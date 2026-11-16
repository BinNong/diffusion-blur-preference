r"""Continuous VP-ODE adapter for OpenAI discrete-time epsilon-prediction checkpoints.

This constructs a continuous-time surrogate to the trained discrete DDPM; it is not a
DDPM variational bound. The discrete cumulative-alpha schedule is log-linearly interpolated.
For tau in [0,T-1], x(tau)=s(tau)x0+sigma(tau)epsilon, and the network is evaluated with
its continuous sinusoidal timestep embedding. The probability-flow drift is

    dx/dtau = (s'/s)x + sigma^2 (s'/s-sigma'/sigma) score(x,tau),

where score=-epsilon_theta/sigma. Integration starts at the last discrete timestep and
uses an isotropic terminal density after division by sigma. This approximation and
its limited endpoint coverage must be reported explicitly.
"""

import math
import os

import numpy as np
import torch
from scipy import integrate

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DiscreteVP:
    def __init__(self, betas, epsilon_fn):
        self.betas = np.asarray(betas, dtype=np.float64)
        self.alpha_bar = np.cumprod(1.0-self.betas)
        self.log_alpha = np.log(self.alpha_bar)
        self.sigma = np.sqrt(1.0-self.alpha_bar)
        self.indices = np.arange(len(self.betas), dtype=np.float64)
        self.epsilon_fn = epsilon_fn

    def parameters(self, tau):
        tau = float(np.clip(tau, 0.0, len(self.betas)-1.0))
        index = min(int(math.floor(tau)), len(self.betas)-2)
        derivative = self.log_alpha[index+1]-self.log_alpha[index]
        log_alpha = float(np.interp(tau, self.indices, self.log_alpha))
        alpha = math.exp(log_alpha)
        s = math.sqrt(alpha)
        sigma = math.sqrt(1.0-alpha)
        s_over_s = 0.5*derivative
        sigma_over_sigma = -0.5*alpha/(1.0-alpha)*derivative
        return s, sigma, s_over_s, sigma_over_sigma

    def tau_for_sigma(self, sigma):
        if sigma < self.sigma[0]-1e-12 or sigma > self.sigma[-1]+1e-12:
            raise ValueError("Requested sigma outside discrete checkpoint coverage")
        return float(np.interp(sigma, self.sigma, self.indices))

    def flow(self, x, tau):
        _, sigma, s_over_s, sigma_over_sigma = self.parameters(tau)
        epsilon = self.epsilon_fn(x, tau)
        return s_over_s*x+sigma*(sigma_over_sigma-s_over_s)*epsilon

    def potential(self, data, target_sigma, rtol=1e-5, atol=1e-5,
                  method="RK45", epsilon=None, terminal_variance=None):
        tau_end = self.tau_for_sigma(target_sigma)
        tau_start = float(len(self.betas)-1)
        shape = data.shape
        batch = shape[0]
        n_dimensions = int(np.prod(shape[1:]))
        if epsilon is None:
            epsilon = torch.randint_like(data, low=0, high=2).to(data.dtype)*2-1
        if terminal_variance is None:
            terminal_variance = self.sigma[-1]**2

        def ode_func(tau, flattened):
            x = torch.from_numpy(flattened[:-batch].reshape(shape)).to(
                device=data.device, dtype=data.dtype)
            with torch.enable_grad():
                x.requires_grad_(True)
                drift = self.flow(x, tau)
                product = torch.sum(drift*epsilon)
                derivative = torch.autograd.grad(product, x)[0]
            divergence = torch.sum(derivative*epsilon, dim=tuple(range(1, x.ndim)))
            return np.concatenate([drift.detach().cpu().numpy().reshape(-1),
                                   divergence.detach().cpu().numpy().reshape(-1)])

        initial = np.concatenate([data.detach().cpu().numpy().reshape(-1),
                                  np.zeros(batch, dtype=np.float64)])
        solution = integrate.solve_ivp(ode_func, (tau_end, tau_start), initial,
                                       rtol=rtol, atol=atol, method=method)
        if not solution.success:
            raise RuntimeError(solution.message)
        final = solution.y[:, -1]
        terminal = torch.from_numpy(final[:-batch].reshape(shape)).to(
            device=data.device, dtype=data.dtype)
        sigma_start = self.parameters(tau_start)[1]
        terminal = terminal/sigma_start
        log_terminal = (-n_dimensions/2*math.log(2*math.pi*terminal_variance)
                        -terminal.square().sum(dim=tuple(range(1, terminal.ndim))) /
                        (2*terminal_variance))
        delta = torch.from_numpy(final[-batch:]).to(device=data.device, dtype=data.dtype)
        return -(log_terminal+delta)


def load_adm(model_name, device="cuda", fixed_label=0):
    if model_name == "imagenet64_uncond":
        from improved_diffusion.gaussian_diffusion import get_named_beta_schedule
        from improved_diffusion.script_util import create_model
        model = create_model(image_size=64, num_channels=128, num_res_blocks=3,
                             learn_sigma=True, class_cond=False, use_checkpoint=True,
                             attention_resolutions="16,8", num_heads=4,
                             num_heads_upsample=-1,
                             use_scale_shift_norm=True, dropout=0.0)
        filename = "adm_imagenet64_uncond_100M_1500K.pt"
        steps, schedule = 4000, "cosine"
        labels = None
    elif model_name == "imagenet128_cond":
        from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
        from guided_diffusion.script_util import create_model
        model = create_model(image_size=128, num_channels=256, num_res_blocks=2,
                             learn_sigma=True, class_cond=True, use_checkpoint=True,
                             attention_resolutions="32,16,8", num_heads=4,
                             use_scale_shift_norm=True, resblock_updown=True,
                             use_fp16=False, use_new_attention_order=True, dropout=0.0)
        filename = "adm_imagenet128_cond_128x128_diffusion.pt"
        steps, schedule = 1000, "linear"
        labels = fixed_label
    else:
        raise ValueError(model_name)
    path = os.path.join(PROJ, "ckpt", filename)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if hasattr(model, "convert_to_fp32"):
        model.convert_to_fp32()

    def epsilon_fn(x, tau):
        timesteps = torch.full((x.shape[0],), float(tau), device=x.device, dtype=torch.float32)
        if labels is None:
            output = model(x, timesteps)
        else:
            y = torch.full((x.shape[0],), labels, device=x.device, dtype=torch.long)
            output = model(x, timesteps, y)
        return output[:, :3]

    betas = get_named_beta_schedule(schedule, steps)
    metadata = {"model_name": model_name, "checkpoint": filename, "steps": steps,
                "schedule": schedule, "class_conditional": labels is not None,
                "fixed_label": labels, "surrogate": "log-linear continuous VP-ODE",
                "min_sigma": float(np.sqrt(betas[0])),
                "max_sigma": float(np.sqrt(1-np.prod(1-betas)))}
    return DiscreteVP(betas, epsilon_fn), model, metadata


def analytic_guard():
    variance = 0.3
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)

    def epsilon_fn(y, tau):
        alpha = math.exp(float(np.interp(tau, np.arange(len(betas)),
                                         np.log(np.cumprod(1-betas)))))
        sigma = math.sqrt(1-alpha)
        s = math.sqrt(alpha)
        return sigma*y/(s*s*variance+sigma*sigma)

    vp = DiscreteVP(betas, epsilon_fn)
    data = torch.linspace(-0.4, 0.5, 24).reshape(1, 3, 2, 4)
    target_sigma = 0.2
    alpha = math.exp(float(np.interp(vp.tau_for_sigma(target_sigma), vp.indices, vp.log_alpha)))
    marginal_variance = alpha*variance+(1-alpha)
    exact = (data.numel()/2*math.log(2*math.pi*marginal_variance)
             +data.square().sum()/(2*marginal_variance))
    terminal_variance = ((vp.alpha_bar[-1]*variance+vp.sigma[-1]**2) /
                         vp.sigma[-1]**2)
    numerical = vp.potential(data, target_sigma, rtol=1e-7, atol=1e-7,
                             method="DOP853", terminal_variance=terminal_variance)[0]
    return float(numerical), float(exact), abs(float(numerical-exact))
