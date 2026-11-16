"""一次性诊断：tw_lev 与 tw_mean 的半径明细，确认 tw_lev 的物理含义。"""
import importlib.util
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd)

tb = hd.build_tables()
r, names, araw = tb["r"], tb["names"], tb["araw"]
print("%-14s %6s %6s %6s %6s %8s %8s %6s %6s %6s" %
      ("profile", "amin", "amax", "mean", "0.9m", "tw_lev", "tw_mean",
       "r10", "r50", "r90"))
for nm in sorted(names, key=lambda n: tb["desc"][n]["tw_lev"]):
    d = tb["desc"][nm]
    a = araw[nm]
    tm = "%.3f" % d["tw_mean"] if np.isfinite(d["tw_mean"]) else "NaN"
    print("%-14s %6.3f %6.3f %6.3f %6.3f %8.3f %8s %6.3f %6.3f %6.3f" %
          (nm, d["_amin"], d["_amax"], a.mean(), 0.9 * a.mean(),
           d["tw_lev"], tm, d.get("_r10", float("nan")),
           d.get("_r50", float("nan")), d.get("_r90", float("nan"))))
