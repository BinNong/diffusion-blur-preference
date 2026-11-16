"""Download official OpenAI diffusion checkpoints on the server, with resumable writes.

Only public checkpoints are fetched, directly into this project's ckpt directory.
No credentials, model conversion, training, or evaluation is performed. File sizes and
SHA-256 digests are recorded in results/external_assets.json.

Run: ./venv2/bin/python repro/fetch_external_assets.py
"""

import hashlib
import json
import os
import time

import requests

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = {
    "adm_imagenet64_uncond_100M_1500K.pt": {
        "url": "https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt",
        "size": 484363296,
        "architecture": "OpenAI improved-diffusion ADM UNet, unconditional ImageNet-64",
    },
    "adm_imagenet128_cond_128x128_diffusion.pt": {
        "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt",
        "size": 1686227960,
        "architecture": "OpenAI guided-diffusion ADM UNet, conditional ImageNet-128",
    },
}


def main():
    session = requests.Session()
    session.trust_env = False
    manifest = {"source": "official OpenAI checkpoint URLs", "files": {}}
    for filename, asset in ASSETS.items():
        url = asset["url"]
        destination = os.path.join(PROJ, "ckpt", filename)
        partial = destination + ".part"
        if os.path.exists(destination):
            with open(destination, "rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if os.path.getsize(destination) != asset["size"]:
                raise RuntimeError("Existing checkpoint has unexpected size: " + destination)
            print("[verified]", filename, flush=True)
        else:
            digest = hashlib.sha256()
            for attempt in range(5):
                offset = os.path.getsize(partial) if os.path.exists(partial) else 0
                headers = {"Range": "bytes=%d-" % offset} if offset else {}
                try:
                    if offset:
                        digest = hashlib.sha256()
                        with open(partial, "rb") as handle:
                            for chunk in iter(lambda: handle.read(4*1024**2), b""):
                                digest.update(chunk)
                    with session.get(url, headers=headers, stream=True,
                                     timeout=(10, 300)) as response:
                        if offset and response.status_code != 206:
                            offset = 0
                            digest = hashlib.sha256()
                        response.raise_for_status()
                        with open(partial, "ab" if offset else "wb") as handle:
                            started = time.time()
                            received = offset
                            next_report = received+128*1024**2
                            for chunk in response.iter_content(1024**2):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                digest.update(chunk)
                                received += len(chunk)
                                if received >= next_report:
                                    print("[download]", filename, received, "bytes",
                                          round(time.time()-started), "s", flush=True)
                                    next_report += 128*1024**2
                    if received != asset["size"]:
                        raise IOError("Incomplete download: %d != %d" % (received, asset["size"]))
                    os.replace(partial, destination)
                    digest = digest.hexdigest()
                    print("[saved]", destination, digest, flush=True)
                    break
                except (requests.RequestException, OSError) as error:
                    print("[retry]", filename, attempt+1, repr(error), flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(5*(attempt+1))
        manifest["files"][filename] = {**asset, "sha256": digest,
                                      "local_path": os.path.join("ckpt", filename)}
        with open(os.path.join(PROJ, "results", "external_assets.json"), "w") as handle:
            json.dump(manifest, handle, indent=1)


if __name__ == "__main__":
    main()
