"""Prepare deterministic BSDS300 real-image pools; no model-generated images.

Use all 100 official test images in sorted filename order. Convert to RGB, resize the
short side with Lanczos, and centre-crop without changing aspect ratio. Save [-1,1]
float32 tensors at 64x64 and 128x128, plus provenance and a contact sheet of the originals.

Run: ./venv2/bin/python repro/prepare_real_images.py
"""

import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(PROJ, "data", "BSDS300")


def main():
    files = []
    for root, _, names in os.walk(SOURCE):
        for name in names:
            path = os.path.join(root, name)
            relative = os.path.relpath(path, SOURCE)
            if name.lower().endswith((".jpg", ".jpeg", ".png")) and "test" in relative.split(os.sep):
                files.append((name, path))
    files.sort(key=lambda item: item[0])
    if len(files) != 100:
        raise RuntimeError("Expected 100 BSDS300 test images, got %d" % len(files))
    records = []
    tensors = {64: [], 128: []}
    sheet = Image.new("RGB", (640, 640), "white")
    for index, (name, path) in enumerate(files):
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        with Image.open(path) as image:
            image = image.convert("RGB")
            original_size = image.size
            scale = 128/min(image.size)
            resized = image.resize((round(image.width*scale), round(image.height*scale)),
                                   Image.Resampling.LANCZOS)
            left = (resized.width-128)//2
            top = (resized.height-128)//2
            crop = resized.crop((left, top, left+128, top+128))
            array = np.asarray(crop, dtype=np.float32)/127.5-1.0
            for size in tensors:
                if size == 128:
                    output = array
                else:
                    output = np.asarray(crop.resize((size, size), Image.Resampling.LANCZOS),
                                        dtype=np.float32)/127.5-1.0
                tensors[size].append(torch.from_numpy(output).permute(2, 0, 1))
            sheet.paste(crop.resize((64, 64), Image.Resampling.LANCZOS),
                        ((index%10)*64, (index//10)*64))
        records.append({"index": index, "filename": name, "sha256": digest,
                        "original_size": original_size, "preprocessing":
                        "RGB; Lanczos short-side resize; centre crop; /127.5-1"})
    for size, values in tensors.items():
        tensor = torch.stack(values)
        if tensor.shape != (100, 3, size, size) or not torch.isfinite(tensor).all():
            raise RuntimeError("Invalid real-image tensor")
        path = os.path.join(PROJ, "data", "bsds300_real_test_%d.pt" % size)
        torch.save(tensor, path)
        with open(path, "rb") as handle:
            tensor_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        print("[saved]", path, tuple(tensor.shape), tensor_digest, flush=True)
    sheet.save(os.path.join(PROJ, "results", "bsds300_real_contact_sheet.png"))
    archive = os.path.join(PROJ, "data", "archives", "BSDS300-images.tgz")
    with open(archive, "rb") as handle:
        archive_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"source": "BSDS300 official test split", "n": len(files),
                "generated": False, "records": records,
                "archive": {"url": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz",
                            "format": "uncompressed tar despite .tgz suffix",
                            "bytes": os.path.getsize(archive), "sha256": archive_sha256},
                "note": "Real-data external-validity pool; no claim about FFHQ training membership."}
    with open(os.path.join(PROJ, "results", "bsds300_real_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=1)
    print("[saved] manifest and contact sheet", flush=True)


if __name__ == "__main__":
    main()
