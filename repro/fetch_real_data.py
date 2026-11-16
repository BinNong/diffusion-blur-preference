"""Fetch public real-image datasets for review-response experiments.

Downloads official archives into data/archives and safely extracts BSDS300 and CIFAR-10
into data/. Records URLs, byte counts, SHA-256 digests and archive members. No generated
images are included in these real-data pools. Does not fetch or redistribute model weights.

Run: ./venv2/bin/python repro/fetch_real_data.py
"""

import hashlib
import json
import os
import tarfile
import time

import requests

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
SOURCES = {
    "BSDS300-images.tgz": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz",
    "cifar-10-python.tar.gz": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
}


def download(session, name, url):
    os.makedirs(os.path.join(DATA, "archives"), exist_ok=True)
    output = os.path.join(DATA, "archives", name)
    partial = output + ".part"
    if os.path.exists(output):
        return output
    for attempt in range(5):
        try:
            with session.get(url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(1024**2):
                        if chunk:
                            handle.write(chunk)
            os.replace(partial, output)
            return output
        except (requests.RequestException, OSError) as error:
            print("[retry]", name, attempt+1, repr(error), flush=True)
            if attempt == 4:
                raise
            time.sleep(5*(attempt+1))


def extract_archive(path):
    destination = os.path.join(DATA, "BSDS300" if os.path.basename(path).startswith("BSDS") else "cifar-10")
    os.makedirs(destination, exist_ok=True)
    with tarfile.open(path, "r:*") as archive:
        members = []
        for member in archive.getmembers():
            target = os.path.realpath(os.path.join(destination, member.name))
            if os.path.commonpath([os.path.realpath(destination), target]) != os.path.realpath(destination):
                raise RuntimeError("Unsafe archive path: " + member.name)
            if not member.isfile():
                continue
            if os.path.exists(target):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = archive.extractfile(member)
            with open(target, "wb") as handle:
                while chunk := source.read(1024**2):
                    handle.write(chunk)
            members.append(member.name)
    return destination, members


def main():
    session = requests.Session()
    session.trust_env = False
    manifest = {"sources": {}, "data_policy": "real-image archives; no model-generated samples"}
    for name, url in SOURCES.items():
        path = download(session, name, url)
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        destination, members = extract_archive(path)
        manifest["sources"][name] = {"url": url, "bytes": os.path.getsize(path),
                                    "sha256": digest, "extracted_to": destination,
                                    "n_files": len(members), "members": members}
        print("[ready]", name, len(members), digest, flush=True)
        with open(os.path.join(PROJ, "results", "real_data_provenance.json"), "w") as handle:
            json.dump(manifest, handle, indent=1)


if __name__ == "__main__":
    main()
