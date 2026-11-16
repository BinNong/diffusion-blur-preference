"""Parallel, resumable downloads of official OpenAI checkpoints.

Range support was verified with HTTP 206 before use. Eight concurrent range requests,
4-MiB independently resumable chunks, strict final byte count, SHA-256 manifest, and atomic
assembly. Only public official URLs are used. Existing complete chunks are never re-downloaded.

Run: ./venv2/bin/python repro/fetch_external_assets_parallel.py
"""

import concurrent.futures
import hashlib
import json
import os
import time

import requests

from fetch_external_assets import ASSETS

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 4*1024**2
WORKERS = 8


def fetch_chunk(url, path, start, end):
    expected = end-start+1
    if os.path.exists(path) and os.path.getsize(path) == expected:
        return path
    if os.path.exists(path):
        raise RuntimeError("Unexpected partial chunk size: " + path)
    for attempt in range(6):
        try:
            session = requests.Session()
            session.trust_env = False
            with session.get(url, headers={"Range": "bytes=%d-%d" % (start, end)},
                             stream=True, timeout=(10, 300)) as response:
                if response.status_code != 206:
                    raise IOError("Range was not honored: HTTP %d" % response.status_code)
                temporary = path+".tmp"
                with open(temporary, "wb") as handle:
                    for block in response.iter_content(256*1024):
                        if block:
                            handle.write(block)
                if os.path.getsize(temporary) != expected:
                    raise IOError("Wrong chunk length: " + path)
                os.replace(temporary, path)
            print("[chunk]", os.path.basename(path), "done", flush=True)
            return path
        except (requests.RequestException, OSError) as error:
            print("[retry]", os.path.basename(path), attempt+1, repr(error), flush=True)
            if attempt == 5:
                raise
            time.sleep(5*(attempt+1))
            if os.path.exists(path+".tmp"):
                os.remove(path+".tmp")
    raise RuntimeError("unreachable")


def main():
    manifest_path = os.path.join(PROJ, "results", "external_assets.json")
    manifest = {"source": "official OpenAI checkpoint URLs",
                "download": "8 parallel HTTP ranges, 4-MiB chunks", "files": {}}
    for filename, asset in ASSETS.items():
        destination = os.path.join(PROJ, "ckpt", filename)
        if os.path.exists(destination):
            if os.path.getsize(destination) != asset["size"]:
                raise RuntimeError("Wrong existing checkpoint size")
        else:
            directory = os.path.join(PROJ, "ckpt", filename+".chunks")
            os.makedirs(directory, exist_ok=True)
            chunks = []
            for index, start in enumerate(range(0, asset["size"], CHUNK)):
                end = min(start+CHUNK-1, asset["size"]-1)
                chunks.append((os.path.join(directory, "%06d.part" % index), start, end))
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(fetch_chunk, asset["url"], path, start, end)
                           for path, start, end in chunks]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            assembled = destination+".assemble"
            digest = hashlib.sha256()
            with open(assembled, "wb") as output:
                for path, _, _ in chunks:
                    with open(path, "rb") as source:
                        for block in iter(lambda: source.read(CHUNK), b""):
                            output.write(block)
                            digest.update(block)
            if os.path.getsize(assembled) != asset["size"]:
                raise RuntimeError("Assembled checkpoint size mismatch")
            os.replace(assembled, destination)
            for path, _, _ in chunks:
                os.remove(path)
            os.rmdir(directory)
            print("[saved]", destination, digest.hexdigest(), flush=True)
        with open(destination, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest["files"][filename] = {**asset, "sha256": digest,
                                      "local_path": os.path.join("ckpt", filename)}
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=1)
    print("[complete]", manifest_path, flush=True)


if __name__ == "__main__":
    main()
