"""Build a non-secret local reproduction manifest; does not publish an archive.

Run with system Python: python3 repro/build_review2_manifest.py
The manifest records hashes, not file contents. Model checkpoints, credentials, VCS metadata,
server connection notes, and generated caches are excluded. Public URL/DOI fields remain null.
"""

import argparse
import datetime
import hashlib
import json
import os
import zipfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(PROJ, "results", "review2_repro_manifest.json")
ARCHIVE = os.path.join(PROJ, "results", "review2_reproducibility.zip")
EXCLUDED_NAMES = {".DS_Store", ".env", "install_key.exp", "remote_server.md",
                  "review2_repro_manifest.json", "review2_reproducibility.zip"}
EXCLUDED_SUFFIXES = (".pkl", ".pem", ".key", ".pyo", ".pyc")
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "edm_ckpt", "save", "search"}
COMMANDS = [
    "./venv2/bin/python repro/analyze_review2.py --nboot 2000",
    "CUDA_VISIBLE_DEVICES='' ./venv2/bin/python repro/validate_review2_physics.py",
    "./venv2/bin/python repro/sweep_score_profile.py --smoke --source generated",
    "./venv2/bin/python repro/sweep_law_stability.py --smoke --out law_stability_smoke.json",
    "./venv2/bin/python repro/sweep_score_profile.py --source generated --out score_profile_generated.json",
    "./venv2/bin/python repro/sweep_law_stability.py --out law_stability.json",
    "./venv2/bin/python repro/analyze_law_stability.py",
    "./venv2/bin/python repro/sweep_score_profile.py --source real --out score_profile_real.json",
    "./venv2/bin/python repro/recheck_score_profile.py --source generated",
    "./venv2/bin/python repro/recheck_score_profile.py --source real",
    "./venv2/bin/python repro/analyze_score_profile.py",
    "./venv2/bin/python repro/check_score_probe_b1.py",
    "cd paper && tectonic -X compile --keep-logs main.tex",
    "cd .. && python3 repro/build_review2_manifest.py --archive",
]


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included(root, filename):
    if filename in EXCLUDED_NAMES or filename.endswith(EXCLUDED_SUFFIXES):
        return False
    if filename.startswith("recover_"):
        return False
    lowered = filename.lower()
    return not ("id_rsa" in lowered or "id_ed25519" in lowered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    records = []
    for directory in ("paper", "repro", "results", "code"):
        root = os.path.join(PROJ, directory)
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in EXCLUDED_DIRS]
            for filename in files:
                if not included(current, filename):
                    continue
                path = os.path.join(current, filename)
                relative = os.path.relpath(path, PROJ)
                if filename.endswith(".pt") and not relative.startswith("results/"):
                    continue
                records.append({"path": relative, "bytes": os.path.getsize(path),
                                "sha256": digest(path)})
    for filename in ("01_bootstrap.sh", "01b_bootstrap_venv.sh", "02_fetch.sh",
                     "02b_fetch.sh", "03_setup_env.sh"):
        path = os.path.join(PROJ, "tools", filename)
        records.append({"path": os.path.relpath(path, PROJ), "bytes": os.path.getsize(path),
                        "sha256": digest(path)})
    records.sort(key=lambda record: record["path"])
    manifest = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_root_name": os.path.basename(PROJ),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
        "commands": COMMANDS,
        "environment": "results/review2_environment.txt",
        "checkpoint_hashes": ["results/review2_edm_checkpoint_sha256.txt",
                              "results/external_assets.json"],
        "image_provenance": ["results/bsds300_real_manifest.json", "repro/make_pool.py"],
        "public_archive": {"url": None, "doi": None, "published": False,
                           "note": "Local manifest only; publication requires author action."},
        "exclusions": sorted(EXCLUDED_NAMES),
        "notes": ["No model checkpoint or credential is included.",
                  "Server-local absolute pool paths must be replaced by the documented pool build.",
                  "This is an integrity manifest, not a claim of external independent reproduction."],
    }
    with open(OUTPUT, "w") as handle:
        json.dump(manifest, handle, indent=1)
    print("[saved]", OUTPUT, "files", len(records), "bytes", manifest["total_bytes"])
    if args.archive:
        with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            archive.write(OUTPUT, os.path.relpath(OUTPUT, PROJ))
            for record in records:
                archive.write(os.path.join(PROJ, record["path"]), record["path"])
        with zipfile.ZipFile(ARCHIVE) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError("Archive integrity check failed: " + corrupt)
        print("[saved and verified]", ARCHIVE, "bytes", os.path.getsize(ARCHIVE))


if __name__ == "__main__":
    main()
