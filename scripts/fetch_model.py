# scripts/fetch_models.py
import os, sys, glob, urllib.request
from huggingface_hub import snapshot_download

REPO = os.environ.get("SBERT_REPO", "sentence-transformers/all-MiniLM-L6-v2")
REV  = os.environ.get("SBERT_REV", "main")   # set to a commit hash later for deterministic builds
DEST = os.environ.get("SBERT_DEST", "/models/sbert/all-MiniLM-L6-v2")

print("[check] touching https://huggingface.co ...", flush=True)
with urllib.request.urlopen("https://huggingface.co", timeout=15) as r:
    print("[ok] status", r.status, flush=True)

os.makedirs(DEST, exist_ok=True)
print(f"[dl] snapshot_download {REPO}@{REV} -> {DEST}", flush=True)
local = snapshot_download(
    repo_id=REPO,
    revision=REV,
    local_dir=DEST,
    allow_patterns=[
        "*.safetensors",
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "data_config.json",
        "tokenizer.*",
        "vocab*",
        "*.txt",
    ],
)

files = [p for p in glob.glob(local + "/**/*", recursive=True) if os.path.isfile(p)]
print(f"[info] downloaded files: {len(files)}", flush=True)
if not files:
    print("[err] no files downloaded", file=sys.stderr, flush=True)
    sys.exit(2)

open(os.path.join(DEST, ".build_stamp"), "w").write(str(len(files)))
print("[ok] snapshot complete", flush=True)
