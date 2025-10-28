import os, sys, glob, urllib.request
from huggingface_hub import snapshot_download

REPO = os.environ.get("CLIP_REPO", "sentence-transformers/clip-ViT-B-32")
REV  = os.environ.get("CLIP_REV", "main")
DEST = os.environ.get("CLIP_DEST", "/models/clip/clip-ViT-B-32")

print("[check] touching https://huggingface.co ...", flush=True)
with urllib.request.urlopen("https://huggingface.co", timeout=15) as r:
    print("[ok] status", r.status, flush=True)

os.makedirs(DEST, exist_ok=True)
print(f"[dl] snapshot_download {REPO}@{REV} -> {DEST}", flush=True)
local = snapshot_download(
    repo_id=REPO,
    revision=REV,
    local_dir=DEST,
)

files = [p for p in glob.glob(local + "/**/*", recursive=True) if os.path.isfile(p)]
print(f"[info] downloaded files: {len(files)}", flush=True)
if not files:
    print("[err] no files downloaded", file=sys.stderr, flush=True)
    sys.exit(2)

open(os.path.join(DEST, ".build_stamp"), "w").write(str(len(files)))
print("[ok] snapshot complete", flush=True)