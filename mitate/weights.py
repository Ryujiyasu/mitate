"""配布している学習済みモデルを取ってくる。

    from mitate.infer import Reader
    reader = Reader()            # 既定のモデル（初回だけ GitHub Releases から取得し、~/.cache/mitate に置く）

モデルは重みだけ（torch.load(weights_only=True) で読める形）。
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RELEASES = "https://github.com/Ryujiyasu/mitate/releases/download"
MODELS = {
    # 名前: (タグ, ファイル名, sha256)
    "mitate-v0.1": ("v0.1.0", "mitate-v0.1.pt", "7996e3d36ae415fd6256ed3f78aa3e0780fc0fbab6004acb205ac4a202bfdc9b"),
}
DEFAULT = "mitate-v0.1"


def cache_dir() -> Path:
    return Path(os.environ.get("MITATE_CACHE", Path.home() / ".cache" / "mitate"))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str = DEFAULT) -> Path:
    """モデルのファイルの場所。なければ取ってきて、中身（sha256）を確かめる。"""
    tag, fname, sha = MODELS[name]
    path = cache_dir() / fname
    if path.exists() and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASES}/{tag}/{fname}"
    print(f"モデルを取得します: {url}", file=sys.stderr)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if _sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"取得したモデルの中身が違います（{url}）")
    tmp.replace(path)
    return path
