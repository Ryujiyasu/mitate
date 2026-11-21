"""Oxi で手書き記入済みの帳票を大量に描き、行画像 + 正解を書き出す。

    python -m mitate.oxigen --n 2000 --out data/oxi_lines --workers 4

行の切り出しは周囲に余白をとるので、表の罫線・隣の欄の文字の端など、実物の帳票にある
「邪魔なもの」も一緒に入る（PIL の合成だけでは出しにくい部分）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from . import oxiforms, render
from .charset import char_to_index, normalize

ROOT = Path(__file__).resolve().parent.parent
OXI_FONTS = ROOT / "tools" / "oxi" / "fonts"


def link_fonts() -> list[tuple[int, str]]:
    """学習用の書き手ごとに Oxi の作業フォルダを用意する（システムのフォントは汚さない）。"""
    writers = []
    for p, _ in render.hw_fonts(heldout=False):
        idx = int(p.stem[-4:])
        family = f"HW SynthC {idx:04d}" if p.stem.startswith("HWSynthC") else f"HW Synth {idx:04d}"
        oxiforms.writer_oxi(family)
        writers.append((idx, family))
    return writers


def pick_writer(rng: random.Random, writers):
    """帳票は人に読ませる前提で丁寧に書かれることが多い。丁寧版があれば 7 割はそちらを選ぶ。"""
    careful = [w for w in writers if "SynthC" in w[1]]
    normal = [w for w in writers if "SynthC" not in w[1]]
    if careful and (not normal or rng.random() < 0.7):
        return rng.choice(careful)
    return rng.choice(normal)


def one_form(args):
    seed, out_dir, writers = args
    rng = random.Random(seed)
    widx, family = pick_writer(rng, writers)
    spec = oxiforms.make_spec(rng, family)
    out_dir = Path(out_dir)
    known = char_to_index()
    hw_values = [v for _, v in spec.rows]
    results = []
    with tempfile.TemporaryDirectory() as td:
        dx, pdf = Path(td) / "f.docx", Path(td) / "f.pdf"
        oxiforms.build_docx(spec, dx, rng)
        try:
            oxiforms.oxi_to_pdf(dx, pdf, family)
        except Exception as e:  # noqa: BLE001
            return [], str(e)[:200]
        pages = oxiforms.extract_lines(pdf, dpi=rng.choice([150, 200, 300]))
    for pi, (img, lines) in enumerate(pages):
        page = np.asarray(img.convert("L"))
        for li, ((x0, y0, x1, y1), text) in enumerate(lines):
            label = " ".join(normalize(text).split())
            if not label or any(c not in known for c in label):
                continue
            h = y1 - y0
            mx, my = rng.uniform(0.05, 0.8) * h, rng.uniform(0.05, 0.6) * h
            X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
            X1, Y1 = int(min(page.shape[1], x1 + mx)), int(min(page.shape[0], y1 + my))
            crop = page[Y0:Y1, X0:X1]
            if crop.size == 0 or crop.shape[0] < 8:
                continue
            crop = render.degrade(crop, rng)
            is_hw = any(label.replace(" ", "") in normalize(v).replace(" ", "") for v in hw_values)
            name = f"{seed:07d}_{pi}_{li}.png"
            Image.fromarray(render.to_height(crop)).save(out_dir / name)
            results.append({"file": name, "text": label, "writer": widx if is_hw else -1})
    return results, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "oxi_lines"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    writers = link_fonts()
    print(f"書き手 {len(writers)} 人のフォントを Oxi に登録", flush=True)
    jobs = [(a.seed + i, str(out), writers) for i in range(a.n)]
    n_lines = n_fail = 0
    with ProcessPoolExecutor(a.workers) as ex, (out / "labels.jsonl").open("a", encoding="utf-8") as f:
        for i, (res, err) in enumerate(ex.map(one_form, jobs, chunksize=4)):
            if err:
                n_fail += 1
                if n_fail <= 3:
                    print("失敗:", err, flush=True)
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_lines += len(res)
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"{i + 1}/{a.n} 帳票, {n_lines} 行, 失敗 {n_fail}", flush=True)
    print(f"完了: {n_lines} 行, 失敗 {n_fail}", flush=True)


if __name__ == "__main__":
    main()
