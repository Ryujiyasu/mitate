"""README 用の「読める / ぎりぎり / 読めない」の見本を作る。

    python -m mitate.gallery runs/crnn_v2/last.pt

条件を 1 つずつ悪くしていき、各段階で N 行を読ませて判定する。
  読める     行の正解率 90% 以上
  ぎりぎり   50〜90%
  読めない   50% 未満
学習に使っていない書体・書き手で測る（あれば）。学習の範囲を超える悪条件まで含めて、どこで壊れるかを見せる。
結果は docs/readability/*.png と README の <!-- readability:start --> 〜 <!-- readability:end --> に書く。
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import render, textgen
from .charset import encode, groups, normalize
from .infer import Reader
from .train import edit_distance

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "readability"
LABEL_FONT = str(Path.home() / ".local/share/fonts/sekou/BIZUDGothic-Regular.ttf")
SAMPLE_TEXT = "山田太郎 令和7年4月1日 ¥12,345"
BASE_PX = 40


# ---------------------------------------------------------------- 劣化（1 つの条件だけを指定の強さで）
def _contrast(a, diff):
    paper = 235.0
    ink = 255.0 - a.astype(np.float32)
    return np.clip(paper - ink / 255.0 * diff, 0, 255).astype(np.uint8)


def _resolution(a, char_px):
    f = char_px / BASE_PX
    h, w = a.shape
    small = cv2.resize(a, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
    return small  # 認識器に入れるときに高さ 48 へ拡大される（低解像度の撮影・FAX と同じ）


def _blur(a, sigma):
    return a if sigma <= 0 else cv2.GaussianBlur(a, (0, 0), sigma)


def _noise(a, sd, rng):
    if sd <= 0:
        return a
    g = np.random.default_rng(rng.randrange(1 << 30))
    return np.clip(a.astype(np.float32) + g.normal(0, sd, a.shape), 0, 255).astype(np.uint8)


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _jpeg(a, q):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=int(q))
    return np.asarray(Image.open(buf).convert("L"))


# (名前, 単位の説明, 段階, 手書きのみか, 描画時の上書き, 描画後の劣化)
SWEEPS = [
    ("解像度", "字の高さ (px)", [32, 20, 14, 11, 9, 7], False,
     lambda v: {}, lambda a, v, r: _resolution(a, v)),
    ("ぼけ", "σ (px, 字の高さ 40px 換算)", [0, 1, 1.5, 2, 3, 4], False,
     lambda v: {}, lambda a, v, r: _blur(a, v)),
    ("薄さ", "紙とインクの濃淡差", [200, 120, 80, 50, 35, 20], False,
     lambda v: {}, lambda a, v, r: _contrast(a, v)),
    ("ノイズ", "σ", [0, 10, 20, 35, 50, 70], False,
     lambda v: {}, lambda a, v, r: _noise(a, v, r)),
    ("JPEG 圧縮", "品質", [95, 50, 25, 12, 6, 3], False,
     lambda v: {}, lambda a, v, r: _jpeg(a, v)),
    ("傾き", "度（行ごと。切り出しは傾いた行を囲む長方形）", [0, 1, 2, 3, 5, 8], False,
     lambda v: {}, lambda a, v, r: _u8(render.tilt(a, v * r.choice([-1, 1])))),
    ("影", "暗くなる割合（境目が行を横切る）", [0, 0.2, 0.35, 0.5, 0.65, 0.8], False,
     lambda v: {}, lambda a, v, r: _u8(render.shadow(a, v, r))),
    ("照明のむら", "端の暗くなる割合", [0, 0.3, 0.5, 0.7, 0.8, 0.9], False,
     lambda v: {}, lambda a, v, r: _u8(render.uneven_light(a, v, r))),
    ("手ぶれ", "ぶれの長さ (px, 字の高さ 40px 換算)", [0, 2, 4, 6, 8, 12], False,
     lambda v: {}, lambda a, v, r: _u8(render.motion_blur(a, v, r.uniform(0, 180)))),
    ("急ぎ", "急ぎ度（学習は 0〜1）", [0, 0.3, 0.6, 1.0, 1.5, 2.0], True,
     lambda v: {"hurry": v, "fatigue": 0.0}, lambda a, v, r: a),
    ("疲れ", "疲れ度（学習は 0〜1）", [0, 0.3, 0.6, 1.0, 1.5, 2.0], True,
     lambda v: {"fatigue": v, "hurry": 0.0}, lambda a, v, r: a),
]


def _fonts(heldout: bool):
    printed = [(str(p), i) for p, i in (render.PRINT_HELDOUT if heldout else render.PRINT_TRAIN) if Path(p).exists()]
    hw = [(str(p), i) for p, i in render.hw_fonts(heldout)]
    return printed, hw


def level(acc: float) -> str:
    return "読める" if acc >= 0.9 else ("ぎりぎり" if acc >= 0.5 else "読めない")


def run_sweep(reader, sweep, printed, hw, n, seed):
    name, unit, steps, hw_only, over, deg = sweep
    pool = groups()["all"]
    results = []
    for v in steps:
        rng = random.Random(seed)
        imgs, refs = [], []
        while len(imgs) < n:
            use_hw = hw_only or (hw and rng.random() < 0.5)
            path, idx = rng.choice(hw if use_hw else printed)
            # 帳票の記入欄らしい、読める言葉（名前・住所・日付・金額・電話番号…）で測る
            from .humantest import _short_text
            text = render.fit_text(_short_text(rng), render.coverage(path, idx), rng, pool)
            if not text:
                continue
            a = render.draw_line(text, path, idx, rng, use_hw, size=BASE_PX, plain=True, **over(v))
            imgs.append(deg(a, v, rng))
            refs.append("".join(normalize(text).split()))  # スペースは無視して採点
        outs = reader.read(imgs)
        hyps = [o.text.replace(" ", "") for o in outs]
        ok = [h == r for h, r in zip(hyps, refs)]
        cer = sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / max(1, sum(len(r) for r in refs))
        results.append({"value": v, "acc": float(np.mean(ok)), "cer": cer})
    return results


def example_strip(reader, sweep, results, font, out_png):
    """同じ 1 行を段階ごとに悪くした見本画像（判定と出力つき）。"""
    name, unit, steps, hw_only, over, deg = sweep
    lab = ImageFont.truetype(LABEL_FONT, 18)
    rows = []
    for v, res in zip(steps, results):
        rng = random.Random(1)
        text = render.fit_text(SAMPLE_TEXT, render.coverage(*font), rng, groups()["all"])
        a = render.draw_line(text, font[0], font[1], rng, "HWSynth" in font[0], size=BASE_PX, plain=True, **over(v))
        a = deg(a, v, rng)
        shown = render.to_height(a)
        out = reader.read([a])[0].text
        rows.append((v, res, shown, out))
    W = max(r[2].shape[1] for r in rows) + 420
    canvas = Image.new("L", (W, 64 * len(rows) + 8), 255)
    d = ImageDraw.Draw(canvas)
    for i, (v, res, shown, out) in enumerate(rows):
        y = 8 + i * 64
        lv = level(res["acc"])
        d.text((8, y), f"{v:g}", font=lab, fill=0)
        d.text((8, y + 24), f"{lv} {res['acc'] * 100:.0f}%", font=lab, fill=0)
        canvas.paste(Image.fromarray(shown), (140, y))
        d.text((150 + shown.shape[1], y + 12), f"→ {out}", font=lab, fill=90)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--only", default=None, help="この条件だけ測る（カンマ区切り。README は書き換えない）")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None

    reader = Reader(a.ckpt)
    printed, hw = _fonts(heldout=True)
    note = "学習に使っていない書体・書き手"
    if not printed or len(hw) < 5:
        printed, hw = _fonts(heldout=False)
        note = "学習に使った書体・書き手（未知の書き手のフォントがまだないため）"
    hw_font = hw[0] if hw else printed[0]

    md = [f"判定は各段階 {a.n} 行（{note}、帳票の記入欄らしい言葉: 名前・住所・日付・金額・電話番号など、スペースは無視）の行の正解率で、"
          "**読める** = 90% 以上、**ぎりぎり** = 50〜90%、**読めない** = 50% 未満。"
          f"見本は「{SAMPLE_TEXT}」を {'手書き' if 'HWSynth' in hw_font[0] else '活字'}で描いたもの。"
          f"モデル: `{a.ckpt}`\n",
          "| 条件 | 読める | ぎりぎり | 読めない |", "|---|---|---|---|"]
    details = []
    for sw in SWEEPS:
        if (sw[3] and not hw) or (only and sw[0] not in only):
            continue
        res = run_sweep(reader, sw, printed, hw, a.n, a.seed)
        key = {"解像度": "resolution", "ぼけ": "blur", "薄さ": "contrast", "ノイズ": "noise",
               "JPEG 圧縮": "jpeg", "急ぎ": "hurry", "疲れ": "fatigue",
               "傾き": "tilt", "影": "shadow", "照明のむら": "light", "手ぶれ": "motion"}[sw[0]]
        png = DOCS / f"{key}.png"
        example_strip(reader, sw, res, hw_font, png)
        by = {"読める": [], "ぎりぎり": [], "読めない": []}
        for r in res:
            by[level(r["acc"])].append(f"{r['value']:g}")
        md.append(f"| {sw[0]}（{sw[1]}） | {', '.join(by['読める']) or '—'} | {', '.join(by['ぎりぎり']) or '—'} | "
                  f"{', '.join(by['読めない']) or '—'} |")
        rows = " / ".join(f"{r['value']:g}: {r['acc'] * 100:.0f}%" for r in res)
        details.append(f"#### {sw[0]}\n\n{rows}\n\n![{sw[0]}](docs/readability/{key}.png)\n")
        print(sw[0], rows, flush=True)
    md.append("")
    md.extend(details)
    block = "\n".join(md)
    (DOCS / "summary.json").write_text(json.dumps({"ckpt": a.ckpt, "note": note}, ensure_ascii=False))

    if only:
        return
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    start, end = "<!-- readability:start -->", "<!-- readability:end -->"
    if start in s and end in s:
        s = s[: s.index(start) + len(start)] + "\n" + block + "\n" + s[s.index(end):]
    readme.write_text(s, encoding="utf-8")
    print("README を更新しました")


if __name__ == "__main__":
    main()
