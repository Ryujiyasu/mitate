"""「読める / 読めない」の境目を測る。

    python -m mitate.boundary runs/crnn_v1/last.pt --n 6000

出すもの（runs/…/boundary.md）:
  1. 確信度で足切りしたときの「自動で受け付けられる割合」と、そのときの正解率
     → 「確信度が高い何割は人が見なくてよいか」の根拠
  2. 条件ごとの誤り率（ぼけ・解像度・コントラスト・ノイズ・JPEG・詰め書き・罫線またぎ・書き手の雑さ…）
     → どこから急に読めなくなるか
合成データは条件が全部分かっているので、境目を正確に測れる。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .charset import groups
from .data import SynthLines
from .infer import Reader
from .render import HW_DIR
from .train import edit_distance


def collect(n: int, seed: int, heldout: bool):
    """n 行を作り、(画像, 正解, 条件) を返す。活字と手書きを半々に。"""
    out = []
    pool = groups()["all"]
    for hw_ratio, k in ((0.0, n // 2), (1.0, n - n // 2)):
        ds = SynthLines(seed, hw_ratio, heldout)
        printed, hw = ds._fonts()
        rng = random.Random(seed + int(hw_ratio * 7))
        for _ in range(k):
            m = {}
            a, _, label, writer = ds.generate(rng, printed, hw, pool, meta=m)
            m["writer"] = writer
            out.append((a, label, m))
    return out


def writer_styles() -> dict[int, dict]:
    p = HW_DIR / "writers.jsonl"
    if not p.exists():
        return {}
    return {d["index"]: d["style"] for d in map(json.loads, p.open(encoding="utf-8"))}


# 条件ごとの区切り（境目が見えるように細かめに）
FACTORS = {
    "解像度（元の字の高さ px × 縮小率）": (lambda m: m["font_px"] * m["downscale"], [0, 14, 18, 22, 26, 32, 40, 60]),
    "ぼけ（σ）": (lambda m: m["blur"], [-0.01, 0.01, 0.5, 0.8, 1.0, 1.3]),
    "コントラスト（紙とインクの差）": (lambda m: m["contrast"], [0, 110, 130, 150, 180, 260]),
    "ノイズ（σ）": (lambda m: m["noise"], [-0.01, 0.01, 5, 9, 15]),
    "JPEG 品質": (lambda m: m["jpeg"], [0, 40, 60, 80, 100, 101]),
    "詰め書き（最後の字の大きさ）": (lambda m: m["cram"], [0, 0.55, 0.65, 0.8, 0.99, 1.01]),
    "字間（字の大きさ比）": (lambda m: m["tracking"], [-1, -0.02, 0.05, 0.15, 1]),
    "行の文字数": (lambda m: m["n_chars"], [0, 5, 10, 15, 20, 40]),
    "急ぎ（hurry）": (lambda m: m.get("hurry"), [-0.01, 0.01, 0.2, 0.5, 0.8, 1.0]),
    "疲れ（fatigue）": (lambda m: m.get("fatigue"), [-0.01, 0.01, 0.2, 0.5, 0.8, 1.0]),
}
FLAGS = {
    "罫線が字を横切る": lambda m: m["crossing"],
    "上下の枠線": lambda m: m["border"],
    "二値化（FAX）": lambda m: m["binarize"],
    "線が太い": lambda m: m["stroke"] == "thick",
    "線が細い": lambda m: m["stroke"] == "thin",
}
STYLE_FACTORS = {
    "書き手の雑さ（jitter）": ("jitter", [0, 0.6, 1.0, 2.0, 3.0, 6.0]),
    "筆画のずれ（stroke_shift）": ("stroke_shift", [-0.01, 0.3, 1.0, 2.0, 5.0]),
    "線の太さ（pen_width）": ("pen_width", [0, 3.0, 4.5, 6.0, 9.0]),
    "傾き（slant の大きさ）": ("abs_slant", [0, 0.04, 0.08, 0.15, 1]),
}


def table(rows, key, bins):
    groups_ = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is None:
            continue
        for lo, hi in zip(bins[:-1], bins[1:]):
            if lo < v <= hi or (lo == bins[0] and v == lo):
                groups_[(lo, hi)].append(r)
                break
    lines = ["| 範囲 | 件数 | 行の正解率 | 文字誤り率 |", "|---|---|---|---|"]
    for (lo, hi) in zip(bins[:-1], bins[1:]):
        g = groups_.get((lo, hi), [])
        if not g:
            continue
        acc = np.mean([r["ok"] for r in g])
        cer = sum(r["err"] for r in g) / max(1, sum(r["len"] for r in g))
        lines.append(f"| {lo:g} 〜 {hi:g} | {len(g)} | {acc * 100:.1f}% | {cer * 100:.2f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--heldout", action="store_true", help="学習に使っていない書体・書き手で測る")
    a = ap.parse_args()

    reader = Reader(a.ckpt)
    samples = collect(a.n, a.seed, a.heldout)
    styles = writer_styles()
    rows = []
    for i in range(0, len(samples), 64):
        chunk = samples[i : i + 64]
        for (img, label, m), r in zip(chunk, reader.read([c[0] for c in chunk])):
            e = edit_distance(r.text.replace(" ", ""), label.replace(" ", ""))  # スペースは無視して採点
            st = styles.get(m["writer"], {})
            rows.append(dict(m, ok=e == 0, err=e, len=len(label.replace(" ", "")), conf=r.conf, ref=label, hyp=r.text,
                             jitter=st.get("jitter"), stroke_shift=st.get("stroke_shift"),
                             pen_width=st.get("pen_width"), abs_slant=abs(st["slant"]) if st else None))

    out = [f"# 読める / 読めないの境目\n\nモデル: `{a.ckpt}`、{len(rows)} 行（活字と手書き半々、"
           f"{'学習に使っていない書体・書き手' if a.heldout else '学習に使った書体・書き手'}、テキストは未学習）\n"]
    for name, sub in (("全体", rows), ("活字", [r for r in rows if not r["handwriting"]]),
                      ("手書き", [r for r in rows if r["handwriting"]])):
        if not sub:
            continue
        acc = np.mean([r["ok"] for r in sub])
        cer = sum(r["err"] for r in sub) / max(1, sum(r["len"] for r in sub))
        out.append(f"- {name}: 行の正解率 {acc * 100:.1f}%、文字誤り率 {cer * 100:.2f}%（{len(sub)} 行）")

    # 1. 確信度で足切り
    out.append("\n## 1. 確信度で足切りしたとき\n\n行の確信度 = その行で一番自信のない字の確信度。"
               "高い順に受け付け、残りは人が確認する想定。\n")
    for name, sub in (("活字", [r for r in rows if not r["handwriting"]]), ("手書き", [r for r in rows if r["handwriting"]])):
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: -r["conf"])
        out.append(f"\n### {name}\n\n| 自動で受け付ける割合 | 確信度の閾値 | 受け付けた行の正解率 |\n|---|---|---|")
        for cov in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
            k = max(1, int(len(sub) * cov))
            acc = np.mean([r["ok"] for r in sub[:k]])
            out.append(f"| {cov * 100:.0f}% | {sub[k - 1]['conf']:.3f} | {acc * 100:.2f}% |")
        for target in (0.99, 0.995, 0.999):
            best = 0
            for k in range(1, len(sub) + 1):
                if np.mean([r["ok"] for r in sub[:k]]) >= target:
                    best = k
            out.append(f"\n正解率 {target * 100:g}% を保てる受け付け割合: **{best / len(sub) * 100:.1f}%**")

    # 2. 条件ごと
    out.append("\n## 2. 条件ごとの読みやすさ\n")
    for name, (key, bins) in FACTORS.items():
        for kind, sub in (("活字", [r for r in rows if not r["handwriting"]]), ("手書き", [r for r in rows if r["handwriting"]])):
            if sub:
                out.append(f"\n### {name}（{kind}）\n\n" + table(sub, key, bins))
    out.append("\n### あり / なし（全体）\n\n| 条件 | あり: 行の正解率 | なし: 行の正解率 |\n|---|---|---|")
    for name, f in FLAGS.items():
        yes = [r for r in rows if f(r)]
        no = [r for r in rows if not f(r)]
        if yes and no:
            out.append(f"| {name} | {np.mean([r['ok'] for r in yes]) * 100:.1f}%（{len(yes)}） | "
                       f"{np.mean([r['ok'] for r in no]) * 100:.1f}%（{len(no)}） |")
    hw = [r for r in rows if r["handwriting"] and r["jitter"] is not None]
    for name, (k, bins) in STYLE_FACTORS.items():
        if hw:
            out.append(f"\n### {name}（手書き）\n\n" + table(hw, lambda r, k=k: r[k], bins))

    # 3. 自信満々の間違い
    bad = sorted([r for r in rows if not r["ok"]], key=lambda r: -r["conf"])[:15]
    out.append("\n## 3. 自信満々に間違えた例（一番まずいもの）\n\n| 確信度 | 正解 | 出力 | 書体 |\n|---|---|---|---|")
    for r in bad:
        out.append(f"| {r['conf']:.3f} | {r['ref']} | {r['hyp']} | {r['font']} |")

    path = Path(a.ckpt).parent / ("boundary_heldout.md" if a.heldout else "boundary.md")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:12]))
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
