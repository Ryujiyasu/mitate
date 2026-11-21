"""人の読み取りテストの問題を作る / 回答を採点してモデルと比べる。

    python -m mitate.humantest make                 # data/humantest/ に画像・問題一覧・正解を作る
    python -m mitate.humantest score answers.json --ckpt runs/crnn_v2/last.pt

問題は gallery.py と同じ「条件を 1 つずつ悪くしていく」画像。書き手・書体は学習に使っていないもの。
正解（key.json）はページに入れない。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from . import render, textgen
from .charset import groups, normalize
from .gallery import BASE_PX, SWEEPS, level

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "humantest"
PER_STEP = 3  # 条件 × 段階 ごとの問題数


# 人が読める（読み方を知っていて入力できる）言葉だけで問題を作る。
# 珍しい漢字は、形がはっきり見えても読み方を知らなければ入力できないので、人との比較に使えない
EASY_SURNAMES = [s for s in textgen.SURNAMES if s not in
                 "髙橋 山﨑 齋藤 濱田 邉見 鬼頭 楢崎 蕪木 嶋田 廣瀬 薮内".split()]
EASY_GIVEN = [g for g in textgen.GIVEN if g not in "巌 毅 勲 猛 稔 千尋 紗希 芽依 凛 湊 蓮".split()]
CITIES = ("札幌 仙台 新宿 渋谷 品川 横浜 川崎 千葉 船橋 大宮 浦和 静岡 浜松 名古屋 豊田 京都 大阪 堺 神戸 "
          "姫路 奈良 岡山 広島 福山 松山 高松 福岡 北九州 熊本 鹿児島 那覇 金沢 富山 新潟 長野 松本 甲府 宇都宮 前橋 水戸").split()
SHOP = "商事 工業 建設 製作所 不動産 運輸 食品 電機 薬局 歯科 病院 保育園".split()
KATA = "サービス センター マンション ハイツ コーポ スポーツ システム デザイン クリニック ファーム".split()


def _easy(kind: str, rng: random.Random) -> str:
    if kind == "name":
        return rng.choice(EASY_SURNAMES) + rng.choice(["", " "]) + rng.choice(EASY_GIVEN)
    if kind == "address":
        t = rng.choice(textgen.PREFS) + rng.choice(CITIES) + rng.choice(["市", "区"])
        return t + f"{rng.randint(1, 9)}-{rng.randint(1, 30)}" if rng.random() < 0.6 else t + f"{rng.randint(1, 9)}丁目"
    if kind == "company":
        n = rng.choice(EASY_SURNAMES) + rng.choice(SHOP)
        return rng.choice(["株式会社", "有限会社", ""]) + n
    if kind == "kata":
        return rng.choice(EASY_SURNAMES) + rng.choice(KATA)
    if kind == "words":
        return " ".join(rng.choice(textgen.FORM_WORDS) for _ in range(rng.randint(2, 3)))
    return textgen.field(kind, rng)  # date / phone / postal / money / number（数字が主で読める）


def _short_text(rng: random.Random) -> str:
    """人が読めて打ち込める、帳票らしい文字列（4〜14 字）。"""
    kinds = ["name", "address", "company", "kata", "words", "date", "phone", "postal", "money", "number"]
    while True:
        t = " ".join(normalize(_easy(rng.choice(kinds), rng)).split())
        if 4 <= len(t) <= 14:
            return t


def make():
    printed = [(str(p), i) for p, i in render.PRINT_HELDOUT if Path(p).exists()]
    hw = [(str(p), 0) for p in sorted((ROOT / "fonts" / "hw").glob("HWSynth09*.ttf"))]
    assert hw, "テスト用の書き手フォント（HWSynth09xx）がない"
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    pool = groups()["all"]
    items, key = [], {}
    rng = random.Random(20261120)
    n = 0
    for sw in SWEEPS:
        name, unit, steps, hw_only, over, deg = sw
        for si, v in enumerate(steps):
            for k in range(PER_STEP):
                use_hw = hw_only or rng.random() < 0.5
                path, idx = rng.choice(hw if use_hw else printed)
                text = render.fit_text(_short_text(rng), render.coverage(path, idx), rng, pool)
                a = render.draw_line(text, path, idx, rng, use_hw, size=BASE_PX, plain=True, **over(v))
                a = deg(a, v, rng)
                n += 1
                iid = f"r{n:03d}"  # 第 2 版（第 1 版 q*** は珍しい漢字が混ざっていた）
                # 表示用: 低解像度の条件も、人が見る大きさ（高さ 64px 前後）に拡大して見せる
                Image.fromarray(a).save(OUT / "img" / f"{iid}.png")
                items.append({"id": iid, "file": f"img/{iid}.png", "h": int(a.shape[0]), "w": int(a.shape[1])})
                key[iid] = {"text": " ".join(text.split()), "condition": name, "unit": unit, "value": v, "step": si,
                            "handwriting": use_hw, "font": Path(path).stem}
    (OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False))
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
    print(f"{len(items)} 問を {OUT} に作成")


def _norm(s: str) -> str:
    return "".join(normalize(s).split())


def score(answers_path: str, ckpt: str | None):
    from .train import edit_distance
    key = json.loads((OUT / "key.json").read_text())
    ans = json.loads(Path(answers_path).read_text())  # [{participant, id, text, confidence, ms}]
    by_item: dict[str, list] = {}
    for a in ans:
        by_item.setdefault(a["id"], []).append(a)
    model = {}
    if ckpt:
        from .infer import Reader
        reader = Reader(ckpt)
        ids = sorted(key)
        imgs = [np.asarray(Image.open(OUT / "img" / f"{i}.png").convert("L")) for i in ids]
        for i, r in zip(ids, reader.read(imgs)):
            model[i] = r
    rows = []
    for iid, k in key.items():
        hs = by_item.get(iid, [])
        if not hs:
            continue
        ref = _norm(k["text"])
        human_acc = np.mean([_norm(h["text"]) == ref for h in hs])
        human_sure = np.mean([h["confidence"] == "sure" for h in hs])
        row = dict(k, id=iid, n=len(hs), human_acc=human_acc, human_sure=human_sure,
                   human_ms=float(np.median([h["ms"] for h in hs])))
        if iid in model:
            row["model_ok"] = _norm(model[iid].text) == ref
            row["model_conf"] = model[iid].conf
            row["model_text"] = model[iid].text
        rows.append(row)
    out = ["# 人とモデルの読み取り比較\n", f"回答 {len(ans)} 件、問題 {len(rows)} 問\n"]
    out.append("| 条件 | 段階 | 人の正解率 | 人が「自信あり」 | モデル正解率 |\n|---|---|---|---|---|")
    groups_: dict[tuple, list] = {}
    for r in rows:
        groups_.setdefault((r["condition"], r["step"], r["value"]), []).append(r)
    for (c, s, v), g in sorted(groups_.items()):
        m = [x["model_ok"] for x in g if "model_ok" in x]
        out.append(f"| {c} | {v:g} | {np.mean([x['human_acc'] for x in g]) * 100:.0f}% | "
                   f"{np.mean([x['human_sure'] for x in g]) * 100:.0f}% | "
                   f"{(np.mean(m) * 100 if m else float('nan')):.0f}% |")
    if model:
        inhuman = [r for r in rows if r["human_acc"] == 1.0 and not r.get("model_ok", True)]
        easy = [r for r in rows if r["human_acc"] == 1.0]
        out.append(f"\n## 人なら間違えない誤り\n\n人が全員正しく読めた {len(easy)} 問のうち、モデルが間違えたのは "
                   f"**{len(inhuman)} 問**（{len(inhuman) / max(1, len(easy)) * 100:.1f}%）\n")
        for r in inhuman[:20]:
            out.append(f"- {r['id']}（{r['condition']} {r['value']:g}）正解「{r['text']}」→ モデル「{r['model_text']}」")
    path = OUT / "report.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:40]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    s = sub.add_parser("score")
    s.add_argument("answers")
    s.add_argument("--ckpt")
    a = ap.parse_args()
    if a.cmd == "make":
        make()
    else:
        score(a.answers, a.ckpt)


if __name__ == "__main__":
    main()
