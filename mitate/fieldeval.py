"""欄の知識の効き目を測る（読み取りテストの 78 問。公開中のテストは書き換えない）。

    python -m mitate.fieldeval runs/full_s1.0/last.pt

画像だけの読み（OCR）と、欄の知識を通した答えを比べ、「直った」「壊した」を数える。
"""
from __future__ import annotations

import csv
import pickle
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from .charset import normalize
from .fields import check_field
from .formtest import FIELDS, LEVELS, OUT, WRITERS, _jobs, _render
from .humantest import EASY_SURNAMES
from .infer import Reader
from .lexicon import KEN_ALL

KIND = dict(FIELDS)


# 罠: 語彙にない（でも実在しそうな）名前。知識がこれを語彙の言葉に書き換えたら「壊した」
ODD_SUFFIX = "屋 堂 組 庵 亭 舎 工房 保守 企画室 書房 本家 製麺 鍼灸 珈琲店 写真室 ドメイン 量子研究室 月面商事".split()


def trap_jobs():
    rng = random.Random(2121)
    rows = [r for r in csv.reader(KEN_ALL.open(encoding="cp932", newline=""))
            if "以下に掲載" not in r[8] and "（" not in r[8] and len(r[8]) <= 6]
    jobs, key, n = [], {}, 0
    for w in WRITERS:
        for lv in LEVELS[1:]:
            for _ in range(5):
                n += 1
                iid = f"t{n:03d}"
                if n % 2:
                    label = "会社名"
                    value = rng.choice(["", "株式会社", "有限会社"]) + rng.choice(EASY_SURNAMES) + rng.choice(ODD_SUFFIX)
                else:
                    label = "住所"
                    r = rng.choice(rows)
                    value = r[6] + r[7] + r[8] + f"{rng.randint(1, 9)}-{rng.randint(1, 30)}"
                jobs.append((iid, label, value, f"HW Test {w:04d} {lv}", False, rng.randrange(1 << 30)))
                key[iid] = {"text": value, "label": label, "writer": w, "level": lv, "cram": False}
    return jobs, key


def cram_jobs():
    """詰め書きの数字: 同じ数字の連続（11, 00）や細い「1」が、欄の端で潰れて書かれたもの。"""
    rng = random.Random(3131)
    def money():
        d = rng.choice(["11", "00", "1", "10", "01", "111"])
        n = int(str(rng.randint(1, 99)) + d + str(rng.randint(0, 999)).zfill(3))
        return f"金{n:,}円"
    def address():
        return rng.choice(EASY_SURNAMES[:0] or ["東京都", "大阪府", "福岡県", "北海道", "愛知県"]) + \
            rng.choice(["中央区", "北区", "南区", "西区"]) + rng.choice(["1-1-11", "11-1", "2-11-1", "10-11", "1-11-11", "9-11"])
    def postal():
        return "〒" + rng.choice(["110-0011", "100-0001", "111-1111", "011-0110", "101-1001"])
    jobs, key, n = [], {}, 0
    for w in WRITERS:
        for lv in LEVELS[:3]:
            for label, fn in [("金額", money)] * 3 + [("住所", address)] * 2 + [("郵便番号", postal)] * 2:
                n += 1
                iid = f"c{n:03d}"
                value = fn()
                jobs.append((iid, label, value, f"HW Test {w:04d} {lv}", True, rng.randrange(1 << 30)))
                key[iid] = {"text": value, "label": label, "writer": w, "level": lv, "cram": True}
    return jobs, key


def load(name: str):
    if name == "test":
        jobs, key = _jobs()
    elif name == "cram":
        jobs, key = cram_jobs()
    else:
        jobs, key = trap_jobs()
    cache = OUT / f"vals_{name}.pkl" if name != "test" else OUT / "vals.pkl"
    if cache.exists():
        return key, pickle.loads(cache.read_bytes())
    with ProcessPoolExecutor(8) as ex:
        vals = {iid: v for iid, _, v in ex.map(_render, jobs)}
    cache.write_bytes(pickle.dumps(vals))
    return key, vals


def main():
    ckpt = sys.argv[1]
    reader = Reader(ckpt)
    titles = {"test": "読み取りテストの 78 問", "trap": "罠（語彙にない名前・町名）", "cram": "詰め書きの数字（11・00・細い 1）"}
    for name in titles:
        print(f"\n# {titles[name]}")
        run(reader, *load(name))


def run(reader, key, vals):
    norm = lambda s: normalize(s).replace(" ", "")
    fixed, broke, still, n_ok = [], [], [], [0, 0]
    sources = Counter()
    for iid, crops in sorted(vals.items()):
        k = key[iid]
        kind = KIND[k["label"]]
        raw, ans, src = [], [], []
        for a in crops:
            res = check_field(reader, a, kind)
            raw.append(res["text"]); ans.append(res["answer"]); src.append(res["source"])
        sources.update(src)
        r_ok = norm("".join(raw)) == norm(k["text"]); a_ok = norm("".join(ans)) == norm(k["text"])
        n_ok[0] += r_ok; n_ok[1] += a_ok
        row = f"{iid} {k['level']}{'・詰め' if k['cram'] else ''} {k['label']}: 正解「{k['text']}」 画像「{''.join(raw)}」 → 答え「{''.join(ans)}」（{'/'.join(src)}）"
        (fixed if a_ok and not r_ok else broke if r_ok and not a_ok else still if not a_ok else []).append(row)
    print(f"画像だけ {n_ok[0]}/{len(vals)}  →  欄の知識あり {n_ok[1]}/{len(vals)}")
    print("答えの出どころ:", dict(sources))
    for name, rows in (("直った", fixed), ("壊した", broke), ("まだ違う", still)):
        print(f"\n## {name}（{len(rows)}）")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
