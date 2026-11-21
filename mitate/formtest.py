"""読み取りテスト（第 2 版）: 帳票の中の手書きを人に読んでもらい、OCR の読みが人の感覚に合うかを判定してもらう。

    python -m mitate.formtest make --ckpt runs/crnn_v1/last.pt   # data/formtest/ に問題と正解を作る
    python -m mitate.formtest score answers.json                 # 回答を集計する

第 1 版との違い:
  - 画像の劣化（ぼけ・低解像度）ではなく、手書きの崩れを軸にする。
    同じ書き手の性格のまま雑さだけを 5 段階に変えたフォント（HW Test NNNN L1〜L5）と、欄が足りない詰め書き
  - 切り出した 1 行ではなく、ラベル（氏名・住所…）と記入欄が並んだ帳票の一部を Oxi で描いて見せる
  - 人が読んだあとに OCR の読みと迷った字を見せ、「人の感覚として自然か」を判定してもらう
  - 問題の文字は、人が読み方を知っていて入力できる言葉だけ
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from docx import Document
from docx.shared import Mm, Pt
from PIL import Image

from . import oxiforms
from .charset import normalize
from .humantest import _easy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "formtest"
WRITERS = [910, 911]
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
FIELDS = [("氏名", "name"), ("住所", "address"), ("電話番号", "phone"), ("生年月日", "date"),
          ("金額", "money"), ("会社名", "company"), ("郵便番号", "postal"), ("担当", "kata")]


def _text(kind: str, rng: random.Random) -> str:
    while True:
        t = " ".join(normalize(_easy(kind, rng)).split())
        if 3 <= len(t) <= 14:
            return t


def _docx(path: Path, label: str, value: str, family: str, cram: bool, rng: random.Random):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    t = doc.add_table(rows=1, cols=2)
    t.autofit = False
    lw, vw = Mm(rng.uniform(24, 32)), Mm(rng.uniform(62, 80) if not cram else rng.uniform(42, 52))
    t.columns[0].width, t.columns[1].width = lw, vw
    c0, c1 = t.rows[0].cells
    c0.width, c1.width = lw, vw
    oxiforms._cell_border(c0)
    oxiforms._cell_border(c1)
    oxiforms._set_font(c0.paragraphs[0].add_run(label), "BIZ UDGothic", 10.5)
    p = c1.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    size = rng.uniform(14, 17)
    n = len(value)
    if cram and n >= 4:
        # 欄が足りなくなって、後ろほど字が小さく・詰まる
        k = max(1, n // 2)
        oxiforms._set_font(p.add_run(value[:k]), family, size)
        for i in range(k, n):
            f = 1.0 - 0.5 * (i - k + 1) / (n - k)
            r = p.add_run(value[i])
            oxiforms._set_font(r, family, max(7, round(size * (0.55 + 0.45 * f) * 2) / 2))
            oxiforms._run_squeeze(r, int(100 * f), -int((1 - f) * 40))
    else:
        oxiforms._set_font(p.add_run(value), family, size)
    doc.save(str(path))


def _render(args):
    """1 問を Oxi で描く。帳票の切り出し画像と、記入欄の行画像（OCR 用）を返す。"""
    iid, label, value, family, cram, seed = args
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as td:
        dx, pdf = Path(td) / "f.docx", Path(td) / "f.pdf"
        _docx(dx, label, value, family, cram, rng)
        oxiforms.oxi_to_pdf(dx, pdf, family)
        img, lines = oxiforms.extract_lines(pdf, dpi=200)[0]
    page = np.asarray(img.convert("L"))
    ink = np.where(page < 200)
    y0, y1, x0, x1 = ink[0].min(), ink[0].max(), ink[1].min(), ink[1].max()
    snippet = page[max(0, y0 - 12): y1 + 12, max(0, x0 - 12): x1 + 12]
    # 記入欄の行（ラベル以外）を、OCR 用に切り出す
    vals = []
    for (a, b, c, d), text in lines:
        if normalize(text).replace(" ", "") == normalize(label):
            continue
        h = d - b
        vals.append(page[int(max(0, b - 0.25 * h)): int(d + 0.25 * h), int(max(0, a - 0.3 * h)): int(c + 0.3 * h)])
    return iid, snippet, vals


def _png_uri(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jobs():
    rng = random.Random(1120)
    jobs, key = [], {}
    n = 0
    for w in WRITERS:
        for lv in LEVELS:
            for label, kind in rng.sample(FIELDS, 6):
                n += 1
                iid = f"f{n:03d}"
                value = _text(kind, rng)
                jobs.append((iid, label, value, f"HW Test {w:04d} {lv}", False, rng.randrange(1 << 30)))
                key[iid] = {"text": value, "label": label, "writer": w, "level": lv, "cram": False}
        for lv in ["L2", "L3", "L4"]:
            for label, kind in rng.sample([f for f in FIELDS if f[1] in ("name", "address", "company", "money")], 3):
                n += 1
                iid = f"f{n:03d}"
                value = _text(kind, rng)
                jobs.append((iid, label, value, f"HW Test {w:04d} {lv}", True, rng.randrange(1 << 30)))
                key[iid] = {"text": value, "label": label, "writer": w, "level": lv, "cram": True}
    return jobs, key


def evaluate(ckpt: str, workers: int = 8):
    """公開中のテストは書き換えずに、同じ 78 問で OCR だけを採点する（追加学習の前後比較用）。"""
    import pickle
    from .infer import Reader
    jobs, key = _jobs()
    cache = OUT / "vals.pkl"
    if cache.exists():
        vals = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor(workers) as ex:
            vals = {iid: v for iid, _, v in ex.map(_render, jobs)}
        cache.write_bytes(pickle.dumps(vals))
    reader = Reader(ckpt)
    norm = lambda s: normalize(s).replace(" ", "")
    ok = {iid: norm("".join(r.text for r in reader.read(v))) == norm(key[iid]["text"]) for iid, v in vals.items()}
    print(f"OCR の正解率 {np.mean(list(ok.values())) * 100:.0f}%（{ckpt}）")
    for lv in LEVELS:
        for cram in (False, True):
            g = [ok[i] for i, k in key.items() if k["level"] == lv and k["cram"] == cram]
            if g:
                print(f"  {lv} {'詰め' if cram else '    '} {sum(g)}/{len(g)}")


def make(ckpt: str, workers: int = 4):
    from .fields import alternatives
    from .infer import Reader
    jobs, key = _jobs()
    OUT.mkdir(parents=True, exist_ok=True)
    reader = Reader(ckpt)
    items = []
    with ProcessPoolExecutor(workers) as ex:
        for iid, snippet, vals in ex.map(_render, jobs):
            texts, doubts, offset = [], [], 0
            for v in vals:
                r = reader.read([v])[0]
                slots = alternatives(reader, v, k=3, conf_below=0.9)
                for i, s in enumerate(slots):
                    if len(s) > 1:
                        doubts.append({"i": offset + i, "c": s[0], "alts": s[1:3]})
                texts.append(r.text)
                offset += len(r.text)
            ocr = "".join(texts)
            key[iid]["ocr"] = ocr
            key[iid]["ocr_ok"] = normalize(ocr).replace(" ", "") == normalize(key[iid]["text"]).replace(" ", "")
            items.append({"id": iid, "src": _png_uri(snippet), "label": key[iid]["label"], "ocr": ocr, "doubts": doubts})
            print(iid, key[iid]["level"], "詰め" if key[iid]["cram"] else "", key[iid]["text"], "→", ocr, flush=True)
    items.sort(key=lambda x: x["id"])
    (OUT / "site").mkdir(exist_ok=True)
    (OUT / "site" / "items.json").write_text(json.dumps(items, ensure_ascii=False))
    (OUT / "key.json").write_text(json.dumps({"ckpt": ckpt, "items": key}, ensure_ascii=False, indent=1))
    ok = np.mean([k["ocr_ok"] for k in key.values()])
    print(f"{len(items)} 問を作成。OCR の正解率 {ok * 100:.0f}%（{ckpt}）")


def score(answers_path: str):
    key = json.loads((OUT / "key.json").read_text())["items"]
    ans = json.loads(Path(answers_path).read_text())  # [{participant, id, text, confidence, judge, ms}]
    norm = lambda s: "".join(normalize(s).split())
    rows = []
    for a in ans:
        k = key.get(a["id"])
        if k:
            rows.append(dict(k, **a, human_ok=norm(a["text"]) == norm(k["text"])))
    out = ["# 読み取りテスト（帳票の中の手書き）の集計\n", f"回答 {len(rows)} 件\n",
           "| 雑さ | 詰め書き | 人の正解率 | 人が「自信あり」 | OCR の正解率 | 「人ならこう間違えない」 |", "|---|---|---|---|---|---|"]
    for lv in LEVELS:
        for cram in (False, True):
            g = [r for r in rows if r["level"] == lv and r["cram"] == cram]
            if not g:
                continue
            wrong = [r for r in g if not r["ocr_ok"]]
            inhuman = [r for r in wrong if r.get("judge") == "inhuman"]
            out.append(f"| {lv} | {'あり' if cram else 'なし'} | {np.mean([r['human_ok'] for r in g]) * 100:.0f}% | "
                       f"{np.mean([r['confidence'] == 'sure' for r in g]) * 100:.0f}% | "
                       f"{np.mean([r['ocr_ok'] for r in g]) * 100:.0f}% | "
                       f"{len(inhuman)}/{len(wrong)} |")
    bad = [r for r in rows if not r["ocr_ok"] and r.get("judge") == "inhuman"]
    out.append("\n## 「人ならこうは間違えない」と判定された OCR の誤り\n")
    for r in bad[:40]:
        out.append(f"- {r['id']}（{r['level']}{'・詰め' if r['cram'] else ''}）正解「{r['text']}」→ OCR「{r['ocr']}」")
    (OUT / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    s = sub.add_parser("score")
    s.add_argument("answers")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.ckpt, a.workers)
    elif a.cmd == "eval":
        evaluate(a.ckpt)
    else:
        score(a.answers)


if __name__ == "__main__":
    main()
