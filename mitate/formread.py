"""記入済みの帳票（スキャン・写真）を読み、元の .docx に書き戻す。

    python -m mitate.formread read  <様式.docx> <写真.png> --out out/    # 読む → out/result.json, out/filled.docx, out/filled.pdf
    python -m mitate.formread test  <様式.docx> --n 20 --out out/         # 合成の記入済み写真で試す

流れ:
  1. 様式（{{欄名}} 入りの .docx）から、Oxi で各欄の位置（docx_field_boxes）と空欄の様式の画像を作る
  2. 写真を空欄の様式に重ねる（特徴点の対応 → 射影変換）。傾き・遠近のゆがみはここで直る
  3. 欄ごとに切り出し、字のある行を見つけて読む。欄の名前から種類（氏名・日付・郵便番号…）を決め、
     欄の知識（fields.check_field）を使う。答えの出どころ（画像 / 推定 / 推測）も返す
  4. 読んだ値を元の .docx に差し込む（oxi-forms の oxi-fill.js）。人が確かめるべき欄には色を付ける
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
FORMFILL = ROOT / "tools" / "formfill.mjs"
DPI = 200
PT = DPI / 72.0

# 欄の名前 → 欄の種類（fields.check_field の種類）。上から順に最初に当たったもの
KIND_RULES = [
    (r"フリガナ|ふりがな|カナ", "kata"),
    (r"郵便", "postal"),
    (r"電話|TEL|FAX|携帯", "phone"),
    (r"住所|所在地", "address"),
    (r"日$|日付|年月日|生年月日", "date"),
    (r"額|金額|価格|料金|円", "money"),
    (r"会社|事業所|勤務先|法人|団体", "company"),
    (r"氏名|名前|担当者|代表者", "name"),
    (r"番号|数量|口座", "digits"),
]
# 答えの出どころ → 書き戻すときの色（Word の蛍光ペン）。画像だけで決まった欄は色なし
HIGHLIGHT = {
    "画像（自信低め）": "yellow",
    "画像（記入内容の確認が必要）": "yellow",
    "推定（画像＋知識）": "cyan",
    "推測（要確認）": "magenta",
}


def kind_of(name: str) -> str:
    base, _, fmt = name.partition(":")
    if fmt in ("和暦", "年月日"):
        return "date"
    for pat, kind in KIND_RULES:
        if re.search(pat, base):
            return kind
    return "free"


def _node(*args) -> dict:
    r = subprocess.run(["node", str(FORMFILL), *map(str, args)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def pdf_page(pdf: Path, dpi: int = DPI) -> np.ndarray:
    page = pdfium.PdfDocument(str(pdf))[0]
    return np.asarray(page.render(scale=dpi / 72, grayscale=True).to_pil().convert("L"))


@dataclass
class Field:
    name: str          # 様式の名前（「生年月日:和暦」）
    base: str          # 差し込みの名前（「生年月日」）
    kind: str
    box: tuple[int, int, int, int]  # 記入する範囲（px、空欄の様式の画像上）


@dataclass
class Template:
    docx: Path
    image: np.ndarray                   # 空欄の様式（DPI px、白地黒字）
    fields: list[Field] = field(default_factory=list)

    @classmethod
    def load(cls, docx: Path, workdir: Path | None = None) -> "Template":
        workdir = Path(workdir or tempfile.mkdtemp(prefix="tpl_"))
        _node("layout", docx, workdir)
        boxes = json.loads((workdir / "boxes.json").read_text())
        img = pdf_page(workdir / "blank.pdf")
        H, W = img.shape
        fields = []
        for f in boxes["fields"]:
            if f["page"] != 1:
                continue  # 試作は 1 ページ目だけ
            if f.get("cell"):
                c = f["cell"]
                box = (c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"])
            else:
                # 表の外の差し込み（「{{申請日:和暦}}」が段落の中にある）: その行の、差し込みの周り
                t = f["text"][0]
                box = (t["x"] - 2.5 * t["width"], t["y"] - 0.6 * t["height"],
                       min(W / PT - 30, t["x"] + t["width"] + 40), t["y"] + 1.6 * t["height"])
            x0, y0, x1, y1 = (int(round(v * PT)) for v in box)
            fields.append(Field(f["name"], f["name"].split(":")[0].split("[")[0], kind_of(f["name"]),
                                (max(0, x0), max(0, y0), min(W, x1), min(H, y1))))
        return cls(Path(docx), img, fields)


# ---------------------------------------------------------------- 写真を様式に重ねる
def register(photo: np.ndarray, tpl: Template, long_side: int = 1400):
    """写真 → 様式の座標の射影変換（3x3）。対応点が足りなければ None。"""
    def small(a):
        f = long_side / max(a.shape)
        return cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f
    t_s, ft = small(tpl.image)
    p_s, fp = small(photo)
    sift = cv2.SIFT_create(nfeatures=6000)
    kt, dt = sift.detectAndCompute(t_s, None)
    kp, dp = sift.detectAndCompute(p_s, None)
    if dt is None or dp is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dp, dt, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None
    src = np.float32([kp[g.queryIdx].pt for g in good]) / fp
    dst = np.float32([kt[g.trainIdx].pt for g in good]) / ft
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0 / ft)
    if Hm is None or inl.sum() < 12:
        return None
    return Hm


def warp_to_template(photo: np.ndarray, tpl: Template) -> np.ndarray | None:
    Hm = register(photo, tpl)
    if Hm is None:
        return None
    H, W = tpl.image.shape
    return cv2.warpPerspective(photo, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- 欄の中の行を切り出す
def field_lines(page: np.ndarray, blank: np.ndarray, box) -> list[np.ndarray]:
    """欄の中の、字が書かれた行の画像（上から順）。罫線と、様式にもともと印刷されている字は除く。"""
    x0, y0, x1, y1 = box
    pad = 5  # 罫線の太さ + 位置合わせのずれ
    crop = page[y0 + pad : y1 - pad, x0 + pad : x1 - pad].astype(np.float32)
    bl = blank[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
    if crop.size == 0:
        return []
    # 影・照明のむらを消す（背景の明るさで割る）→ 字のある所
    bg = cv2.dilate(crop, np.ones((15, 15), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    norm = np.clip(crop / np.maximum(bg, 1) * 255, 0, 255)
    printed = cv2.dilate((bl < 200).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # 様式の字・線
    # 字かどうかの閾値は、欄ごとの紙のざらつき（ノイズ）から決める。細いボールペン・薄い鉛筆でも拾えるように
    d = 255.0 - norm
    noise = 1.48 * float(np.median(np.abs(d - np.median(d)))) + 1.0
    thr = max(14.0, 6.0 * noise)
    ink = cv2.GaussianBlur(d, (3, 3), 0) > thr
    ink = ink.astype(np.uint8)
    faint = (cv2.GaussianBlur(d, (3, 3), 0) > 0.6 * thr).astype(np.uint8)  # 薄い・小さい字（詰め書きの最後の数字など）は、行の範囲に入れる
    ink[printed] = 0
    faint[printed] = 0
    # 残った罫線（位置合わせのずれで見える枠）: 欄の幅・高さの大半にわたる直線を消す
    h, w = ink.shape
    for k, ax in (((max(15, w // 3), 1), 1), ((1, max(15, h * 2 // 3)), 0)):
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        ink[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 0
    # ごみ（小さな点）を除く
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 6
    ink = keep[lab].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(faint, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 10
    faint = keep[lab].astype(np.uint8)
    rows = ink.sum(1)
    if rows.sum() < 30:
        return []
    # 行に分ける: 字のない帯で区切る（ただし字の高さの 1/4 未満の隙間は同じ行）
    on = rows > 0
    bands, start = [], None
    for y, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append([start, y]); start = None
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < max(4, 0.25 * (merged[-1][1] - merged[-1][0])):
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    out = []
    for a, b in merged:
        if b - a < 8 or ink[a:b].sum() < 30:
            continue
        hh = b - a
        cols = np.where(ink[a:b].sum(0) > 0)[0]
        fc = np.where(faint[a:b].sum(0) > 0)[0]
        # 薄い字は、濃い字の続き（隙間が字の高さの 1.5 倍以内）だけを足す
        lo, hi = cols[0], cols[-1]
        for x in fc[fc > hi]:
            if x - hi > 1.5 * hh:
                break
            hi = x
        for x in fc[fc < lo][::-1]:
            if lo - x > 1.5 * hh:
                break
            lo = x
        cols = np.array([lo, hi])
        m = int(0.25 * hh)
        ya, yb = max(0, a - m), min(h, b + m)
        xa, xb = max(0, cols[0] - m), min(w, cols[-1] + 1 + m)
        out.append(np.clip(norm[ya:yb, xa:xb], 0, 255).astype(np.uint8))
    return out


# ---------------------------------------------------------------- 読む
CHOICE_MARGIN = 6.0  # 選択肢が画像の読みより、対数確率でこれ以上悪くなければ「推定」


def read_choice(reader, image, options: list[str]) -> dict:
    """選択式の欄（新規・継続・変更）: 選択肢の中から、画像にいちばん合うものを選ぶ。"""
    from .fields import score
    r = reader.read([image])[0]
    sc = dict(score(reader, image, options + [r.text]))
    best = max(options, key=lambda o: sc.get(o, float("-inf")))
    if r.text in options:
        source = "画像" if r.conf >= 0.9 else "画像（自信低め）"
    elif sc.get(best, float("-inf")) > sc.get(r.text, 0.0) - CHOICE_MARGIN:
        source = "推定（画像＋知識）"
    else:
        source = "推測（要確認）"
    return {"text": r.text, "conf": r.conf, "answer": best if r.text not in options else r.text, "source": source,
            "issues": [], "suggestions": [(o, sc.get(o)) for o in options if o != best][:2]}


def read_form(photo: np.ndarray, tpl: Template, reader, choices: dict[str, list[str]] | None = None) -> dict:
    from .fields import check_field
    choices = choices or {}
    page = warp_to_template(photo, tpl)
    if page is None:
        return {"ok": False, "error": "様式と重ねられない（別の様式か、写りが悪い）", "fields": {}}
    res = {}
    for f in tpl.fields:
        lines = field_lines(page, tpl.image, f.box)
        if not lines:
            res[f.base] = {"answer": "", "source": "空欄", "conf": 1.0, "lines": 0}
            continue
        if f.base in choices:
            parts = [read_choice(reader, lines[0], choices[f.base])]
        else:
            parts = [check_field(reader, im, f.kind if len(lines) == 1 else "free") for im in lines]
        answer = "".join(p["answer"] for p in parts)
        # 欄全体の出どころ: いちばん弱い行に合わせる
        order = ["推測（要確認）", "推定（画像＋知識）", "画像（記入内容の確認が必要）", "画像（自信低め）", "画像"]
        source = min((p["source"] for p in parts), key=lambda s: order.index(s.split("/")[0]) if s.split("/")[0] in order else 0)
        res[f.base] = {"answer": answer, "source": source, "conf": round(min(p["conf"] for p in parts), 3),
                       "kind": f.kind, "lines": len(lines),
                       "issues": sum((p.get("issues", []) for p in parts), []),
                       "suggestions": [s for p in parts for s, _ in p.get("suggestions", [])][:3]}
    return {"ok": True, "fields": res, "_page": page}


# ---------------------------------------------------------------- 書き戻す
SHADE = {"yellow": "FFF2A8", "cyan": "BDEBFF", "magenta": "FFC8EE"}


def _highlight_template(docx: Path, colors: dict[str, str], out: Path):
    """確かめてほしい欄に色を付けた様式の写し。表の欄はセルの網掛け（Oxi の PDF にも出る）、
    表の外の差し込みは蛍光ペン（Word では見えるが、今の Oxi の PDF には出ない）。"""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc = Document(str(docx))

    def paras(container, cell=None):
        for p in container.paragraphs:
            yield p, cell
        for t in getattr(container, "tables", []):
            for row in t.rows:
                for c in row.cells:
                    yield from paras(c, c)

    def add(parent, tag, **attrs):
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        parent.append(el)

    for p, cell in paras(doc):
        for run in p.runs:
            for name, color in colors.items():
                if not re.search(r"\{\{\s*" + re.escape(name) + r"[\s:\[}]", run.text):
                    continue
                if cell is not None:
                    tcpr = cell._tc.get_or_add_tcPr()
                    for old in tcpr.findall(qn("w:shd")):
                        tcpr.remove(old)
                    add(tcpr, "w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": SHADE[color]})
                add(run._element.get_or_add_rPr(), "w:highlight", **{"w:val": color})
    doc.save(str(out))


def write_back(tpl: Template, result: dict, out_docx: Path, out_pdf: Path | None = None):
    values = {k: v["answer"] for k, v in result["fields"].items()}
    colors = {k: HIGHLIGHT[v["source"].split("/")[0]] for k, v in result["fields"].items()
              if v["source"].split("/")[0] in HIGHLIGHT}
    with tempfile.TemporaryDirectory() as td:
        hl = Path(td) / "tpl.docx"
        _highlight_template(tpl.docx, colors, hl)
        vj = Path(td) / "values.json"
        vj.write_text(json.dumps(values, ensure_ascii=False))
        _node("fill", hl, vj, out_docx, *( [out_pdf] if out_pdf else []))


# ---------------------------------------------------------------- 試す: 合成の記入済み写真
GEN_KIND = {"kata": "kana_name", "name": "name", "date": "date", "postal": "postal", "address": "address",
            "phone": "phone", "money": "money", "company": "company", "digits": "digits", "free": "free"}
CHOICES = {"申請区分": ["新規", "継続", "変更"]}
REASONS = ["店舗の改装のため", "新しい機械を導入するため", "事業の拡大に伴う設備投資", "省エネルギー設備への更新",
           "空き家を改修して住むため", "子どもの進学に伴う費用", "地域の祭りの開催費用として", "老朽化した屋根の修理",
           "販路開拓のための展示会出展", "従業員の研修費用", "太陽光発電設備の設置", "バリアフリー化の工事",
           "商品開発のための試作費", "ホームページの作成と広告", "災害で壊れた倉庫の建て替え", "介護のための住宅改修"]


def _value(f: Field, rng: random.Random) -> str:
    from . import textgen
    if f.base in CHOICES:
        return rng.choice(CHOICES[f.base])
    if f.kind == "free":
        return rng.choice(REASONS) + (rng.choice(["。", "", "に使います。"]) if rng.random() < 0.5 else "")
    if f.kind == "money":
        return f"{rng.randint(1, 3000) * 1000:,}"
    if f.kind == "date":
        y = rng.randint(1, 7) if rng.random() < 0.5 else rng.randint(20, 64)
        era = "令和" if y <= 7 else "昭和"
        return f"{era}{y}年{rng.randint(1, 12)}月{rng.randint(1, 28)}日"
    return textgen.field(GEN_KIND[f.kind], rng)


def fill_by_hand(tpl: Template, rng: random.Random, font, values: dict | None = None):
    """空欄の様式に手書きで記入した紙（白地黒字）と、書いた値。"""
    from . import render
    from .charset import groups
    page = tpl.image.astype(np.float32).copy()
    truth = {}
    cov = render.coverage(*font)
    for f in tpl.fields:
        v = (values or {}).get(f.base) or _value(f, rng)
        v = render.fit_text(v, cov, rng, groups()["all"])
        x0, y0, x1, y1 = f.box
        bw, bh = x1 - x0, y1 - y0
        multi = bh > 160 and len(v) > 12
        chunks = [v[i : i + 18] for i in range(0, len(v), 18)] if multi else [v]
        size = int(min(58, bh * (0.55 if not multi else 0.35)) * rng.uniform(0.8, 1.05))
        y = y0 + (bh - size * 1.4 * len(chunks)) / 2 + (8 if multi else 0)
        for ch in chunks:
            a = render.draw_line(ch, font[0], font[1], rng, True, size=size)
            if a.shape[1] > bw - 20:  # 欄に入らない → 縮めて書く（詰め書き）
                a = cv2.resize(a, (bw - 20, int(a.shape[0] * (bw - 20) / a.shape[1] * 0.5 + a.shape[0] * 0.5)))
            ah, aw = a.shape
            x = x0 + 12 + int(rng.uniform(0, max(0, min(60, bw - 24 - aw))))
            if f.box[2] - f.box[0] < 400:  # 段落の中の差し込み（日付）は右に寄せて書く
                x = max(x0, x1 - aw - 10)
            yy = int(np.clip(y + rng.uniform(-6, 6), y0 + 4, max(y0 + 4, y1 - ah - 4)))
            region = page[yy : yy + ah, x : x + aw]
            a = a[: region.shape[0], : region.shape[1]].astype(np.float32)
            ink = rng.uniform(0, 70)
            page[yy : yy + a.shape[0], x : x + a.shape[1]] = np.minimum(region, ink + a / 255 * (255 - ink))
            y += size * 1.4
        truth[f.base] = "".join(chunks)
    return np.clip(page, 0, 255).astype(np.uint8), truth


def photograph(page: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """紙をスマホで撮ったらしい写真: 机の上・斜め・遠近・影・照明のむら・ぼけ・JPEG。"""
    from . import render
    import io
    from PIL import Image
    H, W = page.shape
    m = int(0.08 * max(H, W))
    desk = rng.uniform(60, 150)
    canvas = np.full((H + 2 * m, W + 2 * m), desk, np.float32)
    canvas[m : m + H, m : m + W] = page * rng.uniform(0.85, 1.0)
    src = np.float32([[m, m], [m + W, m], [m + W, m + H], [m, m + H]])
    j = 0.05 * strength * max(H, W)
    dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    ang = np.deg2rad(rng.uniform(-6, 6) * strength)
    c = np.float32([(W + 2 * m) / 2, (H + 2 * m) / 2])
    R = np.float32([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (dst - c) @ R.T + c
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(canvas, M, canvas.shape[::-1], borderValue=float(desk))
    if rng.random() < 0.6:
        img = render.uneven_light(img, rng.uniform(0.1, 0.5) * strength, rng)
    if rng.random() < 0.4:
        img = render.shadow(img, rng.uniform(0.2, 0.5) * strength, rng, soft=rng.uniform(20, 120))
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 6), img.shape)
    f = rng.uniform(0.65, 1.0)  # スマホの画素数（A4 全体で 150〜200dpi 相当）
    img = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=rng.randint(60, 90))
    return np.asarray(Image.open(buf).convert("L"))


def _norm(s: str) -> str:
    from .charset import normalize
    return "".join(normalize(s).split())


def overlay(page: np.ndarray, tpl: Template, result: dict) -> np.ndarray:
    """確認用の画像: 重ねた写真に、欄ごとの読みと出どころの色を描く。"""
    from PIL import Image, ImageDraw, ImageFont
    from .gallery import LABEL_FONT
    col = {"yellow": (240, 200, 0), "cyan": (0, 170, 220), "magenta": (220, 0, 160)}
    im = Image.fromarray(page).convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(LABEL_FONT, 26)
    for f in tpl.fields:
        r = result["fields"].get(f.base, {})
        c = col.get(HIGHLIGHT.get(r.get("source", "").split("/")[0], ""), (0, 150, 0))
        d.rectangle(f.box, outline=c, width=4)
        d.text((f.box[0] + 6, f.box[3] - 32), f"{r.get('answer', '')}  [{r.get('source', '')}]", font=font, fill=c)
    return np.asarray(im)


def test(docx: Path, n: int, out: Path, ckpt: str, seed: int = 7):
    from . import render
    from .infer import Reader
    out.mkdir(parents=True, exist_ok=True)
    tpl = Template.load(docx, out / "template")
    reader = Reader(ckpt, device="cpu")
    writers = [(str(p), i) for p, i in render.hw_fonts(True)]
    rng = random.Random(seed)
    stats = {"fields": 0, "right": 0, "silent_wrong": 0, "flagged": 0, "flagged_wrong": 0, "unaligned": 0}
    by_field: dict[str, list[int]] = {}
    wrong, secs = [], []
    for k in range(n):
        font = rng.choice(writers)
        paper, truth = fill_by_hand(tpl, rng, font)
        photo = photograph(paper, rng)
        t0 = time.time()
        res = read_form(photo, tpl, reader, CHOICES)
        secs.append(time.time() - t0)
        if not res["ok"]:
            stats["unaligned"] += 1
            continue
        for name, t in truth.items():
            r = res["fields"][name]
            ok = _norm(r["answer"]) == _norm(t)
            flagged = r["source"].split("/")[0] in HIGHLIGHT
            stats["fields"] += 1; stats["right"] += ok
            stats["flagged"] += flagged
            stats["flagged_wrong"] += flagged and not ok
            stats["silent_wrong"] += (not ok) and not flagged
            by_field.setdefault(name, []).append(ok)
            if not ok:
                wrong.append(f"{k:02d} {name}: 正解「{t}」 → 「{r['answer']}」（{r['source']}）")
        if k < 3:
            cv2.imwrite(str(out / f"photo{k}.jpg"), photo)
            cv2.imwrite(str(out / f"check{k}.png"), cv2.cvtColor(overlay(res["_page"], tpl, res), cv2.COLOR_RGB2BGR))
            write_back(tpl, {"fields": {a: b for a, b in res["fields"].items()}}, out / f"filled{k}.docx", out / f"filled{k}.pdf")
            (out / f"result{k}.json").write_text(json.dumps({"truth": truth, "fields": res["fields"]}, ensure_ascii=False, indent=1))
    lines = [f"# {docx.name}: 合成の記入済み写真 {n} 枚（学習に使っていない書き手）",
             f"欄の正解 {stats['right']}/{stats['fields']}  色付き（人が確かめる）{stats['flagged']}  "
             f"うち間違い {stats['flagged_wrong']}  **色なしで間違い {stats['silent_wrong']}**  重ねられなかった写真 {stats['unaligned']}",
             f"1 枚の時間（CPU、位置合わせ＋10 欄の読み取り）: 中央値 {np.median(secs):.2f} 秒",
             "", "欄ごと: " + " / ".join(f"{k} {sum(v)}/{len(v)}" for k, v in by_field.items()), "", "## 間違い"] + wrong
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read"); r.add_argument("docx"); r.add_argument("photo"); r.add_argument("--out", default="out")
    r.add_argument("--choices", default=None, help='選択式の欄（JSON）: {"申請区分": ["新規", "継続", "変更"]}')
    t = sub.add_parser("test"); t.add_argument("docx"); t.add_argument("--n", type=int, default=20); t.add_argument("--out", default="runs/formread")
    for p in (r, t):
        p.add_argument("--ckpt", default="runs/s2cam/last.pt")
    a = ap.parse_args()
    if a.cmd == "test":
        test(Path(a.docx), a.n, Path(a.out), a.ckpt)
        return
    from .infer import Reader
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tpl = Template.load(Path(a.docx), out / "template")
    photo = cv2.imread(a.photo, cv2.IMREAD_GRAYSCALE)
    res = read_form(photo, tpl, Reader(a.ckpt, device="cpu"), json.loads(Path(a.choices).read_text()) if a.choices else None)
    page = res.pop("_page", None)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    if res["ok"]:
        write_back(tpl, res, out / "filled.docx", out / "filled.pdf")
        cv2.imwrite(str(out / "check.png"), cv2.cvtColor(overlay(page, tpl, res), cv2.COLOR_RGB2BGR))
    print(json.dumps({k: (v["answer"], v["source"]) for k, v in res["fields"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
