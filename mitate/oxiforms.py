"""Oxi で「手書き記入済みの帳票」を描画し、文字ごとの正解つきデータにする。

流れ:
  1. python-docx で帳票の .docx を作る（表・罫線・ラベルは活字、記入欄は手書きフォント）
  2. Oxi (oxi docx-to-pdf) で PDF にする  ← Word と同じレイアウトで描画される
  3. pypdfium2 でページを画像にし、PDF の文字位置から行ごとの正解（画像の切り出し + テキスト）を作る

PDF のテキストは Oxi が埋め込んだものなので、正解は描画と必ず一致する。
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

from . import textgen

ROOT = Path(__file__).resolve().parent.parent
OXI = ROOT / "tools" / "oxi" / "oxi"

PRINT_FONTS = ["BIZ UDGothic", "BIZ UDMincho", "IPAGothic", "IPAMincho", "Noto Sans JP",
               "BIZ UDPGothic", "IPAPMincho", "Zen Kaku Gothic New", "Shippori Mincho"]


def _set_font(run, family: str, size_pt: float):
    run.font.name = family
    run.font.size = Pt(size_pt)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(k), family)


def _run_squeeze(run, scale_pct: int, spacing_twips: int):
    """Word の「文字の拡大/縮小」(w:w) と字間 (w:spacing) で詰め書きを表す。"""
    rpr = run._element.get_or_add_rPr()
    w = OxmlElement("w:w")
    w.set(qn("w:val"), str(scale_pct))
    rpr.append(w)
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:val"), str(spacing_twips))
    rpr.append(sp)


def add_handwriting(para, value: str, family: str, size_pt: float, rng: random.Random):
    """記入欄に手書きの文字列を入れる。3 割は欄が足りなくなった詰め書きにする。"""
    n = len(value)
    if n >= 4 and rng.random() < 0.3:
        k = rng.randint(n // 3, n - 2)
        end = rng.uniform(0.45, 0.8)
        _set_font(para.add_run(value[:k]), family, size_pt)
        for i in range(k, n):
            f = 1.0 - (1.0 - end) * (i - k + 1) / (n - k)
            r = para.add_run(value[i])
            _set_font(r, family, max(6, round(size_pt * (0.5 + 0.5 * f) * 2) / 2))
            _run_squeeze(r, int(100 * f), -int((1 - f) * 40))
    else:
        _set_font(para.add_run(value), family, size_pt)


def _cell_border(cell, sz=8):
    tcpr = cell._element.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tcpr.append(borders)


FIELDS = [
    ("氏名", "name"), ("フリガナ", "kana_name"), ("生年月日", "date"), ("住所", "address"),
    ("電話番号", "phone"), ("郵便番号", "postal"), ("金額", "money"), ("勤務先", "company"),
    ("備考", "free"), ("申請日", "date"), ("メールアドレス", "email"), ("口座番号", "digits"),
    ("品名", "free"), ("数量", "number"), ("事業所名", "company"), ("担当者", "name"),
]


@dataclass
class FormSpec:
    hw_family: str
    rows: list[tuple[str, str]]  # (ラベル, 記入内容)
    print_font: str
    label_pt: float
    hw_pt: float
    title: str


def make_spec(rng: random.Random, hw_family: str) -> FormSpec:
    n = rng.randint(6, 12)
    rows = []
    for label, kind in rng.sample(FIELDS, k=min(n, len(FIELDS))):
        rows.append((label, textgen.field(kind, rng)))
    return FormSpec(
        hw_family=hw_family,
        rows=rows,
        print_font=rng.choice(PRINT_FONTS),
        label_pt=rng.choice([9, 10, 10.5, 11, 12]),
        hw_pt=rng.uniform(11, 20),
        title=rng.choice(["申請書", "届出書", "申込書", "登録票", "変更届", "請求書", "受付票", "調査票"]),
    )


def build_docx(spec: FormSpec, path: Path, rng: random.Random):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Mm(rng.uniform(15, 25)))
    sec.top_margin = Mm(rng.uniform(15, 25))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_font(p.add_run(spec.title), spec.print_font, 16)
    table = doc.add_table(rows=len(spec.rows), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    label_w = Mm(rng.uniform(25, 40))
    # 記入欄が狭い帳票（折り返して 2 段に書く）も混ぜる
    value_w = Mm(rng.uniform(45, 80)) if rng.random() < 0.25 else Mm(160) - label_w
    table.columns[0].width, table.columns[1].width = label_w, value_w
    for i, (label, value) in enumerate(spec.rows):
        c0, c1 = table.rows[i].cells
        c0.width, c1.width = label_w, value_w
        _cell_border(c0)
        _cell_border(c1)
        _set_font(c0.paragraphs[0].add_run(label), spec.print_font, spec.label_pt)
        para = c1.paragraphs[0]
        para.paragraph_format.space_before = Pt(rng.uniform(2, 10))
        para.paragraph_format.space_after = Pt(rng.uniform(2, 10))
        add_handwriting(para, value, spec.hw_family, spec.hw_pt * rng.uniform(0.9, 1.1), rng)
    doc.save(str(path))


_PRINT_FILES: list[Path] | None = None


def print_font_files() -> list[Path]:
    """帳票の活字（PRINT_FONTS）のファイル。書き手の作業フォルダにも置かないと、Oxi が代替して文字化けする。"""
    global _PRINT_FILES
    if _PRINT_FILES is None:
        _PRINT_FILES = []
        for fam in PRINT_FONTS:
            r = subprocess.run(["fc-match", "-f", "%{file}", f"{fam}:style=Regular"], capture_output=True, text=True)
            if r.stdout and Path(r.stdout).exists():
                _PRINT_FILES.append(Path(r.stdout).resolve())
    return _PRINT_FILES


def _symlink(target, where: Path):
    try:
        os.symlink(target, where)
    except FileExistsError:  # 複数のプロセスが同時に作ることがある
        pass


def writer_oxi(family: str) -> Path:
    """書き手 1 人分の Oxi 作業フォルダ。そのフォントだけを見せる（全部見せると起動のたびに全フォントを読むので遅い）。

    tools/oxi/writers/HWSynthNNNN/
        oxi            本体へのハードリンク（実行ファイルの隣の fonts/ を CLI が探す）
        fonts/         その書き手のフォントと、帳票の活字（PRINT_FONTS）へのシンボリックリンク
        Fonts -> fonts Oxi の core は %SystemRoot%\\Fonts を探すので、SystemRoot をここに向ける
    """
    stem = family.replace(" ", "")  # "HW Synth 0004" -> "HWSynth0004"
    d = OXI.parent / "writers" / stem
    exe = d / "oxi"
    fonts = d / "fonts"
    if not (fonts / ".print").exists():
        fonts.mkdir(parents=True, exist_ok=True)
        src = ROOT / "fonts" / "hw" / f"{stem}.ttf"
        _symlink(src.resolve(), fonts / src.name)
        _symlink("fonts", d / "Fonts")
        for f in print_font_files():
            _symlink(f, fonts / f.name)
        (fonts / ".print").touch()
    if not exe.exists():
        tmp = d / f"oxi.{os.getpid()}"
        os.link(OXI, tmp)
        os.replace(tmp, exe)
    return exe


def oxi_to_pdf(docx: Path, pdf: Path, family: str | None = None):
    exe = writer_oxi(family) if family and family.startswith(("HW Synth", "HW Test")) else OXI
    env = dict(os.environ, SystemRoot=str(exe.parent))
    r = subprocess.run([str(exe), "docx-to-pdf", str(docx), str(pdf)], capture_output=True, text=True,
                       timeout=120, env=env)
    if r.returncode != 0 or not pdf.exists():
        raise RuntimeError(f"oxi failed: {r.stderr[-500:]}")


def _breaks_line(cur, c) -> bool:
    """c が cur（今の行）に続かないなら True。縦の重なりと横の間隔で判定する。"""
    top = min(x[2] for x in cur); bot = max(x[4] for x in cur)
    h = bot - top
    cy = (c[2] + c[4]) / 2
    if not (top - 0.2 * h <= cy <= bot + 0.2 * h):
        return True
    last = cur[-1]
    if c[1] < last[1] - 2:  # 左に戻った = 折り返し
        return True
    return c[1] - last[3] > h * 1.5


def extract_lines(pdf: Path, dpi: int = 200):
    """PDF → [(ページ画像, [(bbox_px, text, is_handwriting?)...])]。行 = 同じ基準線上の連続した文字。"""
    doc = pdfium.PdfDocument(str(pdf))
    out = []
    scale = dpi / 72
    for page in doc:
        h_pt = page.get_height()
        img = page.render(scale=scale, grayscale=True).to_pil()
        tp = page.get_textpage()
        n = tp.count_chars()
        chars = []
        for i in range(n):
            ch = tp.get_text_range(i, 1)
            # pdfium は字の大きさ・基準線が変わるたび（詰め書き、run の切れ目）に \r\n を差し込む。
            # 本当の改行かどうかは位置（_breaks_line）で決めるので、ここでは捨てる。
            if not ch or ch in "\r\n":
                continue
            l, b, r, t = tp.get_charbox(i, loose=True)
            chars.append((ch, l * scale, (h_pt - t) * scale, r * scale, (h_pt - b) * scale))
        lines, cur = [], []
        for c in chars + [None]:
            if c is None or (cur and _breaks_line(cur, c)):
                if cur:
                    text = "".join(x[0] for x in cur).strip()
                    if text:
                        x0 = min(x[1] for x in cur); y0 = min(x[2] for x in cur)
                        x1 = max(x[3] for x in cur); y1 = max(x[4] for x in cur)
                        lines.append(((x0, y0, x1, y1), text))
                cur = []
            if c is not None:
                cur.append(c)
        out.append((img, lines))
    return out


def demo(hw_family: str, out_dir: Path, seed: int = 0):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = make_spec(rng, hw_family)
    with tempfile.TemporaryDirectory() as td:
        dx, pdf = Path(td) / "f.docx", out_dir / f"form{seed}.pdf"
        build_docx(spec, dx, rng)
        oxi_to_pdf(dx, pdf, spec.hw_family)
    pages = extract_lines(pdf)
    img, lines = pages[0]
    img.save(out_dir / f"form{seed}.png")
    (out_dir / f"form{seed}.json").write_text(json.dumps({"spec": spec.__dict__, "lines": lines}, ensure_ascii=False, indent=1))
    return spec, lines


if __name__ == "__main__":
    import sys
    spec, lines = demo(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    for bbox, t in lines:
        print([round(v) for v in bbox], t)
