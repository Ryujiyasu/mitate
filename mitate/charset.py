"""認識対象の文字集合。

cp932（JIS X 0208 + NEC/IBM拡張）を基準にする。帳票で出る ①, ㈱, 髙, 﨑 などを含むため。
ラベルの正規化方針:
  - 全角英数記号（U+FF01-FF5E）は半角に寄せる（見た目で区別しにくいため）
  - 全角スペースは半角スペースに寄せる
  - 半角カタカナは当面対象外（濁点が別文字になり扱いが複雑なため）
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

BLANK = 0  # CTC blank の index


def normalize(text: str) -> str:
    out = []
    for ch in text:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            ch = chr(o - 0xFEE0)
        elif ch == "　":
            ch = " "
        elif ch == "¥":  # ¥ と ￥ は見た目で区別できない
            ch = "￥"
        out.append(ch)
    return "".join(out)


def _cp932_chars() -> list[str]:
    chars = set()
    for b in range(0x20, 0x7F):
        chars.add(chr(b))
    for hi in list(range(0x81, 0xA0)) + list(range(0xE0, 0xFD)):
        for lo in list(range(0x40, 0x7F)) + list(range(0x80, 0xFD)):
            try:
                ch = bytes([hi, lo]).decode("cp932")
            except UnicodeDecodeError:
                continue
            if len(ch) != 1:
                continue
            cat = unicodedata.category(ch)
            if cat.startswith("C"):  # 制御文字・私用領域など
                continue
            chars.add(ch)
    return sorted({normalize(c) for c in chars})


@lru_cache(maxsize=1)
def charset() -> list[str]:
    """index 0 は blank。1.. が文字。"""
    return ["<blank>"] + _cp932_chars()


@lru_cache(maxsize=1)
def char_to_index() -> dict[str, int]:
    return {c: i for i, c in enumerate(charset())}


def encode(text: str) -> list[int]:
    m = char_to_index()
    return [m[c] for c in normalize(text)]


def decode_ctc(indices) -> str:
    """greedy CTC: 連続重複をまとめて blank を除く。"""
    cs = charset()
    out, prev = [], BLANK
    for i in indices:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(cs[i])
        prev = i
    return "".join(out)


# 文字種の分類（テキスト生成で使う）
def _is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿" or "豈" <= c <= "﫿"


def groups() -> dict[str, list[str]]:
    cs = charset()[1:]
    kanji1, kanji2 = [], []
    for c in cs:
        if not _is_kanji(c):
            continue
        # JIS第1水準は cp932 の 0x889F-0x9872
        try:
            b = c.encode("cp932")
        except UnicodeEncodeError:
            kanji2.append(c)
            continue
        (kanji1 if len(b) == 2 and 0x889F <= int.from_bytes(b, "big") <= 0x9872 else kanji2).append(c)
    return {
        "hira": [c for c in cs if "ぁ" <= c <= "ん"],
        "kata": [c for c in cs if "ァ" <= c <= "ヶ"] + ["ー"],
        "kanji1": kanji1,
        "kanji2": kanji2,
        "digit": list("0123456789"),
        "upper": [chr(i) for i in range(0x41, 0x5B)],
        "lower": [chr(i) for i in range(0x61, 0x7B)],
        "ascii_sym": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
        "ja_sym": [c for c in "、。・「」『』（）［］【】〒※→←↑↓〇々〆ヶ〜～！？：；＝＋－×÷±℃％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼①②③④⑤⑥⑦⑧⑨⑩㈱㈲㈹㍉㌔㌢㍍㌘㌧㌃㌶㍑㍗㌍㌦㌣㌫㍊㌻㎜㎝㎞㎎㎏㏄㎡№℡ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ" if c in set(cs)],
        "all": cs,
    }
