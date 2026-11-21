"""1 行の画像を描いて、スキャンらしく汚す。"""
from __future__ import annotations

import glob
import io
import os
import time
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .charset import normalize

ROOT = Path(__file__).resolve().parent.parent
H = 48  # 認識器に入れる行画像の高さ

_LF = Path.home() / ".local/share/fonts"
_SYS = Path("/usr/share/fonts/opentype/noto")

# 活字。学習用と、評価専用（学習に一切使わない書体ファミリー）に分ける
PRINT_TRAIN = [
    (_LF / "sekou/BIZUDGothic-Regular.ttf", 0), (_LF / "sekou/BIZUDGothic-Bold.ttf", 0),
    (_LF / "sekou/BIZUDMincho-Regular.ttf", 0), (_LF / "sekou/BIZUDMincho-Bold.ttf", 0),
    (_LF / "sekou/BIZUDPGothic-Regular.ttf", 0), (_LF / "sekou/BIZUDPGothic-Bold.ttf", 0),
    (_LF / "sekou/BIZUDPMincho-Regular.ttf", 0),
    (_LF / "sekou/ipag.ttf", 0), (_LF / "sekou/ipagp.ttf", 0), (_LF / "sekou/ipam.ttf", 0), (_LF / "sekou/ipamp.ttf", 0),
    (_LF / "NotoSansJP-400.ttf", 0), (_LF / "NotoSansJP-700.ttf", 0), (_LF / "NotoSansJP-900.ttf", 0),
    (_SYS / "NotoSansCJK-Regular.ttc", 0), (_SYS / "NotoSansCJK-Bold.ttc", 0),
    (_SYS / "NotoSerifCJK-Regular.ttc", 0), (_SYS / "NotoSerifCJK-Bold.ttc", 0),
    (_LF / "oxi-candidates/ShipporiMincho-Regular.ttf", 0), (_LF / "oxi-candidates/ShipporiMincho-Bold.ttf", 0),
    (_LF / "oxi-candidates/ZenKakuGothicNew-Regular.ttf", 0), (_LF / "oxi-candidates/ZenKakuGothicNew-Bold.ttf", 0),
    (_LF / "oxi-candidates/IBMPlexSansJP-Regular.ttf", 0), (_LF / "oxi-candidates/IBMPlexSansJP-Bold.ttf", 0),
    (_LF / "oxi-candidates/OxiGothic.ttf", 0), (_LF / "oxi-candidates/OxiMincho.ttf", 0),
]
PRINT_HELDOUT = [
    (_LF / "oxi-candidates/ZenOldMincho-Regular.ttf", 0), (_LF / "oxi-candidates/ZenOldMincho-Bold.ttf", 0),
    (_LF / "oxi-candidates/MPLUS1p-Regular.ttf", 0), (_LF / "oxi-candidates/MPLUS1p-Bold.ttf", 0),
    (_LF / "ZenMaruGothic-Black.ttf", 0), (_LF / "YuseiMagic.ttf", 0),
]
HW_DIR = ROOT / "fonts" / "hw"
HW_HELDOUT_FROM = 270  # これ以降の書き手は評価専用


def hw_fonts(heldout: bool) -> list[tuple[Path, int]]:
    out = []
    now = time.time()
    for p in sorted(glob.glob(str(HW_DIR / "HWSynth*.ttf"))):
        if now - os.path.getmtime(p) < 60:  # 生成中のファイルは使わない
            continue
        idx = int(Path(p).stem[-4:])
        if (idx >= HW_HELDOUT_FROM) == heldout:
            out.append((Path(p), 0))
    return out


@lru_cache(maxsize=None)
def coverage(path: str, index: int) -> frozenset[str]:
    f = TTFont(path, fontNumber=index, lazy=True) if path.endswith(".ttc") else TTFont(path, lazy=True)
    return frozenset(chr(c) for c in f.getBestCmap())


_font_cache: dict = {}


def get_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    key = (path, index, size)
    f = _font_cache.get(key)
    if f is None:
        if len(_font_cache) > 2000:
            _font_cache.clear()
        f = ImageFont.truetype(path, size, index=index)
        _font_cache[key] = f
    return f


def fit_text(text: str, cov: frozenset[str], rng: random.Random, fallback_pool: list[str]) -> str:
    """フォントにない字を、フォントにある字へ置き換える（ラベルと描画を一致させる）。"""
    out = []
    for c in normalize(text):
        if c == " " or c in cov:
            out.append(c)
        else:
            for _ in range(20):
                r = rng.choice(fallback_pool)
                if r in cov:
                    out.append(r)
                    break
    return "".join(out).strip()


@dataclass
class Sample:
    image: np.ndarray  # uint8 (H, W)、白地に黒
    text: str
    writer: int        # 手書きフォントの書き手番号。活字は -1


def draw_line(text: str, font_path: str, font_index: int, rng: random.Random, handwriting: bool,
              meta: dict | None = None, size: int | None = None, hurry: float | None = None,
              fatigue: float | None = None, plain: bool = False, state_cap: float = 1.0) -> np.ndarray:
    """meta を渡すと、使ったパラメータを書き込む（読める/読めないの境目の分析用）。
    size / hurry / fatigue を渡すとその値に固定する。plain=True で字間・詰め書き・揺れの乱数を切る（見本用）。"""
    meta = {} if meta is None else meta
    size_r = rng.randint(28, 52)
    size = size_r if size is None else size
    font = get_font(font_path, font_index, size)
    tracking = rng.uniform(-0.04, 0.15) * size if rng.random() < 0.5 else 0.0
    # 状態の因子（手書きのみ）。ランダムな揺れではなく、決まった方向に字を変える
    #   急ぎ (hurry):   字間が詰まる・右に流れる（傾く）・基準線が流れる
    #   疲れ (fatigue): 行の後ろほど形の揺れと上下のぶれが大きくなる
    hurry_fix, fatigue_fix = hurry, fatigue
    hurry = fatigue = 0.0
    if handwriting:
        tracking = rng.uniform(-0.05, 0.25) * size
        hurry = (rng.random() ** 2 if rng.random() < 0.5 else 0.0) * state_cap
        fatigue = (rng.random() ** 2 if rng.random() < 0.4 else 0.0) * state_cap
    hurry = hurry if hurry_fix is None else hurry_fix
    fatigue = fatigue if fatigue_fix is None else fatigue_fix
    if handwriting:
        if plain:
            tracking = 0.08 * size
        tracking -= 0.12 * hurry * size
    # 詰め書き: 欄が足りなくなり、途中から後ろほど字が小さく・字間が狭くなる
    n = len(text)
    scales = [1.0] * n
    sxs = [1.0] * n  # 横だけの縮み（Word の「文字の拡大/縮小」w:w。欄の端で字が細くなる）
    if handwriting and n >= 4 and rng.random() < 0.35 and not plain:
        k = rng.randint(n // 3, n - 2)
        end = rng.uniform(0.4, 0.8)
        squeeze = rng.choice([0.0, rng.uniform(0.3, 1.2)])  # 0 = 小さくなるだけ、>0 = 横にも潰れる
        for i in range(k, n):
            scales[i] = 1.0 - (1.0 - end) * (i - k + 1) / (n - k)
            sxs[i] = max(0.35, 1.0 - (1.0 - scales[i]) * squeeze)
    meta.update(font_px=size, tracking=round(tracking / size, 3), cram=round(min(s * x for s, x in zip(scales, sxs)), 3),
                handwriting=handwriting, hurry=round(hurry, 3), fatigue=round(fatigue, 3))
    # 1 文字ずつ置く（字間の調整・手書きの上下の揺れ）
    xs, fonts, x = [], [], size * 0.3
    for ch, sc, sx in zip(text, scales, sxs):
        f = font if sc == 1.0 else get_font(font_path, font_index, max(8, int(size * sc)))
        xs.append(x)
        fonts.append((f, sc))
        x += f.getlength(ch) * sx + tracking * sc
    W = int(x + size * 0.3) + 1
    drift = (rng.uniform(-0.04, 0.04) - 0.06 * hurry) if handwriting else 0.0
    rise = abs(drift) * W  # 行が流れる分だけ縦に広げる（字が切れないように）
    Hc = int(size * 2.0 + rise)
    img = Image.new("L", (max(W, 8), Hc), 0)
    d = ImageDraw.Draw(img)
    base = size * 0.45 + (rise if drift < 0 else 0)
    # 毎回の揺れ: 同じ人の同じ字のばらつきは、他人との差よりずっと小さい（筆跡鑑定が成り立つ理由）。
    # なのでごく小さく抑え、大きな変化は状態の因子（急ぎ・疲れ・詰め書き）で表す
    wobble = rng.uniform(0.0, 0.25) if handwriting else 0.0
    if plain:
        wobble = 0.0
    meta["wobble"] = round(wobble, 3)
    for i, (ch, cx, (f, sc), sx) in enumerate(zip(text, xs, fonts, sxs)):
        prog = i / max(1, n - 1)  # 行のどこまで来たか（疲れは後ろほど出る）
        jit = 0.03 + 0.05 * fatigue * prog
        y = base + drift * cx + (rng.gauss(0, size * jit) if handwriting else 0) + size * (1 - sc) * 0.55
        amt = wobble + 0.35 * hurry + 0.6 * fatigue * prog
        if (amt > 0 or sx < 1.0) and not ch.isspace():
            _draw_wobbly(img, ch, f, cx, y, amt, rng, lean=0.12 * hurry, sx=sx)
        else:
            d.text((cx, y), ch, font=f, fill=255)
    a = np.asarray(img)
    ys = np.where(a.max(axis=1) > 0)[0]
    if len(ys) == 0:
        return np.full((H, 16), 255, np.uint8)
    top = max(0, ys[0] - int(rng.uniform(0.05, 0.35) * size))
    bot = min(a.shape[0], ys[-1] + 1 + int(rng.uniform(0.05, 0.35) * size))
    a = a[top:bot]
    return 255 - a  # 白地に黒


def _draw_wobbly(img: Image.Image, ch: str, font, x: float, y: float, amount: float, rng: random.Random,
                 lean: float = 0.0, sx: float = 1.0):
    """1 字を、その場限りの小さな変形（大きさ・回転・せん断・局所のゆがみ）をかけて描く。sx < 1 で横に潰す。"""
    bbox = font.getbbox(ch)
    pad = int(font.size * 0.3)
    w, h = bbox[2] + 2 * pad, bbox[3] + 2 * pad
    if w <= 0 or h <= 0:
        return
    tile = Image.new("L", (w, h), 0)
    ImageDraw.Draw(tile).text((pad, pad), ch, font=font, fill=255)
    t = np.asarray(tile, dtype=np.float32)
    pad_y = pad
    if sx < 1.0:
        w = max(2, int(round(w * sx)))
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
        pad = int(round(pad * sx))
    cx, cy = w / 2, h / 2
    s = 1 + rng.gauss(0, 0.05 * amount)
    M = cv2.getRotationMatrix2D((cx, cy), rng.gauss(0, 3.0 * amount), s)
    M[0, 1] += rng.gauss(0, 0.06 * amount) - lean  # 急ぐと右に流れる
    M[1, 0] += rng.gauss(0, 0.04 * amount)
    t = cv2.warpAffine(t, M, (w, h))
    # 局所のゆがみ（なめらかな変位場）
    if amount > 0.2:
        g = np.random.default_rng(rng.randrange(1 << 30))
        dx = cv2.GaussianBlur(g.normal(0, 1, (h, w)).astype(np.float32), (0, 0), font.size * 0.15)
        dy = cv2.GaussianBlur(g.normal(0, 1, (h, w)).astype(np.float32), (0, 0), font.size * 0.15)
        k = font.size * 0.04 * amount / max(1e-6, float(np.abs(dx).max() + np.abs(dy).max()) / 2)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        t = cv2.remap(t, gx + dx * k, gy + dy * k, cv2.INTER_LINEAR)
    X, Y = int(round(x)) - pad, int(round(y)) - pad_y
    base = np.asarray(img).copy()
    H0, W0 = base.shape
    x0, y0, x1, y1 = max(0, X), max(0, Y), min(W0, X + w), min(H0, Y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = t[y0 - Y : y1 - Y, x0 - X : x1 - X]
    base[y0:y1, x0:x1] = np.maximum(base[y0:y1, x0:x1], np.clip(region, 0, 255).astype(np.uint8))
    img.paste(Image.fromarray(base))


# ---------------------------------------------------------------- 縦書き
# 縦書きで向きが変わる字（90° 回して描く）、右上に寄る字
VERT_ROTATE = set("ー－―‐-–—〜～~（）()「」『』【】［］[]｛｝{}〔〕＜＞<>…‥＝=→←:：;；")
VERT_SMALL = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
VERT_PUNCT = set("、。，．,.")
_DIG = set("0123456789")


def _vtokens(text: str, rng: random.Random) -> list[str]:
    """縦書きの 1 マスずつに分ける。2 桁の数字は半分の確率で縦中横（1 マスに横並び）。"""
    out, i = [], 0
    while i < len(text):
        two = text[i : i + 2]
        if (len(two) == 2 and set(two) <= _DIG and (i + 2 >= len(text) or text[i + 2] not in _DIG)
                and (i == 0 or text[i - 1] not in _DIG) and rng.random() < 0.5):
            out.append(two)
            i += 2
        else:
            out.append(text[i])
            i += 1
    return out


def _tile(tok: str, font, sx: float = 1.0) -> np.ndarray:
    """1 マス分の字を、インクのある範囲だけ切り出した白字（float32）。"""
    bbox = font.getbbox(tok)
    w, h = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
    tile = Image.new("L", (w + 4, h + 4), 0)
    ImageDraw.Draw(tile).text((2 - bbox[0], 2 - bbox[1]), tok, font=font, fill=255)
    t = np.asarray(tile, dtype=np.float32)
    if sx != 1.0:
        t = cv2.resize(t, (max(2, int(t.shape[1] * sx)), t.shape[0]), interpolation=cv2.INTER_AREA)
    return t


def _warp(t: np.ndarray, amount: float, rng: random.Random, lean: float, size: int) -> np.ndarray:
    """_draw_wobbly と同じ種類の小さな変形を、切り出した字にかける。"""
    if amount <= 0 and lean == 0:
        return t
    p = int(size * 0.2)
    t = cv2.copyMakeBorder(t, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    h, w = t.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.gauss(0, 3.0 * amount), 1 + rng.gauss(0, 0.05 * amount))
    M[0, 1] += rng.gauss(0, 0.06 * amount) - lean
    M[1, 0] += rng.gauss(0, 0.04 * amount)
    t = cv2.warpAffine(t, M, (w, h))
    if amount > 0.2:
        g = np.random.default_rng(rng.randrange(1 << 30))
        dx = cv2.GaussianBlur(g.normal(0, 1, (h, w)).astype(np.float32), (0, 0), size * 0.15)
        dy = cv2.GaussianBlur(g.normal(0, 1, (h, w)).astype(np.float32), (0, 0), size * 0.15)
        k = size * 0.04 * amount / max(1e-6, float(np.abs(dx).max() + np.abs(dy).max()) / 2)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        t = cv2.remap(t, gx + dx * k, gy + dy * k, cv2.INTER_LINEAR)
    return t


def draw_vertical(text: str, font_path: str, font_index: int, rng: random.Random, handwriting: bool,
                  meta: dict | None = None, size: int | None = None, hurry: float | None = None,
                  fatigue: float | None = None, plain: bool = False, state_cap: float = 1.0) -> np.ndarray:
    """縦書きの 1 行（上から下）を描き、**左に 90° 倒した横長の画像**で返す（左 = 行の頭）。
    認識器は横書きと同じように左から読めばよい。推論でも縦書きの欄は同じ向きに倒して読む（Reader.read(vertical=True)）。

    縦書きらしさ: 「ー」やかっこは 90° 回す、小さい「ゃ」・句読点は右上に寄せる、2 桁の数字はときどき縦中横。
    状態の因子は横書きと同じ: 急ぎ = 字間が詰まり、行が横に流れ、字が傾く / 疲れ = 後ろほど揺れる / 詰め書き = 後ろほど小さく、縦に潰れる。"""
    meta = {} if meta is None else meta
    size = rng.randint(28, 52) if size is None else size
    hurry_fix, fatigue_fix = hurry, fatigue
    hurry = fatigue = 0.0
    lead = rng.uniform(-0.02, 0.15) * size if rng.random() < 0.5 else 0.0
    if handwriting:
        lead = rng.uniform(-0.05, 0.25) * size
        hurry = (rng.random() ** 2 if rng.random() < 0.5 else 0.0) * state_cap
        fatigue = (rng.random() ** 2 if rng.random() < 0.4 else 0.0) * state_cap
    hurry = hurry if hurry_fix is None else hurry_fix
    fatigue = fatigue if fatigue_fix is None else fatigue_fix
    if handwriting:
        if plain:
            lead = 0.08 * size
        lead -= 0.12 * hurry * size
    toks = _vtokens(text, rng)
    n = len(toks)
    scales, sys_ = [1.0] * n, [1.0] * n
    if handwriting and n >= 4 and rng.random() < 0.35 and not plain:
        k = rng.randint(n // 3, n - 2)
        end = rng.uniform(0.4, 0.8)
        squeeze = rng.choice([0.0, rng.uniform(0.3, 1.2)])
        for i in range(k, n):
            scales[i] = 1.0 - (1.0 - end) * (i - k + 1) / (n - k)
            sys_[i] = max(0.35, 1.0 - (1.0 - scales[i]) * squeeze)
    wobble = 0.0 if plain or not handwriting else rng.uniform(0.0, 0.25)
    meta.update(font_px=size, tracking=round(lead / size, 3), cram=round(min(a * b for a, b in zip(scales, sys_)), 3),
                handwriting=handwriting, hurry=round(hurry, 3), fatigue=round(fatigue, 3), wobble=round(wobble, 3),
                vertical=True)
    drift = (rng.uniform(-0.04, 0.04) + 0.06 * hurry) if handwriting else 0.0  # 急ぐと行が横に流れる
    cells = []  # (字の画像, 横のずれ, 縦の位置, マスの高さ)
    y = size * 0.3
    for i, (tok, sc, sy) in enumerate(zip(toks, scales, sys_)):
        s = max(8, int(size * sc))
        if tok.isspace():
            y += 0.5 * s * sy
            continue
        f = get_font(font_path, font_index, s)
        if len(tok) == 2:  # 縦中横: 2 桁を 1 マスに収める
            t = _tile(tok, f, sx=min(1.0, s * 0.95 / max(1, f.getlength(tok))))
        else:
            t = _tile(tok, f)
        if tok in VERT_ROTATE:
            t = np.ascontiguousarray(np.rot90(t, -1))
        if sy < 1.0:
            t = cv2.resize(t, (t.shape[1], max(2, int(t.shape[0] * sy))), interpolation=cv2.INTER_AREA)
        prog = i / max(1, n - 1)
        amt = wobble + 0.35 * hurry + 0.6 * fatigue * prog
        t = _warp(t, amt, rng, 0.12 * hurry, s)
        cell_h = s * sy
        dx = 0.0
        dy = (cell_h - t.shape[0]) / 2  # マスの中央に置く
        if tok in VERT_SMALL:
            dx, dy = 0.12 * s, dy - 0.12 * s
        if tok in VERT_PUNCT:
            dx, dy = 0.28 * s, -t.shape[0] * 0.2
            cell_h = 0.6 * s
        jit = 0.03 + 0.05 * fatigue * prog
        dx += drift * y + (rng.gauss(0, size * jit) if handwriting else 0.0)
        cells.append((t, dx, y + dy))
        y += cell_h + lead * sc
    if not cells:
        return np.full((H, 16), 255, np.uint8)
    Hc = int(y + size * 0.3) + 1
    span = max(abs(c[1]) for c in cells)
    Wc = int(size * 1.6 + 2 * span) + 1
    cx0 = Wc / 2
    canvas = np.zeros((max(Hc, 8), Wc), np.float32)
    for t, dx, yy in cells:
        X, Y = int(round(cx0 + dx - t.shape[1] / 2)), int(round(yy))
        x0, y0 = max(0, X), max(0, Y)
        x1, y1 = min(Wc, X + t.shape[1]), min(canvas.shape[0], Y + t.shape[0])
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], t[y0 - Y : y1 - Y, x0 - X : x1 - X])
    a = np.clip(canvas, 0, 255).astype(np.uint8)
    xs_ = np.where(a.max(axis=0) > 0)[0]
    left = max(0, xs_[0] - int(rng.uniform(0.05, 0.35) * size))
    right = min(a.shape[1], xs_[-1] + 1 + int(rng.uniform(0.05, 0.35) * size))
    a = 255 - a[:, left:right]  # 白地に黒、縦長
    return np.ascontiguousarray(np.rot90(a, 1))  # 左に 90° 倒す（上 → 左）


# ---------------------------------------------------------------- 撮影の劣化（スマホ・デジカメ）
# どれも白地黒字の float/uint8 画像を受け取り float で返す。能力表（gallery）と学習（degrade）で共通に使う。
def tilt(img: np.ndarray, deg: float, paper: float = 255.0) -> np.ndarray:
    """行ごと傾ける。切り出しは傾いた行を囲む長方形（行の検出器が返す枠と同じ）なので、画像は縦に広がる。"""
    if deg == 0:
        return img.astype(np.float32)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c) + 1, int(h * c + w * s_) + 1
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img.astype(np.float32), M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float(paper))


def shadow(img: np.ndarray, dark: float, rng: random.Random, soft: float | None = None) -> np.ndarray:
    """手やスマホの影: 境目のある暗い領域が行の一部にかかる。dark = 暗くなる割合（0〜1）。"""
    if dark <= 0:
        return img.astype(np.float32)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(-1.2, 1.2)  # 境目の向き（ほぼ縦〜斜め）
    x0 = rng.uniform(0.25, 0.75) * w
    d = (xx - x0) * np.cos(ang) + (yy - h / 2) * np.sin(ang)
    soft = (rng.uniform(0.05, 0.6) * h) if soft is None else soft
    m = 1.0 / (1.0 + np.exp(-d / max(soft, 0.5)))
    if rng.random() < 0.5:
        m = 1.0 - m
    return img.astype(np.float32) * (1.0 - dark * m)


def uneven_light(img: np.ndarray, dark: float, rng: random.Random) -> np.ndarray:
    """照明のむら: 行の端から端へ、なめらかに暗くなる（明るさが掛け算で落ちる）。dark = いちばん暗い所の落ち方。"""
    if dark <= 0:
        return img.astype(np.float32)
    h, w = img.shape
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    if rng.random() < 0.5:
        t = 1 - t
    if rng.random() < 0.3:  # 真ん中が明るく、両端が暗い（ライトの真下）
        t = np.abs(np.linspace(-1, 1, w, dtype=np.float32))[None, :]
    g = 1.0 - dark * t ** rng.uniform(0.7, 1.5)
    return img.astype(np.float32) * g


def motion_blur(img: np.ndarray, length: float, angle: float = 0.0) -> np.ndarray:
    """手ぶれ: 長さ length px の直線に沿ったぼけ。"""
    if length < 1:
        return img.astype(np.float32)
    k = int(np.ceil(length)) | 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0), (k, k))
    ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img.astype(np.float32), -1, ker, borderType=cv2.BORDER_REPLICATE)


CAMERA = True  # False にすると撮影の劣化（傾き・影・照明のむら・強いぼけ・手ぶれ）を使わない（比較用）


def degrade(a: np.ndarray, rng: random.Random, meta: dict | None = None) -> np.ndarray:
    """スキャン・FAX・スマホ撮影らしい劣化。a は白地黒字 uint8。meta を渡すと使ったパラメータを書き込む。"""
    meta = {} if meta is None else meta
    meta.update(stroke="normal", border=False, crossing=False, blur=0.0, downscale=1.0, noise=0.0,
                binarize=False, jpeg=100, rotate=0.0, shadow=0.0, light=0.0, motion=0.0)
    h, w = a.shape
    ink = 255 - a.astype(np.float32)
    # 線の太さ
    r = rng.random()
    if r < 0.15:
        ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
        meta["stroke"] = "thick"
    elif r < 0.25:
        ink = cv2.erode(ink, np.ones((2, 2), np.uint8))
        meta["stroke"] = "thin"
    # 罫線（帳票の枠・下線）
    if rng.random() < 0.3:
        meta["border"] = True
        t = rng.randint(1, 3)
        for _ in range(rng.randint(1, 2)):
            y = rng.choice([rng.randint(0, 3), h - rng.randint(1, 4)])
            ink[max(0, y) : y + t, :] = 255 * rng.uniform(0.5, 1)
    if rng.random() < 0.15:
        x = rng.choice([rng.randint(0, 4), w - rng.randint(1, 5)])
        ink[:, max(0, x) : x + rng.randint(1, 3)] = 255 * rng.uniform(0.5, 1)
    # はみ出し: 枠線が字の上を横切る（欄からはみ出して書いた字）
    if rng.random() < 0.12:
        meta["crossing"] = True
        x = rng.randint(w // 4, max(w // 4 + 1, w - 4))
        t = rng.randint(1, 3)
        ink[:, x : x + t] = np.maximum(ink[:, x : x + t], 255 * rng.uniform(0.6, 1))
    if rng.random() < 0.1:
        meta["crossing"] = True
        y = rng.randint(int(h * 0.6), max(int(h * 0.6) + 1, h - 2))
        t = rng.randint(1, 2)
        ink[y : y + t, :] = np.maximum(ink[y : y + t, :], 255 * rng.uniform(0.6, 1))
    # インクの濃さと紙の色
    paper = rng.uniform(185, 255)
    ink_dark = rng.uniform(0, min(110, paper - 90))  # 紙との濃淡差は最低 90
    img = paper - ink / 255.0 * (paper - ink_dark)
    meta["contrast"] = round(paper - ink_dark, 1)
    # 紙のむら
    if rng.random() < 0.4:
        gx = np.linspace(0, 1, w)[None, :] * rng.uniform(-40, 40)
        gy = np.linspace(0, 1, h)[:, None] * rng.uniform(-20, 20)
        img = img + gx + gy
    # わずかな回転・せん断
    if rng.random() < 0.5:
        ang = rng.gauss(0, 0.8)
        sh = rng.gauss(0, 0.08)
        meta["rotate"] = round(ang, 2)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 1] += sh
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 撮影: 字の大きさ（px）の目安。ぼけ・手ぶれの強さは「字の高さ 40px 換算」で決める（能力表と同じ物差し）
    px = h / 1.4
    k40 = px / 40.0
    cam = 1.0 if CAMERA else 0.0
    if rng.random() < 0.12 * cam:  # 行ごとの傾き（台紙に合わせた後の残り。能力表では 8° まで測る）
        ang = rng.choice([-1, 1]) * rng.uniform(1.0, 4.0)
        img = tilt(img, ang, paper=float(np.median(img)))
        meta["rotate"] = round(ang, 2)
    if rng.random() < 0.12 * cam:
        dark = rng.uniform(0.2, 0.75)
        img = shadow(img, dark, rng)
        meta["shadow"] = round(dark, 2)
    if rng.random() < 0.15 * cam:
        dark = rng.uniform(0.2, 0.8)
        img = uneven_light(img, dark, rng)
        meta["light"] = round(dark, 2)
    # ぼけ・手ぶれ・低解像度
    r = rng.random()
    if r < 0.30:
        sigma = rng.uniform(0.3, 1.5) * k40
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        meta["blur"] = round(sigma / k40, 2)
    elif r < 0.40 and CAMERA:  # 強いぼけ（ピンぼけ）
        sigma = rng.uniform(1.5, 4.0) * k40
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        meta["blur"] = round(sigma / k40, 2)
    elif r < 0.48 and CAMERA:
        ln = rng.uniform(2, 12) * k40
        img = motion_blur(img, ln, rng.uniform(0, 180))
        meta["motion"] = round(ln / k40, 1)
    if rng.random() < 0.3 and meta["blur"] < 1.5:  # 強いぼけに低解像度まで重ねると、人にも読めない
        h, w = img.shape
        f = rng.uniform(0.35, 0.8)
        meta["downscale"] = round(f, 2)
        small = cv2.resize(img, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # ノイズ
    if rng.random() < 0.5:
        sd = rng.uniform(2, 14)
        meta["noise"] = round(sd, 1)
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, sd, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 二値化（FAX・コピー）
    if rng.random() < 0.1 and not (meta["shadow"] or meta["light"]):  # FAX・コピーに影は付かない（付けると影が真っ黒になる）
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        meta["binarize"] = True
    # JPEG
    if rng.random() < 0.4:
        buf = io.BytesIO()
        q = rng.randint(30, 90)
        meta["jpeg"] = q
        Image.fromarray(img).save(buf, format="JPEG", quality=q)
        img = np.asarray(Image.open(buf).convert("L"))
    return img


def to_height(a: np.ndarray, max_w: int = 1280) -> np.ndarray:
    h, w = a.shape
    nw = max(8, int(round(w * H / h)))
    nw = min(nw, max_w)
    return cv2.resize(a, (nw, H), interpolation=cv2.INTER_AREA if h > H else cv2.INTER_LINEAR)
