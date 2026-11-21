"""KanjiVG の筆画データから「架空の書き手」の手書きフォントを大量に作る。

1 フォント = 1 人の書き手。書き手ごとの癖（WriterStyle）を乱数で決め、
さらに字ごとにも揺らす。癖のパラメータは JSON に残す（書き手適応の学習に使う）。

使い方:
    python -m mitate.hwfont --n 200 --out fonts/hw --workers 14

ライセンス: KanjiVG は CC BY-SA 3.0。生成フォントは社内の学習用にとどめ、外部に配布しない。
KanjiVG にない字（記号など）は OFL フォントの字形を書き手の傾き・縮尺で変形して補う。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import parse_path
from fontTools.ttLib import TTFont

from .charset import charset

ROOT = Path(__file__).resolve().parent.parent
KANJIVG_DIR = ROOT / "third_party" / "kanjivg" / "kanji"
FALLBACK_FONT = Path.home() / ".local/share/fonts/sekou/BIZUDGothic-Regular.ttf"
UPM = 1000
KVG = 109.0  # KanjiVG の viewBox


# ---------------------------------------------------------------- 筆画の読み込み
class _FlattenPen(BasePen):
    """SVG パスを折れ線（点列）のリストにする。"""

    def __init__(self):
        super().__init__(None)
        self.polys: list[list[tuple[float, float]]] = []

    def _moveTo(self, p):
        self.polys.append([p])

    def _lineTo(self, p):
        self.polys[-1].append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.polys[-1][-1]
        n = max(4, int(math.dist(p0, p3) / 2.0))
        for i in range(1, n + 1):
            t = i / n
            mt = 1 - t
            self.polys[-1].append((
                mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0],
                mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1],
            ))

    def _closePath(self):
        pass

    def _endPath(self):
        pass


_D_RE = re.compile(r'<path[^>]*?\sd="([^"]+)"')


def load_strokes(ch: str) -> list[np.ndarray] | None:
    """KanjiVG の筆画（筆順どおり）。座標は 0..109、y は下向き。"""
    p = KANJIVG_DIR / f"{ord(ch):05x}.svg"
    if not p.exists():
        return None
    strokes = []
    for d in _D_RE.findall(p.read_text(encoding="utf-8")):
        pen = _FlattenPen()
        parse_path(d, pen)
        for poly in pen.polys:
            if len(poly) >= 2:
                strokes.append(_resample(np.asarray(poly, dtype=np.float64), 1.5))
    return strokes or None


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return pts[:1].repeat(2, axis=0)
    n = max(2, int(total / step) + 1)
    s = np.linspace(0, total, n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


# ---------------------------------------------------------------- 書き手の癖
@dataclass
class WriterStyle:
    seed: int
    pen_width: float      # 線の太さ（KanjiVG 単位, 109 が全角）
    pressure: float       # 0=一定, 1=入り抜きで細くなる
    slant: float          # 右上がりなどの傾き（x への y のせん断）
    rise: float           # 横画の右上がり（y への x のせん断）
    aspect: float         # 横幅の倍率
    jitter: float         # 筆画の低周波の揺れ
    tremor: float         # 高周波の震え
    stroke_shift: float   # 筆画ごとのずれ
    shrink_ends: float    # 筆画の端を縮める/伸ばす（はね・はらいの省略/誇張）
    kana_scale: float     # かなの大きさ（漢字に対する倍率）
    size_var: float       # 字ごとの大きさのばらつき
    baseline_var: float   # 字ごとの上下のばらつき
    roundness: float      # 角を丸める度合い

    @staticmethod
    def sample(seed: int) -> "WriterStyle":
        r = random.Random(seed)
        neat = r.random()  # 0=雑, 1=丁寧
        return WriterStyle(
            seed=seed,
            pen_width=r.choice([r.uniform(2.2, 3.6), r.uniform(3.6, 5.5), r.uniform(5.5, 8.0)]),
            pressure=r.uniform(0.0, 0.8),
            slant=r.gauss(0, 0.08),
            rise=r.gauss(-0.04, 0.05),
            aspect=r.uniform(0.85, 1.08),
            jitter=(1 - neat) ** 1.5 * r.uniform(1.0, 5.0) + 0.4,
            tremor=r.uniform(0.0, 0.45),
            stroke_shift=(1 - neat) ** 1.5 * r.uniform(0.8, 4.5),
            shrink_ends=r.gauss(0, 0.08),
            kana_scale=r.uniform(0.72, 0.98),
            size_var=r.uniform(0.02, 0.10),
            baseline_var=r.uniform(0.0, 0.06),
            roundness=r.uniform(0.0, 1.0),
        )


    def careful(self) -> "WriterStyle":
        """同じ人が「丁寧に書いた」ときの癖。性格（傾き・字形・かなの大きさ・太さ）は残し、雑さだけ下げる。"""
        d = asdict(self)
        for k, f in (("jitter", 0.4), ("stroke_shift", 0.4), ("tremor", 0.5), ("shrink_ends", 0.5),
                     ("size_var", 0.6), ("baseline_var", 0.6), ("roundness", 0.7)):
            d[k] = d[k] * f
        return WriterStyle(**d)


def _smooth_noise(n: int, rng: np.random.Generator, scale: float, freq: float) -> np.ndarray:
    """長さ n の滑らかな 2D ノイズ（少数の正弦波の和）。"""
    t = np.linspace(0, 1, n)[:, None]
    out = np.zeros((n, 2))
    for _ in range(3):
        f = rng.uniform(0.3, 1.0) * freq
        ph = rng.uniform(0, 2 * np.pi, size=2)
        amp = rng.normal(0, scale, size=2)
        out += amp * np.sin(2 * np.pi * f * t + ph)
    return out


def _corner_round(pts: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0 or len(pts) < 5:
        return pts
    k = int(1 + amount * 4)
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    padded = np.pad(pts, ((k, k), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(padded[:, i], ker, mode="valid") for i in range(2)], axis=1)
    sm[0], sm[-1] = pts[0], pts[-1]
    return sm


def perturb_char(strokes: list[np.ndarray], st: WriterStyle, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """書き手の癖 + 字ごとの揺れをかけた筆画。戻り値は (点列, 太さ) のリスト。"""
    out = []
    for s in strokes:
        pts = s.copy()
        n = len(pts)
        # 端の伸び縮み（はね・はらいの省略/誇張）
        if n > 6:
            e = st.shrink_ends + rng.normal(0, 0.05)
            cut = int(abs(e) * n * 0.5)
            if e > 0 and cut > 0:
                pts = pts[: n - cut] if rng.random() < 0.7 else pts[cut:]
            elif e < 0 and cut > 0:
                d = pts[-1] - pts[-2]
                ext = pts[-1] + np.outer(np.arange(1, cut + 1), d)
                pts = np.concatenate([pts, ext])
            n = len(pts)
        pts = _corner_round(pts, st.roundness)
        pts = pts + rng.normal(0, st.stroke_shift, size=2)
        pts = pts + _smooth_noise(n, rng, st.jitter, freq=1.0)
        pts = pts + rng.normal(0, st.tremor, size=pts.shape)
        # 太さ: 入りと抜きで細くなる + 字ごとの筆圧の揺れ
        t = np.linspace(0, 1, n)
        taper = 1 - st.pressure * (np.clip(1 - t / 0.15, 0, 1) ** 2 * 0.5 + np.clip((t - 0.7) / 0.3, 0, 1) ** 1.5)
        w = st.pen_width * taper * rng.uniform(0.85, 1.15)
        out.append((pts, np.maximum(w, 0.8)))
    return out


# ---------------------------------------------------------------- 線 → 輪郭
def _circle(c, r, k=10):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], axis=1)


def _split_corners(pts: np.ndarray, deg: float = 55) -> list[np.ndarray]:
    if len(pts) < 3:
        return [pts]
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    idx = np.where(turn > np.radians(deg))[0] + 1
    parts, start = [], 0
    for i in idx:
        if i - start >= 1:
            parts.append(pts[start : i + 1])
            start = i
    parts.append(pts[start:])
    return [p for p in parts if len(p) >= 2]


def stroke_contours(pts: np.ndarray, w: np.ndarray) -> list[np.ndarray]:
    """可変幅の線を、塗りつぶし用の閉じた輪郭（複数）にする。角で分割し、つなぎ目は円で埋める。"""
    contours = []
    offset = 0
    for part in _split_corners(pts):
        m = len(part)
        pw = w[offset : offset + m] if offset + m <= len(w) else np.full(m, w[-1])
        offset += m - 1
        d = np.gradient(part, axis=0)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        norm[norm < 1e-9] = 1
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / norm
        left = part + nrm * (pw[:, None] / 2)
        right = part - nrm * (pw[:, None] / 2)
        poly = np.concatenate([left, right[::-1]])
        contours.append(poly)
        contours.append(_circle(part[0], pw[0] / 2))
        contours.append(_circle(part[-1], pw[-1] / 2))
    return contours


def _orient_ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area > 0 else poly[::-1]


# ---------------------------------------------------------------- フォントを組む
def _affine(st: WriterStyle, scale: float, dy: float) -> np.ndarray:
    """KanjiVG 座標(0..109, y下向き) → フォント座標(UPM, y上向き) の 2x3 行列。"""
    s = UPM / KVG * scale
    # 中心基準で 縮尺・傾き → 平行移動
    a = np.array([[st.aspect * s, -st.slant * s], [st.rise * s * -1, -s]])
    c = np.array([KVG / 2, KVG / 2])
    t = np.array([UPM / 2, UPM * 0.38 + dy]) - a @ c
    return np.concatenate([a, t[:, None]], axis=1)


def _is_kana(ch: str) -> bool:
    return "ぁ" <= ch <= "ヿ"


def build_font(index: int, out_dir: Path, chars: list[str], careful: bool = False,
               override: dict | None = None, tag: str | None = None) -> dict:
    """careful=True で同じ書き手の「丁寧版」（ファミリー名 HW SynthC NNNN、書き手番号は同じ）。
    override で癖の一部だけ差し替え、tag をつけると別名の試験用フォント（HW Test NNNN <tag>）になる。"""
    st = WriterStyle.sample(1000 + index)
    if careful:
        st = st.careful()
    if override:
        st = WriterStyle(**{**asdict(st), **override})
    prefix = "HWSynthC" if careful else "HWSynth"
    family = f"HW SynthC {index:04d}" if careful else f"HW Synth {index:04d}"
    if tag:
        prefix, family = "HWTest", f"HW Test {index:04d} {tag}"
    fallback = TTFont(FALLBACK_FONT)
    fb_glyphset = fallback.getGlyphSet()
    fb_cmap = fallback.getBestCmap()
    fb_upm = fallback["head"].unitsPerEm

    glyph_order = [".notdef", "space"]
    glyphs = {".notdef": TTGlyphPen(None).glyph(), "space": TTGlyphPen(None).glyph()}
    advances = {".notdef": (UPM, 0), "space": (int(UPM * 0.33), 0)}
    cmap = {0x20: "space"}
    n_kvg = n_fb = 0

    for ch in dict.fromkeys(chars):
        if ch == " ":
            continue
        cp = ord(ch)
        gname = f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:05X}"
        rng = np.random.default_rng((st.seed * 1_000_003 + cp) & 0xFFFFFFFF)
        narrow = cp < 0x80
        scale = (st.kana_scale if _is_kana(ch) else 1.0) * (1 + rng.normal(0, st.size_var))
        if narrow:
            scale *= 0.78
        dy = rng.normal(0, st.baseline_var) * UPM
        pen = TTGlyphPen(None)
        strokes = load_strokes(ch)
        if strokes is not None:
            m = _affine(st, scale, dy)
            for pts, w in perturb_char(strokes, st, rng):
                for poly in stroke_contours(pts, w):
                    q = poly @ m[:, :2].T + m[:, 2]
                    q = _orient_ccw(np.round(q).astype(int))
                    # TrueType は時計回りが外側
                    q = q[::-1]
                    pen.moveTo(tuple(int(v) for v in q[0]))
                    for p in q[1:]:
                        pen.lineTo(tuple(int(v) for v in p))
                    pen.closePath()
            n_kvg += 1
        elif cp in fb_cmap:
            rec = DecomposingRecordingPen(fb_glyphset)
            fb_glyphset[fb_cmap[cp]].draw(rec)
            k = UPM / fb_upm * scale
            tp = TransformPen(pen, (st.aspect * k, -st.rise * k, st.slant * k, k, 0, dy))
            rec.replay(tp)
            n_fb += 1
        else:
            continue
        g = pen.glyph()
        g.recalcBounds(None)
        if narrow:
            xmin = getattr(g, "xMin", 0)
            xmax = getattr(g, "xMax", UPM // 2)
            adv = int(max(xmax - min(xmin, 0), UPM * 0.3) + UPM * 0.12)
            # 左に寄せる（左サイドベアリング 6%）
            if hasattr(g, "coordinates") and g.numberOfContours > 0:
                shift = int(UPM * 0.06 - xmin)
                g.coordinates.translate((shift, 0))
                g.recalcBounds(None)
                adv = int(g.xMax + UPM * 0.06)
        else:
            # 字幅のばらつき（かなは狭め）。字の中心を新しい字幅の中心へ移す
            base = (st.kana_scale + 0.08) if _is_kana(ch) else 1.0
            adv = int(UPM * min(base, 1.0) * rng.uniform(0.92, 1.05))
            if hasattr(g, "coordinates") and g.numberOfContours > 0:
                g.coordinates.translate(((adv - UPM) // 2, 0))
                g.recalcBounds(None)
        glyph_order.append(gname)
        glyphs[gname] = g
        advances[gname] = (adv, getattr(g, "xMin", 0) or 0)
        cmap[cp] = gname

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(advances)
    fb.setupHorizontalHeader(ascent=880, descent=-120)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=880, sTypoDescender=-120, usWinAscent=1000, usWinDescent=200,
                ulCodePageRange1=(1 << 17))  # cp932
    fb.setupPost()
    stem = f"{prefix}{index:04d}{tag or ''}"
    path = out_dir / f"{stem}.ttf"
    tmp = out_dir / f".{stem}.ttf.tmp"
    fb.save(str(tmp))
    os.replace(tmp, path)  # 書き途中のファイルを読ませない
    meta = {"index": index, "family": family, "file": path.name, "careful": careful, "kanjivg_glyphs": n_kvg,
            "fallback_glyphs": n_fb, "style": asdict(st)}
    return meta


def _job(args):
    index, out_dir, chars, careful = args
    return build_font(index, Path(out_dir), chars, careful)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "fonts" / "hw"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chars", default=None, help="この文字だけ入れる（試作用）")
    ap.add_argument("--careful", action="store_true", help="丁寧版を作る（同じ書き手・雑さだけ下げる）")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chars = list(a.chars) if a.chars else charset()[1:]
    jobs = [(i, str(out), chars, a.careful) for i in range(a.start, a.start + a.n)]
    metas = []
    with ProcessPoolExecutor(a.workers) as ex:
        for m in ex.map(_job, jobs):
            metas.append(m)
            print(f"{m['file']}: kanjivg={m['kanjivg_glyphs']} fallback={m['fallback_glyphs']}", flush=True)
    meta_path = out / "writers.jsonl"
    with meta_path.open("a", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
