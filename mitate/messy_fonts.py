"""読み取りテスト用: 同じ書き手の性格のまま、雑さだけを 5 段階に変えたフォント（HW Test NNNN L1〜L5）。"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .charset import charset
from .hwfont import build_font

ROOT = Path(__file__).resolve().parent.parent
LEVELS = {  # L1 丁寧 → L5 かなり雑
    "L1": dict(jitter=0.4, stroke_shift=0.2, tremor=0.05),
    "L2": dict(jitter=1.2, stroke_shift=0.9, tremor=0.12),
    "L3": dict(jitter=2.5, stroke_shift=1.8, tremor=0.2),
    "L4": dict(jitter=4.0, stroke_shift=3.0, tremor=0.3),
    "L5": dict(jitter=6.0, stroke_shift=4.5, tremor=0.4),
}
WRITERS = [910, 911]


def _job(a):
    idx, tag = a
    return build_font(idx, ROOT / "fonts" / "hw", charset()[1:], override=LEVELS[tag], tag=tag)


if __name__ == "__main__":
    jobs = [(w, t) for w in WRITERS for t in LEVELS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        for m in ex.map(_job, jobs):
            print(m["family"], m["style"]["jitter"], flush=True)
