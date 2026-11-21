"""学習データ（その場で無限に生成）と、固定の評価セット。"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from . import render, textgen
from .charset import encode, groups, normalize

STRIDE = 4  # 認識器の横方向の縮小率（既定。stride 2 のモデルでは SynthLines / collate に stride=2 を渡す）


# 守備範囲。狭いモデルは小さく速くでき、範囲外は「自信なし」として人に回せる
PROFILES = {
    # 活字だけ
    "print": dict(p_hw=0.0, max_jitter=None, state_cap=0.0),
    # 帳票: 丁寧〜普通の手書き（雑な書き手・強い急ぎ/疲れは含めない）
    "form": dict(p_hw=0.5, max_jitter=1.8, state_cap=0.5),
    # 全部: 雑な手書きまで
    "full": dict(p_hw=0.5, max_jitter=None, state_cap=1.0),
}


def writer_jitter(path: str) -> float:
    """フォント番号から書き手の雑さを再計算する（生成時と同じ乱数なので writers.jsonl を待たなくてよい）。"""
    from .hwfont import WriterStyle
    stem = Path(path).stem
    st = WriterStyle.sample(1000 + int(stem[-4:]))
    return (st.careful() if stem.startswith("HWSynthC") else st).jitter


class SynthLines(IterableDataset):
    def __init__(self, seed: int, p_handwriting: float = 0.5, heldout: bool = False, profile: str | None = None,
                 stride: int = STRIDE, p_vertical: float = 0.0):
        self.seed = seed
        self.heldout = heldout
        self.stride = stride
        self.p_vertical = p_vertical  # 縦書きの行（左に 90° 倒した形）を混ぜる割合
        prof = PROFILES[profile] if profile else dict(p_hw=p_handwriting, max_jitter=None, state_cap=1.0)
        self.p_hw = prof["p_hw"]
        self.max_jitter = prof["max_jitter"]
        self.state_cap = prof["state_cap"]

    def _fonts(self):
        printed = render.PRINT_HELDOUT if self.heldout else render.PRINT_TRAIN
        hw = [(str(p), i) for p, i in render.hw_fonts(self.heldout)]
        if self.max_jitter is not None:
            hw = [(p, i) for p, i in hw if writer_jitter(p) <= self.max_jitter]
        printed = [(str(p), i) for p, i in printed if Path(p).exists()]
        return printed, hw

    def generate(self, rng: random.Random, printed, hw, pool, meta: dict | None = None, text_fn=None):
        while True:
            use_hw = hw and rng.random() < self.p_hw
            path, idx = rng.choice(hw if use_hw else printed)
            cov = render.coverage(path, idx)
            text = render.fit_text((text_fn or textgen.line)(rng), cov, rng, pool)
            if not text:
                continue
            m = {} if meta is not None else None
            # 縦書きは、英字の多い行（メールアドレスなど）には使わない（縦書きの欄にはまず書かれない）
            vertical = (self.p_vertical > 0 and rng.random() < self.p_vertical
                        and sum(c.isascii() and c.isalpha() for c in text) <= 0.3 * len(text))
            draw = render.draw_vertical if vertical else render.draw_line
            a = draw(text, path, idx, rng, handwriting=use_hw, meta=m, state_cap=self.state_cap)
            if m is not None:
                m.setdefault("vertical", False)
            a = render.degrade(a, rng, meta=m)
            a = render.to_height(a)
            # " " が連続・先頭末尾にあるとラベルが曖昧になるのでまとめる
            label = " ".join(text.split())
            ids = encode(label)
            if a.shape[1] // self.stride < 2 * len(ids) + 1:
                continue
            writer = int(Path(path).stem[-4:]) if use_hw else -1
            if meta is not None:
                meta.clear(); meta.update(m, font=Path(path).stem, n_chars=len(ids))
            return a, ids, label, writer

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid + os.getpid())
        printed, hw = self._fonts()
        pool = groups()["all"]
        n = 0
        while True:
            n += 1
            if n % 500 == 0:  # 生成が進んだフォントを取り込む
                printed, hw = self._fonts()
            yield self.generate(rng, printed, hw, pool)


def make_fixed_set(n: int, seed: int, p_hw: float, heldout: bool, profile: str | None = None, text_fn=None,
                   stride: int = STRIDE, p_vertical: float = 0.0):
    ds = SynthLines(seed, p_hw, heldout, profile, stride=stride, p_vertical=p_vertical)
    if profile:
        ds.p_hw = p_hw  # 評価セットは活字/手書きを指定どおりに分ける
    printed, hw = ds._fonts()
    rng = random.Random(seed)
    pool = groups()["all"]
    return [ds.generate(rng, printed, hw, pool, text_fn=text_fn) for _ in range(n)]


class DiskLines(Dataset):
    """Oxi で描いた帳票から切り出した行（data/oxi_lines/*.png + labels.jsonl）。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.items = [json.loads(l) for l in (self.root / "labels.jsonl").open(encoding="utf-8")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        a = np.asarray(Image.open(self.root / it["file"]).convert("L"))
        a = render.to_height(a)
        label = " ".join(normalize(it["text"]).split())
        return a, encode(label), label, it.get("writer", -1)


def collate(batch, stride: int = STRIDE):
    W = max(a.shape[1] for a, *_ in batch)
    W = (W + stride - 1) // stride * stride
    x = torch.zeros(len(batch), 1, render.H, W)
    widths, targets, lengths, labels, writers = [], [], [], [], []
    for i, (a, ids, label, writer) in enumerate(batch):
        t = 1.0 - torch.from_numpy(a.astype(np.float32)) / 255.0  # 地=0, 字=1
        x[i, 0, :, : a.shape[1]] = t
        widths.append(a.shape[1] // stride)
        targets.extend(ids)
        lengths.append(len(ids))
        labels.append(label)
        writers.append(writer)
    return {
        "x": x,
        "in_lengths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "tgt_lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": labels,
        "writers": torch.tensor(writers, dtype=torch.long),
    }
