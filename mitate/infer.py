"""学習済みの認識器で行画像を読む。

    python -m mitate.infer runs/crnn_v1/last.pt 画像1.png 画像2.png ...

文字ごとの確信度も返す（書き手適応で「自信のある字」を選ぶのに使う）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import render
from .charset import BLANK, charset
from .model import CRNN


@dataclass
class Reading:
    text: str
    char_conf: list[float]  # 文字ごとの確信度（その文字を出したフレームの最大確率）

    @property
    def conf(self) -> float:
        return min(self.char_conf) if self.char_conf else 1.0


class Reader:
    def __init__(self, ckpt: str | Path | None = None, device: str | None = None):
        """ckpt を省くと、配布している既定のモデル（mitate.weights）を使う。"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if ckpt is None:
            from .weights import fetch
            ckpt = fetch()
        try:
            ck = torch.load(ckpt, map_location=self.device, weights_only=True)
        except Exception:  # 学習中のチェックポイント（最適化の状態つき）
            ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model = CRNN(ck["classes"], ck.get("scale", 1.0), stride=ck.get("stride", 4)).to(self.device).eval()
        self.name = Path(ckpt).parent.name
        self.model.load_state_dict(ck["model"])
        self.cs = charset()

    def mask(self, allowed: str | None):
        """欄の種類で使える文字だけを残すためのマスク（None なら全部）。"""
        if allowed is None:
            return None
        from .charset import char_to_index
        m = torch.full((len(self.cs),), float("-inf"))
        m[BLANK] = 0.0
        idx = char_to_index()
        for c in allowed:
            if c in idx:
                m[idx[c]] = 0.0
        return m

    @torch.no_grad()
    def read(self, images: list[np.ndarray], allowed: str | None = None, vertical: bool = False) -> list[Reading]:
        """images: 白地に黒字の uint8 グレースケール（高さは任意）。
        allowed を渡すと、その文字だけから選ぶ（電話番号の欄なら数字とハイフンだけ、など）。
        vertical=True: 縦書きの欄（上から下へ読む）。左に 90° 倒して、横書きの行として読む（学習も同じ向き）。"""
        if vertical:
            images = [np.ascontiguousarray(np.rot90(a, 1)) for a in images]
        STRIDE = self.model.stride
        arrs = [render.to_height(a) for a in images]
        W = max(a.shape[1] for a in arrs)
        W = (W + STRIDE - 1) // STRIDE * STRIDE
        x = torch.zeros(len(arrs), 1, render.H, W)
        for i, a in enumerate(arrs):
            x[i, 0, :, : a.shape[1]] = 1.0 - torch.from_numpy(a.astype(np.float32)) / 255.0
        logits = self.model(x.to(self.device)).float().cpu()  # (T, B, K)
        m = self.mask(allowed)
        if m is not None:
            logits = logits + m
        prob = logits.softmax(-1)
        out = []
        for i, a in enumerate(arrs):
            T = a.shape[1] // STRIDE
            p = prob[:T, i]
            best = p.argmax(-1)
            text, confs, prev = [], [], BLANK
            for t in range(T):
                k = int(best[t])
                if k != BLANK and k != prev:
                    text.append(self.cs[k])
                    confs.append(float(p[t, k]))
                elif k != BLANK and k == prev and confs:
                    confs[-1] = max(confs[-1], float(p[t, k]))
                prev = k
            out.append(Reading("".join(text), confs))
        return out


class Cascade:
    """軽いモデルから順に読み、自信がない行だけ次のモデルへ回す。最後まで自信がなければ人の確認へ。

        Cascade([(Reader("runs/form_s0.5/last.pt"), 0.95), (Reader("runs/full_s1.0/last.pt"), 0.9)])
    """

    def __init__(self, stages: list[tuple["Reader", float]]):
        self.stages = stages

    def read(self, images: list[np.ndarray]) -> list[dict]:
        results: list[dict | None] = [None] * len(images)
        todo = list(range(len(images)))
        for si, (reader, thr) in enumerate(self.stages):
            if not todo:
                break
            outs = reader.read([images[i] for i in todo])
            last = si == len(self.stages) - 1
            nxt = []
            for i, r in zip(todo, outs):
                if r.conf >= thr or last:
                    results[i] = {"text": r.text, "conf": r.conf, "model": reader.name,
                                  "needs_review": r.conf < thr, "char_conf": r.char_conf}
                else:
                    nxt.append(i)
            todo = nxt
        return results


def main():
    reader = Reader(sys.argv[1])
    paths = sys.argv[2:]
    imgs = [np.asarray(Image.open(p).convert("L")) for p in paths]
    for p, r in zip(paths, reader.read(imgs)):
        print(f"{Path(p).name}\t{r.text}\t(最低確信度 {r.conf:.2f})")


if __name__ == "__main__":
    main()
