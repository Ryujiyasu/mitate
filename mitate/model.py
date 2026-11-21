"""軽量な行認識器: CNN + BiLSTM + CTC。

- 入力: (B, 1, 48, W)、出力: (W/stride, B, 文字数) の対数確率（stride = 4 または 2）
- stride 2: 横方向の判断を 2px ごとにする。詰め書きの細い字（「1」など）や、同じ字の連続（「11」）は、
  CTC では間に空白を 1 つ挟まないと出せないので、横の細かさが要る
- CTC は「見えた順に文字を出す」だけで、文脈から文字を作り出さない（勝手な補完をしない）
- 目標: CPU / ブラウザ（ONNX, WebGPU/WASM）で動く大きさ（〜10M パラメータ）
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, p=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class Block(nn.Module):
    """残差つきの畳み込みブロック（depthwise で軽く）。"""

    def __init__(self, c):
        super().__init__()
        self.dw = conv_bn(c, c, 3, 1, 1, groups=c)
        self.pw = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.pw(self.dw(x)))


class CRNN(nn.Module):
    def __init__(self, num_classes: int, scale: float = 1.0, hidden: int | None = None, stride: int = 4):
        """scale で幅を変える（0.5 = 極小、0.75 = 小、1.0 = 標準）。stride は横の縮小率（4 か 2）。
        stride を変えても重みの形は同じなので、stride 4 のモデルの重みをそのまま読み込める。"""
        super().__init__()
        assert stride in (2, 4)
        self.stride = stride
        c1, c2, c3, c4 = (max(8, int(round(c * scale / 8)) * 8) for c in (48, 96, 192, 320))
        hidden = hidden or max(64, int(round(256 * scale / 32)) * 32)
        self.cnn = nn.Sequential(
            conv_bn(1, c1, 3, 1, 1),
            nn.MaxPool2d(2, 2),                       # 24 x W/2
            conv_bn(c1, c2), Block(c2),
            nn.MaxPool2d(2, 2) if stride == 4 else nn.MaxPool2d((2, 1), (2, 1)),  # 12 x W/stride
            conv_bn(c2, c3), Block(c3), Block(c3),
            nn.MaxPool2d((2, 1), (2, 1)),             # 6 x W/4
            conv_bn(c3, c4), Block(c4), Block(c4),
            nn.MaxPool2d((2, 1), (2, 1)),             # 3 x W/4
            conv_bn(c4, c4, (3, 1), 1, (0, 0)),       # 1 x W/4
        )
        # 横方向の文脈（字の幅程度）を広げる 1D 畳み込み
        self.ctx = nn.Sequential(
            nn.Conv1d(c4, c4, 5, padding=2, groups=c4), nn.BatchNorm1d(c4), nn.SiLU(inplace=True),
            nn.Conv1d(c4, c4, 1), nn.BatchNorm1d(c4), nn.SiLU(inplace=True),
        )
        self.rnn = nn.LSTM(c4, hidden, num_layers=2, bidirectional=True, batch_first=False, dropout=0.1)
        # 出力層が全体の半分を占めないよう、256 次元に絞ってから 7,000 字超へ出す
        neck = max(128, int(round(256 * scale / 32)) * 32)
        self.head = nn.Sequential(nn.Linear(hidden * 2, neck), nn.SiLU(inplace=True), nn.Linear(neck, num_classes))

    def features(self, x):
        f = self.cnn(x).squeeze(2)            # (B, C, T)
        f = f + self.ctx(f)
        f = f.permute(2, 0, 1)                # (T, B, C)
        f, _ = self.rnn(f)
        return f                              # (T, B, 2H)

    def forward(self, x):
        return self.head(self.features(x))    # (T, B, K) ロジット
