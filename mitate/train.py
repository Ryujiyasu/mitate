"""認識器の学習。

    python -m mitate.train --steps 60000 --out runs/crnn_v1

評価は 4 種類（すべて学習に使っていないテキスト）:
  print_seen   学習に使った活字フォント
  print_unseen 学習に使っていない活字フォント（書体の汎化）
  hw_seen      学習に使った書き手
  hw_unseen    学習に使っていない書き手（書き手の汎化）
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from .charset import BLANK, charset, decode_ctc
from .data import DiskLines, SynthLines, collate, make_fixed_set
from . import textgen
from .model import CRNN
from .render import hw_fonts


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, samples, device, bs=128):
    model.eval()
    errs = chars = exact = 0
    examples = []
    for i in range(0, len(samples), bs):
        b = collate(samples[i : i + bs], stride=model.stride)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["x"].to(device))
        pred = logits.argmax(-1).cpu()  # (T, B)
        for j, ref in enumerate(b["labels"]):
            hyp = decode_ctc(pred[: b["in_lengths"][j], j])
            e = edit_distance(hyp, ref)
            errs += e
            chars += len(ref)
            exact += e == 0
            if len(examples) < 6 and e:
                examples.append((ref, hyp))
    model.train()
    return {"cer": errs / max(chars, 1), "line_acc": exact / len(samples), "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/crnn_v1")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--p_hw", type=float, default=0.5)
    ap.add_argument("--profile", default=None, choices=["print", "form", "full"], help="守備範囲（data.PROFILES）")
    ap.add_argument("--scale", type=float, default=1.0, help="モデルの大きさ（0.5 / 0.75 / 1.0）")
    ap.add_argument("--oxi_lines", default="data/oxi_lines", help="Oxi 帳票から切り出した行（あれば混ぜる）")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init", default=None, help="重みだけ引き継ぐ（追加学習。最適化と学習率は新しく始める）")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=4, choices=[2, 4], help="横の縮小率。2 = 詰め書き・同じ字の連続に強い")
    ap.add_argument("--p_vert", type=float, default=0.0, help="縦書きの行を混ぜる割合")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.backends.cudnn.benchmark = False  # 幅が毎回変わるので

    print("評価セットを作成中…", flush=True)
    evalsets = {
        "print_seen": make_fixed_set(1000, 101, 0.0, heldout=False),
        "print_unseen": make_fixed_set(1000, 102, 0.0, heldout=True),
        "hw_seen": make_fixed_set(1000, 103, 1.0, heldout=False),
        # 帳票の頻出語（令和・年月日・円・様…）。ここはほぼ 100% でないといけない
        "form_words": make_fixed_set(1000, 105, 0.5, heldout=False, text_fn=textgen.form_phrase),
    }
    if a.p_vert > 0:
        # 縦書き（学習に使っていない書体・書き手）
        evalsets["vert_print_unseen"] = make_fixed_set(500, 106, 0.0, heldout=True, p_vertical=1.0)
        evalsets["vert_hw_unseen"] = make_fixed_set(500, 107, 1.0, heldout=True, p_vertical=1.0)

    synth = SynthLines(seed=int(time.time()), p_handwriting=a.p_hw, profile=a.profile, stride=a.stride, p_vertical=a.p_vert)
    coll = functools.partial(collate, stride=a.stride)
    loader = DataLoader(synth, batch_size=a.bs, num_workers=a.workers, collate_fn=coll,
                        prefetch_factor=4, persistent_workers=True)
    oxi_iter = None
    if Path(a.oxi_lines, "labels.jsonl").exists():
        def _cycle():
            while True:  # 生成中に増えた行も、周回ごとに読み直して使う
                oxi = DiskLines(Path(a.oxi_lines))
                print(f"Oxi 帳票の行: {len(oxi)}", flush=True)
                yield from DataLoader(oxi, batch_size=a.bs // 4, shuffle=True, num_workers=2,
                                      collate_fn=coll, drop_last=True)
        oxi_iter = _cycle()

    K = len(charset())
    model = CRNN(K, a.scale, stride=a.stride).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"parameters: {nparams / 1e6:.2f}M, classes: {K}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = a.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step = 0
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device)["model"])
        print(f"重みを引き継ぎ: {a.init}", flush=True)
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]

    log = (out / "log.jsonl").open("a")
    t0, run_loss = time.time(), 0.0
    it = iter(loader)
    while step < a.steps:
        b = next(it)
        if oxi_iter is not None and step % 4 == 0:
            b = _merge(b, next(oxi_iter))
        x = b["x"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        lp = F.log_softmax(logits.float(), dim=-1)
        loss = F.ctc_loss(lp, b["targets"].to(device), b["in_lengths"].to(device), b["tgt_lengths"].to(device),
                          blank=BLANK, reduction="mean", zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        run_loss += loss.item()
        if step % 100 == 0:
            dt = time.time() - t0
            print(f"step {step} loss {run_loss / 100:.3f} lr {sched.get_last_lr()[0]:.2e} {100 * a.bs / dt:.0f} lines/s", flush=True)
            log.write(json.dumps({"step": step, "loss": run_loss / 100}) + "\n"); log.flush()
            run_loss, t0 = 0.0, time.time()
        if step % a.eval_every == 0 or step == a.steps:
            if "hw_unseen" not in evalsets and len(hw_fonts(heldout=True)) >= 10:
                evalsets["hw_unseen"] = make_fixed_set(1000, 104, 1.0, heldout=True)
                print(f"未知の書き手の評価セットを作成（{len(hw_fonts(heldout=True))} 人）", flush=True)
            res = {k: evaluate(model, v, device) for k, v in evalsets.items()}
            msg = " | ".join(f"{k} CER {r['cer'] * 100:.2f}% line {r['line_acc'] * 100:.1f}%" for k, r in res.items())
            print(f"[eval {step}] {msg}", flush=True)
            for k, r in res.items():
                for ref, hyp in r["examples"][:2]:
                    print(f"   {k}: 正解「{ref}」 → 出力「{hyp}」", flush=True)
            log.write(json.dumps({"step": step, "eval": {k: {"cer": r["cer"], "line_acc": r["line_acc"]} for k, r in res.items()}}) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "classes": K, "scale": a.scale, "profile": a.profile,
                        "stride": a.stride, "p_vert": a.p_vert}, out / "last.pt")


def _merge(a, b):
    """合成行のバッチと Oxi 行のバッチを 1 つにする。"""
    W = max(a["x"].shape[-1], b["x"].shape[-1])
    pad = lambda t: F.pad(t, (0, W - t.shape[-1]))
    return {
        "x": torch.cat([pad(a["x"]), pad(b["x"])]),
        "in_lengths": torch.cat([a["in_lengths"], b["in_lengths"]]),
        "targets": torch.cat([a["targets"], b["targets"]]),
        "tgt_lengths": torch.cat([a["tgt_lengths"], b["tgt_lengths"]]),
        "labels": a["labels"] + b["labels"],
        "writers": torch.cat([a["writers"], b["writers"]]),
    }


if __name__ == "__main__":
    main()
