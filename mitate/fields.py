"""欄の種類を使った確認。方針: **辞書は候補を出すだけ、決めるのは画像**。

oxi-forms の様式は欄の種類（{{氏名}} など）を持っているので、どの欄が名前・日付・郵便番号かが分かる。

    from mitate.fields import check_field
    result = check_field(reader, image, "name")
    # result = {"text": ..., "ok": bool, "issues": [...], "suggestions": [(候補, 画像上の対数確率), ...]}

流れ（名前の例）:
  1. OCR が「山田太昭」と読む
  2. 辞書では「太昭」は名前としてまずない → 問題として記録
  3. 自信のない字について、画像上の次点の字を組み合わせ、辞書にある名前を候補にする
  4. 各候補が「この画像である確率」を CTC でそのまま計算して比べる
  5. 画像の上でも候補が十分ありそうなら「◯◯では？」と提案する。上書きはしない（人が決める）
     画像がはっきり元の読みを支持するなら、珍しい名前でもそのまま通す
"""
from __future__ import annotations

import itertools
import re
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

from . import render, textgen
from .charset import BLANK, charset, encode
from .infer import Reader

ACCEPT_CONF = 0.9  # 画像だけで受け付ける確信度（boundary.py の結果で決め直す）

# 知識の強さは欄ごとに変える。元号は 4 つしかないので強く、名前は弱く
# （珍しい名前や 髙/高・﨑/崎 のような異体字を「よくある字」に直さないため）。
# 候補は「画像上の対数確率が元の読みから MARGIN 以内」のものだけ出す
MARGIN = {"name": 3.0, "date": 8.0}

# 仮の辞書（textgen の名字・名前のリスト）。本番では氏名辞書（ライセンス確認のうえ）に差し替える
SURNAMES = set(textgen.SURNAMES)
GIVEN = set(textgen.GIVEN)
ERAS = {"令和": (2019, 5, 1), "平成": (1989, 1, 8), "昭和": (1926, 12, 25), "大正": (1912, 7, 30)}


# ---------------------------------------------------------------- 画像上の確率
# 欄の種類ごとに使える文字（値は推測しない。文字の種類を絞るだけ）
DIGITS = "0123456789"
ALLOWED = {
    "phone": DIGITS + "-()",
    "postal": DIGITS + "-〒 ",
    "money": DIGITS + ",.-￥円金税込抜（）() ",
    "number": DIGITS + ".,-",
    "digits": DIGITS,
    "date": DIGITS + "年月日元令和平成昭大正/.- RHST",
}


@torch.no_grad()
def _log_probs(reader: Reader, image: np.ndarray, allowed: str | None = None):
    a = render.to_height(image)
    STRIDE = reader.model.stride
    W = (a.shape[1] + STRIDE - 1) // STRIDE * STRIDE
    x = torch.zeros(1, 1, render.H, W)
    x[0, 0, :, : a.shape[1]] = 1.0 - torch.from_numpy(a.astype(np.float32)) / 255.0
    logits = reader.model(x.to(reader.device)).float()
    m = reader.mask(allowed)
    if m is not None:
        logits = logits + m.to(logits.device)
    lp = logits.log_softmax(-1)  # (T, 1, K)
    return lp[: a.shape[1] // STRIDE].cpu()


def score(reader: Reader, image: np.ndarray, candidates: list[str], allowed: str | None = None) -> list[tuple[str, float]]:
    """各候補について log P(候補 | 画像) を CTC で計算する。大きいほど画像に合っている。"""
    lp = _log_probs(reader, image, allowed)
    T = lp.shape[0]
    out = []
    for c in candidates:
        try:
            ids = encode(c)
        except KeyError:
            continue
        if not ids or T < len(ids):
            out.append((c, float("-inf")))
            continue
        nll = F.ctc_loss(lp, torch.tensor([ids]), torch.tensor([T]), torch.tensor([len(ids)]),
                         blank=BLANK, reduction="sum", zero_infinity=True)
        out.append((c, -float(nll)))
    return sorted(out, key=lambda x: -x[1])


def alternatives(reader: Reader, image: np.ndarray, k: int = 4, conf_below: float = 0.9, allowed: str | None = None):
    """greedy で読んだ各字について、画像上の上位 k 候補（自信のある字は 1 候補だけ）。"""
    lp = _log_probs(reader, image, allowed)[:, 0]
    p = lp.exp()
    best = p.argmax(-1)
    cs = charset()
    slots, prev = [], BLANK
    for t in range(lp.shape[0]):
        b = int(best[t])
        if b != BLANK and b != prev:
            top = torch.topk(p[t], k + 1)
            cands = [cs[int(i)] for v, i in zip(top.values, top.indices) if int(i) != BLANK][:k]
            slots.append(cands[:1] if float(p[t, b]) >= conf_below else cands)
        prev = b
    return slots


# ---------------------------------------------------------------- 欄ごとの確認
def _name_ok(text: str) -> bool:
    t = text.replace(" ", "").replace("　", "")
    for i in range(1, len(t)):
        if t[:i] in SURNAMES and t[i:] in GIVEN:
            return True
    return False


def _date_issues(text: str) -> list[str]:
    t = text.replace(" ", "")
    m = re.fullmatch(r"(令和|平成|昭和|大正)(元|\d{1,2})年(\d{1,2})月(\d{1,2})日", t)
    if m:
        era, y, mo, d = m.groups()
        y = 1 if y == "元" else int(y)
        start = ERAS[era]
        year = start[0] + y - 1
    else:
        m = re.fullmatch(r"(\d{4})[年/.\-](\d{1,2})[月/.\-](\d{1,2})日?", t)
        if not m:
            return ["日付の形になっていない"]
        year, mo, d = int(m.group(1)), m.group(2), m.group(3)
        era = None
    try:
        dt = date(year, int(mo), int(d))
    except ValueError:
        return [f"存在しない日付（{year}年{int(mo)}月{int(d)}日）"]
    if era and dt < date(*ERAS[era]):
        return [f"{era}はその日付には始まっていない"]
    return []


def _phone_issues(text: str) -> list[str]:
    digits = re.sub(r"\D", "", text)
    if not digits.startswith("0"):
        return ["0 で始まっていない"]
    if digits[:3] in ("070", "080", "090", "050") and len(digits) != 11:
        return ["携帯・IP 電話の桁数（11 桁）と合わない"]
    if digits[:3] not in ("070", "080", "090", "050") and len(digits) != 10:
        return ["固定電話の桁数（10 桁）と合わない"]
    return []


def _postal_issues(text: str) -> list[str]:
    t = text.replace("〒", "").replace(" ", "")
    return [] if re.fullmatch(r"\d{3}-?\d{4}", t) else ["郵便番号の形（NNN-NNNN）になっていない"]


def _money_issues(text: str) -> list[str]:
    # 円の金額で「469.000」は、桁区切りの「,」を「.」と書いた（または読んだ）可能性が高い。直さずに印を付ける
    return ["桁区切りが「.」になっている"] if re.search(r"\d\.\d{3}(?!\d)", text) else []


VALID = {
    "name": lambda t: _name_ok(t),
    "date": lambda t: not _date_issues(t),
    "phone": lambda t: not _phone_issues(t),
    "postal": lambda t: not _postal_issues(t),
}
ISSUES = {
    "name": lambda t: [] if _name_ok(t) else ["辞書にない名前"],
    "date": _date_issues,
    "phone": _phone_issues,
    "postal": _postal_issues,
    "money": _money_issues,
}


LEXICON_KINDS = ("company", "kata", "address")


def _check_lexicon(reader: Reader, image: np.ndarray, kind: str, r, max_cands: int = 400) -> dict:
    """会社名・住所: 語彙の言葉と 1〜2 字だけ違う部分を言葉に置き換えた候補を作り、画像で比べる。

    採用するのは「語彙でより多く説明できて、しかも画像上の確率が元の読みから強さ（lexicon.STRONG/MEDIUM）
    以内しか下がらない」候補だけ。会社の固有の名前や町名・番地には知識を持たないので、そこは画像のまま。"""
    from . import lexicon
    text = r.text
    reps = sorted(lexicon.repairs(text, kind), key=lambda x: -x[3])
    cands: dict[str, float] = {}
    for i, j, w, s in reps:
        cands.setdefault(text[:i] + w + text[j:], s)
    for (i1, j1, w1, s1), (i2, j2, w2, s2) in itertools.combinations(reps[:60], 2):
        if j1 <= i2 or j2 <= i1:  # 重ならない 2 か所を同時に直す（「有限法社…不勤産」）
            (a, b, wa), (c, d, wb) = sorted([(i1, j1, w1), (i2, j2, w2)])
            cands.setdefault(text[:a] + wa + text[b:c] + wb + text[d:], min(s1, s2))
        if len(cands) >= max_cands:
            break
    base_cov = lexicon.coverage(text, kind)
    cands = {c: s for c, s in cands.items() if lexicon.coverage(c, kind) > base_cov}
    scored = dict(score(reader, image, list(cands) + [text]))
    base = scored.get(text, 0.0)
    ok = sorted(((c, scored[c]) for c in cands if c in scored and scored[c] >= base - cands[c]), key=lambda x: -x[1])
    others = sorted(((c, sc) for c, sc in scored.items() if c != text), key=lambda x: -x[1])
    issues = lexicon.address_issues(ok[0][0] if ok else text) if kind == "address" else []
    if ok:
        return {"text": text, "conf": r.conf, "ok": False, "issues": issues, "notes": [],
                "suggestions": ok[:3], "answer": ok[0][0], "source": "推定（画像＋知識）"}
    return {"text": text, "conf": r.conf, "ok": False, "issues": issues, "notes": [],
            "suggestions": others[:2], "answer": text, "source": "画像（自信低め）"}


def check_field(reader: Reader, image: np.ndarray, kind: str, max_combos: int = 200) -> dict:
    if kind in LEXICON_KINDS:
        r = reader.read([image])[0]
        if r.conf >= ACCEPT_CONF:
            # しっかり読めた → 画像のまま。実在しない住所の組み合わせは、直さずに印だけ付ける
            from .lexicon import address_issues
            content = address_issues(r.text) if kind == "address" else []
            return {"text": r.text, "conf": r.conf, "ok": not content, "issues": content, "notes": [],
                    "suggestions": [], "answer": r.text,
                    "source": "画像（記入内容の確認が必要）" if content else "画像"}
        return _check_lexicon(reader, image, kind, r)
    allowed = ALLOWED.get(kind)
    r = reader.read([image], allowed=allowed)[0]
    text = r.text
    issues: list[str] = []
    suggestions: list[tuple[str, float]] = []
    if r.conf >= ACCEPT_CONF:
        # しっかり読めた → 画像の読みを採用。知識で直さない
        #   辞書にない名前 → 珍しい名前として通す
        #   ありえない内容（2月30日など）→ OCR の誤りではなく記入ミスの可能性。書かれたとおりに残して印を付ける
        notes = []
        if kind == "name" and not _name_ok(text):
            notes.append("辞書にない名前（画像ではっきり読めたのでそのまま採用）")
        content = {"date": _date_issues, "phone": _phone_issues, "postal": _postal_issues,
                   "money": _money_issues}.get(kind, lambda t: [])(text)
        return {"text": text, "conf": r.conf, "ok": not content, "issues": content, "notes": notes,
                "suggestions": [], "answer": text,
                "source": "画像（記入内容の確認が必要）" if content else "画像"}
    # 自信がない → 候補を作って画像で比べる。人と同じく「たぶんこう」は必ず出すが、推測だと印を付ける
    valid = VALID.get(kind, lambda t: True)
    issues = ISSUES.get(kind, lambda t: [])(text)
    slots = alternatives(reader, image, allowed=allowed)
    cands = {"".join(c) for c in itertools.islice(itertools.product(*slots), max_combos)}
    if kind == "date" and text[:2] in ERAS:
        cands |= {era + c[2:] for c in cands for era in ERAS}  # 元号は 4 つしかない
    cands = [c for c in cands if valid(c)]
    scored = score(reader, image, cands + [text], allowed) if cands else []
    base = dict(scored)[text] if scored else 0.0
    others = [(c, sc) for c, sc in scored if c != text]
    if valid(text) and not others or (valid(text) and others and dict(scored)[text] >= others[0][1]):
        source, answer = "画像（自信低め）", text
        suggestions = others[:2]
    elif others and others[0][1] > base - MARGIN.get(kind, 3.0):
        source, answer = "推定（画像＋知識）", others[0][0]
        suggestions = others[:3]
    elif others:
        source, answer = "推測（要確認）", others[0][0]
        suggestions = others[:3]
    else:
        source, answer = "推測（要確認）", text
    return {"text": text, "conf": r.conf, "ok": False, "issues": issues, "notes": [],
            "suggestions": suggestions, "answer": answer, "source": source}
