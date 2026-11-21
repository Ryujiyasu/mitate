"""会社名・住所の語彙（欄の知識で使う）。

方針は fields.py と同じ: **語彙は候補を出すだけ、決めるのは画像**。
語彙は 3 つの強さに分ける（強いほど、画像が少し合わなくても語彙の側に寄せてよい）:
  強  法人の種類（株式会社・有限会社…）と都道府県。数が限られていて、書き間違いもまずない（元号と同じ扱い）
  中  業種の言葉（不動産・製作所・病院・センター…）と市区町村名
  ―  固有の部分（会社の名前・町名・番地）。知識を持たない。画像のまま

出典:
  市区町村  日本郵便「郵便番号データ」（data/dict/ken_all/KEN_ALL.CSV）。
            「郵便番号データに限っては日本郵便株式会社は著作権を主張しません。自由に配布していただいて結構です。」
  そのほか  ここに手で並べた一般的な言葉
評価用の問題（humantest / formtest）の語彙リストとは別に作っている。ただし「不動産」「病院」のような
ありふれた業種の言葉は、どちらにも入っている。
"""
from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEN_ALL = ROOT / "data" / "dict" / "ken_all" / "KEN_ALL.CSV"

STRONG, MEDIUM = 8.0, 6.0  # 画像上の対数確率で、元の読みからどれだけ下がってよいか（fieldeval の罠で確かめた値）

CORP_TYPES = ("株式会社 有限会社 合同会社 合資会社 合名会社 一般社団法人 一般財団法人 公益社団法人 公益財団法人 "
              "社会福祉法人 医療法人 医療法人社団 学校法人 宗教法人 特定非営利活動法人 協同組合 事業協同組合 "
              "農業協同組合 生活協同組合").split()

INDUSTRY = ("商事 商店 商会 工業 産業 興業 建設 建工 工務店 製作所 製造 不動産 運輸 運送 物流 倉庫 交通 食品 "
            "電機 電気 電工 電設 薬局 薬品 歯科 医院 病院 診療所 クリニック 整骨院 接骨院 鍼灸院 保育園 幼稚園 "
            "学園 学院 学校 事務所 法律事務所 会計事務所 税理士事務所 設計 設計事務所 開発 企画 技研 精機 精工 "
            "化学 化成 印刷 自動車 モータース 石油 燃料 ガス 水道 設備 機械 機工 鉄工所 鉄工 金属 木材 林業 "
            "農園 農場 ファーム 牧場 水産 酒造 酒店 米穀 青果 精肉 鮮魚 菓子 製菓 製パン ベーカリー 珈琲 "
            "カフェ 食堂 旅館 ホテル 観光 旅行 保険 銀行 信用金庫 証券 管理 サービス センター システム "
            "システムズ デザイン ソリューションズ テクノロジー コンサルティング ホールディングス グループ "
            "本舗 本店 支店 営業所 工場 研究所 マンション ハイツ コーポ アパート レジデンス スポーツ 美容室 "
            "理容室 介護 福祉会 ケアセンター 商工会 組合 協会 会館 書店 文具 写真館 花店 生花 クリーニング "
            "リフォーム 塗装 板金 電子 通信 情報 放送 出版 広告 物産 貿易 商社 ストア マート").split()

ADDRESS_SUFFIX = "市 区 町 村 郡 丁目 番地 番 号".split()


@lru_cache(maxsize=1)
def municipalities() -> tuple[tuple[str, ...], dict[str, set[str]]]:
    """(都道府県 47, 都道府県 → その中の市区町村名・郡名・それらの語幹)。"""
    by_pref: dict[str, set[str]] = defaultdict(set)
    prefs: list[str] = []
    with KEN_ALL.open(encoding="cp932", newline="") as f:
        for row in csv.reader(f):
            pref, city = row[6], row[7]
            if pref not in by_pref:
                prefs.append(pref)
            # 「上川郡東神楽町」「札幌市中央区」「さいたま市浦和区」→ 郡・市・区・町・村ごとに切る
            unit = ""
            for ch in city:
                unit += ch
                if ch in "市区町村郡" and len(unit) >= 2:
                    by_pref[pref].add(unit)
                    by_pref[pref].add(unit[:-1])  # 語幹（「仙台」「北九州」）も 1 語として持つ
                    unit = ""
    return tuple(prefs), dict(by_pref)


@lru_cache(maxsize=None)
def words(kind: str) -> dict[str, float]:
    """欄の種類 → {言葉: 強さ}。"""
    if kind in ("company", "kata"):
        return {**{w: MEDIUM for w in INDUSTRY}, **{w: STRONG for w in CORP_TYPES}}
    if kind == "address":
        prefs, by_pref = municipalities()
        out = {w: MEDIUM for ws in by_pref.values() for w in ws if len(w) >= 2}
        out.update({w: STRONG for w in prefs})
        out.update({w: MEDIUM for w in ADDRESS_SUFFIX if len(w) >= 2})
        return out
    return {}


@lru_cache(maxsize=None)
def _index(kind: str):
    """1〜2 字が違う言葉を素早く引くための索引（違う位置を「?」にした形 → 言葉）。"""
    idx: dict[str, set[str]] = defaultdict(set)
    for w in words(kind):
        n = len(w)
        for k in (1, 2) if n >= 3 else (1,):
            for pos in itertools.combinations(range(n), k):
                key = "".join("?" if i in pos else c for i, c in enumerate(w))
                idx[key].add(w)
    return idx


def repairs(text: str, kind: str) -> list[tuple[int, int, str, float]]:
    """text の中で「語彙の言葉と 1 字（3 字以上なら 2 字まで）だけ違う」部分を探す。
    返り値: (開始, 終了, 置き換える言葉, 強さ)。言葉の長さは元の部分と同じ（置き換えだけ）。"""
    W = words(kind)
    if not W:
        return []
    idx = _index(kind)
    out = []
    t = text
    for n in range(2, min(len(t), 10) + 1):
        for i in range(0, len(t) - n + 1):
            seg = t[i : i + n]
            if seg in W:
                continue
            found = set()
            for k in (1, 2) if n >= 3 else (1,):
                for pos in itertools.combinations(range(n), k):
                    key = "".join("?" if j in pos else c for j, c in enumerate(seg))
                    found |= idx.get(key, set())
            for w in found:
                if w == seg or " " in seg:
                    continue
                # 市区町村の語幹（「仙台」「川島」）は、直後に 市・区・町・村・郡 が続くときだけ使う
                # （「伊奈川県」の「川県」を「川島」に変えるような直し方を防ぐ）
                if kind == "address" and w[-1] not in "市区町村郡都道府県" and w not in ADDRESS_SUFFIX:
                    if i + n >= len(t) or t[i + n] not in "市区町村郡":
                        continue
                out.append((i, i + n, w, W[w]))
    return out


def coverage(text: str, kind: str) -> int:
    """語彙で説明できる字数（重ならないように長い言葉から貪欲に）。多いほど「ありそうな」文字列。"""
    W = words(kind)
    t = text.replace(" ", "")
    used = [False] * len(t)
    for w in sorted((w for w in W if w in t), key=len, reverse=True):
        start = t.find(w)
        while start >= 0:
            if not any(used[start : start + len(w)]):
                for j in range(start, start + len(w)):
                    used[j] = True
            start = t.find(w, start + 1)
    return sum(used)


def address_issues(text: str) -> list[str]:
    """都道府県と市区町村の組み合わせが実在しないとき（記入ミスか、読み間違いか）。直さずに印だけ付ける。"""
    prefs, by_pref = municipalities()
    t = text.replace(" ", "")
    pref = next((p for p in prefs if t.startswith(p)), None)
    if not pref:
        return ["都道府県で始まっていない"]
    rest = t[len(pref):]
    units = [u for u in by_pref[pref] if u[-1] in "市区町村郡" and rest.startswith(u)]
    if not units:
        others = [p for p in prefs if p != pref and any(rest.startswith(u) for u in by_pref[p] if u[-1] in "市区町村郡")]
        return [f"{pref}に、この市区町村はない" + (f"（{others[0]}にはある）" if others else "")]
    return []
