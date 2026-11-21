"""学習用の文字列を作る。

言語モデルに頼らず「形で読む」認識器にしたいので、自然文に偏らせない:
  - 帳票の記入欄らしい文字列（氏名・住所・日付・金額・電話番号…）
  - かな漢字が混ざった擬似日本語（文字の並びは無意味）
  - 文字集合からの一様サンプリング（めったに出ない字も覚えさせる）
"""
from __future__ import annotations

import random

from .charset import groups

_G = None


def G():
    global _G
    if _G is None:
        _G = groups()
    return _G


SURNAMES = ("佐藤 鈴木 高橋 田中 伊藤 渡辺 山本 中村 小林 加藤 吉田 山田 佐々木 山口 松本 井上 木村 林 斎藤 清水 "
            "山崎 森 池田 橋本 阿部 石川 山下 中島 石井 小川 前田 岡田 長谷川 藤田 後藤 近藤 村上 遠藤 青木 坂本 "
            "斉藤 福田 太田 西村 藤井 金子 岡本 藤原 中野 三浦 原田 中川 松田 竹内 小野 田村 中山 和田 石田 森田 "
            "上田 原 柴田 酒井 工藤 横山 宮崎 宮本 内田 高木 安藤 島田 谷口 大野 丸山 今井 高田 藤本 武田 村田 "
            "上野 杉山 増田 小山 大塚 平野 菅原 久保 松井 千葉 岩崎 桜井 木下 野口 松尾 菊地 野村 新井 渡部 髙橋 "
            "山﨑 齋藤 濱田 邉見 安田 宇野 鬼頭 楢崎 蕪木 嶋田 廣瀬 薮内").split()
GIVEN = ("太郎 一郎 健太 翔太 大輔 拓也 直樹 誠 浩二 和也 健一 隆 博 学 達也 雄一 亮 優斗 蓮 陽翔 湊 悠真 "
         "花子 陽子 恵子 美穂 由美 裕子 真由美 愛 さくら 結衣 陽菜 美咲 葵 凛 結菜 芽依 ひなた みゆき あやか "
         "けんじ ゆうき 智子 久美子 千尋 早紀 彩 紗希 瑞穂 雅之 正樹 秀樹 敏行 昌平 慎吾 稔 勲 猛 巌 毅").split()
PREFS = ("北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県 茨城県 栃木県 群馬県 埼玉県 千葉県 東京都 神奈川県 "
         "新潟県 富山県 石川県 福井県 山梨県 長野県 岐阜県 静岡県 愛知県 三重県 滋賀県 京都府 大阪府 兵庫県 "
         "奈良県 和歌山県 鳥取県 島根県 岡山県 広島県 山口県 徳島県 香川県 愛媛県 高知県 福岡県 佐賀県 長崎県 "
         "熊本県 大分県 宮崎県 鹿児島県 沖縄県").split()
CITY_SUFFIX = ["市", "区", "町", "村", "郡"]
TOWN_SUFFIX = ["町", "丁目", "通", "台", "が丘", "本町", "新田"]
COMPANY = ["株式会社", "有限会社", "合同会社", "㈱", "㈲", "社会福祉法人", "医療法人", "一般社団法人"]
ERA = [("令和", 1, 8), ("平成", 1, 31), ("昭和", 20, 64)]


def _k(rng, n1=1, n2=3, lvl2=0.15):
    g = G()
    return "".join(rng.choice(g["kanji2"] if rng.random() < lvl2 else g["kanji1"]) for _ in range(rng.randint(n1, n2)))


def _hira(rng, a=1, b=4):
    return "".join(rng.choice(G()["hira"]) for _ in range(rng.randint(a, b)))


def _kata(rng, a=2, b=6):
    return "".join(rng.choice(G()["kata"]) for _ in range(rng.randint(a, b)))


def _digits(rng, a, b):
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(a, b)))


def name(rng):
    s = rng.choice(SURNAMES) if rng.random() < 0.7 else _k(rng, 1, 2)
    gname = rng.choice(GIVEN) if rng.random() < 0.6 else _k(rng, 1, 2)
    return s + rng.choice(["", " ", "　"]) + gname


def kana_name(rng):
    return _kata(rng, 2, 5) + rng.choice([" ", "　", ""]) + _kata(rng, 2, 4)


def date(rng):
    r = rng.random()
    y, m, d = rng.randint(1940, 2030), rng.randint(1, 12), rng.randint(1, 28)
    if r < 0.35:
        era, a, b = rng.choice(ERA)
        yy = rng.randint(a, b)
        return f"{era}{'元' if yy == 1 and rng.random() < 0.3 else yy}年{m}月{d}日"
    if r < 0.6:
        return f"{y}年{m}月{d}日"
    if r < 0.85:
        sep = rng.choice(["/", ".", "-"])
        return f"{y}{sep}{m:02d}{sep}{d:02d}" if rng.random() < 0.5 else f"{y}{sep}{m}{sep}{d}"
    return f"{rng.choice(['R', 'H', 'S'])}{rng.randint(1, 60)}.{m}.{d}"


def address(rng):
    s = rng.choice(PREFS) + _k(rng, 1, 3) + rng.choice(CITY_SUFFIX) + _k(rng, 1, 3) + rng.choice(TOWN_SUFFIX)
    r = rng.random()
    if r < 0.5:
        s += f"{rng.randint(1, 9)}-{rng.randint(1, 30)}-{rng.randint(1, 20)}"
    elif r < 0.8:
        s += f"{rng.randint(1, 9)}丁目{rng.randint(1, 30)}番{rng.randint(1, 20)}号"
    else:
        s += f"{rng.randint(1, 999)}番地"
    if rng.random() < 0.3:
        s += rng.choice([" ", ""]) + _kata(rng, 3, 6) + rng.choice(["", "ハイツ", "マンション", "荘"]) + f"{rng.randint(101, 1205)}号室"
    return s


def phone(rng):
    r = rng.random()
    if r < 0.4:
        return f"0{rng.randint(3, 9)}-{_digits(rng, 4, 4)}-{_digits(rng, 4, 4)}"
    if r < 0.8:
        return f"0{rng.choice(['70', '80', '90'])}-{_digits(rng, 4, 4)}-{_digits(rng, 4, 4)}"
    return f"0{_digits(rng, 2, 3)}({_digits(rng, 2, 3)}){_digits(rng, 4, 4)}"


def postal(rng):
    return rng.choice(["〒", "", "〒 "]) + f"{_digits(rng, 3, 3)}-{_digits(rng, 4, 4)}"


def money(rng):
    v = rng.choice([rng.randint(1, 999), rng.randint(1000, 99999), rng.randint(100000, 99999999)])
    s = f"{v:,}" if rng.random() < 0.8 else str(v)
    return rng.choice([f"¥{s}", f"{s}円", f"￥{s}-", f"金{s}円", f"{s}", f"{s}円（税込）"])


def company(rng):
    c = rng.choice(COMPANY)
    body = rng.choice([_k(rng, 2, 4), _kata(rng, 3, 7), _k(rng, 1, 2) + _kata(rng, 2, 4)])
    return c + body if rng.random() < 0.6 else body + c


def email(rng):
    a = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789._") for _ in range(rng.randint(3, 12))).strip("._") or "info"
    d = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8)))
    return f"{a}@{d}.{rng.choice(['jp', 'co.jp', 'ne.jp', 'com', 'or.jp'])}"


def number(rng):
    r = rng.random()
    if r < 0.4:
        return str(rng.randint(0, 9999))
    if r < 0.7:
        return f"{rng.uniform(0, 1000):.{rng.randint(1, 2)}f}"
    return f"{rng.randint(1, 999)}{rng.choice(['個', '本', '枚', '件', 'kg', 'cm', '%', '人', '台', '箱'])}"


def pseudo_japanese(rng, a=4, b=22):
    """かな漢字混じりの擬似文（意味はない）。"""
    out = ""
    target = rng.randint(a, b)
    while len(out) < target:
        r = rng.random()
        if r < 0.4:
            out += _k(rng, 1, 3)
        elif r < 0.75:
            out += _hira(rng, 1, 4)
        elif r < 0.85:
            out += _kata(rng, 2, 6)
        elif r < 0.92:
            out += rng.choice(["、", "。", "・", "「", "」", "（", "）", "：", "ー", "々"])
        else:
            out += _digits(rng, 1, 4)
    return out[:target]


def free(rng):
    return pseudo_japanese(rng, 4, 20)


def uniform_chars(rng, a=3, b=20):
    allc = G()["all"]
    return "".join(rng.choice(allc) for _ in range(rng.randint(a, b))).strip() or "あ"


def alnum(rng):
    r = rng.random()
    if r < 0.3:
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(1, 4))) + "-" + _digits(rng, 2, 6)
    if r < 0.6:
        w = lambda: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 9)))
        return " ".join(w().capitalize() if rng.random() < 0.3 else w() for _ in range(rng.randint(1, 4)))
    return "".join(rng.choice(G()["upper"] + G()["lower"] + G()["digit"] + G()["ascii_sym"]) for _ in range(rng.randint(3, 16)))


def digits(rng):
    return _digits(rng, 4, 14)


_FIELD = {
    "name": name, "kana_name": kana_name, "date": date, "address": address, "phone": phone,
    "postal": postal, "money": money, "company": company, "email": email, "number": number,
    "free": free, "digits": digits,
}


def field(kind: str, rng: random.Random) -> str:
    return _FIELD[kind](rng)


# 帳票によく出る言葉。読み間違えると一番目立つので、学習でも評価でも別扱いにする
FORM_WORDS = ("令和 平成 昭和 年 月 日 年月日 生年月日 申請日 記入日 円 税込 税抜 合計 小計 消費税 様 御中 殿 "
              "株式会社 有限会社 合同会社 代表取締役 氏名 住所 電話番号 郵便番号 フリガナ ふりがな 性別 男 女 "
              "印 署名 捺印 申請者 届出人 代理人 本人 続柄 世帯主 備考 所属 部署 担当者 連絡先 メールアドレス "
              "金額 数量 単価 品名 摘要 請求書 領収書 見積書 申込書 申請書 届出書 契約書 有 無 済 未 "
              "上記 以下 以上 未満 第 号 番地 丁目 番 号室 都 道 府 県 市 区 町 村 郡 年度 期間 から まで").split()


def form_phrase(rng: random.Random) -> str:
    """頻出語を 1〜3 個、記入欄らしい値と組み合わせる。"""
    parts = [rng.choice(FORM_WORDS) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.5:
        parts.insert(rng.randint(0, len(parts)), field(rng.choice(["date", "money", "name", "number"]), rng))
    return rng.choice(["", " ", "　"]).join(parts)


def line(rng: random.Random) -> str:
    """認識器の学習用の 1 行。"""
    r = rng.random()
    if r < 0.15:
        return form_phrase(rng)
    r = rng.random()
    if r < 0.40:
        k = rng.choice(list(_FIELD))
        s = _FIELD[k](rng)
        if rng.random() < 0.3:  # ラベル付き
            s = rng.choice(["氏名", "住所", "電話", "日付", "金額", "備考", "TEL", "No."]) + rng.choice(["：", ":", " ", "　"]) + s
        return s
    if r < 0.70:
        return pseudo_japanese(rng)
    if r < 0.85:
        return uniform_chars(rng)
    return alnum(rng)
