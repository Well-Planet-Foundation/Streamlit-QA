"""data.py — load survey CSV(s), build a codebook, compute composite indices.

Design notes
------------
* Several question batteries appear twice in the export (pandas names the second copy ".1")
  because Google Forms asks them separately of institution staff and independent workers.
  We merge those into one canonical column.
* Every column is auto-typed: ordinal Likert / categorical / multi-select / excluded (PII, free text).
* Several files with the same column layout (one per year) are stacked and get a "年度" column.
* Nothing here talks to an LLM. The LLM only ever sees the codebook (names + option labels),
  never raw rows.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- scales
# Ordered low -> high. Code = position + 1.
SCALES: dict[str, list[str]] = {
    "困擾程度": ["完全沒有", "輕微", "中等", "嚴重", "非常嚴重"],
    "同意程度": ["非常不同意", "不同意", "普通", "同意", "非常同意"],
    "離職念頭頻率": ["從未", "一次或兩次", "一個月一次", "一週一次", "每天"],
    "留任意願": ["確定不會", "應該不會", "不確定", "應該會，但不確定更久以後", "會，而且我打算長期待下去"],
    "有無": ["無", "有"],
    "投入頻率": ["從未", "幾乎從不", "很少", "有時", "經常", "非常經常", "總是"],
    "疲勞頻率": ["從未或幾乎從未", "不常", "有時候", "常常", "總是"],
    "助人感受頻率": ["從未有過", "很少", "有些時候", "經常如此", "總是如此"],
    "年資": ["1 年以內", "1 年以上，未滿 3 年", "3 年以上，未滿 5 年", "5 年以上，未滿 8 年",
           "8 年以上，未滿 15 年", "15 年以上"],
    "機構規模": ["1–4 人", "5–10 人", "11–15 人", "16–30 人", "31–50 人", "51 人以上"],
    "約定工時": ["20 小時以下", "21–40 小時", "41 小時以上"],
    "實際工時": ["20 小時以下", "21–40 小時", "41–50 小時", "51–60 小時", "61 小時以上"],
    "年齡": ["18–24 歲", "25–29 歲", "30–34 歲", "35–39 歲", "40–49 歲", "50 歲以上"],
}
# labels that are real answers (kept for grouping) but have no position on the scale
EXTRA_NA: dict[str, set[str]] = {
    "約定工時": {"沒有明確約定"},
    "實際工時": {"不固定，難以估算"},
    "年齡": {"不願透露"},
}
GLOBAL_NA = {"不適用"}  # dropped completely
# ordinal scales that describe the person/organisation rather than an outcome
DEMOGRAPHIC_SCALES = {"年資", "機構規模", "約定工時", "實際工時", "年齡"}
# prefix for "請依照你的實際感受作答 [..]" style batteries, chosen by detected scale
SCALE_PREFIX = {"投入頻率": "投入", "疲勞頻率": "疲勞", "助人感受頻率": "助人感受"}
STEM_PREFIX = [("困擾程度", "困擾"), ("同意程度", "工作條件"), ("是否曾有過以下行為", "求職行為"),
               ("前三大挑戰", "獨立者挑戰")]

# readable names for non-battery questions (first regex that matches wins)
RENAME = [
    (r"^你目前的助人", "工作型態"),
    (r"^你任職機構的性質", "機構性質"),
    (r"^你的機構目前全職人數", "機構規模"),
    (r"^你的機構主要服務領域", "服務領域"),
    (r"^你的助人工作總年資", "助人年資"),
    (r"^你在目前所屬機構的總年資", "機構年資"),
    (r"約定", "約定每週工時"),
    (r"^你「過去七天」", "過去七天實際工時"),
    (r"^你在機構中的角色", "機構角色"),
    (r"^過去六個月內，你多常認真考慮離開", "離職念頭頻率"),
    (r"^你預期一年後", "一年後是否留任"),
    (r"^就你所知，你的機構目前有以下哪些支持制度", "現有支持制度（受雇者填答）"),
    (r"^如果只能選三項", "希望增加的支持（受雇者填答）"),
    (r"^你認為你的機構之所以還沒有", "未建立制度的原因（受雇者填答）"),
    (r"^你的機構目前有以下哪些支持制度", "現有支持制度（決策者／人資）"),
    (r"^如果至多只能選三項", "希望增加的支持（決策者／人資）"),
    (r"^影響你的機構導入", "導入方案的考量（決策者／人資）"),
    (r"^什麼樣的成效", "認定方案有用的成效（決策者／人資）"),
    (r"^如果有這樣一個免費", "是否希望機構參與免費方案"),
    (r"^你認為你的機構決策者", "決策者可能態度"),
    (r"^你目前主要的工作型態", "獨立者工作型態"),
    (r"^什麼樣的支持，會對現在的你最有幫助", "獨立者最需要的支持"),
    (r"^接下來的題組會詢問", "是否填答創傷題組"),
    (r"^性別$", "性別"),
    (r"^年齡$", "年齡"),
    (r"^主要工作縣市$", "工作縣市"),
]
EXCLUDE_RE = re.compile(r"E-?mail|時間戳記|個人資料|商品卡|你希望之後收到|還有什麼是這份問卷沒問到", re.I)

# --------------------------------------------------------------------------- composites
# (name, description incl. direction, needles in item text, reversed needles, prefix-group or None)
PERSONAL = ["你常覺得疲勞", "體力透支", "你常覺得情緒上心力交瘁", "快要撐不下去了", "你常覺得精疲力竭", "虛弱"]
WORK = ["你的工作會令人情緒上心力交瘁", "快要累垮", "你的工作會讓你覺得挫折", "工作一整天之後",
        "想到又要工作一整天", "每一刻都很難熬"]
WORK_REV = ["有足夠的精力陪朋友或家人"]
COMPOSITES = [
    dict(name="個人倦怠指數", needles=PERSONAL, rev=[], group=None,
         desc="個人層面的疲憊（參考 CBI 個人倦怠 6 題）；0-100，越高越疲憊"),
    dict(name="工作倦怠指數", needles=WORK, rev=WORK_REV, group=None,
         desc="與工作相關的疲憊（參考 CBI 工作倦怠 7 題，最後一題反向計分）；0-100，越高越疲憊"),
    dict(name="工作投入指數", needles=["充滿能量", "充滿熱忱", "沉浸在我的工作"], rev=[], group=None,
         desc="活力/熱忱/專注（參考 UWES-3）；0-100，越高越投入"),
    dict(name="工作要求困擾指數", needles=[], rev=[], group="困擾",
         desc="9 項工作困擾（工作量、角色不清、溝通、薪資…）的平均；0-100，越高困擾越嚴重"),
    dict(name="工作條件與支持指數", needles=[], rev=[], group="工作條件",
         desc="9 項工作資源（期待清楚、自主、同事/主管支持、肯定、平衡、發展）的平均；0-100，越高越正向"),
    dict(name="同情滿足指數", needles=["我喜歡從事助人", "我的工作使我感到滿意", "很高興我選擇"], rev=[], group=None,
         desc="助人工作的滿足感（簡版 ProQOL 自訂分組）；0-100，越高越滿足"),
    dict(name="助人倦怠指數", needles=["陷入困境", "筋疲力竭，因為", "負擔似乎沒有止盡"], rev=[], group=None,
         desc="助人工作造成的倦怠感（簡版 ProQOL 自訂分組）；0-100，越高越倦怠"),
    dict(name="次級創傷壓力指數", needles=["被這些我所幫助的嚴重創傷者所影響", "讓我感到沮喪", "侵擾和令人恐懼"], rev=[], group=None,
         desc="受服務對象創傷經驗影響的程度（簡版 ProQOL 自訂分組）；0-100，越高越嚴重"),
    dict(name="離職意圖指數", needles=["認真考慮離開", "主動搜尋其他工作機會", "更新你的履歷", "申請其他工作機會"],
         rev=["你預期一年後"], group=None,
         desc="離職念頭頻率、預期一年後不留任、實際求職行為的平均；0-100，越高越想離開"),
]
MIN_VALID_FRAC = 0.7


# --------------------------------------------------------------------------- helpers
def short(s, n: int = 18) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _clean_header(c) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def _base(c) -> str:
    return re.sub(r"\.\d+$", "", _clean_header(c))


def _norm(v):
    if not isinstance(v, str):
        return v
    v = v.replace("\u3000", " ").replace("—", "–")
    v = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "–", v).strip()
    return v if v else np.nan


def _strip_q(s: str) -> str:
    return re.sub(r"[。？?\s]+$", "", s.strip())


def _coalesce(df: pd.DataFrame) -> pd.DataFrame:
    groups: dict[str, list] = {}
    for c in df.columns:
        groups.setdefault(_base(c), []).append(c)
    out = {}
    for base, cols in groups.items():
        s = df[cols[0]]
        for c in cols[1:]:
            s = s.where(s.notna(), df[c])
        out[base] = s
    return pd.DataFrame(out)


def detect_scale(s: pd.Series):
    counts = s.dropna().value_counts()
    if counts.empty:
        return None
    total = counts.sum()
    best = None
    for name, labels in SCALES.items():
        vocab = set(labels) | EXTRA_NA.get(name, set()) | GLOBAL_NA
        matched = counts[counts.index.isin(vocab)]
        cov = matched.sum() / total
        if cov >= 0.9 and len(matched) >= 2:
            key = (cov, -len(vocab))
            if best is None or key > best[0]:
                best = (key, name)
    return best[1] if best else None


# --------------------------------------------------------------------------- data classes
@dataclass
class Var:
    name: str
    kind: str                      # ordinal | categorical | multi | composite | year
    alias: str = ""
    item: str = ""                 # question / item text (used to find items for composites)
    group: str | None = None       # battery prefix
    scale: str | None = None
    levels: list[str] = field(default_factory=list)   # ordered labels / categories / options
    desc: str = ""
    n_valid: int = 0
    outcome: bool = False          # True => can be treated as a measured value (mean / correlation)

    @property
    def numeric(self) -> bool:
        return self.kind == "composite" or (self.kind == "ordinal" and self.outcome)


@dataclass
class Bundle:
    vars: dict[str, Var]
    lab: pd.DataFrame
    num: pd.DataFrame
    multi: dict[str, pd.DataFrame]
    years: list[str]
    year_counts: dict[str, int]
    warnings: list[str]
    excluded: list[str]
    index: pd.Index

    @property
    def n(self) -> int:
        return len(self.index)

    # ---- name resolution (exact -> alias -> prefix -> substring -> fuzzy)
    def resolve(self, name):
        if name is None:
            return None
        name = _strip_q(str(name))
        if name in self.vars:
            return self.vars[name]
        for v in self.vars.values():
            if v.alias == name:
                return v
        low = name.lower()
        for v in self.vars.values():
            if v.name.lower() == low or v.alias.lower() == low:
                return v
        cands = [v for v in self.vars.values() if v.name.startswith(name) or v.alias.startswith(name)]
        if len(cands) == 1:
            return cands[0]
        cands = [v for v in self.vars.values() if name in v.name or name in v.item]
        if len(cands) == 1:
            return cands[0]
        import difflib
        pool = {}
        for v in self.vars.values():
            pool[v.name] = v
            pool[v.alias] = v
        m = difflib.get_close_matches(name, list(pool), n=1, cutoff=0.6)
        return pool[m[0]] if m else None

    def suggest(self, name, k=4):
        import difflib
        names = [v.name for v in self.vars.values()]
        return difflib.get_close_matches(str(name), names, n=k, cutoff=0.2)

    # ---- codebook for humans
    def codebook_df(self) -> pd.DataFrame:
        rows = []
        tag = {"ordinal": "序位", "categorical": "類別", "multi": "複選", "composite": "綜合指數", "year": "年度"}
        for v in self.vars.values():
            if v.kind == "ordinal":
                opts = " < ".join(v.levels)
            elif v.kind == "composite":
                opts = v.desc
            else:
                opts = " | ".join(short(x, 24) for x in v.levels[:10]) + (" …" if len(v.levels) > 10 else "")
            rows.append({"變數": v.name, "簡稱": v.alias, "類型": tag[v.kind], "有效樣本": v.n_valid, "選項/說明": opts})
        return pd.DataFrame(rows)

    # ---- compact codebook for the LLM planner
    def codebook_text(self, max_levels: int = 5) -> str:
        L = [f"Respondents: {self.n}. Years loaded: {', '.join(self.years)}."]
        comp = [v for v in self.vars.values() if v.kind == "composite"]
        if comp:
            L.append("== Composite indices (numeric 0-100; best for correlations/means) ==")
            for v in comp:
                L.append(f"- {v.name} (n={v.n_valid}): {v.desc}")
        batteries: dict[str, list[Var]] = {}
        for v in self.vars.values():
            if v.group and v.kind in ("ordinal", "categorical"):
                batteries.setdefault(v.group, []).append(v)
        if batteries:
            L.append("== Item batteries (use the alias, e.g. 困擾3) ==")
            for g, items in batteries.items():
                first = items[0]
                scale = f"scale {first.scale}: " + "<".join(first.levels) if first.kind == "ordinal" else "categorical"
                ns = [i.n_valid for i in items]
                L.append(f"* {g} [{scale}] n={min(ns)}-{max(ns)}")
                L.append("  " + "；".join(f"{i.alias}={short(i.item, 11)}" for i in items))
        L.append("== Single variables ==")
        for v in self.vars.values():
            if v.kind == "composite" or (v.group and v.kind in ("ordinal", "categorical")):
                continue
            if v.kind == "ordinal":
                role = "demographic-ordinal" if not v.outcome else "ordinal"
                L.append(f"- {v.name} [{role}] n={v.n_valid}: " + "<".join(short(x, 7) for x in v.levels))
            elif v.kind == "year":
                L.append(f"- {v.name} [year] values: {' | '.join(v.levels)}")
            else:
                tagk = "multi-select" if v.kind == "multi" else "categorical"
                lv = " | ".join(short(x, 9) for x in v.levels[:max_levels])
                more = f" …(+{len(v.levels) - max_levels})" if len(v.levels) > max_levels else ""
                L.append(f"- {v.name} [{tagk}] n={v.n_valid}: {lv}{more}")
        return "\n".join(L)


# --------------------------------------------------------------------------- loading
def _read_one(src) -> pd.DataFrame:
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    return pd.read_csv(buf, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])


def load_bundle(sources: list[tuple[str, object]]) -> Bundle:
    """sources: [(year_label, csv_bytes | path | file-like), ...]"""
    warnings: list[str] = []
    frames, ref_cols, ref_label = [], None, None
    for label, src in sources:
        raw = _read_one(src)
        df = _coalesce(raw)
        df = df.apply(lambda col: col.map(_norm))
        cols = set(df.columns)
        if ref_cols is None:
            ref_cols, ref_label = cols, label
        else:
            miss, extra = sorted(ref_cols - cols), sorted(cols - ref_cols)
            if miss:
                warnings.append(f"{label} 年度缺少 {len(miss)} 個欄位（與 {ref_label} 年度相比），例如：{short(miss[0], 40)}")
            if extra:
                warnings.append(f"{label} 年度多出 {len(extra)} 個欄位，例如：{short(extra[0], 40)}")
        df.insert(0, "__year__", str(label))
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    year_counts = full["__year__"].value_counts().to_dict()
    years = sorted(year_counts, key=lambda y: (not str(y).isdigit(), str(y)))

    idx = full.index
    vars_: dict[str, Var] = {}
    lab = pd.DataFrame(index=idx)
    num = pd.DataFrame(index=idx)
    multi: dict[str, pd.DataFrame] = {}
    excluded: list[str] = []
    counters: dict[str, int] = {}

    # year variable
    lab["年度"] = full["__year__"]
    vars_["年度"] = Var("年度", "year", alias="年度", levels=years, n_valid=len(full),
                       desc="資料年度（多年趨勢分析用）")
    if all(str(y).isdigit() for y in years):
        num["年度"] = full["__year__"].astype(float)

    def unique_name(n):
        if n not in vars_:
            return n
        k = 2
        while f"{n}({k})" in vars_:
            k += 1
        return f"{n}({k})"

    for col in [c for c in full.columns if c != "__year__"]:
        s = full[col]
        nn = s.dropna()
        if nn.empty or EXCLUDE_RE.search(col) or nn.nunique() < 2:
            excluded.append(col)
            continue
        m = re.match(r"^(.*?)\s*\[(.+)\]\s*$", col, re.S)
        stem, item = (m.group(1), m.group(2)) if m else (col, col)
        stem, item = _strip_q(stem), _strip_q(item)
        scale = detect_scale(s)

        # ---- naming
        group = None
        if m:
            group = SCALE_PREFIX.get(scale) if scale in SCALE_PREFIX else None
            if group is None:
                for key, pref in STEM_PREFIX:
                    if key in stem:
                        group = pref
                        break
            if group is None:
                group = short(stem, 8)
            name = unique_name(f"{group}｜{item}")
            counters[group] = counters.get(group, 0) + 1
            alias = f"{group}{counters[group]}"
        else:
            name = None
            for rx, nice in RENAME:
                if re.search(rx, stem):
                    name = nice
                    break
            name = unique_name(name or short(stem, 40))
            alias = name

        # ---- ordinal Likert / ordered demographics
        if scale:
            labels = SCALES[scale]
            extra = EXTRA_NA.get(scale, set())
            unknown = sorted(set(nn.unique()) - set(labels) - extra - GLOBAL_NA)
            if unknown:
                warnings.append(f"「{short(name, 30)}」有未識別的選項（已視為缺值）：{', '.join(short(u, 16) for u in unknown[:4])}")
            code = {l: i + 1 for i, l in enumerate(labels)}
            clean = s.where(~s.isin(GLOBAL_NA | set(unknown)), np.nan)
            lab[name] = clean
            num[name] = clean.map(code).astype(float)
            present = [l for l in labels + sorted(extra) if l in set(clean.dropna())]
            vars_[name] = Var(name, "ordinal", alias=alias, item=item, group=group, scale=scale, levels=present,
                              n_valid=int(clean.notna().sum()), outcome=scale not in DEMOGRAPHIC_SCALES)
            continue

        # ---- multi-select
        frac_comma = nn.str.contains(r",\s*", regex=True).mean()
        avg_len = nn.str.len().mean()
        if frac_comma >= 0.15:
            parts = nn.map(lambda t: [p.strip() for p in re.split(r",\s*", t) if p.strip()])
            counts = pd.Series([o for lst in parts for o in lst]).value_counts()
            opts = list(counts.index)
            oh = pd.DataFrame(False, index=idx, columns=opts)
            for i, lst in parts.items():
                for o in lst:
                    oh.at[i, o] = True
            multi[name] = oh
            vars_[name] = Var(name, "multi", alias=alias, item=item, group=group, levels=opts,
                              n_valid=int(len(nn)))
            continue

        # ---- free text / identifiers
        if (avg_len > 30 and nn.nunique() / len(nn) > 0.5) or nn.nunique() > 40:
            excluded.append(col)
            continue

        # ---- categorical
        if "縣市" in col:  # people typed names / e-mails here: keep real counties only
            s = s.where(s.str.contains(r"[縣市]$", na=False), "其他／無法辨識").where(s.notna(), np.nan)
        lab[name] = s
        levels = list(s.value_counts().index)
        vars_[name] = Var(name, "categorical", alias=alias, item=item, group=group, levels=levels,
                          n_valid=int(s.notna().sum()))

    bundle = Bundle(vars_, lab, num, multi, years, year_counts, warnings, excluded, idx)
    _add_composites(bundle)
    return bundle


def _add_composites(b: Bundle) -> None:
    for spec in COMPOSITES:
        items: list[tuple[Var, bool]] = []
        if spec["group"]:
            items = [(v, False) for v in b.vars.values() if v.group == spec["group"] and v.kind == "ordinal"]
        else:
            for needle in spec["needles"]:
                hit = [v for v in b.vars.values() if v.kind == "ordinal" and needle in v.item]
                if hit:
                    items.append((hit[0], False))
            for needle in spec["rev"]:
                hit = [v for v in b.vars.values() if v.kind == "ordinal" and needle in v.item]
                if hit:
                    items.append((hit[0], True))
        expected = len(spec["needles"]) + len(spec["rev"]) if not spec["group"] else max(len(items), 1)
        if len(items) < max(2, math.ceil(MIN_VALID_FRAC * expected)):
            b.warnings.append(f"無法建立「{spec['name']}」：只找到 {len(items)}/{expected} 題")
            continue
        units = []
        for v, rev in items:
            k = len(SCALES[v.scale])
            u = (b.num[v.name] - 1) / (k - 1) * 100
            units.append(100 - u if rev else u)
        U = pd.concat(units, axis=1)
        need = math.ceil(MIN_VALID_FRAC * U.shape[1])
        score = U.mean(axis=1).where(U.notna().sum(axis=1) >= need).round(1)
        if score.notna().sum() < 5:
            continue
        b.num[spec["name"]] = score
        note = f"{spec['desc']}；由 {len(items)} 題組成"
        b.vars[spec["name"]] = Var(spec["name"], "composite", alias=spec["name"], desc=note,
                                   n_valid=int(score.notna().sum()), outcome=True)
        if len(items) < expected:
            b.warnings.append(f"「{spec['name']}」只用到 {len(items)}/{expected} 題")
