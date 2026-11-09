"""analysis.py — executes a validated analysis plan: statistics (scipy) + charts (plotly).

The LLM never computes anything. It proposes a plan such as
    {"charts": [{"type": "relationship", "x": "工作要求困擾指數", "y": "個人倦怠指數"}]}
and this module resolves the variable names, runs the numbers and builds the figures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from scipy import stats as sps

from data import Bundle, Var, short

FONT = "Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif"
INK = "#3d4146"
MUTED = "#70777b"
ACCENT = "#4f837a"
CHART_COLORS = (
    ACCENT,
    "#b65d73",
    "#756096",
    "#a77d2e",
    "#4d7599",
    "#66875d",
    "#9c5c80",
    "#447987",
)
MIN_CORR_N = 8
MAX_GROUPS = 12
CHART_TYPES = {"relationship", "distribution", "heatmap", "trend"}


class PlanError(Exception):
    pass


@dataclass
class ChartResult:
    spec: dict
    title: str
    fig: object | None = None
    stats: dict = field(default_factory=dict)
    headline: str = ""          # one deterministic line with the key numbers
    notes: list[str] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- small helpers
def _f(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def p_str(p):
    if p is None:
        return "p=NA"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def corr_strength(r):
    a = abs(r)
    if a < 0.1:
        return "幾乎無相關"
    lab = "弱" if a < 0.3 else "中等" if a < 0.5 else "強"
    return f"{lab}{'正' if r > 0 else '負'}相關"


def eps_strength(e):
    return "可忽略" if e < 0.01 else "小" if e < 0.06 else "中" if e < 0.14 else "大"


def wrap(s, n=12):
    s = str(s)
    return "<br>".join(s[i:i + n] for i in range(0, len(s), n)) if len(s) > n else s


def _layout(fig, title, height=None, **kw):
    layout = dict(
        template="plotly_white",
        colorway=CHART_COLORS,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        title=dict(
            text=wrap(title, 34),
            x=0.02,
            xanchor="left",
            y=0.97,
            font=dict(size=16, color=INK, family=FONT),
        ),
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=22, r=24, t=66, b=52),
        xaxis=dict(
            automargin=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color=MUTED, size=11),
            title=dict(font=dict(color=MUTED, size=12)),
        ),
        yaxis=dict(
            automargin=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color=MUTED, size=11),
            title=dict(font=dict(color=MUTED, size=12)),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(color=INK, size=11),
        ),
        hoverlabel=dict(
            bgcolor=INK,
            bordercolor=INK,
            font=dict(family=FONT, size=12, color="#ffffff"),
        ),
        height=min(height, 500) if height is not None else None,
    )
    for axis in ("xaxis", "yaxis", "legend"):
        if axis in kw:
            options = kw.pop(axis)
            if isinstance(options.get("title"), dict) and isinstance(layout[axis].get("title"), dict):
                options["title"] = layout[axis]["title"] | options["title"]
            layout[axis] = layout[axis] | options
    layout.update(kw)
    fig.update_layout(**layout)
    return fig


def _val_axis(v: Var):
    if v.kind == "composite":
        return dict(range=[0, 100], title=dict(text=f"{v.name}（0–100，平均）"))
    k = len(v.levels)
    return dict(range=[1, k], title=dict(text=f"{short(v.name, 24)} 平均分數（1={short(v.levels[0], 8)} … {k}={short(v.levels[-1], 8)}）"))


def _need(b: Bundle, name, what="變數") -> Var:
    v = b.resolve(name)
    if v is None:
        sug = b.suggest(name)
        raise PlanError(f"找不到{what}「{name}」" + (f"；你是不是想用：{'、'.join(sug)}？" if sug else ""))
    return v


def _match_levels(vals, levels):
    out = []
    for val in vals:
        val = str(val).strip().rstrip("…")
        hit = [l for l in levels if l == val] or [l for l in levels if l.startswith(val) or val in l]
        for h in hit:
            if h not in out:
                out.append(h)
    return out


# --------------------------------------------------------------------------- filters / groups
def build_mask(b: Bundle, flt, notes: list[str]):
    mask = pd.Series(True, index=b.index)
    desc = []
    for key, vals in (flt or {}).items():
        v = b.resolve(key)
        if v is None:
            notes.append(f"篩選變數「{key}」找不到，已忽略")
            continue
        vals = vals if isinstance(vals, list) else [vals]
        if v.kind == "multi":
            hit = _match_levels(vals, v.levels)
            if hit:
                mask &= b.multi[v.name][hit].any(axis=1)
        elif v.kind in ("categorical", "ordinal", "year"):
            hit = _match_levels(vals, v.levels)
            if hit:
                mask &= b.lab[v.name].isin(hit)
        else:
            notes.append(f"「{v.name}」是連續指數，不支援篩選，已忽略")
            continue
        if not hit:
            notes.append(f"篩選「{v.name}」找不到選項 {vals}，已忽略")
            continue
        desc.append(f"{v.name} = {'、'.join(short(h, 20) for h in hit)}")
    return mask, desc


@dataclass
class Groups:
    s: pd.Series            # index = respondent, value = group label (repeated index for multi-select)
    order: list[str]
    ordered: bool
    overlap: bool           # respondents can sit in several groups (multi-select)


def get_groups(b: Bundle, v: Var, mask, min_n: int, notes: list[str], cap=MAX_GROUPS) -> Groups:
    overlap = False
    ordered = False
    if v.kind in ("categorical", "ordinal", "year"):
        s = b.lab[v.name][mask].dropna()
        order = [l for l in v.levels if l in set(s)]
        ordered = v.kind in ("ordinal", "year")
    elif v.kind == "multi":
        oh = b.multi[v.name][mask]
        parts = [pd.Series(c, index=oh.index[oh[c].astype(bool)], dtype=object) for c in oh.columns]
        s = pd.concat(parts) if parts else pd.Series(dtype=object)
        order = list(s.value_counts().index)
        overlap = True
    elif v.kind == "composite":
        x = b.num[v.name][mask].dropna()
        if len(x) < 3 * min_n:
            raise PlanError(f"「{v.name}」有效樣本太少（{len(x)}），無法分成低/中/高三組")
        q = pd.qcut(x.rank(method="first"), 3, labels=["低組（後 1/3）", "中組", "高組（前 1/3）"])
        s = q.astype(str)
        order = ["低組（後 1/3）", "中組", "高組（前 1/3）"]
        ordered = True
        notes.append(f"「{v.name}」依分數分成三等分（相對分組，非臨床切點）")
    else:
        raise PlanError(f"「{v.name}」不能當分組變數")
    counts = s.groupby(level=0).first().value_counts() if not overlap else s.value_counts()
    small = [g for g in order if counts.get(g, 0) < min_n]
    if small:
        notes.append(f"已排除樣本數 < {min_n} 的組別：{'、'.join(short(g, 14) for g in small[:5])}{'…' if len(small) > 5 else ''}")
    order = [g for g in order if g not in small]
    if len(order) > cap:
        if not ordered:
            order = sorted(order, key=lambda g: -counts.get(g, 0))[:cap]
            notes.append(f"組別過多，只顯示樣本數最多的 {cap} 組")
        else:
            order = order[:cap]
    s = s[s.isin(order)]
    return Groups(s, order, ordered, overlap)


def _n_unique(s: pd.Series) -> int:
    return int(s.index.nunique())


def _title_with_filter(title, fdesc):
    return title + (f"（{'；'.join(fdesc)}）" if fdesc else "")


# --------------------------------------------------------------------------- charts
def _group_means(b, val: Var, by: Var, mask, min_n, spec, notes, fdesc) -> ChartResult:
    g = get_groups(b, by, mask, min_n, notes)
    d = g.s.rename("g").to_frame().join(b.num[val.name][mask].rename("v")).dropna()
    rows = []
    for lv in g.order:
        x = d.loc[d["g"] == lv, "v"]
        if len(x) < min_n:
            continue
        ci = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0
        rows.append(dict(group=lv, n=int(len(x)), mean=float(x.mean()), ci=float(ci), median=float(x.median())))
    if len(rows) < 2:
        raise PlanError(f"「{by.name}」可比較的組別不足 2 組（每組至少 {min_n} 人）")
    if not g.ordered:
        rows.sort(key=lambda r: -r["mean"])
    n_total = int(d.index.nunique())
    st: dict = dict(type="group_means", value=val.name, by=by.name, n_total=n_total,
                    groups=[dict(group=r["group"], n=r["n"], mean=_f(r["mean"]), ci95=_f(r["ci"]), median=_f(r["median"])) for r in rows])
    head = []
    # rank correlation when the grouping variable has an order
    if g.ordered and by.kind in ("ordinal", "year") and by.name in b.num.columns:
        z = pd.DataFrame({"x": b.num[by.name][mask], "y": b.num[val.name][mask]}).dropna()
        if len(z) >= MIN_CORR_N and z["x"].nunique() > 1 and z["y"].nunique() > 1:
            rho, p = sps.spearmanr(z["x"], z["y"])
            st.update(spearman_rho=_f(rho), spearman_p=_f(p, 4), spearman_n=int(len(z)))
            head.append(f"Spearman ρ={rho:.2f}（{corr_strength(rho)}，{p_str(p)}，n={len(z)}）")
    if not g.overlap:
        arrs = [d.loc[d["g"] == r["group"], "v"].values for r in rows]
        try:
            H, p = sps.kruskal(*arrs)
            k, n = len(arrs), sum(len(a) for a in arrs)
            eps = max(0.0, (H - k + 1) / (n - k))
            st.update(kruskal_H=_f(H), kruskal_p=_f(p, 4), epsilon_sq=_f(eps, 3))
            head.append(f"組間差異 Kruskal-Wallis {p_str(p)}，效應量 ε²={eps:.3f}（{eps_strength(eps)}）")
        except ValueError:
            notes.append("各組數值完全相同，無法做組間檢定")
    else:
        notes.append("複選題：同一人可能同時出現在多個組別，因此不做組間顯著性檢定，且各組不互斥")
    if n_total < 30:
        notes.append(f"樣本數偏少（n={n_total}），結果請保守解讀")
    st["overlapping_groups"] = g.overlap

    # ---- figure
    title = spec.get("title") or f"各「{by.name}」的{val.name}"
    title = _title_with_filter(title, fdesc)
    labels = [f"{r['group']}（n={r['n']}）" for r in rows]
    means = [r["mean"] for r in rows]
    err = dict(type="data", array=[r["ci"] for r in rows], visible=True,
               color="#82938f", thickness=1.4, width=4)
    base = 0 if val.kind == "composite" else 1
    txt = [f"{m:.1f}" if val.kind == "composite" else f"{m:.2f}" for m in means]
    fig = go.Figure()
    vert = g.ordered
    if vert:
        fig.add_trace(go.Bar(x=[wrap(l, 9) for l in labels], y=means, error_y=err, text=txt, textposition="outside",
                             marker_color=ACCENT, marker_line_color="#ffffff", marker_line_width=1,
                             textfont=dict(color=INK, size=11), base=base,
                             hovertemplate="%{x}<br>平均=%{y:.2f}<extra></extra>"))
        fig.update_layout(yaxis=_val_axis(val) | dict(showgrid=True, gridcolor="#e8efec"))
        _layout(fig, title, height=340)
    else:
        fig.add_trace(go.Bar(y=[wrap(l, 16) for l in labels], x=means, error_x=err, orientation="h", text=txt,
                             textposition="outside", marker_color=ACCENT, marker_line_color="#ffffff",
                             marker_line_width=1, textfont=dict(color=INK, size=11), base=base,
                             hovertemplate="%{y}<br>平均=%{x:.2f}<extra></extra>"))
        fig.update_layout(
            xaxis=_val_axis(val) | dict(showgrid=True, gridcolor="#e8efec"),
            yaxis=dict(autorange="reversed", automargin=True),
        )
        _layout(fig, title, height=max(290, 76 + 42 * len(rows)))
    fig.update_layout(showlegend=False)
    notes.append("長條上的線為 95% 信賴區間（平均值的不確定範圍）")
    return ChartResult(spec, title, fig, st, "；".join(head) or "各組平均如圖", notes)


def _scatter(b, x: Var, y: Var, mask, spec, notes, fdesc) -> ChartResult:
    d = pd.DataFrame({"x": b.num[x.name][mask], "y": b.num[y.name][mask]}).dropna()
    n = len(d)
    if n < MIN_CORR_N:
        raise PlanError(f"兩個變數同時有效的樣本只有 {n}，不足以計算相關")
    if d["x"].nunique() < 2 or d["y"].nunique() < 2:
        raise PlanError("其中一個變數沒有變異（全部相同），無法計算相關")
    rho, p = sps.spearmanr(d["x"], d["y"])
    r, pr = sps.pearsonr(d["x"], d["y"])
    slope, icpt = np.polyfit(d["x"], d["y"], 1)
    st = dict(type="correlation", x=x.name, y=y.name, n=n, spearman_rho=_f(rho), spearman_p=_f(p, 4),
              pearson_r=_f(r), pearson_p=_f(pr, 4), slope_per_10_x=_f(slope * 10), strength=corr_strength(rho))
    if n < 30:
        notes.append(f"樣本數偏少（n={n}），結果請保守解讀")
    title = _title_with_filter(spec.get("title") or f"{x.name} 與 {y.name}", fdesc)
    rng = np.random.default_rng(0)
    yy = d["y"] + (rng.uniform(-0.12, 0.12, n) if y.kind == "ordinal" else 0)
    xs = np.linspace(d["x"].min(), d["x"].max(), 20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["x"],
        y=yy,
        mode="markers",
        name="受訪者",
        marker=dict(size=8, opacity=0.58, color=CHART_COLORS[0], line=dict(width=1, color="#ffffff")),
        hovertemplate=f"{x.name}=%{{x:.2f}}<br>{y.name}=%{{y:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=slope * xs + icpt,
        mode="lines",
        name="線性趨勢線",
        line=dict(color=CHART_COLORS[1], width=3),
        hovertemplate="趨勢線<extra></extra>",
    ))
    fig.update_layout(
        xaxis=_val_axis(x) | dict(title=dict(text=x.name), showgrid=True, gridcolor="#e8efec"),
        yaxis=_val_axis(y) | dict(title=dict(text=y.name), showgrid=True, gridcolor="#e8efec"),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    _layout(fig, title, height=360)
    notes.append("相關不等於因果：不能單憑此圖判斷誰影響誰")
    head = (f"Spearman ρ={rho:.2f}（{corr_strength(rho)}，{p_str(p)}，n={n}）；"
            f"{x.name}每增加 10 分，{y.name}平均約{'增加' if slope >= 0 else '減少'} {abs(slope * 10):.1f}")
    return ChartResult(spec, title, fig, st, head, notes)


def _crosstab(b, x: Var, y: Var, mask, min_n, spec, notes, fdesc) -> ChartResult:
    gx = get_groups(b, x, mask, min_n, notes)
    gy = get_groups(b, y, mask, min_n, notes, cap=10)
    d = gx.s.rename("gx").to_frame().join(gy.s.rename("gy"), how="inner")
    if d.empty:
        raise PlanError("兩個變數沒有同時作答的受訪者")
    tab = pd.crosstab(d["gx"], d["gy"]).reindex(index=gx.order, columns=gy.order).fillna(0).astype(int)
    tab = tab.loc[tab.sum(axis=1) >= min_n]
    tab = tab.loc[:, tab.sum(axis=0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise PlanError("可比較的組別不足（每組至少 %d 人）" % min_n)
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    n_total = int(d.index.nunique())
    st = dict(type="crosstab", x=x.name, y=y.name, n_total=n_total,
              row_percent={str(r): {str(c): _f(pct.loc[r, c], 1) for c in pct.columns} for r in pct.index},
              row_n={str(r): int(tab.loc[r].sum()) for r in tab.index})
    head = []
    if not (gx.overlap or gy.overlap):
        chi2, p, dof, exp = sps.chi2_contingency(tab.values)
        V = math.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        low = float((exp < 5).mean())
        st.update(chi2_p=_f(p, 4), cramers_v=_f(V, 3), expected_lt5_share=_f(low, 2))
        head.append(f"卡方檢定 {p_str(p)}，Cramér's V={V:.2f}（{eps_strength(V ** 2)}關聯）")
        if low > 0.2:
            notes.append("超過兩成的格子期望次數 < 5，卡方檢定不太可靠；請以比例為主、勿過度解讀 p 值")
    else:
        notes.append("複選題：同一人可能出現在多個組別，因此不做卡方檢定，百分比加總可能超過 100%")
    if n_total < 30:
        notes.append(f"樣本數偏少（n={n_total}），結果請保守解讀")
    title = _title_with_filter(spec.get("title") or f"{x.name} × {y.name}", fdesc)
    if y.kind == "ordinal":
        cols = sample_colorscale(
            [[0, "#5d8c80"], [1, "#3d746b"]],
            [i / max(1, len(pct.columns) - 1) for i in range(len(pct.columns))],
        )
    else:
        pal = CHART_COLORS
        cols = [pal[i % len(pal)] for i in range(len(pct.columns))]
    fig = go.Figure()
    ylabels = [wrap(f"{r}（n={int(tab.loc[r].sum())}）", 16) for r in pct.index]
    for c, col in zip(pct.columns, cols):
        vals = pct[c].values
        fig.add_trace(go.Bar(y=ylabels, x=vals, name=short(c, 16), orientation="h", marker_color=col,
                             marker_line_color="#ffffff", marker_line_width=1,
                             text=[f"{v:.0f}%" if v >= 7 else "" for v in vals], textposition="inside",
                             insidetextfont=dict(color="#ffffff", size=11),
                             hovertemplate="%{y}<br>" + str(c) + "：%{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], title=dict(text="該組內的百分比（%）"),
                                                 showgrid=True, gridcolor="#e8efec"),
                      yaxis=dict(autorange="reversed", automargin=True),
                      legend=dict(orientation="h", y=-0.25, x=0, title=dict(text=short(y.name, 24))))
    _layout(fig, title, height=max(300, 118 + 44 * len(pct.index)))
    return ChartResult(spec, title, fig, st, "；".join(head) or "各組分布如圖", notes)


def _relationship(b, spec, min_n, notes, fdesc, mask) -> ChartResult:
    x, y = _need(b, spec.get("x"), "x 變數"), _need(b, spec.get("y"), "y 變數")
    if x.name == y.name:
        raise PlanError("x 與 y 是同一個變數")
    grp_ok = lambda v: v.kind in ("categorical", "ordinal", "year", "multi", "composite")
    if x.numeric and y.numeric:
        if x.kind == "ordinal":                    # few levels -> mean of y at each level of x
            return _group_means(b, y, x, mask, min_n, spec, notes, fdesc)
        return _scatter(b, x, y, mask, spec, notes, fdesc)
    if y.numeric and grp_ok(x):
        return _group_means(b, y, x, mask, min_n, spec, notes, fdesc)
    if x.numeric and grp_ok(y):
        notes.append(f"已改以「{y.name}」分組、比較「{x.name}」")
        return _group_means(b, x, y, mask, min_n, spec, notes, fdesc)
    return _crosstab(b, x, y, mask, min_n, spec, notes, fdesc)


def _distribution(b, spec, min_n, notes, fdesc, mask) -> ChartResult:
    v = _need(b, spec.get("var") or spec.get("x"))
    title = _title_with_filter(spec.get("title") or f"{v.name} 的分布", fdesc)
    fig = go.Figure()
    if v.kind == "composite":
        s = b.num[v.name][mask].dropna()
        if len(s) < 5:
            raise PlanError("有效樣本太少")
        fig.add_trace(go.Histogram(
            x=s,
            xbins=dict(start=0, end=100, size=10),
            marker_color=ACCENT,
            marker_line_color="#ffffff",
            marker_line_width=1,
            hovertemplate="分數：%{x}<br>人數：%{y}<extra></extra>",
        ))
        fig.update_layout(xaxis=dict(range=[0, 100], title=dict(text=f"{v.name}（0–100）")),
                          yaxis=dict(title=dict(text="人數"), showgrid=True, gridcolor="#e8efec"),
                          bargap=0.05)
        st = dict(type="distribution", var=v.name, n=int(len(s)), mean=_f(s.mean(), 1), median=_f(s.median(), 1),
                  sd=_f(s.std(), 1), q25=_f(s.quantile(.25), 1), q75=_f(s.quantile(.75), 1))
        head = f"平均 {s.mean():.1f}、中位數 {s.median():.1f}、標準差 {s.std():.1f}（n={len(s)}）"
        _layout(fig, title, height=320)
    else:
        g = get_groups(b, v, mask, 1, notes, cap=15)
        n_resp = _n_unique(g.s)
        counts = g.s.value_counts()
        order = g.order if g.ordered else list(counts.index)
        vals = [int(counts.get(o, 0)) for o in order]
        pct = [100 * c / n_resp for c in vals]
        st = dict(type="distribution", var=v.name, n=n_resp, multi_select=g.overlap,
                  items=[dict(label=o, count=c, percent=_f(p, 1)) for o, c, p in zip(order, vals, pct)])
        head = f"共 {n_resp} 人作答；最多的是「{short(order[int(np.argmax(vals))], 20)}」（{max(pct):.0f}%）"
        text = [f"{c}（{p:.0f}%）" for c, p in zip(vals, pct)]
        if g.ordered:
            fig.add_trace(go.Bar(
                x=[wrap(o, 9) for o in order], y=vals, text=text, textposition="outside",
                marker_color=ACCENT, marker_line_color="#ffffff", marker_line_width=1,
                textfont=dict(color=INK, size=11),
            ))
            fig.update_layout(yaxis=dict(title=dict(text="人數"), showgrid=True, gridcolor="#e8efec"))
            _layout(fig, title, height=340)
        else:
            fig.add_trace(go.Bar(y=[wrap(o, 18) for o in order], x=vals, orientation="h", text=text, textposition="outside",
                                 marker_color=ACCENT, marker_line_color="#ffffff", marker_line_width=1,
                                 textfont=dict(color=INK, size=11)))
            fig.update_layout(
                yaxis=dict(autorange="reversed", automargin=True),
                xaxis=dict(title=dict(text="人數"), showgrid=True, gridcolor="#e8efec"),
            )
            _layout(fig, title, height=max(290, 86 + 34 * len(order)))
        if g.overlap:
            notes.append("複選題：百分比是「選了該項的人數 ÷ 作答人數」，加總會超過 100%")
    fig.update_layout(showlegend=False)
    return ChartResult(spec, title, fig, st, head, notes)


def _heatmap(b, spec, min_n, notes, fdesc, mask) -> ChartResult:
    names = spec.get("vars") or [v.name for v in b.vars.values() if v.kind == "composite"]
    vs: list[Var] = []
    for nm in names:
        v = b.resolve(nm)
        if v is None or not v.numeric:
            notes.append(f"「{nm}」找不到或不是可計算相關的數值變數，已略過")
        elif v.name not in [u.name for u in vs]:
            vs.append(v)
    vs = vs[:16]
    if len(vs) < 2:
        raise PlanError("相關矩陣至少需要 2 個數值變數")
    k = len(vs)
    R = np.full((k, k), np.nan)
    N = np.zeros((k, k), dtype=int)
    P = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            z = pd.DataFrame({"a": b.num[vs[i].name][mask], "c": b.num[vs[j].name][mask]}).dropna()
            N[i, j] = len(z)
            if i == j:
                R[i, j] = 1.0
            elif len(z) >= MIN_CORR_N and z["a"].nunique() > 1 and z["c"].nunique() > 1:
                R[i, j], P[i, j] = sps.spearmanr(z["a"], z["c"])
    pairs = [dict(a=vs[i].name, b=vs[j].name, rho=_f(R[i, j]), p=_f(P[i, j], 4), n=int(N[i, j]))
             for i in range(k) for j in range(i + 1, k) if not np.isnan(R[i, j])]
    pairs.sort(key=lambda d: -abs(d["rho"]))
    st = dict(type="correlation_matrix", method="spearman", variables=[v.name for v in vs], top_pairs=pairs[:10],
              n_range=[int(N[np.triu_indices(k, 1)].min()), int(N[np.triu_indices(k, 1)].max())])
    labels = [wrap(short(v.name, 14), 7) for v in vs]
    text = [["" if np.isnan(R[i, j]) else f"{R[i, j]:.2f}" for j in range(k)] for i in range(k)]
    fig = go.Figure(go.Heatmap(z=R, x=labels, y=labels, zmin=-1, zmax=1, zmid=0,
                               colorscale=[[0, "#c77f91"], [0.5, "#f7f5f2"], [1, "#6eaaa0"]], text=text,
                               colorbar=dict(title=dict(text="Spearman ρ"), thickness=14, outlinewidth=0),
                               texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>ρ=%{z:.2f}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", automargin=True), xaxis=dict(side="bottom"))
    title = _title_with_filter(spec.get("title") or "變數間的相關矩陣（Spearman）", fdesc)
    _layout(fig, title, height=max(350, 100 + 56 * k))
    top = pairs[0] if pairs else None
    head = (f"最強的一組是「{short(top['a'], 14)}」與「{short(top['b'], 14)}」，ρ={top['rho']}（n={top['n']}）" if top else "")
    notes.append("粉紅色＝負相關、青綠色＝正相關；僅供探索，多組比較未校正顯著性")
    return ChartResult(spec, title, fig, st, head, notes)


def _trend(b, spec, min_n, notes, fdesc, mask) -> ChartResult:
    if len(b.years) < 2:
        raise PlanError("目前只載入了一個年度的資料，無法看年度變化。請在側邊欄加入其他年度的 CSV。")
    v = _need(b, spec.get("var") or spec.get("y") or spec.get("x"))
    by = b.resolve(spec["by"]) if spec.get("by") else None
    yr = b.lab["年度"][mask]
    fig = go.Figure()
    title = _title_with_filter(spec.get("title") or f"{v.name} 的年度變化", fdesc)
    if v.numeric:
        if by is not None:
            g = get_groups(b, by, mask, min_n, notes, cap=6)
            d = g.s.rename("g").to_frame()
        else:
            d = pd.Series("全體", index=yr.dropna().index, name="g").to_frame()
        d = d.join(yr.rename("yr")).join(b.num[v.name][mask].rename("v")).dropna()
        table = []
        for gname, sub in d.groupby("g"):
            xs, ys, es, ns = [], [], [], []
            for y in b.years:
                x = sub.loc[sub["yr"] == y, "v"]
                if len(x) >= min_n:
                    xs.append(y); ys.append(float(x.mean()))
                    es.append(1.96 * x.std(ddof=1) / math.sqrt(len(x))); ns.append(len(x))
                    table.append(dict(group=gname, year=y, mean=_f(x.mean()), ci95=_f(es[-1]), n=len(x)))
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=short(gname, 16),
                    marker=dict(size=8, line=dict(width=1, color="#ffffff")),
                    line=dict(width=2.5),
                    error_y=dict(type="data", array=es, visible=True, color="#9aa9a5", thickness=1),
                ))
        if not table:
            raise PlanError(f"各年度的有效樣本都不足 {min_n} 人")
        fig.update_layout(
            yaxis=_val_axis(v) | dict(showgrid=True, gridcolor="#e8efec"),
            xaxis=dict(type="category", title=dict(text="年度")),
            legend=dict(orientation="h", y=-0.2, x=0),
        )
        st = dict(type="trend", var=v.name, by=by.name if by else None, table=table, n_total=int(d.index.nunique()))
        first, last = b.years[0], b.years[-1]
        a = [t["mean"] for t in table if t["year"] == first and t["group"] == table[0]["group"]]
        z = [t["mean"] for t in table if t["year"] == last and t["group"] == table[0]["group"]]
        head = f"{first}→{last}：{a[0]:.1f}→{z[0]:.1f}" if a and z else "各年度平均如圖"
        notes.append("線上的誤差線為 95% 信賴區間；不同年度受訪者並非同一批人，也未必代表同樣的母體")
    else:
        g = get_groups(b, v, mask, 1, notes, cap=999)   # full set first: denominators must include everyone
        d = g.s.rename("g").to_frame().join(yr.rename("yr")).dropna()
        shown = g.order[:6]
        if len(g.order) > 6:
            notes.append("選項過多，只顯示整體最多人選的前 6 項")
        denom = {y: d.loc[d["yr"] == y].index.nunique() for y in b.years}
        table = []
        for gname in shown:
            ys = []
            for y in b.years:
                nd = int(denom.get(y, 0))
                if nd >= min_n:
                    share = 100 * d[(d["yr"] == y) & (d["g"] == gname)].index.nunique() / nd
                    ys.append((y, share)); table.append(dict(option=gname, year=y, percent=_f(share, 1), n=nd))
            if ys:
                fig.add_trace(go.Scatter(
                    x=[t[0] for t in ys],
                    y=[t[1] for t in ys],
                    mode="lines+markers",
                    name=short(gname, 16),
                    marker=dict(size=8, line=dict(width=1, color="#ffffff")),
                    line=dict(width=2.5),
                ))
        if not table:
            raise PlanError(f"各年度的有效樣本都不足 {min_n} 人")
        fig.update_layout(
            yaxis=dict(title=dict(text="佔該年度作答者的比例（%）"), range=[0, 100],
                       showgrid=True, gridcolor="#e8efec"),
            xaxis=dict(type="category", title=dict(text="年度")),
            legend=dict(orientation="h", y=-0.25, x=0),
        )
        st = dict(type="trend", var=v.name, table=table)
        head = "各選項的年度比例如圖"
    _layout(fig, title, height=360)
    return ChartResult(spec, title, fig, st, head, notes)


# --------------------------------------------------------------------------- entry points
def run_one(b: Bundle, spec: dict, min_n: int = 5) -> ChartResult:
    t = spec.get("type")
    if t not in CHART_TYPES:
        raise PlanError(f"不支援的圖表類型：{t}")
    notes: list[str] = []
    mask, fdesc = build_mask(b, spec.get("filter"), notes)
    if int(mask.sum()) < MIN_CORR_N:
        raise PlanError(f"套用篩選後只剩 {int(mask.sum())} 人，無法分析")
    fn = dict(relationship=_relationship, distribution=_distribution, heatmap=_heatmap, trend=_trend)[t]
    res = fn(b, spec, min_n, notes, fdesc, mask)
    res.stats["filter"] = fdesc or None
    return res


def run_plan(b: Bundle, plan: dict, min_n: int = 5, max_charts: int = 3) -> list[ChartResult]:
    out = []
    for spec in (plan.get("charts") or [])[:max_charts]:
        try:
            out.append(run_one(b, spec, min_n))
        except PlanError as e:
            out.append(ChartResult(spec, spec.get("title") or "（無法產生圖表）", error=str(e)))
        except Exception as e:  # never crash the chat
            out.append(ChartResult(spec, spec.get("title") or "（無法產生圖表）", error=f"{type(e).__name__}: {e}"))
    return out
