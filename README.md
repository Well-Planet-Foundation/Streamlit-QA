# Helper Survey Chat — ask questions, get charts

A chat-style Streamlit app for the 「臺灣助人者困境大調查」 CSV (and future years with the same columns).

```
question ──► Groq LLM ──► JSON "plan" ──► Python (pandas/scipy) ──► plotly charts ──┐
 (chat)     (planner)    (validated)     computes every number                      ├─► chat
                                         └─► stats ─► Groq LLM (explainer) ─────────┘
```

**The LLM never computes anything and never sees raw rows.** It only sees the variable list (names + option
labels) and, afterwards, the aggregated statistics Python produced. E-mails, names and free-text answers are
excluded from the data model entirely.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # then put a NEW Groq key in it (or paste it in the sidebar)
mkdir -p data && cp your_survey.csv data/survey_2026.csv    # year is read from the file name
streamlit run app.py
```

Or just upload CSVs in the sidebar.

## Several years

Add one CSV per year (same columns). The year is guessed from the file name and can be edited in the sidebar.
Files are stacked, a `年度` variable appears, and questions like “各年度的個人倦怠指數有什麼變化？” produce trend lines.
Columns missing/added in a later year are reported under **資料檢查**; unknown answer options are reported rather than silently dropped.

## What it understands

| Chart | Typical question | Statistics |
|---|---|---|
| relationship: number × number | 工作要求困擾和個人倦怠有關嗎？ | Spearman ρ, Pearson r, trend line, n |
| relationship: group × number | 不同機構規模的離職意圖？ | group means ± 95% CI, Kruskal-Wallis, ε², Spearman if groups are ordered |
| relationship: group × group | 性別與機構角色的分布？ | 100% stacked bars, χ², Cramér's V (warns if cells are tiny) |
| distribution | 各年資的人數？ | counts / % / mean-median |
| heatmap | 倦怠、投入、離職意圖的相關？ | Spearman matrix, strongest pairs |
| trend | 各年度的倦怠變化？ | mean ± CI per year (optionally by group), or option share per year |

Any chart can be filtered to a subgroup (“只看第一線助人者”). Follow-up questions work (the last 3 plans are remembered).
Every chart has a 「計算細節」 expander showing the exact plan and numbers.

## Composite indices (built in `data.py`, all 0–100)

個人倦怠 (CBI personal), 工作倦怠 (CBI work; last item reverse-scored), 工作投入 (UWES-3), 工作要求困擾 (9 items),
工作條件與支持 (9 items; higher = better), 同情滿足 / 助人倦怠 / 次級創傷壓力 (3 items each, a *grouping we chose by item
content* from the ProQOL-style block — please confirm with your survey designer), 離職意圖 (5 items).
A person needs ≥70% of a scale's items answered to get a score. Edit `COMPOSITES` in `data.py` to change them.

## Things to know

* Survey branches: institution staff, independent workers (n≈39) and decision-makers/HR (n=26) answered different blocks,
  so some variables have small n. n is always shown; groups below the “每組最少人數” slider are dropped.
* Results come from a self-selected online sample: treat as associations, not causes, and not as national estimates.
* p-values are not corrected for multiple comparisons.
* Groq free-tier limits are per minute; the app retries on 429. If you hit limits, pick a different model in the sidebar.
* Rotate any API key that has ever been pasted into a chat or committed to git.

## Files

`app.py` UI · `pipeline.py` orchestration · `llm.py` Groq client + prompts · `analysis.py` stats + charts · `data.py` loading, codebook, indices
