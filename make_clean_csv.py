"""make_clean_csv.py — create a de-identified copy of a raw survey export for deployment.

    python make_clean_csv.py raw_export.csv data/survey_2026.csv

Removes: timestamps, e-mails, consent / lottery / follow-up columns, the open-ended free-text answer.
Cleans:  the county column (people typed names / e-mails into it) -> non-county entries become "其他".
Collapses answers given by fewer than 5 people (county, org type, role, work style) into "其他".
Shuffles row order so position can't be linked back to submission time.
Note: the app already ignores those columns, so this only reduces what sits on the server / in storage.
"""
import re
import sys

import pandas as pd

from data import EXCLUDE_RE

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
drop = [c for c in df.columns if EXCLUDE_RE.search(re.sub(r"\.\d+$", "", c.strip()))]
df = df.drop(columns=drop)
for c in df.columns:
    if "縣市" in c:
        df[c] = df[c].where(df[c].str.contains(r"[縣市]$", na=False), "其他").where(df[c].notna(), None)
# collapse rare answers in single-choice columns that could single someone out (county, org type, role, work style)
RARE_COLS = re.compile(r"縣市|機構的性質|在機構中的角色|主要的工作型態")
for c in df.columns:
    if RARE_COLS.search(c):
        vc = df[c].value_counts()
        df[c] = df[c].where(~df[c].isin(vc[vc < 5].index), "其他")
df = df.sample(frac=1, random_state=None).reset_index(drop=True)
df.to_csv(dst, index=False, encoding="utf-8-sig")
print(f"dropped {len(drop)} columns: {[c[:16] for c in drop]}\nwrote {len(df)} rows x {df.shape[1]} cols -> {dst}")