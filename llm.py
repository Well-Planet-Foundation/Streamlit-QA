"""llm.py — Groq client + the two prompts (planner and explainer).

Privacy: only the codebook (variable names / option labels) and aggregated statistics are sent
to the LLM. Raw rows, e-mails and free-text answers never leave this machine.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class LLMError(Exception):
    pass


class GroqClient:
    def __init__(self, api_key: str, model: str | None = None, timeout: int = 60):
        if not api_key:
            raise LLMError("尚未設定 Groq API Key（請在側邊欄輸入，或設定環境變數 GROQ_API_KEY）。")
        self.key, self.model, self.timeout = api_key.strip(), (model or DEFAULT_MODEL).strip(), timeout

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(4):
            try:
                r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last = f"網路錯誤：{e}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("retry-after", 2 * (attempt + 1)) or 2)
                last = f"HTTP {r.status_code}（{_err(r)}）"
                if wait > 20:   # daily/minute quota exhausted — don't hang the UI
                    break
                time.sleep(wait)
                continue
            if r.status_code == 401:
                raise LLMError("Groq API Key 無效或已被撤銷（401）。")
            raise LLMError(f"Groq 回應錯誤 HTTP {r.status_code}：{_err(r)}")
        raise LLMError(f"Groq 暫時無法使用：{last}")

    def chat(self, messages, *, json_mode=False, temperature=0.1, max_tokens=1200) -> str:
        payload = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(payload)
        except LLMError as e:
            if json_mode and "HTTP 400" in str(e):      # model without JSON mode -> retry as plain text
                payload.pop("response_format")
                data = self._post(payload)
            else:
                raise
        return data["choices"][0]["message"]["content"] or ""

    def chat_json(self, messages, **kw) -> dict:
        txt = self.chat(messages, json_mode=True, **kw)
        try:
            return parse_json(txt)
        except ValueError:
            fix = messages + [{"role": "assistant", "content": txt},
                              {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."}]
            return parse_json(self.chat(fix, json_mode=True, **kw))


def _err(r) -> str:
    try:
        return r.json()["error"]["message"][:200]
    except Exception:
        return r.text[:200]


def parse_json(txt: str) -> dict:
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise ValueError("no JSON found")
        return json.loads(m.group(0))


# --------------------------------------------------------------------------- planner
PLANNER_SYSTEM = """You are the query planner of a survey-analysis chatbot for a Taiwanese helping-professional well-being survey.
You NEVER see raw data and you NEVER compute anything. You turn the user's question into a small JSON plan; Python then runs the statistics and draws the charts.

Return ONLY one JSON object:
{"charts": [ ... up to 3 chart specs ... ], "assistant_note": "<one short sentence in the user's language, or empty>"}

Chart spec types:
1. {"type":"relationship","x":V,"y":V,"title":T}  - the workhorse. x = explanatory / grouping variable, y = outcome. Python picks the right chart:
   numeric vs numeric -> scatter + Spearman; category vs numeric -> group means + Kruskal-Wallis; category vs category -> 100% stacked bars + chi-square.
2. {"type":"distribution","var":V,"title":T}  - how answers are spread for one variable.
3. {"type":"heatmap","vars":[V,...up to 12 numeric variables],"title":T} - correlation matrix. Omit "vars" to use all composite indices.
4. {"type":"trend","var":V,"by":V (optional),"title":T} - change across years. Only valid when more than one year is loaded.
Any chart may also have "filter": {"<variable>": ["<option text>", ...]} to restrict to a subgroup (e.g. only front-line helpers). Filters only work on categorical/ordinal/multi-select variables.

Rules:
- Variable names must be copied from the codebook below: full names, or the alias for item batteries (e.g. 困擾3). Option texts in the codebook are truncated with "…"; copy the visible part.
- Prefer composite indices (…指數) for broad concepts such as burnout, engagement, intention to leave, job demands, support. Use single items only when the user names that specific item.
- "Does A affect / relate to B" -> relationship with x=A, y=B. "Compare groups" -> relationship with x=group variable, y=numeric outcome.
- For a broad question ("what is related to burnout?") use one heatmap and/or 1-2 relationship charts. Do not exceed 3 charts.
- Many questions only apply to some respondents (see n= in the codebook; e.g. independent workers n≈39). That is fine, Python reports n.
- If the question cannot be answered from this survey (or asks for causes, predictions, individual people), return "charts": [] and explain in assistant_note and suggest what CAN be asked.
- If the question is a follow-up ("what about only front-line staff?", "now by gender"), reuse the previous plan and modify it.
- Titles: short, in the user's language (default 繁體中文).

Codebook:
"""


def planner_messages(codebook: str, history: list[dict], question: str) -> list[dict]:
    msgs = [{"role": "system", "content": PLANNER_SYSTEM + codebook}]
    for turn in history[-3:]:      # previous (question, plan) pairs allow follow-ups
        msgs.append({"role": "user", "content": turn["question"]})
        msgs.append({"role": "assistant", "content": json.dumps(turn["plan"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": question})
    return msgs


def plan_question(client: GroqClient, codebook: str, history: list[dict], question: str) -> dict:
    plan = client.chat_json(planner_messages(codebook, history, question), temperature=0.0, max_tokens=900)
    if not isinstance(plan, dict):
        raise LLMError("模型回傳的格式不正確。")
    plan.setdefault("charts", [])
    if not isinstance(plan["charts"], list):
        plan["charts"] = []
    plan["charts"] = [c for c in plan["charts"] if isinstance(c, dict)]
    return plan


# --------------------------------------------------------------------------- explainer
EXPLAINER_SYSTEM = """You write short explanations of statistical results for NGO staff who are not statisticians.
You receive the user's question and, for each chart, the exact numbers Python computed. Use ONLY those numbers.

Return ONLY JSON: {"explanations": ["<text for chart 1>", ...], "summary": "<1-2 sentence overall takeaway, or empty if only one chart>"}

Style rules:
- Same language as the user's question (default 繁體中文). Plain words, 2-4 sentences per chart, at most ~110 words each.
- Say what the chart shows, the direction and size of the pattern, and the sample size n. For ρ: <0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, >0.5 strong.
- Mention a p-value only briefly; if p >= 0.05 say the difference/association is not statistically distinguishable from chance in this sample.
- Never claim causation. Say "associated with" / 「相關」. Do not recommend interventions beyond a hedged one-liner.
- Remember this is a self-selected online survey (not a random sample) and self-reported; mention the limitation only when it matters (small n, tiny groups, p-values from many comparisons).
- Respect index direction (see var_notes): e.g. for 工作條件與支持指數 higher is better; for burnout indices higher is worse.
- If notes say groups were excluded or tests are unreliable, mention that in one clause.
- Do not invent numbers, group names or findings that are not in the data."""


def _compact(obj, nd=2):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _compact(v, nd) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_compact(v, nd) for v in obj]
    return obj


def explain(client: GroqClient, question: str, results, bundle) -> dict:
    items, used = [], {}
    for r in results:
        items.append(dict(title=r.title, headline=r.headline, notes=r.notes, stats=_compact(r.stats)))
        for key in ("x", "y", "var", "value", "by"):
            nm = r.stats.get(key)
            v = bundle.vars.get(nm) if isinstance(nm, str) else None
            if v is not None:
                used[v.name] = v.desc or ("ordinal: " + " < ".join(v.levels))
    payload = json.dumps(dict(question=question, charts=items, var_notes=used), ensure_ascii=False)
    out = client.chat_json([{"role": "system", "content": EXPLAINER_SYSTEM}, {"role": "user", "content": payload}],
                           temperature=0.2, max_tokens=900)
    ex = out.get("explanations", [])
    if not isinstance(ex, list):
        ex = []
    ex = [str(e) for e in ex] + [""] * (len(results) - len(ex))
    return dict(explanations=ex[: len(results)], summary=str(out.get("summary") or ""))
