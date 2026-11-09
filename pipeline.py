"""pipeline.py — question -> plan (LLM) -> numbers & charts (Python) -> explanation (LLM)."""
from __future__ import annotations

from dataclasses import dataclass, field

from analysis import ChartResult, run_plan
from data import Bundle
from llm import GroqClient, LLMError, explain, plan_question


@dataclass
class Answer:
    question: str
    plan: dict = field(default_factory=dict)
    results: list[ChartResult] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    summary: str = ""
    note: str = ""
    error: str = ""            # fatal problem (e.g. API key) — nothing could be produced
    warning: str = ""          # non-fatal (e.g. explanation unavailable, charts still shown)


def answer_question(client: GroqClient, bundle: Bundle, question: str, history: list[dict], min_n: int = 5) -> Answer:
    ans = Answer(question=question)
    try:
        ans.plan = plan_question(client, bundle.codebook_text(), history, question)
    except LLMError as e:
        ans.error = str(e)
        return ans
    except Exception as e:  # malformed JSON etc.
        ans.error = f"無法理解模型的回覆（{type(e).__name__}）。請換個說法再問一次。"
        return ans
    ans.note = str(ans.plan.get("assistant_note") or "")
    ans.results = run_plan(bundle, ans.plan, min_n=min_n)
    ok = [r for r in ans.results if not r.error]
    if ok:
        try:
            ex = explain(client, question, ok, bundle)
            it = iter(ex["explanations"])
            ans.explanations = [next(it) if not r.error else "" for r in ans.results]
            ans.summary = ex["summary"]
        except Exception as e:
            ans.warning = f"圖表已完成，但文字解讀暫時無法產生（{e}）。下方仍列出由程式計算的數字。"
            ans.explanations = [""] * len(ans.results)
    else:
        ans.explanations = [""] * len(ans.results)
    return ans
