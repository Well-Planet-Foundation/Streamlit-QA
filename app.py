"""app.py — Streamlit chat demo.   Run:  streamlit run app.py
Uses one fixed, anonymized CSV (data/survey_2026.csv). Create it with make_clean_csv.py."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data import load_bundle
from llm import DEFAULT_MODEL, GroqClient, LLMError
from pipeline import answer_question

DATA_FILE = Path(__file__).parent / "data" / "survey_2026.csv"
YEAR_LABEL = "2026"
MIN_N = 3                       # smallest group that will be compared / shown

st.set_page_config(page_title="助人者困境調查・對話式分析", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="載入資料中…")
def get_bundle():
    return load_bundle([(YEAR_LABEL, DATA_FILE.read_bytes())])


def render_answer(msg: dict, idx: int):
    if msg.get("error"):
        st.error(msg["error"])
        return
    if msg.get("note"):
        st.markdown(msg["note"])
    if msg.get("warning"):
        st.warning(msg["warning"])
    for i, r in enumerate(msg["results"]):
        if r.error:
            st.warning(f"**{r.title}**\n\n{r.error}")
            continue
        st.markdown(f"##### {r.title}")
        st.plotly_chart(r.fig, key=f"chart_{idx}_{i}")
        if r.headline:
            st.caption("📐 程式計算：" + r.headline)
        ex = msg["explanations"][i] if i < len(msg["explanations"]) else ""
        if ex:
            st.markdown(ex)
        if r.notes:
            st.caption("ℹ️ " + "　·　".join(dict.fromkeys(r.notes)))
        with st.expander("計算細節（透明度）"):
            st.json({"圖表設定": r.spec, "統計結果": r.stats})
    if msg.get("summary"):
        st.info(msg["summary"])


if not DATA_FILE.exists():
    st.error(f"找不到資料檔：{DATA_FILE.relative_to(DATA_FILE.parent.parent)}")
    st.stop()
try:
    bundle = get_bundle()
except Exception as e:
    st.error(f"讀取 CSV 失敗：{e}")
    st.stop()

api_key = os.getenv("GROQ_API_KEY", "")
model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "你好！你可以用日常語言問我這份問卷的資料，例如：\n\n"
            "- 「工作量過大和離職念頭有關嗎？」\n- 「不同年資的人，倦怠程度有差別嗎？」\n"
            "- 「只看第一線助人者，支持制度和倦怠的關係」\n\n"
            "我會請 AI 理解問題，**由程式計算並畫圖**，再附上簡短說明。")

for i, m in enumerate(st.session_state["messages"]):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_answer(m, i)

if prompt := st.chat_input("輸入你的問題，例如：薪資困擾和離職意圖有關嗎？"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            client = GroqClient(api_key, model)
            with st.spinner("理解問題、計算並繪圖中…"):
                ans = answer_question(client, bundle, prompt, st.session_state["history"], min_n=MIN_N)
            msg = dict(role="assistant", error=ans.error, note=ans.note, warning=ans.warning,
                       results=ans.results, explanations=ans.explanations, summary=ans.summary)
            if not ans.error:
                st.session_state["history"].append({"question": prompt, "plan": ans.plan})
        except LLMError as e:
            msg = dict(role="assistant", error=str(e), results=[], explanations=[])
        st.session_state["messages"].append(msg)
        render_answer(msg, len(st.session_state["messages"]) - 1)