"""app.py — Streamlit chat demo.   Run:  streamlit run app.py
Uses one fixed, anonymized CSV (data/survey_2026.csv). Create it with make_clean_csv.py."""
from __future__ import annotations

import os
from html import escape
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

st.markdown(
    """
    <style>
    :root {
        --ink: #3d4146;
        --muted: #70777b;
        --accent: #638f87;
        --line: #e8e9e7;
        --surface: rgba(255, 255, 255, 0.78);
        --canvas: #fbfaf8;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 12%, rgba(245, 207, 204, 0.26), transparent 32rem),
            radial-gradient(ellipse at 92% 18%, rgba(188, 224, 216, 0.28), transparent 34rem),
            var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(251, 250, 248, 0.88);
    }

    .main .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 1.25rem;
    }

    [data-testid="stMainBlockContainer"] {
        padding-bottom: 1.25rem !important;
    }

    [data-testid="stBottom"] {
        background: transparent;
    }

    [data-testid="stBottomBlockContainer"] {
        max-width: 1160px;
        padding: 0 1rem 0.75rem !important;
    }

    .app-hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.5rem;
        padding: 2rem 2.25rem;
        border: 1px solid #eee8e5;
        border-radius: 20px;
        background: linear-gradient(112deg, #fff4f1 0%, #fffafa 46%, #edf8f5 100%);
        box-shadow: 0 8px 24px rgba(83, 91, 87, 0.045);
    }

    .app-hero::after {
        position: absolute;
        top: -90px;
        right: -25px;
        width: 250px;
        height: 250px;
        border: 1px solid rgba(99, 143, 135, 0.17);
        border-radius: 50%;
        box-shadow: 0 0 0 25px rgba(229, 167, 174, 0.08),
                    0 0 0 52px rgba(126, 179, 169, 0.07);
        content: "";
        pointer-events: none;
    }

    .app-hero .eyebrow {
        margin: 0 0 0.55rem;
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }

    .app-hero h1 {
        margin: 0;
        color: var(--ink);
        font-size: clamp(1.8rem, 4vw, 2.45rem);
        font-weight: 750;
        letter-spacing: 0.015em;
        line-height: 1.25;
    }

    .app-hero .description {
        max-width: 680px;
        margin: 0.7rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.75;
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 0.85rem;
        padding: 1.2rem 1.35rem;
        border: 1px solid rgba(226, 225, 221, 0.72);
        border-radius: 18px;
        background: var(--surface);
        box-shadow: 0 4px 14px rgba(83, 91, 87, 0.025);
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        max-width: 72ch;
        line-height: 1.75;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        margin: 0.55rem 0 0.85rem;
        line-height: 1.75;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        margin: 0.3rem 0;
        line-height: 1.7;
    }

    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
        line-height: 1.75;
    }

    [data-testid="stChatInput"] {
        border: none !important;
        border-radius: 12px;
        background: transparent;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] > div {
        border: 1px solid rgba(226, 225, 221, 0.72) !important;
        border-radius: 18px !important;
        background: var(--surface) !important;
        box-shadow: 0 4px 14px rgba(83, 91, 87, 0.025) !important;
    }

    [data-testid="stChatInputTextArea"] {
        background: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInputSubmitButton"] {
        background: #e6f1ed !important;
        color: var(--accent) !important;
    }

    [data-testid="stChatInputSubmitButton"]:hover {
        background: #d5e9e2 !important;
    }

    [data-testid="stPlotlyChart"] {
        max-width: 840px;
        margin: 0 auto;
        overflow: hidden;
        border-radius: 12px;
        background: transparent;
    }

    .chart-explanation {
        max-width: 72ch;
        color: #515a5d;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        max-width: 900px;
        margin: 0 auto 1rem;
        border-color: var(--line) !important;
        border-radius: 18px !important;
    }

    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(226, 225, 221, 0.72) !important;
        background: var(--surface);
        box-shadow: 0 4px 14px rgba(83, 91, 87, 0.025);
    }

    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: 12px;
        background: #fffdfc;
    }

    h1, h2, h3, h4 {
        color: var(--ink);
        letter-spacing: -0.015em;
    }

    @media (max-width: 640px) {
        .main .block-container {
            padding: 1rem 1rem 5rem;
        }

        .app-hero {
            margin-bottom: 1.25rem;
            padding: 1.5rem 1.25rem;
            border-radius: 16px;
        }

        [data-testid="stChatMessage"] {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        with st.container(border=True):
            st.plotly_chart(
                r.fig,
                key=f"chart_{idx}_{i}",
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False, "responsive": True},
            )
            if r.headline:
                st.caption("📐 程式計算：" + r.headline)
            ex = msg["explanations"][i] if i < len(msg["explanations"]) else ""
            if ex:
                st.markdown(
                    f'<div class="chart-explanation">{escape(ex)}</div>',
                    unsafe_allow_html=True,
                )
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

st.markdown(
    """
    <header class="app-hero">
        <p class="eyebrow">好好星球基金會　·　SURVEY INSIGHTS 2026</p>
        <h1>助人者困境調查</h1>
        <p class="description">用日常語言探索問卷資料。提出問題，查看程式計算的圖表、統計結果與解讀。</p>
    </header>
    """,
    unsafe_allow_html=True,
)

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