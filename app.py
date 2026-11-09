"""app.py — Streamlit chat UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import os
import re
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

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="助人者困境調查・對話式分析", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- data loading
def guess_year(name: str) -> str:
    m = re.search(r"(20\d{2})", name)
    return m.group(1) if m else Path(name).stem[:12]


@st.cache_resource(show_spinner="讀取並整理資料中…")
def get_bundle(sources: tuple):
    return load_bundle(list(sources))


def collect_sources():
    """Uploaded files win; otherwise every CSV in ./data."""
    files = []   # (display name, bytes)
    up = st.sidebar.file_uploader("上傳問卷 CSV（可多個年度）", type="csv", accept_multiple_files=True)
    if up:
        files = [(f.name, f.getvalue()) for f in up]
    elif DATA_DIR.exists():
        files = [(p.name, p.read_bytes()) for p in sorted(DATA_DIR.glob("*.csv"))]
    sources = []
    if files:
        st.sidebar.caption("每個檔案的年度標籤（欄位格式需相同）：")
    for name, raw in files:
        key = "year_" + hashlib.md5(name.encode()).hexdigest()[:8]
        label = st.sidebar.text_input(name[:30], value=guess_year(name), key=key)
        sources.append((label.strip() or guess_year(name), raw))
    return tuple(sources)


# --------------------------------------------------------------------------- rendering
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


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("設定")
api_key = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password",
                                help="只存在這個瀏覽器工作階段；也可放在 .env 的 GROQ_API_KEY")
model = st.sidebar.text_input("模型", value=DEFAULT_MODEL, help="可改成 Groq 提供的其他模型名稱")
min_n = st.sidebar.slider("每組最少人數", 3, 20, 5, help="人數少於這個數字的組別不會被比較，避免被極少數人左右")
st.sidebar.divider()
sources = collect_sources()

st.title("📊 助人者困境調查・對話式分析")

if not sources:
    st.info("請在側邊欄上傳問卷 CSV，或把檔案放進專案的 `data/` 資料夾（檔名含年份，例如 `survey_2026.csv`）。")
    st.stop()

try:
    bundle = get_bundle(sources)
except Exception as e:
    st.error(f"讀取 CSV 失敗：{e}")
    st.stop()

st.caption(f"共 {bundle.n} 份回覆｜年度：" + "、".join(f"{y}（{bundle.year_counts[y]}）" for y in bundle.years)
           + f"｜可分析變數 {len(bundle.vars)} 個")

with st.sidebar.expander("資料檢查"):
    if bundle.warnings:
        for w in bundle.warnings:
            st.warning(w)
    else:
        st.success("欄位格式一致，未發現問題")
    st.caption("已排除（個資／開放式文字）：" + "、".join(c[:14] for c in bundle.excluded))
with st.sidebar.expander("變數清單（可問的欄位）"):
    st.dataframe(bundle.codebook_df(), hide_index=True)

st.sidebar.divider()
st.sidebar.markdown("**範例問題**")
examples = [
    "工作要求困擾越高的人，個人倦怠也越高嗎？",
    "不同機構規模的離職意圖有差異嗎？",
    "第一線助人者與行政人員的次級創傷壓力比較",
    "倦怠、工作投入、工作條件與離職意圖之間的相關",
    "主管支持程度和離職念頭有關嗎？",
]
if len(bundle.years) > 1:
    examples.insert(0, "各年度的個人倦怠指數有什麼變化？")
for q in examples:
    if st.sidebar.button(q, key="ex_" + q):
        st.session_state["pending"] = q
if st.sidebar.button("🗑️ 清除對話"):
    st.session_state["messages"], st.session_state["history"] = [], []
    st.rerun()

# --------------------------------------------------------------------------- chat
st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(
            "你好！你可以用日常語言問我這份問卷的資料，例如：\n\n"
            "- 「工作量過大和離職念頭有關嗎？」\n- 「不同年資的人，倦怠程度有差別嗎？」\n"
            "- 「只看第一線助人者，支持制度和倦怠的關係」\n\n"
            "我會請 AI 理解問題，**由程式計算並畫圖**，再附上簡短說明。也可以直接點左邊的範例問題。")

for i, m in enumerate(st.session_state["messages"]):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_answer(m, i)

prompt = st.chat_input("輸入你的問題，例如：薪資困擾和離職意圖有關嗎？")
if not prompt and st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            client = GroqClient(api_key, model)
            with st.spinner("理解問題、計算並繪圖中…"):
                ans = answer_question(client, bundle, prompt, st.session_state["history"], min_n=min_n)
            msg = dict(role="assistant", error=ans.error, note=ans.note, warning=ans.warning, results=ans.results,
                       explanations=ans.explanations, summary=ans.summary)
            if not ans.error:
                st.session_state["history"].append({"question": prompt, "plan": ans.plan})
        except LLMError as e:
            msg = dict(role="assistant", error=str(e), results=[], explanations=[])
        st.session_state["messages"].append(msg)
        render_answer(msg, len(st.session_state["messages"]) - 1)
