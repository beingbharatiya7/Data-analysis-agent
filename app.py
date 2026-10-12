"""
app.py - Streamlit web interface for the data analysis agent.

Run locally:   streamlit run app.py
On Streamlit Cloud: set "Main file path" to app.py and add a secret
    GEMINI_API_KEY = "your-key"
"""

import os
import tempfile

import streamlit as st

import tools
from agent import DataAgent

SAMPLES = {
    "Sample: store sales": "sample_data/store_sales.csv",
    "Sample: student performance": "sample_data/student_performance.csv",
}
UPLOAD = "Upload my own file"

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊", layout="wide")
st.title("📊 Data Analysis Agent")
st.caption(
    "Ask questions about any CSV or Excel file in plain English. The agent answers by "
    "running real pandas tools, and it remembers useful facts between sessions."
)


def find_api_key():
    """Look for the key in Streamlit secrets, then environment variables.
    The name is matched ignoring upper/lower case, so GEMINI_API_KEY and
    Gemini_API_Key both work."""
    try:
        for name in st.secrets:
            if str(name).lower() == "gemini_api_key":
                return str(st.secrets[name]).strip()
    except Exception:  # no secrets configured
        pass
    for name, value in os.environ.items():
        if name.lower() == "gemini_api_key" and value.strip():
            return value.strip()
    return None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Data")
    choice = st.radio("Choose a dataset", list(SAMPLES) + [UPLOAD])
    dataset_path = None
    if choice == UPLOAD:
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            dataset_path = os.path.join(tempfile.gettempdir(), os.path.basename(uploaded.name))
            with open(dataset_path, "wb") as f:
                f.write(uploaded.getbuffer())
    else:
        dataset_path = SAMPLES[choice]

key = find_api_key()
if not key:
    with st.sidebar:
        key = st.text_input("Gemini API key", type="password",
                            help="Free key from Google AI Studio. Not stored.") or None
if not key:
    st.info("Add your Gemini API key (app secret `GEMINI_API_KEY`, or the box in the sidebar) to start.")
    st.stop()

# ------------------------------------------------------- agent + dataset
if st.session_state.get("key") != key or "agent" not in st.session_state:
    st.session_state.agent = DataAgent(api_key=key)
    st.session_state.key = key
    st.session_state.messages = []
agent = st.session_state.agent

if dataset_path is None:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

# The tools keep the loaded table in one shared place, so make sure it is OUR
# dataset on every rerun (matters if several people use the app at once).
if st.session_state.get("dataset_path") != dataset_path:
    st.session_state.dataset_path = dataset_path
    st.session_state.messages = []
    agent.load(dataset_path)
elif tools.STATE.name != os.path.basename(dataset_path):
    tools.load_dataset(dataset_path)  # reload the table, keep the conversation

with st.expander(f"Preview of {tools.STATE.name} "
                 f"({len(tools.STATE.df)} rows, {len(tools.STATE.df.columns)} columns)"):
    st.dataframe(tools.STATE.df.head(10))

# ---------------------------------------------------------- memory panel
with st.sidebar:
    st.header("2. Memory")
    st.caption("Short useful facts the agent saved (definitions, preferences, filters). "
               "On a shared demo, everyone using this app sees the same memory.")
    notes = agent.memory.show(tools.STATE.name)
    if not notes:
        st.write("_Nothing remembered yet. Try:_ `Remember that 'big orders' means total_sales above 1000.`")
    for i, note in enumerate(notes):
        st.markdown(f"**{note['key']}** ({note['kind']})  \n{note['value']}")
        if st.button("Forget", key=f"forget_{i}_{note['key']}"):
            agent.memory.forget(note["key"], note["dataset"])
            st.rerun()
    if notes and st.button("Clear all memory"):
        agent.memory.clear()
        st.rerun()


# ----------------------------------------------------------------- chat
def show_extras(tool_calls, charts):
    if tool_calls:
        with st.expander(f"🔧 Tools used ({len(tool_calls)})"):
            for t in tool_calls:
                st.code(f"{t['tool']}({t['args']})", language="python")
    for path in charts:
        if os.path.exists(path):
            st.image(path)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_extras(m.get("tools", []), m.get("charts", []))

question = st.chat_input("Ask a question about your data...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analysing... (the agent may wait 30 s if the free plan is busy)"):
                answer = agent.ask(question)
            calls = [{"tool": t["tool"], "args": t["args"]} for t in agent.trace]
            charts = [t["result"]["saved_to"] for t in agent.trace
                      if t["tool"] == "plot_chart" and isinstance(t["result"], dict)
                      and "saved_to" in t["result"]]
        except Exception as e:  # noqa: BLE001
            answer = (f"Sorry, something went wrong: `{str(e)[:300]}`\n\n"
                      "If it mentions 429 or 503, the free Gemini plan is busy. Wait a minute and try again.")
            calls, charts = [], []
        st.markdown(answer)
        show_extras(calls, charts)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": calls, "charts": charts}
    )
    st.rerun()
