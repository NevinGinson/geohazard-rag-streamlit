import streamlit as st
from typing import List, Optional

from geo_core import init_llm, init_vectorstore, answer_question, DOCS_DIR, CHROMA_DIR

st.set_page_config(page_title="Geohazard RAG", layout="wide")

st.title(" Geohazard Management Assistant ")
st.caption("Local PDFs → Chroma → Groq LLM. Fast answers + optional citations + evidence snippets.")

with st.sidebar:
    st.header("Settings")

    rebuild = st.checkbox("Rebuild index (slow)", value=False, help="Deletes and rebuilds chroma_db from PDFs in docs/")
    strict = st.checkbox("Strict mode (refuse if weak evidence)", value=False)
    cite = st.checkbox("Citations ON (page refs)", value=False)

    mode = st.selectbox("Answer mode", ["default", "fast", "brief", "ops", "deep", "ask"], index=0)

    st.divider()
    st.write("📂 **DOCS_DIR**:", DOCS_DIR)
    st.write("💾 **CHROMA_DIR**:", CHROMA_DIR)

    st.divider()
    st.subheader("PDF Routing")
    routing = st.radio("Use PDFs", ["Auto-pick best PDF(s)", "Force one PDF"], index=0)

# Cached init (fast reloads)
@st.cache_resource(show_spinner=True)
def load_resources(rebuild: bool):
    llm = init_llm()
    vs, pdf_names = init_vectorstore(rebuild=rebuild)
    return llm, vs, pdf_names

try:
    with st.spinner("Loading LLM + Vector DB..."):
        llm, vs, pdf_names = load_resources(rebuild=rebuild)
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

forced_pdf: Optional[str] = None
if routing == "Force one PDF":
    forced_pdf = st.selectbox("Choose PDF", pdf_names)

st.divider()

# Chat memory
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None

col1, col2 = st.columns([3, 1])

with col1:
    user_q = st.text_area("Your question", height=90, placeholder="Example: ops: For landslides, give triggers, monitoring, immediate actions, mitigation.")
with col2:
    st.write("Quick tests")
    if st.button("Test: Landslide ops"):
        user_q = "ops: For landslides, give triggers, monitoring, immediate actions, and mitigation."
    if st.button("Test: Hazard list fast"):
        user_q = "fast: List the main geologic hazards mentioned in the PDFs."
    if st.button("Test: Ask mode"):
        user_q = "ask: Areas of highest susceptibility"

ask_btn = st.button("Ask", type="primary", use_container_width=True)

def parse_mode_prefix(q: str, default_mode: str):
    q = (q or "").strip()
    for m in ["fast", "brief", "ops", "deep", "ask", "default"]:
        if q.lower().startswith(m + ":"):
            return m, q.split(":", 1)[1].strip()
    return default_mode, q

if ask_btn and user_q.strip():
    run_mode, clean_q = parse_mode_prefix(user_q, mode)

    with st.spinner("Retrieving + answering..."):
        result = answer_question(
            llm=llm,
            vs=vs,
            question=clean_q,
            strict=strict,
            cite=cite,
            mode=run_mode,
            forced_pdf=forced_pdf,
            last_sources=st.session_state.last_sources,
        )

    # Update follow-up routing memory
    if result.get("chosen_sources"):
        st.session_state.last_sources = result["chosen_sources"]

    st.session_state.history.append(
        {
            "q": user_q.strip(),
            "mode": run_mode,
            "forced_pdf": forced_pdf,
            "answer": result["answer"],
            "confidence": result.get("confidence", "?"),
            "chosen_sources": result.get("chosen_sources", []),
            "evidence": result.get("evidence", []),
        }
    )

st.subheader("Conversation")

for item in reversed(st.session_state.history):
    st.markdown(f"**You:** {item['q']}")
    meta = f"Confidence: **{item['confidence']}**"
    if item["forced_pdf"]:
        meta += f" | Forced PDF: **{item['forced_pdf']}**"
    if item["chosen_sources"] and not item["forced_pdf"]:
        meta += f" | Auto PDFs: **{', '.join(item['chosen_sources'])}**"
    st.caption(meta)

    st.markdown("**Assistant:**")
    st.write(item["answer"])

    if item["evidence"]:
        with st.expander("Evidence snippets"):
            for ev in item["evidence"]:
                st.write(ev)

    st.divider()