import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# .env (local only; Streamlit Cloud uses st.secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Streamlit secrets (Cloud) with fallback to env
try:
    import streamlit as st  # available when running Streamlit
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}

def _get_setting(name: str, default: str = "") -> str:
    """
    Priority:
      1) Streamlit secrets (Cloud)
      2) Environment variables
      3) Default
    """
    try:
        val = str(_SECRETS.get(name, "")).strip()
        if val:
            return val
    except Exception:
        pass
    val = os.getenv(name, "").strip()
    return val if val else default


# Vector DB
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

# Loaders
from langchain_community.document_loaders import PyMuPDFLoader, PDFMinerLoader

# Splitter + Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# LLM
from langchain_groq import ChatGroq


# -------------------- Config (secrets/env) --------------------
DOCS_DIR = _get_setting("DOCS_DIR", str(Path.cwd() / "docs"))
CHROMA_DIR = _get_setting("CHROMA_DIR", str(Path.cwd() / "chroma_db"))

GROQ_API_KEY = _get_setting("GROQ_API_KEY", "")
GROQ_MODEL = _get_setting("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = _get_setting("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(_get_setting("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(_get_setting("CHUNK_OVERLAP", "250"))

TOP_K = int(_get_setting("TOP_K", "10"))
FETCH_K = int(_get_setting("FETCH_K", "60"))
LAMBDA_MULT = float(_get_setting("LAMBDA_MULT", "0.5"))

MIN_SOURCES = int(_get_setting("MIN_SOURCES", "2"))
TEMPERATURE = float(_get_setting("TEMPERATURE", "0.0"))

AUTO_FILTER_TOP_SOURCES = int(_get_setting("AUTO_FILTER_TOP_SOURCES", "2"))
AUTO_FILTER_MIN_HITS = int(_get_setting("AUTO_FILTER_MIN_HITS", "2"))

MAX_CONTEXT_CHUNKS = int(_get_setting("MAX_CONTEXT_CHUNKS", "8"))

EVIDENCE_SNIPPETS = int(_get_setting("EVIDENCE_SNIPPETS", "2"))
EVIDENCE_SNIPPET_CHARS = int(_get_setting("EVIDENCE_SNIPPET_CHARS", "220"))

FOLLOWUP_USE_LAST_SOURCES = _get_setting("FOLLOWUP_USE_LAST_SOURCES", "1") == "1"


# -------------------- Helpers --------------------
def ensure_api_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY missing. Set it in Streamlit Secrets (App settings -> Secrets) "
            "or as an environment variable."
        )
    if not GROQ_API_KEY.startswith("gsk_"):
        raise RuntimeError("GROQ_API_KEY looks invalid (should start with gsk_).")

def list_pdfs(folder: str, only_pdfs: Optional[List[str]] = None) -> List[Path]:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".pdf"])
    if only_pdfs:
        only_set = {s.strip() for s in only_pdfs if s.strip()}
        pdfs = [f for f in pdfs if f.name in only_set]
    return pdfs

def load_pdf(path: str) -> List[Document]:
    try:
        docs = PyMuPDFLoader(path).load()
    except Exception:
        docs = PDFMinerLoader(path).load()

    src = Path(path).name
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = src
    return docs

def split_docs(raw_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(raw_docs)

def build_or_load_chroma(chroma_dir: str, docs: Optional[List[Document]], rebuild: bool) -> Chroma:
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    cpath = Path(chroma_dir)
    has_existing = cpath.exists() and any(cpath.iterdir())

    if has_existing and not rebuild:
        return Chroma(persist_directory=str(cpath), embedding_function=embed)

    if rebuild and cpath.exists():
        import shutil
        shutil.rmtree(str(cpath), ignore_errors=True)

    if not docs:
        raise RuntimeError("No docs provided to build index.")

    vs = Chroma.from_documents(docs, embed, persist_directory=str(cpath))
    return vs

def make_retriever(vs: Chroma, source_allowlist: Optional[List[str]] = None):
    search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT}
    if source_allowlist:
        search_kwargs["filter"] = {"source": {"$in": source_allowlist}}
    return vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

def retrieve_docs(retriever, query: str) -> List[Document]:
    try:
        return retriever.invoke(query)
    except Exception:
        return retriever.get_relevant_documents(query)

def normalize_page(meta: dict) -> str:
    page = (meta or {}).get("page", None)
    if isinstance(page, int):
        return str(page + 1)
    if page is None:
        return "?"
    return str(page)

def tally_sources(docs: List[Document]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for d in docs:
        src = (d.metadata or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts

def choose_top_sources(counts: Dict[str, int], top_n: int, min_hits: int) -> List[str]:
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [src for src, c in ranked if c >= min_hits][:top_n]

def expand_query_if_needed(q: str) -> str:
    ql = q.lower()
    broad = any(x in ql for x in ["list", "main", "overview", "covered", "hazards", "risk", "mitigation", "monitoring", "susceptibility"])
    if not broad:
        return q
    return q + " Include hazard categories and subtypes, triggers, susceptibility, mapping, monitoring, mitigation, and referenced tables/figures."

def build_context(docs: List[Document], max_chunks: int, cite: bool) -> str:
    picked = docs[:max_chunks]
    parts = []
    for d in picked:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        p = normalize_page(meta)
        header = f"[{src} p.{p}] " if cite else f"[{src}] "
        parts.append(header + d.page_content)
    return "\n\n---\n\n".join(parts)

def make_evidence_snippets(docs: List[Document], n: int) -> List[str]:
    out = []
    for d in docs[: max(0, n)]:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        p = normalize_page(meta)
        text = " ".join(d.page_content.split())
        text = text[:EVIDENCE_SNIPPET_CHARS].rstrip()
        if len(text) == EVIDENCE_SNIPPET_CHARS:
            text += "…"
        out.append(f"- {src} (p.{p}): “{text}”")
    return out

def confidence_from_retrieval(docs: List[Document], chosen_sources: List[str]) -> str:
    if not docs or len(docs) < MIN_SOURCES:
        return "Low"
    if chosen_sources:
        return "High"
    return "Medium"

def build_prompt(context: str, question: str, mode: str, cite: bool) -> str:
    if cite:
        citation_rule = "Every factual claim MUST be cited as: (SourceFile.pdf p.X). If you can't cite it, don't say it."
    else:
        citation_rule = "Do NOT invent facts. If you cannot support a claim from the context, omit it."

    mode_instructions = {
        "fast": "Answer in 3–6 lines. Most important points only.",
        "brief": "Answer as 5 bullets max. Compact.",
        "ops": "Answer as an operational checklist: Triggers, Where likely, Monitoring, Immediate actions, Mitigation, Stakeholders.",
        "deep": "Answer with detail and structure. Still avoid filler.",
        "ask": "Do NOT answer yet. Ask up to 3 clarifying questions that would let you answer correctly using the PDFs.",
        "default": "Answer with clear bullets + short headings."
    }.get(mode, "Answer with clear bullets + short headings.")

    return f"""
You are a Geohazard Management assistant (engineering geology / disaster risk reduction).

STRICT GROUNDING RULES:
1) Use ONLY the provided context. Do NOT guess or use outside knowledge.
2) If context is insufficient, reply exactly: "NOT ENOUGH EVIDENCE IN PROVIDED PDFS" and say what document/topic is missing.
3) {citation_rule}
4) Prefer structure and bullets. Be direct.

OUTPUT STYLE:
{mode_instructions}

USER QUESTION:
{question}

CONTEXT:
{context}
""".strip()

def answer_question(
    llm: ChatGroq,
    vs: Chroma,
    question: str,
    strict: bool = False,
    cite: bool = False,
    mode: str = "default",
    forced_pdf: Optional[str] = None,
    last_sources: Optional[List[str]] = None,
) -> Dict:
    q2 = expand_query_if_needed(question)

    # forced PDF
    if forced_pdf:
        retriever = make_retriever(vs, source_allowlist=[forced_pdf])
        docs = retrieve_docs(retriever, q2)
        chosen_sources = [forced_pdf]
    else:
        docs = []
        chosen_sources = []

        # follow-up acceleration
        if FOLLOWUP_USE_LAST_SOURCES and last_sources:
            retriever_last = make_retriever(vs, source_allowlist=last_sources)
            docs_last = retrieve_docs(retriever_last, q2)
            if docs_last and len(docs_last) >= MIN_SOURCES:
                docs = docs_last
                chosen_sources = list(last_sources)

        # full routing if needed
        if not docs:
            retriever_a = make_retriever(vs, source_allowlist=None)
            docs_a = retrieve_docs(retriever_a, q2)
            counts = tally_sources(docs_a)
            chosen_sources = choose_top_sources(counts, AUTO_FILTER_TOP_SOURCES, AUTO_FILTER_MIN_HITS)

            if chosen_sources:
                retriever_b = make_retriever(vs, source_allowlist=chosen_sources)
                docs = retrieve_docs(retriever_b, q2)
            else:
                docs = docs_a

    if strict and (not docs or len(docs) < MIN_SOURCES):
        return {
            "answer": "NOT ENOUGH EVIDENCE IN PROVIDED PDFS",
            "confidence": "Low",
            "chosen_sources": chosen_sources,
            "evidence": [],
            "docs_used": docs,
        }

    context = build_context(docs, MAX_CONTEXT_CHUNKS, cite=cite)
    prompt = build_prompt(context=context, question=question, mode=mode, cite=cite)
    resp = llm.invoke(prompt)

    return {
        "answer": resp.content.strip(),
        "confidence": confidence_from_retrieval(docs, chosen_sources),
        "chosen_sources": chosen_sources,
        "evidence": make_evidence_snippets(docs, EVIDENCE_SNIPPETS),
        "docs_used": docs,
    }

def init_llm() -> ChatGroq:
    ensure_api_key()
    return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=TEMPERATURE)

def init_vectorstore(rebuild: bool = False, only_pdfs: Optional[List[str]] = None) -> Tuple[Chroma, List[str]]:
    pdfs = list_pdfs(DOCS_DIR, only_pdfs=only_pdfs)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {Path(DOCS_DIR).resolve()}")

    raw_docs: List[Document] = []
    for f in pdfs:
        raw_docs.extend(load_pdf(str(f)))

    chunks = split_docs(raw_docs)
    vs = build_or_load_chroma(CHROMA_DIR, chunks, rebuild=rebuild)
    return vs, [p.name for p in pdfs]