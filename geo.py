#!/usr/bin/env python3

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

"""
geo.py — Local Geohazard RAG (Mac/Windows)

Core:
- PDFs in DOCS_DIR (from .env)  [default ./docs]
- Chroma vector DB in CHROMA_DIR (from .env) [default ./chroma_db]
- Groq Chat model for answering

What this version focuses on:
✅ LC 0.2+ safe imports (no langchain.schema / text_splitter / chains)
✅ 2-stage auto PDF routing (wide -> source majority -> focused)
✅ Fast "assistant-like" outputs (modes + confidence) WITHOUT needing citations
✅ Optional citations when you want: --cite or query prefix "cite:"
✅ Evidence snippets (1–2 short lines) to build trust without page refs
✅ usepdf:<filename.pdf> always obeys
✅ Optional: --only-pdfs "a.pdf,b.pdf" to restrict scope
✅ Optional: --strict to refuse if evidence is weak
✅ Session memory: keeps last chosen PDFs for follow-ups

Run:
  python geo.py --rebuild
Then:
  python geo.py
"""

import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# -------------------- .env --------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------- Vector DB --------------------
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback

# -------------------- Embeddings --------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

# -------------------- Loaders --------------------
from langchain_community.document_loaders import PyMuPDFLoader, PDFMinerLoader

# ✅ Modern splitter + Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------- LLM --------------------
from langchain_groq import ChatGroq


# -------------------- Config (from .env) --------------------
DOCS_DIR = os.getenv("DOCS_DIR", "").strip() or str(Path.cwd() / "docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "").strip() or str(Path.cwd() / "chroma_db")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "10"))
FETCH_K = int(os.getenv("FETCH_K", "60"))
LAMBDA_MULT = float(os.getenv("LAMBDA_MULT", "0.5"))

# Strictness
MIN_SOURCES = int(os.getenv("MIN_SOURCES", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Auto source filter
AUTO_FILTER_TOP_SOURCES = int(os.getenv("AUTO_FILTER_TOP_SOURCES", "2"))
AUTO_FILTER_MIN_HITS = int(os.getenv("AUTO_FILTER_MIN_HITS", "2"))

# Answer context size
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))

# Evidence snippets
EVIDENCE_SNIPPETS = int(os.getenv("EVIDENCE_SNIPPETS", "2"))  # show 0..2 snippets
EVIDENCE_SNIPPET_CHARS = int(os.getenv("EVIDENCE_SNIPPET_CHARS", "220"))

# Follow-up behavior
FOLLOWUP_USE_LAST_SOURCES = os.getenv("FOLLOWUP_USE_LAST_SOURCES", "1").strip() == "1"


# -------------------- Helpers --------------------
def die(msg: str, code: int = 1):
    print(msg)
    raise SystemExit(code)


def ensure_api_key():
    if not GROQ_API_KEY:
        die("❌ GROQ_API_KEY missing. Put it in .env as GROQ_API_KEY=gsk_....")
    if not GROQ_API_KEY.startswith("gsk_"):
        die("❌ GROQ_API_KEY looks invalid (should start with gsk_).")


def list_pdfs(folder: str, only_pdfs: Optional[List[str]] = None) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        die(f"📂 Created docs folder at: {p.resolve()}\nPut PDFs inside and rerun.", 0)

    pdfs = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".pdf"])

    if only_pdfs:
        only_set = {s.strip() for s in only_pdfs if s.strip()}
        pdfs = [f for f in pdfs if f.name in only_set]

    if not pdfs:
        die(f"❌ No PDFs found in: {p.resolve()}\nAdd your geohazard PDFs and rerun.", 0)

    return pdfs


def load_pdf(path: str) -> List[Document]:
    """Prefer PyMuPDF (better metadata), fallback to PDFMiner."""
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
        print(f"💿 Loading existing Chroma DB: {cpath.resolve()}")
        return Chroma(persist_directory=str(cpath), embedding_function=embed)

    if rebuild and cpath.exists():
        import shutil
        print(f"🗑️ Rebuilding: deleting old Chroma DB: {cpath.resolve()}")
        shutil.rmtree(str(cpath), ignore_errors=True)

    if not docs:
        die("❌ No docs provided to build the index.")

    print("🛠️ Building Chroma DB…")
    vs = Chroma.from_documents(docs, embed, persist_directory=str(cpath))
    print("✅ Index built.")
    return vs


def make_retriever(vs: Chroma, source_allowlist: Optional[List[str]] = None):
    """MMR retriever + optional source filter."""
    search_kwargs = {
        "k": TOP_K,
        "fetch_k": FETCH_K,
        "lambda_mult": LAMBDA_MULT,
    }
    if source_allowlist:
        search_kwargs["filter"] = {"source": {"$in": source_allowlist}}
    return vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def retrieve_docs(retriever, query: str) -> List[Document]:
    """New-style retriever call (LC 0.2+)."""
    try:
        return retriever.invoke(query)
    except Exception:
        return retriever.get_relevant_documents(query)


def extract_usepdf(query: str) -> Tuple[Optional[str], str]:
    """usepdf:<filename.pdf> <question>"""
    m = re.match(r"^\s*usepdf\s*:\s*([^\s]+)\s+(.*)$", query.strip(), flags=re.IGNORECASE)
    if not m:
        return None, query.strip()
    return m.group(1).strip(), m.group(2).strip()


def extract_prefix_mode(query: str) -> Tuple[Optional[str], str]:
    """
    Supports:
      fast: ...
      brief: ...
      ops: ...
      deep: ...
      ask: ...
      cite: ...
    Returns (mode_or_flag, cleaned_query)
    """
    m = re.match(r"^\s*(fast|brief|ops|deep|ask|cite)\s*:\s*(.+)$", query.strip(), flags=re.IGNORECASE)
    if not m:
        return None, query.strip()
    return m.group(1).lower(), m.group(2).strip()


def expand_query_if_needed(q: str) -> str:
    """Light recall boost for broad queries."""
    ql = q.lower()
    broad = any(
        x in ql
        for x in [
            "list",
            "main",
            "overview",
            "covered",
            "hazards",
            "risk",
            "mitigation",
            "monitoring",
            "susceptibility",
        ]
    )
    if not broad:
        return q
    extra = (
        " Include hazard categories and subtypes, triggers, susceptibility, mapping, "
        "monitoring, mitigation, and any referenced tables/figures."
    )
    return q + extra


def normalize_page(meta: dict) -> str:
    """Display page as 1-indexed if int (many loaders store 0-index)."""
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
    picked = [src for src, c in ranked if c >= min_hits][:top_n]
    return picked


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
    """
    Simple heuristic confidence:
    - High: >= MIN_SOURCES and chosen_sources not empty
    - Medium: >= MIN_SOURCES but routing unclear
    - Low: < MIN_SOURCES
    """
    if not docs or len(docs) < MIN_SOURCES:
        return "Low"
    if chosen_sources:
        return "High"
    return "Medium"


def build_prompt(context: str, question: str, mode: str, cite: bool) -> str:
    """
    If cite=False: don't force page-by-page citations.
    Still strictly grounded: only use context; refuse if insufficient.
    """
    if cite:
        citation_rule = "Every factual claim MUST be cited as: (SourceFile.pdf p.X). If you can't cite it, don't say it."
    else:
        citation_rule = "Do NOT invent facts. If you cannot support a claim from the context, omit it."

    mode_instructions = {
        "fast": "Answer in 3–6 lines. No fluff. Most important points only.",
        "brief": "Answer as 5 bullets max. Compact.",
        "ops": "Answer as an operational checklist: Triggers, Where likely, Monitoring, Immediate actions, Mitigation, Stakeholders.",
        "deep": "Answer with detail and structure. Still avoid filler.",
        "ask": "Do NOT answer yet. Ask up to 3 clarifying questions that would let you answer correctly using the PDFs.",
        "default": "Answer with clear bullets + short headings.",
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


def format_sources(srcs: List[Document], limit: int = 12) -> str:
    if not srcs:
        return "Sources: (none)"
    lines = ["Sources used:"]
    seen = set()
    for d in srcs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        p = normalize_page(meta)
        key = (src, p)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f" - {src} • p.{p}")
        if len(lines) - 1 >= limit:
            break
    return "\n".join(lines)


def answer_question(
    llm: ChatGroq,
    vs: Chroma,
    q: str,
    forced_pdf: Optional[str],
    strict: bool,
    cite: bool,
    mode: str,
    last_sources: Optional[List[str]] = None,
) -> Tuple[str, List[Document], List[str]]:
    """
    2-stage retrieval to auto-pick correct PDF(s):
    - If forced_pdf: filter immediately
    - Else:
        Optionally: if FOLLOWUP_USE_LAST_SOURCES and last_sources exist, try those first (fast follow-up)
        Stage A: retrieve wide
        Stage B: pick top PDF(s) by hit-count, then re-retrieve filtered
    """
    q2 = expand_query_if_needed(q)

    # 1) Forced PDF always wins
    if forced_pdf:
        retriever = make_retriever(vs, source_allowlist=[forced_pdf])
        docs = retrieve_docs(retriever, q2)
        chosen_sources = [forced_pdf]
    else:
        # 2) Follow-up acceleration: try last sources first (if enabled)
        if FOLLOWUP_USE_LAST_SOURCES and last_sources:
            retriever_last = make_retriever(vs, source_allowlist=last_sources)
            docs_last = retrieve_docs(retriever_last, q2)
            if docs_last and len(docs_last) >= MIN_SOURCES:
                docs = docs_last
                chosen_sources = list(last_sources)
            else:
                docs = []
                chosen_sources = []
        else:
            docs = []
            chosen_sources = []

        # 3) If not good enough, do full auto-routing
        if not docs:
            retriever_a = make_retriever(vs, source_allowlist=None)
            docs_a = retrieve_docs(retriever_a, q2)

            counts = tally_sources(docs_a)
            chosen_sources = choose_top_sources(
                counts,
                top_n=AUTO_FILTER_TOP_SOURCES,
                min_hits=AUTO_FILTER_MIN_HITS,
            )

            if chosen_sources:
                retriever_b = make_retriever(vs, source_allowlist=chosen_sources)
                docs = retrieve_docs(retriever_b, q2)
            else:
                docs = docs_a

    # Strict brake
    if strict and (not docs or len(docs) < MIN_SOURCES):
        return "NOT ENOUGH EVIDENCE IN PROVIDED PDFS", docs, chosen_sources

    context = build_context(docs, max_chunks=MAX_CONTEXT_CHUNKS, cite=cite)
    prompt = build_prompt(context=context, question=q, mode=mode, cite=cite)
    resp = llm.invoke(prompt)

    answer_text = resp.content if hasattr(resp, "content") else str(resp)
    return answer_text.strip(), docs, chosen_sources


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Geohazard RAG (Groq + Chroma + local PDFs)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma index from docs/")
    parser.add_argument("--strict", action="store_true", help="Strict mode (refuse if weak evidence)")
    parser.add_argument("--only-pdfs", type=str, default="", help="Comma-separated PDF filenames to include (optional)")
    parser.add_argument("--cite", action="store_true", help="Force citations on all answers")
    args = parser.parse_args()

    ensure_api_key()

    only_pdfs = [s.strip() for s in args.only_pdfs.split(",") if s.strip()] or None

    print("📍 DOCS_DIR   :", Path(DOCS_DIR).resolve())
    print("💾 CHROMA_DIR :", Path(CHROMA_DIR).resolve())
    print("🧠 GROQ_MODEL :", GROQ_MODEL)
    print("🧩 EMBED_MODEL:", EMBED_MODEL)
    print("🧷 CITATIONS  :", "ON" if args.cite else "OFF (default)")
    if only_pdfs:
        print("📌 ONLY_PDFS  :", ", ".join(only_pdfs))

    pdfs = list_pdfs(DOCS_DIR, only_pdfs=only_pdfs)
    available = {p.name for p in pdfs}

    # Load + split
    raw_docs: List[Document] = []
    print("\n📥 Loading PDFs…")
    for f in pdfs:
        print(" -", f.name)
        raw_docs.extend(load_pdf(str(f)))

    print(f"✅ Loaded {len(raw_docs)} page-docs")
    chunks = split_docs(raw_docs)
    print(f"🔪 Total chunks: {len(chunks)}")

    # Vectorstore
    vs = build_or_load_chroma(CHROMA_DIR, chunks, rebuild=args.rebuild)

    # LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=TEMPERATURE,
    )

    print("\n🚀 RAG Ready (geohazards). Type 'exit' to quit.")
    print("👉 Prefix modes: fast:, brief:, ops:, deep:, ask:")
    print("👉 To force citations for one question: cite: <question>  (or run with --cite)")
    print("👉 Tip: usepdf:<filename.pdf> <question>  to force one PDF.\n")

    last_chosen_sources: Optional[List[str]] = None

    while True:
        try:
            raw_q = input("Your Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not raw_q:
            continue
        if raw_q.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        # Extract mode/cite prefix
        prefix, q1 = extract_prefix_mode(raw_q)

        mode = "default"
        cite = args.cite

        if prefix in {"fast", "brief", "ops", "deep", "ask"}:
            mode = prefix
        elif prefix == "cite":
            cite = True

        forced_pdf, clean_q = extract_usepdf(q1)

        if forced_pdf and forced_pdf not in available:
            print(f"⚠️ usepdf file not found in docs/: {forced_pdf}")
            print("Available PDFs:")
            for p in pdfs:
                print(" -", p.name)
            continue

        ans, docs_used, chosen_sources = answer_question(
            llm=llm,
            vs=vs,
            q=clean_q,
            forced_pdf=forced_pdf,
            strict=args.strict,
            cite=cite,
            mode=mode,
            last_sources=last_chosen_sources,
        )

        # Update session memory if we actually routed
        if chosen_sources:
            last_chosen_sources = list(chosen_sources)

        # Show routing info
        if chosen_sources and not forced_pdf:
            print(f"\n[auto-selected PDFs] {', '.join(chosen_sources)}\n")

        # Confidence + Evidence snippets
        conf = confidence_from_retrieval(docs_used, chosen_sources)
        print(f"Confidence: {conf}")

        if EVIDENCE_SNIPPETS > 0 and docs_used:
            snippets = make_evidence_snippets(docs_used, n=EVIDENCE_SNIPPETS)
            if snippets:
                print("Evidence snippets:")
                for s in snippets:
                    print(s)
                print()

        print("Answer:\n" + ans + "\n")

        # Show sources list only when citations are ON
        if cite:
            print(format_sources(docs_used, limit=12))
            print()


if __name__ == "__main__":
    main()