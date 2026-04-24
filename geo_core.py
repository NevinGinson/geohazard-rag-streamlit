import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}


def _get_setting(name: str, default: str = "") -> str:
    try:
        val = str(_SECRETS.get(name, "")).strip()
        if val:
            return val
    except Exception:
        pass
    val = os.getenv(name, "").strip()
    return val if val else default


try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyMuPDFLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq


# ---- Config ----

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


# ---- Core helpers ----

def ensure_api_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY missing. Set it in Streamlit Secrets or as an environment variable."
        )
    if not GROQ_API_KEY.startswith("gsk_"):
        raise RuntimeError("GROQ_API_KEY looks invalid (should start with gsk_).")


def list_pdfs(folder, only_pdfs=None):
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".pdf"])
    if only_pdfs:
        only_set = {s.strip() for s in only_pdfs if s.strip()}
        pdfs = [f for f in pdfs if f.name in only_set]
    return pdfs


def load_pdf(path):
    try:
        docs = PyMuPDFLoader(path).load()
    except Exception:
        docs = PDFMinerLoader(path).load()
    src = Path(path).name
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = src
    return docs


def split_docs(raw_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(raw_docs)


def build_or_load_chroma(chroma_dir, docs, rebuild):
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
    return Chroma.from_documents(docs, embed, persist_directory=str(cpath))


def make_retriever(vs, source_allowlist=None):
    search_kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT}
    if source_allowlist:
        search_kwargs["filter"] = {"source": {"$in": source_allowlist}}
    return vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def retrieve_docs(retriever, query):
    try:
        return retriever.invoke(query)
    except Exception:
        return retriever.get_relevant_documents(query)


def normalize_page(meta):
    page = (meta or {}).get("page", None)
    if isinstance(page, int):
        return str(page + 1)
    if page is None:
        return "?"
    return str(page)


def tally_sources(docs):
    counts = {}
    for d in docs:
        src = (d.metadata or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def choose_top_sources(counts, top_n, min_hits):
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [src for src, c in ranked if c >= min_hits][:top_n]


def expand_query_if_needed(q):
    ql = q.lower()
    broad = any(x in ql for x in [
        "list", "main", "overview", "covered", "hazards", "risk",
        "mitigation", "monitoring", "susceptibility"
    ])
    if not broad:
        return q
    return q + " Include hazard categories and subtypes, triggers, susceptibility, mapping, monitoring, mitigation, and referenced tables/figures."


def build_context(docs, max_chunks, cite):
    picked = docs[:max_chunks]
    parts = []
    for d in picked:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        p = normalize_page(meta)
        header = f"[{src} p.{p}] " if cite else f"[{src}] "
        parts.append(header + d.page_content)
    return "\n\n---\n\n".join(parts)


def make_evidence_snippets(docs, n):
    out = []
    for d in docs[:max(0, n)]:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        p = normalize_page(meta)
        text = " ".join(d.page_content.split())
        text = text[:EVIDENCE_SNIPPET_CHARS].rstrip()
        if len(text) == EVIDENCE_SNIPPET_CHARS:
            text += "…"
        out.append(f"- {src} (p.{p}): \"{text}\"")
    return out


def confidence_from_retrieval(docs, chosen_sources):
    if not docs or len(docs) < MIN_SOURCES:
        return "Low"
    if chosen_sources:
        return "High"
    return "Medium"


def build_prompt(context, question, mode, cite):
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


# ---- Main answer function ----

def answer_question(llm, vs, question, strict=False, cite=False, mode="default",
                    forced_pdf=None, last_sources=None):
    q2 = expand_query_if_needed(question)

    if forced_pdf:
        retriever = make_retriever(vs, source_allowlist=[forced_pdf])
        docs = retrieve_docs(retriever, q2)
        chosen_sources = [forced_pdf]
    else:
        docs = []
        chosen_sources = []
        if FOLLOWUP_USE_LAST_SOURCES and last_sources:
            ret_last = make_retriever(vs, source_allowlist=last_sources)
            docs_last = retrieve_docs(ret_last, q2)
            if docs_last and len(docs_last) >= MIN_SOURCES:
                docs = docs_last
                chosen_sources = list(last_sources)
        if not docs:
            ret_a = make_retriever(vs)
            docs_a = retrieve_docs(ret_a, q2)
            counts = tally_sources(docs_a)
            chosen_sources = choose_top_sources(counts, AUTO_FILTER_TOP_SOURCES, AUTO_FILTER_MIN_HITS)
            if chosen_sources:
                docs = retrieve_docs(make_retriever(vs, chosen_sources), q2)
            else:
                docs = docs_a

    if strict and (not docs or len(docs) < MIN_SOURCES):
        return {
            "answer": "NOT ENOUGH EVIDENCE IN PROVIDED PDFS",
            "confidence": "Low", "chosen_sources": chosen_sources,
            "evidence": [], "docs_used": docs,
            "hazard_tags": [], "risk_level": "Unknown", "locations": [],
            "parameters": [], "citation_checks": [],
        }

    context = build_context(docs, MAX_CONTEXT_CHUNKS, cite=cite)
    prompt = build_prompt(context=context, question=question, mode=mode, cite=cite)

    resp = llm.invoke(prompt)
    answer_text = ""
    if hasattr(resp, "content"):
        answer_text = resp.content
    else:
        answer_text = str(resp)
    answer_text = (answer_text or "").strip()

    tags = _tag_hazards(question, answer_text, docs)
    rlevel = _score_risk(question, answer_text, docs)
    locs = _find_coords(docs)
    params = extract_hazard_params(docs)
    cite_checks = verify_citations(answer_text, docs) if cite else []

    return {
        "answer": answer_text,
        "confidence": confidence_from_retrieval(docs, chosen_sources),
        "chosen_sources": chosen_sources,
        "evidence": make_evidence_snippets(docs, EVIDENCE_SNIPPETS),
        "docs_used": docs,
        "hazard_tags": tags,
        "risk_level": rlevel,
        "locations": locs,
        "parameters": params,
        "citation_checks": cite_checks,
    }


def init_llm():
    ensure_api_key()
    return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=TEMPERATURE)


def init_vectorstore(rebuild=False, only_pdfs=None):
    pdfs = list_pdfs(DOCS_DIR, only_pdfs=only_pdfs)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {Path(DOCS_DIR).resolve()}")
    raw_docs = []
    for f in pdfs:
        raw_docs.extend(load_pdf(str(f)))
    chunks = split_docs(raw_docs)
    vs = build_or_load_chroma(CHROMA_DIR, chunks, rebuild=rebuild)
    return vs, [p.name for p in pdfs]


# ---------- hazard classification ----------

HAZARD_KW = {
    "Landslide": ["landslide", "debris flow", "mudflow", "rockfall", "slope failure",
                   "slope stability", "mass movement", "mass wasting", "translational slide",
                   "rotational slide", "shallow slide", "deep-seated"],
    "Earthquake": ["earthquake", "seismic", "seismicity", "fault", "faulting",
                    "paleoseismic", "neotectonic", "ground motion", "pga", "liquefaction",
                    "site response", "amplification", "epicenter", "epicentre"],
    "Flood": ["flood", "flooding", "inundation", "flash flood", "fluvial", "pluvial",
              "storm surge", "overbank", "return period", "discharge", "floodplain"],
    "Tsunami": ["tsunami", "tidal wave", "run-up", "runup", "tsunami deposit",
                "tsunamigenic", "coastal inundation"],
    "Volcanic": ["volcanic", "volcano", "eruption", "lava", "pyroclastic", "lahar",
                 "tephra", "ash fall", "magma", "caldera", "fumarole", "insar"],
    "Subsidence": ["subsidence", "sinkhole", "karst", "ground settlement",
                   "compaction", "consolidation", "cavity collapse"],
    "Erosion": ["erosion", "coastal erosion", "cliff retreat", "gully", "scour",
                "bank erosion", "shoreline retreat", "sediment transport"],
}


def _tag_hazards(question, answer, docs):
    blob = question.lower() + " " + answer.lower()
    for d in docs[:MAX_CONTEXT_CHUNKS]:
        blob += " " + d.page_content.lower()
    found = []
    for htype, kws in HAZARD_KW.items():
        for kw in kws:
            if kw in blob:
                found.append(htype)
                break
    return sorted(set(found))


# ---------- risk level ----------

_RISK_KW = {
    "Critical": (4, ["immediate danger", "life-threatening", "catastrophic",
                      "imminent failure", "emergency evacuation", "extreme risk",
                      "very high risk", "casualties"]),
    "High":     (3, ["high risk", "significant risk", "major damage", "severe",
                      "extensive damage", "high susceptibility", "high vulnerability",
                      "active fault", "unstable slope"]),
    "Medium":   (2, ["moderate risk", "medium risk", "moderate susceptibility",
                      "potential damage", "possible occurrence", "monitoring recommended"]),
    "Low":      (1, ["low risk", "minimal risk", "low susceptibility", "unlikely",
                      "rare occurrence", "stable", "negligible"]),
}


def _score_risk(question, answer, docs):
    blob = question.lower() + " " + answer.lower()
    for d in docs[:MAX_CONTEXT_CHUNKS]:
        blob += " " + d.page_content.lower()
    best = "Undetermined"
    best_w = 0
    for level, (w, terms) in _RISK_KW.items():
        for t in terms:
            if t in blob:
                if w > best_w:
                    best_w = w
                    best = level
                break
    return best


# ---------- coordinate extraction ----------

_RE_DD = re.compile(
    r"(?P<lat>-?\d{1,3}\.\d{2,8})\s*°?\s*(?P<latd>[NSns])?\s*[,;\s]+\s*"
    r"(?P<lon>-?\d{1,3}\.\d{2,8})\s*°?\s*(?P<lond>[EWew])?"
)
_RE_DMS = re.compile(
    r"(?P<d1>\d{1,3})\s*°\s*(?P<m1>\d{1,2})\s*[''′]\s*(?P<s1>\d{1,2}(?:\.\d+)?)\s*[\"″]?\s*(?P<dir1>[NSns])\s*[,;\s]+\s*"
    r"(?P<d2>\d{1,3})\s*°\s*(?P<m2>\d{1,2})\s*[''′]\s*(?P<s2>\d{1,2}(?:\.\d+)?)\s*[\"″]?\s*(?P<dir2>[EWew])"
)


def _dms2dd(deg, mn, sec, direction):
    dd = float(deg) + float(mn) / 60 + float(sec) / 3600
    if direction.upper() in ("S", "W"):
        dd = -dd
    return dd


def _find_coords(docs):
    out = []
    seen = set()
    for d in docs[:MAX_CONTEXT_CHUNKS]:
        txt = d.page_content
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        pg = normalize_page(meta)
        for pat in [_RE_DD, _RE_DMS]:
            for m in pat.finditer(txt):
                g = m.groupdict()
                if "d1" in g and g["d1"]:
                    lat = _dms2dd(g["d1"], g["m1"], g["s1"], g["dir1"])
                    lon = _dms2dd(g["d2"], g["m2"], g["s2"], g["dir2"])
                else:
                    lat = float(g["lat"])
                    lon = float(g["lon"])
                    if (g.get("latd") or "").upper() == "S": lat = -abs(lat)
                    if (g.get("lond") or "").upper() == "W": lon = -abs(lon)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                key = (round(lat, 3), round(lon, 3))
                if key in seen:
                    continue
                seen.add(key)
                s = max(0, m.start() - 40)
                e = min(len(txt), m.end() + 40)
                out.append({"lat": round(lat, 5), "lon": round(lon, 5),
                            "source": src, "page": pg, "context": txt[s:e].strip()})
    return out


# ---------- THESIS FEATURE: structured hazard parameter extraction ----------
# pulls actual numerical values from the text — magnitudes, PGA,
# recurrence intervals, slip rates, depths, angles etc.
# this is what makes it useful for a geoscientist, not just a chatbot.

_PARAM_PATTERNS = [
    # magnitude: M7.0, Mw 6.5, magnitude 5.8
    ("magnitude", re.compile(r"(?:M[wWsSlL]?\s*=?\s*|magnitude\s+)(\d+\.?\d*)", re.I)),
    # PGA: 0.3g, 0.45 g, PGA of 0.2g
    ("PGA (g)", re.compile(r"(?:PGA|peak ground acceleration)\s*(?:of\s+)?(\d+\.?\d*)\s*g", re.I)),
    # recurrence / return period: 475-year, return period of 500 years
    ("return period (yr)", re.compile(r"(\d+)[\s-]*(?:year|yr)\s*(?:return|recurrence)", re.I)),
    ("return period (yr)", re.compile(r"(?:return period|recurrence interval)\s*(?:of\s+)?(\d+)\s*(?:year|yr)", re.I)),
    # slip rate: 2 mm/yr, 0.5 mm/year
    ("slip rate (mm/yr)", re.compile(r"(\d+\.?\d*)\s*mm\s*/\s*(?:yr|year|a)", re.I)),
    # depth: 10 km depth, depth of 15 km, focal depth 12 km
    ("depth (km)", re.compile(r"(?:depth|focal depth)\s*(?:of\s+)?(\d+\.?\d*)\s*km", re.I)),
    # slope angle: 30°, slope of 25 degrees
    ("slope angle (°)", re.compile(r"(?:slope|angle|inclination|gradient)\s*(?:of\s+)?(\d+\.?\d*)\s*(?:°|degrees?)", re.I)),
    # displacement: 2.5 m displacement, displaced 1.3 m
    ("displacement (m)", re.compile(r"(?:displacement|displaced|offset)\s*(?:of\s+)?(\d+\.?\d*)\s*m(?:eter|etre)?s?\b", re.I)),
    # velocity: 5 cm/year, 12 mm/year movement
    ("velocity (mm/yr)", re.compile(r"(\d+\.?\d*)\s*(?:mm|cm)\s*/\s*(?:yr|year|a)\s*(?:movement|velocity|rate)?", re.I)),
    # area: 2.5 km², 150 hectares
    ("area (km²)", re.compile(r"(\d+\.?\d*)\s*km\s*²", re.I)),
    # intensity: MMI VII, intensity VIII
    ("MMI intensity", re.compile(r"(?:MMI|intensity)\s*(?:of\s+)?([IVXL]+)", re.I)),
]


def extract_hazard_params(docs):
    """Pull quantitative hazard parameters from retrieved chunks.
    Returns a list of {param, value, unit, source, page, context}."""
    found = []
    seen = set()  # avoid duplicate values from overlapping chunks

    for d in docs[:MAX_CONTEXT_CHUNKS]:
        txt = d.page_content
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        pg = normalize_page(meta)

        for param_name, pattern in _PARAM_PATTERNS:
            for m in pattern.finditer(txt):
                val = m.group(1)
                # dedup by param+value+source
                key = (param_name, val, src)
                if key in seen:
                    continue
                seen.add(key)

                s = max(0, m.start() - 30)
                e = min(len(txt), m.end() + 50)
                ctx = txt[s:e].strip()

                found.append({
                    "param": param_name,
                    "value": val,
                    "source": src,
                    "page": pg,
                    "context": ctx,
                })

    return found


# ---------- THESIS FEATURE: citation verification ----------
# when citations mode is on, the LLM outputs things like "(Report.pdf p.12)"
# this function checks whether those citations actually match the retrieved chunks.
# catches hallucinated citations — which is a real problem with RAG systems.

_CITE_RE = re.compile(r"\(([^)]+\.pdf)\s+p\.(\d+)\)", re.I)


def verify_citations(answer_text, docs):
    """Check each (file.pdf p.X) citation in the answer against actual retrieved chunks.
    Returns a list of {citation, file, page, verified, reason}."""
    results = []

    # build a lookup of what we actually have
    available = {}  # (source, page_str) -> chunk text snippet
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown").lower()
        pg = normalize_page(meta)
        available[(src, pg)] = d.page_content[:200]

    # also keep a set of just the filenames we have
    known_files = set(src for (src, _) in available.keys())

    for m in _CITE_RE.finditer(answer_text):
        cited_file = m.group(1).strip()
        cited_page = m.group(2).strip()
        citation_str = f"({cited_file} p.{cited_page})"

        # check
        if cited_file.lower() not in known_files:
            results.append({
                "citation": citation_str,
                "file": cited_file,
                "page": cited_page,
                "verified": False,
                "reason": "File not in retrieved documents",
            })
        elif (cited_file.lower(), cited_page) in available:
            results.append({
                "citation": citation_str,
                "file": cited_file,
                "page": cited_page,
                "verified": True,
                "reason": "Matches retrieved chunk",
            })
        else:
            # file exists but page doesn't match any retrieved chunk
            # could be a real page we just didn't retrieve, or could be hallucinated
            results.append({
                "citation": citation_str,
                "file": cited_file,
                "page": cited_page,
                "verified": False,
                "reason": f"Page {cited_page} not in retrieved chunks (may exist in full document)",
            })

    return results


# ---------- cross-document evidence synthesis ----------
# groups retrieved chunks by source document and identifies where
# different documents agree or cover the same topic.
# useful when you have multiple reports about the same fault zone / region.

def cross_document_synthesis(docs):
    """Group evidence by source and find overlapping topics.
    Returns {sources: {src: [{page, snippet}]}, overlap_topics: []}."""
    by_source = {}
    for d in docs[:MAX_CONTEXT_CHUNKS]:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        pg = normalize_page(meta)
        text = " ".join(d.page_content.split())[:300]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append({"page": pg, "snippet": text})

    # find topics that appear in multiple sources
    # simple approach: check which hazard keywords appear in which sources
    topic_sources = {}  # keyword -> set of sources
    for d in docs[:MAX_CONTEXT_CHUNKS]:
        txt = d.page_content.lower()
        src = (d.metadata or {}).get("source", "unknown")
        for htype, kws in HAZARD_KW.items():
            for kw in kws:
                if kw in txt:
                    if htype not in topic_sources:
                        topic_sources[htype] = set()
                    topic_sources[htype].add(src)
                    break

    # topics covered by 2+ sources = interesting overlaps
    overlaps = []
    for topic, srcs in topic_sources.items():
        if len(srcs) >= 2:
            overlaps.append({
                "topic": topic,
                "sources": sorted(srcs),
                "source_count": len(srcs),
            })

    overlaps.sort(key=lambda x: x["source_count"], reverse=True)

    return {
        "by_source": by_source,
        "overlaps": overlaps,
    }


# ---------- RAG vs plain LLM evaluation ----------

EVAL_QUESTIONS = [
    "What are the main types of geohazards covered in the documents?",
    "What triggers rainfall-induced landslides?",
    "What monitoring methods are used for slope stability?",
    "How is seismic hazard assessed for a given site?",
    "What mitigation measures are recommended for flood risk?",
    "What is the role of InSAR in volcanic monitoring?",
    "How is susceptibility mapping performed for landslides?",
    "What factors control tsunami inundation extent?",
]

_PLAIN_PROMPT = """You are a Geohazard Management assistant.
Answer using only your general knowledge. If unsure, say so.

QUESTION:
{question}"""


def run_evaluation(llm, vs, questions=None):
    questions = questions or EVAL_QUESTIONS
    results = []
    for q in questions:
        rag = answer_question(llm=llm, vs=vs, question=q)
        pr = llm.invoke(_PLAIN_PROMPT.format(question=q))
        pt = ""
        if hasattr(pr, "content"):
            pt = pr.content or ""
        else:
            pt = str(pr)
        pt = pt.strip()
        rag_ok = "NOT ENOUGH EVIDENCE" not in rag["answer"].upper()
        hedges = ["i'm not sure", "i don't know", "i cannot", "i can't",
                  "not certain", "unclear", "no information"]
        plain_ok = not any(h in pt.lower() for h in hedges)
        results.append({
            "question": q, "rag_answer": rag["answer"], "plain_answer": pt,
            "rag_confidence": rag["confidence"], "rag_sources": rag["chosen_sources"],
            "rag_evidence_n": len(rag["evidence"]), "rag_grounded": rag_ok,
            "plain_grounded": plain_ok, "rag_len": len(rag["answer"]), "plain_len": len(pt),
        })
    return results


def eval_summary(results):
    n = len(results)
    if n == 0:
        return {}
    return {
        "total": n,
        "rag_grounded_pct": round(sum(1 for r in results if r["rag_grounded"]) / n * 100, 1),
        "plain_grounded_pct": round(sum(1 for r in results if r["plain_grounded"]) / n * 100, 1),
        "rag_avg_len": round(sum(r["rag_len"] for r in results) / n),
        "plain_avg_len": round(sum(r["plain_len"] for r in results) / n),
        "rag_high_conf_pct": round(sum(1 for r in results if r["rag_confidence"] == "High") / n * 100, 1),
        "rag_avg_evidence": round(sum(r["rag_evidence_n"] for r in results) / n, 1),
        "rag_with_sources_pct": round(sum(1 for r in results if len(r["rag_sources"]) > 0) / n * 100, 1),
    }


# ---------- multi-hazard comparison ----------

_FOCUS = {
    "Landslide": "Focus on landslides, slope stability, mass movements.",
    "Earthquake": "Focus on seismic hazards, earthquakes, faults, ground motion.",
    "Flood": "Focus on flood hazards, inundation, discharge, floodplains.",
    "Tsunami": "Focus on tsunami hazards, coastal inundation, run-up.",
    "Volcanic": "Focus on volcanic hazards, eruptions, lava, lahars, deformation.",
    "Subsidence": "Focus on subsidence, sinkholes, karst, settlement.",
    "Erosion": "Focus on erosion, coastal retreat, gullies, sediment transport.",
}


def compare_hazards(llm, vs, question, hazard_types, cite=False):
    out = []
    for h in hazard_types:
        q = question + " " + _FOCUS.get(h, "")
        r = answer_question(llm=llm, vs=vs, question=q, cite=cite, mode="brief")
        out.append({
            "hazard_type": h, "answer": r["answer"], "confidence": r["confidence"],
            "sources": r["chosen_sources"], "risk_level": r.get("risk_level", "Undetermined"),
            "evidence": r["evidence"],
        })
    return out


# ---------- early warning bulletin ----------

_BULLETIN_TMPL = """You are a geohazard early warning specialist.

Using ONLY the context below, write a PUBLIC SAFETY BULLETIN.
Keep it understandable for non-specialists.

FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOHAZARD ADVISORY BULLETIN
Date: {date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━

HAZARD TYPE: [from context]
RISK LEVEL: [Critical/High/Medium/Low]
AFFECTED AREA: [from context, or "Not specified"]

SITUATION SUMMARY:
[2-3 sentences]

EXPECTED IMPACTS:
[bullets]

RECOMMENDED ACTIONS:
- For authorities: [from context]
- For residents: [from context]
- For emergency services: [from context]

MONITORING STATUS:
[from context]

SOURCE DOCUMENTS:
[list PDFs used]
━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULES:
1) ONLY use provided context. Do NOT invent.
2) If info missing, write "Not available in current documents"
3) Simple actionable language.

QUESTION:
{question}

CONTEXT:
{context}"""


def generate_bulletin(llm, vs, question, forced_pdf=None, last_sources=None):
    q2 = expand_query_if_needed(question)
    if forced_pdf:
        retriever = make_retriever(vs, source_allowlist=[forced_pdf])
        docs = retrieve_docs(retriever, q2)
        chosen_sources = [forced_pdf]
    else:
        docs = []
        chosen_sources = []
        if FOLLOWUP_USE_LAST_SOURCES and last_sources:
            ret = make_retriever(vs, source_allowlist=last_sources)
            dl = retrieve_docs(ret, q2)
            if dl and len(dl) >= MIN_SOURCES:
                docs = dl
                chosen_sources = list(last_sources)
        if not docs:
            ret_a = make_retriever(vs)
            docs_a = retrieve_docs(ret_a, q2)
            counts = tally_sources(docs_a)
            chosen_sources = choose_top_sources(counts, AUTO_FILTER_TOP_SOURCES, AUTO_FILTER_MIN_HITS)
            if chosen_sources:
                docs = retrieve_docs(make_retriever(vs, chosen_sources), q2)
            else:
                docs = docs_a
    if not docs or len(docs) < MIN_SOURCES:
        return {
            "bulletin": "INSUFFICIENT EVIDENCE — cannot generate bulletin.",
            "confidence": "Low", "chosen_sources": chosen_sources,
            "evidence": [], "hazard_tags": _tag_hazards(question, "", docs),
            "risk_level": "Undetermined",
        }
    ctx = build_context(docs, MAX_CONTEXT_CHUNKS, cite=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = _BULLETIN_TMPL.format(date=now, question=question, context=ctx)
    resp = llm.invoke(prompt)
    bt = ""
    if hasattr(resp, "content"):
        bt = resp.content or ""
    else:
        bt = str(resp)
    return {
        "bulletin": bt.strip(),
        "confidence": confidence_from_retrieval(docs, chosen_sources),
        "chosen_sources": chosen_sources,
        "evidence": make_evidence_snippets(docs, EVIDENCE_SNIPPETS),
        "hazard_tags": _tag_hazards(question, bt, docs),
        "risk_level": _score_risk(question, bt, docs),
    }


# ---------- USGS live feed ----------

USGS_FEEDS = {
    "Past hour M2.5+": "2.5_hour.geojson",
    "Past day M2.5+": "2.5_day.geojson",
    "Past day M4.5+": "4.5_day.geojson",
    "Past week M4.5+": "4.5_week.geojson",
    "Past month significant": "significant_month.geojson",
}

def fetch_usgs(feed_key="Past day M4.5+", timeout=10):
    import requests
    if feed_key not in USGS_FEEDS:
        return []
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{USGS_FEEDS[feed_key]}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    quakes = []
    for f in data.get("features", []):
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [None, None, None]
        if coords[0] is None or coords[1] is None or props.get("mag") is None:
            continue
        quakes.append({
            "time": datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc),
            "mag": float(props["mag"]),
            "place": str(props.get("place", "")),
            "lat": float(coords[1]), "lon": float(coords[0]),
            "depth": float(coords[2]) if coords[2] is not None else 0,
            "url": str(props.get("url", "")),
        })
    return quakes


# ---------- place name + hazard zone extraction ----------

_PLACES = {
    "port-au-prince": (18.5392, -72.3350), "haiti": (18.9712, -72.2852),
    "kathmandu": (27.7172, 85.3240), "nepal": (28.3949, 84.1240),
    "istanbul": (41.0082, 28.9784), "san francisco": (37.7749, -122.4194),
    "los angeles": (34.0522, -118.2437), "california": (36.7783, -119.4179),
    "tokyo": (35.6762, 139.6503), "japan": (36.2048, 138.2529),
    "indonesia": (-0.7893, 113.9213), "chile": (-35.6751, -71.5430),
    "new zealand": (-40.9006, 174.8860), "italy": (41.8719, 12.5674),
    "vesuvius": (40.8210, 14.4260), "etna": (37.7510, 14.9934),
    "iceland": (64.9631, -19.0208), "alaska": (64.2008, -152.4937),
    "washington": (47.7511, -120.7401), "puerto rico": (18.2208, -66.5901),
    "aachen": (50.7753, 6.0839), "lower rhine": (51.4500, 6.7500),
    "eifel": (50.3500, 6.9833), "himalaya": (28.5983, 83.9311),
    "andes": (-32.6532, -70.0109), "philippines": (12.8797, 121.7740),
    "taiwan": (23.6978, 120.9605), "mexico city": (19.4326, -99.1332),
    "peru": (-9.1900, -75.0152), "china": (35.8617, 104.1954),
    "sichuan": (30.5728, 104.0668), "greece": (39.0742, 21.8243),
    "turkey": (38.9637, 35.2433), "iran": (32.4279, 53.6880),
    "pakistan": (30.3753, 69.3451), "bangladesh": (23.6850, 90.3563),
    "mediterranean": (35.0, 18.0), "caribbean": (15.0, -75.0),
    "pacific northwest": (46.0, -123.0), "ring of fire": (0.0, 160.0),
    "alpine fault": (-43.5, 170.5), "san andreas": (36.0, -120.5),
    "cascadia": (45.0, -124.0), "sumatra": (0.5897, 101.3431),
}


def extract_place_names(docs):
    found = []
    seen = set()
    for d in docs[:MAX_CONTEXT_CHUNKS]:
        txt = d.page_content.lower()
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        pg = normalize_page(meta)
        for place, (lat, lon) in _PLACES.items():
            if place in txt and place not in seen:
                seen.add(place)
                idx = txt.find(place)
                s = max(0, idx - 30)
                e = min(len(txt), idx + len(place) + 50)
                ctx = d.page_content[s:e].strip()
                found.append({"place": place.title(), "lat": lat, "lon": lon,
                              "source": src, "page": pg, "context": ctx})
    return found


def extract_hazard_zones(docs):
    places = extract_place_names(docs)
    zones = []
    for p in places:
        ctx = p.get("context", "").lower()
        local_risk = "Unknown"
        for level, (w, terms) in _RISK_KW.items():
            for t in terms:
                if t in ctx:
                    local_risk = level
                    break
            if local_risk != "Unknown":
                break
        local_hazards = []
        for htype, kws in HAZARD_KW.items():
            for kw in kws:
                if kw in ctx:
                    local_hazards.append(htype)
                    break
        zones.append({
            "place": p["place"], "lat": p["lat"], "lon": p["lon"],
            "risk": local_risk, "hazards": local_hazards,
            "source": p["source"], "page": p["page"], "context": p["context"],
        })
    return zones


# ---------- location-aware query ----------

_LOCATION_PROMPT = """You are a Geohazard Management assistant.

The user is asking about geohazards near coordinates {lat:.4f}, {lon:.4f}.

Using ONLY the provided context, answer what geohazards are relevant to
this area. If the context mentions this region or nearby areas, describe:
- What hazard types affect this area
- Known risk factors
- Recommended monitoring or mitigation

If the context has no information about this area, say so.

CONTEXT:
{context}"""


def query_location(llm, vs, lat, lon):
    q = f"What geohazards affect the area near {lat:.2f} latitude {lon:.2f} longitude? Risk factors and monitoring?"
    q2 = expand_query_if_needed(q)
    retriever = make_retriever(vs)
    docs = retrieve_docs(retriever, q2)
    if not docs:
        return {"answer": "No relevant documents found.", "sources": [], "hazard_tags": [], "risk_level": "Undetermined"}
    context = build_context(docs, MAX_CONTEXT_CHUNKS, cite=False)
    prompt = _LOCATION_PROMPT.format(lat=lat, lon=lon, context=context)
    resp = llm.invoke(prompt)
    at = ""
    if hasattr(resp, "content"):
        at = resp.content or ""
    else:
        at = str(resp)
    return {
        "answer": at.strip(),
        "sources": list(set((d.metadata or {}).get("source", "?") for d in docs[:5])),
        "hazard_tags": _tag_hazards(q, at, docs),
        "risk_level": _score_risk(q, at, docs),
    }