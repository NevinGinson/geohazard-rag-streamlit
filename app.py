import streamlit as st
import pandas as pd
from typing import Optional

from geo_core import (
    init_llm, init_vectorstore, answer_question,
    DOCS_DIR, CHROMA_DIR, HAZARD_KW,
    compare_hazards, generate_bulletin,
    run_evaluation, eval_summary, EVAL_QUESTIONS,
    fetch_usgs, USGS_FEEDS,
    extract_hazard_zones, query_location,
    cross_document_synthesis,
    make_retriever, retrieve_docs,
)

st.set_page_config(page_title="Geohazard RAG", layout="wide")

with st.sidebar:
    st.header("Settings")
    rebuild = st.checkbox("Rebuild index (slow)", value=False)
    strict = st.checkbox("Strict mode (refuse if weak evidence)", value=False)
    cite = st.checkbox("Citations ON (page refs)", value=False)
    mode = st.selectbox("Answer mode", ["default", "fast", "brief", "ops", "deep", "ask"], index=0)
    st.divider()
    st.write("📂 **DOCS_DIR**:", DOCS_DIR)
    st.write("💾 **CHROMA_DIR**:", CHROMA_DIR)
    st.divider()
    st.subheader("PDF Routing")
    routing = st.radio("Use PDFs", ["Auto-pick best PDF(s)", "Force one PDF"], index=0)


@st.cache_resource(show_spinner=True)
def load_resources(do_rebuild):
    l = init_llm()
    v, names = init_vectorstore(rebuild=do_rebuild)
    return l, v, names

try:
    with st.spinner("Loading LLM + Vector DB..."):
        llm, vs, pdf_names = load_resources(rebuild)
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

forced_pdf: Optional[str] = None
if routing == "Force one PDF":
    forced_pdf = st.selectbox("Choose PDF", pdf_names)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Assistant", " Monitoring & GIS", " Compare Hazards",
    " Bulletin", " Evaluation", " About"
])


# =================== TAB 1: Assistant ===================
with tab1:
    st.title("Geohazard Management Assistant")
    st.caption("Local PDFs → Chroma → Groq LLM. Answers + hazard tags + risk level + parameters.")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_q = st.text_area("Your question", height=90,
            placeholder="Example: ops: For landslides, give triggers, monitoring, immediate actions, mitigation.")
    with col2:
        st.write("Quick tests")
        if st.button("Test: Landslide ops"):
            user_q = "ops: For landslides, give triggers, monitoring, immediate actions, and mitigation."
        if st.button("Test: Hazard list fast"):
            user_q = "fast: List the main geologic hazards mentioned in the PDFs."
        if st.button("Test: Seismic deep"):
            user_q = "deep: How is seismic hazard assessed? Include PGA, magnitude, return periods."

    ask_btn = st.button("Ask", type="primary", use_container_width=True)

    def parse_mode_prefix(q, default_mode):
        q = (q or "").strip()
        for m in ["fast", "brief", "ops", "deep", "ask", "default"]:
            if q.lower().startswith(m + ":"):
                return m, q.split(":", 1)[1].strip()
        return default_mode, q

    if ask_btn and user_q.strip():
        run_mode, clean_q = parse_mode_prefix(user_q, mode)
        with st.spinner("Retrieving + answering..."):
            result = answer_question(llm=llm, vs=vs, question=clean_q,
                strict=strict, cite=cite, mode=run_mode,
                forced_pdf=forced_pdf, last_sources=st.session_state.last_sources)

        if result.get("chosen_sources"):
            st.session_state.last_sources = result["chosen_sources"]

        # cross-doc synthesis — compute now, store as plain dict (not Document objects)
        docs_used = result.get("docs_used", [])
        src_names = set((d.metadata or {}).get("source", "") for d in docs_used)
        synth_data = None
        if len(src_names) > 1:
            try:
                synth_data = cross_document_synthesis(docs_used)
            except Exception:
                synth_data = None

        st.session_state.history.append({
            "q": user_q.strip(), "mode": run_mode,
            "forced_pdf": forced_pdf,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", "?"),
            "chosen_sources": result.get("chosen_sources", []),
            "evidence": result.get("evidence", []),
            "hazard_tags": result.get("hazard_tags", []),
            "risk_level": result.get("risk_level", "Undetermined"),
            "locations": result.get("locations", []),
            "parameters": result.get("parameters", []),
            "citation_checks": result.get("citation_checks", []),
            "cross_doc": synth_data,
        })

    st.subheader("Conversation")
    for item in reversed(st.session_state.history):
        st.markdown(f"**You:** {item['q']}")

        meta = f"Confidence: **{item['confidence']}**"
        rl = item.get("risk_level", "")
        if rl and rl not in ("Undetermined", "Unknown"):
            meta += f" | Risk: **{rl}**"
        ht = item.get("hazard_tags", [])
        if ht:
            meta += f" | Hazards: **{', '.join(ht)}**"
        fp = item.get("forced_pdf")
        cs = item.get("chosen_sources", [])
        if fp:
            meta += f" | Forced PDF: **{fp}**"
        elif cs:
            meta += f" | Auto PDFs: **{', '.join(cs)}**"
        st.caption(meta)

        st.markdown("**Assistant:**")
        st.write(item["answer"])

        # evidence
        ev = item.get("evidence", [])
        if ev:
            with st.expander(f"Evidence snippets ({len(ev)})"):
                for e in ev:
                    st.write(e)

        # extracted parameters
        params = item.get("parameters", [])
        if params:
            with st.expander(f"📐 {len(params)} hazard parameter(s) extracted"):
                try:
                    pf = pd.DataFrame(params)
                    st.dataframe(pf[["param", "value", "source", "page", "context"]],
                                 hide_index=True, use_container_width=True)
                except Exception:
                    for p in params:
                        st.write(f"- {p.get('param')}: {p.get('value')} ({p.get('source')} p.{p.get('page')})")

        # citation verification
        cc = item.get("citation_checks", [])
        if cc:
            with st.expander(f"✓ Citation verification ({len(cc)} checked)"):
                for c in cc:
                    icon = "✅" if c.get("verified") else "❌"
                    st.write(f"{icon} {c.get('citation', '?')} — {c.get('reason', '')}")

        # cross-document synthesis
        cd = item.get("cross_doc")
        if cd and cd.get("overlaps"):
            with st.expander("📚 Cross-document evidence"):
                st.write("**Topics covered by multiple sources:**")
                for o in cd["overlaps"]:
                    st.write(f"- **{o['topic']}** — found in {o['source_count']} documents: {', '.join(o['sources'])}")
                for src, chunks in cd.get("by_source", {}).items():
                    st.caption(f"**{src}** ({len(chunks)} chunks)")

        # map for extracted coordinates
        locs = item.get("locations", [])
        if locs:
            with st.expander(f"📍 {len(locs)} coordinate(s) extracted"):
                ldf = pd.DataFrame(locs).rename(columns={"lat": "latitude", "lon": "longitude"})
                st.map(ldf[["latitude", "longitude"]])
                st.dataframe(pd.DataFrame(locs)[["lat", "lon", "source", "page", "context"]],
                             hide_index=True, use_container_width=True)

        st.divider()


# =================== TAB 2: Monitoring & GIS ===================
with tab2:
    st.title("Monitoring & GIS")
    st.caption("Live earthquake feed, location-aware queries, and hazard zone mapping.")

    gis1, gis2, gis3 = st.tabs([" Live Earthquakes", "Query", " Hazard Zones"])

    # ---- live earthquakes ----
    with gis1:
        st.subheader("Live USGS Seismicity")
        col_a, col_b = st.columns(2)
        with col_a:
            feed = st.selectbox("USGS feed", list(USGS_FEEDS.keys()), index=2)
        with col_b:
            min_mag = st.slider("Min magnitude", 0.0, 8.0, 2.5, 0.1)

        quakes = []
        try:
            with st.spinner("Fetching USGS data..."):
                quakes = fetch_usgs(feed)
            if quakes:
                quakes = [q for q in quakes if q["mag"] >= min_mag]
        except Exception as e:
            st.error(f"USGS fetch failed: {e}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Events", len(quakes))
        c2.metric("Max magnitude", f"{max(q['mag'] for q in quakes):.1f}" if quakes else "—")
        c3.metric("Feed", feed)

        if not quakes:
            st.info("No earthquakes in this selection.")
        else:
            qdf = pd.DataFrame(quakes)
            qdf_map = qdf.rename(columns={"lat": "latitude", "lon": "longitude"})
            st.map(qdf_map[["latitude", "longitude"]], zoom=1)

            with st.expander("Data table"):
                show_df = qdf.copy()
                show_df["time"] = show_df["time"].apply(lambda t: t.strftime("%Y-%m-%d %H:%M"))
                st.dataframe(show_df[["time", "mag", "depth", "place"]].sort_values("mag", ascending=False),
                             hide_index=True, use_container_width=True)

        st.caption("Source: USGS Earthquake Hazards Program")

    # ---- click to query ----
    with gis2:
        st.subheader("Location-Aware Query")
        st.write("Enter coordinates and ask the RAG about hazards near that point.")

        col_lt, col_ln = st.columns(2)
        with col_lt:
            qlat = st.number_input("Latitude", value=50.78, min_value=-90.0, max_value=90.0, format="%.4f")
        with col_ln:
            qlon = st.number_input("Longitude", value=6.08, min_value=-180.0, max_value=180.0, format="%.4f")

        # show the point on a map
        pt_df = pd.DataFrame([{"latitude": qlat, "longitude": qlon}])
        st.map(pt_df, zoom=7)

        if st.button("Query this location", type="primary", use_container_width=True):
            with st.spinner(f"Asking RAG about {qlat:.2f}, {qlon:.2f}..."):
                try:
                    lr = query_location(llm, vs, qlat, qlon)
                except Exception as e:
                    lr = {"answer": f"Error: {e}", "hazard_tags": [], "risk_level": "?", "sources": []}
            st.markdown("**RAG response:**")
            lm = f"Hazards: **{', '.join(lr.get('hazard_tags', []))}**"
            if lr.get("risk_level"):
                lm += f" | Risk: **{lr['risk_level']}**"
            lm += f" | Sources: **{', '.join(lr.get('sources', []))}**"
            st.caption(lm)
            st.write(lr["answer"])

    # ---- hazard zones ----
    with gis3:
        st.subheader("Hazard Zone Map")
        st.write("Scans your PDFs for place names and shows their hazard context on a map.")

        if st.button("Scan PDFs for hazard zones", type="primary", use_container_width=True):
            zones = []
            try:
                with st.spinner("Scanning documents..."):
                    ret = make_retriever(vs)
                    broad_docs = retrieve_docs(ret, "geohazard risk area location region")
                    zones = extract_hazard_zones(broad_docs)
            except Exception as e:
                st.error(f"Scan failed: {e}")

            if not zones:
                st.warning("No place names found. Try adding PDFs that mention specific locations.")
            else:
                st.write(f"Found **{len(zones)}** locations.")

                # show on map
                zdf_map = pd.DataFrame(zones).rename(columns={"lat": "latitude", "lon": "longitude"})
                st.map(zdf_map[["latitude", "longitude"]], zoom=1)

                st.write("**Legend:** 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low | ⚪ Unknown")
                with st.expander("Zone details"):
                    zdf = pd.DataFrame(zones)
                    zdf["hazards"] = zdf["hazards"].apply(lambda x: ", ".join(x) if x else "—")
                    st.dataframe(zdf[["place", "risk", "hazards", "source", "page"]],
                                 hide_index=True, use_container_width=True)


# =================== TAB 3: Compare ===================
with tab3:
    st.title("Multi-Hazard Comparison")
    st.caption("Same question, different hazard lenses — side by side.")

    cmp_q = st.text_input("Question", placeholder="e.g. What monitoring methods are recommended?")
    available = list(HAZARD_KW.keys())
    selected = st.multiselect("Hazard types to compare", available, default=["Landslide", "Earthquake", "Flood"])

    if st.button("Run comparison", type="primary", use_container_width=True) and cmp_q.strip() and selected:
        with st.spinner(f"Comparing {len(selected)} hazard types..."):
            entries = compare_hazards(llm, vs, cmp_q, selected, cite=cite)
        for e in entries:
            st.markdown(f"#### {e['hazard_type']}")
            st.caption(f"Confidence: **{e['confidence']}** | Risk: **{e['risk_level']}** | PDFs: {', '.join(e['sources']) or 'auto'}")
            st.write(e["answer"])
            if e["evidence"]:
                with st.expander("Evidence"):
                    for ev in e["evidence"]:
                        st.write(ev)
            st.divider()


# =================== TAB 4: Bulletin ===================
with tab4:
    st.title("Early Warning Bulletin")
    st.caption("Structured public safety advisories from the PDFs.")

    bull_q = st.text_area("Describe the hazard scenario", height=90,
        placeholder="e.g. Heavy rainfall forecast for region with known landslide susceptibility")
    if st.button("Generate bulletin", type="primary", use_container_width=True) and bull_q.strip():
        with st.spinner("Generating bulletin..."):
            result = generate_bulletin(llm, vs, bull_q,
                forced_pdf=forced_pdf, last_sources=st.session_state.last_sources)
        tags = result.get("hazard_tags", [])
        st.caption(f"Confidence: **{result.get('confidence', '?')}** | Risk: **{result.get('risk_level', '?')}** | Hazards: {', '.join(tags)}")
        st.code(result.get("bulletin", ""), language=None)
        bev = result.get("evidence", [])
        if bev:
            with st.expander("Supporting evidence"):
                for ev in bev:
                    st.write(ev)


# =================== TAB 5: Evaluation ===================
with tab5:
    st.title("RAG vs Plain LLM Evaluation")
    st.caption("Same questions through RAG (grounded) and plain LLM (no docs).")

    st.write("**Test questions:**")
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        st.write(f"{i}. {q}")

    if st.button("Run evaluation (takes ~2 min)", type="primary", use_container_width=True):
        with st.spinner("Running evaluation..."):
            results = run_evaluation(llm, vs)
            summary = eval_summary(results)

        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RAG grounded %", f"{summary.get('rag_grounded_pct', 0)}%")
        c2.metric("Plain grounded %", f"{summary.get('plain_grounded_pct', 0)}%")
        c3.metric("RAG high conf %", f"{summary.get('rag_high_conf_pct', 0)}%")
        c4.metric("RAG w/ sources %", f"{summary.get('rag_with_sources_pct', 0)}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("RAG avg length", f"{summary.get('rag_avg_len', 0)} chars")
        c2.metric("Plain avg length", f"{summary.get('plain_avg_len', 0)} chars")
        c3.metric("Avg evidence", f"{summary.get('rag_avg_evidence', 0)}")

        st.subheader("Per-question breakdown")
        for r in results:
            with st.expander(r["question"]):
                left, right = st.columns(2)
                with left:
                    st.write(f"**RAG** (confidence: {r['rag_confidence']})")
                    ra = r.get("rag_answer", "")
                    st.write(ra[:500] + ("..." if len(ra) > 500 else ""))
                    st.caption(f"Sources: {', '.join(r.get('rag_sources', [])) or 'none'} | Evidence: {r.get('rag_evidence_n', 0)}")
                with right:
                    tag = "grounded" if r.get("plain_grounded") else "hedged/uncertain"
                    st.write(f"**Plain LLM** ({tag})")
                    pa = r.get("plain_answer", "")
                    st.write(pa[:500] + ("..." if len(pa) > 500 else ""))


# =================== TAB 6: About ===================
with tab6:
    st.title("About / Methodology")

    st.markdown("""
### Architecture

```
PDFs → PyMuPDF → chunks → embeddings (MiniLM) → Chroma
  → 2-stage routing → Groq LLM → answer
  → hazard tagger → risk scorer → parameter extractor
  → citation verifier → cross-doc synthesis → GIS
```

### Key features

**Two-stage majority-hit routing.** Retrieves broadly, counts hits per PDF,
re-retrieves from the dominant source(s). Prevents cross-document dilution.

**Structured parameter extraction.** Pulls actual magnitudes (M7.0), PGA
values (0.3g), recurrence intervals (475 yr), fault slip rates (2 mm/yr),
slope angles, depths from the text.

**Citation verification.** Checks whether each (Report.pdf p.12) citation
matches a retrieved chunk. Catches hallucinated references.

**Cross-document evidence synthesis.** When multiple PDFs contribute to an
answer, identifies which topics overlap between sources.

**7-type hazard classification.** Keyword-based scan across landslide,
earthquake, flood, tsunami, volcanic, subsidence, erosion.

**4-level risk scoring.** Weighted keyword matching — Critical, High,
Medium, Low.

**Live USGS earthquake feed.** Real data from USGS GeoJSON feeds.

**Location-aware query.** Enter coordinates, ask the RAG about hazards
at that location.

**Hazard zone overlay.** Place-name gazetteer scans PDFs and maps
locations with risk context.

**Multi-hazard comparison.** Same question with hazard-specific query focus
for side-by-side analysis.

**Early warning bulletin.** Dedicated prompt for structured public safety
advisories.

**RAG vs plain LLM evaluation.** Benchmarks grounding, confidence, and
source traceability.

### Limitations

- Parameter extraction is regex-based — misses unusual formats
- Citation verification only checks retrieved chunks, not full PDFs
- Risk scoring is not calibrated against ISO 31010
- Place-name gazetteer is limited (~40 entries)
- Tables/figures in PDFs don't embed well
- English only
    """)