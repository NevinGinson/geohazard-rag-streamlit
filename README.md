# Geohazard Management RAG Assistant

A Retrieval-Augmented Generation (RAG) system for structured geohazard analysis and decision support.  
Built with Streamlit, Chroma vector database, and Groq-hosted large language models.

This project integrates semantic document retrieval with domain-specific hazard documentation to support operational, technical, and planning-level workflows in geohazard management.

---

## Live Application

Streamlit Deployment:  
https://geohazard-rag-app-ymnjwtmwtemrv7tvsmfqlq.streamlit.app/ 



---

## Overview

This system:

- Ingests geohazard-related PDF reports
- Builds a semantic vector index using Chroma
- Automatically routes queries to the most relevant document(s)
- Generates structured responses tailored to engineering workflows
- Supports optional evidence grounding and citation control

The architecture prioritizes controlled retrieval, document routing, and structured output generation over generic conversational responses.

---

## System Architecture

User Query  
→ Streamlit Interface (`app.py`)  
→ RAG Engine (`geo_core.py`)  
→ Chroma Vector Retrieval  
→ Automatic Source Selection (majority-hit routing)  
→ Groq LLM Response  

The routing mechanism performs:

1. Broad semantic retrieval across indexed documents  
2. Majority source selection  
3. Focused re-retrieval from selected document(s)  

This reduces cross-document drift and improves domain grounding.

---

## Key Features

- Local PDF ingestion (PyMuPDF with PDFMiner fallback)
- Chroma vector database with persistent indexing
- Automatic PDF routing via majority-hit filtering
- Structured output modes (operational, concise, deep analysis)
- Optional strict grounding mode
- Evidence snippet display
- Modular separation between UI and retrieval core

---

## Output Modes

The interface supports structured response control:

- `fast:` concise response
- `brief:` bullet-point summary
- `ops:` operational checklist (Triggers, Monitoring, Actions, Mitigation, Stakeholders)
- `deep:` detailed technical explanation
- `ask:` clarification-driven interaction

---

## Example Queries