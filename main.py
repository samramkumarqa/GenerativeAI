"""Streamlit UI for the AI Test Case Generator.

Flow: load a BRD -> extract requirements (review/edit) -> generate test cases -> download Excel.
"""

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tcgen import config
from tcgen.export import build_workbook
from tcgen.llm import GroqLLM, is_bad_generation
from tcgen.loaders import DocumentError, read_upload, read_url
from tcgen.pipeline import (
    extract_requirements,
    generate_test_cases,
    missing_ids,
    traceability,
)
from tcgen.rag import DocumentIndex, FastEmbedder
from tcgen.schemas import Requirement

SAMPLE_BRD = Path(__file__).parent / "samples" / "sample_brd.pdf"
load_dotenv()  # lets local runs read GROQ_API_KEY from .env

st.set_page_config(page_title="AI Test Case Generator", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------- services

def _configured_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets.toml locally
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return FastEmbedder()


def get_index(text: str) -> DocumentIndex:
    digest = hashlib.sha256(text.encode()).hexdigest()
    if st.session_state.get("index_digest") != digest:
        st.session_state.index = DocumentIndex(text, get_embedder())
        st.session_state.index_digest = digest
    return st.session_state.index


def reset_from(stage: str) -> None:
    """Clear everything downstream of a stage when its input changes."""
    keys = {
        "document": ["doc_text", "doc_name", "requirements", "test_cases"],
        "requirements": ["requirements", "test_cases"],
        "test_cases": ["test_cases"],
    }[stage]
    for k in keys:
        st.session_state.pop(k, None)


def progress_bar(label: str):
    bar = st.progress(0.0, text=label)
    return lambda frac, msg: bar.progress(min(max(frac, 0.0), 1.0), text=msg)


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    key = _configured_key()
    if key:
        st.success("Groq API key configured", icon="🔑")
    else:
        key = st.text_input(
            "Groq API key",
            type="password",
            help="Free at console.groq.com. Used only for this session, never stored.",
        )
    model = st.selectbox(
        "Model", list(config.MODELS), format_func=lambda m: config.MODELS[m]
    )
    st.divider()
    st.caption(
        "Documents are sent to Groq for processing. Don't upload confidential BRDs "
        "to the public demo."
    )
    st.caption("Built with Groq · FAISS · fastembed · Streamlit")


# ---------------------------------------------------------------- header

st.title("🧪 AI Test Case Generator")
st.markdown(
    "Turn a Business Requirements Document into **traceable test cases**. "
    "Every requirement is extracted first so you can review it, then test cases are "
    "generated for each one from the passages of the BRD that describe it (RAG)."
)


# ---------------------------------------------------------------- step 1: document

st.subheader("1 · Load your BRD")
tab_file, tab_url, tab_sample = st.tabs(["Upload file", "From URL", "Try a sample"])

with tab_file:
    upload = st.file_uploader("PDF, Word, TXT or Markdown", type=["pdf", "docx", "txt", "md"])
    if upload and st.button("Use this file", key="use_file"):
        try:
            text = read_upload(upload.name, upload.getvalue())
            reset_from("document")
            st.session_state.doc_text, st.session_state.doc_name = text, upload.name
        except DocumentError as e:
            st.error(str(e))

with tab_url:
    url = st.text_input("Public URL of the BRD (web page or PDF)")
    if url and st.button("Fetch", key="use_url"):
        try:
            with st.spinner("Fetching…"):
                text = read_url(url)
            reset_from("document")
            st.session_state.doc_text, st.session_state.doc_name = text, url
        except DocumentError as e:
            st.error(str(e))

with tab_sample:
    st.write(
        "A short fictional BRD for an online grocery checkout, with numbered "
        "functional and non-functional requirements."
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Use sample BRD", key="use_sample"):
        reset_from("document")
        st.session_state.doc_text = read_upload(SAMPLE_BRD.name, SAMPLE_BRD.read_bytes())
        st.session_state.doc_name = SAMPLE_BRD.name
    c2.download_button("Download sample PDF", SAMPLE_BRD.read_bytes(), SAMPLE_BRD.name)

doc_text = st.session_state.get("doc_text")
if not doc_text:
    st.stop()

st.info(
    f"**{st.session_state.doc_name}** · {len(doc_text):,} characters loaded", icon="📄"
)
with st.expander("Preview extracted text"):
    st.text(doc_text[:5000] + ("\n…" if len(doc_text) > 5000 else ""))

if not key:
    st.warning("Add a Groq API key in the sidebar to continue.")
    st.stop()

llm = GroqLLM(api_key=key, model=model)


def run_safely(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # surface API errors (bad key, rate limit, etc.) without a stack trace
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg.lower():
            st.error("Groq rejected the API key. Check it in the sidebar.")
        elif is_bad_generation(e):
            st.error("The model returned an incomplete answer three times. Click the button again, or switch to the 20B model.")
        elif "429" in msg or "rate" in msg.lower():
            st.error("Groq's rate limit was reached. Wait a minute, or switch to the 20B model.")
        else:
            st.error(f"Something went wrong: {msg}")
        st.stop()


# ---------------------------------------------------------------- step 2: requirements

st.subheader("2 · Review requirements")

if "requirements" not in st.session_state:
    if st.button("Extract requirements", type="primary"):
        reqs = run_safely(
            extract_requirements, doc_text, llm, progress=progress_bar("Reading the BRD…")
        )
        if not reqs:
            st.error("No testable requirements were found in this document.")
            st.stop()
        st.session_state.requirements = reqs
        st.rerun()
    st.stop()

reqs: list[Requirement] = st.session_state.requirements
st.caption(
    f"{len(reqs)} requirements found. Edit any text, untick ones to skip, "
    "or add rows for anything that was missed."
)
gaps = missing_ids(doc_text, reqs)
if gaps:
    st.warning(
        f"These IDs appear in the BRD but no requirement was extracted for them: "
        f"{', '.join(gaps)}. Add them as rows below if they're real requirements."
    )
req_df = pd.DataFrame(
    [{"Include": True, **r.model_dump(exclude={"source_excerpt"})} for r in reqs]
)
edited = st.data_editor(
    req_df,
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    column_config={
        "Include": st.column_config.CheckboxColumn(width="small", default=True),
        "id": st.column_config.TextColumn("ID", width="small", required=True),
        "title": st.column_config.TextColumn("Title", width="medium", required=True),
        "description": st.column_config.TextColumn("Description", width="large"),
        "category": st.column_config.SelectboxColumn(
            "Category", options=["Functional", "Non-Functional", "Business Rule"], default="Functional"
        ),
        "priority": st.column_config.SelectboxColumn(
            "Priority", options=["High", "Medium", "Low"], default="Medium"
        ),
    },
    key="req_editor",
)

excerpts = {r.id: r.source_excerpt for r in reqs}
selected = [
    Requirement(
        id=str(row["id"]).strip(),
        title=str(row["title"]).strip(),
        description=str(row.get("description") or "").strip(),
        category=row.get("category") or "Functional",
        priority=row.get("priority") or "Medium",
        source_excerpt=excerpts.get(str(row["id"]).strip(), ""),
    )
    for row in edited.to_dict("records")
    if row.get("Include", True) and str(row.get("id") or "").strip() and str(row.get("title") or "").strip()
]
dupes = {r.id for r in selected if [s.id for s in selected].count(r.id) > 1}
if dupes:
    st.error(f"Requirement IDs must be unique: {', '.join(sorted(dupes))}")
    st.stop()


# ---------------------------------------------------------------- step 3: generate

st.subheader("3 · Generate test cases")
c1, c2 = st.columns([3, 1])
types = c1.multiselect("Test types", config.TEST_TYPES, default=config.DEFAULT_TEST_TYPES)
per_req = c2.slider("Cases per requirement", 1, 6, 3)

calls = -(-len(selected) // config.GENERATION_BATCH_SIZE)
if st.button(
    f"Generate for {len(selected)} requirements",
    type="primary",
    disabled=not selected or not types,
):
    index = run_safely(get_index, doc_text)
    tcs = run_safely(
        generate_test_cases,
        selected,
        index,
        llm,
        types,
        per_req,
        progress=progress_bar(f"Generating ({calls} calls to Groq)…"),
    )
    st.session_state.test_cases = tcs
    st.session_state.generated_for = selected

if "test_cases" not in st.session_state:
    st.stop()


# ---------------------------------------------------------------- results

tcs = st.session_state.test_cases
used_reqs = st.session_state.generated_for
trace = traceability(used_reqs, tcs)
covered = sum(1 for t in trace if t["Coverage"] == "Covered")

st.subheader("Results")
m1, m2, m3 = st.columns(3)
m1.metric("Requirements", len(used_reqs))
m2.metric("Test cases", len(tcs))
m3.metric("Coverage", f"{covered / len(used_reqs):.0%}" if used_reqs else "–")

t_cases, t_trace = st.tabs(["Test cases", "Traceability matrix"])
with t_cases:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": tc.id,
                    "Req": tc.requirement_id,
                    "Title": tc.title,
                    "Type": tc.test_type,
                    "Priority": tc.priority,
                    "Preconditions": tc.preconditions,
                    "Steps": "\n".join(f"{i}. {s}" for i, s in enumerate(tc.steps, 1)),
                    "Test data": tc.test_data,
                    "Expected result": tc.expected_result,
                }
                for tc in tcs
            ]
        ),
        width="stretch",
        hide_index=True,
    )
with t_trace:
    st.dataframe(pd.DataFrame(trace), width="stretch", hide_index=True)

stem = Path(st.session_state.doc_name).stem if "://" not in st.session_state.doc_name else "brd"
st.download_button(
    "⬇️ Download Excel test pack",
    build_workbook(used_reqs, tcs, st.session_state.doc_name, model),
    file_name=f"{stem}_test_cases.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
st.caption("AI-generated. Review each test case before adding it to your suite.")
