"""Streamlit UI for the AI Test Case Generator.

Flow: load a BRD -> extract requirements (review/edit) -> generate test cases -> download Excel.
"""

import hashlib
import os
import re
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
    ready_to_export,
    review_summary,
    revise_test_cases,
    traceability,
)
from tcgen.rag import DocumentIndex, FastEmbedder
from tcgen.schemas import Requirement, TestCase

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
        "document": ["doc_text", "doc_name", "requirements", "test_cases", "superseded"],
        "requirements": ["requirements", "test_cases"],
        "test_cases": ["test_cases"],
    }[stage]
    for k in keys:
        st.session_state.pop(k, None)


def set_test_cases(tcs: list[TestCase]) -> None:
    """Replace the review set and remount the review table so it shows the new data."""
    st.session_state.test_cases = tcs
    st.session_state.review_version = st.session_state.get("review_version", 0) + 1


def tcs_to_df(tcs: list[TestCase]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Approve": tc.review_status == "Approved",
                "ID": tc.id,
                "Req": tc.requirement_id,
                "Rev": tc.revision,
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
    )


def _text(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def df_to_tcs(df: pd.DataFrame) -> list[TestCase]:
    """Read the review table back, including any inline edits the reviewer made."""
    out = []
    for row in df.to_dict("records"):
        steps = [re.sub(r"^\s*\d+[.)]\s*", "", line).strip() for line in _text(row["Steps"]).splitlines()]
        out.append(
            TestCase(
                id=_text(row["ID"]),
                requirement_id=_text(row["Req"]),
                revision=int(row["Rev"] or 1),
                title=_text(row["Title"]),
                test_type=_text(row["Type"]),
                priority=_text(row["Priority"]),
                preconditions=_text(row["Preconditions"]),
                steps=[s for s in steps if s],
                test_data=_text(row["Test data"]),
                expected_result=_text(row["Expected result"]),
                review_status="Approved" if bool(row["Approve"]) else "Pending",
            )
        )
    return out


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
    set_test_cases(tcs)
    st.session_state.generated_for = selected
    st.session_state.superseded = []

if "test_cases" not in st.session_state:
    st.stop()


# ---------------------------------------------------------------- step 4: human review

# The review and sign-off steps run as a fragment: ticking a box reruns only this part of
# the page instead of the whole app, so it's faster and the page doesn't jump around.


@st.fragment
def review_and_export() -> None:

    used_reqs: list[Requirement] = st.session_state.generated_for

    st.subheader("4 · Review and approve")
    st.caption(
        "Tick **Approve** for each test case you're happy with. You can also edit any text "
        "in the table. Only ticked test cases are exported."
    )

    if flash := st.session_state.pop("flash", None):
        st.success(flash, icon="🔁")

    base: list[TestCase] = st.session_state.test_cases
    s1, s2, _ = st.columns([1, 1, 4])
    if s1.button("☑ Select all"):
        set_test_cases([tc.model_copy(update={"review_status": "Approved"}) for tc in base])
        st.rerun(scope="fragment")
    if s2.button("☐ Clear all"):
        set_test_cases([tc.model_copy(update={"review_status": "Pending"}) for tc in base])
        st.rerun(scope="fragment")

    edited = st.data_editor(
        tcs_to_df(base),
        key=f"review_{st.session_state.review_version}",
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        disabled=["ID", "Req", "Rev"],
        column_config={
            "Approve": st.column_config.CheckboxColumn("✅", width=50, pinned=True, help="Tick to approve"),
            "ID": st.column_config.TextColumn(width="small", pinned=True),
            "Req": st.column_config.TextColumn(width="small"),
            "Rev": st.column_config.NumberColumn(
                width="small", help="Revision: goes up each time the model rewrites a test case"
            ),
            "Title": st.column_config.TextColumn(width="large"),
            "Steps": st.column_config.TextColumn(width="large"),
            "Expected result": st.column_config.TextColumn(width="medium"),
        },
    )
    current = df_to_tcs(edited)
    counts = review_summary(current)
    approved = [tc for tc in current if tc.review_status == "Approved"]
    not_approved = [tc for tc in current if tc.review_status != "Approved"]
    trace = traceability(used_reqs, approved)
    covered = sum(1 for t in trace if t["Coverage"] == "Covered")

    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Approved", len(approved))
    m2.metric("☐ Not approved", len(not_approved))
    m3.metric("Approved coverage", f"{covered / len(used_reqs):.0%}" if used_reqs else "–")

    uncovered = [t["Requirement ID"] for t in trace if t["Coverage"] != "Covered"]
    if approved and uncovered:
        st.warning(f"No approved test case yet for: {', '.join(uncovered)}", icon="⚠️")

    with st.expander("Traceability matrix (approved test cases)"):
        st.dataframe(pd.DataFrame(trace), width="stretch", hide_index=True)

    if not_approved:
        with st.expander(f"Not happy with some? Ask the model to rewrite the {len(not_approved)} unticked test case(s)"):
            feedback = st.text_area(
                "What should change? (optional)",
                key="feedback",
                placeholder="e.g. Use exact boundary values like INR 199 and INR 198; add an invalid input case.",
                height=80,
            )
            if st.button(f"🔁 Regenerate {len(not_approved)} unticked"):
                all_ids = [tc.id for tc in current + st.session_state.superseded]
                next_number = max(int(i.split("-")[-1]) for i in all_ids if i.split("-")[-1].isdigit()) + 1
                replacements = run_safely(
                    revise_test_cases,
                    not_approved,
                    used_reqs,
                    run_safely(get_index, doc_text),
                    llm,
                    next_number,
                    feedback=feedback,
                    progress=progress_bar("Rewriting the unticked test cases…"),
                )
                st.session_state.superseded += [
                    tc.model_copy(update={"reviewer_comment": feedback.strip()})
                    for tc in not_approved
                    if tc.id in replacements
                ]
                set_test_cases([replacements.get(tc.id, tc) for tc in current])
                st.session_state.flash = (
                    f"{len(replacements)} test case(s) rewritten. Review them and tick the ones you approve. "
                    "The originals are kept on the Not Approved sheet of the export."
                )
                st.rerun(scope="fragment")


    # ---------------------------------------------------------------- step 5: sign off and export

    st.subheader("5 · Sign off and download")
    reviewer = st.text_input("Reviewer name", key="reviewer", placeholder="Who approved these test cases?")
    ok, reason = ready_to_export(current, reviewer)
    if ok:
        st.success(
            f"Ready: {len(approved)} approved test case(s) will be exported, signed off by {reviewer.strip()}.",
            icon="✅",
        )
    else:
        st.info(f"Download locked. {reason}", icon="🔒")

    stem = Path(st.session_state.doc_name).stem if "://" not in st.session_state.doc_name else "brd"
    st.download_button(
        f"⬇️ Download {len(approved)} approved test cases (Excel)",
        build_workbook(used_reqs, current + st.session_state.superseded, st.session_state.doc_name, model, reviewer.strip())
        if ok
        else b"",
        file_name=f"{stem}_test_cases_approved.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        disabled=not ok,
    )


review_and_export()
