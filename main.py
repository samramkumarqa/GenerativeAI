import os
import re
import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from functools import lru_cache
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ====== LOAD ENV ======
load_dotenv()

# ====== LLM (Together.ai) ======
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.together.xyz/v1",
    temperature=0.3
)

# ====== STREAMLIT UI ======
st.set_page_config(page_title="AI Testcase Generator", layout="wide")
st.title("AI Test Case Generator (RAG + FAISS)")
st.markdown("Upload a **BRD PDF** or paste a **URL** to generate test cases.")

# ====== HELPERS ======

@lru_cache(maxsize=32)
def fetch_article_text(url: str) -> str:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    return "\n".join(p.get_text() for p in soup.find_all("p"))

def extract_pdf_text(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def clean_md(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^- ", "", text, flags=re.MULTILINE)
    return text.strip()

def parse_testcases(text, testcase_type):
    cases = []
    blocks = re.split(r"\n\d+\.|\nTC-", text)

    for i, block in enumerate(blocks[1:], start=1):
        lines = [clean_md(l) for l in block.split("\n") if l.strip()]

        if len(lines) >= 3:
            cases.append({
                "Test Case ID": f"TC_{i:03}",
                "Title": lines[0],
                "Preconditions": lines[1],
                "Steps": "\n".join(lines[2:-1]),
                "Expected Result": lines[-1],
                "Type": testcase_type
            })
    return pd.DataFrame(cases)

# ====== FAISS ======

@st.cache_resource
def build_faiss_index(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_text(text)
    documents = [Document(page_content=d) for d in docs]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)

def rag_query(vectordb, question, k=4):
    docs = vectordb.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a Senior QA Engineer.

Use ONLY the context below.

CONTEXT:
{context}

TASK:
{question}

FORMAT:
1. Title
2. Preconditions
3. Steps
4. Expected Result
"""
    return llm.invoke(prompt).content

# ====== UI ======
url = st.text_input("BRD URL")
uploaded_file = st.file_uploader("Or Upload BRD PDF", type=["pdf"])

testcase_type = st.selectbox(
    "Select Test Case Type",
    [
        "Functional", "Non-Functional", "Regression", "Smoke", "Sanity",
        "Integration", "System", "UAT", "Database", "API",
        "UI", "Automation", "Exploratory", "Localization", "Recovery"
    ]
)

# ====== ACTION ======
if st.button("Generate Test Cases"):
    try:
        if uploaded_file:
            article_text = extract_pdf_text(uploaded_file)
        elif url:
            article_text = fetch_article_text(url)
        else:
            st.warning("Provide a URL or upload a PDF.")
            st.stop()

        if len(article_text.strip()) < 100:
            st.error("Not enough text found.")
            st.stop()

        status = st.empty()
        status.info("Indexing document...")
        vectordb = build_faiss_index(article_text)

        status.info("Generating test cases...")
        question = f"Generate {testcase_type} test cases."
        result = rag_query(vectordb, question)

        status.success("Done!")

        st.subheader("Generated Test Cases")
        st.write(result)

        df = parse_testcases(result, testcase_type)

        if not df.empty:
            file_name = "testcases.xlsx"
            df.to_excel(file_name, index=False)
            with open(file_name, "rb") as f:
                st.download_button(
                    "Download as Excel",
                    f,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error: {e}")
