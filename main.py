import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

# ====== SET TOGETHER.AI API KEY ======
os.environ["OPENAI_API_KEY"] = "6fdb42ef99b431fae9129c358de43c6844bbd2d7abe8ac0f18a374f44879cd02"  # <-- Replace with your actual key

# ====== DEFINE LLM ======
llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.together.xyz/v1"
)

# ====== PROMPT TEMPLATE ======
prompt = PromptTemplate.from_template(
    "You are a helpful AI assistant. Given the following article text, answer the user's question:\n\n"
    "ARTICLE:\n{text}\n\n"
    "USER QUESTION:\n{question}\n\n"
    "ANSWER:"
)

chain = prompt | llm

# ====== STREAMLIT UI SETUP ======
st.set_page_config(page_title="Testcase creation Tool", layout="wide")
st.title("Test creation tool (Free with Together.ai)")
st.markdown("Enter a **URL** or **PDF** and a **request** below.")

# ====== CACHING ======
@lru_cache(maxsize=32)
def fetch_article_text(url: str) -> str:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text() for p in paragraphs)

# ====== UI INPUTS ======
url = st.text_input("BRD URL", placeholder="https://...")
question = st.text_input("Your Question", placeholder="What type testcases to be prepared?")

if st.button("Process the request"):
    if not url or not question:
        st.warning("Please provide both the URL and the question.")
    else:
        try:
            article_text = fetch_article_text(url)
            if len(article_text.strip()) < 100:
                st.error("Couldn't extract meaningful text from the URL.")
            else:
                st.info("Processing your question with the cloud model...")
                result = chain.invoke({"text": article_text, "question": question})
                st.subheader("Answer:")
                st.write(result)
        except Exception as e:
            st.error(f"Error fetching or processing the URL:\n\n{str(e)}")
