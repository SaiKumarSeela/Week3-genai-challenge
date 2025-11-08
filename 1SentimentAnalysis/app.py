import streamlit as st
import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Pydantic schema ----------
class SentimentResult(BaseModel):
    label: Literal["Positive", "Negative", "Neutral"] = Field(
        ..., description="Overall sentiment of the review."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1."
    )
    rationale: str = Field(
        ..., description="One-sentence reason for the classification."
    )

# ---------- LangChain chain builder ----------
def build_chain(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    structured_llm = llm.with_structured_output(SentimentResult)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise sentiment classifier for product reviews. "
            "Classify the review as exactly one of: Positive, Negative, or Neutral. "
            "Respond using the structured schema provided."
        ),
        (
            "human",
            "Product Review:\n\n{review}\n\n"
            "Return label, confidence (0–1), and a one-sentence rationale."
        )
    ])

    return prompt | structured_llm

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Sentiment Analyzer — Gemini + LangChain",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
    <style>
    body {
        background-color: #f8f9fb;
    }
    .stApp {
        background: linear-gradient(to bottom right, #eef2f3, #ffffff);
    }
    .title {
        font-size: 2.2rem;
        text-align: center;
        color: #333333;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Product Review Sentiment</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Week 3 — Task 1: Analyze review sentiment using Gemini + LangChain</div>", unsafe_allow_html=True)

# API key input
st.markdown("### 🔑 Gemini API Key")
api_key = st.text_input("Enter your Gemini API Key", type="password", placeholder="AIza...")

# Review input
st.markdown("### 💬 Write or Paste Product Review")
review_text = st.text_area(
    "Your Review",
    height=160,
    placeholder="e.g., The headphones sound great and the battery lasts long, but the build feels cheap."
)

if st.button("✨ Analyze Sentiment", use_container_width=True, type="primary"):
    if not api_key:
        st.error("Please provide your Gemini API key.")
        st.stop()
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
        st.stop()

    with st.spinner("Analyzing sentiment with Gemini..."):
        try:
            chain = build_chain(api_key)
            result: SentimentResult = chain.invoke({"review": review_text})

            emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}[result.label]
            color = {
                "Positive": "#22c55e",  # green
                "Negative": "#ef4444",  # red
                "Neutral": "#eab308"   # yellow
            }[result.label]

            st.markdown(f"""
            <div class='result-box'>
                <h3 style='color:{color}; text-align:center;'>{emoji} Sentiment: {result.label}</h3>
                <p style='text-align:center; font-size:1.1rem;'>Confidence: <b>{result.confidence:.2f}</b></p>
                <hr style='opacity:0.3;'>
                <p><b>Rationale:</b> {result.rationale}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("🚀 Built with LangChain `with_structured_output()` + Gemini API · Week 3 — Task 1")

