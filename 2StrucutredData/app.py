import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------- Pydantic schema ----------
class Employee(BaseModel):
    name: str = Field(..., description="Full name of the employee")
    department: str = Field(..., description="Department the employee belongs to")
    skills: List[str] = Field(..., description="List of key skills the employee has")


class EmployeeData(BaseModel):
    employees: List[Employee] = Field(
        ..., description="List of structured employee data extracted from the text"
    )


# ---------- LangChain chain builder ----------
def build_chain(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    structured_llm = llm.with_structured_output(EmployeeData)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an intelligent data extraction assistant. "
            "Extract all employee-related details from the given text and output them "
            "strictly in the structured format (Employee Name, Department, Skills)."
        ),
        (
            "human",
            "Here is the text:\n\n{text}\n\n"
            "Extract all employees and their details."
        )
    ])

    return prompt | structured_llm


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Structured Data Extractor — Gemini + LangChain",
    page_icon="🧾",
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

st.markdown("<div class='title'>🧾 Structured Data Extractor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Week 3 — Task 2: Convert messy HR text into a clean table</div>", unsafe_allow_html=True)

# API key input
st.markdown("### 🔑 Gemini API Key")
api_key = st.text_input("Enter your Gemini API Key", type="password", placeholder="AIza...")

# Text input
st.markdown("### 🧠 Paste Your Unstructured Text")
input_text = st.text_area(
    "Unstructured text input",
    height=180,
    placeholder=(
        "Example:\n"
        "John Doe works in Marketing and has skills in SEO, Social Media, and Content Writing. "
        "Jane Smith from Engineering is skilled in Python, Docker, and AWS."
    )
)

# Analyze button
if st.button("🔍 Extract Structured Data", use_container_width=True, type="primary"):
    if not api_key:
        st.error("Please provide your Gemini API key.")
        st.stop()

    if not input_text.strip():
        st.warning("Please enter some unstructured text.")
        st.stop()

    with st.spinner("Extracting structured data using Gemini..."):
        try:
            chain = build_chain(api_key)
            result: EmployeeData = chain.invoke({"text": input_text})

            if not result.employees:
                st.warning("No employee data found in the text.")
                st.stop()

            # Convert to DataFrame
            df = pd.DataFrame([emp.dict() for emp in result.employees])
            df["skills"] = df["skills"].apply(lambda x: ", ".join(x))

            st.markdown("<div class='result-box'><h3 style='text-align:center;'>📋 Extracted Employee Data</h3></div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            # --- 💾 CSV Download Button ---
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Table as CSV",
                data=csv,
                file_name="structured_employee_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("🚀 Built with LangChain `with_structured_output()` + Gemini API · Week 3 — Task 2")
