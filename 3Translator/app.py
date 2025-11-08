import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Pydantic schema ----------
class TranslationResult(BaseModel):
    source_language: str = Field(..., description="The original language of the text.")
    target_language: str = Field(..., description="The target language for translation.")
    original_text: str = Field(..., description="Original text input by the user.")
    translated_text_script: str = Field(..., description="Translated text in native script of the target language.")
    translated_text_english: str = Field(..., description="Translated text written in English letters (transliteration).")
    meaning_in_english: str = Field(..., description="The English meaning or explanation of the translated sentence.")


# ---------- LangChain chain builder ----------
def build_chain(api_key: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    structured_llm = llm.with_structured_output(TranslationResult)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a multilingual Indian translator. You can understand text written in English, "
            "or English letters that represent Indian words (like 'ela unnavu' for Telugu). "
            "Translate accurately between Indian languages and English. Return all responses "
            "in the structured schema: translated text in target script, English transliteration, "
            "and the English meaning."
        ),
        (
            "human",
            "Source language: {source_lang}\n"
            "Target language: {target_lang}\n\n"
            "Text: {input_text}\n\n"
            "1. Translate to target language script.\n"
            "2. Provide transliteration (English letters).\n"
            "3. Explain the English meaning clearly."
        )
    ])

    return prompt | structured_llm


# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="🇮🇳 Indian Language Translator — Gemini + LangChain",
    page_icon="🌍",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #f0f4f8, #ffffff);
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

st.markdown("<div class='title'>🇮🇳 Indian Language Translator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Week 3 — Task 3: Translate Indian languages (supports phonetic English input)</div>", unsafe_allow_html=True)

# ---------- UI Inputs ----------
st.markdown("### 🔑 Gemini API Key")
api_key = st.text_input("Enter your Gemini API Key", type="password", placeholder="AIza...")

st.markdown("### 🌐 Select Languages")
col1, col2 = st.columns(2)
with col1:
    source_lang = st.selectbox(
        "Source Language",
        ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Gujarati", "Marathi", "Punjabi", "Odia"],
        index=2
    )
with col2:
    target_lang = st.selectbox(
        "Target Language",
        ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Gujarati", "Marathi", "Punjabi", "Odia"],
        index=3
    )

st.markdown(f"### ✍️ Enter Text in {source_lang}")
input_text = st.text_area(
    f"Write your text in {source_lang} (you can also write Indian words in English letters, e.g., 'ela unnavu')",
    height=180,
    placeholder="Example: ela unnavu"
)

# ---------- Translate Button ----------
if st.button("🌐 Translate", use_container_width=True, type="primary"):
    if not api_key:
        st.error("Please provide your Gemini API key.")
        st.stop()

    if not input_text.strip():
        st.warning("Please enter text to translate.")
        st.stop()

    with st.spinner(f"Translating from {source_lang} ➜ {target_lang} using Gemini..."):
        try:
            chain = build_chain(api_key)
            result: TranslationResult = chain.invoke({
                "source_lang": source_lang,
                "target_lang": target_lang,
                "input_text": input_text
            })

            # ---------- Display results ----------
            st.markdown("<div class='result-box'><h3 style='text-align:center;'>✅ Translation Result</h3></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🗣️ Original Text**")
                st.text_area("Source Text", value=result.original_text, height=100)

            with col2:
                st.markdown(f"**🎯 Target Language: {result.target_language}**")
                st.text_area("Translated Text (Script)", value=result.translated_text_script, height=100)

            st.markdown("### 🔡 English Transliteration")
            st.info(result.translated_text_english)

            st.markdown("### 💬 English Meaning")
            st.success(result.meaning_in_english)

            # ---------- Download translation ----------
            txt_data = (
                f"Source: {result.source_language}\n"
                f"Target: {result.target_language}\n\n"
                f"Original: {result.original_text}\n\n"
                f"Translated (Script): {result.translated_text_script}\n"
                f"Transliteration: {result.translated_text_english}\n"
                f"Meaning in English: {result.meaning_in_english}"
            )

            st.download_button(
                label="💾 Download Translation (.txt)",
                data=txt_data.encode('utf-8'),
                file_name="indian_translation_result.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("🚀 Built with LangChain `with_structured_output()` + Gemini API · Week 3 — Task 3 (Indian Translator)")
