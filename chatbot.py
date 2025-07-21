import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

os.environ["GROQ_API_KEY"] = "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"  

# Page config
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")

# Theme switcher radio
theme = st.radio("🌓 Select Theme", ["Dark", "Light"], horizontal=True)

# Define colors based on theme
if theme == "Dark":
    bg_color = "rgba(0, 0, 0, 0.85)"
    app_bg = "rgba(20, 20, 20, 0.6)"
    sidebar_bg = "rgba(10, 10, 10, 0.9)"
    text_color = "#FFFFFF"
    input_bg = "rgba(50, 50, 50, 0.5)"
    accent_color = "#2AA198"
else:
    bg_color = "#f0f2f6"
    app_bg = "rgba(255, 255, 255, 0.6)"
    sidebar_bg = "rgba(255, 255, 255, 0.9)"
    text_color = "#000000"
    input_bg = "rgba(240, 240, 240, 0.9)"
    accent_color = "#2AA198"

# Inject CSS styles
st.markdown(f"""
<style>
body {{
    background: {bg_color};
    font-family: 'Segoe UI', sans-serif;
}}
.stApp {{
    background: {app_bg};
    backdrop-filter: blur(15px);
    padding: 2rem;
    border-radius: 16px;
    margin: 2rem auto;
    max-width: 1000px;
    color: {text_color};
}}
h1, h2, h3, h4, .stRadio label, label {{
    color: {"text_color"} !important;
    font-weight: 600;
}}
[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    color: {text_color} !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}}
[data-testid="stSidebar"] * {{
    color: {text_color} !important;
}}
.stTextInput input {{
    background-color: {input_bg};
    color: {text_color} !important;
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 0.5rem;
}}
.stTextInput input::placeholder {{
    color: #bbbbbb;
}}
.stMarkdown, .stText {{
    color: {text_color} !important;
}}
.stSuccess {{
    background-color: rgba(255, 255, 255, 0.07) !important;
    border-left: 6px solid {accent_color};
    padding: 1rem;
    border-radius: 12px;
    color: {text_color} !important;
    font-size: 1.05rem;
}}
button {{
    background-color: {accent_color} !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
}}
.stRadio > label {{
    font-size: 1rem;
    font-weight: 600;
    color: {"text_color"} !important;
}}
.stRadio div[role="radiogroup"] > label {{
    color: {text_color} !important;
    font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown("### 📁 Upload PDF")
    file = st.file_uploader("Upload your PDF", type="pdf")

    st.write("Then ask your question below!")

# Main UI
st.markdown(f"<h1 style='text-align:center;'>📄✨ Chat With Your PDF</h1>", unsafe_allow_html=True)

if file is not None:
    pdf_pages = PdfReader(file)
    text = ""
    for page in pdf_pages.pages:
        text += page.extract_text()  # Note: was missing += in your original code

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,  
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Query input
    user_query = st.text_input("Ask a question about the file")
    if user_query:
        match = vector_store.similarity_search(user_query)
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            max_retries=2
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_query)

        st.subheader("✅ Response")
        st.markdown(f"<div style='color:{text_color};'>{response}</div>", unsafe_allow_html=True)
else:
    st.info("Please upload a PDF file to get started.")
