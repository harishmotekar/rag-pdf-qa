import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Q&A using RAG", layout="wide")
st.title("📄 PDF / Text Question Answering (RAG)")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return embeddings, llm

embeddings, llm = load_models()

# ---------------- Input ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
manual_text = st.text_area("Or paste text here")

# ---------------- Process Document ----------------
if st.button("Process Document"):
    text = ""

    if uploaded_file:
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text += page.extract_text()

    elif manual_text:
        text = manual_text

    else:
        st.warning("Please upload a PDF or enter text")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.session_state.vs = vectorstore
    st.success("Document processed successfully ✅")

# ---------------- Question Answering ----------------
if "vs" in st.session_state:
    query = st.text_input("Ask a question")

    if query:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vs.as_retriever()
        )

        answer = qa.run(query)
        st.subheader("Answer")
        st.write(answer)
