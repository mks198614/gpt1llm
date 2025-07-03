import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
import os

# ---------------------- Initialize Login State ---------------------- #
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = ""
if "trigger_rerun" not in st.session_state:
    st.session_state["trigger_rerun"] = False

# ---------------------- User DB Setup ---------------------- #
USER_DB_PATH = "user_db.json"
if not os.path.exists(USER_DB_PATH):
    with open(USER_DB_PATH, "w") as f:
        json.dump({"admin": "admin"}, f)  # default admin

with open(USER_DB_PATH, "r") as f:
    USER_DB = json.load(f)

# ---------------------- Page Setup ---------------------- #
st.set_page_config(page_title="Personalized Learning using LLM", page_icon="📘", layout="centered")

# ---------------------- Navigation ---------------------- #
menu = st.sidebar.selectbox("Navigate", ["🏠 Welcome", "🔐 Login", "📝 Register", "📘 Learning Assistant", "🧑‍💼 Admin Panel"])

# ---------------------- Welcome Page ---------------------- #
if menu == "🏠 Welcome":
    st.title("📘 Personalized Learning using LLM")
    st.subheader("Empowered by LLM and your study materials")
    st.markdown("""
    - Upload your personal PDF notes  
    - Ask questions and get instant answers  
    - Powered by Hugging Face models and LangChain
    """)
    st.info("Use the sidebar to Register/Login and Start Learning")

# ---------------------- Login Page ---------------------- #
elif menu == "🔐 Login":
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_DB and USER_DB[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.session_state["trigger_rerun"] = True
            st.success(f"✅ Logged in successfully as {username}!")
        else:
            st.error("❌ Invalid credentials")

# ---------------------- Register Page ---------------------- #
elif menu == "📝 Register":
    st.title("📝 New User Registration")
    new_user = st.text_input("Choose a username")
    new_pass = st.text_input("Choose a password", type="password")
    if st.button("Register"):
        if new_user in USER_DB:
            st.warning("⚠️ Username already exists")
        elif new_user.strip() == "" or new_pass.strip() == "":
            st.warning("⚠️ Username and password cannot be empty")
        else:
            USER_DB[new_user] = new_pass
            with open(USER_DB_PATH, "w") as f:
                json.dump(USER_DB, f)
            st.success("✅ Registration successful! Please login.")

# ---------------------- Admin Panel ---------------------- #
elif menu == "🧑‍💼 Admin Panel":
    if st.session_state.get("authenticated") and st.session_state.get("user") == "admin":
        st.title("🧑‍💼 Admin Panel - Registered Users")
        st.table([{"Username": u} for u in USER_DB.keys()])
    else:
        st.error("❌ Only admin can access this panel.")

# ---------------------- Main LLM App ---------------------- #
elif menu == "📘 Learning Assistant":
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please login to access this page from the sidebar.")
        st.stop()

    st.title(f"📘 Welcome, {st.session_state['user']}")
    uploaded_file = st.file_uploader("Upload your Notes (PDF)", type=["pdf"])

    if uploaded_file:
        with st.sidebar.expander("📄 PDF Info"):
            st.write(f"Filename: `{uploaded_file.name}`")
            st.write(f"Size: {uploaded_file.size / 1024:.2f} KB")

        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("❌ File too large. Please upload a PDF smaller than 200MB.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        with st.spinner("🔄 Loading and Splitting PDF..."):
            try:
                loader = PyMuPDFLoader(pdf_path)
                documents = loader.load()
                st.write("✅ PDF Loaded")
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)
                st.write(f"✅ Split into {len(docs)} chunks")
            except Exception as e:
                st.error(f"❌ Error loading PDF: {e}")

        with st.spinner("🔄 Generating Embeddings..."):
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.from_documents(docs, embeddings)
                st.write("✅ Embeddings done")
            except Exception as e:
                st.error(f"❌ Embedding error: {e}")

        with st.spinner("🔄 Loading LLM... (This may take ~30s on CPU)"):
            try:
                model_name = "google/flan-t5-base"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
                llm = HuggingFacePipeline(pipeline=pipe)
                st.write("✅ Model loaded")
            except Exception as e:
                st.error(f"❌ Model load error: {e}")

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

        level = st.radio("Select your difficulty level:", ["Beginner", "Moderate", "Advanced"])

        sample_questions = {
            "Beginner": "What is AI in simple words?",
            "Moderate": "What are the main types of AI?",
            "Advanced": "How does symbolic AI differ from neural networks?"
        }

        st.markdown("📌 Try a sample question or ask your own:")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Ask your question:")
        with col2:
            if st.button("Use Sample Question"):
                query = sample_questions[level]

        if query:
            with st.spinner("🤖 Generating answer... (be patient on CPU)"):
                try:
                    prompt = f"Answer this as a {level.lower()} learner: {query}"
                    answer = qa_chain.run(prompt)
                    st.write("📗 **Answer:**", answer)
                except Exception as e:
                    st.error(f"❌ Answering failed: {e}")

        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = ""
            st.session_state["trigger_rerun"] = True
            st.success("🔓 Logged out successfully.")

# ---------------------- Safe Rerun Trigger ---------------------- #
if st.session_state.get("trigger_rerun"):
    st.session_state["trigger_rerun"] = False
    st.experimental_rerun()

