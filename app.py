import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from rag_pipeline import load_documents, split_documents, build_vectorstore, load_vectorstore, get_retriever
from agent_graph import build_agent, reset_retriever_cache

load_dotenv()

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Agentic RAG Assistant")
st.caption("Ask questions from your documents ")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = os.path.exists("chroma_db")

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base", type="primary"):
        if uploaded_files:
            if os.path.exists("data"):
                shutil.rmtree("data")
            os.makedirs("data", exist_ok=True)
            for file in uploaded_files:
                file_path = os.path.join("data", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            with st.spinner("Processing documents... this may take a minute"):
                docs = load_documents("data")
                chunks = split_documents(docs)

                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")

                build_vectorstore(chunks)
                reset_retriever_cache()  # clear old cached retriever
                st.session_state.vectorstore_ready = True
                st.session_state.agent = None  # force rebuild agent with new store

            st.success(f"✅ Processed {len(docs)} document(s) into {len(chunks)} chunks!")
        else:
            st.warning("Please upload at least one file first.")

    st.divider()
    st.caption("💡 The agent will answer from your documents first, and automatically fall back to a live web search if the answer isn't found there.")

if st.session_state.vectorstore_ready and st.session_state.agent is None:
    with st.spinner("Loading agent..."):
        st.session_state.agent = build_agent()

if not st.session_state.vectorstore_ready:
    st.info("👈 Upload a document and click **Build Knowledge Base** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "source" in msg:
                badge = "📄 From your documents" if msg["source"] == "docs" else "🌐 From web search"
                st.caption(badge)

    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.agent.invoke({
                    "question": question,
                    "documents": [],
                    "web_search_needed": False,
                    "answer": "",
                    "source": "docs"
                })
                answer = result["answer"]
                source = result["source"]

            st.markdown(answer)
            badge = "📄 From your documents" if source == "docs" else "🌐 From web search"
            st.caption(badge)

        st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})