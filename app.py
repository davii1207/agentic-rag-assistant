import os
import shutil
 
import streamlit as st
from dotenv import load_dotenv
 
from rag_pipeline import (
    SUPPORTED_EXTENSIONS,
    load_file,
    split_documents,
    add_chunks,
    reset_vectorstore,
    knowledge_base_exists,
)
from agent_graph import build_agent, initial_state
 
load_dotenv()
 
DATA_DIR = "data"
 
st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")
 
st.title("🤖 Agentic RAG Assistant")
st.caption("Ask questions from your documents")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
if "agent" not in st.session_state:
    st.session_state.agent = None
 
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = knowledge_base_exists()
 
 
def render_footer(source: str, citations: list):
    badge = "📄 From your documents" if source == "docs" else "🌐 From web search"
    st.caption(badge)
    if citations:
        with st.expander("Sources"):
            for c in citations:
                st.write(f"- {c}")
 
 
# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
 
    if st.button("Add to Knowledge Base", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            all_docs, added, failed = [], [], []
 
            with st.spinner("Processing documents... this may take a minute"):
                for file in uploaded_files:
                    safe_name = os.path.basename(file.name)
                    path = os.path.join(DATA_DIR, safe_name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    try:
                        all_docs.extend(load_file(path))
                        added.append(safe_name)
                    except ValueError as e:
                        failed.append((safe_name, str(e)))
 
                if all_docs:
                    chunks = split_documents(all_docs)
                    add_chunks(chunks)
                    st.session_state.vectorstore_ready = True
                    st.success(f"✅ Added {len(added)} file(s) as {len(chunks)} chunks.")
 
            for name, err in failed:
                st.error(f"{name}: {err}")
 
    if st.session_state.vectorstore_ready:
        if st.button("🗑️ Clear Knowledge Base"):
            reset_vectorstore()
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            st.session_state.vectorstore_ready = False
            st.session_state.messages = []
            st.rerun()
 
    st.divider()
    st.caption(
        "💡 The agent answers from your documents first, and automatically "
        "falls back to a live web search if the answer isn't found there."
    )
 
# ---------------------------------------------------------------- main
if st.session_state.vectorstore_ready and st.session_state.agent is None:
    with st.spinner("Loading agent..."):
        st.session_state.agent = build_agent()
 
if not st.session_state.vectorstore_ready:
    st.info("👈 Upload a document and click **Add to Knowledge Base** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "source" in msg:
                render_footer(msg["source"], msg.get("citations", []))
 
    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
 
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke(initial_state(question))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
 
            answer = result["answer"]
            source = result["source"]
            citations = result.get("citations", [])
 
            st.markdown(answer)
            render_footer(source, citations)
 
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "source": source, "citations": citations}
        )