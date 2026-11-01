import os
from functools import lru_cache
from typing import List
 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
 
load_dotenv()
 
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = (".pdf", ".txt")
 
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
 
 
# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_file(path: str) -> List[Document]:
    """Load a single PDF or TXT file and tag every page with its filename."""
    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)
 
    if ext == ".pdf":
        docs = PyPDFLoader(path).load()
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = [Document(page_content=text, metadata={"source": path})]
    else:
        raise ValueError(f"Unsupported file type: {filename}")
 
    # Drop empty pages (e.g. scanned PDFs with no text layer)
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError(
            f"No extractable text found in {filename}. "
            "If it's a scanned PDF, it needs OCR first."
        )
 
    for d in docs:
        d.metadata["filename"] = filename
    return docs
 
 
def load_documents(data_path: str = "data") -> List[Document]:
    """Load every supported file found under data_path."""
    documents: List[Document] = []
    if not os.path.isdir(data_path):
        print(f"Folder '{data_path}' does not exist.")
        return documents
 
    for root, _, files in os.walk(data_path):
        for name in sorted(files):
            if name.lower().endswith(SUPPORTED_EXTENSIONS):
                try:
                    documents.extend(load_file(os.path.join(root, name)))
                except ValueError as e:
                    print(f"Skipping {name}: {e}")
 
    print(f"Loaded {len(documents)} pages/documents")
    return documents
 
 
# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks
 
 
def make_chunk_ids(chunks: List[Document]) -> List[str]:
    """Deterministic IDs (filename::n) so re-uploading a file updates it
    instead of creating duplicates."""
    counters: dict = {}
    ids = []
    for c in chunks:
        name = c.metadata.get("filename", "unknown")
        n = counters.get(name, 0)
        counters[name] = n + 1
        ids.append(f"{name}::{n}")
    return ids
 
 
# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedding_model():
    """Free, local embedding model. Cached so it loads only once per process."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
 
 
def get_vectorstore(persist_directory: str = PERSIST_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=get_embedding_model(),
    )
 
 
# kept for backward compatibility with older imports
load_vectorstore = get_vectorstore
 
 
def knowledge_base_exists(persist_directory: str = PERSIST_DIR) -> bool:
    """True only if the store actually contains chunks (an empty folder doesn't count)."""
    if not os.path.isdir(persist_directory):
        return False
    try:
        return len(get_vectorstore(persist_directory).get(limit=1)["ids"]) > 0
    except Exception:
        return False
 
 
def add_chunks(chunks: List[Document], persist_directory: str = PERSIST_DIR) -> Chroma:
    """Add chunks to the store. Chunks from a file that was uploaded before
    replace the old version of that file."""
    if not chunks:
        raise ValueError("No chunks to add - the documents contained no text.")
 
    vectorstore = get_vectorstore(persist_directory)
 
    for name in {c.metadata.get("filename") for c in chunks}:
        if name:
            vectorstore.delete(where={"filename": name})
 
    vectorstore.add_documents(chunks, ids=make_chunk_ids(chunks))
    print(f"Added {len(chunks)} chunks to {persist_directory}")
    return vectorstore
 
 
def reset_vectorstore(persist_directory: str = PERSIST_DIR) -> None:
    """Wipe the collection without deleting files on disk (avoids Windows file-lock errors)."""
    try:
        get_vectorstore(persist_directory).delete_collection()
    except Exception as e:
        print(f"Nothing to reset: {e}")
 
 
def build_vectorstore(chunks: List[Document], persist_directory: str = PERSIST_DIR) -> Chroma:
    """Rebuild the store from scratch."""
    reset_vectorstore(persist_directory)
    return add_chunks(chunks, persist_directory)
 
 
def get_retriever(vectorstore: Chroma, k: int = 4):
    """Return a retriever that fetches top-k relevant chunks."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
 
 
if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("Put some .pdf/.txt files in the data/ folder first.")
    else:
        chunks = split_documents(docs)
        build_vectorstore(chunks)
        print("RAG pipeline setup complete!")