import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIR = "chroma_db"

def load_documents(data_path="data"):
    """Load PDFs and text files from the data folder."""
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)

    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def get_embedding_model():
    """Free, local embedding model — no API cost."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(chunks, persist_directory=PERSIST_DIR):
    """Create and persist a Chroma vector store from document chunks."""
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store saved to {persist_directory}")
    return vectorstore


def load_vectorstore(persist_directory=PERSIST_DIR):
    """Load an existing vector store instead of rebuilding it."""
    embeddings = get_embedding_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def get_retriever(vectorstore, k=4):
    """Return a retriever that fetches top-k relevant chunks."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print("✅ RAG pipeline setup complete!")