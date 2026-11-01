import os
from functools import lru_cache
from typing import List, TypedDict
 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
 
from rag_pipeline import get_vectorstore, get_retriever
 
load_dotenv()
 
LLM_MODEL = "openai/gpt-oss-20b"
RETRIEVAL_K = 4
 
 
# --------------------------------------------------------------------------
# Lazy, cached clients (so importing this file never crashes on a missing key)
# --------------------------------------------------------------------------
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file.")
    return value
 
 
@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=_require_env("GROQ_API_KEY"))
 
 
@lru_cache(maxsize=1)
def get_web_search_tool() -> TavilySearch:
    _require_env("TAVILY_API_KEY")  # TavilySearch reads it from the environment
    return TavilySearch(max_results=3)
 
 
# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    documents: List[str]      # retrieved text chunks (or web snippets)
    citations: List[str]      # one label per document: "file.pdf, p. 4" or a URL
    web_search_needed: bool
    answer: str
    source: str               # "docs" or "web"
 
 
def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "documents": [],
        "citations": [],
        "web_search_needed": False,
        "answer": "",
        "source": "docs",
    }
 
 
class GradeDocuments(BaseModel):
    """Structured verdict from the relevance grader."""
    relevant: bool = Field(
        description="True if the context contains information that helps answer the question."
    )
 
 
# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def _citation_label(metadata: dict) -> str:
    name = metadata.get("filename") or os.path.basename(metadata.get("source", "") or "") or "document"
    page = metadata.get("page")
    return f"{name}, p. {page + 1}" if isinstance(page, int) else name
 
 
def retrieve(state: AgentState):
    question = state["question"]
    retriever = get_retriever(get_vectorstore(), k=RETRIEVAL_K)
    docs = retriever.invoke(question)
    print(f"[Retrieve] Found {len(docs)} chunks")
    return {
        "documents": [d.page_content for d in docs],
        "citations": [_citation_label(d.metadata) for d in docs],
    }
 
 
def grade_documents(state: AgentState):
    question = state["question"]
    documents = state["documents"]
 
    if not documents:
        print("[Grade] No documents retrieved -> web search")
        return {"web_search_needed": True}
 
    combined = "\n\n".join(documents)
    try:
        grader = get_llm().with_structured_output(GradeDocuments)
        verdict = grader.invoke([
            SystemMessage(content=(
                "You grade whether retrieved context is relevant to a user question. "
                "Relevant means the context contains information that helps answer it."
            )),
            HumanMessage(content=f"Question: {question}\n\nRetrieved context:\n{combined}"),
        ])
        needs_web = not verdict.relevant
        print(f"[Grade] relevant={verdict.relevant}")
    except Exception as e:
        # If the grader fails, keep the documents we already have rather than crash.
        print(f"[Grade] Grader failed ({e}); using retrieved documents")
        needs_web = False
 
    return {"web_search_needed": needs_web}
 
 
def web_search(state: AgentState):
    question = state["question"]
    print("[Web Search] Searching the web...")
 
    try:
        raw = get_web_search_tool().invoke({"query": question})
    except Exception as e:
        print(f"[Web Search] Failed: {e}")
        raw = {}
 
    results = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    results = [r for r in results if isinstance(r, dict) and r.get("content")]
 
    return {
        "documents": [r["content"] for r in results],
        "citations": [r.get("url", "web") for r in results],
        "source": "web",
    }
 
 
def generate(state: AgentState):
    question = state["question"]
    documents = state["documents"]
    citations = state.get("citations", [])
    source = state.get("source", "docs")
 
    if not documents:
        return {
            "answer": "I couldn't find relevant information in your documents or on the web.",
            "source": source,
            "citations": [],
        }
 
    labels = citations if len(citations) == len(documents) else ["unknown"] * len(documents)
    context = "\n\n".join(f"Source: {lab}\n{text}" for lab, text in zip(labels, documents))
 
    system_prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say so honestly.
Do not use outside knowledge.
 
Context comes from: {"the user's uploaded documents" if source == "docs" else "a live web search"}
 
{context}"""
 
    response = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])
 
    unique_citations = list(dict.fromkeys(labels))  # de-duplicate, keep order
    return {"answer": response.content, "source": source, "citations": unique_citations}
 
 
def decide_next_step(state: AgentState):
    return "web_search" if state.get("web_search_needed") else "generate"
 
 
# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_agent():
    workflow = StateGraph(AgentState)
 
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
 
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {"web_search": "web_search", "generate": "generate"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
 
    return workflow.compile()
 
 
if __name__ == "__main__":
    agent = build_agent()
    test_question = input("Ask a question: ")
    result = agent.invoke(initial_state(test_question))
 
    print("\n" + "=" * 50)
    print(f"ANSWER (source: {result['source']}):")
    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f" - {c}")
    print("=" * 50)