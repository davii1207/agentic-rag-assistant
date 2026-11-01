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
# How many times the agent may rewrite the query and re-retrieve before
# giving up on the documents and falling back to web search.
# (We will tune this value with the evaluation later.)
MAX_RETRIES = 1
 
 
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
    question: str             # the user's original question
    search_query: str         # the query used for retrieval (may be rewritten)
    retries: int              # how many rewrites have happened so far
    documents: List[str]      # retrieved text chunks (or web snippets)
    citations: List[str]      # one label per document: "file.pdf, p. 4" or a URL
    web_search_needed: bool
    answer: str
    source: str               # "docs" or "web"
    steps: List[str]          # human-readable trace of what the agent did
 
 
def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "search_query": question,
        "retries": 0,
        "documents": [],
        "citations": [],
        "web_search_needed": False,
        "answer": "",
        "source": "docs",
        "steps": [],
    }
 
 
def _log(state: AgentState, message: str) -> List[str]:
    """Return the trace with one more line appended."""
    return state.get("steps", []) + [message]
 
 
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
    query = state["search_query"]
    retriever = get_retriever(get_vectorstore(), k=RETRIEVAL_K)
    docs = retriever.invoke(query)
    print(f"[Retrieve] Found {len(docs)} chunks for: {query}")
    return {
        "documents": [d.page_content for d in docs],
        "citations": [_citation_label(d.metadata) for d in docs],
        "steps": _log(state, f'Retrieved {len(docs)} chunks for: "{query}"'),
    }
 
 
def grade_documents(state: AgentState):
    # Always grade against the user's ORIGINAL question, not the rewritten query.
    question = state["question"]
    documents = state["documents"]
 
    if not documents:
        print("[Grade] No documents retrieved")
        return {
            "web_search_needed": True,
            "steps": _log(state, "Grader: nothing retrieved"),
        }
 
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
        note = "Grader: relevant" if verdict.relevant else "Grader: NOT relevant"
    except Exception as e:
        # If the grader fails, keep the documents we already have rather than crash.
        print(f"[Grade] Grader failed ({e}); using retrieved documents")
        needs_web = False
        note = "Grader failed; using retrieved documents"
 
    return {"web_search_needed": needs_web, "steps": _log(state, note)}
 
 
def rewrite_query(state: AgentState):
    """Ask the LLM for a better retrieval query, then loop back to retrieve."""
    question = state["question"]
    previous = state["search_query"]
 
    new_query = previous
    try:
        response = get_llm().invoke([
            SystemMessage(content=(
                "Rewrite the user's question into a better search query for a vector "
                "database of document chunks. Use specific keywords and synonyms likely "
                "to appear in the documents. Return ONLY the rewritten query, nothing else."
            )),
            HumanMessage(content=f"Original question: {question}\nPrevious query that failed: {previous}"),
        ])
        candidate = (response.content or "").strip().strip('"').strip()
        if candidate:
            new_query = candidate
    except Exception as e:
        print(f"[Rewrite] Failed ({e}); keeping previous query")
 
    print(f"[Rewrite] {previous!r} -> {new_query!r}")
    return {
        "search_query": new_query,
        "retries": state["retries"] + 1,
        "steps": _log(state, f'Rewrote query -> "{new_query}"'),
    }
 
 
def web_search(state: AgentState):
    # Web search uses the ORIGINAL question (the rewrite is tuned for the documents).
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
        "steps": _log(state, f"Fell back to web search ({len(results)} results)"),
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
            "steps": _log(state, "No context found; returned a 'could not find' message"),
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
    return {
        "answer": response.content,
        "source": source,
        "citations": unique_citations,
        "steps": _log(state, f"Generated answer from {'documents' if source == 'docs' else 'web results'}"),
    }
 
 
def decide_next_step(state: AgentState):
    """After grading: answer, retry with a rewritten query, or go to the web."""
    if not state.get("web_search_needed"):
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "rewrite_query"
    return "web_search"
 
 
# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_agent():
    workflow = StateGraph(AgentState)
 
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
 
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")   # the loop
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
    print("\nAgent steps:")
    for s in result["steps"]:
        print(f" - {s}")
    print("=" * 50)