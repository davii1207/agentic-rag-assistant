import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from rag_pipeline import load_vectorstore, get_retriever

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

web_search_tool = TavilySearchResults(k=3, api_key=os.getenv("TAVILY_API_KEY"))

retriever = None

def get_current_retriever():
    global retriever
    if retriever is None:
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore, k=4)
    return retriever

def reset_retriever_cache():
    """Force the retriever to reload from disk next time it's needed."""
    global retriever
    retriever = None


class AgentState(TypedDict):
    question: str
    documents: List[str]
    web_search_needed: bool
    answer: str
    source: str  # "docs" or "web"


def retrieve(state: AgentState):
    question = state["question"]
    docs = get_current_retriever().invoke(question)
    doc_texts = [d.page_content for d in docs]
    print(f"[Retrieve] Found {len(doc_texts)} chunks")
    return {"documents": doc_texts}


def grade_documents(state: AgentState):
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("[Grade] No documents retrieved -> web search")
        return {"web_search_needed": True}

    combined = "\n\n".join(documents)
    grading_prompt = f"""You are grading whether retrieved context is relevant to a question.
Question: {question}
Retrieved context: {combined[:2000]}

Respond with only one word: "relevant" or "irrelevant"."""

    response = llm.invoke([HumanMessage(content=grading_prompt)])
    verdict = response.content.strip().lower()
    print(f"[Grade] Verdict: {verdict}")

    needs_web = "irrelevant" in verdict
    return {"web_search_needed": needs_web}


def web_search(state: AgentState):
    question = state["question"]
    print("[Web Search] Searching the web...")
    results = web_search_tool.invoke({"query": question})
    web_content = "\n\n".join([r["content"] for r in results])
    return {"documents": [web_content], "source": "web"}


def generate(state: AgentState):
    question = state["question"]
    documents = state["documents"]
    source = state.get("source", "docs")
    context = "\n\n".join(documents)

    system_prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say so honestly.
Cite whether this came from the uploaded documents or a web search.

Context source: {source}
Context: {context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])
    return {"answer": response.content, "source": source}


def decide_next_step(state: AgentState):
    if state.get("web_search_needed"):
        return "web_search"
    return "generate"


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
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


if __name__ == "__main__":
    agent = build_agent()

    test_question = input("Ask a question: ")
    result = agent.invoke({"question": test_question, "documents": [], "web_search_needed": False, "answer": "", "source": "docs"})

    print("\n" + "="*50)
    print(f"ANSWER (source: {result['source']}):")
    print(result["answer"])
    print("="*50)