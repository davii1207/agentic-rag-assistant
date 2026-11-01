"""
Step 8: build a CANDIDATE evaluation dataset from the documents already in your
knowledge base.
 
Run from the project root (the folder that contains app.py):
 
    python eval/generate_dataset.py            # 35 answerable questions
    python eval/generate_dataset.py --n 40     # choose another number
 
Output: eval/dataset.json  -> you will review/edit this file by hand (Step 9).
 
Two kinds of questions are produced:
  * answerable    - written by an LLM from a real chunk of your documents, so we
                    KNOW which chunk/page should be retrieved (ground truth).
  * out_of_scope  - unrelated to your documents; the agent should use web search.
"""
import argparse
import json
import os
import random
import sys
import time
 
# let this script import rag_pipeline.py from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
 
from rag_pipeline import get_vectorstore, knowledge_base_exists
 
load_dotenv(os.path.join(ROOT, ".env"))
 
# Question writer (and, later, the judge). Deliberately NOT the same model that
# answers questions in the agent (gpt-oss-20b), to reduce self-preference bias.
# If Groq retires this model, change this one line.
GENERATOR_MODEL = "llama-3.3-70b-versatile"
 
OUT_PATH = os.path.join(ROOT, "eval", "dataset.json")
SEED = 42
MIN_CHARS = 400          # skip tiny chunks
MIN_LETTER_RATIO = 0.65  # skip tables of contents, numbers, dotted lines
 
OUT_OF_SCOPE = [
    "What is the recipe for authentic Hyderabadi biryani?",
    "Who won the most recent FIFA World Cup?",
    "What is the capital city of Australia?",
    "How should a beginner train for a half marathon?",
    "What are the health benefits of turmeric?",
    "Explain how photosynthesis works.",
    "Who wrote the novel Pride and Prejudice?",
    "What is the current price of gold per ounce?",
    "How does the offside rule work in football?",
    "What is the tallest mountain in the world?",
]
 
 
class QAPair(BaseModel):
    question: str = Field(description="A self-contained question answered by the passage.")
    reference_answer: str = Field(description="A concise 1-3 sentence answer using only the passage.")
 
 
SYSTEM_PROMPT = (
    "You write evaluation questions for a document question-answering system.\n"
    "Given ONE passage from a document, write ONE question that the passage answers.\n"
    "Rules:\n"
    "- The question must make sense on its own, to someone who has NOT seen the passage. "
    "Never write 'according to the passage/text/document'.\n"
    "- Do NOT copy phrases from the passage. Paraphrase, so the question is not a keyword match.\n"
    "- Ask about a concrete idea, definition, list, cause or process - not trivia like page numbers.\n"
    "- The reference answer must be 1-3 sentences and use ONLY facts in the passage."
)
 
 
def letter_ratio(text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(len(text), 1)
 
 
def pick_chunks(vectorstore, n: int):
    """Choose n good chunks, at most one per (file, page), reproducibly."""
    data = vectorstore.get(include=["documents", "metadatas"])
    candidates = []
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if len(text) >= MIN_CHARS and letter_ratio(text) >= MIN_LETTER_RATIO:
            candidates.append({"chunk_id": cid, "text": text, "meta": meta or {}})
 
    rng = random.Random(SEED)
    rng.shuffle(candidates)
 
    chosen, seen_pages = [], set()
    for c in candidates:
        key = (c["meta"].get("filename"), c["meta"].get("page"))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        chosen.append(c)
        if len(chosen) == n:
            break
    return chosen, len(data["ids"])
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=35, help="number of answerable questions")
    args = parser.parse_args()
 
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is missing - check your .env file.")
    if not knowledge_base_exists():
        sys.exit("Knowledge base is empty. Run the app and upload your PDF first.")
 
    vectorstore = get_vectorstore()
    chunks, total = pick_chunks(vectorstore, args.n)
    print(f"Knowledge base has {total} chunks; sampled {len(chunks)} usable ones.\n")
 
    llm = ChatGroq(model=GENERATOR_MODEL, temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
    writer = llm.with_structured_output(QAPair)
 
    items = []
    for i, c in enumerate(chunks, 1):
        meta = c["meta"]
        page = meta.get("page")
        try:
            qa = writer.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Passage:\n{c['text']}"),
            ])
        except Exception as e:
            print(f"[{i}/{len(chunks)}] skipped ({e})")
            continue
 
        items.append({
            "id": f"ans_{len(items) + 1:03d}",
            "type": "answerable",
            "question": qa.question.strip(),
            "reference_answer": qa.reference_answer.strip(),
            "source_file": meta.get("filename"),
            "source_page": page + 1 if isinstance(page, int) else None,
            "chunk_id": c["chunk_id"],
            "chunk_text": c["text"],
            "keep": True,   # you will set this to false for bad questions in Step 9
        })
        print(f"[{i}/{len(chunks)}] {qa.question.strip()}")
        time.sleep(2)  # stay under free-tier rate limits
 
    for j, q in enumerate(OUT_OF_SCOPE, 1):
        items.append({
            "id": f"oos_{j:03d}",
            "type": "out_of_scope",
            "question": q,
            "reference_answer": None,
            "source_file": None,
            "source_page": None,
            "chunk_id": None,
            "chunk_text": None,
            "keep": True,
        })
 
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
 
    n_ans = sum(i["type"] == "answerable" for i in items)
    print(f"\nSaved {len(items)} items ({n_ans} answerable, {len(OUT_OF_SCOPE)} out-of-scope) to {OUT_PATH}")
    print("Next: open eval/dataset.json and review it (Step 9).")
 
 
if __name__ == "__main__":
    main()
 