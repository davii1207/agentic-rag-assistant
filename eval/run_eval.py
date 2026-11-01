"""
Step 10: run the evaluation.
 
Run from the project root (the folder that contains app.py):
 
    python eval/run_eval.py --name baseline
    python eval/run_eval.py --name no_retry --max-retries 0
    python eval/run_eval.py --name chunk500 --chunk-size 500 --overlap 100 --skip-agent
 
It does two things:
  A) RETRIEVAL TEST (no LLM): does the retriever find the right PAGE for each question?
  B) END-TO-END TEST: runs the whole agent on every question and records the answer,
     route (docs/web), sources, retries and latency.
 
Everything is saved to eval/results/<name>.json. The evaluation uses its OWN index built
only from your textbook (eval/indexes/), so your resume and the app's knowledge base
never affect the results.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
 
from dotenv import load_dotenv
from langchain_chroma import Chroma
 
load_dotenv(os.path.join(ROOT, ".env"))
 
import rag_pipeline as rp
import agent_graph as ag
 
DATASET_PATH = os.path.join(ROOT, "eval", "dataset.json")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")
INDEX_DIR = os.path.join(ROOT, "eval", "indexes")
 
# If one question takes longer than this, we assume a network call hung, give up on it
# and try again (instead of freezing the whole run).
QUESTION_TIMEOUT_S = 120
 
 
# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def find_corpus_file(text: str) -> str:
    data_dir = os.path.join(ROOT, "data")
    matches = [
        f for f in os.listdir(data_dir)
        if text.lower() in f.lower() and f.lower().endswith(rp.SUPPORTED_EXTENSIONS)
    ]
    if len(matches) != 1:
        sys.exit(f"--file '{text}' must match exactly one file in data/, found: {matches}")
    return os.path.join(data_dir, matches[0])
 
 
def get_eval_store(corpus_path: str, chunk_size: int, overlap: int):
    """Build (or reuse) an index containing only the evaluation corpus."""
    rp.CHUNK_SIZE = chunk_size
    rp.CHUNK_OVERLAP = overlap
 
    tag = hashlib.md5(os.path.basename(corpus_path).encode()).hexdigest()[:6]
    index_path = os.path.join(INDEX_DIR, f"cs{chunk_size}_ov{overlap}_{tag}")
 
    store = Chroma(
        collection_name="eval_index",
        persist_directory=index_path,
        embedding_function=rp.get_embedding_model(),
    )
    if len(store.get(limit=1)["ids"]) == 0:
        print(f"Building evaluation index (chunk_size={chunk_size}, overlap={overlap})...")
        chunks = rp.split_documents(rp.load_file(corpus_path))
        store.add_documents(chunks, ids=rp.make_chunk_ids(chunks))
    else:
        print(f"Reusing existing evaluation index: {os.path.basename(index_path)}")
 
    return store, len(store.get()["ids"])
 
 
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
 
 
# --------------------------------------------------------------------------
# A) Retrieval test
# --------------------------------------------------------------------------
def eval_retrieval(store, items, max_k: int = 10):
    """For each answerable question: at what rank does the correct PAGE appear?"""
    retriever = rp.get_retriever(store, k=max_k)
    rows = []
    for it in items:
        docs = retriever.invoke(it["question"])
        rank = None
        for r, d in enumerate(docs, 1):
            page = d.metadata.get("page")
            if (d.metadata.get("filename") == it["source_file"]
                    and isinstance(page, int) and page + 1 == it["source_page"]):
                rank = r
                break
        rows.append({
            "id": it["id"],
            "rank": rank,
            "retrieved_pages": [d.metadata.get("page", -1) + 1 for d in docs],
        })
 
    n = len(rows)
 
    def hit(k):
        return round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3)
 
    metrics = {
        "n_questions": n,
        "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "hit@10": hit(10),
        "mrr@10": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
    }
    return metrics, rows
 
 
# --------------------------------------------------------------------------
# B) End-to-end agent test
# --------------------------------------------------------------------------
def _invoke_with_timeout(agent, question: str, timeout: float):
    """Run agent.invoke in a background thread so a hung network call cannot freeze us."""
    box = {}
 
    def work():
        try:
            box["out"] = agent.invoke(ag.initial_state(question))
        except Exception as e:  # noqa: BLE001
            box["err"] = e
 
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no answer within {timeout:.0f}s (a network call probably hung)")
    if "err" in box:
        raise box["err"]
    return box["out"]
 
 
def run_agent(agent, question: str):
    """Run the agent once; retry if Groq rate-limits us or a call hangs."""
    for attempt in range(1, 4):
        try:
            t0 = time.perf_counter()
            out = _invoke_with_timeout(agent, question, QUESTION_TIMEOUT_S)
            return out, time.perf_counter() - t0, None
        except Exception as e:
            msg = str(e)
            timed_out = isinstance(e, TimeoutError)
            rate_limited = "429" in msg or "rate" in msg.lower()
            if (timed_out or rate_limited) and attempt < 3:
                wait = 0 if timed_out else 15 * attempt
                print(f"      {'timed out' if timed_out else 'rate limited'} - "
                      f"retrying{f' in {wait}s' if wait else ''}...")
                time.sleep(wait)
                continue
            return None, None, msg[:300]
 
 
def eval_agent(items):
    agent = ag.build_agent()
    records = []
    for i, it in enumerate(items, 1):
        out, latency, err = run_agent(agent, it["question"])
        rec = {"id": it["id"], "type": it["type"], "question": it["question"], "error": err}
        if out is not None:
            rec.update({
                "answer": out["answer"],
                "source": out["source"],
                "citations": out["citations"],
                "contexts": out["documents"],
                "steps": out["steps"],
                "retries": out["retries"],
                "latency_s": round(latency, 2),
            })
            print(f"[{i}/{len(items)}] {out['source']:<4} retries={out['retries']} "
                  f"{latency:5.1f}s  {it['question'][:60]}")
        else:
            print(f"[{i}/{len(items)}] ERROR {err[:80]}")
        records.append(rec)
        time.sleep(1)  # be gentle with free-tier rate limits
    return records
 
 
def agent_summary(records):
    ok = [r for r in records if not r["error"]]
    ans = [r for r in ok if r["type"] == "answerable"]
    oos = [r for r in ok if r["type"] == "out_of_scope"]
    routed_ans = sum(r["source"] == "docs" for r in ans)
    routed_oos = sum(r["source"] == "web" for r in oos)
    lat = sorted(r["latency_s"] for r in ok)
    return {
        "n_run": len(records),
        "n_errors": len(records) - len(ok),
        "answerable_routed_to_docs": f"{routed_ans}/{len(ans)}",
        "out_of_scope_routed_to_web": f"{routed_oos}/{len(oos)}",
        "routing_accuracy": round((routed_ans + routed_oos) / max(len(ok), 1), 3),
        "avg_retries": round(sum(r["retries"] for r in ok) / max(len(ok), 1), 2),
        "latency_avg_s": round(sum(lat) / max(len(lat), 1), 2),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_max_s": lat[-1] if lat else None,
    }
 
 
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="name of this run, e.g. baseline")
    p.add_argument("--file", default="Introduction", help="text matching your textbook's file name")
    p.add_argument("--k", type=int, default=4, help="chunks retrieved per query")
    p.add_argument("--max-retries", type=int, default=1, help="query-rewrite retries (0 = off)")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--skip-agent", action="store_true", help="only run the retrieval test")
    args = p.parse_args()
 
    if not os.path.exists(DATASET_PATH):
        sys.exit("eval/dataset.json not found - finish Steps 8 and 9 first.")
    if not args.skip_agent and not (os.getenv("GROQ_API_KEY") and os.getenv("TAVILY_API_KEY")):
        sys.exit("GROQ_API_KEY and TAVILY_API_KEY must be set in .env for the agent test.")
 
    with open(DATASET_PATH, encoding="utf-8") as f:
        items = [i for i in json.load(f) if i.get("keep")]
    answerable = [i for i in items if i["type"] == "answerable"]
    print(f"Dataset: {len(answerable)} answerable + {len(items) - len(answerable)} out-of-scope questions")
 
    corpus = find_corpus_file(args.file)
    store, n_chunks = get_eval_store(corpus, args.chunk_size, args.overlap)
    print(f"Evaluation index has {n_chunks} chunks\n")
 
    # --- A) retrieval ---
    ret_metrics, ret_rows = eval_retrieval(store, answerable)
    print("RETRIEVAL (is the correct page in the top-k?)")
    for k, v in ret_metrics.items():
        print(f"   {k:<12} {v}")
    print()
 
    # --- B) agent ---
    agent_records, agent_metrics = [], None
    if not args.skip_agent:
        ag.get_vectorstore = lambda: store      # point the agent at the evaluation index
        ag.RETRIEVAL_K = args.k
        ag.MAX_RETRIES = args.max_retries
        print(f"END-TO-END AGENT (k={args.k}, max_retries={args.max_retries})")
        agent_records = eval_agent(items)
        agent_metrics = agent_summary(agent_records)
        print("\nAGENT SUMMARY")
        for k, v in agent_metrics.items():
            print(f"   {k:<28} {v}")
 
    # --- save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.name}.json")
    result = {
        "config": {
            "name": args.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "corpus_file": os.path.basename(corpus),
            "n_chunks": n_chunks,
            "k": args.k,
            "max_retries": args.max_retries,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "agent_model": ag.LLM_MODEL,
            "embedding_model": rp.EMBEDDING_MODEL,
            "n_answerable": len(answerable),
            "n_out_of_scope": len(items) - len(answerable),
        },
        "retrieval": {"metrics": ret_metrics, "per_question": ret_rows},
        "agent": {"metrics": agent_metrics, "records": agent_records},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")
 
 
if __name__ == "__main__":
    main()