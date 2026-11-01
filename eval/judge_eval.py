"""
Step 11: score the answers from a finished run with an LLM judge.
 
Run from the project root:
 
    python eval/judge_eval.py --name baseline
 
Reads  eval/results/<name>.json        (made by run_eval.py)
Writes eval/results/<name>.scores.json (so you can re-judge without re-running the agent)
 
Three checks, each returned as a structured verdict with a short reasoning:
  1. CORRECTNESS   (answerable questions) - does the answer match the reference answer?
  2. FAITHFULNESS  (all questions)        - is every claim supported by the retrieved context?
  3. RELEVANCE     (out-of-scope questions) - does the answer actually address the question?
 
Note: free-tier Groq only offers the gpt-oss models, so the default judge is the larger
gpt-oss-120b while the agent uses gpt-oss-20b. Same family, so we validate the judge
against human labels (Step 11b) and say so openly in the write-up.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Literal
 
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "eval", "results")
DATASET_PATH = os.path.join(ROOT, "eval", "dataset.json")
 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
 
load_dotenv(os.path.join(ROOT, ".env"))
 
DEFAULT_JUDGE = "openai/gpt-oss-120b"
MAX_CONTEXT_CHARS = 6000
 
 
# --------------------------------------------------------------------------
# Verdict formats (reasoning comes first so the judge thinks before deciding)
# --------------------------------------------------------------------------
class Correctness(BaseModel):
    reasoning: str = Field(description="1-2 sentences comparing the answer with the reference.")
    verdict: Literal["correct", "partial", "incorrect"]
 
 
class Faithfulness(BaseModel):
    reasoning: str = Field(description="1-2 sentences on whether the claims are supported.")
    unsupported_claims: List[str] = Field(description="Claims NOT supported by the context (empty if none).")
    faithful: bool
 
 
class Relevance(BaseModel):
    reasoning: str = Field(description="1-2 sentences.")
    answers_question: bool
 
 
CORRECTNESS_SYSTEM = (
    "You are a strict grader comparing a candidate answer with a reference answer.\n"
    "- correct: the candidate contains the key facts of the reference and does not contradict it. "
    "Extra correct detail is fine.\n"
    "- partial: it misses some key facts or contains a minor error.\n"
    "- incorrect: it is wrong, contradicts the reference, or says the information is not available.\n"
    "Judge the substance only - ignore wording, style and length."
)
 
FAITHFULNESS_SYSTEM = (
    "You check whether an ANSWER is supported by the CONTEXT it was given.\n"
    "Go through the factual claims in the answer. faithful=true only if EVERY factual claim is "
    "supported by the context. Claims that need outside knowledge are unsupported. "
    "If the answer honestly says the information is not available, it is faithful."
)
 
RELEVANCE_SYSTEM = (
    "You decide whether an ANSWER directly addresses the QUESTION asked. "
    "Ignore whether it is factually correct; only judge if it attempts to answer what was asked."
)
 
 
# --------------------------------------------------------------------------
def make_judge(model: str):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is missing - check your .env file.")
    return ChatGroq(model=model, temperature=0, api_key=key)
 
 
def call_judge(judge, schema, system: str, user: str):
    """One structured judge call with retries. Returns None if it keeps failing."""
    structured = judge.with_structured_output(schema)
    for attempt in range(1, 4):
        try:
            return structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as e:
            msg = str(e)
            rate = "429" in msg or "rate" in msg.lower()
            wait = 15 * attempt if rate else 2
            if attempt < 3:
                print(f"      judge call failed ({'rate limit' if rate else 'error'}); retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"      judge call gave up: {msg[:120]}")
    return None
 
 
def judge_record(judge, rec, reference):
    ctx = "\n\n".join(rec.get("contexts") or [])[:MAX_CONTEXT_CHARS]
    out = {"id": rec["id"], "type": rec["type"], "source": rec.get("source")}
 
    faith = call_judge(
        judge, Faithfulness, FAITHFULNESS_SYSTEM,
        f"CONTEXT:\n{ctx or '(no context was retrieved)'}\n\nANSWER:\n{rec['answer']}",
    )
    out["faithfulness"] = faith.model_dump() if faith else None
 
    if rec["type"] == "answerable":
        corr = call_judge(
            judge, Correctness, CORRECTNESS_SYSTEM,
            f"QUESTION: {rec['question']}\n\nREFERENCE ANSWER: {reference}\n\nCANDIDATE ANSWER: {rec['answer']}",
        )
        out["correctness"] = corr.model_dump() if corr else None
    else:
        rel = call_judge(
            judge, Relevance, RELEVANCE_SYSTEM,
            f"QUESTION: {rec['question']}\n\nANSWER: {rec['answer']}",
        )
        out["relevance"] = rel.model_dump() if rel else None
    return out
 
 
def pct(x, n):
    return round(x / n, 3) if n else None
 
 
def aggregate(rows):
    ans = [r for r in rows if r["type"] == "answerable" and r.get("correctness")]
    oos = [r for r in rows if r["type"] == "out_of_scope" and r.get("relevance")]
    faith = [r for r in rows if r.get("faithfulness")]
 
    c = [r["correctness"]["verdict"] for r in ans]
    metrics = {
        "n_answerable_judged": len(ans),
        "correctness_strict": pct(c.count("correct"), len(c)),
        "correctness_lenient": pct(c.count("correct") + 0.5 * c.count("partial"), len(c)),
        "verdicts": {v: c.count(v) for v in ("correct", "partial", "incorrect")},
        "n_faithfulness_judged": len(faith),
        "faithfulness_overall": pct(sum(r["faithfulness"]["faithful"] for r in faith), len(faith)),
        "n_out_of_scope_judged": len(oos),
        "oos_answers_question": pct(sum(r["relevance"]["answers_question"] for r in oos), len(oos)),
    }
    for src in ("docs", "web"):
        sub = [r for r in faith if r["source"] == src]
        metrics[f"faithfulness_{src}"] = pct(sum(r["faithfulness"]["faithful"] for r in sub), len(sub))
    metrics["n_judge_failures"] = len(rows) - len(
        [r for r in rows if r.get("faithfulness") and (r.get("correctness") or r.get("relevance"))]
    )
    return metrics
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name, e.g. baseline")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    args = p.parse_args()
 
    res_path = os.path.join(RESULTS_DIR, f"{args.name}.json")
    if not os.path.exists(res_path):
        sys.exit(f"{res_path} not found - run run_eval.py --name {args.name} first.")
 
    with open(res_path, encoding="utf-8") as f:
        result = json.load(f)
    with open(DATASET_PATH, encoding="utf-8") as f:
        reference = {i["id"]: i.get("reference_answer") for i in json.load(f)}
 
    records = [r for r in result["agent"]["records"] if not r.get("error")]
    if not records:
        sys.exit("This run has no agent records (it was run with --skip-agent).")
 
    judge = make_judge(args.judge_model)
    print(f"Judging {len(records)} answers with {args.judge_model}...\n")
 
    rows = []
    for i, rec in enumerate(records, 1):
        row = judge_record(judge, rec, reference.get(rec["id"]))
        rows.append(row)
        if rec["type"] == "answerable":
            tag = (row.get("correctness") or {}).get("verdict", "?")
        else:
            tag = "relevant" if (row.get("relevance") or {}).get("answers_question") else "off-topic"
        fa = "faithful" if (row.get("faithfulness") or {}).get("faithful") else "UNFAITHFUL"
        print(f"[{i}/{len(records)}] {rec['id']:<8} {tag:<10} {fa:<10} {rec['question'][:55]}")
        time.sleep(1)
 
    metrics = aggregate(rows)
 
    print("\nSCORES")
    for k, v in metrics.items():
        print(f"   {k:<26} {v}")
 
    questions = {r["id"]: r["question"] for r in records}
    weak = [r for r in rows if r["type"] == "answerable"
            and r.get("correctness") and r["correctness"]["verdict"] != "correct"]
    if weak:
        print("\nANSWERS TO INSPECT (not fully correct):")
        for r in weak[:10]:
            print(f" - {r['id']} [{r['correctness']['verdict']}] {questions[r['id']][:70]}")
            print(f"     {r['correctness']['reasoning'][:160]}")
    unfaithful = [r for r in rows if r.get("faithfulness") and not r["faithfulness"]["faithful"]]
    if unfaithful:
        print("\nUNFAITHFUL ANSWERS:")
        for r in unfaithful[:10]:
            print(f" - {r['id']} ({r['source']}) {questions[r['id']][:60]}")
            for claim in r["faithfulness"]["unsupported_claims"][:2]:
                print(f"     unsupported: {claim[:120]}")
 
    out_path = os.path.join(RESULTS_DIR, f"{args.name}.scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "run": args.name,
            "judge_model": args.judge_model,
            "agent_model": result["config"].get("agent_model"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics,
            "per_question": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved scores to {out_path}")
 
 
if __name__ == "__main__":
    main()