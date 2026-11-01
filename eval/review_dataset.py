"""
Step 9: review the candidate questions one by one.
 
Run from the project root:
 
    python eval/review_dataset.py                 # review new, not-yet-reviewed questions
    python eval/review_dataset.py --flagged       # re-open KEPT questions with suspicious wording
    python eval/review_dataset.py --ids ans_015,ans_020   # re-open specific questions
 
For each answerable question you see the question, the reference answer and the
source text it was written from. Press:
    k = keep      d = drop      e = edit the question (and optionally the answer)
    c = clean up: automatically remove a lead-in like 'According to the passage,'
    q = save and quit (run the script again later to continue where you stopped)
 
Progress is saved after every decision.
"""
import argparse
import json
import os
import re
import sys
 
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "eval", "dataset.json")
 
# Wording that usually means the question is not self-contained or is time-sensitive
SUSPICIOUS = re.compile(
    r"\b(according to|the passage|the text|the document|this passage|this text|"
    r"the author|above|below|recent|recently|latest|currently|this chapter)\b",
    re.IGNORECASE,
)
 
 
LEAD_IN = re.compile(r"^\s*according to (the )?(passage|text|document|author)s?\s*,?\s*", re.IGNORECASE)
 
 
def auto_clean(question: str) -> str:
    """Remove a leading 'According to the passage,' and capitalize what is left."""
    cleaned = LEAD_IN.sub("", question).strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else question
 
 
def save(items):
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
 
 
def summary(items):
    ans = [i for i in items if i["type"] == "answerable"]
    kept = [i for i in ans if i.get("keep")]
    oos = [i for i in items if i["type"] == "out_of_scope" and i.get("keep")]
    print(f"\nAnswerable kept: {len(kept)} of {len(ans)}   |   out-of-scope kept: {len(oos)}")
    if len(kept) < 25:
        print("Tip: aim for roughly 25-30 good answerable questions. "
              "Edit (e) a weak question instead of dropping it when you can.")
 
 
def main():
    if not os.path.exists(PATH):
        sys.exit("eval/dataset.json not found - run generate_dataset.py first.")
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged", action="store_true",
                        help="re-open kept questions that have suspicious wording")
    parser.add_argument("--ids", type=str, default=None,
                        help="comma-separated question ids to re-open, e.g. ans_015,ans_020")
    args = parser.parse_args()
 
    with open(PATH, encoding="utf-8") as f:
        items = json.load(f)
 
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        todo = [i for i in items if i["id"] in wanted and i["type"] == "answerable"]
    elif args.flagged:
        todo = [i for i in items if i["type"] == "answerable" and i.get("keep")
                and SUSPICIOUS.search(i["question"])]
        if not todo:
            print("No kept question has suspicious wording. Nothing to fix.")
            summary(items)
            return
        print(f"{len(todo)} kept question(s) still have suspicious wording:")
        for i in todo:
            print(f"  {i['id']}: {i['question'][:90]}")
    else:
        todo = [i for i in items if i["type"] == "answerable" and not i.get("reviewed")]
        if not todo:
            print("Everything is already reviewed.")
            summary(items)
            return
 
    print("\nDROP a question if:")
    print("  1. it says 'the passage / the text / according to...' (nobody asks that)")
    print("  2. it is vague, so many parts of the book could answer it")
    print("  3. the reference answer is NOT supported by the source text shown")
    print("  4. it depends on when the book was written ('recent', 'latest')")
    print("EDIT it (e) if the idea is good but the wording is bad.\n")
 
    for n, item in enumerate(todo, 1):
        print("=" * 72)
        print(f"[{n}/{len(todo)}]  {item['id']}   (source: page {item['source_page']})")
        flags = sorted({m.lower() for m in SUSPICIOUS.findall(item["question"])})
        if flags:
            print(f"WARNING - suspicious wording: {', '.join(flags)}")
        print(f"\nQUESTION : {item['question']}")
        print(f"ANSWER   : {item['reference_answer']}")
        source = (item.get("chunk_text") or "").replace("\n", " ")
        print(f"\nSOURCE TEXT: {source[:600]}{'...' if len(source) > 600 else ''}\n")
 
        while True:
            choice = input("[k]eep  [d]rop  [e]dit  [c]lean-up  [q]uit & save > ").strip().lower()
            if choice == "c":
                cleaned = auto_clean(item["question"])
                if cleaned == item["question"]:
                    print("Nothing to clean automatically - use e to edit by hand.")
                    continue
                print(f"Cleaned question: {cleaned}")
                if input("Use it? [y/n] > ").strip().lower() == "y":
                    item["question"] = cleaned
                    item["edited"] = True
                    choice = "k"
                    break
                continue
            if choice in ("k", "d", "e", "q"):
                break
            print("Please type k, d, e, c or q.")
 
        if choice == "q":
            save(items)
            print("Saved. Run the script again to continue.")
            summary(items)
            return
 
        if choice == "e":
            new_q = input("New question (Enter = keep current): ").strip()
            if new_q:
                item["question"] = new_q
            new_a = input("New reference answer (Enter = keep current): ").strip()
            if new_a:
                item["reference_answer"] = new_a
            item["edited"] = True
            choice = "k"
 
        item["keep"] = (choice == "k")
        item["reviewed"] = True
        save(items)
 
    print("\nReview finished!")
    summary(items)
 
 
if __name__ == "__main__":
    main()