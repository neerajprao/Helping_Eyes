"""
Expiry test (FR-7, NFR-6): is the "has it expired?" verdict right, for label wordings and date formats seen on real
packs (the formats are the ones in the 148 pharmacy photos, plus food-label wordings)?

    .venv/bin/python eval/run_expiry.py              all stages (resumes: a case already answered is not asked again)
    .venv/bin/python eval/run_expiry.py --redo

Each case is a short label text with a manufacturing date and an expiry date. The correct verdict on the evaluation
date (common.EVAL_DATE, 19 October 2026) was worked out by hand when the case was written, not by the app's code.
A month-only date is valid to the end of that month.

Measured:
  code     assistant.expiry_dates(): the date the app's code finds, and its verdict (no model)
  app      "Has it expired?" through the app's own path (the code's verdict is in the prompt and in the question
           note), for Gemini and Qwen 3B
  llm_only the ablation in the requirements: the same question with the app's date checks taken out (the model gets
           only the text and today's date and has to work it out itself), for Gemini and Qwen 3B

Two "relative" cases (best before N months from manufacture) are reported apart: the app has no rule for them and
must say that the text gives no expiry date.

Output: eval/results/expiry.json (no photo text: these label texts are made up for the test).
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.RESULTS, "expiry.json")
QUESTION = "Has it expired?"
MODELS = ("gemini", "qwen3b")

# (id, label text, expired on 2026-10-19?)  True = expired, False = still good
CASES = [
    ("e01", "Paracetamol Tablets IP 500 mg\nB.No. PT2411\nMFG. DEC.24\nEXP. NOV.26", False),
    ("e02", "Cetirizine Tablets IP 10 mg\nMfg.Date: JAN.2025\nExp.Date: DEC.2026", False),
    ("e03", "Amoxycillin Capsules IP 500 mg\nMFD: 01/2024\nEXP: 12/2025", True),
    ("e04", "Pantoprazole Gastro-resistant Tablets\nMfg. APR. 2024\nExp. APR. 2026", True),
    ("e05", "Vitamin C chewable tablets\nMFG JUL 25\nEXP JUL 28", False),
    ("e06", "Azithromycin Tablets 500 mg\nMfg. Date DEC-2023\nExp. Date NOV-2025", True),
    ("e07", "Ibuprofen Tablets IP 400 mg\nMfd. 01 2024\nExp. 12.2026", False),
    ("e08", "Metformin Tablets 500 mg\nMFG: 01-2024\nEXP: 09-2026", True),
    ("e09", "Cough syrup 100 ml\nMfg. Dt. 11/24\nExp. Dt. 10/26", False),          # valid to 31 Oct 2026
    ("e10", "ORS powder sachet\nMFG: 09/24\nEXP: 08/26", True),
    ("e11", "Moisturising lotion 200 ml\nMFG 12/25\nEXP 06/28", False),
    ("e12", "Antiseptic liquid 500 ml\nMfg. 03/2023\nExp. 02/2026", True),
    ("e13", "Digestive biscuits 200 g\nPacked on: 15/04/2026\nBest before: 14/10/2026", True),
    ("e14", "Instant noodles 70 g\nPKD: 02.06.2026\nUse by: 02.12.2026", False),
    ("e15", "Fruit juice 1 L\nMfd: 20/07/2026\nBest Before 18/10/2026", True),  # one day before the evaluation date
    ("e16", "Milk powder 400 g\nMFG: 05/2026\nBEST BEFORE: 11/2027", False),
    ("e17", "Eye drops 10 ml\nMfg.: SEP.2024\nExp.: SEP.2026\nUse within one month of opening", True),
    ("e18", "Antacid gel 170 ml\nMFG.DATE 03/2025\nEXP.DATE 03/2027", False),
    ("e19", "Hand sanitiser 100 ml\nMfg: AUG 2024\nExpiry: AUG 2026", True),
    ("e20", "Calcium tablets\nMfg. Dt. 02/2026\nExp Dt. 01/2028", False),
    ("e21", "Throat lozenges\nMFG.: NOV.2023\nEXP.: OCT.2025", True),
    ("e22", "Pain relief gel 30 g\nMfg. 06/2025\nExp. 31/10/2026", False),
]
# Relative wordings: no date is printed; the right answer says the text gives no expiry date (or works it out
# correctly and says so). Reported apart, not in the accuracy above.
RELATIVE = [
    ("r01", "Peanut butter 340 g\nMfd: 03/2026\nBest before 6 months from manufacture", True),
    ("r02", "Tomato ketchup 500 g\nPKD: 05/2026\nBest before twelve months from packing", False),
]


def code_verdict(assistant, text):
    dates = assistant.expiry_dates(text)
    if not dates:
        return None, None
    printed, last_day = dates[0]
    return last_day < C.EVAL_DATE, f"{printed} -> valid until {last_day.isoformat()}"


def ask(case_text, provider, llm_only):
    import assistant
    saved = assistant.expiry_checks, assistant._EXPIRY_QUESTION
    if llm_only:                     # the app's date arithmetic taken out: the model works it out alone
        assistant.expiry_checks = lambda text, today=None: "None given: work it out from the text and today's date."
        assistant._EXPIRY_QUESTION = re.compile(r"(?!)")
    try:
        return C.ask(QUESTION, case_text, provider)
    finally:
        assistant.expiry_checks, assistant._EXPIRY_QUESTION = saved


def score(r, truth, relative, found):
    """Verdict right; and for a dated case, is the date it is valid until stated right (when one is stated)?"""
    from photo_report import month_years, says_expired, says_not_in_text
    if r["service_error"]:
        return {"says_expired": None, "correct": None, "date_ok": None}
    said = says_expired(r["answer"])
    if relative:     # no date printed: right = says the text gives none (without a wrong verdict), or the right verdict
        return {"says_expired": said, "correct": said == truth if said is not None else says_not_in_text(r), "date_ok": None}
    date_ok = None
    if found:
        stated = month_years(r["answer"])
        last = found.split("valid until ")[1]                    # YYYY-MM-DD of the last valid day
        date_ok = (int(last[5:7]), int(last[2:4])) in stated if stated else None
    return {"says_expired": said, "correct": said == truth, "date_ok": date_ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    assistant = C.fix_date()

    old = {} if args.redo else {r["id"]: r for r in C.load_json(OUT, {}).get("cases", [])}
    rows = []
    for cid, text, truth in CASES + RELATIVE:
        row = old.get(cid) or {"id": cid, "text": text, "expired": truth, "relative": cid.startswith("r")}
        verdict, found = code_verdict(assistant, text)
        row["code"] = {"verdict": verdict, "found": found,
                       "correct": (verdict == truth) if not row["relative"] else (verdict is None)}
        for cond in ("app", "llm_only"):
            for provider in MODELS:
                key = f"{cond}_{provider}"
                if key not in row or row[key].get("service_error"):
                    print(f"{cid} {key} ...", flush=True)
                    row[key] = ask(text, provider, cond == "llm_only")
                row[key].update(score(row[key], truth, row["relative"], row["code"]["found"]))
        rows.append(row)
        C.save_json(OUT, {"cases": rows})

    def tally(key, relative=False):
        sub = [r for r in rows if r["relative"] == relative]
        vals = [r[key]["correct"] for r in sub]
        scored = [v for v in vals if v is not None]
        dates = [r[key].get("date_ok") for r in sub if key != "code"]
        return {"correct": sum(scored), "scored": len(scored), "service_errors": len(vals) - len(scored),
                "accuracy": round(sum(scored) / len(scored), 4) if scored else None,
                "wrong": [r["id"] for r in sub if r[key]["correct"] is False],
                "date_stated": sum(d is not None for d in dates), "date_right": sum(d is True for d in dates),
                "date_wrong": [r["id"] for r in sub if key != "code" and r[key].get("date_ok") is False]}

    summary = {"date": C.EVAL_DATE.isoformat(), "cases": len(CASES), "relative_cases": len(RELATIVE),
               "expired_cases": sum(1 for _, _, t in CASES if t), "question": QUESTION,
               "models": {"gemini": assistant.LLM_MODEL, "qwen3b": assistant.QWEN_MODELS["qwen3b"]},
               "code": tally("code"), "code_relative": tally("code", True), **C.git_state()}
    for cond in ("app", "llm_only"):
        for provider in MODELS:
            summary[f"{cond}_{provider}"] = tally(f"{cond}_{provider}")
            summary[f"{cond}_{provider}_relative"] = tally(f"{cond}_{provider}", True)
    C.save_json(OUT, {"summary": summary, "cases": rows})
    for k, v in summary.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"{k:28s} {v['correct']}/{v['scored']}  wrong: {', '.join(v['wrong']) or '-'}   "
                  f"date stated {v['date_stated']}, wrong date: {', '.join(v['date_wrong']) or '-'}")


if __name__ == "__main__":
    main()
