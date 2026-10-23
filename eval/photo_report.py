"""
Scoring and report for eval/run_photo_eval.py (stages "score" and "report").

The reference for every photo is Claude's own reading of it: eval/data/private/claude_reference/<id>.json, written by
looking only at the photo (cropped to where text was found and turned upright), before any engine's text was read:

    {"id": "ms01", "item": "short description", "lighting": "good|dim|glare", "angle": "flat|tilted|curved",
     "legibility": "clear|partly|poor",
     "fields": {"name", "strength", "quantity", "maker", "batch", "mfg", "expiry", "price"},   value as printed;
               null = not on the part of the pack in the photo; "?" = printed but not readable in the photo
     "flags": {"rx": true|false|null, "external_use": true|false|null},
     "key_lines": ["the most important printed lines, exactly as printed", ...],
     "notes": ""}

It is one careful reader, not a typed ground truth checked by a second person, and it says so in the report.

Measures:
  OCR key-line accuracy  each key line is found in the engine's text with the fewest character edits (a line may sit
                         anywhere in the text, so this is an approximate substring match); accuracy = 1 - edits /
                         characters, over all key lines. Text is compared after NFKC, case folding and joining
                         whitespace, because the reference cannot be sure of case on every pack.
  OCR field found        the field's letters and digits (case folded) appear in the engine's text, letters and digits only.
  Answer scored          factual questions only (name, strength, quantity, expiry, expired, mfg, batch, price, maker,
                         prescription, external use), by fixed rules against the reference; a field the pack does not
                         show must be answered "not in the text". Free-text questions are listed but not scored, and
                         every answer is checked for numbers that are not in the text the model was given.
"""

import datetime
import json
import os
import re
import statistics
import unicodedata

import numpy as np

import run_photo_eval as R

TODAY = datetime.date.today()
ENGINES = ["apple_vision", "app_capture", "rapidocr", "tesseract"]
FIELDS = ["name", "strength", "quantity", "maker", "batch", "mfg", "expiry", "price"]
SCORED = {"name", "strength", "quantity", "expiry", "expired", "mfg", "batch", "price", "maker", "prescription", "external"}
REPORT_MD = os.path.join(R.ROOT, "markdown", "eval_photo_report.md")
SUMMARY_MD = os.path.join(R.ROOT, "markdown", "eval_photo_summary.md")
SUMMARY_JSON = os.path.join(R.EVAL, "results", "photo_eval_summary.json")
SCORES = os.path.join(R.OUT, "scores")

_NOT_IN_TEXT = re.compile(r"\b(not (?:mentioned|stated|given|listed|shown|provided|specified|found|included|available|"
                          r"visible|printed|clear)|doesn'?t (?:say|mention|show|list|state)|does not (?:say|mention|"
                          r"show|list|state|specify|include|contain)|no (?:information|mention|details?)|"
                          r"isn'?t (?:mentioned|stated|shown|listed)|is not (?:mentioned|stated|shown|listed)|"
                          r"can'?t (?:find|see|tell)|cannot (?:find|see|tell)|unclear|not sure)\b", re.I)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return " ".join(s.split())


def alnum(s):
    return re.sub(r"[^0-9a-z]", "", norm(s))


def substring_edits(ref, text):
    """Fewest character edits to turn `ref` into some substring of `text` (Sellers' algorithm)."""
    if not ref:
        return 0
    if not text:
        return len(ref)
    t = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    prev = np.zeros(len(t) + 1, dtype=np.int64)                     # a match may start anywhere: row 0 is all zeros
    steps = np.arange(len(t) + 1)
    for i, ch in enumerate(ref, 1):
        best = np.empty_like(prev)
        best[0] = i
        best[1:] = np.minimum(prev[:-1] + (t != ord(ch)), prev[1:] + 1)   # substitute / match, or delete
        # inserting text characters: cur[j] = min over k <= j of best[k] + (j - k), a running minimum
        prev = steps + np.minimum.accumulate(best - steps)
    return int(prev.min())


def month_years(s):
    """Every (month, year) written in s, in the forms labels use: 12/2026, 12/26, 12-2026, DEC 2026, Dec-26, 12.2026."""
    s = norm(s)
    found = set()
    for m, y in re.findall(r"(?<!\d)(\d{1,2})\s*[/.\-]\s*(\d{4}|\d{2})(?!\d)", s):
        m, y = int(m), int(y)
        if 1 <= m <= 12:
            found.add((m, y % 100))
    for m, y in re.findall(r"(?<!\d)(\d{1,2})\s+(20\d{2})(?!\d)", s):      # "Mfd. 01 2024"
        if 1 <= int(m) <= 12:
            found.add((int(m), int(y) % 100))
    for name, y in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)", s):
        found.add((_MONTHS[name], int(y) % 100))                          # "December 31, 2026"
    s = re.sub(r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)\s+\d{1,2}(?:st|nd|rd|th)?,(?=\s*\d{4})", r"\1", s)
    for name, y in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*[\-/,.']?\s*(\d{4}|\d{2})(?!\d)", s):
        found.add((_MONTHS[name], int(y) % 100))
    for d, m, y in re.findall(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})(?!\d)", s):
        if 1 <= int(m) <= 12:
            found.add((int(m), int(y) % 100))
    return found


def expired_on(expiry, today=TODAY):
    my = sorted(month_years(expiry))
    if not my:
        return None
    m, y = my[0]
    return (2000 + y, m) < (today.year, today.month)          # a label month is valid to its end


def says_not_in_text(a):
    return bool(a.get("not_in_text")) or bool(_NOT_IN_TEXT.search(a["answer"]))


def says_expired(answer):
    """True / False when the answer says expired / not expired; an explicit statement wins over "valid until"."""
    a = norm(answer)
    if re.search(r"\b(not|hasn'?t|has not|isn'?t|is not)\b[^.]{0,20}\bexpired\b|\bnot yet expired\b", a):
        return False
    if re.search(r"\b(has|have|had|is|already)\b[^.]{0,15}\bexpired\b|\bexpired on\b|\bpast (?:its|the) expiry\b", a):
        return True
    if re.search(r"\bstill (?:valid|good|safe|usable|in date)\b|\bvalid (?:until|till|through|to)\b", a):
        return False
    return None


def code(s):
    """A batch code for comparing: letters and digits only, with O read as 0 and I as 1 (inkjet makes them ambiguous)."""
    return alnum(s).replace("o", "0").replace("i", "1")


def numbers(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s.replace(",", "")))


def score_answer(key, a, ref, document):
    """(True | False | None, why). None = not scored."""
    f = ref["fields"]
    if key not in SCORED:
        return None, "free text, not scored"
    if key == "expired":
        e = f.get("expiry")
        if e in (None, "?"):
            return (says_not_in_text(a), "no expiry date on the photo: should say so") if e is None else (None, "expiry unreadable")
        truth, said = expired_on(e), says_expired(a["answer"])
        if truth is None:
            return None, "expiry date not parseable"
        return (said == truth, f"expiry {e}: {'expired' if truth else 'not expired'} on {TODAY}; answer says "
                               f"{'expired' if said else 'not expired' if said is False else 'neither'}")
    if key == "prescription":
        rx = ref["flags"].get("rx")
        if not rx:
            return None, "no Rx mark seen: not scored"
        if says_not_in_text(a):
            return False, "Rx printed on the pack; the answer says the text does not tell"
        yes = re.search(r"\b(yes|requires?|needs?|only (?:on|with|by)|prescription (?:only|drug|is required|required))\b|schedule\s*h", norm(a["answer"]))
        return bool(yes) and not re.search(r"\b(no|does not|doesn'?t) (?:need|require)", norm(a["answer"])), "Rx printed on the pack"
    if key == "external":
        ext = ref["flags"].get("external_use")
        if not ext:
            return None, "no 'external use' mark seen: not scored"
        return bool(re.search(r"\b(yes|external use only|for external use|only for external|apply)\b", norm(a["answer"]))), "'external use only' printed"
    field = {"name": "name", "strength": "strength", "quantity": "quantity", "expiry": "expiry", "mfg": "mfg",
             "batch": "batch", "price": "price", "maker": "maker"}[key]
    value = f.get(field)
    if value == "?":
        return None, "printed but unreadable in the photo"
    if value is None:
        return says_not_in_text(a), "not on the pack in the photo: should say so"
    ans = a["answer"]
    options = [v.strip() for v in value.split(" / ") if v.strip()]       # "Bayer AG / Bayer Zydus Pharma": either one
    if len(options) > 1:
        results = [score_answer(key, a, {**ref, "fields": {**f, field: o}}, document) for o in options]
        return any(ok for ok, _ in results), f"{field} {value}"
    if field in ("expiry", "mfg"):
        want = month_years(value)
        return (bool(want & month_years(ans)), f"date {value}") if want else (alnum(value) in alnum(ans), f"{value}")
    if field == "batch":
        return code(value) in code(ans), f"batch {value}"
    if field == "price":
        whole = {n.split(".")[0] for n in numbers(value)}
        return bool(whole & {n.split(".")[0] for n in numbers(ans)}), f"price {value}"
    if field in ("strength", "quantity"):
        want = numbers(re.sub(r"^\s*1\s*x\s*", "", value, flags=re.I))      # "1x10 Tablets" is 10 tablets
        return (want <= numbers(ans), f"{value}") if want else (alnum(value) in alnum(ans), value)
    if field == "name":
        brand = alnum(value.split()[0]) if value.split() else ""
        return (brand in alnum(ans) if brand else None), f"name {value}"
    if field == "maker":
        words = [w for w in re.findall(r"[A-Za-z]{3,}", value) if w.lower() not in
                 ("ltd", "limited", "pvt", "private", "the", "and", "pharmaceuticals", "pharma", "healthcare", "industries", "india", "inc")]
        return (alnum(words[0]) in alnum(ans) if words else None), f"maker {value}"
    return None, ""


def unsupported_numbers(answer, document):
    import assistant
    doc = numbers(document) | numbers(assistant.expiry_checks(document))   # the app's own date note is given to the model
    return sorted(n for n in numbers(answer) if n not in doc and len(n.replace(".", "")) >= 2)


def references(pids):
    out = {}
    for pid in pids:
        p = os.path.join(R.REFS, pid + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[pid] = json.load(f)
    return out


def score(pids):
    refs = references(pids)
    os.makedirs(SCORES, exist_ok=True)
    rows = []
    for pid in pids:
        ocr, qa, cv, ref = R.read("ocr", pid), R.read("qa", pid), R.read("cv", pid), refs.get(pid)
        if not (ocr and qa and cv and ref):
            print(f"score {pid}: missing {'ocr ' if not ocr else ''}{'qa ' if not qa else ''}{'cv ' if not cv else ''}{'reference' if not ref else ''}")
            continue
        engines = {}
        for e in ENGINES:
            text = norm(" ".join(l["text"] for l in ocr["engines"][e]["lines"]))
            flat = alnum(text)
            lines = [norm(l) for l in ref["key_lines"] if norm(l)]
            edits = [substring_edits(l, text) for l in lines]
            chars = sum(len(l) for l in lines)
            fields = {k: (any((code(o) in code(text)) if k == "batch" else (alnum(o) in flat)
                              for o in v.split(" / ") if alnum(o)) if alnum(v) else None)
                      for k, v in ref["fields"].items() if v not in (None, "?") and k in FIELDS}
            engines[e] = {"key_line_chars": chars, "key_line_edits": sum(edits),
                          "key_line_accuracy": round(1 - sum(edits) / chars, 4) if chars else None,
                          "per_line": [{"line": l, "edits": d} for l, d in zip(ref["key_lines"], edits)],
                          "fields": fields, "seconds": ocr["engines"][e]["seconds"],
                          "lines": len(ocr["engines"][e]["lines"])}
        document = "\n".join(l["text"] for l in ocr["engines"]["app_capture"]["lines"])
        answers = []
        for a in qa["answers"]:
            ok, why = score_answer(a["key"], a, ref, document) if a["key"] != "read_all" else (None, "the app's own text, no model")
            answers.append({**a, "correct": ok, "why": why,
                            "numbers_not_in_text": unsupported_numbers(a["answer"], document) if a["key"] != "read_all" else []})
        rows.append({"id": pid, "engines": engines, "answers": answers})
        with open(os.path.join(SCORES, pid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rows[-1], f, indent=1, ensure_ascii=False)
    print(f"scored {len(rows)} photos")
    return rows


def _load_scores(pids):
    out = []
    for pid in pids:
        p = os.path.join(SCORES, pid + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def bootstrap(values, weights=None, n=10000, seed=7):
    """95 % percentile interval of a ratio over photos (sum(values) / sum(weights)), resampling photos."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if not len(v) or w.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    ratios = v[idx].sum(1) / np.maximum(w[idx].sum(1), 1e-9)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def summarise(rows, man, refs):
    cats = sorted({man[r["id"]]["category"] for r in rows})
    summary = {"photos": len(rows), "date": TODAY.isoformat(), "reference": "Claude's reading of each photo",
               "qa_model": None, "engines": {}, "by_category": {}, "answers": {}, "checks": {}}
    for e in ENGINES:
        edits = [r["engines"][e]["key_line_edits"] for r in rows]
        chars = [r["engines"][e]["key_line_chars"] for r in rows]
        found = [sum(1 for v in r["engines"][e]["fields"].values() if v) for r in rows]
        total = [sum(1 for v in r["engines"][e]["fields"].values() if v is not None) for r in rows]
        acc = 1 - sum(edits) / max(1, sum(chars))
        ci = bootstrap(np.array(chars) - np.array(edits), chars)
        summary["engines"][e] = {
            "key_line_accuracy": round(acc, 4), "key_line_accuracy_ci95": [round(c, 4) for c in ci] if ci else None,
            "fields_found": sum(found), "fields_total": sum(total),
            "fields_found_rate": round(sum(found) / max(1, sum(total)), 4),
            "fields_found_ci95": [round(c, 4) for c in bootstrap(found, total)] if sum(total) else None,
            "seconds_mean": round(statistics.mean(r["engines"][e]["seconds"] for r in rows), 3),
            "seconds_median": round(statistics.median(r["engines"][e]["seconds"] for r in rows), 3),
            "photos_with_no_text": sum(1 for r in rows if r["engines"][e]["lines"] == 0)}
        per_field = {}
        for k in FIELDS:
            vals = [r["engines"][e]["fields"][k] for r in rows if k in r["engines"][e]["fields"]]
            if vals:
                per_field[k] = [sum(vals), len(vals)]
        summary["engines"][e]["per_field"] = per_field
    for c in cats:
        sub = [r for r in rows if man[r["id"]]["category"] == c]
        summary["by_category"][c] = {"photos": len(sub)}
        for e in ENGINES:
            ed = sum(r["engines"][e]["key_line_edits"] for r in sub)
            ch = sum(r["engines"][e]["key_line_chars"] for r in sub)
            summary["by_category"][c][e] = round(1 - ed / max(1, ch), 4)
    for cond, test in (("good light", lambda m: m["lighting"] == "good"), ("glare", lambda m: m["lighting"] == "glare"),
                       ("dim", lambda m: m["lighting"] == "dim"), ("flat", lambda m: m["angle"] == "flat"),
                       ("tilted", lambda m: m["angle"] == "tilted"), ("curved", lambda m: m["angle"] == "curved")):
        sub = [r for r in rows if test(man[r["id"]])]
        if sub:
            summary["by_category"]["condition: " + cond] = {"photos": len(sub), **{
                e: round(1 - sum(r["engines"][e]["key_line_edits"] for r in sub) / max(1, sum(r["engines"][e]["key_line_chars"] for r in sub)), 4)
                for e in ENGINES}}
    # answers
    by_key = {}
    for r in rows:
        for a in r["answers"]:
            d = by_key.setdefault(a["key"], {"question": a["question"], "asked": 0, "scored": 0, "correct": 0,
                                             "said_not_in_text": 0, "with_numbers_not_in_text": 0, "seconds": []})
            d["asked"] += 1
            d["seconds"].append(a["seconds"])
            if a.get("not_in_text"):
                d["said_not_in_text"] += 1
            if a.get("numbers_not_in_text"):
                d["with_numbers_not_in_text"] += 1
            if a["correct"] is not None:
                d["scored"] += 1
                d["correct"] += bool(a["correct"])
    for k, d in by_key.items():
        d["seconds_mean"] = round(statistics.mean(d.pop("seconds")), 2)
    summary["answers"] = by_key
    per_photo_correct = [sum(1 for a in r["answers"] if a["correct"]) for r in rows]
    per_photo_scored = [sum(1 for a in r["answers"] if a["correct"] is not None) for r in rows]
    summary["answers_overall"] = {"scored": sum(per_photo_scored), "correct": sum(per_photo_correct),
                                  "accuracy": round(sum(per_photo_correct) / max(1, sum(per_photo_scored)), 4),
                                  "ci95": [round(c, 4) for c in bootstrap(per_photo_correct, per_photo_scored)]}
    # image checks
    cvs = {r["id"]: R.read("cv", r["id"]) for r in rows}
    probs = {}
    for c in cvs.values():
        for p in c["quality_text"]["problems"]:
            probs[p] = probs.get(p, 0) + 1
    summary["checks"] = {
        "problems_flagged": probs,
        "photos_with_glare_blobs": sum(1 for c in cvs.values() if c["glare_blobs"]),
        "photos_with_blurry_text_patches": sum(1 for c in cvs.values() if c["blurry_patches"]),
        "photos_with_dark_patches": sum(1 for c in cvs.values() if c["dark_patches"]),
        "photos_with_slant": sum(1 for c in cvs.values() if c["skew_degrees"]),
        "outline_found": sum(1 for c in cvs.values() if c["page_quad"]),
        "capture_version": {v: sum(1 for c in cvs.values() if c["capture_version"] == v)
                            for v in sorted({c["capture_version"] for c in cvs.values()})},
        "glare_vs_reference": _agree(rows, man, cvs, "glare"),
        "sharpness": _spread([c["quality_text"]["sharpness"] for c in cvs.values()]),
        "brightness": _spread([c["quality_full"]["brightness"] for c in cvs.values()]),
        "contrast": _spread([c["quality_full"]["contrast"] for c in cvs.values()]),
        "glare_percent_max": max(c["quality_full"]["glare_percent"] for c in cvs.values()),
        "glare_blobs_by_label": {lab: sum(1 for i, c in cvs.items() if man[i]["lighting"] == lab and c["glare_blobs"])
                                 for lab in ("glare", "good", "dim")},
        "lighting_labels": {lab: sum(1 for i in cvs if man[i]["lighting"] == lab) for lab in ("glare", "good", "dim")},
        "blurry_tiles": [sum(len(c["blurry_patches"]) for c in cvs.values()), sum(c["text_tiles"] for c in cvs.values())],
        "slant_by_angle": {a: [sum(1 for i, c in cvs.items() if man[i]["angle"] == a and c["skew_degrees"]),
                               sum(1 for i in cvs if man[i]["angle"] == a)] for a in ("flat", "tilted", "curved")},
        "slant_abs": _spread([abs(c["skew_degrees"]) for c in cvs.values() if c["skew_degrees"]]),
        "would_flatten": sum(c["would_flatten"] for c in cvs.values()),
        "page_fill": _spread([c["page_fill"] * 100 for c in cvs.values()]),
        "fills_75": sum(1 for c in cvs.values() if c["page_fill"] >= 0.75),
        "hints": sum(1 for c in cvs.values() if c["quality_text"]["hint"]),
    }
    return summary


def _spread(values):
    v = sorted(values)
    return {"min": round(v[0], 2), "median": round(statistics.median(v), 2), "max": round(v[-1], 2)} if v else None


def _agree(rows, man, cvs, problem):
    """How the app's glare check agrees with the reference's lighting label (glare / not glare)."""
    tp = fp = fn = tn = 0
    for r in rows:
        truth = man[r["id"]]["lighting"] == problem
        said = problem in cvs[r["id"]]["quality_text"]["problems"]
        tp += truth and said; fp += said and not truth; fn += truth and not said; tn += not truth and not said
    return {"both": tp, "app_only": fp, "reference_only": fn, "neither": tn}


def pct(x):
    return "–" if x is None else f"{x * 100:.1f} %"


def mark(ok):
    return "✅" if ok else "❌" if ok is False else "–"


def md_escape(s):
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def report(pids):
    rows = _load_scores(pids)
    if not rows:
        print("Nothing scored yet: run the score stage first.")
        return
    man, refs, intake = R.manifest(), references(pids), R.intake()
    s = summarise(rows, man, refs)
    qa0 = R.read("qa", rows[0]["id"])
    s["qa_model"] = qa0["model"]
    s["git"] = {"commit": qa0["commit"], "dirty": qa0["dirty"]}
    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)

    head = _summary_md(s, len(rows))
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(head)

    out = [head.replace("# Photo evaluation: summary", "# Photo evaluation: every photo", 1),
           "\n---\n\n## Every photo\n",
           "Each photo: the picture with the image checks drawn as boxes (legend on its right), the three OCR "
           "engines' boxes side by side, Claude's reading, the OCR scores, and every question with the local "
           "model's answer. ✅ correct, ❌ wrong, – not scored (free text, or the field was unreadable).\n"]
    rel = os.path.relpath(os.path.join(R.OUT, "annotated"), os.path.dirname(REPORT_MD))
    for r in rows:
        pid, ref, cv = r["id"], refs[r["id"]], R.read("cv", r["id"])
        m = man[pid]
        out.append(f"\n### {pid}: {md_escape(ref['item'])}\n")
        out.append(f"`{intake[pid]['original_name']}` · {m['category']} · light: {m['lighting']} · angle: {m['angle']} · "
                   f"legibility (Claude): {ref['legibility']}" + (f" · {md_escape(m['notes'])}" if m["notes"] else "") + "\n")
        out.append(f"![{pid} image checks]({rel}/{pid}_checks.jpg)\n")
        out.append(f"![{pid} OCR engines]({rel}/{pid}_ocr.jpg)\n")
        q = cv["quality_text"]
        out.append("| Image check | Result |\n|---|---|")
        out.append(f"| Sharpness of the text (variance of Laplacian; blurry below {int(R._blur_min())}) | {q['sharpness']:.0f}"
                   f"{' **blurry**' if 'blurry' in q['problems'] else ''}; {len(cv['blurry_patches'])} of {cv['text_tiles']} text patches blurry (red) |")
        out.append(f"| Glare (saturated blobs, magenta) | {cv['quality_full']['glare_percent']:.1f} % of the picture, "
                   f"{len(cv['glare_blobs'])} blob(s){' **glare**' if 'glare' in q['problems'] else ''} |")
        out.append(f"| Light (mean brightness; dark below 50) | {cv['quality_full']['brightness']:.0f}; {len(cv['dark_patches'])} dark patch(es) (blue) |")
        out.append(f"| Contrast (flat below 25) | {cv['quality_full']['contrast']:.0f} |")
        out.append(f"| Text slant (orange arrow) | {cv['skew_degrees']:.1f}° |")
        out.append(f"| Package outline (yellow) | {'found' if cv['page_quad'] else 'not found'}"
                   f"{', the app would flatten it' if cv['would_flatten'] else ''}; fills {cv['page_fill'] * 100:.0f} % of the picture |")
        out.append(f"| What the app would say | {q['hint'] or 'nothing (ready to capture)'} |")
        out.append(f"| Version the capture kept | {cv['capture_version']} |\n")
        out.append("**Claude's reading of the photo**\n")
        out.append("| Field | As printed |\n|---|---|")
        for k in FIELDS:
            v = ref["fields"].get(k)
            out.append(f"| {k} | {'not on the photo' if v is None else 'unreadable' if v == '?' else md_escape(v)} |")
        out.append(f"| Rx mark | {ref['flags'].get('rx')} |\n| External use only | {ref['flags'].get('external_use')} |\n")
        out.append("Key lines: " + " · ".join(f"`{md_escape(l)}`" for l in ref["key_lines"]) + "\n")
        if ref.get("notes"):
            out.append(f"Notes: {md_escape(ref['notes'])}\n")
        out.append("| OCR engine | Lines | Time | Key-line accuracy | Fields found |\n|---|---|---|---|---|")
        for e in ENGINES:
            d = r["engines"][e]
            fields = ", ".join(f"{k} {mark(v)}" for k, v in d["fields"].items()) or "–"
            out.append(f"| {R.ENGINE_NAME[e]} | {d['lines']} | {d['seconds']:.2f} s | {pct(d['key_line_accuracy'])} | {fields} |")
        ocr = R.read("ocr", pid)
        out.append("\n<details><summary>Text the app kept (what the model was given)</summary>\n\n```\n"
                   + R.text_of(ocr, "app_capture") + "\n```\n</details>\n")
        for e in ("apple_vision", "rapidocr", "tesseract"):
            out.append(f"<details><summary>{R.ENGINE_NAME[e]} text</summary>\n\n```\n" + R.text_of(ocr, e) + "\n```\n</details>\n")
        out.append(f"\n**Questions, answered by the local model ({s['qa_model']})**\n")
        out.append("| # | Question | Answer | Time | Says not in text | Score |\n|---|---|---|---|---|---|")
        for i, a in enumerate(r["answers"], 1):
            ans = a["answer"] if a["key"] != "read_all" else "(the app reads the kept text above, word for word)"
            extra = f" ⚠️ numbers not in the text: {', '.join(a['numbers_not_in_text'])}" if a["numbers_not_in_text"] else ""
            out.append(f"| {i} | {a['question']} | {md_escape(ans)}{extra} | {a['seconds']:.2f} s | "
                       f"{'yes' if a.get('not_in_text') else 'no'} | {mark(a['correct'])} {md_escape(a['why']) if a['correct'] is not None else ''} |")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {os.path.relpath(SUMMARY_MD, R.ROOT)}, {os.path.relpath(REPORT_MD, R.ROOT)}, {os.path.relpath(SUMMARY_JSON, R.ROOT)}")


def _correct_cell(d):
    return f"{d['correct']} ({pct(d['correct'] / d['scored'])})" if d["scored"] else "–"


def _summary_md(s, n):
    e = s["engines"]
    L = ["# Photo evaluation: summary", "",
         f"{n} pharmacy photos (blister strips, cartons, bottles, tubes and personal-care items), taken with a phone, "
         f"run on {s['date']} at commit `{s['git']['commit']}`{' (with uncommitted changes)' if s['git']['dirty'] else ''}. "
         "Generated by `eval/run_photo_eval.py score report`; do not edit by hand.", "",
         "**Reference:** Claude (the assistant model) read every photo itself, before seeing any engine's text, and "
         "wrote down the key printed lines and fields. It is one careful reader, not a typed ground truth checked by "
         "a second person: where Claude misread a pack, an engine is marked wrong for it. Fields Claude could not read "
         "are left out of the scores. Numbers in brackets are 95 % bootstrap intervals over photos.", "",
         "## OCR accuracy", "",
         "| Engine | Key-line accuracy | Fields found | Mean time | Photos with no text |", "|---|---|---|---|---|"]
    for k in ENGINES:
        d = e[k]
        ci = d["key_line_accuracy_ci95"]
        fci = d["fields_found_ci95"]
        L.append(f"| {R.ENGINE_NAME[k]} | {pct(d['key_line_accuracy'])} ({pct(ci[0])}–{pct(ci[1])}) | "
                 f"{d['fields_found']}/{d['fields_total']} = {pct(d['fields_found_rate'])}"
                 f"{f' ({pct(fci[0])}–{pct(fci[1])})' if fci else ''} | {d['seconds_mean']:.2f} s | {d['photos_with_no_text']} |")
    L += ["", "Key-line accuracy = 1 − character edits / characters, each key line matched where it best fits in the "
          "engine's text (case and spacing ignored). A field counts as found when its letters and digits appear in the "
          "engine's text. *App capture* is the text the app actually keeps after a capture (Apple Vision on the best of "
          "the original, flattened, tone-corrected and binarised versions).", "",
          "### Fields found, per field", "", "| Field | " + " | ".join(R.ENGINE_NAME[k] for k in ENGINES) + " |",
          "|---|" + "---|" * len(ENGINES)]
    for f in FIELDS:
        cells = []
        for k in ENGINES:
            v = e[k]["per_field"].get(f)
            cells.append(f"{v[0]}/{v[1]}" if v else "–")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "### By category and condition (key-line accuracy)", "",
          "| Group | Photos | " + " | ".join(R.ENGINE_NAME[k] for k in ENGINES) + " |", "|---|---|" + "---|" * len(ENGINES)]
    for c, d in s["by_category"].items():
        L.append(f"| {c} | {d['photos']} | " + " | ".join(pct(d[k]) for k in ENGINES) + " |")
    a = s["answers_overall"]
    L += ["", f"## Answers ({s['qa_model']}, local, through the app's own question path)", "",
          f"Factual questions scored by fixed rules against Claude's reading: **{a['correct']}/{a['scored']} = "
          f"{pct(a['accuracy'])}** ({pct(a['ci95'][0])}–{pct(a['ci95'][1])}). The model was given the text the app "
          "kept (Apple Vision, app capture path), so OCR mistakes reach the answers too.", "",
          "| Question | Asked | Scored | Correct | Said \"not in the text\" | Answers with numbers not in the text | Mean time |",
          "|---|---|---|---|---|---|---|"]
    for k, d in s["answers"].items():
        L.append(f"| {d['question']} | {d['asked']} | {d['scored']} | "
                 f"{_correct_cell(d)} | "
                 f"{d['said_not_in_text']} | {d['with_numbers_not_in_text']} | {d['seconds_mean']:.2f} s |")
    c = s["checks"]
    sh, br, ct, sk, pf = c["sharpness"], c["brightness"], c["contrast"], c["slant_abs"], c["page_fill"]
    gl, lab = c["glare_blobs_by_label"], c["lighting_labels"]
    L += ["", "## Image checks (the app's own measures)", "",
          "| Check | App's rule | On these photos (min / median / max) | Flagged by the app |", "|---|---|---|---|",
          f"| Blur: sharpness of the text (variance of the Laplacian) | blurry below 60 | {sh['min']:.0f} / {sh['median']:.0f} / {sh['max']:.0f} | "
          f"0 photos; {c['blurry_tiles'][0]} of {c['blurry_tiles'][1]} text patches (40 px) below 60, on {c['photos_with_blurry_text_patches']} photos |",
          f"| Glare: saturated blobs (grey 250+) | glare at 1 % of the picture | largest {c['glare_percent_max']:.2f} % | "
          f"0 photos; blobs drawn on {c['photos_with_glare_blobs']} ({gl['glare']} of the {lab['glare']} photos Claude saw glare on, {gl['good']} of the {lab['good']} in good light) |",
          f"| Light: mean brightness | dark below 50, too bright above 205 | {br['min']:.0f} / {br['median']:.0f} / {br['max']:.0f} | 0 photos |",
          f"| Contrast: spread of grey | flat below 25 | {ct['min']:.0f} / {ct['median']:.0f} / {ct['max']:.0f} | 0 photos |",
          f"| Slant of the text | straightened from 1.5° (book path only) | {sk['min']:.1f}° / {sk['median']:.1f}° / {sk['max']:.1f}° where measured | "
          f"measured on {sum(v[0] for v in c['slant_by_angle'].values())} photos (flat {c['slant_by_angle']['flat'][0]}/{c['slant_by_angle']['flat'][1]}, "
          f"tilted {c['slant_by_angle']['tilted'][0]}/{c['slant_by_angle']['tilted'][1]}, curved {c['slant_by_angle']['curved'][0]}/{c['slant_by_angle']['curved'][1]}) |",
          f"| Item outline (grown from its text) and flattening | flatten when tilted over 2° or keystoned | – | outline on {c['outline_found']} photos, the app would flatten {c['would_flatten']} |",
          f"| How much of the picture the item fills | 75 % or more is close enough (no \"Move closer\") | {pf['min']:.0f} % / {pf['median']:.0f} % / {pf['max']:.0f} % | {c['fills_75']} photos at 75 %+ |",
          f"| Spoken hint the app would give | – | – | {c['hints']} photos |",
          "",
          f"- Problems the app's quality check flagged (on the text area): {', '.join(f'{k} {v}' for k, v in c['problems_flagged'].items()) or 'none'}.",
          f"- Photos with glare blobs drawn: {c['photos_with_glare_blobs']}; with blurry text patches: "
          f"{c['photos_with_blurry_text_patches']}; with dark patches: {c['photos_with_dark_patches']}; with a measured slant: {c['photos_with_slant']}.",
          f"- Item outline found: {c['outline_found']} of {n} (none where the item is not one solid four-sided shape, such as a shelf of several products or a crumpled strip).",
          f"- Version the capture path kept: {', '.join(f'{k} {v}' for k, v in c['capture_version'].items())}.",
          f"- Glare check against Claude's lighting label: both say glare {c['glare_vs_reference']['both']}, only the app "
          f"{c['glare_vs_reference']['app_only']}, only Claude {c['glare_vs_reference']['reference_only']}, neither {c['glare_vs_reference']['neither']}.",
          "", "**What the image checks show, and their limits** (seen on the annotated photos):", "",
          "- The quality check flagged no photo: it calls glare only for compact *saturated* white blobs covering at least "
          "1 % of the picture. Most glare on these packs is a reflection on foil or gloss that is bright but not saturated "
          "(or, on mirror-like cartons, a dark reflected image), so it is not caught. Magenta boxes show the saturated "
          "blobs that do exist.",
          "- Blue \"too dark\" patches are 40 px tiles darker than the app's dark threshold (mean below 50). On these photos "
          "they are mostly dark-coloured printing (navy labels, black bands) and shadows at the edges, not poor light: the "
          "whole-picture brightness never fell below the threshold.",
          "- The item outline (yellow) is grown from the item's own text with GrabCut (`find_item_quad()` in `vision.py`, "
          "corrected on 23 October 2026). The earlier outline took the brightest region as the page; on a light floor that "
          "is the floor, so 144 of its 145 outlines were the picture's frame and none enclosed a pack, and \"how much the "
          "item fills\" counted the floor too (median 69 % instead of about 22 %).",
          "- Red blurry patches are tiles on recognised text lines whose sharpness (variance of the Laplacian) is below the "
          "app's threshold of 60.",
          "",
          "**Scoring check:** two random samples of 40 scored answers were read by hand against the photos. The first found "
          "three rule mistakes (a date written \"December 31, 2026\", \"1x10 tablets\" against \"10 tablets\", O against 0 "
          "in batch codes), which were fixed before these numbers; the second found 1 of 40 wrong, caused by a "
          "prescription flag in the reference that the photo did not show (the prescription question is now scored only "
          "where Rx, Schedule H or a prescription line is visible).",
          "",
          "Every photo, with its boxes, OCR text and all answers: `markdown/eval_photo_report.md` (git-ignored, "
          "because it holds the full text of the photos).", ""]
    return "\n".join(L)
