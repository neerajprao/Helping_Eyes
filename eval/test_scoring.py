"""
Tests for the evaluation's scoring rules (photo_report.py, run_expiry.py, run_qa.py, run_book.py, common.py). Every
case here is a mistake a rule once made, or one it must never make. No photos, models or network needed:

    .venv/bin/python eval/test_scoring.py
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402
import photo_report as P  # noqa: E402


def ans(text, not_in_text=False):
    return {"answer": text, "not_in_text": not_in_text}


def test_dates_in_the_forms_labels_and_answers_use():
    assert P.month_years("EXP. NOV.26") == {(11, 26)}
    assert P.month_years("Exp. 12.2026") == {(12, 26)}
    assert P.month_years("Mfd. 01 2024") == {(1, 24)}
    assert (12, 26) in P.month_years("It expires on December 31, 2026.")
    assert (6, 29) in P.month_years("It expires on June 30, 2029.")
    assert (12, 27) in P.month_years("valid until 31/12/2027")


def test_a_label_month_is_valid_to_its_end():
    assert P.expired_on("10/2026", datetime.date(2026, 10, 19)) is False
    assert P.expired_on("09/2026", datetime.date(2026, 10, 19)) is True


def test_expiry_verdicts_in_answers():
    assert P.says_expired("No, it has not expired; it is good until April 2028.") is False
    assert P.says_expired("Yes, this medication has expired.") is True
    assert P.says_expired("It is still valid until November 2026.") is False
    assert P.says_expired("It expired on 18 October 2026.") is True


def test_cannot_tell_is_not_a_verdict():
    assert P.says_expired("The text does not show an expiry date, so I cannot tell if it has expired.") is None
    assert P.says_expired("I can't say whether it has expired.") is None


def test_not_in_text_by_tag_or_by_wording():
    assert P.says_not_in_text(ans("Anything", not_in_text=True))
    assert P.says_not_in_text(ans("The text does not mention the price."))
    assert not P.says_not_in_text(ans("The price is Rs. 32.50."))


def test_batch_codes_read_o_as_zero():
    ref = {"fields": {"batch": "BO24I1"}, "flags": {}}
    assert P.score_answer("batch", ans("The batch number is B02411."), ref, "")[0]


def test_one_by_ten_tablets_is_ten_tablets():
    ref = {"fields": {"quantity": "1x10 Tablets"}, "flags": {}}
    assert P.score_answer("quantity", ans("The pack has 10 tablets."), ref, "")[0]


def test_a_field_the_photo_does_not_show_must_be_said_so():
    ref = {"fields": {"price": None}, "flags": {}}
    assert P.score_answer("price", ans("The text doesn't say the price."), ref, "")[0]
    assert P.score_answer("price", ans("It costs Rs. 40."), ref, "")[0] is False


def test_either_of_two_names_counts():
    ref = {"fields": {"maker": "Bayer AG / Bayer Zydus Pharma"}, "flags": {}}
    assert P.score_answer("maker", ans("It is made by Bayer Zydus Pharma."), ref, "")[0]


def test_per_photo_questions():
    assert P.score_custom({"answer": "14/10/2026", "type": "date"}, ans("Best before 14 October 2026."))[0]
    assert P.score_custom({"answer": "Rs. 20 / ₹20", "type": "number"}, ans("It costs 20 rupees."))[0]
    assert P.score_custom({"answer": None, "type": "text"}, ans("The text doesn't say."))[0]
    assert P.score_custom({"answer": "no", "type": "yesno"}, ans("No, it does not contain nuts."))[0]
    assert P.score_custom({"answer": "no", "type": "yesno"}, ans("No, the text doesn't say whether it has nuts."))[0] is False
    assert P.score_custom({"answer": "no", "type": "yesno"}, ans("No.", not_in_text=True))[0] is False
    assert P.score_custom({"answer": "x", "type": "free"}, ans("anything"))[0] is None


def test_made_up_numbers_are_caught_but_the_apps_date_note_is_not():
    C.fix_date()
    assert P.unsupported_numbers("It costs Rs. 45.", "MRP Rs. 32.50") == ["45"]
    assert P.unsupported_numbers("It is valid until 30 November 2026.", "EXP NOV.26") == []


def test_expiry_rescoring_checks_the_stated_date():
    import run_expiry as E
    found = "EXP 10/26 -> valid until 2026-10-31"
    right = E.score({"answer": "It has not expired; it is valid until 31 October 2026.", "service_error": False}, False, False, found)
    wrong = E.score({"answer": "It has not expired yet; it is valid until November 3rd, 2026.", "service_error": False}, False, False, found)
    assert right["correct"] and right["date_ok"] is True
    assert wrong["correct"] and wrong["date_ok"] is False


def test_relative_best_before_needs_no_wrong_verdict():
    import run_expiry as E
    r = {"answer": "The text does not show an expiry date, so it has not expired.", "not_in_text": False, "service_error": False}
    assert E.score(r, True, True, None)["correct"] is False          # truly expired: "has not expired" is wrong
    r = {"answer": "The text does not show an expiry date.", "not_in_text": True, "service_error": False}
    assert E.score(r, True, True, None)["correct"] is True


def test_value_in_text_matches_dates_by_month():
    import run_qa as Q
    assert Q.in_text("NOV.2026", "MFG DEC.24 EXP 11/26") is True
    assert Q.in_text("B02411", "Batch B0 2411") is True
    assert Q.in_text("Rs. 40", "MRP Rs. 32.50") is False


def test_reading_order_counts_pairs_in_order():
    import run_book as B
    lines = ["The first line of the page", "a second line follows here", "and the third line ends it"]
    good = B.order_score(lines, " ".join(lines))
    bad = B.order_score(lines, " ".join(reversed(lines)))
    assert good["order"] == 1.0 and good["starts_at_first_line"]
    assert bad["order"] == 0.0 and not bad["starts_at_first_line"]


def test_bootstrap_is_reproducible_and_brackets_the_rate():
    groups = [[True, False], [True, True], [False], [True, True, True]]
    a, b = C.bootstrap(groups, C.rate), C.bootstrap(groups, C.rate)
    assert a == b and a[0] <= C.rate(groups) <= a[1]


def test_timing_percentiles():
    t = C.timing([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert t["p50"] == 5.5 and t["min"] == 1 and t["max"] == 10 and t["n"] == 10


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for test_name, fn in tests:
        try:
            fn()
            print(f"PASS  {test_name}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"FAIL  {test_name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
