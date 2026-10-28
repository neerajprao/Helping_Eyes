# User study: how to run it

This covers the **User rating** metric (5+ users, average at least 4.0 out of 5) and **NFR-8** (a first-time user,
following only the spoken prompts, captures an item and hears its expiry date in under 2 minutes without looking at
the screen) from the requirements document. About 15 minutes per person.

Results go in `eval/data/user_study.csv`, one row per person. `eval/make_results.py` reads it and adds a "User study"
section to `markdown/eval_results.md`. Write down **no names**: use P1, P2, ... only.

## Who

- At least 5 people who have not used the app before.
- Include visually impaired people if you can. Otherwise, sighted people **blindfolded or with eyes closed** for task 1,
  and say so in the `vision` column (`blind`, `low_vision`, `sighted_blindfolded`, `sighted`).

## Before each person

1. `python run.py` on the Mac (or `python run.py --share` and the phone), Kokoro voice on, **Answers from:
   Online AI (Gemini)** (or Qwen: write which one in `model`).
2. Have three items ready: a medicine strip with a printed expiry date, a food packet, and a book.
3. Read this to them: *"This app reads printed text aloud and answers questions about it. I will time some tasks. You
   can stop at any time. Nothing is recorded except the times and your ratings, with no name."* Go on only if they
   agree.

## Tasks

| # | Task (say it to them) | What you write down |
|---|---|---|
| 1 | Eyes closed or blindfolded. *"Here is a medicine strip. Use the app to find out when it expires. Press Listen (M key) to talk to it."* Start the timer when they first hold up the strip. | `t1_seconds` until they hear the expiry date; `t1_success` = 1 if they heard the right date within 2 minutes, else 0. No help except repeating the instruction once. |
| 2 | *"Ask whether it has expired, then ask for the batch number."* | `t2_success` = 1 if both answers were right |
| 3 | *"Now show the food packet and ask for the ingredients."* (Press **New item** or say "next" first.) | `t3_success` |
| 4 | *"Switch to book mode and let it read a page; then turn the page."* | `t4_success` = 1 if it read the next page after the turn |
| 5 | *"Ask something that is not printed on the packet, for example what it is used for, and answer the question it asks you."* | `t5_success` = 1 if it said the text doesn't say it and offered or did a web lookup |

## Questions afterwards (1 = strongly disagree, 5 = strongly agree)

| Column | Statement |
|---|---|
| `q_useful` | The app would be useful to someone who cannot read print. |
| `q_easy` | It was easy to get the item into view by following the spoken hints. |
| `q_answers` | The answers were correct and to the point. |
| `q_speed` | The answers came quickly enough. |
| `q_voice` | The voice was clear and pleasant. |
| `q_overall` | Overall, I would use or recommend this app. **(This is the "user rating" in the requirements.)** |

`comments`: anything they said, in a few words, without names. `date`: YYYY-MM-DD.

## The file

`eval/data/user_study.csv` already has the header. Add one line per person, for example:

```
P1,sighted_blindfolded,gemini,48,1,1,1,1,1,5,4,4,5,4,5,hints were clear; wanted a slower voice,2026-10-30
```

Then run `.venv/bin/python eval/make_results.py` again.
