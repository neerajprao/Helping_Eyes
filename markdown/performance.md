# Performance and limitations

What was measured on real photos and the real server, and what the app cannot do (yet). The full tables are in [eval/eval_results.md](eval/eval_results.md). Back to the [README](../README.md).

## Performance

Everything below was measured on 28–29 October 2026 with the scripts in `eval/` (see [Evaluation](setup.md#evaluation)) on an Apple M3 Pro MacBook (18 GB), answers checked against a fixed date (19 October 2026). The full tables, with 95 % confidence intervals and every target of the course requirements marked met or not met, are in **[markdown/eval/eval_results.md](eval/eval_results.md)**; this section is a summary of it.

**How the answers were checked.** The reference for every photo is Claude's own reading of it (the key printed lines, the details such as name, dates, batch and price, and for the non-medicine photos the questions with their expected answers), written before any reader's output was seen. It is one careful reader, not a typed ground truth checked by a second person. Factual answers are scored by fixed rules (tested in `eval/test_scoring.py`); free-text answers by a blind A/B judgement of a random sample.

### Reading the text: 164 real photos

148 pharmacy photos (79 cartons, bottles and ampoules, 43 tablet strips, 26 personal-care items) and 16 others (6 food labels, 5 book pages, 3 printed documents, 2 laptop screens), taken with a phone in good light, glare and dim light, flat, tilted and curved.

| Text reader | Key-line accuracy | Details found (name, strength, batch, dates, price ...) | Time per photo |
|---|---|---|---|
| **The app's capture** (Apple Vision on the best of the original and corrected versions) | **90.5 %** | 70.8 % | 0.31 s |
| Apple Vision alone | 90.2 % | 70.3 % | 0.18 s |
| RapidOCR (the earlier reader) | 88.4 % | 69.3 % | 0.69 s |
| Tesseract | 59.2 % | 25.6 % | 0.36 s |

Printed documents, screens and book pages are read almost perfectly (96–100 %), cartons at 97 %, food labels at 86 % and crumpled foil strips at about 81 %; curved items (80 %) and the two dim photos (76 %) are the hardest. The requirement's character error rate of at most 0.05 is not met (0.095).

### Answers: Gemini Flash-Lite against Qwen 2.5 3B

The same questions about the same 46 photos (30 pharmacy photos chosen at random within each lighting and angle group, plus all 16 others) were asked to both models through the app's own code (708 questions each; Gemini's free tier was busy, so its run took two days and 54 retries, with no question left unanswered).

| | Gemini Flash-Lite | Qwen 2.5 3B (on this Mac) |
|---|---|---|
| Factual answers right (name, strength, quantity, dates, batch, price, maker) | **80.0 %** (95 % interval 75–85) | 64.5 % (57–71) |
| … when the value was in the text the model got | 82.2 % | 57.4 % |
| Says "the text doesn't say" when the photo really doesn't (target ≥ 90 %) | 93.9 % | 93.9 % |
| Says "the text doesn't say" although it does | 12.4 % | 26.0 % |
| Answers with a number that is not in the text | 1.7 % | 0.6 % |
| Free-text answers (use, storage, warnings ...), blind sample of 100 from the first 20 photos | 98 % right, 2 % unsupported | 81 % right, 12 % partial, 7 % wrong |
| First sentence of the answer, median / 90th percentile | 3.1 s / 17.7 s (busy free tier) | 0.45 s / 1.1 s |

On the same 355 factual questions Gemini is right 15 points more often (95 % interval 11 to 21 points; Gemini alone right on 65, Qwen alone on 10). Qwen's main weakness is saying a detail is missing when it is printed, often because the value sits a row away from its label. Neither model reached the requirement's 90 % on factual questions; many of the remaining misses come from the reading itself (a value the OCR missed or garbled can only be right if the model repairs it: Gemini 28 %, Qwen 7 %).

### Expiry dates

22 label texts in the date formats found on the real packs (`NOV.26`, `DEC.2026`, `12/2026`, `10/26`, `JUL 28`, `Best before 14/10/2026` ...), half of them expired, including dates one day either side of the evaluation date:

| Who decides "has it expired?" | Right |
|---|---|
| The app's code (`expiry_dates`) | 22 / 22 |
| The app with Gemini or with Qwen (the code's verdict is given to the model) | 22 / 22 each |
| Qwen alone, without the app's date check, neutral prompt | **1 / 22** |
| Qwen alone, without the app's date check, the app's prompt kept | 5 / 22 |
| Gemini alone, without the app's date check (neutral prompt / app's prompt) | 22 / 22 and 21 / 22 |

**Real packs whose expiry date is printed apart from its label.** On many Indian packs the labels (Batch No., Mfg. Date, Expiry Date) are printed in one column and their dates in another, so the text reader puts a date away from its label (`03/2025 Mfg. Date: 08/2027 Expiry Date: 230.00`). Of the 81 evaluation photos with a printed expiry date, the label-only finder caught 22, and both models then answered "the text doesn't show an expiry date" for the rest. Now every printed date is checked in code (`printed_dates`); Gemini picks the expiry date from the text and Qwen is given the latest date (an expiry date comes after the manufacturing date), and an answer whose date has passed adds that it may be the manufacturing date. On the 34 photos where this applies (8 of them with the real date not read at all), measured on 7 November 2026 against the evaluation date: right expiry date Qwen 21, Gemini 22 (before: 0); right expired / not expired verdict 26 each (before: 0).

The app's prompt tells the model never to say "expired" unless its date check says so, which leans a model left alone towards "not expired", so the test was run again with a neutral expiry rule. Qwen did no better: every one of its 22 answers began "The text doesn't say …" although each label prints its expiry date, and only one went on to a verdict. Its right answers with the app's prompt partly came with wrong reasons (a manufacturing date taken for the expiry). This is why the app works dates out in code. The one slip with the code in place: for "best before 6 months from manufacture", where the code finds no date, Qwen answered "the text does not show an expiry date, so it has not expired" for a jar that had expired. Two expired items photographed for real (a 2024 spice packet and a 2025 ice cream) are part of the photo set.

### Web lookup safety

All 8 web answers started with "According to the web", all 3 health answers ended with the pharmacist reminder, and expiry, batch and price questions are never offered a lookup. But read by hand, 4 of the 8 answers were not about the user's item: for a made-up maker name, the model described other companies as if they made it, and once claimed a medicine "contains meat or fish". "When was it manufactured?" is not blocked from a lookup either.

### Speed (the real server over localhost, 20 runs each)

| Operation | Median | 90th percentile |
|---|---|---|
| Capture of a label (1920 px photo → text) | 0.29 s | 0.47 s |
| Live frame round trip (960 px) | 0.11 s | 0.14 s |
| Question → start of the spoken answer, Qwen (first sentence + its Kokoro audio) | 1.37 s | 1.89 s |
| Question → start of the spoken answer, Gemini (free tier under heavy load) | 9.8 s | 103 s |
| Kokoro voice for one sentence | 1.30 s | 1.33 s |
| Book page with Qwen grouping the paragraphs | 5.8 s | 6.1 s |
| Book page with Gemini grouping the paragraphs | 3.8 s | 12.5 s |
| "Read everything", page-number answers | immediate (no model) | |

The page waits 100 ms after each live reply, so the live check runs about 4.8 times a second (the server alone could take 9.3), below the requirement's 10. Gemini was measured while Google's free tier kept answering "high demand" (HTTP 503): a reply came back in about 2 s or stalled for up to two minutes, so its times say more about that day's load than about the app. Only Qwen meets the 2-second target.

### Book mode on a real book

On 5 photos of a real book (4 two-page spreads) the reading order was right for 98 % of line pairs, but **the spine was not found on any spread**, so each spread was read as one wide page and no printed page number was announced (0 / 5; on synthetic spreads 6 / 6). Qwen did not group the paragraphs of these dense pages within the app's 12-second limit, so they were read in layout order. A 30-second video of 4 page turns, fed to the page-turn detector at the browser's frame rate, gave every new page once plus one early read while the hand was still turning the page.

*Since that evaluation* the book's border was added (see [section 2](book_mode.md#2-two-page-spread-splitting-spine-detection)). On the same 5 photos, read by the app's `read_page()` on 3 November 2026 (not yet re-run with `eval/run_book.py`), the border was found round the open book on all 5 (Apple's detector on 3, the text-grown outline on 2) and every spread was split at its fold, bp01 included, which the reference counts as one page because its left page is an index. The printed page numbers were still not announced on any of them.

### Image checks on the real photos

| Image check (`vision.py`, part B) | On the 164 photos |
|---|---|
| Blur (variance of the Laplacian, below 60) | Never flagged: text sharpness 248 to 7934; 22 of 10087 small text patches were blurry |
| Glare (saturated blobs over 1 % of the picture) | Never flagged; the largest saturated blob was 0.12 %: glare on foil, gloss and screens is bright but not pure white |
| Light and contrast | Never flagged: brightness 66 to 167, contrast 35 to 76 |
| Slant | Measured on 65 photos (median 3°, up to 26°); only the book path straightens it |
| Item outline | Found on 150 photos (none on shelves of several products or crumpled strips) |
| How much of the picture the item fills | Median 23 % (3 % to 79 %) |
| Version a capture kept | The untouched photo on 143, the flattened item on 16 (book pages: the book path on 5) |

The user study of the requirements (5 or more people, a rating of at least 4 out of 5) has not been run yet; its results go in `eval/data/user_study.csv` and are added to `eval_results.md` when it has been run.

---

## Known Limitations

- **Browser:** speech recognition needs Chrome or Edge; other browsers work with typed requests. The camera needs `http://localhost` or HTTPS.
- **Hosting depends on your Mac:** the shared address works only while the Mac is awake and `run.py --share` is running, and every visitor's frames are read one at a time on that Mac (one OCR job at a time), so several visitors at once slow the live guidance. The access code is one shared code: to let one person out, change it for everyone. Answers depend on the language model you configure.
- **Free API limits and privacy:** free model tiers are rate limited (requests per minute and per day vary by model and change over time); when the limit is reached the app says the model is busy. Free tiers may allow the provider to use submitted text to improve its products, so check the provider's terms before sending sensitive documents, or use a paid or self-hosted endpoint. "Read everything" and page-number answers never call the model; expiry answers do (the date is checked in code, and the model only words the verdict). *Offline AI on this computer (Qwen)* needs no key and has no such limit.
- **Internet access is required:** spoken requests are processed by the browser's speech service, and answers come from a hosted language model unless Qwen on this Mac is chosen. The voice needs no internet when Kokoro is set up; without it the voice comes from Microsoft's free Edge voice service.
- **Mac only:** text recognition uses Apple Vision, which exists only on macOS, so the server cannot run on Linux or Windows.
- **Local voice:** Kokoro needs a one-time download of about 350 MB and some CPU while it speaks (about 5 times faster than real time on an M3 Pro, so a long piece takes a few seconds before its first words). It is strongest in English.
- **Unofficial backup voice:** the Edge voice is a free service without a published guarantee; if Microsoft changes it, the page falls back to the browser's own voice. In October 2026 Microsoft refused `edge-tts` 7.2.8 (HTTP 403); 7.3.1 works.
- **Phones:** a phone allows sound only after a real tap, so the first sentence may wait for one (the page says "Tap the screen once to turn the voice on"). Hands-free listening works in Chrome on Android; iPhone Safari only starts listening from a tap, so there use the typed question or the buttons. A hidden tab or a locked screen can pause the audio.
- **Live checks:** about 5 a second as the page sends frames (it waits 100 ms after each reply), below the 10 a second the requirements ask for.
- **No voice interruption:** the microphone is inactive while the application speaks. Use `S` to stop a long answer.
- **Background text:** any text in view (keyboard keys, a screen) may be included in the capture. Holding the item close to the camera reduces this.
- **New captures:** a new item is captured only after the previous text leaves the view briefly, or after **New item** or "next".
- **Recognition errors:** small, curved, reflective or blurred print may be misread, and answers can only be as accurate as the captured text. On real pharmacy photos, crumpled foil strips and the purple inkjet batch and date lines printed over blister bumps were the hardest (about 80 % key-line accuracy against 97 % on cartons).
- **Glare coaching misses shiny packs:** the glare check looks for pure-white patches, so the reflections on foil strips, gold cartons, glossy bottles and screens (bright, but not pure white) produce no "Glare" hint; in 164 real photos it never fired.
- **Labels laid out in two columns:** when a batch number, date or price is printed a row above or below its label, or coded with symbols (`#` made, `@` use before), the language model can pair the wrong value with a label or not find it.
- **Web lookups:** only the search query (the question and the product name) leaves the device; the captured text is not sent. Answers depend on the quality of search results: in the evaluation, when the results were about a different company, the model reported them as facts about the user's item instead of saying the results did not answer. Lookups are never offered for expiry, batch or price ("When was it manufactured?" still is).
- **Relative expiry wording:** "best before 24 months from manufacture" (printed on several real packs) has no date for the code to check, so the model answers alone; in the evaluation Qwen once said "the text does not show an expiry date, so it has not expired" for an item that had expired. Gemini said the date was not printed.
- **A stalled hosted answer:** when Gemini's free tier is overloaded, a streamed reply can start and then stop sending without closing. The app's 120-second timeout only counts silence between pieces, so the user may wait up to two minutes or more with no answer (the evaluation stops its own calls after 90 s). Qwen on this Mac does not have this problem.
- **Book mode:** the spine is the shadow or blank strip inside the book's border, or else the border's centre, so a book held with its spine away from the middle of its own outline (for example one cover folded back) is split in the wrong place, and when no border is found a spread with no visible spine is still read as one wide page. The border needs at least 3 readable lines and costs a whole-frame read (about 0.3 s). On the real book photos the printed page numbers were not announced (0 of 5). A page-turn read can start while the hand is still turning the page. Qwen 3B is too slow to group the paragraphs of a dense page within 12 seconds, so such pages are read in layout order. Strong page curvature, uneven lighting, fingers over the text and illustrations near the centre of a spread can affect spine and column detection. Motion blur prevents recognition, so the book should be held steady or placed on a surface. When the view moves while a page is being read, a few words may be repeated, because the reading position is sent to the server before the new view has been recognised.
- **Model accuracy:** although the model is restricted to the captured text, a small model can still misread or confuse details. Medically important information should be confirmed with a pharmacist or doctor.
- **One reader:** the earlier standalone Gemini reader (which read all text aloud with no questions) was removed; "read everything" does the same without a cloud API.
