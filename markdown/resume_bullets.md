# Resume bullets: Helping Eyes

Every number here comes from [eval_results.md](eval_results.md) (measured 28 October 2026). If a number there changes,
change it here too. Pick 3–4 bullets; the short versions fit one line.

**Helping Eyes — a camera reading assistant for visually impaired people** (Python, FastAPI, Apple Vision, OpenCV,
Gemini / Qwen 2.5 3B via Ollama, Kokoro TTS, JavaScript)

- Built a voice-first web app that guides the user to frame a label, reads it with on-device OCR (Apple Vision) and
  answers spoken questions from the captured text only, using Gemini Flash-Lite or a local Qwen 2.5 3B.
  *Short:* Voice-first reading assistant: on-device OCR + grounded LLM answers + local neural voice.

- Evaluated the pipeline on 164 real phone photos (medicine packs, food labels, books, documents, screens) against
  Tesseract and RapidOCR: 90.5 % key-line accuracy for the app's capture versus 88.4 % (RapidOCR) and 59.2 %
  (Tesseract), at 0.31 s per photo.
  *Short:* 90.5 % OCR key-line accuracy on 164 real photos, beating RapidOCR and Tesseract.

- Kept exact answers out of the LLM: expiry verdicts computed in code were right on 22 / 22 label formats with both
  models, against 5 / 22 for Qwen 3B reasoning about the dates alone.
  *Short:* Code-computed expiry checks: 22/22 correct vs 5/22 for the LLM alone.

- Compared Gemini Flash-Lite with a local Qwen 2.5 3B on the same photos with paired bootstrap confidence intervals:
  87.2 % vs 64.5 % factual accuracy (+16 points paired, 95 % CI 9–25), both reporting missing details correctly over
  92 % of the time, with numbers not in the text in under 2 % of answers.
  *Short:* Paired LLM comparison: Gemini 87 % vs Qwen 3B 65 % factual accuracy, under 2 % unsupported numbers.

- Built book mode with classical computer vision: page-turn detection by frame differencing, spine splitting,
  page-number and running-title detection, XY-cut reading order (98 % of line pairs in order on real pages) and a
  word-by-word reading highlight.
  *Short:* OpenCV book mode: page turns, reading order (98 % on real pages), word highlight.

- Made the evaluation reproducible: resumable scripts, a fixed evaluation date, retry rules for a rate-limited API,
  blind A/B judging of free-text answers, tests for the scoring rules, and results generated straight into the report.
  *Short:* Reproducible evaluation harness with tested scoring rules and generated reports.

Be ready to say what did not work, too (also in eval_results.md): the OCR error rate (0.095) misses the 0.05 target,
neither model reaches 90 % on factual questions, the spine was not found on real two-page spreads, and the live check
runs about 5 times a second, not 10.
