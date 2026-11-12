# Setting up and running Helping Eyes

What you need, how to install, run, share, test and evaluate the app, and what to do when something goes wrong. Back to the [README](../README.md).

## Hardware & Software Requirements

### Hardware

| Component | Specification |
|---|---|
| Server | A Mac (macOS) with Python 3.11 or later, because Apple Vision reads the text; no GPU and no extra hardware |
| Client | Any phone or computer with a modern browser, on the Mac itself or through `run.py --share` |
| Camera | Built-in or USB camera; a phone's rear camera when the page is served over HTTPS |
| Microphone and speakers | For spoken requests and answers |
| Browser | Chrome or Edge (speech recognition); Safari and Firefox work with typed requests |

### Software

| Component | Purpose |
|---|---|
| `pyobjc-framework-Vision`, `pyobjc-framework-Quartz` | Apple Vision, the text recognition built into macOS: text detection and recognition on the Mac's neural hardware, with per-word boxes. No model download and no API key; the app therefore runs on a Mac only |
| A language model API | Question answering. Any OpenAI-compatible chat API; Gemini Flash-Lite (free key) by default, or Qwen 2.5 3B on this Mac through Ollama (no key) |
| FastAPI and uvicorn | The server and its WebSocket |
| `kokoro-onnx`, `lameenc` | The local Kokoro voice (Apache-2.0 model, about 350 MB of model files from `python run.py --get-voice`) and its MP3 encoder. Optional: without them the Edge voice is used. Needs Python below 3.14 (the main `kokoro` package does not support 3.13) |
| `edge-tts` | Free Microsoft Edge neural voice, the backup for speech out; no account or API key required, but it needs an internet connection |
| `ddgs` (Yahoo, Startpage, DuckDuckGo) and Wikipedia's search API | Web search, only with the user's consent; no account or API key required |
| OpenCV and NumPy | Image analysis, book-mode page analysis |
| Browser speech service | Speech recognition (and the last-resort voice if both Kokoro and Edge fail) |

---

## Environment Setup

Only the language model's API key is required. Put it in `helping_eyes/.env`, the one settings file (git-ignored, so the key never reaches git; every optional setting is listed in it as a comment):

```env
# Free key from https://aistudio.google.com/apikey
LLM_API_KEY=<YOUR_KEY>

# Optional: defaults shown. Any OpenAI-compatible chat API works.
# LLM_MODEL=gemini-3.1-flash-lite
# LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
# Tried in turn when LLM_MODEL is busy or slow (empty: none)
# LLM_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-flash-lite-latest
# Web search engines, in order (the ddgs library's names); Wikipedia is asked when none finds anything
# SEARCH_BACKENDS=yahoo,startpage,duckduckgo,auto
```

The **Answers from** menu on the page chooses who answers: *Online AI (Gemini)*, the settings above, or *Offline AI on this computer (Qwen)*, Qwen 2.5 3B running on this Mac through [Ollama](https://ollama.com) (`python run.py` starts Ollama for you if it is installed and not already running (stopping it at the end only if it started it), and loads the Qwen model into memory in the background so the first answer is not slow (it tells you if the model still has to be downloaded with `ollama pull`); `--no-ollama` skips this; no key needed). Qwen settings: `QWEN_3B_MODEL` (default `qwen2.5:3b-instruct`, as listed by `ollama list`) and `QWEN_BASE_URL` (default `http://localhost:11434/v1`). The choice is sent with every request and remembered by the browser. If the model answers "busy" (HTTP 429 or 5xx, as Google's free tier often does) or has not started answering within 8 s, the next model of `LLM_FALLBACK_MODELS` is asked, and the busy one is tried last for 5 minutes (on 12 November 2026 `gemini-3.1-flash-lite` answered 503 "high demand" to nearly every request while `gemini-3.5-flash-lite` answered). To use another provider, change `LLM_BASE_URL`, `LLM_MODEL`, `LLM_FALLBACK_MODELS` and `LLM_API_KEY` (for example Groq or OpenRouter, both of which have free tiers); no code changes are needed. To keep the public address private, set `ACCESS_CODE` (people enter it once per browser; see [Hosting](#hosting)). Optional image settings: `VISION_ENHANCE=0` (turn off the image-correction layer) and `BLUR_MIN`. The camera is chosen in the browser. Optional voice settings: `TTS_VOICE` (default `en-US-JennyNeural`, female), `TTS_RATE` (speed, e.g. `+10%`) and `TTS_ENABLED=0` (always use the browser voice).

---

## Installation

```bash
# 1. Virtual environment (Python 3.11 or later)
python3 -m venv .venv
source .venv/bin/activate

# 2. Python dependencies (Apple Vision is part of macOS; nothing else to download)
python -m pip install -r requirements.txt

# 3. Your language model key
echo "LLM_API_KEY=<YOUR_KEY>" >> helping_eyes/.env     # your key; the only required setting

# 4. Run it
python run.py
```

On first use the browser asks for **camera** and **microphone** access; allow both.

---

## Running the System

### On a laptop

```bash
python run.py                  # checks your setup, starts the app and opens it in Chrome
```

`run.py` is the start button for the app on this Mac: it checks the Python version and the installed packages, tells you if the language-model key is missing, starts the same server, waits until the text reader has loaded and opens the page. Chrome (or Edge) does the camera, the microphone, speech to text and text to speech, exactly as for a visitor on the host. Options: `--no-browser` (print the address only), `--port 8000` (default 7860; the next free port is used if it is taken) and `--host 0.0.0.0` (also reachable from other devices on your network) and `--share` (see below). `python run.py --get-voice` downloads the local voice once (about 350 MB) and stops; the start-up message has a `voice` line that says which voice will speak. Stop it with Ctrl+C. To start the server without `run.py`: `cd helping_eyes && uvicorn server:app --port 7860`.

1. Allow the camera and the microphone: the browser asks for both as soon as the page opens, in one question (the microphone only needs the permission; listening stays off until you press **Listen**). Hold the item in front of the camera and follow the spoken guidance; it is captured automatically once it is steady and in view.
2. After *"Got it"*, ask aloud (press **Listen** once for hands-free listening; the words appear as "You said: ..." while you speak) or type the question and press Enter.
3. When you press **Capture now** (or the round shutter button, or `C`) the photo taken stays on the screen, with the detected text outlined, and the camera is switched off, until you press **New item** (`N`) or say *"next"*. (If no text is found in the photo, the camera comes straight back so you can try again. In book mode, the photo stays on the screen, with the reading overlay and highlight, and the camera is off until the page has been read to the end (then the camera comes back, ready for the next page) or **New item** is pressed; Stop leaves it there so Continue can read on, and if no text is found the camera comes straight back; turned pages are still read hands-free without pressing anything.) To capture another item, press **New item**, say *"next"*, or move the current one out of view briefly when the camera is on.
4. For a book, press **Book mode** (or say *"book mode"*), hold the book open and turn the pages.

The camera works on `http://localhost` and on any `https://` address, but not on a plain `http://` address on the network.

#### Share from your laptop (free, no account, no card)

```bash
brew install cloudflared       # once
python run.py --share          # prints a public https address
```

`--share` opens a free tunnel to the server on your laptop (your fixed ngrok address when `NGROK_DOMAIN` is set, see [Hosting](#hosting); otherwise a Cloudflare Quick Tunnel, as here), so a phone or a friend can open the app over `https` (the camera and microphone need it). On a Mac it also runs `caffeinate` so the laptop does not fall asleep from being idle. The address works only while the laptop is awake and `run.py` is running, and it changes every time. The laptop's own memory covers the roughly 1 GB the OCR needs. Closing the lid on battery still sleeps the Mac. Anyone with the address uses your language-model key's quota, so share it with people you trust.

### Hosting

The Mac is the server (Apple Vision exists only on macOS). `python run.py --share`, described above, gives it a public `https` address with a free tunnel: a fixed ngrok address when `NGROK_DOMAIN` is set in `helping_eyes/.env` (a free ngrok account and its free fixed domain), otherwise a Cloudflare tunnel whose address changes each run. With `ACCESS_CODE` set, visitors first see a page that asks for the code; nothing else works until it is entered. After starting, open `/api/health`: `llm_configured` must be `true` and `ocr` must be `ready`.

### Tests

```bash
cd helping_eyes
for t in tests/test_*.py; do python "$t" || break; done
cd .. && python eval/test_scoring.py     # the evaluation's scoring rules
```

### Evaluation

The real-world evaluation lives in `eval/` and writes its results to [markdown/eval/eval_results.md](eval/eval_results.md). Photos go in `eval/data/private/` (never committed; see [markdown/eval/eval_shotlist.md](eval/eval_shotlist.md)). From the top folder:

```bash
.venv/bin/python eval/run_photo_eval.py ocr cv qa score report   # OCR engines, image checks, Qwen on every photo
.venv/bin/python eval/run_qa.py subset && .venv/bin/python eval/run_qa.py run && .venv/bin/python eval/run_qa.py score
.venv/bin/python eval/run_qa.py web                               # web lookup safety
.venv/bin/python eval/run_expiry.py                               # expiry verdicts and the model-alone ablation
.venv/bin/python eval/run_book.py pages                           # book photos; and: run_book.py video <file> --turns N
.venv/bin/python eval/run_latency.py                              # speed (run with nothing else busy)
.venv/bin/python eval/make_results.py                             # writes markdown/eval/eval_results.md
```

Every script resumes where it stopped, and the Gemini runs retry a busy service (a reply not finished within 90 s counts as busy too) and stop by themselves when the free daily limit is used up. Keep the Mac on its charger with the lid open during long runs: a sleeping Mac freezes them.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| The address asks for an access code | The server has `ACCESS_CODE` set: ask the person who runs it for the code. The browser remembers it for 30 days; after 5 wrong codes, wait a minute. To change the code, edit `ACCESS_CODE` in `helping_eyes/.env` and restart `run.py`. |
| "No API key is set" or "rejected the request" | Put your key in `LLM_API_KEY` (`helping_eyes/.env`) and restart. `/api/health` shows `llm_configured`, and also the OCR status, the memory in use and which voice speaks first (`tts_engine`). |
| "The language model name was not found" (HTTP 404 in the log) | The key works but `LLM_MODEL` (or `LLM_BASE_URL`) is wrong. Google retires and restricts old model names (`gemini-2.5-flash-lite` stopped working for new keys). Look up the current Flash-Lite name on Google's model page and set it in `helping_eyes/.env`; the log line after "Language model returned HTTP 404" quotes Google's own explanation. |
| "The language model is busy" | Every model of `LLM_MODEL` and `LLM_FALLBACK_MODELS` answered "busy" (HTTP 429 or 503: the free-tier limit, or Google's "high demand") or did not start answering within 8 s. The log names each one (`... is busy (HTTP 503): trying ...`). Wait a minute, add other models to `LLM_FALLBACK_MODELS`, or choose *Offline AI on this computer (Qwen)*. |
| "I can't reach the language model" | Check the internet connection and `LLM_BASE_URL`. |
| "Sorry, I couldn't search online right now" | No search engine and not Wikipedia could be reached: check the internet connection. |
| "I searched online but found nothing useful" | The search engines are read from their result pages, and they block a computer that searched a lot (on 12 November 2026 only Yahoo answered from the test Mac, and not always; DuckDuckGo, Brave, Google, Mojeek, Yandex and Wikipedia's page said "No results found"). The app tries each engine of `SEARCH_BACKENDS` in turn and then Wikipedia's own search API, which knows medicines by their generic names. Wait a minute and try again, or change the order in `SEARCH_BACKENDS`. |
| The web lookup is never offered | The offer appears only when the answer is not in the text, and never for expiry, batch or price. Say "look it up" to search directly; the captured text is still checked first, and the web is used only if it cannot answer. |
| The first answer is slow | Free model tiers can be slow at busy times: a model that has not started within 8 s hands over to the next one of `LLM_FALLBACK_MODELS` (measured 12 November 2026: answers came after 1 to 23 s, the web answer after 33 s). For quick answers choose *Offline AI on this computer (Qwen)* (1.4 s to the first spoken word). |
| "The server is not responding..." | The app is starting up (the text reader loads in the background and `/api/health` shows `"ocr":"ready"` when done), `run.py` was stopped, or the Mac went to sleep. Restart `run.py`; with `--share` the public address is new each time. |
| The camera does not start | Allow camera access in the browser (the padlock in the address bar). Another application (FaceTime, Zoom) may be using it. Use `http://localhost` or HTTPS, or Upload photo. |
| The wrong camera opens | Choose another camera in the browser's camera settings; on a phone the rear camera is requested. |
| Stuck on SHOW TEXT HERE | Check that the page is connected (the server log shows the live stream). Move closer and make sure the text is well lit and inside the guide box. |
| Book mode announces no page number | The number is small or at the very edge of the frame. Move the camera so the whole page, including its top and bottom margins, is in view. |
| Book mode does not read the page | Hold the book still for about one second; the `motion` value in the status bar must fall below the `still <` threshold. Keep hands off the page. |
| Book mode restarts a page after the book moved | It no longer does: while a page is being read, nothing the camera sees (the book moving, a hand, even a page turn) interrupts it. Reading stops only when you ask ("stop", the Stop button) or the page is finished; a page turn seen meanwhile is read afterwards. Between pages, a small nudge is ignored because the page layout is compared after sliding it a little. |
| A new item is not captured | Move the previous item out of view briefly, press **New item**, or say "next". |
| Nothing happens when I speak | Press **Listen** and allow the microphone when the browser asks. Watch the answer area: it should show "You said: ..." as you talk. If it shows a message instead, it explains the cause and listening is switched off: "could not reach its speech service" means the browser's speech recognition is offline or blocked (it needs internet and does not work in some browsers, for example Brave); "No microphone was found" or "access was denied" is fixed in the browser's site settings (the icon in the address bar). Type your question meanwhile. |
| A spoken question is not recognised on a phone | Use Chrome on Android (iPhone Safari cannot listen hands-free). Each time the microphone reopens, a phone may beep; speak after the application has stopped talking. |
| A spoken question is not recognised | Use Chrome or Edge, speak after the application has stopped talking (the microphone is off while it speaks), and speak clearly at normal speed. |
| Text is read in the wrong language | Apple Vision is asked for English (`setRecognitionLanguages_(["en-US"])` in `recognize_text()`); change the language list there for other languages. |
| The page does not work after an update, or the console says `Cannot read properties of null` | The browser kept an old copy of the page next to a new script. The server now makes the browser check the page and its files on every load (`Cache-Control: no-cache`), so a normal reload fixes it; if it persists, reload hard (Cmd+Shift+R) or close the tab and reopen it. The script also ignores a missing shutter button instead of stopping. (The `404` for `/.well-known/appspecific/com.chrome.devtools.json` in the server log is only Chrome's developer tools looking for an optional file; ignore it.) |
| No speech output | Check the browser's audio; the page needs one interaction (a click or key press) before some browsers allow sound. Check `/api/health`: `"tts_engine": "kokoro"` means the local voice is first, `"edge"` that the Edge voice is (run `python run.py --get-voice` for the local one; the start-up message has a `voice` line too). If the first voice fails, the next one speaks, and the browser's own voice is the last resort. On a phone, tap the screen once. If the voice pauses, open the browser console (F12) and look for `Server voice failed` or `Voice stalled`, and the server log for `TTS failed` or `TTS was slow`; through ngrok, the request log at `http://127.0.0.1:4040` shows refused calls. |
