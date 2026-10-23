# Helping Eyes — Computer Vision-Based Document Reading System for the Visually Impaired

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Introduction](#introduction)
3. [Objectives](#objectives)
4. [System Architecture](#system-architecture)
5. [Design Approach](#design-approach)
6. [Project Structure](#project-structure)
7. [Hardware & Software Requirements](#hardware--software-requirements)
8. [Environment Setup](#environment-setup)
9. [Installation](#installation)
10. [Running the System](#running-the-system)
11. [Usage & Controls](#usage--controls)
12. [Computer Vision: Book Reading Mode](#computer-vision-book-reading-mode)
13. [Performance](#performance)
14. [Innovations](#innovations)
15. [Known Limitations](#known-limitations)
16. [References](#references)
17. [Troubleshooting](#troubleshooting)

---

## Problem Statement

Visually impaired individuals face significant barriers when accessing printed materials such as books, documents, medicine labels and signage. Existing assistive solutions have several limitations:

- **Cost:** most commercially available reading aids are expensive.
- **Dedicated hardware:** they require special-purpose devices rather than equipment people already own.
- **Low-light performance:** real-time reading in poor lighting remains difficult for traditional OCR systems.
- **Accuracy:** conventional OCR pipelines struggle with blur, complex backgrounds and varied fonts.
- **Information overload:** reading an entire label aloud forces the user to listen to everything to find the one detail they need.

Helping Eyes addresses these gaps with an affordable reading assistant that runs in an ordinary web browser, on any computer or phone with a camera, and answers the user's questions about what the camera sees.

---

## Introduction

**Helping Eyes** is an assistive reading application for visually impaired users. It is one web application: a browser page owns the camera, microphone and speaker, and a Python server does the reading. It runs on your Mac (`python run.py`), and `python run.py --share` gives it a public address; the only thing to configure is the language model's API key (and, optionally, its address and model name). Rather than reading everything aloud, it lets the user ask about the captured text and returns only the information requested.

Core capabilities:

- **Live text detection** on any object (pages, books, medicine packs, labels, signs and screens) from the camera preview.
- **Spoken guidance** ("Move closer", "Move left", "Glare on the right, tilt the page") until the text is in view, followed by automatic capture. "Move closer" is said only when the text is small, the reader is not sure of it (mean confidence below 0.8) **and** the sheet does not already fill the picture: `page_fill()` (in `vision.py`) measures roughly how much of the frame the bright sheet covers (a page, an open book or a lit tablet screen; Otsu threshold on a small copy, about 1 ms), and at 75 % or more that is close enough, so no hint is given. Small text that is read well, or a sheet that fills the picture, is captured without that hint. A round **shutter button** on the camera picture (and the **Capture now** button, key `C`) takes the photo immediately, whatever the guidance says, and uses it.
- **Spoken or typed requests** about the captured text:
  - *"Read everything"* reads the full captured text exactly as recognised.
  - *"Read only the dosage"* reads just the requested part.
  - *"When does it expire?"* or *"Is this safe for children?"* is answered from the text.
- **Hands-free listening:** the microphone listens only while the application is quiet, so it never answers itself.
- **Language model:** answers come from a hosted chat model (Gemini Flash-Lite by default; any OpenAI-compatible API works) and are spoken sentence by sentence as they are generated.
- **Natural spoken voice:** every sentence is spoken by the free **Kokoro** voice, which runs on your own Mac (no internet, about 0.3 s for a short sentence and 2 to 3 s for a long one); if it is not set up (`python run.py --get-voice` sets it up once) or fails, a free Microsoft Edge neural voice speaks instead (female by default, about a second per sentence). Serial and batch numbers above 9999 are read digit by digit. If both fail for a sentence, the browser's own voice speaks that sentence.
- **Optional web lookup:** when the text does not contain the answer, the application offers to search online and does so only with the user's consent. The captured text is always checked first, and web answers give only what was asked.
- **Book reading mode:** a custom computer vision pipeline detects page turns, separates two-page spreads, recognises printed page numbers and running titles on any page, orders columns and paragraphs, and highlights the exact word being read on screen. The language model then groups the recognised lines into paragraphs, so a sentence that runs over a line, a column or a page is read without a pause (the words are never changed).

Text recognition (Apple Vision) and layout analysis run on the server, which is your own Mac. Only the captured text (never the picture) is sent to the language model, and only the search words are sent for a web lookup. Speech recognition uses the browser's speech service, and the spoken voice is made on this Mac by Kokoro (nothing is sent); only if Kokoro is not set up does the free Edge voice service get the sentences to be spoken.

---

## Objectives

- Detect text automatically from a live camera feed, on any object.
- Extract text accurately and quickly with Apple Vision, the text recognition built into macOS.
- Let the user request exactly the information they need instead of listening to everything.
- Answer questions only from the captured text, without inventing details.
- Read books page by page without manual interaction.
- Run on ordinary hardware: a Mac as the server (Apple Vision reads the text), any phone or computer with a browser as the camera, and a free language-model API key; no GPU and no paid service.

---

## System Architecture

<img src="docs/architecture.png" width="900" alt="Life cycle of a request: camera pictures go through the coach (vision.py), the photo fixer and the text reader into saved text; voice or typing goes through speech to text and the command check to the language model; everything ends as speech and on-screen output">

The picture shows the life cycle of one request, from input to output. The diagram below shows the parts it passes through and how they connect.

```mermaid
flowchart TB
    subgraph BR["Browser (web/app.js): camera, microphone, speaker, overlay"]
        CAM[Camera or uploaded photo]
        MODEL[Model menu<br/>Gemini or Qwen 3B]
        MIC[Voice or typed request]
        OUT[Speech out and overlay<br/>guide box, page layout, highlighted word]
    end

    subgraph SV["Server (server.py)"]
        WS["/ws/live<br/>preview frames"]
        CAP["/api/capture<br/>one photo"]
        QLT["/api/quality<br/>uploaded photo: sharpness, light, glare"]
        BK["/api/book/page<br/>one book view"]
        ASK["/api/ask<br/>one request"]
        CMD["commands.py<br/>command or question?"]

        subgraph VIS["vision.py"]
            LIVE["Part D: live guidance<br/>hints, hold still, capture, page turns"]
            BOOKM["Part C: book mode<br/>spine, page numbers, reading order,<br/>resume after the view moves"]
            QUAL["Part B: image quality and page geometry<br/>blur / glare, flatten, dewarp, shadows"]
            OCRP["Part A: reading text (OCR)<br/>Apple Vision"]
        end

        DA["assistant.py<br/>answers only from the captured text,<br/>expiry verdict computed in code"]
    end

    VOICE["commands.py /api/tts<br/>local Kokoro voice<br/>(Edge, then browser voice, if it fails)"]
    LLM[Language model API<br/>Gemini Flash-Lite by default]
    QWEN[Qwen 3B on this Mac<br/>through Ollama]
    WEB[DuckDuckGo<br/>only with consent]

    CAM --> WS
    CAM --> CAP
    CAM --> QLT
    QLT --> QUAL
    MODEL --> ASK
    CAM --> BK
    MIC --> ASK

    WS --> LIVE
    BK --> LIVE
    LIVE --> BOOKM
    LIVE --> QUAL
    CAP --> QUAL
    BOOKM --> QUAL
    QUAL --> OCRP
    LIVE --> OCRP
    BOOKM --> OCRP
    BOOKM -- page text and facts --> DA
    CAP -- text --> DA
    ASK --> CMD
    CMD --> DA
    CMD -- unclear wording --> DA
    DA --> LLM
    DA -- if Qwen is chosen --> QWEN
    DA -- after you agree --> WEB
    DA -- each sentence --> VOICE

    WS -- hints, boxes, capture, page turns --> OUT
    BK -- text to read, page layout --> OUT
    ASK -- sentences, commands --> OUT
```

---

## Design Approach

### Software Design

| Stage | Component | Description |
|---|---|---|
| Image acquisition | Browser camera (`getUserMedia`) | Rear camera at up to 1080p; small preview frames go to the server over a WebSocket, the full-size photo only when capturing |
| Text detection | Apple Vision (`vision.py`, part A) | Locates every line of text in each preview frame |
| User guidance | `vision.py`, part D (`LiveGuide`) | Spoken positioning hints, blur / glare coaching ("Low light" is said at most once every 30 s), then a short hold-still countdown and automatic capture |
| Capture | OCR engine, accurate mode | Recognises all text once and stores it; nothing is read aloud yet |
| Voice input | Browser `SpeechRecognition` | Converts spoken requests to text, hands-free; the page also accepts typed requests |
| Request routing | `commands.py` | Decides whether a request is a command (stop, next, book mode, yes / no, search) or a question; the browser has no command logic of its own. Clear wordings are matched directly; unclear ones that mention the web, searching or looking up (or that answer the "look it up online?" offer) are read by the chosen language model, so any wording that means "look it up" works |
| Understanding | Language model API (Gemini Flash-Lite by default, or Qwen 2.5 3B on this Mac through Ollama, chosen in the page's Model menu; `assistant.py`) | Reads requested parts or answers questions using only the captured text |
| Expiry checks | `assistant.py` | Expiry dates are compared with the current date in code, not by the model |
| Web lookup | DuckDuckGo (`ddgs`) and the language model | Used only with the user's consent; answers are prefixed with "According to the web" |
| Image quality | OpenCV layer (`vision.py`, part B) | Blur, glare and exposure scoring with spoken coaching; item outline grown from its own text (GrabCut, so a pack lighter or darker than the table is found alike) and homography flattening; curved-line dewarping; shadow removal and CLAHE when quality is low |
| Book reading | OpenCV pipeline (`vision.py`, parts C and D) | Page-turn detection, spread splitting, layout analysis, reading order, resuming after the view moves |
| Text-to-speech | Kokoro-82M on this Mac (`kokoro-onnx`, in `commands.py`, `POST /api/tts`), then the Edge neural voice (`edge-tts`), then the browser `speechSynthesis` | Each sentence is turned into MP3 audio, first by the local Kokoro voice (default `af_sarah`, American female; settings `TTS_ENGINE`, `KOKORO_VOICE`, `KOKORO_SPEED`, `KOKORO_MODEL_DIR`; about 5 times faster than real time on an M3 Pro CPU; the model is loaded and warmed up at start, and the audio is encoded with `lameenc`). Set it up once with `python run.py --get-voice` (about 350 MB, kept in `helping_eyes/voice_models/`, not in git). If it is not set up or it fails, the free Microsoft Edge neural voice (default `en-US-JennyNeural`, female; settings `TTS_VOICE`, `TTS_RATE`, `TTS_ENABLED`) speaks instead: no account, key or card. `/api/health` shows which one is first (`tts_engine`). The next few sentences are made while one plays; because the local voice is computer work and not a network call, the page waits 15 s (not 3 s) before sending a second request. A text longer than 300 characters (700 for the Edge voice; a book page is voiced in pieces of that size too) is cut at sentence ends, commas or spaces into pieces that are voiced one by one and joined, so nothing is dropped. A request that never answers does not stall the voice: if the first is not back after 3 s (or has failed) a second is sent and the first to arrive wins (everything is given up after 12 s), and the server itself retries a hung call to the voice service once, with a short limit. If both fail, the browser's own voice speaks that one sentence; only after 3 sentences in a row fail is the browser voice used for 20 seconds before the server voice is tried again. A request that takes over 5 s is logged as `TTS was slow`. Numbers above 9999 (serial, batch and phone numbers) are spoken digit by digit. Queued and interruptible; the spoken word is estimated from the playing time for the on-screen highlight |

---

## Project Structure

```
Helping_Eyes-main/
├── helping_eyes/           # The application: server, vision, commands, assistant and the web page
│   ├── server.py           # FastAPI server: live stream, capture, book pages, requests, health; serves the page with no-cache headers
│   ├── vision.py           # All the computer vision in four parts: A reading text (OCR), B image quality, C book mode, D live guidance
│   ├── commands.py         # What a spoken or typed request means, and the spoken voice (local Kokoro, Edge as backup)
│   ├── assistant.py        # Question answering through a language model API, expiry checks, web lookup, book paragraphs
│   ├── voice_models/       # The Kokoro voice model files (python run.py --get-voice downloads them; not in git)
│   ├── web/                # The page: index.html (with the round shutter button on the camera picture), app.js (camera, uploads, speech, overlay), style.css (fits a laptop screen with no scrolling)
│   ├── tests/              # one test file per part of vision.py (ocr, quality, book, live), plus commands, llm, server, run and web_mic
│   └── .env                # Your settings: the key and optional choices (git-ignored, never committed)
├── markdown/               # Every guide other than this README:
│   ├── working.md          #   plain-language explanation of every file and of a request's life cycle
│   ├── hosting.md          #   plain-language guide to hosting the app from your Mac
│   ├── direction.md        #   what each button does and what you can say or type
│   ├── computer_vision.md  #   every computer vision idea of the project (pixels, maths, each step) in simple words
│   ├── eval_plan.md        #   the real-world evaluation plan (OCR, answers, speed)
│   ├── eval_shotlist.md    #   which photos to take for the evaluation, and how to import them
│   ├── eval_photo_summary.md  # results of the photo evaluation (numbers only; generated)
│   └── eval_photo_report.md   # every photo with its boxes, OCR text and all answers (generated; not in git)
├── eval/                   # The evaluation: tools/ (photo intake, folder import, photo checks), run_photo_eval.py (OCR engines, image checks drawn as boxes, every question to the local model) photo_report.py (scores and report) and make_check_folders.py (one folder of photos per image check: blur, glare, dark, slant, outline); data/private/ is not in git
├── docs/                   # Diagrams and screenshots used in the docs (make_architecture.py redraws the architecture diagram)
├── presentation/           # Slides, narrated video, script and the course requirements document
├── .github/workflows/      # check.yml: runs the tests on every push
├── run.py                  # Runs the whole app on this computer: checks, starts the server, opens Chrome
├── requirements.txt        # The only requirements file: every dependency (app, tests and evaluation)
└── README.md
```

| File | Main components |
|---|---|
| `server.py` | `/ws/live` (preview frames in, guidance and page turns out) · `/api/capture` (photo → text) · `/api/quality` (uploaded photo → sharpness, light, glare, text boxes) · `/api/book/page` (book view → what to read, from where, and the page layout) · `/api/ask` (one request → commands, answers, web lookup, streamed as NDJSON) · `/api/tts` (one sentence → MP3 audio, with its own rate limit) · `/api/health` (status, key configured, OCR ready, memory, voice on) · one session per browser (30 min) and a per-client rate limit |
| `vision.py` | **Part A, reading text (OCR):** `find_text_region()` for live detection · `recognize_text()` for full recognition, including per-word boxes · `read_text_enhanced()` tries corrected versions of a poor frame and keeps the best read · **Part B, image quality and page geometry:** `assess_quality()` (variance of the Laplacian, saturated glare blobs, exposure) · `find_page_quad()` and `warp_page()` (the item outline: grown from the text lines with GrabCut by `find_item_quad()`, or for a book page not yet read the bright paper region; `approxPolyDP`, homography) · `page_fill()` (how much of the picture the item fills, from the same item mask) · `estimate_dewarp()` and `dewarp()` (per-column vertical shift model for curved lines) · `enhance_tone()` (shadow removal, CLAHE) · `binarize()`. Set `VISION_ENHANCE=0` to disable; `BLUR_MIN` tunes the blur threshold · **Part C, book mode:** `PageTurnDetector` (frame differencing state machine) · `estimate_skew()` and `rotate_expand()` (a slanted page is straightened before the layout analysis; every line, word, paragraph and column keeps an outline that follows the slant, so the boxes on screen lie along the text) · `split_spread()` (spine detection) · `split_page_parts()` and `parse_page_number()` (header, footer, printed page number, running title) · `xy_cut()` (reading order) · `read_page()` · `arrange_page()`, `regroup()`, `line_hints()` (the model's paragraphs, built on the original lines) · `continue_from()` (compares a moved view with the page being read) · **Part D, live guidance:** `LiveGuide` (guidance hints, hold-still timing, quality coaching, capture trigger and re-arming) · `BookWatcher` (page turns on the live frames) · `BookReader` (new page, resume from the word reached, keep reading, nothing new) · `speech_chunks()` (page split into paragraphs for tracked speech) |
| `commands.py` | `classify()` (with an optional language-model `judge` for unclear wording) turns a request into stop, repeat, new capture, book mode on / off, yes / no (only while an offer is open), search, read everything or a question · the voice: `synthesize()` turns one sentence into MP3 with the local Kokoro voice (`kokoro-onnx`), or the free Edge neural voice (`edge-tts`) if Kokoro is not set up or fails, `spoken_form()` spells out numbers above 9999 digit by digit, and a small cache avoids repeating work |
| `assistant.py` | `DocAssistant.set_document()` and `ask()` stream answers sentence by sentence from any OpenAI-compatible chat API (`set_provider()` picks Gemini or the local Qwen 3B per request) · `judge_intent()` reads unclear yes / no / search wording · `ask_web()` searches and answers from the results · `wants_read_all()` detects full read-out requests · `expiry_checks()` determines whether expiry dates have passed · `web_lookup_allowed()` excludes item-specific questions such as expiry, batch and price |
| `web/app.js` | Camera and frame streaming · overlay (guide box, page layout, reading highlight) · speech output (plays the server's audio, prefetches the next sentences, estimates the spoken word for the highlight, falls back to the browser voice sentence by sentence; a sound that stops moving is nudged twice, then the browser voice says the rest of that piece from where it stopped, so nothing is skipped; every timer belongs to its sentence and is cleared when speech stops; on a phone the sound is unlocked at the first real tap, and a sentence the phone blocks waits for the next tap without freezing the camera guidance or the microphone (the start-up greeting is simply dropped); console lines `Voice: N characters ready in X ms`, `Server voice failed` and `Voice stalled at …` show where a pause comes from) · asks for the camera and the microphone in ONE request as soon as the page opens (one permission question; the microphone is released at once, and without a microphone the camera alone starts) · hands-free speech input with an echo guard, phone-safe results (a result that repeats the phrase so far replaces the previous one) and a 0.5 s pause before reopening the microphone on a phone, live "Hearing: ..." text and spoken explanations when the microphone or the speech service fails · uploaded photos (shown in place of the camera with the same overlay, glare marker and book-mode reading highlight; New item returns to the camera) · the Model menu · keyboard shortcuts. It holds no decision logic |

---

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
| `ddgs` (DuckDuckGo) | Web search, only with the user's consent; no account or API key required |
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
```

The **Model** menu on the page chooses who answers: Gemini (online, the settings above) or Qwen 2.5 3B running on this Mac through [Ollama](https://ollama.com) (`python run.py` starts Ollama for you if it is installed and not already running (stopping it at the end only if it started it), and loads the Qwen model into memory in the background so the first answer is not slow (it tells you if the model still has to be downloaded with `ollama pull`); `--no-ollama` skips this; no key needed). Qwen settings: `QWEN_3B_MODEL` (default `qwen2.5:3b-instruct`, as listed by `ollama list`) and `QWEN_BASE_URL` (default `http://localhost:11434/v1`). The choice is sent with every request and remembered by the browser. To use another provider, change `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` (for example Groq or OpenRouter, both of which have free tiers); no code changes are needed. Optional image settings: `VISION_ENHANCE=0` (turn off the image-correction layer) and `BLUR_MIN`. The camera is chosen in the browser. Optional voice settings: `TTS_VOICE` (default `en-US-JennyNeural`, female), `TTS_RATE` (speed, e.g. `+10%`) and `TTS_ENABLED=0` (always use the browser voice).

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
2. After *"Got it"*, ask aloud (press **Listen** once for hands-free listening; the words appear as "Hearing: ..." while you speak) or type the question and press Enter.
3. When you press **Capture now** (or the round shutter button, or `C`) the photo taken stays on the screen, with the detected text outlined, and the camera is switched off, until you press **New item** (`R`) or say *"next"*. (If no text is found in the photo, the camera comes straight back so you can try again. In book mode the camera stays on, because pages are turned hands-free.) To capture another item, press **New item**, say *"next"*, or move the current one out of view briefly when the camera is on.
4. For a book, press **Book mode** (or say *"book mode"*), hold the book open and turn the pages.

The camera works on `http://localhost` and on any `https://` address, but not on a plain `http://` address on the network.

#### Share from your laptop (free, no account, no card)

```bash
brew install cloudflared       # once
python run.py --share          # prints a public https address
```

`--share` opens a free Cloudflare Quick Tunnel to the server on your laptop, so a phone or a friend can open the app over `https` (the camera and microphone need it). On a Mac it also runs `caffeinate` so the laptop does not fall asleep from being idle. The address works only while the laptop is awake and `run.py` is running, and it changes every time. The laptop's own memory covers the roughly 1 GB the OCR needs. Closing the lid on battery still sleeps the Mac. Anyone with the address uses your language-model key's quota, so share it with people you trust.

### Hosting

The Mac is the server (Apple Vision exists only on macOS). `python run.py --share`, described above, gives it a public `https` address with a free tunnel: a fixed ngrok address when `NGROK_DOMAIN` is set in `helping_eyes/.env` (free account and domain, see markdown/hosting.md), otherwise a Cloudflare tunnel whose address changes each run. A step-by-step guide, with the settings and what to expect, is in [markdown/hosting.md](markdown/hosting.md). After starting, open `/api/health`: `llm_configured` must be `true` and `ocr` must be `ready`.

### Tests

```bash
cd helping_eyes
for t in tests/test_*.py; do python "$t" || break; done
```

---

## Usage & Controls

### Spoken or typed requests

| Request | Result |
|---|---|
| "Read everything", "read it", "what does it say" | Reads the full captured text exactly as recognised |
| "Read only the directions", "read the ingredients" | Reads only the requested parts |
| Questions such as "When does it expire?", "How much does it cost?", "Can children take this?" | Answered from the captured text |
| Follow-up questions such as "And how often?" | The last few exchanges about the same item are retained |
| "Yes" or "no" after *"Should I look it up online?"* | Performs or declines the web lookup |
| "Look it up", "search online" | Searches the web for the previous question |
| "Look up tartrazine allergy", "search the web for ..." | Searches the web for the given phrase |
| "Repeat" | Repeats the last answer |
| "Stop" | Stops speaking |
| "Next", "new page", "scan again" | Prepares to capture a new item |
| "Book mode", "read my book" | Switches to book reading mode |
| "Normal mode", "stop book mode" | Returns to single-item capture |

### Buttons and keyboard (when the cursor is not in the text box)

| Key | Button | Action |
|---|---|---|
| `C` | Capture now (also the round shutter button on the camera picture) | Capture immediately, with a flash so you see it was taken: the photo stays on the screen and the camera goes off until New item or "next" (in book mode: read the page in view, camera stays on); a second press waits while a photo is being read |
| `R` | New item | Close the photo that stayed on the screen (or an uploaded one) and go back to the live camera, ready for a new item |
| `B` | Book mode | Toggle book mode |
| `A` | Read everything | Read the full captured text |
| `P` | Repeat | Repeat the last answer |
| `S` | Stop | Stop speaking |
| `M` | Listen / Stop listening | Turn hands-free listening on or off. The button shows whether the microphone is open right now: it goes off while the app talks or thinks and turns itself on again when it is quiet. Pressing it while the app talks stops the talking and listens at once |

On a laptop-sized window (at least 1000 × 560) the whole page fits the screen without scrolling: the camera takes the left half and shrinks to the height that is left, and the answer scrolls inside its own box. Phones keep the normal scrolling page.

Upload photo shows the picture where the camera preview is (the camera is switched off while the photo is shown; Back to camera, or New item, switches it back on and returns to the live view). In book mode the uploaded page is read like a camera page: the same overlay (spine, columns, paragraph order) is drawn on it and the line and word being spoken are highlighted. In normal mode, tick the box beside it for a two-page spread. The photo's sharpness, light and glare are shown on it as on the live view, as numbers and a hint only, nothing is drawn on the picture (`POST /api/quality`).

The microphone listens only while the application is silent, so that it does not capture its own voice. Use `S` to interrupt a long answer.

**Overlay states:** `SHOW TEXT HERE` → `ADJUST POSITION` → `HOLD` → `CAPTURING...` → `CAPTURED - ASK ME`, and `THINKING...` / `SPEAKING...` while answering. In book mode the overlay shows the spine, columns, numbered paragraphs, and the line and word being read.

---

## Computer Vision: Book Reading Mode

Book mode reads a book page by page without manual interaction: the user opens the book in front of the camera, listens, and turns the page. The OCR engine recognises the characters of each line; **the decisions of when to read, where each page lies and in what order to read are made by the project's own OpenCV pipeline** in [`vision.py`](helping_eyes/vision.py) (parts C and D).

<img src="docs/book_mode_overlay.jpg" width="720" alt="Book mode overlay on a synthetic two-page spread: spine line in yellow, page numbers and running title outlined in purple, paragraph boxes in green numbered 1 to 6 in reading order">

*Book mode on a synthetic two-page spread: the detected spine (yellow), the printed page numbers and running title (purple, announced but not read in the text flow) and the paragraphs (green), numbered in reading order. The status bar shows the recognised page numbers.*

```mermaid
flowchart LR
    F[Camera frames] --> M[Frame differencing<br/>motion relative to learned noise floor]
    M --> S{State machine<br/>TURNING → SETTLING → STEADY}
    S -- steady for 0.8 s --> SP[Spine detection<br/>split the spread]
    SP --> V[OCR<br/>lines and word boxes]
    V --> H[Header / footer rows<br/>page number, running title]
    V --> X[Recursive XY-cut<br/>columns, headings, paragraphs]
    H --> D{Same page?<br/>page numbers and words}
    X --> D
    D -- new page --> R[Announce page number,<br/>read from the first line]
    D -- same page --> C[Continue from the<br/>word reached]
```

### 1. Page-turn detection: frame differencing and a state machine

- Each frame is reduced to 320×180, converted to grayscale and blurred. The **mean absolute difference** from the previous frame measures motion, and a **median over the last five frames** suppresses isolated noisy frames.
- The detector **learns the camera's noise floor** (a running average of motion while the scene is still) and sets its movement and stillness thresholds relative to it, making it robust across cameras and lighting conditions.
- A state machine (`TURNING` → `SETTLING` → `STEADY`) reports an event once the view has been still for 0.8 s after movement.
- **Layout fingerprint:** an 80×45 binary image of the page layout (adaptive threshold with horizontal dilation, so that each text line forms a band) distinguishes a clearly different layout (`changed`) from the same layout after movement (`moved`). Because dense book pages have almost identical layouts, a `moved` event is still checked against the page's words and printed page number (section 5) rather than being ignored.
- **Vibration and nudges:** the fingerprints are compared after sliding one over the other by up to 4 cells (and a light blur), so a book that shifted or vibrated slightly is still the same page; a different page differs at every alignment (measured: 0.01 for a shifted page, 0.28 for a different one, against a threshold of 0.06). When the layout looks the same (`moved`) but the OCR words are too garbled to line up (a blurred frame), `BookReader` still keeps reading (`MOVED_SAME_PAGE`) and never restarts the page from the first word.

### 2. Two-page spread splitting: spine detection

- **Book localisation:** an Otsu threshold on the blurred image separates the paper from the background; a morphological closing fills in the text, and the union of large paper regions gives the book's outline. Only regions clearly wider than they are tall are treated as spreads; single pages are not split.
- **Shadow cue:** a partially opened book casts a shadow along the spine, visible as a **dark valley** in the column-wise mean brightness. When the valley is at least 12% darker than both pages, the spread is split there.
- **Text-free strip cue:** a fully opened book casts little or no shadow, and the spine region is often the brightest part of the image. Because text never crosses the spine, the two inner margins form the **smoothest vertical strip** of the book. Texture is measured as the **local standard deviation of brightness** per column (text remains textured even when slightly blurred, blank paper does not), and the smooth strip nearest the centre of the book is selected. The strip must be at least 7% of the book's width, which is wider than the gap between two columns of one page.
- **No assumed middle line:** a spine is reported only when one of these cues is found *and* both sides of it hold a real block of text (at least a fifth of the book's width each). A single sheet, a page with two columns, a blank sheet, or a picture where only one side has text gets no spine line, and the page is read as one page. There is no fallback that splits the book down the middle.

### 3. Page structure and reading order

The page structure is derived from the boxes of the lines the OCR recognised, which are reliable whenever the text is readable, even on camera frames that are slightly blurred.

- **Rows:** lines that overlap vertically are grouped into rows (for example, a page number and a running title on the same header line).
- **Header and footer:** the first and last rows are treated as a header or footer when they are separated from the body by a gap clearly larger than the usual line spacing and are either short or contain a page number. They are **announced, not read in the text flow**.
- **Printed page number:** recognised in the header or footer in the forms `47`, `- 47 -`, `Page 47`, `xii` (Roman numerals), `47  THE SILENT RIVER` and `CHAPTER THREE  48`. On a two-page spread, a number missing on one page is inferred from the facing page. A new page is announced as *"Page 47"* or *"Pages 46 and 47"*; when no number is printed, no number is announced.
- **Running title:** the remaining header or footer text (book or chapter title) is announced when it changes.
- **Headings:** lines set in noticeably larger type, or beginning with words such as *Chapter*, *Part* or *Prologue*, are recorded and read at their position in the text.
- **Reading order, recursive XY-cut:** a classic document layout algorithm applied to the line boxes. A region is split at the widest vertical gap between line boxes that is wider than a column gap (columns, read left first); otherwise at a horizontal gap clearly larger than the usual line spacing (headings, paragraph breaks, read top first); the parts are processed recursively. A region that cannot be split is read line by line, and an **indented first line** starts a new paragraph. Words hyphenated across lines are rejoined. **Paragraphs by the language model:** these heuristics cannot tell a sentence that continues in the next column or page from a new paragraph, and every false paragraph is a pause. A sidebar, box or caption beside the main text is worse: its rows are level with the main text's rows, so the reading order alternates between the two. So before a page is spoken, `line_blocks()` groups the lines into blocks from their boxes alone (lines stacked under each other with the same left edge and span: a column, a heading, a sidebar each become a block), and the numbered lines, listed block by block with their position, letter size and layout tags (`[page]`, `[column]`, `[gap]`, `[indent]`, `[short]`), go to the model (`DocAssistant.arrange_lines()`, `ARRANGE_PROMPT` in `assistant.py`), which returns only a grouping and an order, as JSON: the paragraphs in reading order (the main text of a page first, then its sidebars; a sentence that runs from one block into another is joined), each a list of line numbers, plus lines to leave out (page numbers, stray fragments). The code rebuilds the page text from the original lines, so no word is ever changed and the highlight keeps working. A reply that loses or repeats a line, skips more than a quarter of the page, is slow (12 s limit) or fails is ignored, and the page is read with the heuristic paragraphs. This adds about 2 to 3 seconds before a new page starts; the same page is not sent twice.
- The page facts (page numbers, running title, headings) are also passed to the language model, so questions such as *"What page is this?"* or *"Which chapter is this?"* can be answered.

### 4. Reading position highlight

While a page is read aloud, the display follows the speech word by word: the **line being spoken is highlighted** and the **word being spoken is outlined**.

<img src="docs/book_mode_highlight.jpg" width="720" alt="Book mode reading highlight: the line being spoken highlighted in yellow and the current word outlined in orange">

- **Spoken position:** `app.js` estimates the word being spoken from how far through the audio it is (the server gives no word times; the browser fallback voice reports the word through the `boundary` event of `speechSynthesis`) and converts it to a character position in the page text. The server splits the page into paragraphs (`speech_chunks()` in `vision.py`), and each is spoken separately, so the start-up delay coincides with the natural pause between paragraphs. Voices that send no word events still advance the position paragraph by paragraph.
- **On-page location:** during recognition, the OCR engine provides a bounding box for every word, and `read_page()` records which characters of the page text belong to which printed line and word (`PageLayout.lines`). The server sends this layout to the page with each new book page.
- Combining the two yields the exact on-screen location of the word being spoken, drawn on the overlay in the browser.

### 5. Continuing after the view moves

If the camera or the book moves, the new view is compared with the page being read before anything is spoken again (`continue_from()` in `vision.py`):

- The words of both views are aligned with a sequence matcher (`difflib.SequenceMatcher`). Only matching runs of at least three consecutive words count, so common words shared by different pages do not register as an overlap.
- **Printed page numbers take precedence:** if both views show page numbers and they differ, the view is a new page regardless of the words.
- **Same words in view:** reading continues from the exact word it had reached, without announcing a new page or repeating anything. If the book was only slightly disturbed (for example, a hand passed over the page) while reading, speech continues without interruption. If the camera moved so that new lines came into view, those are read after the part already covered.
- **Different words:** the view is treated as a new page and read from the beginning.
- While the view is steady, the page layout is still re-checked once per second, so a slow drift of the camera or book is also noticed.
- The conversation about the page is kept when the view merely moves, so follow-up questions still work.

### Evaluation

| Test | Result |
|---|---|
| Spine detection, spread with a shadow (synthetic) | Exact (x = 960 of 1920; 640 of 1280) |
| Spine detection, fully opened book captured with the USB camera (no shadow) | x = 645, within the blank strip between the pages (approximately 570–665) |
| Spine detection, flat spreads without a shadow (synthetic) | Within the blank strip between the pages |
| Single portrait page | Correctly not split |
| Page structure, 6 synthetic layouts (spreads with header page numbers, chapter heading, hidden page number; single pages with footer number, header and heading, two columns) | 6/6: correct printed page numbers (including one inferred from the facing page), running titles and headings; reading starts at the first body line; page numbers are not read in the text flow |
| Paragraphs marked only by first-line indentation (novel style) | 5 / 5 found |
| Opening a random spread, then turning two pages (pages with near-identical layouts) | Announced "Pages 146 and 147", then "Pages 148 and 149", then "Pages 150 and 151"; the running title was announced once; the chapter heading was read at its position |
| Page-turn sequences (still page, hand passing over, page turned, new page), 10 random noise seeds | 10/10 correct: first page read after about 0.9 s of stillness, no re-read when a hand passes over, new page read about 1 s after the turn |
| Words mapped to their on-screen box (two-page spread) | 169 / 169 |
| View moved mid-read (camera shifted 320 px; top lines left, new lines entered) | Reading continued from the exact word reached, with no new-page announcement; the newly visible lines were then read |
| Different page after reading | Detected as a new page and read from the beginning |
| Processing time | Page-turn detector 0.2 ms per frame; spine and layout analysis 8 ms; full page including recognition and word boxes about 0.4–0.9 s |
| Automated tests | 157 tests: reading text (10, real Apple Vision on rendered text), image quality and page geometry (12, including the item outline on a strip lighter or darker than the floor and with text in the background), book mode (32: page numbers, header and footer, reading order, spine, page turns, resuming), live guidance and book-page decisions (30, including how much of the picture an item on a light floor fills), commands and the voice (14), language-model client against a fake provider, including expiry dates in common label wordings such as `Exp. Date 02/2027` (16), server endpoints and WebSocket with real OCR (12), `run.py` (14: helpers, the key reaching the server, a real start, port fallback, Ctrl+C stop), the page's listening, start-up and voice in headless Chrome with a fake speech engine (17; skipped without Chrome) |
| End to end in Chrome, fake camera filming a medicine label | Guidance, automatic capture (0.6 s), "read everything" and the expiry answer worked; in book mode with a looping two-page video, page turns were detected, announced and read (each page request 0.24–0.41 s). The language model itself was tested against a fake provider, not the real Gemini service |

The principal benefits of book mode are hands-free page turning, identification of the printed page number and title on any page, correct handling of two-page spreads and headers, paragraph structure, and the reading position highlight.

---

## Performance

Measured on an Apple Silicon laptop running the app locally (measured with the earlier RapidOCR reader on the CPU, before the move to Apple Vision; the model API was not measured because no key was available):

| Operation | Time |
|---|---|
| Live frame round trip (browser → server → browser) | about 0.2 s for a label, up to 0.7 s for a dense two-page spread |
| Capture request, full-size photo of a label | 0.6 s |
| Book page request (test pages, with word boxes) | 0.24–0.41 s; a dense 1920×1080 spread about 0.9 s |
| Full read-out request | Immediate (no model involved) |
| Page-number answers | Immediate (computed in code, no model call). Expiry answers use the model, which words the verdict the code computed |
| First spoken sentence of an answer | Depends on the provider and its load; not yet measured with Gemini Flash-Lite |
| Web lookup (search and answer) | Not yet measured (one search and two model calls) |
| Book mode, page turn to start of reading | About 1.5–2 s (0.8 s settling, 0.4–0.9 s recognition, speech start-up) |

### Real photos: 148 pharmacy items (22 October 2026)

148 phone photos from a pharmacy (43 tablet strips, 79 cartons, bottles, ampoules and vials, 26 creams, face washes, lotions, soaps and other personal-care items) were run through the app's own code with `eval/run_photo_eval.py`. The reference is Claude's reading of each photo, not a typed and double-checked ground truth. Full numbers: [markdown/eval_photo_summary.md](markdown/eval_photo_summary.md); every photo with its boxes and answers: `markdown/eval_photo_report.md` (kept out of git).

| Text reader | Key-line accuracy | Details found (name, strength, batch, dates, price ...) | Time per photo |
|---|---|---|---|
| Apple Vision (the app's reader) | 89.7 % | 69.8 % | 0.17 s |
| RapidOCR (the earlier reader) | 87.7 % | 68.4 % | 0.62 s |
| Tesseract | 57.6 % | 24.4 % | 0.32 s |

The text the app keeps after a capture (Apple Vision on the best version of the photo) scores 90.1 %. Apple Vision is best on cartons and bottles (96.7 %) and level with RapidOCR on foil strips (about 80 %); on one crumpled foil strip it found no text at all. The local Qwen 2.5 3B model, given the text the app kept, answered 62.8 % of the factual questions correctly (name, strength, quantity, dates, batch, price, maker); most misses come from the layout (values printed a row away from their labels, dates written as `#06/26 @05/29`), not from invented facts.

| Image check (`vision.py`, part B) | On these photos |
|---|---|
| Blur (variance of the Laplacian, below 60) | Never: text sharpness 248 to 7934 (median 1966); 21 of 8466 small text patches were blurry |
| Glare (saturated blobs over 1 % of the picture) | Never flagged. The largest saturated blob was 0.12 %: glare on foil and gloss is bright but not pure white, so the rule misses it |
| Light and contrast | Never flagged: brightness 96 to 167, contrast 34 to 70 |
| Slant | Measured on 64 photos (median 3°, up to 26°); only the book path straightens it |
| Item outline | Found on 134 photos and drawn around the item itself (none on shelves of several products or crumpled strips). The first version took the brightest region as the page, so on a light floor it outlined the floor; it was corrected on 23 October 2026 (below) |
| How much of the picture the item fills | Median 22 % (3 % to 67 %). The first version also counted the light floor (median 69 %), so on 26 photos the app would wrongly have stopped saying "Move closer" |
| Version a capture kept | The untouched photo on 134, the flattened item on 14 (now that the outline is right, a tilted item is straightened before reading) |

---

## Innovations

### 1. Question-driven reading
A medicine package can contain hundreds of words. Instead of reading everything aloud, the user asks for the specific information needed — the dosage, the expiry date or a warning — and receives a short spoken answer.

### 2. Text detection instead of object detection
Helping Eyes detects text directly rather than specific object categories, so any item with readable text — a page, a medicine strip, a food package, a sign or a screen — can be captured.

### 3. Grounded answers
The model is instructed to use only the captured text and to state when the answer is not present, and expiry dates are evaluated in code rather than by the model (the model is told the verdict and words the answer itself). Only the captured text, never the picture, is sent to the language model.

### 4. Hands-free book reading
A custom OpenCV pipeline detects page turns, splits two-page spreads, orders columns and paragraphs, and highlights the word being read, allowing a book to be read page by page without any manual interaction.

---

## Known Limitations

- **Browser:** speech recognition needs Chrome or Edge; other browsers work with typed requests. The camera needs `http://localhost` or HTTPS.
- **Free hosting is slow:** OCR runs on a shared CPU, so live guidance updates only a few times a second, and a free host sleeps when unused. Answers depend on the language model you configure.
- **Free API limits and privacy:** free model tiers are rate limited (requests per minute and per day vary by model and change over time); when the limit is reached the app says the model is busy. Free tiers may allow the provider to use submitted text to improve its products, so check the provider's terms before sending sensitive documents, or use a paid or self-hosted endpoint. "Read everything", expiry and page-number answers never call the model.
- **Internet access is required:** spoken requests are processed by the browser's speech service, and answers come from a hosted language model. The voice needs no internet when Kokoro is set up; without it the voice comes from Microsoft's free Edge voice service.
- **Mac only:** text recognition uses Apple Vision, which exists only on macOS, so the server cannot run on Linux or Windows.
- **Local voice:** Kokoro needs a one-time download of about 350 MB and some CPU while it speaks (about 5 times faster than real time on an M3 Pro, so a long piece takes a few seconds before its first words). It is strongest in English.
- **Unofficial backup voice:** the Edge voice is a free service without a published guarantee; if Microsoft changes it, the page falls back to the browser's own voice.
- **Phones:** a phone allows sound only after a real tap, so the first sentence may wait for one (the page says "Tap the screen once to turn the voice on"). Hands-free listening works in Chrome on Android; iPhone Safari only starts listening from a tap, so there use the typed question or the buttons. A hidden tab or a locked screen can pause the audio.
- **No voice interruption:** the microphone is inactive while the application speaks. Use `S` to stop a long answer.
- **Background text:** any text in view (keyboard keys, a screen) may be included in the capture. Holding the item close to the camera reduces this.
- **New captures:** a new item is captured only after the previous text leaves the view briefly, or after **New item** or "next".
- **Recognition errors:** small, curved, reflective or blurred print may be misread, and answers can only be as accurate as the captured text. On real pharmacy photos, crumpled foil strips and the purple inkjet batch and date lines printed over blister bumps were the hardest (about 80 % key-line accuracy against 97 % on cartons).
- **Glare coaching misses shiny packs:** the glare check looks for pure-white patches, so the reflections on foil strips, gold cartons and glossy bottles (bright, but not pure white) produce no "Glare" hint; in 148 real photos it never fired.
- **Labels laid out in two columns:** when a batch number, date or price is printed a row above or below its label, or coded with symbols (`#` made, `@` use before), the language model can pair the wrong value with a label or not find it.
- **Web lookups:** only the search query (the question and the product name) leaves the device; the captured text is not sent. Answers depend on the quality of search results. Lookups are never offered for item-specific details such as expiry, batch or price.
- **Book mode:** evaluated on synthetic spreads and a real camera frame. Strong page curvature, uneven lighting, fingers over the text and illustrations near the centre of a spread can affect spine and column detection. Motion blur prevents recognition, so the book should be held steady or placed on a surface. When the view moves while a page is being read, a few words may be repeated, because the reading position is sent to the server before the new view has been recognised.
- **Model accuracy:** although the model is restricted to the captured text, a small model can still misread or confuse details. Medically important information should be confirmed with a pharmacist or doctor.
- **One reader:** the earlier standalone Gemini reader (which read all text aloud with no questions) was removed; "read everything" does the same without a cloud API.

---

## References

1. U. Gawande, N. Rathod, P. Bodkhe, P. Kolhe, H. Amlani and C. Thaokar, "Novel Machine Learning based Text-To-Speech Device for Visually Impaired People," *2023 2nd International Conference on Smart Technologies and Systems for Next Generation Computing (ICSTSN)*, Villupuram, India, 2023, pp. 1–5. doi: 10.1109/ICSTSN57873.2023.10151637

2. D. S R, V. K. Gowda, R. Rai R, S. Kumar S and V. K P, "Smart Reader for Blind People," *2025 International Conference in Advances in Power, Signal, and Information Technology (APSIT)*, Bhubaneswar, India, 2025, pp. 1–4. doi: 10.1109/APSIT63993.2025.11086193

3. A. Sharma, A. Srivastava, and A. Vashishth, "An Assistive Reading System for Visually Impaired using OCR and TTS," *International Journal of Computer Applications*, vol. 95, no. 2, pp. 13–18, Jun. 2014. doi: 10.5120/16566-6231

4. Apple Vision, `VNRecognizeTextRequest`. https://developer.apple.com/documentation/vision/vnrecognizetextrequest

---

## Troubleshooting

| Issue | Solution |
|---|---|
| "No API key is set" or "rejected the request" | Put your key in `LLM_API_KEY` (`helping_eyes/.env`) and restart. `/api/health` shows `llm_configured`, and also the OCR status, the memory in use and whether the Edge voice is on. |
| "The language model name was not found" (HTTP 404 in the log) | The key works but `LLM_MODEL` (or `LLM_BASE_URL`) is wrong. Google retires and restricts old model names (`gemini-2.5-flash-lite` stopped working for new keys). Look up the current Flash-Lite name on Google's model page and set it in `helping_eyes/.env`; the log line after "Language model returned HTTP 404" quotes Google's own explanation. |
| "The language model is busy" | The free-tier limit was reached. Wait a minute, or switch model or provider with `LLM_MODEL` and `LLM_BASE_URL`. |
| "I can't reach the language model" | Check the internet connection and `LLM_BASE_URL`. |
| "Sorry, I couldn't search online right now" | Check the internet connection. DuckDuckGo may rate-limit requests; wait a minute and try again. |
| The web lookup is never offered | The offer appears only when the answer is not in the text, and never for expiry, batch or price. Say "look it up" to search directly; the captured text is still checked first, and the web is used only if it cannot answer. |
| The first answer is slow | Free model tiers can be slow at busy times. Try again, or choose another model with `LLM_MODEL`. |
| "The server is not responding..." | The app is starting up (the text reader loads in the background and `/api/health` shows `"ocr":"ready"` when done), `run.py` was stopped, or the Mac went to sleep. Restart `run.py`; with `--share` the public address is new each time. |
| The camera does not start | Allow camera access in the browser (the padlock in the address bar). Another application (FaceTime, Zoom) may be using it. Use `http://localhost` or HTTPS, or Upload photo. |
| The wrong camera opens | Choose another camera in the browser's camera settings; on a phone the rear camera is requested. |
| Stuck on SHOW TEXT HERE | Check that the page is connected (the server log shows the live stream). Move closer and make sure the text is well lit and inside the guide box. |
| Book mode announces no page number | The number is small or at the very edge of the frame. Move the camera so the whole page, including its top and bottom margins, is in view. |
| Book mode does not read the page | Hold the book still for about one second; the `motion` value in the status bar must fall below the `still <` threshold. Keep hands off the page. |
| Book mode restarts a page after the book moved | It no longer does: while a page is being read, nothing the camera sees (the book moving, a hand, even a page turn) interrupts it. Reading stops only when you ask ("stop", the Stop button) or the page is finished; a page turn seen meanwhile is read afterwards. Between pages, a small nudge is ignored because the page layout is compared after sliding it a little. |
| A new item is not captured | Move the previous item out of view briefly, press **New item**, or say "next". |
| Nothing happens when I speak | Press **Listen** and allow the microphone when the browser asks. Watch the status line: it should show "Hearing: ..." as you talk. If it shows a message instead, it explains the cause and listening is switched off: "could not reach its speech service" means the browser's speech recognition is offline or blocked (it needs internet and does not work in some browsers, for example Brave); "No microphone was found" or "access was denied" is fixed in the browser's site settings (the icon in the address bar). Type your question meanwhile. |
| A spoken question is not recognised on a phone | Use Chrome on Android (iPhone Safari cannot listen hands-free). Each time the microphone reopens, a phone may beep; speak after the application has stopped talking. |
| A spoken question is not recognised | Use Chrome or Edge, speak after the application has stopped talking (the microphone is off while it speaks), and speak clearly at normal speed. |
| Text is read in the wrong language | Apple Vision is asked for English (`setRecognitionLanguages_(["en-US"])` in `recognize_text()`); change the language list there for other languages. |
| The page does not work after an update, or the console says `Cannot read properties of null` | The browser kept an old copy of the page next to a new script. The server now makes the browser check the page and its files on every load (`Cache-Control: no-cache`), so a normal reload fixes it; if it persists, reload hard (Cmd+Shift+R) or close the tab and reopen it. The script also ignores a missing shutter button instead of stopping. (The `404` for `/.well-known/appspecific/com.chrome.devtools.json` in the server log is only Chrome's developer tools looking for an optional file; ignore it.) |
| No speech output | Check the browser's audio; the page needs one interaction (a click or key press) before some browsers allow sound. Check `/api/health`: `"tts_engine": "kokoro"` means the local voice is first, `"edge"` that the Edge voice is (run `python run.py --get-voice` for the local one; the start-up message has a `voice` line too). If the first voice fails, the next one speaks, and the browser's own voice is the last resort. On a phone, tap the screen once. If the voice pauses, open the browser console (F12) and look for `Server voice failed` or `Voice stalled`, and the server log for `TTS failed` or `TTS was slow`; through ngrok, the request log at `http://127.0.0.1:4040` shows refused calls. |
