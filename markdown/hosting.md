# How Helping Eyes is hosted

Your Mac is the server. The app reads text with Apple Vision, which exists only on macOS, so it cannot run on
Linux hosts. Nothing is uploaded anywhere: you run the app on your Mac and, if other people should use it,
give it a public `https` address with a free tunnel. No account, no card.

## One-time setup

1. Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and put it in `helping_eyes/.env` as `LLM_API_KEY=your_key` (copy `helping_eyes/.env.example` to `.env` first).
2. Install the packages: `.venv/bin/python -m pip install -r requirements.txt` (run from the top folder). For the better, local voice, then run `.venv/bin/python run.py --get-voice` once (about 350 MB); without it the free Edge voice is used.
3. Install a tunnel tool. For one fixed address every time (free), use ngrok:
   - Make a free account at [dashboard.ngrok.com](https://dashboard.ngrok.com), then under **Domains** claim your free domain (it looks like `your-name.ngrok-free.app`).
   - Run `brew install ngrok`, then once: `ngrok config add-authtoken <your token from the dashboard>`.
   - Put `NGROK_DOMAIN=your-name.ngrok-free.app` in `helping_eyes/.env`.

   Without that, install `brew install cloudflared` instead: it works with no account, but the address is new each run.

## Each time you want it online

1. Plug in the charger and keep the lid open.
2. Run `.venv/bin/python run.py --share`. On a Mac, `run.py` also stops the laptop from sleeping when idle.
3. Wait for "Ready: the text reader has loaded". The public `https` address is printed: your fixed ngrok address (it never changes), or a `https://...trycloudflare.com` one if `NGROK_DOMAIN` is not set.
4. Open that address on a phone or send it to anyone. Allow the camera and microphone when the browser asks as the page opens (they need `https`). Use Chrome or Edge: speech recognition needs one of them.
5. Press Ctrl+C when you are done. The address stops working.

To use the app only on your Mac, run `python run.py` without `--share`. It opens Chrome at `localhost`.

## Check that it works

Open `<address>/api/health`. You should see `"status":"ok"`, `"llm_configured":true` and `"ocr":"ready"`. It also shows `"memory_mb"` (what the app uses now) `"tts"` (whether a server voice is on) and `"tts_engine"` (`kokoro`, the local voice, or `edge`).

## The settings

Put these in `helping_eyes/.env` (see `.env.example`):

| Setting | Needed? | What it is |
|---|---|---|
| `LLM_API_KEY` | **Yes** | Your free key from Google AI Studio. Keep it secret. |
| `LLM_MODEL` | No | Which AI model to use (default `gemini-3.1-flash-lite`). If the app says the model name was not found, look up the current Flash-Lite name and set it here. |
| `LLM_BASE_URL` | No | Which AI service to talk to. Change it (with the model and key) to use another provider, such as Groq or OpenRouter. |
| `TTS_ENGINE`, `KOKORO_VOICE`, `KOKORO_SPEED` | No | The spoken voice. The default is the local Kokoro voice (`af_sarah`, an American female voice), which needs a one-time `python run.py --get-voice` (about 350 MB); without it the Edge voice is used. |
| `TTS_VOICE`, `TTS_RATE`, `TTS_ENABLED` | No | The backup voice (a free Microsoft Edge voice) and its speed; `TTS_ENABLED=0` uses the browser's own voice. |
| `QWEN_3B_MODEL`, `QWEN_BASE_URL` | No | The local Qwen 3B model in the page's Model menu (default `qwen2.5:3b-instruct` through Ollama at `http://localhost:11434/v1`). `run.py` starts Ollama for you if it is installed and loads the model so the first answer is quick; `--no-ollama` skips that. It runs on your Mac, so visitors using the shared address can pick it too, but it is slower when several people ask at once. |
| `NGROK_DOMAIN` | No | Your free fixed ngrok address, so `--share` always gives the same URL (see step 3 above). Without it `--share` uses a Cloudflare tunnel with a new address each run. |
| `PORT` | No | The port to use (default 7860); `run.py --port` does the same. |

## Things to know

- **The address works only while the Mac is awake and `run.py --share` is running.** With ngrok (`NGROK_DOMAIN`) it is the same every run; with the Cloudflare tunnel it is new every run, so send the new address each time. Visitors may see ngrok's notice page on their first visit.
- **Run only one copy.** Each visitor's captured text and conversation are kept in the server's memory, so two copies would not share them. A restart clears them.
- **Anyone with the address uses your Gemini key's quota**, so share it with people you trust. The app limits how many requests one visitor can make per minute.
- **The free Gemini limits.** The free AI plan has a daily limit. When it is used up, the app says the model is busy. Page-number questions and "read everything" never use the AI, so they keep working. Expiry questions need the AI to word the answer, but the dates are always worked out by the app.
- **Privacy.** The text of what you capture (never the picture) is sent to the AI service, and the sentences to be spoken are made on the Mac by the local Kokoro voice (and sent to Microsoft's voice service only if Kokoro is not set up). Check their terms before using private documents.
- **On a phone**, tap the screen once so the phone allows sound. Hands-free listening works in Chrome on Android; iPhone Safari cannot listen hands-free (type or use the buttons). If the voice pauses, check the ngrok request log at `http://127.0.0.1:4040` and the browser console for `Server voice failed`.
- **Memory.** The text reader needs roughly 0.4 to 1 GB while it reads; a Mac has plenty.

## The files that matter

| File | What it does |
|---|---|
| `run.py` | The start button: checks the setup, starts the server, opens Chrome, and with `--share` opens the tunnel. |
| `helping_eyes/server.py` | The front desk: receives pictures and requests from visitors and sends answers back. |
| `helping_eyes/vision.py` | The eyes: reads text from pictures with Apple Vision, fixes bad photos, understands book pages, and gives the "move closer" hints. |
| `helping_eyes/commands.py` | The listener and the voice: decides if what you said is a command or a question, and turns each sentence into spoken audio (the local Kokoro voice first, the Edge voice as the backup). |
| `helping_eyes/models/` | The Kokoro voice model files, downloaded by `python run.py --get-voice`. Not in git. |
| `helping_eyes/assistant.py` | The answerer: asks the AI model (Gemini, or Qwen 3B on your Mac) and searches the web when you agree. |
| `helping_eyes/web/` | The web page: `index.html` (layout), `app.js` (camera, voice and speaking), `style.css` (looks). |
| `helping_eyes/.env.example` | A template of the settings. Copy it to `.env` and put your key in it. `.gitignore` keeps `.env` out of git. |
| `.github/workflows/check.yml` | Makes GitHub run all the tests on a Mac every time you push. It does not deploy anything. |
