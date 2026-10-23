"""
Run Helping Eyes on this computer, the same app that runs on the host:

    python run.py                    start it and open it in Chrome
    python run.py --no-browser       start it and print the address only
    python run.py --port 8000        use another port (default 7860; the next free one if it is taken)
    python run.py --host 0.0.0.0     also reachable from other devices on your network
    python run.py --share            also give it a public https address: your fixed ngrok address when NGROK_DOMAIN is
                                     set in helping_eyes/.env, otherwise a free Cloudflare tunnel (a new address each run)
    python run.py --no-ollama        do not start Ollama (the local Qwen model in the page's Model menu)

It checks your setup, starts Ollama if it is installed and not already running (and stops it again at the
end only if it started it), starts the server (helping_eyes/server.py), waits until it is ready and opens
the page. Chrome does the camera, the microphone, speech to text and text to speech, exactly as it
does for a visitor on the host. The only thing you need is a language-model key in helping_eyes/.env:

    LLM_API_KEY=<your key from https://aistudio.google.com/apikey>

Without a key the app still starts; "read everything", expiry and page-number answers and the live
guidance work, and questions that need the language model say the key is missing.
Stop it with Ctrl+C.
"""

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT, "helping_eyes")
ENV_FILE = os.path.join(APP_DIR, ".env")
DEFAULT_PORT = 7860
REQUIRED = {"fastapi": "fastapi", "uvicorn": "uvicorn", "multipart": "python-multipart", "cv2": "opencv-python-headless",
            "numpy": "numpy", "Vision": "pyobjc-framework-Vision", "requests": "requests",
            "ddgs": "ddgs", "dotenv": "python-dotenv", "edge_tts": "edge-tts"}


# ---------------------------------------------------------------- checks
def check_python() -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"Helping Eyes needs Python 3.11 or later; this is {sys.version.split()[0]}.")
    if sys.platform != "darwin":
        sys.exit("Helping Eyes reads text with Apple Vision, which is part of macOS, so it runs on a Mac only.")


def missing_packages() -> list:
    """Names (as pip knows them) of the required packages that cannot be imported."""
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    return missing


def voice_line() -> str:
    """Which voice will speak, in one line for the start-up message (the local Kokoro voice needs its packages and files)."""
    sys.path.insert(0, APP_DIR)
    try:
        import commands
    except Exception:
        return "Edge voice"
    if not commands.TTS_ENABLED:
        return "the browser's own voice (TTS_ENABLED=0)"
    if commands.kokoro_ready():
        return f"Kokoro ({commands.KOKORO_VOICE}), on this computer; the Edge voice is the backup"
    return "Edge voice (for the better local voice, run:  python run.py --get-voice)"


def get_voice() -> None:
    """Download the Kokoro voice model files once (about 350 MB) and say how to switch the voice on."""
    import urllib.request
    sys.path.insert(0, APP_DIR)
    import commands
    os.makedirs(commands.KOKORO_DIR, exist_ok=True)
    for name, url in commands.KOKORO_FILES.items():
        target = os.path.join(commands.KOKORO_DIR, name)
        if os.path.exists(target):
            print(f"  have      {name}")
            continue
        print(f"  download  {name} ...", flush=True)
        part = target + ".part"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
                total, done = int(response.headers.get("Content-Length") or 0), 0
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total:
                        print(f"\r  download  {name}  {100 * done // total}%", end="", flush=True)
            os.replace(part, target)
            print()
        except Exception as e:
            if os.path.exists(part):
                os.remove(part)
            sys.exit(f"\nCould not download {name}: {e}")
    missing = [m for m in ("kokoro_onnx", "lameenc") if importlib.util.find_spec(m) is None]
    if missing:
        print("  packages  missing: " + ", ".join(missing) + f"\n            install with:  {sys.executable} -m pip install -r requirements.txt")
    else:
        print("  done      the local voice is ready; start the app as usual.")


def read_setting(wanted: str) -> str:
    """A setting the server will use: the environment wins, even when it is empty (the server never overrides a
    variable that is already set), otherwise helping_eyes/.env; '' when there is none."""
    if wanted in os.environ:
        return os.environ[wanted]
    try:
        for line in open(ENV_FILE):
            name, _, value = line.strip().partition("=")
            if name.strip() == wanted:
                return value.strip().strip("'\"")
    except OSError:
        pass
    return ""


def load_settings() -> None:
    """Put helping_eyes/.env into the environment before any app module is imported. assistant.py and commands.py read
    their settings once, when first imported, and voice_line() imports them before the server (which loads .env
    itself) starts; without this the server ran with no key. A variable already set in the environment is kept."""
    try:
        from dotenv import load_dotenv
    except ImportError:              # reported by missing_packages()
        return
    load_dotenv(ENV_FILE, override=False)


def read_key() -> str:
    """The language-model key the server will use ('' when there is none)."""
    return read_setting("LLM_API_KEY")


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def pick_port(wanted: int, tries: int = 20) -> int:
    """The wanted port if it is free, otherwise the next free one."""
    for port in range(wanted, wanted + tries):
        if port_is_free(port):
            return port
    sys.exit(f"No free port between {wanted} and {wanted + tries - 1}. Close other programs, or use --port.")


# ---------------------------------------------------------------- browser
def browser_command(url: str):
    """The command that opens `url` in Chrome (or Edge: the other browser with speech recognition), or None."""
    if sys.platform == "darwin":
        for app in ("Google Chrome", "Microsoft Edge", "Chromium"):
            if os.path.isdir(f"/Applications/{app}.app"):
                return ["open", "-a", app, url]
    elif sys.platform == "win32":
        base = [os.environ.get(v, "") for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
        for sub in (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe"):
            for root in base:
                if root and os.path.isfile(os.path.join(root, sub)):
                    return [os.path.join(root, sub), url]
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge"):
            if shutil.which(name):
                return [name, url]
    return None


def open_browser(url: str) -> str:
    """Open the page. Returns the name of what opened it."""
    command = browser_command(url)
    if command:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "Chrome" if "Edge" not in " ".join(command) and "edge" not in command[0].lower() else "Edge"
    webbrowser.open(url)
    return "your default browser (Chrome or Edge is recommended: speech recognition needs one of them)"


# ---------------------------------------------------------------- share
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def tunnel_url(line: str):
    """The public address in one line of cloudflared's output, or None."""
    match = TUNNEL_URL.search(line)
    return match.group(0) if match else None


def ngrok_domain() -> str:
    """The fixed ngrok address from NGROK_DOMAIN (a bare name like abc-def.ngrok-free.app), or ''."""
    return re.sub(r"^https?://|/.*$", "", read_setting("NGROK_DOMAIN").strip())


def start_ngrok(port: int, domain: str):
    """Open the ngrok tunnel with your fixed free domain. Returns the processes to stop at the end, or None when
    ngrok is missing or cannot start (it says why)."""
    if not shutil.which("ngrok"):
        print("  share     ngrok is not installed. Install it and add your token (one time):")
        print("            brew install ngrok      then      ngrok config add-authtoken <your token from dashboard.ngrok.com>")
        return None
    tunnel = subprocess.Popen(["ngrok", "http", f"--url={domain}", f"127.0.0.1:{port}", "--log=stdout", "--log-format=logfmt"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    problems = []

    def watch():
        for line in tunnel.stdout:
            if "lvl=eror" in line or "lvl=crit" in line or "ERR_NGROK" in line:
                problems.append(line.strip())
    threading.Thread(target=watch, daemon=True).start()
    for _ in range(20):                                  # up to 5 s for it to come up or fail
        time.sleep(0.25)
        if tunnel.poll() is not None:
            break
    if tunnel.poll() is not None:
        reason = problems[-1] if problems else "it stopped at once"
        print(f"  share     ngrok did not start: {reason[-200:]}")
        print("            (check the token with  ngrok config check  and that the domain is yours; old versions need  brew upgrade ngrok)")
        return None
    procs = [tunnel]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        procs.append(subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]))
    print(f"\nPublic address (always the same; works while this computer is awake and this program runs):\n  https://{domain}\n")
    return procs


def start_share(port: int):
    """Open a public tunnel to this server and keep the computer awake while it runs: the fixed ngrok address when
    NGROK_DOMAIN is set, otherwise a free Cloudflare Quick Tunnel (a new address each run).
    Returns the processes to stop at the end, or None when no tunnel could be started."""
    domain = ngrok_domain()
    if domain:
        procs = start_ngrok(port, domain)
        if procs:
            return procs
        print("            falling back to a Cloudflare tunnel, whose address changes every run.")
    else:
        print("  share     for a fixed address, set NGROK_DOMAIN=<your free ngrok domain> in helping_eyes/.env")
    if not shutil.which("cloudflared"):
        print("  share     cloudflared is not installed, so there is no public address. Install it with:")
        print("            brew install cloudflared      (other systems: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)")
        return None
    tunnel = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    procs = [tunnel]
    if sys.platform == "darwin" and shutil.which("caffeinate"):     # no idle sleep while this program runs
        procs.append(subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())]))

    def watch():
        announced = False
        for line in tunnel.stderr:
            url = tunnel_url(line)
            if url and not announced:
                announced = True
                print(f"\nPublic address (works while this computer is awake and this program runs):\n  {url}\n")
    threading.Thread(target=watch, daemon=True).start()
    return procs


def stop_share(procs) -> None:
    for proc in procs or []:
        proc.terminate()


# ---------------------------------------------------------------- ollama (the local Qwen model)
OLLAMA_URL = "http://127.0.0.1:11434"


def ollama_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1):
            return True
    except Exception:
        return False


QWEN_MODEL = os.environ.get("QWEN_3B_MODEL", "qwen2.5:3b-instruct")


def warm_qwen() -> None:
    """Load the Qwen model into memory now, so the first answer does not wait for it. Runs in the background and
    prints one line. Says what to do if Ollama does not have the model."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
        if QWEN_MODEL not in names:
            print(f"  Qwen      the model {QWEN_MODEL} is not downloaded. Run:  ollama pull {QWEN_MODEL}")
            return
        started = time.time()
        request = urllib.request.Request(f"{OLLAMA_URL}/api/generate", method="POST",
                                         data=json.dumps({"model": QWEN_MODEL, "keep_alive": "1h"}).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=120).read()
        print(f"  Qwen      {QWEN_MODEL} is loaded and ready ({time.time() - started:.1f} s)")
    except Exception as e:
        print(f"  Qwen      could not load {QWEN_MODEL}: {e}")


def start_ollama():
    """Start `ollama serve` unless it is already running. Prints one line about it. Returns the process to
    stop at the end, or None when Ollama was already running (left alone) or is not installed."""
    if ollama_running():
        print("  Qwen      Ollama is already running")
        return None
    if not shutil.which("ollama"):
        print("  Qwen      Ollama is not installed, so the Qwen choice is unavailable (https://ollama.com). Gemini still works.")
        return None
    proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "OLLAMA_KEEP_ALIVE": os.environ.get("OLLAMA_KEEP_ALIVE", "1h")})
    for _ in range(40):                                  # up to 10 s
        if ollama_running():
            print("  Qwen      Ollama started")
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    print("  Qwen      Ollama did not start, so the Qwen choice is unavailable. Gemini still works.")
    proc.terminate()
    return None


# ---------------------------------------------------------------- start
def health(port: int):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


def announce_when_ready(port: int, open_page: bool, url: str) -> None:
    """Wait for the server to answer, open the page, then say when the OCR has finished loading."""
    for _ in range(120):
        if health(port):
            break
        time.sleep(0.5)
    else:
        print("The server did not start answering. Check the messages above.")
        return
    if open_page:
        print(f"Opened {open_browser(url)}.")
    for _ in range(240):
        status = health(port) or {}
        if status.get("ocr") == "ready":
            print("Ready: the text reader has loaded. Hold something up to the camera.")
            return
        if str(status.get("ocr", "")).startswith("failed"):
            print(f"The text reader failed to load: {status['ocr']}")
            return
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Helping Eyes on this computer.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", DEFAULT_PORT)), help="port to use (default 7860)")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1: this computer only)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    parser.add_argument("--no-ollama", action="store_true", help="do not start Ollama (the local Qwen model)")
    parser.add_argument("--share", action="store_true", help="also give it a public https address (free Cloudflare tunnel)")
    parser.add_argument("--get-voice", action="store_true", help="download the local Kokoro voice (once, about 350 MB), then stop")
    args = parser.parse_args()
    load_settings()

    if args.get_voice:
        return get_voice()

    check_python()
    missing = missing_packages()
    if missing:
        sys.exit("These packages are missing: " + ", ".join(missing) +
                 f"\nInstall everything with:  {sys.executable} -m pip install -r requirements.txt")

    port = pick_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is in use, using {port} instead.")
    url = f"http://localhost:{port}"

    print("Helping Eyes")
    print(f"  address   {url}" + ("   (and the other addresses of this computer)" if args.host == "0.0.0.0" else ""))
    if read_key():
        print("  AI model  key found")
    else:
        print(f"  AI model  NO KEY. Add  LLM_API_KEY=...  to {os.path.relpath(ENV_FILE, ROOT)}  to enable answers.")
        print("            (read everything, expiry, page numbers and the live guidance work without it)")
    print(f"  voice     {voice_line()}")
    ollama = None if args.no_ollama else start_ollama()
    if not args.no_ollama and ollama_running():
        threading.Thread(target=warm_qwen, daemon=True).start()          # the model loads while the rest starts
    if args.host == "0.0.0.0":
        print("  note      other devices need https for the camera; over plain http it works only on this computer")
    print("  stop      press Ctrl+C\n")

    threading.Thread(target=announce_when_ready, args=(port, not args.no_browser, url), daemon=True).start()

    shared = start_share(port) if args.share else None
    import uvicorn
    try:
        uvicorn.run("server:app", app_dir=APP_DIR, host=args.host, port=port, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        stop_share(shared)
        if ollama:
            ollama.terminate()
    print("\nStopped.")


if __name__ == "__main__":
    main()
