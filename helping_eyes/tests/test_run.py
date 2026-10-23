"""
Tests for run.py, the script that runs the whole app on this computer. The last tests really start it
in a subprocess (no browser, no key, no model needed):

    python tests/test_run.py        (or: python -m pytest tests/test_run.py)
"""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)

import run


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- the helpers
def test_every_required_package_is_installed():
    assert run.missing_packages() == []


def test_a_missing_package_is_reported_by_its_pip_name():
    original = dict(run.REQUIRED)
    run.REQUIRED["not_a_real_module_xyz"] = "not-a-real-package"
    try:
        assert run.missing_packages() == ["not-a-real-package"]
    finally:
        run.REQUIRED.clear()
        run.REQUIRED.update(original)


def test_port_is_free_and_pick_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not run.port_is_free(busy)
        assert run.pick_port(busy) != busy                  # the next free one
    quiet = free_port()
    assert run.port_is_free(quiet) and run.pick_port(quiet) == quiet


def test_the_server_gets_the_key_even_though_the_voice_line_imports_the_app_first():
    # run.py prints the voice line (which imports commands and so assistant) before the server loads .env; the key
    # must still reach assistant.LLM_API_KEY. A fresh interpreter, so no module is already imported.
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("LLM_API_KEY=abc123\n")
    code = ("import run; run.ENV_FILE = %r; run.load_settings(); run.voice_line(); import assistant; "
            "print(assistant.LLM_API_KEY)" % f.name)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
    try:
        out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=60)
        assert out.stdout.strip().splitlines()[-1] == "abc123", out.stderr[-500:]
        out = subprocess.run([sys.executable, "-c", code], cwd=root, env={**env, "LLM_API_KEY": "from-environment"},
                             capture_output=True, text=True, timeout=60)
        assert out.stdout.strip().splitlines()[-1] == "from-environment"      # the environment still wins
    finally:
        os.unlink(f.name)


def test_key_is_read_from_the_environment_or_the_env_file():
    saved_env, saved_file = os.environ.pop("LLM_API_KEY", None), run.ENV_FILE
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# a comment\nOTHER=1\nLLM_API_KEY = 'abc123'\n")
        run.ENV_FILE = f.name
        assert run.read_key() == "abc123"
        os.environ["LLM_API_KEY"] = "from-environment"
        assert run.read_key() == "from-environment"          # the environment wins
        os.environ["LLM_API_KEY"] = ""
        assert run.read_key() == ""                           # even when empty: the server would not override it
        del os.environ["LLM_API_KEY"]
        run.ENV_FILE = f.name + ".missing"
        assert run.read_key() == ""                           # no file, no key
    finally:
        run.ENV_FILE = saved_file
        os.environ.pop("LLM_API_KEY", None)
        if saved_env is not None:
            os.environ["LLM_API_KEY"] = saved_env
        os.unlink(f.name)


def test_browser_command_opens_the_given_address():
    command = run.browser_command("http://localhost:7860")
    if command is not None:                                   # a computer with Chrome, Edge or Chromium
        assert command[-1] == "http://localhost:7860"


def test_open_browser_falls_back_to_the_default_browser():
    opened, original = [], (run.browser_command, run.webbrowser.open)
    run.browser_command = lambda url: None
    run.webbrowser.open = lambda url: opened.append(url)
    try:
        assert "default browser" in run.open_browser("http://localhost:1")
    finally:
        run.browser_command, run.webbrowser.open = original
    assert opened == ["http://localhost:1"]


def test_tunnel_url_is_found_in_cloudflared_output():
    line = "2026-10-08T10:00:00Z INF |  https://quiet-blue-fox-12.trycloudflare.com  |"
    assert run.tunnel_url(line) == "https://quiet-blue-fox-12.trycloudflare.com"
    assert run.tunnel_url("INF Requesting new quick Tunnel on trycloudflare.com...") is None


def test_the_fixed_ngrok_address_is_read_from_the_settings_and_cleaned():
    saved_env, saved_file = os.environ.pop("NGROK_DOMAIN", None), run.ENV_FILE
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("NGROK_DOMAIN = 'https://quiet-fox.ngrok-free.app/'\n")
        run.ENV_FILE = f.name
        assert run.ngrok_domain() == "quiet-fox.ngrok-free.app"       # a pasted https:// and a trailing / do no harm
        os.environ["NGROK_DOMAIN"] = ""
        assert run.ngrok_domain() == ""                              # no domain: the Cloudflare tunnel is used
    finally:
        run.ENV_FILE = saved_file
        os.environ.pop("NGROK_DOMAIN", None)
        if saved_env is not None:
            os.environ["NGROK_DOMAIN"] = saved_env
        os.unlink(f.name)


def test_a_missing_ngrok_says_how_to_install_it_and_falls_back():
    original, domain = run.shutil.which, run.ngrok_domain
    run.shutil.which, run.ngrok_domain = (lambda name: None), (lambda: "quiet-fox.ngrok-free.app")
    try:
        assert run.start_ngrok(7860, "quiet-fox.ngrok-free.app") is None
        assert run.start_share(7860) is None                          # no ngrok and no cloudflared: nothing starts, no crash
    finally:
        run.shutil.which, run.ngrok_domain = original, domain


def test_share_without_cloudflared_says_how_to_install_it(capsys=None):
    original = run.shutil.which
    run.shutil.which = lambda name: None
    try:
        assert run.start_share(7860) is None
    finally:
        run.shutil.which = original
    run.stop_share(None)                                      # nothing to stop is fine


# ---------------------------------------------------------------- the real thing
def start(*args):
    env = {**os.environ, "LLM_API_KEY": ""}                   # no key, whatever is in the developer's own .env
    return subprocess.Popen([sys.executable, os.path.join(ROOT, "run.py"), "--no-browser", "--no-ollama", *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for(port, timeout=90, ready=False):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                status = json.load(r)
            if not ready or status["ocr"] == "ready":
                return status
        except Exception:
            pass
        time.sleep(0.5)
    raise AssertionError(f"the app did not come up on port {port}")


def stop(proc):
    proc.send_signal(signal.SIGINT)
    try:
        out, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out


def test_it_starts_serves_the_page_and_stops_on_ctrl_c():
    port = free_port()
    proc = start("--port", str(port))
    try:
        status = wait_for(port, ready=True)
        assert status["status"] == "ok" and status["llm_configured"] is False and status["ocr"] == "ready"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert b"Helping Eyes" in r.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/static/app.js", timeout=5) as r:
            assert r.status == 200
        time.sleep(1.5)                                       # run.py checks every 0.5 s before it prints "Ready"
    finally:
        out = stop(proc)
    assert f"http://localhost:{port}" in out
    assert "NO KEY" in out and "LLM_API_KEY" in out           # tells you what to do about the missing key
    assert "Ready: the text reader has loaded" in out
    assert "Stopped." in out and proc.returncode == 0         # a clean stop


def test_it_moves_to_the_next_port_when_the_wanted_one_is_taken():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        wanted = taken.getsockname()[1]
        proc = start("--port", str(wanted))
        try:
            deadline, out_port = time.time() + 90, None
            while time.time() < deadline and out_port is None:
                for candidate in range(wanted + 1, wanted + 6):
                    try:
                        wait_for(candidate, timeout=1)
                        out_port = candidate
                        break
                    except AssertionError:
                        pass
            assert out_port is not None, "the app never appeared on a neighbouring port"
        finally:
            out = stop(proc)
    assert f"Port {wanted} is in use, using {out_port} instead." in out


def test_the_local_voice_is_optional_and_reported_in_one_line():
    import run
    assert "kokoro_onnx" not in run.REQUIRED and "lameenc" not in run.REQUIRED          # a missing one must not stop the start
    assert "voice" not in run.missing_packages()
    line = run.voice_line()
    assert isinstance(line, str) and ("Kokoro" in line or "Edge" in line or "browser" in line)


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
