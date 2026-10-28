"""
Tests for the page's listening (web/app.js, "speech in"). The real app runs in headless Chrome with a fake
speech-recognition engine and a fake voice, so every path can be driven: hearing a sentence, silence,
the speech service being unreachable, a missing or blocked microphone. Skipped when Chrome is not installed.

    python tests/test_web_mic.py        (or: python -m pytest tests/test_web_mic.py)
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CHROME = next((c for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "google-chrome-stable",
                           "chromium", "chromium-browser") if os.path.exists(c) or shutil.which(c)), None)

# Installed before the page's own script runs: a speech engine we control, a voice that finishes at once,
# and a record of what the page asks the server.
FAKES = """
window.__recs = []; window.__spoken = []; window.__asked = [];
class FakeRec {
  constructor() { window.__recs.push(this); }
  start() { this.started = true; }
  abort() { this.finish(); }
  stop() { this.finish(); }
  finish() { if (!this.ended) { this.ended = true; if (this.onend) this.onend(); } }
}
window.SpeechRecognition = window.webkitSpeechRecognition = FakeRec;
const fakeSynth = { speak(u) { window.__spoken.push(u.text + ' [browser]'); setTimeout(() => { if (u.onend) u.onend(); }, 5); }, cancel() {} };
Object.defineProperty(window, "speechSynthesis", { value: fakeSynth });
window.SpeechSynthesisUtterance = function (text) { this.text = text; };
// The server voice (/api/tts) returns audio that an <audio> element plays. Fake both: record the text, "play" for 5 ms.
// window.__ttsDown = true makes /api/tts fail, so the browser voice (fakeSynth) takes over.
window.__media = [];                                    // every camera / microphone request the page makes
const realGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = (constraints) => {
  window.__media.push(JSON.stringify(constraints));
  if (window.__noMic && constraints.audio) return Promise.reject(new DOMException("none", "NotFoundError"));     // no microphone
  return realGetUserMedia(constraints);
};
// A first visit: the browser has not been asked about the microphone yet (the test browser reports "granted" for all)
const realQuery = navigator.permissions.query.bind(navigator.permissions);
navigator.permissions.query = (descriptor) => descriptor.name === "microphone" ? Promise.resolve({ state: "prompt" }) : realQuery(descriptor);
const realFetch = window.fetch;
window.fetch = (url, opts) => {
  if (String(url).includes("/api/ask")) window.__asked.push(JSON.parse(opts.body).question);
  if (String(url).includes("/api/tts")) {
    if (window.__ttsDown) return Promise.resolve(new Response("no", { status: 502 }));
    if (window.__ttsHangFirst) {                        // one request that never answers (until the page gives up on it)
      window.__ttsHangFirst = false;
      return new Promise((resolve, reject) => opts.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));
    }
    window.__spoken.push(JSON.parse(opts.body).text);
    return Promise.resolve(new Response(new Blob(["x"], { type: "audio/mpeg" })));
  }
  if (String(url).includes("/api/health")) return Promise.resolve(new Response(JSON.stringify({ tts: true })));
  if (String(url).includes("/api/capture")) {                  // a photo read by the server: window.__captureOk = false means "no text found"
    const ok = window.__captureOk !== false;
    return Promise.resolve(new Response(JSON.stringify({ ok, text: ok ? "hello world" : "", message: ok ? "Got it." : "I could not find any text.", seconds: 0.1, label: "" })));
  }
  if (String(url).includes("/api/quality")) {
    return Promise.resolve(new Response(JSON.stringify({ quality: "sharp 200", problems: [], hint: "", lines: [], quads: [], box: null })));
  }
  return realFetch(url, opts);
};
window.Audio = class {
  constructor() { this.duration = 0; }
  play() {
    if (window.__soundBlocked) return Promise.reject(new DOMException("blocked", "NotAllowedError"));    // a phone before the first tap
    setTimeout(() => { if (this.onended) this.onended(); }, 5);
    return Promise.resolve();
  }
  pause() {}
};
window.__say = (words, isFinal) => {                       // the speech engine "hears" something
  const rec = window.__recs[window.__recs.length - 1];
  const result = Object.assign([{ transcript: words }], { isFinal });
  rec.onresult({ results: [result] });
  if (isFinal) rec.finish();                               // a real engine ends after a finished sentence
};
window.__fail = (error) => { const rec = window.__recs[window.__recs.length - 1]; rec.onerror({ error }); rec.finish(); };
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Page:
    """The app open in headless Chrome, driven over the DevTools protocol."""

    def __init__(self, app_port):
        from websockets.sync.client import connect
        self.dir = tempfile.mkdtemp()
        blank = os.path.join(self.dir, "blank.mjpeg")             # a camera that sees nothing (no text, so no guidance)
        import cv2
        import numpy as np
        open(blank, "wb").write(cv2.imencode(".jpg", np.full((360, 640, 3), 255, np.uint8))[1].tobytes() * 60)
        self.chrome_port = free_port()
        self.proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={self.chrome_port}",
                                      f"--user-data-dir={self.dir}/profile", "--use-fake-device-for-media-stream",
                                      "--use-fake-ui-for-media-stream", f"--use-file-for-fake-video-capture={blank}",
                                      "--window-size=1000,800", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.chrome_port}/json"))
                break
            except Exception:
                time.sleep(0.2)
        self.ws = connect([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], max_size=None)
        self.n = 0
        self.send("Page.enable")
        self.send("Page.addScriptToEvaluateOnNewDocument", source=FAKES)
        self.send("Page.navigate", url=f"http://localhost:{app_port}/")
        self.wait("typeof listening !== 'undefined' && !!document.getElementById('mic')")

    def send(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.n:
                return message

    def js(self, expression):
        result = self.send("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)["result"]
        if "exceptionDetails" in result:
            raise AssertionError(result["exceptionDetails"]["exception"].get("description", "page error"))
        return result["result"].get("value")

    def wait(self, expression, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.js(expression):
                    return
            except AssertionError:
                pass
            time.sleep(0.15)
        raise AssertionError("timed out waiting for: " + expression + " | status: " + str(self.js("document.getElementById('status').textContent")))

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            shutil.rmtree(self.dir, ignore_errors=True)


def start_app():
    port = free_port()
    env = {**os.environ, "LLM_API_KEY": ""}
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "run.py"), "--no-browser", "--no-ollama", "--port", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2)
            return proc, port
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise AssertionError("the app did not start")


def with_page(scenario):
    proc, port = start_app()
    page = Page(port)
    try:
        scenario(page)
    finally:
        page.close()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


STATUS = "document.getElementById('status').textContent"
LABEL = "document.getElementById('mic').firstChild.textContent.trim()"
PRESSED = "document.getElementById('mic').getAttribute('aria-pressed')"
READY = "window.__recs.length > 0 && !window.__recs[window.__recs.length - 1].ended"       # a listener is running


def press_listen(page):
    page.js("document.getElementById('mic').click()")
    page.wait("listening === true")


# ---------------------------------------------------------------- the scenarios
def scenario_hearing(page):
    press_listen(page)
    page.wait(READY)
    assert page.js(LABEL) == "Listen" and page.js(PRESSED) == "true"     # one fixed name; the pressed state says it is open
    page.js("window.__say('how many tablets', false)")                 # still speaking: shown as it is heard
    assert page.js("document.getElementById('you-said').textContent") == "You said: how many tablets…"
    assert page.js("window.__asked.length") == 0                       # nothing is sent until the sentence is complete
    page.js("window.__say('stop', true)")                              # the finished sentence
    page.wait("window.__asked.length === 1")
    assert page.js("window.__asked[0]") == "stop" and page.js(STATUS).startswith(("Heard: stop", "Answered", "One moment"))
    page.wait("window.__recs.length >= 2 && !window.__recs[window.__recs.length - 1].ended")   # and it listens again


def scenario_button_shows_when_the_mic_is_really_on(page):
    press_listen(page)
    page.wait(READY)
    page.js("speechQueue.push({ text: 'talking' })")                   # the app starts talking
    page.wait(f"{PRESSED} === 'false'")                                 # the button goes off by itself
    assert page.js(LABEL) == "Listen" and page.js("listening") is True   # listening is still switched on
    page.js("speechQueue.length = 0; lastSpeechActivity = performance.now()")   # the app finishes talking
    page.wait(f"{PRESSED} === 'true'")                                  # and the button comes back on by itself
    page.js("speechQueue.push({ text: 'talking again' })")
    page.wait(f"{PRESSED} === 'false'")
    page.js("document.getElementById('mic').click()")                   # pressed while it talks: talking stops, mic opens
    assert page.js("speechQueue.length") == 0 and page.js("listening") is True
    page.wait(f"{PRESSED} === 'true'")


PIXEL = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


def scenario_camera_is_off_while_a_photo_is_shown(page):
    page.wait("cameraReady === true && !!video.srcObject")
    stream = "video.srcObject"
    page.js(f"window.__track = {stream}.getTracks()[0]")
    page.js(f"showStill('{PIXEL}')")                                    # an uploaded photo is shown
    assert page.js("window.__track.readyState") == "ended"              # the camera really stopped, not just hidden
    assert page.js("video.srcObject") is None and page.js("cameraReady") is False
    page.js("backToCamera()")                                           # back to the camera
    page.wait("cameraReady === true && !!video.srcObject")
    assert page.js("video.srcObject.getTracks()[0].readyState") == "live"


def scenario_silence_keeps_listening(page):
    press_listen(page)
    page.wait(READY)
    before = page.js("window.__recs.length")
    page.js("window.__fail('no-speech')")                              # nobody spoke
    page.wait(f"window.__recs.length > {before}")                      # a new listener starts
    assert page.js("listening") is True and page.js(PRESSED) == "true"
    page.js("window.__fail('aborted')")
    assert page.js("listening") is True


def scenario_speech_service_unreachable(page):
    press_listen(page)
    page.wait(READY)
    page.js("window.__fail('network')")
    page.wait("listening === false")
    assert "speech service" in page.js(STATUS)
    assert page.js(LABEL) == "Listen" and page.js(PRESSED) == "false"
    assert any("speech service" in t for t in page.js("window.__spoken"))       # also spoken aloud


def scenario_no_microphone(page):
    page.js("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('none', 'NotFoundError'); }")
    page.js("document.getElementById('mic').click()")
    page.wait("document.getElementById('status').textContent.includes('No microphone')")
    assert page.js("listening") is False and page.js(PRESSED) == "false"


def scenario_microphone_blocked(page):
    page.js("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('no', 'NotAllowedError'); }")
    page.js("document.getElementById('mic').click()")
    page.wait("document.getElementById('status').textContent.includes('Microphone access was denied')")
    assert page.js("listening") is False


def scenario_repeated_unknown_errors_give_up(page):
    press_listen(page)
    for _ in range(3):
        page.wait(READY)
        page.js("window.__fail('something-odd')")
        time.sleep(0.2)
    page.wait("listening === false")
    assert "keeps failing" in page.js(STATUS)


def scenario_button_turns_listening_off(page):
    press_listen(page)
    page.wait(READY)
    page.js("document.getElementById('mic').click()")
    page.wait("listening === false")
    assert page.js(LABEL) == "Listen" and page.js("window.__recs[window.__recs.length - 1].ended") is True
    time.sleep(1)
    n = page.js("window.__recs.length")
    time.sleep(1)
    assert page.js("window.__recs.length") == n                        # and it stays off


def scenario_browser_voice_takes_over_when_the_server_voice_fails(page):
    page.js("window.__ttsDown = true")
    page.js("document.getElementById('question').value = 'next'; document.getElementById('ask-form').requestSubmit()")
    page.wait("window.__spoken.some((t) => t.endsWith('[browser]'))")  # still spoken, by the browser's voice


def scenario_camera_and_microphone_are_asked_for_as_soon_as_the_page_opens(page):
    # ONE request for both (a second request could freeze the camera on a phone)
    page.wait("window.__media.some((c) => c.includes('\"video\"') && c.includes('\"audio\":true'))")
    page.wait("cameraReady")
    # asking only: the microphone is released at once and listening is still off until Listen is pressed
    assert page.js("document.querySelector('video').srcObject.getAudioTracks().length") == 0
    assert page.js("document.getElementById('mic').getAttribute('aria-pressed')") == "false"


def scenario_the_photo_taken_stays_on_screen_and_the_camera_is_off_until_new_item(page):
    page.wait("cameraReady")
    page.js("document.getElementById('capture').click()")
    page.wait("uploadView === true && capturedStill === true")
    assert page.js("document.getElementById('video').hidden") is True                      # the picture taken replaces the video ...
    assert page.js("document.getElementById('still').hidden") is False
    assert page.js("document.getElementById('still').src.startsWith('blob:')") is True
    assert page.js("cameraReady") is False and page.js("document.getElementById('video').srcObject") is None     # ... and the camera is off
    page.wait("document.getElementById('captured-text').textContent === 'hello world'")    # and the photo was read
    import time
    time.sleep(1.5)                                                                         # it stays: nothing brings the camera back by itself
    assert page.js("uploadView") is True and page.js("cameraReady") is False
    page.js("document.getElementById('newitem').click()")                                   # New item
    page.wait("uploadView === false && cameraReady === true")                              # back to the live camera
    assert page.js("document.getElementById('still').hidden") is True


def scenario_a_photo_with_no_text_brings_the_camera_back_at_once(page):
    page.wait("cameraReady")
    page.js("window.__captureOk = false")
    page.js("document.getElementById('capture').click()")
    page.wait("uploadView === false && cameraReady === true")                              # nothing read: try again with the camera
    assert page.js("document.getElementById('still').hidden") is True


def scenario_the_shutter_button_takes_the_photo_now(page):
    page.wait("cameraReady")
    page.wait("!document.getElementById('shutter').hidden")                  # shown while the live camera is in use
    page.js("window.__captures = 0; document.getElementById('capture').addEventListener('click', () => window.__captures++)")
    page.js("document.getElementById('shutter').click()")
    assert page.js("window.__captures") == 1                                 # it does what Capture now does
    assert page.js("document.querySelector('.camera').classList.contains('flash')")        # and flashes, so you see it was taken
    page.js("showStill('data:image/gif;base64,R0lGODlhAQABAAAAACw=')")      # a photo is shown in place of the camera
    page.wait("document.getElementById('shutter').hidden")                  # the shutter is gone then


def scenario_the_camera_still_starts_when_there_is_no_microphone(page):
    page.js("window.__noMic = true; window.__media.length = 0; stopCamera()")
    page.js("startCamera(true)")
    page.wait("cameraReady")
    calls = page.js("window.__media")
    assert '"audio":true' in calls[0] and '"audio":false' in calls[1]          # asked for both, then the camera alone


def scenario_a_phone_that_blocks_sound_does_not_freeze_the_camera_or_the_microphone(page):
    page.js("window.__soundBlocked = true")
    page.js("document.getElementById('question').value = 'next'; document.getElementById('ask-form').requestSubmit()")
    page.wait("soundBlocked !== null")                         # the phone blocked the sound: the sentence waits for a tap
    assert page.js("isSpeaking()") is False                    # waiting for a tap is not speaking: frames and listening go on
    page.js("window.__soundBlocked = false; document.dispatchEvent(new Event('click'))")      # the first tap
    page.wait("soundBlocked === null")


def scenario_a_hung_voice_request_is_answered_by_a_second_one(page):
    page.js("window.__ttsHangFirst = true")
    page.js("document.getElementById('question').value = 'next'; document.getElementById('ask-form').requestSubmit()")
    # the first request never answers: after a few seconds a second one is sent, and the server voice (not the
    # browser's) speaks, long before the page's own give-up time
    page.wait("window.__spoken.length > 0", timeout=8)
    assert not any(t.endswith("[browser]") for t in page.js("window.__spoken"))


def run_scenario(scenario):
    if CHROME is None:
        print(f"SKIP  {scenario.__name__}: Chrome is not installed")
        return
    with_page(scenario)


def test_the_camera_and_the_microphone_are_asked_for_as_soon_as_the_page_opens():
    run_scenario(scenario_camera_and_microphone_are_asked_for_as_soon_as_the_page_opens)


def test_the_photo_taken_stays_on_screen_and_the_camera_is_off_until_new_item():
    run_scenario(scenario_the_photo_taken_stays_on_screen_and_the_camera_is_off_until_new_item)


def test_a_photo_with_no_text_brings_the_camera_back_at_once():
    run_scenario(scenario_a_photo_with_no_text_brings_the_camera_back_at_once)


def test_the_shutter_button_takes_the_photo_now():
    run_scenario(scenario_the_shutter_button_takes_the_photo_now)


def test_the_camera_still_starts_when_there_is_no_microphone():
    run_scenario(scenario_the_camera_still_starts_when_there_is_no_microphone)


def test_a_phone_that_blocks_sound_does_not_freeze_the_camera_or_the_microphone():
    run_scenario(scenario_a_phone_that_blocks_sound_does_not_freeze_the_camera_or_the_microphone)


def test_a_voice_request_that_never_answers_is_replaced_by_a_second_one():
    run_scenario(scenario_a_hung_voice_request_is_answered_by_a_second_one)


def test_it_shows_words_as_they_are_heard_then_sends_the_finished_sentence():
    run_scenario(scenario_hearing)


def test_the_listen_button_is_off_while_the_app_talks_and_pressing_it_stops_the_talking():
    run_scenario(scenario_button_shows_when_the_mic_is_really_on)


def test_the_camera_is_switched_off_while_an_uploaded_photo_is_shown():
    run_scenario(scenario_camera_is_off_while_a_photo_is_shown)


def test_silence_just_listens_again():
    run_scenario(scenario_silence_keeps_listening)


def test_an_unreachable_speech_service_is_explained_and_listening_stops():
    run_scenario(scenario_speech_service_unreachable)


def test_a_missing_microphone_is_explained():
    run_scenario(scenario_no_microphone)


def test_a_blocked_microphone_is_explained():
    run_scenario(scenario_microphone_blocked)


def test_repeated_unknown_errors_stop_listening_with_a_message():
    run_scenario(scenario_repeated_unknown_errors_give_up)


def test_the_button_turns_listening_off_and_it_stays_off():
    run_scenario(scenario_button_turns_listening_off)


def test_the_browser_voice_takes_over_when_the_server_voice_fails():
    run_scenario(scenario_browser_voice_takes_over_when_the_server_voice_fails)


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
