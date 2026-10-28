// Helping Eyes: the browser side.
// The browser owns the camera, the microphone and the speaker and draws the overlay.
// Every decision (guidance hints, when to capture, page turns, what a request means,
// the answers) is made by the server (server.py, vision.py, commands.py).

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), grab = $("canvas"), still = $("still");
const statusEl = $("status"), answerEl = $("answer");

const sid = newId();          // identifies this browser to the server (its item, conversation, book)
let mode = "normal";          // "normal" (one item) or "book" (page by page)
let thinking = false;         // an answer is being generated
let controller = null;        // aborts the current answer stream
let lastSentences = [];

function newId() {
  const bytes = new Uint8Array(16);
  (window.crypto || window.msCrypto).getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
function setStatus(text) { statusEl.textContent = text; statusEl.title = text; }      // the title: a short screen cuts the line
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ================================================================ screen reader (Settings)
// Two ways to hear the app. Normally the app speaks everything important with its own voice, and nothing on the page is
// a live region, so a screen reader does not repeat it. With "My screen reader reads everything" switched on, the app's
// voice stays quiet and everything it would say goes to the screen reader (or a braille display) through #sr-live.
let srMode = false;
let srBusyUntil = 0;          // screen-reader mode: when the screen reader has probably finished what it was given
const SR_MS_PER_CHAR = 60;    // a screen reader at its usual speed, about 15 characters a second

function announce(text) {
  const box = $("sr-live");
  const p = document.createElement("p");
  p.textContent = text;
  box.appendChild(p);                       // a new node in the live region is read out
  while (box.childElementCount > 6) box.firstElementChild.remove();
}

// In screen-reader mode say() ends here: the text is handed to the screen reader, and the microphone and the camera
// guidance wait until it has probably been read (it is heard through the same speakers)
function srSay(text, start) {
  announce(text);
  if (start !== null) speechPos = start + text.length;           // book mode: the piece counts as read
  srBusyUntil = Math.max(srBusyUntil, performance.now()) + 400 + text.length * SR_MS_PER_CHAR;
  lastSpeechActivity = performance.now();
  if (recognizer) recognizer.abort();
}

// ================================================================ speech out
const synth = window.speechSynthesis || null;
const speechQueue = [];       // {text, el, start}: waiting to be spoken
let speakingNow = null;
let speechPos = null;         // book mode: character of the page being spoken right now
let lastSpeechActivity = 0;   // performance.now() when speech last started or ended (echo guard)

let player = null;            // the <audio> element that is playing now
let serverVoice = true;       // false after a failure, until serverVoiceRetry
let serverVoiceRetry = 0;
let serverVoiceFailures = 0;  // sentences in a row that the server voice could not speak
const PREFETCH = 3;           // sentences fetched ahead, so the voice does not pause between them
const START_MS = 10000;       // the server voice must start within this long ...
const STALL_MS = 6000;        // ... and may then be silent for at most this long before the sentence is skipped

// Does the server have a voice? Known before the first sentence is fetched
const voiceKnown = fetch("/api/health").then((r) => r.json()).then((h) => {
  if (h.tts === false) { serverVoice = false; serverVoiceRetry = Infinity; }
  // The local voice is computer work, not a network call: it takes longer, and a second request would only double the work
  if (h.tts_engine === "kokoro") { hedgeMs = 15000; giveUpMs = 30000; }
}).catch(() => {});
const serverVoiceUsable = () => serverVoice || performance.now() > serverVoiceRetry;

// A sentence that waits for the first tap (the phone blocks sound until then) is not speaking: the camera guidance
// and the microphone must not wait for a tap
const isSpeaking = () => ((!!speakingNow || speechQueue.length > 0) && !soundBlocked) || performance.now() < srBusyUntil;

// A number above 9999 is nearly always a serial, batch or phone number: say its digits one by one,
// never "twelve thousand three hundred...". Comma-grouped forms (12,345,678) count too.
const LONG_NUMBER = /(?<![\d,])(?:\d{1,3}(?:,\d{3})+|\d{5,})(?!\d)/g;
function spokenForm(text) {
  return text.replace(LONG_NUMBER, (m) => {
    const digits = m.replace(/,/g, "");
    return Number(digits) > 9999 ? digits.split("").join(" ") : m;
  });
}

// start: for a piece of a book page, its offset in the page text (enables the reading position)
function say(text, { el = null, start = null, optional = false } = {}) {
  if (!text || !text.trim()) return;
  if (start !== null) start += text.length - text.trimStart().length;
  if (srMode) return srSay(text.trim(), start);
  speechQueue.push({ text: text.trim(), el, start, audio: null, optional });
  lastSpeechActivity = performance.now();
  if (recognizer) recognizer.abort();      // the microphone must not hear the app
  prefetch();
  if (!speakingNow) speakNext();
}

function speakNext() {
  const item = speechQueue.shift();
  if (!item) {
    speakingNow = null;
    speechPos = null;
    lastSpeechActivity = performance.now();
    return;
  }
  speakingNow = item;
  prefetch();
  const finished = () => {
    if (speakingNow !== item) return;      // a stale event from a cancelled sentence
    if (item.el) { clearHighlight(); if (item.el.querySelector("mark")) item.el.textContent = item.text; }
    if (item.start !== null) speechPos = item.start + item.text.length;   // voices without word events
    if (item.url) URL.revokeObjectURL(item.url);
    player = null;
    speakNext();
  };
  fetchAudio(item).then((url) => {
    if (speakingNow !== item) { if (url) URL.revokeObjectURL(url); return; }      // stopped while waiting
    if (url) playAudio(item, url, finished);
    else speakWithBrowser(item, finished);
  });
}

// One request for the audio of a text. A request now and then never answers, so a second one is started when the
// first is slow (hedgeMs) or has failed, and whichever delivers first wins; everything is given up after giveUpMs.
let hedgeMs = 3000, giveUpMs = 12000;
function fetchVoiceBlob(text) {
  const controllers = [];
  const request = () => {
    const c = new AbortController();
    controllers.push(c);
    return fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" },
                               body: JSON.stringify({ text }), signal: c.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error("HTTP " + res.status);
        const blob = await res.blob();
        if (!blob.size) throw new Error("empty audio");
        return blob;
      });
  };
  const first = request();
  const second = new Promise((resolve, reject) => {
    const hedge = setTimeout(() => request().then(resolve, reject), hedgeMs);
    first.then(() => clearTimeout(hedge), () => { clearTimeout(hedge); request().then(resolve, reject); });
  });
  const giveUp = setTimeout(() => controllers.forEach((c) => c.abort()), giveUpMs);
  return Promise.any([first, second]).finally(() => {
    clearTimeout(giveUp);
    controllers.forEach((c) => c.abort());           // the loser, if there is one
  });
}

// The server's voice: fetch the audio of one sentence; resolves to an object URL, or null when it cannot
function fetchAudio(item) {
  if (item.audio) return item.audio;
  item.audio = (async () => {
    await voiceKnown;
    if (!serverVoiceUsable()) return null;
    const fetchStart = performance.now();
    try {
      const blob = await fetchVoiceBlob(item.text);
      console.info(`Voice: ${item.text.length} characters ready in ${Math.round(performance.now() - fetchStart)} ms`);
      serverVoiceFailures = 0;
      return URL.createObjectURL(blob);
    } catch (err) {
      console.warn("Server voice failed:", err && err.errors ? err.errors.map((e) => e.message).join("; ") : err.message);
    }
    // This sentence is spoken by the browser's voice, which is the weaker one. One bad sentence must not move
    // the rest of the page onto it, so the server voice is given up only after several in a row fail.
    if (++serverVoiceFailures >= 3) {
      serverVoice = false;
      serverVoiceRetry = performance.now() + 20000;      // browser voice for a short while, then try again
    }
    return null;
  })();
  return item.audio;
}

function prefetch() {
  if (serverVoiceUsable()) speechQueue.slice(0, PREFETCH).forEach(fetchAudio);
}

// Phones let a page play sound only after a tap, and only on an audio element that a tap has started. So one
// element is created and unlocked at the first tap or key press, and reused for every sentence (a new element
// per sentence is blocked on phones, which then fall back to the browser's own weak voice).
const SILENT_WAV = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";
const MOBILE = /Android|iPhone|iPad|Mobi/i.test(navigator.userAgent);
let voiceEl = null, audioUnlocked = false;
let soundBlocked = null;      // a sentence waiting for the first tap, because the phone blocked its sound
// Only some events count as a tap for the phone (a touch's "pointerdown" does not, "touchend" and "click" do),
// so the unlock is tried again at every one of them until the phone has really allowed sound.
function unlockAudio() {
  if (audioUnlocked || player) return;
  try {
    if (!voiceEl) voiceEl = new Audio();
    voiceEl.src = SILENT_WAV;
    const started = voiceEl.play();
    if (started && started.then) started.then(soundUnlocked, () => {});
    else soundUnlocked();
    if (synth && MOBILE) {       // the browser voice needs the same unlock
      const u = new SpeechSynthesisUtterance(" "); u.volume = 0; synth.speak(u);
    }
  } catch (err) { /* nothing to unlock */ }
}
function soundUnlocked() {
  audioUnlocked = true;
  if (soundBlocked) { const play = soundBlocked; soundBlocked = null; play(); }
}
["pointerup", "touchend", "keydown", "click"].forEach((type) => document.addEventListener(type, unlockAudio, { passive: true }));

function playAudio(item, url, finished) {
  item.url = url;
  if (!voiceEl) voiceEl = new Audio();
  const audio = voiceEl;
  player = audio;
  audio.src = url;
  let lastIndex = -1;
  // Never freeze the app if the audio stalls: give up when the sound has not moved for a few seconds. The
  // timer belongs to this sentence and is cleared when speech is stopped (item.stop), because the audio
  // element is shared: a leftover timer would otherwise pause the NEXT sentence in the middle.
  let watchdog = null, kicks = 0;
  const arm = (ms) => { clearTimeout(watchdog); watchdog = setTimeout(() => stalled(), ms); };
  // The sound stopped moving without ending. Nudge it twice; if it stays stuck, give up on this sound but do
  // not skip what is left: the browser's voice says the rest of the piece from where the sound got to.
  const stalled = () => {
    if (speakingNow !== item) return;
    if (audio.ended) return done();
    const at = audio.duration ? audio.currentTime / audio.duration : 0;
    console.warn(`Voice stalled at ${Math.round(at * 100)}% of a ${item.text.length}-character piece ` +
                 `(readyState ${audio.readyState}, paused ${audio.paused}, tab hidden ${document.hidden})`);
    if (at > 0.97) return done();
    if (kicks++ < 2) { audio.play().catch(() => {}); arm(STALL_MS); return; }
    clearTimeout(watchdog);
    audio.onended = audio.onerror = audio.ontimeupdate = audio.onplay = null;
    audio.pause();
    let index = Math.floor(at * item.text.length);
    while (index > 0 && !/\s/.test(item.text[index - 1])) index--;
    const rest = { text: item.text.slice(index), el: null, start: item.start === null ? null : item.start + index };
    setStatus("The voice stalled; the rest is spoken with the browser's voice.");
    speakWithBrowser(rest, finished);
    item.stop = () => { if (rest.stop) rest.stop(); item.stop = null; };
  };
  const done = () => {
    clearTimeout(watchdog);
    item.stop = null;
    audio.onended = audio.onerror = audio.ontimeupdate = audio.onplay = null;
    if (speakingNow !== item) return;      // stopped meanwhile: the element is somebody else's now
    audio.pause();
    finished();
  };
  item.stop = () => { clearTimeout(watchdog); item.stop = null; };
  audio.onplay = () => { arm(STALL_MS); if (item.start !== null) speechPos = item.start; };
  audio.ontimeupdate = () => {
    arm(STALL_MS);
    if (!audio.duration) return;
    // The server gives no word times: estimate the word from how far through the audio we are
    let index = Math.min(item.text.length - 1, Math.floor((audio.currentTime / audio.duration) * item.text.length));
    while (index > 0 && !/\s/.test(item.text[index - 1])) index--;
    if (index !== lastIndex) { lastIndex = index; markWord(item, index); }
  };
  arm(START_MS);                           // the sound has not started yet
  audio.onended = audio.onerror = done;
  audio.play().then(soundUnlocked, (err) => {
    clearTimeout(watchdog);
    audio.onended = audio.onerror = audio.ontimeupdate = audio.onplay = null;
    if (speakingNow !== item) return;
    if (err && err.name === "NotAllowedError") {
      // The phone blocks sound until the page has been tapped. A greeting is simply dropped; any other sentence is
      // kept and played at the next tap.
      item.stop = null; player = null;
      if (item.optional) { URL.revokeObjectURL(url); return finished(); }
      soundBlocked = () => playAudio(item, url, finished);
      setStatus("Tap the screen once to turn the voice on.");
      announce("Tap the screen once to turn the voice on.");      // the voice cannot say this itself
      return;
    }
    item.stop = null; item.url = null; URL.revokeObjectURL(url); player = null;
    speakWithBrowser(item, finished);
  });
}

// The word being spoken is highlighted with the CSS Custom Highlight API where the browser has it: the text itself is
// never rewritten, so a screen reader's reading position in the answer is not disturbed. Older browsers get a <mark>.
const HIGHLIGHT = !!(window.CSS && CSS.highlights && window.Highlight);
function clearHighlight() { if (HIGHLIGHT) CSS.highlights.delete("speaking"); }

// Mark the word at charIndex of the shown sentence, and move the book reading position
function markWord(item, charIndex) {
  if (item.start !== null) speechPos = item.start + charIndex;
  if (!item.el) return;
  const t = item.text;
  const end = t.slice(charIndex).search(/\s|$/) + charIndex;
  const node = item.el.firstChild;
  if (HIGHLIGHT && node && node.nodeType === Node.TEXT_NODE && item.el.childNodes.length === 1 && node.data === t) {
    const range = new Range();
    range.setStart(node, charIndex);
    range.setEnd(node, end);
    CSS.highlights.set("speaking", new Highlight(range));
    return;
  }
  item.el.innerHTML = escapeHtml(t.slice(0, charIndex)) + "<mark>" + escapeHtml(t.slice(charIndex, end)) +
    "</mark>" + escapeHtml(t.slice(end));
}

// The browser's own voice: used when the server's voice is not available
function speakWithBrowser(item, finished) {
  if (!synth) return finished();
  // The spoken text may differ from the shown text (long numbers are spelled out), so the word position
  // is mapped back by counting words.
  const u = new SpeechSynthesisUtterance(spokenForm(item.text));
  u.lang = "en-US";
  u.rate = 1.0;
  u.onstart = () => { if (item.start !== null) speechPos = item.start; };
  // Some browsers never report the end of an utterance: don't let that freeze the app. Every word event shows
  // the voice is alive, so the wait then shrinks to a few seconds.
  let watchdog = null, keepAlive = null;
  const arm = (ms) => { clearTimeout(watchdog); watchdog = setTimeout(() => done(), ms); };
  const done = () => { clearTimeout(watchdog); clearInterval(keepAlive); item.stop = null; finished(); };
  item.stop = () => { clearTimeout(watchdog); clearInterval(keepAlive); item.stop = null; };
  u.onboundary = (e) => {
    arm(8000);
    if (e.name === "word") markWord(item, shownIndex(item.text, u.text, e.charIndex));
  };
  u.onend = u.onerror = done;
  arm(4000 + u.text.length * 120);
  // Chrome silently stops a long utterance after about 15 seconds; a pause and resume keeps it going
  keepAlive = setInterval(() => { if (synth.speaking && !synth.paused) { synth.pause(); synth.resume(); } }, 10000);
  if (synth.paused) synth.resume();
  synth.speak(u);
}

// Position in the shown text of the word that starts at spokenIndex in the spoken text
function shownIndex(shown, spoken, spokenIndex) {
  if (shown === spoken) return spokenIndex;
  const wordNo = spoken.slice(0, spokenIndex).split(/\s+/).filter(Boolean).length;
  const longNumbers = [...shown.matchAll(LONG_NUMBER)].filter((m) => Number(m[0].replace(/,/g, "")) > 9999);
  // Each spelled-out number is several spoken words but one shown word
  let spokenWords = 0, result = shown.length;
  for (const m of shown.matchAll(/\S+/g)) {
    const isLong = longNumbers.some((n) => n.index >= m.index && n.index < m.index + m[0].length);
    const count = isLong ? m[0].replace(/,/g, "").replace(/\D/g, "").length : 1;
    if (wordNo < spokenWords + count) { result = m.index; break; }
    spokenWords += count;
  }
  return Math.min(result, shown.length);
}

// Stop talking (and keep any answer that is still arriving)
function silence() {
  speechQueue.length = 0;
  soundBlocked = null;
  if (speakingNow && speakingNow.stop) speakingNow.stop();      // its timers must not fire into the next sentence
  speakingNow = null;
  speechPos = null;
  lastSpeechActivity = performance.now();
  if (player) { player.onended = player.onerror = null; player.pause(); player = null; }
  if (synth) synth.cancel();
  srBusyUntil = 0;
  clearHighlight();
  answerEl.querySelectorAll("p").forEach((p) => { if (p.querySelector("mark")) p.textContent = p.textContent; });
}

// Stop talking and stop the answer that is being generated
function stopSpeaking() {
  pendingTurn = false;
  silence();
  if (controller) { controller.abort(); controller = null; }
  thinking = false;
}

// ================================================================ camera
let cameraReady = false;
let uploadView = false;       // an uploaded photo is shown in place of the camera

let cameraFailed = "";        // why the camera could not be opened ("" when it works)

// Is the microphone already allowed (so that nothing needs to be asked)?
async function micAlreadyAllowed() {
  try {
    return !!(navigator.permissions && (await navigator.permissions.query({ name: "microphone" })).state === "granted");
  } catch (err) {
    return false;                      // this browser cannot say
  }
}

async function startCamera(first = true) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return cameraUnavailable("This browser has no camera access. Use Upload photo instead.");
  }
  const videoWanted = { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } };
  try {
    let stream = null, micNote = "";
    // The first time the page opens, the camera and the microphone are asked for in ONE request: a single permission
    // question, and no second request that could freeze the camera on a phone. The microphone is released at once
    // (only the permission is wanted; listening starts when Listen is pressed). Without a microphone, or when it is
    // refused, the camera alone still starts.
    if (first && !(await micAlreadyAllowed())) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: videoWanted, audio: true });
        stream.getAudioTracks().forEach((track) => { track.stop(); stream.removeTrack(track); });
      } catch (err) {
        stream = null;
        micNote = MIC_HELP[err.name === "NotFoundError" ? "audio-capture" : "not-allowed"];
      }
    }
    if (!stream) stream = await navigator.mediaDevices.getUserMedia({ video: videoWanted, audio: false });
    if (uploadView) { stream.getTracks().forEach((t) => t.stop()); return; }      // an upload was opened meanwhile
    video.srcObject = stream;
    video.addEventListener("loadedmetadata", () => { cameraReady = true; sizeOverlay(); if (!ws) connectLive(); }, { once: true });
    setStatus(micNote || "Camera ready. Show something to read.");
    if (first) say("System online. Show me something to read.", { optional: true });
  } catch (err) {
    cameraUnavailable(CAMERA_HELP[err.name] || "The camera could not start (" + err.name + "). Reload the page, or use Upload photo instead.");
  }
}

// Switch the camera off completely (the light goes out), not just hide it
function stopCamera() {
  cameraReady = false;
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

// What each camera error means, in words that say what to do next
const CAMERA_HELP = {
  NotAllowedError: "The camera is blocked. Allow it with the camera icon in the address bar and reload the page, or use Upload photo instead.",
  NotFoundError: "No camera was found. Connect one and reload the page, or use Upload photo instead.",
  NotReadableError: "The camera is in use by another app (for example Zoom or FaceTime). Close it and reload the page, or use Upload photo instead.",
};

function cameraUnavailable(message) {
  cameraFailed = message;
  video.hidden = true;
  overlay.hidden = true;
  $("camera-message").hidden = false;
  $("camera-message").textContent = message;
  $("capture").disabled = true;
  setStatus(message);
  say(message);
}

// A JPEG of the current camera frame, scaled so its longer side is at most maxSide
function frameBlob(maxSide, quality) {
  const src = uploadView ? still : video;
  const w = uploadView ? still.naturalWidth : video.videoWidth, h = uploadView ? still.naturalHeight : video.videoHeight;
  return new Promise((resolve) => {
    const scale = Math.min(1, maxSide / Math.max(w, h));
    grab.width = Math.round(w * scale);
    grab.height = Math.round(h * scale);
    grab.getContext("2d").drawImage(src, 0, 0, grab.width, grab.height);
    grab.toBlob(resolve, "image/jpeg", quality);
  });
}

// ================================================================ live stream (guidance, page turns)
let ws = null;
let paused = false;           // frames are not sent (a capture or a page is being read)
let awaiting = false;         // a frame was sent and its reply has not arrived
let sentAt = 0;
let guide = null;             // latest reply, normal mode
let bookState = null;         // latest reply, book mode
let liveFailures = 0;         // connection attempts in a row that failed
let liveLost = false;         // "Lost the connection" has been said

function connectLive() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live?sid=${sid}`;
  ws = new WebSocket(url);
  ws.onopen = () => {
    awaiting = false;
    ws.send(JSON.stringify({ type: "mode", mode }));
    if (liveLost) { liveLost = false; setStatus("Connected to the server again."); say("Connected again."); }
    liveFailures = 0;
  };
  ws.onmessage = (e) => { awaiting = false; onLiveReply(JSON.parse(e.data)); };
  ws.onclose = () => {
    ws = null; awaiting = false;
    // Said once, after a few tries in a row have failed, so a blind user knows why the guidance went quiet
    if (++liveFailures === 3 && !liveLost) {
      liveLost = true;
      setStatus("Lost the connection to the server. Trying again… (check that it is still running)");
      say("Lost the connection to the server. I'll keep trying.");
    }
    setTimeout(connectLive, 1500);
  };
  ws.onerror = () => ws && ws.close();
}

function wsSend(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

async function liveLoop() {
  const stuck = awaiting && performance.now() - sentAt > 8000;   // a lost reply must not stall the loop
  // Normal mode looks for text only while the app is quiet; book mode keeps watching for page turns
  const wanted = mode === "book" || !(isSpeaking() || thinking);
  if (cameraReady && !uploadView && ws && ws.readyState === WebSocket.OPEN && (!awaiting || stuck) && !paused && wanted && !document.hidden) {
    awaiting = true;
    sentAt = performance.now();
    const blob = await frameBlob(960, 0.7);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(blob); else awaiting = false;
  } else if (mode === "book" && bookState && !uploadView) {
    bookTick(bookState, false);       // keep the reading bookkeeping going while no frame is in flight
  }
  setTimeout(liveLoop, 100);
}

function onLiveReply(reply) {
  if (reply.type === "guide" && mode === "normal") {
    guide = reply;
    if (reply.speak) say(reply.speak);
    if (reply.capture) autoCapture();
  } else if (reply.type === "book" && mode === "book") {
    bookState = reply;
    bookTick(reply, true);
  }
}

// What went wrong with a request, in words. The server answers errors with JSON ({"detail": ...});
// when the host itself is down, restarting or out of memory it answers with an HTML page instead.
async function failure(res) {
  try {
    const detail = (await res.json()).detail;
    if (detail) return detail;
  } catch (err) { /* not JSON: fall through */ }
  if (res.status === 502 || res.status === 503 || res.status === 504 || res.status === 0) {
    return "The server is not responding. It may be starting up, or it ran out of memory and is restarting. Wait a minute and try again.";
  }
  return "The server answered with an error (" + res.status + "). Try again.";
}

// The message of a failed request for the status line: a request that never reached the server says so plainly
// instead of the browser's "Failed to fetch"
function problem(err) {
  if (err instanceof TypeError) return "Can't reach the Helping Eyes server. It may have stopped or lost its connection; try again in a moment.";
  return err.message;
}

// Hide the "look it up online?" offer. If the keyboard focus was on one of its buttons, it moves to the question box
// instead of falling back to the top of the page.
function hideOffer() {
  const offer = $("offer");
  if (offer.contains(document.activeElement)) $("question").focus();
  offer.hidden = true;
}

// ================================================================ capture (normal mode)
function showCaptured(data) {
  $("captured").hidden = false;
  $("captured-text").textContent = data.text;
  $("page-label").textContent = [data.label, data.running_title].filter(Boolean).join(" · ");
}

async function postCapture(blob, book) {
  const form = new FormData();
  form.append("image", blob, "capture.jpg");
  form.append("sid", sid);
  form.append("book", book ? "true" : "false");
  const res = await fetch("/api/capture", { method: "POST", body: form });
  if (!res.ok) throw new Error(await failure(res));
  return res.json();
}

// The server said the item is steady and in view: take the full-size photo
async function autoCapture() {
  paused = true;
  setStatus("Capturing…");
  let ok = false;
  try {
    const data = await postCapture(await frameBlob(1920, 0.92), false);
    ok = data.ok;
    if (ok) {
      stopSpeaking();
      showCaptured(data);
      setStatus(`Captured in ${data.seconds} s. ` + data.message);
      say(data.message);
    } else {
      setStatus(data.message);
      say(data.message);
    }
  } catch (err) {
    setStatus("Could not read the photo: " + problem(err));
    say(err instanceof TypeError ? problem(err) : "Could not read the text. Try again.");
  }
  wsSend({ type: "capture_result", ok });
  paused = false;
}

// The Capture now button: take a photo immediately (normal mode) or read the page in view (book mode)
// The shutter on the camera picture does exactly what the Capture now button does: whatever the guidance says,
// the picture in view is taken now and used
const cameraBox = document.querySelector(".camera");
function flash() {
  cameraBox.classList.remove("flash");
  void cameraBox.offsetWidth;                                  // restart the animation
  cameraBox.classList.add("flash");
}
const shutter = $("shutter");                                  // missing when the browser kept an older copy of the page
if (shutter) shutter.addEventListener("click", () => { flash(); $("capture").click(); });
let shutterShown = null;
function showShutter() {                                       // only while the live camera is in use
  const want = cameraReady && !uploadView;
  if (shutter && want !== shutterShown) { shutterShown = want; shutter.hidden = !want; }
}

let capturing = false;                                         // a photo is being read: a second press must wait
$("capture").addEventListener("click", async () => {
  if (uploadView && mode === "book") return handleBookEvent("changed");
  if (!uploadView && !video.videoWidth) { setStatus("The camera is not ready yet."); return say("The camera is not ready yet."); }
  if (mode === "book") return handleBookEvent("changed");
  if (capturing) return;
  capturing = true;
  paused = true;
  stopSpeaking();
  setStatus("Reading the text…");
  say("Reading.");
  let ok = false;
  try {
    const blob = await frameBlob(1920, 0.92);
    // The photo taken stays on the screen and the camera is switched off, until New item (or "next") brings it back
    if (!uploadView) showStill(URL.createObjectURL(blob), true);
    const data = await postCapture(blob, false);
    ok = data.ok;
    stopSpeaking();
    setStatus(ok ? `Captured in ${data.seconds} s. ` + data.message + " Press New item, or say next, for the camera." : data.message);
    if (ok) { showCaptured(data); checkQuality(); } else if (capturedStill) backToCamera();    // nothing read: try again
    say(data.message);
  } catch (err) {
    setStatus("Could not read the photo: " + problem(err));
    say(err instanceof TypeError ? problem(err) : "Sorry, I could not read the photo.");
    if (capturedStill) backToCamera();
  }
  wsSend({ type: "capture_result", ok });
  paused = false;
  capturing = false;
});

// An uploaded photo: shown in place of the camera, with the same overlay
let capturedStill = false;     // the still is a photo just taken with the camera (not an uploaded one)
function showStill(url, captured = false) {
  capturedStill = captured;
  uploadView = true;
  stopCamera();                       // the camera is off while an uploaded photo is used
  still.onload = () => { sizeOverlay(); };
  still.src = url;
  still.hidden = false;
  video.hidden = true;
  overlay.hidden = false;
  $("camera-message").hidden = true;
  $("back-to-camera").hidden = false;
  guide = null; bookState = null;
  sizeOverlay();
}

// Sharpness, light and glare of the uploaded photo: the same figures the camera view shows live
let photoQuality = null;
async function checkQuality() {
  photoQuality = null;
  try {
    const form = new FormData();
    form.append("image", await frameBlob(960, 0.8), "photo.jpg");
    const res = await fetch("/api/quality", { method: "POST", body: form });
    if (!res.ok || !uploadView) return;
    photoQuality = await res.json();
  } catch (err) { /* the picture is still read; only the quality figures are missing */ }
}

// Quality line (red when something is wrong): the figures and the hint only, nothing drawn on the picture
function drawPhotoQuality(ctx, W, H, bottom) {
  const q = photoQuality;
  if (!q) return;
  plateText(ctx, q.quality + (q.hint ? "   " + q.hint : ""), (30 / 1280) * W + 14, bottom,
            q.problems.length ? COLORS.problem : COLORS.idle, "13px sans-serif");
}

function backToCamera() {
  if (!uploadView) return;
  uploadView = false;
  capturedStill = false;
  photoQuality = null;
  still.hidden = true;
  if ((still.getAttribute("src") || "").startsWith("blob:")) URL.revokeObjectURL(still.getAttribute("src"));
  still.removeAttribute("src");
  $("back-to-camera").hidden = true;
  resetBook();
  if (cameraFailed) return cameraUnavailable(cameraFailed);
  video.hidden = false;
  setStatus("Starting the camera…");
  startCamera(false);
}
$("back-to-camera").addEventListener("click", backToCamera);

$("upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  stopSpeaking();
  resetBook();
  const url = URL.createObjectURL(file);
  showStill(url);
  try { await still.decode(); } catch (err) { setStatus("Could not open that image."); return say("Sorry, I could not open that image."); }
  checkQuality();
  if (mode === "book") return handleBookEvent("changed");        // read it page-style, with the reading overlay
  setStatus("Reading the text…");
  say("Reading.");
  try {
    const data = await postCapture(await frameBlob(1920, 0.92), $("book").checked);
    stopSpeaking();
    if (!data.ok) { setStatus(data.message); return say(data.message); }
    showCaptured(data);
    const intro = [data.label, data.running_title].filter(Boolean).join(". ");
    setStatus(`Captured in ${data.seconds} s. ` + data.message);
    say((intro ? intro + ". " : "") + data.message);
    $("question").focus();
  } catch (err) {
    setStatus("Could not read the photo: " + problem(err));
    say(err instanceof TypeError ? problem(err) : "Sorry, I could not read the photo.");
  }
});

// ================================================================ book mode
let pageText = "";            // text of the page being read
let pageLayout = null;        // where its lines and words are on the camera frame
let pageLabel = "";
let readPos = 0;              // how far reading has got in pageText (characters)
let lastChunkStart = 0;       // start of the page's last spoken piece
let reading = false;          // the page is being read aloud
let turnPrompted = true;      // "Turn the page." already said for this page
let bookBusy = false;
let pendingTurn = false;      // a page turn was seen while reading: read the new page when this one is done

function resetBook() {
  pageText = ""; pageLayout = null; pageLabel = ""; readPos = 0; lastChunkStart = 0;
  reading = false; turnPrompted = true; bookState = null;
}

function setMode(next) {
  if (next === mode) return;
  mode = next;
  guide = null;
  resetBook();
  $("bookmode").setAttribute("aria-pressed", String(mode === "book"));
  $("capture").firstChild.textContent = mode === "book" ? "Read this page " : "Capture now ";
  wsSend({ type: "mode", mode });
  if (uploadView && mode === "book") handleBookEvent("changed");   // read the shown photo page-style
}

// Called for every book-mode reply from the server (about ten times a second)
function bookTick(reply, fresh) {
  if (speechPos !== null) readPos = Math.max(readPos, speechPos);
  if (reading && !isSpeaking()) {
    reading = false;
    if (readPos >= lastChunkStart) readPos = pageText.length;       // read to the end
  }
  const readingNow = reading && isSpeaking();
  // While a page is being read, nothing the camera sees (the book moving, a hand, a page turn) interrupts it:
  // it stops only when asked to, or when the page is done. A page turn seen meanwhile is read afterwards.
  if (uploadView) return;
  if (fresh && reply.event && readingNow) { if (reply.event === "changed") pendingTurn = true; return; }
  if (!readingNow && pendingTurn && !bookBusy && !thinking) { pendingTurn = false; return handleBookEvent("changed"); }
  if (fresh && reply.event && !bookBusy) return handleBookEvent(reply.event);
  if (pageText && !turnPrompted && !uploadView && !bookBusy && !(isSpeaking() || thinking)) {
    say("Turn the page.");
    turnPrompted = true;
  }
}

function speakChunks(chunks) {
  for (const c of chunks) { say(c.text, { start: c.start }); lastChunkStart = c.start; }
  reading = chunks.length > 0;
}

// The page view became steady (a page turn or the view moved): read it on the server, then speak
async function handleBookEvent(event) {
  if (bookBusy) return;
  bookBusy = true;
  paused = true;
  setStatus("Reading the page…");
  try {
    const form = new FormData();
    form.append("image", await frameBlob(1600, 0.9), "page.jpg");
    form.append("sid", sid);
    form.append("event", event);
    form.append("read_pos", String(Math.max(readPos, speechPos ?? 0)));
    form.append("reading", String(reading));
    form.append("thinking", String(thinking));
    const res = await fetch("/api/book/page", { method: "POST", body: form });
    if (!res.ok) throw new Error(await failure(res));
    const r = await res.json();
    if (mode !== "book") return;

    if (r.action === "new" || r.action === "resume") {
      pageText = r.text; pageLayout = r.layout; pageLabel = r.label || "";
      answerEl.innerHTML = "";
      hideOffer();
      showCaptured({ text: r.text, label: r.label });
      stopSpeaking();
      if (r.action === "new") {
        if (r.intro) say(r.intro);              // the page number and running title, then the page
        readPos = 0;
        setStatus(`${r.label || "Page"} read in ${r.seconds} s.`);
      } else {
        readPos = r.from_pos;
        setStatus("Same words in view: continuing.");
      }
      speakChunks(r.chunks);
      turnPrompted = false;
    } else if (r.action === "nothing_new") {
      pageText = r.text; pageLayout = r.layout; pageLabel = r.label || "";
      readPos = pageText.length;
      setStatus("Same words in view: nothing new to read.");
    } else if (r.action === "no_text") {
      say("I can't see any text on this page.");
      setStatus("No text on this page.");
    }
    // "keep": the same page barely moved while reading: nothing changes
  } catch (err) {
    setStatus("Could not read the page: " + problem(err));
    say("Could not read the page. " + problem(err));
  } finally {
    bookBusy = false;
    paused = false;
  }
}

// ================================================================ ask (every request goes to the server)
// Show the request (spoken or typed) above the answer, so the user can see what was understood
function showYouSaid(text) {
  const el = $("you-said");
  el.textContent = "You said: " + text;
  el.hidden = false;
}

async function ask(text) {
  text = (text || "").trim();
  if (!text) return;
  stopSpeaking();
  hideOffer();
  showYouSaid(text);
  answerEl.innerHTML = "";
  lastSentences = [];
  setStatus("One moment…");
  const mine = new AbortController();
  controller = mine;
  thinking = true;
  const started = performance.now();
  let firstShown = false;
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sid, question: text, provider: $("model").value }), signal: mine.signal,
    });
    if (!res.ok) throw new Error(await failure(res));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (line) onAskEvent(JSON.parse(line), started, () => firstShown, () => { firstShown = true; });
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      setStatus(err instanceof TypeError ? problem(err) : "Something went wrong: " + err.message);
      say(err instanceof TypeError ? problem(err) : "Sorry, something went wrong. " + err.message);
    }
  } finally {
    if (controller === mine) { controller = null; thinking = false; }
  }
}

function onAskEvent(ev, started, wasShown, markShown) {
  if (ev.type === "sentence") {
    if (!wasShown()) {
      markShown();
      setStatus("Answering…");
    }
    const p = document.createElement("p");
    p.textContent = ev.text;
    if (ev.source === "web") p.className = "web";
    answerEl.appendChild(p);
    lastSentences.push(ev.text);
    say(ev.text, { el: p });
  } else if (ev.type === "command") {
    if (ev.name === "stop") silence();
    else if (ev.name === "book_on") setMode("book");
    else if (ev.name === "book_off") setMode("normal");
    else if (ev.name === "new_capture") { guide = null; resetBook(); backToCamera(); }
  } else if (ev.type === "offer_search") {
    $("offer").hidden = false;
    say(ev.text);
  } else if (ev.type === "done") {
    setStatus(ev.seconds !== undefined ? `Answered in ${ev.seconds} s.` : "Done.");
  }
}

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("question").value);
  $("question").value = "";
});
$("read-all").addEventListener("click", () => ask("read everything"));
$("repeat").addEventListener("click", () => ask("repeat"));
$("stop").addEventListener("click", stopSpeaking);
$("newitem").addEventListener("click", () => ask("next"));
$("bookmode").addEventListener("click", () => ask(mode === "book" ? "normal mode" : "book mode"));
$("offer-yes").addEventListener("click", () => ask("yes"));
$("offer-no").addEventListener("click", () => { hideOffer(); ask("no"); });

// ================================================================ speech in (hands-free)
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null;
let listening = false;        // the user turned listening on
let micFailures = 0;          // speech errors in a row that are not just "nobody spoke"
const ECHO_GUARD = 700;       // ms after the app stops talking before the microphone listens

// What each speech error means, in words the user can act on
const MIC_HELP = {
  "network": "The browser could not reach its speech service. Listening needs an internet connection and does not work in some browsers, for example Brave. Type your question instead.",
  "audio-capture": "No microphone was found. Plug one in or choose it in the browser settings, or type your question instead.",
  "not-allowed": "Microphone access was denied. Allow it with the icon in the address bar, then press Listen again, or type your question instead.",
  "language-not-supported": "This browser cannot recognise English speech. Type your question instead.",
};
MIC_HELP["service-not-allowed"] = MIC_HELP["not-allowed"];

// The button shows whether the microphone is really on: it is off while the app is talking or thinking
// (even though listening is switched on) and turns itself on again when the app is quiet.
function micIsOpen() {
  return listening && !thinking && !isSpeaking() && performance.now() - lastSpeechActivity >= ECHO_GUARD;
}

function renderMic() {
  const open = micIsOpen();
  // The name stays "Listen"; the pressed state (a filled button with a check mark) says whether the microphone is open
  $("mic").setAttribute("aria-pressed", String(open));
  $("mic").title = listening && !open ? "Listening is on and resumes when the app stops talking. Press to stop talking and listen now." : "";
}

function setListening(on, quiet = false) {
  listening = on;
  micFailures = 0;
  renderMic();
  setStatus(on ? "Listening. Ask a question, or say a command." : "Listening is off.");
  if (!quiet) say(on ? "Listening." : "Listening is off.");
  if (!on && recognizer) recognizer.abort();
}

// Listening cannot work: switch it off and say why (spoken too, for someone who cannot read the screen)
function micProblem(message) {
  setListening(false, true);
  setStatus(message);
  say(message);
}

// Ask for the microphone now, from the button press, so the browser's permission question appears at once
async function enableListening() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    setStatus("Asking the browser for the microphone…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());       // only checking: the speech engine opens it itself
    } catch (err) {
      return micProblem(MIC_HELP[err.name === "NotFoundError" ? "audio-capture" : "not-allowed"]);
    }
  }
  setListening(true);
}

if (!Recognition) {
  $("mic").disabled = true;
  $("mic").title = "Speech recognition is not supported in this browser (use Chrome or Edge). Type instead.";
} else {
  $("mic").addEventListener("click", () => {
    if (thinking || isSpeaking()) {                       // pressed while the app talks: stop talking and listen now
      stopSpeaking();
      lastSpeechActivity = performance.now() - ECHO_GUARD + 150;
      return listening ? renderMic() : enableListening();
    }
    listening ? setListening(false) : enableListening();
  });
}

// Listen only while the app is quiet, so it does not hear its own voice
let recognizerEndedAt = 0;
function listenTick() {
  renderMic();
  if (!listening || recognizer || thinking || isSpeaking()) return;
  if (MOBILE && performance.now() - recognizerEndedAt < 500) return;      // phones fail when the microphone is reopened at once
  if (performance.now() - lastSpeechActivity < ECHO_GUARD) return;
  const startedAt = performance.now();
  const r = new Recognition();
  recognizer = r;
  r.lang = "en-US";
  r.interimResults = true;                                // show words as they are heard: proof the microphone works
  r.maxAlternatives = 1;
  r.onresult = (e) => {
    if (lastSpeechActivity > startedAt) return;           // the app started talking: that was its own voice
    // The whole sentence so far: the recognizer can split one utterance into several results
    const results = Array.from(e.results);
    // Phones repeat the whole phrase so far in every result: a result that starts with the previous one replaces it
    const parts = [];
    for (const x of results) {
      const t = x[0].transcript.trim();
      if (!t) continue;
      const last = parts[parts.length - 1];
      if (last !== undefined && t.toLowerCase().startsWith(last.toLowerCase())) parts[parts.length - 1] = t;
      else parts.push(t);
    }
    const text = parts.join(" ");
    const result = results[results.length - 1];
    if (!result.isFinal) return showYouSaid(text + "…");     // shown as it is heard; nothing is announced yet
    micFailures = 0;
    setStatus("Heard: " + text);
    ask(text);
  };
  r.onerror = (e) => {
    if (e.error === "aborted" || e.error === "no-speech") return;      // we stopped it, or nobody spoke: just listen again
    if (MIC_HELP[e.error]) return micProblem(MIC_HELP[e.error]);
    if (++micFailures >= 3) micProblem("Listening keeps failing (" + e.error + "). Type your question instead.");
  };
  r.onend = () => { recognizerEndedAt = performance.now(); if (recognizer === r) recognizer = null; };
  try { r.start(); } catch (err) { recognizer = null; }
}
setInterval(listenTick, 100);

// ================================================================ model menu (Gemini online or Qwen on this computer)
try { const saved = localStorage.getItem("model"); if (saved) $("model").value = saved; } catch (err) { /* storage blocked */ }
$("model").addEventListener("change", () => {
  try { localStorage.setItem("model", $("model").value); } catch (err) { /* storage blocked */ }
  setStatus("Answers now come from " + $("model").selectedOptions[0].textContent.replace(/ \(.*/, "") + ".");
});

// ================================================================ settings
const SHORTCUTS = { c: "capture", r: "newitem", b: "bookmode", a: "read-all", p: "repeat", s: "stop", m: "mic" };
let shortcutsOn = true;
function stored(name, fallback) { try { const v = localStorage.getItem(name); return v === null ? fallback : v === "1"; } catch (err) { return fallback; } }
function store(name, on) { try { localStorage.setItem(name, on ? "1" : "0"); } catch (err) { /* storage blocked */ } }

function applySettings() {
  srMode = $("sr-mode").checked;
  shortcutsOn = $("shortcuts").checked;
  // Without single-key shortcuts the key hints and aria-keyshortcuts go too, so nothing promises a key that does nothing
  document.body.classList.toggle("no-shortcuts", !shortcutsOn);
  for (const [key, id] of Object.entries(SHORTCUTS)) {
    if (shortcutsOn) $(id).setAttribute("aria-keyshortcuts", key.toUpperCase()); else $(id).removeAttribute("aria-keyshortcuts");
  }
}
$("sr-mode").checked = stored("srMode", false);
$("shortcuts").checked = stored("shortcuts", true);
applySettings();
$("sr-mode").addEventListener("change", () => {
  store("srMode", $("sr-mode").checked);
  silence();
  applySettings();
  say(srMode ? "Your screen reader now reads everything. The app's voice is off." : "The app's voice is on again.");
});
$("shortcuts").addEventListener("change", () => {
  store("shortcuts", $("shortcuts").checked);
  applySettings();
  setStatus(shortcutsOn ? "Single-key shortcuts are on." : "Single-key shortcuts are off. Use Tab and the buttons.");
});

// ================================================================ keyboard
document.addEventListener("keydown", (e) => {
  // never while typing or choosing: a letter typed in the question box or the menu is not a shortcut
  const t = e.target;
  if (!shortcutsOn || ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || t.isContentEditable || e.metaKey || e.ctrlKey || e.altKey) return;
  const id = SHORTCUTS[e.key.toLowerCase()];
  if (id && !$(id).disabled) { e.preventDefault(); $(id).click(); }
});

// ================================================================ overlay
// Overlay palette. Labels sit on a dark plate (PLATE), so each one is readable on any camera picture: measured on the
// plate over a white page, idle 8.3:1, detect 9.0:1, hold 6.2:1, ready 6.3:1, problem 5.6:1, white 12.6:1.
const COLORS = { idle: "#d1d1d6", detect: "#ffd60a", hold: "#ff9f0a", ready: "#30d158", line: "#64d2ff",
                 problem: "#ff8a84", text: "#ffffff", gutter: "#ffd60a", margin: "#bf5af2", column: "#64d2ff",
                 block: "#30d158", badge: "#004c9f", reading: "rgba(255, 214, 10, 0.35)", word: "#ff9f0a" };
const PLATE = "rgba(0, 0, 0, 0.8)";

// Text on a rounded dark plate (x, y: the text's baseline start)
function plateText(ctx, text, x, y, color, font) {
  ctx.font = font;
  const m = ctx.measureText(text);
  const up = m.actualBoundingBoxAscent || 16, down = m.actualBoundingBoxDescent || 4, padX = 8, padY = 5;
  ctx.fillStyle = PLATE;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x - padX, y - up - padY, m.width + 2 * padX, up + down + 2 * padY, 8);
  else ctx.rect(x - padX, y - up - padY, m.width + 2 * padX, up + down + 2 * padY);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function sizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const el = uploadView ? still : video;
  // The overlay sits exactly on the picture (which may be narrower than its frame on a laptop screen)
  Object.assign(overlay.style, { left: el.offsetLeft + "px", top: el.offsetTop + "px",
                                 width: el.clientWidth + "px", height: el.clientHeight + "px" });
  overlay.width = Math.round(el.clientWidth * dpr);
  overlay.height = Math.round(el.clientHeight * dpr);
}
window.addEventListener("resize", sizeOverlay);
video.addEventListener("resize", sizeOverlay);
still.addEventListener("load", sizeOverlay);
if (window.ResizeObserver) new ResizeObserver(sizeOverlay).observe(document.querySelector(".camera"));   // the picture's space changes with the layout

function rect(ctx, box, W, H, pad = 0) {
  ctx.strokeRect(box[0] * W - pad, box[1] * H - pad, (box[2] - box[0]) * W + 2 * pad, (box[3] - box[1]) * H + 2 * pad);
}

// An outline that follows slanted text: four corners as fractions of the picture (stroke, or fill when asked)
function poly(ctx, quad, W, H, fill = false) {
  ctx.beginPath();
  quad.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
  ctx.closePath();
  if (fill) ctx.fill(); else ctx.stroke();
}

// Outlines when the server sent them (they follow a slanted page), otherwise plain boxes
function outlines(ctx, quads, boxes, W, H, pad = 0) {
  if (quads && quads.length) quads.forEach((q) => poly(ctx, q, W, H));
  else (boxes || []).forEach((b) => rect(ctx, b, W, H, pad));
}

function drawGuideBox(ctx, W, H, color, label) {
  const mx = (30 / 1280) * W, my = (20 / 720) * H, c = 30;
  ctx.strokeStyle = color; ctx.lineWidth = 3;
  for (const [x, y, dx, dy] of [[mx, my, 1, 1], [W - mx, my, -1, 1], [mx, H - my, 1, -1], [W - mx, H - my, -1, -1]]) {
    ctx.beginPath(); ctx.moveTo(x + dx * c, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * c); ctx.stroke();
  }
  ctx.lineWidth = 1; ctx.strokeRect(mx, my, W - 2 * mx, H - 2 * my);
  if (label) plateText(ctx, label, mx + 16, my + 36, color, `bold ${W < 640 ? 15 : 22}px sans-serif`);
}

function drawNormal(ctx, W, H) {
  const busy = isSpeaking() || thinking;
  if (busy) return drawGuideBox(ctx, W, H, COLORS.idle, thinking && !isSpeaking() ? "THINKING..." : "SPEAKING...  S = stop");
  if (uploadView) {
    if (photoQuality) {
      ctx.strokeStyle = COLORS.line; ctx.lineWidth = 1;
      outlines(ctx, photoQuality.quads, photoQuality.lines, W, H);
      if (photoQuality.box && !(photoQuality.quads || []).length) { ctx.lineWidth = 2; rect(ctx, photoQuality.box, W, H); }
    }
    drawGuideBox(ctx, W, H, capturedStill ? COLORS.ready : COLORS.idle, capturedStill ? "CAPTURED   R = new item" : "UPLOADED PHOTO");
    return drawPhotoQuality(ctx, W, H, H - (20 / 720) * H - 12);
  }
  if (!guide) return drawGuideBox(ctx, W, H, COLORS.idle, "SHOW TEXT HERE");
  ctx.strokeStyle = COLORS.line; ctx.lineWidth = 1;
  outlines(ctx, guide.quads, guide.lines, W, H);
  if (guide.box && !(guide.quads || []).length) { ctx.lineWidth = 2; rect(ctx, guide.box, W, H); }
  const color = { CAPTURED: COLORS.ready, CAPTURING: COLORS.ready, HOLD: COLORS.hold, COACH: COLORS.hold,
                  ADJUST: COLORS.detect }[guide.state] || COLORS.idle;
  const label = guide.state === "CAPTURED" ? "CAPTURED - ASK ME   R = new item   A = read all" : guide.label;
  drawGuideBox(ctx, W, H, color, label);
  plateText(ctx, guide.quality, (30 / 1280) * W + 14, H - (20 / 720) * H - 14,
            guide.problems.length ? COLORS.problem : COLORS.idle, "13px sans-serif");
}

function lineAt(pos) {
  return pageLayout && pageLayout.lines.find((l) => l.start <= pos && pos < l.end);
}

function drawBook(ctx, W, H) {
  const turning = !uploadView && (!bookState || bookState.state === "TURNING");
  if (pageLayout && !turning) {
    const L = pageLayout;
    if (L.gutter) { ctx.strokeStyle = COLORS.gutter; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(L.gutter * W, 0); ctx.lineTo(L.gutter * W, H); ctx.stroke(); }
    ctx.strokeStyle = COLORS.margin; ctx.lineWidth = 1;
    outlines(ctx, L.margin_quads, L.margins, W, H, 3);
    ctx.strokeStyle = COLORS.column;
    if (L.column_quads && L.column_quads.length) L.column_quads.forEach((q) => poly(ctx, q, W, H));
    else for (const b of L.columns) rect(ctx, [b[0], 20 / 720, b[2], 1 - 20 / 720], W, H);
    L.blocks.forEach((b, i) => {
      ctx.strokeStyle = COLORS.block; ctx.lineWidth = 2;
      const quad = L.block_quads && L.block_quads[i];
      if (quad) poly(ctx, quad, W, H); else rect(ctx, b, W, H);
      const bx = Math.max(14, (quad ? Math.min(...quad.map((p) => p[0])) : b[0]) * W - 20),
            by = (quad ? Math.min(...quad.map((p) => p[1])) : b[1]) * H + 12;
      ctx.fillStyle = COLORS.badge; ctx.beginPath(); ctx.arc(bx, by, 14, 0, 7); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "bold 15px sans-serif"; ctx.textAlign = "center";
      ctx.fillText(String(i + 1), bx, by + 5); ctx.textAlign = "left";
    });
    // The line and word being spoken
    const pos = speechPos;
    const span = pos === null ? null : lineAt(pos);
    if (span) {
      ctx.fillStyle = COLORS.reading;
      if (span.quad) poly(ctx, span.quad, W, H, true);
      else ctx.fillRect(span.box[0] * W - 4, span.box[1] * H - 3, (span.box[2] - span.box[0]) * W + 8, (span.box[3] - span.box[1]) * H + 6);
      const word = span.words.find((w) => pos < w[1]);
      if (word) { ctx.strokeStyle = COLORS.word; ctx.lineWidth = 2; if (word[3]) poly(ctx, word[3], W, H); else rect(ctx, word[2], W, H, 3); }
    }
  }
  const bar = 40;
  if (uploadView) drawPhotoQuality(ctx, W, H, H - bar - 10);
  ctx.fillStyle = PLATE; ctx.fillRect(0, H - bar, W, bar);
  const s = bookState;
  ctx.fillStyle = COLORS.text; ctx.font = "bold 16px sans-serif";
  ctx.fillText(`BOOK MODE   ${pageLabel || "Page -"}   ${bookBusy ? "READING PAGE..." : uploadView ? "UPLOADED PHOTO" : s ? s.state : "WAITING"}   ` +
    (s && !uploadView ? `motion ${s.motion} (still < ${s.still_below})   ` : "") + "B = exit", 15, H - 14);
}

function render() {
  const ctx = overlay.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = overlay.width / dpr, H = overlay.height / dpr;
  ctx.clearRect(0, 0, W, H);
  showShutter();
  if ((cameraReady || uploadView) && W > 0) { if (mode === "book") drawBook(ctx, W, H); else drawNormal(ctx, W, H); }
  requestAnimationFrame(render);
}

startCamera();
liveLoop();
render();
