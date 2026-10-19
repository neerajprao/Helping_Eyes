"""
Photo intake for the evaluation: a small local page that saves photos into eval/data/private/images/ and adds a row
to eval/data/manifest.csv for each.

    .venv/bin/python eval/tools/collect.py              then open http://localhost:7871
    .venv/bin/python eval/tools/collect.py --lan        also reachable from a phone on the same Wi-Fi (upload only:
                                                        a phone camera needs https, but picking or taking a photo works)

Three ways in:
  * Take photo   the laptop webcam, captured exactly like the app's Capture now (up to 1920 px, JPEG quality 0.92)
  * Add files    phone photos (JPEG, PNG or HEIC; several at once). They share the details chosen on the form.
  * A folder     eval/tools/import_folder.py imports a whole folder at once; the details are then set on this page
                 (every photo has its own dropdowns; photos still missing details are listed first, outlined in red).

Every photo is stored as <id>.jpg: EXIF rotation applied, all metadata (GPS location, camera, time) removed, JPEG
quality 95. The id is a category prefix and a number (ms01, mb01, fl01, bp01, dc01, ss01). An exact copy of a photo
already collected is refused. Details can be fixed later by editing eval/data/manifest.csv.

Nothing here runs OCR or shows recognised text, so the ground truth typed later cannot be influenced by an engine.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # eval/
DATA = os.path.join(ROOT, "data")
PRIVATE = os.path.join(DATA, "private")
IMAGES = os.path.join(PRIVATE, "images")
INTAKE = os.path.join(PRIVATE, "intake.json")          # original file name and hash of each photo (private)
MANIFEST = os.path.join(DATA, "manifest.csv")

FIELDS = ["id", "category", "lighting", "angle", "device", "notes"]
CATEGORIES = {"medicine_strip": "ms", "medicine_box": "mb", "food_label": "fl", "book_page": "bp",
              "document": "dc", "sign_or_screen": "ss"}
LIGHTING = ["good", "dim", "glare"]
ANGLE = ["flat", "tilted", "curved"]
DEVICE = ["laptop webcam", "phone"]

_lock = threading.Lock()
app = FastAPI(docs_url=None, redoc_url=None)


def read_manifest() -> List[dict]:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: List[dict]) -> None:
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_intake() -> dict:
    if not os.path.exists(INTAKE):
        return {}
    with open(INTAKE, encoding="utf-8") as f:
        return json.load(f)


def write_intake(data: dict) -> None:
    with open(INTAKE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def next_id(category: str, rows: List[dict]) -> str:
    prefix = CATEGORIES.get(category, UNSET)
    used = {r["id"] for r in rows}
    n = 1
    while f"{prefix}{n:02d}" in used:
        n += 1
    return f"{prefix}{n:02d}"


def _is_heic(data: bytes, name: str) -> bool:
    return name.lower().endswith((".heic", ".heif")) or data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1", b"ftyphevc")


def to_clean_image(data: bytes, name: str) -> Image.Image:
    """The photo as an upright RGB image with no metadata. HEIC is converted with macOS's own `sips`."""
    if _is_heic(data, name):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in.heic"), os.path.join(tmp, "out.jpg")
            with open(src, "wb") as f:
                f.write(data)
            subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "best", src, "--out", dst],
                           check=True, capture_output=True)
            with open(dst, "rb") as f:
                data = f.read()
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)          # upright, as a browser shows it
    except Exception as e:
        raise HTTPException(400, f"{name}: not a readable image ({e})")
    return image.convert("RGB")                         # a fresh image: no EXIF, GPS or other metadata is carried over


UNSET = "xx"      # id prefix of a photo whose category is not chosen yet (imported from a folder)


def save_photo(data: bytes, name: str, category: str = "", lighting: str = "", angle: str = "", device: str = "",
               notes: str = "") -> Tuple[Optional[str], str]:
    """Clean and store one photo and add its manifest row. Details may be left empty ("") and filled in later on the
    page. Returns (id, "") or (None, why it was skipped)."""
    digest = hashlib.sha256(data).hexdigest()
    image = to_clean_image(data, name)
    with _lock:
        rows, intake = read_manifest(), read_intake()
        same = next((k for k, v in intake.items() if v.get("sha256") == digest), None)
        if same:
            return None, f"{name}: the same file is already saved as {same}"
        pid = next_id(category, rows)
        image.save(os.path.join(IMAGES, pid + ".jpg"), "JPEG", quality=95)
        rows.append({"id": pid, "category": category, "lighting": lighting, "angle": angle, "device": device,
                     "notes": " ".join(notes.split())})
        write_manifest(rows)
        intake[pid] = {"original_name": name, "sha256": digest, "bytes": len(data),
                       "size": list(image.size), "added": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_intake(intake)
    return pid, ""


@app.post("/api/add")
async def add(files: List[UploadFile] = File(...), category: str = Form(...), lighting: str = Form(...),
              angle: str = Form(...), device: str = Form(...), notes: str = Form("")) -> dict:
    if category not in CATEGORIES or lighting not in LIGHTING or angle not in ANGLE or device not in DEVICE:
        raise HTTPException(400, "Choose a category, lighting, angle and device.")
    added, skipped = [], []
    for upload in files:
        pid, why = save_photo(await upload.read(), upload.filename or "", category, lighting, angle, device, notes)
        (added if pid else skipped).append(pid or why)
    return {"added": added, "skipped": skipped}


@app.post("/api/update/{pid}")
def update(pid: str, category: str = Form(""), lighting: str = Form(""), angle: str = Form(""), device: str = Form(""),
           notes: str = Form("")) -> dict:
    """Change a photo's details. A new category gives the photo a new id with that category's prefix."""
    if (category and category not in CATEGORIES) or (lighting and lighting not in LIGHTING) or \
            (angle and angle not in ANGLE) or (device and device not in DEVICE):
        raise HTTPException(400, "Not an allowed value.")
    with _lock:
        rows = read_manifest()
        row = next((r for r in rows if r["id"] == pid), None)
        if row is None:
            raise HTTPException(404, "No such photo.")
        new_id = pid
        if category != row["category"]:
            new_id = next_id(category, rows)
            os.rename(os.path.join(IMAGES, pid + ".jpg"), os.path.join(IMAGES, new_id + ".jpg"))
            intake = read_intake()
            if pid in intake:
                intake[new_id] = intake.pop(pid)
            write_intake(intake)
        row.update(id=new_id, category=category, lighting=lighting, angle=angle, device=device,
                   notes=" ".join(notes.split()))
        write_manifest(rows)
    return {"id": new_id}


@app.get("/api/items")
def items() -> list:
    return list(reversed(read_manifest()))


@app.post("/api/delete/{pid}")
def delete(pid: str) -> dict:
    with _lock:
        rows = read_manifest()
        if pid not in {r["id"] for r in rows}:
            raise HTTPException(404, "No such photo.")
        write_manifest([r for r in rows if r["id"] != pid])
        intake = read_intake()
        intake.pop(pid, None)
        write_intake(intake)
        path = os.path.join(IMAGES, pid + ".jpg")
        if os.path.exists(path):
            os.remove(path)
    return {"deleted": pid}


@app.get("/img/{pid}")
def image(pid: str) -> FileResponse:
    path = os.path.join(IMAGES, os.path.basename(pid) + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-store"})


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo intake</title><style>
:root{color-scheme:light dark;--bg:#fff;--fg:#111;--muted:#666;--line:#ccc;--accent:#0b5fff}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#999;--line:#444;--accent:#6ea0ff}}
body{font:15px/1.4 system-ui,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--fg);max-width:1100px}
h1{font-size:20px;margin:0 0 4px}.muted{color:var(--muted)}fieldset{border:1px solid var(--line);border-radius:8px;margin:10px 0}
label{display:inline-block;margin:4px 14px 4px 0}select,input[type=text]{font:inherit;padding:4px}
input[type=text]{width:min(520px,100%)}button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);
background:var(--accent);color:#fff;cursor:pointer;margin:4px 6px 4px 0}button.plain{background:transparent;color:var(--fg)}
video{width:min(480px,100%);border-radius:6px;background:#000;display:none}#msg{font-weight:600;min-height:1.4em}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:10px}
.card{border:1px solid var(--line);border-radius:6px;padding:6px;font-size:12px}.card img{width:100%;height:110px;object-fit:cover;border-radius:4px}
.warn{background:#fff3cd;color:#5c4400;padding:8px 10px;border-radius:6px}
</style></head><body>
<h1>Helping Eyes evaluation: photo intake</h1>
<p class="warn">Before each photo: no names (prescriptions, pharmacy stickers), no addresses, no phone numbers, no faces, no card or account numbers. Cover them or pick another item.</p>
<fieldset><legend>Details (apply to the next photo or files)</legend>
<label>Category <select id="category">CATS</select></label>
<label>Lighting <select id="lighting">LIGHT</select></label>
<label>Angle <select id="angle">ANGLE</select></label>
<label>Device <select id="device">DEV</select></label><br>
<label style="width:100%">Notes (no personal details) <input type="text" id="notes" placeholder="e.g. small print, embossed foil, expired item"></label>
</fieldset>
<fieldset><legend>Add</legend>
<button id="cam" class="plain">Start webcam</button><button id="shot" disabled>Take photo</button>
<label><input type="file" id="files" accept="image/*,.heic,.heif" multiple> phone photos</label><button id="upload">Add files</button>
<br><video id="video" autoplay playsinline muted></video>
</fieldset>
<div id="msg"></div><div class="muted" id="count"></div><div id="grid"></div>
<script>
const $ = (id) => document.getElementById(id);
let stream = null;
function form(extra) {
  const f = new FormData();
  for (const k of ["category", "lighting", "angle", "device", "notes"]) f.append(k, $(k).value);
  return f;
}
async function send(f) {
  const r = await fetch("/api/add", {method: "POST", body: f});
  const d = await r.json();
  if (!r.ok) { $("msg").textContent = d.detail || "Failed"; return; }
  $("msg").textContent = (d.added.length ? "Saved " + d.added.join(", ") + ". " : "") + d.skipped.join(" ");
  load();
}
$("cam").onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 1920}, height: {ideal: 1080}}});
  $("video").srcObject = stream; $("video").style.display = "block"; $("shot").disabled = false;
  $("device").value = "laptop webcam";
};
$("shot").onclick = () => {          // the same as the app's Capture now: longer side at most 1920 px, JPEG 0.92
  const v = $("video"), scale = Math.min(1, 1920 / Math.max(v.videoWidth, v.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(v.videoWidth * scale); c.height = Math.round(v.videoHeight * scale);
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  c.toBlob((b) => { const f = form(); f.append("files", b, "webcam.jpg"); send(f); }, "image/jpeg", 0.92);
};
$("upload").onclick = () => {
  if (!$("files").files.length) { $("msg").textContent = "Pick one or more photos first."; return; }
  const f = form(); for (const file of $("files").files) f.append("files", file, file.name);
  $("msg").textContent = "Saving..."; send(f).then(() => { $("files").value = ""; });
};
const CHOICES = {category: CATS_JSON, lighting: LIGHT_JSON, angle: ANGLE_JSON, device: DEV_JSON};
const done = (r) => r.category && r.lighting && r.angle && r.device;
async function load() {
  const rows = await (await fetch("/api/items")).json();
  const by = {}; rows.forEach((r) => by[r.category || "(no category)"] = (by[r.category || "(no category)"] || 0) + 1);
  const hard = rows.filter((r) => r.lighting === "dim" || r.lighting === "glare" || r.angle === "curved").length;
  const todo = rows.filter((r) => !done(r)).length;
  $("count").textContent = rows.length + " photos (" + Object.entries(by).map(([k, v]) => k + " " + v).join(", ") +
    "); " + hard + " difficult (dim, glare or curved); " + (todo ? todo + " still need details (shown first)" : "all details filled in");
  $("grid").innerHTML = "";
  rows.sort((a, b) => done(a) - done(b));
  for (const r of rows) {
    const d = document.createElement("div"); d.className = "card"; if (!done(r)) d.style.borderColor = "#d33";
    d.innerHTML = `<a href="/img/${r.id}" target="_blank"><img loading="lazy" src="/img/${r.id}?t=${Date.now()}"></a><b>${r.id}</b><br>`;
    const fields = {};
    for (const [k, values] of Object.entries(CHOICES)) {
      const s = document.createElement("select"); s.style.width = "100%";
      s.innerHTML = `<option value="">${k}?</option>` + values.map((v) => `<option>${v}</option>`).join("");
      s.value = r[k] || ""; fields[k] = s; d.appendChild(s);
    }
    const n = document.createElement("input"); n.type = "text"; n.placeholder = "notes"; n.value = r.notes || "";
    n.style.width = "100%"; fields.notes = n; d.appendChild(n);
    const save = async () => {
      const f = new FormData(); for (const [k, el] of Object.entries(fields)) f.append(k, el.value);
      const res = await fetch("/api/update/" + r.id, {method: "POST", body: f});
      const out = await res.json(); $("msg").textContent = res.ok ? "Saved " + out.id : (out.detail || "Failed");
      if (res.ok && (out.id !== r.id || done(Object.fromEntries(Object.entries(fields).map(([k, el]) => [k, el.value]))) !== done(r))) load();
      else if (res.ok) Object.entries(fields).forEach(([k, el]) => r[k] = el.value);
    };
    Object.values(fields).forEach((el) => el.onchange = save);
    const del = document.createElement("button"); del.className = "plain"; del.textContent = "Delete";
    del.onclick = async () => { if (confirm("Delete " + r.id + "?")) { await fetch("/api/delete/" + r.id, {method: "POST"}); load(); } };
    d.appendChild(del); $("grid").appendChild(d);
  }
}
load();
</script></body></html>"""


def _options(values) -> str:
    return "".join(f'<option value="{v}">{v}</option>' for v in values)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    page = PAGE
    for key, values in (("CATS", list(CATEGORIES)), ("LIGHT", LIGHTING), ("ANGLE", ANGLE), ("DEV", DEVICE)):
        page = page.replace(key + "_JSON", json.dumps(values)).replace(key, _options(values))
    return page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--lan", action="store_true", help="listen on the local network too (for a phone)")
    args = parser.parse_args()
    os.makedirs(IMAGES, exist_ok=True)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    print(f"Photo intake: http://localhost:{args.port}" + ("  (and this Mac's Wi-Fi address from a phone)" if args.lan else ""),
          file=sys.stderr)
    uvicorn.run(app, host=host, port=args.port, log_level="warning")
