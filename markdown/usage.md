# Using Helping Eyes

Everything you can say or type, and the buttons and keys. The short version is in [direction.md](direction.md). Back to the [README](../README.md).

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
| `C` | Capture now (also the round shutter button on the camera picture) | Capture immediately, with a flash so you see it was taken: the photo stays on the screen and the camera goes off until New item or "next" (in book mode: the photo of the page stays on the screen while it is read, until it has been read to the end or New item is pressed); a second press waits while a photo is being read |
| `N` | New item | Close the photo that stayed on the screen (or an uploaded one) and go back to the live camera, ready for a new item. In book mode (where it stays beside the camera): forget the page and read the next steady view from the start, for another book or document |
| `B` | Book mode | Toggle book mode |
| `A` | Read everything | Read the full captured text |
| `R` | Repeat | Repeat the last answer |
| `S` | Stop | Stop speaking |
| – | Continue | Book mode only: read the rest of the page from the sentence where reading was stopped |
| `L` | Listen | Turn hands-free listening on or off (the app says "Listening" / "Listening is off"). The button is filled with a check mark while the microphone is open right now: it goes off while the app talks or thinks and turns itself on again when it is quiet. Pressing it while the app talks stops the talking and listens at once |

On a laptop-sized window (at least 1000 × 560) the whole page fits the screen without scrolling: the camera takes the left three quarters and shrinks to the height that is left, the Ask and Answer cards share the last quarter, and the answer scrolls inside its own box. Phones keep the normal scrolling page, and at 200 % browser zoom the page switches to that single column.

**Look and accessibility:** all text, including every button label, is at least 7:1 contrast against its background (the answer is the largest text, 20 to 21 px), and the edges of the question box and menu at least 3:1, in both the light and the dark theme. Buttons are at least 48 px tall on a phone and 44 px on a laptop, and the keyboard focus ring is 3 px with a gap. A toggle that is on (Listen, Book mode) is filled and shows a check mark, so its state is not shown by colour alone. The page follows the system settings for reduced motion (no transitions), more contrast (solid borders), reduced transparency (a solid top bar) and Windows High Contrast. The overlay's labels sit on a dark plate, so they stay readable on any camera picture. Text sizes follow the browser's own text-size setting. The round shutter is a large touch target only: keyboard and screen-reader users meet the same action once, as **Capture now**. The answer box can be reached with Tab and scrolled with the arrow keys. Single-letter shortcuts do not fire while you type in the question box or choose in the menu. When the "look it up online?" offer closes, the keyboard focus moves to the question box. **What can I say?** under the Ask buttons lists the voice commands (hidden on short laptop screens). Every problem (camera blocked, microphone blocked, no text, the server unreachable or the live connection lost and back) is said aloud as well as shown. axe-core reports no issues, including the AAA contrast rule.

**Settings** (top right; remembered by the browser):

- *My screen reader reads everything:* the app's own voice stays quiet and everything it would say (guidance, answers, book pages, problems) is passed to the screen reader or braille display through one live region; the microphone waits until the screen reader has probably finished. Off by default: then the app speaks, and the status line and answer box are not live regions, so a screen reader never repeats what the app has just said.
- *Single-key shortcuts:* on by default; switching them off (WCAG 2.1.4) stops C, N, B, A, R, S and L and hides their key hints, for screen-reader users whose own quick keys use the same letters.

The word being read is highlighted with the CSS Custom Highlight API, so the answer text itself is never rewritten while it is spoken (a `<mark>` is the fallback in older browsers).

Upload photo shows the picture where the camera preview is (the camera is switched off while the photo is shown; Back to camera, or New item, switches it back on and returns to the live view). In book mode the uploaded page is read like a camera page: the same overlay (spine, columns, paragraph order) is drawn on it and the line and word being spoken are highlighted. In normal mode, tick the box beside it for a two-page spread. The photo's sharpness, light and glare are shown on it as on the live view, as numbers and a hint only, nothing is drawn on the picture (`POST /api/quality`).

The microphone listens only while the application is silent, so that it does not capture its own voice. Use `S` to interrupt a long answer.

**Overlay states:** `SHOW TEXT HERE` → `ADJUST POSITION` → `HOLD` → `CAPTURING...` → `CAPTURED - ASK ME`, and `THINKING...` / `SPEAKING...` while answering. In book mode the overlay shows the spine, columns, numbered paragraphs, and the line and word being read.
