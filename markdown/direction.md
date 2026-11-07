# Buttons and voice commands

What each button on the page does, and what you can say or type.

If the app was shared with an access code, the first page asks for it: type the code and press **Enter** (or the Enter key). The browser remembers it for 30 days.

## Buttons

| Button | Key | What it does |
|---|---|---|
| **Capture now** (and the round **shutter button** on the camera picture) | `C` | Takes a photo of what the camera sees right now, with a flash, and reads the text on it, whatever the spoken hints say. The photo then stays on the screen and the camera is switched off until you press **New item** or say "next" (if no text is found, the camera comes straight back). In book mode it is called **Read this page** and reads the page in view. On an uploaded photo in book mode, it reads that photo again. |
| **New item** | `N` | Forgets the current item and goes back to the live camera (the photo that stayed on the screen, or an uploaded photo, is closed). Ready for the next thing. Same as saying "next". |
| **Book mode** | `B` | Switches between normal mode (one item) and book mode (page by page, reading aloud). Same as saying "book mode" / "normal mode". In book mode the page shows only the camera and, beside it, **Book mode**, **Upload photo**, then **Stop** and **Continue** last; everything else comes back when book mode is turned off. |
| **Upload photo** | – | Pick a picture from your device instead of using the camera. It is shown where the camera preview is, with sharpness, light and glare info. In book mode it is read like a camera page, with the reading overlay and highlight. |
| **Uploaded photo is a two-page spread** | – | Tick it before uploading a photo of two facing pages, in normal mode, so they are split and read in order. |
| **Back to camera** | – | Appears while an uploaded photo is shown (the camera is switched off meanwhile). Switches the camera back on and returns to the live view. |
| **Answers from** menu | – | Chooses who answers your questions: *Online AI (Gemini)*, or *Offline AI on this computer (Qwen)*, Qwen 3B running on this Mac through Ollama. Your choice is remembered. |
| **What can I say?** | – | Opens a short list of the things you can say or type (hidden on short laptop screens). |
| **Settings** (top right) | – | *My screen reader reads everything*: the app's own voice stays quiet and your screen reader (or braille display) gets everything instead. *Single-key shortcuts*: switch the one-letter keys (C, N, B, A, R, S, L) off if they clash with your screen reader. Both are remembered. |
| **Ask** | `Enter` in the box | Sends the question you typed. |
| **Listen** | `L` | Turns the microphone on or off; the app says "Listening" or "Listening is off". The button is filled, with a check mark, while the microphone is really open: it goes off while the app is talking or thinking and turns itself on again when the app is quiet. Press it while the app is talking to stop the talking and listen straight away. |
| **Read everything** | `A` | Reads all the captured text aloud. Same as saying "read everything". |
| **Repeat** | `R` | Says the last answer again. |
| **Stop** | `S` | Stops speaking straight away. In book mode the app keeps the place where reading stopped (and does not say "Turn the page."). |
| **Continue** | – | Book mode only. Reads the rest of the page from the start of the sentence where reading was stopped (by Stop or by a question). It uses the page already read, so the book does not need to be in view. At the end of a page it says so. |
| **Yes, look it up** / **No** | – | Appear when the text doesn't contain the answer. Yes searches the web for it; No drops it. |

Keys work when the cursor is not in the question box.

## What you can say or type

**Reading**
- "Read everything", "read it", "read this", "what does it say", "what is written"
- "Read only the dosage" (or any part you want)

**Questions about the text**
- "When does it expire?", "What is the batch number?", "Is this safe for children?", "What is the price?"
- Answers come only from the captured text.

**Controls**
- "Stop", "be quiet", "cancel": stop speaking
- "Repeat", "say that again", "again", "pardon": hear the last answer again
- "Next", "next page", "new item", "scan again", "capture again": start a new item
- "Book mode", "reading mode", "read a book": start book mode
- "Normal mode", "exit book mode": back to one item at a time

**Web lookup** (only when you agree)
- After "The text doesn't say. Should I look it up online?": any wording that means yes ("yes", "go ahead", "why not", "sounds good", "search the web for it") or no ("no", "not really", "never mind"). Unusual wording is understood by the language model.
- Any request to search ("look it up", "check that on the internet", "google that", "find out more online") searches for your last question.
- "Search the web for …", "look up …", "google …", "find me … online": search for that phrase

## In book mode

- The page shows only the camera and, on the right (below the camera on a phone), **Book mode** (press it again to leave book mode), **Upload photo** (a photo of a page, read like a camera page; **Back to camera** appears under it while the photo is shown), then **Stop** and **Continue** (reads on from the sentence where reading was stopped). The Ask and Answer cards and the buttons under the camera are hidden until you leave book mode. Their keys still work (`C` reads the page in view, `L` turns listening on or off, `A`, `R`, `N`), and so do the voice commands.
- Once a page is read, the camera view zooms in on the book: the part inside the red border (with a little room round it) is enlarged to fill the view and everything around it is blacked out, so whoever watches can read the words. The status bar keeps its own strip at the bottom, so it never covers the page. Before the first page, when no text is found, and in normal mode, the whole camera picture is shown. Only the screen is zoomed: the app still looks at the whole camera picture.
- Hold the book open and still; each page is read when it settles.
- A red border is drawn round the open book and a yellow line down its spine, between the two pages. The left page is read first, then the right one.
- Turn the page and the next one is read. The app says "Turn the page." when it finishes one.
- The page number and running title are announced before the page.
- The line and word being spoken are highlighted on the picture.
- Move the book mid-page and it carries on from the word it reached.
- When the page opens the browser asks to use the camera and the microphone: allow both (listening itself starts only when you press **Listen**, or the `L` key in book mode, where the button is hidden).
- The voice is Sarah (an American female voice) made on your Mac, so it works without internet. If it is not set up, a free online voice speaks, and the browser's own voice is the last resort.
- The language model groups the lines of a page into paragraphs before it is read, so a sentence that runs over a line, a column or a page is read without a pause, and a sidebar or box is read after the main text. It also reads the page for sense and corrects words the camera misread ("emplovee" is read as "employee"); a correction is used only when it stays close to what is printed. If the model cannot answer, the page is read as laid out, with the words as the camera read them. This works with the online AI (Gemini); the offline Qwen model is usually too slow for a whole book page.
