# Buttons and voice commands

What each button on the page does, and what you can say or type.

## Buttons

| Button | Key | What it does |
|---|---|---|
| **Capture now** (and the round **shutter button** on the camera picture) | `C` | Takes a photo of what the camera sees right now, with a flash, and reads the text on it, whatever the spoken hints say. The photo then stays on the screen and the camera is switched off until you press **New item** or say "next" (if no text is found, the camera comes straight back). In book mode it is called **Read this page** and reads the page in view. On an uploaded photo in book mode, it reads that photo again. |
| **New item** | `R` | Forgets the current item and goes back to the live camera (the photo that stayed on the screen, or an uploaded photo, is closed). Ready for the next thing. Same as saying "next". |
| **Book mode** | `B` | Switches between normal mode (one item) and book mode (page by page, reading aloud). Same as saying "book mode" / "normal mode". |
| **Upload photo** | – | Pick a picture from your device instead of using the camera. It is shown where the camera preview is, with sharpness, light and glare info. In book mode it is read like a camera page, with the reading overlay and highlight. |
| **Uploaded photo is a two-page spread** | – | Tick it before uploading a photo of two facing pages, in normal mode, so they are split and read in order. |
| **Back to camera** | – | Appears while an uploaded photo is shown (the camera is switched off meanwhile). Switches the camera back on and returns to the live view. |
| **Model** menu | – | Chooses who answers your questions: Gemini (online) or Qwen 3B (running on this Mac through Ollama). Your choice is remembered. |
| **Ask** | `Enter` in the box | Sends the question you typed. |
| **Listen** | `M` | Turns the microphone on or off. The button shows when the microphone is really open: it goes off while the app is talking or thinking and turns itself on again when the app is quiet. Press it while the app is talking to stop the talking and listen straight away. |
| **Read everything** | `A` | Reads all the captured text aloud. Same as saying "read everything". |
| **Repeat** | `P` | Says the last answer again. |
| **Stop** | `S` | Stops speaking straight away. |
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

- Hold the book open and still; each page is read when it settles.
- Turn the page and the next one is read. The app says "Turn the page." when it finishes one.
- The page number and running title are announced before the page.
- The line and word being spoken are highlighted on the picture.
- Move the book mid-page and it carries on from the word it reached.
- When the page opens the browser asks to use the camera and the microphone: allow both (listening itself starts only when you press **Listen**).
- The voice is Sarah (an American female voice) made on your Mac, so it works without internet. If it is not set up, a free online voice speaks, and the browser's own voice is the last resort.
- The language model groups the lines of a page into paragraphs before it is read, so a sentence that runs over a line, a column or a page is read without a pause, and a sidebar or box is read after the main text. If the model cannot answer, the page is read as laid out.
