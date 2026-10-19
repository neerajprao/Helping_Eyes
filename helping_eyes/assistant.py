"""
Question answering over captured text, using any chat language model that speaks the
OpenAI-style chat API (Gemini, Groq, OpenRouter, OpenAI, a local server, ...), with an
optional web lookup (DuckDuckGo) when the captured text doesn't contain the answer.

Settings (environment or .env); nothing here is specific to one provider:
    LLM_BASE_URL  chat API address (default: Google Gemini's OpenAI-compatible endpoint)
    LLM_API_KEY   the provider's API key
    LLM_MODEL     model name (default: gemini-3.1-flash-lite)

The page's model menu picks between two providers per request:
    gemini        the settings above
    qwen3b        Qwen 2.5 3B on this computer, served by Ollama (https://ollama.com)
    QWEN_BASE_URL  Ollama's OpenAI-compatible address (default: http://localhost:11434/v1)
    QWEN_3B_MODEL  the 3B model's name in `ollama list` (default: qwen2.5:3b-instruct)
"""

import json
import logging
import os
import re
import calendar
import threading
from datetime import date
from typing import Iterator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1").rstrip("/")
QWEN_MODELS = {"qwen3b": os.getenv("QWEN_3B_MODEL", "qwen2.5:3b-instruct")}
PROVIDERS = ("gemini", *QWEN_MODELS)

# The model replies with exactly this when the user wants everything read out;
# the app then speaks the OCR text itself instead of the model's version.
READ_ALL = "READ_ALL"

# The model starts its reply with this tag when the captured text doesn't hold
# the answer. The tag is never spoken; the app uses it to offer a web lookup.
NOT_IN_TEXT = "[NOT_IN_TEXT]"

SYSTEM_PROMPT = """You help a blind or low-vision person understand text that their camera captured.
The captured text below was extracted by OCR. It may contain recognition mistakes, broken lines,
or text from more than one object.

Rules:
- Your reply is spoken aloud by a text-to-speech voice. Use plain sentences only: no markdown,
  no bullet symbols, no tables, no emojis.
- If the user wants the whole text read out (for example "read everything", "read it all",
  "what does it say"), reply with exactly: READ_ALL
- If the user asks you to read only certain parts (for example the ingredients, the dosage,
  the expiry date, the address), read those parts exactly as written, with no extra commentary.
  Fix an obvious OCR mistake only when you are certain.
- If the user asks a question, answer it briefly, in one to three sentences, using only the
  captured text.
- If the captured text does not contain the answer, start your reply with the tag
  [NOT_IN_TEXT] and then say in one sentence that the text doesn't say. Do not answer from
  your own knowledge, even if you know the answer. Explaining what an ingredient or term
  means also counts as not in the text unless the text itself explains it.
  Examples:
    Question: Is it waterproof?  (the text doesn't mention water resistance)
    Reply: [NOT_IN_TEXT] The text doesn't say whether it is waterproof.
    Question: What is zinc oxide?  (the text only lists zinc oxide as an ingredient)
    Reply: [NOT_IN_TEXT] The text lists zinc oxide as an ingredient but doesn't explain what it is.
- Never invent details that are not in the text. Be precise with medicine, dosage, dates,
  prices and safety information.
- Today's date is {today}.
- For expiry questions, trust the "Date checks" below: they were computed by the app and are
  correct. Do not redo the date arithmetic yourself. Answer in your own natural, conversational
  words, in one or two sentences: say whether it has expired, and when it is or was valid until.
  Never say it has expired unless the Date checks say HAS EXPIRED. If there is no expiry date, say the text does not
  show one; never guess it from a manufacturing date.
- Questions about the page number, book title or chapter are answered from the "Page facts"
  below: they were read from the printed page numbers, header and headings and are reliable.

Date checks:
{date_checks}

Page facts:
{facts}

Captured text:
<<<
{document}
>>>"""

QUERY_PROMPT = """Write one short web search query (at most 8 words) that would answer the
user's question. Use the product or document name from the captured text if it helps.
Reply with the query only, no quotes.

Captured text (first part):
{document}

Question: {question}"""

WEB_PROMPT = """You help a blind or low-vision person. The text their camera captured did not
answer their question, so the app searched the web. Answer using only the web results below.

Rules:
- Your reply is spoken aloud. Plain sentences only, no markdown, no lists, no URLs.
- Say only what the question asks for: the specific fact, value or part they asked about.
  No background, no history, no related facts, no extra tips, and no summary of the results.
  If they ask for one thing (a price, a date, a dosage, an ingredient), give just that.
- Start with "According to the web," so the user knows this is not from their item.
- Keep it to one or two short sentences. If the results don't answer the question, say so in one sentence.
- For medicine, dosage or health questions only, end with: "Please confirm with a pharmacist or doctor."
- Never present web information as if it were printed on the user's item.

Question: {question}

Web results:
{results}"""

ARRANGE_PROMPT = """You prepare a page of a book to be read aloud to a blind person. Below are the lines the text
reader found. The app has already grouped them into blocks from the picture: the lines of a block sit one under
another with the same left edge. A block can be a column of the main text, a heading, a page number, or a
sidebar, box or caption beside or below the main text. Blocks are listed in the order their first line appears,
which is only a guess at the reading order.

Each row looks like:  12 (x62-98 y55 h0.7)[indent] the text of the line
- 12 is the line number. x62-98: the line spans from 62 % to 98 % of the page's width. y55: its top is 55 % of
  the way down. h0.7: its letters are 0.7 times the usual size (1.0 is the main text; a sidebar or caption is
  smaller).
- Tags are further hints and can be wrong: [page] first line of a new page of a two-page spread, [column] first
  line of a new column, [gap] extra space above, [indent] starts indented, [short] ends well before the right edge.

Decide the paragraphs and the order in which they are read.
- Reading order: the left page before the right page. On a page, read ALL the main text first, from top to
  bottom (column by column), up to its very last line. Only then read each sidebar, box or caption of that page.
  Never put a sidebar between two parts of the main text. A heading comes before the text under it.
- A paragraph normally uses lines of one block only. It uses lines of two blocks only when a sentence carries on
  from the end of one block into the start of another (for example from the bottom of one column to the top of the
  next): then they are one paragraph, so the voice does not pause there.
- Inside a block, a new paragraph begins where one really begins: after a sentence has ended and the next line is
  indented, or follows a gap, or follows a [short] line. A [short] line that ends with a full stop, question mark
  or exclamation mark almost always ends its paragraph. A heading, chapter title or list item is its own
  paragraph. A sentence that carries on over the next line belongs to the same paragraph, whatever the tags say.
- You only group and order lines. Never change, add, fix or reorder words inside a line.
- Put a line in "skip" only when it clearly is not part of the book's text: a page number, a running header or
  footer, a stray fragment from the text reader. When unsure, keep the line.
- Every line number appears exactly once, either in one paragraph or in "skip".

Reply with JSON only: the paragraphs in reading order, each a list of line numbers in reading order, for example
{"paragraphs": [[0, 1, 2], [3, 4], [5, 6, 7]], "skip": [8]}"""

ARRANGE_TIMEOUT = 12     # seconds the model gets to arrange a page before the page is read as it is

MSG_UNREACHABLE = "I can't reach the language model. Please check the internet connection."
MSG_NO_KEY = "No API key is set for the language model. Please set LLM_API_KEY."
MSG_BAD_KEY = "The language model rejected the request. Please check the API key."
MSG_NO_MODEL = "The language model name was not found. Please check LLM_MODEL and LLM_BASE_URL."
MSG_BUSY = "The language model is busy right now, or today's free limit is used up. Please try again in a minute."
MSG_QWEN_DOWN = "I can't reach the Qwen model on this computer. Please start Ollama, or choose Gemini."
MSG_FAILED = "Sorry, something went wrong while thinking about that."

# Split finished sentences out of a growing stream of text. A sentence only
# counts as finished once whitespace and a capital letter follow, so "0.5%"
# and "Rs. 25.00" aren't cut in the middle.
_SENTENCE_END = re.compile(r"(.+?[.!?])\s+(?=[A-Z\"'(])|([^\n]+?)\n+", re.S)

# Requests that clearly mean "read the whole thing": handled without the model
_READ_ALL_REQUEST = re.compile(
    r"^(?:please |can you |could you )*"
    r"(?:read(?: it| this| that| out| aloud| everything| all| the whole thing| the (?:whole|entire|full) (?:text|thing|page|label))*"
    r"|(?:read|say|tell me)(?: me)? (?:everything|all(?: of it)?|the (?:whole|entire|full) (?:text|thing|page|label))"
    r"|what does (?:it|this|that) say|what(?:'s| is) written(?: here| on it)?)"
    r"(?: (?:please|for me|out loud|aloud|again))*[?.!]*$"
)

# Facts that only the item itself can tell you: never offered as a web lookup
_ITEM_ONLY = re.compile(r"expir|exp\b|best before|use by|batch|lot\b|mfg|manufactur(ed|ing) date|price|mrp|cost", re.I)


def parse_arrangement(reply: str, count: int) -> Optional[Tuple[List[List[int]], List[int]]]:
    """
    The model's paragraphs and skipped lines from its reply, or None when the reply cannot be trusted: not JSON, a
    line missing or given twice, or too many lines skipped. Nothing may be lost by a wrong answer.
    """
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
        groups = [[int(i) for i in group] for group in data["paragraphs"]]
        skip = [int(i) for i in data.get("skip", [])]
    except (ValueError, TypeError, KeyError):
        return None
    if any(not group for group in groups):
        return None
    seen = [i for group in groups for i in group] + skip
    if sorted(seen) != list(range(count)) or not groups:
        return None
    if len(skip) > max(2, count // 4):
        return None
    return groups, skip


def wants_read_all(request: str) -> bool:
    """True for plain "read it all" requests, so they skip the model entirely."""
    return bool(_READ_ALL_REQUEST.match(request.strip().lower()))


# Backup for when the model forgets the [NOT_IN_TEXT] tag
_SAYS_NOT_IN_TEXT = re.compile(
    r"\b(text|label|it|this|they|directions|instructions|package|packaging)\s+"
    r"(does not|doesn't|do not|don't|did not|didn't)\s+"
    r"(say|mention|show|state|specify|include|contain|provide|explain|list|indicate)"
    r"|\bnot\s+(mentioned|stated|specified|provided|shown|listed|explained|included)\b"
    r"|\bno\s+(information|mention|details?)\s+(about|on|of|regarding)\b",
    re.I,
)


def web_lookup_allowed(question: str) -> bool:
    """False for questions about this particular item (expiry, batch, price)."""
    return not _ITEM_ONLY.search(question)


# ---------------- expiry dates (computed in code; small models get these wrong) ----------------
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
# "Exp", "Exp.", "Expiry Date", "Exp. Date", "Exp.Date", "Exp Dt.", "Use by", "Best before"
_EXPIRY_LABEL = (r"(?:exp(?:iry|iration|\.)?(?:\s*(?:date|dt)\b\.?)?|use\s*(?:by|before)|best\s*before)"
                 r"\.?\s*[:.\-]?\s*")
_EXPIRY_DATE = re.compile(
    _EXPIRY_LABEL + r"(?:"
    r"(?P<d>\d{1,2})[/\-.](?P<m>\d{1,2})[/\-.](?P<y>\d{2,4})"          # 15/04/2028
    r"|(?P<m2>\d{1,2})[/\-.](?P<y2>\d{2,4})"                            # 04/2028, 04-28
    # APR 2028, Apr-28; only real month names, so the "Dat" of "Date" is never taken for a month
    r"|(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s/\-.]*(?P<y3>\d{2,4})"
    r")",
    re.I,
)


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def expiry_dates(text: str) -> List[Tuple[str, date]]:
    """(printed text, last valid day) for every expiry date found in the text."""
    found = []
    for m in _EXPIRY_DATE.finditer(text):
        try:
            if m.group("d"):
                exp = date(_year(m.group("y")), int(m.group("m")), int(m.group("d")))
            else:
                if m.group("m2"):
                    month, year = int(m.group("m2")), _year(m.group("y2"))
                else:
                    month, year = _MONTHS.get(m.group("mon").lower()[:3]), _year(m.group("y3"))
                    if not month:
                        continue
                # Month-only dates are good until the end of that month
                exp = date(year, month, calendar.monthrange(year, month)[1])
        except ValueError:
            continue
        found.append((m.group(0).strip(), exp))
    return found


def expiry_checks(text: str, today: date = None) -> str:
    """One line per expiry date found, saying whether it has passed."""
    today = today or date.today()
    lines = [f'Printed "{printed}" = valid until {exp.strftime("%d %B %Y")}. Today is {today.strftime("%d %B %Y")}, '
             + ("which is AFTER that date, so it HAS EXPIRED." if exp < today
                else "which is BEFORE that date, so it has NOT expired and is still good.")
             for printed, exp in expiry_dates(text)]
    return "\n".join(lines) or "No expiry date found in the text."


# Questions with an exact answer are answered in code, not by the model
_EXPIRY_QUESTION = re.compile(r"\b(expir\w*|exp(iry)? date|best before|use by|out of date|still (good|valid|usable|okay|ok))\b", re.I)
_PAGE_QUESTION = re.compile(r"^\s*(what|which)\s+page(\s+(is\s+(this|it)|am\s+i\s+on|are\s+we\s+on))?\s*\??\s*$", re.I)


# ---------------- web search ----------------
def web_search(query: str, max_results: int = 5) -> List[dict]:
    """DuckDuckGo text search: [{title, href, body}, ...]. Only the query leaves the Mac."""
    from ddgs import DDGS  # imported here so the app still starts if ddgs is missing
    return DDGS(timeout=10).text(query, max_results=max_results) or []


def _why(response) -> str:
    """The provider's own explanation of an error (it says which model or key is wrong), shortened for the log."""
    try:
        return " ".join(response.text.split())[:300] or "(no details)"
    except Exception:
        return "(no details)"


def _endpoint(provider: str) -> Tuple[str, dict, str, str]:
    """(chat API address, headers, model name, API key) of a provider; read at call time."""
    if provider in QWEN_MODELS:
        return QWEN_BASE_URL, {"Content-Type": "application/json"}, QWEN_MODELS[provider], ""
    return (LLM_BASE_URL,
            {"Content-Type": "application/json", **({"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {})},
            LLM_MODEL, LLM_API_KEY)


class DocAssistant:
    """Keeps the captured text plus the conversation about it."""

    def __init__(self):
        self.document = ""
        self.facts = ""
        self.page_label = ""
        self.history: List[dict] = []
        self.provider = "gemini"       # which language model answers: one of PROVIDERS
        self.last_not_in_text = False  # last answer said the text doesn't have it
        self._generation = 0           # bumped by cancel(); stale streams stop early
        self._lock = threading.Lock()

    # ---------------- document ----------------
    def set_document(self, text: str, facts: str = "", page_label: str = "") -> None:
        """New capture: replace the text and forget the previous conversation.
        facts: page facts (page numbers, running title, headings) for book pages;
        page_label: e.g. "Pages 82 and 83", used to answer "what page is this?"."""
        with self._lock:
            self.document = text
            self.facts = facts
            self.page_label = page_label
            self.history = []
            self.last_not_in_text = False
            self._generation += 1

    def update_document(self, text: str, facts: str = "", page_label: str = "") -> None:
        """Same item, slightly different view: replace the text but keep the conversation."""
        with self._lock:
            self.document = text
            self.facts = facts
            self.page_label = page_label

    def set_provider(self, provider: str) -> None:
        """Choose the language model for the next answers (unknown names are ignored)."""
        if provider in PROVIDERS:
            self.provider = provider

    def has_document(self) -> bool:
        return bool(self.document.strip())

    def cancel(self) -> None:
        """Stop any answer currently being generated."""
        with self._lock:
            self._generation += 1

    # ---------------- model ----------------
    def ask(self, question: str) -> Iterator[str]:
        """
        Answer from the captured text, streamed sentence by sentence.
        Yields READ_ALL alone when the user wants the whole text read.
        Sets last_not_in_text when the text doesn't contain the answer.
        """
        with self._lock:
            generation = self._generation
            self.last_not_in_text = False

        # Exact answer from code (reliable with any model size). Expiry is not here: the model words that answer
        # itself, from the "Date checks" the code computes for it.
        direct = None
        if _PAGE_QUESTION.match(question) and self.page_label:
            direct = f"This is {self.page_label[0].lower() + self.page_label[1:]}."
        if direct:
            yield direct
            self._remember(question, direct, generation)
            return

        with self._lock:
            system = SYSTEM_PROMPT.format(document=self.document,
                                          today=date.today().strftime("%d %B %Y"),
                                          date_checks=expiry_checks(self.document),
                                          facts=self.facts or "None (not a book page).")
            messages = [{"role": "system", "content": system}] + self.history
            content = question
            if _EXPIRY_QUESTION.search(question):
                # The code works out the dates; the model only puts the verdict into its own words
                verdict = expiry_checks(self.document)
                content += (f"\n\n[Checked by the app, and certain: {verdict} Answer in your own natural words, in one or two "
                            "sentences, and do not mention this note.]")
            messages.append({"role": "user", "content": content})

        answer = []
        for sentence in self._stream(messages, generation, markers=(READ_ALL, NOT_IN_TEXT)):
            if sentence == READ_ALL:
                self._remember(question, READ_ALL, generation)
                yield READ_ALL
                return
            if sentence == NOT_IN_TEXT:
                self.last_not_in_text = True
                continue
            answer.append(sentence)
            yield sentence
        if answer and _SAYS_NOT_IN_TEXT.search(answer[0]):
            self.last_not_in_text = True
        self._remember(question, " ".join(answer), generation)

    def ask_web(self, question: str) -> Iterator[str]:
        """Search the web for the question and answer from the results."""
        with self._lock:
            generation = self._generation
            document = self.document

        query = self._search_query(question, document) or question
        logger.info(f"Searching the web for: {query}")
        try:
            results = web_search(query)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            yield "Sorry, I couldn't search online right now."
            return
        if generation != self._generation:
            return
        if not results:
            yield "I searched online but found nothing useful."
            return

        sources = "\n\n".join(f"{r.get('title', '')}\n{r.get('body', '')}" for r in results)
        messages = [{"role": "user", "content": WEB_PROMPT.format(question=question, results=sources)}]
        answer = []
        for sentence in self._stream(messages, generation):
            answer.append(sentence)
            yield sentence
        self._remember(question, " ".join(answer), generation)

    # ---------------- internals ----------------
    def _search_query(self, question: str, document: str) -> Optional[str]:
        """Ask the model for a short search query that includes the product name."""
        try:
            base, headers, model, _ = _endpoint(self.provider)
            r = requests.post(f"{base}/chat/completions", headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": QUERY_PROMPT.format(
                    document=document[:1500], question=question)}],
                "temperature": 0,
                "max_tokens": 30,
            }, timeout=30)
            r.raise_for_status()
            query = r.json()["choices"][0]["message"]["content"].strip().strip('"').splitlines()[0]
            return query[:120] or None
        except Exception as e:
            logger.warning(f"Couldn't build a search query: {e}")
            return None

    def arrange_lines(self, numbered: str, count: int) -> Optional[Tuple[List[List[int]], List[int]]]:
        """
        Book mode: ask the model how the numbered lines of a page form paragraphs (see ARRANGE_PROMPT).
        Returns (paragraphs, skipped lines) as parse_arrangement checks them, or None when there is no model to ask,
        it is slow or fails, or its reply is not usable: the caller then keeps the layout analysis' paragraphs.
        """
        base, headers, model, key = _endpoint(self.provider)
        if not key and self.provider not in QWEN_MODELS:
            return None
        try:
            r = requests.post(f"{base}/chat/completions", headers=headers, json={
                "model": model, "temperature": 0, "max_tokens": 2000,
                "messages": [{"role": "system", "content": ARRANGE_PROMPT},
                             {"role": "user", "content": numbered}]}, timeout=(5, ARRANGE_TIMEOUT))
            r.raise_for_status()
            plan = parse_arrangement(r.json()["choices"][0]["message"]["content"] or "", count)
        except Exception as e:
            logger.warning(f"The model could not arrange the page: {e}")
            return None
        if plan is None:
            logger.warning("The model's paragraphs for the page were not usable; reading the page as laid out")
        return plan

    def judge_intent(self, text: str, offer_pending: bool, subject: str = "") -> str:
        """
        What does this request mean for the web lookup? Replies "yes" / "no" (only when the app has just offered
        to look something up), "search" / "search: <topic>" (the user wants something searched online), or "other".
        """
        if offer_pending:
            task = (f'The app just asked: "Should I look up the answer to \'{subject}\' online?" The user replied. '
                    'Answer YES if the reply means go ahead / agree / search / look it up in any wording or language, '
                    'NO if it means decline, otherwise OTHER. Reply with that one word only.')
        else:
            task = ('Does the user ask the app to search the web or look something up online? If yes, reply '
                    '"SEARCH: " followed by the topic to search for, in a few words taken from their request '
                    '(example: "can you google ibuprofen side effects" -> "SEARCH: ibuprofen side effects"). '
                    'If they only say to search without a topic (example: "check that online"), reply just "SEARCH"'
                    + (f' (their previous question was "{subject}")' if subject else '') + '. Otherwise reply "OTHER".')
        base, headers, model, _ = _endpoint(self.provider)
        r = requests.post(f"{base}/chat/completions", headers=headers, json={
            "model": model, "temperature": 0, "max_tokens": 40,
            "messages": [{"role": "system", "content": task},
                         {"role": "user", "content": text}]}, timeout=20)
        r.raise_for_status()
        reply = (r.json()["choices"][0]["message"]["content"] or "").strip().strip(".\"'")
        low = reply.lower()
        if low.startswith("yes"):
            return "yes"
        if low.startswith("no") and not low.startswith("none"):
            return "no"
        if low.startswith("search"):
            topic = reply.partition(":")[2].strip()
            return "search: " + topic if topic.lower() not in ("", "it", "that", "this", "that one", "this one") else "search"
        return "other"

    def _stream(self, messages: List[dict], generation: int, markers=()) -> Iterator[str]:
        """
        Stream a chat reply sentence by sentence. If the reply starts with one
        of `markers`, that marker is yielded on its own first (READ_ALL ends
        the reply; other markers are stripped and the rest is streamed).
        """
        answer = ""
        buffer = ""
        decided = not markers  # whether the reply's opening marker (if any) is known
        base, headers, model, key = _endpoint(self.provider)
        try:
            with requests.post(
                f"{base}/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages, "stream": True, "temperature": 0.2},
                stream=True,
                timeout=(5, 120),
            ) as r:
                if not r.ok:
                    r.content                                  # read the error body now: it is gone once the connection closes
                r.raise_for_status()
                for raw in r.iter_lines():
                    if generation != self._generation:
                        logger.info("Answer cancelled")
                        return
                    line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue                               # blank keep-alive lines, comments
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    choices = json.loads(payload).get("choices") or []
                    choice = choices[0] if choices else {}
                    piece = (choice.get("delta") or {}).get("content") or ""
                    finished = bool(choice.get("finish_reason"))
                    answer += piece
                    buffer += piece

                    if not decided:
                        head = answer.lstrip()
                        found = next((m for m in markers if head.startswith(m)), None)
                        if found == READ_ALL:
                            yield READ_ALL
                            return
                        if found:
                            yield found
                            buffer = head[len(found):].lstrip()
                            decided = True
                        elif any(m.startswith(head) for m in markers) and not finished:
                            continue  # could still turn into a marker
                        else:
                            decided = True

                    while True:
                        m = _SENTENCE_END.match(buffer)
                        if not m:
                            break
                        sentence = (m.group(1) or m.group(2) or "").strip()
                        buffer = buffer[m.end():]
                        if sentence:
                            yield sentence

                    if finished:
                        break
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            logger.error(f"Language model returned HTTP {status} for model {model}: {_why(e.response)}")
            if self.provider in QWEN_MODELS:
                yield MSG_NO_MODEL.replace("LLM_MODEL and LLM_BASE_URL", "the Qwen model name (and that Ollama has it)") if status == 404 else MSG_FAILED
            elif not key and status in (400, 401, 403):
                yield MSG_NO_KEY                               # providers answer a missing key with 400, 401 or 403
            else:
                yield (MSG_BAD_KEY if status in (401, 403) else MSG_NO_MODEL if status == 404
                       else MSG_BUSY if status == 429 else MSG_FAILED)
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            yield MSG_QWEN_DOWN if self.provider in QWEN_MODELS else MSG_UNREACHABLE
            return
        except Exception as e:
            logger.error(f"Model error: {e}")
            yield MSG_FAILED
            return

        if generation == self._generation and buffer.strip():
            yield buffer.strip()

    def _remember(self, question: str, answer: str, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self.history += [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}]
                self.history = self.history[-12:]  # last 6 exchanges
