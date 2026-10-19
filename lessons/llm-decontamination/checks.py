# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-decontamination: ngrams, the n-word runs of a text without punctuation; flag, which test
questions share more than a share of their n-grams with the training data; report, the score on all questions and on
the clean ones; and a choice. The demos take the course's story-cloze questions (data/story-cloze.json, from stories of
Cosmopedia) and the course's TinyStories (data/tinystories-small.txt): first the overlap the two have by themselves,
then a crawl that the course contaminates on purpose, with a look-up model that remembers every text it saw."""
import collections
import html
import json
import random
import time
from functools import cache

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

CLOZE = "data/story-cloze.json"        # the course's story-cloze questions: see data/README.md
STORIES = "data/tinystories-small.txt"   # 6,189 short stories of TinyStories, joined by <|endoftext|>
END = "<|endoftext|>"
PUNCT = ".,!?;:\"'()-"   # what ngrams strips from both ends of a word
N = 13                   # words per n-gram, as GPT-3 checked most of its benchmarks
SIZES = (5, 8, 13)       # the sizes the first demo compares
PLANTED = 50             # questions whose story the course copies into its contaminated crawl
KEY = 8                  # words of the end of a context the look-up model looks for


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


@cache
def cloze() -> dict:
    return json.loads((paths.ROOT / CLOZE).read_text(encoding="utf-8"))


def questions() -> list:
    """The 500 questions of stories no model of the course has seen."""
    return cloze()["items"]


def story_of(q: dict) -> str:
    """The text a question was made from: its context and the sentence that really follows."""
    return q["context"] + " " + q["options"][q["answer"]]


@cache
def stories() -> list:
    return (paths.ROOT / STORIES).read_text(encoding="utf-8").split(END)


@cache
def crawl() -> tuple:
    """The course's contaminated crawl: the stories of TinyStories, and the stories of PLANTED questions (drawn with
    seed 0) copied in between, each as a document of its own; and the places of those questions."""
    rng = random.Random(0)
    planted = sorted(rng.sample(range(len(questions())), PLANTED))
    docs = list(stories())
    for i in planted:
        docs.insert(rng.randrange(len(docs) + 1), story_of(questions()[i]))
    return tuple(docs), tuple(planted)


def _words(text: str) -> list:
    return [w for w in (w.strip(PUNCT) for w in text.lower().split()) if w]


def lookup_model(docs) -> list:
    """The look-up model of ml-train-test-split, for story cloze: it remembers every text it trained on. For a question,
    it looks for the last KEY words of the context in those texts; when it finds them, it picks the option that shares
    the most words with the five words that follow there, and else it guesses, with seed 0. Whether it is right, for
    every question."""
    follows = {}
    for doc in docs:
        w = _words(doc)
        for i in range(len(w) - KEY):
            follows.setdefault(" ".join(w[i:i + KEY]), set(w[i + KEY:i + KEY + 5]))
    rng = random.Random(0)
    right = []
    for q in questions():
        seen = follows.get(" ".join(_words(q["context"])[-KEY:]))
        if seen:
            shared = [len(set(_words(o)[:5]) & seen) for o in q["options"]]
            pick = max(range(len(shared)), key=lambda i: shared[i])
        else:
            pick = rng.randrange(len(q["options"]))
        right.append(pick == q["answer"])
    return right


def natural(ngrams, flag) -> dict:
    """The first demo: for every size of SIZES, how many questions the learner's flag marks, with share 0, against the
    stories of TinyStories, with the learner's ngrams; and, for each size, the n-gram that most questions share with them."""
    out = {}
    texts = [story_of(q) for q in questions()]
    for n in SIZES:
        train = set()
        for doc in stories():
            train |= set(ngrams(doc, n))
        grams = [set(ngrams(t, n)) for t in texts]
        flags = flag(grams, train, 0)
        shared = collections.Counter(x for g in grams for x in g & train)
        example = max(sorted(shared), key=shared.get) if shared else None   # the first of the most shared, in order
        out[n] = {"flagged": sum(bool(f) for f in flags), "example": example}
    return out


def contaminated(ngrams, flag, report) -> dict:
    """The second demo: the look-up model trained on the contaminated crawl, its score on all questions and on those
    the learner's flag does not mark (n = N, share 0), with the learner's report; the documents that decontaminating
    the crawl removes for every size of SIZES (every document that shares an n-gram with a question); and the look-up
    model's score after decontamination with n = N."""
    docs, planted = crawl()
    texts = [story_of(q) for q in questions()]
    train = set()
    for doc in docs:
        train |= set(ngrams(doc, N))
    flags = [bool(f) for f in flag([set(ngrams(t, N)) for t in texts], train, 0)]
    right = lookup_model(docs)
    removed, kept = {}, None
    for n in SIZES:
        test = set()
        for t in texts:
            test |= set(ngrams(t, n))
        keep = [doc for doc in docs if not set(ngrams(doc, n)) & test]
        removed[n] = len(docs) - len(keep)
        if n == N:
            kept = keep
    after = lookup_model(kept)
    return {"docs": len(docs), "planted": len(planted), "flagged": sum(flags),
            "planted_flagged": sum(flags[i] for i in planted), "report": report(flags, right),
            "removed": removed, "after": sum(after) / len(after)}


def flagged_picture(rows: list, labels: dict) -> str:
    """Bars of the questions flagged for every size of n-gram, on one scale of all questions, ink only: a measurement,
    not a role of a model. `rows`: (label, flagged); labels has "title", "of" (the text after a count, with {total}),
    "note", "alt" and "lang"."""
    lang = labels.get("lang", "en")
    total = len(questions())
    W, BAR, ROW, TOP = 330, 14, 40, 30
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = TOP
    for name, value in rows:
        w = max(2.0, (W - 110) * value / total)
        shown = number(value, 0, lang) + " " + labels["of"].format(total=number(total, 0, lang))
        p.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(name)}</text>'
                 f'<rect x="0" y="{y + 17}" width="{w:.1f}" height="{BAR}" rx="2" style="fill:var(--ea-ink)"/>'
                 f'<text x="{w + 6:.1f}" y="{y + 28.5}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(shown)}</text>')
        y += ROW
    p.append(f'<text x="0" y="{y + 10}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["note"])}</text>')
    H = y + 18
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


# --- measured: the course's runs with its own solutions (see the notes) ---
# The first demo (questions flagged against TinyStories, by size), the second (the crawl, the look-up model before
# and after decontamination, the documents removed by size), and the course's GPT on tokens of gpt-train-bpe
# (data/course-gpt-bpe.pt) on the questions of stories it never saw ("items") and of stories it trained on ("seen"):
# right answers of 500, when it picks the option whose log-probability after the context rises most above its
# log-probability after an end-of-text alone.
MEASURED = {"natural": {5: 423, 8: 148, 13: 0}, "docs": 6239, "flagged": 66, "planted_flagged": 50,
            "before": 0.322, "clean": 0.244, "after": 0.244, "removed": {5: 4591, 8: 1391, 13: 50},
            "gpt": {"items": 175, "seen": 165, "of": 500}}
# --- end of measured ---

REQUIRED = ["ngrams", "flag", "report", "after-training"]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...")
                                                          for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


def _shown(v) -> str:
    """A set in a fixed order, so that two sets with the same items look the same; floats without noise."""
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(repr(x) for x in sorted(v, key=repr)) + "}" if v else "set()"
    if isinstance(v, float):
        return repr(round(v, 4))
    if isinstance(v, (tuple, list)):
        inner = ", ".join(_shown(x) for x in v)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    return repr(v)


# --- exercise 1: ngrams ------------------------------------------------------------------------------------------------

_G_HEAD = f"PUNCT = {PUNCT!r}\n\n\ndef ngrams(text, n):\n    words = []\n    for w in text.lower().split():\n"
_G_WORD = "        w = w.strip(PUNCT)\n        if w:\n            words.append(w)\n"
_G_SHORT = ("    if len(words) < n:\n        if words:\n            return {\" \".join(words)}\n        return set()\n")
_G_LOOP = "    out = set()\n    for i in range(len(words) - n + 1):\n        out.add(\" \".join(words[i:i + n]))\n    return out\n"
_G_START = (_G_HEAD + "        # the word without punctuation at its start and end; an empty word does not count\n        ...\n"
            "    # a text shorter than n words: the whole text as one n-gram, or none when it has no words\n    ...\n"
            "    # every run of n words in a row, as one text with spaces between the words\n    ...\n")


def _g(word: str = _G_WORD, short: str = _G_SHORT, loop: str = _G_LOOP, head: str = _G_HEAD) -> str:
    return head + word + short + loop


# --- exercise 2: flag --------------------------------------------------------------------------------------------------

_F_BODY = ("def flag(test_grams, train_grams, share):\n    flags = []\n    for grams in test_grams:\n"
           "        if not grams:\n            flags.append(False)\n        else:\n"
           "            found = len(grams & train_grams)\n            flags.append(found / len(grams) > share)\n"
           "    return flags\n")
_F_START = ("def flag(test_grams, train_grams, share):\n    flags = []\n    for grams in test_grams:\n"
            "        # True when more than `share` of the question's n-grams are in the training data; False without n-grams\n"
            "        ...\n    return flags\n")

# --- exercise 3: report ------------------------------------------------------------------------------------------------

_R_BODY = ("def report(flags, right):\n    clean = []\n    for i in range(len(flags)):\n        if not flags[i]:\n"
           "            clean.append(right[i])\n    score = sum(right) / len(right)\n"
           "    if clean:\n        clean_score = sum(clean) / len(clean)\n    else:\n        clean_score = None\n"
           "    return score, clean_score, sum(flags) / len(flags)\n")
_R_START = ("def report(flags, right):\n    # whether the model was right on the questions that are not flagged: the clean subset\n"
            "    clean = ...\n    score = sum(right) / len(right)\n"
            "    # the share right on the clean subset, or None when no question is clean\n    clean_score = ...\n"
            "    return score, clean_score, sum(flags) / len(flags)\n")


def _same(got, want) -> bool:
    """The same outcome: a tuple or a list with the same items, numbers within 0.0001. The messages show numbers
    rounded to 4 decimals, so two numbers that are not the same never look the same."""
    if isinstance(got, (tuple, list)) and isinstance(want, (tuple, list)):
        return len(got) == len(want) and all(_same(g, w) for g, w in zip(got, want))
    if want is None or got is None:
        return got is None and want is None
    if isinstance(want, bool) or isinstance(got, bool):
        return type(got) is type(want) and got == want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(got - want) <= 1e-4
    return got == want


def _slow_flag(got, text) -> bool:
    """True when the learner's flag goes through all the n-grams of the training data for every question: 120
    questions against 100,000 n-grams then take it a second or more, where `grams & train_grams` takes a millisecond.
    The demos, 500 questions against hundreds of thousands of n-grams, four times, would take minutes."""
    fn = run_code(text).names.get("flag")
    train = {f"w{i} x" for i in range(100_000)}
    grams = [{f"w{i} x", "y z"} for i in range(120)]
    start = time.perf_counter()
    try:
        with_time_limit(lambda: fn(grams, train, 0))
    except Timeout:
        return True
    except Exception:   # every case already worked; an error here is not about speed
        return False
    return time.perf_counter() - start > 0.3


_T = [True, False, True, True, False, False, True, False]
_FL = [False, True, False, False, True, False, False, False]

EXERCISES = {
    "ngrams": FunctionBox(
        name="ngrams",
        function="ngrams",
        start=_G_START,
        solution=_g(),
        visible=[("The dog barked at the post-man, \"Go home!\"", 4)],
        hidden=[("one two three", 3), ("One, two.", 3), ("", 2), (" - ", 1), ("Hi!  Hi?\nHI.", 1),
                ("\"Yes,\" she said. \"Yes!\"", 2), ("a b c d e f", 5), ("Stop. - Now - stop!", 2)],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("no-strip", wrong=_g(word="        if w:\n            words.append(w)\n"),
                    en=_CALL + ": your n-grams keep the punctuation, so a word with a mark at its end and the same word without one count as different words. Take it off both ends of every word: `w.strip(PUNCT)`.",
                    nl=_CALL_NL + ": je n-grams houden de leestekens, dus een woord met een leesteken aan het eind en hetzelfde woord zonder tellen als verschillende woorden. Haal ze van beide kanten van elk woord af: `w.strip(PUNCT)`."),
            Mistake("rstrip", wrong=_g(word="        w = w.rstrip(PUNCT)\n        if w:\n            words.append(w)\n"),
                    en=_CALL + ": the punctuation at the start of a word stays, such as the quote before \"go\". `strip` takes it off both ends; `rstrip` only off the end.",
                    nl=_CALL_NL + ": de leestekens aan het begin van een woord blijven staan, zoals het aanhalingsteken voor \"go\". `strip` haalt ze van beide kanten af; `rstrip` alleen van het eind."),
            Mistake("empty-kept", wrong=_g(word="        words.append(w.strip(PUNCT))\n"),
                    en=_CALL + ": a word that is only punctuation, such as a dash, becomes an empty word, and it still counts. Keep a word only when something is left: `if w:`.",
                    nl=_CALL_NL + ": een woord dat alleen uit leestekens bestaat, zoals een streepje, wordt een leeg woord, en dat telt nog mee. Houd een woord alleen als er iets van over is: `if w:`."),
            Mistake("short-none", wrong=_g(short=""),
                    en=_CALL + ": a text shorter than n words gets no n-gram, so a short question can never be flagged. Give it the whole text as one n-gram, as GPT-3 did: `{{\" \".join(words)}}`.",
                    nl=_CALL_NL + ": een tekst korter dan n woorden krijgt geen n-gram, dus een korte vraag kan nooit gemarkeerd worden. Geef hem de hele tekst als één n-gram, zoals GPT-3 deed: `{{\" \".join(words)}}`."),
            Mistake("short-empty", wrong=_g(short="    if len(words) < n:\n        return {\" \".join(words)}\n"),
                    en=_CALL + ": a text without words gets the n-gram `\"\"`, and every other text without words has it too. Give it no n-grams: `set()`.",
                    nl=_CALL_NL + ": een tekst zonder woorden krijgt het n-gram `\"\"`, en elke andere tekst zonder woorden heeft dat ook. Geef hem geen n-grams: `set()`."),
            Mistake("no-lower", wrong=_g(head=_G_HEAD.replace("text.lower().split()", "text.split()")),
                    en=_CALL + ": the capitals stay in your n-grams, so a word with a capital and the same word without one count as different words. Make the text lower case first: `text.lower().split()`.",
                    nl=_CALL_NL + ": de hoofdletters blijven in je n-grams, dus een woord met een hoofdletter en hetzelfde woord zonder tellen als verschillende woorden. Zet de tekst eerst in kleine letters: `text.lower().split()`."),
            Mistake("last-missing", wrong=_g(loop=_G_LOOP.replace("range(len(words) - n + 1)", "range(len(words) - n)")),
                    en=_CALL + ": the last run of n words is missing. The last one starts at word `len(words) - n`, so the range goes to `len(words) - n + 1`.",
                    nl=_CALL_NL + ": de laatste reeks van n woorden ontbreekt. De laatste begint bij woord `len(words) - n`, dus de range loopt tot `len(words) - n + 1`."),
            Mistake("short-raw", wrong=_g(short="    if len(words) < n:\n        if words:\n            return {text}\n        return set()\n"),
                    en=_CALL + ": a short text is one n-gram of its cleaned words, in lower case and without punctuation, as every other n-gram. Join the words: `{{\" \".join(words)}}`.",
                    nl=_CALL_NL + ": een korte tekst is één n-gram van zijn opgeschoonde woorden, in kleine letters en zonder leestekens, net als elk ander n-gram. Plak de woorden aan elkaar: `{{\" \".join(words)}}`."),
            Mistake("inside", wrong=_g(word="        for c in PUNCT:\n            w = w.replace(c, \"\")\n        if w:\n            words.append(w)\n"),
                    en=_CALL + ": your function also takes the punctuation out of the middle of a word, so \"post-man\" becomes \"postman\". `strip` takes it only off both ends: `w.strip(PUNCT)`.",
                    nl=_CALL_NL + ": je functie haalt de leestekens ook midden uit een woord, dus \"post-man\" wordt \"postman\". `strip` haalt ze alleen van beide kanten af: `w.strip(PUNCT)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: strip every word and keep it when something is left; then the short texts; then every run of n words.",
                      "nl": "Vervang de drie `...`: strip elk woord en houd het als er iets van over is; dan de korte teksten; dan elke reeks van n woorden."},
            "wrong": {"en": _CALL + ". Every run of n words in a row, of the words in lower case without punctuation at their ends, each once?",
                      "nl": _CALL_NL + ". Elke reeks van n woorden achter elkaar, van de woorden in kleine letters zonder leestekens aan hun uiteinden, elk één keer?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `n` and `text`, and handle a text shorter than `n` words?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `n` en `text`, en kan hij een tekst korter dan `n` woorden aan?"},
            "correct": {"en": "Your function gives the n-grams of a text, without capitals and punctuation, so that a question copied in another layout still has the same n-grams.",
                        "nl": "Je functie geeft de n-grams van een tekst, zonder hoofdletters en leestekens, zodat een vraag die in een andere opmaak gekopieerd is nog dezelfde n-grams heeft."},
        },
    ),
    "flag": FunctionBox(
        name="flag",
        function="flag",
        start=_F_START,
        solution=_F_BODY,
        visible=[([{"a b", "b c", "c d"}, {"x y"}, set()], {"b c", "q r"}, 0)],
        hidden=[([{"a b", "b c", "c d"}, {"x y"}, set()], {"b c", "q r"}, 0.5), ([{"a b", "b c"}, {"q r"}], {"a b", "q r"}, 0.5),
                ([{"a b", "b c", "c d", "d e"}], {"a b", "b c"}, 0.5), ([{"a b", "b c", "c d", "d e"}], {"a b", "b c", "c d"}, 0.5),
                ([], {"a b"}, 0), ([{"s t"}, {"u v", "v w"}], set(), 0)],
        show=_shown,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            Mistake("at-least", wrong=_F_BODY.replace("> share", ">= share"),
                    en=_CALL + ": with `>=`, a question with no n-gram in the training data is flagged when `share` is 0. Flag it only when more than `share` is found: `>`.",
                    nl=_CALL_NL + ": met `>=` wordt een vraag zonder enige n-gram in de training data gemarkeerd als `share` 0 is. Markeer hem alleen als er meer dan `share` gevonden wordt: `>`."),
            Mistake("count", wrong=_F_BODY.replace("found / len(grams) > share", "found > share"),
                    en=_CALL + ": your function flags the first question, while only 1 of its 3 n-grams is in `train_grams`: a third, which is not more than 0.5. Compare the share of the question's n-grams that is found with `share`: `found / len(grams) > share`.",
                    nl=_CALL_NL + ": je functie markeert de eerste vraag, terwijl maar 1 van zijn 3 n-grams in `train_grams` zit: een derde, en dat is niet meer dan 0,5. Vergelijk het deel van de n-grams van de vraag dat gevonden is met `share`: `found / len(grams) > share`."),
            Mistake("train-share", wrong=_F_BODY.replace("found / len(grams)", "found / len(train_grams)"),
                    en=_CALL + ": you divide by the n-grams of the training data. The share is of the question's own n-grams: `found / len(grams)`.",
                    nl=_CALL_NL + ": je deelt door de n-grams van de training data. Het deel gaat over de eigen n-grams van de vraag: `found / len(grams)`."),
            Mistake("jaccard", wrong=_F_BODY.replace("found / len(grams)", "found / len(grams | train_grams)"),
                    en=_CALL + ": you divide by the n-grams of the question and of the training data together, as for the overlap of two pages. The share here is of the question's own n-grams: `found / len(grams)`.",
                    nl=_CALL_NL + ": je deelt door de n-grams van de vraag en van de training data samen, zoals bij de overlap van twee pagina's. Het deel gaat hier over de eigen n-grams van de vraag: `found / len(grams)`."),
            Mistake("empty", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="With `{call}`, your function divides by 0. A question without n-grams, such as `set()`, has `len(grams)` 0: give it `False` first, before you divide.",
                    nl="Met `{call}` deelt je functie door 0. Een vraag zonder n-grams, zoals `set()`, heeft `len(grams)` 0: geef hem eerst `False`, voordat je deelt."),
        ],
        form=[Mistake("slow", state="slow", matches=_slow_flag,
                      en="Your `flag` gives the right flags, but slowly: for every question it goes through all the n-grams of the training data. The demo checks 500 questions against hundreds of thousands of n-grams, which would take minutes. Look up only the question's own n-grams: `grams & train_grams`.",
                      nl="Je `flag` geeft de goede markeringen, maar traag: voor elke vraag loopt hij alle n-grams van de training data langs. De demo controleert 500 vragen tegen honderdduizenden n-grams, en dat zou minuten duren. Zoek alleen de eigen n-grams van de vraag op: `grams & train_grams`.")],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: for every question, `False` without n-grams, and else whether the share of its n-grams in the training data is more than `share`.",
                      "nl": "Vervang de `...`: voor elke vraag `False` zonder n-grams, en anders of het deel van zijn n-grams in de training data meer is dan `share`."},
            "wrong": {"en": _CALL + ". For every question: more than `share` of its n-grams in `train_grams`, as `True` or `False`?",
                      "nl": _CALL_NL + ". Voor elke vraag: meer dan `share` van zijn n-grams in `train_grams`, als `True` of `False`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `share`, and the n-grams of each question?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `share`, en de n-grams van elke vraag?"},
            "correct": {"en": "Your function flags the questions whose n-grams are in the training data. With `share` 0, one shared n-gram is enough, as with GPT-3.",
                        "nl": "Je functie markeert de vragen waarvan n-grams in de training data zitten. Met `share` 0 is één gedeeld n-gram genoeg, zoals bij GPT-3."},
        },
    ),
    "report": FunctionBox(
        name="report",
        function="report",
        start=_R_START,
        solution=_R_BODY,
        visible=[(_FL, _T)],
        hidden=[([True, True], [True, False]), ([False, False, False, True], [False, True, False, True]),
                ([False], [True]), ([True, False, False], [True, True, False])],
        show=_shown,
        compare=_same,
        first=[_DOTS],
        mistakes=[
            Mistake("flagged-only", wrong=_R_BODY.replace("if not flags[i]:", "if flags[i]:"),
                    en=_CALL + ": your second number is the score on the flagged questions. The clean subset is the questions that are not flagged: `if not flags[i]:`.",
                    nl=_CALL_NL + ": je tweede getal is de score op de gemarkeerde vragen. De clean subset zijn de vragen die niet gemarkeerd zijn: `if not flags[i]:`."),
            Mistake("all-count", wrong=_R_BODY.replace("sum(clean) / len(clean)", "sum(clean) / len(right)"),
                    en=_CALL + ": you divide the right answers of the clean subset by all questions. Divide by the questions of the clean subset: `len(clean)`.",
                    nl=_CALL_NL + ": je deelt de goede antwoorden van de clean subset door alle vragen. Deel door de vragen van de clean subset: `len(clean)`."),
            Mistake("zero", wrong=_R_BODY.replace("        clean_score = None\n", "        clean_score = 0\n"),
                    en=_CALL + ": with no clean question, your function gives a score of 0, as if the model got every clean question wrong. There is no score then: `None`.",
                    nl=_CALL_NL + ": zonder schone vraag geeft je functie een score van 0, alsof het model elke schone vraag fout had. Er is dan geen score: `None`."),
            Mistake("no-clean", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="With `{call}`, your function divides by 0. When no question is clean, `clean` is empty and there is no score: give `None` then, before you divide.",
                    nl="Met `{call}` deelt je functie door 0. Als geen enkele vraag schoon is, is `clean` leeg en is er geen score: geef dan `None`, voordat je deelt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the outcomes of the questions that are not flagged, and the share right among them.",
                      "nl": "Vervang de twee `...`: de uitkomsten van de vragen die niet gemarkeerd zijn, en het deel daarvan dat goed is."},
            "wrong": {"en": _CALL + ". The share right on all questions, on those that are not flagged, and the share flagged?",
                      "nl": _CALL_NL + ". Het deel goed op alle vragen, op de vragen die niet gemarkeerd zijn, en het deel dat gemarkeerd is?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `flags` and `right`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `flags` en `right`, en geen vaste getallen?"},
            "correct": {"en": "Your function gives the score on all questions and on the clean subset. When the score on the clean subset is much lower, the flagged questions may have lifted the score.",
                        "nl": "Je functie geeft de score op alle vragen en op de clean subset. Is de score op de clean subset veel lager, dan hebben de gemarkeerde vragen de score misschien opgedreven."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ngrams", _G_START, "empty", None),
    ("ngrams", _g(), "correct", None),
    ("ngrams", _g(short="    if len(words) < n:\n        if words:\n            return {text}\n        return set()\n"), "wrong", "short-raw"),
    ("ngrams", _g(word="        for c in PUNCT:\n            w = w.replace(c, \"\")\n        if w:\n            words.append(w)\n"), "wrong", "inside"),
    ("ngrams", _G_HEAD + _G_WORD + "    if len(words) < n:\n        return {\" \".join(words)} if words else set()\n"
     "    return {\" \".join(words[i:i + n]) for i in range(len(words) - n + 1)}\n", "correct", None),
    ("ngrams", _g(word="        if w:\n            words.append(w)\n"), "wrong", "no-strip"),
    ("ngrams", _g(word="        w = w.rstrip(PUNCT)\n        if w:\n            words.append(w)\n"), "wrong", "rstrip"),
    ("ngrams", _g(word="        words.append(w.strip(PUNCT))\n"), "wrong", "empty-kept"),
    ("ngrams", _g(short=""), "wrong", "short-none"),
    ("ngrams", _g(short="    if len(words) < n:\n        return {\" \".join(words)}\n"), "wrong", "short-empty"),
    ("ngrams", _g(head=_G_HEAD.replace("text.lower().split()", "text.split()")), "wrong", "no-lower"),
    ("ngrams", _g(loop=_G_LOOP.replace("range(len(words) - n + 1)", "range(len(words) - n)")), "wrong", "last-missing"),
    ("ngrams", _G_HEAD + _G_WORD + _G_SHORT + "    return {\" \".join(words[i:i + 4]) for i in range(len(words) - 3)}\n", "almost", None),
    ("ngrams", _G_START.replace("        ...\n", "        words.append(w)\n", 1), "wrong", "dots-left"),
    ("flag", _F_START, "empty", None),
    ("flag", _F_BODY, "correct", None),
    ("flag", _F_BODY.replace("found / len(grams)", "found / len(grams | train_grams)"), "wrong", "jaccard"),
    ("flag", "def flag(test_grams, train_grams, share):\n    return [bool(g) and len(g & train_grams) / len(g) > share for g in test_grams]\n", "correct", None),
    ("flag", _F_BODY.replace("> share", ">= share"), "wrong", "at-least"),
    ("flag", _F_BODY.replace("found / len(grams) > share", "found > share"), "wrong", "count"),
    ("flag", _F_BODY.replace("found / len(grams)", "found / len(train_grams)"), "wrong", "train-share"),
    ("flag", "def flag(test_grams, train_grams, share):\n    return [len(g & train_grams) / len(g) > share for g in test_grams]\n", "error", "empty"),
    ("flag", "def flag(test_grams, train_grams, share):\n    return [len(g & train_grams) > 0 for g in test_grams]\n", "wrong", "count"),
    ("flag", "def flag(test_grams, train_grams, share):\n    return [bool(g) and 100 * len(g & train_grams) / len(g) > share for g in test_grams]\n", "wrong", "count"),
    ("flag", _F_BODY.replace("found = len(grams & train_grams)", "found = 0\n            for t in train_grams:\n                if t in grams:\n                    found += 1"), "slow", "slow"),
    ("report", _R_START, "empty", None),
    ("report", _R_BODY, "correct", None),
    ("report", "def report(flags, right):\n    clean = [r for f, r in zip(flags, right) if not f]\n"
     "    return sum(right) / len(right), sum(clean) / len(clean) if clean else None, sum(flags) / len(flags)\n", "correct", None),
    ("report", _R_BODY.replace("if not flags[i]:", "if flags[i]:"), "wrong", "flagged-only"),
    ("report", _R_BODY.replace("sum(clean) / len(clean)", "sum(clean) / len(right)"), "wrong", "all-count"),
    ("report", _R_BODY.replace("        clean_score = None\n", "        clean_score = 0\n"), "wrong", "zero"),
    ("report", "def report(flags, right):\n    clean = [r for f, r in zip(flags, right) if not f]\n"
     "    return round(sum(right) / len(right), 4), round(sum(clean) / len(clean), 4) if clean else None, sum(flags) / len(flags)\n", "correct", None),
    ("report", "def report(flags, right):\n    clean = [r for f, r in zip(flags, right) if not f]\n"
     "    return round(sum(right) / len(right), 3), round(sum(clean) / len(clean), 3) if clean else None, sum(flags) / len(flags)\n", "wrong", None),
    ("report", "def report(flags, right):\n    clean = [r for f, r in zip(flags, right) if not f]\n"
     "    return sum(right) / len(right), sum(clean) / len(clean), sum(flags) / len(flags)\n", "error", "no-clean"),
]
