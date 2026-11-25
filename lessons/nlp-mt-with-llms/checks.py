# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-mt-with-llms: the numbered lines of a whole text for a prompt (numbered), the messages of
one chat that translates a text sentence by sentence (chat_messages), and the consistency of a term's translations in
one text (consistency).

The twelve Dutch texts are the course's own, each with a town whose name is also a Dutch word (Best, Echt, Bergen)
and a term with more than one meaning (bank, blad, monster), and the course's English translation of each sentence.
Qwen3-0.6B, the course's chat model, translated each text in three ways, greedy (temperature 0) and without thinking:
sentence by sentence, each sentence in a chat of its own (alone); sentence by sentence in one chat, with every earlier
message kept (chat); and the whole text in one prompt, with numbered lines (document). The run is
data/course-mt-with-llms-qwen.json, made with data/tools/record_course_mt_with_llms_qwen.py. The course read every
translation and noted the English word Qwen3 used for the term each time (FOUND), and whether that word means what the
term means in the text (SENSES). Consistency follows Wang and others (2023), after Lyu and others (2021): of all pairs
of translations of a term in one text, the share that are the same. BLEU and chrF are those of nlp-bleu, against the
course's translation. Every number in the text and the pictures is computed by the course
(tests/test_nlp_mt_with_llms.py computes them again)."""
import ast
import functools
import html
import json
import math
import re

import numpy as np

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake

QWEN_FILE = paths.ROOT / "data" / "course-mt-with-llms-qwen.json"
DOWNLOADS = {"Qwen3-0.6B": [("Qwen/Qwen3-0.6B", "model.safetensors"), ("Qwen/Qwen3-0.6B-Base", "vocab.json"),
                            ("Qwen/Qwen3-0.6B-Base", "tokenizer_config.json")]}
LANGS = ("en", "nl")
WAYS = ("alone", "chat", "document")
MAX_TOKENS = 64            # for one sentence
DOCUMENT_TOKENS = 400      # for a whole text
INSTRUCTION = "Translate this Dutch sentence into English. Reply with only the English translation."
DOCUMENT_INSTRUCTION = ("Translate this Dutch text into English. Translate each numbered line into one numbered line. "
                        "Reply with only the English translation.")

# The course's twelve Dutch texts: each with a town whose name is also a Dutch word (word: that word in English) and a
# term with more than one meaning, and the course's English translation of each sentence.
DOCUMENTS = (
    {"place": "Dieren", "word": "animals", "term": "beurs",
     "nl": ("Dit weekend is er een boekenbeurs in Dieren.",
            "Op de beurs staan meer dan honderd kramen met oude boeken.",
            "Dieren houdt de beurs al veertig jaar, elk voorjaar.",
            "De beurs is zaterdag en zondag open van tien tot vijf uur.",
            "De toegang is gratis, en wie in Dieren woont, parkeert ook gratis."),
     "en": ("This weekend there is a book fair in Dieren.",
            "At the fair, there are more than a hundred stalls with old books.",
            "Dieren has held the fair every spring for forty years.",
            "The fair is open on Saturday and Sunday from ten to five.",
            "Entry is free, and anyone who lives in Dieren can also park for free.")},
    {"place": "Sluis", "word": "sluice", "term": "bank",
     "nl": ("Langs het water in Sluis staan sinds deze week twintig nieuwe zitbanken.",
            "Elke bank is gemaakt van oud hout uit de haven.",
            "Sluis betaalde ruim veertigduizend euro voor de banken.",
            "Op een bank is plaats voor drie mensen.",
            "Volgend jaar komen er ook banken bij het station van Sluis."),
     "en": ("Since this week, there are twenty new benches along the water in Sluis.",
            "Each bench is made of old wood from the harbor.",
            "Sluis paid more than forty thousand euros for the benches.",
            "A bench has room for three people.",
            "Next year, there will also be benches at the station of Sluis.")},
    {"place": "Bergen", "word": "mountains", "term": "slot",
     "nl": ("Het kasteel van Bergen is na drie jaar werk weer open.",
            "Het slot is meer dan zevenhonderd jaar oud.",
            "Bergen heeft het dak en de toren van het slot laten herstellen.",
            "In de zomer is het slot elke dag open voor bezoekers.",
            "Een kaartje kost tien euro, en voor kinderen uit Bergen vijf euro."),
     "en": ("The castle of Bergen is open again after three years of work.",
            "The castle is more than seven hundred years old.",
            "Bergen has had the roof and the tower of the castle repaired.",
            "In the summer, the castle is open to visitors every day.",
            "A ticket costs ten euros, and five euros for children from Bergen.")},
    {"place": "Haren", "word": "hairs", "term": "kaart",
     "nl": ("Aan de markt in Haren is een nieuw restaurant open.",
            "De kaart verandert elke week.",
            "Op de kaart staan vooral vis en groenten uit de buurt.",
            "Haren had nog geen restaurant met alleen vis.",
            "Deze week staat er ook mosselsoep op de kaart."),
     "en": ("A new restaurant has opened on the market square in Haren.",
            "The menu changes every week.",
            "The menu has mostly fish and vegetables from the area.",
            "Haren did not have a restaurant with only fish yet.",
            "This week, there is also mussel soup on the menu.")},
    {"place": "Best", "word": "best", "term": "vorst",
     "nl": ("Het weerbericht voor Best: vannacht komt er strenge nachtvorst, tot min acht graden.",
            "De vorst houdt aan tot zondag.",
            "Door de vorst zijn de wegen in Best glad.",
            "Best strooit vanavond zout op alle fietspaden.",
            "Na de vorst wordt het volgende week weer zachter."),
     "en": ("The weather forecast for Best: tonight there will be hard night frost, down to minus eight degrees.",
            "The frost will last until Sunday.",
            "Because of the frost, the roads in Best are slippery.",
            "Best will spread salt on all bike paths tonight.",
            "After the frost, it will get milder again next week.")},
    {"place": "Duiven", "word": "pigeons", "term": "vak",
     "nl": ("Een school in Duiven heeft een nieuw schoolvak: programmeren.",
            "Leerlingen krijgen het vak twee uur per week.",
            "Duiven is de eerste plaats in de buurt met dit vak.",
            "In het vak bouwen leerlingen een kleine robot.",
            "Volgend jaar moeten alle leerlingen het vak volgen."),
     "en": ("A school in Duiven has a new school subject: programming.",
            "Students have the subject for two hours a week.",
            "Duiven is the first town in the area with this subject.",
            "In the subject, students build a small robot.",
            "Next year, all students will have to take the subject.")},
    {"place": "Buren", "word": "neighbors", "term": "stof",
     "nl": ("In Buren is zaterdag een markt voor mensen die zelf kleren maken.",
            "Er liggen honderden rollen stof.",
            "De meeste stof komt uit India en Turkije.",
            "Buren houdt deze markt elk jaar in maart.",
            "Een meter stof kost vanaf twee euro."),
     "en": ("On Saturday there is a market in Buren for people who make their own clothes.",
            "There are hundreds of rolls of fabric.",
            "Most of the fabric comes from India and Turkey.",
            "Buren holds this market every year in March.",
            "A meter of fabric costs from two euros.")},
    {"place": "Hoorn", "word": "horn", "term": "blad",
     "nl": ("De bibliotheek van Hoorn geeft een nieuw tijdschrift uit over de stad.",
            "Het blad verschijnt vier keer per jaar.",
            "In het blad staan deze keer oude foto's van de haven.",
            "Alle huizen in Hoorn krijgen het blad gratis.",
            "De volgende keer gaat het blad over de markt."),
     "en": ("The library of Hoorn is publishing a new magazine about the town.",
            "The magazine comes out four times a year.",
            "This time, the magazine has old photos of the harbor.",
            "Every house in Hoorn gets the magazine for free.",
            "Next time, the magazine will be about the market.")},
    {"place": "Velden", "word": "fields", "term": "veer",
     "nl": ("De veerpont over de Maas bij Velden vaart weer.",
            "Het veer lag twee maanden stil voor onderhoud.",
            "Velden is blij, want de brug ligt tien kilometer verderop.",
            "Het veer is gratis voor fietsers en wandelaars.",
            "In de zomer gaat het veer elk kwartier."),
     "en": ("The ferry across the Maas at Velden is running again.",
            "The ferry was out of service for two months for maintenance.",
            "Velden is happy, because the bridge is ten kilometers away.",
            "The ferry is free for cyclists and walkers.",
            "In the summer, the ferry goes every fifteen minutes.")},
    {"place": "Kampen", "word": "camps", "term": "monster",
     "nl": ("Het waterbedrijf neemt elke week een watermonster uit de IJssel bij Kampen.",
            "Het monster gaat naar een lab in Zwolle.",
            "In het laatste monster zat te veel zout.",
            "Kampen vraagt het waterbedrijf om een nieuw monster.",
            "Tot de uitslag van het monster bekend is, mag niemand bij Kampen in de rivier zwemmen."),
     "en": ("Every week, the water company takes a water sample from the IJssel at Kampen.",
            "The sample goes to a lab in Zwolle.",
            "The last sample had too much salt in it.",
            "Kampen is asking the water company for a new sample.",
            "Until the result of the sample is known, nobody may swim in the river at Kampen.")},
    {"place": "Helden", "word": "heroes", "term": "toets",
     "nl": ("De leerlingen van de basisschool in Helden hadden vandaag een rekentoets.",
            "De toets duurde een uur.",
            "Helden doet al jaren mee aan deze toets.",
            "Volgende week krijgen de leerlingen de uitslag van de toets.",
            "De school in Helden gebruikt de toets om te zien wat leerlingen nog moeten oefenen."),
     "en": ("The pupils of the primary school in Helden had a math test today.",
            "The test took an hour.",
            "Helden has taken part in this test for years.",
            "Next week, the pupils will get the result of the test.",
            "The school in Helden uses the test to see what pupils still need to practice.")},
    {"place": "Echt", "word": "real", "term": "bord",
     "nl": ("Bij de ingang van Echt staan sinds gisteren nieuwe verkeersborden.",
            "Op elk bord staat de naam van de stad in twee talen.",
            "Echt betaalde de borden samen met de provincie.",
            "Een bord is twee meter breed.",
            "Volgend jaar krijgt ook het station van Echt een bord."),
     "en": ("Since yesterday, there are new road signs at the entrance of Echt.",
            "Each sign shows the name of the town in two languages.",
            "Echt paid for the signs together with the province.",
            "A sign is two meters wide.",
            "Next year, the station of Echt will also get a sign.")},
)


# --- the course's solutions: the prompt of the whole text, and the messages of one chat ---------------------------------

def numbered(sentences):
    lines = []
    for k in range(len(sentences)):
        lines.append(f"{k + 1}. {sentences[k]}")
    return "\n".join(lines)


def chat_messages(instruction, sentences, answers):
    messages = []
    for k in range(len(answers)):
        messages.append({"role": "user", "content": instruction + "\n\n" + sentences[k]})
        messages.append({"role": "assistant", "content": answers[k]})
    messages.append({"role": "user", "content": instruction + "\n\n" + sentences[len(answers)]})
    return messages


def alone_messages(sentence: str) -> list:
    """The prompt of one sentence on its own: a chat with one message."""
    return [{"role": "user", "content": INSTRUCTION + "\n\n" + sentence}]


def document_messages(sentences) -> list:
    """The prompt of a whole text: the instruction, a blank line and the numbered lines."""
    return [{"role": "user", "content": DOCUMENT_INSTRUCTION + "\n\n" + numbered(sentences)}]


# --- the course's reading of Qwen3's translations -------------------------------------------------------------------------

# The Dutch forms of a term other than the term itself, such as a plural.
FORMS = {"bank": ("bank", "banken"), "bord": ("bord", "borden")}

# For each term, every English word Qwen3 used for it, and whether that word is right: a word for what the term names
# in its text is, also a more general one (building for a castle); a word for another meaning of the term (leaf, board)
# or for another thing (newspaper for a magazine, wool for fabric of any kind, house for a castle) is not.
SENSES = {
    "beurs": {"market": True},
    "bank": {"bank": False},
    "slot": {"slot": False, "building": True, "gate": False, "house": False},
    "kaart": {"card": False, "map": False, "cart": False},
    "vorst": {"front": False, "meeting": False, "forecast": False},
    "vak": {"subject": True, "course": True, "class": True},
    "stof": {"material": True, "wool": False},
    "blad": {"leaf": False, "blade": False, "newspaper": False},
    "veer": {"rope": False},
    "monster": {"monster": False},
    "toets": {"test": True},
    "bord": {"board": False, "sign": True, "traffic signal": False, "toll": False},
}

# The word Qwen3 used for the term in each sentence that has the term, per text and way, written in the singular (banks
# is bank); None where its English has no word for the term, or is the Dutch sentence given back. The course read
# every translation to note these.
FOUND = {
    "Dieren": {"alone": ("market", None, "market"), "chat": ("market",) * 3, "document": ("market",) * 3},
    "Sluis": {"alone": ("bank",) * 4, "chat": ("bank",) * 4, "document": ("bank",) * 4},
    "Bergen": {"alone": ("slot", "building", None), "chat": ("building",) * 3, "document": ("gate", "house", "house")},
    "Haren": {"alone": ("card", "map", "cart"), "chat": ("card",) * 3, "document": (None,) * 3},
    "Best": {"alone": ("front", None, "meeting"), "chat": ("forecast",) * 3, "document": ("forecast",) * 3},
    "Duiven": {"alone": (None, "subject", None, "course"), "chat": ("class", "subject", "subject", "subject"),
               "document": ("subject",) * 4},
    "Buren": {"alone": ("material", "material", None), "chat": ("wool",) * 3, "document": ("wool",) * 3},
    "Hoorn": {"alone": ("leaf", None, "blade", None), "chat": ("newspaper",) * 4, "document": ("newspaper",) * 4},
    "Velden": {"alone": ("rope", "rope", None), "chat": ("rope",) * 3, "document": ("rope",) * 3},
    "Kampen": {"alone": ("monster", None, "monster", "monster"), "chat": ("monster",) * 4, "document": ("monster",) * 4},
    "Helden": {"alone": ("test",) * 4, "chat": ("test",) * 4, "document": ("test",) * 4},
    "Echt": {"alone": ("board", "board", "board", "sign"),
             "chat": ("traffic signal", "toll", "traffic signal", "traffic signal"), "document": ("sign",) * 4},
}


def _word_in(word: str, text_: str, case: bool = False) -> bool:
    """Whether `word` is in `text_` as a whole word (or words), not as part of a longer word."""
    return re.search(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", text_, 0 if case else re.IGNORECASE) is not None


def term_at(i: int) -> list:
    """The places of the sentences of text i that have its term."""
    d = DOCUMENTS[i]
    return [k for k, s in enumerate(d["nl"]) if any(_word_in(f, s) for f in FORMS.get(d["term"], (d["term"],)))]


def place_at(i: int) -> list:
    """The places of the sentences of text i that have its town."""
    d = DOCUMENTS[i]
    return [k for k, s in enumerate(d["nl"]) if _word_in(d["place"], s, case=True)]


@functools.cache
def _qwen() -> dict:
    return json.loads(QWEN_FILE.read_text(encoding="utf-8"))


def lines_of(answer: str, n: int) -> list:
    """The numbered lines 1 to n of an answer to the prompt of a whole text: for each number, the text after "k. " on
    the first line that starts with that number; "" for a number that did not come back."""
    got = {}
    for line in answer.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.*?)\s*$", line)
        if m and int(m.group(1)) not in got:
            got[int(m.group(1))] = m.group(2)
    return [got.get(k + 1, "") for k in range(n)]


def answer(i: int, way: str):
    """Qwen3's answer for text i in one way, as recorded: a list of one answer per sentence, or one answer."""
    return _qwen()["documents"][i][way]


def answers(i: int, way: str) -> list:
    """Qwen3's English for each sentence of text i in one way; for the whole text, the numbered lines of its answer."""
    if way == "document":
        return lines_of(answer(i, way)["text"], len(DOCUMENTS[i]["nl"]))
    return [a["text"] for a in answer(i, way)]


def copied(i: int, way: str, k: int) -> bool:
    """Whether Qwen3 gave back the Dutch of sentence k of text i, instead of a translation."""
    return answers(i, way)[k].strip() == DOCUMENTS[i]["nl"][k].strip()


def found(i: int, way: str) -> list:
    """The words Qwen3 used for the term of text i in one way, in the order of the sentences, None where it used none."""
    return list(FOUND[DOCUMENTS[i]["place"]][way])


def words_found(i: int, way: str) -> list:
    """The words Qwen3 used for the term of text i in one way, without the places where it used none: what the
    consistency of the term is computed from."""
    return [w for w in found(i, way) if w is not None]


def right(i: int, word: str) -> bool:
    return SENSES[DOCUMENTS[i]["term"]][word]


def place_state(i: int, way: str, k: int) -> str:
    """What Qwen3 did with the town in sentence k of text i: kept (the name is in its English), word (the name became
    the English word, such as animals for Dieren), copied (the Dutch given back) or gone (neither)."""
    d, t = DOCUMENTS[i], answers(i, way)[k]
    if copied(i, way, k):
        return "copied"
    if _word_in(d["place"], t, case=True):
        return "kept"
    if _word_in(d["word"], t):
        return "word"
    return "gone"


@functools.cache
def _bleu():
    """The checks of nlp-bleu: BLEU and chrF, and the words of a text as nlp-classic-mt reads them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_bleu_for_mt_with_llms",
                                                  paths.LESSONS / "nlp-bleu" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scores(way: str) -> tuple:
    """(BLEU, chrF) of one way over the 60 sentences of the twelve texts, against the course's translation, as
    nlp-bleu computes them over a whole test set."""
    B = _bleu()
    counts, chars = [], []
    for i, d in enumerate(DOCUMENTS):
        for t, r in zip(answers(i, way), d["en"]):
            counts.append(B.bleu_counts(B.words(t), [B.words(r)]))
            chars.append(B.best_chrf_counts(B.words(t), [B.words(r)]))
    return B.bleu(*B.add_up(counts)), B.chrf(B.add_chrf(chars))


def course_consistency(translations: list):
    """The consistency of a term: the share of pairs of its translations that are the same; None for fewer than two."""
    if len(translations) < 2:
        return None
    return consistency(translations)


def consistency(translations):
    pairs = 0
    same = 0
    for i in range(len(translations)):
        for j in range(i + 1, len(translations)):
            pairs = pairs + 1
            if translations[i] == translations[j]:
                same = same + 1
    return same / pairs


def _mean(values: list):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


@functools.cache
def numbers() -> dict:
    """Every number the text and the pictures name, per way, and for the twelve texts."""
    out = {"texts": len(DOCUMENTS), "sentences": sum(len(d["nl"]) for d in DOCUMENTS),
           "terms": sum(len(term_at(i)) for i in range(len(DOCUMENTS))),
           "places": sum(len(place_at(i)) for i in range(len(DOCUMENTS)))}
    for way in WAYS:
        n = {"right": 0, "not_right": 0, "none": 0, "kept": 0, "word": 0, "gone": 0, "copied": 0}
        texts = []
        for i in range(len(DOCUMENTS)):
            for w in found(i, way):
                n["none" if w is None else "right" if right(i, w) else "not_right"] += 1
            for k in place_at(i):
                n[place_state(i, way, k)] += 1
            texts.append(course_consistency(words_found(i, way)))
        n["consistency"] = texts
        n["mean"] = _mean(texts)
        n["measured"] = sum(c is not None for c in texts)
        n["same_wrong"] = [DOCUMENTS[i]["place"] for i, c in enumerate(texts)
                           if c == 1 and not any(right(i, w) for w in words_found(i, way))]
        n["bleu"], n["chrf"] = scores(way)
        n["copied_sentences"] = sum(copied(i, way, k) for i in range(len(DOCUMENTS)) for k in range(len(DOCUMENTS[i]["nl"])))
        out[way] = n
    out["lines"] = [sum(1 for t in answers(i, "document") if t) for i in range(len(DOCUMENTS))]
    return out


# --- styles and small parts of the pictures ------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_CHIP = ("display:inline-block;margin:0.08rem 0.2rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
_MARK_PRED = "background:var(--ea-pred-wash);color:var(--ea-pred-text);font-weight:600;border-radius:2px;padding:0 0.1rem"
_MARK_LOSS = "background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600;border-radius:2px;padding:0 0.1rem"
_CODE = ("font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;"
         "margin:0.25rem 0;padding:0.4rem 0.55rem;border-radius:4px;background:var(--ea-data-wash)")


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code."""
    return f'<div style="{_UI}">{_marked(text_)}</div>'


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 1) -> str:
    out = f"{round(float(x), digits) + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _chip(text_: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _gray(text_: str) -> str:
    return f'<span style="{_DATA}">{html.escape(text_)}</span>'


def _faint(text_: str) -> str:
    return f'<span style="{_FAINT}">{html.escape(text_)}</span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _rows(rows_: list) -> str:
    """A table without a head: each row a label and HTML that is already escaped."""
    body = "".join(f'<tr><td style="{_CELL};{_FAINT};white-space:nowrap;width:7.5rem">{html.escape(a)}</td>'
                   f'<td style="{_CELL}">{b}</td></tr>' for a, b in rows_)
    return f'<table style="border-collapse:collapse;margin:0.1rem 0 0.5rem;width:100%"><tbody>{body}</tbody></table>'


def _code(text_: str, style: str = "") -> str:
    return f'<div style="{_CODE};{style}">{html.escape(text_)}</div>'


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _empty(w: dict, key: str, seconds: float = PICTURE_SECONDS) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=seconds, kind=w.get("kind", ""))
    return yours.figure(w["title"], _note(text_), alt=text_)


_WHY_EMPTY = {
    "en": {"stopped": "Your function stops with an error, so there is nothing to draw yet. The check above says why.",
           "slow": "Your function took longer than {s} seconds, so there is nothing to draw yet.",
           "odd": "Your function gives something other than {kind}, so there is nothing to draw yet. The check above says what."},
    "nl": {"stopped": "Je functie stopt met een fout, dus er valt nog niets te tekenen. De check hierboven zegt waarom.",
           "slow": "Je functie deed er langer over dan {s} seconden, dus er valt nog niets te tekenen.",
           "odd": "Je functie geeft iets anders dan {kind}, dus er valt nog niets te tekenen. De check hierboven zegt wat."},
}


def _is_count(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:   # an int too large for a float
        return False


def _same(a, b) -> bool:
    """a == b, and False when comparing them stops with an error."""
    try:
        return bool(a == b)
    except Exception:   # noqa: BLE001
        return False


def _highlighted(i: int, text_: str) -> str:
    """Qwen3's English for a sentence of text i, escaped, with the town's name in blue where it stays a name, its
    English word in purple, and each word Qwen3 used for the term in blue when it means what the term means in the
    text, in purple when it does not."""
    d = DOCUMENTS[i]
    marks = [(re.escape(d["place"]), _MARK_PRED, True), (re.escape(d["word"]), _MARK_LOSS, False)]
    for w, ok in SENSES[d["term"]].items():
        marks.append((re.escape(w) + "(?:s|es)?", _MARK_PRED if ok else _MARK_LOSS, False))
    out, at = [], 0
    spans = []
    for rank, (pattern, style, case) in enumerate(marks):   # the name first: Best stays a name, best is the word
        for m in re.finditer(r"(?<![\w-])" + pattern + r"(?![\w-])", text_, 0 if case else re.IGNORECASE):
            spans.append((m.start(), rank, m.end(), style))
    for start, _, end, style in sorted(spans):
        if start < at:
            continue
        out.append(html.escape(text_[at:start]))
        out.append(f'<span style="{style}">{html.escape(text_[start:end])}</span>')
        at = end
    out.append(html.escape(text_[at:]))
    return "".join(out)


WAY_NAMES = {"en": {"alone": "each sentence alone", "chat": "in one chat", "document": "the whole text"},
             "nl": {"alone": "elke zin los", "chat": "in één chat", "document": "de hele tekst"}}


# --- the idea, the table and the texts to try ------------------------------------------------------------------------------

IDEA = 7         # the text of the idea: the magazine of Hoorn
NUMBERED = 11    # the text under exercise 1: the signs of Echt
CHAT = (5, 2)    # under exercise 2: the third sentence of the text about Duiven
SAME_WRONG = 9   # exercise 4: the water sample of Kampen

_IDEA_TEXT = {
    "en": {"title": "One text in three ways: the magazine of Hoorn",
           "lead": "The Dutch text, and under each sentence what Qwen3 wrote in each of the three ways, with the course's translation.",
           "course": "the course",
           "note": "Blue: *Hoorn* kept as a name. Purple: *leaf* and *blade*, other meanings of *blad*, and *newspaper*, another thing than the magazine of the text. Gray: the Dutch and the course's translation."},
    "nl": {"title": "Eén tekst op drie manieren: het blad van Hoorn",
           "lead": "De Nederlandse tekst, en onder elke zin wat Qwen3 op elk van de drie manieren schreef, met de vertaling van de course.",
           "course": "de course",
           "note": "Blauw: *Hoorn* als naam behouden. Paars: *leaf* en *blade*, andere betekenissen van *blad*, en *newspaper*, iets anders dan het tijdschrift van de tekst. Grijs: het Nederlands en de vertaling van de course."},
}


def _blocks(i: int, ways: tuple, lang: str, course_label: str) -> str:
    """For each sentence of text i: the Dutch, what Qwen3 wrote in each of `ways`, and the course's translation."""
    d, out = DOCUMENTS[i], []
    texts = {way: answers(i, way) for way in ways}
    for k, dutch in enumerate(d["nl"]):
        rows_ = [(WAY_NAMES[lang][way], _faint(texts[way][k]) if copied(i, way, k)
                  else _highlighted(i, texts[way][k]) if texts[way][k] else _faint("—")) for way in ways]
        rows_.append((course_label, _gray(d["en"][k])))
        out.append(f'<div style="{_LABEL};{_DATA}">{k + 1}. {html.escape(dutch)}</div>' + _rows(rows_))
    return "".join(out)


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    body = _note(w["lead"]) + _blocks(IDEA, WAYS, lang, w["course"])
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


_TABLE_TEXT = {
    "en": {"title": "Three ways, twelve texts",
           "head": ["", "each sentence alone", "in one chat", "the whole text"],
           "rows": ["town kept as a name, of {places}", "term with a right word, of {terms}", "term without a word",
                    "consistency of the term", "BLEU", "chrF"],
           "note": "Consistency: the mean over the texts where Qwen3 used a word for the term at least twice ({alone}, {chat} and {document} of the 12). BLEU and chrF over the 60 sentences, against the course's translation."},
    "nl": {"title": "Drie manieren, twaalf teksten",
           "head": ["", "elke zin los", "in één chat", "de hele tekst"],
           "rows": ["plaats als naam behouden, van {places}", "term met een goed woord, van {terms}", "term zonder woord",
                    "consistentie van de term", "BLEU", "chrF"],
           "note": "Consistentie: het gemiddelde over de teksten waarin Qwen3 minstens twee keer een woord voor de term gebruikte ({alone}, {chat} en {document} van de 12). BLEU en chrF over de 60 zinnen, tegen de vertaling van de course."},
}


def table_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w, n = _TABLE_TEXT[lang], numbers()
    cells = [[_num(n[way]["kept"], lang) for way in WAYS], [_num(n[way]["right"], lang) for way in WAYS],
             [_num(n[way]["none"], lang) for way in WAYS], [_dec(n[way]["mean"], lang, 2) for way in WAYS],
             [_dec(100 * n[way]["bleu"], lang) for way in WAYS], [_dec(100 * n[way]["chrf"], lang) for way in WAYS]]
    rows_ = [[html.escape(label.format(places=n["places"], terms=n["terms"]))] + c for label, c in zip(w["rows"], cells)]
    note = w["note"].format(**{way: n[way]["measured"] for way in WAYS})
    return yours.figure(w["title"], _table(rows_, w["head"], (1, 2, 3)) + _note(note), alt=w["title"])


_TRY_TEXT = {
    "en": {"title": "{place}: {term}", "course": "the course",
           "blue": "Blue: {place} kept as a name, and a word in the meaning *{term}* has in this text.",
           "purple": "Purple: {place} turned into an English word, or a word with another meaning than *{term}* has here.",
           "gray": "Gray: the Dutch and the course's translation.",
           "dash": "A — means that Qwen3's answer has no line with this number.",
           "copied": "A faint line is the Dutch sentence itself: Qwen3 gave the sentence back without translating it."},
    "nl": {"title": "{place}: {term}", "course": "de course",
           "blue": "Blauw: {place} als naam behouden, en een woord in de betekenis die *{term}* in deze tekst heeft.",
           "purple": "Paars: {place} vertaald als Engels woord, of een woord met een andere betekenis dan *{term}* hier heeft.",
           "gray": "Grijs: het Nederlands en de vertaling van de course.",
           "dash": "Een — betekent dat het antwoord van Qwen3 geen regel met dit nummer heeft.",
           "copied": "Een lichte regel is de Nederlandse zin zelf: Qwen3 gaf de zin terug zonder te vertalen."},
}


def text_options(lang: str) -> dict:
    """The twelve texts for a dropdown: their town and term, by their place in DOCUMENTS."""
    return {f"{d['place']}: {d['term']}": i for i, d in enumerate(DOCUMENTS)}


def try_picture(lang: str, i, way: str) -> str:
    lang = lang if lang in LANGS else "en"
    i = i if isinstance(i, int) and 0 <= i < len(DOCUMENTS) else IDEA
    way = way if way in WAYS else "alone"
    w, d = _TRY_TEXT[lang], DOCUMENTS[i]
    body = _blocks(i, (way,), lang, w["course"])
    texts = answers(i, way)
    marked = "".join(_highlighted(i, t) for k, t in enumerate(texts) if not copied(i, way, k))
    notes = [w[key].format(place=d["place"], term=d["term"]) for key, style in (("blue", _MARK_PRED), ("purple", _MARK_LOSS))
             if style in marked]
    notes.append(w["gray"])
    if not all(texts):
        notes.append(w["dash"])
    if any(copied(i, way, k) for k in range(len(d["nl"]))):
        notes.append(w["copied"])
    title = w["title"].format(place=d["place"], term=d["term"])
    return yours.figure(title, body + "".join(_note(x) for x in notes), alt=title)


# --- the pictures under the checks -------------------------------------------------------------------------------------------

_NUMBERED_TEXT = {
    "en": {"title": "Your prompt for the whole text about Echt", "kind": "a text",
           "same": "With your function, the prompt is the one the course gave Qwen3: the instruction, a blank line and your numbered lines. Under it, Qwen3's answer, line by line.",
           "differs": "With your function, the prompt is not the one the course gave Qwen3: the course's has {want} where yours has {got}. Under it, Qwen3's answer to the course's prompt, line by line.",
           "head": ["", "Dutch", "Qwen3's line"],
           "back": "In all 12 texts, Qwen3 gave back one numbered line for each sentence. In the text about Haren, every line is the Dutch sentence itself.",
           "note": "Gray: the prompt and the Dutch. Blue: *Echt* kept as a name, and *sign*, a word in the meaning *bord* has here."},
    "nl": {"title": "Je prompt voor de hele tekst over Echt", "kind": "een tekst",
           "same": "Met je functie is de prompt dezelfde die de course aan Qwen3 gaf: de instructie, een lege regel en je genummerde regels. Eronder het antwoord van Qwen3, regel voor regel.",
           "differs": "Met je functie is de prompt niet dezelfde die de course aan Qwen3 gaf: de prompt van de course heeft {want} waar de jouwe {got} heeft. Eronder het antwoord van Qwen3 op de prompt van de course, regel voor regel.",
           "head": ["", "Nederlands", "Regel van Qwen3"],
           "back": "In alle 12 teksten gaf Qwen3 voor elke zin één genummerde regel terug. In de tekst over Haren is elke regel de Nederlandse zin zelf.",
           "note": "Grijs: de prompt en het Nederlands. Blauw: *Echt* als naam behouden, en *sign*, een woord in de betekenis die *bord* hier heeft."},
}


def _first_difference(got: str, want: str) -> tuple:
    """The first line where two texts differ, as code: (got, want); the end of a text shows as nothing."""
    a, b = got.split("\n"), want.split("\n")
    for k in range(max(len(a), len(b))):
        x = a[k] if k < len(a) else None
        y = b[k] if k < len(b) else None
        if x != y:
            return x, y
    return None, None


def _code_or_nothing(x, lang: str) -> str:
    if x is None:
        return "nothing" if lang == "en" else "niets"
    x = x if len(x) <= 120 else x[:120] + " …"
    return '"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _note_with(text_: str, lang: str, **lines) -> str:
    """A note whose {name} places hold a line of the learner's text as code, set after the note's own marking, so a *
    or a backtick in the learner's line stays as typed."""
    marked = _marked(text_.format(**{k: f"\x00{k}\x00" for k in lines}))
    for k, x in lines.items():
        shown = html.escape(_code_or_nothing(x, lang))
        marked = marked.replace(f"\x00{k}\x00", shown if x is None else f"<code>{shown}</code>")
    return f'<div style="{_UI}">{marked}</div>'


def numbered_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _NUMBERED_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    d = DOCUMENTS[NUMBERED]
    got = _call(lambda: fn(list(d["nl"])))
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is _SLOW:
        return _empty(w, "slow")
    if not isinstance(got, str):
        return _empty(w, "odd")
    want = numbered(d["nl"])
    if got == want:
        lead = _note(w["same"])
    else:
        x, y = _first_difference(got, want)
        lead = _note_with(w["differs"], lang, got=x, want=y)
    prompt = _code(DOCUMENT_INSTRUCTION + "\n\n" + (got if len(got) <= 3000 else got[:3000] + " …"), _DATA)
    lines = answers(NUMBERED, "document")
    rows_ = [[f"{k + 1}.", _gray(dutch), _highlighted(NUMBERED, line) if line else _faint("—")]
             for k, (dutch, line) in enumerate(zip(d["nl"], lines))]
    body = lead + prompt + _table(rows_, w["head"]) + _note(w["back"]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_CHAT_TEXT = {
    "en": {"title": "Your chat for sentence 3 about Duiven", "kind": "a list of messages",
           "same": "With your function, these are the messages the course gave Qwen3 for sentence 3, in one chat:",
           "differs": "With your function, the messages are not the ones the course gave Qwen3: {why} Below, the messages you made; the answers are to the course's messages.",
           "count": "the course's chat has {want} messages, and yours {got}.",
           "message": "message {k} differs from the course's.",
           "answered": "What Qwen3 answered", "in_chat": "in this chat", "alone": "the sentence alone",
           "note": "Gray: messages with the role `user`. Blue: messages with the role `assistant`, Qwen3's own earlier answers, and in its new answers *Duiven* as a name and *subject*, a word in the meaning *vak* has here."},
    "nl": {"title": "Je chat voor zin 3 over Duiven", "kind": "een lijst met berichten",
           "same": "Met je functie zijn dit de berichten die de course in één chat aan Qwen3 gaf voor zin 3:",
           "differs": "Met je functie zijn de berichten niet dezelfde die de course aan Qwen3 gaf: {why} Hieronder de berichten die je maakte; de antwoorden zijn op de berichten van de course.",
           "count": "de chat van de course heeft {want} berichten, en de jouwe {got}.",
           "message": "bericht {k} verschilt van dat van de course.",
           "answered": "Wat Qwen3 antwoordde", "in_chat": "in deze chat", "alone": "de zin los",
           "note": "Grijs: berichten met de rol `user`. Blauw: berichten met de rol `assistant`, de eigen eerdere antwoorden van Qwen3, en in de nieuwe antwoorden *Duiven* als naam en *subject*, een woord in de betekenis die *vak* hier heeft."},
}


def _text_of(x) -> str:
    """str(x), and a mark when even that stops with an error (an int of thousands of digits)."""
    try:
        return str(x)
    except Exception:   # noqa: BLE001
        return "…"


def _bubble(role, content) -> str:
    role, content = _text_of(role)[:40], _text_of(content)
    content = content if len(content) <= 1200 else content[:1200] + " …"
    style = ("background:var(--ea-pred-wash);color:var(--ea-pred-text);margin-left:2rem" if role == "assistant"
             else "background:var(--ea-data-wash);color:var(--ea-data-text);margin-right:2rem")
    return (f'<div style="margin:0.3rem 0;padding:0.3rem 0.55rem;border-radius:6px;{style}">'
            f'<div style="font-family:var(--ea-font-ui);font-size:0.72rem;opacity:0.8">{html.escape(role)}</div>'
            f'<div style="font-family:var(--ea-font-ui);font-size:0.84rem;white-space:pre-wrap;overflow-wrap:anywhere">'
            f'{html.escape(content)}</div></div>')


def _is_messages(v) -> bool:
    return isinstance(v, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in v)


def chat_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CHAT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    i, k = CHAT
    d = DOCUMENTS[i]
    before = answers(i, "chat")[:k]
    got = _call(lambda: fn(INSTRUCTION, list(d["nl"]), list(before)))
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is _SLOW:
        return _empty(w, "slow")
    if not _is_messages(got):
        return _empty(w, "odd")
    want = chat_messages(INSTRUCTION, d["nl"], before)
    if _same(got, want):
        lead = w["same"]
    elif len(got) != len(want):
        lead = w["differs"].format(why=w["count"].format(want=len(want), got=len(got)))
    else:
        first = next(j for j, (a, b) in enumerate(zip(got, want)) if not _same(a, b))
        lead = w["differs"].format(why=w["message"].format(k=first + 1))
    bubbles = "".join(_bubble(m["role"], m["content"]) for m in got[:12]) + (_faint("…") if len(got) > 12 else "")
    rows_ = [(w["in_chat"], _highlighted(i, answers(i, "chat")[k])), (w["alone"], _highlighted(i, answers(i, "alone")[k]))]
    body = _note(lead) + bubbles + _label(w["answered"]) + _rows(rows_) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_CONSISTENCY_TEXT = {
    "en": {"title": "The consistency of each term, by your function", "kind": "a number",
           "head": ["Text", "each sentence alone", "in one chat", "the whole text"],
           "mean": "mean", "course": "the course: {x}",
           "note": "Each chip is the word Qwen3 used for the term in one sentence, in the singular. Blue: a word in the meaning the term has in its text. Purple: a word with another meaning. Gray —: no word for the term, left out. Under the chips, your function's consistency of the words in the cell, in blue where it is the course's too; — where fewer than two words are left.",
           "off": "A number in purple is not the course's, which follows it."},
    "nl": {"title": "De consistentie van elke term, met je functie", "kind": "een getal",
           "head": ["Tekst", "elke zin los", "in één chat", "de hele tekst"],
           "mean": "gemiddeld", "course": "de course: {x}",
           "note": "Elk blokje is het woord dat Qwen3 in één zin voor de term gebruikte, in het enkelvoud. Blauw: een woord in de betekenis die de term in de tekst heeft. Paars: een woord met een andere betekenis. Grijs —: geen woord voor de term, weggelaten. Onder de blokjes de consistentie van de woorden in de cel volgens je functie, in blauw waar de course hetzelfde krijgt; — waar minder dan twee woorden over zijn.",
           "off": "Een paars getal is niet dat van de course, dat erachter staat."},
}


def consistency_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CONSISTENCY_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    import time
    start, rows_, means, off = time.perf_counter(), [], {way: [] for way in WAYS}, False
    for i, d in enumerate(DOCUMENTS):
        row = [html.escape(f"{d['place']}: {d['term']}")]
        for way in WAYS:
            chips = "".join(_chip("—", _CHIP_DATA) if x is None else _chip(x, _CHIP_PRED if right(i, x) else _CHIP_LOSS)
                            for x in found(i, way))
            words = words_found(i, way)
            if len(words) < 2:
                row.append(chips + f'<div>{_faint("—")}</div>')
                continue
            got = _call(lambda: fn(list(words)), PICTURE_SECONDS - (time.perf_counter() - start))
            if got is _STOPPED:
                return _empty(w, "stopped")
            if got is _SLOW:
                return _empty(w, "slow")
            if not _is_count(got):
                return _empty(w, "odd")
            want = consistency(words)
            if math.isclose(float(got), want, abs_tol=1e-9):
                cell = f'<span style="{_PRED}">{_dec(float(got), lang, 2)}</span>'
            else:
                off = True
                cell = (f'<span style="{_LOSS}">{_dec(float(got), lang, 2)}</span> '
                        + _faint(w["course"].format(x=_dec(want, lang, 2))))
            means[way].append(float(got))
            row.append(chips + f"<div>{cell}</div>")
        rows_.append(row)
    rows_.append([f"<b>{html.escape(w['mean'])}</b>"] + [f"<b>{_dec(sum(v) / len(v), lang, 2)}</b>" for v in means.values()])
    body = _table(rows_, w["head"]) + _note(w["note"]) + (_note(w["off"]) if off else "")
    return yours.figure(w["title"], body, alt=w["title"])
# --- the exercises: the course's solutions, the starts and the wrong versions ---------------------------------------------

_NB_HEAD = "def numbered(sentences):\n    lines = []\n    for k in range(len(sentences)):\n"
_NB_LINE = '        lines.append(f"{k + 1}. {sentences[k]}")\n'
_NB_END = '    return "\\n".join(lines)\n'


def _nb(line: str = _NB_LINE, end: str = _NB_END) -> str:
    return _NB_HEAD + line + end


_NB = _nb()
_NB_START = _nb("        lines.append(...)\n")
_NB_ZERO = _nb('        lines.append(f"{k}. {sentences[k]}")\n')
_NB_NO_SPACE = _nb('        lines.append(f"{k + 1}.{sentences[k]}")\n')
_NB_NO_DOT = _nb('        lines.append(f"{k + 1} {sentences[k]}")\n')
_NB_WHOLE = _nb('        lines.append(f"{k + 1}. {sentences}")\n')
_NB_SENTENCE = _nb("        lines.append(sentences[k])\n")

_CM_HEAD = ("def chat_messages(instruction, sentences, answers):\n    messages = []\n    for k in range(len(answers)):\n"
            '        messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[k]})\n')
_CM_ANSWER = '        messages.append({"role": "assistant", "content": answers[k]})\n'
_CM_NEXT = '    messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[len(answers)]})\n'
_CM_END = "    return messages\n"


def _cm(answer: str = _CM_ANSWER, next_: str = _CM_NEXT, end: str = _CM_END) -> str:
    return _CM_HEAD + answer + next_ + end


_CM = _cm()
_CM_START = _cm("        messages.append(...)\n", "    messages.append(...)\n")
_CM_NO_ANSWER = _cm("")
_CM_USER = _cm('        messages.append({"role": "user", "content": answers[k]})\n')
_CM_SENTENCE = _cm('        messages.append({"role": "assistant", "content": sentences[k]})\n')
_CM_NO_NEXT = _cm(next_="")
_CM_LAST = _cm(next_='    messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[-1]})\n')
_CM_AGAIN = _cm(next_='    messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[len(answers) - 1]})\n')
_CM_BARE = _cm(next_='    messages.append({"role": "user", "content": sentences[len(answers)]})\n')
_CM_NO_BLANK = _cm(next_='    messages.append({"role": "user", "content": instruction + sentences[len(answers)]})\n')

_CS_HEAD = "def consistency(translations):\n    pairs = 0\n    same = 0\n    for i in range(len(translations)):\n"
_CS_INNER = "        for j in range(i + 1, len(translations)):\n"
_CS_COUNT = "            pairs = pairs + 1\n"
_CS_IF = "            if translations[i] == translations[j]:\n"
_CS_END = "                same = same + 1\n    return same / pairs\n"


def _cs(inner: str = _CS_INNER, check: str = _CS_IF, end: str = _CS_END) -> str:
    return _CS_HEAD + inner + _CS_COUNT + check + end


_CS = _cs()
_CS_START = _cs("        for j in range(...):\n", "            if ...:\n")
_CS_ALL = _cs("        for j in range(len(translations)):\n")
_CS_SELF = _cs("        for j in range(i, len(translations)):\n")
_CS_SHORT = _cs("        for j in range(i + 1, len(translations) - 1):\n")
_CS_DIFFER = _cs(check="            if translations[i] != translations[j]:\n")
_CS_BY_LEN = _cs(end="                same = same + 1\n    return same / len(translations)\n")
_CS_FLOOR = _cs(end="                same = same + 1\n    return same // pairs\n")


class _Val:
    """A test value that the check's messages write by its name, such as `C.words_found(...)`; the function gets a
    copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (float, np.floating)):
        return f"{round(float(v), 4)}"
    if isinstance(v, np.integer):
        return repr(int(v))
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _show(v):
    """A value for a message, the way the lesson writes it; a number with at most four decimals."""
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text_, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The small cases the exercise texts show.
TOY_NUMBERED = (["Het regent.", "De weg is glad."],)
TOY_CHAT = ("Translate.", ["Het regent.", "De weg is glad."], ["It is raining."])
TOY_CONSISTENCY = ((["leaf", "paper", "paper"],), (["menu", "menu", "menu", "menu"],))


def _nl_val(i: int) -> _Val:
    return _Val(f'C.DOCUMENTS[{i}]["nl"]', list(DOCUMENTS[i]["nl"]))


def _chat_val(i: int, k: int) -> _Val:
    return _Val(f'C.answers({i}, "chat")[:{k}]', answers(i, "chat")[:k])


def _found_val(i: int, way: str) -> _Val:
    return _Val(f'C.words_found({i}, "{way}")', words_found(i, way))


def _not_text(got, text_) -> bool:
    return not isinstance(got, str)


def _not_messages(got, text_) -> bool:
    return not isinstance(got, list)


def _not_number(got, text_) -> bool:
    return not _is_count(got)


REQUIRED = ["numbered", "chat_messages", "consistency", "same_wrong"]

EXERCISES = {
    "numbered": FunctionBox(
        name="numbered",
        picture=numbered_picture,
        function="numbered",
        start=_NB_START,
        solution=_NB,
        show=_show,
        visible=[tuple(_written(x) for x in TOY_NUMBERED)],
        hidden=[(_written([]),), (_written(["Ja."]),),
                (_written(["Een.", "Twee.", "Drie.", "Vier.", "Vijf.", "Zes.", "Zeven.", "Acht.", "Negen.", "Tien.", "Elf."]),)]
               + [(_nl_val(i),) for i in range(len(DOCUMENTS))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-text", matches=_not_text,
                    en="`{call}` gives {got}, and it should give {want}: one text. Does your function still give back `\"\\n\".join(lines)`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één tekst. Geeft je functie nog steeds `\"\\n\".join(lines)` terug?"),
            Mistake("from-zero", wrong=_NB_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: the first line has the number 0. `k` is the place of a sentence in the list, and places start at 0. Which number should the line of the first sentence have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de eerste regel heeft het nummer 0. `k` is de plaats van een zin in de lijst, en plaatsen beginnen bij 0. Welk nummer hoort de regel van de eerste zin te hebben?"),
            Mistake("no-space", wrong=_NB_NO_SPACE,
                    en="`{call}` gives {got}, and it should give {want}: the sentence comes straight after the full stop. What comes between the full stop and the sentence?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de zin staat direct achter de punt. Wat komt er tussen de punt en de zin?"),
            Mistake("no-dot", wrong=_NB_NO_DOT,
                    en="`{call}` gives {got}, and it should give {want}: the number has no full stop after it. How does a numbered line start, as in `1. Het regent.`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: na het nummer staat geen punt. Hoe begint een genummerde regel, zoals in `1. Het regent.`?"),
            Mistake("whole-list", wrong=_NB_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: every line holds the whole list `sentences`. Which one sentence belongs on the line of place `k`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke regel bevat de hele lijst `sentences`. Welke ene zin hoort op de regel van plaats `k`?"),
            Mistake("no-numbers", wrong=_NB_SENTENCE,
                    en="`{call}` gives {got}, and it should give {want}: the lines have no numbers, so Qwen3 cannot give back a line for each sentence. What comes before each sentence?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de regels hebben geen nummers, dus Qwen3 kan geen regel per zin teruggeven. Waarmee begint elke regel?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: the line for sentence `k`, with its number, a full stop and a space before it.",
                      "nl": "Vervang de `...`: de regel voor zin `k`, met vooraan het nummer, een punt en een spatie."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each line start with the number of its sentence, counted from 1, then a full stop, a space and the sentence?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Begint elke regel met het nummer van de zin, geteld vanaf 1, dan een punt, een spatie en de zin?"},
            "almost": {"en": "Right for the small case. But `{call}` gives {got}, and it should give {want}. Does your function also work for an empty list, and for a text of more than nine sentences?",
                       "nl": "Goed voor het kleine geval. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een lege lijst, en voor een tekst van meer dan negen zinnen?"},
            "correct": {"en": "Right: your function numbers the lines of a text, so Qwen3 can give back one line for each sentence. In the picture below, which lines of Qwen3's answer keep the name Echt, and which word does it use for *bord* each time?",
                        "nl": "Goed: je functie nummert de regels van een tekst, zodat Qwen3 voor elke zin één regel kan teruggeven. Welke regels van het antwoord van Qwen3 in het plaatje hieronder houden de naam Echt, en welk woord gebruikt Qwen3 elke keer voor *bord*?"},
        },
    ),
    "chat_messages": FunctionBox(
        name="chat_messages",
        picture=chat_picture,
        function="chat_messages",
        start=_CM_START,
        solution=_CM,
        show=_show,
        visible=[tuple(_written(x) for x in TOY_CHAT)],
        hidden=[(_written("Translate."), _written(["Ja."]), _written([])),
                (_written("Translate."), _written(["Een.", "Twee.", "Drie."]), _written(["One.", "Two."])),
                (_written("Vertaal."), _written(["Een.", "Twee.", "Drie."]), _written(["One."]))]
               + [(_Val("C.INSTRUCTION", INSTRUCTION), _nl_val(i), _chat_val(i, k)) for i in (5, 7) for k in range(5)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_messages,
                    en="`{call}` gives {got}, and it should give {want}: a list of messages. Does your function still give back `messages`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met berichten. Geeft je functie nog steeds `messages` terug?"),
            Mistake("no-answers", wrong=_CM_NO_ANSWER,
                    en="`{call}` gives {got}, and it should give {want}: the chat has no message with the role `assistant`, so Qwen3 does not see its own earlier translations. What comes after each sentence in the chat?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de chat heeft geen bericht met de rol `assistant`, dus Qwen3 ziet de eigen eerdere vertalingen niet. Wat komt er in de chat na elke zin?"),
            Mistake("user-answers", wrong=_CM_USER,
                    en="`{call}` gives {got}, and it should give {want}: Qwen3's earlier translation has the role `user`, as if you had written it. Which role do the model's own answers have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de eerdere vertaling van Qwen3 heeft de rol `user`, alsof jij het schreef. Welke rol hebben de eigen antwoorden van het model?"),
            Mistake("sentence-answers", wrong=_CM_SENTENCE,
                    en="`{call}` gives {got}, and it should give {want}: the message with the role `assistant` holds the Dutch sentence again. What did Qwen3 answer to sentence `k`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het bericht met de rol `assistant` bevat de Nederlandse zin nog een keer. Wat antwoordde Qwen3 op zin `k`?"),
            Mistake("no-next", wrong=_CM_NO_NEXT,
                    en="`{call}` gives {got}, and it should give {want}: the chat ends with Qwen3's own answer, so it has no new sentence to translate. What should the last message ask?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de chat eindigt met het eigen antwoord van Qwen3, dus er is geen nieuwe zin om te vertalen. Wat moet het laatste bericht vragen?"),
            Mistake("again", wrong=_CM_AGAIN,
                    en="`{call}` gives {got}, and it should give {want}: the last message asks again for a sentence Qwen3 has already translated. How many sentences has Qwen3 translated so far, and at which place in `sentences` is the next one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het laatste bericht vraagt opnieuw om een zin die Qwen3 al vertaalde. Hoeveel zinnen heeft Qwen3 tot nu toe vertaald, en op welke plaats in `sentences` staat de volgende?"),
            Mistake("last", wrong=_CM_LAST,
                    en="`{call}` gives {got}, and it should give {want}: the last message asks for the last sentence of the text, not for the one after the sentences Qwen3 has translated. At which place in `sentences` is that one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het laatste bericht vraagt om de laatste zin van de tekst, en niet om de zin na de zinnen die Qwen3 vertaalde. Op welke plaats in `sentences` staat de eerste zin die Qwen3 nog niet vertaalde?"),
            Mistake("bare", wrong=_CM_BARE,
                    en="`{call}` gives {got}, and it should give {want}: the last message has the sentence without the instruction. What comes before the sentence in the messages of the loop?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het laatste bericht heeft de zin zonder de instructie. Waarmee begint elk bericht van de loop?"),
            Mistake("no-blank", wrong=_CM_NO_BLANK,
                    en="`{call}` gives {got}, and it should give {want}: the sentence comes straight after the instruction. What comes between the two in the messages of the loop?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de zin staat direct achter de instructie. Wat staat er in de berichten van de loop tussen de instructie en de zin?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: in the loop, Qwen3's answer to sentence `k`, and after the loop, the message that asks for the next sentence.",
                      "nl": "Vervang de twee `...`: in de loop het antwoord van Qwen3 op zin `k`, en na de loop het bericht dat om de volgende zin vraagt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each sentence so far come with the role `user`, followed by its answer with the role `assistant`, and does the chat end with the next sentence?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Komt elke zin tot nu toe met de rol `user`, gevolgd door het antwoord met de rol `assistant`, en eindigt de chat met de volgende zin?"},
            "almost": {"en": "Right for the small case. But `{call}` gives {got}, and it should give {want}. Does your function also work at the start of a text, with no answers yet, and in the middle of a longer text?",
                       "nl": "Goed voor het kleine geval. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook aan het begin van een tekst, zonder antwoorden, en in het midden van een langere tekst?"},
            "correct": {"en": "Right: your function makes the chat in which Qwen3 translates a text sentence by sentence, with its own translations so far. In the picture below, what does Qwen3 write for the third sentence in this chat, and what without the chat?",
                        "nl": "Goed: je functie maakt de chat waarin Qwen3 een tekst zin voor zin vertaalt, met de eigen vertalingen tot nu toe. Wat schrijft Qwen3 in het plaatje hieronder bij de derde zin in deze chat, en wat zonder de chat?"},
        },
    ),
    "consistency": FunctionBox(
        name="consistency",
        picture=consistency_picture,
        function="consistency",
        start=_CS_START,
        solution=_CS,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_CONSISTENCY],
        hidden=[(_written(["a", "a"]),), (_written(["a", "b"]),), (_written(["a", "b", "a", "b"]),),
                (_written(["a", "a", "b", "c"]),), (_written(["x", "x", "x", "y", "y"]),)]
               + [(_found_val(i, way),) for i in range(len(DOCUMENTS)) for way in WAYS if len(words_found(i, way)) >= 2],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: a number from 0 to 1. Does your function still give back `same / pairs`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een getal van 0 tot 1. Geeft je functie nog steeds `same / pairs` terug?"),
            Mistake("all", wrong=_CS_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your loops pair each translation with itself, and every other pair twice, once in each order. At which place should `j` start, so that each pair comes once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loops maken van elke vertaling ook een paar met zichzelf, en van elk ander paar twee, in beide volgordes. Op welke plaats moet `j` beginnen, zodat elk paar één keer komt?"),
            Mistake("self", wrong=_CS_SELF,
                    en="`{call}` gives {got}, and it should give {want}: `j` starts at `i`, so each translation is also paired with itself, and that pair is always the same. At which place does the first translation after `i` stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `j` begint bij `i`, dus elke vertaling vormt ook een paar met zichzelf, en dat paar is altijd hetzelfde. Op welke plaats staat de eerste vertaling na `i`?"),
            Mistake("short", wrong=_CS_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: `j` stops one place before the end, so the last translation is never the second of a pair. Where does `range` stop?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `j` stopt één plaats te vroeg, dus de laatste vertaling is nooit de tweede van een paar. Waar stopt `range`?"),
            Mistake("differ", wrong=_CS_DIFFER,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the pairs that differ. Which pairs does consistency count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de paren die verschillen. Welke paren telt consistentie?"),
            Mistake("by-length", wrong=_CS_BY_LEN,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the number of translations, and not by the number of pairs. How many pairs do four translations make?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt door het aantal vertalingen, en niet door het aantal paren. Hoeveel paren maken vier vertalingen?"),
            Mistake("floor", wrong=_CS_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: `//` rounds the share down to a whole number, so anything below 1 becomes 0. Which division gives the share itself?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `//` rondt het aandeel af naar beneden op een heel getal, dus alles onder 1 wordt 0. Welke deling geeft het aandeel zelf?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the numbers `j` runs over, and the question whether the two translations of the pair are the same.",
                      "nl": "Vervang de twee `...`: de getallen waar `j` langs loopt, en de vraag of de twee vertalingen van het paar hetzelfde zijn."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function look at each pair of two different places once, and count the pairs whose translations are the same?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bekijkt je functie elk paar van twee verschillende plaatsen één keer, en telt het de paren met dezelfde vertaling?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function also work for two translations, and for translations that come back later in the list?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor twee vertalingen, en voor vertalingen die later in de lijst terugkomen?"},
            "correct": {"en": "Right: your function computes the consistency of a term, as Wang and colleagues did. In the picture below, in which way is the term most often the same, and is it then also right?",
                        "nl": "Goed: je functie berekent de consistentie van een term, zoals Wang en collega's deden. Op welke manier is de term in het plaatje hieronder het vaakst hetzelfde, en is de term dan ook goed?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("numbered", _NB_START, "empty", None),
    ("numbered", _nb("        lines.append(...)\n", '    return "\\n".join(lines) + "\\n"\n'), "wrong", "dots-left"),
    ("numbered", _nb('        lines.append(f"{k + 1}. {sentences[k]}")  # ...\n'), "correct", None),
    ("numbered", _nb('        lines.append(f"{k + 1}. {sentences[k]}" + "..."[:0])\n'), "correct", None),
    ("numbered", _NB, "correct", None),
    ("numbered", _nb(end='    return lines\n'), "wrong", "not-text"),
    ("numbered", _NB_ZERO, "wrong", "from-zero"),
    ("numbered", _NB_NO_SPACE, "wrong", "no-space"),
    ("numbered", _NB_NO_DOT, "wrong", "no-dot"),
    ("numbered", _NB_WHOLE, "wrong", "whole-list"),
    ("numbered", _NB_SENTENCE, "wrong", "no-numbers"),
    ("numbered", _nb('        lines.append(str(k + 1) + ". " + sentences[k])\n'), "correct", None),
    ("numbered", 'def numbered(sentences):\n    text = ""\n    for k in range(len(sentences)):\n'
                 '        if k > 0:\n            text = text + "\\n"\n        text = text + f"{k + 1}. {sentences[k]}"\n'
                 '    return text\n', "correct", None),
    ("chat_messages", _CM_START, "empty", None),
    ("chat_messages", _CM, "correct", None),
    ("chat_messages", _cm(next_="    messages.append(...)\n"), "wrong", "dots-left"),
    ("chat_messages", _cm(end="    return messages[-1]\n"), "wrong", "not-list"),
    ("chat_messages", _CM_NO_ANSWER, "wrong", "no-answers"),
    ("chat_messages", _CM_USER, "wrong", "user-answers"),
    ("chat_messages", _CM_SENTENCE, "wrong", "sentence-answers"),
    ("chat_messages", _CM_NO_NEXT, "wrong", "no-next"),
    ("chat_messages", _CM_AGAIN, "wrong", "again"),
    ("chat_messages", _CM_LAST, "wrong", "last"),
    ("chat_messages", _CM_BARE, "wrong", "bare"),
    ("chat_messages", _CM_NO_BLANK, "wrong", "no-blank"),
    ("chat_messages", _cm(next_='    k = len(answers)\n    messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[k]})\n'),
     "correct", None),
    ("consistency", _CS_START, "empty", None),
    ("consistency", _CS, "correct", None),
    ("consistency", _cs(check="            if ...:\n"), "wrong", "dots-left"),
    ("consistency", _cs(end="                same = same + 1\n    return str(same / pairs)\n"), "wrong", "not-number"),
    ("consistency", _CS_ALL, "wrong", "all"),
    ("consistency", _CS_SELF, "wrong", "self"),
    ("consistency", _CS_SHORT, "wrong", "short"),
    ("consistency", _CS_DIFFER, "wrong", "differ"),
    ("consistency", _CS_BY_LEN, "wrong", "by-length"),
    ("consistency", _CS_FLOOR, "wrong", "floor"),
    ("consistency", _cs("        for j in range(i):\n"), "correct", None),
    ("consistency", _cs(check="            if translations[j] == translations[i]:\n"), "correct", None),
]


# --- your own text, translated by Qwen3 on your computer -------------------------------------------------------------------

YOUR_TEXT = ("In Doorn krijgt elke inwoner een nieuwe bibliotheekpas.\nMet de pas leen je gratis boeken.\n"
             "Doorn betaalt de pas voor alle kinderen.")
MAX_LINES = 8   # the demo translates at most this many lines of the learner's text


def downloaded() -> bool:
    from everything_ai import chat
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    from everything_ai import chat
    chat.download_or_stop(button, texts)


_LIVE_TEXT = {
    "en": {"title": "Qwen3 on your text", "loading": "Loading Qwen3 …",
           "busy": "Qwen3 is translating: {way}, line {k} of {n} …", "whole": "Qwen3 is translating the whole text …",
           "head": ["Dutch", "each sentence alone", "in one chat", "the whole text"],
           "done": "{n} lines, {tokens} tokens in {seconds} seconds, greedy, as in the course's run.",
           "cut": "Your text has more than {m} lines; Qwen3 translated the first {m}.",
           "note": "Gray: your Dutch. Blue: what Qwen3 wrote on your computer.",
           "dash": "A — means that Qwen3's answer for the whole text has no line with this number.",
           "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
           "memory": "The memory of your device is full. Close other programs, and press the button again."},
    "nl": {"title": "Qwen3 op je tekst", "loading": "Qwen3 laden …",
           "busy": "Qwen3 vertaalt: {way}, regel {k} van {n} …", "whole": "Qwen3 vertaalt de hele tekst …",
           "head": ["Nederlands", "elke zin los", "in één chat", "de hele tekst"],
           "done": "{n} regels, {tokens} tokens in {seconds} seconden, greedy, zoals in de run van de course.",
           "cut": "Je tekst heeft meer dan {m} regels; Qwen3 vertaalde de eerste {m}.",
           "note": "Grijs: je Nederlands. Blauw: wat Qwen3 op je computer schreef.",
           "dash": "Een — betekent dat het antwoord van Qwen3 voor de hele tekst geen regel met dit nummer heeft.",
           "load": "Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
           "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."},
}


def your_lines(text_: str) -> list:
    """The lines of the learner's text that hold something, without spaces at the ends: one sentence per line."""
    return [line.strip() for line in str(text_).splitlines() if line.strip()]


def _live_picture(lines: list, got: dict, w: dict, notes: list) -> str:
    rows_ = []
    for k, dutch in enumerate(lines):
        cells = [_gray(dutch)]
        for way in WAYS:
            t = got[way][k] if k < len(got[way]) else None
            cells.append(f'<span style="{_PRED}">{html.escape(t)}</span>' if t else _faint("…" if t is None else "—"))
        rows_.append(cells)
    return yours.figure(w["title"], _table(rows_, w["head"]) + "".join(_note(x) for x in notes), alt=w["title"])


def translate_live(text_: str, show, lang: str) -> str:
    """Qwen3 on the learner's own Dutch text, one sentence per line, greedy, in the lesson's three ways, with the table
    so far shown by `show(html)` after each answer; then the whole table."""
    from everything_ai import chat
    lang = lang if lang in LANGS else "en"
    w = _LIVE_TEXT[lang]
    all_lines = your_lines(text_)
    lines = all_lines[:MAX_LINES]
    got = {way: [] for way in WAYS}
    show(_live_picture(lines, got, w, [w["loading"]]))
    try:
        model = chat.loaded()
    except Exception as e:   # noqa: BLE001
        return f'<p><em>{html.escape(w["memory" if chat.memory_full(e) else "load"])}</em></p>'
    tokens, seconds = 0, 0.0

    def ask(messages: list, max_tokens: int) -> str:
        nonlocal tokens, seconds
        a = model.answer(messages, max_tokens=max_tokens, temperature=0)
        tokens, seconds = tokens + a["tokens"], seconds + a["seconds"]
        return a["message"]["content"]
    try:
        for k, line in enumerate(lines):
            show(_live_picture(lines, got, w, [w["busy"].format(way=WAY_NAMES[lang]["alone"], k=k + 1, n=len(lines))]))
            got["alone"].append(ask(alone_messages(line), MAX_TOKENS))
        for k in range(len(lines)):
            show(_live_picture(lines, got, w, [w["busy"].format(way=WAY_NAMES[lang]["chat"], k=k + 1, n=len(lines))]))
            got["chat"].append(ask(chat_messages(INSTRUCTION, lines, got["chat"]), MAX_TOKENS))
        show(_live_picture(lines, got, w, [w["whole"]]))
        got["document"] = lines_of(ask(document_messages(lines), DOCUMENT_TOKENS), len(lines))
    except Exception as e:
        if chat.memory_full(e):
            return f'<p><em>{html.escape(w["memory"])}</em></p>'
        raise
    notes = [w["done"].format(n=len(lines), tokens=_num(tokens, lang), seconds=_dec(seconds, lang))]
    if len(all_lines) > MAX_LINES:
        notes.append(w["cut"].format(m=MAX_LINES))
    notes.append(w["note"])
    if not all(got["document"]):
        notes.append(w["dash"])
    return _live_picture(lines, got, w, notes)
