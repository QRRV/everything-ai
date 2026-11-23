# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-coreference: the chains of a text from the chain of each mention (chains), the nearest
earlier mention of the same chain for each mention (antecedents), and the rule "the nearest earlier mention of the
same kind" (pick); with the pictures of what each function gives on LitBank, three pictures of the idea and a demo.

The mentions are the coreference layer of LitBank that the course ships (everything_ai.litbank): about 2,000 words
from each of 100 works of English fiction, so nothing is downloaded. A mention is a person or a thing (its kind, see
everything_ai.litbank). Every number in the text and the pictures is computed by the course
(tests/test_nlp_coreference.py computes them again)."""
import ast
import contextlib
import html
import io
import re

from everything_ai import litbank, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, kept, with_time_limit

_KEY = "nlp-coreference"

# --- the course's data -----------------------------------------------------------------------------------------------

WORKS = litbank.works()        # 100 works, from 1719 to 1922
KINDS = litbank.KINDS          # person, thing
CATEGORIES = ("PRON", "PROP", "NOM")
TYPES = ("PER", "FAC", "LOC", "GPE", "ORG", "VEH")

# The course's own passage, with its eight mentions: (text, kind), and the chain of each.
MUSEUM_TEXT = "The museum bought a painting. It hangs in the hall. A guard straightens it, and visitors love it."
_MUSEUM = (("The museum", "thing"), ("a painting", "thing"), ("It", "thing"), ("the hall", "thing"),
           ("A guard", "person"), ("it", "thing"), ("visitors", "person"), ("it", "thing"))
_MUSEUM_IDS = ("museum", "painting", "painting", "hall", "guard", "painting", "visitors", "painting")
_MUSEUM_PIECES = (0, " bought ", 1, ". ", 2, " hangs in ", 3, ". ", 4, " straightens ", 5, ", and ", 6, " love ", 7, ".")
MUSEUM = list(_MUSEUM)
MUSEUM_IDS = list(_MUSEUM_IDS)

PRIDE = 1342   # Pride and Prejudice at Project Gutenberg: the work of the picture under exercise 1


# --- the course's own exercises --------------------------------------------------------------------------------------

def chains(mentions, ids):
    """Exercise 1: for each chain, the texts of its mentions, in the order of the text."""
    found = {}
    for k in range(len(mentions)):
        text = mentions[k][0]
        c = ids[k]
        if c in found:
            found[c].append(text)
        else:
            found[c] = [text]
    return found


def antecedents(ids):
    """Exercise 2: for each mention, the place of the nearest earlier mention of its chain, or None."""
    last = {}
    out = []
    for k in range(len(ids)):
        if ids[k] in last:
            out.append(last[ids[k]])
        else:
            out.append(None)
        last[ids[k]] = k
    return out


def pick(mentions, i):
    """Exercise 3: the place of the nearest mention before mention i with the same kind, or None."""
    kind = mentions[i][1]
    best = None
    for j in range(i):
        if mentions[j][1] == kind:
            best = j
    return best


def nearest(mentions, i):
    """The rule without the kind: the place of the mention right before mention i, or None."""
    return i - 1 if i > 0 else None


# --- the cases of *it* -----------------------------------------------------------------------------------------------

def is_it(m) -> bool:
    """A mention that is it, its or itself."""
    return m.category == "PRON" and m.text.lower() in litbank.THING_PRONOUNS


def _it_cases() -> tuple:
    """Each it, its or itself of LitBank whose chain has an earlier mention: (work, mention, the mentions before it)."""
    out = []
    for w_i, work in enumerate(WORKS):
        for i, m in enumerate(work.mentions):
            if is_it(m):
                before = litbank.before(work, i)
                if any(work.mentions[j].chain == m.chain for j in before):
                    out.append((w_i, i, tuple(before)))
    return tuple(out)


IT_CASES = kept(f"{_KEY}:cases", _it_cases)   # 318


def case_mentions(case) -> list:
    """The mentions of a case as the exercises write them: a new list of (text, kind), the mentions before the
    pronoun and the pronoun last."""
    w_i, i, before = case
    work = WORKS[w_i]
    return [(work.mentions[j].text, work.mentions[j].kind) for j in before] + [(work.mentions[i].text, "thing")]


def is_right(case, j) -> bool:
    """Whether place j of the mentions of a case is in the chain of the pronoun."""
    w_i, i, before = case
    if not (isinstance(j, int) and not isinstance(j, bool) and 0 <= j < len(before)):
        return False
    work = WORKS[w_i]
    return work.mentions[before[j]].chain == work.mentions[i].chain


def _demo_cases() -> tuple:
    """The cases of the demo: the ones whose pronoun, nearest earlier mention of its chain and both picks of the rule
    lie at most two sentences apart, in sentences a lesson shows, with at most 110 tokens in all."""
    out = []
    for case in IT_CASES:
        w_i, i, before = case
        work = WORKS[w_i]
        m = work.mentions[i]
        mentions = case_mentions(case)
        places = [before[pick(mentions, len(before))], before[-1],
                  max(j for j in before if work.mentions[j].chain == m.chain)]
        first = min(work.mentions[j].sentence for j in places)
        shown = range(first, m.sentence + 1)
        if (m.sentence - first <= 2 and all(work.shown[s] for s in shown)
                and sum(len(work.sentences[s]) for s in shown) <= 110):
            out.append((case, first))
    return tuple(out)


DEMO_CASES = kept(f"{_KEY}:demo", _demo_cases)   # 246


def _demo_start() -> int:
    """The case the slider starts at: *it* for Netherfield Park in Pride and Prejudice, counted from 1."""
    for n, ((w_i, i, _), _) in enumerate(DEMO_CASES):
        work = WORKS[w_i]
        if work.gutenberg == PRIDE and work.mentions[i].sentence == 4:
            return n + 1
    return 1


DEMO_START = _demo_start()


# --- the numbers in the text -----------------------------------------------------------------------------------------

def label_of(work, chain) -> str:
    """How a picture names a chain: its first name, else its first description, else its first mention."""
    found = [m for m in work.mentions if m.chain == chain]
    for category in ("PROP", "NOM"):
        for m in found:
            if m.category == category:
                return m.text
    return found[0].text


_BUCKETS = ("1", "2", "3", "4", "5", "6-9", "10+")


def bucket(d: int) -> str:
    """The bar of a distance in mentions."""
    return str(d) if d <= 5 else "6-9" if d <= 9 else "10+"


def distances(work, out) -> list:
    """For each pronoun of a work with a place before it in out (what antecedents gives): how many mentions back."""
    return [k - out[k] for k, m in enumerate(work.mentions)
            if m.category == "PRON" and isinstance(out[k], int) and not isinstance(out[k], bool) and 0 <= out[k] < k]


def numbers() -> dict:
    """Every number the lesson states, computed from the data and the course's functions."""
    def _go():
        every = [m for w in WORKS for m in w.mentions]
        out = {"works": len(WORKS), "first_year": min(w.year for w in WORKS), "last_year": max(w.year for w in WORKS),
               "mentions": len(every), "tokens": sum(len(s) for w in WORKS for s in w.sentences),
               "categories": {c: sum(m.category == c for m in every) for c in CATEGORIES},
               "types": {t: sum(m.type == t for m in every) for t in TYPES},
               "kinds": {k: sum(m.kind == k for m in every) for k in KINDS}}
        out["per_share"] = out["types"]["PER"] / len(every)
        sizes = [list(chains([(m.text, m.kind) for m in w.mentions], [m.chain for m in w.mentions]).values())
                 for w in WORKS]
        out["entities"] = sum(len(s) for s in sizes)
        out["singletons"] = sum(len(c) == 1 for s in sizes for c in s)
        out["singleton_share"] = out["singletons"] / len(every)
        backs = [antecedents([m.chain for m in w.mentions]) for w in WORKS]
        has = {c: 0 for c in CATEGORIES}
        for w, b in zip(WORKS, backs):
            for m, x in zip(w.mentions, b):
                if x is not None:
                    has[m.category] += 1
        out["anaphors"] = sum(has.values())
        out["anaphor_share"] = out["anaphors"] / len(every)
        out["with_antecedent"] = {c: has[c] / out["categories"][c] for c in CATEGORIES}
        d = sorted(x for w, b in zip(WORKS, backs) for x in distances(w, b))
        out["pron_anaphors"] = len(d)
        out["median"] = d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
        out["within5"] = sum(x <= 5 for x in d) / len(d)
        out["within9"] = sum(x <= 9 for x in d) / len(d)
        out["buckets"] = {b: sum(bucket(x) == b for x in d) / len(d) for b in _BUCKETS}
        out["it_tokens"] = sum(t.lower() in litbank.THING_PRONOUNS for w in WORKS for s in w.sentences for t in s)
        out["it_mentions"] = sum(is_it(m) for m in every)
        out["cases"] = len(IT_CASES)
        kind = [is_right(c, pick(case_mentions(c), len(c[2]))) for c in IT_CASES]
        near = [is_right(c, nearest(case_mentions(c), len(c[2]))) for c in IT_CASES]
        out["right_kind"], out["right_near"] = sum(kind), sum(near)
        out["kind_share"], out["near_share"] = sum(kind) / len(IT_CASES), sum(near) / len(IT_CASES)
        back = {"0": 0, "1": 0, "2": 0, "3+": 0}
        for w_i, i, before in IT_CASES:
            work = WORKS[w_i]
            m = work.mentions[i]
            gap = m.sentence - work.mentions[max(j for j in before if work.mentions[j].chain == m.chain)].sentence
            back[str(gap) if gap < 3 else "3+"] += 1
        out["sentences_back"] = back
        person = [c for c in IT_CASES if any(x.chain == WORKS[c[0]].mentions[c[1]].chain and x.type == "PER"
                                             for x in WORKS[c[0]].mentions)]
        out["it_person"] = len(person)
        out["it_person_right"] = sum(is_right(c, pick(case_mentions(c), len(c[2]))) for c in person)
        out["museum"] = {"kind": [pick(list(_MUSEUM), i) for i in (2, 5, 7)],
                         "near": [nearest(list(_MUSEUM), i) for i in (2, 5, 7)]}
        demo = [c for c, _ in DEMO_CASES]
        out["demo"] = {"n": len(demo), "kind": sum(is_right(c, pick(case_mentions(c), len(c[2]))) for c in demo),
                       "near": sum(is_right(c, nearest(case_mentions(c), len(c[2]))) for c in demo)}
        work = pride()
        found = chains([(m.text, m.kind) for m in work.mentions], [m.chain for m in work.mentions])
        top = sorted(found, key=lambda c: (-len(found[c]), c))[:5]
        out["pride"] = {"mentions": len(work.mentions), "chains": len(found),
                        "top": [(c, label_of(work, c), list(found[c])) for c in top]}
        return out
    return kept(f"{_KEY}:numbers", _go)


def pride():
    return next(w for w in WORKS if w.gutenberg == PRIDE)


# --- pictures: helpers -----------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_HEAD_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);font-weight:600;margin-top:0.6rem"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:1px;border-radius:3px"
_ROW = "display:flex;align-items:center;gap:0.5rem;margin:0.15rem 0"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);width:11rem;flex:none"
_TRACK = "flex:1;min-width:3rem;height:0.65rem;background:var(--ea-line);border-radius:2px"
_NUM = "font-family:var(--ea-font-code);font-size:0.8rem;width:3.6rem;flex:none;text-align:right"
_SMALL = "font-family:var(--ea-font-code);font-size:0.72rem;flex:none"
_CODE = "font-family:var(--ea-font-code);font-size:0.82rem"
_TEXT = "font-family:var(--ea-font-prose);font-size:0.98rem;line-height:2.1"
_PART = "display:inline-block;margin:0 0.1rem;text-align:center;vertical-align:top;line-height:1.3"
_WORD = ("display:block;border-radius:5px;padding:0.1rem 0.35rem;font-family:var(--ea-font-prose);font-size:0.95rem;"
         "color:var(--ea-ink);border:1px solid var(--ea-ink-3)")
_TAG = "display:block;font-family:var(--ea-font-code);font-size:0.7rem;margin-top:0.15rem;color:var(--ea-ink)"
_CELL = "font-family:var(--ea-font-ui);font-size:0.85rem;padding:0.2rem 0.5rem 0.2rem 0;vertical-align:top"
_ROLE = {   # the line or bar and the color of its text: the data's, the course's, the learner's, or wrong
    "data": ("var(--ea-ink-3)", "var(--ea-ink-2)"),
    "course": ("var(--ea-ink)", "var(--ea-ink)"),
    "mine": ("var(--ea-pred)", "var(--ea-pred-text)"),
    "off": ("var(--ea-loss)", "var(--ea-loss-text)"),
}


def _marked(text: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _said(template: str, **values) -> tuple:
    """A note as HTML and as plain text, with the values filled in after the marking: a * or a ` in a value stays as
    it is."""
    out, plain = _marked(template), _bare(template)
    for key, value in values.items():
        out = out.replace("{" + key + "}", html.escape(str(value)))
        plain = plain.replace("{" + key + "}", str(value))
    return out, plain


def _head(text: str) -> str:
    return f'<div style="{_HEAD_UI}">{_marked(text)}</div>'


def _n(x: int, lang: str) -> str:
    """A whole number with a separator for thousands: 28,411 in English, 28.411 in Dutch."""
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _d(x: float, lang: str, digits: int = 3) -> str:
    """A number from 0 to 1 with three decimals: 0.679 in English, 0,679 in Dutch."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _repr(v) -> str:
    """How Python shows a value, also for a value whose repr stops with an error."""
    try:
        return repr(v)
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return "<" + type(v).__name__ + ">"


def _short(v, limit: int = 60) -> str:
    text = _repr(v)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _is_share(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1


def _same_number(a, b) -> bool:
    return _is_share(a) and abs(float(a) - b) < 1e-9


def _bar(label: str, value, role: str, lang: str, off: bool = False) -> str:
    """A row with a label, a bar from 0 to 1 and the number; a value that is not a number from 0 to 1 gets no bar."""
    line, color = _ROLE[role]
    fine = _is_share(value)
    width = float(value) * 100 if fine else 0.0
    number = _d(float(value), lang) if fine else _short(value, 12)
    style = _NUM + f";color:{color}" + (";" + _OFF if off else "")
    return (f'<div style="{_ROW}"><span style="{_LABEL}">{_marked(label)}</span>'
            f'<span style="{_TRACK}"><span style="display:block;width:{width:.1f}%;height:100%;background:{line};'
            f'border-radius:2px"></span></span><span style="{style}">{html.escape(number)}</span></div>')


def _figure(title: str, rows: list, said: list, notes: list) -> str:
    body = "".join(rows) + "".join(f'<div style="{_UI}">{h}</div>' for h, _ in notes)
    return yours.figure(title, body, alt=f'{title}. {"; ".join(said)}. {" ".join(p for _, p in notes)}')


_NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "”", "’", "'", "n't", "n’t", "...", "—"}
_NO_SPACE_AFTER = {"“", "(", "[", "‘", "—"}


def _space_before(prev: str, token: str) -> bool:
    """Whether a space goes between two tokens of LitBank when a sentence is shown as text."""
    if prev in _NO_SPACE_AFTER or token in _NO_SPACE_BEFORE:
        return False
    return not (len(token) > 1 and token[0] in "'’" and token[1:].isalpha())


# --- pictures of the idea --------------------------------------------------------------------------------------------

_LETTERS = "ABCDE"

_MUSEUM_WORDS = {
    "en": {"title": "One passage, eight mentions, five chains",
           "chains": "The chains",
           "legend": "Under each mention: the letter of its chain and its kind. A chain holds the mentions of one "
                     "person or thing, in the order of the text; the course marked them by hand.",
           "names": {"museum": "the museum", "painting": "the painting", "hall": "the hall", "guard": "the guard",
                     "visitors": "the visitors"}},
    "nl": {"title": "Eén stukje tekst, acht mentions, vijf ketens",
           "chains": "De ketens",
           "legend": "Onder elke mention: de letter van de keten en de soort. Een keten bevat de mentions van één "
                     "persoon of ding, in de volgorde van de tekst; de course markeerde de mentions met de hand.",
           "names": {"museum": "het museum", "painting": "het schilderij", "hall": "de hal", "guard": "de bewaker",
                     "visitors": "de bezoekers"}},
}


def museum_picture(lang: str) -> str:
    """The course's passage with each mention in a box, the letter of its chain and its kind under it, and the chains."""
    w = _MUSEUM_WORDS[lang]
    letter = {}
    for c in _MUSEUM_IDS:
        if c not in letter:
            letter[c] = _LETTERS[len(letter)]
    parts = []
    for piece in _MUSEUM_PIECES:
        if isinstance(piece, int):
            text, kind = _MUSEUM[piece]
            parts.append(f'<span style="{_PART}"><span style="{_WORD}">{html.escape(text)}</span>'
                         f'<span style="{_TAG}">{letter[_MUSEUM_IDS[piece]]} · {kind}</span></span>')
        else:
            parts.append(html.escape(piece))
    rows = [f'<div style="{_TEXT}">{"".join(parts)}</div>', _head(w["chains"])]
    said = [MUSEUM_TEXT]
    found = chains(list(_MUSEUM), list(_MUSEUM_IDS))
    for c, texts in found.items():
        rows.append(f'<div style="{_ROW}"><span style="{_SMALL};color:var(--ea-ink);width:1.2rem">{letter[c]}</span>'
                    f'<span style="{_LABEL};width:8rem">{html.escape(w["names"][c])}</span>'
                    f'<span style="{_CODE};color:var(--ea-ink)">{html.escape(" · ".join(texts))}</span></div>')
        said.append(f'{letter[c]} {w["names"][c]}: {", ".join(texts)}')
    return _figure(w["title"], rows, said, [_said(w["legend"])])


_RULE_WORDS = {
    "en": {"title": "Two rules for the three times 'it' in the passage",
           "head": ("Mention", "Nearest earlier mention", "Nearest earlier mention of its kind", "Chain"),
           "chain": "the painting",
           "total": "Right: {near} of 3 without the kind, {kind} of 3 with it.",
           "legend": "In the color of the text: the mention each rule picks, with its place in the passage. Purple "
                     "dashed: a pick outside the chain of the painting. Gray: the chain the course marked."},
    "nl": {"title": "Twee regels voor de drie keer 'it' in de tekst",
           "head": ("Mention", "Dichtstbijzijnde eerdere mention", "Dichtstbijzijnde eerdere mention van dezelfde soort",
                    "Keten"),
           "chain": "het schilderij",
           "total": "Goed: {near} van de 3 zonder de soort, {kind} van de 3 met de soort.",
           "legend": "In de kleur van de tekst: de mention die elke regel kiest, met de plaats in de tekst. Paars "
                     "gestreept: een keuze buiten de keten van het schilderij. Grijs: de keten die de course markeerde."},
}


def rule_picture(lang: str) -> str:
    """The three pronouns of the passage, with the pick of the rule without and with the kind, and their chain."""
    w, nums = _RULE_WORDS[lang], numbers()
    head = "".join(f'<th style="{_CELL};color:var(--ea-ink-2);font-weight:600;text-align:left">{html.escape(h)}</th>'
                   for h in w["head"])
    body, said, right = [], [], {"near": 0, "kind": 0}
    for n, i in enumerate((2, 5, 7)):
        cells = [f'<td style="{_CELL};color:var(--ea-ink)"><code>{i}</code> {html.escape(_MUSEUM[i][0])}</td>']
        for key in ("near", "kind"):
            j = nums["museum"][key][n]
            fine = _MUSEUM_IDS[j] == "painting"
            right[key] += fine
            style = f"{_CELL};color:var(--ea-ink)" + ("" if fine else ";" + _OFF)
            cells.append(f'<td style="{style}"><code>{j}</code> {html.escape(_MUSEUM[j][0])}</td>')
        cells.append(f'<td style="{_CELL};color:var(--ea-ink-2)">{html.escape(w["chain"])}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
        said.append(f'{i} {_MUSEUM[i][0]}: {nums["museum"]["near"][n]}, {nums["museum"]["kind"][n]}')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse"><tr>{head}</tr>{"".join(body)}'
             f'</table></div>')
    return _figure(w["title"], [table], said, [_said(w["total"], **right), _said(w["legend"])])


_BACK_WORDS = {
    "en": {"title": "How far back the chain of 'it' was last mentioned, in sentences",
           "labels": {"0": "In the same sentence", "1": "One sentence back", "2": "Two sentences back",
                      "3+": "Three or more back"},
           "legend": "Each bar: the share of the {cases} cases of *it*, *its* or *itself* in LitBank whose chain was "
                     "mentioned before, by how many sentences back the nearest earlier mention of that chain stands; a "
                     "full bar is 1. In the color of the text: the course's count."},
    "nl": {"title": "Hoe ver terug de keten van 'it' het laatst genoemd werd, in zinnen",
           "labels": {"0": "In dezelfde zin", "1": "Eén zin terug", "2": "Twee zinnen terug",
                      "3+": "Drie of meer terug"},
           "legend": "Elke balk: het deel van de {cases} gevallen van *it*, *its* of *itself* in LitBank waarvan de keten al "
                     "eerder genoemd werd, naar hoeveel zinnen terug de dichtstbijzijnde eerdere mention van die keten "
                     "staat; een volle balk is 1. In de kleur van de tekst: de telling van de course."},
}


def back_picture(lang: str) -> str:
    """For the cases of it: how many sentences back the nearest earlier mention of its chain stands."""
    w, nums = _BACK_WORDS[lang], numbers()
    rows, said = [], []
    for key, count in nums["sentences_back"].items():
        share = count / nums["cases"]
        rows.append(_bar(w["labels"][key], share, "course", lang))
        said.append(f'{w["labels"][key]}: {count}')
    return _figure(w["title"], rows, said, [_said(w["legend"], cases=nums["cases"])])


# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "What the rule picks for 'it' in LitBank",
           "case": "Case {n} of {all}: *{title}*, {author}, {year}",
           "people": "LitBank: *{it}* is in the same chain as *{near}*.",
           "kind": "The rule, with the kind, picks *{got}*: {verdict}.",
           "near": "The rule, without the kind, picks *{got}*: {verdict}.",
           "right": "right", "wrong": "wrong",
           "all": "Over all {all} cases of the slider, the rule gets {kind} right with the kind and {near} without.",
           "legend": "Bold and underlined: the pronoun. Gray frame: the earlier mentions in its chain, as people marked "
                     "them in LitBank. Frame in the color of the text: the mention the rule picks; purple dashed when "
                     "it is outside that chain."},
    "nl": {"title": "Wat de regel kiest voor 'it' in LitBank",
           "case": "Geval {n} van de {all}: *{title}*, {author}, {year}",
           "people": "LitBank: *{it}* hoort bij dezelfde keten als *{near}*.",
           "kind": "De regel, met de soort, kiest *{got}*: {verdict}.",
           "near": "De regel, zonder de soort, kiest *{got}*: {verdict}.",
           "right": "goed", "wrong": "fout",
           "all": "Over alle {all} gevallen van de slider heeft de regel er {kind} goed met de soort en {near} "
                  "zonder.",
           "legend": "Vet en onderstreept: het voornaamwoord. Grijs kader: de eerdere mentions in de keten van het "
                     "voornaamwoord, zoals mensen de keten in LitBank markeerden. Kader in de kleur van de tekst: de "
                     "mention die de regel kiest; paars gestreept als de keuze buiten de keten valt."},
}


def _author(name: str) -> str:
    """LitBank's "Austen, Jane" as "Jane Austen"; a part in parentheses goes."""
    name = re.sub(r"\s*\(.*?\)", "", name)
    last, _, first = name.partition(", ")
    return f"{first} {last}".strip() if first else name


def demo_picture(number, use_kind, lang: str) -> str:
    """One case of it from LitBank: the sentences around it, the earlier mentions of its chain and the rule's pick."""
    w, nums = _DEMO_WORDS[lang], numbers()
    try:
        n = min(max(int(number), 1), len(DEMO_CASES))
    except (TypeError, ValueError):
        n = DEMO_START
    case, first = DEMO_CASES[n - 1]
    w_i, i, before = case
    work = WORKS[w_i]
    m = work.mentions[i]
    mentions = case_mentions(case)
    got = pick(mentions, len(before)) if use_kind else nearest(mentions, len(before))
    chosen = work.mentions[before[got]]
    gold = [work.mentions[j] for j in before if work.mentions[j].chain == m.chain]
    fine = is_right(case, got)

    def covers(x, s, t):
        return x.sentence == s and x.first <= t <= x.last

    lines = []
    for s in range(first, m.sentence + 1):
        groups = []   # runs of tokens with the same marks
        for t, token in enumerate(work.sentences[s]):
            marks = (covers(m, s, t), any(covers(g, s, t) for g in gold), covers(chosen, s, t))
            if groups and groups[-1][0] == marks:
                groups[-1][1].append(token)
            else:
                groups.append((marks, [token]))
        out, prev = [], ""
        for marks, tokens in groups:
            text = ""
            for token in tokens:
                text += (" " if text and _space_before(prev, token) else "") + token
                prev = token
            space = " " if out and _space_before(out[-1][1], tokens[0]) else ""
            style = []
            if marks[0]:
                style.append("font-weight:600;text-decoration:underline;text-underline-offset:3px;color:var(--ea-ink)")
            if marks[1]:
                style.append("border:1px solid var(--ea-ink-3);border-radius:3px;padding:0 2px;color:var(--ea-ink-2)")
            if marks[2]:
                style.append("outline:2px solid var(--ea-ink);outline-offset:1px;border-radius:3px;color:var(--ea-ink)"
                             if fine else _OFF + ";outline-width:2px;color:var(--ea-ink)")
            body = html.escape(text)
            out.append((space + (f'<span style="{";".join(style)}">{body}</span>' if style else body), tokens[-1]))
        lines.append("".join(x for x, _ in out))
    text = " ".join(lines)
    rows = [f'<div style="{_UI};color:var(--ea-ink-2)">'
            f'{_said(w["case"], n=n, all=len(DEMO_CASES), title=work.title, author=_author(work.author), year=work.year)[0]}'
            f'</div>', f'<div style="{_TEXT};line-height:1.9;margin-top:0.4rem">{text}</div>']
    verdict = w["right"] if fine else w["wrong"]
    notes = [_said(w["people"], it=m.text, near=gold[-1].text),
             _said(w["kind" if use_kind else "near"], got=chosen.text, verdict=verdict),
             _said(w["all"], all=nums["demo"]["n"], kind=nums["demo"]["kind"], near=nums["demo"]["near"]),
             _said(w["legend"])]
    plain = " ".join(" ".join(work.sentences[s]) for s in range(first, m.sentence + 1))
    return _figure(w["title"], rows, [plain], notes)


# --- running the learner's function in a picture ---------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _quiet gives for a function that stopped with an error
_TOO_SLOW = object()  # what _timed gives when the learner's function takes too long; None is a value of theirs
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of a picture


def _quiet(fn, args):
    """The learner's function without what it prints; _STOPPED when it stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _quiet_many(fn, calls) -> list:
    """The learner's function on many calls, without what it prints; _STOPPED for a call that stops with an error."""
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for args in calls:
            try:
                out.append(fn(*args))
            except Timeout:
                raise
            except Exception:   # noqa: BLE001   the check above says what went wrong
                out.append(_STOPPED)
    return out


def _timed(make):
    """make() within PICTURE_SECONDS, or _TOO_SLOW when it takes longer."""
    try:
        return with_time_limit(make, PICTURE_SECONDS)
    except Timeout:
        return _TOO_SLOW


_SLOW = {"en": "Your function takes too long on all the mentions. Does a loop of yours go on longer than it needs to?",
         "nl": "Je functie duurt te lang op alle mentions. Gaat een van je loops langer door dan nodig is?"}


_CHAINS_WORDS = {
    "en": {"title": "Your chains on the opening of Pride and Prejudice",
           "found": "Your function finds {mine} chains in the {mentions} mentions; LitBank has {chains}.",
           "gray": "Gray: the five longest chains as LitBank marks them, each named by its first name or description, "
                   "with its number of mentions.",
           "blue": "Blue: the list your function gives for that chain, the first ten texts.",
           "dashed": "Purple dashed: a list that differs from the course's; for *{label}*, the course's list starts "
                     "with {want}.",
           "missing": "Your dict has no chain {chain}, the chain of *{label}*.",
           "odd": "Your function gives {got} instead of a dict.",
           "stop": "Your function stops with an error on the mentions of the work.",
           "result": "One chain holds *Mr. Bennet*, *you* and *My dear*: a name, a pronoun and a description of one "
                     "person."},
    "nl": {"title": "Jouw ketens op het begin van Pride and Prejudice",
           "found": "Je functie vindt {mine} ketens in de {mentions} mentions; LitBank heeft er {chains}.",
           "gray": "Grijs: de vijf langste ketens volgens LitBank, elk genoemd naar de eerste naam of "
                   "beschrijving, met het aantal mentions.",
           "blue": "Blauw: de lijst die je functie voor die keten geeft, de eerste tien teksten.",
           "dashed": "Paars gestreept: een lijst die afwijkt van de lijst van de course; voor *{label}* begint de "
                     "lijst van de course met {want}.",
           "missing": "Je dict heeft geen keten {chain}, de keten van *{label}*.",
           "odd": "Je functie geeft {got} in plaats van een dict.",
           "stop": "Je functie stopt met een fout op de mentions van het werk.",
           "result": "Eén keten bevat *Mr. Bennet*, *you* en *My dear*: een naam, een voornaamwoord en een "
                     "beschrijving van één persoon."},
}

_SHOWN_TEXTS = 10


def _texts(v) -> str:
    """The first texts of a chain, joined, with how many more there are."""
    if not isinstance(v, (list, tuple)):
        return _short(v)
    head = " · ".join(_short(x, 40) if not isinstance(x, str) else x for x in v[:_SHOWN_TEXTS])
    return head + (f" · … +{len(v) - _SHOWN_TEXTS}" if len(v) > _SHOWN_TEXTS else "")


def _same_list(mine, want) -> bool:
    """Whether the learner's value is the list the course has; a value that cannot be compared is not."""
    try:
        return isinstance(mine, (list, tuple)) and list(mine) == want
    except Exception:   # noqa: BLE001   a learner's value can be anything
        return False


def chains_picture(seen) -> str:
    """The learner's chains on the mentions of the opening of Pride and Prejudice: the five longest chains of LitBank,
    each with the list the learner's function gives for it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums = _CHAINS_WORDS[lang], numbers()
    work = pride()
    got = _timed(lambda: _quiet(fn, ([(m.text, m.kind) for m in work.mentions], [m.chain for m in work.mentions])))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    if got is _STOPPED:
        return _figure(w["title"], [], [], [_said(w["stop"])])
    if not isinstance(got, dict):
        return _figure(w["title"], [], [], [_said(w["odd"], got=_short(got))])
    rows, said, notes, differs = [], [], [], None
    for c, label, want in nums["pride"]["top"]:
        rows.append(f'<div style="{_HEAD_UI};color:var(--ea-ink-2);font-weight:500">'
                    f'{html.escape(label)} <span style="{_SMALL}">· {len(want)}</span></div>')
        if c not in got:
            notes.append(_said(w["missing"], chain=c, label=label))
            continue
        mine = got[c]
        off = not _same_list(mine, want)
        if off and differs is None:
            differs = (label, want)
        style = f"{_CODE};color:var(--ea-pred-text);line-height:1.6" + (";" + _OFF if off else "")
        rows.append(f'<div><span style="{style}">{html.escape(_texts(mine))}</span></div>')
        said.append(f"{label}: {_texts(mine)}")
    extra = [_said(w["found"], mine=len(got), mentions=nums["pride"]["mentions"], chains=nums["pride"]["chains"]),
             _said(w["gray"]), _said(w["blue"])]
    if differs:
        extra.append(_said(w["dashed"], label=differs[0], want=_texts(differs[1])))
    elif not notes:
        extra.append(_said(w["result"]))
    return _figure(w["title"], rows, said, extra + notes)


_BACKS_WORDS = {
    "en": {"title": "Your antecedents on all 100 works of LitBank",
           "share": "Mentions with an earlier mention in their chain",
           "cats": {"PRON": "Pronouns", "PROP": "Names", "NOM": "Descriptions"},
           "back": "Pronouns: how many mentions back that earlier mention is",
           "blue": "Blue: what your function gives on the chains of the {mentions} mentions; a full bar is 1. Above, "
                   "the share of each category that gets a place; below, for the {pron} pronouns that the course finds "
                   "one for, how far back your place is.",
           "dashed": "Purple dashed: a share that differs from the course's; for pronouns, the course gets {want} "
                     "with an earlier mention.",
           "odd": "For {k} of the 100 works, your function gives {got} instead of a list with one entry for each "
                  "mention.",
           "bad": "{k} entries of your lists are neither `None` nor the place of an earlier mention, such as {got}.",
           "stop": "For {k} of the 100 works, your function stops with an error.",
           "result": "Half of the pronouns find their antecedent within {median} mentions, {w5} within 5 and {w9} "
                     "within 9."},
    "nl": {"title": "Jouw antecedenten op alle 100 werken van LitBank",
           "share": "Mentions met een eerdere mention in de keten",
           "cats": {"PRON": "Voornaamwoorden", "PROP": "Namen", "NOM": "Beschrijvingen"},
           "back": "Voornaamwoorden: hoeveel mentions terug die eerdere mention staat",
           "blue": "Blauw: wat je functie geeft op de ketens van de {mentions} mentions; een volle balk is 1. Boven het "
                   "deel van de voornaamwoorden, de namen en de beschrijvingen dat een plaats krijgt; onder, voor de "
                   "{pron} voornaamwoorden waarvoor de course er een vindt, hoe ver terug jouw plaats staat.",
           "dashed": "Paars gestreept: een deel dat afwijkt van het deel van de course; bij voornaamwoorden heeft "
                     "volgens de course {want} een eerdere mention.",
           "odd": "Voor {k} van de 100 werken geeft je functie {got} in plaats van een lijst met één element per "
                  "mention.",
           "bad": "{k} elementen van je lijsten zijn geen `None` en ook niet de plaats van een eerdere mention, zoals "
                  "{got}.",
           "stop": "Voor {k} van de 100 werken stopt je functie met een fout.",
           "result": "De helft van de voornaamwoorden vindt de antecedent binnen {median} mentions, {w5} binnen 5 en "
                     "{w9} binnen 9."},
}


def antecedents_picture(seen) -> str:
    """The learner's antecedents on the chains of every work: the share of each category with an earlier mention, and
    for the pronouns how many mentions back the place is."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums = _BACKS_WORDS[lang], numbers()
    got = _timed(lambda: _quiet_many(fn, [([m.chain for m in work.mentions],) for work in WORKS]))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    stopped = sum(g is _STOPPED for g in got)
    odd = [g for g in got if g is not _STOPPED and not (isinstance(g, (list, tuple)))]
    odd += [g for g, work in zip(got, WORKS) if isinstance(g, (list, tuple)) and len(g) != len(work.mentions)]
    notes = []
    if stopped:
        notes.append(_said(w["stop"], k=stopped))
    if odd:
        notes.append(_said(w["odd"], k=len(odd), got=_short(odd[0])))
    fine = [(work, list(g)) for g, work in zip(got, WORKS)
            if isinstance(g, (list, tuple)) and len(g) == len(work.mentions)]
    if not fine:
        return _figure(w["title"], [], [], notes)
    has, bad = {c: 0 for c in CATEGORIES}, []
    d = []
    for work, out in fine:
        for k, (m, x) in enumerate(zip(work.mentions, out)):
            if x is None:
                continue
            if not (isinstance(x, int) and not isinstance(x, bool) and 0 <= x < k):
                bad.append(x)
                continue
            has[m.category] += 1
        d += distances(work, out)
    if bad:
        notes.append(_said(w["bad"], k=_n(len(bad), lang), got=_short(bad[0], 20)))
    rows, said, differs = [_head(w["share"])], [], False
    for c in CATEGORIES:
        share = has[c] / nums["categories"][c]
        off = not _same_number(share, nums["with_antecedent"][c])
        differs = differs or off
        rows.append(_bar(w["cats"][c], share, "mine", lang, off=off))
        said.append(f'{w["cats"][c]}: {_d(share, lang)}')
    rows.append(_head(w["back"]))
    total = nums["pron_anaphors"]
    for b in _BUCKETS:
        share = sum(bucket(x) == b for x in d) / total
        off = not _same_number(share, nums["buckets"][b])
        differs = differs or off
        rows.append(_bar(b, share, "mine", lang, off=off))
        said.append(f"{b}: {_d(share, lang)}")
    extra = [_said(w["blue"], mentions=_n(nums["mentions"], lang), pron=_n(total, lang))]
    if differs:
        extra.append(_said(w["dashed"], want=_d(nums["with_antecedent"]["PRON"], lang)))
    elif len(fine) == len(WORKS) and not bad:
        median = nums["median"]
        extra.append(_said(w["result"], median=median if median != int(median) else int(median),
                           w5=_d(nums["within5"], lang), w9=_d(nums["within9"], lang)))
    return _figure(w["title"], rows, said, extra + notes)


_PICK_WORDS = {
    "en": {"title": "What your function picks in the 318 cases of 'it' in LitBank",
           "mine": "Your function", "near": "Nearest earlier mention",
           "blue": "Blue: the share of the {cases} cases where the mention your function picks is in the chain of the "
                   "pronoun; a full bar is 1. In the color of the text: the nearest earlier mention of any kind, for "
                   "comparison.",
           "dashed": "Purple dashed: a share that differs from the course's; the course gets {want}.",
           "none": "For {k} of the cases, your function gives `None`.",
           "odd": "For {k} of the cases, your function gives something other than a place before the pronoun, such as "
                  "{got}.",
           "stop": "For {k} of the cases, your function stops with an error.",
           "result": "{right} of the {cases} right, with nothing but the order and two kinds."},
    "nl": {"title": "Wat je functie kiest in de 318 gevallen van 'it' in LitBank",
           "mine": "Je functie", "near": "Dichtstbijzijnde eerdere mention",
           "blue": "Blauw: het deel van de {cases} gevallen waarin de mention die je functie kiest in de keten van het "
                   "voornaamwoord zit; een volle balk is 1. In de kleur van de tekst: de dichtstbijzijnde eerdere "
                   "mention van welke soort ook, om mee te vergelijken.",
           "dashed": "Paars gestreept: een deel dat afwijkt van het deel van de course; de course krijgt {want}.",
           "none": "Voor {k} van de gevallen geeft je functie `None`.",
           "odd": "Voor {k} van de gevallen geeft je functie iets anders dan een plaats vóór het voornaamwoord, zoals "
                  "{got}.",
           "stop": "Voor {k} van de gevallen stopt je functie met een fout.",
           "result": "{right} van de {cases} goed, met alleen de volgorde en twee soorten."},
}


def pick_picture(seen) -> str:
    """The learner's pick on every case of it in LitBank: the share whose pick is in the chain of the pronoun, beside
    the nearest earlier mention."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w, nums = _PICK_WORDS[lang], numbers()
    got = _timed(lambda: _quiet_many(fn, [(case_mentions(c), len(c[2])) for c in IT_CASES]))
    if got is _TOO_SLOW:
        return _figure(w["title"], [], [], [_said(_SLOW[lang])])
    stopped = sum(g is _STOPPED for g in got)
    nones = sum(g is None for g in got)
    odd = [g for g, c in zip(got, IT_CASES) if g is not _STOPPED and g is not None
           and not (isinstance(g, int) and not isinstance(g, bool) and 0 <= g < len(c[2]))]
    notes = []
    if stopped:
        notes.append(_said(w["stop"], k=stopped))
    if nones:
        notes.append(_said(w["none"], k=nones))
    if odd:
        notes.append(_said(w["odd"], k=len(odd), got=_short(odd[0], 20)))
    if stopped == len(got):
        return _figure(w["title"], [], [], notes)
    right = sum(is_right(c, g) for g, c in zip(got, IT_CASES))
    share = right / len(IT_CASES)
    off = not _same_number(share, nums["kind_share"])
    rows = [_bar(w["mine"], share, "mine", lang, off=off), _bar(w["near"], nums["near_share"], "course", lang)]
    said = [f'{_bare(w["mine"])}: {_d(share, lang)}', f'{_bare(w["near"])}: {_d(nums["near_share"], lang)}']
    extra = [_said(w["blue"], cases=nums["cases"])]
    if off:
        extra.append(_said(w["dashed"], want=_d(nums["kind_share"], lang)))
    else:
        extra.append(_said(w["result"], right=right, cases=nums["cases"]))
    return _figure(w["title"], rows, said, extra + notes)


# --- the checks ------------------------------------------------------------------------------------------------------

REQUIRED = ["chains", "antecedents", "pick", "trophy"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name, and handed over as a new list,
    as the exercises say."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return list(self.value)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _dots_left(got, text) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _not_dict(got, text) -> bool:
    return got is not None and not isinstance(got, dict)


def _not_list(got, text) -> bool:
    return got is not None and not isinstance(got, (list, tuple))


def _not_place(got, text) -> bool:
    return got is not None and not (isinstance(got, int) and not isinstance(got, bool))


def _rounded(v, depth: int = 0) -> str:
    """A value the way Python shows it, with its words in double quotes, also inside a dict, a tuple or a list; at most
    30 items and three levels deep."""
    if depth > 3:
        return "…"
    if isinstance(v, float):
        text = f"{round(v, 4) + 0.0:.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if type(v) is dict:
        items = list(v.items())
        inner = ", ".join(f"{_rounded(k, depth + 1)}: {_rounded(x, depth + 1)}" for k, x in items[:30])
        return "{" + inner + (", …" if len(items) > 30 else "") + "}"
    if type(v) is tuple:
        inner = ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "")
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if type(v) is list:
        return "[" + ", ".join(_rounded(x, depth + 1) for x in v[:30]) + (", …" if len(v) > 30 else "") + "]"
    if type(v) is set:
        return "{" + ", ".join(sorted(_rounded(x, depth + 1) for x in list(v)[:30])) + "}" if v else "set()"
    if type(v) is str and v.isprintable() and '"' not in v and "\\" not in v:
        return f'"{v}"'   # double quotes, as in the exercise texts
    return _repr(v)


def show_value(v) -> str:
    """How {got} and {want} show a value: as code, cut after 90 characters."""
    text = _rounded(v)
    if len(text) > 90:
        text = text[:89] + "…"
    if isinstance(v, (int, float)) and not isinstance(v, bool) or "`" in text:
        return text
    return f"`{text}`"


_M = Named("C.MUSEUM", _MUSEUM)
_M_IDS = Named("C.MUSEUM_IDS", _MUSEUM_IDS)

# exercise 1: chains. The visible case: the passage of the museum; then no mentions, every mention its own chain,
# chains numbered as in LitBank with one text twice in a chain, the same text "it" in two chains, and one chain.
CH_CASES = [
    (_M, _M_IDS),
    ([], []),
    ([("Paris", "thing"), ("the river", "thing"), ("Sam", "person")], ["paris", "river", "sam"]),
    ([("Sam", "person"), ("the farm", "thing"), ("they", "person"), ("it", "thing"), ("Sam", "person")], [0, 1, 0, 1, 0]),
    ([("The truck", "thing"), ("a trailer", "thing"), ("It", "thing"), ("it", "thing"), ("the hill", "thing")],
     ["truck", "trailer", "truck", "trailer", "hill"]),
    ([("The ship", "thing"), ("It", "thing"), ("its", "thing")], [3, 3, 3]),
]
_CH_HEAD = "def chains(mentions, ids):\n"
_CH_TOP = (_CH_HEAD
           + "    found = {}\n"
           + "    for k in range(len(mentions)):\n"
           + "        text = mentions[k][0]\n"
           + "        c = ids[k]\n")
_CH_START = _CH_TOP + "        ...\n    return found\n"


def _ch(body: str, end: str = "    return found\n") -> str:
    return _CH_TOP + body + end


_CH = _ch("        if c in found:\n            found[c].append(text)\n        else:\n            found[c] = [text]\n")
_CH_COUNTS = _ch("        if c in found:\n            found[c] = found[c] + 1\n        else:\n            found[c] = 1\n")
_CH_PLACES = _ch("        if c in found:\n            found[c].append(k)\n        else:\n            found[c] = [k]\n")
_CH_PAIRS = _ch("        if c in found:\n            found[c].append(mentions[k])\n        else:\n"
                "            found[c] = [mentions[k]]\n")
_CH_TEXT_KEYS = _ch("        if text in found:\n            found[text].append(c)\n        else:\n"
                    "            found[text] = [c]\n")
_CH_SET = _ch("        if c in found:\n            found[c].add(text)\n        else:\n            found[c] = {text}\n")
_CH_LAST = _ch("        found[c] = [text]\n")
_CH_EMPTY = _ch("        if c in found:\n            found[c].append(text)\n        else:\n            found[c] = []\n")
_CH_VALUES = _ch("        if c in found:\n            found[c].append(text)\n        else:\n            found[c] = [text]\n",
                 "    return list(found.values())\n")

# exercise 2: antecedents. The visible case: the chains of the museum; then no mentions, every mention its own chain,
# one chain three times, chains numbered as in LitBank, two chains that cross, and a chain whose first place is 0.
AN_CASES = [
    (_M_IDS,),
    ([],),
    (["a", "b", "c"],),
    (["a", "a", "a"],),
    ([0, 1, 0, 1, 2, 0],),
    (["x", "y", "y", "x"],),
    (["m", "m"],),
]
_AN_HEAD = "def antecedents(ids):\n"
_AN_TOP = (_AN_HEAD
           + "    last = {}\n"
           + "    out = []\n"
           + "    for k in range(len(ids)):\n")
_AN_END = "        last[ids[k]] = k\n    return out\n"
_AN_START = _AN_TOP + "        ...\n" + _AN_END


def _an(body: str, end: str = _AN_END) -> str:
    return _AN_TOP + body + end


_AN = _an("        if ids[k] in last:\n            out.append(last[ids[k]])\n        else:\n            out.append(None)\n")
_AN_FIRST = _an("        if ids[k] in last:\n            out.append(last[ids[k]])\n        else:\n            out.append(None)\n"
                "            last[ids[k]] = k\n", "    return out\n")
_AN_IDS = _an("        if ids[k] in last:\n            out.append(ids[k])\n        else:\n            out.append(None)\n")
_AN_ITSELF = _AN_TOP + "        last[ids[k]] = k\n" + ("        if ids[k] in last:\n            out.append(last[ids[k]])\n"
                                                         "        else:\n            out.append(None)\n    return out\n")
_AN_SKIPS = _an("        if ids[k] in last:\n            out.append(last[ids[k]])\n")
_AN_PREVIOUS = _an("        if ids[k] in last:\n            out.append(k - 1)\n        else:\n            out.append(None)\n")
_AN_BOOLS = _an("        if ids[k] in last:\n            out.append(True)\n        else:\n            out.append(False)\n")
_AN_MINUS = _an("        if ids[k] in last:\n            out.append(last[ids[k]])\n        else:\n            out.append(-1)\n")

# exercise 3: pick. The visible case: the second it of the museum, whose nearest earlier thing is the hall; then the
# first mention, the first it, visitors, the last it, a person with only a thing before it, a person whose nearest
# earlier person is four places back, and a thing with one of its kind after it.
PK_CASES = [
    (_M, 5),
    (_M, 0),
    (_M, 2),
    (_M, 6),
    (_M, 7),
    ([("The ship", "thing"), ("Sam", "person")], 1),
    ([("Sam", "person"), ("the door", "thing"), ("the key", "thing"), ("the lock", "thing"), ("they", "person")], 4),
    ([("the box", "thing"), ("it", "thing"), ("the lid", "thing")], 1),
]
_PK_HEAD = "def pick(mentions, i):\n"
_PK_TOP = (_PK_HEAD
           + "    kind = mentions[i][1]\n"
           + "    best = None\n")
_PK_START = _PK_TOP + "    ...\n    return best\n"
_PK = _PK_TOP + "    for j in range(i):\n        if mentions[j][1] == kind:\n            best = j\n    return best\n"
_PK_EARLIEST = _PK.replace("        if mentions[j][1] == kind:\n", "        if mentions[j][1] == kind and best is None:\n")
_PK_ANY = _PK_TOP + "    for j in range(i):\n        best = j\n    return best\n"
_PK_ITSELF = _PK.replace("range(i)", "range(i + 1)")
_PK_TEXT = _PK.replace("            best = j\n", "            best = mentions[j][0]\n")
_PK_ZERO = _PK.replace("    best = None\n", "    best = 0\n")
_PK_ALL = _PK.replace("range(i)", "range(len(mentions))")
_PK_MENTION = _PK.replace("            best = j\n", "            best = mentions[j]\n")
_PK_THING = _PK.replace("        if mentions[j][1] == kind:\n", '        if mentions[j][1] == "thing":\n')
_PK_SHORT = _PK.replace("range(i)", "range(i - 1)")

EXERCISES = {
    "chains": FunctionBox(
        name="chains",
        picture=chains_picture,
        function="chains",
        start=_CH_START,
        solution=_CH,
        visible=CH_CASES[:1],
        hidden=CH_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("counts", wrong=_CH_COUNTS,
                    en="`{call}` gives {got}, and it should give {want}: the number of mentions of each chain. What should each chain hold?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het aantal mentions van elke keten. Wat moet elke keten bevatten?"),
            Mistake("places", wrong=_CH_PLACES,
                    en="`{call}` gives {got}, and it should give {want}: the places of the mentions in the list. What should a chain hold of each mention?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de plaatsen van de mentions in de lijst. Wat moet een keten van elke mention bevatten?"),
            Mistake("pairs", wrong=_CH_PAIRS,
                    en="`{call}` gives {got}, and it should give {want}: each mention as a whole pair, with its kind. Which part of a mention should the chain hold?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke mention als heel paar, met de soort erbij. Welk deel van een mention moet de keten bevatten?"),
            Mistake("text-keys", wrong=_CH_TEXT_KEYS,
                    en="`{call}` gives {got}, and it should give {want}: the keys of your dict are the texts, and the values the chains. Which should be the key?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de keys van je dict zijn de teksten, en de waarden de ketens. Wat moet de key zijn?"),
            Mistake("set", wrong=_CH_SET,
                    en="`{call}` gives {got}, and it should give {want}: a set keeps each text once and forgets the order. What happens to *it*, which a chain can hold more than once?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een set bewaart elke tekst één keer en vergeet de volgorde. Wat gebeurt er met *it*, dat meer dan één keer in een keten kan staan?"),
            Mistake("last-only", wrong=_CH_LAST,
                    en="`{call}` gives {got}, and it should give {want}: each chain keeps only its last mention, since every mention starts the list over. What should happen to a chain that already has a list?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke keten houdt alleen de laatste mention over, omdat elke mention de lijst opnieuw laat beginnen. Wat moet er gebeuren met een keten die al een lijst heeft?"),
            Mistake("empty-start", wrong=_CH_EMPTY,
                    en="`{call}` gives {got}, and it should give {want}: each chain misses its first mention, since a new chain starts with an empty list. What should a new list hold right away?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke keten mist de eerste mention, omdat een nieuwe keten met een lege lijst begint. Wat moet een nieuwe lijst meteen bevatten?"),
            Mistake("values", wrong=_CH_VALUES,
                    en="`{call}` gives {got}, and it should give {want}: the lists without the chains they belong to. What should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de lijsten zonder de ketens waar de lijsten bij horen. Wat moet je functie teruggeven?"),
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give {want}: a dict. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een dict. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give the chain `c` a list when it has none yet, and add `text` to the list of `c`.",
                      "nl": "Vervang de `...`: geef de keten `c` een lijst als de keten er nog geen heeft, en voeg `text` toe aan de lijst van `c`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which key does each text go under, and what happens when that key is already there?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Onder welke key komt elke tekst, en wat gebeurt er als die key er al is?"},
            "almost": {"en": "Right for the museum. But `{call}` gives {got}, and it should give {want}. Does your function work without mentions, with numbers as chains, and for a text that comes twice?",
                       "nl": "Goed voor het museum. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie zonder mentions, met getallen als ketens, en voor een tekst die twee keer voorkomt?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return` of the dict?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return` van de dict?"},
            "correct": {"en": "Your `chains` is right in all {n} cases. Look at the picture: your function on the opening of *Pride and Prejudice*.",
                        "nl": "Je `chains` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op het begin van *Pride and Prejudice*."},
        },
    ),
    "antecedents": FunctionBox(
        name="antecedents",
        picture=antecedents_picture,
        function="antecedents",
        start=_AN_START,
        solution=_AN,
        visible=AN_CASES[:1],
        hidden=AN_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("first", wrong=_AN_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: each mention gets the first mention of its chain. Which earlier mention of the chain is the nearest?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke mention krijgt de eerste mention van de keten. Welke eerdere mention van de keten is de dichtstbijzijnde?"),
            Mistake("itself", wrong=_AN_ITSELF,
                    en="`{call}` gives {got}, and it should give {want}: each mention gets its own place, since `last` already holds it when your function looks. When should `last` learn the place of mention `k`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke mention krijgt de eigen plaats, omdat `last` de plaats van de mention zelf al bevat als je functie kijkt. Wanneer moet `last` de plaats van mention `k` leren?"),
            Mistake("ids", wrong=_AN_IDS,
                    en="`{call}` gives {got}, and it should give {want}: the chain of the mention, not a place. What should the list hold for a mention whose chain came before?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de keten van de mention, geen plaats. Wat moet de lijst bevatten voor een mention waarvan de keten al eerder kwam?"),
            Mistake("previous", wrong=_AN_PREVIOUS,
                    en="`{call}` gives {got}, and it should give {want}: the place right before the mention, whatever its chain. Which place does `last` hold?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de plaats vlak vóór de mention, van welke keten ook. Welke plaats bevat `last`?"),
            Mistake("skips", wrong=_AN_SKIPS,
                    en="`{call}` gives {got}, and it should give {want}: your list skips the mentions whose chain is new, so it is shorter than `ids`. What should such a mention get?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst slaat de mentions met een nieuwe keten over, dus de lijst is korter dan `ids`. Wat moet zo'n mention krijgen?"),
            Mistake("booleans", wrong=_AN_BOOLS,
                    en="`{call}` gives {got}, and it should give {want}: whether each mention has an earlier one in its chain. Which place should the list hold for such a mention?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: of elke mention een eerdere in de keten heeft. Welke plaats moet de lijst voor zo'n mention bevatten?"),
            Mistake("minus-one", wrong=_AN_MINUS,
                    en="`{call}` gives {got}, and it should give {want}: `-1` for a mention whose chain is new. In Python, `-1` is also a place in a list. What does the exercise ask for such a mention?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `-1` voor een mention met een nieuwe keten. In Python is `-1` ook een plaats in een lijst. Waar vraagt de opdracht om voor zo'n mention?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lijst. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add to `out` the place that `last` holds for the chain of mention `k`, or `None` when its chain is new.",
                      "nl": "Vervang de `...`: voeg aan `out` de plaats toe die `last` voor de keten van mention `k` bevat, of `None` als de keten nieuw is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What does `out` get for a mention whose chain came before, and what for one whose chain is new?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Wat krijgt `out` voor een mention waarvan de keten al eerder kwam, en wat voor een mention met een nieuwe keten?"},
            "almost": {"en": "Right for the museum. But `{call}` gives {got}, and it should give {want}. Does your function work without mentions, for a chain that comes three times, and for a chain whose earlier mention is at place 0?",
                       "nl": "Goed voor het museum. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie zonder mentions, voor een keten die drie keer voorkomt, en voor een keten waarvan de eerdere mention op plaats 0 staat?"},
            "unreached": {"en": "With `{call}`, your function gives `None`. Does it end with a `return` of the list?",
                          "nl": "Met `{call}` geeft je functie `None`. Eindigt je functie met een `return` van de lijst?"},
            "correct": {"en": "Your `antecedents` is right in all {n} cases. Look at the picture: your function on the chains of all 100 works.",
                        "nl": "Je `antecedents` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op de ketens van alle 100 werken."},
        },
    ),
    "pick": FunctionBox(
        name="pick",
        picture=pick_picture,
        function="pick",
        start=_PK_START,
        solution=_PK,
        visible=PK_CASES[:1],
        hidden=PK_CASES[1:],
        show=show_value,
        first=[_DOTS],
        mistakes=[
            Mistake("earliest", wrong=_PK_EARLIEST,
                    en="`{call}` gives {got}, and it should give {want}: the first mention of the same kind, from the start of the text. Which one of them is the nearest?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de eerste mention van dezelfde soort, vanaf het begin van de tekst. Welke daarvan is de dichtstbijzijnde?"),
            Mistake("any-kind", wrong=_PK_ANY,
                    en="`{call}` gives {got}, and it should give {want}: the mention right before, whatever its kind. Which mentions may the rule pick?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de mention vlak ervoor, van welke soort ook. Welke mentions mag de regel kiezen?"),
            Mistake("itself", wrong=_PK_ITSELF,
                    en="`{call}` gives {got}, and it should give {want}: the mention itself, since your loop also looks at place `i`. Which places come before `i`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de mention zelf, omdat je loop ook naar plaats `i` kijkt. Welke plaatsen komen vóór `i`?"),
            Mistake("all", wrong=_PK_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your loop goes over the whole list, also past `i`. Which mentions can a pronoun refer back to?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop gaat over de hele lijst, ook voorbij `i`. Naar welke mentions kan een voornaamwoord terugverwijzen?"),
            Mistake("thing-only", wrong=_PK_THING,
                    en="`{call}` gives {got}, and it should give {want}: your function looks for a thing, also when mention `i` is a person. Which kind should the mention it picks have?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie zoekt een ding, ook als mention `i` een persoon is. Welke soort moet de gekozen mention hebben?"),
            Mistake("short", wrong=_PK_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: your loop stops one place early, so it never looks at the mention right before `i`. Which places come before `i`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je loop stopt één plaats te vroeg, dus de mention vlak vóór `i` komt nooit aan bod. Welke plaatsen komen vóór `i`?"),
            Mistake("text", wrong=_PK_TEXT,
                    en="`{call}` gives {got}, and it should give {want}: the text of the mention. What should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de tekst van de mention. Wat moet je functie teruggeven?"),
            Mistake("mention", wrong=_PK_MENTION,
                    en="`{call}` gives {got}, and it should give {want}: the whole mention. What should your function give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de hele mention. Wat moet je functie teruggeven?"),
            Mistake("zero", wrong=_PK_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: `best` starts at 0, so without a mention of the same kind before `i` your function gives the first place. What should it give then?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `best` begint bij 0, dus zonder mention van dezelfde soort vóór `i` geeft je functie de eerste plaats. Wat moet je functie dan geven?"),
            Mistake("not-place", matches=_not_place,
                    en="`{call}` gives {got}, and it should give {want}: a place in the list, or `None`. What does your `return` line give back?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een plaats in de lijst, of `None`. Wat geeft je `return`-regel terug?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the places before `i`, and remember in `best` each place whose mention has the kind `kind`.",
                      "nl": "Vervang de `...`: ga de plaatsen vóór `i` langs, en onthoud in `best` elke plaats waarvan de mention de soort `kind` heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which places does your loop go over, and which of them does it remember?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welke plaatsen gaat je loop langs, en welke daarvan onthoudt je loop?"},
            "almost": {"en": "Right for the second *it*. But `{call}` gives {got}, and it should give {want}. Does your function work for the first mention, for a person, for a match right before `i`, without an earlier mention of the same kind, and with one of the same kind after `i`?",
                       "nl": "Goed voor de tweede *it*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor de eerste mention, voor een persoon, voor een passende mention vlak vóór `i`, zonder eerdere mention van dezelfde soort, en met een mention van dezelfde soort na `i`?"},
            "unreached": {"en": "With `{call}`, your function gives `None`, and there is an earlier mention of the same kind. Which part of a mention does your `if` compare with `kind`?",
                          "nl": "Met `{call}` geeft je functie `None`, en er is een eerdere mention van dezelfde soort. Welk deel van een mention vergelijkt je `if` met `kind`?"},
            "correct": {"en": "Your `pick` is right in all {n} cases. Look at the picture: your function on the 318 cases of *it* in LitBank.",
                        "nl": "Je `pick` klopt in alle {n} gevallen. Kijk naar de afbeelding: jouw functie op de 318 gevallen van *it* in LitBank."},
        },
    ),
}

CASES = [
    ("chains", _CH_START, "empty", None),
    ("chains", _CH, "correct", None),
    ("chains", "def chains(mentions, ids):\n    found = {}\n    for (text, kind), c in zip(mentions, ids):\n        found.setdefault(c, []).append(text)\n    return found\n", "correct", None),
    ("chains", "def chains(mentions, ids):\n    found = {}\n    for k in range(len(ids)):\n        found[ids[k]] = found.get(ids[k], []) + [mentions[k][0]]\n    return found\n", "correct", None),
    ("chains", _CH_COUNTS, "wrong", "counts"),
    ("chains", _CH_PLACES, "wrong", "places"),
    ("chains", _CH_PAIRS, "wrong", "pairs"),
    ("chains", _CH_TEXT_KEYS, "wrong", "text-keys"),
    ("chains", _CH_SET, "wrong", "set"),
    ("chains", _CH_LAST, "wrong", "last-only"),
    ("chains", _CH_VALUES, "wrong", "values"),
    ("chains", _CH_EMPTY, "wrong", "empty-start"),
    ("chains", "def chains(mentions, ids):\n    return 5\n", "wrong", "not-dict"),
    ("chains", _CH.replace("    return found\n", "    print(found)\n"), "wrong", "prints"),
    ("chains", "def chains(mentions, ids):\n    return {\"museum\": [\"The museum\"], \"painting\": [\"a painting\", \"It\", \"it\", \"it\"], \"hall\": [\"the hall\"], \"guard\": [\"A guard\"], \"visitors\": [\"visitors\"]}\n", "almost", None),
    ("chains", _CH.replace("        c = ids[k]\n", "        c = ids[k]\n        ...\n"), "wrong", "dots-left"),
    ("antecedents", _AN_START, "empty", None),
    ("antecedents", _AN, "correct", None),
    ("antecedents", "def antecedents(ids):\n    out = []\n    for k in range(len(ids)):\n        found = None\n        for j in range(k):\n            if ids[j] == ids[k]:\n                found = j\n        out.append(found)\n    return out\n", "correct", None),
    ("antecedents", _AN_FIRST, "wrong", "first"),
    ("antecedents", _AN_ITSELF, "wrong", "itself"),
    ("antecedents", _AN_IDS, "wrong", "ids"),
    ("antecedents", _AN_PREVIOUS, "wrong", "previous"),
    ("antecedents", _AN_SKIPS, "wrong", "skips"),
    ("antecedents", _AN_BOOLS, "wrong", "booleans"),
    ("antecedents", _AN_MINUS, "wrong", "minus-one"),
    ("antecedents", "def antecedents(ids):\n    return len(ids)\n", "wrong", "not-list"),
    ("antecedents", _AN.replace("    return out\n", "    print(out)\n"), "wrong", "prints"),
    ("antecedents", "def antecedents(ids):\n    return [None, None, 1, None, None, 2, None, 5]\n", "almost", None),
    ("antecedents", _AN.replace("    out = []\n", "    out = []\n    ...\n"), "wrong", "dots-left"),
    ("pick", _PK_START, "empty", None),
    ("pick", _PK, "correct", None),
    ("pick", "def pick(mentions, i):\n    places = [j for j in range(i) if mentions[j][1] == mentions[i][1]]\n    return places[-1] if places else None\n", "correct", None),
    ("pick", _PK_EARLIEST, "wrong", "earliest"),
    ("pick", _PK_ANY, "wrong", "any-kind"),
    ("pick", _PK_ITSELF, "wrong", "itself"),
    ("pick", _PK_ALL, "wrong", "all"),
    ("pick", _PK_TEXT, "wrong", "text"),
    ("pick", _PK_MENTION, "wrong", "mention"),
    ("pick", _PK_ZERO, "wrong", "zero"),
    ("pick", _PK_THING, "wrong", "thing-only"),
    ("pick", _PK_SHORT, "wrong", "short"),
    ("pick", _PK.replace("        if mentions[j][1] == kind:\n", "        if mentions[j][0] == mentions[i][0]:\n"), "wrong", "unreached"),
    ("pick", "def pick(mentions, i):\n    return \"the hall\"\n", "wrong", "not-place"),
    ("pick", _PK.replace("        if mentions[j][1] == kind:\n", "        if mentions[j] == kind:\n"), "wrong", "unreached"),
    ("pick", _PK.replace("    return best\n", "    print(best)\n"), "wrong", "prints"),
    ("pick", "def pick(mentions, i):\n    return 3\n", "almost", None),
    ("pick", _PK.replace("    best = None\n", "    best = None\n    ...\n"), "wrong", "dots-left"),
]
