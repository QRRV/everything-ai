# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-word-relations: a sort of pairs of words by how they relate (senses, synonymy, hyponymy,
meronymy), the number of senses of every word of WordNet (count_senses), and the ladder of first hypernyms from a sense
up to the top (chain).

The senses come from Open English WordNet 2025, which the course ships (data/wordnet-en.tsv.gz, read by
everything_ai.wordnet), with the Dutch words of Open Dutch WordNet beside them. How often a word is used comes from the
order of the 50,000 words of GloVe (everything_ai.wordvectors), from the most to the least frequent. The small lists
and dicts of the cases are the course's own; every number in the text and the pictures is computed by the course
(tests/test_nlp_word_relations.py computes them again)."""
import contextlib
import html
import io

from everything_ai import wordnet, wordvectors, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

_KEY = "nlp-word-relations"

# --- the course's own functions ---------------------------------------------------------------------------------------


def count_senses(pairs) -> dict:
    """A dict from each word of the (word, name) pairs to its number of pairs: its number of senses."""
    counts = {}
    for word, name in pairs:
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1
    return counts


def chain(name, parent) -> list:
    """The list from `name` up to the top: each next item is the parent of the one before, until one has none."""
    out = [name]
    while out[-1] in parent:
        out.append(parent[out[-1]])
    return out


# --- the data -----------------------------------------------------------------------------------------------------------


def _pairs() -> tuple:
    return kept(f"{_KEY}:pairs", lambda: tuple(wordnet.sense_pairs()))


def pairs_for(words) -> list:
    """A new list of (word, name) pairs: one for every sense of every word in `words`, in the order of the list."""
    return [(word, name) for word in words for name in wordnet.senses(word)]


def sense_counts() -> dict:
    """The number of senses of every word of WordNet, as the course counts it: 127,846 words."""
    return kept(f"{_KEY}:counts", lambda: count_senses(_pairs()))


def parents() -> dict:
    """A new dict from every sense that has a hypernym to its first hypernym: 85,586 senses."""
    return wordnet.parents()


PARENT = parents()


def _glove_words() -> tuple:
    return kept(f"{_KEY}:glove", lambda: tuple(wordvectors.glove()[0]))


BANDS = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 50000)]   # places in GloVe's list, from the most frequent


def band_means(counts: dict) -> list:
    """For each band of BANDS and then for the words that are not among GloVe's 50,000: (how many words of `counts`
    fall in it, their mean number of senses)."""
    words = _glove_words()
    out = []
    for a, b in BANDS:
        found = [counts[w] for w in words[a:b] if w in counts]
        out.append((len(found), sum(found) / len(found) if found else 0.0))
    glove = set(words)
    rest = [v for w, v in counts.items() if w not in glove]
    out.append((len(rest), sum(rest) / len(rest) if rest else 0.0))
    return out


def top_words(counts: dict, k: int = 10, ties: int = 5) -> list:
    """The k words with the most senses, most first, and up to `ties` more words with as many senses as the k-th;
    words with as many senses in alphabetical order."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    tied = [kv for kv in ranked[k:k + ties + 1] if ranked[k - 1][1] == kv[1]] if len(ranked) > k else []
    return ranked[:k] + (tied if len(tied) <= ties else [])


def meeting(a: list, b: list):
    """The first item of `a` that is also in `b`: where two ladders meet."""
    return next((x for x in a if x in b), None)


# --- the look of the pictures -------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem")
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_SMALL = "font-family:var(--ea-font-ui);font-size:0.78rem;color:var(--ea-ink-2)"


def _marked(text: str) -> str:
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code, as
    in the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _bare(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _int(n: int, lang: str) -> str:
    """A whole number with a separator per thousand: 127,846 in English, 127.846 in Dutch."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x + 0.0:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _table(rows_: list, head: list, numbers: tuple = (), shade: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers are the columns set to the right; the rows in
    `shade` get a gray band."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join(
        f'<tr style="{"background:var(--ea-data-wash)" if i in shade else ""}">'
        + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>' for j, c in enumerate(row))
        + "</tr>" for i, row in enumerate(rows_))
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _bar(value: float, top: float, style: str, scale: float = 180) -> str:
    width = max(0.0, min(1.0, value / top if top else 0.0)) * scale
    return f'<span style="{_BAR};width:{width:.0f}px;background:{style}"></span>'


def _short(value, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _sense(name: str, lang: str, dutch: bool = True, words: int = 3) -> str:
    """A sense as the pictures write it: its name, then its first words, then its Dutch words in italics."""
    s = wordnet.synset(name)
    shown = ", ".join(w.replace("_", " ") for w, _ in s.words[:words])
    nl = f' <i style="{_SMALL}">{html.escape(", ".join(s.dutch[:3]))}</i>' if dutch and s.dutch else ""
    return f'<code>{html.escape(name)}</code> <span style="{_SMALL}">{html.escape(shown)}</span>{nl}'


# --- the idea's picture: dog.n.01 and its neighbors ---------------------------------------------------------------------

_DOG_WORDS = {
    "en": {"title": "One sense of WordNet and its neighbors: dog.n.01",
           "up": "More general: its hypernyms",
           "words": "Its words",
           "dutch": "Dutch words",
           "part": "Its part",
           "down": "More specific: its {n} hyponyms, such as",
           "more": "and {n} more",
           "legend": "Gray: what WordNet holds for the sense `dog.n.01`, with its definition in English. Each sense is "
                     "named after a word, its part of speech and a number; the Dutch words, in italics, come from Open "
                     "Dutch WordNet.",
           "alt": "The sense dog.n.01 with its words dog, domestic dog and Canis familiaris, its Dutch words hond and "
                  "joekel, its two hypernyms canine.n.02 and domestic_animal.n.01, its part flag.n.07, a tail, and "
                  "{n} hyponyms such as puppy.n.01 and dalmatian.n.01."},
    "nl": {"title": "Eén betekenis uit WordNet en de buren ervan: dog.n.01",
           "up": "Algemener: de hyperoniemen",
           "words": "De woorden",
           "dutch": "Nederlandse woorden",
           "part": "Het deel",
           "down": "Specifieker: de {n} hyponiemen, zoals",
           "more": "en nog {n}",
           "legend": "Grijs: wat WordNet voor de betekenis `dog.n.01` bevat, met de definitie in het Engels. Elke "
                     "betekenis heet naar een woord, de woordsoort en een nummer; de Nederlandse woorden, schuin "
                     "gedrukt, komen uit Open Dutch WordNet.",
           "alt": "De betekenis dog.n.01 met de woorden dog, domestic dog en Canis familiaris, de Nederlandse woorden "
                  "hond en joekel, de twee hyperoniemen canine.n.02 en domestic_animal.n.01, het deel flag.n.07, een "
                  "staart, en {n} hyponiemen zoals puppy.n.01 en dalmatian.n.01."},
}


def hyponyms(name: str) -> list:
    """The senses whose hypernyms include `name`, in the order of the file."""
    def make():
        out = {}
        for s in wordnet.synsets().values():
            for h in s.hypernyms:
                out.setdefault(h, []).append(s.name)
        return out
    return list(kept(f"{_KEY}:hyponyms", make).get(name, ()))


def dog_picture(lang: str) -> str:
    w = _DOG_WORDS[lang]
    dog = wordnet.synset("dog.n.01")
    below = hyponyms("dog.n.01")
    box = "padding:0.35rem 0.5rem;border:1px solid var(--ea-line);border-radius:4px;margin:0.25rem 0"

    def label(text):
        return f'<div style="{_LABEL}">{html.escape(text)}</div>'
    up = label(w["up"]) + "".join(f'<div style="{box}">{_sense(n, lang)}</div>' for n in dog.hypernyms)
    middle = (f'<div style="{box};border:2px solid var(--ea-data)"><code><b>dog.n.01</b></code>'
              f'<div style="{_SMALL}">{html.escape(dog.definition)}</div>'
              + label(w["words"]) + "".join(_chip(x.replace("_", " ")) for x, _ in dog.words)
              + label(w["dutch"]) + "".join(_chip(x) for x in dog.dutch)
              + label(w["part"]) + "".join(f"<div>{_sense(p, lang)}: <span style=\"{_SMALL}\">"
                                            f"{html.escape(wordnet.synset(p).definition)}</span></div>" for p in dog.parts)
              + "</div>")
    shown = below[:8]
    down = (label(w["down"].format(n=len(below)))
            + "".join(_chip(n) for n in shown)
            + f'<span style="{_SMALL}"> {html.escape(w["more"].format(n=len(below) - len(shown)))}</span>')
    return yours.figure(w["title"], up + middle + down + _note(w["legend"]),
                        alt=w["alt"].format(n=len(below)))


# --- running the learner's function in a picture ------------------------------------------------------------------------


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_SECONDS = 10  # what the learner's count_senses gets in a picture, for all 184,845 pairs
CHAIN_SECONDS = 1     # what the learner's chain gets per ladder: a loop that never ends fills the memory meanwhile


def _call(fn, *args, seconds: float = PICTURE_SECONDS):
    """The learner's function within a time limit, on shallow copies (the data holds only strings and tuples);
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(a.copy() if isinstance(a, (list, dict)) else a for a in args)), seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


# --- the picture under exercise 2: the senses of every word, by your count_senses ---------------------------------------

_COUNT_WORDS = {
    "en": {"title": "The senses of all words of WordNet, counted by your function",
           "top": "The ten words with the most senses, and the words with as many as the tenth",
           "head": ["Word", "Senses"],
           "one": "Words with exactly one sense: {one} of {n}, {share}%.",
           "bands": "How often a word is used, and its senses",
           "bands_head": ["Words", "In WordNet", "Senses on average"],
           "band_names": ["the 1,000 most frequent words of GloVe", "the next 4,000", "the next 15,000",
                          "the next 30,000", "not among the 50,000 of GloVe"],
           "legend": "Blue: your function's numbers; purple where a number differs from the course's count. GloVe's "
                     "50,000 words, from the most to the least frequent in its texts, say how often a word is used; "
                     "WordNet has no such numbers. GloVe writes every word in lowercase, so a word with a capital, such as *Monday*, "
                     "counts among the words GloVe does not have. Words that WordNet does not have, such as *the*, are "
                     "left out.",
           "fine": "Your function gives every word of WordNet the same number as the course.",
           "differ": "For {n} words, your function's number differs from the course's, or the word is not in "
                     "WordNet.",
           "error": "Your function stops with an error on the 184,845 pairs of WordNet.",
           "slow": "Your function takes more than {s} seconds for the 184,845 pairs of WordNet.",
           "odd": "On the 184,845 pairs of WordNet, your function gives back {got}, and not a dict."},
    "nl": {"title": "De betekenissen van alle woorden van WordNet, geteld door je functie",
           "top": "De tien woorden met de meeste betekenissen, en de woorden met evenveel als de tiende",
           "head": ["Woord", "Betekenissen"],
           "one": "Woorden met precies één betekenis: {one} van de {n}, {share}%.",
           "bands": "Hoe vaak een woord gebruikt wordt, en de betekenissen ervan",
           "bands_head": ["Woorden", "In WordNet", "Gemiddeld aantal betekenissen"],
           "band_names": ["de 1.000 meest voorkomende woorden van GloVe", "de volgende 4.000", "de volgende 15.000",
                          "de volgende 30.000", "niet bij de 50.000 van GloVe"],
           "legend": "Blauw: de getallen van je functie; paars waar een getal verschilt van de telling van de course. "
                     "De 50.000 woorden van GloVe, van het meest tot het minst voorkomende woord in de teksten van "
                     "GloVe, zeggen hoe vaak een woord gebruikt wordt; WordNet heeft zulke getallen niet. GloVe schrijft elk "
                     "woord met kleine letters, dus een woord met een hoofdletter, zoals *Monday*, telt bij de woorden "
                     "die GloVe niet heeft. Woorden die niet in WordNet staan, zoals *the*, tellen niet mee.",
           "fine": "Je functie geeft elk woord van WordNet hetzelfde getal als de course.",
           "differ": "Je functie geeft {n} woorden van WordNet een ander getal dan de course, of woorden die niet in "
                     "WordNet staan.",
           "error": "Je functie stopt met een fout op de 184.845 paren van WordNet.",
           "slow": "Je functie doet er langer dan {s} seconden over voor de 184.845 paren van WordNet.",
           "odd": "Op de 184.845 paren van WordNet geeft je functie {got} terug, en geen dict."},
}


def count_picture(seen) -> str:
    """The learner's count_senses on all pairs of WordNet: the ten words with the most senses, the share with one
    sense, and the mean number of senses by how often GloVe's texts use a word."""
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _COUNT_WORDS[lang]
    try:
        got = _call(fn, list(_pairs()))
    except Timeout:
        return yours.figure(w["title"], _note(w["slow"].format(s=PICTURE_SECONDS)),
                            alt=w["slow"].format(s=PICTURE_SECONDS))
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["error"]), alt=w["error"])
    if not isinstance(got, dict):
        text = w["odd"].format(got=_short(got))
        return yours.figure(w["title"], _note(text), alt=text)
    course = sense_counts()
    mine = {k: v for k, v in got.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    differ = sum(1 for k in course if mine.get(k) != course[k]) + sum(1 for k in got if k not in course)
    top = top_words(mine)
    high = max([v for _, v in top] + [1])
    rows, alts = [], []
    for word, value in top:
        color = "var(--ea-pred)" if course.get(word) == value else "var(--ea-loss)"
        shown = _int(int(value), lang) if float(value).is_integer() else _dec(value, lang)
        rows.append([html.escape(str(word)), f'{_bar(value, high, color)} '
                     f'<span style="font-variant-numeric:tabular-nums">{shown}</span>'])
        alts.append(f"{word} {value}")
    words = [k for k in mine if isinstance(k, str)]
    one = sum(1 for k in words if mine[k] == 1)
    share = _dec(100 * one / len(words), lang) if words else "0"
    body = f'<div style="{_LABEL}">{html.escape(w["top"])}</div>' + _table(rows, w["head"], numbers=(1,))
    body += f'<div style="{_LABEL}">{html.escape(w["one"].format(one=_int(one, lang), n=_int(len(words), lang), share=share))}</div>'
    course_bands = band_means(course)
    mine_bands = band_means({k: mine[k] for k in words})
    top_mean = max([m for _, m in mine_bands] + [1.0])
    band_rows = []
    for name, (n, m), (_, want) in zip(w["band_names"], mine_bands, course_bands):
        color = "var(--ea-pred)" if abs(m - want) < 1e-9 else "var(--ea-loss)"
        band_rows.append([html.escape(name), _int(n, lang), f'{_bar(m, top_mean, color)} '
                          f'<span style="font-variant-numeric:tabular-nums">{_dec(m, lang)}</span>'])
        alts.append(f"{name}: {_dec(m, lang)}")
    body += f'<div style="{_LABEL}">{html.escape(w["bands"])}</div>' + _table(band_rows, w["bands_head"], numbers=(1,))
    text = (w["fine"] if not differ else w["differ"].format(n=_int(differ, lang))) + " " + w["legend"]
    return yours.figure(w["title"], body + _note(text),
                        alt=f'{w["title"]}. ' + "; ".join(alts) + ". " + _bare(text))


# --- the picture under exercise 3: two pairs of ladders, by your chain --------------------------------------------------

LADDERS = [("poodle.n.01", "cat.n.01"), ("bank.n.01", "depository_financial_institution.n.01")]

_CHAIN_WORDS = {
    "en": {"title": "Up the ladder of first hypernyms, by your function",
           "meet": "The two ladders meet at {name}.",
           "never": "The two ladders do not meet.",
           "legend": "Each column is what your function gives for that sense with `C.PARENT`, drawn from the bottom "
                     "up: the sense at the bottom, its parent above it, and so on up to the top. The two ladders stand "
                     "side by side from their top, so a row on a gray band holds the same sense in both. Blue: your "
                     "function's ladder is the course's; purple: it differs. Beside each name, in italics: its Dutch "
                     "words.",
           "error": "Your function stops with an error on `chain(\"{name}\", C.PARENT)`.",
           "slow": "Your function is still running after {s} second with `chain(\"{name}\", C.PARENT)`.",
           "odd": "With `chain(\"{name}\", C.PARENT)`, your function gives back {got}, and not a list of names."},
    "nl": {"title": "De ladder van eerste hyperoniemen op, door je functie",
           "meet": "De twee ladders komen samen bij {name}.",
           "never": "De twee ladders komen niet samen.",
           "legend": "Elke kolom is wat je functie voor die betekenis met `C.PARENT` geeft, van onder naar boven "
                     "getekend: de betekenis onderaan, de ouder erboven, enzovoort tot de top. De twee ladders staan "
                     "naast elkaar vanaf de top, dus een rij op een grijze band heeft in allebei dezelfde betekenis. "
                     "Blauw: de ladder van je functie is die van de course; paars: de ladder verschilt. Naast elke "
                     "naam, schuin gedrukt: de Nederlandse woorden.",
           "error": "Je functie stopt met een fout op `chain(\"{name}\", C.PARENT)`.",
           "slow": "Je functie is na {s} seconde nog bezig met `chain(\"{name}\", C.PARENT)`.",
           "odd": "Met `chain(\"{name}\", C.PARENT)` geeft je functie {got} terug, en geen lijst met namen."},
}


def _rung(name, lang: str, style: str) -> str:
    if isinstance(name, str) and name in wordnet.synsets():
        s = wordnet.synset(name)
        nl = f' <i style="{_SMALL}">{html.escape(", ".join(s.dutch[:2]))}</i>' if s.dutch else ""
        return f"{_chip(name, style)}{nl}"
    return _chip(_short(name, 40), _CHIP_PURPLE)


def chain_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang = seen.lang
    w = _CHAIN_WORDS[lang]
    body, alts, notes = "", [], []
    for pair in LADDERS:
        ladders = []
        for name in pair:
            try:
                got = _call(fn, name, PARENT, seconds=CHAIN_SECONDS)
            except Timeout:
                notes.append(w["slow"].format(name=name, s=CHAIN_SECONDS))
                break
            if got is _STOPPED:
                notes.append(w["error"].format(name=name))
                break
            if not isinstance(got, (list, tuple)) or not got:
                notes.append(w["odd"].format(name=name, got=_short(got)))
                break
            ladders.append(list(got))
        if len(ladders) < 2:
            break
        tall = max(len(x) for x in ladders)
        padded = [x[::-1] + [None] * (tall - len(x)) for x in ladders]
        styles = [_CHIP_PRED if x == chain(name, PARENT) else _CHIP_PURPLE for x, name in zip(ladders, pair)]
        rows, shade = [], []
        for i in range(tall):
            a, b = padded[0][i], padded[1][i]
            if a is not None and a == b:
                shade.append(i)
            rows.append([_rung(a, lang, styles[0]) if a is not None else "",
                         _rung(b, lang, styles[1]) if b is not None else ""])
        met = meeting(ladders[0], ladders[1])
        line = w["meet"].format(name=met) if met else w["never"]
        body += _table(rows, list(pair), shade=tuple(shade)) + f'<div style="{_LABEL}">{html.escape(line)}</div>'
        alts.append(f"{pair[0]}: {len(ladders[0])}, {pair[1]}: {len(ladders[1])}. {line}")
    text = " ".join(notes + [w["legend"]])
    return yours.figure(w["title"], body + _note(text), alt=f'{w["title"]}. ' + " ".join(alts) + " " + _bare(text))


# --- try it: the senses of a word -------------------------------------------------------------------------------------

LONGEST = 60    # characters of the typed word that count
MOST = 12       # senses in the table

_EXPLORE_WORDS = {
    "en": {"title": "The senses of {word} in WordNet",
           "head": ["", "Sense", "Its words", "Dutch words", "Definition"],
           "more": "And {n} more senses.",
           "ladder": "The ladder of sense {k}, from the top down to {name}",
           "parts": "Its parts", "members": "Its members", "substances": "Made of", "antonyms": "Opposite",
           "down": "More specific: {n} hyponyms",
           "legend": "Gray: what WordNet holds. The Dutch words come from Open Dutch WordNet, which links them to the "
                     "English senses; some of those links are wrong.",
           "none_en": "WordNet has no English word *{word}*. Try another form, such as one without -s, or a word of "
                      "several words with spaces.",
           "none_nl": "Open Dutch WordNet links no sense to the Dutch word *{word}*. Try the base form, such as *hond* "
                      "for *honden*.",
           "empty": "Type a word.",
           "high": "*{word}* has {n} senses; the ladder below is that of sense {n}."},
    "nl": {"title": "De betekenissen van {word} in WordNet",
           "head": ["", "Betekenis", "De woorden", "Nederlandse woorden", "Definitie"],
           "more": "En nog {n} betekenissen.",
           "ladder": "De ladder van betekenis {k}, van de top omlaag naar {name}",
           "parts": "De delen", "members": "De leden", "substances": "Gemaakt van", "antonyms": "Tegendeel",
           "down": "Specifieker: {n} hyponiemen",
           "legend": "Grijs: wat WordNet bevat. De Nederlandse woorden komen uit Open Dutch WordNet, dat de woorden aan "
                     "de Engelse betekenissen koppelt; sommige van die koppelingen kloppen niet.",
           "none_en": "WordNet heeft geen Engels woord *{word}*. Probeer een andere vorm, zoals een zonder -s, of een "
                      "woord van meer woorden met spaties.",
           "none_nl": "Open Dutch WordNet koppelt geen betekenis aan het Nederlandse woord *{word}*. Probeer de "
                      "grondvorm, zoals *hond* voor *honden*.",
           "empty": "Typ een woord.",
           "high": "*{word}* heeft {n} betekenissen; de ladder hieronder is die van betekenis {n}."},
}


def _plain_word(word: str) -> str:
    """A typed word for a note: without asterisks and backticks, which the note reads as italics and code."""
    return word.replace("*", "").replace("`", "")


def explore_names(word: str, language: str) -> list:
    """The senses of a typed word: English through wordnet.senses, Dutch through wordnet.dutch_senses."""
    if language == "Nederlands":
        return wordnet.dutch_senses(word) or wordnet.dutch_senses(word.lower())
    return wordnet.senses(word) or wordnet.senses(word.lower())


def explore_picture(word: str, language: str, number, lang: str) -> str:
    w = _EXPLORE_WORDS[lang]
    word = " ".join(str(word)[:LONGEST].split())
    title = w["title"].format(word=word or "…")
    if not word:
        return yours.figure(title, _note(w["empty"]), alt=w["empty"])
    names = explore_names(word, language)
    if not names:
        text = w["none_nl" if language == "Nederlands" else "none_en"].format(word=_plain_word(word))
        return yours.figure(title, _note(text), alt=_bare(text))
    rows = []
    for k, name in enumerate(names[:MOST], start=1):
        s = wordnet.synset(name)
        rows.append([str(k), f"<code>{html.escape(name)}</code>",
                     html.escape(", ".join(x.replace("_", " ") for x, _ in s.words)),
                     f"<i>{html.escape(', '.join(s.dutch))}</i>", html.escape(s.definition)])
    body = _table(rows, w["head"], numbers=(0,))
    notes = [w["more"].format(n=len(names) - MOST)] if len(names) > MOST else []
    try:
        k = int(number)
    except (TypeError, ValueError):
        k = 1
    k = max(1, k)
    if k > len(names):
        notes.append(w["high"].format(word=_plain_word(word), n=len(names)))
        k = len(names)
    name = names[k - 1]
    s = wordnet.synset(name)
    body += f'<div style="{_LABEL}">{html.escape(w["ladder"].format(k=k, name=name))}</div>'
    body += "<br>".join(_sense(n, lang, words=2) for n in chain(name, PARENT)[::-1])
    for key, found in (("parts", s.parts), ("members", s.members), ("substances", s.substances),
                       ("antonyms", [other for _, _, other in s.antonyms])):
        if found:
            body += (f'<div style="{_LABEL}">{html.escape(w[key])}</div>'
                     + "".join(_chip(n) for n in dict.fromkeys(found)))
    below = hyponyms(name)
    if below:
        body += (f'<div style="{_LABEL}">{html.escape(w["down"].format(n=len(below)))}</div>'
                 + "".join(_chip(n) for n in below[:12]) + (" …" if len(below) > 12 else ""))
    text = " ".join(notes + [w["legend"]])
    return yours.figure(title, body + _note(text),
                        alt=f"{title}: " + "; ".join(f"{n}: {wordnet.synset(n).definition}" for n in names[:MOST])
                        + ". " + _bare(text))


# --- the exercises -------------------------------------------------------------------------------------------------------

REQUIRED = ["relations", "count_senses", "chain"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name, as the lesson's code writes it.
    The learner's function gets a shallow copy: the data holds only strings and tuples."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return self.value.copy()


class _Pairs(list):
    """A small list of (word, name) pairs that a message writes the way the lesson's code does, with double quotes.
    The learner's code gets a plain copy."""

    def __repr__(self):
        return "[" + ", ".join(f'("{a}", "{b}")' for a, b in self) + "]"

    def __deepcopy__(self, memo):
        return list(self)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")


def _shown(v):
    """What a function gave, as a message shows it: a small dict or list in full, as code; a large one by its size."""
    if isinstance(v, dict) and len(v) > 8:
        return {"en": f"a dict with {len(v):,} keys", "nl": f"een dict met {_int(len(v), 'nl')} keys"}
    if isinstance(v, list) and len(v) > 8:
        head = ", ".join(_code(x) for x in v[:3])
        return {"en": f"a list of {len(v)} items that starts with {head}",
                "nl": f"een lijst van {len(v)} items die begint met {head}"}
    return _code(v)


def _code(v) -> str:
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_code(k)}: {_code(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_code(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(_code(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return repr(v)


def _same_dict(got, want) -> bool:
    return isinstance(got, dict) and dict(got) == want


def _same_list(got, want) -> bool:
    return isinstance(got, list) and got == want


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _a_number(got, text) -> bool:
    return isinstance(got, (int, float)) and not isinstance(got, bool)


def _a_name(got, text) -> bool:
    return isinstance(got, str)


def _return_in_loop(got, text) -> bool:
    """A `return` indented under a `for` or `while` line: the loop then ends in its first round."""
    lines = [line.split("#")[0].rstrip() for line in text.splitlines()]
    for i, line in enumerate(lines):
        head = line.lstrip()
        if head.startswith(("for ", "while ")) and head.endswith(":"):
            depth = len(line) - len(head)
            for after in lines[i + 1:]:
                body = after.lstrip()
                if not body:
                    continue
                if len(after) - len(body) <= depth:
                    break
                if body == "return" or body.startswith(("return ", "return(")):
                    return True
    return False


# exercise 2: count_senses. The visible case and an empty list; then one word with three senses, two words that share
# a sense, a word that comes back after another word, and the real senses of three words of WordNet.
COUNT_CASES = [
    (_Pairs([("bank", "bank.n.01"), ("couch", "sofa.n.01"), ("bank", "depository_financial_institution.n.01"),
             ("sofa", "sofa.n.01")]),),
    (_Pairs([]),),
    (_Pairs([("dog", "dog.n.01"), ("dog", "frump.n.01"), ("dog", "dog.n.03")]),),
    (_Pairs([("auto", "car.n.01"), ("car", "car.n.01")]),),
    (_Pairs([("light", "light.n.01"), ("bass", "bass.n.01"), ("light", "light.v.01"), ("bass", "bass.n.02"),
             ("light", "light.a.01")]),),
    (Named('C.pairs_for(["break", "bank", "poodle"])', pairs_for(["break", "bank", "poodle"])),),
]

_CS_HEAD = "def count_senses(pairs):\n"
_CS_START = _CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        ...\n    return counts\n"
_CS = (_CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        if word in counts:\n"
       "            counts[word] = counts[word] + 1\n        else:\n            counts[word] = 1\n    return counts\n")
_CS_ONE = _CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        counts[word] = 1\n    return counts\n"
_CS_ZERO = _CS.replace("        else:\n            counts[word] = 1\n", "        else:\n            counts[word] = 0\n")
_CS_NAMES = _CS.replace("word in counts", "name in counts").replace("counts[word]", "counts[name]")
_CS_KEYERROR = (_CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        counts[word] = counts[word] + 1\n"
                "    return counts\n")
_CS_TOTAL = _CS.replace("    return counts\n", "    return len(counts)\n")
_CS_HIGH = (_CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        if word not in counts:\n"
            "            counts[word] = 1\n        counts[word] = counts[word] + 1\n    return counts\n")
_CS_RETURN = _CS.replace("            counts[word] = 1\n    return counts\n", "            counts[word] = 1\n        return counts\n")
_CS_GET = (_CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        counts[word] = counts.get(word, 0) + 1\n"
           "    return counts\n")

# exercise 3: chain. The visible cases; then a longer chain, a name with no parent at all, a dict with branches, and
# three senses of WordNet with C.PARENT, the first hypernym of every sense that has one.
_SMALL_TREE = {"poodle": "dog", "dog": "animal"}
_BRANCHES = {"oak": "tree", "pine": "tree", "tree": "plant", "rose": "shrub", "shrub": "plant", "plant": "organism"}
_PARENT_NAMED = Named("C.PARENT", PARENT)
CHAIN_CASES = [
    ("poodle", _SMALL_TREE),
    ("animal", _SMALL_TREE),
    ("a", {"a": "b", "b": "c", "c": "d", "d": "e", "e": "f"}),
    ("rock", _SMALL_TREE),
    ("rose", _BRANCHES),
    ("poodle.n.01", _PARENT_NAMED),
    ("bank.n.01", _PARENT_NAMED),
    ("run.v.01", _PARENT_NAMED),
]

_CH_HEAD = "def chain(name, parent):\n"
_CH_START = _CH_HEAD + "    out = [name]\n    ...\n    return out\n"
_CH = _CH_HEAD + "    out = [name]\n    while out[-1] in parent:\n        out.append(parent[out[-1]])\n    return out\n"
_CH_IF = _CH.replace("while out[-1] in parent:", "if out[-1] in parent:")
_CH_NAME = _CH_HEAD + "    out = [name]\n    while name in parent:\n        out.append(parent[name])\n    return out\n"
_CH_FOREVER = _CH_HEAD + "    out = [name]\n    while True:\n        out.append(parent[out[-1]])\n    return out\n"
_CH_RETURN = _CH.replace("        out.append(parent[out[-1]])\n", "        out.append(parent[out[-1]])\n        return out\n")
_CH_EARLY = _CH.replace("while out[-1] in parent:", "while parent[out[-1]] in parent:")
_CH_NO_START = (_CH_HEAD + "    out = []\n    while name in parent:\n        name = parent[name]\n        out.append(name)\n"
                "    return out\n")
_CH_TOP = _CH.replace("    return out\n", "    return out[-1]\n")
_CH_REVERSED = (_CH_HEAD + "    out = [name]\n    while out[0] in parent:\n        out.insert(0, parent[out[0]])\n"
                "    return out\n")
_CH_NAME_MOVES = (_CH_HEAD + "    out = [name]\n    while name in parent:\n        name = parent[name]\n"
                  "        out.append(name)\n    return out\n")

EXERCISES = {
    "relations": Sort(
        name="relations",
        groups={
            "senses": {"en": "Two senses of one word", "nl": "Twee betekenissen van één woord"},
            "synonymy": {"en": "Synonymy: two words, one sense", "nl": "Synonymie: twee woorden, één betekenis"},
            "hyponymy": {"en": "Hyponymy: the first is a kind of the second",
                         "nl": "Hyponymie: het eerste is een soort van het tweede"},
            "meronymy": {"en": "Meronymy: the first is a part or member of the second",
                         "nl": "Meronymie: het eerste is een deel of lid van het tweede"},
        },
        items={
            "tree": {"en": "*tree* and *forest*", "nl": "*boom* en *bos*"},
            "doctor": {"en": "*doctor* and *physician*", "nl": "*dokter* en *arts*"},
            "mouse": {"en": "*mouse* (the small animal) and *mouse* (on a desk, for the pointer)",
                      "nl": "*muis* (het diertje) en *muis* (op een bureau, voor de pijl op het scherm)"},
            "salmon": {"en": "*salmon* and *fish*", "nl": "*zalm* en *vis*"},
            "finger": {"en": "*finger* and *hand*", "nl": "*vinger* en *hand*"},
            "bike": {"en": "*bicycle* and *bike*", "nl": "*fiets* en *rijwiel*"},
            "bat": {"en": "*bat* (an animal that flies at night) and *bat* (the club used in cricket)",
                    "nl": "*slot* (op een deur) en *slot* (een kasteel)"},
            "sparrow": {"en": "*sparrow* and *bird*", "nl": "*mus* en *vogel*"},
        },
        solution={"tree": "meronymy", "doctor": "synonymy", "mouse": "senses", "salmon": "hyponymy",
                  "finger": "meronymy", "bike": "synonymy", "bat": "senses", "sparrow": "hyponymy"},
        nudge={
            "tree": {"en": "Is a tree a kind of forest, or do many trees together make one?",
                     "nl": "Is een boom een soort bos, of maken veel bomen samen één bos?"},
            "doctor": {"en": "Can you swap the two words in *the doctor will see you now* and keep the meaning?",
                       "nl": "Kun je de twee woorden omwisselen in *de dokter komt zo* zonder dat de betekenis verandert?"},
            "mouse": {"en": "How many different words do you see here, and how many meanings?",
                      "nl": "Hoeveel verschillende woorden zie je hier, en hoeveel betekenissen?"},
            "salmon": {"en": "Is every salmon a fish?", "nl": "Is elke zalm een vis?"},
            "finger": {"en": "Is a finger a kind of hand?", "nl": "Is een vinger een soort hand?"},
            "bike": {"en": "Do the two words name the same thing?", "nl": "Noemen de twee woorden hetzelfde ding?"},
            "bat": {"en": "Is this one word or two, and how many meanings does it have here?",
                    "nl": "Is dit één woord of twee, en hoeveel betekenissen heeft het hier?"},
            "sparrow": {"en": "Is a sparrow a part of a bird?", "nl": "Is een mus een deel van een vogel?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
            "empty": {"en": "Put each pair under the relation between its two words.",
                      "nl": "Zet elk paar onder de relatie tussen de twee woorden."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The pair {item} belongs under another heading.",
                      "nl": "Het paar {item} hoort onder een ander kopje."},
            "correct": {"en": "All {n} are right. One form with two senses, two forms with one sense, and two relations between senses: a kind of something more general, and a part or member of a whole.",
                        "nl": "Alle {n} kloppen. Eén vorm met twee betekenissen, twee vormen met één betekenis, en twee relaties tussen betekenissen: een soort van iets algemeners, en een deel of lid van een geheel."},
        },
    ),
    "count_senses": FunctionBox(
        name="count_senses",
        picture=count_picture,
        function="count_senses",
        start=_CS_START,
        solution=_CS,
        visible=COUNT_CASES[:2],
        hidden=COUNT_CASES[2:],
        compare=_same_dict,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("return-in-loop", matches=_return_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: `return counts` stands inside the loop, so your function gives back the dict after the first pair. How far should `return counts` be indented, so that it runs after the loop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `return counts` staat in de loop, dus je functie geeft de dict al na het eerste paar terug. Hoe ver moet `return counts` inspringen, zodat de regel na de loop komt?"),
            Mistake("key-error", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: the first time a word comes by, `counts[word]` does not exist yet, so Python cannot add 1 to it. What can you ask about `counts` before you read the number of a word?",
                    nl="Met `{call}` stopt je functie met een KeyError: de eerste keer dat een woord langskomt, bestaat `counts[word]` nog niet, dus Python kan er geen 1 bij optellen. Wat kun je aan `counts` vragen voordat je het getal van een woord leest?"),
            Mistake("always-one", wrong=_CS_ONE,
                    en="`{call}` gives {got}, and it should give {want}: every word gets 1, also a word that comes by a second time. What should happen to the number of a word that came by before?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk woord krijgt 1, ook een woord dat voor de tweede keer langskomt. Wat moet er gebeuren met het getal van een woord dat al eerder langskwam?"),
            Mistake("one-too-high", wrong=_CS_HIGH,
                    en="`{call}` gives {got}, and it should give {want}: every number is one too high. Does your function count a new word twice, once when it gets its first number and once more right after?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk getal is één te hoog. Telt je functie een nieuw woord twee keer: één keer als het een eerste getal krijgt, en meteen daarna nog een keer?"),
            Mistake("start-zero", wrong=_CS_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: every number is one too low. A word that comes by for the first time has one sense so far: what should its number be?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elk getal is één te laag. Een woord dat voor de eerste keer langskomt, heeft tot dan toe één betekenis: welk getal hoort er dan bij?"),
            Mistake("names", wrong=_CS_NAMES,
                    en="`{call}` gives {got}, and it should give {want}: your dict counts per name of a sense, which is how many words share that sense. That is synonymy, and a fine count too, but the dict needs one key per word: which of the two in a pair is the word?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je dict telt per naam van een betekenis, dus hoeveel woorden die betekenis delen. Dat is synonymie, en ook een mooie telling, maar de dict heeft één key per woord nodig: welke van de twee in een paar is het woord?"),
            Mistake("total", matches=_a_number,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back one number. It should give back the dict itself, with a number for every word: what comes after `return`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft één getal terug, en hoort de dict zelf terug te geven, met een getal voor elk woord: wat komt er na `return`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give each word its number of pairs, 1 the first time it comes by and 1 more each time after that.",
                      "nl": "Vervang de `...`: geef elk woord het aantal paren met dat woord: 1 de eerste keer dat het langskomt, en 1 erbij elke keer daarna."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count for every word the pairs it is in: 1 the first time it comes by, and 1 more each time after that.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel voor elk woord de paren waarin het staat: 1 de eerste keer dat het langskomt, en 1 erbij elke keer daarna."},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function give the right number to every word, also to a word with three senses, and to two words that share one sense?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Geeft je functie elk woord het goede getal, ook een woord met drie betekenissen, en twee woorden die één betekenis delen?"},
            "correct": {"en": "Your `count_senses` is right in all {n} cases, also for the real senses of *break*, *bank* and *poodle*. Look at the picture: your function counted the senses of every word of WordNet.",
                        "nl": "Je `count_senses` klopt in alle {n} gevallen, ook voor de echte betekenissen van *break*, *bank* en *poodle*. Kijk naar de afbeelding: je functie telde de betekenissen van elk woord van WordNet."},
        },
    ),
    "chain": FunctionBox(
        name="chain",
        picture=chain_picture,
        function="chain",
        start=_CH_START,
        solution=_CH,
        visible=CHAIN_CASES[:2],
        hidden=CHAIN_CASES[2:],
        compare=_same_list,
        show=_shown,
        first=[_DOTS],
        time_limit=1.0,
        mistakes=[
            Mistake("key-error", error=_key_error,
                    en="With `{call}`, your function stops with a KeyError: it looks up the parent of a name that has none, because the top is not a key of `parent`. When should the loop stop, and what does the line with `while` ask before every round?",
                    nl="Met `{call}` stopt je functie met een KeyError: je functie zoekt de ouder op van een naam die er geen heeft, want de top is geen key van `parent`. Wanneer moet de loop stoppen, en wat vraagt de regel met `while` vóór elke ronde?"),
            Mistake("return-in-loop", matches=_return_in_loop,
                    en="`{call}` gives {got}, and it should give {want}: `return out` stands inside the loop, so your function gives back the list after the first round. How far should `return out` be indented, so that it runs after the loop?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `return out` staat in de loop, dus je functie geeft de lijst al na de eerste ronde terug. Hoe ver moet `return out` inspringen, zodat de regel na de loop komt?"),
            Mistake("once", wrong=_CH_IF,
                    en="`{call}` gives {got}, and it should give {want}: your list goes up one step and stops there. `if` looks once; which word repeats the lines under it for as long as the condition holds?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst gaat één stap omhoog en stopt daar. `if` kijkt één keer; welk woord herhaalt de regels eronder zolang de voorwaarde klopt?"),
            Mistake("early", wrong=_CH_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your ladder stops one step below the top, and for a name without a parent your function stops with an error. When should the loop go on, and what does your condition ask?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je ladder stopt één stap onder de top, en voor een naam zonder ouder stopt je functie met een fout. Wanneer moet de loop doorgaan, en wat vraagt jouw voorwaarde?"),
            Mistake("no-start", wrong=_CH_NO_START,
                    en="`{call}` gives {got}, and it should give {want}: the list misses its first item, `name` itself. Where does the ladder start?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de lijst mist het eerste item, `name` zelf. Waar begint de ladder?"),
            Mistake("top-only", matches=_a_name,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back one name. It should give back the whole list, from `name` up to the top: what comes after `return`?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft één naam terug, en hoort de hele lijst terug te geven, van `name` tot de top: wat komt er na `return`?"),
            Mistake("reversed", wrong=_CH_REVERSED,
                    en="`{call}` gives {got}, and it should give {want}: your list runs from the top down. The ladder starts at `name` and each next item is the parent of the one before it: where does each new name go in the list?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst loopt van de top naar beneden. De ladder begint bij `name` en elk volgend item is de ouder van het item ervoor: waar komt elke nieuwe naam in de lijst?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a `while` loop that adds a parent to `out` in every round, until it reaches a sense without a parent.",
                      "nl": "Vervang de `...` door een `while`-loop die elke ronde een ouder aan `out` toevoegt, tot de loop bij een betekenis zonder ouder komt."},
            "slow": {"en": "Your function is still running after 1 second on a case of the check: the loop never stops. Does its condition look at something that changes every round, or at something that stays the same, such as `name`? And does the loop add a new name in every round, or the same one?",
                     "nl": "Je functie is na 1 seconde nog bezig met een geval van de check: de loop stopt nooit. Kijkt de voorwaarde naar iets wat elke ronde verandert, of naar iets wat hetzelfde blijft, zoals `name`? En voegt de loop elke ronde een nieuwe naam toe, of steeds dezelfde?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start the list with `name`, and add one parent after another, until a sense has no parent.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin de lijst met `name`, en voeg de ene ouder na de andere toe, tot een betekenis geen ouder heeft."},
            "almost": {"en": "Right for the first two cases. But `{call}` gives {got}, and it should give {want}. Does your function work for a longer ladder too, and for a name that has no parent at all?",
                       "nl": "Goed voor de eerste twee gevallen. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een langere ladder, en voor een naam die helemaal geen ouder heeft?"},
            "correct": {"en": "Your `chain` is right in all {n} cases, also for three senses of WordNet. Look at the picture: where do the ladders of two senses meet?",
                        "nl": "Je `chain` klopt in alle {n} gevallen, ook voor drie betekenissen van WordNet. Kijk naar de afbeelding: waar komen de ladders van twee betekenissen samen?"},
        },
    ),
}

CASES = [
    ("relations", {}, "empty", None),
    ("relations", {"tree": "meronymy", "doctor": "synonymy", "mouse": "senses", "salmon": "hyponymy",
                   "finger": "meronymy", "bike": "synonymy", "bat": "senses", "sparrow": "hyponymy"}, "correct", None),
    ("relations", {"tree": "meronymy", "doctor": "synonymy"}, "wrong", "unfinished"),
    ("relations", {"tree": "hyponymy"}, "wrong", "wrong:tree"),
    ("relations", {"doctor": "senses"}, "wrong", "wrong:doctor"),
    ("relations", {"mouse": "synonymy"}, "wrong", "wrong:mouse"),
    ("relations", {"salmon": "meronymy"}, "wrong", "wrong:salmon"),
    ("relations", {"finger": "hyponymy"}, "wrong", "wrong:finger"),
    ("relations", {"bike": "senses"}, "wrong", "wrong:bike"),
    ("relations", {"bat": "synonymy"}, "wrong", "wrong:bat"),
    ("relations", {"sparrow": "meronymy"}, "wrong", "wrong:sparrow"),
    ("count_senses", _CS_START, "empty", None),
    ("count_senses", _CS, "correct", None),
    ("count_senses", _CS_GET, "correct", None),
    ("count_senses", _CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        if word not in counts:\n"
                     "            counts[word] = 0\n        counts[word] += 1\n    return counts\n", "correct", None),
    ("count_senses", _CS_HEAD + "    counts = {}\n    for word, name in pairs:\n        ...\n    return {w: len([1 for x, _ in pairs if x == w]) for w, _ in pairs}\n",
     "wrong", "dots-left"),
    ("count_senses", _CS_KEYERROR, "error", "key-error"),
    ("count_senses", _CS_ONE, "wrong", "always-one"),
    ("count_senses", _CS_ZERO, "wrong", "start-zero"),
    ("count_senses", _CS_NAMES, "wrong", "names"),
    ("count_senses", _CS_TOTAL, "wrong", "total"),
    ("count_senses", _CS_HIGH, "wrong", "one-too-high"),
    ("count_senses", _CS_RETURN, "wrong", "return-in-loop"),
    ("chain", _CH_START, "empty", None),
    ("chain", _CH, "correct", None),
    ("chain", _CH_NAME_MOVES, "correct", None),
    ("chain", _CH_HEAD + "    out = [name]\n    while name in parent:\n        name = parent[name]\n        out = out + [name]\n"
              "    return out\n", "correct", None),
    ("chain", _CH_IF, "wrong", "once"),
    ("chain", _CH_NAME, "slow", None),
    ("chain", _CH_FOREVER, "error", "key-error"),
    ("chain", _CH_EARLY, "wrong", "early"),
    ("chain", _CH_NO_START, "wrong", "no-start"),
    ("chain", _CH_TOP, "wrong", "top-only"),
    ("chain", _CH_REVERSED, "wrong", "reversed"),
    ("chain", _CH_RETURN, "wrong", "return-in-loop"),
    ("chain", _CH_HEAD + "    out = [name]\n    ...\n    return [name]\n", "wrong", "dots-left"),
]
