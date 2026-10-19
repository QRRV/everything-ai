# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-pretraining-data: where six text datasets may go by the course's own rules for the data
of the models it shares (a Sort), by_license, the share of a corpus's tokens under every license, and two choices in the
lesson itself (open weights against open source, and an opt-out under the EU's text and data mining exception)."""
import html

from everything_ai.check import FunctionBox, Mistake, Sort, same_value

# Dolma v1.6, the open corpus of the Allen Institute for AI: Llama tokens per source, in billions (Soldaini and others,
# 2024, "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research", ACL 2024, table 1).
DOLMA = {"web pages (Common Crawl)": 2479, "code (GitHub)": 411, "social media (Reddit)": 89,
         "scientific papers (Semantic Scholar)": 70, "books (Project Gutenberg)": 6.0, "encyclopedia (Wikipedia, Wikibooks)": 4.3}
DOLMA_NL = {"webpagina's (Common Crawl)": 2479, "code (GitHub)": 411, "sociale media (Reddit)": 89,
            "wetenschappelijke papers (Semantic Scholar)": 70, "boeken (Project Gutenberg)": 6.0,
            "encyclopedie (Wikipedia, Wikibooks)": 4.3}
DOLMA_TOTAL = 3059   # the total in the table; the six sources add up to 3,059.3


def share(part: float, total: float = DOLMA_TOTAL) -> float:
    return 100 * part / total


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


def sources_picture(labels: dict, lang: str) -> str:
    """Dolma's sources as bars of their tokens, on one scale, with the share of each: ink only, as these are counts of
    a corpus. The smallest bars are too short to see, so their share stands next to them."""
    rows = DOLMA_NL if lang == "nl" else DOLMA
    W, BAR, ROW, TOP = 330, 14, 40, 30
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = TOP
    for name, tokens in rows.items():
        w = max(2.0, (W - 110) * tokens / max(rows.values()))
        value = f"{number(tokens, 0 if tokens >= 10 else 1, lang)} · {number(share(tokens), 1, lang)}%"
        p.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(name)}</text>'
                 f'<rect x="0" y="{y + 17}" width="{w:.1f}" height="{BAR}" rx="2" style="fill:var(--ea-ink)"/>'
                 f'<text x="{w + 6:.1f}" y="{y + 28.5}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(value)}</text>')
        y += ROW
    p.append(f'<text x="0" y="{y + 10}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["note"])}</text>')
    H = y + 18
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "open-weights" and "opt-out" are choice questions in the lesson.
REQUIRED = ["where", "by-license", "open-weights", "opt-out"]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...")
                                                          for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_B_HEAD = "def by_license(sources):\n"
_B_TOTAL = "    total = 0\n    for s in sources:\n        total = total + s[\"tokens\"]\n"
_B_ADD = ("    shares = {}\n    for s in sources:\n        if s[\"license\"] in shares:\n"
          "            shares[s[\"license\"]] = shares[s[\"license\"]] + s[\"tokens\"]\n        else:\n"
          "            shares[s[\"license\"]] = s[\"tokens\"]\n")
_B_ROUND = "    for lic in shares:\n        shares[lic] = round(100 * shares[lic] / total, 1)\n    return shares\n"
_B_START = (_B_HEAD + "    # 1. all tokens of the corpus\n    total = 0\n    ...\n"
            "    # 2. the tokens of every license, added up\n    shares = {}\n    ...\n"
            "    # 3. every license as a share of all tokens, in percent, rounded to 1 decimal\n    ...\n    return shares\n")


def _b(total: str = _B_TOTAL, add: str = _B_ADD, rnd: str = _B_ROUND) -> str:
    return _B_HEAD + total + add + rnd


EXAMPLE = [{"name": "Dutch Wikipedia", "tokens": 900, "license": "CC BY-SA 4.0"},
           {"name": "Project Gutenberg", "tokens": 300, "license": "public domain"},
           {"name": "code on GitHub (MIT)", "tokens": 250, "license": "MIT"},
           {"name": "code on GitHub (Apache)", "tokens": 60, "license": "Apache-2.0"},
           {"name": "Dutch web pages", "tokens": 3500, "license": "ODC-By 1.0"}]
def _shown(v) -> str:
    """A dict of shares the way Python shows it, with every float to at most 3 decimals: no float noise in a message."""
    def num(x):
        if isinstance(x, float):
            s = f"{x:.3f}".rstrip("0")
            return s + "0" if s.endswith(".") else s
        return repr(x)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {num(x)}" for k, x in v.items()) + "}"
    return num(v)


def _same_shares(got, want) -> bool:
    """The same licenses with the same shares, in any kind of dict: a defaultdict or a Counter counts too, and
    two ways to the same share, such as 100 * t / total and t / total * 100, are the same (as same_value)."""
    if isinstance(got, dict) and isinstance(want, dict):
        return got.keys() == want.keys() and all(same_value(got[k], want[k]) for k in want)
    return same_value(got, want)


_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_SORT_TEXTS = {
    "missing": {"en": "I cannot see your choices. Run the cell with the datasets again.",
                "nl": "Ik zie je keuzes niet. Voer de cel met de datasets opnieuw uit."},
    "empty": {"en": "Pick a place for each dataset.",
              "nl": "Kies voor elke dataset een plek."},
    "unfinished": {"en": "Everything you picked so far is right. Pick a place for the rest too ({n} to go).",
                   "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een plek (nog {n})."},
    "wrong": {"en": "{item} goes somewhere else.",
              "nl": "{item} hoort ergens anders."},
    "correct": {"en": "All {n} are where the course's rules put them: open texts, and texts by open models, go into the models the course shares; a corpus whose license covers only the collection, and text whose maker's terms rule out a shared model, are only for you to train on; a pirate source not at all.",
                "nl": "Alle {n} staan waar de regels van de course ze zetten: open teksten, en teksten van open modellen, gaan in de modellen die de course deelt; een corpus waarvan de licentie alleen de verzameling dekt, en tekst waarvan de voorwaarden van de maker een gedeeld model uitsluiten, zijn alleen om zelf op te trainen; een piraterijbron helemaal niet."},
}

EXERCISES = {
    "where": Sort(
        name="where",
        groups={
            "share": {"en": "It may go into the models the course shares", "nl": "Het mag in de modellen die de course deelt"},
            "own": {"en": "Only for you to train on: the course shares no model trained on it", "nl": "Alleen om zelf op te trainen: de course deelt geen model dat erop getraind is"},
            "never": {"en": "Not at all", "nl": "Helemaal niet"},
        },
        items={
            "shakespeare": {"en": "**Tiny Shakespeare.** Plays by Shakespeare, about 400 years old.",
                            "nl": "**Tiny Shakespeare.** Toneelstukken van Shakespeare, zo'n 400 jaar oud."},
            "cosmopedia": {"en": "**Cosmopedia.** Stories written by the model Mixtral, whose license is Apache-2.0; the dataset is Apache-2.0 too.",
                           "nl": "**Cosmopedia.** Verhalen geschreven door het model Mixtral, dat de licentie Apache-2.0 heeft; de dataset is ook Apache-2.0."},
            "tinystories": {"en": "**TinyStories.** Stories written by GPT-3.5 and GPT-4; OpenAI's terms forbid using their output to build models that compete with OpenAI.",
                            "nl": "**TinyStories.** Verhalen geschreven door GPT-3.5 en GPT-4; de voorwaarden van OpenAI verbieden hun output te gebruiken voor modellen die met OpenAI concurreren."},
            "fineweb": {"en": "**FineWeb.** Web pages from Common Crawl, cleaned; its license, ODC-By 1.0, covers the collection, not the pages in it.",
                        "nl": "**FineWeb.** Webpagina's uit Common Crawl, opgeschoond; zijn licentie, ODC-By 1.0, dekt de verzameling, niet de pagina's erin."},
            "wikipedia": {"en": "**Wikipedia.** Articles written by volunteers, under CC BY-SA 4.0.",
                          "nl": "**Wikipedia.** Artikelen geschreven door vrijwilligers, onder CC BY-SA 4.0."},
            "books3": {"en": "**Books3.** Nearly 200,000 books, copied from a pirate site.",
                       "nl": "**Books3.** Bijna 200.000 boeken, gekopieerd van een piraterijsite."},
        },
        solution={"shakespeare": "share", "cosmopedia": "share", "tinystories": "own", "fineweb": "own",
                  "wikipedia": "share", "books3": "never"},
        nudge={
            "shakespeare": {"en": "Does anyone still hold rights in text that old? See rule 1.",
                            "nl": "Heeft nog iemand rechten op tekst die zo oud is? Zie regel 1."},
            "cosmopedia": {"en": "Which terms does text written by a model carry, and what do Mixtral's say? See rule 2.",
                           "nl": "Welke voorwaarden draagt tekst die een model schreef, en wat zeggen die van Mixtral? Zie regel 2."},
            "tinystories": {"en": "Who wrote the text, and what do that maker's terms rule out: learning from it yourself, or a model the course shares? See rule 2.",
                            "nl": "Wie schreef de tekst, en wat sluiten de voorwaarden van die maker uit: er zelf van leren, of een model dat de course deelt? Zie regel 2."},
            "fineweb": {"en": "The collection is open and the pages keep their own rights: does that stop you from learning from them, or the course from sharing a model? See rules 1 and 4, and the watch-out.",
                        "nl": "De verzameling is open en de pagina's houden hun eigen rechten: houdt dat jou tegen om ervan te leren, of de course om een model te delen? Zie regel 1 en 4, en de let-op."},
            "wikipedia": {"en": "Is the text itself under an open license? See rule 1: what does the course do with SA for a model it shares?",
                          "nl": "Staat de tekst zelf onder een open licentie? Zie regel 1: wat doet de course met SA bij een model dat ze deelt?"},
            "books3": {"en": "May you use what was copied without permission at all? See rule 3.",
                       "nl": "Mag je gebruiken wat zonder toestemming gekopieerd is? Zie regel 3."},
        },
        texts=_SORT_TEXTS,
    ),
    "by-license": FunctionBox(
        name="by-license",
        function="by_license",
        start=_B_START,
        solution=_b(),
        visible=[(EXAMPLE,)],
        hidden=[([{"name": "a", "tokens": 1, "license": "MIT"}, {"name": "b", "tokens": 2, "license": "MIT"}, {"name": "c", "tokens": 3, "license": "CC0 1.0"}],),
                ([{"name": "one", "tokens": 7, "license": "Apache-2.0"}],),
                ([{"name": "a", "tokens": 1, "license": "x"}, {"name": "b", "tokens": 1, "license": "y"}, {"name": "c", "tokens": 1, "license": "z"}],),
                ([{"name": "web", "tokens": 2479, "license": "ODC-By 1.0"}, {"name": "code", "tokens": 411, "license": "various"},
                  {"name": "books", "tokens": 6.0, "license": "public domain"}, {"name": "wiki", "tokens": 4.3, "license": "CC BY-SA 4.0"},
                  {"name": "papers", "tokens": 70, "license": "various"}],),
                ([{"name": "a", "tokens": 1, "license": "MIT"}, {"name": "b", "tokens": 1, "license": "MIT"}, {"name": "c", "tokens": 1, "license": "MIT"},
                  {"name": "d", "tokens": 1, "license": "CC0 1.0"}, {"name": "e", "tokens": 1, "license": "CC0 1.0"}, {"name": "f", "tokens": 1, "license": "CC0 1.0"}],)],
        show=_shown,
        compare=_same_shares,
        first=[_DOTS],
        mistakes=[
            Mistake("fraction", wrong=_b(rnd="    for lic in shares:\n        shares[lic] = round(shares[lic] / total, 1)\n    return shares\n"),
                    en=_CALL + ": your shares are fractions of 1, rounded to 1 decimal. The task asks for percent: 100 times the fraction.",
                    nl=_CALL_NL + ": je aandelen zijn delen van 1, afgerond op 1 decimaal. De opdracht vraagt om procenten: 100 keer het deel."),
            Mistake("not-rounded", wrong=_b(rnd="    for lic in shares:\n        shares[lic] = 100 * shares[lic] / total\n    return shares\n"),
                    en=_CALL + ": your shares are not rounded. Round each one to 1 decimal: `round(..., 1)`.",
                    nl=_CALL_NL + ": je aandelen zijn niet afgerond. Rond elk af op 1 decimaal: `round(..., 1)`."),
            Mistake("whole-percent", wrong=_b(rnd="    for lic in shares:\n        shares[lic] = round(100 * shares[lic] / total)\n    return shares\n"),
                    en=_CALL + ": your shares are rounded to whole percents. Round to 1 decimal: `round(..., 1)`.",
                    nl=_CALL_NL + ": je aandelen zijn afgerond op hele procenten. Rond af op 1 decimaal: `round(..., 1)`."),
            Mistake("last-wins", wrong=_b(add="    shares = {}\n    for s in sources:\n        shares[s[\"license\"]] = s[\"tokens\"]\n"),
                    en=_CALL + ": a license that two sources share gets only the tokens of the last one. Add the tokens of a source to what its license already has.",
                    nl=_CALL_NL + ": een licentie die twee bronnen delen, krijgt alleen de tokens van de laatste. Tel de tokens van een bron op bij wat zijn licentie al heeft."),
            Mistake("counts", wrong=_b(add="    shares = {}\n    for s in sources:\n        if s[\"license\"] in shares:\n"
                                           "            shares[s[\"license\"]] = shares[s[\"license\"]] + 1\n        else:\n"
                                           "            shares[s[\"license\"]] = 1\n",
                                       total="    total = len(sources)\n"),
                    en=_CALL + ": your shares count sources, not tokens. A source of 3,500 million tokens weighs more than one of 60: add up `s[\"tokens\"]`.",
                    nl=_CALL_NL + ": je aandelen tellen bronnen, geen tokens. Een bron van 3.500 miljoen tokens weegt zwaarder dan een van 60: tel `s[\"tokens\"]` op."),
            Mistake("round-each", wrong=_b(add="    shares = {}\n    for s in sources:\n        part = round(100 * s[\"tokens\"] / total, 1)\n"
                                                "        if s[\"license\"] in shares:\n            shares[s[\"license\"]] = shares[s[\"license\"]] + part\n"
                                                "        else:\n            shares[s[\"license\"]] = part\n",
                                           rnd="    return shares\n"),
                    en=_CALL + ": you round every source before you add them up, so the roundings add up too. Add the tokens of a license first, and round its share once.",
                    nl=_CALL_NL + ": je rondt elke bron af voordat je ze optelt, dus de afrondingen tellen ook op. Tel eerst de tokens van een licentie op, en rond zijn aandeel één keer af."),
            Mistake("names", wrong=_b(add="    shares = {}\n    for s in sources:\n        shares[s[\"name\"]] = s[\"tokens\"]\n"),
                    en=_CALL + ": your dict has a share per source, by its name. The task asks for one per license: use `s[\"license\"]` as the key.",
                    nl=_CALL_NL + ": je dict heeft een aandeel per bron, op zijn naam. De opdracht vraagt er één per licentie: gebruik `s[\"license\"]` als key."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the total, the tokens of every license, and the shares.",
                      "nl": "Vervang de drie `...`: het totaal, de tokens van elke licentie, en de aandelen."},
            "wrong": {"en": _CALL + ". First all tokens, then per license the tokens of its sources added up, and then each as a share of all, in percent, rounded once?",
                      "nl": _CALL_NL + ". Eerst alle tokens, dan per licentie de tokens van zijn bronnen opgeteld, en dan elk als aandeel van alles, in procenten, één keer afgerond?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `sources`, and no fixed licenses or numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `sources`, en geen vaste licenties of getallen?"},
            "correct": {"en": "Your function says how much of a corpus each license covers. On the example corpus, the web pages under ODC-By are 69.9%: the license of the collection, while the pages themselves keep their own rights.",
                        "nl": "Je functie zegt hoeveel van een corpus elke licentie dekt. In het voorbeeldcorpus zijn de webpagina's onder ODC-By 69,9%: de licentie van de verzameling, terwijl de pagina's zelf hun eigen rechten houden."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner gives, state, known mistake).
CASES = [
    ("where", {}, "empty", None),
    ("where", {"shakespeare": "share", "cosmopedia": "share", "tinystories": "own", "fineweb": "own", "wikipedia": "share", "books3": "never"}, "correct", None),
    ("where", {"shakespeare": "share", "cosmopedia": "share"}, "wrong", "unfinished"),
    ("where", {"shakespeare": "share", "cosmopedia": "share", "tinystories": "share"}, "wrong", "wrong:tinystories"),
    ("where", {"shakespeare": "share", "cosmopedia": "share", "tinystories": "own", "fineweb": "share"}, "wrong", "wrong:fineweb"),
    ("where", {"cosmopedia": "own"}, "wrong", "wrong:cosmopedia"),
    ("where", {"books3": "own"}, "wrong", "wrong:books3"),
    ("where", {"wikipedia": "never"}, "wrong", "wrong:wikipedia"),
    ("by-license", _B_START, "empty", None),
    ("by-license", "", "empty", None),
    ("by-license", _b(), "correct", None),
    # another way: the total with sum, the dict with get
    ("by-license", _B_HEAD + "    total = sum(s[\"tokens\"] for s in sources)\n    shares = {}\n    for s in sources:\n"
     "        shares[s[\"license\"]] = shares.get(s[\"license\"], 0) + s[\"tokens\"]\n"
     "    return {lic: round(100 * t / total, 1) for lic, t in shares.items()}\n", "correct", None),
    # a defaultdict and a Counter are dicts too
    ("by-license", "from collections import defaultdict\n\n\n" + _B_HEAD + "    total = sum(s[\"tokens\"] for s in sources)\n    shares = defaultdict(float)\n"
     "    for s in sources:\n        shares[s[\"license\"]] += s[\"tokens\"]\n" + _B_ROUND, "correct", None),
    ("by-license", "from collections import Counter\n\n\n" + _B_HEAD + "    total = sum(s[\"tokens\"] for s in sources)\n    shares = Counter()\n"
     "    for s in sources:\n        shares[s[\"license\"]] += s[\"tokens\"]\n" + _B_ROUND, "correct", None),
    ("by-license", _b(rnd="    for lic in shares:\n        shares[lic] = shares[lic] / total * 100\n    return shares\n"), "wrong", "not-rounded"),
    ("by-license", _B_START.replace("    total = 0\n    ...\n", "    total = 0\n    for s in sources:\n        total = total + s[\"tokens\"]\n"), "wrong", "dots-left"),
    ("by-license", _b(rnd="    for lic in shares:\n        shares[lic] = round(shares[lic] / total, 1)\n    return shares\n"), "wrong", "fraction"),
    ("by-license", _b(add="    shares = {}\n    for s in sources:\n        shares[s[\"license\"]] = s[\"tokens\"]\n"), "wrong", "last-wins"),
    ("by-license", _b(add="    shares = {}\n    for s in sources:\n        if s[\"license\"] in shares:\n"
                          "            shares[s[\"license\"]] = shares[s[\"license\"]] + 1\n        else:\n"
                          "            shares[s[\"license\"]] = 1\n", total="    total = len(sources)\n"), "wrong", "counts"),
    ("by-license", _b(add="    shares = {}\n    for s in sources:\n        part = round(100 * s[\"tokens\"] / total, 1)\n"
                          "        if s[\"license\"] in shares:\n            shares[s[\"license\"]] = shares[s[\"license\"]] + part\n"
                          "        else:\n            shares[s[\"license\"]] = part\n", rnd="    return shares\n"), "wrong", "round-each"),
    ("by-license", _b(add="    shares = {}\n    for s in sources:\n        shares[s[\"name\"]] = s[\"tokens\"]\n"), "wrong", "names"),
    ("by-license", _b(rnd="    for lic in shares:\n        shares[lic] = 100 * shares[lic] / total\n    return shares\n"), "wrong", "not-rounded"),
    ("by-license", _b(rnd="    for lic in shares:\n        shares[lic] = round(100 * shares[lic] / total)\n    return shares\n"), "wrong", "whole-percent"),
]
