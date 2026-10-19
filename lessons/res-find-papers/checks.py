# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-find-papers: one round of following citations in a small citation graph, backward and
forward. And the picture of the papers around the transformer."""
import html
import re

from everything_ai.check import FunctionBox, Mistake


class _Graph(dict):
    """A citation graph that a check message calls by its name, `cites`, instead of writing out all its papers."""

    def __init__(self, name: str, data: dict):
        super().__init__(data)
        self.name = name

    def __repr__(self):
        return self.name


# A small part of the real citation graph: for each paper, some of the papers of this set that it really cites. Not every
# link is here: some within this set are left out (GPT-2 also cites layernorm, ViT also cites gpt, gpt2 and gpt3, GPT-3
# also cites gpt), and every paper cites papers outside this set too. The worked example of the lesson shows the same
# dictionary.
CITES = _Graph("cites", {
    "lstm": [],
    "dropout": [],
    "adam": [],
    "seq2seq": ["lstm"],
    "align": ["seq2seq"],
    "resnet": [],
    "layernorm": [],
    "transformer": ["align", "seq2seq", "lstm", "adam", "dropout", "resnet", "layernorm"],
    "gpt": ["transformer", "adam", "layernorm"],
    "bert": ["transformer", "gpt"],
    "gpt2": ["transformer", "gpt", "bert"],
    "scaling": ["transformer", "gpt2"],
    "gpt3": ["transformer", "gpt2", "bert", "scaling"],
    "vit": ["transformer", "bert", "resnet"],
})
# The papers behind the keys: (key, authors, year, title).
PAPERS = [
    ("lstm", "Hochreiter and Schmidhuber", 1997, "Long Short-Term Memory"),
    ("dropout", "Srivastava et al.", 2014, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"),
    ("adam", "Kingma and Ba", 2014, "Adam: A Method for Stochastic Optimization"),
    ("seq2seq", "Sutskever et al.", 2014, "Sequence to Sequence Learning with Neural Networks"),
    ("align", "Bahdanau et al.", 2014, "Neural Machine Translation by Jointly Learning to Align and Translate"),
    ("resnet", "He et al.", 2015, "Deep Residual Learning for Image Recognition"),
    ("layernorm", "Ba et al.", 2016, "Layer Normalization"),
    ("transformer", "Vaswani et al.", 2017, "Attention Is All You Need"),
    ("gpt", "Radford et al.", 2018, "Improving Language Understanding by Generative Pre-Training"),
    ("bert", "Devlin et al.", 2018, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    ("gpt2", "Radford et al.", 2019, "Language Models are Unsupervised Multitask Learners"),
    ("scaling", "Kaplan et al.", 2020, "Scaling Laws for Neural Language Models"),
    ("gpt3", "Brown et al.", 2020, "Language Models are Few-Shot Learners"),
    ("vit", "Dosovitskiy et al.", 2020, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"),
]
_YEAR = {key: year for key, _, year, _ in PAPERS}

# Another, made-up graph, so that a function that only knows the papers above is caught.
_SMALL = {"a": [], "b": ["a"], "c": ["a", "b"], "d": ["c"]}


def around(paper: str, labels: dict) -> str:
    """The papers one step from `paper`: those that cite it above, the paper in the middle, its references below.
    Ink only: nothing here is a role of a model. HTML chips, not an SVG, so they wrap in a narrow window."""
    back = CITES[paper]
    forward = [p for p in CITES if paper in CITES[p]]

    def chips(keys):
        return "".join(f'<span style="display:inline-block;margin:0.15rem;padding:0.15rem 0.55rem;border:1.2px solid var(--ea-ink-3);'
                       f'border-radius:999px;background:var(--ea-inset);font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink)">'
                       f'{html.escape(k)} <span style="color:var(--ea-ink-3)">{_YEAR[k]}</span></span>' for k in keys)

    def band(title, keys):
        return (f'<div style="text-align:center"><div style="font-size:0.85rem;color:var(--ea-ink-2);margin-bottom:0.2rem">'
                f'{html.escape(title)}</div><div>{chips(keys)}</div></div>')

    arrow = '<div style="text-align:center;font-size:0.85rem;color:var(--ea-ink-3);margin:0.3rem 0">{}</div>'
    middle = (f'<div style="text-align:center"><span style="display:inline-block;padding:0.3rem 0.8rem;border:2px solid var(--ea-ink);'
              f'border-radius:6px;font-family:var(--ea-font-code);font-weight:600;color:var(--ea-ink)">{html.escape(paper)} '
              f'{_YEAR[paper]}</span></div>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:600px;font-family:var(--ea-font-ui)">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'{band(labels["forward"], forward)}{arrow.format(html.escape(labels["cite"]))}{middle}'
            f'{arrow.format(html.escape(labels["cite"]))}{band(labels["back"], back)}</figure>')


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

def _unsorted(got, text) -> bool:
    """A list that is not in alphabetical order, whatever is in it."""
    try:
        return isinstance(got, list) and got != sorted(got)
    except TypeError:
        return False


def _adds_graph(e) -> bool:
    """`+` with all of `cites` on one side: `back = cites`, or `forward = cites`."""
    s = str(e)
    return isinstance(e, TypeError) and (("for +" in s and ("'dict'" in s or "'_Graph'" in s))
                                         or '(not "dict")' in s or '(not "_Graph")' in s)


_START = ("def one_round(cites, paper):\n"
          "    back = ...\n"
          "    forward = []\n"
          "    ...\n"
          "    return sorted(back + forward)\n")
_SOLUTION = ("def one_round(cites, paper):\n"
             "    back = cites[paper]\n"
             "    forward = []\n"
             "    for other in cites:\n"
             "        if paper in cites[other]:\n"
             "            forward.append(other)\n"
             "    return sorted(back + forward)\n")


def _one_round(old: str, new: str) -> str:
    assert old in _SOLUTION, old
    return _SOLUTION.replace(old, new)


# The lesson is done when these are correct. "direction" is the choice question, "plan-search" the open question.
REQUIRED = ["one-round", "direction", "plan-search"]

EXERCISES = {
    "one-round": FunctionBox(
        name="one-round",
        function="one_round",
        start=_START,
        solution=_SOLUTION,
        visible=[(CITES, "bert")],
        hidden=[(CITES, "transformer"), (CITES, "lstm"), (CITES, "gpt3"), (_SMALL, "b"), (_SMALL, "a")],
        first=[_DOTS],
        form=[Mistake("changes-graph", state="almost",
                      matches=lambda got, text: re.search(r"^\s*back\s*=\s*cites\[\s*paper\s*\]\s*$", text, re.MULTILINE) is not None
                      and re.search(r"^\s*back\.(append|extend)\(", text, re.MULTILINE) is not None,
                      en="The papers are right, but your function changes the graph: `back` is not a new list but the list of `paper` inside `cites` itself, so `back.append` writes the papers that cite it into its references. Call it twice and the second answer is longer. Put those papers in `forward`, a list of its own, as the box started.",
                      nl="De papers kloppen, maar je functie verandert de graaf: `back` is geen nieuwe lijst maar de lijst van `paper` in `cites` zelf, dus `back.append` schrijft de papers die hem citeren bij zijn referenties. Roep hem twee keer aan en het tweede antwoord is langer. Zet die papers in `forward`, een eigen lijst, zoals het vak begon.")],
        mistakes=[
            Mistake("quoted-name", error=lambda e: isinstance(e, KeyError) and e.args == ("paper",),
                    en="`cites[\"paper\"]` looks for a paper called *paper*. Use the name `paper` without quotes: it holds the paper the check asks about, such as `\"bert\"`.",
                    nl="`cites[\"paper\"]` zoekt een paper die *paper* heet. Gebruik de naam `paper` zonder aanhalingstekens: daarin zit de paper waarnaar de check vraagt, zoals `\"bert\"`."),
            Mistake("whole-graph", error=_adds_graph,
                    en="Your `+` adds all of `cites`, a dictionary, to a list, and a dictionary and a list cannot be added with `+`. `back` and `forward` both have to be lists of papers: the references of `paper` you look up in `cites` with `paper`.",
                    nl="Je `+` telt heel `cites`, een dictionary, op bij een lijst, en een dictionary en een lijst kun je niet optellen met `+`. `back` en `forward` moeten allebei lijsten met papers zijn: de referenties van `paper` zoek je in `cites` op met `paper`."),
            Mistake("only-back", wrong=_one_round("    for other in cites:\n        if paper in cites[other]:\n            forward.append(other)\n", ""),
                    en="`{call}` gives `{got}`: only the references of the paper, backward. Forward is missing: go through every paper in `cites`, and keep the ones that have `paper` in their list.",
                    nl="`{call}` geeft `{got}`: alleen de referenties van de paper, achteruit. Vooruit ontbreekt: ga door elke paper in `cites`, en houd de papers die `paper` in hun lijst hebben."),
            Mistake("only-forward", wrong=_one_round("    back = cites[paper]\n", "    back = []\n"),
                    en="`{call}` gives `{got}`: only the papers that cite it, forward. Backward is missing: the references of the paper are its own list in `cites`.",
                    nl="`{call}` geeft `{got}`: alleen de papers die hem citeren, vooruit. Achteruit ontbreekt: de referenties van de paper zijn zijn eigen lijst in `cites`."),
            Mistake("wrong-way", wrong=_one_round("        if paper in cites[other]:\n", "        if other in cites[paper]:\n"),
                    en="`{call}` gives `{got}`: every reference twice, once backward and once more where the papers that cite it should be. Forward asks it the other way round: which papers have `paper` in their own list?",
                    nl="`{call}` geeft `{got}`: elke referentie twee keer, één keer achteruit en nog een keer waar de papers moeten staan die hem citeren. Vooruit vraag je het andersom: welke papers hebben `paper` in hun eigen lijst?"),
            Mistake("adds-itself", wrong=_one_round("            forward.append(other)\n", "            forward.append(paper)\n"),
                    en="`{call}` gives `{got}`, with the paper itself in it, once for every paper that cites it. Put the paper that cites it in `forward`: that is `other`.",
                    nl="`{call}` geeft `{got}`, met de paper zelf erin, één keer voor elke paper die hem citeert. Zet de paper die hem citeert in `forward`: dat is `other`."),
            Mistake("not-sorted", matches=_unsorted,
                    en="`{call}` gives `{got}`: not in alphabetical order. Give the papers back with `sorted(back + forward)`, as the box started.",
                    nl="`{call}` geeft `{got}`: niet op alfabet. Geef de papers terug met `sorted(back + forward)`, zoals het vak begon."),
            Mistake("a-set", matches=lambda got, text: isinstance(got, set),
                    en="`{call}` gives a set, and the task asks for a list in alphabetical order. `sorted(...)` turns any collection into one.",
                    nl="`{call}` geeft een set, en de opdracht vraagt om een lijst op alfabet. `sorted(...)` maakt van elke verzameling zo'n lijst."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the first with the references of `paper`, the second with a loop that puts every paper that cites `paper` in `forward`.",
                      "nl": "Vervang de twee `...`: de eerste door de referenties van `paper`, de tweede door een loop die elke paper die `paper` citeert in `forward` zet."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the graph and a paper, and the `def` line of your function has a different number of names between its parentheses. Keep `(cites, paper)`.",
                          "nl": "De check roept `{call}` aan, met de graaf en een paper, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(cites, paper)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the references of `paper` backward, and forward every paper whose list holds `paper`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je achteruit de referenties van `paper`, en vooruit elke paper waarvan de lijst `paper` bevat?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every paper, and for another graph?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elke paper, en voor een andere graaf?"},
            "correct": {"en": "Your function is right in all {n} cases. For `\"bert\"`: backward the transformer and GPT, forward GPT-2, GPT-3 and ViT. Semantic Scholar and Google Scholar do the forward half for you, over millions of papers.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor `\"bert\"`: achteruit de transformer en GPT, vooruit GPT-2, GPT-3 en ViT. Semantic Scholar en Google Scholar doen de vooruit-helft voor je, over miljoenen papers."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_D = "def one_round(cites, paper):\n"
CASES = [
    ("one-round", _START, "empty", None),
    ("one-round", _SOLUTION, "correct", None),
    ("one-round", _D + "    back = cites[paper]\n    forward = []\n    for other, refs in cites.items():\n        if paper in refs:\n            forward.append(other)\n    return sorted(back + forward)\n", "correct", None),
    ("one-round", _D + "    found = []\n    for other in cites:\n        if other in cites[paper] or paper in cites[other]:\n            found.append(other)\n    return sorted(found)\n", "correct", None),
    ("one-round", _D + "    back = cites[\"paper\"]\n    forward = []\n    return sorted(back + forward)\n", "error", "quoted-name"),
    ("one-round", _D + "    back = cites\n    forward = []\n    return sorted(back + forward)\n", "error", "whole-graph"),
    ("one-round", _D + "    back = cites[paper]\n    forward = []\n    return sorted(back + forward)\n", "wrong", "only-back"),
    ("one-round", _one_round("    back = cites[paper]\n", "    back = []\n"), "wrong", "only-forward"),
    ("one-round", _one_round("        if paper in cites[other]:\n", "        if other in cites[paper]:\n"), "wrong", "wrong-way"),
    ("one-round", _one_round("            forward.append(other)\n", "            forward.append(paper)\n"), "wrong", "adds-itself"),
    ("one-round", _one_round("    return sorted(back + forward)\n", "    return back + forward\n"), "wrong", "not-sorted"),
    ("one-round", _one_round("    return sorted(back + forward)\n", "    return set(back + forward)\n"), "wrong", "a-set"),
    ("one-round", _one_round("    return sorted(back + forward)\n", "    return list(set(back + forward))\n"), "wrong", "not-sorted"),
    ("one-round", _one_round("    return sorted(back + forward)\n", "    return forward + back\n"), "wrong", "not-sorted"),
    ("one-round", _D + "    back = cites[paper]\n    forward = cites[paper]\n    return sorted(back + forward)\n", "wrong", "wrong-way"),
    ("one-round", _D + "    back = cites[paper]\n    forward = cites\n    return sorted(back + forward)\n", "error", "whole-graph"),
    ("one-round", _one_round("            forward.append(other)\n", "            back.append(other)\n"), "almost", "changes-graph"),
    ("one-round", _START.replace("    back = ...\n", "    back = cites[paper]\n"), "wrong", "dots-left"),
    ("one-round", _D + "    return [\"gpt\", \"gpt2\", \"gpt3\", \"transformer\", \"vit\"]\n", "almost", None),
]
