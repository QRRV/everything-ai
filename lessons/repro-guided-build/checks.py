# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-guided-build: dataset, the equations a + b = c modulo p split into training and
validation, saved as repro/data.py; a choice of the fraction of training data; and SETTINGS, the paper's settings,
saved as repro/settings.py. The demo builds the data and the learner's ModernGPT with those settings, and sends one
batch through it."""
import functools
import math
import re

from everything_ai import projects
from everything_ai.check import CodeBox, FunctionBox, Mistake

# The model the reproduction adapts: the learner's ModernGPT of llm-modern-gpt, with the modules it imports; where one
# is missing or broken, the course's version stands in.
BUILDS_ON = (("llm", "modern_parts", "llm-modern-gpt", "parts"), ("llm", "modern_attention", "llm-modern-gpt", "modern-attention"),
             ("llm", "modern_gpt", "llm-modern-gpt", "modern-gpt"))
projects.builds_on(*BUILDS_ON)

# The settings of the paper (Power and others, 2022): p from Appendix A.1.1, the rest from A.1.2. FRACTION is the
# reproduction's own choice: a small part of the table, where the delay shows (FRACTIONS).
PAPER = {"p": 97, "layers": 2, "width": 128, "heads": 4, "lr": 1e-3, "weight_decay": 1.0, "betas": (0.9, 0.98),
         "warmup": 10, "batch": 512, "steps": 100_000}
FRACTION = 0.3
# --- measured: the course's ModernGPT with these settings on addition modulo 97, seed 0, one thread on the course's CPU
# (data/tools/measure_fractions.py): the first step, logged every 100, at which training and validation reach 99% ---
FRACTIONS = {0.9: {"t": 400, "v": 1200}, 0.5: {"t": 300, "v": 2100}, 0.3: {"t": 200, "v": 7900}}
SECONDS_PER_STEP = 0.19
# --- end of measured ---
SOURCE = {"p": "A.1.1", "layers": "A.1.2", "width": "A.1.2", "heads": "A.1.2", "lr": "A.1.2", "weight_decay": "A.1.2",
          "betas": "A.1.2", "warmup": "A.1.2", "batch": "A.1.2", "steps": "A.1.2", "fraction": "exercise 2"}
NON_EMBEDDING = 4e5   # "a total of about 4 · 10^5 non-embedding parameters", A.1.2
NON_EMBEDDING_COURSE = 393_856   # the course's ModernGPT with these settings (the tests count it again)


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".") if lang == "nl" else s


def tokens(rows: list, p: int) -> tuple:
    """The equations as the paper gives them to the model, a token each for a, the operation, b and the sign =: the
    numbers 0 to p - 1 are their own tokens, the operation is token p and = is token p + 1. With the answers c."""
    import torch
    x = torch.tensor([[a, p, b, p + 1] for a, b, _ in rows])
    y = torch.tensor([c for _, _, c in rows])
    return x, y


def non_embedding(model) -> int:
    """The parameters of a ModernGPT outside its table of tokens and its head, as the paper counts them."""
    return sum(t.numel() for name, t in model.named_parameters() if not name.startswith(("tok.", "head.")))


def demo() -> dict | str | None:
    """The learner's data and settings, from their project, and their ModernGPT made with those settings: the sizes,
    the parameters, and one batch through the model before any training. None when a part is not saved yet; "data",
    "settings" or "model" when that part stops with an error. Where a module of the model is missing or broken, the
    course's version stands in, and "stand_ins" names it."""
    folder = projects.folder() / "repro"
    if not (folder / "data.py").is_file() or not (folder / "settings.py").is_file():
        return None
    projects.on_path()
    projects.forget()
    stand = projects.use_course_versions()
    try:
        return _demo(stand)
    finally:
        projects.forget()


def _demo(stand) -> dict | str:
    import importlib
    import torch
    import torch.nn.functional as F
    try:
        s = importlib.import_module("repro.settings").SETTINGS
        if not isinstance(s, dict):
            return "settings"
    except Exception:
        return "settings"
    try:
        train, val = importlib.import_module("repro.data").dataset(s["p"], s["fraction"], 0)
    except Exception:
        return "data"
    try:
        torch.manual_seed(0)
        model = importlib.import_module("llm.modern_gpt").ModernGPT(s["p"] + 2, s["width"], s["heads"], s["heads"], s["layers"])
        x, y = tokens(train[:min(s["batch"], len(train))], s["p"])
        with torch.no_grad():
            logits = model(x)
        loss = F.cross_entropy(logits[:, -1], y).item()
    except Exception:
        return "model"
    return {"train": len(train), "val": len(val), "batch": tuple(x.shape), "logits": tuple(logits.shape),
            "params": non_embedding(model), "loss": loss, "chance": math.log(s["p"] + 2), "settings": s,
            "stand_ins": [f"{package}/{module}.py" for package, module, _, _ in stand]}


def _few(rows) -> str:
    rows = [tuple(r) for r in rows]
    return repr(rows) if len(rows) <= 6 else "[" + ", ".join(repr(r) for r in rows[:3]) + f", … {len(rows):,} in all]"


def _show(v):
    """A split as two lists, the long ones shortened to their first equations and their number."""
    try:
        return f"({_few(v[0])}, {_few(v[1])})"
    except Exception:
        return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each one with your own code.",
                nl="Er staat nog een `...` in je code. Vervang ze allemaal door je eigen code.")
_KEEP = {"missing": {"en": "I cannot see the box. Run the cell with the box again.",
                     "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
         "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                     "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}}


# --- exercise 1: dataset -----------------------------------------------------------------------------------------------

_D_HEAD = "import random\n\n\n"
_D_EQ = ("def equations(p):\n    # every equation a + b = c modulo p, as (a, b, c), for every a and every b from 0 to p - 1\n"
         "    rows = []\n    for a in range(p):\n        for b in range(p):\n            rows.append((a, b, (a + b) % p))\n    return rows\n\n\n")
_D_SPLIT = ("def dataset(p, fraction, seed):\n"
            "    # all equations in an order fixed by the seed: the first int(fraction * their number) for training, the rest for validation\n"
            "    rows = equations(p)\n    random.seed(seed)\n    random.shuffle(rows)\n    n = int(fraction * len(rows))\n"
            "    return rows[:n], rows[n:]\n")
_D = _D_HEAD + _D_EQ + _D_SPLIT
_D_START = (_D_HEAD + "def equations(p):\n    # every equation a + b = c modulo p, as (a, b, c), for every a and every b from 0 to p - 1\n    ...\n\n\n"
            "def dataset(p, fraction, seed):\n"
            "    # all equations in an order fixed by the seed: the first int(fraction * their number) for training, the rest for validation\n"
            "    rows = equations(p)\n    random.seed(seed)\n    ...\n")
# The table in The idea opens at 50%, not at the fraction exercise 2 asks for, with an open cell pointed at whose
# mirror cell b + a is open too: 50 + 10 = 60, with 48 of the 97 cells of its answer filled.
TABLE_FRACTION = 0.5
POINT = (50, 10)


@functools.cache
def _dataset():
    from everything_ai.check import run_code
    return run_code(_D).names["dataset"]


def training(p: int, fraction: float, seed: int) -> list:
    """The training equations of the course's own dataset (the solution of exercise 1), for the table in The idea."""
    return _dataset()(p, fraction, seed)[0]


_D_NO_MOD = _D.replace("(a + b) % p", "a + b")
_D_FROM_1 = _D.replace("for a in range(p):\n        for b in range(p):", "for a in range(1, p):\n        for b in range(1, p):")
_D_HALF = _D.replace("        for b in range(p):\n", "        for b in range(a, p):\n")
_D_NO_SHUFFLE = _D.replace("    random.shuffle(rows)\n", "")
_D_ROUND = _D.replace("n = int(fraction * len(rows))", "n = round(fraction * len(rows))")
_D_SWAP = _D.replace("return rows[:n], rows[n:]", "return rows[n:], rows[:n]")
_D_COLUMNS = _D.replace("for a in range(p):\n        for b in range(p):\n            rows.append((a, b,", "for b in range(p):\n        for a in range(p):\n            rows.append((a, b,")


def _same_split(got, want) -> bool:
    """Two lists of equations, training and validation, in the same order; a tuple or a list for each equation."""
    try:
        return (len(got) == 2 and [tuple(r) for r in got[0]] == [tuple(r) for r in want[0]]
                and [tuple(r) for r in got[1]] == [tuple(r) for r in want[1]])
    except TypeError:
        return False


# --- exercise 3: SETTINGS ----------------------------------------------------------------------------------------------

_S_KEYS = ["p", "layers", "width", "heads", "lr", "weight_decay", "betas", "warmup", "batch", "steps", "fraction"]
_S = ("# The settings of the reproduction: the paper's, each from where its comment says, and the fraction of exercise 2.\n"
      "SETTINGS = {\n"
      "    \"p\": 97,                # A.1.1\n    \"layers\": 2,            # A.1.2\n    \"width\": 128,\n    \"heads\": 4,\n"
      "    \"lr\": 1e-3,\n    \"weight_decay\": 1.0,\n    \"betas\": (0.9, 0.98),\n    \"warmup\": 10,\n    \"batch\": 512,\n"
      "    \"steps\": 100_000,\n    \"fraction\": " + repr(FRACTION) + ",     # exercise 2\n}\nSETTINGS\n")
_S_START = ("# The settings of the reproduction: the paper's, each from where its comment says, and the fraction of exercise 2.\n"
            "SETTINGS = {\n"
            "    \"p\": 97,                # A.1.1\n    \"layers\": ...,          # A.1.2\n    \"width\": ...,\n    \"heads\": ...,\n"
            "    \"lr\": ...,\n    \"weight_decay\": ...,\n    \"betas\": ...,\n    \"warmup\": ...,\n    \"batch\": ...,\n"
            "    \"steps\": ...,\n    \"fraction\": ...,      # exercise 2\n}\nSETTINGS\n")
_WANT = {**PAPER, "fraction": FRACTION}


def _equal(got, want) -> bool:
    if isinstance(want, tuple):
        return isinstance(got, (tuple, list)) and len(got) == len(want) and all(_equal(g, w) for g, w in zip(got, want))
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        return False
    return math.isclose(got, want, rel_tol=1e-9)


def _off(got) -> list:
    """The keys of SETTINGS whose value is not the paper's (or not the fraction of exercise 2), in the order of the box."""
    return [k for k in _S_KEYS if not (k in got and _equal(got[k], _WANT[k]))]


# counts: range() and PyTorch's layers take only whole numbers, so 4.0 heads or 1e5 steps stop the model and the runs
_COUNTS = [k for k in _S_KEYS if isinstance(_WANT[k], int)]


def _not_whole(got) -> list:
    """The counts of SETTINGS with the paper's number written as a float, such as 1e5 for the steps."""
    return [k for k in _COUNTS if k in got and isinstance(got[k], float) and _equal(got[k], _WANT[k])]


def _missing(got) -> list:
    return [k for k in _S_KEYS if k not in got]


def _quoted(got) -> list:
    """The settings written between quotes: text, which Python does not read as a number."""
    return [k for k in _S_KEYS if isinstance(got.get(k), str)]


def _fraction_number(got) -> bool:
    f = got.get("fraction")
    return isinstance(f, (int, float)) and not isinstance(f, bool) and 0 <= f <= 1


def _settings_ok(got, want) -> bool:
    return isinstance(got, dict) and not _off(got) and not _not_whole(got)


_QUOTE = {"layers": "2 layers", "width": "width 128", "heads": "4 attention heads", "lr": "learning rate 10⁻³",
          "weight_decay": "weight decay 1", "betas": "β1 = 0.9, β2 = 0.98", "warmup": "warmup over the first 10 updates",
          "batch": "minibatch size 512 or half of training dataset size (whichever was smaller)",
          "steps": "optimization budget of 10⁵ gradient updates", "p": "for a prime number p = 97"}


def _first_off(got, text) -> dict:
    k = _off(got)[0]
    return {"key": k, "value": repr(got[k]), "quote": _QUOTE.get(k, ""), "where": SOURCE[k]}


def _first_float(got, text) -> dict:
    k = _not_whole(got)[0]
    return {"key": k, "value": repr(got[k]), "whole": f"{_WANT[k]:_}"}


EXERCISES = {
    "dataset": FunctionBox(
        name="dataset",
        function="dataset",
        start=_D_START,
        solution=_D,
        visible=[(3, 0.5, 0)],
        hidden=[(5, 0.3, 1), (7, 0.5, 2), (2, 1.0, 0), (4, 0.0, 3), (97, 0.3, 0)],
        compare=_same_split,
        show=_show,
        save_as=("repro", "data"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-mod", wrong=_D_NO_MOD,
                    en="`{call}` gives an equation with a `c` of `p` or more: the sum `a + b` is not taken modulo `p`. On a clock of `p` hours, `c` is `(a + b) % p`, with the brackets: without them, `%` goes before `+`.",
                    nl="`{call}` geeft een vergelijking met een `c` van `p` of meer: de som `a + b` wordt niet modulo `p` genomen. Op een klok van `p` uur is `c` gelijk aan `(a + b) % p`, met de haakjes: zonder gaat `%` voor `+`."),
            Mistake("from-1", wrong=_D_FROM_1,
                    en="`{call}` misses the equations with 0: `range(1, p)` starts at 1. The paper's numbers are 0 to `p` - 1: `range(p)`.",
                    nl="`{call}` mist de vergelijkingen met 0: `range(1, p)` begint bij 1. De getallen van de paper zijn 0 tot en met `p` - 1: `range(p)`."),
            Mistake("half", wrong=_D_HALF,
                    en="`{call}` makes only the equations with `b` at least `a`. `1 + 2` and `2 + 1` are two equations of the table, each with its own tokens: every `b` for every `a`.",
                    nl="`{call}` maakt alleen de vergelijkingen met `b` minstens `a`. `1 + 2` en `2 + 1` zijn twee vergelijkingen van de tabel, elk met zijn eigen tokens: elke `b` voor elke `a`."),
            Mistake("no-shuffle", wrong=_D_NO_SHUFFLE,
                    en="`{call}` splits the equations in the order `equations` makes them, so the training data holds only the smallest values of `a`. Shuffle them first, after `random.seed(seed)`: `random.shuffle(rows)`.",
                    nl="`{call}` splitst de vergelijkingen in de volgorde waarin `equations` ze maakt, dus de training data bevat alleen de kleinste waarden van `a`. Schud ze eerst, na `random.seed(seed)`: `random.shuffle(rows)`."),
            Mistake("round", wrong=_D_ROUND,
                    en="`{call}` puts one equation too many in the training data: `round` rounds 7.5 up to 8. The plan is the first `int(fraction * len(rows))`, which drops the part after the point.",
                    nl="`{call}` zet één vergelijking te veel in de training data: `round` rondt 7,5 af naar 8. Het plan is de eerste `int(fraction * len(rows))`, die het deel na de komma weglaat."),
            Mistake("columns", wrong=_D_COLUMNS,
                    en="`{call}` has the right equations, but `equations` makes them in another order: (0, 0, 0), (1, 0, 1), (2, 0, 2), …, column by column. The shuffle starts from that order, so the same seed gives another split. The table goes row by row: `a` in the outer loop, `b` inside it, and each as `(a, b, c)`.",
                    nl="`{call}` heeft de goede vergelijkingen, maar `equations` maakt ze in een andere volgorde: (0, 0, 0), (1, 0, 1), (2, 0, 2), …, kolom voor kolom. Het schudden begint bij die volgorde, dus dezelfde seed geeft een andere split. De tabel gaat rij voor rij: `a` in de buitenste loop, `b` erbinnen, en elk als `(a, b, c)`."),
            Mistake("shuffle-gives-none", syntax=r"^[ \t]*\w+[ \t]*=[ \t]*random\.shuffle\(",
                    en="On the line `{line}`, `random.shuffle` puts the list itself in another order and gives back nothing, `None`: so the name on the left is `None` after it. Write `random.shuffle(rows)` on a line of its own, and go on with `rows`.",
                    nl="Op de regel `{line}` zet `random.shuffle` de lijst zelf in een andere volgorde en geeft niets terug, `None`: dus de naam links is daarna `None`. Schrijf `random.shuffle(rows)` op een eigen regel, en ga door met `rows`."),
            Mistake("swapped", wrong=_D_SWAP,
                    en="`{call}` gives validation first: the training data comes first, `rows[:n]`, then the rest.",
                    nl="`{call}` geeft de validatie eerst: de training data komt eerst, `rows[:n]`, en dan de rest."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace each `...`: in `equations`, every `(a, b, (a + b) % p)`; in `dataset`, the shuffle and the split.",
                      "nl": "Vervang elke `...`: in `equations` elke `(a, b, (a + b) % p)`, in `dataset` het schudden en het splitsen."},
            "arguments": {"en": "The check calls `{call}`, with `p`, the fraction and the seed, and the `def` line of your `dataset` has another number of names between its parentheses. Keep `(p, fraction, seed)`.",
                          "nl": "De check roept `{call}` aan, met `p`, de fraction en de seed, en de `def`-regel van je `dataset` heeft een ander aantal namen tussen de haakjes. Houd `(p, fraction, seed)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Every `(a, b, (a + b) % p)` in the order of the table, shuffled with `random.shuffle` after `random.seed(seed)`, and the first `int(fraction * len(rows))` for training?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke `(a, b, (a + b) % p)` in de volgorde van de tabel, geschud met `random.shuffle` na `random.seed(seed)`, en de eerste `int(fraction * len(rows))` voor de training?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Other values of `p`, a fraction of 0 or 1, and another seed?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Andere waarden van `p`, een fraction van 0 of 1, en een andere seed?"},
            "correct": {"en": "Your data is that of the paper: every equation of the table, each a training or a validation equation, in an order the seed fixes. It is part of your reproduction kit now, as `{saved}`.",
                        "nl": "Je data is die van de paper: elke vergelijking van de tabel, elk een trainings- of een validatievergelijking, in een volgorde die de seed vastlegt. Hij is nu een deel van je reproductiekit, als `{saved}`."},
            "kept": {"en": "Your `dataset` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `dataset` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "settings": CodeBox(
        name="settings",
        start=_S_START,
        solution=_S,
        compare=_settings_ok,
        save_as=("repro", "settings"),
        first=[_DOTS],
        mistakes=[
            Mistake("not-a-dictionary", matches=lambda got, text: not isinstance(got, dict),
                    en="Your last line does not give a dictionary. Keep the curly brackets of `SETTINGS`, and `SETTINGS` on the last line.",
                    nl="Je laatste regel geeft geen dictionary. Houd de accolades van `SETTINGS`, en `SETTINGS` op de laatste regel."),
            Mistake("name-missing", matches=lambda got, text: isinstance(got, dict) and bool(_missing(got)),
                    detail=lambda got, text: {"key": _missing(got)[0]},
                    en="`SETTINGS` has no `\"{key}\"`. Keep every name of the box as it stands, in quotes and spelled the same, with its value after the colon.",
                    nl="`SETTINGS` heeft geen `\"{key}\"`. Houd elke naam van het vak zoals hij er staat, tussen aanhalingstekens en op dezelfde manier geschreven, met zijn waarde na de dubbele punt."),
            Mistake("quoted", matches=lambda got, text: isinstance(got, dict) and bool(_quoted(got)),
                    detail=lambda got, text: {"key": _quoted(got)[0], "value": repr(got[_quoted(got)[0]])},
                    en="`\"{key}\"` is `{value}`: text between quotes, which Python does not read as a number. Write the value itself, without the quotes.",
                    nl="`\"{key}\"` is `{value}`: tekst tussen aanhalingstekens, die Python niet als getal leest. Schrijf de waarde zelf, zonder de aanhalingstekens."),
            Mistake("fraction-off", matches=lambda got, text: isinstance(got, dict) and _off(got) == ["fraction"] and not _fraction_number(got),
                    en="Every setting of the paper is right. `fraction` is the choice of exercise 2: write it as a number from 0 to 1.",
                    nl="Elke instelling van de paper klopt. `fraction` is de keuze van oefening 2: schrijf hem als getal van 0 tot 1."),
            Mistake("fraction-other", matches=lambda got, text: isinstance(got, dict) and _off(got) == ["fraction"],
                    detail=lambda got, text: {"value": repr(got["fraction"])},
                    en="Every setting of the paper is right, but `fraction` is `{value}`, and that is not the fraction of exercise 2. Which fraction did the course's runs show the delay of your plan at?",
                    nl="Elke instelling van de paper klopt, maar `fraction` is `{value}`, en dat is niet de fraction van oefening 2. Bij welke fraction lieten de runs van de course de vertraging van je plan zien?"),
            Mistake("setting-off", matches=lambda got, text: isinstance(got, dict) and bool(_off(got)), detail=_first_off,
                    en="`\"{key}\"` is `{value}`, and that is not what the paper says. Look in {where}: *{quote}*.",
                    nl="`\"{key}\"` is `{value}`, en dat is niet wat de paper zegt. Kijk in {where}: *{quote}*."),
            Mistake("not-whole", matches=lambda got, text: isinstance(got, dict) and bool(_not_whole(got)), detail=_first_float,
                    en="`\"{key}\"` is `{value}`: the paper's number, but as a float. It is a count, and `range` and the layers of PyTorch take only whole numbers: write `{whole}`.",
                    nl="`\"{key}\"` is `{value}`: het getal van de paper, maar als float. Het is een aantal, en `range` en de layers van PyTorch nemen alleen gehele getallen: schrijf `{whole}`."),
        ],
        texts={
            **_KEEP,
            "empty": {"en": "Replace each `...` with the paper's value: Appendix A.1.2 has them all, and the fraction is your answer to exercise 2.",
                      "nl": "Vervang elke `...` door de waarde van de paper: Appendix A.1.2 heeft ze allemaal, en de fraction is je antwoord op oefening 2."},
            "no_value": {"en": "The last line of the box shows nothing. Put `SETTINGS` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `SETTINGS` op de laatste regel, zodat de check hem kan zien."},
            "wrong": {"en": "Your last line gives `{got}`, and it should be the dictionary of the eleven settings.",
                      "nl": "Je laatste regel geeft `{got}`, en dat moet de dictionary van de elf instellingen zijn."},
            "correct": {"en": "Every setting is the paper's, with its place in the paper next to it, and the fraction is your choice. They are part of your reproduction kit now, as `{saved}`: the later steps take them from there.",
                        "nl": "Elke instelling is die van de paper, met zijn plek in de paper ernaast, en de fraction is je keuze. Ze zijn nu een deel van je reproductiekit, als `{saved}`: de volgende stappen halen ze daar."},
            "kept": {"en": "Your settings are right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je instellingen zijn goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

REQUIRED = ["dataset", "which-fraction", "settings"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("dataset", _D_START, "empty", None),
    ("dataset", _D, "correct", None),
    ("dataset", _D_HEAD + "def equations(p):\n    return [(a, b, (a + b) % p) for a in range(p) for b in range(p)]\n\n\n" + _D_SPLIT, "correct", None),
    ("dataset", _D_HEAD + _D_EQ + _D_SPLIT.replace("    random.seed(seed)\n    random.shuffle(rows)\n", "    random.Random(seed).shuffle(rows)\n"), "correct", None),
    ("dataset", _D_NO_MOD, "wrong", "no-mod"),
    ("dataset", _D_FROM_1, "wrong", "from-1"),
    ("dataset", _D_HALF, "wrong", "half"),
    ("dataset", _D_NO_SHUFFLE, "wrong", "no-shuffle"),
    ("dataset", _D_ROUND, "wrong", "round"),
    ("dataset", _D_SWAP, "wrong", "swapped"),
    ("settings", _S_START, "empty", None),
    ("settings", _S, "correct", None),
    ("settings", _S.replace("\"lr\": 1e-3,", "\"lr\": 0.001,").replace("\"betas\": (0.9, 0.98),", "\"betas\": [0.9, 0.98],"), "correct", None),
    ("settings", _S.replace("\"lr\": 1e-3,", "\"lr\": 1e-4,"), "wrong", "setting-off"),
    ("settings", _S.replace("\"batch\": 512,", "\"batch\": 256,"), "wrong", "setting-off"),
    ("settings", _S.replace("\"fraction\": " + repr(FRACTION), "\"fraction\": 30"), "wrong", "fraction-off"),
    ("settings", "SETTINGS = 3\nSETTINGS\n", "wrong", "not-a-dictionary"),
    ("settings", _S.replace("\"steps\": 100_000,", "\"steps\": 10**5,").replace("\"weight_decay\": 1.0,", "\"weight_decay\": 1,"), "correct", None),
    ("settings", _S.replace("\"steps\": 100_000,", "\"steps\": 1e5,"), "wrong", "not-whole"),
    ("settings", _S.replace("\"heads\": 4,", "\"heads\": 4.0,"), "wrong", "not-whole"),
    ("settings", _S.replace("    \"warmup\": 10,\n", ""), "wrong", "name-missing"),
    ("settings", _S.replace("\"lr\": 1e-3,", "\"lr\": \"1e-3\","), "wrong", "quoted"),
    ("settings", _S.replace("\"fraction\": " + repr(FRACTION), "\"fraction\": 0.5"), "wrong", "fraction-other"),
    ("dataset", _D.replace("(a + b) % p", "a + b % p"), "wrong", "no-mod"),
    ("dataset", _D_COLUMNS, "wrong", "columns"),
    ("dataset", _D.replace("rows.append((a, b, (a + b) % p))", "rows.append((b, a, (a + b) % p))"), "wrong", "columns"),
    ("dataset", _D.replace("    random.shuffle(rows)\n", "    rows = random.shuffle(rows)\n"), "error", "shuffle-gives-none"),
    ("dataset", _D.replace("    return rows\n", ""), "error", None),   # equations without return: Python's own None message
]
