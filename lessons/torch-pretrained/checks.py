# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson torch-pretrained: load_reader, which loads a state_dict from a file into a new DigitCNN, the
course's digit reader, and puts it in eval mode. The course's digit reader is on Hugging Face, with its model card
(everything_ai.hub fetches both, or uses the course's copies in data/ when there is no connection); the picture shows
what is in its repository and how it gets into a model of your own."""
import ast
import functools
import html
import re
import tempfile
from pathlib import Path

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import hub, paths
from everything_ai.check import FunctionBox, Mistake

REPO = "QRRV/everything-ai-qmnist-cnn"   # the course's digit reader on Hugging Face
DATA = "data/qmnist-small.npz"           # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
RIGHT = 929                               # test digits the digit reader reads right, as its model card says


def hub_file(filename: str) -> tuple[Path, str]:
    """A file of the digit reader's repository: its path, and "hub" or "course" for where it came from."""
    return hub.fetch(REPO, filename)


def page() -> str:
    return hub.page(REPO)


def address(filename: str) -> str:
    return hub.address(REPO, filename)


def card() -> str:
    """The model card without its metadata block between the two `---` lines, which the hub shows as tags, and with
    its headings as bold lines: in the lesson, they are part of the worked example, not headings of the lesson."""
    text = hub_file("README.md")[0].read_text(encoding="utf-8")
    text = text.split("---", 2)[2].strip() if text.startswith("---") else text
    lines, fenced = [], False
    for line in text.splitlines():
        fenced = fenced != line.startswith("```")
        heading = None if fenced else re.match(r"#{1,6} (.+)", line)
        lines.append(f"**{heading.group(1)}**" if heading else line)
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def qmnist_test():
    """The 1,000 test images of QMNIST as a tensor of shape (1000, 784) with pixels from 0 to 1, and their digits."""
    d = np.load(paths.ROOT / DATA)
    return (torch.tensor(d["test_images"], dtype=torch.float32) / 255, torch.tensor(d["test_labels"], dtype=torch.int64))


def hub_picture(labels: dict) -> str:
    """The repository on the hub with its two files, and the way from model.pt into a model of your own: torch.load,
    a new DigitCNN, load_state_dict. Ink only: these are files and steps, not parts of a model's computation."""
    W, LEFT = 330, 10
    mono = "font-family:var(--ea-font-code)"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    box = "fill:var(--ea-ink);fill-opacity:0.04;stroke:var(--ea-ink-3);stroke-width:1.2"
    # the repository, with its two files
    parts.append(f'<rect x="1" y="28" width="{W - 2}" height="132" rx="8" style="{box}"/>'
                 f'<text x="{LEFT + 4}" y="50" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(labels["hub"])}</text>'
                 f'<text x="{LEFT + 4}" y="68" style="{mono};font-size:13px;fill:var(--ea-ink)">{html.escape(REPO)}</text>')
    for k, (name, note) in enumerate([("README.md", labels["card"]), ("model.pt", labels["weights"])]):
        y = 82 + k * 38
        parts.append(f'<rect x="{LEFT + 4}" y="{y}" width="{W - 2 * LEFT - 10}" height="30" rx="5" '
                     f'style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"/>'
                     f'<text x="{LEFT + 14}" y="{y + 20}" style="{mono};font-size:13px;fill:var(--ea-ink)">{name}</text>'
                     f'<text x="{LEFT + 104}" y="{y + 20}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(note)}</text>')
    # the steps from model.pt to a model that reads digits
    arrow = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    y = 160
    for code, note in labels["steps"]:
        parts.append(f'<path d="M40 {y + 4} V{y + 26} M35 {y + 20} L40 {y + 26} L45 {y + 20}" {arrow}/>'
                     f'<text x="56" y="{y + 20}" style="font-size:12px;fill:var(--ea-ink-2);font-style:italic">{html.escape(note)}</text>')
        y += 30
        parts.append(f'<rect x="1" y="{y}" width="{W - 2}" height="30" rx="5" style="{box}"/>'
                     f'<text x="{LEFT + 4}" y="{y + 20}" style="{mono};font-size:12.5px;fill:var(--ea-ink)">{html.escape(code)}</text>')
        y += 30
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {y + 4}" role="img" aria-label="{html.escape(labels["title"] + ". " + labels["alt"])}" '
            f'style="width:100%;height:auto;font-family:var(--ea-font-ui)"><title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>'
            + "".join(parts) + "</svg></figure>")


HUB_REPOS = [REPO]   # the lesson downloads from these repositories on the hub: the header says so

# The lesson is done when these are correct. "pick-model" and "safe-load" are the choice questions in the lesson.
REQUIRED = ["pick-model", "load-reader", "safe-load"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
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


class _Path:
    """A file of a test case, written in the check's messages by a short name; the function gets the real path."""

    def __init__(self, shown: str, real: str):
        self.shown, self.real = shown, real

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        torch.manual_seed(0)   # the check copies every argument just before the call: a new DigitCNN then starts the same
        return self.real


@functools.lru_cache(maxsize=1)
def _other_files() -> tuple[str, str]:
    """Two more state_dicts for DigitCNN, in a folder of their own: the course's with its 10 output rows moved one
    place, so that it reads every digit as the next one, and that of a new model with seed 1."""
    from everything_ai.models import DigitCNN
    folder = Path(tempfile.mkdtemp(prefix="eai-torch-pretrained-"))
    state = torch.load(paths.ROOT / "data" / "course-qmnist-cnn.pt", weights_only=True)
    state["out.weight"], state["out.bias"] = state["out.weight"].roll(1, 0), state["out.bias"].roll(1, 0)
    torch.save(state, folder / "next.pt")
    torch.manual_seed(1)
    torch.save(DigitCNN().state_dict(), folder / "new.pt")
    return str(folder / "next.pt"), str(folder / "new.pt")


def _summary(model):
    """What the check compares: the class of what comes back, whether it is in eval mode, and the digits it reads for
    the first 8 test images; for a dictionary, a tensor or a tuple, what it is; anything else, as it is."""
    if isinstance(model, dict):
        return ["dict", len(model)]
    if isinstance(model, torch.Tensor):
        return ["tensor", list(model.shape)]
    if isinstance(model, tuple) and type(model).__repr__ is tuple.__repr__:
        return ["tuple", [_summary(v) for v in model]]
    if not isinstance(model, torch.nn.Module):
        return model
    X, _ = qmnist_test()
    try:
        with torch.no_grad():
            digits = model(X[:8]).argmax(dim=1).tolist()
    except Exception:
        digits = None
    return [type(model).__name__, "eval" if not model.training else "train", digits]


def _shown(v):
    if isinstance(v, list) and v[:1] == ["dict"]:
        return {"en": f"a dictionary of {v[1]} tensors", "nl": f"een dictionary van {v[1]} tensors"}
    if isinstance(v, list) and v[:1] == ["tensor"]:
        return {"en": f"a tensor of shape {tuple(v[1])}", "nl": f"een tensor met shape {tuple(v[1])}"}
    if isinstance(v, list) and v[:1] == ["tuple"]:
        parts = [_shown(x) for x in v[1]]
        return {lang: ("a tuple: " if lang == "en" else "een tuple: ")
                + "; ".join(p[lang] if isinstance(p, dict) else repr(p) for p in parts) for lang in ("en", "nl")}
    if isinstance(v, list) and len(v) == 3 and v[1] in ("eval", "train"):
        digits = " ".join(str(d) for d in v[2]) if v[2] is not None else "?"
        mode = {"eval": ("in eval mode", "in eval mode"), "train": ("in train mode", "in train mode")}[v[1]]
        return {"en": f"a {v[0]} {mode[0]} that reads {digits}", "nl": f"een {v[0]} {mode[1]} dat {digits} leest"}
    return v


_L_HEAD = "import torch\nfrom everything_ai.models import DigitCNN\n\n\ndef load_reader(path):\n"
_L_BODY = ("    state = torch.load(path, weights_only=True)\n    model = DigitCNN()\n    model.load_state_dict(state)\n"
           "    model.eval()\n    return model\n")
_L_START = (_L_HEAD + "    # 3. the state_dict from the file, without code; a new DigitCNN; the numbers into it; eval mode\n"
            "    state = ...\n    model = ...\n    ...\n    ...\n    return model\n")


def _l(body: str = _L_BODY) -> str:
    return _L_HEAD + body


_L_WRONG = {
    "not-loaded": _l(_L_BODY.replace("    model.load_state_dict(state)\n", "")),
    "no-eval": _l(_L_BODY.replace("    model.eval()\n", "")),
    "state-back": _l(_L_BODY.replace("    return model\n", "    return state\n")),
}
def _without_weights_only(text: str) -> bool:
    """A call `torch.load(...)` in the learner's code without `weights_only=True`; comments do not count."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
            if not any(k.arg == "weights_only" and isinstance(k.value, ast.Constant) and k.value.value is True
                       for k in node.keywords):
                return True
    return False


_NO_WEIGHTS_ONLY = Mistake(
    "weights-only", matches=lambda got, text: _without_weights_only(text),
    state="almost",
    en="Your model is right. But write `torch.load(path, weights_only=True)`: since PyTorch 2.6 that is the default, but written out it holds whatever the default is, and anyone who reads your code sees that it loads only tensors.",
    nl="Je model klopt. Maar schrijf `torch.load(path, weights_only=True)`: sinds PyTorch 2.6 is dat de standaard, maar als het er staat, geldt het wat de standaard ook is, en ziet iedereen die je code leest dat die alleen tensors laadt.")

EXERCISES = {
    "load-reader": FunctionBox(
        name="load-reader",
        function="load_reader",
        start=_L_START,
        solution=_l(),
        visible=[(_Path("path", str(paths.ROOT / "data" / "course-qmnist-cnn.pt")),)],
        hidden=[],
        collect=_summary,
        show=_shown,
        first=[_DOTS],
        form=[_NO_WEIGHTS_ONLY],
        mistakes=[
            Mistake("other-model", error=lambda e: isinstance(e, RuntimeError) and "state_dict" in str(e),
                    en="`load_state_dict` stops with an error: the names or shapes in the file are not those of your model. The file holds the state_dict of a `DigitCNN`: load it into a new `DigitCNN()`, the class on the card.",
                    nl="`load_state_dict` stopt met een fout: de namen of shapes in het bestand zijn niet die van je model. Het bestand bevat de state_dict van een `DigitCNN`: laad het in een nieuw `DigitCNN()`, de class op de card."),
            Mistake("not-loaded", wrong=_L_WRONG["not-loaded"],
                    en=_CALL + ": a new DigitCNN that reads digits with its random starting values. Copy the numbers from the file into it: `model.load_state_dict(state)`.",
                    nl=_CALL_NL + ": een nieuw DigitCNN dat cijfers leest met zijn willekeurige beginwaarden. Kopieer de getallen uit het bestand erin: `model.load_state_dict(state)`."),
            Mistake("no-eval", wrong=_L_WRONG["no-eval"],
                    en=_CALL + ": the right numbers, but the model is still in train mode. This model reads the same either way, but one with dropout would not: put it in eval mode with `model.eval()` before you use it.",
                    nl=_CALL_NL + ": de goede getallen, maar het model staat nog in train mode. Dit model leest in beide modi hetzelfde, maar een model met dropout niet: zet het in eval mode met `model.eval()` voordat je het gebruikt."),
            Mistake("state-back", wrong=_L_WRONG["state-back"],
                    en=_CALL + ": you give back the state_dict, the numbers from the file. Give back the model you loaded them into: `return model`.",
                    nl=_CALL_NL + ": je geeft de state_dict terug, de getallen uit het bestand. Geef het model terug waar je ze in geladen hebt: `return model`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: the state_dict from `torch.load`, a new `DigitCNN()`, the state_dict loaded into it, and eval mode.",
                      "nl": "Vervang de `...`: de state_dict uit `torch.load`, een nieuw `DigitCNN()`, de state_dict erin geladen, en eval mode."},
            "arguments": {"en": "The check calls `{call}`, with the path of a file, and the `def` line of your function has another number of names between its parentheses. Keep `(path)`.",
                          "nl": "De check roept `{call}` aan, met het pad van een bestand, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path)`."},
            "wrong": {"en": _CALL + ": the check shows what comes back, its mode, and the digits it reads for the first 8 test images. Step 3 of the recipe, in order?",
                      "nl": _CALL_NL + ": de check laat zien wat er terugkomt, zijn modus, en de cijfers die het leest voor de eerste 8 testafbeeldingen. Stap 3 van het recept, op volgorde?"},
            "almost": {"en": "Right for the course's file. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function load the file it gets, `path`, and not always the same one?",
                       "nl": "Goed voor het bestand van de course. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Laadt je functie het bestand dat hij krijgt, `path`, en niet steeds hetzelfde?"},
            "correct": {"en": "Your function loads a state_dict into a new DigitCNN in eval mode, for every file the check tried. With the course's file, it reads 929 of the 1,000 test digits right, as the model card says.",
                        "nl": "Je functie laadt een state_dict in een nieuw DigitCNN in eval mode, voor elk bestand dat de check probeerde. Met het bestand van de course leest het 929 van de 1.000 testcijfers goed, zoals de model card zegt."},
        },
    ),
}


def _hidden_paths():
    """The hidden cases get their real paths when the check first needs them: two files that this module writes."""
    other, third = _other_files()
    box = EXERCISES["load-reader"]
    box.hidden = [(_Path("other_path", other),), (_Path("third_path", third),)]


_hidden_paths()

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("load-reader", _L_START, "empty", None),
    ("load-reader", _l(), "correct", None),
    ("load-reader", _l("    model = DigitCNN()\n    model.load_state_dict(torch.load(path, weights_only=True))\n"
                       "    return model.eval()\n"), "correct", None),
    ("load-reader", _l(_L_BODY.replace("torch.load(path, weights_only=True)", "torch.load(path)")), "almost", "weights-only"),
    ("load-reader", _l(_L_BODY.replace("torch.load(path, weights_only=True)",
                                       "torch.load(\"" + str(paths.ROOT / "data" / "course-qmnist-cnn.pt") + "\", weights_only=True)")),
     "almost", None),
    ("load-reader", _l(_L_BODY.replace("torch.load(path, weights_only=True)", "torch.load(str(path), weights_only=True)")
                       .replace("    model = DigitCNN()\n", "    # torch.load(path) alone would read more than tensors\n    model = DigitCNN()\n")),
     "correct", None),
    ("load-reader", _l(_L_BODY.replace("weights_only=True", "weights_only=False")), "almost", "weights-only"),
    ("load-reader", _l(_L_BODY.replace("    model = DigitCNN()\n", "    model = torch.nn.Linear(784, 10)\n")), "error", "other-model"),
    ("load-reader", _l(_L_BODY.replace("    return model\n", "    return model, state\n")), "wrong", None),
] + [("load-reader", code, "wrong", key) for key, code in _L_WRONG.items()]
