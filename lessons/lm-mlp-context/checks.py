# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-mlp-context: contexts, the training examples of a context window of k tokens; and
MLPLanguageModel, Bengio's neural language model: k embeddings side by side, a hidden layer, and an output head.
The picture trains it with the Trainer of nn-trainer on Tiny Shakespeare, for 1, 3 and 8 characters of context."""
import functools
import importlib.util
import re

import torch   # imported once, when the lesson opens: the first import takes longer than a check may
from torch.utils.data import DataLoader, TensorDataset

from everything_ai import paths, projects
from everything_ai.check import ClassBox, FunctionBox, Mistake, same_value

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
DIM, HIDDEN, LR = 10, 128, 0.003
N_TRAIN, N_VAL = 128_000, 10_000     # examples drawn from the first 90 % of the text, and from the last 10 %
KS = (1, 3, 8)


@functools.lru_cache(maxsize=1)
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def chars() -> tuple[str, ...]:
    """The 65 characters of Tiny Shakespeare in a fixed order: character chars()[i] has id i."""
    return tuple(sorted(set(text())))


@functools.lru_cache(maxsize=1)
def ids() -> torch.Tensor:
    index = {c: i for i, c in enumerate(chars())}
    return torch.tensor([index[c] for c in text()])


def encode(s: str) -> list[int]:
    """The ids of the characters of s."""
    index = {c: i for i, c in enumerate(chars())}
    return [index[c] for c in s]


# --- exercise 2: contexts -------------------------------------------------------------------------------------------------

_X_START = ("def contexts(ids, k):\n    X = []\n    Y = []\n    for t in range(...):\n"
            "        X.append(...)\n        Y.append(...)\n    return X, Y\n")


def _x(loop: str = "range(k, len(ids))", x: str = "ids[t - k:t]", y: str = "ids[t]") -> str:
    return f"def contexts(ids, k):\n    X = []\n    Y = []\n    for t in {loop}:\n        X.append({x})\n        Y.append({y})\n    return X, Y\n"


_X = _x()

# --- exercise 3: MLPLanguageModel -----------------------------------------------------------------------------------------

_M_HEAD = "import torch\nfrom torch import nn\n\n\nclass MLPLanguageModel(nn.Module):\n    def __init__(self, vocab, k, dim, hidden):\n        super().__init__()\n"
_M_LAYERS = ("        self.emb = nn.Embedding(vocab, dim)\n        self.hidden = nn.Linear(k * dim, hidden)\n"
             "        self.out = nn.Linear(hidden, vocab)\n")
_M_FORWARD = ("        e = self.emb(x)\n        e = e.reshape(len(x), -1)\n        h = torch.relu(self.hidden(e))\n"
              "        return self.out(h)\n")
_M_START = (_M_HEAD + "        # the embedding matrix, the hidden layer on k embeddings side by side, and the output head\n        ...\n\n"
            "    def forward(self, x):\n        # x has shape (B, k): look up, glue the k embeddings into one row, hidden layer with ReLU, output head\n"
            "        ...\n")


def _m(layers: str = _M_LAYERS, forward: str = _M_FORWARD) -> str:
    return _M_HEAD + layers + "\n    def forward(self, x):\n" + forward


def _course_model():
    names: dict = {}
    exec(compile(_m(), "<MLPLanguageModel>", "exec"), names)
    return names["MLPLanguageModel"]


MLPLanguageModel = _course_model()

_M_DATA = "import torch\n\ntorch.manual_seed(0)\n"
_M_SCENARIOS = [
    # 3 characters of context, 2 examples
    _M_DATA + "model = MLPLanguageModel(65, 3, 10, 32)\nx = torch.tensor([[46, 43, 1], [58, 46, 43]])\n"
              "logits = model(x)\n[list(logits.shape), [round(v, 4) for v in logits[0, :4].tolist()]]",
    # other sizes: 5 characters, a vocabulary of 7
    _M_DATA + "model = MLPLanguageModel(7, 5, 4, 8)\nx = torch.tensor([[0, 1, 2, 3, 4], [6, 6, 6, 6, 6], [1, 0, 1, 0, 1]])\n"
              "logits = model(x)\n[list(logits.shape), [round(v, 4) for v in logits[2, :4].tolist()]]",
    # the order of the k tokens matters: their embeddings sit side by side, not added up
    _M_DATA + "model = MLPLanguageModel(7, 2, 4, 8)\na = model(torch.tensor([[1, 2]]))\nb = model(torch.tensor([[2, 1]]))\n"
              "round((a - b).abs().max().item(), 4) > 0",
]


def _out_of_order(got, text) -> bool:
    """The three layers made in another order than the data passes them: other random starting values."""
    at = [text.find(f"self.{n} =") for n in ("emb", "hidden", "out")]
    return min(at) >= 0 and at != sorted(at)


def _close(got, want) -> bool:
    import numpy as np
    if isinstance(want, bool) or isinstance(got, bool):
        return got == want
    if isinstance(want, list):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    try:
        return bool(np.allclose(float(got), float(want), rtol=1e-4, atol=1e-4))
    except (TypeError, ValueError):
        return got == want


# --- the picture ----------------------------------------------------------------------------------------------------------

def _course_trainer():
    """The Trainer of nn-trainer, from the course's solutions there: the same as the learner's, when that is right."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nn_trainer_for_mlp", paths.LESSONS / "nn-trainer" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ours = dict(projects._BUILDS_ON)   # running those checks registers nn-trainer's modules; this lesson's stay
    try:
        return mod._course_trainer()
    finally:
        projects._BUILDS_ON.clear()
        projects._BUILDS_ON.update(ours)


def _examples(k: int, places: torch.Tensor):
    """The context windows of k ids before each place, and the id at each place."""
    all_ids = ids()
    return torch.stack([all_ids[places - k + j] for j in range(k)], dim=1), all_ids[places]


@functools.lru_cache(maxsize=4)
def run(k: int) -> float:
    """MLPLanguageModel with k characters of context, trained one epoch on 128,000 examples by the Trainer (AdamW at
    0.003, warmup of 100 steps, cosine decay, clipping at 1): 1,000 steps of 128. Gives the loss on 10,000
    examples from the last 10 % of the text."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)   # faster on these small layers, and the numbers no longer depend on the number of cores
    try:
        n = len(ids())
        split = int(0.9 * n)
        g = torch.Generator().manual_seed(1000)
        train_at = torch.randint(max(KS), split, (N_TRAIN,), generator=g)
        val_at = torch.randint(split + max(KS), n, (N_VAL,), generator=g)
        torch.manual_seed(0)
        model = MLPLanguageModel(len(chars()), k, DIM, HIDDEN)
        # the mini-batches made at once, in a shuffled order: what a DataLoader with shuffle=True gives, without
        # fetching the 128,000 examples one by one
        X, Y = _examples(k, train_at)
        order = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(1))
        train = [(X[order[i:i + 128]], Y[order[i:i + 128]]) for i in range(0, N_TRAIN, 128)]
        val = DataLoader(TensorDataset(*_examples(k, val_at)), batch_size=5000)
        history = _course_trainer()(model, train, val, epochs=1, lr=LR, warmup=100, max_norm=1.0).fit()
    finally:
        torch.set_num_threads(threads)
    return history[-1]["val_loss"]


def context_picture(labels: dict) -> str:
    """The validation loss after the same training, for 1, 3 and 8 characters of context."""
    from everything_ai import chart
    losses = [run(k) for k in KS]
    panel = chart.Panel((0, 9), (2.0, 2.7), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.xticks = list(KS)
    panel.line(list(KS), losses, "loss", 2.4)
    panel.points(list(KS), losses, "loss")
    for k, loss in zip(KS, losses):   # under the point at k = 3, where the line runs on to the right
        panel.text(k, loss - 0.08 if k == 3 else loss + 0.05, labels["number"](loss, 2), "loss", "middle")
    return chart.figure([panel], alt=labels["alt"], width=330, height=260)


# The lesson is done when these are correct. "window" is the choice question in the lesson.
REQUIRED = ["window", "contexts", "mlp-lm"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_C_EN = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_C_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

EXERCISES = {
    "contexts": FunctionBox(
        name="contexts",
        function="contexts",
        start=_X_START,
        solution=_X,
        visible=[([5, 1, 4, 2, 7], 2)],
        hidden=[([5, 1, 4, 2, 7], 1), ([5, 1, 4, 2, 7], 3), ([0, 1, 2], 3), (list(range(10)), 4)],
        compare=same_value,
        first=[_DOTS],
        save_as=("llm", "contexts"),
        mistakes=[
            Mistake("with-label", wrong=_x(x="ids[t - k:t + 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every window also holds its own label, the id at place `t`. The window is the k ids before it: `ids[t - k:t]`, which stops just before `t`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk venster bevat ook zijn eigen label, het id op plek `t`. Het venster is de k ids ervoor: `ids[t - k:t]`, dat net vóór `t` stopt."),
            Mistake("one-short", wrong=_x(loop="range(k, len(ids) - 1)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the last example is missing. The last id, at place `len(ids) - 1`, is a label too, and `range` stops just before its end.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het laatste voorbeeld ontbreekt. Het laatste id, op plek `len(ids) - 1`, is ook een label, en `range` stopt net voor zijn eind."),
            Mistake("too-early", wrong=_x(loop="range(k - 1, len(ids))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your first window starts before the text does. The first place with k ids before it is `t = k`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je eerste venster begint vóór de tekst. De eerste plek met k ids ervoor is `t = k`."),
            Mistake("one-back", wrong=_x(x="ids[t - k - 1:t - 1]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every window stops one place too early, so the id just before the label is missing. `ids[t - k:t]` ends right before `t`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk venster stopt één plek te vroeg, dus het id vlak voor het label ontbreekt. `ids[t - k:t]` eindigt vlak voor `t`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: which places `t` the loop goes over, the window of k ids before place `t`, and the id at place `t`.",
                      "nl": "Vervang de drie `...`: over welke plekken `t` de loop gaat, het venster van k ids vóór plek `t`, en het id op plek `t`."},
            "arguments": {"en": "The check calls `{call}`, with a list of ids and a number, and the `def` line of your function has another number of names between its parentheses. Keep `(ids, k)`.",
                          "nl": "De check roept `{call}` aan, met een lijst ids en een getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(ids, k)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your loop with step 1 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je loop met stap 1 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every k?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke k?"},
            "correct": {"en": "Your function makes the examples for any context window: a text of n ids gives n − k of them. Your `contexts` is now in your project, as `work/projects/llm/contexts.py`.",
                        "nl": "Je functie maakt de voorbeelden voor elk context window: een tekst van n ids geeft er n − k. Je `contexts` staat nu in je project, als `work/projects/llm/contexts.py`."},
            "kept": {"en": "Your `contexts` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `contexts` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "mlp-lm": ClassBox(
        name="mlp-lm",
        start=_M_START,
        solution=_m(),
        scenarios=_M_SCENARIOS,
        compare=_close,
        digits=4,
        save_as=("llm", "mlp_lm"),
        first=[_DOTS],
        mistakes=[
            Mistake("added-up", wrong=_m(layers=_M_LAYERS.replace("nn.Linear(k * dim, hidden)", "nn.Linear(dim, hidden)"),
                                         forward=_M_FORWARD.replace("        e = e.reshape(len(x), -1)\n", "        e = e.sum(dim=1)\n")),
                    en=_C_EN + ": you add up the k embeddings, so the model no longer knows which token was where: \"ab\" and \"ba\" give the same. Put them side by side: `e.reshape(len(x), -1)`, and `nn.Linear(k * dim, hidden)`.",
                    nl=_C_NL + ": je telt de k embeddings op, dus het model weet niet meer welk token waar stond: \"ab\" en \"ba\" geven hetzelfde. Zet ze naast elkaar: `e.reshape(len(x), -1)`, en `nn.Linear(k * dim, hidden)`."),
            Mistake("sizes", error=lambda e: isinstance(e, RuntimeError) and "cannot be multiplied" in str(e),
                    en="The sizes do not fit: your hidden layer gets an input of another size than it was made for. `self.emb(x)` gives shape (B, k, dim). Glue it in `forward` with `e.reshape(len(x), -1)` before the hidden layer, which gives (B, k * dim), and make the hidden layer for exactly that size, with the names and not with fixed numbers: `nn.Linear(k * dim, hidden)`.",
                    nl="De maten passen niet: je hidden layer krijgt een input van een andere maat dan waarvoor hij gemaakt is. `self.emb(x)` geeft shape (B, k, dim). Plak die in `forward` met `e.reshape(len(x), -1)` vóór de hidden layer, wat (B, k * dim) geeft, en maak de hidden layer voor precies die maat, met de namen en niet met vaste getallen: `nn.Linear(k * dim, hidden)`."),
            Mistake("no-relu", wrong=_m(forward=_M_FORWARD.replace("torch.relu(self.hidden(e))", "self.hidden(e)")),
                    en=_C_EN + ": your hidden layer has no ReLU, so the two `nn.Linear`s together are one linear function. `h = torch.relu(self.hidden(e))`.",
                    nl=_C_NL + ": je hidden layer heeft geen ReLU, dus de twee `nn.Linear`s samen zijn één lineaire functie. `h = torch.relu(self.hidden(e))`."),
            Mistake("averaged", wrong=_m(layers=_M_LAYERS.replace("nn.Linear(k * dim, hidden)", "nn.Linear(dim, hidden)"),
                                         forward=_M_FORWARD.replace("        e = e.reshape(len(x), -1)\n", "        e = e.mean(dim=1)\n")),
                    en=_C_EN + ": you average the k embeddings, so the model no longer knows which token was where: \"ab\" and \"ba\" give the same. Put them side by side: `e.reshape(len(x), -1)`, and `nn.Linear(k * dim, hidden)`.",
                    nl=_C_NL + ": je neemt het gemiddelde van de k embeddings, dus het model weet niet meer welk token waar stond: \"ab\" en \"ba\" geven hetzelfde. Zet ze naast elkaar: `e.reshape(len(x), -1)`, en `nn.Linear(k * dim, hidden)`."),
            Mistake("init-name", error=lambda e: isinstance(e, NameError) and re.search(r"name '(vocab|k|dim|hidden)' is not defined", str(e)) is not None,
                    en="`forward` does not know `vocab`, `k`, `dim` or `hidden`: they are names of `__init__` only. Glue with `e.reshape(len(x), -1)`, which needs none of them, or keep the number in `__init__` as `self.k = k`.",
                    nl="`forward` kent `vocab`, `k`, `dim` en `hidden` niet: dat zijn namen van `__init__` alleen. Plak met `e.reshape(len(x), -1)`, dat geen van hen nodig heeft, of bewaar het getal in `__init__` als `self.k = k`."),
            Mistake("order", matches=_out_of_order,
                    en=_C_EN + ": the right shape, other numbers. Make the layers in the order in which the data passes them, `self.emb`, `self.hidden`, `self.out`, so that they get the same random starting values as in the course's solution.",
                    nl=_C_NL + ": de goede shape, andere getallen. Maak de layers in de volgorde waarin de data erdoor gaat, `self.emb`, `self.hidden`, `self.out`, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the three layers in `__init__`, and in `forward` the lookup, the glue, the hidden layer with ReLU and the output head.",
                      "nl": "Vervang de twee `...`: de drie layers in `__init__`, en in `forward` het opzoeken, het plakken, de hidden layer met ReLU en de output head."},
            "wrong": {"en": _C_EN + ": the shape of the logits and the first four of the first example. Look up, glue, hidden layer with ReLU, output head: steps 2 to 4 of the recipe?",
                      "nl": _C_NL + ": de shape van de logits en de eerste vier van het eerste voorbeeld. Opzoeken, plakken, hidden layer met ReLU, output head: stap 2 tot en met 4 van het recept?"},
            "almost": {"en": "Right for 3 characters of context. But " + _C_EN[0].lower() + _C_EN[1:] + ". Does your class use `vocab`, `k`, `dim` and `hidden`, and no fixed sizes?",
                       "nl": "Goed voor 3 tekens context. Maar " + _C_NL[0].lower() + _C_NL[1:] + ". Gebruikt je class `vocab`, `k`, `dim` en `hidden`, en geen vaste maten?"},
            "correct": {"en": "Your model gives one logit for every token of the vocabulary, from the last k tokens side by side. Your `MLPLanguageModel` is now in your project, as `work/projects/llm/mlp_lm.py`.",
                        "nl": "Je model geeft één logit voor elk token van de vocabulary, uit de laatste k tokens naast elkaar. Je `MLPLanguageModel` staat nu in je project, als `work/projects/llm/mlp_lm.py`."},
            "kept": {"en": "Your `MLPLanguageModel` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `MLPLanguageModel` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("contexts", _X_START, "empty", None),
    ("contexts", "def contexts(ids, k):\n    X = [ids[t - k:t] for t in range(k, len(ids))]\n    Y = ids[k:]\n    return X, Y\n", "correct", None),
    ("contexts", _x(x="ids[t - k:t + 1]"), "wrong", "with-label"),
    ("contexts", _x(loop="range(k, len(ids) - 1)"), "wrong", "one-short"),
    ("contexts", _x(loop="range(k - 1, len(ids))"), "wrong", "too-early"),
    ("contexts", _x(x="ids[t - k - 1:t - 1]"), "wrong", "one-back"),
    ("contexts", _x(x="ids[t - 2:t]"), "almost", None),
    ("mlp-lm", _M_START, "empty", None),
    ("mlp-lm", _m(forward=_M_FORWARD.replace("e.reshape(len(x), -1)", "e.reshape(x.shape[0], -1)")), "correct", None),
    ("mlp-lm", _m(layers=_M_LAYERS.replace("nn.Linear(k * dim, hidden)", "nn.Linear(dim, hidden)"),
                  forward=_M_FORWARD.replace("        e = e.reshape(len(x), -1)\n", "        e = e.sum(dim=1)\n")), "wrong", "added-up"),
    ("mlp-lm", _m(forward=_M_FORWARD.replace("        e = e.reshape(len(x), -1)\n", "")), "error", "sizes"),
    ("mlp-lm", _m(forward=_M_FORWARD.replace("torch.relu(self.hidden(e))", "self.hidden(e)")), "wrong", "no-relu"),
    ("mlp-lm", _m(layers="        self.hidden = nn.Linear(k * dim, hidden)\n        self.emb = nn.Embedding(vocab, dim)\n        self.out = nn.Linear(hidden, vocab)\n"), "wrong", "order"),
    ("mlp-lm", _m(layers=_M_LAYERS.replace("nn.Linear(k * dim, hidden)", "nn.Linear(30, hidden)")), "error", "sizes"),
    ("mlp-lm", _m(layers="        self.emb = nn.Embedding(vocab, dim)\n        self.out = nn.Linear(hidden, vocab)\n        self.hidden = nn.Linear(k * dim, hidden)\n"), "wrong", "order"),
    ("mlp-lm", _m(layers="        self.out = nn.Linear(hidden, vocab)\n        self.hidden = nn.Linear(k * dim, hidden)\n        self.emb = nn.Embedding(vocab, dim)\n"), "wrong", "order"),
    ("mlp-lm", _m(layers=_M_LAYERS.replace("nn.Linear(k * dim, hidden)", "nn.Linear(dim, hidden)"),
                  forward=_M_FORWARD.replace("        e = e.reshape(len(x), -1)\n", "        e = e.mean(dim=1)\n")), "wrong", "averaged"),
    ("mlp-lm", _m(forward=_M_FORWARD.replace("e.reshape(len(x), -1)", "e.reshape(-1, k * dim)")), "error", "init-name"),
    ("mlp-lm", _m(layers=_M_LAYERS.replace("nn.Embedding(vocab, dim)", "nn.Embedding(65, dim)").replace("nn.Linear(hidden, vocab)", "nn.Linear(hidden, 65)")), "almost", None),
]
