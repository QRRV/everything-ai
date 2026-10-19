# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-eval-base: token_losses, the loss of every token of a text under a model, window by
window; bits_per_byte, those losses as bits per byte of the text the tokens stand for; and a choice. The demo scores
the learner's model of llm-pretrain-capstone (or the course's) and GPT-2 small on the same stories, which none of them
trained on, per token and per byte."""
import importlib.util
import math
import time

from everything_ai import hub, models, paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _checks("llm-pretrain-capstone", "everything_ai_checks_llm_pretrain_capstone_for_eval")
S = P.S
B = P.B
BUILDS_ON = P.BUILDS_ON
projects.builds_on(*BUILDS_ON)

STORIES = 400   # the last stories of data/tinystories-small.txt: in the validation shard of llm-data-pipeline
BLOCK = {"yours": P.BLOCK, "gpt2": 1024}   # the windows each model was trained on
DOWNLOADS = {"GPT-2": models.GPT2}


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def text() -> str:
    """The text every model is scored on: the last STORIES stories of data/tinystories-small.txt, joined by
    <|endoftext|>. They lie in the validation shard of llm-data-pipeline, so the learner's model never trained on
    them; GPT-2 is from 2019, TinyStories from 2023."""
    stories = [s.strip() for s in (paths.ROOT / "data" / "tinystories-small.txt").read_text(encoding="utf-8").split(B.END)
               if s.strip()]
    return B.END.join(stories[-STORIES:])


PICTURE = "I dare you to climb the big tree with me."   # a sentence of the lesson's stories, for the picture


def cut_picture(labels: dict):
    """PICTURE cut into the tokens of the model the first button scores and into GPT-2's: the same bytes, other tokens.
    Ink only, as in draw.token_rows: tokens are input."""
    import tiktoken
    from everything_ai import draw, gpt2_tokenizer
    info = P.saved() or P.course_info()
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    tools = run_code("from llm.encode import encode\nfrom llm.decode import decode").names
    ours = [(tools["decode"]([i], merges), None) for i in tools["encode"](PICTURE, merges)]
    gpt2_tokenizer.offline()
    enc = tiktoken.get_encoding("gpt2")
    theirs = [(enc.decode([i]), None) for i in enc.encode(PICTURE)]
    n_bytes = len(PICTURE.encode("utf-8"))
    return draw.token_rows([(labels["ours"].format(n=len(ours)), ours), (labels["gpt2"].format(n=len(theirs)), theirs)],
                           title=labels["title"].format(n=n_bytes), alt=labels["alt"].format(n=n_bytes))


def score(losses: list, pieces: list) -> dict:
    """The tokens, the loss per token, the perplexity and the bits per byte of a scored text."""
    n_bytes = sum(len(p.encode("utf-8")) for p in pieces)
    total = sum(losses)
    return {"tokens": len(losses), "loss": total / len(losses), "ppl": math.exp(total / len(losses)), "bytes": n_bytes,
            "bpb": total / math.log(2) / n_bytes}


def score_yours(token_losses) -> dict | str:
    """The learner's token_losses on their model of llm-pretrain-capstone, or on the course's when they have none.
    "run" when a module of theirs stops with an error, "losses" when their token_losses does, or gives another number
    of losses than there are tokens to score."""
    try:
        model, info, where = P.load()
        tools = run_code("from llm.encode import encode\nfrom llm.decode import decode").names
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
        ids = tools["encode"](text(), merges)
        end = 256 + len(merges)
        pieces = ["" if i == end else tools["decode"]([i], merges) for i in ids[1:]]
    except Exception:
        return "run"
    try:
        start = time.perf_counter()
        losses = [float(v) for v in token_losses(model, ids, BLOCK["yours"])]
        seconds = time.perf_counter() - start
    except Exception:
        return "losses"
    if len(losses) != len(pieces):
        return "losses"
    return {**score(losses, pieces), "seconds": seconds, "source": where, "params": sum(p.numel() for p in model.parameters()),
            "vocab": info["sizes"][0], "losses": losses, "pieces": pieces}


def download_gpt2(show) -> str | None:
    """Download GPT-2 small once, as in lm-peek-real-llm; None when it is there, else why it is not: "net",
    "pinned" or "room"."""
    import urllib.error
    if gpt2_ready():
        return None
    try:
        hub.download(*models.GPT2, progress=lambda done, total: show(done, total))
    except hub.NotPinned:
        return "pinned"
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return "net"
    except OSError:
        return "room"
    return None


def gpt2_ready() -> bool:
    return hub.pinned(*models.GPT2) is not None


def score_gpt2(token_losses) -> dict | str:
    """The learner's token_losses on GPT-2 small, with GPT-2's own tokens of the same text."""
    import tiktoken
    from everything_ai import gpt2_tokenizer
    try:
        gpt2_tokenizer.offline()
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode(text().replace(B.END, "<|endoftext|>"), allowed_special={"<|endoftext|>"})
        model = models.gpt2()
        start = time.perf_counter()
        losses = [float(v) for v in token_losses(model, ids, BLOCK["gpt2"])]
        seconds = time.perf_counter() - start
        pieces = ["" if i == enc.eot_token else enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids[1:]]
    except Exception:
        return "run"
    if len(losses) != len(pieces):
        return "run"
    return {**score(losses, pieces), "seconds": seconds, "params": sum(p.numel() for p in model.parameters()), "vocab": 50257,
            "losses": losses, "pieces": pieces}


# --- measured: the course's runs of the demo with its own solutions, on the CPU, one thread ---
# "yours": the course's model of the demo of llm-pretrain-capstone (TinyStories, so not shared: only its numbers);
# "course": the course's own model, data/course-capstone.pt (Cosmopedia); "gpt2": GPT-2 small, the pinned file.
MEASURED = {
    "yours": {"tokens": 108923, "loss": 1.8295, "ppl": 6.2307, "bytes": 313145, "bpb": 0.9181, "seconds": 5.5, "params": 575337, "vocab": 1257},
    "gpt2": {"tokens": 77744, "loss": 2.4057, "ppl": 11.0857, "bytes": 313145, "bpb": 0.8616, "seconds": 371.2, "params": 124439808, "vocab": 50257},
}
# --- end of measured ---

REQUIRED = ["token-losses", "bits-per-byte", "which-better"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL

# --- exercise 1: token_losses ----------------------------------------------------------------------------------------

_L_HEAD = "import torch\nimport torch.nn.functional as F\n\n\ndef token_losses(model, ids, block):\n"
_L = (_L_HEAD + "    model.eval()\n    losses = []\n    with torch.no_grad():\n"
      "        for start in range(0, len(ids) - 1, block):\n"
      "            window = ids[start:start + block + 1]\n"
      "            x = torch.tensor([window[:-1]])\n"
      "            y = torch.tensor(window[1:])\n"
      "            logits = model(x)[0]\n"
      "            losses += F.cross_entropy(logits, y, reduction=\"none\").tolist()\n"
      "    return losses\n")
_L_START = (_L_HEAD + "    model.eval()\n    losses = []\n    with torch.no_grad():\n"
            "        for start in range(0, len(ids) - 1, block):\n"
            "            # the window of block inputs and the block tokens after each of them\n            ...\n"
            "            # the loss of every token of the window, not their mean\n            ...\n"
            "    return losses\n")
_L_SETUP = ("import torch\ntorch.manual_seed(0)\n"
            "class Toy(torch.nn.Module):  # logits that depend on the token and on its place in a window of `block` places\n"
            "    def __init__(self, block):\n        super().__init__()\n        self.block = block\n"
            "        self.table = torch.nn.Embedding(7, 7)\n        self.places = torch.nn.Embedding(block, 7)\n"
            "    def forward(self, ids):\n"
            "        assert ids.dim() == 2, \"the model takes a batch of windows\"\n"
            "        assert ids.shape[1] > 0, \"a window without inputs\"\n"
            "        assert ids.shape[1] <= self.block, \"a window longer than the model's block\"\n"
            "        return self.table(ids) + self.places(torch.arange(ids.shape[1]))\n"
            "ids = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5]\n")
_L_SCENARIOS = [
    # the lesson's example: every token but the first gets a loss
    _L_SETUP + "model = Toy(4)\nlosses = token_losses(model, ids, 4)\nlen(losses)",
    # and they are the losses of the windows, each from its own start
    _L_SETUP + "model = Toy(4)\nlosses = token_losses(model, ids, 4)\n[round(float(v), 4) for v in losses]",
    # a text whose last window would hold only one id: there is nothing left to predict
    _L_SETUP + "model = Toy(4)\nlosses = token_losses(model, ids[:9], 4)\n[round(float(v), 4) for v in losses]",
    # one window when the block is longer than the text
    _L_SETUP + "model = Toy(16)\nlosses = token_losses(model, ids, 16)\n[round(float(v), 4) for v in losses]",
]
_NO_PLUS = _L.replace("ids[start:start + block + 1]", "ids[start:start + block]")
_STEP_PLUS = _L.replace("range(0, len(ids) - 1, block)", "range(0, len(ids) - 1, block + 1)")
_EVERY = (_L_HEAD + "    model.eval()\n    losses = []\n    with torch.no_grad():\n"
          "        for i in range(1, len(ids)):\n"
          "            x = torch.tensor([ids[max(0, i - block):i]])\n"
          "            logits = model(x)[0]\n"
          "            losses.append(F.cross_entropy(logits[-1:], torch.tensor([ids[i]])).item())\n"
          "    return losses\n")
_WHOLE = (_L_HEAD + "    model.eval()\n    with torch.no_grad():\n"
          "        logits = model(torch.tensor([ids[:-1]]))[0]\n"
          "        return F.cross_entropy(logits, torch.tensor(ids[1:]), reduction=\"none\").tolist()\n")
_MEAN = _L.replace(", reduction=\"none\").tolist()", ").item()").replace("losses += F.cross_entropy", "losses.append(F.cross_entropy")\
    .replace(").item()\n", ").item())\n")

# --- exercise 2: bits_per_byte ---------------------------------------------------------------------------------------

_B = ("import math\n\n\ndef bits_per_byte(losses, pieces):\n    n_bytes = 0\n    for piece in pieces:\n"
      "        n_bytes += len(piece.encode(\"utf-8\"))\n    return sum(losses) / math.log(2) / n_bytes\n")
_B_START = ("import math\n\n\ndef bits_per_byte(losses, pieces):\n    # the bytes of text the tokens stand for, in UTF-8\n    n_bytes = ...\n"
            "    # the losses, in nats, as bits per byte\n    return ...\n")


def _close(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) <= 1e-6 * max(1.0, abs(float(want)))   # a sum in float32 is right too
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "token-losses": ScenarioBox(
        name="token-losses",
        start=_L_START,
        solution=_L,
        scenarios=_L_SCENARIOS,
        first=[_DOTS],
        mistakes=[
            Mistake("one-short", wrong=_NO_PLUS,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: the first token of every window after the first gets no loss. Your windows of `block` ids do not overlap, so that token is only an input. Take `block + 1` ids, `ids[start:start + block + 1]`: `block` inputs, and `block` tokens to predict.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: het eerste token van elk venster na het eerste krijgt geen loss. Je vensters van `block` ids overlappen niet, dus dat token is alleen een input. Neem `block + 1` ids, `ids[start:start + block + 1]`: `block` inputs, en `block` tokens om te voorspellen."),
            Mistake("skipped", wrong=_STEP_PLUS,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: between two windows a token gets no loss. The windows overlap by one id, the last target of one window and the first input of the next: step by `block`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: tussen twee vensters krijgt een token geen loss. De vensters overlappen één id, het laatste doel van het ene venster en de eerste input van het volgende: stap met `block`."),
            Mistake("every-token", wrong=_EVERY,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: your function runs the model once for every token, on a window of its own that ends just before it. That runs the model about `block` times as often: for GPT-2, on the course's CPU, days instead of minutes. Let one window of `block + 1` ids score all its `block` tokens at once.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je functie laat het model één keer per token lopen, op een eigen venster dat vlak ervoor eindigt. Zo loopt het model ongeveer `block` keer zo vaak: voor GPT-2, op de CPU van de course, dagen in plaats van minuten. Laat één venster van `block + 1` ids al zijn `block` tokens tegelijk scoren."),
            Mistake("too-long", error=lambda e: "longer than the model's block" in str(e),
                    en="The check's model has `block` places, as GPT-2 has 1,024, and your function gives it a window with more inputs. Give it at most `block` inputs at a time: the window `ids[start:start + block + 1]` without its last id.",
                    nl="Het model van de check heeft `block` plaatsen, zoals GPT-2 er 1.024 heeft, en je functie geeft het een venster met meer inputs. Geef het hooguit `block` inputs tegelijk: het venster `ids[start:start + block + 1]` zonder zijn laatste id."),
            Mistake("empty-window", error=lambda e: "a window without inputs" in str(e),
                    en="Your function gives the model a window without inputs: at the end of `ids`, a window of one id has no input and nothing to predict. A window needs at least two ids, so stop the loop before that: `range(0, len(ids) - 1, block)`.",
                    nl="Je functie geeft het model een venster zonder inputs: aan het eind van `ids` heeft een venster van één id geen input en niets om te voorspellen. Een venster heeft minstens twee ids nodig, dus stop de loop daarvoor: `range(0, len(ids) - 1, block)`."),
            Mistake("no-batch", error=lambda e: "takes a batch of windows" in str(e),
                    en="A model takes a batch of windows, a tensor with two dimensions, and your function gives it one window of one dimension. Put the window in a list of one: `torch.tensor([window[:-1]])`, and take its logits with `model(x)[0]`.",
                    nl="Een model neemt een batch vensters, een tensor met twee dimensies, en je functie geeft het één venster met één dimensie. Zet het venster in een lijst van één: `torch.tensor([window[:-1]])`, en neem de logits ervan met `model(x)[0]`."),
            Mistake("rows", error=lambda e: "to match target batch_size" in str(e),
                    en="`F.cross_entropy` gets another number of rows of logits than targets. Are the logits those of one window, `model(x)[0]`, one row per input? And are the targets that window without its first id, as many as the inputs?",
                    nl="`F.cross_entropy` krijgt een ander aantal rijen logits dan doelen. Zijn de logits die van één venster, `model(x)[0]`, één rij per input? En zijn de doelen dat venster zonder zijn eerste id, evenveel als de inputs?"),
            Mistake("mean", wrong=_MEAN,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: your function gives one loss per window, their mean. Ask `F.cross_entropy` for the loss of every token, `reduction=\"none\"`, and add them to the list with `.tolist()`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je functie geeft één loss per venster, hun gemiddelde. Vraag `F.cross_entropy` om de loss van elk token, `reduction=\"none\"`, en voeg ze met `.tolist()` aan de lijst toe."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the window with its inputs and targets, and the loss of every token.",
                      "nl": "Vervang de twee `...`: het venster met zijn inputs en doelen, en de loss van elk token."},
            "wrong": {"en": "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does every window take `block + 1` ids, the inputs without the last and the targets without the first, and does it give the loss of every token?",
                      "nl": "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Neemt elk venster `block + 1` ids, de inputs zonder de laatste en de doelen zonder de eerste, en geeft het de loss van elk token?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does every window start its context again, and does a text shorter than `block` give one window?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Begint elk venster zijn context opnieuw, en geeft een tekst korter dan `block` één venster?"},
            "correct": {"en": "Your function gives the loss of every token but the first, each predicted from the tokens before it in its window, as the model was trained.",
                        "nl": "Je functie geeft de loss van elk token behalve het eerste, elk voorspeld uit de tokens ervoor in zijn venster, zoals het model getraind is."},
        },
    ),
    "bits-per-byte": FunctionBox(
        name="bits-per-byte",
        function="bits_per_byte",
        start=_B_START,
        solution=_B,
        visible=[([1.2, 0.7, 2.3], ["The", " café", " is"])],
        hidden=[([2.0], ["a"]), ([0.5, 0.5, 0.5, 0.5], ["Once", " upon", " a", " time"]), ([1.0, 3.0], ["naïve", " 日本"]),
                ([0.0, 1.5], ["", " end"])],
        show=lambda v: repr(float(f"{v:.6g}")) if isinstance(v, float) else S._rounded(v),
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("chars", wrong=_B.replace("len(piece.encode(\"utf-8\"))", "len(piece)"),
                    en=_CALL + ": you count characters, and `é` takes two bytes in UTF-8. Count bytes: `len(piece.encode(\"utf-8\"))`.",
                    nl=_CALL_NL + ": je telt tekens, en `é` kost in UTF-8 twee bytes. Tel bytes: `len(piece.encode(\"utf-8\"))`."),
            Mistake("nats", wrong=_B.replace(" / math.log(2)", ""),
                    en=_CALL + ": the losses are in nats, with the natural log. One bit is `math.log(2)` nats, about 0.693: divide by it.",
                    nl=_CALL_NL + ": de losses zijn in nats, met de natuurlijke log. Eén bit is `math.log(2)` nats, ongeveer 0,693: deel erdoor."),
            Mistake("times-log", wrong=_B.replace(" / math.log(2)", " * math.log(2)"),
                    en=_CALL + ": one bit is `math.log(2)` nats, about 0.693, so a number of nats is more bits: divide by `math.log(2)`, do not multiply.",
                    nl=_CALL_NL + ": één bit is `math.log(2)` nats, ongeveer 0,693, dus een aantal nats is meer bits: deel door `math.log(2)`, niet vermenigvuldigen."),
            Mistake("mean", wrong=_B.replace("sum(losses)", "sum(losses) / len(losses)"),
                    en=_CALL + ": you spread the mean loss per token over the bytes. Spread the total over them: `sum(losses)`, the loss of the whole text.",
                    nl=_CALL_NL + ": je verdeelt de gemiddelde loss per token over de bytes. Verdeel het totaal erover: `sum(losses)`, de loss van de hele tekst."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the bytes of all pieces, and the losses in bits per byte.",
                      "nl": "Vervang de twee `...`: de bytes van alle stukken, en de losses in bits per byte."},
            "wrong": {"en": _CALL + ". The sum of the losses, divided by `math.log(2)` and by the bytes of all pieces in UTF-8?",
                      "nl": _CALL_NL + ". De som van de losses, gedeeld door `math.log(2)` en door de bytes van alle stukken in UTF-8?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `losses` and `pieces`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `losses` en `pieces`, en geen vaste getallen?"},
            "correct": {"en": "Your function measures a model in bits per byte: a number that two models with different tokenizers can share.",
                        "nl": "Je functie meet een model in bits per byte: een getal dat twee modellen met verschillende tokenizers kunnen delen."},
        },
    ),
}

CASES = [
    ("token-losses", _L_START, "empty", None),
    ("token-losses", _L, "correct", None),
    ("token-losses", _L.replace("            logits = model(x)[0]\n            losses += F.cross_entropy(logits, y, reduction=\"none\").tolist()\n",
                                "            logits = model(x)\n            losses += F.cross_entropy(logits[0], y, reduction=\"none\").tolist()\n"), "correct", None),
    ("token-losses", _NO_PLUS, "wrong", "one-short"),
    ("token-losses", _STEP_PLUS, "wrong", "skipped"),
    ("token-losses", _MEAN, "wrong", "mean"),
    ("token-losses", _EVERY, "wrong", "every-token"),
    ("token-losses", _L.replace("model(x)[0]", "model(x)"), "error", "rows"),
    ("token-losses", _WHOLE, "error", "too-long"),
    ("token-losses", _L.replace("x = torch.tensor([window[:-1]])", "x = torch.tensor([window])").replace("model(x)[0]", "model(x)[0][:-1]"), "error", "too-long"),
    ("token-losses", _L.replace("x = torch.tensor([window[:-1]])", "x = torch.tensor(window[:-1])"), "error", "no-batch"),
    ("token-losses", _L.replace("F.cross_entropy(logits, y,", "F.nll_loss(F.log_softmax(logits, dim=-1), y,"), "correct", None),
    ("token-losses", _L.replace("range(0, len(ids) - 1, block)", "range(0, len(ids) - 1, max(1, block // 2))"), "wrong", None),
    ("token-losses", _L.replace("range(0, len(ids) - 1, block)", "range(0, len(ids), block)"), "error", "empty-window"),
    ("bits-per-byte", _B_START, "empty", None),
    ("bits-per-byte", _B, "correct", None),
    ("bits-per-byte", _B.replace("sum(losses) / math.log(2) / n_bytes", "sum(losses) / (math.log(2) * n_bytes)"), "correct", None),
    ("bits-per-byte", _B.replace("len(piece.encode(\"utf-8\"))", "len(piece)"), "wrong", "chars"),
    ("bits-per-byte", _B.replace(" / math.log(2)", ""), "wrong", "nats"),
    ("bits-per-byte", _B.replace(" / math.log(2)", " * math.log(2)"), "wrong", "times-log"),
    ("bits-per-byte", _B.replace("sum(losses)", "sum(losses) / len(losses)"), "wrong", "mean"),
    ("bits-per-byte", "import math\nimport torch\n\n\ndef bits_per_byte(losses, pieces):\n    n_bytes = sum(len(p.encode(\"utf-8\")) for p in pieces)\n    return torch.tensor(losses).sum().item() / math.log(2) / n_bytes\n", "correct", None),
]
