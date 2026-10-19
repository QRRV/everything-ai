# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-kv-cache: attend_step, one step of an attention head with a KV cache, and kv_cache_gb, the
memory of a KV cache."""
import math
import re

from everything_ai.check import FunctionBox, Mistake


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def steps_picture(labels: dict) -> str:
    """Four steps of writing after a start of 3 tokens, without a cache above and with a KV cache below: one row per
    step, one box per token. A box in ink goes through the model in that step and stands out in both themes; a box in ink-3
    is read from the cache and is faint. Nothing here is learned or a loss: ink. An ink box is four washes on top of each other."""
    from everything_ai import chart
    p = chart.Panel((0, 10), (-1.5, 12.2), title=labels["title"], xlabel="", ylabel="", xticks=[], yticks=[])

    def box(x, y, dark):
        for _ in range(4 if dark else 1):
            p.rect(x - 0.38, y - 0.36, x + 0.38, y + 0.36, "ink" if dark else "ink-3")

    for name, cached, top in ((labels["without"], False, 11.3), (labels["with"], True, 5.8)):
        p.text(0.1, top, name, "ink", "start")
        for step in range(4):
            y = top - 1.1 - step * 0.95
            p.text(0.1, y, labels["step"].format(n=step + 1), "ink-2", "start")
            for t in range(3 + step):
                box(2.9 + t * 0.98, y, not cached or step == 0 or t == 2 + step)
    box(0.5, 0.15, True)
    p.text(1.2, 0.15, labels["computed"], "ink", "start")
    box(0.5, -0.85, False)
    p.text(1.2, -0.85, labels["cached"], "ink", "start")
    return chart.figure([p], alt=labels["alt"], width=330, height=360)


# --- exercise 1: attend_step ----------------------------------------------------------------------------------------------------

_HEAD = ("import math\nimport torch\n\n\ndef attend_step(q, k, v, k_cache, v_cache):\n"
         "    # q, k, v: the query, key and value of the new token, (B, 1, d)\n"
         "    # k_cache, v_cache: the keys and values of the earlier tokens, (B, t, d)\n")
_ADD = "    k_cache = torch.cat([k_cache, k], dim=1)\n    v_cache = torch.cat([v_cache, v], dim=1)\n"
_ATTEND = ("    S = q @ k_cache.transpose(1, 2) / math.sqrt(q.shape[2])\n"
           "    out = torch.softmax(S, dim=2) @ v_cache\n")
_TAIL = "    return out, k_cache, v_cache\n"
_MASKED = ("    n = k_cache.shape[1]\n    future = torch.tril(torch.ones(n, n)) == 0\n    S = q @ k_cache.transpose(1, 2) / math.sqrt(q.shape[2])\n    S = S.masked_fill(future, float(\"-inf\"))\n    out = torch.softmax(S, dim=2) @ v_cache\n")
_START = (_HEAD + "    # add the new key and value at the end of the caches\n    ...\n"
          "    # the attention of the new token to every key in the cache, its own included\n    ...\n" + _TAIL)


def _a(add: str = _ADD, attend: str = _ATTEND, tail: str = _TAIL) -> str:
    return _HEAD + add + attend + tail


class _Named:
    """A tensor of a test case, written in the check's messages the way the lesson writes it; made with seed `seed`."""

    def __init__(self, shown: str, seed: int):
        self.shown, self.seed = shown, seed

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        import torch
        torch.manual_seed(self.seed)
        return eval(self.shown, {"torch": torch})


def _case(new: str, cache: str, seed: int) -> tuple:
    return (_Named(new, seed), _Named(new, seed + 1), _Named(new, seed + 2), _Named(cache, seed + 3), _Named(cache, seed + 4))


def _same(a, b) -> bool:
    import torch
    return (isinstance(a, torch.Tensor) and tuple(a.shape) == tuple(b.shape)
            and bool(torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4)))


def _steps_close(got, want) -> bool:
    try:
        return isinstance(got, (tuple, list)) and len(got) == 3 and all(_same(g, w) for g, w in zip(got, want))
    except Exception:
        return False


def _shape(t) -> str:
    return str(tuple(t.shape))


def _row(t) -> str:
    """The numbers of `out` for the first text, at most 4 of them, without float noise."""
    fmt = lambda a: (f"{a:.4f}".rstrip("0").rstrip(".") or "0") if a == a else "nan"
    if t.dim() != 3 or t.numel() == 0:
        return None
    nums = t[0, -1].tolist()
    return "[" + ", ".join(fmt(a) for a in nums[:4]) + (", …]" if len(nums) > 4 else "]")


def _show(v):
    """What a step gives back, in words: the numbers of `out` and the shape of the caches."""
    try:
        import torch
    except ImportError:
        return repr(v)
    if isinstance(v, (tuple, list)) and len(v) == 3 and all(isinstance(t, torch.Tensor) for t in v):
        out, kc, vc = v
        row = _row(out) if out.dim() == 3 and out.shape[1] == 1 else None
        start = ({"en": f"`out` `{row}`", "nl": f"`out` `{row}`"} if row else
                 {"en": f"an `out` of shape {_shape(out)}", "nl": f"een `out` met shape {_shape(out)}"})
        caches = _shape(kc) if _shape(kc) == _shape(vc) else f"{_shape(kc)} {{and}} {_shape(vc)}"
        return {"en": start["en"] + " and caches of shape " + caches.replace("{and}", "and"),
                "nl": start["nl"] + " en caches met shape " + caches.replace("{and}", "en")}
    if isinstance(v, torch.Tensor):
        return {"en": f"one tensor of shape {_shape(v)}", "nl": f"één tensor met shape {_shape(v)}"}
    if isinstance(v, (tuple, list)):
        return {"en": f"{len(v)} values", "nl": f"{len(v)} waarden"}
    if v is None or isinstance(v, (bool, int, float, str)):
        return repr(v)
    return {"en": f"a `{type(v).__name__}`", "nl": f"een `{type(v).__name__}`"}


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _is_tensor(v) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(v, torch.Tensor)


def _cat_sizes(e: BaseException) -> bool:
    return isinstance(e, RuntimeError) and "Sizes of tensors must match except in dimension" in str(e)


# --- exercise 2: kv_cache_gb -----------------------------------------------------------------------------------------------------

_GB_HEAD = "def kv_cache_gb(n_layers, n_kv_heads, d_head, n_tokens, batch, bytes_per_number):\n"
_GB_START = (_GB_HEAD + "    # the numbers one token keeps: a key and a value of every key and value head, in every layer\n    ...\n"
             "    # times the tokens, the texts and the bytes per number, in GB of 10**9 bytes\n    ...\n")


def _k(per: str = "2 * n_layers * n_kv_heads * d_head", rest: str = " * n_tokens * batch * bytes_per_number", scale: str = "1e9") -> str:
    return _GB_HEAD + f"    per_token = {per}\n    return per_token{rest} / {scale}\n"


def _gb_close(got, want) -> bool:
    if isinstance(got, (str, bytes, bool)):
        return False
    try:
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _gb_show(v):
    """A number with 3 significant digits, never in e-notation, with a decimal comma in Dutch."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return repr(v)
    x = float(v)
    if x != x or math.isinf(x):
        return repr(v)
    if isinstance(v, int):
        return {"en": number(x, 0, "en"), "nl": number(x, 0, "nl")}
    a = abs(x)
    digits = 0 if a >= 100 or a == 0 else 1 if a >= 10 else 2 if a >= 1 else min(12, 2 - math.floor(math.log10(a)))
    return {"en": number(x, digits, "en"), "nl": number(x, digits, "nl")}


EXERCISES = {
    "attend-step": FunctionBox(
        name="attend-step",
        function="attend_step",
        start=_START,
        solution=_a(),
        visible=[_case("torch.randn(1, 1, 4)", "torch.randn(1, 2, 4)", 0)],
        hidden=[_case("torch.randn(2, 1, 4)", "torch.zeros(2, 0, 4)", 5), _case("torch.randn(2, 1, 8)", "torch.randn(2, 5, 8)", 10),
                _case("torch.randn(1, 1, 2)", "torch.randn(1, 1, 2)", 15)],
        show=_show,
        compare=_steps_close,
        first=[_DOTS],
        mistakes=[
            Mistake("axis", error=_cat_sizes,
                    en="With `{call}`, `torch.cat` stops: the tensors it joins must have the same size on every axis but the one they are joined on. The tokens are the middle axis of the cache: join the cache and the new key on it, with `dim=1`.",
                    nl="Met `{call}` stopt `torch.cat`: de tensors die hij samenvoegt moeten op elke as even groot zijn, behalve op de as waarop ze samenkomen. De tokens zijn de middelste as van de cache: voeg de cache en de nieuwe key daarop samen, met `dim=1`."),
            Mistake("only-out", matches=lambda got, text: _is_tensor(got) and got.dim() == 3 and got.shape[1] == 1,
                    en="`{call}` gives {got}: only `out`. The next step needs the caches with this token in them too: give back `out, k_cache, v_cache`.",
                    nl="`{call}` geeft {got}: alleen `out`. De volgende stap heeft ook de caches nodig, met dit token erin: geef `out, k_cache, v_cache` terug."),
            Mistake("before-add", wrong=_HEAD + _ATTEND + _ADD + _TAIL,
                    en="`{call}` gives {got}, and it should give {want}: your new token does not look at itself. You compute its attention before its key and value are in the cache. Add them first: the new token sees every earlier token and itself, as in causal attention.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je nieuwe token kijkt niet naar zichzelf. Je berekent zijn attention voordat zijn key en value in de cache staan. Voeg ze eerst toe: het nieuwe token ziet elk eerder token en zichzelf, zoals in causale attention."),
            Mistake("self-only", wrong=_a(attend="    S = q @ k.transpose(1, 2) / math.sqrt(q.shape[2])\n    out = torch.softmax(S, dim=2) @ v\n"),
                    en="`{call}` gives {got}, and it should give {want}: your `out` is the new token's own value. Its query compares with its own key only, so softmax gives that key all the weight. Compare it with every key in the cache, and take the values of the cache.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `out` is de eigen value van het nieuwe token. Zijn query vergelijkt alleen met zijn eigen key, dus softmax geeft die key al het gewicht. Vergelijk hem met elke key in de cache, en neem de values van de cache."),
            Mistake("old-caches", wrong=_a(add="    keys = torch.cat([k_cache, k], dim=1)\n    values = torch.cat([v_cache, v], dim=1)\n",
                                           attend=_ATTEND.replace("k_cache.transpose", "keys.transpose").replace("@ v_cache", "@ values")),
                    en="With `{call}`, your `out` is right, but you give back the caches you got, without the new token. The next step needs them with its key and value at the end.",
                    nl="Met `{call}` klopt je `out`, maar je geeft de caches terug die je kreeg, zonder het nieuwe token. De volgende stap heeft ze nodig met zijn key en value aan het eind."),
            Mistake("front", wrong=_a(add="    k_cache = torch.cat([k, k_cache], dim=1)\n    v_cache = torch.cat([v, v_cache], dim=1)\n"),
                    en="With `{call}`, your `out` is right, but the new token stands at the front of your caches. Keep them in the order of the text, the new token at the end: `torch.cat([k_cache, k], dim=1)`.",
                    nl="Met `{call}` klopt je `out`, maar het nieuwe token staat vooraan in je caches. Houd ze in de volgorde van de tekst, het nieuwe token aan het eind: `torch.cat([k_cache, k], dim=1)`."),
            Mistake("mask", wrong=_a(attend=_MASKED),
                    en="`{call}` gives {got}, and it should give {want}: the mask of causal attention makes your scores a square, a row for every token in the cache, and `out` gets a row for each. The new token has one query, and every key in the cache is of an earlier token or its own: leave the mask out.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de mask van causale attention maakt je scores een vierkant, een rij voor elk token in de cache, en `out` krijgt er een rij voor elk. Het nieuwe token heeft één query, en elke key in de cache hoort bij een eerder token of bij hemzelf: laat de mask weg."),
            Mistake("no-scale", wrong=_a(attend=_ATTEND.replace(" / math.sqrt(q.shape[2])", "")),
                    en="`{call}` gives {got}, and it should give {want}: your scores are not scaled. Divide them by the square root of the width of the head, `math.sqrt(q.shape[2])`, as in causal attention.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je scores zijn niet geschaald. Deel ze door de wortel uit de breedte van de head, `math.sqrt(q.shape[2])`, zoals in causale attention."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the caches with the new key and value added at the end, and the attention of the new token.",
                      "nl": "Vervang de twee `...`: de caches met de nieuwe key en value aan het eind erbij, en de attention van het nieuwe token."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the query, key and value of the new token and the two caches, and the `def` line of your function has another number of names between its parentheses. Keep `(q, k, v, k_cache, v_cache)`.",
                          "nl": "De check roept `{call}` aan, met de query, key en value van het nieuwe token en de twee caches, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(q, k, v, k_cache, v_cache)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you add the new key and value at the end of the caches first, and then compare the new query with every key in the cache, scaled, through softmax, times the values of the cache?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voeg je eerst de nieuwe key en value aan het eind van de caches toe, en vergelijk je daarna de nieuwe query met elke key in de cache, geschaald, door softmax, maal de values van de cache?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your step work for an empty cache, the very first token, for more texts at once, and for keys of another width?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je stap ook met een lege cache, het allereerste token, met meer teksten tegelijk, en met keys van een andere breedte?"},
            "correct": {"en": "Your step gives the same `out` as the last row of causal attention over the whole text, and the cache grows by one token. All that is new per step is one query, one key and one value.",
                        "nl": "Je stap geeft dezelfde `out` als de laatste rij van causale attention over de hele tekst, en de cache groeit met één token. Het enige nieuwe per stap is één query, één key en één value."},
        },
    ),
    "kv-gb": FunctionBox(
        name="kv-gb",
        function="kv_cache_gb",
        start=_GB_START,
        solution=_k(),
        visible=[(32, 8, 128, 8192, 1, 2)],
        hidden=[(80, 8, 128, 4096, 1, 2), (32, 32, 128, 4096, 8, 2), (12, 12, 64, 1024, 4, 4)],
        show=_gb_show,
        compare=_gb_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-two", wrong=_k(per="n_layers * n_kv_heads * d_head"),
                    en="`{call}` gives {got}, and it should give {want}: half. You count only the keys: every token keeps a key and a value in every layer, so times 2.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de helft. Je telt alleen de keys: elk token bewaart in elke layer een key en een value, dus maal 2."),
            Mistake("no-bytes", wrong=_k(rest=" * n_tokens * batch"),
                    en="`{call}` gives {got}, and it should give {want}: you count numbers, not bytes. Multiply by `bytes_per_number`: 2 in BF16, 4 in FP32.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt getallen, geen bytes. Vermenigvuldig met `bytes_per_number`: 2 in BF16, 4 in FP32."),
            Mistake("no-layers", wrong=_k(per="2 * n_kv_heads * d_head"),
                    en="`{call}` gives {got}, and it should give {want}: that is the cache of one layer. Every layer keeps its own keys and values: times `n_layers`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de cache van één layer. Elke layer bewaart zijn eigen keys en values: maal `n_layers`."),
            Mistake("no-batch", wrong=_k(rest=" * n_tokens * bytes_per_number"),
                    en="`{call}` gives {got}, and it should give {want}: that is the cache of one text. Every text of the batch has its own cache: times `batch`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de cache van één tekst. Elke tekst van de batch heeft zijn eigen cache: maal `batch`."),
            Mistake("bytes", wrong=_k(scale="1"),
                    en="`{call}` gives {got}, and it should give {want}: that is in bytes. Divide by `1e9` to get GB.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is in bytes. Deel door `1e9` om GB te krijgen."),
            Mistake("gib", wrong=_k(scale="2**30"),
                    en="`{call}` gives {got}, and it should give {want}: you divide by `2**30`, a GiB. In this lesson a GB is `10**9` bytes.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt door `2**30`, een GiB. In deze les is een GB `10**9` bytes."),
            Mistake("rounded", matches=lambda got, text: not isinstance(got, str) and re.search(r"\bround\s*\(", text) is not None,
                    en="`{call}` gives {got}, and it should give {want}: you round with `round`, and the check compares more digits than it shows. Give back the number as it comes out, without rounding.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rondt af met `round`, en de check vergelijkt meer cijfers dan hij laat zien. Geef het getal terug zoals het uitkomt, zonder af te ronden."),
            Mistake("text", matches=lambda got, text: isinstance(got, str),
                    en="`{call}` gives the text `{got}`, and it should give the number {want}. Give back the number itself, without turning it into text.",
                    nl="`{call}` geeft de tekst `{got}`, en dat moet het getal {want} zijn. Geef het getal zelf terug, zonder er tekst van te maken."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the numbers one token keeps, and a `return` of the total in GB.",
                      "nl": "Vervang de twee `...`: de getallen die één token bewaart, en een `return` van het totaal in GB."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the layers, the key and value heads, their width, the tokens, the texts and the bytes per number, and the `def` line of your function has another number of names between its parentheses. Keep `(n_layers, n_kv_heads, d_head, n_tokens, batch, bytes_per_number)`.",
                          "nl": "De check roept `{call}` aan, met de layers, de key- en value-heads, hun breedte, de tokens, de teksten en de bytes per getal, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n_layers, n_kv_heads, d_head, n_tokens, batch, bytes_per_number)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you count a key and a value for every key and value head in every layer, times the tokens, the texts and the bytes per number, divided by `1e9`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel je een key en een value voor elke key- en value-head in elke layer, maal de tokens, de teksten en de bytes per getal, gedeeld door `1e9`?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for other models, more texts at once, and FP32?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor andere modellen, meer teksten tegelijk en FP32?"},
            "correct": {"en": "Your function is right in all {n} cases. Llama 3 8B keeps 1.07 GB of cache for one text of 8,192 tokens, and Llama 2 7B, with 4 times as many key and value heads, 17.2 GB for 8 texts of 4,096.",
                        "nl": "Je functie klopt in alle {n} gevallen. Llama 3 8B bewaart 1,07 GB cache voor één tekst van 8.192 tokens, en Llama 2 7B, met 4 keer zoveel key- en value-heads, 17,2 GB voor 8 teksten van 4.096."},
        },
    ),
}

# The lesson is done when these are correct. "fits" is the choice question.
REQUIRED = ["attend-step", "kv-gb", "fits"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("attend-step", _START, "empty", None),
    ("attend-step", _a(), "correct", None),
    ("attend-step", _a(add="    k_cache = torch.cat((k_cache, k), 1)\n    v_cache = torch.cat((v_cache, v), 1)\n",
                       attend="    d = k.shape[-1]\n    w = torch.softmax(q @ k_cache.transpose(1, 2) / d ** 0.5, dim=-1)\n    out = w @ v_cache\n"), "correct", None),
    ("attend-step", _a(add=_ADD.replace("dim=1", "dim=2")), "error", "axis"),
    ("attend-step", _a(tail="    return out\n"), "wrong", "only-out"),
    ("attend-step", _HEAD + _ATTEND + _ADD + _TAIL, "wrong", "before-add"),
    ("attend-step", _a(attend="    S = q @ k.transpose(1, 2) / math.sqrt(q.shape[2])\n    out = torch.softmax(S, dim=2) @ v\n"), "wrong", "self-only"),
    ("attend-step", _a(add="    keys = torch.cat([k_cache, k], dim=1)\n    values = torch.cat([v_cache, v], dim=1)\n",
                       attend=_ATTEND.replace("k_cache.transpose", "keys.transpose").replace("@ v_cache", "@ values")), "wrong", "old-caches"),
    ("attend-step", _a(add="    k_cache = torch.cat([k, k_cache], dim=1)\n    v_cache = torch.cat([v, v_cache], dim=1)\n"), "wrong", "front"),
    ("attend-step", _a(attend=_ATTEND.replace(" / math.sqrt(q.shape[2])", "")), "wrong", "no-scale"),
    ("kv-gb", _GB_START, "empty", None),
    ("kv-gb", _k(), "correct", None),
    ("kv-gb", _GB_HEAD + "    return 2 * n_layers * n_kv_heads * d_head * n_tokens * batch * bytes_per_number / 1_000_000_000\n", "correct", None),
    ("kv-gb", _k(per="n_layers * n_kv_heads * d_head"), "wrong", "no-two"),
    ("kv-gb", _k(rest=" * n_tokens * batch"), "wrong", "no-bytes"),
    ("kv-gb", _k(per="2 * n_kv_heads * d_head"), "wrong", "no-layers"),
    ("kv-gb", _k(rest=" * n_tokens * bytes_per_number"), "wrong", "no-batch"),
    ("kv-gb", _k(scale="1"), "wrong", "bytes"),
    ("kv-gb", _k(scale="2**30"), "wrong", "gib"),
    ("attend-step", _a(tail="    return [out, k_cache, v_cache]\n"), "correct", None),
    ("attend-step", _a(attend=_MASKED), "wrong", "mask"),
    ("attend-step", _a(tail="    return k_cache\n"), "wrong", None),
    ("kv-gb", _GB_HEAD + "    return round(2 * n_layers * n_kv_heads * d_head * n_tokens * batch * bytes_per_number / 1e9, 2)\n", "wrong", "rounded"),
    ("kv-gb", _GB_HEAD + "    return str(2 * n_layers * n_kv_heads * d_head * n_tokens * batch * bytes_per_number / 1e9)\n", "wrong", "text"),
]
