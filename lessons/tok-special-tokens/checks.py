# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson tok-special-tokens: encode, which cuts a text at every end-of-text, encodes every part
with pre-tokenization and the merge rules, and puts the id of end-of-text between the parts; and decode, which does the
reverse. Both are saved in the learner's project (work/projects/llm/encode.py and decode.py) when right; together with
train_pre_bpe they are the learner's tokenizer. The demo sends two documents through it and back."""
import re
from functools import cache

from everything_ai import draw, paths, projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

# encode imports merge_ids of tok-byte-level and PATTERN of tok-pre-tokenization; decode imports decode of
# tok-byte-level; the demo trains with train_pre_bpe. Where one of those is missing or broken, the course's version
# stands in (template 1.3). They build on each other in this order.
projects.builds_on(("llm", "merge", "tok-bpe", "merge"), ("llm", "bpe", "tok-bpe", "train-bpe"),
                   ("llm", "byte_bpe", "tok-byte-level", "train-byte-bpe"), ("llm", "byte_decode", "tok-byte-level", "decode"),
                   ("llm", "chunk_pairs", "tok-pre-tokenization", "chunk-pair-counts"),
                   ("llm", "pre_bpe", "tok-pre-tokenization", "train-pre-bpe"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md
TRAIN_CHARS, MERGES = 20_000, 100
PATTERN = r" ?\w+| ?[^\w\s]+|\s+"
END = "<|endoftext|>"
DOCS = ["To be, or not to be.", "Één ijsje 🍦!"]


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


def _merge_ids(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i = i + 2
        else:
            out.append(ids[i])
            i = i + 1
    return out


def plain_encode(text_, merges) -> list:
    """The ids of a text with pre-tokenization and the merge rules, and no special tokens."""
    out = []
    for piece in re.findall(PATTERN, text_):
        ids = list(piece.encode("utf-8"))
        for pair in merges:
            ids = _merge_ids(ids, pair, merges[pair])
        out = out + ids
    return out


def token_text(merges, i) -> str:
    v = {j: bytes([j]) for j in range(256)}
    for pair in merges:
        v[merges[pair]] = v[pair[0]] + v[pair[1]]
    return v[i].decode("utf-8", errors="replace")


def special_picture(labels: dict):
    """Two documents after each other, as they go into training: the tokens of the first, then end-of-text as one
    token of its own, then the tokens of the second. Ink only: tokens are input."""
    rows = [(labels["first"], [(t, None) for t in re.findall(PATTERN, DOCS[0])]),
            (labels["end"], [(END, None)]),
            (labels["second"], [(t, None) for t in re.findall(PATTERN, DOCS[1])])]
    return draw.token_rows(rows, title=labels["title"], alt=labels["alt"])


def shown(token: str) -> str:
    return token.replace(" ", "␣").replace("\n", "↵")


# The lesson is done when these are correct. "end-id" is the choice question in the lesson.
REQUIRED = ["end-id", "encode", "decode"]

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

# --- exercise 2: encode ------------------------------------------------------------------------------------------------------

_E_HEAD = ("import re\nfrom llm.byte_bpe import merge_ids\nfrom llm.pre_bpe import PATTERN\n\nEND = \"<|endoftext|>\"\n\n\n"
           "def encode_part(part, merges):\n    ids = []\n    for piece in re.findall(PATTERN, part):\n"
           "        piece_ids = list(piece.encode(\"utf-8\"))\n        for pair in merges:\n"
           "            piece_ids = merge_ids(piece_ids, pair, merges[pair])\n        ids = ids + piece_ids\n    return ids\n\n\n"
           "def encode(text, merges):\n")
_E_START = (_E_HEAD + "    # the first free id, for end-of-text\n    end_id = ...\n    ids = []\n"
            "    # the parts between the end-of-text tokens: the id of end-of-text between two parts, and the ids of every part\n"
            "    ...\n    return ids\n")
_E_ID = "    end_id = 256 + len(merges)\n    ids = []\n"
_E_BODY = ("    parts = text.split(END)\n    for k in range(len(parts)):\n        if k > 0:\n            ids.append(end_id)\n"
           "        ids = ids + encode_part(parts[k], merges)\n")


def _e(body: str = _E_BODY, end_id: str = _E_ID) -> str:
    return _E_HEAD + end_id + body + "    return ids\n"


# --- exercise 3: decode ------------------------------------------------------------------------------------------------------

_D_HEAD = ("from llm.byte_decode import decode as decode_bytes\n\nEND = \"<|endoftext|>\"\n\n\n"
           "def decode(ids, merges):\n    end_id = 256 + len(merges)\n    texts = []\n    part = []\n    for i in ids:\n")
_D_START = (_D_HEAD + "        # an end-of-text closes the part: its text goes to texts, and a new part starts;\n"
            "        # any other id goes into the part\n        ...\n"
            "    texts.append(decode_bytes(part, merges))\n    return END.join(texts)\n")
_D_BODY = ("        if i == end_id:\n            texts.append(decode_bytes(part, merges))\n            part = []\n"
           "        else:\n            part.append(i)\n")


def _d(body: str = _D_BODY) -> str:
    return _D_HEAD + body + "    texts.append(decode_bytes(part, merges))\n    return END.join(texts)\n"


EXERCISES = {
    "encode": FunctionBox(
        name="encode",
        function="encode",
        start=_E_START,
        solution=_e(),
        visible=[("hi!<|endoftext|>hi", {(104, 105): 256})],
        hidden=[("", {}), ("<|endoftext|>", {}), ("a<|endoftext|><|endoftext|>b", {(97, 98): 256}),
                ("hi hi", {(104, 105): 256, (32, 256): 257}), ("é<|endoftext|>", {(195, 169): 256}), ("hi hi", {(105, 32): 256})],
        python_values=True,
        save_as=("llm", "encode"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-end", wrong=_e(body="    parts = text.split(END)\n    for k in range(len(parts)):\n        ids = ids + encode_part(parts[k], merges)\n"),
                    en=_CALL + ": end-of-text has gone, so its parts follow each other without a mark. Before every part but the first, add `end_id` to `ids`.",
                    nl=_CALL_NL + ": end-of-text is weg, dus de delen volgen elkaar op zonder teken. Voeg vóór elk deel behalve het eerste `end_id` toe aan `ids`."),
            Mistake("first-too", wrong=_e(body="    parts = text.split(END)\n    for k in range(len(parts)):\n        ids.append(end_id)\n        ids = ids + encode_part(parts[k], merges)\n"),
                    en=_CALL + ": your ids start with an end-of-text that the text does not have. There is one end-of-text fewer than there are parts: add it only before a part that is not the first, `if k > 0:`.",
                    nl=_CALL_NL + ": je ids beginnen met een end-of-text die de tekst niet heeft. Er is één end-of-text minder dan er delen zijn: voeg het alleen toe vóór een deel dat niet het eerste is, `if k > 0:`."),
            Mistake("last-too", wrong=_e(body="    parts = text.split(END)\n    for k in range(len(parts)):\n        ids = ids + encode_part(parts[k], merges)\n        ids.append(end_id)\n"),
                    en=_CALL + ": your ids end with an end-of-text that the text does not have. There is one end-of-text fewer than there are parts: only between two parts, `if k > 0:` before the part.",
                    nl=_CALL_NL + ": je ids eindigen met een end-of-text die de tekst niet heeft. Er is één end-of-text minder dan er delen zijn: alleen tussen twee delen, `if k > 0:` vóór het deel."),
            Mistake("not-cut", wrong=_e(body="    ids = ids + encode_part(text, merges)\n"),
                    en=_CALL + ": `<|endoftext|>` is encoded as normal text, letter by letter. Cut the text at every end-of-text first, `text.split(END)`, and put `end_id` between the parts.",
                    nl=_CALL_NL + ": `<|endoftext|>` wordt geëncodeerd als gewone tekst, letter voor letter. Knip de tekst eerst bij elke end-of-text, `text.split(END)`, en zet `end_id` tussen de delen."),
            Mistake("taken-id", wrong=_e(end_id="    end_id = 256\n    ids = []\n"),
                    en=_CALL + ": your end-of-text has id 256, which the first merge rule already has. It needs the first id after the bytes and all the rules: `256 + len(merges)`.",
                    nl=_CALL_NL + ": je end-of-text heeft id 256, dat de eerste merge rule al heeft. Het heeft het eerste id na de bytes en alle regels nodig: `256 + len(merges)`."),
            Mistake("nested", wrong=_e(body=_E_BODY.replace("ids = ids + encode_part(parts[k], merges)", "ids.append(encode_part(parts[k], merges))")),
                    en=_CALL + ": every part is a list inside `ids`. Add the ids of a part one by one, after the others: `ids = ids + encode_part(parts[k], merges)`.",
                    nl=_CALL_NL + ": elk deel is een lijst binnen `ids`. Voeg de ids van een deel één voor één toe, achter de andere: `ids = ids + encode_part(parts[k], merges)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the first free id, after the bytes and the merge rules; then the parts of `text.split(END)`, with `end_id` between two parts and `encode_part` for every part.",
                      "nl": "Vervang de twee `...`: het eerste vrije id, na de bytes en de merge rules; dan de delen van `text.split(END)`, met `end_id` tussen twee delen en `encode_part` voor elk deel."},
            "wrong": {"en": _CALL + ". `end_id` after the bytes and the rules; the parts of `text.split(END)`, each encoded with `encode_part`, and `end_id` before every part but the first: steps 1 and 2 of the recipe?",
                      "nl": _CALL_NL + ". `end_id` na de bytes en de regels; de delen van `text.split(END)`, elk geëncodeerd met `encode_part`, en `end_id` vóór elk deel behalve het eerste: stap 1 en 2 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `merges` for the id of end-of-text, and work for any text, also one that is only end-of-text?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `merges` voor het id van end-of-text, en werkt hij voor elke tekst, ook een die alleen end-of-text is?"},
            "correct": {"en": "Your tokenizer encodes: \"hi\" becomes 256, \"!\" stays 33, and end-of-text is one id of its own, 257, after the merge rules. It is in your project now, as `{saved}`.",
                        "nl": "Je tokenizer encodeert: \"hi\" wordt 256, \"!\" blijft 33, en end-of-text is één eigen id, 257, na de merge rules. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `encode` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `encode` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "decode": FunctionBox(
        name="decode",
        function="decode",
        start=_D_START,
        solution=_d(),
        visible=[([256, 33, 257, 256], {(104, 105): 256})],
        hidden=[([], {}), ([256], {}), ([97, 257, 257, 98], {(97, 98): 256}), ([195, 256], {}), ([195, 169, 256], {}),
                ([256, 257, 32], {(104, 105): 256})],
        python_values=True,
        save_as=("llm", "decode"),
        first=[_DOTS],
        mistakes=[
            Mistake("in-bytes", error=lambda e: isinstance(e, KeyError),
                    en="Your function stops with a KeyError: an id has no bytes. End-of-text is neither a byte nor a merge rule, so it has no bytes: when `i == end_id`, close the part instead of adding `i` to it.",
                    nl="Je functie stopt met een KeyError: een id heeft geen bytes. End-of-text is geen byte en geen merge rule, dus het heeft geen bytes: als `i == end_id`, sluit dan het deel af in plaats van `i` eraan toe te voegen."),
            Mistake("skipped", wrong=_d(body="        if i != end_id:\n            part.append(i)\n"),
                    en=_CALL + ": end-of-text is gone from the text. At every `end_id`, the text of the part so far goes to `texts` and a new part starts; `END.join(texts)` at the end then puts end-of-text between the texts.",
                    nl=_CALL_NL + ": end-of-text is uit de tekst verdwenen. Bij elke `end_id` gaat de tekst van het deel tot nu toe naar `texts` en begint een nieuw deel; `END.join(texts)` zet aan het eind end-of-text tussen de teksten."),
            Mistake("no-reset", wrong=_d(body="        if i == end_id:\n            texts.append(decode_bytes(part, merges))\n        else:\n            part.append(i)\n"),
                    en=_CALL + ": the text before an end-of-text comes back again after it. After a part goes to `texts`, start a new, empty part: `part = []`.",
                    nl=_CALL_NL + ": de tekst voor een end-of-text komt erna nog een keer terug. Begin een nieuw, leeg deel nadat een deel naar `texts` is gegaan: `part = []`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with an `if` on `i == end_id`: then the text of `part` to `texts`, and `part` empty again; else `i` added to `part`.",
                      "nl": "Vervang de `...` door een `if` op `i == end_id`: dan de tekst van `part` naar `texts`, en `part` weer leeg; anders `i` toegevoegd aan `part`."},
            "wrong": {"en": _CALL + ". At every end-of-text, the text of the part so far goes to `texts` and a new part starts; every other id goes into the part: step 3 of the recipe?",
                      "nl": _CALL_NL + ". Bij elke end-of-text gaat de tekst van het deel tot nu toe naar `texts` en begint een nieuw deel; elk ander id gaat in het deel: stap 3 van het recept?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does every end-of-text close a part, also an empty one? And does `decode_bytes` get every part whole, so that the bytes of one character stay together?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Sluit elke end-of-text een deel af, ook een leeg deel? En krijgt `decode_bytes` elk deel in zijn geheel, zodat de bytes van één teken bij elkaar blijven?"},
            "correct": {"en": "Your tokenizer decodes: the parts between the ids of end-of-text become text, with end-of-text between them again. It is in your project now, as `{saved}`.",
                        "nl": "Je tokenizer decodeert: de delen tussen de ids van end-of-text worden tekst, met end-of-text er weer tussen. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `decode` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `decode` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}


# --- the demo: two documents through the learner's whole tokenizer, and back ---------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when encode.py or decode.py is not in the project yet; "tokenizer" when they stop with an
    error; "slow" when training takes too long; else a dict with the ids of the two documents joined by end-of-text,
    the id of end-of-text, whether decode gives the text back, how many ids the 13 characters of end-of-text take as
    normal text, and the modules the course's versions stood in for."""
    if not (_saved("encode") and _saved("decode")):
        return None
    run = run_code("from llm.pre_bpe import train_pre_bpe\nfrom llm.encode import encode\nfrom llm.decode import decode")
    if run.error is not None:
        return "tokenizer"
    train, encode, decode = run.names["train_pre_bpe"], run.names["encode"], run.names["decode"]
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]

    def work():
        merges = dict(train(text()[:TRAIN_CHARS], MERGES))
        joined = END.join(DOCS)
        ids = list(encode(joined, merges))
        return {"ids": ids, "end_id": 256 + len(merges), "back": decode(ids, merges) == joined,
                "plain": len(plain_encode(END, merges))}

    try:
        out = with_time_limit(work, 60)
    except Timeout:
        return "slow"
    except Exception:
        return "tokenizer"
    out["stand_ins"] = stand_ins
    return out


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("encode", _E_START, "empty", None),
    ("encode", _e(), "correct", None),
    ("encode", _e(body=("    parts = text.split(END)\n    ids = ids + encode_part(parts[0], merges)\n    for part in parts[1:]:\n"
                        "        ids.append(end_id)\n        ids = ids + encode_part(part, merges)\n")), "correct", None),
    ("encode", _e(body="    parts = text.split(END)\n    for k in range(len(parts)):\n        ids = ids + encode_part(parts[k], merges)\n"), "wrong", "no-end"),
    ("encode", _e(body="    parts = text.split(END)\n    for k in range(len(parts)):\n        ids.append(end_id)\n        ids = ids + encode_part(parts[k], merges)\n"), "wrong", "first-too"),
    ("encode", _e(body="    parts = text.split(END)\n    for k in range(len(parts)):\n        ids = ids + encode_part(parts[k], merges)\n        ids.append(end_id)\n"), "wrong", "last-too"),
    ("encode", _e(body="    ids = ids + encode_part(text, merges)\n"), "wrong", "not-cut"),
    ("encode", _e(end_id="    end_id = 256\n    ids = []\n"), "wrong", "taken-id"),
    ("encode", _e(body=_E_BODY.replace("ids = ids + encode_part(parts[k], merges)", "ids.append(encode_part(parts[k], merges))")), "wrong", "nested"),
    ("encode", _e(end_id="    end_id = 257\n    ids = []\n"), "almost", None),
    ("decode", _D_START, "empty", None),
    ("decode", _d(), "correct", None),
    ("decode", _d(body="        part.append(i)\n"), "error", "in-bytes"),
    ("decode", _d(body="        if i != end_id:\n            part.append(i)\n"), "wrong", "skipped"),
    ("decode", _d(body="        if i == end_id:\n            texts.append(decode_bytes(part, merges))\n        else:\n            part.append(i)\n"), "wrong", "no-reset"),
    ("decode", _d(body="        if i == end_id:\n            if len(part) > 0:\n                texts.append(decode_bytes(part, merges))\n            part = []\n"
                       "        else:\n            part.append(i)\n"), "almost", None),
]
