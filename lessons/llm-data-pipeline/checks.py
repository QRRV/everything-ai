# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-data-pipeline: write_shards, token ids written once to files of a fixed number of ids, in
the smallest type that holds them; TokenWindows, a Dataset that reads windows of ids from those files with np.memmap;
and a choice. The demos tokenize the course's TinyStories once with the learner's tokenizer of gpt-train-bpe (or the
course's), write the shards in the learner's project, and read random windows from them."""
import html
import importlib.util
import json
import time

import numpy as np

from everything_ai import paths, projects
from everything_ai.check import ClassBox, Mistake, run_code


def _train_bpe():
    """The checks of gpt-train-bpe: its stories, its tokenizer and its trained GPT on tokens."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_gpt_train_bpe_for_pipeline",
                                                  paths.LESSONS / "gpt-train-bpe" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _train_bpe()
_TOKENIZER = ("merge", "bpe", "byte_bpe", "byte_decode", "chunk_pairs", "pre_bpe", "encode", "decode", "fast_encode")
BUILDS_ON = tuple(m for m in B.BUILDS_ON if m[0] == "llm" and m[1] in _TOKENIZER) + (("llm", "shards", "llm-data-pipeline", "write-shards"),
                           ("llm", "token_windows", "llm-data-pipeline", "token-windows"))
projects.builds_on(*BUILDS_ON)

SHARD = 2 ** 18     # ids per shard in the demo: 262,144
BLOCK = 128         # ids per window, as the CPU plan of gpt-train-bpe
BATCH = 32          # windows per mini-batch, as the CPU plan of gpt-train-bpe
FOLDER = "tokens"   # the shards' folder in work/projects/llm/
DTYPE = "uint16" if B.VOCAB <= 2 ** 16 else "uint32"   # the type write_shards chooses for the course's vocabulary


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


def tokenizer() -> dict:
    """The merge rules the demo encodes with: those of the learner's GPT on tokens of gpt-train-bpe ("yours"), else those
    of the course's ("course"), with where they come from."""
    g = B.trained()
    if g["source"] is None:
        g = {**json.loads((paths.ROOT / B.COURSE_GPT).with_suffix(".json").read_text(encoding="utf-8")), "source": "course"}
    return {"merges": {tuple(pair): 256 + i for i, pair in enumerate(g["merges"])}, "source": g["source"]}


def folder():
    return projects.folder() / "llm" / FOLDER


def write_demo(write_shards) -> dict | str:
    """The first demo: encode all stories once with the learner's encode_fast and the merge rules of tokenizer(), then
    write them with the learner's write_shards into work/projects/llm/tokens/, after removing the shards an earlier run
    left there. "run" when encode_fast cannot be imported; else the sizes and the seconds."""
    run = run_code("from llm.fast_encode import encode_fast")
    if run.error is not None:
        return "run"
    tok = tokenizer()
    text = B.text()
    start = time.perf_counter()
    ids = run.names["encode_fast"](text, tok["merges"])
    seconds = time.perf_counter() - start
    out = folder()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard_*.bin"):
        old.unlink()
    written = write_shards(ids, B.VOCAB, str(out), SHARD)
    (out / "info.json").write_text(json.dumps({"vocab": B.VOCAB, "dtype": DTYPE, "shard": SHARD, "ids": len(ids),
                                               "merges": tok["source"], "files": [p.rsplit("/", 1)[-1] for p in written]}) + "\n",
                                   encoding="utf-8")
    size = sum((out / p.rsplit("/", 1)[-1]).stat().st_size for p in written)
    return {"chars": len(text), "text_bytes": len(text.encode("utf-8")), "ids": len(ids), "shards": len(written),
            "bytes": size, "seconds": seconds, "source": tok["source"], "top": max(ids)}


def read_demo(TokenWindows) -> dict | None:
    """The second demo: the learner's TokenWindows on the shards of the first demo, in a DataLoader that shuffles; the
    seconds to open them and to read one mini-batch, a window decoded, and whether its windows are those that
    gpt_examples of gpt-assemble makes of every shard. None when the first demo has not written the shards."""
    import torch
    from torch.utils.data import DataLoader
    info_file = folder() / "info.json"
    if not info_file.exists():
        return None
    info = json.loads(info_file.read_text(encoding="utf-8"))
    files = [str(folder() / name) for name in info["files"]]
    start = time.perf_counter()
    dtype = np.dtype(info["dtype"])
    ds = TokenWindows(files, BLOCK, dtype)
    opened = time.perf_counter() - start
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(0))
    start = time.perf_counter()
    xb, yb = next(iter(loader))
    batch = time.perf_counter() - start
    want = []   # every window of every shard, side by side, as gpt_examples of gpt-assemble makes them
    for name in files:
        ids = np.fromfile(name, dtype=dtype).tolist()
        want += [(ids[s:s + BLOCK], ids[s + 1:s + BLOCK + 1]) for s in range(0, len(ids) - BLOCK, BLOCK)]
    same = len(ds) == len(want) and all(ds[i][0].tolist() == x and ds[i][1].tolist() == y for i, (x, y) in enumerate(want))
    run = run_code("from llm.decode import decode")
    text = run.names["decode"](xb[0].tolist(), tokenizer()["merges"]) if run.error is None else None
    return {"windows": len(ds), "opened": opened, "batch": batch, "shape": tuple(xb.shape), "same": same,
            "text": text, "ids": info["ids"]}


def sizes_picture(rows: list, labels: dict) -> str:
    """Bars of megabytes on one scale, ink only: what the stories take as text and as ids of two types. `rows`:
    (label, megabytes); labels has "title", "note", "alt" and "lang"."""
    lang = labels.get("lang", "en")
    W, BAR, ROW, TOP = 330, 14, 40, 30
    top = max(v for _, v in rows)
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = TOP
    for name, value in rows:
        w = max(2.0, (W - 70) * value / top)
        p.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(name)}</text>'
                 f'<rect x="0" y="{y + 17}" width="{w:.1f}" height="{BAR}" rx="2" style="fill:var(--ea-ink)"/>'
                 f'<text x="{w + 6:.1f}" y="{y + 28.5}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{number(value, 1, lang)} MB</text>')
        y += ROW
    p.append(f'<text x="0" y="{y + 10}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["note"])}</text>')
    H = y + 18
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


# --- measured: the course's run of the first demo with its own solutions and the course's merge rules ---
# The stories, their ids with the course's merge rules (of Cosmopedia), the shards and their bytes, and the seconds
# on the course's computer to encode the stories, to open the shards and to read one mini-batch of BATCH windows.
MEASURED = {"chars": 5000450, "ids": 1704412, "shards": 7, "bytes": 3408824, "windows": 13309,
            "encode": 4.1, "open": 0.0016, "batch": 0.0032}
# --- end of measured ---

REQUIRED = ["write-shards", "token-windows", "why-shards"]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...")
                                                          for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_FN = "With these lines after your function:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_FN_NL = "Met deze regels na je functie:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"
_CLS = "With these lines after your class:\n\n```python\n{scenario}\n```\n\nyou get `{got}`, and it should be `{want}`"
_CLS_NL = "Met deze regels na je class:\n\n```python\n{scenario}\n```\n\nkrijg je `{got}`, en dat moet `{want}` zijn"

# --- exercise 1: write_shards ------------------------------------------------------------------------------------------

_W_HEAD = "import numpy as np\n\n\ndef write_shards(ids, vocab, folder, size):\n"
_W_TYPE = "    if vocab <= 2 ** 16:\n        dtype = np.uint16\n    else:\n        dtype = np.uint32\n"
_W_LOOP = ("    paths = []\n    for start in range(0, len(ids), size):\n        path = f\"{folder}/shard_{len(paths)}.bin\"\n"
           "        np.array(ids[start:start + size], dtype=dtype).tofile(path)\n        paths.append(path)\n    return paths\n")
_W_START = (_W_HEAD + "    # the smallest type that holds every id: 2 bytes for up to 2 ** 16 different ids, else 4\n    ...\n"
            "    paths = []\n    # every `size` ids in a file of their own, shard_0.bin, shard_1.bin, ... in folder; the last may be shorter\n"
            "    ...\n    return paths\n")


def _w(kind: str = _W_TYPE, loop: str = _W_LOOP) -> str:
    return _W_HEAD + kind + loop


def _read(dtype: str) -> str:
    return ("[(p.replace(folder, \"folder\"), os.path.getsize(p), np.fromfile(p, dtype=np." + dtype + ").tolist()) for p in paths]")


def _outside(got, text) -> bool:
    """The function gives back paths that are not in `folder`: the scenario writes "folder/" for it in front."""
    return (isinstance(got, list) and len(got) > 0
            and all(isinstance(t, tuple) and isinstance(t[0], str) and not t[0].startswith("folder/") for t in got))


_W_SCENARIOS = [
    "import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\npaths = write_shards([5, 300, 7, 1256, 0, 42, 9], 1257, folder, 3)\n" + _read("uint16"),
    "import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\npaths = write_shards(list(range(8)), 50257, folder, 4)\n" + _read("uint16"),
    "import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\npaths = write_shards([65535, 1, 2], 65536, folder, 5)\n" + _read("uint16"),
    "import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\npaths = write_shards([128000, 3, 70000, 9], 128256, folder, 2)\n" + _read("uint32"),
    "import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\npaths = write_shards([], 1257, folder, 3)\npaths",
]

# --- exercise 2: TokenWindows ------------------------------------------------------------------------------------------

_T_HEAD = ("import numpy as np\nimport torch\nfrom torch.utils.data import Dataset\n\n\nclass TokenWindows(Dataset):\n"
           "    def __init__(self, paths, block, dtype):\n        self.block = block\n")
_T_OPEN = "        self.shards = [np.memmap(p, dtype=dtype, mode=\"r\") for p in paths]\n"
_T_INDEX = ("        self.starts = []\n        for k in range(len(self.shards)):\n"
            "            for s in range(0, len(self.shards[k]) - block, block):\n                self.starts.append((k, s))\n")
_T_LEN = "\n    def __len__(self):\n        return len(self.starts)\n"
_T_GET = ("\n    def __getitem__(self, i):\n        k, s = self.starts[i]\n"
          "        ids = torch.tensor(self.shards[k][s:s + self.block + 1].tolist())\n        return ids[:-1], ids[1:]\n")
_T_START = (_T_HEAD + "        # every shard as an array on disk, read only\n        self.shards = ...\n"
            "        # (shard, start) of every window: windows of block ids side by side, as long as one more id follows\n"
            "        self.starts = []\n        ...\n" + _T_LEN +
            "\n    def __getitem__(self, i):\n        k, s = self.starts[i]\n"
            "        # the window's block ids and the one after them, as a tensor of int64\n        ids = ...\n"
            "        return ids[:-1], ids[1:]\n")


def _t(opened: str = _T_OPEN, index: str = _T_INDEX, get: str = _T_GET) -> str:
    return _T_HEAD + opened + index + _T_LEN + get


_T_FILES = ("import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\nfiles = []\n"
            "for k, part in enumerate([list(range(10)), list(range(100, 107))]):\n"
            "    files.append(os.path.join(folder, \"shard_\" + str(k) + \".bin\"))\n    np.array(part, dtype=np.uint16).tofile(files[-1])\n")
_T_SCENARIOS = [
    _T_FILES + "ds = TokenWindows(files, 3, np.uint16)\nx, y = ds[0]\n[len(ds), x.tolist(), y.tolist(), x.dtype == torch.int64]",
    _T_FILES + "ds = TokenWindows(files, 3, np.uint16)\n[ds[i][0].tolist() + ds[i][1][-1:].tolist() for i in range(len(ds))]",
    _T_FILES + "ds = TokenWindows(files, 4, np.uint16)\n[(ds[i][0].tolist(), ds[i][1].tolist()) for i in range(len(ds))]",
    ("import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\nfile = os.path.join(folder, \"shard_0.bin\")\n"
     "np.array(list(range(20, 29)), dtype=np.uint16).tofile(file)\nds = TokenWindows([file], 3, np.uint16)\n"
     "[(ds[i][0].tolist(), ds[i][1].tolist()) for i in range(len(ds))]"),
    ("import os\nimport tempfile\n\nfolder = tempfile.mkdtemp()\nfile = os.path.join(folder, \"shard_0.bin\")\n"
     "np.array([70000, 5, 128000, 6, 7], dtype=np.uint32).tofile(file)\nds = TokenWindows([file], 2, np.uint32)\n"
     "[(ds[i][0].tolist(), ds[i][1].tolist()) for i in range(len(ds))]"),
    _T_FILES + "ds = TokenWindows(files, 3, np.uint16)\nfrom torch.utils.data import DataLoader\n"
     "xb, yb = next(iter(DataLoader(ds, batch_size=4)))\n[tuple(xb.shape), tuple(yb.shape)]",
]


def _no_memmap(got, text) -> bool:
    """The class does not use np.memmap outside its comments: it reads the files with np.fromfile, np.load or open."""
    return "memmap" not in "\n".join(line.split("#", 1)[0] for line in text.splitlines())


EXERCISES = {
    "write-shards": ClassBox(
        name="write-shards",
        start=_W_START,
        solution=_w(),
        scenarios=_W_SCENARIOS,
        save_as=("llm", "shards"),
        first=[_DOTS],
        mistakes=[
            Mistake("int64", wrong=_w(loop=_W_LOOP.replace(", dtype=dtype)", ")")),
                    en=_FN + ": `np.array(ids)` without a type makes int64, 8 bytes for every id, so the file is 4 times as big, and read back as 2-byte ids it gives other numbers. Give the type: `np.array(..., dtype=dtype)`.",
                    nl=_FN_NL + ": `np.array(ids)` zonder type maakt int64, 8 bytes voor elke id, dus het bestand is 4 keer zo groot, en teruggelezen als ids van 2 bytes geeft het andere getallen. Geef het type mee: `np.array(..., dtype=dtype)`."),
            Mistake("too-small", error=lambda e: isinstance(e, OverflowError),
                    en="Your function stops with an OverflowError: an id is too large for its type. `np.uint16` holds 0 to 65,535; with more than 2 ** 16 different ids, use `np.uint32`.",
                    nl="Je functie stopt met een OverflowError: een id is te groot voor zijn type. `np.uint16` bevat 0 tot en met 65.535; met meer dan 2 ** 16 verschillende ids gebruik je `np.uint32`."),
            Mistake("below", wrong=_w(kind=_W_TYPE.replace("vocab <= 2 ** 16", "vocab < 2 ** 16")),
                    en=_FN + ": with 65,536 different ids, the ids are 0 to 65,535, and `np.uint16` holds them all. Use 2 bytes up to and including 2 ** 16 ids: `vocab <= 2 ** 16`.",
                    nl=_FN_NL + ": met 65.536 verschillende ids lopen de ids van 0 tot en met 65.535, en `np.uint16` bevat ze allemaal. Gebruik 2 bytes tot en met 2 ** 16 ids: `vocab <= 2 ** 16`."),
            Mistake("tail-lost", wrong=_w(loop=_W_LOOP.replace("range(0, len(ids), size)", "range(0, len(ids) - size + 1, size)")),
                    en=_FN + ": the last ids are missing, when they do not fill a whole shard. Let the loop run to `len(ids)`: the last shard may be shorter.",
                    nl=_FN_NL + ": de laatste ids ontbreken, als ze geen hele shard vullen. Laat de loop tot `len(ids)` lopen: de laatste shard mag korter zijn."),
            Mistake("names", wrong=_w(loop=_W_LOOP.replace("shard_{len(paths)}.bin", "shard_{start}.bin")),
                    en=_FN + ": your shards are named after the place of their first id. Number them 0, 1, 2, ...: `shard_{{len(paths)}}.bin`, so that a later lesson finds them in order.",
                    nl=_FN_NL + ": je shards heten naar de plek van hun eerste id. Nummer ze 0, 1, 2, ...: `shard_{{len(paths)}}.bin`, zodat een latere les ze op volgorde vindt."),
            Mistake("no-folder", matches=_outside,
                    en=_FN + ": your shards are not in `folder`. A name without a folder in front, such as `shard_0.bin`, puts the file in the folder Python runs in; you can delete those files. Put `folder` in front of every name: `f\"{{folder}}/shard_...\"`.",
                    nl=_FN_NL + ": je shards staan niet in `folder`. Een naam zonder map ervoor, zoals `shard_0.bin`, zet het bestand in de map waarin Python draait; die bestanden mag je weggooien. Zet `folder` voor elke naam: `f\"{{folder}}/shard_...\"`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: choose the type from `vocab`, then a loop that writes every `size` ids to a file of its own.",
                      "nl": "Vervang de twee `...`: kies het type met `vocab`, en dan een loop die elke `size` ids naar een eigen bestand schrijft."},
            "wrong": {"en": _FN + ". Every `size` ids in a file of their own, in the smallest type that holds `vocab` ids, numbered from 0?",
                      "nl": _FN_NL + ". Elke `size` ids in een eigen bestand, in het kleinste type dat `vocab` ids bevat, genummerd vanaf 0?"},
            "almost": {"en": "Right for the first example. But " + _FN[0].lower() + _FN[1:] + ". Does your function use `vocab`, `size` and `folder`, and no fixed numbers?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _FN_NL[0].lower() + _FN_NL[1:] + ". Gebruikt je functie `vocab`, `size` en `folder`, en geen vaste getallen?"},
            "correct": {"en": "Your function writes token ids once, in shards, in the smallest type that holds them. It is in your project now, as `{saved}`.",
                        "nl": "Je functie schrijft token-ids één keer weg, in shards, in het kleinste type dat ze bevat. Hij staat nu in je project, als `{saved}`."},
        },
    ),
    "token-windows": ClassBox(
        name="token-windows",
        start=_T_START,
        solution=_t(),
        scenarios=_T_SCENARIOS,
        save_as=("llm", "token_windows"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-shift", wrong=_t(get=_T_GET.replace("        return ids[:-1], ids[1:]\n", "        return ids[:-1], ids[:-1]\n")),
                    en=_CLS + ": the target is the window itself. A GPT learns to predict the next id, so the target is the window one id further: `ids[1:]`.",
                    nl=_CLS_NL + ": de target is het venster zelf. Een GPT leert de volgende id te voorspellen, dus de target is het venster één id verder: `ids[1:]`."),
            Mistake("one-short", wrong=_t(get=_T_GET.replace("s + self.block + 1]", "s + self.block]")),
                    en=_CLS + ": you read `block` ids, so the input and the target are one id short. Read the window and the id after it: `s + self.block + 1`.",
                    nl=_CLS_NL + ": je leest `block` ids, dus de input en de target zijn één id te kort. Lees het venster en de id erna: `s + self.block + 1`."),
            Mistake("every-start", wrong=_t(index=_T_INDEX.replace("range(0, len(self.shards[k]) - block, block)", "range(len(self.shards[k]) - block)")),
                    en=_CLS + ": your windows start at every id, so they overlap and there are about `block` times as many. Start them `block` ids apart: `range(0, len(...) - block, block)`.",
                    nl=_CLS_NL + ": je vensters beginnen bij elke id, dus ze overlappen en het zijn er ongeveer `block` keer zoveel. Laat ze om de `block` ids beginnen: `range(0, len(...) - block, block)`."),
            Mistake("last-short", wrong=_t(index=_T_INDEX.replace("len(self.shards[k]) - block, block)", "len(self.shards[k]) - block + 1, block)")),
                    en=_CLS + ": your last window of a shard has no id after it, so its target is one id short. A window needs `block + 1` ids: the loop goes up to `len(...) - block`.",
                    nl=_CLS_NL + ": je laatste venster van een shard heeft geen id erna, dus zijn target is één id te kort. Een venster heeft `block + 1` ids nodig: de loop gaat tot `len(...) - block`."),
            Mistake("no-long", wrong=_t(get=_T_GET.replace(".tolist())", ")")),
                    en=_CLS + ": your window is not a tensor of type int64. An embedding layer looks up ids of type int64, which `torch.tensor` makes of a list of Python ints: `torch.tensor(....tolist())`.",
                    nl=_CLS_NL + ": je venster is geen tensor van type int64. Een embedding layer zoekt ids van type int64 op, en die maakt `torch.tensor` van een lijst ints van Python: `torch.tensor(....tolist())`."),
            Mistake("across", wrong=(_T_HEAD + "        self.shards = [np.memmap(p, dtype=dtype, mode=\"r\") for p in paths]\n"
                                     "        self.all = np.concatenate(self.shards)\n        self.starts = [(0, s) for s in range(0, len(self.all) - block, block)]\n"
                                     + _T_LEN + "\n    def __getitem__(self, i):\n        k, s = self.starts[i]\n"
                                     "        ids = torch.tensor(self.all[s:s + self.block + 1].tolist())\n        return ids[:-1], ids[1:]\n"),
                    en=_CLS + ": your windows run across the end of one shard into the next. Index the windows of every shard on its own, as `(k, s)`, so that every window lies in one shard.",
                    nl=_CLS_NL + ": je vensters lopen over het eind van een shard door in de volgende. Maak de vensters van elke shard apart, als `(k, s)`, zodat elk venster in één shard ligt."),
        ],
        form=[Mistake("no-memmap", state="almost", matches=_no_memmap,
                      en="Right outcomes, but your class does not open the shards with `np.memmap`. Open every shard with `np.memmap(p, dtype=dtype, mode=\"r\")`: it reads nothing when it opens, and then only the windows you ask for, so a corpus of many gigabytes needs no more memory than a few windows.",
                      nl="Goede uitkomsten, maar je class opent de shards niet met `np.memmap`. Open elke shard met `np.memmap(p, dtype=dtype, mode=\"r\")`: die leest niets bij het openen, en daarna alleen de vensters waar je om vraagt, dus een corpus van vele gigabytes heeft niet meer geheugen nodig dan een paar vensters.")],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: open every shard with `np.memmap`, list the start of every window, and read a window and the id after it.",
                      "nl": "Vervang de drie `...`: open elke shard met `np.memmap`, maak een lijst van het begin van elk venster, en lees een venster en de id erna."},
            "wrong": {"en": _CLS + ". Windows of `block` ids side by side in every shard, as long as one more id follows, with as target every window one id further, as int64?",
                      "nl": _CLS_NL + ". Vensters van `block` ids naast elkaar in elke shard, zolang er nog één id volgt, met als target elk venster één id verder, als int64?"},
            "almost": {"en": "Right for the first example. But " + _CLS[0].lower() + _CLS[1:] + ". Does your class use `block`, `dtype` and every file of `paths`?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _CLS_NL[0].lower() + _CLS_NL[1:] + ". Gebruikt je class `block`, `dtype` en elk bestand van `paths`?"},
            "correct": {"en": "Your Dataset reads windows straight from the shards on disk: only the ids of a window are read when a DataLoader asks for it. It is in your project now, as `{saved}`.",
                        "nl": "Je Dataset leest vensters rechtstreeks uit de shards op de schijf: alleen de ids van een venster worden gelezen als een DataLoader erom vraagt. Hij staat nu in je project, als `{saved}`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("write-shards", _W_START, "empty", None),
    ("write-shards", _w(), "correct", None),
    ("write-shards", _W_HEAD + "    dtype = np.uint16 if vocab <= 65536 else np.uint32\n    paths = []\n    k = 0\n"
     "    while k * size < len(ids):\n        paths.append(f\"{folder}/shard_{k}.bin\")\n"
     "        np.array(ids[k * size:(k + 1) * size], dtype=dtype).tofile(paths[-1])\n        k += 1\n    return paths\n", "correct", None),
    ("write-shards", _w(loop=_W_LOOP.replace(", dtype=dtype)", ")")), "wrong", "int64"),
    ("write-shards", _w(kind="    dtype = np.uint16\n"), "error", "too-small"),
    ("write-shards", _w(kind=_W_TYPE.replace("vocab <= 2 ** 16", "vocab < 2 ** 16")), "wrong", "below"),
    ("write-shards", _w(loop=_W_LOOP.replace("range(0, len(ids), size)", "range(0, len(ids) - size + 1, size)")), "wrong", "tail-lost"),
    ("write-shards", _w(loop=_W_LOOP.replace("shard_{len(paths)}.bin", "shard_{start}.bin")), "wrong", "names"),
    ("write-shards", _w(kind="    dtype = np.uint32\n"), "wrong", None),
    ("write-shards", _w(loop=_W_LOOP.replace("shard_{len(paths)}.bin", "shard_{len(paths):04d}.bin")), "wrong", None),
    ("token-windows", _T_START, "empty", None),
    ("token-windows", _t(), "correct", None),
    ("token-windows", _t(get="\n    def __getitem__(self, i):\n        k, s = self.starts[i]\n"
                             "        window = self.shards[k][s:s + self.block + 1]\n"
                             "        return torch.tensor(window[:-1].tolist()), torch.tensor(window[1:].tolist())\n"), "correct", None),
    ("token-windows", _t(get=_T_GET.replace("        return ids[:-1], ids[1:]\n", "        return ids[:-1], ids[:-1]\n")), "wrong", "no-shift"),
    ("token-windows", _t(get=_T_GET.replace("s + self.block + 1]", "s + self.block]")), "wrong", "one-short"),
    ("token-windows", _t(index=_T_INDEX.replace("range(0, len(self.shards[k]) - block, block)", "range(len(self.shards[k]) - block)")), "wrong", "every-start"),
    ("token-windows", _t(index=_T_INDEX.replace("len(self.shards[k]) - block, block)", "len(self.shards[k]) - block + 1, block)")), "wrong", "last-short"),
    ("token-windows", _t(get=_T_GET.replace(".tolist())", ")")), "wrong", "no-long"),
    ("token-windows", _t(get=_T_GET.replace(".tolist())", ", dtype=torch.long)")), "correct", None),
    ("token-windows", _t(opened="        self.shards = [np.fromfile(p, dtype=dtype) for p in paths]\n"), "almost", "no-memmap"),
    ("token-windows", _t(opened="        # np.memmap would read less\n        self.shards = [np.fromfile(p, dtype=dtype) for p in paths]\n"), "almost", "no-memmap"),
    ("token-windows", _t(get=_T_GET.replace(".tolist())", ".tolist(), dtype=torch.int32)")), "wrong", "no-long"),
]
