# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-triton-kernel: softmax_kernel, a fused softmax in Triton, one program per row. And the demo:
the softmax in five PyTorch operations against PyTorch's own fused one, and the learner's kernel, on the learner's
machine: compiled on an NVIDIA GPU, in Triton's interpreter on a CPU."""
import ast
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, Result, Run, _tb_frames, is_empty_text, run_code, with_time_limit

# Triton runs on Linux, also in WSL2; without it the lesson is read along.
HAS_TRITON = importlib.util.find_spec("triton") is not None
# The checks and the worked example run every kernel in Triton's interpreter, on the CPU: the same answer on every
# computer with Triton, and a kernel from a code box needs no source file. The demo compiles for a GPU itself.
if HAS_TRITON and os.environ.get("TRITON_INTERPRET") != "1":
    os.environ["TRITON_INTERPRET"] = "1"
    # Triton's own functions, such as tl.max, are made when triton is imported: if something imported it before,
    # they are made for the compiler, and the interpreter cannot call them. Import it again.
    for _name in [n for n in sys.modules if (n == "triton" or n.startswith("triton.")) and not n.startswith("triton._C")]:
        del sys.modules[_name]
    if any(n.startswith("triton._C") for n in sys.modules):
        importlib.import_module("triton.language")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def traffic_picture(labels: dict) -> str:
    """The trips to memory of a softmax: as five PyTorch operations, each reading its input from the GPU's memory and
    writing its output back, 5 reads and 3 writes; fused into one kernel, 1 read and 1 write. Data moves, so ink."""
    from everything_ai import chart
    ops = ["max", "−", "exp", "sum", "÷"]
    reads = [1, 1, 1, 1, 1]
    writes = [0, 1, 1, 0, 1]   # max and sum give one number per row: too small to count
    p = chart.Panel((0, 10), (0, 10), title=labels["title"], xlabel="", ylabel="", xticks=[], yticks=[])
    p.rect(0.2, 4.6, 9.8, 5.6, "ink-3")
    p.text(5, 5.1, labels["memory"], "ink", "middle")
    for i, (name, r, w) in enumerate(zip(ops, reads, writes)):
        x = 1.1 + i * 1.95
        p.rect(x - 0.7, 7.6, x + 0.7, 8.8, "data")
        p.text(x, 8.2, name, "ink", "middle")
        p.arrow(x - 0.3, 5.7, x - 0.3, 7.5, "ink-2", 1.6)
        if w:
            p.arrow(x + 0.3, 7.5, x + 0.3, 5.7, "ink-2", 1.6)
    p.text(5, 9.6, labels["five"], "ink-2", "middle")
    p.rect(2.5, 1.2, 7.5, 2.4, "data")
    p.text(5, 1.8, labels["fused"], "ink", "middle")
    p.arrow(4.4, 4.5, 4.4, 2.5, "ink-2", 1.6)
    p.arrow(5.6, 2.5, 5.6, 4.5, "ink-2", 1.6)
    p.text(5, 0.4, labels["one"], "ink-2", "middle")
    svg = chart.figure([p], alt=labels["alt"], width=330, height=300)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- the demo ---------------------------------------------------------------------------------------------------------------

ROWS, COLS = 8192, 1024        # on a CPU: 32 MB of float32 numbers, more than its caches
GPU_ROWS = 65536                # on a GPU: 256 MB, more than the largest L2 caches, of 40 to 72 MB
SMALL = (64, 1024)             # for the interpreter, which runs every program in Python


def agreement(error: float, lang: str) -> str:
    """How close the learner's kernel came to torch.softmax, in words."""
    if error < 1e-6:
        return {"en": "to within a millionth", "nl": "op minder dan een miljoenste na"}[lang]
    return {"en": f"to within {number(error, 6, lang)}", "nl": f"op {number(error, 6, lang)} na"}[lang]


def unfused(x):
    """Softmax over the rows as five PyTorch operations, each one a trip to memory and back."""
    import torch
    m = x.max(dim=1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=1, keepdim=True)


def _sync(where: str):
    import torch
    if where == "cuda":
        torch.cuda.synchronize()


def _timed(fn, where: str, runs: int = 5) -> float:
    """The fastest of a few runs of fn, in seconds, after one run that is not timed."""
    fn()
    _sync(where)
    best = float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        _sync(where)
        best = min(best, time.perf_counter() - t)
    return max(best, 1e-9)


def _launch(kernel, x):
    import torch
    import triton
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    kernel[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out


def _definitions(code: str) -> str:
    """The learner's imports and definitions, without the lines that run something, such as a try-out of softmax on a
    tensor of the CPU."""
    lines = code.splitlines()
    keep = []
    for node in ast.parse(code).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or (
                isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(isinstance(n, ast.Call) for n in ast.walk(node))):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            keep.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n\n\n".join(keep) + "\n"


# Added after the learner's definitions, in a file of its own: launches softmax_kernel on the GPU, compiled, and prints
# its fastest time of a few runs and its largest difference with torch.softmax.
_ON_GPU = '''

import json as _json
import torch
import triton
import time as _time


def _launch(x):
    out = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](x, out, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))
    return out


torch.manual_seed(0)
_x = torch.randn({rows}, {cols}, device="cuda")
_launch(_x)
torch.cuda.synchronize()
_best = float("inf")
for _ in range(5):
    _t = _time.perf_counter()
    _launch(_x)
    torch.cuda.synchronize()
    _best = min(_best, _time.perf_counter() - _t)
print(_json.dumps({{"seconds": _best, "error": float((_launch(_x) - torch.softmax(_x, dim=1)).abs().max())}}))
'''


def _on_gpu(code: str, rows: int = ROWS, cols: int = COLS, env: dict | None = None):
    """The learner's kernel, compiled for the GPU and timed, in a Python process of its own: (seconds, largest
    difference with torch.softmax), or None when it stops. Not in this process: it runs Triton's interpreter for the
    checks, and a kernel compiled here would call the interpreter's tl.max and tl.sum, made when triton was imported. And
    Triton reads a kernel's source from its file, so the code goes into a file first."""
    folder = paths.WORK / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "softmax_kernel.py"
    try:
        path.write_text(_definitions(code) + _ON_GPU.format(rows=rows, cols=cols), encoding="utf-8")
        if env is None:
            env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
        done = subprocess.run([sys.executable, str(path)], cwd=folder, env=env, capture_output=True, text=True, timeout=300)
        found = json.loads(done.stdout.strip().splitlines()[-1])
        return float(found["seconds"]), float(found["error"])
    except (OSError, SyntaxError, ValueError, IndexError, KeyError, TypeError, subprocess.SubprocessError):
        return None


def demo(code: str) -> dict | str:
    """The five operations, PyTorch's softmax and the learner's kernel, timed on the learner's machine: each with its
    seconds and the GB a second its one read and one write would take. On an NVIDIA GPU all on ROWS x COLS, the kernel
    compiled in a process of its own; on a CPU the two PyTorch ones on ROWS x COLS and the kernel, in the interpreter,
    on SMALL. "error" when the kernel stops."""
    import torch
    where = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rows = GPU_ROWS if where == "cuda" else ROWS
    x = torch.randn(rows, COLS, device=where)
    gb = 2 * x.numel() * 4 / 1e9
    out = {"device": where, "rows": []}
    out["rows"].append(("unfused", _timed(lambda: unfused(x), where), gb))
    out["rows"].append(("torch", _timed(lambda: torch.softmax(x, dim=1), where), gb))
    if where == "cuda":
        timed = _on_gpu(code, rows=rows)
        if timed is None:
            return "error"
        seconds, out["error"] = timed
        shape = (rows, COLS)
    else:
        try:
            run = run_code(code)
            kernel = run.names.get("softmax_kernel") if run.error is None else None
            if kernel is None:
                return "error"
            mine = torch.randn(*SMALL)
            seconds = with_time_limit(lambda: _timed(lambda: _launch(kernel, mine), where, runs=2), 60)
            out["error"] = float((_launch(kernel, mine) - torch.softmax(mine, dim=1)).abs().max())
        except Exception:
            return "error"
        shape = SMALL
    out["rows"].append(("yours", seconds, 2 * shape[0] * shape[1] * 4 / 1e9))
    out["small"] = tuple(shape)
    return out


# --- exercise 1: softmax_kernel ----------------------------------------------------------------------------------------------

_HEAD = ("import torch\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\n"
         "def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):\n"
         "    # the row of this program, and the places of its numbers in the row\n"
         "    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n")
_TAIL = ("\n\ndef softmax(x):\n    out = torch.empty_like(x)\n    n_rows, n_cols = x.shape\n"
         "    softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))\n    return out\n")
_START = _HEAD + "    # read the row once, compute its softmax on the chip, and write it once\n    ...\n" + _TAIL

_BODY = ('    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))\n'
         "    x = x - tl.max(x, axis=0)\n    e = tl.exp(x)\n"
         "    tl.store(out_ptr + row * n_cols + cols, e / tl.sum(e, axis=0), mask=mask)\n")

# The check's own launcher, after the learner's code: the same as theirs, but every tensor sits inside a fence, so that a
# kernel that reads outside its rows gets nan, and one that writes outside them is caught, instead of damaging memory
# that belongs to something else. The fence reaches past every likely wrong place, such as a row times BLOCK or a
# column times n_cols: a read past it can stop Python itself. Its places in `out` hold a number no softmax writes.
_GUARD = '''


def softmax(x):
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    fence = (n_rows + 4 + block) * block
    x_all = torch.full((2 * fence + x.numel(),), float("nan"))
    x_all[fence:fence + x.numel()] = x.reshape(-1)
    out_all = torch.full_like(x_all, -12345.0)
    out_all[fence:fence + x.numel()] = float("nan")
    softmax_kernel[(n_rows,)](x_all[fence:], out_all[fence:], n_cols, BLOCK=block)
    if not ((out_all[:fence] == -12345.0).all() and (out_all[fence + x.numel():] == -12345.0).all()):
        return "outside"
    return out_all[fence:fence + x.numel()].reshape(n_rows, n_cols).clone()
'''


def _b(body: str = _BODY) -> str:
    return _HEAD + body + _TAIL


class _Named:
    """A tensor of a test case, written in the check's messages the way the lesson writes it."""

    def __init__(self, shown: str, make):
        self.shown, self.make = shown, make

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        import torch
        torch.manual_seed(0)
        return self.make()


def _t(shown: str):
    import torch   # noqa: F401  (the lines below use it)
    return _Named(shown, lambda: eval(shown, {"torch": __import__("torch")}))


def _close(got, want) -> bool:
    try:
        import torch
        return (isinstance(got, torch.Tensor) and tuple(got.shape) == tuple(want.shape)
                and bool(torch.allclose(got.float(), want.float(), atol=1e-5, equal_nan=True)))
    except Exception:
        return False


def _show(v) -> str:
    try:
        import torch
    except ImportError:
        return repr(v)
    if isinstance(v, torch.Tensor):
        rows = v.tolist() if v.dim() == 2 else [v.tolist()]
        fmt = lambda a: "nan" if a != a else f"{a:.4f}".rstrip("0").rstrip(".")
        if v.numel() <= 12:
            return "[" + ", ".join("[" + ", ".join(fmt(a) for a in r) + "]" for r in rows) + "]"
        return f"a {' × '.join(str(s) for s in v.shape)} tensor whose first row starts [{', '.join(fmt(a) for a in rows[0][:3])}, …]"
    return repr(v)


def _masks_everywhere(text: str) -> bool:
    """Whether every tl.load and tl.store in the box has a mask, by name or in its place."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return True   # the check tells about the syntax itself
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("load", "store") \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl":
            place = 2 if node.func.attr == "store" else 1
            if not any(k.arg == "mask" for k in node.keywords) and len(node.args) <= place:
                return False
    return True


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_MASK = Mistake("no-mask", matches=lambda got, text: not _masks_everywhere(text),
                   en="Every `tl.load` and `tl.store` needs `mask=mask`. A block has `BLOCK` places, and a row can be shorter: without the mask, your kernel reads or writes past the end of its row, in memory that is not its own. The check does not run it like this.",
                   nl="Elke `tl.load` en `tl.store` heeft `mask=mask` nodig. Een block heeft `BLOCK` plekken, en een rij kan korter zijn: zonder de mask leest of schrijft je kernel voorbij het eind van zijn rij, in geheugen dat niet van hem is. De check draait hem zo niet.")
_NO_STORE = Mistake("no-store", matches=lambda got, text: "tl.store" not in text,
                    en="Your kernel has no `tl.store`, so it never writes a result. The check runs your kernel with its own copy of `softmax`: the softmax has to happen inside `softmax_kernel`, between one `tl.load` and one `tl.store`.",
                    nl="Je kernel heeft geen `tl.store`, dus hij schrijft nooit een uitkomst weg. De check draait je kernel met een eigen kopie van `softmax`: de softmax moet in `softmax_kernel` gebeuren, tussen één `tl.load` en één `tl.store`.")


def _def_kept(text: str) -> bool:
    """Whether the box still makes softmax_kernel with the def line of the start: the check launches it itself."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return True   # the check tells about the syntax itself
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "softmax_kernel":
            names = [a.arg for a in node.args.args]
            return len(names) == 4 and names[3] == "BLOCK" and bool(node.decorator_list)
    return False


def _calls(text: str, what: str) -> int:
    """How many times the box calls tl.<what>."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == what and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl")


_DEF_LINE = Mistake("def-line", matches=lambda got, text: not _def_kept(text),
                    en="Keep the two lines `@triton.jit` and `def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):` from the start of the box: the check launches `softmax_kernel` itself, with `x`, `out`, `n_cols` and `BLOCK`.",
                    nl="Houd de twee regels `@triton.jit` en `def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):` uit het begin van het vak: de check start `softmax_kernel` zelf, met `x`, `out`, `n_cols` en `BLOCK`.")
_ONCE = Mistake("twice", matches=lambda got, text: _calls(text, "load") > 1 or _calls(text, "store") > 1, state="almost",
                en="Your kernel gives the right softmax, but it has more than one `tl.load` or `tl.store`, and each one is a trip to memory. Load the row once, keep it on the chip for the max, the powers and the sum, and store the result once.",
                nl="Je kernel geeft de goede softmax, maar hij heeft meer dan één `tl.load` of `tl.store`, en elk ervan is een ritje naar het geheugen. Laad de rij één keer, houd hem op de chip voor het maximum, de machten en de som, en sla de uitkomst één keer op.")


def _has_nan(got) -> bool:
    try:
        import torch
        return isinstance(got, torch.Tensor) and bool(got.isnan().any())
    except Exception:
        return False


_NO_TRITON = {"en": "Triton does not run on this computer, so the check cannot run your kernel. Compare it with the solution under the hint; the lesson is done without this exercise.",
              "nl": "Triton draait niet op deze computer, dus de check kan je kernel niet draaien. Vergelijk hem met de oplossing onder de hint; de les is af zonder deze oefening."}


class _KernelBox(FunctionBox):
    """A FunctionBox for a Triton kernel: the check runs the learner's kernel with its own launcher, `_GUARD`."""

    def check(self, text, inputs=None) -> Result:
        if not HAS_TRITON:
            return Result("empty", _NO_TRITON)   # nothing the check can run; not an error in the learner's code
        if text is None or not isinstance(text, str) or is_empty_text(text, self.start):
            return super().check(text, inputs)
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, m.text, mistake=m.key)
        return super().check(text + _GUARD, inputs)

    def _fn(self, code: str):
        return super()._fn(code + _GUARD)

    def _plain_error(self, run: Run, text: str):
        """Triton's interpreter wraps an error inside the kernel, and its traceback ends at the check's own launch line:
        tell the learner's own error, at the line of their kernel where it happened."""
        e, line = run.error, run.line
        while type(e).__name__ == "InterpreterError" and e.__cause__ is not None:
            e = e.__cause__
            frames = [f for f in _tb_frames(e) if f[0] == "<your code>"]
            line = frames[-1][1] if frames else line
        return super()._plain_error(Run(error=e, line=line), text)


EXERCISES = {
    "softmax-kernel": _KernelBox(
        name="softmax-kernel",
        function="softmax",
        start=_START,
        solution=_b(),
        visible=[(_t("torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])"),)],
        hidden=[(_t("torch.tensor([[1000.0, 1001.0, 999.0]])"),), (_t("torch.tensor([[-5.0, -6.0, -7.0, -8.0, -9.0]])"),),
                (_t("torch.tensor([[5.0]])"),), (_t("torch.randn(8, 100) * 10"),), (_t("torch.randn(3, 1000)"),)],
        show=_show,
        compare=_close,
        first=[_DOTS, _DEF_LINE, _NO_STORE, _NO_MASK],
        form=[_ONCE],
        mistakes=[
            Mistake("outside", matches=lambda got, text: isinstance(got, str) and got == "outside",
                    en="With `{call}`, your kernel writes outside the rows of `out`. Does each program store at its own row, `out_ptr + row * n_cols + cols`, with `mask=mask`?",
                    nl="Met `{call}` schrijft je kernel buiten de rijen van `out`. Schrijft elk programma naar zijn eigen rij, `out_ptr + row * n_cols + cols`, met `mask=mask`?"),
            Mistake("no-max", wrong=_b(_BODY.replace("    x = x - tl.max(x, axis=0)\n", "")), state="almost",
                    en="Right for the example. But `{call}` gives `{got}`: with large numbers, `tl.exp` overflows to `inf`, and `inf / inf` is `nan`. Take the largest number of the row off first, as in the log-sum-exp.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`: bij grote getallen loopt `tl.exp` over naar `inf`, en `inf / inf` is `nan`. Trek eerst het grootste getal van de rij af, zoals bij de log-sum-exp."),
            Mistake("pad-zero", wrong=_b(_BODY.replace('other=float("-inf")', "other=0.0")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the places past the end of the row are read as 0, and then count in your sum, as `exp(0 - max)`. Load them with `other=float(\"-inf\")`: minus infinity never wins the max, and its `exp` is 0.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de plekken voorbij het eind van de rij worden als 0 gelezen, en tellen dan mee in je som, als `exp(0 - max)`. Laad ze met `other=float(\"-inf\")`: min oneindig wint nooit het maximum, en zijn `exp` is 0."),
            Mistake("first-row", wrong=_b(_BODY.replace("row * n_cols + ", "")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every program works on the first row, and the other rows of `out` are never written (the check fills them with `nan`). Add `row * n_cols` to the places, so that program `row` reads and writes its own row.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk programma werkt aan de eerste rij, en de andere rijen van `out` worden nooit geschreven (de check vult ze met `nan`). Tel `row * n_cols` op bij de plekken, zodat programma `row` zijn eigen rij leest en schrijft."),
            Mistake("nan", matches=lambda got, text: _has_nan(got),
                    en="`{call}` gives `{got}`, and it should give `{want}`. A `nan` comes from a place outside the rows, which the check fills with `nan`: your kernel reads there, or leaves a place of `out` unwritten. Does program `row` read and write exactly `row * n_cols + cols`?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een `nan` komt van een plek buiten de rijen, die de check met `nan` vult: je kernel leest daar, of laat een plek van `out` leeg. Leest en schrijft programma `row` precies `row * n_cols + cols`?"),
            Mistake("no-divide", wrong=_b(_BODY.replace("e / tl.sum(e, axis=0)", "e")),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your rows do not add up to 1. Divide the powers of `e` by their sum, `tl.sum(e, axis=0)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rijen tellen niet op tot 1. Deel de machten van `e` door hun som, `tl.sum(e, axis=0)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the body of your kernel: read the row, compute its softmax, and write it.",
                      "nl": "Vervang de `...` door de body van je kernel: lees de rij, reken zijn softmax uit, en schrijf hem weg."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "no_function": {"en": "Your box does not make the function `softmax` any more. Keep the lines of `softmax` from the start of the box: the check uses its own copy, and you may not remove it.",
                            "nl": "Je vak maakt de functie `softmax` niet meer. Houd de regels van `softmax` uit het begin van het vak: de check gebruikt een eigen kopie, en je mag hem niet weghalen."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each program read its own row with the mask, take the largest number off, and divide the powers of `e` by their sum?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Leest elk programma zijn eigen rij met de mask, trekt het er het grootste getal van af, en deelt het de machten van `e` door hun som?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your kernel work for rows of any length, with any numbers?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je kernel voor rijen van elke lengte, met elke getallen?"},
            "correct": {"en": "Your kernel is right in all {n} cases: one read and one write per row, rows of 1 to 1,000 numbers, and numbers of a thousand without overflow.",
                        "nl": "Je kernel klopt in alle {n} gevallen: één keer lezen en één keer schrijven per rij, rijen van 1 tot 1.000 getallen, en getallen van duizend zonder overflow."},
        },
    ),
}

# The lesson is done when these are correct. "grid" is the choice question, "why-fused" the open question. Without
# Triton, the kernel cannot run, and the lesson is done without it.
REQUIRED = (["softmax-kernel"] if HAS_TRITON else []) + ["grid", "why-fused"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("softmax-kernel", _START, "empty", None),
    ("softmax-kernel", _b(), "correct", None),
    ("softmax-kernel", _b('    offs = row * n_cols + cols\n    x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))\n'
                          "    m = tl.max(x)\n    e = tl.exp(x - m)\n    s = tl.sum(e)\n    tl.store(out_ptr + offs, e / s, mask=mask)\n"), "correct", None),
    ("softmax-kernel", _b(_BODY.replace(", mask=mask, other=float(\"-inf\")", "")), "wrong", "no-mask"),
    ("softmax-kernel", _b(_BODY.replace("e / tl.sum(e, axis=0), mask=mask)", "e / tl.sum(e, axis=0))")), "wrong", "no-mask"),
    ("softmax-kernel", _b(_BODY.replace("    x = x - tl.max(x, axis=0)\n", "")), "almost", "no-max"),
    ("softmax-kernel", _b(_BODY.replace('other=float("-inf")', "other=0.0")), "wrong", "pad-zero"),
    ("softmax-kernel", _b(_BODY.replace(', other=float("-inf")', "")), "wrong", "pad-zero"),
    ("softmax-kernel", _b(_BODY.replace("row * n_cols + ", "")), "wrong", "first-row"),
    ("softmax-kernel", _b(_BODY.replace("e / tl.sum(e, axis=0)", "e")), "wrong", "no-divide"),
    ("softmax-kernel", _b(_BODY.replace("out_ptr + row * n_cols", "out_ptr + row * BLOCK")), "wrong", "outside"),
    ("softmax-kernel", _b(_BODY.replace("x_ptr + row * n_cols", "x_ptr + row * BLOCK")), "wrong", "nan"),
    ("softmax-kernel", _HEAD + "    pass\n" + _TAIL.replace("    out = torch.empty_like(x)\n", "    return torch.softmax(x, dim=1)\n"), "wrong", "no-store"),
    ("softmax-kernel", _b(_BODY.replace("x_ptr + row * n_cols + cols", "x_ptr + cols * n_cols + row")), "wrong", "nan"),
    ("softmax-kernel", _b().replace("mask = cols < n_cols", "mask = cols <= n_cols"), "wrong", "outside"),
    ("softmax-kernel", _b().replace("softmax_kernel", "my_kernel"), "wrong", "def-line"),
    ("softmax-kernel", _b(_BODY.replace("    x = x - tl.max(x, axis=0)\n    e = tl.exp(x)\n",
                                        '    m = tl.max(x, axis=0)\n    e = tl.exp(tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf")) - m)\n')), "almost", "twice"),
]
