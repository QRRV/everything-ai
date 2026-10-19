# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson py-generators: two generator functions written in a box. The check collects the values
each one hands out, each copied the moment it comes out, and compares them with the course's solution."""
import re

from everything_ai.check import FunctionBox, Mistake, collect_generator, handed_out

# A `...` that is still there, found before the code runs: a loop whose body is `...` hands out nothing,
# and that should not be explained as some other mistake.
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your function. Replace it with your own lines.",
                nl="Er staat nog een `...` in je functie. Vervang die door je eigen regels.")
_APPEND = Mistake("append-assign", syntax=r"=\s*\w+\s*\.\s*append\s*\(",
                  en="`append` changes the list itself and gives back nothing, so `batch = batch.append(item)` puts nothing, `None`, in `batch`. Write only `batch.append(item)`, on a line of its own.",
                  nl="`append` verandert de lijst zelf en geeft niets terug, dus zet `batch = batch.append(item)` niets, `None`, in `batch`. Schrijf alleen `batch.append(item)`, op een eigen regel.")


def _no_yield(got, text) -> bool:
    """No `yield` outside comments: the function gives everything back at once."""
    return re.search(r"^[^#\n]*\byield\b", text, re.MULTILINE) is None


_TEXTS_NOTHING = {
    "prints": {"en": "Your function shows each value with `print`, but hands out nothing. Use `yield` instead of `print`: then the loop that asks gets the value.",
               "nl": "Je functie laat elke waarde zien met `print`, maar geeft niets af. Gebruik `yield` in plaats van `print`: dan krijgt de loop die erom vraagt de waarde."},
    "no_return": {"en": "Your function hands out nothing: there is no `yield` in it, or it never gives anything back. Write `yield` and the value inside the loop.",
                  "nl": "Je functie geeft niets af: er staat geen `yield` in, of hij geeft nooit iets terug. Schrijf `yield` en de waarde in de loop."},
    "unreached": {"en": "With `{call}`, your generator function hands out nothing: Python never gets to a `yield`. Look at the spaces in front of each line: is every line where it belongs?",
                  "nl": "Met `{call}` geeft je generator function niets af: Python komt nooit bij een `yield`. Kijk naar de spaties voor elke regel: staat elke regel waar hij hoort?"},
}

# The lesson is done when these are correct. "when-runs" is the choice question in the lesson.
REQUIRED = ["squares", "when-runs", "batches"]

_SQUARES = "def squares(n):\n    for i in range(n):\n        yield i * i"
_BATCHES_START = "def batches(items, size):\n    batch = []\n    for item in items:\n        ...\n"
_BATCHES = ("def batches(items, size):\n    batch = []\n    for item in items:\n        batch.append(item)\n"
            "        if len(batch) == size:\n            yield batch\n            batch = []\n"
            "    if len(batch) > 0:\n        yield batch")
_LAST = "\n    if len(batch) > 0:\n        yield batch"

EXERCISES = {
    "squares": FunctionBox(
        name="squares",
        function="squares",
        start="def squares(n):\n    ...\n",
        solution=_SQUARES,
        visible=[(4,)],
        hidden=[(1,), (0,), (6,)],
        collect=collect_generator,
        show=handed_out,
        first=[_DOTS],
        mistakes=[
            Mistake("return-not-yield", wrong="def squares(n):\n    for i in range(n):\n        return i * i",
                    en="`{call}` hands out {got}. `return` ends the function in the first round of the loop. `yield` hands out a value and goes on: write `yield` instead of `return`.",
                    nl="`{call}` geeft {got}. `return` beëindigt de functie in de eerste ronde van de loop. `yield` geeft een waarde af en gaat door: schrijf `yield` in plaats van `return`."),
            Mistake("yield-then-return", wrong="def squares(n):\n    for i in range(n):\n        yield i * i\n        return",
                    en="`{call}` hands out {got}, and it should hand out {want}. After the first `yield`, the `return` ends the function. Leave the `return` out: the loop ends by itself.",
                    nl="`{call}` geeft {got} af, en dat moet {want} zijn. Na de eerste `yield` beëindigt de `return` de functie. Laat de `return` weg: de loop stopt vanzelf."),
            Mistake("yield-list", wrong="def squares(n):\n    result = []\n    for i in range(n):\n        result.append(i * i)\n    yield result",
                    en="`{call}` hands out one value, the whole list. A generator hands out the values one at a time: `yield` each square inside the loop, without a list.",
                    nl="`{call}` geeft één waarde af, de hele lijst. Een generator geeft de waarden één voor één af: `yield` elk kwadraat in de loop, zonder lijst."),
            Mistake("not-squared", wrong="def squares(n):\n    for i in range(n):\n        yield i",
                    en="`{call}` hands out {got}, and it should hand out {want}. Yours hands out the numbers themselves, not their squares: hand out `i * i`.",
                    nl="`{call}` geeft {got} af, en dat moet {want} zijn. De jouwe geeft de getallen zelf af, niet hun kwadraten: geef `i * i` af."),
        ],
        form=[Mistake("no-yield", state="almost", matches=_no_yield,
                      en="Right values, but your function builds them all first and gives them back at once. A generator function hands them out one at a time, with `yield` in the loop: then it never has to hold all of them.",
                      nl="Goede waarden, maar je functie maakt ze eerst allemaal en geeft ze in één keer terug. Een generator function geeft ze één voor één af, met `yield` in de loop: dan hoeft hij ze nooit allemaal tegelijk te bewaren.")],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a loop over `range(n)` that hands out `i * i` with `yield`.",
                      "nl": "Vervang de `...` door een loop over `range(n)` die met `yield` de waarde `i * i` afgeeft."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def squares(n):`, and under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def squares(n):`, met daaronder, ingesprongen, je regels."},
            "wrong": {"en": "`{call}` hands out {got}, and it should hand out {want}: the squares of 0, 1, 2 and 3. Does your loop go over `range(n)`, and hand out `i * i`?",
                      "nl": "`{call}` geeft {got} af, en dat moet {want} zijn: de kwadraten van 0, 1, 2 en 3. Gaat je loop over `range(n)`, en geeft hij `i * i` af?"},
            "almost": {"en": "Right for the example. But `{call}` hands out {got}, and it should hand out {want}. Does your function use `n`, for every `n`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got} af, en dat moet {want} zijn. Gebruikt je functie `n`, voor elke `n`?"},
            "correct": {"en": "Your generator function is right in all {n} cases: it hands out the squares one at a time.",
                        "nl": "Je generator function klopt in alle {n} gevallen: hij geeft de kwadraten één voor één af."},
            **_TEXTS_NOTHING,
        },
    ),
    "batches": FunctionBox(
        name="batches",
        function="batches",
        start=_BATCHES_START,
        solution=_BATCHES,
        visible=[([1, 2, 3, 4, 5], 2)],
        hidden=[([1, 2, 3, 4], 2), ([7], 3), ([], 2), (["a", "b", "c"], 1), ([1, 2, 3, 4, 5, 6, 7], 3)],
        collect=collect_generator,
        show=handed_out,
        first=[_DOTS],
        mistakes=[
            _APPEND,
            Mistake("last-lost", wrong=_BATCHES.replace(_LAST, ""),
                    en="`{call}` hands out {got}, and it should hand out {want}: the last batch is missing. When the loop ends, `batch` can still hold items. After the loop, hand those out too.",
                    nl="`{call}` geeft {got} af, en dat moet {want} zijn: de laatste batch ontbreekt. Als de loop klaar is, kan `batch` nog items bevatten. Geef die na de loop ook af."),
            Mistake("empty-last", state="almost", wrong=_BATCHES.replace("    if len(batch) > 0:\n        yield batch", "    yield batch"),
                    en="Right for the example. But `{call}` hands out {got}, and it should hand out {want}. Yours hands out an empty batch, `[]`, at the end. After the loop, hand out `batch` only if it has items: `if len(batch) > 0:`.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft {got} af, en dat moet {want} zijn. De jouwe geeft aan het eind een lege batch af, `[]`. Geef `batch` na de loop alleen af als hij items heeft: `if len(batch) > 0:`."),
            Mistake("no-reset", wrong=_BATCHES.replace("            yield batch\n            batch = []", "            yield batch"),
                    en="`{call}` hands out {got}, and it should hand out {want}. After you hand out a full batch, `batch` keeps its items, and grows on. Start a new, empty batch right after `yield`: `batch = []`.",
                    nl="`{call}` geeft {got} af, en dat moet {want} zijn. Nadat je een volle batch afgeeft, houdt `batch` zijn items, en groeit hij door. Begin meteen na `yield` een nieuwe, lege batch: `batch = []`."),
            Mistake("reset-first", wrong=_BATCHES.replace("            yield batch\n            batch = []", "            batch = []\n            yield batch"),
                    en="`{call}` hands out {got}, and it should hand out {want}. Your function empties `batch` first, and then hands it out. Swap the two lines: first `yield batch`, then `batch = []`.",
                    nl="`{call}` geeft {got} af, en dat moet {want} zijn. Je functie maakt `batch` eerst leeg, en geeft hem daarna af. Draai de twee regels om: eerst `yield batch`, dan `batch = []`."),
            Mistake("last-in-loop", wrong=_BATCHES.replace(_LAST, "\n        if len(batch) > 0:\n            yield batch"),
                    en="`{call}` hands out {got}, and it should hand out {want}. Your last `if` is indented inside the loop, so it hands out half-full batches in every round. Move it after the loop, as far in as `for`.",
                    nl="`{call}` geeft {got} af, en dat moet {want} zijn. Je laatste `if` is ingesprongen in de loop, dus geeft hij in elke ronde halfvolle batches af. Zet hem na de loop, even ver ingesprongen als `for`."),
            Mistake("return-batch", wrong=_BATCHES.replace("            yield batch\n            batch = []", "            return batch"),
                    en="`{call}` hands out {got}. `return` ends the function at the first full batch. Hand out each batch with `yield`, so the function goes on.",
                    nl="`{call}` geeft {got}. `return` beëindigt de functie bij de eerste volle batch. Geef elke batch af met `yield`, dan gaat de functie door."),
        ],
        form=[Mistake("no-yield", state="almost", matches=_no_yield,
                      en="Right batches, but your function builds them all first and gives them back at once. A generator function hands them out one at a time, with `yield`: then the code that asks can use each batch as soon as it is full.",
                      nl="Goede batches, maar je functie maakt ze eerst allemaal en geeft ze in één keer terug. Een generator function geeft ze één voor één af, met `yield`: dan kan de code die erom vraagt elke batch gebruiken zodra hij vol is.")],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines in the loop: add the item to `batch`, and hand out `batch` once it has `size` items. Then add the lines after the loop, for what is left.",
                      "nl": "Vervang de `...` door de regels in de loop: voeg het item toe aan `batch`, en geef `batch` af zodra hij `size` items heeft. Voeg daarna de regels na de loop toe, voor wat er overblijft."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def batches(items, size):`, then `batch = []` and a loop over `items`.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def batches(items, size):`, dan `batch = []` en een loop over `items`."},
            "wrong": {"en": "`{call}` hands out {got}, and it should hand out {want}: batches of 2, and the 5 that is left in a batch of its own. Does your loop add each item to `batch`, and hand it out when `len(batch) == size`?",
                      "nl": "`{call}` geeft {got} af, en dat moet {want} zijn: batches van 2, en de 5 die overblijft in een eigen batch. Voegt je loop elk item toe aan `batch`, en geeft hij hem af als `len(batch) == size`?"},
            "almost": {"en": "Right for the example. But `{call}` hands out {got}, and it should hand out {want}. Does your function use `size`, and work for every list, also an empty one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got} af, en dat moet {want} zijn. Gebruikt je functie `size`, en werkt hij voor elke lijst, ook een lege?"},
            "correct": {"en": "Your generator function is right in all {n} cases: full batches one at a time, and what is left at the end in a last, smaller batch.",
                        "nl": "Je generator function klopt in alle {n} gevallen: volle batches één voor één, en wat aan het eind overblijft in een laatste, kleinere batch."},
            **_TEXTS_NOTHING,
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_S = "def squares(n):\n"
_B = "def batches(items, size):\n"
CASES = [
    ("squares", _S + "    ...\n", "empty", None),
    ("squares", _SQUARES, "correct", None),
    ("squares", _S + "    for i in range(n):\n        yield i ** 2", "correct", None),
    ("squares", _S + "    for i in range(n):\n        return i * i", "wrong", "return-not-yield"),
    ("squares", _S + "    for i in range(n):\n        yield i * i\n        return", "wrong", "yield-then-return"),
    ("squares", _S + "    result = []\n    for i in range(n):\n        result.append(i * i)\n    yield result", "wrong", "yield-list"),
    ("squares", _S + "    for i in range(n):\n        yield i", "wrong", "not-squared"),
    ("squares", _S + "    for i in range(n):\n        print(i * i)", "wrong", "prints"),
    ("squares", _S + "    result = []\n    for i in range(n):\n        result.append(i * i)\n    return result", "almost", "no-yield"),
    ("squares", _S + "    # use yield later\n    result = []\n    for i in range(n):\n        result.append(i * i)\n    return result", "almost", "no-yield"),
    ("squares", _S + "    for i in range(4):\n        yield i * i", "almost", None),
    ("squares", _S + "    for i in range(1, n + 1):\n        yield i * i", "wrong", None),
    ("squares", _S + "    for i in range(n):\n        if i < 0:\n            yield i * i", "wrong", "unreached"),
    ("squares", _S + "    i = 0\n    for k in range(100000):\n        for j in range(100000):\n            yield i", "slow", None),
    ("batches", _BATCHES_START, "empty", None),
    ("batches", _BATCHES, "correct", None),
    ("batches", _BATCHES.replace("if len(batch) > 0:", "if batch:"), "correct", None),
    ("batches", _BATCHES.replace(_LAST, ""), "wrong", "last-lost"),
    ("batches", _BATCHES.replace("    if len(batch) > 0:\n        yield batch", "    yield batch"), "almost", "empty-last"),
    ("batches", _BATCHES.replace("            yield batch\n            batch = []", "            yield batch"), "wrong", "no-reset"),
    ("batches", _BATCHES.replace("            yield batch\n            batch = []", "            batch = []\n            yield batch"), "wrong", "reset-first"),
    ("batches", _BATCHES.replace(_LAST, "\n        if len(batch) > 0:\n            yield batch"), "wrong", "last-in-loop"),
    ("batches", _BATCHES.replace("            yield batch\n            batch = []", "            return batch"), "wrong", "return-batch"),
    ("batches", _BATCHES.replace("batch.append(item)", "batch = batch.append(item)"), "error", "append-assign"),
    ("batches", _BATCHES.replace("if len(batch) == size:", "if len(batch) == 2:"), "almost", None),
    ("batches", _B + "    result = []\n    batch = []\n    for item in items:\n        batch.append(item)\n        if len(batch) == size:\n"
                     "            result.append(batch)\n            batch = []\n    if len(batch) > 0:\n        result.append(batch)\n    return result", "almost", "no-yield"),
    ("batches", _BATCHES_START.replace("        ...\n", "        ...\n        yield batch\n"), "wrong", "dots-left"),
]
