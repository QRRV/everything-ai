# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-eval-chat: avoids, whether a story leaves out a word; sentences, how many sentences a
story has; held_out, the prompts whose requested words were never a training instruction; and a choice. The demo
scores the answers of three models, before post-training, after SFT and after DPO, to 200 prompts of three kinds of
verifiable instruction (with the words, without a word, at most a number of sentences) with the learner's checks: the
course's models from data/course-eval-chat.json, or the learner's own, which then answer the same prompts."""
import importlib.util
import json
import random
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _checks("llm-dpo", "everything_ai_checks_llm_dpo_for_eval_chat")
T, S = D.T, D.S
BUILDS_ON = D.BUILDS_ON   # the learner's three models, their tokenizer, and words_used
projects.builds_on(*BUILDS_ON)

ANSWERS = "data/course-eval-chat.json"   # the course's three models' answers to every prompt
WITHOUT, SENTENCES = 50, 50              # prompts of the two kinds the models were never trained on
MODELS = ("base", "sft", "dpo")
KINDS = ("with", "clean", "without", "sentences")   # "clean": the prompts of "with" that are held out


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def prompts() -> list:
    """(kind, what it asks, the prompt) for every prompt: the test instructions of llm-sft ("with", their words), then
    WITHOUT asking for a story without one word of the list of llm-sft ("without"), then SENTENCES asking for a story
    of at most 1 to 5 sentences ("sentences"); all in the course's template, its last line "Story:"."""
    out = [("with", list(words), T.instruction(list(words))) for words in T.tests()]
    rng = random.Random(3)
    for w in rng.sample(sorted(T.WORDS), WITHOUT):
        out.append(("without", [w], f"Write a story without the word: {w}.\nStory:\n"))
    for k in range(SENTENCES):
        n = 1 + k % 5
        out.append(("sentences", n, f"Write a story of at most {n} sentence{'s' if n > 1 else ''}.\nStory:\n"))
    return out


def course() -> dict:
    return json.loads((paths.ROOT / ANSWERS).read_text(encoding="utf-8"))


def trained(runs, encode_fast=None, decode=None) -> list:
    """The requested words of every instruction the fine-tuned models were trained on: those of the DPO pairs, and of
    every SFT run in `runs`: "course", the course's SFT run, from data/course-eval-chat.json, and "yours", the
    learner's SFT run of llm-sft, made again as llm-sft made it, on its stories with the merges of their SFT model."""
    out = [it["words"] for it in D.items()]
    if "course" in runs:
        out += course()["sft_instructions"]
    if "yours" in runs:
        merges = {tuple(pair): 256 + i for i, pair in enumerate(T.saved()["merges"])}
        out += [list(words) for words, _, _ in T.pairs(T.stories()[:-T.HELD], merges, encode_fast, decode)]
    return out


def score(answers: dict, train: list, avoids, sentences, held_out, words_used) -> dict | str:
    """For every model of `answers`, how many of its answers pass, per kind, by the learner's checks; with the number
    of prompts of each kind, and for "sentences" the mean number of sentences for every number asked. A string when a
    check stops with an error or gives no bool: "avoids", "sentences", "held_out" or "words"."""
    asked = prompts()
    try:
        clean = held_out([list(a) for k, a, _ in asked if k == "with"], [list(w) for w in train])
        keep = {tuple(sorted(w)) for w in clean}
    except Exception:
        return "held_out"
    out = {"total": {"with": 100, "clean": sum(k == "with" and tuple(sorted(a)) in keep for k, a, _ in asked),
                     "without": WITHOUT, "sentences": SENTENCES}}
    for m, said in answers.items():
        got = {k: 0 for k in KINDS}
        per = {n: [] for n in range(1, 6)}
        for (kind, arg, _), story in zip(asked, said):
            if kind == "with":
                try:
                    ok = words_used(story, list(arg))
                except Exception:
                    return "words"
                if not isinstance(ok, bool):
                    return "words"
                got["with"] += ok
                got["clean"] += ok and tuple(sorted(arg)) in keep
            elif kind == "without":
                try:
                    ok = avoids(story, arg[0])
                except Exception:
                    return "avoids"
                if not isinstance(ok, bool):
                    return "avoids"
                got["without"] += ok
            else:
                try:
                    n = sentences(story)
                except Exception:
                    return "sentences"
                if not isinstance(n, (int, float)) or isinstance(n, bool):   # the check takes 3.0 for 3 too
                    return "sentences"
                got["sentences"] += n <= arg
                per[arg].append(n)
        out[m] = {**got, "per": {n: sum(v) / len(v) for n, v in per.items() if v}}
    return out


def tools() -> dict | str:
    ran = run_code("from llm.fast_encode import encode_fast\nfrom llm.decode import decode\nfrom llm.words_used import words_used")
    return "run" if ran.error is not None else ran.names


def mine() -> dict:
    """Which models of the learner's own there are: their capstone, their SFT model, and it with their DPO adapters."""
    return {"base": T.P.saved() is not None, "sft": T.saved() is not None, "dpo": D.saved() is not None}


def ask(show) -> dict | str:
    """The learner's own models' answers to every prompt, as in data/course-eval-chat.json, for the models they have,
    and the requested words of every instruction those models were fine-tuned on (made once here: making the SFT
    run's instructions again takes half a minute); "run" when one cannot be made or answer."""
    t = tools()
    if t == "run":
        return "run"
    have = [m for m, there in mine().items() if there]
    out, runs = {}, set()
    start = time.perf_counter()
    for m in have:
        try:
            model, info, where = {"base": lambda: T.P.load("yours"), "sft": lambda: T.load("yours"), "dpo": lambda: D.load("yours")}[m]()
            if where != "yours":
                continue
            if m == "sft":
                runs.add("yours")
            if m == "dpo":
                runs.add(info["dpo"]["base"])   # the SFT model the adapters sit on: the learner's or the course's
            merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
            end = 256 + len(merges)
            said = []
            for k, (_, _, prompt) in enumerate(prompts()):
                ids, _ = T.answer(model, t["encode_fast"](prompt, merges), end, seed=k)
                said.append(t["decode"](ids, merges))
                show({"model": m, "done": k + 1, "total": len(prompts()), "seconds": time.perf_counter() - start})
            out[m] = said
        except Exception:
            return "run"
    seconds = time.perf_counter() - start
    try:
        train = trained(runs, t["encode_fast"], t["decode"])
    except Exception:
        return "run"
    return {"answers": out, "train": train, "seconds": seconds}


# --- measured: the course's models with the course's checks, from data/course-eval-chat.json ("course"), and the
# course's models of the demo runs of llm-pretrain-capstone, llm-sft and llm-dpo, which answered the same prompts
# ("yours"; trained on TinyStories, so not shared: only its numbers) ---
MEASURED = {
    "course": {"total": {"with": 100, "clean": 20, "without": 50, "sentences": 50}, "base": {"with": 1, "clean": 0, "without": 48, "sentences": 14, "per": {1: 6.3, 2: 3.8, 3: 7.8, 4: 6.1, 5: 5.2}}, "sft": {"with": 24, "clean": 1, "without": 41, "sentences": 24, "per": {1: 3.6, 2: 3.1, 3: 3.7, 4: 3.4, 5: 3.3}}, "dpo": {"with": 60, "clean": 3, "without": 18, "sentences": 29, "per": {1: 3.8, 2: 3.0, 3: 3.4, 4: 3.1, 5: 3.4}}},
    "yours": {"total": {"with": 100, "clean": 15, "without": 50, "sentences": 50}, "base": {"with": 9, "clean": 0, "without": 43, "sentences": 14, "per": {1: 6.9, 2: 5.7, 3: 5.0, 4: 10.0, 5: 7.6}}, "sft": {"with": 29, "clean": 1, "without": 38, "sentences": 3, "per": {1: 6.5, 2: 6.3, 3: 5.8, 4: 5.9, 5: 6.7}}, "dpo": {"with": 64, "clean": 9, "without": 24, "sentences": 16, "per": {1: 5.3, 2: 4.9, 3: 4.1, 4: 4.7, 5: 4.8}}, "seconds": 297.7},
}
# --- end of measured ---

REQUIRED = ["avoids", "sentences", "held-out", "why-not"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL

# --- exercise 1: avoids ------------------------------------------------------------------------------------------------

_A_HEAD = "def avoids(story, word):\n"
_A = (_A_HEAD + "    for w in story.lower().split():\n        if w.strip(\".,!?;:\\\"'\") == word:\n"
      "            return False\n    return True\n")
_A_START = (_A_HEAD + "    for w in story.lower().split():\n"
            "        # the word without the punctuation around it, as words_used sees it\n        ...\n    return True\n")
_A_SUBSTRING = _A_HEAD + "    return word not in story.lower()\n"
_A_CASE = _A.replace("story.lower().split()", "story.split()")
_A_STRIP = _A.replace("w.strip(\".,!?;:\\\"'\") == word", "w == word")
_A_EARLY = (_A_HEAD + "    for w in story.lower().split():\n        if w.strip(\".,!?;:\\\"'\") == word:\n"
            "            return False\n        return True\n    return True\n")

# --- exercise 2: sentences ---------------------------------------------------------------------------------------------

_S_HEAD = "def sentences(story):\n"
_S = (_S_HEAD + "    count = 0\n    before = False\n    for ch in story:\n        end = ch in \".!?\"\n"
      "        if end and not before:\n            count += 1\n        before = end\n    return count\n")
_S_START = (_S_HEAD + "    count = 0\n    before = False   # was the character before this one a . ! or ?\n    for ch in story:\n"
            "        end = ch in \".!?\"\n        # a sentence ends where a run of . ! ? starts\n        ...\n"
            "        before = end\n    return count\n")
_S_EVERY = _S.replace("        if end and not before:\n            count += 1\n", "        if end:\n            count += 1\n")
_S_DOTS = _S.replace("end = ch in \".!?\"", "end = ch == \".\"")
_S_SPLIT = _S_HEAD + "    return len(story.split(\".\")) - 1\n"

# --- exercise 3: held_out ----------------------------------------------------------------------------------------------

_H_HEAD = "def held_out(prompts, train):\n"
_H = (_H_HEAD + "    seen = set()\n    for words in train:\n        seen.add(tuple(sorted(words)))\n"
      "    return [words for words in prompts if tuple(sorted(words)) not in seen]\n")
_H_START = (_H_HEAD + "    seen = set()\n    for words in train:\n        # the words in one order, as a tuple: a set can hold it\n        ...\n"
            "    # the prompts whose words, in any order, were never a training instruction\n    return ...\n")
_H_ORDER = (_H_HEAD + "    seen = set()\n    for words in train:\n        seen.add(tuple(words))\n"
            "    return [words for words in prompts if tuple(words) not in seen]\n")
_H_ANY = (_H_HEAD + "    seen = set()\n    for words in train:\n        seen.update(words)\n"
          "    return [words for words in prompts if not any(w in seen for w in words)]\n")
_H_KEPT = _H.replace("not in seen]", "in seen]")


def _same(got, want) -> bool:
    """The same prompts in the same order; each as a list or a tuple of its words."""
    try:
        return len(got) == len(want) and all(not isinstance(g, str) and list(g) == list(w) for g, w in zip(got, want))
    except TypeError:
        return False


EXERCISES = {
    "avoids": FunctionBox(
        name="avoids",
        function="avoids",
        start=_A_START,
        solution=_A,
        visible=[("The kite flew over the park.", "dog")],
        hidden=[("Tom's dog barked.", "dog"), ("The dogs barked.", "dog"), ("\"Look, a cat,\" said Mia.", "cat"),
                ("Dog! Come here.", "dog"), ("", "sun"), ("A caterpillar ate a leaf.", "cat"),
                ("They went to the park.\nIt was fun.", "park"), ("\"Dog!\" Tom called.", "dog")],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("substring", wrong=_A_SUBSTRING,
                    en=_CALL + ": you look for the word anywhere in the text, so it is found inside other words, as `dog` in *dogs* or `cat` in *caterpillar*. Look among the whole words of the story, as `words_used` does: `.split()`.",
                    nl=_CALL_NL + ": je zoekt het woord overal in de tekst, dus wordt het in andere woorden gevonden, zoals `dog` in *dogs* of `cat` in *caterpillar*. Zoek tussen de hele woorden van het verhaal, zoals `words_used` doet: `.split()`."),
            Mistake("case", wrong=_A_CASE,
                    en=_CALL + ": at the start of a sentence the word has a capital, *Dog*, and the word to avoid is in small letters. Make the story small first: `story.lower()`.",
                    nl=_CALL_NL + ": aan het begin van een zin heeft het woord een hoofdletter, *Dog*, en het woord dat weg moet staat in kleine letters. Zet het verhaal eerst in kleine letters: `story.lower()`."),
            Mistake("punctuation", wrong=_A_STRIP,
                    en=_CALL + ": after `.split()`, punctuation sticks to the words: `cat,\"` is not `cat`, so you miss it. Strip it from both ends of every word, as `words_used` does.",
                    nl=_CALL_NL + ": na `.split()` plakken leestekens aan de woorden: `cat,\"` is niet `cat`, dus je mist het. Haal ze van beide kanten van elk woord, zoals `words_used` doet."),
            Mistake("early", wrong=_A_EARLY,
                    en=_CALL + ": your function looks at the first word only: the `return True` inside the loop stops it there. Say True only after the loop, when no word was the one to avoid.",
                    nl=_CALL_NL + ": je functie kijkt alleen naar het eerste woord: de `return True` in de loop stopt hem daar. Zeg True pas na de loop, als geen enkel woord het woord was dat weg moest."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: give `False` as soon as a word of the story, without its punctuation, is `word`.",
                      "nl": "Vervang de `...`: geef `False` zodra een woord van het verhaal, zonder zijn leestekens, `word` is."},
            "wrong": {"en": _CALL + ". Every word of the story in small letters, without the punctuation around it, and `False` as soon as one is `word`?",
                      "nl": _CALL_NL + ". Elk woord van het verhaal in kleine letters, zonder de leestekens eromheen, en `False` zodra er één `word` is?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Capitals, punctuation, a line break, *dogs* for *dog*, and an empty story?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Hoofdletters, leestekens, een nieuwe regel, *dogs* voor *dog*, en een leeg verhaal?"},
            "correct": {"en": "Your function checks the instruction *without the word*, and it sees words as `words_used` does: for one word, it says the opposite.",
                        "nl": "Je functie controleert de instructie *zonder het woord*, en ziet woorden zoals `words_used`: voor één woord zegt hij precies het omgekeerde."},
        },
    ),
    "sentences": FunctionBox(
        name="sentences",
        function="sentences",
        start=_S_START,
        solution=_S,
        visible=[("Tom ran. He fell... Was he hurt?",)],
        hidden=[("Wait... what?!",), ("No end here",), ("",), ("One. Two. Three. Four.",), ("\"Hi!\" said Sam.",),
                ("Tom (he was six!) ran home.",)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("every", wrong=_S_EVERY,
                    en=_CALL + ": you count every `.`, `!` and `?`, so `...` counts as three sentences and `?!` as two. Count a run of them once: where it starts, after a character that is none of them.",
                    nl=_CALL_NL + ": je telt elke `.`, `!` en `?`, dus `...` telt als drie zinnen en `?!` als twee. Tel een reeks ervan één keer: waar hij begint, na een teken dat er geen van is."),
            Mistake("dots", wrong=_S_DOTS,
                    en=_CALL + ": you count only the full stops. A sentence can also end at `!` or `?`.",
                    nl=_CALL_NL + ": je telt alleen de punten. Een zin kan ook eindigen bij `!` of `?`."),
            Mistake("split", wrong=_S_SPLIT,
                    en=_CALL + ": you count every full stop and nothing else, so `...` makes three sentences, and `!` and `?` make none. Walk through the characters, and count where a run of `.`, `!` and `?` starts.",
                    nl=_CALL_NL + ": je telt elke punt en verder niets, dus `...` maakt drie zinnen, en `!` en `?` maken er geen. Loop door de tekens, en tel waar een reeks `.`, `!` en `?` begint."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: add one to `count` where a `.`, `!` or `?` comes after a character that is none of them.",
                      "nl": "Vervang de `...`: tel één op bij `count` waar een `.`, `!` of `?` komt na een teken dat er geen van is."},
            "wrong": {"en": _CALL + ". One for every run of `.`, `!` and `?`, where it starts?",
                      "nl": _CALL_NL + ". Eén voor elke reeks `.`, `!` en `?`, waar hij begint?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". `...` and `?!`, a story without an end, and an empty one?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". `...` en `?!`, een verhaal zonder einde, en een leeg verhaal?"},
            "correct": {"en": "Your function counts sentences by a simple rule that a program can follow every time: a run of `.`, `!` and `?` ends one. It counts *Mr.* as an end too; a check need not be perfect, only the same for every model.",
                        "nl": "Je functie telt zinnen met een eenvoudige regel die een programma elke keer kan volgen: een reeks `.`, `!` en `?` sluit er één af. Ook *Mr.* telt als einde; een check hoeft niet perfect te zijn, wel voor elk model dezelfde."},
        },
    ),
    "held-out": FunctionBox(
        name="held-out",
        function="held_out",
        start=_H_START,
        solution=_H,
        visible=[([["park", "kite"], ["dog"], ["sun", "park"]], [["kite", "park"], ["cat"]])],
        hidden=[([["dog"], ["cat"]], []), ([], [["dog"]]), ([["ball", "tree"], ["tree"], ["ball"]], [["tree", "ball"], ["ball"]]),
                ([["fish", "boat"], ["boat", "fish"]], [["fish"]])],
        compare=_same,
        first=[_DOTS],
        mistakes=[
            Mistake("order", wrong=_H_ORDER,
                    en=_CALL + ": *kite and park* and *park and kite* ask for the same words, and your function tells them apart. Put the words of every instruction in one order first: `tuple(sorted(words))`.",
                    nl=_CALL_NL + ": *kite and park* en *park and kite* vragen om dezelfde woorden, en je functie houdt ze uit elkaar. Zet de woorden van elke instructie eerst in één volgorde: `tuple(sorted(words))`."),
            Mistake("any-word", wrong=_H_ANY,
                    en=_CALL + ": you leave out every prompt that shares one word with a training instruction. Then nearly nothing is left, as the same 141 words come back everywhere. Leave out only the prompts whose words, all of them, were a training instruction.",
                    nl=_CALL_NL + ": je laat elke prompt weg die één woord deelt met een trainingsinstructie. Dan blijft er bijna niets over, want dezelfde 141 woorden komen overal terug. Laat alleen de prompts weg waarvan alle woorden samen een trainingsinstructie waren."),
            Mistake("kept", wrong=_H_KEPT,
                    en=_CALL + ": you give back the prompts that *were* training instructions. The clean subset is the rest: `not in`.",
                    nl=_CALL_NL + ": je geeft de prompts terug die *wel* trainingsinstructies waren. De schone subset is de rest: `not in`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: every training instruction's words in one order into `seen`, and the prompts whose words are not in it.",
                      "nl": "Vervang de twee `...`: de woorden van elke trainingsinstructie in één volgorde in `seen`, en de prompts waarvan de woorden er niet in staan."},
            "wrong": {"en": _CALL + ". The prompts, in their order, whose words, sorted, are no training instruction's words, sorted?",
                      "nl": _CALL_NL + ". De prompts, in hun volgorde, waarvan de woorden, gesorteerd, niet de woorden van een trainingsinstructie zijn, gesorteerd?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". No training instructions, no prompts, and the same words in another order?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Geen trainingsinstructies, geen prompts, en dezelfde woorden in een andere volgorde?"},
            "correct": {"en": "Your function makes the clean subset: the prompts whose words, together, were never a training instruction. The rest of the prompt is the same for all, so the words are what has to be held out.",
                        "nl": "Je functie maakt de schone subset: de prompts waarvan de woorden samen nooit een trainingsinstructie waren. De rest van de prompt is voor allemaal gelijk, dus de woorden moeten held out zijn."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("avoids", _A_START, "empty", None),
    ("avoids", _A, "correct", None),
    ("avoids", _A_HEAD + "    words = [w.strip(\".,!?;:\\\"'\") for w in story.lower().split()]\n    return word not in words\n", "correct", None),
    ("avoids", "from llm.words_used import words_used\n\n\n" + _A_HEAD + "    return not words_used(story, [word])\n", "correct", None),
    ("avoids", _A_SUBSTRING, "wrong", "substring"),
    ("avoids", _A_CASE, "wrong", "case"),
    ("avoids", _A_STRIP, "wrong", "punctuation"),
    ("avoids", _A_EARLY, "wrong", "early"),
    ("avoids", _A.replace("story.lower().split()", "story.lower().split(\" \")"), "almost", None),   # a newline is not a space
    ("avoids", _A.replace("w.strip(", "w.rstrip("), "almost", None),
    ("sentences", _S_START, "empty", None),
    ("sentences", _S, "correct", None),
    ("sentences", _S_EVERY, "wrong", "every"),
    ("sentences", _S_DOTS, "wrong", "dots"),
    ("sentences", _S_SPLIT, "wrong", "split"),
    ("sentences", _S.replace("end = ch in \".!?\"", "end = ch in \".!\""), "wrong", None),
    ("sentences", _S_HEAD + "    return story.count(\".\")\n", "wrong", "split"),
    ("sentences", _S.replace("count = 0\n", "count = 0.0\n"), "correct", None),   # the demo takes the float too
    ("sentences", _S_HEAD + "    return sum(1 for w in story.split() if w.strip('\"').endswith(('.', '!', '?')))\n", "almost", None),
    ("held-out", _H_START, "empty", None),
    ("held-out", _H, "correct", None),
    ("held-out", _H_HEAD + "    seen = {tuple(sorted(w)) for w in train}\n    return [w for w in prompts if tuple(sorted(w)) not in seen]\n", "correct", None),
    ("held-out", _H_HEAD + "    seen = [set(w) for w in train]\n    return [w for w in prompts if set(w) not in seen]\n", "correct", None),
    ("held-out", _H_ORDER, "wrong", "order"),
    ("held-out", _H_ANY, "wrong", "any-word"),
    ("held-out", _H_KEPT, "wrong", "kept"),
    ("held-out", _H.replace("if tuple(sorted(words)) not in seen", "if tuple(words) not in seen"), "wrong", None),   # only the training side sorted
    ("held-out", _H.replace("seen.add(tuple(sorted(words)))", "seen.add(tuple(words))"), "almost", None),   # only the prompts sorted
]
