# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-model-card: card_header, the header of a model card the hub reads; limits, the kinds of a
test that got worse or are still too weak; and a choice. The demo writes the model card of a chat model from the
files of its three runs and the scores of the tests of llm-eval-chat and llm-safety-eval: the course's model, or the
learner's own when they have a DPO model, whose models then take both tests."""
import importlib.util
import json
import re
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _checks("llm-safety-eval", "everything_ai_checks_llm_safety_eval_for_card")
E, D, T, S = F.E, F.D, F.T, F.S
BUILDS_ON = F.BUILDS_ON
projects.builds_on(*BUILDS_ON)

CARD = "model_card.md"                          # where the card is saved, in work/projects/llm/
STAGES = ("base", "sft", "dpo")                 # the three runs: pretraining, SFT and DPO
CHAT = ("with", "clean", "without", "sentences")   # the kinds of the test of llm-eval-chat
LICENSES = {"cosmopedia": "Apache-2.0", "tinystories": "CDLA-Sharing-1.0", "course-dpo-answers": "CC-BY-4.0"}
# What the card of the course's model says it is for, and not for: in English, the language of the card and the model.
USES = ("Learning how a tiny chat model is trained and tested, in the lessons of Everything AI, and trying it out: "
        "it writes a short story for young children, with words you ask for.")
NOT_FOR = ("Anything people rely on: facts, advice, or a chat with children without an adult reading along. "
           "It refuses nothing, and it writes stories, whatever it is asked.")


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def _yours() -> dict | None:
    """The information files of the learner's three runs, when each sits on the one before it; else None."""
    run = {"base": T.P.saved(), "sft": T.saved(), "dpo": D.saved()}
    if any(v is None for v in run.values()) or run["sft"].get("base") != "yours" or run["dpo"].get("base") != "yours":
        return None
    return run


def source() -> str:
    """Whose chat model the card is about: the learner's when all three of its runs are their own, else the course's."""
    return "course" if _yours() is None else "yours"


def infos(source: str) -> dict:
    """The information file of each run of the chat model: the course's from data/, or the learner's own."""
    if source == "course":
        return {"base": T.P.course_info(), "sft": T.course_info(), "dpo": D.course_info()}
    return _yours()


def _solutions(checks) -> dict:
    names = {}
    for box in checks.EXERCISES.values():
        if isinstance(box, FunctionBox):
            names.update(run_code(box.solution).names)
    return names


def measured(source: str) -> dict | None:
    """The scores of the course's models in the tests of llm-eval-chat and llm-safety-eval; None for the learner's,
    who measure their own with measure()."""
    if source != "course":
        return None
    return {"chat": E.MEASURED["course"], "safety": F.MEASURED["course"]}


def measure(show) -> dict | str:
    """The learner's own models through both tests, with the course's checks: the scores as in measured(); "run" when
    their models cannot answer."""
    start = time.perf_counter()
    a = E.ask(lambda p: show({"test": "chat", **p}))
    if a == "run" or not a["answers"]:
        return "run"
    e = _solutions(E)
    e["words_used"] = run_code(T.EXERCISES["words-used"].solution).names["words_used"]
    chat = E.score(a["answers"], a["train"], e["avoids"], e["sentences"], e["held_out"], e["words_used"])
    b = F.ask(lambda p: show({"test": "safety", **p}))
    if b == "run" or not b["answers"]:
        return "run"
    f = _solutions(F)
    safety = F.counts(F.test(b["answers"], F.WORDS, f["refuses"], f["blocked"]))
    return {"chat": chat, "safety": safety, "seconds": time.perf_counter() - start}


_DATA = {
    "base": {"course": "{tokens:,} tokens of {stories:,} stories for young children of Cosmopedia (HuggingFaceTB/cosmopedia, Apache-2.0)",
             "yours": "{tokens:,} tokens of the first 200 MB of the training stories of TinyStories (roneneldan/TinyStories, CDLA-Sharing-1.0)"},
    "sft": {"course": "{pairs:,} pairs of an instruction and a story, made from {stories:,} stories for young children of Cosmopedia",
            "yours": "{pairs:,} pairs of an instruction and a story, made from stories of TinyStories"},
    "dpo": {"course": "{pairs:,} preference pairs from four answers of the course's SFT model to each of {instructions:,} instructions (course-dpo-answers, CC-BY-4.0)",
            "yours": "{pairs:,} preference pairs from four answers of the course's SFT model to each of {instructions:,} instructions (course-dpo-answers, CC-BY-4.0); the instructions ask for words of stories of Cosmopedia"},
}
_TRAINING = {
    "base": "{steps:,} steps of a batch of {batch}",
    "sft": "{steps:,} steps at a learning rate of {lr}",
    "dpo": "{steps:,} steps at a learning rate of {lr}, with beta {beta}, LoRA of rank {rank} and alpha {alpha}",
}
_NAMES = {"with": "a story with the asked words", "clean": "a story with words never asked for together in training",
          "without": "a story without a word", "sentences": "a story of at most N sentences"}


def datasets(source: str) -> list:
    """The datasets on the hub that the runs learned from, for the header: for the learner's, TinyStories, and
    Cosmopedia for the words of the DPO instructions."""
    return ["HuggingFaceTB/cosmopedia"] if source == "course" else ["roneneldan/TinyStories", "HuggingFaceTB/cosmopedia"]


def _share(k, n) -> str:
    return f"{k} of {n} ({round(100 * k / n)}%)" if n else "none"


def card(source: str, scores: dict | None, header: str, uses: str, not_for: str, found: list, enough: float, license: str) -> str:
    """The model card, in Markdown, as README.md of the model on the hub: the header, then its sections, in English, the
    language of the model."""
    info = infos(source)
    base, dpo = info["base"], info["dpo"]
    who = "the course Everything AI" if source == "course" else "a learner of the course Everything AI"
    lines = [header.rstrip("\n"), "", "# A tiny chat model for stories", "",
             f"A small language model that writes a short story for young children on request: pretrained, fine-tuned with SFT, "
             f"and given DPO adapters by {who}, with the code of its lessons.", "",
             "## Model", "",
             f"- **Class:** `ModernGPT` of the lesson llm-modern-gpt, {base['params']:,} parameters, with a vocabulary of {base['sizes'][0]:,} tokens: the 256 bytes, {len(base['merges']):,} merges of byte-level BPE, and the end of a story.",
             f"- **Adapters:** LoRA of rank {dpo['rank']} on the SFT model, {dpo['trainable']:,} numbers, trained with DPO.",
             "- **Prompt:** the instruction, then a line `Story:`; the model writes the story after it.", "",
             "## Intended use", "", uses.strip() or "Not written yet.", "",
             "## Not for", "", not_for.strip() or "Not written yet.", "",
             "## Training data", ""]
    for m, what in (("base", "Pretraining"), ("sft", "SFT"), ("dpo", "DPO")):
        lines.append(f"- **{what}:** " + _DATA[m][source].format(**info[m]) + ".")
    lines += ["", "## Training", ""]
    for m, what in (("base", "Pretraining"), ("sft", "SFT"), ("dpo", "DPO")):
        lines.append(f"- **{what}:** " + _TRAINING[m].format(**info[m]) + ".")
    lines += ["", "## Evaluation", ""]
    if scores is None:
        lines.append("Not measured yet.")
    else:
        c, s = scores["chat"], scores["safety"]
        have = [m for m in STAGES if m in c]
        heads = {"base": "before SFT", "sft": "after SFT", "dpo": "after SFT and DPO"}
        lines += ["The tests of the lessons llm-eval-chat (verifiable instructions) and llm-safety-eval (a red-team set), with the course's checks.", "",
                  "| Test | " + " | ".join(heads[m] for m in have) + " |", "|---" * (len(have) + 1) + "|"]
        for k in CHAT:
            lines.append(f"| {_NAMES[k]} | " + " | ".join(_share(c[m][k], c["total"][k]) for m in have) + " |")
        for k, what in (("harmful", "harmful requests refused"), ("jailbreak", "jailbreaks refused"), ("harmless", "harmless requests refused")):
            lines.append(f"| {what} | " + " | ".join(_share(s[m]["alone"][k], s["total"][k]) for m in have if m in s) + " |")
    lines += ["", "## Limits", ""]
    if scores is not None:
        c, s = scores["chat"], scores["safety"]
        for k in found:
            worse = "sft" in c and c["dpo"][k] < c["sft"][k]
            lines.append(f"- {_NAMES[k][0].upper() + _NAMES[k][1:]}: " + (f"worse after DPO, from {_share(c['sft'][k], c['total'][k])} to {_share(c['dpo'][k], c['total'][k])}"
                                                                          if worse else f"only {_share(c['dpo'][k], c['total'][k])}, below {round(100 * enough)}%") + ".")
        if s["dpo"]["alone"]["harmful"] == 0:
            lines.append(f"- It refuses no harmful request: 0 of {s['total']['harmful']}. It was never taught to refuse, so it is not safe to put in front of people as a chat model.")
    lines += ["- It writes only short English stories for young children, and was trained on nothing else.",
              "- Not tested: other kinds of instructions, other languages, and what it writes for a request that asks for no story, other than whether it refuses.", "",
              "## License", "",
              f"- **Model:** {license.strip() or 'not chosen yet'}.",
              "- **Data:** " + "; ".join(f"{name}, {LICENSES[key]}" for name, key in (("Cosmopedia", "cosmopedia"), ("TinyStories", "tinystories"), ("course-dpo-answers", "course-dpo-answers"))
                                         if key != "tinystories" or source == "yours") + "."]
    return "\n".join(lines) + "\n"


def yaml_block(text: str) -> str:
    """Text in YAML as the HTML of a highlighted block, as mo.md shows a fenced block, but with its lines as written:
    mo.md puts a blank line before every line that starts with "- ", also inside a fenced block, and a fence closed on
    the last line of the text (one without a final newline) is no fence at all."""
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import YamlLexer
    block = highlight(text.rstrip("\n") + "\n", YamlLexer(), HtmlFormatter(cssclass="language-yaml codehilite", wrapcode=True))
    return f"<span class='markdown prose dark:prose-invert contents'>{block}</span>"


def digit_reader_header() -> str:
    """The header of the model card of the digit reader of torch-pretrained, as it stands in data/course-qmnist-cnn.md."""
    text = (paths.ROOT / "data/course-qmnist-cnn.md").read_text(encoding="utf-8")
    return text[:text.index("\n---\n", 3) + 5]


def fields(source: str, license: str) -> tuple:
    """The fields and lists of the header of the card, as card_header takes them. The license as an id of the hub, such
    as cc-by-4.0: small letters, digits, dots and dashes; "other" when none is typed. A license typed with a colon or a
    # in it would make every header unreadable, also the course's."""
    lid = re.sub(r"[^a-z0-9.+-]+", "-", license.strip().lower()).strip("-")
    return ({"license": lid or "other", "pipeline_tag": "text-generation"},
            {"language": ["en"], "datasets": datasets(source), "tags": ["everything-ai", "tiny", "stories"]})


def save(text: str) -> str:
    """Save the card in the learner's project; gives back where."""
    folder = projects.folder() / "llm"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / CARD).write_text(text, encoding="utf-8")
    return f"work/projects/llm/{CARD}"


REQUIRED = ["card-header", "limits", "how-well"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL


# --- exercise 1: card_header -------------------------------------------------------------------------------------------

def _header(text):
    """The fields of a card's header as the hub reads them: the YAML between a first line --- and the next line ---,
    with nothing after it; None when there is no such header, or when it is no YAML of fields."""
    import yaml
    if not isinstance(text, str):
        return None
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return None
    ends = [i for i in range(1, len(lines)) if lines[i].strip() == "---"]
    if not ends or any(line.strip() for line in lines[ends[0] + 1:]):
        return None
    try:
        got = yaml.safe_load("\n".join(lines[1:ends[0]]))
    except yaml.YAMLError:
        return None
    return got if isinstance(got, dict) else None


def _lines(text: str) -> list:
    """The lines of a text without the spaces around them, and without blank lines at its end."""
    return [line.strip() for line in text.rstrip().split("\n")]


def _same_header(got, want) -> bool:
    mine, theirs = _header(got), _header(want)
    if mine is None and theirs is None:   # two headers the hub cannot read are the same only as the same lines: a known mistake
        return isinstance(got, str) and isinstance(want, str) and _lines(got) == _lines(want)
    return mine is not None and mine == theirs


_H_HEAD = "def card_header(fields, lists):\n"
_H = (_H_HEAD + "    lines = [\"---\"]\n    for key in fields:\n        lines.append(f\"{key}: {fields[key]}\")\n"
      "    for key in lists:\n        lines.append(f\"{key}:\")\n        for item in lists[key]:\n            lines.append(f\"- {item}\")\n"
      "    lines.append(\"---\")\n    return \"\\n\".join(lines) + \"\\n\"\n")
_H_START = (_H_HEAD + "    lines = [\"---\"]\n    for key in fields:\n        lines.append(f\"{key}: {fields[key]}\")\n"
            "    # each list: a line with its key, then a line for each item; and the closing line\n    ...\n"
            "    return \"\\n\".join(lines) + \"\\n\"\n")
_H_NO_END = _H.replace("    lines.append(\"---\")\n", "")
_H_JOINED = _H.replace("        lines.append(f\"{key}:\")\n        for item in lists[key]:\n            lines.append(f\"- {item}\")\n",
                       "        lines.append(f\"{key}: {', '.join(lists[key])}\")\n")
_H_NO_DASH = _H.replace("lines.append(f\"- {item}\")", "lines.append(item)")
_H_NO_SPACE = _H.replace("lines.append(f\"{key}: {fields[key]}\")", "lines.append(f\"{key}:{fields[key]}\")")

# --- exercise 2: limits ------------------------------------------------------------------------------------------------

_L_HEAD = "def limits(before, after, totals, enough):\n"
_L = (_L_HEAD + "    found = []\n    for kind in totals:\n        share = after[kind] / totals[kind]\n"
      "        if after[kind] < before[kind] or share < enough:\n            found.append(kind)\n    return found\n")
_L_START = (_L_HEAD + "    found = []\n    for kind in totals:\n        share = after[kind] / totals[kind]\n"
            "        # worse than before, or still too weak?\n        ...\n    return found\n")
_L_WORSE = _L.replace("after[kind] < before[kind] or share < enough", "after[kind] < before[kind]")
_L_WEAK = _L.replace("after[kind] < before[kind] or share < enough", "share < enough")
_L_AND = _L.replace("after[kind] < before[kind] or share < enough", "after[kind] < before[kind] and share < enough")
_L_BEFORE = _L.replace("share = after[kind] / totals[kind]", "share = before[kind] / totals[kind]")
_L_EQUAL = _L.replace("after[kind] < before[kind]", "after[kind] <= before[kind]")
_L_TWICE = (_L_HEAD + "    found = []\n    for kind in totals:\n        share = after[kind] / totals[kind]\n"
            "        if after[kind] < before[kind]:\n            found.append(kind)\n        if share < enough:\n            found.append(kind)\n    return found\n")
_L_EX = ({"rhyme": 10, "names": 35, "short": 6}, {"rhyme": 25, "names": 30, "short": 14}, {"rhyme": 50, "names": 40, "short": 20}, 0.6)


def _same_kinds(got, want) -> bool:
    """The same kinds, in any order, as a list, a tuple or a set."""
    if not isinstance(got, (list, tuple, set, frozenset)):
        return False
    try:
        return sorted(got) == sorted(want)   # a kind twice is not the same as once
    except TypeError:
        return False


EXERCISES = {
    "card-header": FunctionBox(
        name="card-header",
        function="card_header",
        start=_H_START,
        solution=_H,
        visible=[({"license": "mit"}, {"language": ["en", "nl"]})],
        hidden=[({"license": "cc-by-4.0", "pipeline_tag": "text-generation"}, {"datasets": ["roneneldan/TinyStories"], "tags": ["stories", "tiny"]}),
                ({"license": "apache-2.0"}, {}), ({}, {"language": ["en"]})],
        compare=_same_header,
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("no-end", wrong=_H_NO_END,
                    en=_CALL + ": the header has no closing line `---`, so the hub does not know where it ends, and reads no header at all. Add the line `---` after the lists.",
                    nl=_CALL_NL + ": de header heeft geen afsluitende regel `---`, dus weet de hub niet waar hij eindigt, en leest hij helemaal geen header. Zet de regel `---` na de lijsten."),
            Mistake("joined", wrong=_H_JOINED,
                    en=_CALL + ": `language: en, nl` is one field with the text *en, nl*, not a list of two. Write the key alone on its line, and each item on a line of its own that starts with `- `.",
                    nl=_CALL_NL + ": `language: en, nl` is één veld met de tekst *en, nl*, geen lijst van twee. Zet de sleutel alleen op zijn regel, en elk item op een eigen regel die begint met `- `."),
            Mistake("no-dash", wrong=_H_NO_DASH,
                    en=_CALL + ": an item of a list is a line that starts with `- `; without it, `en` under `language:` is no item, and the hub cannot read the header. Start each item's line with `- `.",
                    nl=_CALL_NL + ": een item van een lijst is een regel die begint met `- `; zonder dat is `en` onder `language:` geen item, en kan de hub de header niet lezen. Begin de regel van elk item met `- `."),
            Mistake("no-space", wrong=_H_NO_SPACE,
                    en=_CALL + ": `license:mit`, without a space after the colon, is one piece of text and no field. Write `key: value`, with a space.",
                    nl=_CALL_NL + ": `license:mit`, zonder spatie na de dubbele punt, is één stuk tekst en geen veld. Schrijf `key: value`, met een spatie."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: for each list a line `key:` and a line `- item` for each item, then the closing line `---`.",
                      "nl": "Vervang de `...`: voor elke lijst een regel `key:` en een regel `- item` voor elk item, en dan de afsluitende regel `---`."},
            "wrong": {"en": _CALL + ". A line `---`, a line `key: value` for each field, for each list `key:` and a line `- item` per item, and a closing `---`?",
                      "nl": _CALL_NL + ". Een regel `---`, een regel `key: value` voor elk veld, voor elke lijst `key:` en een regel `- item` per item, en een afsluitende `---`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Several fields, a list of one item, no lists, and no fields?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Meer velden, een lijst van één item, geen lijsten, en geen velden?"},
            "correct": {"en": "Your function writes a header the hub reads: read as YAML, it gives the same fields and lists back.",
                        "nl": "Je functie schrijft een header die de hub leest: gelezen als YAML geeft hij dezelfde velden en lijsten terug."},
        },
    ),
    "limits": FunctionBox(
        name="limits",
        function="limits",
        start=_L_START,
        solution=_L,
        visible=[_L_EX],
        hidden=[({"a": 5}, {"a": 5}, {"a": 10}, 0.4), ({"a": 3}, {"a": 6}, {"a": 10}, 0.6), ({}, {}, {}, 0.5),
                ({"a": 9, "b": 2}, {"a": 4, "b": 1}, {"a": 10, "b": 10}, 0.3)],
        compare=_same_kinds,
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("worse-only", wrong=_L_WORSE,
                    en=_CALL + ": you keep only the kinds that got worse. A kind that got better can still be too weak: *rhyme* went up to 25 of 50, below 0.6. Keep it when its share is below `enough` too.",
                    nl=_CALL_NL + ": je houdt alleen de soorten die slechter werden. Een soort die beter werd, kan nog te zwak zijn: *rhyme* ging naar 25 van de 50, onder 0,6. Houd hem ook als zijn aandeel onder `enough` ligt."),
            Mistake("weak-only", wrong=_L_WEAK,
                    en=_CALL + ": you keep only the kinds whose share is below `enough`. A kind that got worse is a limit too, even when it is still above: *names* went down from 35 to 30.",
                    nl=_CALL_NL + ": je houdt alleen de soorten waarvan het aandeel onder `enough` ligt. Een soort die slechter werd, is ook een beperking, ook als hij nog erboven zit: *names* ging van 35 naar 30."),
            Mistake("and", wrong=_L_AND,
                    en=_CALL + ": you keep a kind only when it got worse *and* is too weak. Either is enough to be a limit: `or`.",
                    nl=_CALL_NL + ": je houdt een soort alleen als hij slechter werd *en* te zwak is. Elk van de twee is genoeg voor een beperking: `or`."),
            Mistake("before-share", wrong=_L_BEFORE,
                    en=_CALL + ": you take the share before the last step. What the card says is how the model does now: the share of `after`.",
                    nl=_CALL_NL + ": je neemt het aandeel van vóór de laatste stap. Wat de card zegt, is hoe het model het nu doet: het aandeel van `after`."),
            Mistake("twice", wrong=_L_TWICE, state="almost",
                    en="Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ": a kind that got worse *and* is too weak is in your list twice. Add each kind once: one `if` with `or`.",
                    nl="Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ": een soort die slechter werd *en* te zwak is, staat twee keer in je lijst. Voeg elke soort één keer toe: één `if` met `or`."),
            Mistake("or-equal", wrong=_L_EQUAL,
                    en=_CALL + ": you count a kind that stayed the same as worse. Worse is fewer than before: `<`, not `<=`.",
                    nl=_CALL_NL + ": je telt een soort die gelijk bleef als slechter. Slechter is minder dan ervoor: `<`, niet `<=`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: add `kind` to `found` when it got worse than before, or when its share is below `enough`.",
                      "nl": "Vervang de `...`: voeg `kind` toe aan `found` als hij slechter werd dan ervoor, of als zijn aandeel onder `enough` ligt."},
            "wrong": {"en": _CALL + ". Every kind that is lower after than before, or whose share after is below `enough`?",
                      "nl": _CALL_NL + ". Elke soort die erna lager is dan ervoor, of waarvan het aandeel erna onder `enough` ligt?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". A kind that stayed the same, a share exactly at `enough`, no kinds, and kinds that got worse but are still above?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Een soort die gelijk bleef, een aandeel precies op `enough`, geen soorten, en soorten die slechter werden maar er nog boven zitten?"},
            "correct": {"en": "Your function finds the limits of a model in its scores: what the last training step made worse, and what is still too weak.",
                        "nl": "Je functie vindt de beperkingen van een model in zijn scores: wat de laatste trainingsstap slechter maakte, en wat nog te zwak is."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("card-header", _H_START, "empty", None),
    ("card-header", _H, "correct", None),
    ("card-header", _H.replace("lines.append(f\"{key}:\")\n        for item in lists[key]:\n            lines.append(f\"- {item}\")\n",
                               "lines.append(f\"{key}: {lists[key]}\")\n"), "correct", None),   # ['en', 'nl'] is a YAML list too
    ("card-header", _H.replace("    return \"\\n\".join(lines) + \"\\n\"\n", "    return \"\\n\".join(lines)\n"), "correct", None),
    ("card-header", _H.replace("lines.append(f\"- {item}\")", "lines.append(f\"  - {item}\")"), "correct", None),
    ("card-header", _H_NO_END, "wrong", "no-end"),
    ("card-header", _H_JOINED, "wrong", "joined"),
    ("card-header", _H_NO_DASH, "wrong", "no-dash"),
    ("card-header", _H_NO_SPACE, "wrong", "no-space"),
    ("card-header", _H.replace("    lines = [\"---\"]\n", "    lines = []\n"), "wrong", None),
    ("card-header", _H_NO_END.replace("    return \"\\n\".join(lines) + \"\\n\"\n", "    return \"\\n\".join(lines)\n"), "wrong", "no-end"),
    ("card-header", _H_NO_END.replace("lines.append(f\"- {item}\")", "lines.append(f\"  - {item}\")"), "wrong", "no-end"),
    ("limits", _L_START, "empty", None),
    ("limits", _L, "correct", None),
    ("limits", _L_HEAD + "    return [k for k in totals if after[k] < before[k] or after[k] / totals[k] < enough]\n", "correct", None),
    ("limits", _L_HEAD + "    return sorted(k for k in totals if after[k] < before[k] or after[k] < enough * totals[k])\n", "correct", None),
    ("limits", _L_WORSE, "wrong", "worse-only"),
    ("limits", _L_WEAK, "wrong", "weak-only"),
    ("limits", _L_AND, "wrong", "and"),
    ("limits", _L_BEFORE, "wrong", "before-share"),
    ("limits", _L_EQUAL, "wrong", "or-equal"),
    ("limits", _L_TWICE, "almost", "twice"),
    ("limits", _L_HEAD + "    return {k for k in totals if after[k] < before[k] or after[k] / totals[k] < enough}\n", "correct", None),
    ("limits", _L.replace("share < enough", "share <= enough"), "almost", None),
]
