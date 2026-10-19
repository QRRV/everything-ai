# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-chat-template: gemma_text, a conversation written out in the chat template of Gemma 3."""
import html
import re

from everything_ai import gpt2_tokenizer
from everything_ai.check import FunctionBox, Mistake

gpt2_tokenizer.offline()   # before any tiktoken.get_encoding("gpt2")


def pieces(text: str) -> list[str]:
    """The tokens GPT-2's tokenizer makes of `text`, as text: it knows no chat tokens, so it splits them."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return [enc.decode([i]) for i in enc.encode(text, disallowed_special=())]


def tokens_picture(labels: dict) -> str:
    """The two chat tokens of ChatML, each once as the one special token a chat model's tokenizer makes of it, and once
    as the plain-text tokens of GPT-2's tokenizer, in the code font. Nothing here is learned or a loss: ink and grey; a
    special token has a strong outline, a plain-text token a faint one."""
    W, H_BOX, CH, PAD, GAP = 330, 26, 8.4, 9, 5
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = 34
    for token in ("<|im_start|>", "<|im_end|>"):
        for label, parts, strong in ((labels["special"], [token], True), (labels["plain"].format(n=len(pieces(token))), pieces(token), False)):
            p.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')
            y += 18
            x = 0.0
            for part in parts:
                w = PAD * 2 + CH * len(part)
                stroke = "var(--ea-ink);stroke-width:1.6" if strong else "var(--ea-ink-3);stroke-width:1"
                p.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{H_BOX}" rx="4" style="fill:var(--ea-inset);stroke:{stroke}"/>'
                         f'<text x="{x + w / 2:.1f}" y="{y + 17.5}" style="font-size:13.5px;fill:var(--ea-ink);text-anchor:middle;'
                         f'font-family:var(--ea-font-code);font-variant-ligatures:none;font-feature-settings:\'calt\' 0,\'liga\' 0;white-space:pre">{html.escape(part)}</text>')
                x += w + GAP
            y += H_BOX + 10
        y += 8
    H = y
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# --- exercise 1: gemma_text ---------------------------------------------------------------------------------------------------

_HEAD = "def gemma_text(messages, add_generation_prompt):\n"
_START = (_HEAD + "    # messages: a list of dictionaries with the keys \"role\" and \"content\"\n    text = \"<bos>\"\n"
          "    # a `system` message, every turn, and the generation prompt\n    ...\n    return text\n")
_SYSTEM = ("    prefix = \"\"\n    if len(messages) > 0 and messages[0][\"role\"] == \"system\":\n"
           "        prefix = messages[0][\"content\"] + \"\\n\\n\"\n        messages = messages[1:]\n")
_ROLE = "        role = m[\"role\"]\n        if role == \"assistant\":\n            role = \"model\"\n"
_TURN = "        text = text + \"<start_of_turn>\" + role + \"\\n\" + prefix + m[\"content\"] + \"<end_of_turn>\\n\"\n        prefix = \"\"\n"
_PROMPT = "    if add_generation_prompt:\n        text = text + \"<start_of_turn>model\\n\"\n"


def _g(system: str = _SYSTEM, role: str = _ROLE, turn: str = _TURN, prompt: str = _PROMPT, bos: str = "<bos>") -> str:
    return (_HEAD + f"    text = \"{bos}\"\n" + system + "    for m in messages:\n" + role + turn + prompt + "    return text\n")


def _m(role: str, content: str) -> dict:
    return {"role": role, "content": content}


_VISIBLE = ([_m("user", "Hi")], True)
_HIDDEN = [
    ([_m("system", "Answer in one line."), _m("user", "Who is Romeo?")], True),
    ([_m("user", "Hi"), _m("assistant", "Good day."), _m("user", "Who are you?")], False),
    ([_m("system", "Be kind."), _m("user", "Hello"), _m("assistant", "Hello!"), _m("user", "Bye")], True),
]


def _show(v) -> str:
    return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^\s*\.\.\.\s*(#.*)?$", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_WRONG = {
    "assistant-role": _g(role="        role = m[\"role\"]\n"),
    "system-turn": _g(system="    prefix = \"\"\n"),
    "one-newline": _g(system=_SYSTEM.replace("+ \"\\n\\n\"", "+ \"\\n\"")),
    "no-newline": _g(turn=_TURN.replace("\"<end_of_turn>\\n\"", "\"<end_of_turn>\"")),
    "no-prompt": _g(prompt=""),
    "always-prompt": _g(prompt="    text = text + \"<start_of_turn>model\\n\"\n"),
    "prompt-end": _g(prompt=_PROMPT.replace("<start_of_turn>model\\n", "<start_of_turn>model\\n<end_of_turn>\\n")),
    "no-bos": _g(bos=""),
    "prefix-every": _g(turn=_TURN.replace("        prefix = \"\"\n", "")),
}

EXERCISES = {
    "gemma-text": FunctionBox(
        name="gemma-text",
        function="gemma_text",
        start=_START,
        solution=_g(),
        visible=[_VISIBLE],
        hidden=_HIDDEN,
        show=_show,
        compare=lambda got, want: isinstance(got, str) and got == want,
        first=[_DOTS],
        mistakes=[
            Mistake("assistant-role", wrong=_WRONG["assistant-role"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the turn of an `assistant` message starts with `<start_of_turn>assistant`. Gemma calls that role `model`, as in its generation prompt.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de beurt van een `assistant`-bericht begint met `<start_of_turn>assistant`. Gemma noemt die rol `model`, net als in zijn generation prompt."),
            Mistake("system-turn", wrong=_WRONG["system-turn"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `system` message gets a turn of its own. Gemma has no `system` turn: its content goes at the start of the first `user` message, followed by an empty line.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `system`-bericht krijgt een eigen beurt. Gemma heeft geen `system`-beurt: de inhoud komt aan het begin van het eerste `user`-bericht, met een lege regel erna."),
            Mistake("one-newline", wrong=_WRONG["one-newline"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: after the `system` message comes one `\\n`. It needs an empty line, `\\n\\n`, before the user's own words.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: na het `system`-bericht komt één `\\n`. Er hoort een lege regel, `\\n\\n`, voor de eigen woorden van de gebruiker."),
            Mistake("prefix-every", wrong=_WRONG["prefix-every"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the `system` message comes back in every turn. It belongs at the start of the first `user` message only.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het `system`-bericht komt in elke beurt terug. Het hoort alleen aan het begin van het eerste `user`-bericht."),
            Mistake("no-newline", wrong=_WRONG["no-newline"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: after `<end_of_turn>` there is no new line. Every turn ends with `<end_of_turn>\\n`, so that the next `<start_of_turn>` starts a line.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: na `<end_of_turn>` staat geen nieuwe regel. Elke beurt eindigt met `<end_of_turn>\\n`, zodat de volgende `<start_of_turn>` aan het begin van een regel staat."),
            Mistake("no-prompt", wrong=_WRONG["no-prompt"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the generation prompt is missing. With `add_generation_prompt` true, end with `<start_of_turn>model\\n`, so that what comes next is the model's answer.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de generation prompt ontbreekt. Als `add_generation_prompt` waar is, eindig je met `<start_of_turn>model\\n`, zodat wat volgt het antwoord van het model is."),
            Mistake("always-prompt", wrong=_WRONG["always-prompt"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add the generation prompt also when `add_generation_prompt` is false. Add it only when it is true: a conversation for training ends with the last turn.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je zet de generation prompt er ook bij als `add_generation_prompt` onwaar is. Zet hem er alleen bij als hij waar is: een gesprek om op te trainen eindigt met de laatste beurt."),
            Mistake("prompt-end", wrong=_WRONG["prompt-end"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your generation prompt ends the turn it opens. It is only the start of the model's turn, `<start_of_turn>model\\n`: the model writes the rest, and its own `<end_of_turn>`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je generation prompt sluit de beurt die hij opent. Hij is alleen het begin van de beurt van het model, `<start_of_turn>model\\n`: het model schrijft de rest, en zijn eigen `<end_of_turn>`."),
            Mistake("not-text", matches=lambda got, text: not isinstance(got, str),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function gives back something other than a text. Return the text itself.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie geeft iets anders terug dan een tekst. Geef de tekst zelf terug."),
            Mistake("no-bos", wrong=_WRONG["no-bos"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the text does not start with `<bos>`. Keep the first line, `text = \"<bos>\"`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de tekst begint niet met `<bos>`. Houd de eerste regel, `text = \"<bos>\"`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with your code: a `system` message, every turn, and the generation prompt.",
                      "nl": "Vervang de `...` door je code: een `system`-bericht, elke beurt, en de generation prompt."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with the messages and whether to add the generation prompt, and the `def` line of your function has another number of names between its parentheses. Keep `(messages, add_generation_prompt)`.",
                          "nl": "De check roept `{call}` aan, met de berichten en of de generation prompt erbij moet, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(messages, add_generation_prompt)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Look at the two texts character by character: where do they part?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk de twee teksten teken voor teken: waar gaan ze uiteen?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your template handle a `system` message, earlier answers of the model, and no generation prompt?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Kan je template een `system`-bericht aan, eerdere antwoorden van het model, en geen generation prompt?"},
            "correct": {"en": "Your template writes every conversation as Gemma 3 learned to read them: one exact format, down to the last new line.",
                        "nl": "Je template schrijft elk gesprek zoals Gemma 3 ze heeft leren lezen: één precies formaat, tot de laatste nieuwe regel."},
        },
    ),
}

# The lesson is done when these are correct. "fake-turn" is the choice question, "no-error" the open question.
REQUIRED = ["gemma-text", "fake-turn", "no-error"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gemma-text", _START, "empty", None),
    ("gemma-text", _g(), "correct", None),
    ("gemma-text", _HEAD + "    text = \"<bos>\"\n    first = \"\"\n    for i, m in enumerate(messages):\n"
     "        if m[\"role\"] == \"system\":\n            first = m[\"content\"] + \"\\n\\n\"\n            continue\n"
     "        role = {\"user\": \"user\", \"assistant\": \"model\"}[m[\"role\"]]\n"
     "        text += f\"<start_of_turn>{role}\\n{first}{m['content']}<end_of_turn>\\n\"\n        first = \"\"\n"
     "    if add_generation_prompt:\n        text += \"<start_of_turn>model\\n\"\n    return text\n", "correct", None),
    ("gemma-text", _g().replace("    return text\n", "    return [text]\n"), "wrong", "not-text"),
] + [("gemma-text", code, "wrong", key) for key, code in _WRONG.items()]
