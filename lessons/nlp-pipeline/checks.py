# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-pipeline: a chain of three steps that each take the output of the step before (split a
text into sentences, split the sentences into words, find the names), the outputs of every step, and the first step
whose output differs from what a person gives. The steps, their simple rules and the sentences are made up by the
course."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the course's three steps ----------------------------------------------------------------------------------------


def split_sentences(text: str) -> list:
    """Step 1, the course's simple rule: a new sentence starts after every '. ', '? ' or '! '."""
    sentences, start = [], 0
    for i in range(len(text) - 1):
        if text[i] in ".?!" and text[i + 1] == " ":
            sentences.append(text[start:i + 1])
            start = i + 2
    sentences.append(text[start:])
    return [s for s in sentences if s]


def split_words(sentences: list) -> list:
    """Step 2, the course's simple rule: words at the spaces; a comma at the end of a word, and the '.', '?' or '!'
    at the end of the sentence, become words of their own."""
    out = []
    for sentence in sentences:
        parts = [p for p in sentence.split(" ") if p]
        words = []
        for k, part in enumerate(parts):
            last = k == len(parts) - 1
            if len(part) > 1 and (part[-1] == "," or (last and part[-1] in ".?!")):
                words.extend([part[:-1], part[-1]])
            else:
                words.append(part)
        out.append(words)
    return out


def find_names(sentences: list) -> list:
    """Step 3, the course's simple rule: a word that starts with a capital letter is (part of) a name, except the
    first word of a sentence, since every sentence starts with a capital; capital words next to each other make one
    name."""
    names = []
    for words in sentences:
        current = []
        for i, word in enumerate(words):
            if i > 0 and word[:1].isupper():
                current.append(word)
            elif current:
                names.append(" ".join(current))
                current = []
        if current:
            names.append(" ".join(current))
    return names


STEPS = [split_sentences, split_words, find_names]
STEP_NAMES = {"en": ["split sentences", "split words", "find names"],
              "nl": ["zinnen splitsen", "woorden splitsen", "namen vinden"]}
OUTPUT_NAMES = {"en": ["the text", "sentences", "words", "names"], "nl": ["de tekst", "zinnen", "woorden", "namen"]}


def trace(text, steps: list) -> list:
    """The course's own Exercise 1: give the text to the first step and each output to the next step, and keep the
    output of every step."""
    outputs = []
    result = text
    for step in steps:
        result = step(result)
        outputs.append(result)
    return outputs


def first_wrong(outputs: list, expected: list):
    """The course's own Exercise 2: the number of the first step whose output differs from the expected output,
    counting from 0, and None when every step is right."""
    for i in range(len(outputs)):
        if outputs[i] != expected[i]:
            return i
    return None


# --- the course's sentences ------------------------------------------------------------------------------------------

# Four made-up sentences, and for each what a person gives at each step. The Dutch sentences go wrong at the same
# steps as the English ones; the code and the checks use the English ones.
SENTENCES = [
    "Dr. Smith lives in New York.",
    'I heard Anna say "Paris" twice.',
    "Anna lives in Paris.",
    "I met Anna in Paris. We had tea.",
]
SENTENCES_NL = [
    "Dr. Jansen woont in Den Haag.",
    'Ik hoorde Anna twee keer "Parijs" zeggen.',
    "Anna woont in Parijs.",
    "Ik zag Anna in Parijs. We dronken thee.",
]
EXPECTED = [
    [["Dr. Smith lives in New York."],
     [["Dr.", "Smith", "lives", "in", "New", "York", "."]],
     ["Smith", "New York"]],
    [['I heard Anna say "Paris" twice.'],
     [["I", "heard", "Anna", "say", '"', "Paris", '"', "twice", "."]],
     ["Anna", "Paris"]],
    [["Anna lives in Paris."],
     [["Anna", "lives", "in", "Paris", "."]],
     ["Anna", "Paris"]],
    [["I met Anna in Paris.", "We had tea."],
     [["I", "met", "Anna", "in", "Paris", "."], ["We", "had", "tea", "."]],
     ["Anna", "Paris"]],
]
EXPECTED_NL = [
    [["Dr. Jansen woont in Den Haag."],
     [["Dr.", "Jansen", "woont", "in", "Den", "Haag", "."]],
     ["Jansen", "Den Haag"]],
    [['Ik hoorde Anna twee keer "Parijs" zeggen.'],
     [["Ik", "hoorde", "Anna", "twee", "keer", '"', "Parijs", '"', "zeggen", "."]],
     ["Anna", "Parijs"]],
    [["Anna woont in Parijs."],
     [["Anna", "woont", "in", "Parijs", "."]],
     ["Anna", "Parijs"]],
    [["Ik zag Anna in Parijs.", "We dronken thee."],
     [["Ik", "zag", "Anna", "in", "Parijs", "."], ["We", "dronken", "thee", "."]],
     ["Anna", "Parijs"]],
]
OUTPUTS = [trace(s, STEPS) for s in SENTENCES]

# where each sentence goes wrong, and why, in the lesson's language
WHERE = {
    "en": ["Step 1 starts a new sentence after Dr., so Smith becomes the first word of a sentence, and step 3 skips "
           "first words. Steps 2 and 3 follow their own rules; they only got wrong input.",
           "Step 2 keeps the quotation marks on \"Paris\", so that word starts with a quotation mark, not with a "
           "capital letter, and step 3 misses it.",
           "Steps 1 and 2 are right. Step 3's own rule skips Anna, because it is the first word of the sentence.",
           "Every step gives what a person gives."],
    "nl": ["Stap 1 begint een nieuwe zin na Dr., dus Jansen wordt het eerste woord van een zin, en stap 3 slaat "
           "eerste woorden over. Stap 2 en 3 volgen hun eigen regels; ze kregen alleen verkeerde invoer.",
           "Stap 2 laat de aanhalingstekens aan het woord Parijs vastzitten, dus dat woord begint met een "
           "aanhalingsteken en niet met een hoofdletter, en stap 3 mist het.",
           "Stap 1 en 2 zijn goed. De eigen regel van stap 3 slaat Anna over, omdat Anna het eerste woord van de zin is.",
           "Elke stap geeft wat een mens geeft."],
}


def sentences(lang: str) -> list:
    return SENTENCES_NL if lang == "nl" else SENTENCES


def expected(lang: str) -> list:
    return EXPECTED_NL if lang == "nl" else EXPECTED


def sentence_options(lang: str) -> dict:
    """The sentences for the demo's dropdown, in the lesson's language."""
    return {f"{i + 1}. {s}": i for i, s in enumerate(sentences(lang))}


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


SENTENCES_N = [Named(f"SENTENCES[{i}]", s) for i, s in enumerate(SENTENCES)]
STEPS_N = Named("STEPS", STEPS)
OUTPUTS_N = [Named(f"OUTPUTS[{i}]", o) for i, o in enumerate(OUTPUTS)]
EXPECTED_N = [Named(f"EXPECTED[{i}]", e) for i, e in enumerate(EXPECTED)]

# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_BOX = ("min-width:0;border:1px solid var(--ea-line);border-radius:6px;padding:0.4rem 0.5rem;"
        "font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink);overflow-wrap:anywhere")
_WRONG = ";border:1.5px solid var(--ea-loss)"
_STEP = ("padding:0.3rem 0.2rem;font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);"
         "text-align:right;align-self:center")
_CODE = "font-family:var(--ea-font-code);font-size:12px"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _lines(value) -> list:
    """A value as Python writes it, one line per sentence for a list of sentences or of lists of words."""
    if isinstance(value, list) and value and all(isinstance(x, (str, list)) for x in value) and len(repr(value)) > 34:
        return [repr(x) for x in value]
    return [repr(value)]


def _shown(value) -> str:
    """A value in a box or a cell, as Python writes it, escaped, with long lines cut."""
    out = []
    for line in _lines(value):
        line = line if len(line) <= 70 else line[:67] + "..."
        out.append(f'<div style="{_CODE}">{html.escape(line)}</div>')
    return "".join(out)


def _said(value) -> str:
    text = repr(value)
    return text if len(text) <= 90 else text[:87] + "..."


def _chain(lang: str, text: str, outputs: list, want=None) -> str:
    """The text and the output of each step from top to bottom, with the step beside each output. A box whose output
    is not what a person gives (want) has a purple frame, and shows that under it."""
    names, steps = OUTPUT_NAMES[lang], STEP_NAMES[lang]
    person = {"en": "a person gives:", "nl": "een mens geeft:"}[lang]
    rows = [f'<div style="{_STEP}"></div><div style="{_BOX}"><div style="font-weight:600">{html.escape(names[0])}</div>'
            f'<div style="color:var(--ea-ink-2);margin-top:0.2rem">{html.escape(text)}</div></div>']
    for k, value in enumerate(outputs):
        bad = want is not None and value != want[k]
        extra = (f'<div style="{_UI}">{html.escape(person)}</div>{_shown(want[k])}' if bad else "")
        rows.append(f'<div style="{_STEP}">&darr; {k + 1}. {html.escape(steps[k])}</div>'
                    f'<div style="{_BOX}{_WRONG if bad else ""}"><div style="font-weight:600">{html.escape(names[k + 1])}'
                    f'</div><div style="margin-top:0.2rem">{_shown(value)}</div>{extra}</div>')
    return (f'<div style="display:grid;grid-template-columns:minmax(6rem,9rem) minmax(0,1fr);gap:0.3rem 0.5rem;'
            f'align-items:stretch">{"".join(rows)}</div>')


def _chain_said(lang: str, text: str, outputs: list) -> str:
    names, steps = OUTPUT_NAMES[lang], STEP_NAMES[lang]
    return f"{names[0]}: {text}; " + "; ".join(f"{k + 1}. {steps[k]}, {names[k + 1]}: {_said(v)}"
                                                for k, v in enumerate(outputs))


_PIPE_WORDS = {
    "en": {"title": "The course's pipeline of three steps",
           "note": "Each step gets what the step before gave back: a string, then a list of sentences, then a list of "
                   "lists of words."},
    "nl": {"title": "De pipeline van de course, in drie stappen",
           "note": "Elke stap krijgt wat de stap ervoor teruggaf: een string, dan een lijst zinnen, dan een lijst met "
                   "lijsten woorden."},
}


def pipeline_picture(lang: str) -> str:
    """The fourth sentence through the course's three steps, every step right."""
    w = _PIPE_WORDS[lang]
    text = sentences(lang)[3]
    outputs = trace(text, STEPS)
    return yours.figure(w["title"], _chain(lang, text, outputs) + _note(w["note"]),
                        alt=f'{w["title"]}. {_chain_said(lang, text, outputs)}. {w["note"]}')


_DEMO_WORDS = {
    "en": {"title": "Sentence {i} through the three steps", "start": "The mistake starts at step {k}.",
           "none": "No step makes a mistake."},
    "nl": {"title": "Zin {i} door de drie stappen", "start": "De fout begint bij stap {k}.",
           "none": "Geen enkele stap maakt een fout."},
}


def demo_picture(i: int, lang: str) -> str:
    """One sentence through the course's steps: the output of each step, the first step that differs from what a
    person gives, and why."""
    i = max(0, min(int(i), len(SENTENCES) - 1))
    w = _DEMO_WORDS[lang]
    text, want = sentences(lang)[i], expected(lang)[i]
    outputs = trace(text, STEPS)
    k = first_wrong(outputs, want)
    lines = [w["none"] if k is None else w["start"].format(k=k + 1), WHERE[lang][i]]
    title = w["title"].format(i=i + 1)
    return yours.figure(title, _chain(lang, text, outputs, want) + "".join(_note(x) for x in lines),
                        alt=f'{title}. {_chain_said(lang, text, outputs)}. ' + " ".join(lines))


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_CHECK_WORDS = {
    "en": {"error": "error",
           "trace": {"title": "Your trace(sentence, STEPS) on the four sentences",
                     "head": ["sentence", "your outputs", "the course's outputs"],
                     "fine": "The same as the course's for all four sentences. Only for the last sentence is the "
                             "output what a person gives; Exercise 2 finds where the others go wrong.",
                     "differs": "It differs from the course's for {k} of the four sentences: they have a dashed frame."},
           "first": {"title": "Your first_wrong on the four sentences",
                     "head": ["sentence", "yours", "the course's", "where it goes wrong"],
                     "fine": "The same as the course's for all four sentences: the mistakes start at the first step, "
                             "the second, the third, and not at all.",
                     "differs": "It differs from the course's for {k} of the four sentences: they have a dashed frame."},
           "step": "step {k}", "nostep": "no step",
           "alt": "{title}: {rows}. {after}"},
    "nl": {"error": "fout",
           "trace": {"title": "Jouw trace(zin, STEPS) op de vier zinnen",
                     "head": ["zin", "jouw uitvoer", "die van de course"],
                     "fine": "Bij alle vier de zinnen hetzelfde als die van de course. Alleen bij de laatste zin is de "
                             "uitvoer wat een mens geeft; oefening 2 vindt waar de andere misgaan.",
                     "differs": "Bij {k} van de vier zinnen anders dan die van de course: die hebben een gestreept kader."},
           "first": {"title": "Jouw first_wrong op de vier zinnen",
                     "head": ["zin", "de jouwe", "die van de course", "waar het misgaat"],
                     "fine": "Bij alle vier de zinnen hetzelfde als die van de course: de fouten beginnen bij de eerste "
                             "stap, de tweede, de derde, en helemaal niet.",
                     "differs": "Bij {k} van de vier zinnen anders dan die van de course: die hebben een gestreept kader."},
           "step": "stap {k}", "nostep": "geen stap",
           "alt": "{title}: {rows}. {after}"},
}


def _value_cell(value, lang: str) -> str:
    if value is _STOPPED:
        return f'<span style="{_CODE}">{html.escape(_CHECK_WORDS[lang]["error"])}</span>'
    if isinstance(value, list) and len(value) <= 6:
        return "".join(f'<div style="margin-bottom:0.15rem">{_shown(v)}</div>' for v in value)
    return _shown(value)


def trace_picture(seen) -> str:
    """The four sentences of the lesson's language, with the learner's outputs beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["trace"]
    rows, said, k = [], [], 0
    for text in sentences(seen.lang):
        got, want = _call(fn, text, STEPS), trace(text, STEPS)
        differs = got is _STOPPED or got != want
        k += differs
        rows.append([(html.escape(text), ""), (_value_cell(got, seen.lang), _OFF if differs else ""),
                     (_value_cell(want, seen.lang), "")])
        said.append(f"{text} {w['error'] if got is _STOPPED else _said(got)}, {_said(want)}")
    after = words["differs"].format(k=k) if k else words["fine"]
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def first_wrong_picture(seen) -> str:
    """The four sentences of the lesson's language: the learner's first wrong step beside the course's, and why."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["first"]
    rows, said, k = [], [], 0
    for i, text in enumerate(sentences(seen.lang)):
        outputs = trace(text, STEPS)
        got, want = _call(fn, outputs, expected(seen.lang)[i]), first_wrong(outputs, expected(seen.lang)[i])
        differs = got is _STOPPED or got != want or type(got) is not type(want)
        k += differs
        shown = w["error"] if got is _STOPPED else _said(got)
        where = w["nostep"] if want is None else w["step"].format(k=want + 1)
        rows.append([(html.escape(text), ""), (f'<span style="{_CODE}">{html.escape(shown)}</span>', _OFF if differs else ""),
                     (f'<span style="{_CODE}">{html.escape(repr(want))}</span>', ""),
                     (f'{html.escape(where)}<div style="color:var(--ea-ink-3)">{html.escape(WHERE[seen.lang][i])}</div>', "")])
        said.append(f"{text} {shown}, {want!r}, {where}")
    after = words["differs"].format(k=k) if k else words["fine"]
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["trace", "first_wrong", "fix", "measure"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(SENTENCES|STEPS|OUTPUTS|EXPECTED|split_sentences|split_words|find_names|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `STEPS`, `EXPECTED` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `STEPS`, `EXPECTED` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

# exercise 1: trace; the last two cases use steps of the check's own
TRACE_CASES = [(s, STEPS_N) for s in SENTENCES_N] + [
    ("  Hi  ", Named("[str.strip, str.lower]", [str.strip, str.lower])),
    ("abc", Named("[]", [])),
]
_T_START = "def trace(text, steps):\n    ...\n"
_T = ("def trace(text, steps):\n"
      "    outputs = []\n"
      "    result = text\n"
      "    for step in steps:\n"
      "        result = step(result)\n"
      "        outputs.append(result)\n"
      "    return outputs\n")
_T_LAST = _T.replace("    return outputs\n", "    return result\n")
_T_TEXT = _T.replace("        result = step(result)\n", "        result = step(text)\n")
_T_INPUT = _T.replace("    outputs = []\n", "    outputs = [text]\n")

# exercise 2: first_wrong; the last three cases are lists of the check's own
FIRST_CASES = [(o, e) for o, e in zip(OUTPUTS_N, EXPECTED_N)] + [
    ([1, 5, 3], [1, 2, 4]),
    ([4, 4], [4, 4]),
    ([], []),
]
_F_START = "def first_wrong(outputs, expected):\n    ...\n"
_F = ("def first_wrong(outputs, expected):\n"
      "    for i in range(len(outputs)):\n"
      "        if outputs[i] != expected[i]:\n"
      "            return i\n"
      "    return None\n")
_F_LATE = _F.replace("            return i\n", "            return i + 1\n")
_F_EARLY = _F.replace("            return i\n    return None\n", "            return i\n        return None\n")
_F_ZERO = _F.replace("    return None\n", "    return 0\n")
_F_LAST = ("def first_wrong(outputs, expected):\n"
           "    found = None\n"
           "    for i in range(len(outputs)):\n"
           "        if outputs[i] != expected[i]:\n"
           "            found = i\n"
           "    return found\n")


EXERCISES = {
    "trace": FunctionBox(
        name="trace",
        picture=trace_picture,
        function="trace",
        start=_T_START,
        solution=_T,
        visible=[TRACE_CASES[0]],
        hidden=TRACE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("last", wrong=_T_LAST,
                    en="`{call}` gives {got}, and it should give {want}: that is only what the last step gave back. Keep the output of every step in a list, and give back that list.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is alleen wat de laatste stap teruggaf. Bewaar de uitvoer van elke stap in een lijst, en geef die lijst terug."),
            Mistake("text", wrong=_T_TEXT,
                    en="`{call}` gives {got}, and it should give {want}: every step in your function gets the text you started with. In a pipeline, each step gets what the step before it gave back.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke stap krijgt in jouw functie de tekst waarmee je begon. In een pipeline krijgt elke stap wat de stap ervoor teruggaf."),
            Mistake("input", wrong=_T_INPUT,
                    en="`{call}` gives {got}, and it should give {want}: your list starts with the text itself. Give back only what the steps give back, one output for each step.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw lijst begint met de tekst zelf. Geef alleen terug wat de stappen teruggeven, één uitvoer voor elke stap."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give the text to the first step, each output to the next step, and give back a list of all the outputs.",
                      "nl": "Vervang de `...`: geef de tekst aan de eerste stap, elke uitvoer aan de volgende stap, en geef een lijst met alle uitvoer terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a list with the output of every step of `steps`, in order.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een lijst terug met de uitvoer van elke stap van `steps`, in volgorde."},
            "almost": {"en": "Right for the first sentence. But `{call}` gives {got}, and it should give {want}. Does your function use the `text` and `steps` it gets?",
                       "nl": "Goed voor de eerste zin. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `text` en `steps` die ze krijgt?"},
            "correct": {"en": "Your `trace` is right in all {n} cases. Look at the picture: the first sentence is already two sentences after step 1, and its last output misses Smith.",
                        "nl": "Je `trace` klopt in alle {n} gevallen. Kijk naar de afbeelding: de eerste zin is na stap 1 al twee zinnen, en de laatste uitvoer mist Jansen."},
        },
    ),
    "first_wrong": FunctionBox(
        name="first_wrong",
        picture=first_wrong_picture,
        function="first_wrong",
        start=_F_START,
        solution=_F,
        visible=[FIRST_CASES[0]],
        hidden=FIRST_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("late", wrong=_F_LATE,
                    en="`{call}` gives {got}, and it should give {want}: the function counts from 0, as Python counts the places in a list: step 1, the first step, has number 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de functie telt vanaf 0, zoals Python de plekken in een lijst telt: stap 1, de eerste stap, heeft nummer 0."),
            Mistake("zero", wrong=_F_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: `0` is the number of step 1, so a chain without mistakes would look like one that goes wrong at the start. Give `None` when every step is right.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `0` is het nummer van stap 1, dus een keten zonder fouten zou eruitzien als een keten die bij het begin misgaat. Geef `None` als elke stap goed is."),
            Mistake("last", wrong=_F_LAST,
                    en="`{call}` gives {got}, and it should give {want}: that is the last step that is wrong. The function gives the first one, where the mistake starts; stop as soon as you find it.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de laatste stap die fout is. De functie geeft de eerste, waar de fout begint; stop zodra je die vindt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the steps in order, and give back the number of the first one whose output is not the expected output, or `None`.",
                      "nl": "Vervang de `...`: ga de stappen in volgorde langs, en geef het nummer terug van de eerste waarvan de uitvoer niet de verwachte uitvoer is, of `None`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives the number of the first step whose output is not the expected output, counting from 0, and `None` when every step is right.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft het nummer van de eerste stap waarvan de uitvoer niet de verwachte uitvoer is, geteld vanaf 0, en `None` als elke stap goed is."},
            "almost": {"en": "Right for the first sentence. But `{call}` gives {got}, and it should give {want}. Does your function use the `outputs` and `expected` it gets?",
                       "nl": "Goed voor de eerste zin. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `outputs` en `expected` die ze krijgt?"},
            "correct": {"en": "Your `first_wrong` is right in all {n} cases. Of the four sentences, one goes wrong at each step, and one not at all: the picture says why.",
                        "nl": "Je `first_wrong` klopt in alle {n} gevallen. Van de vier zinnen gaat er bij elke stap één mis, en één helemaal niet: de afbeelding zegt waarom."},
        },
    ),
}

CASES = [
    ("trace", _T_START, "empty", None),
    ("trace", _T, "correct", None),
    ("trace", "def trace(text, steps):\n    outputs = []\n    for step in steps:\n        text = step(text)\n        outputs.append(text)\n    return outputs\n", "correct", None),
    ("trace", _T_LAST, "wrong", "last"),
    ("trace", _T_TEXT, "wrong", "text"),
    ("trace", _T_INPUT, "wrong", "input"),
    ("trace", "def trace(text, steps):\n    return [split_sentences(text)]\n", "error", "lesson-names"),
    ("trace", "def trace(text, steps):\n    return STEPS\n", "error", "lesson-names"),
    ("first_wrong", _F_START, "empty", None),
    ("first_wrong", _F, "correct", None),
    ("first_wrong", "def first_wrong(outputs, expected):\n    i = 0\n    while i < len(outputs):\n        if outputs[i] != expected[i]:\n            return i\n        i = i + 1\n    return None\n", "correct", None),
    ("first_wrong", _F_LATE, "wrong", "late"),
    ("first_wrong", _F_EARLY, "wrong", "unreached"),
    ("first_wrong", _F_ZERO, "wrong", "zero"),
    ("first_wrong", _F_LAST, "wrong", "last"),
    ("first_wrong", "def first_wrong(outputs, expected):\n    return 0\n", "almost", None),
    ("first_wrong", "def first_wrong(outputs, expected):\n    return EXPECTED\n", "error", "lesson-names"),
]
