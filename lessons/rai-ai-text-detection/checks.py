# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ai-text-detection: how surprised a model is by a text, and what a detector that looks
only at that number gets right and wrong. The course's run (data/course-detection-qwen.json, made by
data/tools/record_course_detection_qwen.py with score below) gave Qwen3-0.6B, the chat model, each question of
QUESTIONS, and kept its answer next to the answer a person wrote for the OpenAssistant project (data/oasst-short.json);
it then scored both answers, and the famous texts of FAMOUS, with the same model: the log probability of every token
given the tokens before it."""
import contextlib
import copy
import functools
import html
import io
import json
import re

from everything_ai import chat, paths, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

COURSE = "data/course-detection-qwen.json"
MAX_TOKENS = 256   # what Qwen3 may write per answer
LIVE_TOKENS = 512  # the demo scores at most this many tokens of the learner's text: the logits of more fill the memory
DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
THRESHOLD = 1.75   # where the demo starts, and the threshold of Exercise 2's first case

# Twenty questions people asked the OpenAssistant project, each with an answer a person wrote in prose, without a
# list, a link or code, and without the writer playing an AI; the questions as data/oasst-short.json has them.
QUESTIONS = [
    "Why is the sky blue?",
    "can stainless steel be magnetic in some instances?",
    "Who was Socrates and when did he die?",
    "How would you describe the fur of a swiss mountain dog?",
    "What is the relationship between tensorflow and keras?",
    "Which came first,  the chicken or the egg?",
    "Do cats bath",
    "Can you tell me how tall the burj khalifa is please?",
    "what is a monad in haskell?",
    "Which nation was the first to outlaw animal cruelty?",
    "Tell me about 15th century armor from Poland",
    "Can you play the game Empyrion in first person?",
    "What does manipulation mean?",
    "Please give me some ideas for a jazzy but sombre chord progression.",
    "If Mermaids were real would they lay eggs? Explain your reasoning.",
    "What's a dht node? I saw some people talking about that but I didn't really understood what they were meaning.",
    "What is the relationship between quantum mechanics and general relativity, and how have scientists attempted to reconcile the two theories?",
    "What happens if I hold a piece of paper horizontally with two hands and I let go of one hand?",
    "If we assume that Pluto is a planet, how many planets there are in the solar system?",
    "Can you summarise the bible in 100 words",
]
SKY, SOCRATES = 0, 2   # the questions the worked example and Exercise 1 look at

# Famous texts people wrote long ago, public domain, copied onto many web pages (the same texts as in rai-memorization)
FAMOUS = {
    "constitution": "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
    "declaration": "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
    "gettysburg": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war.",
    "dickens": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us,",
    "moby": "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
}
FAMOUS_NAMES = {
    "en": {"constitution": "the preamble of the US Constitution", "declaration": "the opening of the Declaration of Independence",
           "gettysburg": "the start of the Gettysburg Address", "dickens": "the opening of A Tale of Two Cities",
           "moby": "the opening of Moby-Dick"},
    "nl": {"constitution": "de preambule van de Amerikaanse grondwet", "declaration": "het begin van de Onafhankelijkheidsverklaring",
           "gettysburg": "het begin van de Gettysburg Address", "dickens": "het begin van A Tale of Two Cities",
           "moby": "het begin van Moby-Dick"},
}


def score(model, enc, text: str, context: str = "", limit: int | None = None) -> list:
    """Every token of `text` with its log probability under `model`, given `context` and the tokens of `text` before
    it, as [token, logprob] pairs, logprob the natural log rounded to 4 decimals. Without a context, the first token
    of the text is left out: nothing comes before it to predict it from. With `limit`, only the first `limit` tokens
    of context and text together count."""
    import torch
    from everything_ai import qwen3
    before = qwen3.encode(enc, context) if context else []
    ids = before + qwen3.encode(enc, text)
    if limit is not None:
        ids = ids[:limit]
    if len(ids) < 2:
        return []
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor(ids)).float(), dim=-1)
    return [[enc.decode([ids[i]]), round(float(logp[i - 1, ids[i]]), 4)] for i in range(max(len(before), 1), len(ids))]


# --- the course's run ------------------------------------------------------------------------------------------------

@functools.cache
def _recorded() -> dict:
    """The course's run, or an empty dict before the first run."""
    path = paths.ROOT / COURSE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def items() -> list:
    """The 20 questions of the run, each with both answers and their scores."""
    return _recorded().get("items", [])


def average_nll(pairs: list) -> float:
    """The course's own Exercise 1: the mean surprise, minus the log probability, over every pair."""
    total = 0
    for token, logprob in pairs:
        total = total - logprob
    return total / len(pairs)


def flagged_share(scores: list, threshold: float) -> float:
    """The course's own Exercise 2: the share of the scores below the threshold, the texts the detector calls AI."""
    flagged = 0
    for s in scores:
        if s < threshold:
            flagged = flagged + 1
    return flagged / len(scores)


def share_right(caught: float, false_positive: float, share_ai: float) -> float:
    """The course's own Exercise 3: of the texts the detector flags, the share that is really AI."""
    right = caught * share_ai
    wrong = false_positive * (1 - share_ai)
    return right / (right + wrong)


def averages(kind: str) -> list:
    """The average NLL of each of the 20 answers of one kind, rounded to 3 decimals: "human" or "ai" alone, or
    "human_after_question" or "ai_after_question"."""
    return [round(average_nll(x["scores"][kind]), 3) for x in items()]


def tokens(kind: str) -> list:
    """How many tokens each answer of a kind has: the scored pairs and the first token, which has no score."""
    return [len(x["scores"][kind]) + 1 for x in items()]


HUMAN = averages("human")                  # the 20 answers people wrote, scored alone
AI = averages("ai")                        # Qwen3's own 20 answers, scored alone
HUMAN_Q = averages("human_after_question")   # the same, after the question in Qwen3's chat template
AI_Q = averages("ai_after_question")
FAMOUS_NLL = {k: round(average_nll(v["score"]), 3) for k, v in _recorded().get("famous", {}).items()}
SKY_HUMAN = items()[SKY]["scores"]["human"] if items() else []
SKY_AI = items()[SKY]["scores"]["ai"] if items() else []


def errors(threshold: float, with_question: bool = False) -> dict:
    """What a detector with this threshold gets wrong on the run: answers by people it flags, answers of Qwen3 it lets
    through, and (alone only) the famous texts it flags, each as a list of indexes or keys."""
    human, ai = (HUMAN_Q, AI_Q) if with_question else (HUMAN, AI)
    return {"people": [i for i, v in enumerate(human) if v < threshold],
            "missed": [i for i, v in enumerate(ai) if v >= threshold],
            "famous": [] if with_question else [k for k, v in FAMOUS_NLL.items() if v < threshold]}


def fewest_mistakes() -> tuple:
    """The fewest mistakes any threshold makes on the 40 answers alone, and the thresholds (in steps of 0.01) that
    make so few."""
    counts = {t / 100: len(errors(t / 100)["people"]) + len(errors(t / 100)["missed"]) for t in range(0, 501)}
    least = min(counts.values())
    return least, [t for t, c in counts.items() if c == least]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


HUMAN_N = Named("HUMAN", HUMAN)
AI_N = Named("AI", AI)
SKY_HUMAN_N = Named("SKY_HUMAN", SKY_HUMAN)
SKY_AI_N = Named("SKY_AI", SKY_AI)

# --- pictures --------------------------------------------------------------------------------------------------------
# Every text is shown through html.escape: the answers of Qwen3 and the learner's own text are never read as HTML.

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"


def number(x: float, lang: str, digits: int = 2) -> str:
    """A number for the text of a picture: rounded, with a decimal comma in Dutch and a real minus sign."""
    text = f"{round(x, digits) + 0.0:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _mono(text: str) -> str:
    return f'<span style="{_MONO}">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _token_text(token: str) -> str:
    """A token as a picture writes it: a space at its start as ␣, a new line as ↵."""
    return token.replace("\n", "↵").replace(" ", "␣", 1) if token.startswith(" ") else token.replace("\n", "↵")


def _bars(pairs: list, title: str, alt: str, top: float = 14.0) -> str:
    """A bar per token, as high as its surprise (minus its log probability), with the token under it. Surprise is
    what the loss adds up, so the bars have the loss colour."""
    n = len(pairs)
    W, LEFT, TOP, PH = 540, 28, 26, 110
    bw = (W - LEFT) / max(n, 1)

    def y(v):
        return TOP + PH - PH * min(v, top) / top
    p = [f'<text x="0" y="14" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>']
    for k in range(0, int(top) + 1, 5):
        p.append(f'<line x1="{LEFT}" x2="{W}" y1="{y(k):.1f}" y2="{y(k):.1f}" style="stroke:var(--ea-line);stroke-width:1"/>'
                 f'<text x="{LEFT - 5}" y="{y(k) + 4:.1f}" style="font-size:10px;fill:var(--ea-ink-2);text-anchor:end">{k}</text>')
    for i, (token, logprob) in enumerate(pairs):
        v = -logprob
        x = LEFT + i * bw + 2
        p.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw - 4:.1f}" height="{TOP + PH - y(v):.1f}" style="fill:var(--ea-loss)"/>')
        cx, ly = x + (bw - 4) / 2, TOP + PH + 8
        p.append(f'<text x="{cx:.1f}" y="{ly}" transform="rotate(-40 {cx:.1f} {ly})" style="font-size:10px;'
                 f'font-family:var(--ea-font-code);fill:var(--ea-ink-2);text-anchor:end">{html.escape(_token_text(token))}</text>')
    H = TOP + PH + 70
    return (f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;max-width:{W}px;height:auto;display:block;'
            f'font-family:var(--ea-font-ui)"><title>{html.escape(alt)}</title>{"".join(p)}</svg>')


_TOKENS_WORDS = {
    "en": {"title": "How surprised Qwen3 is by each of the first {n} tokens",
           "human": "The answer a person wrote", "ai": "The answer Qwen3 wrote",
           "note": "The first token, \"The\", has no bar: nothing comes before it. Higher is more surprised.",
           "alt": "{who}: {tokens}"},
    "nl": {"title": "Hoe verrast Qwen3 is door elk van de eerste {n} tokens",
           "human": "Het antwoord dat een mens schreef", "ai": "Het antwoord dat Qwen3 schreef",
           "note": "Het eerste token, \"The\", heeft geen staaf: er komt niets voor. Hoger is meer verrast.",
           "alt": "{who}: {tokens}"},
}


def tokens_picture(lang: str, n: int = 16) -> str:
    """The surprise of the first n scored tokens of both answers to "Why is the sky blue?"."""
    w = _TOKENS_WORDS[lang]
    parts = []
    for kind, pairs in (("human", SKY_HUMAN[:n]), ("ai", SKY_AI[:n])):
        said = ", ".join(f"{_token_text(t)} {number(-lp, lang)}" for t, lp in pairs)
        parts.append(_bars(pairs, w[kind], w["alt"].format(who=w[kind], tokens=said)))
    return yours.figure(w["title"].format(n=n), "".join(parts) + _note(w["note"]), alt=w["title"].format(n=n))


def _strip(rows: list, threshold: float | None, lang: str, high: float = 4.0) -> str:
    """Dots on a line from 0 to `high`, one row per kind of text: (label, values, colour, wrong side). A dot on the
    wrong side of the threshold has a dashed ring in the loss colour; left of the threshold is what the detector
    calls AI."""
    W, LEFT, RIGHT, ROW = 540, 150, 14, 30
    span = W - LEFT - RIGHT

    def x(v):
        return LEFT + span * max(0.0, min(v, high)) / high
    H = ROW * len(rows) + 34
    p = []
    if threshold is not None:
        p.append(f'<rect x="{LEFT}" y="4" width="{x(threshold) - LEFT:.1f}" height="{ROW * len(rows)}" '
                 f'style="fill:color-mix(in srgb, var(--ea-ink-3) 12%, transparent)"/>')
    for k, (name, values, colour, wrong) in enumerate(rows):
        cy = 4 + ROW * k + ROW / 2
        p.append(f'<text x="{LEFT - 8}" y="{cy + 4:.1f}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:end">'
                 f'{html.escape(name)}</text>'
                 f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{cy:.1f}" y2="{cy:.1f}" style="stroke:var(--ea-line);stroke-width:1"/>')
        for v in values:
            p.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="4.5" style="fill:{colour};fill-opacity:0.8"/>')
            if threshold is not None and wrong(v, threshold):
                p.append(f'<circle cx="{x(v):.1f}" cy="{cy:.1f}" r="8" style="fill:none;stroke:var(--ea-loss);'
                         f'stroke-width:1.5;stroke-dasharray:2.5 2"/>')
    base = 4 + ROW * len(rows)
    for t in range(0, int(high) + 1):
        p.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{base}" y2="{base + 4}" style="stroke:var(--ea-ink-3)"/>'
                 f'<text x="{x(t):.1f}" y="{base + 16}" style="font-size:10px;fill:var(--ea-ink-2);text-anchor:middle">{t}</text>')
    if threshold is not None:
        p.append(f'<line x1="{x(threshold):.1f}" x2="{x(threshold):.1f}" y1="2" y2="{base + 2}" '
                 f'style="stroke:var(--ea-ink);stroke-width:1.5;stroke-dasharray:4 3"/>')
    return (f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="width:100%;max-width:{W}px;height:auto;display:block;'
            f'font-family:var(--ea-font-ui)">{"".join(p)}</svg>')


def _person_wrong(v, t):
    return v < t


def _ai_wrong(v, t):
    return v >= t


_RUN_WORDS = {
    "en": {"title": "The average NLL of the 40 answers, scored by Qwen3",
           "alone": "Each answer alone", "question": "Each answer after its question",
           "human": "by people (20)", "ai": "by Qwen3 (20)",
           "note": "Each dot is one answer; further right is more surprising. With the question in front, every answer of Qwen3 is below every answer by people.",
           "alt": "{title}. Alone: by people {human}; by Qwen3 {ai}. After the question: by people {human_q}; by Qwen3 {ai_q}."},
    "nl": {"title": "De gemiddelde NLL van de 40 antwoorden, gescoord door Qwen3",
           "alone": "Elk antwoord los", "question": "Elk antwoord na zijn vraag",
           "human": "door mensen (20)", "ai": "door Qwen3 (20)",
           "note": "Elke stip is één antwoord; verder naar rechts is meer verrassend. Met de vraag ervoor staat elk antwoord van Qwen3 onder elk antwoord van een mens.",
           "alt": "{title}. Los: door mensen {human}; door Qwen3 {ai}. Na de vraag: door mensen {human_q}; door Qwen3 {ai_q}."},
}


def _values(values: list, lang: str) -> str:
    return ", ".join(number(v, lang) for v in sorted(values))


def run_picture(lang: str) -> str:
    """The 40 answers of the run as dots, alone and after their question."""
    w = _RUN_WORDS[lang]
    body = ""
    for head, human, ai in ((w["alone"], HUMAN, AI), (w["question"], HUMAN_Q, AI_Q)):
        body += (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;font-weight:600;color:var(--ea-ink);'
                 f'margin-top:0.4rem">{html.escape(head)}</div>'
                 + _strip([(w["human"], human, "var(--ea-ink)", _person_wrong), (w["ai"], ai, "var(--ea-pred)", _ai_wrong)],
                          None, lang))
    alt = w["alt"].format(title=w["title"], human=_values(HUMAN, lang), ai=_values(AI, lang),
                          human_q=_values(HUMAN_Q, lang), ai_q=_values(AI_Q, lang))
    return yours.figure(w["title"], body + _note(w["note"]), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "A detector with the threshold {t}",
           "human": "by people (20)", "ai": "by Qwen3 (20)", "famous": "famous texts (5)",
           "people": "{k} of the 20 answers by people are called AI.",
           "people_one": "1 of the 20 answers by people is called AI.",
           "missed": "{k} of the 20 answers of Qwen3 get through as human.",
           "missed_one": "1 of the 20 answers of Qwen3 gets through as human.",
           "famous_none": "None of the 5 famous texts is called AI.",
           "famous_some": "Called AI among the famous texts: {names}.",
           "no_famous": "The famous texts have no question, so they are not in this picture.",
           "key": "Left of the dashed line, the detector says AI. A dashed ring marks a mistake."},
    "nl": {"title": "Een detector met de drempel {t}",
           "human": "door mensen (20)", "ai": "door Qwen3 (20)", "famous": "bekende teksten (5)",
           "people": "{k} van de 20 antwoorden van mensen heten AI.",
           "people_one": "1 van de 20 antwoorden van mensen heet AI.",
           "missed": "{k} van de 20 antwoorden van Qwen3 komen erdoor als menselijk.",
           "missed_one": "1 van de 20 antwoorden van Qwen3 komt erdoor als menselijk.",
           "famous_none": "Geen van de 5 bekende teksten heet AI.",
           "famous_some": "Bekende teksten die AI heten: {names}.",
           "no_famous": "De bekende teksten hebben geen vraag, dus ze staan niet in dit plaatje.",
           "key": "Links van de gestreepte lijn zegt de detector AI. Een gestreepte ring markeert een fout."},
}


def demo_picture(threshold: float, with_question: bool, lang: str) -> str:
    """The run with a threshold: what the detector calls AI, and its mistakes on both sides."""
    w = _DEMO_WORDS[lang]
    human, ai = (HUMAN_Q, AI_Q) if with_question else (HUMAN, AI)
    rows = [(w["human"], human, "var(--ea-ink)", _person_wrong), (w["ai"], ai, "var(--ea-pred)", _ai_wrong)]
    if not with_question:
        rows.append((w["famous"], list(FAMOUS_NLL.values()), "var(--ea-ink)", _person_wrong))
    wrong = errors(threshold, with_question)
    lines = [w[key + "_one"] if len(wrong[key]) == 1 else w[key].format(k=len(wrong[key])) for key in ("people", "missed")]
    if with_question:
        lines.append(w["no_famous"])
    elif wrong["famous"]:
        lines.append(w["famous_some"].format(names="; ".join(FAMOUS_NAMES[lang][k] for k in wrong["famous"])))
    else:
        lines.append(w["famous_none"])
    title = w["title"].format(t=number(threshold, lang))
    body = _strip(rows, threshold, lang) + "".join(_note(x) for x in lines) + _note(w["key"])
    return yours.figure(title, body, alt=title + ". " + " ".join(lines))


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


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _close(a, b) -> bool:
    return _is_number(a) and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def _shown(v, lang: str) -> str:
    """What the learner's function gave, for a cell of a table: a number rounded, otherwise as Python writes it."""
    if v is _STOPPED:
        return {"en": "error", "nl": "fout"}[lang]
    if _is_number(v):
        return number(v, lang, 3)
    text = repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


_CHECK_WORDS = {
    "en": {"average": {"title": "Your average_nll on four answers of the run",
                       "head": ["answer", "tokens", "yours", "the course's"],
                       "rows": {"sky-human": "Why is the sky blue? (a person)", "sky-ai": "Why is the sky blue? (Qwen3)",
                                "socrates-human": "Who was Socrates? (a person)", "socrates-ai": "Who was Socrates? (Qwen3)"},
                       "fine": "The same as the course's on all four answers.",
                       "differs": "It differs from the course's on {k} of the four answers: they have a dashed frame."},
           "flagged": {"title": "Your flagged_share at the threshold 1.75",
                       "head": ["call", "yours", "the course's"],
                       "fine": "The same as the course's: 3 of the 20 answers by people flagged, and 15 of the 20 answers of Qwen3.",
                       "differs": "It differs from the course's on {k} of the two lists: they have a dashed frame."},
           "share": {"title": "Your share_right: 10,000 texts, 1 in 100 by AI",
                     "head": ["call", "yours", "the course's"],
                     "squares": "The 189 texts the detector flags: 90 by AI (filled), 99 by people (in the loss colour).",
                     "fine": "The same as the course's: 90 of the 189 accusations are right.",
                     "differs": "It differs from the course's, which gives 90 / 189: the row has a dashed frame."},
           "alt": "{title}: {rows}. {after}"},
    "nl": {"average": {"title": "Jouw average_nll op vier antwoorden van de run",
                       "head": ["antwoord", "tokens", "de jouwe", "die van de course"],
                       "rows": {"sky-human": "Why is the sky blue? (een mens)", "sky-ai": "Why is the sky blue? (Qwen3)",
                                "socrates-human": "Who was Socrates? (een mens)", "socrates-ai": "Who was Socrates? (Qwen3)"},
                       "fine": "Bij alle vier de antwoorden hetzelfde als die van de course.",
                       "differs": "Bij {k} van de vier antwoorden anders dan die van de course: die hebben een gestreept kader."},
           "flagged": {"title": "Jouw flagged_share bij de drempel 1,75",
                       "head": ["aanroep", "de jouwe", "die van de course"],
                       "fine": "Hetzelfde als die van de course: 3 van de 20 antwoorden van mensen aangewezen, en 15 van de 20 antwoorden van Qwen3.",
                       "differs": "Bij {k} van de twee lijsten anders dan die van de course: die hebben een gestreept kader."},
           "share": {"title": "Jouw share_right: 10.000 teksten, 1 op de 100 van AI",
                     "head": ["aanroep", "de jouwe", "die van de course"],
                     "squares": "De 189 teksten die de detector aanwijst: 90 van AI (gevuld), 99 van mensen (in de kleur van de loss).",
                     "fine": "Hetzelfde als die van de course: 90 van de 189 beschuldigingen kloppen.",
                     "differs": "Anders dan die van de course, die 90 / 189 geeft: de rij heeft een gestreept kader."},
           "alt": "{title}: {rows}. {after}"},
}


def average_picture(seen) -> str:
    """Four answers of the run, with the learner's average NLL beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["average"]
    rows, said, k = [], [], 0
    for key, i, kind in (("sky-human", SKY, "human"), ("sky-ai", SKY, "ai"),
                         ("socrates-human", SOCRATES, "human"), ("socrates-ai", SOCRATES, "ai")):
        pairs = items()[i]["scores"][kind]
        got, want = _call(fn, pairs), average_nll(pairs)
        differs = not _close(got, want)
        k += differs
        rows.append([(html.escape(words["rows"][key]), ""), (_mono(str(len(pairs) + 1)), ""),
                     (_mono(_shown(got, seen.lang)), _OFF if differs else ""), (_mono(number(want, seen.lang, 3)), "")])
        said.append(f"{words['rows'][key]}: {_shown(got, seen.lang)}, {number(want, seen.lang, 3)}")
    after = words["differs"].format(k=k) if k else words["fine"]
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def flagged_picture(seen) -> str:
    """The run at the threshold 1.75, with the learner's shares beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["flagged"]
    demo = _DEMO_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for name, values in (("HUMAN", HUMAN_N), ("AI", AI_N)):
        got, want = _call(fn, values, THRESHOLD), flagged_share(values.value, THRESHOLD)
        differs = not _close(got, want)
        k += differs
        call = f"flagged_share({name}, {THRESHOLD})"
        rows.append([(_mono(call), ""), (_mono(_shown(got, seen.lang)), _OFF if differs else ""),
                     (_mono(number(want, seen.lang, 3)), "")])
        said.append(f"{call}: {_shown(got, seen.lang)}, {number(want, seen.lang, 3)}")
    after = words["differs"].format(k=k) if k else words["fine"]
    strip = _strip([(demo["human"], HUMAN, "var(--ea-ink)", _person_wrong), (demo["ai"], AI, "var(--ea-pred)", _ai_wrong)],
                   THRESHOLD, seen.lang)
    return yours.figure(words["title"], strip + _table(rows, words["head"]) + _note(after),
                        alt=w["alt"].format(title=words["title"], rows="; ".join(said), after=after))


def _squares(right: int, wrong: int) -> str:
    """A small square per flagged text: filled ink for a right accusation, the loss colour for a wrong one."""
    cell, per_row = 11, 27
    total = right + wrong
    rows_ = -(-total // per_row)
    p = []
    for i in range(total):
        x, y = (i % per_row) * cell, (i // per_row) * cell
        colour = "var(--ea-ink)" if i < right else "var(--ea-loss)"
        p.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" style="fill:{colour}"/>')
    return (f'<svg width="{per_row * cell}" height="{rows_ * cell}" viewBox="0 0 {per_row * cell} {rows_ * cell}" '
            f'aria-hidden="true" style="display:block;max-width:100%">{"".join(p)}</svg>')


def share_picture(seen) -> str:
    """The texts a detector flags among 10,000, and the learner's share of right accusations beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CHECK_WORDS[seen.lang]
    words = w["share"]
    got, want = _call(fn, 0.9, 0.01, 0.01), share_right(0.9, 0.01, 0.01)
    differs = not _close(got, want)
    call = "share_right(0.9, 0.01, 0.01)"
    rows = [[(_mono(call), ""), (_mono(_shown(got, seen.lang)), _OFF if differs else ""), (_mono(number(want, seen.lang, 3)), "")]]
    after = words["differs"] if differs else words["fine"]
    body = _squares(90, 99) + _note(words["squares"]) + _table(rows, words["head"]) + _note(after)
    return yours.figure(words["title"], body, alt=w["alt"].format(
        title=words["title"], rows=f"{words['squares']} {call}: {_shown(got, seen.lang)}, {number(want, seen.lang, 3)}",
        after=after))


# --- the live demo: a text of your own --------------------------------------------------------------------------------

DEMO_TEXT = "The sky appears blue because molecules in the air scatter blue light from the sun more than other colors."


def live_pairs(text: str):
    """The [token, logprob] pairs of the learner's text under the chat model, at most LIVE_TOKENS tokens: "memory"
    when the memory is full, "load" when loading the model failed otherwise."""
    try:
        bot = chat.loaded()
    except Exception as e:   # noqa: BLE001
        return "memory" if chat.memory_full(e) else "load"
    try:
        return score(bot.model, bot.enc, text, limit=LIVE_TOKENS)
    except Exception as e:
        if chat.memory_full(e):
            return "memory"
        raise


_LIVE_WORDS = {
    "en": {"title": "How surprised Qwen3 is by your text",
           "average": "Average NLL over {n} tokens: {avg}. At the threshold {t}, the detector of this lesson says: {verdict}.",
           "ai": "written by AI", "human": "written by a person",
           "cut": "Only the first {n} tokens count.",
           "key": "The darker a token, the more it surprised Qwen3. The first token has no score, so it is not shown.",
           "short": "Write at least two words: the first token has no score, so one token gives nothing to average.",
           "load": "Loading Qwen3 stops with an error. Close other programs and press the button again; if it happens again, report it on GitHub.",
           "memory": "The memory of your device is full. Close other programs, and press the button again."},
    "nl": {"title": "Hoe verrast Qwen3 is door jouw tekst",
           "average": "Gemiddelde NLL over {n} tokens: {avg}. Bij de drempel {t} zegt de detector van deze les: {verdict}.",
           "ai": "geschreven door AI", "human": "geschreven door een mens",
           "cut": "Alleen de eerste {n} tokens tellen.",
           "key": "Hoe donkerder een token, hoe meer het Qwen3 verraste. Het eerste token heeft geen score, dus het staat er niet bij.",
           "short": "Schrijf minstens twee woorden: het eerste token heeft geen score, dus bij één token valt er niets te middelen.",
           "load": "Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
           "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."},
}


def live_picture(pairs, lang: str, threshold: float = THRESHOLD) -> str:
    """The learner's text, token by token, darker where Qwen3 is more surprised, with its average NLL and what the
    detector says; or what went wrong."""
    w = _LIVE_WORDS[lang]
    if isinstance(pairs, str):
        return f'<p><em>{html.escape(w[pairs])}</em></p>'
    if not pairs:
        return f'<p><em>{html.escape(w["short"])}</em></p>'
    spans = []
    for token, logprob in pairs:
        share = round(min(-logprob, 10.0) * 7)   # up to 70% of the loss colour at a surprise of 10 or more
        spans.append(f'<span title="{html.escape(number(-logprob, lang))}" style="background:color-mix(in srgb, '
                     f'var(--ea-loss) {share}%, transparent);border-radius:2px">{html.escape(token)}</span>')
    avg = average_nll(pairs)
    verdict = w["ai"] if avg < threshold else w["human"]
    lines = [w["average"].format(n=len(pairs), avg=number(avg, lang), t=number(threshold, lang), verdict=verdict)]
    if len(pairs) + 1 >= LIVE_TOKENS:
        lines.append(w["cut"].format(n=LIVE_TOKENS))
    body = (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.7;white-space:pre-wrap;'
            f'overflow-wrap:anywhere;color:var(--ea-ink)">{"".join(spans)}</div>'
            + "".join(_note(x) for x in lines) + _note(w["key"]))
    return yours.figure(w["title"], body, alt=w["title"] + ". " + " ".join(lines))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["average-nll", "flagged-share", "share-right", "toefl", "rewrite"]

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
    return isinstance(e, NameError) and re.search(
        r"'(HUMAN|AI|SKY_HUMAN|SKY_AI|THRESHOLD|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `HUMAN`, `SKY_HUMAN` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `HUMAN`, `SKY_HUMAN` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


def _rounded(v) -> str:
    """A value for {got} and {want}: floats rounded to 3 decimals, also inside a tuple or a list."""
    if _is_number(v) and isinstance(v, float):
        text = f"{round(v, 3) + 0.0:.3f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-") else "0"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_rounded(x) for x in v)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    return repr(v)


AVERAGE_CASES = [(SKY_HUMAN_N,), (SKY_AI_N,), ([["a", -1.0], ["b", -3.0]],), ([["x", -0.5], ["y", -0.5], ["z", -2.0]],),
                 ([[" the", 0.0], [" end", -0.6931]],)]
FLAGGED_CASES = [(HUMAN_N, THRESHOLD), (AI_N, THRESHOLD), (HUMAN_N, 2.0), ([1.5, 2.0, 2.5], 2.0), ([0.4], 1.0),
                 ([3.0, 2.0], 1.0)]
SHARE_CASES = [(0.9, 0.01, 0.01), (0.9, 0.01, 0.1), (0.26, 0.09, 0.1), (0.75, 0.15, 0.2), (1.0, 0.0, 0.3),
               (0.5, 0.5, 0.5)]

# exercise 1: average_nll
_A_START = "def average_nll(pairs):\n    ...\n"
_A = ("def average_nll(pairs):\n"
      "    total = 0\n"
      "    for token, logprob in pairs:\n"
      "        total = total - logprob\n"
      "    return total / len(pairs)\n")
_A_SIGN = _A.replace("total - logprob", "total + logprob")
_A_SUM = _A.replace("return total / len(pairs)", "return total")
_A_LESS = _A.replace("len(pairs)", "(len(pairs) - 1)")

# exercise 2: flagged_share
_F_START = "def flagged_share(scores, threshold):\n    ...\n"
_F = ("def flagged_share(scores, threshold):\n"
      "    flagged = 0\n"
      "    for score in scores:\n"
      "        if score < threshold:\n"
      "            flagged = flagged + 1\n"
      "    return flagged / len(scores)\n")
_F_EQUAL = _F.replace("score < threshold", "score <= threshold")
_F_THROUGH = _F.replace("score < threshold", "score >= threshold")
_F_COUNT = _F.replace("return flagged / len(scores)", "return flagged")

# exercise 3: share_right
_S_START = "def share_right(caught, false_positive, share_ai):\n    ...\n"
_S = ("def share_right(caught, false_positive, share_ai):\n"
      "    right = caught * share_ai\n"
      "    wrong = false_positive * (1 - share_ai)\n"
      "    return right / (right + wrong)\n")
_S_NO_BASE = ("def share_right(caught, false_positive, share_ai):\n"
              "    return caught / (caught + false_positive)\n")
_S_FLAGGED = _S.replace("return right / (right + wrong)", "return right + wrong")
_S_WRONG = _S.replace("return right / (right + wrong)", "return wrong / (right + wrong)")


EXERCISES = {
    "average-nll": FunctionBox(
        name="average-nll",
        picture=average_picture,
        function="average_nll",
        start=_A_START,
        solution=_A,
        visible=[AVERAGE_CASES[0]],
        hidden=AVERAGE_CASES[1:],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("sign", wrong=_A_SIGN,
                    en="`{call}` gives {got}, and it should give {want}: your average is below zero. A log probability is never above zero, so adding them gives a negative total; a surprise is never below zero.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw gemiddelde ligt onder nul. Een log-kans is nooit groter dan nul, dus optellen geeft een negatief totaal; een verrassing is nooit kleiner dan nul."),
            Mistake("sum", wrong=_A_SUM,
                    en="`{call}` gives {got}, and it should give {want}: that is the total surprise of the text. A long text then always scores higher than a short one; the average is per token.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de totale verrassing van de tekst. Een lange tekst scoort dan altijd hoger dan een korte; het gemiddelde is per token."),
            Mistake("less", wrong=_A_LESS,
                    en="`{call}` gives {got}, and it should give {want}: your average divides by one less than the number of pairs. Every pair in `pairs` is a token with its score, so each one counts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw gemiddelde deelt door één minder dan het aantal paren. Elk paar in `pairs` is een token met zijn score, dus elk paar telt mee."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add up the surprise of every token, minus its log probability, and divide by the number of pairs.",
                      "nl": "Vervang de `...`: tel de verrassing van elk token op, min zijn log-kans, en deel door het aantal paren."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The average NLL is the mean surprise per token, and a token's surprise is minus its log probability.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De gemiddelde NLL is de gemiddelde verrassing per token, en de verrassing van een token is min zijn log-kans."},
            "almost": {"en": "Right for `SKY_HUMAN`. But `{call}` gives {got}, and it should give {want}. Does your function use the `pairs` it gets?",
                       "nl": "Goed voor `SKY_HUMAN`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `pairs` die ze krijgt?"},
            "correct": {"en": "Your `average_nll` is right in all {n} cases. The person's answer to \"Why is the sky blue?\" averages 2.29, Qwen3's own answer 1.01: Qwen3 is far less surprised by what it wrote itself. But who wrote it is not always what the number says: for Socrates, the person's answer averages 1.64 and Qwen3's 2.51.",
                        "nl": "Je `average_nll` klopt in alle {n} gevallen. Het antwoord van een mens op \"Why is the sky blue?\" komt gemiddeld op 2,29, het eigen antwoord van Qwen3 op 1,01: Qwen3 is veel minder verrast door wat het zelf schreef. Maar wie het schreef is niet altijd wat het getal zegt: bij Socrates komt het antwoord van een mens op 1,64 en dat van Qwen3 op 2,51."},
        },
    ),
    "flagged-share": FunctionBox(
        name="flagged-share",
        picture=flagged_picture,
        function="flagged_share",
        start=_F_START,
        solution=_F,
        visible=[FLAGGED_CASES[0]],
        hidden=FLAGGED_CASES[1:],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("equal", wrong=_F_EQUAL,
                    en="`{call}` gives {got}, and it should give {want}: your function also flags a score that is exactly the threshold. The detector calls a text AI only when its score is below the threshold.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie wijst ook een score aan die precies gelijk is aan de drempel. De detector noemt een tekst alleen AI als zijn score onder de drempel ligt."),
            Mistake("through", wrong=_F_THROUGH,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the share of the scores at or above the threshold, the texts the detector lets through. The exercise asks for the share it flags.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft het deel van de scores op of boven de drempel, de teksten die de detector doorlaat. De oefening vraagt om het deel dat hij aanwijst."),
            Mistake("count", wrong=_F_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your function gives how many texts the detector flags. The exercise asks for their share of all the texts in the list, a number from 0 to 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft hoeveel teksten de detector aanwijst. De oefening vraagt om hun deel van alle teksten in de lijst, een getal van 0 tot 1."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the scores below the threshold, and give back their share of all the scores.",
                      "nl": "Vervang de `...`: tel de scores onder de drempel, en geef hun deel van alle scores terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The detector flags a text when its score is below the threshold; the function gives the share of the scores it flags.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De detector wijst een tekst aan als zijn score onder de drempel ligt; de functie geeft het deel van de scores dat hij aanwijst."},
            "almost": {"en": "Right for `HUMAN` at 1.75. But `{call}` gives {got}, and it should give {want}. Does your function use the `scores` and `threshold` it gets?",
                       "nl": "Goed voor `HUMAN` bij 1,75. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `scores` en `threshold` die ze krijgt?"},
            "correct": {"en": "Your `flagged_share` is right in all {n} cases. At 1.75, the detector flags 15% of the answers by people, and 75% of Qwen3's: a quarter of Qwen3's answers get through. At 2.0, it flags 35% of the answers by people. Moving the threshold trades one mistake for the other.",
                        "nl": "Je `flagged_share` klopt in alle {n} gevallen. Bij 1,75 wijst de detector 15% van de antwoorden van mensen aan, en 75% van die van Qwen3: een kwart van de antwoorden van Qwen3 komt erdoor. Bij 2,0 wijst hij 35% van de antwoorden van mensen aan. Wie de drempel verplaatst, ruilt de ene fout in voor de andere."},
        },
    ),
    "share-right": FunctionBox(
        name="share-right",
        picture=share_picture,
        function="share_right",
        start=_S_START,
        solution=_S,
        visible=[SHARE_CASES[0]],
        hidden=SHARE_CASES[1:],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-base", wrong=_S_NO_BASE,
                    en="`{call}` gives {got}, and it should give {want}: that is what you get when texts by AI and by people are equally common. Here only a share `share_ai` of all texts is by AI, so the detector meets far more texts by people, and flags more of them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat krijg je als teksten van AI en van mensen even vaak voorkomen. Hier is maar een deel `share_ai` van alle teksten van AI, dus de detector komt veel meer teksten van mensen tegen, en wijst er ook meer van aan."),
            Mistake("flagged", wrong=_S_FLAGGED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the share of all texts that the detector flags, right and wrong together. The exercise asks which part of the flagged texts is really by AI.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft het deel van alle teksten dat de detector aanwijst, goed en fout samen. De oefening vraagt welk deel van de aangewezen teksten echt van AI is."),
            Mistake("wrong-share", wrong=_S_WRONG,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the share of the flagged texts that people wrote, the wrong accusations. The exercise asks for the right ones.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft het deel van de aangewezen teksten dat mensen schreven, de onterechte beschuldigingen. De oefening vraagt om de terechte."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out `right` and `wrong` as shares of all texts, and give back the share of `right` among the flagged texts.",
                      "nl": "Vervang de `...`: reken `right` en `wrong` uit als delen van alle teksten, en geef het deel van `right` onder de aangewezen teksten terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Of all texts, the detector flags the ones by AI that it catches and the ones by people that it gets wrong; the function gives the share of the first among both.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Van alle teksten wijst de detector de teksten van AI aan die hij vangt en de teksten van mensen die hij fout heeft; de functie geeft het deel van de eerste onder beide samen."},
            "almost": {"en": "Right for `share_right(0.9, 0.01, 0.01)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `caught`, `false_positive` and `share_ai` it gets?",
                       "nl": "Goed voor `share_right(0.9, 0.01, 0.01)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `caught`, `false_positive` en `share_ai` die ze krijgt?"},
            "correct": {"en": "Your `share_right` is right in all {n} cases. A detector that catches 90% of AI texts and flags 1% of texts by people is right in fewer than half of its accusations when 1 text in 100 is by AI: 90 of 189. With 1 in 10, it is right in 91%. The detector did not change; only how common AI text is.",
                        "nl": "Je `share_right` klopt in alle {n} gevallen. Een detector die 90% van de teksten van AI vangt en 1% van de teksten van mensen aanwijst, heeft in minder dan de helft van zijn beschuldigingen gelijk als 1 tekst op de 100 van AI is: 90 van de 189. Bij 1 op de 10 heeft hij in 91% gelijk. De detector veranderde niet; alleen hoe vaak tekst van AI voorkomt."},
        },
    ),
}

CASES = [
    ("average-nll", _A_START, "empty", None),
    ("average-nll", _A, "correct", None),
    ("average-nll", "def average_nll(pairs):\n    return -sum(p for t, p in pairs) / len(pairs)\n", "correct", None),
    ("average-nll", _A_SIGN, "wrong", "sign"),
    ("average-nll", _A_SUM, "wrong", "sum"),
    ("average-nll", _A_LESS, "wrong", "less"),
    ("average-nll", f"def average_nll(pairs):\n    return {average_nll(SKY_HUMAN)!r}\n" if SKY_HUMAN else _A, "almost" if SKY_HUMAN else "correct", None),
    ("average-nll", "def average_nll(pairs):\n    return len(SKY_HUMAN)\n", "error", "lesson-names"),
    ("flagged-share", _F_START, "empty", None),
    ("flagged-share", _F, "correct", None),
    ("flagged-share", "def flagged_share(scores, threshold):\n    return len([s for s in scores if s < threshold]) / len(scores)\n", "correct", None),
    ("flagged-share", _F_EQUAL, "wrong", "equal"),
    ("flagged-share", _F_THROUGH, "wrong", "through"),
    ("flagged-share", _F_COUNT, "wrong", "count"),
    ("flagged-share", "def flagged_share(scores, threshold):\n    return 0.15\n", "almost", None),
    ("flagged-share", "def flagged_share(scores, threshold):\n    return len(HUMAN)\n", "error", "lesson-names"),
    ("share-right", _S_START, "empty", None),
    ("share-right", _S, "correct", None),
    ("share-right", "def share_right(caught, false_positive, share_ai):\n    flagged = caught * share_ai + false_positive * (1 - share_ai)\n    return caught * share_ai / flagged\n", "correct", None),
    ("share-right", _S_NO_BASE, "wrong", "no-base"),
    ("share-right", _S_FLAGGED, "wrong", "flagged"),
    ("share-right", _S_WRONG, "wrong", "wrong-share"),
    ("share-right", "def share_right(caught, false_positive, share_ai):\n    return 90 / 189\n", "almost", None),
    ("share-right", "def share_right(caught, false_positive, share_ai):\n    return THRESHOLD\n", "error", "lesson-names"),
]
