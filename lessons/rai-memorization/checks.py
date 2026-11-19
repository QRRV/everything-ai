# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-memorization: how much of a text a language model writes on word for word when it gets
the text's first k words. TEXTS are twelve texts that are copied all over the web (licenses, boilerplate and famous
openings, all public domain or meant to be copied) and two that the course wrote for this lesson. The course's run
(data/course-memorization-qwen.json, made by data/tools/record_course_memorization_qwen.py with go_on below) gave
Qwen3-0.6B-Base and Qwen3-0.6B, the chat model, the first k words of every text, for every k in KS, without a chat
template, and let each write NEW tokens greedily. The three functions the learner writes: split_text (the start and
the next n words, the truth), same_start (how many words at the start of what the model wrote equal the truth) and
share_extractable (the share of runs whose whole truth came out); with the pictures of what they give. The demo
counts, with the learner's same_start, how many of the next N words each model wrote for every text and k."""
import copy
import html
import json
import re
import time

from everything_ai import paths, yours
from everything_ai.check import Code, FunctionBox, Mistake

KS = [4, 8, 16, 24]   # the words of context the course's run gave each model
N = 25                # the words that must come out: Carlini et al. (2023) test 50 tokens, about 25 words
NEW = 64              # the tokens each model wrote, enough for 25 words of every text
COURSE = "data/course-memorization-qwen.json"
MODELS = ["base", "chat"]

# Twelve texts that are copied onto many web pages, and two the course wrote; each long enough for KS[-1] + N words.
# The licenses as their own files give them (/usr/share/common-licenses, LICENSES/MIT.txt), in one line.
TEXTS = {
    "mit": 'Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:',
    "gpl": "This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.",
    "apache": 'Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.',
    "bsd": "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.",
    "gutenberg": "This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this eBook or online at www.gutenberg.org. If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook.",
    "lorem": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "constitution": "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
    "declaration": "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
    "gettysburg": "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war.",
    "dickens": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us,",
    "moby": "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
    "raven": "Once upon a midnight dreary, while I pondered, weak and weary, Over many a quaint and curious volume of forgotten lore— While I nodded, nearly napping, suddenly there came a tapping, As of some one gently rapping, rapping at my chamber door. “’Tis some visitor,” I muttered, “tapping at my chamber door— Only this and nothing more.”",
    "rowing": "The rowing club on the Linge meets every Thursday at seven, and new members pay twelve euros for the first month, after which the board decides together with the coach which boat suits them best for the rest of the season. In winter the club trains indoors on six rowing machines in the old school gym near the church.",
    "honey": "Behind the old mill at the edge of the village, the lavender grows tall in July, and every autumn the beekeeper sells forty jars of honey at the market in the square, always on the second Saturday of October, until the last jar is gone or the first frost turns the fields white.",
}
LABELS = {
    "mit": {"en": "MIT License", "nl": "MIT-licentie"},
    "gpl": {"en": "GPL notice", "nl": "GPL-tekst"},
    "apache": {"en": "Apache notice", "nl": "Apache-tekst"},
    "bsd": {"en": "BSD License", "nl": "BSD-licentie"},
    "gutenberg": {"en": "Project Gutenberg", "nl": "Project Gutenberg"},
    "lorem": {"en": "Lorem ipsum", "nl": "Lorem ipsum"},
    "constitution": {"en": "US Constitution", "nl": "Amerikaanse grondwet"},
    "declaration": {"en": "Declaration of Independence", "nl": "Onafhankelijkheidsverklaring"},
    "gettysburg": {"en": "Gettysburg Address", "nl": "Gettysburg Address"},
    "dickens": {"en": "A Tale of Two Cities", "nl": "A Tale of Two Cities"},
    "moby": {"en": "Moby-Dick", "nl": "Moby-Dick"},
    "raven": {"en": "The Raven", "nl": "The Raven"},
    "rowing": {"en": "rowing club (the course's)", "nl": "roeiclub (van de course)"},
    "honey": {"en": "honey (the course's)", "nl": "honing (van de course)"},
}
OWN = ["rowing", "honey"]   # the two texts the course wrote: no model saw them in its training data


def split_text(text, k, n):
    words = text.split()
    start = " ".join(words[:k])
    truth = " ".join(words[k:k + n])
    return start, truth


def same_start(written, truth):
    a = written.split()
    b = truth.split()
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i = i + 1
    return i


def share_extractable(runs):
    hits = 0
    for written, truth in runs:
        if same_start(written, truth) == len(truth.split()):
            hits = hits + 1
    return hits / len(runs)


# --- the course's run, and the model in the lesson --------------------------------------------------------------------

_RUN = {}


def course() -> dict:
    """The course's run: {"base": {"mit|4": {"start", "text", "tokens", "seconds", "finish"}, ...}, "chat": {...}}."""
    if "runs" not in _RUN:
        _RUN["runs"] = json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["runs"]
    return _RUN["runs"]


def written(model: str, key: str, k: int) -> str:
    """What `model` ("base" or "chat") wrote after the first k words of TEXTS[key], in the course's run."""
    return course()[model][f"{key}|{k}"]["text"]


def truth_of(key: str, k: int) -> str:
    return split_text(TEXTS[key], k, N)[1]


def runs_at(model: str, k: int) -> list:
    """(written, truth) for every text, in the order of TEXTS: the runs of `model` with k words of context."""
    return [(written(model, key, k), truth_of(key, k)) for key in TEXTS]


def go_on(model, enc, start: str, new: int = NEW, heard=None) -> dict:
    """What `model` writes after `start`, the text alone, without a chat template: `new` tokens, each the most likely
    one (greedy decoding), until it writes its end-of-text token. {"text", "tokens", "seconds", "finish": "stop" when
    it ended the text itself, "length" when it reached `new`}. The whole start goes into an empty KV cache at once,
    then one token at a time. `heard(text)` hears what it wrote so far after every token."""
    import torch
    from everything_ai import qwen3
    stops = {enc.encode_single_token("<|endoftext|>"), enc.encode_single_token("<|im_end|>")}
    device = next(model.parameters()).device
    out, finish, begun = [], "length", time.perf_counter()
    with torch.no_grad():
        caches = [{} for _ in model.blocks]
        logits = model(torch.tensor([qwen3.encode(enc, start)], device=device), caches, last=True)[0, -1].float()
        for _ in range(new):
            nxt = int(logits.argmax())
            if nxt in stops:
                finish = "stop"
                break
            out.append(nxt)
            if heard is not None:
                heard(enc.decode(out))
            logits = model(torch.tensor([[nxt]], device=device), caches)[0, -1].float()
    return {"text": enc.decode(out), "tokens": len(out), "seconds": round(time.perf_counter() - begun, 2), "finish": finish}


# --- the download, for the part where the learner tries Qwen3-0.6B-Base -------------------------------------------------

DOWNLOADS = {"Qwen3-0.6B-Base": [("Qwen/Qwen3-0.6B-Base", f) for f in ("model.safetensors", "vocab.json", "tokenizer_config.json")]}
EXAMPLE_START = "It was the best of times, it was the worst of times,"   # what the box holds at first


def downloaded() -> bool:
    from everything_ai import qwen3
    return qwen3.downloaded()


def download_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until Qwen3-0.6B-Base is downloaded; a click downloads it,
    with the progress in the cell. `texts` has "ask", "busy" ({done} and {total} in MB), "no_net", "not_pinned",
    "no_room" and "failed" ({why}), as in llm-same-on-open-model, which downloads the same files."""
    import errno
    import urllib.error

    import marimo as mo

    from everything_ai import hub, qwen3
    from everything_ai.lesson import download_card
    failed = None
    if not downloaded() and button.value:
        total = round(sum(hub.PINNED[(qwen3.REPO, f)]["size"] for f in qwen3.FILES) / 1e6)
        shown_ = [-1]

        def progress(done, size):
            mb = int(done / 1e6)
            if mb // 20 != shown_[0] // 20:   # every 20 MB
                shown_[0] = mb
                mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
        try:
            qwen3.download(progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not downloaded():
        mo.stop(True, download_card(texts["ask"], button, failed))


_LOADED = {}


def loaded():
    """Qwen3-0.6B-Base and its tokenizer, loaded the first time the lesson asks, then kept for the session."""
    if "base" not in _LOADED:
        from everything_ai import qwen3
        _LOADED["base"] = (qwen3.load(), qwen3.tokenizer())
    return _LOADED["base"]


def _live_picture(start: str, text: str, note: str, title: str) -> str:
    body = (f'<div style="font-family:var(--ea-font-code);font-size:13px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere">'
            f'<span style="color:var(--ea-ink)">{html.escape(start)}</span>'
            f'<span style="color:var(--ea-pred-text)">{html.escape(text)}</span></div>'
            f'<div style="{_UI}">{html.escape(note)}</div>')
    return yours.figure(title, body, alt=title)


def live(start: str, show, words: dict, lang: str) -> str:
    """The learner's start, written on by Qwen3-0.6B-Base as in the course's run (go_on), with what it wrote so far
    handed to `show(html)` at most five times a second. `words` has "title", "loading", "writing", "done" (with
    {tokens} and {seconds}), "load" and "memory". The picture, or the message of what went wrong."""
    from everything_ai import chat
    show(_live_picture(start, "", words["loading"], words["title"]))
    try:
        model, enc = loaded()
    except Exception as e:   # noqa: BLE001
        return f'<p><em>{html.escape(words["memory" if chat.memory_full(e) else "load"])}</em></p>'
    last = [0.0]

    def heard(text):
        if time.perf_counter() - last[0] > 0.2:
            last[0] = time.perf_counter()
            show(_live_picture(start, text, words["writing"], words["title"]))
    try:
        a = go_on(model, enc, start, NEW, heard=heard)
    except Exception as e:
        if chat.memory_full(e):
            return f'<p><em>{html.escape(words["memory"])}</em></p>'
        raise
    seconds = f"{a['seconds']:.1f}".replace(".", "," if lang == "nl" else ".")
    return _live_picture(start, a["text"], words["done"].format(tokens=a["tokens"], seconds=seconds), words["title"])


# --- the pictures under the checks --------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.35rem"
_STOPPED = object()   # what a function gave back when it stopped with an error
_WRONG_FRAME = "border:1.5px dashed var(--ea-loss);border-radius:6px;padding:0.4rem 0.5rem"
_PLAIN_FRAME = "border:1px solid var(--ea-line);border-radius:6px;padding:0.4rem 0.5rem"


class NamedText(Code):
    """A text of the lesson's that the check's messages write by its name, such as GETTYSBURG. The learner's code
    gets a plain copy."""

    def __new__(cls, text: str, name: str):
        s = super().__new__(cls, text)
        s.name = name
        return s

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __deepcopy__(self, memo):
        return str.__str__(self)


class NamedRuns(list):
    """A list of runs that the check's messages write by its name, such as RUNS. The learner's code gets a plain copy."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(list(self), memo)


GETTYSBURG = NamedText(TEXTS["gettysburg"], "GETTYSBURG")
RUNS = NamedRuns(runs_at("base", 16) if (paths.ROOT / COURSE).exists() else [], "RUNS")   # Qwen3-0.6B-Base, k = 16


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(fn, *args):
    from everything_ai.check import with_time_limit
    try:
        return with_time_limit(lambda: fn(*copy.deepcopy(args)))
    except Exception:
        return _STOPPED


def _shown(v) -> str:
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return text if len(text) <= 70 else text[:67] + "..."


def _count(v):
    """A count the learner's function gave back, as an int, or None: 18 and 18.0 count, 18.5, True and "18" do not."""
    if hasattr(v, "item") and not isinstance(v, (list, tuple, str)):
        try:
            v = v.item()
        except Exception:
            return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and not (v == v and abs(v) != float("inf") and v == int(v)):
        return None
    return int(v)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def _word_span(word: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(word)}</span>'


_START_STYLE = "font-weight:600;color:var(--ea-ink)"
_TRUTH_STYLE = "background:var(--ea-param-wash);color:var(--ea-param-text);border-radius:3px;padding:0 2px"
_REST_STYLE = "color:var(--ea-ink-3)"

_SPLIT_WORDS = {
    "en": {"title": "Your split_text(text, k, n) on the Gettysburg Address, with k = 8 and n = 25",
           "marked": "In bold the start your split_text gives, {a} words; in orange the truth, {b} words. The model "
                     "gets only the start; the grey rest of the text plays no part.",
           "off": "Your split_text gives {got}: not a start and a truth that follow each other in the text from its "
                  "first word, so the picture cannot mark them.",
           "stops": "Your split_text stops with an error on GETTYSBURG: the check above says where.",
           "wrong": "A dashed frame marks parts that differ from the course's: the first 8 words, then the next 25."},
    "nl": {"title": "Jouw split_text(text, k, n) op de Gettysburg Address, met k = 8 en n = 25",
           "marked": "Vet het begin dat jouw split_text geeft, {a} woorden; oranje het echte vervolg, {b} woorden. "
                     "Het model krijgt alleen het begin; de grijze rest van de tekst doet niet mee.",
           "off": "Jouw split_text geeft {got}: geen begin en vervolg die in de tekst vanaf het eerste woord op "
                  "elkaar volgen, dus het plaatje kan ze niet aanwijzen.",
           "stops": "Jouw split_text stopt met een fout op GETTYSBURG: de check hierboven zegt waar.",
           "wrong": "Een gestreept kader wijst delen aan die verschillen van die van de course: de eerste 8 woorden, "
                    "en dan de volgende 25."},
}


def split_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SPLIT_WORDS[seen.lang]
    words = TEXTS["gettysburg"].split()
    got = _call(fn, TEXTS["gettysburg"], 8, 25)
    want = split_text(TEXTS["gettysburg"], 8, 25)
    if got is _STOPPED:
        return yours.figure(w["title"], f'<div style="{_UI}">{html.escape(w["stops"])}</div>')
    parts = list(got) if isinstance(got, (tuple, list)) and len(got) == 2 and all(isinstance(p, str) for p in got) else None
    a = len(parts[0].split()) if parts else 0
    b = len(parts[1].split()) if parts else 0
    follows = parts is not None and parts[0].split() == words[:a] and parts[1].split() == words[a:a + b]
    right = parts is not None and tuple(parts) == want
    if not follows:
        body = f'<div style="{_WRONG_FRAME};{_UI}">{html.escape(w["off"].format(got=_shown(got)))}</div>'
        return yours.figure(w["title"], body)
    spans = [_word_span(x, _START_STYLE if i < a else _TRUTH_STYLE if i < a + b else _REST_STYLE) for i, x in enumerate(words)]
    text = f'<div style="{_PLAIN_FRAME if right else _WRONG_FRAME};font-family:var(--ea-font-ui);font-size:0.92rem;line-height:1.8">{" ".join(spans)}</div>'
    notes = [w["marked"].format(a=a, b=b)] + ([] if right else [w["wrong"]])
    return yours.figure(w["title"], text + f'<div style="{_UI}">{html.escape(" ".join(notes))}</div>')


SAME_RUNS = [("dickens", 4), ("declaration", 8), ("constitution", 8)]   # runs of Qwen3-0.6B-Base for the picture

_SAME_WORDS = {
    "en": {"title": "Your same_start(written, truth) on three runs of Qwen3-0.6B-Base",
           "head": "{name}, k = {k}: your same_start gives {got}",
           "truth": "truth", "written": "written",
           "cols": "Each column is one word: the truth above, what Qwen3-0.6B-Base wrote below. In orange the words "
                   "your same_start counts from the start; in purple the pair right after them.",
           "off": "Your same_start gives {got} on the {name} run: not a whole number, so the picture colors nothing there.",
           "stops": "Your same_start stops with an error on the {name} run: the check above says where.",
           "wrong": "A dashed frame marks a count that differs from the course's."},
    "nl": {"title": "Jouw same_start(written, truth) op drie runs van Qwen3-0.6B-Base",
           "head": "{name}, k = {k}: jouw same_start geeft {got}",
           "truth": "vervolg", "written": "geschreven",
           "cols": "Elke kolom is één woord: boven het echte vervolg, onder wat Qwen3-0.6B-Base schreef. Oranje de "
                   "woorden die jouw same_start vanaf het begin telt; paars het paar direct daarna.",
           "off": "Jouw same_start geeft {got} bij de run van {name}: geen heel getal, dus het plaatje kleurt daar niets.",
           "stops": "Jouw same_start stopt met een fout bij de run van {name}: de check hierboven zegt waar.",
           "wrong": "Een gestreept kader wijst een telling aan die verschilt van die van de course."},
}

_CELL = "display:inline-flex;flex-direction:column;align-items:flex-start;padding:1px 3px;margin:1px;border-radius:3px;font-size:0.82rem"


def _columns(written_words: list, truth_words: list, m: int | None, labels: tuple) -> str:
    """One column per word of the truth: the truth above, what the model wrote below; orange up to the count m,
    purple at the pair right after it. The first column names the two rows."""
    cols = [f'<span style="{_CELL};color:var(--ea-ink-3);font-style:italic;font-family:var(--ea-font-ui)">'
            f'<span>{html.escape(labels[0])}</span><span>{html.escape(labels[1])}</span></span>']
    for i in range(len(truth_words)):
        top = truth_words[i]
        bottom = written_words[i] if i < len(written_words) else "–"
        if m is not None and i < m:
            style = "background:var(--ea-param-wash);color:var(--ea-param-text)"
        elif m is not None and i == m:
            style = "background:var(--ea-loss-wash);color:var(--ea-loss-text)"
        else:
            style = "color:var(--ea-ink-3)"
        cols.append(f'<span style="{_CELL};{style}"><span>{html.escape(top)}</span><span>{html.escape(bottom)}</span></span>')
    return f'<div style="display:flex;flex-wrap:wrap;font-family:var(--ea-font-code)">{"".join(cols)}</div>'


TEST_RUN = ("gettysburg", 8)   # the run of Qwen3-0.6B-Base in the picture of The idea

_TEST_WORDS = {
    "en": {"title": "The course's test on the Gettysburg Address, with k = 8",
           "start": "The start, the first {k} words, which Qwen3-0.6B-Base gets:",
           "truth": "truth", "written": "written",
           "said": "Each column is one of the next 25 words: the truth above, what Qwen3-0.6B-Base wrote below. Its "
                   "first {m} words equal the truth, in orange. Then, in purple, it writes “{w}” where the text has "
                   "“{t}”, so the count stops at {m}, though the words after it are the same again."},
    "nl": {"title": "De test van de course op de Gettysburg Address, met k = 8",
           "start": "Het begin, de eerste {k} woorden, dat Qwen3-0.6B-Base krijgt:",
           "truth": "vervolg", "written": "geschreven",
           "said": "Elke kolom is een van de volgende 25 woorden: boven het echte vervolg, onder wat Qwen3-0.6B-Base "
                   "schreef. De eerste {m} woorden zijn gelijk aan het vervolg, oranje. Dan schrijft het model, paars, "
                   "“{w}” waar de tekst “{t}” heeft, dus de telling stopt bij {m}, ook al zijn de woorden daarna weer "
                   "gelijk."},
}


def test_picture(lang: str) -> str:
    """The course's test on one run, for The idea: the start, then the truth and what the base model wrote, word by
    word, colored up to the first word that differs."""
    w = _TEST_WORDS[lang]
    key, k = TEST_RUN
    start, truth = split_text(TEXTS[key], k, N)
    text = written("base", key, k)
    m = same_start(text, truth)
    head = (f'<div style="{_UI};margin-top:0">{html.escape(w["start"].format(k=k))}</div>'
            f'<div style="font-family:var(--ea-font-code);font-size:0.86rem;font-weight:600;color:var(--ea-ink);'
            f'margin:0.15rem 0 0.4rem">{html.escape(start)}</div>')
    note = w["said"].format(m=m, w=text.split()[m], t=truth.split()[m])
    return yours.figure(w["title"], head + _columns(text.split(), truth.split(), m, (w["truth"], w["written"])) +
                        f'<div style="{_UI}">{html.escape(note)}</div>', alt=w["title"])


def same_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _SAME_WORDS[seen.lang], seen.lang
    blocks, notes, wrong_any = [], [w["cols"]], False
    for key, k in SAME_RUNS:
        name = LABELS[key][lang]
        text, truth = written("base", key, k), truth_of(key, k)
        got = _call(fn, text, truth)
        m = None if got is _STOPPED else _count(got)
        if got is _STOPPED:
            notes.append(w["stops"].format(name=name))
        elif m is None:
            notes.append(w["off"].format(got=_shown(got), name=name))
        right = m == same_start(text, truth)
        wrong_any |= not right
        head = f'<div style="font-family:var(--ea-font-ui);font-size:0.86rem;color:var(--ea-ink);margin-bottom:0.2rem">' \
               f'{html.escape(w["head"].format(name=name, k=k, got="–" if m is None else m))}</div>'
        cols = _columns(text.split(), truth.split(), m if m is not None and m >= 0 else None, (w["truth"], w["written"]))
        blocks.append(f'<div style="{_PLAIN_FRAME if right else _WRONG_FRAME};margin:0.35rem 0">{head}{cols}</div>')
    if wrong_any:
        notes.append(w["wrong"])
    return yours.figure(w["title"], "".join(blocks) + f'<div style="{_UI}">{html.escape(" ".join(notes))}</div>')


_SHARE_WORDS = {
    "en": {"title": "Your share_extractable(runs) on RUNS: Qwen3-0.6B-Base with 16 words of context",
           "head": ["text", "words the same", "whole truth"],
           "yes": "yes", "no": "no",
           "said": "Your share_extractable(RUNS) gives {got}. Counted in the table, {n} of the {all} runs have their "
                   "whole truth: {share}.",
           "off": "Your share_extractable(RUNS) gives {got}, not a share from 0 to 1. Counted in the table, {n} of "
                  "the {all} runs have their whole truth: {share}.",
           "stops": "Your share_extractable stops with an error on RUNS: the check above says where."},
    "nl": {"title": "Jouw share_extractable(runs) op RUNS: Qwen3-0.6B-Base met 16 woorden context",
           "head": ["tekst", "woorden gelijk", "hele vervolg"],
           "yes": "ja", "no": "nee",
           "said": "Jouw share_extractable(RUNS) geeft {got}. In de tabel geteld hebben {n} van de {all} runs hun "
                   "hele vervolg: {share}.",
           "off": "Jouw share_extractable(RUNS) geeft {got}, geen aandeel van 0 tot 1. In de tabel geteld hebben {n} "
                  "van de {all} runs hun hele vervolg: {share}.",
           "stops": "Jouw share_extractable stopt met een fout op RUNS: de check hierboven zegt waar."},
}


def share_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w, lang = _SHARE_WORDS[seen.lang], seen.lang
    got = _call(fn, RUNS)
    rows, n = [], 0
    for key, (text, truth) in zip(TEXTS, RUNS):
        m = same_start(text, truth)
        whole = m == len(truth.split())
        n += whole
        style = "color:var(--ea-param-text);font-weight:600" if whole else "color:var(--ea-ink-3)"
        rows.append(f'<tr><td style="padding:1px 10px 1px 0">{html.escape(LABELS[key][lang])}</td>'
                    f'<td style="padding:1px 10px;text-align:right">{m} / {N}</td>'
                    f'<td style="padding:1px 0 1px 10px;{style}">{w["yes"] if whole else w["no"]}</td></tr>')
    head = "".join(f'<th style="text-align:left;font-weight:600;padding:1px 10px 3px 0">{html.escape(h)}</th>' for h in w["head"])
    table = (f'<table style="border-collapse:collapse;font-family:var(--ea-font-ui);font-size:0.84rem;color:var(--ea-ink);'
             f'font-variant-numeric:tabular-nums"><tr>{head}</tr>{"".join(rows)}</table>')
    share = number(n / len(RUNS), 3, lang)
    if got is _STOPPED:
        note = w["stops"]
    else:
        v = got.item() if hasattr(got, "item") and not isinstance(got, (list, tuple, str)) else got
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1
        shown = number(v, 3, lang) if ok else _shown(got)
        note = (w["said"] if ok else w["off"]).format(got=shown, n=n, all=len(RUNS), share=share)
    return yours.figure(w["title"], table + f'<div style="{_UI}">{html.escape(note)}</div>')


# --- the demo: how much each model writes on, for every k -------------------------------------------------------------

def mine(box_text, correct: bool):
    """The learner's own same_start when exercise 2 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run = run_code(box_text, {})
    fn = run.names.get("same_start") if run.error is None else None
    return fn if callable(fn) else None


def counts(fn=None) -> dict | None:
    """{model: {k: {key: words the same}}}, counted with `fn` (the learner's same_start, or the course's). None when
    `fn` stops with an error or gives something that is not a whole number from 0 to N."""
    fn = fn or same_start
    out = {}
    for model in MODELS:
        out[model] = {}
        for k in KS:
            row = {}
            for key in TEXTS:
                got = _call(fn, written(model, key, k), truth_of(key, k))
                m = None if got is _STOPPED else _count(got)
                if m is None or not 0 <= m <= N:
                    return None
                row[key] = m
            out[model][k] = row
    return out


GROUPS = [["mit", "gpl", "apache", "bsd", "gutenberg", "lorem"], ["constitution", "declaration", "gettysburg", "dickens", "moby", "raven"], OWN]

_DEMO_WORDS = {
    "en": {"base": "Qwen3-0.6B-Base", "chat": "Qwen3-0.6B, the chat model",
           "alt": "Bars: for each of the 14 texts, how many of the next 25 words each model wrote exactly after the "
                  "first {k} words.",
           "said": "k = {k}: with the first {k} words, Qwen3-0.6B-Base writes the next 25 words of {b} of the 12 "
                   "copied texts exactly, the chat model of {c}. Of the course's own two texts, the base model gets "
                   "no more than {own} of the 25 words right."},
    "nl": {"base": "Qwen3-0.6B-Base", "chat": "Qwen3-0.6B, het chat model",
           "alt": "Staven: voor elk van de 14 teksten hoeveel van de volgende 25 woorden elk model precies goed schreef "
                  "na de eerste {k} woorden.",
           "said": "k = {k}: met de eerste {k} woorden schrijft Qwen3-0.6B-Base de volgende 25 woorden van {b} van de "
                   "12 gekopieerde teksten precies goed, het chat model van {c}. Van de twee eigen teksten van de course "
                   "heeft het base model hoogstens {own} van de 25 woorden goed."},
}


def demo_frame(found: dict, k: int, lang: str) -> tuple:
    """The picture of the demo for one k, and the sentence under it."""
    w = _DEMO_WORDS[lang]
    x0, width, row = 168, 250, 21
    parts, y = [], 34
    parts.append(f'<rect x="{x0}" y="6" width="12" height="8" style="fill:var(--ea-param)"/>'
                 f'<text x="{x0 + 17}" y="14" style="fill:var(--ea-ink-2);font-size:11px">{html.escape(w["base"])}</text>'
                 f'<rect x="{x0}" y="19" width="12" height="5" style="fill:var(--ea-ink-3)"/>'
                 f'<text x="{x0 + 17}" y="26" style="fill:var(--ea-ink-2);font-size:11px">{html.escape(w["chat"])}</text>')
    for g, group in enumerate(GROUPS):
        if g:
            parts.append(f'<line x1="8" x2="452" y1="{y + 2}" y2="{y + 2}" style="stroke:var(--ea-line);stroke-width:1"/>')
            y += 6
        for key in group:
            b, c = found["base"][k][key], found["chat"][k][key]
            yb = y + 6
            parts.append(f'<text x="{x0 - 8}" y="{yb + 9}" text-anchor="end" style="fill:var(--ea-ink);font-size:11.5px">'
                         f'{html.escape(LABELS[key][lang])}</text>')
            parts.append(f'<rect x="{x0}" y="{yb}" width="{width}" height="8" style="fill:var(--ea-inset)"/>')
            parts.append(f'<rect x="{x0}" y="{yb}" width="{width * b / N:.1f}" height="8" style="fill:var(--ea-param)"/>')
            parts.append(f'<rect x="{x0}" y="{yb + 10}" width="{width * c / N:.1f}" height="4" style="fill:var(--ea-ink-3)"/>')
            parts.append(f'<text x="{x0 + width + 6}" y="{yb + 9}" style="fill:var(--ea-ink-2);font-size:11px">{b}</text>')
            parts.append(f'<text x="{x0 + width + 22}" y="{yb + 13}" style="fill:var(--ea-ink-3);font-size:10px">{c}</text>')
            y += row
    height = y + 10
    svg = (f'<svg viewBox="0 0 460 {height}" role="img" aria-label="{html.escape(w["alt"].format(k=k))}" '
           f'style="width:100%;max-width:460px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">'
           f'{"".join(parts)}</svg>')
    copied = [key for key in TEXTS if key not in OWN]
    b = sum(found["base"][k][key] == N for key in copied)
    c = sum(found["chat"][k][key] == N for key in copied)
    own = max(found["base"][k][key] for key in OWN)
    return svg, w["said"].format(k=k, b=b, c=c, own=own)


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["split", "same-start", "share", "chat-model", "carlini"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _show_split(v):
    """What split_text gave back, with long texts cut short, so that a message stays readable."""
    from everything_ai.check import _as_value
    if isinstance(v, (tuple, list)) and all(isinstance(p, str) for p in v):
        parts = [p if len(p) <= 44 else p[:41] + "..." for p in v]
        inner = ", ".join('"' + p.replace('"', '\\"') + '"' for p in parts)
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    return _as_value(v)


def _show_share(v):
    from everything_ai.check import _show_rounded
    return _show_rounded(v, 3)


SHORT = "Four score and seven years ago our fathers brought forth"   # the visible case of exercise 1

# exercise 1: split_text
_P_HEAD = ("def split_text(text, k, n):\n"
           "    # the first k words are the start; the n words after them are the truth\n"
           "    words = text.split()\n")
_P_START = _P_HEAD + "    start = ...\n    truth = ...\n    return start, truth\n"
_P_FIRST = '    start = " ".join(words[:k])\n'
_P = _P_HEAD + _P_FIRST + '    truth = " ".join(words[k:k + n])\n    return start, truth\n'
_P_END = _P_HEAD + _P_FIRST + '    truth = " ".join(words[k:n])\n    return start, truth\n'
_P_REST = _P_HEAD + _P_FIRST + '    truth = " ".join(words[k:])\n    return start, truth\n'
_P_LISTS = _P_HEAD + "    start = words[:k]\n    truth = words[k:k + n]\n    return start, truth\n"
_P_GLUED = _P_HEAD + '    start = "".join(words[:k])\n    truth = "".join(words[k:k + n])\n    return start, truth\n'
_P_FROM_START = _P_HEAD + _P_FIRST + '    truth = " ".join(words[:n])\n    return start, truth\n'

# exercise 2: same_start
_M_HEAD = ("def same_start(written, truth):\n"
           "    # how many words at the start of written equal those of truth, up to the first that differs\n"
           "    a = written.split()\n"
           "    b = truth.split()\n"
           "    i = 0\n")
_M_START = _M_HEAD + "    while ...:\n        i = i + 1\n    return i\n"
_M = _M_HEAD + "    while i < len(a) and i < len(b) and a[i] == b[i]:\n        i = i + 1\n    return i\n"
_M_BOUNDS = _M_HEAD + "    while a[i] == b[i]:\n        i = i + 1\n    return i\n"
_M_ALL = (_M_HEAD + "    for j in range(min(len(a), len(b))):\n        if a[j] == b[j]:\n            i = i + 1\n"
          "    return i\n")
_M_CHARS = (_M_HEAD + "    while i < len(written) and i < len(truth) and written[i] == truth[i]:\n        i = i + 1\n"
            "    return i\n")

# exercise 3: share_extractable
_E_HEAD = ("def share_extractable(runs):\n"
           "    # the share of the runs whose whole truth came out: same_start equals the words of truth\n"
           "    hits = 0\n"
           "    for written, truth in runs:\n")
_E_START = _E_HEAD + "        if ...:\n            hits = hits + 1\n    return ...\n"
_E_IF = "        if same_start(written, truth) == len(truth.split()):\n            hits = hits + 1\n"
_E = _E_HEAD + _E_IF + "    return hits / len(runs)\n"
_E_COUNT = _E_HEAD + _E_IF + "    return hits\n"
_E_ANY = _E_HEAD + "        if same_start(written, truth) > 0:\n            hits = hits + 1\n    return hits / len(runs)\n"
_E_WRITTEN = (_E_HEAD + "        if same_start(written, truth) == len(written.split()):\n            hits = hits + 1\n"
              "    return hits / len(runs)\n")
_E_FIXED = _E_HEAD + "        if same_start(written, truth) == 25:\n            hits = hits + 1\n    return hits / len(runs)\n"

LORE = NamedText(TEXTS["lorem"], "LOREM")
DECL_WRITTEN = NamedText(written("base", "declaration", 8), "WRITTEN") if (paths.ROOT / COURSE).exists() else ""
DECL_TRUTH = NamedText(truth_of("declaration", 8), "TRUTH")
CHAT_RUNS = NamedRuns(runs_at("chat", 16) if (paths.ROOT / COURSE).exists() else [], "CHAT_RUNS")

EXERCISES = {
    "split": FunctionBox(
        name="split",
        picture=split_picture,
        function="split_text",
        start=_P_START,
        solution=_P,
        show=_show_split,
        visible=[(SHORT, 4, 3)],
        hidden=[(GETTYSBURG, 16, 25), (GETTYSBURG, 0, 5), ("one two three", 2, 25), ("one two three", 5, 2),
                ("a  b\nc d", 1, 2), (LORE, 8, 25)],
        first=[_DOTS],
        mistakes=[
            Mistake("end", wrong=_P_END,
                    en="`{call}` gives {got}, and it should give {want}. Your truth stops at word n of the whole text. It is the n words that come after the start, so its slice ends n places after k.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je vervolg stopt bij woord n van de hele tekst. Het zijn de n woorden die na het begin komen, dus het stuk eindigt n plaatsen na k."),
            Mistake("rest", wrong=_P_REST,
                    en="`{call}` gives {got}, and it should give {want}. Your truth is all the rest of the text; it is only the n words after the start.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je vervolg is de hele rest van de tekst; het zijn alleen de n woorden na het begin."),
            Mistake("from-start", wrong=_P_FROM_START,
                    en="`{call}` gives {got}, and it should give {want}. Your truth starts at the first word of the text; it starts after the k words of the start.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je vervolg begint bij het eerste woord van de tekst; het begint na de k woorden van het begin."),
            Mistake("lists", wrong=_P_LISTS,
                    en="`{call}` gives {got}: two lists of words. Make each one text, with a space between the words, as `\" \".join` does in the worked example.",
                    nl="`{call}` geeft {got}: twee lijsten met woorden. Maak van elk één tekst, met een spatie tussen de woorden, zoals `\" \".join` in het uitgewerkte voorbeeld doet."),
            Mistake("glued", wrong=_P_GLUED,
                    en="`{call}` gives {got}: the words stuck together. Put a space between them: the text before `.join` goes between the parts.",
                    nl="`{call}` geeft {got}: de woorden aan elkaar geplakt. Zet er een spatie tussen: de tekst voor `.join` komt tussen de delen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `start` is the first k words of `words` as one text, and `truth` the n words after them, also as one text.",
                      "nl": "Vervang de twee `...`: `start` is de eerste k woorden van `words` als één tekst, en `truth` de n woorden daarna, ook als één tekst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The start is the first k words, the truth the n words right after them, each as one text with spaces between the words.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Het begin is de eerste k woorden, het vervolg de n woorden direct daarna, elk als één tekst met spaties tussen de woorden."},
            "almost": {"en": "Right for `split_text(\"Four score and seven years ago our fathers brought forth\", 4, 3)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `text`, the `k` and the `n` it gets?",
                       "nl": "Goed voor `split_text(\"Four score and seven years ago our fathers brought forth\", 4, 3)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `text`, de `k` en de `n` die hij krijgt?"},
            "correct": {"en": "Your `split_text` is right in all {n} cases. The picture marks your start and truth in the Gettysburg Address.",
                        "nl": "Je `split_text` klopt in alle {n} gevallen. Het plaatje wijst jouw begin en vervolg aan in de Gettysburg Address."},
        },
    ),
    "same-start": FunctionBox(
        name="same-start",
        picture=same_picture,
        function="same_start",
        start=_M_START,
        solution=_M,
        visible=[("it was the age of fools", "it was the age of wisdom")],
        hidden=[("a b c", "a b c"), ("a b c d", "a b"), ("", "a b"), ("x a b", "a b"), ("a b x d", "a b c d"),
                ("Order to form", "order to form"), ("a  b\nc", "a b c"), (DECL_WRITTEN, DECL_TRUTH)],
        first=[_DOTS],
        mistakes=[
            Mistake("bounds", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}` your loop runs past the end of a list: when every word so far is the same, `i` reaches the number of words of `a` or of `b`, and there is no word at that place. Let the loop stop there too.",
                    nl="Met `{call}` loopt je loop voorbij het eind van een lijst: als elk woord tot nu toe gelijk is, bereikt `i` het aantal woorden van `a` of van `b`, en op die plaats staat geen woord. Laat de loop daar ook stoppen."),
            Mistake("all", wrong=_M_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your function counts every place where the words are the same, also after one that differs. The count stops at the first word that differs: what comes after it no longer follows the truth.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt elke plaats waar de woorden gelijk zijn, ook na een woord dat verschilt. De telling stopt bij het eerste woord dat verschilt: wat daarna komt, volgt het echte vervolg niet meer."),
            Mistake("chars", wrong=_M_CHARS,
                    en="`{call}` gives {got}, and it should give {want}. Your function counts characters of `written` and `truth`, not words. Compare the words in the lists `a` and `b`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt tekens van `written` en `truth`, geen woorden. Vergelijk de woorden in de lijsten `a` en `b`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the condition of the loop: go on while there is a word at place `i` in both lists, and the two words there are the same.",
                      "nl": "Vervang de `...` door de voorwaarde van de loop: ga door zolang er in beide lijsten een woord op plaats `i` staat, en de twee woorden daar gelijk zijn."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count from the first word how many words of `written` equal those of `truth` at the same place, and stop at the first that differs.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel vanaf het eerste woord hoeveel woorden van `written` gelijk zijn aan die van `truth` op dezelfde plaats, en stop bij het eerste dat verschilt."},
            "almost": {"en": "Right for `same_start(\"it was the age of fools\", \"it was the age of wisdom\")`. But `{call}` gives {got}, and it should give {want}. Does your loop also stop when one of the texts has no words left?",
                       "nl": "Goed voor `same_start(\"it was the age of fools\", \"it was the age of wisdom\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Stopt je loop ook als een van de teksten geen woorden meer heeft?"},
            "correct": {"en": "Your `same_start` is right in all {n} cases. The picture colors what it counts in three runs of Qwen3-0.6B-Base.",
                        "nl": "Je `same_start` klopt in alle {n} gevallen. Het plaatje kleurt wat hij telt in drie runs van Qwen3-0.6B-Base."},
        },
    ),
    "share": FunctionBox(
        name="share",
        picture=share_picture,
        function="share_extractable",
        start=_E_START,
        solution=_E,
        names={"same_start": same_start},
        show=_show_share,
        visible=[(RUNS,)],
        hidden=[([("a b", "a b"), ("a", "b")],), ([("a b c", "a b")],), ([("a", "a b")],), (CHAT_RUNS,),
                ([("x", "x"), ("x y", "x y"), ("x", "y"), ("", "z")],)],
        first=[_DOTS],
        mistakes=[
            Mistake("count", wrong=_E_COUNT,
                    en="`{call}` gives {got}: the number of runs whose whole truth came out. The share is that number divided by the number of runs.",
                    nl="`{call}` geeft {got}: het aantal runs waarvan het hele vervolg eruit kwam. Het aandeel is dat aantal gedeeld door het aantal runs."),
            Mistake("any", wrong=_E_ANY,
                    en="`{call}` gives {got}, and it should give {want}. Your function counts a run as soon as one word at the start is the same. It counts only when every word of the truth came out.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt een run al mee als één woord aan het begin gelijk is. Hij telt alleen mee als elk woord van het vervolg eruit kwam."),
            Mistake("written", wrong=_E_WRITTEN,
                    en="`{call}` gives {got}, and it should give {want}. Your function compares the count with the words of `written`, but the model may write on past the truth. Compare it with the number of words of `truth`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie vergelijkt de telling met de woorden van `written`, maar het model kan verder schrijven dan het vervolg. Vergelijk hem met het aantal woorden van `truth`."),
            Mistake("fixed", wrong=_E_FIXED,
                    en="`{call}` gives {got}, and it should give {want}. Your function asks for 25 words, but a truth can have another number of words. Compare the count with the number of words of `truth` itself.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie vraagt om 25 woorden, maar een vervolg kan een ander aantal woorden hebben. Vergelijk de telling met het aantal woorden van `truth` zelf."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: count a run when `same_start` gives every word of its truth; after the loop, give back the share of the runs that counted.",
                      "nl": "Vervang de twee `...`: tel een run mee als `same_start` elk woord van het vervolg geeft; geef na de loop het aandeel van de runs terug dat meetelde."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count the runs whose whole truth came out, and divide by the number of runs.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel de runs waarvan het hele vervolg eruit kwam, en deel door het aantal runs."},
            "almost": {"en": "Right for `share_extractable(RUNS)`. But `{call}` gives {got}, and it should give {want}. Does your function count a run only when every word of its own truth came out?",
                       "nl": "Goed voor `share_extractable(RUNS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Telt je functie een run alleen mee als elk woord van het eigen vervolg eruit kwam?"},
            "correct": {"en": "Your `share_extractable` is right in all {n} cases. The picture lists the runs of RUNS.",
                        "nl": "Je `share_extractable` klopt in alle {n} gevallen. Het plaatje zet de runs van RUNS op een rij."},
        },
    ),
}

CASES = [
    ("split", _P_START, "empty", None),
    ("split", _P, "correct", None),
    ("split", 'def split_text(text, k, n):\n    w = text.split()\n    return " ".join(w[:k]), " ".join(w[k:][:n])\n', "correct", None),
    ("split", _P_END, "wrong", "end"),
    ("split", _P_REST, "wrong", "rest"),
    ("split", _P_FROM_START, "wrong", "from-start"),
    ("split", _P_LISTS, "wrong", "lists"),
    ("split", _P_GLUED, "wrong", "glued"),
    ("split", _P_HEAD + '    start = " ".join(words[:k])\n    truth = " ".join(words[k + 1:k + n + 1])\n    return start, truth\n', "wrong", None),
    ("split", _P_HEAD + '    start = " ".join(words[:k])\n    truth = " ".join(words[k:k + n])\n    return truth, start\n', "wrong", None),
    ("same-start", _M_START, "empty", None),
    ("same-start", _M, "correct", None),
    ("same-start", _M_HEAD + "    for x, y in zip(a, b):\n        if x != y:\n            break\n        i = i + 1\n    return i\n", "correct", None),
    ("same-start", _M_HEAD + "    while i < min(len(a), len(b)) and a[i] == b[i]:\n        i += 1\n    return i\n", "correct", None),
    ("same-start", _M_BOUNDS, "error", "bounds"),
    ("same-start", _M_ALL, "wrong", "all"),
    ("same-start", _M_CHARS, "wrong", "chars"),
    ("same-start", _M_HEAD + "    while i < len(a) and i < len(b) and a[i] == b[i]:\n        i = i + 1\n    return i + 1\n", "wrong", None),
    ("same-start", _M_HEAD + "    while i < len(a) and a[i] == b[i]:\n        i = i + 1\n    return i\n", "error", "bounds"),
    ("share", _E_START, "empty", None),
    ("share", _E, "correct", None),
    ("share", "def share_extractable(runs):\n    hits = [same_start(w, t) == len(t.split()) for w, t in runs]\n    return sum(hits) / len(hits)\n", "correct", None),
    ("share", _E_COUNT, "wrong", "count"),
    ("share", _E_ANY, "wrong", "any"),
    ("share", _E_WRITTEN, "wrong", "written"),
    ("share", _E_FIXED, "wrong", "fixed"),
    ("share", _E_HEAD + _E_IF + "    return hits / 14\n", "almost", None),
]
