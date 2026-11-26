# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""ELIZA the way Joseph Weizenbaum's own program of 1965 ran it, for lessons of domain taal, starting with nlp-eliza
(ELIZA: a chatbot from rules).

The code is the course's own. It follows the MAD-SLIP source of ELIZA that Jeff Shrager and the archivist Myles
Crowley found in Weizenbaum's papers at MIT in 2021, as transcribed in https://github.com/rupertl/eliza-ctss (commit
3af426b7), and Anthony Hay's reconstruction https://github.com/anthay/ELIZA (CC0), whose notes explain the SLIP HASH
function and the matching routine YMATCH.

data/eliza-doctor-1965.txt is the DOCTOR script of 1965 (.TAPE. 100), unchanged. data/course-eliza-nl.txt is the
course's own Dutch script in the same form. tests/test_eliza.py replays the conversation of Weizenbaum's paper of 1966
and checks every reply against etc/running-eliza.txt of eliza-ctss, the 1965 program answering that conversation.

One reply, step by step:
1. The input becomes a list of words in capitals; a period and a comma are words of their own. The course reads ? and
   ! as a period: on the typewriter of 1965 a question mark deleted the line, so the program never saw one.
2. The words are scanned from left to right. At a period, a comma or BUT: if no keyword has been seen yet, the words
   up to and including it are dropped and the scan goes on; otherwise it and everything after it are dropped. The
   course adds MAAR, Dutch for but, for the Dutch script.
3. A word of the script may have a substitute (I = YOU): it is swapped in place. A word with rules is a keyword; the
   keyword with the highest rank wins, the first one on a tie. A keyword without a rank has rank 0.
4. With no keyword, ELIZA says its oldest memory when its counter is at 4 (the counter goes 1, 2, 3, 4 and starts
   again, one step per input, and is at 2 for the first input); otherwise it uses the rules of NONE.
5. If the keyword is the MEMORY keyword (MY), one of the four memory rules turns the text into a memory for later;
   the SLIP HASH of the last word picks which one.
6. The keyword's decomposition rules are tried in order; (=WHAT) goes on with the rules of WHAT. The first rule that
   matches cuts the text into parts. Its reassembly rules take turns, and a number in one stands for that part.
7. If no rule matches, ELIZA says one of four fixed lines of its own, picked by the same counter."""
import functools
import re
from dataclasses import dataclass, field

from everything_ai import paths

DOCTOR = paths.ROOT / "data" / "eliza-doctor-1965.txt"
DUTCH = paths.ROOT / "data" / "course-eliza-nl.txt"

DELIMITERS = (".", ",", "BUT", "MAAR")   # the program's own three, and MAAR, Dutch for but, for the Dutch script
NOMACH = ("PLEASE CONTINUE", "HMMM", "GO ON , PLEASE", "I SEE")      # the program's own lines, by counter 1 to 4
NOMACH_NL = ("GA DOOR", "HM", "GA DOOR , ALSJEBLIEFT", "IK BEGRIJP HET")


# --- The script --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A decomposition rule with its reassembly rules. An element of the decomposition is "0" (any number of words),
    a number n (exactly n words), a word, or a tuple: ("*", A, B) is one of those words, ("/", TAG) a word with that
    tag."""
    decomposition: tuple
    reassemblies: tuple


@dataclass(frozen=True)
class Link:
    """(=WHAT): go on with the rules of another keyword."""
    target: str


@dataclass(frozen=True)
class Entry:
    word: str
    substitute: str | None = None
    tags: frozenset = frozenset()
    rank: int = 0
    rules: tuple = ()


@dataclass(frozen=True)
class Script:
    greeting: tuple
    entries: dict
    none: tuple
    memory_key: str
    memory_rules: tuple
    nomatch: tuple = NOMACH

    @property
    def keywords(self):
        """Every word with rules, with its rank."""
        return {w: e.rank for w, e in self.entries.items() if e.rules}

    @property
    def swaps(self):
        """Every word with a substitute, with that substitute."""
        return {w: e.substitute for w, e in self.entries.items() if e.substitute is not None}

    @property
    def tags(self):
        return {w: e.tags for w, e in self.entries.items() if e.tags}


_TOKEN = re.compile(r"[()]|[^\s()]+")


def _tokens(text):
    out = []
    for t in _TOKEN.findall(text):
        if t in "()":
            out.append(t)
            continue
        if t.startswith("=") and len(t) > 1:          # (=WHAT) is (= WHAT)
            out += ["=", t[1:]]
            continue
        parts = re.split(r"([.,])", t)                 # a period and a comma are words of their own
        out += [p for p in parts if p]
    return out


def _lists(tokens):
    """The script's lists, as nested Python lists; DLIST(...) becomes the pair "DLIST", [...]."""
    stack = [[]]
    for t in tokens:
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return stack[0]


def _element(x):
    if isinstance(x, list):
        words = [w for w in x]
        if words and words[0].startswith(("*", "/")) and words[0] not in ("*", "/"):
            words = [words[0][0], words[0][1:]] + words[1:]
        return tuple(words)
    return x


def _rule(x):
    if x and x[0] == "=":
        return Link(x[1])
    decomposition = tuple(_element(e) for e in x[0])
    return Rule(decomposition, tuple(tuple(r) for r in x[1:]))


def read_script(text, nomatch=NOMACH):
    """A script in the form of .TAPE. 100: the greeting, then one list per word, then ()."""
    forms = _lists(_tokens(text))
    greeting, entries, none, memory_key, memory_rules = tuple(forms[0]), {}, (), None, ()
    for form in forms[1:]:
        if not form:
            break
        head = form[0]
        if head == "NONE":
            none = tuple(_rule(r) for r in form[1:])
        elif head == "MEMORY":
            memory_key = form[1]
            rules = []
            for r in form[2:]:
                i = r.index("=")
                rules.append(Rule(tuple(_element(e) for e in r[:i]), (tuple(r[i + 1:]),)))
            memory_rules = tuple(rules)
        else:
            rest, substitute, tags, rank = form[1:], None, frozenset(), 0
            while rest and not isinstance(rest[0], list):
                if rest[0] == "=":
                    substitute, rest = rest[1], rest[2:]
                elif rest[0] == "DLIST":
                    tags, rest = frozenset(_element(rest[1])[1:]), rest[2:]
                else:
                    rank, rest = int(rest[0]), rest[1:]
            entries[head] = Entry(head, substitute, tags, rank, tuple(_rule(r) for r in rest))
    return Script(greeting, entries, none, memory_key, memory_rules, nomatch)


@functools.cache
def doctor():
    """Weizenbaum's DOCTOR script of 1965."""
    return read_script(DOCTOR.read_text(encoding="utf-8"))


@functools.cache
def dutch():
    """The course's Dutch script, in the same form."""
    return read_script(DUTCH.read_text(encoding="utf-8"), NOMACH_NL)


def script(lang):
    return dutch() if lang == "nl" else doctor()


# --- The steps of one reply --------------------------------------------------------------------------------------

def words(text):
    """The input as ELIZA reads it: words in capitals, with a period and a comma as words of their own."""
    text = re.sub(r"[?!]", ".", text.upper())
    return [p for p in re.split(r"\s+|([.,])", text) if p]


def clause(ws, keywords):
    """The words ELIZA keeps (step 2): the first part between periods, commas and BUTs that holds a keyword, or else
    the last part."""
    kept, found = [], False
    for w in ws:
        if w in DELIMITERS:
            if found:
                break
            kept = []
            continue
        kept.append(w)
        if w in keywords:
            found = True
    return kept


def swap(ws, swaps):
    return [swaps[w] if w in swaps else w for w in ws]


def keyword(ws, ranks):
    """The keyword with the highest rank, the first one on a tie; None if there is none."""
    best = None
    for w in ws:
        if w in ranks and (best is None or ranks[w] > ranks[best]):
            best = w
    return best


def _fits(element, word, tags):
    if isinstance(element, tuple):
        if element[0] == "*":
            return word in element[1:]
        return any(t in tags.get(word, ()) for t in element[1:])
    return element == word


def _number(element):
    if isinstance(element, str) and element.isdigit():
        return int(element)
    return None


def decompose(pattern, ws, tags=None):
    """The parts of ws that pattern cuts out, or None if it does not match. Like the SLIP routine YMATCH: the pattern is
    split into segments that each start at a 0; a 0 takes as few words as its segment allows, and the last one takes
    every word that is left. There is no going back to an earlier segment."""
    tags = tags or {}
    parts, w, p = [None] * len(pattern), 0, 0
    while p < len(pattern):
        end = p + 1 if _number(pattern[p]) == 0 else p
        while end < len(pattern) and _number(pattern[end]) != 0:
            end += 1
        wild = _number(pattern[p]) == 0
        fixed = sum((_number(e) or 1) if _number(e) != 0 else 0 for e in pattern[p:end])
        room = len(ws) - w - fixed
        if room < 0:
            return None
        lengths = [room] if wild and end == len(pattern) else range(room + 1) if wild else [0]
        for n in lengths:
            got, i, ok = {}, w + n, True
            for q in range(p + wild, end):
                k = _number(pattern[q])
                if k:
                    got[q], i = ws[i:i + k], i + k
                elif _fits(pattern[q], ws[i], tags):
                    got[q], i = [ws[i]], i + 1
                else:
                    ok = False
                    break
            if ok:
                if wild:
                    got[p] = ws[w:w + n]
                for q, part in got.items():
                    parts[q] = part
                w, p = i, end
                break
        else:
            return None
    return parts if w == len(ws) else None


def reassemble(template, parts):
    """The reply: every number in template is replaced by the words of that part (1 is the first)."""
    out = []
    for w in template:
        n = _number(w)
        out += parts[n - 1] if n else [w]
    return out


def slip_hash(word, n):
    """The SLIP HASH of the last six characters of word, as an n-bit number (Weizenbaum 1966: ALWAYS gives 14 with 7
    bits). The characters are coded the way the IBM 7094 coded them, six bits each, padded with spaces."""
    start = (len(word) - 1) // 6 * 6 if word else 0
    d = 0
    for c in word[start:].ljust(6):
        d = d << 6 | _BCD.get(c, ord(c) & 0x3F)
    return middle_square(d, n)


def middle_square(d, n):
    """The middle n bits of d squared, where d is a 36-bit word of the IBM 7094 whose first bit is its sign."""
    d &= (1 << 35) - 1
    return d * d >> (35 - n // 2) & ((1 << n) - 1)


_BCD = {c: i for i, c in enumerate("0123456789\0='\0\0\0+ABCDEFGHI\0.)\0\0\0-JKLMNOPQR\0$*\0\0\0 /STUVWXYZ\0,(\0\0\0")
        if c != "\0"}


# --- A conversation ----------------------------------------------------------------------------------------------

@dataclass
class Step:
    """What ELIZA did with one input, for the pictures."""
    said: str
    words: list
    kept: list
    swapped: list
    keyword: str | None
    how: str                       # "rule", "memory", "none" or "nomatch"
    rule: Rule | None = None
    via: list = field(default_factory=list)   # keywords that (=WHAT) led to
    parts: list | None = None
    template: tuple | None = None
    reply: list = field(default_factory=list)
    remembered: list | None = None
    counter: int = 0

    @property
    def text(self):
        return " ".join(self.reply)


class Eliza:
    """One conversation. swap=, keyword= and reassemble= put other versions of those steps in place of the course's
    own, so a lesson can run the learner's functions."""

    def __init__(self, script, swap=swap, keyword=keyword, reassemble=reassemble):
        self.script, self.counter, self.memories, self.turns = script, 1, [], {}
        self._swap, self._keyword, self._reassemble = swap, keyword, reassemble

    @property
    def greeting(self):
        return " ".join(self.script.greeting)

    def reply(self, text):
        return self.step(text).text

    def step(self, text):
        s = self.script
        self.counter = self.counter % 4 + 1
        ws = words(text)
        kept = clause(ws, s.keywords)
        swapped = list(self._swap(list(kept), dict(s.swaps)))
        key = self._keyword(list(kept), dict(s.keywords))
        step = Step(text, ws, kept, swapped, key, "rule", counter=self.counter)
        if key is None:
            if self.counter == 4 and self.memories:
                step.how, step.reply = "memory", self.memories.pop(0)
                return step
            return self._apply(step, "NONE", s.none, swapped)
        if key == s.memory_key and swapped:
            rule = s.memory_rules[slip_hash(swapped[-1], 2)]
            parts = decompose(rule.decomposition, swapped, s.tags)
            if parts is not None:
                step.remembered = list(self._reassemble(rule.reassemblies[0], parts))
                self.memories.append(step.remembered)
        return self._apply(step, key, s.entries[key].rules, swapped)

    def _apply(self, step, key, rules, ws):
        s, i = self.script, 0
        while i < len(rules):
            rule = rules[i]
            if isinstance(rule, Link):
                key, rules, i = rule.target, s.entries[rule.target].rules, 0
                step.via.append(key)
                continue
            parts = decompose(rule.decomposition, ws, s.tags)
            if parts is not None:
                turn = self.turns.get((key, i), 0) % len(rule.reassemblies)
                self.turns[(key, i)] = turn + 1
                step.rule, step.parts, step.template = rule, parts, rule.reassemblies[turn]
                step.reply = list(self._reassemble(rule.reassemblies[turn], parts))
                if key == "NONE":
                    step.how = "none"
                return step
            i += 1
        step.how, step.reply = "nomatch", s.nomatch[self.counter - 1].split()
        return step


def conversation(script, lines, **steps):
    """Every Step of one conversation, from a fresh ELIZA."""
    eliza = Eliza(script, **steps)
    return [eliza.step(line) for line in lines if line.strip()]
