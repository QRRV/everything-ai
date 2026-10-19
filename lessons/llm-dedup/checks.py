# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-dedup: shingles, the k-word shingles of a text; minhash, a MinHash signature in Python;
signature, the same with NumPy for many hash functions at once; and a choice. The demo takes the raw web pages of
llm-quality-filter (the first 20 MB of a shard of C4 before its cleaning, downloaded once), cleans them with the
course's versions of that lesson's rules, and finds the near-duplicates among them with the learner's shingles and
signature, and banding (locality-sensitive hashing) by the course."""
import html
import importlib.util
import zlib
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake, run_code, uses_loop


def _quality():
    """The checks of llm-quality-filter: its download of web pages, and the course's versions of its rules."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_llm_quality_filter_for_dedup",
                                                  paths.LESSONS / "llm-quality-filter" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _quality()
C4_SAMPLE, DOWNLOAD_LINE = Q.C4_SAMPLE, Q.DOWNLOAD_LINE
downloaded, sample_or_stop, number = Q.downloaded, Q.sample_or_stop, Q.number

K = 5                 # words per shingle
P = 2 ** 31 - 1       # a prime: every hash is a number from 0 to P - 1
N_HASHES = 128        # hash functions, so numbers per signature
BANDS, ROWS = 16, 8   # banding: 16 bands of 8 numbers
SIMILAR = 0.8         # two pages that share at least this much are near-duplicates
_rng = np.random.default_rng(0)
A = _rng.integers(1, P, N_HASHES).tolist()   # the hash function i turns h into (A[i] * h + B[i]) % P
B = _rng.integers(0, P, N_HASHES).tolist()


def base_hash(shingle: str) -> int:
    """The number of a shingle: CRC-32 of its UTF-8 bytes, a fixed function of Python's zlib, below P."""
    return zlib.crc32(shingle.encode("utf-8")) % P


# --- measured: the course's run of the demo with its own solutions (see the notes) ---
# The course's run of the demo with its own versions, on the download: the kept pages and all their pairs, the
# candidate pairs of banding, those that share at least SIMILAR, the groups and the pages that can go, the largest
# group, the exact copies (groups and extra copies) and the mean error of the estimate on the candidates; and what the
# largest groups are, by the place of a page of each.
MEASURED = {"pages": 5614, "all_pairs": 15755691, "candidates": 594, "similar": 538, "groups": 26, "removed": 74,
            "largest": 28, "exact_groups": 12, "exact_extra": 37, "error": 0.012}
GROUPS = {203: {"en": "a notice that asks you to solve a CAPTCHA", "nl": "een melding die je vraagt een CAPTCHA op te lossen"},
          1188: {"en": "the notice at the top of an XML sitemap", "nl": "de melding boven een XML-sitemap"},
          678: {"en": "a forum notice that you must log in", "nl": "een forummelding dat je moet inloggen"},
          127: {"en": "a page of a parked domain", "nl": "een pagina van een geparkeerd domein"}}
# --- end of measured ---


@cache
def kept_pages() -> list:
    """The pages of the download that the course's versions of llm-quality-filter keep, cleaned, with their place in
    the download: (place, text)."""
    clean = run_code(Q.EXERCISES["clean-lines"].solution).names["clean_lines"]
    keep = run_code(Q.EXERCISES["keep-page"].solution).names["keep_page"]
    out = []
    for i, page in enumerate(Q.pages()):
        text = clean(page)
        if keep(text):
            out.append((i, text))
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def candidates(signatures: np.ndarray) -> set:
    """Banding: pages whose signatures agree in all ROWS numbers of one of the BANDS bands fall in one bucket; only
    the pairs within a bucket are compared."""
    pairs = set()
    for band in range(BANDS):
        buckets = {}
        for j, row in enumerate(signatures[:, band * ROWS:(band + 1) * ROWS]):
            buckets.setdefault(tuple(row.tolist()), []).append(j)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return pairs


def find_duplicates(shingles, signature) -> dict:
    """The demo: the learner's shingles (k = K) and signature on every kept page; then banding, the exact overlap
    of every candidate pair, and groups of pages that share at least SIMILAR with another, of which one stays."""
    pages = kept_pages()
    sets = [set(shingles(text, K)) for _, text in pages]
    sigs = np.array([np.asarray(signature(sorted(base_hash(s) for s in shs), A, B)).tolist() if shs else [P] * N_HASHES
                     for shs in sets], dtype=np.int64)
    exact = {}
    for _, text in pages:
        exact[text] = exact.get(text, 0) + 1
    pairs = []
    for x, y in candidates(sigs):
        pairs.append((x, y, jaccard(sets[x], sets[y]), float((sigs[x] == sigs[y]).mean())))
    parent = list(range(len(pages)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y, j, _ in pairs:
        if j >= SIMILAR:
            parent[find(x)] = find(y)
    groups = {}
    for x in range(len(pages)):
        groups.setdefault(find(x), []).append(x)
    big = [g for g in groups.values() if len(g) > 1]
    similar = [p for p in pairs if p[2] >= SIMILAR]
    return {"pages": len(pages), "all_pairs": len(pages) * (len(pages) - 1) // 2, "candidates": len(pairs),
            "similar": len(similar), "groups": len(big), "removed": sum(len(g) - 1 for g in big),
            "largest": max((len(g) for g in big), default=0),
            "exact_groups": sum(1 for n in exact.values() if n > 1), "exact_extra": sum(n - 1 for n in exact.values() if n > 1),
            "error": float(np.mean([abs(e - j) for _, _, j, e in pairs])) if pairs else 0.0,
            "places": [pages[g[0]][0] for g in sorted(big, key=len, reverse=True)]}


def signatures_picture(agree: list, labels: dict) -> str:
    """The places of two signatures as one row of cells, one per hash function: a cell is filled where the two agree
    (ink: these are places, not roles of a model). Under it, the notes, one per line."""
    n = len(agree)
    W, TOP = 330, 46
    CELL = W / n
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
         f'<text x="0" y="36" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(labels["row"])}</text>']
    for i, same in enumerate(agree):
        style = "fill:var(--ea-ink);stroke:var(--ea-ink)" if same else "fill:var(--ea-inset);stroke:var(--ea-line)"
        p.append(f'<rect x="{i * CELL + 0.5:.1f}" y="{TOP}" width="{CELL - 2:.1f}" height="22" rx="2" style="{style}"/>')
    y = TOP + 22
    for line in labels["notes"]:
        y += 19
        p.append(f'<text x="0" y="{y}" style="font-size:12.5px;fill:var(--ea-ink)">{html.escape(line)}</text>')
    H = y + 8
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


EXAMPLE_A = "the cat sat on the mat and looked at the birds all day"
EXAMPLE_B = "the cat sat on the hat and looked at the birds all day"


def example_agree(n: int = 24) -> tuple:
    """The worked example's two texts, k = 2: which of the first n hash functions agree, and the exact overlap."""
    sa = {" ".join(EXAMPLE_A.split()[i:i + 2]) for i in range(len(EXAMPLE_A.split()) - 1)}
    sb = {" ".join(EXAMPLE_B.split()[i:i + 2]) for i in range(len(EXAMPLE_B.split()) - 1)}
    def sig(s):
        h = np.array(sorted(base_hash(x) for x in s), dtype=np.int64)
        return ((np.outer(h, np.array(A[:n])) + np.array(B[:n])) % P).min(axis=0)
    return (sig(sa) == sig(sb)).tolist(), jaccard(sa, sb)


# The lesson is done when these are correct. "memorize" is the choice question in the lesson.
REQUIRED = ["shingles", "minhash", "signature", "memorize"]

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
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


def _shown(v) -> str:
    """A set in a fixed order, so that two sets with the same items look the same."""
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(repr(x) for x in sorted(v, key=repr)) + "}" if v else "set()"
    if isinstance(v, np.ndarray):
        return repr(v.tolist())
    if isinstance(v, np.generic):
        return repr(v.item())
    return repr(v)


# --- exercise 1: shingles --------------------------------------------------------------------------------------------

_S_HEAD = "def shingles(text, k):\n    words = text.lower().split()\n    out = set()\n"
_S_LOOP = "    for i in range(len(words) - k + 1):\n        out.add(\" \".join(words[i:i + k]))\n"
_S_START = _S_HEAD + "    # every run of k words in a row, as one text with spaces between the words\n    ...\n    return out\n"


def _s(loop: str = _S_LOOP, head: str = _S_HEAD) -> str:
    return head + loop + "    return out\n"


# --- exercise 2: minhash in Python -----------------------------------------------------------------------------------

_M_HEAD = "P = 2 ** 31 - 1\n\n\ndef minhash(hashes, a, b):\n    sig = []\n    for i in range(len(a)):\n"
_M_BODY = ("        smallest = P\n        for h in hashes:\n            v = (a[i] * h + b[i]) % P\n            if v < smallest:\n"
           "                smallest = v\n        sig.append(smallest)\n")
_M_START = (_M_HEAD + "        # hash function i on every hash of the page: keep the smallest number\n        ...\n"
            "        sig.append(smallest)\n    return sig\n")


def _m(body: str = _M_BODY) -> str:
    return _M_HEAD + body + "    return sig\n"


# --- exercise 3: signature with NumPy --------------------------------------------------------------------------------

_N_HEAD = "import numpy as np\n\nP = 2 ** 31 - 1\n\n\ndef signature(hashes, a, b):\n    h = np.array(hashes)\n    a = np.array(a)\n    b = np.array(b)\n"
_N_BODY = "    values = (np.outer(h, a) + b) % P\n    return values.min(axis=0)\n"
_N_START = (_N_HEAD + "    # every hash function on every hash at once: a row per hash, a column per function\n    values = ...\n"
            "    # the smallest number of every column\n    return ...\n")


def _n(body: str = _N_BODY) -> str:
    return _N_HEAD + body


def _same(got, want) -> bool:
    if isinstance(want, np.ndarray) or isinstance(got, np.ndarray):
        try:
            return np.array_equal(np.asarray(got), np.asarray(want))
        except (TypeError, ValueError):
            return False
    return got == want


EXERCISES = {
    "shingles": FunctionBox(
        name="shingles",
        function="shingles",
        start=_S_START,
        solution=_s(),
        visible=[("A dog ran after the Red ball", 3)],
        hidden=[("one two three", 3), ("one two", 3), ("", 2), ("a b a b a b", 2), ("Same same SAME", 1),
                ("We walked  along\nthe beach", 2), ("the cat sat on the mat and the cat sat on the hat", 5)],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("list", wrong=_s(head=_S_HEAD.replace("out = set()", "out = []"), loop=_S_LOOP.replace("out.add(", "out.append(")),
                    en=_CALL + ": your shingles are a list, so a shingle that occurs twice is in it twice. Collect them in a set: `out.add(...)`.",
                    nl=_CALL_NL + ": je shingles zijn een lijst, dus een shingle die twee keer voorkomt, staat er twee keer in. Verzamel ze in een set: `out.add(...)`."),
            Mistake("last-missing", wrong=_s(loop=_S_LOOP.replace("range(len(words) - k + 1)", "range(len(words) - k)")),
                    en=_CALL + ": the last run of k words is missing. The last one starts at word `len(words) - k`, so the range goes to `len(words) - k + 1`.",
                    nl=_CALL_NL + ": de laatste reeks van k woorden ontbreekt. De laatste begint bij woord `len(words) - k`, dus de range loopt tot `len(words) - k + 1`."),
            Mistake("no-lower", wrong=_s(head=_S_HEAD.replace("text.lower().split()", "text.split()")),
                    en=_CALL + ": the capitals stay in your shingles, so a word with a capital and the same word without one count as different words. Make the text lower case first: `text.lower().split()`.",
                    nl=_CALL_NL + ": de hoofdletters blijven in je shingles, dus een woord met een hoofdletter en hetzelfde woord zonder tellen als verschillende woorden. Zet de tekst eerst in kleine letters: `text.lower().split()`."),
            Mistake("split-space", wrong=_s(head=_S_HEAD.replace("text.lower().split()", "text.lower().split(\" \")")),
                    en=_CALL + ": `split(\" \")` cuts only at one space, so two spaces in a row give an empty word, and a new line stays inside a word. `split()` without a value cuts at every run of spaces, tabs and new lines.",
                    nl=_CALL_NL + ": `split(\" \")` knipt alleen bij één spatie, dus twee spaties achter elkaar geven een leeg woord, en een nieuwe regel blijft in een woord staan. `split()` zonder waarde knipt bij elke reeks spaties, tabs en nieuwe regels."),
            Mistake("chars", wrong=_s(head=_S_HEAD.replace("words = text.lower().split()", "words = text.lower()")),
                    en=_CALL + ": your shingles are runs of k characters, not of k words. Cut the text into words first: `text.lower().split()`.",
                    nl=_CALL_NL + ": je shingles zijn reeksen van k tekens, niet van k woorden. Knip de tekst eerst in woorden: `text.lower().split()`."),
            Mistake("tuples", wrong=_s(loop=_S_LOOP.replace("out.add(\" \".join(words[i:i + k]))", "out.add(tuple(words[i:i + k]))")),
                    en=_CALL + ": your shingles are tuples of words. Join each into one text with spaces: `\" \".join(words[i:i + k])`, so that it can be hashed as text.",
                    nl=_CALL_NL + ": je shingles zijn tuples van woorden. Zet elk om in één tekst met spaties: `\" \".join(words[i:i + k])`, zodat hij als tekst gehasht kan worden."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: a loop over the places where a run of k words starts, with `out.add` of that run as one text.",
                      "nl": "Vervang de `...`: een loop over de plekken waar een reeks van k woorden begint, met `out.add` van die reeks als één tekst."},
            "wrong": {"en": _CALL + ". Every run of k words in a row, from the text in lower case, joined with spaces, each once?",
                      "nl": _CALL_NL + ". Elke reeks van k woorden achter elkaar, uit de tekst in kleine letters, met spaties ertussen, elk één keer?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `k` and `text`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `k` en `text`, en geen vaste getallen?"},
            "correct": {"en": "Your function cuts a text into shingles. Two pages are compared by the shingles they share: a different date or menu changes only a few of them.",
                        "nl": "Je functie knipt een tekst in shingles. Twee pagina's vergelijk je via de shingles die ze delen: een andere datum of een ander menu verandert er maar een paar."},
        },
    ),
    "minhash": FunctionBox(
        name="minhash",
        function="minhash",
        start=_M_START,
        solution=_m(),
        visible=[([5, 9, 2], [3, 7], [1, 4])],
        hidden=[([10], [1, 2, 3], [0, 0, 0]), ([2147483646, 1, 7], [1103515245, 48271], [12345, 0]),
                (list(range(50)), [97, 389, 1009], [11, 13, 17]), ([4, 4, 4], [5], [6]), ([], [3], [1])],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("max", wrong=_m(body=_M_BODY.replace("smallest = P", "smallest = -1").replace("if v < smallest", "if v > smallest")),
                    en=_CALL + ": your signature keeps the largest number of each hash function. MinHash keeps the smallest: start at `P` and keep `v` when it is smaller.",
                    nl=_CALL_NL + ": je signature houdt het grootste getal van elke hashfunctie. MinHash houdt het kleinste: begin bij `P` en houd `v` als hij kleiner is."),
            Mistake("start-zero", wrong=_m(body=_M_BODY.replace("smallest = P", "smallest = 0")),
                    en=_CALL + ": every number of your signature is 0: you start at 0, and no number is smaller. Start at `P`, larger than every number a hash function gives.",
                    nl=_CALL_NL + ": elk getal van je signature is 0: je begint bij 0, en geen getal is kleiner. Begin bij `P`, groter dan elk getal dat een hashfunctie geeft."),
            Mistake("no-mod", wrong=_m(body=_M_BODY.replace("v = (a[i] * h + b[i]) % P", "v = a[i] * h + b[i]")),
                    en=_CALL + ": without `% P`, a hash function gives numbers in the order of the hashes, so every function keeps the same shingle: the smallest hash. `% P` mixes the order: `(a[i] * h + b[i]) % P`.",
                    nl=_CALL_NL + ": zonder `% P` geeft een hashfunctie getallen in de volgorde van de hashes, dus elke functie houdt dezelfde shingle: de kleinste hash. `% P` husselt de volgorde: `(a[i] * h + b[i]) % P`."),
            Mistake("one-function", wrong=_m(body=_M_BODY.replace("a[i] * h + b[i]", "a[0] * h + b[0]")),
                    en=_CALL + ": every number of your signature uses the first hash function, `a[0]` and `b[0]`. Number i of the signature uses function i: `a[i]` and `b[i]`.",
                    nl=_CALL_NL + ": elk getal van je signature gebruikt de eerste hashfunctie, `a[0]` en `b[0]`. Getal i van de signature gebruikt functie i: `a[i]` en `b[i]`."),
            Mistake("min-hash-first", wrong=_m(body="        smallest = P\n        for h in hashes:\n            if h < smallest:\n                smallest = h\n"
                                                    "        sig.append((a[i] * smallest + b[i]) % P)\n"),
                    en=_CALL + ": you take the smallest hash first, and then hash function i of that one. MinHash runs function i on every hash, and keeps the smallest outcome: another shingle can win for every function.",
                    nl=_CALL_NL + ": je neemt eerst de kleinste hash, en dan hashfunctie i daarvan. MinHash laat functie i op elke hash los, en houdt de kleinste uitkomst: voor elke functie kan een andere shingle winnen."),
            Mistake("min-empty", error=lambda e: isinstance(e, ValueError) and "empty" in str(e),
                    en="With `{call}`, `min` gets no numbers at all, as when `hashes` is empty, and stops with an error. A page without shingles gets `P` in every place: start at `smallest = P`, and keep the smaller in a loop over `hashes`.",
                    nl="Met `{call}` krijgt `min` helemaal geen getallen, zoals wanneer `hashes` leeg is, en stopt hij met een fout. Een pagina zonder shingles krijgt `P` op elke plek: begin bij `smallest = P`, en houd het kleinere in een loop over `hashes`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: for hash function i, start at `P`, run it on every hash, and keep the smallest outcome.",
                      "nl": "Vervang de `...`: begin voor hashfunctie i bij `P`, laat hem op elke hash los, en houd de kleinste uitkomst."},
            "wrong": {"en": _CALL + ". For every function i: the smallest of `(a[i] * h + b[i]) % P` over all hashes `h`, starting at `P`?",
                      "nl": _CALL_NL + ". Voor elke functie i: de kleinste van `(a[i] * h + b[i]) % P` over alle hashes `h`, beginnend bij `P`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use every function and every hash, also for large hashes and for no hashes at all?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie elke functie en elke hash, ook bij grote hashes en als er geen hashes zijn?"},
            "correct": {"en": "Your function gives a page a MinHash signature: one number per hash function, the smallest it gives any of the page's shingles. For two pages, the chance that a number agrees is about their overlap.",
                        "nl": "Je functie geeft een pagina een MinHash-signature: één getal per hashfunctie, het kleinste dat die aan een van de shingles van de pagina geeft. Voor twee pagina's is de kans dat een getal gelijk is ongeveer hun overlap."},
        },
    ),
    "signature": FunctionBox(
        name="signature",
        function="signature",
        start=_N_START,
        solution=_n(),
        visible=[([5, 9, 2], [3, 7], [1, 4])],
        hidden=[([10], [1, 2, 3], [0, 0, 0]), ([2147483646, 1, 7], [1103515245, 48271], [12345, 0]),
                (list(range(50)), [97, 389, 1009], [11, 13, 17]), ([4, 4, 4], [5], [6])],
        compare=_same,
        show=_shown,
        first=[_DOTS],
        form=[
            Mistake("with-loop", state="almost", matches=lambda got, text: uses_loop(text),
                    en="The right signature, but with a loop of your own in Python. The demo calls `signature` 5,614 times with 128 functions, and with loops that takes several times longer. `np.outer(h, a)` and `.min(axis=0)` do every function on every hash at once, without a loop.",
                    nl="De goede signature, maar met een eigen loop in Python. De demo roept `signature` 5.614 keer aan met 128 functies, en met loops duurt dat een aantal keer zo lang. `np.outer(h, a)` en `.min(axis=0)` doen elke functie op elke hash tegelijk, zonder loop."),
        ],
        mistakes=[
            Mistake("axis", wrong=_n(body=_N_BODY.replace("values.min(axis=0)", "values.min(axis=1)")),
                    en=_CALL + ": you take the smallest number of every row, one per hash. The signature needs one per hash function, the columns: `values.min(axis=0)`.",
                    nl=_CALL_NL + ": je neemt het kleinste getal van elke rij, één per hash. De signature heeft er één per hashfunctie nodig, de kolommen: `values.min(axis=0)`."),
            Mistake("all-min", wrong=_n(body=_N_BODY.replace("values.min(axis=0)", "values.min()")),
                    en=_CALL + ": `values.min()` gives the smallest number of the whole table. Take the smallest of every column: `values.min(axis=0)`.",
                    nl=_CALL_NL + ": `values.min()` geeft het kleinste getal van de hele tabel. Neem het kleinste van elke kolom: `values.min(axis=0)`."),
            Mistake("times", error=lambda e: isinstance(e, ValueError) and "could not be broadcast" in str(e),
                    en="NumPy cannot combine two of your arrays: their shapes do not fit. `h * a` multiplies number by number, so it needs as many hashes as hash functions. `np.outer(h, a)` makes the table of every hash times every `a`, a row per hash and a column per function, and `+ b` then adds `b[c]` to every column `c`.",
                    nl="NumPy kan twee van je arrays niet combineren: hun shapes passen niet. `h * a` vermenigvuldigt getal voor getal, dus het heeft evenveel hashes als hashfuncties nodig. `np.outer(h, a)` maakt de tabel van elke hash keer elke `a`, een rij per hash en een kolom per functie, en `+ b` telt dan `b[c]` bij elke kolom `c` op."),
            Mistake("no-mod", wrong=_n(body=_N_BODY.replace(" % P", "")),
                    en=_CALL + ": every function keeps the same shingle, the one with the smallest hash. That happens without `% P` on every value, or with `% P` only after the smallest is taken: `% P` mixes the order, so first `(np.outer(h, a) + b) % P`, then `.min(axis=0)`.",
                    nl=_CALL_NL + ": elke functie houdt dezelfde shingle, die met de kleinste hash. Dat gebeurt zonder `% P` op elke waarde, of met `% P` pas nadat het kleinste genomen is: `% P` husselt de volgorde, dus eerst `(np.outer(h, a) + b) % P`, dan `.min(axis=0)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the table of every function on every hash, then the smallest number of every column.",
                      "nl": "Vervang de twee `...`: de tabel van elke functie op elke hash, en dan het kleinste getal van elke kolom."},
            "wrong": {"en": _CALL + ". `(np.outer(h, a) + b) % P`, a row per hash and a column per function, and then the smallest of every column?",
                      "nl": _CALL_NL + ". `(np.outer(h, a) + b) % P`, een rij per hash en een kolom per functie, en dan het kleinste van elke kolom?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `hashes`, `a` and `b`, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `hashes`, `a` en `b`, en geen vaste getallen?"},
            "correct": {"en": "Your function gives the same signature as your `minhash`, with all hash functions on all hashes at once. That is fast enough for thousands of pages.",
                        "nl": "Je functie geeft dezelfde signature als je `minhash`, met alle hashfuncties op alle hashes tegelijk. Dat is snel genoeg voor duizenden pagina's."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("shingles", _S_START, "empty", None),
    ("shingles", _s(), "correct", None),
    ("shingles", _S_HEAD + "    return {\" \".join(words[i:i + k]) for i in range(len(words) - k + 1)}\n", "correct", None),
    ("shingles", _s(head=_S_HEAD.replace("out = set()", "out = []"), loop=_S_LOOP.replace("out.add(", "out.append(")), "wrong", "list"),
    ("shingles", _s(loop=_S_LOOP.replace("range(len(words) - k + 1)", "range(len(words) - k)")), "wrong", "last-missing"),
    ("shingles", _s(head=_S_HEAD.replace("text.lower().split()", "text.split()")), "wrong", "no-lower"),
    ("shingles", _s(head=_S_HEAD.replace("words = text.lower().split()", "words = text.lower()")), "wrong", "chars"),
    ("shingles", _s(head=_S_HEAD.replace("text.lower().split()", "text.lower().split(\" \")")), "wrong", "split-space"),
    ("shingles", _s(loop=_S_LOOP.replace("out.add(\" \".join(words[i:i + k]))", "out.add(tuple(words[i:i + k]))")), "wrong", "tuples"),
    ("minhash", _M_START, "empty", None),
    ("minhash", _m(), "correct", None),
    ("minhash", _M_HEAD + "        sig.append(min([(a[i] * h + b[i]) % P for h in hashes] + [P]))\n    return sig\n", "correct", None),
    ("minhash", _m(body=_M_BODY.replace("smallest = P", "smallest = -1").replace("if v < smallest", "if v > smallest")), "wrong", "max"),
    ("minhash", _m(body=_M_BODY.replace("smallest = P", "smallest = 0")), "wrong", "start-zero"),
    ("minhash", _m(body=_M_BODY.replace("v = (a[i] * h + b[i]) % P", "v = a[i] * h + b[i]")), "wrong", "no-mod"),
    ("minhash", _m(body=_M_BODY.replace("a[i] * h + b[i]", "a[0] * h + b[0]")), "wrong", "one-function"),
    ("minhash", _m(body="        smallest = P\n        for h in hashes:\n            if h < smallest:\n                smallest = h\n"
                        "        sig.append((a[i] * smallest + b[i]) % P)\n"), "wrong", "min-hash-first"),
    ("minhash", _M_HEAD + "        sig.append(min((a[i] * h + b[i]) % P for h in hashes))\n    return sig\n", "error", "min-empty"),
    ("signature", _N_START, "empty", None),
    ("signature", _n(), "correct", None),
    ("signature", _N_HEAD + "    return ((h[:, None] * a[None, :] + b) % P).min(axis=0).tolist()\n", "correct", None),
    ("signature", _n(body=_N_BODY.replace("values.min(axis=0)", "values.min(axis=1)")), "wrong", "axis"),
    ("signature", _n(body=_N_BODY.replace("values.min(axis=0)", "values.min()")), "wrong", "all-min"),
    ("signature", _n(body=_N_BODY.replace("np.outer(h, a)", "h * a")), "error", "times"),
    ("signature", _n(body=_N_BODY.replace(" % P", "")), "wrong", "no-mod"),
    ("signature", _n(body=_N_BODY.replace("values.min(axis=0)", "values.min(axis=0) % P").replace(" % P\n    return", "\n    return")), "wrong", "no-mod"),
    ("signature", _n(body=_N_BODY.replace("np.outer(h, a)", "np.outer(a, h)")), "error", "times"),
    ("signature", "import numpy as np\n\nP = 2 ** 31 - 1\n\n\ndef signature(hashes, a, b):\n    sig = []\n    for i in range(len(a)):\n"
                  "        sig.append(((a[i] * np.array(hashes) + b[i]) % P).min())\n    return sig\n", "almost", "with-loop"),
    ("signature", "P = 2 ** 31 - 1\n\n\ndef signature(hashes, a, b):\n    sig = []\n    for i in range(len(a)):\n" + _M_BODY + "    return sig\n", "almost", "with-loop"),
]
