# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-whats-next, and the numbers about the branches that the lesson shows.

Every later branch has its attach points in the route data (course/route/takken.yaml): the lessons of the main route it
connects to, in route order; the branch can begin after the first. The numbers, the figure and the cards come from there
and from the learner's progress, so the lesson stays right when the route changes. The exercises: sort next questions of
a reproduction by their kind, and can_begin, the branches whose first attach point you have done."""
import html
from collections import Counter

from everything_ai import course, draw, progress
from everything_ai.check import FunctionBox, Mistake, Sort

LEVELS = ["basis", "gevorderd", "expert"]
CORE = [s["id"] for s in course.main_route() if not s["extra"]]   # the main route without its extra lessons
LATER = sorted((b for b in course.branches() if b["status"] == "later"), key=lambda b: b["build_order"])
ATTACH = {b["id"]: list(b["attach"]) for b in LATER}
MILESTONES = course.milestones()   # the wrap-up lists all three


def level(lesson_id: str) -> str:
    return course.lessons()[lesson_id]["level"]


def branch(branch_id: str) -> dict:
    return next(b for b in LATER if b["id"] == branch_id)


def short_title(branch_id: str, lang: str) -> str:
    """The name of a branch without what follows a colon: "Building with AI", not all it holds."""
    b = branch(branch_id)
    return (b["title_nl"] if lang == "nl" else b["title"]).split(":")[0]


def first_levels() -> dict[str, int]:
    """How many later branches can begin after each level: the level of their first attach point."""
    c = Counter(level(points[0]) for points in ATTACH.values())
    return {lv: c.get(lv, 0) for lv in LEVELS}


def later_names(lang: str) -> list[str]:
    """The short names of the branches that cannot begin after the basics, in build order."""
    return [short_title(k, lang) for k, points in ATTACH.items() if level(points[0]) != "basis"]


def milestone_lines(lang: str) -> str:
    """The milestones of the main route as a Markdown list, in the learner's language."""
    return "\n".join(f"- {m[lang]}" for m in MILESTONES)


def waiting(ready: list) -> list[str]:
    """The later branches not in `ready`, in the order of their first attach point on the main route."""
    return sorted((k for k in ATTACH if k not in ready), key=lambda k: CORE.index(ATTACH[k][0]))


def done_lessons() -> set[str]:
    """The lessons the learner has done, from their progress."""
    return {lid for lid, p in progress.all_lessons().items() if p.get("done")}


def extra_lessons(branch_id: str) -> list[str]:
    """The built extra lessons of the main route whose home is this branch: a start you can make now."""
    lessons = course.lessons()
    return [s["id"] for s in course.main_route()
            if s["extra"] and lessons[s["id"]]["tak"] == branch_id and course.is_built(s["id"])]


def branch_card(branch_id: str, lang: str, done: set, words: dict) -> str:
    """What the lesson says about one branch, in Markdown: its project, its attach points with a link and whether you
    did them, and the extra lessons of the main route that already belong to it."""
    b = branch(branch_id)
    nl = lang == "nl"
    out = [f"**{b['title_nl'] if nl else b['title']}**  ", f"{words['project']}: {b['project']['title_nl' if nl else 'title']}.", "",
           f"{words['attach']}:", ""]
    for i, a in enumerate(b["attach"], 1):
        state = words["done"] if a in done else words["not_done"]
        out.append(f"{i}. [{course.title(a, lang)}](/course/open/{a}?lang={lang}), "
                   f"{course.level_name(level(a), lang).lower()}: {state}")
    extras = extra_lessons(branch_id)
    if extras:
        out += ["", words["extras"] + " " + ", ".join(f"[{course.title(e, lang)}](/course/open/{e}?lang={lang})" for e in extras) + "."]
    return "\n".join(out)


def tree_picture(branch_id: str, done: set, labels: dict):
    """The main route as a trunk, with its three levels, and the chosen branch growing out of it at its first attach
    point; its other attach points, numbered, join it with a dotted line. Every other later branch is a short grey
    stub where it begins. A dot marks the furthest lesson of the trunk you have done. Ink only: nothing here is learned
    or counted wrong."""
    W, H = 540, 216
    x0, x1, ty, by = 14, 500, 164, 86          # trunk from x0 to x1 at height ty; the chosen branch runs at height by
    n = len(CORE)

    def x(lesson_id: str) -> float:
        return x0 + (x1 - x0) * CORE.index(lesson_id) / (n - 1)

    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    seen = Counter()
    for other, points in ATTACH.items():   # the stubs first, under everything else; two at one lesson side by side
        if other != branch_id:
            xs = x(points[0]) + 6 * seen[points[0]]
            seen[points[0]] += 1
            parts.append(f'<path d="M{xs:.1f} {ty - 3} C{xs:.1f} {ty - 22} {xs + 8:.1f} {ty - 34} {xs + 28:.1f} {ty - 34}" '
                         f'style="fill:none;stroke:var(--ea-axis);stroke-width:2.2;stroke-linecap:round">'
                         f'<title>{html.escape(short_title(other, labels["lang"]))}</title></path>')
    mine = [CORE.index(s) for s in CORE if s in done]
    if mine:   # just under the trunk, above the level names; never cut off at the left edge
        xm = x0 + (x1 - x0) * max(mine) / (n - 1)
        tx, anchor = (xm + 3, "end") if xm > 90 else (max(xm - 3, 4), "start")
        parts.append(f'<path d="M{xm:.1f} {ty + 6} V{ty + 13}" style="stroke:var(--ea-ink-3);stroke-width:1.4"/>'
                     f'<text x="{tx:.1f}" y="{ty + 26}" text-anchor="{anchor}" style="font-size:13px;fill:var(--ea-ink-3)">'
                     f'{html.escape(labels["you"])}</text>')
    starts = [CORE.index(next(s for s in CORE if level(s) == lv)) for lv in LEVELS] + [n]
    for i, lv in enumerate(LEVELS):
        a, b = x0 + (x1 - x0) * starts[i] / (n - 1), x0 + (x1 - x0) * (starts[i + 1] - 1) / (n - 1)
        parts.append(f'<line x1="{a:.1f}" y1="{ty}" x2="{b:.1f}" y2="{ty}" style="stroke:var(--ea-ink-2);stroke-width:6;stroke-linecap:round"/>'
                     f'<text x="{(a + b) / 2:.1f}" y="{ty + 47}" text-anchor="middle" style="font-size:14px;fill:var(--ea-ink-2)">'
                     f'{html.escape(labels["levels"][i])}</text>')
    points = ATTACH[branch_id]
    xf = x(points[0])
    parts.append(f'<path d="M{xf:.1f} {ty} C{xf:.1f} {ty - 50} {xf + 10:.1f} {by} {xf + 50:.1f} {by} H{x1 + 30}" '
                 f'style="fill:none;stroke:var(--ea-ink);stroke-width:3;stroke-linecap:round"/>')
    name = short_title(branch_id, labels["lang"])
    wide = 8.2 * len(name)
    tx, anchor = (xf + 50, "start") if xf + 50 + wide < W - 4 else (W - 4, "end")
    parts.append(f'<text x="{tx:.1f}" y="{by - 10}" text-anchor="{anchor}" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">'
                 f'{html.escape(name)}</text>')
    for xa in (x(p) for p in points[1:]):
        parts.append(f'<line x1="{xa:.1f}" y1="{ty - 9}" x2="{xa:.1f}" y2="{by + 2}" style="stroke:var(--ea-ink);stroke-width:1.6;stroke-dasharray:3 4"/>')
    for i, xa in enumerate((x(p) for p in points), 1):
        parts.append(f'<circle cx="{xa:.1f}" cy="{ty}" r="10" style="fill:var(--ea-ink);stroke:var(--ea-page);stroke-width:2"/>'
                     f'<text x="{xa:.1f}" y="{ty + 4.5}" text-anchor="middle" style="font-size:13px;font-weight:600;fill:var(--ea-page)">{i}</text>')
    return draw._svg(W, H, labels["alt"], "".join(parts))


# --- exercise 2: can_begin ---------------------------------------------------------------------------------------------

_C_HEAD = ("def can_begin(attach, done):\n"
           "    # attach: for each branch, its attach points, in route order\n"
           "    # done: the lessons you have done\n"
           "    # gives a list of the branches whose first attach point is in done, in the order of attach\n")
_C_BODY = ("    ready = []\n"
           "    for branch, points in attach.items():\n"
           "        if points[0] in done:\n"
           "            ready.append(branch)\n"
           "    return ready\n")
_C_START = _C_HEAD + "    ...\n"
C_CODE = _C_HEAD + _C_BODY
_C_WRONG = {
    "any": C_CODE.replace("        if points[0] in done:\n            ready.append(branch)\n",
                          "        for point in points:\n            if point in done:\n                ready.append(branch)\n                break\n"),
    "all": C_CODE.replace("        if points[0] in done:\n",
                          "        if len([p for p in points if p in done]) == len(points):\n"),
    "last": C_CODE.replace("points[0]", "points[-1]"),
    "points": C_CODE.replace("ready.append(branch)", "ready.append(points[0])"),
}
_ATTACH = {"vision": ["cnn-convolution-idea", "cnn-pytorch"], "depth": ["nn-deep-net-project", "llm-modern-gpt"],
           "robots": ["math-matrix-multiply", "rl-what-is-rl"]}
_DONE = ["cnn-convolution-idea", "llm-modern-gpt", "math-matrix-multiply", "rl-what-is-rl"]
_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 1: what kind of next question ----------------------------------------------------------------------------

_SOLUTION = {"shift": "robust", "input-only": "ablation", "wide-mnist": "two", "kmnist": "data",
             "noise": "robust", "noinput-noise": "two", "mnist": "data", "hidden-only": "ablation"}

EXERCISES = {
    "kinds": Sort(
        name="kinds",
        groups={
            "ablation": {"en": "Ablation", "nl": "Ablation"},
            "data": {"en": "New data", "nl": "Nieuwe data"},
            "robust": {"en": "Robustness test", "nl": "Robuustheidstest"},
            "two": {"en": "Two changes at once", "nl": "Twee veranderingen tegelijk"},
        },
        items={
            "shift": {"en": "**Shifted pictures.** The same trained networks, tested on the test pictures moved two pixels to the right.",
                      "nl": "**Verschoven afbeeldingen.** Dezelfde getrainde netwerken, getest op de testafbeeldingen twee pixels naar rechts geschoven."},
            "input-only": {"en": "**Input only.** Dropout at the input, none in the hidden layers; the rest as in the reproduction.",
                           "nl": "**Alleen de input.** Dropout bij de input, geen in de hidden layers; de rest zoals in de reproductie."},
            "wide-mnist": {"en": "**Wider, on MNIST.** Hidden layers of 1024 instead of 512, trained and tested on the handwritten digits.",
                           "nl": "**Breder, op MNIST.** Hidden layers van 1024 in plaats van 512, getraind en getest op de handgeschreven cijfers."},
            "kmnist": {"en": "**Kuzushiji-MNIST.** The same networks and steps, on 28 by 28 pictures of ten Japanese characters in an old cursive script.",
                       "nl": "**Kuzushiji-MNIST.** Dezelfde netwerken en stappen, op afbeeldingen van 28 bij 28 pixels van tien Japanse tekens in een oud, lopend schrift."},
            "noise": {"en": "**Noisy pictures.** The same trained networks, tested on the test pictures with random noise on every pixel.",
                      "nl": "**Afbeeldingen met ruis.** Dezelfde getrainde netwerken, getest op de testafbeeldingen met willekeurige ruis op elke pixel."},
            "noinput-noise": {"en": "**No input dropout, noisy pictures.** No dropout at the input, and tested on pictures with noise.",
                              "nl": "**Geen dropout bij de input, met ruis.** Geen dropout bij de input, en getest op afbeeldingen met ruis."},
            "mnist": {"en": "**MNIST.** The same networks and steps, on the handwritten digits of the paper itself.",
                      "nl": "**MNIST.** Dezelfde netwerken en stappen, op de handgeschreven cijfers van de paper zelf."},
            "hidden-only": {"en": "**Hidden layers only.** Dropout in the hidden layers, none at the input; the rest as in the reproduction.",
                            "nl": "**Alleen de hidden layers.** Dropout in de hidden layers, geen bij de input; de rest zoals in de reproductie."},
        },
        solution=_SOLUTION,
        nudge={
            "input-only": {"en": "Is a part of the method taken away, or does the data or the test change?",
                           "nl": "Wordt er een deel van de methode weggehaald, of verandert de data of de test?"},
            "mnist": {"en": "Do the networks change, or the pictures they learn from?",
                      "nl": "Veranderen de netwerken, of de afbeeldingen waarvan ze leren?"},
            "noise": {"en": "Does the training change, or only what the trained networks are tested on?",
                      "nl": "Verandert de training, of alleen waarop de getrainde netwerken getest worden?"},
            "wide-mnist": {"en": "Count what differs from the reproduction.",
                           "nl": "Tel wat er anders is dan in de reproductie."},
            "hidden-only": {"en": "Which part of the method is still there, and which is gone?",
                             "nl": "Welk deel van de methode is er nog, en welk deel is weg?"},
            "kmnist": {"en": "What is different: the method, or the pictures it learns from and is tested on?",
                       "nl": "Wat is er anders: de methode, of de afbeeldingen waarvan hij leert en waarop hij getest wordt?"},
            "shift": {"en": "Is the model trained again, or only tested under other conditions?",
                      "nl": "Wordt het model opnieuw getraind, of alleen getest onder andere omstandigheden?"},
            "noinput-noise": {"en": "Count what differs from the reproduction.",
                              "nl": "Tel wat er anders is dan in de reproductie."},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the questions again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de vragen opnieuw uit."},
            "empty": {"en": "Put each question in a group: an ablation, new data, a robustness test, or two changes at once.",
                      "nl": "Zet elke vraag in een groep: een ablation, nieuwe data, een robuustheidstest, of twee veranderingen tegelijk."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "“{item}” is not in its group.",
                      "nl": "‘{item}’ staat niet in zijn groep."},
            "correct": {"en": "All {n} are in their group. Two of them change two things at once: split each into two questions, so that whatever you find has one cause.",
                        "nl": "Alle {n} staan in hun groep. Twee ervan veranderen twee dingen tegelijk: splits elk in twee vragen, zodat wat je vindt één oorzaak heeft."},
        },
    ),
    "can-begin": FunctionBox(
        name="can-begin",
        function="can_begin",
        start=_C_START,
        solution=C_CODE,
        visible=[(_ATTACH, _DONE)],
        hidden=[({}, ["cnn-pytorch"]),
                ({"vision": ["cnn-convolution-idea", "cnn-pytorch"]}, []),
                ({"people": ["start-using-ai-tools", "rai-bias-in-data"], "search": ["py-recursion", "prob-expectation"],
                  "speech": ["torch-pretrained", "cnn-pytorch"]},
                 ["py-recursion", "prob-expectation", "torch-pretrained", "cnn-pytorch", "start-using-ai-tools"]),
                ({"one": ["rl-what-is-rl"]}, ["rl-what-is-rl"])],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("any", wrong=_C_WRONG["any"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take a branch when any of its attach points is done. A branch begins after its first one; the later ones are for its later lessons. Look only at `points[0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt een tak als een van zijn aanhaakpunten af is. Een tak begint na zijn eerste; de latere zijn voor zijn latere lessen. Kijk alleen naar `points[0]`."),
            Mistake("all", wrong=_C_WRONG["all"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you wait until every attach point of a branch is done. A branch can begin after its first one; look only at `points[0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je wacht tot elk aanhaakpunt van een tak af is. Een tak kan beginnen na zijn eerste; kijk alleen naar `points[0]`."),
            Mistake("last", wrong=_C_WRONG["last"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you look at the last attach point of each branch. The points are in route order, so the first, where the branch begins, is `points[0]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je kijkt naar het laatste aanhaakpunt van elke tak. De punten staan in de volgorde van de route, dus het eerste, waar de tak begint, is `points[0]`."),
            Mistake("points", wrong=_C_WRONG["points"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: those are lessons, not branches. Append the name of the branch, the key of `attach`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat zijn lessen, geen takken. Voeg de naam van de tak toe, de sleutel van `attach`."),
        ],
        texts={
            "empty": {"en": "Replace the `...`: go through `attach.items()`, and keep each branch whose first attach point is in `done`.",
                      "nl": "Vervang de `...`: loop door `attach.items()`, en houd elke tak waarvan het eerste aanhaakpunt in `done` staat."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does it give a list of branches, the keys of `attach`, each one whose first attach point is in `done`, in the order of `attach`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Geeft hij een lijst met takken, de sleutels van `attach`, elk waarvan het eerste aanhaakpunt in `done` staat, in de volgorde van `attach`?"},
            "almost": {"en": "Your `can_begin` is right for the lesson's example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work without branches, without lessons done, for a branch with one attach point, and when `done` has the lessons in another order than `attach` has the branches?",
                       "nl": "Je `can_begin` klopt voor het voorbeeld van de les. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij zonder takken, zonder afgeronde lessen, voor een tak met één aanhaakpunt, en als `done` de lessen in een andere volgorde heeft dan `attach` de takken?"},
            "correct": {"en": "Your `can_begin` finds the branches you can begin. Below, it runs on the route data of the course and on your own progress.",
                        "nl": "Je `can_begin` vindt de takken waaraan je kunt beginnen. Hieronder draait hij op de routedata van de course en op je eigen voortgang."},
        },
    ),
}


def my_branches(text: str) -> list | None:
    """The learner's can_begin on the attach points of the route data and the lessons they have done; None when it does
    not run."""
    fn = EXERCISES["can-begin"]._fn(text or "")
    if fn is None:
        return None
    done = done_lessons()
    try:
        return list(fn({k: list(v) for k, v in ATTACH.items()}, [s["id"] for s in course.main_route() if s["id"] in done]))
    except Exception:
        return None


# The lesson is done when these are correct. "next-question" is the open question at the end.
REQUIRED = ["kinds", "can-begin", "next-question"]

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
_C_ANY = _C_WRONG["any"]
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _SOLUTION, "correct", None),
    ("kinds", {"input-only": "ablation", "mnist": "data"}, "wrong", "unfinished"),
    ("kinds", {**_SOLUTION, "noise": "data"}, "wrong", "wrong:noise"),
    ("kinds", {**_SOLUTION, "wide-mnist": "data"}, "wrong", "wrong:wide-mnist"),
    ("kinds", {**_SOLUTION, "hidden-only": "two", "noinput-noise": "ablation"}, "wrong", "wrong:noinput-noise"),
    ("kinds", {**_SOLUTION, "hidden-only": "two"}, "wrong", "wrong:hidden-only"),
    ("can-begin", _C_START, "empty", None),
    ("can-begin", C_CODE, "correct", None),
    ("can-begin", _C_ANY, "wrong", "any"),
    ("can-begin", _C_WRONG["all"], "wrong", "all"),
    ("can-begin", _C_WRONG["last"], "wrong", "last"),
    ("can-begin", _C_WRONG["points"], "wrong", "points"),
    ("can-begin", _C_HEAD + "    ready = []\n    ...\n    return ready\n", "wrong", "dots-left"),
    ("can-begin", _C_HEAD + "    return [b for b, p in attach.items() if p[0] in done]\n", "correct", None),
    ("can-begin", _C_HEAD + "    return [b for b, p in attach.items() if p and p[0] in set(done)]\n", "correct", None),
    ("can-begin", _C_HEAD + "    return [b for d in done for b, p in attach.items() if p[0] == d]\n", "almost", None),
]
