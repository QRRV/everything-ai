# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-whats-next, and the numbers about the domains that the lesson shows.

Every domain builds on the core, and borrows shared modules of the foundation at the level where it first needs them
(`gebruikt` in route/domeinen.yaml). The numbers, the picture and the card come from there and from the learner's
progress, so the lesson stays right when the course grows. The exercises: sort next questions of a reproduction by their
kind, and ready, the domains whose shared modules up to the intermediate level you have all done."""
import html
from collections import Counter

from everything_ai import course, progress
from everything_ai.check import FunctionBox, Mistake, Sort
from everything_ai import yours

LEVELS = ["basis", "gevorderd", "expert"]
HERE = "onderzoek"   # the domain this lesson belongs to
DOMAINS = [d["id"] for d in course.structure()["domeinen"]]
OTHERS = [d for d in DOMAINS if d != HERE]
SHARED = [m["id"] for m in course.structure()["fundament"]["modules"] if not m.get("kern")]
USES = {d["id"]: {lv: list(d["gebruikt"].get(lv, [])) for lv in LEVELS} for d in course.structure()["domeinen"]}
CORE = [s["id"] for s in course.kern_route() if not s["extra"]]   # the core without its extra lessons
MILESTONES = course.milestones() + course.milestones(course.route_for("res-whats-next"))   # the wrap-up lists them all


def domain(domain_id: str) -> dict:
    return next(d for d in course.structure()["domeinen"] if d["id"] == domain_id)


def name(domain_id: str, lang: str) -> str:
    """The name of a domain, as the start page shows it."""
    return course.part_name(domain_id, lang)


def module_name(module_id: str, lang: str) -> str:
    """The name of a shared module, as the start page shows it."""
    return course.display()["modules"][f"fundament/{module_id}"][lang]


def module_lessons(module_id: str) -> list[str]:
    """The built lessons of a shared module, in the order of the course."""
    lessons = course.lessons()
    return [lid for lid in course.order()
            if lessons[lid]["domein"] == "fundament" and lessons[lid]["module"] == module_id and course.is_built(lid)]


def borrowers() -> Counter:
    """How many domains borrow each shared module, at any level."""
    return Counter(m for d in DOMAINS for m in {m for lv in LEVELS for m in USES[d][lv]})


def up_to_intermediate(domain_id: str) -> list[str]:
    """The shared modules a domain borrows at the basics and the intermediate level, as far as they have lessons yet."""
    return [m for lv in ("basis", "gevorderd") for m in USES[domain_id][lv] if module_lessons(m)]


def done_lessons() -> set[str]:
    """The lessons the learner has done, from their progress."""
    return {lid for lid, p in progress.all_lessons().items() if p.get("done")}


def done_modules(done: set) -> list[str]:
    """The shared modules with lessons, every one of which the learner has done."""
    return [m for m in SHARED if module_lessons(m) and all(lid in done for lid in module_lessons(m))]


def milestone_lines(lang: str) -> str:
    """The milestones of the core and of the route of this lesson as a Markdown list, in the learner's language."""
    return "\n".join(f"- {m[lang]}" for m in MILESTONES)


def _chip(title: str, line: str, share: float | None) -> str:
    """One shared module in the picture: its name, a line under it, and a bar for how much of it is done; a module
    without lessons yet (share None) has a dashed box and no bar."""
    border = "1px dashed var(--ea-ink-3)" if share is None else "1px solid var(--ea-ink-2)"
    bar = ("" if share is None else
           f'<div style="height:4px;border-radius:2px;background:var(--ea-line);margin-top:0.3rem">'
           f'<div style="height:4px;border-radius:2px;width:{100 * share:.0f}%;background:var(--ea-ink-2)"></div></div>')
    return (f'<div style="border:{border};border-radius:6px;padding:0.35rem 0.5rem">'
            f'<div style="font-size:0.85rem;font-weight:600;color:var(--ea-ink);line-height:1.25">{html.escape(title)}</div>'
            f'<div style="font-size:0.78rem;color:var(--ea-ink-2);margin-top:0.1rem">{html.escape(line)}</div>{bar}</div>')


def domain_picture(domain_id: str, done: set, words: dict) -> str:
    """What a domain builds on, read from the bottom up: the core, filled, with how many of its lessons you have done;
    above it a row per level with the shared modules the domain borrows there, each with how many of its lessons you
    have done and a bar. Ink only: nothing here is learned or counted wrong. HTML, so the names wrap on a phone."""
    lang = words["lang"]

    def count(lessons: list) -> tuple[str, float | None]:
        if not lessons:
            return words["later"], None
        k = sum(1 for lid in lessons if lid in done)
        return words["count_one" if len(lessons) == 1 else "count"].format(k=k, n=len(lessons)), k / len(lessons)

    rows = []
    for lv, label in zip(LEVELS, words["levels"]):
        chips = "".join(_chip(module_name(m, lang), *count(module_lessons(m))) for m in USES[domain_id][lv])
        body = (f'<div style="display:grid;gap:0.35rem;grid-template-columns:repeat(auto-fill,minmax(9.5rem,1fr))">{chips}</div>'
                if chips else f'<div style="font-size:0.85rem;color:var(--ea-ink-3);padding:0.35rem 0">{html.escape(words["nothing"])}</div>')
        rows.append(f'<div style="margin-bottom:0.55rem"><div style="font-size:0.8rem;font-weight:650;color:var(--ea-ink-2);'
                    f'margin-bottom:0.25rem">{html.escape(label)}</div>{body}</div>')
    line, _ = count(CORE)
    core = (f'<div style="background:var(--ea-inset);border:1px solid var(--ea-ink-2);border-radius:6px;padding:0.4rem 0.55rem">'
            f'<div style="font-size:0.85rem;font-weight:600;color:var(--ea-ink)">{html.escape(words["core"])}</div>'
            f'<div style="font-size:0.78rem;color:var(--ea-ink-2);margin-top:0.1rem">{html.escape(line)}</div></div>')
    title = words["title"].format(domain=name(domain_id, lang))
    return yours.figure(title, f'<div style="font-family:var(--ea-font-ui)">{"".join(reversed(rows))}{core}</div>',
                        alt=words["alt"])


def domain_card(domain_id: str, lang: str, done: set, words: dict) -> str:
    """What the lesson says about one domain, in Markdown: its project, the lessons of the core that gave a first taste
    of it, with a link and whether you did them, and how many of its lessons are ready, and in which route."""
    out = [f"**{name(domain_id, lang)}**  ", f"{words['project']}: {course.display()['projects'][domain_id][lang]}."]
    tasters = domain(domain_id).get("proef")
    if isinstance(tasters, list) and tasters:
        out += ["", f"{words['taste']}:", ""]
        out += [f"- [{course.title(t, lang)}](/course/open/{t}?lang={lang}): {words['done'] if t in done else words['not_done']}"
                for t in tasters]
    lessons = [lid for lid in course.built_lessons() if lid in course.lessons() and course.part(lid) == domain_id]
    routes = {r["domein"]: r for r in course.domain_routes()}
    if not lessons:
        ready = words["later"]
    else:
        # its own route, else the first route that has its first lesson, the core included
        others = [course.kern_names()] + course.domain_routes()
        where = routes[domain_id] if domain_id in routes else next(
            (r for r in others if any(s["id"] == lessons[0] for s in course.route_steps(r["id"]))), None)
        count = words["one"] if len(lessons) == 1 else words["many"].format(n=len(lessons))
        ready = count + ("" if where is None else " " + words["in"].format(route=where["short"][lang]))
    out += ["", f"{words['lessons']}: {ready}."]
    return "\n".join(out)


# --- exercise 2: ready ------------------------------------------------------------------------------------------------

_C_HEAD = ("def ready(uses, done):\n"
           "    # uses: for each domain, the shared modules it borrows up to its intermediate level\n"
           "    # done: the shared modules you have finished\n"
           "    # gives a list of the domains whose shared modules are all in done, in the order of uses\n")
_C_BODY = ("    ready = []\n"
           "    for domain, modules in uses.items():\n"
           "        missing = [m for m in modules if m not in done]\n"
           "        if len(missing) == 0:\n"
           "            ready.append(domain)\n"
           "    return ready\n")
_C_START = _C_HEAD + "    ...\n"
C_CODE = _C_HEAD + _C_BODY
_C_WRONG = {
    "any": C_CODE.replace("        missing = [m for m in modules if m not in done]\n        if len(missing) == 0:\n",
                          "        found = [m for m in modules if m in done]\n        if len(found) > 0:\n"),
    "nothing": C_CODE.replace("        missing = [m for m in modules if m not in done]\n        if len(missing) == 0:\n",
                              "        missing = [m for m in modules if m not in done]\n        if len(modules) > 0 and len(missing) == 0:\n"),
    "every": C_CODE.replace("        missing = [m for m in modules if m not in done]\n        if len(missing) == 0:\n            ready.append(domain)\n",
                            "        for m in modules:\n            if m in done:\n                ready.append(domain)\n"),
    "modules": C_CODE.replace("ready.append(domain)", "ready.append(modules)"),
}
_USES = {"vision": ["math", "cnn"], "build": [], "speech": ["math", "cnn", "attention"]}
_DONE = ["cnn", "math"]
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
    "ready": FunctionBox(
        name="ready",
        picture=yours.steps(call=0),
        function="ready",
        start=_C_START,
        solution=C_CODE,
        visible=[(_USES, _DONE)],
        hidden=[({}, ["math"]),
                ({"vision": ["math", "cnn"]}, []),
                ({"build": [], "people": []}, []),
                ({"robots": ["signals", "math"], "build": [], "text": ["attention"], "speech": ["signals", "cnn"]},
                 ["attention", "math", "signals"]),
                ({"one": ["math"]}, ["math"])],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("first", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops: a domain that borrows nothing has an empty list, and an empty list has no first item. Look at every module of a domain, not only the first.",
                    nl="Met `{call}` stopt je functie: een domein dat niets leent, heeft een lege lijst, en een lege lijst heeft geen eerste onderdeel. Kijk naar elke module van een domein, niet alleen naar de eerste."),
            Mistake("any", wrong=_C_WRONG["any"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you take a domain when one of its modules is done. It is ready only when every module it borrows is done, and a domain that borrows nothing is ready right away.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je neemt een domein als een van zijn modules af is. Het is pas klaar als elke module die het leent af is, en een domein dat niets leent, is meteen klaar."),
            Mistake("nothing", wrong=_C_WRONG["nothing"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: a domain that borrows nothing is missing from your list. It needs no module you have not done, so it is ready.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een domein dat niets leent, ontbreekt in je lijst. Het heeft geen module nodig die je nog niet af hebt, dus het is klaar."),
            Mistake("every", wrong=_C_WRONG["every"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: you append a domain once for every module of it that is done. Look at all the modules of a domain first, and then append it once, if none is missing.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je voegt een domein toe voor elke module ervan die af is. Kijk eerst naar alle modules van een domein, en voeg het dan één keer toe, als er geen ontbreekt."),
            Mistake("modules", wrong=_C_WRONG["modules"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: those are lists of modules, not domains. Append the name of the domain, the key of `uses`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat zijn lijsten met modules, geen domeinen. Voeg de naam van het domein toe, de sleutel van `uses`."),
        ],
        texts={
            "empty": {"en": "Replace the `...`: go through `uses.items()`, and keep each domain none of whose modules is missing from `done`.",
                      "nl": "Vervang de `...`: loop door `uses.items()`, en houd elk domein waarvan geen enkele module in `done` ontbreekt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does it give a list of domains, the keys of `uses`, each one whose modules are all in `done`, in the order of `uses`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Geeft hij een lijst met domeinen, de sleutels van `uses`, elk waarvan alle modules in `done` staan, in de volgorde van `uses`?"},
            "almost": {"en": "Your `ready` is right for the lesson's example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work without domains, with nothing done, for domains that borrow nothing, and when `done` has the modules in another order than `uses`?",
                       "nl": "Je `ready` klopt voor het voorbeeld van de les. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij zonder domeinen, als er niets af is, voor domeinen die niets lenen, en als `done` de modules in een andere volgorde heeft dan `uses`?"},
            "correct": {"en": "Your `ready` finds the domains you are ready for. Below, it runs on the route data of the course and on your own progress.",
                        "nl": "Je `ready` vindt de domeinen waar je klaar voor bent. Hieronder draait hij op de routedata van de course en op je eigen voortgang."},
        },
    ),
}


def my_ready(text: str) -> list | None:
    """The learner's ready on what the other domains borrow up to intermediate and the shared modules they have done;
    None when it does not run."""
    fn = EXERCISES["ready"]._fn(text or "")
    if fn is None:
        return None
    try:
        return list(fn({d: up_to_intermediate(d) for d in OTHERS}, done_modules(done_lessons())))
    except Exception:
        return None


def missing(domain_id: str, done: set) -> list[tuple[str, int, int, str]]:
    """The shared modules up to intermediate that a domain still needs: (module, lessons done, lessons, the first lesson
    not done yet)."""
    out = []
    for m in up_to_intermediate(domain_id):
        lessons = module_lessons(m)
        todo = [lid for lid in lessons if lid not in done]
        if todo:
            out.append((m, len(lessons) - len(todo), len(lessons), todo[0]))
    return out


# The lesson is done when these are correct. "next-question" is the open question at the end.
REQUIRED = ["kinds", "ready", "next-question"]

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _SOLUTION, "correct", None),
    ("kinds", {"input-only": "ablation", "mnist": "data"}, "wrong", "unfinished"),
    ("kinds", {**_SOLUTION, "noise": "data"}, "wrong", "wrong:noise"),
    ("kinds", {**_SOLUTION, "wide-mnist": "data"}, "wrong", "wrong:wide-mnist"),
    ("kinds", {**_SOLUTION, "hidden-only": "two", "noinput-noise": "ablation"}, "wrong", "wrong:noinput-noise"),
    ("kinds", {**_SOLUTION, "hidden-only": "two"}, "wrong", "wrong:hidden-only"),
    ("ready", _C_START, "empty", None),
    ("ready", C_CODE, "correct", None),
    ("ready", _C_WRONG["any"], "wrong", "any"),
    ("ready", _C_WRONG["nothing"], "wrong", "nothing"),
    ("ready", _C_WRONG["every"], "wrong", "every"),
    ("ready", _C_WRONG["modules"], "wrong", "modules"),
    ("ready", _C_HEAD + "    ready = []\n    for domain, modules in uses.items():\n        if modules[0] in done:\n"
              "            ready.append(domain)\n    return ready\n", "error", "first"),
    ("ready", _C_HEAD + "    ready = []\n    ...\n    return ready\n", "wrong", "dots-left"),
    ("ready", _C_HEAD + "    return [d for d, ms in uses.items() if all(m in done for m in ms)]\n", "correct", None),
    ("ready", _C_HEAD + "    ready = []\n    for domain, modules in uses.items():\n        ok = True\n        for m in modules:\n"
              "            if m not in done:\n                ok = False\n        if ok:\n            ready.append(domain)\n"
              "    return ready\n", "correct", None),
    ("ready", _C_HEAD + "    return [d for d, ms in uses.items() if len(ms) <= len(done)]\n", "almost", None),
]
