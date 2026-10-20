# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-whats-next. Route: course/route/lessen/onderzoek.yaml. Checks and the numbers about the branches: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-whats-next", "nl")
    C = L.checks   # the attach points of every branch, from the route data
    FIRST = C.first_levels()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De kaart van AI en hoe deze course in elkaar zit](/course/open/start-map-of-ai?lang=nl) en [Je reproductie 8: verslag en delen](/course/open/repro-own-report?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_locked():
    quick_locked = L.choice(
        "quick-locked",
        question="Op de startpagina valt je oog op een les van het niveau expert, terwijl je nog bij de basis bent. Wat gebeurt er als je hem opent?",
        options={
            "locked": "Hij blijft dicht tot je elke les van gevorderd hebt gedaan",
            "right": "Hij gaat open, en *Voorkennis* noemt de lessen waar hij op voortbouwt",
            "count": "Hij gaat open, maar zijn oefeningen tellen pas na de niveaus ervoor",
        },
        correct="right",
        why={
            "right": "Geen enkele les zit op slot: een niveau zegt alleen hoe gevorderd een les is. *Voorkennis* laat zien waar hij op voortbouwt, met een vinkje bij elke les die je af hebt.",
            "locked": "Een niveau is een label, geen level van een spel: geen enkele les zit op slot. Zie [De kaart van AI en hoe deze course in elkaar zit](/course/open/start-map-of-ai?lang=nl).",
            "count": "Elke oefening telt, wanneer je hem ook doet: een niveau zegt alleen hoe gevorderd een les is. Zie [De kaart van AI en hoe deze course in elkaar zit](/course/open/start-map-of-ai?lang=nl).",
        },
    )
    quick_locked
    return (quick_locked,)


@app.cell(hide_code=True)
def quick_check_not_checked():
    quick_not_checked = L.choice(
        "quick-not-checked",
        question="Het deel *Not checked* van het verslag van de course noemt MNIST zelf, grotere netwerken en meer seeds. Waar is het voor?",
        options={
            "failed": "Het noemt de runs die misgingen, zodat een lezer ze kan weglaten",
            "right": "Het voorkomt dat een lezer het resultaat voor meer houdt dan getest is",
            "modest": "Het toont bescheidenheid: een verslag moet niet te zeker van zichzelf klinken",
        },
        correct="right",
        why={
            "right": "Een lezer zou kunnen denken dat MNIST ook getest is, en dat is niet zo. Zo is het deel ook een lijst met open vragen, en deze les komt erop terug.",
            "failed": "Een run die misging, komt in de Results zoals elke andere. *Not checked* noemt wat nooit gedraaid is. Zie [Je reproductie 8](/course/open/repro-own-report?lang=nl).",
            "modest": "Het gaat niet om de toon: het zegt wat niet getest is, zodat een lezer het resultaat kan wegen. Zie [Je reproductie 8](/course/open/repro-own-report?lang=nl).",
        },
    )
    quick_not_checked
    return (quick_not_checked,)


@app.cell(hide_code=True)
def text_why():
    mo.md(rf"""
    ## Waarom

    Je hebt de hoofdroute gedaan: van je eerste regel Python tot een eigen LLM, gepretraind en nagetraind, en een paper die je zelfstandig hebt nagebouwd. En nu? Er zijn twee wegen verder, en je kunt ze allebei nemen. De kaart van de course heeft nog {len(C.LATER)} takken, van computer vision tot robotica, en elk ervan bouwt voort op wat je nu weet. En je reproductie hoeft niet te eindigen bij zijn verslag: veel onderzoek begint als een reproductie, en groeit uit tot een eigen bijdrage als iemand de volgende vraag over de paper stelt.

    **Een voorbeeld uit het dagelijks leven.** Na een jaar les speel je goed genoeg gitaar om liedjes te spelen die je nooit eerder hoorde. Nu kies je: jazz, klassiek, of eigen liedjes. Elk ervan begint bij akkoorden die je al kent, en een goede leraar kan je vertellen welke.
    """)
    return


@app.cell(hide_code=True)
def predict_level():
    guess_level = L.guess(
        "level",
        question=f"Na welk niveau van de hoofdroute kunnen de meeste van de {len(C.LATER)} takken die nog komen, beginnen?",
        options={
            "basis": "Na de basis",
            "gevorderd": "Na gevorderd",
            "expert": "Pas na expert, aan het eind van de hoofdroute",
        },
        reveal=f"**Na de basis.** {FIRST['basis']} van de {len(C.LATER)} takken kunnen meteen na een les van het niveau basis beginnen, en {FIRST['gevorderd']} na een les van gevorderd: {', '.join(C.later_names('nl'))}, die voortbouwt op je project met een diep netwerk. Elke tak bouwt voort op de hoofdroute, maar de meeste hebben alleen het begin ervan nodig om te beginnen. Als je alle drie de niveaus hebt gedaan, kun je aan elk ervan beginnen.",
    )
    guess_level
    return (guess_level,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Aanhaakpunten.** Elke tak die nog komt, groeit uit de hoofdroute, zoals een tak uit een stam. Waar hij aanhaakt, is een aanhaakpunt: een les van de hoofdroute waarop de tak voortbouwt. Een tak kan er meer dan één hebben: hij kan beginnen na het eerste, en zijn latere lessen bouwen ook voort op de andere. Computer vision bijvoorbeeld begint na *Convolutie: één klein filter schuift over de afbeelding*, op het niveau basis, en zijn lessen over diepere CNN's bouwen voort op *Project: je eerste CNN*, op gevorderd. De routedata van de course noemen de aanhaakpunten van elke tak.

        **Van een reproductie naar je eigen vraag.** Een reproductie vraagt of een claim standhoudt waar de paper hem testte. De volgende vraag is waar hij nog meer standhoudt, of waarom. Verander één ding aan de paper, houd al het andere zoals in je reproductie, en je hebt een nieuw experiment, met de claim, het plan, de seeds en de vergelijking die je al hebt. Drie soorten veranderingen komen veel voor:

        - een **ablation** haalt een deel van de methode weg, om te zien hoeveel van de winst dat deel draagt, zoals in Je reproductie 6;
        - **nieuwe data** draait dezelfde methode op een andere dataset, om te zien of de claim daar ook standhoudt;
        - een **robuustheidstest** houdt het model en zijn training, en test het onder een verstoring die de paper niet probeerde, zoals ruis op de afbeeldingen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **aanhaakpunt** | een les van de hoofdroute waarop een tak van de course aanhaakt; de tak kan beginnen na zijn eerste aanhaakpunt, en bouwt vanaf daar voort op wat je weet |
        | **robuustheidstest** | een test van een getraind model onder een verstoring waarvoor het niet getraind is, zoals ruis of verschoven pixels, om te zien of zijn resultaat overeind blijft |
        """),
        L.watch_out("Verander één ding tegelijk. Verander je er twee tegelijk, zoals de data en de grootte van het netwerk, dan heeft een verschil dat je vindt twee mogelijke oorzaken: je kunt niet zeggen welke het was."),
        mo.md(r"""
        Kies hieronder een tak om te zien waar hij uit de hoofdroute groeit.
        """),
    ])
    return


@app.cell(hide_code=True)
def pick_branch():
    branch_pick = mo.ui.dropdown(
        options={C.short_title(_b["id"], L.lang): _b["id"] for _b in C.LATER},
        value=C.short_title(C.LATER[0]["id"], L.lang), label="Tak")
    branch_pick
    return (branch_pick,)


@app.cell(hide_code=True)
def picture_tree(branch_pick):
    _done = C.done_lessons()
    mo.vstack([
        C.tree_picture(branch_pick.value, _done, {
            "lang": L.lang, "title": "Waar de tak uit de hoofdroute groeit",
            "levels": ["Basis", "Gevorderd", "Expert"], "you": "hier ben je",
            "alt": "De hoofdroute als een liggende stam in drie niveaus. De gekozen tak gaat eruit bij zijn eerste aanhaakpunt, en stippellijnen verbinden zijn andere aanhaakpunten ermee. Korte grijze stompjes laten zien waar de andere takken beginnen."}),
        mo.md(C.branch_card(branch_pick.value, L.lang, _done, {
            "project": "Zijn project", "attach": "Zijn aanhaakpunten", "done": "af", "not_done": "nog niet af",
            "extras": "Verdiepingslessen van de hoofdroute uit deze tak, een begin dat je nu al kunt maken:"})),
    ])
    return


@app.cell(hide_code=True)
def text_after_picture():
    mo.md(r"""
    Lees de stam van links naar rechts: de hoofdroute, van de basis tot expert. De genummerde stippen zijn de aanhaakpunten van de tak die je koos; hij groeit eruit bij stip 1. De grijze stompjes laten zien waar de andere takken beginnen, en *hier ben je* staat bij de verste les van de hoofdroute die je af hebt. Elke tak gaat op de startpagina open, bij *De hele kaart*, zodra hij klaar is.

    ### Recept

    Om je volgende stap te kiezen:

    1. **Zeg wat je hierna wilt kunnen.** Een deel van AI, zoals beelden of robots, of een vraag over je paper.
    2. **Een deel van AI?** Zoek hierboven zijn tak en zijn aanhaakpunten op, en kijk terug naar die lessen: daar bouwt de tak op voort. Tot de tak opengaat, zijn zijn verdiepingslessen van de hoofdroute, als hij die heeft, een begin dat je nu al kunt maken.
    3. **Een vraag over je paper?** Begin bij het deel *Not checked* van je verslag, en kies één verandering: een ablation, nieuwe data of een robuustheidstest.
    4. **Schrijf hem op als een plan voordat je draait,** zoals in Je reproductie 2: wat je verandert, wat hetzelfde blijft, en volgens welke regel je zegt dat de claim standhield.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De reproductie die de course van de dropout-paper maakte, zegt onder *Not checked*: MNIST zelf, grotere netwerken, meer seeds. Hier zijn drie volgende vragen, één van elke soort; de eerste komt rechtstreeks uit dat lijstje:

    | Soort | Vraag | Wat verandert | Wat hetzelfde blijft |
    |---|---|---|---|
    | nieuwe data | Houdt de winst stand op MNIST, de eigen cijfers van de paper? | de dataset | de netwerken, de dropout, de stappen en de seeds |
    | ablation | Heeft de winst de volle kans van 0,5 in de hidden layers nodig? | een kans van 0,25 in de hidden layers | de data, de dropout van de input, de stappen en de seeds |
    | robuustheidstest | Houdt de winst stand als de testafbeeldingen ruis hebben? | de testafbeeldingen | de netwerken en hun training |

    *Grotere netwerken* is een vierde vraag. Samen met MNIST gesteld, zou hij twee dingen tegelijk veranderen: stel hem op Fashion-MNIST, als een eigen vraag.

    Welke eerst? Een goede eerste vraag verandert één ding, past in je budget, en heeft een antwoord dat interessant is, welke kant het ook op valt. Dat doen ze alle drie. De course zou met MNIST beginnen: zijn verslag vond fouten die acht tot negen keer zo groot zijn als die van de paper, en een winst die groter is in procentpunten en kleiner als aandeel van de fout. Op MNIST, met fouten dichter bij die van de paper, zie je of dat verschil in grootte van de data komt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Wat voor vraag?

        Elke vraag hieronder verandert iets aan de reproductie die de course van de dropout-paper maakte: netwerken van 784-512-512-10 op Fashion-MNIST, met dropout bij de input en in de hidden layers, drie seeds.
        """),
        L.task("Zet elke vraag in zijn groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Een ablation haalt een deel van de methode weg; nieuwe data verandert waarvan de netwerken leren en waarop ze getest worden; een robuustheidstest verandert alleen waarop de getrainde netwerken getest worden. De dropout bij de input en de dropout in de hidden layers zijn de twee delen van de methode.",
            "Tel eerst de veranderingen in elke vraag. Twee veranderingen zetten hem in de laatste groep, wat ze ook zijn.",
        ],
        why="Ablation: **Alleen de input** en **Alleen de hidden layers**. Nieuwe data: **MNIST** en **Kuzushiji-MNIST**. Robuustheidstest: **Afbeeldingen met ruis** en **Verschoven afbeeldingen**. Twee veranderingen tegelijk: **Breder, op MNIST** en **Geen dropout bij de input, met ruis**.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_can_begin():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Aan welke takken kun je beginnen?

        De routedata geven elke tak zijn aanhaakpunten, in de volgorde van de route, en je voortgang zegt welke lessen je hebt gedaan. Maak `can_begin(attach, done)` af: hij geeft de takken waarvan het eerste aanhaakpunt in `done` staat, in de volgorde van `attach`. In het voorbeeld van de les kunnen `vision` en `robots` beginnen, en `depth` nog niet:

        ```python
        attach = {"vision": ["cnn-convolution-idea", "cnn-pytorch"],
                  "depth": ["nn-deep-net-project", "llm-modern-gpt"],
                  "robots": ["math-matrix-multiply", "rl-what-is-rl"]}
        done = ["cnn-convolution-idea", "llm-modern-gpt", "math-matrix-multiply", "rl-what-is-rl"]
        ```

        De check probeert dit voorbeeld, geen takken, geen afgeronde lessen, een tak met één aanhaakpunt, en afgeronde lessen in een andere volgorde dan de takken.
        """),
        L.task("Maak `can_begin` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_can_begin():
    box_can_begin = L.code_box("can-begin", lines=8)
    box_can_begin
    return (box_can_begin,)


@app.cell(hide_code=True)
def check_can_begin(box_can_begin):
    result_can_begin = L.check("can-begin", box_can_begin.value)
    result_can_begin
    return (result_can_begin,)


@app.cell(hide_code=True)
def hints_can_begin():
    L.hints(
        "can-begin",
        [
            "`for branch, points in attach.items():` geeft elke tak met zijn lijst aanhaakpunten; zijn eerste is `points[0]`.",
            "Begin met een lege lijst, en houd alleen de takken die kunnen beginnen; wat dat beslist, is hun eerste aanhaakpunt.",
        ],
        why="Een tak begint na zijn eerste aanhaakpunt, dus daar kijk je naar. Door `attach` vanaf het begin te doorlopen, blijft zijn volgorde behouden.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Wat je voortgang zegt

    Je `can_begin`, op de aanhaakpunten van alle {len(C.LATER)} takken en op de lessen die je hebt gedaan:
    """)
    return


@app.cell(hide_code=True)
def demo_can_begin(box_can_begin, result_can_begin):
    _ready = C.my_branches(box_can_begin.value) if result_can_begin.state == "correct" else None
    if result_can_begin.state != "correct":
        _out = mo.md("*Maak eerst `can_begin` goed in oefening 2; dan draait hij hier.*")
    elif _ready is None:
        _out = mo.md("*Je `can_begin` stopt met een fout op de routedata van de course en je voortgang. Verander hem in oefening 2 en check hem opnieuw.*")
    else:
        _lines = [f"**Je kunt aan {len(_ready)} van de {len(C.LATER)} takken beginnen.**", ""]
        _lines += [f"- {C.short_title(_k, L.lang)}" for _k in _ready if _k in C.ATTACH]
        _wait = C.waiting(_ready)
        if _wait:
            _lines += ["", "Nog niet:", ""]
            _lines += [f"- {C.short_title(_k, L.lang)}, na [{C.course.title(C.ATTACH[_k][0], L.lang)}](/course/open/{C.ATTACH[_k][0]}?lang={L.lang})" for _k in _wait]
        _out = mo.md("\n".join(_lines))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Oefening 3. Je volgende vraag

    Je verslag is geschreven, en zijn deel *Not checked* noemt wat je niet testte. Maak van een daarvan, of van een andere vraag over je paper, het plan van een volgend experiment.
    """)
    return


@app.cell(hide_code=True)
def explain_next():
    explain_next_cell = L.explain(
        "next-question",
        question="Schrijf de volgende vraag voor de paper die je nabouwde: wat voor verandering het is, wat je verandert en wat hetzelfde blijft, en de regel waarmee je zegt dat de claim standhield, opgeschreven voordat je draait.",
        model_answer="Houdt de winst van dropout stand op MNIST, de eigen cijfers van de paper? Nieuwe data: ik train dezelfde netwerken van 784-512-512-10, met en zonder dropout, op 6000 afbeeldingen van MNIST in plaats van Fashion-MNIST, met dezelfde dropout, dezelfde 12.800 stappen en dezelfde drie seeds. De regel, opgeschreven voordat ik draai: de claim houdt stand op MNIST als de testfout met dropout voor alle drie de seeds lager is. Houdt hij stand, dan vergelijk ik ook de grootte van de winst met de 0,25 procentpunt van de paper, om mijn vermoeden te toetsen dat de moeilijkere afbeeldingen het verschil verklaarden.",
        points=[
            "je noemt de soort verandering: een ablation, nieuwe data of een robuustheidstest",
            "je verandert één ding, en zegt wat hetzelfde blijft",
            "je schrijft de regel voor de claim op voordat je draait",
            "de vraag groeit uit je reproductie, bijvoorbeeld uit zijn *Not checked*",
        ],
    )
    explain_next_cell
    return (explain_next_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.vstack([
        mo.md(rf"""
        ## Afronding

        **Samenhang.** De kaart uit [De kaart van AI en hoe deze course in elkaar zit](/course/open/start-map-of-ai?lang=nl), aan het begin van de course, nu met de aanhaakpunten die laten zien waar elke tak uit de hoofdroute groeit; en het deel *Not checked* van [Je reproductie 8: verslag en delen](/course/open/repro-own-report?lang=nl), omgezet in je volgende vraag. Dit is de laatste les van de hoofdroute, de laatste stap naar de derde van zijn mijlpalen:
        """),
        mo.md(C.milestone_lines(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def wrap_up(result_kinds, result_can_begin, explain_next_cell):
    L.wrap_up(result_kinds, result_can_begin, explain_next_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
