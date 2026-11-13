# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-whats-next. Route: course/route/lessen/onderzoek.yaml. Checks en de getallen over de domeinen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-whats-next", "nl")
    C = L.checks   # wat elk domein leent, uit de routedata
    B = C.borrowers()


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

    Je hebt de kern en de route van onderzoek gedaan: van je eerste regel Python tot een paper die je zelfstandig hebt nagebouwd. En nu? Er zijn twee wegen verder, en je kunt ze allebei nemen. De kaart van de course heeft nog {len(C.OTHERS)} domeinen, van computer vision tot robotica, en elk ervan bouwt voort op wat je nu weet. En je reproductie hoeft niet te eindigen bij zijn verslag: veel onderzoek begint als een reproductie, en groeit uit tot een eigen bijdrage als iemand de volgende vraag over de paper stelt.

    **Een voorbeeld uit het dagelijks leven.** Na een jaar les speel je goed genoeg gitaar om liedjes te spelen die je nooit eerder hoorde. Nu kies je: jazz, klassiek, of eigen liedjes. Elk ervan begint bij akkoorden die je al kent, en een goede leraar kan je vertellen welke.
    """)
    return


@app.cell(hide_code=True)
def predict_shared():
    guess_shared = L.guess(
        "most-borrowed",
        question="Sommige lessen zijn gedeeld: één keer geschreven, en geleend door elk domein dat ze nodig heeft. Welke van deze lenen de meeste domeinen?",
        options={
            "cnn": "CNN's",
            "attention": "Attention en transformers",
            "math": "Wiskunde, formeel",
        },
        reveal=f"**Wiskunde, formeel.** {B['wiskunde']} van de {len(C.DOMAINS)} domeinen lenen die, attention {B['attention']}, en CNN's {B['cnn']}. Voorbij zijn basis heeft bijna elk domein de wiskunde uitgeschreven nodig: afgeleiden, matrices en kansen, die de kern je met voorbeelden liet zien. CNN's zijn voor data in een rooster, zoals beelden en geluid, dus minder domeinen hebben ze nodig.",
    )
    guess_shared
    return (guess_shared,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Gedeelde modules.** Elk domein bouwt voort op de kern. Voorbij hun basis hebben de domeinen meer van dezelfde dingen nodig: de wiskunde formeel uitgeschreven, diepe netwerken trainen, CNN's, attention. Die lessen zijn één keer geschreven, in gedeelde modules van het fundament, en een domein leent de modules die het nodig heeft, op het niveau waarop het ze voor het eerst nodig heeft. Computer vision bijvoorbeeld leent de CNN's op het niveau gevorderd, en spraak ook. Een les die je voor één domein deed, telt voor elk domein dat hem leent: je doet hem nooit twee keer. De routedata van de course noemen wat elk domein leent.

        **Van een reproductie naar je eigen vraag.** Een reproductie vraagt of een claim standhoudt waar de paper hem testte. De volgende vraag is waar hij nog meer standhoudt, of waarom. Verander één ding aan de paper, houd al het andere zoals in je reproductie, en je hebt een nieuw experiment, met de claim, het plan, de seeds en de vergelijking die je al hebt. Drie soorten veranderingen komen veel voor:

        - een **ablation** haalt een deel van de methode weg, om te zien hoeveel van de winst dat deel draagt, zoals in Je reproductie 6;
        - **nieuwe data** draait dezelfde methode op een andere dataset, om te zien of de claim daar ook standhoudt;
        - een **robuustheidstest** houdt het model en zijn training, en test het onder een verstoring die de paper niet probeerde, zoals ruis op de afbeeldingen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **gedeelde module** | een hoofdstuk met lessen in het fundament van de course dat meerdere domeinen lenen, zoals CNN's of attention; elk domein leent het op het niveau waarop het het voor het eerst nodig heeft |
        | **robuustheidstest** | een test van een getraind model onder een verstoring waarvoor het niet getraind is, zoals ruis of verschoven pixels, om te zien of zijn resultaat overeind blijft |
        """),
        L.watch_out("Verander één ding tegelijk. Verander je er twee tegelijk, zoals de data en de grootte van het netwerk, dan heeft een verschil dat je vindt twee mogelijke oorzaken: je kunt niet zeggen welke het was."),
        mo.md(r"""
        Kies hieronder een domein om te zien waar het op voortbouwt.
        """),
    ])
    return


@app.cell(hide_code=True)
def pick_domain():
    domain_pick = mo.ui.dropdown(
        options={C.name(_d, L.lang): _d for _d in C.OTHERS},
        value=C.name("beeld", L.lang), label="Domein")
    domain_pick
    return (domain_pick,)


@app.cell(hide_code=True)
def picture_domain(domain_pick):
    _done = C.done_lessons()
    mo.vstack([
        mo.Html(C.domain_picture(domain_pick.value, _done, {
            "lang": L.lang, "title": "Waar {domain} op voortbouwt", "core": "De kern",
            "levels": ["Basis", "Gevorderd", "Expert"], "count": "{k} van {n} lessen af", "count_one": "{k} van 1 les af",
            "later": "lessen komen nog", "nothing": "niets meer uit het fundament",
            "alt": "Waar het gekozen domein op voortbouwt, van onder naar boven: de kern, en daarboven per niveau de gedeelde modules die het domein leent, elk met hoeveel van zijn lessen je af hebt."})),
        mo.md(C.domain_card(domain_pick.value, L.lang, _done, {
            "project": "Het project", "taste": "Een eerste voorproefje in de kern", "done": "af", "not_done": "nog niet af",
            "lessons": "Lessen", "one": "1 les klaar", "many": "{n} lessen klaar", "in": "in de route ‘{route}’",
            "later": "komen nog"})),
    ])
    return


@app.cell(hide_code=True)
def text_after_picture():
    mo.md(r"""
    Lees de figuur van onder naar boven: de kern, waar elk domein op voortbouwt, en daarboven per niveau de gedeelde modules die het domein leent, elk met hoeveel van zijn lessen je af hebt. Een gestippeld vak is een module waarvan de lessen nog komen. Elk domein gaat op de startpagina open, bij *De hele kaart*, zodra zijn lessen klaar zijn.

    ### Recept

    Om je volgende stap te kiezen:

    1. **Zeg wat je hierna wilt kunnen.** Een deel van AI, zoals beelden of robots, of een vraag over je paper.
    2. **Een deel van AI?** Zoek hierboven zijn domein op: wat het leent, en hoe ver je met elk deel bent. De lessen uit de kern die je er een eerste voorproefje van gaven, als het die heeft, laten zien wat voor werk het is.
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
        ## Zelf doen

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
def text_exercise_ready():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Voor welke domeinen ben je klaar?

        Voor elk domein noemt `uses` de gedeelde modules die het leent tot en met zijn niveau gevorderd, en `done` noemt de gedeelde modules die je af hebt. Maak `ready(uses, done)` af: hij geeft de domeinen waarvan alle gedeelde modules in `done` staan, in de volgorde van `uses`. Een domein dat niets leent, is meteen klaar. In het voorbeeld van de les zijn `vision` en `build` klaar, en heeft `speech` nog `attention` nodig:

        ```python
        uses = {"vision": ["math", "cnn"], "build": [], "speech": ["math", "cnn", "attention"]}
        done = ["cnn", "math"]
        ```

        De check probeert dit voorbeeld, geen domeinen, niets af, domeinen die niets lenen, en modules die in een andere volgorde af zijn dan `uses` ze noemt.
        """),
        L.task("Maak `ready` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ready():
    box_ready = L.code_box("ready", lines=8)
    box_ready
    return (box_ready,)


@app.cell(hide_code=True)
def check_ready(box_ready):
    result_ready = L.check("ready", box_ready.value)
    result_ready
    return (result_ready,)


@app.cell(hide_code=True)
def hints_ready():
    L.hints(
        "ready",
        [
            "`for domain, modules in uses.items():` geeft elk domein met zijn lijst gedeelde modules. Een domein is klaar als geen van zijn modules in `done` ontbreekt.",
            "Verzamel per domein de modules die niet in `done` staan: `[m for m in modules if m not in done]`. Als die lijst leeg is, is het domein klaar, ook als het niets leent.",
        ],
        why="Een domein is klaar als elke module die het leent af is; in een lege lijst ontbreekt niets, dus een domein dat niets leent, is klaar. Door `uses` vanaf het begin te doorlopen, houd je zijn volgorde.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Wat je voortgang zegt

    Je `ready`, op de gedeelde modules die de andere {len(C.OTHERS)} domeinen lenen tot en met hun niveau gevorderd, voor zover die modules al lessen hebben, en op de modules waarvan je alle lessen af hebt:
    """)
    return


@app.cell(hide_code=True)
def demo_ready(box_ready, result_ready):
    _ready = C.my_ready(box_ready.value) if result_ready.state == "correct" else None
    if result_ready.state != "correct":
        _out = mo.md("*Maak `ready` eerst goed in oefening 2; dan draait hij hier.*")
    elif _ready is None:
        _out = mo.md("*Je `ready` stopt met een fout op de routedata van de course en je voortgang. Pas hem aan in oefening 2 en check hem opnieuw.*")
    else:
        _done = C.done_lessons()
        _lines = [f"**Voor {len(_ready)} van de {len(C.OTHERS)} domeinen heb je elke gedeelde module af die ze tot en met gevorderd lenen.**", ""]
        _lines += [f"- {C.name(_d, L.lang)}" for _d in _ready if _d in C.OTHERS]
        _wait = [_d for _d in C.OTHERS if _d not in _ready]
        if _wait:
            _lines += ["", "Nog niet:", ""]
            for _d in _wait:
                _todo = C.missing(_d, _done)
                _parts = [f"{C.module_name(_m, L.lang)}, {_k} van {_n} {"les" if _n == 1 else "lessen"} af, volgende [{C.course.title(_next, L.lang)}](/course/open/{_next}?lang={L.lang})"
                          for _m, _k, _n, _next in _todo]
                _lines.append(f"- {C.name(_d, L.lang)}: " + "; ".join(_parts))
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

        **Samenhang.** De kaart uit [De kaart van AI en hoe deze course in elkaar zit](/course/open/start-map-of-ai?lang=nl), aan het begin van de course, nu met de gedeelde modules die elk domein leent; en het deel *Not checked* van [Je reproductie 8: verslag en delen](/course/open/repro-own-report?lang=nl), omgezet in je volgende vraag. Dit is de laatste les van de route van onderzoek, de laatste stap naar de laatste van deze mijlpalen:
        """),
        mo.md(C.milestone_lines(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def wrap_up(result_kinds, result_ready, explain_next_cell):
    L.wrap_up(result_kinds, result_ready, explain_next_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
