# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-meaning-and-truth. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-meaning-and-truth", "nl")
    C = L.checks   # de wereld van de course, de zinnen, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_part():
    quick_part = L.choice(
        "quick-part",
        question="Welke woordgroep in *de late trein verliet de stad na middernacht* kan plaatsmaken voor het ene woord *toen*?",
        options={
            "right": "*na middernacht*",
            "left": "*verliet de stad na*",
            "train": "*trein verliet*",
        },
        correct="right",
        why={
            "right": "*De late trein verliet de stad toen*: *na middernacht* werkt als één geheel, een constituent, hier een voorzetselgroep. Deze les geeft zulke delen van een zin een eigen betekenis.",
            "left": "Geen enkel woord kan in de plaats komen van *verliet de stad na*: de groep scheidt *na* van *middernacht*. Het is geen constituent. Zie *Constituenten* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
            "train": "*De late toen de stad na middernacht* is geen zin: *trein verliet* is geen constituent. Zie *Constituenten* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
        },
    )
    quick_part
    return (quick_part,)


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="De grammatica van *Zinsbouw en eindeloze zinnen* begint met de regel S → NP VP. Wat zegt die regel?",
        options={
            "right": "S mag je herschrijven als NP en dan VP",
            "or": "S mag je herschrijven als NP, of anders als VP",
            "inside": "NP bevat VP",
        },
        correct="right",
        why={
            "right": "Het symbool links mag je herschrijven als de rij rechts: een zin is een naamwoordgroep gevolgd door een werkwoordgroep. Deze les doet hetzelfde met betekenis: de betekenis van S komt uit die van de NP en de VP.",
            "or": "Een keuze schrijf je met een streep, zoals in NP → DET NOM | PROPN. S → NP VP heeft geen streep: de rechterkant is één rij, NP en dan VP. Zie *Regels die zinnen bouwen* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
            "inside": "De regel herschrijft S, het symbool links, als de rij rechts: NP en VP staan naast elkaar onder S. Zie *Regels die zinnen bouwen* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In *Zinsbouw en eindeloze zinnen* zei een grammatica welke rijen woorden zinnen zijn, en gaf elke zin een boom. Wat een zin betekent, zei die grammatica niet. Een programma dat een menukaart leest om een gerecht te kiezen, of een recept volgt, heeft meer nodig. Jurafsky en Martin openen hun bijlage over de betekenis van zinnen met zulke taken: die taken vragen om voorstellingen die de woorden verbinden met kennis van de wereld buiten de taal.

    Een oud antwoord op de vraag *wat betekent een zin?* begint bij waarheid. De logicus Alfred Tarski (1944) gaf het bekendste voorbeeld: de zin *sneeuw is wit* is waar als, en alleen als, sneeuw wit is. De filosoof Donald Davidson (1967) bouwde erop voort: volgens Davidson is het geven van de waarheidsvoorwaarden van een zin een manier om de betekenis van die zin te geven. Wie weet hoe de wereld moet zijn om een zin waar te maken, weet een flink deel van wat de zin betekent.

    **Een voorbeeld uit het dagelijks leven.** *De winkel is zondag open* is waar of onwaar, afhankelijk van de winkel en de zondag. Om de zin te begrijpen hoef je niet te weten welke van de twee het is: je weet wat je moet nakijken.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Brenden Lake en Marco Baroni (2018) trainden neural networks om opdrachten zoals *walk*, *jump* en *walk twice* (loop, spring, loop twee keer) om te zetten in bewegingen. Na training op een willekeurige 80 van de 100 opdrachten deed het beste netwerk 99,8 van de 100 andere opdrachten goed. Daarna kwam *jump* in de training alleen als losse opdracht voor, terwijl de andere bewegingen ook in langere opdrachten voorkwamen, en testten ze alle langere opdrachten met *jump* erin, zoals *jump twice*. Hoeveel van de 100 deed het beste netwerk ongeveer goed?",
        options={
            "most": "Ongeveer 95 op de 100: bijna evenveel als eerst",
            "half": "Ongeveer 61 op de 100",
            "right": "Ongeveer 1 op de 100",
        },
        reveal="**Ongeveer 1 op de 100**: 1,2 voor het beste netwerk, en 0,08 voor het soort netwerk dat over al hun tests het best was. De netwerken kenden *jump* en kenden *twice*, maar zetten die twee niet samen zoals de betekenis van *jump twice* uit de woorden volgt. In dezelfde opzet met *turn left* (sla linksaf) op de plek van *jump* haalde het beste netwerk 90,3 van de 100. Met 8 langere opdrachten met *jump* in de training haalde het beste netwerk voor *jump* 38,3; met 16 en 32 waren dat 77,8 en 88,4.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een kleine wereld

    Om zinnen te controleren heb je een wereld nodig om ze aan te toetsen. Neem de kleine wereld van de afbeelding: vier dieren. Tom en Mia zijn katten, Rex is een hond en Pip een muis. Mia en Rex slapen. Tom achtervolgt Pip, en Rex achtervolgt Tom.

    Jurafsky en Martin noemen zo'n wereld een **model**: een formeel bouwsel dat een bepaalde stand van zaken in de wereld voorstelt. De dingen erin vormen het domein, de verzameling objecten die het model voorstelt. De woorden krijgen er hun betekenis uit, op drie manieren:

    - een naam staat voor één ding: *Mia* voor Mia;
    - een zelfstandig naamwoord of een werkwoord zonder lijdend voorwerp staat voor een lijst dingen: *kat* voor Tom en Mia, *slaapt* voor Mia en Rex;
    - een werkwoord met een lijdend voorwerp staat voor een lijst paren: *achtervolgt* voor (Tom, Pip) en (Rex, Tom), met wie achtervolgt voorop.

    In het boek staat een eigenschap voor een verzameling dingen en een relatie voor een verzameling tuples, rijtjes dingen zoals deze paren. Deze les gebruikt lijsten.
    """)
    return


@app.cell(hide_code=True)
def world_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_truth():
    mo.md(r"""
    ### Waar in een wereld

    *Mia slaapt* is waar in deze wereld, want Mia staat in de lijst van *slaapt*. *Tom slaapt* is onwaar: Tom staat er niet in. Tarski's *sneeuw is wit* werkt net zo: de zin is waar als, en alleen als, de wereld is zoals de zin zegt. Dat zijn de **waarheidsvoorwaarden** van een zin: wat er in een wereld moet gelden om de zin waar te maken.

    De waarheidsvoorwaarden blijven hetzelfde van de ene wereld naar de andere; of ze gelden, verandert. In een wereld waarin Tom ook slaapt, is *Tom slaapt* waar. Je kunt een zin begrijpen zonder te weten of de zin waar is: je weet waar je naar moet kijken.

    ### Elke en een

    *Elke kat slaapt* gaat over twee lijsten: de katten en de slapers. De zin is waar als elke kat bij de slapers hoort. *Er slaapt een kat* is waar als minstens één kat erbij hoort: *een* betekent hier minstens één. Jurafsky en Martin zeggen hetzelfde over de twee **kwantoren** van de logica, geschreven als ∀ (*elke*) en ∃ (*een*): bij ∃ moet minstens één invulling een ware zin opleveren, bij ∀ moeten alle invullingen dat doen. Een invulling zet één ding uit de wereld op een open plek, zoals de x in *x slaapt*.

    In deze wereld is *er slaapt een kat* waar (Mia) en *elke kat slaapt* onwaar (Tom is wakker). Zo controleer je ze allebei: maak de lijst van de katten die slapen, en kijk naar de lengte van die lijst.

    ### Het geheel uit de delen

    Hoe komt een programma van de woorden bij waar of onwaar? Via de boom. *Rex achtervolgt Tom* heeft een VP, *achtervolgt Tom*, en die VP heeft een eigen betekenis: de lijst van de dieren die Tom achtervolgen. Hier is dat alleen Rex. De zin is waar als het onderwerp, Rex, in die lijst staat. Elk deel krijgt een betekenis uit de delen eronder, van onder in de boom naar boven.

    Dat is het principe van **compositionaliteit**, in een gangbare formulering van de taalkundige Barbara Partee (1984): de betekenis van een samengestelde uitdrukking is een functie van de betekenis van de delen en van de manier waarop ze syntactisch gecombineerd zijn. De Stanford Encyclopedia of Philosophy noemt het een fundamentele vooronderstelling van het meeste hedendaagse werk in de semantiek. Het heet vaak het principe van Frege, naar de logicus Gottlob Frege. De filosoof Max Cresswell, van wie die naam komt (1973), schreef dat de naam niet wil zeggen dat Frege het principe uitdrukkelijk zo opschreef; of Frege het geloofde, is nog altijd een punt van discussie. De logicus Richard Montague bouwde rond 1970 de betekenis van Engelse zinnen op deze manier op, regel voor regel samen met de syntaxis. Tussen natuurlijke talen en de kunstmatige talen van logici bestaat volgens Montague geen belangrijk theoretisch verschil. Montague achtte het mogelijk om de syntaxis en de semantiek van beide soorten talen in één precieze theorie te beschrijven.
    """)
    return


@app.cell(hide_code=True)
def parts_cell():
    mo.Html(C.parts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **semantiek** | het deel van de taalkunde over betekenis, van woorden en van zinnen |
        | **model** | in de logica: een kleine wereld om zinnen aan te toetsen, met de dingen erin en wat er voor ze geldt |
        | **waarheidsvoorwaarden** | wat er in een wereld moet gelden om een zin waar te maken |
        | **kwantor** | een woord zoals *elke* of *een* dat zegt voor hoeveel dingen iets geldt; in de logica ∀ en ∃ |
        | **compositionaliteit** | de betekenis van een geheel komt uit de betekenis van de delen en de manier waarop ze gecombineerd zijn |
        | `"Mia" in group` | `True` als `"Mia"` een van de items van de lijst `group` is, en `False` als dat niet zo is |
        | `("Rex", "Tom")` | een paar, een tuple van twee: Rex achtervolgt Tom; wie achtervolgt, staat voorop |
        | `["S", ["NP", "Mia"], ["VP", "slaapt"]]` | een boom als lijst, nieuw in deze les: eerst het symbool, dan de delen; een woord is een string, zonder de tag (PROPN, VERB) |
        | `x[2][0]` | het item op plek 0 van het item op plek 2: één stap omlaag in de boom voor elk paar haken |
        """),
        L.watch_out("Een model is in deze les een wereld om zinnen aan te toetsen, zoals in de logica. Het is geen model dat van data leert, zoals elders in deze course. En *een* in *er slaapt een kat* betekent minstens één, zoals in de logica: de zin is ook waar als elke kat slaapt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf de wereld op: een lijst voor elk zelfstandig naamwoord en elk werkwoord zonder lijdend voorwerp, en een lijst paren voor elk werkwoord met een lijdend voorwerp.
    2. Geef elk woord een betekenis: een naam het ding, een zelfstandig naamwoord of een werkwoord de lijst.
    3. Werk de boom omhoog: een werkwoord met een lijdend voorwerp geeft de lijst van wie de handeling uitvoert, bij *achtervolgt Tom* wie Tom achtervolgt; een onderwerp met een VP geeft waar of onwaar.
    4. Maak voor *elke* en *een* de lijst van de dingen van het zelfstandig naamwoord die ook in de lijst van de VP staan: *een* heeft er één nodig, *elke* allemaal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel schrijft de wereld van de afbeelding als dict: bij elk woord de lijst. De tweede definieert `some`: de functie maakt de lijst `inside` van de dingen die ook in `group` zitten, en is waar als die lijst niet leeg is. Daarna controleert de cel vier zinnen.
    """)
    return


@app.cell
def world_cell():
    WORLD = {
        "kat": ["Tom", "Mia"],
        "hond": ["Rex"],
        "muis": ["Pip"],
        "slaapt": ["Mia", "Rex"],
        "achtervolgt": [("Tom", "Pip"), ("Rex", "Tom")],
    }
    return (WORLD,)


@app.cell
def example_look(WORLD):
    def some(things, group):
        inside = [thing for thing in things if thing in group]
        return len(inside) > 0

    print("Mia" in WORLD["slaapt"])              # Mia slaapt
    print("Tom" in WORLD["slaapt"])              # Tom slaapt
    print(some(WORLD["kat"], WORLD["slaapt"]))   # er slaapt een kat
    print(some(WORLD["hond"], WORLD["kat"]))      # er is een hond die een kat is
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Mia staat in de lijst van slapers, Tom niet: waar, dan onwaar. Bij *er slaapt een kat* bevat `inside` Mia, dus de lijst is niet leeg: waar. *Er is een hond die een kat is* gebruikt de lijst van katten op de plek van de VP: geen enkele hond staat erin, `inside` blijft leeg, en de zin is onwaar. De woorden werden lijsten, en de lijsten beslisten. Oefening 1 doet hetzelfde voor *elke*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_every():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. every: allemaal

        Schrijf `every(things, group)`: de functie geeft `True` als elk ding uit `things` ook in `group` zit, en anders `False`. *Elke kat slaapt* is dan `every(WORLD["kat"], WORLD["slaapt"])`. Vervang de `...`. De check roept `every(["Tom", "Mia"], ["Mia", "Rex"])` aan, daarna eigen gevallen, waarvan één met een lege eerste lijst.
        """),
        L.task("Beslis of alle dingen uit de eerste lijst in de tweede zitten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_every():
    box_every = L.code_box("every", lines=3)
    box_every
    return (box_every,)


@app.cell(hide_code=True)
def check_every(box_every):
    result_every = L.check("every", box_every.value)
    result_every
    return (result_every,)


@app.cell(hide_code=True)
def hints_every():
    L.hints(
        "every",
        [
            "Begin zoals `some`: maak de lijst `inside` van de dingen die ook in `group` zitten. Vergelijk dan twee lengtes.",
            "*Elke kat slaapt*: hoeveel van de katten moeten er in de lijst van slapers staan?",
            "`    return len(inside) == ___`",
        ],
        why="`inside = [thing for thing in things if thing in group]`, daarna `return len(inside) == len(things)`. `every([\"Tom\", \"Mia\"], [\"Mia\", \"Rex\"])` geeft `False`: alleen Mia zit in `inside`, en er zijn twee katten.",
    )
    return


@app.cell(hide_code=True)
def text_after_every():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op vier zinnen over de wereld, waaronder *elke eenhoorn slaapt*. Er zijn geen eenhoorns, dus de afbeelding geeft je functie `[]` als lijst van eenhoorns. Dan is `inside` ook leeg: 0 is gelijk aan 0, en de zin is waar. Dat verrast veel mensen, maar zo telt de logica: *elke eenhoorn slaapt* is pas onwaar als er een eenhoorn is die niet slaapt. Jurafsky en Martin laten hetzelfde zien met *alle vegetarische restaurants serveren vegetarisch eten*. In de logica staat daar: voor elke x geldt, als x een vegetarisch restaurant is, dan serveert x vegetarisch eten, met een pijl ⇒ voor *als ... dan*. Voor een restaurant dat niet vegetarisch is, is het *als*-deel onwaar, en dan telt de regel als waar. Ze voegen eraan toe dat die pijl maar losjes lijkt op wat we gewoonlijk onder gevolg of oorzaak verstaan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_doers():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. doers: de betekenis van een VP

        Een werkwoord met een lijdend voorwerp, zoals *achtervolgt Tom*, staat voor een lijst dingen: wie Tom achtervolgt. Schrijf `doers(pairs, target)`: geef de lijst terug van wie de handeling uitvoert op `target`, in de volgorde van `pairs`: bij de paren van *achtervolgt* wie `target` achtervolgt. *Achtervolgt Tom* is dan `doers(WORLD["achtervolgt"], "Tom")`, en dat geeft `["Rex"]`. Vervang de `...`. De check roept `doers([("Tom", "Pip"), ("Rex", "Tom")], "Tom")` aan, daarna eigen gevallen.
        """),
        L.task("Zoek wie de handeling uitvoert op target."),
    ])
    return


@app.cell(hide_code=True)
def exercise_doers():
    box_doers = L.code_box("doers", lines=6)
    box_doers
    return (box_doers,)


@app.cell(hide_code=True)
def check_doers(box_doers):
    result_doers = L.check("doers", box_doers.value)
    result_doers
    return (result_doers,)


@app.cell(hide_code=True)
def hints_doers():
    L.hints(
        "doers",
        [
            "Begin met een lege lijst `found`. Loop de paren langs met `for pair in pairs:`, en voeg wie de handeling uitvoert toe als het tweede item van een paar `target` is.",
            "Welk item in `(\"Rex\", \"Tom\")` is wie achtervolgt, en welk wie achtervolgd wordt?",
            "`            found.append(___)`",
        ],
        why="`found = []`; `for pair in pairs:` met daaronder `if pair[1] == target:` en `found.append(pair[0])`; na de loop `return found`. `doers([(\"Tom\", \"Pip\"), (\"Rex\", \"Tom\")], \"Tom\")` geeft `['Rex']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_doers():
    mo.md(r"""
    De afbeelding onder de check vraagt je functie wie elk dier achtervolgt, en geeft de dieren die ze vindt bij *achtervolgt Tom* een dikke rand. Het werkwoord alleen staat voor paren; met het lijdend voorwerp erbij staat de VP voor een lijst dingen, dezelfde soort betekenis als *slaapt*. Zo controleer je een zin met elk van de twee soorten VP op dezelfde manier: staat het onderwerp in de lijst? Oefening 3 zet dat in elkaar.
    """)
    return


@app.cell(hide_code=True)
def text_sentences():
    mo.md(r"""
    ### Oefening 3. is_true: van de boom naar waar of onwaar

    De cel hieronder bevat vijf zinnen als bomen, en een tweede wereld, `NIGHT`, met dezelfde dieren: 's nachts slapen Tom, Mia en Rex, en achtervolgt Pip Tom.
    """)
    return


@app.cell
def sentences_cell():
    SENTENCES = [
        ["S", ["NP", "Mia"], ["VP", "slaapt"]],
        ["S", ["NP", "Tom"], ["VP", "slaapt"]],
        ["S", ["NP", "Rex"], ["VP", "achtervolgt", ["NP", "Tom"]]],
        ["S", ["NP", "Tom"], ["VP", "achtervolgt", ["NP", "Rex"]]],
        ["S", ["NP", "Pip"], ["VP", "achtervolgt", ["NP", "Tom"]]],
    ]
    NIGHT = {
        "kat": ["Tom", "Mia"],
        "hond": ["Rex"],
        "muis": ["Pip"],
        "slaapt": ["Tom", "Mia", "Rex"],
        "achtervolgt": [("Pip", "Tom")],
    }
    return


@app.cell(hide_code=True)
def text_exercise_true():
    mo.vstack([
        mo.md(r"""
        Schrijf `is_true(tree, world)`: geef terug of de zin `tree` waar is in `world`. Zoek de lijst waar de VP voor staat: bij een VP zonder lijdend voorwerp de lijst van het werkwoord in `world`; bij een VP met een lijdend voorwerp de lijst van wie de handeling uitvoert op dat lijdend voorwerp, uit `doers`. Onderscheid de twee soorten VP aan de lengte. Geef dan terug of de naam van het onderwerp in die lijst staat. `doers` staat klaar voor je functie: de check geeft de `doers` van de course mee. Vervang de `...`. De check roept `is_true(SENTENCES[0], WORLD)` aan, daarna alle vijf de zinnen in `WORLD` en in `NIGHT`.
        """),
        L.task("Bouw de waarheid van de zin op uit de betekenis van de delen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_true():
    box_true = L.code_box("is_true", lines=7)
    box_true
    return (box_true,)


@app.cell(hide_code=True)
def check_true(box_true):
    result_true = L.check("is_true", box_true.value)
    result_true
    return (result_true,)


@app.cell(hide_code=True)
def hints_true():
    L.hints(
        "is_true",
        [
            "Neem `vp = tree[2]`. Met `if len(vp) == 2:` is de lijst `world[vp[1]]`; anders roep je `doers` aan met de paren van het werkwoord en de naam in het lijdend voorwerp. Bewaar de lijst in `group`.",
            "Op welke plekken staan in `SENTENCES[2]` de VP, het werkwoord, het lijdend voorwerp en de naam daarin?",
            "`        group = doers(world[vp[1]], ___)`",
        ],
        why="`vp = tree[2]`; `if len(vp) == 2:` dan `group = world[vp[1]]`, `else:` dan `group = doers(world[vp[1]], vp[2][1])`; aan het eind `return tree[1][1] in group`. `is_true(SENTENCES[2], WORLD)` geeft `True`, en `is_true(SENTENCES[2], NIGHT)` geeft `False`.",
    )
    return


@app.cell(hide_code=True)
def text_after_true():
    mo.md(r"""
    De afbeelding onder de check zet je antwoorden voor de vijf zinnen in beide werelden naast elkaar. De zinnen veranderden niet, je functie ook niet; alleen de wereld. *Tom slaapt* is onwaar in `WORLD` en waar in `NIGHT`, en *Rex achtervolgt Tom* net andersom. Dat zijn waarheidsvoorwaarden: een zin zegt wat er moet gelden, en elke wereld geeft antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: 64 werelden

    Zet de zes feiten aan en uit, en kijk welke zinnen waar zijn in jouw wereld. Kies daarna een zin: het rooster laat alle 64 werelden zien die je met de zes schakelaars kunt maken, gevuld waar de zin waar is. Vergelijk *Mia slaapt* met *Rex achtervolgt Tom*: allebei waar in 32 werelden, maar niet in dezelfde. En zoek twee zinnen die in precies dezelfde werelden waar zijn: met alleen deze vier dieren zeggen ze hetzelfde.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_facts = mo.ui.array([mo.ui.checkbox(value=on, label=text) for text, on in C.demo_facts(L.lang)])
    demo_sentence = mo.ui.dropdown(options=C.demo_options(L.lang), value=C.DEMO_LABELS[L.lang][2], label="Zin")
    mo.vstack([demo_facts.hstack(wrap=True, justify="start", gap=1), demo_sentence])
    return demo_facts, demo_sentence


@app.cell(hide_code=True)
def demo_look(demo_facts, demo_sentence):
    mo.Html(C.demo_picture(demo_facts.value, demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_idiom():
    mo.md(r"""
    ### Oefening 4. Waar de delen niet optellen

    Compositionaliteit werkt voor de meeste zinnen, maar misschien niet voor alle. De Stanford Encyclopedia of Philosophy geeft het Engelse *Mary kicked the bucket*, dat kan betekenen dat Mary doodging: er is geen voor de hand liggende manier om die betekenis af te leiden uit de betekenis van de delen. Zulke vaste uitdrukkingen lijken volgens de encyclopedie dus losse uitzonderingen op compositionaliteit. De encyclopedie noemt ook een schatting van de taalkundige Ray Jackendoff (1997): zo'n vijfentwintigduizend vaste uitdrukkingen in het Engels.
    """)
    return


@app.cell(hide_code=True)
def choice_idiom_cell():
    choice_idiom = L.choice(
        "idiom",
        question="Bij welke van deze zinnen volgt de betekenis niet uit de betekenis van de woorden en de manier waarop ze gecombineerd zijn?",
        options={
            "right": "*Het regent pijpenstelen.*",
            "rain": "*De katten en honden staan in de regen.*",
            "every": "*Elke kat slaapt als het regent.*",
            "chase": "*Een hond die een kat achtervolgt, rent de regen in.*",
        },
        correct="right",
        why={
            "right": "Een vaste uitdrukking: het betekent dat het hard regent, en er komt geen pijpensteel aan te pas. Een programma dat de betekenis uit de delen opbouwt, zou in de wereld naar vallende pijpenstelen zoeken. Een woordenboek geeft zo'n vaste uitdrukking dus als geheel, met een eigen betekenis.",
            "rain": "Hier tellen de delen wel op: *de katten en honden* staat voor de dieren, *in de regen* voor waar ze zijn, en de zin is waar als ze daar staan. Zoek een zin waarvan de betekenis niets met de woorden te maken heeft.",
            "every": "Deze zin is opgebouwd uit de delen, zoals in deze les: de katten, de slapers, en *elke*, met *als het regent* voor de momenten waarop het geldt. Zoek een zin waarvan de betekenis niets met de woorden te maken heeft.",
            "chase": "Een lange zin, maar opgebouwd uit de delen: *een hond die een kat achtervolgt* is een lijst honden, gemaakt van *hond* en *achtervolgt een kat*, en de rest zegt wat die hond doet. Zoek een zin waarvan de betekenis niets met de woorden te maken heeft.",
        },
    )
    choice_idiom
    return (choice_idiom,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl) gaf een zin een boom. Deze les gaf elk deel van die boom een betekenis in een kleine wereld: een naam een ding, een zelfstandig naamwoord of een werkwoord een lijst, een werkwoord met een lijdend voorwerp de lijst van wie de handeling uitvoert op dat lijdend voorwerp. Je schreef *elke*, vond de betekenis van een VP, en controleerde zinnen met hun boom in twee werelden. De volgende les laat zien wat dit weglaat: in de logica is *sommige katten slapen* ook waar als elke kat slaapt, maar een luisteraar hoort *niet allemaal*.

    **Een voorbeeld uit het dagelijks leven.** Een spreadsheet beantwoordt *elke factuur is betaald* op dezelfde manier: neem de facturen, neem de betaalde, en vergelijk. Zo bekeken is een vraag aan een database een zin die je tegen een wereld controleert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_idiom, result_doers, result_every, result_true):
    L.wrap_up(result_every, result_doers, result_true, choice_idiom)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
