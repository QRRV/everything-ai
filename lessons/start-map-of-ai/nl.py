# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-map-of-ai. Route: course/route/lessen/stam-1-basis.yaml. Checks en de getallen over de route: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-map-of-ai", "nl")
    C = L.checks   # de getallen over de route, geteld uit de routedata
    GPT = C.goal("doel-eigen-gpt")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over de vier cirkels.
    """)
    return


@app.cell(hide_code=True)
def quick_check_fraud():
    quick_fraud = L.choice(
        "quick-fraud",
        question="De fraudedetector van een bank leerde van miljoenen betalingen, waarbij telkens stond of het fraude was, met een neural network van veel layers. Hij zegt alleen \"verdacht\" of \"in orde\". Wat is de kleinste cirkel waar hij in past?",
        options={"ml": "Machine learning", "dl": "Deep learning", "genai": "Generatieve AI"},
        correct="dl",
        why={
            "ml": "Hij past in machine learning, maar er is een kleinere cirkel: het is een neural network met veel layers, dus deep learning. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "dl": "Hij leerde van voorbeelden, met een neural network van veel layers, en hij maakt niets nieuws. Dus is het deep learning.",
            "genai": "Hij zegt alleen \"verdacht\" of \"in orde\" over betalingen die er al zijn; hij maakt niets nieuws. Dus is het deep learning, geen generatieve AI. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
        },
    )
    quick_fraud
    return (quick_fraud,)


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="Welke volgorde klopt, van de grootste cirkel naar de kleinste?",
        options={
            "swapped": "AI, deep learning, machine learning, generatieve AI",
            "reversed": "Generatieve AI, deep learning, machine learning, AI",
            "right": "AI, machine learning, deep learning, generatieve AI",
        },
        correct="right",
        why={
            "right": "AI is de grootste cirkel, en generatieve AI de kleinste, binnen deep learning.",
            "swapped": "Deep learning is een deel van machine learning: machine learning met neural networks van veel layers. Machine learning is dus de grotere cirkel. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "reversed": "Dat is de volgorde van de kleinste cirkel naar de grootste. De grootste cirkel is AI.",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In een eerdere les heb je vier woorden geleerd voor delen van AI. Maar AI is veel meer dan die vier cirkels. Er horen ook programma's bij die een reis plannen of schaken, robots die leren dingen op te pakken, systemen die spraak omzetten in tekst, en wetten over wat AI wel en niet mag. Niemand leert dat allemaal tegelijk, en ook niet in een willekeurige volgorde: veel ervan bouwt voort op andere delen. Voordat je op weg gaat, heb je dus een kaart nodig: wat er is, wat er eerst komt, en welke kant je op kunt.

    **Een voorbeeld uit het dagelijks leven.** Leren koken. Iedereen begint met dezelfde basis: snijden, garen en op smaak brengen. Daarna kies je een richting, zoals brood bakken of de Italiaanse keuken, en in elke richting ga je van makkelijke naar moeilijke gerechten. Een goed kookboek begint niet met een soufflé, en laat je het hoofdstuk over brood overslaan als je alleen pasta wilt maken.
    """)
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "how-big",
        question="Hoeveel onderwerpen denk je dat het vakgebied AI heeft, van de basis tot het nieuwste onderzoek?",
        options={
            "50": "Ongeveer 50",
            "200": "Ongeveer 200",
            "600": "Meer dan 600",
        },
        reveal="**Meer dan 600.** Toen deze course werd opgezet, werd het hele vakgebied in kaart gebracht: 623 onderwerpen in 27 gebieden, van Python en wiskunde tot robots en recht, en elk onderwerp heeft weer kleinere delen. Voor één doel heeft niemand ze allemaal nodig. Daarom is de course gebouwd als een kaart, met routes erdoorheen.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zie de course als een boom. Dat is beeldspraak, maar het helpt: er is één stam, het **fundament**, en elk ander deel van AI groeit daaruit als een tak. Elke les heeft ook een niveau, en een route zegt in welke volgorde je de lessen doet.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **tak** | een deelgebied van de course, zoals taal of computer vision (AI voor beelden en video), met eigen lessen en een eigen project; elke tak groeit uit het **fundament**, de tak met wiskunde, Python, data, machine learning en deep learning die iedereen doet |
        | **niveau** | het label op een les dat zegt hoe gevorderd die is: **basis**, **gevorderd** of **expert** |
        | **route** | een volgorde van lessen door de course; de **hoofdroute** loopt van je eerste regel Python tot een eigen model dat tekst schrijft, en een wetenschappelijke paper die je zelf nabouwt; een **doelroute** bevat alleen de lessen waar één doel op voortbouwt |

        Drie voorbeelden, van elk één: robotica is een tak; *basis* is een niveau; en *Ik wil zo snel mogelijk mijn eigen GPT*, een doel op de startpagina, is een route.
        """),
        L.watch_out("In een spel is een level een stuk dat pas opengaat als je het vorige hebt uitgespeeld. Hier is een niveau alleen een label dat zegt hoe gevorderd een les is: geen enkele les zit op slot."),
    ])
    return


@app.cell(hide_code=True)
def text_branches():
    mo.md(rf"""
    De course deelt de 27 gebieden van AI in {C.N_BRANCHES} takken in. De hoofdroute gaat door drie ervan, op alle drie de niveaus: het fundament, taal en onderzoek. De figuur telt de lessen van de hoofdroute per niveau en per tak.
    """)
    return


@app.cell(hide_code=True)
def picture_route():
    mo.vstack([
        draw.route_grid(
            ["Basis", "Gevorderd", "Expert"],
            ["Fundament", "Taal", "Onderzoek", "Andere takken"],
            C.grid(), unit=("les", "lessen"), arrows=3,
            title="De hoofdroute: lessen per niveau en per tak"),
        mo.md(r"""
        Lees elke rij van links naar rechts, en ga dan een rij omlaag. Hoe donkerder een vak, hoe meer lessen erin zitten. De gestippelde kolom telt de paar lessen die de hoofdroute uit andere takken leent; die zitten ertussen, op de plek waar je ze nodig hebt. Op elk niveau ga je dus vooral van het fundament naar taal en dan naar onderzoek.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(rf"""
    Je hoeft nu dus geen tak te kiezen: de hoofdroute neemt je mee door het fundament, taal en onderzoek. De andere {C.N_BRANCHES - 3} takken, zoals computer vision en robotica, komen later; de startpagina laat ze allemaal zien bij *De hele kaart*. Ze bouwen allemaal voort op de hoofdroute, en voor de meeste van hun eerste lessen heb je alleen het niveau basis nodig.

    ### Recept

    Zo kies je je route. Loop de vragen in deze volgorde langs:

    1. **Nog geen eigen doel?** Volg de hoofdroute. Op de startpagina brengt *Ga verder* je naar je volgende les.
    2. **Een eigen doel?** Kies het op de startpagina bij *Of volg een doel*. Je ziet dan alleen de lessen waar dat doel op voortbouwt, in de volgorde van de hoofdroute.
    3. **Wil je dieper gaan?** Zet *Toon verdieping* aan. Een verdiepingsles heb je nooit nodig voor de lessen erna.
    4. **Een les verderop geopend?** Kijk bovenaan bij *Voorkennis*: daar staan de lessen waar hij op voortbouwt, met een vinkje bij elke les die je af hebt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(rf"""
    ### Uitgewerkt voorbeeld

    **Geval 1: nog geen doel.** Je wilt AI van het begin af leren, en je hebt nog geen eigen doel. Vraag 1 zegt genoeg: volg de hoofdroute, en *Ga verder* op de startpagina brengt je naar je volgende les. Vraag 3 en 4 helpen je onderweg.

    **Geval 2: een doel.** Je wilt zo snel mogelijk een GPT bouwen: een model dat tekst schrijft. Die bouw je zelf, op niveau gevorderd.

    1. **Nog geen eigen doel?** Je hebt er wel een, dus door naar vraag 2.
    2. **Een eigen doel?** Ja. Op de startpagina kies je *Ik wil zo snel mogelijk mijn eigen GPT*. De startpagina telt de lessen waar dit doel op voortbouwt: **{GPT["lessons"]} lessen**, in plaats van de {GPT["place"]} lessen tot en met de GPT in de hoofdroute (met *Toon verdieping* uit). Kijk welke dat zijn:
        - **Niveaus:** {GPT["levels"]["basis"]} op basis en {GPT["levels"]["gevorderd"]} op gevorderd. Geen enkele op expert: die heb je voor een eerste GPT niet nodig.
        - **Takken:** {GPT["foundation"]} uit het fundament, en de rest vooral uit de tak taal.
    3. **Wil je dieper gaan?** Nu niet: je wilt snel je GPT. Laat *Toon verdieping* uit staan.
    4. **Een les verderop geopend?** Niet nodig: de doelroute geeft je elke les op volgorde.

    De doelroute slaat dus {GPT["place"] - GPT["lessons"]} lessen over waar een eerste GPT niet op voortbouwt.
    """)
    return


@app.cell(hide_code=True)
def text_pick_route():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Kies een route

    Gebruik het recept, zoals in geval 2 van het uitgewerkte voorbeeld.
    """)
    return


@app.cell(hide_code=True)
def choice_pick_route_cell():
    choice_pick_route = L.choice(
        "pick-route",
        question="Sam wil zo snel mogelijk attention snappen, een idee dat je later in de course tegenkomt. Wat kan Sam het best doen?",
        options={
            "jump": "Meteen de les over attention openen.",
            "goal": "Op de startpagina *Ik wil attention snappen* kiezen, en die route volgen.",
            "all": "Eerst de hele hoofdroute doen, tot aan attention.",
        },
        correct="goal",
        why={
            "goal": "Die route bevat precies de lessen waar attention op voortbouwt, in de goede volgorde, en verder niets.",
            "jump": "Die les bouwt voort op veel lessen ervoor: bij *Voorkennis* zou je ze zien staan, zonder vinkjes. Kijk nog eens naar vraag 2 van het recept.",
            "all": "Dat kan, maar het duurt langer dan nodig: vóór attention heeft de hoofdroute ook lessen waar attention niet op voortbouwt. Kijk nog eens naar vraag 2 van het recept.",
        },
    )
    choice_pick_route
    return (choice_pick_route,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tak, niveau of route?

        Elk onderdeel hieronder is een tak, een niveau of een route, net als de drie voorbeelden onder de tabel met nieuwe woorden.
        """),
        L.task("Zet elk onderdeel in de goede groep."),
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
            "Een tak is een deel van AI, een niveau zegt hoe gevorderd een les is, en een route is een volgorde van lessen.",
            "Twee onderdelen zijn labels op lessen: dat zijn de niveaus. Welke twee onderdelen zijn lijsten met lessen die je na elkaar doet?",
            "Computer vision en spraak zijn delen van AI, elk met eigen lessen.",
        ],
        why="Tak: **Computer vision** en **Spraak, audio en muziek**. Niveau: **Gevorderd** en **Expert**. Route: **Ik wil attention snappen** en **De hoofdroute**.",
    )
    return


@app.cell(hide_code=True)
def text_foundation():
    mo.md(r"""
    ### Oefening 3. Hoe de takken samenhangen
    """)
    return


@app.cell(hide_code=True)
def choice_foundation_cell():
    choice_foundation = L.choice(
        "foundation",
        question="Welke zin over de takken klopt?",
        options={
            "separate": "Elke tak is een eigen course: je kunt in elke tak beginnen zonder voorkennis.",
            "levels": "Er zijn drie takken: basis, gevorderd en expert.",
            "grow": "Elke tak groeit uit het fundament: de lessen ervan bouwen voort op lessen uit het fundament.",
        },
        correct="grow",
        why={
            "grow": "Elk deel van AI gebruikt dezelfde basis, dus elke tak begint waar het fundament je gebracht heeft.",
            "separate": "Een eerste les over computer vision slaat bijvoorbeeld een plaatje op als getallen, met Python; zonder het fundament kun je die les niet volgen. Kijk nog eens naar de tabel met nieuwe woorden.",
            "levels": "Basis, gevorderd en expert zijn labels die zeggen hoe gevorderd een les is. Kijk nog eens naar de tabel met nieuwe woorden: wat is een tak?",
        },
    )
    choice_foundation
    return (choice_foundation,)


@app.cell(hide_code=True)
def text_further():
    mo.md(r"""
    ### Verder

    Niet nodig voor deze les, maar leuk om te zien: kies op de startpagina bij *Of volg een doel* voor *Ik wil mijn eerste CNN trainen*. Een CNN is een neural network voor beelden, dat je op niveau gevorderd bouwt. Op welke niveaus liggen de lessen van die route? Kies daarna weer *De hele route*, dan krijg je de hoofdroute terug.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) leerde je vier cirkels kennen; die zijn maar een klein deel van de kaart. Vanaf nu staat elke les ergens op deze kaart: bovenaan een les zie je de plek in de route en het niveau, en de startpagina laat de hele route zien. De volgende les hoort ook nog bij de start: hoe je AI-tools zoals een chatassistent gebruikt, met open ogen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_foundation, choice_pick_route, result_kinds):
    L.wrap_up(choice_pick_route, result_kinds, choice_foundation)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
