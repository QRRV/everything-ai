# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-figures. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-figures", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) en [Is het verschil echt?](/course/open/res-significance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="Welke regel zegt wat de y-as van een grafiek laat zien?",
        options={
            "title": "`plt.title(\"validation loss\")`",
            "right": "`plt.ylabel(\"validation loss\")`",
            "xlabel": "`plt.xlabel(\"validation loss\")`",
        },
        correct="right",
        why={
            "title": "De titel staat boven de grafiek en zegt waar de hele grafiek over gaat. Zie [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "xlabel": "`xlabel` benoemt de as langs de onderkant. Zie [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "right": "`ylabel` zet zijn tekst langs de y-as, de as die omhoog gaat.",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_small():
    quick_small = L.choice(
        "quick-small",
        question="Een verandering verlaagt de loss met 0,006, en de toets geeft p = 0,02. Wat zegt dat over de grootte van de verandering?",
        options={
            "big": "Hij is groot: hoe kleiner p, hoe groter de verandering",
            "right": "Niets: p zegt hoe goed toeval hem verklaart",
            "percent": "Dat de loss ongeveer 2% lager is, zoals p = 0,02 zegt",
        },
        correct="right",
        why={
            "big": "p zegt hoe slecht toeval alleen het verschil verklaart, niet hoe groot het is: met meer seeds krijgt hetzelfde verschil een kleinere p. Zie [Is het verschil echt?](/course/open/res-significance?lang=nl).",
            "percent": "0,02 is de p-waarde, niet de grootte van de verandering. Zie [Is het verschil echt?](/course/open/res-significance?lang=nl).",
            "right": "De grootte is die 0,006 zelf, naast wat de loss verder nog verschuift. Een figuur kan allebei tegelijk laten zien.",
        },
    )
    quick_small
    return (quick_small,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Lezers kijken vaak naar de figuren van een paper voordat ze de tekst lezen, en sommigen lezen verder niets. Een figuur is je resultaat in de vorm die de meeste mensen onthouden. Een duidelijke figuur laat in een paar seconden zien wat je vond; een slordige laat iets anders zien, zonder één fout getal erin.

    **Een voorbeeld uit het dagelijks leven.** Een weerkaart laat het weer van morgen in één oogopslag zien: regen in het noorden, zon in het zuiden. Dezelfde verwachting als tabel met getallen per plaats zegt net zoveel, maar niemand ziet het in één oogopslag. En een kaart die 18 graden donkerrood kleurt, laat een zachte dag op een hittegolf lijken, terwijl elk getal klopt.
    """)
    return


@app.cell(hide_code=True)
def predict_bars():
    guess_bars = L.guess(
        "bars",
        question="De eerste grafiek bij Het idee hieronder tekent de runs van de course met en zonder warmup als twee staven, elk het gemiddelde van tien seeds. De staaf zonder warmup is ongeveer half zo hoog. Hoeveel lager is zijn loss echt?",
        options={
            "half": "Ongeveer de helft",
            "tenth": "Ongeveer een tiende",
            "right": "Minder dan 1%",
        },
        reveal="**Minder dan 1%.** De gemiddelden zijn 2,402 met warmup en 2,396 zonder: 0,006 verschil, een kwart procent. De y-as begint bij 2,390, niet bij 0, dus laten de staven alleen hun toppen zien, en die verschillen een factor twee. En elke staaf verbergt de tien runs erachter.",
    )
    guess_bars
    return (guess_bars,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een duidelijke figuur heeft **één boodschap**, en alles erin dient die boodschap. Zeg de boodschap in de titel, of in de eerste zin van het bijschrift, de tekst onder een figuur die je in [Een abstract, een figuur en een resultatentabel lezen](/course/open/res-figure-and-table?lang=nl) als eerste las. Benoem elke as met wat hij laat zien, en met zijn eenheid als hij er een heeft; staat er bij elke groep een naam langs de as, dan zeggen die namen het. Laat de spreiding over seeds zien: elke run als punt, of de spreiding rond het gemiddelde, niet alleen het gemiddelde. Zet wat je vergelijkt naast elkaar, op één as met één schaal, zodat het oog lengtes of hoogtes direct vergelijkt. Een staaf lees je af aan zijn lengte, dus een staafdiagram begint bij 0; punten en lijnen mogen hoger beginnen, als de as zegt waar.

        Benoem elke groep, langs de as of in een **legenda**, een lijstje in de grafiek dat zegt welk teken welke groep is: geef in Matplotlib elke `plt.plot` of `plt.scatter` een `label="..."` en roep `plt.legend()` aan. En gebruik kleur alleen als die iets betekent, zoals één kleur per instelling.

        De plaatjes: twee keer dezelfde runs. Eerst als twee staven van de gemiddelden op een as die bij 2,390 begint: het verschil lijkt enorm, en de runs zijn weg. Daarna elke run als punt, de twee instellingen naast elkaar op één as, met het gemiddelde als kort streepje: de groepen overlappen, en zonder warmup liggen ze iets lager.
        """),
        mo.Html(C.before_after({"bad": "Misleidend", "good": "Duidelijk", "y": "validation loss", "with": "met warmup", "without": "zonder", "lang": L.lang,
                                "alt_bad": "Twee staven, de gemiddelden met warmup, 2,402, en zonder, 2,396, op een y-as van 2,390 tot 2,404: de staaf zonder warmup is half zo hoog.",
                                "alt_good": "Dezelfde runs als punten, tien met warmup van 2,386 tot 2,431 en tien zonder van 2,382 tot 2,424, naast elkaar op één as van 2,365 tot 2,440, elk met zijn gemiddelde als kort streepje. De twee groepen overlappen."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **legenda** | het lijstje in een grafiek dat zegt welk teken of welke kleur bij welke groep hoort |
        """),
        L.watch_out("Een as die niet bij 0 begint, is prima voor punten en lijnen, als de getallen erbij staan. Bij staven misleidt het, omdat een lezer hun lengtes vergelijkt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo maak je een figuur die iets laat zien:

    1. Kies zijn ene boodschap, en schrijf die als titel of als eerste zin van het bijschrift.
    2. Benoem elke as met wat hij laat zien, en zijn eenheid als hij er een heeft; namen langs een as benoemen hem al.
    3. Laat elke run zien, of de spreiding over seeds, niet alleen het gemiddelde.
    4. Zet wat je vergelijkt naast elkaar, op één as met één schaal; staven beginnen bij 0.
    5. Benoem elke groep, op de as of in een legenda; gebruik kleur alleen als die iets betekent.
    6. Schrijf een bijschrift: wat elk teken is, hoeveel seeds, en de boodschap.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het resultaat van [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl) als figuur: de gemiddelde validation loss van de GPT met 2 en met 4 layers bij negen learning rates. `"o--"` tekent punten die met een gestippelde lijn verbonden zijn, `"o-"` met een doorgetrokken lijn. Het commentaar zegt welke stap elk deel is. Hij is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_figure():
    import matplotlib.pyplot as plt

    _rates = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    _two = [2.400, 2.215, 2.089, 1.962, 1.836, 1.791, 1.779, 1.801, 2.191]
    _four = [2.332, 2.124, 2.005, 1.897, 1.805, 1.784, 1.777, 1.812, 2.787]
    # 1. de ene boodschap, als titel
    plt.title("Op dezelfde manier getuned eindigen 4 layers waar 2 layers eindigen")
    # 2. elke as zegt wat hij laat zien; de rates groeien met factoren, dus een logaritmische schaal
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("validation loss na 6 epochs, gemiddelde van 3 seeds")
    # 4-5. beide modellen op één as, elke lijn met een naam
    plt.plot(_rates, _two, "o--", label="2 layers")
    plt.plot(_rates, _four, "o-", label="4 layers")
    # maak af: plt.legend(), zodat de lezer ziet welke lijn welke is
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zonder `plt.legend()` ziet de lezer twee lijnen en weet hij niet welk model welk is: de labels zijn er, maar niets laat ze zien. Met de legenda noemt een lijstje in een hoek de gestippelde en de doorgetrokken lijn. Stap 3 ontbreekt hier: elk punt is het gemiddelde van drie seeds, en het plaatje in Eerlijke vergelijkingen tekende ook de losse seeds.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_points():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elke run, naast elkaar

        Teken de runs van de course met en zonder warmup, seeds 0 tot en met 9, als een figuur die iets laat zien. Elke run is een punt; de runs met warmup staan bij de naam `"with warmup"` langs de x-as, die zonder bij `"without warmup"`. `["with warmup"] * len(with_warmup)` is die naam één keer voor elke run, zoals `*` een lijst herhaalt, dus `plt.scatter(["with warmup"] * len(with_warmup), with_warmup)` zet heel `with_warmup` bij die naam. Vervang de `...`, en vul een titel in met de ene boodschap en een label voor de y-as. Een gemiddelde dat je er als punt overheen tekent, mag blijven staan.
        """),
        L.task("Teken elke run van `with_warmup` en `without_warmup` als punt, elke instelling bij een eigen naam langs de x-as, met een titel en een label op de y-as."),
    ])
    return


@app.cell(hide_code=True)
def exercise_points():
    box_points = L.code_box("seed-points", lines=10)
    box_points
    return (box_points,)


@app.cell(hide_code=True)
def check_points(box_points):
    result_points = L.check("seed-points", box_points.value)
    result_points
    return (result_points,)


@app.cell(hide_code=True)
def hints_points():
    L.hints(
        "seed-points",
        [
            "Twee regels met `plt.scatter`, één per instelling. De x-waarden zijn de naam van de instelling, zo vaak als er runs zijn, zodat alle runs van een instelling boven die naam staan.",
            "`plt.scatter([\"with warmup\"] * len(with_warmup), ___)` en `plt.scatter([___] * len(without_warmup), without_warmup)`; daarna een titel zoals `\"Zonder warmup eindigen de runs iets lager\"`.",
        ],
        why="```python\nplt.scatter([\"with warmup\"] * len(with_warmup), with_warmup)\nplt.scatter([\"without warmup\"] * len(without_warmup), without_warmup)\nplt.title(\"Zonder warmup eindigen de runs iets lager\")\nplt.ylabel(\"validation loss na 6 epochs\")\nplt.show()\n```\n\nElke instelling krijgt een eigen naam langs de x-as, en elke run een eigen punt erboven, zodat de lezer de spreiding van elke instelling ziet en de twee naast elkaar op één schaal.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Welke figuur is eerlijk?

        Drie figuren van dezelfde runs: de GPT met 2 en met 4 layers bij hun beste learning rate, 0,03, seeds 0 tot en met 2.
        """),
        mo.Html(C.three_ways({"two": "2 layers", "four": "4 layers", "y": "validation loss", "lang": L.lang,
                              "alt_a": "A: twee staven van de gemiddelden, 1,779 en 1,777, op een y-as van 1,776 tot 1,780; de staaf van 4 layers is een vijfde zo hoog.",
                              "alt_b": "B: twee losse panelen, elk met een eigen y-as, met de drie runs van 2 layers en van 4 layers; beide lijken even ver te spreiden.",
                              "alt_c": "C: de drie runs van elk model als punten naast elkaar op één y-as van 1,765 tot 1,795, met de gemiddelden als korte streepjes; de twee groepen overlappen."})),
    ])
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair-figure",
        question="Welke figuur laat eerlijk zien hoe de GPT met 2 en die met 4 layers zich verhouden?",
        options={
            "a": "A: de staven maken het verschil makkelijk te zien",
            "b": "B: elk model krijgt een eigen paneel, dus geen van beide verbergt de ander",
            "right": "C: elke run op één as, naast elkaar",
        },
        correct="right",
        why={
            "a": "De staven beginnen bij 1,776, dus lijkt een verschil van 0,002 een factor vijf, en de drie runs achter elk gemiddelde zijn weg.",
            "b": "Elk paneel heeft een eigen y-as, dus kan het oog de twee niveaus helemaal niet vergelijken: dezelfde hoogte betekent in elk paneel een andere loss. En de runs van 4 layers lijken even ver te spreiden als die van 2 layers, terwijl ze vier keer zo dicht bij elkaar liggen.",
            "right": "Eén as, één schaal, elke run: de lezer ziet dat de twee groepen overlappen, en dat is de eerlijke boodschap.",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_caption():
    mo.md(r"""
    ### Oefening 3. Een bijschrift
    """)
    return


@app.cell(hide_code=True)
def explain_caption():
    explain_caption_cell = L.explain(
        "caption",
        question="Schrijf het bijschrift bij je grafiek van oefening 1, zoals het onder een figuur in een paper zou staan: wat elk punt is, de opzet en de seeds, en de boodschap in één zin.",
        model_answer="**Figuur 1. Zonder warmup eindigen de runs iets lager.** Validation loss na 6 epochs van de kleine GPT uit Eén ding tegelijk veranderen, op het eerste kwart van Tiny Shakespeare, met een warmup van 100 stappen en zonder; elk punt is één run, seeds 0 tot en met 9 voor beide instellingen. Zonder warmup is de loss gemiddeld 0,006 lager, en lager bij elk van de tien seeds, wat klein is naast de spreiding van ongeveer 0,04 tussen seeds.",
        points=[
            "je zegt de boodschap in één zin, aan het begin",
            "je zegt wat elk punt is, en de opzet met zijn seeds",
            "je geeft de grootte van het verschil naast de spreiding, zodat een lezer hem kan beoordelen",
        ],
    )
    explain_caption_cell
    return (explain_caption_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De grafieken zelf komen uit [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl); de runs en hun spreiding uit [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl) en [Is het verschil echt?](/course/open/res-significance?lang=nl). Deze les sluit het deel over experimenten af: je kunt nu één ding veranderen, een deel weghalen, herhalen met seeds, een logboek bijhouden, eerlijk vergelijken, een verschil toetsen en het laten zien. Verderop in de route lees je papers nauwkeuriger en bouw je er zelf een na; de figuren daarvan volgen dit recept. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, explain_caption_cell, result_points):
    L.wrap_up(result_points, choice_fair, explain_caption_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
