# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-no-free-lunch. Route: course/route/lessen/klassiek-ml.yaml. Checks en beelden: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-no-free-lunch", "nl")
    C = L.checks   # vijf verzonnen datasets, de vier modellen van de module met hun scores, en de kleine wereld


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl) en [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tree():
    quick_tree = L.choice(
        "quick-tree",
        question="Een beslisboom met diepte 2 op twee features verdeelt het vlak in delen. Hoe zien de randen van die delen eruit?",
        options={
            "right": "Rechte stukken, elk evenwijdig aan een van de assen",
            "diagonal": "Rechte lijnen in elke richting",
            "curves": "Krommen die om de voorbeelden heen buigen",
        },
        correct="right",
        why={
            "diagonal": "Elke vraag van een boom kijkt naar één feature en één drempel, dus elke rand loopt recht dwars op die ene feature. Zie *Het idee* in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
            "curves": "Een boom stelt alleen drempelvragen over één feature tegelijk; zijn randen zijn recht, en evenwijdig aan de assen. Zie *Het idee* in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
            "right": "Eén feature en één drempel per vraag: de delen zijn rechthoeken. Vandaag blijkt dat een aanname over de data te zijn.",
        },
    )
    quick_tree
    return (quick_tree,)


@app.cell(hide_code=True)
def quick_check_cv():
    quick_cv = L.choice(
        "quick-cv",
        question="Hoe vaak zit elk voorbeeld uit de training set bij 5-fold cross-validation in het deel dat een score krijgt?",
        options={
            "five": "Vijf keer, één keer voor elke fold",
            "right": "Eén keer",
            "never": "Nooit: het wordt alleen gebruikt om te trainen",
        },
        correct="right",
        why={
            "five": "Elke fold krijgt één keer een score, en elk voorbeeld zit in precies één fold: in de andere vier rondes wordt erop getraind. Zie *Het idee* in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "never": "Elke fold is om de beurt het deel met de score, dus elk voorbeeld krijgt één keer een score, van een model dat er niet op trainde. Zie *Het idee* in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "right": "Eén keer, van een model dat het niet zag. Vandaag vergelijk je zo vier modellen op vijf datasets.",
        },
    )
    quick_cv
    return (quick_cv,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je kent nu vier classifiers die elk een ander soort grens trekken: k-NN, de nearest centroid, de breedste straat en de beslisboom. Welke is de beste? Het zou veel werk schelen als je dat eens en voor altijd wist. Probeer ze op een paar verzonnen datasets, en er komt niet één bovenaan. Dat is geen pech: het is een stelling.

    **Een voorbeeld uit het dagelijks leven.** Winterbanden hebben de meeste grip op sneeuw en slijten snel op een hete weg; zomerbanden net andersom. Geen band is de beste op elke weg, want wat een band goed maakt op de ene weg, maakt hem slechter op een andere.
    """)
    return


@app.cell(hide_code=True)
def predict_winner():
    guess_winner = L.guess(
        "winner",
        question="Vier verzonnen datasets met twee features: twee ronde klonten, twee klassen aan weerszijden van een rechte lijn, een klont binnen een ring, en blokken. Welk van de vier modellen is op alle vier het beste?",
        options={
            "knn": "k-NN: het kan elke grens trekken",
            "tree": "De beslisboom: hij kan het vlak in zoveel rechthoeken knippen als nodig is",
            "right": "Geen van allen",
        },
        reveal="**Geen.** Met 5-fold cross-validation is de nearest centroid het beste op de twee ronde klonten (0,823), de breedste straat op de rechte lijn (0,997), k-NN op de ring (1,000) en de beslisboom op de blokken (0,993). Hieronder zie je ze allemaal.",
    )
    guess_winner
    return (guess_winner,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Wat een model aanneemt.** Voorbeelden uit de training set zeggen alleen wat goed is op de plek waar ze liggen. Overal elders moet een model gokken, en hoe het gokt is wat het over de data aanneemt:

    - k-NN: een voorbeeld heeft de klasse van de voorbeelden in de buurt;
    - de nearest centroid: elke klasse is één ronde klont;
    - de breedste straat: een rechte lijn scheidt de klassen;
    - een beslisboom: drempels op één feature tegelijk scheiden de klassen.

    Dit heet de inductive bias van het model. Waar de aanname klopt, leert een model veel van weinig voorbeelden; waar ze niet klopt, helpen meer voorbeelden het weinig.

    **No free lunch.** Is er een model dat niets aanneemt, en daardoor overal wint? Neem een kleine wereld: drie features die elk 0 of 1 zijn, dus 8 mogelijke voorbeelden. De training set laat het label van 5 ervan zien; 3 blijven onbekend.

    | voorbeeld | feature 1 | feature 2 | feature 3 | label |
    |---|---|---|---|---|
    | 1 | 0 | 0 | 0 | 0 |
    | 2 | 0 | 0 | 1 | 0 |
    | 3 | 0 | 1 | 0 | 0 |
    | 4 | 1 | 0 | 0 | 1 |
    | 5 | 1 | 1 | 0 | 1 |
    | 6 | 0 | 1 | 1 | ? |
    | 7 | 1 | 0 | 1 | ? |
    | 8 | 1 | 1 | 1 | ? |

    Elk onbekend voorbeeld kan 0 of 1 zijn: dat geeft $2 \cdot 2 \cdot 2 = 8$ werelden, en elk ervan past perfect bij de training set. Wat een regel ook zegt voor de 3 onbekende voorbeelden, voor elk ervan is de helft van de werelden het ermee eens en de andere helft niet. Dus gemiddeld over de 8 werelden heeft elke regel er precies de helft goed: het beste model net zo goed als een munt. Dat is de **no-free-lunch-stelling**: gemiddeld over alle mogelijke problemen doet geen enkele leermethode het beter op onbekende voorbeelden dan een andere. Een model wint alleen op problemen die bij zijn aanname passen, en de problemen die we tegenkomen zijn niet alle mogelijke problemen: ze hebben structuur.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **no-free-lunch-stelling** | gemiddeld over alle mogelijke problemen doet geen enkele leermethode het beter op onbekende voorbeelden dan een andere | |
    | **wereld** (in deze les) | één manier om de onbekende voorbeelden een label te geven die bij de training set past | `[0, 1, 1]` |
    | `+` tussen lijsten | plakt twee lijsten aan elkaar tot een nieuwe: `[0, 1] + [1]` is `[0, 1, 1]`, en `[0, 1]` blijft zoals hij was | `world + [0]` |
    """)
    return


@app.cell(hide_code=True)
def text_watch_out():
    L.watch_out("No free lunch zegt niet dat alle modellen op jouw probleem even goed zijn. Het zegt het omgekeerde: welk model wint, hangt af van het probleem, dus moet je ze vergelijken op je eigen data, met cross-validation, in plaats van op een favoriet te vertrouwen.")
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Kies een dataset met de slider: 1 twee ronde klonten, 2 een rechte lijn, 3 een ring, 4 blokken, 5 muntworpen, waar de klasse niets met de features te maken heeft. Elk heeft 300 verzonnen voorbeelden met twee features, klasse 0 gevuld en klasse 1 open. De vier grafieken tonen de <span class="c-pred">grens</span> die elk model trekt na training op alle 300, met erboven de accuracy van het model met 5-fold cross-validation; ★ markeert de beste. k-NN kijkt naar 5 buren, de breedste straat heeft C = 1 en de boom is hooguit 4 diep.
    """)
    return


@app.cell(hide_code=True)
def explore_dataset_cell():
    def _frame(_v):
        _d = C.DATASETS[_v - 1]
        _s = C.scores()[_d]
        _alt = (f"Dataset {_v}, {C.NAMES[_d][L.lang]}, vier keer, elk met de grens van één model. Accuracy met "
                f"cross-validation: " + ", ".join(f"{C.MODEL_NAMES[_m][L.lang]} {C.number(_s[_m], L.lang)}" for _m in C.MODELS) + ".")
        _said = C.verdict(_v, L.lang)
        return C.models_picture(_d, {"x": "feature 1", "y": "feature 2", "alt": _alt}, L.lang), _said

    explore_dataset = L.explore("dataset", "dataset", _frame, start=1, values=[1, 2, 3, 4, 5], role="data",
                                task="Zoek voor elk model de dataset waarop het het verst achter het beste blijft.")
    explore_dataset
    return (explore_dataset,)


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Op de twee ronde klonten trekken de nearest centroid (0,823) en de breedste straat (0,817) bijna dezelfde rechte lijn, en dat is wat twee ronde klonten vragen; k-NN (0,743) en de boom (0,727) jagen de voorbeelden na waar de klonten overlappen, en verliezen. Op de rechte lijn haalt de breedste straat 0,997 en k-NN 0,993; de nearest centroid, 0,763, legt zijn lijn halverwege de twee gemiddelden, dwars over de banden. Geen rechte lijn past bij de ring: de nearest centroid haalt 0,567 en de breedste straat 0,650, k-NN 1,000 en de boom 0,990 met een rechthoek. Op de blokken vindt de boom de twee drempels, 0,993. Op de muntworpen zitten alle vier dicht bij 0,5: de breedste straat geeft het op en zegt overal klasse 1, en 163 van de 300 voorbeelden zijn klasse 1, dus 0,543; k-NN komt zelfs onder 0,5 uit, op 0,427. De twee flexibele modellen, k-NN en de boom, raken op deze vijf nooit meer dan ongeveer 0,1 achter: ze kunnen hun grens om de ring en de blokken heen buigen, waar een rechte lijn dat niet kan. Ze betalen daarvoor op de klonten, waar ze de overlap najagen, en k-NN betaalt bij elke vraag, als lazy learner. Dit is één trekking van de datasets: bij een andere kan een verschil van een paar honderdsten omdraaien; een verschil van 0,2 niet.

    ### Recept

    Zo kies je een model zonder free lunch:

    1. Vraag wat elk model aanneemt: voorbeelden in de buurt lijken op elkaar, één ronde klont per klasse, een rechte lijn, drempels op één feature tegelijk.
    2. Vraag wat je over je data weet: teken ze als dat kan, en bedenk waar ze vandaan komen.
    3. Vergelijk de modellen waarvan de aanname zou kunnen passen, met cross-validation op de training data.
    4. Vergelijk bij een nieuw probleem opnieuw: de vorige winnaar bewijst daar niets.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De kleine wereld uit *Het idee*. De cel maakt de 8 werelden van de 3 onbekende voorbeelden, neemt twee regels die allebei bij de 5 voorbeelden uit de training set passen, en telt hoeveel onbekende voorbeelden elke regel in elke wereld goed heeft. De opmerkingen met `#` zeggen welke stap elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_world():
    # de kleine wereld: de onbekende voorbeelden zijn 6 (0, 1, 1), 7 (1, 0, 1) en 8 (1, 1, 1)

    # 1. elke manier om de 3 onbekende voorbeelden een label te geven: 8 werelden, elk past bij de training set
    worlds = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                worlds.append([a, b, c])
    print(len(worlds))

    # 2. twee regels die bij alle 5 voorbeelden uit de training set passen, en wat ze zeggen voor de onbekende
    first = [0, 1, 1]   # het label is feature 1
    other = [1, 1, 0]   # feature 1, omgedraaid als feature 2 en 3 allebei 1 zijn

    # 3. hoeveel onbekende voorbeelden elke regel goed heeft, in elke wereld
    sum_first = 0
    sum_other = 0
    for world in worlds:
        right_first = 0
        right_other = 0
        for i in range(3):
            if first[i] == world[i]:
                right_first = right_first + 1
            if other[i] == world[i]:
                right_other = right_other + 1
        print(world, right_first, right_other)
        sum_first = sum_first + right_first
        sum_other = sum_other + right_other
    print(sum_first, sum_other)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    8 werelden. In elke wereld scoren de regels anders: in de wereld `[0, 1, 1]` heeft de eerste regel alle 3 goed en de andere 1, in `[1, 1, 0]` is het andersom, en in `[1, 0, 0]` heeft de eerste er geen enkele goed. Maar over de 8 werelden hebben ze er allebei 12 van de 24 goed: de helft. Maak van een van de regels drie andere labels, en de som blijft 12. Alleen als je meer over de wereld weet, zoals "het label is altijd feature 1", kan de ene regel het beter doen dan de andere. Daarom begint het recept bij aannames, en daarom geldt stap 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_worlds():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elke wereld

        `all_worlds(n)` moet elke wereld geven voor `n` onbekende voorbeelden: elke lijst van `n` labels die elk 0 of 1 zijn, zoals stap 1 van het uitgewerkte voorbeeld, maar voor elke `n`. Drie geneste lussen werken alleen voor 3, dus deze functie laat de werelden één voorbeeld tegelijk groeien: elke wereld tot nu toe geeft er twee die langer zijn, een met een 0 erachter en een met een 1. Vervang de `...`. De check roept `all_worlds(3)` aan, en probeert 1, 2, 4 en 0 voorbeelden; de volgorde van de werelden maakt niet uit.
        """),
        L.task("Laat `all_worlds` elke wereld voor `n` voorbeelden teruggeven, elk één keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_worlds():
    box_worlds = L.code_box("all-worlds", lines=9)
    box_worlds
    return (box_worlds,)


@app.cell(hide_code=True)
def check_worlds(box_worlds):
    result_worlds = L.check("all-worlds", box_worlds.value)
    result_worlds
    return (result_worlds,)


@app.cell(hide_code=True)
def hints_worlds():
    L.hints(
        "all-worlds",
        [
            "Elke wereld tot nu toe geeft er twee nieuwe. Voor de eerste stap is er één wereld zonder labels, `[]`; na één stap zijn er `[0]` en `[1]`; na twee `[0, 0]`, `[0, 1]`, `[1, 0]` en `[1, 1]`.",
            "`world + [0]` is de wereld met een 0 erachter. `longer.append(...)` zet hem in de nieuwe lijst.",
            "`longer.append(world + [0])` en `longer.append(world + [___])`.",
        ],
        why="Voor 3 voorbeelden geeft dat de 8 werelden uit het uitgewerkte voorbeeld, in dezelfde volgorde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Gemiddeld goed

        `mean_right(predictions, worlds)` moet zeggen hoe vaak een regel met deze `predictions` goed zit op de onbekende voorbeelden, gemiddeld over de `worlds`: in elke wereld het aandeel van de voorspellingen dat gelijk is aan haar labels, en dan het gemiddelde van die aandelen. Vervang de twee `...`. De check roept `mean_right([0, 1, 1], worlds)` aan met de eerste regel en de 8 werelden uit het uitgewerkte voorbeeld, en probeert er nog vijf: de andere regel, de werelden waarin het eerste onbekende voorbeeld 0 is, vier voorbeelden, en twee voorbeelden, met al hun werelden en met maar een paar.
        """),
        L.task("Laat `mean_right` het gemiddelde aandeel goede voorspellingen over de werelden teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-right", lines=9)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-right", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-right",
        [
            "Stap 3 van het uitgewerkte voorbeeld, voor één regel, en dan een gemiddelde: het aandeel goed in elke wereld wordt al opgeteld in `total`.",
            "Tel een goed antwoord als `predictions[i] == world[i]`: `right = right + 1`. Het gemiddelde over de werelden is `total` gedeeld door het aantal werelden.",
            "`if predictions[i] == world[i]:` met `right = right + 1` eronder, en `return total / len(___)`.",
        ],
        why="De eerste regel heeft er over de 8 werelden 12 van de 24 goed: zijn aandelen tellen op tot 4, en 4 / 8 = 0,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ### Oefening 3. Eén overwinning
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-dataset",
        question="Een collega probeerde de vier modellen van deze les op een eigen dataset, en met cross-validation won de beslisboom duidelijk. De conclusie van die collega: de beslisboom is het beste model, en die collega gebruikt hem voortaan voor elk probleem. Wat zeg jij?",
        options={
            "agree": "Mee eens: cross-validation liet het zien, dus de boom is het beste model",
            "right": "De boom past bij deze data; op een ander probleem kan een ander model winnen, dus vergelijk elke keer opnieuw",
            "never": "Cross-validation kan modellen niet onderscheiden: volgens de stelling zijn ze allemaal even goed",
        },
        correct="right",
        why={
            "agree": "Cross-validation laat zien dat de boom op deze data wint: zijn aanname, drempels op één feature tegelijk, past erbij. Op de twee ronde klonten werd hij hier laatste. Zie *No free lunch* in *Het idee*.",
            "never": "De stelling gaat over het gemiddelde over alle mogelijke problemen. Op één echt probleem kunnen de modellen flink verschillen, en cross-validation laat zien hoeveel: op de ring hier van 0,567 tot 1,000. Zie *No free lunch* in *Het idee*.",
            "right": "Gemiddeld over alle problemen wint geen model, dus een overwinning zegt iets over het probleem, niet over het model. Vergelijk bij het volgende probleem opnieuw.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elk van de vier modellen staat voor één aanname: k-NN uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), de nearest centroid uit [Nu leren of later: de nearest centroid](/course/open/ml-nearest-centroid?lang=nl), de breedste straat uit [De breedste straat: de maximum-margin classifier](/course/open/ml-max-margin?lang=nl) en de boom uit [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl). Welke aanname bij je data past, zoek je uit met [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl). Hiermee sluit de module over modellen. In de volgende les, *Clusters in clusters: hierarchical clustering*, zijn de labels weg, en is de vraag welke voorbeelden bij elkaar horen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, result_mean, result_worlds):
    L.wrap_up(result_worlds, result_mean, choice_one)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
