# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-what-is-an-ml-system. Route: course/route/lessen/systemen.yaml. Checks, the course's spam service and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-what-is-an-ml-system", "nl")
    C = L.checks   # the course's spam service, its parts and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De workflow van machine learning](/course/open/ml-workflow?lang=nl) en [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="Wanneer meet je in de workflow van een project het model op de test set?",
        options={
            "rounds": "Na elke ronde van de workflow, om te zien of het model beter werd",
            "start": "Al bij het begin, zodat je het doel kent",
            "right": "Eén keer, aan het eind, na de keuze op de validation set",
            "never": "Nooit: de validation set zegt genoeg",
        },
        correct="right",
        why={
            "right": "De test set blijft buiten elke keuze, zodat de test set meet hoe het model het doet op data die het model nooit zag. Zodra een model in gebruik is, is elk nieuw bericht zulke data, en zulke data blijft binnenkomen: deze les gaat over wat een model dan om zich heen nodig heeft.",
            "rounds": "Als de test set na elke ronde meekiest, passen de keuzes ook bij de test set. Op welke set kiezen de rondes? Zie [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "start": "Het doel spreek je af voordat je traint, maar vanuit wat elke fout kost, niet vanuit de test set. Wanneer komt de test set aan bod? Zie [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "never": "De validation set hielp het model kiezen, dus is de score op de validation set iets te goed. Welke set meet het gekozen model op data die bij geen enkele keuze meedeed, en hoe vaak? Zie [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def quick_check_pipeline():
    quick_pipe = L.choice(
        "quick-pipe",
        question="Een pipeline van `StandardScaler()` en een model is gefit op de training set. Wat doet `pipe.predict` met een nieuw voorbeeld?",
        options={
            "learns": "Eerst een nieuw gemiddelde en een nieuwe spreiding leren van het voorbeeld",
            "right": "Het voorbeeld schalen met wat de training set gaf, en dan voorspellen",
            "raw": "Voorspellen op het ruwe voorbeeld: schalen is alleen voor trainen",
            "refit": "De hele pipeline opnieuw fitten, met het nieuwe voorbeeld erbij",
        },
        correct="right",
        why={
            "right": "`pipe.predict` verandert nieuwe voorbeelden alleen met wat elke stap van de training set leerde. In een draaiend systeem moet elk nieuw bericht door precies de stappen waarmee het model getraind is: een stap die anders werkt, laat het model data zien waarvan het nooit leerde.",
            "learns": "`fit` leert, en `predict` gebruikt alleen wat geleerd is. Wat zou één nieuw voorbeeld geven als eigen gemiddelde en spreiding? Zie [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "raw": "Het model leerde van geschaalde voorbeelden. Wat gebeurt er met de voorspellingen bij getallen van een andere grootte? Zie [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "refit": "`pipe.fit` leert elke stap, en `pipe.predict` gebruikt de stappen. Welke van de twee wordt hier aangeroepen? Zie [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
        },
    )
    quick_pipe
    return (quick_pipe,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2006 bood Netflix een miljoen dollar aan wie 10% beter dan het eigen systeem kon voorspellen hoe de leden films beoordelen. In 2009 won een team, met een combinatie van veel modellen. Netflix heeft het winnende model nooit gebruikt. In 2012 legde Netflix uit waarom. Een jaar na de start van de wedstrijd had een team een eerste, kleinere prijs gewonnen met een combinatie van 107 algoritmen; Netflix bouwde de twee beste daarvan in de eigen dienst en hield deze daar draaiende. De extra accuracy van de uiteindelijke winnaar woog volgens Netflix niet op tegen het werk om het winnende model in deze dienst te bouwen, en de aandacht van Netflix lag toen al ergens anders. Een model met de beste score op een test set is nog niet iets wat mensen kunnen gebruiken: het model moet de data binnenkrijgen, snel antwoorden, blijven werken als de data verandert, en jarenlang onderhouden worden. Deze les gaat over alles rond het model.

    **Een voorbeeld uit het dagelijks leven.** Een goed recept maakt nog geen restaurant. Rond het recept koopt een restaurant ingrediënten in en controleert of de ingrediënten vers zijn, neemt het bestellingen op, bedient het elke tafel binnen redelijke tijd, luistert het naar klachten en past het de kaart aan met de seizoenen. Het recept is één bladzijde; het restaurant draaiende houden is al de rest. En een gerecht dat drie uur kost, werkt niet op een drukke avond, hoe lekker het ook is.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share",
        question="De course schreef een kleine spamdienst in Python, in acht delen, met een model uit scikit-learn. Hoeveel van de code van de dienst is het model, denk je?",
        options={
            "most": "Het meeste: het model doet het echte werk",
            "half": "Ongeveer de helft",
            "quarter": "Ongeveer een kwart",
            "right": "Een paar procent",
        },
        reveal="**Een paar procent.** Van de 118 regels code van de dienst zijn er 5 van het model: 4,2%. Het leren zelf doet scikit-learn, en de vijf regels geven scikit-learn de data. Alle andere regels nemen berichten aan, controleren de berichten, maken er getallen van, geven antwoorden, bekijken de antwoorden en trainen het model opnieuw.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een model is een deel van een systeem.** In de projecten tot nu toe stond de data klaar in een bestand, trainde je een model één keer en mat je het model op een test set. Een model dat mensen gebruiken, heeft veel meer om zich heen nodig. Een **ML-systeem** is het model samen met alles wat het in de praktijk bruikbaar maakt: de data die binnenkomt, controles op deze data, features, trainen, de antwoorden serveren, kijken hoe het gaat, en de mensen die het geheel onderhouden.

        **De delen van de spamdienst van de course.** De course schreef een klein ML-systeem: een dienst die tekstberichten krijgt en antwoordt met spam, ok of geen score. Elk bericht gaat van boven naar beneden door vijf delen. *Verzamelen* neemt elk bericht één keer aan, *controle* kijkt of het bericht bruikbaar is, *features* maakt van de woorden tellingen, het *model* geeft een kans op spam, en *serving* maakt daar een antwoord van: altijd, ook als het model niet kan helpen. Ernaast bewaart *instellingen* de getallen die de andere delen lezen, zoals de threshold, bekijkt *monitoring* de antwoorden, en bewaart *feedback* de berichten waarvan gebruikers het antwoord verbeteren, om het model opnieuw te trainen.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.accordion({"De code van de dienst, deel voor deel": mo.vstack([
            mo.vstack([mo.md(f"**{_name}**"), mo.md("```python\n" + _code + "\n```")])
            for _name, _code in C.parts_accordion(L.lang).items()
        ])}),
        mo.md(r"""
        **Het model is een klein deel.** In 2015 schreven D. Sculley en anderen bij Google de paper *Hidden Technical Debt in Machine Learning Systems*. De bekendste figuur daaruit tekent een echt ML-systeem als blokken: configuratie, data verzamelen, data controleren, features maken, serving, monitoring, tools voor analyse, tools die computers en processen beheren, en in het midden één klein blok: de ML-code. De figuur is een schets, geen meting, maar mensen die ML-systemen bouwen, herkennen het beeld. In de dienst van de course is het model kort omdat scikit-learn het leren doet: `LogisticRegression().fit(rows, labels)` is één regel. Dat is gebruikelijk: het model komt vaak uit een library, en de code die het model met al het andere verbindt, is van het systeem zelf. De paper noemt de code die alleen data in zo'n algemeen pakket stopt en er weer uit haalt *glue code*.

        **Technische schuld.** Een shortcut in code werkt als een lening: je wint nu tijd, en het kost later meer, elke keer dat iemand eromheen moet werken. Ward Cunningham vergeleek dat in 1992 met een schuld; sindsdien heet het **technische schuld**, technical debt. Sculley en anderen laten zien dat ML-systemen er eigen vormen van hebben, bovenop de gewone. Eén is *CACE*, van changing anything changes everything: verander iets, en alles verandert. Een model mengt alle features door elkaar: verander er één, voeg er één toe of haal er één weg, en na opnieuw trainen gebruikt het model alle andere anders, dus staat geen verandering op zichzelf. Een verwant probleem: een model hangt af van de features zoals de features tijdens het trainen gemaakt werden; verander hoe één feature gemaakt wordt, en het model antwoordt anders, zonder dat er één regel van de eigen code verandert. Een andere is de *pipeline jungle*: data-voorbereiding die stukje bij beetje groeit, tot niemand er nog iets aan durft te veranderen. En een model kan de eigen toekomstige data sturen: wat het model nu beslist, komt terecht in de data waarvan het model later leert.

        **Onderzoek en productie.** Van een model in onderzoek en hetzelfde model in productie, in gebruik door echte mensen, wordt iets anders gevraagd:

        | | onderzoek | productie |
        |---|---|---|
        | het doel | de beste score op een vaste test set | goede antwoorden voor echte gebruikers, snel en goedkoop genoeg, jarenlang |
        | de data | één keer verzameld, opgeschoond, vast | blijft binnenkomen, verandert, en is soms kapot |
        | tijd | hoe lang het trainen duurt | hoe lang elk antwoord duurt, voor veel gebruikers tegelijk |
        | klaar | als de resultaten opgeschreven zijn | pas als het systeem uitgezet wordt |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **ML-systeem** | | een model samen met alles eromheen wat het in de praktijk bruikbaar maakt: de data die binnenkomt, controles, features, trainen, serving, monitoring en de mensen die het onderhouden | |
        | **technische schuld** | | wat shortcuts van nu later kosten, doordat een systeem moeilijker te veranderen wordt; ML-systemen hebben er eigen vormen van, zoals CACE | |
        """),
        L.watch_out("Een betere score op de test set maakt nog geen beter systeem. Een model dat een iets hogere accuracy heeft en tegelijk tien keer zo traag is, of veel moeilijker draaiende te houden is, kan het hele systeem slechter maken."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een ML-systeem bekijken:

    1. Volg één invoer vanaf het moment dat de invoer binnenkomt tot het antwoord, en noem elk deel waar de invoer langskomt.
    2. Noem de delen naast deze weg: instellingen, monitoring, feedback en opnieuw trainen.
    3. Vraag bij elk deel wat echte data ermee kan doen: twee keer binnenkomen, leeg zijn, te lang zijn, veranderen in de loop van de tijd.
    4. Tel hoeveel van de code het model is, en hoeveel alles eromheen.
    5. Schrijf op wat productie vraagt: hoe snel een antwoord moet komen, voor hoeveel gebruikers, tegen welke kosten. Kies dan het model dat past, niet alleen het model met de beste score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.part_code("model", L.lang)` geeft de code van het deel van het model als één tekst. De cel laat deze tekst zien, knipt de tekst in regels met `code.split("\n")`, en telt alle regels. Voer de cel uit.
    """)
    return


@app.cell
def example_lines():
    code = C.part_code("model", L.lang)
    print(code)
    lines = code.split("\n")
    print()
    print("alle regels:", len(lines))
    return code, lines


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het deel heeft 11 regels, maar 4 zijn leeg en 2 zijn commentaar, dat de code uitlegt en niets doet. Slechts 5 regels doen werk: de `import`, twee `def`-regels en twee `return`-regels. Oefening 1 telt alleen zulke regels code.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lines():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tel de regels code

        `code_lines(code)` krijgt de code van een deel als één tekst, en geeft het aantal regels code terug: regels die niet leeg zijn en geen commentaar. `code.split("\n")` knipt de tekst in regels, en `text` is de `line` zonder de spaties aan begin en eind, uit `line.strip()`. Bij een regel code gaat `count` één omhoog. Vervang de `...` door de voorwaarde: `text` is niet leeg, en het eerste teken is geen `#`. `and` kijkt eerst naar de linkerkant, en alleen naar de rechterkant als de linkerkant `True` is.

        De check roept `code_lines(C.part_code("model"))` aan, en dat moet 5 geven. Daarna de zeven andere delen van de dienst, een kort stuk code met een ingesprongen commentaar en een regel met alleen spaties, een lege tekst, en een tekst met alleen commentaar.
        """),
        L.task("Tel alleen de regels die niet leeg zijn en geen commentaar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lines():
    box_lines = L.code_box("count-lines", lines=8)
    box_lines
    return (box_lines,)


@app.cell(hide_code=True)
def check_lines(box_lines):
    result_lines = L.check("count-lines", box_lines.value)
    result_lines
    return (result_lines,)


@app.cell(hide_code=True)
def hints_lines():
    L.hints(
        "count-lines",
        [
            "Stap 4 van het recept. De voorwaarde heeft twee delen met `and` ertussen: `text` is niet leeg, en het eerste teken van `text` is geen `#`.",
            "`text != \"\"` is het eerste deel. Het eerste teken van een tekst staat op plaats 0. `and` stopt bij een linkerkant die `False` is, dus bij een lege regel zoekt Python nooit naar een eerste teken dat er niet is.",
            "`        if text != \"\" and ___:`",
        ],
        why="`if text != \"\" and text[0] != \"#\":` laat een regel door waar iets op staat dat niet met `#` begint. De spaties zijn weg uit `text`, dus begint ook een commentaar binnen een functie met `#`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het beste model binnen de tijd

        Een team vergelijkt zes spammodellen: de accuracy op een validation set met veel berichten, en de milliseconden die één bericht kost. De getallen zijn een voorbeeld dat de course verzon. De mailserver neemt 100 berichten per seconde aan op één core, dus mag één bericht hoogstens 10 ms kosten.

        | model | naam in de code | accuracy | ms per bericht |
        |---|---|---|---|
        | woordenlijst | `"word list"` | 0,80 | 0,002 |
        | naive Bayes | `"naive Bayes"` | 0,93 | 0,02 |
        | logistic regression | `"logistic regression"` | 0,95 | 0,05 |
        | klein netwerk | `"small network"` | 0,96 | 1,5 |
        | 30 modellen samen | `"30 models together"` | 0,97 | 40 |
        | large language model | `"large language model"` | 0,98 | 800 |

        `best_in_budget(models, budget)` krijgt een lijst met modellen, elk een tuple `(name, accuracy, ms)`, en een budget in milliseconden per bericht. Van de modellen die binnen het budget passen, geeft de functie de naam terug van het model met de hoogste accuracy, of `None` als geen enkel model past. De loop `for name, accuracy, ms in models` gaat de modellen langs; `best_name` en `best_accuracy` bevatten het beste model tot nu toe. Vervang de `...` door de voorwaarde: het model past binnen `budget`, en de `accuracy` is hoger dan `best_accuracy`. In *Laat de metric kiezen* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl) was een precision van minstens 0,9 de voorwaarde; hier is dat de tijd.

        De check roept `best_in_budget(C.MODELS, 10.0)` aan: de zes modellen uit de tabel, met een budget van 10 ms. Daarna dezelfde modellen met een budget van 1.000, 1,5 en 0,001 ms, een lege lijst, en drie andere modellen.
        """),
        L.task("Geef van de modellen die binnen het budget passen het model met de hoogste accuracy terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_budget():
    box_budget = L.code_box("budget", lines=9)
    box_budget
    return (box_budget,)


@app.cell(hide_code=True)
def check_budget(box_budget):
    result_budget = L.check("budget", box_budget.value)
    result_budget
    return (result_budget,)


@app.cell(hide_code=True)
def hints_budget():
    L.hints(
        "budget",
        [
            "Stap 5 van het recept. De voorwaarde heeft twee delen met `and` ertussen: het tweede, `accuracy > best_accuracy`, vergelijkt met het beste model tot nu toe, en het eerste vergelijkt `ms` met `budget`.",
            "Een model past als één bericht niet langer duurt dan het budget: met een budget van 1,5 ms past het kleine netwerk van precies 1,5 ms nog.",
            "`        if ___ and accuracy > best_accuracy:`",
        ],
        why="`if ms <= budget and accuracy > best_accuracy:` slaat elk model over dat te traag is, en houdt van de andere het model met de hoogste accuracy. Bij 10 ms zijn de 30 modellen samen en het large language model te traag, en is het kleine netwerk het beste van de rest.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De dienst aan het werk

    De demo stuurt tien berichten door de dienst van de course, één voor één, zoals berichten zouden kunnen binnenkomen: met hoofdletters en leestekens, één bericht dat twee keer komt, en een paar berichten die geen score kunnen krijgen. Bij elk bericht laat de afbeelding zien wat elk deel met het bericht deed; een deel dat niet draaide, heeft een rand van streepjes, en een balk links markeert een deel dat het bericht tegenhield, alarm sloeg of een markering van een gebruiker kreeg. Bij bericht 5 haalt een gebruiker het bericht uit de spammap.
    """)
    return


@app.cell(hide_code=True)
def demo_service():
    L.explore("message", "bericht", lambda _v: C.frame(int(_v), L.lang), start=1, values=list(range(1, 11)), role="",
              task="Ga de tien berichten langs: bij hoeveel berichten geeft het model het antwoord?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Het model geeft het antwoord bij 5 van de 10 berichten. Eén bericht kwam twee keer binnen, drie zijn niet bruikbaar (leeg, bijna geen letters, te lang), en bij één bericht kent het model geen enkel woord: de andere delen handelen dit allemaal af zonder het model. Bij bericht 5 zat het model fout, en alleen een gebruiker merkte het: feedback bewaart het bericht om later op te trainen. Bij bericht 8 en 9 geeft de monitoring een alarm omdat de meeste van de laatste antwoorden geen score hebben, en bij bericht 10 omdat alle antwoorden met een score spam zijn. Niets hiervan zie je terug in een score op een test set.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cace():
    mo.md(r"""
    ### Oefening 3. Een kleine verandering

    Het team verandert één regel van features: `text.lower()` wordt `text`, dus vanaf nu houden woorden de hoofdletters. Het model wordt niet opnieuw getraind.
    """)
    return


@app.cell(hide_code=True)
def choice_cace_cell():
    choice_cace = L.choice(
        "cace",
        question="Wat gebeurt er met de antwoorden van de dienst?",
        options={
            "nothing": "Niets: geen enkele regel van het model veranderde",
            "right": "Sommige antwoorden veranderen, al blijft de code van het model gelijk",
            "error": "De dienst stopt met een fout",
            "learns": "Het model leert de nieuwe woorden vanzelf, terwijl het model antwoordt",
        },
        correct="right",
        why={
            "right": "Het model leerde van de features zoals de features tijdens het trainen gemaakt werden, en krijgt nu andere tellingen voor dezelfde berichten. Sculley en anderen noemen dat een unstable data dependency: na het trainen veranderde iets waar het model van afhangt. In een pipeline, zoals in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl), worden de stappen en het model samen gefit, dus leert het model bij opnieuw fitten van de nieuwe features; verander je een stap zonder opnieuw te fitten, dan waarschuwt ook daar niets je.",
            "nothing": "Het model krijgt de tellingen van features. Wat krijgt het model na de verandering voor *WIN*, als de vocabulary alleen *win* bevat?",
            "error": "De rijen met tellingen zijn even lang als eerst, met een getal voor elk woord van de vocabulary. Is er iets dat de code zou stoppen?",
            "learns": "Het model leert alleen als `train` aangeroepen wordt, in feedback. Wat gebeurt er tussen twee keer trainen?",
        },
    )
    choice_cace
    return (choice_cace,)


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.md(r"""
    ### Oefening 4. Leren van de gebruikers

    De dienst wordt elke week opnieuw getraind, op de berichten die gebruikers markeerden: spam die gebruikers tussen de mail vonden, en ok-berichten die gebruikers uit de spammap haalden. Weinig gebruikers kijken ooit in de spammap.
    """)
    return


@app.cell(hide_code=True)
def choice_loop_cell():
    choice_loop = L.choice(
        "loop",
        question="Wat kan er in de loop van de weken met het model gebeuren?",
        options={
            "better": "Het wordt alleen beter, omdat het elke week meer data krijgt",
            "right": "Het leert zelden van de fouten bij ok-berichten",
            "same": "Niets: gebruikers markeren de berichten goed",
            "forget": "Het vergeet alle spam van eerder",
        },
        correct="right",
        why={
            "right": "Een ok-bericht dat het model in de spammap zet, wordt zelden gezien, dus zelden gemarkeerd: wat het model nu beslist, stuurt de data waarvan het model later leert. Sculley en anderen noemen dat een direct feedback loop. Een team kan een steekproef uit de spammap met de hand nakijken, of een paar berichten ongefilterd doorlaten.",
            "better": "Meer data helpt alleen als de data de fouten laat zien. Welke fouten zien gebruikers, en welke fouten zien gebruikers bijna nooit?",
            "same": "De markeringen die binnenkomen, kunnen goed zijn. Maar welke berichten krijgen nooit een markering?",
            "forget": "De oude voorbeelden blijven in de data waarop het model traint. Welke fouten ontbreken in de nieuwe markeringen?",
        },
    )
    choice_loop
    return (choice_loop,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De workflow van machine learning](/course/open/ml-workflow?lang=nl) eindigde een project met één meting op de test set, en in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl) werden de stappen en het model één geheel. Deze les zette dat geheel in een systeem: delen die data aannemen, controleren, antwoorden, bekijken en opnieuw leren, met het model als een klein deel in het midden. In de volgende les, *Van idee tot afscheid: de levenscyclus van een AI-systeem*, volg je zo'n systeem door de tijd, van het eerste idee tot de dag dat het systeem uitgezet wordt, en zie je wie er in elke fase bij betrokken is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cace, choice_loop, result_budget, result_lines):
    L.wrap_up(result_lines, result_budget, choice_cace, choice_loop)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
