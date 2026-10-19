# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-figure-and-table. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-figure-and-table", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl) en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="Een model scoort 2 punten hoger dan een ander model op een benchmark. Wat zegt die score?",
        options={
            "better": "Dat het het betere model is",
            "nothing": "Niets: benchmarkscores betekenen niets",
            "right": "Dat het het beter doet op die taken, op die manier gescoord",
        },
        correct="right",
        why={
            "better": "Beter op die benchmark, niet in het algemeen. Zie [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).",
            "nothing": "Een score betekent wel iets: hoe het model het op die taken doet. Alleen niet meer dan dat. Zie [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).",
            "right": "Een benchmarkscore is een meting op die taken; of dat ook geldt voor waarvoor jij het model wilt gebruiken, zegt de benchmark niet.",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def quick_check_runs():
    quick_runs = L.choice(
        "quick-runs",
        question="Twee modellen, elk één keer getraind, eindigen met een validation loss van 0,31 en 0,33. Wat kun je zeggen?",
        options={
            "right": "Nog weinig: met één run van elk kan het verschil toeval zijn",
            "lower": "Het eerste model is beter: zijn validation loss is lager",
            "same": "Ze zijn precies even goed: het verschil is klein",
        },
        correct="right",
        why={
            "lower": "Lager in deze ene run. Curves springen een beetje op en neer van epoch tot epoch, dus een andere run kan iets hoger of lager eindigen. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "same": "Dat kan één run van elk ook niet laten zien: het verschil kan echt zijn of toeval. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "Elke curve is één run, en curves springen een beetje op en neer, dus een verschil van 0,02 kan toeval zijn. Meerdere runs van elk zouden laten zien hoeveel ze uiteenlopen.",
        },
    )
    quick_runs
    return (quick_runs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De resultaten van een paper staan op drie plekken. Het **abstract** zegt wat de paper beweert, de figuren laten zien hoe getallen veranderen, en de resultatentabel vergelijkt methoden. Veel misverstanden over AI-resultaten komen doordat iemand de claim leest vóór de assen, de eenheden en de omstandigheden: een verschil dat in een figuur enorm lijkt, kan in getallen piepklein zijn, en een rij die in een tabel wint, kan op een makkelijkere test gemeten zijn.

    **Een voorbeeld uit het dagelijks leven.** Voordat je beoordeelt of twee plaatsen op een kaart dicht bij elkaar liggen, kijk je naar de schaalstreep: twee centimeter kan twee kilometer zijn, of tweehonderd.
    """)
    return


@app.cell(hide_code=True)
def predict_gap():
    guess_gap = L.guess(
        "gap",
        question="Een figuur in een paper laat de loss van twee modellen zien over 100 trainingsstappen, met een logaritmische y-as. De twee lijnen lopen naast elkaar, de hele weg met dezelfde afstand ertussen. Wat denk je dat er op een gewone schaal met het verschil tussen de twee losses gebeurt?",
        options={
            "same": "Het blijft de hele weg hetzelfde",
            "right": "Het krimpt terwijl beide losses dalen",
            "grows": "Het groeit naarmate het trainen doorgaat",
        },
        reveal="**Het krimpt terwijl beide losses dalen.** Op een logaritmische schaal betekent dezelfde afstand dezelfde factor: hier is de ene loss bijvoorbeeld steeds twee keer de andere. Aan het begin is dat 2 tegen 1, een verschil van 1; aan het eind 0,02 tegen 0,01, een verschil van 0,01. De figuur laat een vaste verhouding zien, geen vast verschil. Daarom lees je de schaal van een as voordat je iets uit de lijnen afleidt.",
    )
    guess_gap
    return (guess_gap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Het **abstract** is de korte samenvatting bovenaan een paper: het probleem, de aanpak, het belangrijkste resultaat, en waarom het ertoe doet, in één alinea. Lees het als de claim, zoals in [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl): het zegt waarnaar je zoekt in de figuren en tabellen, waar de metingen staan.

        **Een figuur lezen** gaat in een vaste volgorde, voordat je kijkt wat de lijnen doen:

        1. **Het bijschrift**, de tekst onder de figuur: wat er te zien is, op welke data, en hoe vaak het gemeten is, in één run of als het gemiddelde van meerdere.
        2. **De assen**: welke grootheid, in welke eenheid, op een gewone of een logaritmische schaal, en waar elke as begint. Een as die niet bij 0 begint, laat kleine verschillen groot lijken.
        3. **Wat er vergeleken wordt**: welke lijn of staaf welke is, en of ze op dezelfde manier gemeten zijn.
        4. **Dan de claim**: wat de figuur laat zien, en wat niet.

        **Een resultatentabel** heeft de methoden als rijen en de metingen als kolommen, vaak met het beste getal vet. Alleen rijen die onder dezelfde omstandigheden gemeten zijn, kun je vergelijken: dezelfde test set, dezelfde metric, dezelfde hoeveelheid training data. Een getal dat uit een andere paper is overgenomen, kan anders gemeten zijn; het bijschrift of een voetnoot hoort dat te zeggen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **abstract** | | de korte samenvatting bovenaan een paper: probleem, aanpak, belangrijkste resultaat en waarom het ertoe doet | |
        | **bijschrift** | | de tekst onder een figuur of boven een tabel die zegt wat hij laat zien | |
        | **resultatentabel** | | een tabel met methoden als rijen en metingen als kolommen, om ze te vergelijken | `rows` |
        """),
        L.watch_out("Een verschil tussen twee getallen is nog geen echt verschil. Komt elk getal uit één run, dan kan een tweede run met een andere seed net iets anders uitkomen. Een paper die het gemiddelde van meerdere runs geeft, met hoeveel ze uiteenlopen, laat zien of een verschil meer is dan toeval."),
        mo.md(r"""
        De figuur: dezelfde twee accuracies, 91,2% en 91,8%, twee keer. Links begint de as bij 91, rechts bij 0.
        """),
        mo.Html(C.bars_picture({"cut": "As vanaf 91", "full": "As vanaf 0", "y": "accuracy (%)", "point": ",",
                                "alt": "Twee staafdiagrammen van dezelfde twee scores, A 91,2 en B 91,8. Links loopt de as van 91 tot 92, en is de staaf van B vier keer zo hoog als die van A. Rechts loopt de as van 0 tot 100, en lijken de twee staven even hoog."})),
        mo.md(r"""
        Links is de staaf van B vier keer zo hoog als die van A; rechts zie je nauwelijks verschil. Beide geven dezelfde 0,6 procentpunt juist weer. Welke eerlijk is, hangt af van wat de paper erover beweert.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een figuur of een resultatentabel te lezen:

    1. Lees het bijschrift: wat er te zien is, op welke data, en uit hoeveel runs.
    2. Lees de assen, of de kolommen van een tabel: welke grootheid, welke eenheid, welke schaal, waar een as begint, en of hoger of lager beter is.
    3. Kijk wat er vergeleken wordt: welke lijn, staaf of rij welke is, en of ze onder dezelfde omstandigheden gemeten zijn.
    4. Lees pas dan de claim, en leg die naast wat de figuur of tabel laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De linkergrafiek uit de figuur, zoals een verzonnen paper hem zou kunnen laten zien, met de claim "model B is veel beter dan model A".

    1. **Bijschrift:** accuracy op een benchmark van 1.000 vragen, één run per model.
    2. **Assen:** accuracy in procenten, waarbij hoger beter is, op een gewone schaal, maar beginnend bij 91, niet bij 0.
    3. **Wat er vergeleken wordt:** twee modellen op dezelfde benchmark, elk één keer gemeten.
    4. **De claim:** B scoorde 91,8% en A 91,2%: 0,6 procentpunt, of 6 van de 1.000 vragen. "Veel beter" komt van de afgekapte as, niet van de getallen. En met één run per model kunnen 6 vragen toeval zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example_table():
    mo.md(r"""
    Dezelfde stappen op een resultatentabel, in code. Een verzonnen tabel van vier methoden met hun loss op de test set. Methode C is alleen op een klein, makkelijk deel van de test set gemeten. Bij een loss is lager beter, dus de cel bewaart de laagste loss tot nu toe, en begint hoger dan elke loss in de tabel. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_table():
    # 1. het bijschrift: de loss op de test set; methode C alleen op een klein deel ervan
    _rows = [
        ("Methode A", "full", 0.84),
        ("Methode B", "full", 0.79),
        ("Methode C", "small", 0.52),
        ("Methode D", "full", 0.81),
    ]
    # 2. de kolommen: methode, test set, loss; lager is beter
    _best_name = None
    _best_loss = 100
    for _name, _test_set, _loss in _rows:
        # 3. vergelijk alleen de rijen die op dezelfde manier gemeten zijn: op de volledige test set
        if _test_set == "full":
            if _loss < _best_loss:
                _best_name = _name
                _best_loss = _loss
        print(_name, _test_set, _loss, "beste tot nu toe:", _best_name)
    # 4. de beste methode onder dezelfde omstandigheden
    _best_name
    return


@app.cell(hide_code=True)
def text_worked_example_table_result():
    mo.md(r"""
    `for _name, _test_set, _loss in _rows:` haalt elke rij uit elkaar in drie namen, zoals `x, y = point` doet in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl). Methode C heeft de laagste loss, 0,52, maar op een makkelijkere test, dus de buitenste `if` slaat hem over. Onder de rijen op de volledige test set is methode B het best, met 0,79: vanaf die rij blijft `_best_name` `Methode B`. Een tabel die de 0,52 van C vet zet als beste getal, vergelijkt rijen die onder andere omstandigheden gemeten zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_times():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Lees een figuur
        """),
        mo.Html(C.loss_picture({"title": "Training loss, één run", "x": "stap", "y": "loss (logaritmische schaal)",
                                "alt": "Een rechte lijn die daalt van 2 bij stap 0 naar 0,02 bij stap 100, op een logaritmische y-as met de rasterlijnen 0,01, 0,1 en 1."})),
    ])
    return


@app.cell(hide_code=True)
def choice_times_cell():
    choice_times = L.choice(
        "times-smaller",
        question="In deze figuur daalt de loss van 2 bij stap 0 naar 0,02 bij stap 100. Hoeveel keer kleiner is hij bij stap 100 dan bij stap 0?",
        options={
            "grid": "2 keer: de lijn zakt twee rasterlijnen",
            "diff": "1,98 keer: hij gaat van 2 omlaag naar 0,02",
            "right": "100 keer: 2 gedeeld door 0,02",
        },
        correct="right",
        why={
            "grid": "Elke rasterlijn op een logaritmische schaal is een factor 10, dus twee rasterlijnen zijn een factor 10 · 10. Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "diff": "1,98 is hoeveel hij omlaagging: een verschil, geen factor. Hoe vaak past 0,02 in 2?",
            "right": "2 / 0,02 = 100: twee stappen van een factor 10 op de logaritmische schaal. Een rechte lijn op een logaritmische schaal betekent dat de loss elke stap met dezelfde factor krimpt.",
        },
    )
    choice_times
    return (choice_times,)


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De eerlijke winnaar van een tabel

        Een resultatentabel, zoals een paper hem zou laten zien:

        | Methode | Test set | Accuracy (%) |
        |---|---|---:|
        | Ours | `full` | 84,1 |
        | Model X | `full` | 82,7 |
        | Model Y | `easy half` | 89,5 |
        | Model Z | `full` | 83,9 |

        In het vak is dezelfde tabel een lijst van rijen: de naam van een methode, de test set waarop hij gemeten is, en zijn accuracy. Model Y is alleen op de makkelijke helft van de test set gemeten. Vervang de `...`: bewaar de naam van de beste rij in `best_name` en zijn score in `best_score`, maar alleen onder de rijen die op de volledige test set, `"full"`, gemeten zijn. De laatste regel laat `best_name` zien. De check probeert je code ook op een andere tabel.
        """),
        L.task("Zoek de beste rij van de tabel onder de rijen op de volledige test set."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fair():
    box_fair = L.code_box("fair-best", lines=9)
    box_fair
    return (box_fair,)


@app.cell(hide_code=True)
def check_fair(box_fair):
    result_fair = L.check("fair-best", box_fair.value)
    result_fair
    return (result_fair,)


@app.cell(hide_code=True)
def hints_fair():
    L.hints(
        "fair-best",
        [
            "Stap 3 van het recept: alleen rijen die onder dezelfde omstandigheden gemeten zijn, tellen mee. Bewaar daarvan de beste, zoals het uitgewerkte voorbeeld de beste rij bewaart; hier is hoger beter.",
            "Eerst een `if` die alleen de rijen op de volledige test set doorlaat. Daarbinnen een tweede `if`: is deze score hoger dan de beste tot nu toe? Bewaar dan zowel de naam als de score.",
            "`if test_set == \"full\":`, daarbinnen `if score > best_score:`, en daaronder `best_name = name` en `best_score = ___`.",
        ],
        why="Onder de rijen op de volledige test set heeft \"Ours\" 84,1, Model Z 83,9 en Model X 82,7. De 89,5 van Model Y is hoger, maar is op de makkelijke helft gemeten, dus die telt niet mee.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_abstract():
    mo.md(r"""
    ### Oefening 3. Lees een abstract

    Een verzonnen abstract: *"We introduce SparkNet, a small model for reading comprehension. On the QuickRead benchmark it reaches an accuracy of 84.1%, 0.2 points above the previous best model, while it needs half as much computation to train."* Papers zijn bijna altijd in het Engels. In het Nederlands staat er: SparkNet, een klein model voor begrijpend lezen, haalt op de benchmark QuickRead een accuracy van 84,1%, 0,2 punt boven het beste eerdere model, met de helft van het rekenwerk om te trainen.
    """)
    return


@app.cell(hide_code=True)
def explain_abstract():
    explain_read = L.explain(
        "abstract",
        question="Wat beweert dit abstract? En waar zou je in de figuren en tabellen van de paper naar zoeken om het te controleren? Schrijf het op.",
        model_answer="**De claims:** SparkNet haalt 84,1% op QuickRead, 0,2 punt meer dan het beste eerdere model, met de helft van het rekenwerk. **Te controleren:** een resultatentabel waarin alle rijen op dezelfde test set van QuickRead gemeten zijn, onder dezelfde omstandigheden; of de 0,2 punt uit één run komt of het gemiddelde van meerdere is, en hoeveel runs uiteenlopen, want 0,2 punt kan toeval zijn; en een figuur of tabel met het rekenwerk van elk model.",
        points=[
            "je noemt de claims: 84,1%, 0,2 punt boven het beste eerdere model, de helft van het rekenwerk",
            "je zoekt een resultatentabel met alle rijen onder dezelfde omstandigheden",
            "je vraagt of 0,2 punt meer is dan de spreiding tussen runs",
        ],
    )
    explain_read
    return (explain_read,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl) vroeg wat er gemeten is; in de figuren en tabellen van een paper vind je het. De logaritmische schaal uit [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl) en de loss curves uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) zijn de figuren die je het vaakst tegenkomt. In de volgende les, *Hoe onderzoek werkt: vraag, hypothese, experiment*, zie je waar de metingen vandaan komen: een experiment dat bedacht is om een hypothese te toetsen. Een latere les, *Eén run is geen resultaat: seeds en variantie*, meet hoeveel runs uiteenlopen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_times, explain_read, result_fair):
    L.wrap_up(choice_times, result_fair, explain_read)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
