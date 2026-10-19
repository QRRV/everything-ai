# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-one-hot. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow classification

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-one-hot", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Delen van een array kiezen](/course/open/np-indexing?lang=nl) en [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. Wat geeft `m[1]`?",
        options={
            "column": "`array([2, 5])`",
            "right": "`array([4, 5, 6])`",
            "number": "`2`",
        },
        correct="right",
        why={
            "column": "Dat is de kolom `m[:, 1]`. Wat kiest één index tussen de haken? Zie de tabel met nieuwe woorden in [Delen van een array kiezen](/course/open/np-indexing?lang=nl).",
            "number": "Eén index op een matrix geeft meer dan één getal. Zie de tabel met nieuwe woorden in [Delen van een array kiezen](/course/open/np-indexing?lang=nl).",
            "right": "Eén index kiest een hele rij, geteld vanaf 0: rij 1 is de tweede rij.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_label_kind():
    quick_label_kind = L.choice(
        "quick-label-kind",
        question="Welk van deze labels is een klasse die als getal geschreven is?",
        options={
            "right": "De buslijn die iemand neemt, zoals lijn 3",
            "temperature": "De temperatuur van morgen, zoals 14 graden",
            "price": "De prijs van een huis, zoals 300 duizend euro",
        },
        correct="right",
        why={
            "temperature": "14,5 graden kan ook: waarden ertussen hebben betekenis, dus dit label is een getal. Zie stap 4 van het recept in [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "price": "Een prijs ertussen heeft betekenis, en dichtbij is bijna goed: dit label is een getal. Zie stap 4 van het recept in [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "right": "Lijn 2,5 bestaat niet, en lijn 3 is geen drie keer lijn 1: de getallen zijn alleen namen, dus de buslijnen zijn klassen.",
        },
    )
    quick_label_kind
    return (quick_label_kind,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model rekent met getallen, dus een klasse moet ook getallen worden. Maar de klassennummers 0, 1 en 2 brengen iets mee wat niet klopt: een volgorde, en hoeveelheden. Voor een model is "vogel" dan twee keer "hond". Geef daarom elke klasse een eigen plek, in een vector met één plek per klasse. Elke klasse ligt dan even ver van elke andere klasse, en een model kan elke klasse een eigen getal geven, zoals een score of een kans.

    **Een voorbeeld uit het dagelijks leven.** Een stembiljet heeft één vakje per kandidaat, en je kleurt er precies één in. Niemand leest het biljet als "lijst 3 is drie keer lijst 1".
    """)
    return


@app.cell(hide_code=True)
def predict_hesitate():
    guess_hesitate = L.guess(
        "hesitate",
        question="Een model leert de labels 0 = kat, 1 = hond en 2 = vogel als gewone getallen. Bij een foto kan het niet kiezen tussen kat en vogel. Wat voorspelt het volgens jou?",
        options={
            "dog": "Ongeveer 1: een hond",
            "either": "0 of 2, willekeurig",
            "unsure": "Een getal dat 'twijfel' betekent",
        },
        reveal="**Ongeveer 1: een hond.** Halverwege 0 en 2 ligt 1, dus een model dat twijfelt tussen kat en vogel voorspelt juist de klasse waar het niet aan dacht. Met een plek per klasse kan het 0,5 zeggen voor kat en 0,5 voor vogel, en 0 voor hond.",
    )
    guess_hesitate
    return (guess_hesitate,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Leg eenmalig een volgorde vast voor de $K$ klassen, zoals kat, hond, vogel, en nummer ze vanaf 0. De **one-hot**-vector van klasse $c$ heeft $K$ plekken, een 1 op plek $c$ en overal anders 0: kat is $(1; 0; 0)$, hond $(0; 1; 0)$ en vogel $(0; 0; 1)$. "Hot" is de ene plek die aan staat.

        In NumPy is `np.eye(K)` de eenheidsmatrix: $K$ rijen van $K$ getallen, met enen op de diagonaal. Rij $c$ daarvan is precies de one-hot-vector van klasse $c$. Een lijst of array met rijnummers tussen vierkante haken kiest die rijen, in die volgorde: `np.eye(3)[[1, 0]]` geeft de rijen van hond en kat. Dus `np.eye(K)[labels]` maakt de one-hot-vectoren van alle labels tegelijk. De weg terug is `.argmax()`: de plek van de 1 is de klasse.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **one-hot-vector**, **one-hot encoding** | | een klasse als vector met één plek per klasse: een 1 op de plek van de klasse, overal anders 0 | `[0, 1, 0]` |
        | $K$ | K | het aantal klassen | `k` |
        | **eenheidsmatrix** | | een vierkante matrix met enen op de diagonaal en verder nullen; rij $c$ is de one-hot-vector van klasse $c$ | `np.eye(3)` |
        | **rijen op nummer** | | een array of lijst met rijnummers tussen vierkante haken kiest die rijen, in die volgorde | `np.eye(3)[[1, 0]]` |
        """),
        L.watch_out("De volgorde van de klassen is een keuze die je één keer maakt, en waar je je daarna overal aan houdt. $(0; 1; 0)$ betekent alleen hond zolang iedereen het erover eens is dat plek 1 hond is."),
        draw.array_grids([("labels", [["1", "0", "2", "1"]], ["shape (4,)"]),
                          ("np.eye(3)[labels]", [["0", "1", "0"], ["1", "0", "0"], ["0", "0", "1"], ["0", "1", "0"]], ["shape (4, 3)", "kolommen: kat, hond, vogel"],
                           {(0, 1), (1, 0), (2, 2), (3, 1)})],
                         title="Eén rij per label, één kolom per klasse"),
        mo.md(r"""
        Vier labels, hond, kat, vogel, hond, worden vier rijen. Elke rij heeft zijn 1 in de kolom van zijn klasse.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo maak je van klassen one-hot-vectoren, en weer terug:

    1. Leg de volgorde van de $K$ klassen vast, en nummer ze vanaf 0.
    2. Voor één label: een vector van $K$ nullen, met een 1 op de plek van het label.
    3. Voor veel labels tegelijk: `np.eye(K)[labels]`, één rij per label.
    4. Terug naar de klasse: `.argmax()`, de plek van de 1; bij veel rijen `.argmax(axis=1)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Drie klassen, kat, hond en vogel, en vier labels: hond, kat, vogel, hond. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_one_hot():
    import numpy as np

    # 1. de volgorde: 0 = kat, 1 = hond, 2 = vogel
    _classes = ["kat", "hond", "vogel"]
    _labels = np.array([1, 0, 2, 1])
    # 2. één label, hond, met de hand: 3 plekken, met een 1 op plek 1
    _dog = []
    for _i in range(3):
        if _i == 1:
            _dog.append(1)
        else:
            _dog.append(0)
    print(_dog)
    # dezelfde vector is rij 1 van de eenheidsmatrix
    print(np.eye(3))
    # 3. alle labels tegelijk
    _onehot = np.eye(3)[_labels]
    print(_onehot)
    # 4. terug: de plek van de 1 in elke rij
    print(_onehot.argmax(axis=1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_dog` is `[0, 1, 0]`: drie plekken, met de 1 op plek 1, de plek van hond. `np.eye(3)` heeft de one-hot-vectoren van kat, hond en vogel als rijen, en rij 1 is diezelfde vector. `np.eye(3)[_labels]` pakt rij 1, 0, 2 en weer 1: de vier one-hot-vectoren, met een 1 in elke rij. NumPy schrijft `1.` en `0.`, omdat de eenheidsmatrix floats bevat. `.argmax(axis=1)` geeft de plek van de 1 in elke rij terug, `[1 0 2 1]`: de labels waarmee je begon.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_long():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoe lang?
    """)
    return


@app.cell(hide_code=True)
def choice_long_cell():
    choice_long = L.choice(
        "how-long",
        question="Een model leest handgeschreven cijfers, 0 tot en met 9. Hoeveel getallen heeft de one-hot-vector van één label?",
        options={
            "one": "1",
            "nine": "9",
            "right": "10",
            "four": "4: genoeg om 9 in nullen en enen te schrijven",
        },
        correct="right",
        why={
            "one": "Dat is het label zelf, het klassennummer. Hoeveel plekken heeft een one-hot-vector? Zie *Het idee*.",
            "nine": "Tel de cijfers van 0 tot en met 9: hoeveel klassen zijn er? Zie *Het idee*.",
            "four": "Dat is het getal binair schrijven, met meer plekken die aan staan. Een one-hot-vector heeft één plek per klasse, en er staat er maar één aan. Zie *Het idee*.",
            "right": "Tien klassen, dus tien plekken: negen nullen en één 1.",
        },
    )
    choice_long
    return (choice_long,)


@app.cell(hide_code=True)
def text_exercise_one():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén label, met de hand

        Maak `one_hot(label, k)` af: hij krijgt een klassennummer en het aantal klassen, en geeft de one-hot-vector terug als lijst, met gewone Python. Vervang de `...`. De check roept `one_hot(1, 3)` aan: dat moet `[0, 1, 0]` geven. Hij probeert ook andere labels en andere aantallen klassen.
        """),
        L.task("Laat `one_hot` een lijst van `k` getallen teruggeven, met een 1 op plek `label`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one():
    box_one = L.code_box("one-hot", lines=8)
    box_one
    return (box_one,)


@app.cell(hide_code=True)
def check_one(box_one):
    result_one = L.check("one-hot", box_one.value)
    result_one
    return (result_one,)


@app.cell(hide_code=True)
def hints_one():
    L.hints(
        "one-hot",
        [
            "Stap 2 van het recept: een plek voor elke klasse, allemaal 0, behalve de plek van het label.",
            "Een loop over de plekken `i` van 0 tot `k`. Op welke plek voeg je een 1 toe, en hoe test je daarop?",
            "`for i in range(k):`, dan `if i == ___:` met `vector.append(1)`, en onder `else:` `vector.append(___)`.",
        ],
        why="De loop gaat langs alle `k` plekken; alleen bij `i == label` voegt hij een 1 toe. Voor `one_hot(1, 3)`: plek 0, 1 en 2 geven 0, 1 en 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Tellen met one-hot-vectoren

        Vijf handgeschreven cijfers hebben de labels in `labels`. Maak hun one-hot-vectoren met NumPy, en tel ze per klasse op, zodat `counts` tien getallen bevat: hoe vaak elk cijfer voorkomt. Vervang de `...`. De laatste regel is `counts`, zodat de check het ziet. De check probeert ook andere labels.
        """),
        L.task("Zet in `counts` hoe vaak elk van de tien cijfers voorkomt, door de one-hot-vectoren op te tellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-classes", lines=6)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-classes", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-classes",
        [
            "Stap 3 van het recept geeft één rij per label. In welke richting tel je op, om één getal per klasse te krijgen?",
            "Hoeveel rijen heeft `np.eye` nodig bij tien cijfers? Tel daarna elke kolom op, zoals in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "`counts = np.eye(___)[labels].sum(axis=___)`.",
        ],
        why="`np.eye(10)[labels]` is een matrix van 5 rijen en 10 kolommen; `.sum(axis=0)` telt elke kolom op, één telling per cijfer: `[1, 0, 0, 2, 0, 0, 0, 1, 0, 1]`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl) waren klassennummers alleen namen; een one-hot-vector geeft elke klasse een eigen plek, zodat er niets meekomt wat niet klopt. In [Delen van een array kiezen](/course/open/np-indexing?lang=nl) koos je één rij op zijn nummer; nu kiest een hele lijst met rijnummers veel rijen tegelijk. Later in de course geeft een model voor veel klassen één kans per klasse, en die kansen moeten dicht bij de one-hot-vector van het label komen. In de volgende les, *Veel klassen: softmax*, maak je van de scores van een model zulke kansen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long, result_count, result_one):
    L.wrap_up(choice_long, result_one, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
