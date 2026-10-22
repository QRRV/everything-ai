# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-validation-set. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow residuals

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-validation-set", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_which():
    quick_which = L.choice(
        "quick-which",
        question="`w, b = fit(x, y, lr, steps)` traint een lijn. Welke hiervan zijn hyperparameters?",
        options={
            "params": "`w` en `b`",
            "data": "`x` en `y`",
            "right": "`lr` en `steps`",
        },
        correct="right",
        why={
            "params": "Die komen uit het trainen: het zijn parameters. Zie *Het idee* in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).",
            "data": "Dat zijn de trainingsvoorbeelden, geen instellingen. Zie *Het idee* in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).",
            "right": "`lr` en `steps` kies je vóór het trainen, en het trainen verandert ze niet.",
        },
    )
    quick_which
    return (quick_which,)


@app.cell(hide_code=True)
def quick_check_how():
    quick_how = L.choice(
        "quick-how",
        question="Hoe kies je een hyperparameter zoals de graad van een curve?",
        options={
            "right": "Train met elke waarde een model, en vergelijk ze op voorbeelden waarop ze niet trainden",
            "train": "Neem de waarde met de laagste MSE op de training set",
            "gradient": "Laat gradient descent hem veranderen, net als een weight",
        },
        correct="right",
        why={
            "train": "Op de training set ziet meer capaciteit er altijd beter uit. Zie *Het idee* in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).",
            "gradient": "Het trainen verandert een hyperparameter niet. Zie *Het idee* in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).",
            "right": "Elke waarde geeft een eigen getraind model; voorbeelden die het niet zag, laten zien welk model werkt op nieuwe data.",
        },
    )
    quick_how
    return (quick_how,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een hyperparameter kies je op voorbeelden waarop het model niet trainde. De test set lijkt daar de voor de hand liggende plek voor, maar de test set heeft een andere taak: één keer eerlijk zeggen hoe het uiteindelijke model het doet op nieuwe data. Kies je ermee, dan heeft hij het model meegevormd, en ziet zijn score er beter uit dan het model echt is. Daarom houd je een derde set voorbeelden apart, alleen om mee te kiezen.

    **Een voorbeeld uit het dagelijks leven.** Een school probeert drie manieren om wiskunde te geven en houdt de manier waarmee de leerlingen het best scoren op een proefexamen; het echte examen blijft verzegeld tot het eind. Kiest de school met het echte examen, dan beloont dat cijfer ook geluk met die vragen, en lijkt de gekozen manier beter dan ze is.
    """)
    return


@app.cell(hide_code=True)
def predict_best_degree():
    guess_best_degree = L.guess(
        "best-degree",
        question="Negen curves van graad 1 tot en met 9, elk getraind op dezelfde 12 temperaturen, worden gemeten op 10 andere temperaturen. Welke graad krijgt daar volgens jou de laagste MSE?",
        options={
            "line": "1",
            "two": "2",
            "nine": "9",
        },
        reveal="**Graad 2**, met een MSE van ongeveer 0,80 op die 10 temperaturen. Op de 12 trainingstemperaturen had graad 9 gewonnen, met 0,49, maar op de andere 10 krijgt hij ongeveer 36: hij overfit. De regel waarmee de temperaturen gemaakt zijn, heeft graad 2.",
    )
    guess_best_degree
    return (guess_best_degree,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Splits de voorbeelden na het schudden in drie sets, vaak ongeveer 60%, 20% en 20%:

        - de **training set**, om de parameters te leren;
        - de **validation set**, om de hyperparameters te kiezen: train met elke waarde een model op de training set, en houd het model met de laagste loss op de validation set;
        - de **test set**, om het gekozen model één keer te meten, aan het eind.

        De validation set bestaat uit voorbeelden waarop het model niet trainde, dus laat hij zien welke waarde werkt op nieuwe data. De test set blijft erbuiten tot alles gekozen is, dus is zijn score een eerlijke schatting.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **validation set** | | voorbeelden die je naast de training set apart houdt, om hyperparameters mee te kiezen | `x_val`, `y_val` |
        """),
        L.watch_out("Ook de validation-score van het gekozen model is iets te optimistisch: wie uit een paar modellen de winnaar kiest, kiest deels het model dat op die voorbeelden het meeste geluk had. Daarom meet je aan het eind toch op de test set, en meld je dat getal."),
        mo.md(r"""
        De temperaturen uit het uitgewerkte voorbeeld hieronder, gesplitst in 12 om te trainen, 10 om te valideren en 10 om te testen. Voor elke graad de <span class="c-loss">MSE</span> op de training set (gevulde stippen) en op de validation set (open stippen).
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_degrees():
    _shown = {_d: L.number(_e["val"], 0) for _d, _e in C.ERRORS.items()}
    mo.Html(C.picture({"title": "Training- en validation-MSE per graad", "degree": "graad", "mse": "MSE", "chosen": "gekozen",
                       "alt": "De MSE van graad 1 tot en met 9. Op de training set daalt hij van 5,2 naar 0,49. Op de validation set daalt hij van 8,5 naar 0,80 bij graad 2, de gekozen graad, en stijgt dan tot ongeveer 36 bij graad 9."}, _shown))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kies je een hyperparameter en meld je een eerlijke score:

    1. Schud de voorbeelden en splits ze in een training set, een validation set en een test set.
    2. Voor elke waarde van de hyperparameter: train op de training set, en meet op de validation set.
    3. Houd de waarde met de laagste validation-loss: `np.argmin` geeft zijn positie in de lijst.
    4. Meet het gekozen model één keer op de test set, en meld dat getal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De 32 temperaturen uit [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl), nu in drieën gesplitst: dezelfde 12 trainingstemperaturen als daar, met opzet weinig zodat een flexibele curve eerder overfit, en de 20 testtemperaturen van daar, verdeeld over 10 om te valideren en 10 om te testen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_validation():
    import numpy as np

    _rng = np.random.default_rng(75)
    _hours = 24 * _rng.random(size=32)
    _temps = 20 - 0.05 * (_hours - 15) ** 2 + 3 * (_rng.random(size=32) - 0.5)

    # 1. schud en splits: 12 om te trainen, 10 om te valideren, 10 om te testen
    _order = _rng.permutation(32)
    _train, _val, _test = _order[:12], _order[12:22], _order[22:]
    # 2. voor elke graad: train op de training set, meet op de validation set
    _degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    _val_mses = []
    for _degree in _degrees:
        _w = np.polyfit(_hours[_train], _temps[_train], _degree)
        _val_mses.append(np.mean((_temps[_val] - np.polyval(_w, _hours[_val])) ** 2))
    print(np.array(_val_mses))
    # 3. de graad met de laagste validation-MSE
    _best = _degrees[np.argmin(_val_mses)]
    print(_best)
    # 4. het gekozen model, één keer op de test set
    _w = np.polyfit(_hours[_train], _temps[_train], _best)
    print(np.mean((_temps[_test] - np.polyval(_w, _hours[_test])) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_train, _val, _test = _order[:12], _order[12:22], _order[22:]` geeft elk van de drie slices een eigen naam, zoals `x, y = point` deed in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl): de komma's rechts maken een tuple, ook zonder haakjes. De validation-MSE daalt van ongeveer 8,5 bij graad 1 naar 0,80 bij graad 2, en klimt dan tot ongeveer 36 bij graad 9. `np.argmin(_val_mses)` is de `.argmin()` uit [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl), geschreven als functie, en die neemt ook een gewone lijst: hij geeft positie 1, en `_degrees[1]` is graad 2.

    De curve van graad 2 krijgt ongeveer 0,37 op de test set: dat is het getal dat je meldt. Dat het lager is dan de 0,80 op de validation set, is toeval: met maar 10 temperaturen per set schommelt een MSE flink van set tot set. Het let-op gaat over wat er gemiddeld gebeurt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_test_to_choose():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Kiezen op de test set
    """)
    return


@app.cell(hide_code=True)
def choice_test_to_choose_cell():
    choice_test_to_choose = L.choice(
        "test-to-choose",
        question="Iemand slaat de validation set over: die houdt de graad met de laagste test-MSE, graad 3 met 0,34, en meldt 0,34. Wat is er mis met dat getal?",
        options={
            "nothing": "Niets: de test set bestaat uit voorbeelden waarop het model niet trainde",
            "right": "Het is waarschijnlijk te optimistisch: de test set hielp het model kiezen",
            "pessimistic": "Het is waarschijnlijk te pessimistisch: het model trainde niet op de test set",
        },
        correct="right",
        why={
            "nothing": "Het trainde er niet op, maar ze kozen wel de graad. Zie *Waarom*.",
            "pessimistic": "Van negen modellen is het model gehouden dat het op deze voorbeelden het best deed. Maakt dat zijn score te hoog of te laag? Zie het let-op in *Het idee*.",
            "right": "Van negen curves is de curve gehouden die het toevallig het best deed op deze 10 temperaturen, dus 0,34 is te mooi. Er is geen eerlijke testscore meer over.",
        },
    )
    choice_test_to_choose
    return (choice_test_to_choose,)


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kies de graad

        Twaalf trainingsmetingen in `x_train` (uren) en `y_train` (temperaturen), zes metingen om te valideren in `x_val` en `y_val`, en de graden om uit te kiezen in `degrees`. Pas in de loop een curve van elke graad aan op de training set en voeg zijn validation-MSE toe aan `val_mses`. Zet daarna de graad met de laagste validation-MSE in `best`. Vervang de twee `...`. De laatste regel is `best`, zodat de check het ziet. De check probeert ook andere graden om uit te kiezen.
        """),
        L.task("Zet de graad met de laagste validation-MSE in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose-degree", lines=13)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose-degree", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose-degree",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld.",
            "In de loop: `np.polyfit` op de training set, dan de MSE op `x_val` en `y_val`. Na de loop: `np.argmin` geeft een positie. Wat staat er op die positie in `degrees`?",
            "`val_mses.append(np.mean((y_val - np.polyval(w, ___)) ** 2))` en `best = degrees[___]`.",
        ],
        why="De validation-MSE's zijn ongeveer 7,87, 0,96, 1,04, 1,01, 1,16 en 2,21. De laagste staat op positie 1, en `degrees[1]` is 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ### Oefening 3. En dan?
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "test-at-end",
        question="Je koos graad 2 op de validation set. Wat doe je nu met de test set?",
        options={
            "again": "Vergelijk er graad 2 en 3 op, en houd de betere",
            "right": "Meet de curve van graad 2 er één keer op, en meld dat getal",
            "train": "Voeg hem toe aan de training set en train opnieuw, zonder te meten",
        },
        correct="right",
        why={
            "again": "Dan kiest de test set toch. Zie stap 4 van het recept.",
            "train": "Dan is er niets meer over om het model eerlijk te meten. Zie stap 4 van het recept.",
            "right": "De test set heeft nog geen rol gespeeld, dus zijn score is een eerlijke schatting voor nieuwe data.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl) zag je dat de training-loss geen hyperparameter kan kiezen; de validation set kan dat wel. In [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl) liet de test-MSE het patroon zien; nu blijft de test set verzegeld terwijl je kiest, dus is zijn ene score aan het eind eerlijk. In de volgende les, *Vraag het de buren: k-nearest neighbors*, leer je een classifier kennen waarvan de hyperparameter het aantal buren is dat hij raadpleegt, en die kies je op dezelfde manier.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_to_choose, result_choose, choice_end):
    L.wrap_up(choice_test_to_choose, result_choose, choice_end)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
