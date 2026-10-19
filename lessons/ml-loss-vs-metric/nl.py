# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-loss-vs-metric. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-loss-vs-metric", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) en [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_report():
    quick_report = L.choice(
        "quick-report",
        question="Je koos een graad op de validation set, waar hij een MSE van 0,80 kreeg. Welk getal meld je als schatting voor nieuwe data?",
        options={
            "val": "0,80, zijn validation-MSE",
            "train": "Zijn training-MSE",
            "right": "Zijn MSE op de test set",
        },
        correct="right",
        why={
            "val": "De validation set hielp kiezen, dus zijn score is te mooi voor de winnaar. Zie het let-op in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "train": "Van die voorbeelden heeft het model geleerd. Zie *Het idee* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "right": "De test set speelde geen rol bij het trainen of kiezen, dus zijn MSE is een eerlijke schatting.",
        },
    )
    quick_report
    return (quick_report,)


@app.cell(hide_code=True)
def quick_check_cross_entropy():
    quick_cross_entropy = L.choice(
        "quick-cross-entropy",
        question="Een model geeft een foto de kansen 0,1, 0,6 en 0,3, en zijn goede klasse is 1. Hoe groot is zijn cross-entropy ongeveer?",
        options={
            "right": "0,51",
            "first": "2,30",
            "last": "1,20",
        },
        correct="right",
        why={
            "first": "Dat is $-\\log(0{,}1)$, de kans op klasse 0. Waar begin je te tellen? Zie *Het idee* in [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "last": "Dat is $-\\log(0{,}3)$, de kans op klasse 2. Zie *Het idee* in [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "right": "Klasse 1 kreeg 0,6: $-\\log(0{,}6) \\approx 0{,}51$.",
        },
    )
    quick_cross_entropy
    return (quick_cross_entropy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Gradient descent heeft een loss met bruikbare hellingen nodig, zoals cross-entropy. Maar wie een spamfilter gebruikt, geeft niets om zijn cross-entropy: die wil weten hoe vaak het filter goed beslist. Dat is een ander getal, en het kan anders bewegen. Je traint dus op het ene getal en meldt het andere, en je moet weten welk getal waarvoor is.

    **Een voorbeeld uit het dagelijks leven.** In rijlessen zegt de instructeur na elke bocht hoe ver je ernaast zat: iets te ruim, wat te snel. Het rijexamen zegt alleen geslaagd of gezakt. Van het eerste leer je; anderen kijken naar het tweede.
    """)
    return


@app.cell(hide_code=True)
def predict_better():
    guess_better = L.guess(
        "better",
        question="Op vijf foto's heeft model B een veel lagere cross-entropy dan model A: het geeft op de meeste foto's de goede klasse een veel hogere kans. Heeft B volgens jou ook meer van de vijf foto's goed?",
        options={
            "more": "Ja, duidelijk meer",
            "same": "Niet per se: het kunnen er evenveel zijn",
            "fewer": "Nee, minder",
        },
        reveal="**Niet per se.** In het uitgewerkte voorbeeld hieronder hebben ze allebei vier van de vijf foto's goed, terwijl de cross-entropy ongeveer 0,77 is voor A en 0,31 voor B. Of een foto goed telt, hangt alleen af van welke klasse de waarschijnlijkste is, niet van hoe zeker het model is.",
    )
    guess_better
    return (guess_better,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **metric** is een getal dat zegt hoe goed een model is, gemeten op een manier die mensen belangrijk vinden. De bekendste bij classificatie is **accuracy**: het deel van de voorbeelden waarvan de voorspelde klasse het label is. De voorspelde klasse is de waarschijnlijkste, `probs.argmax(axis=1)`; die vergelijken met `labels` geeft per voorbeeld `True` of `False`, en omdat `True` als 1 telt en `False` als 0, is het gemiddelde daarvan het deel dat goed is.

        Je traint op de loss en meldt de metric, en het zijn niet voor niets verschillende getallen. Een heel kleine verandering in een weight verandert de loss een heel klein beetje, dus de loss heeft hellingen voor gradient descent. De accuracy verandert pas als een voorbeeld naar een andere klasse overgaat; bij bijna elke kleine verandering blijft hij precies gelijk, dus zijn helling is bijna overal 0, en gradient descent heeft niets om op af te gaan. Meestal bewegen de twee samen, maar niet altijd: een model kan zekerder worden van zijn goede antwoorden, en beter op de loss, zonder dat er één voorspelling verandert.

        Kies op de validation set met de metric die ertoe doet, en meld hem op de test set, net als bij de loss.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **metric** | | een getal dat zegt hoe goed een model is, gemeten op een manier die mensen belangrijk vinden; je meldt het, je traint er niet op | |
        | **accuracy** | | het deel van de voorbeelden dat een classifier goed heeft | `np.mean(probs.argmax(axis=1) == labels)` |
        """),
        L.watch_out("Accuracy kan misleiden als één klasse veel vaker voorkomt. Zijn 95 van de 100 mails geen spam, dan heeft een filter dat nooit spam zegt een accuracy van 0,95. Een latere les laat getallen zien die dat wel opmerken."),
        mo.md(r"""
        Tien leerlingen, hun uren leren $x$, en of ze slaagden. Een model geeft de kans op slagen, met een vaste weight van 1:
        """),
        L.formula(r"[[pred:\hat{y}]] = \sigma(x + [[param:b]])"),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek van de log loss: dat zet de <span class="c-param">bias term</span> $b$. De ene grafiek toont de <span class="c-loss">log loss</span>, de andere de accuracy: een kans van 0,5 of meer telt als geslaagd.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_b_cell():
    def _frame(b):
        b = round(b, 1)
        right = C.right(b)
        return (C.picture(b, {"loss_title": "Log loss", "loss": "log loss", "acc_title": "Accuracy", "acc": "accuracy",
                              "alt": f"De log loss voor b van −8 tot 0, een glad dal met zijn laagste punt bij ongeveer b = −4,2; bij b = {L.number(b, 1)} is hij {L.number(C.log_loss(b), 2)}. De accuracy voor dezelfde b, een trap van vlakke treden die tussen 0,5 en 0,8 op en neer gaan, het hoogst van b = −4,5 tot −4,1; bij b = {L.number(b, 1)} is hij {L.number(right / 10, 1)}."}),
                f"""Met <i>b</i> = {L.number(b, 1)}: een log loss van <span class="c-loss ea-num">{L.number(C.log_loss(b), 3)}</span>, en een accuracy van <span class="ea-num">{L.number(right / 10, 1)}</span>: het model heeft {right} van de 10 leerlingen goed.""")

    explore_b = L.explore("bias", "b", _frame, start=-5.5, low=-8, high=0, step=0.1, role="param", task="Zoek de $b$ met de laagste log loss, en de $b$ met de hoogste accuracy.")
    explore_b
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De log loss heeft één laagste punt, bij ongeveer $b = -4{,}2$, en bij elke $b$ wijst zijn helling de weg daarheen. De accuracy is het hoogst, 0,8, op een hele vlakke trede van $b = -4{,}5$ tot $-4{,}1$, en springt alleen waar een leerling over 0,5 gaat: een trap, zonder helling om te volgen.

    ### Recept

    Zo beoordeel je een classifier met zijn accuracy:

    1. Neem de voorspelde klasse van elk voorbeeld: de waarschijnlijkste, `probs.argmax(axis=1)`.
    2. Vergelijk met de labels: `== labels` geeft `True` waar de voorspelling goed is.
    3. Neem het gemiddelde: het deel goede voorspellingen.
    4. Train op de loss; kies op de validation set en meld op de test set met de metric die ertoe doet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vijf foto's, drie klassen, en twee modellen, A en B, die elke foto zijn kansen geven. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_metric():
    import numpy as np

    _labels = np.array([0, 1, 2, 1, 0])
    _a = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5], [0.45, 0.35, 0.2], [0.6, 0.2, 0.2]])
    _b = np.array([[0.9, 0.05, 0.05], [0.45, 0.35, 0.2], [0.05, 0.05, 0.9], [0.1, 0.85, 0.05], [0.9, 0.05, 0.05]])

    for _probs in [_a, _b]:
        # 1. de waarschijnlijkste klasse van elke foto
        _predicted = _probs.argmax(axis=1)
        print(_predicted)
        # 2. en 3. goed of niet, en het deel dat goed is
        print(_predicted == _labels, np.mean(_predicted == _labels))
        # de cross-entropy, om te vergelijken
        print(-(np.eye(3)[_labels] * np.log(_probs)).sum(axis=1).mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Model A voorspelt de klassen 0, 1, 2, 0 en 0, en heeft er vier goed: een accuracy van 0,8. Model B voorspelt 0, 0, 2, 1 en 0: het heeft een andere foto fout, maar ook vier goed, dus ook 0,8. De cross-entropy houdt ze uit elkaar: ongeveer 0,77 voor A, en 0,31 voor B, dat op vier foto's de goede klasse 0,85 of 0,9 geeft, waar A hem nooit meer dan 0,6 geeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_slope():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Geen helling
    """)
    return


@app.cell(hide_code=True)
def choice_slope_cell():
    choice_slope = L.choice(
        "no-slope",
        question="Waarom traint gradient descent niet rechtstreeks op accuracy?",
        options={
            "number": "Accuracy is geen getal dat je kunt uitrekenen",
            "right": "Een heel kleine verandering in een weight verandert bijna nooit een voorspelling, dus de helling van accuracy is bijna overal 0",
            "big": "Accuracy kan groter worden dan 1",
        },
        correct="right",
        why={
            "number": "Het is een getal: een deel tussen 0 en 1. Wat heeft gradient descent ervan nodig? Zie *Het idee*.",
            "big": "Een deel van de voorbeelden is nooit meer dan 1. Kijk naar de grafiek van de accuracy: wat ontbreekt er aan een trap?",
            "right": "Vlakke treden geven gradient descent geen richting; de loss, die bij elk klein stapje verandert, wel.",
        },
    )
    choice_slope
    return (choice_slope,)


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Accuracy

        Zes foto's en drie klassen: de kansen van elke foto zijn een rij van `probs`, en de labels staan in `labels`. Zet de accuracy in `accuracy`. Vervang de `...`. De laatste regel is `accuracy`, zodat de check het ziet. De check probeert ook andere labels.
        """),
        L.task("Zet het deel van de foto's waarvan de waarschijnlijkste klasse het label is in `accuracy`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("accuracy", lines=7)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Stap 1 tot en met 3 van het recept, in één regel.",
            "Welke methode geeft de plek van het grootste getal, en langs welke axis staan de klassen van één foto? Wat geeft het gemiddelde van `True` en `False`?",
            "`accuracy = np.mean(probs.argmax(axis=___) == ___)`.",
        ],
        why="De waarschijnlijkste klassen zijn 0, 2, 1, 0, 2 en 1; vergeleken met de labels 0, 2, 1, 1, 0 en 1 zijn er vier goed. Vier van de zes is ongeveer 0,67.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 3. Welke?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-model",
        question="Twee spamfilters op de validation set. Filter A: cross-entropy 0,40, accuracy 0,85. Filter B: cross-entropy 0,45, accuracy 0,88. Wat telt, is hoe vaak het filter goed beslist. Welke kies je?",
        options={
            "a": "Filter A: zijn loss is lager",
            "right": "Filter B: dat beslist vaker goed",
            "test": "Nog geen van beide: vergelijk ze eerst op de test set",
        },
        correct="right",
        why={
            "a": "De loss is waarop elk filter trainde. Waar vroegen de mensen die het gebruiken om? Zie *Het idee*.",
            "test": "De test set blijft verzegeld terwijl je kiest. Zie het recept.",
            "right": "Kies op de validation set met de metric die ertoe doet; meld daarna de accuracy van B op de test set.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) leerde je de loss kennen waarop een classifier traint; accuracy is de metric die je ernaast meldt, en die kijkt alleen naar de waarschijnlijkste klasse. In [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) koos je op de ene set en meldde je op de andere; met een metric gaat het net zo. In de volgende les, *Meer dan accuracy: precision en recall*, zie je welke klassen een model door elkaar haalt, en waarom accuracy alleen kan misleiden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_slope, result_accuracy, choice_which):
    L.wrap_up(choice_slope, result_accuracy, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
