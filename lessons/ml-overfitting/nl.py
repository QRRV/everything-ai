# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-overfitting. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-overfitting", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shuffle():
    quick_shuffle = L.choice(
        "quick-shuffle",
        question="Waarom schud je de voorbeelden voordat je ze splitst in een training set en een test set?",
        options={
            "faster": "Zodat het trainen sneller gaat",
            "right": "Zodat de test set op de rest lijkt, en niet bijvoorbeeld alleen het laatste stuk is",
            "twice": "Zodat het model elk voorbeeld twee keer ziet",
        },
        correct="right",
        why={
            "faster": "Schudden verandert welke voorbeelden waar terechtkomen, niet hoe snel het trainen gaat. Zie *Het idee* in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).",
            "twice": "Na het splitsen zit elk voorbeeld in precies één van de twee sets. Zie *Het idee* in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).",
            "right": "Data staat vaak gesorteerd; na het schudden lijken beide sets op het geheel.",
        },
    )
    quick_shuffle
    return (quick_shuffle,)


@app.cell(hide_code=True)
def quick_check_test_once():
    quick_test_once = L.choice(
        "quick-test-once",
        question="Wanneer meet je de loss op de test set?",
        options={
            "right": "Eén keer, aan het eind",
            "every-step": "Na elke trainingsstap",
            "choose": "Telkens als je tussen modellen kiest",
        },
        correct="right",
        why={
            "every-step": "Dan zou de test set het trainen sturen. Zie het let-op in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).",
            "choose": "Een test set die je hielp kiezen, is geen eerlijke schatting meer. Zie het let-op in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).",
            "right": "Eén keer, aan het eind: dan is het een eerlijke schatting voor nieuwe data.",
        },
    )
    quick_test_once
    return (quick_test_once,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model kan er op twee tegengestelde manieren naast zitten. Is het te eenvoudig, dan mist het het patroon, zoals een rechte lijn door temperaturen die over een dag stijgen en dalen. Is het te flexibel, dan leert het ook de toevallige schommelingen van zijn trainingsvoorbeelden, die in nieuwe data niet terugkomen. De training set en de test set uit de vorige les vertellen welke van de twee je hebt, en dus welke kant je het model op moet veranderen.

    **Een voorbeeld uit het dagelijks leven.** Een weervoorspeller die altijd "morgen 12 graden" zegt, negeert zomer en winter. Een weervoorspeller die onthoudt dat het vorig jaar op deze dag om drie uur 14,3 graden was, ziet toeval aan voor een regel.
    """)
    return


@app.cell(hide_code=True)
def predict_train_mse():
    guess_train_mse = L.guess(
        "train-mse",
        question="Je geeft een curve stap voor stap meer vrijheid om te buigen, en traint hem elke keer op dezelfde 12 temperaturen. Wat gebeurt er volgens jou met zijn MSE op die 12 trainingstemperaturen?",
        options={
            "down": "Die blijft dalen",
            "u": "Die daalt en stijgt dan weer",
            "same": "Die blijft ongeveer gelijk",
        },
        reveal="**Die blijft dalen.** In het uitgewerkte voorbeeld hieronder van 5,2 voor een rechte lijn tot 0,49 voor de meest flexibele curve. Een curve met meer vrijheid kan altijd minstens wat een stijvere kon, dus op de training set wordt hij nooit slechter. Op de test set daalt de MSE en stijgt hij dan weer: van 6,8 naar 0,6, en dan naar 21,9.",
    )
    guess_train_mse
    return (guess_train_mse,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **capaciteit** van een model is hoe ingewikkeld de patronen zijn die het kan volgen. Hier hangt die af van de **graad** van een curve: lineaire regressie op de features $x; x^2; \dots; x^d$ van één getal $x$ geeft een curve $w_1 x + w_2 x^2 + \dots + w_d x^d + b$, een **polynoom** (veelterm) van graad $d$. Graad 1 is een rechte lijn; elke macht erbij laat de curve één keer meer buigen.

        Metingen bevatten ook **ruis**: toevallige schommelingen die geen regel voorspelt, zoals een wolk die net voorbijdrijft als een temperatuur wordt afgelezen. Vergelijk de MSE op de training set en op de test set:

        - **Underfitting**: allebei hoog. Het model is te eenvoudig voor het patroon. Geef het meer capaciteit.
        - **Overfitting**: de training-MSE is laag, maar de test-MSE veel hoger. Het model volgt de ruis van zijn trainingsvoorbeelden. Geef het minder capaciteit, of meer trainingsvoorbeelden.
        - **Ertussenin**: allebei laag en dicht bij elkaar. Op nieuwe data komt geen model onder de ruis zelf.

        `np.polyfit(x, y, d)` vindt de weights en de bias term van de polynoom van graad $d$ met de laagste MSE op `x` en `y`, de waarden waar gradient descent na heel veel stappen op uitkomt; het rekent ze in één keer uit, zonder stappen. `np.polyval(w, x)` rekent de curve uit bij de getallen in `x`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **capaciteit** van een model | | hoe ingewikkeld de patronen zijn die een model kan volgen; meer parameters of flexibelere features geven meer capaciteit | |
        | **polynoom** van graad $d$ | | $w_1 x + w_2 x^2 + \dots + w_d x^d + b$: een som van machten van $x$, elk met een eigen weight, plus een bias term | `np.polyfit(x, y, d)`, `np.polyval(w, x)` |
        | **ruis** | | toevallige schommelingen in de data die geen regel voorspelt | |
        | **underfitting** | | een model dat te eenvoudig is voor het patroon: een hoge MSE op de training set en op de test set | |
        | **overfitting** | | een model dat de ruis van zijn training set volgt: een lage training-MSE, een veel hogere test-MSE | |
        """),
        L.watch_out("Een training-MSE dicht bij 0 is op zich geen goed nieuws: met ongeveer evenveel te leren getallen als trainingsvoorbeelden kan een curve door bijna elk punt gaan. Kijk altijd ook naar de test-MSE."),
        mo.md(r"""
        De ene grafiek toont de temperaturen uit het uitgewerkte voorbeeld hieronder, met een <span class="c-pred">curve</span> van graad $d$: gevulde stippen zijn de 12 trainingsmetingen, open stippen de 20 testmetingen. De andere toont de <span class="c-loss">MSE</span> van elke graad: gevulde stippen op de training set, open stippen op de test set, groter bij graad $d$. Verschuif de slider, of sleep over de grafiek van de MSE, om $d$ te veranderen.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_degree_cell():
    def _frame(d):
        d = int(d)
        train, test = C.ERRORS[d]
        return (C.picture(d, {"data": f"Graad {d}", "hour": "uur van de dag", "temp": "temperatuur (°C)",
                              "mse": "MSE per graad", "degree": "graad", "loss": "MSE",
                              "alt": f"32 temperaturen over een dag, oplopend tot ongeveer 20 °C in de middag, met een curve van graad {d}. De MSE van graad 1 tot en met 9, de stippen van graad {d} groter: op de training set daalt hij van 5,2 naar 0,49; op de test set daalt hij tot ongeveer 0,6 bij graad 2, blijft hij daar dichtbij tot en met graad 5, en stijgt hij tot 21,9 bij graad 9."}),
                f"""Graad {d}: een MSE van <span class="c-loss ea-num">{L.number(train, 2)}</span> op de training set en <span class="c-loss ea-num">{L.number(test, 2)}</span> op de test set.""")

    explore_degree = L.explore("degree", "d", _frame, start=1, low=1, high=9, step=1, role="", task="Zoek de graad met de laagste MSE op de test set.")
    explore_degree
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De MSE op de test set is het laagst bij graad 2, en vanaf graad 6 stijgt hij snel. De MSE op de training set wordt steeds kleiner, tot en met graad 9: een lagere MSE op de training set betekent geen beter model.

    ### Recept

    Zo zoek je uit of een model underfit of overfit:

    1. Train het op de training set.
    2. Meet zijn MSE op de training set en op de test set.
    3. Allebei hoog: underfitting. Geef het model meer capaciteit, zoals een hogere graad of meer features.
    4. Training-MSE laag, test-MSE veel hoger: overfitting. Geef het minder capaciteit, of meer trainingsvoorbeelden.
    5. Verander je het model vanwege de test-MSE, dan hielp de test set je kiezen, en is zijn MSE geen eerlijke schatting voor nieuwe data meer. Een latere les houdt voor die keuze een derde set apart; hier laat de test set alleen het patroon zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Tweeëndertig verzonnen temperaturen, afgelezen op willekeurige uren $h$ van één dag: ze volgen de regel $20 - 0{,}05\,(h - 15)^2$, het hoogst om drie uur 's middags, plus ruis van hooguit 1,5 °C omhoog of omlaag. Twaalf ervan zijn de training set, de andere twintig de test set: met opzet weinig trainingsvoorbeelden, zodat een flexibele curve eerder overfit. Dan drie curves: graad 1, 2 en 9. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_overfitting():
    import numpy as np

    _rng = np.random.default_rng(75)
    _hours = 24 * _rng.random(size=32)
    _temps = 20 - 0.05 * (_hours - 15) ** 2 + 3 * (_rng.random(size=32) - 0.5)
    _order = _rng.permutation(32)
    _x_train = _hours[_order[:12]]
    _y_train = _temps[_order[:12]]
    _x_test = _hours[_order[12:]]
    _y_test = _temps[_order[12:]]

    for _degree in [1, 2, 9]:
        # 1. train een curve van deze graad op de training set
        _w = np.polyfit(_x_train, _y_train, _degree)
        # 2. zijn MSE op de training set en op de test set
        _mse_train = np.mean((_y_train - np.polyval(_w, _x_train)) ** 2)
        _mse_test = np.mean((_y_test - np.polyval(_w, _x_test)) ** 2)
        print(_degree, _mse_train, _mse_test)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Graad 1, een rechte lijn: ongeveer 5,2 op de training set en 6,8 op de test set, allebei hoog. Die underfit: een lijn kan niet stijgen en dalen. Graad 2: ongeveer 0,76 en 0,58, allebei laag en dicht bij elkaar, vlak bij de ruis zelf; de regel waarmee de temperaturen gemaakt zijn, heeft ook graad 2. Graad 9: slechts 0,49 op de training set, maar ongeveer 21,9 op de test set. Die overfit: hij buigt door de toevallige schommelingen van zijn 12 trainingsmetingen, en daartussen en daarbuiten slingert hij er ver naast.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke overfit?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-overfits",
        question="Drie modellen voorspellen huizenprijzen. Hun MSE op de training set en op de test set: A heeft 900 en 950, B heeft 400 en 450, C heeft 20 en 1300. Welk model overfit?",
        options={
            "a": "A",
            "b": "B",
            "right": "C",
        },
        correct="right",
        why={
            "a": "Beide MSE's van A zijn hoog en liggen dicht bij elkaar. Op welke van de twee wijst dat? Zie *Het idee*.",
            "b": "Beide MSE's van B zijn laag en liggen dicht bij elkaar. Zie *Het idee*.",
            "right": "C zit op de huizen uit zijn training set er bijna precies op, maar op nieuwe huizen ver naast: het volgt de ruis. A is op allebei hoog: dat underfit.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_test_mse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De test-MSE van een curve

        Twaalf trainingsmetingen en zes testmetingen, en een graad in `degree`. Laat `np.polyfit` een curve van die graad leren van de training set, en zet zijn MSE op de test set in `mse_test`. Vervang de twee `...`. De laatste regel is `mse_test`, zodat de check het ziet. De check probeert ook een andere graad. Benieuwd naar graad 9? Kopieer je code naar het kladblok en verander daar `degree`.
        """),
        L.task("Zet de test-MSE van een curve van graad `degree`, geleerd van de training set, in `mse_test`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_test_mse():
    box_test_mse = L.code_box("test-mse", lines=11)
    box_test_mse
    return (box_test_mse,)


@app.cell(hide_code=True)
def check_test_mse(box_test_mse):
    result_test_mse = L.check("test-mse", box_test_mse.value)
    result_test_mse
    return (result_test_mse,)


@app.cell(hide_code=True)
def hints_test_mse():
    L.hints(
        "test-mse",
        [
            "Stap 1 en 2 van het recept, maar alleen de test-MSE.",
            "`np.polyfit` krijgt de uren en temperaturen van de training set en de graad. Welke uren gaan in `np.polyval` om de testtemperaturen te voorspellen?",
            "`w = np.polyfit(x_train, y_train, degree)` en `mse_test = np.mean((y_test - np.polyval(w, ___)) ** 2)`.",
        ],
        why="`np.polyfit` leert alleen van de 12 trainingsmetingen. `np.polyval(w, x_test)` voorspelt de 6 testtemperaturen, en het gemiddelde van de gekwadrateerde residuen is ongeveer 0,96.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Waarom hij zo slingert
    """)
    return


@app.cell(hide_code=True)
def explain_why_cell():
    explain_why = L.explain(
        "why-swings",
        question="In het uitgewerkte voorbeeld heeft de curve van graad 9 op de training set een lagere MSE dan de curve van graad 2, maar op de test set een veel hogere. Waarom? Schrijf één of twee zinnen.",
        model_answer="Met negen weights en een bias term, tien te leren getallen, en maar twaalf trainingsmetingen kan de curve van graad 9 door bijna elk trainingspunt buigen, dus volgt hij ook hun ruis. Tussen en buiten die punten slingert hij weg van het echte stijgen en dalen van de dag, dus op de testuren zit hij ver naast.",
        points=[
            "de curve van graad 9 kan door bijna elk trainingspunt buigen",
            "dus volgt hij ook de ruis van de trainingsmetingen",
            "tussen en buiten die punten slingert hij weg van het echte patroon, dus zit hij op de test set ver naast",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl) liet de test set zien dat een opzoekmodel, dat op de huizen uit zijn training set nooit ernaast zit, het op nieuwe huizen slecht doet; dat was overfitting. Nu kun je de capaciteit van een model hoger en lager zetten, en zie je beide manieren om de mist in te gaan: underfitting als het te eenvoudig is, overfitting als het te flexibel is. In de volgende les, *Parameters en hyperparameters*, leer je de instellingen kennen die je zelf kiest, zoals hier de graad of de learning rate.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_test_mse, explain_why):
    L.wrap_up(choice_which, result_test_mse, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
