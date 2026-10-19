# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-feature-scaling. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-feature-scaling", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) en [Broadcasting](/course/open/np-broadcasting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_std():
    quick_std = L.choice(
        "quick-std",
        question="Wat vertelt de standaardafwijking van een kolom getallen je?",
        options={
            "right": "Hoe ver de waarden meestal van hun gemiddelde liggen",
            "range": "De grootste waarde min de kleinste",
            "middle": "De middelste waarde, als je ze sorteert",
        },
        correct="right",
        why={
            "range": "Dat gebruikt alleen de twee buitenste waarden. Wat doet de standaardafwijking met alle waarden? Zie *Het idee* in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "middle": "Dat is de mediaan. Zie de tabel in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "right": "Zo ver liggen de waarden doorgaans van het gemiddelde af, in de eenheid van de waarden: klein als ze dicht bij elkaar liggen.",
        },
    )
    quick_std
    return (quick_std,)


@app.cell(hide_code=True)
def quick_check_lr():
    quick_lr = L.choice(
        "quick-broadcast",
        question="`X` heeft shape (8, 3) en `m` heeft shape (3,). Wat doet `X - m`?",
        options={
            "error": "Python stopt: de shapes passen niet",
            "right": "Hij trekt `m` af van elke rij van `X`: shape (8, 3)",
            "first": "Hij trekt `m` alleen af van de eerste rij van `X`",
        },
        correct="right",
        why={
            "error": "Zet de shapes onder elkaar, rechts uitgelijnd: 3 en 3 zijn gelijk, en een lengte die ontbreekt, telt als 1. Zie de regel in [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "first": "NumPy rekt `m` uit over alle 8 rijen, zonder hem te kopiëren. Zie het plaatje in [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "right": "Rechts uitgelijnd passen (8, 3) en (3,): `m` hoort bij elke rij, dus van elke kolom gaat zijn eigen getal uit `m` af.",
        },
    )
    quick_lr
    return (quick_lr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) stond de oppervlakte in honderden m² en de afstand in km, zodat alle drie de features ongeveer even grote getallen hadden. Echte data krijg je zelden zo: een oppervlakte in m², een afstand in meters, een inkomen in euro's. Eén learning rate moet bij elke weight passen, en met features van heel verschillende grootte past geen enkele learning rate. Je hebt een manier nodig om elke feature op dezelfde schaal te zetten, wat zijn eenheid ook is.

    **Een voorbeeld uit het dagelijks leven.** De ene toets heeft een cijfer van 1 tot 10, de andere een score tot 100. Tel je ze op, dan telt de tweede toets tien keer zo zwaar. Om ze eerlijk te wegen, zet je ze eerst op dezelfde schaal.
    """)
    return


@app.cell(hide_code=True)
def predict_units():
    guess_units = L.guess(
        "units",
        question="De huizen uit de vorige les, maar met de oppervlakte in m² en de afstand in meters. Gradient descent draait met dezelfde learning rate, 0,2. Wat denk je dat er gebeurt?",
        options={
            "same": "Dezelfde fit als eerst",
            "slower": "Het gaat langzamer, maar komt er wel",
            "blows-up": "De loss ontspoort",
        },
        reveal="**De training divergeert: de loss ontspoort.** De getallen zijn nu tot 1000 keer zo groot, en de hellingen dus ook: na 10 stappen is de loss ongeveer 3 × 10¹¹⁵. Een learning rate die klein genoeg is voor de afstand, 0,000001, verschuift de weight van de tuin en de bias term nauwelijks: na 1000 stappen is de loss nog ongeveer 439, terwijl 6,9 haalbaar is.",
    )
    guess_units
    return (guess_units,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De helling voor een weight groeit mee met de grootte van zijn feature: in `r @ X` wordt elk residu vermenigvuldigd met de getallen in de kolom van die feature. Afstanden in meters geven hellingen die duizend keer zo groot zijn als bij een tuin van 0 of 1. Een stap die bij de ene weight past, is veel te groot voor een andere, of veel te klein: het loss landscape is een lang, smal dal.

        Standaardisatie zet elke feature op dezelfde schaal: trek zijn gemiddelde eraf, en deel door zijn standaardafwijking.
        """),
        L.formula(r"\text{gestandaardiseerde } x = \frac{x - \text{gemiddelde van } x}{\text{standaardafwijking van } x}"),
        mo.md(r"""
        Daarna heeft elke feature gemiddelde 0 en standaardafwijking 1, wat zijn eenheid ook is: zijn getallen liggen rond 0, de meeste tussen −2 en 2. Ook het gemiddelde eraf halen telt: liggen alle oppervlaktes ver van 0, dan geven een grotere $w$ en een kleinere $b$ bijna dezelfde voorspellingen, en daarom ligt het dal schuin. Het dal wordt rond, of bijna, want features die samen bewegen kunnen het nog wat uitrekken, en één learning rate past bij elke weight. In NumPy geeft `X.mean(axis=0)` één gemiddelde per kolom, en `X - X.mean(axis=0)` trekt elk gemiddelde van zijn eigen kolom af, zoals in [Broadcasting](/course/open/np-broadcasting?lang=nl).

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **feature scaling** | | features op een vergelijkbare grootte brengen, zodat geen feature zwaarder telt alleen door zijn eenheid | |
        | **standaardisatie** | | feature scaling door het gemiddelde eraf te trekken en door de standaardafwijking te delen: daarna gemiddelde 0 en standaardafwijking 1; het werkwoord is standaardiseren | `(X - X.mean(axis=0)) / X.std(axis=0)` |
        """),
        L.watch_out("Standaardiseer een nieuw voorbeeld met het gemiddelde en de standaardafwijking van de training data, niet met die van het voorbeeld zelf. Het model leerde zijn weights op features die met die getallen gestandaardiseerd zijn."),
        mo.Html(C.picture({"raw": "Oppervlakte in m²", "scaled": "Oppervlakte gestandaardiseerd", "step": "stap {k}",
                           "alt": "Twee contourplots van de loss over w en b, elk met 50 stappen gradient descent vanaf (0, 0). Links, met de oppervlakte in m²: een lang, dun dal dat schuin ligt; de stappen springen heen en weer over het dal rond w = 3,6, b = 0, en blijven ver van het laagste punt bij w = 3,1, b = 46. Rechts, met de oppervlakte gestandaardiseerd: ronde ringen om het laagste punt bij w = 82, b = 306; de stappen lopen er recht naartoe."})),
        mo.md(r"""
        Eén feature, de oppervlakte van de huizen uit de lessen over gradient descent, en het model $w \cdot x + b$. De <span class="c-loss">lijnen</span> verbinden de plekken $(w, b)$ met dezelfde loss; de <span class="c-param">punten</span> zijn 50 stappen gradient descent, en het kruisje is het laagste punt. Links, in m², met $\eta$ = 0,00012, ongeveer de grootste learning rate die daar nog werkt, springen de stappen heen en weer over het dal terwijl $b$ nauwelijks verschuift, en blijven ze ver van het kruisje. Rechts, gestandaardiseerd, met $\eta$ = 0,3 zoals in de lessen over gradient descent, zijn ze binnen ongeveer 10 stappen bij het kruisje.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo train je op features met heel verschillende eenheden:

    1. Reken per feature het gemiddelde en de standaardafwijking over de voorbeelden uit: `mean = X.mean(axis=0)` en `std = X.std(axis=0)`.
    2. Standaardiseer: `Z = (X - mean) / std`.
    3. Train op `Z`, met één learning rate voor elke weight.
    4. Standaardiseer een nieuw voorbeeld met dezelfde `mean` en `std`, en voorspel dan: `(new - mean) / std @ w + b`. `@` tussen twee vectoren geeft hun inproduct, één getal, net als `np.dot`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De acht huizen in hun eigen eenheden: de oppervlakte in m², een tuin (1 = ja, 0 = nee), en de afstand tot het station in meters. `_fit` is gradient descent zoals in de vorige les. Eerst op de ruwe features, dan gestandaardiseerd. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_scaling():
    import numpy as np

    _X = np.array(C.HOUSES_RAW)
    _y = np.array(C.PRICES)

    def _fit(X, lr, steps):
        w = np.zeros(X.shape[1])
        b = 0.0
        for _step in range(steps):
            r = _y - (X @ w + b)
            w = w - lr * -2 * (r @ X) / len(_y)
            b = b - lr * -2 * r.mean()
        return w, b

    def _loss(X, w, b):
        return ((_y - (X @ w + b)) ** 2).mean()

    # zonder schalen: lr 0.2 ontspoort, en een piepkleine lr verschuift nauwelijks iets
    _w, _b = _fit(_X, 0.2, 10)
    print(_loss(_X, _w, _b))
    _w, _b = _fit(_X, 0.000001, 1000)
    print(_loss(_X, _w, _b))
    # 1. per feature: het gemiddelde en de standaardafwijking
    _mean = _X.mean(axis=0)
    _std = _X.std(axis=0)
    print(_mean, _std)
    # 2. en 3. standaardiseer, en train met lr 0.2
    _Z = (_X - _mean) / _std
    _w, _b = _fit(_Z, 0.2, 100)
    print(_loss(_Z, _w, _b))
    # 4. een nieuw huis: standaardiseer het met dezelfde _mean en _std, en voorspel dan
    _new = np.array([120, 0, 700])
    print((_new - _mean) / _std @ _w + _b)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op de ruwe features geeft $\eta$ = 0,2 al na 10 stappen een loss van ongeveer 3 × 10¹¹⁵, en $\eta$ = 0,000001 na 1000 stappen nog een loss van ongeveer 439. De gemiddelden zijn 82,6 m², 0,5 en 850 m, de standaardafwijkingen ongeveer 26,2 m², 0,5 en 384 m. Gestandaardiseerd haalt dezelfde 0,2 in 100 stappen een loss van 6,95: het beste wat dit model kan. Een nieuw huis van 120 m², zonder tuin, op 700 m van het station, gaat door dezelfde `_mean` en `_std`: ongeveer 295 duizend euro.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_z():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén gestandaardiseerd getal
    """)
    return


@app.cell(hide_code=True)
def choice_z_cell():
    choice_z = L.choice(
        "z-value",
        question="Een feature heeft gemiddelde 80 en standaardafwijking 20. Wat is de gestandaardiseerde waarde van 120?",
        options={
            "no-divide": "40",
            "ratio": "1,5",
            "right": "2",
        },
        correct="right",
        why={
            "no-divide": "Dat is 120 min het gemiddelde. Wat is de tweede stap van standaardisatie? Zie de formule in *Het idee*.",
            "ratio": "Dat is 120 gedeeld door het gemiddelde. Wat trek je eerst af, en waardoor deel je? Zie de formule in *Het idee*.",
            "right": "(120 − 80) / 20 = 40 / 20 = 2: twee standaardafwijkingen boven het gemiddelde.",
        },
    )
    choice_z
    return (choice_z,)


@app.cell(hide_code=True)
def text_exercise_standardize():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Uren en minuten

        `X` bevat vier leerlingen, één per rij: uren leren, en minuten slaap. Zet de gestandaardiseerde `X` in `Z`: elke kolom min zijn gemiddelde, gedeeld door zijn standaardafwijking. Vervang de `...`. De laatste regel is `Z`, zodat de check hem ziet. De check probeert ook andere leerlingen.
        """),
        L.task("Zet de gestandaardiseerde `X` in `Z`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_standardize():
    box_standardize = L.code_box("standardize", lines=6)
    box_standardize
    return (box_standardize,)


@app.cell(hide_code=True)
def check_standardize(box_standardize):
    result_standardize = L.check("standardize", box_standardize.value)
    result_standardize
    return (result_standardize,)


@app.cell(hide_code=True)
def hints_standardize():
    L.hints(
        "standardize",
        [
            "Stap 1 en 2 van het recept: het gemiddelde en de standaardafwijking van elke kolom, en dan aftrekken en delen.",
            "Elke feature is een kolom. Welke `axis` geeft één getal per kolom? En let op welk deel Python eerst uitrekent.",
            "`Z = (X - X.mean(axis=___)) / X.std(axis=___)`.",
        ],
        why="`X.mean(axis=0)` en `X.std(axis=0)` geven één getal per kolom. `(X - X.mean(axis=0)) / X.std(axis=0)` trekt van elke kolom zijn eigen gemiddelde af en deelt door zijn standaardafwijking; door de haakjes trekt Python eerst af.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een nieuw huis

        Een model met weights `w` en bias term `b` is getraind op de gestandaardiseerde huizen. Het vak heeft de huizen `X`, hun `mean` en `std`, en een nieuw huis `new`: 100 m², een tuin, 500 m van het station. Zet de voorspelde prijs van `new` in `pred`, in duizenden euro's. Vervang de `...`. De laatste regel is `pred`, zodat de check hem ziet. De check probeert ook een ander nieuw huis.
        """),
        L.task("Zet de voorspelde prijs van het nieuwe huis in `pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict-new", lines=10)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict-new", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict-new",
        [
            "Stap 4 van het recept: het nieuwe huis krijgt dezelfde standaardisatie als de trainingshuizen.",
            "Welk gemiddelde en welke standaardafwijking horen bij de getallen waarop het model getraind is: die van `new`, of die van `X`?",
            "`pred = (new - ___) / ___ @ w + b`.",
        ],
        why="`(new - mean) / std` standaardiseert het nieuwe huis met de getallen van de trainingshuizen; `@ w + b` voorspelt daarna zoals altijd. Dat geeft ongeveer 278,6 duizend euro. Zonder standaardiseren geven dezelfde weights ongeveer −1876.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) waren de eenheden van de huizen met de hand gekozen; standaardisatie, met het gemiddelde en de standaardafwijking uit [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl), doet dat voor elke dataset. Het maakt het loss landscape rond, of bijna, zodat één learning rate bij elke weight past. Het komt terug in *Project: iets echts voorspellen, van begin tot eind*, en in neural networks als batch normalization. In de volgende les, *Hetzelfde model met scikit-learn*, fit een library hetzelfde model voor je, in een paar regels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_z, result_predict, result_standardize):
    L.wrap_up(choice_z, result_standardize, result_predict)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
