# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-regularization. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-regularization", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) en [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_choose():
    quick_choose = L.choice(
        "quick-choose",
        question="Op welke set kies je een hyperparameter?",
        options={
            "train": "De training set",
            "right": "De validation set",
            "test": "De test set",
        },
        correct="right",
        why={
            "train": "Op de training set ziet meer capaciteit er altijd beter uit. Zie *Het idee* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "test": "De test set blijft verzegeld tot het eind, voor één eerlijke score. Zie *Het idee* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "right": "Train met elke waarde op de training set, en houd de waarde met de laagste validation-loss.",
        },
    )
    quick_choose
    return (quick_choose,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="$w = 3$ en $b = 0$. De gradient is daar $(2;\\ -1)$, en de learning rate is 0,5. Waar zijn $w$ en $b$ na één update-stap?",
        options={
            "uphill": "$w = 4$, $b = -0{,}5$",
            "half": "$w = 2$, $b = -0{,}5$",
            "right": "$w = 2$, $b = 0{,}5$",
        },
        correct="right",
        why={
            "uphill": "Dat is een stap met de gradient mee, bergop. Zie het recept in [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl).",
            "half": "Kijk naar het teken bij $b$: zijn helling is negatief. Zie het recept in [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl).",
            "right": "$w = 3 - 0{,}5 \\cdot 2 = 2$ en $b = 0 - 0{,}5 \\cdot (-1) = 0{,}5$: tegen elke helling in.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model met veel features en weinig voorbeelden overfit: het geeft grote weights aan features die toevallig bij de ruis van zijn trainingsvoorbeelden passen. Je zou met de hand features kunnen weghalen, maar vaak weet je niet welke. Laat in plaats daarvan grote weights iets kosten: tel er een straf voor op bij de loss. Gradient descent laat een weight dan alleen groeien waar dat de loss duidelijk verlaagt.

    **Een voorbeeld uit het dagelijks leven.** Een luchtvaartmaatschappij laat je betalen voor elke extra kilo bagage. Wat je echt nodig hebt, neem je nog steeds mee, maar van wat je voor de zekerheid meeneemt, pak je veel minder in.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights",
        question="Twaalf voorbeelden met elk acht features; de labels hangen alleen van de eerste twee af. Gewone lineaire regressie past bijna precies op de twaalf voorbeelden. Hoe zien de weights voor de andere zes features er volgens jou uit?",
        options={
            "zero": "Dicht bij 0: die features doen er niet toe",
            "some": "Een paar ver van 0",
            "same": "Allemaal ongeveer zo groot als de eerste twee",
        },
        reveal="**Een paar ver van 0.** In het uitgewerkte voorbeeld hieronder krijgt feature 3 een weight van ongeveer −1,9, terwijl hij niets met de labels te maken heeft. Met acht features en maar twaalf voorbeelden gebruikt het model ze om de ruis te volgen. Op nieuwe voorbeelden kost dat: een MSE van ongeveer 1,28 op de validation set.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Regularisatie** is elke verandering aan het trainen die een model eenvoudiger houdt, zodat het beter generaliseert. De meest gebruikte soort, **L2-regularisatie**, telt de som van de gekwadrateerde weights op bij de loss, keer een getal $\lambda$:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = \frac{1}{n} \sum_{i=1}^{n} (y_i - [[pred:\hat{y}_i]])^2 + [[hyper:\lambda]] \sum_{j} [[param:w_j]]^2"),
        mo.md(r"""
        Grote weights kosten nu iets, dus laat gradient descent een weight alleen groeien waar dat het eerste deel duidelijk verlaagt. De bias term blijft buiten de straf: die schuift elke voorspelling alleen evenveel op. $\lambda$ is een hyperparameter: met $\lambda = 0$ heb je gewone lineaire regressie; met een heel grote $\lambda$ gaat elke weight naar bijna 0. Je kiest hem op de validation set.

        De helling van $\lambda w_j^2$ voor $w_j$ is $2 \lambda w_j$, dus de helling voor elke weight krijgt $2 \lambda w_j$ erbij. Eén update-stap is dan $w_j - \eta\,(\text{helling van de MSE} + 2 \lambda w_j) = (1 - 2 \eta \lambda)\, w_j - \eta \cdot \text{helling van de MSE}$: vóór de gewone stap krimpt elke weight met dezelfde factor. Daarom heet het ook **weight decay**. Bij een kleine $\lambda$ ligt die factor net onder 1; kies bij een grote $\lambda$ een kleinere learning rate, zodat $2 \eta \lambda$ ruim onder 1 blijft.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **regularisatie** | | elke verandering aan het trainen die een model eenvoudiger houdt, zodat het beter generaliseert | |
        | **L2-regularisatie** | L twee | $\lambda$ keer de som van de gekwadrateerde weights optellen bij de loss | `lam * np.sum(w ** 2)` |
        | $\lambda$ | lambda | hoe zwaar de straf is: een hyperparameter | `lam` |
        | **weight decay** | | elke update-stap laat elke weight een beetje krimpen richting 0; bij gradient descent hetzelfde als L2-regularisatie | `2 * lam * w` in de helling |
        """),
        L.watch_out("Kies $\\lambda$ op de validation set, nooit op de training set. Op de training set wint $\\lambda = 0$ altijd: door de straf past het model alleen maar slechter op de trainingsvoorbeelden."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek van de MSE: dat zet $\lambda$. De ene grafiek toont de acht <span class="c-param">weights</span>; de streepjes geven de weights aan waarmee de labels gemaakt zijn, 3, −2 en zes keer 0. De andere toont de <span class="c-loss">MSE</span> voor elke $\lambda$: op de training set (doorgetrokken lijn, gevulde stip) en op de validation set (gestreepte lijn, open stip). De figuur rekent de weights direct uit, zonder stappen: de weights waar gradient descent na heel veel stappen uitkomt. Zo volgt hij de slider meteen.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lambda_cell():
    def _frame(lam):
        lam = round(lam, 2)
        train, val = C.errors(lam)
        w, _ = C.fit(lam)
        return (C.picture(lam, {"weights": "De acht weights", "feature": "feature", "weight": "weight",
                                "mse": "MSE per λ", "loss": "MSE",
                                "alt": f"Acht staven, één per weight, bij λ = {L.number(lam, 2)}: de eerste {L.number(w[0], 2)}, de tweede {L.number(w[1], 2)}, de andere zes tussen {L.number(min(w[2:]), 2)} en {L.number(max(w[2:]), 2)}. De MSE op de training set stijgt met λ; op de validation set daalt hij steil van 1,28 naar ongeveer 0,15 tussen λ = 0 en 0,01, en stijgt dan weer."}),
                f"""Met <i>λ</i> = {L.number(lam, 2)}: een MSE van <span class="c-loss ea-num">{L.number(train, 3)}</span> op de training set en <span class="c-loss ea-num">{L.number(val, 3)}</span> op de validation set.""")

    explore_lambda = L.explore("lambda", "λ", _frame, start=0, low=0, high=0.2, step=0.01, role="hyper", task="Zoek de $\\lambda$ met de laagste MSE op de validation set. Wat gebeurt er met de eerste twee weights als $\\lambda$ groter wordt dan die?")
    explore_lambda
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Op de validation set is de MSE het laagst rond $\lambda = 0{,}01$. De gestreepte kromme zakt tot ongeveer 0,15 bij een $\lambda$ net onder 0,01; van de standen van de slider geeft 0,01 de laagste MSE, 0,20. Een grotere $\lambda$ krimpt ook de eerste twee weights, de weights die ertoe doen, en de MSE op de validation set stijgt weer: underfitting.

    ### Recept

    Zo train je met L2-regularisatie:

    1. Tel $\lambda$ keer de som van de gekwadrateerde weights op bij de loss: `lam * np.sum(w ** 2)`.
    2. Tel bij de hellingen voor de weights `2 * lam * w` op; laat de helling voor de bias term zoals hij is.
    3. Train met gradient descent zoals eerder.
    4. Probeer een paar waarden van $\lambda$, en kies op de validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Veertig verzonnen voorbeelden met acht features tussen 0 en 1. De labels zijn $1 + 3 \cdot \text{feature 1} - 2 \cdot \text{feature 2}$, in code `_X[:, 0]` en `_X[:, 1]`, plus ruis van hooguit 0,5 omhoog of omlaag; de andere zes features hebben er niets mee te maken. De voorbeelden zijn willekeurig getrokken, dus de eerste 12 kunnen de training set zijn en de andere 28 de validation set. Gradient descent zet 20.000 stappen met $\eta$ = 0,3, één keer met $\lambda = 0$ en één keer met $\lambda = 0{,}01$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_regularization():
    import numpy as np

    _rng = np.random.default_rng(2)
    _X = _rng.random(size=(40, 8))
    _y = 1 + 3 * _X[:, 0] - 2 * _X[:, 1] + (_rng.random(size=40) - 0.5)
    _X_train = _X[:12]
    _y_train = _y[:12]
    _X_val = _X[12:]
    _y_val = _y[12:]

    for _lam in [0, 0.01]:
        _w = np.zeros(8)
        _b = 0.0
        for _step in range(20000):
            _r = _y_train - (_X_train @ _w + _b)
            # 1. de geregulariseerde loss: de MSE plus de straf
            _loss = np.mean(_r ** 2) + _lam * np.sum(_w ** 2)
            # 2. de hellingen ervan: 2 * lam * w erbij, alleen voor de weights
            _grad_w = -2 * (_r @ _X_train) / len(_y_train) + 2 * _lam * _w
            _grad_b = -2 * _r.mean()
            # 3. één stap van gradient descent
            _w = _w - 0.3 * _grad_w
            _b = _b - 0.3 * _grad_b
        # 4. de weights, de loss, en de MSE op de validation set
        print(_lam, _w)
        print(_loss)
        print(np.mean((_y_val - (_X_val @ _w + _b)) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met $\lambda = 0$ zijn de weights ongeveer 3,64, −3,27, −1,91, 0,84 enzovoort: feature 3 krijgt −1,91, terwijl hij er niet toe doet. De loss op de training set is maar ongeveer 0,015, maar de MSE op de validation set ongeveer 1,28. Met $\lambda = 0{,}01$ krimpen de eerste twee weights tot ongeveer 2,28 en −1,60, en blijven de andere zes allemaal binnen ongeveer 0,61 van 0. De loss op de training set stijgt tot ongeveer 0,15, waarvan ongeveer de helft straf, maar de MSE op de validation set daalt tot ongeveer 0,20.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_big():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een heel grote λ
    """)
    return


@app.cell(hide_code=True)
def choice_big_cell():
    choice_big = L.choice(
        "big-lambda",
        question="Wat gebeurt er als je traint met een heel grote $\\lambda$, zoals 100?",
        options={
            "best": "Het model wordt steeds beter op nieuwe data",
            "right": "Alle weights komen dicht bij 0 uit, en het model underfit",
            "nothing": "Er verandert niets: de straf komt niet aan de voorspellingen",
        },
        correct="right",
        why={
            "best": "Schuif in *Het idee* de slider naar rechts: wat doet de validation-MSE?",
            "nothing": "De straf verandert de weights, en de weights maken de voorspellingen. Zie *Het idee*.",
            "right": "De straf weegt dan zwaarder dan al het andere: de weights krimpen tot bijna 0, en het model voorspelt voor elk voorbeeld ongeveer hetzelfde.",
        },
    )
    choice_big
    return (choice_big,)


@app.cell(hide_code=True)
def text_exercise_penalty():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De geregulariseerde loss

        Vier voorbeelden met twee features, een model met de weights `w` en de bias term `b`, en `lam`, de $\lambda$ van de straf. Zet de geregulariseerde loss in `loss`: de mean squared error, plus `lam` keer de som van de gekwadrateerde weights. Vervang de `...`. De laatste regel is `loss`, zodat de check het ziet. De check probeert ook andere weights.
        """),
        L.task("Zet de mean squared error plus de L2-straf in `loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_penalty():
    box_penalty = L.code_box("penalty-loss", lines=9)
    box_penalty
    return (box_penalty,)


@app.cell(hide_code=True)
def check_penalty(box_penalty):
    result_penalty = L.check("penalty-loss", box_penalty.value)
    result_penalty
    return (result_penalty,)


@app.cell(hide_code=True)
def hints_penalty():
    L.hints(
        "penalty-loss",
        [
            "Stap 1 van het recept: de loss die je al kent, met de straf erbij.",
            "De voorspellingen zijn `X @ w + b`. Welke weights gaan in de straf, en hoe tel je hun kwadraten op?",
            "`loss = np.mean((y - (X @ w + b)) ** 2) + ___`.",
        ],
        why="De mean squared error is ongeveer 0,22. De som van de gekwadrateerde weights is $2^2 + (-1{,}5)^2 = 6{,}25$, dus de straf is 0,1 keer 6,25 = 0,625. Samen ongeveer 0,85.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De hellingen met weight decay

        Maak `decay_gradients(X, y, w, b, lam)` af: hij geeft de hellingen van de geregulariseerde loss terug, `grad_w` voor de weights en `grad_b` voor de bias term. De residuen `r` staan er al. Vervang de `...`. De check roept hem aan met de vier voorbeelden van oefening 2: dat moet ongeveer `([-0.0225, -0.7825], -0.925)` geven. Hij probeert ook andere voorbeelden, weights en waarden van `lam`.
        """),
        L.task("Laat `decay_gradients` de hellingen van de geregulariseerde loss teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("decay-gradients", lines=10)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("decay-gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "decay-gradients",
        [
            "Stap 2 van het recept: de hellingen van de mean squared error, zoals bij lineaire regressie, plus de helling van de straf voor de weights.",
            "De hellingen van de mean squared error zijn `-2 * (r @ X) / len(y)` en `-2 * r.mean()`. Wat is de helling van `lam * w ** 2`, en welke van de twee krijgt die?",
            "`grad_w = -2 * (r @ X) / len(y) + ___` en `grad_b = -2 * r.mean()`.",
        ],
        why="De hellingen van de mean squared error zijn ongeveer $(-0{,}4225;\\ -0{,}4825)$ en $-0{,}925$. De straf telt $2 \\cdot 0{,}1 \\cdot (2;\\ -1{,}5) = (0{,}4;\\ -0{,}3)$ op bij de eerste twee, dat geeft $(-0{,}0225;\\ -0{,}7825)$; de helling voor $b$ blijft $-0{,}925$.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [w en b passen met gradient descent](/course/open/ml-gradient-descent-2d?lang=nl) ging elke stap tegen de hellingen van de loss in; met weight decay krijgen de hellingen $2 \lambda w$ erbij, en trekt elke stap de weights ook richting 0. In [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) koos je een graad op de validation set; $\lambda$ kies je op dezelfde manier. In de volgende les, *De loss waarop je traint en de metric die je meldt*, zie je waarom het getal waarop je traint niet altijd het getal is dat je anderen vertelt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_big, result_penalty, result_gradients):
    L.wrap_up(choice_big, result_penalty, result_gradients)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
