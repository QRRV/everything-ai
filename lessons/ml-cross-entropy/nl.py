# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-cross-entropy. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-cross-entropy", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl) en [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Twee klassen krijgen allebei de logit 0. Wat geeft softmax?",
        options={
            "zero": "0 en 0",
            "right": "0,5 en 0,5",
            "one": "1 en 1",
        },
        correct="right",
        why={
            "zero": "Softmax neemt eerst $e$ tot de macht van elke logit. Wat is $e^0$? Zie *Het idee* in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "one": "$e^0 = 1$ voor allebei, maar dan deelt softmax door hun som. Zie *Het idee* in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "$e^0 = 1$ voor allebei, gedeeld door hun som, 2: gelijke logits geven gelijke kansen.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def quick_check_log_loss():
    quick_log_loss = L.choice(
        "quick-log-loss",
        question="Een mail is spam, $y = 1$, en het model geeft $\\hat{y} = 0{,}8$. Hoe groot is de log loss ongeveer?",
        options={
            "other": "1,61",
            "negative": "−0,22",
            "right": "0,22",
        },
        correct="right",
        why={
            "negative": "De log van 0,8 is negatief. Wat doet log loss daarmee? Zie *Het idee* in [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "other": "Dat is $-\\log(0{,}2)$, de loss als de mail geen spam was. Zie *Het idee* in [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "right": "Bij $y = 1$ is de log loss $-\\log(\\hat{y}) = -\\log(0{,}8) \\approx 0{,}22$.",
        },
    )
    quick_log_loss
    return (quick_log_loss,)


@app.cell(hide_code=True)
def quick_check_one_hot():
    quick_one_hot = L.choice(
        "quick-one-hot",
        question="Er zijn vier klassen, 0 tot en met 3. Wat is de one-hot-vector van klasse 2?",
        options={
            "right": "(0; 0; 1; 0)",
            "from-one": "(0; 1; 0; 0)",
            "number": "(0; 0; 2; 0)",
        },
        correct="right",
        why={
            "from-one": "Die heeft zijn 1 op plek 1. Waar begint het tellen? Zie *Het idee* in [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).",
            "number": "Een one-hot-vector bevat alleen nullen en één 1. Zie *Het idee* in [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).",
            "right": "Plek 0, 1, 2 en 3: de 1 staat op plek 2.",
        },
    )
    quick_one_hot
    return (quick_one_hot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Softmax geeft een kans voor elke klasse. Om een model te trainen heb je één getal nodig dat zegt hoe goed die kansen zijn: een loss. Bij twee klassen was dat log loss. Bij veel klassen verdeelt het model zijn kans over al die klassen: naar welke van die getallen moet de loss kijken, en hoe? Cross-entropy is het antwoord, en het is de loss van bijna elk model dat één klasse kiest uit vele: een model dat cijfers leest, een model dat zegt wat er op een foto staat, en een language model, dat zijn volgende woord kiest uit tienduizenden.

    **Een voorbeeld uit het dagelijks leven.** Bij een quiz verdeel je bij elke vraag 10 punten over de antwoorden, om te laten zien hoe zeker je bent. Daarna moet de quizmaster van jouw verdeling één score per vraag maken. Welke punten moeten tellen?
    """)
    return


@app.cell(hide_code=True)
def predict_dog():
    guess_dog = L.guess(
        "dog",
        question="Een model ziet een foto van een hond. Het geeft kat 0,7, hond 0,2 en vogel 0,1. Hoe groot is volgens jou zijn loss?",
        options={
            "best": "Ongeveer 0,36: min de log van 0,7",
            "right": "Ongeveer 1,61: min de log van 0,2",
            "all": "Ongeveer 4,27: min de logs van alle drie, opgeteld",
        },
        reveal="**Ongeveer 1,61.** Alleen de kans op de goede klasse telt, en de foto is van een hond: $-\\log(0{,}2) \\approx 1{,}61$. De 0,7 voor kat kost op zich niets, maar daardoor krijgt hond maar 0,2. Had het model hond 0,7 gegeven, dan was de loss $-\\log(0{,}7) \\approx 0{,}36$ geweest. Net als bij de quiz tellen alleen de punten op het goede antwoord.",
    )
    guess_dog
    return (guess_dog,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een model geeft $K$ kansen $\hat{y}_0; \dots; \hat{y}_{K-1}$, één per klasse, met de klassen genummerd vanaf 0, zoals bij one-hot-vectoren. Het label $c$ is de goede klasse, en als one-hot-vector $y_0; \dots; y_{K-1}$ heeft het een 1 op plek $c$ en overal anders 0. **Cross-entropy** is min de som, over de klassen, van elke $y_k$ maal de log van $\hat{y}_k$:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\sum_{k=0}^{K-1} y_k \log([[pred:\hat{y}_k]]) = -\log([[pred:\hat{y}_c]])"),
        mo.md(r"""
        Elke $y_k$ is 0, behalve die bij de goede klasse, dus er blijft één term over: min de log van de kans op de goede klasse. Dat is de verrassing van wat er echt gebeurde, zoals bij log loss. Een kans van 1 op de goede klasse kost 0; hoe kleiner hij is, hoe meer het kost. Voor veel voorbeelden tegelijk is de loss de gemiddelde cross-entropy van die voorbeelden.

        Met twee klassen is het log loss zelf. Stel dat het label $y$ is, 1 of 0, en dat het model klasse 1 de kans $\hat{y}$ geeft en klasse 0 de rest, $1 - \hat{y}$. Het one-hot-label is dan $(1 - y; y)$, en de som heeft twee termen: $-\left((1 - y) \log(1 - \hat{y}) + y \log(\hat{y})\right)$. Dat is de log loss van één voorbeeld: $-\log(\hat{y})$ voor een label 1, en $-\log(1 - \hat{y})$ voor een label 0.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **cross-entropy** | | de loss voor veel klassen: min de log van de kans die het model aan de goede klasse geeft; voor veel voorbeelden het gemiddelde daarvan | `loss` |
        | $c$ | c | de goede klasse van een voorbeeld: zijn label | `label` |
        | $y_k$, $\hat{y}_k$ | y k, y-dakje k | plek $k$ van het one-hot-label, en de kans die het model aan klasse $k$ geeft | `onehot[k]`, `p[k]` |
        """),
        L.watch_out("Cross-entropy kijkt alleen naar de kans op de goede klasse, maar de andere tellen toch mee: alle kansen zijn samen 1, dus meer voor een foute klasse laat minder over voor de goede. Twee modellen met dezelfde kans op de goede klasse krijgen dezelfde loss, hoe ze de rest ook verdelen."),
        mo.md(r"""
        Schuif met de slider: hij zet de logit van hond, terwijl kat de logit 2 houdt en vogel de logit 0. De foto is van een hond, klasse 1. De ene grafiek toont de <span class="c-pred">kansen</span> uit softmax, die van hond met een dikkere rand; de andere de <span class="c-loss">cross-entropy</span>, min de log van de kans op hond.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        ps = C.probabilities(z)
        shown = [L.number(p, 2) for p in ps]
        dog = L.number(ps[C.RIGHT], 4)
        loss = L.number(C.loss(z), 2)
        return (C.picture(z, {"bars": "Kansen uit softmax", "class": "klasse: 0 kat, 1 hond, 2 vogel", "p": "kans",
                              "curve": "Cross-entropy", "p_right": "kans op hond", "loss": "loss",
                              "alt": f"Drie staven voor kat, hond en vogel: {shown[0]}, {shown[1]} en {shown[2]}. De kromme van min de log van de kans op hond, met een punt bij {dog}: een loss van {loss}."}, shown),
                f"""De foto is van een hond, en het model geeft hond <span class="c-pred ea-num">{dog}</span>. De cross-entropy is min de log daarvan: <span class="c-loss ea-num">{loss}</span>.""")

    explore_z = L.explore("logit-dog", "z₁", _frame, start=0, low=-3, high=5, step=0.5, role="", task="Vanaf welke logit van hond is de cross-entropy onder 0,1? Wat is de kans op hond dan?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Vanaf een logit van 4,5 voor hond is de cross-entropy onder 0,1: de kans op hond is dan 0,9148, en de loss 0,09. De loss wordt nooit 0, hoe hoog de logit ook wordt: daarvoor is een kans van precies 1 nodig.

    ### Recept

    Zo reken je de cross-entropy van veel voorbeelden tegelijk uit, met de kansen in de rijen van `probs` en de labels in `labels`:

    1. Maak van de labels one-hot-rijen: `onehot = np.eye(K)[labels]`.
    2. Neem de log van elke kans: `np.log(probs)`.
    3. Vermenigvuldig met de one-hot-rijen en tel elke rij op: alleen de log van de goede klasse blijft over. Zet er een min voor: `-(onehot * np.log(probs)).sum(axis=1)`.
    4. Neem het gemiddelde over de voorbeelden: `.mean()`.
    5. Controleer één voorbeeld met de hand: min de log van de kans op zijn goede klasse.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Drie foto's en drie klassen: kat (0), hond (1) en vogel (2). De foto's zijn van een hond, een kat en een vogel. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_cross_entropy():
    import numpy as np

    _probs = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    _labels = np.array([1, 0, 2])
    # 1. one-hot-rijen
    _onehot = np.eye(3)[_labels]
    print(_onehot)
    # 2. de log van elke kans
    _logs = np.log(_probs)
    print(_logs)
    # 3. in elke rij blijft alleen de goede klasse over, met een min ervoor; tel dan elke rij op
    _picked = -(_onehot * _logs)
    print(_picked)
    _losses = _picked.sum(axis=1)
    print(_losses)
    # 4. het gemiddelde over de foto's
    print(_losses.mean())
    # 5. de eerste foto met de hand: een hond, met kans 0,7
    print(-np.log(0.7))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke one-hot-rij houdt één log van zijn rij over, en de min maakt hem positief: 0,357 voor de hond, 0,511 voor de kat en 0,357 voor de vogel, met overal anders 0. Per rij opgeteld kosten de foto's 0,357, 0,511 en 0,357, en hun gemiddelde is 0,408. De kat kost het meest: het model gaf hem maar 0,6. De laatste regel geeft weer 0,357, hetzelfde als de eerste foto.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_no_idea():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Geen idee
    """)
    return


@app.cell(hide_code=True)
def choice_no_idea_cell():
    choice_no_idea = L.choice(
        "no-idea",
        question="Een model met vier klassen heeft geen idee: het geeft elke klasse 0,25. Wat is zijn cross-entropy, welke klasse ook de goede is?",
        options={
            "zero": "0: het zegt niets verkeerds",
            "right": "Ongeveer 1,39",
            "four": "Ongeveer 5,55: 1,39 voor elk van de vier klassen",
        },
        correct="right",
        why={
            "zero": "Is 0,25 voor de goede klasse goed? Alleen een kans van 1 kost 0. Zie *Het idee*.",
            "four": "Welke klassen tellen mee in cross-entropy? Zie de formule in *Het idee*.",
            "right": "Alleen de goede klasse telt, en die krijgt 0,25: $-\\log(0{,}25) = \\log(4) \\approx 1{,}39$. Een model dat alle $K$ klassen dezelfde kans geeft, heeft een cross-entropy van $\\log(K)$, welk label het ook is.",
        },
    )
    choice_no_idea
    return (choice_no_idea,)


@app.cell(hide_code=True)
def text_exercise_cross_entropy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Cross-entropy als functie

        Maak `cross_entropy(p, label)` af: hij krijgt de kansen van één voorbeeld, als lijst, en het nummer van zijn goede klasse, en geeft zijn cross-entropy terug. Vervang de `...`. De check roept `cross_entropy([0.7, 0.2, 0.1], 1)` aan, de foto van de hond uit de gok: dat moet ongeveer 1,6094 geven. Hij probeert ook andere kansen en labels.
        """),
        L.task("Laat `cross_entropy` de cross-entropy van de kansen `p` en het label `label` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cross_entropy():
    box_cross_entropy = L.code_box("cross-entropy", lines=6)
    box_cross_entropy
    return (box_cross_entropy,)


@app.cell(hide_code=True)
def check_cross_entropy(box_cross_entropy):
    result_cross_entropy = L.check("cross-entropy", box_cross_entropy.value)
    result_cross_entropy
    return (result_cross_entropy,)


@app.cell(hide_code=True)
def hints_cross_entropy():
    L.hints(
        "cross-entropy",
        [
            "Alleen de kans op de goede klasse telt. Welk element van `p` is dat?",
            "`label` is de plek van de goede klasse in `p`. Wat maakt van een kans zijn verrassing, en welk teken hoort daarbij?",
            "`return ___(p[label])`.",
        ],
        why="`p[label]` is de kans op de goede klasse, en `-np.log` maakt er zijn verrassing van. Voor `[0.7, 0.2, 0.1]` en het label 1: $-\\log(0{,}2) \\approx 1{,}61$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_batch():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Vier foto's tegelijk

        Vier foto's, drie klassen. De kansen van elke foto zijn een rij van `probs`, en hun labels staan in `labels`; `onehot` bevat ze al als one-hot-rijen. Zet de gemiddelde cross-entropy van de vier foto's in `loss`. Vervang de `...`. De laatste regel is `loss`, zodat de check het ziet. De check probeert ook andere labels.
        """),
        L.task("Zet de gemiddelde cross-entropy van `probs` en `onehot` in `loss`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batch():
    box_batch = L.code_box("batch-loss", lines=8)
    box_batch
    return (box_batch,)


@app.cell(hide_code=True)
def check_batch(box_batch):
    result_batch = L.check("batch-loss", box_batch.value)
    result_batch
    return (result_batch,)


@app.cell(hide_code=True)
def hints_batch():
    L.hints(
        "batch-loss",
        [
            "Stap 2 tot en met 4 van het recept, in één regel: `onehot` is stap 1, die is al gedaan.",
            "Vermenigvuldig `onehot` met `np.log(probs)`. Tel dan langs elke rij op, en neem het gemiddelde van wat eruit komt. Waar komt de min?",
            "`loss = -(onehot * np.log(probs)).sum(axis=___).mean()`.",
        ],
        why="`onehot * np.log(probs)` houdt in elke rij de log van de goede klasse over, `.sum(axis=1)` geeft één getal per foto, de min maakt er verrassingen van, en `.mean()` middelt ze: ongeveer 0,83.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) gaf de kansen, en [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl) de labels als vectoren; cross-entropy brengt ze samen in één getal. Het is de verrassing uit [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl), voor elk aantal klassen, en met twee klassen is het log loss. In de volgende les, *De gradient van softmax met cross-entropy*, zoek je uit hoe de loss verandert met elke logit, zodat gradient descent een model voor veel klassen kan trainen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_no_idea, result_cross_entropy, result_batch):
    L.wrap_up(choice_no_idea, result_cross_entropy, result_batch)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
