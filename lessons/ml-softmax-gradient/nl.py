# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-softmax-gradient. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-softmax-gradient", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) en [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cross_entropy():
    quick_cross_entropy = L.choice(
        "quick-cross-entropy",
        question="Een model geeft een voorbeeld de kansen 0,5, 0,25 en 0,25, en zijn goede klasse is 0. Hoe groot is zijn cross-entropy ongeveer?",
        options={
            "other": "1,39",
            "right": "0,69",
            "all": "3,47",
        },
        correct="right",
        why={
            "other": "Dat is $-\\log(0{,}25)$, de loss als klasse 1 of 2 de goede was. Zie *Het idee* in [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "all": "Dat is de verrassing van alle drie de klassen opgeteld. Welke klasse telt? Zie *Het idee* in [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "right": "Alleen de goede klasse telt: $-\\log(0{,}5) \\approx 0{,}69$.",
        },
    )
    quick_cross_entropy
    return (quick_cross_entropy,)


@app.cell(hide_code=True)
def quick_check_log_loss_gradient():
    quick_log_loss_gradient = L.choice(
        "quick-log-loss-gradient",
        question="Een mail is spam, $y = 1$, en heeft één feature, $x = 2$. Het model geeft hem $\\hat{y} = 0{,}3$. Wat is $\\frac{\\partial \\mathcal{L}}{\\partial w}$ voor deze mail?",
        options={
            "right": "−1,4",
            "sign": "1,4",
            "no-x": "−0,7",
        },
        correct="right",
        why={
            "sign": "Wat komt eerst, $\\hat{y}$ of $y$? Zie de formule in [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl).",
            "no-x": "Dat is de helling voor de bias term. Wat komt erbij voor een weight? Zie de formule in [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl).",
            "right": "$(\\hat{y} - y) \\cdot x = (0{,}3 - 1) \\cdot 2 = -1{,}4$.",
        },
    )
    quick_log_loss_gradient
    return (quick_log_loss_gradient,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om een model voor veel klassen met gradient descent te trainen, heb je de helling van de cross-entropy voor elke logit nodig. Met tien klassen zijn er tien logits, en in een model heeft elke logit zijn eigen weights. Elke logit een heel klein stapje veranderen en de loss opnieuw uitrekenen werkt, maar dat is per logit een extra berekening, en elke helling is maar een schatting. De kettingregel geeft alle hellingen tegelijk, precies, en bij softmax met cross-entropy blijken ze net zo eenvoudig als bij log loss.

    **Een voorbeeld uit het dagelijks leven.** Vrienden delen een rekening. Per persoon vergelijk je wat er betaald is met het deel van die persoon: wie te veel betaalde, krijgt het verschil terug, wie te weinig betaalde, legt het bij. Eén getal per persoon, en samen zijn de getallen 0.
    """)
    return


@app.cell(hide_code=True)
def predict_cat():
    guess_cat = L.guess(
        "cat",
        question="Een model ziet een foto van een hond. Het geeft kat 0,7, hond 0,2 en vogel 0,1. De logit van kat gaat een beetje omhoog. Wat gebeurt er volgens jou met de cross-entropy?",
        options={
            "up": "Die stijgt",
            "same": "Niets: kat is niet de goede klasse",
            "down": "Die daalt",
        },
        reveal="**Die stijgt.** Cross-entropy kijkt alleen naar de kans op hond, maar softmax verdeelt 1: meer voor kat laat minder over voor hond. De helling voor de logit van kat blijkt zijn kans te zijn, 0,7. De helling van hond is zijn kans min 1, $0{,}2 - 1 = -0{,}8$: de loss daalt juist als de logit van hond stijgt.",
    )
    guess_cat
    return (guess_cat,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De cross-entropy van één voorbeeld is min de log van de kans op zijn goede klasse $c$, en softmax maakt die kans uit de logits $z_0; \dots; z_{K-1}$: $\hat{y}_c = e^{z_c} / S$, met $S = e^{z_0} + \dots + e^{z_{K-1}}$, de som waardoor softmax deelt. De log van een quotiënt is de log van de teller min de log van de noemer. Dat volgt uit $\log(a \cdot b) = \log(a) + \log(b)$ in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl): $S$ keer $e^{z_c}/S$ is $e^{z_c}$, dus $\log(S) + \log(e^{z_c}/S) = \log(e^{z_c})$. En de log maakt $e$ ongedaan, dus:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\log \frac{e^{z_c}}{S} = -z_c + \log S"),
        mo.md(r"""
        Nu de helling van elk deel voor een logit $z_k$:

        - **De helling van $-z_c$.** Die is $-1$ voor de logit van de goede klasse en 0 voor de andere. Dat is min het one-hot-label: $-y_k$.
        - **De helling van $\log S$.** Volgens de kettingregel: de helling van $\log$ bij $S$, en dat is $1/S$, keer de helling van $S$ voor $z_k$. Alleen de term $e^{z_k}$ van $S$ verandert met $z_k$, en de helling van $e^z$ is $e^z$ zelf. Dus is het $e^{z_k}/S$: dat is $\hat{y}_k$.

        De hellingen van $\log$ en van $e^z$ neem je hier aan, zoals [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl) dat deed met $\log$ en de sigmoid; een latere les leidt ze af. Samen:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial z_k}]] = [[pred:\hat{y}_k]] - y_k"),
        mo.md(r"""
        De gradient voor de logits is de kansen min het one-hot-label: hetzelfde patroon als $\hat{y} - y$ bij log loss, nu met één helling per klasse. De goede klasse krijgt $\hat{y}_c - 1$, en dat is negatief: gradient descent, dat tegen de helling in stapt, verhoogt zijn logit. Elke andere klasse krijgt zijn eigen kans, en die is positief: zijn logit gaat omlaag, het meest bij de klasse die het meest kreeg. Samen zijn de hellingen 0, omdat zowel de kansen als het one-hot-label optellen tot 1.

        In een model is elke logit een inproduct van zijn eigen weights met de features, plus zijn eigen bias term: $z_k = \mathbf{w}_k \cdot \mathbf{x} + b_k$. De kettingregel vermenigvuldigt dan nog één keer met de feature, net als bij log loss: de helling voor weight $j$ van klasse $k$ is $(\hat{y}_k - y_k)\,x_j$, en voor zijn bias term $\hat{y}_k - y_k$.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **gradient van softmax met cross-entropy**, $\hat{y}_k - y_k$ | y-dakje k min y k | de hellingen van de cross-entropy voor de logits, als softmax de kansen maakt: de kansen min het one-hot-label | `softmax(z) - np.eye(K)[label]` |
        | $S$ | S | de som $e^{z_0} + \dots + e^{z_{K-1}}$ waardoor softmax deelt | `np.exp(z).sum()` |
        """),
        L.watch_out("Let op de volgorde: de kansen min het one-hot-label, zoals $\\hat{y} - y$ bij log loss. Draai je ze om, dan krijgt elke helling het verkeerde teken: gradient descent verlaagt dan de logit van de goede klasse."),
        mo.md(r"""
        Schuif met de slider: hij zet $z_1$, de logit van hond, de goede klasse, terwijl kat de logit 2 houdt en vogel de logit 0. De ene grafiek toont de <span class="c-pred">kansen</span> als staven, die van hond met een dikkere rand. De andere toont de <span class="c-grad">helling</span> voor elke logit bij elke $z_1$, de lijn van hond het dikst, met een stip bij elke helling voor de $z_1$ van de slider en stippellijnen bij −0,1 en 0,1; sleep eroverheen om $z_1$ te zetten. De getallen zijn afgerond.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_z_cell():
    def _frame(z):
        shown_p = [L.number(p, 2) for p in C.probabilities(z)]
        shown_g = [L.number(g, 2) for g in C.gradient(z)]
        return (C.picture(z, {"bars": "Kansen uit softmax", "slopes": "Helling bij elke z₁", "class": "klasse: 0 kat, 1 hond, 2 vogel",
                              "p": "kans", "slope": "helling", "names": ["kat", "hond", "vogel"],
                              "alt": f"De kansen op kat, hond en vogel als drie staven: {shown_p[0]}, {shown_p[1]} en {shown_p[2]}. Een tweede grafiek toont de helling voor elke logit bij elke z₁ van −3 tot 5: die van hond stijgt van −0,99 naar 0, die van kat en vogel dalen naar 0. Bij z₁ = {L.number(z, 1)} zijn de hellingen {shown_g[0]}, {shown_g[1]} en {shown_g[2]}."}, shown_p),
                f"""Het model geeft hond <span class="c-pred ea-num">{shown_p[1]}</span>, dus de helling voor de logit van hond is dat min 1: <span class="c-grad ea-num">{shown_g[1]}</span>. Kat en vogel krijgen hun kansen als helling, <span class="c-grad ea-num">{shown_g[0]}</span> en <span class="c-grad ea-num">{shown_g[2]}</span>.""")

    explore_z = L.explore("logit-dog", "z₁", _frame, start=0, low=-3, high=5, step=0.5, role="", task="Vanaf welke $z_1$ liggen alle drie de hellingen binnen 0,1 van 0?")
    explore_z
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Vanaf $z_1 = 4{,}5$ liggen alle drie de hellingen binnen 0,1 van 0: zodra het model de goede klasse bijna alle kans geeft, daar 0,91, krijgen de logits van dit voorbeeld nauwelijks nog een duw. Bij $z_1 = -3$ is de helling van hond −0,99 en die van kat 0,88: de grootste duw komt waar het model het meest fout zit.

    ### Recept

    Zo reken je de hellingen van de cross-entropy uit voor de logits `z` van één voorbeeld met het label `label`:

    1. Maak de kansen met softmax: `p = softmax(z)`.
    2. Haal het one-hot-label eraf: `p - np.eye(K)[label]`.
    3. Controleer: de goede klasse heeft een negatieve helling, de andere een positieve, en samen zijn ze 0.
    4. Controleer één helling door alleen zijn logit een heel klein stapje `h` te veranderen: de verandering van de loss, gedeeld door `h`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De logits 2, 1 en 0 voor kat, hond en vogel, en de foto is van een hond: het label is 1. Eerst de hellingen uit de formule, dan elke helling gemeten met een heel klein stapje. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_softmax_gradient():
    import numpy as np

    _z = np.array([2.0, 1.0, 0.0])
    _label = 1

    def _loss(z):
        _e = np.exp(z - z.max())
        return -np.log(_e[_label] / _e.sum())

    # 1. de kansen uit softmax
    _e = np.exp(_z - _z.max())
    _p = _e / _e.sum()
    print(_p)
    # 2. min het one-hot-label
    _grad = _p - np.eye(3)[_label]
    print(_grad)
    # 3. de controle: alleen negatief voor de goede klasse
    print(_grad < 0)
    # 4. elke logit een heel klein stapje h veranderd
    _h = 0.0001
    for _k in range(3):
        _step = np.eye(3)[_k] * _h
        print((_loss(_z + _step) - _loss(_z)) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De kansen zijn 0,665, 0,245 en 0,090. Min het one-hot-label $(0; 1; 0)$ zijn de hellingen 0,665, −0,755 en 0,090: alleen die van hond is negatief, zoals `[False  True False]` laat zien, en samen zijn ze 0. De kleine stapjes meten 0,66525, −0,75526 en 0,09003: ze verschillen pas in de vijfde decimaal van de formule. `np.eye(3)[_k] * _h` is een vector met `h` op plek `_k` en verder 0, dus elk stapje verandert maar één logit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one_step():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén stap
    """)
    return


@app.cell(hide_code=True)
def choice_one_step_cell():
    choice_one_step = L.choice(
        "one-step",
        question="Een foto van een vogel. Het model geeft kat 0,5, hond 0,3 en vogel 0,2. Gradient descent zet één stap op de logits. Wat gebeurt er?",
        options={
            "only": "Alleen de logit van vogel verandert: die gaat omhoog",
            "equal": "De logit van vogel gaat omhoog; die van kat en hond gaan evenveel omlaag",
            "right": "De logit van vogel gaat omhoog; die van kat en hond gaan omlaag, die van kat het meest",
        },
        correct="right",
        why={
            "only": "Krijgen de andere klassen een helling van 0? Zie de formule in *Het idee*.",
            "equal": "Welke helling krijgt elke foute klasse? Zie de formule in *Het idee*.",
            "right": "De hellingen zijn 0,5 voor kat, 0,3 voor hond en $0{,}2 - 1 = -0{,}8$ voor vogel. Een stap ertegenin verhoogt vogel en verlaagt kat het meest, omdat kat het meest kreeg.",
        },
    )
    choice_one_step
    return (choice_one_step,)


@app.cell(hide_code=True)
def text_exercise_logit_gradient():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De gradient als functie

        Maak `logit_gradient(z, label)` af: hij krijgt de logits van één voorbeeld, als lijst, en het nummer van zijn goede klasse, en geeft de helling van de cross-entropy voor elke logit terug. `softmax` staat er al. Vervang de `...`. De check roept `logit_gradient([2, 1, 0], 1)` aan, het uitgewerkte voorbeeld: dat moet ongeveer `[0.665, -0.755, 0.090]` geven. Hij probeert ook andere logits en labels, ook met een ander aantal klassen.
        """),
        L.task("Laat `logit_gradient` de hellingen van de cross-entropy voor de logits `z` teruggeven, met het label `label`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logit_gradient():
    box_logit_gradient = L.code_box("logit-gradient", lines=10)
    box_logit_gradient
    return (box_logit_gradient,)


@app.cell(hide_code=True)
def check_logit_gradient(box_logit_gradient):
    result_logit_gradient = L.check("logit-gradient", box_logit_gradient.value)
    result_logit_gradient
    return (result_logit_gradient,)


@app.cell(hide_code=True)
def hints_logit_gradient():
    L.hints(
        "logit-gradient",
        [
            "Stap 1 en 2 van het recept: de kansen, min het one-hot-label.",
            "`softmax(z)` geeft de kansen, en `np.eye(K)[label]` het one-hot-label bij `K` klassen. Hoeveel klassen zijn er?",
            "`return softmax(z) - np.eye(___)[label]`.",
        ],
        why="`softmax(z)` geeft 0,665, 0,245 en 0,090 voor `[2, 1, 0]`; het one-hot-label van klasse 1 is $(0; 1; 0)$; het verschil is 0,665, −0,755 en 0,090. Met `len(z)` werkt het voor elk aantal klassen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_nudge():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Eén helling meten

        Drie logits, en de goede klasse is 0. Volgens de formule is de helling voor de logit van klasse 0 gelijk aan $\hat{y}_0 - 1$. Controleer dat zoals in [De afgeleide als helling](/course/open/math-slope?lang=nl): verander alleen de logit van klasse 0 met het heel kleine stapje `h`, en zet de helling van de loss in `slope`. `loss(z, label)` staat er al. Vervang de `...`. De laatste regel is `slope`, zodat de check het ziet. De check probeert ook andere logits.
        """),
        L.task("Zet de helling van `loss` voor de logit van klasse 0 in `slope`, gemeten met het heel kleine stapje `h`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nudge():
    box_nudge = L.code_box("nudge", lines=12)
    box_nudge
    return (box_nudge,)


@app.cell(hide_code=True)
def check_nudge(box_nudge):
    result_nudge = L.check("nudge", box_nudge.value)
    result_nudge
    return (result_nudge,)


@app.cell(hide_code=True)
def hints_nudge():
    L.hints(
        "nudge",
        [
            "Stap 4 van het recept: de loss met het kleine stapje, min de loss zonder, gedeeld door de grootte van het stapje.",
            "`z + np.array([h, 0, 0])` verandert alleen de logit van klasse 0. Welke twee losses trek je van elkaar af, en waardoor deel je?",
            "`slope = (loss(z + np.array([h, 0, 0]), label) - ___) / ___`.",
        ],
        why="Het kleine stapje verlaagt de loss met ongeveer 0,00008; gedeeld door `h`, 0,0001, is dat ongeveer −0,82. De formule geeft hetzelfde: de kans op klasse 0, 0,1753, min 1, is −0,8247.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De gradient van log loss](/course/open/ml-log-loss-gradient?lang=nl) was de helling voor de logit $\hat{y} - y$; met softmax en cross-entropy is het hetzelfde patroon, met één helling per klasse. In [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) mat je hoe goed de kansen zijn; nu weet je welke kant elke logit op moet om ze beter te maken. In *Project: een lineaire classifier voor handgeschreven cijfers* gebruikt gradient descent deze gradient om een model voor de tien cijfers te trainen. In de volgende les, *Werkt het op nieuwe data? Training set en test set*, zoek je uit of een getraind model ook werkt op voorbeelden die het nooit heeft gezien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one_step, result_logit_gradient, result_nudge):
    L.wrap_up(choice_one_step, result_logit_gradient, result_nudge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
