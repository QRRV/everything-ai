# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-backward-by-hand. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-backward-by-hand", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) en [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chain():
    quick_chain = L.choice(
        "quick-chain",
        question="$u = 5 - x$ en $y = u^2$. Wat is $\\frac{dy}{dx}$ bij $x = 2$?",
        options={
            "outer": "6",
            "right": "−6",
            "inner": "−1",
        },
        correct="right",
        why={
            "outer": "Dat is alleen de buitenste helling $2u$. Wat doet $u$ als $x$ stijgt? Zie [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "inner": "Dat is alleen de binnenste helling. Waarmee vermenigvuldigt de kettingregel die? Zie [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl).",
            "right": "Bij $x = 2$ is $u = 3$, dus de buitenste helling $2u$ is 6. De binnenste helling is −1: $u$ daalt als $x$ stijgt. Samen: 6 × (−1) = −6.",
        },
    )
    quick_chain
    return (quick_chain,)


@app.cell(hide_code=True)
def quick_check_passes():
    quick_passes = L.choice(
        "quick-passes",
        question="Je meet elke helling van een model met $N$ parameters door elke parameter om de beurt te verschuiven. Hoeveel forward passes kost dat?",
        options={
            "right": "N + 1",
            "two": "2",
            "square": "N²",
        },
        correct="right",
        why={
            "two": "Dat is genoeg voor één helling. Hoeveel hellingen heb je nodig? Zie [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl).",
            "square": "N² is hoe de FLOPs groeien, niet het aantal forward passes. Zie [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl).",
            "right": "Eén waar je bent, en één voor elke parameter die je verschuift: N + 1 forward passes, elk van ongeveer 2N FLOPs.",
        },
    )
    quick_passes
    return (quick_passes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl) kostte elke helling meten door te verschuiven een forward pass per parameter. Maar de hellingen staan niet los van elkaar: de loss bereikt elke parameter via dezelfde paar stappen, zoals de voorspelling, het residu en het kwadraat. [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) vertelt hoe hellingen samengaan langs zulke stappen. Deze les gebruikt dat, stap voor stap, op een berekening die je met de hand kunt volgen. Elke library die netwerken traint, zoals PyTorch, werkt zo.

    **Een voorbeeld uit het dagelijks leven.** Voor één euro krijg je 1,1 dollar, en voor één dollar 150 yen. Dus voor één euro krijg je 1,1 × 150 = 165 yen: onderweg vermenigvuldig je de koersen.
    """)
    return


@app.cell(hide_code=True)
def predict_reuse():
    guess_reuse = L.guess(
        "reuse",
        question="De loss van één student is $\\mathcal{L} = r^2$, met het residu $r = 7 - \\hat{y}$ en de voorspelling $\\hat{y} = w \\cdot 4 + b$. Zowel $\\frac{\\partial \\mathcal{L}}{\\partial w}$ als $\\frac{\\partial \\mathcal{L}}{\\partial b}$ loopt via $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}}$. Moet je $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}}$ volgens jou twee keer uitrekenen, één keer voor elk?",
        options={
            "twice": "Ja: één keer voor $w$ en één keer voor $b$",
            "right": "Nee: één keer, en allebei gebruiken ze hem",
            "equal": "Alleen als $w$ en $b$ gelijk zijn",
        },
        reveal="**Eén keer.** Werk terug vanaf de loss: eerst $\\frac{\\partial \\mathcal{L}}{\\partial r}$, dan $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}}$, en uit dat ene getal zowel $\\frac{\\partial \\mathcal{L}}{\\partial b}$ als $\\frac{\\partial \\mathcal{L}}{\\partial w}$. Elke helling erachter gebruikt hem opnieuw. Daarom geeft één keer teruglopen elke helling.",
    )
    guess_reuse
    return (guess_reuse,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Schrijf de berekening als een **computational graph**: één node per waarde, met pijlen vanaf de waarden die erin gaan. De forward pass rekent de waarden uit van de inputs naar de loss, en bewaart ze. Voor één student die $x = 4$ uur studeerde en een 7 haalde, bij $w = 1$ en $b = 1$:
        """),
        L.formula(r"m = [[param:w]] \cdot x \qquad [[pred:\hat{y}]] = m + [[param:b]] \qquad [[residual:r]] = 7 - [[pred:\hat{y}]] \qquad [[loss:\mathcal{L}]] = [[residual:r]]^2"),
        mo.md(r"""
        Elke node kent alleen zijn eigen bewerking. Zijn **local gradient** is zijn helling naar één input: voor $m = w \cdot x$ is dat $\frac{\partial m}{\partial w} = x$; voor $\hat{y} = m + b$ is het 1 voor beide inputs; voor $r = 7 - \hat{y}$ is het −1; voor $\mathcal{L} = r^2$ is het $2r$.

        De **backward pass** loopt van de loss terug naar de parameters. Hij begint met $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$, en vermenigvuldigt bij elke pijl met de local gradient: de helling van een input is de helling van de node waar hij in gaat, keer de local gradient ertussen. Dat is de kettingregel, stap voor stap:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w}]] = [[grad:\frac{\partial \mathcal{L}}{\partial r}]] \cdot \frac{\partial r}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial m} \cdot \frac{\partial m}{\partial w} = 4 \cdot (-1) \cdot 1 \cdot 4 = -16"),
        mo.md(r"""
        Een forward pass en dan een backward pass is **backpropagation**. De backward pass doet een of twee vermenigvuldigingen per pijl, dus hij geeft elke helling voor een paar keer de kosten van een forward pass: $O(N)$, niet $O(N^2)$.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **computational graph** | | een berekening getekend als nodes, één per waarde, met pijlen vanaf de waarden die erin gaan | |
        | **local gradient** | | de helling van één node naar één van zijn inputs, alleen uit zijn eigen bewerking, zoals $\frac{\partial m}{\partial w} = x$ | `dm_dw` |
        | **backward pass** | | van de loss terug naar de parameters: de helling van elke input is de helling erna keer de local gradient | |
        | **backpropagation** | | een forward pass die elke waarde bewaart, en dan een backward pass die elke helling geeft | |
        """),
        L.watch_out("Als een waarde in meer dan één node gaat, krijgt hij langs elke pijl een helling terug. Zijn helling is de som daarvan: tel ze op, houd er niet maar één."),
        mo.md(r"""
        Schuif met de slider: hij loopt de backward pass door, één stap per keer. Elke node laat zijn waarde uit de forward pass zien, en, zodra de backward pass hem bereikt, zijn <span class="c-grad">helling</span>; tot dan de stap waarbij hij die krijgt. De <span class="c-grad">getallen bij de pijlen</span> zijn de local gradients.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    _say = [
        "Stap 0: de forward pass is klaar, en elke waarde is bewaard. De backward pass is nog niet begonnen.",
        "Stap 1: de loss naar zichzelf: $\\frac{\\partial \\mathcal{L}}{\\partial \\mathcal{L}} = 1$.",
        "Stap 2: $\\mathcal{L} = r^2$ heeft de local gradient $2r = 4$, dus $\\frac{\\partial \\mathcal{L}}{\\partial r} = 1 \\cdot 4 = 4$.",
        "Stap 3: $r = 7 - \\hat{y}$ heeft de local gradient −1, dus $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}} = 4 \\cdot (-1) = -4$.",
        "Stap 4: $\\hat{y} = m + b$ heeft de local gradient 1 voor allebei, dus $\\frac{\\partial \\mathcal{L}}{\\partial m} = -4$ en $\\frac{\\partial \\mathcal{L}}{\\partial b} = -4$.",
        "Stap 5: $m = w \\cdot x$ heeft de local gradient $x = 4$ voor $w$, dus $\\frac{\\partial \\mathcal{L}}{\\partial w} = -4 \\cdot 4 = -16$. Elke helling is bekend.",
    ]

    def _frame(step):
        step = int(step)
        return (C.picture(step, {"title": "De backward pass, stap voor stap", "later": "bij stap {n}",
                                 "alt": f"Een computational graph van boven naar beneden: w = 1 en x = 4 gaan in m = w · x = 4; m en b = 1 gaan in ŷ = m + b = 5; ŷ gaat in r = 7 − ŷ = 2; r gaat in L = r² = 4. De backward pass bereikt L bij stap 1, r bij stap 2, ŷ bij stap 3, m en b bij stap 4, en w bij stap 5. Bij stap {step} van 5: {', '.join(C.known(step)) or 'nog geen helling'}."}).text,
                _say[step])

    explore_step = L.explore("step", "stap", _frame, start=0, low=0, high=5, step=1, role="", task="Waarom krijgt $b$ zijn helling een stap eerder dan $w$, terwijl het allebei parameters zijn?")
    explore_step
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    $b$ krijgt zijn helling bij stap 4, $w$ pas bij stap 5. $b$ gaat rechtstreeks in $\hat{y}$, terwijl $w$ eerst in $m$ gaat: één pijl verder van de loss. De backward pass gaat één node per keer terug, dus hoe meer pijlen er tussen een waarde en de loss liggen, hoe later zijn helling komt.

    ### Recept

    Zo krijg je elke helling van een berekening:

    1. Forward pass: reken elke waarde uit, van de inputs naar de loss, en bewaar ze.
    2. Begin bij de loss: zijn helling naar zichzelf is 1.
    3. Loop terug, node voor node: de helling van een input is de helling van de node erna, keer de local gradient ertussen.
    4. Waar een waarde in meer dan één node gaat, tel je de hellingen op die terugkomen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De student van de figuur, in code: de forward pass, dan de backward pass. Een naam zoals `dloss_dr` staat voor $\frac{\partial \mathcal{L}}{\partial r}$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_backward():
    _x = 4
    _y = 7
    _w = 1
    _b = 1
    # 1. de forward pass, elke waarde bewaard
    _m = _w * _x
    _y_pred = _m + _b
    _r = _y - _y_pred
    _loss = _r ** 2
    print(_m, _y_pred, _r, _loss)
    # 2. de loss naar zichzelf
    _dloss_dloss = 1
    # 3. terug, node voor node: de helling erna keer de local gradient
    _dloss_dr = _dloss_dloss * 2 * _r
    _dloss_dy_pred = _dloss_dr * -1
    _dloss_dm = _dloss_dy_pred * 1
    _dloss_db = _dloss_dy_pred * 1
    _dloss_dw = _dloss_dm * _x
    print(_dloss_dr, _dloss_dy_pred, _dloss_dw, _dloss_db)
    # 4. r ** 2 is ook r * r: een node waar r twee keer in gaat, dus tel beide pijlen op
    _dloss_dr_twice = _dloss_dloss * _r + _dloss_dloss * _r
    print(_dloss_dr_twice)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De helling voor $w$ is −16, hetzelfde als $2r \cdot (-x) = 2 \cdot 2 \cdot (-4)$ in [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl). De helling voor $b$, −4, kwam bijna gratis: hij gebruikte $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ opnieuw. Allebei zijn negatief, dus $w$ en $b$ groter maken verlaagt de loss: de voorspelling 5 ligt onder het cijfer 7.

    De laatste regel is stap 4 van het recept. $r^2$ is ook $r \cdot r$: een node waar $r$ twee keer in gaat. Langs elke pijl komt de andere input terug, $r = 2$, en opgeteld is dat 4: hetzelfde als $2r$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_backward():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Achteruit door een neuron met ReLU

        Een neuron met ReLU, voor één voorbeeld: $z = w \cdot x + b$, $\hat{y} = \mathrm{ReLU}(z)$, $r = y - \hat{y}$ en $\mathcal{L} = r^2$. De forward pass staat in het vak. Vervang elke `...` door één stap van de backward pass, van de loss terug naar `w` en `b`. ReLU heeft de local gradient 1 waar $z$ boven 0 ligt, en 0 op de andere plekken: daar is hij vlak. De laatste regel is `[dloss_dw, dloss_db]`, zodat de check het ziet. De check probeert ook andere getallen voor `x`, `y`, `w` en `b`.
        """),
        L.task("Maak de backward pass af, zodat `dloss_dw` en `dloss_db` de hellingen van de loss zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_backward():
    box_backward = L.code_box("backward", lines=24)
    box_backward
    return (box_backward,)


@app.cell(hide_code=True)
def check_backward(box_backward):
    result_backward = L.check("backward", box_backward.value)
    result_backward
    return (result_backward,)


@app.cell(hide_code=True)
def hints_backward():
    L.hints(
        "backward",
        [
            "Stap 3 van het recept, zoals in het uitgewerkte voorbeeld, met één node meer: ReLU, tussen $z$ en $\\hat{y}$.",
            "Elke regel is de helling van de regel ervoor keer één local gradient: $2r$, dan −1, dan 1 of 0 voor ReLU, dan $x$ voor `w` en 1 voor `b`.",
            "`dloss_dr = 2 * r`, `dloss_dy_pred = dloss_dr * ___`, `dloss_dz = dloss_dy_pred * ___` waar `z > 0`, `dloss_dw = dloss_dz * ___` en `dloss_db = dloss_dz * ___`.",
        ],
        why="Met $x = 2$, $y = 7$, $w = 1{,}5$ en $b = 1$: $z = 4$, $\\hat{y} = 4$, $r = 3$. Terug: $\\frac{\\partial \\mathcal{L}}{\\partial r} = 6$, $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}} = -6$, en $z$ ligt boven 0, dus $\\frac{\\partial \\mathcal{L}}{\\partial z} = -6$. Dan $\\frac{\\partial \\mathcal{L}}{\\partial w} = -6 \\cdot 2 = -12$ en $\\frac{\\partial \\mathcal{L}}{\\partial b} = -6$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_branch():
    mo.md(r"""
    ### Oefening 2. Eén waarde, twee pijlen
    """)
    return


@app.cell(hide_code=True)
def choice_branch_cell():
    choice_branch = L.choice(
        "branch",
        question="$s = a + b$ en $f = s \\cdot a$, bij $a = 2$ en $b = 3$: dan is $s = 5$ en $f = 10$. De waarde $a$ gaat in twee nodes. Wat is $\\frac{\\partial f}{\\partial a}$?",
        options={
            "direct": "5: alleen langs de pijl van $a$ naar $f$",
            "right": "7: langs beide pijlen, opgeteld",
            "via": "2: alleen langs de pijl van $a$ naar $s$",
        },
        correct="right",
        why={
            "direct": "Dat is wat er langs één pijl terugkomt. $a$ gaat ook in $s$: wat komt er langs die pijl terug? Zie het let-op in *Het idee*.",
            "via": "Dat is wat er via $s$ terugkomt. $a$ gaat ook rechtstreeks in $f$: wat komt er langs die pijl terug? Zie het let-op in *Het idee*.",
            "right": "Rechtstreeks in $f$: $\\frac{\\partial f}{\\partial a} = s = 5$. Via $s$: $\\frac{\\partial f}{\\partial s} \\cdot \\frac{\\partial s}{\\partial a} = a \\cdot 1 = 2$. Samen 7, zoals $f = a^2 + ab$ de helling $2a + b = 7$ heeft.",
        },
    )
    choice_branch
    return (choice_branch,)


@app.cell(hide_code=True)
def text_exercise_check():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Controleer het door te verschuiven

        In [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl) was elke parameter een heel klein stukje verschuiven te traag om mee te trainen, maar goed om te controleren. Controleer de hellingen van het uitgewerkte voorbeeld: vervang de twee `...` door de hellingen van `loss` bij $w = 1$ en $b = 1$, gemeten door $w$, en dan $b$, met `h` te verschuiven. De laatste regel is `[slope_w, slope_b]`, zodat de check het ziet.
        """),
        L.task("Meet beide hellingen door te verschuiven, en vergelijk ze met −16 en −4 uit de backward pass."),
    ])
    return


@app.cell(hide_code=True)
def exercise_check():
    box_check = L.code_box("check", lines=10)
    box_check
    return (box_check,)


@app.cell(hide_code=True)
def check_check(box_check):
    result_check = L.check("check", box_check.value)
    result_check
    return (result_check,)


@app.cell(hide_code=True)
def hints_check():
    L.hints(
        "check",
        [
            "Zoals in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl): de loss met één parameter verschoven, min de loss waar je bent, gedeeld door `h`.",
            "Verschuif alleen `w` voor `slope_w`, en alleen `b` voor `slope_b`; de andere blijft op 1.",
            "`slope_w = (loss(1 + h, 1) - loss(___, ___)) / h`, en hetzelfde voor `b`.",
        ],
        why="$w$ verschuiven met $h$ verandert de loss met ongeveer −16 · $h$, en $b$ verschuiven met ongeveer −4 · $h$; gedeeld door $h$ zijn de hellingen −15,9984 en −3,9999. Ze zijn niet precies −16 en −4, omdat $h$ geen 0 is, maar ze liggen dicht bij wat de backward pass geeft.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Snelheden vermenigvuldigen: de kettingregel](/course/open/math-chain-rule?lang=nl) vermenigvuldigde twee hellingen; de backward pass doet dat bij elke pijl van een graaf, vanaf de loss terug, en gebruikt opnieuw wat hij al heeft. Zo geeft backpropagation elke helling voor een paar keer de kosten van een forward pass, terwijl verschuiven een forward pass per parameter kostte, zoals in [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl). In de volgende les, *Bouw je eigen autograd: een Value die onthoudt*, schrijf je een kleine class die de graaf opbouwt terwijl hij rekent, en de backward pass voor je doet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_backward, choice_branch, result_check):
    L.wrap_up(result_backward, choice_branch, result_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
