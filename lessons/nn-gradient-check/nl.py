# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-gradient-check. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-gradient-check", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl) en [De afgeleide als helling](/course/open/math-slope?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grad():
    quick_grad = L.choice(
        "quick-grad",
        question="Na `backward(loss)` is `w.grad` van een weight `w` gelijk aan −3. Wat zegt dat?",
        options={
            "value": "`w` moet −3 worden",
            "far": "`w` zit 3 van zijn beste waarde af",
            "right": "Als `w` een heel klein beetje stijgt, daalt de loss ongeveer 3 keer zoveel",
        },
        correct="right",
        why={
            "value": "`grad` is geen nieuwe waarde voor `w`, maar de helling van de loss: hoe snel de loss verandert als `w` verandert. Zie [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl).",
            "far": "De helling zegt welke kant het omlaag gaat, en hoe steil het daar is, niet waar de bodem ligt. Zie [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl).",
            "right": "`grad` is de helling van de loss naar `w`: −3 betekent dat de loss ongeveer 3 daalt voor elke eenheid dat `w` stijgt, zolang de stap klein is.",
        },
    )
    quick_grad
    return (quick_grad,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="De loss is 5,00 bij $w = 2{,}00$ en 5,03 bij $w = 2{,}01$. Wat is ongeveer de helling bij $w = 2$?",
        options={
            "right": "3: de loss stijgt 0,03 over een stap van 0,01",
            "small": "0,03: zoveel stijgt de loss",
            "loss": "5,03: de loss na de stap",
        },
        correct="right",
        why={
            "small": "0,03 is hoeveel de loss stijgt, maar de stap was maar 0,01: de helling is de stijging gedeeld door de stap. Zie [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "loss": "De helling is niet de loss zelf, maar hoe snel die verandert. Zie [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "right": "Stijging gedeeld door stap: 0,03 / 0,01 = 3. Met hellingen die je zo meet, controleert deze les gradients.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl) gaf je backward pass elke parameter een `grad`, en de training volgde die. Een fout in een backward pass laat zelden iets crashen: een factor te veel of een verkeerd teken geeft nog steeds getallen, en het netwerk traint nog steeds, alleen slechter, of helemaal niet, en niets vertelt je waarom. Straks schrijf je backward passes voor hele layers. Voordat je er een vertrouwt, controleer je hem met een helling die je zelf kunt meten, zoals je in [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl) één gradient controleerde door een getal te verschuiven.

    **Een voorbeeld uit het dagelijks leven.** Voordat je een keukenweegschaal vertrouwt die je zelf gebouwd hebt, weeg je iets waarvan je het gewicht kent, zoals een liter water.
    """)
    return


@app.cell(hide_code=True)
def predict_nudge():
    guess_nudge = L.guess(
        "up-and-down",
        question="Je meet de helling van $f(x) = x^2$ bij $x = 3$, die echt 6 is, op twee manieren, met $h = 0{,}1$. Welke komt dichterbij?",
        options={
            "up": "Alleen omhoog: $(f(3{,}1) - f(3)) / 0{,}1$, omdat die precies bij 3 begint",
            "same": "Ze zitten er even dicht bij, omdat ze allebei dezelfde $h$ gebruiken",
            "right": "Omhoog en omlaag: $(f(3{,}1) - f(2{,}9)) / 0{,}2$, omdat die naar beide kanten kijkt",
        },
        reveal="**Omhoog en omlaag.** Alleen omhoog geeft $(9{,}61 - 9) / 0{,}1 = 6{,}1$; omhoog en omlaag geeft $(9{,}61 - 8{,}41) / 0{,}2 = 6{,}0$, de helling zelf. De kromme buigt: net rechts van 3 is hij iets steiler, net links iets vlakker. Omhoog en omlaag neemt beide kanten, en hun fouten heffen elkaar op; bij een parabool zelfs precies.",
    )
    guess_nudge
    return (guess_nudge,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Gradient checking** vergelijkt de gradient uit je backward pass met een schatting door te verschuiven, één parameter tegelijk. In [De afgeleide als helling](/course/open/math-slope?lang=nl) zette je alleen een stapje omhoog: een **eindige differentie**. Zet ook een stapje omlaag en neem de helling tussen die twee, de **centrale differentie**:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_j}]] \approx \frac{[[loss:\mathcal{L}(w_j + h)]] - [[loss:\mathcal{L}(w_j - h)]]}{2h}"),
        mo.md(r"""
        De fout ervan krimpt met $h^2$, niet met $h$: tien keer kleinere $h$, honderd keer kleinere fout. Maar $h$ mag niet te klein zijn: $\mathcal{L}(w_j + h)$ en $\mathcal{L}(w_j - h)$ komen dan in bijna al hun cijfers overeen, en hun verschil houdt weinig meer over dan afronding. Met de gewone floats van NumPy is $h = 10^{-5}$ een goede keuze.

        Vergelijk de twee dan met de **relatieve fout**: het verschil, gedeeld door de grootste van de twee groottes, met $a$ de gradient uit backprop en $n$ de schatting door te verschuiven:
        """),
        L.formula(r"\text{relatieve fout} = \frac{|a - n|}{\max(|a|, |n|)}"),
        mo.md(r"""
        Relatief, omdat een gradient 1000 kan zijn of 0,001: een verschil van 0,001 zegt op zichzelf niets. Zijn ze allebei 0, zoals bij een ReLU die voor elk voorbeeld uit staat, dan zijn ze het eens: tel dat als 0. Met $h = 10^{-5}$: onder ongeveer $10^{-7}$ zijn de twee het eens; boven ongeveer $10^{-3}$ klopt de backward pass niet voor die parameter; daartussen kijk je nog eens goed. De check kost twee keer de loss uitrekenen per parameter, dus je controleert een klein netwerk, of een paar willekeurig gekozen parameters, één keer, nadat je een backward pass geschreven hebt; niet tijdens de training.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **gradient checking** | | de gradient uit een backward pass vergelijken met een schatting door te verschuiven, voor elke parameter | |
        | **centrale differentie** | | de helling tussen een stapje omhoog en een stapje omlaag: $(\mathcal{L}(w_j + h) - \mathcal{L}(w_j - h)) / 2h$ | `(f(w + step) - f(w - step)) / (2 * h)` |
        | $h$ | h | de grootte van het stapje; $10^{-5}$ is een goede keuze | `h = 1e-5` |
        | `np.abs` | | de grootte van elk getal, getal voor getal: `np.abs(a - n)` haalt het teken van elk verschil tegelijk weg. `abs(x)` van Python werkt maar op één getal | `np.abs(a - n)` |
        | **relatieve fout** | | het verschil van twee getallen gedeeld door de grootste van hun groottes | `np.abs(a - n) / np.maximum(np.abs(a), np.abs(n))` |
        | `1e-5` | één e min vijf | hoe Python $10^{-5}$ schrijft: 0,00001. Een getal ervoor vermenigvuldigt: `5.6e-12` is $5{,}6 \cdot 10^{-12}$ | `1e-5` |
        | `lambda w: ...` | lambda | een functie op één regel, zonder naam: `lambda w: np.sum(w ** 2)` krijgt `w` en geeft `np.sum(w ** 2)` terug | `lambda w: np.sum(w ** 2)` |
        """),
        L.watch_out("ReLU heeft een knik bij 0. Brengt een stapje een input van een ReLU over 0 heen, dan mengt de schatting twee hellingen, en faalt de check terwijl je backward pass klopt. Faalt één parameter, kijk dan of er een input dicht bij 0 zit voordat je een bug gaat zoeken."),
        mo.md(r"""
        De figuur: hoe ver de twee schattingen van de helling van $e^x$ bij $x = 1$ ernaast zitten, voor elke $h$. Naar links worden ze allebei beter, omhoog en omlaag veel sneller, tot de afronding het overneemt: onder ongeveer $10^{-5}$ voor omhoog en omlaag en $10^{-8}$ voor alleen omhoog groeit de fout weer.
        """),
        mo.Html(C.h_picture({"title": "Hoe ver de schatting ernaast zit, voor elke h", "x": "stap h", "y": "relatieve fout",
                             "forward": "alleen omhoog", "central": "omhoog en omlaag", "rounding": "afronding wint",
                             "alt": "Twee lijnen op twee logschalen: de relatieve fout tegen de stap h, van 10 tot de macht min 12 tot 0,1. Omhoog en omlaag daalt steil rechts van 10 tot de macht min 5 en is daar het laagst, ongeveer 10 tot de macht min 11; alleen omhoog, gestippeld, is het laagst, ongeveer 10 tot de macht min 8, bij h = 10 tot de macht min 8. Links van hun laagste punt groeien ze allebei weer, in een grillige lijn."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een backward pass te controleren:

    1. Reken de gradient uit met je backward pass: één getal $a$ voor elke parameter.
    2. Voor elke parameter: verschuif hem omhoog met $h$ en omlaag met $h$, reken beide keren de loss uit, en deel het verschil door $2h$: de schatting $n$.
    3. Reken de relatieve fout van elke parameter uit: $|a - n| / \max(|a|, |n|)$.
    4. Beoordeel: onder ongeveer $10^{-7}$ zijn ze het eens; boven ongeveer $10^{-3}$ zoek je de bug in de backward pass van die parameter.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén neuron met twee weights, een ReLU en als loss het kwadraat van de fout, voor het voorbeeld $\mathbf{x} = (1, 2)$ met label 0,5. `_grad` is zijn backward pass, met de hand uitgewerkt met de kettingregel. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, doe dan wat het commentaar zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_check():
    import numpy as np

    _x = np.array([1.0, 2.0])
    _y = 0.5

    def _loss(w):
        return (np.maximum(w @ _x, 0) - _y) ** 2

    def _grad(w):
        # de backward pass: 2 · (relu(z) − y), keer 1 waar z > 0, keer x
        _z = w @ _x
        return 2 * (np.maximum(_z, 0) - _y) * (_z > 0) * _x

    _w = np.array([0.3, 0.4])
    # 1. de gradient uit de backward pass
    _a = _grad(_w)
    # 2. voor elke weight: verschuif hem omhoog en omlaag met h, en de helling daartussen
    _h = 1e-5
    _n = np.zeros(2)
    for _j in range(2):
        _step = np.zeros(2)
        _step[_j] = _h
        _n[_j] = (_loss(_w + _step) - _loss(_w - _step)) / (2 * _h)
    print(_a, _n)
    # 3. de relatieve fout van elke weight
    print(np.abs(_a - _n) / np.maximum(np.abs(_a), np.abs(_n)))
    # 4. beoordeel, en test de check zelf: stop een bug in _grad door de 2 weg te laten, en voer opnieuw uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De backward pass en de stapjes geven allebei 1,2 en 2,4. De relatieve fouten zijn ongeveer $6 \cdot 10^{-12}$ en $10^{-12}$ (Python print `5.62...e-12` en `9.99...e-13`): ver onder $10^{-7}$, dus de backward pass klopt. Zonder de 2 geeft de backward pass 0,6 en 1,2, de stapjes nog steeds 1,2 en 2,4, en beide relatieve fouten zijn 0,5: ver boven $10^{-3}$. De check vangt de bug, terwijl de gradients nog steeds de goede kant op wijzen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bug():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waar zit de bug?
    """)
    return


@app.cell(hide_code=True)
def choice_bug_cell():
    choice_bug = L.choice(
        "which-bug",
        question="Gradient checking van vier weights geeft de relatieve fouten $2 \\cdot 10^{-9}$, $4 \\cdot 10^{-10}$, $0{,}31$ en $10^{-8}$. Wat doe je?",
        options={
            "mean": "Niets: gemiddeld zijn de vier fouten klein genoeg",
            "right": "Een bug zoeken: in de backward pass van de derde weight",
            "h": "$h$ kleiner maken: 0,31 is afronding, en een kleinere $h$ haalt die weg",
        },
        correct="right",
        why={
            "mean": "Eén weight met een verkeerde gradient is genoeg om slecht te trainen, en het gemiddelde verstopt dat. Elke parameter wordt apart beoordeeld.",
            "h": "0,31 is geen afronding: afronding geeft fouten ver onder $10^{-3}$, en een kleinere $h$ maakt afronding erger, niet beter.",
            "right": "Drie liggen onder $10^{-7}$, en 0,31 ligt ver boven $10^{-3}$: de backward pass klopt niet voor de derde weight, tenzij daar een input van een ReLU dicht bij 0 zit.",
        },
    )
    choice_bug
    return (choice_bug,)


@app.cell(hide_code=True)
def text_exercise_grad():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De schatting door te verschuiven

        Schrijf `numerical_grad(f, w, h)`: voor elk getal van de array `w` de centrale differentie van de functie `f`, met stap 2 van het recept, als array. De check roept `numerical_grad(lambda w: np.sum(w ** 2), np.array([1.0, -2.0]), 0.001)` aan, en dat moet `[2, -4]` geven; hij probeert ook andere functies, ook functies waarin de getallen van `w` samenwerken, zoals `np.sum(w) ** 2`.
        """),
        L.task("Laat `numerical_grad` de centrale differentie van `f` geven voor elk getal van `w`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grad():
    box_grad = L.code_box("numerical-grad", lines=10)
    box_grad
    return (box_grad,)


@app.cell(hide_code=True)
def check_grad(box_grad):
    result_grad = L.check("numerical-grad", box_grad.value)
    result_grad
    return (result_grad,)


@app.cell(hide_code=True)
def hints_grad():
    L.hints(
        "numerical-grad",
        [
            "Zoals in stap 2 van het uitgewerkte voorbeeld: een loop over `range(len(w))`, en voor elk getal een stap-array van nullen met `h` op één plek.",
            "`f(w + step)` en `f(w - step)` verschuiven alleen dat ene getal, en laten `w` zelf zoals hij is. Deel hun verschil door `2 * h`, en verzamel de hellingen in `np.zeros(len(w))`.",
        ],
        why="Bij `np.sum(w ** 2)` is de helling van elk getal twee keer dat getal: bij `[1, -2]` is dat `[2, -4]`. De centrale differentie geeft dat precies, omdat de functie in elk getal een parabool is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_error():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De relatieve fout

        Schrijf `relative_error(a, n)`: voor elke parameter de relatieve fout tussen de gradient `a` uit backprop en de schatting `n`, met stap 3 van het recept, als array. De check roept `relative_error(np.array([2.0, -3.0, 0.4]), np.array([2.0, -4.0, 0.5]))` aan, en dat moet `[0, 0.25, 0.2]` geven; hij probeert ook grote, kleine en negatieve gradients.
        """),
        L.task("Laat `relative_error` de relatieve fout van elke parameter geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_error():
    box_error = L.code_box("relative-error", lines=6)
    box_error
    return (box_error,)


@app.cell(hide_code=True)
def check_error(box_error):
    result_error = L.check("relative-error", box_error.value)
    result_error
    return (result_error,)


@app.cell(hide_code=True)
def hints_error():
    L.hints(
        "relative-error",
        [
            "De grootte van een getal is zijn waarde zonder teken: `np.abs`. Werk met hele arrays tegelijk, zoals `np.abs(a - n)`.",
            "`np.maximum(x, y)` neemt plek voor plek de grootste van twee arrays, zoals bij een ReLU.",
        ],
        why="Voor de tweede parameter: $|-3 - (-4)| = 1$, en de grootste grootte is 4: $1 / 4 = 0{,}25$. Voor de derde: $0{,}1 / 0{,}5 = 0{,}2$.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De helling die je in [De afgeleide als helling](/course/open/math-slope?lang=nl) door te verschuiven mat, genomen aan beide kanten, wordt een test voor de backward pass uit [Je MLP trainen: de training loop](/course/open/nn-micrograd-train?lang=nl): twee keer de loss per parameter, en een relatieve fout die zegt of een gradient klopt. In de volgende les, *Backprop door hele layers met de hand*, schrijf je de backward pass van een heel netwerk in NumPy, en deze check, en PyTorch, vertellen je of hij klopt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bug, result_grad, result_error):
    L.wrap_up(choice_bug, result_grad, result_error)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
