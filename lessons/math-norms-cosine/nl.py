# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-norms-cosine. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-norms-cosine", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="Wat is het inproduct van $(2, 3)$ en $(4, -1)$?",
        options={
            "products": "$(8, -3)$",
            "right": "5",
            "sign": "11",
        },
        correct="right",
        why={
            "products": "Dat zijn de producten. Wat is de laatste stap van een inproduct? Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "sign": "Kijk naar het tweede product: 3 keer −1. Welk teken krijgt dat? Zie het recept in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "right": "2 × 4 + 3 × (−1) = 8 − 3 = 5.",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def quick_check_np_dot():
    quick_np_dot = L.choice(
        "quick-np-dot",
        question="`a` en `b` zijn arrays van elk 3 getallen. Wat geeft `np.dot(a, b)`?",
        options={
            "right": "Eén getal",
            "three": "Een array van 3 getallen",
            "matrix": "Een matrix van 3 bij 3",
        },
        correct="right",
        why={
            "three": "Dat is `a * b`. Wat doet `np.dot` met die producten? Zie het uitgewerkte voorbeeld in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "matrix": "Een inproduct vermenigvuldigt de getallen op dezelfde plek, niet elk getal met elk getal. Zie *Het idee* in [Veel features: het inproduct](/course/open/math-dot-product?lang=nl).",
            "right": "`np.dot` vermenigvuldigt de getallen op dezelfde plek en telt de producten op: één getal.",
        },
    )
    quick_np_dot
    return (quick_np_dot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model ziet alles als vectoren: een huis, een student, later in de course een woord of een hele tekst. Om er twee te vergelijken, wil je weten hoe ver ze uit elkaar liggen, en of ze dezelfde kant op wijzen. Zoekmachines herkennen teksten over hetzelfde onderwerp aan de richting van hun vectoren.

    **Een voorbeeld uit het dagelijks leven.** Twee vrienden lopen vanaf dezelfde straathoek, allebei naar het noordoosten. De een stopt na 1 km, de ander loopt door tot 5 km. Ze zijn 4 km van elkaar af, maar ze liepen precies dezelfde kant op.
    """)
    return


@app.cell(hide_code=True)
def predict_same_way():
    guess_same_way = L.guess(
        "same-way",
        question="Twee vectoren: $\\mathbf{a} = (1, 1)$ en $\\mathbf{b} = (5, 5)$. Wat denk je dat klopt?",
        options={
            "close": "Ze liggen dicht bij elkaar, en wijzen dezelfde kant op",
            "far-same": "Ze liggen ver uit elkaar, maar wijzen dezelfde kant op",
            "far-other": "Ze liggen ver uit elkaar, en wijzen verschillende kanten op",
        },
        reveal="**Ver uit elkaar, maar dezelfde kant op.** $\\mathbf{b}$ is $\\mathbf{a}$ keer 5: dezelfde richting, vijf keer zo lang. De afstand tussen hun punten is ongeveer 5,7, maar hun cosine similarity, die alleen naar de richting kijkt, is precies 1.",
    )
    guess_same_way
    return (guess_same_way,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zie een vector als een pijl van de oorsprong, het punt (0, 0), naar het punt dat zijn getallen aangeven. Zijn norm is zijn lengte: de wortel uit de som van zijn kwadraten. Ken je Pythagoras van school: het is zijn regel, met zoveel kwadraten als de vector getallen heeft. Dat is ook de wortel uit het inproduct van de vector met zichzelf.
        """),
        L.formula(r"\lVert \mathbf{x} \rVert = \sqrt{x_1^2 + x_2^2 + \dots + x_d^2} = \sqrt{\mathbf{x} \cdot \mathbf{x}}"),
        mo.md(r"""
        De afstand tussen twee vectoren is de norm van hun verschil, $\lVert \mathbf{a} - \mathbf{b} \rVert$: de lengte van de rechte lijn tussen hun punten. De cosine similarity, geschreven als $\cos(\mathbf{a}, \mathbf{b})$, kijkt alleen naar de richting: het inproduct, gedeeld door de twee normen, zodat de lengtes wegvallen.
        """),
        L.formula(r"\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}"),
        mo.md(r"""
        Hij is 1 als $\mathbf{a}$ en $\mathbf{b}$ dezelfde kant op wijzen, 0 als ze haaks op elkaar staan, en −1 als ze tegengesteld wijzen. De naam komt van de cosinus van de hoek ertussen, uit de meetkunde; die heb je hier niet nodig, want de formule rekent hem uit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **norm** | | de lengte van een vector: de wortel uit de som van zijn kwadraten; nooit negatief | `np.linalg.norm(x)` |
        | **afstand** | | hoe ver twee vectoren uit elkaar liggen: de norm van hun verschil | `np.linalg.norm(a - b)` |
        | **cosine similarity** | | het inproduct gedeeld door het product van de twee normen: 1 bij dezelfde richting, 0 als ze haaks staan, −1 bij tegengestelde richtingen | `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` |
        | $\lVert \mathbf{x} \rVert$ | de norm van x | de norm van de vector $\mathbf{x}$ | `np.linalg.norm(x)` |
        """),
        L.watch_out("De cosine similarity negeert de lengte: (1, 1) en (100, 100) hebben een cosine similarity van 1. Telt de grootte mee, zoals bij de oppervlakte van twee huizen, gebruik dan de afstand."),
        mo.md(r"""
        Beweeg de slider: hij draait $\mathbf{b}$ weg van $\mathbf{a}$. $\mathbf{a}$ is 4 lang, $\mathbf{b}$ 3, en de stippellijn verbindt hun pijlpunten: de afstand. De andere grafiek toont de cosine similarity bij elke hoek; sleep eroverheen om de hoek te zetten.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_angle_cell():
    def _frame(angle):
        distance, similarity = C.measures(angle, 3)
        return (C.picture(angle, 3, {"title": "Twee vectoren", "every": "Bij elke hoek", "x": "hoek (°)", "y": "cosine similarity",
                                 "alt": f"Twee pijlen vanuit de oorsprong: a, 4 lang, en b, 3 lang, met {L.number(angle, 0)} graden ertussen. Een stippellijn verbindt hun pijlpunten: de afstand. Een tweede grafiek toont de cosine similarity bij elke hoek van 0 tot 180 graden: hij daalt van 1 via 0 bij 90 graden naar −1, langzaam bij 0 en bij 180 graden en het snelst rond 90."}, "angle"),
                f"""Met {L.number(angle, 0)}° ertussen, en $\\mathbf{{b}}$ 3 lang: de stippellijn, de afstand tussen de pijlpunten, is <span class="ea-num">{L.number(distance, 2)}</span>, en de cosine similarity is <span class="ea-num">{L.number(similarity, 2)}</span>.""")

    explore_angle = L.explore("angle", "hoek", _frame, start=45, low=0, high=180, step=15, role="", task="Bij welke hoek is de cosine similarity 0,5?")
    explore_angle
    return


@app.cell(hide_code=True)
def text_between():
    mo.md(r"""
    Bij 60° is de cosine similarity 0,5. Hij daalt niet gelijkmatig met de hoek: van 0° tot 60° verliest hij de helft, en van 60° tot 90° de andere helft.

    In het volgende blok blijft de hoek 45°, en maak je $\mathbf{b}$ alleen langer of korter. De andere grafiek toont beide getallen bij elke lengte: de afstand gestippeld, zoals de lijn tussen de pijlpunten, en de cosine similarity doorgetrokken; sleep eroverheen om de lengte te zetten.
    """)
    return


@app.cell(hide_code=True)
def explore_length_cell():
    def _frame(length):
        distance, similarity = C.measures(45, length)
        return (C.picture(45, length, {"title": "Twee vectoren", "every": "Bij elke lengte van b", "x": "lengte van b", "distance": "afstand", "similarity": "cosine similarity",
                                 "alt": f"Twee pijlen vanuit de oorsprong: a, 4 lang, en b, {L.number(length, 1)} lang, met {L.number(45, 0)} graden ertussen. Een stippellijn verbindt hun pijlpunten: de afstand. Een tweede grafiek toont bij elke lengte van b van 1 tot 5 de afstand, gestippeld, die het laagst is bij ongeveer 3, en de cosine similarity, doorgetrokken, die 0,71 blijft."}, "length"),
                f"""Met {L.number(45, 0)}° ertussen, en $\\mathbf{{b}}$ {L.number(length, 1)} lang: de stippellijn, de afstand tussen de pijlpunten, is <span class="ea-num">{L.number(distance, 2)}</span>, en de cosine similarity is <span class="ea-num">{L.number(similarity, 2)}</span>.""")

    explore_length = L.explore("length", "lengte", _frame, start=4, low=1, high=5, step=0.5, role="", task="Bij welke lengte van $\\mathbf{b}$ is de afstand het kleinst?")
    explore_length
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Alleen de afstand verandert. Die is het kleinst bij een lengte van ongeveer 3, waar hij 2,83 is, en niet bij 4, waar $\mathbf{b}$ even lang is als $\mathbf{a}$: bij 45° komt de pijlpunt van $\mathbf{b}$ het dichtst bij die van $\mathbf{a}$ als $\mathbf{b}$ iets korter is dan $\mathbf{a}$, en gaat hij weer verder weg als $\mathbf{b}$ nog korter wordt. De cosine similarity blijft bij elke lengte 0,71: hij kijkt alleen naar de richting.

    ### Recept

    Zo reken je de cosine similarity van twee vectoren uit:

    1. Reken hun inproduct uit: `np.dot(a, b)`.
    2. Reken beide normen uit: `np.linalg.norm(a)` en `np.linalg.norm(b)`.
    3. Deel het inproduct door het product van de normen, met haakjes om dat product.
    4. Lees het af: dicht bij 1 is dezelfde richting, dicht bij 0 haaks, dicht bij −1 tegengesteld.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Drie teksten, elk als vector: hoe vaak ze de woorden "bal", "doelpunt" en "stem" gebruiken. Tekst `_b` is lang en gaat over hetzelfde als `_a`; tekst `_c` gaat over verkiezingen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_cosine():
    import numpy as np

    _a = np.array([3, 2, 0])
    _b = np.array([12, 8, 1])
    _c = np.array([0, 1, 4])
    # 1. de inproducten
    print(np.dot(_a, _b), np.dot(_a, _c))
    # 2. de normen: van _a stap voor stap, dan alle drie met np.linalg.norm
    _total = 0
    for _i in range(len(_a)):
        _total = _total + _a[_i] ** 2
    print(_total, _total ** 0.5)
    print(np.linalg.norm(_a), np.linalg.norm(_b), np.linalg.norm(_c))
    # 3. deel door het product van de normen
    print(np.dot(_a, _b) / (np.linalg.norm(_a) * np.linalg.norm(_b)))
    print(np.dot(_a, _c) / (np.linalg.norm(_a) * np.linalg.norm(_c)))
    # en de afstanden, om te vergelijken
    print(np.linalg.norm(_a - _b), np.linalg.norm(_a - _c))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De loop telt voor `_a` 9 + 4 + 0 = 13 op, en de wortel daaruit, 3,61, is wat `np.linalg.norm` geeft. De cosine similarity van `_a` en `_b` is 0,998, bijna 1: dezelfde woorden in ongeveer dezelfde verhouding, alleen vier keer zo vaak. `_a` en `_c` krijgen 0,13: ze lijken nauwelijks op elkaar. De afstand vertelt iets anders: `_c` ligt op 5,10 van `_a`, dichterbij dan `_b` op 10,86, alleen omdat `_b` een langere tekst is. Om teksten over hetzelfde onderwerp te vinden, is de cosine similarity de betere maat. Woordtellingen zijn nooit negatief, dus hier zakt hij nooit onder 0; de slider voor de hoek bij 180° laat −1 zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_norm():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een norm met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_norm_cell():
    choice_norm = L.choice(
        "norm",
        question="Wat is de norm van de vector $(2, -3, 6)$?",
        options={
            "no-root": "49",
            "sum": "5",
            "right": "7",
        },
        correct="right",
        why={
            "no-root": "Dat is de som van de kwadraten. Welke stap komt daarna? Zie de formule in *Het idee*.",
            "sum": "Dat telt de getallen zelf op. Wat gebeurt er eerst met elk getal? Zie de formule in *Het idee*.",
            "right": "2² + (−3)² + 6² = 4 + 9 + 36 = 49, en de wortel uit 49 is 7. Door het kwadraat telt de −3 even zwaar als een 3.",
        },
    )
    choice_norm
    return (choice_norm,)


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Dezelfde muzieksmaak

        Hoeveel nummers twee mensen deze week beluisterden: pop, jazz en rock. `a` is de eerste persoon, `b` de tweede. Zet de cosine similarity van `a` en `b` in `similarity`. Vervang de `...`. De laatste regel is `similarity`, zodat de check hem ziet. De check probeert ook een andere tweede persoon.
        """),
        L.task("Zet de cosine similarity van `a` en `b` in `similarity`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("cosine", lines=7)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("cosine", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "cosine",
        [
            "Het recept heeft de stappen: het inproduct, de twee normen, en een deling.",
            "Het inproduct komt boven de streep. Wat komt eronder, en wat heeft Python nodig om door dat hele stuk te delen?",
            "`similarity = np.dot(a, b) / (___ * ___)`.",
        ],
        why="`np.dot(a, b)` is 10 × 4 + 2 × 1 + 5 × 2 = 52. De normen zijn ongeveer 11,36 en 4,58, en hun product is ongeveer 52,05, dus de cosine similarity is ongeveer 0,999. Door de haakjes vermenigvuldigt Python de normen voordat hij deelt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een afstand zonder NumPy

        Maak `distance(a, b)` af. Hij krijgt twee lijsten met getallen van dezelfde lengte, en geeft de afstand ertussen terug. Vervang de `...`. De check roept `distance([1, 2], [4, 6])` aan: dat moet `5.0` geven. Hij probeert ook andere lijsten, sommige met negatieve getallen.
        """),
        L.task("Laat `distance` de afstand tussen `a` en `b` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_distance():
    box_distance = L.code_box("distance-loop", lines=6)
    box_distance
    return (box_distance,)


@app.cell(hide_code=True)
def check_distance(box_distance):
    result_distance = L.check("distance-loop", box_distance.value)
    result_distance
    return (result_distance,)


@app.cell(hide_code=True)
def hints_distance():
    L.hints(
        "distance-loop",
        [
            "De afstand is de norm van het verschil: plek voor plek aftrekken, kwadrateren, optellen en de wortel nemen.",
            "Een loop over de plekken, `for i in range(len(a)):`, met een totaal dat groeit. Wat tel je erbij op voor plek `i`?",
            "`total = total + (a[i] - b[i]) ** ___` in de loop, en daarna `return total ** ___`.",
        ],
        why="Begin met `total = 0`. Voor elke plek is `(a[i] - b[i]) ** 2` het gekwadrateerde verschil; de loop telt ze op, en `** 0.5` neemt de wortel. In het voorbeeld: 3² + 4² = 25, en de wortel uit 25 is `5.0`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Veel features: het inproduct](/course/open/math-dot-product?lang=nl) vermenigvuldigde je twee vectoren plek voor plek en telde je op. De wortel uit het inproduct van een vector met zichzelf is zijn lengte, de norm; de norm van een verschil is een afstand; en het inproduct gedeeld door twee normen zegt of twee vectoren dezelfde kant op wijzen. Later in de course worden woorden vectoren, en woorden met een vergelijkbare betekenis krijgen een hoge cosine similarity. In de volgende les, *Snelheden vermenigvuldigen: de kettingregel*, zie je hoe snelheden van verandering met elkaar vermenigvuldigd worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_norm, result_cosine, result_distance):
    L.wrap_up(choice_norm, result_cosine, result_distance)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
