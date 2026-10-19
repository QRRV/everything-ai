# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-derivations. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-derivations", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van formule naar code](/course/open/res-formula-to-code?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Een paper schrijft $q(k)$ voor een verdeling over $K$ klassen, voor elk voorbeeld. Welke shape krijgt die in je code, voor een batch van $B$ voorbeelden?",
        options={
            "k": "(K,)",
            "right": "(B, K)",
            "kb": "(K, B)",
        },
        correct="right",
        why={
            "k": "Elk voorbeeld heeft een eigen verdeling: de batch-as komt eerst. Zie [Van formule naar code](/course/open/res-formula-to-code?lang=nl).",
            "kb": "De batch-as komt eerst, zoals bij de logits: een rij per voorbeeld. Zie [Van formule naar code](/course/open/res-formula-to-code?lang=nl).",
            "right": "Een rij per voorbeeld, een kolom per klasse. Deze les gaat een stap verder: van een formule naar de regels die ertoe leidden.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="$X$ is 0 of 2, elk met kans ½. Wat is $\\mathrm{Var}(X)$?",
        options={
            "two": "2",
            "half": "0,5",
            "right": "1",
        },
        correct="right",
        why={
            "two": "2 is het gemiddelde van de kwadraten van de waarden zelf, $\\tfrac12 \\cdot 0^2 + \\tfrac12 \\cdot 2^2$. De variantie kwadrateert hun afstanden tot de verwachtingswaarde 1: $\\tfrac12 \\cdot 1^2 + \\tfrac12 \\cdot 1^2$. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "half": "Elke waarde ligt 1 af van de verwachtingswaarde 1, en die afstand wordt gekwadrateerd: $\\tfrac12 \\cdot 1 + \\tfrac12 \\cdot 1$. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "$\\mathbb{E}[X] = 1$, en $\\tfrac12 (0 - 1)^2 + \\tfrac12 (2 - 1)^2 = 1$. Let op: $\\mathbb{E}[X^2] = 2$, en dat is niet de variantie. Deze les heeft dat verschil nodig.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Papers besparen ruimte: "hieruit volgt", "een eenvoudige berekening laat zien". De stap die ze overslaan, is vaak waar een aanname zich verstopt, en een aanname die voor jouw model niet geldt, maakt de uitkomst voor dat model fout. Een afleiding regel voor regel doorwerken, invullen wat is overgeslagen en elke regel met getallen controleren, vertelt je waar een uitkomst op rust.
    """)
    return


@app.cell(hide_code=True)
def predict_variance():
    guess_variance = L.guess(
        "relu-inputs",
        question="De formule van [Initialisatie](/course/open/nn-initialization?lang=nl) is $\\mathrm{Var}(z) = n_\\text{in} \\cdot \\mathrm{Var}(w) \\cdot \\mathrm{Var}(x)$. Een layer heeft 256 inputs en weights met $\\mathrm{Var}(w) = 1/256$. Zijn inputs komen uit een ReLU, $x = \\max(0, y)$ met $y$ standaardnormaal, en $\\mathrm{Var}(x) \\approx 0{,}34$. Wat is $\\mathrm{Var}(z)$, gemeten op 20.000 inputs?",
        options={
            "formula": "Ongeveer 0,34",
            "right": "Ongeveer 0,5",
            "half": "Ongeveer 0,17",
        },
        reveal=f"**Ongeveer 0,5:** {L.number(C.guess_value(), 3)}. De formule ging uit van inputs met gemiddelde 0, en na een ReLU is het gemiddelde ongeveer 0,4. Wat telt is niet $\\mathrm{{Var}}(x)$ maar het gemiddelde van de kwadraten, $\\mathbb{{E}}[x^2] = 0{{,}}5$. Deze les zoekt uit waar dat vandaan komt.",
    )
    guess_variance
    return (guess_variance,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De afleiding.** He en anderen (2015) leiden hun initialisatie af in een paar regels, voor een layer $y_l = W_l x_l$ met bias terms 0, waarvan de $n_l$ inputs uit een ReLU komen, $x_l = \max(0, y_{l-1})$. Elke weight wordt onafhankelijk getrokken, met gemiddelde 0:
        """),
        L.formula(r"(1)\quad \mathrm{Var}[[[hidden:y_l]]] = n_l \, \mathrm{Var}[[[param:w_l]] [[hidden:x_l]]]"),
        L.formula(r"(2)\quad \mathrm{Var}[[[param:w_l]] [[hidden:x_l]]] = \mathrm{Var}[[[param:w_l]]] \, \mathbb{E}[[[hidden:x_l]]^2]"),
        L.formula(r"(3)\quad \mathbb{E}[[[hidden:x_l]]^2] = \tfrac{1}{2} \mathrm{Var}[[[hidden:y_{l-1}]]]"),
        L.formula(r"(4)\quad \mathrm{Var}[[[hidden:y_l]]] = \tfrac{1}{2} n_l \, \mathrm{Var}[[[param:w_l]]] \, \mathrm{Var}[[[hidden:y_{l-1}]]]"),
        mo.md(r"""
        Dus blijft de variantie van layer tot layer gelijk als $\tfrac12 n_l \mathrm{Var}[w_l] = 1$: $\mathrm{Var}[w_l] = 2 / n_l$, He initialization. De nummers (1) tot en met (4) zijn van deze les: de paper nummert zijn vergelijkingen op zijn eigen manier, en je vindt deze vier regels in paragraaf 2.2. De paper geeft regels (2) en (3) elk één zin. Vul ze in.

        **Regel (1)** is de rekenregel van [Initialisatie](/course/open/nn-initialization?lang=nl): $y_l$ telt $n_l$ onafhankelijke producten op, en de varianties van onafhankelijke termen tellen op.

        **Regel (2).** Schrijf de definitie van de variantie uit: $\mathrm{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$. Het kwadraat is $X^2 - 2X\,\mathbb{E}[X] + (\mathbb{E}[X])^2$; de verwachtingswaarde van een som is de som van de verwachtingswaarden, en vaste getallen zoals $\mathbb{E}[X]$ komen ongewijzigd uit een verwachtingswaarde. Dus is de variantie $\mathbb{E}[X^2] - 2(\mathbb{E}[X])^2 + (\mathbb{E}[X])^2 = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$. Voor $X = w x$, met $w$ en $x$ onafhankelijk, is de verwachtingswaarde van een product het product van de verwachtingswaarden: $\mathbb{E}[w^2 x^2] = \mathbb{E}[w^2]\,\mathbb{E}[x^2]$ en $\mathbb{E}[w x] = \mathbb{E}[w]\,\mathbb{E}[x]$. Met $\mathbb{E}[w] = 0$ is de tweede 0, en is $\mathbb{E}[w^2] = \mathrm{Var}[w]$. Wat overblijft is regel (2): $\mathbb{E}[x^2]$, niet $\mathrm{Var}[x]$, omdat $x$ geen gemiddelde 0 heeft.

        **Regel (3).** $y_{l-1}$ heeft gemiddelde 0 en ligt symmetrisch rond 0, omdat zijn weights dat doen. Dan dragen de positieve helft en de negatieve helft van $y$ evenveel bij aan $\mathbb{E}[y^2]$. De ReLU houdt de positieve helft en maakt de rest 0, dus $\mathbb{E}[x^2] = \tfrac12 \mathbb{E}[y^2]$, en met gemiddelde 0 is $\mathbb{E}[y^2] = \mathrm{Var}[y]$.

        **Controleer elke regel met getallen.** Trek veel samples die aan de aannames voldoen, reken beide kanten van een regel uit, en vergelijk: ze moeten gelijk zijn binnen de ruis van de samples. Breek dan één aanname, en kijk welke regel niet meer klopt.

        De figuur: de variantie van $y_l$ na elk van 10 layers van 256 neuronen, zoals regel (4) hem voorspelt, de grijze lijnen, en zoals hij uitkomt, de stippen, gemiddeld over 8 runs. De y-as heeft een logaritmische schaal: gelijke stappen omlaag zijn gelijke factoren.
        """),
        mo.Html(C.layers_picture({"title": "Regel (4) en de meting", "x": "layer", "y": "variantie van y",
                                  "two": "Var[w] = 2/n", "one": "Var[w] = 1/n", "lang": L.lang,
                                  "alt": "Twee grijze lijnen met stippen op een logaritmische schaal over 10 layers. Met Var[w] = 2/n blijft de voorspelling op 1, en liggen de stippen tussen ongeveer 0,97 en 1. Met Var[w] = 1/n halveert de voorspelling bij elke layer, van 0,5 tot ongeveer 0,001, en liggen de stippen erop."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **afleiding** | | de regels die van wat aangenomen wordt naar een uitkomst leiden, elk volgend uit de regels ervoor met een rekenregel | |
        | **numerieke controle** van een regel | | beide kanten van de regel uitgerekend op veel samples die aan de aannames voldoen, en vergeleken | |
        | $\mathbb{E}[x^2]$ | E van x kwadraat | de verwachtingswaarde van het kwadraat van $x$: het gemiddelde van de kwadraten, niet het kwadraat van het gemiddelde | `np.mean(x ** 2)` |
        | $\mathrm{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | | de variantie, uitgeschreven: het gemiddelde van de kwadraten min het kwadraat van het gemiddelde | `np.var(x)` |
        """),
        L.watch_out("Twee kanten die op samples gelijk zijn, bewijzen een regel niet: je samples kunnen aan een aanname voldoen waar je model niet aan voldoet, zoals weights met precies gemiddelde 0 of een symmetrische y. Controleer de regel op de samples die de afleiding aanneemt, en daarna op samples zoals die in je model."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een afleiding te volgen:

    1. Schrijf hem uit, een regel per stap, en nummer de regels.
    2. Noem bij elke regel de rekenregel die hij gebruikt en wat hij aanneemt, zoals onafhankelijkheid, gemiddelde 0 of symmetrie.
    3. Waar een stap is overgeslagen, schrijf je een definitie of een rekenregel uit tot de volgende regel volgt.
    4. Controleer elke regel met getallen: op samples die aan de aannames voldoen, zijn beide kanten gelijk binnen de ruis; breek dan één aanname tegelijk, en kijk welke regel misgaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4 voor regel (1): de variantie van een som van $n = 16$ onafhankelijke producten, tegen $n$ keer de variantie van één product, op 100.000 samples. `np.var(a)` is de variantie van een array: het kwadraat van `np.std(a)`.
    """)
    return


@app.cell
def example_line1():
    import numpy as np

    _rng = np.random.default_rng(0)
    _n = 16
    _w = _rng.normal(0, 0.25, size=(100_000, _n))  # weights met gemiddelde 0
    _x = np.maximum(0, _rng.normal(size=(100_000, _n)))  # inputs uit een ReLU
    _terms = _w * _x  # de n producten van elk sample
    _left = np.var(_terms.sum(axis=1))  # Var[y]: y telt de n producten op
    _right = _n * np.var(_terms)  # n keer Var[w x]
    print(round(_left, 4), round(_right, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De twee kanten zijn gelijk op ongeveer 1% na: de ruis van 100.000 samples. Regel (1) klopt hier. Zijn aanname is dat de 16 producten onafhankelijk zijn: elke weight en elke input wordt apart getrokken. Met $\mathrm{Var}[w] = 0{,}25^2$ en $\mathbb{E}[x^2] = 0{,}5$ zijn ze allebei ongeveer $16 \times 0{,}0625 \times 0{,}5 = 0{,}5$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_line2():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De twee kanten van regel (2)

        Maak `line2_sides(ws, xs)` af: geef voor samples van de weights `ws` en van de inputs `xs` de linkerkant van regel (2), $\mathrm{Var}[w x]$, en de rechterkant, $\mathrm{Var}[w]\,\mathbb{E}[x^2]$, terug. De check roept `line2_sides([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 3.0, 3.0])` aan, wat ongeveer (3,3333; 3,3333) moet geven, en probeert lijsten waarin de twee kanten verschillen.
        """),
        L.task("Laat `line2_sides` beide kanten van regel (2) geven voor de samples."),
    ])
    return


@app.cell(hide_code=True)
def exercise_line2():
    box_line2 = L.code_box("line2-sides", lines=10)
    box_line2
    return (box_line2,)


@app.cell(hide_code=True)
def check_line2(box_line2):
    result_line2 = L.check("line2-sides", box_line2.value)
    result_line2
    return (result_line2,)


@app.cell(hide_code=True)
def hints_line2():
    L.hints(
        "line2-sides",
        [
            "`ws * xs` geeft de producten. $\\mathbb{E}[x^2]$ is het gemiddelde van de kwadraten: eerst kwadrateren, dan middelen.",
        ],
        why="`left = np.var(ws * xs)` en `right = np.var(ws) * np.mean(xs ** 2)`. In het voorbeeld zijn de weights ±1 met gemiddelde 0, dus $\\mathrm{Var}[w] = 1$, en $\\mathbb{E}[x^2] = 20 / 6$; de producten hebben dezelfde variantie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_line3():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De twee kanten van regel (3)

        Maak `line3_sides(ys)` af: geef voor samples `ys` van de waarden $y$ vóór de ReLU de linkerkant van regel (3), $\mathbb{E}[x^2]$ met $x = \max(0, y)$, en de rechterkant, $\tfrac12 \mathrm{Var}[y]$, terug. De check roept `line3_sides([-2.0, -1.0, 0.0, 1.0, 2.0])` aan, wat (1; 1) moet geven, en probeert lijsten die niet symmetrisch rond 0 liggen.
        """),
        L.task("Laat `line3_sides` beide kanten van regel (3) geven voor de samples."),
    ])
    return


@app.cell(hide_code=True)
def exercise_line3():
    box_line3 = L.code_box("line3-sides", lines=9)
    box_line3
    return (box_line3,)


@app.cell(hide_code=True)
def check_line3(box_line3):
    result_line3 = L.check("line3-sides", box_line3.value)
    result_line3
    return (result_line3,)


@app.cell(hide_code=True)
def hints_line3():
    L.hints(
        "line3-sides",
        [
            "`np.maximum(0, ys)` is de ReLU van elk sample. Kwadrateer na de ReLU, niet ervoor.",
        ],
        why="`left = np.mean(np.maximum(0, ys) ** 2)` en `right = 0.5 * np.var(ys)`. Voor het voorbeeld is $x = (0, 0, 0, 1, 2)$, dus $\\mathbb{E}[x^2] = 5 / 5 = 1$, en $\\mathrm{Var}[y] = 10 / 5 = 2$.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je controles op veel samples

    De cel hieronder geeft je twee functies 200.000 samples: één keer zoals de afleiding aanneemt, en één keer met één aanname gebroken: weights met gemiddelde 1 voor regel (2), en een $y$ met gemiddelde 0,5 voor regel (3). Zolang je functies niet goed zijn, nemen die van de course hun plaats in.
    """)
    return


@app.cell(hide_code=True)
def demo_lines(box_line2, box_line3, result_line2, result_line3):
    _d = C.demo(box_line2.value if result_line2.state == "correct" else None,
                box_line3.value if result_line3.state == "correct" else None)
    _r = _d["rows"]
    _who = ("je functies" if all(_d["yours"]) else "je `line2_sides` en de `line3_sides` van de course" if _d["yours"][0]
            else "de `line2_sides` van de course en je `line3_sides`" if _d["yours"][1] else "de functies van de course")
    mo.md("\n".join([
        f"Met {_who}:",
        "",
        "| regel | samples | links | rechts |",
        "|---|---|---|---|",
        f"| (2) | zoals aangenomen | {L.number(_r['2-holds'][0], 4)} | {L.number(_r['2-holds'][1], 4)} |",
        f"| (2) | weights met gemiddelde 1 | {L.number(_r['2-broken'][0], 4)} | {L.number(_r['2-broken'][1], 4)} |",
        f"| (3) | zoals aangenomen | {L.number(_r['3-holds'][0], 4)} | {L.number(_r['3-holds'][1], 4)} |",
        f"| (3) | $y$ met gemiddelde 0,5 | {L.number(_r['3-broken'][0], 4)} | {L.number(_r['3-broken'][1], 4)} |",
        "",
        "Zoals aangenomen zijn beide kanten gelijk binnen de ruis. Met weights waarvan het gemiddelde niet 0 is, mist regel (2) de term $\\mathbb{E}[w]^2 \\, \\mathrm{Var}[x]$; met een $y$ die omhoog geschoven is, houdt de ReLU er meer dan de helft van over, en klopt regel (3) niet. Een bias term die niet 0 is, schuift $y$ op dezelfde manier op. En regel (3) geldt alleen voor ReLU: GELU, de gladdere ReLU van GPT-2, houdt niet de helft van $\\mathbb{E}[y^2]$ over, ook niet bij een symmetrische $y$.",
    ]))
    return


@app.cell(hide_code=True)
def text_exercise_assumption():
    mo.md(r"""
    ### Oefening 3. Wat regel (3) aanneemt
    """)
    return


@app.cell(hide_code=True)
def choice_assumption_cell():
    choice_assumption = L.choice(
        "assumption",
        question="Welke aanname over $y_{l-1}$ gebruikt regel (3), $\\mathbb{E}[x_l^2] = \\tfrac12 \\mathrm{Var}[y_{l-1}]$?",
        options={
            "right": "Dat $y_{l-1}$ gemiddelde 0 heeft en er symmetrisch omheen ligt",
            "independent": "Dat $y_{l-1}$ onafhankelijk is van de weights $w_l$",
            "unit": "Dat $y_{l-1}$ variantie 1 heeft",
        },
        correct="right",
        why={
            "independent": "Onafhankelijkheid van de weights is wat regel (2) gebruikt. Regel (3) gaat alleen over de ReLU: hoeveel van $y$ hij overhoudt.",
            "unit": "Regel (3) klopt voor elke variantie: beide kanten schalen ermee. Wat hij nodig heeft, is dat de helft van het kwadraat van elke kant van 0 komt.",
            "right": "Dan houdt de ReLU precies de helft van $\\mathbb{E}[y^2]$ over, en is $\\mathbb{E}[y^2] = \\mathrm{Var}[y]$. De demo hierboven brak die aanname met een gemiddelde van 0,5.",
        },
    )
    choice_assumption
    return (choice_assumption,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De He initialization van [Initialisatie](/course/open/nn-initialization?lang=nl) is nu regel voor regel afgeleid, met de variantie van [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) uitgeschreven, en met getallen gecontroleerd, zoals [Van formule naar code](/course/open/res-formula-to-code?lang=nl) een formule tegen code controleerde. In een reproductie vertelt een afleiding die je hebt gecontroleerd aan welke aannames jouw opzet moet voldoen voordat je de uitkomst van de paper mag verwachten. De volgende les, [Kritisch lezen: claims tegen bewijs](/course/open/res-critical-reading?lang=nl), vraagt of de experimenten van een paper echt steunen wat hij beweert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_assumption, result_line2, result_line3):
    L.wrap_up(result_line2, result_line3, choice_assumption)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
