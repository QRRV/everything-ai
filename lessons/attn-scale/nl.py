# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-scale. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-scale", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Queries, keys en values](/course/open/attn-qkv?lang=nl) en [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_value():
    quick_value = L.choice(
        "quick-value",
        question="In self-attention is de nieuwe vector van een woord een gewogen gemiddelde. Waarvan?",
        options={
            "query": "Van de queries van de woorden",
            "key": "Van de keys van de woorden",
            "right": "Van de values van de woorden",
        },
        correct="right",
        why={
            "query": "De query bepaalt alleen waar het woord zelf kijkt. Zie [Queries, keys en values](/course/open/attn-qkv?lang=nl).",
            "key": "De key bepaalt hoeveel andere woorden ervan nemen, niet wat ze nemen. Zie [Queries, keys en values](/course/open/attn-qkv?lang=nl).",
            "right": "Queries en keys maken alleen de scores. Vandaag kijk je naar de grootte van die scores.",
        },
    )
    quick_value
    return (quick_value,)


@app.cell(hide_code=True)
def quick_check_saturation():
    quick_saturation = L.choice(
        "quick-saturation",
        question="Waar een activatiefunctie verzadigt, zoals de sigmoid bij $z = 10$, wat komt er dan op de terugweg aan bij de layers ervoor?",
        options={
            "right": "Bijna geen gradient: de helling is daar bijna 0",
            "full": "De hele gradient: verzadiging verandert alleen de forward pass",
            "more": "Een grotere gradient, omdat de input van de functie groot is",
        },
        correct="right",
        why={
            "full": "Op de terugweg wordt de gradient vermenigvuldigd met de helling, en in het vlakke deel is de helling bijna 0. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "more": "Een grote input komt in het vlakke deel van de kromme, waar de helling het kleinst is. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "right": "Vandaag blijkt de softmax van attention op dezelfde manier te verzadigen, als zijn scores groot worden.",
        },
    )
    quick_saturation
    return (quick_saturation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een inproduct telt één product op voor elk getal van de twee vectoren. Echte queries en keys hebben veel getallen: 64 in GPT-2. Hoe langer ze zijn, hoe groter de scores worden, en grote scores maken de softmax heel scherp: bijna alle weight gaat naar één woord. Dat lijkt een duidelijke keuze, maar zo stokt het leren: de softmax verzadigt dan, zoals een sigmoid ver van 0, en er gaat nauwelijks nog gradient terug naar de queries en keys. Het model kan dan niet meer leren waar het moet kijken. De oplossing is één deling.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "spread",
        question="Een query en een key hebben elk $d_k$ willekeurige getallen met gemiddelde 0 en standaardafwijking 1. Met 4 getallen spreiden hun inproducten ongeveer 2 rond 0. Hoe ver spreiden ze met 64 getallen?",
        options={
            "same": "Ongeveer 2: de getallen zelf zijn niet groter",
            "linear": "Ongeveer 32: 16 keer zoveel getallen, 16 keer de spreiding",
            "right": "Ongeveer 8: 16 keer zoveel getallen, 4 keer de spreiding",
        },
        reveal="**Ongeveer 8.** De spreiding groeit met de wortel uit het aantal producten: $\\sqrt{64} = 8$. De producten hebben gemiddelde 0, dus veel vallen tegen elkaar weg: hun varianties tellen op, niet hun groottes. Met 1.024 getallen is de spreiding ongeveer 32.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Stel dat de getallen van een query en een key willekeurig en onafhankelijk zijn, zodat het ene getal niets zegt over een ander, elk met gemiddelde 0 en variantie 1, de variantie uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl). Twee regels uit de kansrekening, die deze les aanneemt zoals ze zijn: het product van twee zulke getallen, $q_i k_i$, heeft weer gemiddelde 0 en variantie 1; en de variantie van een som van onafhankelijke getallen is de som van hun varianties. Dus heeft de score $\mathbf{q} \cdot \mathbf{k}$, een som van $d_k$ producten, variantie $d_k$, en een standaardafwijking van $\sqrt{d_k}$. Als je elke score deelt door $\sqrt{d_k}$, is de spreiding weer 1, wat $d_k$ ook is. Dat is **scaled dot-product attention**, de attention van de transformer, uit *Attention Is All You Need* (Vaswani en anderen, 2017):
        """),
        L.formula(r"[[hidden:\mathbf{H}]] = \mathrm{softmax}\!\left(\frac{[[hidden:Q]] [[hidden:K]]^\top}{\sqrt{[[hyper:d_k]]}}\right) [[hidden:V]]"),
        mo.md(r"""
        De softmax gaat nog steeds langs elke rij. Echte queries en keys zijn niet willekeurig, maar ze beginnen bijna willekeurig, en de deling houdt de scores in een gebied waar de softmax nog gradient doorlaat.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **scaled dot-product attention** | | attention waarvan de scores vóór de softmax gedeeld worden door $\sqrt{d_k}$ | `torch.softmax(Q @ K.T / math.sqrt(d_k), dim=1) @ V` |
        | $\sqrt{d_k}$ | wortel uit d k | de spreiding van het inproduct van twee willekeurige vectoren van $d_k$ getallen met spreiding 1 | `math.sqrt(d_k)` |
        """),
        L.watch_out("Een scherpe softmax is geen teken dat het model weet waar het moet kijken: met lange, ongeschaalde keys is hij al scherp vanaf het begin, voordat het model iets geleerd heeft, en dan leert het nauwelijks. De schaal is ook geen geleerde weight: $\\sqrt{d_k}$ ligt vast door de lengte van de keys."),
        mo.md(r"""
        De figuur: willekeurige queries, met 16 willekeurige keys voor elk, getallen met spreiding 1; voor elke lengte van de keys het gemiddelde van de grootste van de 16 attention weights, over 2.000 queries, seed 0. Zonder schalen klimt de grootste weight van 0,43 naar 0,96: bijna one-hot. Geschaald blijft hij rond 0,25. De gradient van de softmax krimpt mee met de scherpte: zonder schalen is hij bij 1.024 getallen 6 keer kleiner dan bij 4; geschaald verandert hij nauwelijks.
        """),
        mo.Html(C.sharpness_picture({"title": "De grootste van 16 attention weights", "x": "lengte van de keys, d_k", "y": "gemiddelde grootste weight",
                                     "plain": "niet geschaald", "scaled": "gedeeld door √d_k", "even": "alle 16 gelijk: 1/16", "decimal": ",",
                                     "alt": "Twee lijnen tegen de lengte van de keys 4, 16, 64, 256 en 1.024. Niet geschaald klimt de gemiddelde grootste attention weight van 0,43 naar 0,96. Gedeeld door de wortel uit d_k blijft hij rond 0,25. Een gestippelde lijn bij 1/16 geeft alle weights gelijk aan."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Scaled dot-product attention, met één query, key en value per rij:

    1. De lengte van de keys, $d_k$: het aantal kolommen van `Q`.
    2. De scores, elke query met elke key, gedeeld door $\sqrt{d_k}$.
    3. De weights: de softmax van elke rij.
    4. De nieuwe vectoren: de weights keer de values.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén willekeurige query tegen 4 willekeurige keys van elk 1.024 getallen. `_rng.normal(size=(4, 1024))` vraagt de generator uit [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) om willekeurige getallen met gemiddelde 0 en standaardafwijking 1, in een array van die shape. Het commentaar met `#` zegt welke stap van het recept elk deel is; stap 4, de weights maal de values, gaat zoals in [Queries, keys en values](/course/open/attn-qkv?lang=nl). Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_scale():
    import math
    import numpy as np

    _rng = np.random.default_rng(1)
    _q = _rng.normal(size=(1, 1024))
    _K = _rng.normal(size=(4, 1024))
    # 1. de lengte van de keys
    _d_k = _q.shape[1]
    # 2. de scores
    _S = _q @ _K.T  # maak af: gedeeld door math.sqrt(_d_k)
    print(np.round(_S, 2))
    # 3. de weights: de softmax van elke rij
    _S = _S - _S.max(axis=1, keepdims=True)
    _A = np.exp(_S) / np.sum(np.exp(_S), axis=1, keepdims=True)
    print(np.round(_A, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, lopen de scores van ongeveer −27 tot 52, en geeft de softmax de grootste alle weight: 1, en de andere drie 0, op drie decimalen. Een kleine verandering in een score verandert daar niets: de gradient is ongeveer 0. Afgemaakt wordt elke score gedeeld door $\sqrt{1{.}024} = 32$: de scores worden −0,86, 1,61, −0,8 en −0,41, en de weights 0,065, 0,765, 0,069 en 0,101. De beste key krijgt nog steeds het meest, maar een kleine verandering in een score verandert nu de weights, dus de gradient heeft iets om mee te werken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_divide():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Te veel delen
    """)
    return


@app.cell(hide_code=True)
def choice_divide_cell():
    choice_divide = L.choice(
        "divide-too-much",
        question="Met keys van 1.024 getallen deel je de scores door $d_k = 1{.}024$ in plaats van door $\\sqrt{d_k} = 32$. Wat gebeurt er met de attention weights?",
        options={
            "sharper": "Die worden scherper: één key krijgt bijna alle weight",
            "right": "Die worden bijna gelijk: elke key krijgt ongeveer evenveel weight",
            "same": "Niets: de softmax geeft bij elke deling dezelfde weights",
        },
        correct="right",
        why={
            "sharper": "Door meer delen worden de scores kleiner, niet groter: hun spreiding wordt $32 / 1{.}024 = 1/32$. Kleine scores geven een vlakke softmax.",
            "same": "De softmax verandert niet als je bij elke score hetzelfde getal optelt, maar delen verandert hem wel: dat drukt de scores bij elkaar.",
            "right": "De scores spreiden dan maar $1/32$: allemaal dicht bij 0, dus $e$ tot de macht van elk is ongeveer 1, en elke key krijgt ongeveer hetzelfde deel. Het model zou nauwelijks verschil tussen keys zien.",
        },
    )
    choice_divide
    return (choice_divide,)


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Meet de spreiding

        Maak `score_spread(d_k)` af: de functie maakt 10.000 willekeurige queries en 10.000 willekeurige keys van `d_k` getallen, met gemiddelde 0 en spreiding 1, neemt het inproduct van elke query met de key in dezelfde rij, en geeft de standaardafwijking van die scores. Vervang de twee `...`. De check roept `score_spread(64)` aan, wat ongeveer `8.08` moet geven, en probeert andere lengtes.
        """),
        L.task("Laat `score_spread` meten hoe ver de scores van willekeurige queries en keys spreiden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spread():
    box_spread = L.code_box("score-spread", lines=11)
    box_spread
    return (box_spread,)


@app.cell(hide_code=True)
def check_spread(box_spread):
    result_spread = L.check("score-spread", box_spread.value)
    result_spread
    return (result_spread,)


@app.cell(hide_code=True)
def hints_spread():
    L.hints(
        "score-spread",
        [
            "`q * k` vermenigvuldigt de twee arrays getal voor getal, rij voor rij. Een inproduct telt de producten van één rij op: een som langs `axis=1`.",
            "De spreiding van een kolom getallen is zijn standaardafwijking, zoals in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl): `np.std`.",
        ],
        why="Elke rij geeft één score, een som van `d_k` producten, en de standaardafwijking over de 10.000 scores is ongeveer $\\sqrt{d_k}$: 8,08 bij 64, dicht bij 8, omdat een meting over willekeurige getallen nooit precies klopt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scaled():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Geschaalde attention

        Maak `scaled_attention(Q, K, V)` af: één query, key en value per rij; de functie geeft de nieuwe vectoren van scaled dot-product attention. De eerste regels maken van de lijsten tensors en vinden $d_k$; stap 3 en 4 staan er al. Vervang de `...` volgens stap 2 van het recept. De check roept `scaled_attention` aan met 3 woorden waarvan de queries en keys 4 getallen hebben, wat `[[0.9841, 0.1332], [0.1332, 0.9841], [0.6667, 0.6667]]` moet geven, en probeert andere lengtes.
        """),
        L.task("Laat `scaled_attention` de scores delen door de wortel uit de lengte van de keys."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scaled():
    box_scaled = L.code_box("scaled-attention", lines=12)
    box_scaled
    return (box_scaled,)


@app.cell(hide_code=True)
def check_scaled(box_scaled):
    result_scaled = L.check("scaled-attention", box_scaled.value)
    result_scaled
    return (result_scaled,)


@app.cell(hide_code=True)
def hints_scaled():
    L.hints(
        "scaled-attention",
        [
            "Stap 2 van het afgemaakte uitgewerkte voorbeeld, met `Q`, `K` en `d_k`.",
            "`math.sqrt(d_k)` is de wortel; de scores zijn elke query met elke key, zoals in [Queries, keys en values](/course/open/attn-qkv?lang=nl).",
        ],
        why="Met 4 getallen per key wordt elke score door 2 gedeeld: de scores 8, 0 en 4 van het eerste woord worden 4, 0 en 2, en zijn weights 0,87, 0,02 en 0,12 in plaats van 0,98, 0,00 en 0,02.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De scores uit [Queries, keys en values](/course/open/attn-qkv?lang=nl) groeien als $\sqrt{d_k}$, omdat hun varianties optellen; te groot laten ze de softmax verzadigen, zoals de sigmoid in [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl). Als je ze deelt door $\sqrt{d_k}$, krijg je scaled dot-product attention. In de volgende les, *De causal mask*, leert een language model het volgende woord te voorspellen zonder vooruit te kijken: elk woord mag alleen nemen van de woorden ervoor.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_divide, result_spread, result_scaled):
    L.wrap_up(choice_divide, result_spread, result_scaled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
