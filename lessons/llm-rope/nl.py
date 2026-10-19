# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-rope. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-rope", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl) en [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl). <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def quick_check_turn():
    quick_turn = L.choice(
        "quick-turn",
        question="Je draait het paar $(5, 12)$ over 90 graden, tegen de klok in. Wat krijg je?",
        options={
            "right": "$(-12, 5)$",
            "clock": "$(12, -5)$",
            "same": "$(5, 12)$: een draaiing laat de getallen gelijk",
        },
        correct="right",
        why={
            "right": "$(a\\cos\\theta - b\\sin\\theta;\\; a\\sin\\theta + b\\cos\\theta)$ met $\\cos 90° = 0$ en $\\sin 90° = 1$. De lengte blijft 13.",
            "clock": "Dat is een kwartslag met de klok mee, over $-90$ graden. Zie [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl).",
            "same": "Een draaiing laat de lengte gelijk, 13, maar niet de getallen zelf. Zie [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl).",
        },
    )
    quick_turn
    return (quick_turn,)


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="Hoeveel parameters heeft de sinusoidal positional encoding uit [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl)?",
        options={
            "places": "Eén per plek: een getal dat zegt waar het token staat",
            "both": "Eén per plek en per feature",
            "right": "Geen: die wordt uitgerekend",
        },
        correct="right",
        why={
            "places": "Dat zou een geleerde tabel zijn, met één getal per plek. De sinusoïden komen uit een formule. Zie [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl).",
            "both": "Dat is de geleerde position embedding van GPT-2. De sinusoïden komen uit een formule, zonder iets te leren.",
            "right": "Sinussen en cosinussen van $t / 10000^{2j/d}$: niets te leren. RoPE gebruikt dezelfde snelheden, en heeft ook geen parameters.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je GPT telt bij elk token een geleerde vector op voor zijn plek: één rij per plek, tot de langste tekst waarop hij trainde. Llama, Mistral, Qwen en de meeste open modellen sindsdien doen iets anders: **rotary position embeddings**, RoPE. Die tellen niets op bij de tokens. In elke attention-layer draaien ze elke query en elke key over een hoek die van zijn plek afhangt.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same-distance",
        question="Met RoPE vergelijk je een query op plek 10 met een key op plek 7. Dan zet je dezelfde twee vectoren, vóór het draaien, op de plekken 110 en 107. Wat gebeurt er met hun score?",
        options={
            "lower": "Die is lager: de tokens staan verder van het begin",
            "right": "Die blijft gelijk",
            "other": "Die is anders, op een manier die van de vectoren afhangt",
        },
        reveal="**Die blijft gelijk.** De query draait over zijn plek en de key over de zijne, dus hun score hangt alleen af van het verschil van hun plekken: in beide gevallen 3. Zoals [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl) liet zien, ziet een inproduct van gedraaide paren alleen de verschillen van hun hoeken.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: draai elk paar over zijn plek.** Lees een query of key van $d$ getallen als $d/2$ paren: getal 0 en 1, 2 en 3, enzovoort. Op plek $t$ draait paar $j$ over de hoek
        """),
        L.formula(r"t \, \theta_j \qquad \theta_j = 10000^{-2j/d}"),
        mo.md(r"""
        Dat zijn de snelheden van de sinusoïden uit [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl): paar 0 draait 1 radiaal per plek, het laatste paar nauwelijks. In code heet de 10000 `base`. Snelle paren houden buren uit elkaar, langzame paren tokens die ver uit elkaar staan.

        **Stap 2: alleen de afstand telt.** Een query op plek $m$ draait over $m\theta_j$, een key op plek $n$ over $n\theta_j$, en hun inproduct hangt alleen af van de verschillen, $(m - n)\theta_j$. Dus de score tussen de twee hangt af van $m - n$: twee tokens die 3 plekken uit elkaar staan, scoren overal in de tekst hetzelfde.

        **Stap 3: waar het zit.** In elke attention-layer, na de projecties, op de queries en keys van elke head, met $d$ de grootte van één head. De values worden niet gedraaid. Er zijn geen position embeddings bij de input, en RoPE heeft geen parameters.

        **Stap 4: wat het oplevert.** Relatieve posities, zonder tabel. Wijzen een query en een key dezelfde kant op, zoals de 64 enen van het plaatje, dan zakt hun score meestal naarmate de tokens verder uit elkaar staan. Een model kent nog steeds alleen de afstanden waarop het trainde: daarbuiten bereiken de langzame paren hoeken die het nooit zag. Manieren om de context op te rekken, zoals position interpolation, laten het draaien langzamer gaan zodat een langere tekst binnen de hoeken blijft die het model kent, meestal met een korte extra training op lange teksten.

        Het plaatje: de score van een query en een key van 64 enen, allebei gedraaid met RoPE en gedeeld door $\sqrt{64}$ zoals in attention, tegen de afstand tussen hun plekken.
        """),
        mo.Html(C.distance_picture({"title": "Score tegen afstand", "x": "afstand tussen de plekken", "y": "score", "lang": L.lang,
                                    "alt": "Een kromme over afstanden 0 tot 60. De score is 8 bij afstand 0, zakt tot ongeveer 5,3 bij afstand 10, en schommelt daarna op en neer terwijl hij langzaam zakt, tussen ongeveer 3,3 en 5,3, tot ongeveer 4 bij afstand 60."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **RoPE** | rotary position embeddings: elk paar van een query of key draaien over zijn plek maal zijn snelheid | `rope(q)`, `rope(k)` |
        | $\theta_j$ | de snelheid van paar $j$: radialen per plek | `base ** (-2 * j / d)` |
        """),
        L.watch_out("Code gebruikt twee indelingen. Deze les, het artikel over RoPE en de Llama-code van Meta koppelen buren, getal 0 en 1; de Llama van Hugging Face koppelt de eerste helft van een vector aan de tweede helft. De weights van een getraind model verwachten de indeling waarmee het getraind is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo pas je RoPE toe op een tensor met queries of keys, met de plekken langs de op een na laatste axis:

    1. Maak de snelheden, $\theta_j = 10000^{-2j/d}$, één per paar.
    2. Maak de hoeken: plek maal snelheid, een tabel met één rij per plek en één kolom per paar.
    3. Draai elk paar over zijn hoek: $(a\cos - b\sin;\; a\sin + b\cos)$.
    4. Doe het met de queries en de keys, vóór hun scores.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tabel met hoeken is nieuw: de snelheden van 4 paren, $d = 8$, en hun hoeken op de plekken 0 tot 3.
    """)
    return


@app.cell
def example_angles():
    import torch

    _d, _n = 8, 4
    _theta = 10000 ** (-torch.arange(0, _d, 2) / _d)
    _angles = torch.arange(_n)[:, None] * _theta[None, :]
    print("snelheden:", [round(_s, 3) for _s in _theta.tolist()])
    for _t in range(_n):
        print("plek", _t, [round(_a, 3) for _a in _angles[_t].tolist()])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `torch.arange(0, 8, 2)` geeft de getallen 0, 2, 4 en 6, het eerste van elk paar, dus de snelheden zijn $10000^{-0/8}, 10000^{-2/8}, \ldots$: 1; 0,1; 0,01 en 0,001. `[:, None]` maakt van de plekken een kolom en `[None, :]` van de snelheden een rij; hun product is de tabel met hoeken, plek maal snelheid, met een rij nullen voor plek 0: het eerste token wordt niet gedraaid.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rope():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Je eigen rope

        Schrijf `rope(x, base=10000.0)`: `x` heeft de plekken langs zijn op een na laatste axis en een even aantal getallen langs zijn laatste, met willekeurige axes ervoor, zoals (teksten, heads, plekken, getallen). Geef een tensor met dezelfde shape terug, met elk paar gedraaid over zijn hoek. `x[..., 0::2]` pakt de getallen 0, 2, 4, … van de laatste axis, welke axes er ook voor staan: `...` staat voor al die axes. `torch.cos` en `torch.sin` nemen de cosinus en de sinus van elk getal van een tensor, zoals `np.cos` en `np.sin` dat doen voor een array. De check roept `rope(torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]))` aan, wat ongeveer `[[1, 0, 1, 0], [0.5403, 0.8415, 0.9999, 0.01]]` moet geven, en probeert andere shapes.
        """),
        L.task("Schrijf `rope`: de hoeken van alle plekken en paren, en elk paar van `x` gedraaid over zijn hoek."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rope():
    box_rope = L.code_box("rope", lines=11)
    box_rope
    return (box_rope,)


@app.cell(hide_code=True)
def check_rope(box_rope):
    result_rope = L.check("rope", box_rope.value)
    result_rope
    return (result_rope,)


@app.cell(hide_code=True)
def hints_rope():
    L.hints(
        "rope",
        [
            "De tabel met hoeken uit het uitgewerkte voorbeeld, en dan de draaiing uit [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl) op alle paren tegelijk: `x[..., 0::2]` zijn de eerste getallen van alle paren, `x[..., 1::2]` de tweede.",
        ],
        why="Met `a = x[..., 0::2]` en `b = x[..., 1::2]` zijn de gedraaide paren `a * cos - b * sin` en `a * sin + b * cos`, met `cos` en `sin` van de tabel met hoeken; teruggeschreven in de even en de oneven getallen van een nieuwe tensor houden ze de indeling van `x`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Dezelfde afstand, andere plekken

    De cel hieronder draait één willekeurige query en één willekeurige key van 64 getallen met je `rope`, op vijf paren plekken, en geeft hun score, gedeeld door $\sqrt{64}$ zoals in attention. Hij werkt zodra oefening 1 goed is.
    """)
    return


@app.cell(hide_code=True)
def demo_places(box_rope, result_rope):
    if result_rope.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 af: deze cel gebruikt je `rope`.*")
    else:
        _rows = C.demo(box_rope.value)
        if _rows == "error":
            _out = mo.md("*Je `rope` stopte met een fout op vectoren van 64 getallen op 504 plekken.*")
        else:
            _table = "\n".join(f"| {_m} | {_k} | {_m - _k} | {C.number(_s, 3, L.lang)} |" for _m, _k, _s in _rows)
            _out = mo.md(f"""
    | query op | key op | afstand | score |
    |---|---|---|---|
    {_table}

    Op afstand 3 is de score hetzelfde op de plekken 3 en 0, 10 en 7, en 503 en 500. Op andere afstanden is het een andere score.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Oefening 2. Waar de plek binnenkomt
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "where",
        question="Waar komt in een GPT met RoPE de plek van een token binnen?",
        options={
            "input": "Opgeteld bij de embeddings van de tokens, één keer, bij de input",
            "right": "In elke attention-layer: queries en keys worden over hun plek gedraaid",
            "values": "In elke attention-layer: queries, keys en values worden gedraaid, zodat de output de plek ook meedraagt",
        },
        correct="right",
        why={
            "input": "Daar komen de geleerde position embeddings van GPT-2 binnen. RoPE telt niets op bij de input.",
            "values": "RoPE laat de values met rust: de plek doet ertoe voor de scores, die bepalen waar een token kijkt.",
            "right": "Na de projecties, in elke head van elke layer, en alleen op de queries en keys.",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_exercise_values():
    mo.md(r"""
    ### Oefening 3. Niet de values
    """)
    return


@app.cell(hide_code=True)
def explain_values():
    explain_values_cell = L.explain(
        "not-the-values",
        question="RoPE draait de queries en de keys, maar niet de values. Waarom is dat genoeg om een model te laten gebruiken waar zijn tokens staan, en wat zou het ook draaien van de values doen?",
        model_answer="**Waarom genoeg:** de scores, uit queries en keys, bepalen hoeveel elk token van elk ander token neemt. Met RoPE hangen ze af van de afstand tussen de twee, dus de attention weights dragen al mee waar alles staat: een head kan leren naar het token ervoor te kijken, of naar tokens ver terug. **De values:** die dragen de inhoud die gemengd wordt. Gedraaid over hun eigen plek zouden de values van verschillende tokens voor dezelfde inhoud verschillende kanten op wijzen, en zou de gewogen som kopieën mengen die over verschillende hoeken gedraaid zijn en elkaar deels opheffen; de layers na attention zouden die som niet zomaar terug kunnen draaien.",
        points=[
            "je zegt dat de scores bepalen waar een token kijkt, en dat ze met RoPE van de afstand afhangen",
            "je zegt dat de values de inhoud dragen die gemengd wordt",
            "je zegt dat het draaien van de values dezelfde inhoud in verschillende richtingen zou mengen, wat de layers na attention niet zomaar terug kunnen draaien",
        ],
    )
    explain_values_cell
    return (explain_values_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De draaiingen uit [Vectoren draaien en complexe getallen](/course/open/math-complex-rotation?lang=nl), met de snelheden van de sinusoïden uit [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl), zitten nu in de attention van [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl). De volgende les, *Grouped-query attention*, verandert hoeveel keys en values attention bewaart. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, explain_values_cell, result_rope):
    L.wrap_up(result_rope, choice_where, explain_values_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
