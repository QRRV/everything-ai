# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-causal-mask. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-causal-mask", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De scores schalen](/course/open/attn-scale?lang=nl) en [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Waarom deelt scaled dot-product attention zijn scores door $\\sqrt{d_k}$?",
        options={
            "sum": "Zodat de weights van elke rij optellen tot 1",
            "right": "Zodat de scores niet meegroeien met de lengte van de keys, en de softmax niet verzadigt",
            "speed": "Zodat de matrixvermenigvuldiging sneller gaat",
        },
        correct="right",
        why={
            "sum": "De softmax laat elke rij al optellen tot 1, met of zonder de deling. Zie [De scores schalen](/course/open/attn-scale?lang=nl).",
            "speed": "De deling maakt niets sneller; ze houdt de spreiding van de scores rond 1. Zie [De scores schalen](/course/open/attn-scale?lang=nl).",
            "right": "Vandaag verander je de scores nog één keer, vóór dezelfde softmax: een deel ervan wordt min oneindig.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def quick_check_next_token():
    quick_next_token = L.choice(
        "quick-next-token",
        question="Wat geeft een language model terug voor de tekst tot nu toe?",
        options={
            "right": "Een kans voor elk mogelijk volgend token",
            "text": "De rest van de tekst, in één keer",
            "one": "Het ene token dat volgt, zonder kansen",
        },
        correct="right",
        why={
            "text": "Het voorspelt één token tegelijk; verder schrijven betekent opnieuw vragen met het nieuwe token erbij. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "one": "Het geeft een kans voor elk mogelijk volgend token; er een kiezen is een aparte stap. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "En om het te trainen is elke plek van een tekst een voorbeeld: de tokens tot nu toe, en het token dat volgt.",
        },
    )
    quick_next_token
    return (quick_next_token,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een GPT leert van elke plek van een tekst tegelijk: op positie $i$ voorspelt het token $i + 1$. Met de self-attention van de vorige lessen neemt positie $i$ van elke positie, ook van $i + 1$, precies het token dat het model moet voorspellen. Trainen is dan te makkelijk: het model leert het antwoord over te nemen van de volgende positie. En als het schrijft, is er nog geen volgende positie. Elke positie mag alleen zichzelf gebruiken en wat ervoor kwam.
    """)
    return


@app.cell(hide_code=True)
def predict_zeros():
    guess_zeros = L.guess(
        "zeros",
        question="Een tekst van 4 tokens gaat door attention waarin elke positie alleen mag nemen van zichzelf en de posities ervoor. Hoeveel van de 4 × 4 = 16 attention weights zijn 0?",
        options={
            "four": "4: één verboden token voor elke positie",
            "eight": "8: de helft ervan",
            "right": "6: alles boven de diagonaal",
        },
        reveal="**6.** Positie 1 mag niet nemen van positie 2, 3 en 4, positie 2 niet van 3 en 4, positie 3 niet van 4: 3 + 2 + 1 = 6, alles boven de diagonaal. Positie 4 neemt van alle vier. Bij $n$ tokens zijn $n(n - 1)/2$ weights 0: bij 1.024 tokens bijna de helft.",
    )
    guess_zeros
    return (guess_zeros,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **causal mask** verbergt de toekomst van elke positie. Vóór de softmax wordt elke score van een latere positie $-\infty$: $e^{-\infty} = 0$, dus zijn weight is precies 0, en de softmax verdeelt de hele weight van de rij over de posities die overblijven. Met $M_{ij} = 0$ voor $j \le i$ en $-\infty$ voor $j > i$:
        """),
        L.formula(r"[[hidden:\mathbf{H}]] = \mathrm{softmax}\!\left(\frac{[[hidden:Q]] [[hidden:K]]^\top}{\sqrt{[[hyper:d_k]]}} + M\right) [[hidden:V]]"),
        mo.md(r"""
        De eerste positie ziet alleen zichzelf, en de laatste ziet alles. Een transformer met deze mask is een *decoder*, zoals GPT. In PyTorch geeft `torch.tril(torch.ones(n, n))` een matrix van n bij n met enen op en onder de diagonaal, en nullen erboven; `== 0` maakt van de nullen `True`: de toekomst. `S.masked_fill(future, float("-inf"))` geeft een kopie van `S` met $-\infty$ overal waar `future` `True` is; `float("-inf")` is min oneindig als Python-getal.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **causal mask** | | zet de scores van latere posities vóór de softmax op $-\infty$, zodat elke positie alleen neemt van zichzelf en de posities ervoor | `S.masked_fill(future, float("-inf"))` |
        | $M$ | M | de mask: 0 waar een positie mag kijken, $-\infty$ waar dat niet mag | |
        | `torch.tril` | torch tril | de onderste driehoek van een matrix: de getallen op en onder de diagonaal, nullen erboven | `torch.tril(A)` |
        | `masked_fill` | masked fill | een kopie van een tensor met één waarde overal waar een mask van `True` en `False` `True` is | `S.masked_fill(future, float("-inf"))` |
        """),
        L.watch_out("Maskeer de scores vóór de softmax, niet de weights erna: weights die je achteraf op 0 zet, laten rijen achter die niet meer optellen tot 1. De mask verbergt een woord ook niet voor zichzelf: de diagonaal blijft, anders zou de eerste positie helemaal niets zien."),
        mo.md(r"""
        De figuur: de attention weights van "the cat sat down", met verzonnen embeddings van 2 getallen als queries, keys en values, en de causal mask. Boven de diagonaal is elke weight 0. "the" neemt alles van zichzelf; "down" neemt van alle vier woorden, met dezelfde weights als zonder mask, want er komt niets meer na.
        """),
        C.mask_picture({"title": "Causale attention van \"the cat sat down\"", "rows": "woord", "cols": "neemt van", "decimal": ",",
                        "alt": "Een matrix van 4 bij 4 attention weights met nullen boven de diagonaal. the neemt 1 van zichzelf; cat neemt 0,33 van the en 0,67 van zichzelf; sat neemt 0,25, 0,25 en 0,50; down neemt 0,31, 0,15, 0,22 en 0,31."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Causale attention, met één query, key en value per rij, voor $n$ posities:

    1. De scores: elke query met elke key, gedeeld door $\sqrt{d_k}$.
    2. De toekomst: `True` boven de diagonaal van een matrix van n bij n, waar `torch.tril` nullen overlaat in een matrix van enen.
    3. De scores van de toekomst op $-\infty$, met `masked_fill`.
    4. De softmax van elke rij, en de weights keer de values.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "the cat sat down", met de verzonnen embeddings van de figuur. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_mask():
    import math
    import torch

    _X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
    _n, _d_k = _X.shape
    # 1. de scores
    _S = _X @ _X.T / math.sqrt(_d_k)
    # 2. de toekomst
    _future = torch.tril(torch.ones(_n, _n)) == 0
    print(_future)
    # 3. de scores van de toekomst op min oneindig
    _S = _S  # maak af: _S.masked_fill(_future, float("-inf"))
    # 4. de weights
    print(torch.softmax(_S, dim=1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_future` is `True` boven de diagonaal: 6 plekken. Zoals de cel nu is, heeft elke rij 4 weights boven 0: "the" neemt 0,313 van "sat", dat erna komt. Afgemaakt zijn de scores boven de diagonaal $-\infty$, en de weights daar 0: "the" neemt 1 van zichzelf, "cat" 0,330 van "the" en 0,670 van zichzelf, "sat" 0,248, 0,248 en 0,503. De laatste rij, "down", is met en zonder mask hetzelfde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_infinity():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom min oneindig
    """)
    return


@app.cell(hide_code=True)
def choice_infinity_cell():
    choice_infinity = L.choice(
        "minus-infinity",
        question="Waarom zet de mask de scores van de toekomst op $-\\infty$, en niet op 0?",
        options={
            "zero-ok": "0 werkt even goed: een score van 0 betekent geen match",
            "right": "Een score van 0 krijgt in de softmax nog $e^0 = 1$, dus de toekomst zou nog meetellen",
            "faster": "Met $-\\infty$ is de softmax sneller uit te rekenen",
        },
        correct="right",
        why={
            "zero-ok": "Een score van 0 is een gewone score: naast scores rond 0 krijgt hij ongeveer evenveel weight als de andere.",
            "faster": "De softmax kost evenveel; wat verandert, is dat $e^{-\\infty} = 0$ die weights precies 0 maakt.",
            "right": "Alleen $e^{-\\infty} = 0$ geeft een weight van precies 0, wat de andere scores ook zijn. Een heel negatief getal, zoals $-10^9$, werkt bijna even goed.",
        },
    )
    choice_infinity
    return (choice_infinity,)


@app.cell(hide_code=True)
def text_exercise_future():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De toekomst van elke positie

        Maak `future_mask(n)` af: de functie geeft een tensor van n bij n met `True` waar kolom `j` na rij `i` komt, en `False` op en onder de diagonaal. Vervang de `...` volgens stap 2 van het recept. De check roept `future_mask(3)` aan, wat `[[False, True, True], [False, False, True], [False, False, False]]` moet geven, en probeert andere maten.
        """),
        L.task("Laat `future_mask` de toekomst van elke positie markeren met True."),
    ])
    return


@app.cell(hide_code=True)
def exercise_future():
    box_future = L.code_box("future-mask", lines=6)
    box_future
    return (box_future,)


@app.cell(hide_code=True)
def check_future(box_future):
    result_future = L.check("future-mask", box_future.value)
    result_future
    return (result_future,)


@app.cell(hide_code=True)
def hints_future():
    L.hints(
        "future-mask",
        [
            "`torch.tril` houdt de enen op en onder de diagonaal; erboven staan de nullen, en de nullen zijn de toekomst.",
            "Stap 2 van het uitgewerkte voorbeeld, met `n` in plaats van `_n`.",
        ],
        why="`torch.tril(torch.ones(n, n))` heeft enen waar een positie mag kijken, zichzelf en ervoor; `== 0` maakt van de rest, boven de diagonaal, `True`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_causal():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Causale attention

        Maak `causal_attention(Q, K, V)` af: één query, key en value per rij; de functie geeft de nieuwe vectoren van scaled dot-product attention met de causal mask. De eerste regels maken tensors van de lijsten en rekenen de geschaalde scores uit; stap 4 staat er al. Vervang de `...` volgens stap 2 en 3 van het recept. De check roept `causal_attention` aan met 3 posities waarvan de queries en keys `[[2, 0], [0, 2], [1, 1]]` zijn en de values `[[1, 0], [0, 1], [0.5, 0.5]]`, wat `[[1, 0], [0.0558, 0.9442], [0.5, 0.5]]` moet geven, en probeert andere maten.
        """),
        L.task("Laat `causal_attention` de toekomst van elke positie verbergen vóór de softmax."),
    ])
    return


@app.cell(hide_code=True)
def exercise_causal():
    box_causal = L.code_box("causal-attention", lines=11)
    box_causal
    return (box_causal,)


@app.cell(hide_code=True)
def check_causal(box_causal):
    result_causal = L.check("causal-attention", box_causal.value)
    result_causal
    return (result_causal,)


@app.cell(hide_code=True)
def hints_causal():
    L.hints(
        "causal-attention",
        [
            "Stap 2 en 3 van het afgemaakte uitgewerkte voorbeeld in één regel, met `S` en `n`.",
            "`masked_fill` krijgt de mask van de toekomst en de waarde die het daar neerzet: min oneindig, `float(\"-inf\")`.",
        ],
        why="De eerste positie ziet alleen zichzelf, dus houdt hij zijn eigen value, `[1, 0]`. De tweede neemt een beetje van de eerste: zijn score met zichzelf is 2,83, met de eerste 0, dus gaat 0,944 van zijn weight naar zichzelf.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De scaled dot-product attention uit [De scores schalen](/course/open/attn-scale?lang=nl), met de toekomst verborgen, zodat een model de next-token prediction uit [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) van elke plek van een tekst tegelijk kan leren, zonder de antwoorden te zien. In de volgende les, *Multi-head attention*, draaien meerdere kleinere attentions naast elkaar, die elk vrij zijn om naar iets anders te zoeken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_infinity, result_future, result_causal):
    L.wrap_up(choice_infinity, result_future, result_causal)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
