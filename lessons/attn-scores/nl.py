# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-scores. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-scores", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl) en [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_new_vector():
    quick_new_vector = L.choice(
        "quick-new-vector",
        question="Hoe bepaalt een woord bij attention hoeveel het van een ander woord neemt?",
        options={
            "right": "Uit het inproduct van hun embeddings, via een softmax",
            "best": "Uit hoe dicht de twee woorden bij elkaar staan in de zin",
            "sum": "Uit een weight die het model voor dat paar woorden geleerd heeft",
        },
        correct="right",
        why={
            "best": "Attention kijkt naar de embeddings, niet naar de plekken: een woord aan de andere kant van de zin kan net zo zwaar meetellen. Zie [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl).",
            "sum": "De attention weights worden niet geleerd: de softmax rekent ze voor elke zin opnieuw uit, uit de inproducten. Zie [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl).",
            "right": "Vandaag reken je die weights uit voor alle woorden van een zin tegelijk, als één matrix.",
        },
    )
    quick_new_vector
    return (quick_new_vector,)


@app.cell(hide_code=True)
def quick_check_shift():
    quick_shift = L.choice(
        "quick-shift",
        question="Voordat je e tot de macht van een lijst getallen neemt, trek je van elk getal het grootste getal af. Wat verandert dat aan de softmax van de lijst?",
        options={
            "smaller": "De kansen worden kleiner: alle getallen gingen omlaag",
            "largest": "Het grootste getal krijgt kans 0: het werd 0",
            "right": "Niets: alleen de overflow van exp is weg",
        },
        correct="right",
        why={
            "smaller": "Elke $e^{z}$ wordt gedeeld door dezelfde $e^{m}$, boven en onder de streep, dus de breuken blijven gelijk. Zie [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).",
            "largest": "$e^0$ is 1, niet 0: het grootste getal krijgt het grootste deel, net als eerst. Zie [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).",
            "right": "$e^{z_j - m} / \\sum_k e^{z_k - m}$ is $e^{z_j} / \\sum_k e^{z_k}$, omdat $e^{-m}$ boven en onder de streep tegen elkaar wegvalt, en er blijft geen $e$ tot een macht boven 0 over die kan overlopen.",
        },
    )
    quick_shift
    return (quick_shift,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl) rekende je de nieuwe vector van één woord uit. Een language model doet dat voor elk token van zijn tekst tegelijk, en doet het opnieuw in elke layer: bij de kleinste GPT-2 met tot 1.024 tokens, in elk van zijn 12 layers. Eén woord tegelijk in een Python-loop is veel te traag; met matrices geschreven zijn het twee matrixvermenigvuldigingen en een softmax. En echte embeddings bevatten honderden getallen, dus hun inproducten kunnen oplopen tot honderden of duizenden; vanaf ongeveer 710 loopt $e$ tot de macht van een score over, en geeft de softmax nan.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="Een tekst van 1.024 tokens gaat door één attention, zoals in deze les: elk token neemt een gewogen gemiddelde van alle tokens. Hoeveel attention weights zijn daarvoor nodig?",
        options={
            "per-token": "1.024: één weight voor elk token",
            "right": "1.048.576: één voor elk paar tokens",
            "neighbors": "3.072: één voor elk token en zijn twee buren",
        },
        reveal="**1.048.576**, dat is 1.024 × 1.024. Elk token heeft een weight voor elk token, zichzelf meegerekend, dus de weights vullen een matrix met een rij en een kolom per token. Twee keer zoveel tekst geeft vier keer zoveel weights: een van de redenen dat lange teksten een transformer traag maken en veel geheugen kosten.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zet de embeddings van de $n$ woorden van een zin in een matrix $\mathbf{X}$, één rij per woord, elk $d$ getallen: shape $(n, d)$. Dan bevat $\mathbf{X}\mathbf{X}^\top$ het inproduct van elk woord met elk woord, de **attention scores**: getal $(i, j)$ is $\mathbf{x}_i \cdot \mathbf{x}_j$, en de shape is $(n, n)$. De softmax van elke rij geeft de **attention weights** $\mathbf{A}$: rij $i$ zegt hoeveel woord $i$ van elk woord neemt, en telt op tot 1. Nog één matrixvermenigvuldiging geeft de nieuwe vectoren van alle woorden, $\mathbf{H}$, met shape $(n, d)$:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{S}]] &= [[hidden:\mathbf{X}]] [[hidden:\mathbf{X}]]^\top \\ [[hidden:\mathbf{A}]] &= \mathrm{softmax}([[hidden:\mathbf{S}]]) \text{, per rij} \\ [[hidden:\mathbf{H}]] &= [[hidden:\mathbf{A}]] [[hidden:\mathbf{X}]] \end{aligned}"),
        mo.md(r"""
        Rij $i$ van $\mathbf{A}\mathbf{X}$ is $\sum_j A_{ij} \mathbf{x}_j$: het gewogen gemiddelde uit [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl), voor woord $i$. Trek vóór de softmax van elke rij de grootste score van die rij af, zoals bij de log-sum-exp in [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl): de weights blijven gelijk, en geen $e$ tot de macht van een score kan nog overlopen. In NumPy geeft `S.max(axis=1, keepdims=True)` het grootste getal van elke rij als kolom, zodat elke rij zijn eigen grootste getal aftrekt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **attention score** | | hoe goed twee woorden bij elkaar passen: het inproduct van hun embeddings | `X @ X.T` |
        | **attention weight** | | hoeveel het ene woord van het andere neemt: de softmax van een rij scores, dus elke rij is positief en telt op tot 1 | `A[i, j]` |
        | $\mathbf{S}$ | hoofdletter S | de attention scores, één rij en één kolom per woord: shape $(n, n)$ | `S` |
        | $\mathbf{A}$ | hoofdletter A | de attention weights, één rij en één kolom per woord: shape $(n, n)$ | `A` |
        | $\mathbf{H}$ | hoofdletter H | de nieuwe vectoren, één rij per woord: shape $(n, d)$ | `A @ X` |
        """),
        L.watch_out("$\\mathbf{S}$ is symmetrisch, want $\\mathbf{x}_i \\cdot \\mathbf{x}_j = \\mathbf{x}_j \\cdot \\mathbf{x}_i$, maar $\\mathbf{A}$ niet: elke rij wordt gedeeld door zijn eigen som. En de softmax gaat langs elke rij, `axis=1`: over de hele matrix, of langs de kolommen, tellen de weights van een woord niet meer op tot 1."),
        mo.md(r"""
        De figuur: de attention weights van "the river bank", met de verzonnen embeddings uit [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl), waarvan de twee getallen zeggen hoeveel een woord over geld gaat en hoeveel over water. Elke rij telt op tot 1. "river" neemt 0,84 van zichzelf, "the" neemt 0,38 van "river", maar "river" neemt maar 0,04 van "the".
        """),
        C.weights_picture({"title": "Attention weights van \"the river bank\"", "rows": "woord", "cols": "neemt van", "decimal": ",",
                           "alt": "Een matrix van 3 bij 3 attention weights. the neemt 0,23 van the, 0,38 van river en 0,38 van bank; river neemt 0,04, 0,84 en 0,11; bank neemt 0,16, 0,42 en 0,42."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Attention voor alle woorden tegelijk, met één embedding per rij van `X`:

    1. De scores: `X @ X.T`, het inproduct van elk woord met elk woord.
    2. Trek van elke rij zijn grootste score af: `S.max(axis=1, keepdims=True)`.
    3. De weights: e tot de macht van elke score, gedeeld door de som van zijn rij.
    4. De nieuwe vectoren: de weights keer `X`, één rij per woord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "the river bank" met de verzonnen embeddings, elk 20 keer zo groot: net als bij embeddings met veel getallen lopen de scores op tot in de honderden. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_scores():
    import numpy as np

    _X = 20 * np.array([[0.5, 0.5], [0.0, 2.0], [1.0, 1.0]])
    # 1. de scores van elk woord met elk woord
    _S = _X @ _X.T
    print(_S)
    # 2. de grootste score van elke rij eraf
    _S = _S  # maak af: _S - _S.max(axis=1, keepdims=True)
    # 3. de weights: de softmax van elke rij
    _A = np.exp(_S) / np.sum(np.exp(_S), axis=1, keepdims=True)
    print(np.round(_A, 3))
    # 4. de nieuwe vectoren
    _H = _A @ _X
    print(np.round(_H, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De scores lopen op tot 1.600, voor "river" met zichzelf. Zoals de cel nu is, lopen $e^{1600}$ en $e^{800}$ over tot `inf`, NumPy waarschuwt daarvoor, en `inf / inf` is `nan`: de weights van "river" en "bank" worden `nan` waar een score overliep, en hun nieuwe vectoren helemaal. Alleen "the", met als grootste score 400, werkt nog. Afgemaakt trekt elke rij eerst zijn grootste score af: de weights worden `[0. 0.5 0.5]`, `[0. 1. 0.]` en `[0. 0.5 0.5]`, en de nieuwe vectoren `[10. 30.]`, `[0. 40.]` en `[10. 30.]`. Zulke grote scores maken de softmax heel scherp: elke rij geeft bijna al zijn weight aan wat het best past. Twee lessen verder, *De scores schalen*, maakt ze expres kleiner.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_symmetric():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Symmetrisch of niet
    """)
    return


@app.cell(hide_code=True)
def choice_symmetric_cell():
    choice_symmetric = L.choice(
        "symmetric",
        question="De scores zijn symmetrisch: $S_{ij} = S_{ji}$. Zijn de attention weights ook symmetrisch, $A_{ij} = A_{ji}$?",
        options={
            "yes": "Ja: de softmax houdt elk paar scores gelijk op",
            "same-word": "Alleen op de diagonaal, waar een woord zichzelf tegenkomt",
            "right": "Nee: elke rij wordt gedeeld door zijn eigen som",
        },
        correct="right",
        why={
            "yes": "$A_{ij}$ is $e^{S_{ij}}$ gedeeld door de som van rij $i$, en $A_{ji}$ hetzelfde getal gedeeld door de som van rij $j$. In de figuur neemt \"the\" 0,38 van \"river\", en \"river\" 0,04 van \"the\".",
            "same-word": "Op de diagonaal vergelijk je een weight met zichzelf: $A_{ii} = A_{ii}$ zegt niets. Buiten de diagonaal hebben rij $i$ en rij $j$ meestal verschillende sommen, dus verschillen $A_{ij}$ en $A_{ji}$: in de figuur 0,38 en 0,04.",
            "right": "$A_{ij}$ en $A_{ji}$ delen de score $S_{ij}$, maar rij $i$ en rij $j$ hebben hun eigen som. Een woord dat elders sterk past, neemt van elk woord minder.",
        },
    )
    choice_symmetric
    return (choice_symmetric,)


@app.cell(hide_code=True)
def text_exercise_softmax_rows():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een softmax voor elke rij

        Maak `softmax_rows(S)` af: de functie geeft de softmax van elke rij van de matrix `S`, zonder overflow, ook bij scores van duizend of meer. Vervang de twee `...` volgens stap 2 en 3 van het recept. De check roept `softmax_rows([[1, 2, 3], [1000, 1000, 1001]])` aan, wat `[[0.09, 0.2447, 0.6652], [0.2119, 0.2119, 0.5761]]` moet geven, en probeert andere matrices, ook niet-vierkante.
        """),
        L.task("Laat `softmax_rows` de softmax van elke rij van een matrix geven, zonder overflow."),
    ])
    return


@app.cell(hide_code=True)
def exercise_softmax_rows():
    box_softmax_rows = L.code_box("softmax-rows", lines=9)
    box_softmax_rows
    return (box_softmax_rows,)


@app.cell(hide_code=True)
def check_softmax_rows(box_softmax_rows):
    result_softmax_rows = L.check("softmax-rows", box_softmax_rows.value)
    result_softmax_rows
    return (result_softmax_rows,)


@app.cell(hide_code=True)
def hints_softmax_rows():
    L.hints(
        "softmax-rows",
        [
            "Stap 2 en 3 van het afgemaakte uitgewerkte voorbeeld, met `S` in plaats van `_S`.",
            "Het grootste getal van elke rij, als kolom: `S.max(axis=1, keepdims=True)`. Zonder `keepdims` legt NumPy de platte uitkomst langs de kolommen in plaats van langs de rijen.",
        ],
        why="Als je de grootste score van de rij zelf aftrekt, blijft elke breuk gelijk, en is $e^0 = 1$ de grootste exp, dus niets loopt over. De tweede rij van het voorbeeld, 1.000, 1.000 en 1.001, wordt −1, −1 en 0: dezelfde weights als voor 0, 0 en 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_attend_all():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Alle woorden tegelijk

        Maak `attend_all(X)` af: `X` bevat één embedding per rij; de functie geeft de nieuwe vectoren van alle woorden, één per rij, volgens het recept. Stap 2 en 3 staan er al. Vervang de twee `...` volgens stap 1 en 4. De check roept `attend_all([[1, 0], [0, 1], [1, 1]])` aan, de zin van de oefening in [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl), wat `[[0.8446, 0.5777], [0.5777, 0.8446], [0.7881, 0.7881]]` moet geven, en probeert andere zinnen, ook van één woord en met 3 getallen per woord.
        """),
        L.task("Laat `attend_all` de nieuwe vectoren van alle woorden tegelijk geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attend_all():
    box_attend_all = L.code_box("self-attention", lines=11)
    box_attend_all
    return (box_attend_all,)


@app.cell(hide_code=True)
def check_attend_all(box_attend_all):
    result_attend_all = L.check("self-attention", box_attend_all.value)
    result_attend_all
    return (result_attend_all,)


@app.cell(hide_code=True)
def hints_attend_all():
    L.hints(
        "self-attention",
        [
            "Stap 1 geeft een matrix met een rij en een kolom voor elk woord, dus met shape $(n, n)$: welke volgorde van `X` en `X.T` geeft dat?",
            "Stap 4: rij $i$ van de uitkomst is het gewogen gemiddelde met de weights van rij $i$ van `A`. Welke matrixvermenigvuldiging van `A` en `X` neemt voor elke rij van `A` de rijen van `X`?",
        ],
        why="`X @ X.T` heeft shape $(n, d) \\times (d, n) = (n, n)$; `A @ X` heeft shape $(n, n) \\times (n, d) = (n, d)$: één nieuwe vector per woord, zo lang als een embedding. Rij 0 van het voorbeeld is wat `attend` voor woord 0 gaf.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het gewogen gemiddelde uit [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl), voor alle woorden tegelijk met de matrixvermenigvuldiging uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl), en zonder overflow dankzij de truc uit [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl). Tot nu toe vergelijkt een woord zijn embedding met de embeddings van de andere, en geeft het zijn embedding door zoals hij is. In de volgende les, *Queries, keys en values*, krijgt elk woord in plaats daarvan drie geleerde vectoren: één om mee te zoeken, één om gevonden te worden, en één om door te geven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_symmetric, result_softmax_rows, result_attend_all):
    L.wrap_up(choice_symmetric, result_softmax_rows, result_attend_all)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
