# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-matrix-rank. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-matrix-rank", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) en [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Welke shape heeft `B @ C`, met `B` van shape (5, 2) en `C` van shape (2, 5)?",
        options={
            "inner": "(2, 2)",
            "error": "Geen: het stopt met een fout",
            "right": "(5, 5)",
        },
        correct="right",
        why={
            "inner": "De binnenste getallen, 2 en 2, moeten gelijk zijn en vallen dan weg; de buitenste blijven over. Zie [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "error": "De binnenste getallen zijn allebei 2, dus de shapes passen. Zie [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "right": "(5, 2) @ (2, 5): de binnenste tweeën passen, en de buitenste 5 en 5 blijven over.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_column():
    quick_column = L.choice(
        "quick-column",
        question="`c` is een array van shape (3, 1), een kolom. Wat is de shape van `c.T`?",
        options={
            "right": "(1, 3): een rij",
            "same": "(3, 1): `.T` verandert hier niets",
            "flat": "(3,): de haakjes vallen weg",
        },
        correct="right",
        why={
            "same": "`.T` verandert alleen niets bij een array met één getal in zijn shape, zoals (3,). Deze heeft er twee. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "flat": "Transponeren wisselt de twee getallen van de shape om; het laat er nooit een vallen. Zie [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "right": "Transponeren wisselt rijen en kolommen: (3, 1) wordt (1, 3).",
        },
    )
    quick_column
    return (quick_column,)


@app.cell(hide_code=True)
def text_why():
    mo.md(f"""
    ## Waarom

    Een groot language model heeft weight-matrices van 4.096 bij 4.096: elk {C.grouped(4096 * 4096, L.lang)} getallen, en tientallen daarvan. Om zo'n model een nieuwe taak te leren, hoef je niet al die getallen vrij te veranderen: een verandering waarvan alle rijen combinaties zijn van een paar rijen, is vaak genoeg. Zo'n verandering kun je bewaren als twee dunne matrices, met een klein deel van de getallen. Later in de course pas je een model op die manier aan, met LoRA.

    **Een voorbeeld uit het dagelijks leven.** Een verfwinkel verkoopt honderden kleuren, maar mengt ze allemaal uit een paar basispigmenten. Schrijf de pigmenten op en één recept per kleur, en je hebt elke kleur, met veel minder om te bewaren.
    """)
    return


@app.cell(hide_code=True)
def predict_rank():
    guess_rank = L.guess(
        "rank-guess",
        question="Een matrix heeft drie rijen: `[1, 2, 3]`, `[2, 4, 6]` en `[1, 0, 1]`. Hoeveel echt verschillende richtingen denk je dat deze rijen hebben?",
        options={
            "three": "3: er zijn drie rijen",
            "right": "2: één rij voegt niets nieuws toe",
            "one": "1: ze beginnen allemaal met een klein getal",
        },
        reveal="**2.** De tweede rij is precies twee keer de eerste: hij wijst dezelfde kant op, alleen langer, dus hij voegt geen nieuwe richting toe. De derde rij kun je niet uit de eerste maken door te schalen, dus die voegt er wel een toe. Twee onafhankelijke rijen: de rang van deze matrix is 2.",
    )
    guess_rank
    return (guess_rank,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een rij is **onafhankelijk** van andere rijen als je hem niet kunt maken door die rijen te schalen en op te tellen. De **rang** van een matrix is zijn aantal onafhankelijke rijen: hoeveel richtingen hij echt heeft. Dat is altijd ook het aantal onafhankelijke kolommen, dus hooguit het kleinste van de twee getallen in zijn shape, en `A.T` heeft dezelfde rang als `A`. In code geeft `np.linalg.matrix_rank(A)` hem.

        Vermenigvuldig een hoge, dunne matrix $\mathbf{B}$ van shape $(d, r)$ met een brede, dunne matrix $\mathbf{C}$ van shape $(r, d)$, en je krijgt een vierkante matrix $\mathbf{A} = \mathbf{B}\mathbf{C}$ van shape $(d, d)$. Zijn rang is hooguit $r$. Je ziet waarom als je het product schrijft als een som van $r$ stukken: kolom $i$ van $\mathbf{B}$ keer rij $i$ van $\mathbf{C}$. Bij LoRA zijn $\mathbf{B}$ en $\mathbf{C}$ wat het model leert, dus hier zijn ze oranje, en $\mathbf{A}$ en zijn stukken ook, want die maken ze samen.
        """),
        L.formula(r"[[param:\mathbf{A}]] = [[param:\mathbf{B}]][[param:\mathbf{C}]] = [[param:\mathbf{b}_1 \mathbf{c}_1]] + [[param:\mathbf{b}_2 \mathbf{c}_2]] + \dots + [[param:\mathbf{b}_r \mathbf{c}_r]]"),
        mo.md(f"""
        Elk stuk $\\mathbf{{b}}_i \\mathbf{{c}}_i$ is een kolom van shape $(d, 1)$ keer een rij van shape $(1, d)$: een volle $(d, d)$-matrix waarvan de rijen allemaal veelvouden zijn van dezelfde rij $\\mathbf{{c}}_i$, dus met rang 1, tenzij het helemaal uit nullen bestaat. Elk stuk voegt hooguit één richting toe, dus $r$ stukken geven er hooguit $r$.

        Een matrix met een rang ver onder zijn grootte is **low-rank**. Bewaard als $\\mathbf{{B}}$ en $\\mathbf{{C}}$ heeft hij $2 \\cdot d \\cdot r$ getallen nodig in plaats van $d \\cdot d$. Voor $d$ = 4.096 en $r$ = 8 zijn dat er {C.grouped(2 * 4096 * 8, L.lang)} in plaats van {C.grouped(4096 * 4096, L.lang)}: minder dan een half procent.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **onafhankelijk** | | lineair onafhankelijk: een rij die je niet kunt maken door andere rijen te schalen en op te tellen | |
        | **rang** | | het aantal onafhankelijke rijen van een matrix, dat ook zijn aantal onafhankelijke kolommen is | `np.linalg.matrix_rank(A)` |
        | **low-rank** | | met een rang ver onder zijn grootte; zo'n matrix kun je bewaren als twee dunne matrices | `B @ C` |
        | $\\mathbf{{b}}_i \\mathbf{{c}}_i$ | b i c i | kolom $i$ van $\\mathbf{{B}}$ keer rij $i$ van $\\mathbf{{C}}$: een matrix met rang 1, of 0 als hij alleen nullen bevat | `B[:, 0:1] @ C[0:1, :]` |
        """),
        L.watch_out("Rang gaat over richtingen, niet over grootte of nullen: een matrix vol grote getallen kan rang 1 hebben, en de matrix met de rijen `[1, 0]` en `[0, 1]`, voor de helft nullen, heeft rang 2. En `matrix_rank` telt richtingen op afronding na: een rij die op een piepkleine float-fout na een combinatie van andere is, telt toch als afhankelijk."),
        mo.md(r"""
        De figuur: een hoge matrix $\mathbf{B}$ van shape (8, 2) keer een brede matrix $\mathbf{C}$ van shape (2, 8) geeft een vierkante matrix $\mathbf{A}$ van shape (8, 8), elk op schaal getekend.
        """),
        mo.Html(C.thin_picture({"title": "Twee dunne matrices maken één vierkante", "numbers": "{n} getallen",
                                "note": "A heeft 64 getallen, maar rang hooguit {r}:\nB en C bevatten er maar 32.",
                                "alt": "Drie vakken, op schaal getekend: B, hoog en dun, (8, 2), 16 getallen; keer C, breed en dun, (2, 8), 16 getallen; geeft A, vierkant, (8, 8), 64 getallen. Daaronder: A heeft 64 getallen, maar rang hooguit 2: B en C bevatten er 32."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een low-rank matrix te maken en te controleren:

    1. Kies de rang $r$, en maak een hoge matrix $\mathbf{B}$ van shape $(d, r)$ en een brede matrix $\mathbf{C}$ van shape $(r, d)$.
    2. Vermenigvuldig ze: $\mathbf{A} = \mathbf{B}\mathbf{C}$, van shape $(d, d)$.
    3. Dezelfde matrix is de som van $r$ stukken: kolom $i$ van $\mathbf{B}$ keer rij $i$ van $\mathbf{C}$.
    4. Controleer de rang met `np.linalg.matrix_rank`, en tel de getallen: $2 \cdot d \cdot r$ voor $\mathbf{B}$ en $\mathbf{C}$, tegen $d \cdot d$ voor $\mathbf{A}$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een matrix van shape (6, 6) en rang 2. In `B[:, 0:1]` zorgt de `0:1` ervoor dat de kolom een kolom blijft, van shape (6, 1); in `C[0:1, :]` blijft de rij zo een rij, van shape (1, 6). Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 en 4 zijn nog niet af: voer de cel eerst zo uit, maak ze dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_rank():
    import numpy as np

    # 1. r = 2: B is (6, 2), C is (2, 6)
    _B = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [0, 3], [1, 2]])
    _C = np.array([[1, 2, 0, 1, 0, 3], [0, 1, 1, 0, 2, 1]])
    # 2. het product: een matrix van shape (6, 6)
    _A = _B @ _C
    print(_A)
    # 3. dezelfde matrix als stukken met rang 1: het eerste stuk staat er; tel het tweede erbij op, met 1:2
    _pieces = _B[:, 0:1] @ _C[0:1, :]
    print(_pieces - _A)
    # 4. de rang, en de getallen: A heeft er 6 * 6; zet erachter hoeveel B en C er samen hebben
    print(np.linalg.matrix_rank(_A), 6 * 6)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\mathbf{A}$ heeft 6 rijen, maar rang 2: de derde rij, `[1, 3, 1, 1, 2, 4]`, is de eerste plus de tweede, en de andere zijn ook combinaties van de eerste twee. Met alleen het eerste stuk is het verschil nog niet 0: het is min het tweede stuk, het deel dat nog ontbreekt. Met het tweede stuk erbij is het overal 0: de twee stukken tellen precies op tot $\mathbf{A}$. $\mathbf{B}$ en $\mathbf{C}$ bevatten 24 getallen tegen 36; hoe groter de matrix, hoe groter de besparing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De rang van een matrix
    """)
    return


@app.cell(hide_code=True)
def choice_rank_cell():
    choice_rank = L.choice(
        "rank-3",
        question="Wat is de rang van de matrix met de rijen `[1, 0, 1]`, `[0, 1, 1]` en `[1, 1, 2]`?",
        options={
            "three": "3",
            "one": "1",
            "right": "2",
        },
        correct="right",
        why={
            "three": "De derde rij is de eerste plus de tweede, dus die voegt geen nieuwe richting toe.",
            "one": "De eerste twee rijen wijzen verschillende kanten op: geen van beide is een veelvoud van de andere.",
            "right": "De eerste twee rijen zijn onafhankelijk, en de derde is hun som: twee richtingen.",
        },
    )
    choice_rank
    return (choice_rank,)


@app.cell(hide_code=True)
def text_exercise_numbers():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel getallen?

        Een weight-matrix hoeft niet vierkant te zijn: hij heeft de shape $(d_\text{out}, d_\text{in})$. `thin_numbers(d_out, d_in, r)` moet geven hoeveel getallen $\mathbf{B}$, van shape $(d_\text{out}, r)$, en $\mathbf{C}$, van shape $(r, d_\text{in})$, samen bevatten. Vervang de `...`. De check roept `thin_numbers(5, 4, 2)` aan, en dat moet `18` geven; hij probeert ook andere shapes.
        """),
        L.task("Laat `thin_numbers` de getallen van `B` en `C` samen tellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_numbers():
    box_numbers = L.code_box("thin-numbers", lines=4)
    box_numbers
    return (box_numbers,)


@app.cell(hide_code=True)
def check_numbers(box_numbers):
    result_numbers = L.check("thin-numbers", box_numbers.value)
    result_numbers
    return (result_numbers,)


@app.cell(hide_code=True)
def hints_numbers():
    L.hints(
        "thin-numbers",
        [
            "Stap 4 van het recept, voor een matrix die niet vierkant is.",
            "Een matrix van shape $(n, m)$ bevat $n \\cdot m$ getallen. Tel `B` en `C` elk met hun eigen shape, en tel op.",
        ],
        why="`B` is (5, 2), 10 getallen; `C` is (2, 4), 8 getallen: samen 18, tegen 20 voor de volle (5, 4)-matrix.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pieces():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een matrix uit stukken

        `from_pieces(cols, rows)` krijgt $r$ kolommen van shape $(d, 1)$ en $r$ rijen van shape $(1, d)$, en moet de som van de $r$ stukken geven, kolom $i$ keer rij $i$, zoals in stap 3 van het recept. Schrijf de body van de functie. De check roept hem aan met twee kolommen van 3 getallen en twee rijen van 3 getallen, en probeert ook andere stukken.
        """),
        L.task("Laat `from_pieces` de som van alle stukken geven, elk een kolom keer een rij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pieces():
    box_pieces = L.code_box("from-pieces", lines=9)
    box_pieces
    return (box_pieces,)


@app.cell(hide_code=True)
def check_pieces(box_pieces):
    result_pieces = L.check("from-pieces", box_pieces.value)
    result_pieces
    return (result_pieces,)


@app.cell(hide_code=True)
def hints_pieces():
    L.hints(
        "from-pieces",
        [
            "Stap 3 van het recept: stuk `i` is `cols[i]` keer `rows[i]`. Begin met het eerste stuk, en tel de andere erbij op in een loop over `range(1, len(cols))`.",
            "Een kolom keer een rij is (d, 1) @ (1, d): een (d, d)-matrix. Welke volgorde geeft die shape?",
        ],
        why="Het eerste stuk is `[1, 0, 2]` als kolom keer `[1, 2, 0]` als rij; het tweede is `[0, 1, 1]` keer `[0, 1, 3]`. Opgeteld geven ze een (3, 3)-matrix met rang 2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een matrixproduct uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) kun je lezen als een som van stukken, elk een kolom keer een rij; [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl) wisselt rijen en kolommen om, en laat de rang gelijk. Een matrix uit $r$ zulke stukken heeft rang hooguit $r$, en twee dunne matrices bewaren hem met veel minder getallen. In de volgende les, *Veelgebruikte verdelingen*, laat je matrices even los en beschrijf je kans met formules.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rank, result_numbers, result_pieces):
    L.wrap_up(choice_rank, result_numbers, result_pieces)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
