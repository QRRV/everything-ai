# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les np-broadcasting. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow vectorization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-broadcasting", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) en [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_times():
    quick_times = L.choice(
        "quick-times",
        question="`a = np.array([1, 2])` en `b = np.array([3, 4])`. Wat geeft `a * b`?",
        options={
            "total": "`11`",
            "pairs": "`array([3, 8])`",
            "joined": "`array([1, 2, 3, 4])`",
        },
        correct="pairs",
        why={
            "total": "Dat zou één getal voor alles zijn. Wat doet `*` met twee arrays met dezelfde shape? Zie de tabel met nieuwe woorden in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "pairs": "De getallen op dezelfde plek gaan samen: 1 keer 3, en 2 keer 4.",
            "joined": "`*` plakt arrays niet aan elkaar. Zie de figuur in [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
        },
    )
    quick_times
    return (quick_times,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="`m` heeft shape `(3, 4)`. Wat is de shape van `m.mean(axis=0)`?",
        options={
            "rows": "`(3,)`",
            "one": "Eén getal, zonder shape",
            "cols": "`(4,)`",
        },
        correct="cols",
        why={
            "rows": "`axis=0` gaat langs de rijen omlaag, dus verdwijnen de rijen. Wat blijft er over? Zie het blok *Let op* in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "one": "Dat krijg je zonder axis. Zie het recept in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "cols": "Langs de rijen omlaag verdwijnen de 3 rijen, en blijft er één gemiddelde per kolom over: 4 getallen.",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vaak wil je één getal per kolom van een hele tabel aftrekken, zoals elk cijfer min het gemiddelde van zijn toets, zodat je ziet wie het per toets beter deed dan de anderen. Later in de course telt een model dezelfde getallen op bij elk van de vele voorbeelden die het tegelijk verwerkt. Met wat je in deze les leert, doet NumPy dat in één regel code, zonder de kleine array te kopiëren.

    **Een voorbeeld uit het dagelijks leven.** Een docent die bij elk cijfer van een moeilijke toets één punt optelt, en twee bij elk cijfer van een nog moeilijkere: dezelfde correctie voor elke leerling, maar een andere per toets.
    """)
    return


@app.cell(hide_code=True)
def predict_stretch():
    guess_stretch = L.guess(
        "stretch",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])` heeft shape `(2, 3)`. Wat denk je dat `m + np.array([10, 20, 30])` geeft?",
        options={
            "rows": "`array([[11, 22, 33], [14, 25, 36]])`",
            "error": "Een fout: de twee shapes verschillen",
            "first": "Alleen de eerste rij verandert: `array([[11, 22, 33], [4, 5, 6]])`",
        },
        reveal="**`array([[11, 22, 33], [14, 25, 36]])`.** De shapes verschillen, maar ze passen: NumPy rekt de vector uit over de twee rijen, alsof hij er twee keer staat, en telt getal voor getal op. Dat is broadcasting.",
    )
    guess_stretch
    return (guess_stretch,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Bij twee arrays met verschillende shapes rekt NumPy elke lengte 1, of lengte die ontbreekt, uit tot de lengte van de andere array op die plek, tot ze allebei dezelfde shape hebben, en rekent dan getal voor getal. Dat is broadcasting. Het werkt alleen als de shapes passen, en de regel daarvoor is kort.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **broadcasting** | NumPy rekt elke lengte 1, of lengte die ontbreekt, uit tot de andere lengte, zonder iets te kopiëren, en rekent dan getal voor getal | `m + np.array([10, 20, 30])` |
    | **de regel** | zet de shapes onder elkaar, rechts uitgelijnd; elk paar moet gelijk zijn, of een van de twee moet 1 zijn; een lengte die ontbreekt, telt als 1 | `(2, 3)` en `(3,)` passen |
    | **een vector** | shape `(3,)` tegen `(2, 3)`: hij komt bij elke rij | `m + np.array([10, 20, 30])` |
    | **een kolom** | shape `(2, 1)` tegen `(2, 3)`: één getal per rij, uitgerekt over de kolommen | `m + np.array([[100], [200]])` |
    | **`keepdims=True`** | laat de axis staan die een samenvatting weghaalt, met lengte 1, zodat één getal per rij een kolom blijft | `m.sum(axis=1, keepdims=True)` heeft shape `(2, 1)` |

    Passen de shapes niet, dan stopt NumPy met een fout zoals `operands could not be broadcast together with shapes (3,4) (3,)`. Met operands bedoelt NumPy de twee arrays van de berekening, en daarna volgen hun shapes, zonder spaties geschreven. Maar bij een vierkante matrix, zoals `(3, 3)`, past een vector met shape `(3,)` altijd, ook als je één getal per rij bedoelde: dan krijg je een verkeerde uitkomst, en geen fout. Een kolom gemaakt met `keepdims=True` houdt het goed.
    """)
    return


@app.cell(hide_code=True)
def picture_stretch():
    mo.vstack([
        draw.array_grids(C.picture(["m", "v, uitgerekt", "m + v"], ["uitgerekt tot (2, 3), niet gekopieerd", "elke rij plus v"]),
                         title="Broadcasting: een vector uitgerekt over de rijen"),
        mo.md(r"""
        `v` is de vector `[10 20 30]`: de rij met de dikke rand. Om op `m` te passen, doet NumPy alsof `v` er twee keer staat: de grijze rij is de uitgerekte kopie, die NumPy nooit echt maakt. Dan telt NumPy getal voor getal op, dus krijgt elke rij van `m` er `10`, `20` en `30` bij.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een vector met shape `(2,)` past niet op een matrix met shape `(2, 3)`: rechts uitgelijnd verschillen 2 en 3. Voor één getal per rij heb je een kolom nodig, shape `(2, 1)`: gebruik `keepdims=True`."),
        mo.md(r"""
        ### Recept

        Zo reken je met twee arrays met verschillende shapes:

        1. Zet beide shapes onder elkaar, rechts uitgelijnd.
        2. Vergelijk ze paar voor paar, van rechts. Ze passen als elk paar gelijk is of een 1 heeft; een lengte die ontbreekt, telt als 1.
        3. Reken: NumPy rekt elke 1 uit tot de lengte van de andere shape op die plek.
        4. Eén getal per rij tegen een matrix: maak er een kolom van, met `keepdims=True` als het uit een samenvatting komt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cijfers uit [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl): een rij per leerling, een kolom per toets. Eerst elk cijfer min het gemiddelde van zijn toets, dan elk cijfer min het gemiddelde van zijn leerling. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_centered():
    import numpy as np

    _grades = np.array([[7, 8, 6, 8], [5, 7, 8, 6], [9, 9, 7, 10]])
    _test_means = _grades.mean(axis=0)
    # 1. en 2. de shapes: (3, 4) en (4,); van rechts zijn 4 en 4 gelijk, en de ontbrekende telt als 1
    print(_grades.shape, _test_means.shape)
    # 3. elk cijfer min het gemiddelde van zijn toets: de gemiddelden worden over de rijen uitgerekt
    print(_grades - _test_means)
    # 4. één gemiddelde per leerling, bewaard als kolom, shape (3, 1)
    _student_means = _grades.mean(axis=1, keepdims=True)
    # 1. en 2. (3, 4) en (3, 1): van rechts 4 en 1, en een van de twee is 1; dan 3 en 3, gelijk
    print(_grades.shape, _student_means.shape)
    # 3. elk cijfer min het gemiddelde van zijn leerling: de kolom wordt over de kolommen uitgerekt
    _centered = _grades - _student_means
    print(_centered)
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De shapes `(3, 4)` en `(4,)` passen, dus gaat van elk cijfer het gemiddelde van zijn toets af: `0` betekent net zo goed als het gemiddelde van die toets, en `2` betekent 2 punten beter. Leerling 2, de laatste rij, deed het in drie toetsen beter dan het gemiddelde. De gemiddelden per leerling hebben met `keepdims=True` shape `(3, 1)`: een kolom. Afgetrokken van de cijfers laat die zien hoe elk cijfer zich verhoudt tot het eigen gemiddelde van die leerling. Zonder `keepdims=True` hadden de gemiddelden per leerling shape `(3,)`, en `(3, 4)` min `(3,)` stopt met een fout: rechts uitgelijnd verschillen 4 en 3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke past?
    """)
    return


@app.cell(hide_code=True)
def choice_fits_cell():
    choice_fits = L.choice(
        "which-fits",
        question="`m` heeft shape `(3, 4)`. Welke array kan NumPy bij `m` optellen?",
        options={
            "four": "Een array met shape `(4,)`",
            "three": "Een array met shape `(3,)`",
            "turned": "Een array met shape `(4, 3)`",
        },
        correct="four",
        why={
            "three": "Rechts uitgelijnd verschillen 4 en 3, en geen van beide is 1. Kijk nog eens naar het blok *Let op* boven het recept.",
            "turned": "Van rechts: 4 en 3 verschillen, en geen van beide is 1. Dezelfde getallen, maar op de verkeerde plekken.",
            "four": "Van rechts zijn 4 en 4 gelijk, en de ontbrekende lengte telt als 1: NumPy rekt hem uit over de 3 rijen.",
        },
    )
    choice_fits
    return (choice_fits,)


@app.cell(hide_code=True)
def text_exercise_centered():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Warmer of kouder dan gewoonlijk?

        De temperatuur op vier dagen, 's ochtends, 's middags en 's avonds: `temps` heeft een rij per dag, shape `(4, 3)`. Maak `centered`: elke temperatuur min het gemiddelde van zijn dagdeel, zodat je ziet welke dagen warmer waren dan gewoonlijk. Vervang de `...`. De laatste regel is `centered`, zodat de check het ziet.
        """),
        L.task("Maak `centered`: elke temperatuur van `temps` min het gemiddelde van zijn kolom."),
    ])
    return


@app.cell(hide_code=True)
def exercise_centered():
    box_centered = L.code_box("centered", lines=6)
    box_centered
    return (box_centered,)


@app.cell(hide_code=True)
def check_centered(box_centered):
    result_centered = L.check("centered", box_centered.value)
    result_centered
    return (result_centered,)


@app.cell(hide_code=True)
def hints_centered():
    L.hints(
        "centered",
        [
            "Eerst het gemiddelde van elk dagdeel: één per kolom. Haal die gemiddelden dan af van `temps`.",
            "Eén gemiddelde per kolom: de samenvatting gaat langs de rijen omlaag. Zijn shape, `(3,)`, past van rechts op `(4, 3)`, dus rekt NumPy hem uit over de rijen.",
            "`centered = temps - temps.mean(axis=___)`.",
        ],
        why="`temps.mean(axis=0)` geeft het gemiddelde van elk dagdeel, shape `(3,)`. NumPy rekt het uit over de vier rijen, dus gaat van elke temperatuur het gemiddelde van zijn eigen kolom af.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het aandeel binnen elke stad

        Een verkiezing: `votes` heeft een rij per stad en een kolom per partij, met hoeveel stemmen elke partij daar kreeg. Maak `shares(votes)` af: die geeft voor elke stad het aandeel van elke partij terug, als array met dezelfde shape. Een aandeel is de stemmen van de partij gedeeld door alle stemmen in die stad, dus telt elke rij op tot 1. De functie maakt al een array van `votes`. Vervang de `...`. De check roept `shares([[30, 50, 20], [10, 10, 30]])` aan: dat moet `array([[0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])` geven.
        """),
        L.task("Laat `shares(votes)` elk getal delen door het totaal van zijn eigen rij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=8)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Stap 4 van het recept: één getal per rij, het totaal van elke stad, tegen een matrix.",
            "Het totaal van elke rij is een som langs elke rij. Houd het als kolom, zodat het op `votes` past; deel dan.",
            "`totals = votes.sum(axis=1, ___)`, en dan `return votes / ___`.",
        ],
        why="`votes.sum(axis=1, keepdims=True)` geeft het totaal van elke stad als kolom, shape `(rijen, 1)`. NumPy rekt het uit over de kolommen, dus wordt elk getal gedeeld door het totaal van zijn eigen stad.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) moesten twee arrays dezelfde shape hebben; nu rekt NumPy de kleinste uit, als de shapes passen. In [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) verdween de axis van een samenvatting; met `keepdims=True` blijft hij staan, met lengte 1, zodat de uitkomst past op de array waar hij vandaan kwam. Later in de course deel je elke rij van een tabel met tellingen door zijn totaal, zoals in oefening 3, om van tellingen kansen te maken. In de volgende les, *Willekeurige getallen en seeds*, maak je arrays vol willekeurige getallen, zoals een model begint voordat het iets geleerd heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fits, result_centered, result_shares):
    L.wrap_up(choice_fits, result_centered, result_shares)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
