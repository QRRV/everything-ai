# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-channels-pooling. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-channels-pooling", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_odd():
    quick_odd = L.choice(
        "quick-odd",
        question="Waarom zijn filters bijna altijd oneven groot, zoals 3 bij 3 of 5 bij 5?",
        options={
            "right": "Zodat padding aan beide kanten gelijk is en de grootte blijft",
            "fast": "Omdat een computer oneven maten sneller vermenigvuldigt",
            "center": "Omdat een even filter helemaal niet over een afbeelding kan schuiven",
        },
        correct="right",
        why={
            "fast": "De snelheid hangt nauwelijks af van even of oneven. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "center": "Een even filter schuift prima; alleen de padding om de grootte te houden kan niet aan beide kanten gelijk zijn. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "right": "Met $k = 3$ en $p = 1$ houdt de output de grootte van de input. Veel CNN's houden zo de grootte van hun layers gelijk, en laten pooling de afbeelding expres kleiner maken.",
        },
    )
    quick_odd
    return (quick_odd,)


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="Wat doet een stride van 2 met de breedte van de output van een layer, vergeleken met een stride van 1?",
        options={
            "double": "Die verdubbelt: het filter bestrijkt per stap meer",
            "same": "Niets: de stride verandert alleen hoe het filter beweegt",
            "right": "Die halveert ongeveer: het filter slaat elke tweede plek over",
        },
        correct="right",
        why={
            "double": "Het filter zet grotere stappen, dus het stopt op minder plekken: er komen minder getallen uit. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "same": "De stride staat in de formule voor de grootte van de output: $\\lfloor (n + 2p - k)/s \\rfloor + 1$. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "right": "Vandaag zie je nog een manier om een afbeelding kleiner te maken: pooling, die helemaal geen weights heeft.",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Eén filter vindt één soort patroon, zoals een verticale rand. Een afbeelding bevat er veel: randen in elke richting, hoeken, vlekken, texturen. Een echte layer heeft tientallen filters, elk met een eigen feature map, en de volgende layer combineert die maps tot grotere patronen. En een kleurenfoto is niet één rooster van getallen maar drie: rood, groen en blauw. Ondertussen moet het netwerk van een grote afbeelding naar één antwoord, dus ergens moeten de roosters kleiner worden, zonder te verliezen wat er gevonden is.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "rgb-filters",
        question="Een convolutional layer heeft 64 filters van 3 bij 3 en werkt op een kleurenfoto, met 3 channels: rood, groen en blauw. Hoeveel weights hebben zijn filters samen, zonder bias terms?",
        options={
            "flat": "576: 64 filters van 3 × 3",
            "right": "1.728: 64 filters van 3 × 3 × 3",
            "depends": "Dat hangt van de foto af: meer pixels vragen meer weights",
        },
        reveal="**1.728.** Elk filter heeft een plak van 3 bij 3 voor elk input channel, dus 3 × 3 × 3 = 27 weights, en 64 filters hebben er 64 × 27 = 1.728. Met een bias term per filter heeft de layer 1.792 parameters, evenveel voor een foto van 32 bij 32 pixels als voor een van 1.000 bij 1.000.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een afbeelding met $C_\text{in}$ **channels**, zoals rood, groen en blauw, heeft hier shape $(C_\text{in}, n, n)$: de channels staan voorop, dus `x[i]` is channel `i`. De foto's uit [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl) hadden ze achteraan, met shape (rijen, kolommen, 3). Een filter daarvoor heeft een plak van $k$ bij $k$ voor elk channel: shape $(C_\text{in}, k, k)$. Op elke plek vermenigvuldigt elke plak het stuk van zijn eigen channel, en alle producten van alle channels worden opgeteld tot één getal. Eén filter maakt dus één feature map, een **output channel**. Een layer met $C_\text{out}$ filters, samen van shape $(C_\text{out}, C_\text{in}, k, k)$, maakt $C_\text{out}$ output channels, en dat zijn de input channels van de volgende layer:
        """),
        L.formula(r"\begin{aligned} & [[hidden:\text{out}[o, r, c]]] = \sum_{i=0}^{C_\text{in}-1} \sum_{a=0}^{k-1} \sum_{b=0}^{k-1} \\ & \qquad x[i, r+a, c+b] \cdot [[param:W[o, i, a, b]]] \end{aligned}"),
        mo.md(r"""
        In NumPy neemt `x[:, r:r + k, c:c + k]` het stuk op `r`, `c` uit elk channel tegelijk: shape $(C_\text{in}, k, k)$, de shape van één filter.

        **Pooling** maakt elk channel kleiner. **Max pooling** knipt een channel in blokken van $s$ bij $s$, naast elkaar, en houdt het grootste getal van elk blok: het sterkste signaal blijft, ook als het een pixel binnen zijn blok verschuift. Pooling van 2 bij 2 halveert de breedte en de hoogte, en heeft geen weights: het leert niets. **Average pooling** houdt in plaats daarvan het gemiddelde van elk blok.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **output channel** | | de feature map die één filter maakt; het input channel van de volgende layer | `out[o]` |
        | **pooling** | | een channel kleiner maken door elk blok samen te vatten in één getal | |
        | **max pooling** | | pooling die het grootste getal van elk blok houdt | `.max()` |
        | **average pooling** | | pooling die het gemiddelde van elk blok houdt | `.mean()` |
        | $C_\text{in}$, $C_\text{out}$ | C in, C out | het aantal input channels en het aantal output channels | `len(x)`, `len(W)` |
        """),
        L.watch_out("Een filter is niet 2D: het dekt altijd alle input channels, dus het heeft $C_\\text{in} \\times k \\times k$ weights. En pooling werkt op elk channel apart; het mengt nooit channels. Channels mengen is het werk van de filters."),
        mo.md(r"""
        De figuur: max pooling van 2 bij 2 op één channel van 4 bij 4. In elk van de vier blokken blijft het grootste getal, in een vakje met een dikke rand; de rest wordt grijs.
        """),
        C.pool_picture({"title": "Max pooling van 2 bij 2", "input": "één channel", "output": "na pooling",
                        "input_caption": "(4, 4)", "output_caption": "(2, 2)"}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een layer met channels, dan max pooling:

    1. Controleer de shapes: input $(C_\text{in}, n, n)$, filters $(C_\text{out}, C_\text{in}, k, k)$, output $(C_\text{out}, m, m)$ met $m = n - k + 1$, hier zonder padding en met stride 1.
    2. Output channel `o` op `r`, `c`: het stuk op `r`, `c` van elk channel, `x[:, r:r + k, c:c + k]`, keer filter `o`, getal voor getal, alles opgeteld.
    3. Max pooling met blokken van $s$ bij $s$: voor elk channel begint blok `r`, `c` bij rij `r * s` en kolom `c * s`, en zijn `.max()` blijft; voor C channels heeft de output shape `(C, n // s, n // s)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 voor een input van 2 channels van 4 bij 4 en één filter van shape (2, 3, 3): een diagonaal in elke plak, van linksboven in de eerste en van rechtsboven in de tweede. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_channels():
    import numpy as np

    _x = np.array([[[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [1, 0, 1, 2]],
                   [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]])
    _W = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
    _k = 3
    # 1. de shapes: (2, 4, 4) en (2, 3, 3) geven één output channel van 2 bij 2
    print(_x.shape, _W.shape)
    _m = len(_x[0]) - _k + 1
    _out = np.zeros((_m, _m))
    # 2. het filter op het stuk van elk channel, alles opgeteld
    for _r in range(_m):
        for _c in range(_m):
            _out[_r, _c] = (_x[0, _r:_r + _k, _c:_c + _k] * _W[0]).sum()  # maak af: elk channel, _x[:, ...] * _W
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, kijkt het filter alleen naar channel 0, met zijn eerste plak: `[[2, 2], [2, 3]]`, de sommen langs de diagonalen vanaf linksboven. Afgemaakt, met `(_x[:, _r:_r + _k, _c:_c + _k] * _W).sum()`, telt het tweede channel zijn eigen sommen erbij op, `[[0, 3], [3, 0]]`: samen `[[2, 5], [5, 3]]`. Het tweede channel is een dambord: langs de tegendiagonaal van het filter heeft het drie enen op twee van de vier plekken, en drie nullen op de andere twee. Het ene output channel bevat nu wat beide input channels laten zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De weights van een layer
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights",
        question="Een layer heeft 16 filters van 3 bij 3 en krijgt een input van 8 channels. Hoeveel weights hebben zijn filters, zonder bias terms?",
        options={
            "flat": "144: 16 filters van elk 3 × 3",
            "in": "72: 8 channels van elk 3 × 3",
            "right": "1.152: 16 filters van 8 × 3 × 3",
        },
        correct="right",
        why={
            "flat": "Elk filter dekt alle 8 input channels: een plak van 3 bij 3 voor elk.",
            "in": "Dat is één filter. De layer heeft er 16, één voor elk output channel.",
            "right": "$C_\\text{out} \\times C_\\text{in} \\times k \\times k = 16 \\times 8 \\times 9 = 1.152$, plus 16 bias terms: de output van de layer heeft 16 channels.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_conv_channels():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een layer met channels

        Maak `conv_channels(x, W)` af: de input `x` heeft shape $(C_\text{in}, n, n)$ en de filters `W` shape $(C_\text{out}, C_\text{in}, k, k)$; de functie geeft de $C_\text{out}$ output channels, zonder padding en met stride 1. Vervang de `...` volgens stap 2 van het recept. De check roept `conv_channels` aan met een input van 2 channels van 3 bij 3 en 2 filters van shape (2, 2, 2), wat `[[[4, 6], [4, 5]], [[4, 5], [3, 6]]]` moet geven, en probeert andere inputs, ook van 1 en van 3 channels.
        """),
        L.task("Laat `conv_channels` één output channel per filter geven, uit alle input channels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_conv_channels():
    box_conv_channels = L.code_box("conv-channels", lines=16)
    box_conv_channels
    return (box_conv_channels,)


@app.cell(hide_code=True)
def check_conv_channels(box_conv_channels):
    result_conv_channels = L.check("conv-channels", box_conv_channels.value)
    result_conv_channels
    return (result_conv_channels,)


@app.cell(hide_code=True)
def hints_conv_channels():
    L.hints(
        "conv-channels",
        [
            "Stap 2 van het recept, zoals de afgemaakte regel uit het uitgewerkte voorbeeld. Daar was `_W` één filter; hier bevat `W` alle $C_\\text{out}$ filters, één per output channel.",
            "Het filter van output channel `o` is `W[o]`, met shape $(C_\\text{in}, k, k)$: dezelfde shape als het stuk van elk channel op `r`, `c`, dus de twee vermenigvuldigen getal voor getal.",
        ],
        why="Output channel `o` komt alleen van filter `o`, en elk getal ervan telt producten op uit alle input channels: zo mengt een layer de channels ervoor tot nieuwe.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_max_pool():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Max pooling

        Maak `max_pool(x, size)` af: de input `x` heeft shape $(C, n, n)$; de functie geeft de max pooling van elk channel, met blokken van `size` bij `size` naast elkaar. Vervang de `...` volgens stap 3 van het recept. De check roept `max_pool([[[2, 0, 1, 3], [1, 4, 0, 2], [3, 1, 5, 0], [0, 2, 1, 1]]], 2)` aan, wat `[[[4, 3], [3, 5]]]` moet geven, en probeert andere channels en blokgroottes.
        """),
        L.task("Laat `max_pool` het grootste getal van elk blok houden, per channel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_max_pool():
    box_max_pool = L.code_box("max-pool", lines=13)
    box_max_pool
    return (box_max_pool,)


@app.cell(hide_code=True)
def check_max_pool(box_max_pool):
    result_max_pool = L.check("max-pool", box_max_pool.value)
    result_max_pool
    return (result_max_pool,)


@app.cell(hide_code=True)
def hints_max_pool():
    L.hints(
        "max-pool",
        [
            "Blok `r`, `c` van een channel begint bij rij `r * size` en kolom `c * size`, en is `size` breed.",
            "Neem dat blok uit channel `ch` van `x`, zoals bij de convolutie uit [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl) met een stride van `size`, en houd zijn `.max()`.",
        ],
        why="Pooling stapt met de blokgrootte en houdt het grootste getal in plaats van een gewogen som: de blokken overlappen niet, en elk getal van de input hoort bij precies één blok, of, aan een rafelige rand, bij geen enkel blok.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De convolutie uit [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl) heeft nu channels: een filter dekt alle input channels, en een layer maakt één output channel per filter. Pooling stapt met de blokgrootte en houdt het maximum in plaats van een gewogen som. Een CNN wisselt die twee af: convoluties vinden patronen, pooling maakt de afbeelding kleiner, tot er per channel een paar getallen over zijn voor de laatste layers. In de volgende les, *Convolutie als één grote matrixvermenigvuldiging*, schrijf je al deze loops om tot één matrixvermenigvuldiging, een van de manieren waarop een GPU een convolutie berekent.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_conv_channels, result_max_pool):
    L.wrap_up(choice_weights, result_conv_channels, result_max_pool)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
