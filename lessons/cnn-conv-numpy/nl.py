# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-conv-numpy. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-conv-numpy", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl) en [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_channels():
    quick_channels = L.choice(
        "quick-channels",
        question="Een layer krijgt een input met shape (3, 10, 10) en heeft 6 filters van 3 bij 3, zonder padding en met stride 1. Welke shape heeft zijn output?",
        options={
            "per-in": "(3, 8, 8): één output channel per input channel",
            "right": "(6, 8, 8): één output channel per filter",
            "both": "(18, 8, 8): één voor elk filter en input channel",
        },
        correct="right",
        why={
            "per-in": "Elk filter dekt alle 3 input channels tegelijk en telt ze op. Zie [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl).",
            "both": "Een filter maakt geen channel per input channel: het telt de producten van allemaal op tot één getal. Zie [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl).",
            "right": "Elk van de 6 filters geeft één output channel van $10 - 3 + 1 = 8$ bij 8. Elk van die $6 \\times 64$ getallen is één som van 27 producten: vandaag reken je ze allemaal tegelijk uit.",
        },
    )
    quick_channels
    return (quick_channels,)


@app.cell(hide_code=True)
def quick_check_matmul():
    quick_matmul = L.choice(
        "quick-matmul",
        question="`A` en `B` zijn matrices waarvan de maten passen. Welk getal staat in rij `i` en kolom `j` van `A @ B`?",
        options={
            "same-place": "Het getal op rij `i`, kolom `j` van `A`, keer dat op dezelfde plek van `B`",
            "sums": "De som van rij `i` van `A`, keer de som van kolom `j` van `B`",
            "right": "Het inproduct van rij `i` van `A` en kolom `j` van `B`",
        },
        correct="right",
        why={
            "same-place": "Dat is `A * B`, getal voor getal. Zie [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "sums": "Eerst worden de getallen paarsgewijs vermenigvuldigd, en dan opgeteld: een inproduct. Zie [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "right": "En omdat kolom `j` van `B.T` rij `j` van `B` is, bevat rij `i` van `A @ B.T` de inproducten van rij `i` van `A` met elke rij van `B`.",
        },
    )
    quick_matmul
    return (quick_matmul,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De loops uit [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl) rekenen elk getal van de output apart uit. Voor een layer van 16 filters van 3 bij 3 op een kleurenafbeelding van 32 bij 32 loopt de regel binnen de loops 16 × 30 × 30 = 14.400 keer, en elke keer kost Python een beetje tijd. Een computer is het snelst met één grote, regelmatige klus, en de klus waarvoor rekenbibliotheken het best zijn afgesteld, is de matrixvermenigvuldiging: NumPy geeft die aan een bibliotheek waaraan al tientallen jaren geschaafd is, en een GPU voert duizenden van die vermenigvuldigingen tegelijk uit. Dus schrijf je een convolutie als één matrixvermenigvuldiging.
    """)
    return


@app.cell(hide_code=True)
def predict_copies():
    guess_copies = L.guess(
        "copies",
        question="Een kleurenfoto van 32 bij 32 pixels bevat 3 × 32 × 32 = 3.072 getallen. Voor filters van 3 bij 3, met padding 1 en stride 1, schrijf je elk stuk dat het filter bedekt als één rij van een nieuwe matrix. Hoeveel getallen bevat die matrix?",
        options={
            "same": "3.072: dezelfde getallen, alleen in een andere volgorde en vorm",
            "right": "27.648: 9 keer zoveel, want een pixel komt in hooguit 9 stukken terecht",
            "per-piece": "82.944: 27 keer zoveel, want elk stuk heeft 27 getallen",
        },
        reveal="**27.648, precies 9 keer zoveel.** Padding 1 houdt 32 × 32 = 1.024 plekken, en het stuk op elke plek heeft 3 × 3 × 3 = 27 getallen: 1.024 × 27 = 27.648. Een pixel die niet aan de rand ligt, zit in de stukken van 9 plekken, dus hij wordt 9 keer gekopieerd; de nullen van de padding vullen aan wat aan de randen ontbreekt.",
    )
    guess_copies
    return (guess_copies,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Elk getal van de output van een convolutie is een inproduct: het stuk van de input onder het filter, zijn $C_\text{in} \cdot k \cdot k$ getallen, met de weights van één filter. Dat stuk heet een **patch**. **im2col** schrijft elke patch als één platte rij van een matrix $\mathbf{P}$, plek na plek, rij voor rij van de afbeelding: $m \cdot m$ rijen van $C_\text{in} k^2$ getallen. Elk filter wordt één rij van $\mathbf{W}$, in dezelfde volgorde platgemaakt: $C_\text{out}$ rijen van $C_\text{in} k^2$ getallen. Alle inproducten tegelijk zijn dan één matrixvermenigvuldiging:
        """),
        L.formula(r"[[hidden:\text{out}]] = [[param:\mathbf{W}]] \, \mathbf{P}^\top"),
        mo.md(r"""
        $\mathbf{W}$ heeft shape $(C_\text{out}, C_\text{in} k^2)$ en $\mathbf{P}^\top$ shape $(C_\text{in} k^2, m^2)$, dus de binnenste maten passen en out heeft shape $(C_\text{out}, m^2)$. Rij $o$ van out bevat filter $o$ op alle $m^2$ plekken, in de volgorde van de rijen van $\mathbf{P}$, en `.reshape(c_out, m, m)` maakt van elke rij weer een output channel van $m$ bij $m$. De naam im2col, kort voor *image to columns*, komt van de eerste versies, `im2col` van MATLAB en daarna Caffe, die elke patch in een kolom zetten; met rijen krijg je dezelfde matrix, getransponeerd. Stride en padding veranderen alleen welke patches je knipt: eerst padden, en dan stappen van $s$.

        ### Nieuwe woorden en tekens

        | Woord of symbool | Uitspraak | Betekenis | In code |
        |---|---|---|---|
        | **patch** | | het stuk van de input onder het filter op één plek, van elk channel: $C_\text{in} \cdot k \cdot k$ getallen | `x[:, r:r + k, c:c + k]` |
        | **im2col** | image to columns | elke patch als één rij van een matrix schrijven, zodat de convolutie één matrixvermenigvuldiging wordt | `patches(x, k)` |
        | $\mathbf{P}$ | hoofdletter P | de patches als rijen: één rij per plek, shape $(m^2, C_\text{in} k^2)$ | `P` |
        | $\mathbf{W}$ | hoofdletter W | de filters als rijen: shape $(C_\text{out}, C_\text{in} k^2)$ | `W.reshape(c_out, c_in * k * k)` |
        | **`.reshape(n)`** | reshape | met één getal: alle getallen in één platte rij, channel na channel, rij na rij | `patch.reshape(c_in * k * k)` |
        """),
        L.watch_out("im2col kopieert: een pixel die niet aan de rand ligt, komt in de patches van $k^2$ plekken terecht. Voor 64 channels van 224 bij 224 en filters van 3 bij 3 met padding 1 bevat $\\mathbf{P}$ bijna 29 miljoen getallen, tegen 3,2 miljoen in de input. Daarom knippen GPU-bibliotheken de patches tegel voor tegel terwijl ze vermenigvuldigen, in plaats van heel $\\mathbf{P}$ in het geheugen te houden."),
        mo.md(r"""
        De figuur: één channel van 3 bij 3 en een filter van 2 bij 2, dus 2 × 2 = 4 plekken en 4 patches van 4 getallen. De patch op rij 0, kolom 1 van de afbeelding wordt rij 1 van $\mathbf{P}$: de plekken worden rij voor rij geteld.
        """),
        C.patches_picture({"title": "Elke patch wordt een rij", "input": "één channel", "output": "P, de patches als rijen",
                           "input_note": "patch op rij 0, kolom 1", "output_note": "rij 1"}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een convolutie als één matrixvermenigvuldiging, zonder padding en met stride 1:

    1. Tel de plekken per kant: $m = n - k + 1$.
    2. Voor elke plek, rij voor rij: de patch `x[:, r:r + k, c:c + k]`, plat met `.reshape(c_in * k * k)`, is de volgende rij van $\mathbf{P}$.
    3. Elk filter als één rij, in dezelfde volgorde: `W.reshape(c_out, c_in * k * k)`.
    4. Vermenigvuldig, `Wm @ P.T`, en geef elk filter zijn output channel: `.reshape(c_out, m, m)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén channel van 4 bij 4 en twee filters van shape (1, 2, 2): een diagonaal, en een klein verschil tussen rechtsboven en linksonder. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_im2col():
    import numpy as np

    _x = np.array([[[1, 2, 0, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 0, 3, 2]]])
    _W = np.array([[[[1, 0], [0, 1]]], [[[0, 1], [-1, 0]]]])
    _c_in, _k = 1, 2
    # 1. de plekken per kant
    _m = len(_x[0]) - _k + 1
    # 2. elke patch als één rij, plek na plek
    _rows = []
    for _r in range(_m):
        for _c in range(_m):
            _rows.append(_x[:, _r:_r + _k, _c:_c + _k].reshape(_c_in * _k * _k))
    _P = np.array(_rows)
    print(_P.shape)
    # 3. elk filter als één rij
    _Wm = _W.reshape(2, _c_in * _k * _k)
    # 4. alle inproducten tegelijk
    _out = _Wm @ _P.T  # maak af: geef elk filter zijn output channel, .reshape(2, _m, _m)
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\mathbf{P}$ heeft shape (9, 4): 3 × 3 plekken, en 4 getallen in elke patch. Zoals hij is, print de cel het product, met shape (2, 9): de eerste rij is het diagonaalfilter op de 9 plekken, `[2 4 0 5 2 5 0 5 3]`, en de tweede rij het andere filter. Afgemaakt, met `(_Wm @ _P.T).reshape(2, _m, _m)`, print hij twee output channels van 3 bij 3, `[[2 4 0] [5 2 5] [0 5 3]]` en `[[-1 -1 -1] [1 0 -1] [1 1 0]]`: precies wat je `conv_channels` uit [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl) voor deze input geeft, getal voor getal.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De shapes
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape",
        question="Een input met shape (3, 8, 8) en 16 filters met shape (3, 3, 3), zonder padding en met stride 1. Welke shape heeft $\\mathbf{W}\\mathbf{P}^\\top$, vóór de reshape?",
        options={
            "right": "(16, 36): een rij voor elk filter",
            "per-place": "(36, 16): een rij voor elke plek",
            "per-pixel": "(16, 64): een kolom voor elke pixel",
        },
        correct="right",
        why={
            "per-place": "Dat is $\\mathbf{P}\\mathbf{W}^\\top$, dezelfde getallen getransponeerd. $\\mathbf{W}$ staat voorop, dus zijn rijen, de filters, zijn de rijen van het resultaat.",
            "per-pixel": "Een filter van 3 bij 3 past op $8 - 3 + 1 = 6$ plekken per kant, niet op alle 8: het resultaat heeft een kolom voor elke plek.",
            "right": "$\\mathbf{W}$ is (16, 27) en $\\mathbf{P}^\\top$ is (27, 36): 16 filters, elk op 6 × 6 = 36 plekken. `.reshape(16, 6, 6)` geeft de 16 output channels.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_patches():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. im2col

        Maak `patches(x, k)` af: de input `x` heeft shape $(C_\text{in}, n, n)$, en de functie geeft $\mathbf{P}$, elke patch voor een filter van `k` bij `k` als één rij, zonder padding en met stride 1. Vervang de `...` volgens stap 2 van het recept. De check roept `patches` aan met een input van 2 channels van 3 bij 3 en `k` = 2, wat 4 rijen van 8 getallen moet geven, `[[1, 2, 0, 1, 0, 1, 1, 0], …]`, en probeert andere inputs en kernel sizes.
        """),
        L.task("Laat `patches` elke patch van de input als één platte rij schrijven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_patches():
    box_patches = L.code_box("patches", lines=13)
    box_patches
    return (box_patches,)


@app.cell(hide_code=True)
def check_patches(box_patches):
    result_patches = L.check("patches", box_patches.value)
    result_patches
    return (result_patches,)


@app.cell(hide_code=True)
def hints_patches():
    L.hints(
        "patches",
        [
            "Stap 2 van het uitgewerkte voorbeeld, met `x`, `r`, `c`, `k` en `c_in` in plaats van de namen met `_`.",
            "`x[:, r:r + k, c:c + k]` neemt de patch uit elk channel tegelijk: shape $(C_\\text{in}, k, k)$. Eén platte rij heeft al zijn $C_\\text{in} \\cdot k \\cdot k$ getallen.",
        ],
        why="Elke rij van $\\mathbf{P}$ is alles waar één filter op één plek naar kijkt, dus één inproduct met een filterrij geeft één getal van de output. De loops knippen alleen; het vermenigvuldigen komt later, allemaal tegelijk.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_conv_matmul():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De convolutie als één `@`

        Maak `conv_matmul(P, W, m)` af: `P` is het resultaat van `patches`, `W` bevat de filters, met shape $(C_\text{out}, C_\text{in}, k, k)$, en `m` is het aantal plekken per kant. De functie geeft de $C_\text{out}$ output channels van $m$ bij $m$. Vervang de twee `...` volgens stap 3 en 4 van het recept. De check roept `conv_matmul` aan met de 4 patches van 2 channels van 3 bij 3 en 2 filters met shape (2, 2, 2), wat `[[[2, 0], [5, 2]], [[2, 2], [4, 1]]]` moet geven, en probeert andere filters en channels.
        """),
        L.task("Laat `conv_matmul` alle output channels uitrekenen met één matrixvermenigvuldiging."),
    ])
    return


@app.cell(hide_code=True)
def exercise_conv_matmul():
    box_conv_matmul = L.code_box("conv-matmul", lines=11)
    box_conv_matmul
    return (box_conv_matmul,)


@app.cell(hide_code=True)
def check_conv_matmul(box_conv_matmul):
    result_conv_matmul = L.check("conv-matmul", box_conv_matmul.value)
    result_conv_matmul
    return (result_conv_matmul,)


@app.cell(hide_code=True)
def hints_conv_matmul():
    L.hints(
        "conv-matmul",
        [
            "Een rij van `Wm` moet even lang zijn als een rij van `P`: `len(P[0])` getallen. Stap 3 en 4 van het uitgewerkte voorbeeld laten de rest zien.",
            "`Wm @ P.T` heeft een rij per filter en een kolom per plek, plek na plek, rij voor rij; `.reshape(c_out, m, m)` knipt elke rij in `m` rijen van `m`.",
        ],
        why="Een filter en een patch worden in dezelfde volgorde platgemaakt, channel na channel en rij na rij, dus getal $j$ van een filterrij ontmoet getal $j$ van een patchrij: het inproduct is precies de som uit [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl).",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De loops uit [Channels en pooling](/course/open/cnn-channels-pooling?lang=nl) en de matrixvermenigvuldiging uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) zijn dezelfde convolutie: elk getal van de output is het inproduct van een patch en een filter. Zelfs als een Python-loop de patches knipt, is de matrixversie van een layer met 16 filters op een kleurenafbeelding van 32 bij 32 in NumPy meer dan 10 keer zo snel als die loops, en het verschil groeit met het aantal filters. Kumar Chellapilla, Sidd Puri en Patrice Simard schreven CNN's al zo in 2006, om ze snel te laten draaien, ook op een GPU; cuDNN, de bibliotheek voor deep learning op GPU's van NVIDIA, rekent nog steeds veel convoluties uit als matrixvermenigvuldigingen, en knipt de patches daarbij tegel voor tegel. In de volgende les, *Project: je eerste CNN*, doet `nn.Conv2d` van PyTorch dit allemaal voor je, en train je een kleine CNN op Fashion-MNIST.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_patches, result_conv_matmul):
    L.wrap_up(choice_shape, result_patches, result_conv_matmul)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
