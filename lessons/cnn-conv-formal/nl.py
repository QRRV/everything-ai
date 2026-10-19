# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-conv-formal. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-conv-formal", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_depends():
    quick_depends = L.choice(
        "quick-depends",
        question="Van welke pixels van de afbeelding hangt één getal van een feature map af?",
        options={
            "all": "Van alle pixels: het filter schuift over de hele afbeelding",
            "right": "Alleen van de pixels van het stuk onder het filter op die plek",
            "row": "Van de pixels van zijn eigen rij van de afbeelding, van links naar rechts",
        },
        correct="right",
        why={
            "all": "Het filter schuift over de hele afbeelding, maar elk getal van de feature map is de som voor één plek. Zie [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).",
            "row": "Het stuk onder het filter is k bij k: rijen en kolommen rond de plek. Zie [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).",
            "right": "Dat is locality: elk getal kijkt naar een klein stuk. Vandaag zie je hoe dat stuk groeit als je layers stapelt.",
        },
    )
    quick_depends
    return (quick_depends,)


@app.cell(hide_code=True)
def quick_check_edge():
    quick_edge = L.choice(
        "quick-edge",
        question="In `convolve` gaat het filter alleen naar plekken waar het helemaal past. Wat betekent dat voor de pixels aan de rand van de afbeelding?",
        options={
            "right": "Ze liggen minder vaak onder het filter dan pixels in het midden",
            "skip": "Geen enkele plek van het filter dekt ze, dus ze worden overgeslagen",
            "same": "Niets: elke pixel, aan de rand of in het midden, ligt even vaak onder het filter",
        },
        correct="right",
        why={
            "skip": "De hoekpixel ligt op één plek onder het filter, de eerste; hij wordt niet overgeslagen, maar telt maar één keer. Zie [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).",
            "same": "Een pixel in het midden ligt onder k × k plekken van het filter, een hoekpixel onder één. Zie [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl).",
            "right": "En de feature map is kleiner dan de afbeelding. Vandaag laat padding het filter ook over de rand gaan: pixels aan de rand tellen vaker mee, en de feature map kan even groot blijven als de afbeelding.",
        },
    )
    quick_edge
    return (quick_edge,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een echte convolutional layer heeft naast zijn weights drie instellingen: hoe groot het filter is, hoe ver het per keer opschuift, en wat het aan de rand doet. Samen bepalen ze hoe groot de output is, en dus hoeveel getallen de volgende layer krijgt en hoeveel rekenwerk het kost. Ze bepalen ook hoeveel van de afbeelding een getal diep in het netwerk kan zien: een voorwerp wordt pas herkend als het netwerk er genoeg tegelijk van ziet. Elk CNN dat je bouwt of leest, van LeNet tot ResNet, wordt in deze instellingen beschreven.
    """)
    return


@app.cell(hide_code=True)
def predict_field():
    guess_field = L.guess(
        "three-layers",
        question="Drie convolutional layers met filters van 3 bij 3, op elkaar gestapeld, waarvan elk filter één plek per keer opschuift. Hoeveel pixels breed is het stuk van de afbeelding waarvan één getal van de output van de derde layer afhangt?",
        options={
            "three": "3: de grootte van één filter",
            "right": "7: elke layer voegt er 2 toe",
            "nine": "9: drie filters van 3",
        },
        reveal="**7 bij 7.** Een getal van de eerste layer ziet 3 bij 3 pixels. Een getal van de tweede ziet 3 bij 3 getallen van de eerste, die elk hun eigen 3 bij 3 pixels zien, één plek verschoven: samen 5 bij 5. De derde voegt er weer 2 toe: 7 bij 7. Kleine filters, gestapeld, kijken naar een groot deel van de afbeelding.",
    )
    guess_field
    return (guess_field,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **kernel size** $k$ is de grootte van het filter, $k$ bij $k$: *kernel* is een ander woord voor filter, en de course houdt het alleen in deze naam. De **stride** $s$ is hoeveel plekken het filter per keer opschuift: met $s = 2$ slaat het elke tweede plek over, en wordt de output ongeveer half zo breed. **Padding** $p$ zet een rand van nullen, $p$ breed, rond de afbeelding, zodat het filter ook aan de rand past; met $k = 3$, $p = 1$ en $s = 1$ is de output even groot als de afbeelding. Samen geven ze de grootte van de output van een afbeelding van $n$ bij $n$:
        """),
        L.formula(r"n_\text{out} = \left\lfloor \frac{n + 2[[hyper:p]] - [[hyper:k]]}{[[hyper:s]]} \right\rfloor + 1"),
        mo.md(r"""
        $n + 2p$ is de breedte van de afbeelding met padding, $n + 2p - k$ het aantal plekken dat het filter rechts van zijn eerste plek kan opschuiven, gedeeld door $s$ het aantal stappen dat het zet, en $+1$ telt de eerste plek. De haken $\lfloor \, \rfloor$ ronden af naar beneden: een plek waar het filter zou uitsteken, telt niet. In Python is dat `//`, delen en naar beneden afronden: `7 // 2` geeft `3`.

        Elk getal van de output hangt af van een stuk van $k$ bij $k$ van de input. Stapel je layers, dan hangt een getal van de tweede layer af van $k$ bij $k$ getallen van de eerste, die elk hun eigen stuk zien: samen een breder stuk. Dat stuk van de afbeelding is het **receptive field**. Met filters van 3 bij 3 en stride 1 voegt elke layer 2 toe aan de breedte: 3, 5, 7. Een stride maakt groter wat de layers erna toevoegen: na een layer met stride 2 liggen naburige getallen 2 pixels uit elkaar, dus voegt elke layer van 3 bij 3 daarna er 4 toe in plaats van 2; elke volgende stride van 2 verdubbelt dat weer.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **kernel size** | | de grootte van het filter, $k$ bij $k$ | `k` |
        | **stride** | | hoeveel plekken het filter per keer opschuift | `s` |
        | **padding** | | een rand van nullen rond de input, zodat het filter ook aan de rand past | `p` |
        | **receptive field** | | het stuk van de afbeelding waarvan één getal van een layer afhangt | |
        | $\lfloor x \rfloor$ | entier van x | $x$ naar beneden afgerond op een heel getal | `//` |
        """),
        L.watch_out("Met een even kernel size kan de padding niet aan beide kanten gelijk zijn en de grootte houden bij stride 1: $k = 3$ met $p = 1$ en $k = 5$ met $p = 2$ houden hem, wat een van de redenen is dat filters bijna altijd oneven zijn. En de nullen van de padding horen niet bij de afbeelding: rond een lichte afbeelding ziet het filter voor verticale randen een rand langs de hele linker- en rechterkant."),
        mo.md(r"""
        De figuur: een afbeelding van 5 bij 5 met een rand van nullen, 1 breed, een filter van 3 bij 3 en een stride van 2. Het doorgetrokken kader is het filter op zijn eerste plek, het gestippelde op zijn tweede, 2 plekken verder; de eerste twee vakjes van de output zijn de getallen van die twee plekken. De output is $\lfloor (5 + 2 - 3) / 2 \rfloor + 1 = 3$ bij 3.
        """),
        C.conv_picture({"input": "input, met padding 1", "output": "output", "caption": "n = 5, k = 3, s = 2, p = 1: een output van 3 bij 3",
                        "alt": "Een rooster van 7 bij 7 vakjes: een afbeelding van 5 bij 5 met een rand van nullen. Een doorgetrokken kader van 3 bij 3 dekt de hoek linksboven, een gestippeld kader begint 2 vakjes verder naar rechts. Ernaast een output van 3 bij 3, waarvan de eerste twee vakjes gekleurd zijn."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een convolutie met kernel size $k$, stride $s$ en padding $p$:

    1. Padding: een array van nullen van $n + 2p$ bij $n + 2p$, met de afbeelding in het midden.
    2. Bereken de grootte van de output, $\lfloor (n + 2p - k)/s \rfloor + 1$: in Python rondt `//` naar beneden af.
    3. Voor elke outputplek `r`, `c`: het stuk van $k$ bij $k$ van de afbeelding met padding dat begint bij rij `r * s` en kolom `c * s`, keer het filter, opgeteld.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een afbeelding van 4 bij 4 met padding 1, een filter van 3 bij 3 enen, dat het stuk eronder optelt, en stride 2. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_stride():
    import numpy as np

    _img = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])
    _K = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    _n, _k, _s, _p = 4, 3, 2, 1
    # 1. padding: een rand van nullen
    _x = np.zeros((_n + 2 * _p, _n + 2 * _p))
    _x[_p:_p + _n, _p:_p + _n] = _img
    # 2. de grootte van de output
    _m = (_n + 2 * _p - _k) // _s + 1
    print(_m)
    # 3. het filter op elke plek, _s plekken uit elkaar
    _out = np.zeros((_m, _m))
    for _r in range(_m):
        for _c in range(_m):
            _out[_r, _c] = (_x[_r:_r + _k, _c:_c + _k] * _K).sum()  # maak af: begin bij rij _r * _s en kolom _c * _s
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De grootte van de output is $\lfloor (4 + 2 - 3) / 2 \rfloor + 1 = 2$. Zoals hij is, schuift het filter één plek per keer op: het telt de stukken bij rij en kolom 0 en 1 op, `[[4, 9], [7, 13]]`, de hoek linksboven van een output met stride 1. Afgemaakt, met `_x[_r * _s:_r * _s + _k, _c * _s:_c * _s + _k]`, beginnen de stukken bij 0 en 2: `[[4, 11], [9, 13]]`. Het eerste getal, 4, is de som van het stuk linksboven: vijf nullen van de rand en 1 + 2 + 0 + 1 van de afbeelding.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_receptive():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een stride in de stapel
    """)
    return


@app.cell(hide_code=True)
def choice_receptive_cell():
    choice_receptive = L.choice(
        "receptive",
        question="Twee layers: eerst een filter van 3 bij 3 met stride 2, dan een filter van 3 bij 3 met stride 1. Hoeveel pixels breed is het stuk van de afbeelding waarvan één getal van de output van de tweede layer afhangt?",
        options={
            "five": "5: elke layer voegt er 2 toe, zoals zonder stride",
            "nine": "9: drie getallen van de eerste layer, elk 3 pixels breed",
            "right": "7: de tweede layer voegt na de stride twee keer zoveel toe",
        },
        correct="right",
        why={
            "five": "Na de stride van 2 liggen naburige getallen van de eerste layer 2 pixels uit elkaar, dus dekken de 2 extra getallen van de tweede layer 4 extra pixels.",
            "nine": "De drie getallen van de eerste layer overlappen: elk ziet 3 pixels, en ze beginnen 2 pixels uit elkaar, dus samen zien ze 2 + 2 + 3 = 7.",
            "right": "De eerste layer ziet 3 pixels; de tweede voegt 2 getallen van de eerste toe, elk 2 pixels verder: 3 + 2 × 2 = 7.",
        },
    )
    choice_receptive
    return (choice_receptive,)


@app.cell(hide_code=True)
def text_exercise_out_size():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De grootte van de output

        Maak `out_size(n, k, s, p)` af: de breedte van de output van een afbeelding van `n` bij `n`, met kernel size `k`, stride `s` en padding `p`, als heel getal. Vervang de `...` door de formule uit Het idee, in Python. De check roept `out_size(28, 5, 2, 0)` aan, wat `12` moet geven, en probeert andere maten.
        """),
        L.task("Laat `out_size` de grootte van de output van een convolutie geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_out_size():
    box_out_size = L.code_box("out-size", lines=3)
    box_out_size
    return (box_out_size,)


@app.cell(hide_code=True)
def check_out_size(box_out_size):
    result_out_size = L.check("out-size", box_out_size.value)
    result_out_size
    return (result_out_size,)


@app.cell(hide_code=True)
def hints_out_size():
    L.hints(
        "out-size",
        [
            "De breedte van de afbeelding met padding is `n + 2 * p`; het filter kan `n + 2 * p - k` plekken rechts van zijn eerste plek opschuiven.",
            "Deel dat door `s` en rond naar beneden af met `//`, en tel dan de eerste plek erbij.",
        ],
        why="Voor `out_size(28, 5, 2, 0)`: het filter kan 23 plekken rechts van zijn eerste plek opschuiven; in stappen van 2 zijn dat 11 stappen, naar beneden afgerond, en met de eerste plek erbij 12 plekken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_conv2d():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een convolutie met stride en padding

        Maak `conv2d(img, K, s, p)` af: de convolutie van de vierkante afbeelding `img` met het vierkante filter `K`, met stride `s` en padding `p`. Vervang de drie `...`, volgens stap 1 tot en met 3 van het recept. De check roept `conv2d([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], [[1, 0], [0, -1]], 2, 0)` aan, wat `[[0, -1], [2, -2]]` moet geven, en probeert andere filters, strides en paddings.
        """),
        L.task("Laat `conv2d` een filter over een afbeelding schuiven met stride en padding."),
    ])
    return


@app.cell(hide_code=True)
def exercise_conv2d():
    box_conv2d = L.code_box("conv2d", lines=17)
    box_conv2d
    return (box_conv2d,)


@app.cell(hide_code=True)
def check_conv2d(box_conv2d):
    result_conv2d = L.check("conv2d", box_conv2d.value)
    result_conv2d
    return (result_conv2d,)


@app.cell(hide_code=True)
def hints_conv2d():
    L.hints(
        "conv2d",
        [
            "Stap 1 is een array van nullen met `np.zeros`, `n + 2 * p` bij `n + 2 * p`; de regel eronder zet de afbeelding in het midden. Stap 2 is je formule uit `out_size`.",
            "Stap 3 is de regel uit `convolve`, met het stuk uit `x`, beginnend bij rij `r * s` en kolom `c * s`.",
        ],
        why="Met stride 2 begint outputplek 1 bij rij 2 van de afbeelding met padding, dus het begin van een stuk is `r * s`, en het eindigt `k` verder. Door het stuk uit `x` te nemen in plaats van uit `img` komen aan de rand de nullen van de padding mee.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** `convolve` uit [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl) was het geval $s = 1$, $p = 0$: zijn outputgrootte, $H - k + 1$, is de formule van vandaag met $n = H$ en die waarden. Kernel size, stride en padding zijn hyperparameters van een layer, gekozen vóór het trainen; de weights van het filter zijn wat de layer leert. In de volgende les, *Channels en pooling*, krijgt een layer veel filters tegelijk, die elk hun eigen output maken, en maakt pooling de afbeelding kleiner terwijl de sterkste signalen blijven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_receptive, result_out_size, result_conv2d):
    L.wrap_up(choice_receptive, result_out_size, result_conv2d)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
