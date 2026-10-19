# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-convolution-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cnn-convolution-idea", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl) en [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_piece():
    quick_piece = L.choice(
        "quick-piece",
        question="`img` is een afbeelding met shape (28, 28). Welke shape heeft `img[5:8, 10:13]`?",
        options={
            "right": "(3, 3)",
            "stop": "(4, 4): van 5 tot en met 8",
            "values": "(8, 13)",
        },
        correct="right",
        why={
            "stop": "Een slice stopt net voor zijn eind: `5:8` is rij 5, 6 en 7. Zie [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl).",
            "values": "De getallen in een slice zeggen waar hij begint en stopt, niet hoe groot hij is. Zie [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl).",
            "right": "Rij 5, 6 en 7, en kolom 10, 11 en 12: een stuk van 3 bij 3 pixels.",
        },
    )
    quick_piece
    return (quick_piece,)


@app.cell(hide_code=True)
def quick_check_neuron():
    quick_neuron = L.choice(
        "quick-neuron",
        question="Een hidden layer krijgt een afbeelding van 28 bij 28 pixels als 784 inputs. Hoeveel weights heeft elk van zijn neuronen, zonder zijn bias term?",
        options={
            "one": "1: één weight per neuron",
            "nine": "9: één per pixel en de pixels eromheen",
            "right": "784: één per pixel",
        },
        correct="right",
        why={
            "one": "Een neuron maakt een gewogen som van al zijn inputs, elk met zijn eigen weight. Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "nine": "Een neuron van een hidden layer ziet al zijn inputs, niet een klein stuk ervan. Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "right": "Elk neuron maakt een gewogen som van alle 784 inputs, dus het heeft een rij van 784 weights in `W`.",
        },
    )
    quick_neuron
    return (quick_neuron,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een neuron van een hidden layer heeft één weight voor elke pixel. Een streep linksboven in een afbeelding en dezelfde streep rechtsonder komen bij totaal andere weights uit, dus moet het netwerk die streep opnieuw leren voor elke plek waar hij kan staan. Toch ziet een rand, een hoek of een lus er overal hetzelfde uit. Je kunt er beter naar zoeken met één klein setje weights, dat op elke plek in de afbeelding hetzelfde is.

    **Een voorbeeld uit het dagelijks leven.** Om elke "de" in een boek te vinden, leer je geen aparte regel voor bladzijde 1, bladzijde 2 enzovoort. Je schuift dezelfde twee letters langs elke regel, en noteert waar ze overeenkomen.
    """)
    return


@app.cell(hide_code=True)
def predict_places():
    guess_places = L.guess(
        "places",
        question="Een filter van 3 bij 3 pixels schuift over een afbeelding van 28 bij 28 pixels, naar elke plek waar het er helemaal op past. Hoeveel plekken denk je dat dat zijn?",
        options={
            "all": "784: één voor elke pixel",
            "right": "676",
            "nine": "9: zoveel als het filter pixels heeft",
        },
        reveal="**676: 26 bij 26.** Langs een rij kan de linkerrand van het filter beginnen bij kolom 0 tot en met 25: bij 26 zou het over kolom 27 heen steken. Voor de rijen geldt hetzelfde. De uitkomst is dus een rooster van 26 bij 26: twee rijen en twee kolommen kleiner dan de afbeelding.",
    )
    guess_places
    return (guess_places,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **filter** is een klein rooster van weights, `K`, van $k$ bij $k$, vaak 3 bij 3. Een **convolutie** schuift het over de afbeelding. Op elke plek waar het past, vermenigvuldigt het elke pixel eronder met de weight erboven, en telt de producten op. Die som is één getal van de **feature map**, `out`:
        """),
        L.formula(r"[[hidden:\text{out}[r, c]]] = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} \text{img}[r+i, c+j] \cdot [[param:K[i, j]]]"),
        mo.md(r"""
        In NumPy is dat `(img[r:r + k, c:c + k] * K).sum()`: het stuk van de afbeelding van $k$ bij $k$ dat begint bij rij $r$ en kolom $c$, keer het filter, getal voor getal, en dan de som. Een afbeelding van $H$ rijen bij $W$ kolommen, haar hoogte en breedte, geeft een feature map van $H - k + 1$ bij $W - k + 1$.

        Twee ideeën maken dit anders dan een hidden layer:

        - **Weight sharing.** Op elke plek worden dezelfde $k \cdot k$ weights gebruikt: een filter van 3 bij 3 heeft 9 weights voor de hele afbeelding, waar een neuron van een hidden layer er 784 heeft.
        - **Locality.** Elk getal van de feature map hangt alleen af van een klein stuk van de afbeelding, de $k$ bij $k$ pixels onder het filter, niet van allemaal.

        Een feature map is groot waar de afbeelding op het filter lijkt. Het filter `[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]` neemt de kolom rechts min de kolom links. Het geeft dus een groot positief getal waar de getallen van links naar rechts van laag naar hoog gaan: een verticale rand. Andersom, van hoog naar laag, geeft het een groot negatief getal; op een egaal vlak 0. In een netwerk dat leert, worden de weights van de filters ook geleerd. Hier stel je ze met de hand in. Strikt genomen draait een convolutie in de wiskunde het filter eerst ondersteboven en van links naar rechts om; deep learning slaat die stap over en noemt het toch een convolutie.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **filter** | | een klein rooster van weights, $k$ bij $k$, dat over een afbeelding schuift; ook kernel genoemd | `K` |
        | $k$ | k | de grootte van het filter: $k$ rijen en $k$ kolommen | `k = len(K)` |
        | **convolutie** | | een filter over een afbeelding schuiven en op elke plek de producten van pixels en weights optellen | `(img[r:r + k, c:c + k] * K).sum()` |
        | **feature map** | | het rooster van sommen dat een convolutie geeft: groot waar de afbeelding op het filter lijkt | `out` |
        | **weight sharing** | | dezelfde weights op elke plek van de input | |
        | **locality** | | elke output hangt alleen af van een klein stuk van de input | |
        | **`np.zeros((rows, cols))`** | | een rooster van nullen met `rows` rijen en `cols` kolommen; de shape gaat erin als tuple, vandaar de twee paar haakjes | `np.zeros((2, 3))` |
        | **`out[r, c] = x`** | | zet het getal `x` in rij `r` en kolom `c` van `out`; met een slice op elke plek van dat stuk: `img[:, 3:] = 1` | `out[0, 1] = 5` |
        """),
        L.watch_out("\"Kernel\" betekent in AI meerdere dingen. Bij een convolutie is het een ander woord voor het filter. Maar het Python-proces achter een marimo-notebook heet ook een kernel, en een klein programma dat op een GPU (graphics processing unit) draait ook. Deze course zegt filter voor de weights van een convolutie."),
        mo.md(r"""
        De figuur: beeld 0 van QMNIST, een verzameling handgeschreven cijfers die de course in `data/qmnist-small.npz` heeft. Het is een 5. Ernaast staan zijn feature maps voor het filter voor verticale randen hierboven en voor `[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]`, dat zoekt naar een horizontale rand met lage getallen erboven en hoge eronder. Magenta is een groot positief getal, grijs een groot negatief getal.
        """),
        mo.Html(C.picture({"title": "Eén cijfer, twee filters", "image": "de afbeelding", "vertical": "verticale randen",
                           "horizontal": "horizontale randen",
                           "alt": "Een handgeschreven 5, en twee feature maps van 26 bij 26. De map van het filter voor verticale randen is magenta langs de linkerkant van elke streep en grijs langs de rechterkant; de map van het filter voor horizontale randen is magenta langs de bovenkant van elke streep en grijs langs de onderkant."})),
        mo.md(r"""
        In de getallen is een pixel 1 voor inkt en 0 voor papier. Daarom is de map van het filter voor verticale randen magenta aan de linkerkant van elke streep, waar 0 voor papier overgaat in 1 voor inkt, en grijs aan de rechterkant. De map van het horizontale filter doet hetzelfde boven en onder elke streep. De platte bovenkant van de 5 valt duidelijk op in de horizontale map, de rechtopstaande streep in de verticale.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een afbeelding met een filter te convolueren:

    1. Kies een filter `K` van $k$ bij $k$.
    2. Maak de feature map, allemaal nullen, van $H - k + 1$ bij $W - k + 1$: `out = np.zeros((rows, cols))`.
    3. Voor elke rij `r` en kolom `c` van `out`: `out[r, c] = (img[r:r + k, c:c + k] * K).sum()`.
    4. Lees de feature map: grote positieve getallen waar de afbeelding op het filter lijkt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een afbeelding van 6 bij 6 met links 0 en rechts 1, met het filter voor verticale randen; daarna dezelfde stappen op de 5 uit de figuur, die `C.digit(0)` geeft als array van 28 bij 28. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_convolve():
    import numpy as np
    import matplotlib.pyplot as plt

    _img = np.zeros((6, 6))
    _img[:, 3:] = 1   # links 0, rechts 1
    # 1. een filter van 3 bij 3 voor een verticale rand: links laag, rechts hoog
    _K = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    _k = len(_K)
    # 2. de feature map, allemaal nullen: 6 - 3 + 1 = 4 bij 4
    _out = np.zeros((len(_img) - _k + 1, len(_img[0]) - _k + 1))
    # 3. op elke plek: het stuk keer het filter, opgeteld
    for _r in range(len(_out)):
        for _c in range(len(_out[0])):
            _out[_r, _c] = (_img[_r:_r + _k, _c:_c + _k] * _K).sum()
    print(_out)

    # dezelfde stappen op een handgeschreven cijfer
    _digit = C.digit(0)
    _out = np.zeros((len(_digit) - _k + 1, len(_digit[0]) - _k + 1))
    for _r in range(len(_out)):
        for _c in range(len(_out[0])):
            _out[_r, _c] = (_digit[_r:_r + _k, _c:_c + _k] * _K).sum()
    # 4. lees de feature map: waar is hij groot?
    print(_out.shape, _out.max(), _out.min())
    plt.imshow(_out, cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de afbeelding van 6 bij 6 is de feature map 0 in zijn eerste en laatste kolom, waar het filter alleen nullen of alleen enen ziet, en 3 in de twee middelste kolommen, waar de rand eronder loopt: rechts drie enen keer +1, links drie nullen keer −1. Op het cijfer is de feature map 26 bij 26. Zijn grootste getallen liggen langs de linkerkant van de strepen en zijn meest negatieve langs de rechterkant; met `cmap="gray"` zijn dat de witte en de zwarte lijnen in het beeld.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel weights?
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights",
        question="Een filter van 5 bij 5 schuift over een afbeelding van 28 bij 28 en maakt een feature map van 24 bij 24. Hoeveel weights gebruikt het voor de hele feature map?",
        options={
            "per-place": "14.400: 25 weights op elk van de 576 plekken",
            "right": "25",
            "per-pixel": "784: één per pixel",
        },
        correct="right",
        why={
            "per-place": "Het filter neemt dezelfde 25 weights mee naar elke plek. Zie *Weight sharing* in *Het idee*.",
            "per-pixel": "Zoveel heeft één neuron van een hidden layer. Zie *Weight sharing* in *Het idee*.",
            "right": "5 · 5 = 25 weights, dezelfde op alle 576 plekken: weight sharing. Een neuron van een hidden layer heeft 784 weights voor één getal."
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_convolve():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een convolutie voor elk filter

        `convolve(img, K)` convolueert een afbeelding met een filter van elke grootte $k$ bij $k$, en geeft de feature map terug. Vervang de drie `...`: op hoeveel plekken het filter past, omlaag, `rows`, en opzij, `cols`, en voor elke plek de som van het stuk afbeelding keer het filter. De check roept `convolve([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], [[1, 0], [0, -1]])` aan, een filter van 2 bij 2, en dat moet `[[0, -1, -1], [-1, 1, 3], [2, 0, -2]]` geven. Hij probeert ook filters van 1 bij 1 en 3 bij 3, en afbeeldingen die niet vierkant zijn.
        """),
        L.task("Laat `convolve` een filter van elke grootte over de afbeelding schuiven, en de feature map teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_convolve():
    box_convolve = L.code_box("convolve", lines=14)
    box_convolve
    return (box_convolve,)


@app.cell(hide_code=True)
def check_convolve(box_convolve):
    result_convolve = L.check("convolve", box_convolve.value)
    result_convolve
    return (result_convolve,)


@app.cell(hide_code=True)
def hints_convolve():
    L.hints(
        "convolve",
        [
            "Stap 2 en 3 van het recept, met `k` in plaats van 3.",
            "Een filter van `k` past omlaag op zoveel plekken als de afbeelding rijen heeft, min `k`, plus 1; opzij hetzelfde met de lengte van een rij. Neem op elke plek het stuk van `k` rijen en `k` kolommen dat daar begint.",
            "`rows = len(img) - k + ___`, `cols = len(img[0]) - ___ + 1`, en `out[r, c] = (img[r:r + k, c:c + ___] * K).sum()`.",
        ],
        why="Met een filter van 2 bij 2 op een afbeelding van 4 bij 4 past het filter op 3 bij 3 plekken. Linksboven is het stuk `[[1, 2], [0, 1]]`, en keer het filter `[[1, 0], [0, -1]]` geeft dat 1 · 1 + 1 · (−1) = 0. De andere plekken gaan net zo.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_filter():
    mo.md(r"""
    ### Oefening 3. Welk filter?
    """)
    return


@app.cell(hide_code=True)
def choice_filter_cell():
    choice_filter = L.choice(
        "which-filter",
        question="Je zoekt horizontale randen met papier erboven en inkt eronder: een pixel van 0 boven, 1 onder. Welk filter geeft daar een groot positief getal?",
        options={
            "vertical": "`[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]`",
            "flipped": "`[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]`",
            "right": "`[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]`",
            "flat": "`[[1, 1, 1], [1, 1, 1], [1, 1, 1]]`",
        },
        correct="right",
        why={
            "vertical": "Dit filter vergelijkt links met rechts, niet boven met onder. Zie het filter voor randen in *Het idee*.",
            "flipped": "Hier komt de inkt onder bij de −1'en, dus dit filter geeft daar een groot negatief getal. Het is groot waar inkt boven zit en papier onder. Zie het filter voor randen in *Het idee*.",
            "flat": "Dit filter telt alle negen pixels op: het is groot waar veel inkt is, rand of niet. Zie *Het idee*.",
            "right": "De bovenste rij weegt de pixels erboven met −1, de onderste rij de pixels eronder met +1: papier boven en inkt onder geven drie nullen keer −1 plus drie enen keer +1, en dat is 3; een egaal vlak geeft 0.",
        },
    )
    choice_filter
    return (choice_filter,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een convolutie werkt op afbeeldingen als de roosters van getallen uit [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl), met de slices uit [Delen van een array kiezen](/course/open/np-indexing?lang=nl). Waar een hidden layer uit [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl) elk neuron één weight per pixel geeft, deelt een filter een paar weights over de hele afbeelding, en kijkt het naar één klein stuk tegelijk. Een convolutional neural network, een CNN, stapelt veel filters en leert hun weights. In de volgende les, *Recurrentie: een netwerk met geheugen*, werkt hetzelfde idee van weights hergebruiken langs een zin in plaats van over een afbeelding.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_filter, choice_weights, result_convolve):
    L.wrap_up(choice_weights, result_convolve, choice_filter)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
