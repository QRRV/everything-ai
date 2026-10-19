# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rnn-memory-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rnn-memory-idea", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl) en [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`W` heeft shape (3, 2), `x` heeft shape (2,), en `b` heeft shape (3,). Welke shape heeft `W @ x + b`?",
        options={
            "inputs": "(2,): één getal per input",
            "matrix": "(3, 2), net als `W`",
            "right": "(3,): één getal per rij van `W`",
        },
        correct="right",
        why={
            "inputs": "`W @ x` neemt het inproduct van elke rij van `W` met `x`. Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "matrix": "Een matrix keer een vector geeft een vector: één inproduct per rij. Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "right": "(3, 2) @ (2,) -> (3,): één inproduct voor elk van de 3 rijen van `W`, en `b` telt bij elk één getal op.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="Geven `[ord(ch) for ch in \"ab\"]` en `[ord(ch) for ch in \"ba\"]` dezelfde lijst?",
        options={
            "right": "Nee: dezelfde getallen, in een andere volgorde",
            "same": "Ja: beide teksten hebben een a en een b",
            "other": "Nee: de b krijgt een ander getal als hij voorop staat",
        },
        correct="right",
        why={
            "same": "Een lijst houdt zijn volgorde: `[97, 98]` is niet `[98, 97]`. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "other": "`ord` geeft elk teken zijn eigen getal, waar het ook staat. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "right": "`[97, 98]` tegen `[98, 97]`: een tekst is een rij, en de volgorde hoort erbij.",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een layer zoals in [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl) neemt een vast aantal inputs: zijn `W` heeft één kolom per input. Met een tekst gaat dat niet. De ene zin heeft 12 tekens, de volgende 80, en wat een woord betekent, hangt af van wat ervoor kwam: "niet goed" betekent iets anders dan "goed". Een netwerk voor tekst moet een rij van elke lengte lezen, op volgorde, en onthouden wat het tot nu toe gelezen heeft.

    **Een voorbeeld uit het dagelijks leven.** Je volgt een routebeschrijving door een stad: links, twee straten rechtdoor, rechts, enzovoort. Je onthoudt alleen waar je bent en welke kant je op kijkt, en past dat bij elke nieuwe aanwijzing aan. Tien aanwijzingen of honderd, je houdt dezelfde paar dingen in je hoofd, en dezelfde aanwijzingen in een andere volgorde brengen je ergens anders.
    """)
    return


@app.cell(hide_code=True)
def predict_trace():
    guess_trace = L.guess(
        "trace",
        question="Een netwerk leest een tekst teken voor teken en houdt een korte samenvatting bij van wat het gelezen heeft: twee getallen, bij elk teken opnieuw gemaakt uit het nieuwe teken en de samenvatting ervoor. Het leest één \"a\" en dan twintig keer \"b\". Wat denk je dat er aan het eind nog van de \"a\" in de samenvatting zit?",
        options={
            "all": "Alles: wat in de samenvatting zit, blijft erin",
            "more": "Meer dan aan het begin: hij is langer meegenomen",
            "right": "Bijna niets",
        },
        reveal=f"**Bijna niets.** Met het netwerk uit het uitgewerkte voorbeeld hieronder halveert het spoor van de a bij elke stap: {L.number(C.trace(0), 2)}, {L.number(C.trace(1), 2)}, {L.number(C.trace(2), 2)}, {L.number(C.trace(3), 2)}, … Na twintig b's is er minder dan een miljoenste over. Elke stap maakt een nieuwe samenvatting uit het nieuwe teken en maar een deel van de oude. Een getraind netwerk kan weights leren die dingen langer bewaren, maar een samenvatting van een paar getallen kan niet alles bevatten.",
    )
    guess_trace
    return (guess_trace,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **recurrent neural network**, een RNN, leest een rij stap voor stap, zoals een tekst teken voor teken. Het neemt een **hidden state** $\mathbf{h}$ mee van stap naar stap: een vector van getallen die samenvat wat het tot nu toe gelezen heeft. Bij stap $t$ leest het de input $\mathbf{x}_t$, en maakt het de nieuwe hidden state uit die input en de hidden state van de stap ervoor:
        """),
        L.formula(r"[[hidden:\mathbf{h}_t]] = \tanh([[param:\mathbf{W}_x]] [[data:\mathbf{x}_t]] + [[param:\mathbf{W}_h]] [[hidden:\mathbf{h}_{t-1}]] + [[param:\mathbf{b}]])"),
        mo.md(r"""
        - De hidden state begint bij nullen: $\mathbf{h}_0 = \mathbf{0}$.
        - $\mathbf{W}_x$ weegt de nieuwe input en $\mathbf{W}_h$ de hidden state ervoor, net als de `W` van een layer: één rij voor elk getal van de hidden state. Met een hidden state van 2 getallen en inputs van 2 getallen zijn ze allebei (2, 2).
        - Bij elke stap dezelfde $\mathbf{W}_x$, $\mathbf{W}_h$ en $\mathbf{b}$. Zo leest het netwerk een rij van elke lengte met dezelfde weights, en blijft de hidden state bij elke stap even groot.
        - **tanh** perst elk getal tussen −1 en 1, met 0 in het midden, zoals de sigmoid uit [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) tussen 0 en 1 perst. Zo kan de hidden state niet stap na stap groeien.

        De laatste hidden state vat de hele rij samen. Een output layer kan hem lezen, bijvoorbeeld om te zeggen of een recensie positief of negatief is. In een netwerk dat leert, vindt het trainen $\mathbf{W}_x$, $\mathbf{W}_h$ en $\mathbf{b}$; in deze les stel je ze met de hand in, zodat je ziet wat de hidden state doet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **recurrent neural network** | RNN, er-en-en | een netwerk dat een rij stap voor stap leest, en een hidden state van elke stap aan de volgende doorgeeft, met dezelfde weights bij elke stap | |
        | **hidden state** | | de vector die een recurrent neural network van stap naar stap meeneemt: een samenvatting van wat het tot nu toe gelezen heeft | `h` |
        | $\mathbf{x}_t$ | x t | de input bij stap $t$ | `x` |
        | $\mathbf{h}_t$, $\mathbf{h}_{t-1}$ | h t, h t min 1 | de hidden state na stap $t$, en na de stap ervoor | `h` |
        | $\mathbf{W}_x$, $\mathbf{W}_h$ | W x, W h | de weights voor de input, en voor de hidden state ervoor | `W_x`, `W_h` |
        | **tanh** | tanh | perst elk getal tussen −1 en 1, met 0 in het midden | `np.tanh(z)` |
        """),
        L.watch_out("*State* betekent in AI meerdere dingen. Hier is het de **hidden state**, de vector die een recurrent neural network van stap naar stap meeneemt: zeg altijd voluit hidden state. In reinforcement learning is een *state* de situatie waarin een model zijn volgende actie kiest, en dat is iets anders."),
        mo.md(r"""
        De figuur: het netwerk uit het uitgewerkte voorbeeld leest "abbb". Na elke stap laten de staven de twee getallen van de hidden state zien. Het eerste onthoudt de a's, het tweede de b's.
        """),
        mo.Html(C.picture("abbb", {"title": "Eén netwerk, vier stappen", "step": "stap",
                                   "alt": "Vijf vakken van links naar rechts, h0 tot en met h4, verbonden door pijlen met W h; onder vak 1 tot en met 4 de tekens a, b, b, b, elk met een pijl W x omhoog. In elk vak twee staven, a en b. h0: allebei 0. h1: a 0,76, b 0. h2: a 0,36, b 0,76. h3: a 0,18, b 0,88. h4: a 0,09, b 0,89. De staaf van de a wordt bij elke stap korter."}, L.lang)),
        mo.md(r"""
        De a komt binnen bij stap 1 en vervaagt daarna: elke stap houdt er maar een deel van over. De b's duwen het tweede getal omhoog, maar tanh houdt het onder 1.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een tekst te lezen met een recurrent neural network:

    1. Kies een vector voor elk teken, de weights `W_x` en `W_h`, en de bias terms `b`: bij elke stap dezelfde.
    2. Begin met een hidden state van nullen: `h = np.zeros(size)`.
    3. Voor elk teken, op volgorde: `h = np.tanh(W_x @ x + W_h @ h + b)`, met `x` de vector van dat teken.
    4. Lees de laatste hidden state: die vat de hele tekst samen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het netwerk uit de figuur, met een hidden state van 2 getallen, leest drie teksten van a's en b's. De dictionary `vec` geeft elk teken zijn vector, zoals in [Dictionaries](/course/open/py-dicts?lang=nl). Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_rnn():
    import numpy as np

    # 1. een vector voor elk teken, de weights en de bias terms: bij elke stap dezelfde
    _vec = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    _W_x = np.array([[1.0, 0.0], [0.0, 1.0]])
    _W_h = np.array([[0.5, 0.0], [0.0, 0.5]])
    _b = np.array([0.0, 0.0])

    for _text in ["ab", "ba", "abbbb"]:
        # 2. de hidden state begint bij nullen
        _h = np.zeros(2)
        # 3. één stap voor elk teken, op volgorde
        for _ch in _text:
            _h = np.tanh(_W_x @ _vec[_ch] + _W_h @ _h + _b)   # (2, 2) @ (2,) + (2, 2) @ (2,) + (2,) -> (2,)
        # 4. de laatste hidden state vat de tekst samen
        print(_text, _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    _ab, _ba, _long = C.states("ab")[-1], C.states("ba")[-1], C.states("abbbb")[-1]
    mo.md(f"""
    `W_x` zet een nieuwe a in het eerste getal en een nieuwe b in het tweede, en `W_h` houdt de helft over van wat er al stond. "ab" en "ba" hebben dezelfde tekens, maar hun hidden states verschillen: ({L.number(_ab[0], 2)}; {L.number(_ab[1], 2)}) tegen ({L.number(_ba[0], 2)}; {L.number(_ba[1], 2)}). Het laatste teken telt het zwaarst, dus de volgorde zit in de hidden state. In "abbbb" halveert de a bij elke stap, tot {L.number(_long[0], 2)}. De vier b's duwen het tweede getal niet boven 1 uit: het is {L.number(_long[1], 2)}, omdat tanh het onder 1 houdt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_size():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoe groot is het geheugen?
    """)
    return


@app.cell(hide_code=True)
def choice_size_cell():
    choice_size = L.choice(
        "size",
        question="Een recurrent neural network met een hidden state van 8 getallen leest een tekst van 1.000 tekens. Hoeveel getallen bevat zijn hidden state na het laatste teken?",
        options={
            "per-char": "8.000: 8 voor elk teken",
            "chars": "1.000: één voor elk teken",
            "right": "8, net als na het eerste teken",
        },
        correct="right",
        why={
            "per-char": "Elke stap maakt een nieuwe hidden state van dezelfde grootte, en die vervangt de oude. Zie *Het idee*.",
            "chars": "De hidden state heeft zoveel getallen als `W_h` rijen heeft, hoe lang de tekst ook is. Zie *Het idee*.",
            "right": "Elke stap vervangt de hidden state door een nieuwe van dezelfde grootte: 8 getallen moeten 1.000 tekens samenvatten.",
        },
    )
    choice_size
    return (choice_size,)


@app.cell(hide_code=True)
def text_exercise_rnn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een recurrent neural network voor elke rij

        `rnn(xs, W_x, W_h, b)` leest de inputs in `xs` op volgorde, één vector per stap, en geeft de laatste hidden state terug. `W_x` heeft shape (hidden, inputs), `W_h` shape (hidden, hidden), en `b`, de bias terms, shape (hidden,): de hidden state heeft zoveel getallen als `b`. De eerste regel maakt er arrays van. Vervang de twee `...`: de hidden state om mee te beginnen, en één stap van het netwerk. De check roept `rnn([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]], [0, 0])` aan, de "ab" uit het uitgewerkte voorbeeld, en dat moet ongeveer `[0.3634, 0.7616]` geven. Hij probeert ook hidden states van andere groottes, en langere rijen.
        """),
        L.task("Laat `rnn` beginnen met een hidden state van nullen en één stap doen voor elke input."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rnn():
    box_rnn = L.code_box("rnn", lines=10)
    box_rnn
    return (box_rnn,)


@app.cell(hide_code=True)
def check_rnn(box_rnn):
    result_rnn = L.check("rnn", box_rnn.value)
    result_rnn
    return (result_rnn,)


@app.cell(hide_code=True)
def hints_rnn():
    L.hints(
        "rnn",
        [
            "Stap 2 en 3 van het recept, met de namen uit de functie.",
            "`h` begint met één nul voor elk getal van de hidden state: zoveel heeft `b` er. Elke stap gebruikt de input `x`, de hidden state `h` van de stap ervoor, allebei de weight-matrices en de bias terms.",
            "`h = np.zeros(len(___))` voor de loop, en `h = np.tanh(W_x @ x + W_h @ ___ + b)` erin.",
        ],
        why="Met het voorbeeld leest stap 1 (1, 0): `W_x @ x` is (1, 0), en `W_h @ h` en `b` zijn nullen, dus tanh van (1, 0) geeft (0,7616; 0). Stap 2 leest (0, 1), en `W_h @ h` telt de helft van de hidden state ervoor erbij: tanh van (0,3808; 1) geeft (0,3634; 0,7616).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_memory():
    mo.md(r"""
    ### Oefening 3. Geen geheugen
    """)
    return


@app.cell(hide_code=True)
def choice_memory_cell():
    choice_memory = L.choice(
        "no-memory",
        question="In een recurrent neural network zet je elke weight van `W_h` op 0, en houd je `W_x` en `b`. Het leest een tekst. Waar hangt de laatste hidden state nu van af?",
        options={
            "right": "Alleen van het laatste teken",
            "all": "Van alle tekens, elk evenveel",
            "nothing": "Van niets: hij blijft vol nullen",
        },
        correct="right",
        why={
            "all": "Via `W_h @ h` komt de hidden state van de stap ervoor in een stap. Met `W_h` vol nullen is dat deel nul. Zie de formule in *Het idee*.",
            "nothing": "`W_x @ x + b` brengt bij elke stap nog steeds het nieuwe teken binnen. Zie de formule in *Het idee*.",
            "right": "Elke stap is dan `np.tanh(W_x @ x + b)`: de hidden state ervoor wordt met 0 vermenigvuldigd. Elke stap vergeet alles, en de laatste hidden state komt alleen uit het laatste teken.",
        },
    )
    choice_memory
    return (choice_memory,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een recurrent neural network doet bij elke stap wat een layer uit [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl) doet: weights keer inputs, plus bias terms, door een activation function. Nieuw is de input die het van zichzelf krijgt: de hidden state van de stap ervoor. Zo leest het een tekst uit [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl) op volgorde, van elke lengte. Zijn geheugen heeft een vaste grootte en vervaagt, dus raakt het begin van een lange tekst kwijt. In de volgende les, *Attention: elk woord kijkt naar de andere*, kijkt elke positie rechtstreeks naar elke andere, hoe ver terug die ook staat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_memory, choice_size, result_rnn):
    L.wrap_up(choice_size, result_rnn, choice_memory)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
