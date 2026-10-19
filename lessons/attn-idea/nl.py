# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-idea", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl) en [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`X` heeft shape (4, 2): 4 woorden met elk 2 getallen, één woord per rij. Welke shape heeft `X @ X.T`?",
        options={
            "small": "(2, 2): één getal voor elk paar van de 2 getallen",
            "right": "(4, 4): één getal voor elk paar woorden",
            "same": "(4, 2): dezelfde shape als `X`",
        },
        correct="right",
        why={
            "small": "Dat zou `X.T @ X` zijn: (2, 4) @ (4, 2). Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "same": "(4, 2) @ (2, 4) geeft (4, 4): de binnenste getallen, 2 en 2, zijn gelijk, en de buitenste blijven over. Zie [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl).",
            "right": "(4, 2) @ (2, 4) geeft (4, 4): rij $i$, kolom $j$ is het inproduct van woord $i$ met woord $j$. Vandaag zeggen zulke inproducten hoe goed twee woorden bij elkaar passen.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_equal():
    quick_equal = L.choice(
        "quick-equal",
        question="Twee klassen krijgen dezelfde logit, en een derde een veel lagere. Wat geeft de softmax de eerste twee?",
        options={
            "right": "Elk dezelfde kans, iets onder de helft",
            "one": "Allebei 1: ze zijn allebei de grootste",
            "logit": "Elk zijn eigen logit terug",
        },
        correct="right",
        why={
            "one": "De kansen van de softmax tellen op tot 1, dus twee ervan kunnen niet allebei 1 zijn. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "logit": "De softmax maakt van logits kansen tussen 0 en 1 die optellen tot 1. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "Gelijke logits geven gelijke kansen, en de lage derde neemt maar een beetje, dus elk krijgt iets onder de helft. Vandaag maakt de softmax op precies deze manier weights van hoe goed woorden passen.",
        },
    )
    quick_equal
    return (quick_equal,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een embedding geeft een woord één lijst getallen, wat de zin ook is. Maar "bank" in "de houten bank" betekent iets anders dan in "de bank betaalt". Een model dat tekst leest, heeft de betekenis nodig die een woord *hier* heeft, in deze zin. Dus moet elk woord naar de andere woorden kunnen kijken, en het meest opnemen van de woorden die er het best bij passen.

    **Een voorbeeld uit het dagelijks leven.** Als je een woord leest dat twee dingen kan betekenen, kijk je naar de woorden eromheen om te weten welke, en je kijkt het meest naar de woorden die helpen: "houten" zegt je meer over "bank" dan "de".
    """)
    return


@app.cell(hide_code=True)
def predict_bank():
    guess_bank = L.guess(
        "bank-looks",
        question="In \"de houten bank\" neemt \"bank\" een gewogen gemiddelde van de embeddings van alle drie de woorden, waarbij de woorden waarvan de embedding het best bij de zijne past, volgens het inproduct, het zwaarst meetellen. Met de verzonnen embeddings van deze les, de (0,5; 0,5), houten (0; 2) en bank (1; 1), welk woord telt het zwaarst mee?",
        options={
            "river": "\"houten\", veruit: meer dan \"de\" of \"bank\"",
            "equal": "Alle drie ongeveer even zwaar",
            "right": "\"bank\" zelf, net zo zwaar als \"houten\"",
        },
        reveal="**\"bank\" zelf, net zo zwaar als \"houten\".** Het inproduct van \"bank\", (1; 1), met \"houten\", (0; 2), is 2, en met zichzelf ook 2; met \"de\" is het 1. Dus krijgen \"houten\" en \"bank\" dezelfde weight, elk 0,42, en \"de\" 0,16: het uitgewerkte voorbeeld hieronder rekent het uit. Een woord past altijd goed bij zijn eigen embedding. Toch schuift de nieuwe vector van \"bank\" op naar hout, omdat \"houten\" hem die kant op trekt. In een latere les geeft echte attention elk woord aparte vectoren om mee te zoeken en om gevonden te worden.",
    )
    guess_bank
    return (guess_bank,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Met **attention** neemt elk woord, op elke plek in de zin, een gewogen gemiddelde van de embeddings van alle woorden, zijn eigen embedding ook. De weights hangen af van hoe goed twee woorden bij elkaar passen, gemeten met het inproduct van hun embeddings. De softmax maakt van de inproducten weights die positief zijn en optellen tot 1: de attention weights. Voor woord $i$, met embeddings $\mathbf{x}_j$:
        """),
        L.formula(r"\begin{aligned} s_{j} &= [[hidden:\mathbf{x}_i]] \cdot [[hidden:\mathbf{x}_j]] \\ [[hidden:a_{j}]] &= \frac{e^{s_{j}}}{\sum_k e^{s_{k}}} \\ [[hidden:\mathbf{h}_i]] &= \sum_j [[hidden:a_{j}]] \, [[hidden:\mathbf{x}_j]] \end{aligned}"),
        mo.md(r"""
        $\mathbf{h}_i$ is de nieuwe vector van woord $i$. Elk woord doet dit tegelijk, en zijn nieuwe vector gaat door naar de volgende layer. Attention is het hart van de **transformer**, het soort netwerk achter bijna elk groot language model: layer na layer attention, elk gevolgd door een klein MLP. Echte attention heeft nog een paar onderdelen meer, die latere lessen één voor één bouwen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **attention** | | elk woord neemt een gewogen gemiddelde van de embeddings van alle woorden; de woorden die er het best bij passen, tellen het zwaarst mee | `weights @ X` |
        | $\mathbf{h}_i$ | h i | de nieuwe vector van woord $i$ na attention | `weights @ X` |
        | **transformer** | | een netwerk van layers attention, elk gevolgd door een klein MLP | |
        """),
        L.watch_out("In gewone taal is aandacht iets wat je geeft. Hier is attention een berekening, een gewogen gemiddelde; het model merkt niets op. De attention weights worden ook niet geleerd, zoals de weights van een layer: de softmax rekent ze voor elke zin opnieuw uit. En de embeddings in deze les heeft de course verzonnen, met 2 getallen: hoeveel een woord over geld gaat, en hoeveel over hout. Een echt model leert zijn embeddings, met honderden getallen."),
        mo.md(r"""
        De figuur: de verzonnen embeddings van "houten", "bank" en "betaalt" als pijlen, en de nieuwe vector van "bank" na attention in "de houten bank" en in "de bank betaalt". Dezelfde embedding van "bank" draait in de ene zin naar hout, en in de andere naar geld.
        """),
        mo.Html(C.bank_picture({"title": "Bank, voor en na attention", "x": "geld", "y": "hout",
                                "river": "houten", "bank": "bank", "account": "betaalt", "in_river": "naast houten", "in_account": "naast betaalt",
                                "alt": "Pijlen vanuit de oorsprong: houten wijst recht omhoog, betaalt recht naar rechts, bank schuin daartussen. Na attention wijst bank naast houten meer omhoog, en bank naast betaalt meer naar rechts."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De nieuwe vector van één woord, na attention:

    1. De scores: het inproduct van de embedding van het woord met de embedding van elk woord in de zin, zijn eigen ook.
    2. De attention weights: de softmax van de scores.
    3. De nieuwe vector: elke embedding keer zijn weight, alles opgeteld.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "bank" in "de houten bank", met de verzonnen embeddings van de figuur, één woord per rij van `_X`. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_attention():
    import numpy as np

    _words = ["de", "houten", "bank"]
    _X = np.array([[0.5, 0.5], [0.0, 2.0], [1.0, 1.0]])
    _i = 2
    # 1. de scores: het inproduct van de embedding van woord _i met elke rij van _X
    _scores = _X @ _X[_i]
    print(_words[_i], _scores)
    # 2. de attention weights: de softmax van de scores
    _weights = np.exp(_scores) / np.sum(np.exp(_scores))
    print(np.round(_weights, 3), np.sum(_weights))
    # 3. de nieuwe vector: elke rij van _X keer zijn weight, opgeteld
    _h = _weights @ _X
    print(np.round(_h, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_X @ _X[_i]` geeft het inproduct van "bank" met elke rij: 1 met "de", 2 met "houten" en 2 met zichzelf. De softmax maakt daar de weights 0,155, 0,422 en 0,422 van, die optellen tot 1. `_weights @ _X` vermenigvuldigt elke rij van `_X` met zijn weight en telt ze op. De shapes zijn (3,) @ (3, 2): de drieën passen, en (2,) blijft over, één vector van 2 getallen. Met afgeronde weights: $0{,}155 \cdot (0{,}5;\ 0{,}5) + 0{,}422 \cdot (0;\ 2) + 0{,}422 \cdot (1;\ 1) \approx (0{,}5;\ 1{,}345)$. De embedding van "bank" was $(1;\ 1)$; zijn nieuwe vector is opgeschoven naar hout.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Weights die optellen tot 1
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights-sum",
        question="De attention weights van één woord tellen altijd op tot 1. Wat levert dat de nieuwe vector op?",
        options={
            "right": "Hij is een gemiddelde: niet groter dan de grootste embedding, hoe lang de zin ook is",
            "largest": "Hij is de embedding van het woord met de grootste weight: het best passende woord krijgt alles",
            "zero": "Zijn getallen tellen ook op tot 1: net als de weights",
        },
        correct="right",
        why={
            "largest": "Alle woorden tellen mee, elk naar zijn weight; de grootste weight telt het meest, niet als enige.",
            "zero": "De weights tellen op tot 1, niet de getallen van de nieuwe vector: in het voorbeeld tellen die op tot 1,845.",
            "right": "Weights die optellen tot 1 maken een gewogen gemiddelde. Zonder de softmax zou een zin van 100 woorden 100 embeddings optellen, en zou de nieuwe vector groeien met de lengte van de zin.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_attend():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De nieuwe vector van elk woord

        Laat `attend(X, i)` de nieuwe vector van woord `i` na attention geven, volgens het recept. `X` is een NumPy-array met één embedding per rij, één rij per woord; `i` is het nummer van de rij. Vervang de drie `...`, één per stap, zoals in het uitgewerkte voorbeeld. De check roept `attend(np.array([[1, 0], [0, 1], [1, 1]]), 0)` aan, wat `np.array([0.8446, 0.5777])` moet geven, en probeert andere woorden en andere zinnen, ook met 3 getallen per woord, en een zin van één woord.
        """),
        L.task("Laat `attend` de nieuwe vector van woord `i` na attention geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attend():
    box_attend = L.code_box("attend", lines=10)
    box_attend
    return (box_attend,)


@app.cell(hide_code=True)
def check_attend(box_attend):
    result_attend = L.check("attend", box_attend.value)
    result_attend
    return (result_attend,)


@app.cell(hide_code=True)
def hints_attend():
    L.hints(
        "attend",
        [
            "Dezelfde drie stappen als in het uitgewerkte voorbeeld, met `X` en `i` in plaats van `_X` en `_i`.",
            "Stap 1 is één `@`, van `X` en rij `i` van `X`. Stap 2 is de softmax van die scores, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "`scores = X @ X[i]`, `weights = np.exp(scores) / np.sum(___)`, en `return weights @ ___`.",
        ],
        why="`scores = X @ X[i]`; `weights = np.exp(scores) / np.sum(np.exp(scores))`; `return weights @ X`. In het voorbeeld zijn de scores 1, 0 en 1, de weights 0,4223, 0,1554 en 0,4223, en de nieuwe vector is $0{,}4223 \\cdot (1;\\ 0) + 0{,}1554 \\cdot (0;\\ 1) + 0{,}4223 \\cdot (1;\\ 1) = (0{,}8446;\\ 0{,}5777)$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_transformer():
    mo.md(r"""
    ### Oefening 3. Drie bouwstenen
    """)
    return


@app.cell(hide_code=True)
def choice_transformer_cell():
    choice_transformer = L.choice(
        "transformer",
        question="Welke hiervan is een transformer?",
        options={
            "rnn": "Een netwerk dat de woorden één voor één leest en een geheugen meeneemt van elk woord naar het volgende",
            "cnn": "Een netwerk dat één klein filter langs de zin schuift, een paar woorden tegelijk",
            "right": "Een netwerk waarin elk woord tegelijk naar alle andere woorden kijkt, layer na layer",
        },
        correct="right",
        why={
            "rnn": "Dat is een recurrent neural network, uit [Recurrentie: een netwerk met geheugen](/course/open/rnn-memory-idea?lang=nl): dat moet alles woord voor woord doorgeven.",
            "cnn": "Dat is een convolutie, uit [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl), op een zin in plaats van een afbeelding: elke plek ziet alleen zijn buren.",
            "right": "In een transformer neemt elk woord in één stap elk ander woord op, met attention. Woorden die ver uit elkaar staan, komen elkaar net zo makkelijk tegen als buren, en alle plekken worden tegelijk uitgerekend.",
        },
    )
    choice_transformer
    return (choice_transformer,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De embeddings uit [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl) geven elk woord één vaste lijst getallen; attention geeft het een nieuwe die van de zin afhangt. De scores zijn inproducten, zoals in de matrixproducten van [De forward pass in NumPy, met shapes](/course/open/nn-forward-numpy?lang=nl), en de weights komen uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl). Attention begon in 2014 als toevoeging aan recurrent neural networks die vertalen, zodat ze naar elk woord van de zin terug konden kijken; de transformer van 2017 hield alleen de attention over. In het gevorderde deel bouw je echte attention stap voor stap, en daarna een eigen transformer. De volgende les, *Leren van rewards*, gaat over een andere manier van leren: van rewards in plaats van labels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_attend, choice_transformer):
    L.wrap_up(choice_weights, result_attend, choice_transformer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
