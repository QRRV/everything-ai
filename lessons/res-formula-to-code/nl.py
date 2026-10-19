# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-formula-to-code. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-formula-to-code", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl) en [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pass():
    quick_pass = L.choice(
        "quick-pass",
        question="In welke van de drie rondes werk je elke formule door, tot je de methode zelf zou kunnen bouwen?",
        options={
            "first": "De eerste",
            "second": "De tweede",
            "right": "De derde",
        },
        correct="right",
        why={
            "first": "De eerste ronde is vijf tot tien minuten voor de claim: titel, abstract, inleiding, kopjes en conclusie. Zie [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl).",
            "second": "De tweede ronde volgt de methode en de figuren, en mag een formule overslaan die je nog niet volgt. Zie [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl).",
            "right": "Over die ronde gaat deze les: van een formule naar code die je kunt vertrouwen.",
        },
    )
    quick_pass
    return (quick_pass,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="$\\mathbf{X}$ heeft shape (B, 8) en $\\mathbf{W}$ shape (8, 3). Welke shape heeft $\\mathbf{X}\\mathbf{W}$?",
        options={
            "right": "(B, 3)",
            "w": "(8, 3)",
            "x": "(B, 8)",
        },
        correct="right",
        why={
            "w": "De 8 is de binnenste maat, waarover het product optelt: die valt weg. Zie [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl).",
            "x": "De 8 kolommen van X ontmoeten de 8 rijen van W en vallen weg; de 3 kolommen van W blijven. Zie [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl).",
            "right": "De buitenste maten blijven. Deze les geeft elk teken van een formule zo'n shape.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een paper zegt in formules wat er gedaan is, en een formule heb je pas iets aan als hij draait. Daartussen liggen drie valkuilen: notatie die je niet kent, tekens waarvan de paper de shape nooit noemt, en code die draait maar net iets anders uitrekent. De derde is het moeilijkst te vangen: die geeft helemaal geen foutmelding. Een vaste aanpak vangt alle drie: ontcijfer elk teken, geef het een tensor met een shape, en controleer de code op een voorbeeld dat klein genoeg is om met de hand uit te rekenen, en tegen een implementatie die je vertrouwt.
    """)
    return


@app.cell(hide_code=True)
def predict_target():
    guess_target = L.guess(
        "right-token",
        question="De transformer-paper traint met label smoothing van $\\varepsilon = 0{,}1$, over een vocabulary van ongeveer 37.000 tokens, en verwijst daarvoor naar Szegedy en anderen (2016). Welke kans geeft het target volgens hun definitie aan het juiste token?",
        options={
            "exact": "Precies 0,9",
            "right": "Net iets meer dan 0,9",
            "small": "0,1",
        },
        reveal="**Net iets meer dan 0,9.** Hun formule, hieronder, geeft het juiste token $1 - \\varepsilon + \\varepsilon / K = 0{,}9 + 0{,}1 / 37{.}000 \\approx 0{,}900003$, en elk ander token $0{,}1 / 37{.}000$. De eigen code van de auteurs (tensor2tensor) koos een variant: precies 0,9 voor het juiste token, en de 0,1 verdeeld over alleen de andere tokens. In woorden is het allebei \"ongeveer 0,9\"; in code moet je er een kiezen, en alleen de formule of de code van de paper zegt welke.",
    )
    guess_target
    return (guess_target,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: ontcijfer de notatie.** Label smoothing komt van Szegedy en anderen (2016). In plaats van het one-hot-target trainen zij tegen
        """),
        L.formula(r"q'(k \mid x) = (1 - \varepsilon)\,\delta_{k,y} + \varepsilon\, u(k)"),
        mo.md(r"""
        Lees hem teken voor teken, met de tekst eromheen: $x$ is de input; $k$ is een klasse, van 1 tot $K$ (in code van 0 tot $K - 1$); $y$ is de juiste klasse van $x$; $\delta_{k,y}$, de **Kronecker-delta**, is 1 als $k = y$ en anders 0, dus het one-hot-target als formule; $u(k)$ is een verdeling over de klassen, in de paper de uniforme, $1/K$. En $\varepsilon$ is hoeveel van het target naar $u$ gaat: geen klein getal tegen delen door nul, en geen ruis. Dezelfde letter kan in een andere paper iets anders betekenen; de tekst zegt wat. De loss is de cross-entropy tegen dat target, met $p(k)$ de kansen van het model:
        """),
        L.formula(r"H(q', p) = -\sum_{k=1}^{K} q'(k) \log p(k)"),
        mo.md(r"""
        Het plaatje: het target van één voorbeeld met 5 klassen en $\varepsilon = 0{,}2$, naast het one-hot-target.
        """),
        mo.Html(C.targets_picture({"title": "One-hot of verzacht", "x": "klasse k", "y": "target", "right": "0,84", "rest": "0,04",
                                   "hard": "one-hot", "soft": "verzacht", "lang": L.lang,
                                   "alt": "Staven voor 5 klassen; de juiste klasse is klasse 2. Het one-hot-target is 1 voor klasse 2 en 0 voor de andere. Het verzachte target met epsilon 0,2 is 0,84 voor klasse 2 en 0,04 voor elk van de andere."})),
        mo.md(r"""
        **Stap 2: een tensor en een shape voor elk teken.** De paper schrijft één voorbeeld; code werkt op een batch van $B$ voorbeelden.

        | Teken | Tensor | Shape |
        |---|---|---|
        | $p(k)$ | `p = torch.softmax(z, dim=1)` | (B, K) |
        | $y$ | `y` | (B,) |
        | $\delta_{k,y}$ | `torch.eye(K)[y]` | (B, K) |
        | $u(k) = 1/K$ | `1 / K` | één getal |
        | $q'(k)$ | `q` | (B, K) |
        | $\sum_k$ | `.sum(dim=1)` | (B, K) naar (B,) |

        Eén ding laat de formule open: de som of het gemiddelde over de batch? PyTorch neemt het gemiddelde.

        **Stap 3: een toy-voorbeeld dat je met de hand uitrekent.** Met $K = 2$, $p = (0{,}75;\ 0{,}25)$, $y$ de eerste klasse en $\varepsilon = 0{,}2$: $q' = (0{,}8 + 0{,}1;\ 0{,}1) = (0{,}9;\ 0{,}1)$, en $H = -(0{,}9 \log 0{,}75 + 0{,}1 \log 0{,}25) \approx 0{,}3975$. De logits $z = (\log 3;\ 0)$ geven precies die $p$. Je code moet ook 0,3975 geven.

        **Stap 4: een controle tegen een implementatie die je vertrouwt.** PyTorch heeft deze loss: `nn.CrossEntropyLoss(label_smoothing=0.2)`. Vergelijk op willekeurige inputs met `torch.allclose`, dat waar is als twee tensors op afronding na gelijk zijn. Is er geen implementatie, controleer dan bijzondere gevallen: met $\varepsilon = 0$ moet de loss de gewone cross-entropy zijn.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **Kronecker-delta** $\delta_{k,y}$ | 1 als $k = y$, anders 0; in code `torch.eye(K)[y]` |
        | **label smoothing** | trainen tegen een target dat de juiste klasse $1 - \varepsilon + \varepsilon/K$ geeft en elke andere klasse $\varepsilon/K$ |
        | **toy-voorbeeld** | inputs zo klein dat je de juiste output met de hand kunt uitrekenen |
        | `torch.allclose(a, b)` | waar als twee tensors op afronding na gelijk zijn |
        """),
        L.watch_out("Een formule in een paper is meestal voor één voorbeeld, en laat open wat code moet beslissen: het gemiddelde of de som over een batch, welk grondtal de log heeft, welke tokens meetellen. Zoek de code van de paper, of controleer tegen een implementatie, voordat je je eigen code vertrouwt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo maak je van een formule uit een paper code:

    1. Ontcijfer: schrijf op wat elk teken betekent, uit de tekst rond de formule, en noteer elk teken dat iets anders betekent dan je gewend bent.
    2. Shapes: geef elk teken een tensor met een shape, voeg de batch-as toe, en noteer wat de formule openlaat.
    3. Toy: kies inputs zo klein dat je de output met de hand kunt uitrekenen, en vergelijk.
    4. Getallen: vergelijk met een implementatie die je vertrouwt, of met bijzondere gevallen, met `torch.allclose`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw zijn de Kronecker-delta als `torch.eye`, en de twee controles, met de hand en met `torch.allclose`. De formule is de gewone cross-entropy, $-\sum_k \delta_{k,y} \log p(k)$, op een toy-batch van twee voorbeelden.
    """)
    return


@app.cell
def example_formula():
    import math
    import torch
    import torch.nn as nn

    _z = torch.tensor([[math.log(3), 0.0], [0.0, 0.0]])  # logits, (B, K) = (2, 2)
    _y = torch.tensor([0, 1])  # de juiste klassen, (B,)
    _K = _z.shape[1]
    _p = torch.softmax(_z, dim=1)  # p(k), (B, K)
    _delta = torch.eye(_K)[_y]  # de Kronecker-delta: de one-hot-targets, (B, K)
    _loss = -(_delta * torch.log(_p)).sum(dim=1).mean()  # som over k, dan gemiddelde over de batch
    print(round(_loss.item(), 4))
    print(torch.allclose(_loss, nn.CrossEntropyLoss()(_z, _y)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de hand: het eerste voorbeeld heeft $p = (0{,}75;\ 0{,}25)$ en juiste klasse 0, dus $-\log 0{,}75 \approx 0{,}2877$; het tweede heeft $p = (0{,}5;\ 0{,}5)$, dus $-\log 0{,}5 \approx 0{,}6931$. Hun gemiddelde is 0,4904, zoals de code laat zien. En `torch.allclose` zegt dat de loss van PyTorch dezelfde is: het gemiddelde over de batch was de goede keuze.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_smoothed():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Label smoothing uit de formule

        Schrijf `smoothed_loss(z, y, eps)`: de cross-entropy $H(q', p)$ tegen het verzachte target $q'$ van stap 1, gemiddeld over de batch. `z` bevat de logits, shape (B, K); `y` de juiste klassen, (B,); `eps` is $\varepsilon$. De check roept `smoothed_loss(torch.tensor([[math.log(3), 0.0]]), torch.tensor([0]), 0.2)` aan, het toy-voorbeeld van stap 3, wat 0,3975 moet geven, en probeert meer voorbeelden tegelijk, meer klassen en $\varepsilon = 0$.
        """),
        L.task("Schrijf `smoothed_loss`: het target q' als tensor, dan H(q', p) voor elk voorbeeld, en het gemiddelde daarvan over de batch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_smoothed():
    box_smoothed = L.code_box("smoothed-loss", lines=10)
    box_smoothed
    return (box_smoothed,)


@app.cell(hide_code=True)
def check_smoothed(box_smoothed):
    result_smoothed = L.check("smoothed-loss", box_smoothed.value)
    result_smoothed
    return (result_smoothed,)


@app.cell(hide_code=True)
def hints_smoothed():
    L.hints(
        "smoothed-loss",
        [
            "De tabel van stap 2 geeft elk teken van de formule zijn tensor; het uitgewerkte voorbeeld laat de som en het gemiddelde zien. Alleen het target verandert: $(1 - \\varepsilon)$ maal de delta, plus $\\varepsilon / K$.",
        ],
        why="`K = z.shape[1]`, `p = torch.softmax(z, dim=1)`, `q = (1 - eps) * torch.eye(K)[y] + eps / K`, en `return -(q * torch.log(p)).sum(dim=1).mean()`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_factor():
    mo.md(r"""
    ### Oefening 2. Een factor vier
    """)
    return


@app.cell(hide_code=True)
def choice_factor_cell():
    choice_factor = L.choice(
        "factor-four",
        question="Je vergelijkt je eigen label smoothing met `nn.CrossEntropyLoss(label_smoothing=0.1)` op een batch van 4 voorbeelden. De jouwe is precies 4 keer zo groot. Wat controleer je eerst?",
        options={
            "eps": "Of je ε 4 keer te groot is",
            "log": "Of je log een ander grondtal heeft",
            "right": "Of je over de batch optelt",
        },
        correct="right",
        why={
            "eps": "Een grotere ε verandert het target, en dus de loss, maar niet met een nette factor die gelijk is aan de batch size.",
            "log": "Een ander grondtal vermenigvuldigt elke log met dezelfde factor, maar die factor is ongeveer 1,44 voor grondtal 2 en 0,43 voor grondtal 10, geen 4.",
            "right": "Een factor die gelijk is aan de batch size wijst naar de batch-as: de formule is voor één voorbeeld, en PyTorch neemt het gemiddelde over de batch.",
        },
    )
    choice_factor
    return (choice_factor,)


@app.cell(hide_code=True)
def text_exercise_toy():
    mo.md(r"""
    ### Oefening 3. Waarom een toy-voorbeeld
    """)
    return


@app.cell(hide_code=True)
def explain_toy():
    explain_toy_cell = L.explain(
        "why-toy",
        question="Je code komt op willekeurige inputs overeen met PyTorch. Waarom controleer je hem dan toch op een toy-voorbeeld dat je met de hand uit de formule hebt uitgerekend?",
        model_answer="**De implementatie is de paper niet:** overeenkomen met PyTorch laat zien dat je code doet wat PyTorch doet, en PyTorch kan een andere variant of een andere standaard gekozen hebben dan de paper, zoals het gemiddelde over de batch, of ε alleen verdeeld over de andere klassen. Een toy-voorbeeld uit de formule zelf toetst je code aan de paper. **Het is onafhankelijk:** de berekening met de hand deelt geen code met de jouwe, dus een fout in allebei tegelijk is onwaarschijnlijk, en als ze verschillen, laten de kleine getallen zien waar. **Vaak is er geen:** voor een nieuwe methode is er geen implementatie om mee te vergelijken, en het toy-voorbeeld, met bijzondere gevallen zoals ε = 0, is alles wat je hebt.",
        points=[
            "je zegt dat een implementatie van de paper kan afwijken, zodat overeenkomen ermee niet bewijst dat je de formule volgt",
            "je zegt dat het toy-voorbeeld uit de formule zelf komt, los van elke code, en klein genoeg om te vinden waar het misgaat",
            "je zegt dat er voor nieuwe methoden vaak geen implementatie is om mee te vergelijken",
        ],
    )
    explain_toy_cell
    return (explain_toy_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De label smoothing die je in je derde ronde door de transformer-paper vond, in [Een paper lezen in drie rondes](/course/open/res-three-pass?lang=nl), is nu code, met elk teken een shape zoals in [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl). De volgende les, *Kritisch lezen: claims tegen bewijs*, vraagt of de experimenten van een paper echt steunen wat hij beweert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_factor, explain_toy_cell, result_smoothed):
    L.wrap_up(result_smoothed, choice_factor, explain_toy_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
