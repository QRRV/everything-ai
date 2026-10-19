# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les opt-gradient-clipping. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-gradient-clipping", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl) en [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_half():
    quick_half = L.choice(
        "quick-half",
        question="Je vermenigvuldigt elk getal van een vector met 0,5. Wat gebeurt er met zijn norm, en met zijn richting?",
        options={
            "right": "De norm halveert, en de richting blijft hetzelfde",
            "quarter": "De norm wordt een kwart, en de richting blijft hetzelfde",
            "turns": "De norm halveert, en de richting draait",
        },
        correct="right",
        why={
            "quarter": "De getallen worden gekwadrateerd, en de kwadraten worden wel een kwart, maar de norm is de wortel uit hun som: de wortel uit een kwart is een half. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "turns": "De cosine similarity met de oude vector blijft 1: hij kijkt alleen naar de richting, en vermenigvuldigen met een positief getal verandert die niet. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "right": "De pijl wordt half zo lang en wijst dezelfde kant op: zijn cosine similarity met de oude vector is 1. Vandaag gebruik je precies dat om een gradient korter te maken zonder hem te draaien.",
        },
    )
    quick_half
    return (quick_half,)


@app.cell(hide_code=True)
def quick_check_layers():
    quick_layers = L.choice(
        "quick-layers",
        question="Op de terugweg door 20 layers wordt een gradient in elke layer met ongeveer 1,5 vermenigvuldigd. Bij de laatste layer is hij 1. Hoe groot is hij ongeveer bij de eerste?",
        options={
            "sum": "Ongeveer 30: 20 keer 1,5",
            "same": "Ongeveer 1,5: elke layer verandert hem met dezelfde factor",
            "right": "Ongeveer 3.300: 1,5 twintig keer met zichzelf vermenigvuldigd",
        },
        correct="right",
        why={
            "sum": "De factoren worden vermenigvuldigd, niet opgeteld: 1,5 keer 1,5 keer … , 20 keer. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "same": "De factor van elke layer komt bovenop die van de layers erna: ze vermenigvuldigen zich. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "right": "$1{,}5^{20} \\approx 3300$: een exploding gradient. Weights met de goede spreiding houden de factoren rond 1; de les van vandaag is het vangnet voor de stappen waarin een gradient toch te groot wordt.",
        },
    )
    quick_layers
    return (quick_layers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Ook in een netwerk dat goed traint, kan één mini-batch een gradient geven die honderd keer groter is dan de andere: een voorbeeld met een fout erin, of een steile plek in de loss. Eén stap met zo'n gradient kan de weights ver weg gooien van waar ze waren. Grote language models trainen wekenlang op miljarden voorbeelden, en daartussen zit altijd wel een slecht voorbeeld. **Gradient clipping** bewaakt elke stap: als de gradient groter is dan een grens die je kiest, wordt hij vóór de stap korter gemaakt.
    """)
    return


@app.cell(hide_code=True)
def predict_bad_batch():
    guess_bad_batch = L.guess(
        "bad-batch",
        question="Het MLP uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) traint één epoch van 94 mini-batches, met SGD en een learning rate van 0,5. In mini-batch 41 is iemand vergeten de pixels door 255 te delen: ze lopen van 0 tot 255. De norm van zijn gradient is 146; bij de mini-batches ervoor is die ongeveer 1. Zonder die fout leest het netwerk 865 van de 1.000 testcijfers goed. Hoeveel denk je dat het er met de fout goed leest?",
        options={
            "same": "Ongeveer 865: één mini-batch van de 94 telt nauwelijks mee",
            "right": "Ongeveer 400: één enorme stap maakt veel van wat het leerde ongedaan",
            "zero": "Ongeveer 100, zo goed als gokken: de loss wordt `nan` en blijft dat",
        },
        reveal="**405.** Eén stap met een gradient die 146 keer te groot is, gooit de weights zo ver weg dat de loss naar meer dan 1.000 springt, en in de 53 mini-batches die nog over zijn komt het netwerk niet meer terug op waar het was. Met de gradient geclipt op een norm van 5 leest dezelfde epoch, met de slechte mini-batch erbij, er 870 goed.",
    )
    guess_bad_batch
    return (guess_bad_batch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zet alle getallen van de gradient, van alle layers, achter elkaar in één lange vector $\mathbf{g}$. Zijn norm $\lVert \mathbf{g} \rVert$ zegt hoe groot de hele stap zou worden. Kies een grens $c$. Ligt de norm boven $c$, vermenigvuldig dan elk getal van de gradient met $c / \lVert \mathbf{g} \rVert$; anders laat je hem met rust:
        """),
        L.formula(r"\lVert [[grad:\mathbf{g}]] \rVert = \sqrt{\textstyle\sum_i [[grad:g_i]]^2}"),
        L.formula(r"[[grad:\mathbf{g}]] \leftarrow [[grad:\mathbf{g}]] \cdot \frac{[[hyper:c]]}{\lVert [[grad:\mathbf{g}]] \rVert} \quad \text{als } \lVert [[grad:\mathbf{g}]] \rVert > [[hyper:c]]"),
        mo.md(r"""
        Daarna is de norm precies $c$. Omdat elk getal met dezelfde factor wordt vermenigvuldigd, wijst de gradient nog dezelfde kant op: de stap gaat in dezelfde richting, alleen minder ver. Een gewone stap, onder de grens, verandert helemaal niet.

        In PyTorch doet één regel tussen `loss.backward()` en `opt.step()` het: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`. De `_` aan het eind van de naam zegt dat hij de gradients zelf verandert. Hij geeft de norm vóór het clippen terug, als tensor; als je die bijhoudt, zie je wanneer er een slechte mini-batch langskomt, en hoe groot de normen meestal zijn, wat helpt om `max_norm` te kiezen: iets boven de gewone normen, zodat alleen de uitschieters geclipt worden. Grote language models worden meestal getraind met `max_norm=1.0`. Er is ook `torch.nn.utils.clip_grad_value_`, die elk getal apart afkapt op $-c$ en $c$, een methode die je zelden nodig hebt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **gradient clipping** | | de gradient vóór de stap korter maken als zijn norm boven een grens ligt | `clip_grad_norm_` |
        | $\lVert \mathbf{g} \rVert$ | de norm van g | de norm van alle getallen van de gradient samen, van alle layers | wat `clip_grad_norm_` teruggeeft |
        | $c$ | c | de grens: de grootste norm die de gradient mag hebben | `max_norm` |
        """),
        L.watch_out("Clippen verandert de stap, niet de loss: zet het na `loss.backward()`, als de gradients er zijn, en vóór `opt.step()`, die ze gebruikt. En ook [Adam](/course/open/opt-adam?lang=nl) heeft het nodig. Adam houdt elke stap op ongeveer $\\eta$, dus met Adam op 0,001 doet de slechte mini-batch uit de gok minder schade: 810 goed in plaats van 845. Maar de enorme gradient blijft lang in de gemiddelden $m$ en $s$ van Adam zitten, en de stappen erna zijn slechter: met clippen op 5 leest Adam er 840 goed."),
        mo.md(r"""
        De figuur: de epoch uit de gok, zonder clippen (gestippeld) en met clippen op 5 (doorgetrokken). Links de norm van de gradient van elke mini-batch, vóór het clippen: ongeveer 1, en 146 bij mini-batch 41: met clippen de enige boven de grens. Zonder clippen geven de twee mini-batches daarna ook normen van meer dan 100: de weights staan ver van waar ze waren. Rechts de loss van elke mini-batch: zonder clippen springt hij naar meer dan 1.000 en blijft hij boven 1, gemiddeld 1,6 over de laatste tien mini-batches; met clippen 0,4. Allebei op een logaritmische schaal.
        """),
        mo.Html(C.clip_picture({"norm_title": "Norm van de gradient", "loss_title": "Training loss", "x": "mini-batch", "norm": "norm vóór clippen", "loss": "loss",
                                "limit": "grens 5", "without": "zonder clippen", "with": "met clippen",
                                "alt": "Twee panelen op een logaritmische schaal over 94 mini-batches. Links de norm van de gradient: ongeveer 1, en 146 bij mini-batch 41, ver boven de gestippelde grens op 5; zonder clippen blijft hij nog twee mini-batches boven 100. Rechts de loss: zonder clippen springt hij naar ongeveer 1.000 en blijft hij boven 1; met clippen heeft hij een korte piek en zakt hij verder tot ongeveer 0,4."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Gradient clipping op de norm, in elke trainingsstap:

    1. Reken na `loss.backward()` de norm van alle gradients samen uit: de wortel uit de som van de kwadraten van al hun getallen.
    2. Ligt die boven de grens $c$, reken dan de factor $c / \lVert \mathbf{g} \rVert$ uit; anders is de factor 1.
    3. Vermenigvuldig elke gradient met die factor.
    4. Zet daarna de stap, `opt.step()`, en houd de norm van stap 1 bij.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een netwerk met een weight-matrix en één bias term, waarvan de gradient geclipt wordt op $c = 5$, in NumPy. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_clip():
    import numpy as np

    _g_w = np.array([[3.0, 0.0], [0.0, 4.0]])
    _g_b = np.array([12.0])
    _c = 5.0
    # 1. de norm van alle getallen samen
    _norm = (np.sum(_g_w ** 2) + np.sum(_g_b ** 2)) ** 0.5
    # 2. maak af: als _norm boven _c ligt, maak _scale dan _c / _norm
    _scale = 1.0
    # 3. elke gradient keer de factor
    _g_w = _g_w * _scale
    _g_b = _g_b * _scale
    print("norm:", _norm, "factor:", round(_scale, 4))
    print(np.round(_g_w, 4), np.round(_g_b, 4))
    print("norm erna:", round((np.sum(_g_w ** 2) + np.sum(_g_b ** 2)) ** 0.5, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De norm van de vijf getallen samen is $\sqrt{9 + 16 + 144} = 13$, ver boven 5. Zoals hij is, vermenigvuldigt de cel met 1, en is de norm erna nog steeds 13. Met stap 2 af is de factor $5 / 13 \approx 0{,}3846$: de 12 wordt 4,6154, de 3 en de 4 worden 1,1538 en 1,5385, en de norm erna is 5. De getallen houden hun verhouding 3 : 4 : 12, dus de gradient wijst dezelfde kant op. Op zichzelf heeft de weight-matrix een norm van 5, precies de grens: elke array clippen op zijn eigen norm zou hem met rust laten en alleen de bias term korter maken, en dan verandert de richting.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De norm, niet elk getal
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "keeps-direction",
        question="`torch.nn.utils.clip_grad_value_` kapt elk getal van de gradient apart af op $-c$ en $c$. Waarom clippen bijna alle trainingsscripts in plaats daarvan de norm?",
        options={
            "less-work": "De norm clippen is minder werk: het kijkt naar één getal in plaats van naar allemaal",
            "larger": "Elk getal afkappen kan de gradient groter maken dan hij was",
            "right": "Alle getallen met één factor vermenigvuldigen houdt de richting van de stap; getallen één voor één afkappen verandert die",
        },
        correct="right",
        why={
            "less-work": "Om de norm uit te rekenen heb je ook elk getal nodig: de wortel uit de som van al hun kwadraten.",
            "larger": "Afkappen maakt een getal alleen kleiner in grootte, nooit groter. Wat het wel verandert, is de verhouding tussen de getallen.",
            "right": "Met $c = 5$ wordt de gradient $(3, 4, 12)$ uit het voorbeeld $(3, 4, 5)$: het grote getal wordt afgekapt en de kleine niet, dus de stap draait. Clippen op de norm geeft $(1{,}15;\\ 1{,}54;\\ 4{,}62)$, dezelfde richting.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_clip():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een lijst gradients clippen

        Schrijf `clip_grads(grads, max_norm)`: `grads` is een lijst NumPy-arrays, de gradients van de layers van een netwerk, elk met een eigen vorm. Geef een lijst terug met de arrays geclipt volgens stap 1 tot en met 3 van het recept: één norm over alle getallen van alle arrays, en als die boven `max_norm` ligt, elke array keer `max_norm / norm`. Onder de grens geef je de arrays terug zoals ze zijn. De check roept `clip_grads([np.array([1, 2]), np.array([[2], [4]])], 2.5)` aan, wat `[np.array([0.5, 1]), np.array([[1], [2]])]` moet geven, en probeert andere lijsten: onder de grens, op de grens, één array, en alleen nullen.
        """),
        L.task("Laat `clip_grads` een lijst gradients clippen op hun norm samen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clip():
    box_clip = L.code_box("clip-grads", lines=10)
    box_clip
    return (box_clip,)


@app.cell(hide_code=True)
def check_clip(box_clip):
    result_clip = L.check("clip-grads", box_clip.value)
    result_clip
    return (result_clip,)


@app.cell(hide_code=True)
def hints_clip():
    L.hints(
        "clip-grads",
        [
            "De norm van alle arrays samen: tel `np.sum(g ** 2)` op voor elke `g` in `grads`, en neem de wortel uit het totaal, met `** 0.5`.",
            "Dan `if norm > max_norm:` een nieuwe lijst maken, er met `append` `g * (max_norm / norm)` aan toevoegen voor elke `g`, en die teruggeven; anders `grads` teruggeven.",
        ],
        why="`total = 0` en `for g in grads: total = total + np.sum(g ** 2)`; `norm = total ** 0.5`; `if norm > max_norm:` dan `out = []`, `for g in grads: out.append(g * (max_norm / norm))` en `return out`; na de `if` `return grads`. In het voorbeeld is de norm $\\sqrt{1 + 4 + 4 + 16} = 5$, dus de factor is $2{,}5 / 5 = 0{,}5$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een trainingsstap met clippen

        Schrijf `train_step(model, opt, xb, yb, max_norm)`: één trainingsstap op de mini-batch `xb`, `yb`, zoals in de loop van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), met gradient clipping op `max_norm` op de goede plek. Geef de norm vóór het clippen terug, als getal: wat `clip_grad_norm_` teruggeeft, met `.item()`. De check maakt een klein `nn.Linear`-model met een SGD-optimizer, zet een stap met grens 1, en in een tweede poging twee stappen na elkaar, de eerste met grens 100 en de tweede met grens 2; hij vergelijkt de normen die je teruggeeft en de weights en bias terms na de stappen.
        """),
        L.task("Laat `train_step` één trainingsstap met gradient clipping zetten en de norm teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("train-step", lines=12)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("train-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "train-step",
        [
            "De stappen van de loop uit het project blijven: de loss, `opt.zero_grad()`, `loss.backward()` en `opt.step()`. Waar daartussen zijn de gradients er al, en nog niet gebruikt?",
            "Tussen `loss.backward()` en `opt.step()` komt `torch.nn.utils.clip_grad_norm_`, met de parameters van het model en `max_norm`; wat hij teruggeeft is de norm die je nodig hebt.",
        ],
        why="`loss = loss_fn(model(xb), yb)`; `opt.zero_grad()`; `loss.backward()`; `norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`; `opt.step()`; `return norm.item()`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De norm uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl) meet hoe groot een stap zou worden, en schalen met één factor houdt zijn richting. [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) liet zien hoe een gradient layer na layer kan groeien als de weights de verkeerde spreiding hebben; clippen vangt een te grote gradient op, wat de oorzaak ook is, zoals één slechte mini-batch. [Adam](/course/open/opt-adam?lang=nl) begrenst elke stap zelf, maar zijn gemiddelden onthouden een enorme gradient: ook daar helpt clippen. In de volgende les, *Dropout*, bestrijd je overfitting door tijdens het trainen een deel van de neuronen uit te zetten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, result_clip, result_step):
    L.wrap_up(choice_direction, result_clip, result_step)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
