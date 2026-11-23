# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-logistic-text. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("nlp-logistic-text", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl), en een over [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tfidf():
    quick_tfidf = L.choice(
        "quick-tfidf",
        question="Van de 2.400 trainingszinnen van deze les hebben er 170 *great* en 747 *and*. Een nieuwe zin heeft twee keer *and* en één keer *great*. Welk van de twee woorden krijgt in die zin de hoogste TF-IDF-waarde?",
        options={
            "twice": "*and*, omdat de zin dat woord twee keer heeft",
            "right": "*great*, omdat minder zinnen het hebben",
            "zero": "Allebei 0, omdat elk van de twee in meer dan één zin staat",
        },
        correct="right",
        why={
            "right": r"*great* krijgt $1 \times \log_{10}(2400 / 170) = 1{,}15$, en *and* $2 \times \log_{10}(2400 / 747) = 1{,}01$: twee keer een woord dat in bijna een derde van alle zinnen staat, telt minder dan één keer een woord dat in 7% ervan staat. In deze les zijn zulke waarden wat logistische regressie voor elke zin krijgt.",
            "twice": "Het aantal is maar een van de twee factoren. De IDF van *and* is laag, omdat 747 van de 2.400 zinnen het woord hebben. Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "zero": "Alleen een woord in elk document krijgt IDF 0, en *and* staat in 747 van de 2.400 zinnen. Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
        },
    )
    quick_tfidf
    return (quick_tfidf,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="Een voorbeeld heeft label 1 en een feature met waarde 2, en logistische regressie geeft het voorbeeld kans 0,8. Wat is de helling van de log loss van dat voorbeeld voor de weight van die feature?",
        options={
            "plus": "0,4, dus de stap verlaagt de weight",
            "right": "−0,4, dus de stap verhoogt de weight",
            "error": "−0,2, de fout zonder de waarde van de feature",
            "times": "1,6, de kans maal de waarde van de feature",
        },
        correct="right",
        why={
            "right": r"De helling is $(p - y) \times x = (0{,}8 - 1) \times 2 = -0{,}4$. Gradient descent trekt de learning rate maal de helling eraf, dus een negatieve helling verhoogt de weight, en de kans van dit positieve voorbeeld gaat omhoog. Deze les doet hetzelfde met één weight voor elk woord.",
            "plus": "Met $y = 1$ en $p = 0{,}8$ is de fout $p - y$ negatief: het voorbeeld is positief en krijgt een te lage kans. Zie [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl).",
            "error": "−0,2 is alleen de fout $p - y$; de helling voor een weight is de fout maal de waarde van de feature. Zie [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl).",
            "times": "De helling gebruikt de fout $p - y$, niet de kans zelf. Zie [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl).",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In *Een document als vector* kregen 600 achtergehouden zinnen uit reviews het label van de trainingszin die er het meest op leek, en 397 daarvan kregen zo het goede label. Naive Bayes, dat uit alle 2.400 trainingszinnen leert hoeveel elk woord over het label zegt, had er 497 goed. Deze les traint de classifier van [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) op dezelfde zinnen, met de gradient descent van [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl). De features zijn de TF-IDF-waarden van [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl): één weight voor elk woord, geleerd uit de labels.

    Die weights kun je lezen, en de weights vertellen wat de classifier geleerd heeft. Daniel Jurafsky en James H. Martin schrijven in *Speech and Language Processing* dat je bij sentiment mag verwachten dat het woord *awesome* een hoge positieve weight krijgt, en *abysmal* een sterk negatieve. De zinnen zijn de 3.000 van *Sentiment Labelled Sentences*, verzameld door Dimitrios Kotzias en collega's: 1.000 van Amazon, 1.000 van IMDb en 1.000 van Yelp, de helft positief en de helft negatief. Net als in *Een document als vector* houdt de les elke vijfde zin achter, dus 2.400 zinnen trainen het model en 600 testen het.

    **Een voorbeeld uit het dagelijks leven.** Een docent beoordeelt een opstel met een puntenlijst: een heldere opening levert 2 punten op, een spelfout kost er 1. Het cijfer is de som. Logistische regressie leert zo'n lijst uit voorbeelden, met een getal voor elk woord.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Welk van deze vier woorden heeft na de training op de 2.400 zinnen de meest negatieve weight?",
        options={
            "bad": "*bad*",
            "worst": "*worst*",
            "right": "*not*",
            "terrible": "*terrible*",
        },
        reveal="***not***, met −1,78. *bad* krijgt −1,49, *worst* −1,06 en *terrible* −0,85. *not* staat in 179 negatieve trainingszinnen en in 40 positieve, zoals *The last 15 minutes of movie are also not bad as well.* (*Het laatste kwartier van de film is ook niet slecht.*); *bad* staat in 72 negatieve en 2 positieve. Elke zin past alleen de weights van de eigen woorden aan, en de vele negatieve zinnen met *not* trekken de weight van *not* verder omlaag dan de 40 positieve zinnen de weight omhoog trekken. *bad* staat in maar 74 zinnen, dus veel minder zinnen veranderen de weight van *bad*: al bij het begin is de helling voor *not* groter, 0,032 tegen 0,026.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Eén weight voor elk woord

    Elke zin wordt een rij TF-IDF-waarden, zoals in [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl): een dictionary van elk woord van de zin naar het aantal keer dat het woord in de zin staat, maal $\log_{10}$ van 2.400 gedeeld door het aantal trainingszinnen met dat woord. De vocabulary bestaat uit de 4.518 woorden van de trainingszinnen; een woord dat in geen enkele trainingszin staat, valt weg. Logistische regressie geeft elk woord $t$ een weight $w_t$, en heeft één bias $b$. De logit $z$ en de kans $p$ dat de zin positief is, zijn

    $$z = b + \sum_{t} x_t \, w_t \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

    De som loopt over de woorden van de zin, en $x_t$ is de TF-IDF-waarde van woord $t$ in die zin. Een woord dat niet in de zin staat, heeft waarde 0 en telt niet mee. Dit is $\mathbf{w} \cdot \mathbf{x} + b$ uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), uitgeschreven voor een sparse rij: van de 4.518 waarden van een zin van tien woorden zijn er hoogstens tien niet 0. Een positieve weight maakt $z$ groter, richting positief, en een negatieve weight maakt $z$ kleiner. De afbeelding laat dat zien voor een achtergehouden zin, met de weights na de training.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    *I did not expect this to be so good!* (*Ik had niet verwacht dat het zo goed zou zijn!*) heeft het label positief, maar krijgt $p = 0{,}423$: *not* draagt −1,852 bij en *good* +1,505, en de andere woorden en de bias brengen $z$ op −0,309. Een weight hoort bij één woord, wat er ook naast staat, dus *not* trekt $z$ hier ook omlaag.

    ### Training: elke zin past de weights van de eigen woorden aan

    De weights worden geleerd met gradient descent op de gemiddelde log loss, zoals in [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl). Over de $n$ trainingszinnen is de helling voor de weight van woord $t$

    $$\text{helling}(w_t) = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) \, x_{i,t}$$

    met $p_i$ de kans voor zin $i$, $y_i$ het label, 1 of 0, en $x_{i,t}$ de waarde van woord $t$ in die zin. Dit is `grad_w = (diff @ X) / n` uit die les, met `diff = p - y`, uitgewerkt voor één woord: alleen de zinnen met woord $t$ tellen mee, want voor de andere zinnen is $x_{i,t} = 0$. Een positieve zin met een te lage kans heeft een negatieve fout $p_i - y_i$, dus de stap verhoogt de weights van de woorden in die zin; een negatieve zin met een te hoge kans verlaagt de weights van de eigen woorden. De helling voor de bias is de gemiddelde fout, $\frac{1}{n} \sum_{i} (p_i - y_i)$. Elke stap trekt van elke weight en van de bias de learning rate maal de helling af.

    ### Een classifier met twee namen

    In NLP heeft logistische regressie een tweede naam. Jurafsky en Martin vertellen in hun historische noten waar die naam vandaan komt: begin jaren negentig werkte men bij IBM Research aan logistische regressie voor NLP, onder de naam maximum entropy modeling of maxent (Berger en collega's, 1996), ogenschijnlijk los van wat er in de statistiek gebeurde. Elders noemen Jurafsky en Martin maximum entropy een andere formulering van logistische regressie: een andere weg naar hetzelfde model. De **maximum entropy classifier**, kortweg maxent, is dus logistische regressie onder een andere naam.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **maximum entropy classifier** (maxent) | een naam in NLP voor logistische regressie, uit de jaren negentig |
        | **weight van een woord** | het getal dat logistische regressie voor een woord leert: een positieve weight trekt een zin richting positief, een negatieve richting negatief |
        | $x_t$ | de TF-IDF-waarde van woord $t$ in een zin |
        | $w_t$ | de weight van woord $t$ |
        """),
        L.watch_out("Twee getallen met een vergelijkbare naam. De TF-IDF-waarde van een woord komt uit tellen en ligt vast. De weight van logistische regressie wordt geleerd uit de labels. In [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl) heette de TF-IDF-waarde in het Engels ook weight; deze les zegt TF-IDF-waarde voor het eerste getal en weight voor het tweede."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Maak van elke zin een rij: een dictionary van elk van de woorden naar de TF-IDF-waarde.
    2. Begin met weight 0 voor elk woord van de vocabulary, en bias 0.
    3. Reken voor elke trainingszin $p$ uit met de huidige weights, en de fout $p - y$.
    4. Tel voor elk woord de fout maal de waarde van het woord op over de zinnen met dat woord, en deel door het aantal zinnen: de helling van de weight. De helling van de bias is de gemiddelde fout.
    5. Trek van elke weight en van de bias de learning rate maal de helling af, en ga terug naar stap 3.
    6. Geef een nieuwe zin het label positief als $p$ 0,5 of meer is, en tel hoeveel achtergehouden zinnen het eigen label krijgen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder neemt de achtergehouden zin *I will not be eating there again.* (*Ik ga daar niet meer eten.*), met het label negatief, en maakt er een rij van met `row` van de course. `parts` van de course geeft voor elk woord van de rij de TF-IDF-waarde en de weight na de training; de code print die twee en het product van de twee. Daarna print de code de bias na de training, uit `bias` van de course.
    """)
    return


@app.cell
def example_cell():
    example = C.row("I will not be eating there again.")
    for word, value, weight in C.parts(example):
        print(word, round(value, 3), round(weight, 3), round(value * weight, 3))
    print("bias", round(C.bias(), 3))
    return (example,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Tel alles op: $z = -0{,}181 + 0{,}016 + 0{,}509 - 1{,}852 - 0{,}132 - 0{,}127 - 0{,}532 + 0{,}450 = -1{,}849$, en $p = \sigma(-1{,}849) = 0{,}136$. Het model zegt negatief, net als het label. *not* doet het meeste werk, *there* helpt mee, en *will* en *again* trekken de andere kant op, met +0,509 en +0,450.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_probability():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. probability: de kans op positief

        Schrijf `probability(row, weights, bias)`, met `row` een dictionary van elk woord van een zin naar de TF-IDF-waarde, `weights` een dictionary van elk woord naar de weight, en `bias` een getal. Geef de kans terug dat de zin positief is: begin een som $z$ bij de bias, tel voor elk woord van de rij de waarde maal de weight op, en geef $\sigma(z)$ terug, `1 / (1 + np.exp(-z))`.

        `small_weights`, `SMALL_WEIGHTS` van de course, bevat vijf weights: *great* 2,21, *good* 1,33, *food* −0,16, *not* −1,78 en *bad* −1,49; de voorbeelden gebruiken bias −0,18. Voor `probability({"great": 1.15, "food": 1.363}, small_weights, -0.18)` is $z = -0{,}18 + 1{,}15 \times 2{,}21 + 1{,}363 \times (-0{,}16) = 2{,}143$, en de uitkomst is ongeveer 0,895. Vervang de `...`. De check roept dat aan en `probability({"not": 1.04, "good": 1.13}, small_weights, -0.18)`, en daarna eigen gevallen: een rij zonder woorden, een woord met een hogere waarde, een andere bias, en de rij van het uitgewerkte voorbeeld met `weights` en `bias`, de weights en de bias na de training, uit `weights` en `bias` van de course.
        """),
        L.task("Reken de kans op positief uit voor één rij TF-IDF-waarden."),
    ])
    return


@app.cell
def weights_cell():
    small_weights = C.SMALL_WEIGHTS
    weights = C.weights()
    bias = C.bias()
    print(small_weights)
    print(len(weights), round(weights["great"], 3), round(weights["food"], 3), round(bias, 3))
    return bias, small_weights, weights


@app.cell(hide_code=True)
def exercise_probability():
    box_probability = L.code_box("probability", lines=5)
    box_probability
    return (box_probability,)


@app.cell(hide_code=True)
def check_probability(box_probability):
    result_probability = L.check("probability", box_probability.value)
    result_probability
    return (result_probability,)


@app.cell(hide_code=True)
def hints_probability():
    L.hints(
        "probability",
        [
            "Begin met `z = bias`, en ga de woorden van de rij langs met `for word in row:`. Tel voor elke `word` de waarde `row[word]` maal de weight van het woord bij `z` op. Geef na de loop `1 / (1 + np.exp(-z))` terug.",
            "Voor de eerste rij krijgt de som 1,15 maal 2,21 voor *great*. Waar in de argumenten van de functie staat 2,21, en hoe zoek je daar het getal van een woord op?",
            r"`        z = z + row[word] * ___`",
        ],
        why=r"`z = bias`; `for word in row:` met `z = z + row[word] * weights[word]`; na de loop `return 1 / (1 + np.exp(-z))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_probability():
    mo.md(r"""
    In de afbeelding gaf je functie zeven achtergehouden zinnen een kans op positief, met de weights na de training. *Everyone is very attentive, providing excellent customer service.* (*Iedereen is erg attent, en de service is uitstekend.*) krijgt 0,94, en *It's like a bad two hour TV movie.* (*Het is net een slechte tv-film van twee uur.*) 0,096. *I did not expect this to be so good!*, uit het idee, krijgt 0,423: het verkeerde label, voor een positieve zin.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gradients():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. gradients: de hellingen

        Schrijf `gradients(examples, weights, bias)`, met `examples` een lijst van paren `(row, label)`, met label 1 voor positief en 0 voor negatief. Geef twee dingen terug: een dictionary van elk woord van `weights` naar de helling van de gemiddelde log loss voor de weight van dat woord, en de helling voor de bias. Elk voorbeeld telt voor elk van de woorden de fout $p - y$ maal de waarde van het woord op, gedeeld door $n$, het aantal voorbeelden; en voor de bias de fout gedeeld door $n$. Hier is $p$ wat `probability` de rij geeft, en `probability` van de course staat voor je klaar. Een woord van `weights` dat in geen enkel voorbeeld staat, krijgt helling 0.

        `small_examples`, `SMALL_EXAMPLES` van de course, bevat drie voorbeelden, en `start_weights`, `START_WEIGHTS` van de course, geeft de vijf woorden weight 0. Bij `gradients(small_examples, start_weights, 0.0)` krijgt elk voorbeeld $p = 0{,}5$. *great* staat alleen in het eerste voorbeeld, en dat is positief, dus de helling is $(0{,}5 - 1) \times 1{,}15 / 3 = -0{,}192$; *food* staat in het eerste en in het derde, en het derde is negatief: $((0{,}5 - 1) \times 1{,}363 + (0{,}5 - 0) \times 1{,}363) / 3 = 0$. De uitkomst is `({"great": -0.191667, "good": 0.188333, "food": 0.0, "not": 0.173333, "bad": 0.251833}, 0.166667)`. Vervang de `...`. De check roept dat aan en `gradients(small_examples, small_weights, -0.18)`, en daarna eigen gevallen: één voorbeeld, `more_examples` met een andere bias, en de 2.400 trainingszinnen, `examples`, uit `training_examples` van de course, met elke weight op 0 in `zero_weights` en met de weights na de training. De cel hieronder print de eerste trainingszin als rij, en als tekst uit `training` van de course.
        """),
        L.task("Reken de helling van de gemiddelde log loss uit voor elke weight en voor de bias."),
    ])
    return


@app.cell
def examples_cell(weights):
    small_examples = C.SMALL_EXAMPLES
    start_weights = C.START_WEIGHTS
    more_examples = C.MORE_EXAMPLES
    examples = C.training_examples()
    zero_weights = {key: 0.0 for key in weights}
    print(small_examples[0])
    print(len(examples), examples[0])
    print(C.training()[0][1])
    return examples, more_examples, small_examples, start_weights, zero_weights


@app.cell(hide_code=True)
def exercise_gradients():
    box_gradients = L.code_box("gradients", lines=10)
    box_gradients
    return (box_gradients,)


@app.cell(hide_code=True)
def check_gradients(box_gradients):
    result_gradients = L.check("gradients", box_gradients.value)
    result_gradients
    return (result_gradients,)


@app.cell(hide_code=True)
def hints_gradients():
    L.hints(
        "gradients",
        [
            "Zet `n = len(examples)`, begin `grad` met 0.0 voor elk woord van `weights`, en zet `grad_bias = 0.0`. Ga de voorbeelden langs met `for row, label in examples:`, en zet `error` op de kans van de rij min `label`. Tel daarna voor elke `word` van de rij de fout maal de waarde van het woord, gedeeld door `n`, bij `grad[word]` op, en tel `error / n` bij `grad_bias` op. Geef na de loops `grad, grad_bias` terug.",
            "Het eerste kleine voorbeeld heeft *great* met waarde 1,15, en de fout is aan het begin 0,5 − 1 = −0,5. Wat telt dat voorbeeld op bij de helling van *great*, voordat je door `n` deelt?",
            r"`            grad[word] = grad[word] + ___ / n`",
        ],
        why=r"`n = len(examples)`, `grad = {word: 0.0 for word in weights}` en `grad_bias = 0.0`; `for row, label in examples:` met `error = probability(row, weights, bias) - label`, dan `for word in row:` met `grad[word] = grad[word] + error * row[word] / n`, en na die loop `grad_bias = grad_bias + error / n`; aan het eind `return grad, grad_bias`.",
    )
    return


@app.cell(hide_code=True)
def text_after_gradients():
    mo.md(r"""
    In de afbeelding rekende je functie de hellingen van de eerste stap uit, over de 2.400 trainingszinnen, met elke weight op 0. *great* heeft de meest negatieve helling, dus de eerste stap verhoogt de weight van *great* het meest, en *not* heeft de meest positieve helling. Ook *and* en *that* staan in de lijsten: *and* staat in 450 positieve en 297 negatieve trainingszinnen, *that* in 86 positieve en 133 negatieve.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train het model

    De knop traint logistische regressie op de 2.400 trainingszinnen: 300 stappen gradient descent met learning rate 1, vanaf weight 0 voor elk woord en bias 0. Elke stap haalt de hellingen uit je eigen `gradients` zodra oefening 2 goed is, en tot dan uit `gradients` van de course. Elke stap rekent 4.518 hellingen uit, dus de 300 stappen duren een paar seconden. Tijdens de training toont de grafiek na elke stap de gemiddelde log loss van de trainingszinnen; ernaast zie je de zes woorden met de sterkste weight naar elke kant, en de kans op positief van drie achtergehouden zinnen. Daarna kun je teruggaan door de stappen.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train 300 stappen")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(box_gradients, demo_button, result_gradients):
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om te trainen.*")
    else:
        _yours = C.mine(box_gradients.value, result_gradients.state == "correct", "gradients", C.EXERCISES["gradients"].names)
        _words = C.live_words(L.lang)
        _now = {}

        def _progress(_step, _loss, _ws, _b):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                      C.STEPS, low=0.0, high=0.7)
            _now["live"].watch(lambda: C.now_parts(_ws, _b, L.lang))
            _now["live"].add(_step, _loss, _words["progress"].format(s=_step, n=C.STEPS, loss=L.number(_loss, 3)))

        mo.output.replace(mo.md("*Stap 1 · bezig met starten …*"))
        _run = C.demo(_yours, _progress)
        if isinstance(_run, str):
            _note = {"gradients": "*Je `gradients` stopt met een fout op de trainingszinnen, of geeft iets anders dan een helling voor elk woord en een voor de bias. De afbeelding bij oefening 2 laat zien wat er misgaat.*",
                     "slow": "*De training duurt langer dan twee minuten. Gaat je `gradients` de voorbeelden maar één keer langs? Of is de computer met iets anders bezig?*",
                     "course": "*De training duurt langer dan twee minuten. Is de computer met iets anders bezig?*"}[_run if _yours else "course"]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _right = C.rewind_right(_run, C.STEPS)
            _who = " Je eigen `gradients` rekende de hellingen uit." if _yours else ""
            _out = mo.vstack([
                mo.md(f"""Na {C.STEPS} stappen is de gemiddelde log loss van de trainingszinnen {L.number(_run["curve"][-1][1], 3)}, en geeft het model <span class="ea-num">{_right}</span> van de 600 achtergehouden zinnen het goede label.{_who}"""),
                L.explore("rewind", "stap", lambda _s: (C.rewind_frame(_run, int(_s), L.lang),
                                                        f"Na stap {int(_s)}: {C.rewind_right(_run, int(_s))} van de 600 achtergehouden zinnen goed."),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Ga terug naar stap 1, en dan weer vooruit: welke woorden krijgen als eerste een sterke weight, en wat gebeurt er met *I did not expect this to be so good!*?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met elke weight op 0 krijgt elke zin $p = 0{,}5$, en is de gemiddelde log loss $\log 2 = 0{,}693$; na 300 stappen is de log loss 0,165. Het model geeft dan 2.354 van de 2.400 trainingszinnen het goede label, 98,1%, en 491 van de 600 achtergehouden zinnen, 81,8%: van 456 na de eerste stap naar 491 na stap 70, en vanaf daar tussen 488 en 491. De sterkste woorden richting positief zijn *great* (2,21), *good* (1,33), *excellent* (1,16), *nice* (1,14), *love* (1,10) en *works* (0,99); richting negatief *not* (−1,78), *bad* (−1,49), *worst* (−1,06), *poor* (−0,99), *terrible* (−0,85) en *don't* (−0,82). Van de twee woorden van Jurafsky en Martin krijgt *awesome* 0,74, een hoge positieve weight; *abysmal* staat in maar twee trainingszinnen en krijgt −0,25: negatief, maar niet sterk. *and* en *that*, uit de afbeelding bij oefening 2, eindigen op +0,67 en −0,43.

    *I did not expect this to be so good!* zakt eerst, naar 0,393 na stap 50: wat *not* eraf haalt, groeit sneller dan wat *good* bijdraagt, en *did*, *be* en de bias trekken $z$ ook omlaag. Daarna klimt de kans een beetje terug, naar 0,423, doordat *did* positief wordt en *so* groeit; *good* haalt *not* nooit in, dus de zin krijgt nog steeds niet het eigen label.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. accuracy: hoeveel zijn goed?

        Schrijf `accuracy(examples, weights, bias)`: geef elk voorbeeld het label 1 als `probability` de rij 0,5 of meer geeft, en anders 0, tel de voorbeelden waarbij dat het eigen label is, en geef het aandeel terug: dat aantal gedeeld door het aantal voorbeelden.

        Voor `accuracy(small_examples, small_weights, -0.18)` is de uitkomst `1.0`: de drie voorbeelden krijgen 0,895, 0,371 en 0,066, dus de labels 1, 0 en 0, en dat zijn de eigen labels. `more_examples`, `MORE_EXAMPLES` van de course, bevat vier voorbeelden, en `accuracy(more_examples, small_weights, -0.18)` geeft `0.5`: twee van de vier krijgen het eigen label. Vervang de `...`. De check roept die twee aan, en daarna eigen gevallen: één voorbeeld dat het andere label krijgt, een bias die elk voorbeeld positief maakt, en de 600 achtergehouden zinnen, `held`, uit `held_out_examples` van de course, met de weights na de training. De cel hieronder print de eerste achtergehouden zin als rij, en als tekst uit `held_out` van de course.
        """),
        L.task("Tel welk aandeel van de voorbeelden het eigen label krijgt."),
    ])
    return


@app.cell
def held_cell(more_examples):
    held = C.held_out_examples()
    print(more_examples[0])
    print(len(held), held[0])
    print(C.held_out()[0][1])
    return (held,)


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("accuracy", lines=9)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Begin met `right = 0`, en ga de voorbeelden langs met `for row, label in examples:`. Zet in elke ronde `guess = 0`, en zet `guess` op 1 als `probability(row, weights, bias)` hoog genoeg is. Is `guess` gelijk aan `label`, tel dan 1 bij `right` op. Geef na de loop `right / len(examples)` terug.",
            "Het tweede kleine voorbeeld krijgt kans 0,371. Welk label geeft het model dat voorbeeld, en vanaf welke kans zegt het model positief?",
            r"`        if probability(row, weights, bias) >= ___:`",
        ],
        why=r"`right = 0`; `for row, label in examples:` met `guess = 0`, `if probability(row, weights, bias) >= 0.5:` en `guess = 1`, dan `if guess == label:` en `right = right + 1`; na de loop `return right / len(examples)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracy():
    mo.md(r"""
    In de afbeelding mat je functie logistische regressie op de 600 achtergehouden zinnen: 491 goed, 81,8%. Naive Bayes, op de kale aantallen, heeft er 497 goed, 82,8%: zes meer. De twee hebben niet dezelfde zinnen fout. Beide hebben 83 zinnen fout; alleen logistische regressie heeft er nog 26 fout, en alleen naive Bayes nog 20.

    ### Hoe goed is het?

    Op deze korte zinnen doet logistische regressie het niet beter dan naive Bayes. Sida Wang en Christopher Manning zetten in 2012 naive Bayes naast support vector machines, een ander soort classifier, en vonden iets vergelijkbaars: naive Bayes doet het bij sentiment van korte fragmenten beter, en bij langere documenten juist niet. De zinnen uit reviews van deze les zijn zulke korte fragmenten. Andrew Ng en Michael Jordan deden in 2001 een verwante ontdekking, over het aantal trainingsvoorbeelden. Volgens hen maakt discriminatief leren op den duur minder fouten, maar kan een generatieve classifier veel sneller in de buurt komen van het eigen, hogere aantal fouten. Op den duur betekent hier: met steeds meer voorbeelden. Naive Bayes noemen ze generatief: het model beschrijft hoe elk label woorden voortbrengt. Logistische regressie is discriminatief: het model leert het label rechtstreeks uit de woorden. Met weinig voorbeelden kan naive Bayes het dus beter doen, en met veel voorbeelden kan logistische regressie het inhalen.

    De weights helpen wel te zien waarom het model beslist zoals het beslist. Jurafsky en Martin schrijven over het uitleggen van een model dat de grootte van de weight van een feature kan helpen begrijpen waarom de classifier een beslissing nam. De meest gelijkende zin uit *Een document als vector* kan dat niet: die methode wijst alleen naar één andere zin.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Welke weight?

    Elke regel geeft een woord, en in hoeveel positieve en hoeveel negatieve trainingszinnen het woord voorkomt. Zet elk woord bij de weight die het krijgt na de 300 stappen training.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_weight = L.sort("what-weight")
    sort_weight
    return (sort_weight,)


@app.cell(hide_code=True)
def check_sort(sort_weight):
    result_sort = L.check("what-weight", sort_weight)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: een eigen zin

    Typ een zin in het Engels. De afbeelding laat zien wat elk woord bijdraagt aan $z$, met de weights na de training, en de kans op positief die daaruit volgt. Probeer *Not bad at all.* (*Helemaal niet slecht.*), of een zin over een maaltijd, een film of een telefoon.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_text = mo.ui.text(value=C.DEMO_TEXT, label="Zin", full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(try_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    *The food was not great, but the staff were lovely.* (*Het eten was niet geweldig, maar het personeel was heel aardig.*) krijgt 0,732. *great* draagt +2,537 bij en *not* −1,852: logistische regressie telt op wat elk woord op zichzelf zegt, en ziet niet dat *not* bij *great* hoort. *Not bad at all.* krijgt 0,008: *not* en *bad* hebben allebei een sterk negatieve weight, en *at* haalt er nog 0,689 af, dus het model is zeker van de zaak, en heeft het mis. Een latere les van het domein, *Een lijst met goede en slechte woorden*, draait de score van een woord om na een ontkenning.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl) maakte van elke zin een rij TF-IDF-waarden; deze les gaf die rijen aan de logistische regressie van [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), met één weight voor elk woord, getraind met de gradient descent van [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl). De weights laten zien welke woorden richting positief trekken en welke richting negatief, en op korte zinnen doet naive Bayes het ongeveer even goed. De volgende les van het domein, *Veel labels: micro- en macrogemiddelden*, meet een classifier die uit meer dan twee labels kiest.

    **Een voorbeeld uit het dagelijks leven.** Een spamfilter kan zo werken: elk woord van een e-mail heeft een geleerde weight, en de som beslist of de e-mail in de map met spam belandt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_accuracy, result_gradients, result_probability, result_sort):
    L.wrap_up(result_probability, result_gradients, result_accuracy, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
