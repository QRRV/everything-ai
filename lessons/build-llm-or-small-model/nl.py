# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-llm-or-small-model. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-llm-or-small-model", "nl")
    C = L.checks   # de vragen van de bank, de opgenomen antwoorden van Qwen3, het kleine model van de course, en de demo met het jouwe


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Kant-en-klare AI: zien, spraak, vertalen en klassieke NLP](/course/open/build-ai-services?lang=nl), één over [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl), en één over [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parse():
    quick_parse = L.choice(
        "quick-parse",
        question="Een model antwoordt `Card_Payment_Wrong_Exchange_Rate`. Waarom geeft de `parse_label` van de course het langste label terug dat hij erin vindt?",
        options={
            "piece": "Omdat `exchange_rate` er ook in staat, als stuk van het langere label",
            "fast": "Omdat een langer label sneller te vinden is",
            "rare": "Omdat lange labels zeldzamer zijn, dus eerder goed",
        },
        correct="piece",
        why={
            "piece": "Beide labels staan in het antwoord; het model bedoelde het langere.",
            "fast": "Snelheid heeft er niets mee te maken. Welke labels staan er in dat antwoord? Zie [Kant-en-klare AI: zien, spraak, vertalen en klassieke NLP](/course/open/build-ai-services?lang=nl).",
            "rare": "Hoe vaak een label voorkomt, doet hier niet ter zake. Welke labels staan er in dat antwoord? Zie [Kant-en-klare AI: zien, spraak, vertalen en klassieke NLP](/course/open/build-ai-services?lang=nl).",
        },
    )
    quick_parse
    return (quick_parse,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="Wat doen `model.fit(X, y)` en `model.predict(X_new)` in scikit-learn?",
        options={
            "same": "Allebei trainen ze het model, de tweede op nieuwe data",
            "both": "`fit` leert van `X` en de goede antwoorden `y`; `predict` geeft de antwoorden van het model voor nieuwe rijen",
            "check": "`fit` controleert of het model bij de data past; `predict` traint het",
        },
        correct="both",
        why={
            "both": "Elk model van scikit-learn heeft deze twee: leren, en dan gebruiken.",
            "same": "`predict` krijgt geen goede antwoorden, dus waarvan zou het moeten leren? Zie [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl).",
            "check": "Het is andersom: bij `fit` gebeurt het leren. Zie [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl).",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="Een model heeft precision 1,0 en recall 0,1. Wat is de F1?",
        options={
            "mean": "0,55, het gemiddelde van de twee",
            "one": "1,0, de hoogste van de twee",
            "low": "Ongeveer 0,18, dicht bij de laagste",
        },
        correct="low",
        why={
            "low": "De F1 is het harmonisch gemiddelde, 2 · 1,0 · 0,1 / 1,1 ≈ 0,18: het blijft dicht bij de laagste van de twee.",
            "mean": "Dat is het gewone gemiddelde. Welk gemiddelde neemt de F1? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "one": "Dan zou een model dat bijna niets vindt de volle score halen. Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Kant-en-klare AI: zien, spraak, vertalen en klassieke NLP](/course/open/build-ai-services?lang=nl) sorteerde Qwen3 vragen van klanten in zes labels met alleen een prompt, en had er 99 van de 120 goed. Voor één vaste taak die steeds opnieuw draait, is er een andere manier: een eigen klein model, getraind op gesorteerde voorbeelden, dat die ene taak doet en verder niets. Het kan beter zijn, het is veel sneller, en het kost bijna niets per bericht. Maar het heeft die gesorteerde voorbeelden nodig, en het kent alleen zijn eigen taak. Deze les meet ze allebei op dezelfde 120 vragen, zodat je kunt kiezen.

    **Een voorbeeld uit het dagelijks leven.** Een goede kok maakt elk gerecht waar je om vraagt. Een koffiemachine maakt één ding, maar wel in een paar seconden, elke keer hetzelfde, en voor bijna niets. Voor koffie gebruik je de machine; voor een nieuw gerecht heb je de kok nodig. Dat is alleen een vergelijking; hieronder staan de echte getallen.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **taakspecifiek model** | een klein model, getraind voor één taak op voorbeelden van die taak, zoals berichten in zes labels sorteren; verder weet het niets |
    | **TF-IDF** | *term frequency, inverse document frequency*: elke tekst wordt een vector met een getal voor elk woord dat de teksten gebruiken; het getal is hoger als het woord in deze tekst staat, en lager als het woord in veel teksten staat, zoals `my` of `the`; elke vector heeft lengte 1 |
    | `TfidfVectorizer` | uit scikit-learn: `fit_transform(texts)` leert de woorden van de trainingsteksten en maakt hun vectoren; `transform(new_texts)` maakt vectoren van nieuwe teksten met dezelfde woorden |
    | `LogisticRegression` | de logistische regressie van [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), in scikit-learn; met meer dan twee labels geeft hij elk label een kans, zoals softmax doet, en `predict` kiest het waarschijnlijkste |
    | **macro F1** | de F1 score van elk label, waarbij dat label ja telt en alle andere nee, en dan het gemiddelde van die F1 scores: elk label telt even zwaar, ook een label dat weinig voorkomt |
    | `a & b` | met twee NumPy-arrays van `True` en `False`: `True` waar ze allebei `True` zijn |

    ### Het kleine model tegen Qwen3

    De course trainde zo'n klein model, TF-IDF en logistische regressie, op gesorteerde vragen van dezelfde bank: eerst op 2 vragen van elk label, dan op 5, 10, 20, 30 en ten slotte op alle 50. Elke keer mat hij de macro F1 op dezelfde 120 testvragen die Qwen3 daar beantwoordde, en die het kleine model nooit zag:
    """)
    return


@app.cell(hide_code=True)
def picture_curve():
    mo.Html(C.curve_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_compare():
    mo.vstack([
        mo.md(r"""
        Met 2 of 5 vragen per label is het kleine model slechter dan Qwen3: het heeft te weinig woorden gezien. Vanaf 10 per label is het beter, en met alle 50 haalt het een macro F1 van 0,934, met 112 van de 120 goed, tegen 0,823 en 99 van Qwen3. De lijn wiebelt een beetje: met 20 testvragen per label verandert één antwoord meer of minder goed de score met ongeveer 0,01.

        | | Qwen3-0.6B met een prompt | het kleine model, 50 per label |
        |---|---|---|
        | macro F1 op de 120 testvragen | 0,823 | 0,934 |
        | goed van de 120 | 99 | 112 |
        | tijd voor de 120 | 125 seconden op de computer van de course | minder dan een tiende seconde, het trainen op de 300 vragen erbij |
        | grootte | 1,5 GB | 32 kB: 443 woorden, en 6 weights voor elk |
        | verzonnen prijs van een miljoen berichten, bij 3 dollar per miljoen tokens erin en 15 eruit | 393 dollar: 116 tokens erin en 3 eruit per bericht | alleen de computer waarop het draait |
        | heeft nodig | een prompt | 300 gesorteerde vragen |

        Voor één vaste taak met gesorteerde voorbeelden wint het kleine model dus op alle drie. Het LLM wint als er geen gesorteerde voorbeelden zijn, als de labels vaak veranderen, als een bericht kennis van de wereld nodig heeft of in een taal komt die de voorbeelden niet hebben, en als het antwoord een tekst is in plaats van een label.
        """),
        L.watch_out("Meet ze allebei op dezelfde testteksten, en nooit op teksten waarvan een model leerde: een klein model kent zijn 300 trainingsvragen uit zijn hoofd, en zou er beter uitzien dan het is. De training set en test set van [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl) zijn hier net zo nodig."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een klein model voor één vaste taak

    1. **Verzamel gesorteerde voorbeelden:** teksten met hun labels, en houd er een deel apart om te testen.
    2. **Maak vectoren van de teksten:** `vectorizer = TfidfVectorizer()`, en dan `X = vectorizer.fit_transform(texts)`.
    3. **Train:** `model = LogisticRegression(max_iter=1000)`, en dan `model.fit(X, labels)`. De `max_iter=1000` geeft het genoeg stappen om het leren af te maken.
    4. **Sorteer nieuwe teksten:** `model.predict(vectorizer.transform(new_texts))`, met `transform`, niet `fit_transform`: nieuwe teksten krijgen vectoren met de woorden van de trainingsteksten.
    5. **Meet** de macro F1 op de test set, en vergelijk met het LLM op dezelfde teksten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Wat TF-IDF maakt van drie korte teksten: `my card is lost`, `my card has not arrived` en `my pin is blocked`.

    1. **De woorden:** `fit_transform` vindt 9 verschillende woorden, op volgorde van het alfabet: `arrived`, `blocked`, `card`, `has`, `is`, `lost`, `my`, `not`, `pin`. Elke tekst wordt een vector van 9 getallen.
    2. **De vector van `my card is lost`:** `card` 0,480, `is` 0,480, `lost` 0,632, `my` 0,373, en 0 voor de vijf woorden die er niet in staan.
    3. **Waarom die getallen:** `lost` staat alleen in deze tekst, dus het zegt er het meest over en krijgt het hoogste getal; `card` en `is` staan in twee van de drie teksten; `my` staat in alle drie, dus het zegt het minst en krijgt het laagste. De vier getallen vormen een vector met lengte 1.
    4. **Wat het model dan leert:** een weight voor elk woord en elk label, zoals een hoge weight voor `lost` bij `lost_or_stolen_card`.
    """)
    return


@app.cell(hide_code=True)
def text_train():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Train het kleine model

        Maak `train_small(texts, labels)` af: hij maakt TF-IDF-vectoren van de teksten, traint er een logistische regressie op, en geeft ze allebei terug, de vectorizer en het model. De check traint je functie op de 300 gesorteerde vragen, `texts` en `labels`, en op de eerste 60, en sorteert de 120 testvragen met wat hij teruggeeft; het plaatje zet je model op de lijn van de course.
        """),
        L.task("Maak `X` met de vectorizer, en train `model` op `X` en `labels`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train_small", lines=10)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train_small", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train_small",
        [
            "Stap 2 en 3 van het recept: eerst vectoren van de teksten, dan leert het model van die vectoren en de labels.",
            "De vectorizer leert de woorden en maakt de vectoren in één keer, met `fit_transform`; het model leert met `fit`, van de vectoren en de goede labels.",
            "`X = vectorizer.fit_transform(texts)`, en op de regel met de puntjes `model.fit(..., labels)`, met de vectoren op de plaats van de puntjes.",
        ],
        why="""```python
def train_small(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    return vectorizer, model
```

De functie geeft ook de vectorizer terug: nieuwe teksten hebben vectoren met dezelfde woorden nodig voordat het model ze kan sorteren.""",
    )
    return


@app.cell(hide_code=True)
def text_f1():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De macro F1

        `macro_f1(true, predicted, labels)` telt al voor elk label de true positives `tp`, false positives `fp` en false negatives `fn`. Reken daaruit de F1 van dat label uit, zoals in de les over de F1 score, en geef het gemiddelde van al die F1 scores terug. De check probeert een klein voorbeeld, de labels van Qwen3, die van het kleine model, en een model dat altijd goed is.
        """),
        L.task("Voeg de F1 van elk label toe aan `scores`, en geef het gemiddelde van `scores` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("macro_f1", lines=12)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("macro_f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "macro_f1",
        [
            "Twee stappen: de F1 van één label in de loop, het gemiddelde van allemaal erna.",
            "De F1 uit de tellingen is `2 * tp / (2 * tp + fp + fn)`; `np.mean` geeft het gemiddelde van een lijst.",
            "`scores.append(2 * tp / (...))`, met `2 * tp + fp + fn` op de plaats van de puntjes, en `return np.mean(...)`, met de lijst met scores.",
        ],
        why="""```python
def macro_f1(true, predicted, labels):
    true, predicted = np.array(true), np.array(predicted)
    scores = []
    for label in labels:
        tp = np.sum((true == label) & (predicted == label))
        fp = np.sum((true != label) & (predicted == label))
        fn = np.sum((true == label) & (predicted != label))
        scores.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(scores)
```

Elk label is één keer ja, en de rest nee; het gemiddelde laat elk label even zwaar tellen.""",
    )
    return


@app.cell(hide_code=True)
def text_when():
    mo.md(r"""
    ### Oefening 3. Wanneer wint het LLM?
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when-llm",
        question="Een winkel wil de reviews van zijn producten op onderwerp sorteren. De onderwerpen veranderen elke maand, en niemand heeft tijd om voorbeelden te sorteren. Wat past?",
        options={
            "small": "Een klein model, getraind op de gesorteerde reviews van vorige maand",
            "nothing": "Niets: zonder voorbeelden kan geen model sorteren",
            "llm": "Een LLM met een prompt die de onderwerpen van deze maand opsomt",
        },
        correct="llm",
        why={
            "llm": "Een prompt verander je in een minuut en heeft geen voorbeelden nodig; een klein model zou elke maand nieuwe gesorteerde voorbeelden nodig hebben.",
            "small": "De onderwerpen van vorige maand zijn niet die van deze maand. Waarvan zou het kleine model de nieuwe moeten leren?",
            "nothing": "Qwen3 sorteerde 99 van de 120 vragen zonder één voorbeeld. Wat kreeg het in plaats daarvan?",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_test_set():
    mo.md(r"""
    ### Oefening 4. Waarom de testvragen?
    """)
    return


@app.cell(hide_code=True)
def choice_test_set_cell():
    choice_test_set = L.choice(
        "test-set",
        question="Waarom meet de course het kleine model op de 120 testvragen, en niet op de 300 waarvan het leerde?",
        options={
            "seen": "Op de 300 zou het er beter uitzien dan het is, omdat het die al gezien heeft",
            "more": "Omdat 120 vragen meer zijn dan 300",
            "faster": "Omdat meten op minder vragen sneller gaat",
        },
        correct="seen",
        why={
            "seen": "Een model kan zijn trainingsvragen uit zijn hoofd leren; alleen vragen die het nooit zag, laten zien hoe het het doet op het volgende bericht van een klant.",
            "more": "120 is minder dan 300. Wat is er anders aan de 120?",
            "faster": "Allebei duurt het minder dan een seconde. Wat is er anders aan de 120?",
        },
    )
    choice_test_set
    return (choice_test_set,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Train je kleine model op minder of meer vragen

    Zodra oefening 1 goed is, leert je `train_small` hier van de eerste vragen van elk label, zoveel als de schuif zegt, en zie je waar het uitkomt tegenover Qwen3. Schrijf ook een eigen bericht, en kijk welk label het krijgt. Probeer een bericht in het Nederlands: het kleine model kent alleen de woorden van zijn Engelse vragen.
    """)
    return


@app.cell(hide_code=True)
def slider_cell():
    n_slider = mo.ui.slider(start=2, stop=50, step=1, value=10, label="Gesorteerde vragen per label", show_value=True)
    message_box = mo.ui.text(value="Someone took my card out of my bag.", label="Een eigen bericht", full_width=True)
    mo.vstack([n_slider, message_box])
    return message_box, n_slider


@app.cell(hide_code=True)
def demo_answer(box_train, message_box, n_slider):
    _train = C.learner_train(box_train.value)
    if _train is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo traint je eigen `train_small`.*")
    else:
        _words = {"title": "Je kleine model, op de eerste vragen van elk label",
                  "note": "Getraind op {n} vragen per label, {total} in totaal, in {s} seconden: macro F1 {f1} op de 120 testvragen (de cirkel), tegenover {q} van Qwen3.",
                  "label": "Je bericht krijgt het label",
                  "failed": "Je train_small stopte met een fout: {why}"}
        _out = mo.Html(C.demo(_train, n_slider.value, message_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De 120 vragen en de antwoorden van Qwen3 komen uit [Kant-en-klare AI: zien, spraak, vertalen en klassieke NLP](/course/open/build-ai-services?lang=nl); `fit` en `predict` uit [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl), en de F1 score uit [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl). De prijs is geteld zoals in [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl). Latere lessen van de module brengen het model naar je eigen apparaat en in de browser, waar een klein model het best past.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_set, choice_when, result_f1, result_train):
    L.wrap_up(result_train, result_f1, choice_when, choice_test_set)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
