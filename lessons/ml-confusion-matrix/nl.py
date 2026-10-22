# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-confusion-matrix. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-confusion-matrix", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_accuracy():
    quick_accuracy = L.choice(
        "quick-accuracy",
        question="Een model voorspelt de klassen 0, 1, 1 en 2 voor vier foto's met de labels 0, 1, 2 en 2. Wat is zijn accuracy?",
        options={
            "right": "0,75",
            "count": "3",
            "half": "0,5",
        },
        correct="right",
        why={
            "count": "Dat is hoeveel er goed zijn. Accuracy is een deel. Zie *Het idee* in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "half": "Vergelijk de voorspellingen één voor één met de labels: hoeveel kloppen er? Zie het recept in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "right": "Drie van de vier kloppen: 0,75.",
        },
    )
    quick_accuracy
    return (quick_accuracy,)


@app.cell(hide_code=True)
def quick_check_metric():
    quick_metric = L.choice(
        "quick-metric",
        question="Waarom meld je een metric zoals accuracy, en niet de cross-entropy waarop het model trainde?",
        options={
            "slope": "Omdat de metric betere hellingen heeft voor gradient descent",
            "lower": "Omdat de metric altijd lager is",
            "right": "Omdat de metric zegt hoe goed het model is in termen waar mensen om geven",
        },
        correct="right",
        why={
            "slope": "Het is andersom: de loss heeft de hellingen. Zie *Het idee* in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "lower": "Een deel en een loss staan helemaal niet op dezelfde schaal. Zie *Het idee* in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "right": "Je traint op de loss, die hellingen heeft, en meldt wat de mensen die het model gebruiken willen weten.",
        },
    )
    quick_metric
    return (quick_metric,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Accuracy gooit alle fouten op één hoop. Maar een spamfilter kan op twee manieren de fout ingaan: het kan spam doorlaten, of een mail weggooien die je nodig had. Die fouten kosten niet hetzelfde, en als de ene klasse veel zeldzamer is dan de andere, kan accuracy er goed uitzien terwijl het filter niets nuttigs doet. Om te zien wat een model echt doet, tel je welke klassen het door elkaar haalt.

    **Een voorbeeld uit het dagelijks leven.** Een metaaldetector op een vliegveld moet piepen bij elk mes. Maar als hij ook piept bij elke riemgesp, letten de medewerkers er niet meer op. Alle messen vinden en alleen bij messen piepen zijn twee verschillende doelen.
    """)
    return


@app.cell(hide_code=True)
def predict_raise():
    guess_raise = L.guess(
        "raise",
        question="Een spamfilter geeft elke mail een kans op spam, en noemt hem spam vanaf 0,5. Wat gebeurt er volgens jou als het een mail pas vanaf 0,8 spam noemt?",
        options={
            "both": "Het laat minder spam door, en gooit minder mails weg die je nodig hebt",
            "same": "Er verandert niets: de kansen blijven hetzelfde",
            "right": "Het laat meer spam door, maar gooit minder mails weg die je nodig hebt",
        },
        reveal="**Het laat meer spam door, maar gooit minder mails weg die je nodig hebt.** In het uitgewerkte voorbeeld hieronder vangt het filter met 0,8 in plaats van 0,5 nog maar 4 in plaats van 7 van de 10 spammails, en noemt het 0 in plaats van 3 andere mails ten onrechte spam. De accuracy is beide keren 0,85: daaraan zie je het verschil niet. Welke fout meer kost, bepaalt waar je de grens legt.",
    )
    guess_raise
    return (guess_raise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Noem bij twee klassen de klasse die je zoekt **positief**: hier spam. Elke mail komt dan in een van de vier vakjes terecht: een **true positive** is spam die het filter spam noemt, een **false positive** is een mail die geen spam is maar wel spam genoemd wordt, een **false negative** is spam die het filter mist, en een **true negative** is een mail die geen spam is en ook geen spam genoemd wordt. Zet de aantallen in een tabel, met een rij voor elke echte klasse en een kolom voor elke voorspelde klasse, en je hebt een **confusion matrix**. Bij $K$ klassen is hij $K$ bij $K$, en de goede voorspellingen liggen op de diagonaal.

        Uit de vakjes komen twee metrics:

        - **Precision**: van de mails die het filter spam noemt, het deel dat echt spam is: $\dfrac{TP}{TP + FP}$. Hoge precision betekent weinig vals alarm.
        - **Recall**: van de mails die echt spam zijn, het deel dat het filter vindt: $\dfrac{TP}{TP + FN}$. Hoge recall betekent dat er weinig spam doorglipt.

        Een filter dat geen enkele mail spam noemt, heeft helemaal geen precision: $TP + FP$ is 0, en door 0 kun je niet delen.

        Een filter geeft elke mail een kans op spam, en noemt hem spam vanaf een **decision threshold**, vaak 0,5. Verhoog je de threshold, dan noemt het filter minder mails spam: recall kan alleen maar dalen, en precision stijgt meestal, al zakt hij telkens even als er een spammail onder de threshold valt. Wat zwaarder weegt, hangt af van welke fout meer kost; de threshold kies je op de validation set.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **true positive**, **false positive**, **false negative**, **true negative**; $TP$, $FP$, $FN$, $TN$ | | de vier uitkomsten van een voorspelling bij twee klassen: positief of negatief voorspeld, en goed (true) of fout (false) | `tp`, `fp`, `fn`, `tn` |
        | **confusion matrix** | | een tabel die voor elke echte klasse (rij) telt hoe vaak elke klasse voorspeld werd (kolom) | `cm[label, pred]` |
        | **precision** | | van de voorbeelden die positief voorspeld zijn, het deel dat positief is | `tp / (tp + fp)` |
        | **recall** | | van de voorbeelden die positief zijn, het deel dat positief voorspeld is | `tp / (tp + fn)` |
        | **decision threshold** | | de kleinste kans waarbij een model de positieve klasse voorspelt | `p >= 0.5` |
        """),
        L.watch_out("Als deze course alleen precision zegt, bedoelt hij deze metric. De precisie van getallen in een computer, hoeveel cijfers die bewaart, is iets anders: numerieke precisie."),
        mo.md(r"""
        Veertig mails, waarvan tien spam, elk met de kans op spam die een filter eraan gaf. Schuif met de slider, of sleep over de grafiek van precision en recall, om de decision threshold te zetten. Onder de slider staat de confusion matrix, met geen spam als klasse 0 en spam als klasse 1, dus de true positives staan rechtsonder, in `cm[1, 1]`. Weer daaronder staat de grafiek: precision (doorgetrokken lijn) en recall (gestreepte lijn) voor elke threshold.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        t = round(t, 2)
        c = C.counts(t)
        s = C.scores(t)
        precision = "—" if s["precision"] is None else L.number(s["precision"], 2)
        return (draw.array_grids([("aantallen", [[str(c["tn"]), str(c["fp"])], [str(c["fn"]), str(c["tp"])]],
                                   ["cm[label, pred]", "rijen: het label, 0 geen spam, 1 spam", "kolommen: de voorspelling, 0 geen spam, 1 spam"])],
                                 indices=True, title="Confusion matrix").text
                + C.picture(t, {"title": "Precision en recall", "threshold": "decision threshold", "share": "deel",
                                "precision": "precision", "recall": "recall",
                                "alt": f"Precision stijgt, met kleine dipjes, van ongeveer 0,33 naar 1 als de threshold omhoog gaat, en recall daalt van 1 naar 0,1. Bij de threshold {L.number(t, 2)}: precision {precision}, recall {L.number(s['recall'], 2)}."}),
                f"""Met de threshold {L.number(t, 2)}: true positives {c["tp"]}, false positives {c["fp"]}, false negatives {c["fn"]}, true negatives {c["tn"]}. Precision <span class="ea-num">{precision}</span>, recall <span class="ea-num">{L.number(s["recall"], 2)}</span>, accuracy <span class="ea-num">{L.number(s["accuracy"], 3)}</span>.""")

    explore_threshold = L.explore("threshold", "threshold", _frame, start=0.5, low=0.05, high=0.95, step=0.05, role="hyper",
                                  task="Zoek de hoogste threshold die nog minstens 8 van de 10 spammails vangt. Wat is de precision daar?")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Om minstens 8 van de 10 spammails te vangen, kan de threshold tot 0,45 omhoog. Daar is de precision 0,73: van de 11 mails die het filter spam noemt, zijn er 3 geen spam. Een hogere threshold laat meer spam door en verhoogt meestal de precision, maar niet altijd: bij 0,5 valt de spammail met 0,45 onder de threshold, en zakt de precision naar 0,70.

    ### Recept

    Zo zie je wat een classifier met twee klassen echt doet:

    1. Kies de positieve klasse, de klasse die je zoekt, en een decision threshold.
    2. Tel de vier vakjes: true positives, false positives, false negatives en true negatives.
    3. Reken precision uit, $TP / (TP + FP)$, en recall, $TP / (TP + FN)$.
    4. Schuif de threshold om de een tegen de ander in te ruilen, afhankelijk van welke fout meer kost, en kies hem op de validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De veertig mails uit de figuur: de kansen die het filter gaf aan de tien spammails, en aan de dertig andere mails. Dan de threshold 0,5, en 0,8. `np.sum(a)` doet hetzelfde als `a.sum()`: het telt alle getallen van `a` op. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_confusion():
    import numpy as np

    _spam = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2])
    _ham = np.array([0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08,
                     0.08, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01])

    # 1. de positieve klasse is spam; twee thresholds
    for _t in [0.5, 0.8]:
        # 2. de vier vakjes: de som van True en False telt de True's
        _tp = np.sum(_spam >= _t)
        _fn = np.sum(_spam < _t)
        _fp = np.sum(_ham >= _t)
        _tn = np.sum(_ham < _t)
        print(_t, _tp, _fn, _fp, _tn)
        # 3. precision en recall, en de accuracy om te vergelijken
        print(_tp / (_tp + _fp), _tp / (_tp + _fn), (_tp + _tn) / 40)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de threshold 0,5 vindt het filter 7 van de 10 spammails en noemt het 3 andere mails ten onrechte spam: precision 0,7, recall 0,7. Met 0,8 vindt het er maar 4, maar elke mail die het spam noemt, is spam: precision 1,0, recall 0,4. De accuracy is beide keren 0,85, en zegt niets over dit verschil.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_never():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een filter dat nooit spam zegt
    """)
    return


@app.cell(hide_code=True)
def choice_never_cell():
    choice_never = L.choice(
        "never-spam",
        question="Van 100 mails zijn er 5 spam, en een filter zegt nooit spam. Welk getal laat meteen zien dat het nutteloos is?",
        options={
            "accuracy": "Zijn accuracy, 0,95",
            "right": "Zijn recall, 0",
            "tn": "Zijn 95 true negatives",
        },
        correct="right",
        why={
            "accuracy": "0,95 ziet er goed uit. Welk getal telt hoeveel van de spam het vindt? Zie *Het idee*.",
            "tn": "Dat zijn de mails die het goed heeft zonder iets te doen. Zie *Het idee*.",
            "right": "Van de 5 spammails vindt het er geen: recall 0. Zijn precision is niet eens uit te rekenen, omdat het geen enkele mail spam noemt.",
        },
    )
    choice_never
    return (choice_never,)


@app.cell(hide_code=True)
def text_exercise_precision():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Precision en recall

        Twaalf mails: `pred` zegt welke een filter spam noemt (1) of niet (0), en `y` welke spam zijn. Omdat ze allebei alleen 0 en 1 bevatten, is `pred * y` precies 1 waar ze allebei 1 zijn, dus `tp = np.sum(pred * y)` telt de true positives. Tel op dezelfde manier de false positives in `fp` en de false negatives in `fn`, en zet daarna precision in `precision` en recall in `recall`. Vervang de vier `...`. De laatste regel laat ze allebei zien, zodat de check ze ziet. De check probeert ook andere voorspellingen.
        """),
        L.task("Zet de precision en de recall van het filter in `precision` en `recall`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision():
    box_precision = L.code_box("precision-recall", lines=10)
    box_precision
    return (box_precision,)


@app.cell(hide_code=True)
def check_precision(box_precision):
    result_precision = L.check("precision-recall", box_precision.value)
    result_precision
    return (result_precision,)


@app.cell(hide_code=True)
def hints_precision():
    L.hints(
        "precision-recall",
        [
            "Stap 2 en 3 van het recept: eerst de vakjes, dan de twee delen.",
            "Een false positive heeft `pred` 1 en `y` 0, dus daar is `1 - y` gelijk aan 1. Welk product is precies 1 bij een false negative? En welke vakjes komen onder de streep bij precision, en bij recall?",
            "`fp = np.sum(pred * (1 - y))`, `fn = np.sum(___)`, `precision = tp / (tp + ___)` en `recall = tp / (tp + ___)`.",
        ],
        why="Het filter noemt 5 mails spam, en 4 daarvan zijn spam: 4 true positives en 1 false positive, dus precision is 4 / 5 = 0,8. Van de 6 spammails mist het er 2: recall is 4 / 6, ongeveer 0,67.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_matrix():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een confusion matrix voor drie klassen

        Tien foto's van katten (0), honden (1) en vogels (2): hun labels in `labels`, en wat een model voorspelde in `preds`. De matrix `cm` begint met allemaal nullen. Tel in de loop 1 op bij het vakje in de rij van de echte klasse en de kolom van de voorspelde klasse. Vervang de `...`. De laatste regel is `cm`, zodat de check het ziet. De check probeert ook andere voorspellingen.
        """),
        L.task("Tel elke foto in zijn vakje van de confusion matrix `cm`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matrix():
    box_matrix = L.code_box("matrix", lines=9)
    box_matrix
    return (box_matrix,)


@app.cell(hide_code=True)
def check_matrix(box_matrix):
    result_matrix = L.check("matrix", box_matrix.value)
    result_matrix
    return (result_matrix,)


@app.cell(hide_code=True)
def hints_matrix():
    L.hints(
        "matrix",
        [
            "Eén regel in de loop: één foto, één vakje, plus 1.",
            "Foto `i` hoort in rij `labels[i]` en kolom `preds[i]`. Hoe koos je één element van een matrix op zijn rij en kolom?",
            "`cm[labels[i], preds[i]] = ___ + 1`.",
        ],
        why="De katten zijn allemaal goed: 3 linksboven. De honden: 2 goed en 1 aangezien voor een vogel. De vogels: 2 goed, 1 aangezien voor een kat en 1 voor een hond.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl) was accuracy de metric; de confusion matrix laat zien wat erachter zit, en precision en recall laten zien wat hij verbergt als één klasse zeldzaam is. De decision threshold is nog een instelling die je op de validation set kiest. In de volgende les, *Precision en recall in één getal: de F1 score*, vat je die twee samen in één getal dat eerlijk blijft als één klasse zeldzaam is. En in het project met handgeschreven cijfers laat een confusion matrix van tien klassen zien welke cijfers je model door elkaar haalt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_never, result_precision, result_matrix):
    L.wrap_up(choice_never, result_precision, result_matrix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
