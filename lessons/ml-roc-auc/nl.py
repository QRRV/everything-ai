# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-roc-auc. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-roc-auc", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="Een detector heeft een precision van 0,5 en een recall van 0,5. Wat is zijn F1 score?",
        options={
            "product": "0,25",
            "right": "0,5",
            "sum": "1",
        },
        correct="right",
        why={
            "product": "Dat is het product. F1 is twee keer het product gedeeld door de som. Zie *Het idee* in [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "sum": "Dat is de som. F1 ligt tussen precision en recall in, dicht bij de kleinste. Zie *Het idee* in [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "right": "2 · 0,5 · 0,5 / (0,5 + 0,5) = 0,5: als de twee gelijk zijn, is F1 dat getal ook.",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def quick_check_cells():
    quick_cells = L.choice(
        "quick-cells",
        question="Welke vakjes van de confusion matrix laat de F1 score weg?",
        options={
            "fp": "De false positives",
            "right": "De true negatives",
            "none": "Geen: hij gebruikt ze alle vier",
        },
        correct="right",
        why={
            "fp": "De false positives zitten erin: die trekken precision omlaag. Zie *Het idee* in [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "none": "Kijk naar 2 · TP / (2 · TP + FP + FN): er ontbreekt één vakje. Zie *Het idee* in [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "right": "F1 is 2 · TP / (2 · TP + FP + FN): de vele voorbeelden die een model terecht met rust laat, tellen niet mee.",
        },
    )
    quick_cells
    return (quick_cells,)




@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Precision, recall en F1 horen allemaal bij één decision threshold: verschuif hem, en ze veranderen. Terwijl je een model bouwt, heb je de threshold vaak nog niet gekozen, en twee filters kunnen van plaats wisselen als hij verschuift. Je wilt één getal voor hoe goed de kansen de spam boven de andere mails zetten, welke threshold je ook kiest.

    **Een voorbeeld uit het dagelijks leven.** Een docent kan dertig opstellen van beste naar slechtste op volgorde leggen, goed of slecht, voordat iemand de grens voor een voldoende heeft vastgesteld: de volgorde is het werk van de docent, de grens een aparte keuze.
    """)
    return


@app.cell(hide_code=True)
def predict_random():
    guess_random = L.guess(
        "random",
        question="Een filter geeft elke mail een willekeurige kans. Maak alle paren van één spammail en één mail die geen spam is. In welk deel van de paren krijgt de spammail volgens jou de hogere kans?",
        options={
            "none": "Bijna geen",
            "right": "Ongeveer de helft",
            "threshold": "Dat hangt af van de threshold",
        },
        reveal="**Ongeveer de helft.** Een willekeurige kans zet de spammail bij toeval in de helft van de paren voorop. Dat deel is de AUC: 0,5 voor gokken, 1 als elke spammail boven elke andere mail staat, en 0,935 voor het filter van de veertig mails. Er komt geen threshold aan te pas.",
    )
    guess_random
    return (guess_random,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Elk van de veertig mails uit [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) heeft de kans op spam die het filter eraan gaf; tien ervan zijn spam. Reken bij elke threshold twee delen uit:

        - de **true positive rate**: van de spammails, het deel dat spam genoemd wordt, $TP / (TP + FN)$: recall, onder een andere naam;
        - de **false positive rate**: van de andere mails, het deel dat ten onrechte spam genoemd wordt, $FP / (FP + TN)$.

        Elke threshold is één punt, met de false positive rate naar rechts en de true positive rate omhoog. Verbind de punten, en je hebt de **ROC curve**: van (0, 0), waar niets spam genoemd wordt, tot (1, 1), waar alles spam genoemd wordt. Gaat de threshold omlaag, dan stapt de curve 1/10 omhoog voor elke spammail die de threshold passeert, en 1/30 naar rechts voor elke andere mail; een spammail en een andere mail met dezelfde kans maken één schuine stap. Een filter dat alle spam boven de andere mails zet, gaat eerst recht omhoog naar de hoek linksboven; gokken blijft dicht bij de diagonaal.

        De oppervlakte onder de curve is de **ROC AUC**, kortweg AUC: een getal van 0 tot 1. Elk van de 30 andere mails is een stap van 1/30 naar rechts, op de hoogte van het deel van de spammails erboven. De oppervlakte is dus het deel van de 10 × 30 paren van een spammail en een andere mail waarin de spammail de hogere kans heeft, waarbij een gelijke stand half telt. Hij hangt alleen af van de volgorde van de kansen, niet van een threshold.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **true positive rate**, TPR | | van de positieve voorbeelden, het deel dat positief voorspeld wordt: een andere naam voor recall | `tpr` |
        | **false positive rate**, FPR | | van de negatieve voorbeelden, het deel dat ten onrechte positief voorspeld wordt | `fpr` |
        | **ROC curve** | er oo cee | de true positive rate tegen de false positive rate, voor elke decision threshold | |
        | **ROC AUC**, AUC | aa uu cee | de oppervlakte onder de ROC curve: het deel van de paren van een positief en een negatief voorbeeld in de goede volgorde, waarbij een gelijke stand half telt | `auc` |
        | **precision-recall curve** | | precision tegen recall, voor elke decision threshold; bij een zeldzame klasse laat hij zien wat de ROC curve verbergt | |
        """),
        L.watch_out("ROC staat voor receiver operating characteristic, uit de radar van de Tweede Wereldoorlog. In de geneeskunde heet recall sensitiviteit en 1 min de false positive rate specificiteit, dus een ROC curve laat daar sensitiviteit zien tegen 1 − specificiteit."),
        mo.md(r"""
        Schuif met de slider, of sleep over de grafiek van de mails, om de threshold te zetten: elke mail rechts van de lijn wordt spam genoemd. De andere grafiek laat de ROC curve zien, de stip van die threshold, de gestreepte diagonaal van gokken en, in grijs, de oppervlakte onder de curve.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_roc_cell():
    def _frame(t):
        t = round(t, 2)
        r = C.rates(t)
        return (C.picture(t, {"mails": "De 40 mails", "probability": "kans op spam", "spam": "spam", "other": "geen spam",
                              "roc": f"ROC curve, AUC {L.number(C.auc(), 3)}", "fpr": "false positive rate", "tpr": "true positive rate", "guess": "gokken",
                              "alt": f"Eerste grafiek: de tien spammails als gevulde stippen in een rij bovenaan en de dertig andere mails als open stippen daaronder, naar hun kans, met de threshold {L.number(t, 2)} als lijn en {r['tp']} spammails en {r['fp']} andere mails rechts ervan. Tweede grafiek: de ROC curve, een trap ver boven de gestreepte diagonaal van gokken, de oppervlakte eronder in grijs, en een stip bij de false positive rate {L.number(r['fpr'], 2)} en de true positive rate {L.number(r['tpr'], 2)}."}),
                f"""Met de threshold {L.number(t, 2)} noemt het filter {r["tp"]} van de 10 spammails spam, een true positive rate van <span class="ea-num">{L.number(r["tpr"], 2)}</span>, en {r["fp"]} van de 30 andere mails, een false positive rate van <span class="ea-num">{L.number(r["fpr"], 2)}</span>.""")

    explore_roc = L.explore("threshold", "threshold", _frame, start=0.5, low=0, high=1, step=0.05, role="hyper",
                            task="Zoek de laagste threshold waarbij het filter hooguit 3 van de 30 andere mails spam noemt. Welk deel van de spam vangt het daar?")
    explore_roc
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De laagste is 0,45: daar noemt het filter 3 van de 30 andere mails spam, een false positive rate van 0,1, en vangt het 8 van de 10 spammails, een true positive rate van 0,8; bij 0,4 komt er een vierde andere mail over de lijn. De curve beantwoordt die vraag voor elke threshold tegelijk. De oppervlakte, de AUC van 0,935, beweegt niet mee met de slider.

    ### Recept

    De score van een classifier is het getal dat hij aan elk voorbeeld geeft, hier de kans op spam. Zo teken je zijn ROC curve en reken je zijn AUC uit:

    1. Neem elke threshold, van boven de hoogste score tot onder de laagste.
    2. Reken bij elke threshold de true positive rate en de false positive rate uit.
    3. Zet de true positive rate omhoog uit tegen de false positive rate naar rechts, en verbind de punten.
    4. De AUC is de oppervlakte eronder: het deel van de paren van een positief en een negatief voorbeeld waarin het positieve de hogere score heeft, waarbij een gelijke stand half telt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De veertig mails uit de figuur: zes thresholds van hoog naar laag, elk één punt van de curve, en daarna de AUC, uit de spammails boven elk van de 30 andere mails en die met dezelfde kans. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_roc():
    import numpy as np

    _spam = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2])
    _ham = np.array([0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.08,
                     0.08, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01])

    # 1. thresholds van hoog naar laag
    for _t in [1.0, 0.8, 0.6, 0.45, 0.2, 0.0]:
        # 2. de true positive rate en de false positive rate
        _tpr = np.sum(_spam >= _t) / len(_spam)
        _fpr = np.sum(_ham >= _t) / len(_ham)
        # 3. één punt van de curve: naar rechts, dan omhoog
        print(_t, _fpr, _tpr)
    # 4. de AUC: voor elke andere mail de spammails erboven en die met dezelfde kans; zo'n gelijke stand telt half
    _higher = 0
    _ties = 0
    for _h in _ham:
        _higher = _higher + np.sum(_spam > _h)
        _ties = _ties + np.sum(_spam == _h)
    _above = _higher + 0.5 * _ties
    print(_higher, _ties, _above, _above / (len(_spam) * len(_ham)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste zes regels zijn punten van de curve: threshold, false positive rate, true positive rate. De laatste regel: in 279 van de 300 paren heeft de spammail de hogere kans, en 3 paren staan gelijk, bij 0,35 en bij 0,2. Als een gelijke stand half telt, maakt dat 280,5: een AUC van 0,935.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Twee rates bij één threshold

        Twaalf mails, gesplitst naar hun label: `pos` bevat de kansen op spam van de 5 spammails, en `neg` die van de 7 andere. Zet de true positive rate bij de threshold `t` in `tpr`, en de false positive rate in `fpr`, zoals in het uitgewerkte voorbeeld. Vervang de twee `...`. De laatste regel laat ze allebei zien, zodat de check ze ziet. De check probeert ook een andere threshold.
        """),
        L.task("Zet de true positive rate in `tpr` en de false positive rate in `fpr`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("rates", lines=8)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "rates",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld: een aantal gedeeld door het aantal mails van één klasse.",
            "`np.sum(pos >= t)` telt de spammails die spam genoemd worden. Welke mails worden ten onrechte spam genoemd, en hoeveel mails heeft elke klasse?",
            "`tpr = np.sum(pos >= t) / len(___)` en `fpr = np.sum(___ >= t) / len(neg)`.",
        ],
        why="Bij 0,5 noemt het filter 4 mails spam: 3 van de 5 spammails, een true positive rate van 0,6, en 1 van de 7 andere, een false positive rate van ongeveer 0,14.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_auc():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De AUC door paren te tellen

        Dezelfde twaalf mails; één spammail en één andere mail hebben allebei 0,4. Tel in de lus voor elke andere mail `n` bij `above` de spammails op met een hogere kans, en de helft van die met dezelfde kans. Zet daarna de AUC in `auc`. Vervang de twee `...`. De laatste regel is `auc`, zodat de check hem ziet. De check probeert ook andere spammails.
        """),
        L.task("Zet de AUC van de twaalf mails in `auc`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_auc():
    box_auc = L.code_box("auc", lines=9)
    box_auc
    return (box_auc,)


@app.cell(hide_code=True)
def check_auc(box_auc):
    result_auc = L.check("auc", box_auc.value)
    result_auc
    return (result_auc,)


@app.cell(hide_code=True)
def hints_auc():
    L.hints(
        "auc",
        [
            "Stap 4 van het recept, zoals in het uitgewerkte voorbeeld.",
            "`np.sum(pos > n)` telt de spammails boven `n`, en `np.sum(pos == n)` die met dezelfde kans. Hoeveel paren van één spammail en één andere mail zijn er?",
            "`above = above + np.sum(pos > n) + 0.5 * ___` en `auc = above / (len(pos) * ___)`.",
        ],
        why="Van de 35 paren heeft de spammail in 26 de hogere kans, en één paar staat gelijk: 26,5 / 35, ongeveer 0,757.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ### Oefening 3. Alleen de volgorde
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="Een tweede filter geeft elk van de veertig mails de helft van de kans die het eerste filter eraan geeft. Wat is zijn AUC?",
        options={
            "half": "De helft: ongeveer 0,47",
            "right": "Dezelfde: 0,935",
            "threshold": "Dat hangt af van de threshold",
        },
        correct="right",
        why={
            "half": "Als je elke kans halveert, veranderen de getallen, maar blijft een spammail die boven een andere mail stond daar ook boven? Zie *Het idee*.",
            "threshold": "De AUC hoort bij geen enkele threshold: hij telt paren. Zie stap 4 van het recept.",
            "right": "Halveren verandert de volgorde van geen enkel paar, en de AUC kijkt alleen naar die volgorde. Bij de threshold 0,5 noemt het tweede filter echter geen enkele mail spam: geen kans komt nog tot 0,5.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De ROC curve laat alle thresholds uit [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) tegelijk zien. De F1 score uit [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) hoort bij één threshold; de AUC bij geen enkele, dus daarmee vergelijk je modellen voordat je een threshold kiest. Bij een zeldzame klasse deelt de false positive rate door de vele negatieve voorbeelden, dus hij blijft klein, ook als er meer vals alarm is dan treffers: de detector van vijven met de threshold 0,05 uit die les had een false positive rate van 0,13, terwijl 120 van zijn 197 alarmen vals waren. Kijk dan ook naar de precision-recall curve. In de volgende les, *Versla altijd een eenvoudige baseline*, vergelijk je een model met de eenvoudigste regel die je kunt bedenken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rates, result_auc, choice_order):
    L.wrap_up(result_rates, result_auc, choice_order)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
