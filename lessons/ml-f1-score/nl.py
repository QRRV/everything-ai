# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-f1-score. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-f1-score", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="Een filter noemt 8 mails spam, en 6 daarvan zijn spam. Het mist 4 andere spammails. Wat zijn de precision en de recall?",
        options={
            "swapped": "Precision 0,6, recall 0,75",
            "right": "Precision 0,75, recall 0,6",
            "all": "Precision 0,5, recall 0,6",
        },
        correct="right",
        why={
            "swapped": "Andersom. Precision deelt door de mails die het filter spam noemt, recall door de mails die spam zijn. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "all": "Precision kijkt alleen naar de 8 mails die het filter spam noemt. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "right": "6 van de 8 mails die spam genoemd worden, zijn spam: precision 0,75. Het filter vindt 6 van de 10 spammails: recall 0,6.",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="Een filter noemt een mail spam vanaf een kans van 0,5. Je zet de decision threshold hoger, op 0,7. Wat gebeurt er met de recall?",
        options={
            "up": "Die stijgt",
            "same": "Die blijft altijd gelijk",
            "right": "Die daalt, of blijft gelijk",
        },
        correct="right",
        why={
            "up": "Minder mails halen een hogere threshold, dus het filter kan niet meer spam vinden dan eerst. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "same": "Hij blijft alleen gelijk als geen enkele spammail een kans van 0,5 tot 0,7 heeft. Zie de grafiek in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "right": "Er worden minder mails spam genoemd, dus er kan geen spammail gevonden worden die eerst gemist werd.",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les had een filter dat nooit spam zegt een accuracy van 0,95, op 100 mails met 5 spam. Als één klasse zeldzaam is, kan accuracy een model dat niets vindt zelfs hoger zetten dan een model dat bijna alles vindt. Precision en recall laten zien wat een model doet, maar het zijn twee getallen, en de threshold ruilt de een in tegen de ander. Om twee modellen te vergelijken, of twee thresholds, wil je één getal dat alleen hoog is als ze allebei hoog zijn.

    **Een voorbeeld uit het dagelijks leven.** Een rookmelder die nooit afgaat, heeft het bijna elke dag van het jaar goed, en is nutteloos op de ene dag dat er brand is. Een rookmelder die afgaat zodra je een tosti maakt, mist geen enkele brand, maar al snel luistert niemand er meer naar. Een goede rookmelder doet allebei: hij gaat af bij brand, en is verder stil.
    """)
    return


@app.cell(hide_code=True)
def predict_everything():
    guess_everything = L.guess(
        "everything",
        question="1.000 handgeschreven cijfers, waarvan 80 een 5. Een detector noemt elk cijfer een 5: zijn recall is 1, en zijn precision 0,08. Het gewone gemiddelde van die twee is 0,54. Wat zou één eerlijk getal voor allebei volgens jou over deze detector moeten zeggen?",
        options={
            "mean": "Ongeveer 0,54: de helft is perfect",
            "right": "Iets dicht bij 0,08: zijn zwakste getal houdt hem laag",
            "one": "1: hij vindt elke 5",
        },
        reveal="**Iets dicht bij zijn zwakste getal.** Zijn F1 score is 0,15. Eén getal voor allebei moet laag zijn zodra een van de twee laag is; anders levert elk cijfer een 5 noemen al meer dan de helft van de hoogste score op.",
    )
    guess_everything
    return (guess_everything,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **F1 score** vat precision $P$ en recall $R$ samen in één getal:
        """),
        L.formula(r"F_1 = \frac{2 \cdot P \cdot R}{P + R}"),
        mo.md(r"""
        Dat is hun **harmonisch gemiddelde**: bij twee getallen twee keer hun product, gedeeld door hun som. Anders dan het gewone gemiddelde blijft het dicht bij de kleinste van de twee: het ligt altijd tussen het kleinste getal en twee keer het kleinste. Het is alleen 1 als ze allebei 1 zijn, en 0 zodra een van de twee 0 is.

        Met de vakjes van de confusion matrix is het $F_1 = 2\,TP / (2\,TP + FP + FN)$. De true negatives zitten er niet in. Daarom misleidt F1 niet als één klasse veel zeldzamer is dan de andere, wat **class imbalance** heet, zolang de zeldzame klasse de positieve is: de vele voorbeelden van de veelvoorkomende klasse die een model terecht met rust laat, maken de accuracy hoog, maar tellen voor F1 niet mee. Een model dat de zeldzame klasse nooit voorspelt, heeft 0 true positives, dus zijn F1 score is 0, ook al kun je zijn precision niet uitrekenen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **F1 score**, $F_1$ | F één | het harmonisch gemiddelde van precision en recall; alleen hoog als ze allebei hoog zijn | `2 * tp / (2 * tp + fp + fn)` |
        | **harmonisch gemiddelde** | | bij twee getallen $a$ en $b$: $2ab / (a + b)$ | |
        | **class imbalance** | | één klasse is veel zeldzamer dan de andere, zoals de vijven tussen alle cijfers | |

        Sleep de stip over het vlak, of verplaats hem met de pijltjestoetsen: precision van links naar rechts, recall van onder naar boven, en de balken ernaast volgen.
        """),
    ])
    return


@app.cell(hide_code=True)
def plane_f1_cell():
    def _frame(_p, _r):
        _mean, _f1 = (_p + _r) / 2, C.f1(_p, _r)
        _words = {"title": "Precision, recall en twee gemiddelden", "precision": f"precision {L.number(_p)}", "recall": f"recall {L.number(_r)}",
                  "mean": f"gewoon gemiddelde {L.number(_mean)}", "f1": f"F1 score {L.number(_f1)}",
                  "alt": f"Vier balken van 0 tot 1: precision {L.number(_p)}, recall {L.number(_r)}, hun gewone gemiddelde {L.number(_mean)} en hun F1 score {L.number(_f1)}."}
        return (C.bars(_p, _r, _words),
                f"""Precision {L.number(_p)} en recall {L.number(_r)}: het gewone gemiddelde is <span class="ea-num">{L.number(_mean)}</span>, de F1 score <span class="ea-num">{L.number(_f1)}</span>.""")

    _plane = L.plane("precision-recall", "precision", "recall", _frame, C.STOPS, C.STOPS, start=(0.9, 0.3), role="",
                     task="Zoek een punt waar het gewone gemiddelde van precision en recall 0,5 of meer is, maar de F1 score onder 0,2.")
    _plane
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Zulke punten liggen bij twee hoeken: het ene getal 0,9 of meer, het andere 0,1 of minder. F1 blijft dicht bij het kleinste getal, en is alleen hoog in de hoek rechtsboven, waar ze allebei hoog zijn. Op de linker- en de onderrand ligt geen echte detector, behalve in hun hoek: een 0 voor een van de twee betekent 0 true positives, en dan is de ander ook 0, of niet uit te rekenen.

    ### Recept

    Zo beoordeel je een classifier met zijn F1 score:

    1. Tel de true positives, false positives en false negatives.
    2. Reken precision uit, $TP / (TP + FP)$, en recall, $TP / (TP + FN)$.
    3. Reken $F_1 = 2PR / (P + R)$ uit, of rechtstreeks uit de aantallen $2\,TP / (2\,TP + FP + FN)$, wat ook werkt als je precision niet kunt uitrekenen.
    4. Is één klasse zeldzaam, maak die dan de positieve klasse, en vergelijk modellen, of thresholds op de validation set, met F1 in plaats van accuracy.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier detectoren voor vijven, op 1.000 handgeschreven cijfers van een validation set uit QMNIST, waarvan 80 een 5: een die nooit 5 zegt; een logistische regressie die de course trainde op 5.000 andere cijfers, één keer met de decision threshold 0,05 en één keer met 0,5; en een die elk cijfer een 5 noemt. Bij elke detector de aantallen $TP$, $FP$, $FN$ en $TN$. `for _name, _tp, _fp, _fn, _tn in _detectors:` geeft elke waarde van elke tuple een eigen naam, zoals `x, y = point` doet. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_f1():
    # 1. de aantallen van elke detector
    _detectors = [
        ("nooit 5", 0, 0, 80, 920),
        ("threshold 0.05", 77, 120, 3, 800),
        ("threshold 0.5", 52, 3, 28, 917),
        ("altijd 5", 80, 920, 0, 0),
    ]
    for _name, _tp, _fp, _fn, _tn in _detectors:
        # 2. precision, als de detector ooit 5 zegt, en recall
        if _tp + _fp == 0:
            _precision = None
        else:
            _precision = _tp / (_tp + _fp)
        _recall = _tp / (_tp + _fn)
        # 3. F1 uit de aantallen, en de accuracy om te vergelijken
        _f1 = 2 * _tp / (2 * _tp + _fp + _fn)
        print(f"{_name}:", _precision, _recall, _f1, (_tp + _tn) / 1000)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke regel geeft de detector, zijn precision, recall, F1 score en accuracy. Accuracy zet de detector die nooit 5 zegt, met 0,92, boven die met de threshold 0,05, met 0,877, terwijl die tweede 77 van de 80 vijven vindt. F1 zet ze in de goede volgorde: 0 tegen 0,56. Met de threshold 0,5 vindt de detector minder vijven, 52, maar geeft hij maar 3 keer vals alarm: de hoogste F1 score van de vier, 0,77. Dat werkt alleen met de vijven als positieve klasse: met de andere cijfers als positief zou de detector die nooit 5 zegt een F1 score van 0,96 krijgen, en die met de threshold 0,05 maar 0,93.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lopsided():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Dicht bij de kleinste
    """)
    return


@app.cell(hide_code=True)
def choice_lopsided_cell():
    choice_lopsided = L.choice(
        "lopsided",
        question="Een detector heeft een precision van 0,2 en een recall van 0,8. Hoe groot is zijn F1 score?",
        options={
            "mean": "0,5",
            "right": "0,32",
            "product": "0,16",
        },
        correct="right",
        why={
            "mean": "Dat is het gewone gemiddelde: één hoog getal trekt het omhoog, hoe laag het andere ook is. F1 blijft dicht bij het kleinste getal. Zie *Het idee*.",
            "product": "Dat is het product van de twee. F1 is twee keer het product, gedeeld door de som. Zie *Het idee*.",
            "right": "2 · 0,2 · 0,8 / (0,2 + 0,8) = 0,32: veel dichter bij 0,2 dan bij 0,8.",
        },
    )
    choice_lopsided
    return (choice_lopsided,)


@app.cell(hide_code=True)
def text_exercise_f1():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De F1 score van een detector

        Twintig cijfers: `pred` zegt welke een detector een 5 noemt (1) of niet (0), en `y` welke een 5 zijn. Net als in de vorige les telt `tp = np.sum(pred * y)` de true positives. Tel de false positives in `fp` en de false negatives in `fn`, en zet de F1 score in `f1`. Vervang de drie `...`. De laatste regel is `f1`, zodat de check hem ziet. De check probeert ook andere cijfers.
        """),
        L.task("Zet de F1 score van de detector in `f1`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("f1", lines=9)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "f1",
        [
            "Stap 1 en 3 van het recept: eerst de aantallen, dan F1.",
            "Een false positive heeft `pred` 1 en `y` 0, een false negative `pred` 0 en `y` 1. Voor F1 uit de aantallen hoef je precision en recall niet eerst uit te rekenen.",
            "`fp = np.sum(pred * (1 - y))`, `fn = np.sum(___)` en `f1 = 2 * tp / (___)`.",
        ],
        why="De detector noemt 8 cijfers een 5, en 3 daarvan zijn het: 3 true positives en 5 false positives. Hij mist 1 van de 4 vijven. F1 = 2 · 3 / (2 · 3 + 5 + 1) = 0,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rare():
    mo.md(r"""
    ### Oefening 3. Een zeldzame klasse
    """)
    return


@app.cell(hide_code=True)
def choice_rare_cell():
    choice_rare = L.choice(
        "rare-pick",
        question="Van 1.000 kaartbetalingen zijn er 10 fraude. Model A zegt nooit fraude. Model B vangt 8 van de 10, en wijst ten onrechte 40 eerlijke betalingen aan. Welk model vindt fraude beter, en welk getal laat dat zien?",
        options={
            "accuracy": "A: zijn accuracy is 0,99, tegen 0,958 voor B",
            "right": "B: zijn F1 score is ongeveer 0,28, tegen 0 voor A",
            "neither": "Geen van beide: ze zitten allebei weleens fout",
        },
        correct="right",
        why={
            "accuracy": "Accuracy telt de 990 eerlijke betalingen die A met rust laat zonder iets te doen. Welk getal laat de true negatives weg? Zie *Het idee*.",
            "neither": "Elk model maakt fouten; de vraag is welk model zijn werk beter doet. Reken F1 uit voor allebei: stap 3 van het recept.",
            "right": "B heeft 8 true positives, 40 false positives en 2 false negatives: F1 = 16 / 58, ongeveer 0,28. A vindt helemaal geen fraude: F1 0.",
        },
    )
    choice_rare
    return (choice_rare,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) lieten precision en recall zien wat accuracy verbergt; F1 vat ze samen in één getal dat de true negatives weglaat. Hij laat ze even zwaar wegen: kost de ene fout veel meer dan de andere, kijk dan naar die twee zelf, of naar de F-beta score, die recall zwaarder laat wegen dan precision, of juist lichter, en kies de threshold naar wat elke fout kost. En F1 hoort nog steeds bij één decision threshold. In de volgende les, *Alle thresholds tegelijk: de ROC curve en AUC*, bekijk je alle thresholds samen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lopsided, result_f1, choice_rare):
    L.wrap_up(choice_lopsided, result_f1, choice_rare)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
