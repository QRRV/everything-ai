# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-fair-thresholds. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-fair-thresholds", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) en [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_opportunity():
    quick_opportunity = L.choice(
        "quick-opportunity",
        question="Een model geeft een ja aan 0,44 van de vrouwen die meer verdienen en aan 0,57 van de mannen die meer verdienen. Aan welk criterium voldoet het niet?",
        options={
            "calibration": "Calibration within groups",
            "right": "Equal opportunity",
            "none": "Geen, als het deel met een ja gelijk is",
        },
        correct="right",
        why={
            "calibration": "Calibration within groups gaat over wat een score betekent: van de mensen met een kans van ongeveer 0,7 is ongeveer 70 procent positief. Deze getallen zijn recalls, het deel van de mensen die echt meer verdienen dat een ja krijgt. Zie *Het idee* in [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).",
            "right": "Equal opportunity vraagt dezelfde true positive rate, de recall, in elke groep: hier 0,44 tegen 0,57.",
            "none": "Equal opportunity kijkt niet naar het deel met een ja, alleen naar de recall, en 0,44 is geen 0,57. Zie *Het idee* in [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).",
        },
    )
    quick_opportunity
    return (quick_opportunity,)


@app.cell(hide_code=True)
def quick_check_reweighing():
    quick_reweighing = L.choice(
        "quick-reweighing",
        question="Door reweighing behandelde het model van UCI Adult vrouwen en mannen meer gelijk. Wat veranderde er?",
        options={
            "labels": "De labels van een deel van de vrouwen en de mannen",
            "right": "Hoe zwaar elke persoon meetelt bij het trainen",
            "threshold": "De thresholds voor de vrouwen en de mannen",
        },
        correct="right",
        why={
            "labels": "Labels veranderen is massaging. Reweighing houdt elk label zoals het is en geeft elke persoon een gewicht: voor hoeveel mensen die persoon telt terwijl het model leert. Zie *Het idee* in [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl).",
            "right": "Elke persoon kreeg een gewicht naar geslacht en label, en het model werd met die gewichten opnieuw getraind. De threshold bleef 0,5 voor iedereen.",
            "threshold": "De threshold bleef 0,5 voor iedereen. Reweighing gaf elke persoon een gewicht terwijl het model leerde, en vroeg om een nieuwe training. Zie *Het idee* in [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl).",
        },
    )
    quick_reweighing
    return (quick_reweighing,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) gaf een lagere threshold voor de vrouwen hun dezelfde recall als de mannen, en zat het model ongeveer even vaak goed als eerst. In [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl) brachten gewichten het deel met een ja dichter bij elkaar, maar moest het model opnieuw getraind worden. Dat kan vaak niet: het model komt van een ander team of van een leverancier, en je krijgt alleen de kans die het voor elke persoon geeft. Wat je dan nog kiest, is de threshold. Deze les zoekt voor elk fairness-criterium het paar thresholds dat de meeste mensen goed heeft, en meet wat elk paar kost en wat het ongelijk laat.

    **Een voorbeeld uit het dagelijks leven.** Een school geeft twee klassen hetzelfde vak, elk met een eigen versie van de toets, en één versie blijkt moeilijker. Om in beide klassen hetzelfde deel van de leerlingen die de stof beheersen te laten slagen, legt de school de grens voor een voldoende bij de moeilijke versie lager. Niemands antwoorden veranderen, alleen het cijfer dat als voldoende telt. Maar nu kunnen een leerling uit de ene klas en een leerling uit de andere met dezelfde score een andere uitslag krijgen.
    """)
    return


@app.cell(hide_code=True)
def predict_cost():
    guess_cost = L.guess(
        "cost",
        question="Met één threshold van 0,5 zit het model goed bij 0,834 van de mensen in de test set. Met het beste paar thresholds dat vrouwen en mannen dezelfde recall geeft: hoe vaak denk je?",
        options={
            "less": "0,78, duidelijk minder vaak",
            "right": "0,832, bijna even vaak",
            "more": "0,85, vaker dan eerst",
        },
        reveal="**0,832, bijna even vaak.** De vrouwen die van een nee naar een ja gaan, hebben een kans net onder 0,5, en ongeveer vier op de tien van hen verdienen echt meer: hun nieuwe ja is goed bij ongeveer vier op de tien van hen. Andere criteria kosten meer. De demo hieronder laat het zien.",
    )
    guess_cost
    return (guess_cost,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Na het trainen, op de kans.** Moritz Hardt, Eric Price en Nathan Srebro (2016) boden een oplossing voor een model dat al getraind is: in plaats van een misschien ingewikkelde training te veranderen, volstaat een eenvoudige stap achteraf, post-processing, met zo min mogelijk verlies aan nut. De nieuwe beslissing mag maar twee dingen gebruiken: de kans die het model een persoon geeft, en de groep van die persoon. Geen andere kolom is nodig, en geen weight van het model verandert. Met een threshold per groep vergelijk je de kans van een vrouw met de threshold van de vrouwen, en die van een man met die van de mannen.

    **Thresholds zoeken.** Kies een criterium, zoals gelijke recall, en hoe groot een verschil je toestaat, zoals kleiner dan 0,01. Probeer dan paren: een threshold voor de vrouwen en een voor de mannen, elk van 0,01 tot 0,99. Reken voor elk paar het verschil tussen de twee recalls uit. Houd de paren met een verschil onder 0,01, en neem daarvan het paar dat de meeste mensen goed heeft. Die zoektocht heet **threshold optimization**. Hoeveel een paar goed heeft, telt per groep op: de vrouwen die het goed heeft plus de mannen die het goed heeft. Je telt dus één keer per groep en threshold, en daarna tel je alleen op. De library Fairlearn heeft er een tool voor, `ThresholdOptimizer`: die kiest de thresholds zo dat de gekozen maat, zoals accuracy, zo goed mogelijk is, binnen de grenzen van het fairness-criterium. Volgens de handleiding probeert hij daarvoor alle mogelijke thresholds en kiest hij de beste combinatie.

    **Kies op het ene deel, meet op het andere.** Als je de thresholds op de test set kiest en daarna op dezelfde test set meet, lijken de groepen gelijker dan ze bij nieuwe mensen zullen zijn. Daarom zoekt de demo op de mensen waarop het model getraind is, en meet de test set alleen. Dat kan hier omdat het model de mensen waarop het trainde nauwelijks uit het hoofd leerde: met één threshold van 0,5 zit het bij 0,835 van hen goed en bij 0,834 van de test set. Bij een model dat dat wel doet, houd je een [validation set](/course/open/ml-validation-set?lang=nl) apart voor de zoektocht.

    **Ook gelijke false positive rates: een muntworp.** Met één threshold per groep kunnen gelijke recall en gelijke false positive rates samen alleen bij de paren waar ze toevallig allebei gelijk uitkomen; op UCI Adult vindt de demo die alleen bij hoge thresholds, waar weinig mensen een ja krijgen. Hardt en collega's staan daarom twee thresholds voor een groep toe: boven allebei een ja, onder allebei een nee, en daartussen beslist een muntworp. Ook de handleiding van Fairlearn zegt dat je voor een precieze gelijkheid meestal tussen twee thresholds moet loten. Dan kunnen twee mensen met dezelfde kans en uit dezelfde groep een ander antwoord krijgen. Deze les gebruikt geen muntworp.

    **Wat het nodig heeft: de groep, bij het beslissen.** De handleiding van Fairlearn benadrukt dat `ThresholdOptimizer` de groep van elke persoon nodig heeft op het moment dat het model in gebruik is. Barocas, Hardt en Narayanan noemen in *Fairness and Machine Learning* een veelgehoord bezwaar tegen post-processing: de beslissing gebruikt het lidmaatschap van een groep heel openlijk, met een andere threshold voor elke groep. Of dat mag, hangt af van het land en de beslissing; zie [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl). Het Algoritmekader van de overheid vraagt om na te gaan of de inputvariabelen leiden tot direct onderscheid op gronden als ras of geslacht: "Gebruik van deze factoren wordt dan gezien als direct onderscheid en is verboden." Het voegt eraan toe dat zo'n onderscheid niet altijd fout of verboden is, bijvoorbeeld als een wettelijke uitzondering het toelaat. Over het recht in de VS schrijven Barocas, Hardt en Narayanan dat het antwoord genuanceerd is.

    **Probeer het.** De cel hieronder traint het model zonder de kolom `sex` één keer, en gebruikt daarna alleen de kansen ervan. `_rates` telt voor één geslacht in één deel van de tabel en één threshold hoeveel het goed heeft, het deel met een ja, de recall en de false positive rate. Stap 1 vult `_table` voor elke threshold op de mensen om op te trainen. Stap 2 probeert alle 99 × 99 paren `_t_women` en `_t_men`: `_gap` bevat per criterium hoe ver het paar ervan af zit, en per criterium bewaart `_best` het paar met de meeste mensen goed, `_most`, van de paren met een verschil onder 0,01. Eén threshold van 0,5 voor iedereen blijft zoals hij is. Stap 3 meet elk gekozen paar op de test set. Voer de cel uit; dat duurt een paar seconden.
    """)
    return


@app.cell
def demo_search():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaanse enquête van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _rich = _people["income"] == ">50K"
    _poor = _people["income"] == "<=50K"

    # het model zonder de kolom sex, zoals in de vorige lessen: één keer getraind, en hieronder niet veranderd
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _rich[_train])
    _score = _model.predict_proba(_X)[:, 1]   # voor elke persoon de kans op een ja


    def _rates(part, sex, threshold):
        # één geslacht in één deel van de tabel, met één threshold: hoeveel het goed heeft, en drie delen
        group = (_people["split"] == part) * (_people["sex"] == sex)
        yes = _score > threshold
        return {"mensen": group.sum(), "goed": (yes == _rich)[group].sum(),
                "deel met een ja": yes[group].mean(), "recall": yes[group * _rich].mean(),
                "false positive rate": yes[group * _poor].mean()}


    # 1. per geslacht en per threshold van 0,01 tot 0,99, op de mensen om op te trainen
    _thresholds = np.arange(1, 100) / 100
    _table = {"Female": {}, "Male": {}}
    for _sex in _table:
        for _t in _thresholds:
            _table[_sex][_t] = _rates("train", _sex, _t)

    # 2. per criterium, van de paren met een verschil onder 0,01, het paar met de meeste mensen goed
    _best = {"één threshold": (0.5, 0.5), "gelijk deel met een ja": None, "gelijke recall": None,
             "gelijke recall en false positive rate": None}
    _most = {"gelijk deel met een ja": 0, "gelijke recall": 0, "gelijke recall en false positive rate": 0}
    for _t_women in _thresholds:
        for _t_men in _thresholds:
            _w = _table["Female"][_t_women]
            _m = _table["Male"][_t_men]
            _right = _w["goed"] + _m["goed"]
            _recall_gap = abs(_w["recall"] - _m["recall"])
            _gap = {"gelijk deel met een ja": abs(_w["deel met een ja"] - _m["deel met een ja"]),
                    "gelijke recall": _recall_gap,
                    "gelijke recall en false positive rate":
                        max(_recall_gap, abs(_w["false positive rate"] - _m["false positive rate"]))}
            for _name in _gap:
                if _gap[_name] < 0.01:
                    if _right > _most[_name]:
                        _most[_name] = _right
                        _best[_name] = (_t_women, _t_men)

    # 3. elk gekozen paar op de test set
    for _name in _best:
        _t_women, _t_men = _best[_name]
        _w = _rates("test", "Female", _t_women)
        _m = _rates("test", "Male", _t_men)
        print(_name, "| thresholds", round(_t_women, 2), "en", round(_t_men, 2),
              "| goed", round((_w["goed"] + _m["goed"]) / (_w["mensen"] + _m["mensen"]), 3))
        for _sex, _r in {"vrouwen": _w, "mannen": _m}.items():
            print("  ", _sex, "| deel met een ja", round(_r["deel met een ja"], 2),
                  "| recall", round(_r["recall"], 2),
                  "| false positive rate", round(_r["false positive rate"], 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_criteria({"title": "Eén model, vier paren thresholds, op de test set",
                                    "rows": ["Eén threshold", "Gelijk deel met een ja", "Gelijke recall", "Bijna equalized odds"],
                                    "xlabels": ["deel met een ja", "recall", "false positive rate"], "right": "accuracy",
                                    "key": "open: vrouwen, gevuld: mannen; rechts bij elke rij hoe vaak het model goed zit", "decimal": ",",
                                    "alt": "Drie panelen met elk vier rijen, voor vrouwen en mannen. Eén threshold van 0,5: deel met een ja 0,07 en 0,25, recall 0,44 en 0,57, false positive rate 0,03 en 0,11, accuracy 0,834. Gelijk deel met een ja: 0,14 en 0,15, recall 0,64 en 0,37, false positive rate 0,07 en 0,05, accuracy 0,817. Gelijke recall: deel met een ja 0,10 en 0,24, recall 0,55 en 0,55, false positive rate 0,05 en 0,10, accuracy 0,832. Bijna equalized odds: deel met een ja 0,03 en 0,08, recall 0,22 en 0,22, false positive rate 0,01 en 0,02, accuracy 0,805."})),
        mo.md(r"""
        Geen weight van het model veranderde; alleen de thresholds. **Gelijke recall** koos 0,38 voor de vrouwen en 0,52 voor de mannen: op de test set krijgt 0,55 van de vrouwen en van de mannen die meer verdienen een ja, en zit het model goed bij 0,832 van de mensen in plaats van 0,834. Dat is goedkoop omdat de kansen in beide groepen ongeveer hetzelfde betekenen. 173 vrouwen in de test set hebben een kans tussen 0,38 en 0,5, en 0,40 van hen verdient echt meer: hun nieuwe ja is bij ongeveer 70 van hen goed en bij ongeveer 100 fout. Maar het deel met een ja blijft verschillen, 0,10 tegen 0,24, en de false positive rate ook, 0,05 tegen 0,10. In [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) was de threshold voor de vrouwen 0,37, gekozen op de test set zelf; hier is hij 0,38, gekozen op de mensen om op te trainen.

        **Gelijk deel met een ja** koos 0,29 en 0,67: 0,14 van de vrouwen en 0,15 van de mannen krijgt een ja, en het model zit goed bij 0,817. Hier betalen de mannen: de 1114 mannen in de test set met een kans tussen 0,5 en 0,67 krijgen nu een nee, terwijl 0,58 van hen echt meer verdient. En de recall draait om: 0,64 van de vrouwen die meer verdienen krijgt een ja, en 0,37 van zulke mannen. In [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) kreeg elke groep even vaak een ja als alle mensen samen met één threshold, 0,19; deze zoektocht zoekt in plaats daarvan het paar met de meeste mensen goed.

        Voor **gelijke recall en false positive rate** was het beste paar met één threshold per groep 0,71 en 0,77: een recall van 0,22 voor allebei, en het model zit goed bij 0,805, niet ver boven de 0,764 die een nee voor iedereen haalt. Op de test set zijn de false positive rates 0,007 en 0,019: het verschil onder 0,01 dat op de mensen om op te trainen gold, houdt bij nieuwe mensen net geen stand.

        Op kredietscores vonden Hardt en collega's equal opportunity ook het goedkoopste van de drie fairness-criteria, gemeten in winst: bij hun voorbeeld haalde één threshold voor iedereen, zonder te letten op afkomst, 99,3 procent van de grootste winst, equal opportunity 92,8 procent, equalized odds 80,2 procent en demographic parity 69,8 procent. Wat een criterium kost, hangt af van de data, en elk criterium laat een ander ongelijk: het criterium kiezen is een keuze over de beslissing, niet over het model.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **threshold optimization** | na de training een threshold per groep kiezen: van de paren thresholds die aan een fairness-criterium voldoen, het paar dat de meeste mensen goed heeft, of het best scoort op een andere maat |
        | **randomized threshold** | twee thresholds voor één groep, met een muntworp voor de mensen van wie de kans ertussen ligt; meestal nodig voor precies equalized odds |
        | `ThresholdOptimizer` | de tool van Fairlearn voor threshold optimization; die heeft de groep van elke persoon nodig als hij beslist |
        """),
        L.watch_out("Gelijk op de test set is niet overal gelijk. Frances Ding en collega's kozen zulke thresholds op enquêtedata van Amerikaanse staten, en zagen dat een methode die in de ene staat demographic parity haalt, daar in een andere staat niet meer aan kan voldoen als je haar zomaar overzet. Meet opnieuw op de mensen over wie het model echt beslist."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kies je na de training een threshold per groep:

    1. Neem de kansen van het getrainde model voor mensen van wie je het label kent en die niet in je test set zitten, zoals de mensen waarop het trainde of een validation set.
    2. Kies het criterium en het verschil dat je toestaat. Tel per groep en per threshold hoeveel het goed heeft en de maten die het criterium vergelijkt.
    3. Probeer elk paar thresholds; houd de paren met een verschil onder wat je toestaat, en neem het paar dat de meeste mensen goed heeft.
    4. Meet het paar op de test set: per groep het deel met een ja, de recall en de false positive rate, en hoe vaak het goed zit. Zeg aan welk criterium het voldoet, wat het ongelijk laat, en dat elke beslissing nu de groep van de persoon nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee verzonnen groepen, a en b, van elk 100 mensen, en drie thresholds. `_yes_a` bevat per threshold het deel van groep a dat een ja krijgt, `_right_a` hoeveel mensen uit groep a de threshold goed heeft, en `_yes_b` en `_right_b` hetzelfde voor groep b. De cel zoekt het paar met een gelijk deel met een ja: voor elk paar het verschil tussen de delen met een ja; bij een verschil onder 0,02 hoeveel het paar goed heeft; en in `_best` het paar met de meeste, `_most`.
    """)
    return


@app.cell
def example_search():
    # twee verzonnen groepen van 100 mensen: per threshold het deel met een ja en hoeveel het goed heeft
    _yes_a = {0.4: 0.30, 0.5: 0.20, 0.6: 0.12}
    _yes_b = {0.4: 0.45, 0.5: 0.31, 0.6: 0.21}
    _right_a = {0.4: 80, 0.5: 84, 0.6: 82}
    _right_b = {0.4: 78, 0.5: 85, 0.6: 83}

    _best = None
    _most = 0
    for _t_a in _yes_a:
        for _t_b in _yes_b:
            _gap = abs(_yes_a[_t_a] - _yes_b[_t_b])
            if _gap < 0.02:
                _right = _right_a[_t_a] + _right_b[_t_b]
                print(_t_a, "en", _t_b, "| verschil", round(_gap, 2), "| goed", _right)
                if _right > _most:
                    _most = _right
                    _best = (_t_a, _t_b)
    print("beste:", _best, "met", _most, "van de 200 goed")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Twee van de negen paren hebben een verschil onder 0,02: 0,4 voor a en 0,5 voor b, met 165 goed, en 0,5 voor a en 0,6 voor b, met 167 goed. Het tweede wint. Het paar met allebei 0,5 zou er 169 goed hebben, maar de delen met een ja zijn dan 0,20 en 0,31. Oefening 1 doet dezelfde zoektocht op UCI Adult, voor gelijke recall.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_search():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het beste paar met gelijke recall

        Op de mensen om op te trainen bevat `recall_women` per threshold de recall van de vrouwen, en `right_women` hoeveel vrouwen de threshold goed heeft; `recall_men` en `right_men` hetzelfde voor de mannen. `best` begint als `None` en `most` als 0. De loops gaan langs de thresholds van de vrouwen met `t_women` en die van de mannen met `t_men`. Vervang de `...` door de zoektocht van het uitgewerkte voorbeeld: `gap`, het verschil tussen de twee recalls; bij een verschil onder 0,02 `right`, hoeveel het paar goed heeft; en als `right` meer is dan `most`, onthoud het dan in `most` en het paar in `best`. De check probeert ook een verzonnen tabel.
        """),
        L.task("Vind `best`: het paar `(t_women, t_men)` met een verschil in recall onder 0,02 dat de meeste mensen goed heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("search", lines=16)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Doe wat het uitgewerkte voorbeeld deed, met de recall in plaats van het deel met een ja: voor elk paar het verschil, en alleen bij een verschil onder 0,02 hoeveel het paar goed heeft.",
            "`gap = abs(recall_women[t_women] - recall_men[t_men])`, dan `if gap < 0.02:` en daaronder ingesprongen `right = right_women[t_women] + right_men[t_men]`.",
            "Onder `if right > most:` komen twee regels: `most = right` en `best = ___`.",
        ],
        why="`gap = abs(recall_women[t_women] - recall_men[t_men])`; onder `if gap < 0.02:` komt `right = right_women[t_women] + right_men[t_men]`, en onder `if right > most:` de regels `most = right` en `best = (t_women, t_men)`. Dat geeft 0,4 voor de vrouwen en 0,55 voor de mannen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decide():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Dezelfde kans, een ander antwoord

        `people` bevat vier nieuwe mensen, elk als een tuple van het geslacht (`"women"` of `"men"`) en de kans op een ja volgens het model. `threshold` bevat per geslacht de thresholds voor gelijke recall uit de demo. De loop gaat langs de mensen met `sex` en `score`, en `yes` begint als een lege lijst. Vervang de `...` door één regel die aan `yes` toevoegt of `score` boven de threshold van het geslacht van die persoon ligt. De check probeert ook andere mensen en andere thresholds.
        """),
        L.task("Maak `yes`: voor elke persoon `True` bij een ja en `False` bij een nee, met de threshold van het eigen geslacht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decide():
    box_decide = L.code_box("decide", lines=9)
    box_decide
    return (box_decide,)


@app.cell(hide_code=True)
def check_decide(box_decide):
    result_decide = L.check("decide", box_decide.value)
    result_decide
    return (result_decide,)


@app.cell(hide_code=True)
def hints_decide():
    L.hints(
        "decide",
        [
            "Elke persoon heeft de threshold van het eigen geslacht, en een ja betekent dat de kans erboven ligt.",
            "`threshold[sex]` is de threshold van het geslacht van deze persoon, en `score > threshold[sex]` is `True` bij een ja.",
            "`yes.append(score > ___)`.",
        ],
        why="`yes.append(score > threshold[sex])`. Dat geeft `[True, False, False, True]`: de vrouw en de man met 0,45 krijgen een ander antwoord.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unequal():
    mo.md(r"""
    ### Oefening 3. Wat gelijke recall ongelijk laat
    """)
    return


@app.cell(hide_code=True)
def choice_unequal_cell():
    choice_unequal = L.choice(
        "unequal",
        question="Op UCI Adult kosten de thresholds voor gelijke recall bijna geen accuracy. Wat laten ze ongelijk tussen vrouwen en mannen?",
        options={
            "nothing": "Niets: gelijke recall maakt dan alles gelijk",
            "right": "Het deel met een ja en de false positive rate",
            "accuracy": "Alleen hoe vaak het model over iedereen goed zit",
        },
        correct="right",
        why={
            "nothing": "Alleen de recall werd gelijk. Op de test set bleef het deel met een ja 0,10 tegen 0,24, en de false positive rate 0,05 tegen 0,10. Zie de demo.",
            "right": "Met gelijke recall krijgt 0,10 van de vrouwen en 0,24 van de mannen een ja, en krijgt 0,05 en 0,10 van de mensen die niet meer verdienen er toch een. Elk criterium laat een ander ongelijk.",
            "accuracy": "Hoe vaak het model goed zit, is één getal voor iedereen: 0,832, dicht bij 0,834. Wat tussen de groepen ongelijk blijft, is het deel met een ja, 0,10 tegen 0,24, en de false positive rate, 0,05 tegen 0,10.",
        },
    )
    choice_unequal
    return (choice_unequal,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl) veranderde voor het trainen hoe zwaar elke persoon meetelt. Een threshold per groep verandert de beslissing na het trainen, en verandert geen weight van het model: het heeft alleen de kansen van het model nodig en, bij het beslissen, de groep van elke persoon. Het paar zoeken dat de meeste mensen goed heeft, laat zien wat elk criterium op UCI Adult kost: gelijke recall bijna niets, een gelijk deel met een ja meer, en gelijke recall en false positive rate samen, zonder muntworp, het meest. Het laat ook zien dat elk criterium een ander ongelijk laat, zoals [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) al voorspelde. De volgende les, *Fairness-tools en benchmarks*, laat de libraries zien die deze berekeningen voor je doen, en de tabellen waarop fairness-methoden meestal vergeleken worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_unequal, result_decide, result_search):
    L.wrap_up(result_search, result_decide, choice_unequal)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
