# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-project-sentiment. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nlp-project-sentiment", "nl")
    C = L.checks   # de zinnen uit de recensies in het Engels en het Nederlands, de classifier en het lexicon van de course, de demo's en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl), en een over [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_macro():
    quick_macro = L.choice(
        "quick-macro",
        question="Een classifier noemt alle 10 zinnen van een test positief, en 6 ervan zijn positief. Wat is de macro F1 over de twee labels, positief en negatief?",
        options={
            "accuracy": "0,6",
            "positive": "0,75",
            "right": "0,375",
            "none": "Niet uit te rekenen, want geen enkele zin heet negatief",
        },
        correct="right",
        why={
            "right": r"Voor positief: $2 \times 6 / (2 \times 6 + 4 + 0) = 0{,}75$. Voor negatief: geen true positives, geen false positives en 4 false negatives, dus $0 / (0 + 0 + 4) = 0$. Het gemiddelde is $0{,}375$. In dit project kan een zin ook helemaal geen label krijgen, en je `macro_f1` telt dat mee.",
            "accuracy": "0,6 is hoeveel van de 10 de classifier goed heeft. Macro F1 rekent de F1 score van elk label uit, en neemt het gemiddelde. Wat is de F1 score van negatief, dat de classifier nooit geeft? Zie [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl).",
            "positive": r"0,75 is alleen de F1 score van positief, $2 \times 6 / (2 \times 6 + 4 + 0)$. Macro F1 neemt het gemiddelde over beide labels. Wat voegt negatief toe? Zie [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl).",
            "none": r"De F1 score van negatief heeft 4 false negatives onder de breukstreep: $0 / (0 + 0 + 4)$, en dat is gewoon een getal. Welk getal, en wat is het gemiddelde met de F1 score van positief? Zie [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl).",
        },
    )
    quick_macro
    return (quick_macro,)


@app.cell(hide_code=True)
def quick_check_negation():
    quick_negation = L.choice(
        "quick-negation",
        question="In een lexicon heeft *niet* de score −0,6 en *slecht* −0,7. Een ontkenning draait de scores van de volgende twee woorden om, en houdt de eigen score. Welke som krijgt *niet slecht*?",
        options={
            "plain": "−1,3",
            "right": "0,1",
            "only": "0,7",
            "both": "1,3",
        },
        correct="right",
        why={
            "right": "*niet* telt de eigen −0,6 mee, en *slecht*, direct erna, telt omgekeerd: +0,7. De som is 0,1, dus *niet slecht* komt positief uit. In dit project maakt `lexicon_labels` een label uit zulke sommen.",
            "plain": "−1,3 telt beide scores op zoals de scores in het lexicon staan, zonder de ontkenning. Wat gebeurt er met de score van *slecht*, direct na *niet*? Zie [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl).",
            "only": "0,7 draait *slecht* om, maar laat de score van *niet* zelf weg. Houdt een ontkenning de eigen score? Zie [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl).",
            "both": "1,3 draait ook *niet* om. De ontkenning draait de woorden erna om; wat gebeurt er met de eigen score? Zie [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl).",
        },
    )
    quick_negation
    return (quick_negation,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Project: een spamfilter met TF-IDF](/course/open/nlp-project-spam?lang=nl) bouwde je stukje bij beetje een eigen tekstclassifier: `features` voor de rijen met TF-IDF-waarden en `step` voor de training, allebei in je project `text`. In [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl) bouwde je een sentiment lexicon uit tellingen. Twee manieren om een goede recensie van een slechte te onderscheiden: de ene leert alle weights samen, de andere geeft elk woord een score uit de eigen twee aantallen. In deze stap van het project laat je allebei los op dezelfde recensies, in het Engels en in het Nederlands, en je meet de twee op dezelfde manier.

        Eén getal voor elk is het begin. Macro F1, uit [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl), telt beide labels even zwaar, en een lexicon kan een zin zonder label laten, dus je project telt dat ook mee. Dan de fouten zelf: als twee classifiers met zo'n verschillende werking op verschillende zinnen de mist ingaan, vangt de een de ander op. Gaan de twee op dezelfde zinnen de mist in, dan zit het probleem in de zinnen, en een betere methode van hetzelfde soort komt dan niet ver.

        **Een voorbeeld uit het dagelijks leven.** Twee docenten beoordelen dezelfde opstellen, de een met een beoordelingsmodel, de ander door het hele opstel te lezen. Waar de twee het oneens zijn, helpt een tweede blik. Waar allebei een laag cijfer geven aan een opstel waarvan de leerling zeker is dat het goed is, zit de oorzaak waarschijnlijk in het opstel, of in de opdracht. Dat is maar een vergelijking; hieronder tel je waar je twee classifiers het eens zijn.
        """),
        draw.project_steps(["Je spamfilter", "Je sentimentclassifier"], 1,
                           here_label="← vandaag", title="Waar je project staat: text"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_shared = L.guess(
        "shared",
        question="De classifier van *Logistische regressie op woorden* heeft 120 van de 600 Nederlandse achtergehouden zinnen fout, en het lexicon van *Een lijst met goede en slechte woorden*, met ontkenning, 119. In hoeveel van de 600 zinnen zijn allebei fout, denk je?",
        options={
            "chance": "Ongeveer 25, alsof de fouten niets met elkaar te maken hebben",
            "some": "Ongeveer 55",
            "right": "Ongeveer 90",
            "all": "Ongeveer 115: bijna elke fout",
        },
        reveal="**89.** Als de fouten van de twee niets met elkaar te maken hadden, zouden ongeveer $120 \\times 119 / 600 \\approx 24$ zinnen fout zijn in allebei. 89 is bijna vier keer zoveel. Slechts 31 zinnen zijn alleen in de classifier fout en 30 alleen in het lexicon. In het Engels is het 74 in allebei, 35 en 32. Oefening 3 vindt deze zinnen, en aan het eind van de les lees je de zinnen.",
    )
    guess_shared
    return (guess_shared,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **De zinnen.** Dezelfde zinnen uit recensies als in [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) en [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl): 3.000 zinnen van Amazon, IMDb en Yelp, in 2015 verzameld door Dimitrios Kotzias, Misha Denil, Nando de Freitas en Padhraic Smyth, elk gelabeld als positief of negatief, en de Nederlandse vertaling van alle 3.000 door de course. Elke vijfde zin wordt achtergehouden, zoals eerder. In elke taal geeft dat 2.400 trainingszinnen en 600 achtergehouden zinnen, waarvan 291 positief.

    **Je classifier.** Het spamfilter van je project, op andere tekst: je `document_frequencies` en `features` maken van elke zin een rij met TF-IDF-waarden, en je `step` traint een weight voor elk woord van de trainingszinnen, 4.518 woorden in het Engels en 4.737 in het Nederlands, in 300 stappen met learning rate 1,0, vanaf weight 0 en bias 0. Een achtergehouden zin is positief als de kans op positief, $p$, 0,5 of meer is. De code is in beide talen dezelfde; alleen de woorden verschillen.

    **Je lexicon.** De lijst van [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl): elk woord van de trainingszinnen krijgt een score uit de twee aantallen, $\log_{10}\frac{\text{positief} + 1}{\text{negatief} + 1}$, en een zin krijgt de som van de scores van de woorden, met de twee woorden na een ontkenning omgedraaid. Boven 0 is positief, onder 0 negatief, en bij precies 0 krijgt de zin **geen label**: het lexicon zegt niets over de zin. De lijst van Pattern doet als derde mee, deels met de hand gemaakt en zonder deze recensies.

    **Macro F1 zonder label.** [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl) rekende de F1 score van elk label uit, $2\,TP / (2\,TP + FP + FN)$, en nam het gemiddelde. Hier zijn er twee labels, positief en negatief, en een zin kan geen van beide krijgen. De zin telt dan als false negative van het eigen label, en voor geen enkel label als false positive. Een zin zonder label kost dus recall, en laat precision met rust.

    **De fouten naast elkaar.** Eén getal voor elke classifier verbergt welke zinnen elk fout heeft. Dus je project verdeelt de achtergehouden zinnen in drie groepen: de zinnen die allebei fout hebben, de zinnen die alleen je classifier fout heeft, en de zinnen die alleen je lexicon fout heeft.

    De afbeelding volgt één achtergehouden zin door allebei, met de weights na de training, zoals je de weights hieronder traint.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        De classifier telt voor elk woord een waarde keer een weight op. *niet* en *slecht* hebben allebei een sterk negatieve weight, −2,167 en −1,473, want allebei staan vooral in klachten. Dus $z = -5{,}151$ en $p = 0{,}006$: negatief, en fout. Het lexicon draait de scores van *slecht* en *om*, de twee woorden na *niet*, om: *slecht* telt nu +1,57, en de som is +0,29: positief, het eigen label van de zin. Een rij met TF-IDF-waarden kent geen volgorde, dus de classifier kan niet zien dat *niet* bij *slecht* hoort.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **geen label** | wat een lexicon geeft aan een zin met een som van precies 0; in code `None` |
        | `C.training(lang)` | de 2.400 trainingszinnen in het Engels (`"en"`) of het Nederlands (`"nl"`), als lijst van `[words, label]`: eerst de woorden in kleine letters, dan het label, 1 voor positief en 0 voor negatief |
        | `C.held_out(lang)` | de 600 achtergehouden zinnen, in dezelfde vorm |
        """),
        L.watch_out("De twee classifiers liggen dicht bij elkaar: 480 tegen 481 van de 600 Nederlandse achtergehouden zinnen goed. Bij 600 zinnen zegt een verschil van een paar zinnen weinig over welke van de twee beter is: 600 andere recensies kunnen het omdraaien. Voor een verschil waar je op kunt bouwen, heb je veel meer zinnen nodig, of hetzelfde verschil opnieuw op andere data."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Houd in elke taal elke vijfde zin achter: 2.400 om op te trainen, 600 achtergehouden.
    2. Je classifier: rijen met TF-IDF-waarden uit je `features`, 300 stappen van je `step` met learning rate 1,0, en positief vanaf $p = 0{,}5$.
    3. Je lexicon: een score voor elk woord uit de aantallen in de trainingszinnen, en een label voor elke zin uit de som, met ontkenning. Dat is `lexicon_labels`.
    4. De macro F1 van allebei, in elke taal. Dat is `macro_f1`.
    5. De zinnen die allebei fout hebben, en de zinnen die maar één van de twee fout heeft. Dat is `compare`. Lees de zinnen daarna.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel hieronder volgt stap 1, voor de taal van de les. De cel print hoeveel trainingszinnen en achtergehouden zinnen er zijn, en de tweede trainingszin, om de vorm te laten zien. Daarna telt de cel de positieve achtergehouden zinnen, door de labels op te tellen. Voer de cel uit.
    """)
    return


@app.cell
def example_cell():
    # de trainingszinnen en de achtergehouden zinnen: paren [words, label], label 1 voor positief
    _training = C.training(L.lang)
    _held = C.held_out(L.lang)
    print(len(_training), "trainingszinnen,", len(_held), "achtergehouden")
    print(_training[1])
    _positive = 0
    for _words, _label in _held:
        _positive = _positive + _label
    print(_positive, "achtergehouden zinnen zijn positief")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke zin is een paar: eerst de lijst met woorden, dan het label. Dat is de vorm van de berichten van je spamfilter, dus je `features` en `step` kunnen deze zinnen zo aan. *Een lijst met goede en slechte woorden* zette het label vooraan; in dit project komen de woorden overal eerst, ook in `lexicon` van de course in oefening 1. *uitstekende prijs-kwaliteitverhouding* wordt twee woorden, *prijs* en *kwaliteitverhouding*: een streepje scheidt woorden. 291 van de 600 achtergehouden zinnen zijn positief en 309 negatief: beide labels komen ongeveer even vaak voor, dus macro F1 en hoe vaak een classifier goed zit, liggen hier dicht bij elkaar. Het verschil wordt groter als een classifier zinnen zonder label laat, of het ene label veel vaker geeft dan het andere.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lexicon_labels():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. lexicon_labels: de labels van je lexicon

        `lexicon_labels(texts, scores, negations)` krijgt een lijst `texts` met de woorden van elke zin, een lexicon `scores` met een score voor elk woord, en de lijst `negations`. De functie geeft een lijst terug met één label voor elke zin: 1 als de som van de scores boven 0 ligt, 0 als de som onder 0 ligt, en `None` als de som precies 0 is. In het vak staan al de versies van de course van je `lexicon` en je `score` uit *Een lijst met goede en slechte woorden*, met twee veranderingen: `lexicon` neemt elke zin als `[words, label]`, en `score` noemt het lexicon `scores`, zodat de naam `lexicon` vrij blijft voor de functie.

        `small_texts`, de `SMALL_TEXTS` van de course, bevat vier korte zinnen: *good food*, *not good*, *not bad* en *fine*. `small_scores`, de `SMALL_SCORES` van de course, geeft *good* 0,5, *bad* −0,7 en *not* −0,6. *good food* telt op tot 0,5: positief. *not good* telt op tot −0,6 − 0,5 = −1,1: negatief. *not bad* telt op tot −0,6 + 0,7 = 0,1: positief. Het lexicon heeft *fine* niet, dus deze som is 0: geen label. `lexicon_labels(small_texts, small_scores, ["not"])` geeft `[1, 0, 1, None]`. Vervang de `...`. De check roept dat aan, en daarna eigen gevallen: helemaal geen zinnen; *not fine bad*, waar *bad* het tweede woord na *not* is; *bad not*, met de ontkenning als laatste woord; twee woorden met scores die samen precies 0 zijn; twee Nederlandse zinnen met *niet* en *geen*; en de eerste vijf Engelse achtergehouden zinnen, met het lexicon van de course uit de trainingszinnen. Als alles klopt, bewaart de check het vak in je project, in `work/projects/text/lexicon.py`.
        """),
        L.task("Geef elke zin een label uit de som van de scores, en geen label bij een som van precies 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lexicon_labels():
    box_lexicon_labels = L.code_box("lexicon_labels", lines=24)
    box_lexicon_labels
    return (box_lexicon_labels,)


@app.cell(hide_code=True)
def check_lexicon_labels(box_lexicon_labels):
    result_lexicon_labels = L.check("lexicon_labels", box_lexicon_labels.value)
    result_lexicon_labels
    return (result_lexicon_labels,)


@app.cell(hide_code=True)
def hints_lexicon_labels():
    L.hints(
        "lexicon_labels",
        [
            "In de loop is `total` de som van de zin, uit `score` van de course. Dan `label`: geef het eerst een waarde, en verander de waarde met één `if` voor een som boven 0 en één voor een som onder 0. Voeg `label` als laatste met `append` toe aan `labels`.",
            "Een som van precies 0 ligt niet boven 0 en niet onder 0, dus geen van beide `if`-regels verandert het label. Welke waarde moet het label dan hebben vóór de twee `if`-regels?",
            "`        label = ___`",
        ],
        why="`total = score(words, scores, negations)`, dan `label = None`, `if total > 0:` met `label = 1` eronder, `if total < 0:` met `label = 0`, en `labels.append(label)`; na de loop `return labels`. De eerste vijf Engelse achtergehouden zinnen krijgen `[1, 0, 0, 0, 0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_lexicon_labels():
    mo.md(r"""
    Het lexicon uit de trainingszinnen geeft 296 Nederlandse achtergehouden zinnen positief en 303 negatief, en laat één zin zonder label, *Magische hulp.*: *magische* en *hulp* staan in geen enkele trainingszin, dus de som is 0. In het Engels krijgt elke zin een label. De lijst van Pattern laat 175 Nederlandse zinnen zonder label, en 131 Engelse. In 156 van de Nederlandse zinnen zonder label staat geen enkel woord op de lijst. Van de 175 zijn er 103 negatief: de lijst van Pattern kent vooral bijvoeglijke naamwoorden, en een klacht kan vaak zonder, zoals *Wat een verspilling van geld en tijd!* En *Dit is een prachtige telefoon.* blijft ook zonder label: de lijst heeft *prachtig*, maar de course zoekt woorden op zoals de woorden geschreven staan, en *prachtige* staat er niet in.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_macro_f1():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. macro_f1: macro F1 zonder label

        `macro_f1(gold, said)` krijgt de labels van de zinnen, `gold`, en de labels die een classifier aan de zinnen geeft, `said`: 1, 0, of `None` voor geen label. De functie geeft de macro F1 terug over de labels 1 en 0: het gemiddelde van de F1 scores, elk $2\,TP / (2\,TP + FP + FN)$, zoals in [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl). Een `None` is voor geen van beide een label: een false negative van het eigen label van de zin, en voor geen enkel label een false positive. Beide labels komen voor in `gold`. In het vak staan al de loop over de twee labels en de laatste regel.

        `small_gold`, de `SMALL_GOLD` van de course, bevat drie positieve en daarna drie negatieve zinnen, en `small_said`, de `SMALL_SAID` van de course, geeft deze zinnen de labels `[1, 1, 0, 0, None, 1]`. Voor label 1: twee true positives, één false positive (de zesde) en één false negative (de derde), dus $F_1 = 4 / (4 + 1 + 1) = 0{,}667$. Voor label 0: één true positive (de vierde), één false positive (de derde) en twee false negatives (de vijfde, zonder label, en de zesde), dus $F_1 = 2 / (2 + 1 + 2) = 0{,}4$. `macro_f1(small_gold, small_said)` geeft `0.533`. Vervang de `...`. De check roept dat aan, en daarna eigen gevallen: vier zinnen die allemaal het eigen label krijgen; vijf zinnen die allemaal 1 krijgen, waarvan twee positief; twee zinnen zonder label; drie zinnen die allemaal het andere label krijgen; de labels van de classifier van de course op de 600 Engelse achtergehouden zinnen; en de labels van de lijst van Pattern op de 600 Nederlandse. De check bewaart je functie in `work/projects/text/scores.py`.
        """),
        L.task("Reken de F1 score van elk label uit, met geen label als false negative, en geef het gemiddelde terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_macro_f1():
    box_macro_f1 = L.code_box("macro_f1", lines=18)
    box_macro_f1
    return (box_macro_f1,)


@app.cell(hide_code=True)
def check_macro_f1(box_macro_f1):
    result_macro_f1 = L.check("macro_f1", box_macro_f1.value)
    result_macro_f1
    return (result_macro_f1,)


@app.cell(hide_code=True)
def hints_macro_f1():
    L.hints(
        "macro_f1",
        [
            "Voor elk label: `tp`, `fp` en `fn`, 0 aan het begin, en een loop `for i in range(len(gold)):` met drie `if`-regels die `gold[i]` en `said[i]` vergelijken met `label`. Tel na deze loop, nog binnen de loop over de labels, de F1 score van het label op bij `total`.",
            "Een true positive van een label heeft dat label in beide lijsten, een false positive alleen in `said`, een false negative alleen in `gold`. Een `None` in `said` is nooit gelijk aan een label. Hoe maken de drie aantallen de F1 score?",
            "`        total = total + ___`",
        ],
        why="Onder de loop over de plaatsen: `if gold[i] == label and said[i] == label:` met `tp = tp + 1`, `if gold[i] != label and said[i] == label:` met `fp = fp + 1`, en `if gold[i] == label and said[i] != label:` met `fn = fn + 1`; na deze loop `total = total + 2 * tp / (2 * tp + fp + fn)`. Met de labels van de classifier van de course op de Engelse achtergehouden zinnen geeft dat 0,818.",
    )
    return


@app.cell(hide_code=True)
def text_after_macro_f1():
    mo.md(r"""
    In het Nederlands krijgt de classifier 0,800 en het lexicon uit de trainingszinnen 0,802; in het Engels 0,818 en 0,823. De lijst van Pattern krijgt 0,637 in het Nederlands en 0,703 in het Engels. De F1 score voor negatief is bij Pattern het laagst van allemaal, 0,568 in het Nederlands en 0,626 in het Engels: de lijst van Pattern noemt maar 149 Nederlandse zinnen negatief, tegen 276 positief, en laat 175 zinnen zonder label, waarvan 103 negatief. De twee classifiers uit deze recensies zijn in beide talen ongeveer even goed, en allebei doen het in het Nederlands ongeveer twee punten slechter.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train je classifier

    De knop traint je classifier op de 2.400 trainingszinnen, eerst in het Engels, dan in het Nederlands. Je `document_frequencies` en `features` maken de rijen, en je `step` zet 300 stappen met learning rate 1,0, vanaf weight 0 voor elk woord en bias 0. Een bestand dat niet in je project staat, of niet klopt, wordt vervangen door de versie van de course, zoals de notitie onder *Waarom* zegt. De grafiek toont de gemiddelde log loss van de trainingszinnen na elke stap. Ernaast zie je de vijf woorden met de sterkste weight naar elke kant, drie achtergehouden zinnen met de kans op positief, en hoeveel achtergehouden zinnen bij 0,5 het eigen label krijgen. Het duurt ongeveer tien seconden. Daarna kun je terug door de stappen van de run in het Nederlands.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train in het Engels en in het Nederlands, elk 300 stappen")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(demo_button):
    trained = None
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om te trainen.*")
    else:
        _now = {}

        def _progress(_lang, _step, _loss, _ws, _b, _held_rows):
            if _lang not in _now:
                _words = C.live_words(L.lang, _lang)
                _now[_lang] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                     C.STEPS, low=0.0, high=0.7)
                _now[_lang].above = [mo.Html(_now["en"].picture())] if _lang == "nl" and "en" in _now else []
                _now["words"] = _words
            _now[_lang].watch(lambda: C.now_parts(_held_rows, _ws, _b, L.lang, _lang))
            _now[_lang].add(_step, _loss, _now["words"]["progress"].format(s=_step, n=C.STEPS, loss=L.number(_loss, 3)))

        mo.output.replace(mo.md("*De rijen van de zinnen maken …*"))
        _run = C.demo(_progress)
        if isinstance(_run, str):
            _note = C.broken(_run, L.lang)
            _last = _now.get("nl", _now.get("en"))
            _out = _last.stopped(_note) if _last is not None else mo.md(_note)
        else:
            trained = _run
            _mine = _run[L.lang]
            _who = {(): "de functies van de course", ("features",): "je `features` en `step` van de course",
                    ("train",): "`features` van de course en je `step`"}.get(tuple(_run["yours"]), "je `features` en je `step`")
            _out = mo.vstack([
                mo.md(f"""Na {C.STEPS} stappen met {_who} geeft je classifier bij 0,5 <span class="ea-num">{C.rewind_right(_run["en"], C.STEPS)}</span> van de 600 Engelse achtergehouden zinnen het eigen label, en <span class="ea-num">{C.rewind_right(_run["nl"], C.STEPS)}</span> van de 600 Nederlandse. De gemiddelde log loss van de trainingszinnen is {L.number(_run["en"]["curve"][-1][1], 3)} in het Engels en {L.number(_run["nl"]["curve"][-1][1], 3)} in het Nederlands."""),
                L.explore("rewind", "stap", lambda _s: (C.rewind_frame(_mine, int(_s), L.lang),
                                                        "Na stap {s}: {r} van de 600 Nederlandse achtergehouden zinnen krijgen het eigen label.".format(
                                                            s=int(_s), r=C.rewind_right(_mine, int(_s)))),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Ga terug naar stap 1, en dan weer vooruit: welke woorden zijn het eerst sterk, en hoe ver zakt de kans op positief van *Voorspelbaar, maar niet slecht om te kijken.*?"),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met elke weight op 0 krijgt elke zin $p = 0{,}5$ en is de gemiddelde log loss $\log 2 = 0{,}693$; na 300 stappen is het 0,158 in het Nederlands en 0,165 in het Engels. Bij 0,5 geeft je classifier 480 van de 600 Nederlandse achtergehouden zinnen het eigen label, en 491 van de Engelse. De sterkste woorden richting positief zijn *geweldig* (1,86), *geweldige* (1,74), *goed* (1,23), *beste* (1,06) en *goede* (0,96); richting negatief *niet* (−2,17), *slecht* (−1,47), *slechte* (−1,22), *slechtste* (−1,01) en *verschrikkelijk* (−0,76). In het Engels staan *great* (2,21) en *good* (1,33) bovenaan, en *not* (−1,78) en *bad* (−1,49) onderaan. Het Nederlands zet een *e* achter een bijvoeglijk naamwoord vóór een zelfstandig naamwoord, dus *geweldig* en *geweldige* krijgen elk een eigen weight. De Nederlandse trainingszinnen hebben meer verschillende woorden, 4.737 tegen 4.518, en meer woorden die maar één keer voorkomen, 2.922 tegen 2.625; of dat de 11 zinnen verschil tussen de twee talen verklaart, laten deze getallen alleen niet zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. compare: waar allebei de mist ingaan

        `compare(gold, mine, theirs)` krijgt de labels van de zinnen, `gold`, en de labels die twee classifiers aan de zinnen geven, `mine` en `theirs`. De functie geeft drie lijsten met plaatsen terug: de zinnen die allebei fout hebben, de zinnen die alleen `mine` fout heeft, en de zinnen die alleen `theirs` fout heeft. Een label is fout als het niet het eigen label van de zin is, dus een ontbrekend label, `None`, is ook fout. In het vak staan al de drie lege lijsten, de loop over de plaatsen en de laatste regel.

        `small_mine`, de `SMALL_MINE` van de course, is `[1, 0, 1, 1, 0, 0]`, en `small_theirs`, de `SMALL_THEIRS` van de course, is `[1, 0, 0, 0, None, 0]`, allebei bij `small_gold`, en dat is `[1, 1, 1, 0, 0, 0]`. Met de plaatsen geteld vanaf 0, zoals Python telt, is de zin op plaats 1 in allebei fout; de zin op plaats 3 alleen in `mine`; de zinnen op plaats 2 en 4 alleen in `theirs`; `theirs` geeft de zin op plaats 4 geen label. `compare(small_gold, small_mine, small_theirs)` geeft `([1], [3], [2, 4])`. Vervang de `...`. De check roept dat aan, en daarna eigen gevallen: drie lege lijsten; drie zinnen die allebei goed hebben; twee zinnen die allebei fout hebben, waarvan één zonder label in `theirs`; drie zinnen, één alleen fout in `mine` en twee alleen fout in `theirs`; en de classifier van de course tegen het lexicon van de course op de 600 Engelse achtergehouden zinnen. De check bewaart je functie in `work/projects/text/compare.py`.
        """),
        L.task("Verdeel de plaatsen in fout in allebei, alleen fout in `mine` en alleen fout in `theirs`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compare():
    box_compare = L.code_box("compare", lines=16)
    box_compare
    return (box_compare,)


@app.cell(hide_code=True)
def check_compare(box_compare):
    result_compare = L.check("compare", box_compare.value)
    result_compare
    return (result_compare,)


@app.cell(hide_code=True)
def hints_compare():
    L.hints(
        "compare",
        [
            "Drie `if`-regels in de loop, één voor elke lijst. Elke regel vergelijkt `mine[i]` en `theirs[i]` met `gold[i]`, verbindt de twee vergelijkingen met `and`, en voegt de plaats `i` met `append` toe aan de eigen lijst.",
            "Een zin die alleen `theirs` fout heeft, is goed in `mine`, en heeft in `theirs` een ander label dan in `gold`, of helemaal geen. Welke vergelijking klopt voor twee labels die verschillen?",
            "`        if mine[i] == gold[i] and theirs[i] ___ gold[i]:`",
        ],
        why="`if mine[i] != gold[i] and theirs[i] != gold[i]:` met `both.append(i)`, `if mine[i] != gold[i] and theirs[i] == gold[i]:` met `only_mine.append(i)`, en `if mine[i] == gold[i] and theirs[i] != gold[i]:` met `only_theirs.append(i)`. Voor de classifier en het lexicon van de course op de Engelse achtergehouden zinnen bevatten de drie lijsten 74, 35 en 32 plaatsen.",
    )
    return


@app.cell(hide_code=True)
def text_project_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je project heeft nu een classifier en een lexicon, en de functies om de twee te meten en de fouten te vergelijken. De afbeelding neemt de classifier die je hierboven trainde, je `lexicon_labels` met het lexicon van de course uit de trainingszinnen, je `macro_f1` en je `compare`. De afbeelding geeft de macro F1 van allebei in beide talen, en voor de taal van de les een paar zinnen uit elke groep: eerst de zinnen die de tekst noemt, dan de kortste. Een bestand dat nog niet in je project staat, telt mee met de versie van de course, en de afbeelding zegt dat erbij.
    """)
    return


@app.cell(hide_code=True)
def project_demo(result_compare, result_lexicon_labels, result_macro_f1, trained):
    _r = (result_lexicon_labels, result_macro_f1, result_compare)   # zodat deze cel na elke check opnieuw draait
    if trained is None:
        _out = mo.md("*Train eerst je classifier hierboven: de afbeelding vergelijkt de classifier met je lexicon.*")
    else:
        _out = mo.Html(C.project_picture(trained, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_after_project_demo():
    mo.md(r"""
    Lees de drie groepen. Fout in allebei: *Uitstekende draadloze headset voor beginners.* heeft het label negatief, terwijl er niets in klaagt; misschien vergiste wie het label gaf zich, misschien ging de recensie verder in een andere zin. Een label dat je niet uit de zin zelf kunt verklaren, verslaat elke classifier die alleen de zin leest. Korte zinnen zoals *Niet missen.* gaan ook in allebei fout: *niet* hangt in deze recensies naar negatief, en hier is het een advies om te gaan. Alleen fout in je classifier: *Voorspelbaar, maar niet slecht om te kijken.*, de zin van de afbeelding, en *Deze film heeft geen enkel pluspunt.*, waar het lexicon *pluspunt* omdraait na *geen*. Alleen fout in je lexicon: *Nogmaals: helemaal geen plot.* *plot* heeft al een eigen negatieve score, −0,80: in deze recensies komt een plot vooral ter sprake in een klacht, 18 van de 20 keer in negatieve zinnen. Na *geen* omgedraaid telt *plot* als lof. De regel die de ene zin redt, breekt een andere, zoals in *Een lijst met goede en slechte woorden*. En *Mooi stuk troep.* is ironisch: in het lexicon weegt *mooi* (+1,08) zwaarder dan *troep* (−0,74).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shared():
    mo.md(r"""
    ### Oefening 4. Wat zeggen de gedeelde fouten?
    """)
    return


@app.cell(hide_code=True)
def choice_shared_cell():
    choice_shared = L.choice(
        "shared-mistakes",
        question="Je classifier en je lexicon delen 89 van de fouten op de Nederlandse achtergehouden zinnen, waar toeval er ongeveer 24 zou geven. Wat zegt dat?",
        options={
            "combine": "De twee samen zouden de meeste fouten oplossen",
            "right": "Veel van wat misgaat, zit in de zinnen, niet in een van de methodes",
            "copied": "Het lexicon leerde de fouten van de classifier",
            "chance": "Weinig: met meer dan 100 fouten elk ontstaat zoveel overlap ook door toeval",
        },
        correct="right",
        why={
            "right": "Twee methodes met een heel verschillende werking gaan op dezelfde zinnen de mist in: een label dat de zin niet verklaart, een advies zoals *Niet missen.*, een woord dat hier iets anders betekent dan in de meeste recensies. Allebei lezen losse woorden, en deze zinnen zeggen iets wat de woorden, één voor één, niet zeggen. Een betere methode van dezelfde soort lost dat niet op; een classifier die elk woord in de context leest, zoals de BERT van *BERT: het ontbrekende woord invullen*, maakt meer kans, en sommige labels krijgt geen enkele classifier goed.",
            "combine": "Waar allebei fout zijn, geven de twee bijna altijd hetzelfde verkeerde label: elk geeft 1 of 0, en alleen in *Magische hulp.* geeft het lexicon geen label. In de andere 88 zinnen zijn de twee het eens, en een stemming van de twee zegt daar hetzelfde. Samen kunnen de twee alleen helpen in de 61 zinnen waar precies één van de twee goed zit.",
            "copied": "Het lexicon komt uit de aantallen van elk woord in de trainingszinnen; de classifier krijgt het lexicon nooit te zien. De overlap komt uit de zinnen: allebei leren van dezelfde trainingszinnen, en allebei worden getest op dezelfde achtergehouden zinnen.",
            "chance": "Door toeval zouden ongeveer $120 \\times 119 / 600 \\approx 24$ zinnen fout zijn in allebei. 89 is bijna vier keer zoveel, dus de overlap is geen toeval.",
        },
    )
    choice_shared
    return (choice_shared,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Project: een spamfilter met TF-IDF](/course/open/nlp-project-spam?lang=nl) gaf je project een classifier en de code om te trainen; hier leerde dezelfde code sentiment, in het Engels en in het Nederlands, zonder één regel te veranderen. [Een lijst met goede en slechte woorden](/course/open/nlp-sentiment-lexicon?lang=nl) gaf het lexicon, nu met een label voor elke zin en geen label waar het lexicon niets zegt. [Veel labels: micro- en macrogemiddelden](/course/open/nlp-micro-macro?lang=nl) gaf macro F1, nu met een ontbrekend label als false negative. Op deze recensies komen de twee classifiers ongeveer even goed uit, en de twee delen de meeste fouten: wat overblijft, zit in zinnen die losse woorden niet verklaren. Met het spamfilter erbij bevat je project `text` nu `logistic.py`, `features.py`, `train.py`, `judge.py`, `lexicon.py`, `scores.py` en `compare.py`. De volgende les, *Waar gaat deze verzameling over? Topic models*, vindt waar een verzameling documenten over gaat, zonder enig label.

    **Een voorbeeld uit het dagelijks leven.** Twee vrienden geven een film een duim omhoog of omlaag, de een op basis van de trailer, de ander op basis van recensies. Als allebei het mis hebben over dezelfde film, was de film waarschijnlijk moeilijk van buitenaf te beoordelen: een trage start, een grap die je pas aan het eind snapt. Een derde vriend vragen die ook alleen de trailer zag, helpt dan niet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shared, result_compare, result_lexicon_labels, result_macro_f1):
    L.wrap_up(result_lexicon_labels, result_macro_f1, result_compare, choice_shared)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
