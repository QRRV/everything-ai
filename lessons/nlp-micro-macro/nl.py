# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-micro-macro. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-micro-macro", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl), en een over [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="Een spamfilter vindt 30 van de 50 spammails, en noemt daarbij 12 andere mails spam. Wat is de F1 score van het filter voor spam?",
        options={
            "recall": "0,6",
            "right": "0,65",
            "precision": "0,71",
            "mean": "0,657",
        },
        correct="right",
        why={
            "right": r"$2 \times 30 / (2 \times 30 + 12 + 20) = 60 / 92 = 0{,}652$: tussen de precision, 0,714, en de recall, 0,6, en dichter bij de kleinste van de twee. Deze les rekent zo'n F1 score uit voor elk van 59 labels.",
            "precision": "0,71 is de precision: 30 van de 42 mails die het filter spam noemt. De F1 score telt ook de 20 spammails mee die het filter mist. Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "recall": "0,6 is de recall: 30 van de 50 spammails. De F1 score telt ook de 12 mails mee die het filter ten onrechte spam noemt. Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "mean": r"0,657 is het gewone gemiddelde van de precision, 0,714, en de recall, 0,6. De F1 score is $2PR / (P + R)$, en ligt dichter bij de kleinste van de twee: 0,652. Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def quick_check_sentence():
    quick_sentence = L.choice(
        "quick-sentence",
        question="In [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) kreeg *I did not expect this to be so good!* (*Ik had niet verwacht dat het zo goed zou zijn!*), een zin met het label positief, de kans 0,423 op positief. Met positief als positief label: wat is deze zin voor de classifier?",
        options={
            "tp": "Een true positive, omdat de zin positief is",
            "fp": "Een false positive, omdat de kans van de zin te laag is",
            "right": "Een false negative: positief, negatief genoemd",
            "tn": "Een true negative, omdat 0,423 onder 0,5 ligt",
        },
        correct="right",
        why={
            "right": "Met 0,423, onder 0,5, zegt de classifier negatief terwijl het label positief is: een false negative. In deze les is elk label om de beurt het positieve, en één fout is dan een false negative voor het ene label en een false positive voor een ander.",
            "tp": "Voor een true positive moeten twee dingen kloppen: het label is positief, en de classifier zegt positief. Met 0,423 zegt de classifier negatief. Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "fp": "Een false positive is een zin die de classifier positief noemt terwijl het label negatief is; hier is het andersom. Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "tn": "Een true negative is een negatieve zin die negatief genoemd wordt, en deze zin heeft het label positief. Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
        },
    )
    quick_sentence
    return (quick_sentence,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe koos een classifier in dit domein tussen twee labels, zoals positief en negatief. Een assistent waar je tegen praat, kiest uit veel meer: *wake me up at nine am on friday* (*maak me vrijdag om negen uur wakker*) vraagt om een wekker, *is it raining* (*regent het*) om het weer. Elk soort verzoek is een **intent**, met een naam zoals alarm_set of weather_query, en een classifier geeft elke opdracht één intent.

    De opdrachten van deze les komen uit MASSIVE, een set die Jack FitzGerald en collega's van Amazon in 2022 publiceerden: een miljoen realistische opdrachten aan een assistent, met labels, in 51 talen, over 18 onderwerpen zoals muziek en de agenda, met 60 intents en 55 soorten slots. Slots zijn de woorden van een opdracht die ertoe doen, zoals *five am*; deze les gebruikt de slots niet. Professionele vertalers zetten de Engelse opdrachten om in 50 andere talen, waaronder het Nederlands. Deze les gebruikt de 16.521 Engelse opdrachten, elk met een van de 60 intents: 11.514 trainen een classifier, 2.974 testen de classifier, en de andere 2.033 gebruikt deze les niet. Ook de Nederlandse les gebruikt de Engelse opdrachten, zodat alle getallen dezelfde zijn als in de Engelse les; onder *Hoe goed is het?* staan de getallen voor de Nederlandse opdrachten. De intents komen lang niet even vaak voor. 810 trainingsopdrachten zijn calendar_set, de intent voor een herinnering in de agenda, zoals *remind me at one p. m.* (*herinner me om één uur 's middags*); 14 zijn music_dislikeness, de intent om te zeggen dat je een liedje niet leuk vindt.

    De classifier is naive Bayes, uit *Spam of niet: naive Bayes*, met 60 intents in plaats van twee labels. De score van een intent is de log van het aandeel van de intent in de trainingsopdrachten, de prior, plus voor elk woord van de opdracht de log van het aandeel van dat woord in de woorden van de trainingsopdrachten van die intent, met de add-k smoothing van *Paren die je nooit zag: smoothing* en $k = 0{,}1$. Elke testopdracht krijgt de intent met de hoogste score, en 2.437 van de 2.974 krijgen de eigen intent: 81,9%. Maar is de classifier bij elke intent even goed?

    **Een voorbeeld uit het dagelijks leven.** Een busbedrijf meldt dat 95% van alle ritten op tijd is. De meeste ritten rijden op drie drukke lijnen; op een rustige lijn met een paar ritten per dag is de helft te laat. In het aandeel van alle ritten valt die rustige lijn nauwelijks op; in het gemiddelde van de aandelen per lijn, waarin elke lijn één keer telt, wel.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Naive Bayes geeft 81,9% van de 2.974 testopdrachten de eigen intent. Reken voor elk van de 59 intents van de testopdrachten apart de F1 score uit, en neem het gewone gemiddelde van de 59 getallen. Wat verwacht je?",
        options={
            "higher": "Ongeveer 0,95: hoger, want de meeste intents zijn makkelijk",
            "same": "Ongeveer 0,82: hetzelfde als het aandeel dat goed gaat",
            "right": "Ongeveer 0,75: lager dan het aandeel dat goed gaat",
            "far": "Ongeveer 0,45: veel lager dan het aandeel dat goed gaat",
        },
        reveal="**Ongeveer 0,75: lager**, 0,752. De intents met veel opdrachten krijgen meestal een hoge F1 score: calendar_set 0,888, play_music 0,883, weather_query 0,909. Maar drie zeldzame intents krijgen 0: general_greet, iot_hue_lighton en music_dislikeness, die naive Bayes aan geen enkele opdracht geeft. Elk van de drie telt in het gemiddelde even zwaar als calendar_set, met 209 testopdrachten. De 30 intents met de meeste trainingsopdrachten hebben een gemiddelde F1 score van 0,825, de andere 29 van 0,676.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Eén label tegelijk

    Precision, recall en de F1 score van [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) hebben een positief label nodig. Met veel labels neem je elk label om de beurt als positief, en alle andere samen als negatief. Voor play_music is een true positive een opdracht van play_music die de classifier play_music geeft; een false positive is een opdracht van een andere intent die play_music krijgt; een false negative is een opdracht van play_music die een andere intent krijgt. In de confusion matrix van *Meer dan accuracy: precision en recall*, met een rij voor elk echt label en een kolom voor elk label dat de classifier geeft, zijn de true positives van een label de cel op de diagonaal, de false positives de rest van de kolom van dat label, en de false negatives de rest van de rij.

    Eén fout telt dus twee keer. *do not play rock metal* (*speel geen rock metal*) is een opdracht van music_dislikeness, en naive Bayes geeft de opdracht play_music: een false positive voor play_music, en een false negative voor music_dislikeness.

    ### Twee manieren om te middelen

    Zo krijgt elk van de 59 intents een F1 score. Daniel Jurafsky en James H. Martin beschrijven in *Speech and Language Processing* twee manieren om daar één getal van te maken. Bij macro-averaging reken je de score voor elk label apart uit, en neem je het gemiddelde over de labels. Bij micro-averaging verzamel je de beslissingen voor alle labels in één confusion matrix, en reken je precision en recall uit met die ene tabel. Voor de F1 score, met $L$ labels en de aantallen $TP_c$, $FP_c$ en $FN_c$ van label $c$:

    $$\text{macro } F_1 = \frac{1}{L} \sum_{c} \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c} \qquad \text{micro } F_1 = \frac{2\,TP}{2\,TP + FP + FN}$$

    met $TP = \sum_c TP_c$, de true positives van alle labels opgeteld, en net zo voor $FP$ en $FN$. **Macro-averaging** laat elk label even zwaar tellen, hoe weinig opdrachten het label ook heeft. **Micro-averaging** laat elke opdracht even zwaar tellen, dus een label dat vaak voorkomt, weegt zwaarder. De afbeelding laat beide zien voor naive Bayes op de testopdrachten.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Elke grijze balk is één intent, van de meeste trainingsopdrachten links naar de minste rechts. De doorgetrokken lijn is de macro F1, 0,752; de gestreepte lijn de micro F1, 0,819. Volgens Jurafsky en Martin wordt een microgemiddelde vooral bepaald door het label dat het vaakst voorkomt, omdat de aantallen bij elkaar worden opgeteld; het macrogemiddelde laat beter zien hoe het met de kleinere labels gaat, en past dus beter als elk label even belangrijk is. Hun voorbeeld heeft drie labels, urgent, normal en spam; het uitgewerkte voorbeeld hieronder gebruikt de getallen daarvan.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **intent** | wat iemand met een opdracht wil, als label, zoals alarm_set |
        | **macro-averaging** | voor elk label een score uitrekenen en daar het gewone gemiddelde van nemen: elk label telt even zwaar |
        | **micro-averaging** | de aantallen van alle labels optellen en de score één keer uitrekenen: elk voorbeeld telt even zwaar |
        | macro $F_1$, micro $F_1$ | de F1 score met macro-averaging en met micro-averaging |
        | $TP_c$, $FP_c$, $FN_c$ | de true positives, false positives en false negatives van label $c$ |
        """),
        L.watch_out("Sommige papers rekenen een macro F1 anders uit: als de F1 score van de gemiddelde precision en de gemiddelde recall van de labels. Voor naive Bayes op de testopdrachten geeft dat 0,767 in plaats van 0,752. In deze course is de macro F1 het gemiddelde van de F1 scores. Kijk bij een macro F1 die je leest of zelf opschrijft welke van de twee het is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Leg de lijst van labels vast: hier de 59 intents van de testopdrachten.
    2. Tel voor elk label de true positives, false positives en false negatives, met dat label als positief en alle andere als negatief.
    3. Voor de macro F1: reken de F1 score van elk label uit met de eigen aantallen, $2\,TP_c / (2\,TP_c + FP_c + FN_c)$, en neem het gewone gemiddelde over de labels.
    4. Voor de micro F1: tel de true positives, false positives en false negatives van alle labels op, en reken met de drie sommen één F1 score uit.
    5. Geef beide, en kijk naar de labels met de laagste F1 score: daar faalt de classifier.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het voorbeeld van Jurafsky en Martin deelt 367 mails in bij urgent, normal en spam. `SLP_MATRIX` van de course bevat de confusion matrix, met een rij voor elk echt label en een kolom voor elk label dat het systeem gaf, zoals in *Meer dan accuracy: precision en recall*; Jurafsky en Martin tekenen de matrix andersom. De code neemt voor elk label de true positives van de diagonaal, de false positives uit de rest van de kolom van het label, en de false negatives uit de rest van de rij.
    """)
    return


@app.cell
def example_cell():
    names = ["urgent", "normal", "spam"]
    cm = C.SLP_MATRIX   # een rij voor elk echt label, een kolom voor elk label dat het systeem gaf
    for k in range(3):
        tp = cm[k][k]
        fp = cm[0][k] + cm[1][k] + cm[2][k] - tp
        fn = cm[k][0] + cm[k][1] + cm[k][2] - tp
        print(names[k], tp, fp, fn)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Urgent krijgt $F_1 = 2 \times 8 / (2 \times 8 + 11 + 8) = 16 / 35 = 0{,}457$, normal $120 / 215 = 0{,}558$ en spam $400 / 484 = 0{,}826$. De macro F1 is het gemiddelde daarvan, $(0{,}457 + 0{,}558 + 0{,}826) / 3 = 0{,}614$. Voor de micro F1 tel je op: $TP = 8 + 60 + 200 = 268$, $FP = 11 + 55 + 33 = 99$ en $FN = 8 + 40 + 51 = 99$, dus de micro F1 is $536 / (536 + 99 + 99) = 0{,}730$. Spam heeft 251 van de 367 mails en trekt de micro F1 naar de eigen 0,826; urgent, met 16 mails, telt voor een derde mee in de macro F1. Jurafsky en Martin rekenen zo de precision uit: 0,60 met macro-averaging en 0,73 met micro-averaging.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. counts: één label tussen vele

        `gold` bevat de echte intent van elke testopdracht, en `said` de intent die naive Bayes eraan geeft, in dezelfde volgorde; de cel hieronder print de eerste drie opdrachten met allebei, uit `test` van de course. Schrijf `counts(gold, said, label)`: ga de plekken `i` van de opdrachten langs, en tel de true positives (`gold[i]` en `said[i]` zijn allebei `label`), de false positives (alleen `said[i]` is `label`) en de false negatives (alleen `gold[i]` is `label`). Geef `tp, fp, fn` terug.

        `small_gold` en `small_said`, `SMALL_GOLD` en `SMALL_SAID` van de course, zijn zes opdrachten: vijf van play_music en één van alarm_set, en een classifier die bij alle zes play_music zegt. Dus `counts(small_gold, small_said, "play_music")` geeft `(5, 1, 0)`, en `counts(small_gold, small_said, "alarm_set")` geeft `(0, 0, 1)`. Vervang de `...`. De check roept die twee aan, en daarna eigen gevallen: de labels van `more_gold` en `more_said`, `MORE_GOLD` en `MORE_SAID` van de course, een label dat in geen van beide lijsten staat, en drie intents van de testopdrachten, met `gold` en `said` uit `gold` en `said` van de course.
        """),
        L.task("Tel de true positives, false positives en false negatives van één label."),
    ])
    return


@app.cell
def commands_cell():
    small_gold = C.SMALL_GOLD
    small_said = C.SMALL_SAID
    more_gold = C.MORE_GOLD
    more_said = C.MORE_SAID
    gold = C.gold()
    said = C.said()
    print(small_gold)
    print(small_said)
    print(len(gold), len(said))
    for i in range(3):
        print(C.test()[i][1], "|", gold[i], "|", said[i])
    return gold, more_gold, more_said, said, small_gold, small_said


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("counts", lines=12)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "counts",
        [
            "Zet `tp`, `fp` en `fn` eerst op 0, en ga de plekken langs met `for i in range(len(gold)):`. Vergelijk `gold[i]` en `said[i]` met `label`, en tel 1 op bij het aantal dat past: `tp` als allebei `label` zijn, `fp` als alleen `said[i]` het is, `fn` als alleen `gold[i]` het is. Geef na de loop `tp, fp, fn` terug.",
            "In `small_gold` is de derde opdracht alarm_set, en de classifier zegt play_music. Bij welk aantal hoort die opdracht voor het label play_music, en wat geldt er voor de eigen intent?",
            r"`        if ___ and said[i] == label:`",
        ],
        why=r"`tp`, `fp` en `fn` beginnen bij 0; `for i in range(len(gold)):` met `if gold[i] == label and said[i] == label:` en `tp = tp + 1`, `if gold[i] != label and said[i] == label:` en `fp = fp + 1`, en `if gold[i] == label and said[i] != label:` en `fn = fn + 1`; na de loop `return tp, fp, fn`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    In de afbeelding telde je functie voor acht intents. Van de 209 testopdrachten van calendar_set, de intent die het vaakst voorkomt, gaan er 191 goed, en daarnaast krijgen 30 opdrachten van andere intents calendar_set. general_quirky, voor vreemde vragen zoals *check my car is ready* (*kijk of mijn auto klaar is*), heeft de meeste fouten, naar beide kanten: 54 false positives en 92 false negatives. Van de vier zeldzame intents in de afbeelding krijgen er drie nooit een opdracht: *what's up olly* (*hoe gaat het, olly*), de enige testopdracht van general_greet, krijgt news_query, en *do not play rock metal* krijgt play_music.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_macro():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. macro_f1: elk label telt even zwaar

        Schrijf `macro_f1(gold, said, labels)`: haal voor elk label van `labels` de aantallen op met `counts`, reken de F1 score uit, $2\,TP / (2\,TP + FP + FN)$, en geef het gewone gemiddelde over de labels terug. Elk label in `labels` komt voor in `gold`, dus geen noemer is 0. `counts` van de course staat voor je klaar.

        `small_labels`, `SMALL_LABELS` van de course, bevat alarm_set en play_music. Bij het kleine voorbeeld krijgt play_music $2 \times 5 / (2 \times 5 + 1 + 0) = 10 / 11 = 0{,}909$ en alarm_set $0 / (0 + 0 + 1) = 0$, dus `macro_f1(small_gold, small_said, small_labels)` geeft $(0{,}909 + 0) / 2 = 0{,}4545$. Vervang de `...`. De check roept dat aan en `macro_f1(more_gold, more_said, more_labels)`, met `more_labels` uit `MORE_LABELS` van de course, en daarna eigen gevallen: elke opdracht goed, het tweede voorbeeld met alleen alarm_set en weather_query als labels, en de testopdrachten met `labels`, de 59 intents uit `labels` van de course: met naive Bayes, zonder de prior uit `said_flat` van de course, en met maar 10 bekende intents uit `said_known` van de course.
        """),
        L.task("Reken de macro F1 uit: het gemiddelde van de F1 scores van de labels."),
    ])
    return


@app.cell
def labels_cell():
    small_labels = C.SMALL_LABELS
    more_labels = C.MORE_LABELS
    labels = C.labels()
    print(small_labels, more_labels)
    print(labels[:4], labels[-1])
    return labels, more_labels, small_labels


@app.cell(hide_code=True)
def exercise_macro():
    box_macro = L.code_box("macro", lines=6)
    box_macro
    return (box_macro,)


@app.cell(hide_code=True)
def check_macro(box_macro):
    result_macro = L.check("macro", box_macro.value)
    result_macro
    return (result_macro,)


@app.cell(hide_code=True)
def hints_macro():
    L.hints(
        "macro",
        [
            "Begin met `total = 0`, en ga de labels langs met `for label in labels:`. Haal bij elk label `tp, fp, fn` op met `counts(gold, said, label)`, en tel de F1 score op bij `total`. Maak na de loop van `total` het gewone gemiddelde over de labels.",
            "Bij het kleine voorbeeld zijn de twee F1 scores 0,909 en 0, en de macro F1 is 0,4545. Waardoor deel je de som: door de 6 opdrachten, of door iets anders?",
            r"`    return total / ___`",
        ],
        why=r"`total = 0`; `for label in labels:` met `tp, fp, fn = counts(gold, said, label)` en `total = total + 2 * tp / (2 * tp + fp + fn)`; na de loop `return total / len(labels)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_macro():
    mo.md(r"""
    In de afbeelding nam je functie het gemiddelde van de 59 balken: 0,752. Drie balken zijn 0, en music_settings en audio_volume_other krijgen 0,286: elk van de twee vindt 1 van de 6 testopdrachten. Zonder de drie intents met 0 zou het gemiddelde van de andere 56 op 0,792 uitkomen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_micro():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. micro_f1: elke opdracht telt even zwaar

        Schrijf `micro_f1(gold, said, labels)`: tel de true positives, false positives en false negatives van alle labels in `labels`, die `counts` geeft, op in drie sommen, en geef één F1 score van de drie sommen terug.

        Bij het kleine voorbeeld zijn de sommen $TP = 0 + 5 = 5$, $FP = 0 + 1 = 1$ en $FN = 1 + 0 = 1$, dus `micro_f1(small_gold, small_said, small_labels)` geeft $10 / (10 + 1 + 1) = 0{,}8333$: bijna twee keer de macro F1. Vervang de `...`. De check roept dat aan en `micro_f1(more_gold, more_said, more_labels)`, en daarna dezelfde eigen gevallen als bij `macro_f1`.
        """),
        L.task("Reken de micro F1 uit: één F1 score van de aantallen van alle labels opgeteld."),
    ])
    return


@app.cell(hide_code=True)
def exercise_micro():
    box_micro = L.code_box("micro", lines=10)
    box_micro
    return (box_micro,)


@app.cell(hide_code=True)
def check_micro(box_micro):
    result_micro = L.check("micro", box_micro.value)
    result_micro
    return (result_micro,)


@app.cell(hide_code=True)
def hints_micro():
    L.hints(
        "micro",
        [
            "Zet `all_tp`, `all_fp` en `all_fn` eerst op 0, en ga de labels langs met `for label in labels:`. Haal bij elk label `tp, fp, fn` op met `counts(gold, said, label)`, en tel elk van de drie op bij de eigen som. Geef na de loop de F1 score van de drie sommen terug.",
            "Bij het kleine voorbeeld zijn de sommen 5, 1 en 1. Wat is de F1 score van die aantallen, en reken je de F1 score in de loop uit, of erna?",
            r"`        all_fp = all_fp + ___`",
        ],
        why=r"`all_tp`, `all_fp` en `all_fn` beginnen bij 0; `for label in labels:` met `tp, fp, fn = counts(gold, said, label)`, dan `all_tp = all_tp + tp`, `all_fp = all_fp + fp` en `all_fn = all_fn + fn`; na de loop `return 2 * all_tp / (2 * all_tp + all_fp + all_fn)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_micro():
    mo.md(r"""
    In de afbeelding is je micro F1 0,8194: hetzelfde getal als het aandeel van de testopdrachten dat de eigen intent krijgt, 2.437 van de 2.974. Dat is geen toeval. Een opdracht die de eigen intent krijgt, is één true positive, voor die intent. Een opdracht die een andere intent krijgt, is één false positive, voor de intent die de opdracht krijgt, en één false negative, voor de eigen intent. Dus $TP$ is het aantal opdrachten dat goed gaat, 2.437, en $FP$ en $FN$ zijn allebei het aantal dat fout gaat, 537:

    $$\text{micro } F_1 = \frac{2 \times 2437}{2 \times 2437 + 537 + 537} = \frac{2437}{2437 + 537} = 0{,}819$$

    Als elk voorbeeld precies één label heeft, en de classifier alleen labels uit de lijst geeft, zijn de micro-precision, de micro-recall en de micro F1 alle drie de accuracy. Jurafsky en Martin merken in een voetnoot op dat de micro-precision altijd gelijk is aan de micro-recall; in het voorbeeld uit het boek staat elk label dat de classifier geeft in de tabel. Zonder de prior geeft naive Bayes drie testopdrachten, zoals *baseball scores* (*honkbaluitslagen*), de intent cooking_query, die 4 trainingsopdrachten heeft en geen testopdrachten. Die drie fouten zijn false negatives voor de eigen intent, maar voor geen van de 59 labels een false positive. Over de 59 intents is de micro-precision dan 2.434 van de 2.971 en de micro-recall 2.434 van de 2.974, dus de micro F1, 0,8188, is iets hoger dan de accuracy, 0,8184.

    ### Hoe goed is het?

    De makers van MASSIVE geven per taal het aandeel testopdrachten dat de eigen intent krijgt, en voor de slots een F1 score met micro-averaging. Drie veel grotere modellen, met 258 tot 580 miljoen parameters en getraind op de trainingsopdrachten van alle 51 talen samen, hebben 87,9% tot 89,0% van de Engelse testopdrachten goed; naive Bayes, dat woorden telt, haalt 81,9%. Op de Nederlandse opdrachten halen dezelfde modellen 86,8% tot 87,6%, en naive Bayes 79,9%, met een macro F1 van 0,730: daar krijgen vier intents 0. De Nederlandse vertalingen bevatten hier en daar fouten, zoals *maakt mijn wakker om negen uur in de ochtend op vrijdag* voor *wake me up at nine am on friday*.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: minder intents

    Wat als de classifier alleen de intents kende die het vaakst voorkomen? Met de slider kies je hoeveel van de 60 intents naive Bayes kent, geteld vanaf de intent met de meeste trainingsopdrachten. Naive Bayes leert elk van die intents uit dezelfde trainingsopdrachten als eerst, en geeft elke testopdracht de bekende intent met de hoogste score, dus de opdrachten van een intent die naive Bayes niet kent, gaan altijd fout. De afbeelding laat de F1 score van elke intent zien, met de onbekende intents in het paarse vlak, en de twee gemiddelden, uitgerekend door je `micro_f1` en `macro_f1` zodra oefening 2 en 3 goed zijn, en tot dan door de functies van de course.
    """)
    return


@app.cell(hide_code=True)
def demo_cell(box_macro, box_micro, result_macro, result_micro):
    _names = C.EXERCISES["macro"].names
    _macro = C.mine(box_macro.value, result_macro.state == "correct", "macro_f1", _names)
    _micro = C.mine(box_micro.value, result_micro.state == "correct", "micro_f1", _names)
    _values = C.demo_values(_macro, _micro)
    demo_known = L.explore("known", "bekende intents", lambda _k: C.demo_frame(_k, _values, L.lang),
                           start=60, values=C.KNOWN, role="",
                           task="Schuif naar 30 intents, ongeveer de helft. Hoeveel zakt elk gemiddelde?")
    demo_known
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met 30 intents kent naive Bayes nog de intent van 2.382 van de 2.974 testopdrachten, 80%, en de micro F1 zakt van 0,819 naar 0,677. De macro F1 valt naar 0,380: 29 van de 59 intents krijgen nu 0. Met 50 intents horen maar 74 testopdrachten bij een intent die naive Bayes niet kent: de micro F1 gaat van 0,819 naar 0,806, de macro F1 van 0,752 naar 0,682. Met 1 krijgt elke opdracht calendar_set: een micro F1 van 0,070, de 209 opdrachten van calendar_set op 2.974, en een macro F1 van 0,002. Een classifier die de zeldzame labels weglaat, kan er op de micro F1 bijna even goed uitzien als een classifier die dat niet doet; de macro F1 laat het verschil zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prior():
    mo.md(r"""
    ### Oefening 4. Met of zonder de prior?
    """)
    return


@app.cell(hide_code=True)
def choice_prior_cell():
    choice_prior = L.choice(
        "prior",
        question="De score van naive Bayes begint met de log van het aandeel van de trainingsopdrachten van elke intent: de prior. Zonder de prior, alsof elke intent even vaak voorkomt, geeft naive Bayes de 29 intents van de testopdrachten met de minste trainingsopdrachten aan 635 testopdrachten in plaats van 534; deze 29 intents zijn de echte intent van 592 testopdrachten. Dan gaan 2.434 testopdrachten goed in plaats van 2.437, en de macro F1 wordt 0,763 in plaats van 0,752. Welke van de twee is beter?",
        options={
            "prior": "Met de prior, omdat er meer opdrachten goed gaan",
            "flat": "Zonder, omdat de macro F1 dan hoger is",
            "right": "Dat hangt af van welk gemiddelde de taak nodig heeft",
            "same": "Allebei even goed, omdat de micro F1 bij beide 0,819 is",
        },
        correct="right",
        why={
            "right": "De micro F1 telt elke opdracht: met de prior gaan er 3 van de 2.974 meer goed, en de micro F1 is 0,8194 tegen 0,8188. De macro F1 telt elke intent: zonder de prior worden de zeldzame intents vaker gegeven, music_dislikeness gaat van 0 naar 0,571 en general_greet van 0 naar 0,333. Zelfs calendar_set krijgt een hogere F1 score, 0,903 in plaats van 0,888, omdat minder opdrachten van andere intents calendar_set krijgen. De 29 zeldzame intents zijn de echte intent van 592 opdrachten: met de prior geeft naive Bayes die intents te weinig, 534 keer, en zonder de prior iets te vaak, 635 keer. Welke versie beter is, hangt af van wat er voor de taak telt: elke opdracht, of elke intent.",
            "prior": "Meer opdrachten goed is wat de micro F1 telt, en het verschil is 3 van de 2.974. De macro F1 zegt het omgekeerde. Welk gemiddelde telt, hangt af van de taak. Zie *Het idee*.",
            "flat": "Een hogere macro F1 betekent dat de intents het gemiddeld beter doen, maar er gaan 3 opdrachten meer fout. Welk gemiddelde telt, hangt af van de taak. Zie *Het idee*.",
            "same": "De micro F1 is ongeveer gelijk, maar de macro F1 verschilt 0,012: zonder de prior doen de zeldzame intents het beter. Zie *Het idee*.",
        },
    )
    choice_prior
    return (choice_prior,)


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 5. Welk gemiddelde?

    Elke situatie vraagt om één getal voor een classifier met veel labels. Zet elke situatie bij het gemiddelde dat er het best bij past.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_average = L.sort("which-average")
    sort_average
    return (sort_average,)


@app.cell(hide_code=True)
def check_sort(sort_average):
    result_sort = L.check("which-average", sort_average)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) gaf één F1 score voor één positief label; deze les nam elk label om de beurt als positief, bij de intents van MASSIVE, en maakte op twee manieren één getal van de 59 F1 scores. Macro-averaging laat elk label even zwaar tellen, dus de zeldzame intents die naive Bayes mist, trekken de macro F1 omlaag; micro-averaging laat elke opdracht even zwaar tellen, en met één intent per opdracht is de micro F1 de accuracy. Dezelfde twee gemiddelden meten elke classifier met veel labels, wat er binnenin ook gebeurt. De volgende les van het domein, *Project: een spamfilter met TF-IDF*, bouwt een hele classifier voor echte sms-berichten.

    **Een voorbeeld uit het dagelijks leven.** Een rapport geeft een cijfer voor elk vak. Het gemiddelde van de cijfers is als de macro F1: een klein vak telt even zwaar als een groot vak. Alle goede antwoorden van alle toetsen samen tellen is als de micro F1: een vak met veel toetsen telt zwaarder.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_prior, result_counts, result_macro, result_micro, result_sort):
    L.wrap_up(result_counts, result_macro, result_micro, choice_prior, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
