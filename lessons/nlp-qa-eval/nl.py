# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-qa-eval. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-qa-eval", "nl")
    C = L.checks   # de vragen van SQuAD en SQuAD-NL met elk antwoord dat mensen aanwezen, de scores, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Het antwoord vinden in een passage](/course/open/nlp-extractive-qa?lang=nl), en één over [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_people():
    quick_people = L.choice(
        "quick-people",
        question="In *Het antwoord vinden in een passage* is elk antwoord van SQuAD een span die mensen aanwezen in een alinea van Wikipedia. Hoeveel mensen wezen een antwoord aan bij een vraag van de development set?",
        options={
            "one": "Eén: de persoon die de vraag schreef",
            "right": "Minstens drie: wie de vraag schreef, en nog minstens twee anderen",
            "program": "Niemand: een programma wees de antwoorden aan, en mensen controleerden alles achteraf",
            "two": "Precies twee, zodat de een de ander kan controleren",
        },
        correct="right",
        why={
            "right": "Bij de development set en de test set werd nog minstens twee mensen gevraagd om elk een antwoord aan te wijzen bij elke vraag, naast wie de vraag schreef. De spans verschillen vaak een beetje, en deze les scoort een antwoord tegen allemaal.",
            "one": "Dat geldt voor de training set. Wat voegde SQuAD toe bij de vragen van de development set? Zie [Het antwoord vinden in een passage](/course/open/nlp-extractive-qa?lang=nl).",
            "program": "Mensen op een platform voor crowdsourcing markeerden de antwoorden zelf. Hoeveel mensen bij elke vraag van de development set? Zie [Het antwoord vinden in een passage](/course/open/nlp-extractive-qa?lang=nl).",
            "two": "Het waren er meer dan twee. Wie wees er een antwoord aan, naast wie de vraag schreef? Zie [Het antwoord vinden in een passage](/course/open/nlp-extractive-qa?lang=nl).",
        },
    )
    quick_people
    return (quick_people,)


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="In *Precision en recall in één getal: de F1 score* is de F1 score het harmonisch gemiddelde van precision en recall. Een detector heeft een precision van 0,5 en een recall van 1. Wat is de F1 score?",
        options={
            "mean": "0,75",
            "min": "0,5",
            "right": "Ongeveer 0,67",
            "one": "1, want de recall is perfect",
        },
        correct="right",
        why={
            "right": "Twee keer het product, gedeeld door de twee opgeteld: 2 × 0,5 × 1 / (0,5 + 1) = 1 / 1,5, ongeveer 0,67. Het harmonisch gemiddelde blijft dicht bij de kleinste van de twee. Deze les rekent precision en recall uit over de woorden van een antwoord.",
            "mean": "0,75 is het gewone gemiddelde van 0,5 en 1. Welk gemiddelde blijft dicht bij het kleinste getal? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "min": "0,5 is de kleinste van de twee. Is het harmonisch gemiddelde van 0,5 en 1 precies de kleinste? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "one": "De F1 score is alleen 1 als beide 1 zijn. Wat doet een precision van 0,5 ermee? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een leraar die een toets nakijkt, vergelijkt elk antwoord met het antwoordmodel. Daar staat *1850*; de ene leerling schrijft *in 1850*, de andere *het jaar 1850*, en de leraar geeft allebei het punt. Een computer die duizenden antwoorden scoort, kan niet meelezen. De computer heeft dus een regel nodig voor wanneer een antwoord goed telt, en voor hoe dicht een bijna goed antwoord erbij komt. SQuAD kwam met twee zulke regels, exact match en F1, en systemen voor reading comprehension worden er sindsdien mee gescoord.

    In deze les schrijf je allebei, en meet je ermee hoe goed mensen het met elkaar eens zijn, en hoe ver een hele zin, het antwoord van je reader uit *Het antwoord vinden in een passage*, van het antwoord af zit.

    **Een voorbeeld uit het dagelijks leven.** Bij een pubquiz staat op het antwoordblad *Amsterdam*. Een team schrijft *de stad Amsterdam*: de meeste quizmasters geven het punt, een strenge quizmaster niet, en sommige geven een half punt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_people = L.guess(
        "guess",
        question="In SQuAD wezen meerdere mensen een antwoord aan bij elke vraag van de development set. Neem het antwoord van de tweede persoon, en vergelijk het met de antwoorden van de anderen, zonder hoofdletters, leestekens en de woorden *a*, *an* en *the*. Bij hoeveel van de 1.190 Engelse vragen is het precies gelijk aan minstens één ervan?",
        options={
            "half": "Ongeveer de helft",
            "two-thirds": "Ongeveer twee derde",
            "right": "Ongeveer vier op de vijf",
            "all": "Bijna allemaal, op een enkele na",
        },
        reveal="Ongeveer vier op de vijf: 980 van de 1.190, of 82,4%. Mensen die dezelfde alinea en dezelfde vraag lezen, wijzen bij ongeveer één op de zes vragen toch een andere span aan: de een *between 1500 and 1850*, de ander *1500 and 1850*. Over het antwoord is men het meestal eens, maar niet over hoeveel tekst eromheen erbij hoort. Met een deel van de punten voor de woorden die in beide antwoorden staan, de F1 score van deze les, scoren mensen 91,8%. Dit kan alleen met de Engelse vragen: SQuAD-NL houdt bij de meeste vragen elk verschillend antwoord één keer.",
    )
    guess_people
    return (guess_people,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Het antwoord van een reader is een span, een stuk van de alinea. Om het te scoren, vergelijk je het met de **referentieantwoorden**: de spans die mensen bij dezelfde vraag aanwezen. In de development set van SQuAD wees wie de vraag schreef een antwoord aan, en werd nog minstens twee mensen gevraagd om er elk een aan te wijzen; wie in de alinea geen antwoord vond, mocht de vraag zonder antwoord insturen. 1.175 van de 1.190 Engelse vragen van deze les hebben drie referentieantwoorden of meer, de andere 15 twee. SQuAD-NL houdt bij de meeste vragen elk verschillend antwoord één keer: 604 van de 1.183 Nederlandse vragen hebben er één. Rajpurkar en anderen scoorden een antwoord op twee manieren, en lieten bij allebei de leestekens en de lidwoorden (*a*, *an* en *the*) buiten beschouwing.

    **Exact match** rekent een antwoord goed als het gelijk is aan één van de referentieantwoorden, nadat beide genormaliseerd zijn: in kleine letters, zonder leestekens en zonder lidwoorden. In het paper meet exact match het percentage antwoorden dat precies gelijk is aan één van de referentieantwoorden. Dus *De Horniman* is gelijk aan *Horniman*, maar *in 2018* niet aan *2018*, al zeggen de twee hetzelfde.

    **Token F1** geeft een deel van de punten voor de woorden die een antwoord en een referentieantwoord delen. Het paper behandelt een antwoord en een referentieantwoord als bags met tokens, en berekent daarvan de F1. Een bag is een soort set waarin een woord vaker kan staan; een woord dat beide twee keer hebben, is twee keer gedeeld. Precision is het deel van de woorden van het antwoord dat gedeeld is, recall het deel van de woorden van het referentieantwoord, en F1 het harmonisch gemiddelde van de twee, zoals in *Precision en recall in één getal*. Een antwoord met te veel woorden scoort lager op precision, een antwoord met ontbrekende woorden lager op recall.

    Een vraag heeft meerdere referentieantwoorden, en SQuAD neemt de beste: de hoogste F1 over alle referentieantwoorden van een vraag, en daarna het gemiddelde over alle vragen. Zo rekent exact match een vraag ook goed als het antwoord gelijk is aan één van de referentieantwoorden. Het plaatje scoort één antwoord op een vraag over de pest tegen de referentieantwoorden ervan.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **referentieantwoord** | een antwoord dat mensen aanwezen, waarmee het antwoord van een systeem vergeleken wordt; SQuAD spreekt van *ground truth answers* |
    | **exact match** (EM) | een antwoord telt alleen als goed als het gelijk is aan een referentieantwoord, nadat beide genormaliseerd zijn; over veel vragen het deel dat goed is |
    | **token F1** | de F1 score van de woorden van een antwoord tegen de woorden van een referentieantwoord: precision over het antwoord, recall over het referentieantwoord |
    | **bag** | een verzameling zoals een set, waarin iets vaker kan staan: *zwembad zwembad* bevat *zwembad* twee keer |
    | **lidwoorden** | *de*, *het* en *een* in het Nederlands; *a*, *an* en *the* in het Engels |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de antwoorden van een reader te scoren zoals SQuAD dat doet:

    1. Normaliseer het antwoord en elk referentieantwoord: de woorden in kleine letters, zonder leestekens en zonder *de*, *het* en *een*.
    2. Exact match: 1 als het antwoord dezelfde woorden heeft, in dezelfde volgorde, als minstens één referentieantwoord; anders 0.
    3. Token F1 tegen één referentieantwoord: tel de gedeelde woorden, elk zo vaak als het in beide staat. Zonder gedeelde woorden is F1 gelijk aan 0. Anders is precision = gedeeld / woorden van het antwoord, recall = gedeeld / woorden van het referentieantwoord, en F1 = 2 × precision × recall / (precision + recall).
    4. Neem voor elke vraag de beste exact match en de beste F1 over de referentieantwoorden.
    5. Neem van elk het gemiddelde over de vragen, als percentage.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem de vraag *In welke jaren was de pest aanwezig in islamitische landen?* De referentieantwoorden zijn *tussen 1500 en 1850* en *1500 en 1850*. Stel dat een reader *tussen 1500 en 1850 vrijwel elk jaar* aanwijst: goed, maar met drie woorden te veel. Het is niet gelijk aan een referentieantwoord, dus de exact match is 0. Met het eerste referentieantwoord deelt het vier woorden, *tussen*, *1500*, *en* en *1850*. Een paar regels rekenen precision, recall en F1 uit; `C.words` is de `words` van *Soorten question answering*:
    """)
    return


@app.cell
def example_cell():
    _answer = C.words("tussen 1500 en 1850 vrijwel elk jaar")
    _reference = C.words("tussen 1500 en 1850")
    print(_answer)
    print(_reference)
    _precision = 4 / len(_answer)
    _recall = 4 / len(_reference)
    print(_precision, _recall, 2 * _precision * _recall / (_precision + _recall))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het antwoord heeft 7 woorden, en 4 ervan zijn gedeeld, dus de precision is 4/7 = 0,57. Alle 4 woorden van het referentieantwoord staan in het antwoord, dus de recall is 1. De F1 score is 2 × 0,57 × 1 / 1,57 = 0,73. Tegen *1500 en 1850* is de precision 3/7 en de F1 score 0,6. SQuAD neemt de beste: bij deze vraag exact match 0 en F1 0,73. De drie woorden te veel kosten een kwart van de score; de hele zin, met nog meer woorden, zou meer kosten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_match():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. exact_match: hetzelfde antwoord?

        Schrijf `exact_match`, dat `prediction` krijgt, het antwoord van een reader, `answer`, een referentieantwoord, en `articles`, een set lidwoorden zoals `C.ARTICLES["nl"]`, met *de*, *het* en *een*. Geef 1 terug als de twee dezelfde woorden in dezelfde volgorde hebben nadat beide genormaliseerd zijn, en anders 0. Het vak kent `words`, dat de woorden van een tekst geeft in kleine letters, zonder leestekens. In de startcode moet `p` de woorden van `prediction` zonder lidwoorden bevatten, en `a` de woorden van `answer`; de regels erna vergelijken de twee. Vervang de twee `...`.

        De check roept `exact_match("Het zwembad.", "zwembad", C.ARTICLES["nl"])` aan, wat `1` geeft. Hetzelfde in het Engels, met `C.ARTICLES["en"]`: *a*, *an* en *the*. Daarna eigen gevallen: woorden in een andere volgorde, een woord te veel, een hoofdletter of een punt, een woord dat twee keer staat, *the* binnen een langer woord, en antwoorden van SQuAD.
        """),
        L.task("Zeg of een antwoord gelijk is aan een referentieantwoord, nadat beide genormaliseerd zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_match():
    box_match = L.code_box("exact_match", lines=6)
    box_match
    return (box_match,)


@app.cell(hide_code=True)
def check_match(box_match):
    result_match = L.check("exact_match", box_match.value)
    result_match
    return (result_match,)


@app.cell(hide_code=True)
def hints_match():
    L.hints(
        "exact_match",
        [
            "`p` is een list comprehension over de woorden die `words` van `prediction` geeft, en houdt elk woord `w` dat niet in `articles` staat; `a` is hetzelfde voor `answer`. Twee lijsten zijn gelijk met `==` als de lijsten dezelfde items in dezelfde volgorde bevatten.",
            "Houd de woorden als lijst, niet als set: een set vergeet de volgorde, dus *1850 and 1500* zou gelijk zijn aan *1500 and 1850*. En gebruik `words`, niet `split`: `split` houdt de hoofdletters en de punten.",
            "`    a = [w for w in ___ if w not in articles]`",
        ],
        why="`p = [w for w in words(prediction) if w not in articles]` en `a = [w for w in words(answer) if w not in articles]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_match():
    mo.md(r"""
    Normaliseren maakt verschil. Van de 1.190 Engelse vragen is het antwoord van de tweede persoon bij 76,3% precies dezelfde tekst als een antwoord van iemand anders. Zonder hoofdletters en leestekens is 78,7% gelijk, en ook zonder lidwoorden 82,4%. Het script van SQuAD normaliseert op dezelfde manier, met één verschil: het schrapt de leestekens binnen een woord en plakt de twee delen aan elkaar, dus *27-30* wordt *2730*, waar `words` er *27* en *30* van maakt. Op deze vragen verandert dat geen enkele exact match, en gaat de F1 score van mensen van 91,8% naar 91,6%. Voor het Nederlands laat de course *de*, *het* en *een* weg.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_f1():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. token_f1: een deel van de punten

        Schrijf `token_f1`, dat `prediction`, `answer` en `articles` krijgt zoals in oefening 1, en de F1 score teruggeeft van de woorden van `prediction` tegen de woorden van `answer`. De startcode maakt `p` en `a` zoals in oefening 1, en `same`, het aantal woorden dat de twee delen, met de `shared` van de course, die een woord zo vaak telt als het in beide staat: `shared(["pool", "pool"], ["pool"])` geeft `1`. Hebben de twee geen enkel woord gemeen, dan zijn precision en recall allebei 0, en zou de F1 score door 0 delen, dus de startcode geeft eerst `0.0` terug. Vervang de drie `...`: `precision`, `recall` en de F1 score.

        De check roept `token_f1("om 9 uur", "9 uur", C.ARTICLES["nl"])` aan, wat `0.8` geeft: *9* en *uur* zijn gedeeld, dus de precision is 2/3 en de recall 1. Hetzelfde in het Engels, en daarna eigen gevallen: een antwoord met te veel woorden, een antwoord met ontbrekende woorden, een woord dat twee keer staat, en de antwoorden van het uitgewerkte voorbeeld.
        """),
        L.task("Reken de F1 score uit van de woorden die een antwoord met een referentieantwoord deelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("token_f1", lines=10)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("token_f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "token_f1",
        [
            "`precision` is `same` gedeeld door het aantal woorden in `p`, met `len`; `recall` is `same` gedeeld door het aantal woorden van het referentieantwoord. De `return` geeft het harmonisch gemiddelde van de twee, zoals in *Precision en recall in één getal*: twee keer het product van `precision` en `recall`, gedeeld door `precision` plus `recall`.",
            "Tel woorden, geen tekens: `len(prediction)` is het aantal tekens van de tekst, `len(p)` het aantal woorden. En zet haakjes om `precision + recall`: Python deelt eerst en telt daarna op.",
            "`    recall = same / ___`",
        ],
        why="`precision = same / len(p)`, `recall = same / len(a)` en `return 2 * precision * recall / (precision + recall)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_f1():
    mo.md(r"""
    Als de antwoorden van twee mensen verschillen, delen de antwoorden meestal toch woorden. Van de 1.190 Engelse vragen is het antwoord van de tweede persoon bij 210 aan geen enkel ander antwoord precies gelijk. Bij 171 daarvan deelt het woorden met minstens één ander antwoord, en slechts bij 39 met geen enkel. Rajpurkar en anderen zagen hetzelfde: mensen verschillen vooral doordat de een een bijzaak wel meeneemt en de ander niet, zoals *monsoon trough* tegenover *movement of the monsoon trough*, en zelden doordat men het oneens is over het antwoord. De mensen die de extra antwoorden toevoegden, kregen de opdracht om de kortste span te kiezen die de vraag beantwoordt. Bij de vraag over de pest is het eerste antwoord *between 1500 and 1850*, en zijn de twee antwoorden erna *1500 and 1850*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. squad_scores: de scores over veel vragen

        Schrijf `squad_scores`, dat `predictions` krijgt, de antwoorden van een reader op een lijst vragen, `answer_lists`, bij elke vraag de lijst met de referentieantwoorden, en `articles`. Geef de exact match en de F1 score over alle vragen terug, in procenten: bij elke vraag de beste score van het antwoord over de referentieantwoorden, en daarna het gemiddelde over de vragen. Het vak kent de `exact_match` en de `token_f1` van de course, dus je antwoorden op oefening 1 en 2 doen hier niet mee. In de startcode tellen `em` en `f1` de beste score van elke vraag op, en de `return` maakt van de totalen percentages. Vervang de twee `...`.

        De check roept `squad_scores(["om 9 uur", "het zwembad"], [["9 uur"], ["het grote zwembad", "zwembad"]], C.ARTICLES["nl"])` aan, wat `(50.0, 90.0)` geeft: *om 9 uur* is aan geen enkel referentieantwoord gelijk en heeft F1 0,8, en *het zwembad* is precies gelijk aan *zwembad*. Hetzelfde in het Engels, en daarna alle vragen met als antwoorden de zinnen die de reader van *Het antwoord vinden in een passage* kiest (in de les: `C.reader_answers("nl")` en `C.answer_lists("nl")`), en de antwoorden van mensen.
        """),
        L.task("Scoor veel antwoorden zoals SQuAD dat doet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("squad_scores", lines=8)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("squad_scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "squad_scores",
        [
            "Bij vraag `i` is het antwoord `predictions[i]`, en staan de referentieantwoorden op plaats `i` van `answer_lists`. Een list comprehension geeft `exact_match` van het antwoord met elk referentieantwoord `a`, en `articles`; `max` neemt de beste daarvan, en `em` gaat daarmee omhoog. De regel van `f1` doet hetzelfde met `token_f1`.",
            "Neem de beste, niet het eerste referentieantwoord: *het zwembad* is precies gelijk aan *zwembad*, het tweede referentieantwoord. En houd één score per vraag: een loop over de referentieantwoorden die elke score optelt, telt een vraag meer dan één keer.",
            "`        em = em + max([exact_match(predictions[i], a, articles) for a in ___])`",
        ],
        why="`em = em + max([exact_match(predictions[i], a, articles) for a in answer_lists[i]])` en `f1 = f1 + max([token_f1(predictions[i], a, articles) for a in answer_lists[i]])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_scores():
    mo.md(r"""
    Mensen scoren tegen elkaar 82,4% exact match en 91,8% F1 op de Engelse vragen; op de hele development set maten Rajpurkar en anderen op deze manier 80,3% en 90,5%. De zin met het antwoord scoort op de Nederlandse vragen 0,3% en 23,6%. De recall is bijna altijd 1, want de zin bevat het hele antwoord, maar de zin heeft 27,8 woorden zonder lidwoorden, tegen ongeveer 3 bij een antwoord, dus de precision is laag. De zin die je reader kiest, scoort 17,6% F1, iets onder de 20,2% van het sliding window, een eenvoudige baseline van het paper op de hele Engelse development set, en de eerste zin van de alinea 9,0%. De logistic regression van het paper, die een span aanwijst, haalde 51,0% F1: het meeste van de score zit in het aanwijzen van de span.

    ### Probeer het: scoor een antwoord

    Kies een vraag, typ een antwoord, en zie hoe het scoort tegen elk referentieantwoord dat mensen aanwezen. Zolang je niets typt, scoort het vak de zin die de reader van *Het antwoord vinden in een passage* kiest. Het vak laat de alinea's weg die de course niet laat zien: het heeft 260 van de 1.183 vragen.
    """)
    return


@app.cell(hide_code=True)
def box_number():
    try_n = mo.ui.slider(1, C.box_count(L.lang), value=1, label="Vraag", show_value=True)
    try_n
    return (try_n,)


@app.cell(hide_code=True)
def box_answer():
    try_answer = mo.ui.text(value="", label="Je antwoord", full_width=True)
    try_answer
    return (try_answer,)


@app.cell(hide_code=True)
def box_cell(try_answer, try_n):
    mo.Html(C.box_picture(L.lang, try_n.value, try_answer.value))
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.md(r"""
    ### Oefening 4. Waarom de beste?

    Bij elke vraag neemt SQuAD de beste score over de referentieantwoorden, en niet het gemiddelde ervan, en ook niet de score tegen het eerste referentieantwoord.
    """)
    return


@app.cell(hide_code=True)
def choice_best_cell():
    why_best = L.choice(
        "why_best",
        question="Waarom neemt SQuAD de beste score over de referentieantwoorden van een vraag?",
        options={
            "fast": "De beste is sneller uit te rekenen dan het gemiddelde van allemaal",
            "right": "Mensen wijzen verschillende spans aan voor één antwoord; wie gelijk is aan één ervan, doet het zo goed als een mens",
            "long": "Zodat een langer antwoord, met meer van de woorden van de mensen, altijd hoger scoort dan een kort antwoord met minder woorden",
            "first": "Het eerste referentieantwoord telt, en de andere beslissen alleen bij een gelijkspel",
        },
        correct="right",
        why={
            "right": "Mensen die dezelfde alinea lezen, wijzen *2018* of *in 2018* aan, en beide zijn goed. Alleen tegen het eerste van de andere antwoorden zouden mensen tegen elkaar 68,1% exact match scoren in plaats van 82,4%, en 84,6% F1 in plaats van 91,8%. De beste over allemaal vraagt dat een antwoord gelijk is aan het antwoord van één persoon, niet aan dat van iedereen. Alleen verschillende antwoorden geven meer keus: een kopie van een antwoord verandert geen enkele beste score. 554 van de 1.190 Engelse vragen en 643 van de 1.183 Nederlandse hebben slechts één antwoordtekst, en daar scoort een antwoord dat in andere woorden goed is laag.",
            "fast": "Allebei zijn snel uit te rekenen. Wat hebben de referentieantwoorden van één vraag gemeen?",
            "long": "Een antwoord met te veel woorden scoort lager op precision. Geeft de beste over de referentieantwoorden lange antwoorden een voordeel?",
            "first": "Alle referentieantwoorden tellen even zwaar. Wat gebeurt er met een antwoord dat precies gelijk is aan het tweede?",
        },
    )
    why_best
    return (why_best,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Het antwoord vinden in een passage](/course/open/nlp-extractive-qa?lang=nl) koos je reader een zin; hier scoorde je antwoorden zoals SQuAD dat doet, met de precision, recall en F1 score van [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) over woorden in plaats van voorbeelden, na het normaliseren van [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl). Mensen zijn het bij 82,4% van de antwoorden precies eens, en scoren tegen elkaar 91,8% F1; een hele zin bevat het antwoord, maar scoort laag. De volgende les, *Vertalen met regels*, begint met machinevertaling, en later scoort *Een vertaling scoren: BLEU en chrF* vertalingen tegen referentievertalingen op een vergelijkbare manier: met gedeelde woorden, precision en recall.

    **Een voorbeeld uit het dagelijks leven.** Een taalapp die je een zin laat vertalen, accepteert vaak meerdere vertalingen, en rekent je antwoord goed als het gelijk is aan één ervan: de beste over de referentieantwoorden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_match, result_f1, result_scores, why_best):
    L.wrap_up(result_match, result_f1, result_scores, why_best)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
