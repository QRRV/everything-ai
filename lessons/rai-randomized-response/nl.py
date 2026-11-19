# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-randomized-response. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-randomized-response", "nl")
    C = L.checks   # een verzonnen enquête: TRUTH, de munten FIRST en SECOND, ANSWERS, en acht mensen FEW_*


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Tabellen koppelen](/course/open/rai-linkage-attacks?lang=nl), [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) en [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_name():
    quick_name = L.choice(
        "quick-name",
        question="Een vrijgegeven bezoek past bij precies één kiezer, op een kiezerslijst met ongeveer twee derde van het stadje. Wat levert die match op?",
        options={
            "right": "Een naam die fout kan zijn",
            "sure": "Een naam die zeker klopt",
            "none": "Geen naam: één match is te weinig",
        },
        correct="right",
        why={
            "right": "Iemand met dezelfde waarden die niet op de lijst staat, zie je niet, dus de ene match kan de verkeerde persoon noemen: in het verzonnen stadje waren 48 van de 622 namen fout.",
            "sure": "De lijst bevat niet iedereen, dus iemand met dezelfde waarden kan erop ontbreken. Zie *Een naam uit een onvolledige lijst* in *Het idee* van [Tabellen koppelen](/course/open/rai-linkage-attacks?lang=nl).",
            "none": "Een vrijgegeven rij die bij precies één kiezer past, krijgt de naam van die kiezer; de vraag is of die naam klopt. Zie *Twee tabellen, één sleutel* in *Het idee* van [Tabellen koppelen](/course/open/rai-linkage-attacks?lang=nl).",
        },
    )
    quick_name
    return (quick_name,)


@app.cell(hide_code=True)
def quick_check_prob():
    quick_prob = L.choice(
        "quick-prob",
        question="Van 200 antwoorden zijn er 50 ja. Wat is de kans op ja, geteld uit deze antwoorden?",
        options={
            "count": "50, het aantal ja-antwoorden",
            "right": "0,25, of 50 van de 200",
            "coin": "0,5: het is ja of nee",
        },
        correct="right",
        why={
            "right": "Een kans die je uit pogingen telt, is hoe vaak het gebeurde, gedeeld door het aantal pogingen: 50 / 200 = 0,25.",
            "count": "50 is hoe vaak het gebeurde; een kans deelt dat door het aantal pogingen en ligt tussen 0 en 1. Zie *Het idee* van [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "coin": "Twee mogelijke uitkomsten hoeven niet even waarschijnlijk te zijn; geteld uit deze antwoorden kwam ja 50 keer op 200. Zie *Het idee* van [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
        },
    )
    quick_prob
    return (quick_prob,)


@app.cell(hide_code=True)
def quick_check_rng():
    quick_rng = L.choice(
        "quick-rng",
        question="`rng = np.random.default_rng(7)`. Welke getallen kan `rng.integers(0, 2, size=5)` geven?",
        options={
            "three": "0, 1 en 2",
            "right": "Alleen 0 en 1",
            "floats": "Elk getal van 0 tot 2",
        },
        correct="right",
        why={
            "right": "`rng.integers(low, high)` geeft gehele getallen tot `high`, zonder `high` zelf: hier 0 en 1, net als een munt.",
            "three": "`high` hoort er niet bij: `rng.integers(0, 2)` geeft 0 of 1. Zie *Nieuwe woorden en tekens* in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "floats": "`rng.integers` geeft gehele getallen; `rng.random` geeft floats. Zie *Nieuwe woorden en tekens* in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
        },
    )
    quick_rng
    return (quick_rng,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Stel mensen een vraag die ze liever niet beantwoorden, en een deel van hen zegt niet de waarheid. Stanley Warner noemde de fout die daardoor ontstaat in 1965 een vertekening door ontwijkende antwoorden, en bedacht een uitweg: iedereen voegt toeval toe aan het eigen antwoord, zodat geen enkel antwoord iemand verraadt, en wie de antwoorden optelt, corrigeert voor dat toeval. Local differential privacy, een manier om zulke bescherming te meten, heeft zijn wortels in dat idee, en Google en Apple hebben er versies van gebruikt om gegevens van hun gebruikers te verzamelen. In deze les houd je een verzonnen enquête onder 10.000 mensen met munten, schat je uit de antwoorden hoeveel er in werkelijkheid ja zeiden, en meet je hoeveel privacy de munten geven.

    **Een voorbeeld uit het dagelijks leven.** Een docent vraagt wie het huiswerk heeft overgeschreven, en er gaat nauwelijks een hand omhoog. De volgende keer gooit iedereen eerst onder de tafel een munt op: bij kop gaat de hand omhoog, wat de waarheid ook is; bij munt alleen bij een ja. Een opgestoken hand zegt nu weinig over wie dan ook, en toch kan de docent ongeveer uitrekenen hoeveel leerlingen hebben overgeschreven.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="In een voorbeeld wil Warner het aandeel van een groep weten, ongeveer de helft, met een standaardafwijking van 0,05. Rechtstreeks gevraagd, als iedereen de waarheid zegt, zijn ongeveer 100 interviews genoeg. Hoeveel zijn er nodig met de draaischijf die Warner voorstelde, ingesteld op p = 0,75, die je hieronder tegenkomt?",
        options={
            "same": "Zo'n 100, ongeveer evenveel",
            "twice": "Rond de 200",
            "right": "Ongeveer 400",
        },
        reveal="**Ongeveer 400**: Warner (1965) rekent uit dat de draaischijf zo'n 400 interviews nodig heeft, waar de directe vraag het met ongeveer 100 af kan. Het toeval dat iedereen beschermt, maakt de groep ook vager, dus zijn er meer antwoorden nodig. In Oefening 2 zie je een vergelijkbare prijs met twee munten.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een draaischijf in plaats van een antwoord

    Warner (1965) wilde af van de vertekening door ontwijkende antwoorden: de fout in een enquête die ontstaat doordat mensen een gevoelige vraag niet naar waarheid beantwoorden. Het hulpmiddel was een draaischijf. Elke interviewer krijgt dezelfde schijf, waarvan de wijzer met kans p op de letter A uitkomt en met kans 1 − p op de letter B. Bij elk interview draait de geïnterviewde de schijf zonder dat de interviewer meekijkt, en zegt alleen of de wijzer op de letter van de eigen groep staat. Groep A kunnen de mensen zijn die het gevoelige ding deden. De interviewer hoort alleen ja of nee, en ziet de schijf nooit. De titel van Warner noemt de methode al: **randomized response**.

    Eén antwoord zegt dan weinig, maar de antwoorden van velen zeggen samen nog iets. Met een aandeel π van de mensen in groep A komt een ja met kans πp + (1 − π)(1 − p), en Warner lost π daaruit op, uit het aandeel ja. De schijf moet wel naar één kant overhellen: bij p = 1/2 geeft de geïnterviewde helemaal geen informatie, en bij p = 1 is het gewoon de directe vraag. Daartussen kost de privacy antwoorden: met π = 0,5 en p = 0,75 heeft Warner ongeveer 400 interviews nodig voor een standaardafwijking van 0,05, tegen maar ongeveer 100 voor de directe vraag, als iedereen de waarheid zou zeggen.

    ### Twee munten

    Cynthia Dwork en Aaron Roth (2014) beschrijven een latere variant van het idee van Warner, met munten. Wie meedoet, krijgt de opdracht: gooi een munt op; valt die op munt, antwoord dan naar waarheid; valt die op kop, gooi dan een tweede munt op en antwoord ja bij kop en nee bij munt. Ook dit is randomized response: iedereen voegt toeval toe aan het eigen antwoord voordat die het geeft.

    Iemand bij wie de waarheid ja is, antwoordt ja met kans 1/2 + 1/4 = 3/4: de helft van de keren vanuit de waarheid, een kwart van de keren door de tweede munt. Iemand bij wie de waarheid nee is, antwoordt ja met kans 1/4. Zelfs een ja is dus niet belastend, schrijven Dwork en Roth, want dat antwoord komt met een kans van minstens 1/4, of iemand het nu gedaan heeft of niet. Bij een echt aandeel p ja is het aandeel ja in de antwoorden gemiddeld (1/4)(1 − p) + (3/4)p = 1/4 + p/2, en omgekeerd: schat p als twee keer het aandeel ja-antwoorden, min 1/2. Twee zuivere munten geven een ja met dezelfde kansen als de draaischijf van Warner met p = 3/4 (daar is p de kans dat de wijzer op A uitkomt, niet het aandeel), en dus dezelfde schatting.

    ### Hoeveel privacy: ε

    Voor één persoon is een ja 3/4 waarschijnlijk bij een echte ja en 1/4 bij een echte nee: drie keer zo waarschijnlijk. Die verhouding is de maat. Een manier van antwoorden is ε-differentially private, in de lokale zin, als voor elke twee waarheden die iemand kan hebben elk antwoord hoogstens e<sup>ε</sup> keer zo waarschijnlijk is bij de ene als bij de andere. Volgens Dwork en Roth is de randomized response hierboven (ln 3, 0)-differentially private. Daarbij is ln 3 de log van 3, ongeveer 1,10, en de 0 zegt dat de grens zonder uitzonderingen geldt. Een kleinere ε betekent antwoorden die minder zeggen over iemand; bij ε = 0 hangt een antwoord helemaal niet van de waarheid af. En de privacy zit in de werkwijze, schrijven Dwork en Roth: er bestaan geen goede of slechte antwoorden.

    Dit is **local differential privacy**: iedereen voegt het toeval zelf toe, voordat iemand het antwoord verzamelt. Omdat niemand behalve de eigenaar ooit aan een stukje persoonsgegevens komt, noemen Dwork en Roth deze lokale opzet veel veiliger, en ze schrijven dat dit lokale model zijn wortels heeft in randomized response, zoals Warner het in 1965 voor het eerst voorstelde.

    ### De prijs

    Munten verbergen één persoon, en ze maken ook de groep vager: dezelfde precisie vraagt meer antwoorden, en in Oefening 2 zie je hoeveel meer. Úlfar Erlingsson, Vasyl Pihur en Aleksandra Korolova (2014), die RAPPOR ontwierpen, een vorm van randomized response die Chrome van Google heeft gebruikt, geven een voorbeeld bij ε = ln 3: om iets te vinden dat bij 1% van de mensen voorkomt, zijn een miljoen antwoorden nodig. Opnieuw vragen verbruikt privacy: de garantie wordt zwakker als de enquête herhaald wordt, dus maakt RAPPOR per persoon één keer een antwoord met ruis, bewaart dat, en voegt er bij elk bericht nieuwe ruis aan toe. RAPPOR verzamelde zijn gegevens bij een deel van de gebruikers van Chrome die ervoor gekozen hadden gebruiksstatistieken naar Google te sturen.

    Ook Apple gebruikt local differential privacy, met methoden die ingewikkelder zijn dan twee munten, maar met dezelfde maat. Volgens het *Differential Privacy Technical Overview* van Apple is dat voor emoji een privacybudget van ε = 4, met één inzending per dag, en voor QuickType ε = 8, met twee inzendingen per dag. Bij ε = 8 mag een antwoord tot e<sup>8</sup>, ongeveer 2981 keer, zo waarschijnlijk zijn bij de ene waarheid als bij de andere; bij ln 3 is dat 3 keer.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **randomized response** | iedereen voegt toeval toe aan het eigen antwoord, zoals met de twee munten, zodat geen enkel antwoord iemand verraadt, terwijl je het aandeel in de groep toch kunt schatten |
        | **local differential privacy** | privacy waarbij iedereen het toeval toevoegt voordat het antwoord de deur uit gaat, zodat zelfs wie verzamelt de echte antwoorden niet ziet; ε zegt hoeveel één antwoord mag zeggen |
        | **ε** (epsilon) | hoeveel één antwoord mag zeggen: elk antwoord is hoogstens e<sup>ε</sup> keer zo waarschijnlijk bij de ene waarheid als bij de andere; een kleinere ε is meer privacy |
        | **ln 3** | de log van 3 met grondtal e, `np.log(3)`, ongeveer 1,10: de ε van twee zuivere munten |
        | **2 × aandeel − 1/2** | de schatting van het echte aandeel ja, uit het aandeel ja in antwoorden met twee zuivere munten |
        """),
        L.watch_out("De twee munten zijn een latere variant van het idee van Warner: Warner gebruikte een draaischijf. ε = ln 3 geldt voor één antwoord: dezelfde vraag opnieuw gesteld, met nieuwe munten, zegt elke keer meer. En de schatting hoeft niet tussen 0 en 1 te liggen: Warner merkt op dat die erbuiten kan vallen, en 2 × aandeel − 1/2 is kleiner dan 0 zodra minder dan een kwart van de antwoorden ja is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een vraag te stellen die mensen liever niet beantwoorden, met twee munten:

    1. Iedereen gooit de eerste munt op. Bij munt antwoorden ze naar waarheid; bij kop gooien ze de tweede munt op en antwoorden ja bij kop en nee bij munt. Dat is `answer`.
    2. Tel de antwoorden op: het aandeel ja, keer 2, min 1/2, schat het echte aandeel. Dat is `estimate`.
    3. Meet de privacy: hoeveel keer zo waarschijnlijk een ja is bij een echte ja als bij een echte nee. De log van die verhouding is ε. Dat is `epsilon`.
    4. Reken de prijs: de schatting spreidt meer dan bij de directe vraag, dus heb je meer antwoorden nodig, en elk nieuw antwoord van dezelfde persoon verbruikt privacy.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel gooit de twee munten op voor 10.000 mensen, met 1 voor kop en 0 voor munt, en telt wat de munten in hun eentje beslissen.
    """)
    return


@app.cell
def example_coins():
    import numpy as np

    _rng = np.random.default_rng(1965)
    _first = _rng.integers(0, 2, size=10000)    # 1 = kop: de tweede munt beslist
    _second = _rng.integers(0, 2, size=10000)   # 1 = kop: antwoord ja
    print("heads on the first coin:", _first.mean())
    # twee keer kop: een ja, wat de waarheid ook is
    print("forced yes:", (_first * _second).mean())
    print("forced no:", (_first * (1 - _second)).mean())
    print("tells the truth:", (1 - _first).mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Ongeveer de helft van de mensen zegt de waarheid, ongeveer een kwart zegt ja en ongeveer een kwart zegt nee, wat de waarheid ook is: hier 50,15%, 24,15% en 25,7%. Arrays van nullen en enen met elkaar vermenigvuldigen pikt mensen eruit: `_first * _second` is alleen 1 waar beide munten op kop vallen, en `1 - _first` is alleen 1 waar de eerste munt op munt valt. Iemand bij wie de waarheid ja is, antwoordt dus ongeveer 1/2 + 1/4 = 3/4 van de keren ja, en iemand bij wie de waarheid nee is ongeveer 1/4 van de keren. In Oefening 1 zet je de munten en de waarheid samen om in antwoorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_answer():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. answer: wat iedereen zegt

        `TRUTH` bevat de echte antwoorden van de 10.000 mensen van een verzonnen enquête over een vraag die ze liever niet beantwoorden, zoals of ze ooit bij een toets hebben gespiekt: 1 voor ja, 0 voor nee; 2000 ervan zijn 1. `FIRST` en `SECOND` bevatten hun twee munten: 1 voor kop, 0 voor munt. `FEW_TRUTH`, `FEW_FIRST` en `FEW_SECOND` zijn acht mensen, één voor elke combinatie van waarheid en munten.

        `answer(truth, first, second)` krijgt drie NumPy-arrays van nullen en enen, even lang: de echte antwoorden `truth`, de eerste munten `first` en de tweede munten `second`. Hij geeft terug wat iedereen antwoordt, als array van nullen en enen: de waarheid waar de eerste munt op munt (0) valt, en de tweede munt waar de eerste op kop (1) valt. Vervang de `...`. De check roept `answer(FEW_TRUTH, FEW_FIRST, FEW_SECOND)` aan, daarna `TRUTH`, `FIRST` en `SECOND`, en andere munten.
        """),
        L.task("Geef de waarheid terug waar first 0 is, en second waar first 1 is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("answer", lines=3)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "answer",
        [
            "Stap 1 van het recept. In het uitgewerkte voorbeeld is `1 - _first` precies 1 voor de mensen die de waarheid zeggen.",
            "Vermenigvuldigen met een array van nullen en enen houdt de waarden waar die 1 is en maakt de rest 0, dus `(1 - first) * truth` is de waarheid voor wie die zegt, en 0 voor de anderen. Tel er het deel bij op voor de mensen bij wie de eerste munt op kop valt.",
            "`return (1 - first) * truth + ___`",
        ],
        why="`return (1 - first) * truth + first * second` geeft de waarheid waar `first` 0 is en de tweede munt waar `first` 1 is. In de enquête antwoorden 1499 van de 2000 mensen bij wie de waarheid ja is ja, en 1988 van de 8000 bij wie de waarheid nee is: ongeveer 3 op 4 en 1 op 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_estimate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. estimate: het echte aandeel uit de antwoorden

        `ANSWERS` bevat de antwoorden van de 10.000 mensen van de enquête, gemaakt met de `answer` van de course uit `TRUTH`, `FIRST` en `SECOND`.

        `estimate(answers)` krijgt een NumPy-array `answers` van nullen en enen: de antwoorden van een enquête met twee zuivere munten. Hij geeft de schatting terug van het echte aandeel ja, zoals Dwork en Roth die maken. Vervang de `...`. De check roept `estimate(ANSWERS)` aan, daarna andere antwoorden.
        """),
        L.task("Geef de schatting van het echte aandeel ja terug, uit antwoorden met twee zuivere munten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_estimate():
    box_estimate = L.code_box("estimate", lines=3)
    box_estimate
    return (box_estimate,)


@app.cell(hide_code=True)
def check_estimate(box_estimate):
    result_estimate = L.check("estimate", box_estimate.value)
    result_estimate
    return (result_estimate,)


@app.cell(hide_code=True)
def hints_estimate():
    L.hints(
        "estimate",
        [
            "Stap 2 van het recept, en de schatting van Dwork en Roth in *Twee munten*.",
            "Het aandeel ja in nullen en enen is hun gemiddelde. Gemiddeld is dat aandeel 1/4 + p/2, met p het echte aandeel: los dat op naar p.",
            "`return 2 * ___ - 0.5`",
        ],
        why="`return 2 * answers.mean() - 0.5` rekent het aandeel ja terug naar het echte aandeel. `estimate(ANSWERS)` geeft 0,1974: 34,87% van de antwoorden is ja, en het echte aandeel is 0,2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_epsilon():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. epsilon: hoeveel privacy

        `epsilon(t)` krijgt een getal `t` van 0 tot onder 1: de kans dat de eerste munt zegt "zeg de waarheid". Met een zuivere munt is `t` 0,5. Zegt de eerste munt dat niet, dan beslist de tweede munt, nog steeds zuiver, tussen ja en nee: de helft van die keren ja. Hij geeft ε terug: de log van hoeveel keer zo waarschijnlijk een ja is bij iemand bij wie de waarheid ja is, als bij iemand bij wie de waarheid nee is. `yes_if_yes` is de kans op ja voor iemand bij wie de waarheid ja is, en `yes_if_no` hetzelfde voor iemand bij wie de waarheid nee is. Vervang de drie `...`. De check roept `epsilon(0.5)` aan, daarna andere waarden van `t`.
        """),
        L.task("Geef de log terug van hoeveel keer zo waarschijnlijk een ja is bij een echte ja als bij een echte nee."),
    ])
    return


@app.cell(hide_code=True)
def exercise_epsilon():
    box_epsilon = L.code_box("epsilon", lines=8)
    box_epsilon
    return (box_epsilon,)


@app.cell(hide_code=True)
def check_epsilon(box_epsilon):
    result_epsilon = L.check("epsilon", box_epsilon.value)
    result_epsilon
    return (result_epsilon,)


@app.cell(hide_code=True)
def hints_epsilon():
    L.hints(
        "epsilon",
        [
            "Stap 3 van het recept. Met zuivere munten is `t` 0,5: een ja komt 3/4 van de keren bij een echte ja en 1/4 bij een echte nee, en ε is ln 3.",
            "Iemand bij wie de waarheid ja is, zegt ja als de eerste munt zegt dat de waarheid moet, met kans `t`, of als die dat niet zegt en de tweede munt ja zegt, met kans `(1 - t) / 2`. Iemand bij wie de waarheid nee is, zegt alleen in het tweede geval ja. `np.log` geeft de log, zoals in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "`yes_if_yes = t + (1 - t) / 2`, `yes_if_no = (1 - t) / 2` en `return np.log(___)`",
        ],
        why="`yes_if_yes = t + (1 - t) / 2` en `yes_if_no = (1 - t) / 2` zijn de twee kansen op ja, en `return np.log(yes_if_yes / yes_if_no)` is ε. `epsilon(0.5)` geeft 1,10, ln 3; `epsilon(0.9)` geeft 2,94: een ja is dan 19 keer zo waarschijnlijk bij een echte ja.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_a_yes():
    mo.md(r"""
    ### Oefening 4. Eén ja

    In de enquête met twee zuivere munten antwoordde iemand ja.
    """)
    return


@app.cell(hide_code=True)
def choice_a_yes_cell():
    choice_a_yes = L.choice(
        "a-yes",
        question="Wat laat dat ene antwoord zien over die persoon?",
        options={
            "sure": "Dat het waar is: een ja is nooit gedwongen",
            "right": "Weinig: bij een echte ja 3/4 kans, anders 1/4",
            "nothing": "Niets: de munten beslissen elk antwoord in hun eentje",
        },
        correct="right",
        why={
            "right": "Een ja komt met kans 3/4 bij iemand bij wie de waarheid ja is, en met kans 1/4 bij iemand bij wie de waarheid nee is: drie keer zo waarschijnlijk, ε = ln 3. Dat verschuift de kansen, maar nooit tot zekerheid, en daarom noemen Dwork en Roth een ja niet belastend.",
            "sure": "Twee keer kop dwingt een ja af, wat de waarheid ook is: dat gebeurt een kwart van de keren. Zie *Twee munten* in *Het idee*.",
            "nothing": "De helft van de keren zegt de eerste munt dat de waarheid moet, dus een ja is drie keer zo waarschijnlijk bij een echte ja als bij een echte nee. Zie *Hoeveel privacy: ε* in *Het idee*.",
        },
    )
    choice_a_yes
    return (choice_a_yes,)


@app.cell(hide_code=True)
def text_exercise_every_day():
    mo.md(r"""
    ### Oefening 5. Elke dag opnieuw

    Een bedrijf stelt dezelfde mensen elke dag dezelfde vraag, elke keer met twee nieuwe munten.
    """)
    return


@app.cell(hide_code=True)
def choice_every_day_cell():
    choice_every_day = L.choice(
        "every-day",
        question="Wat gebeurt er met hun privacy?",
        options={
            "same": "Niets: elk antwoord heeft nog steeds ε = ln 3",
            "right": "Die slijt weg als de antwoorden zich opstapelen",
            "grows": "Die groeit, want elke dag komt er nieuwe ruis bij",
        },
        correct="right",
        why={
            "right": "Elk antwoord is ja in 3/4 van de keren bij een echte ja en in 1/4 bij een echte nee, dus over veel dagen komt het aandeel ja van één persoon dicht bij 3/4 of 1/4, en dat verraadt de waarheid. Erlingsson, Pihur en Korolova schrijven dat de garantie zwakker wordt als de enquête herhaald wordt; daarom bewaart RAPPOR één antwoord met ruis per persoon, en voegt er bij elk bericht nieuwe ruis aan toe.",
            "same": "Elk antwoord heeft ε = ln 3, maar ze stapelen zich op: over veel dagen komt het aandeel ja van één persoon dicht bij 3/4 of 1/4, en dat verraadt de waarheid. Zie *De prijs* in *Het idee*.",
            "grows": "Nieuwe munten elke dag geven nieuwe antwoorden, en hun aandeel over veel dagen komt dicht bij 3/4 of 1/4: de waarheid schemert erdoorheen. Zie *De prijs* in *Het idee*.",
        },
    )
    choice_every_day
    return (choice_every_day,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tabellen koppelen](/course/open/rai-linkage-attacks?lang=nl) werden de gegevens eerst verzameld, en kwamen de namen terug door ze te koppelen aan een kiezerslijst. Hier beschermt iedereen het eigen antwoord al voordat die het geeft, en ε zegt hoeveel: geen tabel om mee te koppelen kan teruggeven wat de munten verborgen, en toch kun je het aandeel in de groep schatten, zoals in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl), tegen de prijs van meer antwoorden. In de volgende les, *Trainen zonder de data te verzamelen*, blijven de gegevens op elk apparaat, en reist alleen wat een model ervan leert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_a_yes, choice_every_day, result_answer, result_epsilon, result_estimate):
    L.wrap_up(result_answer, result_estimate, result_epsilon, choice_a_yes, choice_every_day)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
