# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les prob-conditional. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-conditional", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_spam():
    quick_spam = L.choice(
        "quick-spam",
        question="8 van de 20 mails zijn spam. Wat is P(spam)?",
        options={
            "right": "0,4",
            "count": "8",
            "inverse": "2,5",
        },
        correct="right",
        why={
            "count": "Zoveel zijn het er. Een kans is een deel: waarvan? Zie de formule in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "inverse": "Dat is 20 / 8, andersom. Een kans ligt tussen 0 en 1. Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "right": "8 van de 20 mails: 8 / 20 = 0,4.",
        },
    )
    quick_spam
    return (quick_spam,)


@app.cell(hide_code=True)
def quick_check_rest():
    quick_rest = L.choice(
        "quick-rest",
        question="Elke dag regent het of blijft het droog, en P(regen) = 0,3. Wat is P(droog)?",
        options={
            "same": "0,3",
            "right": "0,7",
            "one": "1",
        },
        correct="right",
        why={
            "same": "De twee uitkomsten hoeven niet even waarschijnlijk te zijn. Waar komen alle kansen van een kansverdeling samen op uit? Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "one": "1 is de kans op alle uitkomsten samen. Wat blijft er over voor droog? Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "right": "De twee kansen zijn samen 1: 1 − 0,3 = 0,7.",
        },
    )
    quick_rest
    return (quick_rest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het maakt een spamfilter niet uit hoe waarschijnlijk spam in het algemeen is. Hij wil weten hoe waarschijnlijk het is dat deze mail spam is, nu hij het woord "free" erin heeft gezien. Iets weten verandert de kans, en tellen laat zien hoeveel.

    **Een voorbeeld uit het dagelijks leven.** Het regent op 40% van de dagen. Maar op een ochtend met donkere wolken kijk je alleen naar ochtenden zoals deze, en op een veel groter deel daarvan regent het.
    """)
    return


@app.cell(hide_code=True)
def predict_free():
    guess_free = L.guess(
        "free",
        question="Van 100 mails zijn er 20 spam. Het woord \"free\" staat in 15 van de spammails, en in 10 van de andere. Een nieuwe mail bevat \"free\". Hoe groot denk je dat de kans is dat hij spam is?",
        options={
            "overall": "0,2",
            "swapped": "0,75",
            "given": "0,6",
        },
        reveal="**0,6.** 25 mails bevatten \"free\", en 15 daarvan zijn spam: 15 / 25 = 0,6. 0,2 is P(spam) over alle 100 mails, en 0,75 is andersom: het deel van de spammails dat \"free\" bevat.",
    )
    guess_free
    return (guess_free,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een voorwaardelijke kans telt alleen onder de gevallen waarin iets anders gebeurde. Houd alleen de gevallen over waarin B gebeurde, en neem daaronder het deel waarin ook A gebeurde.
        """),
        L.formula(r"P(A \mid B) = \frac{\text{hoe vaak } A \text{ en } B \text{ allebei gebeurden}}{\text{hoe vaak } B \text{ gebeurde}}"),
        mo.md(r"""
        De verticale streep lees je als "gegeven": alles erachter weet je al.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **voorwaardelijke kans** | | de kans op $A$ als je alleen kijkt naar de gevallen waarin $B$ gebeurde | `both / with_b` |
        | $P(A \mid B)$ | de kans op A gegeven B | de voorwaardelijke kans op $A$, gegeven $B$ | |
        """),
        L.watch_out("P(A | B) is niet P(B | A). Bij de mails is P(spam | free) = 15 / 25 = 0,6, maar P(free | spam) = 15 / 20 = 0,75: de voorwaarde bepaalt onder welke gevallen je telt."),
        mo.Html(C.picture({"title": "100 mails", "spam": "spam", "normal": "geen spam", "total": "totaal", "free": "met \"free\"", "plain": "zonder", "caption": "P(spam | free) = 15 / 25 = 0,6",
                           "alt": "Een tabel van 100 mails. Spam: 15 met het woord free, 5 zonder, 20 in totaal. Geen spam: 10 met free, 70 zonder, 80 in totaal. Samen: 25 met free, 75 zonder, 100. De kolom met free is gemarkeerd: P(spam gegeven free) is 15 van de 25, 0,6."})),
        mo.md(r"""
        De voorwaarde "free" houdt één kolom over: 25 mails. Daarvan zijn er 15 spam. De andere 75 mails doen niet mee.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vind je P(A | B) door te tellen:

    1. Houd alleen de gevallen over waarin B gebeurde.
    2. Tel ze: dat is het getal waardoor je deelt.
    3. Tel daaronder de gevallen waarin A ook gebeurde.
    4. Deel de telling van stap 3 door de telling van stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Tien dagen, elk een paar: de lucht in de ochtend en het weer later die dag. `for _sky, _weather in _days:` geeft de twee waarden van elk paar een eigen naam, zoals `x, y = point` deed in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl). Eerst P(regen) over alle dagen, dan P(regen | wolken). Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_rain():
    _days = [("clouds", "rain"), ("clouds", "rain"), ("clouds", "dry"), ("sun", "dry"), ("clouds", "rain"),
             ("sun", "dry"), ("sun", "dry"), ("clouds", "dry"), ("sun", "rain"), ("sun", "dry")]
    # P(regen), over alle dagen
    _rain = 0
    for _sky, _weather in _days:
        if _weather == "rain":
            _rain = _rain + 1
    print(_rain)
    print(_rain / len(_days))
    # 1. en 2. houd de dagen met wolken over, en tel ze
    _clouds = 0
    _both = 0
    for _sky, _weather in _days:
        if _sky == "clouds":
            _clouds = _clouds + 1
            # 3. daaronder de dagen met regen
            if _weather == "rain":
                _both = _both + 1
    print(_clouds)
    print(_both)
    # 4. deel
    print(_both / _clouds)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op 4 van de 10 dagen regende het: P(regen) = 0,4. Van de 5 ochtenden met wolken brachten er 3 regen: P(regen | wolken) = 0,6. Weten dat er wolken zijn, maakt de kans op regen groter: van 0,4 naar 0,6.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_studied():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hielp het leren?
    """)
    return


@app.cell(hide_code=True)
def choice_studied_cell():
    choice_studied = L.choice(
        "studied",
        question="50 leerlingen: 30 hebben geleerd, en 24 daarvan slaagden. Van de 20 die niet leerden, slaagden er 8. Wat is P(geslaagd | geleerd)?",
        options={
            "joint": "0,48",
            "overall": "0,64",
            "right": "0,8",
        },
        correct="right",
        why={
            "joint": "Dat is 24 / 50: het deel van alle leerlingen dat leerde en slaagde. Welke leerlingen houdt de voorwaarde over? Zie het recept.",
            "overall": "Dat is 32 / 50, P(geslaagd) over alle leerlingen. De voorwaarde is dat ze geleerd hebben: zie stap 1 van het recept.",
            "right": "Houd de 30 over die leerden; daarvan slaagden er 24: 24 / 30 = 0,8. Van de 20 die niet leerden, slaagden er 8: 8 / 20 = 0,4. Wie leerde, slaagde twee keer zo vaak.",
        },
    )
    choice_studied
    return (choice_studied,)


@app.cell(hide_code=True)
def text_exercise_given():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. P(a | b) als functie

        Maak `p_given(cases, a, b)` af. `cases` is een lijst paren: een uitkomst en een voorwaarde, zoals `("spam", "free")`. De functie geeft P(uitkomst is `a` | voorwaarde is `b`) terug. Vervang de `...`. De check roept `p_given(mails, "spam", "free")` aan met zeven mails, en dat moet `0.75` geven. Hij probeert ook andere gevallen.
        """),
        L.task("Laat `p_given` de voorwaardelijke kans teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_given():
    box_given = L.code_box("p-given", lines=10)
    box_given
    return (box_given,)


@app.cell(hide_code=True)
def check_given(box_given):
    result_given = L.check("p-given", box_given.value)
    result_given
    return (result_given,)


@app.cell(hide_code=True)
def hints_given():
    L.hints(
        "p-given",
        [
            "Het recept, met twee tellingen in één loop, zoals in het uitgewerkte voorbeeld.",
            "Loop de paren langs met `for outcome, condition in cases:`. Welke paren tellen mee voor het getal waardoor je deelt, en welke voor het getal erboven?",
            "`if condition == b:` telt 1 op bij `with_b`, en daarbinnen telt `if outcome == ___:` 1 op bij `both`; aan het eind `return both / ___`.",
        ],
        why="De loop telt de gevallen met voorwaarde `b`, en daaronder die met uitkomst `a`. Bij de zeven mails: 4 bevatten \"free\", 3 daarvan zijn spam, dus 3 / 4 = 0,75.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een test, beide kanten op

        Een test op een ziekte, gedaan bij 1000 mensen. `counts` bevat hoeveel mensen in elke groep vallen: ziek of gezond, en een positieve of negatieve test. Elke key is een paar, zoals `("ill", "positive")`, en `counts[("ill", "positive")]` geeft de telling van die groep. Zet P(ziek | positief) in `p_ill_given_pos`, en P(positief | ziek) in `p_pos_given_ill`. Vervang beide `...`. De laatste regel laat allebei zien, zodat de check ze ziet. De check probeert ook andere tellingen.
        """),
        L.task("Reken de voorwaardelijke kans beide kanten op uit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_test():
    box_test = L.code_box("test-both-ways", lines=6)
    box_test
    return (box_test,)


@app.cell(hide_code=True)
def check_test(box_test):
    result_test = L.check("test-both-ways", box_test.value)
    result_test
    return (result_test,)


@app.cell(hide_code=True)
def hints_test():
    L.hints(
        "test-both-ways",
        [
            "Voor elk van de twee eerst stap 1 van het recept: welke twee groepen houdt zijn voorwaarde over?",
            "P(ziek | positief) houdt de positieve tests over, van zieke en van gezonde mensen. P(positief | ziek) houdt de zieke mensen over, met een positieve of een negatieve test. Wat komt er bij allebei boven de streep?",
            "`counts[(\"ill\", \"positive\")] / (counts[(\"ill\", \"positive\")] + counts[(___, ___)])`, en dezelfde vorm voor de andere.",
        ],
        why="Allebei zetten de 9 mensen die ziek zijn en positief testen boven de streep. P(ziek | positief) deelt door alle 108 positieve tests: ongeveer 0,083. P(positief | ziek) deelt door de 10 zieke mensen: 0,9.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) telde je onder alle pogingen; een voorwaardelijke kans telt alleen onder de gevallen waarin de voorwaarde geldt, en het maakt uit welke kant op. Later in de course geeft een model dat spam van andere mail onderscheidt precies dit: de kans op spam, gegeven de woorden in de mail. In de volgende les, *Verwachtingswaarde en variantie*, reken je uit wat een uitkomst die van het toeval afhangt gemiddeld oplevert, en hoe ver de uitkomsten rond dat gemiddelde schommelen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_studied, result_given, result_test):
    L.wrap_up(choice_studied, result_given, result_test)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
