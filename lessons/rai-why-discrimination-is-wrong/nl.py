# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-why-discrimination-is-wrong. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-why-discrimination-is-wrong", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) en [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_milder():
    quick_milder = L.choice(
        "quick-milder",
        question="Een neutrale regel raakt een groep in het bijzonder. Het doel is legitiem en de regel is passend, maar een milder middel zou ook werken. Wat is het?",
        options={
            "direct": "Directe discriminatie",
            "right": "Verboden indirecte discriminatie",
            "fine": "Toegestaan, want twee van de drie antwoorden zijn ja",
        },
        correct="right",
        why={
            "direct": "De regel noemt geen grond, dus hij is niet direct. Zie *Het idee* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
            "right": "Met een milder middel binnen bereik is de regel niet noodzakelijk, en één nee is genoeg.",
            "fine": "Alle drie de antwoorden moeten ja zijn. Zie *Wanneer indirect mag* in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl).",
        },
    )
    quick_milder
    return (quick_milder,)


@app.cell(hide_code=True)
def quick_check_theory():
    quick_theory = L.choice(
        "quick-theory",
        question="Welke theorie beoordeelt een keuze op wat ze teweegbrengt, opgeteld over iedereen?",
        options={
            "right": "Consequentialisme, dat uitkomsten optelt",
            "duty": "Deontologie",
            "virtue": "Deugdethiek, die naar een goed karakter vraagt",
        },
        correct="right",
        why={
            "right": "Het consequentialisme telt de uitkomsten op voor iedereen die erdoor geraakt wordt.",
            "duty": "Deontologie houdt vast aan plichten en rechten die tellen wat de uitkomst ook is. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "virtue": "Deugdethiek vraagt wat iemand met een goed karakter zou doen. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_theory
    return (quick_theory,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De wet zegt wanneer discriminatie verboden is, maar niet waarom het fout is. Die vraag doet ertoe bij modellen. Een model dat elke persoon beoordeelt naar het gemiddelde van de eigen groep, kan gemiddeld gelijk hebben en niemand iets willen aandoen, en toch zijn wat veel filosofen discriminatie noemen die moreel fout is. Om te zeggen wat er mis mee is, en wat een oplossing moet herstellen, heb je de redenen nodig.

    **Een voorbeeld uit het dagelijks leven.** Een winkeleigenaar loopt elke tiener achterna door de winkel, omdat tieners vaker stelen, zegt de eigenaar. Ook als dat gemiddeld zo zou zijn, wordt de tiener die nooit iets stal beoordeeld naar een groep, niet naar iets wat die tiener zelf deed.
    """)
    return


@app.cell(hide_code=True)
def predict_taste():
    guess_taste = L.guess(
        "taste",
        question="Economen noemen het beoordelen van iemand naar het gemiddelde van de eigen groep statistische discriminatie. Gaat hun basismodel ervan uit dat de werkgever een hekel heeft aan de groep?",
        options={
            "yes": "Ja, die afkeer maakt het juist discriminatie",
            "little": "Een beetje: wat afkeer, en wat gokken",
            "right": "Nee, de werkgever gokt zo goed als het gaat",
        },
        reveal="**Nee.** In de modellen van Edmund Phelps (1972) en Kenneth Arrow (1973) heeft de werkgever aan geen enkele groep een hekel, en in het basisgeval van Phelps kloppen de gemiddelden die de werkgever gebruikt zelfs. Daarom noemen economen het rationeel. Deze les vraagt of het toch fout kan zijn.",
    )
    guess_taste
    return (guess_taste,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Filosofen geven verschillende antwoorden op de vraag wat discriminatie fout maakt. Drie worden vaak genoemd.

        **Het schaadt.** Kasper Lippert-Rasmussen (*Born Free and Equal?*, 2014) betoogt dat discriminatie fout is, als ze fout is, door de schade die ze doet, en dat schade aan mensen die al slechter af zijn zwaarder weegt. Dan heeft discriminatie geen eigen kwaad: het is een manier om schade te doen. Een tegenwerping: een bedrijf ontslaat iemand om een seksistische reden; die persoon vindt een betere baan en hoort nooit waarom. Niemand lijkt schade te hebben, toch vinden velen het ontslag fout.

        **Het vernedert.** Deborah Hellman (*When Is Discrimination Wrong?*, 2008) betoogt dat mensen verschillend behandelen naar een eigenschap fout is als het vernedert: als het uitdrukt dat een persoon of groep een lagere status heeft, gedaan door iemand met genoeg macht om dat te laten aankomen. Het voorbeeld van Hellman: een werkgever die eist dat vrouwelijk personeel make-up draagt, zegt dat het lichaam van een vrouw er is om door anderen bekeken te worden. Het gaat om wat de daad betekent in de samenleving, niet om wat degene die het doet ervan denkt. Een tegenwerping: ook een klasgenoot zonder enige macht over je kan je vernederen.

        **Het respecteert de persoon niet.** Benjamin Eidelson (*Discrimination and Disrespect*, 2015) betoogt dat discriminatie fout is als ze mensen niet het respect geeft dat personen toekomt: ze zijn even veel waard, en ze geven hun eigen leven vorm. Iemand als individu behandelen verbiedt niet elke generalisatie, die volgens Eidelson (2013) "alledaags en onvermijdelijk" is; het vraagt dat je kijkt naar wat iemand van het eigen leven heeft gemaakt. Een brandweer die elke vrouw afwijst, in plaats van elke sollicitant een conditietest te geven, vraagt nooit of de sollicitant de kracht heeft opgebouwd.

        **Groepen klein houden.** Een ander antwoord kijkt naar groepen. Verschillen tussen groepen kunnen voortkomen uit onrecht in het verleden, en een beslissing die erop leunt, maakt dat onrecht groter. Barocas, Hardt en Narayanan noemen in hun boek over fairness zes antwoorden, waaronder dit, dat ze *compounding injustice* noemen, en schrijven dat geen ervan een volledige theorie is.

        **Statistische discriminatie.** De economen Phelps en Arrow beschreven werkgevers die het echte kunnen van iemand niet zien en het gemiddelde van de groep van die persoon gebruiken om het gat te vullen: geen afkeer, alleen gokken. In het model van Phelps, zoals Aigner en Cain het in 1977 schreven, wordt de schatting van een persoon naar het gemiddelde van de groep getrokken, en sterker naarmate de test minder te vertrouwen is: de schatting is (1 − trust) × groepsgemiddelde + trust × score. Twee mensen met dezelfde score krijgen dan verschillende schattingen, alleen door hun groep. Economen voegen toe dat in het basismodel elke werknemer betaald krijgt wat de informatie voorspelt, en noemen het daarom geen discriminatie in economische zin. Veel filosofen vinden het toch fout, ook als het gemiddelde klopt: een groep is een onnodig grove gok als je naar de persoon zelf kunt kijken; het verschil tussen de groepen kan uit vroeger onrecht komen; en het kijkt niet naar wat de persoon van het eigen leven maakte. Een model dat leert van gegevens over groepen kan hetzelfde doen, zonder dat iemand ertoe besluit.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **statistische discriminatie** | wat gemiddeld geldt voor de groep van iemand gebruiken in plaats van wat je niet van die persoon zelf kunt zien; in het model van Phelps mengt de schatting het groepsgemiddelde met de eigen score van de persoon |
        | **trust** | in de formule: hoeveel de test meetelt, van 0 (helemaal niet) tot 1 (volledig); Aigner en Cain schrijven het als γ |
        """),
        L.watch_out("Statistische discriminatie is niet hetzelfde als je vergissen in een groep. De gemiddelden kunnen kloppen, en niemand hoeft een hekel aan iemand te hebben; de vraag is of het juist is om een persoon daarnaar te beoordelen."),
        mo.Html(C.picture({"title": "Dezelfde score, verschillende schattingen",
                           "group": "groep", "mean": "groepsgemiddelde", "score": "score",
                           "xlabel": "schatting van de werkgever", "decimal": ",",
                           "alt": "Twee rijen op een schaal van 40 tot 80. Beide mensen scoren 70. Groep A heeft gemiddelde 60, en de schatting is 65,0; groep B heeft gemiddelde 50, en de schatting is 60,0. Elke schatting ligt halverwege het groepsgemiddelde en de score, omdat trust 0,5 is."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zoek je uit wat er mis is met een regel of een model dat groepen verschillend behandelt:

    1. Vraag wie wat verliest, en of wie verliest al slechter af was: schade.
    2. Vraag wat de regel in zijn samenleving over een groep zegt, en of wie hem gebruikt macht heeft: vernedering.
    3. Vraag of de regel kijkt naar wat elke persoon deed en van het eigen leven maakte, of alleen naar de groep: respect voor de persoon.
    4. Vraag waar het verschil tussen de groepen vandaan komt, en of erop leunen een oud onrecht groter maakt.
    5. Schrijf op welke antwoorden van toepassing zijn: ze zeggen wat een oplossing moet herstellen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het model van Phelps, met getallen die de course verzon. Twee mensen scoren allebei 70 op een test; `_means` bevat de gemiddelde score van hun groepen. `_trust` is 0,5: de test telt voor de helft. De loop rekent met `_group` de schatting uit voor een persoon van elke groep. Voer de cel uit.
    """)
    return


@app.cell
def example_estimates():
    # verzonnen getallen: één testscore, het gemiddelde van elke groep, en hoeveel de werkgever de test vertrouwt
    _score = 70
    _means = {"A": 60, "B": 50}
    _trust = 0.5

    _estimates = {}
    for _group in _means:
        _estimates[_group] = (1 - _trust) * _means[_group] + _trust * _score
    print(_estimates)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Dezelfde score, maar 65,0 tegen 60,0: 5 punten verschil, alleen door de groep. Python schrijft getallen met een punt. Nu een test die de werkgever minder vertrouwt: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe ver uit elkaar?

        Dezelfde mensen, maar de werkgever vertrouwt de test minder: `trust` is 0,2. Vervang de `...` door de loop van het uitgewerkte voorbeeld, met de namen zonder liggend streepje: `estimates` en `group`. Reken daarna `gap` uit, de schatting van `"A"` min die van `"B"`. De laatste regel is `gap`, zodat de check het ziet. De check probeert ook andere getallen.
        """),
        L.task("Reken `gap` uit: hoeveel punten twee mensen met dezelfde score uit elkaar komen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gap():
    box_gap = L.code_box("gap", lines=11)
    box_gap
    return (box_gap,)


@app.cell(hide_code=True)
def check_gap(box_gap):
    result_gap = L.check("gap", box_gap.value)
    result_gap
    return (result_gap,)


@app.cell(hide_code=True)
def hints_gap():
    L.hints(
        "gap",
        [
            "Neem de loop van het uitgewerkte voorbeeld over, zonder de `print`, en haal het liggend streepje van elke naam. Daarna komt nog één regel.",
            "In de loop is elke schatting `(1 - trust)` keer het groepsgemiddelde plus `trust` keer de score. Trek na de loop de twee schattingen van elkaar af.",
            "Na de loop: `gap = estimates[\"A\"] - ___`.",
        ],
        why="`estimates = {}`, dan `for group in means:` met `estimates[group] = (1 - trust) * means[group] + trust * score` eronder, en na de loop `gap = estimates[\"A\"] - estimates[\"B\"]`. Dat geeft `8.0`: 62 tegen 54.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_accounts():
    mo.md(r"""
    ### Oefening 2. Van wie is het antwoord?

    Zes beweringen. Deel elke bewering in bij het antwoord waar ze bij hoort.
    """)
    return


@app.cell(hide_code=True)
def sort_accounts_cell():
    sort_accounts = L.sort("accounts")
    sort_accounts
    return (sort_accounts,)


@app.cell(hide_code=True)
def check_accounts(sort_accounts):
    result_accounts = L.check("accounts", sort_accounts)
    result_accounts
    return (result_accounts,)


@app.cell(hide_code=True)
def hints_accounts():
    L.hints(
        "accounts",
        [
            "Vraag bij elke bewering: gaat ze over wat iemand verliest, over wat een daad over een groep zegt, of over kijken naar de persoon?",
            "De make-upregel is het eigen voorbeeld van Hellman, en de brandweer komt van Eidelson.",
            "Schade: de kosten en \\_\\_\\_. Vernedering: de make-up en \\_\\_\\_.",
        ],
        why="**Het schaadt:** de kosten en wie slechter af is. **Het vernedert:** de make-up en de macht. **Het respecteert de persoon niet:** de brandweer en de schrijver van het eigen leven.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.md(r"""
    ### Oefening 3. Een gemiddelde dat klopt
    """)
    return


@app.cell(hide_code=True)
def choice_average_cell():
    choice_average = L.choice(
        "average",
        question="Een werkgever weet, terecht, dat mensen uit groep B gemiddeld lager scoren, en geeft iedereen uit groep B een lagere schatting, wat de eigen score ook is. Wat zeggen veel filosofen uit deze les?",
        options={
            "fine": "Er is niets mis",
            "taste": "Het is alleen fout als de werkgever een hekel heeft aan groep B",
            "right": "Het kan fout zijn, ook als het gemiddelde klopt",
        },
        correct="right",
        why={
            "fine": "Een gemiddelde dat klopt, kan toch een onnodig grove gok zijn, uit vroeger onrecht komen en negeren wat iemand van het eigen leven maakte. Zie *Statistische discriminatie* in *Het idee*.",
            "taste": "Afkeer is niet nodig: statistische discriminatie gaat over gokken, en de antwoorden over respect en schade gelden nog steeds. Zie *Het idee*.",
            "right": "De persoon wordt beoordeeld naar de groep, niet naar de eigen score, en het verschil tussen de groepen kan uit vroeger onrecht komen.",
        },
    )
    choice_average
    return (choice_average,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) leerde je het consequentialisme en de deontologie kennen. Het antwoord over schade beoordeelt discriminatie op de uitkomsten, met meer gewicht voor wie slechter af is; de antwoorden over vernedering en respect kijken naar wat een daad zegt en hoe die een persoon behandelt, dichter bij de deontologie. In [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl) keek de wet naar effecten; hier zag je waarom die effecten fout kunnen zijn. De volgende les, *Wat is een gelijke kans?*, vraagt wat het zou betekenen om iedereen dezelfde kans te geven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_average, result_accounts, result_gap):
    L.wrap_up(result_gap, result_accounts, choice_average)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
