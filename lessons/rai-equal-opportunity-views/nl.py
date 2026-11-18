# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-equal-opportunity-views. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-equal-opportunity-views", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom is discriminatie fout?](/course/open/rai-why-discrimination-is-wrong?lang=nl)
    """)
    return


@app.cell(hide_code=True)
def quick_check_respect():
    quick_respect = L.choice(
        "quick-respect",
        question="Welk antwoord zegt dat discriminatie fout is als ze niet kijkt naar wat iemand van het eigen leven heeft gemaakt?",
        options={
            "right": "Dat van Eidelson, over respect voor personen",
            "harm": "Dat van Lippert-Rasmussen, over de schade die ze doet",
            "demean": "Dat van Hellman, over vernedering",
        },
        correct="right",
        why={
            "right": "Eidelson vraagt om mensen te respecteren als gelijken die hun eigen leven vormgeven.",
            "harm": "Lippert-Rasmussen kijkt naar de schade, zeker aan wie slechter af is. Zie *Het idee* in [Waarom is discriminatie fout?](/course/open/rai-why-discrimination-is-wrong?lang=nl)",
            "demean": "Hellman kijkt naar wat een daad over een groep uitdrukt. Zie *Het idee* in [Waarom is discriminatie fout?](/course/open/rai-why-discrimination-is-wrong?lang=nl)",
        },
    )
    quick_respect
    return (quick_respect,)


@app.cell(hide_code=True)
def quick_check_trust():
    quick_trust = L.choice(
        "quick-trust",
        question="In het model van Phelps zakt het vertrouwen van de werkgever in de test van 0,5 naar 0,2. Wat gebeurt er met het verschil tussen twee mensen met dezelfde score?",
        options={
            "right": "Het wordt groter, want de groep telt meer",
            "shrinks": "Het wordt kleiner",
            "same": "Het blijft gelijk, want de score is gelijk",
        },
        correct="right",
        why={
            "right": "Hoe minder de test telt, hoe meer het groepsgemiddelde telt: 5 punten werden er 8.",
            "shrinks": "Als de test minder telt, telt het groepsgemiddelde meer, en dat trekt de twee verder uit elkaar. Zie Oefening 1 in [Waarom is discriminatie fout?](/course/open/rai-why-discrimination-is-wrong?lang=nl)",
            "same": "De scores zijn gelijk, maar de groepsgemiddelden verschillen, en die tellen zwaarder als de test minder telt. Zie Oefening 1 in [Waarom is discriminatie fout?](/course/open/rai-why-discrimination-is-wrong?lang=nl)",
        },
    )
    quick_trust
    return (quick_trust,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Van een model dat sollicitanten kiest, een toelating tot een school of een besluit over een lening zeggen we dat het fair is als het iedereen een gelijke kans geeft. Maar wat is een gelijke kans? Dezelfde test voor iedereen, of een test die rekening houdt met waar mensen begonnen? Het antwoord bepaalt welke fairnesstest je op een model loslaat, en de tests uit latere lessen zijn het niet met elkaar eens.

    **Een voorbeeld uit het dagelijks leven.** Twee kinderen maken dezelfde toelatingstoets. Het ene had een jaar lang bijles; het andere paste elke middag op jongere broertjes en zusjes. De toets is voor allebei gelijk. De kans ook?
    """)
    return


@app.cell(hide_code=True)
def predict_warriors():
    guess_warriors = L.guess(
        "warriors",
        question="Een samenleving stelt haar klasse van krijgers, ooit alleen voor de rijken, open voor iedereen, en kiest krijgers met een krachttest. De armen zijn zo slecht gevoed dat ze de kracht missen om te winnen. Is de kans nu gelijk?",
        options={
            "yes": "Ja, want de test is voor iedereen gelijk",
            "right": "Alleen op papier, de armen staan achter",
            "no": "Nee, een krachttest is nooit fair",
        },
        reveal="**Alleen op papier.** Dit is het voorbeeld van de filosoof Bernard Williams (1962). De wedstrijd is open, en wordt beoordeeld op wat telt voor de taak, maar de armen beginnen te ver achter om hem te winnen. Het gat tussen een open wedstrijd en een echte kans is waar deze les over gaat.",
    )
    guess_warriors
    return (guess_warriors,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Bijna iedereen vindt dat mensen een gelijke kans moeten krijgen. Over wat dat betekent, verschillen ze van mening. Barocas, Hardt en Narayanan delen de antwoorden in hun boek over fairness in drie opvattingen in.

        **De smalle opvatting.** Behandel mensen die nu gelijk zijn op dezelfde manier. Beoordeel iedereen alleen op wat telt voor de taak, zoals een testscore, en geef de plek aan wie het best scoort. Dit is het oude ideaal van loopbanen die openstaan voor talent; filosofen noemen het formele gelijkheid van kansen. Als dat leidt tot minder aanstellingen uit één groep, is wie beslist volgens de smalle opvatting niets te verwijten. Het boek vindt dat ze weinig meer vraagt dan consequent zijn.

        **De brede opvatting.** Richt de samenleving zo in dat mensen met hetzelfde talent en dezelfde bereidheid om het te gebruiken dezelfde vooruitzichten krijgen, in welk gezin ze ook geboren zijn. De filosoof John Rawls noemde dit *fair equality of opportunity* (*A Theory of Justice*, 1971). Talent en inzet mogen nog steeds verschil maken; de plek waar iemand begon niet. Deze opvatting gaat minder over één besluit dan over scholen, zorg en de andere instellingen van een samenleving.

        **De middenopvatting.** Wie beslist, moet mensen die nu verschillen soms toch hetzelfde behandelen, als het verschil komt door iets wat niet mag tellen, zoals vroeger onrecht of pech. Texas geeft de beste 10 procent van elke eindexamenklas een gegarandeerde plek aan de staatsuniversiteiten: de beste leerling van een arme school hoeft de beste van een rijke school niet te verslaan op kale scores.

        **Wanneer begint de wedstrijd?** De econoom John Roemer (*Equality of Opportunity*, 1998) zegt dat de opvattingen verschillen in wanneer de wedstrijd begint. Hij splitst wat een uitkomst bepaalt in omstandigheden, waar iemand niets aan kan doen, zoals het inkomen van de ouders, en inzet, waar iemand zelf verantwoordelijk voor is. Alleen inzet mag tellen. Omdat omstandigheden ook de inzet vormen, vergelijkt Roemer mensen op hun plaats tussen wie dezelfde omstandigheden heeft: wie zich het meest inzet onder de armen, telt even zwaar als wie zich het meest inzet onder de rijken.

        **Elke opvatting, een andere test.** Heidari, Loi, Gummadi en Krause (2019) laten zien wat dat met een test doet. Alice en Bob hebben elk 5 jaar opleiding, Anna 3 en Ben 7; Alice en Anna zijn vrouwen, Bob en Ben mannen. Als de jaren zelf tellen als inzet, moeten Alice en Bob dezelfde vooruitzichten krijgen. Als inzet wordt gemeten als plaats binnen de eigen groep, zoals Roemer doet, moet Alice, de eerste van de vrouwen, dezelfde vooruitzichten krijgen als Ben, de eerste van de mannen, en kan Bob onder Alice uitkomen. Latere lessen maken van zulke ideeën tests die groepen vergelijken; het boek waarschuwt dat "moral notions never map exactly to technical criteria": morele ideeën passen nooit precies op technische maatstaven.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **gelijke kansen** | het ideaal dat mensen een gelijke kans krijgen op een plek of een goed; de opvattingen verschillen over wat een gelijke kans is |
        | **omstandigheden** | in de woorden van Roemer: wat een uitkomst bepaalt en waar iemand niets aan kan doen, zoals waar die persoon geboren is |
        | **inzet** | in de woorden van Roemer: wat een uitkomst bepaalt en waar iemand zelf verantwoordelijk voor is |
        """),
        L.watch_out("De brede opvatting vraagt geen gelijke uitkomsten. Volgens Rawls mogen talent en inzet nog steeds tot verschillende resultaten leiden; alleen de plek waar iemand begon niet."),
        mo.Html(C.picture({"title": "Drie opvattingen van een gelijke kans",
                           "rows": ["Kijkt naar", "De wedstrijd begint", "Voorbeeld"],
                           "views": [{"name": "Smal", "texts": ["wat mensen nu kunnen",
                                                                "bij het besluit",
                                                                "één lat voor elke sollicitant"]},
                                     {"name": "Midden", "texts": ["mensen binnen hun eigen groep",
                                                                  "eerder: sommige verschillen tellen niet",
                                                                  "de beste 10 procent van elke school"]},
                                     {"name": "Breed", "texts": ["de vooruitzichten van mensen met hetzelfde talent en dezelfde inzet",
                                                                 "bij de geboorte",
                                                                 "goede scholen voor elk kind"]}],
                           "alt": "Drie kaarten. Smal: kijkt naar wat mensen nu kunnen; de wedstrijd begint bij het besluit; voorbeeld: één lat voor elke sollicitant. Midden: kijkt naar mensen binnen hun eigen groep; de wedstrijd begint eerder, want sommige verschillen tellen niet; voorbeeld: de beste 10 procent van elke school. Breed: kijkt naar de vooruitzichten van mensen met hetzelfde talent en dezelfde inzet; de wedstrijd begint bij de geboorte; voorbeeld: goede scholen voor elk kind."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zoek je uit welke gelijke kans een regel of een model geeft:

    1. Schrijf op waarop hij mensen vergelijkt, zoals een score of jaren ervaring.
    2. Vraag je af of een verschil daarin kan komen door omstandigheden waar mensen niets aan kunnen doen.
    3. Kan dat niet, of vind je dat het niet mag uitmaken: dan kan de smalle opvatting, met één lat voor iedereen, genoeg zijn.
    4. Kan het wel: vergelijk mensen binnen de groep met dezelfde omstandigheden, zoals de middenopvatting doet, bijvoorbeeld de besten van elke school.
    5. Vraag je af wat er vóór het besluit moet veranderen, op school of op het werk, zodat de beginpunten dichter bij elkaar komen: de brede opvatting.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De smalle opvatting, met scores die de course verzon. `_scores` bevat de testscores van zes leerlingen van elk van twee scholen, van hoog naar laag. Er is één lat voor iedereen: wie minstens `_bar` scoort, komt binnen. Een loop in een loop gaat langs elke school, en langs elke score van die school. Voer de cel uit.
    """)
    return


@app.cell
def example_narrow():
    # verzonnen testscores van zes leerlingen per school, van hoog naar laag
    _scores = {"North": [91, 88, 85, 82, 76, 70], "South": [83, 77, 74, 69, 65, 61]}
    # de smalle opvatting: één lat voor iedereen
    _bar = 80

    _admitted = {}
    for _school in _scores:
        _admitted[_school] = []
        for _score in _scores[_school]:
            if _score >= _bar:
                _admitted[_school].append(_score)
    print(_admitted)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vier van North, één van South. Volgens de smalle opvatting valt daar niemand iets te verwijten. De middenopvatting neemt, zoals in Texas, in plaats daarvan de besten van elke school: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_admitted():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De besten van elke school

        De middenopvatting: de beste `top` leerlingen van elke school komen binnen, wat hun score ook is. De lijsten lopen van hoog naar laag. Vervang de `...` door een loop over de scholen, zoals in het uitgewerkte voorbeeld, met de namen zonder hun liggend streepje: `admitted` en `school`. Voor elke school krijgt `admitted[school]` de eerste `top` scores van die school. De laatste regel is `admitted`, zodat de check het ziet. De check probeert ook andere getallen.
        """),
        L.task("Reken `admitted` uit: de beste `top` scores van elke school."),
    ])
    return


@app.cell(hide_code=True)
def exercise_admitted():
    box_admitted = L.code_box("admitted", lines=9)
    box_admitted
    return (box_admitted,)


@app.cell(hide_code=True)
def check_admitted(box_admitted):
    result_admitted = L.check("admitted", box_admitted.value)
    result_admitted
    return (result_admitted,)


@app.cell(hide_code=True)
def hints_admitted():
    L.hints(
        "admitted",
        [
            "Gebruik de buitenste loop van het uitgewerkte voorbeeld, `for school in scores:`. Je hebt geen `if` en geen binnenste loop nodig: de lijsten lopen al van hoog naar laag.",
            "Een slice geeft de eerste items van een lijst: `scores[school][:2]` geeft de eerste 2 scores van één school. Gebruik `top` in plaats van 2.",
            "In de loop: `admitted[school] = ___`.",
        ],
        why="`admitted = {}`, dan `for school in scores:` met daaronder `admitted[school] = scores[school][:top]`. Dat geeft `{'North': [91, 88], 'South': [83, 77]}`: de 77 van South komt binnen, de 85 van North niet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_views():
    mo.md(r"""
    ### Oefening 2. Welke opvatting?

    Zes beweringen. Deel elke bewering in bij de opvatting waar ze bij hoort.
    """)
    return


@app.cell(hide_code=True)
def sort_views_cell():
    sort_views = L.sort("views")
    sort_views
    return (sort_views,)


@app.cell(hide_code=True)
def check_views(sort_views):
    result_views = L.check("views", sort_views)
    result_views
    return (result_views,)


@app.cell(hide_code=True)
def hints_views():
    L.hints(
        "views",
        [
            "Vraag je bij elke bewering af: kijkt ze naar mensen zoals ze nu zijn, houdt één besluit rekening met hun beginpunt, of gaat ze over hoe een hele samenleving is ingericht?",
            "De beste van de klas is de regel van Texas, en het gezin waarin iemand geboren wordt komt van Rawls.",
            "Smal: open voor talent en \\_\\_\\_. Midden: de beste van de klas en \\_\\_\\_.",
        ],
        why="**Smal:** open voor talent, en zelfde score, zelfde besluit. **Midden:** de beste van de klas, en vroeger onrecht. **Breed:** betere scholen, en overal geboren.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.md(r"""
    ### Oefening 3. Plaats in de eigen groep
    """)
    return


@app.cell(hide_code=True)
def choice_rank_cell():
    choice_rank = L.choice(
        "rank",
        question="Vier verzonnen sollicitanten. In groep A heeft Cleo 4 jaar opleiding en Dana 6; in groep B heeft Eli er 6 en Finn 8. Als inzet wordt gemeten als plaats binnen de eigen groep, zoals Roemer doet, wie staat dan gelijk met Dana?",
        options={
            "eli": "Eli, met dezelfde 6 jaar",
            "right": "Finn, eerste in groep B",
            "cleo": "Cleo, ook in groep A",
        },
        correct="right",
        why={
            "eli": "Dezelfde jaren is de vergelijking als de jaren zelf tellen als inzet. Op plaats is Eli tweede in groep B en Dana eerste in groep A. Zie *Elke opvatting, een andere test* in *Het idee*.",
            "right": "Dana is eerste in groep A en Finn eerste in groep B: op plaats binnen de eigen groep staan ze gelijk, ook met 6 tegen 8 jaar.",
            "cleo": "Binnen één groep verschillen de plaatsen: Dana is eerste, Cleo tweede. De plaats vergelijkt over groepen heen. Zie *Wanneer begint de wedstrijd?* in *Het idee*.",
        },
    )
    choice_rank
    return (choice_rank,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waarom is discriminatie fout?](/course/open/rai-why-discrimination-is-wrong?lang=nl) trok een groepsgemiddelde twee mensen met dezelfde score uit elkaar; de smalle opvatting verbiedt precies dat, door alleen te kijken naar wat telt voor de taak. De middenopvatting gaat verder: die behandelt mensen met verschillende scores soms hetzelfde, omdat het verschil komt van waar ze begonnen, zoals jouw code deed met de besten van elke school. Latere lessen maken van deze opvattingen tests die groepen vergelijken. Een van die tests heet *equal opportunity*; ondanks de naam is het één technische regel, en die past alleen onder aannames bij de ideeën uit deze les. De volgende les, *Uitgefilterd door een filter*, kijkt naar wat een automatisch filter doet met mensen met een handicap.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rank, result_admitted, result_views):
    L.wrap_up(result_admitted, result_views, choice_rank)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
