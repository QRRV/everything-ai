# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-principles. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-principles", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_duty():
    quick_duty = L.choice(
        "quick-duty",
        question="Als een ziekenhuis patiëntgegevens deelt zonder toestemming te vragen, kan het meer levens redden. Waarom kan de deontologie toch nee zeggen?",
        options={
            "outcome": "Omdat het uiteindelijk minder levens redt dan het belooft",
            "character": "Omdat een dokter met een goed karakter zich er ongemakkelijk bij voelt",
            "right": "Omdat mensen recht hebben op toestemming, wat de uitkomst ook is",
        },
        correct="right",
        why={
            "outcome": "Dat is nog steeds een argument over uitkomsten, de vraag van het consequentialisme. Welke theorie houdt vast aan plichten en rechten, wat de uitkomst ook is? Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "character": "Dat is de vraag van de deugdethiek. Welke theorie houdt vast aan plichten en rechten? Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "right": "De deontologie houdt vast aan plichten en rechten, wat de uitkomst ook is. Kant vond dat je een mens nooit alleen als middel mag gebruiken.",
        },
    )
    quick_duty
    return (quick_duty,)


@app.cell(hide_code=True)
def quick_check_total():
    quick_total = L.choice(
        "quick-total",
        question="In het ziekenhuis van die les voorkomt nearby first 32 heropnames en need first 28, maar need first helpt een groter aandeel van de groep die het slechtst af is. Welke theorie wijst naar nearby first?",
        options={
            "right": "Het consequentialisme, dat het goede over iedereen optelt",
            "contract": "Een eerlijke afspraak, die zorgt voor wie het slechtst af is",
            "care": "Zorgethiek, die begint bij wie in de regio van wie afhangt",
        },
        correct="right",
        why={
            "right": "Het consequentialisme telt op wat elke keuze teweegbrengt, en 32 is meer dan 28.",
            "contract": "Rawls betoogde dat je achter de sluier van onwetendheid zorgt voor wie het slechtst af is, en dat wijst naar need first. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "care": "Zorgethiek kijkt naar relaties en naar wat mensen nodig hebben, niet naar een totaal. Welke theorie telt op? Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_total
    return (quick_total,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sinds 2016 hebben overheden, bedrijven, universiteiten en anderen tientallen documenten met principes voor AI gepubliceerd. In 2019 verzamelden Anna Jobin, Marcello Ienca en Effy Vayena er 84 en telden ze welke principes elk document noemt. Op het eerste gezicht is de wereld het eens: een handvol woorden komt bijna overal terug. Deze les telt die woorden, en kijkt daarna wat de documenten ermee bedoelen.

    **Een voorbeeld uit het dagelijks leven.** Bijna elk restaurant noemt zijn eten vers. Bij het ene betekent dat vanochtend geplukt, bij het andere vorige maand ingevroren en vandaag ontdooid. Het woord is hetzelfde; wat je krijgt niet.
    """)
    return


@app.cell(hide_code=True)
def predict_transparency():
    guess_transparency = L.guess(
        "transparency",
        question="Hoeveel van de 84 documenten met principes voor AI noemen transparantie, denk je?",
        options={
            "quarter": "Ongeveer 20",
            "half": "Ongeveer 40",
            "right": "Ongeveer 70",
        },
        reveal="**73 van de 84**, meer dan elk ander principe. Rechtvaardigheid en fairness kwamen op de tweede plaats, met 68 (Jobin, Ienca en Vayena, 2019). *Het idee* laat de rest zien.",
    )
    guess_transparency
    return (guess_transparency,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Overheden, bedrijven en anderen schrijven op wat zij goede AI vinden, in korte principes. Drie reeksen zijn bekend:

        - De **OESO-principes voor AI** (2019, bijgewerkt in 2024) waren de eerste waar regeringen het samen over eens werden. Ze noemen er vijf: inclusieve groei, duurzame ontwikkeling en welzijn; respect voor de rechtsstaat, mensenrechten en democratische waarden, waaronder fairness en privacy; transparantie en explainability; robuustheid, beveiliging en veiligheid; en verantwoording.
        - De *Aanbeveling over de ethiek van kunstmatige intelligentie* van UNESCO werd in november 2021 aangenomen door alle 193 lidstaten.
        - In 2019 beschreef de High-Level Expert Group on AI van de EU **betrouwbare AI**: wettig, ethisch en robuust, met zeven eisen, waaronder menselijk toezicht, transparantie en fairness. In 2020 maakte die groep er **ALTAI** van, een lijst vragen die een organisatie over haar eigen systeem beantwoordt.

        Bedrijven spreken vaak van **verantwoorde AI**: AI bouwen en gebruiken volgens zulke principes, met de werkwijzen die daarbij helpen.

        Het plaatje laat zien wat Jobin, Ienca en Vayena in hun 84 documenten vonden. Vijf principes komen in meer dan de helft terug, en vier van de woorden, weldoen, niet-schaden, autonomie en rechtvaardigheid, waren al de kern van de medische ethiek (Beauchamp en Childress, 1979). Dat leest als overeenstemming. Maar hetzelfde onderzoek vond dat de documenten met hetzelfde woord heel verschillende dingen bedoelen, er andere redenen voor geven en er iets anders over zeggen hoe je het in de praktijk brengt. **Transparantie**, het vaakst genoemd, kan betekenen dat je zegt dat er AI wordt gebruikt, dat je de data of de code publiceert, of dat je een beslissing uitlegt.

        Ook wie ze schreef, is ongelijk verdeeld. De Verenigde Staten en het Verenigd Koninkrijk schreven samen meer dan een derde van de documenten; geen enkel document kwam uit een land in Afrika of Zuid-Amerika alleen: die werelddelen komen alleen voor via internationale organisaties.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **OESO-principes voor AI** | de eerste principes voor AI waar regeringen het samen over eens werden (2019, bijgewerkt in 2024): vijf, zoals mensenrechten, transparantie en verantwoording |
        | **betrouwbare AI** | de naam die de Europese expertgroep gebruikt voor AI die wettig, ethisch en robuust is, met zeven eisen zoals menselijk toezicht, transparantie en fairness |
        | **ALTAI** | de Assessment List for Trustworthy AI (2020): de vragen van de expertgroep om een systeem aan die zeven eisen te toetsen |
        | **verantwoorde AI** | AI bouwen en gebruiken volgens zulke principes, en de werkwijzen en regels die daarbij helpen |
        | **transparantie** | openheid over een AI-systeem: dat het gebruikt wordt, wat het doet, hoe het gemaakt is en met welke data |
        """),
        L.watch_out("Hoe vaak een principe genoemd wordt, zegt hoe populair het woord is, niet wat het betekent, en ook niet of iemand ernaar handelt. Twee documenten kunnen allebei fairness noemen en dingen vragen die botsen."),
        mo.Html(C.picture({"title": "In hoeveel van de 84 documenten elk principe genoemd wordt", "x": "documenten", "half": "helft",
                           "names": ["transparantie", "rechtvaardigheid en fairness", "niet-schaden", "verantwoordelijkheid", "privacy", "weldoen",
                                     "vrijheid en autonomie", "vertrouwen", "duurzaamheid", "waardigheid", "solidariteit"],
                           "alt": "Van 84 documenten met principes voor AI noemen er 73 transparantie, 68 rechtvaardigheid en fairness, 60 niet-schaden, 60 verantwoordelijkheid, 47 privacy, 41 weldoen, 34 vrijheid en autonomie, 28 vertrouwen, 14 duurzaamheid, 13 waardigheid en 6 solidariteit. Een gestippelde lijn markeert de helft, bij 42."})),
        mo.md(r"""
        Aantallen uit Jobin, Ienca en Vayena (2019), tabel 2. De gestippelde lijn ligt bij de helft van de 84 documenten.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo lees je een reeks principes voor AI:

    1. Zet de principes op een rij die erin staan.
    2. Zoek bij elk principe wat dit document ermee bedoelt: een definitie, een voorbeeld, een regel.
    3. Kijk hoe het in de praktijk wordt gebracht: een lijst vragen zoals ALTAI, iemand die ervoor verantwoordelijk is.
    4. Vergelijk het met een andere reeks: hetzelfde woord, dezelfde betekenis?
    5. Schrijf op wat er ontbreekt, en wie het schreef.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De zes principes onderaan het plaatje. Voor elk het aandeel van de 84 documenten dat het noemt: het aantal gedeeld door het aantal documenten. Voer de cel uit.
    """)
    return


@app.cell
def example_shares():
    _named = {"beneficence": 41, "freedom and autonomy": 34, "trust": 28, "sustainability": 14, "dignity": 13, "solidarity": 6}
    _documents = 84

    for _principle in _named:
        print(_principle, round(_named[_principle] / _documents, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zelfs weldoen staat in minder dan de helft van de documenten, en duurzaamheid in maar 0,17 ervan. Welke principes noemt meer dan de helft? Dat is oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_common():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Meer dan de helft

        `named` bevat de aantallen van alle 11 principes en `documents` het aantal documenten; de loop gaat met `principle` langs de principes. Vervang de `...` door twee regels: een `if` die het aandeel, uitgerekend zoals in het uitgewerkte voorbeeld, vergelijkt met de helft, en daaronder de regel die het principe overhoudt. De laatste regel is `common`, zodat de check het ziet. De check probeert ook een ander aantal documenten.
        """),
        L.task("Zet in `common` de namen van de principes die meer dan de helft van de documenten noemt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_common():
    box_common = L.code_box("common", lines=9)
    box_common
    return (box_common,)


@app.cell(hide_code=True)
def check_common(box_common):
    result_common = L.check("common", box_common.value)
    result_common
    return (result_common,)


@app.cell(hide_code=True)
def hints_common():
    L.hints(
        "common",
        [
            "Het aandeel zoals in het uitgewerkte voorbeeld, nu in een `if`: twee regels in de loop.",
            "Welk getal staat voor de helft als je een aandeel vergelijkt? En wat doet de loop met een principe dat erdoor komt?",
            "`if named[principle] / documents > ___:` en daaronder `common.___(principle)`.",
        ],
        why="`if named[principle] / documents > 0.5:` met `common.append(principle)` eronder houdt `transparency`, `justice and fairness`, `non-maleficence`, `responsibility` en `privacy` over. `beneficence` haalt het net niet, met 41 van de 84.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_meanings():
    mo.md(r"""
    ### Oefening 2. Eén woord, drie betekenissen

    Elk van deze zes gevallen heet in een of andere richtlijn transparantie. Zet elk geval bij wat het transparant maakt.
    """)
    return


@app.cell(hide_code=True)
def sort_meanings_cell():
    sort_meanings = L.sort("meanings")
    sort_meanings
    return (sort_meanings,)


@app.cell(hide_code=True)
def check_meanings(sort_meanings):
    result_meanings = L.check("meanings", sort_meanings)
    result_meanings
    return (result_meanings,)


@app.cell(hide_code=True)
def hints_meanings():
    L.hints(
        "meanings",
        [
            "Vraag bij elk geval: gaat het over de beslissing over één persoon, over mensen vertellen dat er AI is, of over hoe het hele systeem gemaakt is?",
            "Twee gevallen gaan over één geval van één persoon. Twee andere vertellen alleen dat er AI bij komt kijken, en leggen niets uit.",
            "Eén beslissing uitleggen: de geweigerde lening en \\_\\_\\_. Zeggen dat er AI wordt gebruikt: het chatvenster en \\_\\_\\_. Het systeem openleggen: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Eén beslissing uitleggen:** de geweigerde lening en het gemarkeerde bericht. **Zeggen dat er AI wordt gebruikt:** het chatvenster en het label. **Het systeem zelf openleggen:** de data en de grenzen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fairness():
    mo.md(r"""
    ### Oefening 3. Hetzelfde woord
    """)
    return


@app.cell(hide_code=True)
def choice_fairness_cell():
    choice_fairness = L.choice(
        "fairness",
        question="Twee bedrijven noemen allebei fairness als principe. Voor bedrijf A betekent dat dat zijn systeem voor de ene groep klanten evenveel fouten maakt als voor de andere; voor bedrijf B dat voor elke klant dezelfde regel geldt. Wat zegt het gedeelde woord je?",
        options={
            "same": "Dat hun systemen dezelfde mensen op dezelfde manier behandelen",
            "right": "Weinig: één woord kan heel verschillende dingen vragen",
            "stronger": "Dat bedrijf A fairness serieuzer neemt dan bedrijf B",
        },
        correct="right",
        why={
            "same": "De twee betekenissen kunnen zelfs uit elkaar lopen: één regel voor elke klant kan toch meer fouten maken voor één groep. Wat vonden Jobin, Ienca en Vayena over de betekenissen? Zie *Het idee*.",
            "right": "De woorden komen overeen, de betekenissen niet, zoals Jobin, Ienca en Vayena vonden. Wat een principe vraagt, leer je pas uit wat het in dat document betekent.",
            "stronger": "Beide bedrijven bedoelen iets met fairness; ze bedoelen verschillende dingen. Niets hier zegt welk bedrijf er meer voor doet. Zie *Het idee*.",
        },
    )
    choice_fairness
    return (choice_fairness,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) leerde je theorieën kennen die elk hun eigen vraag stellen. Een principe is korter: een woord als fairness of niet-schaden, dat elke theorie op haar eigen manier kan invullen, en dat is een deel van de reden dat zoveel documenten het erover eens zijn. In de volgende les, *Als principes alleen woorden zijn*, test je of een principe wel iets verandert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fairness, result_common, result_meanings):
    L.wrap_up(result_common, result_meanings, choice_fairness)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
