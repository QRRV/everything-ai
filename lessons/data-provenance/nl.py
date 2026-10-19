# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-provenance. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("data-provenance", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vragen stellen aan data](/course/open/data-explore?lang=nl) en [Dictionaries](/course/open/py-dicts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_missing():
    quick_missing = L.choice(
        "quick-missing",
        question="`df.isna().sum()` geeft `3` voor de kolom `temp`. Wat betekent dat?",
        options={
            "missing": "Er ontbreken 3 temperaturen",
            "zero": "3 temperaturen zijn 0",
            "low": "3 temperaturen liggen onder het gemiddelde",
        },
        correct="missing",
        why={
            "zero": "`df.isna()` gaat niet over de waarde 0. Wat markeert het met `True`? Zie de tabel met nieuwe woorden in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "low": "`df.isna()` vergelijkt niet met het gemiddelde. Wat markeert het met `True`? Zie de tabel met nieuwe woorden in [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "missing": "`df.isna()` is `True` voor elke ontbrekende waarde, en `.sum()` telt ze per kolom: 3 in `temp`.",
        },
    )
    quick_missing
    return (quick_missing,)


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="`card = {\"license\": \"CC0\"}`. Wat geeft `\"source\" in card`?",
        options={
            "true": "`True`",
            "false": "`False`",
            "error": "Een fout",
        },
        correct="false",
        why={
            "true": "`in` kijkt naar de keys van `card`. Is `\"source\"` daar een van? Zie de tabel met nieuwe woorden in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "error": "`in` stopt niet met een fout als een key er niet is. Wat geeft het dan? Zie de tabel met nieuwe woorden in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "false": "`card` heeft één key, `\"license\"`; `\"source\"` is er niet, dus `in` geeft `False`.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een dataset kan er perfect uitzien en toch niet de goede zijn om te gebruiken. Misschien mag je hem helemaal niet gebruiken, misschien hebben de mensen erin nooit toestemming gegeven, of misschien dekt hij maar één plek in één maand. Een model dat erop getraind is, neemt dat allemaal mee. Daarom zoek je, voordat je data gebruikt, uit waar die vandaan komt, en schrijf je dat op, zodat de volgende het niet opnieuw hoeft uit te zoeken.

    **Een voorbeeld uit het dagelijks leven.** Het etiket op een verpakking in de supermarkt: waar het vandaan komt, wat erin zit en tot wanneer het houdbaar is. Een dataset heeft ook zo'n etiket nodig.
    """)
    return


@app.cell(hide_code=True)
def predict_faces():
    guess_faces = L.guess(
        "faces",
        question="Een website biedt een dataset van 10.000 foto's van gezichten aan, gratis te downloaden. Mag je er volgens jou een model op trainen en dat model delen?",
        options={
            "free": "Ja: hij is gratis te downloaden",
            "depends": "Alleen als de licentie het toestaat, en de mensen op de foto's toestemming gaven",
            "no-sale": "Ja, zolang je het model niet verkoopt",
        },
        reveal="**Alleen als de licentie het toestaat, en de mensen op de foto's toestemming gaven.** Gratis te downloaden is niet hetzelfde als vrij te gebruiken: de licentie zegt wat je met de data mag. En foto's van gezichten gaan over mensen, dus het doet er ook toe of zij toestemming gaven voor dit gebruik.",
    )
    guess_faces
    return (guess_faces,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Elke dataset heeft een verhaal: wie hem maakte, hoe en wanneer, en wat je ermee mag. Dat verhaal is zijn herkomst. Met vier vragen kom je een heel eind: Waar komt hij vandaan? Onder welke licentie? Gaven de mensen erin toestemming? En wat dekt hij niet? Je beantwoordt ze voordat je de data gebruikt, en je schrijft de antwoorden ernaast op, in een datasheet.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **dataherkomst** | waar een dataset vandaan komt: wie hem maakte, uit welke bronnen, wanneer, en onder welke licentie en toestemming | |
    | **licentie** | de voorwaarden waaronder je een dataset, een model of code mag gebruiken, aanpassen en delen | `"CC BY 4.0"` |
    | **CC0** | de maker ziet af van zijn rechten: je mag er alles mee, zonder iemand te noemen; het zegt niets over de mensen in de data | `"CC0 1.0"` |
    | **CC BY** | je mag het gebruiken, aanpassen en delen, ook om er geld mee te verdienen, als je de maker noemt, naar de licentie verwijst en zegt wat je veranderde; met SA krijgt wat je ervan maakt en deelt dezelfde licentie; met NC mag je er geen geld mee verdienen | `"CC BY-SA 4.0"` |
    | **geen licentie** | heeft een dataset geen licentie, ga er dan van uit dat je hem niet mag kopiëren of delen: vraag het de maker voordat je hem deelt of erop voortbouwt | |
    | **toestemming** | instemming die mensen vrij en bewust geven met het gebruik van hun gegevens voor een genoemd doel | |
    | **beperkingen** | wat de data niet dekt: andere plekken, andere tijden, mensen of gevallen die ontbreken | |
    | **datasheet** | een document bij een dataset dat deze vragen beantwoordt; hier een dictionary | `card["license"]` |
    """)
    return


@app.cell(hide_code=True)
def picture_card():
    mo.vstack([
        mo.md("**De datasheet van Palmer penguins**"),
        draw.dict_boxes([("source", "Palmer Station, Antarctica"), ("collected", "2007 tot 2009"), ("license", "CC0 1.0"),
                         ("consent", "geen mensen erin"), ("limits", "alleen 3 eilanden, 2007–2009")],
                        key_label="veld", value_label="antwoord"),
        mo.md(r"""
        Een echte dataset, kort: een paar lichaamsmaten van 344 pinguïns van drie soorten, op drie eilanden bij Palmer Station op Antarctica. Elk veld beantwoordt één vraag. De licentie, CC0, laat je er alles mee doen; er zitten geen mensen in om iets te vragen; en de beperkingen zeggen wat je er niet uit kunt leren, zoals iets over pinguïns ergens anders.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Gratis te downloaden is niet vrij te gebruiken. Zonder licentie ga je ervan uit dat je de data niet mag kopiëren of delen, en vraag je het de maker. En bij data over mensen kijk je waarvoor ze toestemming gaven: toestemming voor het ene gebruik is geen toestemming voor elk gebruik."),
        mo.md(r"""
        ### Recept

        Voordat je een dataset gebruikt:

        1. Zoek de bron: wie hem maakte, waaruit, en wanneer. Kijk of er een README of een datasheet naast de data staat.
        2. Zoek de licentie, en lees wat die toestaat: gebruiken, aanpassen, delen, er geld mee verdienen.
        3. Gaat de data over mensen: ga na of ze toestemming gaven voor dit gebruik; zo niet, zoek dan uit of je de data wel mag gebruiken.
        4. Schrijf de beperkingen op: welke plekken, welke tijden, wie of wat erin zit, en wie of wat ontbreekt.
        5. Bewaar de antwoorden naast de data, als datasheet, zodat wie de dataset daarna gebruikt, ze kan lezen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De datasheet van Palmer penguins als dictionary, met een veld voor elke vraag. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. De course houdt zulke notities zelf ook bij: `data/README.md` zegt waar elke dataset vandaan komt die hij zelf aanbiedt, zoals de toneelstukken van Shakespeare die een latere les gebruikt. Voer de cel uit.
    """)
    return


@app.cell
def example_card():
    _card = {
        # 1. de bron, en wanneer
        "source": "Kristen Gorman en het Palmer Station Long Term Ecological Research-programma, Antarctica",
        "collected": "2007 tot 2009",
        # 2. de licentie
        "license": "CC0 1.0",
        # 3. mensen die toestemming moesten geven
        "consent": "niet nodig: er zitten geen mensen in",
        # 4. de beperkingen
        "limits": "alleen 344 pinguïns, op 3 eilanden bij één station, 2007 tot 2009; sommige waarden ontbreken",
    }
    # 5. lees hem terug, veld voor veld
    for _field in ["source", "collected", "license", "consent", "limits"]:
        print(f"{_field}: {_card[_field]}")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke regel beantwoordt één vraag. De licentie, CC0 1.0, is de meest open die er is. Toestemming is niet nodig, want er zitten geen mensen in de data. De beperkingen waarschuwen je: een model dat op deze pinguïns getraind is, weet niets van pinguïns op andere eilanden of uit andere jaren, en over de ontbrekende waarden moet je beslissen voordat je erop traint, zoals in [Vragen stellen aan data](/course/open/data-explore?lang=nl). Bij de foto's van gezichten uit de gok zou het veld `consent` moeten zeggen waarmee de mensen instemden; weet niemand dat, dan is dat een reden om ze niet te gebruiken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_license():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat staat CC BY toe?
    """)
    return


@app.cell(hide_code=True)
def choice_license_cell():
    choice_license = L.choice(
        "cc-by",
        question="Een dataset met vogelgezang heeft de licentie CC BY 4.0. Je wilt er een model op trainen, de opnamen een beetje aanpassen, en ze delen op je eigen website. Mag dat?",
        options={
            "never": "Nee: je mag hem alleen voor jezelf gebruiken",
            "silent": "Ja, en je hoeft niemand te noemen",
            "credit": "Ja, als je de maker noemt, naar de licentie verwijst en zegt wat je veranderde",
        },
        correct="credit",
        why={
            "never": "Zoek CC BY op in de tabel met nieuwe woorden: wat mag je ermee?",
            "silent": "Dat is CC0. Waar vragen de letters BY om? Zie de tabel met nieuwe woorden.",
            "credit": "CC BY staat gebruiken, aanpassen en delen toe, ook om er geld mee te verdienen, zolang je de maker noemt, naar de licentie verwijst en zegt wat je veranderde. BY staat voor het noemen: gemaakt door wie.",
        },
    )
    choice_license
    return (choice_license,)


@app.cell(hide_code=True)
def text_exercise_card():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De datasheet van het weerstation

        In [Vragen stellen aan data](/course/open/data-explore?lang=nl) verkende je de maand van een weerstation. De datasheet ervan heeft al twee velden. Voeg de andere drie toe: `license`, `consent` en `limits`, elk met een tekst. De feiten: de course heeft de data verzonnen en deelt het bestand onder CC BY-SA 4.0, net als de tekst van zijn lessen; er zitten geen mensen in de data; het is één verzonnen station voor één maand, met twee ontbrekende temperaturen en -999 voor een kapotte windmeter. Vervang de `...`. De laatste regel is `card`, zodat de check het ziet.
        """),
        L.task("Maak `card` compleet: voeg `license`, `consent` en `limits` toe."),
    ])
    return


@app.cell(hide_code=True)
def exercise_card():
    box_card = L.code_box("station-card", lines=9)
    box_card
    return (box_card,)


@app.cell(hide_code=True)
def check_card(box_card):
    result_card = L.check("station-card", box_card.value)
    result_card
    return (result_card,)


@app.cell(hide_code=True)
def hints_card():
    L.hints(
        "station-card",
        [
            "Stap 2 tot en met 4 van het recept: de licentie, de toestemming en de beperkingen, elk als één paar van de dictionary.",
            "Elk paar is een key tussen aanhalingstekens, een dubbele punt, en een tekst tussen aanhalingstekens, met een komma erachter, net als de twee paren die er al staan.",
            "`\"license\": \"___\",` op een eigen regel, en dan hetzelfde voor `consent` en `limits`.",
        ],
        why="De licentie van de stationsdata is CC BY-SA 4.0, dezelfde als die van de tekst van de course. Toestemming is niet nodig, want er zitten geen mensen in. De beperkingen zeggen wat een model er niet uit kan leren: één verzonnen station, één maand, twee ontbrekende temperaturen, en -999 waar de windmeter kapot was.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Welke velden ontbreken?

        Maak `missing_fields(card)` af. Hij krijgt een datasheet als dictionary, en geeft een lijst terug van de velden uit `FIELDS` die de datasheet mist: een veld telt als ontbrekend als het er niet is, of als zijn tekst leeg is, `""`. Houd de volgorde van `FIELDS` aan. Vervang de `...`. De check roept `missing_fields({"source": "a weather station", "license": "CC0 1.0"})` aan: dat moet `["collected", "consent", "limits"]` geven. De check probeert ook andere datasheets.
        """),
        L.task("Laat `missing_fields(card)` de velden uit `FIELDS` teruggeven die `card` mist."),
    ])
    return


@app.cell(hide_code=True)
def exercise_missing():
    box_missing = L.code_box("missing-fields", lines=12)
    box_missing
    return (box_missing,)


@app.cell(hide_code=True)
def check_missing(box_missing):
    result_missing = L.check("missing-fields", box_missing.value)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing-fields",
        [
            "Ga veld voor veld door `FIELDS`, en houd de ontbrekende bij in een lijst.",
            "Een veld ontbreekt als het geen key van `card` is, of als zijn tekst leeg is. Kijk eerst met `in` of het er is: pas dan mag je naar de tekst kijken.",
            "`if field in card:`, daaronder `if card[field] == ___:` en `missing.append(field)`; onder `else:` ook `missing.append(field)`. Na de loop `return missing`.",
        ],
        why="De loop gaat door `FIELDS`, dus de lijst houdt hun volgorde. `if field in card:` kijkt alleen naar de tekst van een veld dat er is, dus een ontbrekende key geeft geen fout; `else:` vangt de velden die er helemaal niet zijn.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Vragen stellen aan data](/course/open/data-explore?lang=nl) vroeg je een tabel wat erin zit; nu vraag je ook waar hij vandaan komt, en dat kan geen tabel je vertellen. In [Dictionaries](/course/open/py-dicts?lang=nl) zocht je waarden op met een key; een datasheet is zo'n dictionary. Later in de course controleer je de bron en de licentie van elke dataset waarop je traint, en krijgt een getraind model een eigen model card. In de volgende les, *Voorbeelden, features en labels*, leer je de woorden voor wat er in een dataset zit waar een model van leert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_license, result_card, result_missing):
    L.wrap_up(choice_license, result_card, result_missing)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
