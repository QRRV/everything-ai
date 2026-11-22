# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-what-is-nlp. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-what-is-nlp", "nl")
    C = L.checks   # negen verzonnen systemen, en negen verzonnen verzoeken aan de hulp van een fietsenwinkel


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl) en [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_numbers():
    quick_numbers = L.choice(
        "quick-numbers",
        question="Wat geeft `[ord(ch) for ch in \"hi\"]`?",
        options={
            "bytes": "b'hi', de twee tekens als bytes van UTF-8-tekst",
            "right": "[104, 105], één code point voor elk teken",
            "count": "[1, 1], hoe vaak elk teken voorkomt",
        },
        correct="right",
        why={
            "right": "`ord` geeft het code point van een teken: 104 voor h en 105 voor i. Voor een computer is tekst een rij getallen; deze les vraagt wat een computer met die rij kan doen.",
            "bytes": "Bytes krijg je met `\"hi\".encode()`. `ord` geeft het code point van één teken, een getal. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "count": "`ord` telt niets: het geeft het code point van elk teken, 104 en 105. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
        },
    )
    quick_numbers
    return (quick_numbers,)


@app.cell(hide_code=True)
def quick_check_bigrams():
    quick_bigrams = L.choice(
        "quick-bigrams",
        question="Een bigram-model telt de paren tekens in de tekst `banana`. Welk teken geeft het na `n`?",
        options={
            "n": "n, want een bigram herhaalt het teken dat het krijgt",
            "b": "b, het eerste teken van de tekst",
            "right": "a, want elke n in de tekst wordt gevolgd door een a",
        },
        correct="right",
        why={
            "right": "Beide paren die met n beginnen zijn na, dus het model geeft a. Het schrijft tekst zonder te weten wat die betekent: één kant van waar deze les over gaat.",
            "n": "Een bigram is twee tekens naast elkaar in de tekst. In banana wordt elke n gevolgd door een a. Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
            "b": "Het model kijkt naar het teken ervoor, niet naar het begin van de tekst: na n komt a, twee keer. Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
        },
    )
    quick_bigrams
    return (quick_bigrams,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je typt drie woorden in een zoekvak en krijgt de pagina die je bedoelde. Een mailprogramma zet een bericht over een prijs waar je nooit aan meedeed in de spammap. Een app maakt van een Spaanse menukaart een Nederlandse. Elke keer werkt een computer met menselijke taal, die voor mensen geschreven is en niet voor computers.

    Met deze les begint het domein over taal en zoeken. Ze zegt hoe dat vakgebied heet en wat erbij hoort, deelt systemen in naar wat ze met taal doen, en bouwt de eerste stap van een klein programma dat vragen beantwoordt: vinden waar een vraag over gaat.

    **Een voorbeeld uit het dagelijks leven.** Een postkantoor doet twee dingen met dezelfde brief: het adres lezen, en een etiket printen. Het ene haalt iets uit tekst; het andere maakt tekst.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een computer kan iets uit tekst halen, zoals waar een vraag over gaat, of tekst schrijven. Een chatassistent beantwoordt vragen over een winkel. Welke van de twee doet zo'n assistent?",
        options={
            "read": "Alleen iets uit tekst halen: het sorteert vragen, en mensen beantwoorden ze",
            "write": "Alleen tekst schrijven: wat er waarschijnlijk volgt",
            "right": "Allebei: snappen wat je vraagt, dan een antwoord schrijven",
        },
        reveal="**Allebei.** Om te antwoorden moet de assistent eerst snappen wat je vraagt, en daarna een antwoord in woorden schrijven. *Begrijpen en maken* in *Het idee* geeft de twee kanten hun namen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Natural language processing

    Een natuurlijke taal is een taal die mensen spreken, schrijven of gebaren, zoals Nederlands of Engels; Python is er geen. Julia Hirschberg en Christopher Manning schreven in 2015 in *Science* over een vakgebied dat rekentechnieken inzet om menselijke taal te leren, te begrijpen en te maken. **Natural language processing**, kortweg NLP, is dat vakgebied: computers die met menselijke taal werken.

    De Association for Computational Linguistics (ACL) noemt voorbeelden van zulke systemen: spraakherkenning, text-to-speech, telefoonsystemen die je met je stem bedient, zoekmachines op het web, tekstverwerkers en lesmateriaal voor talen, en dat is maar een greep. Hirschberg en Manning wezen op producten uit 2015, zoals Siri van Apple en Skype Translator. Het leerboek van Jurafsky en Martin noemt in zijn versie van 19 augustus 2026 spam herkennen een belangrijke commerciële toepassing.

    ### Computationele linguïstiek

    De pagina van de ACL zegt onder het kopje over het vakgebied: **computationele linguïstiek** is de wetenschap die taal bestudeert vanuit het rekenen, met computers. Is dat een ander vakgebied dan NLP? De bronnen zijn het niet eens. Dezelfde pagina zegt dat het vakgebied vaak computationele linguïstiek of NLP heet, en Hirschberg en Manning schrijven dat computationele linguïstiek ook NLP heet. Jurafsky en Martin schrijven dat de twee soms uit elkaar worden gehouden: de structuur van taal berekenen helpt bij vragen over taal zelf; dat onderzoek heet computationele linguïstiek en wordt soms van NLP onderscheiden.

    Wie ze wel uit elkaar houdt, ziet computationele linguïstiek als de wetenschap die wil weten hoe taal werkt, en NLP als de techniek die systemen wil die werken. De pagina van de ACL noemt beide drijfveren: soms wetenschappelijk, en soms vooral technisch, omdat men een onderdeel wil bouwen dat werkt in een systeem voor spraak of taal.

    ### Begrijpen en maken

    De omschrijving van Hirschberg en Manning heeft drie werkwoorden: leren, begrijpen en maken. De laatste twee geven NLP twee kanten.

    **Natural language understanding**, kortweg NLU, gaat van taal naar wat die betekent. Emily Bender en Alexander Koller (2020) zien betekenis als het verband tussen wat er gezegd wordt en wat iemand ermee wil overbrengen. Begrijpen is voor hen terugvinden wat iemand bedoelde, gegeven de woorden: de uitdrukking *e* en de bedoeling *i* erachter. Ze noemen NLU op het niveau van mensen een grote uitdaging voor AI. Een spamfilter zit aan deze kant: er gaat tekst in, en er komt een label uit.

    **Natural language generation**, kortweg NLG, gaat de andere kant op. Albert Gatt en Emiel Krahmer (2018) omschrijven het als tekst of spraak maken uit invoer die geen taal is, zoals de getallen in een tabel. Een weersite die een verwachting schrijft uit de metingen zit aan deze kant. Ze wijzen erop dat het volgens anderen nogal lastig is om NLG precies te omschrijven, en ze rekenen tekst maken uit tekst, zoals een samenvatting, ook tot NLG.

    Veel systemen doen allebei. Een menukaart vertalen is snappen wat het Spaans zegt en dat in het Nederlands schrijven. Deze les deelt systemen in op een eenvoudige manier van de course zelf: tekst erin is begrijpen, tekst eruit is maken, tekst erin en tekst eruit is allebei, en zonder tekst aan een van beide kanten is het geen NLP. Echte systemen zijn minder netjes: een zoekmachine leest je woorden, en geeft een lijst pagina's terug die anderen schreven.
    """)
    return


@app.cell(hide_code=True)
def sides_cell():
    mo.Html(C.sides_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **natural language processing (NLP)** | computers die met menselijke taal werken: taal begrijpen, maken, of ervan leren |
        | **computationele linguïstiek** | taal zelf bestuderen met computers; vaak een andere naam voor NLP |
        | **natural language understanding (NLU)** | de kant van NLP die van taal naar betekenis gaat |
        | **natural language generation (NLG)** | de kant van NLP die van data of betekenis naar taal gaat |
        | `C.SYSTEMS` | negen verzonnen systemen: dicts met `"text"`, `"input"` en `"output"` |
        | `C.REQUESTS`, `C.KEYWORDS` | negen verzonnen verzoeken aan een fietsenwinkel, en een dict van elke soort verzoek naar woorden die erop wijzen |
        | `elif` | else if: Python bekijkt deze voorwaarde alleen als de `if` ervoor niet waar was |
        | `a in s` | `True` als de string `s` de string `a` bevat: `"pen" in "open"` is `True` |
        """),
        L.watch_out("NLU is de naam van een taak, geen bewering dat een systeem begrijpt zoals mensen dat doen. Bender en Koller stellen, in wat ze zelf een position paper noemen, dat een systeem dat alleen op vorm getraind is bij voorbaat geen betekenis kan leren."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        ### Recept

        1. Kijk wat erin gaat: menselijke taal, geschreven of gesproken, of data zoals getallen en labels.
        2. Kijk op dezelfde manier wat eruit komt.
        3. Vraag eerst of er tekst in gaat en tekst uit komt. Dan doet het systeem beide kanten.
        4. Anders: gaat er alleen tekst in, dan is het begrijpen (NLU); komt er alleen tekst uit, dan is het maken (NLG).
        5. Gaat er geen taal in en komt er geen uit, dan is het geen NLP, hoe slim het systeem ook is.
        """),
        mo.Html(C.grid_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt het derde systeem van de course en drukt de tekst af, wat erin gaat en wat eruit komt. `C.SYSTEMS[2]` is een dict; `system["input"]` zoekt de waarde onder de key `"input"` op.
    """)
    return


@app.cell
def example_system():
    system = C.SYSTEMS[2]   # het derde systeem
    print(system["text"])
    print("in:", system["input"])
    print("out:", system["output"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De cel drukt de tekst in het Engels af: een app vertaalt een Spaanse menukaart in het Nederlands. Er gaat tekst in en er komt tekst uit: volgens het recept beide kanten. De app moet snappen wat het Spaans zegt, en het dan in het Nederlands schrijven.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_side():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. side: welke kant van NLP

        `C.SYSTEMS` bevat negen verzonnen systemen, elk een dict met de keys `"text"`, `"input"` (erin) en `"output"` (eruit); onder de laatste twee staat `"text"` of `"data"`. Schrijf `side(system)`: de kant van NLP van de dict `system`, als een van de strings `"both"` (allebei), `"NLU"`, `"NLG"` of `"none"` (geen). Stel de vragen van het recept in de volgorde van het recept, met `if` en `and`, daarna `elif`. Vervang de `...`. De check roept `side(SYSTEMS[0])` aan, daarna de andere systemen en een eigen systeem.
        """),
        L.task("Vergelijk wat erin gaat en wat eruit komt met \"text\", en geef de kant terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_side():
    box_side = L.code_box("side", lines=8)
    box_side
    return (box_side,)


@app.cell(hide_code=True)
def check_side(box_side):
    result_side = L.check("side", box_side.value)
    result_side
    return (result_side,)


@app.cell(hide_code=True)
def hints_side():
    L.hints(
        "side",
        [
            "*Recept* geeft de vier antwoorden, en welke vraag eerst komt.",
            "Welke vraag moet eerst, zodat een systeem met aan beide kanten tekst niet stopt bij een vraag over één kant? En welk antwoord blijft over voor de laatste regel?",
            "`    if system[\"input\"] == \"text\" and ___:`",
        ],
        why="`if system[\"input\"] == \"text\" and system[\"output\"] == \"text\":` met `return \"both\"` eronder, dan `elif system[\"input\"] == \"text\":` met `return \"NLU\"`, `elif system[\"output\"] == \"text\":` met `return \"NLG\"`, en als laatste `return \"none\"`. `side(SYSTEMS[0])` geeft `\"NLU\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_side():
    mo.md(r"""
    Twee systemen begrijpen alleen, drie maken alleen tekst, drie doen allebei, en één is helemaal geen NLP. De spammap en de ingedeelde mails eindigen in een label; de weersverwachting, het wedstrijdverslag en de melding over vertraging beginnen bij getallen. De spreadsheet is slim met getallen, maar er gaat geen taal in en er komt geen uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_intent():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. intent: waar een verzoek over gaat

        Een verzonnen fietsenwinkel wil een programma dat vragen beantwoordt. De eerste stap is begrijpen: waar gaat een verzoek over? `C.REQUESTS` bevat negen verzoeken, in het Engels, en `C.KEYWORDS` is een dict van elke soort verzoek (`"hours"` voor openingstijden, `"repair"` voor reparatie, `"price"` voor prijs) naar een lijst woorden die erop wijzen. Schrijf `intent(request, keywords)`: de eerste key van de dict `keywords` met een woord dat in de string `request` voorkomt, en `"unknown"` als geen enkel woord erin voorkomt. De woorden staan in kleine letters, dus zet het verzoek eerst in kleine letters, met `.lower()`. Ga de keys en hun lijsten langs met `for name, words in keywords.items():`, en elke lijst met een tweede `for`-loop. Vervang de `...`. De check roept `intent(REQUESTS[0], KEYWORDS)` aan, daarna de andere verzoeken en eigen dicts.
        """),
        L.task("Ga de soorten en hun woorden langs, en geef de eerste soort terug met een woord in het verzoek."),
    ])
    return


@app.cell(hide_code=True)
def exercise_intent():
    box_intent = L.code_box("intent", lines=7)
    box_intent
    return (box_intent,)


@app.cell(hide_code=True)
def check_intent(box_intent):
    result_intent = L.check("intent", box_intent.value)
    result_intent
    return (result_intent,)


@app.cell(hide_code=True)
def hints_intent():
    L.hints(
        "intent",
        [
            "Lees het verzoek zoals het programma dat doet: één soort tegelijk, en binnen een soort één woord tegelijk.",
            "Wat vraag je van elk woord, en wat geef je terug zodra het antwoord ja is?",
            "`            if ___:`",
        ],
        why="`text = request.lower()`, `for name, words in keywords.items():`, `for word in words:`, dan `if word in text:` met `return name` eronder, en na beide loops `return \"unknown\"`. `intent(REQUESTS[0], KEYWORDS)` geeft `\"hours\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_intent():
    mo.md(r"""
    Zeven van de negen zijn goed, en twee fout: zoek ze op in de afbeelding onder de check.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: beide kanten aan het werk

    Kies een verzoek, en zie de twee kanten na elkaar: de `intent` van de course vindt waar het verzoek over gaat, en een vast antwoord voor elke soort verzoek maakt de reactie: de eenvoudigste vorm van maken.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_request = mo.ui.dropdown(options=C.request_options(L.lang), value=list(C.request_options(L.lang))[0], label="Een verzoek")
    demo_request
    return (demo_request,)


@app.cell(hide_code=True)
def demo_look(demo_request):
    mo.Html(C.demo_picture(demo_request.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_names():
    mo.md(r"""
    ### Oefening 3. Twee projecten die woorden tellen

    Een onderzoeksgroep telt de woorden die kinderen van twee tot vier jaar gebruiken, om uit te zoeken op welke leeftijd ze de verleden tijd gaan gebruiken. Een bedrijf telt de woorden die gebruikers typen, om een spellingapp te bouwen die het goede woord voorstelt.
    """)
    return


@app.cell(hide_code=True)
def choice_names_cell():
    choice_names = L.choice(
        "names",
        question="Wie computationele linguïstiek en NLP uit elkaar houdt: welk van de twee projecten is dan computationele linguïstiek?",
        options={
            "right": "Het onderzoek naar kinderen, omdat het vraagt hoe taal geleerd wordt",
            "app": "De spellingapp, want daar zet men rekenen op taal in voor echte gebruikers",
            "neither": "Geen van beide, want een programma dat telt doet geen linguïstiek",
        },
        correct="right",
        why={
            "right": "De groep wil weten hoe taal werkt: wanneer kinderen de verleden tijd leren. Jurafsky en Martin noemen dat onderzoek naar vragen over taal zelf. De spellingapp is NLP: een systeem dat moet werken.",
            "app": "Een app die werkt is de technische kant: NLP. Computationele linguïstiek stelt, als je die apart houdt, vragen over taal zelf. Zie *Computationele linguïstiek* in *Het idee*.",
            "neither": "Tellen is rekenen, en de vraag over kinderen is een vraag over taal: dat maakt het computationele linguïstiek. Zie *Computationele linguïstiek* in *Het idee*.",
        },
    )
    choice_names
    return (choice_names,)


@app.cell(hide_code=True)
def text_exercise_open():
    mo.md(r"""
    ### Oefening 4. Een slot dat niet open wil

    Voor het verzoek *The lock on my bike won't open.* (het slot van mijn fiets gaat niet open) geeft de `intent` van de course `"hours"`.
    """)
    return


@app.cell(hide_code=True)
def choice_open_cell():
    choice_open = L.choice(
        "open",
        question="Waarom gaat dit verzoek mis?",
        options={
            "list": "Bij de reparatiewoorden ontbreekt lock, en dat toevoegen lost elk verzoek als dit voorgoed op",
            "right": "Open betekent hier iets anders, en een trefwoord ziet het verschil niet",
            "lower": "Het verzoek klein maken veranderde wat het betekent",
        },
        correct="right",
        why={
            "right": "Het woord open staat in het verzoek, maar het gaat over een slot, niet over de winkel. Een programma dat alleen naar letters kijkt, ziet dat niet: begrijpen vraagt meer dan woorden vinden.",
            "list": "Lock toevoegen helpt niet: hours staat eerst in de dict, dus open wint nog steeds. En het volgende verzoek over een slot gaat misschien over het kopen van een slot. Zie `C.KEYWORDS`.",
            "lower": "Kleine letters veranderen niet wat een verzoek betekent; zonder kleine letters zou *Open on Sunday?* gemist worden. Het probleem is het woord open, dat hier iets anders betekent.",
        },
    )
    choice_open
    return (choice_open,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl) liet zien hoe een computer tekst opslaat, en [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) maakte tekst door te tellen, zonder er iets van te begrijpen. Deze les gaf het vakgebied een naam: NLP, met twee kanten, begrijpen (NLU) van tekst naar betekenis en maken (NLG) van betekenis naar tekst. Computationele linguïstiek bestudeert taal zelf met computers, en is vaak een andere naam voor hetzelfde vakgebied. Een programma met trefwoorden vond waar een verzoek over ging, en had er twee van de negen fout. De volgende les, *Waarom taal moeilijk is voor computers*, laat zien waar zulke fouten vandaan komen.

    **Een voorbeeld uit het dagelijks leven.** Een goede tolk doet beide kanten: eerst snappen wat er gezegd wordt, dan het in een andere taal opnieuw zeggen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_names, choice_open, result_intent, result_side):
    L.wrap_up(result_side, result_intent, choice_names, choice_open)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
