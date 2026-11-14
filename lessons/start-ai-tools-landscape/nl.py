# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-ai-tools-landscape. Route: course/route/lessen/bouwen.yaml. Checks en de verzonnen chat: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-ai-tools-landscape", "nl")
    C = L.checks   # de verzonnen chat voor het plaatje van een context window


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bakery():
    quick_bakery = L.choice(
        "quick-bakery",
        question="Een assistent schrijft dat de kleine bakker in jouw straat in 1987 is begonnen. Waarom controleer je dat voordat het in de schoolkrant komt?",
        options={
            "likely": "Hij schrijft een waarschijnlijk jaartal, en bij een zeldzaam feit is waarschijnlijk vaak niet waar.",
            "never": "Assistenten hebben jaartallen nooit goed.",
            "purpose": "Assistenten verzinnen jaartallen expres, om zeker te klinken.",
        },
        correct="likely",
        why={
            "likely": "Het model schrijft steeds een waarschijnlijk volgend woord. Weinig teksten noemen jouw bakker, dus een waarschijnlijk jaartal kan makkelijk fout zijn, en het klinkt even zeker.",
            "never": "Bij dingen die veel teksten noemen, heeft een assistent jaartallen meestal goed. Wat is er anders aan dit jaartal? Zie [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).",
            "purpose": "Hij heeft geen eigen bedoeling: hij kiest woord voor woord een waarschijnlijk volgend woord. Waarom kan dat bij deze bakker misgaan? Zie [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).",
        },
    )
    quick_bakery
    return (quick_bakery,)


@app.cell(hide_code=True)
def quick_check_no_check():
    quick_no_check = L.choice(
        "quick-no-check",
        question="Welk deel van het antwoord van een assistent kun je gebruiken zonder het te controleren?",
        options={
            "quote": "Een citaat van een beroemd persoon",
            "title": "Een idee voor een titel dat je zelf kiest",
            "date": "De datum van een gebeurtenis",
        },
        correct="title",
        why={
            "title": "Een idee waar je zelf uit kiest, is geen feit, dus er valt niets te controleren.",
            "quote": "Een citaat is een feit: heeft die persoon het echt gezegd? Stap 2 van het recept in [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl) haalt de feiten eruit.",
            "date": "Een datum is een feit, en precieze getallen zijn juist waar assistenten uitglijden. Stap 2 van het recept in [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl) haalt de feiten eruit.",
        },
    )
    quick_no_check
    return (quick_no_check,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Er zijn honderden AI-tools, en elke maand komen er nieuwe bij. Welke je kiest, bepaalt wat hij voor je kan doen, hoeveel hij in één keer kan lezen, en wie er nog meer krijgt wat jij hem geeft. Namen veranderen snel, maar de soorten tools blijven dezelfde, en de vragen die je bij elke soort stelt ook.

    **Een voorbeeld uit het dagelijks leven.** Je kunt uit eten gaan, een maaltijd laten bezorgen met je boodschappen, of zelf koken. Het restaurant heeft de grootste keuken, maar het personeel ziet wat je bestelt. Thuis is je keuken kleiner, maar niemand ziet wat je kookt. Dat is alleen een vergelijking; hieronder zie je de drie soorten AI-tools waar die voor staat.
    """)
    return


@app.cell(hide_code=True)
def predict_where():
    guess_where = L.guess(
        "where-it-goes",
        question="Je plakt een brief van je dokter in een gratis chatassistent op het web, en vraagt om een samenvatting. Waar is de brief daarna?",
        options={
            "only": "Alleen op je eigen computer",
            "theirs": "Ook op de computers van het bedrijf achter de assistent",
            "gone": "Nergens: hij wordt gewist zodra je het antwoord hebt",
        },
        reveal="**Ook op de computers van het bedrijf.** Het model draait niet op jouw computer: je tekst gaat via internet naar het bedrijf dat het draait, de aanbieder. Daar wordt je chat meestal bewaard. Afhankelijk van de aanbieder en je instellingen lezen mensen van het bedrijf soms delen mee, bijvoorbeeld om misbruik op te sporen, en kan hij gebruikt worden om latere modellen te trainen. Je brief ligt nu dus bij een bedrijf waarvan je de voorwaarden waarschijnlijk nooit hebt gelezen.",
    )
    guess_where
    return (guess_where,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **context window** | hoeveel tekst een model hooguit kan bekijken als het het volgende woord schrijft: jouw berichten, zijn antwoorden en alles wat aan de chat is toegevoegd |
    | **open weights** | een model is gepubliceerd, zodat iedereen het kan downloaden en op eigen computers kan draaien; kleine zelfs op een laptop |
    | **privacyhygiëne** | de gewoonte om na te gaan wat je een tool geeft voordat je het plakt of uploadt, en weg te laten wat niet voor iedereen is |

    ### Drie soorten AI-tools

    | Soort | Voorbeelden in november 2026 | Waar je tekst heen gaat |
    |---|---|---|
    | een **chatassistent** in een browser of een app | ChatGPT, Claude, Gemini, Microsoft Copilot, Le Chat | naar de computers van de aanbieder |
    | **AI in een programma** dat je al gebruikt | schrijfhulp in een mail-app, suggesties in een code-editor, een foto-app die dingen uit een foto weghaalt | meestal naar de computers van de maker, of van een AI-bedrijf waarmee die samenwerkt; een telefoon doet een deel op de telefoon zelf |
    | een **model dat je zelf draait** | modellen met open weights, zoals Qwen, Mistral, Gemma en gpt-oss, met een programma dat ze draait | nergens heen: het blijft op je computer |

    De namen in de middelste kolom zijn de bekende op het moment van schrijven; volgend jaar zijn er een paar nieuw. De rechterkolom is de vraag die blijft.

    ### Wat een assistent kan behalve schrijven

    Veel assistenten kunnen **op het web zoeken**, **bestanden lezen** die je ze geeft (een document, een spreadsheet, een foto), **code draaien** om een som uit te rekenen of een grafiek te tekenen, en dingen **onthouden** uit eerdere chats. Sommige maken plaatjes of praten hardop. Een latere les laat assistenten zien die stap voor stap dingen voor je doen.

    Dat alles geeft het model meer tekst om naar te kijken: de zoekresultaten, het bestand, de uitkomst van de code. Het language model schrijft het antwoord nog steeds woord voor woord. En alles waar het naar kijkt, moet in zijn context window passen.
    """)
    return


@app.cell(hide_code=True)
def text_window():
    mo.md(r"""
    ### Het context window

    Voor elk volgend woord kijkt het model naar de chat tot nu toe. Het kan alleen niet alles tegelijk bekijken: er past maar zoveel in zijn context window. Dat wordt geteld in stukjes van woorden (een latere les noemt ze tokens en laat zien hoe tekst wordt opgeknipt). Wordt een chat langer dan het window, dan past het begin er niet meer in. Afhankelijk van de tool laat hij dan de oudste berichten vallen, zoals in het plaatje hieronder, maakt hij er een korte samenvatting van, of vraagt hij je een nieuwe chat te beginnen. Hoe dan ook: het model ziet het begin niet meer woord voor woord.

    Echte assistenten kunnen veel meer kwijt dan het plaatje: van zo'n tienduizend stukjes tot een miljoen of meer, afhankelijk van de tool en van wat je betaalt. Een lange chat waarin je lange teksten plakt, kan nog steeds te veel zijn. En ook binnen het window wordt een detail in een heel lange tekst vaker gemist dan in een korte.
    """)
    return


@app.cell(hide_code=True)
def picture_window():
    window_story = L.explore(
        "window-story", "berichten in de chat", lambda n: C.window_frame(n, L.lang), start=8, low=1, high=8, step=1, role="",
        task="Een assistent schrijft samen met jou een verhaal. Schuif terug vanaf 8 berichten: vanaf welk bericht ziet het model niet meer dat Noor bang is voor water?")
    mo.vstack([
        window_story,
        L.watch_out("Een context window is geen venster op je scherm, zoals het chatvenster: het is hoeveel van de chat het model in één keer kan bekijken."),
        mo.md(r"""
        Elke rij is een bericht, met zijn lengte in stukjes van woorden; de lengtes en het window van 1.000 stukjes zijn verzonnen, zodat het in het plaatje past. De **haak** wijst de nieuwste berichten aan die samen in het context window passen; de grijze zijn eruit gevallen. Vanaf bericht 6 ziet het model bericht 1 niet meer, en in hoofdstuk 4 laat het Noor de zee in duiken.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_open_weights():
    mo.md(r"""
    ### Open weights

    De grootste assistenten draaien op rijen speciale computers in een datacenter: je krijgt alleen hun antwoorden, nooit het model zelf. Sommige bedrijven publiceren hun modellen juist wel. Een model met **open weights** kan iedereen downloaden (een latere les laat zien wat de getallen erin zijn) en op eigen computers draaien, zonder internet; kleine draaien zelfs op een laptop. Dan verlaat niets wat je typt je computer. Het nadeel: een model dat klein genoeg is voor een laptop, weet minder en maakt meer fouten dan de grootste assistenten. Later in de course draai je er zelf een.

    Open weights is niet hetzelfde als open source: het bedrijf houdt de teksten waarop het trainde en zijn eigen code vaak voor zich, en zijn licentie kan nog voorwaarden stellen.
    """)
    return


@app.cell(hide_code=True)
def picture_where():
    mo.vstack([
        draw.flow(
            [("Jouw computer", ["je typt of plakt"]), ("Computers van de aanbieder", ["het model schrijft het antwoord", "je chat wordt bewaard"])],
            ["je tekst, via\ninternet"], ["het antwoord"], title="Een chatassistent op het web"),
        draw.flow(
            [("Jouw computer", ["je typt of plakt", "een model met open weights schrijft het antwoord", "niets verlaat je computer"])],
            [], [], title="Een model dat je zelf draait"),
        mo.md(r"""
        Bij een chatassistent, en bij de meeste AI in andere programma's, reist je tekst naar de aanbieder; bij een model op je eigen computer blijft hij waar hij is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een chat wissen in de app wist hem niet altijd meteen op de computers van de aanbieder: veel aanbieders bewaren hem nog een tijd, bijvoorbeeld voor veiligheidscontroles. Het moment dat telt, is het moment dat je plakt."),
        mo.md(r"""
        ### Recept: denk na voordat je plakt

        1. **Gaat het je computer uit?** Bij een chatassistent en bij de meeste AI in programma's: ja. Bij een model dat je zelf draait: nee, en dan ben je klaar.
        2. **Wat staat erin dat niet voor iedereen is?** Wachtwoorden en toegangscodes; gegevens over jezelf, zoals je adres, je bankrekening of je gezondheid; gegevens over andere mensen; en wat van je school of werk is.
        3. **Laat het weg of vervang het.** Schrijf *[naam]* in plaats van een naam, gebruik verzonnen getallen, schrijf *een collega*. Een wachtwoord gaat nooit een chat in.
        4. **Materiaal van school of werk** gaat alleen in een tool die je school of werkgever heeft goedgekeurd, volgens hun regels.
        5. **Kijk één keer naar de instellingen**: gebruikt de aanbieder je chats om latere modellen te trainen? Bij veel tools kun je dat uitzetten.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je energierekening is veel te hoog, en je wilt hulp bij een klachtbrief. Op de rekening staan je naam, adres, klantnummer en rekeningnummer, en het verbruik per maand met de bedragen. Je wilt een gratis chatassistent op het web gebruiken.

    1. **Gaat het je computer uit?** Ja: het is een chatassistent op het web. Ga door.
    2. **Wat staat erin dat niet voor iedereen is?** Je naam, adres, klantnummer en rekeningnummer. Het verbruik en de bedragen zeggen niets over jou als persoon, en de brief heeft ze nodig.
    3. **Laat het weg of vervang het.** Je typt het verbruik en de bedragen, en schrijft *[naam]* en *[klantnummer]* waar die horen. De assistent schrijft de brief; je eigen gegevens vul je daarna zelf in, in je eigen tekstverwerker.
    4. **Materiaal van school of werk?** Nee, het is je eigen rekening. Stap 4 telt hier niet.
    5. **Kijk naar de instellingen.** Je kijkt of je chats gebruikt worden om te trainen, en zet dat uit.

    En de bedragen in de brief die terugkomt? Dat zijn feiten waar de klacht van afhangt: die controleer je met je rekening erbij, met het recept van [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_paste():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Voordat je plakt

        Je wilt bij elk van de zes dingen hieronder hulp van een gratis chatassistent op het web. Stap 1 van het recept is al gedaan: alle zes zouden je computer uitgaan. Doe nu stap 2 tot en met 4.
        """),
        L.task("Zet elk van de zes in de goede groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paste():
    sort_paste = L.sort("before-you-paste")
    sort_paste
    return (sort_paste,)


@app.cell(hide_code=True)
def check_before_you_paste(sort_paste):
    result_paste = L.check("before-you-paste", sort_paste)
    result_paste
    return (result_paste,)


@app.cell(hide_code=True)
def hints_paste():
    L.hints(
        "before-you-paste",
        [
            "Stap 2 van het recept: wat staat erin dat niet voor iedereen is? Denk aan wachtwoorden, gegevens over mensen, en wat van je werk is.",
            "Iets wat iedereen al mag lezen, zoals een recept uit een boek of een nieuwsbericht op internet, kan erin zoals het is.",
            "Vraag je bij elk af: wie kan er last van krijgen als een vreemde het leest? Bij drie is dat zo.",
        ],
        why="Plak zoals het is: **Pannenkoeken**, **Nieuws** en **Wikipedia**. Niet zoals het is: **Wachtwoord** (nooit in een chat), **Brief** (laat de naam weg, en de andere gegevens die naar je buur wijzen) en **Klanten** (die blijft op je werk, of gaat in een tool die je werkgever heeft goedgekeurd).",
    )
    return


@app.cell(hide_code=True)
def text_which_tool():
    mo.md(r"""
    ### Oefening 2. Welke soort tool?
    """)
    return


@app.cell(hide_code=True)
def choice_which_tool_cell():
    choice_which_tool = L.choice(
        "which-tool",
        question="Een ziekenhuis wil patiëntdossiers laten samenvatten door een AI-model, en de dossiers mogen de eigen computers van het ziekenhuis niet uit. Welke soort tool past?",
        options={
            "own": "Een model met open weights dat op de eigen computers van het ziekenhuis draait",
            "chat": "Een chatassistent op het web, met trainen op chats uitgezet",
            "inside": "De AI in de mail-app die het ziekenhuis al gebruikt",
        },
        correct="own",
        why={
            "own": "Alleen een model dat op de eigen computers van het ziekenhuis draait, houdt de dossiers daar. Dat het misschien minder weet dan de grootste assistenten, is de prijs.",
            "chat": "Trainen uitzetten houdt de dossiers niet in het ziekenhuis: ze reizen nog steeds naar de computers van de aanbieder. Kijk naar de rechterkolom van de tabel met de drie soorten.",
            "inside": "AI in een programma stuurt de tekst meestal ook naar de computers van de maker, of van een AI-bedrijf waarmee die samenwerkt. Kijk naar de rechterkolom van de tabel met de drie soorten.",
        },
    )
    choice_which_tool
    return (choice_which_tool,)


@app.cell(hide_code=True)
def text_long_chat():
    mo.md(r"""
    ### Oefening 3. Een lange chat
    """)
    return


@app.cell(hide_code=True)
def choice_long_chat_cell():
    choice_long_chat = L.choice(
        "long-chat",
        question="Je plant een reis in één lange chat. In je eerste bericht schreef je dat je geen vlees eet. Sindsdien heb je drie lange reisgidsen in de chat geplakt. Nu stelt de assistent een steakhouse voor. Wat is de waarschijnlijkste reden?",
        options={
            "latest": "Een assistent leest alleen je laatste bericht, nooit de berichten ervoor.",
            "window": "De reisgidsen hebben je eerste bericht uit zijn context window geduwd.",
            "mind": "Hij besloot dat je van gedachten bent veranderd, omdat je het niet nog eens zei.",
        },
        correct="window",
        why={
            "window": "De lange reisgidsen vulden het context window, en je eerste bericht viel eruit. Herhaal wat ertoe doet, of begin een nieuwe chat en zet het in je eerste bericht.",
            "latest": "Hij leest alles van de chat wat in zijn context window past, niet alleen het laatste bericht. Waardoor zou je eerste bericht er niet meer in passen?",
            "mind": "Als je eerste bericht nog in zijn context window stond, zag het model het bij elk volgend woord, en niets in de chat zegt dat je van gedachten bent veranderd. Waardoor zou het eerste bericht verdwijnen?",
        },
    )
    choice_long_chat
    return (choice_long_chat,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl) leerde je controleren wat een assistent zegt. Nu weet je ook wat je hem geeft: welke soort tool je tekst waarheen stuurt, hoeveel van een chat het model in één keer ziet, en wat je weglaat voordat je plakt. Verderop in deze module leer je een assistent goed iets te vragen, en later in de course draai je een model met open weights op je eigen computer en zie je een context window van binnen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long_chat, choice_which_tool, result_paste):
    L.wrap_up(result_paste, choice_which_tool, choice_long_chat)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
