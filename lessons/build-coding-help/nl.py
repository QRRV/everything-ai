# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-coding-help. Route: course/route/lessen/bouwen.yaml. Checks en het plaatje: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-coding-help", "nl")
    C = L.checks   # het plaatje van de drie soorten hulp, en de twee sorteeroefeningen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl), en één over [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_centaur():
    quick_centaur = L.choice(
        "quick-centaur",
        question="Je geeft een assistent een heel rapport om te vertalen, en kijkt de vertaling na als hij klaar is. Welke manier van werken is dat?",
        options={
            "cyborg": "Cyborg",
            "frontier": "De jagged frontier",
            "centaur": "Centaur",
        },
        correct="centaur",
        why={
            "centaur": "Een heel deel gaat naar de assistent en komt terug om nagekeken te worden: een centaur. Een cyborg wisselt stap voor stap af met de assistent.",
            "cyborg": "Een cyborg gaat vaak heen en weer. Hier gaat het werk één keer heen en komt het één keer terug. Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
            "frontier": "De jagged frontier is de grens tussen de taken die een model goed en slecht doet, geen manier van werken. Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
        },
    )
    quick_centaur
    return (quick_centaur,)


@app.cell(hide_code=True)
def quick_check_agents_md():
    quick_agents_md = L.choice(
        "quick-agents-md",
        question="Waarom staat er een bestand AGENTS.md in de map van de course?",
        options={
            "memory": "De assistent schrijft erin op wat hij over jou heeft geleerd",
            "instructions": "Het vertelt een AI-assistent hoe de course in elkaar zit en hoe hij je helpt zonder antwoorden weg te geven",
            "list": "Het somt de AI-agents op die de course gebruikt",
        },
        correct="instructions",
        why={
            "instructions": "Een assistent die in de map werkt, leest het, zodat hij de course kent en weet hoe hij helpt. Hij volgt het niet altijd, dus je kijkt nog steeds na wat hij zegt.",
            "memory": "De course schreef AGENTS.md, niet de assistent. Wat staat erin? Zie [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl).",
            "list": "Het is bedoeld voor elke assistent die je op de course loslaat. Wat vraagt het van hem? Zie [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl).",
        },
    )
    quick_agents_md
    return (quick_agents_md,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Code schrijven is een van de dingen waar AI-tools het meest voor gebruikt worden. De hulp is er in heel verschillende maten: een tool die de rest van de regel raadt die je typt, een chat waaraan je vragen stelt over je code, of een agent die zelf tien bestanden verandert en ze draait. Hoe meer een tool zelf doet, hoe minder je van elke stap ziet. Dus hangt er meer af van wat je achteraf nakijkt, en van wat je hem laat aanraken.

    **Een voorbeeld uit het dagelijks leven.** Een brief schrijven. Je telefoon stelt het volgende woord voor, en je tikt het aan of typt door: je ziet elk woord. Een vriend leest je brief en zegt wat beter kan: jij beslist wat je verandert. Of een assistent schrijft de brief en post hem: snel, maar je ziet hem alleen als je vraagt hem te lezen voordat hij weggaat. Dat is alleen een vergelijking; hieronder zie je de drie soorten hulp bij code.
    """)
    return


@app.cell(hide_code=True)
def predict_trust():
    guess_trust = L.guess(
        "trust",
        question="In zijn enquête van 2025 vroeg Stack Overflow developers naar AI-tools. Van de ruim 33.000 die antwoordden, gebruikte 84% ze of was dat van plan. Hoeveel vertrouwden erop dat wat de tools geven klopt?",
        options={
            "most": "De meesten, ongeveer 70%",
            "third": "Ongeveer een derde",
            "few": "Bijna niemand, ongeveer 3%",
        },
        reveal="**Ongeveer een derde: 33%.** Meer van hen wantrouwden het: 46%. Maar 3% vertrouwde het sterk. De grootste ergernis, genoemd door 66%: antwoorden die *bijna goed zijn, maar net niet*. Developers gebruiken de tools veel, en kijken na wat eruit komt. Over dat nakijken gaat deze les.",
    )
    guess_trust
    return (guess_trust,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **code completion** | terwijl je typt, stelt de tool de rest van de regel of de volgende paar regels voor, vaak in grijs; met één toets, vaak Tab, neem je het voorstel over |
    | **coding agent** | een assistent die in je projectmap werkt: hij leest bestanden, verandert ze en voert commando's uit, zoals de tests, en gaat zelf door tot hij denkt dat de taak af is |
    | **tests** | kleine programma's die je code op gevallen uitvoeren en zeggen of hij geeft wat hij moet geven; een latere les laat zien hoe je ze schrijft |
    | **permissies** | wat je een agent laat doen zonder dat hij het jou eerst vraagt |

    ### Drie soorten hulp

    | Soort | Wat het ziet | Wat het doet | Wat jij nakijkt |
    |---|---|---|---|
    | **code completion** | het bestand waarin je zit, rond waar je typt | stelt de volgende regel of regels voor | elk voorstel, voordat je het overneemt |
    | **chat** over code | wat je erin plakt of selecteert, en je vraag | legt uit, vindt een fout, schrijft een stuk code | het antwoord, voordat je het gebruikt |
    | **coding agent** | de hele projectmap | leest, verandert en draait, in veel stappen | wat hij veranderde, en of de tests echt slagen |

    Achter alle drie zit een language model. Veel tools doen alle drie: editors zoals Cursor, of Visual Studio Code met GitHub Copilot, vullen aan, chatten en hebben een agent; Claude Code, Codex en Aider zijn agents die je in een terminal start, een venster waarin je commando's typt, in de projectmap (voorbeelden in november 2026).
    """)
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.Html(C.ways_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_setup():
    mo.vstack([
        mo.md(r"""
        ### Een tool instellen

        Een agent werkt op jouw computer, met jouw bestanden. Drie dingen bepalen wat hij kan doen:

        - **Waar je hem start.** Hij ziet de map waarin je hem start, en alles daaronder. Start hem in de projectmap, niet in je persoonlijke map.
        - **Wat hij mag zonder te vragen.** De meeste agents vragen het vóór elk commando of elke verandering, en laten je sommige dingen voorgoed toestaan: de permissies. Sta toe wat makkelijk terug te draaien is en niets buiten het project verandert, zoals het project lezen en de tests draaien. Laat hem eerst vragen bij wat moeilijk terug te draaien is, of wat van internet binnenkomt of ernaartoe gaat.
        - **Wat hij van je project weet.** Een bestand zoals AGENTS.md, dat je kent uit [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl), vertelt hem hoe het project in elkaar zit en wat je wilt. De volgende les gaat over die context.

        Houd wachtwoorden en keys uit de map waarin hij werkt: wat een agent leest, gaat naar het model, en bij een model online naar de aanbieder, net als alles wat je in een chat plakt ([Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl)). Sommige agents kunnen ook in een *sandbox* draaien: een afgesloten ruimte waar een fout de rest van je computer niet kan bereiken.
        """),
        L.watch_out("Een agent die *klaar, alle tests slagen* schrijft, heeft ze niet altijd gedraaid, en een test die hij zelf schreef, kan het verkeerde testen. Wat hij over zijn eigen werk zegt, is tekst van een language model, net als elk ander antwoord: je kijkt zelf naar de veranderingen en de testresultaten."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: hulp bij code kiezen en instellen

    1. **Hoeveel wil je zien?** Elke regel: code completion. Elk antwoord: een chat. Alleen het resultaat: een agent, voor werk dat groter is dan een paar regels.
    2. **Start een agent in de projectmap,** met een kopie van de map waar je naar terug kunt.
    3. **Sta toe wat makkelijk terug te draaien is,** zoals het project lezen en de tests draaien. Laat hem eerst vragen voordat hij iets weggooit, installeert of verstuurt.
    4. **Houd wachtwoorden en keys** uit de map.
    5. **Lees wat hij veranderde, en draai de tests zelf,** voordat je het houdt.

    In deze course zijn de oefeningen om zelf te schrijven: hulp die het antwoord schrijft, neemt het leren weg, zoals AGENTS.md zegt. Daar past een chat die uitlegt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je kleine programma van vier bestanden leest een lijst boeken en drukt de boeken af die je nog niet gelezen hebt. Je wilt een optie die ze op jaar sorteert, met een test.

    1. **Hoeveel zien:** meer dan een paar regels, in meer dan één bestand, met tests om te draaien: een agent past.
    2. **Map:** je kopieert de map *books* naar *books-copy*, en start de agent in *books*.
    3. **Permissies:** hij mag de vier bestanden lezen en de tests zelf draaien; hij vraagt het voordat hij iets installeert.
    4. **Geheimen:** de map heeft er geen; het programma leest alleen *books.csv*.
    5. **Nakijken:** de agent schrijft dat hij twee bestanden veranderde, een test toevoegde, en dat alle tests slagen. Je leest de twee veranderingen en draait de tests zelf: ze slagen. Maar hij gaf ook een functie een nieuwe naam, waar je niet om vroeg. Je vraagt hem dat terug te draaien, en kijkt opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_which_help():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke soort hulp?

    Zet elke taak bij de soort hulp die er het best bij past.
    """)
    return


@app.cell(hide_code=True)
def exercise_which_help():
    sort_help = L.sort("which-help")
    sort_help
    return (sort_help,)


@app.cell(hide_code=True)
def check_which_help(sort_help):
    result_help = L.check("which-help", sort_help)
    result_help
    return (result_help,)


@app.cell(hide_code=True)
def hints_help():
    L.hints(
        "which-help",
        [
            "Kijk naar de laatste kolom van de tabel: wat jij nakijkt. Hoeveel van het werk wil je zien?",
            "Een regel of twee waar je typt: completion. Een vraag waarvan je het antwoord zelf gebruikt: een chat. Veel bestanden en tests om te draaien: een agent.",
            "Twee van de zes zijn code completion, twee een chat en twee een agent.",
        ],
        why="Code completion: **De volgende regel** en **Een laatste regel**. Chat: **Een foutmelding** en **Twee manieren**. Coding agent: **Twaalf bestanden** en **Een nieuwe optie**.",
    )
    return


@app.cell(hide_code=True)
def text_permissions():
    mo.md(r"""
    ### Oefening 2. Wat mag een agent zelf?

    Je stelt een coding agent in voor een eigen project. Welke van deze dingen mag hij zelf doen, en bij welke vraagt hij het jou eerst?
    """)
    return


@app.cell(hide_code=True)
def exercise_permissions():
    sort_permissions = L.sort("permissions")
    sort_permissions
    return (sort_permissions,)


@app.cell(hide_code=True)
def check_permissions(sort_permissions):
    result_permissions = L.check("permissions", sort_permissions)
    result_permissions
    return (result_permissions,)


@app.cell(hide_code=True)
def hints_permissions():
    L.hints(
        "permissions",
        [
            "Vraag bij elk: kan ik het terugdraaien, en komt er iets van internet binnen of gaat er iets naartoe?",
            "Lezen en kijken veranderen niets. Weggooien, installeren en online zetten wel.",
            "Drie van de zes doet hij zelf, en bij drie vraagt hij het jou eerst.",
        ],
        why="Zelf: **Het project lezen**, **De tests draaien** en **In de code zoeken**: er verandert niets wat je niet kunt terugdraaien. Vraagt jou eerst: **Een package installeren** (code van onbekenden die op je computer draait), **Een map weggooien** (die kan voorgoed weg zijn) en **Online zetten** (wat online staat, blijft online).",
    )
    return


@app.cell(hide_code=True)
def text_after_done():
    mo.md(r"""
    ### Oefening 3. De agent zegt dat hij klaar is
    """)
    return


@app.cell(hide_code=True)
def choice_after_done_cell():
    choice_after_done = L.choice(
        "after-done",
        question="De agent schrijft: *Klaar. Ik heb drie bestanden veranderd, en alle tests slagen.* Wat doe je voordat je de veranderingen houdt?",
        options={
            "ask": "De agent vragen of hij het zeker weet",
            "look": "De drie veranderingen lezen en de tests zelf draaien",
            "keep": "Niets: de tests slagen, dus het werkt",
        },
        correct="look",
        why={
            "look": "Wat de agent over zijn werk zegt, is tekst van een language model, die net als elk ander antwoord fout kan zijn. De veranderingen en de testresultaten die je zelf ziet, zijn dat niet.",
            "keep": "Heb jij de tests zien slagen? Een agent kan zeggen dat ze slagen zonder ze te draaien, of een test veranderen tot hij slaagt. Waar kun je zelf naar kijken?",
            "ask": "Zijn antwoord daarop is weer tekst van hetzelfde model, dat makkelijk ja zegt. Waar kun je zelf naar kijken?",
        },
    )
    choice_after_done
    return (choice_after_done,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl) zag je twee manieren om met AI te werken: code completion is cyborgwerk, regel voor regel; een agent aan wie je een hele taak geeft, is centaurwerk, en je kijkt het resultaat na. In [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl) liet je een assistent met AGENTS.md op de course los. De volgende les, [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl), laat zien hoe je een agent een taak geeft zodat hij doet wat je bedoelde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_after_done, result_help, result_permissions):
    L.wrap_up(result_help, result_permissions, choice_after_done)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
