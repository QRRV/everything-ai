# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-coding-workflow. Route: course/route/lessen/bouwen.yaml. Checks, de 32 programma's en hun plaatje: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-coding-workflow", "nl")
    C = L.checks   # de 32 programma's die Qwen3 op vier manieren van vragen schreef, wat ze geven, en het plaatje


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Je wilt een nieuwe optie in je programma, verdeeld over drie bestanden, met een test, terwijl jij iets anders doet. Welke soort hulp past?",
        options={
            "agent": "Een coding agent",
            "completion": "Code completion",
            "chat": "Een chat over je code",
        },
        correct="agent",
        why={
            "agent": "Werk in meer bestanden, met tests om te draaien, zonder dat jij elke stap doet: daar is een coding agent voor. Deze les gaat over hoe je hem zo'n taak geeft.",
            "completion": "Code completion stelt een regel voor waar je typt; de bestanden en de test blijven jouw werk. Zie [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).",
            "chat": "Een chat geeft antwoord, en het werk doe je zelf. Zie [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_ask():
    quick_ask = L.choice(
        "quick-ask",
        question="Over welke van deze moet een coding agent het jou vragen voordat hij het doet?",
        options={
            "tests": "De tests draaien",
            "install": "Een package van internet installeren",
            "search": "In de code van het project zoeken",
        },
        correct="install",
        why={
            "install": "Dan zou code van onbekenden op je computer draaien, en dat is moeilijk terug te draaien. De tests draaien en zoeken veranderen niets wat je niet kunt terugdraaien.",
            "tests": "De tests draaien verandert niets wat je niet kunt terugdraaien, en de agent heeft het nodig om zijn eigen werk na te kijken. Zie [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).",
            "search": "Zoeken verandert helemaal niets. Zie [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).",
        },
    )
    quick_ask
    return (quick_ask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een coding agent doet wat je vraagt, en vult wat je niet vroeg in met gokken. Bij een kleine taak zie je die gokken meteen; bij een grote taak zitten ze in bestanden die je nooit opent. Hoe je de taak overdraagt, bepaalt hoeveel hij moet gokken, en hoe snel je merkt waar hij verkeerd gokte.

    **Een voorbeeld uit het dagelijks leven.** Je laat een aannemer je keuken opnieuw doen. Met *maak hem mooier* krijg je een keuken, maar niet die je bedoelde. Een tekening met maten brengt je dichterbij. Samen het plan doorlopen voordat er een muur uit gaat, brengt een vergeten stopcontact aan het licht terwijl dat nog niets kost. En de aantekeningen over het huis, waar de waterleidingen lopen, besparen de aannemer het gokken. Dat is alleen een vergelijking; hieronder zie je wat Qwen3 deed toen het op vier manieren om hetzelfde kleine programma werd gevraagd.
    """)
    return


@app.cell(hide_code=True)
def predict_examples():
    guess_examples = L.guess(
        "spec-examples",
        question="De course gaf Qwen3 een precieze beschrijving met drie voorbeelden van wat eruit moet komen, en liet het het programma acht keer schrijven. Hoeveel van de acht programma's hadden alle drie die voorbeelden goed?",
        options={
            "half": "Ongeveer de helft",
            "none": "Geen enkel",
            "all": "Alle acht",
        },
        reveal="**Geen enkel.** Zeven van de acht hadden het eerste voorbeeld goed en alle acht het derde, maar niet één het tweede: *'+31 6 1234 5678' geeft '0612345678'*. Dat voorbeeld stond gewoon in de beschrijving. Hieronder zie je wat het meteen had gevangen.",
    )
    guess_examples
    return (guess_examples,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **spec** | kort voor specificatie: wat een programma moet doen, opgeschreven vóór de code, met voorbeelden, de randgevallen ook |
    | **spec-driven development** | werken met een coding agent vanuit een spec: eerst opschrijven wat eruit moet komen, en de agent daarna laten bouwen en testen tegen die spec |
    | **plan** | de stappen die een agent van plan is te zetten, opgeschreven voordat hij iets verandert, zodat jij ze kunt lezen en verbeteren |
    | **context** | alles wat een model voor zich heeft als het antwoordt: je verzoek, de bestanden die het las, en notities zoals AGENTS.md |

    ### Vier delen van een taak

    | Deel | Wat het zegt | Waarom het helpt |
    |---|---|---|
    | de spec | *wat* eruit moet komen, met voorbeelden | een voorbeeld kun je niet op twee manieren lezen, en je kunt het proberen |
    | het plan | *hoe*, in stappen, in een volgorde | je leest het voordat er code is, terwijl verbeteren nog niets kost |
    | kleine stappen | één verandering tegelijk, met na elke stap de tests | een fout komt boven in de stap die hem maakte |
    | de context | wat voor elke taak geldt, zoals hoe je de tests draait | de agent begint er elke taak mee, en jij hoeft het niet te herhalen |

    Veel agents lezen een bestand met de naam AGENTS.md vóór je verzoek; Claude Code leest CLAUDE.md, dat naar AGENTS.md kan verwijzen. Voorbeelden in november 2026: Kiro en Spec Kit van GitHub bouwen de hele manier van werken rond een spec, met een lijst taken die een voor een worden afgevinkt; Claude Code en Cursor hebben een plan mode, waarin de agent alleen een plan maakt en niets verandert totdat jij akkoord bent.

    **De run.** De course vroeg Qwen3-0.6B op vier manieren om dezelfde kleine functie, acht keer per manier: een functie die een Nederlands telefoonnummer opschoont. De verzoeken waren in het Engels; hier staan ze in het Nederlands.

    - **Vaag verzoek:** *Schrijf een Python-functie clean_phone(text) die een telefoonnummer opschoont.*
    - **Spec met voorbeelden:** *Schrijf een Python-functie clean_phone(text) voor Nederlandse telefoonnummers. Houd alleen de cijfers; een nummer dat met +31 of 0031 begint, begint in plaats daarvan met 0; geef de 10 cijfers terug als string, of een lege string als er geen 10 cijfers over zijn. Voorbeelden: '06-12345678' geeft '0612345678', '+31 6 1234 5678' geeft '0612345678', '12345' geeft ''.*
    - **Spec, eerst een plan:** dezelfde spec, die eindigt met *Schrijf eerst een kort plan in genummerde stappen, en dan de code.*
    - **Projectnotities, vaag verzoek:** het vage verzoek, met dezelfde regels en voorbeelden ervoor als notities over het project, zoals een agent AGENTS.md vóór je verzoek leest.

    Een *string* is een stuk tekst; een lege string, '', is tekst zonder iets erin. De course probeerde elk programma op negen nummers, waarvan drie de eigen voorbeelden uit de spec, zonder er een regel aan te veranderen:
    """)
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.vstack([
        mo.Html(C.ways_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_code():
    L.explore("program-code", "programma", lambda _v: C.program_frame(int(_v), L.lang), start=20, low=1, high=len(C.OUTPUTS), step=1, role="",
              task="Schuif door de 32 programma's: acht van het vage verzoek, dan acht van de spec, acht met eerst een plan, en acht met projectnotities. Lees bij de programma's met een plan ook het plan: heeft het een stap voor +31?")
    return


@app.cell(hide_code=True)
def text_plans():
    mo.md(C.plans_after(L.lang))
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("Een grote coding agent schrijft deze functie veel vaker goed dan Qwen3-0.6B. Maar een grote taak laat veel meer te gokken over, en een agent kan ook zeggen dat de tests slagen zonder dat hij ze heeft gedraaid. Dus kijk je zelf wat de tests zeggen, zoals hier elk programma werd gedraaid in plaats van geloofd.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een taak voor een coding agent

    1. **Laat hem eerst kijken.** Vraag hem de bestanden te lezen die ertoe doen en te vertellen wat hij vond, zonder iets te veranderen.
    2. **Schrijf de spec:** wat eruit moet komen, met voorbeelden, de randgevallen ook, zoals te kort, te lang en de vreemde gevallen.
    3. **Vraag om een plan, en lees het** voordat er code wordt geschreven. Ontbreekt er een stap, of staat er een in de verkeerde volgorde? Zeg het nu.
    4. **Werk in kleine stappen.** Na elke stap draaien de tests, de voorbeelden uit de spec ook, en kijk jij wat er veranderde.
    5. **Zet wat altijd geldt in AGENTS.md,** zoals hoe je de tests draait en wat hij niet mag aanraken, zodat de volgende taak ermee begint.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je laat een coding agent de telefoonnummers in de ledenlijst van je club netjes maken: sommige staan er met streepjes, sommige met +31.

    1. **Eerst kijken:** *Lees het project en vertel me waar telefoonnummers worden opgeslagen en gebruikt. Verander nog niets.* Hij vindt ze in twee bestanden: de ledenlijst, en de pagina die hem laat zien.
    2. **Spec:** je schrijft de regels uit de run hierboven op, met vijf voorbeelden: de drie uit de spec, plus '0031612345678' geeft '0612345678' en '06 1234 567' geeft '', omdat het te kort is.
    3. **Plan:** zijn plan heeft drie stappen: alles weghalen wat geen cijfer is, kijken of er 10 cijfers zijn, opslaan. Je ziet dat +31 ontbreekt: na de eerste stap wordt '+31 6 1234 5678' '31612345678', 11 cijfers, en dat nummer gaat weg. Je vraagt om een stap ervoor: maak eerst van +31 of 0031 een 0.
    4. **Kleine stappen:** de agent maakt van je vijf voorbeelden tests en draait ze na elke stap. Je leest zelf wat de tests zeggen, niet alleen wat de agent erover zegt.
    5. **AGENTS.md:** je voegt één regel toe, *Telefoonnummers worden opgeslagen als 10 cijfers, beginnend met 0.* De volgende taak die aan telefoonnummers komt, begint daarmee.
    """)
    return


@app.cell(hide_code=True)
def text_spec_plan_context():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Spec, plan of AGENTS.md?

    Je geeft een agent de taak om de telefoonnummers van je club op te schonen. Waar hoort elke zin?
    """)
    return


@app.cell(hide_code=True)
def exercise_spec_plan_context():
    sort_parts = L.sort("spec-plan-context")
    sort_parts
    return (sort_parts,)


@app.cell(hide_code=True)
def check_spec_plan_context(sort_parts):
    result_parts = L.check("spec-plan-context", sort_parts)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "spec-plan-context",
        [
            "Vraag bij elke zin: zegt hij wat eruit moet komen, hoe je daar komt, of wat in dit project geldt, wat de taak ook is?",
            "Een spec gaat over het resultaat van deze ene taak, een plan is stappen in een volgorde, en AGENTS.md bevat wat voor elke taak waar is.",
            "Twee zinnen per groep. *Een voorbeeld* en *Te kort* zeggen wat eruit moet komen; *Eerst* en *Daarna* zijn stappen.",
        ],
        why="Spec: **Een voorbeeld** en **Te kort**. Plan: **Eerst** en **Daarna**. AGENTS.md: **Tests** en **De map data**.",
    )
    return


@app.cell(hide_code=True)
def text_which_helped():
    mo.md(r"""
    ### Oefening 2. Wat hielp het meest?
    """)
    return


@app.cell(hide_code=True)
def choice_which_helped_cell():
    choice_which_helped = L.choice(
        "which-helped",
        question="Welke verandering aan het vage verzoek hielp Qwen3 het meest?",
        options={
            "notes": "De projectnotities vóór het verzoek",
            "spec": "De spec met voorbeelden",
            "plan": "Eerst het plan, bovenop de spec",
        },
        correct="spec",
        why={
            "spec": "Van 11 naar 39 van de 72 nummers. Wat een model moet gokken, gokt het vaak verkeerd, en een spec met voorbeelden laat veel minder te gokken over. Het plan erbovenop voegde hier niets toe: 38.",
            "notes": "De notities hielpen wel: 11 werd 21. Vergelijk dat met de totalen onder de andere kolommen.",
            "plan": "Kijk naar de totalen onderaan het plaatje: *Spec, eerst een plan* tegen *Spec met voorbeelden*. Wat voegde het plan toe?",
        },
    )
    choice_which_helped
    return (choice_which_helped,)


@app.cell(hide_code=True)
def text_plus_31():
    mo.md(r"""
    ### Oefening 3. Wat had het gevangen?
    """)
    return


@app.cell(hide_code=True)
def choice_plus_31_cell():
    choice_plus_31 = L.choice(
        "plus-31",
        question="Geen van de 32 programma's maakte van +31 een 0, ook niet de 24 die de spec kregen, met zijn voorbeeld. Wat had dat meteen gevangen?",
        options={
            "tests": "De voorbeelden uit de spec als tests, gedraaid na elke stap",
            "longer": "Een langere spec, met de regel over +31 er twee keer in",
            "ask": "De agent vragen of zijn code klopt",
        },
        correct="tests",
        why={
            "tests": "Een test van het voorbeeld *'+31 6 1234 5678' geeft '0612345678'* faalt bij alle 32 programma's, de eerste keer dat hij draait. Na elke stap gedraaid, vangen de tests ook een stap die kapotmaakt wat al werkte.",
            "longer": "De regel stond al in de spec, met een voorbeeld, en alle 24 programma's die hem kregen, misten hem. Twee keer zeggen is nog steeds alleen zeggen: wat zou je *laten zien* dat het misging?",
            "ask": "In tien antwoorden staat het voorbeeld al onder de code, alsof de code het geeft. Geen van die programma's doet dat. Wat een model over zijn code zegt, is geen controle.",
        },
    )
    choice_plus_31
    return (choice_plus_31,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl) koos je de soort hulp en stelde je een agent in; hier gaf je hem een taak: een spec om tegen na te kijken, een plan om te lezen, kleine stappen, en AGENTS.md voor wat altijd geldt. Je voorbeelden helpen pas als ze draaien: verderop in deze module laat [Code nakijken die AI schreef](/course/open/build-check-ai-code?lang=nl) zien hoe je ze als tests schrijft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_plus_31, choice_which_helped, result_parts):
    L.wrap_up(result_parts, choice_which_helped, choice_plus_31)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
