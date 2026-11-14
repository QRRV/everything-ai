# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-make-without-code. Route: course/route/lessen/bouwen.yaml. Checks, de 16 programma's en hun plaatje: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-make-without-code", "nl")
    C = L.checks   # de 16 programma's die Qwen3 uit een beschrijving schreef, wat ze geven, en het plaatje


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frontier():
    quick_frontier = L.choice(
        "quick-frontier",
        question="Qwen3 vermenigvuldigde 23 met 47 goed in alle acht runs, en telde de r'en in *strawberry* fout in alle acht. Wat zegt dat over een nieuwe taak?",
        options={
            "easy": "Als het er makkelijk uitziet, doet het model het goed",
            "try": "Nog niets: dat zoek je uit door het te proberen op gevallen die je kunt nakijken",
            "never": "Dat je een model met geen enkele taak kunt vertrouwen",
        },
        correct="try",
        why={
            "try": "De frontier is grillig: twee taken die er even makkelijk uitzien, kunnen aan verschillende kanten ervan liggen.",
            "easy": "Letters tellen ziet er makkelijk uit, en ging elke keer fout. Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
            "never": "Hetzelfde model vermenigvuldigde elke keer goed. Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
        },
    )
    quick_frontier
    return (quick_frontier,)


@app.cell(hide_code=True)
def quick_check_centaur():
    quick_centaur = L.choice(
        "quick-centaur",
        question="Je geeft een assistent een hele taak, en kijkt het resultaat na als hij klaar is. Welke manier van werken is dat?",
        options={
            "centaur": "Centaur",
            "cyborg": "Cyborg",
        },
        correct="centaur",
        why={
            "centaur": "Een heel deel gaat naar de assistent, en je kijkt het als geheel na. Zo werkt vibe coding ook, zoals je zult zien.",
            "cyborg": "Een cyborg wisselt stap voor stap af met de assistent. Hier gaat het werk één keer heen en komt het één keer terug. Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
        },
    )
    quick_centaur
    return (quick_centaur,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Met AI kun je dingen maken vanuit een beschrijving: een plaatje, een liedje, een korte video, en ook een klein programma, zonder code te schrijven. Dat gaat snel, en voor een eerste versie is het vaak goed genoeg. Maar wat je niet zelf maakte, ken je niet van binnen. Waar een fout zit, merk je pas als je het probeert.

    **Een voorbeeld uit het dagelijks leven.** Je bestelt telefonisch een verjaardagstaart: *chocolade, voor twaalf personen*. Hij komt aan en ziet er prachtig uit. Of er noten in zitten, merk je pas als een gast met een allergie een hap neemt, tenzij je het van tevoren vraagt. Dat is alleen een vergelijking; hieronder zie je wat Qwen3 maakte uit een beschrijving van een klein programma.
    """)
    return


@app.cell(hide_code=True)
def predict_programs():
    guess_programs = L.guess(
        "programs",
        question="De course vroeg Qwen3 zestien keer om een klein programma dat zegt hoeveel dagen een maand heeft. Hoeveel van de zestien programma's hadden elke maand van 1890 tot en met 2110 goed?",
        options={
            "most": "De meeste",
            "few": "Twee of drie",
            "none": "Niet één",
        },
        reveal="**Niet één.** Elk van de zestien programma's heeft een maand fout; een ervan alleen februari 1900 en februari 2100, andere meer dan duizend maanden. De meeste zagen er goed uit, en ze draaiden allemaal zonder foutmeldingen. Hieronder zie je waar elk programma de fout in gaat, zonder een regel code te lezen.",
    )
    guess_programs
    return (guess_programs,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **vibe coding** | een AI een programma laten maken vanuit een beschrijving, en het alleen beoordelen door het te gebruiken, zonder de code te lezen; de naam komt van Andrej Karpathy (2025) |

    ### Dingen maken vanuit een beschrijving

    | Wat je maakt | Voorbeelden van tools (november 2026) | Waar het misgaat |
    |---|---|---|
    | een plaatje | Midjourney, de beeldgeneratoren in ChatGPT en Gemini | tekst in het plaatje, kleine details, dingen waar je niet om vroeg |
    | muziek | Suno, Udio | songteksten, en een liedje dat klinkt als een bestaand liedje |
    | een video | Veo, Sora | bewegingen die niet kunnen, dingen die van het ene moment op het andere veranderen |
    | een programma | Lovable, Bolt, Replit, of elke chatassistent | de gevallen die niemand probeerde |

    Een programma, een lijst instructies voor een computer, verschilt op één punt van een plaatje: een plaatje zie je in zijn geheel, maar een programma zie je pas als het draait, één geval tegelijk. Een fout getal in een geval dat je nooit probeert, zie je nooit.

    **De run.** De course gaf Qwen3-0.6B twee beschrijvingen, elk acht keer. Beschrijving A vraagt om een Python-functie `days_in_month(month, year)` die zegt hoeveel dagen een maand heeft, en om alleen de code. Beschrijving B voegt toe: *also for February in leap years* (ook voor februari in schrikkeljaren). De prompts zijn in het Engels, omdat Qwen3 Engels het best schrijft. Een schrikkeljaar is een jaar dat deelbaar is door 4, behalve eeuwjaren, die alleen schrikkeljaren zijn als ze deelbaar zijn door 400: 2000 was een schrikkeljaar, 1900 niet. De course probeerde elk programma op acht maanden, zonder er een regel aan te veranderen:
    """)
    return


@app.cell(hide_code=True)
def picture_programs():
    mo.vstack([
        mo.Html(C.programs_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_code():
    L.explore("program-code", "program", lambda _v: C.program_frame(int(_v), L.lang), start=4, low=1, high=len(C.OUTPUTS), step=1, role="",
              task="Schuif langs de zestien programma's. Je hoeft de code niet te lezen om te zien waar elk programma de fout in gaat; als je hem kunt lezen, kijk dan hoe zeker van zichzelf hij eruitziet.")
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("Een grote assistent schrijft dit programma veel vaker goed dan Qwen3-0.6B. Maar een groter programma heeft meer gevallen die niemand probeert, en daar maakt ook een groot model fouten die je pas ziet als je test.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: iets maken zonder code

    1. **Beschrijf wat je wilt,** met de gevallen die ertoe doen: *ook voor februari in schrikkeljaren*.
    2. **Gebruik het als iemand die het kapot wil krijgen:** probeer de gewone gevallen, en de randen, zoals de kortste maand, een schrikkeljaar en een eeuwjaar.
    3. **Schrijf op wat je probeerde,** zodat je het na elke verandering opnieuw kunt proberen.
    4. **Probeer na een verandering alles opnieuw,** niet alleen wat je liet veranderen.
    5. **Hangt er geld, persoonsgegevens of iemands gezondheid van af?** Dan is vibe coding niet genoeg: iemand moet de code lezen en testen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je laat een assistent een kleine webpagina voor een feestje maken: gasten vullen hun naam in en wat ze meenemen.

    1. **Beschrijf:** *Een pagina waar gasten hun naam invullen en wat ze meenemen, en zien wat de anderen meenemen. Dezelfde naam twee keer telt niet dubbel.*
    2. **Gebruik het om het kapot te krijgen:** je vult je eigen naam in, dan dezelfde naam nog eens, dan een naam met een spatie aan het eind, en een lege naam.
    3. **Schrijf op:** vier proefjes, en wat de pagina deed: de naam met een spatie telde dubbel.
    4. **Na de verbetering alle vier opnieuw:** de spatie is opgelost, maar nu wordt een lege naam geaccepteerd. Daar vraag je ook om, en je probeert alle vier opnieuw.
    5. **Geld, persoonsgegevens, gezondheid?** Namen en gerechten voor een feestje: nee. Vibe coding is hier prima. Als de pagina ook betalingen aannam, niet meer.
    """)
    return


@app.cell(hide_code=True)
def text_vibe_or_read():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Vibe coding, of iemand die de code leest?

    Bij welke van deze programma's is vibe coding prima, en bij welke moet iemand de code lezen en testen?
    """)
    return


@app.cell(hide_code=True)
def exercise_vibe_or_read():
    sort_vibe = L.sort("vibe-or-read")
    sort_vibe
    return (sort_vibe,)


@app.cell(hide_code=True)
def check_vibe_or_read(sort_vibe):
    result_vibe = L.check("vibe-or-read", sort_vibe)
    result_vibe
    return (result_vibe,)


@app.cell(hide_code=True)
def hints_vibe():
    L.hints(
        "vibe-or-read",
        [
            "Vraag bij elk: als het misgaat zonder dat iemand het merkt, wat kost dat dan, en wie?",
            "Je eigen foto's met een kopie, een spelletje, een quiz: een fout kost je een poging. Loon, rekeningnummers, medicijnen: een fout kost iemand anders iets.",
            "Bij drie van de zes is vibe coding prima, en bij drie leest iemand de code.",
        ],
        why="Vibe coding is prima: **Vakantiefoto's**, **Een spelletje** en **Een quiz**: een fout kost je een poging. Iemand leest de code: **Salarissen**, **Leden** en **Medicijnen**: daar hangen geld, persoonsgegevens en gezondheid van af.",
    )
    return


@app.cell(hide_code=True)
def text_only_recent():
    mo.md(r"""
    ### Oefening 2. Te makkelijke proefjes
    """)
    return


@app.cell(hide_code=True)
def choice_only_recent_cell():
    choice_only_recent = L.choice(
        "only-recent",
        question="Stel dat je alleen maanden van 2024 en 2025 had geprobeerd. Welk programma had er dan goed uitgezien, terwijl het dat niet is?",
        options={
            "a1": "Programma A1",
            "a4": "Programma A4",
            "b1": "Programma B1",
        },
        correct="a4",
        why={
            "a4": "A4 maakt elk vierde jaar een schrikkeljaar, en dat klopt van 1901 tot en met 2099. Alleen een eeuwjaar zoals 1900 laat zien dat het fout is. Daarom probeer je de randen, niet alleen de gevallen die als eerste bij je opkomen.",
            "a1": "Kijk naar de kolom *feb 2024* in het plaatje: wat geeft A1, en wat is goed?",
            "b1": "Kijk naar de kolommen *feb 2025* en *feb 2024*: B1 zit er in allebei naast.",
        },
    )
    choice_only_recent
    return (choice_only_recent,)


@app.cell(hide_code=True)
def text_longer_description():
    mo.md(r"""
    ### Oefening 3. Een langere beschrijving
    """)
    return


@app.cell(hide_code=True)
def choice_longer_description_cell():
    choice_longer_description = L.choice(
        "longer-description",
        question="Beschrijving B voegde *also for February in leap years* toe. Wat veranderde dat?",
        options={
            "all": "Nu waren alle acht programma's goed",
            "trade": "Februari was veel vaker goed, maar bij nog meer programma's gingen de andere maanden fout",
            "same": "Niets: de programma's bleven hetzelfde",
        },
        correct="trade",
        why={
            "trade": "Februari ging van goed in alle vier de jaren bij geen van de acht programma's naar goed bij zes; maar de programma's met een andere maand fout gingen van vijf van de acht naar zeven. De langere beschrijving maakte goed wat ze noemde, en verder niets. Dus probeer je na elke verandering alles opnieuw.",
            "all": "Kijk naar de rijen van beschrijving B in het plaatje: hoeveel hebben geen gemarkeerd getal?",
            "same": "Vergelijk de kolom *feb 2024* voor de programma's van A en van B.",
        },
    )
    choice_longer_description
    return (choice_longer_description,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl) vond je de frontier van Qwen3 door taken te proberen die je kon nakijken; hier deed je hetzelfde voor een programma: je beoordeelde het op wat het geeft, geval voor geval. Vibe coding is centaurwerk, een hele taak uit handen gegeven en achteraf nagekeken. Verderop in deze module laat [Code nakijken die AI schreef](/course/open/build-check-ai-code?lang=nl) zien hoe je zulke proefjes opschrijft als tests die vanzelf draaien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_longer_description, choice_only_recent, result_vibe):
    L.wrap_up(result_vibe, choice_only_recent, choice_longer_description)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
