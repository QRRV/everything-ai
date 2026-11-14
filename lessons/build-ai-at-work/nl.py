# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-ai-at-work. Route: course/route/lessen/bouwen.yaml. Checks en de run van Qwen3 van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("build-ai-at-work", "nl")
    C = L.checks   # de tien taken, hoe een antwoord beoordeeld wordt, en de run van Qwen3 van de course


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_steps():
    quick_steps = L.choice(
        "quick-steps",
        question="Qwen3 controleerde getallen met een kaal ja of nee, en zei ja tegen alle 30, ook tegen de foute. Welke controle ving meer van de foute?",
        options={
            "twice": "Dezelfde ja-of-nee-vraag, twee keer gesteld",
            "steps": "Een controle die het antwoord eerst uitwerkt, met stappen die je kunt lezen",
            "none": "Geen: een model kan niets controleren",
        },
        correct="steps",
        why={
            "steps": "Het antwoord uitwerken gaf het model ruimte, en je kunt lezen waar het misging. Het miste er nog steeds een paar, dus je leest de stappen, niet alleen de laatste regel.",
            "twice": "Een model dat één keer ja zegt, zegt waarschijnlijk weer ja. Wat gaf het model meer ruimte? Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
            "none": "Eén soort controle ving wel een deel van de foute getallen. Welke? Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
        },
    )
    quick_steps
    return (quick_steps,)


@app.cell(hide_code=True)
def quick_check_source():
    quick_source = L.choice(
        "quick-source",
        question="Je geeft een assistent de openingstijden van een winkel als tekst, en vraagt hoe laat hij zaterdag sluit. Waarom vraag je de assistent de zin te noemen die hij gebruikte?",
        options={
            "faster": "Dan antwoordt het model sneller.",
            "always": "Dan is het antwoord altijd goed.",
            "check": "Dan kun je die zin lezen en zien of het antwoord erin staat.",
        },
        correct="check",
        why={
            "check": "Een citatie wijst je de zin aan, dus nakijken kost een paar seconden. En de citatie zelf kan fout zijn, dus je leest hem echt.",
            "faster": "Een citatie maakt het antwoord iets langer, niet sneller. Wat heb je aan het nummer van de zin? Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
            "always": "Qwen3 noemde soms de verkeerde zin, zelfs met de bron ervoor. Waar is een citatie dan voor? Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
        },
    )
    quick_source
    return (quick_source,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op je werk kan AI je bij de ene taak uren schelen en je bij de volgende een fout opleveren, terwijl die twee taken er even moeilijk uitzien. Weten welke welke is, en hoe je bij elke taak met AI samenwerkt, bepaalt of het je werk beter maakt of slechter.

    **Een voorbeeld uit het dagelijks leven.** Een navigatie-app kent de straten van een heel land, en loodst je zonder één verkeerde afslag door een vreemde stad. Maar af en toe stuurt hij je een weg in die al weken is afgesloten, en hij klinkt net zo zeker als altijd. Wie hem vaak gebruikt, leert waar je hem kunt vertrouwen en waar je zelf uit het raam moet kijken. Dat is alleen een vergelijking; hieronder zie je hetzelfde voor AI op je werk.
    """)
    return


@app.cell(hide_code=True)
def predict_outside():
    guess_outside = L.guess(
        "outside-frontier",
        question="In 2023 gaven onderzoekers 758 consultants van Boston Consulting Group realistische taken, met of zonder GPT-4. Eén taak was gekozen omdat GPT-4 hem fout zou doen. Hoe vaak vonden de consultants met GPT-4 bij die taak het goede antwoord, vergeleken met de consultants zonder?",
        options={
            "more": "Vaker: met AI werken mensen beter",
            "same": "Ongeveer even vaak: de AI maakte geen verschil",
            "less": "Minder vaak",
        },
        reveal="**Minder vaak.** Zonder AI vond 84,5% van de consultants het goede antwoord; met GPT-4 60% en 71% in de twee groepen die het hadden: gemiddeld 19 procentpunt minder. Velen namen de overtuigende maar foute analyse van het model over, en controleerden die minder. Bij 18 andere taken, die het model goed kon, maakten dezelfde consultants met GPT-4 12,2% meer taken af, 25,1% sneller, en hun werk kreeg een meer dan 40% hogere beoordeling. (Dell'Acqua en anderen, *Navigating the Jagged Technological Frontier*, 2023.)",
    )
    guess_outside
    return (guess_outside,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **jagged frontier** | de onregelmatige grens tussen taken die AI goed doet en taken die het fout doet; twee taken die voor een mens even moeilijk lijken, kunnen aan verschillende kanten ervan liggen |
    | **centaur** | een manier van samenwerken met AI waarbij je het werk verdeelt: sommige delen gaan in hun geheel naar de AI, de rest doe je zelf, en je kijkt na wat terugkomt |
    | **cyborg** | een manier van samenwerken met AI waarbij jij en de AI bij elke stap heen en weer gaan, zin voor zin, zo dicht op elkaar dat je nauwelijks nog ziet welk deel van wie is |

    Het onderzoek met de consultants gebruikte alle drie de woorden. De naam *centaur* komt uit het schaken, waar een mens en een schaakprogramma samen als één team spelen, zoals de centaur uit de oude Griekse verhalen: half mens, half paard.

    ### De frontier is grillig

    De grens is geen rechte lijn, met makkelijke taken binnen en moeilijke erbuiten. Hij gaat in en uit: een model kan dingen die voor jou moeilijk zijn, en faalt bij dingen die een kind kan. Aan de taak zie je het niet; je ontdekt het door hem te proberen, op gevallen waarvan je het antwoord weet.

    De course gaf Qwen3-0.6B, het kleine chat model van [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl), tien kleine taken, elk acht keer. Elk antwoord werd goed of fout gerekend volgens een regel die de course voor die taak schreef. De prompts zijn in het Engels, omdat Qwen3 Engels het best schrijft.
    """)
    return


@app.cell(hide_code=True)
def picture_frontier():
    mo.vstack([
        mo.Html(C.frontier(L.lang)),
        mo.md(C.frontier_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_tasks():
    L.explore("task-answers", "taak", lambda _v: C.task_frame(int(_v), L.lang), start=2, low=1, high=len(C.TASKS), step=1, role="",
              task="Schuif door de tien taken en lees wat Qwen3 schreef. Welke taak verrast je het meest?")
    return


@app.cell(hide_code=True)
def text_moves():
    mo.vstack([
        L.watch_out("De frontier van het ene model is niet die van het andere, en hij beweegt met elke nieuwe versie. Een grote assistent noemt de hoofdstad van Australië zonder moeite, en heeft toch taken die er net zo makkelijk uitzien en die hij fout doet. Dus test je het model dat je echt gebruikt."),
        mo.md(r"""
        Waarom ziet een model dat 23 keer 47 uitrekent geen drie r'en in *strawberry*? Een deel van het antwoord: een model leest tekst in tokens, stukjes van woorden, niet letter voor letter, dus de letters van een woord tellen is lastig voor een model. Later in de course zie je hoe een tekst in tokens wordt geknipt. Maar de reden heb je niet nodig om de frontier te zien: proberen is genoeg.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: AI in je dagelijks werk

    1. **Deel het werk op in taken.** Niet *de nieuwsbrief*, maar: de uitslagen verzamelen, de opening schrijven, de data controleren, de samenvatting vertalen.
    2. **Zoek uit waar elke taak ligt.** Probeer hem een paar keer op een geval waarvan je het antwoord weet. Elke keer goed: binnen de frontier. Ook maar één keer fout, op een manier die je niet zou merken: erbuiten.
    3. **Binnen: kies je manier van werken.** Als centaur geef je de hele taak uit handen en kijk je het resultaat als geheel na. Als cyborg werk je mee, zin voor zin.
    4. **Buiten: doe het zelf,** of laat de AI een deel doen dat je regel voor regel nakijkt.
    5. **Prompt zoals je een collega zou inlichten.** Zeg voor wie het is en waarvoor, geef de tekst die nodig is, vraag om de vorm die je wilt, en vraag opnieuw met wat er ontbrak.
    6. **Test opnieuw** als je overstapt op een ander model, of als er een nieuwe versie uitkomt: de frontier beweegt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je schrijft de maandelijkse nieuwsbrief van een sportclub, en wilt er een assistent voor gebruiken.

    1. **Opdelen.** Vier taken: de uitslagen van de maand, een openingsalinea, de data van de volgende wedstrijden, en een korte samenvatting in het Engels voor de leden die uit het buitenland komen.
    2. **Waar ligt elke taak?** Je probeert de uitslagen: de assistent maakt een nette tabel van je aantekeningen, maar één keer verwisselt hij twee scores. Dat ligt buiten de frontier, op een manier die je niet zou merken. De data van de volgende wedstrijden kan hij helemaal niet weten. De openingsalinea en de samenvatting gaan elke keer dat je het probeert goed.
    3. **Binnen.** De opening schrijf je als cyborg: je vraagt om drie versies, neemt een zin uit één ervan, en vraagt opnieuw met een grap uit de bestuursvergadering. De samenvatting doe je als centaur: je geeft de hele nieuwsbrief, en leest het Engels één keer door.
    4. **Buiten.** De uitslagen en de data typ je zelf over van de website van de bond. Of de assistent maakt de tabel, en jij controleert elke score met de website.
    5. **Prompt.** *Dit is de nieuwsbrief van onze voetbalclub, voor leden en ouders. Schrijf een samenvatting in het Engels van hoogstens 100 woorden, op een vriendelijke toon, met de data precies zoals ze er staan.*
    6. **Test opnieuw.** Volgende maand krijgt de assistent van de club een nieuwe versie: je probeert de tabel met uitslagen opnieuw voordat je erop vertrouwt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Centaur of cyborg?

        Zes mensen gebruiken op hun werk een assistent. Wie verdeelt het werk, en wie gaat bij elke stap heen en weer?
        """),
        L.task("Zet elk van de zes in de goede groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_ways = L.sort("centaur-or-cyborg")
    sort_ways
    return (sort_ways,)


@app.cell(hide_code=True)
def check_centaur_or_cyborg(sort_ways):
    result_ways = L.check("centaur-or-cyborg", sort_ways)
    result_ways
    return (result_ways,)


@app.cell(hide_code=True)
def hints_ways():
    L.hints(
        "centaur-or-cyborg",
        [
            "Kijk hoe vaak het werk heen en weer gaat tussen de persoon en de assistent.",
            "Eén keer heen en één keer terug, met een controle aan het eind: dat is het werk verdelen. Veel korte beurten: dat is meewerken.",
            "Drie van de zes geven een heel deel uit handen en kijken het daarna na.",
        ],
        why="Centaur: **Notulen**, **Vertaling** en **Verdeeld**: een heel deel gaat naar de assistent, en komt terug om nagekeken te worden. Cyborg: **Zin voor zin**, **Sparren** en **Regel voor regel**: veel korte beurten, zodat het resultaat door allebei gemaakt is.",
    )
    return


@app.cell(hide_code=True)
def text_frontier_qwen():
    mo.md(r"""
    ### Oefening 2. Binnen of buiten?
    """)
    return


@app.cell(hide_code=True)
def choice_frontier_qwen_cell():
    choice_frontier_qwen = L.choice(
        "frontier-qwen",
        question="Op je werk wil je Qwen3 zinnen laten omzetten in boodschappenlijstjes voor de kantine, en hoofdsteden laten opzoeken voor een quiz. Wat doe je, met het plaatje van de tien taken in gedachten?",
        options={
            "both": "Je gebruikt het voor allebei, en controleert af en toe",
            "lists": "Je gebruikt het voor de lijstjes, die je één keer doorleest; de hoofdsteden zoek je ergens anders op",
            "neither": "Je gebruikt het voor geen van beide: een model dat een hoofdstad fout heeft, kun je nergens mee vertrouwen",
        },
        correct="lists",
        why={
            "lists": "De lijstjes waren in alle acht runs goed: binnen de frontier, dus even doorlezen is genoeg. De hoofdstad was in alle acht fout, en het klonk zeker: erbuiten.",
            "both": "Kijk naar de hoofdstad in het plaatje: fout in elke run. Af en toe controleren laat de meeste foute antwoorden door. Welke van de twee taken ligt buiten de frontier?",
            "neither": "De frontier is grillig: een model dat bij de ene taak faalt, kan een andere goed doen. Kijk naar de boodschappenlijst in het plaatje.",
        },
    )
    choice_frontier_qwen
    return (choice_frontier_qwen,)


@app.cell(hide_code=True)
def text_outside_study():
    mo.md(r"""
    ### Oefening 3. De taak buiten de frontier
    """)
    return


@app.cell(hide_code=True)
def choice_outside_study_cell():
    choice_outside_study = L.choice(
        "outside-study",
        question="In het onderzoek vonden de consultants met GPT-4 bij de taak buiten de frontier minder vaak het goede antwoord. Wat ging er volgens de onderzoekers mis?",
        options={
            "slow": "GPT-4 was zo traag dat ze tijd tekortkwamen.",
            "trusted": "Ze vertrouwden op de analyse van het model, die overtuigend maar fout was, en stelden er minder vragen bij.",
            "refused": "GPT-4 weigerde bij die taak te helpen.",
        },
        correct="trusted",
        why={
            "trusted": "De cijfers leken compleet, maar in de aantekeningen van de interviews stonden details die de andere kant op wezen, en die miste het model. Zijn analyse las goed, dus velen namen hem over. Hun geschreven advies kreeg zelfs een hogere beoordeling, ook als het fout was.",
            "slow": "De consultants met GPT-4 waren bij die taak sneller, niet trager. Kijk nog eens naar de uitleg onder de vraag over de consultants.",
            "refused": "GPT-4 hielp en gaf een antwoord, een overtuigend antwoord. Kijk nog eens naar de uitleg onder de vraag over de consultants.",
        },
    )
    choice_outside_study
    return (choice_outside_study,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Zoek de frontier zelf

    Bedenk een kleine taak uit je eigen werk of school, waarvan je het goede antwoord weet. Vraag het Qwen3, en druk een paar keer op de knop met een andere seed om te zien of het elke keer goed is. Qwen3 draait op je eigen computer, dus niets wat je typt verlaat hem. Het schrijft Engels het best.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value=C.TASKS["letters"][0], label="Je taak", rows=3, full_width=True)
    seed_box = mo.ui.number(start=0, stop=99, step=1, value=0, label="Seed")
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([prompt_box, mo.hstack([seed_box, ask_button], justify="start")])
    return ask_button, prompt_box, seed_box


@app.cell(hide_code=True)
def demo_answer(ask_button, prompt_box, seed_box):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Schrijf eerst een taak in het vak.*")
    else:
        _words = {"title": "Wat Qwen3 antwoordt", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden.",
                  "load": "Het laden van Qwen3 stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(chat.demo([{"role": "user", "content": prompt_box.value}], lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang,
                                 max_tokens=C.MAX_TOKENS, seed=int(seed_box.value)))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl) liet je een model zijn eigen antwoorden nakijken; hier beslis je welke taken dat nodig hebben, omdat ze buiten de frontier liggen, en hoe je het werk bij de rest verdeelt. De volgende lessen van deze module gaan door het werk zelf: zoeken en leren met AI, dingen maken zonder code, werk automatiseren, en hulp bij code.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_frontier_qwen, choice_outside_study, result_ways):
    L.wrap_up(result_ways, choice_frontier_qwen, choice_outside_study)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
