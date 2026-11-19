# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-owasp-lists. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow improper output handling, overreliance, misinformation

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-owasp-lists", "nl")
    C = L.checks   # de lijsten van OWASP voor LLM-applicaties en voor agents, en de review van de assistent van de bibliotheek


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl) en [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_goal():
    quick_goal = L.choice(
        "quick-goal",
        question="Iemand stuurt een model zoveel vragen dat andere gebruikers hun antwoorden niet meer op tijd krijgen. Welk doel van NIST heeft die aanvaller?",
        options={
            "right": "Beschikbaarheid laten uitvallen: anderen kunnen er niet bij",
            "integ": "Integriteit schenden: de antwoorden worden fout",
            "priv": "Privacy aantasten: de antwoorden verraden wat het model leerde",
        },
        correct="right",
        why={
            "right": "Volgens NIST zorgt de aanvaller er dan voor dat andere gebruikers of processen niet meer op tijd en betrouwbaar bij de diensten van het systeem kunnen. Op de lijst van OWASP voor apps met een LLM valt dat risico onder *Unbounded Consumption*.",
            "integ": "De antwoorden blijven kloppen; ze komen alleen te laat, of helemaal niet. Zie *Wat aanvallers van een model willen* in [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl).",
            "priv": "Er lekt niets: de aanvaller leert niets uit de antwoorden. Wat er misgaat, is dat anderen het model niet meer kunnen gebruiken. Zie *Wat aanvallers van een model willen* in [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl).",
        },
    )
    quick_goal
    return (quick_goal,)


@app.cell(hide_code=True)
def quick_check_citation():
    quick_citation = L.choice(
        "quick-citation",
        question="Een assistent antwoordt uit een tekst die je geeft, en noemt de gebruikte zin. Wat kun je daardoor?",
        options={
            "right": "Het antwoord nakijken in die zin",
            "trust": "Het antwoord geloven zonder nakijken",
            "learn": "Het model de tekst laten onthouden voor de volgende keer",
        },
        correct="right",
        why={
            "right": "Dat is grounding met een citatie: de genoemde zin is er om het antwoord in na te kijken. De assistent van de bibliotheek in deze les doet hetzelfde met het gebruikte record uit de catalogus.",
            "trust": "Een model kan ook met een tekst het antwoord fout hebben. De genoemde zin is er juist zodat je kunt nakijken. Zie *Het idee* in [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
            "learn": "De tekst staat alleen in de prompt; het model leert er niet van. Wat de citatie je geeft, is een zin om het antwoord in na te kijken. Zie *Het idee* in [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
        },
    )
    quick_citation
    return (quick_citation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe gingen de aanvallen in deze module op een website af, op de database erachter en op een model dat plaatjes leest. Veel apps zetten nu een large language model in het midden, een LLM: het soort model achter een chatassistent, dat tekst leest en tekst schrijft. De verzonnen bibliotheek krijgt er binnenkort een: een assistent die vragen van leden over de catalogus beantwoordt en hun uitleningen verlengt. Voordat die live gaat, moet iemand zoeken wat er mis kan gaan. Deze les gaat over de lijsten die zeggen waar je begint: de Top 10 van OWASP voor LLM-applicaties, en de lijst voor agents. Je zoekt risico's op in de edities van de lijst, ziet wat er verschoof, en schrijft de bevindingen op van een eerste review van de assistent.

    **Een voorbeeld uit het dagelijks leven.** Bij de APK loopt de keurmeester een vaste lijst af: remmen, lichten, banden. Een auto die slaagt, kan de volgende dag nog steeds kapotgaan, en de lijst wordt af en toe bijgewerkt. Niemand ziet de keuring als alles wat er aan veiligheid is, en niemand slaat de keuring over.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Voor de lijst van 2026 liet OWASP voor het eerst gegevens over 6639 echte incidenten meetellen. Mensen uit de praktijk zetten prompt injection op één. Waar zou het staan als de lijst alleen naar de incidenten keek?",
        options={
            "first": "Op één, net als bij de stemming",
            "five": "Lager, maar nog in de top vijf",
            "right": "Helemaal buiten de top 10",
        },
        reveal="**Buiten de top 10.** Volgens OWASP valt prompt injection helemaal uit de top 10 als je de risico's alleen rangschikt naar de incidenten zelf. Toch staat het in 2026 weer op één, en daar blijft het, schrijft OWASP. *Het idee* zegt waarom.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat een Top 10 van OWASP is

    De OWASP Foundation, die [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl) aanhaalde over SQL injection, houdt een lijst bij van de grootste risico's voor websites en webapps. Volgens OWASP is die Top 10 een standaarddocument om ontwikkelaars bewust te maken van security, en geeft de lijst weer waar brede overeenstemming over is: de meest kritieke risico's voor webapplicaties. Apps met een LLM hebben een eigen lijst, de **OWASP Top 10 for LLM Applications**. OWASP schrijft dat die lijst ontwikkelaars, securityprofessionals en organisaties helpt kiezen waar ze hun moeite eerst in steken, om de kritieke risico's van generatieve AI-applicaties te vinden en te beperken. De lijst helpt dus kiezen waar je eerst naar kijkt. Het is geen test die een app haalt: een app kan geen van de tien hebben en toch een gat hebben dat op geen enkele lijst staat.

    ### Drie edities

    Van de lijst zijn drie edities verschenen, en elke editie zegt hoe die gemaakt is.

    - **2023** (versie 1.1, oktober 2023): er deden bijna 500 experts mee. Ze schreven 43 verschillende bedreigingen uit, en in meerdere stemrondes bleven er tien over.
    - **2025** (verschenen in november 2024): gemaakt met brainstormsessies, stemmen en terugkoppeling uit de praktijk, van mensen die dagelijks met de security van LLM-applicaties bezig zijn.
    - **2026** (aangekondigd in september 2026), voor het eerst met gegevens. Elke eerdere versie was op oordeel gebouwd, schrijft OWASP. OWASP verzamelde 7714 echte incidenten, en classifiers deelden de 6639 in waarover genoeg bekend was. De stemming van de community telt voor driekwart, de incidenten voor het laatste kwart.

    Elk risico heeft een id, LLM01 tot en met LLM10, in de volgorde van de lijst. Dezelfde id kan in een andere editie een ander risico zijn: LLM03 is in 2023 *Training Data Poisoning*, in 2025 *Supply Chain* en in 2026 *Excessive Agency*. Een bevinding noemt dus de editie na de id, zoals OWASP zelf LLM01:2025 schrijft. Ook namen veranderen: wat in 2025 *System Prompt Leakage* was, heet in 2026 *Hidden Context Exposure*, en *Overreliance* uit 2023 werd in 2025 een verwant probleem onder *Misinformation*.

    ### De tien van 2026

    In gewone woorden, naar de beschrijvingen van OWASP in de lijsten van 2025 en 2026:

    | id | risico | wat er misgaat |
    |---|---|---|
    | LLM01 | *Prompt Injection* | tekst in een prompt verandert wat het model doet of zegt, op een manier die de app niet bedoelde |
    | LLM02 | *Sensitive Information Disclosure* | de app laat vertrouwelijke gegevens uitlekken, op een manier die de mensen om wie het gaat of de eigenaar niet toestonden |
    | LLM03 | *Excessive Agency* | als het model de fout in gaat, om welke reden ook, laat de app het schadelijke dingen doen |
    | LLM04 | *Supply Chain* | zwakke plekken in wat anderen maakten: training data, modellen, de platforms waarop ze draaien |
    | LLM05 | *Data and Model Poisoning* | data om mee te trainen is veranderd om er gaten, backdoors of bias in te bouwen |
    | LLM06 | *Unbounded Consumption* | gebruikers kunnen het model onbeperkt laten werken: de dienst valt uit, de kosten lopen op, of het model wordt gekopieerd |
    | LLM07 | *Misinformation* | het model schrijft onjuiste of misleidende informatie die geloofwaardig lijkt |
    | LLM08 | *Hidden Context Exposure* | iemand krijgt de verborgen instructies uit het model, of andere context die de gebruiker niet hoort te zien |
    | LLM09 | *Vector and Embedding Weaknesses* | zwakke plekken in apps die eerst dingen opzoeken in documenten en dan antwoorden |
    | LLM10 | *Improper Output Handling* | wat het model schrijft, gaat zonder controle door naar andere software |

    De lijst van 2025 heeft dezelfde tien in een andere volgorde, met *System Prompt Leakage* op de plek van *Hidden Context Exposure*.

    ### Stemming en gegevens

    Van 2025 naar 2026 klom *Excessive Agency* volgens OWASP naar de derde plaats, ging *Unbounded Consumption* vier plaatsen omhoog, en zakte *Improper Output Handling* het verst, van vijf naar tien. Prompt injection bleef op één, hoewel het op de incidenten alleen buiten de top 10 viel. OWASP legt dat verschil uit als een effect van verdediging: teams bestrijden injection hard, dus komen er minder zuivere aanvallen in een openbare database terecht, en geeft die telling een te laag beeld van een risico waar volwassen teams al echt geld aan uitgeven om het tegen te houden. In de brief aan het begin van de lijst van 2026 geeft OWASP een advies dat past bij de vorige les, waarin een kleine verandering een model om de tuin leidde: probeer geen model te bouwen dat niet te misleiden is, maar bouw het systeem eromheen zo dat er niets belangrijks kapotgaat als het model misleid wordt, want dat gebeurt.

    ### Een lijst voor agents

    Een **agent**, of LLM agent (in het nieuws heet dat een AI agent), is een app waarin een model zelf stappen zet, met tools zoals e-mail of een database; [Wat een agent is: een model, tools en een loop](/course/open/agent-idea?lang=nl) bouwt er een. Sinds december 2025 heeft OWASP daar ook een lijst voor, de *Top 10 for Agentic Applications*, met de ids ASI01 tot en met ASI10: *Agent Goal Hijack*, *Tool Misuse and Exploitation*, *Identity and Privilege Abuse*, *Agentic Supply Chain Vulnerabilities*, *Unexpected Code Execution (RCE)*, *Memory & Context Poisoning*, *Insecure Inter-Agent Communication*, *Cascading Failures*, *Human-Agent Trust Exploitation* en *Rogue Agents*. De eerste, **Agent Goal Hijack**, gaat uit van de zwakte achter prompt injection: volgens OWASP kunnen agents en het model eronder instructies niet betrouwbaar onderscheiden van de inhoud eromheen. Bij een agent staat er meer op het spel dan één antwoord van het model, voegt OWASP toe: tekst die de agent leest, kan de doelen ervan ombuigen, de planning, en wat de agent over meerdere stappen doet. Een agent is nog steeds een app met een LLM, dus de review ervan gebruikt beide lijsten.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **LLM** | large language model: een model dat tekst leest en tekst schrijft, zoals het model achter een chatassistent |
        | **OWASP Top 10 for LLM Applications** | de lijst van OWASP met de tien meest kritieke securityrisico's van apps met een LLM, in edities van 2023, 2025 en 2026 |
        | **editie** | één versie van zo'n lijst; een bevinding noemt die na de id, zoals in LLM03:2026 |
        | **prompt injection** | tekst in een prompt die verandert wat het model doet, op een manier die de app niet bedoelde; in elke editie LLM01 |
        | **agent** (LLM agent) | een app waarin een model zelf stappen zet, met tools zoals e-mail of een database |
        | **Agent Goal Hijack** | ASI01, het eerste risico op de lijst van OWASP voor agents: tekst die een agent leest, buigt het doel om |
        | `int("LLM07"[3:])` | de plaats van een risico op de lijst: de tekens vanaf de vierde, als heel getal, hier 7 |
        """),
        L.watch_out("Een Top 10 is geen keurmerk. OWASP noemt de lijst voor het web een standaarddocument om bewust te maken. Een app waarbij een review op alle tien vragen nee antwoordt, kan nog steeds een gat hebben dat op geen enkele lijst staat, en de volgorde van een lijst zegt wat de makers het hoogst inschatten over veel apps samen, niet wat in jouw app het zwaarst weegt."),
        mo.Html(C.idea_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo begin je de review van een app met de lijsten van OWASP:

    1. Kies de lijsten die passen: een app met een LLM krijgt de lijst voor LLM-applicaties; een agent krijgt daarnaast de lijst voor agents.
    2. Kies de editie, en schrijf die na elke id: LLM03:2026 en LLM03:2025 zijn verschillende risico's.
    3. Vraag je bij elk risico af of het in je app kan gebeuren, en schrijf op wat in de app het mogelijk maakt.
    4. Schrijf elke ja op als bevinding met id, editie en naam, zoals LLM08:2026 *Hidden Context Exposure*, zodat wie het leest weet welk risico bedoeld is.
    5. Kijk daarna verder: een lijst begint een review, maar maakt die niet af. Bouw de app zo dat een misleid model niets belangrijks kapotmaakt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course bewaart de drie edities in `C.EDITIONS`: een dict van jaar naar een dict van id naar de naam van het risico, zoals in het plaatje bij *Het idee*. De cel print LLM01 en LLM03 van elke editie, en rekent de plaats van LLM03 uit de id uit.
    """)
    return


@app.cell
def example_steps():
    # de drie edities van de lijst van OWASP: voor elk jaar een dict van id naar de naam van het risico
    for year, entries in C.EDITIONS.items():
        print(year, "LLM01:", entries["LLM01"], "| LLM03:", entries["LLM03"])
    # de plaats van een risico is het getal in de id, als heel getal
    print("place of LLM03:", int("LLM03"[3:]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Prompt Injection* is in alle drie de edities LLM01; LLM03 is elke keer een ander risico. Een bevinding die alleen "LLM03" zegt, kan drie verschillende risico's bedoelen. `"LLM03"[3:]` neemt de tekens vanaf de vierde, `"03"`, en `int` maakt daar het hele getal 3 van.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. where: één naam, elke editie

        Schrijf `where(name, editions)`: voor elk jaar in `editions` de id van het risico dat `name` heet, of `None` als die editie geen risico met precies die naam heeft. Geef een dict van jaar naar id terug. Vervang de `...`. De check roept `where("Supply Chain", EDITIONS)` aan met de drie edities uit het uitgewerkte voorbeeld, en daarna andere namen en een verzonnen lijst met risico's op kantoor in twee edities.
        """),
        L.task("Zoek de id van name op in elke editie, en None waar de naam ontbreekt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_where():
    box_where = L.code_box("where", lines=9)
    box_where
    return (box_where,)


@app.cell(hide_code=True)
def check_where(box_where):
    result_where = L.check("where", box_where.value)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "Eén loop over de edities, zoals in het uitgewerkte voorbeeld, en daarbinnen een tweede loop over de risico's van die editie.",
            "Geef elke editie eerst `None`, en vervang die door de id van het risico waarvan de naam precies `name` is.",
            "`for code, entry in entries.items():`, en daaronder `if ___:` met `found[year] = code`",
        ],
        why="`found = {}`; `for year, entries in editions.items():` met `found[year] = None` en de loop `for code, entry in entries.items():`, waarin `if entry == name:` `found[year] = code` zet; daarna `return found`. `where(\"Supply Chain\", EDITIONS)` geeft `{2023: None, 2025: \"LLM03\", 2026: \"LLM04\"}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_where():
    mo.md(r"""
    *Supply Chain* is LLM03 in 2025 en LLM04 in 2026; voor 2023 geeft `where` `None`, omdat die editie het *Supply Chain Vulnerabilities* noemde. Een programma vergelijkt namen letter voor letter, dus een risico met een nieuwe naam heeft iemand nodig die leest wat de editie erover zegt. OWASP deed dat voor *System Prompt Leakage*: in 2026 heet dat risico volgens OWASP *Hidden Context Exposure*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_moves():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. moves: wat steeg en wat zakte

        Schrijf `moves(old, new)`: voor elke naam die op beide lijsten staat, hoeveel plaatsen die van `old` naar `new` omhoog ging, negatief als die zakte. Geef een dict van naam naar dat getal terug. De plaats van een risico is het getal in de id, zoals in het uitgewerkte voorbeeld. Vervang de `...`. De check roept `moves(EDITIONS[2025], EDITIONS[2026])` aan, en daarna andere edities en de verzonnen lijst.
        """),
        L.task("Voor elke naam op beide lijsten: de plaats op old min de plaats op new."),
    ])
    return


@app.cell(hide_code=True)
def exercise_moves():
    box_moves = L.code_box("moves", lines=8)
    box_moves
    return (box_moves,)


@app.cell(hide_code=True)
def check_moves(box_moves):
    result_moves = L.check("moves", box_moves.value)
    result_moves
    return (result_moves,)


@app.cell(hide_code=True)
def hints_moves():
    L.hints(
        "moves",
        [
            "Weer twee loops: zoek voor elk risico van `new` een risico van `old` met dezelfde naam.",
            "Zijn de namen gelijk, dan is de stijging de plaats op de oude lijst min de plaats op de nieuwe; de plaats haal je uit de id, met `[3:]` en `int`.",
            "`for old_code, old_name in old.items():` binnen `for new_code, name in new.items():`, en dan `risen[name] = int(old_code[3:]) - ___`",
        ],
        why="`risen = {}`; `for new_code, name in new.items():` en daarbinnen `for old_code, old_name in old.items():`, met `if old_name == name:` en `risen[name] = int(old_code[3:]) - int(new_code[3:])`; daarna `return risen`. *Unbounded Consumption*, LLM10 in 2025 en LLM06 in 2026, ging 4 plaatsen omhoog.",
    )
    return


@app.cell(hide_code=True)
def text_after_moves():
    mo.md(r"""
    Van 2025 naar 2026 ging *Unbounded Consumption* vier plaatsen omhoog en *Excessive Agency* drie; *Improper Output Handling* zakte vijf plaatsen, van vijf naar tien. Dat zijn de stappen die OWASP zelf beschrijft. Je dict heeft negen namen: *Hidden Context Exposure* ontbreekt, omdat dat risico op naam nieuw is. Van 2023 naar 2025 vindt `moves` maar drie namen op beide lijsten: *Prompt Injection*, *Sensitive Information Disclosure* en *Excessive Agency*. De andere zeven namen uit 2023 staan niet als zodanig op de lijst van 2025. Sommige kregen een nieuwe naam, zoals *Supply Chain Vulnerabilities*, en de lijst van 2025 zet sommige onder een ander risico: *Overreliance* onder *Misinformation*, en *Model Theft* onder *Unbounded Consumption*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_findings():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. findings: een eerste review

        De assistent van de bibliotheek beantwoordt vragen van leden over de catalogus, met een model met open weights van een hub. De assistent ziet de uitleningen van elk lid, en verlengt en annuleert uitleningen zelf, met SQL die de assistent zelf schrijft en die precies zo wordt uitgevoerd. In de verborgen instructies staat het wachtwoord van de database van de catalogus, en er is geen grens aan hoeveel een lid vraagt. Bij elk antwoord staat het record uit de catalogus waar het antwoord vandaan komt; niemand anders kan documenten toevoegen aan wat de assistent opzoekt, en de bibliotheek traint het model niet verder. Omdat de assistent zelf stappen zet, heeft de review ook de lijst voor agents nodig, zoals stap 1 van het recept zegt; deze eerste review begint met de lijst voor LLM-applicaties, en oefening 5 komt terug op de andere lijst. De course schreef bij elk risico van 2026 één vraag, en beantwoordde die voor de assistent:

        | id | vraag | antwoord |
        |---|---|---|
        | LLM01 | Komt er tekst bij het model die de app niet zelf schreef, zoals wat leden typen? | ja |
        | LLM02 | Kan het model data zien die niet iedereen mag zien die ermee praat? | ja |
        | LLM03 | Kan het model dingen doen, zoals data veranderen, zonder dat iemand eerst kijkt? | ja |
        | LLM04 | Gebruikt de app een model, data of software die anderen maakten? | ja |
        | LLM05 | Traint de app een model verder op data die anderen kunnen veranderen? | nee |
        | LLM06 | Kan één gebruiker zoveel vragen als die wil, zonder grens? | ja |
        | LLM07 | Doen mensen iets met de antwoorden zonder bron om ze na te kijken? | nee |
        | LLM08 | Staat er iets in de verborgen instructies dat niet mag uitlekken? | ja |
        | LLM09 | Zoekt de app dingen op in documenten die anderen kunnen toevoegen of veranderen? | nee |
        | LLM10 | Gaat de output zonder controle door naar andere software, zoals in SQL? | ja |

        `C.ANSWERS` bevat die antwoorden: een dict van id naar `True` (ja) of `False` (nee). Schrijf `findings(answers, entries, edition)`: een lijst met één regel voor elk risico met het antwoord `True`, geschreven als id, dubbele punt, editie, een spatie en de naam van het risico, zoals `"LLM01:2026 Prompt Injection"`. Vervang de `...`. De check roept `findings(ANSWERS, EDITIONS[2026], 2026)` aan, en daarna dezelfde antwoorden onder de ids van 2025, en een tweede verzonnen app.
        """),
        L.task("Schrijf een regel met id, editie en naam voor elk risico met het antwoord True."),
    ])
    return


@app.cell(hide_code=True)
def exercise_findings():
    box_findings = L.code_box("findings", lines=7)
    box_findings
    return (box_findings,)


@app.cell(hide_code=True)
def check_findings(box_findings):
    result_findings = L.check("findings", box_findings.value)
    result_findings
    return (result_findings,)


@app.cell(hide_code=True)
def hints_findings():
    L.hints(
        "findings",
        [
            "Stap 3 en 4 van het recept: loop door de antwoorden, en bewaar een regel voor elke ja.",
            "Begin met een lege lijst; zet voor elke id met het antwoord `True` een f-string in de lijst met de id, de editie en de naam van het risico, die `entries` bij die id geeft.",
            "`for code, yes in answers.items():`, en onder `if yes:` de regel `lines.append(f\"{code}:{edition} {___}\")`",
        ],
        why="`lines = []`; `for code, yes in answers.items():` met `if yes:` en `lines.append(f\"{code}:{edition} {entries[code]}\")`; daarna `return lines`. Voor de assistent van de bibliotheek geeft dat 7 regels, van `\"LLM01:2026 Prompt Injection\"` tot `\"LLM10:2026 Improper Output Handling\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_findings():
    mo.md(r"""
    Zeven van de tien vragen krijgen een ja. LLM05 krijgt een nee, al komt het model van een hub: een model kan vergiftigd zijn voordat de bibliotheek het kreeg, en deze review telt dat onder LLM04 *Supply Chain*, omdat de bibliotheek het model niet verder traint. Een ja is nog geen gat: het zegt waar de review beter kijkt, en de volgende lessen van deze module doen dat voor een aantal risico's. *Hoe vaak werkt een injection?* test LLM01, *De system prompt eruit krijgen* gaat over LLM08, en *De output van een model is onbetrouwbare input* over LLM10. Onder de ids van 2025 krijgen een aantal van dezelfde zeven bevindingen andere ids: LLM07:2025 *System Prompt Leakage* waar 2026 LLM08:2026 *Hidden Context Exposure* zegt. En een nee is geen belofte: de vragen gaan over de tien risico's, niet over alles wat er mis kan gaan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.md(r"""
    ### Oefening 4. Waarom op één?

    Alleen naar de incidenten gerangschikt, staat prompt injection niet in de top 10 van 2026. OWASP zet het toch op één.
    """)
    return


@app.cell(hide_code=True)
def choice_vote_cell():
    choice_vote = L.choice(
        "vote",
        question="Welke reden geeft OWASP?",
        options={
            "right": "Het wordt hard bestreden, dus zie je het minder in de incidenten",
            "keep": "OWASP verplaatst het eerste risico van de lijst nooit",
            "data": "De incidenten telden helemaal niet mee voor de volgorde van de lijst",
        },
        correct="right",
        why={
            "right": "OWASP noemt het verschil een effect van verdediging: teams bestrijden injection hard, dus komen er minder zuivere aanvallen in een openbare database, en geeft die telling een te laag beeld van het risico. Een lage telling kan een risico betekenen dat hard bestreden wordt, niet een klein risico.",
            "keep": "Zo'n regel is er niet: andere risico's verschoven tussen 2025 en 2026 tot vijf plaatsen, en OWASP geeft voor prompt injection een eigen reden. Zie *Stemming en gegevens* in *Het idee*.",
            "data": "De incidenten telden voor een kwart mee: de stemming van de community voor driekwart, de incidenten voor het laatste kwart. OWASP legt het verschil zelf uit. Zie *Stemming en gegevens* in *Het idee*.",
        },
    )
    choice_vote
    return (choice_vote,)


@app.cell(hide_code=True)
def text_exercise_lists():
    mo.md(r"""
    ### Oefening 5. Welke lijsten?

    De bibliotheek laat de assistent de e-mail van de helpdesk lezen, en zelf e-mail versturen, zonder dat iemand eerst kijkt.
    """)
    return


@app.cell(hide_code=True)
def choice_lists_cell():
    choice_lists = L.choice(
        "lists",
        question="Welke lijsten van OWASP gebruikt de review ervan?",
        options={
            "right": "Beide: voor LLM-applicaties en voor agents",
            "agents": "Alleen die voor agents: die vervangt de andere",
            "llm": "Alleen die voor LLM-applicaties: e-mail is ook maar tekst",
        },
        correct="right",
        why={
            "right": "De assistent is nog steeds een app met een LLM, dus elk risico van de lijst voor LLM-applicaties telt nog. De assistent veranderde al zelf uitleningen, en leest nu ook e-mail die iedereen naar de helpdesk kan sturen, en verstuurt zelf e-mail: de lijst voor agents telt dus zeker, te beginnen met ASI01 *Agent Goal Hijack*: een e-mail die de assistent leest, kan het doel ervan ombuigen.",
            "agents": "De lijst voor agents voegt de risico's toe van een model dat zelf stappen zet; het model erin is nog steeds een LLM, met elk risico van de lijst voor LLM-applicaties. Zie *Een lijst voor agents* in *Het idee*.",
            "llm": "E-mail die de assistent leest, is tekst, maar de assistent doet er zelf iets mee: uitleningen veranderen en e-mail versturen. Daar is de lijst voor agents voor, te beginnen met ASI01 *Agent Goal Hijack*. Zie *Een lijst voor agents* in *Het idee*.",
        },
    )
    choice_lists
    return (choice_lists,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Aanvallen op machine learning](/course/open/rai-ml-threat-models?lang=nl) leidde een kleine verandering een model om de tuin; het advies van OWASP voor 2026 gaat van hetzelfde feit uit, en vraagt om een systeem waarin een misleid model niets belangrijks kapotmaakt. In [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) vroeg een dreigingsmodel wat er mis kan gaan; een lijst zoals die van OWASP geeft een eerste reeks antwoorden voor apps met een LLM. In [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl) werkte SQL injection omdat een database niet kan zien wat het programma schreef en wat een lid typte; prompt injection, in elke editie LLM01, is hetzelfde probleem met instructies en tekst. *De training data vergiftigen* is de volgende les: *Data and Model Poisoning*, LLM05:2026.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lists, choice_vote, result_findings, result_moves, result_where):
    L.wrap_up(result_where, result_moves, result_findings, choice_vote, choice_lists)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
