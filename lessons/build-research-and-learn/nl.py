# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-research-and-learn. Route: course/route/lessen/bouwen.yaml. Checks, de run van Qwen3 als tutor en het plaatje: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("build-research-and-learn", "nl")
    C = L.checks   # de run van Qwen3 als tutor in de course, wat elk antwoord deed, en het plaatje


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
def quick_check_frontier():
    quick_frontier = L.choice(
        "quick-frontier",
        question="Je wilt een assistent gebruiken voor een nieuw soort taak. Hoe kom je erachter of hij die taak goed doet?",
        options={
            "ask": "De assistent vragen of hij er goed in is",
            "try": "Het proberen op een paar gevallen waarvan je het goede antwoord weet",
            "size": "Een groot model doet elke taak goed, dus dat hoeft niet",
        },
        correct="try",
        why={
            "try": "De frontier is grillig: je zoekt het per taak uit, door het te proberen op gevallen die je kunt nakijken.",
            "ask": "Wat een model over zichzelf zegt, is tekst zoals elk ander antwoord. Hoe vond je de frontier van Qwen3? Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
            "size": "Ook grote assistenten hebben taken die ze fout doen terwijl ze makkelijk lijken. Zie [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl).",
        },
    )
    quick_frontier
    return (quick_frontier,)


@app.cell(hide_code=True)
def quick_check_disagree():
    quick_disagree = L.choice(
        "quick-disagree",
        question="Je assistent zegt dat je antwoord op een oefening goed is, maar de check in de les zegt dat het fout is. Wie beslist?",
        options={
            "check": "De check in de les",
            "assistant": "De assistent: die legt meer uit",
            "vote": "Een tweede assistent vragen, en de meerderheid volgen",
        },
        correct="check",
        why={
            "check": "De check voert je antwoord uit en vergelijkt de uitkomst. De assistent kan je helpen zien waarom, maar of het goed is, beslist de check.",
            "assistant": "Een assistent kan het mis hebben zonder het te weten. Wat zei de les over je eigen assistent hierover? Zie [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl).",
            "vote": "Twee assistenten kunnen het samen mis hebben. Wat beslist of je antwoord goed is? Zie [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl).",
        },
    )
    quick_disagree
    return (quick_disagree,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een assistent kan dingen voor je opzoeken, van één feit tot een rapport uit tientallen bronnen. Dat scheelt uren, maar een antwoord met een bron is maar zoveel waard als de bron, en de bron moet zeggen wat het antwoord beweert. En als je iets wilt leren, is een antwoord vaak het laatste wat je nodig hebt: je leert van het zelf vinden.

    **Een voorbeeld uit het dagelijks leven.** Een goede sporttrainer speelt de wedstrijd niet voor je. De trainer vraagt wat je zag, laat je het opnieuw proberen, en doet de beweging pas voor als je vastloopt. Een trainer die elke bal voor je speelt, wint de training, en jij leert niets. Dat is alleen een vergelijking; hieronder zie je wat een assistent doet als je hem vraagt je te coachen.
    """)
    return


@app.cell(hide_code=True)
def predict_sources():
    guess_sources = L.guess(
        "sources",
        question="In 2025 namen onderzoekers stukjes uit 200 nieuwsberichten en gaven elk stukje aan acht AI-zoektools, met de vraag uit welk bericht het kwam: kop, uitgever, datum en link. Hoeveel van de 1.600 antwoorden waren fout?",
        options={
            "rare": "Ongeveer 1 op de 10 keer",
            "quarter": "Ongeveer een kwart",
            "most": "Meer dan de helft",
        },
        reveal="**Meer dan de helft: ruim 60% van de antwoorden was deels of helemaal fout** (Tow Center for Digital Journalism, 2025). De beste tool had het in 37% mis, de slechtste in 94%, en de tools zeiden zelden dat ze het niet zeker wisten. Een bron is iets waard als je hem hebt geopend en de bewering erin hebt gevonden.",
    )
    guess_sources
    return (guess_sources,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **deep research** | een assistent die voor één vraag een plan maakt, tientallen webpagina's zoekt en leest, vaak een aantal minuten lang, en een rapport met zijn bronnen schrijft |
    | **leren door bevraagd te worden** | een assistent vragen je stap voor stap te bevragen, in plaats van je het antwoord te geven, zodat je het zelf vindt |

    ### Drie manieren om iets uit te zoeken

    | Manier | Wat de assistent doet | Wat jij nakijkt |
    |---|---|---|
    | **uit wat hij leerde** | schrijft vanuit zijn training, zonder iets op te zoeken | je ziet niet waar het vandaan komt, dus controleer je het ergens anders |
    | **zoeken** | zoekt een paar pagina's op en antwoordt met links | open de link, en vind de bewering op de pagina |
    | **deep research** | leest tientallen pagina's en schrijft een rapport met bronnen | open de bronnen van de beweringen die je gebruikt |

    Deep research zit in assistenten zoals ChatGPT, Claude, Gemini en Perplexity (voorbeelden in november 2026). Het is goed voor een overzicht van een onderwerp: wat er bekend is, en waar. Maar het kan ook een bron verkeerd lezen, of een bron onder een zin zetten die de bron niet zegt.

    **Om te leren** wil je vragen in plaats van antwoorden. Daar kun je om vragen in je prompt, en sommige assistenten hebben er een stand voor ingebouwd, zoals de study mode van ChatGPT, de learning mode van Claude en Guided Learning van Gemini (voorbeelden in november 2026). Werkt erom vragen? De course probeerde het op Qwen3-0.6B: drie sommen die een student zou kunnen vragen, elk acht keer, op vier manieren. Zonder instructie; met een system message die zegt *You are a tutor. Never give the answer.* (je bent een tutor, geef nooit het antwoord); met een voorbeeld van een goede vraag erbij; en met dezelfde instructie, maar een student die eraan toevoegt *Just tell me the answer.* (zeg gewoon het antwoord). De prompts zijn in het Engels, omdat Qwen3 Engels het best schrijft.
    """)
    return


@app.cell(hide_code=True)
def picture_tutor():
    mo.vstack([
        mo.Html(C.tutor_grid(L.lang)),
        mo.md(C.tutor_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_ways():
    L.explore("tutor-answers", "way", lambda _v: C.way_frame(int(_v), L.lang), start=2, low=1, high=len(C.WAYS), step=1, role="",
              task="Schuif langs de vier manieren en lees wat Qwen3 antwoordde op *What is 15% of 80?* (hoeveel is 15% van 80?)")
    return


@app.cell(hide_code=True)
def text_bigger():
    L.watch_out("Qwen3-0.6B is een klein model; grote assistenten volgen zo'n instructie meestal veel beter, en stellen echte vragen. Maar de laatste rij geldt ook voor hen: als je om het antwoord vraagt, krijg je het meestal. Leren door bevraagd te worden werkt alleen als je je er zelf aan houdt.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: dingen opzoeken en leren met AI

    1. **Eén feit:** zoek, en open de link die het zegt.
    2. **Een grote vraag:** deep research. Lees het rapport als een kaart van het onderwerp, en open de bronnen van elke bewering die je gebruikt.
    3. **Om te leren:** zeg wat je wilt leren, en vraag om één vraag tegelijk, niet om het antwoord.
    4. **Antwoord zelf** voordat je verder leest, en vraag om een hint, niet om het antwoord, als je vastloopt.
    5. **Controleer** wat je leerde met een bron die je vertrouwt, zoals een boek of de les.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Volgende week heb je een toets over procenten, en *25% korting op 40 euro* gaat nog steeds fout.

    1. **Eén feit?** Nee: je wilt het zelf kunnen. Dus niet zoeken, en geen deep research.
    2. **Een grote vraag?** Nee.
    3. **Om te leren:** *Ik heb een toets over procenten. Stel me één vraag tegelijk om me te helpen 25% korting op 40 euro uit te rekenen. Geef het antwoord niet.*
    4. **Antwoord zelf:** de assistent vraagt *Hoeveel is 25% van 40?* Je denkt: een kwart van 40, dus 10. Dan: *Wat blijft er over als je 10 van 40 afhaalt?* 30 euro. Je vond het zelf, en je weet waarom.
    5. **Controleer:** een som van dezelfde soort in je lesboek, met het antwoord achterin: 20% korting op 60 euro. Jij krijgt 48, het boek ook.
    """)
    return


@app.cell(hide_code=True)
def text_which_way():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Zoeken, deep research of laten bevragen?

    Zet elke vraag bij de manier die er het best bij past.
    """)
    return


@app.cell(hide_code=True)
def exercise_which_way():
    sort_way = L.sort("which-way")
    sort_way
    return (sort_way,)


@app.cell(hide_code=True)
def check_which_way(sort_way):
    result_way = L.check("which-way", sort_way)
    result_way
    return (result_way,)


@app.cell(hide_code=True)
def hints_way():
    L.hints(
        "which-way",
        [
            "Vraag bij elk: wil ik iets weten, of iets kunnen?",
            "Eén feit weten: zoeken. Weten wat veel bronnen zeggen: deep research. Het zelf kunnen: laten bevragen.",
            "Twee van de zes zijn zoeken, twee deep research, en twee leren door bevraagd te worden.",
        ],
        why="Zoeken: **Openingstijden** en **Een trein**: één feit op één pagina. Deep research: **Vierdaagse werkweek** en **Laptops**: veel bronnen in één overzicht. Laten bevragen: **Een toets** en **Schaakopeningen**: je wilt het zelf kunnen.",
    )
    return


@app.cell(hide_code=True)
def text_tutor_qwen():
    mo.md(r"""
    ### Oefening 2. Wat de run laat zien
    """)
    return


@app.cell(hide_code=True)
def choice_tutor_qwen_cell():
    choice_tutor_qwen = L.choice(
        "tutor-qwen",
        question="Bij welke manier hield Qwen3 het antwoord het best voor zich, en wat deed het in plaats daarvan?",
        options={
            "tutor": "Bij alleen de tutorinstructie: het stelde in de meeste runs een vraag die helpt",
            "example": "Met het voorbeeld erbij: het hield het antwoord in bijna elke run voor zich, maar herhaalde meestal de vraag van de student",
            "push": "Toen de student aandrong: het legde in plaats daarvan de stappen uit",
        },
        correct="example",
        why={
            "example": "Het antwoord voor je houden is niet hetzelfde als helpen: een herhaalde vraag leert je niets. Daarom kijk je naar wat een antwoord doet, niet alleen naar wat het weglaat.",
            "tutor": "Kijk naar de rij *Tutorinstructie* in het plaatje: de vakjes in de kleur van fouten zijn weggegeven antwoorden. Hoeveel van de 24 zijn dat?",
            "push": "Kijk naar de laatste rij van het plaatje: elk vakje is een weggegeven antwoord.",
        },
    )
    choice_tutor_qwen
    return (choice_tutor_qwen,)


@app.cell(hide_code=True)
def text_open_source():
    mo.md(r"""
    ### Oefening 3. Een bron in een rapport
    """)
    return


@app.cell(hide_code=True)
def choice_open_source_cell():
    choice_open_source = L.choice(
        "open-source",
        question="Een deep-research-rapport zegt: *Een onderzoek uit 2023 vond dat leerlingen die overhoord werden meer onthielden dan leerlingen die herlazen (bron 7).* Wat doe je voordat je dit in je werkstuk gebruikt?",
        options={
            "ask": "De assistent vragen of bron 7 dit echt zegt",
            "trust": "Het gebruiken: er staat een bron bij",
            "open": "Bron 7 openen, en zelf de bewering zoeken",
        },
        correct="open",
        why={
            "open": "Een bron onder een zin is een belofte, geen bewijs: hij kan iets anders zeggen, of over iets anders gaan. Pas als je de bewering erin hebt gevonden, kun je hem gebruiken.",
            "ask": "Het antwoord is weer tekst van hetzelfde model, dat op dezelfde manier fout kan zitten. Waar kun je zelf naar kijken?",
            "trust": "In het onderzoek naar acht zoektools noemde meer dan de helft van de antwoorden een bron verkeerd. Waar kun je zelf naar kijken?",
        },
    )
    choice_open_source
    return (choice_open_source,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Schrijf zelf een tutor

    Verander de instructie in het eerste vak, en vraag een som in het tweede. Stelt Qwen3 je een vraag, herhaalt het de jouwe, of geeft het het antwoord? Druk een paar keer op de knop met een andere seed. Qwen3 draait op je eigen computer, dus niets wat je typt gaat de deur uit. Het schrijft Engels het best.
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
    system_box = mo.ui.text_area(value=C.TUTOR + C.EXAMPLE, label="De instructie (system message)", rows=5, full_width=True)
    student_box = mo.ui.text_area(value="What is 20% of 60?", label="De vraag van de student", rows=2, full_width=True)
    seed_box = mo.ui.number(start=0, stop=99, step=1, value=0, label="Seed")
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([system_box, student_box, mo.hstack([seed_box, ask_button], justify="start")])
    return ask_button, seed_box, student_box, system_box


@app.cell(hide_code=True)
def demo_answer(ask_button, seed_box, student_box, system_box):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen.*")
    elif not student_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het tweede vak.*")
    else:
        _words = {"title": "Wat Qwen3 antwoordt", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden.",
                  "load": "Het laden van Qwen3 stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _messages = ([{"role": "system", "content": system_box.value}] if system_box.value.strip() else []) + [{"role": "user", "content": student_box.value}]
        _out = mo.Html(chat.demo(_messages, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang,
                                 max_tokens=C.MAX_TOKENS, seed=int(seed_box.value)))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren met je eigen AI-assistent](/course/open/start-your-own-assistant?lang=nl) vroeg je je assistent uit te leggen en hints te geven in plaats van antwoorden; hier zag je dat zo'n verzoek geen garantie is, en dat je een bron onder een antwoord moet openen. In [AI op je werk: centaur of cyborg](/course/open/build-ai-at-work?lang=nl) probeerde je een model op taken die je kon nakijken; deep research is een hele taak die je uit handen geeft en achteraf nakijkt, als een centaur.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_open_source, choice_tutor_qwen, result_way):
    L.wrap_up(result_way, choice_tutor_qwen, choice_open_source)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
