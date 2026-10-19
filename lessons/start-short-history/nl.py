# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-short-history. Route: course/route/lessen/stam-1-basis.yaml. Checks en de gebeurtenissen op de tijdlijn: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-short-history", "nl")
    C = L.checks   # de gebeurtenissen, de winters en de geschetste golf voor de tijdlijn


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over de vier cirkels.
    """)
    return


@app.cell(hide_code=True)
def quick_check_typing():
    quick_typing = L.choice(
        "quick-typing",
        question="Rond 2010 stelde het toetsenbord van een telefoon je volgende woord voor door in miljoenen berichten te tellen welk woord meestal volgde. Er zat geen neural network in. Wat is de kleinste cirkel waar het in past?",
        options={"ai": "AI", "ml": "Machine learning", "dl": "Deep learning"},
        correct="ml",
        why={
            "ai": "Het past in AI, maar er is een kleinere cirkel: het vond zijn regel in voorbeelden. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "ml": "Het vond zijn regel in voorbeelden, dus het is machine learning, en zonder neural network is het geen deep learning.",
            "dl": "Voor deep learning is een neural network met veel layers nodig, en dat zat niet in dit toetsenbord. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
        },
    )
    quick_typing
    return (quick_typing,)


@app.cell(hide_code=True)
def quick_check_rules():
    quick_rules = L.choice(
        "quick-rules",
        question="Een schaakprogramma beoordeelt elke stelling alleen met regels die schakers hebben opgeschreven. In welke cirkel hoort het?",
        options={"ml": "Machine learning", "ai": "AI, maar niet machine learning", "genai": "Generatieve AI"},
        correct="ai",
        why={
            "ai": "Mensen schreven de regels, dus het leerde niet van voorbeelden: het is AI, maar geen machine learning.",
            "ml": "Machine learning vindt de regels zelf in voorbeelden. Dit programma kreeg zijn regels van mensen. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "genai": "Generatieve AI ligt binnen machine learning en leert dus altijd van voorbeelden. Dit programma kreeg zijn regels van mensen. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
        },
    )
    quick_rules
    return (quick_rules,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    AI lijkt iets van de laatste jaren, maar de naam is al zo'n zeventig jaar oud. Sindsdien is het een paar keer op en neer gegaan: grote beloften, veel geld, dan teleurstelling, en jaren later een nieuwe start. Ken je die geschiedenis, dan lees je het nieuws over AI beter: wat is echt nieuw, en wat werd er al eerder beloofd?

    **Een voorbeeld uit het dagelijks leven.** Elektrische auto's. Rond 1900 reden er al flink wat rond, maar de benzineauto won, en bijna een eeuw lang waren elektrische auto's zo goed als verdwenen. Ze kwamen terug toen accu's goed en goedkoop genoeg waren geworden, en overheden schonere lucht wilden. Het idee was oud; het moest wachten. Hetzelfde zie je in de geschiedenis van AI.
    """)
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-learning",
        question="Wanneer denk je dat de eerste neural networks werden gebouwd die van voorbeelden leren?",
        options={
            "1950s": "In de jaren vijftig",
            "1980s": "In de jaren tachtig",
            "2012": "Rond 2012",
        },
        reveal="**In de jaren vijftig.** Het bekendste is het perceptron uit 1958, en er waren er al een paar eerder. Neural networks zijn een oud idee. In deze les zie je waarom het tot ongeveer 2012 duurde voordat ze echt doorbraken.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    De geschiedenis van AI verloopt in golven. In een goede periode is er veel hoop en geld, en wordt er veel beloofd. Komen die beloften niet uit, dan drogen geld en belangstelling jarenlang op.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **AI-winter** | een periode waarin de grote beloften van AI niet uitkwamen, waardoor geld en belangstelling opdroogden; de bekendste vielen in de jaren zeventig en eind jaren tachtig |

    Het woord *winter* is beeldspraak: AI stond niet stil, maar er was veel minder geld en aandacht voor. Niet elke historicus vindt dat de jaren zeventig een echte winter waren: in de Verenigde Staten bleef het totale geld voor AI-onderzoek groeien (Haigh, 2023). In Groot-Brittannië was de bezuiniging wel echt.
    """)
    return


@app.cell(hide_code=True)
def picture_timeline():
    mo.vstack([
        draw.timeline(
            [_e[0] for _e in C.EVENTS["nl"]], C.WINTERS, C.WAVE, start=1950, end=2026,
            winter_label="AI-winter", wave_label="hoop en geld",
            title="Zeventig jaar AI, in golven"),
        mo.md(r"""
        Elke winter volgt op een piek, en de laatste golf is tot nu toe de hoogste (2026). De lijn is een schets van hoop en geld, geen meting; de grijze banden zijn de twee AI-winters. De precieze jaartallen verschillen per boek; zie ze als een benadering. De nummers op de tijdlijn horen bij de lijst hieronder.
        """),
        mo.md("\n".join(f"{_i}. **{_e[0]} · {_e[1]}.** {_e[2]}" for _i, _e in enumerate(C.EVENTS["nl"], 1))),
    ])
    return


@app.cell(hide_code=True)
def text_two_ways():
    mo.md(r"""
    ### Twee manieren, om de beurt

    In [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) was de eerste vraag: schreven mensen de regels, of vond de computer ze in voorbeelden? De geschiedenis van AI gaat grotendeels over die twee manieren, die elkaar afwisselen.

    1. **Regels van mensen (1956 tot de jaren negentig).** Programma's die redeneren met regels, en in de jaren tachtig *expertsystemen*: programma's die advies geven in één smal vakgebied, met honderden regels die samen met menselijke experts zijn opgeschreven. In een kleine wereld werkte dat, maar de echte wereld vroeg om meer regels dan iemand kon opschrijven.
    2. **Leren van voorbeelden.** Dat begon al vroeg. In 1958 leerde het *perceptron* van zijn fouten: in de kern één enkel kunstmatig *neuron*, een kleine berekening met getallen die het na elke fout een beetje bijstelt. (De machine van Rosenblatt had er meerdere; één neuron is een vereenvoudiging.) In 1986 raakte *backpropagation* breed bekend: een manier om neural networks met meerdere layers van hun fouten te laten leren. Maar lange tijd was er te weinig data, en waren computers te traag.
    3. **Deep learning (vanaf ongeveer 2012).** Drie dingen kwamen samen: veel meer data, zoals foto's en tekst op internet; snelle GPU's, chips die oorspronkelijk voor de graphics van games zijn gemaakt en het rekenwerk van een neural network vele malen sneller doen; en betere manieren om netwerken met veel layers te laten leren. In 2012 herkende AlexNet, een diep neural network, dingen op foto's veel beter dan alles wat er tot dan toe was.
    4. **Large language models (vanaf ongeveer 2018).** Heel grote neural networks die van enorme hoeveelheden tekst leerden hoe tekst verdergaat. Met ChatGPT, eind 2022, bereikten ze het grote publiek; in oktober 2025 meldde OpenAI dat ongeveer 800 miljoen mensen ChatGPT elke week gebruikten.

    Twee keer werd er meer beloofd dan de programma's konden waarmaken, en volgde er een winter: in de jaren zeventig, toen zowel redenerende programma's als de eerste neural networks tegenvielen, en eind jaren tachtig, toen expertsystemen teleurstelden. En net als bij de elektrische auto waren neural networks een oud idee dat moest wachten op data en snelle computers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee systemen uit die geschiedenis. Stel bij elk de eerste vraag uit [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl): schreven mensen de regels, of vond de computer ze in voorbeelden?

    1. **XCON (1980)** hielp computermaker DEC om voor elke bestelde computer de juiste onderdelen samen te stellen, eerst met een paar honderd regels en later met duizenden, allemaal door de makers opgeschreven. **Mensen schreven de regels.**
    2. **Een neural network uit 1989** leerde de cijfers van handgeschreven Amerikaanse postcodes lezen (die bestaan alleen uit cijfers), van duizenden voorbeelden, elk met het cijfer erbij dat erop staat. Niemand schreef op hoe een 7 eruitziet; dat vond het netwerk zelf in de voorbeelden. **Het leerde van voorbeelden.**
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ways():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Regels of leren?

        Zes systemen uit de geschiedenis van AI. Stel dezelfde vraag als in het uitgewerkte voorbeeld.
        """),
        L.task("Zet elk systeem in de goede groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ways():
    sort_ways = L.sort("rules-or-learning")
    sort_ways
    return (sort_ways,)


@app.cell(hide_code=True)
def check_ways(sort_ways):
    result_ways = L.check("rules-or-learning", sort_ways)
    result_ways
    return (result_ways,)


@app.cell(hide_code=True)
def hints_ways():
    L.hints(
        "rules-or-learning",
        [
            "Vraag je bij elk systeem af: wie bepaalde hoe het antwoordt, een mens of de voorbeelden?",
            "Een systeem dat zelf zijn getallen bijstelt, of goed wordt door veel voorbeelden te zien, heeft geleerd. Een systeem waarvan iemand de patronen, regels of stappen opschreef, niet.",
            "Drie systemen kregen hun regels van mensen, en drie leerden zelf. Welke drie zijn een neural network, of een neuron?",
        ],
        why="Regels van mensen: **ELIZA**, **Expertsysteem** en **SHRDLU**; de makers schreven de patronen, de regels en de stappen op. Geleerd van voorbeelden: **Perceptron**, **AlexNet** en **ChatGPT**; die stelden hun eigen getallen bij door voorbeelden te zien.",
    )
    return


@app.cell(hide_code=True)
def text_winter():
    mo.md(r"""
    ### Oefening 2. Wat is een AI-winter?
    """)
    return


@app.cell(hide_code=True)
def choice_winter_cell():
    choice_winter = L.choice(
        "winter",
        question="Welke zin beschrijft een AI-winter?",
        options={
            "broken": "Een tijd waarin AI-programma's niet meer werkten.",
            "impossible": "Een tijd waarin wetenschappers bewezen dat AI onmogelijk is.",
            "promises": "Een tijd waarin AI meer had beloofd dan het kon, en geld en aandacht wegvielen.",
        },
        correct="promises",
        why={
            "promises": "De programma's bleven werken, en het onderzoek ging door; er was alleen veel minder geld en aandacht.",
            "broken": "De programma's werkten gewoon door. Kijk nog eens naar de tabel met het nieuwe woord: wat droogde er op?",
            "impossible": "Dat heeft niemand bewezen, en het onderzoek ging gewoon door. Kijk nog eens naar de tabel met het nieuwe woord.",
        },
    )
    choice_winter
    return (choice_winter,)


@app.cell(hide_code=True)
def text_why_2012():
    mo.md(r"""
    ### Oefening 3. Waarom pas in 2012?
    """)
    return


@app.cell(hide_code=True)
def choice_why_2012_cell():
    choice_why_2012 = L.choice(
        "why-2012",
        question="Het perceptron is van 1958. Waarom brak deep learning pas rond 2012 door?",
        options={
            "invented": "Voor 2012 had niemand aan neural networks gedacht.",
            "together": "Veel meer data, snelle GPU's en betere manieren om netwerken te laten leren kwamen samen.",
            "rules": "In 2012 kregen neural networks eindelijk regels over wat er op een foto staat.",
        },
        correct="together",
        why={
            "together": "Net als bij de elektrische auto was het idee oud; het moest wachten op genoeg data en computers die snel genoeg waren.",
            "invented": "Op de tijdlijn staan het perceptron in 1958 en backpropagation in 1986. Kijk nog eens naar stap 3 van *Twee manieren, om de beurt*.",
            "rules": "Een neural network krijgt geen regels van mensen: het vindt ze zelf in voorbeelden. Kijk nog eens naar stap 3 van *Twee manieren, om de beurt*.",
        },
    )
    choice_why_2012
    return (choice_why_2012,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) leerde je de twee manieren uit elkaar houden; hier zag je hoe ze elkaar zeventig jaar lang afwisselden. Later in de course bouw je die geschiedenis zelf na: één neuron, dan een netwerk met layers dat leert met backpropagation die je zelf schrijft, en aan het eind van de basis een klein language model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why_2012, choice_winter, result_ways):
    L.wrap_up(result_ways, choice_winter, choice_why_2012)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
