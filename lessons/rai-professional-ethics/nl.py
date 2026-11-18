# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-professional-ethics. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-professional-ethics", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Principes en richtlijnen](/course/open/rai-principles?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="Jobin, Ienca en Vayena lazen 84 documenten met principes voor AI. Welk principe noemden de meeste?",
        options={
            "right": "Transparantie",
            "solidarity": "Solidariteit",
            "freedom": "Vrijheid en autonomie",
        },
        correct="right",
        why={
            "right": "73 van de 84 documenten noemen transparantie, meer dan enig ander principe.",
            "solidarity": "Solidariteit kwam als laatste: maar 6 van de 84 documenten noemen het. Zie de afbeelding in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "freedom": "34 van de 84 documenten noemen vrijheid en autonomie, minder dan de helft. Zie de afbeelding in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def quick_check_same_word():
    quick_same_word = L.choice(
        "quick-same-word",
        question="Twee documenten noemen allebei fairness als principe. Wat weet je daarmee?",
        options={
            "same": "Dat ze precies hetzelfde vragen",
            "right": "Weinig; lees wat elk ervan ermee bedoelt",
            "source": "Dat ze allebei van dezelfde organisatie komen",
        },
        correct="right",
        why={
            "same": "Hetzelfde woord kan om verschillende dingen vragen, en twee documenten kunnen zelfs botsen. Zie het blok *Let op* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "right": "Het gedeelde woord verbergt verschillende betekenissen: wil je weten wat een principe vraagt, lees dan wat het in dat document betekent.",
            "source": "De 84 documenten kwamen van overheden, bedrijven en anderen. Twee documenten kunnen een woord delen en toch om verschillende dingen vragen. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
        },
    )
    quick_same_word
    return (quick_same_word,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Principes worden geschreven door organisaties. Maar wie het systeem bouwt, ziet vaak als eerste dat het iemand kan schaden, en krijgt toch de vraag om het op te leveren. Beroepsgroepen schrijven op wat hun leden aan het publiek verschuldigd zijn, zodat iemand in die positie iets heeft om naar te wijzen. En zelfs een keuze die puur technisch lijkt, zoals welke instelling van een model je gebruikt, rust op waarden. Deze les gaat over allebei.

    **Een voorbeeld uit het dagelijks leven.** Een arts belooft de patiënt voorop te zetten, wie de rekening ook betaalt. En hoe zeker een arts wil zijn voordat die een vlekje op een scan onschuldig noemt, hangt af van hoe erg het zou zijn om een ziekte te missen.
    """)
    return


@app.cell(hide_code=True)
def predict_need():
    guess_need = L.guess(
        "need",
        question="Onderzoekers lazen 100 veel geciteerde papers over machine learning van twee van de grootste conferenties. In hoeveel denk je dat de paper uitlegde in welke behoefte van de samenleving het werk voorzag?",
        options={
            "most": "Ongeveer 75",
            "half": "Ongeveer 50",
            "right": "Ongeveer 15",
        },
        reveal="**15 van de 100.** Maar 1 besprak mogelijke schade. Wat de papers wel het vaakst belangrijk vonden, waren prestaties, generalisatie, kwantitatief bewijs, efficiëntie, voortbouwen op eerder werk en nieuwheid (Birhane en collega's, 2022).",
    )
    guess_need
    return (guess_need,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Gedragscodes.** De ACM, een grote internationale vereniging van informatici, werkte in 2018 haar *Code of Ethics and Professional Conduct* bij. De eerste principes luiden: 1.1 draag bij aan de samenleving en aan het welzijn van mensen; 1.2 vermijd schade; 1.3 wees eerlijk en betrouwbaar; 1.4 wees rechtvaardig en ga discriminatie actief tegen; 1.6 respecteer privacy. Principe 3.1 vraagt leidinggevenden ervoor te zorgen dat het algemeen belang bij al het werk met computers voorop staat, en de code noemt het algemeen belang de belangrijkste overweging. De IEEE, een grote vereniging van ingenieurs in elektrotechniek en informatica, opent haar code met de belofte dat de veiligheid, de gezondheid en het welzijn van het publiek boven alles gaan. Een gedragscode is geen wet: wie hem breekt, pleegt geen misdrijf. Maar hij geeft een professional houvast om nee te zeggen, en een maatstaf waar anderen die professional aan kunnen houden.

        **Onderzoek is nooit vrij van waarden.** Het *value-free ideal* zegt dat wetenschap maatschappelijke en morele waarden buiten moet laten. In 1953 gaf de filosoof Richard Rudner een argument daartegen waarover nog steeds wordt gediscussieerd. Geen enkel bewijs maakt een bewering zeker, dus een wetenschapper moet beslissen hoeveel bewijs genoeg is, en dat hangt af van hoe erg een fout zou zijn: je wilt zekerder weten dat een medicijn geen dodelijke hoeveelheid gif bevat dan dat een partij riemgespen geen gebreken heeft. De filosoof Heather Douglas bouwde daar in 2009 op voort. In machine learning geldt hetzelfde voor welke vragen je onderzoekt, welke data je gebruikt en wat je het beste model noemt. Sinds 2020 vraagt NeurIPS, een van de grootste conferenties, auteurs om na te denken over de gevolgen van hun werk: eerst met een *broader impact statement* in elke paper, en vanaf 2021 met een checklist. In Europa vraagt *verantwoord onderzoeken en innoveren* (in het Engels *responsible research and innovation*) hetzelfde van onderzoek: denk vooruit over de gevolgen, betrek het publiek, en stuur bij als dat nodig is.

        De afbeelding laat een keuze zien zoals die van Rudner. Een check die haatberichten op een forum markeert, verzonnen door de course, kan streng of ruim worden ingesteld. Een strenge instelling laat minder haatberichten door, maar markeert meer onschuldige berichten. Welke instelling is de beste? De aantallen zeggen het niet: het hangt af van hoe erg een gemist bericht is vergeleken met een vals alarm, en dat is een vraag over wiens schade zwaarder telt.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **gedragscode** | regels die een beroepsgroep of organisatie afspreekt over hoe haar leden zich gedragen, zoals de ACM Code of Ethics voor informatici |
        | **value-free ideal** | het ideaal dat wetenschap vrij hoort te zijn van maatschappelijke en morele waarden; wetenschapsfilosofen stellen dat waarden altijd meespelen, bijvoorbeeld bij de keuze hoeveel bewijs genoeg is |
        """),
        L.watch_out("Fouten tellen is ook niet neutraal. Een prijs zetten op elke soort fout maakt de waarden zichtbaar, maar iemand koos die prijzen nog steeds: schrijf op wie, en waarom."),
        mo.Html(C.picture({"title": "Wat elke instelling fout doet in 1.000 berichten", "y": "berichten",
                           "misses": "haatberichten doorgelaten", "alarms": "onschuldige berichten gemarkeerd",
                           "alt": "Drie instellingen van een check op haatberichten, getest op 1.000 berichten. Strict laat 4 haatberichten door en markeert 60 onschuldige. Middle laat er 10 door en markeert 25. Loose laat er 25 door en markeert 8."})),
        mo.md(r"""
        Aantallen verzonnen door de course. Hoe strenger de instelling, hoe minder haatberichten erdoor komen, en hoe meer onschuldige berichten per vergissing gemarkeerd worden.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo verantwoord je een keuze in je werk:

    1. Benoem de keuze: wat je meet, welke data, welke instelling, wat je publiceert.
    2. Benoem wie er wint en wie er betaalt als het misgaat.
    3. Zoek op wat je gedragscode erover zegt, zoals 1.2 *vermijd schade*.
    4. Maak de waarden zichtbaar, bijvoorbeeld als een prijs voor elke soort fout.
    5. Schrijf op wat je koos en waarom, zodat anderen het kunnen nagaan en het oneens kunnen zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De check uit de afbeelding. `misses` bevat hoeveel haatberichten elke instelling doorlaat, `alarms` hoeveel onschuldige berichten hij markeert. De moderators van het forum tellen een gemist bericht als 10 keer zo erg als een vals alarm, omdat een haatbericht de mensen raakt op wie het gericht is: `miss_cost` is 10 en `alarm_cost` is 1. De kosten van een instelling zijn de kosten van de gemiste berichten plus de kosten van de valse alarmen. Voer de cel uit.
    """)
    return


@app.cell
def example_moderators():
    _misses = {"strict": 4, "middle": 10, "loose": 25}
    _alarms = {"strict": 60, "middle": 25, "loose": 8}
    _miss_cost = 10
    _alarm_cost = 1

    for _setting in _misses:
        print(_setting, _miss_cost * _misses[_setting] + _alarm_cost * _alarms[_setting])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met deze kosten is `strict` het goedkoopst: 100, tegen 125 en 258. De mensen die op het forum schrijven, kunnen dat anders zien: een onschuldig bericht dat wordt verwijderd, legt hen het zwijgen op. Oefening 1 rekent met hun kosten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wiens kosten?

        Dezelfde check, maar nu met de kosten van de schrijvers: een vals alarm telt 3 keer zo erg als een gemist bericht. `misses` en `alarms` bevatten de aantallen uit het uitgewerkte voorbeeld, `miss_cost` en `alarm_cost` wat één fout van elke soort kost, en de loop gaat met `setting` langs de instellingen. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld, maar zet de kosten van elke instelling in `cost` in plaats van ze te printen. De laatste regel is `cost`, zodat de check het ziet. De check probeert ook andere kosten.
        """),
        L.task("Zet in `cost`, onder elke instelling, de kosten van de gemiste berichten plus de kosten van de valse alarmen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=12)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "De som uit het uitgewerkte voorbeeld, met de uitkomst opgeslagen onder de instelling.",
            "Elk aantal wordt vermenigvuldigd met wat één zo'n fout kost. Welke van de twee kosten hoort bij de gemiste berichten?",
            "`cost[setting] = ___ * misses[setting] + ___ * alarms[setting]`",
        ],
        why="`cost[setting] = miss_cost * misses[setting] + alarm_cost * alarms[setting]` geeft `{'strict': 184, 'middle': 85, 'loose': 49}`: met de kosten van de schrijvers is `loose` het goedkoopst.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_principles():
    mo.md(r"""
    ### Oefening 2. Welk principe?

    Zes situaties die een engineer kan tegenkomen. Zet elke situatie bij het principe uit de ACM-code waar die vooral over gaat.
    """)
    return


@app.cell(hide_code=True)
def sort_principles_cell():
    sort_principles = L.sort("principles")
    sort_principles
    return (sort_principles,)


@app.cell(hide_code=True)
def check_principles(sort_principles):
    result_principles = L.check("principles", sort_principles)
    result_principles
    return (result_principles,)


@app.cell(hide_code=True)
def hints_principles():
    L.hints(
        "principles",
        [
            "Vraag bij elke situatie: loopt iemand gevaar, wordt iemand ongelijk behandeld, of wordt er iets over iemand bewaard of gelezen?",
            "Twee situaties gaan over mensen die gewond of ziek kunnen raken. Twee andere gaan over groepen die slechter behandeld worden dan andere.",
            "Vermijd schade: de brug en \\_\\_\\_. Wees rechtvaardig: \\_\\_\\_ en de lening. Respecteer privacy: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Vermijd schade:** de brug en het dieet. **Wees rechtvaardig:** het accent en de lening. **Respecteer privacy:** de locatie en de berichten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_values():
    mo.md(r"""
    ### Oefening 3. Alleen getallen?
    """)
    return


@app.cell(hide_code=True)
def choice_values_cell():
    choice_values = L.choice(
        "values",
        question="Een onderzoeker schrijft: \"Mijn paper laat alleen zien dat mijn model een hogere accuracy heeft. Er zitten geen waarden in.\" Wat zegt deze les?",
        options={
            "numbers": "Dat klopt: accuracy is een getal, dus vrij van waarden",
            "ethics": "Waarden spelen alleen mee in papers over ethiek of recht",
            "right": "Kiezen wat je meet, was al een keuze over waarden",
        },
        correct="right",
        why={
            "numbers": "Het getal kan goed gemeten zijn, maar iemand koos ervoor om accuracy te meten, op deze data, en het model met de hoogste accuracy het beste te noemen. Zie *Het idee*.",
            "ethics": "Het argument van Rudner geldt voor elke bewering die op bewijs rust: hoeveel bewijs genoeg is, hangt af van wat een fout zou kosten. Zie *Het idee*.",
            "right": "Wat je meet, op welke data, en wat je het beste noemt, zijn keuzes, en accuracy telt elke fout als even erg, wat zelf een waarde is.",
        },
    )
    choice_values
    return (choice_values,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Principes en richtlijnen](/course/open/rai-principles?lang=nl) zag je principes die organisaties voor AI schrijven. Een gedragscode doet hetzelfde voor een beroepsgroep: hij zegt wat je het publiek verschuldigd bent, voor wie je ook werkt. En de kosten in Oefening 1 lieten zien dat zelfs een technische keuze, welke instelling je gebruikt, rust op waarden die je kunt benoemen en verantwoorden. In de volgende les, *Toestemming en licenties van data*, vraag je of je een dataset überhaupt mag gebruiken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_values, result_cost, result_principles):
    L.wrap_up(result_cost, result_principles, choice_values)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
