# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-your-own-assistant. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-your-own-assistant", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over de lessen hiervoor.
    """)
    return


@app.cell(hide_code=True)
def quick_check_monday():
    quick_monday = L.choice(
        "quick-monday",
        question="Een assistent vertelt je dat het museum ook op maandag open is. Je wilt op een maandag gaan. Wat doe je?",
        options={
            "sure": "Je vraagt de assistent of hij het zeker weet.",
            "source": "Je kijkt op de website van het museum zelf.",
            "trust": "Je gaat: hij zei het zonder enige twijfel.",
        },
        correct="source",
        why={
            "sure": "Dezelfde assistent is geen tweede bron: hij kan zijn fout herhalen en net zo zeker klinken. Kijk nog eens naar het recept in [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).",
            "source": "Het museum kent zijn eigen openingsdagen. Zo controleer je bij de bron.",
            "trust": "Hoe zeker een antwoord klinkt, zegt weinig over hoe juist het is. Kijk nog eens naar het recept in [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).",
        },
    )
    quick_monday
    return (quick_monday,)


@app.cell(hide_code=True)
def quick_check_outcome():
    quick_outcome = L.choice(
        "quick-outcome",
        question="Waar kijkt een check in deze course naar?",
        options={
            "form": "Of je het precies zo schreef als de oplossing van de course.",
            "time": "Hoe lang je erover deed.",
            "outcome": "Naar de uitkomst van je antwoord.",
        },
        correct="outcome",
        why={
            "form": "Een check vergelijkt je tekst niet met de oplossing van de course: `5 + 5` telt net zo goed als `3 + 7`. Kijk nog eens in [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).",
            "time": "Een check houdt geen tijd bij. Kijk nog eens in [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).",
            "outcome": "Een check kijkt naar de uitkomst, dus een ander antwoord met de goede uitkomst telt ook als goed.",
        },
    )
    quick_outcome
    return (quick_outcome,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Misschien gebruik je al een AI-assistent. Tijdens het leren kan die je enorm helpen: hij wordt nooit moe van je vragen, en legt hetzelfde op tien manieren uit. Maar hij kan ook het denkwerk voor je doen. Geeft hij je elk antwoord, dan heb je de les af en niets geleerd. En hij kan ernaast zitten, zoals je zag in [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl).

    **Een voorbeeld uit het dagelijks leven.** Denk aan een rijinstructeur met eigen pedalen. Je leert pas rijden als je zelf mag sturen. Neemt de instructeur in elke bocht het stuur over, dan kom je veilig aan, maar rijden kun je nog steeds niet. Een goede instructeur pakt niet het stuur, maar zegt: "Kijk even in je spiegel."
    """)
    return


@app.cell(hide_code=True)
def predict_help():
    guess_help = L.guess(
        "which-help",
        question="Je zit vast bij een oefening. Van welke hulp denk je dat je het meest leert?",
        options={
            "answer": "Het hele antwoord, zodat je het kunt bestuderen.",
            "hint": "Een hint, zodat je de rest zelf vindt.",
            "nothing": "Helemaal geen hulp: je blijft het zelf proberen.",
        },
        reveal="**Meestal een hint.** Een uitgewerkt antwoord bestuderen helpt ook; daarom begint elke les met een uitgewerkt voorbeeld. Maar zit je eenmaal vast, dan laat een hint de laatste stap aan jou, en een stap die je zelf zet, blijft beter hangen. De hints van de course werken net zo: ze gaan één voor één open, met de oplossing als laatste.",
    )
    guess_help
    return (guess_help,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Je kunt je eigen assistent laten helpen bij deze course, zolang hij helpt zoals een goede instructeur dat doet.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **AGENTS.md** | een bestand in de map van de course met aanwijzingen voor een AI-assistent: hoe de course in elkaar zit, en hoe hij je helpt zonder de antwoorden voor te zeggen |

    ### Hoe je assistent de course kent

    Sommige assistenten werken in een map op je computer, zoals Claude Code, Codex en Cursor (voorbeelden in oktober 2026). Open er een in de map van de course: de map waarin je de course hebt gestart. In oktober 2026 lezen de meeste AGENTS.md vanzelf; doet die van jou dat niet, vraag hem dan eerst om AGENTS.md te lezen. Hij kan dan ook zien waar je bent: je eigen kopie van elke les en je voortgang, ook wat je in de oefenvakken typte, staan in de map *work*. Een chatprogramma in je browser ziet de map niet. Zeg daar dus bij welke les en oefening je bent, en plak de oefening erin.

    AGENTS.md vraagt je assistent onder meer om het antwoord niet voor te zeggen, alleen te gebruiken wat je tot nu toe hebt geleerd, en in jouw taal te antwoorden. Vraag je zelf ronduit om het antwoord, dan kan hij het toch geven: die keuze blijft van jou. En een assistent volgt aanwijzingen niet altijd, nog een reden om te controleren wat hij zegt.
    """)
    return


@app.cell(hide_code=True)
def picture_flow():
    mo.vstack([
        draw.flow(
            [("Jij", ["je vraagt, zet zelf", "de laatste stap, en de", "check zegt of het klopt"]),
             ("Je assistent", ["legt uit, geeft hints,", "kan ernaast zitten"]),
             ("Map van de course", ["AGENTS.md", "de lessen", "work: je voortgang"])],
            forward=["je vraag", "leest"],
            back=["een hint,\neen voorbeeld of\neen vraag terug"],
            title="Wie doet wat als je leert met een assistent"),
        mo.md(r"""
        Met een assistent in de map van de course: je vraag gaat naar de assistent, die leest AGENTS.md en je werk in de map, en geeft je iets terug waarbij jij blijft denken. Of je antwoord goed is, beslist nog steeds de check in de les.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: leren met een assistent

    1. **Laat hem de course zien.** Open hem in de map van de course, of zeg bij welke les en oefening je bent en plak de oefening erin.
    2. **Vraag om hulp, niet om het antwoord.** Bijvoorbeeld: "Leg dit uit met een ander voorbeeld", "Geef me één hint" of "Stel me een vraag die de goede kant op wijst."
    3. **Controleer wat hij zegt.** Klopt het met de les? Gebruikt hij woorden die de course je nog niet heeft geleerd? Vraag hem dan om het zonder die woorden uit te leggen, of om te zeggen in welke les de course ze behandelt.
    4. **Zet de stap zelf, en laat de check beslissen.** Je antwoord is goed als de check onder de oefening *Goed* zegt, niet als de assistent dat zegt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je bent bij oefening 1 van [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl): de som in het vak moet 10 opleveren. Je typte `3 + 5`, en de check zegt *Nog niet*. Je assistent staat open in de map van de course.

    1. **Laat hem de course zien.** Hij staat open in de map van de course, dus hij ziet in je voortgang wat je in het vak typte, en kan opzoeken wat de check daarover zegt. (Met een chatprogramma in je browser typ je zelf de les en de oefening, en plak je `3 + 5` en de melding van de check erin.)
    2. **Vraag om hulp, niet om het antwoord.** Je typt: *"Ik ben bij oefening 1 van Zo werkt een les, en de check zegt Nog niet. Geef me één hint, niet het antwoord."* Je typt niet *"Wat moet ik in het vak zetten?"*: dan zet de assistent de laatste stap voor je. Hij antwoordt: *"Kijk wat je som nu oplevert. Hoeveel mist er nog tot 10?"*
    3. **Controleer wat hij zegt.** De hint klopt met de les: ook de melding van de check zegt wat je som oplevert. Er staan geen nieuwe woorden in. Stel dat hij had gezegd: *"Sla de som eerst op in een variabele."* De course heeft je nog niet geleerd wat een variabele is, dus dan vraag je hem om het zonder dat woord uit te leggen, of om te zeggen in welke les de course het behandelt.
    4. **Zet de stap zelf, en laat de check beslissen.** Je ziet dat `3 + 5` samen 8 is: je komt er 2 tekort. Je maakt van de 5 een 7, en de check onder het vak zegt *Goed*.

    Een tweede geval: stel dat de assistent had gezegd *"Je antwoord is goed"*, terwijl de check nog steeds *Nog niet* zei. Dan ga je af op de check: die is gemaakt voor precies deze oefening. Terug naar stap 2, en vraag de assistent wat de melding van de check betekent.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ask():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Bij welke verzoeken blijf jij denken?

        Zes verzoeken die je een assistent zou kunnen sturen. Kijk naar stap 2 van het recept, en naar het verzoek dat je in het uitgewerkte voorbeeld niet typt.
        """),
        L.task("Zet elk verzoek in de goede groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    sort_ask = L.sort("ask-kinds")
    sort_ask
    return (sort_ask,)


@app.cell(hide_code=True)
def check_ask(sort_ask):
    result_ask = L.check("ask-kinds", sort_ask)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask-kinds",
        [
            "Kijk naar stap 2 van het recept: welke soorten hulp noemt die?",
            "Vraag je bij elk verzoek af: blijft er na het antwoord van de assistent nog een stap voor jou over?",
            "Bij drie verzoeken blijft er een stap voor jou over, bij drie zet de assistent die stap. *Invullen* is er een waarbij de assistent de stap zet.",
        ],
        why="Jij blijft denken: **Ander voorbeeld**, **Eén hint** en **Een vraag terug**; daarna is de laatste stap nog van jou. De assistent denkt voor je: **Invullen**, **Alleen het antwoord** en **Maak het goed**; daarmee krijg je een kant-en-klaar antwoord.",
    )
    return


@app.cell(hide_code=True)
def text_trust_check():
    mo.md(r"""
    ### Oefening 2. De assistent en de check zijn het oneens
    """)
    return


@app.cell(hide_code=True)
def choice_trust_check_cell():
    choice_trust_check = L.choice(
        "trust-check",
        question="Je assistent zegt dat je antwoord goed is, maar de check onder de oefening zegt *Nog niet*. Wat doe je?",
        options={
            "assistant": "Je gaat af op de assistent: die legt alles zo goed uit.",
            "check": "Je gaat af op de check, en vraagt de assistent om uitleg bij de melding.",
            "skip": "Je slaat de oefening over: de check zit er vast naast.",
        },
        correct="check",
        why={
            "assistant": "Goed uitleggen en gelijk hebben zijn twee verschillende dingen: een assistent kan ernaast zitten en net zo zeker klinken. Kijk nog eens naar stap 4 van het recept.",
            "check": "De check is gemaakt voor precies deze oefening en kijkt naar de uitkomst van je antwoord. Een assistent kan ernaast zitten en toch zeker klinken.",
            "skip": "Ook een check kan een fout bevatten; weet je dat zeker, meld het dan met de link onderaan de les. Van overslaan leer je niets. Kijk nog eens naar stap 4 van het recept.",
        },
    )
    choice_trust_check
    return (choice_trust_check,)


@app.cell(hide_code=True)
def text_new_word():
    mo.md(r"""
    ### Oefening 3. Een woord dat je nog niet kent
    """)
    return


@app.cell(hide_code=True)
def choice_new_word_cell():
    choice_new_word = L.choice(
        "new-word",
        question="Je assistent legt een stap uit met een woord dat de course je nog niet heeft geleerd. Wat doe je?",
        options={
            "nod": "Je knikt mee: het wordt later vanzelf duidelijk.",
            "stop": "Je gebruikt geen assistent meer: die is niet voor beginners bedoeld.",
            "ask": "Je vraagt hem om het zonder dat woord uit te leggen, of om te zeggen in welke les de course het behandelt.",
        },
        correct="ask",
        why={
            "nod": "Snap je één woord niet, dan hangt de rest van de uitleg ook in de lucht. Kijk nog eens naar stap 3 van het recept.",
            "stop": "Een assistent kan beginners juist veel helpen, zolang hij woorden gebruikt die je kent. Kijk nog eens naar stap 3 van het recept.",
            "ask": "Zo blijft de uitleg binnen wat je al kent, en weet je waar het woord terugkomt.",
        },
    )
    choice_new_word
    return (choice_new_word,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI-tools gebruiken met open ogen](/course/open/start-using-ai-tools?lang=nl) leerde je controleren wat een assistent zegt; hier gebruik je dat tijdens het leren. In [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl) leerde je de checks en de hints kennen; je assistent komt daar nu bij, en komt er niet voor in de plaats. Vanaf de volgende les schrijf je Python, en juist dan heb je het meest aan een assistent die hints geeft in plaats van antwoorden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_word, choice_trust_check, result_ask):
    L.wrap_up(result_ask, choice_trust_check, choice_new_word)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
