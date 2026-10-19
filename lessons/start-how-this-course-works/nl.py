# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-how-this-course-works. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("start-how-this-course-works", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In deze course leer je AI door het te doen. Je leest niet alleen over een model: je schrijft de code, voert hem uit en ziet wat er gebeurt. Na elke oefening zegt een check meteen of je antwoord klopt, en zo niet, waarom niet.

    Dat werkt alleen als je de weg weet in een les. Daarom is deze eerste les een korte rondleiding, en doe je elke stap één keer. Daarna kun je je hoofd houden bij waar elke les over gaat.

    **Een voorbeeld uit het dagelijks leven.** De eerste rijles begint met een auto die stilstaat: waar de pedalen zitten, waar de spiegels voor zijn. Tien minuten zo besparen je een hoop gezoek als je eenmaal op de weg zit.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Elke les heeft dezelfde vijf delen, altijd in dezelfde volgorde:

    | Deel | Wat je daar doet |
    |---|---|
    | **De kop** | zien waar je bent in de route, wat het doel is en wat je nodig hebt |
    | **Waarom** | ontdekken welk probleem de les oplost |
    | **Het idee** | de nieuwe woorden, de uitleg en een uitgewerkt voorbeeld lezen |
    | **Zelf doen** | oefeningen maken, en een check zegt meteen of ze kloppen |
    | **Afronding** | zien wat je nu kunt, en waar het terugkomt |

    Deze les heeft dezelfde delen. Op dit moment ben je bij *Het idee*.

    ### Nieuwe woorden

    | Woord | Wat het betekent |
    |---|---|
    | **cel** | één blok van de les; er staat tekst in, zoals in dit blok, of code |
    | **code** | opdrachten voor de computer; in deze course geschreven in Python |
    | een cel **uitvoeren** | de computer de code in een cel laten doen; de uitkomst verschijnt direct onder de code |
    | **check** | een deel van de les dat naar je antwoord kijkt en zegt of het klopt, waarom, en wat je nu doet |
    | **kladblok** | een plek om code uit te proberen zonder de les te veranderen; in marimo heet het *Scratchpad* |

    Een les is een pagina van cellen, een *notebook*. De cellen die bij de les horen, zijn al uitgevoerd toen je hem opende.
    """)
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ### Een cel uitvoeren

        Hieronder staat een codecel. Je hoeft de code nog niet te snappen: de eerste les over Python legt hem uit. Voor nu telt alleen dat de cel de tekst tussen de aanhalingstekens laat zien.
        """),
        L.task("Klik in de codecel hieronder en verander `world` in je eigen naam. Voer de cel daarna uit: druk op **Shift + Enter**, of klik op de knop ▶ rechts van de cel."),
    ])
    return


@app.cell
def example_hello():
    print("Hello, world!")
    return


@app.cell(hide_code=True)
def text_after_run():
    mo.md(r"""
    Staat je naam direct onder de code? Dan heb je zojuist je eerste cel uitgevoerd. Alles wat je verandert, wordt vanzelf bewaard, in je eigen kopie van de les.
    """)
    return


@app.cell(hide_code=True)
def text_guess():
    mo.md(r"""
    ### Gok eerst

    In veel lessen gok je voordat je het antwoord ziet. Een gok wordt nooit beoordeeld. Door te gokken merk je wat je verwachtte, en juist daardoor blijft het antwoord hangen, ook als je gok fout was.
    """)
    return


@app.cell(hide_code=True)
def predict_mistake():
    guess_mistake = L.guess(
        "mistake",
        question="Wat denk je dat er in deze course gebeurt als een oefening nog niet goed is?",
        options={
            "points": "Je verliest punten.",
            "restart": "Je moet de les opnieuw beginnen.",
            "explain": "Je hoort wat er nog niet klopt, en je probeert het opnieuw.",
        },
        reveal="Er zijn geen punten, geen straffen en geen reeksen om vol te houden. Een check zegt wat er nog niet klopt en wat je eraan kunt doen, en dan probeer je het gewoon opnieuw. Eerst iets fout doen hoort bij leren.",
    )
    guess_mistake
    return (guess_mistake,)


@app.cell(hide_code=True)
def text_exercise_sum():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een oefening met een check

        De meeste oefeningen in deze course zijn code die je zelf schrijft. In de eerste lessen typ je die in een vak zoals hieronder. De check onder het vak leest wat je typte, voert het uit en zegt wat hij ziet.
        """),
        L.task("Verander de som in het vak, zodat de uitkomst 10 is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sum():
    box_sum = L.code_box("sum", lines=2)
    box_sum
    return (box_sum,)


@app.cell(hide_code=True)
def check_sum(box_sum):
    result_sum = L.check("sum", box_sum.value)
    result_sum
    return (result_sum,)


@app.cell(hide_code=True)
def hints_sum():
    L.hints(
        "sum",
        [
            "Lees wat de check zegt: wat geeft `3 + 4`, en hoeveel komt er nog tekort tot 10?",
            "Laat de `3` staan en maak van de `4` een groter getal.",
            "`3 + ___`: welk getal past in het gat?",
        ],
        why="3 + 7 is 10. Elke andere som die 10 geeft, is net zo goed, zoals `5 + 5` of `12 - 2`: de check kijkt naar de uitkomst.",
    )
    return


@app.cell(hide_code=True)
def text_choice():
    mo.md(r"""
    ### Oefening 2. Een keuzevraag

    Sommige vragen hebben antwoorden om uit te kiezen. Bij elk antwoord hoort een eigen uitleg, ook bij het goede, dus lees die, wat je ook kiest. Dit soort vraag zie je terug bovenaan elke les, om te checken wat je nodig hebt uit eerdere lessen.
    """)
    return


@app.cell(hide_code=True)
def choice_parts_cell():
    choice_parts = L.choice(
        "parts",
        question="In welk deel van een les maak je oefeningen met een check?",
        options={"why": "*Waarom*", "idea": "*Het idee*", "turn": "*Zelf doen*"},
        correct="turn",
        why={
            "why": "*Waarom* laat zien welk probleem de les oplost. Er valt nog niets te oefenen.",
            "idea": "*Het idee* legt uit, met een uitgewerkt voorbeeld dat je uitvoert en leest. Direct daarna ga je oefenen.",
            "turn": "De oefeningen met een check staan in *Zelf doen*, direct na het uitgewerkte voorbeeld. Zo oefen je in precies de vorm die je net zag.",
        },
    )
    choice_parts
    return (choice_parts,)


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Oefening 3. In je eigen woorden

    Sommige doelen kan een computer niet nakijken, zoals iets uitleggen. Dan schrijf je eerst je antwoord, leg je het naast een modelantwoord, en vink je aan welke punten in het jouwe staan.
    """)
    return


@app.cell(hide_code=True)
def explain_check_does():
    explain_check = L.explain(
        "check-does",
        question="Wat doet een check als je antwoord nog niet klopt? Schrijf één of twee zinnen.",
        model_answer="Een check kijkt naar de uitkomst van je antwoord. Klopt die nog niet, dan zegt hij **wat** er niet klopt, **waarom**, en **wat je nu kunt doen**. Hij geeft je niet zomaar het antwoord: daarvoor zijn de hints en de oplossing, voor als je vastzit.",
        points=[
            "hij zegt wat er nog niet klopt",
            "hij zegt wat je nu kunt doen",
            "hij geeft niet zomaar het antwoord",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_tools():
    mo.md(r"""
    ### Twee dingen die je soms nodig hebt

    **Iets uitproberen.** Wil je een stukje code proberen zonder de les te veranderen? Gebruik het **kladblok**. Klik op de twee kleine tellers linksonder in beeld (een kruisje en een driehoekje, elk met een getal). Onderaan opent een paneel; kies daar het tabblad *Scratchpad*. Wat je daar typt, komt nooit in de les terecht.

    **Opnieuw beginnen.** Ben je de draad kwijt, of wil je een les nog eens van voren af aan doen? Op de startpagina staat bij elke les die je hebt geopend **Opnieuw beginnen**. Dat zet de les terug zoals de course hem maakte. Wat je af had, blijft af.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elke les in de course werkt zoals deze. De volgende les gaat over vier woorden die je overal hoort: AI, machine learning, deep learning en generatieve AI. De eerste les over Python, *Je eerste programma*, legt de code uit die je in deze les uitvoerde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_parts, explain_check, result_sum):
    L.wrap_up(result_sum, choice_parts, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
