# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-research-cycle. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-research-cycle", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_review():
    quick_review = L.choice(
        "quick-review",
        question="Wie doet de peer review van een paper?",
        options={
            "right": "Een paar andere onderzoekers uit hetzelfde vakgebied",
            "journalists": "De journalisten die over de paper schrijven",
            "authors": "De auteurs zelf, voordat ze hem online zetten",
        },
        correct="right",
        why={
            "journalists": "Journalisten schrijven over een paper, vaak nadat hij verschenen is; ze beoordelen hem niet voor een conferentie of tijdschrift. Zie [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).",
            "authors": "Je eigen werk controleren is geen peer review: de peers zijn andere onderzoekers. Zie [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).",
            "right": "Een conferentie of tijdschrift stuurt de paper naar een paar experts, de peers, die adviseren of hij gepubliceerd wordt en wat er beter moet.",
        },
    )
    quick_review
    return (quick_review,)


@app.cell(hide_code=True)
def quick_check_paper():
    quick_paper = L.choice(
        "quick-paper",
        question="Wat staat er in een paper?",
        options={
            "results": "Alleen de resultaten, zodat anderen ze kunnen gebruiken",
            "news": "Een korte samenvatting van een resultaat, voor het nieuws",
            "right": "Een onderzoek, van de vraag tot de resultaten",
        },
        correct="right",
        why={
            "results": "Alleen resultaten kun je niet controleren: een paper zegt ook wat de vraag was en wat de onderzoekers deden. Zie [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).",
            "news": "Dat is een nieuwsbericht of een persbericht, iemands samenvatting van een paper. Zie [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).",
            "right": "En hoe het zich verhoudt tot eerder werk, zodat anderen het kunnen lezen, controleren en erop voortbouwen.",
        },
    )
    quick_paper
    return (quick_paper,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een paper doet verslag van een onderzoek. Maar hoe gaat dat onderzoek zelf? Stel dat je denkt dat één zin een chatbot betere antwoorden laat geven. Hoe kom je erachter of dat klopt, op een manier die anderen kunnen controleren en vertrouwen? Als je weet hoe onderzoekers dat doen, lees je een paper beter: wat wilden ze weten, wat verwachtten ze, en wat deden ze? En later in deze course doe je zelf experimenten, tot het herhalen van het resultaat van een paper toe.

    **Een voorbeeld uit het dagelijks leven.** Een arts die denkt dat je griep hebt en het zeker wil weten, zegt dat niet zomaar. Die laat een test doen waarvan de uitslag kan laten zien dat het iets anders is.
    """)
    return


@app.cell(hide_code=True)
def predict_plan():
    guess_plan = L.guess(
        "plan",
        question="Onderzoekers willen weten of een chatbot verhaaltjessommen beter oplost als \"Let's think step by step.\" (\"Laten we stap voor stap denken.\") na elke som staat. Welk plan vertelt ze het meest?",
        options={
            "until": "Sommen blijven proberen tot ze er een paar vinden waar de zin helpt, en die laten zien",
            "right": "200 sommen, elk met en zonder de zin; tel de goede antwoorden",
            "ask": "De chatbot vragen of de zin hem helpt om beter na te denken",
        },
        reveal="**200 sommen, elk met en zonder de zin.** Dan is de zin het enige verschil tussen de twee tellingen, dus een duidelijk verschil in goede antwoorden komt daarvandaan. Wie blijft proberen tot het helpt, vindt wel een paar sommen waar het helpt, ook als het gemiddeld niets doet. En wat de chatbot over zichzelf zegt, meet niets: hij kan in beide gevallen ja zeggen.",
    )
    guess_plan
    return (guess_plan,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Onderzoek gaat rond in een cyclus. Het begint met een **onderzoeksvraag**: wat wil je te weten komen? "Lost een chatbot verhaaltjessommen beter op als je hem vraagt stap voor stap te denken?"

        Een vraag kun je nog niet testen. Dus maak je er een **hypothese** van: een precies antwoord dat je verwacht, zo geformuleerd dat een test het kan weerleggen. "Met de zin erbij heeft de chatbot meer van deze 200 sommen goed." Ze zegt wat je verwacht, waarop, en wat je gaat tellen. Vóór de test zeg je ook welke uitkomst haar weerlegt: evenveel goede antwoorden met de zin als zonder, of minder.

        Dan het **experiment**: een test van de hypothese. Je verandert één ding, houdt al het andere gelijk, en meet. Hier verander je alleen de zin; de sommen, de chatbot en de manier waarop je de antwoorden nakijkt, blijven gelijk. Verander je twee dingen tegelijk, dan kun je niet zien welk van de twee het verschil maakt. Een klein verschil kan ook toeval zijn: een chatbot geeft niet altijd hetzelfde antwoord. Later in de course leer je hoe groot een verschil moet zijn.

        Tot slot schrijf je het op als paper, met wat je deed en vond, zodat anderen het kunnen controleren en herhalen. Een uitkomst roept nieuwe vragen op, en de cyclus begint opnieuw.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **onderzoeksvraag** | wat je te weten wilt komen |
        | **hypothese** | een precies antwoord dat je op de vraag verwacht, zo geformuleerd dat een test het kan weerleggen |
        | **experiment** | een test van een hypothese: je verandert één ding, houdt de rest gelijk, en meet wat er gebeurt |
        """),
        L.watch_out("Een uitkomst die met de hypothese klopt, bewijst haar niet voorgoed: ze heeft deze test doorstaan, op deze sommen, met deze chatbot. Een uitkomst ertegen uit een eerlijke test laat zien dat ze niet klopt, in elk geval voor deze test. Daarom is een goed experiment er een dat ook tegen je had kunnen uitvallen."),
        mo.md(r"""
        De figuur: de onderzoekscyclus.
        """),
        mo.Html(C.cycle({"title": "De onderzoekscyclus",
                         "boxes": [("Vraag", "wat wil je te weten komen?"),
                                   ("Hypothese", "een precies antwoord dat je verwacht, dat een test kan weerleggen"),
                                   ("Experiment", "verander één ding, houd de rest gelijk, en meet"),
                                   ("Paper", "schrijf op wat je deed en vond, zodat anderen het kunnen controleren")],
                         "again": "nieuwe vragen",
                         "alt": "Vier vakken in een vierkant, met pijlen rondom. 1, Vraag: wat wil je te weten komen? 2, Hypothese: een precies antwoord dat je verwacht, dat een test kan weerleggen. 3, Experiment: verander één ding, houd de rest gelijk, en meet. 4, Paper: schrijf op wat je deed en vond, zodat anderen het kunnen controleren. Een pijl gaat van de paper terug naar de vraag, met erbij: nieuwe vragen."})),
        mo.md(r"""
        De pijl terug doet ertoe: een uitkomst roept nieuwe vragen op, en andere onderzoekers gaan verder waar een paper ophoudt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om van een vraag een experiment te maken:

    1. Schrijf de vraag op: wat wil je te weten komen?
    2. Maak er een hypothese van: een precieze bewering over wat je verwacht, over iets wat je kunt tellen of meten.
    3. Zeg vóór het experiment welke uitkomst de hypothese zou weerleggen.
    4. Ontwerp het experiment: verander één ding, houd al het andere gelijk, en meet.
    5. Schrijf op wat je deed en vond, ook als het tegen je uitviel, zodat anderen het kunnen controleren en herhalen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een echt onderzoek uit 2022, van Takeshi Kojima en vier co-auteurs, met een generatieve AI voor tekst, van het soort achter de chatbots van nu. De paper zelf gebruikt deze kopjes niet; hier staat het onderzoek in de stappen van het recept.

    1. **Vraag:** lost zo'n programma verhaaltjessommen beter op als je hem vraagt stap voor stap te denken?
    2. **Hypothese:** met "Let's think step by step." na elke som heeft het programma meer van een vaste set van 600 verhaaltjessommen goed.
    3. **Weerlegd als:** het programma met de zin evenveel goed heeft als zonder, of minder.
    4. **Experiment:** dezelfde 600 sommen en hetzelfde programma, een keer met elke som alleen en een keer met de zin erbij. Elk antwoord wordt goed of fout gerekend, en de goede antwoorden worden beide keren geteld.
    5. **Paper:** ze zetten het in mei 2022 op arXiv, en het kwam datzelfde jaar door de peer review van NeurIPS. Het belangrijkste programma dat ze testten, had zonder de zin ongeveer 18% van de sommen goed, en met de zin ongeveer 79%.

    De hypothese heeft deze test doorstaan. En de uitkomst riep nieuwe vragen op, die andere onderzoekers oppakten: waarom helpt de zin, en is er een nog betere?
    """)
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke stap?

    Zes zinnen, door elkaar, uit één onderzoek dat een chatbot in het Nederlands en in het Engels vergelijkt. Zet elke zin bij zijn stap van de cyclus.
    """)
    return


@app.cell(hide_code=True)
def exercise_steps():
    sort_steps = L.sort("cycle-steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("cycle-steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "cycle-steps",
        [
            "Vraag bij elke zin: is het iets wat ze willen weten, een precieze verwachting, iets wat ze doen, of iets wat ze achteraf delen?",
            "Kijk nog eens naar de figuur: de pijl vanaf de paper gaat terug naar de vraag. Welke zin komt na de uitkomst?",
            "Twee zinnen vragen iets wat ze willen weten, een daarvan pas na de uitkomst. Precies één zin is een precieze verwachting die een telling kan weerleggen. Twee zinnen beschrijven wat ze doen om die te testen, en één wat ze daarna doen.",
        ],
        why="**Vraag:** slechter in het Nederlands dan in het Engels; even groot verschil voor andere talen. **Hypothese:** minder van de 300 testvragen goed in het Nederlands. **Experiment:** een vertaler vertaalt de vragen; dezelfde chatbot beantwoordt ze in beide talen. **Paper:** opschrijven wat ze deden en vonden, en het online zetten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_testable():
    mo.md(r"""
    ### Oefening 2. Kan het fout zijn?
    """)
    return


@app.cell(hide_code=True)
def choice_testable_cell():
    choice_testable = L.choice(
        "testable",
        question="Welke van deze zinnen is een hypothese die een experiment kan weerleggen?",
        options={
            "understands": "De chatbot begrijpt echt de betekenis van elke vraag die hij beantwoordt",
            "future": "Ooit zullen chatbots beter zijn dan mensen in alles wat ze doen",
            "right": "Als de chatbot in één woord moet antwoorden, heeft hij minder van deze 200 vragen fout",
            "maybe": "In het Nederlands maakt de chatbot misschien af en toe een fout, of misschien niet, afhankelijk van de vraag",
        },
        correct="right",
        why={
            "understands": "Welke telling zou \"begrijpt echt\" laten zien? Zoals het er staat, kan geen uitkomst het weerleggen, dus kan geen experiment het testen.",
            "future": "\"Ooit\" hoeft nooit te komen: geen uitkomst van vandaag kan het weerleggen.",
            "maybe": "Elke uitkomst past bij \"misschien wel, misschien niet\": het kan niet fout zijn, dus het zegt niets.",
            "right": "Ze zegt wat er verwacht wordt, op welke vragen, en wat je telt. Heeft de chatbot in één woord evenveel fout, of meer, dan is de hypothese weerlegd.",
        },
    )
    choice_testable
    return (choice_testable,)


@app.cell(hide_code=True)
def text_exercise_design():
    mo.md(r"""
    ### Oefening 3. Ontwerp een experiment
    """)
    return


@app.cell(hide_code=True)
def explain_design():
    explain_check = L.explain(
        "design",
        question="Iemands hypothese: \"Een chatbot heeft meer vragen goed als ze zonder spelfouten geschreven zijn.\" Ontwerp een experiment met het recept. Schrijf op wat je verandert, wat je gelijk houdt, wat je telt, en welke uitkomst de hypothese zou weerleggen.",
        model_answer="**Veranderen:** alleen de spelling: elke vraag een keer met spelfouten en een keer zonder. **Gelijk houden:** de vragen zelf, de chatbot, en de manier waarop de antwoorden worden nagekeken. **Tellen:** hoeveel vragen hij in elke versie goed heeft. **Weerlegd als:** de versie zonder spelfouten evenveel goed heeft als de versie met, of minder.",
        points=[
            "je verandert alleen de spelling, bij dezelfde vragen",
            "je houdt de chatbot en het nakijken gelijk",
            "je telt de goede antwoorden in beide versies",
            "je zegt welke uitkomst de hypothese weerlegt",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl) liet zien hoe onderzoek gedeeld wordt; nu weet je wat er vóór de paper komt: een vraag, een hypothese die fout kan zijn, en een experiment dat haar test. In het onderzoeksdeel van de course ontwerp je zelf zulke experimenten, en aan het eind herhaal je het resultaat van een paper. Met deze les is het basisdeel van de course af. Latere lessen bouwen erop voort, zoals *Eén ding tegelijk veranderen* en *Is het verschil echt?*. De volgende les, *Rekenregels voor afgeleiden*, begint het gevorderde deel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_testable, explain_check, result_steps):
    L.wrap_up(result_steps, choice_testable, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
