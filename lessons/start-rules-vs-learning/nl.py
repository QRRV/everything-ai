# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-rules-vs-learning. Route: course/route/lessen/stam-1-basis.yaml. Checks en ochtenden: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-rules-vs-learning", "nl")
    C = L.checks   # de ochtenden, en hoe de fouten van een regel geteld worden

    def line(mornings, t=None, drag=False):
        """De ochtenden op een lijn; met een regel krijgen de fouten een ring. Met drag zet een speelblok de regel
        waar je langs de lijn sleept. De titel zegt wat het plaatje laat zien, voor een schermlezer."""
        def listed(temps):
            return ", ".join(str(d) for d in temps[:-1]) + " en " + str(temps[-1]) if len(temps) > 1 else str(temps[0])

        title = (f"Twaalf ochtenden op een lijn van 0 tot 26 °C: met jas bij {listed([d for d, c in sorted(mornings) if c])} °C, "
                 f"zonder jas bij {listed([d for d, c in sorted(mornings) if not c])} °C.")
        if t is not None:
            wrong = [d for d, c in sorted(mornings) if (d < t) != c]
            title += (f" De regel jas als het kouder is dan {t} °C maakt {count(len(wrong))}"
                      + (f", met een ring bij {listed(wrong)} °C." if wrong else "."))
        return draw.threshold_line(
            mornings, t, lo=0, hi=26, axis="temperatuur 's ochtends (°C)", yes="jas", no="geen jas",
            rule=f"jas als het kouder is dan {t} °C" if t is not None else "", wrong="fout", title=title, drag=drag)

    def count(n):
        """Een aantal fouten, in woorden."""
        return f"{n} fout" if n == 1 else f"{n} fouten"

    def bars(mornings):
        """Hoeveel fouten elke regel maakt."""
        return draw.mistakes_per_rule(
            {t: C.mistakes(mornings, t) for t in C.RULES}, axis="regel: jas als het kouder is dan … °C",
            value="fouten", best="minste fouten")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over de vorige les.
    """)
    return


@app.cell(hide_code=True)
def quick_check_spam():
    quick_spam = L.choice(
        "quick-spam",
        question="Een spamfilter leerde van heel veel e-mails welke spam zijn. Er zit geen neural network in. Wat is de kleinste cirkel waar hij in past?",
        options={"ai": "AI", "ml": "Machine learning", "dl": "Deep learning"},
        correct="ml",
        why={
            "ai": "Hij past in AI, maar er is een kleinere cirkel: hij leerde van voorbeelden, dus het is machine learning. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "ml": "Hij leerde van voorbeelden, dus het is machine learning, en zonder neural network is het geen deep learning.",
            "dl": "Voor deep learning is een neural network met veel layers nodig, en dit filter heeft er geen. Kijk nog eens in [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
        },
    )
    quick_spam
    return (quick_spam,)


@app.cell(hide_code=True)
def quick_check_genai():
    quick_genai = L.choice(
        "quick-genai",
        question="Welke van deze is generatieve AI?",
        options={
            "poem": "Een app die een nieuw gedicht schrijft bij drie woorden die je geeft.",
            "birds": "Een app die zegt welke vogel je hoort.",
            "route": "Een routeplanner met regels die mensen schreven.",
        },
        correct="poem",
        why={
            "poem": "Hij maakt iets nieuws, een gedicht, dus het is generatieve AI.",
            "birds": "Hij noemt een vogel die er al zingt; hij maakt niets nieuws. Het is dus geen generatieve AI.",
            "route": "Mensen schreven de regels, dus het is AI, maar geen machine learning, en dus ook geen generatieve AI.",
        },
    )
    quick_genai
    return (quick_genai,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je wilt dat een computer iets voor je beslist: of een e-mail spam is, of er een kat op een foto staat. Eén manier is de regels zelf opschrijven. Voor simpele dingen werkt dat, maar voor spam heb je duizenden regels nodig, en spammers veranderen hun e-mails elke week. Hoe komt een computer anders aan een regel?

    **Een voorbeeld uit het dagelijks leven.** Je leert een kind niet wat een hond is door regels te geven: vier poten, een staart en een vacht passen ook bij een kat. Je wijst honden aan en zegt "hond", en katten en zegt "kat". Na genoeg voorbeelden vindt het kind het verschil zelf. Voorbeelden geven is makkelijk; de regel opschrijven is moeilijk.
    """)
    return


@app.cell(hide_code=True)
def predict_can_find():
    guess_can_find = L.guess(
        "can-find",
        question="Je geeft een computer twaalf ochtenden: de temperatuur, en of je een jas droeg. Denk je dat hij een regel voor \"jas of geen jas\" kan vinden zonder dat jij hem een regel geeft?",
        options={
            "yes": "Ja, alleen uit de voorbeelden.",
            "no": "Nee, iemand moet hem de regel geven.",
            "understand": "Alleen als hij begrijpt wat kou is.",
        },
        reveal="**Ja.** In deze les zie je hoe: de computer probeert regels en houdt de regel over die het best bij de voorbeelden past. Hij weet niet wat een jas is, of wat kou is. Hij ziet alleen getallen en antwoorden.",
    )
    guess_can_find
    return (guess_can_find,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **voorbeeld** | één geval in de data; hier één ochtend, met de temperatuur en of je een jas droeg |
    | **regel** | wat het antwoord geeft voor een geval; hier "jas als het kouder is dan 14 °C" |
    | **leren van data** | in plaats van dat een mens de regel opschrijft, haalt de computer hem zelf uit voorbeelden; dit is het idee achter alle machine learning |

    ### Jouw twaalf ochtenden

    Twaalf ochtenden lang schreef je de temperatuur op en of je een jas droeg. Een dichte stip is een ochtend met jas, een open stip een ochtend zonder.
    """)
    return


@app.cell(hide_code=True)
def picture_mine():
    line(C.MINE)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    Een regel van de soort "jas als het kouder is dan T °C" deelt de lijn in tweeën: links een jas, rechts geen jas. Elke ochtend aan de verkeerde kant is een **fout**, en krijgt een ring. Verschuif de schuifbalk, of sleep langs de lijn, om T te zetten, en let op de ringen.
    """)
    return


@app.cell(hide_code=True)
def explore_mine_cell():
    def _frame(t):
        return (line(C.MINE, t, drag=True).text,
                f"""Deze regel maakt <span class="c-loss">{count(C.mistakes(C.MINE, t))}</span> op jouw ochtenden.""")

    explore_mine = L.explore("mine", "T", _frame, start=20, low=0, high=26, step=1, role="param", task="Zoek een regel die op jouw ochtenden geen fouten maakt.")
    explore_mine
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met T op 14 of op 15 °C maakt de regel geen fouten op jouw ochtenden. Dat was jij die de regel schreef. Nu de computer. Die volgt een recept, en heeft niets anders nodig dan de voorbeelden.

    ### Recept

    1. Neem een regel: "jas als het kouder is dan T °C", te beginnen bij T = 0.
    2. Tel de fouten van die regel op de voorbeelden.
    3. Doe hetzelfde voor elke T, tot en met 26.
    4. Houd de regel met de minste fouten over.

    Dit is een vereenvoudigde manier van leren. Hier werkt hij omdat er maar 27 regels zijn om te proberen. Later in de course zie je hoe een computer een regel stap voor stap verbetert als er veel te veel regels zijn om ze allemaal te proberen.
    """)
    return


@app.cell(hide_code=True)
def worked_example():
    mo.md(rf"""
        ### Uitgewerkt voorbeeld

        De computer volgt het recept op jouw twaalf ochtenden.

        1. **T = 0**: "jas als het kouder is dan 0 °C". Zo koud was het geen enkele ochtend, dus de regel zegt elke keer "geen jas". Je droeg 6 ochtenden een jas: **{count(C.mistakes(C.MINE, 0))}**.
        2. **T = 10**: de regel zegt "jas" bij 4, 7 en 9 °C, en "geen jas" bij de rest. Fout bij 10, 12 en 13 °C: **{count(C.mistakes(C.MINE, 10))}**.
        3. **Elke andere T**: de computer telt op dezelfde manier. De staven laten alle 27 regels zien:
        """)
    return


@app.cell(hide_code=True)
def worked_example_bars():
    bars(C.MINE)
    return


@app.cell(hide_code=True)
def worked_example_end():
    _found = C.learn(C.MINE)
    mo.md(rf"""
        4. **Minste fouten**: de regels voor **{_found[0]} °C** en **{_found[-1]} °C** maken allebei {count(C.mistakes(C.MINE, _found[0]))}. De computer houdt de eerste over: jas als het kouder is dan {_found[0]} °C.
        """)
    return


@app.cell(hide_code=True)
def text_exercise_neighbour():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De ochtenden van je buur

    Je buur schreef ook twaalf ochtenden op, met de temperatuur en of er een jas aan ging. Nu doe jij wat de computer doet, op de ochtenden van je **buur**. De schuifbalk begint bij je eigen regel, 14 °C.
    """)
    return


@app.cell(hide_code=True)
def exercise_neighbour():
    explore_neighbour = L.explore("neighbour", "T", lambda t: line(C.NEIGHBOUR, t, drag=True).text, start=14, low=0, high=26, step=1,
                                  role="param", task="Zet de schuifbalk op de regel die de minste fouten maakt op de ochtenden van je buur.")
    explore_neighbour
    return (explore_neighbour,)


@app.cell(hide_code=True)
def check_neighbour(explore_neighbour):
    result_neighbour = L.check("neighbour", explore_neighbour)
    result_neighbour
    return (result_neighbour,)


@app.cell(hide_code=True)
def hints_neighbour():
    L.hints(
        "neighbour",
        [
            "Verschuif de schuifbalk telkens één graad en tel elke keer de ringen.",
            "Je buur droeg een jas bij 17 en 18 °C. Een regel die die goed heeft, moet boven de 18 uitkomen.",
            "Vergelijk 19 en 20 °C. Welke maakt minder fouten?",
        ],
        why="**19 °C**: jas als het kouder is dan 19 °C. Alleen de ochtend van 16 °C is fout: toen droeg je buur geen jas. Geen enkele regel van deze soort krijgt die ochtend goed zonder andere ochtenden fout te krijgen.",
    )
    return


@app.cell(hide_code=True)
def text_learn():
    mo.md(r"""
    ### Laat de computer het doen

    Je deed net met de hand wat de computer doet. Druk op de knop om de computer het recept te laten volgen op de ochtenden van je buur.
    """)
    return


@app.cell(hide_code=True)
def learn_button_cell():
    learn_button = mo.ui.run_button(label="Leer van de voorbeelden")
    learn_button
    return (learn_button,)


@app.cell(hide_code=True)
def learned_neighbour(learn_button):
    mo.stop(not learn_button.value)
    _found = C.learn(C.NEIGHBOUR)
    mo.vstack([
        bars(C.NEIGHBOUR),
        line(C.NEIGHBOUR, _found[0]),
        mo.md(f"De computer houdt **jas als het kouder is dan {_found[0]} °C** over, met {count(C.mistakes(C.NEIGHBOUR, _found[0]))}. Hetzelfde recept als net, andere voorbeelden, een andere regel."),
    ])
    return


@app.cell(hide_code=True)
def text_choice():
    mo.md(r"""
    ### Oefening 2. Wat veranderde er?
    """)
    return


@app.cell(hide_code=True)
def choice_changed_cell():
    choice_changed = L.choice(
        "what-changed",
        question="Jouw regel is 14 °C, die van je buur 19 °C. Wat veranderde er om de regel van je buur te krijgen?",
        options={
            "examples": "Alleen de voorbeelden. Het recept bleef hetzelfde.",
            "recipe": "Het recept: de computer had een nieuwe manier van leren nodig.",
            "understand": "De computer begreep dat je buur het sneller koud heeft.",
        },
        correct="examples",
        why={
            "examples": "De computer volgde hetzelfde recept op andere voorbeelden, en vond een andere regel. Dat is de kracht van leren van data: voor een nieuwe situatie heb je alleen nieuwe voorbeelden nodig.",
            "recipe": "Het recept was precies hetzelfde: elke regel proberen, de fouten tellen, de minste overhouden. Alleen de voorbeelden waren anders.",
            "understand": "De computer begrijpt niets van kou of jassen. Hij telde alleen fouten op andere voorbeelden, en die gaven toevallig een andere regel.",
        },
    )
    choice_changed
    return (choice_changed,)


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Oefening 3. Waarom leren?
    """)
    return


@app.cell(hide_code=True)
def explain_why_learn():
    explain_why = L.explain(
        "why-learn",
        question="Een spamfilter beslist bij elke e-mail of het spam is. Waarom laat je hem liever leren van voorbeelden dan dat je zelf zijn regels schrijft? Schrijf één of twee zinnen.",
        model_answer="Spam komt in veel te veel vormen om te vangen met regels die je zelf schrijft, en spammers veranderen hun e-mails steeds. Om te leren heb je alleen e-mails nodig waar spam of geen spam bij staat, en als spam verandert, leert het filter opnieuw van nieuwe voorbeelden.",
        points=[
            "spam heeft te veel vormen om er regels voor te schrijven",
            "spam verandert steeds, en met nieuwe voorbeelden leert het filter opnieuw",
            "om te leren heb je alleen e-mails nodig waar spam of geen spam bij staat",
        ],
    )
    explain_why
    return (explain_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) was de eerste vraag van het recept: schreven mensen de regels, of vond de computer ze in voorbeelden? Nu zag je de tweede manier aan het werk. Fouten tellen is de eerste vorm van een getal dat je steeds weer tegenkomt: een getal dat zegt hoe slecht een regel is. Hier leerde de computer één getal, T. De modellen die je later in de course bouwt, leren duizenden tot miljoenen getallen, maar het idee blijft hetzelfde: leer de regel uit voorbeelden, en meet hoe goed hij is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_changed, explain_why, result_neighbour):
    L.wrap_up(result_neighbour, choice_changed, explain_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
