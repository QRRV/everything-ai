# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-ai-ml-genai. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("start-ai-ml-genai", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over de vorige les. Ze houden niets tegen: gaat er een fout, dan zegt de uitleg waar je nog eens kunt kijken.
    """)
    return


@app.cell(hide_code=True)
def quick_check_run():
    quick_run = L.choice(
        "quick-run",
        question="Hoe voer je een codecel uit?",
        options={
            "keys": "Druk op **Shift + Enter**, of klik op de knop ▶ naast de cel.",
            "reopen": "Sluit de les en open hem opnieuw.",
            "word": "Typ `run` onderaan de cel.",
        },
        correct="keys",
        why={
            "keys": "**Shift + Enter** voert de cel uit waar je in staat, en de uitkomst verschijnt direct onder de code.",
            "reopen": "Als je een les opent, worden de cellen uitgevoerd die bij de les horen, maar niet de cellen die jij veranderde. Eén cel uitvoeren doe je met **Shift + Enter**. Kijk nog eens in [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).",
            "word": "`run` is geen Python-woord, dus Python zou stoppen met een fout. Een cel voer je uit met **Shift + Enter**. Kijk nog eens in [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl).",
        },
    )
    quick_run
    return (quick_run,)


@app.cell(hide_code=True)
def quick_check_not_yet():
    quick_not_yet = L.choice(
        "quick-not-yet",
        question="Een check zegt *Nog niet*. Wat kun je het beste doen?",
        options={
            "read": "Lezen wat de check zegt, je antwoord aanpassen en opnieuw kijken.",
            "solution": "Meteen de oplossing openen.",
            "restart": "De les opnieuw beginnen.",
        },
        correct="read",
        why={
            "read": "Een check zegt wat er nog niet klopt en wat je eraan kunt doen. Daarmee opnieuw proberen is hoe je het meest leert.",
            "solution": "De oplossing is er voor als je echt vastzit. Probeer eerst wat de check voorstelt, en daarna de hints, één voor één.",
            "restart": "Eén fout antwoord is geen reden om opnieuw te beginnen. Lees wat de check zegt en probeer het nog eens.",
        },
    )
    quick_not_yet
    return (quick_not_yet,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    "AI schrijft je e-mails." "Machine learning spoort fraude op." "Een deep learning-model vond de tumor." Je hoort deze woorden elke dag, en vaak alsof ze hetzelfde betekenen. Dat doen ze niet. Als iemand zegt dat een app "AI gebruikt", weet je nog niet wat die app doet of hoe hij gemaakt is.

    **Een voorbeeld uit het dagelijks leven.** Voertuig, auto, elektrische auto. Elke elektrische auto is een auto, en elke auto is een voertuig, maar een fiets is een voertuig en geen auto. Zegt iemand alleen "voertuig", dan weet je niet welk voertuig er bedoeld wordt. De vier woorden in deze les werken net zo.
    """)
    return


@app.cell(hide_code=True)
def predict_learning():
    guess_learning = L.guess(
        "every-ai-learns",
        question="Denk je dat elk AI-systeem wat het doet uit voorbeelden heeft geleerd?",
        options={
            "yes": "Ja, dat maakt het juist AI.",
            "no": "Nee, sommige AI volgt regels die mensen schreven.",
            "new": "Alleen de AI van de laatste paar jaar.",
        },
        reveal="**Nee.** Veel AI volgt regels die mensen schreven, zoals een schaakprogramma dat stellingen een score geeft met regels van schakers. Leren van voorbeelden is *één* manier om AI te bouwen, en die heet machine learning. Tegenwoordig is het verreweg de meest gebruikte manier, en daarom lopen de woorden zo makkelijk door elkaar.",
    )
    guess_learning
    return (guess_learning,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    De vier woorden zijn cirkels in elkaar. Elke cirkel is een deel van de cirkel eromheen.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **kunstmatige intelligentie** (AI) | het vakgebied dat computersystemen bouwt voor taken waarvan we dachten dat er menselijke intelligentie voor nodig was, zoals beelden herkennen, taal begrijpen of plannen |
    | **machine learning** (ML) | het deel van AI waarin de computer zelf een regel vindt in voorbeelden, in plaats van dat een mens die regel opschrijft |
    | **deep learning** | machine learning met *neural networks* die veel *layers* hebben; later in de course bouw je er zelf een |
    | **generatieve AI** | AI die iets nieuws maakt, zoals tekst, beelden, geluid of code, in plaats van alleen te sorteren of te voorspellen |

    Na deze les zeggen we gewoon AI en ML.
    """)
    return


@app.cell(hide_code=True)
def picture_circles():
    mo.Html("""
    <figure class="ea-figure" style="margin:0.5rem 0 1rem;max-width:40rem">
    <svg viewBox="0 0 640 370" role="img" aria-labelledby="circles-title" style="width:100%;height:auto;font-family:var(--ea-font-ui)">
      <title id="circles-title">Vier cirkels in elkaar: AI, machine learning, deep learning en generatieve AI</title>
      <g style="stroke:var(--ea-ink-2);stroke-width:1.5">
        <ellipse cx="320" cy="190" rx="312" ry="170" style="fill:var(--ea-ink);fill-opacity:0.03"/>
        <ellipse cx="320" cy="228" rx="248" ry="130" style="fill:var(--ea-ink);fill-opacity:0.05"/>
        <ellipse cx="320" cy="266" rx="178" ry="92" style="fill:var(--ea-ink);fill-opacity:0.07"/>
        <ellipse cx="320" cy="303" rx="88" ry="55" style="fill:var(--ea-ink);fill-opacity:0.10"/>
      </g>
      <g style="fill:var(--ea-ink);text-anchor:middle">
        <text x="320" y="50" style="font-weight:680;font-size:19px">AI</text>
        <text x="320" y="72" style="font-size:14px;fill:var(--ea-ink-2)">bijvoorbeeld een routeplanner</text>
        <text x="320" y="128" style="font-weight:680;font-size:19px">Machine learning</text>
        <text x="320" y="150" style="font-size:14px;fill:var(--ea-ink-2)">bijvoorbeeld een schatting van huizenprijzen</text>
        <text x="320" y="203" style="font-weight:680;font-size:19px">Deep learning</text>
        <text x="320" y="225" style="font-size:14px;fill:var(--ea-ink-2)">bijvoorbeeld ontgrendelen met je gezicht</text>
        <text x="320" y="292" style="font-weight:680;font-size:17px">Generatieve AI</text>
        <text x="320" y="315" style="font-size:13px;fill:var(--ea-ink-2)">bijvoorbeeld een</text>
        <text x="320" y="332" style="font-size:13px;fill:var(--ea-ink-2)">beeldgenerator</text>
      </g>
    </svg>
    <figcaption class="ea-needs">Elk systeem staat in de kleinste cirkel waar het in past, en daarmee ook in elke cirkel eromheen.</figcaption>
    </figure>
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Zo worden de cirkels meestal getekend. In het echt zijn de grenzen wat vager, maar dit beeld is genoeg voor de hele course.
        """),
        L.watch_out("In het nieuws bedoelt men met \"AI\" vaak alleen generatieve AI: een chatassistent of een beeldgenerator. In deze course is AI de hele grote cirkel."),
        mo.md(r"""
        ### Recept

        Om de kleinste cirkel te vinden waar een systeem in past, stel je drie vragen, in deze volgorde:

        1. **Schreven mensen de regels, of vond de computer ze in voorbeelden?** Mensen schreven ze: stop, het staat in de cirkel AI.
        2. **Is het een neural network met veel layers?** Nee: stop, het is machine learning.
        3. **Maakt het iets nieuws, zoals tekst, beelden of geluid?** Nee: het is deep learning. Ja: het is generatieve AI.

        Sommige echte systemen mengen regels en leren. Kijk dan naar het deel dat het meeste werk doet.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    **Foto's zoeken.** Je telefoon vindt elke foto van je hond. Hij leerde hoe een hond eruitziet van miljoenen foto's met "hond" of "geen hond" erbij, met een neural network van veel layers.

    1. **Regels of voorbeelden?** Hij leerde van foto's met "hond" of "geen hond" erbij: voorbeelden. Hij staat dus binnen machine learning.
    2. **Een neural network met veel layers?** Ja, dat staat in de beschrijving. Hij staat dus binnen deep learning.
    3. **Iets nieuws?** Nee, hij vindt foto's die er al zijn. Het is dus geen generatieve AI.

    De kleinste cirkel is **deep learning**. Foto's zoeken staat ook in de twee cirkels eromheen: het is machine learning, en het is AI.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_circles():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Vijf systemen, vier cirkels

        Elk systeem hieronder krijgt één cirkel. Gebruik het recept: de drie vragen, in volgorde.
        """),
        L.task("Zet elk systeem in de **kleinste** cirkel waar het in past."),
    ])
    return


@app.cell(hide_code=True)
def exercise_circles():
    sort_circles = L.sort("circles")
    sort_circles
    return (sort_circles,)


@app.cell(hide_code=True)
def check_circles(sort_circles):
    result_circles = L.check("circles", sort_circles)
    result_circles
    return (result_circles,)


@app.cell(hide_code=True)
def hints_circles():
    L.hints(
        "circles",
        [
            "Begin bij elk systeem met vraag 1: welke volgen regels die mensen schreven? Die blijven in de grote cirkel AI.",
            "Vraag 2: bij welke van de andere staat *neural network*? Die zonder blijven bij machine learning.",
            "Vraag 3: de chatassistent en de muziekmaker maken iets nieuws. De planten-app noemt alleen een plant.",
        ],
        why="Planten-app: **deep learning**. Spamfilter: **machine learning**. Chatassistent: **generatieve AI**. Schaakprogramma: **AI**. Muziekmaker: **generatieve AI**.",
    )
    return


@app.cell(hide_code=True)
def text_inside():
    mo.md(r"""
    ### Oefening 2. In elkaar
    """)
    return


@app.cell(hide_code=True)
def choice_inside_cell():
    choice_inside = L.choice(
        "inside",
        question="Welke zin klopt?",
        options={
            "all-learn": "Elk AI-systeem leert van voorbeelden.",
            "genai-ml": "Elk systeem met generatieve AI is ook machine learning.",
            "same": "Deep learning en generatieve AI betekenen hetzelfde.",
        },
        correct="genai-ml",
        why={
            "all-learn": "Niet elk AI-systeem leert: het schaakprogramma volgt regels die mensen schreven, en is ook AI. Machine learning is het deel van AI dat van voorbeelden leert.",
            "genai-ml": "Generatieve AI is de kleinste cirkel. Die ligt binnen deep learning, en dat ligt binnen machine learning, dus ook generatieve AI leerde van voorbeelden.",
            "same": "Generatieve AI is het deel van deep learning dat iets nieuws maakt. De planten-app is deep learning, maar maakt niets nieuws.",
        },
    )
    choice_inside
    return (choice_inside,)


@app.cell(hide_code=True)
def text_ask():
    mo.md(r"""
    ### Oefening 3. De goede vragen stellen
    """)
    return


@app.cell(hide_code=True)
def explain_ask():
    explain_questions = L.explain(
        "ask",
        question="In een nieuwsbericht staat: \"Deze app gebruikt AI.\" Welke vragen zou je stellen om te ontdekken in welke cirkel hij hoort? Schrijf ze op.",
        model_answer="**Schreven mensen de regels, of leerde hij ze van voorbeelden?** Als hij leerde: **is het een neural network met veel layers?** En: **maakt hij iets nieuws, zoals tekst of plaatjes, of sorteert en voorspelt hij?**",
        points=[
            "je vraagt of hij van voorbeelden leerde of regels volgt die mensen schreven",
            "je vraagt of het een neural network met veel layers is",
            "je vraagt of hij iets nieuws maakt",
        ],
    )
    explain_questions
    return (explain_questions,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Zo werkt een les](/course/open/start-how-this-course-works?lang=nl) leerde je de weg in een les; vanaf nu gaat elke les over AI zelf. De volgende les zoomt in op de eerste vraag van het recept: regels schrijven tegenover leren van voorbeelden. Later in de course bouw je elke cirkel zelf: eerst iets dat leert van voorbeelden, dan je eigen neural network, en aan het eind van de hoofdroute een klein generatief model dat tekst schrijft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_inside, explain_questions, result_circles):
    L.wrap_up(result_circles, choice_inside, explain_questions)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
