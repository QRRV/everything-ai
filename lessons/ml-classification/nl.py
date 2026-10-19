# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-classification. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-classification", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_photos():
    quick_photos = L.choice(
        "quick-photos",
        question="Een model leert van 5.000 foto's die mensen als kat of hond markeerden. Welke soort leren is dat?",
        options={
            "right": "Supervised learning",
            "unsupervised": "Unsupervised learning",
            "reinforcement": "Reinforcement learning",
        },
        correct="right",
        why={
            "unsupervised": "Hier heeft elke foto een antwoord: mensen markeerden hem. Zie stap 1 van het recept in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "reinforcement": "Het model doet niets en krijgt geen rewards; het leert van voorbeelden met antwoorden. Zie het recept in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "right": "Elke foto heeft een label dat mensen eraan gaven: kat of hond.",
        },
    )
    quick_photos
    return (quick_photos,)


@app.cell(hide_code=True)
def quick_check_next_word():
    quick_next_word = L.choice(
        "quick-next-word",
        question="Een language model leert het volgende woord van een tekst te voorspellen. Waar komt elk label vandaan?",
        options={
            "people": "Mensen schreven elk volgend woord op als label",
            "rewards": "Uit rewards voor goede woorden",
            "right": "Uit de tekst zelf",
        },
        correct="right",
        why={
            "people": "Niemand hoeft deze labels te schrijven. Waar staat het volgende woord al? Zie *Het idee* in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "rewards": "Rewards horen bij reinforcement learning. Wat voorspelt het model hier? Zie *Het idee* in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "right": "Verberg het woord dat erna komt, en dat woord is het label: self-supervised learning.",
        },
    )
    quick_next_word
    return (quick_next_word,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De tabel met leerlingen in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) had twee kolommen die je kon voorspellen: het cijfer, en of ze slaagden. Had een leerling een 6,5 en voorspelt een model 6,3, dan is dat bijna goed. Maar is het antwoord "yes" en voorspelt het model "no", dan is dat gewoon fout: er zit niets tussen. Dat zijn twee soorten taken, en later hebben ze andere modellen nodig en een andere manier om fouten te tellen. Dus voordat je iets bouwt, vraag je je af welke van de twee je taak is.

    **Een voorbeeld uit het dagelijks leven.** Een weerbericht zegt dat het morgen 18 graden wordt, en dat het gaat regenen. De temperatuur is een getal, en 17 ligt dicht bij 18. Regen is ja of nee.
    """)
    return


@app.cell(hide_code=True)
def predict_digit():
    guess_digit = L.guess(
        "digit",
        question="Een model leest een handgeschreven cijfer, 0 tot en met 9, van een plaatje. Welke soort taak is dat volgens jou?",
        options={
            "number": "Een getal voorspellen: het antwoord is een cijfer",
            "answers": "Een van tien vaste antwoorden",
            "both": "Allebei tegelijk",
        },
        reveal="**Een van tien vaste antwoorden.** Het antwoord schrijf je met een cijfer, maar een voorspelling van 7,5 betekent niets, en een 7 die als 8 gelezen wordt, is niet beter dan een 7 die als 1 gelezen wordt: allebei gewoon fout. De tien cijfers zijn tien klassen, die er alleen uitzien als getallen. Deze taak is classificatie.",
    )
    guess_digit
    return (guess_digit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Bij supervised learning bepaalt het label de taak.

        - Bij **regressie** is het label een getal, zoals een prijs, een toetscijfer of een temperatuur. Waarden ertussen hebben betekenis, en een voorspelling dicht bij het label is bijna goed.
        - Bij **classificatie** is het label een van een vaste set antwoorden, de **klassen**: spam of geen spam, ja of nee, de handgeschreven cijfers 0 tot en met 9. Een voorspelling is goed of fout; er zit niets tussen.

        Klassen worden vaak als getallen geschreven, zoals 0 = kat, 1 = hond en 2 = vogel. Het blijven klassen: de getallen zijn alleen namen, en een kat en een vogel maken gemiddeld geen hond.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **regressie** | | een taak waarbij het label een getal is, zoals een prijs | `df["grade"]` |
        | **classificatie** | | een taak waarbij het label een van een vaste set antwoorden is, zoals spam of geen spam | `df["passed"]` |
        | **klasse** | | een van de antwoorden bij classificatie, zoals spam, of yes | `"yes"` |
        """),
        L.watch_out("*Regressie* en *klasse* betekenen ook iets anders. In software is een regressie een fout in een programma die na een wijziging terugkomt; hier is het een taak die een getal voorspelt. En een klasse is hier een van de antwoorden: geen schoolklas, en geen `class` uit Python-code."),
        draw.data_tables(C.picture(["Regressie: het label is een getal", "Classificatie: het label is een klasse"]),
                         alt="De tabel met 7 leerlingen twee keer, met de kolommen hours, sleep, grade en passed. Links is de kolom grade gemarkeerd: getallen zoals 4,5, 7,5 en 6,0. Rechts is de kolom passed gemarkeerd: yes of no."),
        mo.md(r"""
        Dezelfde zeven leerlingen twee keer, met dezelfde features: de uren die ze leerden en sliepen. Links is het label het cijfer, een getal: regressie. Rechts is het label passed, yes of no: classificatie, met twee klassen. De tabel is dezelfde; het label dat je kiest, maakt de taak.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien of een taak regressie of classificatie is:

    1. Zoek het label: de kolom die je wilt voorspellen.
    2. Kijk naar de waarden. Een getal, waarbij waarden ertussen betekenis hebben en dichtbij bijna goed is? Regressie.
    3. Een van een vaste set antwoorden? Classificatie: die antwoorden zijn de klassen. `.value_counts()` zet ze op een rij.
    4. Klassen die als getallen geschreven zijn, blijven klassen: vraag je af of een waarde ertussen iets zou betekenen.

    Een getal als label kun je ook omzetten in klassen:

    5. Kies een grens, en geef elke waarde vanaf die grens de ene klasse en elke waarde eronder de andere.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tabel met leerlingen, met twee taken: het cijfer voorspellen, of voorspellen of een leerling slaagt. Aan het eind wordt het cijfer klassen, met een grens: vanaf een 5,5 ben je geslaagd. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_tasks():
    import pandas as pd

    _df = pd.read_csv("lessons/ml-vocabulary/students.csv")
    # 1. de eerste taak: het label is grade
    # 2. getallen: het laagste en het hoogste cijfer
    print(_df["grade"].min(), _df["grade"].max())
    # 1. de tweede taak: het label is passed
    # 3. een vaste set antwoorden: de klassen, en hoeveel leerlingen er in elke klasse zitten
    print(_df["passed"].value_counts())
    # 5. van getal naar klassen: de eerste vier cijfers, met 5,5 als grens
    _classes = []
    for _grade in [4.5, 7.5, 6.5, 3.0]:
        if _grade >= 5.5:
            _classes.append("yes")
        else:
            _classes.append("no")
    print(_classes)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De cijfers lopen van 3,0 tot 8,5, en een 6,8 of een 7,25 zou ook betekenis hebben: het cijfer voorspellen is regressie. `passed` heeft maar twee waarden, `yes` voor 4 leerlingen en `no` voor 3: dat voorspellen is classificatie, met de klassen yes en no. De 3 leerlingen zonder cijfer tellen niet mee; hun labels zijn nog niet bekend. Met 5,5 als grens worden de cijfers 4,5, 7,5, 6,5 en 3,0 no, yes, yes en no: precies wat `passed` voor die leerlingen zegt. Het getal als label van de ene taak geeft de klassen van de andere.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_regression():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke is regressie?
    """)
    return


@app.cell(hide_code=True)
def choice_regression_cell():
    choice_regression = L.choice(
        "which-regression",
        question="Welke van deze taken is regressie?",
        options={
            "spam": "Of een mail spam is",
            "language": "In welke taal een tekst geschreven is",
            "right": "Hoeveel minuten een trein te laat zal zijn",
            "digit": "Welk cijfer, 0 tot en met 9, op een plaatje staat",
        },
        correct="right",
        why={
            "spam": "Spam of geen spam: is dat een getal, of een van twee antwoorden? Zie stap 3 van het recept.",
            "language": "Nederlands, Engels, Frans: een vaste set antwoorden. Zie stap 3 van het recept.",
            "digit": "Een handgeschreven cijfer ziet eruit als een getal, maar wat zou een voorspelling van 7,5 betekenen: een 7 of een 8? Zie stap 4 van het recept.",
            "right": "De minuten zijn een getal: 4,5 minuut heeft betekenis, en 6 minuten voorspellen bij 7 is bijna goed.",
        },
    )
    choice_regression
    return (choice_regression,)


@app.cell(hide_code=True)
def text_exercise_numbers():
    mo.md(r"""
    ### Oefening 2. Klassen als getallen
    """)
    return


@app.cell(hide_code=True)
def choice_numbers_cell():
    choice_numbers = L.choice(
        "class-numbers",
        question="Vier foto's hebben de labels 0, 0, 2 en 2, met 0 = kat, 1 = hond en 2 = vogel. Het gemiddelde van de labels is 1. Wat zegt dat?",
        options={
            "dog": "Op de gemiddelde foto staat een hond",
            "right": "Niets: de getallen zijn alleen namen van klassen",
            "half": "Op de helft van de foto's staat een hond",
        },
        correct="right",
        why={
            "dog": "Op geen van de vier foto's staat een hond. Is een gemiddelde van klassennummers een klasse? Zie *Het idee*.",
            "half": "Op twee foto's staat een kat en op twee een vogel: op geen enkele een hond. Zie *Het idee*, over klassen als getallen.",
            "right": "Twee katten en twee vogels, en geen hond: dit gemiddelde van klassennummers zegt niets over de foto's. Klassen beschrijf je door ze te tellen.",
        },
    )
    choice_numbers
    return (choice_numbers,)


@app.cell(hide_code=True)
def text_exercise_delays():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Van minuten naar klassen

        Het aantal minuten dat een trein te laat is, is een getal: dat voorspellen is regressie. Een reisapp wil alleen laten zien of een trein te laat is: een trein die 5 minuten of meer te laat is, telt als te laat. Maak `delay_classes(delays)` af. Hij krijgt een lijst vertragingen in minuten en geeft een lijst terug met een klasse voor elke vertraging: `"late"` of `"on time"`. Vervang de `...`. De check roept `delay_classes([2, 12, 0, 7])` aan: dat moet `["on time", "late", "on time", "late"]` geven. Hij probeert ook andere lijsten.
        """),
        L.task("Laat `delay_classes` de klasse van elke vertraging teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_delays():
    box_delays = L.code_box("delay-classes", lines=9)
    box_delays
    return (box_delays,)


@app.cell(hide_code=True)
def check_delays(box_delays):
    result_delays = L.check("delay-classes", box_delays.value)
    result_delays
    return (result_delays,)


@app.cell(hide_code=True)
def hints_delays():
    L.hints(
        "delay-classes",
        [
            "Ga de vertragingen een voor een langs, en voeg voor elke vertraging een klasse toe aan `classes`.",
            "In de loop een `if` met een `else`. Bij welke voorwaarde is een vertraging te laat? Let op een vertraging van precies 5.",
            "`for delay in delays:`, dan `if delay ___ 5:` met `classes.append(\"late\")`, en onder `else:` `classes.append(___)`.",
        ],
        why="`delay >= 5` is `True` bij 5 minuten of meer: dan is de klasse `\"late\"`, en anders `\"on time\"`. Voor `[2, 12, 0, 7]`: on time, late, on time, late.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl) had supervised learning labels; nu weet je dat de soort label de taak bepaalt: een getal bij regressie, een klasse bij classificatie. In [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) had de tabel met leerlingen twee labels om uit te kiezen; elk geeft een van de twee taken. Later in de course kiest een model voor classificatie niet alleen een klasse: het geeft een kans voor elke klasse. De volgende les, *Exponenten en logaritmen*, geeft je twee functies waarmee zo'n model gebouwd wordt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_numbers, choice_regression, result_delays):
    L.wrap_up(choice_regression, choice_numbers, result_delays)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
