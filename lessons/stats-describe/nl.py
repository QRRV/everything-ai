# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les stats-describe. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("stats-describe", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) en [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mean():
    quick_mean = L.choice(
        "quick-mean",
        question="`a = np.array([2, 4, 9])`. Wat geeft `a.mean()`?",
        options={
            "mean": "`5.0`",
            "sum": "`15`",
            "middle": "`4`",
        },
        correct="mean",
        why={
            "sum": "Dat is de som, `a.sum()`. Wat doe je met de som om het gemiddelde te krijgen? Zie de tabel met nieuwe woorden in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "middle": "Dat is het middelste van de drie getallen. Hoe reken je een gemiddelde uit? Zie de tabel met nieuwe woorden in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "mean": "2 + 4 + 9 is 15, en 15 gedeeld door 3 getallen is 5,0.",
        },
    )
    quick_mean
    return (quick_mean,)


@app.cell(hide_code=True)
def quick_check_bar():
    quick_bar = L.choice(
        "quick-bar",
        question="Wat tekent `plt.bar([\"mon\", \"tue\"], [3, 5])`?",
        options={
            "line": "Een lijn van 3 naar 5",
            "points": "Twee losse punten",
            "bars": "Twee staven, 3 en 5 hoog",
        },
        correct="bars",
        why={
            "line": "Een lijn door punten is `plt.plot`. Welke grafiek heeft een staaf voor elke naam? Zie de tabel met nieuwe woorden in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "points": "Losse punten zijn `plt.scatter`. Welke grafiek heeft een staaf voor elke naam? Zie de tabel met nieuwe woorden in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "bars": "`plt.bar` tekent een staaf voor elke naam, zo hoog als zijn waarde.",
        },
    )
    quick_bar
    return (quick_bar,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een kolom data kan duizenden waarden hebben, te veel om één voor één te lezen. Een paar getallen en één grafiek vertellen je het meeste: waar het midden ligt, en hoe ver de waarden eromheen verspreid liggen. Zo bekijk je data voordat je er een model op traint, en later beschrijf je met dezelfde getallen hoe ver een model ernaast zit.

    **Een voorbeeld uit het dagelijks leven.** Een weerbericht: "morgen rond de 18 graden, tussen 14 en 21". Eén getal voor het midden, en twee voor de spreiding.
    """)
    return


@app.cell(hide_code=True)
def predict_bus():
    guess_bus = L.guess(
        "bus",
        question="Vijftien leerlingen hebben gemeten hoe lang hun reis naar school duurde. De meesten deden er 12 tot 45 minuten over; één miste de bus en deed er 95 over. Wat denk je dat groter is: het gemiddelde of de mediaan, de middelste waarde als je ze sorteert?",
        options={
            "mean": "Het gemiddelde",
            "median": "De mediaan",
            "same": "Ze zijn even groot",
        },
        reveal="**Het gemiddelde.** Het gemiddelde is 30,7 minuten en de mediaan 25. De rit van 95 minuten maakt de som veel groter en trekt zo het gemiddelde omhoog. Voor de mediaan is het maar één waarde meer bovenaan de rij.",
    )
    guess_bus
    return (guess_bus,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Twee vragen beschrijven een kolom getallen: waar ligt het midden, en hoe ver liggen de waarden eromheen verspreid? Het gemiddelde is het midden dat je kent: de som gedeeld door het aantal waarden. De mediaan is de middelste waarde als je ze sorteert. Eén extreme waarde trekt aan het gemiddelde, maar verschuift de mediaan nauwelijks. De standaardafwijking zegt hoe ver de waarden doorgaans van het gemiddelde af liggen, en een kwantiel zegt onder welke waarde een deel ervan ligt. Een histogram laat alle waarden tegelijk zien.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **mediaan** | de middelste waarde als je ze sorteert: de helft van de waarden ligt eronder, de helft erboven | `np.median(x)` |
    | **uitschieter** | een waarde ver van de rest; het kan een fout zijn of een zeldzaam echt geval, dus kijk ernaar voordat je hem weghaalt | |
    | **standaardafwijking** | hoe ver de waarden doorgaans van het gemiddelde af liggen, in hun eigen eenheid; klein betekent dicht bij elkaar | `x.std()` |
    | **kwantiel** | de waarde waaronder een bepaald deel van de waarden ligt; de mediaan is het 0,5-kwantiel | `np.quantile(x, 0.9)` |
    | **`.min()`, `.max()`** | de kleinste en de grootste waarde | `x.max()` |
    | **histogram** | een grafiek die het bereik van de waarden in gelijke bins verdeelt, en voor elke bin een staaf tekent, zo hoog als het aantal waarden dat erin valt | `plt.hist(x, bins=...)` |
    | **bin** | een van die gelijke stukken van het bereik | `bins=range(0, 101, 10)` |

    `range(0, 101, 10)` telt van 0 tot 100 in stappen van 10: het derde getal is de stap, en 101 zorgt dat 100 meedoet. Dat zijn de randen van de bins.

    De standaardafwijking reken je in vier stappen uit: het verschil van elke waarde met het gemiddelde, `x - x.mean()`; elk verschil in het kwadraat, zodat een verschil onder het gemiddelde, dat negatief is, niet wegvalt tegen een verschil erboven; het gemiddelde van die kwadraten; en daarvan de wortel, `** 0.5`, om terug te gaan naar de eenheid van de waarden.
    """)
    return


@app.cell(hide_code=True)
def picture_times():
    mo.vstack([
        mo.Html(C.picture({"title": "Reistijden naar school", "x": "minuten", "y": "leerlingen",
                           "median": "mediaan, 25", "mean": "gemiddelde, 30,7",
                           "alt": "Een histogram van 15 reistijden, met staven van 10 minuten: 3 leerlingen deden er 10 tot 20 minuten over, 6 deden er 20 tot 30 over, 3 deden er 30 tot 40 over, 2 deden er 40 tot 50 over, en 1 deed er 90 tot 100 over. Een doorgetrokken lijn staat bij de mediaan, 25, een gestippelde lijn bij het gemiddelde, 30,7."})),
        mo.md(r"""
        De 15 reistijden als histogram: een staaf voor elke 10 minuten, zo hoog als het aantal leerlingen erin. De meeste ritten duurden 10 tot 50 minuten; één duurde 95. De mediaan, 25, ligt midden in de grote groep. Het gemiddelde, 30,7, wordt door die ene lange rit naar rechts getrokken.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`x.std()` en `np.std(x)` delen door het aantal waarden. Sommige andere tools delen door één minder, en geven op dezelfde getallen dus een iets grotere uitkomst. Beide heten standaardafwijking; zeg welke je gebruikt als het ertoe doet."),
        mo.md(r"""
        ### Recept

        Zo beschrijf je een kolom getallen:

        1. Teken een histogram, om de vorm en eventuele uitschieters te zien: `plt.hist(x, bins=...)`.
        2. Zoek het midden: `x.mean()` en `np.median(x)`. Verschillen ze veel, zoek dan naar uitschieters.
        3. Zoek de spreiding: `x.std()`, de kleinste en de grootste waarde, en een kwantiel zoals `np.quantile(x, 0.9)`.
        4. Zeg in een zin wat je ziet, met de eenheid: "de helft van de ritten duurt minder dan 25 minuten".
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De reistijden van de 15 leerlingen, in minuten. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_times():
    import matplotlib.pyplot as plt
    import numpy as np

    _times = np.array([25, 18, 32, 20, 95, 27, 12, 40, 22, 25, 35, 15, 30, 20, 45])
    # 1. het histogram: een staaf voor elke 10 minuten
    plt.hist(_times, bins=range(0, 101, 10))
    plt.xlabel("minuten")
    plt.ylabel("leerlingen")
    plt.title("Reistijden naar school")
    plt.show()
    # 2. het midden
    print(_times.mean())
    print(np.median(_times))
    # 3. de spreiding
    print(_times.std())
    print(_times.min())
    print(_times.max())
    print(np.quantile(_times, 0.9))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het histogram heeft één grote groep van 10 tot 50 minuten en één eenzame staaf bij 90 tot 100: de leerling die de bus miste. Het gemiddelde is 30,7 minuten en de mediaan 25: de lange rit trekt het gemiddelde omhoog. De standaardafwijking is 19,3 minuten; zonder de rit van 95 minuten zou hij 9,1 zijn, dus één uitschieter verdubbelt hem. De ritten duren van 12 tot 95 minuten. Het 0,9-kwantiel is 43: 90% van 15 ritten is 13,5, dus die plek valt tussen de 13e en de 14e van de gesorteerde tijden, 40 en 45, en NumPy neemt een waarde daartussen. Stap 4, in een zin: "de helft van de leerlingen is binnen 25 minuten op school, en bijna iedereen binnen 45; één leerling miste de bus en deed er 95 over".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_house():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een doorsnee huis
    """)
    return


@app.cell(hide_code=True)
def choice_house_cell():
    choice_house = L.choice(
        "typical-house",
        question="De vijf huizen in een straat kosten 210, 230, 250, 260 en 1450 duizend euro. Welk getal zegt het best wat een doorsnee huis in die straat kost?",
        options={
            "mean": "Het gemiddelde: 480 duizend",
            "median": "De mediaan: 250 duizend",
            "highest": "De hoogste prijs: 1450 duizend",
        },
        correct="median",
        why={
            "mean": "Vier van de vijf huizen kosten veel minder dan 480 duizend. Wat trekt het gemiddelde omhoog?",
            "highest": "Dat is één huis, het duurste. Zegt dat wat de meeste huizen kosten?",
            "median": "Twee huizen kosten minder en twee meer. Het ene heel dure huis trekt het gemiddelde omhoog, maar de mediaan blijft in het midden.",
        },
    )
    choice_house
    return (choice_house,)


@app.cell(hide_code=True)
def text_exercise_slow():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel tijd nodig?

        Het vak heeft de reistijden als `times`. De school wil weten hoeveel tijd de meeste leerlingen nodig hebben: maak `slow`, de tijd waarbinnen 3 van elke 4 ritten klaar zijn. Vervang de `...`. De laatste regel is `slow`, zodat de check het ziet. De check probeert ook een andere groep leerlingen.
        """),
        L.task("Maak `slow`: de tijd waarbinnen 3 van elke 4 ritten in `times` klaar zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slow():
    box_slow = L.code_box("slow-trips", lines=5)
    box_slow
    return (box_slow,)


@app.cell(hide_code=True)
def check_slow(box_slow):
    result_slow = L.check("slow-trips", box_slow.value)
    result_slow
    return (result_slow,)


@app.cell(hide_code=True)
def hints_slow():
    L.hints(
        "slow-trips",
        [
            "Stap 3 van het recept: een kwantiel, waaronder een bepaald deel van de waarden ligt.",
            "Welke functie in de tabel met nieuwe woorden krijgt de array en een fractie van 0 tot 1? Welke fractie is 3 van elke 4?",
            "`slow = np.___(times, ___)`.",
        ],
        why="3 van elke 4 is de fractie 0,75, en `np.quantile(times, 0.75)` is de tijd waaronder 3 van elke 4 ritten liggen: 33,5 minuut. Die plek valt tussen twee van de gesorteerde tijden, 32 en 35, dus neemt NumPy een waarde daartussen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De standaardafwijking, stap voor stap

        Maak `spread(values)` af. Hij rekent de standaardafwijking van een lijst getallen zelf uit, in de vier stappen onder de tabel met nieuwe woorden, zonder `std`. Vervang de `...`. De check roept `spread([2, 4, 4, 4, 5, 5, 7, 9])` aan: dat moet `2.0` geven. De check probeert ook andere lijsten.
        """),
        L.task("Laat `spread(values)` de standaardafwijking van `values` uitrekenen, stap voor stap."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spread():
    box_spread = L.code_box("spread", lines=8)
    box_spread
    return (box_spread,)


@app.cell(hide_code=True)
def check_spread(box_spread):
    result_spread = L.check("spread", box_spread.value)
    result_spread
    return (result_spread,)


@app.cell(hide_code=True)
def hints_spread():
    L.hints(
        "spread",
        [
            "Vier stappen: de verschillen met het gemiddelde, in het kwadraat, hun gemiddelde, en de wortel.",
            "`x - x.mean()` geeft het verschil van elke waarde met het gemiddelde, als array. Welke macht kwadrateert elk getal van een array in één keer, en welke neemt de wortel?",
            "`differences = x - x.mean()`, en dan `return (differences ___).mean() ___`.",
        ],
        why="Het gemiddelde van het voorbeeld is 5. De verschillen ermee zijn -3, -1, -1, -1, 0, 0, 2 en 4; in het kwadraat 9, 1, 1, 1, 0, 0, 4 en 16. Hun gemiddelde is 32 / 8 = 4, en de wortel uit 4, `4 ** 0.5`, is 2,0.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) rekende je een gemiddelde uit; nu ken je ook een midden dat één uitschieter niet wegtrekt, en meet je de spreiding. In [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) tekende je een staaf voor elke naam; een histogram tekent een staaf voor elke bin. Later in de course beoordeel je een model op hoe ver het gemiddeld ernaast zit, en komt de standaardafwijking terug om getallen op dezelfde schaal te zetten. In de volgende les, *Vragen stellen aan data*, gebruik je deze getallen om een hele dataset te bekijken voordat je er een model op bouwt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_house, result_slow, result_spread):
    L.wrap_up(choice_house, result_slow, result_spread)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
