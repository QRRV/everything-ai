# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les prob-expectation. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-expectation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_die():
    quick_die = L.choice(
        "quick-die",
        question="Je gooit één keer met een eerlijke dobbelsteen. Wat is P(6)?",
        options={
            "six": "6",
            "right": "1/6",
            "half": "1/2",
        },
        correct="right",
        why={
            "six": "Dat is de kant, niet de kans erop. Een kans ligt tussen 0 en 1. Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "half": "Een 6 of geen 6 zijn twee uitkomsten, maar ze zijn niet even waarschijnlijk. Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "right": "Zes kanten, elk even waarschijnlijk: elk 1/6, ongeveer 0,167.",
        },
    )
    quick_die
    return (quick_die,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="Welke stappen geven de standaardafwijking van een kolom getallen?",
        options={
            "right": "De verschillen met het gemiddelde, gekwadrateerd; hun gemiddelde; en de wortel daaruit",
            "no-square": "De verschillen met het gemiddelde, en hun gemiddelde",
            "range": "De grootste waarde min de kleinste, gehalveerd",
        },
        correct="right",
        why={
            "no-square": "Die verschillen zijn samen 0, dus hun gemiddelde is altijd 0. Wat gebeurt er eerst met elk verschil? Zie *Het idee* in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "range": "Dat gebruikt alleen de twee buitenste waarden. Zie *Het idee* in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "right": "Gekwadrateerd, zodat verschillen onder het gemiddelde die erboven niet opheffen; en de wortel brengt het terug naar de eenheid van de waarden.",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een weddenschap, een spel of de gok van een model heeft een uitkomst die van het toeval afhangt. Twee vragen bepalen of het iets waard is: wat levert het gemiddeld op als je het heel vaak herhaalt, en hoe ver schommelen de uitkomsten rond dat gemiddelde? Een model trainen is net zo: het maakt de loss klein, gemiddeld over alle voorbeelden die het zou kunnen zien.

    **Een voorbeeld uit het dagelijks leven.** Een lot kost €2, en 1 lot op de 100 wint €150. Gemiddeld levert een lot €1,50 op: per lot krijg je 50 cent minder terug dan je betaalde, ook al gaat één koper op de honderd blij naar huis.
    """)
    return


@app.cell(hide_code=True)
def predict_average():
    guess_average = L.guess(
        "average",
        question="Je gooit duizenden keren met een eerlijke dobbelsteen. Wat denk je dat het gemiddelde van alle worpen wordt?",
        options={
            "three": "3",
            "three-half": "3,5",
            "six": "6",
        },
        reveal="**3,5:** elke kant van 1 tot 6 komt even vaak boven, en hun gemiddelde is 21 / 6 = 3,5. Dat getal gooi je nooit.",
    )
    guess_average
    return (guess_average,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een getal dat van het toeval afhangt, $X$, zoals wat je met een dobbelsteen gooit of wat een weddenschap uitbetaalt, heeft een kansverdeling: elk van zijn waarden $x_1, x_2, \dots$ met zijn kans. De verwachtingswaarde is het gemiddelde dat je op de lange duur krijgt: elke waarde keer zijn kans, opgeteld.
        """),
        L.formula(r"\mathbb{E}[X] = x_1 \cdot P(x_1) + x_2 \cdot P(x_2) + \dots"),
        mo.md(r"""
        De variantie meet hoe ver de waarden rond de verwachtingswaarde uiteenliggen: voor elke waarde de gekwadrateerde afstand tot de verwachtingswaarde, keer zijn kans, opgeteld.
        """),
        L.formula(r"\mathrm{Var}(X) = (x_1 - \mathbb{E}[X])^2 \cdot P(x_1) + (x_2 - \mathbb{E}[X])^2 \cdot P(x_2) + \dots"),
        mo.md(r"""
        De wortel eruit is de standaardafwijking uit [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl), nu gewogen met de kansen. Die is weer in de eenheid van de waarden, en zegt hoe ver ze doorgaans van de verwachtingswaarde liggen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **verwachtingswaarde** | | het gemiddelde op de lange duur van een getal dat van het toeval afhangt: elke waarde keer zijn kans, opgeteld | `(values * probs).sum()` |
        | **variantie** | | de gekwadrateerde afstand tot de verwachtingswaarde, gewogen met de kans; de wortel eruit is de standaardafwijking | `((values - mean) ** 2 * probs).sum()` |
        | $x_1, x_2, \dots$ | x één, x twee, enzovoort | de waarden die $X$ kan aannemen | |
        | $\mathbb{E}[X]$ | de verwachtingswaarde van X | de verwachtingswaarde van $X$ | |
        | $\mathrm{Var}(X)$ | de variantie van X | de variantie van $X$ | |
        """),
        L.watch_out("De verwachtingswaarde hoeft geen waarde te zijn die kan voorkomen: met een dobbelsteen gooi je nooit 3,5. Het is ook niet de waarschijnlijkste waarde: het is het gemiddelde over heel veel pogingen."),
        mo.Html(C.picture({"safe": "Een veilige gok", "wild": "Een wilde gok", "x": "wat je wint", "y": "kans",
                           "alt": "Twee staafdiagrammen. Een veilige gok betaalt 4 of 6, elk met kans 0,5. Een wilde gok betaalt 0 of 10, elk met kans 0,5. Bij allebei markeert een stippellijn de verwachtingswaarde, 5."})),
        mo.md(r"""
        Beide gokken leveren gemiddeld 5 op: de stippellijn. De veilige gok betaalt 4 of 6, elk 1 van 5 af, dus zijn variantie is 1² × 0,5 + 1² × 0,5 = 1. De wilde gok betaalt 0 of 10, elk 5 ervan af, dus zijn variantie is 5² × 0,5 + 5² × 0,5 = 25. Hun standaardafwijkingen, 1 en 5, zeggen hoe ver elke gok doorgaans van 5 af uitbetaalt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo beschrijf je een getal dat van het toeval afhangt:

    1. Zet elke waarde op een rij met zijn kans: de kansverdeling.
    2. Verwachtingswaarde: vermenigvuldig elke waarde met zijn kans, en tel op.
    3. Variantie: neem voor elke waarde de gekwadrateerde afstand tot de verwachtingswaarde, keer zijn kans, en tel op.
    4. Standaardafwijking: de wortel uit de variantie.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een eerlijke dobbelsteen: zijn verwachtingswaarde, variantie en standaardafwijking, en daarna een controle met 6000 worpen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_die():
    import random

    # 1. de kansverdeling: elke kant met kans 1/6
    _die = {}
    for _face in range(1, 7):
        _die[_face] = 1 / 6
    # 2. de verwachtingswaarde
    _mean = 0
    for _face in _die:
        _mean = _mean + _face * _die[_face]
    print(_mean)
    # 3. de variantie
    _var = 0
    for _face in _die:
        _var = _var + (_face - _mean) ** 2 * _die[_face]
    print(_var)
    # 4. de standaardafwijking
    _std = _var ** 0.5
    print(_std)
    # controle: het gemiddelde van 6000 worpen
    random.seed(3)
    _total = 0
    for _i in range(6000):
        _total = _total + random.randint(1, 6)
    print(_total / 6000)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De verwachtingswaarde is 3,5, de variantie ongeveer 2,92, en de standaardafwijking ongeveer 1,71: een worp ligt doorgaans zo'n 1,7 van 3,5 af. Het gemiddelde van 6000 worpen met `random.randint` is 3,48, dicht bij 3,5. Hoe meer worpen, hoe dichterbij.
    """)
    return


@app.cell(hide_code=True)
def text_numpy():
    mo.md(r"""
    **Hetzelfde in NumPy.** Zet de waarden in één array en hun kansen in een tweede, in dezelfde volgorde. Dan is elke stap van het recept één regel, zonder loop:

    | Stap | Gewone Python | NumPy |
    |---|---|---|
    | 1 | een dictionary `_die` | twee arrays, `_faces` en `_probs` |
    | 2 | een loop die `_face * _die[_face]` optelt | `(_faces * _probs).sum()` |
    | 3 | een loop die `(_face - _mean) ** 2 * _die[_face]` optelt | `((_faces - _mean) ** 2 * _probs).sum()` |
    """)
    return


@app.cell
def example_die_numpy():
    import numpy as np

    _faces = np.array([1, 2, 3, 4, 5, 6])          # 1. de waarden
    _probs = np.array([1, 1, 1, 1, 1, 1]) / 6      # 1. hun kansen, in dezelfde volgorde
    _mean = (_faces * _probs).sum()                # 2. de verwachtingswaarde
    _var = ((_faces - _mean) ** 2 * _probs).sum()  # 3. de variantie
    print(_mean, _var, _var ** 0.5)                # 2, 3 en 4 in één keer
    return


@app.cell(hide_code=True)
def text_exercise_game():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De moeite waard?
    """)
    return


@app.cell(hide_code=True)
def choice_game_cell():
    choice_game = L.choice(
        "game",
        question="Een spel: je wint €10 met kans 0,2, en anders niets. Wat is de verwachtingswaarde van wat je wint?",
        options={
            "prize": "€10",
            "half": "€5",
            "right": "€2",
        },
        correct="right",
        why={
            "prize": "Zoveel win je als je wint. Hoe vaak is dat? Zie de formule in *Het idee*.",
            "half": "Dat is het gewone gemiddelde van €0 en €10, alsof beide even waarschijnlijk zijn. Zie het blok *Let op* en stap 2 van het recept.",
            "right": "10 × 0,2 + 0 × 0,8 = 2: gemiddeld €2 per spel. Kost een spel meer dan €2, dan kost het je op de lange duur geld.",
        },
    )
    choice_game
    return (choice_game,)


@app.cell(hide_code=True)
def text_exercise_expected():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De verwachtingswaarde als functie

        Maak `expected(dist)` af. Hij krijgt een kansverdeling als dictionary, elke waarde met zijn kans, en geeft de verwachtingswaarde terug. Vervang de `...`. De check roept `expected({1: 0.5, 2: 0.25, 4: 0.25})` aan: dat moet `2.0` geven. Hij probeert ook andere kansverdelingen, zoals een eerlijke dobbelsteen.
        """),
        L.task("Laat `expected` de verwachtingswaarde van `dist` teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_expected():
    box_expected = L.code_box("expected", lines=7)
    box_expected
    return (box_expected,)


@app.cell(hide_code=True)
def check_expected(box_expected):
    result_expected = L.check("expected", box_expected.value)
    result_expected
    return (result_expected,)


@app.cell(hide_code=True)
def hints_expected():
    L.hints(
        "expected",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld.",
            "Een loop over de keys van `dist` geeft elke waarde; `dist[value]` is zijn kans. Wat tel je voor elke waarde bij het totaal op?",
            "`total = total + value * ___`, en na de loop `return total`.",
        ],
        why="De loop telt `value * dist[value]` op voor elke waarde: 1 × 0,5 + 2 × 0,25 + 4 × 0,25 = 2,0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_variance():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoe wild is de prijs?

        Een loterij betaalt €0, €5 of €20. `values` bevat de prijzen en `probs` hun kansen, als NumPy-arrays in dezelfde volgorde, en `mean` is de verwachtingswaarde, al voor je uitgerekend. Zet de variantie in `variance`. Vervang de `...`. De laatste regel is `variance`, zodat de check hem ziet. De check probeert ook andere kansen.
        """),
        L.task("Zet de variantie van de prijs in `variance`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_variance():
    box_variance = L.code_box("variance", lines=8)
    box_variance
    return (box_variance,)


@app.cell(hide_code=True)
def check_variance(box_variance):
    result_variance = L.check("variance", box_variance.value)
    result_variance
    return (result_variance,)


@app.cell(hide_code=True)
def hints_variance():
    L.hints(
        "variance",
        [
            "Stap 3 van het recept, met arrays: NumPy rekent met alle waarden tegelijk.",
            "`values - mean` geeft de afstand van elke waarde tot de verwachtingswaarde. Wat gebeurt er met die afstanden, en waarmee vermenigvuldig je ze?",
            "`variance = ((values - mean) ** 2 * ___).sum()`.",
        ],
        why="`(values - mean) ** 2` kwadrateert elke afstand tot 5,5; `* probs` weegt die met zijn kans; `.sum()` telt op: 30,25 × 0,5 + 0,25 × 0,3 + 210,25 × 0,2 = 57,25.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) maakte je kansverdelingen; de verwachtingswaarde is hun gemiddelde op de lange duur, en de variantie meet hoe ver uitkomsten daaromheen schommelen. De wortel eruit is de standaardafwijking uit [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl), gewogen met de kansen. Later in de course is een loss een verwachtingswaarde: het gemiddelde over alle voorbeelden die een model zou kunnen zien. In de volgende les, *Soorten leren*, houd je de soorten leren uit elkaar aan waar de goede antwoorden vandaan komen. Met *Toon verdieping* aan komt eerst *De regel van Bayes*: die draait een voorwaardelijke kans om.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_game, result_expected, result_variance):
    L.wrap_up(choice_game, result_expected, result_variance)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
