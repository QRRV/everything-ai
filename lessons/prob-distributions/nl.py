# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les prob-distributions. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-distributions", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) en [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="Snavellengtes hebben een variantie van 9 mm². Wat is hun standaardafwijking?",
        options={
            "right": "3 mm: de wortel eruit",
            "same": "9 mm: hetzelfde getal",
            "half": "4,5 mm: de helft ervan",
        },
        correct="right",
        why={
            "same": "De variantie is in het kwadraat van de eenheid, mm²; de standaardafwijking is weer in mm. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "half": "De standaardafwijking is niet de helft van de variantie. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "√9 = 3: de standaardafwijking is de wortel uit de variantie, in de eenheid van de waarden.",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def quick_check_classes():
    quick_classes = L.choice(
        "quick-classes",
        question="Een softmax over drie klassen geeft de eerste twee de kansen 0,7 en 0,2. Wat geeft hij de derde?",
        options={
            "zero": "0: de eerste klasse heeft gewonnen",
            "mean": "0,45: het gemiddelde van die twee",
            "right": "0,1: alle drie samen zijn 1",
        },
        correct="right",
        why={
            "zero": "Softmax geeft elke klasse een kans boven 0. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "mean": "De kansen worden niet gemiddeld: samen zijn ze 1. Zie [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "1 − 0,7 − 0,2 = 0,1: de kansen van een softmax zijn samen altijd 1.",
        },
    )
    quick_classes
    return (quick_classes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Of een e-mail spam is, welk cijfer er op een afbeelding staat, hoe lang de snavel van een pinguïn is: deep learning beschrijft elk daarvan met een kansverdeling, en de losses waarmee je traint komen daaruit voort. Drie verdelingen komen steeds terug. Ken je ze, dan kun je lezen wat de output van een model betekent, nagaan of data eruitziet zoals een model aanneemt, en nieuwe waarden trekken die op echte lijken.

    **Een voorbeeld uit het dagelijks leven.** Een weer-app geeft een kans op regen: ja of nee. Een peiling geeft een aandeel per partij: één uit vele. Een arts legt de lengte van een kind naast de groeicurve voor zijn leeftijd: een meting.
    """)
    return


@app.cell(hide_code=True)
def predict_band():
    _x = C.bills()
    guess_band = L.guess(
        "within-one",
        question=f"De {len(_x)} Adéliepinguïns in de data hebben een gemiddelde snavellengte van {L.number(_x.mean(), 1)} mm, met een standaardafwijking van {L.number(_x.std(), 1)} mm. Welk deel van hen heeft volgens jou een snavel binnen één standaardafwijking van het gemiddelde, tussen {L.number(_x.mean() - _x.std(), 1)} en {L.number(_x.mean() + _x.std(), 1)} mm?",
        options={
            "third": "Ongeveer een derde: de band is smal",
            "right": "Ongeveer twee derde: de meeste liggen dicht bij het gemiddelde",
            "all": "Bijna allemaal: maar weinig pinguïns wijken echt af van de rest",
        },
        reveal=f"**Ongeveer twee derde: {L.number(100 * C.within(1), 1)}%.** Binnen twee standaardafwijkingen is het {L.number(100 * C.within(2), 1)}%. Een normale verdeling voorspelt ongeveer 68% en 95%: de snavellengtes van één soort volgen de klokvorm goed.",
    )
    guess_band
    return (guess_band,)


@app.cell(hide_code=True)
def text_idea():
    _d = C.penguins()
    _males = int((_d["sex"] == "male").sum())
    _known = int(_d["sex"].notna().sum())
    _shares = _d["species"].value_counts() / len(_d)
    mo.vstack([
        mo.md(f"""
        ## Het idee

        Een kansverdeling geeft elke mogelijke uitkomst zijn kans, zoals in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl). Drie soorten komen overal terug.

        De **Bernoulli-verdeling** is voor één uitkomst ja of nee: 1 met kans $p$, 0 met kans $1 - p$. De verwachtingswaarde is $p$ en de variantie $p(1 - p)$, met de formules uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl). Van de {_known} pinguïns waarvan het geslacht bekend is, zijn er {_males} mannetjes: $p$ = {L.number(_males / _known, 3)}.

        De **categorische verdeling** is voor één uit $K$ klassen, elk met een eigen kans; samen tellen ze op tot 1. De softmax-output van een classifier is er een, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), en de Bernoulli-verdeling is het geval $K = 2$. De soorten van de pinguïns: Adélie {L.number(_shares["Adelie"], 2)}, Gentoo {L.number(_shares["Gentoo"], 2)}, Chinstrap {L.number(_shares["Chinstrap"], 2)}.

        De **normale verdeling** $\\mathcal{{N}}(\\mu, \\sigma^2)$ is voor metingen die zich rond een gemiddelde verzamelen: de klokvorm, met het gemiddelde $\\mu$ en de standaardafwijking $\\sigma$. Haar **kansdichtheid**, kortweg dichtheid, zegt hoe dicht de waarden bij elke $x$ liggen:
        """),
        L.formula(r"[[pred:p(x)]] = \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2\sigma^2}}"),
        mo.md(r"""
        Een dichtheid is geen kans: de kans op een bereik van waarden is de oppervlakte onder de kromme daar, en de hele oppervlakte is 1. Een dichtheid kan dus boven 1 komen, als de waarden dicht bij elkaar liggen: $\mathcal{N}(0, 0{,}2^2)$ is bij zijn gemiddelde ongeveer 2. Ongeveer 68% van de oppervlakte ligt binnen één $\sigma$ van het gemiddelde, en ongeveer 95% binnen twee. Formules schrijven de variantie in $\mathcal{N}(\mu, \sigma^2)$; `random.gauss` van Python en NumPy nemen de standaardafwijking $\sigma$.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Bernoulli-verdeling** | ber-noe-lie | één uitkomst ja of nee: 1 met kans $p$, 0 met kans $1 - p$ | `random.random() < p` |
        | **categorische verdeling** | | één uit $K$ klassen, elk met een eigen kans, samen 1 | `random.choices(classes, weights=probs)` |
        | **normale verdeling**, $\mathcal{N}(\mu, \sigma^2)$ | normaal met gemiddelde mu en variantie sigma-kwadraat | de klokvorm van metingen rond het gemiddelde $\mu$, met standaardafwijking $\sigma$ | `random.gauss(mu, sigma)` |
        | **kansdichtheid**, $p(x)$ | p van x | hoe dicht de waarden bij $x$ liggen; de oppervlakte eronder over een bereik is de kans op dat bereik | |
        | $\mu$, $\sigma$ | mu, sigma | het gemiddelde en de standaardafwijking van een normale verdeling | `mu`, `sigma` |

        De code in de tabel trekt één waarde uit elke verdeling. `random.random()` geeft een getal van 0 tot 1, elke waarde even waarschijnlijk, dus het ligt met kans `p` onder `p`: één trekking uit een Bernoulli-verdeling, als `True` of `False`. `random.choices(classes, weights=probs)` kiest één klasse, elke naar verhouding van haar getal in `weights`, en geeft die terug in een lijst; in `weights` mogen ook aantallen staan. `random.gauss(mu, sigma)` geeft één getal uit $\mathcal{N}(\mu, \sigma^2)$.
        """),
        L.watch_out("Niet elke meting is normaal. Inkomens, de grootte van steden of het aantal woorden in een tekst hebben een lange staart naar één kant, en de 68%-regel geldt daar niet. En een mengsel van groepen, zoals de snavels van alle drie de soorten samen, kan twee of drie bulten hebben. Kijk naar een histogram voordat je een klokvorm aanneemt."),
        mo.md(r"""
        De figuur: de snavellengtes van de Adéliepinguïns in bins van 1 mm, en de <span class="c-pred">klokvorm</span> met hun gemiddelde en standaardafwijking. De stippellijnen liggen op één standaardafwijking van het gemiddelde.
        """),
        mo.Html(C.picture({"title": "Snavellengte van Adéliepinguïns", "x": "snavellengte (mm)", "y": "pinguïns", "band": "binnen één σ",
                           "alt": f"Een histogram van de snavellengtes van {len(C.bills())} Adéliepinguïns, van ongeveer 32 tot 46 mm, het hoogst bij 37 tot 40 mm, met een klokvorm erover die de staven goed volgt. Stippellijnen op één standaardafwijking aan elke kant van het gemiddelde."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om data te beschrijven met een verdeling:

    1. Kijk wat voor uitkomst het is: ja of nee is Bernoulli, één uit een paar klassen is categorisch, een meting kan normaal zijn.
    2. Schat de parameters uit de data: $p$ is het aandeel ja, elke klasse krijgt haar aandeel, $\mu$ is het gemiddelde en $\sigma$ de standaardafwijking.
    3. Gebruik de verdeling: haar kansen, haar verwachtingswaarde en variantie, of nieuwe waarden die je eruit trekt.
    4. Vergelijk haar met de data, bij een normale verdeling met het deel binnen één en twee standaardafwijkingen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns met alle drie de verdelingen. `C.penguins()` is de tabel, `C.bills()` zijn de snavellengtes van de Adéliepinguïns, en `C.within(k)` het deel daarvan binnen `k` standaardafwijkingen, dat je in oefening 3 zelf schrijft. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 en 4 zijn niet af: voer de cel uit zoals hij is, maak ze dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_distributions():
    import random
    import numpy as np

    _d = C.penguins()
    # 1. geslacht is ja of nee: Bernoulli. De soort is één uit drie: categorisch.
    #    De snavellengte is een meting: normaal?
    # 2. de parameters: p, de aandelen van de soorten, mu en sigma
    _males = np.sum(_d["sex"] == "male")
    _females = np.sum(_d["sex"] == "female")
    _p = _males / (_males + _females)
    print(_p)
    print(_d["species"].value_counts() / len(_d))
    _x = C.bills()
    _mu = np.mean(_x)
    _sigma = np.std(_x)
    print(_mu, _sigma)
    # 3. gebruik ze: de verwachtingswaarde bij Bernoulli staat er; voeg de variantie toe. Dan één trekking uit elk
    print(_p)
    random.seed(1)
    print(random.random() < _p)
    print(random.choices(["Adelie", "Gentoo", "Chinstrap"], weights=[152, 124, 68]))
    print(random.gauss(_mu, _sigma))
    # 4. controleer de normale verdeling: het deel binnen één σ staat er; voeg het deel binnen twee toe
    print(C.within(1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Van de pinguïns met een bekend geslacht is 50,5% een mannetje: $p$ = 0,505, met variantie 0,25, de grootste die een Bernoulli-verdeling kan hebben. De `weights` van `random.choices` zijn de aantallen van de soorten, 152, 124 en 68: `value_counts` zonder te delen. De trekkingen geven een mannetje, een Chinstrap-pinguïn en een snavel van 39,0 mm. Stap 4: 68,2% binnen één standaardafwijking en 96,7% binnen twee, dicht bij de 68% en 95% van een normale verdeling.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke verdeling?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="Een model kijkt naar een afbeelding en geeft een kans voor elk van de 10 cijfers. Welke verdeling vormen die 10 kansen?",
        options={
            "right": "Categorisch: één uit 10 klassen",
            "bernoulli": "Bernoulli: het antwoord is goed of fout",
            "normal": "Normaal: de cijfers liggen rond een gemiddelde",
        },
        correct="right",
        why={
            "bernoulli": "Een Bernoulli-verdeling heeft twee uitkomsten. Hier toont de afbeelding één van 10 cijfers, elk met een eigen kans.",
            "normal": "De cijfers zijn klassen, geen metingen: een 7 ligt op geen bruikbare manier tussen een 6 en een 8.",
            "right": "Eén uit 10 klassen, elk met een eigen kans, samen 1: een categorische verdeling.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_density():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De klokvorm als functie

        Schrijf `normal_density(x, mu, sigma)`: de dichtheid van $\mathcal{N}(\mu, \sigma^2)$ bij `x`, met de formule. Het vak importeert `math`: gebruik `math.exp`, `math.sqrt` en `math.pi`. De check roept `normal_density(4, 1, 3)` aan, en dat moet ongeveer `0.0807` geven; hij probeert ook andere waarden.
        """),
        L.task("Laat `normal_density` de dichtheid van de normale verdeling geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_density():
    box_density = L.code_box("density", lines=6)
    box_density
    return (box_density,)


@app.cell(hide_code=True)
def check_density(box_density):
    result_density = L.check("density", box_density.value)
    result_density
    return (result_density,)


@app.cell(hide_code=True)
def hints_density():
    L.hints(
        "density",
        [
            "De formule heeft twee delen: het getal ervoor, 1 gedeeld door $\\sigma \\sqrt{2\\pi}$, en $e$ tot de macht $-\\frac{(x - \\mu)^2}{2\\sigma^2}$.",
            "`math.exp(v)` is $e$ tot de macht `v`, en `**` is een macht. Zet haakjes om alles waardoor je deelt.",
        ],
        why="Bij $x = 4$, $\\mu = 1$, $\\sigma = 3$: het getal ervoor is $\\frac{1}{3 \\cdot 2{,}507} \\approx 0{,}133$, en $e^{-9/18} = e^{-1/2} \\approx 0{,}6065$. Samen ongeveer 0,0807.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_within():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Controleren met de klokvorm

        Schrijf `share_within(x, k)`: het deel van de waarden van de array `x` dat binnen `k` standaardafwijkingen van hun gemiddelde ligt, dus meer dan het gemiddelde min `k` standaardafwijkingen en minder dan het gemiddelde plus `k`. Dat is stap 4 van het recept. De check roept `share_within(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1)` aan, en dat moet `0.6` geven; hij probeert ook andere arrays en andere waarden van `k`.
        """),
        L.task("Laat `share_within` het deel van de waarden geven dat binnen `k` standaardafwijkingen van het gemiddelde ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_within():
    box_within = L.code_box("share-within", lines=10)
    box_within
    return (box_within,)


@app.cell(hide_code=True)
def check_within(box_within):
    result_within = L.check("share-within", box_within.value)
    result_within
    return (result_within,)


@app.cell(hide_code=True)
def hints_within():
    L.hints(
        "share-within",
        [
            "Reken het gemiddelde en de standaardafwijking uit, en daarmee de twee randen van de band.",
            "Een boolean mask houdt de waarden boven de onderrand; een tweede mask, op wat er over is, houdt die onder de bovenrand. Deel hoeveel er over zijn door hoeveel er waren.",
        ],
        why="Het gemiddelde is 5,5 en de standaardafwijking ongeveer 2,87, dus de band loopt van ongeveer 2,63 tot 8,37. De waarden 3 tot en met 8 liggen erbinnen: 6 van de 10.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De kansen uit [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl) en de verwachtingswaarde en variantie uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) krijgen nu drie vormen: Bernoulli voor ja of nee, categorisch voor de klassen uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), en normaal voor metingen. In de volgende les, *Entropie: gemiddelde verrassing*, meet je hoe onzeker een verdeling is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_density, result_within):
    L.wrap_up(choice_which, result_density, result_within)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
