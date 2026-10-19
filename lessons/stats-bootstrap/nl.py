# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les stats-bootstrap. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("stats-bootstrap", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_integers():
    quick_integers = L.choice(
        "quick-integers",
        question="`rng` is een random number generator. Welke hiervan kan `rng.integers(0, 3, size=4)` geven?",
        options={
            "right": "`[0, 2, 2, 1]`: vier getallen van 0 tot en met 2",
            "three": "`[1, 3, 2, 3]`: vier getallen van 1 tot en met 3",
            "short": "`[0, 1, 2]`: elk getal één keer",
        },
        correct="right",
        why={
            "three": "Het tweede getal van `rng.integers` doet niet mee: 3 komt nooit. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "short": "`size=4` vraagt om vier getallen, en een getal kan vaker voorkomen. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "right": "Vier hele getallen vanaf 0 tot 3, waarbij 3 zelf niet meedoet; hetzelfde getal kan vaker voorkomen.",
        },
    )
    quick_integers
    return (quick_integers,)


@app.cell(hide_code=True)
def quick_check_median():
    quick_median = L.choice(
        "quick-median",
        question="Welk getal geeft `np.quantile(x, 0.5)`?",
        options={
            "mean": "Het gemiddelde van `x`: het midden",
            "right": "De mediaan van `x`: de helft van de waarden ligt eronder",
            "half": "De helft van de grootste waarde van `x`: 0,5 keer de top",
        },
        correct="right",
        why={
            "mean": "Het gemiddelde is de som gedeeld door het aantal; een kwantiel telt juist waarden. Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "half": "Een kwantiel is een waarde van `x` waaronder een deel van de waarden ligt, geen deel van een waarde. Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "right": "Onder het 0,5-kwantiel ligt de helft van de waarden: de mediaan. Deze les gebruikt het 0,025- en het 0,975-kwantiel.",
        },
    )
    quick_median
    return (quick_median,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je rekent een getal uit met de data die je hebt: een model heeft 87 van de 100 testvoorbeelden goed, of 68 pinguïns hebben een gemiddelde snavellengte van 48,8 mm. Met andere testvoorbeelden, of andere pinguïns, zou het getal een beetje anders uitvallen. Voordat je zegt dat het ene model beter is dan het andere, of dat een resultaat klopt, wil je weten hoe ver zo'n getal alleen door toeval kan verschuiven. Je hebt maar die ene set data, en voor de meeste getallen is er geen formule.

    **Een voorbeeld uit het dagelijks leven.** Je proeft één lepel om te weten hoe zout een hele pan soep is. Een tweede lepel smaakt net iets anders, en hoeveel die verschilt, zegt je hoeveel je op de eerste kunt vertrouwen.
    """)
    return


@app.cell(hide_code=True)
def predict_accuracy():
    guess_accuracy = L.guess(
        "accuracy-range",
        question="Een model heeft 87 van de 100 testvoorbeelden goed: een accuracy van 87%. Zijn echte accuracy is het deel dat het goed zou hebben van eindeloos veel voorbeelden van dezelfde soort. Tussen welke waarden denk je dat die ligt?",
        options={
            "close": "Van ongeveer 86% tot 88%: het is hetzelfde model",
            "wide": "Van ongeveer 50% tot 100%: 100 voorbeelden zeggen bijna niets",
            "right": "Van ongeveer 80% tot 93%: toeval verschuift een paar voorbeelden",
        },
        reveal="**Van ongeveer 80% tot 93%.** Met 100 voorbeelden verschuift alleen al het toeval het aantal met een paar, omhoog of omlaag. Resample de 100 uitkomsten 2000 keer, zoals deze les laat zien, en de middelste 95% van de accuracy's loopt van 0,80 tot 0,93: het 95%-betrouwbaarheidsinterval van de echte accuracy. Een tweede model met 85% op 100 andere voorbeelden is dus niet aantoonbaar slechter.",
    )
    guess_accuracy
    return (guess_accuracy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een getal dat je uit data uitrekent, zoals een gemiddelde of een accuracy, komt uit je **steekproef**: de voorbeelden die je toevallig hebt. Een andere steekproef van dezelfde soort geeft een net iets ander getal. Hoeveel het van steekproef tot steekproef varieert, is de **standaardfout**: de standaardafwijking die het getal over veel steekproeven zou hebben.

        Je hebt maar één steekproef. De **bootstrap** maakt er nieuwe van: trek $n$ waarden uit je $n$ waarden **met teruglegging**, zodat een waarde twee keer of helemaal niet getrokken kan worden, en reken het getal opnieuw uit. Zo'n **resample** is even groot als je data en lijkt erop, maar wijkt een beetje af, zoals een andere steekproef zou doen. Doe dat een paar duizend keer, en de spreiding van de getallen die je krijgt laat zien hoeveel je getal door toeval beweegt.

        Een **betrouwbaarheidsinterval** geeft dat als een bereik. Het 95%-interval van de bootstrap loopt van het 0,025-kwantiel tot het 0,975-kwantiel van de geresamplede getallen: de middelste 95%, met aan elke kant 2,5% eraf. Op deze manier gemaakt vangt het interval de echte waarde, die eindeloos veel data zou geven, in ongeveer 95 van elke 100 steekproeven.

        Voor een gemiddelde is er ook een formule: de standaardfout is ongeveer de standaardafwijking van de data gedeeld door de wortel uit het aantal voorbeelden.
        """),
        L.formula(r"\mathrm{SE} \approx \frac{\text{standaardafwijking van de data}}{\sqrt{n}}"),
        mo.md(r"""
        Vier keer zoveel voorbeelden halveren de standaardfout, en daarmee de breedte van het interval. De bootstrap geeft hetzelfde zonder formule, ook voor getallen die er geen hebben.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **steekproef** | | de voorbeelden die je toevallig hebt, uit alle voorbeelden die er hadden kunnen zijn | `x` |
        | **standaardfout**, $\mathrm{SE}$ | standaardfout | hoeveel een getal uit een steekproef van steekproef tot steekproef varieert: de standaardafwijking ervan over veel steekproeven | `means.std()` |
        | **bootstrap** | | nieuwe steekproeven maken door met teruglegging uit je eigen data te trekken, om te zien hoeveel een getal varieert | |
        | **met teruglegging** | | elke waarde gaat terug nadat hij getrokken is, dus hij kan opnieuw getrokken worden | `rng.choice(x, size=len(x))` |
        | **resample** | | zo'n nieuwe steekproef, even groot als de data | |
        | **betrouwbaarheidsinterval** | | een bereik rond een getal dat, op deze manier gemaakt, de echte waarde vangt in een gegeven deel van de steekproeven, zoals 95% | `np.quantile(means, [0.025, 0.975])` |
        | `rng.choice(x, size=n)` | | `n` waarden met teruglegging getrokken uit `x` | |
        """),
        L.watch_out("De bootstrap kent alleen je steekproef. Is die klein, bijvoorbeeld 5 waarden, of lijkt die niet op de rest, bijvoorbeeld pinguïns van één eiland, dan is het interval te smal of ligt het op de verkeerde plek, en meer resamples lossen dat niet op: alleen meer of betere data. En een 95%-interval van een gemiddelde bevat geen 95% van de waarden: het gaat over het gemiddelde, niet over losse pinguïns."),
        mo.md(r"""
        De figuur: de accuracy's van 2000 resamples van de 100 testuitkomsten uit de gok. Ze stapelen zich op rond 0,87, en de middelste 95% loopt van 0,80 tot 0,93: het 95%-betrouwbaarheidsinterval van de accuracy. Het is niet helemaal symmetrisch: boven 0,87 is minder ruimte dan eronder.
        """),
        mo.Html(C.boot_picture({"title": "2000 resamples van 100 testuitkomsten", "x": "accuracy van de resample", "y": "resamples",
                                "band": "middelste 95%: {low} tot {high}", "number": L.number,
                                "alt": "Een histogram van 2000 accuracy's, van ongeveer 0,77 tot 0,97, het hoogst rond 0,87. Stippellijnen bij 0,80 en 0,93 markeren de middelste 95%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om met de bootstrap het 95%-betrouwbaarheidsinterval van een getal te vinden:

    1. Reken het getal uit op je data, bijvoorbeeld het gemiddelde.
    2. Trek een resample: $n$ waarden uit je data met teruglegging, en reken het getal opnieuw uit.
    3. Herhaal stap 2 een paar duizend keer; de standaardafwijking van de uitkomsten is de standaardfout.
    4. Haal aan elke kant 2,5% eraf: het 0,025- en het 0,975-kwantiel van de uitkomsten zijn het 95%-betrouwbaarheidsinterval.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De snavellengtes van 68 Chinstrap-pinguïns op Dream, een eiland van de Palmer-archipel: `C.chinstrap_bills()` geeft ze als array. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_bootstrap():
    import numpy as np

    _x = C.chinstrap_bills()
    # 1. het getal: de gemiddelde snavellengte van de 68 pinguïns
    print(len(_x), _x.mean())
    _rng = np.random.default_rng(0)
    _means = []
    for _i in range(2000):
        # 2. een resample: 68 waarden met teruglegging getrokken, en het gemiddelde ervan
        _resample = _rng.choice(_x, size=len(_x))
        _means.append(_resample.mean())
    _means = np.array(_means)
    # 3. de standaardfout: de spreiding van de 2000 gemiddelden, en de formule ernaast
    print(_means.std(), _x.std() / len(_x) ** 0.5)
    # 4. de onderkant van het 95%-interval staat er; voeg de bovenkant toe, het 0,975-kwantiel
    print(np.quantile(_means, 0.025))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De 68 pinguïns hebben een gemiddelde snavellengte van 48,83 mm. De 2000 geresamplede gemiddelden hebben een spreiding van 0,41 mm, dicht bij de formule, 3,31 / $\sqrt{68}$ = 0,40 mm. De onderkant van het interval is 48,03 mm; met stap 4 af is de bovenkant 49,61 mm. Het 95%-betrouwbaarheidsinterval van de gemiddelde snavellengte loopt dus van ongeveer 48,0 tot 49,6 mm, voor Chinstrap-pinguïns op Dream, het enige eiland waar alle 68 gemeten zijn. Het zegt niet dat het echte gemiddelde met een kans van 95% in dit ene bereik ligt: het ligt erin of niet. Het zegt dat intervallen die zo gemaakt zijn het echte gemiddelde in ongeveer 95 van elke 100 steekproeven vangen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_four():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Negen keer zoveel data
    """)
    return


@app.cell(hide_code=True)
def choice_four_cell():
    choice_four = L.choice(
        "more-data",
        question="Je meet negen keer zoveel Chinstrap-pinguïns: 612 in plaats van 68. Hoe breed wordt het 95%-interval van hun gemiddelde snavellengte ongeveer?",
        options={
            "right": "Ongeveer een derde zo breed: de standaardfout krimpt met de wortel uit n",
            "ninth": "Ongeveer een negende zo breed: er is negen keer zoveel data",
            "same": "Ongeveer even breed: de bootstrap haalt altijd aan elke kant 2,5% eraf",
        },
        correct="right",
        why={
            "ninth": "De standaardfout is de standaardafwijking gedeeld door $\\sqrt{n}$: negen keer $n$ deelt hem door $\\sqrt{9} = 3$, niet door 9.",
            "same": "Er gaat evenveel af, maar met meer data liggen de geresamplede gemiddelden dichter bij elkaar, dus de middelste 95% is smaller.",
            "right": "De standaardfout is de standaardafwijking gedeeld door $\\sqrt{n}$: negen keer $n$ deelt hem door $\\sqrt{9} = 3$. Elk extra cijfer nauwkeurigheid kost honderd keer zoveel data.",
        },
    )
    choice_four
    return (choice_four,)


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De bootstrap-gemiddelden

        Schrijf `bootstrap_means(x, n_boot, seed)`: hij maakt een random number generator met `seed` en geeft de gemiddelden van `n_boot` resamples van de array `x` terug, als array: stap 2 en 3 van het recept. De check roept `bootstrap_means(np.array([2, 4, 6, 8]), 1000, 0)` aan, en dat moet 1000 gemiddelden rond 5 geven, met een spreiding van ongeveer 1,1; hij probeert ook andere data. Willekeurige getallen vallen met een andere seed of andere aanroepen anders uit, dus de check kijkt hoeveel gemiddelden er zijn, waar ze liggen en hoeveel ze spreiden, en of dezelfde seed dezelfde gemiddelden geeft.
        """),
        L.task("Laat `bootstrap_means` de gemiddelden van `n_boot` resamples van `x` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("boot-means", lines=9)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("boot-means", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "boot-means",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld: een loop van `n_boot` rondes, elk met `rng.choice(x, size=len(x))` en het gemiddelde daarvan.",
            "Maak `rng = np.random.default_rng(seed)` één keer, vóór de loop; verzamel de gemiddelden in een lijst met `append`, en geef `np.array` van die lijst terug.",
        ],
        why="Voor `[2, 4, 6, 8]` liggen de gemiddelden van 1000 resamples rond 5, het gemiddelde van de data, met een spreiding van ongeveer de standaardafwijking van de data, 2,24, gedeeld door $\\sqrt{4} = 2$: ongeveer 1,1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_interval():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het interval

        Schrijf `interval(means, level)`: het betrouwbaarheidsinterval dat het deel `level` van de gemiddelden in het midden heeft, als twee getallen, laag en hoog, met stap 4 van het recept. Bij `level` 0,95 zijn dat het 0,025- en het 0,975-kwantiel. De check roept `interval(np.arange(1, 101), 0.9)` aan, en dat moet `[5.95, 95.05]` geven; hij probeert ook andere niveaus.
        """),
        L.task("Laat `interval` de onder- en bovenkant geven van het middelste deel `level` van `means`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_interval():
    box_interval = L.code_box("interval", lines=6)
    box_interval
    return (box_interval,)


@app.cell(hide_code=True)
def check_interval(box_interval):
    result_interval = L.check("interval", box_interval.value)
    result_interval
    return (result_interval,)


@app.cell(hide_code=True)
def hints_interval():
    L.hints(
        "interval",
        [
            "Het deel buiten het interval is `1 - level`; de helft daarvan gaat naar elke kant.",
            "`np.quantile(means, [low_fraction, high_fraction])` geeft beide kanten tegelijk.",
        ],
        why="Bij `level` 0,9 blijft 0,1 over: 0,05 aan elke kant. Het 0,05- en het 0,95-kwantiel van de getallen 1 tot en met 100 zijn 5,95 en 95,05.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De standaardafwijking uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) zegt hoe ver waarden van hun gemiddelde liggen; de standaardfout zegt hoe ver een gemiddelde, of elk ander getal uit een steekproef, van het echte kan liggen. De random number generator uit [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) trekt de resamples, en de kwantielen uit [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) geven het interval. Geef vanaf nu bij elk getal dat je meet ook zijn interval: een verschil tussen twee modellen telt pas als het groter is dan toeval. De bootstrap komt terug in *Eén run is geen resultaat: seeds en variantie* en *Is het verschil echt?*. In de volgende les, *Gradient checking*, controleer je de gradients van een netwerk met de helling van de loss zelf.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_four, result_means, result_interval):
    L.wrap_up(choice_four, result_means, result_interval)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
