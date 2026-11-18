# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-uber-crash. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-uber-crash", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bias():
    quick_bias = L.choice(
        "quick-bias",
        question="Een systeem markeert niets, en de persoon die het controleert mist een probleem dat de eigen informatie wel liet zien. Hoe heet dat?",
        options={
            "gap": "Een responsibility gap, want niemand staat ervoor in",
            "right": "Automation bias",
            "fine": "Toezicht dat werkt zoals het moet, want het systeem beslist",
        },
        correct="right",
        why={
            "gap": "Een responsibility gap gaat over wie ervoor instaat wat een systeem deed. Zie *Het idee* in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
            "right": "Missen wat het systeem niet markeerde is een van de twee vormen van automation bias; de foute suggesties volgen is de andere.",
            "fine": "De persoon was er juist om te vangen wat het systeem miste. Zie *Het idee* in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
        },
    )
    quick_bias
    return (quick_bias,)


@app.cell(hide_code=True)
def quick_check_three():
    quick_three = L.choice(
        "quick-three",
        question="Toezicht werkt alleen als de persoon drie dingen heeft. Welke drie?",
        options={
            "course": "Een cursus over de wet, een handleiding en een manager die het werk nakijkt",
            "right": "Tijd, het geval zelf, en de macht om nee te zeggen",
            "trust": "Vertrouwen in het systeem, en snelheid",
        },
        correct="right",
        why={
            "course": "Dat kan helpen, maar het geeft de persoon geen ruimte om een fout te vangen. Zie *Het idee* in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
            "right": "Tijd om te kijken, informatie over het geval zelf en niet alleen een score, en de macht om nee te zeggen zonder daarvoor te boeten.",
            "trust": "Vertrouwen in het systeem is juist waar automation bias van gemaakt is. Zie *Het idee* in [Een mens in de loop](/course/open/rai-human-oversight?lang=nl).",
        },
    )
    quick_three
    return (quick_three,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In maart 2018 reed een testauto 's avonds zelf door Tempe, in Arizona, met één persoon achter het stuur die moest opletten. De auto reed een voetganger aan die met een fiets aan de hand de weg overstak; de voetganger kwam om het leven. Het systeem van de auto had de voetganger seconden eerder gedetecteerd; de persoon achter het stuur keek op een telefoon. De onderzoekers vonden niet één fout maar een hele rij, en een aantal daarvan waren keuzes in het ontwerp.

    **Een voorbeeld uit het dagelijks leven.** Houd een paar plakken kaas met gaten voor een lamp. Het licht komt er alleen door waar elke plak op dezelfde plek een gat heeft. Elke plak is een laag veiligheid; voor een ongeluk moet er in alle lagen tegelijk een gat zitten.
    """)
    return


@app.cell(hide_code=True)
def predict_detected():
    guess_detected = L.guess(
        "detected",
        question="De auto reed 's avonds ongeveer 71 kilometer per uur. Hoe lang voor het ongeluk denk je dat het systeem de voetganger voor het eerst detecteerde?",
        options={
            "half": "Ongeveer een halve seconde ervoor",
            "one": "Ongeveer anderhalve seconde",
            "right": "5,6 seconden ervoor",
        },
        reveal="**5,6 seconden ervoor**, eerst met de radar. Maar het herkende nooit een voetganger, en de auto begon pas 0,2 seconde voor het ongeluk af te remmen.",
    )
    guess_detected
    return (guess_detected,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het ongeluk.** Op zondag 18 maart 2018, om 21.58 uur, reed een testauto van de Advanced Technologies Group van Uber zelf over Mill Avenue in Tempe: een Volvo XC90 met het systeem voor automatisch rijden van Uber, dat nog in ontwikkeling was en al zo'n 19 minuten zelf reed. Elaine Herzberg, 49 jaar, stak met een fiets aan de hand de weg over, op een plek zonder zebrapad. De auto kwam aanrijden met 45 mijl per uur (72 kilometer per uur), de maximumsnelheid, en raakte de voetganger met 39 mijl per uur (63 kilometer per uur). Herzberg kwam om het leven. De NTSB, de Amerikaanse raad die ongelukken in het verkeer en vervoer onderzoekt, bracht in november 2019 haar rapport uit.

        **Wat het systeem zag.** Het systeem detecteerde de voetganger 5,6 seconden voor de botsing, eerst met de radar, als een voertuig. Daarna hield het de voetganger voor *other* (iets anders), weer voor een voertuig, sprong het een paar keer heen en weer tussen die twee, hield het de voetganger voor een fiets, *other*, en bij 1,2 seconde weer voor een fiets. Elke keer dat de soort veranderde, gebruikte het systeem niet meer wat het eerder van het object had gevolgd, de tracking history, dus kon het het pad niet goed voorspellen; en een object van de soort *other* behandelde het alsof het stilstond. Het herkende nooit een voetganger: het ontwerp hield geen rekening met voetgangers die buiten een zebrapad oversteken.

        **Wat het systeem deed.** Bij 1,2 seconde zag het systeem een noodgeval. Toen volgde *action suppression*: een seconde lang hield het systeem het remmen tegen, terwijl het het gevaar controleerde of de operator het overnam, en het waarschuwde de operator niet. Uber had dat ingebouwd uit zorg over vals alarm. Het systeem remde met hoogstens 7 m/s², en alleen als dat een botsing voorkwam, niet om een botsing minder zwaar te maken. 0,2 seconde voor het ongeluk klonk er een geluid en begon de auto af te remmen. De eigen noodrem van Volvo, die voetgangers kon detecteren, stond uit zodra het systeem van Uber reed: de radars van de twee gebruikten dezelfde frequenties, en de remmen konden niet kiezen tussen twee opdrachten. In een simulatie van Volvo, met ideale sensoren, voorkwam dat systeem de botsing bij 17 van 20 varianten van hoe de voetganger bewoog, en raakte het bij de andere 3 met minder dan 10 mijl (16 kilometer) per uur.

        **De persoon in de auto.** Er zat één operator op de bestuurdersstoel; tot de herfst van 2017 zette Uber twee mensen in elke auto. De operator keek via een telefoon naar een televisieprogramma, had 34 procent van de tijd dat de auto reed de blik omlaag, en keek vanaf 6 seconden voor het ongeluk 5 seconden lang omlaag. Uber controleerde maar zelden de camerabeelden van zijn operators.

        **Het oordeel.** De NTSB noemde als waarschijnlijke oorzaak dat de operator niet op de weg en het systeem lette, afgeleid door een telefoon. Daaraan droegen drie tekortkomingen van de groep van Uber bij: gebrekkige procedures om veiligheidsrisico's te beoordelen, toezicht op de operators dat niet werkte, en geen goede manier om te voorkomen dat operators te veel op het systeem leunden, alle drie een gevolg van een veiligheidscultuur die tekortschoot. Verdere factoren waren dat de voetganger buiten een zebrapad overstak, volgens het rapport onder invloed van drugs, en dat het vervoersdepartement van Arizona te weinig toezicht hield op de tests.

        **Daarna.** Uber trof nog dezelfde maand een schikking met de familie; de voorwaarden werden niet bekendgemaakt. In maart 2019 schreef de officier van justitie van Yavapai County dat er geen grond was voor strafrechtelijke aansprakelijkheid van Uber. De operator werd in 2020 aangeklaagd voor dood door schuld, erkende in juli 2023 schuld aan het in gevaar brengen van een ander, en kreeg drie jaar proeftijd onder toezicht.

        Geen enkele fout veroorzaakte het ongeluk alleen. Het systeem detecteerde de voetganger maar herkende nooit een mens, gooide bij elke verandering het pad weg, en hield de remmen tegen; een tweede systeem dat kon remmen stond uit. De persoon die dat allemaal moest opvangen, zat alleen in een auto waar eerst twee mensen in zaten, keek omlaag, en werd zelden gecontroleerd. Alleen die persoon werd strafrechtelijk vervolgd.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **veiligheidsbestuurder** | iemand in een zelfrijdende testauto die de weg en het systeem in de gaten houdt en moet ingrijpen als er iets misgaat; de NTSB noemt die persoon de operator van het voertuig |
        """),
        L.watch_out("Gedetecteerd is niet herkend. Het systeem zag 5,6 seconden voor het ongeluk *iets*, maar wist niet wat het was of waar het heen ging. \"De auto zag de voetganger niet\" is niet wat er gebeurde."),
        mo.Html(C.picture({"title": "De laatste zes seconden voor het ongeluk",
                           "rows": ["Waar het systeem de voetganger voor hield", "Wat het systeem deed", "De operator"],
                           "kinds": {"vehicle": "voertuig", "other": "iets anders", "bicycle": "fiets"},
                           "switches": "heen en weer", "held": "noodgeval: remmen tegengehouden",
                           "key": "Middelste rij, omlijnd: het remmen wordt tegengehouden; gevuld: een geluid, en de auto remt af. "
                                  "Onderste rij, omlijnd: de operator kijkt omlaag; gevuld: kijkt naar de weg.",
                           "down": "kijkt omlaag", "impact": "ongeluk, 63 km/u",
                           "xlabel": "seconden voor het ongeluk", "decimal": ",",
                           "alt": "Een tijdlijn van de laatste zes seconden. Het systeem detecteert de voetganger bij 5,6 seconden als voertuig, en springt daarna tussen iets anders, voertuig en fiets. Bij 1,2 seconde ziet het een noodgeval en houdt het het remmen een seconde tegen; bij 0,2 seconde klinkt er een geluid en remt de auto af. De operator kijkt omlaag van 6 seconden tot 1 seconde voor het ongeluk."})),
        mo.md(r"""
        Uit het rapport van de NTSB. Tussen 3,8 en 2,7 seconden sprong het systeem nog een paar keer heen en weer tussen voertuig en iets anders.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de lagen te zoeken in een ongeluk met een geautomatiseerd systeem:

    1. Zet alle lagen op een rij die het hadden kunnen tegenhouden: wat het systeem waarneemt, wat het voorspelt, hoe het remt, een tweede systeem, de persoon, de controles van het bedrijf, de toezichthouder.
    2. Schrijf bij elke laag op wat die deed, en of dat een keuze in het ontwerp was.
    3. Markeer de lagen die iemand uitzette of weghaalde, en waarom.
    4. Vraag hoe de persoon in de loop geholpen werd om op te letten, en hoe die persoon gecontroleerd werd.
    5. Vergelijk waar de schuld terechtkwam met waar de gaten zaten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De regel uit het rapport, vereenvoudigd: als de soort van een object verandert, vergeet het systeem het pad tot dan toe en telt het vanaf dat moment opnieuw. `_kinds` bevat een object dat de course verzon, elke seconde een keer gezien, met de seconden voor een ongeluk als sleutels. De loop gaat met `_time` langs de momenten; als de soort anders is dan `_previous`, schuift `_start` naar dat moment, en `_previous` onthoudt elke soort. `_history` is hoeveel seconden pad het systeem bij `_now` nog telt. -5,0 betekent 5 seconden ervoor. `_previous` begint als een lege string, dus telt het eerste moment altijd als verandering en krijgt `_start` een waarde. Voer de cel uit.
    """)
    return


@app.cell
def example_history():
    _kinds = {-5.0: "bicycle", -4.0: "bicycle", -3.0: "other", -2.0: "other", -1.0: "other"}
    _now = -1.0

    _previous = ""
    for _time in _kinds:
        if _kinds[_time] != _previous:
            _start = _time
        _previous = _kinds[_time]
    _history = _now - _start
    print(_start, _history)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het object werd 4 seconden gezien, maar bij -1,0 telt het systeem er maar 2,0 van: vanaf -3,0, toen het *other* werd. Nu dezelfde regel voor de momenten uit het rapport: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_history():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoeveel van het pad was er nog?

        De momenten uit het rapport van de NTSB. `kinds` bevat de seconden voor het ongeluk en waar het systeem de voetganger voor hield, met de woorden van het rapport: `vehicle` (voertuig), `other` (iets anders) en `bicycle` (fiets). Het heen en weer springen tussen -3,8 en -2,7 is weggelaten, en dat verandert het antwoord niet. `now` is -1,2, het moment waarop het systeem het noodgeval zag. Vervang de `...` door de regels van het uitgewerkte voorbeeld, met de namen zonder hun liggende streepje: `previous`, `time`, `start` en `history`. De loop gaat met `time` langs `kinds`. Laat de `print` weg: de laatste regel is `history`, zodat de check het ziet. De check probeert ook andere momenten.
        """),
        L.task("Reken `history` uit: hoeveel seconden van het pad van de voetganger het systeem nog telde toen het het noodgeval zag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_history():
    box_history = L.code_box("history", lines=12)
    box_history
    return (box_history,)


@app.cell(hide_code=True)
def check_history(box_history):
    result_history = L.check("history", box_history.value)
    result_history
    return (result_history,)


@app.cell(hide_code=True)
def hints_history():
    L.hints(
        "history",
        [
            "Het uitgewerkte voorbeeld doet dit voor een verzonnen object. Neem de regels over, zonder de `print`, en haal bij elke naam het liggende streepje weg.",
            "Begin `previous` als `\"\"`. Schuif in de loop `start` naar `time` als de soort anders is dan `previous`, en laat `previous` daarna de soort onthouden. Trek na de loop af.",
            "In de loop: `if kinds[time] != ___:` met daaronder `start = time`; na de loop: `history = now - ___`.",
        ],
        why="`previous = \"\"`, dan `for time in kinds:` met `if kinds[time] != previous:` en daaronder `start = time`, dan `previous = kinds[time]` in de loop, en na de loop `history = now - start`. Dat geeft `0.0`: bij -1,2 was de soort net veranderd.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.md(r"""
    ### Oefening 2. Het oordeel

    Zes bevindingen van de NTSB. Deel elke bevinding in zoals de NTSB dat deed.
    """)
    return


@app.cell(hide_code=True)
def sort_verdict_cell():
    sort_verdict = L.sort("verdict")
    sort_verdict
    return (sort_verdict,)


@app.cell(hide_code=True)
def check_verdict(sort_verdict):
    result_verdict = L.check("verdict", sort_verdict)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "De NTSB noemde één waarschijnlijke oorzaak, en daarna factoren die bijdroegen: sommige van de groep van Uber, sommige van anderen.",
            "De groep van Uber had er drie: over risico's, over de operators, en over leunen op het systeem.",
            "Oorzaak: \\_\\_\\_. Uber: de risico's, \\_\\_\\_ en het vertrouwen. Verder: de oversteek en \\_\\_\\_.",
        ],
        why="**De waarschijnlijke oorzaak:** de telefoon. **Droeg bij, Uber:** de risico's, het toezicht en het vertrouwen. **Verdere factoren:** de oversteek en de staat.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_court():
    mo.md(r"""
    ### Oefening 3. Wat gebeurde er bij de rechter?
    """)
    return


@app.cell(hide_code=True)
def choice_court_cell():
    choice_court = L.choice(
        "court",
        question="Hoe liep het ongeluk strafrechtelijk af?",
        options={
            "uber": "Uber werd veroordeeld, en de operator vrijgesproken",
            "right": "De operator erkende schuld; Uber werd niet vervolgd",
            "nobody": "Niemand werd vervolgd, want het systeem reed de auto zelf",
        },
        correct="right",
        why={
            "uber": "Uber werd nooit vervolgd: in 2019 zag de officier van justitie van Yavapai County geen grond voor strafrechtelijke aansprakelijkheid van Uber. Zie *Het idee*.",
            "right": "In 2019 zag de officier van justitie van Yavapai County geen grond voor strafrechtelijke aansprakelijkheid van Uber; in 2023 erkende de operator schuld aan het in gevaar brengen van een ander en kreeg drie jaar proeftijd onder toezicht. De schuld kwam terecht bij de persoon die het dichtst bij de machine stond: de morele kreukelzone uit [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl).",
            "nobody": "De operator werd in 2020 aangeklaagd voor dood door schuld. Zie *Het idee*.",
        },
    )
    choice_court
    return (choice_court,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een mens in de loop](/course/open/rai-human-oversight?lang=nl) zag je dat iemand die een systeem bewaakt dat meestal gelijk heeft, vaak mist wat het niet markeert, en dat toezicht tijd, informatie en de macht om nee te zeggen nodig heeft. Hier moest de operator de hele rit alleen opletten, kreeg die geen waarschuwing toen het systeem de remmen tegenhield, en werd die zelden gecontroleerd. In [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl) leerde je de morele kreukelzone kennen: bij de rechter kwam de schuld alleen bij de operator terecht. Hiermee eindigt de module *Ethiek*; de volgende module, *Fairness*, begint met *Directe en indirecte discriminatie*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_court, result_history, result_verdict):
    L.wrap_up(result_history, result_verdict, choice_court)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
