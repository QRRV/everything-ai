# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les info-entropy. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("info-entropy", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sure():
    quick_sure = L.choice(
        "quick-surprise",
        question="Een weermodel gaf regen op maandag de kans 0,8 en op dinsdag de kans 0,2. Het regende op allebei de dagen. Op welke dag was de verrassing groter?",
        options={
            "right": "Dinsdag: toen vond het model regen onwaarschijnlijk",
            "monday": "Maandag: toen was de kans het grotere getal",
            "same": "Geen van beide: het regende op allebei de dagen, dus de verrassing is gelijk",
        },
        correct="right",
        why={
            "monday": "De verrassing is min de log van de kans: hoe groter de kans, hoe kleiner de verrassing. Zie [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "same": "De verrassing hangt af van de kans die het model gaf aan wat er gebeurde, niet alleen van wat er gebeurde. Zie [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "right": "$-\\log(0{,}2) \\approx 1{,}61$ en $-\\log(0{,}8) \\approx 0{,}22$: wat het model onwaarschijnlijk vond, verrast het meer.",
        },
    )
    quick_sure
    return (quick_sure,)


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="Een getal is 1, 2 of 3, met de kansen 0,2; 0,5 en 0,3. Wat is de verwachtingswaarde?",
        options={
            "middle": "2: de middelste waarde",
            "likely": "2, want 2 is het waarschijnlijkst",
            "right": "2,1: elke waarde keer haar kans, opgeteld",
        },
        correct="right",
        why={
            "middle": "De middelste waarde telt alleen als de kansen eromheen in evenwicht zijn; hier is 3 waarschijnlijker dan 1. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "likely": "De waarschijnlijkste waarde is niet het gemiddelde op de lange duur: de andere tellen ook mee. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "1 · 0,2 + 2 · 0,5 + 3 · 0,3 = 0,2 + 1 + 0,9 = 2,1.",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Hoeveel informatie zit er in een bericht? Een muntworp vertelt je minder dan een worp met een dobbelsteen, en een voorspelling van zon in de woestijn vertelt je bijna niets. Informatietheorie meet dit in bits. Hetzelfde getal zegt hoe ver een bestand te comprimeren is, en in de volgende lessen leidt het tot de loss waarmee language models getraind en vergeleken worden.

    **Een voorbeeld uit het dagelijks leven.** Bij het spel Twintig vragen zoek je met ja-of-neevragen uit waar iemand aan denkt. Een goede vraag splitst wat er nog over is in twee helften, dus twintig vragen kunnen ongeveer een miljoen dingen uit elkaar houden.
    """)
    return


@app.cell(hide_code=True)
def predict_bits():
    guess_bits = L.guess(
        "plays-bits",
        question=f"De toneelstukken van Shakespeare gebruiken {C.kinds()} verschillende tekens. Een computer slaat elk teken op in 8 bits, en als alle {C.kinds()} even vaak voorkwamen, zou een code {L.number(C.uniform_bits(), 2)} bits per teken nodig hebben. Een slimmere code geeft de veelvoorkomende tekens korte codes en de zeldzame lange. Hoeveel bits per teken denk je dat die gemiddeld minstens nodig heeft?",
        options={
            "six": "Ongeveer 5,9: hoe vaak een teken voorkomt, maakt weinig uit",
            "right": "Ongeveer 4,8: de veelvoorkomende tekens schelen veel",
            "three": "Ongeveer 3,1: de veelvoorkomende tekens besparen nog meer",
        },
        reveal=f"**Ongeveer 4,8: {L.number(C.plays_entropy(), 2)} bits.** Dat is de entropie van de tekens van de toneelstukken: de spatie alleen is al ongeveer 15% ervan, en `e` ongeveer 8%. Geen code die naar één teken tegelijk kijkt, doet het gemiddeld beter. gzip, dat ook naar de tekens ervoor kijkt, haalt {L.number(C.gzip_bits(), 2)} bits per teken: de context helpt.",
    )
    guess_bits
    return (guess_bits,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **bit** is het antwoord op één ja-of-neevraag waarvan de twee antwoorden even waarschijnlijk zijn. De verrassing $-\log(p)$ uit [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl), met de log met grondtal 2, meet je in bits: een uitkomst met kans $p$ draagt $-\log_2(p)$ bits. Een eerlijke munt geeft 1 bit; één uit 8 even waarschijnlijke uitkomsten geeft 3 bits, de drie ja-of-neevragen die 8 halveren tot 4, 2 en 1.

        **Entropie** is de verwachte verrassing van een verdeling: de verrassing van elke uitkomst, gewogen met haar kans, opgeteld, als een verwachtingswaarde zoals in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
        """),
        L.formula(r"H(p) = -\sum_i p_i \log_2 p_i"),
        mo.md(r"""
        Entropie is 0 als één uitkomst zeker is, en het grootst, $\log_2 K$, als alle $K$ uitkomsten even waarschijnlijk zijn. In bits is ze een ondergrens: volgens een stelling van Claude Shannon uit 1948 kan geen code die elk symbool apart codeert, gemiddeld minder bits per symbool gebruiken, en heeft geen strategie met ja-of-neevragen gemiddeld minder vragen nodig. De beste codes komen er tot op minder dan 1 bit bij, en halen haar precies als elke kans 1/2, 1/4, 1/8 enzovoort is. Met de natuurlijke log, zoals bij log loss, is de eenheid de **nat**; 1 nat is ongeveer 1,44 bits.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **bit** | | de eenheid van informatie: het antwoord op één eerlijke ja-of-neevraag | |
        | **entropie**, $H(p)$ | H van p | de verwachte verrassing van een verdeling, in bits: hoe onzeker je bent voordat je de uitkomst ziet | `-np.sum(p * np.log2(p))` |
        | $\log_2$ | log met grondtal 2 | de exponent van 2 die een getal geeft: $\log_2 8 = 3$ | `np.log2(x)` |
        | **nat** | | de eenheid van informatie met de natuurlijke log in plaats van $\log_2$; 1 nat is ongeveer 1,44 bits | `np.log(x)` |
        """),
        L.watch_out("Entropie meet onzekerheid, geen waarde: een rij willekeurige tekens, elk van de 65 even waarschijnlijk, heeft meer entropie per teken dan een bladzijde Shakespeare. En ze bekijkt elk symbool apart; als symbolen van elkaar afhangen, zoals letters in woorden, kan een code die de context gebruikt eronder komen, zoals gzip deed."),
        mo.md(r"""
        De figuur: de entropie van een munt die met kans $p$ op kop valt. Ze is 0 voor een munt die altijd of nooit op kop valt, en het grootst, 1 bit, voor een eerlijke munt.
        """),
        mo.Html(C.coin_picture({"title": "De entropie van een munt", "x": "kans op kop", "y": "entropie (bits)",
                                "point": "p = {p}: {h} bits", "number": L.number,
                                "alt": "Een kromme in de vorm van een boog, van 0 bits bij kans 0, omhoog naar 1 bit bij kans 0,5, en omlaag naar 0 bits bij kans 1. Gemarkeerd: 1 bit bij 0,5, en 0,47 bits bij 0,9."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de entropie van een verdeling te vinden:

    1. Zet de kansen van alle uitkomsten op een rij; samen zijn ze 1.
    2. Laat de uitkomsten met kans 0 weg: die gebeuren nooit en voegen geen verrassing toe.
    3. Reken de verrassing van elke uitkomst uit in bits, $-\log_2(p)$.
    4. Weeg elke verrassing met haar kans en tel ze op; vergelijk met $\log_2 K$, de grootst mogelijke waarde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De entropie van de soorten van de 344 Palmer-pinguïns: Adélie, Gentoo en Chinstrap. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_entropy():
    import numpy as np

    # 1. de kansen, uit de aantallen van elke soort
    _counts = np.array(C.species_counts())
    _p = _counts / _counts.sum()
    # 2. geen soort heeft hier kans 0, maar de regel houdt het recept heel
    _p = _p[_p > 0]
    # 3. de verrassing van elke soort, in bits
    _surprise = -np.log2(_p)
    print(_surprise)
    # 4. de grootst mogelijke waarde bij 3 soorten staat er; weeg de verrassingen met _p, tel ze op, en print dat ook
    print(np.log2(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    _p = C.species_probs()
    mo.md(f"""
    Chinstrap is het zeldzaamst, dus verrast het meest: {L.number(_p[2], 2)} als kans geeft ongeveer {L.number(C.surprise_bits(_p[2]), 2)} bits. Met stap 4 af is de entropie {L.number(C.species_entropy(), 3)} bits, iets onder de {L.number(C.surprise_bits(1 / 3), 3)} van drie even vaak voorkomende soorten: de soorten komen niet helemaal even vaak voor, dus raden is iets makkelijker.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_coins():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De onzekerste uitkomst
    """)
    return


@app.cell(hide_code=True)
def choice_coins_cell():
    choice_coins = L.choice(
        "most-entropy",
        question="Welke hiervan heeft de hoogste entropie?",
        options={
            "coin": "Een eerlijke munt: twee kanten, elk de helft van de keren",
            "skewed": "Een rad met vier vakken: één vak komt in 70% van de keren, de andere drie elk in 10%",
            "right": "Een rad met drie gelijke vakken, elk een derde van de keren",
        },
        correct="right",
        why={
            "coin": "Een eerlijke munt heeft 1 bit, het meeste wat twee uitkomsten kunnen hebben. Met drie even waarschijnlijke uitkomsten weet je vooraf nog minder.",
            "skewed": "Vier uitkomsten, maar één komt het vaakst, dus je raadt vaak goed: ongeveer 1,36 bits. Drie gelijke vakken geven meer.",
            "right": "Drie even waarschijnlijke uitkomsten: $\\log_2 3 \\approx 1{,}585$ bits. Meer uitkomsten geven alleen meer entropie als ze niet scheef verdeeld zijn: het rad met vier vakken heeft ongeveer 1,36 bits, de munt 1.",
        },
    )
    choice_coins
    return (choice_coins,)


@app.cell(hide_code=True)
def text_exercise_entropy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Entropie als functie

        Schrijf `entropy(p)`: de entropie in bits van de kansen in de array `p`, met het recept. De check roept `entropy(np.array([0.5, 0.25, 0.25]))` aan, en dat moet `1.5` geven; hij probeert ook andere verdelingen, ook een met een kans van 0.
        """),
        L.task("Laat `entropy` de verwachte verrassing van `p` geven, in bits."),
    ])
    return


@app.cell(hide_code=True)
def exercise_entropy():
    box_entropy = L.code_box("entropy", lines=7)
    box_entropy
    return (box_entropy,)


@app.cell(hide_code=True)
def check_entropy(box_entropy):
    result_entropy = L.check("entropy", box_entropy.value)
    result_entropy
    return (result_entropy,)


@app.cell(hide_code=True)
def hints_entropy():
    L.hints(
        "entropy",
        [
            "Stap 2 tot en met 4 van het recept: laat eerst de nullen weg, dan de verrassing in bits, dan wegen en optellen.",
            "Een boolean mask, `p[p > 0]`, laat de nullen weg. `np.log2` geeft de log met grondtal 2 van elk getal tegelijk.",
        ],
        why="De verrassingen zijn 1, 2 en 2 bits. Gewogen: $0{,}5 \\cdot 1 + 0{,}25 \\cdot 2 + 0{,}25 \\cdot 2 = 1{,}5$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bits():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoe klein kan een tekst worden?

        Schrijf `min_bits(text)`: het kleinste aantal bits dat een code voor de hele tekst nodig heeft als ze alleen gebruikt hoe vaak elk teken voorkomt. Tel elk teken, zet de aantallen om in kansen, en vermenigvuldig de entropie met het aantal tekens. De check roept `min_bits("aaab")` aan, en dat moet ongeveer `3.245` geven; hij probeert ook andere teksten.
        """),
        L.task("Laat `min_bits` de entropie van de tekens van `text` keer de lengte ervan geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bits():
    box_bits = L.code_box("min-bits", lines=16)
    box_bits
    return (box_bits,)


@app.cell(hide_code=True)
def check_bits(box_bits):
    result_bits = L.check("min-bits", box_bits.value)
    result_bits
    return (result_bits,)


@app.cell(hide_code=True)
def hints_bits():
    L.hints(
        "min-bits",
        [
            "Tel met een dictionary, zoals in [Dictionaries](/course/open/py-dicts?lang=nl): één sleutel per teken, met het aantal als waarde. Elk aantal gedeeld door de lengte van de tekst is dan een kans.",
            "De entropie van die kansen is het aantal bits per teken; de hele tekst heeft dat keer zijn lengte nodig.",
        ],
        why="In \"aaab\" heeft `a` kans 0,75 en `b` 0,25. De entropie is $0{,}75 \\cdot 0{,}415 + 0{,}25 \\cdot 2 \\approx 0{,}811$ bits per teken, en 4 tekens hebben ongeveer 3,245 bits nodig.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De verrassing uit [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl), in bits, en gemiddeld als verwachtingswaarde zoals in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl), is de entropie: hoe onzeker een verdeling is, en met hoe weinig bits een code toe kan. In de volgende les, *Cross-entropy en KL-divergentie*, meet je de verrassing als je de ene verdeling verwacht en de waarheid een andere is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_coins, result_entropy, result_bits):
    L.wrap_up(choice_coins, result_entropy, result_bits)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
