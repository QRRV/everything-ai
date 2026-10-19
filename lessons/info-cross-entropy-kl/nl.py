# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les info-cross-entropy-kl. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow entropy

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("info-cross-entropy-kl", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl) en [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_die():
    quick_die = L.choice(
        "quick-die",
        question="Een dobbelsteen heeft 8 kanten, allemaal even waarschijnlijk. Wat is de entropie van één worp?",
        options={
            "eight": "8 bits: één voor elke kant",
            "one": "1 bit: één worp",
            "right": "3 bits: drie ja-of-neevragen",
        },
        correct="right",
        why={
            "eight": "Elke vraag halveert wat er over is: 8 kanten vragen $\\log_2 8 = 3$ vragen, niet 8. Zie [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl).",
            "one": "Eén bit is één eerlijke ja-of-neevraag, genoeg voor een munt maar niet voor 8 kanten. Zie [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl).",
            "right": "Met 8 even waarschijnlijke kanten is de entropie $\\log_2 8 = 3$ bits.",
        },
    )
    quick_die
    return (quick_die,)


@app.cell(hide_code=True)
def quick_check_onehot():
    quick_onehot = L.choice(
        "quick-onehot",
        question="In de cross-entropy-loss $-\\sum_k y_k \\log(\\hat{y}_k)$ is het label $y$ een one-hot-vector. Wat blijft er van de som over?",
        options={
            "all": "De termen van alle klassen, elk gewogen door het model",
            "right": "Alleen de term van de goede klasse",
            "wrong": "Alleen de termen van de foute klassen",
        },
        correct="right",
        why={
            "all": "De one-hot-vector is 0 voor elke klasse behalve de goede, dus de andere termen vallen weg. Zie [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "wrong": "De foute klassen hebben $y_k = 0$, dus hun termen zijn 0. Zie [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl).",
            "right": "Alleen de goede klasse heeft $y_k = 1$: de loss is $-\\log(\\hat{y}_c)$.",
        },
    )
    quick_onehot
    return (quick_onehot,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model kent de echte verdeling van zijn data nooit; het heeft er een eigen schatting van. Wat kost een verkeerde schatting? De cross-entropy-loss waarmee je traint geeft het antwoord, en dat antwoord valt in twee delen uiteen: de verrassing die de data hoe dan ook heeft, en het extra dat komt van de verkeerde schatting van het model. Alleen het tweede deel kan training weghalen.

    **Een voorbeeld uit het dagelijks leven.** Je pakt in voor een week zon, en het regent. Wat gedoe had je hoe dan ook gehad, hoe je ook had ingepakt; de rest komt doordat je voor het verkeerde weer inpakte.
    """)
    return


@app.cell(hide_code=True)
def predict_coin():
    guess_coin = L.guess(
        "fair-coin",
        question="Een munt is eerlijk, maar jij verwacht dat hij in 90% van de worpen op kop valt. Een eerlijke munt heeft 1 bit entropie. Hoeveel bits verrassing per worp denk je dat je gemiddeld krijgt?",
        options={
            "right": "Ongeveer 1,74 bits: de worpen met munt, de helft, verrassen je flink",
            "heads": "Ongeveer 1,15 bits: de worpen met kop, de helft, verrassen je amper",
            "tails": "Ongeveer 3,32 bits: munt, waar je maar 0,1 aan gaf, kost zoveel",
        },
        reveal=f"**Ongeveer 1,74 bits: {L.number(C.coin(0.5, 0.9), 3)}.** De helft van de worpen is kop, en dat verwachtte je met 0,9: {L.number(C.coin(1.0, 0.9), 2)} bits per keer. De andere helft is munt, waar je maar 0,1 aan gaf: {L.number(C.coin(0.0, 0.9), 2)} bits per keer. Gemiddeld {L.number(C.coin(0.5, 0.9), 2)} bits, {L.number(C.coin(0.5, 0.9) - 1, 2)} meer dan de 1 bit die een eerlijke munt toch al heeft.",
    )
    guess_coin
    return (guess_coin,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Stel dat de waarheid de verdeling $p$ is, en dat jij $q$ verwacht. Elke uitkomst komt zo vaak als $p$ zegt, maar verrast je zoveel als $q$ zegt. De **cross-entropy** is de verwachte verrassing:
        """),
        L.formula(r"[[loss:H(p, q)]] = -\sum_i p_i \log_2 q_i"),
        mo.md(r"""
        Het is de algemene vorm van de cross-entropy uit [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl): daar was $p$ het one-hot-label, dus telde alleen de goede klasse, en was de log de natuurlijke. Als $q$ gelijk is aan $p$, is de cross-entropy de entropie $H(p)$ uit [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl). Voor elke andere $q$ is ze groter: de verkeerde verdeling verwachten kost gemiddeld altijd iets.

        De **KL-divergentie**, naar Kullback en Leibler, is dat extra:
        """),
        L.formula(r"[[loss:D_{\mathrm{KL}}(p \,\|\, q)]] = H(p, q) - H(p) = \sum_i p_i \log_2 \frac{p_i}{q_i}"),
        mo.md(r"""
        Ze is 0 als $q$ gelijk is aan $p$, en nooit negatief. Ze is geen afstand: $D_{\mathrm{KL}}(p \,\|\, q)$ en $D_{\mathrm{KL}}(q \,\|\, p)$ verschillen meestal. Train je een model met de cross-entropy-loss, dan is $p$ de data en verandert $H(p)$ niet, dus een lagere loss is een lagere KL-divergentie: de $q$ van het model komt dichter bij de data. En als $q$ kans 0 geeft aan een uitkomst die $p$ heeft, zijn de cross-entropy en de KL-divergentie oneindig.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | $H(p, q)$ | H van p en q | de cross-entropy: de verwachte verrassing als de waarheid $p$ is en jij $q$ verwacht | `-np.sum(p * np.log2(q))` |
        | **KL-divergentie**, $D_{\mathrm{KL}}(p \,\|\, q)$ | KL-divergentie van p ten opzichte van q | de extra verrassing boven de entropie van $p$ als je $q$ verwacht; alleen 0 als $q$ gelijk is aan $p$ | `np.sum(p * np.log2(p / q))` |
        """),
        L.watch_out("De volgorde doet ertoe. In $H(p, q)$ en $D_{\\mathrm{KL}}(p \\,\\|\\, q)$ is $p$ de waarheid die de uitkomsten weegt, en $q$ wat je verwacht. Wissel ze om en je krijgt een ander getal, en in een loss een andere training."),
        mo.md(r"""
        De figuur: een eerlijke munt, dus $p$ is half kop, half munt, en de $q$ die je verwacht. De cross-entropy is het laagst, 1 bit, als je verwacht wat waar is; bij $q$ = 0,9 is het stuk boven 1 bit de KL-divergentie.
        """),
        mo.Html(C.coin_picture({"title": "Cross-entropy van een eerlijke munt", "x": "verwachte kans op kop, q", "y": "bits",
                                "floor": "entropie van de eerlijke munt: 1 bit", "gap": "KL = {kl} bits", "number": L.number,
                                "alt": "Een U-vormige kromme van de cross-entropy, het laagst bij q = 0,5 met 1 bit, en steil oplopend naar q = 0 en q = 1. Een stippellijn bij 1 bit. Bij q = 0,9 een pijl van 1 bit omhoog naar de kromme, bij ongeveer 1,74 bits: de KL-divergentie van 0,74 bits."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te meten wat een verkeerde verwachting kost:

    1. Schrijf de echte verdeling $p$ en de verwachte $q$ op, over dezelfde uitkomsten.
    2. De cross-entropy: de verrassing onder $q$, $-\log_2 q_i$, gewogen met $p_i$, opgeteld; laat de uitkomsten met $p_i = 0$ weg.
    3. De entropie van $p$, zoals in [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl).
    4. De KL-divergentie: de cross-entropy min de entropie, de extra bits omdat je $q$ verwachtte.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De soorten van de 344 Palmer-pinguïns als $p$, en als $q$ de schatting dat alle drie de soorten even vaak voorkomen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_kl():
    import numpy as np

    # 1. de waarheid, uit de aantallen, en een verwachting van drie gelijke delen
    _p = np.array(C.species_counts()) / 344
    _q = np.array([1, 1, 1]) / 3
    # 2. de cross-entropy, in bits, over de uitkomsten die p heeft (hier alle drie)
    _keep = _p > 0
    _cross = -np.sum(_p[_keep] * np.log2(_q[_keep]))
    # 3. de entropie van p: de 0 houdt de plek vrij; vervang hem, zoals stap 2 maar met p in de log
    _entropy = 0
    # 4. het extra: de KL-divergentie
    print(_cross, _entropy, _cross - _entropy)
    return


@app.cell(hide_code=True)
def text_after_example():
    _p = [n / 344 for n in C.species_counts()]
    _biscoe = C.species_counts("Biscoe")
    _q = [n / sum(_biscoe) for n in _biscoe]
    mo.md(f"""
    Met de 0 in stap 3 is het laatste getal de cross-entropy zelf, en dat is te veel. Met stap 3 af: drie gelijke delen verwachten kost {L.number(C.bits(_p, [1 / 3] * 3), 3)} bits per pinguïn, tegen {L.number(C.bits(_p, _p), 3)} voor de echte aandelen, de twee getallen die het uitgewerkte voorbeeld van [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl) printte: een KL-divergentie van {L.number(C.bits(_p, [1 / 3] * 3) - C.bits(_p, _p), 3)} bits, klein, omdat de soorten niet veel verschillen in hoe vaak ze voorkomen. Neem als $q$ in plaats daarvan de aandelen op het eiland Biscoe, waar de data geen Chinstrap heeft: $q$ geeft Chinstrap kans 0, de 68 Chinstraps zijn oneindig verrassend, en de cross-entropy en de KL-divergentie zijn oneindig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_equal():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wanneer is er geen extra?
    """)
    return


@app.cell(hide_code=True)
def choice_equal_cell():
    choice_equal = L.choice(
        "when-equal",
        question="Wanneer is de cross-entropy $H(p, q)$ precies de entropie $H(p)$?",
        options={
            "uniform": "Als $p$ elke uitkomst dezelfde kans geeft",
            "sure": "Als $q$ zeker is van één uitkomst",
            "right": "Als $q$ gelijk is aan $p$",
        },
        correct="right",
        why={
            "uniform": "Een uniforme $p$ kost nog steeds extra als $q$ niet ook uniform is: het gaat erom of $q$ bij $p$ past.",
            "sure": "Een zekere $q$ geeft de andere uitkomsten kans 0; heeft $p$ er daar een van, dan is de cross-entropy oneindig.",
            "right": "De KL-divergentie is alleen 0 als $q$ gelijk is aan $p$; elke andere $q$ kost gemiddeld extra bits.",
        },
    )
    choice_equal
    return (choice_equal,)


@app.cell(hide_code=True)
def text_exercise_cross():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Cross-entropy als functie

        Schrijf `cross_entropy(p, q)`: de cross-entropy in bits als de waarheid `p` is en jij `q` verwacht, met stap 2 van het recept. De check roept `cross_entropy(np.array([0.5, 0.25, 0.125, 0.125]), np.array([0.25, 0.125, 0.5, 0.125]))` aan, en dat moet `2.25` geven; hij probeert ook andere paren, ook met kansen van 0.
        """),
        L.task("Laat `cross_entropy` de verwachte verrassing onder `q` geven als de waarheid `p` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cross():
    box_cross = L.code_box("cross-entropy", lines=7)
    box_cross
    return (box_cross,)


@app.cell(hide_code=True)
def check_cross(box_cross):
    result_cross = L.check("cross-entropy", box_cross.value)
    result_cross
    return (result_cross,)


@app.cell(hide_code=True)
def hints_cross():
    L.hints(
        "cross-entropy",
        [
            "Stap 2 van het recept: welke verdeling staat in de log, en welke weegt?",
            "Houd alleen de uitkomsten waar `p` boven 0 is, met een boolean mask, en gebruik dat op beide arrays.",
        ],
        why="De verrassingen onder `q` zijn 2, 3, 1 en 3 bits; gewogen met `p`: $0{,}5 \\cdot 2 + 0{,}25 \\cdot 3 + 0{,}125 \\cdot 1 + 0{,}125 \\cdot 3 = 2{,}25$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kl():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De KL-divergentie

        Schrijf `kl(p, q)`: de KL-divergentie in bits, de extra verrassing boven de entropie van `p`, met stap 2 tot en met 4 van het recept. De check roept `kl(np.array([0.5, 0.25, 0.125, 0.125]), np.array([0.25, 0.125, 0.5, 0.125]))` aan, en dat moet `0.5` geven; hij probeert ook andere paren, in beide richtingen.
        """),
        L.task("Laat `kl` de extra verrassing geven als je `q` verwacht en de waarheid `p` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kl():
    box_kl = L.code_box("kl", lines=7)
    box_kl
    return (box_kl,)


@app.cell(hide_code=True)
def check_kl(box_kl):
    result_kl = L.check("kl", box_kl.value)
    result_kl
    return (result_kl,)


@app.cell(hide_code=True)
def hints_kl():
    L.hints(
        "kl",
        [
            "Stap 3 en 4 van het recept: de cross-entropy, min de entropie van `p`. Of in één som, met $\\log_2 \\frac{p_i}{q_i}$.",
            "Laat de uitkomsten weg waar `p` 0 is, zoals in oefening 2; houd `p` als gewicht.",
        ],
        why="De cross-entropy is 2,25 bits en de entropie van `p` is 1,75 bits, dus de KL-divergentie is 0,5 bits.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De entropie uit [Entropie: gemiddelde verrassing](/course/open/info-entropy?lang=nl) is wat de waarheid hoe dan ook kost; de cross-entropy uit [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl) is wat je betaalt met je eigen verwachting, en de KL-divergentie is het verschil. Trainen met de cross-entropy-loss maakt dat verschil kleiner. In de volgende les, *Maximum likelihood*, zie je dat dit hetzelfde is als de data zo waarschijnlijk mogelijk maken, en waarom zoveel modellen op deze loss trainen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_equal, result_cross, result_kl):
    L.wrap_up(choice_equal, result_cross, result_kl)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
