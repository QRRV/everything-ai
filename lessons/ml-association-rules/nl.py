# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-association-rules. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-association-rules", "nl")
    C = L.checks   # de 19.773 bestellingen van de winkel, Apriori erop voor elke min support, de regels, en het plaatje


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl) en één over [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_conditional():
    quick_conditional = L.choice(
        "quick-conditional",
        question="Van 200 bestellingen bevatten er 40 thee, en 30 van die 40 bevatten ook taart. Wat is P(taart | thee)?",
        options={
            "right": "0,75",
            "both": "0,15",
            "tea": "0,2",
        },
        correct="right",
        why={
            "both": "0,15 is 30 / 200: het deel van alle bestellingen met thee én taart. P(taart | thee) kijkt alleen naar de 40 bestellingen met thee. Zie *Het idee* in [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "tea": "0,2 is 40 / 200: P(thee). P(taart | thee) is het deel van de bestellingen met thee dat ook taart bevat. Zie *Het idee* in [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "right": "30 / 40. Vandaag krijgt dat getal een eigen naam: de confidence van de regel thee → taart.",
        },
    )
    quick_conditional
    return (quick_conditional,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question="`df` heeft een rij voor elk product in een bestelling, met de kolommen `invoice`, de bestelling, en `product`. Wat geeft `df.groupby(\"invoice\")[\"product\"].count()`?",
        options={
            "right": "Voor elke bestelling hoeveel producten erin zitten",
            "product": "Voor elk product in hoeveel bestellingen het zit",
            "total": "Hoeveel bestellingen er zijn",
        },
        correct="right",
        why={
            "product": "`groupby(\"invoice\")` maakt een groep per bestelling, niet per product. Voor elk product groepeer je op `\"product\"`. Zie *Het idee* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "total": "`groupby` geeft één getal per groep, dus een getal voor elke bestelling. Hoeveel bestellingen er zijn, is het aantal groepen. Zie *Het idee* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "right": "Eén getal per bestelling. Vandaag maakt `groupby` per bestelling één set producten.",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een Britse webwinkel die cadeaus verkoopt, veel ervan aan andere winkels, had van december 2010 tot december 2011 19.773 bestellingen, met 3.765 verschillende producten. Welke producten belanden in dezelfde bestelling? Een winkel die dat weet, kan ze naast elkaar zetten, het tweede voorstellen als het eerste in het winkelmandje gaat, of ze als set verkopen. Maar 3.765 producten geven 7.085.730 paren, en 3.658.456 daarvan zijn minstens één keer samen gekocht: veel te veel om één voor één te bekijken.

    **Een voorbeeld uit het dagelijks leven.** Een supermarkt zet de salsa naast de tortillachips. Niet omdat ze op elkaar lijken, maar omdat wie het ene pakt, vaak ook het andere pakt.
    """)
    return


@app.cell(hide_code=True)
def predict_together():
    guess_together = L.guess(
        "together-guess",
        question="De GREEN REGENCY TEACUP AND SAUCER, een theekop met schotel in een groen patroon, zit in 1.013 van de 19.773 bestellingen. Dezelfde kop met rozen, de ROSES REGENCY TEACUP AND SAUCER, zit in 1.065: 5,4% van alle bestellingen. Hoeveel van de 1.013 bestellingen met de groene kop bevatten ook de kop met rozen?",
        options={
            "chance": "Ongeveer 5%, net als in alle bestellingen",
            "some": "Ongeveer 25%",
            "right": "Ongeveer 75%",
        },
        reveal="**Ongeveer 75%.** 767 van de 1.013: 75,7%, tegen 5,4% van alle bestellingen. De kop met rozen zit 14 keer zo vaak in een bestelling met de groene kop als in een willekeurige bestelling.",
    )
    guess_together
    return (guess_together,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Van bestellingen naar regels.** Elke bestelling is een set producten. Drie getallen zeggen hoe producten samengaan. Hieronder is GREEN de groene theekop en ROSES de theekop met rozen.

        - De **support** van een set producten is het deel van alle bestellingen dat elk product van de set bevat. GREEN zit in 1.013 van de 19.773 bestellingen: een support van 0,051. GREEN en ROSES samen zitten in 767: 0,039.
        - Een **association rule** A → B zegt dat bestellingen met A vaak ook B bevatten. A en B zijn elk één product of een set ervan.
        - De **confidence** van A → B is de support van A en B samen gedeeld door de support van A: van de bestellingen met A, het deel dat ook B bevat. Dat is de voorwaardelijke kans P(B | A). GREEN → ROSES: 767 / 1.013 = 0,757.
        - De **lift** van A → B is de confidence gedeeld door de support van B: hoeveel keer zo vaak B in een bestelling met A zit als in een willekeurige bestelling. ROSES zit in 5,4% van alle bestellingen en in 75,7% van de bestellingen met GREEN, dus GREEN → ROSES heeft een lift van 14,1. Een lift van 1 betekent dat A geen verschil maakt voor B; onder 1 komt B juist minder vaak voor bij A.

        Een **frequent itemset** is een set producten met een support van minstens een minimum dat jij kiest, de min support. Alleen voor frequente sets loont het om regels te maken: een set in twee bestellingen zegt weinig.

        **Apriori.** Elke bestelling met GREEN en ROSES bevat ook GREEN, dus de support van een set is nooit groter dan die van een deel ervan. Een set kan alleen frequent zijn als al zijn delen het zijn. **Apriori** gebruikt dat: het telt de losse producten en houdt de frequente, maakt alleen daarvan paren en houdt de frequente paren, maakt alleen sets van drie als hun drie paren allemaal frequent zijn, en zo verder, tot er geen set meer frequent is.

        ### Nieuwe woorden en tekens

        | Woord of teken | Zeg je als | Wat het betekent | In code |
        |---|---|---|---|
        | **support** | | het deel van alle bestellingen dat elk product van een set bevat | `count / len(orders)` |
        | **association rule** | | een regel die zegt dat bestellingen met A vaak ook B bevatten | |
        | A → B | als A, dan B | de regel met A links en B rechts | |
        | **confidence** | | de support van A en B samen gedeeld door de support van A: P(B \| A) | `n_both / n_a` |
        | **lift** | | de confidence gedeeld door de support van B | `confidence / (n_b / n)` |
        | **frequent itemset** | | een set producten met een support van minstens de min support | |
        | **Apriori** | | de frequent itemsets vinden door grotere sets alleen van frequente kleinere te maken | |
        | `items <= order` | | bij twee sets: zit elke waarde van `items` ook in `order`? | `{"a"} <= {"a", "b"}` is `True` |
        """),
        L.watch_out("Lift zegt niets over hoe vaak. 125 producten zijn maar in één bestelling gekocht, en tien paren ervan in dezelfde bestelling: zo'n paar heeft een lift van 19.773, de hoogste lift die hier kan, door één bestelling. Daarom komen regels alleen uit frequente sets. En confidence alleen misleidt als B toch al in veel bestellingen zit: JUMBO BAG RED RETROSPOT → WHITE HANGING HEART T-LIGHT HOLDER heeft een confidence van 0,214, maar de houder zit in 11,5% van alle bestellingen, dus de lift is maar 1,87."),
        mo.md(r"""
        Het plaatje laat Apriori los op de 19.773 bestellingen. Links, voor losse producten, paren en sets van drie: hoeveel sets Apriori telt, de open staven, en hoeveel daarvan frequent zijn, de gevulde staven, met het aantal. De gestreepte lijn boven de paren is alle 7.085.730 paren. Elke lijn hoger is tien keer de lijn eronder. Rechts elke regel van de frequente paren, twee per paar: de confidence en de lift, de lift ook op die schaal. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_support_cell():
    def _frame(ms):
        said = C.said(ms, L.lang)
        alt = ("Links een staafdiagram op een logaritmische schaal met open staven voor de sets die Apriori telt en gevulde staven voor de frequente, "
               "voor losse producten, paren en sets van drie, met een gestreepte lijn bij alle 7.085.730 paren. Rechts de regels van de "
               "frequente paren als stippen, confidence van 0 tot 1 tegen lift van 1 tot 100 op een logaritmische schaal. " + said)
        return (C.picture(ms, {"left": f"Apriori, min support {C.short(ms, L.lang)}", "right": "Regels van de frequente paren", "sets": "sets",
                               "sizes": ["producten", "paren", "sets van 3"], "lang": L.lang, "alt": alt}), said)

    explore_support = L.explore("min-support", "min support", _frame, start=0.02, values=C.SUPPORTS, role="hyper",
                                task="Ga van min support 0,01 naar 0,04. Hoeveel paren telt Apriori bij 0,02, en vanaf welke min support is geen enkele set van drie meer frequent?")
    explore_support
    return


@app.cell(hide_code=True)
def text_after_support():
    mo.md(r"""
    Bij min support 0,01 zijn 830 producten frequent, en telt Apriori 344.035 paren ervan in plaats van 7.085.730; 910 zijn frequent. Bij 0,02 telt het 45.451 paren, houdt er 90, en maakt het maar 67 sets van drie, waarvan er 3 frequent zijn, zoals de groene, de rozen- en de roze regency-theekop samen. Vanaf 0,03 is geen enkele set van drie meer frequent; bij 0,04 blijft één paar over, JUMBO BAG PINK POLKADOT en JUMBO BAG RED RETROSPOT. Een lagere min support vindt meer regels, met lifts tot 76,83 bij 0,01, maar ook meer sets die door een paar bestellingen ontstaan.

    ### Recept

    Om te vinden welke producten samen gekocht worden:

    1. Maak van elke bestelling een set producten.
    2. Kies een min support, en vind de frequente sets met Apriori: losse producten, dan paren van frequente producten, dan sets van drie waarvan alle paren frequent zijn.
    3. Maak voor elk frequent paar beide regels, A → B en B → A, met hun confidence en lift.
    4. Houd de regels met een hoge confidence en een lift ruim boven 1, en bekijk zelf een paar van hun bestellingen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De regel GREEN → ROSES, met de hand geteld. De cel leest de bestellingen, maakt voor elke bestelling een set producten met `groupby("invoice")["product"].agg(set)`: behalve een naam zoals `"mean"` neemt `agg` ook een functie, en `set` maakt van de waarden van elke groep een set. Hij telt de bestellingen met GREEN, met ROSES, en met allebei. Hij print het aantal bestellingen en die drie aantallen, daarna de support van GREEN, de confidence en de lift. De opmerkingen met `#` zeggen welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_rule():
    import pandas as pd

    # 1. elke bestelling als een set producten
    _df = pd.read_csv("data/online-retail-baskets.csv.gz")
    _orders = list(_df.groupby("invoice")["product"].agg(set))

    # 3. de regel GREEN -> ROSES: tel de bestellingen met A, met B, en met allebei
    _a = "GREEN REGENCY TEACUP AND SAUCER"
    _b = "ROSES REGENCY TEACUP AND SAUCER"
    _n_a = 0
    _n_b = 0
    _n_both = 0
    for _order in _orders:
        if _a in _order:
            _n_a = _n_a + 1
        if _b in _order:
            _n_b = _n_b + 1
        if _a in _order and _b in _order:
            _n_both = _n_both + 1

    _n = len(_orders)
    _confidence = _n_both / _n_a
    _lift = _confidence / (_n_b / _n)
    print(_n, _n_a, _n_b, _n_both)
    print(round(_n_a / _n, 3), round(_confidence, 3), round(_lift, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `agg(set)` geeft één set per groep: hier de producten van elke bestelling. 19.773 bestellingen; GREEN zit in 1.013, ROSES in 1.065, en allebei in 767. De support van GREEN is 0,051, de confidence van GREEN → ROSES 0,757, en de lift 14,06: de getallen uit *Het idee*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_support():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Support

        `support(orders, items)` moet de support van de set `items` teruggeven: het deel van de bestellingen dat elk product ervan bevat. `orders` is een lijst van sets. Vervang de twee `...`: wanneer een bestelling meetelt, en wat de functie teruggeeft. De check roept `support(orders, both)` aan met zes kleine bestellingen en `both` de GREEN TEACUP en de ROSES TEACUP, en probeert er nog vier: één product, een set die in geen enkele bestelling zit, drie producten, en acht andere bestellingen.
        """),
        L.task("Laat `support` het deel van de bestellingen geven dat een set bevat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_support():
    box_support = L.code_box("support", lines=7)
    box_support
    return (box_support,)


@app.cell(hide_code=True)
def check_support(box_support):
    result_support = L.check("support", box_support.value)
    result_support
    return (result_support,)


@app.cell(hide_code=True)
def hints_support():
    L.hints(
        "support",
        [
            "Een bestelling telt mee als elk product van `items` erin zit. De support is dan een deel: van alle bestellingen, hoeveel telden er mee?",
            "In de tabel met nieuwe woorden staat een vergelijking die vraagt of de ene set in de andere ligt. Het deel is het aantal gedeeld door het aantal bestellingen.",
            "`if items ___ order:` en `return count / ___`.",
        ],
        why="Van de zes bestellingen bevatten de eerste en de vierde zowel de GREEN TEACUP als de ROSES TEACUP: 2 van de 6, een support van 0,333.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Confidence en lift

        `rule(n_a, n_b, n_both, n)` moet de confidence en de lift van A → B teruggeven, uit vier aantallen: de bestellingen met A, de bestellingen met B, de bestellingen met allebei, en alle bestellingen. Vervang de twee `...`. De check roept `rule(1013, 1065, 767, 19773)` aan, de aantallen van GREEN → ROSES, en probeert er nog vier: een regel met een lift van 2,5, een met een lift onder 1, een met een lift van precies 1, en een met een confidence van 1.
        """),
        L.task("Laat `rule` de confidence en de lift van een regel geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rule():
    box_rule = L.code_box("rule", lines=5)
    box_rule
    return (box_rule,)


@app.cell(hide_code=True)
def check_rule(box_rule):
    result_rule = L.check("rule", box_rule.value)
    result_rule
    return (result_rule,)


@app.cell(hide_code=True)
def hints_rule():
    L.hints(
        "rule",
        [
            "De confidence is het deel van de bestellingen met A dat ook B bevat. De lift is de confidence gedeeld door de support van B.",
            "Hoeveel van de `n_a` bestellingen met A bevatten ook B? En de support van B is het aantal bestellingen met B als deel van alle bestellingen.",
            "`confidence = n_both / ___` en `lift = confidence / (___ / n)`.",
        ],
        why="De confidence is 767 / 1.013 = 0,757. De support van B is 1.065 / 19.773 = 0,0539, en de confidence gedeeld daardoor is 14,06.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_apriori():
    mo.md(r"""
    ### Oefening 3. Waarom Apriori sets mag overslaan
    """)
    return


@app.cell(hide_code=True)
def choice_apriori_cell():
    choice_apriori = L.choice(
        "apriori",
        question="De groene, de rozen- en de roze regency-theekop zitten samen in 541 bestellingen: een support van 0,027. Wat weet je, zonder te tellen, over de support van GREEN en ROSES samen?",
        options={
            "right": "Die is minstens 0,027: elke bestelling met alle drie bevat ook die twee",
            "less": "Die is hoogstens 0,027: minder bestellingen bevatten een paar dan een set van drie",
            "unknown": "Niets: dat moet je tellen",
        },
        correct="right",
        why={
            "less": "Het is andersom: een paar vraagt minder van een bestelling dan een set van drie, dus bevatten minstens zoveel bestellingen het. GREEN en ROSES zitten in 767 bestellingen, de drie theekoppen in 541.",
            "unknown": "Elke bestelling met alle drie de theekoppen bevat ook GREEN en ROSES, dus zit het paar in minstens die 541 bestellingen, een support van 0,027, nog voor je telt. Tellen geeft 767.",
            "right": "Het paar zit in minstens die 541 bestellingen; tellen geeft 767, een support van 0,039. Apriori gebruikt dit andersom: een set met een deel dat niet frequent is, kan zelf niet frequent zijn, dus telt het die nooit.",
        },
    )
    choice_apriori
    return (choice_apriori,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De confidence van een regel is de voorwaardelijke kans uit [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl), en de lift vergelijkt die met de kans op B op zich. De bestellingen zijn de sets uit [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl), gemaakt met de `groupby` uit [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl). Een regel zegt wat samengaat, niet waarom, en niet wat één klant wil: de lessen over aanbevelen beginnen bij dezelfde bestellingen, met de klanten die ze plaatsten. In de volgende les, *Tijd als index: datums en resampling in pandas*, krijgt data een klok.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_apriori, result_rule, result_support):
    L.wrap_up(result_support, result_rule, choice_apriori)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
