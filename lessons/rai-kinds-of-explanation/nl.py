# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-kinds-of-explanation. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-kinds-of-explanation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) en [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logit():
    quick_logit = L.choice(
        "quick-logit",
        question="Een logistische regressie heeft de weights (2; −1) en de bias 0. Hij krijgt het voorbeeld (1; 3). Welke kans geeft hij?",
        options={
            "right": "Onder 0,5",
            "above": "Boven 0,5",
            "half": "Precies 0,5",
        },
        correct="right",
        why={
            "right": "De logit is 2 · 1 + (−1) · 3 + 0 = −1. Onder 0, dus de sigmoid geeft een kans onder 0,5: het model zegt nee.",
            "above": "Reken eerst de logit uit: 2 · 1 + (−1) · 3 + 0 = −1. Een logit onder 0 geeft een kans onder 0,5. Zie [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "half": "Precies 0,5 vraagt om een logit van precies 0: het voorbeeld zou op de decision boundary liggen. Hier is de logit 2 · 1 + (−1) · 3 + 0 = −1. Zie [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
        },
    )
    quick_logit
    return (quick_logit,)


@app.cell(hide_code=True)
def quick_check_leaf():
    quick_leaf = L.choice(
        "quick-leaf",
        question="Een beslisboom stuurt een pinguïn langs zijn vragen naar een blad. Wat zegt het blad?",
        options={
            "near": "De klasse van de meest gelijkende pinguïn",
            "right": "De meest voorkomende klasse in dat blad",
            "average": "Het gemiddelde over alle pinguïns",
        },
        correct="right",
        why={
            "near": "Een boom zoekt geen gelijkende pinguïn. Elk blad zegt de klasse die het vaakst voorkomt bij de trainingsvoorbeelden die erin terechtkwamen. Zie [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
            "right": "Elk blad zegt de klasse die het vaakst voorkomt bij de trainingsvoorbeelden die erin terechtkwamen, en elke pinguïn die het blad bereikt, krijgt die klasse.",
            "average": "Een boom neemt geen gemiddelde over alle pinguïns. Elk blad zegt de klasse die het vaakst voorkomt bij zijn eigen trainingsvoorbeelden. Zie [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
        },
    )
    quick_leaf
    return (quick_leaf,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2016 kondigde DARPA, het onderzoeksagentschap van het Amerikaanse ministerie van Defensie, een programma over explainable AI aan. De slides voor de Proposers Day in augustus 2016 noemden machine learning-modellen ondoorzichtig, niet intuïtief en moeilijk te begrijpen voor mensen, en lieten een gebruiker vragen: "Waarom deed je dat?" en "Waarom niet iets anders?". De vragen zijn ouder dan het programma. In 2004 beschreven Michael van Lent en collega's een systeem met een onderdeel "Explainable AI (XAI)", voortbouwend op werk over uitleg in expertsystemen zoals Mycin, beschreven in 1976. Maar een "waarom" kan verschillende dingen vragen. Waarom kreeg deze ene aanvrager nee? Hoe beslist het model in het algemeen? En komt het antwoord uit het model zelf, of van iemand die het achteraf heeft bestudeerd? Wat een uitleg je kan vertellen, hangt van die verschillen af.

    **Een voorbeeld uit het dagelijks leven.** Je vraagt een collega waarom die vandaag over de ringweg reed. "Vanwege de werkzaamheden op de brug" legt deze ene rit uit. "Ik neem altijd de weg die mijn telefoon het snelst noemt" legt uit hoe je collega in het algemeen kiest. En beide antwoorden zijn wat je collega je achteraf vertelt, en dat hoeft niet te zijn hoe de keuze echt ging.
    """)
    return


@app.cell(hide_code=True)
def predict_curve():
    guess_curve = L.guess(
        "curve",
        question="De slides van DARPA uit 2016 tekenden een curve: hoe beter een soort model voorspelt, hoe moeilijker het uit te leggen is. Op hoeveel gemeten modellen is die curve gebaseerd?",
        options={
            "dozens": "Tientallen, uit benchmarks",
            "few": "Een handvol, uit één studie",
            "none": "Geen",
        },
        reveal="**Geen.** De slides noemen de grafiek \"(notional)\", en Cynthia Rudin (2019), die er een eigen versie van tekende, schreef dat hij op geen enkele data gebaseerd is. Of een model dat mensen kunnen lezen slechter moet voorspellen, is iets om per taak te meten; een latere les in deze module komt erop terug.",
    )
    guess_curve
    return (guess_curve,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén voorspelling, of het hele model.** Finale Doshi-Velez en Been Kim (2017) zetten het verschil zo neer: globale interpretability is weten welke patronen er in het algemeen zijn, lokale interpretability is de redenen kennen voor één specifiek besluit, zoals waarom een bepaalde leningaanvraag werd afgewezen. Het eerste kan belangrijk zijn als je wetenschappelijk inzicht wilt of bias wilt opsporen, het tweede als je één besluit moet rechtvaardigen. In het kredietmodel van deze les zijn de weights een global explanation: elke gemiste betaling verlaagt de logit met 1,2, voor elke aanvrager. Wat elke weight bijdraagt voor één aanvrager, is een local explanation.

        **Uit het model, of achteraf gemaakt.** Zachary Lipton (2016) deelde uitleg op in twee families: transparantie (hoe werkt het model?) en post-hoc explanations (wat kan het model me nog meer vertellen?). Het boek *Interpretable Machine Learning* van Christoph Molnar (derde editie, 2025) noemt het eerste interpretability by design: je traint een model dat mensen kunnen lezen, bijvoorbeeld een logistische regressie in plaats van een random forest. Een post-hoc explanation wordt na het trainen gemaakt, met een aparte methode die het afgeronde model bestudeert. De aantrekkingskracht volgens Lipton: je kunt ondoorzichtige modellen achteraf uitleggen zonder in te leveren op hoe goed ze voorspellen. Het risico staat in het blok *Let op* hieronder.

        **Elk model, of één soort.** Een model-agnostic methode heeft alleen de inputs en outputs van een model nodig. Molnar: je kijkt niet naar wat er in het model zit, alleen naar hoe de output verandert als je de features verandert. Een model-specifieke methode gebruikt de binnenkant van één soort model, zoals de weights van een logistische regressie of de vragen van een boom. Alleen een model-agnostic methode werkt voor een model waaraan je alleen vragen kunt stellen, een black box in de strikte zin van deze course. Het boek van Molnar gebruikt het woord ruimer, voor modellen die je niet kunt begrijpen door naar hun parameters te kijken, zoals een neural network; voor zulke modellen bestaan ook model-specifieke methoden die erin kijken. Cynthia Rudin (2019) noemt nog een reden waarom een model een black box kan zijn: het is eigendom van een bedrijf, dat het niet laat zien.

        Deze drie vragen zijn de manier waarop deze course uitleg indeelt. Het boek van Molnar deelt het iets anders in: het splitst alleen de model-agnostic methoden in lokaal en globaal, en waarschuwt dat de grens tussen by design en post hoc in werkelijkheid vaag is.

        **Drie soorten ondoorzichtigheid.** Jenna Burrell (2016) noemde drie redenen waarom een model ondoorzichtig blijft: geheimhouding door een bedrijf of de overheid, technisch analfabetisme, en een ondoorzichtigheid die voortkomt uit de eigenschappen van machine learning-algoritmen en de schaal waarop ze pas bruikbaar worden. De tweede gaat over de lezer: code schrijven en lezen is een vak apart. De derde geldt zelfs voor experts: optimalisatie in heel veel dimensies past niet bij de manier waarop mensen redeneren. Herkennen welke soort er speelt, schrijft Burrell, is een sleutel om te bepalen welke technische en niet-technische oplossingen schade kunnen voorkomen. Een model openbaar maken neemt de derde soort niet weg; een model dat mensen kunnen lezen, of een methode die het uitlegt, misschien wel.

        **De woorden.** Interpretability heeft op dit moment geen formele technische betekenis, schreef Lipton, en de bronnen zijn het nog steeds niet eens. Tim Miller (2019) gebruikt interpretability en explainability als hetzelfde; Alejandro Barredo Arrieta en collega's (2019) noemen interpretability een passieve eigenschap van een model en explainability een actieve; Rudin ziet een kloof tussen de twee. Deze course houdt ze uit elkaar: interpretability gaat over begrijpen hoe een model vanbinnen werkt, explainability over het uitleggen van zijn outputs. Explainable AI, kort XAI, is het vakgebied met zijn methoden.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **explainable AI (XAI)** | het vakgebied en de methoden die de outputs van AI-modellen begrijpelijk maken voor mensen |
        | **interpretability** | hoe goed mensen kunnen begrijpen hoe een model vanbinnen werkt, bijvoorbeeld door de weights of de vragen te lezen |
        | **explainability** | hoe goed de redenen achter de outputs van een model aan mensen uit te leggen zijn |
        | **local explanation** | een uitleg van één voorspelling, zoals waarom deze aanvrager nee kreeg |
        | **global explanation** | een uitleg van hoe een model zich in het algemeen gedraagt, voor elke input |
        | **post-hoc explanation** | een uitleg die na het trainen gemaakt wordt, met een aparte methode die het afgeronde model bestudeert |
        | **model-agnostic** | gezegd van een methode die alleen de inputs en outputs van een model nodig heeft, zodat ze voor elk model werkt; een model-specifieke methode gebruikt de binnenkant van één soort |
        | **black box** | een model dat je alleen van buitenaf kunt bekijken: je ziet wat erin gaat en wat eruit komt, niet hoe het van het een naar het ander komt |
        """),
        L.watch_out("Een uitleg die achteraf gemaakt is, beschrijft het model vaak maar bij benadering, en hij kan goed klinken en toch fout zijn. Lipton waarschuwde dat een methode die is afgesteld om mensen tevreden te stellen, misleidende maar geloofwaardige uitleg kan geven. En de curve op de slides van DARPA heet \"(notional)\": hij meet niets."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een uitleg in te delen, stel je vier vragen:

    1. Waar komt hij vandaan: uit het model zelf gelezen, of achteraf gemaakt met een aparte methode?
    2. Waarover gaat hij: één voorspelling, of het hele model?
    3. Wat heeft hij nodig: de binnenkant van één soort model, of alleen inputs en outputs?
    4. Als niemand het model kan uitleggen: wordt het geheimgehouden, mist de lezer de kennis, of werkt het op een manier die geen mens kan volgen?
    """)
    return


@app.cell(hide_code=True)
def picture_two():
    mo.vstack([
        mo.md("**Eén model, twee soorten uitleg**"),
        mo.Html(C.two_explanations("nl")),
        mo.md(r"""
        Links de weights van het kredietmodel, voor elke aanvrager hetzelfde: een global explanation. Rechts wat ze bijdragen aan de logit van aanvrager B, die 2000 euro per maand verdient, geen betaling miste en al 6 jaar in dezelfde baan werkt: een local explanation. Allebei lees je uit het model zelf, dus allebei zijn ze model-specifiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het kredietmodel van deze les is een logistische regressie die de course heeft verzonnen, met drie features: `income`, het inkomen in duizenden euro's per maand, `missed_payments`, de gemiste betalingen in het afgelopen jaar, en `years_at_job`, het aantal jaren in de huidige baan. Voer de cel uit: hij rekent de logit en de kans op ja uit voor aanvrager A, zoals in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).
    """)
    return


@app.cell
def example_logit():
    import math

    # een kredietmodel dat de course heeft verzonnen: een logistische regressie op drie features
    _weights = {"income": 0.8, "missed_payments": -1.2, "years_at_job": 0.3}
    _bias = -3.0

    # aanvrager A: inkomen in duizenden euro's per maand
    _person = {"income": 3.5, "missed_payments": 2, "years_at_job": 1}

    _logit = _bias
    for _feature in _weights:
        _logit = _logit + _weights[_feature] * _person[_feature]
    print("logit", round(_logit, 2))
    print("kans op ja", round(1 / (1 + math.exp(-_logit)), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De logit is −2,3 en de kans op ja 0,091: het model zegt nee. De weights vertellen hoe het model voor iedereen werkt, een global explanation die je uit het model zelf leest. Waarom aanvrager A nee krijgt, is een andere vraag, over één aanvrager. Daarvoor houd je apart wat elke feature bijdraagt: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_contributions():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat elke feature bijdraagt

        `weights` en `person` zijn het kredietmodel en aanvrager A uit het uitgewerkte voorbeeld. Reken `contributions` uit: een dict met voor elke feature de weight keer de waarde van `person`. De loop in het uitgewerkte voorbeeld telde deze getallen een voor een op bij de logit; nu houd je ze apart. De check probeert ook aanvrager B.
        """),
        L.task("Reken `contributions` uit: voor elke feature de weight keer de waarde van `person`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contributions():
    box_contributions = L.code_box("contributions", lines=10)
    box_contributions
    return (box_contributions,)


@app.cell(hide_code=True)
def check_contributions(box_contributions):
    result_contributions = L.check("contributions", box_contributions.value)
    result_contributions
    return (result_contributions,)


@app.cell(hide_code=True)
def hints_contributions():
    L.hints(
        "contributions",
        [
            "Eén getal voor elke feature: wat die bijdraagt aan de logit van deze aanvrager.",
            "Begin met een lege dict, loop over de features met `for feature in weights:`, en zet `weights[feature] * person[feature]` onder elke feature.",
            "`contributions[feature] = ___`",
        ],
        why="`contributions = {}`, dan `for feature in weights:` met `contributions[feature] = weights[feature] * person[feature]` in de loop. Voor aanvrager A geeft dat income `2.8`, missed_payments `-2.4` en years_at_job `0.3`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 2. Deel de uitleg in

    Zet elke uitleg in zijn groep: waar komt hij vandaan, en waarover gaat hij?
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Twee vragen bij elke uitleg: lees je hem uit het model zelf, of wordt hij achteraf gemaakt door het model vragen te stellen? En gaat hij over één aanvrager, of over allemaal?",
            "Drie worden achteraf gemaakt, door het afgeronde model vragen te stellen; twee van die drie beschrijven alle aanvragers.",
            "Uit het model, het hele model: de weights en \\_\\_\\_. Achteraf gemaakt, één voorspelling: \\_\\_\\_.",
        ],
        why="**Uit het model, één voorspelling:** één pad, jouw bijdragen. **Uit het model, het hele model:** de weights, de hele boom. **Achteraf gemaakt, één voorspelling:** een model in de buurt. **Achteraf gemaakt, het hele model:** een kolom schudden, een gemiddelde curve.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_agnostic():
    mo.md(r"""
    ### Oefening 3. Een model waaraan je alleen vragen kunt stellen
    """)
    return


@app.cell(hide_code=True)
def choice_agnostic_cell():
    choice_agnostic = L.choice(
        "agnostic",
        question="Het kredietmodel van een bank is een black box: je geeft het de gegevens van een aanvrager, en het zegt ja of nee. Hoe kun je het toch uitleggen?",
        options={
            "weights": "De weights lezen",
            "path": "Een aanvrager langs de vragen van de boom volgen",
            "right": "Het vragen naar veranderde aanvragers",
        },
        correct="right",
        why={
            "weights": "Een black box laat geen weights zien. Wat wel werkt, is een model-agnostic methode: die heeft alleen inputs en outputs nodig, dus hij vraagt het model naar veranderde aanvragers en kijkt naar de antwoorden. Zie *Elk model, of één soort* in *Het idee*.",
            "path": "Een pad volgen kan alleen met de boom zelf, de binnenkant van één soort model: een model-specifieke methode. Een black box geeft alleen antwoord. Zie *Elk model, of één soort* in *Het idee*.",
            "right": "Een model-agnostic methode heeft alleen inputs en outputs nodig, dus ze werkt voor elk model, ook een model waaraan je alleen vragen kunt stellen. Een kolom schudden, de gemiddelde curve en het model in de buurt uit oefening 2 werken allemaal zo.",
        },
    )
    choice_agnostic
    return (choice_agnostic,)


@app.cell(hide_code=True)
def text_exercise_opacity():
    mo.md(r"""
    ### Oefening 4. Welke ondoorzichtigheid?
    """)
    return


@app.cell(hide_code=True)
def choice_opacity_cell():
    choice_opacity = L.choice(
        "opacity",
        question="Een team heeft de code en alle 20 miljoen weights van een kredietmodel, en de leden lezen goed code. Toch kan niemand in begrijpelijke termen zeggen waarom het één aanvrager afwees. Welke soort ondoorzichtigheid van Burrell is dit?",
        options={
            "secrecy": "Geheimhouding: het bedrijf verbergt het model",
            "skill": "Te weinig kennis om code te lezen",
            "right": "Hoe het werkt, op zijn schaal",
        },
        correct="right",
        why={
            "secrecy": "Het team heeft de code en de weights: er wordt niets voor het team verborgen. Wat overblijft, is de derde soort van Burrell: optimalisatie over miljoenen weights past niet bij de manier waarop mensen redeneren. Zie *Drie soorten ondoorzichtigheid* in *Het idee*.",
            "skill": "Het team leest goed code. De derde soort van Burrell geldt zelfs voor mensen met een gespecialiseerde opleiding, zelfs voor informatici. Zie *Drie soorten ondoorzichtigheid* in *Het idee*.",
            "right": "De derde soort van Burrell: optimalisatie in heel veel dimensies past niet bij de manier waarop mensen redeneren, ook niet bij informatici. Het model openbaar maken of mensen leren code te lezen, helpt hier niet; een model dat mensen kunnen lezen, of een methode die het uitlegt, misschien wel.",
        },
    )
    choice_opacity
    return (choice_opacity,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) was de logit de weights keer de features, plus de bias; in deze les splitste je die som in zijn delen, en die delen zijn een local explanation. In [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl) volgde je een pinguïn langs een boom: ook een local explanation, uit het model zelf gelezen. De volgende les, [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), maakt van bijdragen de redenen die een kredietverstrekker bij een nee geeft. De methoden die achteraf gemaakt worden, zoals een kolom schudden en het model in de buurt, komen later in deze module.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agnostic, choice_opacity, result_contributions, result_kinds):
    L.wrap_up(result_contributions, result_kinds, choice_agnostic, choice_opacity)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
