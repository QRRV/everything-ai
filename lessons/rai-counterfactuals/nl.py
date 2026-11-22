# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-counterfactuals. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-counterfactuals", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl) en een over [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reason():
    quick_reason = L.choice(
        "quick-reason",
        question="Tegen de 1608 aanvragers net boven de grens was opleiding de belangrijkste reden voor het nee van de aanvrager van 30, met een verschil van −1,344. Wat vertelt die reden de aanvrager?",
        options={
            "amount": "Hoeveel niveaus meer een ja zouden geven",
            "right": "Wat het zwaarst tegen die persoon woog",
            "only": "Dat alleen de opleiding het nee bepaalde",
        },
        correct="right",
        why={
            "right": "Goed: een reden meet de aanvrager af aan een referentiegroep en noemt wat het zwaarst woog. Een hoeveelheid geeft hij niet: hoeveel er zou moeten veranderen voor een ja is een andere vraag, en die van deze les.",
            "amount": "Een reden is een verschil tegen een referentiegroep, hier −1,344, in eenheden van de logit. Hij zegt wat het zwaarst woog, niet hoeveel er zou moeten veranderen: dat is een vraag voor een counterfactual, die van deze les. Zie [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl).",
            "only": "Leeftijd en uren wogen ook tegen, −0,789 en −0,302 tegen dezelfde groep; opleiding woog het zwaarst. Zie [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl).",
        },
    )
    quick_reason
    return (quick_reason,)


@app.cell(hide_code=True)
def quick_check_direction():
    quick_direction = L.choice(
        "quick-direction",
        question="In de vorige lessen kreeg education_num een Shapley-waarde van +0,181 voor de persoon van 41. Betekent die waarde dat één opleidingsniveau meer de kans van die persoon zou verhogen?",
        options={
            "more": "Ja: een positieve waarde betekent dat meer beter is",
            "step": "Ja, met 0,181 voor elk extra niveau",
            "right": "Nee: ze zegt niets over een verandering",
        },
        correct="right",
        why={
            "right": "Goed. De waarde verdeelt het verschil tussen de kans van deze persoon en het gemiddelde van de achtergrond; ze zegt niet wat een verandering zou doen. Wat het antwoord zou veranderen, is de vraag van deze les.",
            "more": "Een positieve Shapley-waarde betekent niet dat de voorspelling stijgt als de feature stijgt: ze verdeelt het verschil tussen deze persoon en een achtergrond. Zie [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).",
            "step": "+0,181 is het deel van education_num in het verschil tussen de kans van deze persoon, 0,451, en het gemiddelde van de achtergrond, 0,225; het is geen toename per niveau. Zie [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).",
        },
    )
    quick_direction
    return (quick_direction,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De aanvrager van 30 uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl) kreeg een nee: de logistische regressie van de bank geeft een logit van −2,315, een kans op ja van 0,090. De redenen vertelden wat het zwaarst tegen de aanvrager woog. De volgende vraag van de aanvrager is een andere: wat zou ik moeten veranderen om een ja te krijgen? Sandra Wachter, Brent Mittelstadt en Chris Russell stelden voor om precies dat te beantwoorden, zonder het model open te maken. Hun voorbeeld: je kreeg de lening niet omdat je jaarinkomen £30.000 was; met een inkomen van £45.000 had je hem wel gekregen. Zo'n zin is makkelijk te volgen en vraagt van het model alleen zijn antwoorden. Maar voor één persoon zijn er veel van zulke zinnen, welke je krijgt hangt af van hoe je een verandering meet, en sommige vragen om veranderingen die niemand kan maken.

    **Een voorbeeld uit het dagelijks leven.** Een webwinkel zegt: "Nog €12 in je mandje en je verzending is gratis." Hoe de winkel de verzendkosten uitrekent, hoor je niet; je hoort alleen een kleine verandering die je de andere uitkomst geeft. Of je €12 meer wilt kopen, beslis je zelf, en volgende maand ligt de grens misschien ergens anders.
    """)
    return


@app.cell(hide_code=True)
def predict_closest():
    guess_closest = L.guess(
        "closest",
        question="De aanvrager verandert één kolom. Gemeten in stappen van de typische spreiding van elke kolom, zoals Wachter en anderen voorstellen: welke enkele verandering die een ja geeft, ligt het dichtst bij?",
        options={
            "age": "81 zijn in plaats van 30",
            "education": "Opleidingsniveau 16 in plaats van 9",
            "hours": "96 uur per week in plaats van 40",
        },
        reveal="**81 zijn.** In stappen van de mediane absolute afwijking van elke kolom, 10 jaar, 1 niveau en 3 uur, is 51 jaar ouder 5,1 stappen, 7 opleidingsniveaus hoger 7 stappen, en 56 uur meer 18,7 stappen. De dichtstbijzijnde counterfactual vraagt de aanvrager 51 jaar ouder te zijn: waar voor het model, niets waard voor de aanvrager. *Het idee* laat zien waar deze getallen vandaan komen.",
    )
    guess_closest
    return (guess_closest,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een counterfactual explanation.** Een **counterfactual explanation** zegt hoe de input anders had moeten zijn om het model een ander antwoord te laten geven: als dit dat was geweest, was het antwoord ja geweest. Wachter en anderen zetten dat tegenover uitleggen hoe een model van binnen werkt: een counterfactual beschrijft hoe het besluit afhing van de externe feiten die ertoe leidden. Er zijn alleen de antwoorden van het model voor nodig, dus het werkt voor elk model. Er is er ook niet maar één: er kunnen meerdere counterfactuals zijn, omdat er meerdere gewenste uitkomsten kunnen bestaan, en zij geven de voorkeur aan de dichtstbijzijnde mogelijke wereld (alles op p. 6).

        **Er een vinden.** Voor een model met een gradient, zoals hun neural network, zoeken Wachter en anderen: ze schuiven een rij met gradient descent op tot het model het gewenste antwoord geeft, terwijl de rij dicht bij de persoon blijft, en vanuit verschillende beginrijen komen ze bij verschillende counterfactuals uit. Bij een logistische regressie hoef je niet te zoeken. De bank zegt ja vanaf een logit van 0. Als alleen kolom $k$ verandert, met $\Delta_k$, beweegt de logit $z$ met $w_k \Delta_k$, dus de logit komt op 0 als

        $$\Delta_k = -\frac{z}{w_k}.$$

        Voor de aanvrager van 30 is $z = -2{,}315$, en education_num, met gewicht 0,345, moet 6,71 niveaus omhoog. Dit schrijf je in oefening 1.

        **Dichtbij in wat?** Om een verandering in jaren af te wegen tegen een verandering in niveaus en in uren, heeft elke kolom een schaal nodig. Wachter en anderen stellen de L1-afstand voor, ook Manhattan-afstand genoemd, waarin de verandering in elke kolom gedeeld wordt door de mediane absolute afwijking van die kolom (p. 17). De **mediane absolute afwijking** (MAD) van een kolom is de mediaan van hoe ver de waarden van hun mediaan af liggen, en de afstand telt de verandering van elke kolom op in stappen van zijn MAD:

        $$d(x, x') = \sum_k \frac{|x_k - x'_k|}{\mathrm{MAD}_k}.$$

        In UCI Adult zijn de MAD's 10 jaar leeftijd, 1 opleidingsniveau en 3 uur per week. Deze afstand schrijf je in oefening 2. Een andere afstand kan een ander antwoord geven, ook in het eigen experiment van Wachter en anderen: met gekwadrateerde afstanden veranderden hun counterfactuals het gemiddelde cijfer veel liever dan de score op het toelatingsexamen, en dat kwam volgens hen door hun keuze van de afstand (p. 19). Solon Barocas, Andrew D. Selbst en Manish Raghavan noemen schalen op basis van de data enigszins willekeurig: de ene kredietverstrekker telt 5000 dollar meer inkomen per jaar misschien even zwaar als een jaar langer in dienst, en een concurrent die op andere data traint 10.000 dollar (p. 7). De schuifbalk hieronder verandert zo'n schaal.

        **Veranderingen die niemand kan maken.** De dichtstbijzijnde counterfactual kan veranderen wat iemand niet kan veranderen. In het experiment van Wachter en anderen met data van rechtenstudenten, met een gewone gekwadrateerde afstand, kreeg ras, gecodeerd als 0 of 1, eerst betekenisloze waarden, gebroken of negatief; toen ze het in twee runs op 0 of 1 vastzetten, stelde hun methode voortaan steeds voor om het ras in 'wit' te veranderen (p. 19). Berk Ustun, Alexander Spangher en Yang Liu vragen in plaats daarvan om **recourse**, nu vaak **algorithmic recourse** genoemd: dat iemand het besluit van een model kan veranderen door input te veranderen waarop die persoon zelf kan handelen, zoals inkomen, en niet leeftijd of burgerlijke staat (p. 1). Amir-Hossein Karimi en anderen delen de kolommen in drie soorten in: actionable, zoals een banksaldo; veranderlijk maar niet actionable, zoals een kredietscore; en onveranderlijk, zoals een geboorteplaats (p. 5). Leeftijd verandert wel, maar niet door iets wat de aanvrager doet. Een counterfactual biedt recourse als hij alleen actionable kolommen verandert, wat je in DiCE vraagt met `features_to_vary`.

        **Veel antwoorden.** Er kunnen nul, één of meerdere beste counterfactuals zijn (Karimi en anderen, p. 7), en welke een methode vindt, hangt af van de afstand, van waar het zoeken begint en van de seed. Welke een bank laat zien, is ook een keuze: Karimi en anderen waarschuwen dat iemand uit veel even aannemelijke verklaringen de maatschappelijk meest aanvaardbare kan kiezen (p. 10).

        **Wat hij niet laat zien.** Een counterfactual zegt wat het antwoord van dit model voor deze persoon zou veranderen, meer niet. Wachter en anderen schrijven zelf dat een counterfactual die een protected attribute niet verandert, geen bewijs is dat dat attribuut niets met het besluit te maken had (p. 15). En hij geldt alleen zolang het model hetzelfde blijft. Kaivalya Rawal, Ece Kamar en Himabindu Lakkaraju gaven recourse onder één model, trainden het opnieuw op nieuwere data en telden hoeveel recourses het nieuwe model afwees: met data over borgtocht uit 1978 en 1980 was dat 96,6% van de recourses van de methode van Ustun en anderen bij een logistische regressie. Over modellen, methoden en veranderingen in de data liep het aandeel van 0% tot 100%, met 0% of bijna 0% bij een paar modellen met bomen en neural networks op dezelfde borgtochtdata (p. 7).

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **counterfactual explanation** | hoe de input anders had moeten zijn om het model een ander antwoord te laten geven, vaak de dichtstbijzijnde zulke verandering, gemeten met een gekozen afstand |
        | **algorithmic recourse** | of iemand het besluit van een model kan veranderen door te veranderen waarop die persoon kan handelen, zoals inkomen, niet leeftijd |
        | **actionable** | een kolom die iemand door te handelen kan veranderen, zoals werkuren; niet leeftijd of geboorteplaats |
        | **mediane absolute afwijking** (MAD) | de mediaan van hoe ver de waarden van een kolom van hun mediaan af liggen: een typische stap in die kolom |
        | $z$ | de logit van de aanvrager; de bank zegt ja vanaf $z = 0$ |
        | $\Delta_k$ | hoeveel kolom $k$ moet veranderen, als alleen die verandert, om de logit op 0 te brengen |
        | $d(x, x')$ | de afstand van Wachter en anderen: de verandering van elke kolom in stappen van zijn MAD, opgeteld |
        """),
        L.watch_out("Een counterfactual explanation is waar voor één model, één afstand en één keuze van kolommen die mogen veranderen. Zeg welke kolommen mochten veranderen en hoe je dichtbij hebt gemeten, en lees een kolom die hij laat staan nooit als een kolom die er niet toe deed."),
    ])
    return


@app.cell(hide_code=True)
def explore_years_cell():
    explore_years = L.explore("years", "jaar leeftijd in één stap", lambda v: C.years_picture(v, "nl"), start=10,
                              values=C.YEARS, role="hyper",
                              task="Schuif van 10, de MAD van leeftijd, omlaag naar 7 en dan naar 1. Bij hoeveel jaar in een stap is 81 zijn niet meer de dichtstbijzijnde van de drie?")
    mo.vstack([
        mo.md("**Hoeveel is een jaar?**"),
        explore_years,
        mo.md(r"""
        Elke staaf is de afstand van de aanvrager van 30 tot een counterfactual die één kolom verandert met het kleinste hele getal dat een ja geeft. Opleiding en uren houden hun MAD's, 1 niveau en 3 uur; de schuifbalk zet hoeveel jaar leeftijd één stap is. Van deze drie ligt 81 zijn het dichtst bij met 8 jaar of meer in een stap, en opleidingsniveau 16 met 7 of minder. Aan de aanvrager en aan het model veranderde niets, alleen aan hoe zwaar een jaar weegt tegen een opleidingsniveau. Met de standaardafwijkingen als schalen, 13,71 jaar, 2,57 niveaus en 12,39 uur, ligt opleidingsniveau 16 het dichtst bij, op 2,72 stappen tegen 3,72 voor 81 zijn.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo geef je een counterfactual explanation:

    1. Schrijf op welk antwoord je wilt: hier een logit van 0 of meer, het ja van de bank.
    2. Zeg welke kolommen mogen veranderen. Voor recourse alleen de actionable.
    3. Kies een schaal voor elke kolom, zoals zijn MAD in de data, en zeg welke.
    4. Zoek veranderingen die het gewenste antwoord geven: bij een logistische regressie $-z / w_k$ voor één kolom; bij elk ander model door te zoeken.
    5. Meet ze elk met de afstand, en laat de paar dichtstbijzijnde zien, niet maar één.
    6. Zeg wat hij niet laat zien: een kolom die hij laat staan, kan toch hebben meegeteld, en een nieuw model kan anders antwoorden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het model en de aanvrager uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), met de afstand van Wachter en anderen. `np.median(..., axis=0)` geeft de mediaan van elke kolom en `np.abs` de grootte van elk getal. `math.ceil` rondt naar boven af op een heel getal, omdat leeftijden, niveaus en uren in de tabel hele getallen zijn, en `.copy()` maakt een nieuwe array, zodat de eigen rij van de aanvrager blijft zoals hij is. Voer de cel uit.
    """)
    return


@app.cell
def example_counterfactuals():
    import math

    import numpy as np
    import pandas as pd

    # de drie kolommen van UCI Adult, het model uit de les over redenen, en de aanvrager van 30
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _weights = np.array([0.046, 0.345, 0.042])
    _bias = -8.48
    _person = np.array([30, 9, 40])
    _logit = _person @ _weights + _bias
    print("logit", round(_logit, 3), " kans op ja", round(1 / (1 + math.exp(-_logit)), 3))

    # de schaal van Wachter en anderen: de mediane absolute afwijking (MAD) van elke kolom
    _mad = np.median(np.abs(_X - np.median(_X, axis=0)), axis=0)
    print("MAD", _mad)

    # één kolom tegelijk: de kleinste hele verandering die de logit op 0 brengt, en de afstand ervan
    for _k in range(3):
        _other = _person.copy()
        _other[_k] = _person[_k] + math.ceil(-_logit / _weights[_k])
        _new = _other @ _weights + _bias
        _distance = (np.abs(_other - _person) / _mad).sum()
        print(_names[_k], "naar", _other[_k], " logit", round(_new, 3), " afstand", round(_distance, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De dichtstbijzijnde counterfactual is 81 zijn, op 5,1 stappen. Waarom leeftijd? Per stap van zijn schaal beweegt elke kolom de logit met zijn gewicht keer zijn MAD: leeftijd met 0,046 · 10 = 0,46, opleiding met 0,345 · 1 = 0,345 en uren met 0,042 · 3 = 0,126. Leeftijd beweegt de logit per stap het meest, dus wie elke stap aan leeftijd besteedt, is er in de minste stappen, en geen mix van kolommen komt dichterbij: een mix besteedt een deel van zijn stappen aan een kolom die de logit minder beweegt. Maar leeftijd is niet actionable. Laat die weg, en opleidingsniveau 16, een doctorstitel voor iemand met een diploma van de middelbare school, ligt het dichtst bij, op 7 stappen. Verder weg liggen onder meer niveau 15 met 46 uur per week (8 stappen), niveau 14 met 55 uur (10), niveau 13 met 63 uur (11,7) en alleen 96 uur per week (18,7). Elk is een echte counterfactual van dit model. Welke de bank stuurt, is een keuze. (De print schrijft de getallen met een punt, zoals Python dat doet.)
    """)
    return


@app.cell(hide_code=True)
def text_exercise_needed():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoeveel moet er veranderen?

        Schrijf `change_needed(weights, bias, person, k)`: `weights` en `person` zijn lijsten met een getal voor elke kolom, `bias` is een getal en `k` is de plaats van een kolom. Begin `logit` bij `bias` en tel er voor elke kolom `j` `weights[j] * person[j]` bij op. Geef terug hoeveel kolom `k` moet veranderen, als alleen die verandert, om de logit op 0 te brengen: min de logit gedeeld door `weights[k]`. De check roept eerst `change_needed([0.046, 0.345, 0.042], -8.48, [30, 9, 40], 1)` aan: de aanvrager van 30 heeft 6,71 opleidingsniveaus meer nodig. De tekening voert je functie uit voor elke kolom van die aanvrager.
        """),
        L.task("Schrijf `change_needed(weights, bias, person, k)`: reken de logit uit, en geef terug hoeveel kolom `k` moet veranderen, als alleen die verandert, om hem op 0 te brengen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_needed():
    box_needed = L.code_box("needed", lines=6)
    box_needed
    return (box_needed,)


@app.cell(hide_code=True)
def check_needed(box_needed):
    result_needed = L.check("needed", box_needed.value)
    result_needed
    return (result_needed,)


@app.cell(hide_code=True)
def hints_needed():
    L.hints(
        "needed",
        [
            "De logit is een som die bij de bias begint. Als kolom `k` met een hoeveelheid verandert, beweegt de logit met `weights[k]` keer die hoeveelheid, en de logit moet met min zichzelf bewegen om op 0 te komen.",
            "Begin met `logit = bias`, en tel binnen `for j in range(len(person)):` er `weights[j] * person[j]` bij op. Na de loop is de verandering min de logit, gedeeld door het gewicht van kolom `k`.",
            "`return ___ / weights[k]`",
        ],
        why="`logit = bias`, dan `for j in range(len(person)):` met `logit = logit + weights[j] * person[j]`, en na de loop `return -logit / weights[k]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoe ver ligt een counterfactual?

        Schrijf `distance(person, other, scale)`, de afstand van Wachter en anderen: `person` en `other` zijn twee rijen, en `scale` heeft een getal voor elke kolom. Begin `total` bij 0 en tel er voor elke kolom `k` de grootte van zijn verandering bij op, `abs(other[k] - person[k])`, gedeeld door `scale[k]`; `abs` is de absolute waarde in Python, het getal zonder minteken. Geef `total` terug. De check roept eerst `distance([30, 9, 40], [30, 15, 46], [10, 1, 3])` aan: opleidingsniveau 15 met 46 uur per week ligt 0 + 6 + 2 = 8 stappen van de aanvrager af. De tekening voert je functie uit op zes counterfactuals van de aanvrager, met de MAD's als schaal.
        """),
        L.task("Schrijf `distance(person, other, scale)`: tel voor elke kolom `k` de grootte van zijn verandering op, gedeeld door `scale[k]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_distance():
    box_distance = L.code_box("distance", lines=6)
    box_distance
    return (box_distance,)


@app.cell(hide_code=True)
def check_distance(box_distance):
    result_distance = L.check("distance", box_distance.value)
    result_distance
    return (result_distance,)


@app.cell(hide_code=True)
def hints_distance():
    L.hints(
        "distance",
        [
            "Een verandering omlaag is even ver als een verandering omhoog, en een jaar is geen niveau: neem de grootte van elke verandering, en tel die in stappen van de schaal van zijn eigen kolom.",
            "Begin met `total = 0` en loop met `for k in range(len(person)):`. Tel er elke keer de grootte van de verandering bij op, `abs(other[k] - person[k])`, gedeeld door de schaal van die kolom. Geef na de loop `total` terug.",
            "`total = total + abs(other[k] - person[k]) / ___`",
        ],
        why="`total = 0`, dan `for k in range(len(person)):` met `total = total + abs(other[k] - person[k]) / scale[k]`, en na de loop `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_race():
    mo.md(r"""
    ### Oefening 3. Een kolom die hij laat staan

    Een counterfactual noemt één weg naar een ander antwoord. Wat zegt hij over de kolommen die hij niet verandert?
    """)
    return


@app.cell(hide_code=True)
def choice_race_cell():
    choice_race = L.choice(
        "race",
        question="Het model van een bank gebruikt ras als een van zijn kolommen. De counterfactual die de bank een aanvrager stuurt, verandert alleen de opleiding. Laat dat zien dat ras geen rol speelde bij het nee?",
        options={
            "right": "Nee, hij zegt niets over ras",
            "closest": "Ja: anders lag de dichtstbijzijnde verandering bij ras",
            "distance": "Ja, als de bank de afstand van Wachter en anderen gebruikte",
        },
        correct="right",
        why={
            "right": "Goed. Een kolom die een counterfactual laat staan, kan toch hebben meegeteld; de dichtstbijzijnde verandering gaat alleen naar waar een stap het antwoord het meest beweegt. Wachter en anderen zeggen het zelf: een counterfactual die een protected attribute niet verandert, is geen bewijs dat dat attribuut niets met het besluit te maken had. Of ras meetelt, is een vraag over veel besluiten, niet over één uitleg.",
            "closest": "De dichtstbijzijnde verandering gaat naar waar een stap de logit het meest beweegt, niet naar de kolom die het zwaarst tegen de aanvrager woog. Voor de aanvrager van 30 verandert de dichtstbijzijnde counterfactual alleen leeftijd, terwijl opleiding en uren ook tegen de aanvrager wogen. Wachter en anderen: een counterfactual die een protected attribute laat staan, is geen bewijs dat het er niet toe deed (p. 15).",
            "distance": "De afstand bepaalt welke counterfactual het dichtst bij ligt, niet of ras ertoe deed. Een counterfactual die ras laat staan, zegt er niets over, welke afstand je ook neemt (Wachter en anderen, p. 15). Zie *Wat hij niet laat zien* in *Het idee*.",
        },
    )
    choice_race
    return (choice_race,)


@app.cell(hide_code=True)
def text_exercise_later():
    mo.md(r"""
    ### Oefening 4. Als het model verandert

    Een counterfactual is een uitspraak over één model. Banken trainen hun modellen opnieuw.
    """)
    return


@app.cell(hide_code=True)
def choice_later_cell():
    choice_later = L.choice(
        "later",
        question="De aanvrager haalt opleidingsniveau 16, de recourse die de bank noemde. Intussen heeft de bank zijn model opnieuw getraind op nieuwere data. Krijgt de aanvrager nu een ja?",
        options={
            "promised": "Ja: de counterfactual van de bank beloofde het",
            "right": "Niet zeker: het nieuwe model kan nee zeggen",
            "cutoff": "Ja, zolang de grens nog op 0 ligt",
        },
        correct="right",
        why={
            "right": "Goed. Een counterfactual is waar voor het model dat hem maakte. Rawal, Kamar en Lakkaraju trainden modellen opnieuw op nieuwere data en telden hoeveel recourses het nieuwe model afwees: met data over borgtocht uit 1978 en 1980 was dat 96,6% van de recourses van de methode van Ustun en anderen bij een logistische regressie, al was het bij een paar modellen met bomen en neural networks 0% of bijna 0%. Barocas en anderen rekenen een model dat in de tijd hetzelfde blijft tot de verborgen aannames van counterfactual explanations.",
            "promised": "De counterfactual is een uitspraak over het oude model: met niveau 16 zegt dat model ja. Een opnieuw getraind model heeft andere weights. Met data over borgtocht uit 1978 en 1980 vonden Rawal, Kamar en Lakkaraju dat een nieuwe logistische regressie 96,6% van de recourses van de methode van Ustun en anderen afwees.",
            "cutoff": "De grens kan op 0 blijven terwijl de weights veranderen: een opnieuw getraind model heeft andere weights, dus dezelfde rij krijgt een andere logit. Met data over borgtocht uit 1978 en 1980 vonden Rawal, Kamar en Lakkaraju dat een nieuwe logistische regressie 96,6% van de recourses van de methode van Ustun en anderen afwees.",
        },
    )
    choice_later
    return (choice_later,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl) vertelde wat het zwaarst tegen de aanvrager woog; een counterfactual zegt wat het antwoord zou veranderen, en die twee hoeven niet dezelfde kolom te noemen. [Shapley-waarden](/course/open/rai-shapley-values?lang=nl) verdeelde één voorspelling over de features; een counterfactual zoekt een andere voorspelling in de buurt. DiCE uit [Tools voor uitleg](/course/open/rai-xai-tools?lang=nl) zoekt voor elk model meerdere counterfactuals tegelijk, en met `features_to_vary` vraag je het om recourse. De volgende les, [Project: leg je classifier uit](/course/open/rai-project-audit-explain?lang=nl), Daarin pas je deze les en de lessen ervoor toe op een eigen model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_later, choice_race, result_distance, result_needed):
    L.wrap_up(result_needed, result_distance, choice_race, choice_later)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
