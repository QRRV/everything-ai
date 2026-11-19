# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-k-anonymity. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-k-anonymity", "nl")
    C = L.checks   # UCI Adult als PEOPLE en BANDED, de vijf quasi-identifiers als QI, en een verzonnen kliniek


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl) en [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hash():
    quick_hash = L.choice(
        "quick-hash",
        question="Een app vervangt elk e-mailadres in zijn rijen door de SHA-256-hash van dat adres, en deelt de rijen. Is het resultaat anoniem?",
        options={
            "yes": "Ja: een hash is niet terug te draaien naar tekst",
            "right": "Nee: hash waarschijnlijke adressen en vergelijk",
            "key": "Ja, zolang er nergens een sleutellijst bewaard wordt",
        },
        correct="right",
        why={
            "right": "Wie de adressen kan raden, hasht ze en vergelijkt, zoals `unmask` deed met de lijst van het café. Dat is pseudonimisering: de rijen blijven persoonsgegevens.",
            "yes": "Een hash hoeft niet teruggedraaid te worden: wie de adressen kan raden, hasht ze en vergelijkt. Zie *Een hash is geen schuilplaats* in *Het idee* van [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl).",
            "key": "Een gewone hash heeft geen sleutellijst nodig: wie de adressen kan raden, hasht ze en vergelijkt. Zie *Een hash is geen schuilplaats* in *Het idee* van [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl).",
        },
    )
    quick_hash
    return (quick_hash,)


@app.cell(hide_code=True)
def quick_check_size():
    quick_size = L.choice(
        "quick-size",
        question="`df` is de tabel met 344 pinguïns. Wat geeft `df.groupby(\"species\").size()`?",
        options={
            "right": "Het aantal rijen van elke soort",
            "groups": "Hoeveel soorten er zijn",
            "mean": "Het gemiddelde van elke kolom, per soort",
        },
        correct="right",
        why={
            "right": "`size` geeft per groep hoeveel rijen hij heeft: één getal voor elke soort.",
            "groups": "`size` geeft één getal per groep: hoeveel rijen die groep heeft. Hoeveel groepen er zijn, is de lengte van die uitkomst. Zie *Nieuwe woorden en tekens* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "mean": "Een gemiddelde vraagt om `.mean()`; `size` telt de rijen van elke groep. Zie *Nieuwe woorden en tekens* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
        },
    )
    quick_size
    return (quick_size,)


@app.cell(hide_code=True)
def quick_check_pairs():
    quick_pairs = L.choice(
        "quick-pairs",
        question="Wat geeft `df.groupby([\"island\", \"species\"]).size()`?",
        options={
            "apart": "Eén telling per eiland, en één per soort",
            "right": "Eén telling per paar van eiland en soort",
            "first": "Eén telling per eiland; de soort telt niet mee",
        },
        correct="right",
        why={
            "right": "Met een lijst kolommen is een groep een combinatie van waarden die voorkomt, zoals Biscoe en Gentoo; `size` telt de rijen van elk zo'n paar.",
            "apart": "Met een lijst kolommen zijn de groepen de combinaties van waarden, niet elke kolom apart: Biscoe en Gentoo is één groep. Zie stap 4 van het *Recept* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "first": "Beide kolommen maken de groepen: elke combinatie van eiland en soort die voorkomt, is één groep. Zie stap 4 van het *Recept* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
        },
    )
    quick_pairs
    return (quick_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een gangbare manier om gegevens over mensen veilig te delen, is iedereen te laten opgaan in een menigte: zolang elke combinatie van feiten vaak genoeg voorkomt, is niemand eruit te pikken. Dat idee heet k-anonimiteit, en het is een van de bekendste antwoorden op het probleem van de vorige les, waar drie feiten genoeg waren om één rij eruit te pikken. Het is makkelijk te meten en te bereiken: je telt, maakt grover en laat weg. Het heeft ook een bekend gat: als iedereen in een groep hetzelfde geheim heeft, beschermt het opgaan in die groep het geheim niet. In deze les meet je allebei op UCI Adult, een tabel met 48.842 mensen: k, de grootte van de kleinste groep, en l, het kleinste aantal verschillende geheimen in een groep.

    **Een voorbeeld uit het dagelijks leven.** Een bedrijf houdt een enquête onder het personeel, en laat de antwoorden alleen zien voor teams van minstens vijf mensen, zodat geen enkel antwoord terug te leiden is. Maar als alle vijf in een team nee antwoorden op "Ben je tevreden over je leidinggevende?", kent de leidinggevende toch ieders antwoord.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="UCI Adult bevat 48.842 mensen uit een Amerikaanse enquête van 1994. Kijk alleen naar hun leeftijd, geslacht, ras, burgerlijke staat en opleiding. Hoeveel van hen zijn de enige in de tabel met hun combinatie van die vijf?",
        options={
            "none": "Niet één persoon",
            "few": "Ongeveer 400",
            "right": "Ongeveer 4000",
        },
        reveal="**Ongeveer 4000**: 3948 mensen, 8% van de tabel, zijn de enige met hun leeftijd, geslacht, ras, burgerlijke staat en opleiding. Voor elk van hen pikken die vijf feiten één rij eruit. De kleinste groep reken je zelf uit in Oefening 1.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Schuilen in een groep

        In [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl) lieten drie feiten over Bo één rij over: Bo was eruit gepikt. Latanya Sweeney (2002) maakte daar een eis aan een hele tabel van: een vrijgegeven tabel biedt k-anonimiteit als de gegevens van elke persoon erin niet te onderscheiden zijn van die van minstens k−1 andere mensen die er ook in staan. Het gaat om de feiten die iemand ook van elders kan weten, zoals leeftijd, postcode of geslacht. Samen vormen zulke kolommen een **quasi-identifier**, een term die Sweeney van Dalenius overneemt. **k-anonimiteit** vraagt dat elke combinatie van waarden van de quasi-identifier in minstens k rijen voorkomt. In pandas-taal: groepeer de rijen op de quasi-identifier, en de kleinste groep heeft minstens k rijen. Wie alleen de quasi-identifier van iemand kent, vindt dan minstens k rijen die passen, zoals `candidates` zou doen, en kan niet zien welke van die persoon is.

        Het idee is ouder dan 2002. Volgens Sweeney is de publicatie van 2002 een flink aangepaste en uitgebreide versie van een publicatie met Pierangela Samarati, ingediend in 1998, en Sweeney geeft Samarati de eer voor de naam. Die eerdere publicatie zegt waarom het nodig is: er bestaat nog steeds het wijdverbreide misverstand dat gegevens die er anoniem uitzien, ook anoniem zijn.
        """),
        mo.Html(C.clinic_picture("nl")),
        mo.md(r"""
        ### Grover maken of weglaten

        Een tabel is zelden vanzelf k-anoniem. In een tweede publicatie uit 2002 noemt Sweeney twee manieren om er te komen. **Generaliseren** is een waarde vervangen door een minder precieze die er nog bij past; **weglaten** (suppression) is een waarde helemaal niet vrijgeven. Het voorbeeld van generaliseren in die publicatie: de postcodes 02138 en 02139 worden allebei 0213*. In het plaatje werd elke leeftijd de tien jaar waarin hij valt, en verloor elke postcode zijn laatste cijfer; weglaten zou een waarde wissen, of een hele rij waarvan de groep nog te klein is.

        Allebei kost het iets: een grovere waarde zegt minder, en een weggelaten rij is verloren voor onderzoek. Voor een studie naar studenten en belastingen in Estland schrijven Bogdanov e.a. (2016) dat 3-anonimiteit 10 tot 30% van de steekproef kostte, afhankelijk van de groep; de studie koos een andere weg, die terugkomt in *Rekenen met geheimen*. In UCI Adult zitten, met leeftijd, geslacht, ras, burgerlijke staat en opleiding, 9810 van de 48.842 mensen in een groep van minder dan 5: de tabel 5-anoniem maken door alleen rijen weg te laten, zou 20% ervan kosten.

        ### Wat k niet belooft

        Sweeney schrijft het zelf: de eigenschap garandeert niet dat niemand in de vrijgegeven tabel te identificeren is, want er kunnen andere aanvallen zijn die iemands identiteit verraden. En k-anonimiteit rust op een aanname: dat wie de gegevens beheert, de quasi-identifiers goed kan aanwijzen. Een kolom die buiten de quasi-identifier blijft, omdat niemand dacht dat iemand hem kon weten, is een weg naar binnen. Dezelfde publicatie beschrijft drie aanvallen op tabellen die k-anoniem zijn: een op de volgorde van de rijen (de oplossing is ze te schudden), een die een tweede, anders grover gemaakte versie van dezelfde gegevens erbij legt, en een op gegevens die in de loop van de tijd veranderen.

        ### Als de hele groep het geheim deelt

        Ashwin Machanavajjhala, Johannes Gehrke, Daniel Kifer en Muthuramakrishnan Venkitasubramaniam (2006) lieten twee aanvallen op k-anonimiteit zien. Ten eerste kan een aanvaller de geheime waarden achterhalen als die binnen een groep weinig verschillen. Ten tweede hebben aanvallers vaak achtergrondkennis, en daartegen biedt k-anonimiteit geen garantie. Een kolom die geheim moet blijven, zoals een aandoening of een inkomen, noemt deze les een **gevoelige kolom**.

        Hun voorbeeld is een verzonnen ziekenhuis, met een tabel die 4-anoniem is op postcode, leeftijd en nationaliteit. Alice weet dat Bob, een buurman, 31 is, Amerikaans, en in postcode 13053 woont. Er passen vier rijen, en in alle vier staat kanker: Alice weet nu dat Bob kanker heeft, zonder te weten welke rij van Bob is. Dat is de **homogeneity attack**. Bij de tweede aanval maakt wat Alice al weet de andere waarden van een groep heel onwaarschijnlijk. Op hun versie van UCI Adult, zonder de rijen met een ontbrekende waarde (45.222 mensen), met leeftijd, geslacht, ras, burgerlijke staat en opleiding als quasi-identifier, maakten ze elke minimale 6-anonieme versie van de tabel, niet grover dan nodig: met de inkomensklasse (de kolom `income`) als geheim waren er 8 van de 9 kwetsbaar voor de homogeneity attack; met het beroep 1 van de 12.

        ### l-diversity: verschillende geheimen in elke groep

        Hun antwoord: een groep is ℓ-divers als hij minstens ℓ goed vertegenwoordigde waarden van de gevoelige kolom bevat, en een tabel is ℓ-divers als elke groep dat is. Een groep heet bij hen een q⋆-blok, en de course schrijft **l-diversity** met een gewone l. Ze noemen het een principe, nog geen definitie, en maken "goed vertegenwoordigd" op twee manieren precies, allebei strenger dan de eenvoudigste lezing. De eenvoudigste lezing, minstens l verschillende waarden in elke groep, kreeg de naam distinct l-diversity van Ninghui Li, Tiancheng Li en Suresh Venkatasubramanian (2007). Die waarschuwen dat ze aanvallen die met kansen redeneren niet tegenhoudt. Verschillende waarden kunnen ook bijna hetzelfde betekenen: in hun voorbeeld bevat een groep een maagzweer, een maagontsteking en maagkanker, en leert een aanvaller dat iemand erin een maagziekte heeft. Hun oordeel over l-diversity: het is niet nodig en niet genoeg om te voorkomen dat een gevoelige waarde uitlekt. Ze stellen t-closeness voor: in elke groep moet de verdeling van de geheime waarden binnen een afstand t blijven van hun verdeling in de hele tabel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **quasi-identifier** | feiten die op zichzelf niet zeggen wie iemand is, zoals leeftijd, postcode of geslacht, maar in combinatie wel; in een tabel een set kolommen |
        | **k-anonimiteit** | elke combinatie van waarden van de quasi-identifier komt in minstens k rijen voor; de tabel is dan k-anoniem |
        | **k** | het aantal rijen in de kleinste groep |
        | **generaliseren** | een waarde vervangen door een minder precieze die er nog bij past, zoals 37 door 30-39 |
        | **weglaten** (suppression) | een waarde helemaal niet vrijgeven, of een hele rij |
        | **gevoelige kolom** | een kolom die geheim moet blijven, zoals een aandoening of een inkomen; in de publicaties heet zo'n kolom een sensitive attribute |
        | **homogeneity attack** | iemands geheim leren omdat alle rijen van diens groep het delen, zonder te weten welke rij van die persoon is |
        | **l-diversity** | elke groep heeft minstens l goed vertegenwoordigde waarden van de gevoelige kolom; in de eenvoudigste lezing minstens l verschillende waarden |
        | **l** | in de eenvoudigste lezing het kleinste aantal verschillende waarden van de gevoelige kolom in één groep |
        | **nunique** | per groep hoeveel verschillende waarden een kolom heeft, zonder de ontbrekende waarden: `df.groupby("sex")["race"].nunique()` |
        """),
        L.watch_out("k-anonimiteit beschermt alleen tegen iemand van wie de kennis past bij de quasi-identifier die jij koos, en alleen tegen het eruit pikken van een rij. Het zegt niets over wat de rijen van een groep gemeen hebben: tel l naast k, en noem een tabel niet anoniem omdat k groot is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan hoe goed een tabel de mensen erin verbergt:

    1. Kies de quasi-identifier: elke kolom die iemand ook van elders kan weten, zoals leeftijd, geslacht of waar iemand woont.
    2. Groepeer de rijen op die kolommen, en neem het aantal rijen in de kleinste groep: dat is k. Dat is `k_of`.
    3. Is k te klein, maak dan de kolommen grover die de groepen klein maken, zoals leeftijd in banden van tien jaar, en laat de rijen weg waarvan de groep nog steeds minder dan k rijen heeft. Dat is `band`.
    4. Tel in elke groep de verschillende waarden van de gevoelige kolom; de kleinste telling is l. Is l gelijk aan 1, dan is iemands geheim bekend zonder dat je diens rij vindt. Dat is `l_of`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest UCI Adult in, groepeert de mensen op geslacht en ras, telt de mensen in elke groep, en neemt de kleinste telling: de k voor deze twee kolommen. Daarna voegt hij de burgerlijke staat toe.
    """)
    return


@app.cell
def example_groups():
    import pandas as pd

    _people = pd.read_csv("data/adult.csv.gz")
    # hoeveel mensen elke combinatie van geslacht en ras delen
    _sizes = _people.groupby(["sex", "race"]).size()
    print(_sizes)
    print("smallest group:", _sizes.min())
    print("with marital status too:", _people.groupby(["sex", "race", "marital_status"]).size().min())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met geslacht en ras heeft de kleinste groep 155 mensen: de vrouwen van wie het ras als Other staat genoteerd. Op die twee kolommen is de tabel 155-anoniem. Voeg de burgerlijke staat toe, en twee mensen zijn de enige met hun combinatie: k zakt naar 1. Elke kolom die je toevoegt, kan de groepen verder splitsen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_k():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. k_of: de kleinste groep

        `PEOPLE` is UCI Adult: 48.842 mensen, elk een rij, met kolommen zoals `age`, `sex`, `race`, `marital_status`, `education`, `occupation` en `income` (`"<=50K"` of `">50K"`: hoogstens of meer dan 50.000 dollar per jaar). `QI` is de lijst `["age", "sex", "race", "marital_status", "education"]`: de quasi-identifier van Machanavajjhala en collega's.

        `k_of(df, columns)` krijgt een tabel `df` en een lijst kolomnamen `columns`. Hij geeft het aantal rijen in de kleinste groep rijen met dezelfde waarden in `columns` terug: de k waarvoor `df` k-anoniem is op die kolommen. Vervang de `...` door één expressie: groepeer `df` op `columns`, tel de rijen van elke groep, en neem de kleinste telling. De check roept `k_of(PEOPLE, QI)` aan, dan minder kolommen, en de kliniek uit het plaatje: `CLINIC` vóór en `RELEASED` na het grover maken.
        """),
        L.task("Geef het aantal rijen in de kleinste groep rijen met dezelfde waarden in columns terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_k():
    box_k_of = L.code_box("k_of", lines=3)
    box_k_of
    return (box_k_of,)


@app.cell(hide_code=True)
def check_k(box_k_of):
    result_k_of = L.check("k_of", box_k_of.value)
    result_k_of
    return (result_k_of,)


@app.cell(hide_code=True)
def hints_k():
    L.hints(
        "k_of",
        [
            "Stap 2 van het recept. Eén groep per combinatie van waarden, een telling per groep, en dan één getal: de kleinste telling.",
            "`df.groupby(columns)` maakt de groepen. Welke methode telt de rijen van elke groep, en welke neemt de kleinste van die tellingen?",
            "`return df.groupby(columns).size().___()`",
        ],
        why="`df.groupby(columns).size()` telt de rijen van elke groep, en `.min()` neemt de kleinste. `k_of(PEOPLE, QI)` geeft 1: 3948 mensen zijn de enige met hun combinatie van de vijf kolommen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_band():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. band: tien jaar in plaats van één

        `band(age)` krijgt een heel getal `age`, en geeft de tien jaar waarin het valt terug, als tekst: `"30-39"` voor 37, en ook voor 30 en voor 39. `low` is de laagste leeftijd van de band: 30 voor 37. Vervang de eerste `...` door een expressie voor `low`, en de tweede door de tekst van de band: van `low` tot de hoogste leeftijd van de band, met een streepje en zonder spaties. De check roept `band(37)` aan, en dan andere leeftijden, van 5 tot 100.
        """),
        L.task("Geef de tien jaar waarin de leeftijd valt terug, als tekst zoals \"30-39\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_band():
    box_band = L.code_box("band", lines=4)
    box_band
    return (box_band,)


@app.cell(hide_code=True)
def check_band(box_band):
    result_band = L.check("band", box_band.value)
    result_band
    return (result_band,)


@app.cell(hide_code=True)
def hints_band():
    L.hints(
        "band",
        [
            "Stap 3 van het recept. 37 ligt tussen 30 en 39: rond het naar beneden af op een tiental. Deel door 10, laat weg wat na de punt komt, en vermenigvuldig weer met 10.",
            "`int(...)` laat weg wat na de punt komt, zoals in [Variabelen en getallen](/course/open/py-variables?lang=nl): `int(3.7)` geeft `3`. De band loopt dan van `low` tot de hoogste leeftijd van de band, in een f-string: voor 37 van 30 tot 39.",
            "`low = int(age / 10) * 10` en `return f\"{low}-{___}\"`",
        ],
        why="`low = int(age / 10) * 10` geeft 30 voor 37, en `return f\"{low}-{low + 9}\"` geeft `\"30-39\"`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_l():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. l_of: verschillende geheimen in elke groep

        `BANDED` is `PEOPLE` met één kolom meer, `band`: de tien jaar van ieders leeftijd, gemaakt met de `band` van de course. Gegroepeerd op `band` en `sex` heeft hij 18 groepen, en de kleinste heeft 17 mensen: `BANDED` is 17-anoniem op die twee kolommen.

        `l_of(df, columns, secret)` krijgt een tabel `df`, een lijst kolomnamen `columns`, en de naam van nog een kolom, `secret`. Hij geeft het kleinste aantal verschillende waarden van `secret` in één groep rijen met dezelfde waarden in `columns` terug: de l waarvoor `df` l-divers is, in de eenvoudigste lezing. Vervang de `...` door één expressie: groepeer `df` op `columns`, neem de kolom `secret`, tel in elke groep de verschillende waarden, en neem de kleinste telling. De check roept `l_of(BANDED, ["band", "sex"], "income")` aan, dan andere kolommen en geheimen, de kliniek, en een kleine tabel `GAPS` met een ontbrekende waarde.
        """),
        L.task("Geef het kleinste aantal verschillende waarden van secret in één groep rijen met dezelfde waarden in columns terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_l():
    box_l_of = L.code_box("l_of", lines=3)
    box_l_of
    return (box_l_of,)


@app.cell(hide_code=True)
def check_l(box_l_of):
    result_l_of = L.check("l_of", box_l_of.value)
    result_l_of
    return (result_l_of,)


@app.cell(hide_code=True)
def hints_l():
    L.hints(
        "l_of",
        [
            "Stap 4 van het recept. Zoals bij `k_of`, maar in elke groep tel je de verschillende waarden van één kolom in plaats van de rijen.",
            "Na `df.groupby(columns)` neem je met vierkante haken één kolom, zoals in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl). Welke methode uit *Nieuwe woorden en tekens* telt de verschillende waarden?",
            "`return df.groupby(columns)[secret].___().min()`",
        ],
        why="`df.groupby(columns)[secret].nunique()` telt in elke groep de verschillende waarden van `secret`, en `.min()` neemt de kleinste. `l_of(BANDED, [\"band\", \"sex\"], \"income\")` geeft 1: in één groep heeft iedereen dezelfde inkomensklasse.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bigger():
    mo.md(r"""
    ### Oefening 4. Een grotere k

    Machanavajjhala en collega's maakten UCI Adult ook k-anoniem op leeftijd, geslacht, ras, burgerlijke staat en opleiding, met de inkomensklasse als geheim: op elke minimale manier, één keer voor k = 2, één keer voor k = 5 en één keer voor k = 50.
    """)
    return


@app.cell(hide_code=True)
def choice_bigger_cell():
    choice_bigger = L.choice(
        "bigger-k",
        question="Wat gebeurde er met de homogeneity attack toen k groter werd?",
        options={
            "gone": "Vanaf k = 5 was hij verdwenen",
            "right": "Er lekten minder rijen, maar hij bleef",
            "grew": "Hij verraadde meer rijen naarmate k groeide",
        },
        correct="right",
        why={
            "right": "In de tijdschriftversie van hun publicatie werd de inkomensklasse van gemiddeld 558 rijen verraden bij k = 2, van 381,58 bij k = 5, en van 202,80 bij k = 50, waar alle 5 minimale tabellen kwetsbaar waren. Een grotere k vraagt om grotere groepen, niet om gevarieerdere.",
            "gone": "Bij k = 5 waren 11 van de 12 minimale tabellen nog kwetsbaar, en bij k = 50 alle 5. Een grotere k vraagt om grotere groepen, niet om gevarieerdere. Zie *Als de hele groep het geheim deelt* in *Het idee*.",
            "grew": "Er werden minder rijen verraden naarmate k groeide: gemiddeld 558 bij k = 2, 202,80 bij k = 50. Maar de aanval verdween niet. Zie *Als de hele groep het geheim deelt* in *Het idee*.",
        },
    )
    choice_bigger
    return (choice_bigger,)


@app.cell(hide_code=True)
def text_exercise_skewness():
    mo.md(r"""
    ### Oefening 5. Twee waarden, één antwoord

    Een test is positief voor 1% van de 10.000 mensen in een tabel. In de vrijgegeven tabel heeft één groep 49 mensen met een positieve test en 1 met een negatieve. De groep heeft twee verschillende waarden, dus in de eenvoudigste lezing is hij 2-divers.
    """)
    return


@app.cell(hide_code=True)
def choice_skewness_cell():
    choice_skewness = L.choice(
        "skewness",
        question="Wat leert een aanvaller over iemand van wie bekend is dat die in die groep zit?",
        options={
            "low": "Dat de kans 1% is, net als voor iedereen",
            "half": "Dat het 50-50 is: er zijn twee waarden",
            "right": "Dat die heel waarschijnlijk positief is",
        },
        correct="right",
        why={
            "right": "Li, Li en Venkatasubramanian (2007): iemand in zo'n groep zou voor 98% als positief gelden, in plaats van 1%. Twee verschillende waarden betekenen niet twee even waarschijnlijke.",
            "low": "De 1% geldt voor de hele tabel, niet voor deze groep: daarin testten 49 van de 50 positief. Zie *l-diversity: verschillende geheimen in elke groep* in *Het idee*.",
            "half": "Twee waarden kunnen verre van even waarschijnlijk zijn: 49 van de 50 mensen in de groep testten positief. Zie *l-diversity: verschillende geheimen in elke groep* in *Het idee*.",
        },
    )
    choice_skewness
    return (choice_skewness,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl) pikten drie feiten één rij eruit. k-anonimiteit zorgt dat bij elke combinatie van waarden van de quasi-identifier minstens k rijen passen, en l-diversity dat de rijen van een groep niet allemaal hun geheim delen. Allebei beschermen ze alleen tegen iemand van wie de kennis past bij de quasi-identifier die jij koos. In de volgende les, *Tabellen koppelen*, koppel je een tabel zonder namen aan een openbare tabel op zulke kolommen, en zie je de namen terugkomen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bigger, choice_skewness, result_band, result_k_of, result_l_of):
    L.wrap_up(result_k_of, result_band, result_l_of, choice_bigger, choice_skewness)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
