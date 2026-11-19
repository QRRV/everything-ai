# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-linkage-attacks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-linkage-attacks", "nl")
    C = L.checks   # een verzonnen stadje: de kiezerslijst VOTERS, de vrijgegeven bezoeken VISITS, het hele stadje PEOPLE, en QI


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Verstoppen in een groep: k-anonimiteit](/course/open/rai-k-anonymity?lang=nl) en [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_k():
    quick_k = L.choice(
        "quick-k",
        question="Gegroepeerd op postcode, geboortejaar en geslacht heeft een tabel drie groepen: van 4, 9 en 2 rijen. Wat is de grootste k waarvoor de tabel k-anoniem is?",
        options={
            "right": "k = 2, de kleinste groep",
            "largest": "k = 9, de grootste groep",
            "all": "k = 15, alle rijen bij elkaar",
        },
        correct="right",
        why={
            "right": "Elke combinatie van waarden moet in minstens k rijen voorkomen, dus de kleinste groep bepaalt k: 2.",
            "largest": "k-anonimiteit vraagt dat elke groep minstens k rijen heeft, dus de kleinste groep bepaalt k, niet de grootste. Zie *Verstoppen in een groep* in *Het idee* van [Verstoppen in een groep: k-anonimiteit](/course/open/rai-k-anonymity?lang=nl).",
            "all": "k gaat over de groepen, niet over de hele tabel: de kleinste groep heeft 2 rijen. Zie *Verstoppen in een groep* in *Het idee* van [Verstoppen in een groep: k-anonimiteit](/course/open/rai-k-anonymity?lang=nl).",
        },
    )
    quick_k
    return (quick_k,)


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="Dinsdag staat één keer in `left` en twee keer in `right`. Hoeveel rijen voor dinsdag geeft `pd.merge(left, right, on=\"day\")`?",
        options={
            "first": "Eén: merge houdt de eerste match",
            "right": "Twee, één per match",
            "none": "Geen: de sleutel moet uniek zijn",
        },
        correct="right",
        why={
            "right": "Elk paar rijen met dezelfde sleutel wordt één rij, dus dinsdag komt er twee keer uit. Daarom tel je de rijen voor en na een merge.",
            "first": "merge houdt elke match: elk paar rijen met dezelfde sleutel wordt één rij. Zie *Het idee* van [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).",
            "none": "Een sleutel mag vaker voorkomen; elk paar rijen met dezelfde sleutel wordt één rij. Zie *Het idee* van [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def quick_check_inner():
    quick_inner = L.choice(
        "quick-inner",
        question="Maandag staat in `left` maar niet in `right`. Staat er een rij voor maandag in `pd.merge(left, right, on=\"day\")`?",
        options={
            "missing": "Ja, met ontbrekende waarden aan de rechterkant",
            "right": "Nee: een inner join laat maandag weg",
            "longer": "Ja, als `right` meer rijen heeft",
        },
        correct="right",
        why={
            "right": "Zonder `how` doet merge een inner join: een sleutel die niet in beide tabellen staat, valt weg, zonder melding.",
            "missing": "Dat doet `how=\"left\"`. Zonder `how` doet merge een inner join, en die laat maandag weg. Zie *Nieuwe woorden en tekens* in [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).",
            "longer": "Het aantal rijen doet er niet toe: zonder `how` doet merge een inner join, en die laat een sleutel weg die niet in beide staat. Zie *Nieuwe woorden en tekens* in [Twee tabellen samenvoegen: merge](/course/open/data-merge?lang=nl).",
        },
    )
    quick_inner
    return (quick_inner,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tabel zonder namen kan toch verraden wie erin staat, zodra iemand hem naast een tabel met namen legt. Latanya Sweeney liet dat in de jaren negentig zien door de kiezerslijst van de stad Cambridge naast medische gegevens te leggen die een instantie van de staat Massachusetts over ambtenaren had vrijgegeven. Later opnieuw bekeken, is het geval minder eenvoudig dan de korte versie: een kiezerslijst bevat niet iedereen, dus een naam die past is niet altijd de juiste naam. In deze les doe je allebei in een verzonnen stadje: je koppelt een vrijgegeven tabel van 1000 bezoeken aan de kiezerslijst van het stadje, telt de bezoeken die een naam krijgen, en ziet daarna, met wat alleen de course weet, hoeveel van die namen fout zijn.

    **Een voorbeeld uit het dagelijks leven.** Een hardloopclub zet de uitslag van een wedstrijd online zonder namen: alleen leeftijdsgroep, geslacht en eindtijd. In de hardloopapp van een buurman staat een loopje van die ochtend, op hetzelfde parcours, met dezelfde tijd. Leg de twee naast elkaar, en één uitslag heeft een naam.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="De kiezerslijst van Cambridge, Massachusetts, die Sweeney kocht, bevatte 54.805 kiezers, met hun geboortedatum en een van vijf postcodes. Hoeveel van hen hadden een combinatie van geboortedatum en postcode die geen andere kiezer op de lijst had?",
        options={
            "few": "Bijna niemand",
            "half": "Minder dan de helft",
            "right": "Ongeveer 7 op 10",
        },
        reveal="**Ongeveer 7 op 10**: 69%, volgens Pierangela Samarati en Latanya Sweeney (1998). Met het geslacht erbij telt Sweeney (2000) 2 × 5 × 365 × 100 = 365.000 mogelijke combinaties over honderd jaar geboortedata, voor maar 54.805 kiezers. In Oefening 1 meet je hetzelfde op een verzonnen kiezerslijst.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Twee tabellen, één sleutel

        Sweeney (2002) beschrijft een gangbare praktijk: gegevens over mensen worden vrijgegeven zonder naam, adres en telefoonnummer, in de veronderstelling dat ze anoniem blijven omdat ze er anoniem uitzien. In Massachusetts kocht de Group Insurance Commission (GIC) de zorgverzekering in voor ambtenaren van de staat. De gegevens van de GIC bevatten bijna honderd velden per bezoek, voor ongeveer 135.000 ambtenaren en hun gezinnen, en omdat men dacht dat ze anoniem waren, gaf de GIC een kopie aan onderzoekers en verkocht een kopie aan bedrijven. Sweeney kocht voor twintig dollar de kiezerslijst van Cambridge, Massachusetts, en kreeg die op twee diskettes. De lijst bevatte van elke kiezer de naam, het adres, de postcode, de geboortedatum en het geslacht, en via postcode, geboortedatum en geslacht kon je hem koppelen aan de medische gegevens.

        De toenmalige gouverneur van Massachusetts, William Weld, woonde in Cambridge, en de medische gegevens van Weld zaten in de data van de GIC. Volgens de kiezerslijst hadden zes mensen de geboortedatum van Weld, van wie drie mannen, en daarvan woonde er maar één in de postcode van Weld.

        Dat is een **linkage attack**: je koppelt een vrijgegeven tabel aan een tweede tabel die namen heeft en een quasi-identifier deelt, hier postcode, geboortedatum en geslacht. In pandas is het een merge op alle drie: `pd.merge(released, voters, on=["zip", "birth", "sex"])`. Een rij past alleen als alle drie de kolommen gelijk zijn. Een vrijgegeven rij die bij precies één kiezer past, krijgt de naam van die kiezer; een rij die bij twee kiezers past, komt er twee keer uit, met twee kandidaten; een rij die bij geen enkele past, valt weg, zoals bij elke inner join.
        """),
        mo.Html(C.link_picture("nl")),
        mo.md(r"""
        ### Waarom zoveel mensen alleen staan

        Sweeney (2000) legt uit waarom de kiezerslijst zoveel verraadde: op de lijst waren er 2 mogelijke waarden voor het geslacht en 5 postcodes, en de geboortedata lagen verspreid over 365 dagen in 100 jaar. Dat geeft 365.000 mogelijke combinaties, voor maar 54.805 kiezers. Met veel meer combinaties dan mensen hebben veel mensen een combinatie voor zich alleen: ze zijn **uniek**, alleen op die kolommen. Samarati en Sweeney (1998) geven, met een verwijzing naar Sweeney, de getallen voor de lijst: van de 54.805 kiezers was 12% uniek op geboortedatum alleen, 29% op geboortedatum en geslacht, 69% op geboortedatum en postcode, en 97% was herkenbaar aan de volledige postcode en geboortedatum.

        Voor de hele Verenigde Staten schatte Sweeney (2000) dat 87% van de bevolking (216 miljoen van de 248 miljoen) waarschijnlijk uniek was op alleen postcode, geslacht en geboortedatum. De schatting komt uit de tellingen van de volkstelling van 1990, niet uit namen: per postcode geeft de volkstelling het aantal mensen in elke leeftijdsgroep, en waar een groep niet meer mensen heeft dan er combinaties van geslacht en geboortedatum in zitten, telt Sweeney ze allemaal als mensen die waarschijnlijk alleen staan. Philippe Golle (2006) telde opnieuw met de volkstelling, in de veronderstelling dat geboortes gelijk over de dagen van het jaar verdeeld zijn, en kwam op 61% voor 1990 en 63% voor 2000. Beide getallen tellen mensen die waarschijnlijk alleen staan, niet mensen die een naam kregen; de publicatie van Sweeney waarschuwt zelf dat weten dat een of enkele mensen bepaalde waarden delen niet precies hetzelfde is als zeggen wie dat zijn.

        Met een grovere geboortedatum staat bijna niemand meer alleen. Met postcode, geslacht en alleen maand en jaar van geboorte geeft de tabel van Sweeney 3,7% waarschijnlijk uniek; met alleen het jaar 0,04%. De tabel van Golle voor 2000 geeft 4,2% en 0,2%.

        ### Een naam uit een onvolledige lijst

        Daniel Barth-Jones (2012) bekeek het geval opnieuw. De kiezerslijst bevatte 54.805 mensen, terwijl Cambridge in 1996 en 1997 ongeveer 99.435 inwoners had, volgens de schatting van Barth-Jones uit de volkstellingen van 1990 en 2000: de lijst bevatte 55% van de stad. Van de ongeveer 88.000 mensen die oud genoeg waren om te stemmen, stonden er zo'n 31.000 (35%) niet op. Veel mensen die op de kiezerslijst uniek leken, zouden dus niet echt de enige in Cambridge zijn geweest met hun geboortedatum, geslacht en postcode. Barth-Jones noemt de aanname dat een lijst iedereen bevat de mythe van het perfecte bevolkingsregister: een kiezerslijst is geen **bevolkingsregister**, een lijst van iedereen die ergens woont.

        De volkstelling van 2000 telde 174 mannen van 50 in de postcode van Weld. Met de aanname dat dat aantal in 1997 hetzelfde was, en dat verjaardagen gelijk over het jaar verdeeld zijn, zoals Golle aannam, komt Barth-Jones op een kans van maar 62% dat Weld de enige van hen met die verjaardag was. Een korte regel geeft hetzelfde getal: elk van de 173 andere mannen is met kans 364/365 op een andere dag jarig. De verjaardag van de een hangt niet af van die van een ander, dus de kans dat alle 173 op een andere dag jarig zijn, is het product van hun kansen: 364/365, 173 keer met zichzelf vermenigvuldigd, (364/365)<sup>173</sup>, ongeveer 0,62. Met 150 mannen, het kleinste aantal dat Barth-Jones aannemelijk vindt, geeft de regel 66%. Die regel schrijf je in Oefening 3.

        Barth-Jones stelt dat Weld waarschijnlijk alleen herkenbaar was omdat Weld een publiek figuur was van wie de ziekenhuisopname in het nieuws kwam: Weld zakte in mei 1996 in elkaar tijdens een diploma-uitreiking op een hogeschool. En over de kiezerslijst: het koppelen aan die lijst was voor het herkennen van Weld overbodig, niet nodig en, belangrijker nog, niet genoeg. Dat maakt gegevens zonder namen niet veilig, en dat zegt Barth-Jones ook niet: wie alleen de direct identificerende gegevens zoals namen en adressen weghaalt, moet blijven letten op het risico van heridentificatie.
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
        | **linkage attack** | namen zetten op een vrijgegeven tabel door die te koppelen aan een tweede tabel die namen heeft en een quasi-identifier deelt, zoals een kiezerslijst |
        | **uniek** | de enige rij met zijn waarden in bepaalde kolommen: alleen op die kolommen |
        | **bevolkingsregister** | een lijst van iedereen die ergens woont; een kiezerslijst is dat niet, want niet iedereen staat erop |
        | **`on=[...]`** | een merge op meerdere kolommen: een rij past alleen als ze allemaal gelijk zijn, zoals in `pd.merge(visits, voters, on=["zip", "birth", "sex"])` |
        | **(364/365)<sup>n−1</sup>** | de kans dat geen van n − 1 andere mensen op jouw verjaardag jarig is, als verjaardagen gelijk over 365 dagen verdeeld zijn |
        """),
        L.watch_out("Uniek is nog niet herkend: een rij die alleen staat in een tabel krijgt pas een naam als een tweede tabel met namen dezelfde waarden bevat. En één match in een lijst die niet iedereen bevat, geeft een naam die waarschijnlijk klopt, maar niet zeker. Twee details van het verhaal over Weld raken vaak door elkaar: 54.805 is het aantal kiezers, niet het aantal inwoners; en het verhaal dat Sweeney de medische gegevens van de gouverneur naar diens kantoor stuurde, staat niet in de publicatie van Sweeney uit 2002: het komt uit een navertelling van Paul Ohm (2010), die er een andere auteur voor aanhaalt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan op hoeveel rijen van een vrijgegeven tabel iemand een naam kan zetten:

    1. Zoek de kolommen die de vrijgegeven tabel deelt met een tabel die namen heeft, zoals postcode, geboortedatum en geslacht.
    2. Meet hoeveel mensen alleen staan op die kolommen: dat zijn de mensen die een match eruit kan pikken. Dat is `share_alone`.
    3. Doe een merge op al die kolommen, en tel per vrijgegeven rij hoeveel rijen met een naam erbij passen: een rij met precies één krijgt een naam. Dat is `linked`.
    4. Vraag je af, voordat je zo'n naam vertrouwt, of de tabel met namen iedereen bevat. Zo niet, reken dan de kans uit dat de match alleen staat in de hele bevolking. Dat is `alone`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course verzon een stadje van 30.000 volwassenen, met een kiezerslijst en de vrijgegeven tabel van een zorgverzekeraar: niemand erin bestaat. De cel leest beide tabellen in, doet een merge op postcode, geboortedatum en geslacht, en telt wat eruit komt.
    """)
    return


@app.cell
def example_link():
    import pandas as pd

    _visits = pd.read_csv("data/course-town-visits.csv.gz")
    _voters = pd.read_csv("data/course-town-voters.csv.gz")
    print(len(_visits), "visits and", len(_voters), "voters")
    # elk bezoek naast elke kiezer met dezelfde postcode, geboortedatum en hetzelfde geslacht
    _both = pd.merge(_visits, _voters, on=["zip", "birth", "sex"])
    print(_both.head(3))
    print("rows after the merge:", len(_both))
    print("visits with a match:", _both["visit"].nunique())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vrijgegeven tabel heeft geen namen, en toch zetten de eerste rijen van de merge een diagnose naast een naam. De merge heeft 805 rijen voor 711 bezoeken: 289 van de 1000 bezoeken passen bij geen enkele kiezer en vallen weg, en sommige bezoeken passen bij meer dan één kiezer, dus die komen er vaker uit. Welke van de 711 precies één naam krijgen, tel je in Oefening 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. share_alone: hoeveel mensen alleen staan

        `VOTERS` is de kiezerslijst van het verzonnen stadje: 19.656 rijen met de kolommen `name`, `zip`, `birth` (een datum als tekst, zoals `"1961-04-12"`) en `sex`. `PEOPLE` is het hele stadje zoals alleen de course het kent: 30.000 mensen, met dezelfde kolommen en nog twee, `voter` (`True` voor de mensen op de kiezerslijst) en `visit` (de id van hun bezoek in de vrijgegeven tabel, of leeg). `QI` is de lijst `["zip", "birth", "sex"]`.

        `share_alone(table, columns)` krijgt een tabel `table` en een lijst kolomnamen `columns`. Hij geeft het aandeel terug van de rijen die de enige zijn met hun waarden in `columns`: een getal van 0 tot 1. Vervang de `...` na `sizes =` door de grootte van elke groep rijen met dezelfde waarden in `columns`, en de `...` na `return` door het aandeel van de rijen waarvan de groep één rij heeft. De check roept `share_alone(VOTERS, QI)` aan, daarna andere kolommen, `PEOPLE`, en de kleine kiezerslijst `FEW_VOTERS` van het plaatje in *Het idee*.
        """),
        L.task("Geef het aandeel terug van de rijen die de enige zijn met hun waarden in columns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share_alone = L.code_box("share_alone", lines=4)
    box_share_alone
    return (box_share_alone,)


@app.cell(hide_code=True)
def check_share(box_share_alone):
    result_share_alone = L.check("share_alone", box_share_alone.value)
    result_share_alone
    return (result_share_alone,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share_alone",
        [
            "Stap 2 van het recept. De grootte van de groepen ken je van `k_of`; nu tel je de groepen met grootte 1, en maak je van die telling een aandeel van alle rijen.",
            "`sizes == 1` geeft `True` voor elke groep van één rij, en `.sum()` telt de `True`-waarden, zoals in [Vragen stellen aan data](/course/open/data-explore?lang=nl). Welke functie geeft het aantal rijen van `table`?",
            "`sizes = table.groupby(columns).size()` en `return (sizes == 1).sum() / ___`",
        ],
        why="`sizes = table.groupby(columns).size()` geeft de grootte van elke groep, en `return (sizes == 1).sum() / len(table)` het aandeel van de rijen dat alleen staat. `share_alone(VOTERS, QI)` geeft 0,86: 86% van de kiezers is de enige op de lijst met hun postcode, geboortedatum en geslacht.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_linked():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. linked: de rijen die een naam krijgen

        `VISITS` is de vrijgegeven tabel van een verzonnen zorgverzekeraar in hetzelfde stadje: 1000 rijen met de kolommen `visit` (een id van `"V0001"` tot `"V1000"`), `zip`, `birth`, `sex` en `diagnosis`, en geen namen.

        `linked(released, voters, columns)` krijgt een vrijgegeven tabel `released` met een kolom `visit`, een kiezerslijst `voters`, en een lijst kolomnamen `columns`. Hij geeft terug hoeveel rijen van `released` op alle `columns` bij precies één rij van `voters` passen: de rijen die een naam krijgen. Vervang de drie `...`: `both` is de merge van `released` en `voters` op `columns`, `per_visit` het aantal rijen van elk bezoek in `both`, en de laatste regel telt de bezoeken met precies één. De check roept `linked(VISITS, VOTERS, QI)` aan, daarna minder kolommen, en de kleine tabellen `FEW_VISITS` en `FEW_VOTERS` van het plaatje in *Het idee*.
        """),
        L.task("Geef terug hoeveel rijen van released op columns bij precies één rij van voters passen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_linked():
    box_linked = L.code_box("linked", lines=8)
    box_linked
    return (box_linked,)


@app.cell(hide_code=True)
def check_linked(box_linked):
    result_linked = L.check("linked", box_linked.value)
    result_linked
    return (result_linked,)


@app.cell(hide_code=True)
def hints_linked():
    L.hints(
        "linked",
        [
            "Stap 3 van het recept. Een bezoek waar twee kiezers bij passen, staat twee keer in de merge, dus tel de rijen van elk bezoek, en dan de bezoeken met precies één rij.",
            "`pd.merge` neemt met `on=` een lijst kolommen, zoals in het uitgewerkte voorbeeld. Voor de telling per bezoek groepeer je `both` op een van zijn kolommen en neem je `.size()`; `.sum()` van een vergelijking telt de `True`-waarden.",
            "`both = pd.merge(released, voters, on=columns)`, `per_visit = both.groupby(___).size()` en `return (per_visit == 1).sum()`",
        ],
        why="`both = pd.merge(released, voters, on=columns)` zet elk bezoek naast elke kiezer die past, `per_visit = both.groupby(\"visit\").size()` telt ze per bezoek, en `return (per_visit == 1).sum()` telt de bezoeken met precies één. `linked(VISITS, VOTERS, QI)` geeft 622: 622 van de 1000 bezoeken krijgen een naam.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_alone():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. alone: de kans dat een match de enige is

        `alone(n, values)` krijgt een geheel getal `n`: een aantal mensen dat alles gemeen heeft op één ding na, zoals de mannen van 50 in één postcode, die alleen in hun verjaardag verschillen. `values` is het aantal waarden dat dat ene ding kan hebben, allemaal even waarschijnlijk, zoals 365 verjaardagen. Hij geeft de kans terug dat een van de n mensen als enige een bepaalde waarde heeft: dat geen van de andere n − 1 die waarde ook heeft. `miss` is de kans dat één ander een andere waarde heeft dan jij. Vervang de eerste `...` door die kans, en de tweede door de kans dat alle n − 1 anderen missen. De check roept `alone(174, 365)` aan, daarna andere getallen.
        """),
        L.task("Geef de kans terug dat geen van de andere n − 1 mensen jouw waarde heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_alone():
    box_alone = L.code_box("alone", lines=4)
    box_alone
    return (box_alone,)


@app.cell(hide_code=True)
def check_alone(box_alone):
    result_alone = L.check("alone", box_alone.value)
    result_alone
    return (result_alone,)


@app.cell(hide_code=True)
def hints_alone():
    L.hints(
        "alone",
        [
            "Stap 4 van het recept, en de regel in *Een naam uit een onvolledige lijst*: met 365 verjaardagen mist één ander de jouwe met kans 364/365.",
            "Kansen die allemaal moeten uitkomen, vermenigvuldig je: `miss` keer zichzelf, één keer voor elke andere persoon. `**` is tot de macht, zoals in [Variabelen en getallen](/course/open/py-variables?lang=nl).",
            "`miss = (values - 1) / values` en `return miss ** ___`",
        ],
        why="`miss = (values - 1) / values` is de kans dat één ander jouw waarde mist, en `return miss ** (n - 1)` de kans dat alle n − 1 dat doen. `alone(174, 365)` geeft 0,62: de 62% van Barth-Jones.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_voter_list():
    mo.md(r"""
    ### Oefening 4. Wat de lijst liet zien

    Op de kiezerslijst van Cambridge had maar één man de geboortedatum en postcode van de gouverneur: Weld.
    """)
    return


@app.cell(hide_code=True)
def choice_voter_list_cell():
    choice_voter_list = L.choice(
        "voter-list",
        question="Wat liet dat zien?",
        options={
            "sure": "Dat de passende gegevens zeker van Weld waren",
            "right": "Alleen dat geen andere kiezer erbij paste",
            "nothing": "Niets: op een kiezerslijst staat geen geboortedatum",
        },
        correct="right",
        why={
            "right": "De lijst bevatte 54.805 van de ongeveer 99.435 mensen in Cambridge, volgens de schatting van Barth-Jones, dus een man van 50 met dezelfde verjaardag die er niet op stond, zag je niet. Met 174 mannen van 50 in die postcode geeft `alone(174, 365)` 0,62: de kans van Barth-Jones dat Weld de enige was.",
            "sure": "De lijst was niet de hele stad: van de ongeveer 88.000 mensen die oud genoeg waren om te stemmen, stonden er zo'n 31.000 niet op. Iemand met dezelfde waarden kon op de lijst ontbreken. Zie *Een naam uit een onvolledige lijst* in *Het idee*.",
            "nothing": "De lijst van Sweeney bevatte van elke kiezer de naam, het adres, de postcode, de geboortedatum en het geslacht. De vraag is wat één match op die lijst laat zien. Zie *Een naam uit een onvolledige lijst* in *Het idee*.",
        },
    )
    choice_voter_list
    return (choice_voter_list,)


@app.cell(hide_code=True)
def text_exercise_estimate():
    mo.md(r"""
    ### Oefening 5. Wat 87% telt

    Sweeney (2000) schatte dat 87% van de mensen in de VS in 1990 waarschijnlijk als enige hun postcode, geslacht en geboortedatum had. Golle (2006) telde opnieuw voor 1990 en kwam op 61%.
    """)
    return


@app.cell(hide_code=True)
def choice_estimate_cell():
    choice_estimate = L.choice(
        "estimate",
        question="Wat tellen deze twee getallen?",
        options={
            "named": "Mensen van wie de gegevens een naam kregen",
            "right": "Mensen die waarschijnlijk uniek zijn",
            "voters": "Kiezers van wie de naam op een openbare lijst stond",
        },
        correct="right",
        why={
            "right": "Allebei zijn het schattingen, uit tellingen van de volkstelling per postcode en leeftijd, van hoeveel mensen waarschijnlijk alleen staan op die waarden. Niemand kreeg een naam. Wie alleen staat, krijgt pas een naam als een tweede tabel met namen dezelfde waarden bevat.",
            "named": "Niemand kreeg een naam: beide getallen zijn schattingen uit de volkstelling van hoeveel mensen waarschijnlijk alleen staan. Voor een naam is een tweede tabel nodig met dezelfde waarden. Zie *Waarom zoveel mensen alleen staan* in *Het idee*.",
            "voters": "De getallen gaan over de hele bevolking in de volkstelling, niet over een kiezerslijst, en ze tellen mensen die waarschijnlijk alleen staan, geen namen. Zie *Waarom zoveel mensen alleen staan* in *Het idee*.",
        },
    )
    choice_estimate
    return (choice_estimate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Verstoppen in een groep: k-anonimiteit](/course/open/rai-k-anonymity?lang=nl) maakte een quasi-identifier de groepen; hier is hij de sleutel van een merge, en elke rij die erop alleen staat, kan een naam krijgen uit een tweede tabel. Een grovere geboortedatum maakt de groepen groter en de bezoeken met precies één match minder, zoals het grover maken bij k-anonimiteit deed; een lijst die niet iedereen bevat, maakt een deel van de overgebleven namen fout. Dit alles werkt op de gegevens nadat ze zijn vrijgegeven. In de volgende les, *Antwoorden met een munt*, komt de bescherming eerder: elke persoon voegt toeval toe aan het eigen antwoord voordat het wordt verzameld.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_estimate, choice_voter_list, result_alone, result_linked, result_share_alone):
    L.wrap_up(result_share_alone, result_linked, result_alone, choice_voter_list, choice_estimate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
