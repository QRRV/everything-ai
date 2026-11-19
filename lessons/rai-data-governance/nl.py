# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-data-governance. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-data-governance", "nl")
    C = L.checks   # de data catalog en de medewerkers van een verzonnen bibliotheek


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Privacy by design](/course/open/rai-privacy-by-design?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kept():
    quick_kept = L.choice(
        "quick-kept",
        question="In de verzonnen bibliotheek geeft `values_kept(MEMBERS, PURPOSES[\"recommend\"])` de waarde `7`. Wat telt die 7?",
        options={
            "members": "De leden die aanbevelingen willen",
            "right": "De bewaarde waarden van de leden die blijven",
            "keys": "De keys die het aanbevelingsmodel nodig heeft",
        },
        correct="right",
        why={
            "right": "Voor elk van de 7 leden die blijven, bewaart de pipeline één waarde voor het aanbevelingsmodel: de soorten boeken die dat lid leende. 7 leden, elk 1 waarde.",
            "members": "Niemand heeft de leden gevraagd of ze aanbevelingen willen. De 7 telt waarden: per lid dat blijft de keys die `minimize` bewaart. Zie Oefening 3 in [Privacy by design](/course/open/rai-privacy-by-design?lang=nl).",
            "keys": "Het aanbevelingsmodel heeft één key nodig, `\"books\"`. De 7 is die ene key voor elk van de 7 leden die blijven. Zie Oefening 3 in [Privacy by design](/course/open/rai-privacy-by-design?lang=nl).",
        },
    )
    quick_kept
    return (quick_kept,)


@app.cell(hide_code=True)
def quick_check_left():
    quick_left = L.choice(
        "quick-left",
        question="Een lid vertrok in 2019 bij de bibliotheek. Wat vraagt opslagbeperking van de gegevens over dat lid?",
        options={
            "keep": "Bewaren voor het geval het lid terugkomt",
            "right": "Wissen zodra ze niet meer nodig zijn",
            "filter": "Bewaren, met de naam verborgen door een filter",
        },
        correct="right",
        why={
            "right": "Opslagbeperking: gegevens worden niet langer bewaard in een vorm die de betrokkenen identificeert dan nodig is. Met het lidmaatschap is ook het doel voorbij: de gegevens gaan weg, of worden geanonimiseerd zodat niemand de persoon nog kan terugvinden.",
            "keep": "Dat het lid misschien terugkomt, is geen doel: het lidmaatschap is voorbij. Zie *Wissen als het klaar is* in [Privacy by design](/course/open/rai-privacy-by-design?lang=nl).",
            "filter": "Als je de naam verbergt, blijft de rest van het record staan, en die kan nog steeds naar het lid wijzen. Zie *Wissen als het klaar is* in [Privacy by design](/course/open/rai-privacy-by-design?lang=nl).",
        },
    )
    quick_left
    return (quick_left,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Joris van Dam, lid van de verzonnen bibliotheek uit [Privacy by design](/course/open/rai-privacy-by-design?lang=nl), zegt het lidmaatschap op en vraagt de bibliotheek alle gegevens over Joris te wissen. De balie kan de rij in het ledenbestand wissen. Maar de bibliotheek heeft meer bestanden: de uitleningen, de data waarop het aanbevelingsmodel getraind is, het model zelf, een lijst voor de nieuwsbrief, een back-up van oktober 2026 en een enquête uit 2024 in een gedeelde map. Welke daarvan bevatten gegevens van Joris van Dam? Wie beslist per bestand of de gegevens weg moeten? En wie mag ze intussen lezen? Zonder antwoorden bereikt een verzoek om te wissen het bestand waar iemand aan denkt, en mist het de rest. In deze les geef je de bibliotheek die antwoorden in code: wie welke dataset mag lezen, waar iemands gegevens staan, en waar niemand de leiding heeft.

    **Een voorbeeld uit het dagelijks leven.** Je zegt je lidmaatschap van de sportclub op en vraagt de club je gegevens te wissen. De secretaris haalt je van de ledenlijst. Maar je naam staat nog in het toernooischema van vorig jaar op iemands laptop en in het mailprogramma van de club, en niemand bij de club weet wie daarover moet beslissen.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Onderzoekers verdeelden de training data van een model in 20 delen en trainden één model per deel, zodat je voor het vergeten van de gegevens van één persoon maar één deel opnieuw hoeft te trainen. Hoeveel sneller was dat bij 8 van zulke verzoeken op één dataset dan helemaal opnieuw trainen?",
        options={
            "few": "ongeveer 1,5 keer",
            "right": "ongeveer 4,6 keer",
            "many": "ongeveer 20 keer",
        },
        reveal="**4,63 keer** zo snel, op de dataset *Purchase*, met elk deel weer verdeeld in 50 stukken, volgens Bourtoule e.a., die de methode SISA noemen. Gratis is het niet: *En het model?* in *Het idee* zegt wat het kost, en waarom een model het überhaupt nodig heeft.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Iemand moet beslissen

    In [Privacy by design](/course/open/rai-privacy-by-design?lang=nl) besliste je per doel wat de pipeline bewaart en wanneer gegevens weggaan. In een organisatie moet iemand die beslissingen nemen, voor elke dataset, en ze blijven nemen. Dat is **data governance**. Een definitie in de woordenlijst van NIST, het Amerikaanse instituut voor standaarden, noemt het een reeks processen die ervoor zorgt dat data overal in een organisatie formeel beheerd worden. Het gaat niet om wie de data toevallig heeft. De richtlijn van de Britse overheid over data-eigenaarschap zegt dat eigenaarschap niet over het bezit van data gaat, maar over het vastleggen van de rollen van de mensen die verantwoordelijk zijn voor het beheer van data, gedurende de hele levensloop van die data.

    ### Eigenaar, steward, custodian

    De Britse richtlijn noemt drie rollen rond een dataset. De **eigenaar** (*data owner*) is iemand hoog in de organisatie met eigen verantwoordelijkheden voor data. Een eigenaar hoeft de data niet tot in detail te kennen, maar is onder meer aanspreekbaar op een bewaarschema en op het beperken van de toegang tot wie daarvoor bevoegd is. De steward (*data steward*) weet wat er in de dataset zit en regelt het dagelijkse werk, ter ondersteuning van de beslissingen van de eigenaar. En de custodian (*data custodian*) zorgt voor het systeem waar de data in staan: die legt ze vast, slaat ze op en verwijdert ze, zoals de eigenaar dat vraagt. De eigenaar beslist en legt er verantwoording over af, de steward beheert de dataset van dag tot dag: de richtlijn noemt de een *accountable* en de ander *responsible*. Die namen zijn geen vaste standaard, en andere bronnen verdelen het werk anders; deze les volgt de Britse richtlijn. Die raadt ook aan om eens per jaar na te gaan wie waarvan eigenaar is, en verder telkens als rollen of verantwoordelijkheden flink veranderen.

    ### Wie wat mag zien

    **Toegangsbeheer** is volgens een definitie van NIST de beslissing om iemand wel of geen toegang te geven tot de onderdelen van een systeem, zoals een netwerk, data, een applicatie of een dienst. Welke toegang je geeft, volgt uit het principe van *least privilege*: een systeem hoort de rechten van gebruikers, en van programma's die namens hen werken, te beperken tot het minimum dat nodig is voor hun taken. *Least privilege* doet voor toegang wat minimale gegevensverwerking doet voor het verzamelen. En het is niet iets wat je één keer instelt: de lijst van beveiligingsmaatregelen van NIST vraagt om de toegekende rechten geregeld na te lopen, om te zien of de reden voor elk recht nog geldt. In de verzonnen bibliotheek gaat toegang per rol: de balie, het uitlenen, het datateam, communicatie en IT mogen elk lezen wat hun taken vragen, en een persoon mag lezen wat een van de eigen rollen toestaat.

    ### Een overzicht van wat er is

    Niemand kan zorgen voor data waar niemand van weet. Een **data catalog** zet de datasets van een organisatie op een rij, met per dataset wie ervoor zorgt en wie hem mag gebruiken. Kopieën tellen hier ook als dataset: een back-up, een export in een gedeelde map, een model dat op de data getraind is. De datasheet uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) gaat dieper in op één dataset. Gebru e.a., die datasheets voorstelden, vragen onder meer wie de dataset ondersteunt, host en onderhoudt, en, bij een dataset over mensen, of er grenzen gelden aan hoe lang de gegevens bewaard worden.

    ### Een verzoek om te wissen

    In [De AVG voor AI](/course/open/rai-gdpr?lang=nl) leerde je het recht op gegevenswissing kennen: op een van zes gronden kan iemand de eigen persoonsgegevens "zonder onredelijke vertraging" laten wissen, met uitzonderingen. Artikel 12 legt de termijn vast: de verwerkingsverantwoordelijke laat de betrokkene "onverwijld en in ieder geval binnen één maand na ontvangst van het verzoek" weten wat er met het verzoek gedaan is. "Afhankelijk van de complexiteit van de verzoeken en van het aantal verzoeken kan die termijn indien nodig met nog eens twee maanden worden verlengd", en de betrokkene moet dat binnen die eerste maand horen, "met vermelding van de redenen voor de vertraging". Hooguit drie maanden in totaal dus. Om op tijd te antwoorden, moet de bibliotheek weten waar de gegevens staan, en wie per kopie beslist of ze weggaan of, onder een van de uitzonderingen, blijven. Een data catalog met eigenaren geeft de bibliotheek allebei.

    ### En het model?

    Het model `recommender` is getraind op de rijen van de leden die blijven. Een rij uit de training data wissen verandert niets aan een model dat al getraind is. Het advies van de EDPB over AI-modellen, dat je in [De AVG voor AI](/course/open/rai-gdpr?lang=nl) tegenkwam, zegt dat informatie uit de training data, persoonsgegevens inbegrepen, opgenomen kan blijven in de parameters van het model. Het concludeert dat AI-modellen die op persoonsgegevens getraind zijn niet in alle gevallen als anoniem kunnen gelden: of een model anoniem is, hoort per geval beoordeeld te worden. De zekere manier om te vergeten is opnieuw trainen zonder de data, wat Bourtoule e.a. een naïeve manier noemen: het werkt, maar het kost elke keer een hele training. Hun SISA-training (*Sharded, Isolated, Sliced, and Aggregated*) verdeelt de training data in delen, *shards*, traint één model per shard en combineert hun antwoorden, bijvoorbeeld door ze te laten stemmen: het label dat de meeste modellen voorspellen, wint. Een verzoek om te vergeten betekent dan dat je alleen het model van die ene shard opnieuw traint. Met 20 shards, elk weer verdeeld in 50 stukken (*slices*), ging dat 4,63 keer zo snel als helemaal opnieuw trainen bij 8 verzoeken op de dataset *Purchase*, en 2,45 keer zo snel bij 18 verzoeken op *SVHN*, terwijl de accuracy minder dan 2 procentpunt daalde. Het kost extra opslag, het scheelt alleen tijd zolang er minder verzoeken zijn dan drie keer het aantal shards, en op *ImageNet* ging het 1,36 keer zo snel, met 19,45 procentpunt minder top-5-accuracy. Een naam verbergen in de antwoorden van het model is niet hetzelfde: vergeten vraagt volgens hen dat een trainingspunt helemaal niets meer bijdraagt aan het model.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **data governance** | de processen en rollen die voor elke dataset bepalen wie ervoor aanspreekbaar is, wie hem mag gebruiken en hoe lang hij bewaard wordt |
        | **toegangsbeheer** | bepalen en afdwingen wie welke data en systemen mag lezen of veranderen |
        | **data catalog** | een overzicht van de datasets van een organisatie, met per dataset wie ervoor zorgt en wie hem mag gebruiken |
        """),
        L.watch_out("De eigenaar van een dataset is niet wie hem toevallig in handen heeft: in de Britse richtlijn is opslaan en verwijderen het werk van de custodian, en is de eigenaar iemand hoog in de organisatie die beslist. En rijen uit de training data wissen laat een getraind model ze niet vergeten."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te weten wie voor de data zorgt, en om een verzoek om te wissen te beantwoorden:

    1. Zet elke dataset in een data catalog, kopieën inbegrepen, elk met een eigenaar, een steward en een custodian, en de mensen van wie hij gegevens bevat.
    2. Geef elke rol toegang tot wat de taken vragen, en niet meer, en loop dat geregeld na. Dat is `may_read`.
    3. Als iemand vraagt de eigen gegevens te wissen, zoek dan elke dataset die ze bevat. Dat is `holding`.
    4. Zoek de gaten: datasets die de gegevens bevatten en waar niemand eigenaar van is. Dat is `no_owner`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel print de data catalog van de bibliotheek: per dataset de eigenaar, de steward en de custodian, van hoeveel leden hij gegevens bevat en welke rollen hem mogen lezen. `C.CATALOG` is een lijst van acht dicts; `None` betekent dat niemand die rol heeft. Daarna telt de cel de datasets waar niemand eigenaar van is.
    """)
    return


@app.cell
def example_catalog():
    # de data catalog van de verzonnen bibliotheek: elke dataset met wie ervoor zorgt
    for _d in C.CATALOG:
        print(f'{_d["name"]}: owner {_d["owner"]}, steward {_d["steward"]}, custodian {_d["custodian"]}')
        print("    cards:", len(_d["cards"]), "| may read:", _d["access"])

    # hoeveel datasets geen eigenaar hebben
    _count = 0
    for _d in C.CATALOG:
        if _d["owner"] is None:
            _count = _count + 1
    print("datasets without an owner:", _count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zes datasets hebben een eigenaar, een steward en een custodian. Twee hebben alleen een custodian: de back-up van oktober 2026 en de enquête van 2024. Allebei bevatten ze gegevens van leden, van tien en van drie leden, en niemand beslist wat ermee gebeurt. De openingstijden bevatten geen gegevens van leden, en elke rol mag ze lezen. Bij het model `recommender` noemt de data catalog de kaarten van de zeven leden waarop het getraind is: hun gegevens kunnen nog in de parameters zitten, en daar gaat het de EDPB om in *En het model?*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. may_read: wie wat mag zien

        `may_read(person, dataset)` krijgt een medewerker als dict `person`, met een `"name"` en een lijst `"roles"`, en een dataset uit de catalog als dict `dataset`, met in de lijst `"access"` de rollen die hem mogen lezen. De functie geeft `True` terug als minstens één van de rollen van de persoon in die lijst staat, en anders `False`. `STAFF` is de lijst van de medewerkers van de bibliotheek; `STAFF[0]` is Noor Jansen, van de balie. De loop gaat langs de rollen van de persoon, en de laatste regel geeft `False` terug voor als geen enkele rol paste. Vervang de `...`. De check roept `may_read(STAFF[0], CATALOG[0])` aan, en daarna andere mensen en datasets.
        """),
        L.task("Geef True terug zodra een van de rollen van de persoon de dataset mag lezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("may-read", lines=6)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("may-read", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "may-read",
        [
            "Stap 2 van het recept. `return` beëindigt een functie meteen, ook midden in een loop.",
            "Zodra één rol van de persoon de dataset mag lezen, is het antwoord `True`, en kan de functie stoppen. Pas als de loop alle rollen heeft gehad zonder dat er een paste, is het antwoord `False`.",
            '`if ___ in dataset["access"]:` en daaronder `return True`',
        ],
        why='`if role in dataset["access"]:` met daaronder `return True` geeft `True` bij de eerste rol die de dataset mag lezen; na de loop geeft `return False` de rest. `may_read(STAFF[0], CATALOG[0])` geeft `True`: Noor Jansen heeft de rol `"desk"`, en de balie mag het ledenbestand lezen.',
    )
    return


@app.cell(hide_code=True)
def text_after_read():
    mo.md(r"""
    Met een goede `may_read` zijn 25 van de 72 vakjes gevuld. Noor Jansen van de balie mag het ledenbestand, de uitleningen en de openingstijden lezen, en niets waar het datateam aan werkt. Daan Smit heeft twee rollen, en mag lezen wat een van beide toestaat. Mees Peters, een nieuwe vrijwilliger zonder rol, mag helemaal niets lezen. En de enquête van 2024, waar niemand eigenaar van is, kan door vier mensen gelezen worden: de toegang is ooit gegeven, en niemand heeft er sindsdien nog naar gekeken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_holding():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. holding: waar iemands gegevens staan

        `holding(catalog, card)` krijgt de catalog, een lijst datasets, en een kaartnummer `card`. De functie geeft de namen terug van de datasets waarvan `"cards"` de kaart `card` bevat, in de volgorde van de catalog: elke plek die een verzoek om te wissen moet bereiken. Binnenin begint `names` als lege lijst, en de loop gaat langs de datasets. Vervang de `...`. De check roept `holding(CATALOG, "L-1004")` aan, met de kaart van Joris van Dam, en daarna andere kaarten en lijsten.
        """),
        L.task("Geef de namen terug van de datasets die de kaart bevatten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_holding():
    box_holding = L.code_box("holding", lines=7)
    box_holding
    return (box_holding,)


@app.cell(hide_code=True)
def check_holding(box_holding):
    result_holding = L.check("holding", box_holding.value)
    result_holding
    return (result_holding,)


@app.cell(hide_code=True)
def hints_holding():
    L.hints(
        "holding",
        [
            'Stap 3 van het recept. `"b" in ["a", "b"]` geeft `True`: `in` vraagt of een lijst een waarde bevat.',
            "Vraag voor elke dataset of de kaart tussen de kaartnummers staat. Zo ja, voeg dan de naam van de dataset toe aan `names`.",
            '`if card in ___:` en daaronder `names.append(dataset["name"])`',
        ],
        why='`if card in dataset["cards"]:` met daaronder `names.append(dataset["name"])` verzamelt elke dataset die de kaart bevat. `holding(CATALOG, "L-1004")` geeft `["members", "loans", "recommend-train", "recommender", "backup-2026-10", "survey-2024"]`: zes van de acht datasets.',
    )
    return


@app.cell(hide_code=True)
def text_after_holding():
    mo.md(r"""
    De data catalog noemt de kaart van Joris van Dam bij zes van de acht datasets: niet alleen bij het ledenbestand en de uitleningen, maar ook bij de training data, de back-up, de enquête en het model, waarvan de parameters de gegevens nog kunnen bevatten. En Nora Willems, die in juni 2025 vertrok, staat nog in twee datasets: de back-up van oktober 2026 is gemaakt voordat de pipeline uit [Privacy by design](/course/open/rai-privacy-by-design?lang=nl) de vertrokken leden wiste, en de enquête van 2024 hoorde nooit bij die pipeline. Een verzoek om te wissen dat alleen het ledenbestand bereikt, mist die kopieën.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_owner():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. no_owner: waar niemand beslist

        `no_owner(catalog, card)` geeft de namen terug van de datasets die `card` bevatten en geen eigenaar hebben: hun `"owner"` is `None`. Daar is niemand aanspreekbaar op het verzoek, en beslist niemand of de gegevens mogen blijven. Binnenin begint `names` als lege lijst, en de loop gaat langs de datasets. Een dataset moet door twee vragen heen, dus je hebt een `if` binnen een andere `if` nodig. Vervang de `...`. De check roept `no_owner(CATALOG, "L-1004")` aan, en daarna andere kaarten en lijsten.
        """),
        L.task("Geef de namen terug van de datasets die de kaart bevatten en geen eigenaar hebben."),
    ])
    return


@app.cell(hide_code=True)
def exercise_owner():
    box_owner = L.code_box("no-owner", lines=8)
    box_owner
    return (box_owner,)


@app.cell(hide_code=True)
def check_owner(box_owner):
    result_owner = L.check("no-owner", box_owner.value)
    result_owner
    return (result_owner,)


@app.cell(hide_code=True)
def hints_owner():
    L.hints(
        "no-owner",
        [
            "Stap 4 van het recept. Een `if` binnen een andere `if` loopt alleen als beide voorwaarden `True` zijn; de binnenste `if` staat vier spaties verder in.",
            "Vraag eerst of de dataset de kaart bevat, zoals in Oefening 2. Vraag pas dan of niemand er eigenaar van is, en voeg zo ja de naam toe aan `names`.",
            '`if card in dataset["cards"]:` en daaronder `if ___ is None:`',
        ],
        why='`if card in dataset["cards"]:`, met daaronder `if dataset["owner"] is None:` en daar weer onder `names.append(dataset["name"])`, bewaart de datasets die de kaart bevatten en geen eigenaar hebben. `no_owner(CATALOG, "L-1004")` geeft `["backup-2026-10", "survey-2024"]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_owner():
    mo.md(r"""
    Twee van de zes datasets met de gegevens van Joris van Dam hebben geen eigenaar: de back-up en de enquête. Kees Mulder beheert ze allebei als custodian, maar een custodian verwijdert data zoals de eigenaar dat vraagt, en hier vraagt geen eigenaar iets. De oplossing is niet meer code: iemand moet eigenaar worden van elk van de twee, en beslissen wat ermee gebeurt. Elk lid van de bibliotheek staat in de back-up, dus voor elk van de tien vindt `no_owner` er minstens die ene.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wiens gegevens, en waar

    Kies een lid. Het plaatje laat zien welke datasets de gegevens van dat lid bevatten, met wie voor elk zorgt: met jouw `holding` zodra Oefening 2 goed is, en tot dan met die van de course. Probeer Sem Brouwer, die in november 2023 vertrok.
    """)
    return


@app.cell(hide_code=True)
def try_member_cell():
    try_member = mo.ui.dropdown(options=C.MEMBER_CARDS, value="Joris van Dam", label="Wiens gegevens?")
    try_member
    return (try_member,)


@app.cell(hide_code=True)
def demo_whose(box_holding, result_holding, try_member):
    _yours = C.mine(box_holding.value, result_holding.state == "correct")
    _out = mo.Html(C.whose_picture(try_member.value, _yours, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_deadline():
    mo.md(r"""
    ### Oefening 4. De termijn

    Joris van Dam vraagt de bibliotheek op 3 maart om alle gegevens over Joris te wissen. Elke kopie vinden kost tijd: de back-up, de enquête, het model.
    """)
    return


@app.cell(hide_code=True)
def choice_deadline_cell():
    choice_deadline = L.choice(
        "deadline",
        question="Wat vraagt de AVG van de bibliotheek?",
        options={
            "one": "Eén maand, en nooit langer dan dat",
            "right": "Eén maand, of drie als het nodig is en gemeld",
            "free": "Geen termijn, als het uiteindelijk maar gebeurt",
        },
        correct="right",
        why={
            "right": "Artikel 12: een antwoord \"in ieder geval binnen één maand na ontvangst van het verzoek\", een termijn die \"indien nodig met nog eens twee maanden\" verlengd kan worden, als Joris van Dam dat binnen die eerste maand hoort, met de redenen. Hooguit drie maanden in totaal.",
            "one": "Artikel 12 staat twee maanden extra toe als de verzoeken complex of talrijk zijn, mits de betrokkene dat binnen de eerste maand hoort, met de redenen. Zie *Een verzoek om te wissen* in *Het idee*.",
            "free": "Artikel 12 geeft één maand, en hooguit twee erbij. Zie *Een verzoek om te wissen* in *Het idee*.",
        },
    )
    choice_deadline
    return (choice_deadline,)


@app.cell(hide_code=True)
def text_exercise_model():
    mo.md(r"""
    ### Oefening 5. En het model?

    De bibliotheek wist de rijen van Joris van Dam uit `recommend-train`. Daarvoor is het model `recommender` op die rijen getraind.
    """)
    return


@app.cell(hide_code=True)
def choice_model_cell():
    choice_model = L.choice(
        "model",
        question="Wat betekent dat voor het model?",
        options={
            "numbers": "Joris zit er niet in: een model bevat alleen getallen",
            "right": "Joris kan er nog in zitten, in de parameters",
            "gone": "Het model vergeet Joris samen met de gewiste rijen",
        },
        correct="right",
        why={
            "right": "Volgens de EDPB kan informatie uit de training data opgenomen blijven in de parameters van het model. Rijen wissen verandert niets aan een model dat al getraind is. Om het zeker te weten, traint de bibliotheek het model opnieuw zonder die rijen, of traint het in shards, zoals SISA, zodat dat minder kost.",
            "numbers": "Getallen kunnen persoonsgegevens bevatten: volgens de EDPB kunnen AI-modellen die op persoonsgegevens getraind zijn niet in alle gevallen als anoniem gelden. Zie *En het model?* in *Het idee*.",
            "gone": "Een getraind model verandert niet als de training data veranderen. Zie *En het model?* in *Het idee*.",
        },
    )
    choice_model
    return (choice_model,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Privacy by design](/course/open/rai-privacy-by-design?lang=nl) besliste je per doel wat een pipeline bewaart en wanneer gegevens weggaan. Hier zag je wie in een organisatie die beslissingen neemt, wie wat mag lezen, en hoe je elke kopie van iemands gegevens vindt, ook de kopieën waar niemand voor zorgt. Dit is de laatste les van de module privacy. De module security begint met *Denken als een aanvaller*, en vraagt wat er gebeurt als iemand met opzet bij de data of het model probeert te komen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_deadline, choice_model, result_holding, result_owner, result_read):
    L.wrap_up(result_read, result_holding, result_owner, choice_deadline, choice_model)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
