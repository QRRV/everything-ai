# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-privacy-by-design. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-privacy-by-design", "nl")
    C = L.checks   # het ledenbestand van een verzonnen bibliotheek, en wat elk doel nodig heeft


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_typo():
    quick_typo = L.choice(
        "quick-typo",
        question="Een filter verbergt negen cijfers als ze door de elfproef komen. Iemand vult een echt BSN in, met één cijfer fout. Wat doet het filter?",
        options={
            "close": "Het verbergt het: één cijfer fout is nog dichtbij",
            "right": "Het laat het nummer staan: de proef mislukt",
            "stop": "Het stopt: het nummer is geen BSN",
        },
        correct="right",
        why={
            "right": "Eén verkeerd cijfer laat de elfproef altijd mislukken, dus het filter ziet geen mogelijk BSN en laat het nummer staan, terwijl het bijna iemands echte nummer is. Een mislukte proef maakt een nummer niet onschuldig.",
            "close": "De elfproef kijkt niet hoe dicht een nummer erbij zit: één verkeerd cijfer laat hem mislukken, en het filter laat het nummer in de tekst staan. Zie *Wat een geslaagde proef zegt* in [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl).",
            "stop": "Het filter vraagt de proef, krijgt `False` en gaat door: het nummer blijft in de tekst staan. Zie *Wat een geslaagde proef zegt* in [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl).",
        },
    )
    quick_typo
    return (quick_typo,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="Op tien berichten verbergt een filter 23 van de 32 persoonsgegevens. Wat zegt dat?",
        options={
            "any": "Het verbergt 72% van de persoonsgegevens in elke tekst",
            "right": "De recall op deze berichten is 0,71875",
            "wrong": "9 dingen die het verborg, waren geen persoonsgegevens",
        },
        correct="right",
        why={
            "right": "23 van de 32 is 0,71875: het deel van de persoonsgegevens in deze berichten dat het filter vindt, de recall. Op andere teksten kan het meer of minder vinden.",
            "any": "Tien verzonnen berichten zeggen weinig over andere teksten: daarop kan het filter meer of minder vinden. Zie Oefening 3 in [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl).",
            "wrong": "De 9 zijn de persoonsgegevens die het filter miste: die blijven in de tekst staan. Wat het per vergissing verborg, tel je apart, bij de andere getallen in de berichten. Zie Oefening 3 in [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een verzonnen bibliotheek wil een model trainen dat boeken aanbeveelt op basis van wat leden leenden. Het ledenbestand bevat voor elk van de tien leden een kaartnummer, een naam, een e-mailadres, een telefoonnummer (één lid gaf er nooit een), een geboortedatum, een postcode, de soorten boeken die ze leenden, of ze de nieuwsbrief willen, en de maand waarin hun lidmaatschap stopte, als het stopte. In [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl) verborg je namen en nummers die al in de tekst stonden, en er glipte altijd iets doorheen. Hier komt de vraag eerst: wat heeft elk doel nodig, en wanneer mogen de gegevens weg? Wat de pipeline nooit binnenhaalt, hoeft geen filter te vinden, en wat gewist is, kan niet lekken. In deze les schrijf je die regels voor de bibliotheek: alleen bewaren wat een doel nodig heeft, de leden laten gaan die vertrokken zijn, en tellen wat er overblijft.

    **Een voorbeeld uit het dagelijks leven.** Een winkel vraagt je geboortedatum en je telefoonnummer om je een boek te verkopen. Twee jaar nadat je je sportschool opzegde, heeft die nog een kopie van je paspoort. In beide gevallen dienen de gegevens geen doel meer, en in beide gevallen kunnen ze nog lekken.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="De Belastingdienst had de dubbele nationaliteit van Nederlanders in januari 2014 moeten wissen: die speelt geen rol bij het beoordelen van een aanvraag voor kinderopvangtoeslag. Van hoeveel mensen stond die in mei 2018 nog in de systemen van de Belastingdienst?",
        options={
            "few": "14 duizend",
            "mid": "140 duizend",
            "right": "1,4 miljoen",
        },
        reveal="**1,4 miljoen**, volgens de Autoriteit Persoonsgegevens (AP): \"Toch bewaarde en gebruikte de Belastingdienst deze gegevens.\" Pas in de zomer van 2020 waren ze er volledig uit verwijderd. *Gegevens die gewist hadden moeten worden* in *Het idee* vertelt wat de AP eraan deed.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Vooraf, niet achteraf

    In [De AVG voor AI](/course/open/rai-gdpr?lang=nl) leerde je doelbinding kennen: gegevens die voor het ene doel verzameld zijn, mogen later niet worden gebruikt voor een doel dat daar niet mee te verenigen is. **Privacy by design** gaat een stap verder: bouw de bescherming in een systeem in voordat het iets verzamelt. Ann Cavoukian, toen *Information and Privacy Commissioner* van Ontario in Canada, legde dat jaren vóór de AVG vast in zeven principes. De kern: privacy by design komt vooraf, niet achteraf. De zeven, vrij vertaald:

    1. vooruitlopen in plaats van reageren: voorkomen in plaats van herstellen;
    2. privacy als standaard;
    3. privacy ingebouwd in het ontwerp;
    4. volledige functionaliteit: alle kanten winnen, niemand hoeft iets in te leveren;
    5. beveiliging van begin tot eind, de hele levensloop van de gegevens lang;
    6. zichtbaarheid en transparantie;
    7. respect voor de privacy van de gebruiker.

    Privacy ingebouwd in het ontwerp betekent volgens Cavoukian dat privacy er niet achteraf aan wordt vastgeschroefd. En Cavoukian schrijft dat het verzamelen van persoonsgegevens tot een strikt minimum beperkt moet blijven.

    ### Wat de wet vraagt

    De AVG heeft een eigen term, en artikel 25 noemt de zeven principes niet. De titel van dat artikel is "Gegevensbescherming door ontwerp en door standaardinstellingen", en het vraagt om maatregelen, "zoals pseudonimisering", die de beginselen van artikel 5 in de praktijk brengen. Drie van die beginselen dragen deze les. Doelbinding: gegevens worden "voor welbepaalde, uitdrukkelijk omschreven en gerechtvaardigde doeleinden" verzameld, en later niet gebruikt op een manier die daar niet mee te verenigen is. **Minimale gegevensverwerking**: gegevens moeten "toereikend zijn, ter zake dienend en beperkt tot wat noodzakelijk is voor de doeleinden waarvoor zij worden verwerkt". En opslagbeperking: gegevens "worden bewaard in een vorm die het mogelijk maakt de betrokkenen niet langer te identificeren dan voor de doeleinden waarvoor de persoonsgegevens worden verwerkt noodzakelijk is". Artikel 25, lid 2, zegt dan wat standaard geldt: "De verwerkingsverantwoordelijke treft passende technische en organisatorische maatregelen om ervoor te zorgen dat in beginsel alleen persoonsgegevens worden verwerkt die noodzakelijk zijn voor elk specifiek doel van de verwerking. Die verplichting geldt voor de hoeveelheid verzamelde persoonsgegevens, de mate waarin zij worden verwerkt, de termijn waarvoor zij worden opgeslagen en de toegankelijkheid daarvan." Waar de titel "door standaardinstellingen" zegt, staat in de tekst zelf "in beginsel".

    Let op het eerste woord van minimale gegevensverwerking: toereikend. De regel vraagt niet om zo weinig mogelijk gegevens, maar om wat het doel nodig heeft, en niet meer.

    ### Wat standaard geldt

    De European Data Protection Board, de EDPB, waarin de Europese toezichthouders op gegevensbescherming samenwerken, legt artikel 25 uit in richtlijnen. Een standaardinstelling is daar, net als in de informatica, de waarde die een instelling al heeft voordat iemand iets kiest: wat geldt als niemand iets verandert. Volgens de EDPB verzamelt een organisatie dus standaard niet meer gegevens dan nodig, verwerkt ze die niet verder dan de doelen vragen, en bewaart ze die niet langer dan nodig. Cavoukian zei hetzelfde van de andere kant: als iemand niets doet, blijft de privacy van die persoon intact.

    ### Minder detail, of helemaal niets

    De EDPB begint nog eerder: wie gegevens verwerkt, hoort eerst te bepalen of er voor het doel überhaupt persoonsgegevens nodig zijn. En als ze nodig zijn, is minder detail misschien genoeg: wie leden per geboortedecennium telt, heeft geen geboortedata nodig. Het voorbeeld van de EDPB zelf is een boekwinkel waarvan het bestelformulier om een geboortedatum en een telefoonnummer vraagt, terwijl het kopen van een boek die niet nodig heeft; voor e-books, die niet per post gaan, hoeft het formulier ook geen adres te vragen.

    ### Wissen als het klaar is

    Persoonsgegevens die niet meer nodig zijn voor het doel, moeten volgens de EDPB standaard gewist of geanonimiseerd worden, en het wissen van bepaalde persoonsgegevens hoort geautomatiseerd te zijn, zodat het niet wacht tot iemand eraan denkt. Bij een lidmaatschap: de gegevens worden gewist als het lidmaatschap stopt en er geen wettelijke grond is om ze langer te bewaren. De AVG noemt geen vaste termijn; volgens de EDPB hangt de bewaartermijn af van het doel. En anonimiseren in plaats van wissen is alleen genoeg als niemand de persoon nog kan terugvinden, zoals in [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl): de EDPB vraagt de verwerkingsverantwoordelijke dat te controleren.

    ### Gegevens die gewist hadden moeten worden

    De Autoriteit Persoonsgegevens (AP) beschreef in 2021 een zaak: "De Belastingdienst had de gegevens over de dubbele nationaliteit van Nederlanders al in januari 2014 moeten wissen. De dubbele nationaliteit van Nederlanders speelt geen rol bij het beoordelen van een aanvraag voor kinderopvangtoeslag. Toch bewaarde en gebruikte de Belastingdienst deze gegevens. In mei 2018 stond in totaal van nog 1,4 miljoen mensen de dubbele nationaliteit geregistreerd in systemen van de Belastingdienst." Pas in de zomer van 2020 waren ze er volledig uit verwijderd. De AP legde de Belastingdienst in totaal een boete op van 2,75 miljoen euro, waarvan € 750.000 voor het bewaren van deze gegevens. In het besluit beboette de AP dit als verwerken zonder rechtmatige grondslag, niet als overtreding van de opslagbeperking. De overtreding liep van 2014 tot 2020; voor de boete telde de AP alleen de jaren vanaf 1 januari 2016, toen de AP de bevoegdheid kreeg om zulke overtredingen te beboeten.

    ### Waarom dit vóór een filter komt

    In [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl) miste elk filter iets. Een key waar het formulier nooit om vraagt, heeft geen filter nodig, en een lid dat gewist is, kan niet lekken. Wat de pipeline wél bewaart, zijn vaak nog persoonsgegevens: minder om te lekken is niet niets om te lekken.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **privacy by design** | privacybescherming vanaf het begin in een systeem inbouwen, met privacyvriendelijke standaardinstellingen, in plaats van die achteraf toe te voegen |
        | **minimale gegevensverwerking** | alleen de persoonsgegevens verwerken die een doel nodig heeft: toereikend, ter zake dienend en beperkt tot wat noodzakelijk is |
        | `x is None` | `True` als `x` gelijk is aan `None`, het Python-woord voor niets; `x is not None` is het omgekeerde |
        """),
        L.watch_out("Minimale gegevensverwerking is niet \"zo weinig mogelijk\": de gegevens moeten toereikend zijn voor het doel, dus te weinig is ook een probleem. En wat een pipeline na het minimaliseren bewaart, zijn vaak nog persoonsgegevens: minder om te lekken is niet niets om te lekken."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een pipeline die alleen bewaart wat de doelen nodig hebben:

    1. Schrijf eerst elk doel op, en per doel de keys die het nodig heeft. Vraag eerst of het doel überhaupt persoonsgegevens nodig heeft, en dan of minder detail genoeg is, zoals een decennium in plaats van een geboortedatum.
    2. Bewaar van elk record alleen die keys. Dat is `minimize`.
    3. Bepaal wanneer gegevens weggaan, en laat ze gaan zonder te wachten tot iemand eraan denkt: hier, als een lidmaatschap stopt. Dat is `to_delete`.
    4. Tel wat er overblijft: de waarden die zichtbaar worden als de gegevens van de pipeline lekken. Dat is `values_kept`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel laat het eerste lid van het bestand zien, key voor key, en dan wat een telling van leden per geboortedecennium nodig heeft: het decennium, niet de datum. `C.MEMBERS` is het bestand, een lijst van tien dicts. `birth_date[:3]` neemt de eerste drie tekens, zoals `"195"`, en `+ "0s"` maakt er `"1950s"` van; `postcode[:4]` neemt het gebied, `"3511"`. Een dict telt de leden per decennium.
    """)
    return


@app.cell
def example_decades():
    # het eerste lid van de verzonnen bibliotheek, zoals het bestand het heeft
    _member = C.MEMBERS[0]
    for _key in _member:
        print(_key, _member[_key])

    # voor een telling per geboortedecennium en per gebied is minder detail genoeg
    print("decade:", _member["birth_date"][:3] + "0s", "area:", _member["postcode"][:4])

    # hoeveel leden in elk decennium geboren zijn
    _count = {}
    for _m in C.MEMBERS:
        _decade = _m["birth_date"][:3] + "0s"
        if _decade in _count:
            _count[_decade] = _count[_decade] + 1
        else:
            _count[_decade] = 1
    print(_count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Anouk de Vries heeft negen waarden in het bestand, van het kaartnummer `L-1001` tot `ended`, dat `None` is: Anouk de Vries is nog lid. Voor een telling per decennium is `1950s` genoeg, en het gebied `3511` zegt minder over waar iemand woont dan `3511 KL`. De telling geeft twee leden voor elk decennium van de jaren 70 tot en met de jaren 2000, en één voor de jaren 50 en één voor de jaren 60: tien leden, en voor de telling is het decennium van elk lid genoeg: een formulier kan daarnaar vragen in plaats van naar de geboortedatum. In de oefeningen bouw je hetzelfde idee in de pipeline in: per doel alleen de keys die het nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minimize():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. minimize: alleen wat het doel nodig heeft

        `minimize(record, needed)` krijgt één lid als dict `record`, en een lijst `needed` met de keys die een doel nodig heeft. Hij geeft een nieuwe dict terug met alleen de keys van `record` die in `needed` staan, elk met zijn waarde. Een key in `needed` die het lid niet heeft, zoals het telefoonnummer dat Lina Postma nooit gaf, komt er niet in. `PURPOSES` bevat wat elk doel nodig heeft: `PURPOSES["lending"]` is `["card", "name", "email"]`, voor het uitlenen met een herinnering per e-mail als een boek te laat is, en `PURPOSES["recommend"]` is `["books"]`, voor het aanbevelingsmodel. Binnenin begint `kept` als een lege dict, en de loop gaat langs de keys van `record`. Vervang de `...`. De check roept `minimize(MEMBERS[0], PURPOSES["lending"])` aan, en daarna andere leden en keys.
        """),
        L.task("Geef een dict terug met alleen de keys die het doel nodig heeft, elk met zijn waarde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minimize():
    box_minimize = L.code_box("minimize", lines=7)
    box_minimize
    return (box_minimize,)


@app.cell(hide_code=True)
def check_minimize(box_minimize):
    result_minimize = L.check("minimize", box_minimize.value)
    result_minimize
    return (result_minimize,)


@app.cell(hide_code=True)
def hints_minimize():
    L.hints(
        "minimize",
        [
            "Stap 2 van het recept. In [Dictionaries](/course/open/py-dicts?lang=nl) geeft `for fruit in prices:` één key tegelijk, en `prices[fruit]` de waarde erbij.",
            "Vraag voor elke key of die in `needed` staat. Zo ja, zet hem in `kept`, met dezelfde waarde als in `record`.",
            "`if key in needed:` en daaronder `kept[key] = ___`",
        ],
        why='`if key in needed:` met daaronder `kept[key] = record[key]` bewaart een key met zijn waarde als het doel hem nodig heeft. `minimize(MEMBERS[0], PURPOSES["lending"])` geeft `{"card": "L-1001", "name": "Anouk de Vries", "email": "anouk.devries@example.org"}`: zes van de negen waarden van het lid blijven buiten.',
    )
    return


@app.cell(hide_code=True)
def text_after_minimize():
    mo.md(r"""
    Met een goede `minimize` zijn in het plaatje 30 van de 89 vakjes gevuld: drie keys voor elk van de tien leden. Voor het uitlenen zijn geen geboortedatum en geen telefoonnummer nodig, dus de pipeline voor het uitlenen haalt ze nooit binnen. Maar de pipeline bewaart nog wel de drie leden die vertrokken zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_delete():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. to_delete: wie weg moet

        `to_delete(members)` krijgt een lijst leden, en geeft de kaartnummers terug van de leden van wie het lidmaatschap gestopt is, in de volgorde van de lijst: de leden van wie de bibliotheek de gegevens wist. Een lid dat vertrokken is, heeft een maand in `"ended"`, zoals `"2019-03"`; een lid dat nog lid is, heeft `None`. Binnenin begint `cards` als een lege lijst, en de loop gaat langs de leden. Vervang de `...`. De check roept `to_delete(MEMBERS)` aan, en daarna andere lijsten.
        """),
        L.task("Geef de kaartnummers terug van de leden van wie het lidmaatschap gestopt is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_delete():
    box_delete = L.code_box("to-delete", lines=7)
    box_delete
    return (box_delete,)


@app.cell(hide_code=True)
def check_delete(box_delete):
    result_delete = L.check("to-delete", box_delete.value)
    result_delete
    return (result_delete,)


@app.cell(hide_code=True)
def hints_delete():
    L.hints(
        "to-delete",
        [
            "Stap 3 van het recept. `x is None` vraagt of `x` gelijk is aan `None`; `x is not None` vraagt het omgekeerde.",
            'Een lid is vertrokken als `"ended"` een maand bevat, dus als het niet `None` is. Voeg voor elk zo\'n lid het kaartnummer toe aan `cards`, met `append`.',
            '`if member["ended"] is not None:` en daaronder `cards.append(___)`',
        ],
        why='`if member["ended"] is not None:` met daaronder `cards.append(member["card"])` verzamelt de kaartnummers van de leden die vertrokken zijn. `to_delete(MEMBERS)` geeft `["L-1003", "L-1006", "L-1009"]`: Yara El Amrani vertrok in maart 2019, Sem Brouwer in november 2023 en Nora Willems in juni 2025.',
    )
    return


@app.cell(hide_code=True)
def text_after_delete():
    mo.md(r"""
    Zeven leden blijven. In een echte pipeline gaat zo'n stap vanzelf, zoals de EDPB vraagt. Ook kopieën tellen mee: de EDPB vraagt verwerkingsverantwoordelijken te bepalen welke persoonsgegevens back-ups en logs bewaren, en hoe lang. De AVG noemt geen vaste termijn; deze bibliotheek wist de gegevens van een lid als het lidmaatschap stopt, zoals in het voorbeeld van de EDPB.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kept():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. values_kept: wat er nog kan lekken

        `values_kept(members, needed)` geeft terug hoeveel waarden de pipeline voor een doel bewaart: voor elk lid dat blijft het aantal keys dat `minimize` voor dat lid bewaart, bij elkaar opgeteld. `len` van een dict geeft het aantal keys. Een lid bij wie `"ended"` niet `None` is, telt niet mee: die gegevens zijn gewist. Het vak kent de `minimize` van de course. Binnenin begint `count` bij 0. Vervang de `...`. De check roept `values_kept(MEMBERS, PURPOSES["recommend"])` aan, en daarna andere lijsten en doelen.
        """),
        L.task("Geef terug hoeveel waarden de pipeline bewaart van de leden die blijven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kept():
    box_kept = L.code_box("values-kept", lines=7)
    box_kept
    return (box_kept,)


@app.cell(hide_code=True)
def check_kept(box_kept):
    result_kept = L.check("values-kept", box_kept.value)
    result_kept
    return (result_kept,)


@app.cell(hide_code=True)
def hints_kept():
    L.hints(
        "values-kept",
        [
            'Stap 4 van het recept. `len({"a": 1, "b": 2})` geeft `2`.',
            "Sla een lid over van wie het lidmaatschap gestopt is. Laat voor elk ander lid de functie van de course uit Oefening 1 het bewaren doen, en tel bij `count` op hoeveel keys die bewaarde.",
            '`if member["ended"] is None:` en daaronder `count = count + len(___)`',
        ],
        why='`if member["ended"] is None:` met daaronder `count = count + len(minimize(member, needed))` telt op wat de pipeline bewaart. `values_kept(MEMBERS, PURPOSES["recommend"])` geeft `7`: één waarde voor elk van de 7 leden die blijven. Voor het uitlenen geeft hij `21`, en met elke key geeft `values_kept(MEMBERS, FIELDS)` `62`: 7 leden keer 9 keys, min het telefoonnummer dat Lina Postma nooit gaf.',
    )
    return


@app.cell(hide_code=True)
def text_after_kept():
    mo.md(r"""
    Van 89 waarden naar 7: wat het aanbevelingsmodel krijgt, is minder dan een tiende van het bestand. Als die gegevens zouden lekken, lieten ze zien welke soorten boeken zeven leden leenden, zonder hun namen, e-mailadressen of geboortedata. Dat kunnen nog steeds persoonsgegevens zijn: een leeslijst kan naar één persoon wijzen. En het uitlenen heeft nog steeds namen en e-mailadressen nodig. Privacy by design maakt een pipeline niet veilig; het maakt kleiner wat er mis kan gaan.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Een doel van jezelf

    Kies de keys die een doel van jezelf nodig heeft. Het plaatje laat zien wat de pipeline bewaart van de zeven leden die blijven: met jouw `minimize` zodra Oefening 1 klopt, en met die van de course tot dan. Probeer een nieuwsbrief: welke keys heeft die nodig? En naar welke leden zou je hem sturen?
    """)
    return


@app.cell(hide_code=True)
def try_keys_cell():
    try_keys = mo.ui.multiselect(options=list(C.FIELDS), value=["books"], label="Keys die je doel nodig heeft")
    try_keys
    return (try_keys,)


@app.cell(hide_code=True)
def demo_try(box_minimize, result_minimize, try_keys):
    _yours = C.mine(box_minimize.value, result_minimize.state == "correct")
    _out = mo.Html(C.try_picture(try_keys.value, _yours, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_minimum():
    mo.md(r"""
    ### Oefening 4. Minder dan nodig

    Om nog minder te laten lekken, haalt de bibliotheek het e-mailadres uit `PURPOSES["lending"]`. Nu kan de bibliotheek geen herinnering meer sturen als een boek te laat is.
    """)
    return


@app.cell(hide_code=True)
def choice_minimum_cell():
    choice_minimum = L.choice(
        "minimum",
        question="Wat vraagt minimale gegevensverwerking van de bibliotheek?",
        options={
            "few": "Zo weinig mogelijk, wat het doel ook is",
            "right": "Wat het doel nodig heeft, en niets meer",
            "filter": "Alleen wat een filter achteraf kan verbergen",
        },
        correct="right",
        why={
            "right": "De AVG vraagt om gegevens die toereikend, ter zake dienend en beperkt tot het noodzakelijke zijn voor het doel. Voor het uitlenen met een herinnering is het e-mailadres noodzakelijk: zonder dat adres heeft de bibliotheek te weinig voor het doel.",
            "few": "Het eerste woord van de regel is toereikend: genoeg voor het doel. Een bibliotheek die geen herinnering kan sturen, heeft te weinig. Zie *Wat de wet vraagt* in *Het idee*.",
            "filter": "Een filter komt nadat de gegevens binnen zijn; minimale gegevensverwerking bepaalt, aan de hand van het doel, wat er überhaupt binnenkomt. Zie *Wat de wet vraagt* in *Het idee*.",
        },
    )
    choice_minimum
    return (choice_minimum,)


@app.cell(hide_code=True)
def text_exercise_default():
    mo.md(r"""
    ### Oefening 5. De nieuwsbrief

    Een nieuw lid schrijft zich in op de website van de bibliotheek. De bibliotheek stuurt een nieuwsbrief naar de leden die hem willen.
    """)
    return


@app.cell(hide_code=True)
def choice_default_cell():
    choice_default = L.choice(
        "default",
        question="Welk inschrijfformulier volgt privacy als standaard?",
        options={
            "right": "Het vakje voor de nieuwsbrief is eerst leeg; leden kunnen het aanvinken",
            "ticked": "Het vakje voor de nieuwsbrief is eerst aangevinkt; leden kunnen het uitvinken",
            "link": "Geen vakje: elke nieuwsbrief heeft een link om te stoppen",
        },
        correct="right",
        why={
            "right": "Privacy als standaard: als iemand niets doet, blijft de privacy van die persoon intact. Een lid dat niets doet, krijgt geen nieuwsbrief, en de bibliotheek gebruikt het e-mailadres er niet voor.",
            "ticked": "Een lid dat niets doet, krijgt dan de nieuwsbrief: de standaard werkt tegen de privacy in. Zie *Wat standaard geldt* in *Het idee*.",
            "link": "Een link om te stoppen vraagt het lid om achteraf iets te doen; standaard hoort er niets te gebeuren als het lid niets doet. Zie *Wat standaard geldt* in *Het idee*.",
        },
    )
    choice_default
    return (choice_default,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Persoonsgegevens vinden en verbergen](/course/open/rai-find-pii?lang=nl) verborg je namen en nummers die al in de tekst stonden, en er glipte altijd iets doorheen. Hier besliste je voordat er iets binnenkwam: per doel welke keys de pipeline meeneemt, en wanneer de gegevens van een lid weggaan. In de volgende les, *Wie beheert de data?*, wordt de vraag wie dat in een organisatie beslist, wie wat mag zien, en hoe iemand weet waar alle kopieën van iemands gegevens staan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_default, choice_minimum, result_delete, result_kept, result_minimize):
    L.wrap_up(result_minimize, result_delete, result_kept, choice_minimum, choice_default)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
