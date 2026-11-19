# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-pseudonymization. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-pseudonymization", "nl")
    C = L.checks   # de twaalf gebruikers van een verzonnen taalapp, dezelfde rijen met codes, en de lijst van een café


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [De AVG voor AI](/course/open/rai-gdpr?lang=nl) en [Dictionaries](/course/open/py-dicts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_identifiable():
    quick_identifiable = L.choice(
        "quick-identifiable",
        question="Volgens de AVG zijn gegevens persoonsgegevens als ze gaan over een persoon die…",
        options={
            "named": "erin genoemd wordt",
            "right": "geïdentificeerd of identificeerbaar is",
            "eu": "in een van de landen van de Europese Unie woont",
        },
        correct="right",
        why={
            "right": "Persoonsgegevens zijn \"alle informatie over een geïdentificeerde of identificeerbare natuurlijke persoon\". Een persoon is identificeerbaar als die direct of indirect te identificeren is, ook zonder naam.",
            "named": "Een naam is niet nodig: wie indirect te identificeren is, met een nummer of met feiten over die persoon, is ook identificeerbaar. Zie *Persoonsgegevens, en wie verantwoordelijk is* in *Het idee* van [De AVG voor AI](/course/open/rai-gdpr?lang=nl).",
            "eu": "Bij persoonsgegevens gaat het erom of een persoon geïdentificeerd of identificeerbaar is, niet waar die woont. Zie *Persoonsgegevens, en wie verantwoordelijk is* in *Het idee* van [De AVG voor AI](/course/open/rai-gdpr?lang=nl).",
        },
    )
    quick_identifiable
    return (quick_identifiable,)


@app.cell(hide_code=True)
def quick_check_controller():
    quick_controller = L.choice(
        "quick-controller",
        question="Een bedrijf heeft een taalapp en beslist wat het met de gegevens van de gebruikers doet, en waarom. Een hostingbedrijf bewaart de gegevens op zijn servers, en doet alleen wat het bedrijf van de app vraagt. Wie is de verwerkingsverantwoordelijke?",
        options={
            "host": "Het hostingbedrijf, dat de gegevens bewaart",
            "right": "Het bedrijf achter de app",
            "users": "Elke gebruiker, voor de eigen gegevens",
        },
        correct="right",
        why={
            "right": "De verwerkingsverantwoordelijke is wie \"het doel van en de middelen voor de verwerking van persoonsgegevens vaststelt\": het bedrijf van de app. Het hostingbedrijf verwerkt de gegevens namens dat bedrijf, en is een verwerker.",
            "host": "De gegevens bewaren maakt het hostingbedrijf geen verwerkingsverantwoordelijke: het volgt het bedrijf van de app, dat beslist waarom en hoe. Dat maakt het een verwerker. Zie *Persoonsgegevens, en wie verantwoordelijk is* in *Het idee* van [De AVG voor AI](/course/open/rai-gdpr?lang=nl).",
            "users": "De gebruikers zijn de betrokkenen: de gegevens gaan over hen. De verwerkingsverantwoordelijke is wie beslist waarom en hoe de gegevens verwerkt worden: het bedrijf van de app. Zie *Persoonsgegevens, en wie verantwoordelijk is* in *Het idee* van [De AVG voor AI](/course/open/rai-gdpr?lang=nl).",
        },
    )
    quick_controller
    return (quick_controller,)


@app.cell(hide_code=True)
def quick_check_keys():
    quick_keys = L.choice(
        "quick-keys",
        question="`row` is de dictionary `{\"email\": \"ana@mail.example\", \"hour\": 21}`. Waar gaat `key` langs in `for key in row:`?",
        options={
            "right": "De keys",
            "values": "De waarden",
            "pairs": "Paren van een key en zijn waarde",
        },
        correct="right",
        why={
            "right": "Een loop over een dictionary gaat langs de keys: hier eerst `\"email\"` en dan `\"hour\"`. De waarde van elk is `row[key]`.",
            "values": "Een loop over een dictionary gaat langs de keys; een waarde krijg je met `row[key]`. Zie *Oefening 2. Een loop over een dictionary* in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "pairs": "Een loop over een dictionary gaat alleen langs de keys, hier `\"email\"` en `\"hour\"`; een waarde krijg je met `row[key]`. Zie *Oefening 2. Een loop over een dictionary* in [Dictionaries](/course/open/py-dicts?lang=nl).",
        },
    )
    quick_keys
    return (quick_keys,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Gegevens over mensen zijn veel waard voor onderzoek en om modellen te trainen, en vaak komen ze met een belofte: de namen zijn eruit gehaald. Die belofte kan twee heel verschillende dingen betekenen. Een naam of een e-mailadres vervangen door een code is pseudonimisering: het blijven persoonsgegevens, in elk geval voor wie de sleutel heeft, en de AVG blijft gelden. Anoniem betekent veel meer: niemand kan de persoon terugvinden, met geen enkel middel dat redelijkerwijs gebruikt kan worden. In 2006 verving AOL de gebruikers-ID's in zijn zoekgegevens door nummers, en toch wisten journalisten mensen te identificeren aan de hand van hun zoekopdrachten; Netflix zei dat alle informatie die klanten identificeert verwijderd was, en twee onderzoekers lieten zien hoe weinig beoordelingen en datums er nodig zijn om het record van een abonnee eruit te pikken. In deze les pseudonimiseer je zelf de rijen van een verzonnen app, en daarna draai je het twee keer terug: één keer met een lijst e-mailadressen, één keer met drie feiten over één persoon.

    **Een voorbeeld uit het dagelijks leven.** Een groep leerlingen levert opstellen in met een nummer in plaats van een naam, zodat de docent nakijkt zonder te weten van wie een opstel is. Maar de docent kent de handschriften, weet wie altijd over paarden schrijft, en weet wie als enige vorige zomer in Japan was. De namen zijn weg; de leerlingen zijn nog te vinden.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Voor de Netflix Prize publiceerde Netflix de filmbeoordelingen van 480.189 abonnees, zonder hun namen. Stel dat je 8 beoordelingen van één abonnee kent, waarvan er 2 helemaal fout mogen zijn, en de datums tot op 14 dagen. Voor welk deel van de records van de abonnees is dat genoeg om er precies één uit te pikken?",
        options={
            "tenth": "Ongeveer 1 op de 10",
            "half": "Ongeveer de helft",
            "right": "99%",
        },
        reveal="**99%**: Arvind Narayanan en Vitaly Shmatikov (2008) vonden dat je met 8 beoordelingen, waarvan er 2 fout mogen zijn, en datums die 14 dagen ernaast mogen zitten, 99% van de records in de dataset uniek kunt identificeren. Voor 68% waren twee beoordelingen met datums binnen 3 dagen genoeg. Hieronder zie je waarom de AVG zulke data niet anoniem zou noemen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Een code in plaats van een adres

        De verzonnen taalapp uit [De AVG voor AI](/course/open/rai-gdpr?lang=nl) wil een onderzoeksteam laten bestuderen hoe mensen leren. De app haalt de e-mailadressen uit de rijen, en zet er een code voor in de plaats. De AVG noemt zo'n stap **pseudonimisering**: "het verwerken van persoonsgegevens op zodanige wijze dat de persoonsgegevens niet meer aan een specifieke betrokkene kunnen worden gekoppeld zonder dat er aanvullende gegevens worden gebruikt, mits deze aanvullende gegevens apart worden bewaard en technische en organisatorische maatregelen worden genomen", zegt artikel 4, punt 5. De aanvullende gegevens kunnen een lijst zijn die zegt welke code bij welk adres hoort: de sleutel.

        Het resultaat blijven persoonsgegevens: "Gepseudonimiseerde persoonsgegevens die door het gebruik van aanvullende gegevens aan een natuurlijke persoon kunnen worden gekoppeld, moeten als gegevens over een identificeerbare natuurlijke persoon worden beschouwd" (overweging 26). Dat maakt pseudonimisering niet nutteloos. Volgens overweging 28 kan het de risico's voor de betrokkenen verkleinen, en verwerkingsverantwoordelijken en verwerkers helpen hun verplichtingen na te komen. En het is niet overal verplicht: de AVG legt geen algemene plicht tot pseudonimisering op, schrijft het Europees Comité voor gegevensbescherming (EDPB) in zijn richtsnoeren over pseudonimisering van januari 2025, een conceptversie voor openbare raadpleging.
        """),
        mo.Html(C.shared_picture("nl")),
        mo.md(r"""
        ### Anoniem: niemand is terug te vinden

        Voor anonieme gegevens geldt de AVG niet: voor informatie die niet over een geïdentificeerde of identificeerbare persoon gaat, of voor persoonsgegevens die zo anoniem gemaakt zijn dat de betrokkene niet of niet meer identificeerbaar is (overweging 26). **Anonimisering** is gegevens zo verwerken dat dat klopt. Of iemand nog identificeerbaar is, is een brede vraag. Volgens dezelfde overweging tellen alle middelen, "bijvoorbeeld selectietechnieken" (in het Engels: singling out), waarvan redelijkerwijs te verwachten is dat de verwerkingsverantwoordelijke of een ander ze gebruikt, en tellen objectieve factoren mee, zoals wat identificeren kost en hoeveel tijd het vraagt. Het gaat dus niet alleen om de sleutel, en niet alleen om de app: iedereen telt, met elk middel dat redelijkerwijs gebruikt kan worden. **Heridentificatie** is achterhalen wie er achter gegevens zit die anoniem leken. Het Engelse woord, *re-identification*, staat niet in de Engelse tekst van de AVG, en *anonymisation* evenmin; *anonymous* staat er alleen in overweging 26.

        ### Drie risico's

        In 2014, onder de wet van vóór de AVG, schreef de Groep gegevensbescherming artikel 29 een advies over technieken om te anonimiseren. Het stelt elke techniek drie vragen: kun je een persoon er nog uit pikken, kun je records over een persoon nog aan elkaar koppelen, en kun je nog iets over een persoon afleiden? **Singling out**, iemand eruit pikken, is de mogelijkheid om de records van één persoon in de dataset af te zonderen. **Koppelbaarheid** is het kunnen koppelen van minstens twee records over dezelfde betrokkene of groep betrokkenen, in één database of in twee. Afleiden is met een aanzienlijke kans de waarde van een eigenschap achterhalen uit de waarden van andere eigenschappen, zoals een language model raadde waar een schrijver woont in [Informatie op de verkeerde plek](/course/open/rai-contextual-integrity?lang=nl).

        Voor pseudonimisering beantwoordt de tabel van het advies alle drie de vragen met ja: elk risico blijft. Het oordeel van de werkgroep: pseudonimisering is geen manier van anonimiseren; het maakt alleen de koppeling van een dataset met de echte identiteit van een betrokkene moeilijker, en is daarom een nuttige beveiligingsmaatregel. Anonimiseren moet daarentegen blijvend zijn: het moet identificatie onomkeerbaar onmogelijk maken.

        ### Een hash is geen schuilplaats

        Een **hashfunctie** maakt van elke tekst een korte reeks letters en cijfers: dezelfde tekst geeft altijd dezelfde hash, en een tekst die één teken verschilt een totaal andere. SHA-256 is zo'n functie. De werkgroep noemt de hashfunctie bij de meest gebruikte technieken om te pseudonimiseren, en waarschuwt: als bekend is welke invoer er kan zijn, kun je die invoer opnieuw door de hashfunctie halen en zo de echte waarde van een record terugvinden. Het concept van de EDPB geeft een voorbeeld: weet je dat de omzetting alleen een SHA-256-hash van een naam is, dan pas je die toe op alle namen die je elders hebt, en kijk je welke hashes in de gepseudonimiseerde dataset voorkomen. De EDPB wil daarom een geheim in de omzetting: alleen wie de geheime informatie heeft, moet het pseudoniem bij een identificerend gegeven kunnen berekenen. Geen van beide teksten noemt e-mailadressen; dat je een gehasht adres op dezelfde manier vindt, volgt uit dezelfde redenering, en je probeert het in Oefening 2.

        ### Namen weg, mensen gevonden

        In 2006 publiceerde AOL een database met twintig miljoen zoektermen van meer dan 650.000 gebruikers over drie maanden; het enige wat het deed om privacy te beschermen, was de gebruikers-ID vervangen door een nummer. Volgens de werkgroep werden daardoor sommige van die gebruikers publiekelijk geïdentificeerd en gelokaliseerd; The New York Times bracht het verhaal onder de kop *A Face Is Exposed for AOL Searcher No. 4417749*.

        Voor de Netflix Prize publiceerde Netflix 100.480.507 filmbeoordelingen van 480.189 abonnees, van december 1999 tot december 2005. In zijn FAQ schreef Netflix dat alle informatie die klanten identificeert verwijderd was, en dat alleen beoordelingen en datums over waren. Arvind Narayanan en Vitaly Shmatikov (2008) lieten zien wat beoordelingen en datums waard zijn. Met 8 beoordelingen, waarvan er 2 helemaal fout mogen zijn, en datums die tot 14 dagen ernaast zitten, waren 99% van de records in de dataset uniek te identificeren; voor 68% waren twee beoordelingen met datums binnen 3 dagen genoeg. Zonder datums was 84% van de abonnees nog eruit te pikken, als je 6 van 8 van hun films kent die buiten de top 500 vallen. Om het op echte mensen te proberen, namen ze een heel kleine steekproef van ongeveer 50 gebruikers van IMDb, en vonden ze de records van twee van hen in de data van Netflix.

        Luc Rocher, Julien Hendrickx en Yves-Alexandre de Montjoye (2019) schatten met een model dat 15 demografische eigenschappen 99,98% van de mensen in Massachusetts uniek maken. Hun conclusie: zulke datasets voldoen waarschijnlijk niet aan de moderne eisen voor anonimisering van de AVG.

        ### Pseudoniem voor wie?

        In september 2025 deed het Hof van Justitie van de Europese Unie uitspraak over precies deze vraag. De Gemeenschappelijke Afwikkelingsraad (SRB), een orgaan van de EU, had 1104 reacties van aandeelhouders en schuldeisers van een bank naar het adviesbureau Deloitte gestuurd, elk met een code in plaats van de naam van de schrijver. Alleen de SRB kon met die code de reacties aan de gegevens van de schrijvers koppelen. De zaak ging over de regels voor organen van de EU, die het Hof op dit punt net zo leest als de AVG. Het Hof: gepseudonimiseerde gegevens zijn niet in alle gevallen en voor iedereen persoonsgegevens, voor zover pseudonimisering, afhankelijk van de omstandigheden, daadwerkelijk kan verhinderen dat anderen dan de verwerkingsverantwoordelijke de betrokkene identificeren. Daarvoor mag de ontvanger de maatregelen niet ongedaan kunnen maken, ook niet met andere middelen, zoals gegevens naast elkaar leggen. Voor de SRB, die de sleutel heeft, bleven de reacties persoonsgegevens. De Europese Toezichthouder voor gegevensbescherming (EDPS) had betoogd dat gepseudonimiseerde gegevens voor iedereen persoonsgegevens zijn, en het Hof wees dat af als ongegrond. Toch vernietigde het Hof het arrest van het Gerecht, dat het besluit van de EDPS nietig had verklaard, en stuurde het de zaak terug naar het Gerecht, dat de feiten nog moet wegen: voor de plicht om mensen te vertellen wie hun gegevens krijgt, moet volgens het Hof worden beoordeeld of de betrokkene identificeerbaar is op het moment dat de gegevens worden verzameld, en vanuit de verwerkingsverantwoordelijke.

        Het concept van de EDPB, van vóór de uitspraak, gaat verder: gepseudonimiseerde gegevens blijven persoonsgegevens, ook als de gepseudonimiseerde gegevens en de aanvullende gegevens niet bij dezelfde persoon liggen. Voor de kant van de ontvanger is de uitspraak de latere en de bindende tekst.
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
        | **pseudonimisering** | vervangen wie iemand is, zoals een naam of een e-mailadres, door een code; met apart bewaarde extra informatie leidt de code terug naar de persoon, dus het blijven persoonsgegevens, in elk geval voor wie die informatie heeft |
        | **anonimisering** | gegevens zo verwerken dat niemand er nog uit te identificeren is, met geen enkel middel dat redelijkerwijs gebruikt kan worden; anonieme gegevens vallen buiten de AVG |
        | **heridentificatie** | achterhalen wie er achter gegevens zit die anoniem leken |
        | **singling out** | de records van één persoon in een dataset eruit pikken, ook zonder de naam te kennen |
        | **koppelbaarheid** | twee records over dezelfde persoon aan elkaar kunnen koppelen, in één dataset of in twee |
        | **hashfunctie** | een functie die van een tekst een korte reeks letters en cijfers maakt, altijd dezelfde voor dezelfde tekst; SHA-256 is er een |
        """),
        L.watch_out("Pseudonimisering is een goede maatregel, geen slechte: het kan de risico's verlagen, en volgens de AVG kan het helpen om aan de verplichtingen te voldoen. De fout is alleen het resultaat anoniem te noemen. Hooguit verbergt een code wie iemand is voor wie de sleutel niet heeft; de rest van de rij verbergt hij niet."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan of gegevens zonder namen anoniem zijn:

    1. Zoek wat direct zegt wie iemand is, zoals een naam of een e-mailadres, en vervang het door een code. Dat is `pseudonymize`.
    2. Vraag je af wie een code terug kan leiden naar een persoon: wie de sleutel heeft, en bij een gewone hash iedereen die de invoer kan raden en hashen. Dat is `unmask`.
    3. Kijk naar wat er in de rijen overblijft. Tel de rijen die passen bij wat iemand over een persoon kan weten; blijft er één rij over, dan is die persoon eruit gepikt. Dat is `candidates`.
    4. Noem de gegevens alleen anoniem als niemand, met middelen die redelijkerwijs gebruikt kunnen worden, een persoon eruit kan pikken, records kan koppelen of iets kan afleiden. Anders blijven het persoonsgegevens.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel hasht drie e-mailadressen met SHA-256, en houdt de eerste 8 tekens van elke hash als code. `hashlib.sha256` berekent de hash; die werkt op bytes, dus `.encode("utf-8")` maakt eerst bytes van de tekst, `.hexdigest()` schrijft de hash als letters en cijfers, en `[:8]` houdt de eerste 8 daarvan.
    """)
    return


@app.cell
def example_hash():
    import hashlib

    for _email in ["ana@mail.example", "bo@mail.example", "ana@mail.example"]:
        # de code van een adres: de eerste 8 tekens van de SHA-256-hash
        _code = hashlib.sha256(_email.encode("utf-8")).hexdigest()[:8]
        print(_email, "→", _code)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het eerste en het laatste adres zijn hetzelfde, en hun codes ook. Daardoor is zo'n code nuttig: een onderzoeksteam kan één gebruiker door al die rijen heen volgen, zonder ooit het adres te zien. Daardoor is hij ook zwak: wie hetzelfde adres hasht, krijgt dezelfde code. In de oefeningen doet `code_of(text)` wat de loop met elk adres doet: de versie van de course staat klaar.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pseudonymize():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. pseudonymize: een code in plaats van het adres

        `USERS` is een lijst met de twaalf gebruikers van de verzonnen app. Elke gebruiker is een dictionary met de keys `"email"`, `"language"` (de taal die iemand leert), `"joined"` (het jaar van aanmelden), `"hour"` (het uur van de dag waarop iemand meestal oefent), `"streak"` (hoeveel dagen achter elkaar iemand oefende) en `"goal"` (het doel). De eerste drie:

        | | email | language | joined | hour | streak | goal |
        |---|---|---|---|---|---|---|
        | `USERS[0]` | ana@mail.example | Spanish | 2022 | 21 | 15 | travel through Peru |
        | `USERS[1]` | bo@mail.example | Japanese | 2023 | 7 | 41 | a new job in Osaka |
        | `USERS[2]` | chen@mail.example | Japanese | 2023 | 22 | 3 | films without subtitles |

        `pseudonymize(row)` krijgt één zo'n dictionary `row`, en geeft een nieuwe dictionary terug met dezelfde keys en waarden, behalve dat `"email"` weg is en `"code"` de `code_of` van het adres bevat. Het vak mag `code_of` gebruiken. `new` begint als lege dictionary, en de `for`-loop gaat elke `key` van `row` langs. Vervang de `...` door een `if` en een `else`: als `key` gelijk is aan `"email"`, zet dan `code_of(row[key])` in `new` onder `"code"`; zet anders `row[key]` in `new` onder `key`. De functie geeft `new` terug. De check roept `pseudonymize(USERS[0])` aan, daarna andere rijen, ook met andere keys.
        """),
        L.task("Geef een kopie van de rij terug, met het e-mailadres vervangen door de code ervan onder de key \"code\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pseudonymize():
    box_pseudonymize = L.code_box("pseudonymize", lines=9)
    box_pseudonymize
    return (box_pseudonymize,)


@app.cell(hide_code=True)
def check_pseudonymize(box_pseudonymize):
    result_pseudonymize = L.check("pseudonymize", box_pseudonymize.value)
    result_pseudonymize
    return (result_pseudonymize,)


@app.cell(hide_code=True)
def hints_pseudonymize():
    L.hints(
        "pseudonymize",
        [
            "Stap 1 van het recept. Elke key van de rij komt in `new` terecht, alleen gaat het adres erin als de code ervan, onder een andere key.",
            "Een `if key == \"email\":` met één regel eronder, en een `else:` met één regel eronder. Onder welke key komt de code, en onder welke key komt elke andere waarde?",
            "Onder de `if`: `new[\"code\"] = code_of(row[key])`; onder de `else`: `new[___] = row[key]`",
        ],
        why="`if key == \"email\":` met `new[\"code\"] = code_of(row[key])` eronder, en `else:` met `new[key] = row[key]` eronder. `pseudonymize(USERS[0])` geeft `{\"code\": \"7f0d4910\", \"language\": \"Spanish\", \"joined\": 2022, \"hour\": 21, \"streak\": 15, \"goal\": \"travel through Peru\"}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unmask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. unmask: wie zit er achter een code?

        `SHARED` is de lijst met de twaalf rijen na pseudonimisering, zoals het onderzoeksteam ze krijgt, en `CODES` de lijst met hun twaalf codes, in dezelfde volgorde. Een van de onderzoekers heeft ook een taalcafé, met een inschrijflijst: `CAFE` is een lijst met tien e-mailadressen.

        `unmask(codes, emails)` krijgt een lijst codes `codes` en een lijst adressen `emails`. De functie geeft een dictionary terug met, voor elk adres waarvan de code in `codes` staat, die code als key en het adres als waarde. Het vak mag `code_of` gebruiken. `found` begint als lege dictionary, en de `for`-loop gaat elk adres `email` langs. Vervang de `...` door twee regels: als `code_of(email)` in `codes` staat, zet `email` dan onder die code in `found`. De functie geeft `found` terug. De check roept `unmask(CODES, CAFE)` aan, daarna kortere lijsten.
        """),
        L.task("Geef een dictionary terug met de codes die bij een van de adressen horen, elk met het adres erbij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unmask():
    box_unmask = L.code_box("unmask", lines=7)
    box_unmask
    return (box_unmask,)


@app.cell(hide_code=True)
def check_unmask(box_unmask):
    result_unmask = L.check("unmask", box_unmask.value)
    result_unmask
    return (result_unmask,)


@app.cell(hide_code=True)
def hints_unmask():
    L.hints(
        "unmask",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld: hetzelfde adres geeft altijd dezelfde code, dus hash elk adres op de lijst en zoek de code ervan.",
            "Een `if` met `in codes`, en eronder één regel die iets aan de dictionary toevoegt. Wat staat er links van `in`: het adres, of de code ervan?",
            "`if code_of(email) in codes:` en eronder `found[___] = email`",
        ],
        why="`if code_of(email) in codes:` en eronder `found[code_of(email)] = email`. `unmask(CODES, CAFE)` geeft vier adressen: die van ana, chen, ivo en lou.",
    )
    return


@app.cell(hide_code=True)
def text_after_unmask():
    mo.md(r"""
    Vier van de tien adressen op de lijst van het café horen bij gebruikers van de app: ana, chen, ivo en lou. Van die vier kan het onderzoeksteam nu alles in hun rijen lezen: de taal, het uur, het doel. Het had geen sleutel nodig. De code was een gewone hash, en de lijst van het café gaf de invoer om te proberen: precies waar de werkgroep voor waarschuwt. Met een geheim in de omzetting, zoals de EDPB vraagt, had de lijst van het café niets gevonden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_candidates():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. candidates: eruit gepikt zonder naam

        Bo gebruikt de app, en een collega van Bo weet drie dingen: Bo leert Japans, meldde zich aan in 2023, en oefent om 7 uur 's ochtends. `KNOWN` is de dictionary `{"language": "Japanese", "joined": 2023, "hour": 7}`.

        `candidates(known, rows)` krijgt een dictionary met feiten `known` en een lijst rijen `rows`, zoals `SHARED`. De functie geeft een lijst terug met de codes van de rijen die bij elk feit kloppen: voor elke key in `known` heeft de rij dezelfde waarde. `found` begint als lege lijst, en de `for`-loop gaat elke `row` langs. Vervang de `...` door je eigen regels: begin met `fits = True`; ga elke `key` van `known` langs, en zet `fits` op `False` als `row[key]` verschilt van `known[key]`; voeg na die loop `row["code"]` toe aan `found` als `fits` nog `True` is. De functie geeft `found` terug. De check roept `candidates(KNOWN, SHARED)` aan, daarna minder feiten, andere feiten en andere rijen.
        """),
        L.task("Geef de codes terug van de rijen die bij elk feit in known kloppen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_candidates():
    box_candidates = L.code_box("candidates", lines=11)
    box_candidates
    return (box_candidates,)


@app.cell(hide_code=True)
def check_candidates(box_candidates):
    result_candidates = L.check("candidates", box_candidates.value)
    result_candidates
    return (result_candidates,)


@app.cell(hide_code=True)
def hints_candidates():
    L.hints(
        "candidates",
        [
            "Stap 3 van het recept. Een rij blijft alleen als geen van de feiten in `known` ertegen spreekt.",
            "Binnen de loop over de rijen: `fits = True`, dan een loop over de keys van `known` met een `if` die de rij als niet passend markeert, en na die loop een `if fits:`. Wat gaat er in `found`: de rij, of de code ervan?",
            "`if row[key] ___ known[key]:` en eronder `fits = False`",
        ],
        why="`fits = True`; `for key in known:` met `if row[key] != known[key]:` en `fits = False` eronder; dan `if fits:` met `found.append(row[\"code\"])` eronder. `candidates(KNOWN, SHARED)` geeft `[\"0f631e11\"]`: één rij, die van Bo.",
    )
    return


@app.cell(hide_code=True)
def text_after_candidates():
    mo.md(r"""
    Met één feit blijven er drie rijen over; met twee feiten twee rijen; met alle drie één: de rij van Bo, eruit gepikt zonder naam, adres of sleutel. De collega kan nu ook de rest van die rij lezen, zoals het doel van Bo: a new job in Osaka, een nieuwe baan in Osaka. Dat is singling out, het eerste van de drie risico's van de werkgroep, en het is wat Narayanan en Shmatikov deden met een paar beoordelingen en datums. Twaalf rijen maken het makkelijk; hun 480.189 abonnees maakten het niet moeilijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_recipient():
    mo.md(r"""
    ### Oefening 4. Persoonsgegevens voor wie?

    De app maakt zijn codes nu met een geheime sleutel die alleen de app heeft, in plaats van met een gewone hash, en geeft de rijen aan een onderzoeksteam. Het team heeft geen sleutel, en geen andere manier om uit te vinden wie wie is: geen lijst van een café, en geen collega die Bo kent.
    """)
    return


@app.cell(hide_code=True)
def choice_recipient_cell():
    choice_recipient = L.choice(
        "recipient",
        question="Zijn deze rijen persoonsgegevens, volgens de uitspraak van het Hof van Justitie in 2025?",
        options={
            "none": "Voor niemand, want de adressen zijn eruit gehaald",
            "right": "Voor de app wel, voor het team misschien niet",
            "always": "Ja, altijd, voor iedereen die ze in handen heeft",
        },
        correct="right",
        why={
            "right": "Het Hof: gepseudonimiseerde gegevens zijn niet in alle gevallen en voor iedereen persoonsgegevens. Voor de app, die de sleutel heeft, blijven de rijen persoonsgegevens. Voor een team dat de pseudonimisering niet ongedaan kan maken, ook niet door gegevens naast elkaar te leggen, misschien niet. De zaak ging terug naar het Gerecht, dat de feiten nog moet wegen.",
            "none": "Voor de app, die de sleutel heeft, blijven de rijen persoonsgegevens: dat zei het Hof over de SRB, die de codes aan de schrijvers kon koppelen. Zie *Pseudoniem voor wie?* in *Het idee*.",
            "always": "Het concept van de EDPB van januari 2025 komt in de buurt: gepseudonimiseerde gegevens blijven persoonsgegevens, ook als de rijen en de sleutel bij verschillende partijen liggen. Bij het Hof betoogde de EDPS dat ze voor iedereen persoonsgegevens zijn, en het Hof wees dat af als ongegrond: niet in alle gevallen en voor iedereen. Zie *Pseudoniem voor wie?* in *Het idee*.",
        },
    )
    choice_recipient
    return (choice_recipient,)


@app.cell(hide_code=True)
def text_exercise_deleted():
    mo.md(r"""
    ### Oefening 5. De sleutel is weg

    Stel dat de app elke gebruiker een willekeurige code had gegeven, en de lijst met welke code bij welk adres hoort apart had bewaard. Nu wist de app die lijst, zodat niemand een code nog kan terugzetten naar een adres.
    """)
    return


@app.cell(hide_code=True)
def choice_deleted_cell():
    choice_deleted = L.choice(
        "deleted",
        question="Zijn de rijen van het onderzoeksteam nu anoniem?",
        options={
            "yes": "Ja: zonder de lijst vindt niemand iemand terug",
            "right": "Alleen als niemand op een andere manier te vinden is",
            "never": "Nee, gepseudonimiseerde gegevens blijven altijd persoonsgegevens",
        },
        correct="right",
        why={
            "right": "Het concept van de EDPB: ook als alle aanvullende gegevens gewist zijn, kun je de gegevens pas als anoniem beschouwen als aan de voorwaarden voor anonimiteit is voldaan. De rij van Bo laat zien waarom: drie feiten waren genoeg, zonder sleutel. Anoniem betekent dat niemand, met middelen die redelijkerwijs gebruikt kunnen worden, een persoon eruit kan pikken, records kan koppelen of iets kan afleiden.",
            "yes": "Oefening 3 vond de rij van Bo zonder sleutel: drie feiten waren genoeg om die rij eruit te pikken. De lijst wissen sluit één weg terug af, niet elke weg. Zie *Drie risico's* in *Het idee*.",
            "never": "Gepseudonimiseerde gegevens kunnen anoniem worden, maar alleen als aan de voorwaarden voor anonimiteit is voldaan, ook nadat de sleutel gewist is, zegt het concept van de EDPB. Zie *Anoniem: niemand is terug te vinden* in *Het idee*.",
        },
    )
    choice_deleted
    return (choice_deleted,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De AVG voor AI](/course/open/rai-gdpr?lang=nl) zag je persoonsgegevens: informatie over een persoon die geïdentificeerd of identificeerbaar is. Hier zag je hoe ruim *identificeerbaar* is. Een code in plaats van een adres laat een persoon identificeerbaar: voor wie de sleutel heeft, voor iedereen die de invoer van een gewone hash kan raden, en voor iedereen die een paar feiten over die persoon weet. In de volgende les, *Verstoppen in een groep: k-anonimiteit*, zorg je dat elke combinatie van zulke feiten in meerdere rijen voorkomt, zodat geen rij alleen staat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_deleted, choice_recipient, result_candidates, result_pseudonymize, result_unmask):
    L.wrap_up(result_pseudonymize, result_unmask, result_candidates, choice_recipient, choice_deleted)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
