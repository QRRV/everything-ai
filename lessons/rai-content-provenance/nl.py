# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-content-provenance. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Content Provenance

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-content-provenance", "nl")
    C = L.checks   # de speelgoedkey uit 1978, de tekeningen van de course en hun labels


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_integrity():
    quick_integrity = L.choice(
        "quick-integrity",
        question="Op een nieuwssite verandert iemand zonder het recht daartoe het bijschrift onder een foto. Welke eigenschap van de CIA-triade breekt dat?",
        options={
            "secret": "Vertrouwelijkheid: alleen wie het mag, kan het lezen",
            "right": "Integriteit: alleen wie het mag, kan het veranderen",
            "there": "Beschikbaarheid: het is er als je het nodig hebt",
        },
        correct="right",
        why={
            "right": "Integriteit gaat verloren als informatie zonder recht wordt veranderd of vernietigd. Deze les gaat over hoe iedereen kan nagaan of een plaatje veranderd is sinds iemand het ondertekende.",
            "secret": "Iedereen mag het bijschrift lezen; wat misging, is dat iemand het veranderde. Zie *Drie dingen om te beschermen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
            "there": "De pagina is er nog voor iedereen; wat misging, is dat iemand hem veranderde. Zie *Drie dingen om te beschermen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
        },
    )
    quick_integrity
    return (quick_integrity,)


@app.cell(hide_code=True)
def quick_check_spoofing():
    quick_spoofing = L.choice(
        "quick-spoofing",
        question="Een plaatje gaat rond met een bijschrift dat een persbureau het maakte, maar dat persbureau deed dat nooit. Welke dreiging van STRIDE is dat?",
        options={
            "deny": "Repudiation: ontkennen dat je iets deed",
            "right": "Spoofing: je voordoen als iemand anders",
            "rights": "Elevation of privilege: rechten krijgen die je niet hoort te hebben",
        },
        correct="right",
        why={
            "right": "Iemand zet de naam van een ander onder het plaatje. Een digitale handtekening werkt daartegen: alleen wie de private key heeft, kan een handtekening maken die klopt, zoals deze les laat zien.",
            "deny": "Niemand ontkent hier iets te hebben gedaan: iemand doet zich voor als het persbureau. Zie *Wat er mis kan gaan: STRIDE* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
            "rights": "Niemand krijgt hier meer rechten dan hoort: iemand doet zich voor als het persbureau. Zie *Wat er mis kan gaan: STRIDE* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
        },
    )
    quick_spoofing
    return (quick_spoofing,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een plaatje van een overstroomde haven gaat rond, met de naam van een nieuwsredactie eronder. Heeft die redactie het gemaakt, en is het nog het plaatje dat ze maakte? Zoeken naar sporen van vervalsing is lastig. Een watermerk, een verborgen signaal in het plaatje waar een detector naar zoekt, kun je verwijderen: NIST, het Amerikaanse instituut voor normen, schreef in 2024 dat de meeste watermerkmethoden van nu steeds weer te verwijderen bleken, en dat het vinden van een watermerk altijd een kans op een fout heeft.

    Herkomst draait de vraag om. In plaats van te vragen of een plaatje nep is, vraag je wie ervoor instaat, en of het sindsdien veranderd is. De maker ondertekent het plaatje, en iedereen kan de handtekening controleren. Deze les bouwt dat met een key uit 1978 die klein genoeg is om met de hand te volgen, en kijkt daarna naar C2PA, de standaard achter Content Credentials, naar wat zo'n label je niet vertelt, naar watermerken, en naar wat de AI Act van de EU vraagt.

    **Een voorbeeld uit het dagelijks leven.** Een lakzegel op een brief laat zien van wie het zegel is, en dat niemand de brief sindsdien openmaakte. Het laat niet zien dat wat er in de brief staat, waar is.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Google stopte een watermerk in de antwoorden van zijn model Gemini, en vergeleek hoe vaak zo'n 20 miljoen antwoorden met en zonder watermerk een duimpje omhoog kregen. Hoe ver lagen de twee percentages duimpjes omhoog uit elkaar?",
        options={
            "right": "0,01%",
            "one": "Ongeveer 1%",
            "five": "Ongeveer 5%, in het voordeel van geen watermerk",
        },
        reveal="**0,01%.** Volgens Dathathri et al. in *Nature* (2024) scheelde het percentage duimpjes omhoog 0,01% tussen de twee modellen, met het model met watermerk hoger, en het percentage duimpjes omlaag 0,02%, met het model met watermerk lager. Beide verschillen waren statistisch niet significant. Een watermerk kan de antwoorden dus even goed laten als ze waren; of het een aanvaller overleeft, is een andere vraag, waar *Watermerken* in *Het idee* op ingaat.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Ondertekenen met twee keys

    Een **digitale handtekening** is volgens FIPS 186-5, de standaard van NIST, de uitkomst van een cryptografische bewerking op data waarmee je, als die goed is uitgevoerd, kunt nagaan wie de afzender is en dat de data niet veranderd zijn, en waardoor de ondertekenaar niet kan ontkennen te hebben getekend. In 1978 beschreven Rivest, Shamir en Adleman hoe dat kan: een bericht onderteken je met een key die je geheimhoudt, en iedereen kan die handtekening controleren met de bijbehorende key die je openbaar maakt. Een elektronische handtekening moet volgens hen afhangen van het bericht én van wie tekent.

    In hun methode, RSA, is de **public key** twee getallen, `n` en `e`, die iedereen mag kennen; de **private key** is één getal, `d`, dat alleen de ondertekenaar heeft. Om een getal `m` te ondertekenen, rekent de ondertekenaar `pow(m, d, n)` uit. Om een handtekening `s` te controleren, rekent iedereen `pow(s, e, n)` uit: geeft dat `m` terug, dan past de handtekening bij `m`. In Python verheft `pow` met drie getallen het eerste tot de macht het tweede, en geeft het de rest na delen door het derde: `pow(2, 10, 1000)` is `24`, want 2 tot de macht 10 is 1024. Het voorbeeld in het paper zelf is klein: p = 47, q = 59, dus n = 47 · 59 = 2773, met d = 157 en e = 17, waarbij 17 de inverse is van 157 modulo 2668. Daarbij is 2668 gelijk aan 46 · 58, beide priemgetallen min één, vermenigvuldigd, en de inverse zijn betekent dat 17 · 157 na delen door 2668 rest 1 geeft. Onderteken je 920 met deze key, dan krijg je 192, en controleren van 192 geeft 920 terug; 921 ondertekenen geeft 2129; maak je van de handtekening 192 een 193, dan geeft de controle 1260 en niet 920. Wie alleen `n` en `e` heeft, kan een handtekening controleren, maar geen getal naar keuze ondertekenen: de veiligheid berust volgens het paper deels op hoe moeilijk het is om de openbare deler `n` te ontbinden in factoren. Wie `p` en `q` uit `n` haalt, kan `d` uitrekenen.

    Echte keys zijn veel groter. FIPS 186-5, uit 2023, vraagt om een modulus met een even aantal bits, en minstens 2048 bits. De `n` = 2773 uit het paper heeft 12 bits, en iedereen vindt zijn `p` en `q` in een oogwenk: de key in deze les laat de rekensom zien, geen veiligheid.

    ### Een plaatje ondertekenen

    Een plaatje is niet één klein getal, maar een heleboel bytes. Een hashfunctie maakt van elk aantal bytes een waarde van één vaste lengte, een vingerafdruk: verander één byte, en de vingerafdruk verandert. De ondertekenaar ondertekent de vingerafdruk in plaats van het plaatje. Volgens de uitleg van C2PA combineert de standaard gewone cryptografische hashes, zoals SHA2-256, met een aanpak die lijkt op een Merkle tree. Een Merkle tree is een boom van hashes: elke hash dekt de hashes eronder, zodat die ene hash bovenaan elk deel dekt. In deze les neemt `C.toy_hash(picture, n)` de SHA-256 van de bytes van het plaatje, en houdt daarvan de rest na delen door `n`, zodat hij bij de speelgoedkey past. Met maar 2773 mogelijke waarden delen twee plaatjes makkelijk een speelgoedhash; een echte ondertekenaar ondertekent de hele SHA-256, met een key van 2048 bits of meer.

    ### Content Credentials

    **Contentherkomst** is informatie over waar een stuk inhoud vandaan komt en hoe het gemaakt of bewerkt is. In februari 2021 richtten Adobe, Arm, BBC, Intel, Microsoft en Truepic de Coalition for Content Provenance and Authenticity op, **C2PA**, om tot een standaard voor herkomst te komen en zo misleidende inhoud te bestrijden. Die standaard hangt een ondertekend label, een manifest, aan een bestand, zoals een plaatje of een video. **Content Credentials** is de naam die gebruikers zien: de term die de specificatie aanraadt voor een C2PA-manifest in niet-technische taal. Het Content Authenticity Initiative, dat door Adobe geleid wordt, noemt ze een soort voedingswaardelabel voor digitale inhoud.

    Een manifest bevat assertions: uitspraken over het bestand die de ondertekenaar doet of verzamelt als de claim gemaakt wordt, zoals welke camera of tool het plaatje maakte. Het bevat een claim: een digitaal ondertekende datastructuur, waaraan je ziet of ermee geknoeid is, die naar de assertions verwijst en naar wat het label aan het bestand bindt. En het bevat de claim signature, gemaakt met de private key van de ondertekenaar. Die binding houdt het label bij het plaatje. Een hard binding is een of meer cryptografische hashes die het hele bestand of een deel ervan uniek aanwijzen; een soft binding is een vingerafdruk die niet uniek is, of een onzichtbaar watermerk, waarmee je een plaatje terugvindt als zijn label kwijt is. Volgens het Content Authenticity Initiative zijn Content Credentials, anders dan andere metadata, cryptografisch ondertekend, zodat je ziet of ermee geknoeid is. Metadata zijn gegevens over een bestand, zoals wanneer het gemaakt is. Dezelfde site noemt de Leica M11-P de eerste camera ter wereld met Content Credentials ingebouwd, en contentcredentials.org meldt meer dan 500 bedrijven. De specificatie die deze les las, is versie 2.5, van oktober 2026.

    ### Wat een label je niet vertelt

    Een label dat klopt, zegt wie tekende en dat er sindsdien niets veranderde. Het zegt niet dat het plaatje laat zien wat er gebeurde. De uitleg van C2PA zegt het zelf: Content Credentials oordelen niet over de vraag of herkomstgegevens waar zijn, alleen over de vraag of ze correct opgebouwd, ongeschonden, geldig en vertrouwd zijn, en ze zijn geen wondermiddel tegen desinformatie. NIST voegt eraan toe dat C2PA-metadata kunnen ontbreken, te vaag kunnen zijn om te helpen, of zelfs onjuist, ook al zijn ze ondertekend. Wie een onware uitspraak ondertekent, ondertekent hem evengoed.

    *Geldig* en *vertrouwd* zijn twee dingen. Een handtekening klopt met de key van wie hem maakte, dus een controle moet weten van wie ze de keys vertrouwt: in deze les is `C.KEYS` die vertrouwenslijst, een dict met bij de naam van elke ondertekenaar de bijbehorende public key. De handtekening van een vreemde vertelt je alleen dat die vreemde tekende.

    Ook een ontbrekend label zegt weinig. Herkomst toevoegen is niet verplicht, en volgens de uitleg van C2PA kun je de herkomstgegevens gewoon verwijderen. NIST: metadata kunnen helemaal verdwijnen, en dat gebeurt vaak als bestanden gedeeld worden, bijvoorbeeld via sociale media. Dat kan ooit veranderen: als herkomstgegevens gewoon worden, kan het ontbreken ervan op een plek waar je ze zou verwachten volgens NIST zelf een signaal worden, in elk geval in sommige situaties. Het gaat erom makers te goeder trouw te helpen, niet om elke vervalser te pakken. contentcredentials.org zegt dat er altijd kwaadwillenden zullen blijven die synthetische inhoud als echt labelen, en dat het doel is om goedwillenden een manier te geven om te laten zien dat hun inhoud echt is.

    ### Watermerken

    Een **watermerk** is een verborgen signaal in de inhoud zelf, waar een detector later naar kan zoeken. SynthID van Google DeepMind stopt watermerken rechtstreeks in plaatjes, geluid, tekst en video die door AI gemaakt zijn. Een large language model (LLM) schrijft tekst token voor token, een woord of een stuk van een woord tegelijk, en kiest elk volgend token met toeval uit kansen die het uitrekent; dat kiezen heet sampling. Bij tekst past SynthID de kansen op de volgende tokens aan: het model kiest zijn volgende token net iets anders, op een manier die een detector herkent. Volgens Dathathri et al. in *Nature* (2024) verandert SynthID-Text niets aan de training van een LLM, alleen aan het samplen, en kost het vinden van het watermerk weinig rekenwerk; het LLM zelf is daarvoor niet nodig. Het is gebruikt om Gemini en Gemini Advanced van een watermerk te voorzien, en in hun meting op Gemma 7B-IT, een open model van Google, maakte het elk token 0,57% trager.

    Google schrijft dat zijn watermerk in plaatjes is ontworpen om bewerkingen te doorstaan zoals bijsnijden, filters, een andere frame rate of lossy compressie, die ruimte spaart door details weg te gooien. De auteurs van het paper over tekst zijn voorzichtig over de grenzen: watermerken zijn geen volledige oplossing om tekst van AI te herkennen, maar een aanvulling op andere aanpakken; ze zijn kwetsbaar voor stelen, vervalsen en wegpoetsen; en bewerkingen van de tekst verzwakken ze, zoals herschrijven door een LLM. Open modellen zijn een probleem: bij modellen die iedereen zelf draait, valt een watermerk moeilijk af te dwingen. NIST: laat een ander model zonder watermerk de tekst herschrijven, en vaak verdwijnt het watermerk, terwijl de tekst maar een beetje slechter wordt.

    Een handtekening en een watermerk doen ander werk. De handtekening zegt wie voor een bestand instaat en dat het sindsdien niet veranderd is, maar ze zit in de metadata, en die kun je verwijderen. Het watermerk zit in de inhoud en kan blijven als de metadata weg zijn, maar ook dat kun je verwijderen of vervalsen, en het vinden ervan gaat met fouten gepaard. C2PA kan allebei dragen: een watermerk is een van zijn soft bindings.

    ### Wat de wet vraagt

    De AI Act van de EU vraagt om markering. Volgens artikel 50, lid 2, moeten aanbieders van AI-systemen die synthetisch geluid, beeld, video of tekst maken, ook AI-systemen voor algemene doeleinden, ervoor zorgen dat de output in een machineleesbaar formaat gemarkeerd is en te herkennen is als kunstmatig gemaakt of bewerkt. Hun technische oplossingen moeten doeltreffend, interoperabel, robuust en betrouwbaar zijn, voor zover dat technisch kan. Wie een AI-systeem inzet dat een deepfake maakt, een plaatje, geluid of video dat met AI gemaakt of veranderd is zodat het echt lijkt, moet dat volgens artikel 50, lid 4, bekendmaken. De wet geldt vanaf 2 augustus 2026; een aanbieder van een systeem dat vóór die dag al op de markt was, moet uiterlijk 2 december 2026 aan lid 2 voldoen. De tekst die de course las, noemt geen techniek: geen C2PA, en geen watermerken.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.label_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **digitale handtekening** | een getal bij een bericht dat alleen de houder van de private key kan maken, en dat iedereen met de public key kan controleren |
        | **public key**, **private key** | `n` en `e`, die iedereen mag kennen; `d`, dat alleen de ondertekenaar heeft |
        | `pow(m, d, n)` | de rest van `m` tot de macht `d` na delen door `n` |
        | **contentherkomst** | informatie over waar inhoud vandaan komt en hoe die gemaakt of bewerkt is |
        | **C2PA** | de standaard voor ondertekende herkomstlabels, van de coalitie met dezelfde naam |
        | **Content Credentials** | de naam die gebruikers zien voor een C2PA-manifest: zijn assertions, zijn claim en de claim signature |
        | **watermerk** | een verborgen signaal in de inhoud zelf, waar een detector naar kan zoeken |
        | `C.toy_hash(picture, n)` | de SHA-256 van de bytes van het plaatje, ingekort tot een getal onder `n` |
        | `C.BOAT`, `C.KEYS` | de tekening van een boot van de course, 108 bytes; de vertrouwenslijst, met bij de naam van elke ondertekenaar de bijbehorende public key |
        """),
        L.watch_out("De key in deze les heeft 12 bits, en iedereen kan hem kraken. Bescherm er nooit iets echts mee, en ook niet met zelfgeschreven code voor handtekeningen: gebruik een library die een standaard uitvoert, met keys van 2048 bits of meer."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Onderteken met de private key en controleer met de public key: alleen de ondertekenaar kan een handtekening maken, en iedereen kan hem controleren.
    2. Onderteken de hash van de inhoud, zodat elke verandering in de inhoud de handtekening breekt.
    3. Controleer een label op volgorde: is er een, is de ondertekenaar iemand die je vertrouwt, is de inhoud nog dezelfde als bij het ondertekenen, en klopt de handtekening?
    4. Lees een label dat klopt als "deze ondertekenaar staat ervoor in, en het is sindsdien niet veranderd", nooit als "dit is waar".
    5. Lees een ontbrekend label als ontbrekend, niet als nep.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel ondertekent 920, het getal uit het voorbeeld in het paper, met de key uit het paper, en controleert de handtekening.
    """)
    return


@app.cell
def example_sign():
    n, e, d = C.N, C.E, C.D   # de speelgoedkey van Rivest, Shamir en Adleman, 1978: 2773, 17 en 157
    m = 920
    s = pow(m, d, n)          # onderteken met de private exponent d
    print("signature:", s)
    print("checking gives:", pow(s, e, n))
    print("checking 921 with it:", pow(s, e, n) == 921)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De controle geeft 920 terug, dus de handtekening past bij 920; bij 921 past hij niet. Ondertekenen vroeg om `d`, controleren alleen om `e` en `n`. In oefening 1 komen getallen binnen met hun handtekening, en onderweg zijn sommige paren veranderd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_genuine():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. genuine: welke handtekeningen kloppen?

        `C.SIGNED` bevat vijf getallen die `photo-desk.example`, een verzonnen nieuwsredactie, ondertekend zou hebben, elk met zijn handtekening, als paren `(m, s)`. Schrijf `genuine(pairs, e, n)`: de getallen `m` uit de lijst `pairs` waarvan de handtekening `s` klopt met de public key `e` en `n`, in de volgorde van `pairs`. `for m, s in pairs:` haalt elk paar uit elkaar in zijn getal en zijn handtekening, en de lijst `found` verzamelt de getallen die kloppen. Vervang de `...`. De check roept `genuine(SIGNED, 17, 2773)` aan, en daarna een paar eigen lijsten en keys.
        """),
        L.task("Controleer elke handtekening met de public key, en houd het getal als de controle het getal teruggeeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_genuine():
    box_genuine = L.code_box("genuine", lines=6)
    box_genuine
    return (box_genuine,)


@app.cell(hide_code=True)
def check_genuine(box_genuine):
    result_genuine = L.check("genuine", box_genuine.value)
    result_genuine
    return (result_genuine,)


@app.cell(hide_code=True)
def hints_genuine():
    L.hints(
        "genuine",
        [
            "*Ondertekenen met twee keys* in *Het idee* zegt wat de controle van een goede handtekening teruggeeft.",
            "Binnen de loop beslist één `if` of je `m` houdt; daaronder staat `found.append(m)`.",
            "`        if ___ == m:`",
        ],
        why="`if pow(s, e, n) == m:` met daaronder `found.append(m)`. `genuine(SIGNED, 17, 2773)` geeft `[920, 1110]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_genuine():
    mo.md(r"""
    Twee van de vijf kloppen. Verander het getal en houd de handtekening, en de handtekening past niet meer; verander de handtekening en houd het getal, en ook dan past hij niet. Om een getal naar keuze te ondertekenen, heb je `d` nodig: wie alleen `e` en `n` heeft, kan controleren, niet ondertekenen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sign():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. sign_picture: een label voor een plaatje

        Schrijf `sign_picture(picture, signer, d, n)`: het label voor de bytes `picture`, als een dict met drie keys. `"signer"` is de tekst `signer`, `"hash"` is de hash van het plaatje, `toy_hash(picture, n)`, en `"signature"` is de handtekening van die hash met de private exponent `d`. `toy_hash` is die van de course, en het vak kent hem. Vervang de `...`. De check roept `sign_picture(BOAT, "photo-desk.example", 157, 2773)` aan, en daarna andere plaatjes, ondertekenaars en keys.
        """),
        L.task("Hash het plaatje, onderteken de hash met d, en geef de drie delen terug als een dict."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sign():
    box_sign = L.code_box("sign-picture", lines=3)
    box_sign
    return (box_sign,)


@app.cell(hide_code=True)
def check_sign(box_sign):
    result_sign = L.check("sign-picture", box_sign.value)
    result_sign
    return (result_sign,)


@app.cell(hide_code=True)
def hints_sign():
    L.hints(
        "sign-picture",
        [
            "Eerst de vingerafdruk, dan de handtekening: het label ondertekent de hash, niet de bytes.",
            "Twee regels: één geeft `h` zijn waarde met `toy_hash`, en één geeft een dict terug met de keys `\"signer\"`, `\"hash\"` en `\"signature\"`.",
            "`    return {\"signer\": signer, \"hash\": h, \"signature\": ___}`",
        ],
        why="`h = toy_hash(picture, n)`, dan `return {\"signer\": signer, \"hash\": h, \"signature\": pow(h, d, n)}`. `sign_picture(BOAT, \"photo-desk.example\", 157, 2773)` geeft `{\"signer\": \"photo-desk.example\", \"hash\": 1110, \"signature\": 1378}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_sign():
    mo.md(r"""
    Dat is een speelgoedclaim met zijn claim signature. Verander één pixel van de boot en zijn hash verandert, dus de oude handtekening past niet meer; om het veranderde plaatje te ondertekenen, heb je `d` nodig. Een echt manifest ondertekent ook zijn assertions, zodat niemand kan veranderen wat het over het plaatje zegt zonder de handtekening te breken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. verdict: wat zegt het label?

        Schrijf `verdict(picture, manifest, keys)`: wat het label `manifest` zegt over de bytes `picture`, als een van vijf teksten. `manifest` is een label zoals `sign_picture` het maakt, of `None` als het plaatje er geen heeft; `manifest is None` vraagt of er geen is. `keys` is de vertrouwenslijst, een dict met bij de naam van elke ondertekenaar de bijbehorende public key, als paar `(e, n)`, zoals `C.KEYS`; `e, n = keys[signer]` haalt de key van de ondertekenaar uit elkaar, en `signer` is de ondertekenaar van het label. Controleer in deze volgorde: geen label geeft `"no credentials"`, een ondertekenaar die niet in `keys` staat geeft `"unknown signer"`, een plaatje waarvan de eigen hash verschilt van de `"hash"` in het label geeft `"changed"`, een handtekening die niet klopt met de `"hash"` in het label geeft `"bad signature"`, en anders is het verdict `"signed by "` gevolgd door de ondertekenaar. De eerste regels en de laatste staan er al. Vervang de `...`. De check roept `verdict(BOAT, BOAT_LABEL, KEYS)` aan, en daarna andere plaatjes, labels en vertrouwenslijsten.
        """),
        L.task("Geef een verdict terug bij de eerste controle die faalt, in de volgorde van de oefening."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=12)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Nog drie controles, elk een `if` die zijn verdict teruggeeft: eerst de vertrouwenslijst, dan het plaatje, dan de handtekening.",
            "`e, n = keys[signer]` komt na de controle op de vertrouwenslijst. De twee controles daarna vergelijken allebei iets met `manifest[\"hash\"]`: eerst de eigen hash van het plaatje, dan wat controleren van de handtekening geeft.",
            "`    if ___ != manifest[\"hash\"]:`",
        ],
        why="`if signer not in keys:` geeft `\"unknown signer\"` terug; `e, n = keys[signer]`; `if toy_hash(picture, n) != manifest[\"hash\"]:` geeft `\"changed\"` terug; `if pow(manifest[\"signature\"], e, n) != manifest[\"hash\"]:` geeft `\"bad signature\"` terug. `verdict(BOAT, BOAT_LABEL, KEYS)` geeft `\"signed by photo-desk.example\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_verdict():
    mo.md(r"""
    Het label van `pixel-mill.example` heeft een goede handtekening, gemaakt met een eigen key, maar `KEYS` bevat die key niet. Voeg hem toe, en `verdict` zegt `signed by pixel-mill.example`: wie je vertrouwt, kies je zelf, en dat beslist de handtekening niet. Het vervalste label heeft de juiste hash voor de bewerkte boot, maar alleen wie `d` heeft, had die hash kunnen ondertekenen. En de boot waarvan een platform het label weghaalde, is pixel voor pixel dezelfde boot als de ondertekende.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: verander één pixel

    Kies een pixel van de boot op zijn nummer, van 0 linksboven tot 107 rechtsonder. Het plaatje zet die pixel om: een lege wordt gevuld, een gevulde wordt leeg. Ernaast staat het `verdict` van de course met het label van de boot, of zonder label als het platform het weghaalt.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_pixel = mo.ui.slider(start=0, stop=107, step=1, value=17, label="Pixel", show_value=True)
    demo_strip = mo.ui.checkbox(value=False, label="Het platform haalt het label weg")
    mo.hstack([demo_pixel, demo_strip], justify="start", gap=2)
    return demo_pixel, demo_strip


@app.cell(hide_code=True)
def demo_look(demo_pixel, demo_strip):
    mo.Html(C.demo_picture(demo_pixel.value, demo_strip.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.md(r"""
    ### Oefening 4. Geen label

    Een plaatje van een overstroomde straat gaat rond op sociale media, zonder Content Credentials.
    """)
    return


@app.cell(hide_code=True)
def choice_label_cell():
    choice_label = L.choice(
        "no-label",
        question="Wat vertelt het ontbrekende label je?",
        options={
            "fake": "Dat het met AI gemaakt of veranderd is, want een camera had er een toegevoegd",
            "right": "Dat het er nu geen heeft: nooit gehad, of bij het delen weggehaald",
            "real": "Dat het echt is, want tools die met AI plaatjes maken, voegen een label toe",
        },
        correct="right",
        why={
            "right": "Volgens de uitleg van C2PA is herkomst toevoegen niet verplicht, en volgens NIST verdwijnen metadata vaak als bestanden gedeeld worden, bijvoorbeeld via sociale media. Alleen waar je een label zou verwachten, kan het ontbreken ervan ooit een signaal worden, schrijft NIST.",
            "fake": "Een label toevoegen is niet verplicht, en een platform kan het weghalen: in oefening 3 is de boot zonder label pixel voor pixel de ondertekende. Zie *Wat een label je niet vertelt* in *Het idee*.",
            "real": "De AI Act vraagt aanbieders om te markeren wat hun systemen maken, maar een markering kan een watermerk zijn in plaats van een label, en een label kan onderweg verdwijnen. Een ontbrekend label wijst geen kant op. Zie *Wat een label je niet vertelt* in *Het idee*.",
        },
    )
    choice_label
    return (choice_label,)


@app.cell(hide_code=True)
def text_exercise_watermark():
    mo.md(r"""
    ### Oefening 5. Twee uitspraken over watermerken

    Google schrijft dat het watermerk van SynthID in plaatjes ontworpen is om bewerkingen te doorstaan zoals bijsnijden, filters, een andere frame rate of lossy compressie. NIST schrijft dat de meeste watermerkmethoden van nu steeds weer te verwijderen bleken.
    """)
    return


@app.cell(hide_code=True)
def choice_watermark_cell():
    choice_watermark = L.choice(
        "watermark",
        question="Spreken die twee elkaar tegen?",
        options={
            "yes": "Ja: wat NIST vond, geldt ook voor SynthID, dus de claim van Google klopt niet",
            "right": "Nee: bestand tegen bewerkingen is niet bestand tegen een aanvaller",
            "text": "Nee: NIST schreef alleen over watermerken in tekst, Google over plaatjes",
        },
        correct="right",
        why={
            "right": "Bestand tegen bijsnijden of een filter is niet bestand tegen iemand die het watermerk weg wil hebben. Google claimt het eerste voor zijn watermerk in plaatjes; NIST schrijft over verwijderen. Voor watermerken in tekst noemen de auteurs van het paper over SynthID-Text zelf aanvallen die het watermerk stelen, vervalsen of wegpoetsen, en herschrijven door een LLM. Vraag bij een claim over robuustheid waartegen er getest is.",
            "yes": "Google claimt niet dat zijn watermerk elke aanval overleeft, alleen de bewerkingen waarvoor het ontworpen is; een watermerk kan dat en toch sneuvelen tegen een aanvaller. Beide uitspraken kunnen kloppen. Zie *Watermerken* in *Het idee*.",
            "text": "Wat NIST vond, gaat over watermerkmethoden in het algemeen, niet alleen over tekst. Wat de twee uitspraken onderscheidt, is waar ze over gaan: gewone bewerkingen, of een aanvaller. Zie *Watermerken* in *Het idee*.",
        },
    )
    choice_watermark
    return (choice_watermark,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) was integriteit iets om te beschermen en spoofing een dreiging; met een digitale handtekening merkt iedereen die de public key heeft allebei op: een veranderd bericht, en een handtekening die niet van de genoemde ondertekenaar is. Ondertekenen maakt een plaatje niet waar: het vertelt je wie ervoor instaat. De volgende les, *Tekst van AI herkennen*, vraagt wat er overblijft als niemand tekende: tekst van AI herkennen aan de tekst alleen.

    **Een voorbeeld uit het dagelijks leven.** Een verzegelde pot in de winkel laat zien dat niemand hem sinds de fabriek openmaakte, en welke fabriek hem verzegelde. Het laat niet zien dat wat erin zit goed is; en een pot waarvan het zegel losliet, is daarom nog niet bedorven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_label, choice_watermark, result_genuine, result_sign, result_verdict):
    L.wrap_up(result_genuine, result_sign, result_verdict, choice_label, choice_watermark)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
