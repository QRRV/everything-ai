# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-secret-sharing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-secret-sharing", "nl")
    C = L.checks   # vijf verzonnen bedrijven: SECRETS, en SHARES, hun shares voor drie partijen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sends():
    quick_sends = L.choice(
        "quick-sends",
        question="Wat stuurt een gekozen telefoon in een ronde van FedAvg terug naar de server?",
        options={
            "data": "De eigen cijfers, om de server erop te laten trainen",
            "right": "De weights na het trainen",
            "score": "Alleen hoeveel cijfers die goed las",
        },
        correct="right",
        why={
            "right": "Elke gekozen telefoon laadt de weights van de server, traint op de eigen data, en stuurt alleen de nieuwe weights en het aantal voorbeelden terug; de cijfers blijven op de telefoon.",
            "data": "De cijfers verlaten de telefoon nooit: daar is federated learning juist voor. Zie *De data blijven op de telefoon* in *Het idee* van [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).",
            "score": "De server heeft weights nodig om te middelen, geen score. Zie *FedAvg: middel de weights* in *Het idee* van [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).",
        },
    )
    quick_sends
    return (quick_sends,)


@app.cell(hide_code=True)
def quick_check_weighted():
    quick_weighted = L.choice(
        "quick-weighted",
        question="Twee telefoons sturen 0,2 en 0,6 voor dezelfde weight. De eerste trainde op 100 voorbeelden, de tweede op 300. Wat maakt FedAvg ervan?",
        options={
            "plain": "0,4: het gewone gemiddelde van de twee weights",
            "right": "0,5: elk telt mee naar de voorbeelden",
            "big": "0,6: de weight van de grootste telefoon",
        },
        correct="right",
        why={
            "right": "Elke telefoon telt mee naar het aantal voorbeelden: (100 · 0,2 + 300 · 0,6) / 400 = 200 / 400 = 0,5.",
            "plain": "Het gewone gemiddelde telt beide telefoons even zwaar; FedAvg telt elke telefoon mee naar het aantal voorbeelden. Zie *FedAvg: middel de weights* in *Het idee* van [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).",
            "big": "De kleinere telefoon telt ook mee, voor een kwart van de voorbeelden. Zie *FedAvg: middel de weights* in *Het idee* van [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).",
        },
    )
    quick_weighted
    return (quick_weighted,)


@app.cell(hide_code=True)
def quick_check_server():
    quick_server = L.choice(
        "quick-server",
        question="Wat heeft de server na een ronde van FedAvg gezien?",
        options={
            "nothing": "Alleen het gemiddelde van de weights",
            "right": "De nieuwe weights van elke gekozen telefoon",
            "data": "De cijfers van elke telefoon en ook de weights",
        },
        correct="right",
        why={
            "right": "Elke gekozen telefoon stuurt de eigen nieuwe weights, en de server middelt die zelf. Hard e.a. schrijven dat gebruikers erop moeten vertrouwen dat de server de losse uploads van weights niet onder de loep neemt.",
            "nothing": "De server rekent het gemiddelde zelf uit, uit de weights die elke telefoon stuurt. Zie *Wat de weights nog verraden* in *Het idee* van [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).",
            "data": "De cijfers blijven op de telefoons; alleen weights reizen. Zie *De data blijven op de telefoon* in *Het idee* van [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl).",
        },
    )
    quick_server
    return (quick_server,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les hielden de telefoons hun cijfers, maar zag de server de weights van elke telefoon, en moesten gebruikers erop vertrouwen dat die er niet naar keek. Kan het ook zo dat de server alleen sommen te zien krijgt? Daar bestaat het gemiddelde van FedAvg uit. De vraag is ouder dan federated learning: kunnen een paar partijen samen iets uitrekenen zonder dat iemand de input van een ander ziet? In Boston deden 114 bedrijven dat in 2017 met hun loongegevens: de Boston Women's Workforce Council analyseerde de lonen van 166.705 werknemers, terwijl de eigen gegevens van elk bedrijf privé bleven. In deze les verdeel je getallen in shares die op willekeurige ruis lijken, tel je ze op zonder één getal te zien, en ontdek je waarom de willekeurige getallen uit het hele bereik moeten komen.

    **Een voorbeeld uit het dagelijks leven.** Drie vrienden willen hun gemiddelde salaris weten zonder het eigen salaris te noemen. De eerste bedenkt een groot willekeurig getal, telt het eigen salaris erbij op en fluistert de uitkomst tegen de tweede. De tweede telt het eigen salaris erbij op en fluistert de uitkomst tegen de derde, die hetzelfde doet en de uitkomst terugfluistert tegen de eerste. De eerste haalt het willekeurige getal eraf: wat overblijft is het totaal van de drie salarissen, en een derde daarvan is het gemiddelde. De eerste twee gefluisterde getallen klinken voor wie ze hoort als willekeurige getallen; het laatste vertelt de eerste alleen het totaal.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Shamir (1979) begint met een puzzel van Liu: elf wetenschappers sluiten hun documenten op in een kast die open moet gaan als, en alleen als, er zes of meer van hen bij zijn. Hoeveel sloten zijn daar met gewone sloten en sleutels minstens voor nodig?",
        options={
            "six": "6: één per wetenschapper die nodig is",
            "eleven": "11: één per wetenschapper",
            "right": "462",
        },
        reveal="**462 sloten**, met 252 sleutels voor elke wetenschapper: volgens Shamir is het niet moeilijk te laten zien dat de kleinste oplossing zoveel nodig heeft. Elke groep van vijf moet de sleutel missen van een slot dat elke zesde wetenschapper wel kan openen, en twee groepen van vijf kunnen zo'n slot niet delen, dus heeft elk van de 462 manieren om 5 van de 11 te kiezen een eigen slot nodig. Shamir deelt in plaats van sleutels een getal. In deze les gebruik je de eenvoudigste vorm, waarin alle shares nodig zijn.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een geheim in shares

    Adi Shamir (1979) beantwoordde de puzzel van de kast met getallen. Shamir laat zien hoe je gegevens D in n stukken verdeelt, zo dat elke k stukken D makkelijk teruggeven, terwijl zelfs alles weten over k − 1 stukken helemaal niets over D verraadt. Zo'n schema noemt Shamir een (k, n) threshold scheme. David Evans, Vladimir Kolesnikov en Mike Rosulek (2018) noemen de stukken shares: elke t − 1 shares zeggen niets over het geheim s, en elke t shares geven het helemaal terug. Een geheim zo verdelen heet **secret sharing**. Evans e.a. gebruiken vaak het geval waarin alle n shares nodig zijn en ook genoeg om het geheim terug te krijgen, en deze les doet dat ook.

    ### Shares die optellen

    Een eenvoudige manier telt op: Evans e.a. noemen eenvoudig optellend delen, additive sharing, een geschikte methode. Met hele getallen gaat dat zo. Kies een groot getal M: elke share is een heel getal van 0 tot M − 1. Trek alle shares op één na willekeurig uit dat hele bereik, met `random.randint(0, M - 1)`. De laatste share is wat er van het geheim over is, modulo M: het geheim min de andere shares, en dan de rest bij deling door M. In Python is die rest `%`: `104200 % 100000` is 4200, en ook bij een negatief getal is het een heel getal van 0 tot M − 1: `-75633 % 100000` is 24367. Denk aan een klok met M streepjes in plaats van 12: optellen gaat rond en rond, en alleen waar je uitkomt telt. Alle shares samen gaan de klok rond en komen uit bij het geheim.

    Eén share, of elke n − 1 van de n shares, is niets dan willekeurige getallen: wat het geheim ook is, elke rij waarden van 0 tot M − 1 is even waarschijnlijk. Volgens Evans e.a. is optellend delen veilig tegen n − 1 corrupte partijen, en Andrei Lapets en collega's (2018) beschrijven shares voor een breed publiek als stukjes die op zichzelf niet van willekeurige ruis te onderscheiden zijn. Dat klopt alleen als de willekeurige getallen uit het hele bereik komen: trek je ze uit een klein bereik, dan blijft de laatste share dicht bij het geheim, zoals je na Oefening 1 ziet.

    ### Optellen zonder te kijken

    Shares kun je share voor share optellen. Evans e.a. noemen dat additieve homomorfie: uit [x] en [y] en een openbaar getal z kunnen de partijen zonder berichten uit te wisselen [x + y], [x + z] en [xz] maken. Hier staat [x] voor x verdeeld in shares. Heeft partij 1 de eerste shares van x en y, en partij 2 de tweede, dan telt elk op wat die heeft, en de twee totalen zijn shares van x + y. Zo kunnen ook meerdere partijen elk één share van een heleboel geheimen optellen: samen geven hun totalen de som van alle geheimen, terwijl elke partij alleen getallen zag die op ruis lijken. Twee verdeelde getallen vermenigvuldigen is lastiger: daarvoor moeten de partijen volgens Evans e.a. met elkaar communiceren.

    Een uitkomst berekenen uit de input van een paar partijen zonder dat iemand de input van de anderen ziet, heet **secure multi-party computation**, kort MPC. De OECD (2023) omschrijft MPC als een verzameling hulpmiddelen waarmee de deelnemende partijen samen een functie van hun input berekenen en die input toch privé houden; zo is er geen vertrouwde derde partij meer nodig die de data inziet en beheert. Het verbergt de input, niet wat de uitkomst zelf zegt: met twee partijen is het totaal min je eigen getal het getal van de ander. Andrew Yao (1982) stelde de klassieke vraag: twee miljonairs willen weten wie de rijkste is, zonder per ongeluk iets meer over elkaars vermogen te weten te komen; hoe voeren ze zo'n gesprek? Yao loste het op met eenrichtingsfuncties, niet met secret sharing. Sindsdien is MPC volgens Evans e.a. uitgegroeid van een theoretische curiositeit in de jaren tachtig tot een middel om echte systemen mee te bouwen.

    ### Wie je moet vertrouwen

    De toepassingen in Boston en Estland hieronder gaan uit van partijen die het protocol volgen, maar wel kijken naar alles wat ze krijgen. Evans e.a. noemen zo'n tegenstander semi-honest: die volgt het protocol zoals het is voorgeschreven, maar probeert zoveel mogelijk te leren uit de berichten van de andere partijen. Zulke partijen heten ook vaak *honest-but-curious*. Waar het niet tegen beschermt: als de partijen hun shares samenleggen, liggen de geheimen op straat. Bij optellend delen over n partijen zijn daar alle n voor nodig.

    ### Waar het gebruikt wordt

    Volgens Evans e.a. geldt een veiling- en biedplatform voor contracten voor de teelt van suikerbieten, dat Deense onderzoekers samen met de Deense overheid en belanghebbenden bouwden, algemeen als de eerste commerciële toepassing van MPC. In Boston gebruikte Web-MPC, dat Lapets e.a. beschrijven, een variant van gewone additive secret sharing. De bedrijven stuurden hun gegevens in als shares, Boston University verzorgde de dienst, en de Boston Women's Workforce Council (BWWC) analyseerde de uitkomst, in 2015, 2016 en 2017. In 2017 ging dat om 166.705 werknemers van 114 bedrijven, ongeveer 16% van de beroepsbevolking van de regio Boston. Volgens de analyse van de BWWC verdienden witte vrouwen 75 cent voor elke dollar van een witte man; Aziatische, zwarte en Latijns-Amerikaanse vrouwen 71, 52 en 49 cent. Volgens Lapets e.a. past het *honest-but-curious*-model als de dienstverlener en de analist geen belang hebben bij samenspannen, bijvoorbeeld omdat beide willen dat het initiatief slaagt.

    In Estland zochten statistici in 2015 naar een verband tussen werken tijdens de studie en niet op tijd afstuderen. Dan Bogdanov en collega's (2016) beschrijven hoe dat liep op drie servers met additive secret sharing, met tien miljoen gegevens uit de belastingadministratie en een half miljoen uit het onderwijs; volgens de auteurs is het de grootste cryptografisch private statistische studie op echte data ooit. Over het alternatief, de gegevens anonimiseren, melden de auteurs dat 3-anonimiteit, afhankelijk van de demografische groep, 10 tot 30% van de steekproef kostte; 3-anonimiteit, waarbij elk record de identificerende waarden deelt met minstens twee andere, is de k-anonimiteit uit [Verstoppen in een groep: k-anonimiteit](/course/open/rai-k-anonymity?lang=nl) met k = 3. Shares veranderen niets aan de gegevens zelf, alleen aan wie ze kan zien. De studie vond volgens Evans e.a. geen verband tussen werken tijdens de studie en niet op tijd afstuderen, wel tussen meer opleiding en een hoger inkomen.

    ### Privacy-enhancing technologies

    De OECD (2023) brengt zulke methoden onder één naam: privacy-enhancing technologies (PET's) zijn digitale technieken en werkwijzen waarmee je informatie kunt verzamelen, verwerken, analyseren en delen, terwijl de vertrouwelijkheid van persoonsgegevens beschermd blijft. De OECD deelt ze in vier groepen in, en elke **privacy-enhancing technology** valt in een daarvan: data verhullen, versleuteld verwerken, federated en gedistribueerde analyse, en hulpmiddelen voor verantwoording. Federated learning, uit de vorige les, hoort bij federated en gedistribueerde analyse. MPC hoort bij versleuteld verwerken, naast homomorphic encryption (HE), dat rekent op versleutelde data. Threshold secret sharing, waarbij een vast aantal sleutels nodig is om versleutelde data te openen, is voor de OECD een hulpmiddel voor verantwoording: het digitale evenbeeld van een kluis met een paar losse sloten, waarvan de sleutels bij verschillende mensen liggen, zoals de kast van de elf wetenschappers. Over rekenen op shares schrijft de OECD dat technieken voor secret sharing of HE data van meerdere partijen kunnen samenvoegen en erop kunnen rekenen.

    De twee kunnen samengaan. In de vorige les wezen McMahan e.a. voor federated learning naar secure multi-party computation. Telefoons zouden hun weights, omgezet in hele getallen, kunnen verdelen in shares voor een paar servers: die servers zouden dan alleen sommen te weten komen, en het gemiddelde van FedAvg is gemaakt van sommen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **secret sharing** | een geheim verdelen in shares, zo dat ze samen (bij Shamir: elke k ervan) het teruggeven en minder ervan er niets over zeggen |
        | **share** | één stuk van een verdeeld geheim: op zichzelf een getal dat op willekeurige ruis lijkt |
        | **`%`**, modulo M | de rest bij deling door M, een heel getal van 0 tot M − 1: `104200 % 100000` is 4200; zoals een klok met M streepjes |
        | **secure multi-party computation** (MPC) | methoden waarmee een paar partijen een uitkomst berekenen uit hun input, zonder dat iemand de input van de anderen ziet |
        | **honest-but-curious** | een partij die het protocol volgt, maar kijkt naar alles wat die krijgt; ook semi-honest genoemd |
        | **privacy-enhancing technology** (PET) | een techniek om data te verzamelen, te gebruiken of te delen terwijl de vertrouwelijkheid van persoonsgegevens beschermd blijft, zoals federated learning of MPC |
        """),
        L.watch_out("Een losse share zegt alleen niets als de willekeurige getallen uit het hele bereik 0 tot M − 1 komen. En modulo M geven de shares alleen een getal onder M terug: kies M groter dan elk totaal dat je uitrekent, anders gaat de som de klok rond en komt er iets verkeerds uit."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een som van geheimen uit te rekenen zonder dat iemand er een ziet, met additive secret sharing:

    1. Kies M groter dan elk totaal dat je uitrekent. Verdeel elk geheim: willekeurige getallen van 0 tot M − 1 voor alle shares op de laatste na, en als laatste share het geheim min die getallen, modulo M. Dat is `share`.
    2. Alle shares van een geheim, opgeteld modulo M, geven het terug; minder ervan zeggen niets. Dat is `reveal`.
    3. Geef share j van elk geheim aan partij j. Elke partij telt de eigen shares op, modulo M. Dat is `party_totals`.
    4. De partijen leggen hun totalen samen: `reveal` op de totalen geeft de som van alle geheimen. Alleen de som komt naar buiten, zolang de partijen hun shares niet samenleggen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel laat zien wat `%` doet, verdeelt twee geheimen, x en y, in elk twee shares, en laat twee partijen elk hun eigen shares optellen.
    """)
    return


@app.cell
def example_shares():
    import random

    M = 100000                     # elke share is een heel getal van 0 tot M - 1
    print(104200 % M, -75633 % M)  # % geeft de rest bij deling door M

    random.seed(1979)
    x = 4200
    x_first = random.randint(0, M - 1)   # een willekeurig getal van 0 tot M - 1
    x_second = (x - x_first) % M         # wat er van x over is, modulo M
    print("shares of x:", x_first, x_second, "together:", (x_first + x_second) % M)

    y = 1300
    y_first = random.randint(0, M - 1)
    y_second = (y - y_first) % M
    print("shares of y:", y_first, y_second, "together:", (y_first + y_second) % M)

    # partij 1 heeft beide eerste shares, partij 2 beide tweede; elk telt op wat die heeft
    total_1 = (x_first + y_first) % M
    total_2 = (x_second + y_second) % M
    print("party totals:", total_1, total_2, "together:", (total_1 + total_2) % M)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `104200 % M` is `4200`, en `-75633 % M` is `24367`. De shares van x, `11351` en `92849`, lijken in niets op 4200, maar ze tellen op tot `104200`, en dat is modulo M `4200`; voor y net zo. Partij 1 heeft `11351` en `69401`, partij 2 heeft `92849` en `31899`: getallen die op zichzelf niets zeggen. Hun totalen, `80752` en `24748`, tellen op tot `105500`, en dat is modulo M `5500`: de som van x en y, terwijl geen van beide partijen x of y zag. In Oefening 1 verdeel je een geheim in een willekeurig aantal shares.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. share: een geheim verdelen

        `share(secret, randoms, M)` krijgt een heel getal `secret` van 0 tot M − 1, een lijst `randoms` met hele getallen van 0 tot M − 1, en het getal `M`. De functie geeft de lijst `shares` terug: de willekeurige getallen, in hun volgorde, en daarna de laatste share, wat er van `secret` over is als alle `randoms` eraf zijn, modulo `M`. Binnenin begint `left` bij `secret`, en de loop zet elk willekeurig getal `r` in `shares`. Vervang de twee `...`. De check roept `share(4200, [71320, 8513], 100000)` aan, en dan andere geheimen en willekeurige getallen.
        """),
        L.task("Geef de willekeurige getallen terug, en daarna de share die ze samen modulo M op het geheim laat uitkomen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=9)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Stap 1 van het recept. In het uitgewerkte voorbeeld is `x_second` wat er van `x` over is als `x_first` eraf is, modulo `M`.",
            "Hier kunnen er zoveel willekeurige getallen zijn als je wilt. De loop zet ze al stuk voor stuk in `shares`; haal elk ook af van `left`, zodat `left` na de loop het geheim min al die getallen is. De laatste share moet een heel getal van 0 tot M − 1 zijn, net als de andere.",
            "`left = left - r` en `shares.append(___)`",
        ],
        why="`left = left - r` haalt elk willekeurig getal eraf, en `shares.append(left % M)` voegt na de loop de laatste share toe. `share(4200, [71320, 8513], 100000)` geeft `[71320, 8513, 24367]`: samen `104200`, en `104200 % 100000` is `4200`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat één share verraadt

    Het plaatje verdeelt twee geheimen, 20.000 en 70.000, elk 200 keer in twee shares, met jouw `share` zodra Oefening 1 klopt, en tot dan met die van de course. Elke keer wordt het willekeurige getal, de eerste share, getrokken uit 0 tot W − 1. Elke stip is de laatste share van één keer verdelen: gevuld voor 20.000, open voor 70.000, met een gestreepte lijn bij het geheim. De eerste shares hebben geen plaatje nodig: dat zijn willekeurige getallen van 0 tot W − 1, wat het geheim ook is. Met de schuif kies je W.
    """)
    return


@app.cell(hide_code=True)
def demo_leak(box_share, result_share):
    _yours = C.mine(box_share.value, result_share.state == "correct")
    _runs = C.leak_runs(_yours) if _yours else None
    _who = "jouw `share`"
    if _runs is None:
        _who = "de `share` van de course, omdat die van jou hier een fout geeft" if _yours else "de `share` van de course, omdat Oefening 1 nog niet klopt"
        _runs = C.leak_runs()
    _out = mo.vstack([
        mo.md(f"Gemaakt met {_who}."),
        L.explore("leak", "W", lambda _w: C.leak_frame(_runs, int(_w), L.lang), start=1000, values=C.WIDTHS, role="",
                  task="Sleep W van 1000 naar 100.000: vanaf welke W kan een laatste share de twee geheimen niet meer uit elkaar houden?"),
    ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_reveal():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. reveal: een geheim weer samenvoegen

        `reveal(shares, M)` krijgt een lijst `shares` met hele getallen en het getal `M`. De functie geeft de shares opgeteld terug, modulo `M`. Binnenin begint `total` bij 0, en de loop gaat de shares langs, één share `s` per keer. Vervang de twee `...`. De check roept `reveal([71320, 8513, 24367], 100000)` aan, en dan andere shares.
        """),
        L.task("Geef de shares opgeteld terug, modulo M."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reveal():
    box_reveal = L.code_box("reveal", lines=6)
    box_reveal
    return (box_reveal,)


@app.cell(hide_code=True)
def check_reveal(box_reveal):
    result_reveal = L.check("reveal", box_reveal.value)
    result_reveal
    return (result_reveal,)


@app.cell(hide_code=True)
def hints_reveal():
    L.hints(
        "reveal",
        [
            "Stap 2 van het recept. In het uitgewerkte voorbeeld voegt `(x_first + x_second) % M` x weer samen uit de twee shares.",
            "Hier kunnen er zoveel shares zijn als je wilt: laat de loop elke share bij `total` optellen, zodat `total` na de loop ze allemaal opgeteld bevat. Die som kan M of meer zijn.",
            "`total = ___` en `return total % M`",
        ],
        why="`total = total + s` telt de shares op, en `return total % M` neemt de rest. `reveal([71320, 8513, 24367], 100000)` geeft `4200`; zonder de laatste share geeft `reveal([71320, 8513], 100000)` `79833`, dat er in niets op lijkt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_party_totals():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. party_totals: wat elke partij optelt

        `SECRETS` bevat wat vijf verzonnen bedrijven hun vrouwelijke werknemers in een jaar betalen, in duizenden dollars: `[3120, 870, 12450, 5300, 760]`. Elk bedrijf verdeelde het eigen getal met de `share` van de course in drie shares, één voor elk van drie partijen. `SHARES` bevat ze: `SHARES[i][j]` is share j van bedrijf i, en partij j heeft die. Tellen begint bij 0, dus de partijen zijn 0, 1 en 2.

        `party_totals(table, M)` krijgt zo'n tabel, een lijst `table` met één rij per bedrijf en in elke rij één share per partij, en het getal `M`. De functie geeft de lijst `totals` terug: voor elke partij j de eigen shares opgeteld, modulo `M`. Binnenin gaat de loop over `j` de partijen langs; voor elke partij begint `total` bij 0, en de loop over de rijen neemt telkens één rij `row`. Vervang de twee `...`. De check roept `party_totals(SHARES, 100000)` aan, en dan andere tabellen.
        """),
        L.task("Geef voor elke partij de eigen shares opgeteld terug, modulo M."),
    ])
    return


@app.cell(hide_code=True)
def exercise_party_totals():
    box_party_totals = L.code_box("party-totals", lines=9)
    box_party_totals
    return (box_party_totals,)


@app.cell(hide_code=True)
def check_party_totals(box_party_totals):
    result_party_totals = L.check("party-totals", box_party_totals.value)
    result_party_totals
    return (result_party_totals,)


@app.cell(hide_code=True)
def hints_party_totals():
    L.hints(
        "party-totals",
        [
            "Stap 3 van het recept. In het uitgewerkte voorbeeld telt partij 1 `x_first` en `y_first` op: de eerste share van elk geheim.",
            "Partij j heeft kolom j: uit de rij van elk bedrijf de share op plek j. De binnenste loop gaat de rijen langs; tel die share van elke rij op bij `total`. Na die loop krijgt `totals` de rest van `total` bij deling door `M`.",
            "`total = total + ___` en `totals.append(total % M)`",
        ],
        why="`total = total + row[j]` telt kolom j op, en `totals.append(total % M)` houdt het totaal als share. `party_totals(SHARES, 100000)` geeft `[38021, 53499, 30980]`; opgeteld, modulo 100.000, is dat `22500`, de som van de vijf geheimen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two_companies():
    mo.md(r"""
    ### Oefening 4. Twee bedrijven

    Er doen maar twee bedrijven mee aan een verdeling zoals die van Oefening 3, en de partijen maken het totaal van hun twee getallen bekend.
    """)
    return


@app.cell(hide_code=True)
def choice_two_companies_cell():
    choice_two_companies = L.choice(
        "two-companies",
        question="Wat kan elk bedrijf uit het totaal afleiden?",
        options={
            "right": "Het getal van de ander: het totaal min het eigen",
            "nothing": "Niets: de shares verborgen elk getal",
            "half": "Alleen dat het getal van de ander ongeveer de helft is",
        },
        correct="right",
        why={
            "right": "Secure multi-party computation verbergt de input, niet wat de uitkomst zelf zegt: met twee bedrijven is het totaal min je eigen getal het getal van de ander. Met 114 bedrijven, zoals in Boston in 2017, is het totaal min je eigen getal de som van de andere 113.",
            "nothing": "De shares verborgen de getallen onderweg, maar het totaal is de uitkomst, en die zegt wat die zegt: het totaal min je eigen getal is dat van de ander. Zie *Optellen zonder te kijken* in *Het idee*.",
            "half": "Het totaal min je eigen getal is precies het getal van de ander, geen schatting. Zie *Optellen zonder te kijken* in *Het idee*.",
        },
    )
    choice_two_companies
    return (choice_two_companies,)


@app.cell(hide_code=True)
def text_exercise_collude():
    mo.md(r"""
    ### Oefening 5. Twee partijen samen

    Stel dat Boston precies het schema van deze les had gebruikt, met twee partijen: elk bedrijf verdeelt het eigen getal in twee shares, één voor Boston University en één voor de BWWC.
    """)
    return


@app.cell(hide_code=True)
def choice_collude_cell():
    choice_collude = L.choice(
        "collude",
        question="Wat kunnen die twee te weten komen door samen te leggen wat elk heeft?",
        options={
            "total": "Nog steeds alleen de totalen",
            "right": "Het eigen getal van elk bedrijf",
            "noise": "Alleen ruis, want elke share lijkt op ruis",
        },
        correct="right",
        why={
            "right": "Samen hebben de twee beide shares van elk bedrijf, en beide shares van een getal, opgeteld modulo M, geven het terug. Het honest-but-curious-model vertrouwt erop dat de twee dat niet doen; volgens Lapets e.a. past het als de twee geen belang hebben bij samenspannen.",
            "total": "De totalen worden met opzet uitgerekend, maar samen hebben de twee ook beide shares van elk bedrijf, en die geven het getal van elk bedrijf terug. Zie *Wie je moet vertrouwen* in *Het idee*.",
            "noise": "Eén share alleen lijkt op ruis, maar alle shares van een getal samen geven het terug, en de twee samen hebben die allemaal. Zie *Wie je moet vertrouwen* in *Het idee*.",
        },
    )
    choice_collude
    return (choice_collude,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Trainen zonder de data te verzamelen](/course/open/rai-federated-learning?lang=nl) hielden de telefoons hun data, maar zag de server de weights van elke telefoon. Hier rekenen een paar partijen een som uit zonder dat iemand ook maar één input ziet: shares die op ruis lijken, share voor share opgeteld, geven alleen het totaal terug. Net als federated learning is het een privacy-enhancing technology, en de twee kunnen samengaan. In de volgende les, *Wat een model onthoudt*, gaat de vraag over het model zelf: wat een getraind model kan verraden over de data waarvan het leerde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_collude, choice_two_companies, result_party_totals, result_reveal, result_share):
    L.wrap_up(result_share, result_reveal, result_party_totals, choice_two_companies, choice_collude)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
