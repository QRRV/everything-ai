# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-secrets-and-endpoints. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-secrets-and-endpoints", "nl")
    C = L.checks   # de verzonnen repository van de winkel met twee keys in de geschiedenis, adressen, tokens, de server


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl) en [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_uber():
    quick_uber = L.choice(
        "quick-uber",
        question="Waar vonden indringers in 2016 de toegangssleutel tot bestanden van Uber, volgens een klacht van de Amerikaanse Federal Trade Commission?",
        options={
            "right": "In code in een privérepository op GitHub",
            "public": "In een openbare repository op GitHub",
            "laptop": "Op een laptop die een medewerker in de trein liet liggen",
        },
        correct="right",
        why={
            "right": "Volgens de klacht stond de sleutel leesbaar in code in een privérepository op GitHub. Wie in de repository kwam, had de sleutel.",
            "public": "De repository was privé; de indringers zeiden dat ze binnenkwamen met wachtwoorden die bij eerdere datalekken waren uitgelekt. Zie *Geheimen in de code* in [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl).",
            "laptop": "De sleutel stond in code, in een privérepository op GitHub. Zie *Geheimen in de code* in [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl).",
        },
    )
    quick_uber
    return (quick_uber,)


@app.cell(hide_code=True)
def quick_check_environ():
    quick_environ = L.choice(
        "quick-environ",
        question="Je code leest `os.environ[\"MODEL_API_KEY\"]`, en die environment variable is niet gezet. Wat gebeurt er?",
        options={
            "right": "Python stopt met een KeyError",
            "empty": "De key is de lege tekst",
            "none": "De waarde is None, en het request gaat zonder key de deur uit",
        },
        correct="right",
        why={
            "right": "`os.environ` bevat de environment variables als dictionary, en een naam opzoeken die niet in een dictionary staat, geeft een KeyError.",
            "empty": "Met vierkante haken is een naam die ontbreekt een KeyError, zoals in elke dictionary. Zie *Nieuwe woorden en tekens* in [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl).",
            "none": "Met vierkante haken is een naam die ontbreekt een KeyError, zoals in elke dictionary; het request wordt nooit verstuurd. Zie *Nieuwe woorden en tekens* in [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl).",
        },
    )
    quick_environ
    return (quick_environ,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een key in code is een key voor iedereen die de code kan lezen, en een server is een deur voor iedereen die hem kan bereiken. Allebei komt veel voor. Michael Meli, Matthew McNiece en Bradley Reaves doorzochten in 2017 en 2018 het openbare deel van GitHub, en keken alleen naar bestanden met private keys en naar de keys van 11 diensten waarvan de keys een eigen vorm hebben. Ze vonden 201.642 verschillende geheimen, in meer dan 100.000 repository's; ongeveer twee weken nadat ze ze vonden, stond 81% er nog. Het beveiligingsbedrijf GitGuardian telde in 2025 28,65 miljoen nieuwe geheimen in openbare commits op GitHub, 34% meer dan in 2024; de geheimen van AI-diensten die het vond, stegen met 81%, naar 1.275.105. En onderzoekers van Cisco zochten in 2025 naar servers van Ollama, een programma dat LLM's draait, die vanaf internet bereikbaar waren: ze vonden er 1139, en 214 daarvan antwoordden met een model.

    **Een voorbeeld uit het dagelijks leven.** Een reservesleutel onder de deurmat. Wie weet waar de sleutel ligt, kan naar binnen, en juist de mensen die je het minst binnen wilt hebben, weten waar ze moeten kijken.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Meli, McNiece en Reaves zetten steeds opnieuw een eigen tekst in een openbare commit op GitHub, en maten hoe snel de zoekfunctie van GitHub hem liet zien. Wat was de mediaan?",
        options={
            "right": "20 seconden",
            "minutes": "15 minuten",
            "day": "Een hele dag",
        },
        reveal="**20 seconden.** De onderzoekers schrijven dat de mediaan 20 seconden was, met tijden van een halve seconde tot ruim 4 minuten, en dat het tijdstip van de dag er niet toe deed. Zo snel kan iemand die de zoekfunctie van GitHub in de gaten houdt een nieuwe key zien; een aanvaller hebben ze niet gemeten.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Het aanvalsoppervlak

    NIST noemt het **aanvalsoppervlak** de punten aan de rand van een systeem, een onderdeel of een omgeving waar een aanvaller kan proberen binnen te komen, iets te veroorzaken of gegevens naar buiten te halen. Voor de hulpassistent van Spoke & Chain keken de vorige lessen naar één punt: de tekst die het model leest. Deze les kijkt naar nog twee: de keys in de code, en de server waarop hij draait. De cheat sheet van OWASP over geheimen raadt aan om geheimen mee te nemen als deel van het aanvalsoppervlak als je een dreigingsmodel maakt. Een punt dat je weghaalt, hoef je niet meer te verdedigen: een key die niet in de code staat, kan niet uit de code lekken.

    ### Keys uit de code

    Een key in de code gaat overal mee waar de code heen gaat. *The Twelve-Factor App*, een verzameling regels voor apps van Adam Wiggins, maakt er een test van: je configuratie staat goed buiten je code als je de code op elk moment openbaar zou kunnen maken zonder dat er inloggegevens uitlekken. De regel die daarbij hoort: zet configuratie in environment variables, zoals [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl) deed met `os.environ`. Niet iedereen vindt dat genoeg. In het deel over containers waarschuwt de cheat sheet van OWASP over geheimen dat environment variables meestal te lezen zijn voor alle processen, en in logs of geheugendumps terecht kunnen komen; hij raadt ze alleen aan als andere manieren niet kunnen. Over één ding zijn ze het eens: niet in de code.

    ### Secret scanning

    **Secret scanning** is code en de geschiedenis ervan doorzoeken op teksten met de vorm van een key, een wachtwoord of een token. Veel diensten geven hun keys een eigen vorm, zoals een vast begin en een vaste lengte, en naar die vorm zoekt een scanner. Meli en collega's deden hetzelfde: ze zochten naar bestanden met private keys en naar keys van 11 belangrijke diensten met een herkenbare vorm. *Secret scanning* van GitHub doorzoekt de hele geschiedenis van alle branches van een repository op inloggegevens die in de code staan. Het draait vanzelf alleen bij openbare repository's; privérepository's van een organisatie hebben er *GitHub Secret Protection* voor nodig, bij een abonnement *GitHub Team* of *Enterprise Cloud*. *Push protection* komt een stap eerder: het houdt een push met een geheim erin tegen voordat het in de repository komt. Voor je eigen account staat het standaard aan, maar dan houdt het alleen pushes naar openbare repository's tegen. Voor een repository staat het standaard uit en is *GitHub Secret Protection* nodig, en standaard kan iedereen met schrijfrechten op de repository push protection omzeilen met een reden, zoals *I'll fix it later*.

    Een commit die een key weghaalt, haalt de commit die hem toevoegde niet weg: de key blijft in de geschiedenis staan. Daarom is het eerste advies van GitHub bij een key die een scanner vond hetzelfde als in [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl): vervang de inloggegevens meteen, zodat niemand er nog mee binnenkomt. Meli en collega's kregen zelfs commits terug die met het herschrijven van de geschiedenis waren weggehaald, op GitHub zoals het in 2018 werkte.

    ### Open deuren

    Een server luistert op een adres en een poort, en het adres bepaalt wie hem kan bereiken. Adressen die met 127 beginnen, zijn volgens RFC 1122, de regels voor hosts op internet uit 1989, de eigen adressen van een host, en ze mogen nooit buiten die host verschijnen. Een server op 127.0.0.1 is dus alleen te bereiken vanaf de computer waarop hij draait, en `localhost` is de naam van dat adres. Voor een server betekent het adres 0.0.0.0, of de lege tekst in Python, het omgekeerde: elk adres van de computer, ook het adres waarop het netwerk hem bereikt. `python -m http.server` van Python zelf doet dat: volgens de documentatie luistert de server standaard op alle netwerkinterfaces.

    Ollama luistert standaard op 127.0.0.1, poort 11434; de FAQ laat zien hoe de environment variable `OLLAMA_HOST` dat verandert, bijvoorbeeld in 0.0.0.0. Wiz Research schreef in 2024 dat het image van Ollama voor Docker van buitenaf bereikbaar is, en dat Ollama zelf geen authenticatie heeft: het vraagt om geen key of token. Wiz vond meer dan 1000 servers van Ollama die open stonden. Het vond ook een gat in Ollama voor versie 0.1.34 (CVE-2024-37032) waarmee een aanvaller bestanden op de server kon overschrijven, en dat volgens Wiz verder te misbruiken was tot het uitvoeren van willekeurige code op afstand. Luisteren op localhost maakt het risico van aanvallen op afstand volgens Wiz flink kleiner. Het advies van Cisco uit 2025: zet een server met een LLM nooit open voor iedereen zonder authenticatie met een veilige key of token.

    ### Vraag om een token

    Een server die anderen moeten bereiken, kan om een token vragen, zoals de API's in [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl) om je key vroegen. De gebruikelijke vorm is de header `Authorization: Bearer` met het token erachter, uit RFC 6750 (2012). Die noemt zo'n token een *bearer token*: iedereen die het token heeft, kan het gebruiken op elke manier waarop een ander die het heeft dat kan. Wie het heeft, komt binnen, dus het moet net zo geheim blijven als een key, en de RFC maakt TLS, de versleuteling achter https, verplicht. Op de server vergelijkt de controle de header met het token dat de server verwacht.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **aanvalsoppervlak** | de punten van een systeem waar een aanvaller kan proberen binnen te komen, iets te veroorzaken of gegevens naar buiten te halen |
        | **secret scanning** | code en de geschiedenis ervan doorzoeken op teksten met de vorm van een key, een wachtwoord of een token |
        | push protection | een controle voor een push die een commit met een geheim erin tegenhoudt |
        | `127.0.0.1`, `localhost` | het adres en de naam van de computer zelf; een server daar is alleen vanaf die computer te bereiken |
        | `0.0.0.0` | voor een server: elk adres van de computer, ook het adres waarop het netwerk hem bereikt |
        | bearer token | een token dat iedereen binnenlaat die het meestuurt, in de header `Authorization: Bearer` |
        | `rest.isalnum()` | True als de tekst `rest` niet leeg is en alleen letters en cijfers bevat |
        """),
        L.watch_out("Een key weghalen in een nieuwe commit haalt hem niet uit de geschiedenis, en een key die ooit gepusht is, kan al gekopieerd zijn. Trek hem eerst in, en maak dan een nieuwe."),
        mo.Html(C.surface_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_repository():
    mo.md(r"""
    ### De repository van de winkel

    De course verzon een repository voor de hulpassistent van Spoke & Chain, met acht commits. De assistent gebruikt een verzonnen betaaldienst, PedalPay, waarvan de keys met `pp_live_` beginnen, gevolgd door 24 letters en cijfers. De scanner van de course knipt elk bestand in woorden, bij spaties, nieuwe regels en de tekens `= " ' ( ) , :`, en vraagt van elk woord of het de vorm van een key heeft. Het plaatje laat zien wat hij in elke commit vindt.
    """)
    return


@app.cell(hide_code=True)
def picture_history():
    mo.Html(C.history_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_history():
    mo.md(r"""
    In b7d2f41 kwam de key in `refunds.py` te staan; een commit later las c90e5a2 hem in plaats daarvan uit de environment. De nieuwste versie bevat geen key, maar de geschiedenis nog twee: de tweede kwam mee in notities over een terugbetaling die misging, met de header zoals hij verstuurd was, en werd een commit later weer weggehaald. In de README staat `pp_live_your_key_here`, een tekst die begint als een key maar er geen is: het stuk na `pp_live_` is te kort, en bevat `_`. En commit 0a9e4c1 zette een deur van een andere soort open: de server luisterde vanaf toen op 0.0.0.0, zodat de laptops van de winkel de assistent konden gebruiken.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Breng het aanvalsoppervlak in kaart: de code en de geschiedenis ervan, de keys, elke server en het adres waarop hij luistert, en de tekst die een model leest.
    2. Houd keys uit de code: lees ze uit een environment variable of een dienst voor het beheer van geheimen.
    3. Scan voor elke push, en scan één keer de hele geschiedenis: zoek naar de vormen van de keys die je gebruikt.
    4. Trek een key in die ooit gepusht is, en maak dan een nieuwe; hem uit de code halen is niet genoeg.
    5. Laat een server op 127.0.0.1 luisteren, tenzij andere computers hem moeten bereiken; moeten ze dat, vraag dan om een token, via https.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.start_server(token)` start de server van de course: de `ThreadingHTTPServer` van Python op `("127.0.0.1", port)`, met een vrije poort, die om `token` vraagt in de header Authorization. `C.status(url, header)` stuurt hem een request voor zijn modellen, met `header` in Authorization of zonder header bij `None`, en geeft de statuscode van het antwoord terug: 200 als hij antwoordt, 401 als hij weigert. De cel start de server met het token `C.TOKEN`, stuurt drie requests, en `_server.stop()` stopt hem weer.
    """)
    return


@app.cell
def example_server():
    # de server van de course, alleen op deze computer, die om een token vraagt
    _server = C.start_server(C.TOKEN)
    print("listens on", _server.host)
    for _header in [None, "Bearer wrong-token", "Bearer " + C.TOKEN]:
        print(_header, "->", C.status(_server.url, _header))
    _server.stop()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De server luistert op 127.0.0.1, dus geen andere computer kan hem bereiken. Zonder token, en met een verkeerd token, antwoordt hij 401; alleen het goede token, `shop-token-4417-chain`, krijgt antwoord. Alles draaide op je eigen computer, en de server is weer gestopt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_key():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. looks_like_key: de vorm van een key

        Schrijf `looks_like_key(word)`: True als `word` de vorm van een key van PedalPay heeft, `pp_live_` en dan precies 24 letters en cijfers; anders False. `word.startswith(start)` geeft True als `word` met `start` begint, en `rest.isalnum()` geeft True als `rest` niet leeg is en alleen letters en cijfers bevat. Vervang de twee `...`: geef eerst False terug als `word` niet met `pp_live_` begint; controleer dan `rest`, het stuk daarna. De check roept `looks_like_key(KEY)` aan, waar `KEY` de key uit commit b7d2f41 is, en daarna lastige woorden; het plaatje eronder laat je functie ook op de geschiedenis van de winkel los.
        """),
        L.task("Geef alleen True terug voor pp_live_ gevolgd door precies 24 letters en cijfers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_key():
    box_key = L.code_box("looks-like-key", lines=6)
    box_key
    return (box_key,)


@app.cell(hide_code=True)
def check_key(box_key):
    result_key = L.check("looks-like-key", box_key.value)
    result_key
    return (result_key,)


@app.cell(hide_code=True)
def hints_key():
    L.hints(
        "looks-like-key",
        [
            "Stap 3 van het recept: een scanner zoekt naar de vorm van een key. Die vorm heeft drie delen: het begin, de lengte van de rest, en de tekens erin.",
            "De eerste `...` wordt een `if` met `not` en `word.startswith`, met `return False` eronder. De laatste regel geeft terug of `rest` het goede aantal tekens heeft en `rest.isalnum()` geldt, met `and` ertussen.",
            "`    return ___ == 24 and rest.isalnum()`",
        ],
        why="`if not word.startswith(\"pp_live_\"):` met `return False` eronder, en aan het eind `return len(rest) == 24 and rest.isalnum()`. De tekst in de README heeft 13 tekens na `pp_live_`, waarvan twee `_`, dus het is geen key.",
    )
    return


@app.cell(hide_code=True)
def text_after_key():
    mo.md(r"""
    Beide keys in de geschiedenis worden gevonden, en de tekst in de README niet. Een scanner die de vorm van een key kent, heeft geen lijst met de keys zelf nodig, maar hij vindt alleen de vormen die hij kent: een wachtwoord, of een key van een dienst zonder eigen vorm voor zijn keys, glipt erdoor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_local():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. local_only: wie de server kan bereiken

        Schrijf `local_only(host)`: True als een server die op `host` luistert alleen te bereiken is vanaf de computer waarop hij draait. `localhost` en `::1`, de naam en het IPv6-adres van de computer zelf, staan al in de code. Vervang de `...` na `return`: welke andere adressen bij de computer zelf horen, staat bij *Open deuren* in *Het idee*. De check roept `local_only("127.0.0.1")` aan, en daarna de adressen in het plaatje eronder.
        """),
        L.task("Geef alleen True terug voor de adressen die een server op deze computer houden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_local():
    box_local = L.code_box("local-only", lines=5)
    box_local
    return (box_local,)


@app.cell(hide_code=True)
def check_local(box_local):
    result_local = L.check("local-only", box_local.value)
    result_local
    return (result_local,)


@app.cell(hide_code=True)
def hints_local():
    L.hints(
        "local-only",
        [
            "Stap 5 van het recept, en *Open deuren* in *Het idee*: welke adressen horen bij de computer zelf?",
            "Volgens RFC 1122 is elk adres dat met 127 begint een adres van de computer zelf, en `host.startswith(start)` kijkt hoe `host` begint. `0.0.0.0` en de lege tekst zijn alle adressen tegelijk, dus niet lokaal.",
            "`    return host.startswith(___)`",
        ],
        why="`return host.startswith(\"127.\")`. `0.0.0.0` en `\"\"` geven False: een server daar luistert op elk adres, ook op het adres dat het netwerk bereikt.",
    )
    return


@app.cell(hide_code=True)
def text_after_local():
    mo.md(r"""
    `10.0.0.127` eindigt op 127 en is een adres op een netwerk; `127.0.0.2` is net zo goed de computer zelf als `127.0.0.1`. `0.0.0.0` en de lege tekst lijken niets, en zijn de ruimste keuze van allemaal: alle adressen tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_authorized():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. authorized: geen token, geen toegang

        De server van de winkel leest zijn token met `os.environ.get("SHOP_TOKEN", "")`, die de waarde van de environment variable `SHOP_TOKEN` geeft, of `""` als die niet gezet is. Bij elk request roept hij `authorized(header, key)` aan, met de header Authorization van het request (`""` als die er niet is) en dat token als `key`. Schrijf `authorized`: alleen True als `header` gelijk is aan `"Bearer "` gevolgd door het token van de server, en nooit als de server geen token heeft. De laatste regel vergelijkt de header al. Vervang de `...`. De check roept `authorized("Bearer " + TOKEN, TOKEN)` aan, waar `TOKEN` het token uit het uitgewerkte voorbeeld is, en daarna andere headers, en een server waarvan het token ontbreekt.
        """),
        L.task("Weiger elk request als de server geen token heeft; vergelijk anders de header."),
    ])
    return


@app.cell(hide_code=True)
def exercise_authorized():
    box_authorized = L.code_box("authorized", lines=5)
    box_authorized
    return (box_authorized,)


@app.cell(hide_code=True)
def check_authorized(box_authorized):
    result_authorized = L.check("authorized", box_authorized.value)
    result_authorized
    return (result_authorized,)


@app.cell(hide_code=True)
def hints_authorized():
    L.hints(
        "authorized",
        [
            "Wat geeft de laatste regel terug als de server geen token heeft en de header `\"Bearer \"` is, met niets erachter?",
            "Voor de laatste regel komt een `if` op `key`, met `return False` eronder. Wat `key` is als `SHOP_TOKEN` niet gezet is, staat in de tekst van de oefening.",
            "`    if key == ___:`",
        ],
        why="`if key == \"\":` met `return False` eronder. Zonder die regels geeft `authorized(\"Bearer \", \"\")` True: `\"Bearer \" + \"\"` is alleen `\"Bearer \"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_authorized():
    mo.md(r"""
    Zonder de eerste regels zou een server die zonder `SHOP_TOKEN` gestart is elk request binnenlaten met als header `Bearer ` en niets erachter: één vergeten instelling, en de deur staat open. Met die regels betekent een ontbrekend token dat niemand binnenkomt, en zie je de fout meteen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Voor de push

    Typ of plak een commit in het vak. Het plaatje laat de scanner er regel voor regel op los, zoals een controle voor een push dat zou doen, met jouw `looks_like_key` zodra Oefening 1 klopt. Probeer een key met een letter minder, of de tekst uit de README. Wat je typt, blijft op je computer.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    demo_commit = mo.ui.text_area(value=C.DEMO_TEXT, label="Je commit", rows=5, full_width=True)
    demo_commit
    return (demo_commit,)


@app.cell(hide_code=True)
def demo_push(box_key, demo_commit, result_key):
    _fn = C.mine(box_key.value, result_key.state == "correct", "looks_like_key")
    _out = mo.Html(C.push_picture(demo_commit.value, L.lang, _fn))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_reach():
    mo.md(r"""
    ### Oefening 4. Lokaal, maar voor wie?

    Na commit 1c7f2e5 luistert de modelserver van de assistent weer op 127.0.0.1, en vraagt hij om geen token.
    """)
    return


@app.cell(hide_code=True)
def choice_reach_cell():
    choice_reach = L.choice(
        "reach",
        question="Wie kan er nog requests naartoe sturen?",
        options={
            "right": "Elk programma op de server van de winkel zelf",
            "nobody": "Niemand behalve de assistent: één programma per adres",
            "network": "Elke computer op het netwerk van de winkel",
        },
        correct="right",
        why={
            "right": "Adressen die met 127 beginnen, mogen volgens RFC 1122 nooit buiten de host verschijnen, dus andere computers bereiken hem niet; maar elk programma en elke gebruiker op dezelfde computer wel. Volgens Wiz maakt luisteren op localhost het risico van aanvallen op afstand flink kleiner: het risico vanaf het netwerk, niet alles. Een token houdt ook de programma's op dezelfde computer buiten die het niet hebben.",
            "nobody": "Een adres hoort niet bij één programma: elk programma op de computer kan een request naar 127.0.0.1 sturen. Zie *Open deuren* in *Het idee*.",
            "network": "127.0.0.1 is de computer zelf, en volgens RFC 1122 mogen zulke adressen nooit buiten de host verschijnen: andere computers bereiken hem niet. Zie *Open deuren* in *Het idee*.",
        },
    )
    choice_reach
    return (choice_reach,)


@app.cell(hide_code=True)
def text_exercise_environment():
    mo.md(r"""
    ### Oefening 5. In de environment, en dan?

    De winkel bewaart de key van PedalPay nu in de environment variable `PEDALPAY_KEY`, zoals commit c90e5a2 deed, en niet meer in de code. Als de assistent crasht, stuurt de server een crashrapport naar een dienst buiten de winkel die fouten verzamelt, en dat rapport bevat de environment van het proces.
    """)
    return


@app.cell(hide_code=True)
def choice_environment_cell():
    choice_environment = L.choice(
        "environment",
        question="Wat betekent dat voor de key van PedalPay?",
        options={
            "right": "Wie het rapport kan lezen, kan de key nu gebruiken",
            "none": "Niets: een key buiten de code komt niet in een rapport",
            "push": "Push protection houdt het rapport tegen",
        },
        correct="right",
        why={
            "right": "OWASP waarschuwt dat environment variables meestal te lezen zijn voor alle processen, en in logs of geheugendumps terecht kunnen komen. Een crashrapport met de environment erin is zo'n plek: de key gaat overal mee waar het rapport heen gaat. Trek hem in, net als een key die gepusht is. Uit de code is uit de repository, niet buiten bereik.",
            "none": "OWASP waarschuwt dat environment variables in logs of geheugendumps terecht kunnen komen, en een crashrapport met de environment erin is er zo een. Zie *Keys uit de code* in *Het idee*.",
            "push": "Push protection controleert wat er naar een repository op GitHub gepusht wordt, niet wat een server ergens anders heen stuurt. Zie *Secret scanning* in *Het idee*.",
        },
    )
    choice_environment
    return (choice_environment,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Injection en geheimen](/course/open/rai-classic-vulnerabilities?lang=nl) noemde een key in code een zwakke plek, en [Batches, keys en de rekening](/course/open/build-batch-keys-costs?lang=nl) las een key uit de environment en stuurde hem mee met `Bearer`. Deze les keek naar de andere kant: hoe een scanner een key vindt die toch in de code terechtkwam, en hoe een server beslist wie hij binnenlaat. Het aanvalsoppervlak van een LLM-app is meer dan de prompt. De volgende les, *Een modelbestand dat code uitvoert*, kijkt naar nog een punt: het modelbestand dat je laadt.

    **Een voorbeeld uit het dagelijks leven.** Een hotel geeft elke gast een kaart die één kamer opent. Als het systeem dat de kaarten controleert uitvalt, moeten de deuren dicht blijven, en niet voor iedereen opengaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_environment, choice_reach, result_authorized, result_key, result_local):
    L.wrap_up(result_key, result_local, result_authorized, choice_reach, choice_environment)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
