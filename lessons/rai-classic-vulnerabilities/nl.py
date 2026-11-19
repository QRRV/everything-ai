# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-classic-vulnerabilities. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-classic-vulnerabilities", "nl")
    C = L.checks   # de website van de verzonnen bibliotheek: een database in het geheugen, de zoekfunctie voor uitleningen en de logins


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) en [Tekst: strings](/course/open/py-strings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="Via een zoekvak op de website van de bibliotheek kan iedereen de uitleningen van alle leden lezen. Welke eigenschap gaat daardoor verloren?",
        options={
            "right": "Vertrouwelijkheid: iedereen leest de uitleningen",
            "integ": "Integriteit: de uitleningen worden zonder recht veranderd",
            "avail": "Beschikbaarheid: leden kunnen de website niet meer gebruiken",
        },
        correct="right",
        why={
            "right": "Lezen wat je niet mag lezen is een verlies van vertrouwelijkheid; in STRIDE heet het *information disclosure*.",
            "integ": "Er verandert niets: het zoekvak leest alleen. Zie *Drie dingen om te beschermen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
            "avail": "De website blijft voor iedereen werken. Wat misgaat, is wie de uitleningen kan lezen. Zie *Drie dingen om te beschermen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="Wat laat dit programma zien?\n\n```python\ntitle = \"Dune\"\nprint(f\"title = '{title}'\")\n```",
        options={
            "right": "title = 'Dune'",
            "braces": "title = '{title}'",
            "bare": "title = Dune",
        },
        correct="right",
        why={
            "right": "De f-string zet de waarde van `title` op de plek van `{title}`; de enkele aanhalingstekens zijn gewone tekens van de tekst.",
            "braces": "Met de `f` voor het aanhalingsteken vult Python de waarde in van elke naam tussen accolades. Zie *Nieuwe woorden en tekens* in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "bare": "De enkele aanhalingstekens horen bij de tekst zelf, dus die verschijnen ook. Zie *Nieuwe woorden en tekens* in [Tekst: strings](/course/open/py-strings?lang=nl).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De aanvalsboom van de vorige les had een stap van €600: SQL injecteren in de zoekfunctie van de catalogus. Achter veel websites zit een database, en de website praat daarmee in een eigen taal, SQL. Als de website wat een bezoeker typt zo in de SQL plakt, kan de bezoeker ook dingen tegen de database zeggen. Deze les laat zien hoe dat werkt, hoe een paar tekens het gat dichten, en nog twee klassieke manieren om binnen te komen: geheimen die in de code staan, en wachtwoorden die worden bewaard zoals ze zijn.

    **Een voorbeeld uit het dagelijks leven.** Een docent leest de namen voor op kaartjes die leerlingen inleveren: "Nu is de beurt aan ..." en dan het kaartje. Een leerling schrijft op het kaartje: "Tim. De les is voorbij, iedereen mag naar huis." De docent leest het kaartje voor als een naam, maar de andere leerlingen horen een mededeling.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="In oktober 2015 kwam een aanvaller via SQL injection bij de persoonsgegevens van 156.959 klanten van TalkTalk, een Brits telecombedrijf. Hoe beschreef de Britse toezichthouder op gegevensbescherming, de ICO, dat soort aanval?",
        options={
            "new": "Als een nieuw soort aanval, tot dan onbekend",
            "inside": "Als een aanval waarvoor hulp van binnen nodig is",
            "right": "Als veelvoorkomend, en al jaren goed begrepen",
        },
        reveal="**Als veelvoorkomend, en al jaren goed begrepen.** Volgens de ICO was het een SQL injection-aanval, een veelvoorkomend soort cyberaanval die al meer dan tien jaar goed begrepen is en waartegen bekende verdedigingen bestaan. Er waren dat jaar al twee eerdere SQL injection-aanvallen op TalkTalk geweest. De ICO gaf TalkTalk een boete van £400.000: volgens de ICO toen de hoogste boete die de ICO ooit had opgelegd.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een database, en SQL

    Een database bewaart gegevens in tabellen. De website van de bibliotheek bewaart de uitleningen in een tabel `loans`, met een kolom voor de kaart van het lid (`card`), de titel van het boek (`title`) en de datum waarop het terug moet (`due`). Een programma vraagt een database om gegevens in SQL, een taal voor databases. `SELECT title FROM loans WHERE card = 'L-1004'` vraagt om de titels van de uitleningen met kaart L-1004. Een tekstwaarde, zoals de kaart, staat tussen enkele aanhalingstekens.

    ### SQL injection

    De website maakt die opdracht met een f-string uit wat een lid typt: `f"SELECT title FROM loans WHERE card = '{card}'"`. De OWASP Foundation, die adviezen publiceert over de beveiliging van webapplicaties, omschrijft de aanval als het invoegen, of injecteren, van een SQL-query via de invoer die de gebruiker naar de applicatie stuurt. Het werkt doordat SQL volgens OWASP geen echt onderscheid maakt tussen opdrachten en gegevens: de database kan niet zien welk deel van de opdracht het programma schreef en welk deel een lid typte. De Python-documentatie zegt hoe een aanvaller dat gebruikt: sluit het enkele aanhalingsteken en voeg OR TRUE toe, en je krijgt alle rijen. Wie `x' OR 'a'='a` typt, sluit de kaart af met het eerste aanhalingsteken en voegt een voorwaarde toe die voor elke rij waar is. Wat een aanvaller daarna kan doen, dekt alle zes de letters van STRIDE. Volgens OWASP kan een aanvaller met SQL injection zich voordoen als iemand anders, bestaande gegevens veranderen, problemen met onweerlegbaarheid veroorzaken, zoals transacties ongedaan maken of saldo's aanpassen, alle gegevens in het systeem inzien, gegevens vernietigen of onbruikbaar maken, of beheerder van de databaseserver worden.

    Injectie is niet alleen SQL. Op de lijst van 2025 van de 25 gevaarlijkste zwakke plekken in software, van MITRE en CISA, staat SQL injection op nummer 2. Nummer 1 is *cross-site scripting*; OWASP noemt die aanvallen een soort injectie, waarbij kwaadaardige scripts in verder onschuldige en vertrouwde websites worden gestopt.

    ### De oplossing: placeholders

    De oplossing houdt de opdracht en de gegevens gescheiden. De Python-documentatie zegt: gebruik altijd placeholders in plaats van stringopmaak om Python-waarden aan SQL-opdrachten te koppelen, om SQL injection te voorkomen. Een **placeholder** is een `?` in de opdracht op de plek van een waarde; de waarden gaan apart naar de database, in een lijst: `run("SELECT title FROM loans WHERE card = ?", [card])`. De database leest eerst de opdracht, en zet daarna elke waarde op de juiste plek, als gegevens, nooit als SQL. OWASP geeft een voorbeeld: invoer als `tom' or '1'='1` wordt dan letterlijk behandeld als de gebruikersnaam die moet kloppen, niet als een voorwaarde die aan de query wordt toegevoegd. De aanhalingstekens in plaats daarvan met de hand onschadelijk maken (*escaping*) is volgens OWASP foutgevoelig en wordt sterk afgeraden.

    ### Geheimen in de code

    Een geheim is wat toegang geeft: een wachtwoord, of een API key waarmee een programma inlogt bij een dienst. MITRE telt het als zwakke plek als een product vaste inloggegevens in zijn code heeft, zoals een wachtwoord of een cryptografische sleutel. Wie de code ziet, heeft de sleutel. MITRE: als het wachtwoord ooit ontdekt of gepubliceerd wordt, wat op internet vaak gebeurt, kan iedereen die het kent bij het product. Een privérepository houdt een geheim ook niet veilig. Volgens een klacht van de Amerikaanse Federal Trade Commission zeiden indringers dat ze in 2016 binnenkwamen op de GitHub van Uber met wachtwoorden die bij eerdere datalekken waren uitgelekt; Uber verplichtte daar geen multi-factor authentication, die hieronder bij *Wachtwoorden* terugkomt. Volgens de klacht vonden ze een toegangssleutel die leesbaar in code stond in een privérepository op GitHub, en kwamen daarmee bij bestanden met onder meer zo'n 25,6 miljoen namen en e-mailadressen. Het advies van GitHub: zet geen geheimen in je code, maar gebruik environment variables of een dienst voor het beheer van geheimen. *Sleutels en open deuren*, verderop in deze module, laat zien hoe.

    ### Wachtwoorden

    Een website waarop leden inloggen, moet wachtwoorden controleren zonder ze te bewaren. OWASP: bewaar wachtwoorden nooit leesbaar, maar bescherm ze met sterke, trage hashfuncties zoals Argon2id, bcrypt of PBKDF2, en voeg aan elk wachtwoord een eigen salt toe, zodat aanvallers geen vooraf berekende opzoektabellen kunnen gebruiken. Een **hash** komt uit een functie die maar één kant op werkt: uit het wachtwoord krijg je de hash, maar uit de hash krijg je het wachtwoord niet terug. Een **salt** is een willekeurige tekst die bij elk account wordt bewaard en samen met het wachtwoord wordt gehasht, zodat twee leden met hetzelfde wachtwoord verschillende hashes krijgen. Om een login te controleren, hasht de website het getypte wachtwoord met de salt van het account, en vergelijkt dat met de bewaarde hash. De richtlijnen van NIST uit 2025 (SP 800-63B-4) vragen om wachtwoorden van minstens 15 tekens als een wachtwoord de enige factor is, verbieden regels die een mix van soorten tekens eisen, en zeggen dat je een nieuw wachtwoord moet vergelijken met een lijst van veelgebruikte, voorspelbare of gelekte wachtwoorden. **Multi-factor authentication** vraagt om meer dan één soort factor; NIST noemt er drie: iets wat je weet, iets wat je hebt en iets wat je bent.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **SQL** | een taal om een database om gegevens te vragen, zoals in `SELECT title FROM loans WHERE card = 'L-1004'` |
        | **SQL injection** | een aanval waarbij tekst die in een programma wordt getypt de SQL verandert die het naar zijn database stuurt |
        | **placeholder** | een `?` in een SQL-opdracht op de plek van een waarde; de waarde gaat apart naar de database, als gegevens |
        | **hard-coded secret** | een wachtwoord of sleutel die in de code zelf staat, leesbaar voor iedereen die de code ziet |
        | **hash** | wat een functie die maar één kant op werkt van een wachtwoord maakt; het wachtwoord is er niet uit terug te rekenen |
        | **salt** | een willekeurige tekst die bij elk account wordt bewaard en samen met het wachtwoord wordt gehasht |
        | **multi-factor authentication** | een login die om meer dan één soort factor vraagt, zoals een wachtwoord en een telefoon |
        """),
        L.watch_out("Een placeholder is alleen voor waarden. Een tabel- of kolomnaam kan niet in een `?`: OWASP raadt aan die te kiezen uit een vaste lijst die bij het programma hoort. En hashen is geen versleutelen: versleuteling kun je met de sleutel terugdraaien, een hash niet."),
        mo.Html(C.idea_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo dicht je de klassieke gaten in een programma:

    1. Zoek elke plek waar tekst van buiten in SQL terechtkomt, of in een andere taal zoals HTML.
    2. Probeer het als aanvaller: een aanhalingsteken om de waarde af te sluiten, dan een voorwaarde die altijd waar is.
    3. Vervang de f-string door placeholders: een `?` in de SQL voor elke waarde, en de waarden in een lijst.
    4. Houd geheimen uit de code: lees ze uit een environment variable of een dienst voor geheimen.
    5. Bewaar wachtwoorden als trage hash met een salt per account, en controleer een login door opnieuw te hashen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De zoekfunctie van de catalogus vindt de schrijver van een boek bij de titel. `C.run(sql, values)` voert één SQL-opdracht uit op de database van de bibliotheek en geeft de eerste kolom van elke gevonden rij terug, als lijst. De database bestaat alleen in het geheugen en wordt bij elke aanroep opnieuw gemaakt, dus er kan niets veranderen op je computer. De cel zoekt twee keer met de titel in de SQL geplakt, een keer naar een echte titel en een keer met een aanval, en daarna twee keer met een placeholder.
    """)
    return


@app.cell
def example_search():
    # de zoekfunctie van de catalogus, geplakt: de titel gaat zo de SQL in
    def _search(title):
        _sql = f"SELECT author FROM books WHERE title = '{title}'"
        print(_sql)
        return C.run(_sql)

    print(_search("Frankenstein"))
    print(_search("x' OR 'a'='a"))
    # dezelfde zoekopdracht met een placeholder: de titel gaat als gegevens naar de database
    print(C.run("SELECT author FROM books WHERE title = ?", ["x' OR 'a'='a"]))
    print(C.run("SELECT author FROM books WHERE title = ?", ["Gulliver's Travels"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Voor *Frankenstein* gaat de geplakte zoekopdracht goed: één schrijver. Met de aanval vraagt de opdracht om de rijen met titel x of waarvoor `'a'='a'` geldt, en dat geldt voor elke rij: alle zes de schrijvers komen terug. Met een placeholder is dezelfde tekst alleen een titel die geen enkel boek heeft, dus er komt niets terug; en een echte titel met een aanhalingsteken erin, *Gulliver's Travels*, wordt gewoon gevonden. Geplakt zou dat aanhalingsteken de titel te vroeg afsluiten en de SQL breken: ook zonder kwade bedoelingen laat een naam als van 't Hof een geplakte zoekopdracht struikelen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_attack():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. attack: alle uitleningen

        Een lid dat inlogt, ziet de eigen uitleningen: de website roept `loans_for` aan met de kaart van het lid.

        ```python
        def loans_for(card):
            return run(f"SELECT title FROM loans WHERE card = '{card}'")
        ```

        `loans_for("L-1004")` geeft de twee uitleningen van Joris van Dam. Denk als een aanvaller: geef `attack` een tekst waardoor `loans_for(attack)` alle 9 uitleningen van de bibliotheek teruggeeft. De database bestaat alleen in het geheugen, dus er verandert niets.
        """),
        L.task("Geef attack een tekst waardoor loans_for alle uitleningen teruggeeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attack():
    box_attack = L.code_box("attack", lines=3)
    box_attack
    return (box_attack,)


@app.cell(hide_code=True)
def check_attack(box_attack):
    result_attack = L.check("attack", box_attack.value)
    result_attack
    return (result_attack,)


@app.cell(hide_code=True)
def hints_attack():
    L.hints(
        "attack",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld: de geplakte SQL zet een `'` voor je tekst en nog een erachter.",
            "Begin met een willekeurige kaart en een `'` om die af te sluiten, dan `OR` en een voorwaarde die altijd waar is. Laat het laatste aanhalingsteken van die voorwaarde open: de `'` die loans_for achter je tekst zet, sluit het af.",
            "`attack = \"x' OR ___\"`",
        ],
        why="Met `attack = \"x' OR 'a'='a\"` wordt de SQL `SELECT title FROM loans WHERE card = 'x' OR 'a'='a'`: geen enkele kaart is x, maar `'a'='a'` geldt voor elke rij, dus alle 9 uitleningen komen terug.",
    )
    return


@app.cell(hide_code=True)
def text_after_attack():
    mo.md(r"""
    Negen uitleningen van zeven leden, uit één zoekvak. Deze zoekfunctie geeft alleen titels terug, maar OWASP waarschuwt dat hetzelfde gat alle gegevens in het systeem kan blootgeven. Andere teksten werken net zo goed, zoals `' OR '1'='1`; de check kijkt alleen naar wat er terugkomt. Let op wat de aanval niet nodig had: geen wachtwoord, geen speciaal programma, alleen een aanhalingsteken op de juiste plek.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_due():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. loans_due: de veilige zoekopdracht

        Leden willen zien welke van hun uitleningen binnenkort terug moeten. Schrijf `loans_due(card, before)`: de titels van de uitleningen van `card` die terug moeten voor de datum `before`. Datums zijn teksten zoals `"2026-12-01"`, dus een eerdere datum is ook de kleinere tekst, en SQL kan ze vergelijken met `<`. Twee voorwaarden verbind je met `AND`. Gebruik placeholders, zodat er geen tekst van buiten in de SQL terechtkomt. `run(sql, values)` staat klaar. Vervang de `...`. De check roept `loans_due("L-1004", "2026-12-01")` aan, dan andere kaarten en datums, en twee aanvallen.
        """),
        L.task("Geef de titels terug van de uitleningen van card die voor de datum terug moeten, met placeholders."),
    ])
    return


@app.cell(hide_code=True)
def exercise_due():
    box_due = L.code_box("loans-due", lines=3)
    box_due
    return (box_due,)


@app.cell(hide_code=True)
def check_due(box_due):
    result_due = L.check("loans-due", box_due.value)
    result_due
    return (result_due,)


@app.cell(hide_code=True)
def hints_due():
    L.hints(
        "loans-due",
        [
            "Stap 3 van het recept, zoals in de laatste regels van het uitgewerkte voorbeeld.",
            "Schrijf de SQL met een `?` op de plek van elke waarde, de kaart en de datum, en geef `run` de waarden in een lijst, in de volgorde van de vraagtekens.",
            "`return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", ___)`",
        ],
        why="`return run(\"SELECT title FROM loans WHERE card = ? AND due < ?\", [card, before])`. `loans_due(\"L-1004\", \"2026-12-01\")` geeft `[\"Max Havelaar\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_due():
    mo.md(r"""
    Met placeholders is `x' OR 'a'='a` alleen een kaart die geen enkel lid heeft: geen uitleningen. De aanval in de datum is alleen een vreemde datum, en *Max Havelaar*, dat op 26 november terug moet, ligt er nog steeds vóór: één vergelijking van twee teksten, meer niet. Geplakt had dezelfde datum alle 9 uitleningen gegeven.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Allebei tegelijk

    Typ wat je wilt als kaart: `L-1004`, de aanval van oefening 1, of een tekst met een aanhalingsteken erin, zoals `van 't Hof`. Het plaatje zet je tekst twee keer in de zoekopdracht voor uitleningen: in de SQL geplakt, zoals loans_for doet, en met een placeholder.
    """)
    return


@app.cell(hide_code=True)
def try_card_cell():
    try_card = mo.ui.text(value="x' OR 'a'='a", label="Kaart")
    try_card
    return (try_card,)


@app.cell(hide_code=True)
def demo_both(try_card):
    _out = mo.Html(C.demo_picture(try_card.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_login():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. check_login: inloggen zonder het wachtwoord

        `USERS` heeft een account voor vijf leden: voor elke kaart een salt en een hash, gemaakt met `slow_hash(password, salt)`, zoals in de tabel hieronder. Schrijf `check_login(users, card, password)`: `True` als `card` een account heeft in `users` en de hash van `password`, gemaakt met de salt van dat account, de bewaarde hash is; anders `False`. `slow_hash` staat klaar. Om de checks snel te houden, doet die 1000 rondes PBKDF2. OWASP geeft de voorkeur aan Argon2id, en vraagt om 600.000 rondes of meer waar PBKDF2 moet vanwege FIPS-140, een standaard van de Amerikaanse overheid. Vervang de `...`. De check roept `check_login(USERS, "L-1004", ...)` aan met het juiste wachtwoord van Joris, en daarna andere kaarten en wachtwoorden.
        """),
        mo.Html(C.users_picture(L.lang)),
        L.task("Geef alleen True terug als de kaart een account heeft en de hash van het getypte wachtwoord gelijk is aan de bewaarde hash."),
    ])
    return


@app.cell(hide_code=True)
def exercise_login():
    box_login = L.code_box("login", lines=6)
    box_login
    return (box_login,)


@app.cell(hide_code=True)
def check_login_cell(box_login):
    result_login = L.check("login", box_login.value)
    result_login
    return (result_login,)


@app.cell(hide_code=True)
def hints_login():
    L.hints(
        "login",
        [
            "Stap 5 van het recept. In [Dictionaries](/course/open/py-dicts?lang=nl) controleerde je met `in` of een key er is voordat je die opzocht.",
            "Als de kaart in `users` staat, neem dan het account, hash het getypte wachtwoord met `slow_hash` en de salt van dat account, en vergelijk met de hash van dat account. Geef anders `False` terug.",
            "`return slow_hash(password, ___) == user[\"hash\"]`, onder `if card in users:` en `user = users[card]`, en daarna `return False`",
        ],
        why="`if card in users:` met `user = users[card]` en `return slow_hash(password, user[\"salt\"]) == user[\"hash\"]` eronder, en daarna `return False`. `check_login(USERS, \"L-1004\", ...)` met het juiste wachtwoord geeft `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_login():
    mo.md(r"""
    Eén hoofdletter anders, en de hash is een andere: de login mislukt. Het account van Mila gaat niet open met het wachtwoord van Joris, en L-1003 heeft helemaal geen account. De bibliotheek hoefde nooit een wachtwoord te bewaren om er een te controleren. Wie `USERS` steelt, krijgt salts en hashes, en moet wachtwoord na wachtwoord proberen, elke poging een trage hash; met een snelle hash zoals SHA-256 kunnen aanvallers volgens OWASP in korte tijd heel veel gokken doen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_secret():
    mo.md(r"""
    ### Oefening 4. Een sleutel op GitHub

    Een vrijwilliger pusht een notebook met de API key van de e-maildienst van de bibliotheek naar een openbare repository op GitHub. Een uur later ziet iemand het.
    """)
    return


@app.cell(hide_code=True)
def choice_secret_cell():
    choice_secret = L.choice(
        "secret",
        question="Wat doe je eerst?",
        options={
            "right": "De key intrekken, en een nieuwe maken",
            "rewrite": "De commit uit de geschiedenis halen en opnieuw pushen",
            "ignore": "Het notebook aan .gitignore toevoegen",
        },
        correct="right",
        why={
            "right": "Volgens de documentatie van GitHub trek je als eerste stap het geheim in of vervang je het; daarna geeft het geen toegang meer. Dan kan de key uit de code, en kan de code een nieuwe key uit een environment variable lezen.",
            "rewrite": "Anderen kunnen de key al hebben. GitHub waarschuwt dat de commits na het herschrijven van de geschiedenis elders nog te vinden kunnen zijn: in clones en forks, in bewaarde weergaven en in pull requests. Trek eerst de key in.",
            "ignore": "Volgens de documentatie van git verandert .gitignore niets aan bestanden die git al bijhoudt. En de commit die al openbaar is, houdt de key. Trek eerst de key in.",
        },
    )
    choice_secret
    return (choice_secret,)


@app.cell(hide_code=True)
def text_exercise_access():
    mo.md(r"""
    ### Oefening 5. Veilig, maar voor wie?

    De website gebruikt nu jouw `loans_due`, met placeholders. Joris is ingelogd als L-1004, en de kaart staat in het adres van de pagina. Joris verandert die in L-1007.
    """)
    return


@app.cell(hide_code=True)
def choice_access_cell():
    choice_access = L.choice(
        "access",
        question="Wat geeft loans_due terug?",
        options={
            "right": "Mila's uitleningen: placeholders kijken niet wie vraagt",
            "error": "Een fout: de kaart is niet van Joris",
            "none": "Niets: een placeholder laat alleen jouw eigen kaart door",
        },
        correct="right",
        why={
            "right": "Volgens OWASP voorkomt een placeholder dat de waarde de SQL verandert, maar beslist die niet of wie het vraagt bij het bijbehorende account mag. Wie welke uitleningen mag lezen, is autorisatie, een van de zes eigenschappen van de vorige les: de website moet nog steeds controleren of de kaart hoort bij wie er is ingelogd.",
            "error": "`loans_due` krijgt een kaart en een datum, en weet niets over wie er is ingelogd. L-1007 is een kaart als elke andere, dus de uitleningen ervan komen terug. Zie *De oplossing: placeholders* in *Het idee*.",
            "none": "Een placeholder maakt van de kaart gegevens, en de database zoekt die kaart op, wie het ook vraagt. Zie *De oplossing: placeholders* in *Het idee*.",
        },
    )
    choice_access
    return (choice_access,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) was SQL injection de stap van €600 in de aanvalsboom; placeholders nemen die stap weg, zoals de verdediging in de demo daar deed. In [Tekst: strings](/course/open/py-strings?lang=nl) maakte je teksten met f-strings: voor SQL is dat precies het gat, en de oplossing is de database de waarden te laten invullen. Dezelfde fout komt terug bij language models. Bruce Schneier vergelijkt SQL injection met een oude truc op het telefoonnet, die niet te verhelpen was omdat het probleem algemeen was: gegevens en besturing gebruikten hetzelfde kanaal. Volgens Schneier geldt: zolang een aanvaller een computer gegevens kan laten aanzien voor opdrachten, is die kwetsbaar. Maar Schneier voegt eraan toe dat we, anders dan bij het telefoonnet, de gegevens van een LLM niet van zijn opdrachten kunnen scheiden. De lijst van OWASP met risico's voor LLM-applicaties uit 2026 zegt hetzelfde: er is geen nette tegenhanger van placeholders. *Aanvallen op machine learning* kijkt naar wat aanvallers van een model willen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_access, choice_secret, result_attack, result_due, result_login):
    L.wrap_up(result_attack, result_due, result_login, choice_secret, choice_access)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
