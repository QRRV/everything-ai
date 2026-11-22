# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-regex. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-regex", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_keywords():
    quick_keywords = L.choice(
        "quick-keywords",
        question="Het programma van de fietsenwinkel vond met lijsten trefwoorden waar een verzoek over ging, en zag *The lock on my bike won't open.* (het slot van mijn fiets gaat niet open) aan voor een vraag over openingstijden. Waar keek het programma naar?",
        options={
            "meaning": "Wat de klant wilde weten, gelezen uit de hele zin",
            "right": "Of bepaalde letters in het verzoek voorkomen",
            "order": "De volgorde van de woorden, van eerste tot laatste",
        },
        correct="right",
        why={
            "right": "Het programma kijkt of een woord zoals *open* tussen de letters van het verzoek staat, en die letters zeggen niet wat het verzoek betekent. Deze les schrijft patronen die meer vinden dan één vast woord, zoals elk jaartal of elke kloktijd in een tekst, en ook die vinden letters, geen betekenis.",
            "meaning": "Lezen wat een zin betekent, is juist wat de trefwoorden niet konden. Zie Oefening 4, *Een slot dat niet open wil*, in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "order": "Het programma keek niet naar de volgorde van de woorden. Zie Oefening 4, *Een slot dat niet open wil*, in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_keywords
    return (quick_keywords,)


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="Een programma leest je e-mails en zet elke datum en kloktijd die het vindt in je agenda. Welke kant van NLP is dat, volgens de regel van [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl)?",
        options={
            "right": "Begrijpen: er gaat tekst in, en er komt data uit",
            "making": "Maken: er gaat data in, en er komt tekst uit",
            "neither": "Geen van beide, want het programma zet alleen getallen in een agenda",
        },
        correct="right",
        why={
            "right": "Er gaat tekst in, en er komen datums en tijden uit als data voor de agenda: begrijpen. Deze les schrijft de eerste stap van zo'n programma: patronen die kloktijden en andere getallen in tekst vinden.",
            "making": "Maken gaat van data naar tekst, zoals een weerbericht dat uit metingen geschreven wordt. Welke kant gaat dit programma op? Zie *Begrijpen en maken* in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "neither": "Wat eruit komt is geen tekst, maar wat erin gaat wel: de e-mails. Zie *Begrijpen en maken* in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tekst staat vol dingen met een vaste vorm: datums, kloktijden, bedragen, postcodes, telefoonnummers. Een **reguliere expressie**, kort regex, is een patroon dat zo'n vorm beschrijft, en een programma dat er een heeft, kan elk stuk van een tekst vinden dat erin past. Jurafsky en Martin schrijven in het hoofdstuk *Words and Tokens* van *Speech and Language Processing* dat een reguliere expressie formeel een algebraïsche notatie is voor een verzameling strings, en dat je er in de praktijk een string mee zoekt in een tekst en mee aangeeft hoe je die string verandert, wat allebei de kern is van tokenization. Ook schrijven ze dat elk systeem voor tokenization reguliere expressies als een stap gebruikt. De code van GPT-2 knipt een tekst bijvoorbeeld eerst in stukken met één reguliere expressie, zoals een woord met de spatie ervoor, een reeks cijfers of een reeks leestekens, en splitst pas daarna elk stuk verder. Het programma ELIZA, dat zinnen zoals *I need X* omzet in antwoorden zoals *What would it mean to you if you got X?*, werkt volgens Jurafsky en Martin met een reeks vervangingen door reguliere expressies, die elk een deel van de invoer vinden en veranderen. Deze les schrijft patronen voor getallen, postcodes en kloktijden, probeert de getallen en de tijden uit op de Nederlandse zinnen van Tatoeba, en telt wat elk patroon vindt, en wat het fout doet.

    **Een voorbeeld uit het dagelijks leven.** Een webformulier dat je postcode weigert omdat je een spatie typte, kan die vergelijken met een patroon van hoe een postcode eruitziet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In de 199.970 Nederlandse zinnen van Tatoeba vindt een patroon 472 getallen van precies vier cijfers. Hoeveel daarvan liggen tussen 1000 en 2099, waar de meeste jaartallen in een tekst zouden liggen?",
        options={
            "few": "Zo'n 50",
            "half": "Zo'n 240",
            "right": "431",
            "all": "Alle 472, stuk voor stuk",
        },
        reveal="**431**, ongeveer 9 op de 10. De meeste zijn jaartallen, zoals 2013, dat alleen al 42 keer voorkomt. Maar een getal tussen 1000 en 2099 is niet altijd een jaartal, zoals in *bijna 2000 nieuwe leraren* en *Tussen de 1490 en 1635 mensen*, en een jaartal kan erbuiten liggen, zoals in *het jaar 2345*. Een patroon vindt de vorm van een stuk tekst, niet waar het voor staat.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een patroon voor een verzameling strings

    Het eenvoudigste patroon is een stuk tekst zelf: het patroon `de` past op elke plaats waar een d en een e naast elkaar staan. Elke plaats waar een patroon past, is een **match**. Andere tekens in een patroon staan voor meer dan zichzelf, zodat één patroon veel strings kan beschrijven. De documentatie van Python noemt reguliere expressies in de kern een kleine, sterk gespecialiseerde programmeertaal binnen Python, die je gebruikt via de module re. Deze les gebruikt vier soorten stukken.

    Een **tekenset**, tussen vierkante haken, matcht één teken uit een set: `[dD]` matcht d of D, `[A-Z]` één hoofdletter van A tot en met Z, en `[0-9]` één cijfer; `\d` is kort voor een cijfer. Een punt `.` matcht elk teken, behalve een regeleinde.

    Een **herhaling** zegt hoe vaak het stuk ervoor komt: `*` nul keer of vaker, `+` één keer of vaker, `?` nul keer of één keer, `{3}` precies drie keer, en `{1,3}` één tot drie keer. Jurafsky en Martin noemen de eerste twee naar Stephen Kleene: de Kleene star betekent nul of meer keer het teken of de reguliere expressie ervoor. Een herhaling neemt zoveel tekens als mogelijk: patronen zijn greedy, schrijven ze, en breiden zich zo ver mogelijk uit over een string.

    Een **anker** matcht een plaats, geen teken: `^` het begin van de tekst, `$` het eind, en `\b` een woordgrens: een plaats met aan de ene kant een letter, een cijfer of `_`, en aan de andere kant een teken dat geen van drieën is, of het begin of eind van de tekst.

    Een **groep**, tussen haakjes, houdt een deel van het patroon bij elkaar, en haalt het deel van een match eruit dat je wilt, zoals de uren van een kloktijd.

    ### Patronen in Python

    De module `re` van Python werkt met patronen. `re.search(pattern, text)` zoekt de eerste match. De documentatie zegt dat de functie de string afzoekt naar de eerste plaats waar het patroon een match geeft, en die match teruggeeft, of None als het patroon nergens in de string past. `re.findall(pattern, text)` geeft elke match: alle matches die elkaar niet overlappen, als lijst met strings of met tuples. Met groepen geeft de functie wat de groepen matchten: bij één groep een lijst met strings van die groep, bij meer groepen een lijst met tuples van strings.

    Patronen schrijf je als **raw strings**, met een `r` voor de aanhalingstekens. In een gewone string leest Python zelf al sommige backslashes: `"\n"` is een regeleinde, en `"\b"` een backspace. Volgens de documentatie behandelt Python backslashes in een string met een `r` ervoor op geen enkele bijzondere manier: `r"\n"` is een string van twee tekens, een backslash en een n, en `"\n"` een string van één teken, een regeleinde. Dus `r"\b"` komt bij `re` aan als een backslash en een b, een woordgrens, en `"\b"` als een backspace, die in geen enkele zin staat.

    ### Twee soorten fouten

    Een patroon kan op twee manieren fout gaan. Jurafsky en Martin bouwen een patroon voor het Engelse woord *the*, van `the` via `[tT]he` naar `\b[tT]he\b`, en schrijven dat ze daarbij twee soorten fouten herstelden: false positives, strings die ze ten onrechte matchten, zoals other en there, en false negatives, strings die ze ten onrechte misten, zoals The. Een **false positive** is een match die er geen had mogen zijn; een **false negative** is een stuk tekst dat het patroon had moeten matchen, en miste. Minder fouten maken, schrijven ze, vraagt om twee inspanningen die elkaar tegenwerken: de **precision** verhogen, dus minder false positives, en de **recall** verhogen, dus minder false negatives. De afbeelding doet hetzelfde voor het Nederlandse woord *de*, in een zin van Tatoeba, en telt de matches van elk patroon in alle Nederlandse zinnen.

    ### Waar reguliere expressies vandaan komen

    In een memorandum uit 1951 voor de RAND Corporation, *Representation of Events in Nerve Nets and Finite Automata*, beschreef Stephen Kleene een soort gebeurtenissen die Kleene regular events noemde, met de vraag of iemand een betere naam wist. Het memorandum verscheen in 1956 opnieuw, in het boek *Automata Studies*, en Jurafsky en Martin noemen dat werk uit 1951 en 1956 de eerste definitie van reguliere expressies. In 1968 beschreef Ken Thompson in *Communications of the ACM* een compiler die van een reguliere expressie een programma maakt dat tekst doorzoekt, voor de computer IBM 7094. Reguliere expressies gingen van Thompsons versie van de editor QED naar de Unix-editor ed, met het commando `g/re/p`, van global regular expression print, dat de regels met een match liet zien. Toen Thompson gevraagd werd hoe je een bestand doorzoekt dat te groot is voor de editor, schrijft Doug McIlroy in *A Research UNIX Reader*, maakte Thompson er een apart programma van en noemde het naar dat commando: grep. Het was meteen een succes, en werd al snel een werkwoord. grep staat in de Unix-handleiding van november 1973.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **reguliere expressie** (regex) | een patroon dat een verzameling strings beschrijft, zoals elk getal van vier cijfers |
        | **match** | een stuk tekst waar een patroon op past |
        | **tekenset** `[...]` | één teken uit een set: `[aeiou]` een klinker, `[A-Z]` een hoofdletter, `[1-9]` een cijfer van 1 tot en met 9 |
        | `\d` | één cijfer, kort voor `[0-9]`; in Python ook een cijfer uit een ander schrift, zoals ٣ |
        | `.` | elk teken, behalve een regeleinde; `\.` is een punt zelf |
        | `*`, `+`, `?` | het stuk ervoor nul keer of vaker, één keer of vaker, nul keer of één keer: `colou?r` matcht color en colour |
        | `{3}`, `{1,3}` | het stuk ervoor precies drie keer, of één tot drie keer |
        | **anker** | een plaats in plaats van een teken: `^` het begin van de tekst, `$` het eind, `\b` een woordgrens |
        | **groep** `( )` | een deel van een patroon tussen haakjes, bij elkaar gehouden; `re.findall` geeft wat elke groep matchte |
        | **raw string** `r"..."` | een string waarin Python elke backslash met rust laat, zodat `re` het patroon krijgt zoals het geschreven is |
        | `re.search(pattern, text)` | de eerste match in de tekst, of `None` als er geen is |
        | `m.group()` | de tekst die de match `m` beslaat |
        | `re.findall(pattern, text)` | elke match, als lijst met strings, of met tuples als het patroon meer dan één groep heeft |
        | `x is not None` | `True` als `x` iets is, `False` als het `None` is |
        | `int("08")` | het hele getal waar een string van cijfers voor staat, hier 8 |
        | **false positive**, **false negative** | een match die er geen had mogen zijn; een stuk tekst dat had moeten matchen en gemist werd |
        | **precision**, **recall** | minder false positives betekent meer precision; minder false negatives meer recall |
        """),
        L.watch_out("Een patroon kan heel lang duren. Als het op veel manieren kan proberen een tekst te laten passen, probeert het programma dat het patroon uitvoert de ene manier na de andere; dat heet backtracking. Op 2 juli 2019 zette Cloudflare een regel live met een reguliere expressie die volgens het eigen verslag enorm veel backtrackte: de processors die webverkeer afhandelen, liepen in het hele netwerk op tot bijna 100%, en de dienst lag 27 minuten plat. Op 20 juli 2016 lag Stack Overflow 34 minuten plat, toen een bericht met zo'n 20.000 tekens witruimte achter elkaar een patroon tegenkwam dat witruimte weghaalt, in het verslag vereenvoudigd tot `\\s+$`, waarin `\\s` elk witruimteteken is. De spaties stonden niet aan het eind, dus `\\s+` begon bij de eerste spatie, nam alle spaties, vond geen eind, en begon opnieuw bij de tweede spatie, en zo verder voor elke spatie. Probeer een patroon ook op lange en vreemde tekst."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Beschrijf wat je zoekt als een vorm: welke tekens, hoeveel, en waar die staan.
    2. Schrijf die vorm als patroon in een raw string: tekensets voor de tekens, herhaling voor hoeveel, ankers en woordgrenzen voor waar.
    3. Gebruik `re.search` om te weten of en waar er een match is, en `re.findall` voor elke match; zet de delen die je wilt in groepen.
    4. Kijk wat het patroon vindt en mist in echte tekst: de false positives en de false negatives.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `ECLIPSE` bevat een zin van Tatoeba over een maansverduistering, een eclipse van de maan, met twee kloktijden. De cel zoekt er een reeks cijfers in met `re.search` en laat de match zien met `m.group()`, vindt elke reeks cijfers met `re.findall`, en zoekt een euroteken.
    """)
    return


@app.cell
def example_search():
    import re

    ECLIPSE = C.eclipse()
    print(ECLIPSE)
    m = re.search(r"\d+", ECLIPSE)
    print(m.group())
    print(re.findall(r"\d+", ECLIPSE))
    print(re.search(r"€", ECLIPSE))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `\d+` neemt een reeks cijfers zo lang als het kan: 21, niet 2. `re.search` stopt bij de eerste match, en `re.findall` geeft alle vier de reeksen. Maar de tijden vallen uit elkaar: 21 en 30 horen bij elkaar. Oefening 3 houdt de tijden bij elkaar met groepen. De zin heeft geen euroteken, dus `re.search` geeft `None`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_four():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. four_digits: elk getal van vier cijfers

        `SENTENCE` hieronder bevat een zin van Tatoeba met een jaartal en het nummer van een ster. Schrijf `four_digits(text)`: elk getal van precies vier cijfers in `text`, als strings, in de volgorde waarin die in de tekst staan. Een getal dat deel is van een langer getal telt niet mee, dus 28678 geeft niets. Gebruik `re.findall`, en schrijf het patroon als raw string. Voor `four_digits(SENTENCE)` is de uitkomst `["2019"]`. Vervang de `...`. De check roept `four_digits(SENTENCE)` aan, daarna eigen gevallen: een telefoonnummer, een jaartal na 2099, een gratis nummer met streepjes, twee jaartallen, een aandeel met een decimale komma, een tekst zonder getallen, en een lege tekst.
        """),
        L.task("Vind elk getal van precies vier cijfers in een tekst met re.findall."),
    ])
    return


@app.cell
def sentence_cell():
    SENTENCE = C.star()
    print(SENTENCE)
    return


@app.cell(hide_code=True)
def exercise_four():
    box_four = L.code_box("four-digits", lines=4)
    box_four
    return (box_four,)


@app.cell(hide_code=True)
def check_four(box_four):
    result_four = L.check("four-digits", box_four.value)
    result_four
    return (result_four,)


@app.cell(hide_code=True)
def hints_four():
    L.hints(
        "four-digits",
        [
            "Geef terug wat `re.findall` in `text` vindt. Schrijf het patroon als raw string: een woordgrens, dan de cijfers, dan weer een woordgrens.",
            "Welk stuk uit de tabel zegt hoe vaak het stuk ervoor komt?",
            r'`    return re.findall(r"\b___\b", text)`',
        ],
        why=r'`return re.findall(r"\b\d{4}\b", text)`: een woordgrens, precies vier keer een cijfer, en een woordgrens.',
    )
    return


@app.cell(hide_code=True)
def text_after_four():
    mo.md(r"""
    Van de 472 getallen liggen er 431 tussen 1000 en 2099, en de meeste daarvan zijn jaartallen: 2013 komt alleen al 42 keer voor. Bij de 41 andere zitten bedragen en maten, zoals *3000 yen*, dat 6 keer voorkomt, en de delen van telefoonnummers zoals *9876-5432*, waar het streepje ook een woordgrens is. Zonder de woordgrenzen vindt `\d{4}` 487 matches: 15 daarvan zijn vier cijfers binnen een langer getal of woord, zoals 2867 in 28678, false positives.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_postcode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. is_postcode: is dit een postcode?

        Een Nederlandse postcode bestaat uit vier cijfers en twee letters, zegt PostNL. De bibliotheek python-stdnum, die nummers zoals postcodes controleert, laat de vier cijfers met 1 tot en met 9 beginnen, niet met 0. Het CBS schrijft een postcode met een spatie, zoals in 2514 AT, en zonder, zoals in 1234AB. Schrijf `is_postcode(s)`: `True` als de hele string `s` een postcode is, en anders `False`: vier cijfers waarvan het eerste geen 0 is, dan één spatie of geen, dan twee hoofdletters, en niets ervoor of erna. Gebruik `re.search` met een patroon dat vastzit aan het begin en aan het eind van de tekst, en vergelijk wat het geeft met `None`. Voor `is_postcode("1234 AB")` is de uitkomst `True`. Vervang de `...`. De check roept `is_postcode("1234 AB")` aan, daarna eigen gevallen: geen spatie, een andere postcode, kleine letters, een eerste cijfer 0, te weinig en te veel cijfers, te veel letters, twee spaties, een streepje, een woord voor de postcode, en niets.
        """),
        L.task("Zeg of een hele string een Nederlandse postcode is, met re.search en ankers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_postcode():
    box_postcode = L.code_box("is-postcode", lines=4)
    box_postcode
    return (box_postcode,)


@app.cell(hide_code=True)
def check_postcode(box_postcode):
    result_postcode = L.check("is-postcode", box_postcode.value)
    result_postcode
    return (result_postcode,)


@app.cell(hide_code=True)
def hints_postcode():
    L.hints(
        "is-postcode",
        [
            "Bouw het patroon van links op: een cijfer van 1 tot en met 9, nog drie cijfers, de spatie die mag ontbreken, en twee hoofdletters. Zet een anker voor het begin ervoor en een voor het eind erachter. Geef dan terug of `re.search` een match vindt, door te vergelijken wat het geeft met `None`.",
            "Een spatie is ook een teken. Welk teken maakt het stuk ervoor optioneel?",
            r'`    return re.search(r"^[1-9]\d{3}___[A-Z]{2}$", s) is not None`',
        ],
        why=r'`return re.search(r"^[1-9]\d{3} ?[A-Z]{2}$", s) is not None`: het begin, een cijfer van 1 tot en met 9, drie cijfers, nul keer of één keer een spatie, twee hoofdletters, en het eind.',
    )
    return


@app.cell(hide_code=True)
def text_after_postcode():
    mo.md(r"""
    `^` en `$` maken het patroon vast aan de hele tekst, dus *Postcode: 1234 AB* en *12345 AB* worden geweigerd. Python heeft ook `re.fullmatch`, dat alleen matcht als de hele string matcht, en dat op één plaats strenger is: `$` matcht ook vlak voor een regeleinde helemaal aan het eind, dus `re.search(r"^[1-9]\d{3} ?[A-Z]{2}$", "1234 AB\n")` vindt een match, en `re.fullmatch(r"[1-9]\d{3} ?[A-Z]{2}", "1234 AB\n")` geeft `None`. Het patroon controleert alleen de vorm: of een postcode van die vorm bestaat, kan een patroon niet weten. Daarvoor heb je de lijst met postcodes zelf nodig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_times():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. clock_times: elke tijd in een tekst

        In gewone Nederlandse tekst schrijf je een kloktijd met een punt, zoals in *8.30 uur*, zegt Taaladvies.net; technische teksten kunnen een dubbele punt gebruiken, zoals in *17:03:16*. In Tatoeba komen allebei de schrijfwijzen voor. Schrijf `clock_times(text)`: elke kloktijd in `text`, als paar hele getallen `(uren, minuten)`, op volgorde. Een tijd is één of twee cijfers voor de uren, een punt of een dubbele punt, en twee cijfers voor de minuten, zonder cijfer er direct voor of na. Gebruik `re.findall` met een groep om de uren en een groep om de minuten, en maak van elk paar strings hele getallen met `int`. Voor `clock_times(ECLIPSE)` is de uitkomst `[(21, 30), (23, 10)]`. Vervang de `...`. De check roept `clock_times(ECLIPSE)` aan, daarna eigen gevallen: een tijd met een punt, twee tijden in één zin, minuten die met een nul beginnen, een jaartal voor een tijd, het nummer van een versie, een getal met een punt tussen de duizendtallen, een bedrag in dollars, en een lege tekst.
        """),
        L.task("Vind elke kloktijd in een tekst, en haal die met groepen uit elkaar in uren en minuten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_times():
    box_times = L.code_box("clock-times", lines=5)
    box_times
    return (box_times,)


@app.cell(hide_code=True)
def check_times(box_times):
    result_times = L.check("clock-times", box_times.value)
    result_times
    return (result_times,)


@app.cell(hide_code=True)
def hints_times():
    L.hints(
        "clock-times",
        [
            "Schrijf het patroon van links op: een woordgrens, een groep met één of twee cijfers, wat er tussen de uren en de minuten staat, een groep met twee cijfers, en een woordgrens. `re.findall` geeft dan een paar strings voor elke tijd. Maak daar een lijst met paren hele getallen van, met een list comprehension en `int`.",
            "Er kunnen twee verschillende tekens tussen de uren en de minuten staan. Hoe neemt één plaats in een patroon elk van die twee?",
            r'`    found = re.findall(r"\b(\d{1,2})___(\d\d)\b", text)`',
        ],
        why=r'`found = re.findall(r"\b(\d{1,2})[.:](\d\d)\b", text)`, en dan `return [(int(h), int(m)) for h, m in found]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_times():
    mo.md(r"""
    Tussen vierkante haken is een punt gewoon een punt, dus `[.:]` neemt een punt of een dubbele punt; daarbuiten is een punt elk teken. In Tatoeba vindt het patroon 67 kloktijden, 50 met een dubbele punt en 17 met een punt, en 31 daarvan zijn 2:30. Een ervan is geen tijd: de 3.11 van *Windows voor werkgroepen 3.11* is het nummer van een versie, een false positive. En het patroon mist elke tijd zonder punt of dubbele punt, zoals *om 8 uur*: *om*, een getal en *uur* staan 52 keer achter elkaar in Tatoeba, false negatives. Een tweede patroon kan die vinden, en brengt eigen fouten mee.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: een eigen patroon

    Typ een patroon. De afbeelding laat zien hoeveel matches het heeft in de Nederlandse zinnen van Tatoeba, in hoeveel zinnen, en een paar van die zinnen met elke match gemarkeerd. Nog twee stukken die je kunt gebruiken: `\w` matcht een letter, een cijfer of `_`, en `{2,}` betekent twee keer of vaker. Probeer `\bTom\b`, `\b\w+heid\b`, `\b[A-Z]{2,}\b`, `\d+%`, `€ ?\d+` of een eigen patroon.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_pattern = mo.ui.text(value=r"\b\d{1,2} uur\b", label="Patroon", full_width=True)
    demo_pattern
    return (demo_pattern,)


@app.cell(hide_code=True)
def demo_look(demo_pattern):
    mo.Html(C.demo_picture(demo_pattern.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    `\bTom\b` matcht in 25.328 zinnen: de zinnen van Tatoeba gebruiken die naam veel. `\b\w+heid\b` vindt *waarheid* 440 keer en *gezondheid* 166 keer, en ook *afscheid*, 40 keer, dat alleen op dezelfde letters eindigt. Elk patroon dat je probeert, vindt letters in een vorm; of de matches zijn wat je bedoelde, zie je pas als je de zinnen leest.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Welk gereedschap past?

    Elke vetgedrukte taak vraagt iets van een tekst. Zet de taak bij het gereedschap dat erbij past.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_tool = L.sort("which-tool")
    sort_tool
    return (sort_tool,)


@app.cell(hide_code=True)
def check_sort(sort_tool):
    result_sort = L.check("which-tool", sort_tool)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl) vond met trefwoorden waar een verzoek over ging: vaste stukken tekst. Een reguliere expressie beschrijft een hele verzameling strings: tekensets voor welke tekens, herhaling voor hoeveel, ankers en woordgrenzen voor waar, en groepen voor de delen die je wilt. `re.search` zegt of en waar een patroon matcht, `re.findall` geeft elke match, en een raw string houdt de backslashes voor `re`. Je vond elk getal van vier cijfers, controleerde postcodes zoals een formulier dat doet, en haalde kloktijden uit elkaar in uren en minuten. Op echte tekst had elk patroon false positives of false negatives, en kijken wat een patroon vindt en mist, hoort bij het schrijven ervan. De volgende les van het domein, *Tekst in zinnen en woorden knippen*, knipt tekst met regels in zinnen en woorden, en laat zien waar eenvoudige regels tekortschieten.

    **Een voorbeeld uit het dagelijks leven.** In een teksteditor die met reguliere expressies kan zoeken, vindt `\b\d{4}\b` in één keer elk getal van vier cijfers in een document.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_four, result_postcode, result_sort, result_times):
    L.wrap_up(result_four, result_postcode, result_times, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
