# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-tokenize. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-tokenize", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_digits():
    quick_digits = L.choice(
        "quick-digits",
        question="Wat geeft `re.findall(r\"\\d+\", \"Om 8.30 uur\")`?",
        options={
            "whole": "`[\"8.30\"]`",
            "right": "`[\"8\", \"30\"]`",
            "dots": "`[\"8\", \".\", \"30\"]`",
        },
        correct="right",
        why={
            "right": "`\\d` matcht één cijfer en `+` maakt er een reeks van, zo lang als het kan. De punt is geen cijfer, dus de reeks stopt daar, en erna begint een nieuwe. Deze les knipt hele teksten in stukken met zulke reeksen.",
            "whole": "De punt is geen cijfer, dus `\\d+` kan de punt niet nemen: de reeks cijfers stopt bij de punt. Zie *Patronen in Python* in [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "dots": "`re.findall` geeft alleen de matches, en de punt matcht `\\d+` niet. Zie *Patronen in Python* in [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
        },
    )
    quick_digits
    return (quick_digits,)


@app.cell(hide_code=True)
def quick_check_doctor():
    quick_doctor = L.choice(
        "quick-doctor",
        question="Een programma zegt dat een zin eindigt na elk woord dat op een punt eindigt. In *Morgen komt dr. Visser langs.* zegt het dus ook dat er een zin eindigt na *dr.* Wat voor fout is dat?",
        options={
            "right": "Een false positive: het programma ziet een einde waar er geen is",
            "negative": "Een false negative: het programma mist een einde dat er echt staat",
            "none": "Geen fout, want het woord *dr.* eindigt wel op een punt",
        },
        correct="right",
        why={
            "right": "*dr.* is een afkorting, en de zin gaat erna verder: het programma vindt een einde dat er niet hoort, een false positive. Deze les zorgt ervoor dat afkortingen en initialen geen zin beëindigen.",
            "negative": "Een false negative is een einde dat het programma mist. Hier vindt het één einde te veel. Zie *Twee soorten fouten* in [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "none": "De punt van *dr.* hoort bij de afkorting, en de zin gaat erna verder. Zie *Twee soorten fouten* in [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
        },
    )
    quick_doctor
    return (quick_doctor,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voordat een programma woorden kan tellen, opzoeken of de namen in een tekst kan vinden, moet het weten waar elk woord begint en ophoudt, en waar elke zin eindigt. In het Nederlands en het Engels doet een spatie veel van dat werk, maar niet alles. Een punt kan een zin afsluiten, of een afkorting zoals *dr.*, of allebei tegelijk; of *zo'n* en *Zuid-Afrika* één woord zijn of meer, hangt af van de regels die je volgt. En Jurafsky en Martin schrijven in het hoofdstuk *Words and Tokens* van *Speech and Language Processing* dat het schrift van het Chinees, het Japans en het Thai geen losse woorden laat zien: er staan geen spaties tussen de woorden. Deze les knipt Nederlandse zinnen met één patroon in woorden, beslist met een lijst en een patroon waar een zin eindigt, en knipt tekst zonder spaties met een dictionary. De tokens en de zinseinden vergelijkt de les met een **treebank**: een verzameling zinnen waarin bij elke zin de woorden en de grammatica uitgeschreven staan. De treebank hier is UD_Dutch-Alpino van **Universal Dependencies**, een project dat voor veel talen treebanks op dezelfde manier maakt; de les noemt die treebank Alpino, en gebruikt alleen de woorden.

    **Een voorbeeld uit het dagelijks leven.** Een programma dat een tekst voorleest, moet weten waar elke zin eindigt: als het de punt van *dr.* in *dr. Visser* voor het eind van een zin hield, zou het midden in een naam pauzeren.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een regel knipt een tekst bij de spaties, en haalt elk leesteken los als een eigen token. In hoeveel van de 13.603 zinnen van Alpino geeft die regel precies de woorden van de treebank?",
        options={
            "few": "Ongeveer 2.000",
            "half": "Ongeveer 6.800",
            "right": "11.074",
            "all": "Alle 13.603 zinnen",
        },
        reveal="**11.074**, ongeveer 4 op de 5. In de andere zinnen houdt de treebank bij elkaar wat de regel splitst: woorden met een koppelteken zoals *Zuid-Afrika*, initialen zoals *J.*, afkortingen zoals *mr.*, woorden met een apostrof zoals *zo'n* en *'t*, en getallen zoals *10.000*. Alleen bij de spaties knippen geeft in 1.635 zinnen precies de woorden van de treebank. Welke stukken woorden zijn, legt een standaard vast, en een regel moet die standaard volgen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Tokens en standaarden

    Een **token** is een stuk tekst dat een programma als één geheel behandelt, zoals een woord, een getal of een leesteken, en een tekst in tokens knippen heet **tokenization**. Welke stukken één token zijn, is een keuze. Jurafsky en Martin schrijven dat je bij tokenization met regels eerst een standaard vastlegt, en dan regels schrijft die precies die standaard volgen. De standaard van de Penn Treebank voor het Engels haalt volgens hen clitics los (*doesn't* wordt *does* plus *n't*), houdt woorden met een koppelteken bij elkaar, en zet elk leesteken apart. Een **clitic** is een deel van een woord dat niet op zichzelf kan staan, en alleen voorkomt als het vastzit aan een ander woord. Universal Dependencies maakt andere keuzes. De Engelse treebank EWT splitst *don't* ook in *do* en *n't*, maar splitst daarnaast de meeste woorden met een koppelteken: volgens de richtlijnen wordt *search-engine* drie woorden, *search*, *-* en *engine*. In Alpino zijn *zo'n* en *'t* elk één woord, en volgens de richtlijnen voor het Nederlands telt daar geen enkel geschreven woord als meer dan één woord. Een woord mag er elk leesteken bevatten, zoals in *o.a.* en *vroeg-renaissance*. De afbeelding laat één zin van Alpino zien, geknipt bij de spaties, geknipt bij de spaties met elk leesteken apart, en geknipt door de treebank.

    ### Meer stukken van een patroon

    Een **keuze** `|` tussen twee patronen matcht wat een van beide matcht: `kat|hond` vindt *kat* en *hond*. Op elke plaats in de tekst probeert `re.findall` de keuzes van links naar rechts, en neemt de eerste die past. Een tekenset met `^` direct na de `[` matcht één teken dat *niet* in de set staat: `[^aeiou]` matcht elk teken behalve deze vijf klinkers, ook een spatie of een cijfer. `\s` matcht één witruimteteken, zoals een spatie, een tab of een regeleinde, en `\S` één teken dat geen witruimte is, dus `\S+` is een reeks tekens tussen spaties. En `\w` matcht een letter, een cijfer of `_`, ook een letter met een accent, zoals é. Tussen vierkante haken houden `\d`, `\w` en `\s` hun betekenis: `[\d.]` matcht een cijfer of een punt.

    ### Waar eindigt een zin?

    Vinden waar de zinnen van een tekst eindigen, heet **sentence segmentation**. Vraagtekens en uitroeptekens laten volgens Jurafsky en Martin vrij duidelijk zien dat een zin ophoudt. Een punt kan volgens hen twee dingen aangeven: het eind van een zin, of een afkorting zoals *Dr.* of *Inc.* In het Nederlands schrijf je een **afkorting** met een of meer punten, zoals *bijv.*, *blz.* en *m.a.w.*, zegt regel 17.A van de Leidraad, de spellingregels bij de Woordenlijst Nederlandse Taal. Een **initiaal**, de eerste letter van een naam, zoals *J.*, heeft ook een punt. En als een zin op een afkorting eindigt, komt er geen tweede punt achter, zegt Taaladvies.net: de punt van *enz.* aan het eind van een zin sluit de afkorting en de zin tegelijk af.

    Een regel voor zinseinden kan op twee manieren de mist in gaan, net als een patroon: een **false positive** is een einde dat de regel noemt en dat geen einde is, en een **false negative** een echt einde dat de regel mist. De **precision** is het deel van de genoemde einden dat echt is, en de **recall** het deel van de echte einden dat de regel vindt. Een lijst met afkortingen verhoogt de precision: zo'n lijst helpt volgens Jurafsky en Martin bepalen of een punt bij een veelgebruikte afkorting hoort, en je kunt die lijst met de hand maken of een computer laten leren, zoals Kiss en Strunk in 2006 beschreven. Het programma van Tibor Kiss en Jan Strunk, Punkt, leert de afkortingen uit de tekst die het knipt, zonder lijst. Op teksten uit de Nederlandse krant *De Limburger* nam Punkt voor 0,97% van de punten aan het eind van een token de verkeerde beslissing, en de regel dat elke punt aan het eind van een token een zin beëindigt, voor 6,50%.

    ### Tekst zonder spaties

    Een tekst zonder spaties in woorden knippen heet **word segmentation**. Zelfs mensen die de woorden kennen, knippen niet allemaal op dezelfde manier. Jurafsky en Martin laten de Chinese zin 姚明进入总决赛 zien, *Yao Ming bereikt de finale*, die volgens de standaard van de Chinese Treebank 3 woorden telt, en volgens de standaard van de Universiteit van Peking 5. Richard Sproat en drie collega's lieten zes moedertaalsprekers 100 Chinese zinnen in woorden knippen; de overeenkomst tussen twee sprekers, gemeten als het gemiddelde van precision en recall, was gemiddeld 0,76, waarbij 1 betekent dat twee sprekers elke zin hetzelfde knippen.

    Een eenvoudige manier om te knippen is **maximum matching**, of MaxMatch. De conceptversie van 2017 van hetzelfde boek beschrijft dat zo: het algoritme begint aan het begin van een string, kiest het langste woord uit de dictionary dat daar past, en gaat verder na dat woord. Past er geen enkel woord, dan neemt het één teken als woord, en gaat het één teken verder. Volgens die versie werkt het beter voor het Chinees dan voor het Engels, omdat Chinese woorden veel korter zijn. **Backward MaxMatch** doet hetzelfde vanaf het eind van de tekst: het neemt het langste woord waarop de tekst eindigt.
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
        | **token** | een stuk tekst dat een programma als één geheel behandelt: een woord, een getal of een leesteken |
        | **tokenization** | een tekst in tokens knippen |
        | **treebank** | een verzameling zinnen, elk met de woorden en de grammatica uitgeschreven |
        | **clitic** | een deel van een woord dat niet op zichzelf kan staan, zoals *n't* in *doesn't* |
        | **afkorting**, **initiaal** | een ingekort woord met een punt, zoals *bijv.*; de eerste letter van een naam met een punt, zoals *J.* |
        | **sentence segmentation** | vinden waar elke zin van een tekst eindigt |
        | **false positive**, **false negative** | een einde dat een regel noemt en dat geen einde is; een echt einde dat de regel mist |
        | **precision**, **recall** | het deel van de einden die een regel noemt, dat echt is; het deel van de echte einden dat de regel vindt |
        | **word segmentation** | een tekst zonder spaties in woorden knippen |
        | **maximum matching** (MaxMatch) | een tekst in woorden knippen door steeds het langste woord uit een dictionary te nemen waarmee de tekst begint |
        | **backward MaxMatch** | hetzelfde vanaf het eind: steeds het langste woord uit de dictionary waarop de tekst eindigt |
        | **dictionary** | hier: de verzameling woorden waaruit MaxMatch mag kiezen, in de code een set strings, geen Python-dict |
        | `A|B` | wat A matcht, of anders wat B matcht; op elke plaats de eerste keuze van links die past |
        | `[^...]` | één teken dat niet in de set staat: `[^aeiou]` is elk teken behalve deze vijf klinkers |
        | `\w` | één letter, cijfer of `_`, ook een letter met een accent, zoals é |
        | `\s`, `\S` | één witruimteteken: een spatie, een tab of een regeleinde; één teken dat geen witruimte is |
        | `word.lower()` | de string in kleine letters: `"Dr.".lower()` is `"dr."` |
        | `range(n, 0, -1)` | de hele getallen van n tot en met 1, aftellend: `range(3, 0, -1)` geeft 3, 2 en 1 |
        | `text[:k]` | de eerste k tekens van `text`: `"kat"[:2]` is `"ka"` |
        """),
        L.watch_out("MaxMatch is gretig: op elke plaats neemt MaxMatch het langste woord, en één woord dat te lang is, brengt de woorden erna in de war. In het hoofdstuk *Tokenisation and Sentence Segmentation* van het *Handbook of Natural Language Processing* geeft David Palmer het Engelse voorbeeld *thetabledownthere*: gretig geknipt wordt dat *theta bled own there*, en backward MaxMatch vindt volgens Palmer *the table down there*. Met de woorden van Tatoeba als dictionary wordt *ikbenzoterug* vooruit *ik ben zot er u g*, en achteruit *ik ben zo terug*. Maar achteruit is niet altijd beter: *neeikweethetniet* wordt achteruit *ne eik weet het niet*, en vooruit klopt het."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Kies een standaard: welke stukken tekst tellen als één token, zoals *dr.*, *zo'n* of *10.000*.
    2. Knip met een patroon, een reeks letters en cijfers of één leesteken, en vergelijk het resultaat met de woorden van een treebank.
    3. Zorg dat afkortingen en initialen geen zin beëindigen: een lijst voor de woorden, een patroon voor de vorm.
    4. Staan er geen spaties, knip dan met een dictionary: neem steeds het langste woord dat past, en vergelijk vooruit met achteruit.
    5. Tel wat er misgaat, en bekijk de gevallen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `S` bevat een zin van Alpino. De cel knipt die zin bij de spaties met `\S+`, en laat dan de woorden zien waarvan het laatste teken een punt, een vraagteken of een uitroepteken is: elk van die woorden zou een zin kunnen beëindigen.
    """)
    return


@app.cell
def example_cut():
    import re

    S = C.example()
    print(S)
    words = re.findall(r"\S+", S)
    print(words)
    print([w for w in words if re.search(r"[.?!]$", w)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `\S+` laat elke punt aan het woord ervoor zitten, dus vier woorden eindigen op een punt, en alleen het laatste beëindigt de zin: *drs.* is een afkorting, en *J.* en *D.* zijn initialen. Een regel die na elke punt een zin laat eindigen, zou deze zin in vier stukken knippen. Oefening 2 zorgt ervoor dat de eerste drie de zin niet beëindigen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tokens():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. tokens: woorden en leestekens

        `SENTENCE` hieronder bevat een zin van Alpino met de namen van drie steden. Schrijf `tokens(text)`: de tokens van `text`, op volgorde, waarbij een token een reeks letters, cijfers en `_` is, of één teken dat geen letter, geen cijfer en geen witruimte is. Een komma of een vraagteken is dus een eigen token, en witruimte is geen token. Gebruik `re.findall` met één patroon van twee keuzes, verbonden door `|`. Voor `tokens(SENTENCE)` is de uitkomst `["Maar", "waar", "blijven", "Amsterdam", ",", "Rotterdam", "en", "Den", "Haag", "?"]`. Vervang de `...`. De check roept `tokens(SENTENCE)` aan, en daarna eigen gevallen: een kloktijd, een apostrof, twee tekens achter elkaar, drie punten, letters met accenten, haakjes, extra spaties, een regeleinde en een lege tekst.
        """),
        L.task("Knip een tekst in woorden en leestekens met één patroon en re.findall."),
    ])
    return


@app.cell
def sentence_cell():
    SENTENCE = C.cities()
    print(SENTENCE)
    return


@app.cell(hide_code=True)
def exercise_tokens():
    box_tokens = L.code_box("tokens", lines=4)
    box_tokens
    return (box_tokens,)


@app.cell(hide_code=True)
def check_tokens(box_tokens):
    result_tokens = L.check("tokens", box_tokens.value)
    result_tokens
    return (result_tokens,)


@app.cell(hide_code=True)
def hints_tokens():
    L.hints(
        "tokens",
        [
            "Geef terug wat `re.findall` in `text` vindt, met één patroon van twee keuzes verbonden door `|`: eerst een reeks letters en cijfers, dan één teken dat iets anders is, maar geen witruimte.",
            "Welk stuk uit de tabel matcht één teken dat niet in een set staat? En welke twee korte stukken horen hier in die set?",
            r'`    return re.findall(r"\w+|___", text)`',
        ],
        why=r'`return re.findall(r"\w+|[^\w\s]", text)`: een reeks letters, cijfers en `_`, of anders één teken dat geen van die drie is en geen witruimte.',
    )
    return


@app.cell(hide_code=True)
def text_after_tokens():
    mo.md(r"""
    De afbeelding laat zien waar je tokens en de treebank uit elkaar gaan. Meestal is het een koppelteken: Alpino houdt *Zuid-Afrika* en *1-0* heel, en de regel splitst zulke woorden, in 1.305 zinnen. Daarna komen afkortingen (465 zinnen), woorden met een apostrof zoals *'t* en *zo'n* (432), initialen (320) en getallen zoals *10.000* en *2,5* (227). Sommige zinnen van Alpino schrijven een openend aanhalingsteken als twee komma's, `,,`, en een sluitend aanhalingsteken als twee apostroffen, `''`; de treebank houdt elk paar bij elkaar als één woord. Op de Engelse treebank EWT knipt dezelfde regel 7.808 van de 11.723 zinnen precies: EWT splitst *don't* in *do* en *n't*, en de regel knipt het in *don*, *'* en *t*. Een regel past bij een standaard, en elke treebank heeft een eigen standaard.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ends():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ends_sentence: een einde, of een afkorting?

        Als je een tekst bij de spaties knipt, zoals in het uitgewerkte voorbeeld, beëindigt een woord een zin als het laatste teken `.`, `?` of `!` is, tenzij het een afkorting of een initiaal is. Schrijf `ends_sentence(word, abbreviations)`: `False` als `word` in kleine letters in de set `abbreviations` staat, of als `word` één hoofdletter van A tot en met Z is met een punt erachter, zoals *J.*; anders `True` als het laatste teken van `word` `.`, `?` of `!` is, en `False` als dat niet zo is. `ABBREVIATIONS` hieronder is de eigen korte lijst van de course met 30 Nederlandse afkortingen, in kleine letters. Gebruik `.lower()`, `in` en `re.search`, en vergelijk wat `re.search` geeft met `None`. Voor `ends_sentence("drs.", ABBREVIATIONS)` is de uitkomst `False`. Vervang de `...`. De check roept `ends_sentence("drs.", ABBREVIATIONS)` aan, en daarna eigen gevallen: een woord aan het eind van een zin, een initiaal, een afkorting met een hoofdletter, een woord in hoofdletters, een vraag, een uitroep, een getal met een punt erin, een woord zonder teken, nog een afkorting met een hoofdletter, en een woord met een lege set afkortingen.
        """),
        L.task("Zeg of een woord een zin beëindigt, met een lijst afkortingen en een patroon voor initialen."),
    ])
    return


@app.cell
def abbreviations_cell():
    ABBREVIATIONS = C.abbreviations()
    print(sorted(ABBREVIATIONS))
    return


@app.cell(hide_code=True)
def exercise_ends():
    box_ends = L.code_box("ends-sentence", lines=6)
    box_ends
    return (box_ends,)


@app.cell(hide_code=True)
def check_ends(box_ends):
    result_ends = L.check("ends-sentence", box_ends.value)
    result_ends
    return (result_ends,)


@app.cell(hide_code=True)
def hints_ends():
    L.hints(
        "ends-sentence",
        [
            "Geef eerst `False` terug voor een afkorting of een initiaal, met één `if` en `or`: staat het woord in kleine letters in `abbreviations`, of vindt `re.search` er een initiaal in? Geef daarna terug of het woord op een van de drie tekens eindigt.",
            "Een initiaal is het hele woord, geen deel van een langer woord: welke ankers heeft het patroon nodig?",
            r'`    if word.lower() in abbreviations or re.search(r"___", word):`',
        ],
        why=r'`if word.lower() in abbreviations or re.search(r"^[A-Z]\.$", word):` geeft `False` terug, en daarna `return re.search(r"[.?!]$", word) is not None`.',
    )
    return


@app.cell(hide_code=True)
def text_after_ends():
    mo.md(r"""
    Met de lijst en de initialen gaat de precision van 89,2% naar 96,4%: van de 1.326 onterechte einden van de eenvoudigste regel blijven er 415 over, zoals *v.*, *nl.* en *zg.*, afkortingen die niet op de lijst staan, en een paar echte einden binnen wat Alpino als één zin telt, zoals *Cryptogram.* De recall daalt een beetje, van 80,8% naar 80,7%: in 13 zinnen is een afkorting of een initiaal het laatste woord, zoals *etc.*, *pct.* en *N.V.*, en dan sluit de punt ook de zin af. De meeste gemiste einden hebben geen enkel teken: 1.790 zinnen van Alpino eindigen zonder teken, zoals koppen, en 409 eindigen op een dubbele punt. Kiss en Strunk wijzen erop dat er steeds nieuwe afkortingen bij komen, zodat geen lijst ooit compleet is; Punkt leert de afkortingen in plaats daarvan uit de tekst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_start():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. longest_start: de stap van MaxMatch

        MaxMatch zet steeds dezelfde stap: het neemt het langste begin van de tekst dat een woord is, en gaat dan verder na dat woord. Schrijf `longest_start(text, words)`: het langste begin `text[:end]` van `text` dat in de set `words` staat; als geen enkel begin van `text` in `words` staat, het eerste teken van `text`. Probeer het langste begin eerst, zodat het eerste begin dat je in `words` vindt, het begin is dat je teruggeeft. `WORDS` hieronder bevat de 38.682 woorden van de Nederlandse zinnen van Tatoeba, in kleine letters. Voor `longest_start("zoterug", WORDS)` is de uitkomst `"zot"`: *zo* is ook een woord, maar *zot* is langer. Vervang de `...`. De check roept `longest_start("zoterug", WORDS)` aan, en daarna eigen gevallen: een hele zin, een tekst die zelf een woord is, een tekst die begint met een woord met kortere woorden erin, een tekst die met geen enkel woord begint, drie letters, een lange zin, een euroteken en één letter. In de afbeelding onder de check roept de eigen MaxMatch van de course je functie aan.
        """),
        L.task("Vind het langste begin van een tekst dat in een dictionary staat: de stap van MaxMatch."),
    ])
    return


@app.cell
def words_cell():
    WORDS = C.dictionary()
    print(len(WORDS))
    return


@app.cell(hide_code=True)
def exercise_start():
    box_start = L.code_box("longest-start", lines=5)
    box_start
    return (box_start,)


@app.cell(hide_code=True)
def check_start(box_start):
    result_start = L.check("longest-start", box_start.value)
    result_start
    return (result_start,)


@app.cell(hide_code=True)
def hints_start():
    L.hints(
        "longest-start",
        [
            "Ga met een `for`-loop over een `range` die aftelt langs de mogelijke einden, zodat het langste begin eerst komt. Geef het eerste `text[:end]` terug dat in `words` staat. Geef na de loop `text[:1]` terug.",
            "Hoeveel tekens heeft het langste begin van `text`?",
            r'`    for end in range(___, 0, -1):`',
        ],
        why=r'`for end in range(len(text), 0, -1):`, dan `if text[:end] in words:` met `return text[:end]`, en na de loop `return text[:1]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_start():
    mo.md(r"""
    Met je functie knipt de MaxMatch van de course 1.247 van de 2.000 zinnen goed, 62,4%; backward MaxMatch knipt er 1.162 goed. Elk woord van deze zinnen staat in de dictionary, dus elke fout komt doordat MaxMatch steeds het langste woord neemt: *ikwastelaatopschool* wordt *ik waste laat op school*. Het Chinees heeft kortere woorden, en daar doet MaxMatch het bijna zo goed als mensen: Sproat en collega's vergeleken een gretige MaxMatch met hun zes moedertaalsprekers, en de overeenkomst was gemiddeld 0,73, tegen 0,76 tussen de sprekers onderling.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: MaxMatch op een eigen tekst

    Typ een tekst, met of zonder spaties: de spaties gaan eruit. De afbeelding knipt de tekst met MaxMatch vooruit en achteruit, met de woorden van Tatoeba als dictionary. Probeer *neemdeanderestoel*, *dekatisweggelopen*, *gisterenwasunietopschool*, of een eigen hashtag.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="ikbenzoterug", label="Tekst", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    *neemdeanderestoel* wordt vooruit *neem dean der es toe l*, en achteruit klopt het. *dekatisweggelopen* wordt vooruit *dek at is weggelopen*. En *gisterenwasunietopschool* gaat beide kanten op mis, als *gisteren was unie top school*. Een dictionary zegt welke woorden er bestaan, niet welke woorden de schrijver bedoelde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Welke oplossing past?

    Elk vetgedrukt geval vraagt iets van een programma dat tekst in woorden en zinnen knipt. Zet het bij de oplossing die het nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_fix = L.sort("which-fix")
    sort_fix
    return (sort_fix,)


@app.cell(hide_code=True)
def check_sort(sort_fix):
    result_sort = L.check("which-fix", sort_fix)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl) vond stukken tekst met een vaste vorm. Hier knipte een patroon een hele tekst in tokens: een reeks letters en cijfers, of één leesteken, met `|` voor de keuze en `[^...]` voor één teken buiten een set. Welke stukken één token zijn, legt een standaard vast, en de standaard van Alpino houdt bij elkaar wat een eenvoudige regel splitst: woorden met een koppelteken of een apostrof, afkortingen, initialen, getallen. Bij zinnen is de punt het probleem: een lijst zorgt dat afkortingen geen zin beëindigen, een patroon doet dat voor initialen, en je mat wat dat met de precision en de recall doet. Waar een tekst geen spaties heeft, knipt MaxMatch de tekst met een dictionary en het langste woord, en gaat het mis waar toevallig een langer woord past. De volgende les van het domein, *Normaliseren: hoofdletters, accenten en stop words*, maakt van *Café* en *cafe* één woord, laat heel gewone woorden weg, en laat zien wat daarbij verloren gaat.

    **Een voorbeeld uit het dagelijks leven.** De woordenteller van een tekstverwerker knipt tekst ook in woorden: of *zo'n* of *e-mail* als één woord telt of als twee, hangt af van de regels van die woordenteller.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ends, result_sort, result_start, result_tokens):
    L.wrap_up(result_tokens, result_ends, result_start, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
