# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-information-extraction. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-information-extraction", "nl")
    C = L.checks   # de alinea's van 48 Wikipedia-pagina's in het Engels en het Nederlands, de lijst van de course met de datums erin, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl), en een over [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_groups():
    quick_groups = L.choice(
        "quick-groups",
        question=r'In *Patronen vinden: reguliere expressies* haalde een patroon met groepen een tijd uit elkaar. Wat geeft `re.findall(r"(\d{1,2}):(\d\d)", "om 9:30 en 14:05")`?',
        options={
            "whole": '`["9:30", "14:05"]`: de hele matches',
            "right": '`[("9", "30"), ("14", "05")]`',
            "numbers": "`[(9, 30), (14, 5)]`: de delen als gehele getallen",
            "flat": '`["9", "30", "14", "05"]`: elk deel in één lijst van strings',
        },
        correct="right",
        why={
            "right": "Met meer dan één groep geeft `re.findall` voor elke match een tuple met wat elke groep matchte, als strings: de 05 houdt de 0. Deze les maakt rijen van zulke tuples.",
            "whole": "Dat geeft `re.findall` voor een patroon zonder groepen. Wat verandert er als het patroon er twee heeft? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "numbers": "De delen komen rechtstreeks uit de tekst. Wat voor waarde is een stuk tekst? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "flat": "`re.findall` houdt de delen van één match bij elkaar. Waarin? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
        },
    )
    quick_groups
    return (quick_groups,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="Een regel vindt 50 datums in een tekst. Daarvan zijn er 40 echte datums uit de tekst; de andere 10 niet. De tekst bevat in totaal 80 datums. Wat zijn de precision en de recall van de regel?",
        options={
            "swapped": "Precision 0,5, recall 0,8",
            "right": "Precision 0,8, recall 0,5",
            "share": "Precision 0,625, recall 0,8: 50 van de 80 datums, en 40 van de 50",
            "errors": "Precision 0,2, recall 0,5",
        },
        correct="right",
        why={
            "right": "Precision is het deel van wat de regel vond dat goed is: 40 van de 50, 0,8. Recall is het deel van de datums in de tekst dat de regel vond: 40 van de 80, 0,5. Deze les telt allebei voor regels die datums uit tekst halen.",
            "swapped": "Welke van de twee gaat over wat de regel vond, en welke over wat de tekst bevat? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "share": "Precision en recall tellen allebei de 40 datums die goed zijn. Van hoeveel? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "errors": "0,2 is het deel van de gevonden datums dat fout is, 10 van de 50. Welk deel is precision? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tekst vertelt je feiten, maar pas in een tabel kan een programma de feiten sorteren, tellen of opzoeken. Jurafsky en Martin beginnen het hoofdstuk over information extraction met een analist die aandelen van luchtvaartmaatschappijen volgt: bij elke aankondiging van hogere tarieven wil de analist weten welke maatschappij het was, hoeveel erbij kwam, op welke datum, en welke maatschappijen volgden. Zulke feiten staan in nieuwsberichten, zoals een bericht dat United Airlines op vrijdag de tarieven met $6 per retourvlucht verhoogde vanwege de hoge brandstofprijzen, en dat American Airlines meteen volgde. Information extraction zet volgens Jurafsky en Martin de ongestructureerde informatie in teksten om in gestructureerde data, bijvoorbeeld om er een relationele database mee te vullen voor verdere verwerking.

    Deze les haalt de datums uit 2.067 alinea's van 48 pagina's van de Engelse Wikipedia: de pagina's van de development set van SQuAD (Rajpurkar en anderen, 2016), een verzameling vragen over Wikipedia, met het Nederlands uit SQuAD-NL (de Vries en anderen, 2023), grotendeels machinevertaling. De course zocht elke datum erin die een dag, een maand en een jaar noemt, eerst met patronen zoals in deze les, en daarna met de hand op elke plek waar een dag en een maand naast elkaar staan: 245 datums, in beide talen dezelfde. Met deze lijst kun je precies tellen hoeveel datums een regel vindt, mist en fout heeft.

    **Een voorbeeld uit het dagelijks leven.** Je krijgt een mail: het etentje is op 12 maart om zeven uur. Je mailprogramma biedt aan om het in je agenda te zetten. Het haalde de datum en de tijd uit de tekst, en maakte er een dag en een uur van die de agenda begrijpt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_forms = L.guess(
        "guess",
        question="Hoeveel van de 245 datums in de Engelse alinea's zijn geschreven met de maand eerst, zoals *February 7, 2016*, en niet met de dag eerst, zoals *7 February 2016*?",
        options={
            "few": "Ongeveer 10, van een paar Amerikaanse pagina's",
            "fifty": "Ongeveer 50",
            "right": "Ongeveer 110",
            "all": "Bijna allemaal: Wikipedia begon in de Verenigde Staten en schrijft zoals Amerikanen het doen",
        },
        reveal="**108 van de 245: bijna de helft.** De stijlgids van de Engelse Wikipedia, het *Manual of Style*, vraagt om één vorm binnen een pagina: de dag eerst of de maand eerst. En een onderwerp met sterke banden met een Engelstalig land krijgt in het algemeen de vorm die in dat land het gewoonst is. Zo schrijven *Super Bowl 50*, *Apollo program* en *American Broadcasting Company* de maand eerst, en *Martin Luther*, *Kenya* en *Doctor Who* de dag eerst; 28 van de 30 pagina's met datums houden zich aan één vorm. De Nederlandse vertaling schrijft elke datum met de dag eerst. Een regel voor één vorm vindt ongeveer de helft van de Engelse datums.",
    )
    guess_forms
    return (guess_forms,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Information extraction** zet vrije tekst om in gestructureerde data: rijen met vaste velden. Vaak weet je van tevoren welke velden je wilt. Jurafsky en Martin noemen dat template filling: in documenten de gebeurtenissen of situaties vinden die steeds op dezelfde manier terugkomen, en de slots van een template invullen. Een **template** bevat de velden voor één soort gebeurtenis, en een **slot** is een van de velden. Voor het bericht over de tarieven heeft de template vier slots, en Jurafsky en Martin vullen de slots met *United Airlines* als de maatschappij die begon, *$6* als het bedrag, *2006-10-26* als de datum waarop de verhoging inging, en *American Airlines* als de maatschappij die volgde.

    Drie van de vier slots bevatten tekst die rechtstreeks uit het bericht komt. De datum niet: volgens Jurafsky en Martin kan een slot een stuk van de tekst zelf bevatten, of iets als een tijd, een bedrag of een begrip dat met extra verwerking is afgeleid. Het bericht zegt alleen *vrijdag*, en in een latere zin *donderdag*; welke dagen dat zijn, volgt uit de dag waarop het bericht verscheen. Echte teksten zijn vaak zo: volgens Jurafsky en Martin noemt een aanduiding van een tijd in echte teksten vrij zelden de volledige datum. Deze les houdt het bij datums die een dag, een maand en een jaar noemen.

    De vorm van zo'n datum kun je schrijven als reguliere expressie, met een groep voor elk veld. `re.findall` geeft dan voor elke datum een tuple van strings, en van elke string maak je de waarde van het veld: cijfers met `int`, de naam van een maand met een dictionary die het nummer geeft. Een verticale streep in een patroon betekent *of*: `(mei|juni)` matcht *mei* of *juni*. De haakjes houden de keuze binnen de groep; zonder haakjes splitst de streep het hele patroon in tweeën. Zo werken de meeste systemen voor datums nog steeds: volgens Jurafsky en Martin werken de huidige aanpakken om tijden en datums in een vaste vorm te zetten meestal met regels.

    Een veld van een tabel schrijf je in één vaste vorm, zodat een programma de waarden kan vergelijken en sorteren. Voor datums is dat de internationale standaard **ISO 8601**: het jaar in vier cijfers, de maand in twee, de dag in twee, met streepjes ertussen. Jurafsky en Martin geven als voorbeeld 2 juli 2007, in de vorm YYYY-MM-DD: 2007-07-02. *23 november 1963* en *November 23, 1963* worden allebei 1963-11-23.

    Hoe goed is een regel? Je vergelijkt de rijen ervan met een lijst van de goede rijen, een **gold standard**. Net als in *Meer dan accuracy: precision en recall* is een rij die de regel vindt en die in de lijst staat een true positive, een rij die de regel vindt en die niet in de lijst staat een false positive, en een rij uit de lijst die de regel niet vindt een false negative. Precision is het deel van de gevonden rijen dat goed is, en recall het deel van de goede rijen dat gevonden is. Een rij met één fout veld, zoals de verkeerde dag, telt dubbel: als false positive, en de goede rij die het had moeten zijn als false negative.

    De afbeelding laat vier zinnen van vier pagina's zien, en de tabel die de course van de datums maakt.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Elke datum wordt één rij met een jaar, een maand en een dag, en de naam van de maand wordt een nummer: *juli* wordt 7. De laatste kolom schrijft elke rij in de vorm van ISO 8601. De tijd in de derde zin, 20:17:40, is ook een feit, maar heeft geen slot in deze tabel.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **information extraction** | vrije tekst omzetten in gestructureerde data, zoals rijen met vaste velden |
    | **template** | de velden die je wilt voor één soort gebeurtenis of feit, zoals de maatschappij, het bedrag en de datum van hogere tarieven |
    | **slot** | één veld van een template |
    | **ISO 8601** | de internationale standaard om datums te schrijven: YYYY-MM-DD, zoals 1963-11-23 |
    | **gold standard** | de lijst met goede antwoorden waarmee je de uitkomst van een regel vergelijkt |
    | `"-".join(d)` | een dictionary `d` samenvoegen voegt de keys samen, hier met `-` ertussen: `"-".join({"mei": 5, "juni": 6})` geeft `"mei-juni"` |
    | `for day, month, year in re.findall(pattern, text)` | bij een patroon met drie groepen geeft `re.findall` voor elke match een tuple van drie strings, en de loop haalt elk tuple uit elkaar in drie namen |
    | `f"{n:02d}"` | een f-string die het gehele getal `n` schrijft met minstens twee cijfers, met een 0 ervoor als het getal één cijfer heeft: `f"{7:02d}"` geeft `"07"` |
    | `C.MONTHS_EN`, `C.MONTHS_NL` | de namen van de maanden in het Engels en in het Nederlands, met de nummers erbij: `{"January": 1, …}` en `{"januari": 1, …}` |
    | `C.text(lang, i)` | alinea `i` van de 2.067, vanaf 0 geteld, in het Engels (`"en"`) of het Nederlands (`"nl"`) |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Kies de velden van een rij: hier het jaar, de maand en de dag.
    2. Schrijf een patroon met een groep voor elk veld, en kijk wat het patroon in echte tekst vindt.
    3. Maak van elke groep de waarde van het veld: cijfers met `int`, een naam met een dictionary.
    4. Schrijf elke waarde in één vaste vorm, zoals ISO 8601 voor datums.
    5. Tel de goede, foute en gemiste rijen tegen een gold standard, en verbeter het patroon waar het tekortschiet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code vult drie slots van de template uit de eerste twee zinnen van het bericht, hier in het Nederlands: de maatschappijen met een groep voor het woord vóór *Airlines*, en het bedrag met een groep voor de cijfers na het dollarteken. In een patroon is `\$` het dollarteken zelf; een `$` alleen betekent het einde van de tekst.
    """)
    return


@app.cell
def example_cell():
    import re

    _report = ("Vanwege de hoge brandstofprijzen zei United Airlines vrijdag de tarieven met $6 per retourvlucht te "
               "hebben verhoogd op vluchten naar steden waar ook goedkopere maatschappijen vliegen. American Airlines, "
               "een onderdeel van AMR Corp., volgde meteen")
    _airlines = re.findall(r"([A-Z][a-z]+) Airlines", _report)
    _amounts = re.findall(r"\$(\d+)", _report)
    _row = {"eerste": _airlines[0] + " Airlines", "bedrag": int(_amounts[0]), "volger": _airlines[1] + " Airlines"}
    print(_airlines)
    print(_amounts)
    print(_row)
    return (re,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het patroon voor de maatschappijen is een gok die hier werkt: een woord met een hoofdletter vóór *Airlines*. *United* krijgt het slot van de maatschappij die begon omdat het als eerste komt, en *American* het slot van de volger omdat het als tweede komt: de regel weet niets van wie wie volgt, alleen van de volgorde in de tekst. Het bedrag is nu een geheel getal, 6, zodat een programma het kan optellen. De datum waarop de verhoging inging blijft leeg: het bericht zegt *vrijdag*, en in een latere zin *donderdag*, en welke dagen dat zijn, hangt af van de datum van het bericht, en het bericht zelf geeft geen datum.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. date_rows: de datums van een tekst als rijen

        Schrijf `date_rows(text, months)`, waarbij `text` een string is en `months` een dictionary met namen van maanden en de nummers ervan, zoals `{"januari": 1, "februari": 2, …}`. Geef een lijst terug met een rij voor elke datum in `text` die geschreven is als een dag, een naam van een maand en een jaar, zoals *6 december 1989*, in de volgorde van de tekst. Een dag is één of twee cijfers en een jaar vier, en geen van beide is een deel van een langer getal of woord. Een rij is een tuple `(year, month, day)` van drie gehele getallen. De eerste regel van de functie voegt de namen van de maanden samen tot `names`, zoals `januari|februari|…`. Vervang de twee keer `...`: het patroon, met een groep voor de dag, een voor de maand en een voor het jaar, en de regel die de rij toevoegt aan `rows`.

        De check roept `date_rows(sentence_en, months_en)` en `date_rows(sentence_nl, months_nl)` aan, met de zin *Doctor Who originally ran for 26 seasons on BBC One, from 23 November 1963 until 6 December 1989.*, het Nederlands ervan, *Doctor Who liep oorspronkelijk 26 seizoenen op BBC One, van 23 november 1963 tot 6 december 1989.*, en de namen van de maanden in het Engels en in het Nederlands (in de les: `C.SENTENCE_EN`, `C.SENTENCE_NL`, `C.MONTHS_EN` en `C.MONTHS_NL`). Beide aanroepen geven `[(1963, 11, 23), (1989, 12, 6)]`. Daarna eigen gevallen: een Engelse zin met *1 May 1191*; een met *4 August 1915* en *November 1918*; een met *February 7, 2016*; een met *series 24 in 1991*; een Nederlandse zin met *9 oktober 2004*; een lege tekst; een tekst met *3 Feb 2020* en *4 March 2020*, met de maanden `{"Jan": 1, "Feb": 2, "Mar": 3}`; en de alinea's 371 (*Martin Luther*) en 6 (*Super Bowl 50*), vanaf 0 geteld, in het Engels en in het Nederlands (in de les: `C.text("en", 371)` enzovoort).
        """),
        L.task("Haal de datums van een tekst uit elkaar in rijen met een jaar, een maand en een dag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_rows = L.code_box("date_rows", lines=10)
    box_rows
    return (box_rows,)


@app.cell(hide_code=True)
def check_rows(box_rows):
    result_rows = L.check("date_rows", box_rows.value)
    result_rows
    return (result_rows,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "date_rows",
        [
            r"`pattern` is een raw string met drie groepen: één of twee cijfers met `\d{1,2}`, dan een spatie en een groep rond `names`, ertussen gezet met `+`, dan een spatie en vier cijfers, met `\b` aan beide kanten. In de loop voegt `rows.append` een tuple toe: `int` maakt van `year` en `day` gehele getallen, en de dictionary `months` geeft het nummer van de naam van de maand in `month`.",
            r'Zet `names` met `+` tussen `r"\b(\d{1,2}) ("` en `r") (\d{4})\b"`: de haakjes eromheen maken van de keuze van een maand één groep. Dan de rij: hoe zoek je de waarde van een key op in een dictionary?',
            "`        rows.append((int(year), ___, int(day)))`",
        ],
        why=r'`pattern = r"\b(\d{1,2}) (" + names + r") (\d{4})\b"` en, in de loop, `rows.append((int(year), months[month], int(day)))`. Het voorbeeld geeft `[(1963, 11, 23), (1989, 12, 6)]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_rows():
    mo.md(r"""
    Op de Nederlandse alinea's vindt je regel 239 van de 245 datums, en maakt de regel geen enkele foute rij. De Nederlandse vertaling schrijft elke datum met de dag eerst. De 6 datums die de regel mist, zijn op een andere manier geschreven, zoals *28 januari tot 25 mei 1521*, waar één jaar bij twee dagen hoort. Op de Engelse alinea's vindt dezelfde regel slechts 132 datums: *Super Bowl 50*, *Apollo program* en *American Broadcasting Company* schrijven de maand eerst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. score: goed, fout en gemist

        Schrijf `score(found, gold)`, waarbij `found` de lijst is met de rijen die een regel vond, en `gold` de lijst met de goede rijen, de gold standard. Een rij kan elk tuple zijn, en geen enkele rij staat twee keer in een lijst. Geef een tuple `(right, wrong, missed)` terug: hoeveel rijen uit `found` in `gold` staan, hoeveel niet, en hoeveel rijen uit `gold` niet in `found` staan. De eerste regels maken van beide lijsten een set, zodat `in` snel is, ook bij lange lijsten. Vervang de twee keer `...`.

        De check roept `score(found_rows, gold_rows)` aan, met de rijen `[(1963, 11, 23), (1989, 12, 7), (2005, 3, 26)]` en de goede rijen `[(1963, 11, 23), (1989, 12, 6), (2005, 3, 26), (2008, 7, 27)]` (in de les: `C.FOUND_ROWS` en `C.GOLD_ROWS`). Dat geeft `(2, 1, 2)`: de rij met 7 december is fout, en 6 december is gemist, net als de rij van 2008. Daarna eigen gevallen: niets gevonden; geen goede rijen; beide lijsten leeg; elke goede rij gevonden; twee rijen tegen drie; en, als rijen `(alinea, jaar, maand, dag)` tegen de 245 datums van de course, de rijen van de regel van de course voor de dag eerst op alle Engelse en op alle Nederlandse alinea's, en van een lossere regel op de Engelse.
        """),
        L.task("Tel de goede, foute en gemiste rijen van een regel tegen de gold standard."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=8)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "In de loop telt `if row in gold:` 1 op bij `right`. Na de loop geeft `return` drie getallen: `right`; de rijen uit `found` die niet goed zijn, `len(found) - right`; en de rijen uit `gold` die niet gevonden zijn.",
            "`right` telt de rijen uit `found` die ook in `gold` staan, en elke andere rij uit `found` is fout. Hoeveel rijen uit `gold` blijven er over die de regel niet vond?",
            "`    return right, len(found) - right, ___`",
        ],
        why="In de loop: `if row in gold:` met daaronder `right = right + 1`, en na de loop `return right, len(found) - right, len(gold) - right`. Het voorbeeld geeft `(2, 1, 2)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    De regel voor de dag eerst heeft op de Nederlandse alinea's een precision van 1,00: elke rij die de regel vindt, is goed. De recall is 0,98: de regel mist alleen de 6 datums die op een andere manier geschreven zijn. Een lossere regel, met elk woord tussen de dag en het jaar, vindt dezelfde datums en voegt er 7 foute rijen aan toe, zoals *5 in 1970* uit *13,5 in 1970*: een precision van 0,97. Een tweede patroon voor de maand eerst verandert hier niets. Op de Engelse alinea's heeft de regel voor de dag eerst een recall van slechts 0,54, en daar tilt het tweede patroon de recall naar 0,98.

    Deze getallen zeggen net zoveel over de lijst als over de regels: de course vond 239 van de 245 datums met dezelfde twee patronen, en de andere 6 door met de hand elke plek na te lopen waar een dag en een maand naast elkaar staan. Een gold standard die met een regel gemaakt is, laat pas zien wat de regel mist als mensen de lijst nalopen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_iso():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. iso_dates: één vaste vorm

        Schrijf `iso_dates(rows)`, waarbij `rows` een lijst is met rijen `(year, month, day)`. Geef een lijst terug met elke rij geschreven als datum van ISO 8601, een tekst zoals `"1969-07-20"`, gesorteerd. Vervang de twee keer `...`.

        De check roept `iso_dates(apollo_rows)` aan, met drie datums van de pagina *Apolloprogramma*: `[(1969, 7, 20), (1961, 5, 25), (1967, 1, 27)]` (in de les: `C.APOLLO_ROWS`). Dat geeft `["1961-05-25", "1967-01-27", "1969-07-20"]`. Daarna eigen gevallen: geen rijen; één rij; drie datums van *Super Bowl 50* in 2012 en 2013; de 15 datums van *Apolloprogramma*; en alle 245 datums van de course.
        """),
        L.task("Schrijf rijen als datums van ISO 8601, op volgorde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iso():
    box_iso = L.code_box("iso_dates", lines=6)
    box_iso
    return (box_iso,)


@app.cell(hide_code=True)
def check_iso(box_iso):
    result_iso = L.check("iso_dates", box_iso.value)
    result_iso
    return (result_iso,)


@app.cell(hide_code=True)
def hints_iso():
    L.hints(
        "iso_dates",
        [
            "In de loop voegt `dates.append` een f-string toe met `year`, `month` en `day`, een streepje tussen elke twee, en `:02d` na elke naam die twee cijfers moet krijgen. Na de loop geeft `return` de teksten op volgorde met `sorted`.",
            "ISO 8601 schrijft de maand en de dag elk met twee cijfers: 7 wordt 07. Tekst sorteert op het eerste teken, dan op het volgende. Waarom zet dat datums in deze vorm op volgorde van de tijd?",
            '`        dates.append(f"{year}-{month:02d}-{___}")`',
        ],
        why='In de loop: `dates.append(f"{year}-{month:02d}-{day:02d}")`, en daarna `return sorted(dates)`. Het voorbeeld geeft `["1961-05-25", "1967-01-27", "1969-07-20"]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_iso():
    mo.md(r"""
    Gesorteerd als tekst begint de lijst van datums zoals op de pagina met *11 juli 1962* en *11 juni 1962*, en eindigt de lijst met *8 februari 1974* en *9 november. 1967* (met een punt erin, zo staat het in de vertaling): op volgorde van het eerste cijfer, niet van de tijd. Jouw ISO-datums lopen van 1961-04-12 tot 1974-02-08. In de vorm van ISO 8601 is sorteren als tekst hetzelfde als sorteren op datum, omdat de grootste eenheid vooraan staat en elk deel een vast aantal cijfers heeft.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: een eigen patroon

    Het vak begint met een patroon voor een dag, een willekeurig woord en een jaar. De afbeelding legt wat het patroon vindt in de 2.067 alinea's van jouw taal naast de 245 datums van de course: een datum telt als gevonden als één match de hele datum bevat, en een match zonder datum is fout. Jurafsky en Martin waarschuwen bij regels voor tijden en datums: de grootste moeilijkheid is om uitdrukkingen te vermijden die een false positive geven, zoals de 1984 in een zin over het boek *1984*. Kun je de foute matches laten verdwijnen zonder datums te verliezen? Kun je in het Engels ook de datums vinden met de maand eerst? Welke datums vindt geen enkel patroon van jou, en waarom niet?
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    pattern_box = mo.ui.text(value=C.PATTERN_START, label="Patroon", full_width=True)
    pattern_box
    return (pattern_box,)


@app.cell(hide_code=True)
def try_look(pattern_box):
    mo.Html(C.pattern_picture(pattern_box.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_names():
    mo.md(r"""
    ### Oefening 4. Datums en namen

    Datums hebben een paar vaste vormen, dus een patroon kan de datums beschrijven. Probeer hetzelfde voor de namen van mensen: een patroon voor een reeks van twee of meer woorden die elk met een hoofdletter beginnen en met kleine letters verdergaan, zoals *Nikola Tesla*. In de 11.722 Engelse zinnen van Universal NER (Mayhew en anderen, 2024) markeerden mensen elke naam van een persoon: 2.028 namen. Het patroon vindt 3.000 reeksen, en 510 daarvan zijn namen van mensen: een precision van 0,17, en een recall van 0,25.
    """)
    return


@app.cell(hide_code=True)
def choice_names_cell():
    choice_names = L.choice(
        "names",
        question="Waarom werkt een patroon zoveel beter voor datums dan voor de namen van mensen?",
        options={
            "case": "`re` kan een hoofdletter niet van een kleine letter onderscheiden",
            "right": "Een naam heeft geen vaste vorm: hoofdletters beginnen ook plaatsen, bedrijven en zinnen, en veel namen zijn één woord",
            "rare": "Namen komen minder vaak voor in tekst dan datums",
            "marks": "Het patroon klopt, maar de mensen die de zinnen markeerden misten de meeste namen, dus de goede reeksen tellen als fout",
        },
        correct="right",
        why={
            "right": "Een datum is een getal van 1 tot 31, een van twaalf namen van maanden en vier cijfers; een naam heeft niet zo'n vorm. *Iguazu Falls* en *Buenos Aires* zien eruit als namen en zijn plaatsen, en *Callum* is een naam van één woord. Of een reeks woorden een naam is, hangt af van de woorden eromheen. De les [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl) leert dat uit voorbeelden, met een model.",
            "case": "`[A-Z]` matcht alleen hoofdletters, en `[a-z]` alleen kleine letters. Waar komen de foute reeksen dan vandaan?",
            "rare": "2.028 namen in 11.722 zinnen is genoeg om te vinden. Waarom zijn de meeste reeksen geen namen?",
            "marks": "Een markering kan af en toe fout zijn, maar *Iguazu Falls* is een plaats, en het patroon neemt de plaats toch. Wat vertelt een hoofdletter je over een woord?",
        },
    )
    choice_names
    return (choice_names,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl) vond een patroon stukken tekst. Hier werd elke groep van een patroon een veld van een rij, en de rijen werden een tabel: information extraction met regels. Je `date_rows` haalde de datums van een tekst uit elkaar in jaren, maanden en dagen, je `score` telde goede, foute en gemiste rijen tegen een gold standard, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), en je `iso_dates` schreef de datums in de ene vorm van ISO 8601, waarin sorteren als tekst hetzelfde is als sorteren op datum. Een patroon vindt alleen de vormen die het beschrijft: bijna de helft van de Engelse datums stond in een andere vorm, en de namen van mensen hebben geen vorm die een patroon kan vangen. De volgende les, *Keywords en termen*, vindt de woorden die zeggen waar een document over gaat.

    **Een voorbeeld uit het dagelijks leven.** Een app voor je uitgaven leest de bonnetjes die je fotografeert: de datum, de winkel en het bedrag komen in de kolommen. Een bonnetje dat de datum op een andere manier schrijft, komt op de verkeerde dag terecht, of verschijnt helemaal niet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_names, result_iso, result_rows, result_score):
    L.wrap_up(result_rows, result_score, result_iso, choice_names)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
