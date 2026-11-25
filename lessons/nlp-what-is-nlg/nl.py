# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-what-is-nlg. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-what-is-nlg", "nl")
    C = L.checks   # de drie weken weer van de course, de grenzen, de woorden, de templates, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl), en één over [Tekst: strings](/course/open/py-strings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="In *Wat NLP is* heeft NLP twee kanten: begrijpen (NLU), van tekst naar betekenis, en maken (NLG), van data of betekenis naar tekst. Welk van deze systemen zit alleen aan de kant van NLG?",
        options={
            "spam": "Een spamfilter dat elke mail leest en indeelt",
            "right": "Een programma dat een wedstrijdverslag schrijft uit de uitslag",
            "translate": "Een programma dat een Nederlandse tekst in het Engels omzet",
            "steps": "Een stappenteller die de stappen van een dag optelt en er een grafiek van tekent",
        },
        correct="right",
        why={
            "right": "Er gaan getallen in en er komt tekst uit: dat is alleen NLG. Deze les bouwt zo'n programma, voor het weer.",
            "spam": "Een spamfilter leest tekst en geeft een label, geen tekst. Welke kant gaat van tekst naar betekenis? Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "translate": "Er gaat tekst in en er komt tekst uit, dus een vertaler zit aan beide kanten. Bij welk systeem gaat er geen tekst in? Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "steps": "Bij een stappenteller gaat er geen tekst in en komt er geen tekst uit: het systeem zit aan geen van beide kanten. Welk systeem maakt tekst uit getallen? Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_fstring():
    quick_fstring = L.choice(
        "quick-fstring",
        question="In *Tekst: strings* vult een f-string de waarde in van elke naam tussen accolades. Wat geeft `f\"Op {day} wordt het {word}.\"` met `day = \"maandag\"` en `word = \"droog\"`?",
        options={
            "plain": "`\"Op {day} wordt het {word}.\"`, met de accolades",
            "names": "`\"Op day wordt het word.\"`",
            "right": "`\"Op maandag wordt het droog.\"`",
            "error": "Een fout, want een f-string vult slechts één naam in",
        },
        correct="right",
        why={
            "right": "Python zet de waarde van `day` en van `word` in de gaten. Zo'n f-string is een template: vaste woorden met gaten, zoals deze les de zinnen schrijft.",
            "plain": "Zonder de `f` aan het begin zouden de accolades blijven staan. Wat laat de `f` Python met de accolades doen? Zie [Tekst: strings](/course/open/py-strings?lang=nl).",
            "names": "Python vult de waarde van elke naam in, niet de naam zelf. Wat is de waarde van `day`? Zie [Tekst: strings](/course/open/py-strings?lang=nl).",
            "error": "Een f-string mag zoveel namen bevatten als je wilt, elk tussen eigen accolades. Wat komt er in elk gat? Zie [Tekst: strings](/course/open/py-strings?lang=nl).",
        },
    )
    quick_fstring
    return (quick_fstring,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op 17 maart 2014 trilde de grond bij Beverly Hills, in Californië: een kleine aardbeving. De *Los Angeles Times* meldde de beving als eerste krant, binnen drie minuten. Albert Gatt en Emiel Krahmer, die in 2018 een overzicht van het vakgebied schreven, vertellen hoe: een 'robotjournalist' maakte het bericht zelf, door de binnenkomende metingen van de beving in de gaten van een vaste tekst in te vullen.

    Programma's schreven al jaren eerder weersverwachtingen. In 2005 beschreven Somayajulu Sripada, Ehud Reiter en Lezan Hawizy van de Universiteit van Aberdeen een programma dat verwachtingen schreef voor booreilanden op zee. Het weerbedrijf Weathernews liet het elke dag 150 conceptverwachtingen maken, en de meteorologen van het bedrijf verbeterden elk concept, zodat de klant pas de verbeterde tekst kreeg.

    In deze les schrijf je een weerbericht uit een tabel met getallen, in de drie stappen die zulke programma's zetten, en zie je wat elke stap beslist.

    **Een voorbeeld uit het dagelijks leven.** Een weer-app toont *Regen in het komende uur* op het vergrendelscherm van een telefoon: een zin die gemaakt is uit de getallen van een regenradar, niet getypt door een mens.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_readers = L.guess(
        "guess",
        question="SumTime-Mousam schreef verwachtingen voor booreilanden uit de getallen van een weermodel. Ehud Reiter en collega's lieten mensen die in het werk zulke verwachtingen lezen twee soorten teksten zien, geschreven uit dezelfde getallen: de teksten van het programma, en de teksten van meteorologen. Welke vonden de lezers beter?",
        options={
            "much": "De teksten van de meteorologen, met grote voorsprong",
            "slightly": "De teksten van de meteorologen, maar net",
            "same": "Geen van beide: de twee soorten teksten deden het ongeveer even goed",
            "right": "De teksten van het programma",
        },
        reveal="De teksten van het programma. Volgens Reiter en collega's wees een evaluatie door gebruikers van verwachtingen erop dat de lezers de teksten van SumTime-Mousam verkozen boven teksten van mensen, voor een deel door betere woordkeus. Misschien liet een evaluatie toen voor het eerst zien dat teksten van NLG beter zijn dan teksten van mensen, schrijven de onderzoekers. De meteorologen gebruikten woorden niet op dezelfde manier: verschillende woorden voor hetzelfde, en hetzelfde woord met verschillende betekenissen. Voor het tijdstip 00:00, merken Gatt en Krahmer op, kon een meteoroloog *late evening*, *midnight* of gewoon *evening* schrijven. Het programma gebruikte vaste regels van getallen naar woorden, en vermeed woorden die slechts een paar meteorologen gebruikten, en woorden die mensen verschillend lazen.",
    )
    guess_readers
    return (guess_readers,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    *Wat NLP is* noemde de kant van NLP die zelf tekst maakt: **natural language generation** (NLG). Deze les gaat over **data-to-text generation**, waarbij er data in gaat, zoals een tabel met getallen. Ehud Reiter, die zulke systemen bouwde aan de Universiteit van Aberdeen, schreef in 1996 dat bijna elk NLG-systeem in de praktijk, technisch gezien, drie taken uitvoert. Deze les gebruikt deze namen:

    1. **Content selection** bepaalt wat je zegt. Een tabel bevat meer dan een lezer nodig heeft, en het programma houdt wat ertoe doet als **boodschappen**, zoals *6,5 mm regen op woensdag*, en zet de boodschappen op volgorde. Gatt en Krahmer: in data zit meestal meer informatie dan je met een tekst wilt overbrengen, of de data is gedetailleerder dan je in een tekst wilt zeggen.
    2. **Microplanning** plant de zinnen. Deze stap kiest de woorden voor elke boodschap, zoals *winderig* bij windkracht 6; bepaalt welke boodschappen een zin delen, **aggregatie** geheten; en bepaalt hoe je verwijst naar iets wat de tekst al noemde.
    3. **Surface realization** schrijft elke zin uit volgens de regels van de taal: de volgorde van de woorden, de uitgangen, de komma's en de punt.

    Reiter noemde de drie stappen content determination en text planning, sentence planning, en realization. In het boek dat Reiter in 2000 met Robert Dale schreef, heten de stappen document planning, microplanning en realization, en Sripada en collega's beschrijven SumTime-Mousam in deze drie fasen. Gatt en Krahmer verdelen hetzelfde werk over zes taken: content determination en text structuring vormen samen de eerste fase, die Gatt en Krahmer de text planner noemen; sentence aggregation, lexicalization en referring expression generation vormen microplanning; en linguistic realization is surface realization. Deze les noemt de eerste stap naar de belangrijkste taak: de inhoud kiezen.

    Het plaatje toont de drie stappen op een week in april die de course heeft verzonnen.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **data-to-text generation** | NLG waarbij er data in gaat, zoals een tabel met getallen, en geen tekst |
    | **content selection** | de eerste stap: kiezen welke feiten uit de data de tekst noemt, en in welke volgorde; ook document planning of text planning genoemd |
    | **boodschap** | één feit dat content selection houdt, zoals `("rain", 6.5)`: een soort en de waarde |
    | **microplanning** | de middelste stap: de woorden kiezen, bepalen welke feiten een zin delen, en bepalen hoe je naar iets verwijst |
    | **aggregatie** | feiten in één zin samennemen, zoals twee dagen met hetzelfde weer |
    | **surface realization** | de laatste stap: elke zin uitschrijven volgens de regels van de taal, met de goede volgorde van de woorden, de uitgangen en de leestekens |
    | **template** | een zin met gaten die een programma invult, zoals een f-string |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo schrijf je een weerbericht uit een week met getallen:

    1. Houd voor elke dag de metingen die de grens halen, als boodschappen van een soort en de waarde, zoals `("rain", 6.5)`.
    2. Geef elke dag woorden voor de boodschappen, zoals *regenachtig en winderig*. Een dag zonder boodschappen is *droog*.
    3. Geef dagen achter elkaar met dezelfde woorden één zin.
    4. Vul de template van de taal met de dagen en de woorden: een komma tussen de dagen, en *en* bij de laatste.

    Stap 1 is content selection, stap 2 en 3 zijn microplanning, en stap 4 is surface realization. De dagen houden de volgorde, met maandag als eerste.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem woensdag in de week in april. `C.WEEKS` bevat de drie weken van de course, elk een lijst van zeven dagen; een dag is een dictionary met vijf metingen: de hoogste en de laagste temperatuur in °C, regen in mm, windkracht in Beaufort, en uren zon. `C.LIMITS` geeft voor elke soort de grens: de waarde waarbij het weerbericht de soort gaat noemen. `C.WORDS` geeft het woord voor elke soort in de taal van de les, `C.AND` het woord voor *en*, en `C.DAYS` de namen van de dagen:
    """)
    return


@app.cell
def example_cell():
    _day = C.WEEKS["april"][2]
    print(C.DAYS[L.lang][2], _day)
    print(C.LIMITS)
    print(_day["rain"] >= C.LIMITS["rain"], _day["wind"] >= C.LIMITS["wind"], _day["sun"] >= C.LIMITS["sun"])
    _words = [C.WORDS[L.lang]["rain"], C.WORDS[L.lang]["wind"]]
    print(_words)
    print(f"Op {C.DAYS[L.lang][2]} wordt het {_words[0]} {C.AND[L.lang]} {_words[1]}.")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Van de vijf metingen halen er twee de grens: 6,5 mm regen, bij een grens van 1 mm, en windkracht 6, bij een grens van 6: een waarde gelijk aan de grens telt mee. De hoogste temperatuur van 11 graden, de laagste van 7 en het ene uur zon vallen weg. De woorden komen uit een tabel, één per soort, en de laatste regel is een template: de f-string bevat de vaste woorden, en in de gaten komen de dag en de woorden. Reiter schrijft dat het eenvoudiger en goedkoper kan zijn om voor de realisatie templates met gaten te gebruiken in plaats van 'echte' syntactische verwerking, zeker als een systeem slechts een paar soorten zinnen maakt.

    Elke taal heeft een eigen template. De Engelse les schrijft *On Wednesday, it will be rainy and windy*: het Nederlands zet het werkwoord *wordt* op de tweede plaats, met *het* erna, waar het Engels eerst *it* schrijft en dan *will be*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_select():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. select: kiezen wat je zegt

        Schrijf `select`, dat `day` krijgt, een dictionary met de metingen van één dag, en `limits`, een dictionary van een soort meting naar de grens, zoals `C.LIMITS`. Geef een lijst met boodschappen terug: voor elke key van `limits`, in de volgorde van `limits`, een tuple van de soort en de waarde van de dag, als de waarde minstens de grens is. In de startcode begint `messages` als een lege lijst, en de loop neemt steeds één key `kind` van `limits`. Vervang de twee `...`.

        De check roept `select({"high": 11, "low": 7, "rain": 6.5, "wind": 6, "sun": 1}, {"rain": 1, "wind": 6})` aan, en dat geeft `[("rain", 6.5), ("wind", 6)]`: windkracht 6 haalt de grens 6. Dan `select({"high": 27, "low": 16, "rain": 0.0, "wind": 2, "sun": 13}, {"high": 25, "sun": 10, "rain": 1, "wind": 6})`, en eigen gevallen: een dag waarop niets de grens haalt, een waarde gelijk aan de grens, helemaal geen grenzen, grenzen in een andere volgorde, en dagen uit de weken van de course met `C.LIMITS`.
        """),
        L.task("Houd de metingen van een dag die de grens halen, als boodschappen van een soort en de waarde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_select():
    box_select = L.code_box("select", lines=6)
    box_select
    return (box_select,)


@app.cell(hide_code=True)
def check_select(box_select):
    result_select = L.check("select", box_select.value)
    result_select
    return (result_select,)


@app.cell(hide_code=True)
def hints_select():
    L.hints(
        "select",
        [
            "`day` en `limits` zijn dictionaries met dezelfde soorten als keys, zoals in *Dictionaries*: `limits[kind]` is de grens van één soort, en de waarde van de soort op de dag zoek je op dezelfde manier op in `day`. `messages.append` voegt één item toe aan het eind van de lijst, en dat item kan een tuple van twee waarden zijn, zoals in *Tuples, sets en sorteren*.",
            "De `if` zet `>=` tussen de waarde van de dag en de grens, zodat een waarde gelijk aan de grens meetelt, zoals windkracht 6 bij de grens 6. Daarna gaat de soort samen met de waarde in de lijst, de soort eerst: de volgende stap zoekt het woord bij de soort op, en de waarde zegt hoeveel.",
            "`            messages.append((kind, ___))`",
        ],
        why="`if day[kind] >= limits[kind]:` en `messages.append((kind, day[kind]))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_select():
    mo.md(r"""
    Met de grenzen van de course blijven van de 105 getallen van de drie weken er 24 over: 9 keer regen, 6 keer wind, 6 keer zon en 3 keer warmte. De laagste temperatuur van de nacht komt nooit in het weerbericht, en 3 dagen met een beetje regen onder 1 mm, zoals de 0,4 mm op vrijdag in april, tellen als droog. Content determination draait om kiezen, schrijven Gatt en Krahmer. De grenzen zijn zo'n keuze, en volgens Gatt en Krahmer kan de keuze ook afhangen van wie de tekst leest: een verwachting voor zeilers heeft de wind al bij een lagere kracht nodig dan een verwachting voor mensen in de stad.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_aggregate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. aggregate: één zin voor dagen met hetzelfde weer

        Schrijf `aggregate`, dat `phrases` krijgt, de woorden voor elke dag van een week, op volgorde. Geef een lijst met groepen terug, elk een tuple van een lijst met dagnummers, geteld vanaf 0, en de woorden van de groep. Dagen achter elkaar met dezelfde woorden delen een groep; een dag met andere woorden dan de vorige dag begint een nieuwe groep. In de startcode begint `groups` leeg, de loop gaat langs de plaatsen `i`, en de `if` vraagt of de laatste groep tot nu toe dezelfde woorden heeft als dag `i`: dan komt dag `i` erbij. Een tuple kan niet veranderen, maar de lijst erin wel: `groups[-1][0].append(i)` voegt dag `i` toe aan de dagen van de laatste groep. Vervang de twee `...`.

        De check roept `aggregate(["dry", "dry", "rainy", "rainy", "dry"])` aan, met de woorden van de Engelse les, en dat geeft `[([0, 1], "dry"), ([2, 3], "rainy"), ([4], "dry")]`: vrijdag is weer droog, maar ligt niet naast maandag en dinsdag. Dan `aggregate(["droog", "zonnig", "zonnig"])`, met de woorden van deze les, en eigen gevallen: één dag, zeven dagen met hetzelfde weer, zeven dagen met elk ander weer, woorden die later terugkomen, en de woorden van de weken van de course.
        """),
        L.task("Geef dagen achter elkaar met dezelfde woorden één groep, als één zin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_aggregate():
    box_aggregate = L.code_box("aggregate", lines=8)
    box_aggregate
    return (box_aggregate,)


@app.cell(hide_code=True)
def check_aggregate(box_aggregate):
    result_aggregate = L.check("aggregate", box_aggregate.value)
    result_aggregate
    return (result_aggregate,)


@app.cell(hide_code=True)
def hints_aggregate():
    L.hints(
        "aggregate",
        [
            "`groups[-1]` is de laatste groep tot nu toe: een tuple met op plaats 0 de lijst met dagen en op plaats 1 de woorden, zoals in *Tuples, sets en sorteren*. `len(groups) > 0` staat als eerste in de `if`, zodat de rest van de voorwaarde alleen loopt als er een groep is: een lege lijst heeft geen laatste item, en de eerste dag zou stoppen met een `IndexError`.",
            "Zet `==` tussen de woorden van de laatste groep en de woorden van dag `i`. In de `else` begint `groups.append` een nieuwe groep: een tuple van een lijst met alleen `i` en de woorden van dag `i`. Begin de dagen met een lijst `[i]`, niet met het getal `i`, zodat een latere dag erbij kan komen met `append`.",
            "`        if len(groups) > 0 and groups[-1][1] == ___:`",
        ],
        why="`if len(groups) > 0 and groups[-1][1] == phrases[i]:` en `groups.append(([i], phrases[i]))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_aggregate():
    mo.md(r"""
    Aggregatie maakt van de 21 dagen van de drie weken 15 zinnen: 5 in april, 6 in juli en 4 in oktober, waar dinsdag, woensdag en donderdag één zin delen. De feiten blijven hetzelfde; alleen de verdeling over de zinnen verandert. Volgens Reiter veranderen zulke bewerkingen niets aan de informatie in de tekst, maar maken de bewerkingen de tekst wel vloeiender en makkelijker te lezen. Binnen één dag delen regen en wind op dezelfde manier een zin, als *regenachtig en winderig*. Het eigen voorbeeld van Reiter maakt van twee zinnen over Sam, dat Sam een hoge bloeddruk heeft en dat Sam een lage bloedsuiker heeft, één zin: Sam heeft een hoge bloeddruk en een lage bloedsuiker.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_join():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. join_names: dagen op een rij, zoals de taal dat schrijft

        Schrijf `join_names`, dat `names` krijgt, een lijst met minstens één naam, en `and_word`, het woord voor *en* in de taal van de tekst. Geef één tekst terug: één naam onveranderd; bij meer namen de namen met een komma en een spatie ertussen, en `and_word` tussen de laatste twee. In de startcode regelt de `if` het geval van één naam, en de laatste `return` het geval van meer namen. Vervang de twee `...`.

        De check roept `join_names(["Monday", "Wednesday", "Thursday"], "and")` aan, en dat geeft `"Monday, Wednesday and Thursday"`. Dan `join_names(["zaterdag", "zondag"], "en")`, met de Nederlandse woorden, en dat geeft `"zaterdag en zondag"`, en eigen gevallen: één naam, de zeven dagen, en woorden die geen dagen zijn, zoals `["warm", "sunny", "windy"]`.
        """),
        L.task("Schrijf een lijst met namen zoals een zin dat doet: komma's, en het woord voor en bij de laatste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_join():
    box_join = L.code_box("join_names", lines=4)
    box_join
    return (box_join,)


@app.cell(hide_code=True)
def check_join(box_join):
    result_join = L.check("join_names", box_join.value)
    result_join
    return (result_join,)


@app.cell(hide_code=True)
def hints_join():
    L.hints(
        "join_names",
        [
            "`names[0]` is de eerste naam en `names[-1]` de laatste, zoals in *Lijsten*. `\", \".join` plakt een lijst met teksten aan elkaar met een komma en een spatie ertussen, zoals `\"\".join` dat zonder iets ertussen deed in *Tekst is een rij getallen*. Een slice geeft een stuk van een lijst; het stuk dat naar `join` gaat is elke naam behalve de laatste, en `and_word` komt tussen dat stuk en de laatste naam.",
            "Met elke naam in de `join` staat de laatste naam er twee keer: één keer in de `join` en één keer na `and_word`. Een negatief getal in een slice telt vanaf het eind, zoals in *Lijsten*. En `and_word` heeft aan beide kanten een spatie nodig, `\" \"`.",
            "`    return \", \".join(___) + \" \" + and_word + \" \" + names[-1]`",
        ],
        why="`return names[0]` bij één naam, en `return \", \".join(names[:-1]) + \" \" + and_word + \" \" + names[-1]` bij meer.",
    )
    return


@app.cell(hide_code=True)
def text_after_join():
    mo.md(r"""
    Bij surface realization gaan talen elk een eigen weg. Het Engels schrijft de dagen van de week met een hoofdletter, het Nederlands met kleine letters: *On Tuesday, Wednesday and Thursday* is *Op dinsdag, woensdag en donderdag*. Veel Engelse schrijvers zetten bij een opsomming van drie een komma tussen het voorlaatste woord en *and*, en Nederlandse schrijvers doen dat zelden; de course laat de komma in beide talen weg. Reiter somt op wat een realizer goed moet doen, zoals de vormen van woorden: het Engelse meervoud van *box* is *boxes*, niet *boxs*. En de congruentie: *I am here*, niet *I is here*. Een template met *het* als onderwerp heeft geen congruentie nodig: *wordt het* blijft hetzelfde bij één dag of bij drie.

    ### Probeer het: het weerbericht van een week

    Kies een week en stel de grenzen van content selection in, en zie de drie stappen meebewegen. Een getal met een achtergrond haalt de grens. Zet de grens voor regen op 5 mm in de week in oktober: maandag, met 4 mm regen, wordt *droog*. Een feit weglaten is niet neutraal, want het woord voor een dag zonder boodschappen zegt ook iets.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_week = mo.ui.dropdown(options={C.WEEK_NAMES[L.lang][_week]: _week for _week in C.WEEKS},
                              value=C.WEEK_NAMES[L.lang]["october"], label="Week")
    try_rain = mo.ui.slider(0.5, 20, step=0.5, value=1, label="Grens voor regen (mm)", show_value=True)
    try_wind = mo.ui.slider(1, 12, step=1, value=6, label="Grens voor wind (Beaufort)", show_value=True)
    try_high = mo.ui.slider(15, 35, step=1, value=25, label="Grens voor warmte (°C)", show_value=True)
    try_sun = mo.ui.slider(1, 15, step=1, value=10, label="Grens voor zon (uren)", show_value=True)
    mo.vstack([try_week, mo.hstack([try_rain, try_wind, try_high, try_sun], justify="start", gap=1.5, wrap=True)])
    return try_high, try_rain, try_sun, try_week, try_wind


@app.cell(hide_code=True)
def try_look(try_high, try_rain, try_sun, try_week, try_wind):
    mo.Html(C.try_picture(L.lang, try_week.value, try_rain.value, try_wind.value, try_high.value, try_sun.value))
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ### Oefening 4. Welke stap beslist het?

    Elke beslissing hieronder geeft vorm aan het weerbericht van een week. Zet elke beslissing bij de stap die de beslissing neemt.
    """)
    return


@app.cell(hide_code=True)
def exercise_steps():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl) noemde NLG, de kant van NLP die zelf tekst maakt; hier bouwde je er een in drie stappen, van een tabel naar een weerbericht. De stappen zijn regels, zoals in [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl): elk woord en elke grens is een keuze van iemand, en het weerbericht zegt alleen wat de regels erin zetten. Een language model, zoals in [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl), kan uit dezelfde tabel ook een weerbericht schrijven, vrijer en met meer verschillende woorden. Ziwei Ji en collega's waarschuwen in een overzicht uit 2023 dat generatie met deep learning gemakkelijk tekst verzint die niet de bedoeling was: tekst die niet steunt op wat erin ging. De volgende les, *Samenvatten: kiezen of schrijven*, komt dezelfde afweging tegen met een tekst als invoer in plaats van een tabel.

    **Een voorbeeld uit het dagelijks leven.** Een bank-app die schrijft *Je gaf deze maand meer uit in de supermarkt dan vorige maand*, zet dezelfde drie stappen: de app kiest één feit uit veel betalingen, kiest de woorden, en vult een template.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_select, result_aggregate, result_join, result_steps):
    L.wrap_up(result_select, result_aggregate, result_join, result_steps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
