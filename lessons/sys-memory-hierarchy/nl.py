# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-memory-hierarchy. Route: course/route/lessen/systemen.yaml. Checks, the H100, the measurements, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-memory-hierarchy", "nl")
    C = L.checks   # het geheugen van een H100, de optellingen gemeten op de computer van de course, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) en [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vector():
    quick_vector = L.choice(
        "quick-vector",
        question="`prices` is een NumPy-array met een miljoen getallen. Wat geeft `prices * 1.1`?",
        options={
            "right": "Een nieuwe array met een miljoen getallen, elk 1,1 keer de eigen prijs",
            "repeat": "De miljoen getallen herhaald, zoals `*` bij een lijst doet",
            "one": "Eén getal: 1,1 keer de som van alle prijzen",
            "error": "Een fout, want `*` werkt op één getal tegelijk en heeft een loop eromheen nodig",
        },
        correct="right",
        why={
            "right": "Een berekening op een array werkt op elk getal tegelijk, in één snelle loop binnen NumPy. In deze les zie je nog een reden dat zo'n loop snel is: de getallen van een array liggen in het geheugen naast elkaar.",
            "repeat": "Herhalen doet `*` met een lijst en een heel getal. Wat doet een berekening op een array met elk getal? Zie [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "one": "Hier wordt niets opgeteld. Wat gebeurt er met elk getal van de array? Zie [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
            "error": "NumPy doet de loop voor je, binnen de berekening. Wat geeft een berekening op een array? Zie [Rekenen zonder loops](/course/open/np-vectorization?lang=nl).",
        },
    )
    quick_vector
    return (quick_vector,)


@app.cell(hide_code=True)
def quick_check_fastest():
    quick_fastest = L.choice(
        "quick-fastest",
        question="Je meet dezelfde berekening vijf keer met `time.perf_counter()`: 3,1; 2,4; 2,5; 2,4 en 5,0 milliseconde. Welke tijd houdt de course aan?",
        options={
            "first": "3,1 milliseconde, de eerste keer",
            "right": "2,4 milliseconde, de snelste van de vijf",
            "mean": "3,08 milliseconde, het gemiddelde van alle vijf",
            "slow": "5,0 milliseconde, de traagste, zodat de tijd nooit te laag is",
        },
        correct="right",
        why={
            "right": "Het werk duurt minstens de snelste tijd; de tragere keren wachtten ook op andere dingen, en de eerste keer deed wat werk dat alleen de eerste keer nodig is. In deze les houd je de snelste van een paar optellingen aan om te meten hoe snel je computer getallen leest.",
            "first": "De eerste keer duurt vaak langer, omdat sommige dingen alleen de eerste keer gebeuren. Welke keer zegt het meest over het werk zelf? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
            "mean": "Het gemiddelde neemt de 5,0 mee, een keer die op iets anders wachtte. Welke tijd zegt het meest over het werk zelf? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
            "slow": "De traagste keer wachtte het langst op andere dingen. Welke tijd zegt het meest over het werk zelf? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
        },
    )
    quick_fastest
    return (quick_fastest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een GPU kan alleen rekenen met getallen die in de chip zelf staan, in een paar honderd kilobyte vlak naast de cores. Al het andere staat verder weg: in een groter geheugen op de chip, in het eigen geheugen van de GPU naast de chip, in het RAM van de computer, op de SSD. Elke stap verder weg bevat meer en levert de bytes trager. De snelheid van een berekening hangt dus niet alleen af van het aantal FLOPs, maar ook van de plek waar de getallen vandaan komen. In deze les zie je hoe het geheugen van een GPU en van je eigen computer in elkaar zit, reken je uit hoe lang het lezen van data van elke plek duurt, en meet je op je eigen computer hoeveel sneller een optelling is als de getallen dichtbij passen.

    **Een voorbeeld uit het dagelijks leven.** Een kok heeft zout en olie op het aanrecht, meel in de kast en zakken aardappelen in een voorraadkamer achterin. Het aanrecht is klein en binnen handbereik; de voorraadkamer bevat veel meer, en elke tocht ernaartoe kost minuten. Een goede kok haalt dus een hele zak in één keer, en houdt wat snel weer nodig is dichtbij.
    """)
    return


@app.cell(hide_code=True)
def predict_hbm():
    guess_hbm = L.guess(
        "hbm",
        question="Een NVIDIA H100 heeft 80 GB geheugen naast de chip, en de GPU leest dat geheugen met 3,35 terabyte per seconde. Hoe lang duurt het om alle 80 GB één keer te lezen, denk je?",
        options={
            "micro": "Ongeveer 24 microseconde",
            "right": "Ongeveer 24 milliseconde",
            "second": "Ongeveer 2,4 seconde",
            "minute": "Ongeveer een minuut, want 80 GB is veel data",
        },
        reveal="**Ongeveer 24 milliseconde.** 80 GB gedeeld door 3.350 GB per seconde is 0,024 seconde: de H100 kan het hele geheugen ongeveer 42 keer per seconde lezen. Dezelfde 80 GB uit het RAM van de computer, via de PCIe-verbinding met 64 GB per seconde, duren 1,25 seconde, 52 keer zo lang; van een snelle SSD met 7,45 GB per seconde ongeveer 11 seconden.",
    )
    guess_hbm
    return (guess_hbm,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Geheugenhiërarchie.** Een computer bewaart de getallen in lagen. Dicht bij de cores is het geheugen klein en snel; elke laag verder weg is groter en trager. Deze ladder is de **geheugenhiërarchie**. Op een NVIDIA H100 ziet de ladder er zo uit:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Een H100 heeft 132 groepen cores. Elke groep houdt de getallen voor de berekening van dit moment in de registers, 256 KB, en getallen die snel weer nodig zijn in de L1-cache, nog eens 256 KB. Alle groepen delen een L2-cache van 50 MB. Naast de chip zit 80 GB HBM3, en de data van de computer komt binnen via een PCIe Gen5-verbinding. Het eigen advies van NVIDIA voor snelle code is om de data "zo dicht mogelijk" bij de cores te zetten.

        **Bandbreedte.** De **bandbreedte** van een geheugen, of van een verbinding, is hoeveel bytes per seconde er doorheen kunnen. Lezen duurt het aantal bytes gedeeld door de bandbreedte: een bestand van 1 GB duurt 1 / 7,45 = 0,13 seconde van een SSD die 7,45 GB per seconde leest. Het HBM3 van de H100 levert 3,35 TB per seconde, de PCIe-verbinding 64 GB per seconde in elke richting, en één module DDR5-geheugen in een computer 38,4 GB per seconde. NVIDIA geeft geen los getal voor de registers en de caches, en de bandbreedte daar is nog hoger.

        **Cache.** Een **cache** is een klein, snel geheugen dat kopieën bewaart van data die net gebruikt is. De cache haalt een getal dat er nog niet in staat van verder weg, samen met de getallen ernaast: een blok van 64 bytes op de meeste processors, acht float64-getallen. De getallen van een NumPy-array liggen in het geheugen naast elkaar, dus vindt een optelling die in volgorde door de getallen loopt de meeste al in de cache. Getallen in een willekeurige volgorde hebben elk een eigen tocht nodig.

        **HBM.** **HBM** staat voor geheugen met een hoge bandbreedte: geheugenchips op elkaar gestapeld, vlak naast de GPU-chip, met duizenden korte draadjes verbonden. Daardoor is HBM veel sneller dan het RAM van een computer, dat op losse modules op het moederbord zit. De H100 heeft vijf stapels, samen 80 GB.

        **Je eigen computer.** De processor van je computer heeft dezelfde ladder: registers, twee of drie niveaus cache, dan het RAM, dan de SSD. Op de computer van de course leest een optelling van 512 KB aan getallen 34 GB per seconde, en een optelling van 512 MB slechts 8,2: de grote array past niet in de caches en komt uit het RAM. Heel kleine arrays zijn ook traag, om de reden uit [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl): elke aanroep heeft vaste kosten.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **geheugenhiërarchie** | | de lagen van geheugen, van klein en snel dicht bij de cores tot groot en traag verder weg: registers, caches, HBM of RAM, SSD | |
        | **bandbreedte** | | hoeveel bytes per seconde een geheugen of een verbinding levert; lezen duurt het aantal bytes gedeeld door de bandbreedte | `nbytes / bandwidth` |
        | **cache** | "kesj" | een klein, snel geheugen dat kopieën bewaart van data die net gebruikt is, samen met de data ernaast | |
        | **HBM** | "haa bee em" | geheugen met een hoge bandbreedte: stapels geheugenchips vlak naast een GPU-chip | |
        | `a.nbytes` | "a punt n bytes" | de bytes die een array inneemt: het aantal waarden keer de bytes van elke waarde | `np.zeros(1000).nbytes` geeft `8000` |
        """),
        L.watch_out("Een bandbreedte zegt hoe snel de bytes komen zodra de bytes stromen, niet hoe lang de eerste byte onderweg is. Veel kleine stukjes lezen, elk van een andere plek, is trager dan één groot stuk met dezelfde bytes."),
        L.watch_out("GB en TB tellen hier in duizendtallen: 1 TB is 1.000 GB, en 1 GB is 1.000.000.000 bytes, in Python `1e9`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De tijd om data te lezen, en de bandbreedte die je haalt:

    1. Tel de bytes: `a.nbytes` voor een array, of het aantal waarden keer de bytes van elke waarde.
    2. Zoek de bandbreedte op van de plek waar de bytes vandaan komen: het HBM, PCIe, het RAM of de SSD.
    3. Deel de bytes door de bandbreedte: de seconden om de bytes één keer te lezen.
    4. Om een bandbreedte zelf te meten: doe het werk eerst één keer, lees `time.perf_counter()` voor en na het werk, en deel de bytes door de seconden.
    5. Deel door `1e9` voor gigabytes per seconde, en vergelijk kleine arrays met grote: de snelheid zakt in bij de grootte die niet meer in de cache past.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een array met een miljoen float64-getallen: hoeveel bytes de array inneemt, en hoe lang het HBM3 van een H100 erover doet om de array één keer te lezen. Voer de cel uit.
    """)
    return


@app.cell
def example_bytes():
    import numpy as np

    _a = np.random.default_rng(0).random(1_000_000)   # een miljoen float64-getallen
    print(_a.size, "getallen,", _a.nbytes, "bytes")
    _seconds = _a.nbytes / 3.35e12                     # één keer lezen uit het HBM3 van een H100
    print(round(_seconds * 1e6, 2), "microseconde")
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Een float64-getal neemt 8 bytes in, dus nemen een miljoen getallen 8.000.000 bytes in. Uit het HBM is dat 2,39 microseconde; van een SSD met 7,45 GB per seconde zou het 1.074 microseconde duren, 450 keer zo lang. Oefening 1 rekent zulke tijden uit voor meerdere plekken tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Leestijden

        `read_seconds(nbytes, bandwidths)` krijgt een aantal bytes en een dict met de bandbreedte van elke plek, in bytes per seconde, en geeft een dict terug met de seconden die het lezen van de bytes van elke plek kost. Vervang de `...` zodat elke ronde van de loop `nbytes` gedeeld door de bandbreedte van `name` in `seconds[name]` zet.

        De check roept `read_seconds(16_000_000_000, C.LEVELS)` aan: een model van 16 GB, en `C.LEVELS` is `{"HBM": 3.35e12, "PCIe": 64e9, "SSD": 7.45e9}`. Daarna andere aantallen bytes en andere dicts, waarvan één leeg.
        """),
        L.task("Laat elke ronde van de loop de leestijd van `name` in `seconds` zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read", lines=6)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read",
        [
            "In de loop is `name` één key van `bandwidths`, zoals `\"HBM\"`, en een dict geeft de waarde van een key tussen vierkante haken. `seconds[name]` krijgt `nbytes` gedeeld door de bandbreedte van deze plek. Wat komt er na de `/`?",
            "Met 1.000 bytes en `{\"a\": 10}` duurt de plek `\"a\"` 1.000 / 10 = 100 seconden.",
            "`        seconds[name] = nbytes / ___`",
        ],
        why="`seconds[name] = nbytes / bandwidths[name]` deelt de bytes door de bandbreedte van elke plek. Een model van 16 GB duurt 4,8 ms uit het HBM, 0,25 seconde via PCIe en 2,1 seconde van de SSD.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een bandbreedte meten

        `gb_per_second(values)` telt een array eerst één keer op met `values.sum()`, zet de klok `time.perf_counter()` in `start`, telt nog een keer op, en bewaart de tijd van deze optelling in `seconds`. Vervang de `...` door de gigabytes per seconde die de optelling las: de bytes van `values` gedeeld door `seconds`, en door `1e9`.

        De check roept je functie aan met stand-ins voor een array: `slow_array(800000000, 0.1)` bevat 800.000.000 bytes, en elke optelling ervan duurt 0,1 seconde. Daarna roept het plaatje je functie aan op echte arrays van 8 KB tot 128 MB, op je eigen computer.
        """),
        L.task("Geef de gigabytes per seconde van één optelling van `values` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rate():
    box_rate = L.code_box("rate", lines=10)
    box_rate
    return (box_rate,)


@app.cell(hide_code=True)
def check_rate(box_rate):
    result_rate = L.check("rate", box_rate.value)
    result_rate
    return (result_rate,)


@app.cell(hide_code=True)
def hints_rate():
    L.hints(
        "rate",
        [
            "Een array weet hoeveel bytes de array inneemt: de tabel met nieuwe woorden noemt het attribuut. `seconds` is de tijd van één optelling, en een gigabyte is `1e9` bytes. Welke bytes komen er voor `/ seconds`?",
            "Een array met een miljoen float64-getallen neemt 8.000.000 bytes in. Gelezen in 0,001 seconde is dat 8.000.000 / 0,001 / 1e9 = 8 GB per seconde.",
            "`    return ___ / seconds / 1e9`",
        ],
        why="`return values.nbytes / seconds / 1e9` deelt de bytes van de array door de tijd van één optelling. De stand-in leest 800.000.000 bytes in 0,1 seconde: 8 GB per seconde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Oefening 3. Waar de getallen staan

    Acht plekken uit deze les. Deel elke plek in: *Op de GPU-chip*, *Naast de chip* of *Buiten de GPU*.
    """)
    return


@app.cell(hide_code=True)
def sort_where_cell():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "De registers en de caches horen bij de chip zelf. HBM is ernaast gestapeld. Buiten de GPU zit alles op het moederbord, en alles wat erop aansluit.",
            "Drie horen op de chip, twee ernaast en drie buiten de GPU.",
            "Op de GPU-chip: registers, L1-cache, \\_\\_\\_. Naast de chip: HBM3, \\_\\_\\_. Buiten de GPU: RAM, \\_\\_\\_, \\_\\_\\_.",
        ],
        why="**Op de GPU-chip:** registers, L1-cache, L2-cache. **Naast de chip:** HBM3, een model tijdens de training. **Buiten de GPU:** RAM, PCIe Gen5, SSD.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Optellingen van elke grootte

    Op de computer van de course de snelste van veel optellingen van float64-getallen met NumPy, voor arrays van 1 KB tot 512 MB: hoeveel gigabyte per seconde elke optelling leest. Kies een grootte.
    """)
    return


@app.cell(hide_code=True)
def demo_size():
    L.explore("size", "kilobyte", lambda _v: C.frame(_v, L.lang), start=512, values=C.DEMO_VALUES, role="",
              task="Ga van 1 naar 512.000 kilobyte: waar is de snelheid het hoogst, en waar zakt de snelheid in?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Van 1 KB tot 512 KB stijgt de snelheid, omdat de vaste kosten van een aanroep steeds minder wegen. Tot 32 MB komen de getallen uit de caches van de processor, met 26 tot 34 GB per seconde. Vanaf 128 MB komen de getallen uit het RAM, met ongeveer 8 GB per seconde: dezelfde optellingen, met een kwart van de snelheid. Een GPU heeft dezelfde ladder, met elke laag groter en sneller: het HBM3 alleen levert al 3,35 TB per seconde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ### Oefening 4. In volgorde of geschud

    Op de computer van de course bevat `a` 16 miljoen float64-getallen, 128 MB. `order` bevat de posities 0, 1, 2 enzovoort, en `shuffled` dezelfde posities in een willekeurige volgorde, gemaakt voordat de klok start. `a[order].sum()` duurt 67 milliseconde, en `a[shuffled].sum()` duurt 229 milliseconde.
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="Dezelfde getallen en dezelfde optellingen. Waarom duurt de geschudde volgorde meer dan drie keer zo lang?",
        options={
            "shuffle": "Het schudden kost de extra tijd",
            "right": "In volgorde wordt elk blok dat de cache ophaalt helemaal gebruikt; geschud blijft het meeste van elk blok ongebruikt",
            "bytes": "Geschudde getallen nemen meer bytes in het geheugen in, dus valt er meer te lezen",
            "adds": "Optellen in een willekeurige volgorde kost meer FLOPs dan optellen in volgorde, omdat elk getal een eigen stap nodig heeft",
        },
        correct="right",
        why={
            "right": "De cache haalt bij elk getal uit het RAM een heel blok met de getallen ernaast op. In volgorde staan de volgende getallen al in dat blok. Geschud heeft bijna elk getal een eigen blok uit het RAM nodig, en het meeste van elk blok wordt nooit gebruikt.",
            "shuffle": "`shuffled` was al gemaakt voordat de klok startte; beide regels kiezen alleen de getallen en tellen de getallen op. Het verschil zit in de plek van het volgende getal: staat het volgende getal dicht bij het vorige?",
            "bytes": "Beide regels lezen dezelfde 16 miljoen float64-getallen, van elk 8 bytes. Het verschil zit in de volgorde van lezen: wat haalt de cache bij elk getal op?",
            "adds": "Beide regels tellen dezelfde 16 miljoen getallen op, dus doen beide evenveel FLOPs. Het verschil zit in de volgorde van lezen: wat haalt de cache bij elk getal op?",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Rekenen zonder loops](/course/open/np-vectorization?lang=nl) werkte NumPy op een hele array tegelijk; hier zie je dat zo'n array in het geheugen op één rij ligt, precies wat een cache nodig heeft. In [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl) kostte een berekening de FLOPs gedeeld door de FLOP/s die je haalde; hier kost lezen de bytes gedeeld door de bandbreedte, en vaak is dat de langste van de twee tijden. De volgende les is *GPU-generaties: van Volta tot Blackwell*: hoe de GPU's van NVIDIA groeiden, in FLOP/s en in geheugen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_rate, result_read, result_where):
    L.wrap_up(result_read, result_rate, result_where, choice_order)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
