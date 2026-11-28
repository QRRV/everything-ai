# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-gpu-generations. Route: course/route/lessen/systemen.yaml. Checks, the five GPUs, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-gpu-generations", "nl")
    C = L.checks   # de gepubliceerde getallen van vijf GPU's, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Dichtbij en snel, ver weg en traag: de geheugenhiërarchie](/course/open/sys-memory-hierarchy?lang=nl) en [De wet van Moore en de groei van compute](/course/open/sys-moores-law?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bandwidth():
    quick_bandwidth = L.choice(
        "quick-bandwidth",
        question="Wat zegt de bandbreedte van een geheugen, zoals het HBM naast een GPU?",
        options={
            "hold": "Hoeveel bytes het geheugen kan bevatten",
            "right": "Hoeveel bytes per seconde het geheugen kan leveren",
            "first": "Hoe lang de eerste byte na een verzoek onderweg is",
            "flops": "Hoeveel FLOPs per seconde de GPU kan doen met de getallen in het geheugen",
        },
        correct="right",
        why={
            "right": "Lezen duurt het aantal bytes gedeeld door de bandbreedte. In deze les zie je hoe de bandbreedte van GPU-geheugen van generatie tot generatie groeide, naast de FLOP/s.",
            "hold": "Dat is de grootte van het geheugen. Wat telt de bandbreedte? Zie [Dichtbij en snel, ver weg en traag: de geheugenhiërarchie](/course/open/sys-memory-hierarchy?lang=nl).",
            "first": "Een bandbreedte zegt hoe snel de bytes komen zodra de bytes stromen, niet hoe lang de eerste byte onderweg is. Wat telt een bandbreedte? Zie [Dichtbij en snel, ver weg en traag: de geheugenhiërarchie](/course/open/sys-memory-hierarchy?lang=nl).",
            "flops": "FLOP/s tellen het rekenwerk van de cores, niet het geheugen. Wat telt de bandbreedte van een geheugen? Zie [Dichtbij en snel, ver weg en traag: de geheugenhiërarchie](/course/open/sys-memory-hierarchy?lang=nl).",
        },
    )
    quick_bandwidth
    return (quick_bandwidth,)


@app.cell(hide_code=True)
def quick_check_doubling():
    quick_doubling = L.choice(
        "quick-doubling",
        question="Een getal groeit in 6 jaar 8 keer, elk jaar in hetzelfde tempo. Wat is de verdubbelingstijd?",
        options={
            "right": "2 jaar, want 8 keer is drie verdubbelingen",
            "divide": "0,75 jaar: 6 jaar gedeeld door 8",
            "whole": "6 jaar, de hele periode",
            "half": "3 jaar, want na de helft van de 6 jaar is het getal verdubbeld",
        },
        correct="right",
        why={
            "right": "8 is 2 × 2 × 2: drie verdubbelingen in 6 jaar, één per 2 jaar. In deze les vergelijk je hoe snel de FLOP/s en de bandbreedte van GPU's verdubbelden.",
            "divide": "De jaren delen door de groei telt geen verdubbelingen. Hoeveel verdubbelingen maken 8 keer? Zie [De wet van Moore en de groei van compute](/course/open/sys-moores-law?lang=nl).",
            "whole": "In 6 jaar groeide het getal 8 keer, dus verdubbelde het getal meer dan één keer. Hoeveel verdubbelingen maken 8 keer? Zie [De wet van Moore en de groei van compute](/course/open/sys-moores-law?lang=nl).",
            "half": "Groei in hetzelfde tempo verdubbelt in gelijke stappen; na 3 jaar is het getal 2,8 keer zo groot. Hoeveel verdubbelingen maken 8 keer? Zie [De wet van Moore en de groei van compute](/course/open/sys-moores-law?lang=nl).",
        },
    )
    quick_doubling
    return (quick_doubling,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Ongeveer elke twee jaar brengt NVIDIA een nieuwe GPU voor servers uit, en AMD antwoordt met een eigen GPU. Elke nieuwe GPU rekent sneller, bevat meer geheugen en leest dat geheugen sneller dan de vorige. Toch groeien deze getallen niet even snel, en het getal dat achterblijft bepaalt hoe snel veel taken lopen. In deze les lees je de getallen van vijf GPU's van 2017 tot 2024 zoals de makers de getallen publiceerden, reken je uit wat elke stap won, en reken je uit hoeveel GPU's een model nodig heeft.

    **Een voorbeeld uit het dagelijks leven.** Een drukpers wordt elke twee jaar twee keer zo snel. De vrachtwagen met papier wordt intussen slechts een beetje sneller. Na een paar jaar staat de pers de helft van de tijd stil, wachtend op papier.
    """)
    return


@app.cell(hide_code=True)
def predict_bandwidth():
    guess_bandwidth = L.guess(
        "bandwidth",
        question="Van de V100 uit 2017 naar de B200 uit 2024 groeiden de FLOP/s van NVIDIA voor getallen van 16 bits 18 keer. Hoeveel keer groeide de bandbreedte van het eigen geheugen van de GPU, denk je?",
        options={
            "pace": "Ook ongeveer 18 keer, in hetzelfde tempo",
            "right": "Ongeveer 9 keer",
            "more": "Ongeveer 40 keer, want geheugen is eenvoudiger te maken dan cores",
            "none": "Nauwelijks: ongeveer 1,5 keer",
        },
        reveal="**Ongeveer 9 keer.** Van 0,9 TB/s naar 8 TB/s is 8,9 keer: de helft van de groei van de FLOP/s. In de tijd van het lezen van één byte doet een V100 139 FLOPs, een B200 281. Elke nieuwe GPU rekent sneller, en wacht langer op de getallen.",
    )
    guess_bandwidth
    return (guess_bandwidth,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **GPU-generatie.** NVIDIA ontwerpt de GPU's ongeveer elke twee jaar opnieuw, en vernoemt elk ontwerp naar een wetenschapper. De GPU's van één ontwerp, van één maker, vormen een **GPU-generatie**: Volta met de V100 in 2017, Ampere met de A100 in 2020, Hopper met de H100 in 2022 en Blackwell met de B200 in 2024. De MI300X van AMD uit 2023 hoort bij de generatie CDNA 3. De makers publiceren de getallen van elke GPU in een datasheet:

        | GPU | Generatie | Jaar | FLOP/s bij 16 bits | VRAM | Bandbreedte | Verbindingen |
        |---|---|---|---|---|---|---|
        | V100 | NVIDIA Volta | 2017 | 125 biljoen | 32 GB HBM2 | 0,9 TB/s | 300 GB/s |
        | A100 | NVIDIA Ampere | 2020 | 312 biljoen | 80 GB HBM2e | 2,04 TB/s | 600 GB/s |
        | H100 | NVIDIA Hopper | 2022 | 989 biljoen | 80 GB HBM3 | 3,35 TB/s | 900 GB/s |
        | MI300X | AMD CDNA 3 | 2023 | 1.307 biljoen | 192 GB HBM3 | 5,3 TB/s | 896 GB/s |
        | B200 | NVIDIA Blackwell | 2024 | 2.250 biljoen | 180 GB HBM3e | 8 TB/s | 1.800 GB/s |

        De verbindingen gaan van de ene GPU naar de andere GPU's in dezelfde server: NVLink bij NVIDIA, Infinity Fabric bij AMD. De B200 bestaat uit twee chips in één behuizing, zo verbonden dat de twee als één GPU werken.

        **VRAM.** Het geheugen van een GPU zelf heet **VRAM**, van video-RAM: de naam stamt uit de tijd dat GPU's alleen plaatjes tekenden. Op een server-GPU is het VRAM het HBM naast de chip. De parameters van een model moeten erin passen, of over meer GPU's verdeeld worden. Een byte is 8 bits, dus neemt een parameter van 16 bits 2 bytes in: een model met 70 miljard parameters neemt 140 GB in, te veel voor één H100 van 80 GB, en het model past op één MI300X of B200. Het draaien van het model vraagt extra ruimte, voor de getallen die onderweg berekend worden.

        **Minder bits.** Elke generatie voegde FLOP/s toe voor getallen met minder bits. De V100 rekent het snelst met getallen van 16 bits; de H100 doet met getallen van 8 bits twee keer zoveel FLOP/s als met 16, en de B200 doet 9.000 biljoen FLOP/s met getallen van 4 bits, 72 keer de V100 bij 16 bits. Datasheets drukken graag het getal voor de minste bits af, en vaak een getal dat twee keer zo groot is, met de vermelding "with sparsity": dat getal telt op matrices waarin de helft van de getallen nul is, in een vast patroon. Deze les vergelijkt elke GPU bij 16 bits, zonder sparsity. Een latere les laat zien wat minder bits kosten: *Hoeveel bits per getal: van FP32 tot INT4*.

        **Rekenen tegenover geheugen.** Deel de FLOP/s door de bandbreedte, en je krijgt de FLOPs die een GPU kan doen in de tijd van het lezen van één byte uit het VRAM. Een berekening met minder FLOPs per gelezen byte wacht op het geheugen. Het plaatje zet elke GPU tegenover de V100: de donkere stip is de FLOP/s, de open stip de bandbreedte.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Van generatie tot generatie groeit de afstand tussen de twee stippen: de memory wall uit [De wet van Moore en de groei van compute](/course/open/sys-moores-law?lang=nl), in de getallen van vijf GPU's.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **GPU-generatie** | | de GPU's van één ontwerp van één maker, zoals Hopper of Blackwell van NVIDIA; elke nieuwe generatie brengt meer FLOP/s, meer en sneller geheugen, en snellere verbindingen | |
        | **VRAM** | "vee-ram" | het geheugen van een GPU zelf; op een server-GPU het HBM naast de chip | |
        | FLOPs per byte | | de FLOP/s gedeeld door de bandbreedte: de FLOPs die een GPU kan doen in de tijd van het lezen van één byte | `flops / bandwidth` |
        """),
        L.watch_out("Een datasheet geeft het maximum: het meeste dat de GPU kan, op het werk dat het best past. Een echte taak haalt daar een deel van, vaak ruim onder de helft."),
        L.watch_out("Vergelijk gelijk met gelijk: dezelfde bits, en allebei met of allebei zonder sparsity. NVIDIA drukt voor de H100 bij 16 bits 1.979 biljoen FLOP/s af, met sparsity; zonder is het 989 biljoen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om twee GPU's te vergelijken, en om te zien of een model past:

    1. Neem van beide datasheets hetzelfde soort getal: dezelfde bits, allebei met of allebei zonder sparsity.
    2. Deel het nieuwe getal door het oude: hoeveel keer het groeide.
    3. Doe dat voor de FLOP/s en voor de bandbreedte. Groeien de FLOP/s sneller, dan wint een taak met veel lezen en weinig rekenen minder dan de FLOP/s beloven.
    4. Om te zien of een model past, tel je de bytes: de parameters keer de bytes van elke parameter. Deel door het VRAM van één GPU, en rond naar boven af.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De FLOPs per byte van vier GPU's van NVIDIA: de FLOP/s bij 16 bits gedeeld door de bandbreedte, allebei per seconde. Voer de cel uit.
    """)
    return


@app.cell
def example_per_byte():
    _gpus = {"V100": (125e12, 0.9e12), "A100": (312e12, 2.039e12),
             "H100": (989.4e12, 3.35e12), "B200": (2250e12, 8e12)}
    for _name in _gpus:
        _flops, _bandwidth = _gpus[_name]          # FLOP/s bij 16 bits, bytes per seconde
        print(_name, round(_flops / _bandwidth), "FLOPs per byte")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de tijd van het lezen van één byte doet een V100 139 FLOPs; een B200 doet er 281, ongeveer twee keer zoveel. De H100 doet de meeste: van de H100 naar de B200 groeide de bandbreedte iets sneller dan de FLOP/s bij 16 bits. Oefening 1 rekent de groei van elk getal in één keer uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gain():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat elke stap won

        `gains(old, new)` krijgt de getallen van twee GPU's, elk een dict zoals `C.GPUS["V100"]`, en geeft een dict terug met hoeveel keer elk getal groeide van `old` naar `new`. Vervang de `...` zodat elke ronde van de loop het getal van `new` gedeeld door het getal van `old` voor `key` in `out[key]` zet.

        De check roept `gains(C.GPUS["V100"], C.GPUS["B200"])` aan. `C.GPUS["V100"]` is `{"flops": 125, "bandwidth": 0.9, "vram": 32, "link": 300}`: biljoenen FLOP/s bij 16 bits, TB/s, GB en GB/s. Daarna andere GPU's en andere dicts.
        """),
        L.task("Laat elke ronde van de loop de groei van `key` in `out` zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gain():
    box_gain = L.code_box("gain", lines=6)
    box_gain
    return (box_gain,)


@app.cell(hide_code=True)
def check_gain(box_gain):
    result_gain = L.check("gain", box_gain.value)
    result_gain
    return (result_gain,)


@app.cell(hide_code=True)
def hints_gain():
    L.hints(
        "gain",
        [
            "In de loop is `key` één key van `old`, zoals `\"flops\"`, en `old[key]` is het oude getal. `out[key]` krijgt het getal van `new` bij dezelfde key, gedeeld door `old[key]`. Wat komt er voor de `/`?",
            "Met `{\"a\": 2}` als `old` en `{\"a\": 6}` als `new` groeide het getal van `\"a\"` 6 / 2 = 3 keer.",
            "`        out[key] = ___ / old[key]`",
        ],
        why="`out[key] = new[key] / old[key]` deelt het nieuwe getal door het oude, voor elke key. Van de V100 naar de B200: de FLOP/s ×18, de bandbreedte ×8,9, het VRAM ×5,6 en de verbindingen ×6.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Past het model?

        `gpus_needed(params, bits, vram_gb)` krijgt het aantal parameters van een model, de bits van elke parameter en het VRAM van één GPU in GB, en geeft terug hoeveel GPU's van dat formaat er nodig zijn om alle parameters te bevatten. `math.floor(gb / vram_gb)` zet de GPU's die helemaal vol raken in `count`, en de `if` telt er één bij voor de rest. Vervang de `...` zodat `gb` de gigabytes van alle parameters is: `params` keer de bytes van één parameter, gedeeld door `1e9`.

        De check roept `gpus_needed(70_000_000_000, 16, 80)` aan: 70 miljard parameters van 16 bits, op H100's van 80 GB. Daarna andere modellen en andere GPU's; het plaatje toont twee modellen op alle vijf de GPU's.
        """),
        L.task("Maak van `gb` de gigabytes van alle parameters van het model."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit", lines=10)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit",
        [
            "`gb` is `params` keer de bytes van één parameter, gedeeld door `1e9`. Een parameter heeft `bits` bits, en een byte is 8 bits: hoeveel bytes is dat?",
            "Een parameter van 16 bits neemt 2 bytes in, dus nemen 70 miljard parameters 140.000.000.000 bytes in: 140 GB.",
            "`    gb = params * ___ / 1e9`",
        ],
        why="`gb = params * bits / 8 / 1e9` zet de bits van elke parameter om in bytes, en de bytes in gigabytes. 70 miljard parameters van 16 bits zijn 140 GB: twee H100's van 80 GB, of één MI300X of B200.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_helps():
    mo.md(r"""
    ### Oefening 3. Wat helpt welke taak

    Acht taken voor een GPU. Zet elke taak onder het getal dat de taak sneller of mogelijk maakt: *Meer FLOP/s*, *Meer bandbreedte*, *Meer VRAM* of *Snellere verbindingen*.
    """)
    return


@app.cell(hide_code=True)
def sort_helps_cell():
    sort_helps = L.sort("helps")
    sort_helps
    return (sort_helps,)


@app.cell(hide_code=True)
def check_helps(sort_helps):
    result_helps = L.check("helps", sort_helps)
    result_helps
    return (result_helps,)


@app.cell(hide_code=True)
def hints_helps():
    L.hints(
        "helps",
        [
            "Vraag bij elke taak waarop de GPU wacht: op de cores, als elk gelezen getal vaak gebruikt wordt; op het geheugen, als elk getal één keer gebruikt wordt; op ruimte voor alle getallen; of op de andere GPU's.",
            "Onder elk van de vier horen twee taken.",
            "Meer FLOP/s: trainen met grote batches, \\_\\_\\_. Meer bandbreedte: één gebruiker, één token per keer, \\_\\_\\_. Meer VRAM: een model van 140 GB op één GPU, \\_\\_\\_. Snellere verbindingen: trainen op 1.000 GPU's, \\_\\_\\_.",
        ],
        why="**Meer FLOP/s:** trainen met grote batches, 256 gebruikers tegelijk. **Meer bandbreedte:** één gebruiker, één token per keer, de optelling van een enorme array. **Meer VRAM:** een model van 140 GB op één GPU, een groter model trainen op één GPU. **Snellere verbindingen:** trainen op 1.000 GPU's, een model verdeeld over acht GPU's.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Vijf GPU's

    Elke GPU tegenover de V100 uit 2017, op een logaritmische schaal: de FLOP/s bij 16 bits, de bandbreedte van het VRAM, het VRAM en de verbindingen. Kies een jaar.
    """)
    return


@app.cell(hide_code=True)
def demo_year():
    L.explore("year", "jaar", lambda _v: C.frame(_v, L.lang), start=2022, values=C.DEMO_VALUES, role="",
              task="Ga van 2017 naar 2024: welk getal groeit het snelst, en welk getal blijft achter?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Van de V100 naar de B200 verdubbelden de FLOP/s bij 16 bits ongeveer elke 1,7 jaar, de bandbreedte elke 2,2 jaar en het VRAM elke 2,8 jaar; de verbindingen groeiden 6 keer in de zeven jaar. Elke nieuwe GPU wacht dus langer op de getallen, tenzij de taak elk gelezen getal vaak gebruikt. De MI300X laat zien dat een maker kan kiezen: minder FLOP/s dan de B200, en meer VRAM dan elk van de vijf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_token():
    mo.md(r"""
    ### Oefening 4. Eén token per keer

    Een model van 16 GB schrijft een antwoord voor één gebruiker, één token per keer. Voor elk token leest de GPU alle 16 GB uit het VRAM, en doet ongeveer één FLOP per byte. Op een V100 duurt het lezen van de 16 GB 17,8 milliseconde, en de FLOPs duren 0,13 milliseconde.
    """)
    return


@app.cell(hide_code=True)
def choice_token_cell():
    choice_token = L.choice(
        "token",
        question="Van de V100 naar de B200 groeiden de FLOP/s 18 keer en de bandbreedte 8,9 keer. Hoeveel sneller schrijft de B200 dit antwoord, ongeveer?",
        options={
            "flops": "Ongeveer 18 keer, net als de FLOP/s",
            "right": "Ongeveer 9 keer, net als de bandbreedte: het lezen van de 16 GB kost de meeste tijd",
            "both": "Ongeveer 160 keer: de groei van de FLOP/s en van de bandbreedte vermenigvuldigd",
            "vram": "Helemaal niet sneller: de 16 GB passen in het VRAM van allebei, en het VRAM bepaalt hoe snel een antwoord komt",
        },
        correct="right",
        why={
            "right": "Elk token wacht tot de 16 GB gelezen zijn: 17,8 milliseconde op een V100, 2 op een B200, 8,9 keer zo snel. De FLOPs kosten op allebei een fractie van een milliseconde, dus is de groei van 18 keer nauwelijks te zien.",
            "flops": "De FLOPs kosten 0,13 van de 17,8 milliseconde op een V100. Wat kost de rest van de tijd, en hoeveel groeide dat?",
            "both": "De twee tellen niet op tot één groei: elk token wacht vooral op een van de twee. Welke van de twee kost het grootste deel van de 17,8 milliseconde?",
            "vram": "Het VRAM zegt of het model past, niet hoe snel het model gelezen wordt. Wat bepaalt de tijd van het lezen van 16 GB?",
        },
    )
    choice_token
    return (choice_token,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Dichtbij en snel, ver weg en traag: de geheugenhiërarchie](/course/open/sys-memory-hierarchy?lang=nl) duurde lezen het aantal bytes gedeeld door de bandbreedte; hier zie je dat de bandbreedte van GPU's trager groeit dan de FLOP/s, dus wachten steeds meer taken op de bytes. In [De wet van Moore en de groei van compute](/course/open/sys-moores-law?lang=nl) was de memory wall een trend over 20 jaar; hier staat de memory wall in de getallen van vijf GPU's, van 139 FLOPs per byte voor de V100 tot 281 voor de B200. De volgende les is *Hoe een computer een getal bewaart: integers en floats*: wat een getal van 16, 8 of 4 bits kan bevatten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_token, result_fit, result_gain, result_helps):
    L.wrap_up(result_gain, result_fit, result_helps, choice_token)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
