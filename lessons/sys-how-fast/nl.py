# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-how-fast. Route: course/route/lessen/systemen.yaml. Checks, de bigram-schrijver van de course, de metingen en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-how-fast", "nl")
    C = L.checks   # de bigram-schrijver van de course, de metingen en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl) en [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="Een netwerk heeft 50.000 weights. Ongeveer hoeveel FLOPs kost één trainingsstap voor één voorbeeld?",
        options={
            "forward": "Ongeveer 100.000: 2 per weight, zoals in de forward pass",
            "right": "Ongeveer 300.000: 6 per weight",
            "one": "Ongeveer 50.000: 1 per weight",
            "pairs": "Ongeveer 2,5 miljard: elke weight met elke andere",
        },
        correct="right",
        why={
            "right": "De forward pass kost 2 FLOPs per weight, en de backward pass gebruikt elke weight nog twee keer: 6 · 50.000 = 300.000. Deze les maakt van FLOPs tijd: hoeveel FLOPs een computer in één seconde doet.",
            "forward": "2 FLOPs per weight is alleen de forward pass. Wat komt er met de backward pass bij? Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "one": "Alleen al in de forward pass kost elke weight een vermenigvuldiging en een optelling, en de backward pass gebruikt de weight opnieuw. Hoeveel FLOPs per weight geeft dat? Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "pairs": "Een weight wordt in de forward pass één keer per voorbeeld gebruikt, niet één keer voor elke andere weight. Hoeveel FLOPs per weight kost een trainingsstap? Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`W` is de 65 × 65-tabel van je bigram-netwerk, en `x` bevat de nummers van 1.000 tekens. Wat is `W[x]`?",
        options={
            "one": "Eén rij van 65 getallen: de rij van het laatste teken",
            "table": "Een 65 × 65-tensor: de tabel zelf",
            "right": "Een 1.000 × 65-tensor: de rij van elk teken",
            "pairs": "Een 1.000 × 1.000-tensor met alle paren",
        },
        correct="right",
        why={
            "right": "`W[x]` kiest de rij van `W` voor elk nummer in `x`: hetzelfde als `torch.eye(65)[x] @ W`, een one-hot-rij per teken keer de tabel. In deze les is dat product wat het netwerk uitrekent voor elk teken dat het schrijft.",
            "one": "`x` bevat 1.000 nummers, en `W[x]` kiest voor elk nummer een rij. Hoeveel rijen geeft dat? Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
            "table": "Indexeren met `x` kiest rijen van `W`: één voor elk nummer in `x`. Hoeveel nummers bevat `x`? Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
            "pairs": "`W[x]` kiest rijen van `W`, en elke rij heeft 65 getallen, één voor elk volgend teken. Hoeveel getallen heeft elke rij van de uitkomst? Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Aan een model dat goede antwoorden langzaam geeft, heb je weinig: niemand wacht een minuut op elk woord van een antwoord, en een training van een jaar is nooit op tijd klaar. In [Te veel hellingen om één voor één te meten](/course/open/nn-why-backprop?lang=nl) telde je het werk van een berekening in FLOPs, en dat aantal is op elke computer hetzelfde. Hoe lang het werk duurt, is dat niet: dat hangt af van hoeveel FLOPs de computer per seconde doet, en van hoeveel tijd opgaat aan werk dat niets met FLOPs te maken heeft. Deze les meet allebei op je eigen computer, voor een matrixproduct en voor het bigram-netwerk van je project: hoeveel FLOPs per seconde, hoeveel tekens per seconde, en waar de tijd van één teken heen gaat.

    **Een voorbeeld uit het dagelijks leven.** Een rit van 30 km is over de snelweg even ver als door de stad. Over de snelweg, met 100 km/u, duurt de rit 18 minuten. Door de stad, met stoplichten, kruispunten en parkeren, kan dezelfde 30 km een uur duren, in een auto die makkelijk de hele weg 100 km/u kan rijden. De afstand is het werk, de snelheid is het tempo, en de tijd is het werk gedeeld door het tempo dat je echt haalt.
    """)
    return


@app.cell(hide_code=True)
def predict_speed():
    guess_speed = L.guess(
        "speed",
        question="Eén teken van je bigram-netwerk kost een one-hot-rij keer de 65 × 65-tabel: 2 · 65 · 65 = 8.450 FLOPs. De computer van de course doet bij een groot matrixproduct ongeveer 450 miljard FLOPs per seconde. Hoeveel tekens per seconde schrijft het netwerk op deze computer, denk je?",
        options={
            "flops": "Ongeveer 50 miljoen: 450 miljard gedeeld door 8.450",
            "billions": "Miljarden: een product is wat een computer het snelst doet",
            "right": "Tienduizenden",
            "few": "Een paar honderd: Python is traag",
        },
        reveal="**Tienduizenden.** Op de computer van de course kost één teken 20,5 microseconde: ongeveer 49.000 tekens per seconde, ongeveer duizend keer minder dan de 53 miljoen die de FLOPs alleen zouden toelaten. Het product zelf neemt 1,9 van deze microseconden, en het trekken van het volgende teken 14,5. Een product zo klein als één rij keer een 65 × 65-tabel haalt slechts een paar miljard FLOPs per seconde, en de meeste tijd gaat op aan werk dat niets met FLOPs te maken heeft.",
    )
    guess_speed
    return (guess_speed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Werk en tempo.** FLOPs tellen het werk van een berekening. Een product van twee n × n-matrices geeft n · n getallen, en elk daarvan kost n vermenigvuldigingen en n optellingen, dus kost het product $2n^3$ FLOPs. **FLOP/s**, FLOPs per seconde, is het tempo: hoeveel FLOPs een computer in één seconde doet. De tijd van het werk is het werk gedeeld door het tempo. Let op de schrijfwijze: FLOPs, met een kleine s, is een aantal bewerkingen; FLOP/s, ook geschreven als FLOPS, is een snelheid.

        **FLOP/s meten.** Om een tempo te meten, neem je de tijd op van werk waarvan je de FLOPs kent. `time.perf_counter()`, uit de module `time` van Python, leest een klok in seconden af, een klok die gemaakt is om korte tijden te meten: lees de klok af voor en na het werk, en het verschil is de tijd van het werk. Eén meting kan pech hebben, omdat de computer af en toe ook iets anders doet, dus meet je een paar keer en houd je de snelste meting: de meting met de minste onderbrekingen. De eerste keer is een berekening trager, omdat PyTorch de berekening dan voorbereidt, dus telt deze keer niet mee. Korte tijden lees je het makkelijkst in **microseconden**, µs, miljoenste delen van een seconde: één seconde is $10^6$ µs.

        **Klein is traag.** De afbeelding laat de FLOP/s zien van een n × n-product op de computer van de course. Bij 8 × 8 kost een product 1.024 FLOPs en duurt het 1,1 µs: ongeveer 0,9 miljard FLOP/s. Bij 1.024 × 1.024 kost het 2,1 miljard FLOPs en duurt het 4.800 µs: ongeveer 450 miljard FLOP/s, 480 keer zo snel. Elk product heeft vaste kosten bovenop de FLOPs: Python roept PyTorch aan, PyTorch controleert de shapes, kiest een manier van rekenen en maakt ruimte voor de uitkomst. Op de computer van de course duurt dat ongeveer een microseconde, hoe groot het product ook is. Bij een klein product zijn de vaste kosten bijna alle tijd; bij een groot product vallen de vaste kosten weg naast de FLOPs, en kunnen alle cores bezig blijven.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Tokens per seconde.** Voor een model dat schrijft, telt **tokens per seconde**: hoeveel tokens het model in één seconde schrijft. Voor je bigram-netwerk is een token één teken, en elk token kost één stap: de one-hot-rij van het laatste teken keer de tabel, softmax, en het trekken van het volgende teken, zoals in [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl). Alleen het product doet echt werk met de tabel; de andere delen hebben vooral vaste kosten.

        **Throughput.** Eén stap kan ook veel teksten tegelijk elk een volgend teken geven: zet de laatste tekens van alle teksten in `x`, en `torch.eye(65)[x] @ W` geeft alle rijen in één product. **Throughput** is het werk dat alles samen per seconde doet: met 100 teksten tegelijk de tokens per seconde van alle 100 teksten. De vaste kosten van elk deel betaal je dan één keer per stap, en niet één keer per tekst.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **FLOP/s** | flop per seconde | het aantal rekenbewerkingen met floating-point getallen dat een computer in één seconde doet: hoe snel de computer rekent; ook geschreven als FLOPS | `flops / seconds` |
        | **tokens per seconde** | | hoeveel tokens een model in één seconde schrijft | `n / seconds` |
        | **throughput** | | het werk dat alles samen per seconde doet, zoals de tokens per seconde van alle teksten tegelijk | |
        | `time.perf_counter()` | time punt perf counter | een klok in seconden, gemaakt om korte tijden te meten: alleen het verschil tussen twee standen betekent iets | `time.perf_counter()` |
        | **microseconde**, µs | | een miljoenste deel van een seconde | `seconds * 10 ** 6` |
        """),
        L.watch_out("De FLOPs van een model zeggen hoeveel het model rekent, niet hoe lang het duurt. Twee modellen met evenveel FLOPs kunnen vele keren verschillen in snelheid, als het ene een paar grote producten doet en het andere veel kleine."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Meten hoe snel iets draait:

    1. Tel het werk: FLOPs voor een berekening, tokens voor een model dat schrijft.
    2. Draai het eerst één keer, zodat PyTorch alles heeft voorbereid.
    3. Lees `time.perf_counter()` af voor en na, en neem het verschil: de seconden.
    4. Herhaal een paar keer, en houd de snelste.
    5. Deel het werk door de seconden: FLOP/s, of tokens per seconde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt vijf keer de tijd op van een product van twee 512 × 512-matrices, houdt de snelste, en rekent de FLOP/s uit. `torch.ones((size, size))` maakt een matrix vol enen: welke getallen er in een matrix staan, verandert niets aan hoe lang een product duurt. Voer de cel uit.
    """)
    return


@app.cell
def example_measure():
    import time

    import torch

    size = 512
    a = torch.ones((size, size))
    b = torch.ones((size, size))
    flops = 2 * size ** 3
    a @ b   # het eerste product van een grootte bereidt het product ook voor
    fastest = None
    for _ in range(5):
        before = time.perf_counter()
        a @ b
        seconds = time.perf_counter() - before
        if fastest is None or seconds < fastest:
            fastest = seconds
    print("FLOPs:", flops)
    print("snelste tijd in µs:", round(fastest * 10 ** 6, 1))
    print("miljarden FLOP/s:", round(flops / fastest / 10 ** 9, 1))
    return a, b, before, fastest, flops, seconds, size, time, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het product kost 268.435.456 FLOPs. Op de computer van de course duurt de snelste van de vijf ongeveer 650 µs, en dat geeft ongeveer 413 miljard FLOP/s. Je eigen computer geeft andere getallen: minder cores geven minder FLOP/s, een nieuwere processor meer. Voer de cel nog eens uit: de tijd verandert elke keer een beetje; de snelste houden geeft een getal dat je kunt vergelijken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flop():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. FLOP/s uit gemeten tijden

        `flop_per_second(sizes, micros)` krijgt een lijst met groottes van matrices en een lijst met tijden: `micros[i]` is de tijd in microseconden van één product van twee `sizes[i]` × `sizes[i]`-matrices. De functie geeft de lijst `rates` terug, met de FLOP/s van elke grootte. De loop `for i in range(len(sizes))` gaat langs de groottes; `n` is de grootte, en `seconds` de tijd in seconden. Vervang de `...` door een regel die de FLOP/s van deze grootte aan `rates` toevoegt.

        De check roept `flop_per_second` aan met de groottes `[16, 512]` en de tijden `[1.3, 650.0]` van de computer van de course. Daarna vier andere gevallen: alle acht groottes van de afbeelding met de tijden erbij, een 2 × 2-product van 1 µs, een 1.000 × 1.000-product van 2.000 µs, en twee lege lijsten.
        """),
        L.task("Voeg de FLOP/s van elke grootte toe aan `rates`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flop():
    box_flop = L.code_box("flop-rate", lines=8)
    box_flop
    return (box_flop,)


@app.cell(hide_code=True)
def check_flop(box_flop):
    result_flop = L.check("flop-rate", box_flop.value)
    result_flop
    return (result_flop,)


@app.cell(hide_code=True)
def hints_flop():
    L.hints(
        "flop-rate",
        [
            "Stap 5 van het recept: de FLOP/s van een grootte is het aantal FLOPs gedeeld door de seconden. De regel roept `rates.append` aan met de FLOPs van een product van twee `n` × `n`-matrices, gedeeld door `seconds`.",
            "Een product van twee n × n-matrices geeft n · n getallen, en elk daarvan kost n vermenigvuldigingen en n optellingen. Voor n = 16 zijn dat 8.192 FLOPs, en met 1,3 µs, 0,0000013 seconde, ongeveer 6,3 miljard FLOP/s.",
            "`        rates.append(___ / seconds)`",
        ],
        why="`rates.append(2 * n ** 3 / seconds)` deelt het werk van één product door de tijd. `n ** 3` telt de n · n getallen van elk n stappen, en de 2 de vermenigvuldiging en de optelling van elke stap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_tokens():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tokens per seconde

        `tokens_per_second(write, n)` krijgt een functie `write` die `n` tokens schrijft, en meet hoeveel tokens per seconde `write` schrijft. De eerste regel, `import time`, maakt de klok beschikbaar, en de `return`-regel deelt `n` door `seconds`. Vervang de `...` door de klok om `write(n)`: lees `time.perf_counter()` af voor `write(n)`, roep `write(n)` aan, en maak `seconds` de tijd die verstreek.

        De check roept `tokens_per_second` aan met een stand-in voor een schrijver: `slow_writer(0.05)` doet er 0,05 seconde over, wat `n` ook is, dus 1.000 tokens geven 20.000 tokens per seconde. Een klok geeft nooit twee keer precies dezelfde tijd, dus keurt de check een antwoord goed dat minder dan 30% daarvan afwijkt. Daarna twee andere gevallen: 500 tokens in 0,1 seconde, en 30 tokens in 0,05 seconde.
        """),
        L.task("Meet één aanroep van `write(n)`, en maak `seconds` de tijd daarvan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tokens():
    box_tokens = L.code_box("tokens-rate", lines=8)
    box_tokens
    return (box_tokens,)


@app.cell(hide_code=True)
def check_tokens(box_tokens):
    result_tokens = L.check("tokens-rate", box_tokens.value)
    result_tokens
    return (result_tokens,)


@app.cell(hide_code=True)
def hints_tokens():
    L.hints(
        "tokens-rate",
        [
            "Stap 3 van het recept, in drie regels: een stand van `time.perf_counter()` met een eigen naam, dan `write(n)`, dan `seconds`: een nieuwe stand van `time.perf_counter()` min de eerste.",
            "In het uitgewerkte voorbeeld heette de eerste stand `before`. Met `slow_writer(0.05)` liggen de twee standen ongeveer 0,05 seconde uit elkaar, dus geven 1.000 tokens ongeveer 20.000 tokens per seconde.",
            "`    seconds = time.perf_counter() - ___`",
        ],
        why="`start = time.perf_counter()` voor `write(n)`, en `seconds = time.perf_counter() - start` erna: de klok loopt om precies één aanroep van `write(n)`, en verder niets.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Waar de tijd van een teken heen gaat

    De demo schrijft 2.000 tekens op je computer met de bigram-tabel van de course, geteld op de toneelstukken van Shakespeare: de tabel die je netwerk in het project leerde. De demo meet elk deel van elke stap: de one-hot-rij, het product met de tabel, softmax, en het trekken van het volgende teken. Daarna schrijft de demo 1, 10, 100 en 1.000 teksten tegelijk, elk 200 stappen, en meet hoe lang één stap duurt. Druk op de knop; het meten duurt een paar seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    measure_button = mo.ui.run_button(label="Meet op mijn computer")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(measure_button):
    if not measure_button.value:
        _out = mo.md("*Druk op de knop om op je computer te meten.*")
    else:
        _m = C.demo()
        _out = mo.vstack([
            mo.Html(C.parts_picture(_m["parts"], L.lang)),
            mo.md(C.parts_say(_m["parts"], L.lang)),
            L.explore("texts", "teksten tegelijk", lambda _v: C.texts_frame(_m["texts"], int(_v), L.lang), start=1,
                      values=list(C.TEXTS), role="",
                      task="Ga van 1 tekst naar 1.000: hoeveel keer zoveel tekens per seconde geven 1.000 teksten tegelijk op je computer?"),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Op de computer van de course kost één teken 20,5 µs: ongeveer 49.000 tekens per seconde. Het product neemt daarvan 1,9 µs, minder dan een tiende, en het trekken 14,5 µs, meer dan twee derde; de one-hot-rij kost 2,5 µs en softmax 1,3 µs, en de laatste 0,3 µs gaan naar de loop om de delen heen en naar het aflezen van de klok. Een stap met 100 teksten duurt 180 µs, ongeveer 9 keer zo lang als een stap met één tekst, en geeft 100 tekens: ongeveer 560.000 tekens per seconde, 11 keer de throughput van één tekst. Met 1.000 teksten duurt een stap 1.300 µs, en is de throughput ongeveer 770.000 tekens per seconde, 16 keer zoveel als met één tekst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_part_faster():
    mo.md(r"""
    ### Oefening 3. Een sneller product

    Iemand vindt een manier om het product van het bigram-netwerk tien keer zo snel te maken, en verandert verder niets.
    """)
    return


@app.cell(hide_code=True)
def choice_part_faster_cell():
    choice_part_faster = L.choice(
        "part-faster",
        question="Hoeveel minder tijd kost één teken op de computer van de course?",
        options={
            "ten": "Tien keer minder: 2,1 µs in plaats van 20,5",
            "half": "Ongeveer de helft: het product is het grote werk",
            "none": "Niets: FLOPs kosten geen tijd",
            "right": "Ongeveer 8% minder: 1,7 van de 20,5 µs",
        },
        correct="right",
        why={
            "right": "Het product neemt 1,9 van de 20,5 µs, en tien keer zo snel 0,19: 1,7 µs minder, ongeveer 8%. Versnel je het deel dat de FLOPs doet, dan helpt dat weinig als de meeste tijd ergens anders heen gaat: het trekken alleen al kost 14,5 µs.",
            "ten": "Alleen het product wordt tien keer zo snel, en het product neemt 1,9 van de 20,5 µs. Wat gebeurt er met de andere 18,6 µs?",
            "half": "Het product doet bijna alle FLOPs, maar hoeveel van de tijd neemt het product op de computer van de course?",
            "none": "Het product kost wel tijd: 1,9 µs op de computer van de course. Hoeveel daarvan verdwijnt als het product tien keer zo snel is?",
        },
    )
    choice_part_faster
    return (choice_part_faster,)


@app.cell(hide_code=True)
def text_exercise_many_texts():
    mo.md(r"""
    ### Oefening 4. Veel mensen tegelijk

    Een dienst schrijft met één model teksten voor veel mensen. In plaats van een stap per persoon geeft de dienst in één stap 100 mensen elk het volgende token.
    """)
    return


@app.cell(hide_code=True)
def choice_many_texts_cell():
    choice_many_texts = L.choice(
        "many-texts",
        question="Wat gebeurt er met een stap, vergeleken met een stap voor één persoon?",
        options={
            "right": "De stap duurt langer, maar veel minder dan 100 keer zo lang",
            "same": "De stap duurt 100 keer zo lang, dus blijft de throughput gelijk",
            "shared": "De stap duurt even lang als voor één persoon: het werk wordt gedeeld",
            "fewer": "De stap duurt zo lang dat er minder tokens per seconde uitkomen",
        },
        correct="right",
        why={
            "right": "Op de computer van de course duurt een stap met 100 teksten 180 µs in plaats van 20,5: ongeveer 9 keer zo lang, voor 100 keer zoveel tekens. De vaste kosten van elk deel betaal je één keer per stap, dus stijgt de throughput 11 keer. Zo houden diensten die voor veel mensen tegelijk schrijven de computers bezig.",
            "same": "De 100 one-hot-rijen gaan samen door één product, één softmax en één keer trekken. Betaal je de vaste kosten van elk deel dan één keer, of 100 keer?",
            "shared": "Een stap met 100 teksten doet 100 keer zoveel FLOPs als een stap met één tekst. Kan de stap dan helemaal niet langer duren?",
            "fewer": "De teksten delen elke stap: één product voor de rijen van alle teksten. Gingen de tekens per seconde in de demo omlaag of omhoog met meer teksten?",
        },
    )
    choice_many_texts
    return (choice_many_texts,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Te veel hellingen om één voor één te meten](/course/open/nn-why-backprop?lang=nl) telde je FLOPs, en in [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl) loonde een GPU alleen voor grote producten. Deze les zette een klok naast het tellen: FLOP/s voor een matrixproduct, tokens per seconde voor het bigram-netwerk van [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl), en de throughput van veel teksten tegelijk. Het domein *ML-systemen, hardware en MLOps* gaat veel verder: de chips waarop modellen trainen, veel mensen tegelijk bedienen, en wat het trainen en draaien van een model kost. Met *Toon verdieping* aan komt nu *Spam of niet: naive Bayes*, waarin getelde woorden berichten in spam en ok indelen. In de volgende les, *Wat een paper, een preprint en peer review zijn*, ga je zelf onderzoek lezen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_many_texts, choice_part_faster, result_flop, result_tokens):
    L.wrap_up(result_flop, result_tokens, choice_part_faster, choice_many_texts)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
