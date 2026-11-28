# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-why-training-costs-millions. Route: course/route/lessen/systemen.yaml. Checks, the published GPU hours, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-why-training-costs-millions", "nl")
    C = L.checks   # de GPU-uren die de makers publiceerden, de runs van de oefeningen, de demo, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_time():
    quick_time = L.choice(
        "quick-time",
        question="Een trainingsstap heeft 6 biljoen FLOPs nodig, en je GPU haalt daarbij 3 biljoen FLOP/s. Hoe lang duurt één stap?",
        options={
            "half": "0,5 seconde",
            "right": "Ongeveer 2 seconden",
            "big": "18 seconden",
            "depends": "Dat hangt af van de grootte van het model, dus FLOPs en FLOP/s alleen zeggen het niet",
        },
        correct="right",
        why={
            "right": "Tijd is het werk gedeeld door het tempo: 6 biljoen FLOPs bij 3 biljoen per seconde duren 2 seconden. In deze les loopt dezelfde deling over maanden, op duizenden GPU's.",
            "half": "0,5 is het tempo gedeeld door het werk. Welke van de twee staat boven de deelstreep als je een tijd wilt? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
            "big": "18 is het werk keer het tempo, en dat is geen tijd. Wat doe je met het werk en het tempo om seconden te krijgen? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
            "depends": "De grootte van het model zit al in de 6 biljoen FLOPs. Wat geven het werk en het tempo samen? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
        },
    )
    quick_time
    return (quick_time,)


@app.cell(hide_code=True)
def quick_check_many():
    quick_many = L.choice(
        "quick-many",
        question="Je netwerk schrijft 1 tekst met 50 tekens per seconde. Met 100 teksten tegelijk schrijft het netwerk in totaal 2.000 tekens per seconde. Wat ging omhoog?",
        options={
            "each": "De snelheid van elke tekst apart, van 50 naar 2.000 tekens per seconde",
            "flops": "De FLOPs van één teken",
            "right": "De throughput: het werk van alle teksten samen per seconde",
            "rate": "De hoogste FLOP/s die de GPU kan halen",
        },
        correct="right",
        why={
            "right": "Throughput telt de tekens van alle teksten samen: 2.000 per seconde, terwijl elke tekst er 20 krijgt, trager dan alleen. Duizenden GPU's werken net zo: samen doen de GPU's meer per seconde, en geen enkele GPU wordt sneller.",
            "each": "2.000 per seconde zijn alle 100 teksten samen, dus krijgt elke tekst er 20. Welk woord telt het werk van alles samen per seconde? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
            "flops": "Een teken kost evenveel FLOPs als eerst; er lopen er alleen meer tegelijk. Wat ging omhoog? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
            "rate": "Wat een GPU hoogstens kan verandert niet door wat je erop draait. Wat ging omhoog? Zie [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl).",
        },
    )
    quick_many
    return (quick_many,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je bigram-netwerk trainde in seconden op je eigen computer. Een groot language model traint op dezelfde manier, met miljarden parameters en biljoenen tokens, en het werk groeit met allebei. Volgens Meta kostte de training van Llama 3.1 405B 30.840.000 GPU-uren, en elk van deze uren wordt betaald: als huur, of als gekochte GPU's, stroom en mensen. Krantenkoppen maken van zulke getallen één prijs, en de prijs hangt af van wat er wordt meegeteld. In deze les reken je de rekening zelf uit, met de getallen die de makers publiceerden, en zie je wat erbuiten valt.

    **Een voorbeeld uit het dagelijks leven.** Een schilder maakt een offerte voor een huis: 3 schilders, 4 dagen van 8 uur, samen 96 werkuren, en tegen 45 euro per uur is dat 4.320 euro. Met 6 schilders is het werk in 2 dagen klaar, voor dezelfde 4.320 euro. De offerte laat de verf weg, en het bezoek om eerst naar de muren te kijken.
    """)
    return


@app.cell(hide_code=True)
def predict_years():
    guess_years = L.guess(
        "years",
        question="Volgens Meta kostte de training van Llama 3.1 405B 30.840.000 GPU-uren op de H100. Als één GPU al dat werk alleen had gedaan, hoe lang had dat geduurd, denk je?",
        options={
            "month": "Ongeveer een maand, want een GPU doet elke seconde biljoenen FLOPs",
            "years": "Ongeveer 35 jaar",
            "centuries": "Ongeveer 350 jaar",
            "right": "Ongeveer 3.500 jaar",
        },
        reveal="**Ongeveer 3.500 jaar.** 30.840.000 uur is 1.285.000 dagen, en dat is 3.518 jaar. Een groot model traint dus op duizenden GPU's tegelijk: Meta trainde het model van 405B op maximaal 16.384 H100-GPU's, en zo werd de tijd maanden. Het werk blijft hetzelfde, en de rekening ook: elk van de 30.840.000 GPU-uren wordt betaald.",
    )
    guess_years
    return (guess_years,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een GPU-uur.** Een **GPU-uur** is één GPU die één uur wordt gebruikt. Een run op 8 GPU's van 10 uur gebruikt 80 GPU-uren, en een run op 80 GPU's van 1 uur ook. De makers van grote modellen melden de training in GPU-uren:

        | Model | Maker, jaar | GPU | GPU-uren |
        |---|---|---|---|
        | Llama 2 7B | Meta, 2023 | A100 | 184.320 |
        | Llama 2 70B | Meta, 2023 | A100 | 1.720.320 |
        | DeepSeek-V3 | DeepSeek, 2024 | H800 | 2.788.000 |
        | Llama 3.1 405B | Meta, 2024 | H100 | 30.840.000 |

        De H100 en de H800 zijn nieuwer dan de A100, en een uur van een nieuwere GPU doet meer werk.

        **De rekening.** De **trainingskosten** zijn de GPU-uren keer de prijs van één GPU-uur. DeepSeek telde de training van V3 in drie delen, en rekende 2 dollar per GPU-uur op de H800, een prijs om een H800 te huren:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Lees de afbeelding van boven naar beneden: drie delen maken 2.788.000 GPU-uren, en tegen 2 dollar per uur is dat 5.576.000 dollar. DeepSeek schrijft zelf wat hier niet in zit: het eerdere onderzoek, en de proefruns om de architectuur, de methode en de data te kiezen.

        **De tijd.** De GPU-uren gedeeld door het aantal GPU's geven de uren die de run duurt, en gedeeld door 24 de dagen. Het cluster van DeepSeek heeft 2.048 H800-GPU's, en de pretraining kostte 2.664.000 GPU-uren; volgens DeepSeek duurde de pretraining minder dan twee maanden. Meer GPU's maken een run korter, niet goedkoper: de GPU-uren blijven gelijk.

        **Wat er nog meer geld kost.** Runs stoppen. Meta trainde hetzelfde model van 405B op maximaal 16.384 H100-GPU's, en in een periode van 54 dagen pretraining stopte de run 466 keer, 419 keer onverwacht, waarvan ongeveer de helft door kapotte GPU's en het geheugen van de GPU's. Toch trainde het cluster meer dan 90 procent van de tijd. Aan de laatste run gaan de tests en de proefruns vooraf, en mensen bouwen het model. En een huurprijs is niet wat een eigenaar betaalt: een eigenaar koopt de GPU's, en betaalt de stroom en het gebouw. Een latere les telt op wat het kost om computers te bezitten. Alle getallen hier zijn gelezen op 28 november 2026.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **GPU-uur** | | één GPU die één uur wordt gebruikt; de gebruikelijke eenheid voor de kosten van een training | `gpus * hours` |
        | **trainingskosten** | | de GPU-uren van een training keer de prijs van één GPU-uur; het onderzoek, de proefruns en de mensen eromheen komen erbij | `gpu_hours * price` |
        | `( )` | "haakjes" | Python rekent eerst uit wat ertussen staat | `12 / (2 * 3)` |
        """),
        L.watch_out("Kosten in een krantenkop zijn vaak alleen de laatste run. De 5.576.000 dollar van DeepSeek zegt dat zelf, en een lab betaalt voor veel meer runs dan de run die werkte."),
        L.watch_out("Vergelijk GPU-uren alleen op dezelfde GPU. Een uur van een H100 doet meer werk dan een uur van een A100, dus betekenen minder GPU-uren op een nieuwere GPU niet per se minder werk."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De rekening van een training:

    1. Zoek de GPU-uren van elke run, of de GPU's en de tijd: GPU-uren zijn GPU's keer uren.
    2. Tel elke run op: de tests, de proefruns, de runs die stopten, en de laatste run.
    3. Vermenigvuldig met de prijs van één GPU-uur: de huur, of wat een uur van een GPU je kost als je de GPU bezit.
    4. Voor de tijd: deel de GPU-uren door het aantal GPU's, en door 24 voor dagen.
    5. Zeg wat er niet in zit: het onderzoek, de mensen, de data, en de GPU's zelf als je de GPU's kocht.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.DEEPSEEK` is een dict met de GPU-uren van de drie delen van de training van DeepSeek-V3, en `C.PRICE` is de 2 dollar per GPU-uur waarmee DeepSeek rekende. De cel telt de delen op en vermenigvuldigt daarna met de prijs. Voer de cel uit.
    """)
    return


@app.cell
def example_bill():
    _hours = 0
    for _part in C.DEEPSEEK:
        print(_part, C.DEEPSEEK[_part])
        _hours = _hours + C.DEEPSEEK[_part]
    print("GPU hours", _hours)
    print("dollars", _hours * C.PRICE)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    2.788.000 GPU-uren tegen 2 dollar is 5.576.000 dollar, het getal dat DeepSeek meldde, en de pretraining is daarvan 96 procent. De uitkomst is een float, `5576000.0`, want `C.PRICE` is de float `2.0`. Oefening 1 doet hetzelfde voor alle runs van een project.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De rekening van een project

        `cost(runs, price)` krijgt een lijst van runs, elk een dict met het aantal GPU's onder `"gpus"` en het aantal dagen onder `"days"`, en de prijs van één GPU-uur. De loop gaat langs elke `run`, en `total = total + hours * price` telt de kosten van de run op bij `total`. Vervang de `...` door de GPU-uren van de run: de GPU's keer de uren, waarbij een dag 24 uur heeft.

        `C.RUNS` bevat de vijf runs van een verzonnen project: een kleine test op 8 GPU's van 2 dagen, twee proefruns op 64 GPU's van 5 dagen, een run op 512 GPU's, gestopt na 9 dagen, en de laatste run op 512 GPU's van 30 dagen. De check roept `cost(C.RUNS, 2.0)` aan. Daarna één GPU voor één dag, dezelfde runs tegen 1 dollar, helemaal geen runs, en eigen runs.
        """),
        L.task("Tel de GPU-uren van elke run, zodat cost de rekening van alle runs geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=7)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "`run[\"gpus\"]` is het aantal GPU's van de run, en `run[\"days\"]` het aantal dagen. `hours` moet de GPU-uren bevatten: elke GPU, voor elk uur van de run. Hoeveel uur heeft een dag, en wat vermenigvuldig je?",
            "8 GPU's voor 2 dagen: `8 * 2 * 24` geeft `384` GPU-uren.",
            "`        hours = run[\"gpus\"] * ___`",
        ],
        why="`hours = run[\"gpus\"] * run[\"days\"] * 24` telt elke GPU voor elk uur van de run. De vijf runs van `C.RUNS` zijn samen 494.976 GPU-uren: 989.952 dollar tegen 2 dollar.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_days():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel dagen?

        `days(gpu_hours, gpus)` krijgt de GPU-uren van een run en het aantal GPU's waarop de run draait, en geeft terug hoeveel dagen de run duurt. Schrijf na `return` één regel die de GPU-uren deelt door de GPU's en door 24. Python deelt en vermenigvuldigt van links naar rechts: `12 / 2 * 3` geeft `18.0`, en `12 / (2 * 3)` geeft `2.0`.

        De check roept `days(2664000, 2048)` aan: de pretraining van DeepSeek-V3 op het eigen cluster. Daarna één biljoen tokens van dezelfde training, Llama 3.1 405B op 16.384 GPU's, en eigen kleine getallen.
        """),
        L.task("Geef terug hoeveel dagen een run van gpu_hours duurt op gpus GPU's."),
    ])
    return


@app.cell(hide_code=True)
def exercise_days():
    box_days = L.code_box("days", lines=3)
    box_days
    return (box_days,)


@app.cell(hide_code=True)
def check_days(box_days):
    result_days = L.check("days", box_days.value)
    result_days
    return (result_days,)


@app.cell(hide_code=True)
def hints_days():
    L.hints(
        "days",
        [
            "`gpu_hours` gedeeld door de GPU's geeft de uren van de run, en nog eens gedeeld door 24 de dagen. Haakjes laten Python vermenigvuldigen voordat Python deelt. Wat komt er tussen de haakjes?",
            "`2664000 / 2048` geeft `1300.78125` uur van het hele cluster, en `1300.78125 / 24` geeft ongeveer `54.2` dagen.",
            "`    return gpu_hours / (___)`",
        ],
        why="`return gpu_hours / (gpus * 24)` deelt in één keer door de GPU's en door de 24 uren van een dag. 2.664.000 GPU-uren op 2.048 GPU's duren 54,2 dagen, en op 16.384 GPU's, acht keer zoveel, 6,8 dagen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bill():
    mo.md(r"""
    ### Oefening 3. Wat zit er in de 5.576.000 dollar?

    Zeven onderdelen van het maken van DeepSeek-V3. Zet elk onderdeel onder *In de 5.576.000 dollar* als DeepSeek het onderdeel meetelde, of onder *Er niet in*.
    """)
    return


@app.cell(hide_code=True)
def sort_bill_cell():
    sort_bill = L.sort("bill")
    sort_bill
    return (sort_bill,)


@app.cell(hide_code=True)
def check_bill(sort_bill):
    result_bill = L.check("bill", sort_bill)
    result_bill
    return (result_bill,)


@app.cell(hide_code=True)
def hints_bill():
    L.hints(
        "bill",
        [
            "Vraag bij elk onderdeel: is het één van de delen van de training van V3 zelf, zoals DeepSeek de delen telde?",
            "Drie onderdelen zitten erin, en vier niet.",
            "Erin: pretraining, \\_\\_\\_ en \\_\\_\\_. Er niet in: proefruns, \\_\\_\\_, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Erin:** pretraining, langere teksten en post-training. **Er niet in:** proefruns, eerder onderzoek, de mensen, en de GPU's zelf.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_same():
    mo.md(r"""
    ### Oefening 4. Twee labs

    Lab A huurt 1.024 GPU's voor 40 dagen. Lab B huurt 2.048 van dezelfde GPU's voor 20 dagen, tegen dezelfde prijs per GPU-uur.
    """)
    return


@app.cell(hide_code=True)
def choice_same_cell():
    choice_same = L.choice(
        "same",
        question="Welk lab betaalt meer?",
        options={
            "long": "Lab A, want de run van A duurt twee keer zo lang",
            "right": "Geen van beide: allebei betalen 983.040 GPU-uren",
            "many": "Lab B, want B huurt twee keer zoveel GPU's",
            "order": "Lab B, want een twee keer zo groot cluster kost meer per GPU, zoals elke grotere bestelling",
        },
        correct="right",
        why={
            "right": "De rekening is GPU's keer uren keer de prijs: 1.024 GPU's voor 960 uur en 2.048 GPU's voor 480 uur maken allebei 983.040 GPU-uren. Lab B heeft het model na 20 dagen in plaats van 40: meer GPU's kopen tijd, geen kleinere rekening. Op een echt cluster gaat er bij B iets meer tijd op aan GPU's die op elkaar wachten.",
            "long": "A draait twee keer zo lang, op half zoveel GPU's. Wat geven 1.024 keer 960 uur en 2.048 keer 480 uur?",
            "many": "B huurt twee keer zoveel GPU's, voor half zo lang. Op hoeveel GPU-uren komen de twee labs?",
            "order": "De prijs per GPU-uur is voor allebei gelijk, zoals de vraag zegt. Wat kan B verder meer laten betalen?",
        },
    )
    choice_same
    return (choice_same,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Zeven trainingen

    De GPU-uren die Meta en DeepSeek publiceerden voor zeven modellen, van de minste naar de meeste. Voor het gekozen model zie je de kosten bij 2 dollar per GPU-uur, hoe lang één GPU erover zou doen, en hoe lang het cluster van DeepSeek met 2.048 GPU's erover zou doen. Kies een model.
    """)
    return


@app.cell(hide_code=True)
def demo_model():
    L.explore("model", "model", lambda _v: C.frame(_v, L.lang), start=5, values=C.DEMO_VALUES, role="",
              task="Ga van het kleinste model naar het grootste: hoe groeien de kosten en de tijd?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Llama 3.1 405B kostte 30.840.000 GPU-uren, 11 keer de 2.788.000 van DeepSeek-V3: tegen 2 dollar 61.680.000 dollar tegenover 5.576.000. Op de 2.048 GPU's van DeepSeek had het model van 405B 627 dagen nodig gehad; Meta gebruikte er maximaal 16.384, acht keer zoveel. Llama 2 7B, het kleinste, kostte 184.320 GPU-uren: 21 jaar op één GPU, en 4 dagen op 2.048. De balken vergelijken uren van verschillende GPU's: de drie balken van Llama 2 zijn uren van de oudere A100. En elk getal is wat de maker meetelde: Meta telt de pretraining van Llama 2 en de training van elk model van Llama 3.1, en DeepSeek telt drie delen en laat het onderzoek weg. Niemand telt de mensen.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Hoe snel is je model? FLOPs, FLOP/s en tokens per seconde](/course/open/sys-how-fast?lang=nl) was de tijd het werk gedeeld door het tempo van één computer; hier is het werk zo groot dat duizenden GPU's het werk delen, en heeft elk GPU-uur een prijs. De volgende les is *De wet van Moore en de groei van compute*: waarom elke nieuwe generatie GPU's meer werk per uur doet, en hoe snel de compute van de grootste trainingen groeide.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same, result_bill, result_cost, result_days):
    L.wrap_up(result_cost, result_days, result_bill, choice_same)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
