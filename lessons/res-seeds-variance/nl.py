# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-seeds-variance. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-seeds-variance", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl) en [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="In je ablation-tabel eindigden de GPT zonder posities en de volledige GPT dicht bij elkaar. Elke rij is één run. Wat ontbreekt er nog voordat je kunt zeggen of het weghalen van de posities verschil maakte?",
        options={
            "bigger": "Een run met een grotere GPT",
            "right": "Meer runs van elke rij, met andere seeds",
            "longer": "Een langere run van de GPT zonder posities",
        },
        correct="right",
        why={
            "bigger": "Een grotere GPT is een andere opzet; die zegt niets over toeval in deze. Zie [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl).",
            "longer": "Een langere run is ook een andere opzet. Zie [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl).",
            "right": "Eén run per rij is één worp met de dobbelsteen: met andere seeds beweegt elk getal een beetje. Deze les meet hoeveel.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_interval():
    quick_interval = L.choice(
        "quick-interval",
        question="Een 95%-bootstrapinterval van een gemiddelde loopt van 2,39 tot 2,41. Wat zegt het?",
        options={
            "runs": "95% van alle losse waarden ligt tussen 2,39 en 2,41",
            "true": "Het echte gemiddelde is 2,40",
            "right": "Hoe ver het gemiddelde zou bewegen met andere steekproeven zoals deze",
        },
        correct="right",
        why={
            "runs": "De losse waarden spreiden wijder dan hun gemiddelde: het interval gaat over het gemiddelde. Zie [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl).",
            "true": "Het interval legt het gemiddelde niet vast op één getal; het geeft een gebied. Zie [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl).",
            "right": "De bootstrap trekt nieuwe steekproeven uit de steekproef die je hebt, en de middelste 95% van hun gemiddelden is het interval.",
        },
    )
    quick_interval
    return (quick_interval,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) eindigde de kleine GPT zonder warmup iets lager dan met warmup. Is dat de warmup, of toeval? De seed bepaalt de beginwaarden van de weights van een model en de volgorde van zijn mini-batches, en met een andere seed eindigt dezelfde run op een ander getal. Voordat je zegt dat een verandering helpt, moet je weten hoeveel een getal beweegt als er niets verandert behalve de seed. <!-- termen:ok -->

    **Een voorbeeld uit het dagelijks leven.** Je weegt jezelf, stapt eraf en weer erop: de weegschaal geeft 70,2, dan 70,4. Je bent geen 200 gram aangekomen. Voordat je een verandering van 200 gram gelooft, wil je weten hoeveel de weegschaal uit zichzelf beweegt.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "five-seeds",
        question="Je traint de kleine GPT uit Eén ding tegelijk veranderen vijf keer, met seeds 0 tot en met 4 en verder niets anders. Hoe ver denk je dat de laagste en de hoogste validation loss uit elkaar liggen?",
        options={
            "same": "Helemaal niet: de instellingen en de data zijn gelijk",
            "right": "Een paar honderdsten",
            "tenths": "Een paar tienden, zo ver als een andere learning rate",
        },
        reveal="**Een paar honderdsten.** In de runs van de course eindigden seeds 0 tot en met 4 op 2,390; 2,392; 2,417; 2,394 en 2,401: van laagste tot hoogste 0,027 uit elkaar, terwijl alleen de seed veranderde. Dat is meer dan de warmup veranderde in Eén ding tegelijk veranderen, 0,004 tot 0,008.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Run-to-run variance** is hoeveel het resultaat van dezelfde run verandert als alleen de seed verandert. De seed bepaalt de beginwaarden van de weights en de volgorde van de mini-batches; allebei veranderen ze waar het trainen uitkomt. Met één run kun je een echt effect niet van deze ruis onderscheiden.

        Dus train je elke instelling met meerdere seeds, dezelfde seeds voor elke instelling, en rapporteer je drie dingen:

        - het **gemiddelde** over de seeds: het getal dat je als resultaat rapporteert;
        - de **spreiding**: de laagste en de hoogste run;
        - een **95%-bootstrapinterval** van het gemiddelde, zoals in [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl): hoe ver het gemiddelde zelf zou kunnen bewegen met andere seeds.

        Dan vergelijk je een verschil tussen twee instellingen met die ruis. Overlappen hun intervallen veel, dan laten de intervallen geen verschil zien: met andere seeds hadden de twee gemiddelden andersom kunnen uitvallen. Dat bewijst niet dat er geen verschil is. Vijf seeds geven een ruw interval, vaak te smal; dat is nog altijd veel beter dan één getal. En dezelfde seeds voor beide instellingen koppelen de runs, seed voor seed: een klein verschil dat twee intervallen missen, kan per paar toch zichtbaar worden, zoals een latere les laat zien.

        Het plaatje: de runs van de course met de kleine GPT, seeds 0 tot en met 4, met warmup en zonder. Elk punt is één run; de balk is het 95%-bootstrapinterval van het gemiddelde, het streepje het gemiddelde.
        """),
        mo.Html(C.strip_picture([("met warmup", C.REFERENCE["control"]), ("zonder warmup", C.REFERENCE["warmup=0"])],
                                {"title": "Vijf seeds per rij", "x": "validation loss na 6 epochs", "lang": L.lang, "alt": "Twee rijen van vijf punten. Met warmup liggen de runs van 2,390 tot 2,417, gemiddelde 2,399, interval van het gemiddelde 2,392 tot 2,409. Zonder warmup van 2,384 tot 2,412, gemiddelde 2,394, interval 2,387 tot 2,403. De twee intervallen overlappen grotendeels."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **run-to-run variance** | hoeveel het resultaat van dezelfde run verandert met alleen een andere seed |
        """),
        L.watch_out("Een 95%-interval van het gemiddelde is niet waar 95% van de runs landt: de runs zelf spreiden wijder. Het zegt hoe zeker je van het gemiddelde kunt zijn."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een resultaat te rapporteren dat standhoudt:

    1. Kies de seeds: minstens 3, hier 5, dezelfde voor elke instelling.
    2. Train elke instelling met elke seed; verder verandert er niets.
    3. Rapporteer per instelling het gemiddelde, de laagste en de hoogste run, en het 95%-bootstrapinterval van het gemiddelde.
    4. Vergelijk: overlappen de intervallen van twee instellingen veel, dan laten de intervallen geen verschil zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 met verzonnen validation losses: vijf seeds van één instelling; stap 4 volgt onder de cel. Het commentaar zegt welk deel van stap 3 elke regel is. Hij is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_summary():
    import numpy as np

    _losses = np.array([1.652, 1.678, 1.664, 1.662, 1.671])
    # 3. het gemiddelde, en de laagste en de hoogste run
    print("gemiddelde:", _losses.mean(), " laagste:", _losses.min(), " hoogste:", _losses.max())
    # 3. het 95%-bootstrapinterval van het gemiddelde: de gemiddelden van 10.000 resamples, en hun middelste deel
    _rng = np.random.default_rng(0)
    _means = []
    for _ in range(10000):
        _means.append(np.mean(_rng.choice(_losses, size=len(_losses))))
    _low, _high = np.quantile(_means, [0.25, 0.75])  # maak af: de middelste 95%, [0.025, 0.975]
    print("interval:", _low, "tot", _high)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vijf runs liggen 0,026 uit elkaar, van 1,652 tot 1,678. Met de kwantielen 0,25 en 0,75 bevat het interval de middelste helft van de gemiddelden: veel te smal. Met 0,025 en 0,975 bevat het de middelste 95%, ongeveer 1,658 tot 1,672: smaller dan de spreiding van de runs, omdat een gemiddelde van vijf minder beweegt dan één run. **Stap 4:** een tweede instelling krijgt op dezelfde manier een eigen interval. Overlappen de twee veel, dan kunnen de intervallen de instellingen niet uit elkaar houden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_summary():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een samenvatting van meerdere seeds

        Maak `seed_summary(losses, n_boot, seed)` af: stap 3 van het recept als functie. Hij geeft drie getallen terug: het gemiddelde van `losses`, en de onder- en bovenkant van het 95%-bootstrapinterval van dat gemiddelde, uit `n_boot` resamples met een random number generator die met `seed` gemaakt is. Vervang de twee `...`. De check roept `seed_summary([2.41, 2.38, 2.44, 2.40, 2.39], 10000, 0)` aan, en probeert andere runs. Een bootstrap is willekeurig, dus de check staat de kleine verschillen toe die een andere manier van trekken geeft.
        """),
        L.task("Laat `seed_summary(losses, n_boot, seed)` het gemiddelde van `losses` teruggeven en de twee kanten van het 95%-bootstrapinterval van dat gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_summary():
    box_summary = L.code_box("seed-summary", lines=10)
    box_summary
    return (box_summary,)


@app.cell(hide_code=True)
def check_summary(box_summary):
    result_summary = L.check("seed-summary", box_summary.value)
    result_summary
    return (result_summary,)


@app.cell(hide_code=True)
def hints_summary():
    L.hints(
        "seed-summary",
        [
            "Dezelfde stappen als in het uitgewerkte voorbeeld, met de namen `losses`, `n_boot` en `rng`: `n_boot` keer het gemiddelde van een resample, met teruglegging getrokken; dan de twee kwantielen van die gemiddelden.",
            "In de loop: `means.append(np.mean(rng.choice(___, size=len(losses))))`. Daarna: `low, high = np.quantile(___, [___, ___])`.",
        ],
        why="```python\nimport numpy as np\n\n\ndef seed_summary(losses, n_boot, seed):\n    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n        means.append(np.mean(rng.choice(losses, size=len(losses))))\n    low, high = np.quantile(means, [0.025, 0.975])\n    return [np.mean(losses), low, high]\n```\n\nElke resample trekt evenveel runs als er zijn, met teruglegging, dus een run kan twee keer of helemaal niet voorkomen. De middelste 95% van de gemiddelden van de resamples is het interval van het gemiddelde.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je GPT met vijf seeds

    De knop traint de kleine GPT uit [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) vijf keer, met seeds 0 tot en met 4, met jouw `GPT`, `gpt_loaders` en `Trainer` op de CPU: een halve tot een hele minuut per run, dus samen een paar minuten. De rest van de les wacht terwijl hij traint, en de stopknop van de cel maakt er een eind aan. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def seeds_button_cell():
    seeds_button = mo.ui.run_button(label="Train je GPT met vijf seeds")
    seeds_button
    return (seeds_button,)


@app.cell(hide_code=True)
def demo_seeds(box_summary, result_summary, seeds_button):
    if not seeds_button.value:
        _out = mo.md("*Druk op de knop om de vijf runs te starten.*")
    else:
        _runs = C.run_seeds(lambda _i, _share: mo.output.replace(mo.md(f"*Bezig met seed {_i} van 0 tot en met 4: {round(100 * _share)}%*")))
        if _runs == "stopped":
            _out = mo.md("*Je hebt de runs gestopt. Druk op de knop om opnieuw te beginnen.*")
        elif isinstance(_runs, str):
            _out = mo.md("*Een run stopt met een fout. Komen je modules in `work/projects/llm/` en `work/projects/nn/` nog door hun checks? Verwijder een bestand dat je veranderde, en controleer zijn oefening opnieuw.*")
        else:
            _summary, _yours = C.summary_of(box_summary.value if result_summary.state == "correct" else None)
            _mean, _low, _high = _summary(_runs, 10000, 0)
            _who = "" if _yours else " Oefening 1 klopt nog niet, dus de `seed_summary` van de course maakte de samenvatting."
            _out = mo.vstack([
                mo.Html(C.strip_picture([("jij, met warmup", _runs), ("course, met warmup", C.REFERENCE["control"]), ("course, zonder warmup", C.REFERENCE["warmup=0"])],
                                        {"title": "Vijf seeds per rij", "x": "validation loss na 6 epochs", "lang": L.lang,
                                         "alt": f"Drie rijen van elk vijf runs. Jouw runs liggen van {C.number(min(_runs), 3, L.lang)} tot {C.number(max(_runs), 3, L.lang)}, met een gemiddelde van {C.number(_mean, 3, L.lang)}."})),
                mo.md(f"Je vijf runs eindigden op {'; '.join(C.number(_v, 3, L.lang) for _v in _runs)}: een spreiding van {C.number(max(_runs) - min(_runs), 3, L.lang)}, een gemiddelde van {C.number(_mean, 3, L.lang)}, en een 95%-interval van het gemiddelde van {C.number(_low, 3, L.lang)} tot {C.number(_high, 3, L.lang)}.{_who}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.md(r"""
    ### Oefening 2. Doet warmup er hier toe?
    """)
    return


@app.cell(hide_code=True)
def choice_overlap_cell():
    choice_overlap = L.choice(
        "overlap",
        question="De vijf seeds van de course: met warmup gemiddelde 2,399 en 95%-interval 2,392 tot 2,409; zonder warmup gemiddelde 2,394 en interval 2,387 tot 2,403. Wat kun je zeggen op grond van deze twee intervallen?",
        options={
            "lower": "Zonder warmup is hier beter: het gemiddelde is lager",
            "none": "Warmup maakt hier geen verschil: de intervallen overlappen",
            "right": "Ze houden de twee niet uit elkaar: ze overlappen veel",
        },
        correct="right",
        why={
            "lower": "Het gemiddelde is 0,005 lager, minder dan een gemiddelde van vijf runs door toeval beweegt: de twee intervallen alleen laten dat niet zien. Zie *Het idee*.",
            "none": "Overlappende intervallen laten niet zien dat er geen verschil is. Met elk van de vijf seeds eindigde de run zonder warmup lager: seed voor seed kunnen de runs laten zien wat twee intervallen missen. Zie *Het idee*.",
            "right": "Als twee groepen is het verschil van 0,005 kleiner dan de ruis tussen seeds. Maar dat de run zonder warmup met elk van de vijf seeds lager eindigde, is een aanwijzing dat de intervallen iets missen; een latere les toetst die, door de runs seed voor seed te vergelijken.",
        },
    )
    choice_overlap
    return (choice_overlap,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Oefening 3. Rapporteer je resultaat
    """)
    return


@app.cell(hide_code=True)
def explain_report():
    explain_report_cell = L.explain(
        "report",
        question="Schrijf het resultaat van je vijf runs op zoals je dat in een paper zou doen: de opzet, het aantal seeds, het gemiddelde, de spreiding en het interval, en één zin over wat het betekent. Nog geen demo gedraaid? Gebruik de getallen van de course in het plaatje van Het idee.",
        model_answer="**Opzet:** de kleine GPT uit Eén ding tegelijk veranderen, met warmup, 6 epochs op het eerste kwart van Tiny Shakespeare, seeds 0 tot en met 4. **Resultaat:** een validation loss van gemiddeld 2,399, van 2,390 tot 2,417, met een 95%-bootstrapinterval van het gemiddelde van 2,392 tot 2,409. **Betekenis:** twee runs van deze opzet kunnen alleen door toeval 0,03 uit elkaar liggen, dus een verschil van een paar duizendsten tussen twee instellingen, gemeten met één run elk, zegt niets.",
        points=[
            "je noemt de opzet en de vijf seeds",
            "je geeft het gemiddelde, de spreiding van de laagste tot de hoogste run, en het 95%-interval van het gemiddelde",
            "je zegt wat het betekent: hoeveel één run door toeval kan bewegen, dus hoe groot een verschil moet zijn voordat het iets zegt",
        ],
    )
    explain_report_cell
    return (explain_report_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) en [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl) vergeleken elk één run met één run; nu weet je hoeveel één run door toeval beweegt. Het interval komt uit [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl). Met meerdere seeds per instelling heb je veel runs om bij te houden: de volgende les, *Een experimentlogboek bijhouden*, laat zien hoe je elke run opschrijft. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_overlap, explain_report_cell, result_summary):
    L.wrap_up(result_summary, choice_overlap, explain_report_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
