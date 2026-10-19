# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-experiment-log. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-experiment-log", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl) en [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl). <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def quick_check_seeds():
    quick_seeds = L.choice(
        "quick-seeds",
        question="Je vergelijkt twee instellingen met elk vijf seeds. Welke seeds krijgt de tweede instelling?",
        options={
            "right": "Dezelfde vijf als de eerste",
            "new": "Vijf nieuwe, zodat je meer toeval ziet",
            "any": "Het maakt niet uit welke",
        },
        correct="right",
        why={
            "new": "Nieuwe seeds brengen andere beginwaarden van de weights en een andere volgorde van de mini-batches in de vergelijking. Zie [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
            "any": "De seed alleen verschuift het resultaat al een paar honderdsten. Zie [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
            "right": "Met dezelfde seeds beginnen de twee instellingen met dezelfde willekeurige getallen, dus kun je hun runs seed voor seed vergelijken.",
        },
    )
    quick_seeds
    return (quick_seeds,)


@app.cell(hide_code=True)
def quick_check_ablation():
    quick_ablation = L.choice(
        "quick-ablation",
        question="Bij een ablation train je je GPT zonder één deel. Wat blijft hetzelfde als bij de run van de volledige GPT?",
        options={
            "data": "Alleen de data en het aantal epochs",
            "right": "Al het andere",
            "nothing": "Niets hoeft: het is toch een ander model",
        },
        correct="right",
        why={
            "data": "Verschillen de instellingen of de seed ook, dan heeft een verschil meer dan één oorzaak. Zie [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl).",
            "nothing": "Dan kun je niet zeggen wat het ontbrekende deel deed. Zie [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl).",
            "right": "Data, instellingen, epochs en seed blijven gelijk; alleen het deel dat je weghaalt verschilt, dus komt een verschil van dat deel of van toeval.",
        },
    )
    quick_ablation
    return (quick_ablation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de laatste drie lessen trainde je je GPT vaak: een control run en zijn varianten in [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl), een GPT zonder posities in [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl), vijf seeds in [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl). Over een maand vraagt iemand welke instellingen 2,39 gaven, en met welke versie van je code. Een checkpoint bevat weights, geen instellingen, en een notebook vergeet zijn getallen als het dichtgaat. <!-- termen:ok -->

    Ook de code zelf hoort bij een experiment. In 2018 lieten Peter Henderson en zijn collega's zien dat openbare implementaties van dezelfde leeralgoritmen, met dezelfde instellingen, duidelijk verschillende resultaten gaven ([Henderson et al., 2018](https://arxiv.org/abs/1709.06560)). Dus schrijf je ook op welke code er draaide.

    **Een voorbeeld uit het dagelijks leven.** Een bakker die twaalf broden probeert en niets opschrijft, weet achteraf dat het zevende het beste was, maar niet wat erin zat. Een bakker die bij elk brood de bloem, het water, de oven en de tijd opschrijft, kan het zevende opnieuw bakken.
    """)
    return


@app.cell(hide_code=True)
def predict_left():
    guess_left = L.guess(
        "runs-left",
        question="Als je op elke knop drukte, trainden die drie lessen je GPT acht keer of vaker. Hoeveel van die runs kun je nu nog terugvinden, met hun instellingen en hun resultaat?",
        options={
            "all": "Allemaal: de course bewaart elke run",
            "right": "Hooguit één",
            "none": "Geen enkele",
        },
        reveal="**Hooguit één.** Van die runs bewaart je project er maar één: de beste run van de GPT zonder posities, in `work/projects/llm/gpt_nopos.json`. Zelfs dat bestand heeft geen learning rate of warmup; die staan alleen in je code. (Ernaast bewaart `gpt_chars.json` de beste run van je volledige GPT uit Project: je GPT op een boek trainen.) De runs van Eén ding tegelijk veranderen en van de vijf seeds bestonden alleen in het notebook: toen dat dichtging, waren hun getallen weg.",
    )
    guess_left
    return (guess_left,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **experimentlogboek** is een bestand met één regel voor elke run die je maakt: de codeversie, alle instellingen, de seed en het resultaat. De code die traint schrijft die regel zelf, direct na de run, en een regel verander je daarna nooit meer. Dan staat in het logboek wat er draaide, niet wat je je herinnert.

        Elke regel is geschreven in **JSON**, een tekstformaat voor data zoals dictionaries en lijsten: `json.dumps(run)` maakt van een dictionary één regel tekst, en `json.loads(line)` maakt van die regel weer een dictionary. Een bestand met één JSON-regel per run heet **JSON Lines**, vaak met de extensie `.jsonl`. `open(path, "a")` opent een bestand om iets aan het eind toe te voegen; `"w"` maakt het eerst leeg.

        De **codeversie** zegt welke code er draaide. Hier is dat een **hash** van de code: een korte reeks letters en cijfers die uit de tekst berekend wordt, dezelfde voor dezelfde tekst, en een totaal andere als er één teken verandert. `hashlib.sha256` berekent hem, en de eerste 8 tekens zijn genoeg om je versies uit elkaar te houden. Houd je je code bij in Git, dan doet de commit-hash hetzelfde. Tools zoals MLflow en Weights & Biases houden zo'n logboek voor je bij, met een website om erin te bladeren; onder de motorkap is het hetzelfde idee.

        Het plaatje: één regel uit het logboek van de course, met zijn vier delen.
        """),
        mo.Html(C.line_picture({"title": "Eén run, één regel", "code": "codeversie", "settings": "alle instellingen", "seed": "seed", "val_loss": "resultaat",
                                "under": "De puntjes staan voor de vijf andere instellingen; de echte regel bevat ze alle acht.",
                                "alt": "Eén regel JSON met vier delen: de codeversie, een dictionary met alle instellingen, de seed 0 en de validation loss 2,3902."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **experimentlogboek** | een bestand met één regel per run: codeversie, instellingen, seed en resultaat, geschreven door de code die traint |
        | **JSON** | een tekstformaat voor data zoals dictionaries en lijsten |
        | **JSON Lines** | een bestand met één JSON-regel per record, vaak `.jsonl` |
        | **hash** | een korte reeks letters en cijfers die uit een tekst berekend wordt: dezelfde voor dezelfde tekst, een andere als er iets verandert |
        | **codeversie** | met welke code een run draaide, zoals de hash van die code |
        """),
        L.watch_out("Een regel die je later met de hand intypt, bevat wat je je herinnert. Laat de code die traint de regel schrijven, direct na de run, met de instellingen die hij echt gebruikte."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo houd je een experimentlogboek bij:

    1. Verzamel vóór de run de codeversie, alle instellingen, ook die je niet veranderde, en de seed.
    2. Noteer na de run het resultaat, zoals de validation loss.
    3. Voeg de run als één JSON-regel toe aan het eind van het logboek, met `"a"`; verander nooit een regel die er al staat.
    4. Om een vraag te beantwoorden lees je de regels terug met `json.loads` en kies je de runs die je nodig hebt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee runs van één instelling, met seeds 0 en 1, in een nieuw logboek, en er weer uit. Het commentaar zegt welke stap elk deel is. Hij is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit. `.encode()` maakt van de tekst bytes, en die neemt `sha256` aan; `.hexdigest()` geeft de hash terug als tekst.
    """)
    return


@app.cell
def example_log():
    import hashlib
    import json

    _path = C.scratch_path()  # een nieuw bestand in een tijdelijke map, niet in je project
    _code = "lr = 0.001\nwarmup = 100\n"  # staat voor de code van de runs
    # 1. de codeversie: de eerste 8 tekens van de hash van de code
    _version = hashlib.sha256(_code.encode()).hexdigest()[:8]
    for _seed, _loss in [(0, 2.3902), (1, 2.3924)]:
        # 1-2. het verslag van één run, met zijn resultaat
        _run = {"code": _version, "settings": {"lr": 0.001, "warmup": 100}, "seed": _seed, "val_loss": _loss}
        # 3. één JSON-regel aan het eind van het logboek
        with open(_path, "w") as _f:  # maak af: "a" voegt toe aan het eind, "w" maakt het bestand eerst leeg
            _f.write(json.dumps(_run) + "\n")
    # 4. lees elke regel terug
    with open(_path) as _f:
        _log = [json.loads(_line) for _line in _f]
    print("runs in het logboek:", len(_log))
    for _r in _log:
        print(_r)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met `"w"` bevat het logboek één run, seed 1: elke `open` maakte het bestand leeg, dus de run met seed 0 is weg. Met `"a"` bevat het ze allebei, elk op een eigen regel. Verander één teken van `_code`, bijvoorbeeld `0.001` in `0.002`, en voer de cel opnieuw uit: de codeversie is een totaal andere.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een run in het logboek schrijven

        Maak `log_run(path, settings, seed, val_loss, version)` af: stap 1 tot en met 3 van het recept als functie. Hij maakt het verslag van één run, een dictionary met de keys `"code"` (de versie), `"settings"`, `"seed"` en `"val_loss"`, en voegt dat als één JSON-regel toe aan het eind van het bestand op `path`. Vervang de twee `...`. De check roept hem twee keer aan op een nieuw bestand en leest de regels terug, en probeert een bestand waar al een run in staat.
        """),
        L.task("Laat `log_run(path, settings, seed, val_loss, version)` één JSON-regel met de keys `\"code\"`, `\"settings\"`, `\"seed\"` en `\"val_loss\"` toevoegen aan het eind van het bestand op `path`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_log():
    box_log = L.code_box("log-run", lines=8)
    box_log
    return (box_log,)


@app.cell(hide_code=True)
def check_log(box_log):
    result_log = L.check("log-run", box_log.value)
    result_log
    return (result_log,)


@app.cell(hide_code=True)
def hints_log():
    L.hints(
        "log-run",
        [
            "Het verslag is een dictionary, zoals in het uitgewerkte voorbeeld, met `version` onder de key `\"code\"`. Open daarna het bestand met `\"a\"` en schrijf het verslag als één regel.",
            "`run = {\"code\": version, \"settings\": ___, \"seed\": ___, \"val_loss\": ___}`, daarna `with open(path, ___) as f:` en daaronder `f.write(json.dumps(___) + \"\\n\")`.",
        ],
        why="```python\nimport json\n\n\ndef log_run(path, settings, seed, val_loss, version):\n    run = {\"code\": version, \"settings\": settings, \"seed\": seed, \"val_loss\": val_loss}\n    with open(path, \"a\") as f:\n        f.write(json.dumps(run) + \"\\n\")\n```\n\n`\"a\"` voegt toe aan het eind, dus elke run die er al stond blijft staan. `json.dumps` maakt één regel van de dictionary, en `\"\\n\"` sluit die regel af, zodat de volgende run op een eigen regel begint.",
    )
    return


@app.cell(hide_code=True)
def text_version():
    mo.md(r"""
    ### Je eigen codeversie

    De cel hieronder berekent de codeversie van je GPT-project: de hash van alle `.py`-bestanden in `work/projects/llm/` en `work/projects/nn/`, de code waarmee je GPT traint. Elke keer dat een check daar nieuwe code bewaart, of dat je zelf een bestand verandert, verandert de versie.
    """)
    return


@app.cell(hide_code=True)
def demo_version():
    _version, _files = C.project_version()
    if _version is None:
        _out = mo.md("*Je project heeft nog geen bestanden in `work/projects/llm/`. Die komen met de lessen over de GPT, zoals [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl).*")
    else:
        _out = mo.md(f"De {_files} bestanden van je GPT-project hebben nu codeversie **`{_version}`**. Een run die je vandaag logt krijgt deze versie; een run na een verandering van je code krijgt een andere.")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_find():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Runs terugvinden

        Het logboek van de course bevat zijn eigen 16 runs van de kleine GPT, uit [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) en [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl), met al hun instellingen. Maak `find_runs(log, wanted)` af: stap 4 van het recept. Hij geeft, in de volgorde van het logboek, elke run terug waarvan de instellingen voor elke key van `wanted` de waarde van `wanted` hebben. `wanted` bevat alleen de instellingen waar je naar vraagt, zoals `{"warmup": 0}`. Vervang de `...`. De check roept `find_runs(log, {"warmup": 0})` aan, wat de 5 runs zonder warmup moet geven, en stelt ook andere vragen. <!-- termen:ok -->
        """),
        L.task("Laat `find_runs(log, wanted)` elke run van `log` teruggeven waarvan de instellingen voor elke key van `wanted` de waarde van `wanted` hebben, in de volgorde van het logboek."),
    ])
    return


@app.cell(hide_code=True)
def exercise_find():
    box_find = L.code_box("find-runs", lines=8)
    box_find
    return (box_find,)


@app.cell(hide_code=True)
def check_find(box_find):
    result_find = L.check("find-runs", box_find.value)
    result_find
    return (result_find,)


@app.cell(hide_code=True)
def hints_find():
    L.hints(
        "find-runs",
        [
            "Begin voor elke run met `same = True`. Loop langs de keys van `wanted`; als de instelling van de run verschilt van `wanted[key]`, zet je `same = False`. Houd de run als `same` nog `True` is.",
            "`for key in wanted:` met daaronder `if run[\"settings\"][key] != wanted[key]:` en `same = False`; na die loop `if same:` en `found.append(run)`.",
        ],
        why="```python\ndef find_runs(log, wanted):\n    found = []\n    for run in log:\n        same = True\n        for key in wanted:\n            if run[\"settings\"][key] != wanted[key]:\n                same = False\n        if same:\n            found.append(run)\n    return found\n```\n\nEen run past als geen enkele key van `wanted` een andere waarde heeft in zijn instellingen. Een lege `wanted` vraagt nergens naar, dus dan past elke run.",
    )
    return


@app.cell(hide_code=True)
def text_demo_find():
    mo.md(r"""
    ### Vraag het logboek van de course

    Kies een vraag: jouw `find_runs` haalt de runs uit het logboek van de course. Zolang oefening 2 nog niet goed is, doet de `find_runs` van de course het.
    """)
    return


@app.cell(hide_code=True)
def question_cell():
    question = mo.ui.dropdown(
        options={"de control run": "control", "learning rate 0,003": "lr=0.003", "geen warmup": "warmup=0", "4 layers": "layers=4"},
        value="geen warmup",
        label="Welke runs?",
    )
    question
    return (question,)


@app.cell(hide_code=True)
def demo_find(box_find, question, result_find):
    _find, _yours = C.find_runs_of(box_find.value if result_find.state == "correct" else None)
    _wanted = C.QUESTIONS[question.value]
    _runs = _find(C.LOG, _wanted)
    _who = "Jouw `find_runs`" if _yours else "De `find_runs` van de course"
    if _runs:
        _mean = sum(_r["val_loss"] for _r in _runs) / len(_runs)
        _out = mo.md(f"{_who} vindt {len(_runs)} runs met `{C.as_code(_wanted)}`, met een gemiddelde validation loss van {C.number(_mean, 3, L.lang)}:\n\n" + C.runs_table(_runs, L.lang))
    else:
        _out = mo.md(f"{_who} vindt geen runs met `{C.as_code(_wanted)}`.")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_settings():
    mo.md(r"""
    ### Oefening 3. Welke instellingen gaan erin?
    """)
    return


@app.cell(hide_code=True)
def choice_settings_cell():
    choice_settings = L.choice(
        "all-settings",
        question="Je probeert een learning rate van 0,003 in plaats van 0,001, en verandert verder niets. Welke instellingen gaan in de regel van deze run?",
        options={
            "changed": "Alleen `lr`, de instelling die je veranderde",
            "right": "Allemaal, ook die je niet veranderde",
            "defaults": "Die afwijken van de standaardwaarden in je code",
        },
        correct="right",
        why={
            "changed": "Dan zegt de regel niet wat de andere instellingen waren. Verander je er later een, dan weet je niet meer welke runs welke hadden.",
            "defaults": "Standaardwaarden staan in je code, en code verandert: na een verandering betekent een regel die erop leunt iets anders. Zie stap 1 van het recept.",
            "right": "Een regel met elke instelling staat op zichzelf: hij zegt wat elke instelling was, ook als de standaardwaarden in je code intussen veranderd zijn, en je vindt hem terug op elke instelling.",
        },
    )
    choice_settings
    return (choice_settings,)


@app.cell(hide_code=True)
def text_exercise_version():
    mo.md(r"""
    ### Oefening 4. Twee runs, twee getallen
    """)
    return


@app.cell(hide_code=True)
def explain_version():
    explain_version_cell = L.explain(
        "code-version",
        question="Twee regels in je logboek hebben dezelfde instellingen en dezelfde seed. De ene eindigde op 2,3902, de andere op 2,4317, en hun codeversies zijn `4f9a1c2e` en `b81d07a3`. Wat concludeer je, en hoe had je deze twee regels gelezen zonder codeversie?",
        model_answer="**Wat het laat zien:** de code veranderde tussen de twee runs. Met dezelfde instellingen en dezelfde seed kan het verschil van 0,04 van die verandering komen, niet alleen van toeval; op dezelfde computer eindigt dezelfde code met dezelfde instellingen en seed meestal weer op hetzelfde getal. De hash zegt niet wat er veranderde: daarvoor heb je de oude code nodig, en daarom houden mensen hun code bij in Git en loggen ze de commit. **Zonder codeversie** zou het lijken alsof dezelfde run op twee getallen eindigde. Je zou 0,04 ruis kunnen noemen, of de verkeerde run vertrouwen, zonder te weten welke code welk getal gaf.",
        points=[
            "je zegt dat de code veranderde tussen de twee runs",
            "je zegt dat het verschil van die verandering kan komen, en niet alleen van toeval",
            "je zegt wat je zonder versie had gedacht: dezelfde run met twee resultaten, ruis of een fout die je niet kunt terugvinden",
        ],
    )
    explain_version_cell
    return (explain_version_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De runs in het logboek van de course komen uit [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) en [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl); het bestand lees je zoals in [Een tekstbestand lezen](/course/open/py-files?lang=nl). De volgende les, *Eerlijke vergelijkingen*, vraagt of de baseline in je logboek evenveel zorg kreeg als je eigen methode. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_settings, explain_version_cell, result_find, result_log):
    L.wrap_up(result_log, result_find, choice_settings, explain_version_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
