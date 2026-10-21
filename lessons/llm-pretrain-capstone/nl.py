# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-pretrain-capstone. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-pretrain-capstone", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl) en [Project: je eigen scaling law fitten](/course/open/llm-fit-scaling-law?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mfu():
    quick_mfu = L.choice(
        "quick-mfu",
        question="Een run traint 20.000 tokens per seconde, en elk token kost 120 miljoen FLOPs. De snelste vermenigvuldiging die deze GPU doet, haalt 12.000 miljard FLOP/s. Wat is de MFU van de run?",
        options={
            "tenth": "2%",
            "right": "20%",
            "half": "50%",
        },
        correct="right",
        why={
            "tenth": "20.000 keer 120 miljoen is 2.400 miljard FLOP/s, een vijfde van de 12.000 miljard: tel de nullen nog eens. Zie [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl).",
            "half": "De run heeft 20.000 keer 120 miljoen nodig, 2.400 miljard FLOP/s; dat is een vijfde van de piek. Zie [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl).",
            "right": "In dit project meet je de MFU van je eigen run, op je eigen computer.",
        },
    )
    quick_mfu
    return (quick_mfu,)


@app.cell(hide_code=True)
def quick_check_optimal():
    quick_optimal = L.choice(
        "quick-optimal",
        question="Voor een budget geeft je `compute_optimal` 1 miljoen parameters bij 20 tokens per parameter. Wat geeft hij voor vier keer zoveel compute?",
        options={
            "params": "4 miljoen parameters op dezelfde tokens",
            "tokens": "1 miljoen parameters op vier keer zoveel tokens",
            "right": "2 miljoen parameters op twee keer zoveel tokens",
        },
        correct="right",
        why={
            "params": "Bij een vast aantal tokens per parameter groeien ze allebei: de compute is 6 keer 20 keer de parameters in het kwadraat. Zie [Project: je eigen scaling law fitten](/course/open/llm-fit-scaling-law?lang=nl).",
            "tokens": "Bij een vast aantal tokens per parameter groeit het model ook: de compute is 6 keer 20 keer de parameters in het kwadraat. Zie [Project: je eigen scaling law fitten](/course/open/llm-fit-scaling-law?lang=nl).",
            "right": "De parameters en de tokens groeien elk met de wortel van de compute. In dit project is de compute wat je computer in 15 minuten doet.",
        },
    )
    quick_optimal
    return (quick_optimal,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Alles wat je voor je LLM bouwde, komt hier samen: je tokenizer en zijn shards, je ModernGPT, het recept van een open project, je scaling law, mixed precision en de MFU. Labs pretrainen zo op grote schaal: ze meten wat hun computers doen, kiezen een model dat in het budget past, trainen het, en meten de uitkomst. Eldan en Li (2023) lieten met TinyStories zien dat modellen van minder dan 10 miljoen parameters al vloeiende, samenhangende verhalen kunnen schrijven. In dit project pretrain je zelf zo'n model, in een budget van 15 minuten, en laat je het schrijven.

        **Een voorbeeld uit het dagelijks leven.** Je hebt een kwartier om te koken. Je kijkt eerst hoe heet je fornuis wordt, kiest dan een gerecht dat in die tijd klaar is, en aan het eind proef je.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           5, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_lower():
    _m = C.MEASURED
    _run = _m["run"]
    guess_lower = L.guess(
        "lower",
        question=f"De course plande de run van dit project op haar eigen CPU: een model met diepte {_run['depth']}, {C.number(_run['params'], 0, L.lang)} parameters, op {C.number(_run['tokens'] / 1e6, 1, L.lang)} miljoen tokens. In [Project: je eigen scaling law fitten](/course/open/llm-fit-scaling-law?lang=nl) haalde het model met diepte {C.S.BIG}, {C.number(C.S.MEASURED['big']['params'], 0, L.lang)} parameters op {C.number(C.S.MEASURED['big']['tokens'] / 1e6, 2, L.lang)} miljoen tokens, een validation loss van {C.number(C.S.MEASURED['big']['val'], 2, L.lang)} op dezelfde validatieverhalen. Wat haalt het kleinere model van deze run?",
        options={
            "higher": "Een hogere loss: het heeft minder parameters",
            "same": "Ongeveer dezelfde loss",
            "right": "Een lagere loss",
        },
        reveal=f"**Een lagere loss: {C.number(_run['val'], 2, L.lang)} tegen {C.number(C.S.MEASURED['big']['val'], 2, L.lang)}.** Het model is kleiner, maar het ziet {C.number(_run['tokens'] / C.S.MEASURED['big']['tokens'], 0, L.lang)} keer zoveel tokens: {C.number(_run['tokens'] / _run['params'], 0, L.lang)} per parameter in plaats van {C.number(C.S.MEASURED['big']['tokens'] / C.S.MEASURED['big']['params'], 1, L.lang)}. Bij een budget kan een kleiner model op meer data een groter model op te weinig data verslaan.",
    )
    guess_lower
    return (guess_lower,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een tijdsbudget is een compute budget.** Voor de run meet een **proefmeting** deze computer: een paar stappen van het model met diepte 6, die zijn tokens per seconde geven. Keer de 6 FLOPs per parameter en token die Chinchilla rekent, $6N$, zijn dat de FLOP/s waarop een plan kan rekenen; en keer de 15 minuten van de run is dat de compute $C$.

        **De grootte: compute-optimal, en dan het dichtstbijzijnde model.** Je `compute_optimal` verdeelt $C$ over parameters en tokens bij 20 tokens per parameter. De familie uit [Project: je eigen scaling law fitten](/course/open/llm-fit-scaling-law?lang=nl) heeft een model voor elke diepte, dus de run neemt de diepte waarvan de parameters $N$ het dichtst bij komen, en besteedt er het hele budget aan: $C / 6N$ tokens.

        **Eén stap van de run.** De learning rate volgt het schema uit [Een open trainingsrecept lezen](/course/open/llm-open-recipes?lang=nl); de forward pass en de loss draaien onder autocast in BF16, zoals in [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl), maar alleen waar dat sneller is: de proefmeting meet allebei; en de gradients worden geclipt op een norm van 1, zoals in [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl).

        **De run zelf.** Na elk tiende van zijn stappen meet de run zijn loss op 64 vensters van de validatieverhalen uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), en bewaart hij zijn toestand, zodat een run die stopt verder kan. Aan het eind meet hij de loss op alle vensters, en zijn MFU, zoals in [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl).

        ### Nieuwe woorden

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **tijdsbudget** | de tijd die een run mag duren; keer de FLOP/s van de computer een compute budget | `flops_per_second * seconds` |
        | **proefmeting** | een paar stappen die meten wat een computer doet, voordat de run gepland wordt | 8 getimede aanroepen van `pretrain_step` |
        """),
        mo.md(r"""
        Het plaatje: de validation loss van de run van de course op haar eigen CPU, na elk tiende van zijn stappen, tegen de tokens die hij had gezien.
        """),
        mo.Html(C.curve_picture([(_m["run"]["history"], False)], C.S.MEASURED["big"]["val"],
                                {"title": "Validation loss tijdens de run", "x": "geziene tokens", "y": "validation loss per token",
                                 "reference": f"diepte {C.S.BIG}, scaling law", "lang": L.lang,
                                 "alt": f"De validation loss daalt van {C.number(_m['run']['history'][0]['val'], 2, L.lang)} na {C.number(_m['run']['history'][0]['tokens'] / 1e6, 1, L.lang)} miljoen tokens naar {C.number(_m['run']['history'][-1]['val'], 2, L.lang)} na {C.number(_m['run']['tokens'] / 1e6, 1, L.lang)} miljoen, onder de {C.number(C.S.MEASURED['big']['val'], 2, L.lang)} van diepte {C.S.BIG} van de scaling law."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een model te pretrainen in een tijdsbudget:

    1. Maak van de verhalen shards; een snelle computer ziet ze meer dan eens.
    2. Meet deze computer met een proefmeting, in FP32 en in BF16, en neem de snelste.
    3. Plan: de compute van het budget, compute-optimal verdeeld, en het dichtstbijzijnde model.
    4. Train: elke stap op de learning rate van het schema, met geclipte gradients; valideer af en toe, en bewaar de toestand.
    5. Meet de uitkomst, de loss, de tokens per seconde en de MFU, en laat het model schrijven.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 in getallen: het plan dat de course op haar eigen CPU maakte, uit de FLOP/s die haar proefmeting mat.
    """)
    return


@app.cell
def example_plan():
    _flops = 7.38e10    # FLOP/s van de proefmeting van de course op haar CPU, 2 threads: tokens per seconde keer 6N
    _compute = _flops * 15 * 60
    _params = (_compute / (6 * 20)) ** 0.5    # compute-optimal bij 20 tokens per parameter
    _sizes = {2: 260649, 3: 575337, 4: 1110633, 5: 1940457, 6: 3141801}
    for _depth in _sizes:
        print(_depth, _sizes[_depth], round(abs(_sizes[_depth] - _params)))
    _best = 3    # de diepte met het kleinste verschil
    print(round(_compute / 1e12, 1), round(_params), round(_compute / (6 * _sizes[_best])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het budget is ongeveer $66 \cdot 10^{12}$ FLOPs, en compute-optimal is dat ongeveer 744.000 parameters. Diepte 3, met 575.337 parameters, komt het dichtst bij: 169.000 eronder, tegen 367.000 erboven voor diepte 4. Als je het hele budget aan diepte 3 besteedt, zijn dat ongeveer 19,2 miljoen tokens, zo'n 33 per parameter: het model is iets kleiner dan compute-optimal, dus het krijgt iets meer data.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het plan

        Maak `plan_run(flops_per_second, seconds, sizes, tokens_per_param)` af: stap 3. `sizes` geeft de parameters van elke diepte van de familie, zoals `{2: 260649, 3: 575337}`. Reken de compute van het budget en zijn compute-optimal parameters uit met je eigen `compute_optimal`, zoek de diepte waarvan de parameters het dichtst bij komen, en geef die diepte terug met de tokens die de hele compute eraan besteden, als heel getal. De check roept `plan_run(8e12, 900, {6: 3141801, 7: 4782537, 8: 6938601, 9: 9691305, 10: 13100457}, 20)` aan, voor een computer die honderd keer zo snel is als de CPU van de course, en andere.
        """),
        L.task("Laat `plan_run` de diepte en de tokens geven van een compute-optimal run in het budget."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plan():
    box_plan = L.code_box("plan-run", lines=12)
    box_plan
    return (box_plan,)


@app.cell(hide_code=True)
def check_plan(box_plan):
    result_plan = L.check("plan-run", box_plan.value)
    result_plan
    return (result_plan,)


@app.cell(hide_code=True)
def hints_plan():
    L.hints(
        "plan-run",
        [
            "De compute is `flops_per_second * seconds`. `params, _ = compute_optimal(compute, tokens_per_param)` geeft de parameters; de `_` laat zijn tokens weg, die bij precies die grootte horen.",
            "Loop door `sizes` en houd de beste diepte tot nu toe bij, zoals je `best_floor` zijn beste bodem bijhield: een diepte is beter als `abs(sizes[depth] - params)` kleiner is.",
        ],
        why="Geen model van de familie is precies compute-optimal. Het dichtstbijzijnde, met het hele budget, komt dicht bij de laagste loss voor die compute. Labs plannen net zo: vanuit de compute die ze hebben, niet vanuit het model dat ze zouden willen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén stap van de run

        Maak `pretrain_step(model, opt, x, y, lr, device_type, mixed)` af: één stap van stap 4. Zet de learning rate `lr` in elke groep van de optimizer, draai de forward pass en de loss alleen onder autocast in BF16 als `mixed` True is, en clip de gradients tussen de backward pass en de stap op een norm van hooguit 1,0. De check draait je stap op een kleine layer met reeksen van 5 posities, zoals een GPT ze krijgt, en probeert onder meer of een enorme gradient geclipt wordt.
        """),
        L.task("Laat `pretrain_step` één stap zetten op de learning rate die hij krijgt, alleen in BF16 als dat gevraagd wordt, met geclipte gradients."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("pretrain-step", lines=16)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("pretrain-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "pretrain-step",
        [
            "`opt.param_groups` is een lijst van dicts, elk met een eigen `\"lr\"`. `torch.autocast(..., enabled=mixed)` zet autocast aan of uit: met `enabled=False` doet hij niets.",
            "Een GPT geeft logits met shape (batch, posities, vocabulary), en `F.cross_entropy` wil één rij per positie: `logits.reshape(-1, logits.shape[-1])` en `y.reshape(-1)`, zoals in de regel voor de loss uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl). Het clippen komt na `loss.backward()` en voor `opt.step()`, zoals in [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl).",
        ],
        why="Het schema werkt alleen als elke stap zijn learning rate gebruikt; clippen voorkomt dat één vreemde mini-batch het model laat ontsporen; en autocast loont alleen waar de hardware BF16 sneller vermenigvuldigt dan FP32.",
    )
    return


@app.cell(hide_code=True)
def text_demo_data():
    _d = C.MEASURED["data"]
    mo.vstack([
        mo.md(f"""
        ### Je run, stap 1: de verhalen

        De knop downloadt één keer de eerste 200 MB van de trainingsverhalen van TinyStories, {C.number(_d['stories'], 0, L.lang)} verhalen, en maakt er shards van met je `encode_fast` en `write_shards`, {C.number(C.CHUNK / 1e6, 0, L.lang)} miljoen tekens tegelijk, in `work/projects/llm/{C.FOLDER}/`. Op de computer van de course, met de merge rules van de course, gaf dat {C.number(_d['ids'] / 1e6, 1, L.lang)} miljoen tokens in ongeveer {round(_d['seconds'] / 60)} minuten. De validatieverhalen blijven die uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl): geen verhaal van de run zit ertussen.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def data_button_cell():
    data_button = mo.ui.run_button(label="Haal de verhalen op")
    data_button
    return (data_button,)


@app.cell(hide_code=True)
def demo_data(data_button):
    _note = None
    if data_button.value:
        def _progress(_p):
            if _p["phase"] == "download":
                mo.output.replace(mo.md(f"**Downloaden** · {C.number(_p['done'] / 1e6, 0, L.lang)} van {C.number(_p['total'] / 1e6, 0, L.lang)} MB"))
            else:
                mo.output.replace(mo.md(f"**Coderen** · {C.number(_p['stories'], 0, L.lang)} van {C.number(_p['total'], 0, L.lang)} verhalen · {C.number(_p['ids'] / 1e6, 1, L.lang)} miljoen tokens · {round(_p['seconds'])} s"))
        _r = C.prepare(_progress)
        if _r == "download":
            _note = "*Het downloaden lukte niet. Is er verbinding met internet? Druk nog eens op de knop om het opnieuw te proberen.*"
        elif _r == "run":
            _note = "*Het coderen stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*"
    _info = C.shards_info()
    if _info is None:
        _out = mo.md(_note or "*Nog geen shards van de run. Druk op de knop.*")
    else:
        _whose = "je eigen merge rules" if _info["merges"] == "yours" else "de merge rules van de course"
        _out = mo.md(f"Je shards staan klaar: **{C.number(_info['ids'] / 1e6, 1, L.lang)} miljoen tokens** uit {C.number(_info['stories'], 0, L.lang)} verhalen, gecodeerd met {_whose}.")
    _out
    return


@app.cell(hide_code=True)
def text_demo_measure():
    mo.md(f"""
    ### Je run, stap 2: meten en plannen

    De knop klokt {C.PROBE_STEPS} stappen van je `pretrain_step` op het model met diepte {C.PROBE}, in FP32 en in BF16, neemt de snelste, en plant met je `plan_run` een run van {C.MINUTES} minuten. Het plan wordt in je project bewaard: om verder te gaan met een run die stopte, druk je opnieuw op *Pretrain* zonder opnieuw te meten.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Meet deze computer")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(box_plan, box_step, measure_button, result_plan, result_step):
    _note = None
    _ready = result_plan.state == "correct" and result_step.state == "correct"
    if not _ready:   # no mo.stop: the course's run below shows meanwhile
        _note = "*Maak eerst oefening 1 en 2 af: de run plant met je `plan_run` en traint met je `pretrain_step`.*"
    elif measure_button.value:
        mo.output.replace(mo.md("**Meten** · in FP32 en in BF16 …"))
        _m = C.measure(run_code(box_step.value).names["pretrain_step"], lambda _p: None)
        if _m == "run" or C.save_plan(_m, run_code(box_plan.value).names["plan_run"]) == "run":
            _note = "*Het meten stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*"
    plan = C.saved_plan() if _ready else None
    if _note is not None:
        _out = mo.md(_note)
    elif plan is None:
        _out = mo.md("*Nog niet gemeten. Druk op de knop.*")
    else:
        _fast = "BF16" if plan["mixed"] else "FP32"
        _where = "de CPU" if plan["device"] == "cpu" else plan["name"]
        _other = plan["tps_bf16"] if not plan["mixed"] else plan["tps_fp32"]
        _out = mo.md(f"""Op {_where} trainde je stap **{C.number(plan['tps'], 0, L.lang)} tokens per seconde** in {_fast}{'' if _other is None else f', tegen {C.number(_other, 0, L.lang)} in het andere formaat'}: {C.number(plan['flops'] / 1e9, 1, L.lang)} miljard FLOP/s bij 6N. In {C.MINUTES} minuten is dat $\\text{{{C.number(plan['compute'] / 1e12, 0, L.lang)}}} \\cdot 10^{{12}}$ FLOPs. Je `plan_run` kiest diepte **{plan['depth']}**, {C.number(plan['params'], 0, L.lang)} parameters, op **{C.number(plan['tokens'] / 1e6, 1, L.lang)} miljoen tokens**: {C.number(plan['steps'], 0, L.lang)} stappen van {plan['batch']} vensters.""")
    _out
    return (plan,)


@app.cell(hide_code=True)
def train_button_cell(plan):
    train_button = mo.ui.run_button(label=f"Pretrain diepte {plan['depth']} op {C.number(plan['tokens'] / 1e6, 1, L.lang)} miljoen tokens" if plan else "Pretrain", disabled=plan is None)
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_step, plan, train_button):
    _labels = {"title": "Validation loss tijdens de run", "x": "geziene tokens", "y": "validation loss per token",
               "reference": f"diepte {C.S.BIG}, scaling law", "lang": L.lang, "alt": "De validation loss van je run tot nu toe."}
    trained = None
    _note = None
    if train_button.value and plan is None:
        _note = "*Meet eerst deze computer: de run volgt zijn plan.*"
    elif train_button.value:
        _seen = []

        def _progress(_p):
            if _p["phase"] == "eval":
                _seen.append(_p)
            _parts = [mo.md(f"**Stap {C.number(_p['step'], 0, L.lang)} van {C.number(_p['steps'], 0, L.lang)}** · {C.number(_p['seconds'] / 60, 1, L.lang)} minuten")]
            if _seen:
                _parts.append(mo.Html(C.curve_picture([(_seen, False)], C.S.MEASURED["big"]["val"], _labels)))
            mo.output.replace(mo.vstack(_parts))
        _r = C.run_plan(plan, run_code(box_step.value).names["pretrain_step"], _progress)
        if _r == "data":
            _note = "*Er zijn nog geen shards van de run: druk eerst op* Haal de verhalen op*.*"
        elif _r == "merges":
            _note = "*Je merge rules zijn veranderd sinds je de verhalen ophaalde: druk opnieuw op* Haal de verhalen op*.*"
        elif _r == "memory":
            _note = "*Het geheugen van de GPU is vol: dit model, met mini-batches van 64 vensters, past er niet naast wat hem verder gebruikt. Sluit andere programma's die de GPU gebruiken, en druk opnieuw op* Pretrain*; tot dan neemt de run van de course hieronder de plaats in.*"
        elif isinstance(_r, str):
            _note = "*De run stopt met een fout, of zijn loss is geen getal meer. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*"
        else:
            trained = _r
    mo.md(_note) if _note else mo.md("*Klaar: je model staat in `work/projects/llm/capstone.pt`.*") if trained else None
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _m = C.MEASURED
    _info = C.saved()
    _mine = _info is not None
    _run = _info if _mine else _m["run"]
    _whose = "Je model" if _mine else "Het model van de course"
    _stand = "" if _mine else " Tot je je eigen model pretraint, neemt de run van de course zijn plaats in."
    _on = "de CPU" if _run["device"] == "cpu" else _run["name"]
    _runs = [(_run["history"], False)] + ([(_m["run"]["history"], True)] if _mine else [])
    _plan = C.saved_plan() if _mine else None
    _probe = _plan["flops"] if _plan else _m["measure"]["flops"]
    _did = _run["tps"] * 6 * _run["params"]
    mo.vstack([
        mo.Html(C.curve_picture(_runs, C.S.MEASURED["big"]["val"],
                                {"title": "Validation loss tijdens de run", "x": "geziene tokens", "y": "validation loss per token",
                                 "reference": f"diepte {C.S.BIG}, scaling law", "lang": L.lang,
                                 "alt": f"De validation loss van {'je run, met die van de course gestreept' if _mine else 'de run van de course'}: aan het eind {C.number(_run['val'], 2, L.lang)} per token."})),
        mo.md(f"""{_whose} met diepte {_run['depth']}, {C.number(_run['params'], 0, L.lang)} parameters, trainde {C.number(_run['seconds'] / 60, 0, L.lang)} minuten lang op {_on}, op {C.number(_run['tokens'] / 1e6, 1, L.lang)} miljoen tokens: {C.number(_run['tps'], 0, L.lang)} tokens per seconde, een MFU van {C.number(100 * _run['mfu'], 0, L.lang)}%. Zijn validation loss is **{C.number(_run['val'], 3, L.lang)}** per token, tegen {C.number(C.S.MEASURED['big']['val'], 3, L.lang)} voor diepte {C.S.BIG} van de scaling law. Het plan rekende op {C.MINUTES} minuten: bij diepte {_run['depth']} deden de stappen {C.number(_did / 1e9, 1, L.lang)} miljard FLOP/s bij 6N, tegen {C.number(_probe / 1e9, 1, L.lang)} miljard voor de proefmeting met diepte {C.PROBE}, dus de tijd wijkt af van het plan.{_stand}"""),
        mo.md(f"Het schrijft, vanaf *{C.START}*:\n\n" + C.quote(trained["story"] if trained else _info["story"] if _mine else _m["story"])),
    ])
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(f"""
    ### De run, meting voor meting

    Elke keer dat de run zijn validation loss meet, laat hij het model ook schrijven: {C.SNAP} tokens vanaf *{C.START}*, telkens getrokken met dezelfde seed. Hier ga je door die momenten: de validation loss tijdens de run, met een gestreepte lijn op het moment dat je kiest, en wat het model toen schreef.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _info = trained if trained is not None else C.saved()
    _mine = _info is not None and bool(C.moments(_info["history"]))
    _run = _info if _mine else C.MEASURED["run"]
    _whose = "je run" if _mine else "de run van de course"
    if _mine:
        _note = ""
    elif _info is not None:
        _note = "*Je bewaarde run is gemaakt voordat de les bijhield wat het model schreef, dus hier neemt de run van de course zijn plaats in. Pretrain opnieuw om door je eigen run te gaan.*"
    else:
        _note = "*Tot je je eigen model pretraint, ga je hier door de run van de course.*"
    _labels = {"title": f"Validation loss tijdens {_whose}", "x": "geziene tokens", "y": "validation loss per token", "lang": L.lang,
               "alt": f"De validation loss van {_whose} tegen de geziene tokens, op een logschaal, met een gestreepte lijn op het gekozen moment.",
               "after": "Na {tokens} miljoen tokens", "val": "validation loss"}
    _ks = C.moments(_run["history"])   # numbered from 1 on the slider
    _block = L.explore("replay", "meting", lambda _v: C.replay_frame(_run["history"], _ks[int(_v) - 1], _labels), start=len(_ks), values=list(range(1, len(_ks) + 1)), role="",
                       task="Schrijft het model bij de eerste meting al zinnen? Wat gaat daar nog mis dat bij de laatste goed gaat?")
    mo.vstack([mo.md(_note), _block]) if _note else _block
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(f"""
    In de run van de course (je eigen model schrijft andere verhalen) schrijft het model al zinnen vanaf de eerste meting, na {C.number(C.MEASURED['run']['history'][0]['tokens'] / 1e6, 1, L.lang)} miljoen tokens: namen, *One day*, het begin van wat iemand zegt. Daarna daalt de validation loss steeds langzamer, van {C.number(C.MEASURED['run']['history'][0]['val'], 2, L.lang)} naar {C.number(C.MEASURED['run']['history'][-1]['val'], 2, L.lang)}, en wat dat oplevert, zie je minder snel: of het verhaal klopt. Eerst wordt Tim Tom, die dan een meisje is; aan het eind heeft Tim een speelgoedbeer die Lily heet, en spelen ze samen. Daartussen schrijft het model meer dan eens *Tim had a toy boy named Tim*: elk moment is één trekking, en een iets ander model slaat ergens een andere weg in. De loss telt {C.number(C.QUICK * C.BLOCK, 0, L.lang)} tokens van de validatieverhalen tegelijk, in {C.QUICK} vensters; één verhaal van {C.SNAP} tokens laat één weg zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measure():
    mo.md(r"""
    ### Oefening 3. Waarom meten?
    """)
    return


@app.cell(hide_code=True)
def choice_measure_cell():
    _p = C.MEASURED["measure"]
    choice_measure = L.choice(
        "why-measure",
        question=f"De proefmeting van de CPU van de course trainde {C.number(_p['tps_fp32'], 0, L.lang)} tokens per seconde in FP32 en {C.number(_p['tps_bf16'], 0, L.lang)} in BF16. Waarom meet de run allebei, in plaats van altijd BF16 te nemen?",
        options={
            "right": "Of BF16 sneller is, hangt af van de hardware, en op deze CPU was het trager",
            "precision": "BF16 maakt de loss minder nauwkeurig, dus de run neemt het alleen als de loss gelijk blijft",
            "memory": "BF16 kost op een CPU meer geheugen, dus het model past er misschien niet in",
        },
        correct="right",
        why={
            "right": f"Mixed precision is een manier om sneller te gaan, en alleen een meting zegt of dat op jouw computer lukt. Op de CPU van de course duurde een stap in BF16 {C.number(_p['tps_fp32'] / _p['tps_bf16'], 1, L.lang)} keer zo lang.",
            "precision": "Onder autocast blijven de weights en de loss FP32, zoals in [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl), en de proefmeting vergelijkt nooit losses: hij klokt alleen de stappen, om het snelste formaat te nemen.",
            "memory": f"Een BF16-getal kost de helft van de bytes van een FP32-getal, op elk apparaat. De vraag is de snelheid: op de CPU van de course was BF16 {C.number(_p['tps_fp32'] / _p['tps_bf16'], 1, L.lang)} keer zo traag.",
        },
    )
    choice_measure
    return (choice_measure,)


@app.cell(hide_code=True)
def text_longer():
    _c = C.course_info()
    mo.vstack([
        mo.md(f"""
        ## Afronding

        **Een langere run van dezelfde code.** De course trainde haar eigen model met deze code {C.number(_c['hours'], 1, L.lang)} uur lang op haar CPU: diepte {_c['depth']}, {C.number(_c['params'], 0, L.lang)} parameters, compute-optimal op {C.number(_c['tokens'] / 1e6, 0, L.lang)} miljoen tokens aan verhalen voor jonge kinderen uit Cosmopedia, omdat de course geen model deelt dat op TinyStories getraind is. Zijn losses zijn niet te vergelijken met de jouwe: andere verhalen, andere merge rules. Het schrijft, vanaf *{C.START}*:
        """),
        mo.md(C.quote(_c["story"])),
        mo.md(r"""
        De lessen hierna gaan verder met je model, of met dat van de course als je er geen hebt: ze meten het, en leren het instructies te volgen.

        **Samenhang.** Je `compute_optimal` uit [Project: je eigen scaling law fitten](/course/open/llm-fit-scaling-law?lang=nl), die nu een echte run op maat maakt; de stap uit [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl), nu met het schema uit [Een open trainingsrecept lezen](/course/open/llm-open-recipes?lang=nl) en het clippen uit [Gradient clipping](/course/open/opt-gradient-clipping?lang=nl); de MFU uit [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl), nu van je eigen run; en de shards uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), nu van 200 MB aan verhalen. Je project `llm` heeft nu `plan_run.py` en `pretrain_step.py`, en na de run je model in `capstone.pt`, met zijn groottes en merge rules in `capstone.json`.
        """),
    ])
    return


@app.cell(hide_code=True)
def wrap_up(result_plan, result_step, choice_measure):
    L.wrap_up(result_plan, result_step, choice_measure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
