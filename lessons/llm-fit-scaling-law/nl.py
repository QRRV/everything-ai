# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-fit-scaling-law. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-fit-scaling-law", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Scaling laws](/course/open/llm-scaling-laws?lang=nl) en [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Drie runs geven de losses 3,2, 2,7 en 2,45, en je neemt een bodem van 2,0. Waar fit je een rechte lijn door, tegen de log van de groottes?",
        options={
            "loss": "De log van de loss: de bodem telt alleen bij grote modellen",
            "rest": "De loss min de bodem: 1,2, 0,7 en 0,45, zonder log",
            "right": "De log van de loss min de bodem: de logs van 1,2, 0,7 en 0,45",
        },
        correct="right",
        why={
            "loss": "Zonder de bodem buigt de kromme op een log-log plot, en voorspelt de lijn een te lage loss voor grotere runs. Zie [Scaling laws](/course/open/llm-scaling-laws?lang=nl).",
            "rest": "Een machtswet is alleen op een log-log plot een rechte lijn: beide assen hebben een log nodig. Zie [Scaling laws](/course/open/llm-scaling-laws?lang=nl).",
            "right": "Deze les gaat een stap verder: welke bodem maakt van je eigen runs de rechtste lijn?",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="Je ModernGPT heeft een breedte van d = 160 en 5 query heads. Met hoeveel getallen werkt elke query head?",
        options={
            "right": "32: de breedte wordt over de heads verdeeld",
            "all": "160: elke head ziet de hele breedte",
            "each": "800: vijf heads van elk 160",
        },
        correct="right",
        why={
            "all": "De queries van alle heads samen hebben d getallen: elke head krijgt er d / h. Zie [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).",
            "each": "De heads verdelen de breedte; ze krijgen er niet elk een kopie van. Zie [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).",
            "right": "In deze les heeft elk model heads van 32 getallen, hoe groot het ook is.",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een grote trainingsrun kan maanden en miljoenen kosten; je wilt weten wat hij oplevert voordat je begint. De makers van GPT-4 fitten een scaling law met een bodem, $L(C) = a C^b + c$ in de compute $C$, door modellen die op dezelfde manier getraind waren met hooguit 10.000 keer minder compute, en voorspelden daarmee de uiteindelijke loss van GPT-4 kort nadat de run begon, zonder iets van zijn resultaten te gebruiken. De voorspelling was zeer nauwkeurig (OpenAI, 2023). In dit project doe je hetzelfde op kleine schaal: train vier kleine versies van je ModernGPT, fit er een wet door, voorspel de loss van een grotere, en train die pas dan.

        **Een voorbeeld uit het dagelijks leven.** Voor een lange wandeling klok je jezelf over 1, 2 en 4 kilometer, en daaruit voorspel je de 10. Je weet dat vermoeidheid de laatste kilometers trager maakt: de voorspelling uit korte wandelingen is wat te snel.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           5, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_close():
    _m = C.MEASURED
    _small, _big = _m["small"], _m["big"]
    guess_close = L.guess(
        "close",
        question=f"De course trainde vier kleine ModernGPT's, van {C.number(_small[0]['params'] / 1e6, 2, L.lang)} tot {C.number(_small[-1]['params'] / 1e6, 2, L.lang)} miljoen parameters, op dezelfde verhalen, en fitte een wet met een bodem door hun losses. Die voorspelt de loss van een model van {C.number(_big['params'] / 1e6, 2, L.lang)} miljoen voordat het getraind is. Hoeveel zit de voorspelling er volgens jou naast?",
        options={
            "tenths": "Hij zit er een paar tienden naast",
            "right": "Hij zit er een paar honderdsten naast",
            "fails": "Niet dichtbij: het grotere model leert iets nieuws, dat geen wet van kleine modellen kan zien",
        },
        reveal=f"**Een paar honderdsten: de wet voorspelde {C.number(_m['predicted'], 2, L.lang)}, en het model haalde {C.number(_big['val'], 2, L.lang)}.** De voorspelling is {C.number(abs(_big['val'] - _m['predicted']), 3, L.lang)} te {'laag' if _m['predicted'] < _big['val'] else 'hoog'}: waarschijnlijk omdat een model dat dezelfde data ziet als de kleine, minder tokens per parameter krijgt, en minder leert dan de wet van de kleine verwacht.",
    )
    guess_close
    return (guess_close,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een familie van modellen.** Om groottes te vergelijken, mag alleen de grootte veranderen. Dit project maakt elk model met één knop, zijn diepte $L$: een breedte van $d = 32 L$, en één head per 32 getallen. Elke run traint op dezelfde vensters van dezelfde verhalen, één epoch over de training shards uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), met hetzelfde schema en dezelfde seed.

        **De wet, met een bodem die je niet kent.** [Scaling laws](/course/open/llm-scaling-laws?lang=nl) fitte $L(N) = E + a N^k$ met de bodem $E$ gegeven. Voor je eigen runs geeft niemand je $E$. Probeer dus bodems, en houd de bodem die de runs de rechtste lijn op een log-log plot geeft: de kleinste gemiddelde kwadratische afstand tussen de logs van $L - E$ en de lijn erdoor. Vier runs leggen een bodem maar ruw vast, dus hoe verder je voorspelt, hoe minder je erop kunt vertrouwen.

        **Compute-optimal.** Een run van $N$ parameters op $D$ tokens kost ongeveer $C = 6 N D$ FLOPs. Bij een gegeven compute kan een model te groot zijn voor zijn data, of te klein: Hoffmann en anderen (2022) vonden de laagste loss bij ongeveer 20 tokens per parameter. In dit project ziet elk model dezelfde tokens, dus hoe groter het model, hoe minder tokens per parameter het krijgt: de grote zitten ver van compute-optimal, en hun loss kan minder dalen dan de wet van de kleine verwacht.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **compute-optimal** | de verdeling van een compute budget over parameters en tokens die de laagste loss geeft; bij Chinchilla ongeveer 20 tokens per parameter | `compute_optimal(c, 20)` |
        | **familie van modellen** | modellen gemaakt met één recept, die alleen in grootte verschillen, zodat hun losses te vergelijken zijn | `width = 32 * depth` |
        """),
        mo.md(r"""
        Het beeld: de vier kleine runs van de course, de wet met de bodem die er het best bij past, zijn voorspelling voor het grotere model, en de echte loss van dat model.
        """),
        mo.Html(C.law_picture(_m["small"], _m["floor"], _m["big"], _m["predicted"],
                              {"title": "Validation loss tegen grootte", "x": "parameters", "y": "validation loss per token", "lang": L.lang,
                               "measured": "gemeten", "predicted": "voorspeld", "law": "de wet",
                               "alt": f"Vier gemeten losses, van {C.number(_m['small'][0]['val'], 2, L.lang)} bij {C.number(_m['small'][0]['params'] / 1e6, 2, L.lang)} miljoen parameters tot {C.number(_m['small'][-1]['val'], 2, L.lang)} bij {C.number(_m['small'][-1]['params'] / 1e6, 2, L.lang)} miljoen, met een gestippelde wet erdoor. Bij {C.number(_m['big']['params'] / 1e6, 2, L.lang)} miljoen voorspelt de wet {C.number(_m['predicted'], 2, L.lang)}, en het model haalde {C.number(_m['big']['val'], 2, L.lang)}."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de loss van een grote run te voorspellen uit kleine:

    1. Maak een familie van modellen met één knop.
    2. Train elke grootte op dezelfde data, met hetzelfde schema en dezelfde seed.
    3. Train vier groottes, elk ongeveer twee keer de vorige.
    4. Zoek de bodem die de losses de rechtste lijn op een log-log plot geeft.
    5. Fit de wet met die bodem, en voorspel de loss van het grote model voordat je het traint.
    6. Train het grote, en vergelijk.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 in getallen: de parameters, de tokens per parameter en de compute van de vier kleine runs van de course en de grote.
    """)
    return


@app.cell
def example_budget():
    _runs = [(2, 260649), (3, 575337), (4, 1110633), (5, 1940457), (6, 3141801)]
    _tokens = 6138 * 256   # één epoch: de 6.138 vensters van 256 tokens van de training shards
    for _depth, _params in _runs:
        # tokens per parameter, en de compute volgens de regel 6ND, in eenheden van 10 ** 12 FLOPs
        print(_depth, _params, round(_tokens / _params, 1), round(6 * _params * _tokens / 1e12, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke grootte is 1,6 tot 2,2 keer de vorige, en elke run ziet dezelfde 1.571.328 tokens. Dus dalen de tokens per parameter van 6,0 bij diepte 2 naar 0,5 bij diepte 6, ver onder de 20 van Chinchilla: deze runs hebben meer parameters dan hun data kan voeden. De compute groeit met de parameters, van ongeveer $2{,}5 \cdot 10^{12}$ FLOPs bij diepte 2 tot $30 \cdot 10^{12}$ bij diepte 6: de grote run kost ongeveer vier vijfde van de vier kleine samen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_floor():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De rechtste lijn

        Maak `best_floor(x, loss, floors)` af: stap 4. Fit voor elke bodem van `floors` de rechte lijn door `np.log10(x)` en `np.log10(np.array(loss) - floor)`, en meet hoe ver de logs ervan af liggen: het gemiddelde van hun gekwadrateerde verschillen. Geef de bodem met de kleinste fout terug. De check roept `best_floor([1e5, 3e5, 1e6, 3e6], [3.4, 2.9, 2.55, 2.35], [0.0, 0.5, 1.0, 1.5, 2.0, 2.2])` aan, en andere.
        """),
        L.task("Laat `best_floor` de bodem geven die de rechtste lijn maakt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_floor():
    box_floor = L.code_box("best-floor", lines=14)
    box_floor
    return (box_floor,)


@app.cell(hide_code=True)
def check_floor(box_floor):
    result_floor = L.check("best-floor", box_floor.value)
    result_floor
    return (result_floor,)


@app.cell(hide_code=True)
def hints_floor():
    L.hints(
        "best-floor",
        [
            "De fit is die uit [Scaling laws](/course/open/llm-scaling-laws?lang=nl): `k, b = np.polyfit(log_x, log_rest, 1)`. De waarden van de lijn zelf zijn `k * log_x + b`.",
            "Houd in de loop twee namen bij: de beste bodem tot nu toe en zijn fout. Vervang ze als een fout kleiner is, of als er nog geen is.",
        ],
        why="Een te lage bodem buigt de logs de ene kant op, een te hoge de andere kant; daartussen ligt de bodem waarvan de logs een rechte lijn zijn. Dat is de bodem waar de runs zelf naar wijzen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De voorspelling

        Maak `predict(x_new, x, loss, floor)` af: stap 5. Fit de lijn door de logs van `x` en van `loss - floor`, met helling $k$ en snijpunt $b$, en geef de bodem plus de machtswet bij `x_new` terug: $E + 10^b \cdot x_{\text{new}}^k$. De check roept `predict(1e7, [1e5, 3e5, 1e6, 3e6], [3.4, 2.9, 2.55, 2.35], 1.5)` aan, en andere.
        """),
        L.task("Laat `predict` de loss geven die de wet bij `x_new` voorspelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict", lines=10)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict",
        [
            "`np.polyfit` geeft eerst de helling, dan het snijpunt, dat de log van $a$ is.",
            "De wet geeft het deel van de loss boven de bodem; de voorspelling is de bodem plus dat deel.",
        ],
        why="De voorspelling rust op twee aannames: dat de wet doorgaat zoals hij deed, en dat de bodem klopt. Hoe verder `x_new` van de runs af ligt, hoe meer allebei tellen.",
    )
    return


@app.cell(hide_code=True)
def text_demo_small():
    _m = C.MEASURED
    mo.vstack([
        mo.md(f"""
        ### Je vier kleine modellen

        De knop hieronder traint je ModernGPT met diepte {', '.join(str(_d) for _d in C.SMALL[:-1])} en {C.SMALL[-1]}: elk één epoch over de training shards, mini-batches van {C.BATCH} vensters van {C.BLOCK} tokens, AdamW met de warmup, het plateau en de warmdown uit [Een open trainingsrecept lezen](/course/open/llm-open-recipes?lang=nl), seed 0, en de laatste shard om op te valideren. De runs worden bewaard in `work/projects/llm/{C.RUNS}`. Op de computer van de course, op 2 CPU-threads, duurden de vier samen ongeveer {round(sum(_r['seconds'] for _r in _m['small']) / 60)} minuten; een GPU is veel sneller. Tot je je eigen runs traint, nemen die van de course hun plaats in. Terwijl elk model traint, laat een grafiek de training loss van elke stap zien, met de kleinere modellen die al klaar zijn als gestippelde lijnen erbij; de grafiek van het laatste model blijft boven de tabel staan. Heb je alle vier getraind, dan laat de run van diepte {C.BIG} verderop ze ook gestippeld zien.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def small_button_cell():
    small_button = mo.ui.run_button(label="Train de vier kleine modellen")
    small_button
    return (small_button,)


@app.cell(hide_code=True)
def demo_small(small_button):
    _failed = False
    _kept = []   # the chart of the last depth, with the depths before it dashed: it stays above the table
    if small_button.value:
        _done = []   # the curves of the depths trained before, dashed beside the one that trains now
        _now = {}    # the live chart of the depth that trains now

        def _progress(_p):
            if _now.get("depth") != _p["depth"]:
                _now["depth"], _now["live"] = _p["depth"], L.live({"title": f"Diepte {_p['depth']}: de training loss van elke stap", "x": "stap", "y": "loss per token",
                                                                  "alt": f"De training loss bij elke stap van diepte {_p['depth']} tot nu toe, in grijs, en als dikkere lijn het gemiddelde over de laatste 10 stappen{'; gestippeld de kleinere modellen die al klaar zijn' if _done else ''}."}, _p["steps"], window=10)
                for _c in _done:
                    _now["live"].reference(_c, "", role="loss")
            _now["live"].extend(_p["fresh"], f"**Diepte {_p['depth']}** · stap {_p['step']} van {_p['steps']} · loss {C.number(_p['loss'], 2, L.lang)} per token · {round(_p['seconds'])} s")
        for _depth in C.SMALL:
            _r = C.train(_depth, _progress)
            if _r == "run":
                _failed = True
                break
            C.save_run(_r)
            _done.append(_r["curve"])
        if _now:
            _kept = [mo.Html(_now["live"].picture())]
    _saved = C.saved_runs()
    _mine = all(_d in _saved for _d in C.SMALL)
    small_runs = [_saved[_d] for _d in C.SMALL] if _mine else C.MEASURED["small"]
    _whose = "Je runs" if _mine else "De runs van de course, tot je je eigen traint"
    _rows = "\n".join(f"| {_r['depth']} | {C.number(_r['params'], 0, L.lang)} | {C.number(_r['tokens'] / _r['params'], 1, L.lang)} | {C.number(_r['val'], 3, L.lang)} |" for _r in small_runs)
    _note = [mo.md("*De run stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*")] if _failed else []
    mo.vstack(_note + _kept + [mo.md(f"{_whose}:\n\n| diepte | parameters | tokens per parameter | validation loss per token |\n|---|---|---|---|\n{_rows}")])
    return (small_runs,)


@app.cell(hide_code=True)
def demo_law(box_floor, box_predict, result_floor, result_predict, small_runs):
    mo.stop(not (result_floor.state == "correct" and result_predict.state == "correct"),
            mo.md("*Maak eerst oefening 1 en 2 af: dit deel fit de wet met je `best_floor` en je `predict`, en het grote model traint pas als je zijn loss voorspeld hebt.*"))
    _x = [_r["params"] for _r in small_runs]
    _loss = [_r["val"] for _r in small_runs]
    law_floor = run_code(box_floor.value).names["best_floor"](_x, _loss, C.floors_for(_loss))
    big_params = C.MEASURED["big"]["params"]
    law_prediction = float(run_code(box_predict.value).names["predict"](big_params, _x, _loss, law_floor))
    _plain = float(run_code(box_predict.value).names["predict"](big_params, _x, _loss, 0.0))
    mo.md(f"""Je `best_floor` kiest voor deze runs een bodem van **{C.number(law_floor, 2, L.lang)}**. Daarmee voorspelt je `predict` een validation loss van **{C.number(law_prediction, 3, L.lang)}** per token voor het model met diepte {C.BIG}, met {C.number(big_params, 0, L.lang)} parameters. Zonder bodem, een rechte lijn door de losses zelf, zou hij {C.number(_plain, 3, L.lang)} zeggen.""")
    return (law_floor, law_prediction)


@app.cell(hide_code=True)
def big_button_cell(law_prediction):
    big_button = mo.ui.run_button(label=f"Train het model met diepte {C.BIG}: je wet zegt {C.number(law_prediction, 3, L.lang)}")
    big_button
    return (big_button,)


@app.cell(hide_code=True)
def demo_big(big_button, law_floor, law_prediction, small_runs):
    _m = C.MEASURED
    _failed = False
    if big_button.value:
        _now = {}   # the live chart, made at the first step: then the run knows its steps

        def _progress(_p):
            if not _now:
                _runs = C.saved_runs()
                _curves = [_runs[_d]["curve"] for _d in C.SMALL if "curve" in _runs.get(_d, {})]   # none for the course's runs, or for older runs saved without a curve
                _now["live"] = L.live({"title": f"Diepte {_p['depth']}: de training loss van elke stap", "x": "stap", "y": "loss per token",
                                       "alt": f"De training loss bij elke stap van diepte {_p['depth']} tot nu toe, in grijs, en als dikkere lijn het gemiddelde over de laatste 10 stappen{'; gestippeld je kleine modellen' if _curves else ''}."}, _p["steps"], window=10)
                for _c in _curves:
                    _now["live"].reference(_c, "", role="loss")
            _now["live"].extend(_p["fresh"], f"**Diepte {_p['depth']}** · stap {_p['step']} van {_p['steps']} · loss {C.number(_p['loss'], 2, L.lang)} per token · {round(_p['seconds'])} s")
        _r = C.train(C.BIG, _progress)
        if _r == "run":
            _failed = True
        else:
            C.save_run(_r)
    _saved = C.saved_runs()
    _big = _saved.get(C.BIG) or _m["big"]
    _whose = "Je model" if C.BIG in _saved else "Het model van de course"
    _stand = "" if C.BIG in _saved else " Tot je je eigen model traint, neemt dat van de course zijn plaats in."
    _diff = _big["val"] - law_prediction
    _note = [mo.md("*De run stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*")] if _failed else []
    mo.vstack(_note + [
        mo.Html(C.law_picture(small_runs, law_floor, _big, law_prediction,
                              {"title": "Validation loss tegen grootte", "x": "parameters", "y": "validation loss per token", "lang": L.lang,
                               "measured": "gemeten", "predicted": "voorspeld", "law": "de wet",
                               "alt": f"De vier kleine runs, de wet erdoor, de voorspelling van {C.number(law_prediction, 2, L.lang)} voor diepte {C.BIG}, en zijn echte loss van {C.number(_big['val'], 2, L.lang)}."})),
        mo.md(f"""{_whose} met diepte {C.BIG} haalde een validation loss van **{C.number(_big['val'], 3, L.lang)}** per token, tegen de {C.number(law_prediction, 3, L.lang)} die je wet voorspelde: {C.number(abs(_diff), 3, L.lang)} {'hoger' if _diff > 0 else 'lager'} dan voorspeld. Het kreeg {C.number(_big['tokens'] / _big['params'], 1, L.lang)} tokens per parameter, het minst van alle vijf.{_stand}"""),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_optimal():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een compute-optimal verdeling

        Maak `compute_optimal(compute, tokens_per_param)` af. Een run kost `compute = 6 * params * tokens` FLOPs; bij de compute-optimal verdeling is `tokens = tokens_per_param * params`. Los de parameters op, en geef de parameters en de tokens terug. De check roept `compute_optimal(6e18, 20)` aan, en andere. Zodra hij goed is, wordt hij in je project bewaard: de capstone bepaalt er de grootte van zijn model mee.
        """),
        L.task("Laat `compute_optimal` een compute budget verdelen over parameters en tokens."),
    ])
    return


@app.cell(hide_code=True)
def exercise_optimal():
    box_optimal = L.code_box("compute-optimal", lines=8)
    box_optimal
    return (box_optimal,)


@app.cell(hide_code=True)
def check_optimal(box_optimal):
    result_optimal = L.check("compute-optimal", box_optimal.value)
    result_optimal
    return (result_optimal,)


@app.cell(hide_code=True)
def hints_optimal():
    L.hints(
        "compute-optimal",
        [
            "Zet `tokens = tokens_per_param * params` in `compute = 6 * params * tokens`: de compute is `6 * tokens_per_param * params ** 2`.",
            "Los dat op naar `params`: eerst delen, dan het kwadraat ongedaan maken.",
        ],
        why="Twee keer zoveel compute maakt het model niet twee keer zo groot: bij een vast aantal tokens per parameter groeien de parameters en de tokens elk met de wortel van de compute.",
    )
    return


@app.cell(hide_code=True)
def demo_optimal(box_optimal, result_optimal):
    mo.stop(result_optimal.state != "correct", mo.md("*Maak eerst oefening 3 af: dit deel gebruikt je `compute_optimal`.*"))
    _big = C.MEASURED["big"]
    _compute = 6 * _big["params"] * _big["tokens"]
    _p, _t = run_code(box_optimal.value).names["compute_optimal"](_compute, C.CHINCHILLA)
    mo.md(f"""De run met diepte {C.BIG} kost ongeveer ${C.number(_compute / 1e12, 0, L.lang)} \\cdot 10^{{12}}$ FLOPs. Je `compute_optimal` verdeelt dat budget, bij de {C.CHINCHILLA} tokens per parameter van Chinchilla, in een model van **{C.number(_p, 0, L.lang)} parameters** op **{C.number(_t / 1e6, 1, L.lang)} miljoen tokens**: een model van een zesde van de grootte, op meer verhalen dan de course heeft. Met weinig data is een groter model niet het beste gebruik van compute; de capstone krijgt meer data voor hij groeit.""")
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.md(r"""
    ### Oefening 4. Waarom te laag?
    """)
    return


@app.cell(hide_code=True)
def choice_low_cell():
    choice_low = L.choice(
        "why-low",
        question="De wet door de vier kleine runs voorspelde de loss van het grote model iets te laag. Welke reden past het best bij deze runs?",
        options={
            "seed": "Toeval: met een andere seed had het grote model precies op de wet gelegen",
            "right": "Te weinig data: het grote model kreeg de minste tokens per parameter",
            "floor": "De bodem was te laag: met een hogere klopt elke voorspelling",
        },
        correct="right",
        why={
            "seed": "Toeval verschuift een loss een beetje, en met één run per grootte kun je dat niet uitsluiten. Maar de grote run verschilt van de kleine op een punt waar de wet niets van weet: zijn tokens per parameter, het minst van allemaal.",
            "floor": "Een hogere bodem buigt de wet en verhoogt de voorspelling, maar de bodem is gekozen om bij de kleine runs te passen; hij kan niet weten wat zo weinig tokens per parameter met de grote doen.",
            "right": "Waarschijnlijk wel. Een wet van kleine runs is een gids voor runs die op dezelfde manier gemaakt zijn, en de grote run wijkt het meest af: dezelfde data voor veel meer parameters, 0,5 tokens per parameter, ver buiten het bereik waarop Chinchilla zijn formule fitte. Labs laten de data meegroeien met het model, zoals `compute_optimal` doet.",
        },
    )
    choice_low
    return (choice_low,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De wetten uit [Scaling laws](/course/open/llm-scaling-laws?lang=nl), nu gefit door je eigen runs, met een bodem die je ze zelf laat kiezen; je ModernGPT uit [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl), nu in vijf groottes; en de shards uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), nu de data die elke grootte ziet, met het schema uit [Een open trainingsrecept lezen](/course/open/llm-open-recipes?lang=nl). De capstone bepaalt met je `compute_optimal` hoe groot het kleine LLM wordt dat je pretraint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_floor, result_predict, result_optimal, choice_low):
    L.wrap_up(result_floor, result_predict, result_optimal, choice_low)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
