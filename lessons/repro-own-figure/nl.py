# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-figure. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-figure", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 4: sanity checks](/course/open/repro-own-sanity?lang=nl) en [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_learns():
    quick_learns = L.choice(
        "quick-learns",
        question="Je sanity check geeft `\"learns\": False` voor een model van tekst dat alleen het vorige token ziet. In zijn batch komt het token *de* drie keer voor *kat* en twee keer voor *hond*. Wat zegt dat over het model?",
        options={
            "frozen": "Een deel ervan is bevroren of losgekoppeld van de loss: een model dat werkt, leert elke batch uit het hoofd",
            "right": "Nog niets: geen model dat alleen het vorige token ziet, kan die batch uit het hoofd leren",
            "rate": "Zijn learning rate is te laag: een model van tekst heeft meer nodig dan `1e-3`",
        },
        correct="right",
        why={
            "frozen": "Na *de* geeft een model dat alleen *de* ziet één antwoord, en de batch wil drie keer *kat* en twee keer *hond*: zelfs het beste antwoord na *de*, 3/5 *kat* en 2/5 *hond*, laat op die vijf plekken een loss van ongeveer 0,67. Zie [Je reproductie 4](/course/open/repro-own-sanity?lang=nl).",
            "rate": "Met elke learning rate is het beste na *de* 3/5 *kat* en 2/5 *hond*: de batch houdt het tegen, niet de rate. Zie [Je reproductie 4](/course/open/repro-own-sanity?lang=nl).",
            "right": "Kies een batch waarin elke input zijn label onderscheidt, en de check zegt weer iets over het model.",
        },
    )
    quick_learns
    return (quick_learns,)


@app.cell(hide_code=True)
def quick_check_resume():
    quick_resume = L.choice(
        "quick-resume",
        question="Je `run_seeds` van Begeleide reproductie 4 stopt tijdens seed 2, als seed 0 en 1 in het logboek staan. Je drukt opnieuw op de knop. Wat draait er?",
        options={
            "right": "Seed 2, vanaf zijn eerste stap: een run die stopte, schreef geen regel",
            "where": "Seed 2, vanaf de stap waar hij stopte, want zijn log is bewaard",
            "all": "Seed 0, 1 en 2, zodat ze alle drie uit dezelfde druk op de knop komen",
        },
        correct="right",
        why={
            "where": "Een run voegt zijn regel pas toe als hij klaar is: van seed 2 is niets bewaard, dus hij begint opnieuw. Zie [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).",
            "all": "`run_seeds` slaat elke seed over waarvan de regel in het logboek staat: daardoor kost een stop één run. Zie [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).",
            "right": "Dat is wat een stop kost: de run die bezig was. De seeds die klaar zijn, blijven klaar.",
        },
    )
    quick_resume
    return (quick_resume,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Het hoofdresultaat is waar je reproductie om draait: de figuur of tabel van de paper, opnieuw gemaakt uit je eigen runs, en het oordeel van je plan erover. Het is het langste deel van je project: elke seed van je plan, beide settings, urenlang. Dus komt elke run in één logboek, een regel meteen erna, en gaat de runner verder waar hij stopte: een stop kost één run, niet de avond. En elke stip van je figuur en elk getal van je oordeel komt uit een regel van dat logboek, die iedereen opnieuw kan lezen.

        **Een voorbeeld uit het dagelijks leven.** Op een verkiezingsavond schrijft elk stembureau zijn telling op zodra die klaar is, en stuurt hem in. Een stroomstoring kost de telling van één stembureau, niet van het land, en iedereen kan de uitslagen opnieuw optellen.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_ladder():
    guess_ladder = L.guess(
        "ladder",
        question="Het budget van je plan is 12.800 stappen. Om ook te zien hoe het resultaat groeit met de training, draai je het budget en zijn halveringen: 100, 200, 400, enzovoort tot 12.800 stappen, elk een eigen run. Hoeveel stappen is dat samen, tegen de 12.800 van het budget alleen?",
        options={
            "eight": "Ongeveer acht keer zoveel: één run voor elk van de acht budgetten",
            "free": "Nauwelijks meer: naast de lange run kosten de korte bijna niets",
            "right": "Ongeveer twee keer zoveel: alle kortere runs samen zijn nog één budget",
        },
        reveal="**Ongeveer twee keer zoveel.** 100 + 200 + 400 + 800 + 1600 + 3200 + 6400 = 12.700: alle kortere runs samen zijn 100 stappen korter dan de langste. De hele ladder is 25.500 stappen, voor elke seed en setting. Elke run is zo lang als alle kortere runs samen, plus de kortste, dus kost een ladder minder dan twee keer zijn top.",
    )
    guess_ladder
    return (guess_ladder,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Eerst het budget.** De regel van je plan beoordeelt de runs van één budget: de stappen van je hoofdrun. Je legt het vast vóór de eerste run, op basis van hoe lang de paper traint en hoeveel uur je plan voor een seed heeft, en schrijft het in je kit. Wie het kiest na een blik op de resultaten, kiest het antwoord.

    **Een ladder van budgetten.** De paper laat zijn resultaat zien naarmate de training vordert. Je hoofdresultaat draait het budget en zeven halveringen ervan, elk een eigen run, van zijn eigen begin tot zijn eigen eind: acht punten van dezelfde figuur, voor elke seed en setting.

    **Eén regel per run.** De runner van je kit, `run_plan`, gaat door elke seed, elk budget en elke setting, en voegt elk resultaat meteen na zijn run aan het logboek toe, als één regel JSON: `{"setting": "claim", "seed": 0, "steps": 12800, "result": 12.5}`. De setting, de seed en de stappen samen wijzen een run aan: een run waarvan de regel er staat, draait niet opnieuw, dus kost een stop de run die bezig was.

    **Het oordeel uit het logboek.** `results_at` neemt de regels van één budget en geeft ze aan de regel van je plan zoals die ze wil: onder de namen van het plan, de kleinste seed eerst. De regel neemt de eerste resultaten van elke lijst als de seeds van het plan, en vergelijkt de twee settings seed voor seed.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | ladder van budgetten | het budget en zijn halveringen, elk een eigen run: de punten van de figuur naarmate de training vordert |

    Het plaatje: het hoofdresultaat van de course voor de dropout-paper, terwijl zijn logboek volloopt. 3 seeds, 8 budgetten van 100 tot 12.800 stappen, beide settings: 48 runs, in de volgorde van de runner. Elke stip is de testfout van één run, gevuld met dropout en open zonder; de lijnen gaan door de medianen van de budgetten waar elke seed tot nu toe zijn run heeft, doorgetrokken met dropout en gestreept zonder; de grote ring markeert de nieuwste run. De dunne lijn is het budget van het plan. De schuif begint als elke run klaar is, en laat het logboek zien zoals het onderweg was.
    """)
    return


@app.cell(hide_code=True)
def explore_log_cell():
    _names = C.P.course_plan("dropout", L.lang)["settings"]
    _labels = {"title": "Het hoofdresultaat van de course, run voor run", "x": "stappen van de run", "y": "testfout (%)", "budget": "budget van het plan",
               "alt": "De testfout van het experiment van de course voor de dropout-paper tegen de stappen van elke run, op een log-as van 100 tot 12.800, met en zonder dropout, drie seeds elk. Met elke run klaar: met dropout is de fout hoger tot 800 stappen, ongeveer 27,6% tegen 23,8% bij 100 stappen, en lager vanaf 1600 stappen, 12,5% tegen 13,6% in de mediaan bij 12.800 stappen."}
    _verdicts = {"too few runs": "te weinig runs", "supported": "gesteund", "not supported": "niet gesteund"}
    _plan = C.P.course_plan("dropout", "en")

    def _frame(n):
        _lines = C.course_log()[:int(n)]
        _next = C.stop_at(int(n))
        _v = _verdicts[C.course_verdict(_lines, _plan)]
        _stop = "Elke run is klaar." if _next is None else f"Een stop kost nu de run die bezig is: \"{_next['setting']}\", seed {_next['seed']}, {C.whole(_next['steps'], L.lang)} stappen."
        _pic = C.figure_picture(_lines, _names, C.COURSE_STEPS, _labels, last=_lines[-1] if _lines else None, ylim=(10, 30))
        return _pic, f"{int(n)} van {len(C.course_log())} runs in het logboek. Het oordeel van het plan bij {C.whole(C.COURSE_STEPS, L.lang)} stappen: {_v}. {_stop}"

    explore_log = L.explore("runs", "runs", _frame, start=48, low=0, high=48, step=1, role="",
                            task="Schuif terug vanaf het eind: op welke run wacht het oordeel? En wat kost een stop na 30 runs?")
    explore_log
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Als elke run klaar is, is de testfout bij 12.800 stappen 12,5% met dropout en 13,6% zonder, in de mediaan; seed voor seed loopt dropout 0,4, 0,7 en 1,2 procentpunt voor, een mediaan van 0,7, en de regel van het plan zegt gesteund. Het oordeel wacht op de allerlaatste run, `"other"` van seed 2 bij 12.800 stappen: tot dan heeft het plan te weinig runs, hoe mooi de figuur er ook uitziet. Een stop na 30 runs kost één run, de 31e, `"claim"` van seed 1 bij 12.800 stappen; de volgende druk op de knop gaat daar verder. De figuur vertelt het verhaal van de paper op de schaal van de course: met dropout begint de fout hoger, omdat het netwerk er langzamer mee leert, en vanaf 1600 stappen is hij lager.
        """),
        L.watch_out("Elk punt van de ladder is het eind van een eigen run, met zijn schedule over zijn eigen stappen. De figuur van de paper volgt één lange run onderweg. Die twee kunnen verschillen: een korte run die rustig eindigt, kan het beter doen dan een lange run bij dezelfde stap, die nog onderweg is. Zeg dat onder je figuur."),
        mo.md(r"""
        ### Recept

        Zo draai je het hoofdresultaat van je reproductie:

        1. Leg het budget vast op basis van de paper en de uren van je plan, en schrijf het in je kit.
        2. Draai het budget en zijn zeven halveringen, elke seed, beide settings, met de runner van je kit: een regel in het logboek meteen na elke run.
        3. Draai het na een stop opnieuw: de runner gaat verder met de runs die geen regel hebben.
        4. Teken de figuur uit het logboek: elke run een stip, elke setting een lijn door de medianen.
        5. Lees de runs bij het budget met `results_at`, en geef ze aan de regel van je plan.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een logboek waarin seed 1 vóór seed 0 draaide, zoals wanneer je seed 1 eerst draaide om het uit te proberen. Nieuw is `key=` in `sorted`: dat sorteert de regels op wat de functie voor elke regel geeft, hier zijn seed. Voer de cel uit.
    """)
    return


@app.cell
def example_order():
    # een logboek waarin "claim" van seed 1 eerst draaide, om het uit te proberen
    _rows = [("claim", 1, 17.5), ("claim", 0, 16.4),
             ("other", 0, 15.5), ("other", 1, 15.3)]
    _log = []
    for _s, _seed, _r in _rows:
        _log.append({"setting": _s, "seed": _seed,
                     "result": _r})
    _sorted = sorted(_log, key=lambda _l: _l["seed"])
    print("logboek:", [_l["seed"] for _l in _log])
    print("gesorteerd:", [_l["seed"] for _l in _sorted])
    for _s in ["claim", "other"]:
        print(_s, [_l["result"] for _l in _sorted
                   if _l["setting"] == _s])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de volgorde van het logboek zouden de lijsten `[17.5, 16.4]` en `[15.5, 15.3]` zijn: op plek 0 seed 1 van `"claim"` tegen seed 0 van `"other"`, twee runs die andere startwaarden en andere batches zagen. Gesorteerd op seed zijn ze `[16.4, 17.5]` en `[15.5, 15.3]`: plek 0 van allebei is seed 0, en plek 1 is seed 1. Dat heeft de regel nodig, want hij neemt de eerste resultaten als de seeds van het plan en vergelijkt ze plek voor plek.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_runner():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De runner

        Maak `run_plan(run, seeds, budgets, path, progress=None, settings=("claim", "other"))` af. Hij draait `run(setting, seed, steps)` van je experiment voor elke seed, elk budget en elk van `settings`, in de volgorde van de loops, en voegt meteen na elke run één regel JSON toe aan het logboek op `path`. Eerst leest hij welke runs klaar zijn: een tuple van de setting, de seed en de stappen van elke regel in het logboek. De check geeft hem een snelle stand-in voor je experiment, die bij elke run noteert hoeveel regels het logboek heeft: met twee seeds en twee budgetten in een nieuw logboek, met een logboek dat al drie runs heeft, met een plan dat in zijn tweede run stopt en opnieuw draait, en met alleen de setting `"ablation"`, zoals de volgende les hem draait.
        """),
        L.task("Maak `run_plan` af: eerst `done`, dan elke run die niet klaar is, met zijn regel meteen erna geschreven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_runner():
    box_runner = L.code_box("run-plan", lines=26)
    box_runner
    return (box_runner,)


@app.cell(hide_code=True)
def check_runner(box_runner):
    result_runner = L.check("run-plan", box_runner.value)
    result_runner
    return (result_runner,)


@app.cell(hide_code=True)
def hints_runner():
    L.hints(
        "run-plan",
        [
            "`done` maak je zoals `run_seeds` van Begeleide reproductie 4 het maakte, uit elke regel van `open(path)` met `json.loads`, maar met een tuple van drie voor elke regel in plaats van alleen zijn seed.",
            "In de loops: `if (setting, seed, steps) not in done:`, dan de regel als dictionary met zijn resultaat, geschreven met `open(path, \"a\")` als `json.dumps(line) + \"\\n\"`, en gegeven aan `progress` als die er is.",
        ],
        why="De tuple van drie wijst een run aan, dus kost een stop tussen twee settings, of tussen twee budgetten van een seed, alleen de run die bezig was; en een regel die meteen na zijn run geschreven is, blijft bewaard, wat er ook na komt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_results():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De resultaten bij het budget

        Maak `results_at(lines, steps, plan)` af. Hij geeft de regel van je plan de runs van één budget: een dictionary met de twee namen van de settings uit het plan, elk met de resultaten van die setting bij `steps` stappen, de kleinste seed eerst. `names` maakt van `"claim"` en `"other"` van de kit de namen van het plan. De check probeert een verzonnen logboek van twee seeds en twee budgetten, hetzelfde logboek met seed 1 eerst, en een ander plan.
        """),
        L.task("Maak `results_at` af: de regels in de volgorde van hun seeds, alleen die van `steps` stappen, elk resultaat onder de naam in het plan van zijn setting."),
    ])
    return


@app.cell(hide_code=True)
def exercise_results():
    box_results = L.code_box("results-at", lines=12)
    box_results
    return (box_results,)


@app.cell(hide_code=True)
def check_results(box_results):
    result_results = L.check("results-at", box_results.value)
    result_results
    return (result_results,)


@app.cell(hide_code=True)
def hints_results():
    L.hints(
        "results-at",
        [
            "Loop door de regels zoals het uitgewerkte voorbeeld ze sorteert: `sorted(lines, key=...)`, op hun seed.",
            "Voor elke regel van `steps` stappen is `names[line[\"setting\"]]` de naam van zijn setting in het plan: voeg `line[\"result\"]` toe aan die lijst van `results`.",
        ],
        why="Gesorteerd op seed zijn de eerste resultaten van elke lijst de seeds van het plan, en hebben beide lijsten op elke plek dezelfde seed, zoals de regel ze vergelijkt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Leg je budget vast

        Leg vóór de eerste run van je hoofdresultaat zijn budget vast: de stappen van zijn langste runs, op basis van hoe lang de paper traint en hoeveel uur je plan voor een seed heeft. De ladder van zeven halveringen komt erbij, voor minder dan twee keer de kosten. De uren van je plan zijn voor één seed, beide settings, dus de ladder van een seed duurt ongeveer 2 × 2 × het budget × de seconden van één stap: meet één stap met een korte run van je experiment. De check bewaart het budget in je kit, als `work/projects/own/budget.json`, met de datum.
        """),
        L.task("Vul het budget van je hoofdrun in, en in een zin waar het vandaan komt."),
    ])
    return


@app.cell(hide_code=True)
def budget_inputs():
    _old = C.saved_budget() or {}
    budget_steps = mo.ui.number(start=0, stop=10_000_000, step=1, value=_old.get("steps") or 0, label="Het budget van je hoofdrun, in stappen")
    budget_why = mo.ui.text_area(label="Waar het vandaan komt: hoe lang de paper traint, en de uren van je plan", full_width=True, rows=3, value=_old.get("why", ""))
    _plan = C.P.saved_plan()
    _head = (mo.md(f"*Je plan: {_plan['seeds']} seeds, {C.short(_plan['hours_per_seed'], L.lang)} uur voor een seed, beide settings.*") if _plan
             else mo.md("*Nog geen plan: schrijf het eerst in [Je reproductie 2](/course/open/repro-own-claim?lang=nl).*"))
    mo.vstack([_head, budget_steps, budget_why])
    return budget_steps, budget_why


@app.cell(hide_code=True)
def check_budget(budget_steps, budget_why):
    result_budget = L.check("budget", {"steps": budget_steps.value, "why": budget_why.value})
    result_budget
    return (result_budget,)


@app.cell(hide_code=True)
def hints_budget():
    L.hints(
        "budget",
        [
            "De opzet van de paper zegt hoe lang hij traint: in stappen, in epochs, of in keren door de data. Eén keer door N voorbeelden, in batches van B, is N / B stappen.",
            "Meet een run van 200 stappen van je experiment en deel door 200. De ladder van een seed duurt ongeveer 2 × 2 × het budget × dat: de uren van je plan voor een seed moeten dat ruim aankunnen.",
        ],
        why=f"Voor het experiment van de course voor de dropout-paper: **{C.whole(C.COURSE_STEPS, L.lang)} stappen.** " + C._B_WHY[L.lang],
    )
    return


@app.cell(hide_code=True)
def text_main():
    mo.md(rf"""
    ### Wat je project nu kan

    De knop draait je hoofdresultaat met je kit: je `run_plan` met de `run` van je experiment, elke seed van je plan, de ladder van je budget, in je logboek `work/projects/own/runs.jsonl`. Je ziet elke run zodra zijn regel binnenkomt. Voor het experiment van de course voor de dropout-paper duren alle 48 runs ongeveer {C.COURSE_MINUTES} minuten op één thread van de computer van de course; je plan zegt hoeveel uur de jouwe mag duren. Sluit je de les halverwege, dan blijven de runs die klaar zijn in je logboek staan: de volgende druk op de knop gaat verder met de andere. Zolang hij draait, wacht de rest van deze les: je kunt verder lezen, of in een andere les werken.
    """)
    return


@app.cell(hide_code=True)
def button_main():
    run_main = mo.ui.run_button(label="Draai mijn hoofdresultaat")
    run_main
    return (run_main,)


@app.cell(hide_code=True)
def demo_run(run_main, result_runner, result_budget):
    import time as _time

    result_runner, result_budget  # opnieuw uitvoeren als een oefening een deel van de kit bewaart
    _plan = C.P.saved_plan()
    _names = C.P._settings(_plan) if _plan else ("claim", "other")
    _labels = {"title": "Je hoofdresultaat, run voor run", "x": "stappen van de run", "y": (_plan or {}).get("metric") or "je metric", "budget": "je budget", "alt": "Je runs tot nu toe."}
    main_lines = None
    if not run_main.value:
        _out = mo.md("*Druk op de knop om je hoofdresultaat met je kit te draaien.*")
    else:
        _began, _done = _time.perf_counter(), []
        _budget = C.saved_budget()

        def _progress(_line):
            _done.append(_line)
            _steps = sum(_l["steps"] for _l in _done)
            _todo = 2 * int(_plan["seeds"]) * sum(C.ladder(_budget["steps"])) - sum(_l["steps"] for _l in (C.my_log() or [])) if _budget else 0
            _spent = _time.perf_counter() - _began
            _left = (f", nog ongeveer {round(_spent / _steps * max(_todo, 0) / 60)} minuten" if _spent / _steps * _todo >= 90 else ", nog minder dan twee minuten") if _steps and _todo > 0 else ""
            _all = C.my_log() or []
            mo.output.replace(mo.vstack([
                mo.md(f"**{len(_all)} runs in je logboek** · de nieuwste: `\"{_line['setting']}\"`, seed {_line['seed']}, {C.whole(_line['steps'], L.lang)} stappen: {C.short(_line['result'], L.lang)} · {C.short(round(_spent / 60, 1), L.lang)} minuten{_left}"),
                mo.Html(C.figure_picture(_all, _names, _budget["steps"], _labels, last=_line, role="ink")),
            ]))

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Je kit heeft eerst drie delen nodig, en een ervan is er nog niet: je runner, oefening 1 hier; je experiment, oefening 2 van [Je reproductie 3](/course/open/repro-own-build?lang=nl); en je plan, van [Je reproductie 2](/course/open/repro-own-claim?lang=nl).*")
        elif _r == "budget":
            _out = mo.md("*Leg eerst je budget vast in oefening 3: de knop draait zijn ladder.*")
        elif _r == "log":
            _out = mo.md("*Een regel van je logboek, `work/projects/own/runs.jsonl`, is geen run zoals `run_plan` hem schrijft: `\"setting\"` `\"claim\"` of `\"other\"`, een hele `\"seed\"` en `\"steps\"`, en een getal als `\"result\"`. Heb je je `own/runner.py` veranderd na zijn check, of het logboek met de hand?*")
        elif _r == "error":
            _out = mo.md("*Je `own/runner.py` of je `own/experiment.py` stopt met een fout. Heb je er een veranderd na zijn check? Check hem dan opnieuw. De runs die klaar waren, staan in je logboek.*")
        else:
            main_lines = _r
            _out = mo.md(f"*Klaar: {len(_r)} runs staan in je logboek, `work/projects/own/runs.jsonl`.*")
    _out
    return (main_lines,)


@app.cell(hide_code=True)
def demo_main(main_lines, result_results):
    main_lines, result_results  # het logboek opnieuw lezen na een run
    _lines, _plan, _budget = C.my_log(), C.P.saved_plan(), C.saved_budget()
    if not _lines or not _plan or not _budget:
        _out = mo.md("*Je hoofdresultaat verschijnt hier zodra zijn runs in je logboek staan.*")
    else:
        _names = C.P._settings(_plan)
        _labels = {"title": "Je hoofdresultaat", "x": "stappen van de run", "y": _plan.get("metric") or "je metric", "budget": "je budget",
                   "alt": f"Je runs: {len(_lines)} in je logboek, tegen de stappen van elke run, op een log-as."}
        _verdicts = {"too few runs": "te weinig runs", "supported": "gesteund", "not supported": "niet gesteund"}
        _said = C.my_verdict(_lines, _plan, _budget["steps"])
        _rule = (f"De regel van je plan, op de runs van {C.whole(_budget['steps'], L.lang)} stappen, gelezen met je `results_at`: **{_verdicts.get(_said, _said)}**." if _said
                 else "Je `own/results.py` of de regel in je `own/plan.py` stopt met een fout, of zit nog niet in je kit: check oefening 2, en de regel van Je reproductie 2.")
        _out = mo.vstack([
            mo.Html(C.figure_picture(_lines, _names, _budget["steps"], _labels, role="ink")),
            mo.md(_rule),
            mo.Html(C.main_table(_lines, _plan, _budget["steps"], {"seed": "seed", "gain": "winst"}, L.lang)),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 4. Wat je hoofdresultaat zegt

    Je hoofdresultaat staat in je logboek. Zeg wat het over de claim zegt, en waar je opzet verschilt van die van de paper.
    """)
    return


@app.cell(hide_code=True)
def explain_why_main():
    explain_why_main_cell = L.explain(
        "why-main",
        question="Wat zegt je hoofdresultaat over de claim? En waar kan je opzet verschillen van die van de paper, zodat je getallen verschillen van zijn figuur?",
        model_answer="**Mijn resultaat:** bij 12.800 stappen is de testfout met dropout lager dan zonder, voor alle drie de seeds, met 0,4, 0,7 en 1,2 procentpunt; de mediaan, 0,7, ligt boven de 0,5 van mijn plan, dus mijn regel zegt gesteund. Tot 800 stappen loopt dropout achter: het netwerk leert er langzamer mee. **Waar het verschilt:** mijn data is Fashion-MNIST, 6000 trainingsafbeeldingen, niet de 60.000 van MNIST; mijn netwerk heeft twee hidden layers van 512, kleiner dan dat van de paper; ik train met SGD en een cosine decay die ik zelf koos; en ik meet de testfout op 1000 afbeeldingen, dus één afbeelding is 0,1 procentpunt. Mijn fouten zijn veel hoger dan die van de paper, ongeveer 12% tegen ongeveer 1%: wat ik kan vergelijken is het effect, niet het getal.",
        points=[
            "je geeft het oordeel van je regel bij je budget, met de winst in de mediaan",
            "je noemt een verschil in de data of hoe die is voorbereid",
            "je noemt een verschil in het netwerk of de training",
            "je noemt een verschil in hoe het resultaat gemeten wordt",
        ],
    )
    explain_why_main_cell
    return (explain_why_main_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De runner van [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl), nu voor elk experiment van je kit; het contract en de delen van [Je reproductie 3](/course/open/repro-own-build?lang=nl); en de regel van [Je reproductie 2](/course/open/repro-own-claim?lang=nl), die nu echte runs beoordeelt. In de volgende les, *Je reproductie 6: de ablation*, test je van welk deel van de methode je resultaat afhangt, met dezelfde runner.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_runner, result_results, result_budget, explain_why_main_cell):
    L.wrap_up(result_runner, result_results, result_budget, explain_why_main_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
