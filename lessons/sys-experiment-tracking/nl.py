# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-experiment-tracking. Route: course/route/lessen/systemen.yaml. Checks, het log van de course, de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-experiment-tracking", "nl")
    C = L.checks   # het log van de course met twaalf runs, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Data als tekst: JSON](/course/open/py-json?lang=nl) en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question="Een regel van een log bevat de tekst `{\"lr\": 0.5, \"done\": true}`. Waar maakt `json.loads` deze tekst van?",
        options={
            "text": "Dezelfde tekst, als string",
            "right": "Een dict met `True` als waarde bij `\"done\"`",
            "strings": "Een dict met als waarden de strings `\"0.5\"` en `\"true\"`",
            "error": "Een fout, want `true` is geen Python",
        },
        correct="right",
        why={
            "right": "`json.loads` zet JSON-tekst om in Python-waarden: het object wordt een dict, `0.5` een float en `true` de `True` van Python. Zo komt een log met runs ook terug: elke regel tekst wordt weer een dict.",
            "text": "`json.loads` leest de tekst en maakt er Python-waarden van. Wat wordt een JSON-object? Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
            "strings": "In JSON zijn een getal en `true` geen strings: strings hebben dubbele aanhalingstekens. Wat worden een getal en `true` in Python? Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
            "error": "`true` is JSON, geen Python, en `json.loads` weet dat. Wat wordt het? Zie [Data als tekst: JSON](/course/open/py-json?lang=nl).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def quick_check_epoch():
    quick_epoch = L.choice(
        "quick-epoch",
        question="Na elke epoch meet je de training loss en de validation loss. De training loss blijft dalen; de validation loss is na epoch 8 het laagst en stijgt daarna. Van welke epoch bewaar je het model?",
        options={
            "first": "Epoch 1: het overfit nog niet",
            "last": "De laatste epoch, met de laagste training loss",
            "right": "Epoch 8, met de laagste validation loss",
            "test": "De epoch die de meeste testcijfers goed leest",
        },
        correct="right",
        why={
            "right": "Bij overfitting is het beste model het model van de epoch met de laagste validation loss. Een log heeft dus de validation loss van elke epoch nodig, niet alleen van de laatste.",
            "first": "Na epoch 1 heeft het model nog nauwelijks geleerd: beide losses zijn nog hoog. Welke epoch deed het het best op cijfers waarop het model nooit trainde? Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "last": "Een dalende training loss met een stijgende validation loss is overfitting: het model leert de trainingscijfers uit het hoofd. Welke epoch deed het het best op de validation set? Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "test": "Kies je op de testcijfers, dan meten de testcijfers niet langer hoe goed het model is op cijfers die de keuze nooit zag. Welke set is er om te kiezen? Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
        },
    )
    quick_epoch
    return (quick_epoch,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In maart 2018 schreef de engineer Pete Warden een blogpost met de titel *The Machine Learning Reproducibility Crisis*. Modellen waren moeilijk opnieuw te trainen met wat de makers hadden opgeschreven: soms lukte het volgens de post zelfs de oorspronkelijke maker niet om hetzelfde model opnieuw te trainen met vergelijkbare uitkomsten. Bijhouden wat er veranderde en een model opnieuw opbouwen was volgens Warden nog middeleeuws. In juni 2018 bracht het bedrijf Databricks MLflow uit, een open-source tool voor precies dat probleem. De aankondiging zei het eenvoudig: experimenten bijhouden is lastig.

    Je bent het probleem in deze course al tegengekomen. Een training-run in een cel laat de losses zien, en voer je de cel opnieuw uit met een andere learning rate, dan zijn de oude getallen weg. Welke learning rate gaf na een stuk of twaalf runs de laagste validation loss, en na welke epoch? Deze les bewaart elke run in een log, zodat je dat volgende week nog weet.

    **Een voorbeeld uit het dagelijks leven.** Een thuisbakker verandert de ene week de bloem en de volgende week de temperatuur van de oven. Zonder aantekeningen is een goed brood geluk dat je niet kunt herhalen. Met een schrift waarin per keer staat wat erin ging, hoe heet de oven was en hoe het brood werd, zoek je het beste brood op en bak je het opnieuw.
    """)
    return


@app.cell(hide_code=True)
def predict_last():
    guess_last = L.guess(
        "last-epoch",
        question="De course trainde het cijfer-MLP 12 keer op 1.000 trainingscijfers: drie learning rates, twee groottes van de hidden layer en twee seeds, elke keer 20 epochs, met na elke epoch de validation loss. Bij hoeveel van de 12 runs is de validation loss na de laatste epoch het laagst?",
        options={
            "all": "Alle 12: langer trainen helpt altijd",
            "most": "Ongeveer 9 van de 12",
            "right": "2 van de 12",
            "none": "Geen van de 12",
        },
        reveal="**2 van de 12.** Alleen de twee runs met learning rate 0,05 en seed 1 halen in epoch 20 de laagste validation loss. De twee runs met learning rate 0,05 en seed 0 zijn het laagst na epoch 19, en na epoch 20 hooguit 0,001 hoger: ook deze runs leren nog. Bij de 8 runs met learning rate 0,2 en 0,5 ligt het laagste punt na epoch 4 tot en met 10, en stijgt de validation loss daarna. Een log met alleen de laatste epoch zou bij deze 8 runs het beste model verbergen. De demo hieronder loopt alle 12 langs.",
    )
    guess_last
    return (guess_last,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een run en het log.** Een run is één keer trainen, met eigen instellingen: een learning rate, een grootte, een seed. **Experiment tracking** betekent dat elke run zelf opschrijft wat erin ging en wat eruit kwam: de instellingen voordat de run begint, de metingen tijdens de training en de uitkomst aan het eind. Later kun je alle runs vergelijken, de beste vinden en herhalen, omdat de instellingen in het log staan, ook de seed uit [Zelfde run, zelfde uitkomst: seeds en determinisme](/course/open/sys-reproducibility?lang=nl).

        **Params en metrics.** Tools voor tracking gebruiken twee woorden. Een **param** is een instelling van de run, één keer gelogd, voordat de training begint: de learning rate, het aantal neuronen, de seed, de versie van PyTorch. Een **metric** is een meting, steeds opnieuw gelogd: de training loss en de validation loss na elke epoch, elk met een **step**, het nummer van de epoch. De API-documentatie van MLflow zegt het in twee zinnen: een param log je één keer per run, een metric zo vaak als je wilt. Met de steps bevat het log de hele curve uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl), dus kun je later nog de epoch met de laagste validation loss vinden.

        **Eén run per regel.** Het log is een tekstbestand in het formaat **JSON Lines**: elke regel is één JSON-waarde, hier één run, zoals `json.dumps` uit [Data als tekst: JSON](/course/open/py-json?lang=nl) de run schrijft. Een nieuwe run voegt één regel toe aan het eind van het bestand en laat elke eerdere regel ongemoeid. Om het log te lezen, loop je de regels langs en maak je van elke regel weer een dict met `json.loads`. De eigen site van het formaat, jsonlines.org, noemt het formaat zeer geschikt voor logbestanden en stelt de extensie `.jsonl` voor.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **MLflow.** Je schrijft zelf een kleine tracker, in een paar regels Python. MLflow heeft dezelfde aanroepen, en veel meer: `mlflow.log_param("lr", 0.5)` logt een instelling, `mlflow.log_metric("val_loss", 0.48, step=2)` een meting, en `mlflow.search_runs()` geeft alle runs als een pandas DataFrame, met een kolom per param en metric, zoals `params.lr` en `metrics.val_loss`. Deze kolom bevat de laatst gelogde waarde: sorteer je erop, dan sorteer je op de laatste epoch. Sinds versie 3.7 bewaart MLflow de runs in een databasebestand, `mlflow.db`, en `mlflow server` laat de runs als tabel zien in je browser, op `http://localhost:5000`. MLflow is open source, onder de Apache 2.0-licentie; TensorBoard en Weights & Biases doen hetzelfde werk, elk op een eigen manier. Deze course installeert geen van deze tools: je tracker laat het idee zien, en de tools leveren er een server bij, met een tabel om te sorteren en grafieken om runs te vergelijken.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **experiment tracking** | | elke training-run legt de instellingen, de metrics en de uitkomst vast, zodat je runs later kunt vergelijken en herhalen | |
        | **MLflow** | em-el-flow | een open-source tool voor experiment tracking, met de aanroepen `log_param` en `log_metric` | `mlflow.log_metric("val_loss", 0.48, step=2)` |
        | **JSON Lines** | djee-son leins | een tekstformaat met één JSON-waarde per regel, meestal in een bestand dat eindigt op `.jsonl` | `json.dumps(run) + "\n"` |
        | **`open(path, "a")`** | | opent een bestand om tekst aan het eind toe te voegen, en maakt het bestand als het er nog niet is | `with open(path, "a") as f:` |
        | **`f.write(text)`** | | schrijft de tekst letterlijk in het open bestand, zonder regeleinde aan het eind | `f.write(json.dumps(run) + "\n")` |
        """),
        L.watch_out("`open(path, \"w\")` maakt een bestand eerst leeg en schrijft dan pas. Open je je log met `\"w\"` in plaats van `\"a\"`, dan is elke eerdere run weg. In het domein Onderzoek schrijft de les *Een experimentlogboek bijhouden* ook de versie van de code bij elke run op."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een run die zichzelf bijhoudt:

    1. Geef elke run een naam, en een dict met `"params"` en `"metrics"`.
    2. Log met `log_param`, voordat de training begint, elke instelling die de uitkomst kan veranderen: de learning rate, de grootte, de seed, de versie van PyTorch.
    3. Log na elke epoch elke meting met `log_metric`, met de epoch als step.
    4. Voeg na de training de run als één JSON-regel onderaan het log toe, met `open(path, "a")`.
    5. Om runs te vergelijken, lees je het log regel voor regel terug met `json.loads`, en zoek je per run de laagste validation loss, in welke epoch dan ook.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1, 2 en 4 van het recept, met twee epochs van de validation loss met de hand ingetypt, en dan stap 5: het log teruggelezen. `C.log_path("example.jsonl")` geeft een bestand in je werkmap, `work/runs/example.jsonl`. Voer de cel uit.
    """)
    return


@app.cell
def example_tracker():
    import json
    import torch

    # 1. een run: de naam, de instellingen (params) en de metingen (metrics)
    def start_run(name):
        return {"name": name, "params": {}, "metrics": {}}

    def log_param(run, key, value):
        run["params"][key] = value

    # 2. de instellingen, één keer, voordat de training begint
    run = start_run("first-try")
    log_param(run, "lr", 0.5)
    log_param(run, "seed", 0)
    log_param(run, "torch", torch.__version__)
    run["metrics"]["val_loss"] = [[1, 0.61], [2, 0.48]]   # twee epochs, voorlopig met de hand

    # 4. de run als één regel JSON, toegevoegd aan het eind van het log
    path = C.log_path("example.jsonl")
    with open(path, "a") as _f:
        _f.write(json.dumps(run) + "\n")

    # 5. het log teruggelezen: één run per regel
    with open(path) as _f:
        runs = [json.loads(line) for line in _f]
    print(len(runs))
    print(runs[-1])
    return json, log_param, path, run, runs, start_run, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het eerste getal telt de runs in het log. Voer de cel nog eens uit, en het getal gaat één omhoog: elke run voegt een regel toe, en de eerdere regels blijven. De laatste regel komt terug als dict, met de instellingen onder `"params"` en de twee metingen onder `"metrics"`, elk een lijst van twee: de step, dan de waarde. Ook de versie van PyTorch staat erin: `torch.__version__` geeft de versie als tekst, dus zegt het log welke PyTorch de run maakte. Er ontbreekt nog een functie die na elke epoch een meting voor je toevoegt: dat is je eerste oefening.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_metric():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je eigen log_metric

        `log_metric(run, key, value, step)` voegt één meting toe aan een run, zoals `log_param` een instelling toevoegt. Onder `run["metrics"]` heeft elke metric een lijst, zoals `[[1, 0.61], [2, 0.48]]` voor de validation loss na epoch 1 en 2: één item per aanroep, elk een lijst van twee, met eerst de `step` en dan de `value`. De `if`-regel en de regel eronder maken de lijst als `key` nog niet in `run["metrics"]` staat. Vervang de `...` door één regel die het item toevoegt aan het eind van de lijst van `key`.

        De check voert je functie uit met een paar regels erna, zoals de run uit het uitgewerkte voorbeeld met de twee epochs, en vergelijkt wat het log bevat. De afbeelding eronder vult een log voor een van de runs van de course.
        """),
        L.task("Voeg elke meting met de step toe aan het eind van de lijst van de metric."),
    ])
    return


@app.cell(hide_code=True)
def exercise_metric():
    box_metric = L.code_box("log-metric", lines=5)
    box_metric
    return (box_metric,)


@app.cell(hide_code=True)
def check_metric(box_metric):
    result_metric = L.check("log-metric", box_metric.value)
    result_metric
    return (result_metric,)


@app.cell(hide_code=True)
def hints_metric():
    L.hints(
        "log-metric",
        [
            "De lijst van `key` is `run[\"metrics\"][key]`: de twee regels erboven maken deze lijst als `key` nog ontbreekt. `append` voegt één item toe aan het eind van een lijst.",
            "Het item is zelf een lijst van twee: eerst de step, dan de waarde, zoals in `[2, 0.48]`. Welke twee namen van de functie bevatten deze getallen?",
            "`    run[\"metrics\"][key].append(___)`",
        ],
        why="`run[\"metrics\"][key].append([step, value])` voegt de meting toe aan het eind van de lijst van `key` en bewaart wat er al stond: na 20 epochs bevat de lijst 20 items, één per step.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Het log van de course

    De course trainde de twaalf runs uit de vraag hierboven, logde elke run met `log_param` en `log_metric` zoals in deze les, en voegde elke run als één regel toe aan een log. `C.RUNS` bevat de twaalf runs, teruggelezen met `json.loads`. De slider kiest een run. Bovenaan staan de training loss (doorgetrokken lijn) en de validation loss (lijn van streepjes) na elke epoch, op een logaritmische schaal, met de laagste validation loss gemarkeerd; daaronder de laagste validation loss van alle twaalf runs, met de gekozen run groter getekend; onderaan het begin van de regel in het log.
    """)
    return


@app.cell(hide_code=True)
def demo_log():
    L.explore("run", "run", lambda _v: C.frame(_v, L.lang), start=0, low=0, high=11, step=1,
              role="", task="Zoek de run met de laagste validation loss, en een run waarvan de validation loss nog daalt.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Gesorteerd op de laagste validation loss staat `lr0.5-h128-s0` bovenaan: 0,3965, na epoch 8. Gesorteerd op de validation loss na de laatste epoch zou een andere run bovenaan staan: `lr0.2-h128-s1`, met 0,4474. De beste run hangt af van wat je vergelijkt, en het log bevat elke epoch, dus kun je kiezen. Dezelfde instellingen met seed 1, `lr0.5-h128-s1`, halen 0,4183, na epoch 4: zoals [Zelfde run, zelfde uitkomst: seeds en determinisme](/course/open/sys-reproducibility?lang=nl) liet zien, zegt één run per instelling weinig. De vier runs met learning rate 0,05 leren na 20 epochs nog; een volgende run kan met deze instellingen langer trainen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De beste run in het log

        `best_run(runs, key)` krijgt een lijst runs, zoals teruggelezen uit een log, en de naam van een metric, en geeft de naam terug van de run met de laagste waarde van deze metric, bij welke step dan ook. `lowest(run, key)` is klaar: met `min` en een list comprehension over de `step` en `value` van elk item geeft de functie de laagste waarde van `key` in één run. In `best_run` begint `best` als de eerste van de `runs`, en de loop `for run in runs` gaat alle runs langs; `best = run` onthoudt een run die het beter doet dan `best` tot nu toe. Vervang de `...` door deze vergelijking.

        De check roept `best_run(C.RUNS, "val_loss")` aan, met de twaalf runs uit de demo, en daarna andere runs en andere metrics.
        """),
        L.task("Houd de run waarvan de laagste waarde van de metric lager is dan de laagste waarde van de beste run tot nu toe."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-run", lines=10)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-run", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-run",
        [
            "`lowest(run, key)` geeft de laagste waarde van `key` in `run`, over alle steps. Bij een loss is lager beter. Welke run moet `run` verslaan om de plek van de beste in te nemen?",
            "`run` moet de beste run tot nu toe verslaan, en `best` bevat deze run. Welke aanroep geeft de laagste waarde van `key` in deze run?",
            "`        if lowest(run, key) < ___:`",
        ],
        why="`lowest(run, key) < lowest(best, key)` is `True` als `run` lager komt dan alle eerdere runs; dan onthoudt `best = run` deze run. In het log van de course is dat `lr0.5-h128-s0`, met 0,3965 na epoch 8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lines():
    mo.md(r"""
    ### Oefening 3. Eén regel per run

    Je log voegt elke run als één regel toe aan het eind van `runs.jsonl`. Een collega bewaart alle runs in plaats daarvan in één JSON-lijst in `runs.json`: na elke run leest het programma de lijst, voegt de nieuwe run eraan toe, en schrijft de hele lijst opnieuw.
    """)
    return


@app.cell(hide_code=True)
def choice_lines_cell():
    choice_lines = L.choice(
        "lines",
        question="Wat voorkom je met één regel per run?",
        options={
            "size": "Een bestand dat meer ruimte op de schijf inneemt",
            "right": "Alle runs kwijtraken als het schrijven halverwege stopt",
            "read": "Een log dat Python niet kan teruglezen",
            "order": "Runs die in het bestand door elkaar raken, niet in de volgorde van draaien",
        },
        correct="right",
        why={
            "right": "Schrijf je de hele lijst opnieuw, dan herschrijf je ook elke eerdere run. Stopt het programma halverwege, bijvoorbeeld doordat de stroom uitvalt, dan kan het bestand midden in de lijst eindigen, en kan `json.load` er niets meer van lezen. Een nieuwe regel voegt alleen tekst toe aan het eind: een kapotte laatste regel kost alleen deze run, en omdat de eerdere regels heel blijven, kan je code de kapotte regel overslaan.",
            "size": "Beide bestanden bevatten dezelfde runs als dezelfde JSON-tekst, op een komma en twee haken na. Wat gebeurt er met de ene lijst als het schrijven halverwege stopt?",
            "read": "`json.load` leest een JSON-lijst net zo makkelijk als `json.loads` een regel leest. Wat gebeurt er met de lijst als het schrijven halverwege stopt?",
            "order": "Een lijst houdt de items in de volgorde van toevoegen, net als een bestand de regels. Wat is er nog meer anders als het programma stopt tijdens het schrijven?",
        },
    )
    choice_lines
    return (choice_lines,)


@app.cell(hide_code=True)
def text_exercise_repeat():
    mo.md(r"""
    ### Oefening 4. Genoeg om opnieuw te trainen?

    Je log van de run `run-7` bevat `lr`, `hidden` en `epochs` als params, en de training loss en de validation loss van elke epoch als metrics. Een collega wil dezelfde run opnieuw trainen, op een computer van hetzelfde type, en hetzelfde model krijgen.
    """)
    return


@app.cell(hide_code=True)
def choice_repeat_cell():
    choice_repeat = L.choice(
        "repeat",
        question="Wat moet het log nog meer bevatten?",
        options={
            "time": "Hoe lang elke epoch duurde",
            "loss": "De test loss van elke epoch, naast de andere twee",
            "name": "Wie de run trainde, en op welke dag",
            "right": "De seed en de versies van de packages",
        },
        correct="right",
        why={
            "right": "Met dezelfde seed, dezelfde versies en dezelfde hardware geeft een run op een CPU opnieuw hetzelfde model, zoals in [Zelfde run, zelfde uitkomst: seeds en determinisme](/course/open/sys-reproducibility?lang=nl). De losses zijn wat de run opleverde, niet wat de run nodig heeft. Een run logt dus elke instelling die de uitkomst kan veranderen, als param.",
            "time": "De duur van een epoch zegt iets over de computer, niet over wat de run nodig heeft om dezelfde uitkomst te geven. Van welke willekeurige getallen hangt de run af?",
            "loss": "Een test loss is iets wat de run oplevert, net als de andere twee losses. Wat moet de collega instellen voordat de training begint, om hetzelfde model te krijgen?",
            "name": "Een naam en een datum helpen je een run terug te vinden, toch hangt de training er niet van af. Waar hangt het model nog meer van af, naast `lr`, `hidden` en `epochs`?",
        },
    )
    choice_repeat
    return (choice_repeat,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Zelfde run, zelfde uitkomst: seeds en determinisme](/course/open/sys-reproducibility?lang=nl) liet zien wat een run nodig heeft om dezelfde uitkomst te geven; deze les schrijft dat bij elke run op. [Data als tekst: JSON](/course/open/py-json?lang=nl) gaf `json.dumps` en `json.loads`, hier één regel per run, en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) de reden om elke epoch te bewaren: het beste model kan in het midden liggen. In de volgende les, *Welk model is dit? Versies en een model registry*, krijgt het model van de beste run een eigen versie, zodat je altijd weet welk model je gebruikt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lines, choice_repeat, result_best, result_metric):
    L.wrap_up(result_metric, result_best, choice_lines, choice_repeat)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
