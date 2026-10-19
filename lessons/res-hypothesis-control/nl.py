# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-hypothesis-control. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-hypothesis-control", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl) en [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_wrong():
    quick_wrong = L.choice(
        "quick-wrong",
        question="Een hypothese: \"Met een warmup eindigt mijn GPT zijn run met een lagere validation loss dan zonder.\" Welke uitkomst laat zien dat ze niet klopt?",
        options={
            "longer": "De run met een warmup duurt een paar seconden langer",
            "right": "De run met een warmup eindigt op dezelfde of een hogere validation loss",
            "train": "Zonder warmup daalt de training loss sneller in de eerste epoch",
        },
        correct="right",
        why={
            "longer": "De hypothese gaat over de validation loss aan het eind, niet over tijd. Zie [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl).",
            "train": "De hypothese zegt wat er aan het eind van de run gebeurt, en met de validation loss. Een snellere start laat niet zien dat ze niet klopt. Zie [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl).",
            "right": "De hypothese verwacht aan het eind een lagere validation loss; dezelfde of een hogere laat zien dat ze niet klopt.",
        },
    )
    quick_wrong
    return (quick_wrong,)


@app.cell(hide_code=True)
def quick_check_val():
    quick_val = L.choice(
        "quick-val",
        question="Je vergelijkt twee runs van je GPT. Waarom kijk je naar hun validation loss, en niet naar hun training loss?",
        options={
            "lower": "De validation loss is altijd lager dan de training loss",
            "faster": "De validation loss kost minder tijd om uit te rekenen",
            "right": "De validation loss is gemeten op tekst waarop niet getraind is",
        },
        correct="right",
        why={
            "lower": "Hij is meestal hoger, niet lager: het model heeft die tekst niet gezien. Zie [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl).",
            "faster": "Snelheid is niet de reden. Zie [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl).",
            "right": "Een model kan zijn trainingstekst uit het hoofd leren; de validation loss laat zien hoe goed het het doet op nieuwe tekst.",
        },
    )
    quick_val
    return (quick_val,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Stel, je GPT eindigde zijn run op een validation loss van 1,66. Je wilt een lagere, dus je verhoogt de learning rate, maakt het model dieper en maakt de vensters langer, alles tegelijk, en de volgende run eindigt op 1,60. Welke verandering deed het? Misschien hielp de ene en schaadde de andere. Misschien was het geen van alle, en liep de run gewoon anders. Dat kun je niet weten. Om te weten wat een verandering doet, verander je één ding, en vergelijk je met een run zonder die verandering.

    **Een voorbeeld uit het dagelijks leven.** Je bakt brood met nieuw meel, een hetere oven en minder zout, en het smaakt beter. Volgende week wil je het weer bakken: welke van de drie houd je?
    """)
    return


@app.cell(hide_code=True)
def predict_lr():
    guess_lr = L.guess(
        "three-times",
        question="Een kleine GPT traint 6 epochs met een learning rate van 0,001. Een tweede run is in alles gelijk, maar met een learning rate van 0,003. Wat denk je dat zijn validation loss na de 6 epochs doet?",
        options={
            "right": "Hij is duidelijk lager",
            "same": "Hij is ongeveer gelijk",
            "higher": "Hij is duidelijk hoger: een te grote stap",
        },
        reveal="**Duidelijk lager.** In de runs van de course met 3 seeds eindigde de control op 2,39 tot 2,42, en de runs met 0,003 op 2,08 tot 2,10: met elke seed lager, met ongeveer 0,3. Voor deze kleine GPT, die maar 6 korte epochs traint, is 0,001 te voorzichtig: met grotere stappen leert hij meer in dezelfde tijd. Of 0,003 ook beter is voor je grotere GPT, zegt dit experiment niet: het toetste deze opzet.",
    )
    guess_lr
    return (guess_lr,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een experiment vergelijkt runs. De **control run** is je opzet zoals hij is, zonder de verandering die je wilt toetsen. Een **variant** is een kopie ervan waarin je precies één instelling verandert: de **experimentele variabele**, die op school bij biologie en natuurkunde de onafhankelijke variabele heet. Al het andere blijft gelijk: de data, de grootte van het model, het aantal epochs, de code. Wat je meet, hier de validation loss aan het eind, vergelijk je tussen de control en de variant. <!-- termen:ok -->

        Waarom één instelling? Verschilt de variant in twee, dan heeft een verschil in het resultaat twee mogelijke oorzaken, en kun je die niet uit elkaar houden. Een run met een hogere learning rate *en* een extra epoch die lager eindigt, zegt niets over de learning rate.

        Geef beide runs dezelfde **seed**. De seed bepaalt de beginwaarden van de weights en de volgorde van de mini-batches. Met dezelfde seed begint een variant met een andere learning rate of warmup met dezelfde weights als de control, en krijgt hij de mini-batches in dezelfde volgorde: de instelling is het enige verschil. Een verandering in het model zelf, zoals meer layers, gebruikt de willekeurige getallen anders, dus daar verandert ook de volgorde van de mini-batches. En één run per instelling blijft één worp met de dobbelsteen: een latere les herhaalt elke run met meer seeds.

        In code passen instellingen in een dictionary. `dict(control)` maakt een nieuwe dictionary met dezelfde paren; daarin verander je dan één paar. `variant = control` doet dat niet: dat geeft dezelfde dictionary een tweede naam, zodat een verandering in de variant ook de control verandert.
        """),
        mo.Html(C.settings_picture(C.CONTROL, {**C.CONTROL, "lr": 0.003},
                                   {"title": "Een control run en een variant", "setting": "instelling", "control": "control", "variant": "variant", "changed": "veranderd",
                                    "alt": "Een tabel met negen instellingen. De control run en de variant zijn in elke instelling gelijk behalve de learning rate, lr: 0,001 in de control, 0,003 in de variant, gemarkeerd als veranderd."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **control run** | de run van een experiment met je opzet zoals hij is, zonder de verandering die je toetst; elke andere run vergelijk je ermee |
        | **experimentele variabele** | de ene instelling die je tussen de runs van een experiment bewust verandert |
        | **variant** | een run die alleen in de experimentele variabele van de control run verschilt |
        """),
        L.watch_out("Een control run is geen baseline. Een baseline is een andere methode waarmee je vergelijkt, zoals in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl); een control run is je eigen opzet zonder de verandering. Een experiment heeft vaak allebei nodig."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te toetsen wat één verandering doet:

    1. Schrijf de hypothese op: welke instelling je verandert, waarin, wat je verwacht, en wat je meet.
    2. Schrijf de control run op: elke instelling van de run die je nu hebt.
    3. Maak de variant: een kopie van de control waarin die ene instelling veranderd is.
    4. Train ze allebei, met dezelfde seed, en vergelijk wat je meet.
    5. Controleer, voordat je iets concludeert, of de twee runs alleen in die ene instelling verschillen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    **1. De hypothese:** met een learning rate van 0,003 in plaats van 0,001 eindigt de kleine GPT van deze les zijn 6 epochs met een lagere validation loss. Stap 2, 3 en 5 in code; het commentaar zegt welke stap elk deel is. Stap 3 is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_variant():
    # 2. de control run: elke instelling van de run
    _control = {"lr": 0.001, "warmup": 100, "layers": 2, "d": 64, "heads": 4, "block": 64, "batch": 32, "epochs": 6, "seed": 0}
    # 3. de variant: een nieuwe dictionary met dezelfde paren, en één instelling veranderd
    _variant = _control  # maak af: dict(_control)
    _variant["lr"] = 0.003
    # 5. controle: in welke instellingen verschillen de twee runs?
    _differ = []
    for _name in _control:
        if _control[_name] != _variant[_name]:
            _differ.append(_name)
    print("control lr:", _control["lr"], " variant lr:", _variant["lr"])
    print("verschillen in:", _differ)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, geeft `_variant = _control` dezelfde dictionary een tweede naam, dus de regel erna verandert ook de control: allebei hebben ze een learning rate van 0,003, stap 5 vindt helemaal geen verschil, en je control run is weg. Afgemaakt maakt `dict(_control)` een nieuwe dictionary: de control houdt zijn 0,001, en de twee runs verschillen in `lr` en in niets anders. Dan kan stap 4, ze allebei trainen, laten zien wat de learning rate doet. De demo hieronder doet stap 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_change():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een variant met één verandering

        Maak `one_change(control, name, value)`: een nieuwe dictionary met elk paar van `control`, waarin de instelling `name` de waarde `value` heeft. `control` zelf moet blijven zoals hij was. Vervang de twee `...`. De check roept `one_change(control, "warmup", 0)` aan met de control van het uitgewerkte voorbeeld, en probeert andere instellingen en een andere dictionary.
        """),
        L.task("Laat `one_change(control, name, value)` een kopie van `control` teruggeven waarin alleen de instelling `name` veranderd is in `value`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_change():
    box_change = L.code_box("one-change", lines=7)
    box_change
    return (box_change,)


@app.cell(hide_code=True)
def check_change(box_change):
    result_change = L.check("one-change", box_change.value)
    result_change
    return (result_change,)


@app.cell(hide_code=True)
def hints_change():
    L.hints(
        "one-change",
        [
            "Twee stappen, zoals in stap 3 van het uitgewerkte voorbeeld: eerst een nieuwe dictionary met dezelfde paren als `control`, dan dat ene paar veranderd, met de namen `name` en `value`.",
            "`variant = dict(___)`, en dan `variant[___] = ___`.",
        ],
        why="```python\ndef one_change(control, name, value):\n    variant = dict(control)\n    variant[name] = value\n    return variant\n```\n\n`dict(control)` maakt een nieuwe dictionary met dezelfde paren, dus de variant veranderen laat `control` zoals hij was. `variant[name] = value` verandert die ene instelling.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Stap 4: draai het experiment

    Kies de instelling die verandert; de control run heeft een learning rate van 0,001, een warmup van 100 stappen en 2 layers. De knop traint de control run van het uitgewerkte voorbeeld, en daarna de variant die jouw `one_change` ervan maakt: twee runs van een kleine GPT van ongeveer 110.000 parameters (210.000 met 4 layers), met jouw `GPT`, `gpt_loaders` en `Trainer`, op het eerste kwart van Tiny Shakespeare, op de CPU. Elke run duurt ongeveer een halve tot een hele minuut, een variant met 4 layers ongeveer twee keer zo lang. Terwijl hij traint, wacht de rest van de les; de stopknop van de cel maakt er een eind aan.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    change_choice = mo.ui.dropdown(
        options={"learning rate 0,003": 0, "geen warmup": 1, "4 layers": 2},
        value="learning rate 0,003", label="Verander:")
    run_button = mo.ui.run_button(label="Draai het experiment")
    mo.vstack([change_choice, run_button], align="start")
    return change_choice, run_button


@app.cell(hide_code=True)
def demo_experiment(box_change, change_choice, result_change, run_button):
    _labels = {"title": "Validation loss", "x": "epoch", "y": "loss", "lang": L.lang, "control": "control", "variant": "variant"}
    _name, _value = C.CHANGES[change_choice.value]
    if not run_button.value:
        _out = mo.md("*Kies een verandering en druk op de knop.*")
    else:
        _fn, _yours = C.one_change_of(box_change.value if result_change.state == "correct" else None)
        _variant = _fn(dict(C.CONTROL), _name, _value)
        _differ = C.changed(C.CONTROL, _variant)
        if _differ != [_name]:
            _out = mo.md(f"*De variant verschilt van de control in {', '.join(f'`{_d}`' for _d in _differ) or 'niets'}, niet alleen in `{_name}`. Kijk oefening 1 nog eens na.*")
        else:
            _runs = []
            for _which, _settings in (("de control run", C.CONTROL), ("de variant", _variant)):
                _h = C.run(_settings, lambda _share, _w=_which: mo.output.replace(mo.md(f"*Bezig met {_w}: {round(100 * _share)}%*")))
                if isinstance(_h, str):
                    break
                _runs.append(_h)
            if len(_runs) < 2:
                _out = mo.md("*Je stopte het experiment voordat beide runs klaar waren. Druk op de knop om het opnieuw te starten.*" if _h == "stopped" else "*Een run stopt met een fout. Komen je modules in `work/projects/llm/` en `work/projects/nn/` nog door hun checks? Verwijder een bestand dat je veranderde, en controleer zijn oefening opnieuw.*")
            else:
                _c, _v = _runs[0][-1]["val_loss"], _runs[1][-1]["val_loss"]
                _ref = C.REFERENCE.get(f"{_name}={_value}")
                _course = "" if not _ref else f" In de runs van de course met seeds 0, 1 en 2 eindigde de control op {C.number(_ref['control'][0], 3, L.lang)} tot {C.number(_ref['control'][1], 3, L.lang)}, en deze variant met elke seed lager dan de control, met {C.number(_ref['lower'][0], _ref['lower'][2], L.lang)} tot {C.number(_ref['lower'][1], _ref['lower'][2], L.lang)}."
                _who = "" if _yours else " Oefening 1 klopt nog niet, dus de `one_change` van de course maakte de variant."
                _out = mo.vstack([
                    mo.Html(C.curves_picture(_runs[0], _runs[1], {**_labels, "alt": f"De validation loss na elke epoch: de control eindigt op {C.number(_c, 3, L.lang)}, de variant op {C.number(_v, 3, L.lang)}."})),
                    mo.md(f"Na {len(_runs[0])} epochs eindigde de control run op een validation loss van {C.number(_c, 3, L.lang)}, en de variant, met `{_name} = {_value}`, op {C.number(_v, 3, L.lang)}.{_course}{_who}"),
                ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_confound():
    mo.md(r"""
    ### Oefening 2. Wat veranderde er?
    """)
    return


@app.cell(hide_code=True)
def choice_confound_cell():
    choice_confound = L.choice(
        "confound",
        question="Je control run traint 6 epochs met een learning rate van 0,001 en seed 0. Een variant traint 8 epochs met een learning rate van 0,003 en seed 0, en eindigt met een lagere validation loss. Wat kun je concluderen?",
        options={
            "lr": "De hogere learning rate helpt: dat wilde je toetsen",
            "right": "Niet welke verandering hielp: de variant veranderde twee dingen",
            "epochs": "Langer trainen helpt: de variant had twee epochs meer",
        },
        correct="right",
        why={
            "lr": "De variant trainde ook twee epochs langer, dus die extra epochs kunnen het verschil net zo goed verklaren. Zie *Het idee*.",
            "epochs": "De variant had ook een hogere learning rate, dus die kan het verschil net zo goed verklaren. Zie *Het idee*.",
            "right": "Er veranderden twee instellingen, dus het verschil heeft twee mogelijke oorzaken. Maak twee varianten, elk met één verandering, en vergelijk elk met de control.",
        },
    )
    choice_confound
    return (choice_confound,)


@app.cell(hide_code=True)
def text_exercise_design():
    mo.md(r"""
    ### Oefening 3. Ontwerp een experiment

    Je hypothese: een warmup van 100 stappen geeft de kleine GPT van deze les na 6 epochs een lagere validation loss dan helemaal geen warmup.
    """)
    return


@app.cell(hide_code=True)
def explain_design():
    explain_design_cell = L.explain(
        "design",
        question="Ontwerp het experiment met het recept: schrijf de control run op, de variant, wat gelijk blijft, wat je meet, en welke uitkomst zou laten zien dat de hypothese niet klopt.",
        model_answer="**Control run:** de instellingen van het uitgewerkte voorbeeld, met `warmup` 100. **Variant:** dezelfde instellingen met `warmup` 0, gemaakt met `one_change`. **Gelijk:** de data, de learning rate, de grootte van de GPT, de batch size, de 6 epochs en de seed. **Gemeten:** de validation loss na epoch 6. **Klopt niet als:** de variant zonder warmup op dezelfde of een lagere validation loss eindigt dan de control.\n\nDe course deed dit experiment met seeds 0, 1 en 2: zonder warmup eindigde de run elke keer iets lager, met 0,004 tot 0,008. Voor deze kleine GPT, met deze learning rate, klopt de hypothese dus niet; ook dat is een uitkomst. Of zo'n klein verschil meer is dan toeval, laat een latere les zien.",
        points=[
            "de control heeft een warmup van 100 stappen en de variant geen, en verder verschilt er niets",
            "beide runs gebruiken dezelfde seed en dezelfde 6 epochs",
            "je zegt wat je meet, en welke uitkomst laat zien dat de hypothese niet klopt",
        ],
    )
    explain_design_cell
    return (explain_design_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl) zei dat een experiment bewust iets verandert en de rest gelijk houdt; nu doe je dat met de GPT van [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl). In de volgende les, *Ablation: wat gebeurt er als je een deel weghaalt?*, is die ene verandering een heel deel van je GPT, weggelaten. En omdat één run per instelling één worp met de dobbelsteen is, herhaalt een les daarna elke run met meer seeds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_confound, explain_design_cell, result_change):
    L.wrap_up(result_change, choice_confound, explain_design_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
