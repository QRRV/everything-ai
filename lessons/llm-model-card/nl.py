# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-model-card. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-model-card", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) en [Red teaming van je eigen chat model](/course/open/llm-safety-eval?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_datasheet():
    quick_datasheet = L.choice(
        "quick-datasheet",
        question="Wat schrijft de datasheet van een dataset op dat de tabel met data zelf niet laat zien?",
        options={
            "right": "Zijn bron, zijn licentie, toestemming, en zijn beperkingen",
            "stats": "Het gemiddelde en de spreiding van elke kolom",
            "models": "Welke modellen erop getraind zijn, en hoe goed ze het deden",
        },
        correct="right",
        why={
            "stats": "Die reken je uit met de tabel zelf; een datasheet bevat wat de tabel niet kan vertellen. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "models": "Hoe goed modellen het deden, hoort op hun eigen cards: een datasheet gaat over de data. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Een model card doet hetzelfde voor een model: hij zegt wat zijn bestanden en zijn scores niet uit zichzelf zeggen.",
        },
    )
    quick_datasheet
    return (quick_datasheet,)


@app.cell(hide_code=True)
def quick_check_filter():
    quick_filter = L.choice(
        "quick-filter",
        question="Met een filter van 11 woorden ervoor weigerde het model van de course 14 van de 20 schadelijke verzoeken. Wat deed het filter nog meer?",
        options={
            "only": "Niets: het hield alleen schadelijke verzoeken tegen",
            "jail": "Het hield ook alle 10 jailbreaks tegen",
            "right": "Het hield 4 onschuldige verzoeken tegen, en geen jailbreak",
        },
        correct="right",
        why={
            "only": "Het hield ook onschuldige verzoeken tegen, zoals *How do I kill a Python process?* Zie [Red teaming van je eigen chat model](/course/open/llm-safety-eval?lang=nl).",
            "jail": "Geen enkele: een jailbreak is zo geschreven dat het filter zijn woorden niet ziet. Zie [Red teaming van je eigen chat model](/course/open/llm-safety-eval?lang=nl).",
            "right": "Een card die alleen zei *het weigert 14 van de 20 schadelijke verzoeken*, zou ze allebei verzwijgen.",
        },
    )
    quick_filter
    return (quick_filter,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt een chat model getraind, en misschien wil iemand anders het gebruiken. Zonder een woord van jou weet die niet dat het alleen verhalen voor jonge kinderen schrijft, dat het nooit weigert, of dat *60 van de 100 testinstructies* minder betekent dan het lijkt. Een **model card** (Mitchell en anderen, 2019) hoort bij het model en zegt waarvoor het is en waarvoor niet, waarop het getraind is, hoe het getest is en met welke uitkomst, en waar het tekortschiet. Op Hugging Face is de card het bestand README.md van de repository van het model, met een header die de hub leest, zoals op de card van de cijferlezer uit [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).

    **Een voorbeeld uit het dagelijks leven.** De bijsluiter in een doosje medicijnen zegt waarvoor het is, hoeveel je neemt, wat de bijwerkingen zijn, en wanneer je het niet moet nemen. Je neemt toch ook geen pil uit een doosje zonder bijsluiter.
    """)
    return


@app.cell(hide_code=True)
def predict_section():
    guess_section = L.guess(
        "section",
        question="Liang en anderen (2024) onderzochten 32.111 model cards op Hugging Face. Welke sectie vonden ze het vaakst ingevuld?",
        options={
            "limits": "Limitations: wat het model niet kan",
            "right": "Training: hoe het model getraind is",
            "eval": "Evaluation: hoe goed het model het doet",
        },
        reveal=f"**Training: in {C.number(74.3, 1, L.lang)}% van de cards.** Evaluation was maar in {C.number(15.4, 1, L.lang)}% ervan ingevuld, en Limitations in {C.number(17.4, 1, L.lang)}%: twee van de vier secties die het minst ingevuld zijn. Wat een lezer het meest nodig heeft om te beslissen of een model bij zijn taak past, laten cards juist vaak weg.",
    )
    guess_section
    return (guess_section,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Feiten, geen hoop.** Alles op een card komt uit een bestand dat een run schreef, of uit een test die je deed. Een getal noemt zijn test en zijn aantal: *60 van de 100 testinstructies*, niet *60%*, en met de schone subset ernaast.

        **Waarvoor het is, en waarvoor niet.** Het **beoogde gebruik** zegt voor wie het model is en waarvoor. Ernaast staat waarvoor het niet is: wat iemand zou kunnen proberen, en niet moet doen.

        **Beperkingen komen uit de tests.** Wat de laatste trainingsstap slechter maakte, en wat nog te zwak is, zijn allebei bevindingen; en wat je niet getest hebt ook.

        **Een header die de hub leest.** Bovenaan README.md staat de **metadata** van de card: tussen twee regels `---` een regel `key: value` voor elk veld, en voor een lijst een regel `key:` met een regel `- item` voor elk item. Dat formaat is **YAML**. De hub toont er de licentie uit, en vindt het model via zijn datasets, taal en tags. De card van de cijferlezer begint zo:
        """),
        mo.Html(C.yaml_block(C.digit_reader_header())),
        mo.md(r"""
        **Licenties.** De card noemt de licentie van het model, en die van elke dataset waarvan het leerde. De licenties van de data kunnen voorwaarden stellen aan wat je ermee maakt: lees ze allemaal voordat je deelt.
        """),
        draw.flow(
            [("Je runs", ["capstone, SFT, DPO", "elk een .json-bestand"]),
             ("Je tests", ["verifieerbare instructies", "en red teaming"]),
             ("Model card", ["header en secties", "README.md"])],
            forward=["drie modellen", "scores, beperkingen"],
            back=[],
            title="Van je runs naar een model card"),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **beoogd gebruik** | voor wie een model is, en waarvoor; de card zegt ernaast waarvoor het niet is | `C.USES` |
    | **metadata** | data over data: op een model card de header met zijn licentie, datasets, taal en tags | `card_header(fields, lists)` |
    | **YAML** | het tekstformaat van die header: `key: value`, en een regel `- item` voor elk item van een lijst | `"license: cc-by-4.0"` |

    ### Recept

    Zo schrijf je een model card:

    1. Verzamel de feiten van elke run uit zijn bestanden: het model, zijn grootte, zijn data met hun licenties, zijn training.
    2. Schrijf waarvoor het model is, en waarvoor niet.
    3. Geef elke test met zijn aantal, voor en na elke trainingsstap, en de schone subset ernaast.
    4. Schrijf de beperkingen op die je tests vonden, en wat je niet getest hebt.
    5. Zet de licentie, de datasets, de taal en de tags in de header.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 voor het chat model van de course: de feiten van zijn drie runs, uit de bestanden die de runs schreven. `C.infos("course")` geeft een dict per run.
    """)
    return


@app.cell
def example_facts():
    _info = C.infos("course")
    _base, _sft, _dpo = _info["base"], _info["sft"], _info["dpo"]
    print("model:", _base["params"], "parameters, en", _dpo["trainable"], "in de adapters van rang", _dpo["rank"])
    print("pretraining:", _base["tokens"], "tokens van", _base["stories"], "verhalen, in", _base["steps"], "stappen")
    print("SFT:", _sft["pairs"], "paren van een instructie en een verhaal, in", _sft["steps"], "stappen")
    print("DPO:", _dpo["pairs"], "preference pairs uit", _dpo["instructions"], "instructies, in", _dpo["steps"], "stappen")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elk getal op deze regels komt uit een bestand dat een run schreef: niets is met de hand ingetypt, dus niets is gegokt. Wat de bestanden niet kunnen zeggen, is hoe goed het model het doet, en waar het tekortschiet: daarvoor heb je je tests nodig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_header():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De header

        Maak `card_header(fields, lists)` af. `fields` bevat de velden met één waarde, en `lists` die met een lijst waarden. Geef de header terug als één string: een regel `---`, een regel `key: value` voor elk veld, voor elke lijst een regel `key:` en een regel `- item` voor elk item, en een afsluitende regel `---`. De check leest je header zoals de hub dat doet, en roept `card_header({"license": "mit"}, {"language": ["en", "nl"]})` aan, en andere.
        """),
        L.task("Laat `card_header` de header van een model card schrijven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_header():
    box_header = L.code_box("card-header", lines=11)
    box_header
    return (box_header,)


@app.cell(hide_code=True)
def check_header(box_header):
    result_header = L.check("card-header", box_header.value)
    result_header
    return (result_header,)


@app.cell(hide_code=True)
def hints_header():
    L.hints(
        "card-header",
        [
            "Een lijst heeft twee soorten regels: één voor zijn sleutel, en één voor elk van zijn items. Welke loop komt in welke?",
            "De header eindigt zoals hij begint. Wat is zijn laatste regel?",
        ],
        why="De hub leest de header om de licentie te laten zien en het model te vinden op zijn datasets en taal; een header die hij niet kan lezen, is geen header.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_limits():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De beperkingen

        Maak `limits(before, after, totals, enough)` af. `before` en `after` geven voor elke soort van een test hoeveel er slaagden voor en na de laatste trainingsstap, en `totals` hoeveel er van die soort waren. Geef een lijst terug van de soorten die een beperking zijn: die slechter werden, en die waarvan het aandeel erna onder `enough` ligt. De check roept `limits({"rhyme": 10, "names": 35, "short": 6}, {"rhyme": 25, "names": 30, "short": 14}, {"rhyme": 50, "names": 40, "short": 20}, 0.6)` aan, en andere.
        """),
        L.task("Laat `limits` de soorten vinden die slechter werden of te zwak zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_limits():
    box_limits = L.code_box("limits", lines=8)
    box_limits
    return (box_limits,)


@app.cell(hide_code=True)
def check_limits(box_limits):
    result_limits = L.check("limits", box_limits.value)
    result_limits
    return (result_limits,)


@app.cell(hide_code=True)
def hints_limits():
    L.hints(
        "limits",
        [
            "Twee redenen maken een soort tot een beperking. Wat vergelijk je bij elk ervan?",
            "Elk van de twee redenen is genoeg. Welk woord verbindt twee voorwaarden zo dat er één genoeg is?",
        ],
        why="Een card die alleen noemt wat beter werd, verzwijgt wat een lezer het meest nodig heeft: waar het model tekortschiet.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### De card van je chat model

        De les schrijft de card uit de bestanden van de drie runs en de scores van de twee tests: van je eigen chat model als zijn capstone-, SFT- en DPO-run allemaal van jou zijn, en anders van dat van de course. Je `card_header` schrijft de header, en je `limits` vindt de beperkingen, onder het aandeel dat je instelt. Schrijf in de vakken waarvoor het model is en waarvoor niet, in het Engels, zoals de card. Je eigen modellen doen beide tests als je op de knop drukt: op de CPU van de course, met één thread, duurde dat ongeveer {round((C.E.MEASURED['yours']['seconds'] + C.F.MEASURED['yours']['seconds']) / 60)} minuten voor haar modellen van de demoruns. De knop onder de card bewaart hem als `work/projects/llm/{C.CARD}`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def card_inputs():
    _yours = C.source() == "yours"
    uses_box = mo.ui.text_area(value="" if _yours else C.USES, label="Waarvoor het is (in het Engels)", full_width=True)
    not_for_box = mo.ui.text_area(value="" if _yours else C.NOT_FOR, label="Waarvoor het niet is (in het Engels)", full_width=True)
    license_box = mo.ui.text(value="" if _yours else "cc-by-4.0", label="Licentie van het model", full_width=True)
    enough_slider = mo.ui.slider(10, 90, step=10, value=50, label="Genoeg, in %: een soort onder dit aandeel is een beperking", show_value=True)
    measure_button = mo.ui.run_button(label="Laat mijn modellen beide tests doen", disabled=not _yours)
    mo.vstack([uses_box, not_for_box, license_box, enough_slider, measure_button])
    return enough_slider, license_box, measure_button, not_for_box, uses_box


@app.cell(hide_code=True)
def demo_measure(measure_button):
    scores = C.measured(C.source())
    _note = None
    if measure_button.value:
        _names = {"base": "voor post-training", "sft": "na SFT", "dpo": "na SFT en DPO"}
        _tests = {"chat": "verifieerbare instructies", "safety": "red teaming"}

        def _progress(_p):
            mo.output.replace(mo.md(f"**{_tests[_p['test']]}** · {_names[_p['model']]} · prompt {_p['done']} van {_p['total']} · {C.number(_p['seconds'] / 60, 1, L.lang)} minuten"))
        _r = C.measure(_progress)
        if _r == "run":
            _note = "*Je modellen kunnen niet antwoorden: een module van je project stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks?*"
        else:
            scores = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (scores,)


@app.cell(hide_code=True)
def demo_card(scores, uses_box, not_for_box, license_box, enough_slider, box_header, box_limits, result_header, result_limits):
    _course = {"card_header": run_code(C.EXERCISES["card-header"].solution).names["card_header"],
               "limits": run_code(C.EXERCISES["limits"].solution).names["limits"]}
    _boxes = {"card_header": (box_header, result_header), "limits": (box_limits, result_limits)}
    _fns, _notes = {}, []
    for _k, (_box, _result) in _boxes.items():
        _fns[_k] = run_code(_box.value).names.get(_k) if _result.state == "correct" else None
        if _fns[_k] is None:
            _fns[_k] = _course[_k]
            _notes.append(f"*De `{_k}` van de course staat er tot de jouwe slaagt voor zijn check.*")
    _src = C.source()
    _enough = enough_slider.value / 100   # the slider is in %
    _fields, _lists = C.fields(_src, license_box.value)
    try:
        _head = _fns["card_header"](_fields, _lists)
        _ok = C._header(_head) is not None
    except Exception:
        _ok = False
    if not _ok:   # yours stops on these fields, or writes a header the hub cannot read: the course's stands in
        _head = _course["card_header"](_fields, _lists)
        _notes.append("*Je `card_header` schrijft voor deze velden geen header die de hub kan lezen: die van de course staat er.*")
    _found = []
    if scores is not None:
        _c = scores["chat"]
        _totals = {_k: _c["total"][_k] for _k in C.CHAT if _c["total"][_k]}
        try:
            _found = [_k for _k in _fns["limits"](_c["sft"], _c["dpo"], _totals, _enough) if _k in C.CHAT]
        except Exception:
            _found = _course["limits"](_c["sft"], _c["dpo"], _totals, _enough)
    card_text = C.card(_src, scores, _head, uses_box.value, not_for_box.value, _found, _enough, license_box.value)
    _body = card_text[len(_head.rstrip("\n")):].replace("\n## ", "\n#### ").replace("\n# ", "\n### ")   # its headings below the lesson's
    _whose = "je chat model" if _src == "yours" else "het chat model van de course"
    mo.vstack([
        mo.md(f"**De card van {_whose}.** De header:"),
        mo.Html(C.yaml_block(_head)),
        mo.Html(f"<div style='border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.2rem 1rem; overflow-x: auto'>{mo.md(_body).text}</div>"),
        *[mo.md(_n) for _n in _notes],
    ])
    return (card_text,)


@app.cell(hide_code=True)
def save_cell():
    save_button = mo.ui.run_button(label="Bewaar de card")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def demo_save(save_button, card_text):
    _msg = None
    if save_button.value:
        try:
            _msg = f"*Bewaard als `{C.save(card_text)}`: zet hem als README.md in de repository van je model als je het deelt.*"
        except OSError:
            _msg = "*De card kon niet bewaard worden: bestaat de map `work/projects/llm/`, en mag de course erin schrijven?*"
    mo.output.replace(mo.md(_msg)) if _msg else mo.output.clear()
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De bestanden van de runs vullen de secties over het model, zijn data en zijn training. De scores voegen toe wat die bestanden niet kunnen zeggen: hoe het model het doet op woorden die in de training nooit samen gevraagd werden, wat de laatste stap slechter maakte of te zwak liet, en of het weigert. Dat zijn de delen die de meeste cards weglaten. Waarvoor het model is en waarvoor niet, kan geen programma schrijven: dat is aan jou, en de licentie ook: lees eerst de licenties van zijn data.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_how_well():
    mo.md(r"""
    ### Oefening 3. Hoe goed doet het het?
    """)
    return


@app.cell(hide_code=True)
def choice_how_well_cell():
    choice_how_well = L.choice(
        "how-well",
        question="Welke regel onder *Evaluation* vertelt een lezer wat de 60 van de 100 van het chat model van de course waard is?",
        options={
            "vague": "Het volgt instructies goed: 60% op onze test",
            "right": "60 van de 100 testinstructies, de meeste ook in de training; 3 van de 20 die dat niet waren",
            "up": "60 van de 100 testinstructies na DPO, tegen 24 na SFT en 1 daarvoor",
        },
        correct="right",
        why={
            "vague": "Een aandeel zonder zijn test en zijn aantal geeft een lezer niets om te controleren, en *goed* is een hoop, geen meting.",
            "up": "Klopt, maar de meeste van die instructies waren ook trainingsinstructies: zonder de schone subset houdt een lezer 60 van de 100 voor wat het model met nieuwe doet.",
            "right": "Een getal met zijn test, zijn aantal, en de schone subset ernaast: een lezer ziet wat die 60 waard is.",
        },
    )
    choice_how_well
    return (choice_how_well,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De datasheet uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl), nu voor een model; de model card uit [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl), nu door jou geschreven; en de tests uit [Heeft post-training geholpen?](/course/open/llm-eval-chat?lang=nl) en [Red teaming van je eigen chat model](/course/open/llm-safety-eval?lang=nl), nu als zijn evaluatie en zijn beperkingen. Hierna: de formule van een paper omzetten in code.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_header, result_limits, choice_how_well):
    L.wrap_up(result_header, result_limits, choice_how_well)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
