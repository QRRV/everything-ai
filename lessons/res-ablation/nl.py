# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-ablation. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("res-ablation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) en [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl). <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def quick_check_control():
    quick_control = L.choice(
        "quick-control",
        question="Welke run is in een experiment met je GPT de control run?",
        options={
            "change": "De run met de verandering die je wilt toetsen",
            "right": "Je run zonder de verandering, verder in alles gelijk",
            "other": "Een run van een ander soort model, zoals het bigram-model",
        },
        correct="right",
        why={
            "change": "Dat is de variant. Zie [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl).",
            "other": "Een ander soort model is een baseline, geen control. Zie [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl).",
            "right": "Elke variant wordt ermee vergeleken, dus hij verschilt van elke variant in maar één instelling.",
        },
    )
    quick_control
    return (quick_control,)


@app.cell(hide_code=True)
def quick_check_positions():
    quick_positions = L.choice(
        "quick-positions",
        question="Je GPT telt een rij van zijn tabel `self.pos` op bij de embedding van elk token. Wat vertelt die rij de blocks?",
        options={
            "meaning": "Wat het token betekent",
            "right": "Op welke plek in het venster het token staat",
            "next": "Hoe groot de kans is dat elk token er direct na komt",
        },
        correct="right",
        why={
            "meaning": "Dat is de rij uit de tabel met tokens, `self.tok`. Zie [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl).",
            "next": "Dat komt er aan het eind uit, uit de head. Zie [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl).",
            "right": "De tabel heeft een rij voor elke plek; zonder die tabel kan één block niet zien waar een token staat.",
        },
    )
    quick_positions
    return (quick_positions,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een paper presenteert een model met veel delen, en laat zien dat het werkt. Maar welke delen heeft het nodig? Papers beantwoorden dat met een ablation: haal één deel weg, train opnieuw op precies dezelfde manier, en meet. Een tabel met zulke runs, het volledige model bovenaan, laat zien wat elk deel toevoegt. In [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) was die ene verandering een instelling; hier is het een deel van je GPT, weggehaald. In deze projectstap doe je het met de position embedding van je eigen GPT. <!-- termen:ok -->
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het leren instructies en voorkeuren te volgen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_positions():
    guess_positions = L.guess(
        "no-positions",
        question="Haal de position embedding uit je GPT: hij ziet dan welke tekens ervoor kwamen, maar niet waar ze staan. Wat denk je dat er met zijn validation loss gebeurt na dezelfde run?",
        options={
            "right": "Ongeveer gelijk: attention ziet nog elk teken ervoor",
            "worse": "Duidelijk slechter: hij weet niet meer in welke volgorde de tekens kwamen",
            "guess": "Zo slecht als gelijkmatig gokken tussen de 65 tekens",
        },
        reveal="**Ongeveer gelijk, na de hele run.** In de runs van de course met het CPU-plan, een GPT van 5 blocks die 10 epochs traint, met seeds 0 tot en met 2, eindigde de volledige GPT op 1,660 tot 1,674 en de GPT zonder posities op 1,663 tot 1,668: geen verschil groter dan de spreiding tussen seeds. Zonder posities leerde hij wel langzamer: na 2 epochs stond hij op 2,23 tot 2,26, de volledige GPT op 2,06 tot 2,09. Met meer blocks en de causal mask kan een GPT de volgorde zelf uitzoeken. In 2022 vonden Adi Haviv en zijn collega's hetzelfde voor veel grotere language models: zonder posities kwamen ze in de buurt van modellen met posities, al bleven ze iets achter ([Haviv et al., 2022](https://arxiv.org/abs/2203.16634)). Op het GPU-plan, een grotere GPT die 20 epochs traint, draaide de course beide GPT's één keer, met seed 0. De GPT zonder posities lag na elke epoch behalve de eerste achter: na 6 epochs met 0,25, en aan het eind met 0,014: 1,538 tegen 1,524. Eén run per rij kan niet zeggen of 0,014 meer is dan toeval: de volgende les laat zien hoeveel één run verschuift.",
    )
    guess_positions
    return (guess_positions,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **ablation** haalt één deel uit een model, traint het model opnieuw op precies dezelfde manier, en meet hoeveel slechter het wordt. Het is een experiment zoals in de vorige les: het volledige model is de control run, het model zonder dat deel is de variant, en al het andere blijft gelijk: de data, de grootte, de epochs, de seed. Een paar ablations samen, die elk een ander deel weghalen, vormen een **ablation study**, die papers als tabel laten zien: het volledige model in de eerste rij, dan één rij per weggehaald deel.

        Om de position embedding weg te halen, zou je je GPT kunnen herschrijven. Het kan eenvoudiger: zet elk getal van zijn tabel op 0, zodat hij niets bij de tokens optelt, en **bevries** hem, zodat trainen de getallen niet terug kan leren. In PyTorch is een parameter bevroren als zijn `requires_grad` `False` is: hij houdt geen hellingen bij, dus de optimizer laat hem met rust. De rest van je GPT, de tabel met tokens, de blocks en de head, traint als vanouds.

        Zonder posities ziet attention nog elk teken ervoor, maar niet in welke volgorde. Eén block kan "abc" op de laatste plek niet van "bac" onderscheiden: het ziet dezelfde drie tekens. Met meer blocks kan een GPT de volgorde weer uitzoeken, via de causal mask: de eerste plek ziet één teken, de tweede twee, enzovoort, en latere blocks kunnen de plekken daaraan herkennen.
        """),
        mo.Html(C.flow_picture({"title": "Je GPT, met de posities weggehaald",
                                "tok": "tabel met tokens", "pos": "tabel met plekken", "pos_note": "elk getal 0, bevroren", "plus": "+",
                                "blocks": "de blocks", "head": "norm en head", "out": "het volgende teken",
                                "alt": "De delen van je GPT van boven naar beneden: de tabel met tokens en de tabel met plekken, opgeteld; de tabel met plekken doorgestreept, elk getal 0 en bevroren; dan de blocks, dan de norm en de head, dan het volgende teken."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **ablation** | één deel van een model weghalen, op dezelfde manier opnieuw trainen, en meten hoeveel slechter het wordt | |
        | **ablation study** | een paar ablations, die elk een ander deel weghalen, vaak als tabel | |
        | **bevriezen** | parameters vast laten staan terwijl de rest traint | `p.requires_grad = False` |
        """),
        L.watch_out("Een ablation laat zien wat een deel toevoegt in deze opzet: dit model, deze data, zoveel epochs. Met een groter model of een langere run kan hetzelfde deel meer of minder uitmaken. En één run per rij is één worp met de dobbelsteen, zoals de volgende lessen laten zien."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om uit te zoeken wat één deel van een model toevoegt:

    1. Kies het deel, en schrijf op wat je zonder dat deel verwacht.
    2. De control run: het volledige model, gewoon getraind.
    3. De variant: hetzelfde model met het deel weggehaald, zijn getallen op 0 en bevroren.
    4. Train de variant precies zoals de control: dezelfde data, grootte, epochs en seed. Vergelijk de validation losses.
    5. Schrijf het resultaat op als rij van een ablation-tabel, en zeg wat het niet laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en 3 op een kleine GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), met 1 block en 4 plekken, die nog niet getraind is. **1. De verwachting:** zonder posities geeft hij op de laatste plek dezelfde logits voor "abc" en voor "bac". Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_order():
    import torch

    _model = C.tiny_gpt()
    _abc = torch.tensor([[0, 1, 2]])
    _bac = torch.tensor([[1, 0, 2]])
    with torch.no_grad():
        # de logits op de laatste plek, voor "abc" en voor "bac"
        _before = torch.allclose(_model(_abc)[0, -1], _model(_bac)[0, -1])
        # 3. haal de posities weg: elk getal van de tabel met plekken op 0
        _model.pos.weight[0] = 0  # maak af: elke rij, [:], niet alleen de eerste
        _after = torch.allclose(_model(_abc)[0, -1], _model(_bac)[0, -1])
    print("gelijk met posities:", _before)
    print("gelijk zonder posities:", _after)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met posities krijgt de laatste plek andere logits voor "abc" en "bac": de eerste twee tekens staan op andere plekken. Met alleen rij 0 op 0 markeert rij 1 nog de tweede plek, dus verschillen ze nog. Met elke rij op 0 geven de twee dezelfde logits: één block ziet dezelfde drie tekens, en niets zegt waar. In het uitgewerkte voorbeeld traint het model niet, dus bevriezen doet er nog niet toe; in je run wel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_function():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Haal de posities weg

        Maak `no_positions(model)` af: stap 3 van het recept voor elke GPT van jou. Zet elk getal van `model.pos.weight` op 0, zonder graaf, en bevries hem. De functie verandert het model zelf en geeft niets terug. De check roept hem aan op de GPT van het uitgewerkte voorbeeld, ook met daarna een stap training, en op een GPT met 2 blocks. Klopt hij, dan bewaart de check hem in je project, als `work/projects/llm/ablate.py`.
        """),
        L.course_versions(),
        L.task("Laat `no_positions(model)` de tabel met plekken van `model` op 0 zetten en bevriezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_function():
    box_nopos = L.code_box("no-positions", lines=7)
    box_nopos
    return (box_nopos,)


@app.cell(hide_code=True)
def check_function(box_nopos):
    result_nopos = L.check("no-positions", box_nopos.value)
    result_nopos
    return (result_nopos,)


@app.cell(hide_code=True)
def hints_function():
    L.hints(
        "no-positions",
        [
            "Twee regels, zoals in stap 3 van het uitgewerkte voorbeeld en in *Het idee*: de ene zet elk getal van de tabel op 0, onder `with torch.no_grad():`; de andere laat hem stoppen met trainen.",
            "`model.pos.weight[___] = 0` onder `with torch.no_grad():`, en dan `model.pos.weight.requires_grad = ___`.",
        ],
        why="```python\nimport torch\n\n\ndef no_positions(model):\n    with torch.no_grad():\n        model.pos.weight[:] = 0\n    model.pos.weight.requires_grad = False\n```\n\n`[:]` zet elke rij op 0, en met `torch.no_grad()` mag je een parameter ter plekke veranderen. Met `requires_grad` op `False` krijgt de tabel geen hellingen, dus laat de optimizer hem op 0.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _p = C.T.plan()
    _gpu = "de GPU van je Mac" if _p["device"] == "mps" else f"je GPU, {_p['name']}"
    _where = "op de CPU, de kleinere GPT van het CPU-plan" if _p["device"] == "cpu" else f"op {_gpu}, de GPT van {C.T.grouped(_p['params'], L.lang)} parameters"
    _time = f" Op de computer van de course, een CPU met 4 cores, duurde zo'n run ongeveer {round(C.T.REFERENCE['cpu'][0]['minutes'])} minuten." if _p["device"] == "cpu" else ""
    mo.md(f"""
    ### Stap 4: de run zonder posities

    De knop traint je GPT nog een keer, {_where}, precies zoals in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl): hetzelfde plan, {_p['epochs']} epochs en seed 0, met je `no_positions` toegepast voordat hij traint.{_time} Tijdens het trainen laat de cel na elke epoch de loss zien, en wacht de rest van de les; met de stopknop van de cel stop je de run eerder. Hieronder staat je run naast de run met posities: die van jou, als je er een trainde op deze computer, en anders die van de course.
    """)
    return


@app.cell(hide_code=True)
def ablation_button_cell():
    ablation_button = mo.ui.run_button(label="Train je GPT zonder posities")
    ablation_button
    return (ablation_button,)


@app.cell(hide_code=True)
def demo_ablation(ablation_button, result_nopos):
    result_nopos  # run again when exercise 1 saves your no_positions
    _labels = {"title": "Je run zonder posities", "x": "epoch", "y": "loss", "lang": L.lang, "guess": "gelijkmatig gokken", "mlp": "je MLP", "train": "training", "val": "validatie"}

    def _progress(_p):
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · nog ongeveer {_m} {'minuten' if _m > 1 else 'minuut'} te gaan"
        _parts = [mo.md(f"**Epoch {_p['epoch']} van {_p['epochs']}** · stap {_p['step']} van {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.T.history_picture(_p["history"], {**_labels, "alt": "De training en validation loss van je run zonder posities tot nu toe, na elke epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    _result = C.train(_progress) if ablation_button.value else C.last_run()
    if _result is None and result_nopos.state == "correct" and not ablation_button.value:
        _out = mo.md("*Druk op de knop om de run te starten.*")
    elif _result is None:
        _out = mo.md("*Maak eerst oefening 1 af, en druk dan op de knop.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*De run stopt met een fout. Komen je modules in `work/projects/llm/` en `work/projects/nn/` nog door hun checks? Verwijder een bestand dat je veranderde, en controleer zijn oefening opnieuw.*",
            "memory": "*Je GPU heeft geen geheugen meer. Sluit andere programma's die hem gebruiken, zoals een game of een ander notebook, en druk opnieuw op de knop.*",
            "stopped": "*Je stopte de run tijdens zijn eerste epoch, dus er is niets van bewaard. Druk op de knop om opnieuw te beginnen.*",
        }[_result])
    else:
        _control, _who = C.control()
        _h = _result["history"]
        _stopped = f" Je stopte de run na epoch {len(_h)} van {_result['epochs']}, dus vergelijk voorzichtig." if _result["stopped"] else ""
        _stand = "" if not _result.get("stand_ins") else " De versie van de course sprong in voor " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + ", zoals de melding bij oefening 1 zegt."
        if _control is None:
            _out = mo.md(f"Zonder posities haalde je GPT een validation loss van {C.number(_result['best'], 3, L.lang)} na epoch {_result['best_epoch']}.{_stopped}{_stand} Train je GPT met posities in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl) om te vergelijken.")
        else:
            _whose = "je run uit [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl)" if _who == "yours" else "de run van de course met hetzelfde plan en seed 0"
            _d = _result["best"] - _control["best"]
            _diffs = C.course_differences("cpu" if _result["device"] == "cpu" else "gpu")
            _course = "" if not _diffs else f" In de {'runs' if len(_diffs) > 1 else 'run'} van de course met dit plan{', seed voor seed' if len(_diffs) > 1 else ', met seed 0'}, was het verschil zonder posities min met posities {'; '.join(C.signed(_x, 3, L.lang) for _x in _diffs)}."
            _rows = [("volledige GPT", C.number(_control["best"], 3, L.lang)), ("zonder posities", C.number(_result["best"], 3, L.lang))]
            _out = mo.vstack([
                mo.Html(C.pair_picture(_control["history"], _h, {"title": "Validation loss", "x": "epoch", "y": "loss", "lang": L.lang, "guess": "gelijkmatig gokken",
                                                                  "with": "met posities", "without": "zonder",
                                                                  "alt": f"De validation loss na elke epoch, met posities omlaag tot {C.number(_control['best'], 2, L.lang)} en zonder posities omlaag tot {C.number(_result['best'], 2, L.lang)}."},
                                       _result["epochs"])),
                mo.Html(C.table_picture(_rows, {"model": "model", "loss": "laagste validation loss"})),
                mo.md(f"De volledige GPT is {_whose}. Zonder posities eindigde je GPT {C.number(abs(_d), 3, L.lang)} {'hoger' if _d > 0 else 'lager'}.{_stopped}{_course}{_stand}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_table():
    mo.md(r"""
    ### Oefening 2. Een ablation-tabel lezen
    """)
    return


@app.cell(hide_code=True)
def choice_table_cell():
    choice_table = L.choice(
        "table",
        question="Een ablation-tabel zoals een paper hem laat zien, verzonnen voor deze vraag: het volledige model 1,52; zonder de MLP's 1,95; zonder layer normalization 1,71; zonder de position embedding 1,53. Elke rij is één keer getraind. Wat kun je zeggen?",
        options={
            "right": "De MLP's voegen hier het meest toe; de position embedding voegt misschien niets toe, of weinig",
            "pos": "De position embedding kan uit elk model weg, want hij veranderde de loss maar 0,01",
            "order": "Layer normalization doet er meer toe dan de MLP's",
        },
        correct="right",
        why={
            "pos": "In deze tabel maakte hij een verschil van 0,01, in één run van één opzet. Dat bewijst niet dat hij uit elk model kan. Zie het let-op in *Het idee*.",
            "order": "Zonder de MLP's steeg de loss meer, 0,43, dan zonder layer normalization, 0,19. Zie *Het idee*.",
            "right": "Elke rij vergelijk je met het volledige model: hoe groter de stijging, hoe meer het deel toevoegt. Een stijging van 0,01 in één run is te klein om veel te zeggen, zoals de runs zonder posities in deze les laten zien.",
        },
    )
    choice_table
    return (choice_table,)


@app.cell(hide_code=True)
def text_exercise_shows():
    mo.md(r"""
    ### Oefening 3. Wat laat je ablation zien?
    """)
    return


@app.cell(hide_code=True)
def explain_shows():
    explain_shows_cell = L.explain(
        "what-it-shows",
        question="Schrijf je ablation op als rij van een tabel, met de volledige GPT erboven. Geen eigen run? Neem die van de course, op het CPU-plan met seed 0: 1,662 voor de volledige GPT; 1,668 zonder posities. En dan: wat laat hij zien over de position embedding, en wat niet?",
        model_answer="**Tabel:** volledige GPT 1,662; zonder posities 1,668 (de runs van de course met het CPU-plan, seed 0). **Het laat zien** dat in deze opzet, een GPT van 5 blocks die 10 epochs traint, de position embedding aan het eind niets meetbaars toevoegt: het verschil, 0,006, is niet groter dan hoe ver runs met andere seeds uit elkaar vallen. Met posities leerde hij in de eerste epochs wel sneller. **Het laat niet zien** dat posities er nooit toe doen: met één block, langere teksten of andere data kan het anders zijn, en elke rij is één run.",
        points=[
            "je geeft beide rijen, met hun validation losses",
            "je zegt wat het verschil betekent voor de position embedding, in deze opzet",
            "je noemt een grens: één run per rij, of alleen dit model, deze data en deze lengte van trainen",
        ],
    )
    explain_shows_cell
    return (explain_shows_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl) gaf je GPT zijn posities; nu heb je gemeten wat ze toevoegen. Het experiment is dat van [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl), met een deel weggehaald in plaats van een instelling veranderd. Elke rij van je tabel is één run. In de volgende les, *Eén run is geen resultaat: seeds en variantie*, herhaal je een run met andere seeds, en zie je hoeveel een getal alleen door toeval beweegt. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_table, explain_shows_cell, result_nopos):
    L.wrap_up(result_nopos, choice_table, explain_shows_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
