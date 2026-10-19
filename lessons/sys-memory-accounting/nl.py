# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-memory-accounting. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-memory-accounting", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Adam](/course/open/opt-adam?lang=nl) en [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_adam():
    quick_adam = L.choice(
        "quick-adam",
        question="Hoeveel getallen per parameter neemt Adam van stap naar stap mee, naast de parameter zelf en zijn gradient?",
        options={
            "one": "Eén: zijn eigen learning rate",
            "right": "Twee: een gemiddelde van zijn gradients en een van hun kwadraten",
            "none": "Geen: hij gebruikt alleen de nieuwste gradient en de learning rate",
        },
        correct="right",
        why={
            "one": "De learning rate van Adam is één getal voor alle parameters. Wat hij per parameter bijhoudt, zijn twee gemiddelden. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "none": "Adam houdt voor elke parameter een gemiddelde van de gradients bij en een gemiddelde van hun kwadraten. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "right": "Twee getallen voor elke parameter, dus twee keer zoveel getallen als het model zelf.",
        },
    )
    quick_adam
    return (quick_adam,)


@app.cell(hide_code=True)
def quick_check_master():
    quick_master = L.choice(
        "quick-master",
        question="Je zet een trainingsrun om van FP32 naar mixed precision met autocast. Wat gebeurt er met het geheugen van de weights?",
        options={
            "half": "Het halveert, want de berekeningen zijn in BF16",
            "right": "Het blijft gelijk, want de weights blijven FP32",
            "double": "Het verdubbelt, want elke weight heeft nu een kopie ernaast",
        },
        correct="right",
        why={
            "half": "De berekeningen zijn in BF16, maar de master weights blijven FP32. Zie [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl).",
            "double": "De BF16-kopieën van autocast zijn half zo groot als de FP32-weights, en na de backward pass weer weg; de FP32-weights blijven zoals ze waren. Zie [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl).",
            "right": "De master weights zijn even groot als eerst. Mixed precision bespaart op de activations, niet op de weights.",
        },
    )
    quick_master
    return (quick_master,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een run die na twintig minuten stopt met "out of memory", heeft twintig minuten verspild. Voordat je begint, kun je optellen wat een trainingsrun in het geheugen houdt, en weten of het past. De sommen zijn eenvoudig, en de uitkomsten verrassen vaak: een model van 7 miljard parameters kun je in BF16 gebruiken op een GPU van 24 GB, maar het helemaal finetunen, verder trainen op je eigen data, vraagt ruim meer dan 100 GB.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt():
    guess_gpt = L.guess(
        "gpt-memory",
        question="De GPT van [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl) heeft ongeveer 10,8 miljoen parameters. Hoeveel geheugen nemen zijn weights, hun gradients en de optimizer state van AdamW in FP32?",
        options={
            "weights": "Ongeveer 43 MB",
            "right": "Ongeveer 173 MB",
            "gb": "Ongeveer 1 GB",
        },
        reveal="**Ongeveer 173 MB.** Elke parameter neemt 4 bytes voor de weight, 4 voor zijn gradient en 8 voor de twee gemiddelden van AdamW: 16 bytes. $10{,}8 \\times 10^6 \\times 16 \\approx 1{,}73 \\times 10^8$ bytes. 43 MB zijn alleen de weights. De activations komen erbovenop, en bij een klein model zijn die vaak het grootste deel.",
    )
    guess_gpt
    return (guess_gpt,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: vier delen.** Tijdens het trainen staan in het geheugen:

        - de **weights**: 4 bytes per parameter in FP32;
        - hun **gradients**: evenveel, 4 bytes per stuk;
        - de **optimizer state**: wat de optimizer per parameter van stap naar stap meeneemt. Gewone SGD houdt niets bij, momentum één getal (4 bytes), Adam en AdamW twee (8 bytes);
        - de **activations**: wat de forward pass bewaart voor de backward pass, zoals de input van elke linear layer.

        **Stap 2: 16 bytes per parameter.** Met AdamW tellen de eerste drie delen op tot $4 + 4 + 8 = 16$ bytes per parameter. Dat geldt in FP32, en ook in mixed precision, waar de master weights, hun gradients en de optimizer state van AdamW FP32 blijven.

        **Stap 3: de activations.** Hun geheugen hangt minder af van het aantal parameters dan van wat er door het model stroomt: het groeit met het aantal vensters in een batch, het aantal tokens per venster, het aantal layers en de breedte $d$; de attention scores groeien met het kwadraat van de lengte van het venster, en de logits van de output head met de grootte van de vocabulary. Je kunt ze meten: `torch.autograd.graph.saved_tensors_hooks` roept een functie van jou aan voor elke tensor die de forward pass bewaart voor de backward pass.

        **Stap 4: wat waar helpt.** Een kleinere batch, met gradient accumulation om de effectieve batch size te houden, maakt de activations kleiner; mixed precision ook, al is het minder dan de helft, omdat softmax, sommen en de loss FP32 blijven. Een optimizer met minder optimizer state, of een die zijn optimizer state in 8 bits opslaat, maakt de optimizer state kleiner. Een kleiner model maakt de weights en gradients kleiner; maar een paar van de parameters trainen laat de gradients en de optimizer state van de rest weg. Echte runs hebben ook wat geheugen nodig voor PyTorch zelf en verliezen wat aan fragmentatie, dus houd 10 tot 20% vrij.

        Het plaatje: een model van 7 miljard parameters gebruiken in BF16, links, tegenover het helemaal finetunen met AdamW, rechts, met stippellijnen bij het geheugen van GPU's van 24 en 80 GB. De getallen in de staven zijn GB.
        """),
        mo.Html(C.bars_picture({"title": "Een model van 7 miljard parameters", "y": "GB", "weights": "weights", "grads": "gradients", "state": "optimizer state",
                                "bars": ["gebruiken", "trainen"], "gpu24": "24 GB", "gpu80": "80 GB", "lang": L.lang,
                                "alt": "Twee staven. Gebruiken in BF16: 14 GB weights, onder de lijn van 24 GB. Trainen met AdamW: 28 GB weights, 28 GB gradients en 56 GB optimizer state van AdamW op elkaar, 112 GB nog vóór de activations, boven de lijnen voor GPU's van 24 en 80 GB."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **optimizer state** | de getallen die een optimizer per parameter van stap naar stap meeneemt, zoals de twee gemiddelden van Adam | `opt.state` |
        | **activation memory** | het geheugen van de tensors die de forward pass bewaart voor de backward pass | |
        """),
        L.watch_out("Een GB is in deze course $10^9$ bytes. Sommige tools tonen GiB, $2^{30}$ bytes, ongeveer 7% meer per eenheid, en het geheugen van een GPU die als 80 GB verkocht wordt, ligt dicht bij 80 GiB: iets meer ruimte dan de sommen hier zeggen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo weet je vóór een run of hij past:

    1. Tel de parameters.
    2. Vermenigvuldig met de bytes per parameter: 4 voor de weight, 4 voor zijn gradient, en de optimizer state, 8 voor AdamW.
    3. Meet de activations voor één batch, of neem ze uit een run met een kleinere batch: ze groeien evenredig met de batch.
    4. Tel het op, en houd 10 tot 20% van de GPU vrij.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Activations meten is nieuw: een kleine MLP, en twee functies voor `saved_tensors_hooks`. Die roept de eerste aan voor elke tensor die de forward pass bewaart, en bewaart wat die functie teruggeeft, hier de tensor zelf; de tweede geeft hem terug aan de backward pass, hier onveranderd. `t.numel()` is het aantal getallen in een tensor `t`, `t.element_size()` de bytes van elk, en `t.data_ptr()` de plek in het geheugen waar zijn getallen beginnen. De functie laat de parameters weg, die al geteld zijn, en telt elke plek één keer, omdat twee bewerkingen dezelfde tensor kunnen bewaren.
    """)
    return


@app.cell
def example_saved():
    import torch
    from torch.autograd.graph import saved_tensors_hooks

    _hidden, _out = torch.nn.Linear(64, 256), torch.nn.Linear(256, 10)
    _parameters = list(_hidden.parameters()) + list(_out.parameters())
    _places = {p.data_ptr() for p in _parameters}
    _kept = {}

    def _pack(t):
        if t.data_ptr() not in _places:
            _kept[t.data_ptr()] = t.numel() * t.element_size()
        return t

    with saved_tensors_hooks(_pack, lambda t: t):
        _loss = _out(torch.relu(_hidden(torch.randn(32, 64)))).sum()
    print("activations bewaard voor de backward pass:", sum(_kept.values()), "bytes")
    print("parameters:", sum(p.numel() * 4 for p in _parameters), "bytes")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De MLP bewaart 40.960 bytes voor een batch van 32: de input, 32 × 64 getallen, en de output van ReLU, 32 × 256 getallen, die zowel de ReLU als de tweede layer nodig hebben. Zijn parameters nemen 76.840 bytes. Met een batch van 320 nemen de activations tien keer zoveel, en blijven de weights gelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gb():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Tel het op

        Schrijf `training_gb(n_params, optimizer, activation_gb)`: het geheugen van een trainingsrun in GB, $10^9$ bytes, in FP32 of mixed precision: de weights, hun gradients en de optimizer state van `optimizer`, die `"sgd"`, `"momentum"` of `"adamw"` is, plus de activations, die al in GB binnenkomen. De check roept `training_gb(7e9, "adamw", 0)` aan, wat 112 moet geven, en probeert andere modellen, optimizers en activations.
        """),
        L.task("Schrijf `training_gb(n_params, optimizer, activation_gb)`: de weights, gradients en optimizer state van `n_params` parameters in GB, plus `activation_gb`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gb():
    box_gb = L.code_box("training-gb", lines=7)
    box_gb
    return (box_gb,)


@app.cell(hide_code=True)
def check_gb(box_gb):
    result_gb = L.check("training-gb", box_gb.value)
    result_gb
    return (result_gb,)


@app.cell(hide_code=True)
def hints_gb():
    L.hints(
        "training-gb",
        [
            "Tel eerst de bytes van één parameter op, stap 1 en 2 van het recept; de rest is vermenigvuldigen, omrekenen naar GB, en de activations erbij optellen.",
        ],
        why="Met AdamW is dat $4 + 4 + 8 = 16$ bytes per parameter: 112 GB voor 7 miljard parameters, nog vóór één activation.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De activations van jouw GPT

    De knop meet jouw `GPT` uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), in een kleine maat: 2 blocks, $d = 64$, vensters van 64 tokens. Waar een module van jou ontbreekt, springt de versie van de course in. Hij telt wat de forward pass bewaart voor batches van 1 tot 64 vensters, naast de 16 bytes per parameter van trainen met AdamW. Het duurt een seconde.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Meet mijn GPT")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(measure_button):
    if not measure_button.value:
        _out = mo.md("*Druk op de knop om te meten.*")
    else:
        _m = C.measure()
        if _m == "error":
            _out = mo.md("*Je GPT stopte met een fout bij deze maten: werkt hij nog in [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl)?*")
        else:
            _fixed = 16 * _m["params"]
            _n = lambda _x, _d=2: C.number(_x, _d, L.lang)
            _rows = "\n".join(f"| {_b} | {_n(_v / 1e6)} MB | {_n(_v / _fixed, 1)}× |" for _b, _v in _m["rows"])
            _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_s}`" for _s in _m["stand_ins"]) + ".") if _m["stand_ins"] else ""
            _out = mo.md(f"""
    Jouw GPT heeft {_n(_m['params'], 0)} parameters: {_n(_fixed / 1e6)} MB aan weights, gradients en optimizer state van AdamW.{_stand}

    | vensters per batch | activations | keer de 16 bytes per parameter |
    |---|---|---|
    {_rows}

    De activations groeien evenredig met de batch. Bij een model dat zo klein is, zijn ze vanaf een handvol vensters het grootste deel.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_fine():
    mo.md(r"""
    ### Oefening 2. Fine-tunen op één GPU
    """)
    return


@app.cell(hide_code=True)
def choice_fine_cell():
    choice_fine = L.choice(
        "fine-tune",
        question="Je wilt alle parameters van een model van 7 miljard parameters finetunen met AdamW, in mixed precision, op één GPU van 24 GB. Past dat?",
        options={
            "bf16": "Ja: in BF16 neemt het model maar 14 GB",
            "batch": "Ja, met een batch van één venster",
            "right": "Nee, zelfs niet zonder activations",
        },
        correct="right",
        why={
            "bf16": "14 GB is het model in BF16 om te gebruiken. Trainen houdt FP32-master weights, hun gradients en de optimizer state van AdamW bij: 16 bytes per parameter.",
            "batch": "Een kleine batch maakt alleen de activations kleiner; de 112 GB aan weights, gradients en optimizer state van AdamW blijft.",
            "right": "Weights, gradients en de optimizer state van AdamW nemen alleen al $7 \\times 10^9 \\times 16 = 112$ GB. Op één GPU lukt dit door maar een paar van de parameters te trainen, met de rest in BF16 of nog minder bits: een latere les.",
        },
    )
    choice_fine
    return (choice_fine,)


@app.cell(hide_code=True)
def text_exercise_oom():
    mo.md(r"""
    ### Oefening 3. Out of memory
    """)
    return


@app.cell(hide_code=True)
def explain_oom():
    explain_oom_cell = L.explain(
        "out-of-memory",
        question="Het trainen van je GPT stopt met \"out of memory\" op een GPU. Noem drie dingen die je kunt veranderen, en bij elk welk deel van het geheugen het kleiner maakt, en wat het kost.",
        model_answer="**Een kleinere batch, met gradient accumulation:** maakt de activations kleiner, die evenredig met de batch groeien; accumulation houdt de effectieve batch size gelijk, ten koste van meer stappen per update en wat meer tijd. **Kortere vensters:** maakt de activations kleiner, de attention scores zelfs met het kwadraat van de lengte; het model ziet dan minder context tegelijk. **Mixed precision:** houdt een groot deel van de activations in BF16, zodat ze kleiner worden, al is het minder dan de helft, en draait op een GPU vaak sneller; de prijs is grovere stappen in de berekeningen, en op een GPU zonder BF16 heb je FP16 met loss scaling nodig. De weights, gradients en optimizer state blijven even groot. **Een optimizer met minder optimizer state,** zoals SGD met momentum in plaats van AdamW, of een die zijn optimizer state in 8 bits opslaat: maakt de optimizer state kleiner; hij traint misschien slechter of moet opnieuw afgesteld worden. **Een kleiner model:** maakt alles kleiner, ten koste van wat het model kan leren.",
        points=[
            "je noemt minstens drie veranderingen",
            "bij elke noem je het deel dat kleiner wordt: activations, optimizer state, of weights en gradients",
            "bij elke noem je een prijs, zoals meer tijd, minder context, slechter trainen of een zwakker model",
        ],
    )
    explain_oom_cell
    return (explain_oom_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De twee gemiddelden van Adam komen uit [Adam](/course/open/opt-adam?lang=nl), de FP32-master weights uit [Mixed-precision training](/course/open/sys-mixed-precision?lang=nl), en gradient accumulation uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl). De volgende les, *Trainingscompute schatten: de 6ND-vuistregel*, telt het andere budget van een run op: zijn FLOPs, en dus zijn tijd.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fine, explain_oom_cell, result_gb):
    L.wrap_up(result_gb, choice_fine, explain_oom_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
