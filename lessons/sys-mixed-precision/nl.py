# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-mixed-precision. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-mixed-precision", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl) en [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_range():
    quick_range = L.choice(
        "quick-range",
        question="Wat neemt BF16 over van FP32?",
        options={
            "digits": "De 23 mantissebits, dus de fijne stappen",
            "right": "De 8 exponentbits, dus het bereik",
            "both": "Allebei, samengeperst in de helft van de bits",
        },
        correct="right",
        why={
            "digits": "BF16 heeft maar 7 mantissebits: zijn stappen bij 1 zijn 1/128. Zie [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl).",
            "both": "In 16 bits passen geen 8 exponentbits en 23 mantissebits. Zie [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl).",
            "right": "Dus krijgt BF16 zelden een overflow of underflow, maar zijn stappen zijn grof. Deze les houdt de getallen die fijne stappen nodig hebben in FP32.",
        },
    )
    quick_range
    return (quick_range,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="Wat verandert `opt.step()` in je trainer?",
        options={
            "grads": "De gradients van de parameters",
            "right": "De parameters zelf",
            "loss": "De loss van de volgende mini-batch",
        },
        correct="right",
        why={
            "grads": "De gradients komen uit `backward()`; `opt.zero_grad()` maakt ze leeg. Zie [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).",
            "loss": "De loss verandert omdat de parameters veranderden; de stap zelf raakt alleen de parameters. Zie [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).",
            "right": "De optimizer telt bij elke parameter een kleine update op. Of die update blijft, hangt af van het formaat waarin de parameters staan.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl) kwam een weight van 1,0 in BF16 nooit van zijn plek: elke kleine update rondde weg. Toch worden grote modellen in 16 bits getraind. De truc is niet alles in 16 bits te zetten. **Mixed precision**, in 2018 gepubliceerd door Paulius Micikevicius en collega's, bewaart de weights zelf in FP32 en doet alleen de matrixvermenigvuldigingen in 16 bits. `torch.autocast` van PyTorch regelt de boekhouding, dus er komt in een trainingsstap maar één regel bij.
    """)
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights-format",
        question="Je draait de forward pass van een trainingsstap binnen `torch.autocast(..., dtype=torch.bfloat16)`. In welk formaat staan de weights van je model na de stap?",
        options={
            "bf16": "BF16, net als de berekeningen",
            "right": "FP32, zoals eerst",
            "mixed": "Sommige layers in BF16, sommige in FP32",
        },
        reveal="**FP32, zoals eerst.** Autocast verandert alleen het formaat van de berekeningen binnen zijn blok: een lineaire layer krijgt voor de vermenigvuldiging een BF16-kopie van zijn weights en zijn input, en de weights zelf blijven FP32. Hun gradients komen ook in FP32 terug, en de optimizer werkt de FP32-weights bij, dus een kleine update gaat niet verloren.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: drie soorten getallen.** Een trainingsstap heeft weights, activations en gradients. De activations zijn de outputs van de layers. De optimizer houdt voor elke weight ook eigen getallen bij, zoals de twee gemiddelden van AdamW. Mixed precision houdt de weights in FP32, als de **master weights**, en de gemiddelden van de optimizer ook. De matrixvermenigvuldigingen van de forward en de backward pass gaan in BF16.

        **Stap 2: autocast.** Zet de forward pass en de loss binnen `with torch.autocast(device_type=..., dtype=torch.bfloat16):`. Daarbinnen maken bewerkingen op de 16-bitslijst van autocast, zoals `nn.Linear`, matrixvermenigvuldigingen en convoluties, BF16-kopieën van hun inputs en rekenen ze in BF16. Bewerkingen op zijn FP32-lijst, die bereik of fijne stappen nodig hebben, zoals de cross-entropy-loss, rekenen in FP32; op een NVIDIA-GPU staan daar ook softmax, logaritmen en sommen op. Andere bewerkingen, zoals ReLU, houden het formaat van hun input. Roep `backward()` na het blok aan: de backward van elke bewerking draait in het formaat van zijn forward, en de gradients van FP32-weights komen in FP32 terug. De stap van de optimizer verandert niet.

        **Stap 3: FP16 heeft loss scaling nodig.** Met FP16 in plaats van BF16 vallen kleine gradients onder zijn bereik: onder ongeveer $3 \times 10^{-8}$ worden ze 0. **Loss scaling** vermenigvuldigt de loss vóór `backward()` met een grote factor $S$, zoals 65.536, zodat elke gradient $S$ keer groter is en binnen het bereik blijft; vóór de stap worden ze weer door $S$ gedeeld. Werd een gradient `inf` of `nan`, dan wordt die stap overgeslagen en $S$ gehalveerd; na 2000 goede stappen op rij wordt $S$ weer verdubbeld. `torch.amp.GradScaler` doet dit allemaal. BF16 heeft het bereik van FP32 en heeft er niets van nodig.

        **Stap 4: wat je bespaart.** Veel van de activations die de backward pass nodig heeft, worden in BF16 bewaard, wat geheugen bespaart, al is het minder dan de helft: een deel blijft FP32. Op een GPU met tensor cores draaien de matrixvermenigvuldigingen meerdere keren sneller. De weights, hun gradients en de gemiddelden van de optimizer blijven FP32, even groot als eerst, en de BF16-kopieën van de weights komen erbij; de volgende les telt het allemaal op.

        Het plaatje: één trainingsstap in mixed precision.
        """),
        mo.Html(C.flow_picture({"title": "Eén stap in mixed precision", "weights": "master weights, FP32", "copy": "een BF16-kopie voor elke matrixvermenigvuldiging",
                                "forward": "forward pass: matrixvermenigvuldigingen in BF16", "loss": "loss, FP32",
                                "backward": "backward pass, elke bewerking in het formaat van zijn forward", "grads": "gradients van de weights, FP32",
                                "update": "stap van de optimizer op de master weights, FP32",
                                "alt": "Zeven vakken van boven naar beneden: master weights in FP32; een BF16-kopie voor elke matrixvermenigvuldiging; de forward pass met de matrixvermenigvuldigingen in BF16; de loss in FP32; de backward pass, elke bewerking in het formaat van zijn forward; de gradients van de weights in FP32; de stap van de optimizer op de master weights in FP32."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **mixed precision** | trainen met de berekeningen in 16 bits en de master weights in FP32 | |
        | **master weights** | de FP32-weights die de optimizer bijwerkt; de 16-bits getallen zijn kopieën | |
        | **autocast** | de schakelaar van PyTorch die de bewerkingen in een blok in een lager formaat draait waar dat veilig is | `with torch.autocast(device_type, dtype=torch.bfloat16):` |
        | **loss scaling** | de loss vóór `backward()` met een grote factor vermenigvuldigen, en de gradients vóór de stap erdoor delen, zodat kleine FP16-gradients geen 0 worden | `torch.amp.GradScaler()` |
        """),
        L.watch_out("Het model zelf omzetten naar BF16, met `model.to(torch.bfloat16)`, is geen mixed precision: dan zijn ook de weights die de optimizer bijwerkt BF16, en gaan kleine updates verloren."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo train je in mixed precision:

    1. Laat het model en de optimizer zoals ze zijn, in FP32.
    2. Zet de forward pass en de loss binnen `torch.autocast` met `dtype=torch.bfloat16`.
    3. Roep `backward()` en de stap van de optimizer na het blok aan, zoals eerst.
    4. Met FP16 in plaats daarvan voeg je een `GradScaler` toe voor de loss scaling.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Autocast is nieuw: één lineaire layer als model, een forward hook zoals in [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl) die meldt in welk formaat de layer rekent, en de formaten van de loss, de weights en hun gradients.
    """)
    return


@app.cell
def example_autocast():
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    _model = torch.nn.Linear(4, 3)
    _model.register_forward_hook(lambda m, i, o: print("de layer rekent in", o.dtype))
    _x, _y = torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _logits = _model(_x)
        _loss = F.cross_entropy(_logits, _y)
    _loss.backward()
    print("logits", _logits.dtype, " loss", _loss.dtype)
    print("weights", _model.weight.dtype, " gradients", _model.weight.grad.dtype)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De layer rekent in BF16, dus de logits zijn BF16; de cross-entropy-loss is FP32, omdat autocast die in FP32 draait. De weights en hun gradients zijn FP32. Op een NVIDIA-GPU schrijf je `device_type="cuda"`, op een Mac `"mps"`; de rest blijft gelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een stap in mixed precision

        Maak `train_step(model, opt, x, y, device_type)` af: één trainingsstap met de forward pass en de cross-entropy-loss onder autocast in BF16, en het model gewoon in FP32. Hij geeft de loss terug als getal. De check draait hem op één lineaire layer, kijkt in welk formaat de layer rekent, in welk formaat de weights staan en welke loss hij teruggeeft, en probeert een kleine update en twee stappen achter elkaar.
        """),
        L.task("Maak `train_step(model, opt, x, y, device_type)` af: de forward pass en de loss onder autocast in BF16, het model in FP32."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("train-step", lines=11)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("train-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "train-step",
        [
            "De twee regels van de forward pass en de loss komen ingesprongen onder één `with`-regel, zoals in het uitgewerkte voorbeeld, met `device_type` in plaats van `\"cpu\"`.",
        ],
        why="Het `with`-blok omvat de forward pass en de loss, waar autocast per bewerking het formaat kiest; `backward()` en `opt.step()` blijven erbuiten, en het model wordt nooit omgezet.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### FP32 tegen mixed precision op je machine

    De knop traint een MLP van 784, 1024, 1024 en 10 op de 6000 cijfers van de course, drie epochs, twee keer vanaf hetzelfde begin: één keer gewoon in FP32, één keer met jouw `train_step` (zolang oefening 1 niet goed is, die van de course). Of mixed precision sneller is, hangt af van je hardware. Een GPU met snelle BF16, zoals een NVIDIA-GPU van 2020 of later, doet de vermenigvuldigingen op zijn tensor cores: hoe groter de matrices, hoe groter de winst, en deze zijn klein. Een CPU zonder snelle BF16, zoals in veel laptops, moet BF16 nadoen met andere instructies, en kan er vele keren zo lang over doen. Het duurt van een paar seconden tot een paar minuten.
    """)
    return


@app.cell(hide_code=True)
def compare_button_cell():
    compare_button = mo.ui.run_button(label="Train allebei")
    compare_button
    return (compare_button,)


@app.cell(hide_code=True)
def demo_compare(box_step, compare_button, result_step):
    if not compare_button.value:
        _out = mo.md("*Druk op de knop om allebei te trainen.*")
    else:
        _step, _yours = C.train_step_of(box_step.value if result_step.state == "correct" else None)
        _runs = {"fp32": "FP32", "mixed": "mixed precision"}
        _r = C.compare(_step, show=lambda _run, _i, _n: mo.output.replace(mo.md(f"*{_runs[_run]}: epoch {_i + 1} van {_n}*")))
        _where = {"cuda": "je GPU", "mps": "de GPU van je Mac", "cpu": "je CPU"}[_r["device"]]
        _n = lambda _x, _d: C.number(_x, _d, L.lang)
        _who = "jouw `train_step`" if _yours else "de `train_step` van de course"
        _f, _m = _r["fp32"], _r["mixed"]
        _ratio = _f["seconds"] / _m["seconds"]
        _speed = (f"Hier is mixed precision {_n(_ratio, 1)} keer zo snel als FP32." if _ratio >= 1.1 else
                  f"Hier duurt mixed precision {_n(1 / _ratio, 1)} keer zo lang als FP32." if _ratio <= 1 / 1.1 else
                  "Hier zijn ze ongeveer even snel.")
        _moved = ("Je GPU rekent onder autocast niet in BF16 (op een Mac is daar macOS 14 of later voor nodig), dus beide runs draaien op je CPU. "
                  if _r["moved"] else "")
        _out = mo.md(f"""
    | | FP32 | mixed precision |
    |---|---|---|
    | seconden per epoch | {_n(_f['seconds'], 2)} | {_n(_m['seconds'], 2)} |
    | testcijfers goed | {_n(100 * _f['accuracy'], 1)}% | {_n(100 * _m['accuracy'], 1)}% |
    | eerste layer rekent in | {_f['calculates']} | {_m['calculates']} |
    | weights na het trainen | {_f['weights']} | {_m['weights']} |

    {_moved}Op {_where}, met {_who}. {_speed} Beide runs beginnen met dezelfde weights en krijgen de mini-batches in dezelfde volgorde; de accuracy verschilt een beetje, omdat BF16 de berekeningen afrondt. De tijd per epoch laat de eerste epoch weg, die ook opwarmt.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_stays():
    mo.md(r"""
    ### Oefening 2. Wat blijft FP32?
    """)
    return


@app.cell(hide_code=True)
def choice_stays_cell():
    choice_stays = L.choice(
        "what-stays",
        question="Je traint met autocast in BF16 en AdamW. Welke getallen blijven de hele tijd FP32?",
        options={
            "acts": "De activations die de backward pass nodig heeft",
            "right": "De weights en de twee gemiddelden van AdamW",
            "all": "Geen: autocast zet alle getallen om naar BF16",
        },
        correct="right",
        why={
            "acts": "Die komen uit BF16-vermenigvuldigingen en worden in BF16 bewaard: daar komt de geheugenbesparing vandaan.",
            "all": "Autocast verandert alleen de berekeningen binnen zijn blok; de weights en de gemiddelden van AdamW worden nooit omgezet.",
            "right": "De master weights en de gemiddelden van AdamW blijven FP32, dus elke update landt in fijne stappen. Ze krimpen niet: de volgende les telt hun geheugen.",
        },
    )
    choice_stays
    return (choice_stays,)


@app.cell(hide_code=True)
def text_exercise_scaling():
    mo.md(r"""
    ### Oefening 3. Waarom de loss schalen?
    """)
    return


@app.cell(hide_code=True)
def explain_scaling():
    explain_scaling_cell = L.explain(
        "scaling",
        question="In een backward pass in FP16 zou een gradient $10^{-8}$ moeten worden. Wat gebeurt ermee zonder loss scaling, en wat verandert een schaal van $S = 65.536$? Waarom heeft BF16 dit niet nodig?",
        model_answer="**Zonder schalen:** FP16 maakt van getallen onder ongeveer $3 \\times 10^{-8}$ een 0, en houdt weinig cijfers onder ongeveer $6 \\times 10^{-5}$, dus $10^{-8}$ wordt 0: dat deel van de gradient gaat verloren. **Met $S = 65.536$:** de loss wordt vóór `backward()` met $S$ vermenigvuldigd, dus elke gradient is $S$ keer groter: $10^{-8} \\times 65.536 \\approx 6{,}6 \\times 10^{-4}$, ruim binnen het bereik van FP16. Vóór de stap worden de gradients weer door $S$ gedeeld, in FP32. Loopt een gradient over naar `inf` of wordt hij `nan`, dan wordt de stap overgeslagen en $S$ gehalveerd. **BF16** heeft de 8 exponentbits van FP32, dus $10^{-8}$ past er gewoon in.",
        points=[
            "je zegt dat de gradient zonder schalen 0 wordt, onder het bereik van FP16",
            "je zegt dat de loss met $S$ schalen elke gradient $S$ keer groter maakt, hier ongeveer $6{,}6 \\times 10^{-4}$, en dat hij vóór de stap terug wordt gedeeld",
            "je zegt dat BF16 het bereik van FP32 heeft, dus geen schalen nodig heeft",
        ],
    )
    explain_scaling_cell
    return (explain_scaling_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De formaten komen uit [Minder bits: FP32, FP16 en BF16](/course/open/sys-number-formats?lang=nl), de trainingsstap uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl). De volgende les, *Waar het geheugen heen gaat*, telt op wat een trainingsrun in het geheugen houdt: de weights, hun gradients, wat de optimizer ervoor bijhoudt en de activations, en hoeveel mixed precision daarvan bespaart.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stays, explain_scaling_cell, result_step):
    L.wrap_up(result_step, choice_stays, explain_scaling_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
