# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-torch-compile. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.
# termen:allow graph

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-torch-compile", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je eigen GPU kernel schrijven](/course/open/sys-triton-kernel?lang=nl) en [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kernel():
    quick_kernel = L.choice(
        "quick-kernel",
        question="Wat is een GPU kernel?",
        options={
            "core": "Een van de cores van een GPU",
            "right": "Een functie die op de GPU draait, in veel programma's naast elkaar",
            "driver": "De driver waarmee PyTorch met de GPU praat",
        },
        correct="right",
        why={
            "core": "Een kernel is code, geen hardware: één launch draait hem in veel programma's over de cores. Zie [Je eigen GPU kernel schrijven](/course/open/sys-triton-kernel?lang=nl).",
            "driver": "Een kernel is de code van één operatie, zoals je fused softmax. Zie [Je eigen GPU kernel schrijven](/course/open/sys-triton-kernel?lang=nl).",
            "right": "Elke operatie van PyTorch die op een GPU rekent, start er minstens één. In deze les schrijft een compiler fused kernels voor je.",
        },
    )
    quick_kernel
    return (quick_kernel,)


@app.cell(hide_code=True)
def quick_check_bound():
    quick_bound = L.choice(
        "quick-bound",
        question="Wat remt de snelheid van `x * 2 + 1` op een tensor van 100 miljoen getallen af?",
        options={
            "right": "Het geheugenverkeer: elk getal lezen en schrijven",
            "flops": "De FLOPs: twee per getal",
            "launch": "Het starten van de kernel",
        },
        correct="right",
        why={
            "flops": "Twee FLOPs per 8 verplaatste bytes ligt ver onder wat een GPU of CPU per byte kan rekenen. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "launch": "Een kernel starten kost microseconden; 800 MB verplaatsen duurt veel langer. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "right": "Het is memory-bound. Als twee aparte operaties sturen ze de tensor zelfs twee keer door het geheugen: ze samenvoegen is wat een compiler doet.",
        },
    )
    quick_bound
    return (quick_bound,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je fused softmax was sneller dan de vijf operaties van PyTorch, maar voor elke keten van operaties in een model een kernel schrijven is veel werk. **`torch.compile`** doet het voor je: één regel, en PyTorch traceert je model, voegt samen wat kan, en schrijft de kernels. Het kan het trainen duidelijk sneller maken, en het kan ook niets opleveren. Als je weet wat het doet, weet je wanneer je het gebruikt, en hoe je meet of het geholpen heeft.
    """)
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-call",
        question="Je compileert een kleine functie met `torch.compile`, en roept hem één keer aan. Vergeleken met dezelfde aanroep zonder compileren is die ene aanroep…",
        options={
            "fast": "Sneller: gecompileerde code is snel",
            "same": "Ongeveer even snel",
            "right": "Veel trager",
        },
        reveal=f"**Veel trager:** de eerste aanroep compileert. Voor een functie van drie elementsgewijze operaties en een som op 4 miljoen getallen kostte dat op de CPU waarop de course gemaakt is ongeveer {L.number(C.COMPILE_S, 0)} seconden; daarna kostte een aanroep {L.number(C.COMPILED_S * 1000, 1)} ms in plaats van {L.number(C.EAGER_S * 1000, 1)} ms. Compileren verdient zich alleen terug als de functie vaak draait.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: traceren.** Bij de eerste aanroep leest **Dynamo**, de voorkant van `torch.compile`, je Python-functie terwijl die draait, en legt de operaties van PyTorch vast in een graaf. Het voegt **guards** toe: controles op de inputs, zoals hun shapes en getaltypes. Bij latere aanroepen gebruikt het de gecompileerde code opnieuw zolang de guards kloppen; als er een faalt, bijvoorbeeld bij een nieuwe shape, compileert het opnieuw.

        **Stap 2: compileren.** **Inductor**, de achterkant, neemt de graaf, voegt de operaties samen die samen kunnen, en schrijft de kernels: in Triton op een NVIDIA-GPU, in C++ op een CPU. Dat is het trage deel van de eerste aanroep.

        **Stap 3: graph breaks.** Wat Dynamo niet in een graaf kan zetten, laat het aan Python over: een `print`, `.item()` dat een getal uit een tensor haalt, een `if` op de waarde van een tensor, een library die geen PyTorch is. Daar breekt de graaf in tweeën, en over de breuk heen wordt niets samengevoegd. `torch._dynamo.explain(fn)(x)` traceert een functie, zonder te compileren, en telt de grafen en de breuken.

        **Stap 4: verdient het zich terug?** Een gecompileerde aanroep is $t_\text{eager} - t_\text{compiled}$ sneller; compileren kost één keer $t_\text{compile}$. Het heeft zich terugverdiend na
        """),
        L.formula(r"n^* = \left\lceil \frac{t_\text{compile}}{t_\text{eager} - t_\text{compiled}} \right\rceil \text{ aanroepen}"),
        mo.md(r"""
        en nooit als een gecompileerde aanroep niet sneller is. Een trainingsrun van 100.000 stappen met dezelfde shape verdient het makkelijk terug; een script dat één keer draait niet.

        De figuur: de totale tijd tegen het aantal aanroepen, voor de meting van de voorspelling: zonder compileren een lijn door 0; gecompileerd een lijn die bij de compileertijd begint maar minder steil stijgt. Ze kruisen bij $n^*$.
        """),
        mo.Html(C.calls_picture({"title": "Wanneer compileren loont", "x": "aanroepen", "y": "seconden in totaal",
                                 "eager": "zonder compileren", "compiled": "gecompileerd", "even": "n* ≈ {n}", "lang": L.lang,
                                 "alt": "Twee lijnen van de totale tijd tegen 0 tot 2.100 aanroepen. Zonder compileren, gestreept, stijgt hij van 0 tot ongeveer 47 seconden. Gecompileerd, doorgetrokken, begint hij bij 16 seconden en stijgt hij tot ongeveer 22. Ze kruisen bij ongeveer 800 aanroepen."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **`torch.compile`** | | traceert een model of functie en compileert het tot fused kernels, bij de eerste aanroep | `torch.compile(model)` |
        | **graph break** | | een plek waar het traceren stopt en Python het overneemt, zodat de graaf in tweeën breekt | `torch._dynamo.explain(fn)(x)` |
        | **guard** | | een controle op de inputs waaronder de gecompileerde code opnieuw gebruikt mag worden | |
        | `torch.where` | torch where | getal voor getal: `a` waar de voorwaarde `c` klopt, anders `b`; één graaf, geen `if` | `torch.where(c, a, b)` |
        | $n^*$ | n ster | het aantal aanroepen waarna het compileren zich heeft terugverdiend | |
        """),
        L.watch_out("Meet een gecompileerde functie pas na zijn eerste aanroep, en op een GPU met `torch.cuda.synchronize()` voordat je de klok afleest, zoals in [Je eigen GPU kernel schrijven](/course/open/sys-triton-kernel?lang=nl). En let op opnieuw compileren: een nieuwe shape laat een guard falen. Een model dat teksten van elke lengte krijgt, compileert bij de tweede lengte opnieuw, nu voor elke lengte, en code voor elke lengte kan trager zijn dan code voor één lengte; vul aan tot een paar vaste lengtes, of geef `dynamic=True` mee om bij de eerste aanroep al voor elke lengte te compileren."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om uit te zoeken of `torch.compile` helpt:

    1. Tel de graph breaks met `torch._dynamo.explain`, en haal die weg in het deel van de code dat het vaakst draait.
    2. Time een aanroep zonder compileren, na een aanroep om op te warmen.
    3. Compileer, time de eerste aanroep, die compileert, en time dan latere aanroepen.
    4. Reken $n^*$ uit, en vergelijk het met hoe vaak de code zal draaien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 op een functie met een `print` in het midden: `torch._dynamo.explain` traceert hem, zonder te compileren, en vertelt hoeveel grafen het vond en waarom de graaf brak.
    """)
    return


@app.cell
def example_explain():
    import torch
    import torch._dynamo as dynamo

    def _halves(x):
        _y = torch.relu(x) * 2
        print("halverwege")
        return _y + 1

    _e = dynamo.explain(_halves)(torch.tensor([-1.0, 2.0]))
    print("grafen:", _e.graph_count, "breuken:", _e.graph_break_count)
    _why = str(_e.break_reasons[0].reason).splitlines()
    print(_why[0])
    print(_why[1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `explain` draait de functie, dus wordt "halverwege" afgedrukt. Dynamo kon de `print` niet in een graaf zetten, zoals de tweede regel van de reden zegt: het maakte één graaf ervoor en één erna, met één breuk. De ReLU en het maal 2 kunnen nog samengevoegd worden, en het plus 1 staat apart; over de breuk heen wordt niets samengevoegd. Zonder de `print` is het één graaf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_even():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Wanneer het loont

        Maak `break_even(compile_s, eager_s, compiled_s)` af: geef met de compileertijd en de seconden van één aanroep zonder en met compileren $n^*$ terug, naar boven afgerond, of `None` als een gecompileerde aanroep niet sneller is. De check roept `break_even(30.0, 0.05, 0.02)` aan, wat `1000` moet geven, en probeert de meting van de figuur, een deling die precies uitkomt, gelijke tijden, en een tragere gecompileerde aanroep.
        """),
        L.task("Laat `break_even` het aantal aanroepen geven waarna het compileren zich heeft terugverdiend."),
    ])
    return


@app.cell(hide_code=True)
def exercise_even():
    box_even = L.code_box("break-even", lines=8)
    box_even
    return (box_even,)


@app.cell(hide_code=True)
def check_even(box_even):
    result_even = L.check("break-even", box_even.value)
    result_even
    return (result_even,)


@app.cell(hide_code=True)
def hints_even():
    L.hints(
        "break-even",
        [
            "Eerst het geval waarin een gecompileerde aanroep niet sneller is; dan de formule van stap 4, met `math.ceil` om naar boven af te ronden.",
        ],
        why="`if compiled_s >= eager_s: return None`, en dan `return math.ceil(compile_s / (eager_s - compiled_s))`. Voor het voorbeeld: 30 / 0,03 = 1.000 aanroepen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_graph():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén graaf

        `one_graph(x)` deelt `x` door zijn grootste absolute waarde als die boven 1 ligt, en neemt de ReLU. Zoals hij er staat, breekt hij de graaf: `.item()` haalt het maximum als Python-getal uit de graaf, en de `if` beslist dan in Python. Schrijf hem zo om dat hij dezelfde uitkomst geeft, als één graaf: houd het maximum een tensor, en laat `torch.where(voorwaarde, a, b)` kiezen: dat neemt `a` waar de voorwaarde klopt, en anders `b`. De check roept `one_graph(torch.tensor([0.5, -2.0, 1.5, 3.0]))` aan, wat `[0.1667, 0.0, 0.5, 1.0]` moet geven, probeert andere tensors, en telt de graph breaks met `torch._dynamo.explain`.
        """),
        L.task("Laat `one_graph` dezelfde uitkomst geven zonder graph break."),
    ])
    return


@app.cell(hide_code=True)
def exercise_graph():
    box_graph = L.code_box("one-graph", lines=9)
    box_graph
    return (box_graph,)


@app.cell(hide_code=True)
def check_graph(box_graph):
    result_graph = L.check("one-graph", box_graph.value)
    result_graph
    return (result_graph,)


@app.cell(hide_code=True)
def hints_graph():
    L.hints(
        "one-graph",
        [
            "`m = x.abs().max()` blijft een tensor met één getal; `m > 1` is dan ook een tensor, en `torch.where` neemt die als voorwaarde.",
        ],
        why="`m = x.abs().max()` en `return torch.relu(torch.where(m > 1, x / m, x))`. Voor het voorbeeld is het maximum 3: de vector wordt door 3 gedeeld, en de ReLU maakt −0,667 nul.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Compileren en timen

    De knop compileert je `one_graph` zodra die goed is, anders die van de course, en timet hem op 4 miljoen getallen, op je NVIDIA-GPU als je die hebt: de eerste aanroep, die compileert, en dan een aanroep zonder en met compileren. Compileren kan tot een minuut duren; PyTorch bewaart wat het compileerde in een cache op schijf, dus de volgende keer is de eerste aanroep veel sneller. Op een computer zonder C++-compiler werkt het niet.
    """)
    return


@app.cell(hide_code=True)
def compile_button_cell():
    compile_button = mo.ui.run_button(label="Compileer en time")
    compile_button
    return (compile_button,)


@app.cell(hide_code=True)
def demo_compile(box_graph, compile_button, result_graph):
    if not compile_button.value:
        _out = mo.md("*Druk op de knop om te compileren en te timen.*")
    else:
        _d = C.demo(box_graph.value if result_graph.state == "correct" else None)
        if _d == "error":
            _out = mo.md("*Compileren werkte niet op deze computer: `torch.compile` heeft op een CPU een C++-compiler nodig, of een NVIDIA-GPU met Triton.*")
        else:
            _where = "je GPU" if _d["device"] == "cuda" else "je CPU"
            _who = "je `one_graph`" if _d["yours"] else "de `one_graph` van de course"
            _gain = _d["eager"] / _d["compiled"]
            _out = mo.md(
                f"Op {_where}, {_who}: de eerste aanroep kostte {L.number(_d['first'], 1)} s; daarna kostte een aanroep "
                f"{L.number(_d['eager'] * 1000, 2)} ms zonder compileren en {L.number(_d['compiled'] * 1000, 2)} ms gecompileerd, "
                f"{L.number(_gain, 1)} keer zo snel. De uitkomsten zijn {'gelijk' if _d['same'] else 'verschillend'}."
            )
    _out
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Oefening 3. Wanneer compileren
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "when-compile",
        question="Welke van deze drie wint het meest met `torch.compile`?",
        options={
            "once": "Een script dat een model één keer op één input draait",
            "right": "Een trainingsrun van 100.000 stappen op batches van één shape",
            "lengths": "Een loop die bij elke aanroep een tekst van een andere lengte krijgt",
        },
        correct="right",
        why={
            "once": "Eén aanroep verdient de compileertijd nooit terug: $n^*$ ligt ver boven 1.",
            "lengths": "Een nieuwe lengte laat een guard falen, dus compileert hij nog een keer, nu voor elke lengte: dat kost extra tijd, en code voor elke lengte kan trager zijn dan code voor één shape.",
            "right": "Eén keer compileren, dan 100.000 aanroepen met dezelfde guards: ver voorbij $n^*$.",
        },
    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het samenvoegen dat je met de hand deed in [Je eigen GPU kernel schrijven](/course/open/sys-triton-kernel?lang=nl), nu gedaan door een compiler, die op een NVIDIA-GPU zijn eigen Triton-kernels schrijft; wat het bespaart, is het geheugenverkeer van [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl). Compileer in een eigen trainingsrun het model één keer, controleer de graph breaks, en time een paar stappen na de eerste.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_even, result_graph):
    L.wrap_up(result_even, result_graph, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
