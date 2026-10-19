# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-triton-kernel. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-triton-kernel", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl) en [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_small():
    quick_small = L.choice(
        "quick-small",
        question="In een profiel draaien optellingen en ReLU's duizenden keren en nemen ze de helft van de tijd van een stap, hoewel ze bijna geen FLOPs doen. Wat maakt ze traag?",
        options={
            "flops": "Ze doen meer FLOPs dan de profiler laat zien",
            "right": "Ze lezen en schrijven elk getal voor ongeveer één FLOP per stuk",
            "count": "De profiler telt hun tijd dubbel",
        },
        correct="right",
        why={
            "flops": "Een optelling doet één FLOP per getal, en de profiler meet de bewerking terwijl ze draait. Zie [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl).",
            "right": "Ze zijn memory-bound: de cores wachten tot de getallen uit het geheugen komen en er weer heen gaan.",
            "count": "De profiler telt de tijd van elke bewerking één keer, los van de rest. Zie [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl).",
        },
    )
    quick_small
    return (quick_small,)


@app.cell(hide_code=True)
def quick_check_max():
    quick_max = L.choice(
        "quick-max",
        question="Een stabiele softmax trekt het grootste getal van een rij van elk getal af voordat hij de machten van $e$ neemt. Waarom?",
        options={
            "sum": "Zodat de uitkomst optelt tot 1",
            "fast": "Zodat hij sneller rekent",
            "right": "Zodat de machten niet kunnen overlopen, terwijl de uitkomst gelijk blijft",
        },
        correct="right",
        why={
            "sum": "Delen door de som laat hem optellen tot 1, met of zonder de verschuiving. Zie [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).",
            "fast": "De verschuiving is één aftrekking meer, niet minder werk. Zie [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl).",
            "right": "De grootste macht wordt $e^0 = 1$, en de factor $e^{-m}$ valt weg als je door de som deelt.",
        },
    )
    quick_max
    return (quick_max,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De profiler liet kleine bewerkingen zien die op het geheugen wachten. Geschreven als bewerkingen van PyTorch is een softmax er vijf: het grootste getal van elke rij, een aftrekking, de machten van $e$, hun som, en een deling. Elk ervan leest zijn input uit het geheugen van de GPU en schrijft zijn output terug. Eén programma dat alle vijf op een rij doet, terwijl de rij op de chip blijft, leest hem één keer en schrijft hem één keer. De eigen `torch.softmax` van PyTorch is zo'n programma. Voor een combinatie waarvoor niemand er een schreef, schrijf je je eigen **GPU kernel**: in Triton, in Python.
    """)
    return


@app.cell(hide_code=True)
def predict_fused():
    guess_fused = L.guess(
        "fused-speed",
        question="Over een grote matrix lezen de vijf bewerkingen van een softmax de hele matrix 5 keer en schrijven ze hem 3 keer. Eén samengevoegde kernel leest hem één keer en schrijft hem één keer, en doet dezelfde FLOPs. Hoeveel sneller kan de samengevoegde kernel zijn op een GPU, waar softmax op het geheugen wacht?",
        options={
            "same": "Nauwelijks sneller: de FLOPs zijn hetzelfde",
            "right": "Tot ongeveer 4 keer zo snel",
            "hundred": "Tot ongeveer 100 keer zo snel",
        },
        reveal="**Tot ongeveer 4 keer zo snel.** Een memory-bound bewerking duurt ongeveer zo lang als het verplaatsen van haar bytes: 8 ritjes van de matrix tegen 2. Sneller dan de input één keer lezen en de output één keer schrijven kan niet. De demo hieronder meet het op je machine.",
    )
    guess_fused
    return (guess_fused,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: een kernel en zijn grid.** Een GPU draait een kernel als heel veel kopieën tegelijk, elk op een eigen stuk van de data. In CUDA, de manier van NVIDIA om zijn GPU's te programmeren, zijn de kopieën threads, gegroepeerd in **thread blocks**, en een launch start een grid van blocks. In Triton schrijf je wat één block doet, op een heel block getallen tegelijk, en Triton verdeelt dat werk over de threads. Elke kopie die draait is een programma: `tl.program_id(0)` zegt welk het is, en de launch `kernel[grid](...)` zegt hoeveel er zijn.

        **Stap 2: pointers, plekken en een mask.** Een tensor komt een kernel binnen als pointer: de plek in het geheugen van zijn eerste getal, zoals `data_ptr()` hem geeft. `x_ptr + offsets` zijn de plekken van een block getallen, `tl.load` leest ze en `tl.store` schrijft ze. De grootte van een block is een macht van twee die Triton kent als hij compileert, een `tl.constexpr`: de def-regel zegt dat met `BLOCK: tl.constexpr`. Waar een block voorbij het eind van de data reikt, zet de mask plekken uit: `tl.load(..., mask=mask, other=v)` geeft daar `v`, en `tl.store(..., mask=mask)` schrijft daar niets.

        **Stap 3: samenvoegen.** Tussen `tl.load` en `tl.store` blijven de getallen op de chip, en kost elke bewerking daartussen geen ritje naar het geheugen. Op de chip werken `tl.max(x, axis=0)`, `tl.exp(x)` en `tl.sum(x, axis=0)` op een block zoals `max`, `exp` en `sum` op een NumPy-array. Meerdere bewerkingen in één kernel heet **operator fusion**. Een samengevoegde softmax geeft elke rij een eigen programma: dat laadt de rij, trekt het grootste getal ervan af, neemt de machten van $e$, deelt ze door hun som, en slaat de rij op. Twee ritjes in plaats van acht.

        **Stap 4: zonder NVIDIA-GPU.** Met `TRITON_INTERPRET=1` draait Triton een kernel in Python op de CPU: goed, maar traag. Deze les draait het uitgewerkte voorbeeld en de checks altijd zo, op CPU-tensors, zodat elke check op elke computer met Triton hetzelfde antwoord geeft. Om de snelheid van een kernel op een GPU te meten, meet je hem zoals in de vorige les, en deel je de bytes die hij leest en schrijft door de tijd: GB per seconde, om te vergelijken met de geheugenbandbreedte van de GPU.

        Het plaatje: de ritjes naar het geheugen van een softmax, als vijf bewerkingen en als één kernel.
        """),
        mo.Html(C.traffic_picture({"title": "Ritjes naar het geheugen", "memory": "geheugen van de GPU", "five": "5 keer lezen, 3 keer schrijven",
                                   "fused": "één kernel", "one": "1 keer lezen, 1 keer schrijven", "lang": L.lang,
                                   "alt": "Boven vijf vakjes voor max, aftrekken, exp, som en delen, elk met een pijl uit het geheugen van de GPU, en drie met een pijl terug: 5 keer lezen en 3 keer schrijven. Onder één vakje voor de samengevoegde kernel, met één pijl uit het geheugen en één terug."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **GPU kernel** | een kleine functie die in heel veel kopieën tegelijk op de GPU draait, elk op een eigen stuk van de data | `@triton.jit` |
        | **thread block** | in CUDA een groep threads die samen draaien; in Triton één programma, dat aan een block data werkt | `tl.program_id(0)` |
        | **operator fusion** | meerdere bewerkingen in één kernel, zodat de data één keer uit het geheugen komt | |
        | **CUDA** | het platform van NVIDIA om zijn GPU's te programmeren; in PyTorch is een NVIDIA-GPU het device `"cuda"` | |
        | **Triton** | een taal met compiler, te gebruiken vanuit Python, om GPU kernels te schrijven | `import triton` |
        """),
        L.watch_out("Een kernel kan elke plek in het geheugen lezen en schrijven, ook voorbij het eind van een tensor: niets houdt hem tegen. Zonder de mask leest een kernel getallen die niet van hem zijn, of beschadigt hij ze, en kan het programma crashen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo schrijf je een kernel die op rijen werkt:

    1. Geef elke rij een programma: `row = tl.program_id(0)`, en een launch met één programma per rij.
    2. Maak de plekken van de rij, `row * n_cols + tl.arange(0, BLOCK)`, en een mask voor de plekken voorbij het eind.
    3. Laad de rij één keer, met de mask, en een `other` die niets verandert aan wat volgt.
    4. Reken op de chip, en sla één keer op, met de mask.
    5. Controleer de uitkomst met PyTorch, en meet dan de snelheid.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een Triton-kernel is nieuw: een optelling en een ReLU samengevoegd, `relu(x + y)`, over vectoren van 1.000 getallen, in blocks van 256. Zonder Triton, op macOS of op Windows zonder WSL2, stopt deze cel bij `import triton`; lees dan mee.
    """)
    return


@app.cell
def example_add_relu():
    import torch
    import triton
    import triton.language as tl


    @triton.jit
    def add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        start = tl.program_id(0) * BLOCK
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, tl.maximum(x + y, 0.0), mask=mask)


    torch.manual_seed(0)
    _x, _y = torch.randn(1000), torch.randn(1000)
    _out = torch.empty_like(_x)
    add_relu_kernel[(triton.cdiv(1000, 256),)](_x, _y, _out, 1000, BLOCK=256)
    print("programma's:", triton.cdiv(1000, 256))
    print("gelijk aan PyTorch:", torch.equal(_out, torch.relu(_x + _y)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `torch.empty_like(_x)` maakt een tensor met dezelfde shape voor de uitkomst, zonder zijn getallen in te vullen: dat doet de kernel. `triton.cdiv(1000, 256)` deelt en rondt naar boven af: 4 programma's. Programma 3 begint bij plek 768, en zijn block reikt tot 1.023; de mask zet de 24 plekken vanaf 1.000 uit. Elk programma laadt zijn stuk van `x` en van `y`, telt ze op en neemt de ReLU op de chip, en slaat de uitkomst op: `x + y` gaat nooit naar het geheugen, waar `torch.relu(x + y)` van PyTorch het wegschrijft en terugleest. `torch.equal` zegt of twee tensors precies gelijk zijn: de uitkomst is gelijk aan die van PyTorch, tot op het laatste bit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kernel():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Een samengevoegde softmax

        Maak `softmax_kernel` af: het programma van één rij van `x`, met `n_cols` getallen, dat de softmax van die rij naar dezelfde rij van `out` schrijft. De functie `softmax` eronder start één programma per rij, met als `BLOCK` de macht van twee uit `triton.next_power_of_2(n_cols)`. De check start je kernel met een eigen kopie van `softmax`, die een hek om de tensors zet, om te zien of je kernel erbinnen blijft. Hij roept `softmax(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))` aan, wat ongeveer `[[0.09, 0.2447, 0.6652], [0.3333, 0.3333, 0.3333]]` moet geven, en probeert rijen van andere lengtes en getallen van duizend.
        """),
        L.task("Maak `softmax_kernel` af: laad de rij één keer met de mask, reken stabiel zijn softmax uit, en sla hem één keer op met de mask."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kernel():
    box_kernel = L.code_box("softmax-kernel", lines=22)
    box_kernel
    return (box_kernel,)


@app.cell(hide_code=True)
def check_kernel(box_kernel):
    result_kernel = L.check("softmax-kernel", box_kernel.value)
    result_kernel
    return (result_kernel,)


@app.cell(hide_code=True)
def hints_kernel():
    L.hints(
        "softmax-kernel",
        [
            "Stap 2 tot 4 van het recept, met het uitgewerkte voorbeeld als vorm. Voor `other`: welk getal verandert, op de plekken voorbij het eind van de rij, niet het grootste getal en niet de som van de machten?",
        ],
        why="`float(\"-inf\")` voorbij het eind wint nooit het maximum, en zijn macht, $e^{-\\infty} = 0$, voegt niets toe aan de som. Het maximum aftrekken voorkomt dat `tl.exp` overloopt, en `tl.max` en `tl.sum` met `axis=0` werken langs het block.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Drie softmaxen, gemeten

    De knop meet de softmax over de rijen van een matrix van 1.024 kolommen, meer dan de caches van je chip bevatten: 8.192 rijen, 32 MB, op een CPU, en 65.536 rijen, 256 MB, op een NVIDIA-GPU: als de vijf bewerkingen van PyTorch, als de eigen `torch.softmax` van PyTorch, en als je kernel, gecompileerd voor je GPU als je een NVIDIA-GPU hebt. Op een CPU draait de interpreter van Triton je kernel in Python, dus krijgt die een kleinere matrix, 64 × 1.024, en zegt zijn tijd niets over een GPU. De GB per seconde tellen één keer lezen en één keer schrijven van de matrix. Het duurt een paar seconden; hij werkt zodra oefening 1 goed is.
    """)
    return


@app.cell(hide_code=True)
def time_button_cell():
    time_button = mo.ui.run_button(label="Meet de drie softmaxen")
    time_button
    return (time_button,)


@app.cell(hide_code=True)
def demo_time(box_kernel, result_kernel, time_button):
    if result_kernel.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 af: deze cel gebruikt je kernel.*")
    elif not time_button.value:
        _out = mo.md("*Druk op de knop om te meten.*")
    else:
        _m = C.demo(box_kernel.value)
        if _m == "error":
            _out = mo.md("*Je kernel stopte met een fout op deze machine.*")
        else:
            _n = lambda _x, _d=1: C.number(_x, _d, L.lang)
            _names = {"unfused": "vijf bewerkingen van PyTorch", "torch": "`torch.softmax`", "yours": "je kernel"}
            _rows = "\n".join(f"| {_names[_k]} | {_n(_s * 1000, 2)} ms | {_n(_g / _s, 1 if _g / _s >= 1 else 3)} GB/s |" for _k, _s, _g in _m["rows"])
            _speed = _m["rows"][0][1] / _m["rows"][1][1]
            _where = "je GPU" if _m["device"] == "cuda" else "je CPU"
            _note = ("" if _m["device"] == "cuda" else
                     f" Je kernel draaide in de interpreter op {_m['small'][0]} × {C.number(_m['small'][1], 0, L.lang)} getallen: zijn tijd laat zien hoe traag Python is, niet je kernel op een GPU.")
            _out = mo.md(f"""
    Op {_where}:

    | softmax | tijd | GB/s |
    |---|---|---|
    {_rows}

    De samengevoegde `torch.softmax` is {_n(_speed)} keer zo snel als de vijf bewerkingen.{_note} Je kernel geeft hetzelfde als `torch.softmax`, {C.agreement(_m['error'], L.lang)}.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_grid():
    mo.md(r"""
    ### Oefening 2. Programma's en blocks
    """)
    return


@app.cell(hide_code=True)
def choice_grid_cell():
    choice_grid = L.choice(
        "grid",
        question="Je `softmax` draait op een matrix van 512 rijen van 3.000 getallen. Hoeveel programma's start de launch, en hoeveel plekken van elk block zet de mask uit?",
        options={
            "cols": "3.000 programma's, en geen plekken",
            "right": "512 programma's, en 1.096 plekken",
            "exact": "512 programma's, en geen plekken: het block is 3.000",
        },
        correct="right",
        why={
            "cols": "De launch start één programma per rij: `(n_rows,)`, 512. Elk programma werkt aan een hele rij.",
            "exact": "De grootte van een block moet een macht van twee zijn: `triton.next_power_of_2(3000)` is 4.096.",
            "right": "Eén programma per rij, en een block van 4.096 plekken, waarvan er $4.096 - 3.000 = 1.096$ voorbij het eind van de rij liggen.",
        },
    )
    choice_grid
    return (choice_grid,)


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Waarom samengevoegd sneller is
    """)
    return


@app.cell(hide_code=True)
def explain_why_fused():
    explain_fused_cell = L.explain(
        "why-fused",
        question="Je samengevoegde softmax doet dezelfde FLOPs als de vijf bewerkingen van PyTorch, en toch is hij op een GPU sneller. Waarom? En waarom kan hij niet veel sneller worden dan ongeveer vier keer?",
        model_answer="**Waarom sneller:** softmax is memory-bound: zijn tijd gaat naar het verplaatsen van de getallen tussen het geheugen van de GPU en de chip, niet naar rekenen. De vijf bewerkingen lezen de matrix 5 keer en schrijven hem 3 keer; de samengevoegde kernel leest hem één keer en schrijft hem één keer, en houdt hem daartussen op de chip: een kwart van de bytes. **Waarom niet meer:** elke softmax moet zijn input één keer lezen en zijn output één keer schrijven. Dan verplaatst de kernel zoveel bytes per seconde als het geheugen toelaat, en alleen sneller geheugen maakt hem sneller.",
        points=[
            "je zegt dat softmax memory-bound is: zijn tijd gaat naar het verplaatsen van getallen, niet naar FLOPs",
            "je vergelijkt de ritjes naar het geheugen: 8 voor de vijf bewerkingen, 2 voor de samengevoegde kernel",
            "je zegt dat één keer lezen en één keer schrijven het minste is wat elke softmax nodig heeft, dus dat de bandbreedte de grens legt",
        ],
    )
    explain_fused_cell
    return (explain_fused_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De memory-bound bewerkingen van [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl), gevonden met de profiler van [Profiling: tokens per seconde en MFU](/course/open/sys-profiling-mfu?lang=nl), zijn nu samengevoegd tot één kernel, met de stabiele softmax van [Floating point en de log-sum-exp-truc](/course/open/num-floats-stability?lang=nl) erin. De volgende les, *RMSNorm*, begint aan de onderdelen van een moderne GPT.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_grid, explain_fused_cell, result_kernel):
    L.wrap_up(result_kernel, choice_grid, explain_fused_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
