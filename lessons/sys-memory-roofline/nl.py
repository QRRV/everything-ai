# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-memory-roofline. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-memory-roofline", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Ook van buiten de course:** twee woorden. Een **thread** is een van de vele kopieën van hetzelfde kleine programma die een GPU tegelijk draait, elk op zijn eigen getallen. Een **cache** is een klein, snel geheugen dat een kopie bewaart van data die net gebruikt is, zodat die niet opnieuw uit het tragere geheugen hoeft te komen. Is een van beide nieuw, lees dan eerst [Thread (computing)](https://en.wikipedia.org/wiki/Thread_(computing)) of [Cache (computing)](https://en.wikipedia.org/wiki/Cache_(computing)), allebei in het Engels.
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl) en [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_device():
    quick_device = L.choice(
        "quick-device",
        question="Je model staat op de GPU, en een batch staat nog op de CPU. Wat gebeurt er als je `model(batch)` aanroept?",
        options={
            "copy": "PyTorch kopieert de batch voor je naar de GPU",
            "right": "PyTorch stopt met een fout",
            "cpu": "Het model draait deze keer op de CPU",
        },
        correct="right",
        why={
            "copy": "PyTorch verplaatst nooit zelf een tensor: je verplaatst de batch met `.to(device)`. Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "cpu": "De weights blijven waar ze zijn, op de GPU, en kunnen een batch op de CPU niet ontmoeten. Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "right": "De weights en de batch moeten op hetzelfde device staan. Data tussen de twee verplaatsen is traag, en dat hoort bij deze les.",
        },
    )
    quick_device
    return (quick_device,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="Een GPT heeft 2 miljoen parameters. Ongeveer hoeveel FLOPs kost een forward pass over 1000 tokens?",
        options={
            "params": "Ongeveer 2 miljoen",
            "right": "Ongeveer 4 miljard",
            "train": "Ongeveer 12 miljard",
        },
        correct="right",
        why={
            "params": "Dat is het aantal parameters. Elk token kost ongeveer 2 FLOPs per parameter. Zie [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl).",
            "train": "Dat is een trainingsstap, forward en backward, ongeveer 6 FLOPs per parameter per token. Zie [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl).",
            "right": "Ongeveer 2 FLOPs per parameter per token: $2 \\times 2.000.000 \\times 1000 = 4 \\times 10^9$.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De specificaties van een GPU noemen twee snelheden. Een NVIDIA RTX 4090 doet 82,6 biljoen FLOPs per seconde op float32-getallen, kortweg FP32, met zijn gewone cores: 82,6 TFLOP/s. Zijn geheugen leest hij met 1,008 biljoen bytes per seconde, 1,008 TB/s. Om zijn cores bezig te houden, moet hij dus ongeveer 82 FLOPs doen voor elke byte die hij leest. Veel bewerkingen in een neural network doen er veel minder: ze zijn het grootste deel van hun tijd aan het wachten op data. Of een bewerking op het geheugen wacht of op het rekenen, bepaalt wat haar sneller maakt: minder bytes, of minder FLOPs.
    """)
    return


@app.cell(hide_code=True)
def predict_add():
    guess_add = L.guess(
        "add-vs-matmul",
        question="Op een RTX 4090 tel je twee arrays van 100 miljoen float32-getallen op, en vermenigvuldig je twee 4096 × 4096-matrices van float32, wat ongeveer 1400 keer zoveel FLOPs kost. Hoe lang duurt het optellen, vergeleken met het vermenigvuldigen, volgens de specificaties van de GPU?",
        options={
            "thousandth": "Ongeveer een duizendste zo lang",
            "tenth": "Ongeveer een tiende zo lang",
            "right": "Ongeveer even lang",
        },
        reveal="**Ongeveer even lang: 1,2 ms tegen 1,7 ms.** Het optellen doet $10^8$ FLOPs, maar verplaatst 1,2 GB: twee arrays gelezen en één geschreven, 4 bytes per getal. Met 1,008 TB/s duurt dat 1,2 ms. Het vermenigvuldigen doet $2 \\cdot 4096^3 \\approx 1{,}4 \\times 10^{11}$ FLOPs, 1,7 ms met 82,6 TFLOP/s, en verplaatst maar 0,2 GB. Dit zijn schattingen uit de specificaties; echte code haalt wat minder dan de piek.",
    )
    guess_add
    return (guess_add,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: de geheugenhiërarchie.** Een GPU bewaart data op meerdere niveaus, van klein en snel tot groot en traag:

        | Niveau | Grootte op een RTX 4090 | Wat erin staat |
        |---|---|---|
        | registers | tot ongeveer 1 KB per thread | de getallen waarmee een berekening nu werkt |
        | shared memory en L1-cache | 128 KB per streaming multiprocessor, de groep cores die het deelt | data die een groep threads opnieuw gebruikt |
        | L2-cache | 72 MB voor de hele chip | data die net gebruikt is |
        | GPU-geheugen | 24 GB, gelezen met ongeveer 1 TB/s | de weights, de activations, de batch |
        | hostgeheugen | het eigen RAM van de computer, via de PCIe-verbinding met tientallen GB/s | wat nog niet op de GPU staat |

        Deze les telt het verkeer tussen het GPU-geheugen en de chip: elk getal dat een bewerking uit het GPU-geheugen leest of erin terugschrijft.

        **Stap 2: arithmetic intensity.** De **arithmetic intensity** van een bewerking is haar aantal FLOPs gedeeld door het aantal bytes dat ze verplaatst. Twee float32-arrays optellen doet 1 FLOP per element en verplaatst 12 bytes: een intensity van $1/12$. Twee $n \times n$-matrices vermenigvuldigen doet $2n^3$ FLOPs en verplaatst $3 n^2 \cdot 4$ bytes, als elk getal één keer reist: een intensity van $n/6$, ongeveer 683 bij $n = 4096$.

        **Stap 3: de roofline.** Een bewerking kan niet sneller rekenen dan de piek, en niet sneller dan haar data aankomt: intensity maal geheugenbandbreedte. De snelheid die ze kan halen is dus
        """),
        L.formula(r"\text{FLOP/s} = \min(\text{piek},\ \text{intensity} \times \text{bandbreedte})"),
        mo.md(r"""
        Op log-log-assen is dat een schuin dak dat plat wordt: de **roofline**. Waar de twee elkaar raken, het **ridge point**, ligt bij piek gedeeld door bandbreedte: ongeveer 82 FLOPs per byte voor de RTX 4090. Een bewerking links daarvan is **memory-bound**, rechts ervan **compute-bound**. Haar tijd volgens deze schatting is de langzaamste van de twee delen, als de GPU rekent terwijl de data beweegt:
        """),
        L.formula(r"t = \max\left(\frac{\text{FLOPs}}{\text{piek}},\ \frac{\text{bytes}}{\text{bandbreedte}}\right)"),
        mo.md(r"""
        **Stap 4: je GPT.** Elementsgewijze bewerkingen, zoals ReLU en de optelling van een residual connection, en ook LayerNorm en softmax, doen een paar FLOPs per getal dat ze lezen: memory-bound. Een lineaire layer leest zijn weights één keer per batch en gebruikt ze voor elk token erin, dus groeit zijn intensity met het aantal tokens. Die groeit niet eindeloos: de inputs en outputs van elk token reizen ook, dus vlakt de intensity voor een $d \times d$-layer af bij ongeveer $d/4$. Een grote layer met een grote batch is compute-bound. Je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), met $d = 64$, blijft ver links van het ridge point van een RTX 4090, hoe groot de batch ook is: $d/4 = 16$. Met $d = 384$, zoals in de GPT van 10,8 miljoen parameters, ligt $d/4 = 96$ er net rechts van. En als een model één token tegelijk genereert, zijn zelfs de lineaire layers van een groot model memory-bound.

        Het plaatje: de roofline van de RTX 4090 bij FP32 volgens zijn specificaties, met de twee bewerkingen uit de gok.
        """),
        mo.Html(C.roof_picture({"title": "RTX 4090, FP32", "x": "FLOPs per byte", "y": "FLOP/s", "add": "optellen", "matmul": "matmul 4096", "ridge": "ridge 82", "lang": L.lang,
                                "alt": "Log-log-assen: FLOP/s tegen FLOPs per byte. Het dak stijgt langs een schuine lijn en wordt plat bij 82,6 biljoen FLOP/s, bij het ridge point van 82 FLOPs per byte. Het optellen ligt ver links op het schuine deel, bij 1/12 FLOP per byte en 84 miljard FLOP/s; de matrixvermenigvuldiging ligt op het platte deel, bij 683 FLOPs per byte."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **FLOP/s** | FLOPs per seconde: hoe snel een chip rekent, waar FLOPs tellen hoeveel er te rekenen is |
        | **geheugenhiërarchie** | de niveaus waarop een computer data bewaart, van klein en snel, zoals registers, tot groot en traag, zoals GPU-geheugen |
        | **geheugenbandbreedte** | hoeveel bytes per seconde er tussen twee niveaus kunnen bewegen, zoals van het GPU-geheugen naar de chip |
        | **arithmetic intensity** | de FLOPs van een bewerking gedeeld door de bytes die ze verplaatst |
        | **roofline** | de hoogste FLOP/s die een bewerking bij elke intensity kan halen: de bandbreedte maal de intensity, hoogstens de piek |
        | **ridge point** | de intensity waar de twee daken elkaar raken: piek gedeeld door bandbreedte |
        | **memory-bound**, **compute-bound** | begrensd door het verplaatsen van data, of door het rekenen |
        """),
        L.watch_out("De roofline is een ideaal. Hij neemt aan dat rekenen en data verplaatsen helemaal overlappen, dat elk getal maar één keer tussen het GPU-geheugen en de chip reist, en dat de bewerking de piek en de volle bandbreedte haalt. Echte code haalt minder, en kleine bewerkingen veel minder: elke bewerking op een GPU starten kost ook een paar microseconden."),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Bytes tellen is nieuw: de roofline-schatting van de 4096 × 4096-matrixvermenigvuldiging op de RTX 4090.
    """)
    return


@app.cell
def example_roofline():
    peak, bandwidth = 82.6e12, 1008e9  # RTX 4090, FP32, volgens zijn specificaties
    n = 4096
    flops = 2 * n**3  # n * n uitkomsten, elk n vermenigvuldigingen en n optellingen
    bytes_moved = 3 * n**2 * 4  # A en B één keer gelezen, C één keer geschreven, 4 bytes per float32
    print("intensity:", flops / bytes_moved, " ridge point:", peak / bandwidth)
    print("rekenen:", flops / peak, "s  geheugen:", bytes_moved / bandwidth, "s")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De intensity, 683, ligt ver rechts van het ridge point, 82: het rekenen duurt 1,7 ms en de data 0,2 ms, dus is de vermenigvuldiging compute-bound, en is de schatting 1,7 ms. Met $n = 64$ is de intensity ongeveer 11, links van het ridge point: een kleine matrixvermenigvuldiging is ook memory-bound.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_time():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De roofline-schatting

        Schrijf `roofline_time(flops, bytes_moved, peak_flops, bandwidth)`: de tijd die een bewerking volgens de roofline-schatting kost, in seconden, en wat haar begrenst, `"compute"` of `"memory"`, als `[time, limit]`. De check roept `roofline_time(1e8, 1.2e9, 82.6e12, 1008e9)` aan, het optellen uit de gok, wat ongeveer `[0.00119, "memory"]` moet geven, en probeert andere bewerkingen en GPU's.
        """),
        L.task("Schrijf `roofline_time(flops, bytes_moved, peak_flops, bandwidth)`: geef de tijd volgens de roofline-schatting terug en `\"compute\"` of `\"memory\"`, welke van de twee haar begrenst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_time():
    box_time = L.code_box("roofline-time", lines=8)
    box_time
    return (box_time,)


@app.cell(hide_code=True)
def check_time(box_time):
    result_time = L.check("roofline-time", box_time.value)
    result_time
    return (result_time,)


@app.cell(hide_code=True)
def hints_time():
    L.hints(
        "roofline-time",
        [
            "Tijd is werk gedeeld door snelheid, één keer voor de FLOPs en één keer voor de bytes. De bewerking is klaar als de langzaamste van de twee klaar is.",
        ],
        why="```python\ndef roofline_time(flops, bytes_moved, peak_flops, bandwidth):\n    t_compute = flops / peak_flops\n    t_memory = bytes_moved / bandwidth\n    if t_compute >= t_memory:\n        return [t_compute, \"compute\"]\n    return [t_memory, \"memory\"]\n```\n\nDe GPU rekent terwijl hij data verplaatst, dus de twee tijden overlappen en de grootste bepaalt het tempo. Welke van de twee groter is, zegt welke grens de bewerking raakt.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je eigen machine

    De knop meet je machine: matrixvermenigvuldigingen van 64 × 64 tot 2048 × 2048, en het optellen van twee arrays van 32 miljoen getallen, op je GPU als je die hebt, anders op de CPU. Het duurt een paar seconden. Het plaatje zet elke meting bij haar intensity, met een dak dat getekend is uit je snelste vermenigvuldiging en de bandbreedte van je optelling. Op een CPU zijn de getallen lager, en de vorm is dezelfde.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Meet mijn machine")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(measure_button):
    if not measure_button.value:
        _out = mo.md("*Druk op de knop om je machine te meten.*")
    else:
        _m = C.measure(lambda _i, _n: mo.output.replace(mo.md(f"*Meting {_i + 1} van {_n}*")))
        _peak = max(_f for _n, _x, _f in _m["matmul"])
        _bw = _m["add"][2]
        _points = [(_x, _f, f"{_n}" if _n in (64, 2048) else "") for _n, _x, _f in _m["matmul"]] + [(_m["add"][0], _m["add"][1], "optellen")]
        _where = {"cuda": "je GPU", "mps": "de GPU van je Mac", "cpu": "je CPU"}[_m["device"]]
        _ridge = C.number(_peak / _bw, 0 if _peak / _bw >= 10 else 1, L.lang)
        _out = mo.vstack([
            mo.Html(C.roof_picture({"title": f"Gemeten op {_where}", "x": "FLOPs per byte", "y": "FLOP/s", "ridge": f"ridge {_ridge}", "lang": L.lang,
                                    "alt": f"Log-log-assen met je metingen: matrixvermenigvuldigingen van 64 tot 2048, en het optellen. Je snelste vermenigvuldiging haalde {C.number(_peak / 1e9, 0, L.lang)} miljard FLOP/s; het optellen verplaatste {C.number(_bw / 1e9, 0, L.lang)} GB per seconde."},
                                   points=_points, peak=_peak, bandwidth=_bw)),
            mo.md(f"Op {_where} haalde de snelste matrixvermenigvuldiging {C.number(_peak / 1e9, 0, L.lang)} miljard FLOP/s, en verplaatste het optellen {C.number(_bw / 1e9, 0, L.lang)} GB per seconde: een ridge point bij ongeveer {_ridge} FLOPs per byte. Punten onder het dak verliezen elders tijd: elke bewerking starten kost tijd, wat het zwaarst weegt bij kleine, en caches, andere programma's en de temperatuur van de chip spelen allemaal mee. Meet twee keer voordat je één getal vertrouwt."),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_bound():
    mo.md(r"""
    ### Oefening 2. Welk deel wacht op het geheugen?
    """)
    return


@app.cell(hide_code=True)
def choice_bound_cell():
    choice_bound = L.choice(
        "memory-bound",
        question="Een GPT met de maten van de kleinste GPT-2, $d = 768$, MLP's die elke vector naar 3072 getallen brengen en terug, en een vocabulary van 50.257 tokens, traint op een RTX 4090 met batches van 8 vensters van 1024 tokens. Welk deel is het meest waarschijnlijk memory-bound?",
        options={
            "mlp": "De matrixvermenigvuldigingen van de MLP's",
            "right": "LayerNorm",
            "head": "De output layer, van $d$ naar alle tokens van de vocabulary",
        },
        correct="right",
        why={
            "mlp": "Hun weights worden één keer per batch gelezen en gebruikt voor 8192 tokens: ongeveer 290 FLOPs per byte, rechts van het ridge point, dus compute-bound.",
            "head": "Dat is ook een grote matrixvermenigvuldiging, met zijn weights gebruikt voor elk token van de batch: ongeveer 350 FLOPs per byte, compute-bound.",
            "right": "LayerNorm leest elk getal, doet er een handvol FLOPs mee, en schrijft het terug: een intensity van ongeveer één FLOP per byte, ver links van het ridge point.",
        },
    )
    choice_bound
    return (choice_bound,)


@app.cell(hide_code=True)
def text_exercise_batch():
    mo.md(r"""
    ### Oefening 3. Grotere batches
    """)
    return


@app.cell(hide_code=True)
def explain_batch():
    explain_batch_cell = L.explain(
        "batch",
        question="Een lineaire layer met een $d \\times d$-weightmatrix draait op een batch van $B$ tokens. Waarom daalt de tijd per token als $B$ groeit, en waarom houdt dat na een tijd op? Gebruik de arithmetic intensity.",
        model_answer="**FLOPs:** ongeveer $2 B d^2$, groeiend met $B$. **Bytes:** de weights, $d^2$ getallen, één keer per batch, plus de inputs en outputs, $2 B d$ getallen, elk 4 bytes. Bij een kleine $B$ overheersen de weights de bytes, dus is de intensity ongeveer $2 B d^2 / (4 d^2) = B/2$: die groeit met de batch. De layer is memory-bound, zijn tijd is vooral het lezen van de weights, en die tijd wordt over meer tokens verdeeld als $B$ groeit. Bij een grote $B$ overheersen de inputs en outputs, en vlakt de intensity af bij ongeveer $2 B d^2 / (8 B d) = d/4$. Ligt $d/4$ voorbij het ridge point, dan wordt de layer compute-bound, voegt elk extra token zijn eigen FLOPs toe, en blijft de tijd per token ongeveer gelijk; zo niet, zoals bij $d = 64$ op een RTX 4090, dan blijft hij memory-bound en vlakt de tijd per token toch af, omdat elk token dan zijn eigen bytes meebrengt.",
        points=[
            "je zegt dat de weights één keer per batch gelezen worden, dus dat de bytes veel langzamer groeien dan de FLOPs",
            "je zegt dat de intensity met de batch groeit, dus dat een memory-bound layer per token sneller wordt",
            "je zegt dat de intensity afvlakt, bij ongeveer $d/4$, dus dat de tijd per token niet meer daalt: compute-bound als dat voorbij het ridge point ligt, anders nog steeds memory-bound",
        ],
    )
    explain_batch_cell
    return (explain_batch_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De FLOPs komen uit [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl), de GPU uit [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl). De volgende les, *Minder bits: FP32, FP16 en BF16*, halveert de bytes van elk getal: voor een memory-bound bewerking is dat tot twee keer zo snel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bound, explain_batch_cell, result_time):
    L.wrap_up(result_time, choice_bound, explain_batch_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
