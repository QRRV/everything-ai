# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-profiling-mfu. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-profiling-mfu", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl) en [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_time():
    quick_time = L.choice(
        "quick-time",
        question="De 6ND-vuistregel geeft de FLOPs van een trainingsrun. Wat heb je nog meer nodig om daar een tijd van te maken?",
        options={
            "layers": "Het aantal layers van het model",
            "lr": "De learning rate",
            "right": "Hoeveel FLOPs per seconde je GPU's echt doen",
        },
        correct="right",
        why={
            "layers": "De layers zitten al in $N$, het aantal parameters. Zie [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).",
            "lr": "De learning rate verandert wat het model leert, niet wat een stap kost. Zie [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).",
            "right": "De piek maal het deel ervan dat een run haalt, maal het aantal GPU's. Deze les meet dat deel.",
        },
    )
    quick_time
    return (quick_time,)


@app.cell(hide_code=True)
def quick_check_bound():
    quick_bound = L.choice(
        "quick-bound",
        question="Een bewerking doet 2 FLOPs voor elke byte die hij leest. Hij draait op een GPU met een ridge point bij 80 FLOPs per byte. Wat bepaalt zijn snelheid?",
        options={
            "right": "Het geheugen: hij wacht op data",
            "compute": "De berekeningen: hij wacht op de cores",
            "neither": "Geen van beide: 2 FLOPs per byte is te weinig om mee te tellen",
        },
        correct="right",
        why={
            "right": "Ver links van het ridge point wachten de cores op de bytes. Veel kleine bewerkingen in een netwerk zijn zo.",
            "compute": "Compute-bound bewerkingen liggen rechts van het ridge point, bij meer dan 80 FLOPs per byte. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
            "neither": "Elke bewerking wordt door iets begrensd: hier door de bandbreedte, omdat hij veel bytes leest voor weinig FLOPs. Zie [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl).",
        },
    )
    quick_bound
    return (quick_bound,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De 6ND-vuistregel geeft een tijd als je weet welk deel van de piek je run haalt. Dat deel meet je: draai een paar stappen, tel de tokens per seconde, en vergelijk de FLOPs per seconde van het model met de piek. Valt dat laag uit, dan laat een profiler zien waar de tijd heen gaat, bewerking voor bewerking, en dus wat je het eerst verandert. Een run die van 15% naar 45% van de piek gaat, duurt een derde van de tijd.
    """)
    return


@app.cell(hide_code=True)
def predict_async():
    guess_async = L.guess(
        "async-clock",
        question="Op een GPU meet je één trainingsstap: `time.perf_counter()` ervoor en erna, en verder niets. Wat laat het verschil zien?",
        options={
            "right": "Minder dan de stap duurt, vaak veel minder",
            "same": "De tijd van de stap",
            "more": "Meer dan de stap duurt: de klok telt ook het versturen van het werk naar de GPU",
        },
        reveal="**Minder dan de stap duurt, vaak veel minder.** PyTorch zet het werk van elke bewerking in de wachtrij van de GPU en gaat meteen door met de volgende regel Python. De GPU werkt de wachtrij in zijn eigen tempo af, dus de tweede `time.perf_counter()` kan draaien terwijl de GPU nog bezig is. `torch.cuda.synchronize()` wacht tot de wachtrij leeg is.",
    )
    guess_async
    return (guess_async,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: tokens per seconde.** Draai een aantal stappen, en deel de tokens waarop ze trainden door de seconden die ze duurden: de **throughput** van de run. Drie dingen maken de meting eerlijk. Sla de eerste stappen over: daarin vraagt PyTorch geheugen aan en kiest het hoe het elke bewerking uitrekent, dus zijn ze trager dan de rest. Wacht op een GPU op het device voordat je de klok afleest, met `torch.cuda.synchronize()`, zowel voordat je de klok start als voordat je hem stopt. En meet meerdere stappen, niet één: stappen verschillen een beetje van elkaar.

        **Stap 2: MFU.** Tokens per seconde maal de FLOPs per token, $6N$ plus attention, geeft de FLOPs per seconde die het model nodig heeft. Gedeeld door de piek is dat de **MFU**, de model FLOPs utilization:
        """),
        L.formula(r"\text{MFU} = \frac{\text{tokens/s} \times \text{FLOPs per token}}{\text{piek}}"),
        mo.md(r"""
        De MFU telt alleen de FLOPs die het model nodig heeft. Sommige runs rekenen meer: ze gooien activations weg en berekenen ze in de backward pass opnieuw, om geheugen te besparen. De **HFU**, de hardware FLOPs utilization, telt die FLOPs ook mee, dus is hij nooit lager dan de MFU. Neem als piek de specificaties van je GPU in je getalformaat; op je eigen machine neemt de demo in plaats daarvan de snelste matrixvermenigvuldiging van je device, zoals in [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl), die iets lager uitkomt dan de specificaties.

        **Stap 3: de profiler.** `torch.profiler.profile` legt elke bewerking vast die PyTorch draait, met de tijd die ze nam; `key_averages()` telt ze per bewerking op. De *self time* van een bewerking laat de bewerkingen weg die ze zelf aanroept, zodat geen tijd dubbel telt. De meeste namen beginnen met `aten::`, de bibliotheek van bewerkingen van PyTorch: `aten::mm`, `aten::addmm` en `aten::bmm` zijn matrixvermenigvuldigingen, en de meeste andere zijn kleine bewerkingen die elk getal één keer lezen en schrijven, en dus memory-bound zijn. Op een NVIDIA-GPU laat Nsight Systems hetzelfde zien op een tijdlijn, met de CPU en de GPU naast elkaar.

        **Stap 4: wat je verandert.** Nemen de matrixvermenigvuldigingen de meeste tijd, dan werkt de GPU aan waar hij goed in is. Nemen kleine bewerkingen de meeste tijd, dan geven een grotere batch of langere vensters elke bewerking meer werk, halveert mixed precision de bytes, en haalt het samenvoegen van bewerkingen, zodat de data één keer gelezen wordt voor een paar ervan, ritjes naar het geheugen weg: de volgende les doet dat. Wacht de GPU op de CPU, zoek dan naar stappen die expres wachten. Meet na elke verandering opnieuw.

        Het plaatje: drie trainingsstappen op een GPU, zoals Python en de GPU ze zien.
        """),
        mo.Html(C.queue_picture({"title": "Python loopt voor de GPU uit", "time": "tijd", "python": "Python", "gpu": "GPU", "waits": "sync: wacht",
                                 "early": "te vroeg", "after": "na sync", "lang": L.lang,
                                 "alt": "Twee rijen over de tijd. Boven zet Python stap 1, 2 en 3 elk in een oogwenk in de wachtrij, en wacht dan in een sync tot de GPU klaar is. Onder werkt de GPU stap 1, 2 en 3 na elkaar af, veel langer. Een klok die meteen na stap 3 wordt afgelezen, stopt te vroeg; na de sync stopt hij als de GPU klaar is."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **throughput** | hoeveel werk een run per seconde afmaakt, zoals tokens per seconde | |
        | **MFU** | model FLOPs utilization: de FLOPs per seconde die het model nodig heeft, als deel van de piek | |
        | **HFU** | hardware FLOPs utilization: hetzelfde, met de dubbel berekende FLOPs ook meegeteld | |
        | **profiler** | een tool die meet waar een programma zijn tijd aan besteedt, bewerking voor bewerking | `torch.profiler` |
        """),
        L.watch_out("`loss.item()` en `print(loss)` wachten ook op de GPU: ze hebben het getal nodig. Een loop die elke stap de loss leest, wacht elke stap, en Python kan niet meer vooruitlopen. Lees hem in plaats daarvan elke honderd stappen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zoek je uit hoe snel een run is, en waarom:

    1. Draai een paar stappen zonder te meten, wacht op het device, en start de klok.
    2. Draai een aantal stappen, wacht op het device, en stop de klok: tokens gedeeld door seconden.
    3. Vermenigvuldig met de FLOPs per token en deel door de piek: de MFU.
    4. Profileer een paar stappen, en kijk welke bewerkingen de tijd nemen.
    5. Verander de grootste, en meet opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De profiler is nieuw: tien trainingsstappen van een kleine MLP op de CPU, vastgelegd, en elke bewerking die meer dan 5% van de tijd nam, met hoe vaak ze draaide.
    """)
    return


@app.cell
def example_profile():
    import torch
    from torch.profiler import profile

    torch.manual_seed(0)
    _hidden, _out = torch.nn.Linear(256, 1024), torch.nn.Linear(1024, 10)
    _opt = torch.optim.AdamW(list(_hidden.parameters()) + list(_out.parameters()))
    _x, _y = torch.randn(512, 256), torch.randint(0, 10, (512,))

    with profile() as _prof:
        for _ in range(10):
            _loss = torch.nn.functional.cross_entropy(_out(torch.relu(_hidden(_x))), _y)
            _opt.zero_grad()
            _loss.backward()
            _opt.step()

    _total = sum([_e.self_cpu_time_total for _e in _prof.key_averages()])
    for _e in _prof.key_averages():
        _share = _e.self_cpu_time_total / _total
        if _share > 0.05:
            print(_e.key, round(100 * _share), "%,", _e.count, "keer")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De matrixvermenigvuldigingen nemen de grootste delen: `aten::addmm` in de forward pass, een vermenigvuldiging met de bias erbij, 20 keer voor 2 layers in 10 stappen, en `aten::mm` in de backward pass. De rest is verdeeld over kleine bewerkingen: de ReLU, `aten::clamp_min`, en zijn backward pass, `aten::threshold_backward`; `aten::copy_`, dat tensors kopieert; en de updates van AdamW, zoals `aten::sqrt`, die elke stap een paar keer langs elke parameter gaan. `Optimizer.step#AdamW.step` is geen bewerking maar een label: de tijd die de `step` van AdamW in Python zelf doorbrengt, buiten de bewerkingen die hij aanroept. Hoe de delen verdeeld zijn, hangt af van je machine: met meer CPU-cores gaan de vermenigvuldigingen sneller, en wegen de kleine bewerkingen zwaarder.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tps():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Tokens per seconde

        Schrijf `tokens_per_second(run, n_tokens, skip, steps)`. `run.step()` start één trainingsstap van `n_tokens` tokens, `run.sync()` wacht tot het device klaar is, en `run.clock()` geeft de tijd in seconden. Draai `skip` stappen zonder te meten, meet dan `steps` stappen, en geef de tokens per seconde terug. De check speelt een GPU na: `run.step()` komt meteen terug terwijl het device doorwerkt, de eerste stappen zijn traag, en de andere verschillen een beetje van elkaar. Zijn klok is een eigen klok, dus de check geeft op elke machine hetzelfde antwoord. De check roept `tokens_per_second(run, 2048, 5, 10)` aan, wat ongeveer 409.600 moet geven, en probeert andere runs.
        """),
        L.task("Schrijf `tokens_per_second(run, n_tokens, skip, steps)`: sla de eerste stappen over, en meet de andere met het device klaar bij de start en aan het eind."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tps():
    box_tps = L.code_box("tokens-per-second", lines=10)
    box_tps
    return (box_tps,)


@app.cell(hide_code=True)
def check_tps(box_tps):
    result_tps = L.check("tokens-per-second", box_tps.value)
    result_tps
    return (result_tps,)


@app.cell(hide_code=True)
def hints_tps():
    L.hints(
        "tokens-per-second",
        [
            "Stap 1 en 2 van het recept: twee loops, met een `run.sync()` na elke loop, en de klok afgelezen meteen na elke `run.sync()`.",
        ],
        why="Na de overgeslagen stappen zorgt `run.sync()` dat ze af zijn voordat de klok start; na de gemeten stappen zorgt het dat ze af zijn voordat de klok stopt. De 10 gemeten stappen duren samen 0,05 seconde: $2048 \\times 10 / 0{,}05 = 409.600$ tokens per seconde.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je GPT, gemeten en geprofileerd

    De knop laat je `tokens_per_second` je `GPT` uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) meten, die met AdamW traint op willekeurige tokens, op je GPU als je die hebt, anders op de CPU. Waar een module van jou ontbreekt, springt de versie van de course in. Op een NVIDIA-GPU traint hij de GPT van het boek, mini-batches van 32 vensters van 256 tokens; op een CPU of een Mac een kleinere, 4 blocks met $d = 128$ en mini-batches van 8 vensters van 128 tokens. Daarna meet hij de snelste matrixvermenigvuldiging van je device als zijn piek, en profileert hij twee stappen. Op een CPU duurt het een paar seconden. Hij werkt zodra oefening 1 goed is.
    """)
    return


@app.cell(hide_code=True)
def measure_button_cell():
    measure_button = mo.ui.run_button(label="Meet en profileer mijn GPT")
    measure_button
    return (measure_button,)


@app.cell(hide_code=True)
def demo_measure(box_tps, measure_button, result_tps):
    if result_tps.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 af: deze cel gebruikt je `tokens_per_second`.*")
    elif not measure_button.value:
        _out = mo.md("*Druk op de knop om te meten.*")
    else:
        _m = C.demo(box_tps.value)
        if _m == "error":
            _out = mo.md("*De meting stopte met een fout: werkt je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) nog?*")
        else:
            _n = lambda _x, _d=0: C.number(_x, _d, L.lang)
            _where = {"cuda": "je GPU", "mps": "de GPU van je Mac", "cpu": "je CPU"}[_m["device"]]
            _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_s}`" for _s in _m["stand_ins"]) + ".") if _m["stand_ins"] else ""
            _rows = "\n".join(f"| `{_k}` | {_n(100 * _s)}% | {_c} |" for _k, _s, _c in _m["top"])
            _out = mo.md(f"""
    Op {_where} trainde je GPT van {_n(_m['params'])} parameters met **{_n(_m['tps'])} tokens per seconde**. Met {_n(_m['per_token'] / 1e6, 1)} miljoen FLOPs per token is dat {_n(_m['flops'] / 1e9, 1)} miljard FLOP/s. De snelste matrixvermenigvuldiging haalde {_n(_m['peak'] / 1e9, 1)} miljard FLOP/s: een MFU van **{_n(100 * _m['mfu'])}%**.{_stand}

    De bewerkingen die in twee stappen de meeste tijd namen:

    | bewerking | deel van de tijd | keer |
    |---|---|---|
    {_rows}

    Meet twee keer voordat je één getal vertrouwt: andere programma's, de temperatuur van de chip en de caches spelen allemaal mee.
    """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_hfu():
    mo.md(r"""
    ### Oefening 2. MFU en HFU
    """)
    return


@app.cell(hide_code=True)
def choice_hfu_cell():
    choice_hfu = L.choice(
        "hfu",
        question="Om geheugen te besparen gooit een run in de forward pass de activations van elk block weg, en berekent hij tijdens de backward pass de hele forward pass opnieuw. Zijn MFU is 30%. Ongeveer hoeveel is zijn HFU?",
        options={
            "less": "Ongeveer 22%",
            "same": "30%, hetzelfde",
            "right": "Ongeveer 40%",
        },
        correct="right",
        why={
            "less": "De HFU telt meer FLOPs dan de MFU, nooit minder: het opnieuw berekende werk komt erbovenop.",
            "same": "De MFU laat de tweede forward pass weg, de HFU telt hem mee: de HFU valt hoger uit.",
            "right": "Per token rekent de hardware $6N + 2N = 8N$ FLOPs terwijl het model er $6N$ nodig heeft: de HFU is $30\\% \\times 8/6 = 40\\%$.",
        },
    )
    choice_hfu
    return (choice_hfu,)


@app.cell(hide_code=True)
def text_exercise_low():
    mo.md(r"""
    ### Oefening 3. Een lage MFU
    """)
    return


@app.cell(hide_code=True)
def explain_low():
    explain_low_cell = L.explain(
        "low-mfu",
        question="Je GPT traint op een GPU met een MFU van 8%. De profiler zegt dat de matrixvermenigvuldigingen een vijfde van de tijd nemen. Wat probeer je, en hoe weet je of het hielp?",
        model_answer="**Wat het profiel zegt:** vier vijfde van de tijd gaat naar andere bewerkingen, de meeste kleine die op het geheugen wachten, of naar wachten. **Wat je probeert:** een grotere batch of langere vensters, zodat elke vermenigvuldiging meer werk heeft en de kleine bewerkingen minder wegen; mixed precision, dat de bytes van de meeste bewerkingen halveert; kleine bewerkingen samenvoegen, zodat de data één keer gelezen wordt voor een paar ervan; en zoeken naar wachten, zoals `loss.item()` elke stap of een data loader die het niet bijhoudt. **Hoe je het weet:** verander één ding tegelijk, meet de tokens per seconde en de MFU opnieuw op dezelfde manier, en profileer opnieuw om te zien welke bewerkingen nu de tijd nemen.",
        points=[
            "je zegt dat de meeste tijd naar kleine, memory-bound bewerkingen of naar wachten gaat",
            "je noemt minstens twee veranderingen, zoals een grotere batch, mixed precision, bewerkingen samenvoegen of wachten weghalen",
            "je meet opnieuw na elke verandering, één verandering tegelijk",
        ],
    )
    explain_low_cell
    return (explain_low_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De schatting van [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl) heeft nu een gemeten deel van de piek, en de profiler laat de memory-bound bewerkingen van [Geheugen of compute: wat remt je GPU af?](/course/open/sys-memory-roofline?lang=nl) in je eigen GPT zien. De volgende les, *Je eigen GPU kernel schrijven*, voegt zulke kleine bewerkingen samen tot één.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hfu, explain_low_cell, result_tps):
    L.wrap_up(result_tps, choice_hfu, explain_low_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
