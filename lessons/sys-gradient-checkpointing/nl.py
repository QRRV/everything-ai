# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-gradient-checkpointing. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-gradient-checkpointing", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl) en [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_batch():
    quick_batch = L.choice(
        "quick-batch",
        question="Welk deel van het geheugen van een trainingsrun groeit mee met de batch size?",
        options={
            "weights": "De weights en hun gradients",
            "right": "De activations",
            "state": "De optimizer state",
        },
        correct="right",
        why={
            "weights": "De weights en hun gradients hebben één getal per parameter, wat de batch ook is. Zie [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl).",
            "state": "De optimizer bewaart een vast aantal waarden per parameter, wat de batch ook is. Zie [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl).",
            "right": "De forward pass bewaart tensors voor elk voorbeeld in de batch, en elke layer. Deze les ruilt een deel ervan in voor compute.",
        },
    )
    quick_batch
    return (quick_batch,)


@app.cell(hide_code=True)
def quick_check_input():
    quick_input = L.choice(
        "quick-input",
        question="Een linear layer rekent $\\mathbf{Y} = \\mathbf{X}\\mathbf{W}^\\top$ uit. Wat heeft de backward pass nodig uit de forward pass, om de gradient van $\\mathbf{W}$ uit te rekenen?",
        options={
            "right": "De input $\\mathbf{X}$",
            "output": "De output $\\mathbf{Y}$",
            "nothing": "Niets: $\\mathbf{W}$ is genoeg",
        },
        correct="right",
        why={
            "output": "De gradient van $\\mathbf{W}$ is $\\mathbf{G}^\\top\\mathbf{X}$, met $\\mathbf{G}$ de gradient van $\\mathbf{Y}$: daarvoor is de input nodig. Zie [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl).",
            "nothing": "De gradient van $\\mathbf{W}$ hangt af van wat er binnenkwam: $\\mathbf{G}^\\top\\mathbf{X}$, met $\\mathbf{G}$ de gradient van $\\mathbf{Y}$. Zie [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl).",
            "right": "$\\mathbf{G}^\\top\\mathbf{X}$, met $\\mathbf{G}$ de gradient van $\\mathbf{Y}$. Dus bewaart de forward pass elke input tot de backward pass: dat zijn de activations.",
        },
    )
    quick_input
    return (quick_input,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Met lange teksten en grote batches nemen de activations meer geheugen in dan de weights, de gradients en de optimizer state samen. Als een run niet past, kun je de batch kleiner maken, maar dan ziet elke stap minder voorbeelden. **Gradient checkpointing** houdt de batch: het gooit de meeste activations weg in de forward pass, en rekent ze in de backward pass opnieuw uit als ze nodig zijn. Minder geheugen, voor wat extra compute.
    """)
    return


@app.cell(hide_code=True)
def predict_compute():
    guess_compute = L.guess(
        "extra-flops",
        question="Een trainingsstap is een forward pass en een backward pass, en de backward pass kost ongeveer twee keer de FLOPs van de forward pass. Je zet een checkpoint om elk block van een diep netwerk. Ongeveer hoeveel meer FLOPs kost een stap?",
        options={
            "few": "Ongeveer 5% meer",
            "right": "Ongeveer een derde meer",
            "double": "Ongeveer twee keer zoveel",
        },
        reveal="**Ongeveer een derde meer:** de forward pass van elke layer draait nog één keer, tijdens de backward pass. Een stap kostte 1 + 2 = 3 forward passes; nu kost hij er 4. Hoeveel geheugen dat bespaart, hangt af van hoe lang de segmenten zijn, zoals deze les laat zien.",
    )
    guess_compute
    return (guess_compute,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Segmenten.** Verdeel de $n$ layers in segmenten van $k$ layers. Bewaar in de forward pass alleen de input van elk segment, en gooi de activations binnen een segment weg. Als de gradient in de backward pass bij een segment aankomt, draai je zijn forward pass opnieuw vanaf zijn bewaarde input, deze keer met zijn activations bewaard, en ga je er terug doorheen. Gooi ze dan weg, en ga door naar het segment ervoor.

        **Geheugen.** Tel in eenheden van de activations van één layer. Zonder checkpoints worden de input van het netwerk en de output van elke layer bewaard: $n + 1$. Met segmenten van $k$ bewaart de forward pass $\lceil n / k \rceil$ inputs van segmenten, en komen er tijdens de backward pass de $k$ van één segment bij:
        """),
        L.formula(r"[[hidden:\text{geheugen}(k)]] = \lceil n / k \rceil + k"),
        mo.md(r"""
        Korte segmenten bewaren veel inputs; lange rekenen veel tegelijk opnieuw uit. De som is het kleinst bij ongeveer $k = \sqrt{n}$: ongeveer $2\sqrt{n}$ in plaats van $n + 1$. Voor 16 layers is dat 8 in plaats van 17; voor 100 layers 20 in plaats van 101. Segmenten van 1 layer, of één segment van alle $n$, besparen in deze telling niets: weer $n + 1$.

        **Compute.** Wat $k$ ook is, de forward pass van elke layer draait één keer extra: ongeveer een derde meer FLOPs per stap. Dat is de prijs.

        **In PyTorch.** `checkpoint(fn, *inputs, use_reentrant=False)`, uit `torch.utils.checkpoint`, draait `fn` op de inputs zonder te bewaren wat erbinnen gebeurt, en draait hem opnieuw in de backward pass. `use_reentrant=False` vraagt om de manier van checkpointen die PyTorch aanraadt: die werkt ook als de input geen gradient nodig heeft, zoals de data die het eerste segment in gaat. Een checkpoint is hier geen model dat op schijf is opgeslagen, zoals de checkpoints van een trainingsrun: het is een bewaarde input van een segment, een punt waarvandaan de backward pass opnieuw rekent.

        De figuur: voor 16 layers het geheugen van de activations voor elke lengte $k$ van een segment: de bewaarde inputs van de segmenten, gevuld, en het opnieuw berekende segment erbovenop, licht. De gestreepte lijn is de 17 eenheden zonder checkpoints.
        """),
        mo.Html(C.segments_picture({"title": "16 layers in segmenten van k", "x": "lengte van een segment", "y": "eenheden van één layer",
                                    "plain": "zonder checkpoints", "lang": L.lang,
                                    "alt": "Vijf gestapelde balken, voor k = 1, 2, 4, 8 en 16, met hoogtes 17, 10, 8, 10 en 17: de bewaarde inputs, 16, 8, 4, 2 en 1, met het opnieuw berekende segment, 1, 2, 4, 8 en 16, erbovenop. Een gestreepte lijn op 17 geeft het geheugen zonder checkpoints, even hoog als de balken voor k = 1 en 16; de laagste balk is k = 4."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **gradient checkpointing** | | in de forward pass maar een deel van de activations bewaren, en de rest in de backward pass opnieuw uitrekenen | `checkpoint(fn, x, use_reentrant=False)` |
        | **segment** | | layers die onder één checkpoint draaien: alleen hun input wordt bewaard | |
        | $\lceil n / k \rceil$ | n gedeeld door k, naar boven afgerond | $n / k$ naar boven afgerond: het aantal segmenten | `math.ceil(n / k)` |
        """),
        L.watch_out("Een segment met toeval erin, zoals dropout, moet bij het opnieuw draaien dezelfde willekeurige getallen geven, anders verschillen de opnieuw berekende activations van de eerste. De functie `checkpoint` van PyTorch bewaart daarvoor de toestand van de random number generator en zet hem terug; eigen code die dingen opnieuw uitrekent, moet dat ook doen. En heel korte segmenten kosten bovenop de extra FLOPs ook nog wat extra werk van zichzelf."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een run passend te maken met gradient checkpointing:

    1. Meet de activations, zoals in [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl), en kijk hoeveel er weg moet.
    2. Verdeel de layers in segmenten; ongeveer $\sqrt{n}$ layers per segment bewaart het minst. Een transformer block bewaart vanbinnen veel tensors, dus één block per segment, dat alleen zijn input bewaart, bespaart al het meeste en is vaak het eenvoudigst.
    3. Stuur elk segment door `checkpoint(..., use_reentrant=False)`.
    4. Meet opnieuw: het bewaarde geheugen, en de tijd per stap. De FLOPs zijn ongeveer een derde meer; de tijd is vaak meer dan dat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en stap 4 op één linear layer en een ReLU, met `saved_tensors_hooks` zoals in [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl): wat de forward pass bewaart, zonder en met een checkpoint om de layer en de ReLU.
    """)
    return


@app.cell
def example_one_segment():
    import torch
    from torch.autograd.graph import saved_tensors_hooks
    from torch.utils.checkpoint import checkpoint

    _layer = torch.nn.Linear(100, 100)
    _x = torch.randn(10, 100)
    _places = {p.data_ptr() for p in _layer.parameters()}

    def _kept(run):
        _sizes = {}

        def _pack(t):
            if t.data_ptr() not in _places:
                _sizes[t.data_ptr()] = t.numel() * t.element_size()
            return t

        with saved_tensors_hooks(_pack, lambda t: t):
            _out = run()
        _out.sum().backward()
        return sum(_sizes.values())

    def _segment(x):
        return torch.relu(_layer(x))

    print("zonder:", _kept(lambda: _segment(_x)), "bytes")
    print("met een checkpoint:", _kept(lambda: checkpoint(_segment, _x, use_reentrant=False)), "bytes")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zonder checkpoint bewaart de forward pass de input van de layer, voor de gradient van zijn weights, en de output van de ReLU, voor de helling van de ReLU: 2 tensors van 10 × 100 float32-getallen, 8.000 bytes. Met een checkpoint bewaart hij alleen de input, 4.000 bytes; de output van de ReLU wordt in de backward pass opnieuw uitgerekend.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_checkpointed():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Segmenten van k blocks

        Het vak begint met `kept_bytes(run, blocks)`, de meting van het uitgewerkte voorbeeld voor een lijst blocks, en `run_blocks(blocks, x)`, die `x` op volgorde door de blocks stuurt, elk een linear layer met een ReLU erachter. Maak `checkpointed(blocks, x, k)` af: stuur de blocks in segmenten van `k`, elk segment door `checkpoint`, en geef de output terug. De check draait 8 linear layers van 32 getallen op 16 voorbeelden: de output moet dezelfde zijn als zonder checkpoints, en in segmenten van 2 moet de forward pass 4 inputs van segmenten bewaren, 8.192 bytes. Hij probeert ook segmenten van 3, en kijkt of de gradient het eerste block bereikt.
        """),
        L.task("Laat `checkpointed` elk segment van `k` blocks door `checkpoint` sturen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_checkpointed():
    box_checkpointed = L.code_box("checkpointed", lines=30)
    box_checkpointed
    return (box_checkpointed,)


@app.cell(hide_code=True)
def check_checkpointed(box_checkpointed):
    result_checkpointed = L.check("checkpointed", box_checkpointed.value)
    result_checkpointed
    return (result_checkpointed,)


@app.cell(hide_code=True)
def hints_checkpointed():
    L.hints(
        "checkpointed",
        [
            "Een loop met `range(0, len(blocks), k)`; een segment is `blocks[start:start + k]`, en `checkpoint` geeft alles na de functie aan die functie door.",
        ],
        why="`for start in range(0, len(blocks), k): x = checkpoint(run_blocks, blocks[start:start + k], x, use_reentrant=False)`, en `return x`. Het laatste segment is korter als het aantal blocks niet deelbaar is door `k`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De beste lengte van een segment

        Maak `best_k(n)` af: voor $n$ layers de lengte $k$ van een segment, van 1 tot en met $n$, met het minste geheugen, $\lceil n / k \rceil + k$ eenheden, en de kleinste $k$ als twee gelijk zijn. De check roept `best_k(16)` aan, wat `4` moet geven, en probeert 1 layer en aantallen layers die niet deelbaar zijn door de lengte van de segmenten.
        """),
        L.task("Laat `best_k` de lengte van een segment geven met het minste geheugen voor activations."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-k", lines=9)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-k", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-k",
        [
            "Houd de beste `k` tot nu toe bij en zijn som; vervang ze alleen als een nieuwe `k` een kleinere som geeft, zodat bij gelijke sommen de eerste, kleinste `k` blijft.",
        ],
        why="Laat `k` over `range(1, n + 1)` lopen, reken `math.ceil(n / k) + k` uit, en houd de `k` met de kleinste; of `min(range(1, n + 1), key=lambda k: math.ceil(n / k) + k)`, die bij gelijke sommen de eerste houdt. Voor 16 layers geeft `k = 4` 4 + 4 = 8.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je checkpoints op 16 layers

    De cel hieronder draait 16 linear layers van 512 getallen, elk met een ReLU, op 256 voorbeelden, met je `checkpointed` zodra die goed is, anders met die van de course: voor elke lengte van een segment wat de forward pass bewaart, en de tijd van een forward en backward pass op je CPU. Het duurt een paar seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_segments(box_checkpointed, result_checkpointed):
    _d = C.demo(box_checkpointed.value if result_checkpointed.state == "correct" else None)
    _base = _d["rows"][0][2]
    _who = "je `checkpointed`" if _d["yours"] else "de `checkpointed` van de course"
    _lines = [f"Met {_who}:", "",
              "| segmenten van | bewaarde bytes | tijd |", "|---|---|---|"]
    for _k, _bytes, _t in _d["rows"]:
        _lines.append(f"| {'geen checkpoints' if _k == '-' else _k} | {C.grouped(_bytes, 'nl')} | {L.number(_t / _base, 2)}× |")
    _slower = all(_t > _base for _, _, _t in _d["rows"][1:])
    _end = ("Elke lengte van een segment kostte meer tijd dan geen checkpoints: de extra forward passes, en wat extra werk voor elk checkpoint." if _slower else
            "De extra forward passes en het extra werk voor elk checkpoint kosten tijd, maar op een drukke computer wisselen tijden: voer de cel opnieuw uit als een lengte van een segment even snel uitkwam als geen checkpoints.")
    _lines += ["", "De tijd als veelvoud van de run zonder checkpoints. De bewaarde bytes halveren elke keer dat de segmenten twee keer zo lang worden. " + _end]
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def text_exercise_same():
    mo.md(r"""
    ### Oefening 3. Wat meer compute kost
    """)
    return


@app.cell(hide_code=True)
def choice_same_cell():
    choice_same = L.choice(
        "same-compute",
        question="Je zet checkpoints in een netwerk van 16 layers, één keer in segmenten van 1 layer, en één keer in segmenten van 4. Welke voegt meer FLOPs toe aan een trainingsstap?",
        options={
            "one": "Segmenten van 1: meer checkpoints, meer opnieuw uitrekenen",
            "four": "Segmenten van 4: langere segmenten om opnieuw uit te rekenen",
            "right": "Ongeveer evenveel: elke layer draait één keer extra vooruit",
        },
        correct="right",
        why={
            "one": "Elke layer zit in precies één segment, en elk segment draait zijn forward pass één keer extra: het aantal segmenten verandert de FLOPs niet.",
            "four": "Een langer segment rekent meer layers opnieuw uit, maar er zijn minder segmenten: samen elke layer één keer.",
            "right": "De lengte van een segment verandert het geheugen, niet de FLOPs. Alleen heel korte segmenten kosten wat meer tijd, door het extra werk van elk checkpoint.",
        },
    )
    choice_same
    return (choice_same,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het geheugen voor activations van [Waar het geheugen heen gaat](/course/open/sys-memory-accounting?lang=nl), gemeten met dezelfde `saved_tensors_hooks`, nu ingeruild voor compute: de inputs van [Backprop door hele layers met de hand](/course/open/nn-backprop-tensors?lang=nl) worden alleen aan het begin van elk segment bewaard. Met mixed precision en gradient accumulation is het een van de gebruikelijke manieren om een run passend te maken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same, result_best, result_checkpointed):
    L.wrap_up(result_checkpointed, result_best, choice_same)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
