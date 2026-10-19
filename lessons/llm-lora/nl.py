# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-lora. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-lora", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Rang: hoeveel richtingen een matrix heeft](/course/open/math-matrix-rank?lang=nl) en [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rank():
    quick_rank = L.choice(
        "quick-rank",
        question="Je telt bij een weight matrix met vorm (96, 96) het product `B @ A` op, met `B` van vorm (96, 8) en `A` van vorm (8, 96). Wat is de hoogste rang die deze verandering kan hebben?",
        options={
            "full": "96",
            "sum": "104",
            "right": "8",
        },
        correct="right",
        why={
            "full": "Elke rij van `B @ A` is een mengsel van de 8 rijen van `A`, dus hij heeft hooguit 8 richtingen. Zie [Rang: hoeveel richtingen een matrix heeft](/course/open/math-matrix-rank?lang=nl).",
            "sum": "De rang van een product is hooguit zijn kleinste binnenmaat: de 8 kolommen van `B` en rijen van `A`, niet 96 en 8 bij elkaar opgeteld. Zie [Rang: hoeveel richtingen een matrix heeft](/course/open/math-matrix-rank?lang=nl).",
            "right": "Een (96, 8)-matrix keer een (8, 96)-matrix heeft hooguit rang 8. Vandaag is zo'n product alles wat een model leert.",
        },
    )
    quick_rank
    return (quick_rank,)


@app.cell(hide_code=True)
def quick_check_padding():
    quick_padding = L.choice(
        "quick-padding",
        question="Waarom krijgen bij SFT de doelen van de padding in een batch het label -100?",
        options={
            "stop": "Zodat het model leert stoppen na zijn antwoord",
            "right": "Zodat ze nergens voor tellen in de loss",
            "id": "Omdat -100 het id van het paddingtoken is",
        },
        correct="right",
        why={
            "stop": "Het model leert stoppen van het end-of-text aan het eind van elk antwoord, met zijn label meegeteld. De padding erna hoort niets te leren. Zie [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).",
            "id": "-100 is helemaal geen token: het is het label dat `F.cross_entropy` overslaat. De inputs krijgen het end-of-text als padding. Zie [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).",
            "right": "`F.cross_entropy` slaat elk doel van -100 over. Vandaag finetune je op dezelfde batches, met veel minder getallen die leren.",
        },
    )
    quick_padding
    return (quick_padding,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Full fine-tuning traint elk getal van het model. AdamW houdt voor elk getal dat leert twee eigen getallen bij, en er is voor elk ook een gradient: bij een model van miljarden parameters zijn dat een paar kopieën van het model in het geheugen. En elke gefinetunede versie is een heel nieuw model om op te slaan. **LoRA**, low-rank adaptation (Hu en anderen, 2021), bevriest het model en leert voor elke weight matrix alleen een kleine low-rank correctie. Op GPT-3, met 175 miljard parameters, kan het tot 10.000 keer zo weinig getallen trainen als full fine-tuning met Adam, met een derde van het GPU-geheugen.

    **Een voorbeeld uit het dagelijks leven.** Je schrijft je aantekeningen in de kantlijn van een dik boek, in plaats van het boek over te schrijven. Het boek blijft zoals het was; de aantekeningen zijn klein, je kunt ze overnemen in een ander exemplaar van hetzelfde boek, en uitgummen.
    """)
    return


@app.cell(hide_code=True)
def predict_lora():
    _m, _f = C.MEASURED["yours"], C.T.MEASURED["yours"]
    guess_lora = L.guess(
        "lora",
        question=f"In het project over SFT deed het model uit de demorun van de capstone van de course {_f['after']['used']} van de {_f['after']['total']} testinstructies goed na {C.number(C.STEPS, 0, L.lang)} stappen SFT op al zijn {C.number(_m['params'], 0, L.lang)} getallen. LoRA van rang {_m['rank']} bevriest ze allemaal en traint {C.number(_m['trainable'], 0, L.lang)} nieuwe, op dezelfde voorbeelden en even veel stappen. Hoeveel instructies doet het goed?",
        options={
            "fewer": f"Veel minder: ongeveer {round(_f['after']['used'] / 3)}",
            "right": f"Ongeveer evenveel: {round(_f['after']['used'] * 2 / 3)} tot {round(_f['after']['used'] * 4 / 3)}",
            "more": f"Veel meer: ongeveer {2 * _f['after']['used']}",
        },
        reveal=f"**{_m['after']['used']} van de {_m['after']['total']}**, tegen {_f['after']['used']} voor full fine-tuning. Met {C.number(100 * _m['trainable'] / _m['params'], 1, L.lang)}% zoveel getallen die leren, en een vijf keer zo hoge learning rate, een paar meer; ook Hu en anderen vonden LoRA even goed als full fine-tuning of beter. Ze vermoedden dat de verandering die fine-tuning in een weight matrix maakt low-rank is; voor dit model en deze instructies is een update van rang {_m['rank']} in elke linear layer van de blocks genoeg.",
    )
    guess_lora
    return (guess_lora,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Een bevroren weight en een low-rank update.** Een linear layer rekent $W x$ uit. LoRA houdt $W$ zoals hij is, en telt er een matrix $B A$ bij op: $A$ heeft $r$ rijen, $B$ heeft $r$ kolommen, dus $B A$ heeft de vorm van $W$ maar een rang van hooguit $r$, de **rang** van de update. Alleen $A$ en $B$ leren. De update wordt geschaald met $\alpha / r$. Hu en anderen houden $\alpha$ vast als ze de rang veranderen, en dat helpt om de learning rate te houden; velen nemen in plaats daarvan $\alpha$ twee keer de rang, een schaal van 2, zoals de demo:
    """)
    return


@app.cell(hide_code=True)
def formula_lora():
    L.formula(r"h = W x + \frac{\alpha}{r} \, B A x")
    return


@app.cell(hide_code=True)
def picture_lora():
    _d, _r = C.MEASURED["yours"]["d"], C.RANK
    mo.vstack([
        mo.md(f"Het beeld, op schaal: een weight matrix van het model uit de demorun van de capstone van de course, {_d} bij {_d}, en een update van rang {_r}. De bevroren weight is grijs; wat leert heeft de kleur van wat een model leert."),
        C.lora_picture(_d, _r, {"title": f"Een bevroren weight en een update van rang {_r}",
                                "w": f"W: {_d} × {_d}, bevroren: {C.number(_d * _d, 0, L.lang)} getallen",
                                "ba": f"B: {_d} × {_r}, A: {_r} × {_d}: {C.number(2 * _d * _r, 0, L.lang)} getallen die leren",
                                "alt": f"Een groot grijs vierkant W van {_d} bij {_d} getallen, plus een hoge smalle matrix B en een brede smalle matrix A in oranje: {C.number(2 * _d * _r, 0, L.lang)} getallen die leren."}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    **Begin op nul.** $A$ begint willekeurig, $B$ op nul (Hu en anderen, 2021). Dan is $B A$ nul, en begint het model precies zoals het was. $B$ krijgt vanaf de eerste stap een gradient, omdat $A$ niet nul is; zodra $B$ beweegt, leert $A$ ook.

    **Samenvoegen, of uitzetten.** Na het trainen is $W + \frac{\alpha}{r} B A$ één matrix met de vorm van $W$: samengevoegd kost de update niets als het model schrijft. Apart gehouden is de update een **adapter**: een klein bestand, en uitgezet is het model weer het model waarmee je begon.

    **Waar.** In de meeste van hun experimenten zetten Hu en anderen LoRA alleen op de query en de value van attention. Dettmers en anderen (2023) vonden dat LoRA op elke linear layer van de blocks moet staan om full fine-tuning te evenaren. Je `add_lora` doet dat: zeven layers per block van je ModernGPT. De embedding en de head blijven bevroren.

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **LoRA** | low-rank adaptation: alleen een low-rank update $B A$ van elke bevroren weight matrix finetunen | `LoRALinear(base, r, alpha)` |
    | $A$, $B$ | de twee smalle matrices van de update: $A$ met vorm $(r, \text{in})$, $B$ met vorm $(\text{out}, r)$ | `self.A`, `self.B` |
    | $r$ | de **rang** van de update: het aantal rijen van $A$ en kolommen van $B$ | `r` |
    | $\alpha$ | alfa: met $r$ de schaal $\alpha / r$ van de update | `alpha / r` |
    | **adapter** | de update van een model, apart ervan bewaard, die aan en uit kan | `layer.on = False` |
    | **samenvoegen** | de geschaalde update bij de weight optellen, zodat er één matrix overblijft | `W + scale * B @ A` |

    ### Recept

    Om een model met LoRA te finetunen:

    1. Bevries elke parameter van het model.
    2. Zet een LoRA-layer rond elke linear layer van zijn blocks: de bevroren layer, $A$ willekeurig, $B$ nul.
    3. Train alleen $A$ en $B$, met de stap en de voorbeelden van full fine-tuning, met een hogere learning rate.
    4. Bewaar de adapters, of voeg ze samen met de weights.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een weight met vorm (4, 4) en een update van rang 1. De getallen zijn willekeurig.
    """)
    return


@app.cell
def example_lora():
    torch.manual_seed(0)
    _W = torch.randn(4, 4)
    _A = torch.randn(1, 4)
    _B = torch.zeros(4, 1)
    _x = torch.randn(4)
    print(torch.allclose(_W @ _x + _B @ _A @ _x, _W @ _x))   # B is nul: er verandert nog niets
    _B = torch.tensor([[1.0], [0.5], [0.0], [-1.0]])         # zoals na wat training
    print(_W.numel(), _A.numel() + _B.numel())
    print(np.linalg.matrix_rank((_B @ _A).numpy()))
    _merged = _W + _B @ _A
    print(torch.allclose(_merged @ _x, _W @ _x + _B @ (_A @ _x)))   # samengevoegd: één matrix, dezelfde output
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij de start is $B$ nul, dus geeft de layer wat $W$ geeft. Na wat training heeft de update 8 getallen tegen de 16 van $W$, en een rang van 1: elke rij van $B A$ is een veelvoud van $A$. Samengevoegd tot één matrix geeft hij dezelfde output. Voor een matrix van 96 bij 96 en een rang van 8 zijn het 1.536 getallen tegen 9.216.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_layer():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een LoRA-layer

        Maak `LoRALinear(base, r, alpha)` af: de layer van stap 2. Hij houdt de linear layer `base`, bevroren, en maakt `A` met vorm `(r, base.in_features)`, willekeurig, en `B` met vorm `(base.out_features, r)`, nul, allebei als `nn.Parameter`. Zijn `forward` geeft wat `base` geeft, plus, als `self.on` True is, de update `x @ self.A.T @ self.B.T` keer `self.scale`. De check maakt `LoRALinear(nn.Linear(6, 4), 2, 4)` en probeert hem op drie inputs van 6 getallen, en meer.
        """),
        L.task("Laat `LoRALinear` beginnen als zijn basis, alleen `A` en `B` leren, en uit kunnen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layer():
    box_layer = L.code_box("lora-linear", lines=20)
    box_layer
    return (box_layer,)


@app.cell(hide_code=True)
def check_layer(box_layer):
    result_layer = L.check("lora-linear", box_layer.value)
    result_layer
    return (result_layer,)


@app.cell(hide_code=True)
def hints_layer():
    L.hints(
        "lora-linear",
        [
            "Een parameter `p` bevriezen is `p.requires_grad = False`, voor elke parameter van de basis. Een tensor wordt een eigen parameter met `nn.Parameter(...)`; `torch.zeros` en `torch.randn` maken tensors met de vormen die de opdracht geeft, en de willekeurige getallen delen door `base.in_features ** 0.5` houdt ze klein.",
            "`x` heeft `in_features` getallen in zijn laatste dimensie. `x @ self.A.T` brengt ze naar `r` getallen, `@ self.B.T` naar `out_features`: de vorm van wat `base(x)` geeft, ook voor een batch vensters.",
        ],
        why="Met `B` op nul begint de layer als zijn basis. Alleen `A` en `B` hebben een gradient nodig, en met `self.on` kan een latere les de update uitzetten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_add():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Adapters in je model

        Maak `add_lora(model, r, alpha)` af: stap 1 en 2. Bevries elke parameter van het model, en zet dan een `LoRALinear` rond elk van de zeven linear layers van elk block: `query`, `key`, `value` en `out` van zijn attention `a`, en `gate`, `up` en `down` van zijn MLP `m`. Geef de lijst van de nieuwe layers terug. De check bouwt een kleine ModernGPT van 2 blocks, `ModernGPT(40, 16, 2, 1, 2)`, en roept `add_lora(model, 2, 4)` aan.
        """),
        L.task("Laat `add_lora` het model bevriezen en elke linear layer van zijn blocks een adapter geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_add():
    box_add = L.code_box("add-lora", lines=16)
    box_add
    return (box_add,)


@app.cell(hide_code=True)
def check_add(box_add):
    result_add = L.check("add-lora", box_add.value)
    result_add
    return (result_add,)


@app.cell(hide_code=True)
def hints_add():
    L.hints(
        "add-lora",
        [
            "Bevries eerst, zoals in je `LoRALinear`, maar voor `model.parameters()`. De `A` en `B` die je daarna maakt zijn nieuwe parameters, en die leren.",
            "`a.query = LoRALinear(a.query, r, alpha)` zet een adapter rond één layer: de nieuwe layer neemt de plaats van de oude in in het model, en houdt hem als zijn `base`.",
        ],
        why="De volgorde telt: wat je na het toevoegen van de adapters bevriest, bevriest de adapters ook. Zeven adapters per block is wat volgens Dettmers en anderen LoRA nodig heeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Voeg de update samen

        Maak `merge_lora(layer)` af: stap 4. Geef een nieuwe `nn.Linear` terug waarvan de weight de weight van `layer.base` is plus `layer.scale` keer `layer.B @ layer.A`, met de bias van de basis als die er een heeft. De basis zelf blijft zoals hij was. De check voegt een `LoRALinear(nn.Linear(6, 4), 2, 4)` samen waarvan `B` niet meer nul is, en meer.
        """),
        L.task("Laat `merge_lora` één linear layer geven die doet wat de layer en zijn update doen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_merge():
    box_merge = L.code_box("merge-lora", lines=12)
    box_merge
    return (box_merge,)


@app.cell(hide_code=True)
def check_merge(box_merge):
    result_merge = L.check("merge-lora", box_merge.value)
    result_merge
    return (result_merge,)


@app.cell(hide_code=True)
def hints_merge():
    L.hints(
        "merge-lora",
        [
            "`layer.B @ layer.A` heeft de vorm van de weight: (out_features, in_features). De samengevoegde weight is de weight van de basis plus dat product, geschaald zoals in `forward`.",
            "`merged.weight.copy_(...)` zet getallen in de nieuwe layer, onder `torch.no_grad()`; `merged.bias.copy_(base.bias)` doet hetzelfde voor de bias.",
        ],
        why="De update is lineair, dus kan hij bij de weight worden opgeteld: één matrix geeft dezelfde output, en het model schrijft even snel als eerst.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _m = C.MEASURED["yours"]
    _r = _m["rank"]
    mo.vstack([
        mo.md(f"""
        ### Finetune je model met LoRA

        De knop doet wat de demo van [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl) deed, met je `sft_example`, `sft_batch` en `pretrain_step`, maar op je model uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl) met je `add_lora` van de rang die je kiest en alfa twee keer de rang: {C.number(C.STEPS, 0, L.lang)} stappen met een learning rate van {C.number(C.LR, 3, L.lang)}, vijf keer die van full fine-tuning, omdat de update op nul begint en weinig getallen heeft. Dan beantwoordt het model dezelfde {C.T.TESTS} testinstructies, geteld met je `words_used`, en voegt je `merge_lora` elke adapter samen, om de logits te vergelijken. Op de CPU van de course, met één thread, duurde rang {_r} ongeveer {round((_m['seconds'] + _m['other_seconds']) / 60)} minuten voor haar model uit de demorun van de capstone; het eigen model van de course, ongeveer vijf keer zo groot, doet er op twee threads ongeveer een halfuur over. De adapters worden bewaard in `work/projects/llm/lora.pt`.

        Op een CPU, en met een model dat zo klein is, is een stap LoRA niet sneller dan een stap full fine-tuning: de backward pass gaat nog door elke layer. LoRA spaart geheugen, en bestandsgrootte: de optimizer houdt zijn twee getallen alleen bij voor wat leert.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def rank_cell():
    rank_pick = mo.ui.dropdown(options={str(_r): _r for _r in C.RANKS}, value=str(C.RANK), label="Rang")
    lora_button = mo.ui.run_button(label="Finetune met LoRA")
    mo.hstack([rank_pick, lora_button], align="end", justify="start")
    return lora_button, rank_pick


@app.cell(hide_code=True)
def demo_lora(box_add, box_layer, box_merge, lora_button, rank_pick, result_add, result_layer, result_merge):
    adapted = None
    _note = None
    if not all(_r.state == "correct" for _r in (result_layer, result_add, result_merge)):
        _note = "*Maak eerst oefening 1 tot en met 3 af: de demo finetunet met je `LoRALinear` en `add_lora`, en voegt samen met je `merge_lora`.*"
    elif lora_button.value:
        def _progress(_p):
            if _p["phase"] == "pairs":
                mo.output.replace(mo.md(f"**{C.number(_p['pairs'], 0, L.lang)} voorbeelden** · adapters van rang {rank_pick.value} …"))
            elif _p["phase"] == "after":
                mo.output.replace(mo.md(f"**Na LoRA** · instructie {_p['done']} van {_p['total']} · elk woord gebruikt: {_p['used']}"))
            else:
                mo.output.replace(mo.md(f"**Stap {C.number(_p['step'], 0, L.lang)} van {C.number(_p['steps'], 0, L.lang)}** · loss {C.number(_p['loss'], 3, L.lang)} · {C.number(_p['seconds'] / 60, 1, L.lang)} minuten"))
        _r = C.run(rank_pick.value, _progress)
        if _r == "run":
            _note = "*De demo stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks? Verwijder een bestand dat je veranderde, en check zijn oefening opnieuw.*"
        elif _r == "lora":
            _note = "*Je adapters stoppen met een fout op het model of bij het samenvoegen, veranderen wat het geeft voor er getraind is, of laten niets over om te leren. Slagen je `LoRALinear`, `add_lora` en `merge_lora` nog voor hun checks?*"
        elif _r == "memory":
            _note = "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop.*"
        elif _r == "diverged":
            _note = "*De loss is geen getal meer: de training is ontspoord. Probeer een lagere rang, of druk opnieuw op de knop.*"
        else:
            adapted = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (adapted,)


@app.cell(hide_code=True)
def demo_result(adapted):
    _m = C.MEASURED["yours"]
    _run = adapted if adapted is not None else C.saved() or _m
    _mine = adapted is not None or C.saved() is not None
    _f = C.full_sft(_run.get("base", _run.get("source")) if _mine else None)
    _same = (_f["source"] == "yours") == _mine   # both rows on one model: the learner's two runs, or the course's two
    _rows = [f"| full fine-tuning | {C.number(_f['params'], 0, L.lang)} | {C.number(2 * _f['params'], 0, L.lang)} | **{_f['used']}** van {_f['total']} |",
             f"| LoRA, rang {_run['rank']} | {C.number(_run['trainable'], 0, L.lang)} | {C.number(2 * _run['trainable'], 0, L.lang)} | **{_run['after']['used']}** van {_run['after']['total']} |"]
    _whose = "je model" if _mine and _run.get("base", _run.get("source")) == "yours" else "het model van de course" if _mine else "het model uit de demorun van de capstone van de course"
    _stand = "" if _mine else "\n\nDe runs van de course staan er tot je je eigen model finetunet."
    _sft = "" if _f["source"] == "yours" else " De rij van full fine-tuning is de run van de course van de demo uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl)."
    _gap = f" Samengevoegd geeft het model dezelfde logits, op {C.small(_run['gap'], L.lang)} na." if _mine else ""
    _size = f" De adapters nemen {C.number(_run['bytes'] / 1000, 0, L.lang)} kB in `lora.pt`; het hele model neemt {C.number(4 * _run['params'] / 1e6, 1, L.lang)} MB in float32."
    _before = f" (vóór SFT: {_f['before']} goed)" if _same else ""
    mo.md(f"De antwoorden van {_whose} op {_run['after']['total']} achtergehouden instructies{_before}:\n\n"
          "| | getallen die leren | getallen die AdamW bijhoudt | elk woord gebruikt |\n|---|---|---|---|\n" + "\n".join(_rows) + _stand
          + f"\n\n{_sft}{_gap}{_size}")
    return


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Oefening 4. Waarom nul?
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "why-zero",
        question="LoRA laat `B` op nul beginnen en `A` willekeurig. Waarom niet allebei op nul?",
        options={
            "memory": "Nullen kosten minder geheugen, dus één matrix met nullen is genoeg",
            "right": "Met allebei op nul krijgt geen van beide ooit een andere gradient dan nul",
            "random": "Allebei op nul zou net zo goed werken; een willekeurige `A` is alleen een gewoonte",
        },
        correct="right",
        why={
            "memory": "Een matrix met nullen kost evenveel geheugen als elke andere. De reden is de gradient: de gradient van `B` loopt via `A`, en die van `A` via `B`.",
            "random": "Met allebei op nul is de gradient van `B` nul omdat `A` nul is, en die van `A` omdat `B` nul is: er zou nooit iets leren.",
            "right": "Eén van beide op nul is genoeg om het model te laten beginnen zoals het was; de andere moet willekeurig zijn, zodat de eerste een gradient krijgt.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De low-rank matrices uit [Rang: hoeveel richtingen een matrix heeft](/course/open/math-matrix-rank?lang=nl), nu als wat een model leert; de gradients van matrixproducten uit [Gradients van matrixuitdrukkingen: volg de shapes](/course/open/math-matrix-calculus?lang=nl), die `B` via `A` bereiken; en de voorbeelden, batches en check uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), nu voor adapters. Je project `llm` heeft nu `lora.py`, `add_lora.py` en `merge_lora.py`, en na de demo je adapters in `lora.pt`. In een latere les, over DPO, is een adapter wat leert, en is hetzelfde model met zijn adapters uit het model waarmee het vergeleken wordt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_layer, result_add, result_merge, choice_zero):
    L.wrap_up(result_layer, result_add, result_merge, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
