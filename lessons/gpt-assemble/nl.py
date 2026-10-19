# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les gpt-assemble. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-assemble", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Het transformer block](/course/open/attn-transformer-block?lang=nl) en [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stack():
    quick_stack = L.choice(
        "quick-stack",
        question="Een stapel van 12 transformer blocks krijgt een mini-batch met shape (B, n, d). Welke shape komt er uit het laatste block?",
        options={
            "wide": "(B, n, 12d): elk block voegt d getallen toe",
            "right": "(B, n, d): de shape die erin ging",
            "one": "(B, d): één vector per tekst",
        },
        correct="right",
        why={
            "wide": "Een block telt zijn uitkomsten getal voor getal op bij de residual stream; de stream houdt zijn d getallen. Zie [Het transformer block](/course/open/attn-transformer-block?lang=nl).",
            "one": "Elk token houdt zijn eigen vector, door elk block heen. Zie [Het transformer block](/course/open/attn-transformer-block?lang=nl).",
            "right": "Elk block geeft de shape terug die het krijgt, dus je kunt ze stapelen. Vandaag stapel je ze tot een GPT.",
        },
    )
    quick_stack
    return (quick_stack,)


@app.cell(hide_code=True)
def quick_check_modulelist():
    quick_modulelist = L.choice(
        "quick-modulelist",
        question="Waarom bewaart een netwerk een lijst layers in een `nn.ModuleList`, en niet in een gewone Python-lijst?",
        options={
            "right": "Zodat PyTorch hun parameters ziet, en een optimizer ze traint",
            "faster": "Omdat een gewone lijst te traag is om de layers van een netwerk te bewaren",
            "order": "Zodat de layers in de goede volgorde draaien",
        },
        correct="right",
        why={
            "faster": "Snelheid heeft er niets mee te maken: een gewone lijst verbergt de layers voor `model.parameters()`. Zie [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).",
            "order": "Een gewone lijst houdt ook zijn volgorde. Het verschil is dat PyTorch er niet in kijkt. Zie [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).",
            "right": "Vandaag bevat een `nn.ModuleList` de transformer blocks van je GPT.",
        },
    )
    quick_modulelist
    return (quick_modulelist,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Alle onderdelen zijn er. Je tokenizer maakt van een tekst token-ids, je embeddings uit [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl) geven elk token en elke plek een vector, en je blocks uit [Het transformer block](/course/open/attn-transformer-block?lang=nl) laten de tokens kijken naar de tokens ervoor. Een GPT is die onderdelen op een rij, met aan het eind een output head die op elke plek een score geeft voor elk mogelijk volgend token. De kleinste GPT-2 is zo gebouwd: 12 blocks met $d = 768$, en een vocabulary van 50.257 tokens. In deze projectstap zet je je eigen GPT in elkaar, en maak je zijn trainingsvoorbeelden klaar.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_loss():
    guess_loss = L.guess(
        "untrained-loss",
        question="Je GPT voor Tiny Shakespeare, met zijn 65 tekens, krijgt vóór het trainen vensters tekst, met willekeurige weights. Welke loss verwacht je op de volgende tekens?",
        options={
            "zero": "Ongeveer 0: hij heeft alle weights die hij nodig heeft",
            "right": "Ongeveer 4,2: even goed als gelijkmatig gokken uit 65",
            "big": "Ongeveer 65: één voor elk teken dat hij fout heeft",
        },
        reveal="**Ongeveer 4,2.** Gelijkmatig gokken uit 65 tekens geeft elk een kans van 1/65, en een cross-entropy van $\\ln 65 = 4{,}17$. Willekeurige weights geven logits die bijna gelijk zijn, dus de loss ligt daar dichtbij; de demo hieronder vindt 4,40. Het trainen moet hem vanaf daar omlaag brengen.",
    )
    guess_loss
    return (guess_loss,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **GPT** is een **decoder-only transformer**: alleen causale attention, dus elk token kijkt terug en nooit vooruit. De transformer van 2017 had ook een encoder, die voor een vertaling een hele invoerzin tegelijk las; een GPT heeft zo'n deel niet. Van token-ids naar logits gaat het zo:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{X}]] &= [[param:E]][\text{ids}] + [[param:P]][0, \dots, n-1] \\ [[hidden:\mathbf{X}]] &\leftarrow \mathrm{Block}([[hidden:\mathbf{X}]]) \\ [[pred:\text{logits}]] &= \mathrm{LN}([[hidden:\mathbf{X}]])\,[[param:\mathbf{W}]] + [[param:\mathbf{b}]] \end{aligned}"),
        mo.md(r"""
        $E$ is de tabel van tokens en $P$ de tabel van plekken, `block` rijen lang: een GPT leest hooguit `block` tokens tegelijk. De tweede regel draait één keer voor elk block, `n_layers` keer, en elk block heeft eigen weights: `nn.ModuleList([Block(d, h) for _ in range(n_layers)])` maakt een nieuw `Block` voor elke plek in de lijst. In de blocks werden alleen kopieën van de stream genormaliseerd, dus vóór de output head komt nog een laatste layer normalization, en de output head is een lineaire layer van $d$ naar de grootte van de vocabulary. De logits hebben shape $(B, n, \text{vocab})$: een score voor elk mogelijk volgend token, op elke plek van elke tekst.

        Zo leert een GPT ook: een venster van `block` token-ids gaat erin, en het target op elke plek is het token erna, dus de targets zijn hetzelfde venster, één id verder. Eén venster geeft `block` voorbeelden tegelijk, en de loss is de cross-entropy over alle plekken.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **decoder-only transformer** | een transformer van alleen causale blocks, zonder encoder | `GPT(vocab, d, h, n_layers, block)` |
        | **GPT** | een decoder-only transformer die het volgende token voorspelt; generative pretrained transformer | `model(ids)` |
        | `block` | het grootste aantal tokens dat een GPT tegelijk leest: de rijen van zijn tabel van plekken | `GPT(..., block)` |
        """),
        L.watch_out("De plekken die je opzoekt zijn die van deze tekst, `torch.arange(n)` met `n = ids.shape[1]`, niet alle `block`: een tekst kan korter zijn dan `block`, en dan passen de shapes niet. En een venster en zijn targets zijn even lang: het target van de laatste plek is het id direct na het venster."),
        mo.md(r"""
        De figuur: je GPT van boven naar beneden, met de shape van wat er uit elk deel komt. De ids zijn data, de residual stream loopt van de embeddings door alle blocks, en de output head geeft de logits.
        """),
        mo.Html(C.gpt_picture({"title": "Je GPT", "ids": "token-ids", "embed": "tokens + plekken", "block1": "block 1", "blockL": "laatste block",
                               "norm": "layer normalization", "head": "output head: de logits",
                               "alt": "Van boven naar beneden: token-ids met shape (B, n); tokens plus plekken, (B, n, d); block 1 tot en met het laatste block, elk (B, n, d); layer normalization, (B, n, d); en de output head met de logits, (B, n, vocab)."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een GPT met een vocabulary van `vocab` tokens, vectoren van $d$ getallen, $h$ heads, `n_layers` blocks, en teksten van hooguit `block` tokens:

    1. De ids van B teksten van n tokens, een tensor met shape (B, n), met n hooguit `block`.
    2. De embeddings van de tokens plus die van hun plekken: de residual stream, (B, n, d).
    3. Elk block om de beurt, dan layer normalization en de output head: de logits, (B, n, vocab).
    4. Trainingsvoorbeelden: vensters van `block` ids naast elkaar, zolang er nog één id volgt, elk met hetzelfde venster één verder als targets.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 tot en met 3 voor 2 teksten van 3 tokens, met een vocabulary van 10 en $d = 8$. De blocks houden de shape, dus die blijven hier weg. De eerste tekst heeft token 1 op plek 0 en op plek 2; de cel print hoeveel hun logits verschillen. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_gpt():
    import torch
    from torch import nn

    torch.manual_seed(0)
    # 1. 2 teksten van elk 3 ids
    _ids = torch.tensor([[1, 2, 1], [4, 5, 6]])
    _n = _ids.shape[1]
    _tok, _pos = nn.Embedding(10, 8), nn.Embedding(6, 8)
    _norm, _head = nn.LayerNorm(8), nn.Linear(8, 10)
    with torch.no_grad():
        # 2. de tokens plus hun plekken: de residual stream
        _x = _tok(_ids)  # maak af: + _pos(torch.arange(_n))
        print(_x.shape)
        # 3. (de blocks houden de shape) layer normalization en de output head: de logits
        _logits = _head(_norm(_x))
        print(_logits.shape)
        print((_logits[0, 0] - _logits[0, 2]).abs().max())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De stream heeft shape (2, 3, 8), en de logits (2, 3, 10): 10 scores op elk van 3 plekken van 2 teksten. Zoals de cel nu is, is het verschil tussen de logits van token 1 op plek 0 en op plek 2 `tensor(0.)`: zonder zijn plek heeft token 1 op beide plekken dezelfde vector, en dus dezelfde logits. Afgemaakt is het grootste verschil 1,0345.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De shape van de logits
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "logit-shape",
        question="Je GPT met een vocabulary van 65 tekens krijgt 32 vensters van 64 ids. Welke shape hebben zijn logits?",
        options={
            "window": "(32, 65): één gok per venster, voor het id erna",
            "place": "(32, 64): één getal per plek",
            "right": "(32, 64, 65): een gok voor het volgende id op elke plek",
        },
        correct="right",
        why={
            "window": "Dat was je model uit [Een neural language model dat verder terugkijkt](/course/open/lm-mlp-context?lang=nl). Een GPT gokt op elke plek tegelijk.",
            "place": "Eén getal kan niet zeggen welk van 65 tekens volgt: elke plek heeft een score voor elk teken nodig.",
            "right": "32 × 64 = 2.048 gokken uit één forward pass, elk met een eigen target: één pass traint op ze allemaal tegelijk.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_gpt():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je GPT

        Maak `GPT(vocab, d, h, n_layers, block)` af: in `__init__` zijn vijf onderdelen, in de volgorde waarin de data erdoor gaat; in `forward` stap 2 en 3 van het recept, voor ids met shape (B, n). De eerste regels importeren je `Block` uit je project. De check maakt hem na `torch.manual_seed(0)` en geeft hem 2 teksten van 3 tokens; daarna probeert hij andere maten, en telt hij de tensors met parameters die PyTorch ziet. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/gpt.py`.
        """),
        L.task("Laat `GPT` van token-ids logits maken: embeddings, elk block, layer normalization, output head."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gpt():
    box_gpt = L.code_box("gpt", lines=18)
    box_gpt
    return (box_gpt,)


@app.cell(hide_code=True)
def check_gpt(box_gpt):
    result_gpt = L.check("gpt", box_gpt.value)
    result_gpt
    return (result_gpt,)


@app.cell(hide_code=True)
def hints_gpt():
    L.hints(
        "gpt",
        [
            "`__init__`: de vijf delen van het idee, in de volgorde van de data: een tabel met een rij voor elk token en een met een rij voor elke plek, elke rij `d` getallen; `n_layers` nieuwe blocks, in een lijst waar PyTorch in kan kijken; een layer normalization van `d` getallen; en de output head, van `d` getallen naar een score voor elk token.",
            "`forward`: stap 2 van het afgemaakte uitgewerkte voorbeeld, met `self.tok`, `self.pos` en `n`; dan elk block om de beurt op de stream, `x = ...`; dan stap 3.",
        ],
        why="De blocks draaien na elkaar, elk op wat het vorige gaf, langs de residual stream. Pas na het laatste block krijgt de stream zijn laatste normalisatie en wordt hij logits.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_examples():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainingsvoorbeelden

        Maak `gpt_examples(ids, block)` af: de functie geeft de trainingsvoorbeelden van een lijst token-ids, stap 4 van het recept, als twee lijsten: `X`, de vensters van `block` ids, en `Y`, dezelfde vensters één id verder. Het is de `contexts` uit [Een neural language model dat verder terugkijkt](/course/open/lm-mlp-context?lang=nl), met een heel venster als target. De check roept `gpt_examples([5, 6, 7, 8, 9, 10, 11], 3)` aan, wat `([[5, 6, 7], [8, 9, 10]], [[6, 7, 8], [9, 10, 11]])` moet geven, en probeert andere lijsten. Als hij goed is, bewaart de check hem als `work/projects/llm/gpt_data.py`.
        """),
        L.task("Laat `gpt_examples` een lijst ids in vensters naast elkaar knippen, elk met zijn targets één id verder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_examples():
    box_examples = L.code_box("gpt-examples", lines=10)
    box_examples
    return (box_examples,)


@app.cell(hide_code=True)
def check_examples(box_examples):
    result_examples = L.check("gpt-examples", box_examples.value)
    result_examples
    return (result_examples,)


@app.cell(hide_code=True)
def hints_examples():
    L.hints(
        "gpt-examples",
        [
            "De vensters beginnen bij 0, `block`, `2 * block`, enzovoort: een `range` met een stap. Het laatste begin moet ruimte laten voor het venster en één id erna.",
            "Een venster is `ids[i:i + block]`; zijn targets zijn dezelfde slice, met beide uiteinden één verder.",
        ],
        why="`range(0, len(ids) - block, block)` stopt vóór een begin waar het venster op het laatste id zou eindigen, dus elk venster heeft een volledig target: met 6 ids en vensters van 3 zou een venster dat op plek 3 begint op het laatste id eindigen, en zou zijn laatste plek geen target hebben.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij maakt van heel Tiny Shakespeare ids van tekens en maakt je trainingsvoorbeelden met vensters van 64. Dan maakt hij je GPT voor zijn 65 tekens, met $d = 64$, 4 heads en 4 blocks, na `torch.manual_seed(0)`, en laat hem vóór het trainen de volgende tekens van 8 vensters gokken. Het werkt zodra oefening 2 en 3 af zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_gpt(result_gpt, result_examples):
    result_gpt, result_examples  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `GPT` en je `gpt_examples` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je GPT duurt hier te lang: meer dan een minuut. Doet je `GPT` of je `gpt_examples` meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "gpt": "*`GPT` of `gpt_examples` in `work/projects/llm/` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan opnieuw.*",
        }[_result])
    else:
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
        _out = mo.md(f"""Tiny Shakespeare geeft **{C.grouped(_result['windows'], L.lang)}** vensters van 64 tekens. Je GPT heeft {C.grouped(_result['params'], L.lang)} parameters, en voor 8 vensters hebben zijn logits shape {tuple(_result['shape'])}.

Vóór het trainen is zijn loss op die vensters **{C.number(_result['loss'], 2, L.lang)}**; gelijkmatig gokken uit {_result['chars']} tekens geeft {C.number(_result['guess_loss'], 2, L.lang)}. Je GPT weet nog niets, en heeft alles wat hij nodig heeft om te leren.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De embeddings van tokens en plekken uit [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl), je blocks uit [Het transformer block](/course/open/attn-transformer-block?lang=nl) in een `nn.ModuleList` zoals in [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl), en de voorbeelden uit [Een neural language model dat verder terugkijkt](/course/open/lm-mlp-context?lang=nl), nu met een target op elke plek: je eigen GPT. Je project `llm` heeft nu `gpt.py` en `gpt_data.py`. Hierna: *Parameters en FLOPs per token tellen*, waar de parameters van je GPT zitten en wat één token kost; daarna train je hem op een boek.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_gpt, result_examples):
    L.wrap_up(choice_shape, result_gpt, result_examples)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
