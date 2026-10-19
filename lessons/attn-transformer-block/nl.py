# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-transformer-block. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("attn-transformer-block", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl) en [Layer normalization](/course/open/nn-layernorm?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_place():
    quick_place = L.choice(
        "quick-place",
        question="Een GPT-2 heeft geleerde position embeddings. Waar komt de plek van een token binnen?",
        options={
            "right": "Eén keer, vóór de eerste layer: opgeteld bij de embedding van het token",
            "scores": "In elke attention head, opgeteld bij zijn scores",
            "mask": "Nergens: de causal mask geeft de volgorde al",
        },
        correct="right",
        why={
            "scores": "Sommige nieuwere modellen veranderen de scores, zoals ALiBi. GPT-2 telt een geleerde vector voor elke plek op bij de token-embeddings. Zie [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl).",
            "mask": "De causal mask zegt welke tokens ervoor kwamen, niet in welke volgorde. Zie [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl).",
            "right": "Vandaag bouw je het block dat die vectoren krijgt, en ze doorgeeft aan het volgende block.",
        },
    )
    quick_place
    return (quick_place,)


@app.cell(hide_code=True)
def quick_check_prenorm():
    quick_prenorm = L.choice(
        "quick-prenorm",
        question="In een pre-norm block, `x + f(norm(x))`, wat gaat er langs het directe pad, van de input rechtdoor naar de output?",
        options={
            "normalized": "De genormaliseerde x",
            "right": "x zelf, onveranderd",
            "f": "Wat f maakt van de genormaliseerde x",
        },
        correct="right",
        why={
            "normalized": "De normalisatie zit alleen op de weg naar f. Het directe pad draagt x zoals hij binnenkwam. Zie [Layer normalization](/course/open/nn-layernorm?lang=nl).",
            "f": "Dat is het andere pad, door f; dat wordt aan het eind bij x opgeteld. Zie [Layer normalization](/course/open/nn-layernorm?lang=nl).",
            "right": "Vandaag heeft een block er twee na elkaar: één met attention als f, één met een MLP.",
        },
    )
    quick_prenorm
    return (quick_prenorm,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een GPT is één block, keer op keer: de kleinste GPT-2 stapelt er 12, GPT-3 96. Elk block heeft attention, zoals in [Multi-head attention](/course/open/attn-multi-head?lang=nl), en een MLP, elk met layer normalization en een residual connection. In deze projectstap bouw je dat block in je project, voor een mini-batch van teksten tegelijk, zoals een GPT traint. Het is het eerste deel van je eigen GPT; je eigen tokenizer komt hierna op de route, en dan de hele GPT.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           4, here_label="← vandaag: zijn block", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_params():
    guess_params = L.guess(
        "params",
        question="Eén block van GPT-2 heeft $d = 768$. Zijn attention heeft vier matrices van 768 bij 768; zijn MLP maakt van elke vector 3.072 getallen en dan weer 768, met één matrix voor elke kant op. Welk deel heeft de meeste parameters?",
        options={
            "attention": "De attention: vier matrices tegen twee",
            "right": "Het MLP: ongeveer twee keer zoveel als de attention",
            "equal": "Allebei ongeveer evenveel",
        },
        reveal="**Het MLP: ongeveer twee keer zoveel als de attention.** De attention heeft vier matrices van 768 bij 768, voor de queries, keys, values en de output-projectie: 2.359.296 weights, en 3.072 bias terms. Het MLP heeft twee matrices van 768 bij 3.072: 4.718.592 weights, en 3.840 bias terms. De twee layer normalizations hebben er samen 3.072. Eén block heeft 7.087.872 parameters.",
    )
    guess_params
    return (guess_params,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **transformer block** krijgt de vectoren van $n$ tokens, van elk $d$ getallen, en geeft $n$ nieuwe vectoren van dezelfde grootte terug. Het heeft twee delen, elk met een layer normalization ervoor en een residual connection eromheen, zoals je `PreNormBlock` uit [Layer normalization](/course/open/nn-layernorm?lang=nl): eerst multi-head causale attention, dan een MLP, een hidden layer voor de vector van elk token:
        """),
        L.formula(r"\begin{aligned} [[hidden:\mathbf{X}]] &\leftarrow [[hidden:\mathbf{X}]] + \mathrm{MultiHead}\big(\mathrm{LN}_1([[hidden:\mathbf{X}]])\big) \\ [[hidden:\mathbf{H}]] &= \mathrm{ReLU}\big(\mathrm{LN}_2([[hidden:\mathbf{X}]])\,[[param:\mathbf{W}_1]] + [[param:\mathbf{b}_1]]\big) \\ [[hidden:\mathbf{X}]] &\leftarrow [[hidden:\mathbf{X}]] + [[hidden:\mathbf{H}]]\,[[param:\mathbf{W}_2]] + [[param:\mathbf{b}_2]] \end{aligned}"),
        mo.md(r"""
        Het MLP maakt van elke vector $4d$ getallen, $\mathbf{H}$, en dan weer $d$, zoals in GPT-2. GPT-2 gebruikt daar GELU, een gladdere ReLU; hier is het ReLU. Elk deel telt op bij $\mathbf{X}$ en vervangt het nooit, dus de vector van elk token loopt rechtdoor door alle blocks: de **residual stream**. Elk deel leest een genormaliseerde kopie ervan en schrijft wat het vond terug, door op te tellen. Elk block van een GPT heeft eigen weights.

        Een GPT traint op een mini-batch van $B$ teksten tegelijk, een tensor met shape $(B, n, d)$. `nn.Linear` en `nn.LayerNorm` werken op de laatste axis, wat er ook voor staat, dus ze werken ongewijzigd op een mini-batch. Attention moet anders: elke tekst mag alleen zijn eigen tokens vergelijken. `@` op twee tensors met 3 axes vermenigvuldigt ze tekst voor tekst, en `k.transpose(1, 2)` wisselt axis 1 en 2, zodat elke tekst de getransponeerde van zijn eigen keys krijgt. `q @ k.transpose(1, 2)` geeft dan $B$ matrices met scores van $n$ bij $n$, één per tekst. De kolommen van een head zijn `q[:, :, a:b]`. De softmax en het naast elkaar zetten van de heads gaan langs axis 2, en de mask van $n$ bij $n$ geldt voor elke tekst tegelijk, door broadcasting.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **transformer block** | | multi-head causale attention en een MLP, elk met layer normalization ervoor en een residual connection eromheen | `Block(d, h)` |
        | **residual stream** | | de vectoren van de tokens op weg door alle blocks; elk deel van elk block telt erbij op | `x = x + ...` |
        | $(B, n, d)$ | B bij n bij d | de shape van een mini-batch van teksten: teksten, tokens, getallen per token | `x.shape` |
        | `transpose(1, 2)` | transpose 1 2 | wisselt axis 1 en 2 van een tensor: in een mini-batch de getransponeerde van elke tekst | `k.transpose(1, 2)` |
        """),
        L.watch_out("Normaliseer een kopie, niet de stream: in `x = x + self.attention(self.norm1(x))` gaat `x` zelf onveranderd verder. En `.T` is niet de getransponeerde van een mini-batch: op een tensor met 3 axes draait het alle axes om, $(B, n, d)$ wordt $(d, n, B)$."),
        mo.md(r"""
        De figuur: één block. De residual stream loopt recht naar beneden; elk deel neemt er een genormaliseerde kopie van en telt zijn uitkomst er weer bij op, bij ⊕.
        """),
        mo.Html(C.block_picture({"title": "Eén transformer block", "input": "x: n vectoren van d getallen", "output": "naar het volgende block",
                                 "norm": "layer normalization", "attention": "multi-head causale attention", "mlp": "MLP: d → 4d → d",
                                 "stream": "residual stream",
                                 "alt": "De residual stream loopt van x, n vectoren van d getallen, recht naar beneden naar het volgende block. Er gaan twee takken af: layer normalization, dan multi-head causale attention, weer opgeteld bij de stream; dan layer normalization, dan een MLP van d naar 4d naar d getallen, opnieuw opgeteld."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een transformer block voor een mini-batch $x$ met shape $(B, n, d)$, met $h$ heads:

    1. Een mini-batch $x$ met shape $(B, n, d)$; zijn queries, keys en values, met dezelfde shape, en een mask van $n$ bij $n$.
    2. Head `i`, voor elke tekst tegelijk: zijn scores `q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2)`, geschaald en gemaskeerd, de softmax langs axis 2, maal zijn values `v[:, :, a:b]`.
    3. De outputs van de heads naast elkaar langs axis 2, dan de output-projectie.
    4. Het block: $x$ plus de attention van de genormaliseerde $x$; dan $x$ plus het MLP van de genormaliseerde $x$: van $d$ naar $4d$ getallen, ReLU, en terug naar $d$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee teksten van 3 tokens, met elk 4 getallen: de scores van één head voor beide teksten tegelijk, en de MLP-helft van een block. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_block():
    import math
    import torch
    from torch import nn

    torch.manual_seed(0)
    # 1. een mini-batch van 2 teksten van 3 tokens, elk 4 getallen
    _x = torch.randn(2, 3, 4)
    _query, _key = nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
    _norm, _lin1, _lin2 = nn.LayerNorm(4), nn.Linear(4, 16), nn.Linear(16, 4)
    with torch.no_grad():
        # 2. de scores van één head van 4 getallen, voor beide teksten tegelijk
        _q, _k = _query(_x), _key(_x)
        _S = _q @ _k.transpose(1, 2) / math.sqrt(4)
        print(_k.shape, _k.transpose(1, 2).shape, _S.shape)
        # 4. de MLP-helft van een block: x plus het MLP van de genormaliseerde x
        _y = _lin2(torch.relu(_lin1(_norm(_x))))  # maak af: zet _x + ervoor
        print(_x[0, 0])
        print(_y[0, 0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De keys hebben shape (2, 3, 4); gewisseld (2, 4, 3); en de scores (2, 3, 3): een matrix van 3 bij 3 voor elke tekst. Zoals de cel nu is, is `_y[0, 0]` gelijk aan `[-0.1175, 0.0982, -0.0675, -0.3728]`: alleen wat het MLP van het eerste token maakt, en zijn eigen vector `[-1.1258, -1.1524, -0.2506, -0.4339]` is weg. Afgemaakt is het `[-1.2433, -1.0542, -0.3181, -0.8067]`: de vector van het token, plus die kleine verandering. Dat is de residual stream.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_no_attention():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een GPT zonder attention
    """)
    return


@app.cell(hide_code=True)
def choice_no_attention_cell():
    choice_no_attention = L.choice(
        "no-attention",
        question="Je haalt de attention uit elk block van een GPT, en houdt de layer normalizations en de MLP's. Waaruit kan die GPT dan nog het volgende token voorspellen?",
        options={
            "before": "Uit alle tokens ervoor: de residual stream draagt ze mee",
            "few": "Uit de laatste paar tokens, zoals het neural language model deed",
            "right": "Alleen uit het token zelf en zijn plek",
        },
        correct="right",
        why={
            "before": "De residual stream van een token draagt alleen wat bij de eigen vector van dat token wordt opgeteld. Zonder attention wordt er nooit iets van andere tokens bij opgeteld.",
            "few": "Dat model uit [Een neural language model dat verder terugkijkt](/course/open/lm-mlp-context?lang=nl) zette de laatste paar embeddings naast elkaar vóór zijn MLP. Het MLP van een block krijgt de vector van één token tegelijk.",
            "right": "Niets anders bereikt zijn vector: layer normalization en het MLP werken op de vector van elk token apart. Attention is de enige plek waar de tokens van een tekst informatie uitwisselen.",
        },
    )
    choice_no_attention
    return (choice_no_attention,)


@app.cell(hide_code=True)
def text_exercise_attention():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Attention voor een mini-batch

        Maak `CausalSelfAttention(d, h)` af: de multi-head attention uit [Multi-head attention](/course/open/attn-multi-head?lang=nl), voor een mini-batch `x` met shape $(B, n, d)$. `__init__`, de mask en de loop over de heads zijn gegeven; vervang de twee `...` door stap 2 en 3 van het recept. De check maakt hem na `torch.manual_seed(0)` en geeft hem 2 teksten van 3 tokens; daarna probeert hij 3 heads en langere teksten. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/attention.py`.
        """),
        L.task("Laat `CausalSelfAttention` causale attention draaien in elke head, voor elke tekst van een mini-batch tegelijk."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attention():
    box_attention = L.code_box("causal-self-attention", lines=26)
    box_attention
    return (box_attention,)


@app.cell(hide_code=True)
def check_attention(box_attention):
    result_attention = L.check("causal-self-attention", box_attention.value)
    result_attention
    return (result_attention,)


@app.cell(hide_code=True)
def hints_attention():
    L.hints(
        "causal-self-attention",
        [
            "De regels van causale attention uit [De causal mask](/course/open/attn-causal-mask?lang=nl), met `[:, :, a:b]` voor de kolommen van head `i`, `transpose(1, 2)` in plaats van `.T`, en axis 2 in plaats van axis 1.",
            "Stap 2 van het uitgewerkte voorbeeld geeft de scores, gedeeld door `math.sqrt(self.d_head)`; `S.masked_fill(future, float(\"-inf\"))` verbergt de toekomst in elke tekst tegelijk. Na de loop gaat `torch.cat(outputs, dim=2)` door `self.out`.",
        ],
        why="De scores hebben shape $(B, n, n)$ en de mask $(n, n)$: PyTorch gebruikt dezelfde mask voor elke tekst. `@` vermenigvuldigt tekst voor tekst, dus de teksten van een mini-batch mengen nooit.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block(result_attention):
    result_attention  # opnieuw gemaakt na de check van oefening 2, die je attention opslaat
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je transformer block

        Maak `Block(d, h)` af: in `__init__` de vijf layers, in de volgorde waarin de data erdoor gaat; in `forward` stap 4 van het recept. De eerste regels importeren je `CausalSelfAttention` uit je project. De check maakt het block na `torch.manual_seed(0)` en geeft het 2 teksten van 3 tokens met elk 8 getallen; daarna stuurt hij een tekst twee keer door het block, en bekijkt hij de shapes van de parameters. Als het goed is, bewaart de check het als `work/projects/llm/block.py`.
        """),
        L.course_versions(),
        L.task("Laat `Block` de attention en dan het MLP, elk van een genormaliseerde kopie, optellen bij de residual stream."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("block", lines=16)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block, result_attention):
    result_attention  # opnieuw checken als oefening 2 je attention opslaat
    result_block = L.check("block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "block",
        [
            "`__init__` is dat van je `PreNormBlock` uit [Layer normalization](/course/open/nn-layernorm?lang=nl), twee keer: `nn.LayerNorm(d)` en `CausalSelfAttention(d, h)`, dan `nn.LayerNorm(d)` en de twee `nn.Linear`-layers van het MLP, `d` naar `4 * d` en terug.",
            "`forward` heeft twee regels zoals `x = x + self.attention(self.norm1(x))`: de tweede met het MLP uit het uitgewerkte voorbeeld, `self.lin2(torch.relu(self.lin1(...)))`, van `self.norm2(x)`; dan `return x`.",
        ],
        why="Het block geeft de shape terug die het krijgt, $(B, n, d)$, dus blocks kun je stapelen: wat uit het ene komt, gaat het volgende in, langs de residual stream.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij stapelt 4 van je blocks, met $d = 32$ in 4 heads, gemaakt na `torch.manual_seed(0)`, en stuurt 2 teksten van 8 willekeurige vectoren erdoorheen. Dan verandert hij het laatste token van de eerste tekst, en kijkt hij wat er verandert. Ook maakt hij één block van de grootte van GPT-2, $d = 768$ in 12 heads, en telt hij zijn parameters. Het werkt zodra oefening 2 en 3 af zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_blocks(result_attention, result_block):
    result_attention, result_block  # opnieuw uitvoeren als oefening 2 of 3 je module opslaat
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `CausalSelfAttention` en je `Block` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je blocks duren hier te lang: meer dan een halve minuut. Doet je `Block` of je `CausalSelfAttention` meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "block": "*`Block` of `CausalSelfAttention` in `work/projects/llm/` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan opnieuw.*",
        }[_result])
    else:
        _by = lambda v: "helemaal niet" if v == 0 else "met hooguit " + C.number(v, 3, L.lang)
        _ahead = ("je blocks kijken nooit vooruit" if _result["before"] == 0
                  else "dus je blocks kijken vooruit: bekijk de mask van je `CausalSelfAttention`")
        _mixed = ("en de teksten van een mini-batch mengen niet" if _result["other"] == 0
                  else "dus de teksten van de mini-batch mengen: bekijk de axes in je `CausalSelfAttention`")
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij oefening 3 zegt."
        _out = mo.md(f"""Uit je 4 blocks komt shape {tuple(_result['shape'])}, de shape die erin ging. Als het laatste token van de eerste tekst verandert, verandert zijn eigen nieuwe vector met hooguit {C.number(_result['last'], 3, L.lang)}, die van de 7 tokens ervoor {_by(_result['before'])}, en die van de tweede tekst {_by(_result['other'])}: {_ahead}, {_mixed}.

Eén block van de grootte van GPT-2 heeft **{C.grouped(_result['params'], L.lang)}** parameters, dus de 12 blocks van de kleinste GPT-2 hebben er {C.grouped(12 * _result['params'], L.lang)}. GPT-2 zelf heeft er 2.304 meer per block: bias terms op zijn queries, keys en values.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De multi-head attention uit [Multi-head attention](/course/open/attn-multi-head?lang=nl), nu voor een mini-batch, en het pre-norm block met een residual connection uit [Layer normalization](/course/open/nn-layernorm?lang=nl), twee keer: dat is het transformer block. De posities uit [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl) gaan er aan het begin in, en de residual stream draagt ze door elk block. Je project `llm` heeft nu `attention.py` en `block.py`. In *Je GPT in elkaar zetten* worden embeddings, een stapel van je blocks en een output head je GPT. Eerst bouw je op de hoofdroute je eigen tokenizer, te beginnen met *Byte pair encoding: bouw je eigen tokenizer*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_no_attention, result_attention, result_block):
    L.wrap_up(choice_no_attention, result_attention, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
