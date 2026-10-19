# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-layernorm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow residual

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-layernorm", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Batch normalization](/course/open/nn-batchnorm?lang=nl) en [Residual connections](/course/open/nn-residual?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_over_what():
    quick_over_what = L.choice(
        "quick-over-what",
        question="In train mode standaardiseert batch normalization de $z$'s van elk neuron. Waarover worden het gemiddelde en de variantie genomen?",
        options={
            "all-data": "Over alle training data tegelijk, voordat het trainen begint",
            "right": "Over de voorbeelden in de mini-batch",
            "running": "Over zijn running averages, die het na elke batch bijwerkt",
        },
        correct="right",
        why={
            "all-data": "Het ziet nooit alle data tegelijk: in train mode gebruikt het de mini-batch die het krijgt. Zie [Batch normalization](/course/open/nn-batchnorm?lang=nl).",
            "running": "De running averages zijn voor eval mode. In train mode gebruikt het het gemiddelde en de variantie van de mini-batch zelf. Zie [Batch normalization](/course/open/nn-batchnorm?lang=nl).",
            "right": "Per neuron, over de voorbeelden in de mini-batch: dus elk antwoord hangt af van de andere voorbeelden erin.",
        },
    )
    quick_over_what
    return (quick_over_what,)


@app.cell(hide_code=True)
def quick_check_direct_path():
    quick_direct_path = L.choice(
        "quick-direct-path",
        question="Een block geeft $y = x + f(x)$. Wat komt er op de terugweg bij $x$ aan?",
        options={
            "only-f": "Alleen wat er door $f$ terugkomt, zoals bij elke layer",
            "nothing": "Niets: het optellen houdt de gradient tegen",
            "right": "De gradient voor $y$ zelf, plus wat er door $f$ terugkomt",
        },
        correct="right",
        why={
            "only-f": "Het optellen geeft $x$ een tweede weg, met afgeleide 1. Zie [Residual connections](/course/open/nn-residual?lang=nl).",
            "nothing": "Een optelling geeft de gradient door aan allebei zijn termen. Zie [Residual connections](/course/open/nn-residual?lang=nl).",
            "right": "$\\partial y / \\partial x = 1 + \\partial f / \\partial x$: de directe weg brengt de gradient voor $y$ ongewijzigd terug. Vandaag zie je wat er wel en niet op die weg mag staan.",
        },
    )
    quick_direct_path
    return (quick_direct_path,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Batch normalization heeft een batch nodig: elk antwoord hangt af van de andere voorbeelden erin, kleine batches geven ruwe schattingen, en eval mode heeft running averages nodig. Een language model dat tekst schrijft, werkt aan één reeks tegelijk, van willekeurige lengte. Layer normalization standaardiseert elk voorbeeld op zichzelf, over zijn eigen features, dus dat speelt allemaal niet. Transformers gebruiken het in elk block. Dat roept een tweede vraag op: waar komt de normalisatie in een block met een residual connection?
    """)
    return


@app.cell(hide_code=True)
def predict_where():
    guess_where = L.guess(
        "post-or-pre",
        question="Een MLP voor de cijfers heeft 30 blocks met residual connections, en traint één epoch met een learning rate van 0,3. Zonder normalisatie is zijn loss na 6 stappen `nan`. Nu krijgt elk block een layer normalization: óf na het optellen, zoals in de eerste transformer van 2017, óf aan het begin van $f$, met nog één na het laatste block. Welke traint volgens jou?",
        options={
            "both": "Allebei, ongeveer even goed: een LayerNorm is een LayerNorm",
            "after": "Alleen die met de LayerNorm na het optellen",
            "right": "Alleen die met de LayerNorm aan het begin van $f$",
        },
        reveal="**Alleen die met de LayerNorm aan het begin van $f$.** Na één epoch leest die 784 van de 1.000 testcijfers goed. Met de LayerNorm na het optellen blijft de loss op 2,30, de loss van gokken, en leest hij er 126 goed. Elke LayerNorm op de directe weg verandert wat erdoor terugkomt; in 30 blocks zijn dat 30 veranderingen na elkaar.",
    )
    guess_where
    return (guess_where,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Layer normalization**, kort LayerNorm, standaardiseert elk voorbeeld over zijn eigen features. Voor één voorbeeld met $d$ features $x_1, \dots, x_d$:
        """),
        L.formula(r"\mu = \frac{1}{d} \sum_{j=1}^{d} x_j \qquad \sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2"),
        L.formula(r"\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + [[hyper:\varepsilon]]}} \qquad y_j = [[param:\gamma_j]] \, \hat{x}_j + [[param:\beta_j]]"),
        mo.md(r"""
        Het zijn de formules van batch normalization, omgedraaid: het gemiddelde en de variantie worden genomen over de $d$ features van één voorbeeld, niet over de voorbeelden van een batch. $\gamma$ en $\beta$ hebben nog steeds één getal per feature, geleerd, hetzelfde voor elk voorbeeld. Er zijn geen running averages: LayerNorm doet hetzelfde in train mode en in eval mode, en het werkt voor een batch van één. In PyTorch doet `nn.LayerNorm(64)` dit voor 64 features.

        In een block met een residual connection kan het op twee plekken. **Post-norm**, zoals in de eerste transformer van 2017, normaliseert na het optellen: $y = \mathrm{LN}(x + f(x))$. **Pre-norm**, zoals in GPT-2 en de meeste transformers daarna, normaliseert op weg naar $f$: $y = x + f(\mathrm{LN}(x))$. Met pre-norm staat er niets op de directe weg, dus de gradient komt ongewijzigd terug door elk block, zoals in [Residual connections](/course/open/nn-residual?lang=nl). Omdat de weg zelf nooit genormaliseerd wordt, kunnen zijn getallen van block tot block groeien; een netwerk met pre-norm zet daarom nog één LayerNorm na het laatste block, vóór de output-layer. Zonder die LayerNorm eindigt ook het pre-norm-netwerk uit de gok in `nan`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **layer normalization** | | voor elk voorbeeld: zijn features standaardiseren, en dan keer $\gamma$ plus $\beta$ | `nn.LayerNorm(64)` |
        | $\mu$, $\sigma^2$ | mu, sigma kwadraat | het gemiddelde en de variantie van de features van één voorbeeld, gedeeld door $d$ | `X.mean(dim=1, keepdim=True)` |
        | **post-norm** | | de LayerNorm na het optellen: $\mathrm{LN}(x + f(x))$ | `self.norm(x + f(x))` |
        | **pre-norm** | | de LayerNorm op weg naar $f$: $x + f(\mathrm{LN}(x))$ | `x + f(self.norm(x))` |
        | $\mathrm{LN}(x)$ | L N van x | layer normalization als functie: $x$ gestandaardiseerd over zijn features, keer $\gamma$ plus $\beta$ | `self.norm(x)` |
        | `keepdim=True` | keep dim true | houdt het gemiddelde van elke rij als kolom, zodat je het van elk getal in die rij kunt aftrekken | `X.mean(dim=1, keepdim=True)` |
        """),
        L.watch_out("`keepdim` in PyTorch, `keepdims` in NumPy: hetzelfde, met en zonder s. En post-norm is niet fout: met een kleinere learning rate, of een die aan het begin langzaam groeit, traint het ook goed. Met 0,1 in plaats van 0,3 leest het netwerk van 30 blocks met allebei ongeveer 84% van de testcijfers goed. Pre-norm is de keuze die de minste zorg vraagt."),
        mo.md(r"""
        De figuur: de training loss van het netwerk van 30 blocks uit de gok tijdens zijn ene epoch, het gemiddelde van de laatste 10 mini-batches, met post-norm en met pre-norm. In de eerste 20 stappen stijgen ze allebei tot boven 3 en dalen ze weer. Daarna blijft de loss met post-norm op 2,30. Met pre-norm blijft hij dalen, tot ongeveer 0,7.
        """),
        mo.Html(C.loss_picture({"title": "30 blocks, learning rate 0,3, één epoch", "x": "mini-batch", "y": "training loss",
                                "post": "post-norm", "pre": "pre-norm",
                                "alt": "Twee loss-curves over 94 mini-batches. Allebei stijgen ze in de eerste 20 stappen van ongeveer 2,6 tot boven 3 en dalen ze weer. Met post-norm, gestippeld, blijft de loss daarna op ongeveer 2,3. Met pre-norm daalt hij tot ongeveer 0,7."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Layer normalization van een batch, één rij per voorbeeld en één kolom per feature:

    1. Per rij: het gemiddelde en de variantie over zijn features, gedeeld door het aantal features, en als kolom gehouden met `keepdim=True`.
    2. Standaardiseer: min het gemiddelde, gedeeld door de wortel uit de variantie plus $\varepsilon$.
    3. Herschaal: keer $\gamma$, plus $\beta$, één van elk per feature.
    4. Gebruik in een block met een residual connection pre-norm: $x + f(\mathrm{LN}(x))$, en nog één LayerNorm vóór de output-layer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De batch uit [Batch normalization](/course/open/nn-batchnorm?lang=nl), omgedraaid: nu 2 voorbeelden met elk 4 features, het eerste tussen 1 en 6, het tweede tussen 40 en 70. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_layernorm():
    import torch
    from torch import nn

    _X = torch.tensor([[1.0, 2.0, 3.0, 6.0], [50.0, 70.0, 40.0, 60.0]])
    # 1. per rij: het gemiddelde en de variantie over zijn features, als kolom gehouden
    _mean = _X.mean(dim=1, keepdim=True)
    _var = ((_X - _mean) ** 2).mean(dim=1, keepdim=True)
    print(_mean, _var)
    # 2. en 3. standaardiseer; gamma en beta beginnen op 1 en 0, dus herschalen verandert nog niets
    print((_X - _mean) / torch.sqrt(_var + 1e-5))
    _ln = nn.LayerNorm(4)
    with torch.no_grad():
        print(_ln(_X))
        print(_ln(_X[:1]))
        # 4. een pre-norm block, met f één layer: _X plus f van de genormaliseerde _X
        torch.manual_seed(0)
        _f = nn.Linear(4, 4)
        # daarna: print _X plus _f van _ln(_X), en voer opnieuw uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De gemiddelden zijn 3 en 55, de varianties 3,5 en 125: de getallen uit Batch normalization, omdat de batch dezelfde is, omgedraaid. Elke rij heeft nu gemiddelde 0 en standaardafwijking 1: −1,07, −0,53, 0 en 1,60 voor het eerste voorbeeld, −0,45, 1,34, −1,34 en 0,45 voor het tweede. `nn.LayerNorm(4)` geeft hetzelfde. Krijgt het alleen het eerste voorbeeld, dan geeft het precies dezelfde eerste rij: het tweede voorbeeld heeft geen invloed. Met BatchNorm was een batch van één gestandaardiseerd tot allemaal nullen. In stap 4 telt $f$ hooguit ongeveer 1 bij elk getal op: de eerste rij wordt 0,47, 3,07, 2,48 en 6,50, en de tweede blijft tussen 40 en 71. De directe weg draagt `_X` zelf; alleen $f$ krijgt de genormaliseerde rijen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een batch van één
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "batch-of-one",
        question="Een model traint met één lange tekst per stap: een batch van één. Welke normalisatie standaardiseert in train mode dan nog zinvol zijn features?",
        options={
            "batch": "BatchNorm: zijn running averages vullen de ontbrekende batch aan",
            "right": "LayerNorm: het standaardiseert elk voorbeeld over zijn eigen features",
            "both": "Allebei: een batch van één is gewoon een kleine batch",
        },
        correct="right",
        why={
            "batch": "In train mode gebruikt BatchNorm de batch zelf, niet zijn running averages. Voor één voorbeeld is elke $z$ min zijn eigen gemiddelde 0: alles wordt $\\beta$.",
            "both": "Voor BatchNorm is het gemiddelde van één getal dat getal zelf, dus elke $z$ wordt 0 vóór het herschalen; PyTorch weigert in train mode zelfs een batch van één. LayerNorm kijkt nooit naar andere voorbeelden.",
            "right": "LayerNorm neemt het gemiddelde en de variantie over de features van elk voorbeeld, dus de grootte van de batch doet er niet toe, in train mode en in eval mode.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_exercise_ln():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Layer normalization met de hand

        Schrijf `layer_norm(X, gamma, beta)`: layer normalization, volgens stap 1 tot en met 3 van het recept, met $\varepsilon$ = 0,00001. `X` heeft één rij per voorbeeld en één kolom per feature; `gamma` en `beta` hebben één getal per feature. De check roept `layer_norm(torch.tensor([[1.0, 2.0, 6.0], [10.0, 20.0, 30.0]]), torch.tensor([1.0, 2.0, 0.5]), torch.tensor([0.0, 1.0, -1.0]))` aan, en dat moet `[[-0.9258, 0.0742, -0.3056], [-1.2247, 1.0, -0.3876]]` geven; daarna probeert hij andere batches, ook een batch van één voorbeeld.
        """),
        L.task("Laat `layer_norm` elke rij standaardiseren over zijn eigen features, en dan keer gamma, plus beta."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ln():
    box_ln = L.code_box("layer-norm", lines=6)
    box_ln
    return (box_ln,)


@app.cell(hide_code=True)
def check_ln(box_ln):
    result_ln = L.check("layer-norm", box_ln.value)
    result_ln
    return (result_ln,)


@app.cell(hide_code=True)
def hints_ln():
    L.hints(
        "layer-norm",
        [
            "Zoals `batch_norm` in [Batch normalization](/course/open/nn-batchnorm?lang=nl), maar over elke rij in plaats van elke kolom: `dim=1`.",
            "Met `dim=1` is het gemiddelde één getal per rij; `keepdim=True` houdt het als kolom, zodat `X - mean` het eigen gemiddelde van elke rij van die rij aftrekt.",
        ],
        why="`mean = X.mean(dim=1, keepdim=True)`, `var = ((X - mean) ** 2).mean(dim=1, keepdim=True)`, en `return gamma * (X - mean) / torch.sqrt(var + 1e-5) + beta`. In het voorbeeld heeft de eerste rij gemiddelde 3 en variantie $14/3 \\approx 4{,}667$, dus zijn 1 wordt $(1 - 3)/\\sqrt{4{,}667} \\approx -0{,}926$, keer 1 plus 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een pre-norm block

        Schrijf een class `PreNormBlock(nn.Module)`: je `ResidualBlock` uit [Residual connections](/course/open/nn-residual?lang=nl) met pre-norm. `PreNormBlock(d, hidden)` heeft `self.norm = nn.LayerNorm(d)`, `self.lin1 = nn.Linear(d, hidden)` en `self.lin2 = nn.Linear(hidden, d)`, met `lin1` gemaakt vóór `lin2`, zodat ze dezelfde willekeurige weights krijgen als in de oplossing van de course. Zijn `forward(x)` geeft `x + f(norm(x))`, met `f` eerst `lin1`, dan ReLU, dan `lin2`. De check maakt `PreNormBlock(4, 8)` na `torch.manual_seed(0)` en vergelijkt wat het geeft voor 3 willekeurige voorbeelden; daarna probeert hij andere groottes, een block dat twee keer na elkaar wordt toegepast, en één voorbeeld alleen.
        """),
        L.task("Laat `PreNormBlock` zijn input plus f van zijn genormaliseerde input geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("pre-norm-block", lines=13)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("pre-norm-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "pre-norm-block",
        [
            "`__init__` is die van je `ResidualBlock`, met `self.norm = nn.LayerNorm(d)` erbij: `d` features, de grootte van `x`.",
            "In `forward` krijgt alleen `f` de genormaliseerde input: eerst `self.norm`, dan `lin1`, ReLU en `lin2`. De `x` die erbij wordt opgeteld, blijft zoals hij is.",
        ],
        why="`__init__`: `super().__init__()`, `self.norm = nn.LayerNorm(d)`, `self.lin1 = nn.Linear(d, hidden)` en `self.lin2 = nn.Linear(hidden, d)`. `forward`: `return x + self.lin2(torch.relu(self.lin1(self.norm(x))))`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De formules uit [Batch normalization](/course/open/nn-batchnorm?lang=nl), genomen over de features van elk voorbeeld in plaats van over de batch. De directe weg uit [Residual connections](/course/open/nn-residual?lang=nl) blijft met pre-norm schoon, zodat de gradient ongewijzigd blijft terugkomen. Je `PreNormBlock` is de helft van een transformer block: daar heeft één zo'n block attention als $f$, en het volgende een MLP. In de volgende les, *Momentum*, krijgt gradient descent zelf een geheugen van zijn eerdere stappen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, result_ln, result_block):
    L.wrap_up(choice_one, result_ln, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
