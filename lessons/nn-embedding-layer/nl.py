# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-embedding-layer. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-embedding-layer", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl) en [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="`W` is de tabel van 65 × 65 uit het bigram-project, en `x = torch.tensor([3, 3, 10])`. Wat is `W[x]`?",
        options={
            "one": "Eén rij: rij 3, het eerste id van de tensor; de rest valt weg",
            "right": "Drie rijen: rij 3, nog eens rij 3 en rij 10, shape (3, 65)",
            "sum": "Eén rij: de som van rij 3, 3 en 10",
        },
        correct="right",
        why={
            "one": "Indexeren met een tensor van ids geeft één rij voor elke id erin. Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
            "sum": "Er wordt niets opgeteld: elke id geeft zijn eigen rij. Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
            "right": "Elke id geeft zijn eigen rij, dus een id die twee keer voorkomt, geeft twee keer dezelfde rij. Vandaag krijgt zo'n tabel minder kolommen en een eigen layer.",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_params():
    quick_params = L.choice(
        "quick-params",
        question="Hoeveel parameters heeft `nn.Linear(8, 65)`?",
        options={
            "weights": "520: één weight voor elk paar van een input en een neuron",
            "plus": "73: 8 inputs en 65 neuronen",
            "right": "585: 8 weights voor elk van de 65 neuronen, plus 65 bias terms",
        },
        correct="right",
        why={
            "weights": "Elk van de 65 neuronen heeft ook een bias term. Zie [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl).",
            "plus": "Elk neuron heeft een weight voor elke input, niet één in totaal. Zie [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl).",
            "right": "`weight` heeft shape (65, 8), 520 getallen, en `bias` nog 65. In oefening 3 komt zo'n layer na een embedding layer.",
        },
    )
    quick_params
    return (quick_params,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In het bigram-project had elk teken een eigen rij van 65 getallen, en elke rij leerde op zichzelf, ook voor `!` en `?`, waar bijna dezelfde tekens op volgen. Dat schaalt niet: een model dat naar de 2 tekens ervoor kijkt, heeft een rij nodig voor elk paar, 4.225 rijen; voor 3 tekens 274.625. Een **embedding layer** geeft elk teken in plaats daarvan een korte rij getallen, en laat de rest van het netwerk daarmee rekenen. Tekens die zich hetzelfde gedragen, kunnen dan rijen krijgen die op elkaar lijken, en delen wat ze leren.
    """)
    return


@app.cell(hide_code=True)
def predict_dims():
    guess_dims = L.guess(
        "how-many-numbers",
        question="Het bigram-model krijgt nu een embedding layer met een paar getallen per teken, en dan `nn.Linear` van die getallen naar 65 logits. Getraind op alle paren van Tiny Shakespeare haalt de tabel van 65 × 65 uit het bigram-project een loss van 2,4526. Wat is van deze drie het kleinste aantal getallen per teken waarmee de embedding layer daar tot op 0,01 bij komt?",
        options={
            "two": "2: een klinker of niet, en een hoofdletter of niet",
            "right": "16: een kwart van de 65",
            "all": "65: één voor elk mogelijk volgend teken, zoals in de tabel",
        },
        reveal="**16.** Met 16 getallen per teken is de loss 2,4543; met 8 is hij 2,4839, met 2 nog 2,7341. 16 getallen per teken geven samen met de linear layer 2.145 parameters, de helft van de 4.225 van de tabel. In zo'n korte rij getallen komen tekens die zich hetzelfde gedragen dicht bij elkaar: in deze run zijn de vier die volgens de cosine similarity het dichtst bij `.` liggen `?`, `;`, `:` en `!`. Met een andere seed zijn het andere vier, maar bijna altijd leestekens.",
    )
    guess_dims
    return (guess_dims,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **embedding layer** is een tabel die het netwerk leert: de **embedding matrix** $E$, één rij van $d$ getallen voor elk van de $\lvert\mathcal{V}\rvert$ ids, zoals de 65 tekens. Voor id $x$ geeft hij rij $x$, zijn embedding. Dat geeft een one-hot-vector keer de tabel ook, zoals in het bigram-project, maar dan zonder te vermenigvuldigen met 64 nullen:
        """),
        L.formula(r"[[hidden:\mathbf{e}]] = [[param:E]][x] = \mathrm{onehot}(x) \, [[param:E]]"),
        mo.md(r"""
        Op de terugweg krijgen alleen de rijen die werden opgezocht een gradient, en een rij die op meerdere plekken werd opgezocht, krijgt al hun gradients opgeteld. Voor rij $v$ van $E$ loopt de som over de plekken $k$ waar de id $x_k$ gelijk is aan $v$:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial E_v}]] = \sum_{k \,:\, x_k = v} [[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{e}_k}]]"),
        mo.md(r"""
        Met gradient descent blijven de rijen van ids die niet voorkwamen in die stap zoals ze zijn.

        In PyTorch is de layer `nn.Embedding(vocab, d)`. Zijn `weight` is $E$, met shape $(\lvert\mathcal{V}\rvert, d)$, aan het begin met willekeurige getallen. Hij neemt een tensor van ids van elke shape, hele getallen, en geeft één rij van $d$ getallen per id: ids met shape (3,) geven shape (3, $d$). Het bigram-model met een embedding layer is dan `nn.Linear(d, vocab)` na `nn.Embedding(vocab, d)`: van een id naar zijn embedding, en van de embedding naar $\lvert\mathcal{V}\rvert$ logits voor het volgende teken.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **embedding layer** | | een layer die voor elke id een rij van een geleerde tabel opzoekt | `nn.Embedding(vocab, d)` |
        | **embedding matrix** | | de tabel van die layer: één rij van $d$ getallen per id | `emb.weight` |
        | $\lvert\mathcal{V}\rvert$, $d$ | de grootte van de vocabulary, d | het aantal ids, en het aantal getallen in een embedding | `num_embeddings`, `embedding_dim` |
        """),
        L.watch_out("`nn.Embedding` neemt ids, hele getallen van 0 tot en met $\\lvert\\mathcal{V}\\rvert - 1$, geen one-hot-vectoren of kommagetallen: bij een id van $\\lvert\\mathcal{V}\\rvert$ of meer stopt Python met een fout. En de embeddings beginnen als willekeurige getallen; hun betekenis krijgen ze pas door te trainen."),
        mo.md(r"""
        De figuur: het bigram-model met een embedding layer van 1 tot 32 getallen per teken, getraind op alle paren tekens van Tiny Shakespeare, en zijn loss na het trainen. De gestippelde lijn is de tabel van 65 × 65 uit het bigram-project: het beste wat een bigram-model kan. Vanaf 16 getallen is de embedding layer bijna zo goed als de tabel.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_dims():
    mo.Html(C.dims_picture({"title": "Loss per grootte van de embedding", "x": "getallen per teken", "y": "loss na het trainen", "table": "tabel van 65 × 65",
                            "alt": "De loss na het trainen tegen het aantal getallen per teken, 1 tot 32, op een logaritmische schaal: 2,92 met 1, 2,73 met 2, 2,59 met 4, 2,48 met 8, en 2,45 met 16 en 32, op de gestippelde lijn van de tabel van 65 bij 65."}))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een embedding layer:

    1. Nummer de dingen: elk teken, woord of token krijgt een id van 0 tot en met $\lvert\mathcal{V}\rvert - 1$.
    2. Maak de tabel: $\lvert\mathcal{V}\rvert$ rijen van $d$ getallen, eerst willekeurig; in PyTorch `nn.Embedding(vocab, d)`.
    3. Zoek op: voor ids `x` de rijen `E[x]`, één per id.
    4. Train: op de terugweg krijgt elke rij de gradients van alle plekken waar zijn id voorkwam, opgeteld; daarna zet de optimizer zijn stap.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een tabel van 4 ids met elk 2 getallen, in NumPy, en de ids 2, 0 en 2. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_lookup():
    import numpy as np

    # 2. de tabel: 4 ids, elk 2 getallen
    _E = np.array([[0.1, 0.2], [0.5, -0.3], [-0.4, 0.8], [0.0, 1.0]])
    _ids = np.array([2, 0, 2])
    # 3. zoek op: één rij per id, hetzelfde als one-hot-rijen keer de tabel
    print(_E[_ids])
    print(np.eye(4)[_ids] @ _E)
    # 4. de gradient van de tabel, uit de gradients van de drie opgezochte rijen
    _grad_out = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    _grad_E = np.zeros(_E.shape)
    for _k in range(len(_ids)):
        # maak af: tel bij de rij op, in plaats van hem te overschrijven
        _grad_E[_ids[_k]] = _grad_out[_k]
    print(_grad_E)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_E[_ids]` geeft rij 2, 0 en 2: $(-0{,}4;\ 0{,}8)$, $(0{,}1;\ 0{,}2)$ en nog eens $(-0{,}4;\ 0{,}8)$, en de one-hot-rijen keer de tabel geven hetzelfde. Rij 1 en 3 werden niet opgezocht, dus hun gradient is 0. Id 2 kwam twee keer voor, met de gradients $(1;\ 0)$ en $(2;\ 2)$. Zoals hij is, overschrijft de loop rij 2, en blijft alleen de laatste over: $(2;\ 2)$. Afgemaakt, met `_grad_E[_ids[_k]] = _grad_E[_ids[_k]] + _grad_out[_k]`, krijgt rij 2 ze allebei: $(3;\ 2)$.
    """)
    return


@app.cell(hide_code=True)
def text_torch_step():
    mo.md(r"""
    ### Een trede hoger: hetzelfde in PyTorch

    `nn.Embedding` doet dit allemaal zelf, het optellen ook. Hier met een eigen willekeurige tabel: `.backward()` van de som van elke opgezochte rij keer zijn gradient geeft elke rij van de tabel zijn gradients, opgeteld, zoals in de loop.
    """)
    return


@app.cell
def example_torch():
    import torch
    from torch import nn

    torch.manual_seed(0)
    _emb = nn.Embedding(4, 2)
    _rows = _emb(torch.tensor([2, 0, 2]))
    print(_emb.weight.shape, _rows.shape)
    (_rows * torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])).sum().backward()
    print(_emb.weight.grad)
    return


@app.cell(hide_code=True)
def text_after_torch():
    mo.md(r"""
    De tabel heeft shape (4, 2), en de drie opgezochte rijen shape (3, 2). De gradient van de tabel is die van het afgemaakte NumPy-voorbeeld: $(0;\ 1)$ voor rij 0, $(3;\ 2)$ voor rij 2, en 0 voor rij 1 en 3.

    | NumPy | PyTorch |
    |---|---|
    | `E = np.array(...)`, shape (4, 2) | `emb = nn.Embedding(4, 2)`, `emb.weight` |
    | `E[ids]` | `emb(ids)` |
    | de loop die `grad_out[k]` optelt bij rij `ids[k]` | `.backward()`, in `emb.weight.grad` |
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lookup():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom opzoeken?
    """)
    return


@app.cell(hide_code=True)
def choice_lookup_cell():
    choice_lookup = L.choice(
        "why-lookup",
        question="Een one-hot-vector keer de embedding matrix geeft dezelfde rij als opzoeken. Waarom zoekt `nn.Embedding` rijen op?",
        options={
            "right": "Het geeft dezelfde rij zonder het werk: niet vermenigvuldigen met al die nullen",
            "better": "Opzoeken geeft een betere rij: het vermenigvuldigen voegt afrondingsfouten toe",
            "grad": "One-hot-vectoren hebben geen gradient, dus de tabel zou niet kunnen leren",
        },
        correct="right",
        why={
            "better": "Vermenigvuldigen met nullen en één 1 geeft precies de rij, zonder afrondingsfouten: beide manieren geven dezelfde getallen.",
            "grad": "De tabel krijgt op beide manieren dezelfde gradient; de one-hot-vectoren zelf hebben er geen nodig.",
            "right": "Met 50.000 verschillende tokens heeft een one-hot-vector 49.999 nullen, en daarmee vermenigvuldigen is werk voor niets. Opzoeken geeft dezelfde rij, voor een fractie van de moeite.",
        },
    )
    choice_lookup
    return (choice_lookup,)


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén stap op de tabel

        Schrijf `sgd_step(E, ids, grad_out, lr)`: één stap gradient descent op de embedding-tabel `E`, shape $(\lvert\mathcal{V}\rvert, d)$, nadat de rijen van `ids` zijn opgezocht en de gradients `grad_out` kregen, één rij per id. Maak eerst de gradient van de tabel, volgens stap 4 van het recept; geef dan `E - lr * grad_E` terug. De check roept `sgd_step(np.array([[0, 0], [1, 1], [2, 2]]), np.array([0, 0]), np.array([[1, 0], [0, 1]]), 0.5)` aan, wat `np.array([[-0.5, -0.5], [1, 1], [2, 2]])` moet geven, en probeert andere tabellen en ids.
        """),
        L.task("Laat `sgd_step` één stap gradient descent zetten op een embedding-tabel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("sgd-step", lines=10)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("sgd-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "sgd-step",
        [
            "De loop van het uitgewerkte voorbeeld, afgemaakt: voor elke `k` krijgt rij `ids[k]` van `grad_E` er `grad_out[k]` bij.",
            "Dan de stap zelf: de tabel min de learning rate keer zijn gradient. Rijen die niet werden opgezocht, hebben gradient 0, dus die blijven.",
        ],
        why="`for k in range(len(ids)): grad_E[ids[k]] = grad_E[ids[k]] + grad_out[k]`, dan `return E - lr * grad_E`. In het voorbeeld wordt rij 0 twee keer opgezocht en krijgt hij $(1;\\ 0) + (0;\\ 1) = (1;\\ 1)$; met een learning rate van 0,5 gaat hij van $(0;\\ 0)$ naar $(-0{,}5;\\ -0{,}5)$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het bigram-model met een embedding layer

        Schrijf een class `CharBigram(nn.Module)`: `CharBigram(vocab, dim)` maakt eerst `self.emb`, een `nn.Embedding` met `vocab` rijen van `dim` getallen, en dan `self.out`, een `nn.Linear` van `dim` naar `vocab`, in die volgorde, zodat ze dezelfde willekeurige startgetallen krijgen als het model van de check. `forward(x)` zoekt de ids in `x` op en geeft de rijen aan de linear layer: één rij van `vocab` logits per id. De check maakt `CharBigram(5, 2)` met een vaste seed en vergelijkt zijn logits voor de ids 0, 3 en 3; daarna maakt hij `CharBigram(65, 8)` en vergelijkt hij de shapes van zijn parameters en van zijn logits voor ids in een tabel van 2 bij 2.
        """),
        L.task("Laat `CharBigram` met een embedding layer de logits van het volgende teken geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_model():
    box_model = L.code_box("char-bigram", lines=12)
    box_model
    return (box_model,)


@app.cell(hide_code=True)
def check_model(box_model):
    result_model = L.check("char-bigram", box_model.value)
    result_model
    return (result_model,)


@app.cell(hide_code=True)
def hints_model():
    L.hints(
        "char-bigram",
        [
            "Beide layers zijn attributes die je in `__init__` maakt, na `super().__init__()`, zoals in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl): de embedding layer krijgt `vocab` en `dim`, de linear layer gaat van `dim` naar `vocab`.",
            "`forward` is één regel, van binnen naar buiten gelezen: eerst de embedding layer op `x`, dan de linear layer op wat eruit komt.",
        ],
        why="`__init__`: `self.emb = nn.Embedding(vocab, dim)` en dan `self.out = nn.Linear(dim, vocab)`. `forward`: `return self.out(self.emb(x))`. `CharBigram(65, 8)` heeft $65 \\cdot 8 = 520$ getallen in zijn tabel en 585 in zijn linear layer: 1.105 parameters.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De rijen `W[x]` uit [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl) waren al een opzoeking in een geleerde tabel; een embedding layer maakt die rijen kort en geeft ze een eigen layer, gebouwd zoals de layers uit [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl). Wat ze leren, is wat [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl) liet zien: dingen die zich hetzelfde gedragen, krijgen rijen die op elkaar lijken. In de volgende les, *Een herbruikbare training loop*, schrijf je de training loop één keer, zodat elk model hierna hem kan gebruiken. Later zet een language model dat naar meerdere tekens tegelijk kijkt hun embeddings naast elkaar, en begint een transformer met een embedding layer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lookup, result_step, result_model):
    L.wrap_up(choice_lookup, result_step, result_model)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
