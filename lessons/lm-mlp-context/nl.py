# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-mlp-context. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-mlp-context", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Embedding layers](/course/open/nn-embedding-layer?lang=nl) en [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`nn.Embedding(65, 10)` krijgt een tensor met ids van shape (32, 3). Welke shape komt eruit?",
        options={
            "flat": "(32, 30): de 3 embeddings van een rij naast elkaar",
            "right": "(32, 3, 10): een rij van 10 getallen voor elk id",
            "same": "(32, 3): één getal voor elk id",
        },
        correct="right",
        why={
            "flat": "Naast elkaar zetten is een eigen stap, die je vandaag zet; de layer zelf voegt aan het eind een nieuwe as van 10 toe. Zie [Embedding layers](/course/open/nn-embedding-layer?lang=nl).",
            "same": "Elk id krijgt een hele rij van de embedding matrix, 10 getallen. Zie [Embedding layers](/course/open/nn-embedding-layer?lang=nl).",
            "right": "De shape van de ids blijft, en er komt een as van 10 achteraan. Vandaag worden de 3 embeddings van elke rij naast elkaar geplakt, tot 30 getallen.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_bigram():
    quick_bigram = L.choice(
        "quick-bigram",
        question="Hoeveel tekens ervoor gebruikt het bigram-model uit het bigram-project om het volgende teken te voorspellen?",
        options={
            "two": "Twee: het teken ervoor en het teken daarvoor",
            "all": "Allemaal, vanaf het begin van de tekst",
            "right": "Eén: alleen het laatste teken",
        },
        correct="right",
        why={
            "two": "Een bigram is een paar: het laatste teken en het volgende. Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
            "all": "Zijn enige input is het laatste teken, dat één rij van de tabel kiest. Zie [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl).",
            "right": "Na \"th\", \"ch\" en \"sh\" ziet het alleen \"h\", dus het geeft alle drie dezelfde kansen voor het volgende teken. Vandaag kijkt het model verder terug.",
        },
    )
    quick_bigram
    return (quick_bigram,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het bigram-model ziet alleen het laatste teken. Na "th" ziet het "h", en een "h" na "c" of "s" wordt door andere tekens gevolgd dan een "h" na "t". Een model dat de laatste paar tokens ziet, kan het veel beter doen, maar tellen schaalt niet: met 3 tekens context heeft een tabel een rij nodig voor elk van de $65^3 = 274.625$ combinaties, waarvan er maar 11.556 ooit voorkomen in Tiny Shakespeare. In 2003 gaven Yoshua Bengio en zijn collega's elk token een embedding, zetten ze de embeddings van de laatste paar tokens naast elkaar, en lieten ze een MLP de kansen op het volgende token geven: een **neural language model**. Tokens die zich vergelijkbaar gedragen, krijgen embeddings die dicht bij elkaar liggen, dus wat het model over de ene combinatie leert, geldt ook voor vergelijkbare.
    """)
    return


@app.cell(hide_code=True)
def predict_window():
    guess_window = L.guess(
        "three-or-eight",
        question="Het model van deze les traint 1.000 stappen van 128 voorbeelden op Tiny Shakespeare: één keer kijkt het 3 tekens terug, één keer 8. Welke eindigt met de laagste loss op tekst waarop het niet getraind heeft?",
        options={
            "eight": "8, duidelijk: het ziet meer van de tekst",
            "right": "Ongeveer gelijk: met 3 tekens zelfs iets lager",
            "three": "3, duidelijk: 8 tekens zijn te veel om te leren",
        },
        reveal="**Ongeveer gelijk, met 3 iets lager.** Over 4 seeds eindigt het model met 3 tekens op 2,20 tot 2,22, en met 8 op 2,22 tot 2,24. In het MLP heeft elke plek van het venster zijn eigen weights, dus het moet los leren wat een \"e\" twee plekken terug betekent en wat hij vijf plekken terug betekent: meer context is meer om te leren. Met 8 keer zoveel training, 8.000 stappen, wint 8 net: 1,94 tot 1,95 tegen 1,96 tot 1,99.",
    )
    guess_window
    return (guess_window,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Het model kijkt een vast aantal tokens terug, $k$: zijn **context window**. Voor het volgende token op plek $t$ neemt het de ids van de $k$ tokens ervoor, zoekt hun embeddings op, elk $d$ getallen, en plakt die naast elkaar tot één vector van $k \cdot d$ getallen. Daarna komt een hidden layer met ReLU, en dan de **output head**: een lineaire layer van de hidden layer naar één logit voor elk token van de vocabulary, 65 voor Tiny Shakespeare. Net als in [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl) geeft de softmax van de logits de kansen, en geeft cross-entropy met het token dat echt volgt de loss. Bengio en zijn collega's gebruikten tanh, de S-curve uit [Initialisatie](/course/open/nn-initialization?lang=nl), waar deze les ReLU gebruikt; met tanh is het verhaal van het plaatje hieronder hetzelfde, met een iets hogere loss.
        """),
        L.formula(r"[[hidden:\mathbf{x}]] = \big[\, [[hidden:\mathbf{e}_{t-k}]] ; \dots ; [[hidden:\mathbf{e}_{t-1}]] \,\big]"),
        L.formula(r"[[pred:\mathbf{z}]] = [[param:\mathbf{W}_2]] \, \mathrm{ReLU}([[param:\mathbf{W}_1]] [[hidden:\mathbf{x}]] + [[param:\mathbf{b}_1]]) + [[param:\mathbf{b}_2]]"),
        mo.md(r"""
        In PyTorch maakt `nn.Embedding` van ids met shape (B, k) embeddings met shape (B, k, d), en `e.reshape(len(x), -1)` plakt de $k$ embeddings van elke rij tot één rij van $k \cdot d$ getallen. Het venster ligt vast: het model kan niets zien van wat ervoor komt, hoeveel tekst er ook is. En $\mathbf{W}_1$ heeft $k \cdot d$ kolommen, een blok van $d$ voor elke plek in het venster, dus een langer venster betekent meer weights, en elke plek wordt los geleerd.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **context window** | | het vaste aantal tokens dat een model terugkijkt | `k` |
        | **neural language model** | | een neural network dat de kansen op het volgende token geeft | `MLPLanguageModel(65, 3, 10, 128)` |
        | **output head** | | de laatste layer: van de laatste vector van het netwerk naar één logit voor elk token van de vocabulary | `nn.Linear(hidden, vocab)` |
        | $\mathbf{x}$ | x | de $k$ embeddings van het venster naast elkaar, $k \cdot d$ getallen | `e.reshape(len(x), -1)` |
        | $\mathbf{z}$ | z | de logits, één voor elk token van de vocabulary | `model(x)` |
        """),
        L.watch_out("Plak de embeddings naast elkaar; tel ze niet op. Opgeteld geven \"ab\" en \"ba\" dezelfde vector, en weet het model niet meer welk token waar stond. En de eerste $k$ tokens van een tekst hebben minder dan $k$ tokens voor zich: deze les maakt voor hen gewoon geen voorbeelden."),
        mo.md(r"""
        De figuur: de loss op tekst waarop het model niet getraind heeft, het laatste tiende van Tiny Shakespeare, na dezelfde training met de Trainer, voor 1, 3 en 8 tekens context, in één run, met seed 0. Van 1 naar 3 daalt de loss veel; van 3 naar 8 daalt hij niet meer: met zoveel training kan het model de extra tekens nog niet benutten.
        """),
        mo.Html(C.context_picture({"title": "Loss op onbekende tekst", "x": "tekens context, k", "y": "validation loss", "number": L.number,
                                   "alt": "De validation loss na 1.000 stappen training, tegen het aantal tekens context: 2,51 met 1, 2,20 met 3, en 2,22 met 8."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een neural language model met een context window van $k$ tokens:

    1. Maak de voorbeelden: voor elke plek $t$ vanaf $k$ de $k$ ids ervoor, en het id op $t$ als label.
    2. Zoek de embeddings op: ids met shape (B, k) geven shape (B, k, d).
    3. Plak de $k$ embeddings van elk voorbeeld naast elkaar: shape (B, k · d).
    4. Een hidden layer met ReLU, dan de output head: één logit voor elk token van de vocabulary.
    5. Train met cross-entropy tegen het label: je Trainer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 en 3 voor twee vensters van 3 tekens, "the" en "he ", met een embedding layer van 10 getallen per teken. `C.encode` geeft de ids van de tekens van een tekst. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_glue():
    import torch
    from torch import nn

    torch.manual_seed(0)
    _emb = nn.Embedding(65, 10)
    _x = torch.tensor([C.encode("the"), C.encode("he ")])
    print(_x)
    # 2. opzoeken: een rij van 10 getallen voor elk id
    _e = _emb(_x)
    print(_e.shape)
    # 3. plak de 3 embeddings van elk venster naast elkaar
    _glued = _e  # maak af: _e.reshape(len(_x), -1)
    print(_glued.shape)
    # getallen 10 tot en met 19 van de eerste rij: de embedding van zijn tweede teken, "h"
    print(_glued[0, 10:20].tolist() == _e[0, 1].tolist())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    "the" heeft de ids 58, 46 en 43, en "he " 46, 43 en 1: de spatie heeft id 1. Het opzoeken geeft shape (2, 3, 10). Zoals hij is, heeft `_glued` nog die shape, en vraagt `_glued[0, 10:20]` om de plekken 10 tot en met 19 van een as van 3: niets, dus de vergelijking print `False`. Afgemaakt heeft `_glued` shape (2, 30), en zijn de getallen 10 tot en met 19 van de eerste rij de embedding van "h": `True`. De embedding van "h" staat in het midden van de eerste rij en aan het begin van de tweede: de plek in het venster is de plek in de rij.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_window():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat het venster ziet
    """)
    return


@app.cell(hide_code=True)
def choice_window_cell():
    choice_window = L.choice(
        "window",
        question="Een model met een context window van 3 tekens voorspelt de laatste \"e\" van \"to be or not to be\". Welke tekens kan het gebruiken?",
        options={
            "all": "Heel \"to be or not to b\": het venster zegt alleen welke het zwaarst tellen",
            "right": "Alleen \"o\", de spatie en \"b\": de 3 vlak ervoor",
            "first": "De eerste 3, \"to \": het venster begint bij het begin",
        },
        correct="right",
        why={
            "all": "Het model krijgt alleen de k ids van het venster als input; alles daarvoor is er gewoon niet.",
            "first": "Het venster schuift mee: voor elke plek bevat het de k tokens vlak voor die plek.",
            "right": "Het venster bevat de 3 tokens vlak voor de plek. Dat dezelfde \"to b\" eerder in de tekst al voorkwam, kan het model niet zien: het venster ligt vast.",
        },
    )
    choice_window
    return (choice_window,)


@app.cell(hide_code=True)
def text_exercise_contexts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De voorbeelden van een venster

        Maak `contexts(ids, k)` af: die geeft de trainingsvoorbeelden van een lijst ids voor een context window van `k`, als twee lijsten: `X`, de vensters van `k` ids, en `Y`, het id dat op elk venster volgt. Vervang de drie `...`, volgens stap 1 van het recept. De check roept `contexts([5, 1, 4, 2, 7], 2)` aan, wat `([[5, 1], [1, 4], [4, 2]], [4, 2, 7])` moet geven, en probeert andere vensters en lijsten. Als hij goed is, slaat de check hem op in je project, als `work/projects/llm/contexts.py`.
        """),
        L.task("Laat `contexts` de vensters van `k` ids geven en het id dat op elk venster volgt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contexts():
    box_contexts = L.code_box("contexts", lines=8)
    box_contexts
    return (box_contexts,)


@app.cell(hide_code=True)
def check_contexts(box_contexts):
    result_contexts = L.check("contexts", box_contexts.value)
    result_contexts
    return (result_contexts,)


@app.cell(hide_code=True)
def hints_contexts():
    L.hints(
        "contexts",
        [
            "De eerste plek met `k` ids ervoor is `k`, en de laatste plek is `len(ids) - 1`, die ook een label is.",
            "Het venster vóór plek `t` is een slice van `ids` die net vóór `t` stopt en `k` lang is.",
        ],
        why="Voor `contexts([5, 1, 4, 2, 7], 2)` loopt `t` van 2 tot en met 4. Bij `t` = 2 is het venster `ids[0:2]`, `[5, 1]`, en het label `ids[2]`, 4; bij `t` = 4 is het venster `[4, 2]` en het label 7: een lijst van 5 ids geeft 5 − 2 = 3 voorbeelden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het model

        Maak de class `MLPLanguageModel` af: `MLPLanguageModel(vocab, k, dim, hidden)` is het model van het recept, voor een vocabulary van `vocab` tokens en een context window van `k`. Vervang de twee `...`:

        - In `__init__` drie layers, gemaakt in deze volgorde, zodat ze dezelfde willekeurige startwaarden krijgen als in de oplossing van de course: `self.emb`, een `nn.Embedding` met `dim` getallen per token; `self.hidden`, een `nn.Linear` van de `k` embeddings naast elkaar naar `hidden` neuronen; en `self.out`, de output head.
        - In `forward(x)`, met `x` van shape (B, k): stap 2 tot en met 4 van het recept.

        De check maakt kleine modellen na `torch.manual_seed(0)` en vergelijkt hun logits, en of de volgorde van de tokens in een venster uitmaakt. Als hij goed is, slaat de check hem op als `work/projects/llm/mlp_lm.py`.
        """),
        L.task("Laat `MLPLanguageModel` één logit voor elk token geven, uit de laatste `k` tokens naast elkaar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_model():
    box_model = L.code_box("mlp-lm", lines=16)
    box_model
    return (box_model,)


@app.cell(hide_code=True)
def check_model(box_model):
    result_model = L.check("mlp-lm", box_model.value)
    result_model
    return (result_model,)


@app.cell(hide_code=True)
def hints_model():
    L.hints(
        "mlp-lm",
        [
            "De input van de hidden layer is de geplakte rij uit het uitgewerkte voorbeeld: `k * dim` getallen. De output head gaat van `hidden` naar `vocab`.",
            "In `forward`: `self.emb(x)`, dan de reshape uit het afgemaakte uitgewerkte voorbeeld met `x` in plaats van `_x`, dan `torch.relu` van de hidden layer, dan `self.out`.",
        ],
        why="`self.emb(x)` geeft (B, k, dim), de reshape (B, k * dim), de hidden layer (B, hidden), en de output head (B, vocab): één logit voor elk token van de vocabulary, voor elk voorbeeld. De ReLU ertussen voorkomt dat de twee lineaire layers samen één worden.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het bigram-model uit [Project: hetzelfde bigram-model, geleerd met gradient descent](/course/open/lm-bigram-neural?lang=nl) zocht één rij op voor het laatste teken; dit model zoekt de embeddings uit [Embedding layers](/course/open/nn-embedding-layer?lang=nl) op voor de laatste $k$, en een MLP maakt de logits. Het traint met je Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl), en je llm-project heeft nu `contexts.py` en `mlp_lm.py`. Zijn venster ligt vast, en het leert elke plek van het venster los: recurrent neural networks en daarna attention, later in de course, gebruiken dezelfde weights op elke plek, en kijken veel verder terug. In de volgende les, *Perplexity*, maak je van de loss een getal dat je je kunt voorstellen: tussen hoeveel tokens het model kiest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_window, result_contexts, result_model):
    L.wrap_up(choice_window, result_contexts, result_model)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
