# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-weight-tying. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-weight-tying", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) en [Embedding layers](/course/open/nn-embedding-layer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_logits():
    quick_logits = L.choice(
        "quick-logits",
        question="Waar komen de logits van je GPT vandaan?",
        options={
            "embedding": "Uit de embedding van de tokens, `self.tok`",
            "block": "Uit de attention van het laatste block",
            "right": "Uit `self.head`, de output head",
        },
        correct="right",
        why={
            "embedding": "`self.tok` staat aan het begin: hij maakt van ids vectoren. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "block": "Een block geeft de residual stream, d getallen per plek; die moet nog een score voor elk token worden. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "right": "`nn.Linear(d, vocab)` maakt van de d getallen van elke plek een logit voor elk token. Die head is de helft van deze les.",
        },
    )
    quick_logits
    return (quick_logits,)


@app.cell(hide_code=True)
def quick_check_lookup():
    quick_lookup = L.choice(
        "quick-lookup",
        question="`tok = nn.Embedding(vocab, d)`. Wat geeft `tok(torch.tensor([7]))`?",
        options={
            "column": "Kolom 7 van `tok.weight`",
            "right": "Rij 7 van `tok.weight`",
            "times": "Elke weight van `tok.weight` maal 7",
        },
        correct="right",
        why={
            "column": "De tabel heeft een rij van d getallen voor elk token, shape (vocab, d). Zie [Embedding layers](/course/open/nn-embedding-layer?lang=nl).",
            "times": "Een embedding layer zoekt een rij op; hij vermenigvuldigt niet met de id. Zie [Embedding layers](/course/open/nn-embedding-layer?lang=nl).",
            "right": "Eén rij per token, shape (vocab, d). De andere helft van deze les.",
        },
    )
    quick_lookup
    return (quick_lookup,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je GPT heeft twee tabellen met iets voor elk token van de vocabulary: de embedding aan het begin, die van een token een vector maakt, en de output head aan het eind, die van een vector een score voor elk token maakt. Met een vocabulary van 50.000 tokens zijn ze allebei groot. **Weight tying** gebruikt één tabel voor beide kanten. GPT-2 doet het, en veel kleine modellen ook.
    """)
    return


@app.cell(hide_code=True)
def predict_gpt2():
    guess_gpt2 = L.guess(
        "untied-gpt2",
        question="GPT-2 small, met $d = 768$ en een vocabulary van 50.257 tokens, heeft 124 miljoen parameters, met zijn head gekoppeld aan zijn embedding. Hoeveel zou het er hebben met een eigen head?",
        options={
            "same": "Ongeveer 125 miljoen",
            "right": "Ongeveer 163 miljoen",
            "double": "Ongeveer 250 miljoen",
        },
        reveal=f"**Ongeveer 163 miljoen:** de weight van de head heeft 50.257 × 768 = {C.grouped(C.VOCAB * 768, 'nl')} getallen, bijna een kwart van het ongekoppelde model. De twaalf blocks samen hebben er ongeveer 85 miljoen.",
    )
    guess_gpt2
    return (guess_gpt2,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Twee tabellen met dezelfde shape.** De embedding `nn.Embedding(vocab, d)` bewaart een tabel $\mathbf{E}$ met shape (vocab, d): rij $v$ is de vector van token $v$. De head `nn.Linear(d, vocab)` bewaart zijn weight met een rij voor elke output, zoals elke `nn.Linear` doet: ook shape (vocab, d). Zijn logit voor token $v$ is het inproduct van de laatste vector $\mathbf{h}$ met rij $v$, plus een bias term.

        **Eén tabel.** Weight tying neemt $\mathbf{E}$ voor allebei. Dan is de logit van token $v$ het inproduct van $\mathbf{h}$ met de eigen embedding $\mathbf{e}_v$ van het token:
        """),
        L.formula(r"[[pred:z_v]] = [[hidden:\mathbf{h}]] \cdot [[param:\mathbf{e}_v]] + [[param:b_v]]"),
        mo.md(r"""
        Voor alle tokens tegelijk: $\mathbf{h} \mathbf{E}^\top$. Het model geeft token $v$ een hoge logit als zijn laatste vector dezelfde kant op wijst als de embedding van $v$. In PyTorch doet één regel in `__init__` het: `self.head.weight = self.tok.weight`. De head en de embedding hebben nu dezelfde `nn.Parameter`; `model.parameters()` geeft hem één keer, en de optimizer verandert hem één keer per stap.

        **Twee gradients, opgeteld.** De tabel wordt twee keer gebruikt, zoals een convolutie zijn filter op elke plek gebruikt in [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl): weight sharing. Dus is zijn gradient de som van de gradient van het begin en die van het eind.

        **Waarom het kan helpen.** Zonder weight tying geeft een trainingsstap alleen een gradient aan de rijen van de embedding van de tokens in de batch. De head krijgt er een voor elke rij: softmax met cross-entropy geeft elke logit een gradient, zoals in [De gradient van softmax met cross-entropy](/course/open/ml-softmax-gradient?lang=nl). Gekoppeld leert de rij van een zeldzaam token ook bij elke stap. Press en Wolf (2017) vonden dat gekoppelde language models een lagere loss haalden op nieuwe tekst, met minder parameters.

        De figuur: het deel van de parameters dat weight tying bespaart, voor de vier maten van GPT-2. Hoe groter het model, hoe meer ervan in de blocks zit, en hoe minder weight tying bespaart.
        """),
        mo.Html(C.saved_picture({"title": "Wat weight tying bespaart in GPT-2", "x": "deel van de parameters zonder tying (%)",
                                 "row": "{name}, {n} miljoen zonder tying", "lang": L.lang,
                                 "alt": "Vier liggende balken. Weight tying bespaart 24% van GPT-2 small, 163 miljoen parameters zonder tying; 13% van GPT-2 medium, 406 miljoen; 8% van GPT-2 large, 838 miljoen; en 5% van GPT-2 XL, 1.638 miljoen."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **weight tying** | één tabel voor de embedding en de output head, zodat de logit van een token het inproduct met zijn eigen embedding is | `self.head.weight = self.tok.weight` |
        | $\mathbf{E}$, $\mathbf{e}_v$ | de tabel van de embedding, shape (vocab, d), en zijn rij voor token $v$ | `self.tok.weight`, `self.tok.weight[v]` |
        | $\mathbf{h} \mathbf{E}^\top$ | de logits van een gekoppelde head voor elk token, zonder de bias terms | |
        """),
        L.watch_out("Koppel de parameter, niet een kopie. `nn.Parameter(self.tok.weight.clone())` geeft de head dezelfde getallen om mee te beginnen, maar als een eigen tabel: PyTorch telt hem mee, en het trainen drijft de twee uit elkaar. En `a is b` is alleen `True` als `a` en `b` één en hetzelfde object zijn: zo controleer je dat een model gekoppeld is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de head van een language model aan zijn embedding te koppelen:

    1. Maak de embedding `nn.Embedding(vocab, d)` en de head `nn.Linear(d, vocab)`, zoals eerder: beide weights hebben shape (vocab, d).
    2. Geef de head de parameter van de embedding: `self.head.weight = self.tok.weight`.
    3. Controleer het: `model.head.weight is model.tok.weight` geeft `True`, en `model.parameters()` bevat de tabel één keer.
    4. Train zoals eerder: de gradient van de tabel is de som van de gradients van beide kanten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een vocabulary van 4 tokens met $d = 2$, één stap: na token 1 moet token 3 volgen. Eerst niet gekoppeld, met twee tabellen die met dezelfde getallen beginnen; dan gekoppeld, met één. Het commentaar met `#` zegt welke stap van het recept elk deel is. De som is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_grad():
    import torch
    from torch import nn

    _start = [[0.5, -1.0], [1.0, 0.5], [-0.5, 0.5], [0.0, 1.0]]
    _ids, _target = torch.tensor([1]), torch.tensor([3])
    _loss_fn = nn.CrossEntropyLoss()
    # 1. niet gekoppeld: een tabel voor de embedding en een voor de head, met dezelfde getallen om mee te beginnen
    _E_in = torch.tensor(_start, requires_grad=True)
    _E_out = torch.tensor(_start, requires_grad=True)
    _loss_fn(_E_in[_ids] @ _E_out.T, _target).backward()
    print("embedding:", _E_in.grad)
    print("head:", _E_out.grad)
    # 2. gekoppeld: één tabel aan beide kanten
    _E = torch.tensor(_start, requires_grad=True)
    _loss_fn(_E[_ids] @ _E.T, _target).backward()
    print("gekoppeld:", _E.grad)
    # 4. de gradient van de gekoppelde tabel, uit de twee delen
    _sum = _E_in.grad  # maak af: _E_in.grad + _E_out.grad
    print("som:", _sum)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De embedding krijgt alleen in rij 1 een gradient: token 1 was de enige input. De head krijgt er een in elke rij, omdat softmax de gradient over elke logit verdeelt. Zoals hij is, is de som alleen het deel van de embedding, met nullen in rij 0, 2 en 3. Afgemaakt is hij gelijk aan de gekoppelde gradient, rij voor rij: in rij 1 is $0{,}5205 + 0{,}5045 = 1{,}0251$ (na afronden), en de andere rijen komen alleen van de head.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logits():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De logits van een gekoppelde head

        Maak `tied_logits(h, E)` af: geef voor de vectoren in de rijen van `h`, shape (n, d), en de tabel van de embedding `E`, shape (vocab, d), de logits van een gekoppelde head zonder bias terms, shape (n, vocab). De check roept `tied_logits([[1.0, 0.0, 2.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.5, -1.0, 2.0]])` aan, wat `[[1.0, 0.0, 3.0, 4.5]]` moet geven, en probeert meer vectoren en andere tabellen.
        """),
        L.task("Laat `tied_logits` het inproduct van elke vector van `h` met elke rij van `E` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logits():
    box_logits = L.code_box("tied-logits", lines=6)
    box_logits
    return (box_logits,)


@app.cell(hide_code=True)
def check_logits(box_logits):
    result_logits = L.check("tied-logits", box_logits.value)
    result_logits
    return (result_logits,)


@app.cell(hide_code=True)
def hints_logits():
    L.hints(
        "tied-logits",
        [
            "Een matrixproduct van `h` met iets geeft een rij voor elke vector van `h`. Welke shape moet dat iets hebben, om een kolom voor elk token te geven?",
            "`E` heeft een rij voor elk token, shape (vocab, d); je hebt (d, vocab) nodig: een kolom voor elk token.",
        ],
        why="`return h @ E.T`. Voor het voorbeeld geeft de vector (1, 0, 2) tegen de vier rijen 1, 0, 1 + 2 = 3 en 0,5 + 4 = 4,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_class():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een gekoppeld model

        `TiedLM` is het kleinste language model met beide tabellen: een embedding en een head, met niets ertussen. Vervang de `...` in `__init__` door de twee layers, zoals in je GPT, en de regel die ze koppelt. De check maakt `TiedLM(10, 4)` en vraagt of `model.head.weight is model.tok.weight`; dan telt hij de tensors met parameters, en controleert hij dat de head en de embedding na een trainingsstap nog gelijk zijn.
        """),
        L.task("Laat `TiedLM` één tabel gebruiken voor zijn embedding en zijn head."),
    ])
    return


@app.cell(hide_code=True)
def exercise_class():
    box_class = L.code_box("tied-lm", lines=12)
    box_class
    return (box_class,)


@app.cell(hide_code=True)
def check_class(box_class):
    result_class = L.check("tied-lm", box_class.value)
    result_class
    return (result_class,)


@app.cell(hide_code=True)
def hints_class():
    L.hints(
        "tied-lm",
        [
            "Stap 1 en 2 van het recept. Beide weights hebben shape (vocab, d), dus er hoeft niets getransponeerd te worden.",
            "Wijs aan `self.head.weight` de parameter van de embedding zelf toe, geen berekening ermee.",
        ],
        why="`self.tok = nn.Embedding(vocab, d)`, `self.head = nn.Linear(d, vocab)` en `self.head.weight = self.tok.weight`. Andersom, `self.tok.weight = self.head.weight`, koppelt ze ook: dan begint de tabel met de getallen van de head.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je model op de maat van GPT-2

    De cel hieronder maakt je `TiedLM` met de vocabulary van GPT-2 en $d = 768$, alleen de shapes, en telt zijn parameters zoals een optimizer ze ziet.
    """)
    return


@app.cell(hide_code=True)
def demo_counts(box_class, result_class):
    _both = C.counts(box_class.value) if result_class.state == "correct" else None
    mo.md(
        f"Je `TiedLM({C.VOCAB}, 768)` heeft {C.grouped(_both[0], L.lang)} parameters: één tabel en {C.grouped(C.VOCAB, L.lang)} bias terms. Niet gekoppeld zou hij er {C.grouped(_both[1], L.lang)} hebben: bijna de helft daarvan zou een tweede tabel zijn."
        if _both else "Als je `TiedLM` goed is, worden zijn parameters hier geteld."
    )
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.md(r"""
    ### Oefening 3. Welke rijen leren
    """)
    return


@app.cell(hide_code=True)
def choice_rows_cell():
    choice_rows = L.choice(
        "rows-grad",
        question="Een ongekoppeld model doet een trainingsstap op een batch waarin alleen de tokens \"de\" en \"kat\" als input voorkomen. Welke rijen krijgen een gradient?",
        options={
            "right": "Embedding: de rijen van \"de\" en \"kat\"; head: elke rij",
            "both-two": "Embedding en head allebei: alleen de twee rijen van \"de\" en \"kat\"",
            "both-all": "Embedding en head: elke rij",
        },
        correct="right",
        why={
            "both-two": "De head geeft een logit voor elk token, en softmax met cross-entropy geeft elke logit een gradient: elke rij van de head krijgt er een.",
            "both-all": "De embedding zoekt alleen de rijen van de tokens in de batch op; de andere rijen worden in deze stap niet gebruikt, en krijgen geen gradient.",
            "right": "Zoals in het uitgewerkte voorbeeld. Gekoppeld krijgt de tabel allebei, dus de rij van een token dat zelden input is, leert toch bij elke stap van de head.",
        },
    )
    choice_rows
    return (choice_rows,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De embedding van [Embedding layers](/course/open/nn-embedding-layer?lang=nl) en de output head van [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), nu één tabel, met de weight sharing van [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl): één parameter op twee plekken gebruikt, en een gradient van elk. Je eigen GPT leest maar 65 tekens, dus weight tying zou hem maar 65 × d getallen besparen, en het zou veranderen hoe zijn head begint: met de getallen van `nn.Embedding`, getrokken met een standaardafwijking van 1, zou zijn eerste loss ver boven 4,17 liggen, de loss van gelijkmatig gokken. Weight tying loont bij een vocabulary van tienduizenden tokens, zoals bij GPT-2, dat al zijn weights klein laat beginnen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rows, result_class, result_logits):
    L.wrap_up(result_logits, result_class, choice_rows)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
