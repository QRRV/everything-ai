# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les attn-positions. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-positions", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Multi-head attention](/course/open/attn-multi-head?lang=nl) en [Embedding layers](/course/open/nn-embedding-layer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_projection():
    quick_projection = L.choice(
        "quick-projection",
        question="Wat doet de output-projectie van multi-head attention?",
        options={
            "split": "Die verdeelt de vectoren over de heads, vóór de attention",
            "right": "Die mengt de outputs van de heads, nadat ze naast elkaar zijn gezet",
            "scale": "Die deelt de scores van elke head door de wortel uit zijn lengte",
        },
        correct="right",
        why={
            "split": "De heads nemen hun kolommen rechtstreeks uit $Q$, $K$ en $V$; de projectie komt erna. Zie [Multi-head attention](/course/open/attn-multi-head?lang=nl).",
            "scale": "Dat is het schalen binnen elke head, vóór zijn softmax. Zie [Multi-head attention](/course/open/attn-multi-head?lang=nl).",
            "right": "Een lineaire layer van $d$ naar $d$. Vandaag komt er een ander soort vector bij, vóór de attention: één voor elke plek.",
        },
    )
    quick_projection
    return (quick_projection,)


@app.cell(hide_code=True)
def quick_check_embedding():
    quick_embedding = L.choice(
        "quick-embedding",
        question="Hoeveel parameters heeft `nn.Embedding(50, 8)`?",
        options={
            "right": "400: een rij van 8 getallen voor elk van de 50 ids",
            "sum": "58: 50 ids en 8 getallen",
            "bias": "408: 400 weights en 8 bias terms",
        },
        correct="right",
        why={
            "sum": "Het is een tabel: elk id heeft een eigen rij, van 8 geleerde getallen. Zie [Embedding layers](/course/open/nn-embedding-layer?lang=nl).",
            "bias": "Een embedding layer zoekt alleen rijen op; hij heeft geen bias terms. Zie [Embedding layers](/course/open/nn-embedding-layer?lang=nl).",
            "right": "50 × 8 = 400. Vandaag geeft een tweede tabel zoals deze elke plek in de tekst een eigen rij.",
        },
    )
    quick_embedding
    return (quick_embedding,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    "dog bites man" en "man bites dog" bestaan uit dezelfde drie woorden, en betekenen iets anders. Een model dat tekst moet begrijpen, moet weten waar elk woord staat. Maar attention vergelijkt inhoud: de scores van een woord hangen af van de embeddings van de woorden, niet van waar ze staan.
    """)
    return


@app.cell(hide_code=True)
def predict_swap():
    guess_swap = L.guess(
        "swap",
        question="Attention zonder mask en zonder enige informatie over plekken: je verwisselt het eerste en het laatste woord van een zin. Wat gebeurt er met de nieuwe vectoren?",
        options={
            "all": "Die veranderen allemaal, omdat elk woord een andere volgorde ziet",
            "right": "Die wisselen ook van plek, en verder verandert er niets",
            "none": "Niets: elke rij blijft precies zoals hij was",
        },
        reveal="**Die wisselen ook van plek, en verder verandert er niets.** Elk woord vergelijkt zich met dezelfde woorden als eerder en middelt dezelfde embeddings: alleen de rijen komen in een andere volgorde naar buiten. Zo krijgt \"dog\" dezelfde nieuwe vector in \"dog bites man\" en in \"man bites dog\": attention behandelt een zin als een zak met woorden. Met de causal mask is het niet veel beter: in één layer ziet een woord welke woorden ervoor kwamen, maar niet in welke volgorde.",
    )
    guess_swap
    return (guess_swap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Geef elke plek een eigen vector, en tel die op bij de embedding van het token dat er staat: $\mathbf{x}_t = \mathbf{e}_\text{token} + \mathbf{p}_t$. Hetzelfde token op een andere plek heeft nu een andere vector, dus zijn scores en zijn nieuwe vector hangen af van waar het staat. Dit is **positional encoding**, en er zijn twee gangbare soorten.

        De oorspronkelijke transformer van 2017 gebruikte een **sinusoidal positional encoding**: vaste golven van $\sin$ en $\cos$, die tussen $-1$ en $1$ op en neer gaan, beginnend bij $\sin 0 = 0$ en $\cos 0 = 1$, en zich elke $2\pi \approx 6{,}28$ herhalen. Kolom $2j$ van plek $t$ krijgt een sinus, en kolom $2j + 1$ een cosinus van dezelfde hoek, elk paar wat trager dan het paar ervoor:
        """),
        L.formula(r"\begin{aligned} [[hidden:p_{t,\,2j}]] &= \sin\frac{t}{10000^{2j/d}} \\ [[hidden:p_{t,\,2j+1}]] &= \cos\frac{t}{10000^{2j/d}} \end{aligned}"),
        mo.md(r"""
        Geen twee plekken krijgen dezelfde rij. GPT-2 heeft in plaats daarvan een **learned position embedding**: een embedding layer van 1.024 rijen, één voor elke plek tot aan zijn langste tekst. In PyTorch is dat een tweede `nn.Embedding(block, d)`, met `block` het grootste aantal tokens dat het model tegelijk leest, en `torch.arange(n)` geeft de plekken 0, 1, ..., n − 1 om op te zoeken. Veel nieuwere modellen tellen niets bij de embeddings op, maar veranderen de scores, naar hoe ver twee plekken uit elkaar liggen: ALiBi trekt er een straf af die groeit met de afstand, en RoPE draait de queries en keys over een hoek die van de plek afhangt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **positional encoding** | | een vector voor elke plek, opgeteld bij de embedding van het token dat er staat, zodat attention plekken uit elkaar kan houden | `tok(ids) + pos(places)` |
        | **sinusoidal positional encoding** | | vaste posities uit sinus- en cosinusgolven, trager in latere kolommen | `math.sin(angle)` |
        | **learned position embedding** | | een geleerde tabel met een rij voor elke plek, zoals GPT-2 heeft | `nn.Embedding(block, d)` |
        | $\mathbf{p}_t$ | p t | de positievector van plek $t$ | `pos(torch.arange(n))[t]` |
        | `torch.arange(n)` | torch arange | de gehele getallen 0, 1, ..., n − 1, als tensor | `torch.arange(3)` |
        """),
        L.watch_out("De positie wordt opgeteld, niet naast het token gezet: dezelfde $d$ getallen dragen wat een token is en waar het staat, en het trainen zoekt dat uit. En een geleerde tabel heeft een vaste lengte: GPT-2 heeft geen rij voor plek 1.024 of later, dus het leest hooguit 1.024 tokens tegelijk."),
        mo.md(r"""
        De figuur: drie kolommen van de sinusoïden voor $d = 16$, over 64 plekken. Kolom 0 is een snelle golf, kolom 4 (gestreept) een die tien keer trager is, kolom 8 (de dunne lijn) een die honderd keer trager is. Elke plek heeft een eigen mengeling van de 16 golven, een beetje zoals de wijzers van een klok.
        """),
        mo.Html(C.waves_picture({"title": "Sinusoïden voor d = 16", "x": "plek t", "y": "waarde", "fast": "kolom 0: snel", "slow": "kolom 8: traag",
                                 "decimal": ",",
                                 "alt": "Drie golven over 64 plekken: kolom 0 gaat ongeveer tien keer op en neer, kolom 4 ongeveer één keer, en kolom 8 stijgt langzaam van 0 naar ongeveer 0,6."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Posities voor een tekst van $n$ tokens en embeddings van $d$ getallen:

    1. Sinusoïden: de hoek van plek $t$ in kolom $i$ en $i + 1$, voor elke even $i$, is $t / 10000^{i/d}$.
    2. De sinus van die hoek in kolom $i$, de cosinus in kolom $i + 1$.
    3. Geleerd in plaats daarvan: een tabel van plekken naast de tabel van tokens, beide van $d$ getallen.
    4. Elk token: de embedding van het token plus de embedding van zijn plek.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "dog bites man" als de ids 2, 7 en 5, en "man bites dog" als 5, 7 en 2, met een kleine attention zonder mask; `torch.no_grad()` omdat hier niets getraind wordt. De cel print hoeveel de nieuwe vector van "dog" verschilt tussen de twee zinnen. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_positions():
    import math
    import torch
    from torch import nn

    def _attention(x):
        A = torch.softmax(x @ x.T / math.sqrt(x.shape[1]), dim=1)
        return A @ x

    torch.manual_seed(0)
    # 3. een tabel van tokens, en ernaast een tabel van plekken
    _tok = nn.Embedding(10, 4)
    _pos = nn.Embedding(8, 4)
    _dog_bites_man = torch.tensor([2, 7, 5])
    _man_bites_dog = torch.tensor([5, 7, 2])
    with torch.no_grad():
        # 4. elk token: de embedding van het token plus die van zijn plek
        _x = _tok(_dog_bites_man)  # maak af: + _pos(torch.arange(3))
        _y = _tok(_man_bites_dog)  # maak af: + _pos(torch.arange(3))
        print(_attention(_x)[0] - _attention(_y)[2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, is het verschil `[0., 0., 0., 0.]`: "dog" krijgt precies dezelfde nieuwe vector aan het begin van de ene zin en aan het eind van de andere. Afgemaakt, met de plekken erbij opgeteld, is het verschil `[0.4660, -0.3531, -0.2565, -0.0249]`: de vector van "dog" hangt nu af van waar het staat, hier op plek 0 of op plek 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_too_long():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een tekst die te lang is
    """)
    return


@app.cell(hide_code=True)
def choice_too_long_cell():
    choice_too_long = L.choice(
        "too-long",
        question="Een model heeft geleerde position embeddings voor 1.024 plekken, zoals GPT-2. Het krijgt een tekst van 1.500 tokens. Wat gebeurt er?",
        options={
            "learns": "Het leert rijen voor de nieuwe plekken terwijl het ze leest",
            "again": "Het begint na plek 1.023 weer bij plek 0",
            "right": "Het heeft geen rijen voor plek 1.024 en later",
        },
        correct="right",
        why={
            "learns": "Een model leert alleen terwijl het getraind wordt, niet terwijl het leest. De tabel heeft 1.024 rijen, en meer heeft het niet.",
            "again": "Dan zouden twee tokens dezelfde plek krijgen, en zou het model ze door elkaar halen. De tabel houdt gewoon op bij 1.024.",
            "right": "Rij 1.024 opzoeken in een tabel van 1.024 rijen stopt met een fout. Het leest dus hooguit 1.024 tokens tegelijk: een langere tekst wordt afgekapt, of in delen gelezen. Dit is het context window van het model.",
        },
    )
    choice_too_long
    return (choice_too_long,)


@app.cell(hide_code=True)
def text_exercise_sinusoids():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Sinusoïden

        Maak `sinusoids(n, d)` af: de functie geeft de sinusoidal positional encoding van $n$ plekken met elk $d$ getallen, als NumPy-array van $n$ bij $d$, volgens stap 1 en 2 van het recept. Vervang de twee `...`. De check roept `sinusoids(3, 4)` aan, wat `[[0.0, 1.0, 0.0, 1.0], [0.8415, 0.5403, 0.01, 1.0], [0.9093, -0.4161, 0.02, 0.9998]]` moet geven, en probeert andere maten.
        """),
        L.task("Laat `sinusoids` elke plek zijn sinus- en cosinusgolven geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sinusoids():
    box_sinusoids = L.code_box("sinusoids", lines=12)
    box_sinusoids
    return (box_sinusoids,)


@app.cell(hide_code=True)
def check_sinusoids(box_sinusoids):
    result_sinusoids = L.check("sinusoids", box_sinusoids.value)
    result_sinusoids
    return (result_sinusoids,)


@app.cell(hide_code=True)
def hints_sinusoids():
    L.hints(
        "sinusoids",
        [
            "De hoek is de plek gedeeld door een getal dat groeit met de kolom: `10000 ** (i / d)`, 1 voor kolom 0.",
            "`math.sin(angle)` komt in `pe[t, i]`, en `math.cos(angle)` in de kolom erna.",
        ],
        why="Met $d = 4$ draaien kolom 0 en 1 met de hoek $t$, en kolom 2 en 3 met $t / 100$: plek 1 geeft $\\sin 1 \\approx 0{,}8415$ en $\\cos 1 \\approx 0{,}5403$, dan $\\sin 0{,}01 \\approx 0{,}01$ en $\\cos 0{,}01 \\approx 1$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_tokens_positions():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Tokens en plekken

        Maak de class `TokensAndPositions(vocab, d, block)` af: in `__init__` een tabel `self.tok` met een rij van $d$ getallen voor elk token van de vocabulary, en daarna een tabel `self.pos` met een rij voor elke plek tot aan `block`; in `forward` de embeddings van een tensor van token-ids `ids`, één tekst, plus die van hun plekken. De check maakt de module na `torch.manual_seed(2)` en geeft hem token 3, token 1 en weer token 3: de eerste en de laatste rij moeten nu verschillen.
        """),
        L.task("Laat `TokensAndPositions` de embedding van elke plek optellen bij de embedding van zijn token."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tokens_positions():
    box_tokens_positions = L.code_box("tokens-and-positions", lines=12)
    box_tokens_positions
    return (box_tokens_positions,)


@app.cell(hide_code=True)
def check_tokens_positions(box_tokens_positions):
    result_tokens_positions = L.check("tokens-and-positions", box_tokens_positions.value)
    result_tokens_positions
    return (result_tokens_positions,)


@app.cell(hide_code=True)
def hints_tokens_positions():
    L.hints(
        "tokens-and-positions",
        [
            "Twee `nn.Embedding`-layers, zoals in [Embedding layers](/course/open/nn-embedding-layer?lang=nl): één van `vocab` rijen en één van `block` rijen, beide met `d` getallen.",
            "De plekken van een tekst van `len(ids)` tokens zijn `torch.arange(len(ids))`; zoek ze op in `self.pos`, en tel de uitkomst op bij `self.tok(ids)`.",
        ],
        why="Beide tabellen geven één rij van $d$ getallen per token, dus ze kunnen getal voor getal opgeteld worden. Token 3 krijgt op plek 0 en op plek 2 dezelfde token-rij, maar een andere plek-rij, dus zijn vector verschilt.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tabellen uit [Embedding layers](/course/open/nn-embedding-layer?lang=nl), nu ook voor plekken, geven de multi-head attention uit [Multi-head attention](/course/open/attn-multi-head?lang=nl) wat die niet kon zien: waar elk token staat. Je `TokensAndPositions` is de input van een GPT. In de volgende les, *Het transformer block*, krijgt attention een klein MLP erachter, allebei met layer normalization en een residual connection: het blok dat een GPT keer op keer stapelt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_too_long, result_sinusoids, result_tokens_positions):
    L.wrap_up(choice_too_long, result_sinusoids, result_tokens_positions)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
