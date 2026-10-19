# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-count. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("gpt-count", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) en [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vocab():
    quick_vocab = L.choice(
        "quick-vocab",
        question="Welke twee delen van je GPT groeien mee met de grootte van de vocabulary?",
        options={
            "places": "De tabel van plekken en de output head",
            "right": "De tabel van tokens en de output head",
            "blocks": "De tabel van tokens en de blocks",
        },
        correct="right",
        why={
            "places": "De tabel van plekken heeft een rij voor elke plek, `block` stuks, hoe groot de vocabulary ook is. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "blocks": "Een block werkt op vectoren van d getallen; het weet niet hoeveel tokens er zijn. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "right": "Een rij voor elk token in de tabel, en een score voor elk token uit de head. Vandaag tel je hoe groot ze zijn.",
        },
    )
    quick_vocab
    return (quick_vocab,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-flops",
        question="Een computer die twee keer zo snel is, doet dezelfde forward pass van een netwerk. Wat gebeurt er met het aantal FLOPs van die forward pass?",
        options={
            "half": "Het halveert: de computer heeft de helft van de tijd nodig",
            "double": "Het verdubbelt: de computer doet er twee keer zoveel per seconde",
            "right": "Het blijft gelijk: FLOPs tellen het werk, niet de tijd",
        },
        correct="right",
        why={
            "half": "De tijd halveert, maar het netwerk heeft evenveel vermenigvuldigingen en optellingen nodig als eerst. Zie [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl).",
            "double": "Twee keer zoveel per seconde is de snelheid van de computer; de forward pass zelf heeft dezelfde berekeningen nodig. Zie [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl).",
            "right": "Vandaag tel je ze voor je GPT: per token, voor de parameters en voor attention.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voordat je een model traint, wil je twee dingen weten: hoe groot het is, en wat één token kost. Samen bepalen ze hoe lang het trainen duurt, en wat elk antwoord van het model kost. GPT-3 heeft 175 miljard parameters: één token lezen kost hem ongeveer 350 miljard FLOPs. Beide getallen kun je uitrekenen voordat je iets draait, uit de maten van het model: tel waar de parameters zitten, reken ongeveer 2 FLOPs per parameter voor elk token, en tel erbij op wat attention kost, en dat groeit met de lengte van de tekst.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "blocks-share",
        question="Een GPT met de maten van de kleinste GPT-2, een vocabulary van 50.257 tokens, $d = 768$ en 12 blocks, heeft een eigen output head. Welk deel van zijn parameters zit volgens jou in de 12 blocks?",
        options={
            "most": "Ongeveer 95%: de blocks zijn het model",
            "right": "Ongeveer de helft",
            "few": "Ongeveer 10%",
        },
        reveal="**Ongeveer de helft**: 85 miljoen van de 163 miljoen. De tabel van tokens en de output head hebben elk 50.257 × 768, ongeveer 38,6 miljoen, elk een kwart van het model. GPT-2 zelf gebruikt één matrix als tabel van tokens én als output head, en heeft 124 miljoen parameters, waarvan 68% in de blocks.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Het **parameter count** van je GPT, deel voor deel, bij een vocabulary van $|\mathcal{V}|$ tokens, vectoren van $d$ getallen, $L$ blocks en teksten van hooguit `block` tokens:

        - de tabel van tokens, $|\mathcal{V}| \cdot d$, en de tabel van plekken, `block` $\cdot\, d$;
        - elk block: attention, $4d^2 + d$ (query, key, value en de output-projectie, die bias terms heeft); twee layer normalizations, elk $2d$ (een $\gamma$ en een $\beta$ voor elk getal); en het MLP, $8d^2 + 5d$ (van $d$ naar $4d$ en terug, met bias terms);
        - de laatste layer normalization, $2d$, en de output head, $d \cdot |\mathcal{V}| + |\mathcal{V}|$.

        Bij een grote $d$ is een block ongeveer $12d^2$, en het hele model:
        """),
        L.formula(r"\begin{gathered} N \approx 12\,L\,d^2 + 2\,|\mathcal{V}|\,d \\ \text{FLOPs} \approx 2N + 2\,L\,n\,d \end{gathered}"),
        mo.md(r"""
        **FLOPs per token.** Net als in [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl) gebruikt een forward pass elke parameter één keer per token, voor een vermenigvuldiging en een optelling: ongeveer $2N$ FLOPs. Attention doet er een deel bij dat groeit met de lengte $n$ van de tekst. In elk block vergelijkt een token op plek $t$ zijn query met de keys van de $t$ tokens tot en met zichzelf, en mengt het hun values: ongeveer $4\,t\,d$ FLOPs. Door de causal mask kijkt een token van een tekst van $n$ tokens gemiddeld naar ongeveer $n/2$ ervan, dus kost attention ongeveer $2\,L\,n\,d$ per token, het tweede deel van de formule. Je attention uit [De causal mask](/course/open/attn-causal-mask?lang=nl) berekent de scores van de latere tokens toch, en verbergt ze pas daarna, dus die doet ongeveer twee keer zoveel: de formule telt het werk dat een token nodig heeft. Trainen heeft ook de backward pass nodig, die ongeveer twee keer zoveel kost als de forward pass: ongeveer drie keer de FLOPs van de forward pass per token, ongeveer $6N$ bij een korte tekst.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **parameter count** | hoeveel getallen een model leert, deel voor deel: 124 miljoen in GPT-2 |
        | **FLOPs per token** | de berekeningen voor één token: ongeveer $2N$ en attention vooruit, drie keer zoveel om te trainen |
        | $n$ | het aantal tokens van de tekst, hooguit `block` |
        """),
        L.watch_out("Niet elke parameter kost FLOPs: een rij van een tabel opzoeken vermenigvuldigt niets. De regel van $2N$ telt de tabellen toch mee, dus bij een klein model met een grote vocabulary schat hij te hoog. En $6N$ is per token: een mini-batch van 64 vensters van 256 tokens is 16.384 tokens, elk $6N$."),
        mo.md(r"""
        De figuur: waar de parameters zitten, als deel van elk model. In je kleine GPT zit bijna alles in de blocks; met de maten van GPT-2 nemen de tabel van tokens en de output head elk een kwart.
        """),
        mo.Html(C.parts_picture({"title": "Waar de parameters zitten", "small": "je GPT van de vorige les: {n}", "gpt2": "de maten van GPT-2: {n}",
                                 "tok": "tabel van tokens", "pos": "tabel van plekken", "blocks": "blocks", "head": "output head", "lang": "nl",
                                 "alt": "Twee groepen van vier balken, één per deel. Je GPT van de vorige les, 211.777 parameters: de tabel van tokens 2%, de tabel van plekken 2%, de blocks 94%, de output head 2%. De maten van GPT-2, 163.059.793 parameters: de tabel van tokens 24%, de tabel van plekken minder dan 1%, de blocks 52%, de output head 24%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De grootte van een GPT, en wat één token hem kost:

    1. De tabellen: $|\mathcal{V}| \cdot d$ voor de tokens, `block` $\cdot\, d$ voor de plekken.
    2. Elk block: attention $4d^2 + d$, twee layer normalizations van $2d$, en het MLP $8d^2 + 5d$; keer $L$.
    3. Aan het eind: $2d$ voor de laatste layer normalization, en $d \cdot |\mathcal{V}| + |\mathcal{V}|$ voor de output head.
    4. Eén token vooruit: ongeveer $2N$, en $2\,L\,n\,d$ voor attention in een tekst van $n$ tokens.
    5. Trainen op dat token: ongeveer drie keer de FLOPs van stap 4.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    GPT-2 small: zijn parameters, en de FLOPs van het trainen op één token van een tekst van 1.024 tokens. GPT-2 gebruikt één matrix als tabel van tokens én als output head, dus zijn eind is alleen de laatste layer normalization. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_count():
    _V, _d, _L, _block = 50257, 768, 12, 1024
    # 1. de tabellen
    _tables = _V * _d + _block * _d
    # 2. de blocks: attention, twee layer normalizations, het MLP
    _blocks = _L * ((4 * _d * _d + _d) + 2 * (2 * _d) + (8 * _d * _d + 5 * _d))
    # 3. het eind: de output head van GPT-2 is zijn tabel van tokens
    _end = 2 * _d
    _N = _tables + _blocks + _end
    print(_tables, _blocks, _end, _N)
    # 4. één token vooruit, in een tekst van 1.024 tokens
    _forward = 2 * _N  # maak af: + 2 * _L * _block * _d, voor attention
    print(_forward)
    # 5. trainen op dat token: vooruit, en terug voor ongeveer twee keer zoveel
    print(3 * _forward)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    GPT-2 small heeft volgens deze telling 124.412.160 parameters, 39 miljoen in de tabellen en 85 miljoen in de blocks: 27.648 minder dan GPT-2 zelf, want daar hebben de queries, keys en values ook bias terms. Zoals hij is, telt de cel $2N$: 248.824.320 FLOPs om één token te lezen, en drie keer zoveel om erop te trainen. Afgemaakt doet attention er 18.874.368 bij voor een tekst van 1.024 tokens, 7,6% meer: 803.096.064 FLOPs om op één token te trainen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flops():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén token, vooruit
    """)
    return


@app.cell(hide_code=True)
def choice_flops_cell():
    choice_flops = L.choice(
        "forward-flops",
        question="De GPT die je op een boek gaat trainen heeft ongeveer 10,8 miljoen parameters. Ongeveer hoeveel FLOPs heeft hij nodig om één token te lezen, vooruit?",
        options={
            "right": "Ongeveer 22 miljoen: 2 per parameter",
            "one": "Ongeveer 10,8 miljoen: één per parameter",
            "six": "Ongeveer 65 miljoen: 6 per parameter",
        },
        correct="right",
        why={
            "one": "Elke parameter kost twee FLOPs per token: een vermenigvuldiging en een optelling.",
            "six": "6 per parameter is trainen, vooruit en terug. Een token lezen is alleen de forward pass.",
            "right": "2 × 10,8 miljoen, en een beetje meer voor attention: 2 · 6 · 256 · 384, ongeveer 1,2 miljoen, voor vensters van 256 tokens. Trainen kost per token drie keer zoveel.",
        },
    )
    choice_flops
    return (choice_flops,)


@app.cell(hide_code=True)
def text_exercise_params():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tel je GPT

        Maak `gpt_params(vocab, d, n_layers, block)` af: die geeft hoeveel parameters je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) heeft, stap 1 tot en met 3 van het recept. Je GPT heeft een eigen output head, met bias terms. De check roept `gpt_params(65, 64, 4, 64)` aan, wat `211777` moet geven, zoals PyTorch je GPT van de vorige les telt, en probeert andere maten.
        """),
        L.task("Laat `gpt_params` de parameters van je GPT tellen, deel voor deel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_params():
    box_params = L.code_box("gpt-params", lines=11)
    box_params
    return (box_params,)


@app.cell(hide_code=True)
def check_params(box_params):
    result_params = L.check("gpt-params", box_params.value)
    result_params
    return (result_params,)


@app.cell(hide_code=True)
def hints_params():
    L.hints(
        "gpt-params",
        [
            "`embeddings` en `one_block`: stap 1 en 2 van het uitgewerkte voorbeeld, in de namen van deze functie, en voor één block.",
            "`end` is niet die van GPT-2: je GPT heeft een eigen output head, met een weight voor elk paar van een van de d getallen en een van de vocab tokens, en een bias term voor elk token.",
        ],
        why="Een tabel heeft een rij van d getallen voor elk token of elke plek. Een lineaire layer heeft een weight voor elk paar van een invoer en een uitvoer, en, met bias terms, één per uitvoer; alleen query, key en value hebben er geen. Een layer normalization heeft een γ en een β voor elk getal.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flops3():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen op één token

        Maak `train_flops(n_params, n_layers, n, d)` af: die geeft de FLOPs van het trainen van een GPT met `n_params` parameters, met `n_layers` blocks van `d` getallen, op één token van een tekst van `n` tokens, stap 4 en 5 van het recept. De check roept `train_flops(1_000_000, 4, 128, 128)` aan, wat `6393216` moet geven, en probeert andere GPT's. In Python is `1_000_000` een miljoen: de `_` maakt het alleen makkelijker te lezen.
        """),
        L.task("Laat `train_flops` de FLOPs tellen van het trainen op één token: parameters en attention, vooruit en terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flops3():
    box_flops = L.code_box("train-flops", lines=6)
    box_flops
    return (box_flops,)


@app.cell(hide_code=True)
def check_flops3(box_flops):
    result_flops = L.check("train-flops", box_flops.value)
    result_flops
    return (result_flops,)


@app.cell(hide_code=True)
def hints_flops3():
    L.hints(
        "train-flops",
        [
            "`forward`: stap 4 van het afgemaakte uitgewerkte voorbeeld, in de namen van deze functie.",
            "Trainen is de forward pass, en een backward pass van ongeveer twee keer zijn FLOPs. Hoeveel forward passes is dat samen?",
        ],
        why="In de backward pass geeft elke vermenigvuldiging van de forward pass er twee: één voor de gradient van elk van de twee getallen die ze vermenigvuldigde, zoals een weight en wat erin kwam. Dus kost de backward pass ongeveer twee keer de forward pass, en trainen ongeveer drie keer.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je functies kunnen

    De cel hieronder telt drie GPT's met je `gpt_params`: je GPT van de vorige les, `GPT(65, 64, 4, 4, 64)`; die je op een boek gaat trainen, `GPT(65, 384, 6, 6, 256)`; en een met de maten van GPT-2. PyTorch telt de eerste twee, gemaakt door je `GPT` uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl); de derde zou zo'n 650 MB geheugen kosten, dus die blijft weg. Daarna telt je `train_flops` wat trainen op één token kost voor de GPT van het boek, en voor een met de maten van GPT-2 bij steeds langere teksten, met een tabel van plekken zo lang als de tekst. Hij werkt zodra oefening 2 en 3 goed zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_count(box_params, box_flops, result_params, result_flops):
    _ok = result_params.state == "correct" and result_flops.state == "correct"
    if not _ok:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: deze cel gebruikt je `gpt_params` en je `train_flops`.*")
    else:
        _result = C.demo(box_params.value, box_flops.value)
        if isinstance(_result, str):
            _out = mo.md({
                "slow": "*Je GPT maken duurt hier te lang: meer dan een minuut.*",
                "gpt": "*Je `GPT` in `work/projects/llm/gpt.py` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check oefening 2 van [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) opnieuw.*",
            }[_result])
        else:
            _rows = []
            for _name, _r in zip(["vorige les", "voor het boek", "maten van GPT-2"], _result["rows"]):
                _pt = "weggelaten" if _r["pytorch"] is None else C.grouped(_r["pytorch"], L.lang)
                _rows.append(f"| {_name} | {C.grouped(_r['yours'], L.lang)} | {_pt} |")
            _long = [f"| {C.grouped(_g['n'], L.lang)} | {C.grouped(_g['flops'], L.lang)} | {C.number(100 * _g['attention'], 0, L.lang)}% |" for _g in _result["gpt2"]]
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + "."
            _book = _result["book"]
            _out = mo.md("| GPT | jouw telling | PyTorch |\n|---|---|---|\n" + "\n".join(_rows) + f"""

Je GPT voor het boek trainen op één token van een venster van 256 kost **{C.grouped(_book['flops'], L.lang)} FLOPs**, waarvan {C.number(100 * _book['attention'], 0, L.lang)}% voor attention. Met de andere maten van GPT-2 en vensters zo lang als de tekst neemt attention het over:

| tekst van | FLOPs om op één token te trainen | attention |
|---|---|---|
""" + "\n".join(_long) + f"""

Bij een korte tekst zit je met ongeveer 6 FLOPs per parameter dichtbij; bij een lange kan attention meer kosten dan alle parameters samen. Daarom zijn lange teksten duur.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De FLOPs uit [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl), nu voor elk token dat een GPT leest of waarvan hij leert, en de delen van je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), één voor één geteld. Hierna: *Project: je GPT op een boek trainen*, waar je GPT van 10,8 miljoen parameters leert van Tiny Shakespeare, voor ongeveer 68 miljoen FLOPs per token waarop hij traint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flops, result_params, result_flops):
    L.wrap_up(choice_flops, result_params, result_flops)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
