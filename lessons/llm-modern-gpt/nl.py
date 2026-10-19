# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-modern-gpt. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("llm-modern-gpt", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Grouped-query attention](/course/open/llm-gqa?lang=nl) en [De KV cache](/course/open/llm-kv-cache?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="Een layer heeft 4 query heads en 2 key en value heads. Welke key head gebruikt query head 3? Heads tellen vanaf 0.",
        options={
            "right": "Key head 1: query heads 2 en 3 delen hem",
            "own": "Key head 3: elke query head heeft zijn eigen",
            "first": "Key head 0: alle query heads delen de eerste",
        },
        correct="right",
        why={
            "own": "Dat is multi-head attention, met evenveel key heads als query heads. Met 2 key heads bedient elk een groep van 4 / 2 = 2 query heads: zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "first": "Dat is multi-query attention, met één key head. Met 2 delen query heads 0 en 1 key head 0, en 2 en 3 key head 1: zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "right": "`int(3 / (4 / 2))` = 1. Je ModernAttention deelt zijn key en value heads op dezelfde manier.",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def quick_check_cache():
    quick_cache = L.choice(
        "quick-cache",
        question="Je GPT schrijft met een KV cache en heeft 50 tokens geschreven. Wat gaat er bij de volgende stap door zijn layers?",
        options={
            "all": "Alle 50 tokens: de cache maakt elk ervan sneller",
            "none": "Helemaal niets: de cache heeft het volgende token en zijn logits al",
            "right": "Alleen het nieuwste token: de cache heeft de keys en values van de andere",
        },
        correct="right",
        why={
            "all": "Dan zou de cache niets besparen. De keys en values van de 50 tokens worden bewaard, dus alleen het nieuwste token wordt geprojecteerd en gaat door de MLP's: zie [De KV cache](/course/open/llm-kv-cache?lang=nl).",
            "none": "De cache bewaart keys en values, geen tokens. Het volgende token komt nog steeds uit de logits van een stap: zie [De KV cache](/course/open/llm-kv-cache?lang=nl).",
            "right": "Zijn query leest alle keys en values in de cache. In oefening 4 schrijft je eigen GPT zo.",
        },
    )
    quick_cache
    return (quick_cache,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(f"""
        ## Waarom

        Je GPT van [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) lijkt op GPT-2 uit 2019: layer normalization, een MLP (met ReLU, waar GPT-2 GELU heeft), een tabel van geleerde plekken bij de input, en evenveel key en value heads als query heads. Llama, Mistral en Qwen veranderden deze vier onderdelen, en ze schrijven met een KV cache. Je bouwde ze in de vorige vijf lessen elk apart. In dit project gaan ze in je GPT, en meet je wat elk met de loss doet: dezelfde verhalen, dezelfde tokens en evenveel stappen als je GPT op tokens, zodat alleen de verandering verschilt.

        **Een voorbeeld uit het dagelijks leven.** Een bakker die vier dingen tegelijk in een recept verandert en een beter brood krijgt, weet niet welke verandering het deed. Verander één ding per bakbeurt, bak elke versie een paar keer, en proef.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_change():
    _best = {_k: C.mean_best(_k) for _k in C.CHANGES}
    _step = {_k: _best[_k] - _best[C.CHANGES[_i - 1]] for _i, _k in enumerate(C.CHANGES) if _i > 0}
    _top = min(_step, key=_step.get)
    _label = {"rms": "RMSNorm", "swiglu": "SwiGLU", "rope": "RoPE", "gqa": "Grouped-query attention"}
    guess_change = L.guess(
        "change",
        question=f"De course trainde de GPT op tokens uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) met het CPU-plan, en voegde daarna de veranderingen één voor één toe: RMSNorm, dan SwiGLU, dan RoPE, dan grouped-query attention, elke versie drie keer. De GPT op tokens eindigde op {C.number(_best['gpt'], 2, L.lang)} per token. Welke verandering verlaagde de validation loss volgens jou het meest?",
        options={
            "rms": "RMSNorm in plaats van layer normalization",
            "swiglu": "SwiGLU in plaats van de MLP met ReLU",
            "rope": "RoPE in plaats van de tabel van plekken",
            "gqa": "Grouped-query attention, 2 key en value heads in plaats van 4",
        },
        reveal=f"**{_label[_top]}: {C.signed(_step[_top], 2, L.lang)} per token.** " + "; ".join(f"{_label[_k]}: {C.signed(_step[_k], 2, L.lang)}" for _k in _step) + ", elk de verandering in het gemiddelde van drie runs. De demo laat elke run zien, naast de spreiding tussen seeds.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vier onderdelen veranderen, de rest blijft.** De embedding van de tokens, de residual stream, pre-norm, de output head en de trainingsrun blijven zoals in je GPT. Vier onderdelen veranderen:

        | In je GPT | In de ModernGPT |
        |---|---|
        | layer normalization | RMSNorm: geen gemiddelde, geen bias terms |
        | MLP van $d$ naar $4d$ naar $d$, met ReLU | SwiGLU met een hidden size van `int(8 * d / 3)`: drie weight-matrices in plaats van twee, dus met ongeveer evenveel parameters |
        | een tabel van geleerde plekken, opgeteld bij de input | RoPE in elke attention, op de queries en keys; geen tabel, dus bij de maten van een ModernGPT hoort geen venstergrootte |
        | evenveel key en value heads als query heads | grouped-query attention: `kv_heads` key en value heads, elk gedeeld door `h / kv_heads` query heads |

        **RoPE op elke head apart.** Je `rope` draait de paren van een vector van $d$ getallen, elk paar met zijn eigen snelheid. In attention is een vector één head: de `d_head` getallen van die head. Knip de queries dus in heads, draai elke head met `rope`, en zet ze weer naast elkaar met `torch.cat`; hetzelfde voor de keys, met `kv_heads` heads.

        **De KV cache in de GPT.** Tijdens het schrijven krijgt elke attention een eigen cache: een dict. De eerste aanroep, met de hele start, vult hem met de keys en values van die tokens. Elke volgende aanroep krijgt één token. Dat token staat op plek $t$, het aantal keys dat al in de cache zit, dus RoPE moet het draaien als een token op plek $t$, niet 0: je `rope` krijgt een argument `start`, de plek van zijn eerste rij. Daarna komen de keys en values uit de cache eerst, en blijven ze allemaal in de cache. Met één query maakt je `gqa` een masker van 1 bij 1 dat niets verbergt, dus het nieuwe token leest elke key: dat zijn allemaal eerdere tokens, of het token zelf.

        **Meet elke verandering apart.** De course voegde de veranderingen één voor één toe, elk bovenop de vorige, en trainde elke versie drie keer, met seeds 0, 1 en 2, op de verhalen, tokens en stappen van je GPT op tokens. Zelfs met dezelfde seed trekt een versie met een nieuw onderdeel andere startwaarden en zijn mini-batches in een andere volgorde, dus twee runs verschillen al een beetje; een verandering telt pas als hij de loss meer verschuift dan die spreiding.
        """),
        L.watch_out("RoPE draait paren getallen, dus een head heeft er een even aantal nodig: `d / h` moet even zijn. De plannen van deze les hebben heads van 32 en 64 getallen."),
        mo.md(r"""
        Het beeld: het block van de ModernGPT, met in grijs onder elk onderdeel wat het vervangt. De residual stream loopt recht naar beneden, zoals in je Block.
        """),
        mo.Html(C.block_picture({"title": "Eén block van de ModernGPT", "input": "x: de embeddings van de tokens, zonder plekken", "input_was": "was: tokens plus de tabel van plekken",
                                 "output": "naar het volgende block", "norm": "RMSNorm", "norm_was": "was: layer normalization",
                                 "attention": "grouped-query attention, RoPE", "attention_was": "was: h key en value heads",
                                 "mlp": "SwiGLU: d → 8d/3 → d", "mlp_was": "was: d → 4d → d, met ReLU", "stream": "residual stream",
                                 "alt": "De residual stream loopt van x, de embeddings van de tokens zonder plekken, recht naar beneden naar het volgende block. Er gaan twee takken van af: RMSNorm, dan grouped-query attention met RoPE, terug opgeteld bij de stream; dan RMSNorm, dan SwiGLU van d naar 8d/3 naar d getallen, weer terug opgeteld. Onder elk onderdeel staat wat het vervangt: layer normalization, attention met h key en value heads, en een MLP van d naar 4d naar d met ReLU; bij de input de tabel van plekken."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om je GPT modern te maken, en elke verandering te controleren:

    1. Zet de vier onderdelen in je project, en geef `rope` een plek om bij te beginnen: `start`.
    2. Attention: queries voor `h` heads, keys en values voor `kv_heads` heads, elke head van de queries en keys door RoPE gedraaid vanaf `start`; met een cache eerst de keys en values daaruit, en bewaar ze allemaal.
    3. Block en GPT: je Block en GPT met RMSNorm en SwiGLU, zonder de tabel van plekken; elk block krijgt zijn eigen cache.
    4. Train hem op de tokens van je GPT op tokens, met hetzelfde plan en dezelfde seed, en vergelijk de loss met de runs van de course, verandering na verandering.
    5. Schrijf met de caches: de start één keer, daarna één token per stap.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 en 3 in getallen, voor het CPU-plan: wat elke verandering met de parameters doet, en wat grouped-query attention met de KV cache van een vol venster doet.
    """)
    return


@app.cell
def example_counts():
    _d, _h, _kv_heads, _n_layers, _block = 128, 4, 2, 5, 128
    _d_head = int(_d / _h)
    # 2. keys en values: voor h heads, en voor kv_heads heads
    _kv = 2 * (_d * _d)
    _kv_gqa = 2 * (_d * _kv_heads * _d_head)
    print("keys en values:", _kv, "->", _kv_gqa, "per block")
    # 3. de MLP van je Block, d -> 4d -> d met bias terms, en SwiGLU met een hidden size van int(8 * d / 3)
    _hidden = int(8 * _d / 3)
    _mlp = _d * 4 * _d + 4 * _d + 4 * _d * _d + _d
    _swiglu = 3 * _d * _hidden
    print("MLP:", _mlp, "-> SwiGLU:", _swiglu, "met hidden size", _hidden)
    # 3. de normalisaties, twee per block en één aan het eind: RMSNorm heeft geen bias terms
    _norms = 2 * _n_layers + 1
    print("normalisaties:", _norms * 2 * _d, "->", _norms * _d)
    # 3. de tabel van plekken: block rijen van d getallen, weg
    print("tabel van plekken:", _block * _d, "-> 0")
    # de KV cache van een vol venster: elk block bewaart per token een key en een value, voor elke key en value head
    _cache = _n_layers * 2 * _h * _d_head * _block
    _cache_gqa = _n_layers * 2 * _kv_heads * _d_head * _block
    print("KV cache:", _cache, "->", _cache_gqa, "getallen")
    return


@app.cell(hide_code=True)
def text_after_example():
    _p = C.plan_for("cpu")
    _hidden = int(8 * 128 / 3)
    mo.md(f"""
    Grouped-query attention halveert de keys en values, in de parameters en in de cache. SwiGLU met een hidden size van {_hidden} heeft {C.grouped(3 * 128 * _hidden, L.lang)} parameters waar de MLP er {C.grouped(2 * 128 * 512 + 512 + 128, L.lang)} had: bijna evenveel, dus een lagere loss kan niet van meer parameters komen. Alles bij elkaar heeft de ModernGPT van het CPU-plan {C.grouped(_p['params'], L.lang)} parameters, je GPT op tokens {C.grouped(_p['params_gpt'], L.lang)}.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je onderdelen in je project

        Stap 1 van het recept. Kopieer je `RMSNorm` uit [RMSNorm](/course/open/llm-rmsnorm?lang=nl), je `SwiGLU` uit [Moderne MLP's: GELU, SiLU en SwiGLU](/course/open/llm-swiglu?lang=nl), je `rope` uit [Rotary position embeddings (RoPE)](/course/open/llm-rope?lang=nl) en je `gqa` uit [Grouped-query attention](/course/open/llm-gqa?lang=nl) naar het vak: open elke les en kopieer daar de code uit je vak. Geef `rope` daarna nog één argument, `start=0`: de plek van zijn eerste rij. De check draait elk onderdeel zoals zijn eigen les deed, en ook `rope(x, start=3)`: twee rijen die op plek 3 en 4 staan.
        """),
        L.task("Zet je vier onderdelen in het vak, en laat de plekken van `rope` bij `start` beginnen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parts():
    box_parts = L.code_box("parts", lines=40)
    box_parts
    return (box_parts,)


@app.cell(hide_code=True)
def check_parts(box_parts):
    result_parts = L.check("parts", box_parts.value)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "parts",
        [
            "De vier onderdelen blijven zoals ze waren; alleen `rope` verandert, op twee plekken: zijn `def`-regel en zijn hoeken.",
            "Geef `start` op de `def`-regel de waarde 0 voor als hij niet wordt meegegeven, zodat `rope(x)` werkt zoals eerst. In de hoeken schuift elke plek `start` op: tel hem bij `torch.arange(n)` op, tussen haakjes, vóór `[:, None]`.",
        ],
        why="Een rij van `rope` draait over zijn plek keer `theta`. Met `start` draaien de rijen van een stuk tekst zoals ze in de hele tekst zouden draaien, vanaf plek `start`: zo wordt een nieuw token in oefening 2 op zijn eigen plek gedraaid, en zijn zijn scores met de eerdere keys die van de hele tekst.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_attention():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ModernAttention

        Stap 2 van het recept. Maak `ModernAttention(d, h, kv_heads)` af, met je `rope` en `gqa` uit oefening 1. `turn(x, n_heads, start)` draait elke head van `x` apart. `forward(x, cache=None)` werkt zonder cache tijdens het trainen; tijdens het schrijven is `cache` een dict die de keys en values van de eerdere tokens bewaart, onder `"k"` en `"v"`. De check maakt kleine, zoals `ModernAttention(8, 4, 2)`, en kijkt naar hun weights, naar `turn`, en naar een tekst van 5 tokens: 4 met een lege cache, dan de laatste alleen, die moet geven wat de hele tekst op zijn plek geeft.
        """),
        L.task("Laat `ModernAttention` de queries en keys van elke head draaien, en zijn keys en values in de cache bewaren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attention():
    box_attention = L.code_box("modern-attention", lines=34)
    box_attention
    return (box_attention,)


@app.cell(hide_code=True)
def check_attention(box_attention, result_parts):
    result_parts  # check again when exercise 1 saves your parts
    result_attention = L.check("modern-attention", box_attention.value)
    result_attention
    return (result_attention,)


@app.cell(hide_code=True)
def hints_attention():
    L.hints(
        "modern-attention",
        [
            "`__init__`: de queries `d` getallen breed, de keys en values `kv_heads * self.d_head`, alle drie zonder bias terms, en `out` zoals in je CausalSelfAttention. `turn`: een loop over de `n_heads` heads, `rope` met `start=start` op de kolommen van elke head, en de gedraaide heads naast elkaar met `torch.cat` op de laatste as.",
            "`forward`: draai de queries met `turn` en `self.h` heads, de keys met `self.kv_heads`; de values blijven zoals ze komen. Met keys in de cache zet je die vóór de nieuwe, met `torch.cat` op de as van de tokens; met een cache, ook een lege, bewaar je alle keys en values erin, onder `\"k\"` en `\"v\"`.",
        ],
        why="Alleen de queries en keys worden gedraaid, elke head als een eigen vector, zodat een score afhangt van hoe ver twee tokens uit elkaar staan. De cache bewaart de keys gedraaid op hun eigen plek; een nieuw token wordt gedraaid op plek `start`, erachter, en zijn query leest ze allemaal.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gpt():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. ModernBlock en ModernGPT

        Stap 3 van het recept. Maak `ModernBlock(d, h, kv_heads)` en `ModernGPT(vocab, d, h, kv_heads, n_layers)` af: je Block uit [Het transformer block](/course/open/attn-transformer-block?lang=nl) en je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), met de namen die ze hadden, `mlp` voor de SwiGLU, en zonder de tabel van plekken. `forward(ids, caches=None)` krijgt tijdens het schrijven een lijst caches, één dict per block. De check maakt kleine, zoals `ModernGPT(20, 16, 4, 2, 3)`, telt hun parameters, en geeft een tekst van 5 tokens één keer in zijn geheel en één keer als 4 en dan 1, met caches.
        """),
        L.task("Maak `ModernBlock` en `ModernGPT`, en geef elk block zijn eigen cache."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gpt():
    box_gpt = L.code_box("modern-gpt", lines=36)
    box_gpt
    return (box_gpt,)


@app.cell(hide_code=True)
def check_gpt(box_gpt, result_parts, result_attention):
    result_parts, result_attention  # check again when exercise 1 or 2 saves your module
    result_gpt = L.check("modern-gpt", box_gpt.value)
    result_gpt
    return (result_gpt,)


@app.cell(hide_code=True)
def hints_gpt():
    L.hints(
        "modern-gpt",
        [
            "Een block: je Block, met je `RMSNorm(d)` waar het `nn.LayerNorm(d)` had, `ModernAttention(d, h, kv_heads)` als `attention`, en je `SwiGLU` met een hidden size van `int(8 * d / 3)` als `mlp`, waar het `lin1` en `lin2` had. Zijn `forward` is die van je Block, met `cache` doorgegeven aan de attention.",
            "De GPT: je GPT zonder `pos`, met `RMSNorm` als `norm`. Ga in `forward` de blocks langs met hun nummer `i`, zodat block `i` `caches[i]` kan krijgen; als `caches` None is, roep je elk block aan zonder cache.",
        ],
        why="Het block is je Block met RMSNorm en SwiGLU; de attention brengt de plekken, dus de GPT begint met alleen de tokens. Elk block heeft eigen keys en values, dus elk block heeft een eigen cache nodig.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_gpt):
    result_gpt  # the note changes when exercise 3 saves your module
    _p = C.plan()
    _d, _h, _layers, _block = _p["sizes"]
    if _p["device"] == "cpu":
        _where = "Deze computer heeft geen GPU waar de grotere GPT op past" + (f" ({_p['small']} heeft te weinig geheugen)" if _p["small"] else "") + ", dus het CPU-plan draait"
        _time = f" Op de computer van de course, een CPU met 4 cores, duurde de run ongeveer {round(C.minutes('gqa'))} minuten, op één thread."
    else:
        _where = f"Je computer heeft een GPU, {_p['name']}, dus je ModernGPT traint daarop"
        _time = ""
    mo.vstack([
        mo.md(f"""
        ### Wat je project nu kan

        De knop hieronder traint je ModernGPT zoals [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) je GPT traint: je tokenizer op de eerste 90% van de verhalen, die daarna allemaal worden geëncodeerd, en je `gpt_loaders` en `Trainer`, met hetzelfde plan en dezelfde seed. {_where}: een ModernGPT van {C.grouped(_p['params'], L.lang)} parameters, met $d = {_d}$, {_h} query heads en {_p['kv_heads']} key en value heads, {_layers} blocks en vensters van {_block} tokens, {_p['epochs']} epochs met mini-batches van {_p['batch']}. Daarna schrijft hij een verhaal dat begint met "{C.B.START}".{_time} De stopknop van de cel beëindigt de run eerder; de beste parameters tot dan blijven bewaard, in `work/projects/llm/gpt_modern.pt`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je ModernGPT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_gpt):
    result_gpt  # run again when exercise 3 saves your module
    _labels = {"title": "Jouw run", "x": "epoch", "y": "loss per token", "lang": L.lang, "guess": "gelijk gokken", "train": "training", "val": "validatie"}
    _names = {"gpt": "je GPT op tokens", "rms": "+ RMSNorm", "swiglu": "+ SwiGLU", "rope": "+ RoPE, geen tabel van plekken", "gqa": "+ grouped-query attention"}

    def _progress(_p):
        if _p["phase"] == "tokenizer":
            mo.output.replace(mo.md("**Je tokenizer traint** op de eerste 90% van de verhalen…"))
            return
        if _p["phase"] == "encode":
            mo.output.replace(mo.md("**De verhalen worden geëncodeerd** met je `encode_fast`…"))
            return
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · nog ongeveer {_m} {'minuten' if _m > 1 else 'minuut'} te gaan"
        _parts = [mo.md(f"**Epoch {_p['epoch']} van {_p['epochs']}** · stap {_p['step']} van {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.history_picture(_p["history"], {**_labels, "alt": "De training loss en validation loss van je run tot nu toe, na elke epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    _best = {_k: C.mean_best(_k) for _k in C.CHANGES}
    _step = {_k: _best[_k] - _best[C.CHANGES[_i - 1]] for _i, _k in enumerate(C.CHANGES) if _i > 0}
    _short = {"gpt": "GPT op tokens", "rms": "+ RMSNorm", "swiglu": "+ SwiGLU", "rope": "+ RoPE", "gqa": "+ GQA"}
    _course = mo.vstack([
        mo.Html(C.changes_picture([(_names[_k], [_r["best"] for _r in C.runs(_k)]) for _k in C.CHANGES],
                                  {"title": "De runs van de course, per verandering", "x": "beste validation loss per token, CPU-plan", "lang": L.lang,
                                   "alt": "Eén rij per GPT, elk met de veranderingen van de rijen erboven, en een stip voor elk van zijn drie runs: " + "; ".join(f"{_names[_k]}: " + ", ".join(C.number(_r['best'], 2, L.lang) for _r in C.runs(_k)) for _k in C.CHANGES) + "."})),
        mo.md("De beste validation loss per token met het CPU-plan: het gemiddelde van de drie runs van elke versie, en de spreiding, de hoogste min de laagste.\n\n| | parameters | gemiddeld | spreiding |\n|---|---|---|---|\n" + "\n".join(
            f"| {_short[_k]} | {C.grouped(C.runs(_k)[0].get('params', C.plan_for('cpu')['params_gpt']), L.lang)} | {C.number(C.mean_best(_k), 3, L.lang)} | {C.number(C.spread(_k), 3, L.lang)} |" for _k in C.CHANGES)),
        mo.md(f"""RoPE verlaagt de loss verreweg het meest, met {C.number(-_step['rope'], 2, L.lang)}: een tabel van plekken moet eerst geleerd worden voordat attention hem kan gebruiken, en RoPE geeft elke attention vanaf de eerste stap de afstand tussen twee tokens. SwiGLU verlaagt hem met {C.number(-_step['swiglu'], 2, L.lang)}. RMSNorm laat hem zoals hij was, {C.signed(_step['rms'], 3, L.lang)}, binnen de spreiding. Grouped-query attention verhoogt hem een beetje, met {C.number(_step['gqa'], 3, L.lang)}: meer dan de spreiding, maar klein naast wat het bespaart, de helft van de KV cache. De tijd gaat ook omhoog: alleen op één thread duurt een run van de GPT op tokens ongeveer {round(C.minutes('gpt'))} minuten, een van de ModernGPT {round(C.minutes('gqa'))}. Je RMSNorm is een paar kleine stappen van PyTorch waar `nn.LayerNorm` er één is, in C++ geschreven, en `turn` draait elke head apart, in een loop."""),
        mo.md(f"""In de runs van de course met het GPU-plan, met $d = {C.plan_for('gpu')['sizes'][0]}$, {C.plan_for('gpu')['sizes'][2]} blocks en vensters van {C.plan_for('gpu')['sizes'][3]} tokens, {C.grouped(C.plan_for('gpu')['params'], L.lang)} parameters op dezelfde verhalen, gaat het anders. Zijn ModernGPT haalde zijn laagste validation loss, {C.number(C.runs('gqa', 'gpu')[0]['best'], 2, L.lang)}, na epoch {C.runs('gqa', 'gpu')[0]['best_epoch']}, en ging daarna overfitten; de GPT op tokens haalde {C.number(C.runs('gpt', 'gpu')[0]['best'], 2, L.lang)} na epoch {C.runs('gpt', 'gpu')[0]['best_epoch']}. De ModernGPT leert sneller, dus met zoveel parameters voor deze verhalen begint hij ze ook eerder uit zijn hoofd te leren. Een verandering die in de ene opzet helpt, kan in een andere tegenvallen: meet haar in de opzet waarin je traint."""),
    ])
    if train_button.value:
        _result = C.train(_progress)
    else:
        _result = C.last_run()
    _ready = result_gpt.state == "correct"
    if _result is None:
        if _ready and not train_button.value:
            _out = mo.vstack([mo.md("*Druk op de knop om de run te starten. Tot dan de runs van de course:*"), _course])
        else:
            _out = mo.vstack([mo.md("*Maak eerst oefening 1 tot en met 3 af, en druk dan op de knop: de run gebruikt je ModernGPT. Tot dan de runs van de course:*"), _course])
        if C.trained()["source"] == "course":
            _c, _b = C.trained(), C.course_bpe()
            _out = mo.vstack([_out, mo.md(f"""En de eigen ModernGPT van de course: de ModernGPT van het CPU-plan, door de course getraind op {C.grouped(_c['chars'], L.lang)} tekens aan verhalen uit Cosmopedia, waarvan de licentie de course een model laat delen dat erop getraind is, en niet op TinyStories. Op deze verhalen, moeilijker dan TinyStories, is zijn validation loss {C.number(_c['best'], 2, L.lang)} per token; de GPT op tokens van de course uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) eindigde op dezelfde verhalen op {C.number(_b['best'], 2, L.lang)}. Hij schreef, met temperatuur 0,8:

> {_c['story'].replace(chr(10), chr(10) + '> ')}""")])
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*De run stopt met een fout. Slagen je modules in `work/projects/llm/` en `work/projects/nn/` nog voor hun checks? Verwijder een bestand dat je veranderd hebt, en check zijn oefening opnieuw.*",
            "memory": "*Je GPU had te weinig geheugen. Sluit andere programma's die hem gebruiken, zoals een game of een ander notebook, en druk opnieuw op de knop.*",
            "stopped": "*Je stopte de run tijdens zijn eerste epoch, dus er is niets van bewaard. Druk op de knop om opnieuw te beginnen.*",
        }[_result])
    else:
        _h = _result["history"]
        _plan = "gpu" if _result["device"] != "cpu" else "cpu"
        _on = {"cpu": "de CPU", "mps": "de GPU van Apple"}.get(_result["device"], _result["name"])
        _limit = C.number(C.THRESHOLD[_plan], 2, L.lang)
        if _result["stopped"]:
            _verdict = f"Je stopte de run na epoch {len(_h)} van {_result['epochs']}; de runs van de course met de ModernGPT eindigden, met alle {_result['epochs']} epochs, onder {_limit} per token."
        elif _result["best"] <= C.THRESHOLD[_plan]:
            _verdict = f"De runs van de course met de ModernGPT eindigden onder {_limit} per token, en de jouwe ook: je vier veranderingen werken zoals ze moeten."
        else:
            _verdict = f"De runs van de course met de ModernGPT eindigden onder {_limit} per token, en de jouwe niet. Bekijk oefening 1 tot en met 3 nog eens, en die van de lessen waar je tokenizer en Trainer vandaan komen."
        _old = C.B.last_run()
        if _old is not None and _old["device"] == _result["device"] and _old["epochs"] == _result["epochs"] and not _old["stopped"] and not _result["stopped"]:
            _diff = _old["best"] - _result["best"]
            _versus = f" Je GPT op tokens met hetzelfde plan eindigde op {C.number(_old['best'], 2, L.lang)} per token: " + (f"je ModernGPT eindigde {C.number(abs(_diff), 2, L.lang)} {'lager' if _diff > 0 else 'hoger'}." if abs(_diff) >= 0.005 else "je ModernGPT eindigde ongeveer even hoog.")
        else:
            _versus = f" De GPT op tokens van de course met het {_plan.upper()}-plan eindigde op {C.number(C.mean_best('gpt', _plan), 2, L.lang)} per token."
        _saved = "" if not _result.get("kept") else " Je eerdere run had een lagere validation loss, dus `work/projects/llm/gpt_modern.pt` houdt de parameters van die run."
        _who = "" if not _result["stand_ins"] else " De versie van de course viel in voor " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + "."
        _story = "" if not _result.get("story") else "\n\nHet verhaal dat je ModernGPT schreef, met temperatuur 0,8:\n\n> " + _result["story"].replace("\n", "\n> ")
        _out = mo.vstack([
            mo.Html(C.history_picture(_h, {**_labels, "alt": f"De training loss en validation loss van je run na elke epoch: de validation loss is het laagst na epoch {_result['best_epoch']}, met {C.number(_result['best'], 2, L.lang)} per token."}, _result["epochs"])),
            mo.md(f"""Je ModernGPT van {C.grouped(_result['params'], L.lang)} parameters trainde {len(_h)} {'epochs' if len(_h) > 1 else 'epoch'} op {_on}, in {C.number(_result['minutes'], 1, L.lang)} minuten, tokenizer en encoderen meegerekend. Zijn validation loss daalde tot **{C.number(_result['best'], 2, L.lang)} per token** na epoch {_result['best_epoch']}: {C.number(_result['best_per_char'], 3, L.lang)} per teken.{_versus}{_saved} {_verdict}{_who}{_story}"""),
            _course,
        ])
    trained_run = _result
    _out
    return (trained_run,)


@app.cell(hide_code=True)
def text_exercise_generate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Schrijven met de cache

        Stap 5 van het recept. Maak `generate(model, ids, n)` af: hij schrijft `n` ids na `ids`, elke keer de waarschijnlijkste volgende, zoals greedy decoding doet, met een cache voor elk block van `model`. De check geeft hem een kleine ModernGPT, vergelijkt zijn ids met die van de langzame manier, die bij elke stap alle ids door het model stuurt, en telt hoeveel tokens het model per aanroep krijgt.
        """),
        L.task("Laat `generate` de start één keer sturen, en daarna één nieuw id per stap, met de caches."),
    ])
    return


@app.cell(hide_code=True)
def exercise_generate():
    box_generate = L.code_box("generate", lines=18)
    box_generate
    return (box_generate,)


@app.cell(hide_code=True)
def check_generate(box_generate, result_gpt):
    result_gpt  # check again when exercise 3 saves your module
    result_generate = L.check("generate", box_generate.value)
    result_generate
    return (result_generate,)


@app.cell(hide_code=True)
def hints_generate():
    L.hints(
        "generate",
        [
            "Stap 5 van het recept: eerst een lijst met een nieuwe lege dict voor elk block in `model.blocks`, dan de hele start één keer door het model, als tensor van één rij, met de caches.",
            "In de loop: het id met de grootste logit op de laatste plek, achteraan in `out`; dan alleen dat id door het model, als tensor van één rij en één kolom, met dezelfde caches.",
        ],
        why="De start gaat één keer door het model en vult elke cache; daarna kost een stap één token per layer, en zijn query leest de keys en values van alles ervoor. De ids zijn die van de langzame manier, op afronding na.",
    )
    return


@app.cell(hide_code=True)
def demo_speed(result_generate, trained_run):
    result_generate, trained_run  # run again when exercise 4 saves your module, or after a run
    _s = C.speed()
    if _s is None:
        _out = mo.md("*Maak eerst oefening 4 af: dan schrijft je `generate` met je getrainde ModernGPT, of met die van de course zolang je er zelf geen getraind hebt.*")
    elif isinstance(_s, str):
        _out = mo.md("*Het schrijven stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks? Verwijder een bestand dat je veranderd hebt, en check zijn oefening opnieuw.*")
    else:
        _whose = "je ModernGPT uit de demo hierboven" if _s["source"] == "yours" else "de ModernGPT van de course, getraind op verhalen uit Cosmopedia, zolang je er zelf geen getraind hebt"
        _same = "dezelfde ids" if _s["same"] else "bijna dezelfde ids: bij een stap lagen de logits van twee ids zo dicht bij elkaar dat afronding de andere koos"
        _out = mo.md(f"""
        ### Wat je project nu kan: schrijven met een cache

        Je `generate` schreef {_s['tokens']} ids na "{C.B.START}" met {_whose}, in {C.number(_s['cache'], 2, L.lang)} seconden. De langzame manier, bij elke stap alle ids door het model, deed er {C.number(_s['full'], 2, L.lang)} seconden over, en schreef {_same}. Met een venster van {_s['block']} tokens bevatten de caches van zijn {_s['layers']} blocks {C.grouped(_s['numbers'], L.lang)} getallen; met evenveel key en value heads als query heads, {_s['h']} in plaats van {_s['kv_heads']}, zouden ze er {C.grouped(_s['numbers_mha'], L.lang)} bevatten.
        """)
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gqa():
    mo.md(r"""
    ### Oefening 5. Waarom grouped-query attention?
    """)
    return


@app.cell(hide_code=True)
def choice_gqa_cell():
    _rope, _gqa = C.mean_best("rope"), C.mean_best("gqa")
    choice_gqa = L.choice(
        "gqa-why",
        question=f"In de runs van de course verandert grouped-query attention, 2 key en value heads in plaats van 4, de validation loss gemiddeld van {C.number(_rope, 3, L.lang)} naar {C.number(_gqa, 3, L.lang)} per token. Waarom gebruiken Llama, Mistral en Qwen het toch?",
        options={
            "loss": "Het verlaagt de loss van grote modellen, ook als het dat bij kleine niet doet",
            "right": "Het halveert de KV cache tijdens het schrijven, en de loss verandert nauwelijks",
            "flops": "Het halveert de FLOPs van attention tijdens het trainen",
        },
        correct="right",
        why={
            "loss": "Het paper claimt geen lagere loss: het laat een kwaliteit zien dicht bij die van multi-head attention, met een veel kleinere cache. Wat het bespaart is geheugen tijdens het schrijven: zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "flops": "Elke query head vergelijkt zijn queries nog steeds met alle keys, dus de scores kosten evenveel FLOPs als eerst; alleen de projecties van de keys en values worden kleiner. Zie [Grouped-query attention](/course/open/llm-gqa?lang=nl).",
            "right": "Met `h / kv_heads` query heads per key en value head is de cache `h / kv_heads` keer zo klein: twee keer in het CPU-plan, drie keer in het GPU-plan, 4 of 8 keer in Llama en Mistral, waar de cache van lange teksten gigabytes kost.",
        },
    )
    choice_gqa
    return (choice_gqa,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je GPT en zijn run uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl), met de onderdelen uit [RMSNorm](/course/open/llm-rmsnorm?lang=nl), [Moderne MLP's: GELU, SiLU en SwiGLU](/course/open/llm-swiglu?lang=nl), [Rotary position embeddings (RoPE)](/course/open/llm-rope?lang=nl) en [Grouped-query attention](/course/open/llm-gqa?lang=nl), en de cache uit [De KV cache](/course/open/llm-kv-cache?lang=nl). Je project `llm` heeft nu `modern_parts.py`, `modern_attention.py`, `modern_gpt.py` en `modern_generate.py`, en na de run een ModernGPT in `gpt_modern.pt`, met zijn merge rules in `gpt_modern.json`. De lessen over pretraining hierna trainen deze GPT op data die je zelf klaarmaakt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_parts, result_attention, result_gpt, result_generate, choice_gqa):
    L.wrap_up(result_parts, result_attention, result_gpt, result_generate, choice_gqa)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
