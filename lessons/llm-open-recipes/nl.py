# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-open-recipes. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-open-recipes", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl) en [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_read():
    quick_read = L.choice(
        "quick-read",
        question="Je `TokenWindows` opent 7 shards met `np.memmap`. Wanneer worden de ids van venster 500 van de schijf gelezen?",
        options={
            "right": "Als een DataLoader om venster 500 vraagt: `__getitem__` leest dan zijn ids",
            "start": "Als de Dataset gemaakt wordt: `__init__` leest elk venster aan het begin één keer",
            "never": "Nooit: een memory-mapped file houdt al zijn ids in het geheugen vanaf het moment dat hij geopend wordt",
        },
        correct="right",
        why={
            "start": "`__init__` opent alleen de shards en houdt bij waar elk venster begint. Zie [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl).",
            "never": "Een memory-mapped file blijft op de schijf; indexeren leest die ids. Zie [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl).",
            "right": "De demo van deze les traint op die manier uit je shards.",
        },
    )
    quick_read
    return (quick_read,)


@app.cell(hide_code=True)
def quick_check_swiglu():
    quick_swiglu = L.choice(
        "quick-swiglu",
        question="Wat doet de gate van SwiGLU in de MLP van je ModernGPT?",
        options={
            "next": "Hij kiest door welk block de tokens hierna gaan",
            "norm": "Hij normaliseert de input van de MLP, zoals RMSNorm doet",
            "right": "Hij schaalt elk getal van de andere tak, zodat de MLP sommige delen doorlaat en andere tegenhoudt",
        },
        correct="right",
        why={
            "next": "Elk token gaat door elk block, op volgorde. De gate werkt binnen één MLP: zie [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).",
            "norm": "De normalisatie komt vóór de MLP. De gate vermenigvuldigt: `F.silu(self.gate(x)) * self.up(x)`. Zie [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl).",
            "right": "nanochat, het recept van deze les, gebruikt een andere MLP: ReLU in het kwadraat, zonder gate.",
        },
    )
    quick_swiglu
    return (quick_swiglu,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een paper vertelt wat een team vond; een volledig open trainingsrecept laat je het zelf doen. nanochat, van Andrej Karpathy, is zo'n recept: zijn code, zijn data en elke instelling zijn openbaar, en één script doet alles, van de tokenizer tot een model waarmee je kunt chatten. In 2019 kostte het trainen van GPT-2 zo'n 43.000 dollar. nanochat houdt een leaderboard bij van hoe snel zijn recept de score van GPT-2 op CORE verslaat, een verzameling benchmarks, op één computer met 8 H100-GPU's, en elke inzending noemt de commit van de code die het deed. Zo'n recept van begin tot eind lezen laat zien welke keuzes een echte run maakt, en laat je er één veranderen om te zien wat die doet.

    **Een voorbeeld uit het dagelijks leven.** Een foto van een brood vertelt je dat het kan. Een recept met het meel, de hoeveelheden, de tijden en de oventemperatuur laat je het bakken, en nog eens bakken met minder gist om te zien wat de gist doet.
    """)
    return


@app.cell(hide_code=True)
def predict_speedrun():
    guess_speedrun = L.guess(
        "speedrun",
        question="In 2019 kostte het trainen van GPT-2 zo'n 43.000 dollar. Hoe lang heeft het beste recept van nanochat van maart 2026 volgens jou nodig, op één computer met 8 H100-GPU's, om de score van GPT-2 op CORE te verslaan?",
        options={
            "month": "Ongeveer een maand",
            "day": "Ongeveer een dag",
            "right": "Minder dan twee uur",
        },
        reveal="**Minder dan twee uur: 1,65 uur, het record van 14 maart 2026 op het leaderboard van nanochat.** De eerste inzending, van januari 2026, duurde 3,04 uur. Getallen met minder precisie (FP8), een grotere batch, betere data en twee rondes geautomatiseerd onderzoek, met veranderingen die AI-agents vonden, halveerden de tijd bijna in twee maanden, en elke verandering staat in de code van zijn inzending.",
    )
    guess_speedrun
    return (guess_speedrun,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een trainingsrecept** is alles wat een run nodig heeft, opgeschreven: de data, de tokenizer, het model, de optimizer en zijn schema, hoe lang je traint, hoe je evalueert, en de volgorde van de fasen. Dat van nanochat is één script, `runs/speedrun.sh`. Zonder zijn setup, zijn opties voor logging en de regels die op de download wachten, en met commentaar van de course in plaats van het zijne, zijn dit de stappen (nanochat, MIT-licentie, © 2025 Andrej Karpathy; op commit `92d63d4`):

        ```bash
        python -m nanochat.dataset -n 8            # download de eerste 8 shards van de data voor pretraining
        python -m nanochat.dataset -n 170 &        # en de rest van de 170 shards, terwijl de tokenizer traint
        python -m scripts.tok_train                # train de tokenizer op zo'n 2 miljard tekens
        python -m scripts.tok_eval                 # evalueer hem
        torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
        torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
        torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
        ```

        **De fasen.** De data: nanochats eigen opnieuw verpakte kopie van ClimbMix van NVIDIA, webtekst in parquet-shards, gedownload als ze nodig zijn. De tokenizer: byte-level BPE met een vocabulary van $2^{15}$ = 32.768 ids, een BOS-token aan het begin van elk document, en cijfers in groepjes van hooguit 2. Pretraining: `base_train`. Dan evaluatie, supervised fine-tuning voor chat (`chat_sft`), en de evaluatie daarvan. Deze les leest de pretraining; latere lessen doen de rest.

        **Eén knop voor de grootte.** `--depth` legt het aantal blocks vast, en daarmee al de rest: de breedte is de diepte keer 64, naar boven afgerond op een veelvoud van 128, de grootte van een head; het aantal heads is de breedte gedeeld door 128. De scaling-parameters, de matrices van de blocks en de output head, zijn ongeveer $12 \, L \, d^2 + V d$ voor $L$ blocks met breedte $d$ en een vocabulary van $V$. De tokens om op te trainen zijn een vast getal keer die parameters, 8 in de speedrun. De batch groeit met de tokens, als een machtswet, en het aantal stappen is de tokens gedeeld door de batch.

        **Het schema.** De learning rate stijgt 40 stappen lang in een rechte lijn, blijft op zijn piek, en daalt in de laatste 65% van de stappen in een rechte lijn naar 5% van de piek: een warmdown, in plaats van de cosinus uit [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl).

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **trainingsrecept** | | alles wat een trainingsrun nodig heeft, zo opgeschreven dat anderen hem kunnen draaien: data, tokenizer, model, optimizer, schema, lengte, evaluatie en de volgorde van de fasen | `runs/speedrun.sh` |
        | **warmdown** | | het eind van een schema waarin de learning rate in een rechte lijn daalt naar een klein deel van zijn piek | `lr_multiplier(step, ...)` |
        | **diepte** | | het aantal blocks van een model; nanochat legt er de breedte, de heads en de lengte van de run mee vast | `--depth=24` |
        """),
        mo.md(r"""
        Het beeld: het schema van nanochat over de 300 stappen van de demo van deze les, naast een cosinusschema met dezelfde warmup.
        """),
        mo.Html(C.schedule_picture({"title": "Learning rate, als deel van zijn piek", "x": "stap", "y": "deel van de piek", "lang": L.lang,
                                    "nanochat": "nanochat", "cosine": "cosinus",
                                    "alt": "Twee lijnen over 300 stappen. Die van nanochat stijgt in 40 stappen naar 1, blijft 1 tot stap 105, en daalt in een rechte lijn naar 0,05 bij stap 300. De cosinus stijgt op dezelfde manier en daalt dan in een halve cosinus naar 0."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een trainingsrecept te lezen:

    1. Zoek het script dat alles draait, en zet zijn stappen op volgorde.
    2. Voor elke stap: welke data hij leest, en welke instellingen hij krijgt.
    3. Zoek de regel die de maat van de run bepaalt: het model, de tokens, de batch en de stappen.
    4. Zoek het schema van de learning rate.
    5. Leg elk onderdeel naast wat je zelf bouwde: hetzelfde, of nieuw?
    6. Verander één instelling tegelijk, en meet wat die doet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 voor de speedrun, `--depth=24`, met de regels van nanochat, en een vocabulary van $2^{15}$.
    """)
    return


@app.cell
def example_speedrun():
    import math

    _depth, _vocab, _ratio = 24, 2 ** 15, 8
    # de breedte: diepte keer 64, naar boven afgerond op een veelvoud van 128; één head per 128
    _dim = math.ceil(_depth * 64 / 128) * 128
    _heads = int(_dim / 128)
    # de scaling-parameters, en de tokens om op te trainen
    _params = 12 * _depth * _dim * _dim + _vocab * _dim
    _tokens = _ratio * _params
    # de batch, vanuit het referentiemodel d12 van nanochat: 2 ** 19 tokens, groeiend als de tokens tot de macht 0,383
    _ref = _ratio * (12 * 12 * 768 * 768 + _vocab * 768)
    _batch = 2 ** round(math.log2(2 ** 19 * (_tokens / _ref) ** 0.383))
    print(_dim, _heads, _params, _tokens, _batch, int(_tokens / _batch))
    return


@app.cell(hide_code=True)
def text_after_example():
    _s = C.speedrun()
    mo.md(f"""
    Diepte 24 geeft een breedte van {C.number(_s['dim'], 0, L.lang)}, al een veelvoud van 128, dus {_s['heads']} heads, en {C.number(_s['params'], 0, L.lang)} scaling-parameters volgens de formule; nanochat telt er {C.number(729810624, 0, L.lang)}, omdat een paar kleine gates er niet in zitten. 8 tokens per parameter geven {C.number(_s['tokens'] / 1e9, 2, L.lang)} miljard tokens. Het referentiemodel, diepte 12, traint op `2 ** 19` tokens per stap; de speedrun heeft {C.number(_s['tokens'] / (8 * C.size_of(12, 2 ** 15)[2]), 1, L.lang)} keer zijn tokens, dus zijn batch is `2 ** 19` keer dat tot de macht 0,383, afgerond op een macht van 2: `2 ** 20`, {C.number(_s['batch'], 0, L.lang)} tokens. Dat geeft {C.number(_s['steps'], 0, L.lang)} stappen. De hele run volgt uit één getal, en daarom kan een inzending op het leaderboard "d24" of "d26" zeggen en een hele run bedoelen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat je bouwde, en wat nieuw is

        `nanochat/gpt.py` somt op wat het model van nanochat anders maakt dan de GPT-2 van 2019. Hier staan zes van die onderdelen, en drie andere uit zijn recept. Zet elk onderdeel op zijn plek: heeft je ModernGPT uit [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl) het ook, of is het nieuw in nanochat?
        """),
        L.task("Zet elk onderdeel van het recept van nanochat op zijn plek."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    sort_features = L.sort("features")
    sort_features
    return (sort_features,)


@app.cell(hide_code=True)
def check_features(sort_features):
    result_features = L.check("features", sort_features)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def text_exercise_size():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De grootte uit de diepte

        Maak `size(depth, vocab)` af: stap 3 van het recept voor het model. Hij geeft de breedte terug, de diepte keer 64 naar boven afgerond op een veelvoud van 128; het aantal heads, één per 128 getallen van de breedte; en de scaling-parameters, `12 * depth * dim * dim + vocab * dim`. De check roept `size(20, 32768)` aan, en andere.
        """),
        L.task("Laat `size` de breedte, de heads en de parameters van nanochat voor een diepte geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_size():
    box_size = L.code_box("size", lines=12)
    box_size
    return (box_size,)


@app.cell(hide_code=True)
def check_size(box_size):
    result_size = L.check("size", box_size.value)
    result_size
    return (result_size,)


@app.cell(hide_code=True)
def hints_size():
    L.hints(
        "size",
        [
            "Naar boven afronden op een veelvoud van 128 is: deel door 128, rond naar boven af met `math.ceil`, en vermenigvuldig weer met 128.",
            "Diepte 5 geeft 320, geen veelvoud van 128: naar boven afgerond 384, dus 3 heads.",
        ],
        why="Met hele heads van 128 heeft elk model van de familie heads van dezelfde grootte, wat zijn diepte ook is: een dieper model krijgt meer heads, geen grotere.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lr():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het schema van nanochat

        Maak `lr_multiplier(step, steps, warmup, warmdown_ratio, final)` af: stap 4, zoals `get_lr_multiplier` in `scripts/base_train.py` van nanochat het doet. Hij geeft het deel van de piek van de learning rate bij stap `step`, geteld vanaf 0, van een run van `steps` stappen. Tijdens de eerste `warmup` stappen stijgt het in een rechte lijn, van `1 / warmup` bij stap 0 naar 1. Daarna is het 1, tot en met stap `steps - warmdown`, waarbij `warmdown` gelijk is aan `round(warmdown_ratio * steps)`. Daarna daalt het in een rechte lijn, van 1 bij stap `steps - warmdown` naar `final` bij stap `steps`, één stap na de laatste. De check roept `lr_multiplier(10, 100, 20, 0.5, 0.1)` aan, en andere.
        """),
        L.task("Laat `lr_multiplier` het schema van nanochat geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lr():
    box_lr = L.code_box("lr-multiplier", lines=12)
    box_lr
    return (box_lr,)


@app.cell(hide_code=True)
def check_lr(box_lr):
    result_lr = L.check("lr-multiplier", box_lr.value)
    result_lr
    return (result_lr,)


@app.cell(hide_code=True)
def hints_lr():
    L.hints(
        "lr-multiplier",
        [
            "Drie delen: `step < warmup`, dan `step <= steps - warmdown`, dan de rest.",
            "In de warmdown: hoeveel stappen zijn er bij stap `step` nog over, tot stap `steps`? Gedeeld door `warmdown` is dat het deel dat nog komt: 1 aan het begin.",
        ],
        why="Het plateau op de piek leert snel; de rechte warmdown aan het eind verlaagt de loss flink in zijn laatste stappen. En de lengte van de warmdown volgt de run, dus een langere run heeft geen nieuw schema nodig.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _dim, _heads, _ = C.size_of(C.DEPTH, C.D.B.VOCAB)
    mo.vstack([
        mo.md(f"""
        ### Een paar honderd stappen van het recept

        De knop hieronder traint je ModernGPT met wat deze les uit nanochat las: de grootte van diepte {C.DEPTH} volgens de regel van nanochat, een breedte van {_dim} met {_heads} heads, en je `lr_multiplier` met de standaardwaarden van nanochat, een warmup van {C.SCHEDULE['warmup']} stappen en een warmdown over de laatste {round(100 * C.SCHEDULE['warmdown'])}% naar {round(100 * C.SCHEDULE['final'])}% van de piek, {C.STEPS} stappen lang, met {C.BATCH} vensters van {C.BLOCK} tokens. Hij leest de vensters met je `TokenWindows` uit de shards van [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), en houdt de laatste shard apart om op te valideren. De optimizer is AdamW, met een piek van {C.number(C.PEAK, 3, L.lang)}, niet Muon. Op de computer van de course, op zijn CPU, duurde de run ongeveer {round(C.MEASURED['seconds'] / 60)} minuten, op 2 threads; op een GPU gaat het veel sneller.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train 300 stappen")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, box_lr, result_lr):
    _m = C.MEASURED
    _labels = {"title": "Training loss, om de 10 stappen", "x": "stap", "y": "loss per token", "lang": L.lang}
    _course = mo.Html(C.curve_picture([(_m["curve"], True)], {**_labels, "key": "- - de run van de course",
                                                                "alt": f"De training loss van de run van de course, van {C.number(_m['curve'][0], 1, L.lang)} na 10 stappen omlaag naar {C.number(_m['curve'][-1], 2, L.lang)} na 300."}))
    _verdict = f"De run van de course eindigde met een validation loss van {C.number(_m['val'], 2, L.lang)} per token, {C.number(_m['per_char'], 2, L.lang)} per teken, in {round(_m['seconds'] / 60)} minuten op 2 threads. De GPT op tokens van de course uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) haalde {C.number(C.BPE_RUN['per_char'], 2, L.lang)} per teken na {C.BPE_RUN['epochs']} epochs, in {round(C.BPE_RUN['minutes'])} minuten op één thread. Die GPT was kleiner, 1,3 miljoen parameters met een context van 128 tokens, en had een andere tokenizer; per teken zijn de twee te vergelijken. Deze {C.STEPS} stappen zien de training shards nog niet eens één keer, en komen al dicht in de buurt."
    if result_lr.state != "correct":
        _out = mo.vstack([mo.md("*Maak eerst oefening 3 af en druk dan op de knop: de run gebruikt je `lr_multiplier`. Tot dan de run van de course:*"), _course, mo.md(_verdict)])
    elif not train_button.value:
        _out = mo.vstack([mo.md("*Druk op de knop om de run te starten. Tot dan de run van de course:*"), _course, mo.md(_verdict)])
    else:
        def _progress(_p):
            mo.output.replace(mo.md(f"**Stap {_p['step']} van {_p['steps']}** · training loss {C.number(_p['loss'], 2, L.lang)} per token · {round(_p['seconds'])} s"))
        _r = C.run(run_code(box_lr.value).names["lr_multiplier"], _progress)
        if _r == "run":
            _out = mo.md("*De run stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*")
        else:
            _on = "de CPU" if _r["device"] == "cpu" else "de GPU"
            _out = mo.vstack([
                mo.Html(C.curve_picture([(_m["curve"], True), (_r["curve"], False)], {**_labels, "key": "— jouw run   - - die van de course",
                                                                                      "alt": f"De training loss van jouw run, van {C.number(_r['curve'][0], 1, L.lang)} na 10 stappen omlaag naar {C.number(_r['curve'][-1], 2, L.lang)} na 300, naast de run van de course."})),
                mo.md(f"""Je ModernGPT van {C.number(_r['params'], 0, L.lang)} parameters trainde {C.STEPS} stappen op {_on}, in {round(_r['seconds'])} seconden. Zijn validation loss op de apart gehouden shard is **{C.number(_r['val'], 2, L.lang)} per token**, {C.number(_r['per_char'], 2, L.lang)} per teken. {_verdict}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_recipe():
    mo.md(r"""
    ### Oefening 4. Waarom een recept open is
    """)
    return


@app.cell(hide_code=True)
def choice_recipe_cell():
    choice_recipe = L.choice(
        "recipe",
        question="Elke inzending op het leaderboard van nanochat noemt de commit van de code die hem maakte. Wat kun je daardoor?",
        options={
            "best": "Weten dat de nieuwste commit altijd het beste model traint",
            "right": "Elk record opnieuw draaien, en zien welke veranderingen in code of instellingen het sneller maakten",
            "private": "De data geheimhouden, want de commit bevat alleen de code en niet waar de data vandaan komt",
        },
        correct="right",
        why={
            "best": "Een nieuwere commit kan sneller zijn en op een andere manier slechter; het leaderboard noemt de score van elke inzending, en met de code kun je die controleren.",
            "private": "De commit bevat de code die de data downloadt, dus de data is net zo openbaar als de code.",
            "right": "Dat maakt een recept open: je kunt het opnieuw draaien, en één ding tegelijk veranderen, zoals in stap 6 van het recept.",
        },
    )
    choice_recipe
    return (choice_recipe,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De ModernGPT uit [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl), nu op maat gemaakt met de regel van nanochat; de shards uit [Project: een datapipeline voor pretraining](/course/open/llm-data-pipeline?lang=nl), nu om uit te trainen; en de schedules uit [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl), nu met een warmdown. Latere lessen fitten een scaling law voor je model en pretrainen het langer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_features, result_size, result_lr, choice_recipe):
    L.wrap_up(result_features, result_size, result_lr, choice_recipe)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
