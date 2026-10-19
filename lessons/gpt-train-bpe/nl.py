# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les gpt-train-bpe. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-train-bpe", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl) en [Tekst genereren met je GPT](/course/open/gpt-generate?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_end():
    quick_end = L.choice(
        "quick-end",
        question="Je tokenizer heeft 1.000 merge rules. Welk id krijgt end-of-text?",
        options={
            "thousand": "1000: het eerste id na de 1.000 regels",
            "first": "256: het eerste id na de bytes",
            "right": "1256: na de bytes en de regels",
        },
        correct="right",
        why={
            "thousand": "De 256 bytes komen eerst, met ids 0 tot en met 255, en de regels nemen 256 tot en met 1255. Zie [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl).",
            "first": "256 is de eerste merge rule. End-of-text heeft een id nodig dat geen regel heeft: `256 + len(merges)`. Zie [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl).",
            "right": "`256 + len(merges)`: je GPT van deze les heeft dus een vocabulary van 1.257 ids.",
        },
    )
    quick_end
    return (quick_end,)


@app.cell(hide_code=True)
def quick_check_tau():
    quick_tau = L.choice(
        "quick-tau",
        question="Je GPT schrijft met temperatuur 0,8 in plaats van 1. Wat verandert er?",
        options={
            "right": "Waarschijnlijke ids worden nog waarschijnlijker: veiliger tekst",
            "random": "Elk id krijgt een gelijkere kans: wildere tekst",
            "nothing": "Niets: de temperatuur doet er alleen toe tijdens het trainen",
        },
        correct="right",
        why={
            "random": "Dat doet een temperatuur boven 1. Onder 1 deel je de logits door minder dan 1, dus komen de grootste verder van de rest te liggen. Zie [Tekst genereren met je GPT](/course/open/gpt-generate?lang=nl).",
            "nothing": "De temperatuur deelt de logits vóór de softmax, elke keer dat de GPT een stap schrijft. Zie [Tekst genereren met je GPT](/course/open/gpt-generate?lang=nl).",
            "right": "De demo van deze les schrijft zijn verhaal met 0,8.",
        },
    )
    quick_tau
    return (quick_tau,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(f"""
        ## Waarom

        Je GPT uit [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl) leest en schrijft één teken tegelijk: zijn venster van 128 tekens is ongeveer 25 woorden. Je tokenizer uit [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl) knipt tekst in stukken van een paar tekens: op {C.grouped(5_000_000, L.lang)} tekens aan korte verhalen bevat een token gemiddeld {C.number(C.CHARS_PER_TOKEN, 1, L.lang)} tekens. Dan past er in hetzelfde venster {C.number(C.CHARS_PER_TOKEN, 1, L.lang)} keer zoveel verhaal, en leert elke stap van het trainen van {C.number(C.CHARS_PER_TOKEN, 1, L.lang)} keer zoveel tekst.

        Alleen gaat je `train_pre_bpe` in elke ronde langs elk stuk van de tekst. De eerste 90% van de verhalen heeft {C.grouped(C.PIECES, L.lang)} stukken, dus 1.000 rondes duren op de computer van de course {C.NAIVE_MINUTES} minuten. Deze les maakt je tokenizer snel genoeg, en traint dan je GPT op zijn tokens.

        **Een voorbeeld uit het dagelijks leven.** Bij de zelfscankassa, twaalf blikken van dezelfde soep: je scant er één en tikt × 12 in, in plaats van alle twaalf te scannen.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_speed():
    guess_speed = L.guess(
        "speed",
        question=f"Van de {C.grouped(C.PIECES, L.lang)} stukken zijn er veel hetzelfde: \" the\" alleen al komt {C.grouped(C.TOP_PIECES[2][1], L.lang)} keer voor. Een BPE die één keer langs elk verschillend stuk gaat, en elk paar erin zo vaak telt als het stuk voorkomt, leert dezelfde 1.000 regels. Hoeveel sneller is die volgens jou?",
        options={
            "two": "Ongeveer 2 keer: hij doet nog steeds 1.000 rondes",
            "ten": "Ongeveer 10 keer: de meeste stukken zijn verschillend",
            "right": "Ongeveer 100 keer: er zijn weinig verschillende stukken",
        },
        reveal=f"**Ongeveer {C.SPEEDUP} keer: {C.FAST_SECONDS} seconden in plaats van {C.NAIVE_MINUTES} minuten**, op de computer van de course. De verhalen hebben maar {C.grouped(C.DISTINCT, L.lang)} verschillende stukken, {C.SPEEDUP_PIECES} keer minder dan alle {C.grouped(C.PIECES, L.lang)}: kinderverhalen gebruiken weinig woorden, steeds weer. Elke ronde telt nog steeds paren en voegt samen, maar in {C.grouped(C.DISTINCT, L.lang)} stukken.",
    )
    guess_speed
    return (guess_speed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Tel elk verschillend stuk één keer.** Knip de tekst in stukken, zoals in [Pre-tokenization](/course/open/tok-pre-tokenization?lang=nl), en tel in een dict `counts_of` hoe vaak elk stuk voorkomt. Een paar in een stuk dat 3 keer voorkomt, telt dan 3 keer: tel de telling van het stuk op, niet 1. De tellingen van de paren zijn precies die van je `train_pre_bpe`, en dus ook de merge rules, in dezelfde volgorde, uit veel minder stukken.

        **Encodeer elk verschillend stuk één keer.** Hetzelfde bij het encoderen: een stuk gaat door `encode_part` als je het voor het eerst tegenkomt, en zijn ids gaan in een dict `known`. Elke volgende keer haal je ze daar. Een uitkomst onthouden om hem opnieuw te gebruiken heet **caching**.

        **Je GPT op tokens.** Dezelfde GPT en Trainer als op tekens, met 1.257 ids in plaats van de tekens: alleen de embedding en de output layer worden groter. Zijn loss is per token, en een token bevat een paar tekens. Om hem te vergelijken met een GPT op tekens, verdeel je hem dus over de tekens: loss per teken = loss per token × tokens ÷ tekens.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **caching** | | een uitkomst onthouden, zodat dezelfde vraag later geen werk meer kost | `known[piece]` |
        | **tekens per token** | | hoeveel tekens van een tekst een token gemiddeld bevat: de lengte van de tekst gedeeld door zijn aantal tokens | `len(text) / len(ids)` |
        """),
        L.watch_out("Voeg toe aan een lange lijst met `append`. `ids = ids + more` bouwt elke keer een hele nieuwe lijst: met 1,5 miljoen ids duurt dat ongeveer een uur in plaats van seconden."),
        mo.md(r"""
        De figuur: de stukken van "the the the cat cow cup", het voorbeeld van oefening 1. Links elk stuk, rechts elk verschillend stuk één keer met zijn telling. Het paar `t h` zit één keer in "the" en twee keer in "␣the": 1 + 2 = 3, zoals links.
        """),
        mo.Html(C.pieces_picture({"title": "Elk stuk, of elk verschillend stuk één keer", "all": "elk stuk", "once": "elk verschillend stuk één keer",
                                  "count": "telling", "pair": "paar t h: 1 + 2 = 3",
                                  "alt": "Links de zes stukken the, ␣the, ␣the, ␣cat, ␣cow en ␣cup. Rechts de vijf verschillende stukken met hun tellingen: the 1, ␣the 2, ␣cat 1, ␣cow 1, ␣cup 1. Het paar t h telt 1 + 2 = 3 keer."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om je GPT op je eigen tokens te trainen:

    1. Tel de stukken: elk stuk van elk deel tussen twee end-of-texts, met hoe vaak het voorkomt.
    2. Train BPE op die stukken, met elk paar zo vaak geteld als zijn stuk voorkomt: dezelfde regels als je `train_pre_bpe`.
    3. Encodeer de hele tekst met elk verschillend stuk één keer geëncodeerd, en end-of-text als id `256 + len(merges)`.
    4. Train je GPT op de ids, met een vocabulary van `257 + len(merges)`, en vergelijk zijn loss per teken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 op de eerste 20.000 tekens van de verhalen: hoeveel stukken, hoeveel verschillende, en de meest voorkomende. Dan de controle: de tellingen van de paren uit elk verschillend stuk één keer, elk met zijn telling, zijn die uit elk stuk.
    """)
    return


@app.cell
def example_counts():
    import re

    _text = C.text()[:20000]
    _pieces = []
    for _part in _text.split("<|endoftext|>"):
        for _piece in re.findall(C.PATTERN, _part):
            _pieces.append(_piece)
    # 1. elk verschillend stuk één keer, met hoe vaak het voorkomt
    _counts_of = {}
    for _piece in _pieces:
        if _piece in _counts_of:
            _counts_of[_piece] = _counts_of[_piece] + 1
        else:
            _counts_of[_piece] = 1
    print(len(_pieces), "stukken,", len(_counts_of), "verschillende")
    # repr laat een stuk tussen aanhalingstekens zien, zodat je een spatie aan het begin ziet
    for _piece in sorted(_counts_of, key=_counts_of.get, reverse=True)[:5]:
        print(repr(_piece), _counts_of[_piece])
    # de paren van elk stuk, en de paren van elk verschillend stuk, zo vaak geteld als het voorkomt
    _every = {}
    for _piece in _pieces:
        _ids = list(_piece.encode("utf-8"))
        for _i in range(len(_ids) - 1):
            _pair = (_ids[_i], _ids[_i + 1])
            if _pair in _every:
                _every[_pair] = _every[_pair] + 1
            else:
                _every[_pair] = 1
    _once = {}
    for _piece in _counts_of:
        _ids = list(_piece.encode("utf-8"))
        for _i in range(len(_ids) - 1):
            _pair = (_ids[_i], _ids[_i + 1])
            if _pair in _once:
                _once[_pair] = _once[_pair] + _counts_of[_piece]
            else:
                _once[_pair] = _counts_of[_piece]
    print(_every == _once)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(f"""
    Op 20.000 tekens zijn {C.grouped(C.FIRST_PIECES['pieces'], L.lang)} stukken maar {C.grouped(C.FIRST_PIECES['different'], L.lang)} verschillende, en de punten, de komma's en \" and\" komen het meest voor. De twee dicts met tellingen van paren zijn gelijk: `True`. Op alle verhalen is het verschil veel groter: {C.grouped(C.PIECES, L.lang)} stukken, {C.grouped(C.DISTINCT, L.lang)} verschillende. `sorted(..., key=_counts_of.get, reverse=True)` zet de stukken op volgorde van hun telling, de grootste eerst; in de oefeningen heb je het niet nodig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bpe():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. BPE op elk verschillend stuk één keer

        Maak `train_bpe_fast(text, n)` af: stap 1 en 2 van het recept. De rondes zijn die van je `train_pre_bpe`, met `words` en `weights` in plaats van de stukken: `words[k]` heeft de ids van een verschillend stuk, en `weights[k]` hoe vaak het voorkomt. De check roept `train_bpe_fast("the the the cat cow cup", 2)` aan, en andere teksten, ook een van 300.000 tekens, die binnen 2 seconden klaar moet zijn.
        """),
        L.task("Laat `train_bpe_fast` elk verschillend stuk één keer tellen, en elk paar zo vaak als zijn stuk voorkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bpe():
    box_bpe = L.code_box("train-bpe-fast", lines=36)
    box_bpe
    return (box_bpe,)


@app.cell(hide_code=True)
def check_bpe(box_bpe):
    result_bpe = L.check("train-bpe-fast", box_bpe.value)
    result_bpe
    return (result_bpe,)


@app.cell(hide_code=True)
def hints_bpe():
    L.hints(
        "train-bpe-fast",
        [
            "Stap 1 is de eerste loop van het uitgewerkte voorbeeld, binnen een loop over de delen van `text.split(END)`.",
            "Voor elk stuk van `re.findall(PATTERN, part)`: één meer in `counts_of`, of 1 voor een stuk dat je voor het eerst tegenkomt. In de rondes telt een paar `weights[k]` op bij zijn telling.",
        ],
        why="Een paar in een stuk komt even vaak voor als het stuk zelf, dus de telling van het stuk optellen geeft dezelfde telling als langs elk stuk van de tekst gaan. De dicts houden de volgorde waarin stukken en paren voor het eerst verschijnen, dus zelfs een gelijkspel tussen twee paren eindigt zoals in je `train_pre_bpe`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_encode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Encodeer elk verschillend stuk één keer

        Maak `encode_fast(text, merges)` af: stap 3 van het recept. De functie geeft dezelfde ids als je `encode` uit [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl), en gebruikt je `encode_part` van daar. De check roept `encode_fast("hi hi<|endoftext|>hi", {(104, 105): 256})` aan, en andere teksten, ook lange met 300 regels, die binnen 2 seconden klaar moeten zijn.
        """),
        L.task("Laat `encode_fast` elk verschillend stuk één keer encoderen, en zijn ids uit `known` halen elke keer dat het voorkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_encode():
    box_encode = L.code_box("encode-fast", lines=18)
    box_encode
    return (box_encode,)


@app.cell(hide_code=True)
def check_encode(box_encode):
    result_encode = L.check("encode-fast", box_encode.value)
    result_encode
    return (result_encode,)


@app.cell(hide_code=True)
def hints_encode():
    L.hints(
        "encode-fast",
        [
            "Twee delen: een stuk dat je voor het eerst tegenkomt, gaat in `known`; dan gaan, elke keer, zijn ids in `ids`.",
            "`if piece not in known:` met `encode_part(piece, merges)`; na de `if` een loop over `known[piece]`.",
        ],
        why="Elk verschillend stuk gaat één keer door alle merge rules, en zijn ids staan vanaf dan in `known`. `append` voegt één id toe aan het eind van dezelfde lijst, dus 1,5 miljoen ids kosten een ogenblik, geen uur.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_bpe, result_encode):
    result_bpe, result_encode  # the note changes when exercise 1 or 2 saves your module
    _p = C.plan()
    _d, _h, _layers, _block = _p["sizes"]
    if _p["device"] == "cpu":
        _where = "Deze computer heeft geen GPU waar de grotere GPT op past" + (f" ({_p['small']} heeft te weinig geheugen)" if _p["small"] else "") + ", dus draait er een kleinere versie op de CPU"
        _time = f" Op de computer van de course, een CPU met 4 cores, duurde de run ongeveer {round(C.REFERENCE['cpu'][0]['minutes'])} minuten, op één thread."
    else:
        _where = f"Je computer heeft een GPU, {_p['name']}, dus je GPT traint daarop"
        _time = ""
    mo.vstack([
        mo.md(f"""
        ### Wat je project nu kan

        De knop hieronder traint je tokenizer, je `train_bpe_fast`, met {C.grouped(C.MERGES, L.lang)} regels op de eerste 90% van de verhalen, encodeert ze allemaal met je `encode_fast`, en traint je GPT op hun tokens, met je `gpt_loaders` en je `Trainer`, zoals in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl). {_where}: een GPT van {C.grouped(_p['params'], L.lang)} parameters, met $d = {_d}$, {_h} heads, {_layers} blocks en vensters van {_block} tokens, {_p['epochs']} epochs met mini-batches van {_p['batch']}. Daarna schrijft hij een verhaal dat begint met "{C.START}".{_time} Tijdens het trainen laat de cel na elke epoch de loss zien en de tijd die nog over is, en wacht de rest van de les. De stopknop van de cel beëindigt de run eerder; de beste parameters tot dan blijven bewaard, in `work/projects/llm/gpt_bpe.pt`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je tokenizer en je GPT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_bpe, result_encode):
    result_bpe, result_encode  # run again when exercise 1 or 2 saves your module
    _labels = {"title": "Jouw run", "x": "epoch", "y": "loss per token", "lang": L.lang, "guess": "gelijk gokken", "train": "training", "val": "validatie"}

    def _progress(_p):
        if _p["phase"] == "tokenizer":
            mo.output.replace(mo.md("**Je tokenizer traint** op de eerste 90% van de verhalen…"))
            return
        if _p["phase"] == "encode":
            mo.output.replace(mo.md(f"**De verhalen worden geëncodeerd** met je `encode_fast` · je tokenizer deed er {C.number(_p['seconds_bpe'], 0, L.lang)} seconden over"))
            return
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · nog ongeveer {_m} {'minuten' if _m > 1 else 'minuut'} te gaan"
        _parts = [mo.md(f"**Epoch {_p['epoch']} van {_p['epochs']}** · stap {_p['step']} van {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.history_picture(_p["history"], {**_labels, "alt": "De training loss en validation loss van je run tot nu toe, na elke epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    if train_button.value:
        _result = C.train(_progress)
    else:
        _result = C.last_run()
    _ready = result_bpe.state == "correct" and result_encode.state == "correct"
    if _result is None and _ready and not train_button.value:
        _out = mo.md("*Druk op de knop om de run te starten.*")
    elif _result is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af, en druk dan op de knop: de run gebruikt je `train_bpe_fast` en je `encode_fast`.*")
    if _result is None and C.trained()["source"] == "course":
        _c = C.trained()
        _out = mo.vstack([_out, mo.md(f"""Tot dan de eigen GPT op tokens van de course: dezelfde GPT en tokenizer als in het CPU-plan, door de course getraind op {C.grouped(_c['chars'], L.lang)} tekens aan verhalen uit Cosmopedia, waarvan de licentie de course een model laat delen dat erop getraind is, en niet op TinyStories. Op deze verhalen, moeilijker dan TinyStories, is zijn validation loss {C.number(_c['best'], 2, L.lang)} per token, {C.number(_c['best_per_char'], 3, L.lang)} per teken. Hij schreef, met temperatuur 0,8:

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
        _chars = C.CHARS[_plan] if _plan in C.CHARS else C.CHARS["cpu"]
        _on = {"cpu": "de CPU", "mps": "de GPU van Apple"}.get(_result["device"], _result["name"])
        _ratio = _result["val_chars"] / _result["val_tokens"]
        _row = "ongeveer evenveel stappen" if _plan in C.CHARS else "het CPU-plan"
        _steps = "ongeveer evenveel stappen als je run met het CPU-plan" if _plan in C.CHARS else "ongeveer evenveel stappen als een run van deze les met het CPU-plan, niet als je run met het GPU-plan, met zijn grotere GPT"
        _limit = C.number(C.THRESHOLD[_plan], 2, L.lang)
        if _result["stopped"]:
            _verdict = f"Je stopte de run na epoch {len(_h)} van {_result['epochs']}; de runs van de course met deze GPT eindigden, met alle {_result['epochs']} epochs, onder {_limit} per token."
        elif _result["best"] <= C.THRESHOLD[_plan]:
            _verdict = f"De runs van de course met deze GPT eindigden onder {_limit} per token, en de jouwe ook: je tokenizer en je GPT leerden wat ze moesten leren."
        else:
            _verdict = f"De runs van de course met deze GPT eindigden onder {_limit} per token, en de jouwe niet. Bekijk oefening 1 en 2 nog eens, en die van de lessen waar je GPT en Trainer vandaan komen."
        _saved = "" if not _result.get("kept") else f" Je eerdere run had een lagere validation loss, dus `work/projects/llm/gpt_bpe.pt` houdt de parameters van die run."
        _who = "" if not _result["stand_ins"] else " De versie van de course viel in voor " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + "."
        _story = "" if not _result.get("story") else "\n\nHet verhaal dat je GPT schreef, met temperatuur 0,8:\n\n> " + _result["story"].replace("\n", "\n> ")
        _out = mo.vstack([
            mo.Html(C.history_picture(_h, {**_labels, "alt": f"De training loss en validation loss van je run na elke epoch: de validation loss is het laagst na epoch {_result['best_epoch']}, met {C.number(_result['best'], 2, L.lang)} per token."}, _result["epochs"])),
            mo.md(f"""Je GPT van {C.grouped(_result['params'], L.lang)} parameters trainde {len(_h)} {'epochs' if len(_h) > 1 else 'epoch'} op {_on}, in {C.number(_result['minutes'], 1, L.lang)} minuten, tokenizer en encoderen meegerekend. De verhalen werden {C.grouped(_result['tokens'], L.lang)} tokens, {C.number(_ratio, 2, L.lang)} tekens per token op het deel waarop hij valideert. Zijn validation loss daalde tot **{C.number(_result['best'], 2, L.lang)} per token** na epoch {_result['best_epoch']}: **{C.number(_result['best_per_char'], 3, L.lang)} per teken**.{_saved} {_verdict}

| | loss per teken |
|---|---|
| je GPT op tokens | {C.number(_result['best_per_char'], 3, L.lang)} |
| de GPT van de course op tekens, dezelfde verhalen, {_row} | {C.number(_chars['best'], 3, L.lang)} |

De GPT op tekens is de run van de course met het CPU-plan van [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl) op deze verhalen, {_chars['epochs']} epochs lang: {_steps}.{_who}{_story}

De GPT van de course op tekens schreef, met hetzelfde begin:

> {_chars['story'].replace(chr(10), chr(10) + '> ')}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_char():
    mo.md(r"""
    ### Oefening 3. Per token of per teken
    """)
    return


@app.cell(hide_code=True)
def choice_char_cell():
    _r = C.REFERENCE["cpu"][0]
    choice_char = L.choice(
        "per-char",
        question=f"In de run van de course met de demo hierboven eindigt de GPT op tokens met een validation loss van {C.number(_r['best'], 2, L.lang)} per token; zijn tokens bevatten gemiddeld {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} tekens. De GPT van de course op tekens eindigt op dezelfde verhalen met {C.number(C.CHARS['cpu']['best'], 2, L.lang)} per teken. Welke voorspelt de verhalen beter?",
        options={
            "chars": f"De GPT op tekens: {C.number(C.CHARS['cpu']['best'], 2, L.lang)} is minder dan {C.number(_r['best'], 2, L.lang)}",
            "right": f"De GPT op tokens: {C.number(_r['best'], 2, L.lang)} per token is {C.number(_r['best_per_char'], 2, L.lang)} per teken",
            "same": "Je kunt ze niet vergelijken: ze hebben een andere vocabulary",
        },
        correct="right",
        why={
            "chars": f"Een token bevat {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} tekens, dus één loss per token verdeel je over zoveel tekens: {C.number(_r['best'], 2, L.lang)} ÷ {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} = {C.number(_r['best_per_char'], 2, L.lang)} per teken.",
            "same": "Per teken kan het wel: beide geven dezelfde verhalen een kans, dus de loss van de hele tekst is te vergelijken, verdeeld over zijn tekens. Verdeel de loss per token over de tekens die een token bevat.",
            "right": f"{C.number(_r['best'], 2, L.lang)} ÷ {C.number(_r['val_chars'] / _r['val_tokens'], 2, L.lang)} = {C.number(_r['best_per_char'], 2, L.lang)} per teken, tegen {C.number(C.CHARS['cpu']['best'], 2, L.lang)}. Met ongeveer evenveel stappen las de GPT op tokens ongeveer {C.number(_r['val_chars'] / _r['val_tokens'], 1, L.lang)} keer zoveel verhaal.",
        },
    )
    choice_char
    return (choice_char,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je tokenizer uit [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl), [Pre-tokenization](/course/open/tok-pre-tokenization?lang=nl) en [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl), nu snel genoeg voor megabytes; je GPT en run uit [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl), nu op zijn tokens; en de temperatuur uit [Tekst genereren met je GPT](/course/open/gpt-generate?lang=nl) voor zijn verhaal. Je project `llm` heeft nu `fast_bpe.py` en `fast_encode.py`, en na de run een GPT op tokens in `gpt_bpe.pt`, met zijn merge rules in `gpt_bpe.json`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_bpe, result_encode, choice_char):
    L.wrap_up(result_bpe, result_encode, choice_char)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
