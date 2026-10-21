# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-reasoning-from-scratch. Route: course/route/lessen/taal.yaml. Checks en de runs van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-reasoning-from-scratch", "nl")
    C = L.checks
    M = C.MEASURED["cpu"]   # de drie seeds van de course op het CPU-plan: goed van 200 per run en lengte
    K = C.course_result("cpu")   # de runs van de course met seed 0 op het CPU-plan


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl) en [Hoe een LLM redeneert: tussenstappen en test-time compute](/course/open/llm-reasoning-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    _quick_mask = L.choice(
        "quick-mask",
        question="Bij SFT zijn de labels van de tokens van de instructie -100. Waarom?",
        options={
            "right": "Zodat de loss alleen het antwoord telt",
            "hidden": "Zodat het model de instructie nooit ziet",
            "padding": "Om aan te geven waar de padding begint",
        },
        correct="right",
        why={
            "right": "`F.cross_entropy` slaat elk label van -100 over: het model leest de instructie, en leert alleen het antwoord schrijven. Vandaag is de prompt een vermenigvuldiging.",
            "hidden": "Het model leest de instructie wel: die is zijn input. Alleen de loss laat die weg. Zie [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).",
            "padding": "De padding van de doelen krijgt ook -100, maar de instructie krijgt het om een andere reden: de loss moet alleen het antwoord tellen. Zie [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).",
        },
    )
    _quick_mask
    return


@app.cell(hide_code=True)
def quick_check_steps():
    _quick_steps = L.choice(
        "quick-steps",
        question="Waarom kunnen tussenstappen een model met een vaste diepte helpen om te antwoorden?",
        options={
            "right": "Elk token dat het schrijft is één forward pass meer, en elke stap kan een probleem zijn dat één pass aankan",
            "weights": "Het schrijven van de stappen verandert zijn weights terwijl het antwoordt",
            "copy": "De stappen bevatten het antwoord, dus het model hoeft het alleen over te nemen",
        },
        correct="right",
        why={
            "right": "Meer rekenwerk voor dezelfde vraag, in kleine stukjes. Vandaag test je dat op een model waarvan je elk trainingsvoorbeeld zelf kiest.",
            "weights": "Terwijl een model antwoordt, verandert er geen weight: de stappen zijn tekst die het schrijft en daarna leest. Zie [Hoe een LLM redeneert](/course/open/llm-reasoning-idea?lang=nl).",
            "copy": "Het model schrijft de stappen zelf; ze bevatten het antwoord pas als het dat heeft uitgerekend. Zie [Hoe een LLM redeneert](/course/open/llm-reasoning-idea?lang=nl).",
        },
    )
    _quick_steps
    return


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Qwen3 antwoordde beter met tussenstappen, maar het is gepretraind op biljoenen tokens, waarschijnlijk met uitgewerkte oplossingen ertussen: je kunt niet zien wat de stappen daar deden. Test het dus waar je elk voorbeeld zelf kiest: train hetzelfde kleine model twee keer van nul op dezelfde vermenigvuldigingen, één keer op alleen het antwoord en één keer met de tussenstappen uitgeschreven. Alleen de vorm van de data verschilt: een eerlijke vergelijking, zoals in [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl), met alleen het antwoord als control run.

        Het model is nieuw en klein, gebouwd uit alleen onderdelen van je project: ModernGPT, `sft_example`, `sft_batch` en `pretrain_step`. Je verhalenmodel blijft zoals het is.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen",
                            "Extra: een klein model dat zijn tussenstappen schrijft"],
                           7, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_formats():
    _s, _p = M["steps-1800"]["3"], M["plain-1800"]["3"]
    _guess_formats = L.guess(
        "formats",
        question=f"Hetzelfde model, dezelfde {C.number(C.RUNS['cpu'][0][1], 0, 'nl')} stappen van {C.BATCH} paren, op dezelfde manier getrokken. Neem {C.HELD} producten van twee getallen van 3 cijfers die geen van beide ooit zag: hoe vaak heeft elk model het goed?",
        options={
            "none": "Allebei bijna nooit: producten van 3 cijfers zijn te moeilijk voor zo'n klein model",
            "both": "Allebei ongeveer even vaak: ze zagen dezelfde paren",
            "steps": "Met tussenstappen de meeste, met alleen het antwoord bijna geen",
        },
        reveal=f"**Met tussenstappen {C.listed(_s, 'nl')} van {C.HELD}; met alleen het antwoord {C.listed(_p, 'nl')}**, in de runs van de course met seeds 0, 1 en 2. Hetzelfde model en dezelfde paren: alleen de vorm van de antwoorden veranderde.",
    )
    _guess_formats
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(rf"""
        ## Het idee

        **Trainen op uitgeschreven tussenstappen.** De antwoorden in de training data bevatten de stappen van hun berekening, zodat het model leert die vóór de uitkomst te schrijven: een chain of thought, als training data. Elke stap is een klein probleem, en elk token dat het model schrijft is een forward pass die de stappen ervoor leest.

        **De twee vormen.** De tokens zijn tekens: de tien cijfers, `*`, `+`, `=`, `,` en `>`, en het end-of-text, {C.SIZES[0]} in totaal. De prompt is `17*21=`, de loss telt alleen het antwoord (loss masking, zoals bij SFT), en het antwoord eindigt met het end-of-text.

        - alleen het antwoord: `357`
        - met tussenstappen: `17*1=17,17*2=34>17+340=357>357`. Eerst de deelproducten, 17 keer elk cijfer van 21, van rechts af; dan `>` en de lopende sommen, met de plek van elk cijfer: 17 + 340; dan `>` en het product. Met een `b` van één cijfer zijn er geen sommen: `7*1=7>>7`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **trainen op uitgeschreven tussenstappen** | trainen op antwoorden die de stappen van hun berekening bevatten, zodat het model leert die vóór de uitkomst te schrijven |
        | **length generalization** | of een model dat getraind is op invoer tot een bepaalde lengte, ook langere invoer goed doet |
        """),
        L.watch_out("*Schrijft zijn tussenstappen* is niet *redeneert*. Dit model leert één berekening in één vorm; een reasoning model, zoals o1 van OpenAI (stand oktober 2026), schrijft lange chains of thought over veel soorten problemen."),
    ])
    return


@app.cell(hide_code=True)
def text_play_formats():
    mo.md(r"""
    **Beide vormen, token voor token.** Grijs is de prompt, die het model krijgt; blauw is wat het leert schrijven, waar de loss telt. De schuif kiest een paar, van 1 × 1 cijfer tot 4 × 4.
    """)
    return


@app.cell(hide_code=True)
def play_formats():
    _labels = {"title": "{a} × {b}: de tokens van elke vorm", "plain": "alleen het antwoord", "steps": "met tussenstappen",
               "tokens": "{n} tokens in het antwoord", "end": "einde", "lang": "nl",
               "said": "**{a} × {b} = {product}** · alleen het antwoord {plain} tokens, met tussenstappen {steps}, het end-of-text meegeteld; {parts}.",
               "part": "{n} deelproduct", "parts": "{n} deelproducten",
               "alt": "De prompt van het gekozen paar in grijs, dan zijn antwoord in blauw, één keer met alleen het product en één keer met zijn tussenstappen, elk met het aantal tokens van het antwoord."}
    _play = L.explore("formats", "voorbeeld", lambda _v: C.formats_frame(int(_v), _labels), start=2, values=list(range(1, len(C.PAIRS) + 1)),
                      role="", task="Hoeveel tokens heeft het antwoord van elke vorm nodig bij 3 × 3? En hoeveel deelproducten heeft een product van 4 × 4 nodig, wat de training nooit laat zien?")
    _play
    return


@app.cell(hide_code=True)
def text_play_formats_after():
    _a, _b = C.PAIRS[4]
    _n = C.tokens_frame(5, {"title": "", "plain": "", "steps": "", "tokens": "", "end": "", "alt": ""})[1]
    mo.md(rf"""
    Bij 3 × 3 kost {_a} × {_b} met tussenstappen {_n['steps']} tokens in het antwoord tegen {_n['plain']}: ongeveer {round(_n['steps'] / _n['plain'])} keer zoveel, bij het trainen en in elk antwoord. Een product van 4 × 4 heeft vier deelproducten en drie sommen nodig; de training laat er hooguit drie en twee zien.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(rf"""
    ### Recept

    1. Trek paren van getallen van 1, 2 of 3 cijfers elk; houd {C.HELD} paren van elke lengte buiten de training: 2 × 2, 3 × 3, en 4 × 4, langer dan alles in de training.
    2. Schrijf elk paar als zijn prompt en zijn antwoord, in één vorm.
    3. Maak er ids en labels van met `sft_example`, met de prompt gemaskeerd, en batches met `sft_batch`.
    4. Train hetzelfde model op elke vorm met dezelfde stappen, batch, learning rate en seed.
    5. Laat elk model de achtergehouden paren greedy beantwoorden, lees zijn product met een programma, en tel.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 en 3 voor 12 × 34: het antwoord met zijn tussenstappen, dan zijn ids en labels. `C.encode` geeft het id van elk teken, zoals in [Project: je GPT op een boek trainen](/course/open/gpt-train-chars?lang=nl): de tekens gesorteerd, teken i heeft id i. Voer de cel uit.
    """)
    return


@app.cell
def example_twelve():
    _a, _b = 12, 34
    # 2. de cijfers van b van rechts af, een deelproduct voor elk, en de lopende som
    _digits = [ord(_c) - ord("0") for _c in reversed(str(_b))]
    _parts = [f"{_a}*{_d}={_a * _d}" for _d in _digits]
    _first = _a * _digits[0]
    _add = _a * _digits[1] * 10 ** 1
    _answer = ",".join(_parts) + ">" + f"{_first}+{_add}={_first + _add}" + ">" + f"{_a * _b}"
    print(_digits, _parts)
    print(_answer)
    # 3. de ids, en een label voor elk id: -100 voor de prompt, dan het antwoord en het end-of-text
    _prompt = C.encode(f"{_a}*{_b}=")
    _ids = _prompt + C.encode(_answer) + [C.END]
    _labels = [-100] * len(_prompt) + C.encode(_answer) + [C.END]
    print(_ids[:10], "...", _ids[-3:], len(_ids))
    print(_labels[:10], "...", _labels[-3:], len(_labels))
    return


@app.cell(hide_code=True)
def text_after_example():
    _answer = C.course_functions()[0](12, 34)
    mo.md(rf"""
    De 3 van 34 staat voor 30, dus 12 × 3 = 36 telt als 360: 48 + 360 = 408. De prompt krijgt zes labels van -100, de {len(_answer)} tokens van het antwoord en het end-of-text, id {C.END}, hun eigen ids: {len(_answer) + 7} ids en evenveel labels. In de demo maakt je `sft_example` deze lijsten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_answer():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het antwoord met zijn tussenstappen

        Maak `steps_answer(a, b)` af: stap 2 voor de vorm met tussenstappen, zonder het end-of-text. Het vak begint met de cijfers van `b` van rechts af, als getallen: `ord(c) - ord("0")`, zoals in [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl). De check roept `steps_answer(17, 21)` aan, wat `"17*1=17,17*2=34>17+340=357>357"` moet geven, en probeert één cijfer, nullen en vier cijfers. Is hij goed, dan wordt hij bewaard als `llm/steps_answer.py`.
        """),
        L.task("Maak `steps_answer` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_answer():
    box_answer = L.code_box("steps-answer", lines=17)
    box_answer
    return (box_answer,)


@app.cell(hide_code=True)
def check_answer(box_answer):
    result_answer = L.check("steps-answer", box_answer.value)
    result_answer
    return (result_answer,)


@app.cell(hide_code=True)
def hints_answer():
    L.hints(
        "steps-answer",
        ["Eén deelproduct per cijfer: `parts.append(f\"{a}*{d}={a * d}\")`. Het cijfer op plek `i` telt `10 ** i` keer: schrijf de som van `total` en `a * digits[i] * 10 ** i`, en tel die daarna bij `total` op."],
        why="Elke stap is er een die één forward pass aankan: een getal keer één cijfer, of één optelling. Het product komt als laatste, na de tweede `>`, waar de checker van oefening 2 het leest.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een programma dat het product leest

        Maak `read_product(text)` af, de checker van stap 5: het deel na de laatste `>`, of de hele tekst als die er geen heeft, als het alleen uit cijfers bestaat (`.isdigit()`), en anders `None`. Het product blijft tekst: de demo vergelijkt het met `f"{a * b}"`. De check leest ook de lege tekst en antwoorden die halverwege stoppen. Is hij goed, dan wordt hij bewaard als `llm/read_product.py`.
        """),
        L.task("Maak `read_product` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read-product", lines=6)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read-product", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read-product",
        ["`text.split(\">\")[-1]` is het laatste deel, ook als de tekst geen `>` heeft; `\"\".isdigit()` is `False`."],
        why="Een model dat nog leert, schrijft kapotte antwoorden. Een checker die daar `None` voor zegt, telt ze als fout, en nooit als product.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _s, _p = C.per_step("cpu", "steps", 1800), C.per_step("cpu", "plain", 1800)
    mo.vstack([
        mo.md(rf"""
        ### Train beide modellen

        De knop traint een nieuwe ModernGPT van {C.number(C.PARAMS, 0, 'nl')} parameters (d = {C.SIZES[1]}, {C.SIZES[4]} blocks, {C.SIZES[2]} query heads, {C.SIZES[3]} key en value heads) op elke vorm om de beurt: {C.number(C.RUNS['cpu'][0][1], 0, 'nl')} stappen van {C.BATCH} paren, getrokken met seed 0 en nooit een achtergehouden paar, met je `steps_answer`, `sft_example`, `sft_batch` en `pretrain_step`; AdamW met {C.number(C.LR, 3, 'nl')}, weight decay {C.number(C.WEIGHT_DECAY, 1, 'nl')}, met de warmup en warmdown uit [Een open trainingsrecept lezen](/course/open/llm-open-recipes?lang=nl). Daarna beantwoordt elk model de {len(C.TESTS) * C.HELD} achtergehouden paren greedy, en leest je `read_product` elk antwoord.

        Op de CPU van de course met {C.threads('cpu')} threads duurden beide runs samen {C.number(C.plan_minutes('cpu'), 0, 'nl')} minuten, {C.number(_s, 2, 'nl')} s per stap met tussenstappen en {C.number(_p, 2, 'nl')} s zonder. Op een GPU traint de demo ook nog alleen het antwoord met {C.number(C.RUNS['gpu'][2][1], 0, 'nl')} stappen, vier keer zoveel, ongeveer de tijd die de run met tussenstappen op die CPU kost; de course nam die run ook op haar CPU op, met {C.threads('gpu')} threads, niet op een GPU. Een proefmeting van {C.PROBE} stappen van elke vorm zegt eerst hoe lang het op de jouwe duurt; zonder die tijd nemen de runs en modellen van de course het over.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def probe_button_cell():
    probe_button = mo.ui.run_button(label=f"Meet {C.PROBE} stappen van elke vorm")
    probe_button
    return (probe_button,)


@app.cell(hide_code=True)
def demo_probe(box_answer, probe_button, result_answer):
    _note = None
    if not probe_button.value:
        _note = None
    elif result_answer.state != "correct":
        _note = "*Maak eerst `steps_answer` goed in oefening 1: de proefmeting traint ermee.*"
    else:
        mo.output.replace(mo.md("**Meten** …"))
        _t = C.tools()
        _p = "run" if _t == "run" else C.probe(run_code(box_answer.value).names["steps_answer"], _t)
        if isinstance(_p, str):
            _note = "*De proefmeting stopt met een fout. Halen je modules in `work/projects/llm/` hun checks nog? Verwijder een bestand dat je hebt veranderd, en check de oefening ervan opnieuw.*"
        else:
            _long = " Dat is lang: de runs van de course hieronder nemen het over, en haar modellen antwoorden bij *Vraag het beide modellen*. Druk op *Train beide modellen* als je de tijd hebt." if _p["minutes"] > 20 else ""
            _note = f"Op deze computer duurde een stap {C.number(_p['seconds']['steps'], 2, 'nl')} s met tussenstappen en {C.number(_p['seconds']['plain'], 2, 'nl')} s met alleen het antwoord: alle runs van het {_p['kind'].upper()}-plan duren ongeveer **{C.number(_p['minutes'], 0, 'nl')} minuten**.{_long}"
    mo.md(_note) if _note else None
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train beide modellen")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_answer, box_read, result_answer, result_read, train_button):
    trained = None
    if result_answer.state != "correct" or result_read.state != "correct":
        _out = mo.md("*Maak eerst `steps_answer` en `read_product` goed in oefening 1 en 2: de demo traint met de eerste en telt met de tweede.*")
    elif not train_button.value:
        _out = mo.md("*Druk op de knop om beide modellen te trainen.*")
    else:
        _plan = C.where()["kind"]
        _names = {"steps": "Met tussenstappen", "plain": "Alleen het antwoord"}
        _lives = {}

        def _show(_d):
            _key = (_d.get("format"), _d.get("steps"))
            _name = f"{_names[_key[0]]}, {C.number(_key[1], 0, 'nl')} stappen" if _key[0] else ""
            if _d["phase"] == "start":
                _live = L.live({"title": f"{_names[_key[0]]}: loss per stap", "x": "stap", "y": "loss",
                                "alt": f"De loss van elke stap tot nu toe van je run, {_name.lower()}, in grijs, met zijn gemiddelde als de dikkere lijn, en de run van de course op hetzelfde plan gestippeld."},
                               _key[1], low=0.0)
                _live.reference(C.live_reference(_plan, *_key), "", role="loss")
                _live.above = [mo.Html(_l.picture()) for _l in _lives.values()]
                _lives[_key] = _live
                _live.line(f"**{_name}** · run {_d['i']} van {_d['n']} …")
            elif _d["phase"] == "train":
                _lives[_key].extend(_d["fresh"], f"**{_name}** · stap {C.number(_d['step'], 0, 'nl')} van {C.number(_d['steps'], 0, 'nl')} · loss {C.number(_d['loss'], 3, 'nl')} · {C.number(_d['seconds'] / 60, 1, 'nl')} minuten · gestippeld: de run van de course")
            elif _d["phase"] == "measure":
                _lives[_key].line(f"**{_name}** · het model beantwoordt de {len(C.TESTS) * C.HELD} achtergehouden paren …")

        _r = C.run(run_code(box_answer.value).names["steps_answer"], run_code(box_read.value).names["read_product"], _show)
        _charts = [mo.Html(_l.picture()) for _l in _lives.values()]   # ze blijven staan na de run
        if isinstance(_r, str):
            _out = mo.vstack(_charts + [mo.md({
                "steps": "*Je `steps_answer` stopt met een fout op een paar van de training, of schrijft een teken dat geen token van het model is. Haalt hij zijn check nog?*",
                "example": "*Je `sft_example` stopt met een fout op een paar. Haalt hij zijn check nog in [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl)?*",
                "batch": "*Je `sft_batch` stopt met een fout op een batch paren. Haalt hij zijn check nog in [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl)?*",
                "read": "*Je `read_product` stopt met een fout op een antwoord van het model. Haalt hij zijn check nog?*",
                "run": "*De run stopt met een fout. Halen je modules in `work/projects/llm/` hun checks nog? Verwijder een bestand dat je hebt veranderd, en check de oefening ervan opnieuw.*",
                "memory": "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk nog eens op de knop.*",
                "diverged": "*De loss is geen getal meer: de training is ontspoord. Gebruikt je `pretrain_step` de learning rate die hij krijgt?*",
            }[_r])])
        else:
            trained = _r
            _out = mo.vstack(_charts)
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _mine = trained if trained is not None else C.saved()
    _plan = _mine["kind"] if _mine else "cpu"
    _names = {"steps": "met tussenstappen", "plain": "alleen het antwoord"}
    _rows = []
    for _i, (_fmt, _steps) in enumerate(C.RUNS[_plan]):
        _label = f"{_names[_fmt]}, {C.number(_steps, 0, 'nl')} stappen"
        _course = C.course_runs(_plan, _fmt, _steps)
        if _mine:
            _r = _mine["runs"][_i]
            _rows.append(f"| {_label}, de jouwe | " + " | ".join(f"**{_r['accuracy'][str(_n)]['right']}**" for _n in C.TESTS)
                         + f" | {C.number(_r['seconds'] / _r['steps'], 2, 'nl')} | {C.number(C.answer_tokens(_r), 0, 'nl')} |")
        _rows.append(f"| {_label}, de seeds van de course | " + " | ".join(C.listed([_c['accuracy'][str(_n)]['right'] for _c in _course], 'nl') for _n in C.TESTS)
                     + f" | {C.number(C.per_step(_plan, _fmt, _steps), 2, 'nl')} | {C.number(C.answer_tokens(_course[0]), 0, 'nl')} |")
    _table = "| run | 2 × 2 | 3 × 3 | 4 × 4 | s per stap | tokens per antwoord |\n|---|---|---|---|---|---|\n" + "\n".join(_rows)
    _stand = "" if _mine else "\n\n*De runs van de course nemen het over tot je je eigen traint.*"
    _verdict = ""
    if _mine:
        _r = _mine["runs"][0]
        _ok = all(_r["accuracy"][str(_n)]["right"] >= C.THRESHOLD[_plan][str(_n)] for _n in ("2", "3"))
        _verdict = (f"\n\nJe run met tussenstappen begon bij een loss van {C.number(_r['first'], 1, 'nl')}, dicht bij ln {C.SIZES[0]} = {C.number(C.START_LOSS, 1, 'nl')}, zoals een nieuw model hoort. "
                    f"De check vraagt er minstens {C.THRESHOLD[_plan]['2']} van {C.HELD} goed bij 2 × 2 en {C.THRESHOLD[_plan]['3']} bij 3 × 3: de laagste van de drie seeds van de course, {min(C.MEASURED[_plan]['steps-1800']['2'])} en {min(C.MEASURED[_plan]['steps-1800']['3'])}, min een marge voor een andere computer. "
                    + ("De jouwe haalt dat: je `steps_answer` en de modules van je project trainen zoals het hoort." if _ok else
                       "De jouwe haalt dat niet. Halen je modules in `work/projects/llm/` hun checks nog? Vergelijk dan je `steps_answer` met de oplossing onder oefening 1."))
    mo.md(f"Goed van {C.HELD} achtergehouden paren bij elke lengte, met de seconden per trainingsstap en de tokens die een antwoord kost, het end-of-text meegeteld:\n\n{_table}{_verdict}{_stand}")
    return


@app.cell(hide_code=True)
def text_after_demo():
    _s, _p = K["runs"][0], K["runs"][1]
    mo.md(rf"""
    **De runs lezen.** Met tussenstappen eindigt de loss veel lager, {C.number(_s['losses'][-1]['loss'], 3, 'nl')} tegen {C.number(_p['losses'][-1]['loss'], 2, 'nl')} in de runs van de course met seed 0, maar dat zegt weinig: de meeste tokens van een antwoord met tussenstappen volgen uit de tokens ervoor, dus hun loss is laag, ook als het product fout is. Vergelijk wat elk model goed heeft. En de tussenstappen kosten iets: ongeveer {C.number(_s['tokens'] / _p['tokens'], 0, 'nl')} keer de tokens om op te trainen voor evenveel paren, en ongeveer {C.number(C.answer_tokens(_s) / C.answer_tokens(_p), 0, 'nl')} keer de tokens voor elk antwoord, test-time compute die alleen het antwoord nooit besteedt.
    """)
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(rf"""
    ### Je run, stap voor stap

    Bij de start en na elk vijfde deel van elke run laat de demo beide modellen de eerste {C.SNAP} achtergehouden paren van 3 × 3 beantwoorden. Ga door die momenten: beide losses en de goede antwoorden tot dan, doorgetrokken met tussenstappen en gestippeld met alleen het antwoord, en wat elk model schreef bij {C.HELD_OUT[3][0][0]} × {C.HELD_OUT[3][0][1]}.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _info = trained if trained is not None else C.saved()
    _run = _info if _info is not None else K
    _whose = "je runs" if _info is not None else "de runs van de course"
    _labels = {"title_loss": "De loss langs beide runs", "title_right": f"Goed van {C.SNAP} paren van 3 × 3", "x": "stap",
               "y_loss": "loss", "y_right": "goed", "lang": "nl",
               "alt": f"Links de loss van {_whose}, doorgetrokken met tussenstappen en gestippeld met alleen het antwoord, met een gestippelde lijn op het gekozen moment. Rechts hoeveel van {C.SNAP} achtergehouden paren van 3 × 3 elk model goed had op elk moment tot dan, doorgetrokken met tussenstappen en gestippeld met alleen het antwoord.",
               "start": "Voor de eerste stap", "after": "Na {step} stappen", "right": "goed van {n}: met tussenstappen {steps}, alleen het antwoord {plain}",
               "steps": "Met tussenstappen", "plain": "Alleen het antwoord", "longer": "En drie van zijn antwoorden met tussenstappen op paren van 4 × 4:"}
    _steps = [_s["step"] for _s in C.pair_of(_run)[0]["snapshots"]]
    _block = L.explore("replay", "stap", lambda _v: C.replay_frame(_run, _steps.index(int(_v)), _labels), start=_steps[-1], values=_steps, role="",
                       task=f"Op welk moment heeft het model met tussenstappen voor het eerst producten goed? Welk deel van zijn antwoord klopt al voordat het product klopt?")
    mo.vstack([_block] if _info is not None else [mo.md("*Tot je je eigen modellen traint, ga je hier door de runs van de course met seed 0.*"), _block])
    return


@app.cell(hide_code=True)
def text_replay_after():
    _s, _p = K["runs"][0]["snapshots"], K["runs"][1]["snapshots"]
    _first = next(_x for _x in _s if _x["right"] > 0)
    mo.md(rf"""
    In de run van de course met seed 0 (je eigen run kan een beetje anders zijn) heeft het model met tussenstappen na {C.number(_first['step'], 0, 'nl')} stappen voor het eerst producten goed, {_first['right']} van {C.SNAP}, en aan het eind {_s[-1]['right']}. Het schrijft vanaf het eerste moment na de start de vorm van een antwoord, met verkeerde getallen: {C.shown(_s[1]['text'])}. Na {C.number(_s[3]['step'], 0, 'nl')} stappen kloppen zijn lopende sommen, terwijl 169 × 6 = {C.first_part(_s[3]['text'])} nog fout is, en het product dus ook; na {C.number(_s[4]['step'], 0, 'nl')} klopt het hele antwoord. Het model met alleen het antwoord heeft er op geen enkel moment een goed: na de start schrijft het een getal van ongeveer de goede lengte, {C.shown(_p[-1]['text'])} aan het eind.
    """)
    return


@app.cell(hide_code=True)
def text_ask():
    mo.md(r"""
    ### Vraag het beide modellen

    Typ twee getallen van hooguit 4 cijfers, zoals `169*186`. Beide modellen antwoorden greedy: de jouwe zodra je ze hebt getraind, anders die van de course.
    """)
    return


@app.cell(hide_code=True)
def ask_cell():
    ask_box = mo.ui.text(value="169*186", label="Je paar")
    ask_button = mo.ui.run_button(label="Vraag het beide modellen")
    mo.hstack([ask_box, ask_button], align="end", justify="start")
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_read, result_read, trained):
    _note = None
    if ask_button.value:
        _pair = C.asked(ask_box.value)
        if _pair is None:
            _note = "*Typ twee getallen van 1 tot 4 cijfers met een `*` ertussen, zoals `169*186`.*"
        else:
            _read = run_code(box_read.value).names["read_product"] if result_read.state == "correct" else C.course_functions()[1]
            _out = C.ask(_pair, _read, C.course_functions()[0])
            _whose = "je modellen" if _out[0][4] == "yours" else "de modellen van de course"
            _lines = [f"- {'Met tussenstappen' if _f == 'steps' else 'Alleen het antwoord'}: {C.shown(_text)} → " + ("geen product" if _got is None else f"{_got}, " + ("goed" if _ok else "fout"))
                      for _f, _text, _got, _ok, _ in _out]
            _note = f"{_pair[0]} × {_pair[1]} = {C.whole(_pair[0] * _pair[1], 'nl')}, door {_whose}:\n\n" + "\n".join(_lines)
    mo.md(_note) if _note else None
    return


@app.cell(hide_code=True)
def text_exercise_longer():
    mo.md(r"""
    ### Oefening 3. Langere getallen
    """)
    return


@app.cell(hide_code=True)
def explain_longer():
    explain_longer_cell = L.explain(
        "longer",
        question=f"Het model met tussenstappen heeft de meeste producten van 3 × 3 goed en geen van {C.HELD} van 4 × 4. Kijk wat het bij 4 × 4 schreef op het laatste moment van het terugspoelen. Waarom faalt het, terwijl elke soort stap die het schrijft er een is die het leerde?",
        model_answer="Het zag nooit vier deelproducten, en geen getal van vier cijfers: meestal schrijft het het plan van een product van 3 cijfers, drie deelproducten en hun sommen, en soms raakt het de vorm helemaal kwijt. Het leerde de vorm van zijn voorbeelden, geen methode voor elke lengte, en de plekken van een langer antwoord zijn nooit getraind. Kleine transformers die op rekensommen getraind zijn, kunnen zelden langere getallen aan dan in de training (Lee en anderen, 2023); langere getallen in de training, of een manier om aan te geven waar elk cijfer staat, helpen, maar niet betrouwbaar (McLeish en anderen, 2024; Zhou en anderen, 2024).",
        points=[
            "de training bepaalde de lengte: die liet nooit vier deelproducten of een getal van 4 cijfers zien",
            "het model schrijft de vorm die het leerde, drie deelproducten, geen algemene methode",
            "dit is length generalization, en dat is een open probleem voor kleine transformers",
        ],
    )
    explain_longer_cell
    return (explain_longer_cell,)


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Oefening 4. Is de vergelijking eerlijk?
    """)
    return


@app.cell(hide_code=True)
def explain_fair():
    _long = C.MEASURED["gpu"]["plain-7400"]["3"]
    explain_fair_cell = L.explain(
        "fair",
        question=f"Met dezelfde {C.number(C.RUNS['cpu'][0][1], 0, 'nl')} stappen zag het model op alleen het antwoord evenveel paren, maar veel minder tokens. In de runs van de course op het GPU-plan trainde het vier keer zoveel stappen, op ongeveer evenveel tokens als de run met tussenstappen, en had het nog steeds {C.listed(_long, 'nl')} van {C.HELD} producten van 3 × 3 goed. Is de vergelijking eerlijk? Wat zou je toevoegen om het zeker te weten?",
        model_answer="De vergelijking is eerlijk in het model, de data, de stappen en de seed, en vier keer zoveel stappen, op ongeveer evenveel tokens, laten zien dat de extra tokens niet het hele verhaal zijn. Ze is niet eerlijk in tuning effort: één learning rate voor beide, die het onderzoek voor deze les voor de vorm met tussenstappen koos. Ik zou hem per vorm afstellen, meer seeds draaien, en een groter model of een langere run proberen: GPT-2 Small haalde 4 × 4 zonder tussenstappen (Deng en anderen, 2024). En ik zou de kosten melden: meer trainingstijd per stap en veel meer tokens per antwoord.",
        points=[
            "één ding veranderde, de vorm; evenveel stappen en vier keer zoveel stappen allebei bekeken",
            "de learning rate is niet afgesteld voor alleen het antwoord: tuning effort",
            "meer seeds, en een groter model of een langere run",
            "de tussenstappen kosten trainingstijd en tijd bij het antwoorden",
        ],
    )
    explain_fair_cell
    return (explain_fair_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(rf"""
    ## Afronding

    **Samenhang.** [Hoe een LLM redeneert](/course/open/llm-reasoning-idea?lang=nl) vroeg een groot model om zijn tussenstappen; hier leerde een model ze uit zijn data, en zo komen chains of thought überhaupt in een model terecht. Je project `llm` heeft nu `steps_answer.py` en `read_product.py`, en na de demo beide modellen in `arith_steps.pt` en `arith_plain.pt`, met hun runs in `arith.json`.

    **En reinforcement learning?** De volgende stap zou [Reinforcement learning met verifiable rewards](/course/open/llm-rlvr?lang=nl) zijn, met je `read_product` als reward. De course probeerde het bij het onderzoek voor deze les, op een model met tussenstappen dat {C.number(C.RL['before'], 1, 'nl')}% van 3 × 3 goed had, en {C.RL['steps']} stappen GRPO hielpen niet: {C.number(C.RL['1e-4'], 1, 'nl')}% erna, bij een learning rate van 0,0001; {C.number(C.RL['3e-4'], 1, 'nl')}% bij 0,0003; en 0% bij 0,001. Toch zegt pass@8 van {C.RL['pass8']}%, tegen {C.number(C.RL['one'], 1, 'nl')}% voor één sample, dat er ruimte was. Hoe het wel werkt, is open; de course voegt de stap toe zodra het lukt.

    **Verder.** Stem over gesamplede antwoorden van je model met tussenstappen, zoals in *Hoe een LLM redeneert*. Train beide vormen op optellen, waar een klein model ook zonder tussenstappen ver komt. Of kijk elke stap `x*y=z` en `x+y=z` van een antwoord na, niet alleen het product, en zoek goede producten met een foute stap.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_answer, result_read, explain_longer_cell, explain_fair_cell):
    L.wrap_up(result_answer, result_read, explain_longer_cell, explain_fair_cell)
    return


@app.cell(hide_code=True)
def look_back():
    _s = M["steps-1800"]["3"]
    _look = L.choice(
        "look-seeds",
        question=f"### Terugblik\n\nDe course draaide elke vorm met drie seeds. Met tussenstappen hadden ze {C.listed(_s, 'nl')} van {C.HELD} producten van 3 × 3 goed. Wat vertellen drie seeds je dat één run niet vertelt?",
        options={
            "right": "Hoeveel de uitkomst verandert met alleen de seed, zodat je ziet dat het verschil tussen de vormen veel groter is",
            "exact": "De precieze accuracy van de methode, als het gemiddelde van de drie",
            "nothing": "Niets: dezelfde code geeft met elke seed hetzelfde getal",
        },
        correct="right",
        why={
            "right": "Een andere seed geeft andere beginwaarden van de weights en andere paren, en een ander getal. Een verschil tussen de vormen dat veel groter is dan die spreiding, is geen toeval.",
            "exact": "Drie runs geven een gemiddelde en een spreiding, geen precieze waarde. Zie [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
            "nothing": "De seed bepaalt de beginwaarden van de weights en de volgorde van de paren: de drie runs verschillen. Zie [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
        },
    )
    _look
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
