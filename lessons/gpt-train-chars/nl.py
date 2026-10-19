# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-train-chars. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-train-chars", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) en [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_window():
    quick_window = L.choice(
        "quick-window",
        question="Je GPT krijgt een venster van 256 ids, en als targets hetzelfde venster één id verder. Van hoeveel voorspellingen leert hij?",
        options={
            "one": "1: het id na het venster",
            "right": "256: één op elke plek van het venster",
            "fewer": "255: de laatste plek heeft geen target in het venster",
        },
        correct="right",
        why={
            "one": "Dat was je MLP uit [Een neural language model dat verder terugkijkt](/course/open/lm-mlp-context?lang=nl): één context, één target. Je GPT gokt op elke plek. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "fewer": "De targets zijn het venster één id verder, dus de laatste plek heeft ook een target: het id vlak na het venster. Zie [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).",
            "right": "256 voorspellingen, elk met een eigen target: vandaag geeft een mini-batch van 64 vensters er 16.384.",
        },
    )
    quick_window
    return (quick_window,)


@app.cell(hide_code=True)
def quick_check_copy():
    quick_copy = L.choice(
        "quick-copy",
        question="Een mini-batch `X` staat op de CPU, en je model op de GPU. Je schrijft `X.to(\"cuda\")` op een eigen regel, en daarna `model(X)`. Wat gebeurt er?",
        options={
            "moved": "Het werkt: `X` staat nu op de GPU",
            "right": "Een fout: `X` zelf staat nog op de CPU",
            "slow": "Het werkt, maar traag: PyTorch kopieert `X` bij elke aanroep naar de GPU",
        },
        correct="right",
        why={
            "moved": "`.to(device)` geeft een verplaatste kopie terug, en laat `X` staan waar hij stond. Bewaar de kopie: `X = X.to(device)`. Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "slow": "PyTorch verplaatst niets uit zichzelf, ook niet traag: het model en `X` staan op twee devices, en dat is een fout. Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "right": "Vandaag bewaart je `gpt_loaders` de verplaatste kopieën, in een `TensorDataset` op het device.",
        },
    )
    quick_copy
    return (quick_copy,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je GPT heeft al zijn delen, en weet nog niets: zijn loss is ongeveer 4,17, alsof hij gelijkmatig gokt uit 65 tekens. In deze projectstap leert hij van Tiny Shakespeare, ongeveer een miljoen tekens toneel. Op een GPU met genoeg geheugen is dat je GPT van 10,8 miljoen parameters uit [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl); op een computer zonder, een kleinere GPT van ongeveer een miljoen parameters, op de CPU. Daarna heeft je project een getrainde GPT, en in de volgende les laat je hem schrijven.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_loss():
    guess_loss = L.guess(
        "final-loss",
        question="Je MLP uit [Een neural language model dat verder terugkijkt](/course/open/lm-mlp-context?lang=nl) kwam na zijn 1.000 stappen tot een loss van 2,20 op tekst waarop hij niet getraind had. Waar eindigt de validation loss van je GPT na deze run, denk je?",
        options={
            "mlp": "Ongeveer 2,1: iets onder de MLP, want het is hetzelfde boek",
            "zero": "Bijna 0: dan kent hij het boek uit zijn hoofd",
            "right": "Duidelijk onder 2: hij kijkt veel verder terug dan de MLP",
        },
        reveal="**Duidelijk onder 2.** In de runs van de course met de kleinere GPT op een CPU, met 3 seeds, eindigde de validation loss op 1,66 tot 1,67; de GPT van 10,8 miljoen parameters kwam in een run van de course tot 1,52. Je MLP keek 3 tekens terug; je GPT kijkt 128 of 256 tekens terug, en leert zo woorden, namen en de vorm van een toneelstuk. Bijna 0 wordt het nooit: de tekst waarop hij valideert, is nieuw voor hem, en wat er in een toneelstuk volgt, is nooit zeker.",
    )
    guess_loss
    return (guess_loss,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **run** is één training van begin tot eind, met vaste instellingen en een vaste seed. Een run van je GPT heeft vijf stappen, en bijna elk stuk ervan heb je al geschreven:

        - **de tokens:** de tekens van het boek, gesorteerd, dus teken $i$ heeft id $i$: 65 stuks;
        - **de splitsing:** de eerste 90% van de ids om op te trainen, de laatste 10% om op te valideren. Niet geschud vóór het splitsen, anders dan in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl): vensters naast elkaar zijn één tekst, dus de vensters om op te valideren zijn een deel van het boek dat de GPT nooit leest, geen vensters tussen vensters waarop hij traint;
        - **de mini-batches:** vensters van `block` ids uit elk deel, met je `gpt_examples`, als tensors op het device. Alle vensters van het boek met hun targets zijn samen minder dan 20 MB aan ids, dus ze gaan in één keer naar het device, en tijdens het trainen hoeft geen mini-batch te verhuizen;
        - **het model:** je GPT op het device, en je `Trainer` uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl): AdamW, een warmup en cosine decay, clipping, en de parameters met de laagste validation loss bewaard;
        - **de run:** elke epoch vensters uit het hele boek, in een nieuwe volgorde; na elke epoch de loss op de vensters om op te valideren.

        In elke epoch na de eerste laat de demo de vensters op een andere plek in het boek beginnen, een willekeurig aantal tekens later. Met elke epoch dezelfde vensters leerde de grotere GPT ze uit zijn hoofd: in een run van de course ging zijn validation loss na 8 epochs weer omhoog. Dezelfde tekst, op andere plekken geknipt, geeft hem in elke epoch nieuwe voorbeelden.

        Hoeveel werk dat is, weet je uit [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl): je GPT van 10,8 miljoen parameters trainen op één token van een venster van 256 kost ongeveer 68 miljoen FLOPs, attention erbij. Een mini-batch van 64 vensters is 16.384 tokens, dus ongeveer $10^{12}$ FLOPs per stap, en 20 epochs van 62 stappen zijn ongeveer $1{,}4 \cdot 10^{15}$.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **run** | één training van begin tot eind, met vaste instellingen en een vaste seed |
        """),
        L.watch_out("`.to(device)` verplaatst de parameters van je GPT, niet de tensors die zijn `forward` onderweg maakt: de causal mask in je attention, gemaakt met `torch.ones(n, n)`, en de plekken in je GPT, `torch.arange(n)`. Die worden op de CPU gemaakt, en op een GPU stopt PyTorch dan met een fout. De demo hieronder draait je GPT binnen `with torch.device(device):`, dat nieuwe tensors op het device maakt. In je eigen code kun je `torch.arange(n, device=ids.device)` schrijven."),
        mo.md(r"""
        De figuur: de training loss (doorgetrokken) en de validation loss (gestippeld) na elke epoch, in een van de runs van de course met de kleinere GPT op de CPU, met seed 0. Beide dalen snel vanaf 4,17, de loss van gelijkmatig gokken, en zakken in de tweede epoch onder de 2,20 van je MLP; daarna dalen ze trager, terwijl de learning rate afneemt. In de eerste twee epochs is de training loss de hoogste: dat is het gemiddelde over de stappen van een epoch, terwijl de GPT nog snel leerde. Vanaf de derde epoch blijft de validation loss erboven: tekst die de GPT nooit zag, is moeilijker dan tekst waarvan hij leert. De runs met seed 1 en 2 eindigden binnen 0,02 van deze.
        """),
        mo.Html(C.history_picture(C.reference("cpu"), {"title": "Een run van de kleinere GPT op de CPU", "x": "epoch", "y": "loss", "lang": "nl",
                                                                    "guess": "gelijkmatig gokken", "mlp": "je MLP", "train": "training", "val": "validation", "alt": "De training loss en de validation loss na elk van de 10 epochs van de run van de course: de validation loss daalt van 2,38 na epoch 1 naar 1,66 na epoch 10, ver onder de gestippelde lijnen van gelijkmatig gokken op 4,17 en van de MLP op 2,20; de training loss eindigt op 1,47."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een run van een GPT op de tekens van een tekst:

    1. De tekens van de tekst, gesorteerd, als tokens; de tekst als ids.
    2. De eerste 90% van de ids om op te trainen, de laatste 10% om op te valideren.
    3. Vensters van `block` ids uit elk deel, als tensors op het device, in mini-batches: geschud om op te trainen, op volgorde om op te valideren.
    4. Een GPT voor deze tekens op het device, en een `Trainer` met de instellingen van de run en een pad voor de beste parameters.
    5. Train met `fit()`, en kijk na elke epoch naar beide losses: het checkpoint bewaart de parameters met de laagste validation loss.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en 2 voor de eerste regels van Tiny Shakespeare, met vensters van 4: hoeveel ids en vensters elk deel krijgt. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_split():
    _text = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n"
    # 1. de tekens, gesorteerd, en de tekst als ids
    _chars = sorted(set(_text))
    _index = {}
    for _i in range(len(_chars)):
        _index[_chars[_i]] = _i
    _ids = []
    for _ch in _text:
        _ids.append(_index[_ch])
    print(len(_chars), "tekens,", len(_ids), "ids")
    # 2. de eerste 90% om op te trainen, de laatste 10% om op te valideren
    _cut = len(_ids)  # maak af: int(0.9 * len(_ids))
    _train, _val = _ids[:_cut], _ids[_cut:]
    # vensters van 4 naast elkaar, zolang er nog een id volgt: zoals je gpt_examples ze maakt
    print("om op te trainen:", len(_train), "ids,", len(range(0, len(_train) - 4, 4)), "vensters")
    print("om op te valideren:", len(_val), "ids,", len(range(0, len(_val) - 4, 4)), "vensters")
    print(_text[_cut:])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De regels hebben 30 verschillende tekens, en 81 ids. Zoals hij is, gaan alle 81 ids naar de training, 20 vensters, en blijft er niets over om op te valideren. Afgemaakt geven 72 ids 17 vensters om op te trainen, en de laatste 9, `, speak.` en een nieuwe regel, 2 vensters om op te valideren: tekst waarop de GPT nooit traint. Voor het hele boek is dat ongeveer een miljoen ids om op te trainen, en honderdduizend om op te valideren.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke parameters worden bewaard?
    """)
    return


@app.cell(hide_code=True)
def choice_best_cell():
    choice_best = L.choice(
        "best-epoch",
        question="Een run duurt 30 epochs. De training loss daalt elke epoch; de validation loss is het laagst na epoch 18, en stijgt daarna een beetje. Welke parameters staan aan het eind in het checkpoint?",
        options={
            "last": "Die na epoch 30: de laatste",
            "right": "Die na epoch 18: de laagste validation loss",
            "train": "Die na epoch 30: hun training loss is het laagst",
        },
        correct="right",
        why={
            "last": "Je Trainer bewaart alleen als de validation loss de laagste tot nu toe is: de laatste keer was na epoch 18.",
            "train": "De training loss wordt gemeten op de vensters waarvan de GPT leert: laag daar kan betekenen dat hij die uit zijn hoofd kent. Het checkpoint gaat af op de validation loss.",
            "right": "Na epoch 18 leerde je GPT de vensters waarop hij traint beetje bij beetje uit zijn hoofd, en werd hij slechter op tekst die hij nooit zag: overfitting. Het checkpoint bewaart de beste.",
        },
    )
    choice_best
    return (choice_best,)


@app.cell(hide_code=True)
def text_exercise_loaders():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De mini-batches

        Maak `gpt_loaders(ids, block, batch_size, device)` af: stap 2 en 3 van het recept, voor elke lijst ids, zodat een latere projectstap hem ook kan gebruiken voor de tokens van je eigen tokenizer. De eerste regels importeren je `gpt_examples` uit je project. De check roept `gpt_loaders(list(range(50)), 4, 2, "cpu")` aan, waarvan de mini-batches om op te valideren één venster moeten zijn, `[45, 46, 47, 48]` met targets `[46, 47, 48, 49]`, en probeert daarna de volgorde van de mini-batches en het `"meta"`-device van PyTorch, dat een GPU nadoet, zoals in [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl). Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/gpt_loaders.py`.
        """),
        L.task("Laat `gpt_loaders` de ids splitsen, en mini-batches van vensters op het device geven: geschud om op te trainen, op volgorde om op te valideren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loaders():
    box_loaders = L.code_box("gpt-loaders", lines=16)
    box_loaders
    return (box_loaders,)


@app.cell(hide_code=True)
def check_loaders(box_loaders):
    result_loaders = L.check("gpt-loaders", box_loaders.value)
    result_loaders
    return (result_loaders,)


@app.cell(hide_code=True)
def hints_loaders():
    L.hints(
        "gpt-loaders",
        [
            "`cut` zoals in stap 2 van het afgemaakte uitgewerkte voorbeeld. Dan je `gpt_examples` één keer voor elk deel: de ids vóór `cut`, en de ids vanaf `cut`.",
            "De vensters en de targets van één deel: twee tensors op `device`, samen in een `TensorDataset`, zoals in [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl). Een `DataLoader` van `batch_size` voor elk deel; maar één van de twee wordt geschud.",
        ],
        why="De vensters om op te valideren moeten tekst zijn waarop de GPT nooit traint: het eind van het boek. Schudden doet er alleen toe bij het trainen, waar de volgorde van de mini-batches de stappen verandert; de validation loss is het gemiddelde over alle vensters, in elke volgorde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run(result_loaders):
    result_loaders  # de melding verandert als oefening 2 je gpt_loaders bewaart
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De run

        Maak `gpt_run(text, sizes, batch_size, epochs, device, path)` af: stap 4 van het recept, een `Trainer` die klaarstaat om je GPT te trainen op de tekens van `text`. Stap 1, de tekens en de ids, is gegeven; de eerste regels importeren je `GPT`, je `gpt_loaders` en je `Trainer` uit je project. De check roept `gpt_run(text, (16, 2, 1, 8), 4, 3, "cpu", path)` aan voor een korte tekst, met `path` een bestand in een tijdelijke map, en kijkt naar de logits van 3 ids en naar wat de Trainer kreeg; daarna traint hij 2 epochs, en probeert hij het `"meta"`-device. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/gpt_run.py`.
        """),
        L.course_versions(),
        L.task("Laat `gpt_run` een Trainer geven met je mini-batches en je GPT op het device, klaar om te trainen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("gpt-run", lines=21)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("gpt-run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "gpt-run",
        [
            "Je `gpt_loaders` met `ids`, `block`, `batch_size` en `device`; dan je `GPT` met het aantal verschillende tekens en de vier maten, verplaatst naar `device`.",
            "`Trainer(model, train_loader, val_loader, epochs, ...)`, met de instellingen uit het commentaar als keyword arguments, zoals in [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl): `lr`, `warmup`, `max_norm` en `path`.",
        ],
        why="De vocabulary is het aantal verschillende tekens, `len(chars)`, niet de lengte van de tekst. En alle tensors van één berekening moeten op één device staan: de mini-batches en de parameters.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _p = C.plan()
    _d, _h, _layers, _block = _p["sizes"]
    if _p["device"] == "cpu":
        _where = "Deze computer heeft geen GPU waarop de grotere GPT past" + (f" ({_p['small']} heeft te weinig geheugen)" if _p["small"] else "") + ", dus draait een kleinere versie op de CPU"
        _time = f" Op de computer van de course, een CPU met 4 cores, duurde de run ongeveer {round(C.REFERENCE['cpu'][0]['minutes'])} minuten."
    else:
        _where = f"Je computer heeft een GPU, {_p['name']}, dus je GPT traint daarop"
        _time = ""
    mo.md(f"""
    ### Wat je project nu kan

    De knop hieronder start een run op Tiny Shakespeare met je `gpt_run`, en dus met je `gpt_loaders`, je `GPT` en je `Trainer`. {_where}: een GPT van {C.grouped(_p['params'], L.lang)} parameters, met $d = {_d}$, {_h} heads, {_layers} blocks en vensters van {_block} tekens, {_p['epochs']} epochs van mini-batches van {_p['batch']}.{_time} Tijdens het trainen laat de cel na elke epoch de loss zien en de tijd die nog over is, en wacht de rest van de les. Met de stopknop van de cel stop je de run eerder; de beste parameters tot dan blijven bewaard, in `work/projects/llm/gpt_chars.pt`.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je GPT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_loaders, result_run):
    result_loaders, result_run  # opnieuw draaien als oefening 2 of 3 je module bewaart
    _labels = {"title": "Je run", "x": "epoch", "y": "loss", "lang": L.lang, "guess": "gelijkmatig gokken", "mlp": "je MLP", "train": "training", "val": "validation"}

    def _progress(_p):
        _m = 1 if _p["left"] is None else max(1, round(_p["left"] / 60))
        _left = "" if _p["left"] is None else f" · ongeveer {_m} {'minuten' if _m > 1 else 'minuut'} te gaan"
        _parts = [mo.md(f"**Epoch {_p['epoch']} van {_p['epochs']}** · stap {_p['step']} van {_p['steps']}{_left}")]
        if _p["history"]:
            _parts.append(mo.Html(C.history_picture(_p["history"], {**_labels, "alt": "De training loss en de validation loss van je run tot nu toe, na elke epoch."}, _p["epochs"])))
        mo.output.replace(mo.vstack(_parts))

    if train_button.value:
        _result = C.train(_progress)
    else:
        _result = C.last_run()
    _ready = result_loaders.state == "correct" and result_run.state == "correct"
    if _result is None and _ready and not train_button.value:
        _out = mo.md("*Druk op de knop om de run te starten.*")
    elif _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af, en druk dan op de knop: de run gebruikt je `gpt_loaders` en je `gpt_run`.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "run": "*De run stopt met een fout. Halen je modules in `work/projects/llm/` en `work/projects/nn/` hun checks nog? Verwijder een bestand dat je veranderd hebt, en check zijn oefening opnieuw.*",
            "memory": "*Je GPU heeft geen geheugen meer. Sluit andere programma's die hem gebruiken, zoals een game of een ander notebook, en druk opnieuw op de knop.*",
            "stopped": "*Je stopte de run tijdens zijn eerste epoch, dus er is niets van bewaard. Druk op de knop om opnieuw te beginnen.*",
        }[_result])
    else:
        _h = _result["history"]
        _plan = "gpu" if _result["device"] != "cpu" else "cpu"
        _limit = C.number(C.THRESHOLD[_plan], 2, L.lang)
        if _result["stopped"]:
            _verdict = f"Je stopte de run na epoch {len(_h)} van {_result['epochs']}; de runs van de course met deze GPT, met alle {_result['epochs']} epochs, eindigden onder {_limit}. Druk op de knop voor een hele run."
        elif _result["best"] <= C.THRESHOLD[_plan]:
            _verdict = f"De runs van de course met deze GPT eindigden onder {_limit}, en de jouwe ook: je GPT heeft geleerd wat hij moest leren."
        else:
            _verdict = f"De runs van de course met deze GPT eindigden onder {_limit}, en de jouwe niet. Check oefening 2 en 3 opnieuw, en die van de lessen in de melding bij oefening 3."
        if _result.get("kept"):
            _saved = f" Je eerdere run had een lagere validation loss, {C.number(_result['kept']['best'], 3, L.lang)} tegen {C.number(_result['best'], 3, L.lang)}, dus `work/projects/llm/gpt_chars.pt` houdt de parameters van die run."
        else:
            _saved = " Die parameters staan nu in `work/projects/llm/gpt_chars.pt`."
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + ", zoals de melding bij oefening 3 zegt."
        _on = {"cpu": "de CPU", "mps": "de Apple-GPU"}.get(_result["device"], _result["name"])
        _rows = []
        for _g in _result.get("guesses") or []:
            _top = ", ".join(f"`{C.shown_char(_c)}` {C.number(100 * _pr, 0, L.lang)}%" for _c, _pr in _g["top"])
            _rows.append(f"| …{C.shown_text(_g['read'])} | `{C.shown_char(_g['next'])}` | {_top} |")
        _table = "" if not _rows else ("\n\nWat je GPT als volgende verwacht, op drie plekken bij het eind van het boek, in de tekst waarop hij valideert en nooit getraind heeft (␣ is een spatie, ↵ een nieuwe regel):\n\n"
                                       "| wat hij las | volgt | de drie gokken van je GPT |\n|---|---|---|\n" + "\n".join(_rows))
        _out = mo.vstack([
            mo.Html(C.history_picture(_h, {**_labels, "alt": f"De training loss en de validation loss van je run na elke epoch: de validation loss is het laagst na epoch {_result['best_epoch']}, op {C.number(_result['best'], 2, L.lang)}."}, _result["epochs"])),
            mo.md(f"""Je GPT van {C.grouped(_result['params'], L.lang)} parameters trainde {len(_h)} {'epochs' if len(_h) > 1 else 'epoch'} op {_on}, in {C.number(_result['minutes'], 1, L.lang)} minuten: {C.grouped(_result['tokens'], L.lang)} tokens, ongeveer ${C.sci(_result['flops'], L.lang)}$ FLOPs. Zijn validation loss ging van {C.number(_h[0]['val_loss'], 2, L.lang)} na de eerste epoch naar **{C.number(_result['best'], 2, L.lang)}** na epoch {_result['best_epoch']}.{_saved} Gelijkmatig gokken geeft 4,17, en je MLP kwam tot 2,20. {_verdict}{_who}{_table}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je GPT uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl), je Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl), het device uit [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl) en de FLOPs uit [Parameters en FLOPs per token tellen](/course/open/gpt-count?lang=nl), samen in één run; het checkpoint bewaart de parameters van de epoch met de laagste validation loss, tegen de overfitting uit [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl). Je project `llm` heeft nu `gpt_loaders.py` en `gpt_run.py`, en na de run een getrainde GPT in `gpt_chars.pt`. Hierna: *Tekst genereren met je GPT*, waar je sampling-loop uit [Sampling: laat het model schrijven](/course/open/lm-sampling?lang=nl) ermee schrijft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_best, result_loaders, result_run):
    L.wrap_up(choice_best, result_loaders, result_run)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
