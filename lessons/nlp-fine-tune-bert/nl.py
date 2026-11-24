# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-fine-tune-bert. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-fine-tune-bert", "nl")
    C = L.checks   # de getallen van BERT-Tiny voor zes zinnen met een head, de scores van de gefinetunede BERT-Tiny van de course voor namen, de demo, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl) en [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_phases():
    quick_phases = L.choice(
        "quick-phases",
        question="Wat gebeurt er in *Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen* in fase 2?",
        options={
            "head": "Alleen de nieuwe laatste layer leert, met een grotere learning rate",
            "right": "Elke layer leert, met een tien keer kleinere learning rate",
            "scratch": "Elke layer leert opnieuw, vanaf willekeurige getallen",
            "swap": "De nieuwe laatste layer is bevroren, en de rest leert",
        },
        correct="right",
        why={
            "right": "Fase 1 traint alleen de nieuwe laatste layer op de bevroren features; fase 2 laat elke layer leren, met tien keer kleinere stappen, zodat de geleende features maar een beetje veranderen. In deze les is het geleende netwerk BERT-Tiny.",
            "head": "Dat is fase 1, en daar hoort de grotere learning rate bij. Welke parameters leren in fase 2, en hoe groot zijn de stappen? Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
            "scratch": "Dan zouden de geleende features weggegooid worden, en begint het netwerk opnieuw. Wat houdt fase 2 over van het getrainde netwerk? Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
            "swap": "De nieuwe laatste layer begint met willekeurige getallen, dus juist dat deel moet het meest leren. Welke parameters ontdooit fase 2? Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
        },
    )
    quick_phases
    return (quick_phases,)


@app.cell(hide_code=True)
def quick_check_bio():
    quick_bio = L.choice(
        "quick-bio",
        question="Welke tag krijgt in *Een label per woord: sequence labeling en BIO* het tweede woord van de naam van een persoon?",
        options={
            "b": "Weer B-PER, want het hoort bij de naam",
            "o": "O, want alleen het eerste woord markeert de naam",
            "right": "I-PER: binnen de naam",
            "bare": "PER, alleen het type",
        },
        correct="right",
        why={
            "right": "B begint een naam, I zet een naam voort, en O valt buiten elke naam: *Ada Lovelace* is `B-PER I-PER`. In deze les leert BERT-Tiny deze tags, voor personen, plaatsen en organisaties.",
            "b": "Dan zou *Ada Lovelace* twee namen van elk één woord zijn. Welke letter zegt dat een woord de naam ervoor voortzet? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
            "o": "Dan zou een naam van twee woorden eruitzien als een naam van één woord met een gewoon woord erachter. Welke letter markeert de woorden binnen een naam? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
            "bare": "Met alleen het type zouden twee namen direct na elkaar in elkaar overlopen. Welke letter zet BIO vóór het type van een woord dat een naam voortzet? Zie [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl).",
        },
    )
    quick_bio
    return (quick_bio,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    BERT-Tiny leerde door gaten in te vullen: het model zag nooit een recensie met het label *positief*, of een naam met de tag van een persoon. Toch zijn de vectoren na het laatste block een goed begin voor zulke taken. Jacob Devlin en de anderen die BERT maakten, beschrijven hoe weinig een taak toevoegt: per taak stop je de invoer en de uitvoer van de taak in BERT, en je finetunet alle parameters van begin tot eind. Vergeleken met de pretraining noemen de makers fine-tuning ook nog relatief goedkoop.

    In deze les zet je twee soorten head op BERT-Tiny: één die een zin een label geeft, en één die elk woord een tag geeft. Je finetunet BERT-Tiny op de 2.400 zinnen uit recensies van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl), en je leest de namen die een door de course gefinetunede BERT-Tiny vindt in de testzinnen van Universal NER.

    **Een voorbeeld uit het dagelijks leven.** Iemand die goed Engels leest, leert in een middag brieven van klanten te sorteren in lof en klachten. Niemand leert diegene eerst opnieuw Engels: alleen wat het nieuwe werk vraagt, komt erbij.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_frozen = L.guess(
        "guess",
        question="In *Logistische regressie op woorden* kreeg een weight voor elk woord 491 van de 600 achtergehouden zinnen uit recensies goed. Nu blijft BERT-Tiny bevroren, en leert alleen een nieuwe layer op de vector bij [CLS], van dezelfde 2.400 trainingszinnen, 4 epochs lang. Hoeveel van de 600 krijgt de nieuwe layer goed?",
        options={
            "coin": "Zo'n 300, niet beter dan kop of munt gooien",
            "right": "Zo'n 440, minder dan met de woorden",
            "same": "Zo'n 490, evenveel als met de woorden",
            "more": "Zo'n 560, veel meer dan met de woorden",
        },
        reveal="**Zo'n 440**: over 5 seeds mat de course 430 tot 449. Bevroren vertelt de vector bij [CLS] van BERT-Tiny de nieuwe layer minder over lof en klachten dan de woorden zelf aan logistische regressie vertelden. De demo hieronder laat zien wat er gebeurt als heel BERT-Tiny ook leert.",
    )
    guess_frozen
    return (guess_frozen,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een head voor de taak

    BERT-Tiny geeft na het laatste block op elke plek van een tekst een vector van 128 getallen. In [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl) maakte een head van de vector van een plek een score voor elk stukje. Voor een nieuwe taak krijgt BERT-Tiny een nieuwe **head**: een layer die van de vectoren de scores voor de taak maakt. **Fine-tuning** traint daarna op voorbeelden van de taak met een label, zoals in *Een getraind netwerk hergebruiken*: alleen de head, met BERT-Tiny bevroren, of de head en heel BERT-Tiny, met een kleine learning rate.

    **Om een zin een label te geven** leest de head één vector: de vector bij **[CLS]**, het stukje dat BERT vóór elke tekst zet. `nn.Linear(128, 2)` maakt daar 2 scores van, negatief en positief, en de loss is zoals altijd de cross-entropy. Daniel Jurafsky en James H. Martin schrijven in *Speech and Language Processing* dat de vector bij [CLS] in de laatste layer voor de hele invoer staat, en dient als invoer van een classifier head: logistische regressie of een neural network dat de beslissing neemt. Jurafsky en Martin noemen ook een andere manier: de som of het gemiddelde van de laatste vectoren van alle tokens.

    ### Token classification

    **Token classification** zet een head van dezelfde soort op elke plek: `nn.Linear(128, 7)` geeft bij elk stukje 7 scores, één voor elke tag van BIO met personen, plaatsen en organisaties: `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG` en `I-ORG`, genummerd van 0 tot 6.

    De tags horen bij woorden, en BERT-Tiny leest stukjes: *Lovelace* is *love* en *##lace*. De makers van BERT lazen een woord bij het eerste stukje: dat stukje gaat naar de classifier voor de tags van de namen. In deze les krijgt bij de training het eerste stukje van elk woord de tag van het woord, en krijgt elke andere plek het label **-100**: [CLS], [SEP] en elk stukje dat een woord voortzet. `nn.CrossEntropyLoss` slaat een plek met -100 over. Volgens de documentatie van PyTorch wordt een label gelijk aan `ignore_index`, -100 tenzij je iets anders kiest, genegeerd en draagt het niets bij aan de gradient. Het gemiddelde gaat dan over de andere plekken. Om daarna de tags te lezen, krijgt elk woord de tag met de hoogste score bij het eerste stukje.

    Jurafsky en Martin beschrijven een andere manier om te trainen: elk stukje van een woord krijgt de tag van het woord. Om de tags te lezen, nemen ook Jurafsky en Martin het eerste stukje.
    """)
    return


@app.cell(hide_code=True)
def idea_heads():
    mo.Html(C.heads_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        ### Bevroren of gefinetuned

        Met BERT-Tiny bevroren zijn de vectoren features, zoals in [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl): alleen de head leert. Gefinetuned leert elke parameter. De makers van BERT finetuneden heel BERT, met een learning rate van 0,00005, 0,00003 of 0,00002, batches van 16 of 32 voorbeelden en 2 tot 4 epochs. Voor de 24 kleine modellen, met BERT-Tiny erbij, noemen de aantekeningen in de BERT-repository van Google grotere learning rates, van 0,0003 tot 0,00003, en 4 epochs. De demo in deze les finetunet met een learning rate van 0,0001, en traint de head alleen met 0,001.

        Hoeveel fine-tuning toevoegt, hangt af van het model, de taak en wat erbovenop staat. Matthew Peters, Sebastian Ruder en Noah A. Smith vergeleken de twee manieren voor BERT-base, een model 25 keer zo groot als BERT-Tiny. Bevroren was de head in het onderzoek meer dan één layer: een gewogen combinatie van alle layers van BERT ging naar een eigen, groter netwerk voor de taak. Voor de namen in CoNLL 2003, een verzameling nieuwsteksten, vonden de onderzoekers een span F1 van 0,922 bevroren en 0,924 gefinetuned; voor zinnen uit filmrecensies een accuracy van 0,930 bevroren en 0,935 gefinetuned. Met maar één layer op de bevroren BERT-Tiny meet deze les veel grotere verschillen.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **head** (voor een taak) | een nieuwe layer op een pretrained model, die van de vectoren de scores van een taak maakt |
        | **token classification** | sequence labeling met een pretrained encoder: een head geeft de vector van elk token een label |
        | **-100** | het label dat `nn.CrossEntropyLoss` overslaat: de `ignore_index` |
        | **[PAD]** | stukje nummer 0: een korte zin wordt ermee aangevuld tot de lengte van de langste in de batch |
        | `model(ids, mask)` | BERT-Tiny op een batch: `ids` bevat de nummers van de stukjes, één rij per zin, en `mask` is True op de echte plekken en False op de [PAD]s; het geeft een lijst van 3 tensors met de vorm (zinnen, plekken, 128), vóór het eerste block en na elk block |
        | `x[:, k]` | plek `k` van elke rij van de tensor `x`: van de vorm (n, t, 128) naar (n, 128) |
        """),
        L.watch_out("-100 is geen tag. Tag nummer 0 is `O`, buiten elke naam: een label 0 zou het model leren dat de plek buiten elke naam valt. Een label -100 leert het model niets over de plek."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zet een head op BERT-Tiny: `nn.Linear(128, 2)` op de vector bij [CLS] om een zin een label te geven, `nn.Linear(128, 7)` op elke plek voor token classification.
    2. Geef elk trainingsvoorbeeld de labels: één voor een zin; voor token classification een tag bij het eerste stukje van elk woord en -100 op elke andere plek.
    3. Train met de cross-entropy: alleen de head, met BERT-Tiny bevroren, of alles, met een kleine learning rate.
    4. Lees de antwoorden: de hoogste van de 2 scores voor een zin; voor een woord de tag met de hoogste score bij het eerste stukje.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De knop hieronder downloadt BERT-Tiny. De code laat het model dan los op één zin, *I love this device.* (Ik ben dol op dit apparaat.) `bert.load()` geeft BERT-Tiny zelf, bevroren, en `bert.tokenizer().encode` de nummers van de stukjes van een tekst, met *[CLS]* en *[SEP]*. De zin is een batch van één, dus `mask` is overal True: geen stukje heeft nummer 0, het nummer van [PAD]. De code zet een nieuwe head voor 2 scores op de vector bij [CLS] van de eerste zin, en een nieuwe head voor 7 scores op elke plek. Beide heads hebben willekeurige getallen: geen van beide leerde nog iets.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="BERT-Tiny downloaden, 18 MB")
    return (get_button,)


@app.cell(hide_code=True)
def bert_cell(get_button):
    bert = C.bert_or_stop(get_button, {
        "ask": "*BERT-Tiny staat nog niet op je computer. De knop downloadt het model één keer van Hugging Face, in een vaste versie: `google/bert_uncased_L-2_H-128_A-2`, 18 MB, onder de Apache-2.0-licentie. Het uitgewerkte voorbeeld, de demo en het onderdeel om zelf te proberen hebben het model nodig; de oefeningen niet.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*BERT-Tiny is gedownload.*")
    return (bert,)


@app.cell
def example_cell(bert):
    import torch
    from torch import nn

    model = bert.load()
    ids = torch.tensor([bert.tokenizer().encode("I love this device.")])
    mask = ids > 0
    print(bert.pieces("I love this device."))
    layers = model(ids, mask)
    print(len(layers), layers[-1].shape)
    torch.manual_seed(0)
    sentence_head = nn.Linear(128, 2)
    tag_head = nn.Linear(128, 7)
    with torch.no_grad():
        print(sentence_head(layers[-1][0, 0]))
        print(tag_head(layers[-1]).shape)
    print(bert.pieces("Ada Lovelace lived in London."))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De zin heeft 7 stukjes, *[CLS]* en *[SEP]* meegeteld, en `model` geeft 3 tensors; de laatste heeft de vorm (1, 7, 128): 1 zin, 7 plekken, 128 getallen. `layers[-1][0, 0]` is de vector op plek 0 van zin 0, [CLS], en de head voor zinnen geeft daar 2 scores voor. De head voor tags geeft op elk van de 7 plekken 7 scores: de vorm (1, 7, 7). De stukjes van *Ada Lovelace lived in London.* laten zien waar token classification mee te maken krijgt: *Lovelace* werd *love* en *##lace*, en BERT-Tiny schreef elke letter klein.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_classify():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. classify: 2 scores voor elke zin

        Schrijf `classify(model, head, ids, mask)`. `ids` bevat de nummers van de stukjes van n zinnen, één rij per zin, met [PAD] aangevuld tot dezelfde lengte; `mask` is True op de echte plekken. Geef de 2 scores terug die `head` geeft voor de vector bij [CLS] van elke zin, na het laatste block: een tensor met de vorm (n, 2).

        De check gebruikt een kleine vervanger van BERT-Tiny, `model`, met 3 getallen per plek in plaats van 128, en een head `head` van 3 getallen naar 2 scores. De check roept `classify(model, head, ids, mask)` aan met twee zinnen, de tweede met twee [PAD]s, en met één zin van vijf stukjes. Daarna een eigen geval: drie zinnen van verschillende lengte, twee ervan met [PAD]s. Vervang de `...`.
        """),
        L.task("Geef de 2 scores van de head terug voor de vector bij [CLS] van elke zin, na het laatste block."),
    ])
    return


@app.cell(hide_code=True)
def exercise_classify():
    box_classify = L.code_box("classify", lines=3)
    box_classify
    return (box_classify,)


@app.cell(hide_code=True)
def check_classify(box_classify):
    result_classify = L.check("classify", box_classify.value)
    result_classify
    return (result_classify,)


@app.cell(hide_code=True)
def hints_classify():
    L.hints(
        "classify",
        [
            "Eén regel met drie stappen: laat `model` los op `ids` en `mask`, neem de laatste van de 3 tensors die het model geeft, en geef `head` de vector bij [CLS] van elke zin.",
            "De laatste tensor heeft de vorm (n, t, 3) bij de vervanger, en (n, t, 128) bij BERT-Tiny: zin, plek, getal. [CLS] staat op plek 0 van elke zin, en de tabel met woorden laat zien hoe je in één keer één plek van elke rij neemt.",
            "De ene regel: `    return head(model(ids, mask)[-1][___])`",
        ],
        why=r"`return head(model(ids, mask)[-1][:, 0])`: `model(ids, mask)[-1]` is de tensor na het laatste block, `[:, 0]` neemt plek 0, [CLS], van elke zin, en `head` maakt van elk van deze vectoren 2 scores. Zonder `mask` zou het model de [PAD]s als echte stukjes behandelen, en zou de vector bij [CLS] van een korte zin veranderen.",
    )
    return


@app.cell(hide_code=True)
def text_after_classify():
    mo.md(r"""
    De afbeelding gebruikt je `classify` op de getallen van BERT-Tiny voor zes achtergehouden zinnen, met een head die de course trainde op de bevroren vectoren bij [CLS]: 5 van de 6 komen aan de goede kant uit. *I did not expect this to be so good!* (Ik had niet verwacht dat het zo goed zou zijn!), een zin vol lof, krijgt 0,48.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je classify op BERT-Tiny: bevroren, dan gefinetuned

    De knop hieronder traint BERT-Tiny met een nieuwe head voor 2 scores op de 2.400 trainingszinnen, met je `classify` als oefening 1 goed is, anders met de `classify` van de course. Het model traint twee keer, elke keer 4 epochs van 75 stappen van 32 zinnen: eerst met BERT-Tiny bevroren, zodat alleen de head leert, daarna terwijl heel BERT-Tiny leert. Dat duurt ongeveer een minuut, het meeste voor de tweede run. Een grafiek toont de loss van elke stap, en gestreept de loss op de 600 achtergehouden zinnen na elke epoch; in de tweede run, dun en gestreept in grijs, de loss van de eerste. Ernaast staat de kans op positief voor de zes zinnen van de afbeelding, en eronder de vectoren bij [CLS] na het laatste block, allebei opnieuw getekend terwijl BERT-Tiny traint. Daarna kun je door de tweede run terugspoelen.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="BERT-Tiny finetunen")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_tune(bert, box_classify, demo_button, result_classify):
    bert   # BERT-Tiny is gedownload: deze cel wacht op de downloadknop
    demo_result = None
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om BERT-Tiny te finetunen.*")
    else:
        _now = {}   # de live grafiek van elke run, gemaakt bij de eerste melding, en de loss op de achtergehouden zinnen tot nu toe
        _titles = {"frozen": "Alleen de head leert", "tuned": "Heel BERT-Tiny leert"}

        def _progress(_p):
            _run = _p["run"]
            if _run not in _now or _p["step"] == 1:   # een nieuwe run, ook als de run van de course opnieuw draait in plaats van de jouwe
                _now[_run] = L.live({"title": _titles[_run], "x": "stap", "y": "loss",
                                     "alt": "De loss van elke stap tot nu toe, en gestreept de loss op de 600 achtergehouden zinnen na elke epoch; dun en gestreept in grijs, de loss op de achtergehouden zinnen van de run waarin alleen de head leert." if _run == "tuned" else "De loss van elke stap tot nu toe, en gestreept de loss op de 600 achtergehouden zinnen na elke epoch."}, _p["steps"], low=0.0,
                                    key=[("loss", False, "2.400 trainingszinnen, elke stap"), ("loss", True, "600 achtergehouden zinnen, na elke epoch")])
                _now[_run].watch(lambda: C.now_parts(_p["look"](), _run, L.lang))
                _now[_run + "-held"] = [(0, _p["held_start"])]
                if _run == "tuned" and "frozen-held" in _now:   # de loss op de achtergehouden zinnen als alleen de head leert, in grijs
                    _now["tuned"].reference(_now["frozen-held"], "alleen de head", role="ink-2")
            _line = f"**{_titles[_run]}** · epoch {_p['epoch']} van {_p['epochs']} · stap {_p['step']} van {_p['steps']} · loss {L.number(_p['loss'], 3)}"
            if "held_loss" in _p:   # de achtergehouden zinnen na elke epoch, gestreept
                _now[_run + "-held"].append((_p["step"], _p["held_loss"]))
                _line += f" · {_p['right']} van de 600 achtergehouden zinnen goed"
            _now[_run].reference(_now[_run + "-held"], "", role="loss", replace=True)
            _now[_run].add(_p["step"], _p["loss"], _line)

        demo_result = C.demo(box_classify.value if result_classify.state == "correct" else None, _progress)
        _who = "je eigen `classify`" if demo_result["yours"] else "de `classify` van de course"
        _labels = {"title": "Heel BERT-Tiny leert", "x": "stap", "y": "loss", "frozen": "alleen de head",
                   "alt": "De loss van elke stap van de run waarin heel BERT-Tiny leert, gestreept de loss op de 600 achtergehouden zinnen na elke epoch, en dun en gestreept in grijs de loss van de run waarin alleen de head leert."}

        def _frame(_s):
            _s = int(_s)
            _right = C.held_at(demo_result, _s)
            _said = f"Na stap {_s}: {_right} van de 600 achtergehouden zinnen goed." if _right is not None else f"Na stap {_s}, in epoch {(_s - 1) // 75 + 1}."
            return C.rewind_frame(demo_result, _s, _labels, L.lang), _said

        _table = mo.md(f"""Met {_who}, van de 600 achtergehouden zinnen, na 4 epochs:

| | goed |
|---|---|
| alleen de head leert, BERT-Tiny bevroren | {C.held_at(demo_result, C.STEPS, "frozen")} |
| heel BERT-Tiny leert: gefinetuned | {C.held_at(demo_result, C.STEPS)} |
| logistische regressie op woorden, in *Logistische regressie op woorden* | {C.REFERENCE['words']} |

Over 5 seeds, met de code van de course: {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]} als alleen de head leert, {C.REFERENCE['tuned'][0]} tot {C.REFERENCE['tuned'][1]} gefinetuned.""")
        _out = mo.vstack([L.explore("rewind", "stap", _frame, start=C.STEPS, values=C.rewind_steps(demo_result), role="",
                                    task="Ga door de run vanaf stap 0: wanneer schuiven de zes zinnen naar de goede kant, en veranderen de rijen vakjes mee?"), _table])
    _out
    return (demo_result,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In de runs van de course krijgt de head alleen na 4 epochs 430 van de 600 achtergehouden zinnen goed, en de gefinetunede BERT-Tiny 497. In de bevroren run blijven de rijen vakjes bij elke stap gelijk, en verschuift alleen de head de balken. In de gefinetunede run veranderen de rijen zelf, en na epoch 2 staan vijf van de zes zinnen aan de goede kant. De gefinetunede BERT-Tiny krijgt er ongeveer evenveel goed als logistische regressie op woorden, 491: een klein model op 2.400 zinnen wint weinig ten opzichte van een weight per woord. In de laatste epoch gaat de loss op de achtergehouden zinnen weer omhoog, van 0,402 naar 0,491, terwijl het aantal goed nauwelijks verandert, van 498 naar 497.

    Beide runs zetten *I did not expect this to be so good!* aan de negatieve kant. In *Logistische regressie op woorden* had *not* (niet) van alle woorden de sterkste weight richting negatief, −1,78.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_labels():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. piece_labels: een label voor elke plek

        Schrijf `piece_labels(word_pieces, tags)`. `word_pieces` bevat de stukjes van elk woord van een zin, een lijst per woord, en `tags` het nummer van de tag van elk woord. Geef een label terug voor elke plek die BERT-Tiny leest: -100 voor [CLS]; dan voor elk woord de tag bij het eerste stukje en -100 bij elk later stukje; dan -100 voor [SEP].

        Voor `piece_labels([["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]], [1, 2, 0, 0, 3, 0])` is de uitkomst `[-100, 1, 2, -100, 0, 0, 3, 0, -100]`. Vervang de `...`. De check roept dat aan en hetzelfde voor *Marie Curie moved to Paris.*, met `[["marie"], ["cu", "##rie"], ["moved"], ["to"], ["paris"], ["."]]`, en daarna eigen gevallen: een woord van drie stukjes, een zin van één woord, een zin waarin elk woord één stukje is, met een organisatie, en een testzin van Universal NER, `C.word_pieces(913)` met `C.gold_tags(913)`: *I just had the best experience at this Kal Tire location .*
        """),
        L.task("Geef -100 terug voor [CLS], de tag van elk woord bij het eerste stukje en -100 bij de andere stukjes, en -100 voor [SEP]."),
    ])
    return


@app.cell(hide_code=True)
def exercise_labels():
    box_labels = L.code_box("piece_labels", lines=7)
    box_labels
    return (box_labels,)


@app.cell(hide_code=True)
def check_labels(box_labels):
    result_labels = L.check("piece_labels", box_labels.value)
    result_labels
    return (result_labels,)


@app.cell(hide_code=True)
def hints_labels():
    L.hints(
        "piece_labels",
        [
            "`labels` begint met -100, voor [CLS]. Ga door de woorden met `for i in range(len(word_pieces)):`. Voeg voor woord `i` met `labels.append` één label voor het eerste stukje toe aan `labels`, en daarna -100 voor elk stukje in `word_pieces[i][1:]`. Na de loop nog één -100, voor [SEP].",
            "Het eerste stukje van woord `i` draagt de tag van woord `i`. Welk van de twee argumenten bevat de tags, en welke plek daarin hoort bij woord `i`?",
            "De regel in de loop die het eerste stukje een label geeft: `        labels.append(___)`",
        ],
        why=r"`labels.append(tags[i])`: het eerste stukje van woord `i` krijgt de tag van het woord. Helemaal, na `labels = [-100]`: `for i in range(len(word_pieces)):` met `labels.append(tags[i])` eronder, en daaronder `for piece in word_pieces[i][1:]:` met `labels.append(-100)`; na de loop `labels.append(-100)` voor [SEP]. In het voorbeeld krijgt *love* 2, `I-PER`, en *##lace* -100.",
    )
    return


@app.cell(hide_code=True)
def text_after_labels():
    mo.md(r"""
    In de afbeelding gaf je functie de plekken van drie testzinnen een label. Van de 41 plekken krijgen er 28 een tag; de andere 13, de twee uiteinden en de latere stukjes van *ginny*, *Travelocity*, *Expedia*, *carval* en *walmart*, krijgen -100.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Oefening 3. Waarom -100 bij een later stukje?
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-piece",
        question="Bij de training krijgt *##lace*, het tweede stukje van *Lovelace*, het label -100. Waarom?",
        options={
            "outside": "Zodat het model leert dat *##lace* buiten elke naam valt",
            "novector": "Omdat BERT-Tiny geen vector geeft bij een stukje dat een woord voortzet",
            "right": "Zodat elk woord één keer telt, op de plek waar de tag gelezen wordt",
            "faster": "Om de training sneller te maken, met minder plekken om te rekenen",
        },
        correct="right",
        why={
            "right": "Met een tag bij elk stukje zou *Travelocity*, drie stukjes, drie keer meetellen in de loss, en *Paris* één keer. Met -100 geeft elk woord de loss één label, bij het eerste stukje, de plek waar de tag na de training gelezen wordt. Jurafsky en Martin beschrijven de andere manier, de tag van het woord bij elk stukje; om de tags te lezen, nemen beide manieren het eerste stukje.",
            "outside": "Dat zou het label 0 zijn, de tag `O`. Wat doet `nn.CrossEntropyLoss` met een plek met het label -100?",
            "novector": "BERT-Tiny geeft op elke plek een vector, ook bij *##lace*, en de head geeft daar 7 scores. Wat doet -100 met deze scores in de loss?",
            "faster": "BERT-Tiny en de head rekenen hoe dan ook op elke plek, en de loss is maar een klein deel van het werk. Wat zou er in de loss veranderen als *##lace* de tag van *Lovelace* kreeg?",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_exercise_tags():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. word_tags: een tag voor elk woord

        Schrijf `word_tags(scores, word_pieces)`. `scores` is een NumPy-array met een rij van 7 scores voor elke plek van een zin, [CLS] en [SEP] meegeteld, zoals de head voor tags de scores geeft; `word_pieces` is zoals in oefening 2. Geef het nummer van de tag van elk woord terug: de tag met de hoogste score bij het eerste stukje.

        `ada_scores` bevat scores van de course zelf voor *Ada Lovelace lived in London.*, 9 plekken: op elke plek heeft één tag 2,0 en hebben de andere tags 0,0. Voor `word_tags(ada_scores, [["ada"], ["love", "##lace"], ["lived"], ["in"], ["london"], ["."]])` is de uitkomst `[1, 2, 0, 0, 3, 0]`. Vervang de `...`. De check roept dat aan en hetzelfde voor *Marie Curie moved to Paris.*, met `curie_scores`, en daarna eigen gevallen: een woord van drie stukjes, een zin van één woord waarvan het eerste stukje twee tags met dezelfde hoogste score heeft, waarbij het laagste nummer telt, en de scores van de gefinetunede BERT-Tiny van de course voor twee testzinnen, `C.test_scores(379)` met `C.word_pieces(379)` en `C.test_scores(913)` met `C.word_pieces(913)`.
        """),
        L.task("Geef voor elk woord het nummer terug van de tag met de hoogste score bij het eerste stukje."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tags():
    box_tags = L.code_box("word_tags", lines=10)
    box_tags
    return (box_tags,)


@app.cell(hide_code=True)
def check_tags(box_tags):
    result_tags = L.check("word_tags", box_tags.value)
    result_tags
    return (result_tags,)


@app.cell(hide_code=True)
def hints_tags():
    L.hints(
        "word_tags",
        [
            "`tags` begint leeg, en `place` bij 1: het eerste stukje van het eerste woord, direct na [CLS]. Ga door de woorden met `for pieces in word_pieces:`. Voeg voor elk woord `int(np.argmax(scores[place]))` toe aan `tags`, en schuif `place` dan door naar het eerste stukje van het volgende woord.",
            "Een woord van drie stukjes neemt drie plekken in, en een woord van één stukje één plek. Hoe ver schuift `place` op na een woord?",
            "De regel in de loop die doorschuift: `        place = place + ___`",
        ],
        why=r"`place = place + len(pieces)`: na een woord springt `place` over alle stukjes van het woord heen, naar het eerste stukje van het volgende woord. In het voorbeeld staat *love* op plek 2; *Lovelace* heeft twee stukjes, dus het volgende woord, *lived*, begint op plek 4, en *##lace* op plek 3 wordt overgeslagen.",
    )
    return


@app.cell(hide_code=True)
def text_after_tags():
    mo.md(r"""
    In de afbeelding las je functie de tags van alle 1.471 testzinnen van Universal NER uit de scores van een BERT-Tiny die de course finetunede voor namen: met de head voor tags, de labels van oefening 2, een learning rate van 0,0001 en 3 epochs over de 8.773 trainingszinnen. Het model vindt 783 namen, waarvan 477 goed, van de 802 namen in de testset: een span F1 van 0,602. De tagger van [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl) haalde 0,551 in [Een tagger beoordelen: accuracy en span F1](/course/open/nlp-span-f1?lang=nl); dezelfde head op een bevroren BERT-Tiny, getraind met een learning rate van 0,01, haalt 0,411. Mayhew en anderen, de makers van Universal NER, geven 0,858 voor een veel groter model, getraind op dezelfde Engelse trainingsset, op de eigen testset, waarin ook de e-mails staan die deze course weglaat.

    BERT-Tiny schrijft elke tekst in kleine letters voordat het model de tekst in stukjes knipt: *Travelocity* en *travelocity* zijn voor BERT-Tiny hetzelfde. De tagger van *Namen vinden* keek naar woordvormen, met een hoofdletter als `X`. In de afbeelding mist de gefinetunede BERT-Tiny *Travelocity* en *Expedia*, en noemt het model *carval* en *walmart* plaatsen, waar de testset organisaties heeft.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: een eigen zin

    Typ een Engelse zin, bijvoorbeeld een regel uit een recensie. De afbeelding laat de kans op positief zien die de twee runs van de demo de zin geven: de head op de bevroren BERT-Tiny, en de gefinetunede BERT-Tiny. Het onderdeel heeft de demo hierboven nodig. De zin hieronder, *Not bad at all.* (Helemaal niet slecht.), prijst met twee negatieve woorden; probeer een zin met *but* (maar) in het midden, en dezelfde zin met de twee helften omgedraaid.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_text = mo.ui.text(value="Not bad at all.", label="Zin", full_width=True)
    explore_text
    return (explore_text,)


@app.cell(hide_code=True)
def explore_look(bert, demo_result, explore_text):
    mo.stop(demo_result is None, mo.md("*Finetune BERT-Tiny eerst met de knop van de demo.*"))
    mo.Html(C.explore_picture(bert, demo_result, explore_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** BERT-Tiny leerde gaten invullen in [BERT: het ontbrekende woord invullen](/course/open/nlp-bert?lang=nl). Een nieuwe head maakt er een model voor een taak van: op de vector bij [CLS] om een zin een label te geven, op elke plek om elk woord een tag te geven, zoals in [Een label per woord: sequence labeling en BIO](/course/open/nlp-bio-tags?lang=nl), met de tag bij het eerste stukje van een woord en -100 op elke andere plek. Fine-tuning traint de head en heel BERT-Tiny samen, met een kleine learning rate; de twee fases van [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl) trainden eerst de head alleen. Op de recensies krijgt de gefinetunede BERT-Tiny 497 van de 600 achtergehouden zinnen goed, tegen 430 als alleen de head leert; voor namen haalt het model een span F1 van 0,602, tegen 0,551 voor de tagger met woordvormen.

    **Een voorbeeld uit het dagelijks leven.** RobBERT is een Nederlands model uit de BERT-familie. Pieter Delobelle en de anderen die het model maakten, finetuneden RobBERT op recensies van Nederlandstalige boeken, en het model geeft 95,1% van de recensies in de testset het goede label, positief of negatief. Voor de namen in de Nederlandse teksten van CoNLL-2002 haalt RobBERT een span F1 van 0,891.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_classify, result_labels, result_tags):
    L.wrap_up(result_classify, result_labels, choice_first, result_tags)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
