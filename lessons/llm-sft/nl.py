# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-sft. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-sft", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Chat templates](/course/open/llm-chat-template?lang=nl) en [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_turn():
    quick_turn = L.choice(
        "quick-turn",
        question="Een chat model schrijft zijn antwoord, token voor token. Hoe weet de loop die het laat schrijven dat het antwoord klaar is?",
        options={
            "right": "Het model schrijft zijn end-of-turn-token",
            "length": "Na een vast aantal tokens",
            "dot": "Bij de eerste punt",
        },
        correct="right",
        why={
            "length": "Een vast aantal is alleen een grens, zodat een loop altijd stopt. Een chat model beëindigt zelf zijn beurt: de loop stopt bij zijn end-of-turn-token. Zie [Chat templates](/course/open/llm-chat-template?lang=nl).",
            "dot": "Een antwoord kan veel zinnen hebben. Het model beëindigt zelf zijn beurt, met zijn end-of-turn-token. Zie [Chat templates](/course/open/llm-chat-template?lang=nl).",
            "right": "Het model leerde dat token aan het eind van elk antwoord te schrijven. Vandaag zie je waar het dat leert.",
        },
    )
    quick_turn
    return (quick_turn,)


@app.cell(hide_code=True)
def quick_check_lr():
    quick_lr = L.choice(
        "quick-lr",
        question="Je traint verder met elke parameter van een netwerk dat al getraind is. Welke learning rate neem je?",
        options={
            "bigger": "Een grotere dan bij de eerste training",
            "same": "Dezelfde als bij de eerste training",
            "right": "Een kleinere dan bij de eerste training",
        },
        correct="right",
        why={
            "bigger": "Met een grote stap bewegen de getrainde weights te ver en verliezen ze wat ze leerden. Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
            "same": "De getrainde weights zijn al goed; de learning rate van de eerste training beweegt ze te ver. Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
            "right": "Kleine stappen houden wat het netwerk leerde. Vandaag finetune je je language model op dezelfde manier.",
        },
    )
    quick_lr
    return (quick_lr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je model uit de capstone schrijft tekst verder. Geef het *Write a story with the word: kite.* en het schrijft door: misschien een verhaal, met of zonder vlieger, of nog een instructie. Een chat model is een base model dat verder getraind is, eerst op voorbeelden van een instructie met een goed antwoord. Die training heet **supervised fine-tuning**, SFT. InstructGPT, een GPT-3 dat OpenAI leerde instructies te volgen (Ouyang en anderen, 2022), begint ermee, en het open recept van nanochat uit [Een open trainingsrecept lezen](/course/open/llm-open-recipes?lang=nl) heeft het ook.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe kok heeft duizend recepten gelezen: lees het begin van een recept voor, en hij maakt het af. In zijn eerste dagen in een restaurant ziet hij de bestellingen binnenkomen, elk met het gerecht dat ervoor gemaakt werd. Na een paar honderd kookt hij wat er besteld wordt, en stopt hij als het bord vol is.
    """)
    return


@app.cell(hide_code=True)
def predict_after():
    _m = C.MEASURED["yours"]
    guess_after = L.guess(
        "after",
        question=f"De course gaf haar model uit de demorun van de capstone {_m['before']['total']} instructies, zoals *Write a story with the words: kite and park.* Zijn verhalen gebruikten elk gevraagd woord bij {_m['before']['used']} ervan. Dan {C.number(C.STEPS, 0, L.lang)} stappen SFT op voorbeelden uit andere verhalen, ongeveer {round(_m['seconds'] / 60)} minuten op één thread van de CPU van de course. Bij hoeveel van dezelfde instructies nu?",
        options={
            "same": "Ongeveer evenveel: een paar minuten training veranderen weinig",
            "right": "Een paar keer zoveel, maar lang niet allemaal",
            "all": "Bijna allemaal",
        },
        reveal=f"**{_m['after']['used']} van de {_m['after']['total']}.** Hetzelfde model, dezelfde instructies: een paar minuten voorbeelden leerden het een instructie te beantwoorden met een verhaal dat de woorden gebruikt. Zijn verhalen zijn vaak nog vreemd: een model van {C.number(_m['params'], 0, L.lang)} parameters leert in {round(C.P.MEASURED['run']['seconds'] / 60)} minuten pretraining nu eenmaal niet alles.",
    )
    guess_after
    return (guess_after,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Supervised fine-tuning.** Schrijf elk voorbeeld als één tekst: een instructie, en een goed antwoord erop, in een vaste template. Train het gepretrainde model dan op die teksten zoals bij pretraining: voorspel elk volgend token, met dezelfde loss en dezelfde stap. Alleen de data zijn anders, en zoals in [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl) zijn er minder stappen en is de learning rate lager. Trainen op instructies heet ook **instruction tuning**.

    **De template.** Je tokenizer heeft één special token: het end-of-text. Dus is de template van de course gewone tekst: de instructie, en op een eigen regel `Story:`, de generation prompt. Het antwoord begint op de regel erna en eindigt met het end-of-text, dat hier het end-of-turn-token is. De chat template van een echt model, zoals in [Chat templates](/course/open/llm-chat-template?lang=nl), doet hetzelfde met special tokens rond elk bericht.

    **Loss masking.** De instructie is gegeven: het model hoeft niet te leren die te schrijven. Dus krijgt elk token van de instructie het label **-100**, een getal dat geen token is. `F.cross_entropy` slaat elk doel van -100 over, en neemt het gemiddelde van de andere: de loss, en dus elke gradient, komt alleen van het antwoord en zijn end-of-text.
    """)
    return


@app.cell(hide_code=True)
def picture_example():
    mo.vstack([
        mo.md(f"Het beeld: het voorbeeld *{C.PICTURE[1]}* voor het woord *{C.PICTURE[0][0]}*, in de tokens van het model dat deze les finetunet, met onder elk token zijn label. ↵ is een nieuwe regel, ␣ een spatie."),
        C.example_picture({"title": "Een voorbeeld, en het label van elk token", "end": "einde",
                           "alt": "Een instructie, een antwoord en een end-of-text in tokens geknipt. Onder de tokens van de instructie staat het label -100; onder het antwoord en het end-of-text hun eigen ids."}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_batch():
    mo.md(r"""
    **Verschuiven en padding.** Zoals bij pretraining zijn de inputs de ids zonder de laatste, en de doelen de labels zonder de eerste: op elke plek voorspelt het model het volgende token. De voorbeelden verschillen in lengte, en een tensor is een rechthoek, dus krijgt elke rij van een batch **padding** tot de langste: de inputs met het end-of-text, al zou elk id werken, en de doelen met -100, zodat de padding nergens voor telt. De padding komt achteraan: door het causal mask kijkt een token alleen naar de tokens ervoor, dus ziet geen echt token ooit de padding.

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **supervised fine-tuning**, SFT | een gepretraind model trainen op voorbeelden van een instructie met een goed antwoord | |
    | **instruction tuning** | een andere naam voor SFT op instructies | |
    | **loss masking** | de doelen die niet mogen tellen het label -100 geven: de instructie, en de padding | `[-100] * len(prompt_ids)` |
    | -100 | het label dat `F.cross_entropy` overslaat: zijn `ignore_index` | `F.cross_entropy(logits, y)` |
    | **padding** | de kortere rijen van een batch aanvullen tot de lengte van de langste | `[end_id] * pad` |

    ### Recept

    Om een base model op instructies te finetunen:

    1. Schrijf elk voorbeeld in de template: de ids van de instructie, van het antwoord, en het end-of-text.
    2. Geef elk id een label: -100 voor de instructie, de ids zelf voor het antwoord en het einde.
    3. Maak batches: de inputs zonder het laatste id, de doelen zonder het eerste label, achteraan aangevuld met het end-of-text en -100.
    4. Train met de stap van de pretraining, met een lagere learning rate, een paar honderd tot een paar duizend stappen.
    5. Test op instructies die het model nooit zag, met een check die een programma kan doen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een instructie van drie ids, `[5, 9, 2]`, en een antwoord van twee, `[7, 3]`; het end-of-text is 0. De ids zijn verzonnen, en de logits van het model op de vijf plekken ook.
    """)
    return


@app.cell
def example_mask():
    _prompt, _answer, _end = [5, 9, 2], [7, 3], 0
    _ids = _prompt + _answer + [_end]
    _labels = [-100] * len(_prompt) + _answer + [_end]
    _x, _y = _ids[:-1], _labels[1:]
    print(_x)
    print(_y)
    torch.manual_seed(0)
    _logits = torch.randn(len(_x), 10)   # 10 tokens in de vocabulary
    print(round(F.cross_entropy(_logits, torch.tensor(_y)).item(), 4))
    print(round(F.cross_entropy(_logits[2:], torch.tensor(_y[2:])).item(), 4))   # alleen de laatste drie plekken
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op plek 2 heeft het model 5, 9 en 2 gezien, en moet het 7 voorspellen, het eerste token van het antwoord; op plek 4 moet het het einde voorspellen. De eerste twee plekken hebben het label -100, dus is de loss het gemiddelde van alleen de laatste drie: hetzelfde getal als wanneer je de eerste twee weglaat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_example():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een voorbeeld en zijn labels

        Maak `sft_example(prompt_ids, answer_ids, end_id)` af: stap 1 en 2. Geef de ids van het hele voorbeeld terug en een label voor elk id. De check roept `sft_example([31, 8, 14, 2], [50, 51, 52], 99)` aan, en andere.
        """),
        L.task("Laat `sft_example` de ids en de labels van een voorbeeld geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_example():
    box_example = L.code_box("sft-example", lines=7)
    box_example
    return (box_example,)


@app.cell(hide_code=True)
def check_example(box_example):
    result_example = L.check("sft-example", box_example.value)
    result_example
    return (result_example,)


@app.cell(hide_code=True)
def hints_example():
    L.hints(
        "sft-example",
        [
            "`+` zet lijsten aan elkaar: `[1, 2] + [3]` is `[1, 2, 3]`. `end_id` is één getal, dus `[end_id]` is een lijst van één.",
            "`[-100] * 3` is `[-100, -100, -100]`: één -100 voor elk id van de instructie, en dan het antwoord en het einde, net als in de ids.",
        ],
        why="De labels zeggen wat meetelt in de loss: -100 voor wat gegeven was, de ids zelf voor wat het model moet schrijven, tot en met het end-of-text, zodat het leert stoppen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_batch():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een batch

        Maak `sft_batch(examples, end_id)` af: stap 3. `examples` is een lijst van `(ids, labels)`, zoals je `sft_example` ze geeft. Geef twee tensors terug: de inputs `x` en de doelen `y`, een rij per voorbeeld, één plek verschoven en achteraan aangevuld tot de langste: de inputs met `end_id`, de doelen met -100. De check laat je functie lopen op twee voorbeelden van verschillende lengte, en meer.
        """),
        L.task("Laat `sft_batch` voorbeelden van verschillende lengte omzetten in één batch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_batch():
    box_batch = L.code_box("sft-batch", lines=12)
    box_batch
    return (box_batch,)


@app.cell(hide_code=True)
def check_batch(box_batch):
    result_batch = L.check("sft-batch", box_batch.value)
    result_batch
    return (result_batch,)


@app.cell(hide_code=True)
def hints_batch():
    L.hints(
        "sft-batch",
        [
            "Het langste voorbeeld heeft `max(len(ids) for ids, labels in examples)` ids, en één input minder: zijn laatste id is alleen een doel. Een rij van `len(ids) - 1` inputs heeft er `pad = n - (len(ids) - 1)` bij nodig.",
            "`[7, 3, 0][:-1]` is `[7, 3]`, en `[7, 3] + [0] * 2` is `[7, 3, 0, 0]`: een rij zonder zijn laatste id, en dan de padding. De doelen gaan net zo, vanuit `labels[1:]`, met `-100` als padding.",
        ],
        why="Een tensor heeft rijen van één lengte nodig. De doelen zijn één plek verschoven, zoals bij pretraining, en de padding krijgt -100, zodat die nergens voor telt in de loss.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_words():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Deed het wat er gevraagd werd?

        Maak `words_used(story, words)` af: stap 5. Geef `True` terug als elk woord van `words` als heel woord in het verhaal staat, en anders `False`. De gevraagde woorden zijn in kleine letters; in het verhaal kan een woord met een hoofdletter beginnen, en leestekens ervoor of erachter hebben, zoals `"Kite!`. *Kites* is een ander woord dan *kite*. De check roept `words_used("Tom flew his kite in the park.", ["kite", "park"])` aan, en andere.
        """),
        L.task("Laat `words_used` zeggen of een verhaal elk gevraagd woord gebruikt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("words-used", lines=8)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("words-used", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "words-used",
        [
            "`story.lower().split()` geeft de woorden van het verhaal in kleine letters. `word.strip(\".,!?;:\\\"'\")` haalt leestekens van beide kanten van een woord, zoals in [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl).",
            "Zet de gestripte woorden in een set met `found.add(...)`. Dan een loop over `words` die `False` teruggeeft zodra er één niet in `found` staat; na de loop `True`.",
        ],
        why="Een programma kan alleen nakijken wat het kan zien: hele woorden, wat hun hoofdletters en leestekens ook zijn. De demo telt met je functie hoeveel antwoorden doen wat er gevraagd werd.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _m = C.MEASURED["yours"]
    mo.vstack([
        mo.md(f"""
        ### Finetune je model

        De knop schrijft een instructie voor elk verhaal van `data/tinystories-small.txt` behalve de laatste {C.HELD} dat een van de {len(C.WORDS)} zelfstandige naamwoorden van de lijst van de course noemt: het verhaal afgeknipt na zijn laatste zin die met zijn instructie in {C.MAX_LEN} tokens past, en een of twee van zijn zelfstandige naamwoorden als gevraagde woorden. Hij maakt de voorbeelden met je `sft_example`: ongeveer {C.number(round(_m['pairs'], -2), 0, L.lang)}. Dan beantwoordt je model uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl), of dat van de course als je er geen hebt, {C.TESTS} instructies uit de laatste {C.HELD} verhalen, waar geen van de voorbeelden uit komt, en telt je `words_used` de antwoorden die elk woord gebruiken. Met maar {len(C.WORDS)} woorden om naar te vragen staan de meeste van deze instructies ook tussen de voorbeelden, elk met een ander verhaal: wat telt, is of het eigen verhaal van het model de woorden gebruikt. Dan {C.number(C.STEPS, 0, L.lang)} stappen SFT, met je `sft_batch` en je `pretrain_step` uit de capstone met een learning rate van {C.number(C.LR, 3, L.lang)}, een derde van die van de pretraining, en weer dezelfde {C.TESTS} instructies.

        Je `pretrain_step` hoeft niet te veranderen: `F.cross_entropy` slaat elk doel van -100 vanzelf over. Op de CPU van de course, met één thread, deed de knop er in totaal ongeveer {round((_m['seconds'] + _m['other_seconds']) / 60)} minuten over voor haar model uit de demorun van de capstone, van {C.number(_m['params'], 0, L.lang)} parameters; het eigen model van de course is ongeveer vijf keer zo groot, en doet er ongeveer vier keer zo lang over. Het gefinetunede model wordt bewaard in `work/projects/llm/sft.pt`, voor de lessen hierna.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def tune_button_cell():
    tune_button = mo.ui.run_button(label="Finetune je model")
    tune_button
    return (tune_button,)


@app.cell(hide_code=True)
def demo_tune(box_batch, box_example, box_words, result_batch, result_example, result_words, tune_button):
    tuned = None
    _note = None
    if not all(_r.state == "correct" for _r in (result_example, result_batch, result_words)):
        _note = "*Maak eerst oefening 1 tot en met 3 af: de demo finetunet met je `sft_example` en `sft_batch`, en telt met je `words_used`.*"
    elif tune_button.value:
        def _progress(_p):
            if _p["phase"] == "pairs":
                mo.output.replace(mo.md(f"**{C.number(_p['pairs'], 0, L.lang)} voorbeelden** · het model beantwoordt de instructies …"))
            elif _p["phase"] in ("before", "after"):
                _when = "Voor SFT" if _p["phase"] == "before" else "Na SFT"
                mo.output.replace(mo.md(f"**{_when}** · instructie {_p['done']} van {_p['total']} · elk woord gebruikt: {_p['used']}"))
            else:
                mo.output.replace(mo.md(f"**Stap {C.number(_p['step'], 0, L.lang)} van {C.number(_p['steps'], 0, L.lang)}** · loss {C.number(_p['loss'], 3, L.lang)} · {C.number(_p['seconds'] / 60, 1, L.lang)} minuten"))
        _r = C.run(run_code(box_example.value).names["sft_example"], run_code(box_batch.value).names["sft_batch"],
                   run_code(box_words.value).names["words_used"], _progress)
        if _r == "run":
            _note = "*De demo stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks? Verwijder een bestand dat je veranderde, en check zijn oefening opnieuw.*"
        elif _r == "example":
            _note = "*Je `sft_example` stopt met een fout op de verhalen, of geeft ids en labels van verschillende lengte. Werkt hij voor een antwoord van honderd ids?*"
        elif _r == "batch":
            _note = "*Je `sft_batch` stopt met een fout op een batch van de verhalen. Werkt hij voor 16 voorbeelden van verschillende lengte?*"
        elif _r == "words":
            _note = "*Je `words_used` stopt met een fout op een antwoord van het model, of geeft iets anders dan `True` of `False`. Werkt hij voor een verhaal zonder een van de woorden?*"
        elif _r == "memory":
            _note = "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop.*"
        elif _r == "diverged":
            _note = "*De loss is geen getal meer: de training is ontspoord. Is de learning rate van je `pretrain_step` de learning rate die hij krijgt?*"
        else:
            tuned = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (tuned,)


@app.cell(hide_code=True)
def demo_result(tuned):
    _m = C.MEASURED["yours"]
    _info = C.saved()
    _run = tuned if tuned is not None else _info if _info is not None else _m
    _base = tuned["source"] if tuned is not None else _info["base"] if _info is not None else None
    _whose = {"yours": "je model", "course": "het model van de course", None: "het model uit de demorun van de capstone van de course"}[_base]
    _rows = "\n".join(f"| {_label} | **{_run[_when]['used']}** van {_run[_when]['total']} | {_run[_when]['stopped']} van {_run[_when]['total']} |"
                      for _label, _when in (("voor SFT", "before"), ("na SFT", "after")))
    _stand = "" if _base is not None else "\n\nDe run van de course staat er tot je je eigen model finetunet."
    _example = _run["example"] if tuned is None else C.example(tuned["before"], tuned["after"])
    mo.vstack([
        mo.md(f"De antwoorden van {_whose} op de {_run['after']['total']} testinstructies:\n\n| | elk woord gebruikt | zelf gestopt |\n|---|---|---|\n{_rows}{_stand}"),
        mo.md(("De eerste van deze instructies die het model na SFT wel uitvoerde en ervoor niet: " if _example.get("better") else "De eerste van deze instructies: ") + f"*{C.instruction(_example['words']).splitlines()[0]}*\n\nVoor SFT:\n\n" + C.P.quote(_example["before"] or "…")),
        mo.md("Na SFT:\n\n" + C.P.quote(_example["after"] or "…")),
    ])
    return


@app.cell(hide_code=True)
def text_ask():
    mo.md(r"""
    ### Vraag het je model

    Typ een of twee woorden, en laat je gefinetunede model er een verhaal mee schrijven. Het model kent alleen Engels: woorden die de verhalen van TinyStories vaak gebruiken, werken het best.
    """)
    return


@app.cell(hide_code=True)
def ask_cell():
    ask_box = mo.ui.text(value="kite park", label="Je woorden", full_width=True)
    ask_button = mo.ui.run_button(label="Schrijf")
    mo.hstack([ask_box, ask_button], align="end")
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, tuned):
    _note = None
    if ask_button.value:
        _words = C.asked_words(ask_box.value)
        _t = C.tools()
        if not _words:
            _note = "*Typ eerst een woord, in letters van a tot z.*"
        elif _t == "run":
            _note = "*Schrijven stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks?*"
        else:
            _model, _info, _ = (tuned["model"], C.saved(), "yours") if tuned is not None else C.load()
            _merges = {tuple(_pair): 256 + _i for _i, _pair in enumerate(_info["merges"])}
            _text, _ended = C.write_for(_model, _merges, _words, _t["encode_fast"], _t["decode"])
            _whose = "" if C.saved() is not None else " Het SFT-model van de course schreef dit: finetune eerst je eigen."
            _note = f"*{C.instruction(_words).splitlines()[0]}*{_whose}\n\n" + C.P.quote(_text or "…") + ("" if _ended else "\n\n*Het stopte niet vanzelf binnen " + str(C.LENGTH) + " tokens.*")
    mo.md(_note) if _note else None
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ### Oefening 4. Het einde van een antwoord
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "no-end",
        question="Je vergeet het end-of-text aan het eind van elk antwoord in de voorbeelden, en finetunet. Wat gebeurt er als het model een instructie beantwoordt?",
        options={
            "error": "Het genereren stopt met een fout: het model heeft geen token om mee te eindigen",
            "nothing": "Er verandert niets: het leerde het end-of-text bij de pretraining",
            "right": "Het stopt niet vanzelf: het schrijft door na zijn verhaal",
        },
        correct="right",
        why={
            "error": "Er gaat niets kapot: het model heeft zijn end-of-text-token nog. Het leerde alleen nooit dat na een antwoord te schrijven, dus gaat de loop door tot zijn grens.",
            "nothing": f"Voor SFT stopte het model van de course vanzelf bij {C.MEASURED['yours']['before']['stopped']} van de {C.MEASURED['yours']['before']['total']} antwoorden. Na SFT zonder end-of-text stopte het bij {C.MEASURED['no-end']['stopped']} van de {C.MEASURED['no-end']['total']}: geen enkel voorbeeld liet het een einde na een antwoord zien.",
            "right": "Het model leert stoppen van de voorbeelden: elk antwoord dat eindigt met het end-of-text, met zijn label meegeteld, leert het waar een beurt voorbij is.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het finetunen uit [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl), nu van een language model, met dezelfde stap als je pretraining in [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl); de generation prompt en het end-of-turn-token uit [Chat templates](/course/open/llm-chat-template?lang=nl), nu in een eigen template; en een check die een programma kan doen, van wat er gevraagd werd. Je project `llm` heeft nu `sft_example.py`, `sft_batch.py` en `words_used.py`, en na de demo je gefinetunede model in `sft.pt`. Hierna finetunet LoRA op dezelfde manier met veel minder getallen om te leren, en gebruikt DPO je `words_used` om een beter antwoord van een slechter te onderscheiden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_example, result_batch, result_words, choice_end):
    L.wrap_up(result_example, result_batch, result_words, choice_end)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
