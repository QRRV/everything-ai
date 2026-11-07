# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-try-classifier. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-try-classifier", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl) en over [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="De foto van de kat, 451 bij 300 pixels, gaat door de vijf stappen van de vorige les. Welke shape heeft de tensor die het model ingaat?",
        options={
            "right": "`(1, 3, 224, 224)`",
            "last": "`(1, 224, 224, 3)`",
            "nobatch": "`(3, 224, 224)`",
            "resized": "`(1, 3, 256, 384)`",
        },
        correct="right",
        why={
            "right": "Hoe groot de foto ook is, de center crop geeft 224 bij 224; `.permute(2, 0, 1)` zet de 3 kleuren vooraan, en `.unsqueeze(0)` maakt er een batch van één van.",
            "last": "Zo bewaart NumPy een beeld. Het model wil de kleuren vooraan: `.permute(2, 0, 1)` geeft `(3, 224, 224)`, en `.unsqueeze(0)` zet er een batch van één voor.",
            "nobatch": "Dat is één foto, maar het model neemt een batch: `.unsqueeze(0)` zet er een as van lengte 1 voor, `(1, 3, 224, 224)`.",
            "resized": "Dat is de foto na stap 1. Stap 2 knipt er het middelste vierkant van 224 bij 224 uit.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Een model geeft drie klassen de logits 2, 0 en 0. Welke kans geeft softmax aan de eerste klasse?",
        options={
            "right": "0,79",
            "all": "1",
            "even": "0,33",
            "logit": "2",
        },
        correct="right",
        why={
            "right": "e² / (e² + e⁰ + e⁰) = 7,39 / 9,39 = 0,79: de grootste logit krijgt de grootste kans, en de andere twee krijgen elk 0,11.",
            "all": "Softmax geeft elke klasse een deel: e tot de macht van elke logit, gedeeld door hun som. 7,39 / (7,39 + 1 + 1) = 0,79, en 0,11 voor elk van de andere.",
            "even": "De logits verschillen, dus de kansen ook: e² / (e² + e⁰ + e⁰) = 0,79 voor de eerste klasse.",
            "logit": "Een kans ligt tussen 0 en 1. Softmax neemt e tot de macht van elke logit en deelt door hun som: 7,39 / 9,39 = 0,79.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een pretrained beeldclassifier herkent duizend soorten dingen op een foto, zonder dat je zelf iets traint: een foto klaargemaakt zoals in de vorige les, een paar regels code, en je ziet wat hij denkt. Maar wat betekent het als hij "goed" is? Op een foto van een kopje espresso staat ook een kopje, en één kat past soms bij meerdere klassen die op elkaar lijken. Daarom worden classifiers op twee manieren beoordeeld: op hun beste gok, en op hun vijf beste gokken.

    **Een voorbeeld uit het dagelijks leven.** Een vriend hoort een vogel in de tuin en zegt: "Een merel, of anders een zanglijster, of een roodborst." Het was een zanglijster. De eerste gok is fout, maar de goede vogel staat op het korte lijstje, en dat zegt meer over hoe goed je vriend vogels kent dan alleen "fout".
    """)
    return


@app.cell(hide_code=True)
def predict_cat():
    guess_cat = L.guess(
        "cat",
        question="Een classifier die op ImageNet is getraind, met 1000 klassen, kijkt naar de foto van de kat uit de vorige les. Zijn beste gok is een klasse van katten. Hoe zeker is hij van die gok?",
        options={
            "sure": "Meer dan 90 procent",
            "forty": "Ongeveer 40 procent",
            "ten": "Ongeveer 10 procent",
            "tiny": "Minder dan 1 procent",
        },
        reveal="**Ongeveer 40 procent.** Zijn beste gok is Egyptian cat, met 0,39; tiger cat volgt, ook met 0,39, en tabby met 0,22. Samen hebben deze drie 99,5 procent: het model twijfelt er niet aan dat het een kat ziet, alleen aan welke van drie klassen voor huiskatten in ImageNet het is. Die klassen lijken op elkaar, dus een beste gok van maar 40 procent zegt niet dat het model slecht is.",
    )
    guess_cat
    return (guess_cat,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **ImageNet** is een verzameling foto's van het web, waarbij mensen elke foto één klasse uit 1000 hebben gegeven: van tench (zeelt) en goldfish tot toilet tissue, met 118 hondenrassen ertussen. De klassen hebben Engelse namen, en die houdt deze les zoals ze zijn. De versie waarop de meeste beeldclassifiers worden getraind en vergeleken, heeft 1,28 miljoen trainingsfoto's en 50.000 validatiefoto's. Een classifier die erop is getraind, hier ResNet-18 met 11,7 miljoen parameters, geeft een foto een logit per klasse: 1000 getallen. Softmax maakt er kansen van die samen 1 zijn, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl). Hoe ResNet-18 zijn logits berekent, laten latere lessen zien; hier is het een model dat je gebruikt.

        **Top-1 en top-5.** De klasse met de hoogste kans is de beste gok van het model, zijn top-1; de vijf klassen met de hoogste kansen zijn zijn top-5. **Top-k accuracy** is het deel van de foto's waarbij het label bij hun k beste gokken zit. Top-1 accuracy is de accuracy uit [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl); top-5 accuracy telt een foto ook als goed wanneer het label op het korte lijstje van het model staat. Voor ImageNet worden ze allebei gemeld, omdat klassen op elkaar lijken, er op een foto vaak meer dan één ding staat, en elke foto maar één label heeft. Op de 50.000 validatiefoto's van ImageNet mat torchvision voor deze ResNet-18 een top-1 accuracy van 69,8 procent, en een top-5 accuracy van 89,1 procent.

        **In code.** Het model geeft de logits als een tensor met shape `(1, 1000)`. `logits[0].numpy()` maakt van de eerste rij een NumPy-array, zodat softmax werkt zoals je hem kent: `np.exp(z - z.max())`, gedeeld door de som ervan. Dan paren: `(p[i], names[i])` voor elke klasse. `sorted` sorteert paren op hun eerste waarde, de kans, en `reverse=True` draait de volgorde om, van groot naar klein, zodat `[:5]` de vijf beste gokken pakt.

        **Eén keer downloaden.** Het bestand van ResNet-18 is 47 MB. De knop in het uitgewerkte voorbeeld downloadt het één keer van Hugging Face, van `timm`, een bibliotheek met beeldmodellen die een kopie van de eigen weights van torchvision bewaart, in een vaste versie, en controleert dat elke byte klopt met het vastgelegde bestand van de course. De code van ResNet-18 is open, onder een BSD-licentie, maar zijn weights zijn geleerd van de foto's van ImageNet, waarvan de rechten bij hun fotografen liggen, dus of de weights doorgegeven mogen worden, is onduidelijk. Daarom levert de course ze niet mee, en blijven ze op je eigen computer.
        """),
        L.watch_out("De kansen zijn altijd samen 1, ook voor een foto van iets dat in geen van de 1000 klassen past. Een hoge kans betekent dat het model deze klasse verkiest boven de andere 999, niet dat die klasse klopt."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **ImageNet** | een verzameling foto's met bij elke foto één klasse uit 1000; de foto's waarop de meeste beeldclassifiers worden getraind en vergeleken | |
        | **top-1, top-5** | de klasse met de hoogste kans; de vijf klassen met de hoogste kansen | `sorted(pairs, reverse=True)[:5]` |
        | **top-k accuracy** | het deel van de foto's waarbij het label bij hun k beste gokken zit | |
        | **`reverse=True`** | laat `sorted` van groot naar klein gaan | `sorted([0.2, 0.5, 0.3], reverse=True)` geeft `[0.5, 0.3, 0.2]` |
        | **`.numpy()`** | maakt van een tensor een NumPy-array, met dezelfde getallen | `logits[0].numpy()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_bars():
    _sep = ","
    mo.vstack([
        mo.md("**De vijf beste gokken van ResNet-18**"),
        mo.hstack([
            mo.Html(C.bars("coffee", 5, {"title": "De koffiefoto", "alt": "Vijf balken, de beste gok eerst: " + ", ".join(f"{_n} {C.shown(_p, _sep)}" for _n, _p in C.REFERENCE["coffee"][:5]) + "."}, _sep)),
            mo.Html(C.bars("cat", 5, {"title": "De kattenfoto", "alt": "Vijf balken, de beste gok eerst: " + ", ".join(f"{_n} {C.shown(_p, _sep)}" for _n, _p in C.REFERENCE["cat"][:5]) + "."}, _sep)),
        ], wrap=True, justify="start", gap=1.5),
        mo.md(r"""
        Bij de koffiefoto heeft espresso 0,85: goed en zeker. Zijn andere gokken zijn bruine dranken en gerechten, en een kopje, dat ook op de foto staat. Bij de kattenfoto staan drie klassen van katten bijna naast elkaar, en lynx, een wilde kat, ver daarachter. De kansen zijn afgerond op twee decimalen, en onder 0,01 zeggen de balken dat: lynx heeft 0,0045, Persian cat 0,0002.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Geef de kattenfoto het label tabby, zoals iemand die foto's labelt voor ImageNet zou kunnen doen. Met de slider kies je k; je ziet de k beste gokken, en of tabby erbij zit.
    """)
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _say(k):
        _k = int(k)
        _in = C.rank("cat") <= _k
        _held = C.held("cat", _k)
        _part = "meer dan 99,9" if _held >= 0.9995 else C.percent(100 * _held, ",")
        if _in:
            _verdict = f" Tabby zit erbij, op plaats {C.rank('cat')}: voor top-{_k} telt de kattenfoto als goed."
        elif _k == 1:
            _verdict = " Tabby is niet de beste gok: voor top-1 telt de kattenfoto als fout."
        else:
            _verdict = f" Tabby zit er niet bij: voor top-{_k} telt de kattenfoto als fout."
        _start = "De beste gok heeft" if _k == 1 else f"De {_k} beste gokken hebben samen"
        return f"{_start} {_part} procent van de kans." + _verdict

    def _frame(k):
        _html = C.bars("cat", int(k), {"title": f"De top-{int(k)} van de kattenfoto", "alt": f"De {int(k)} beste gokken van ResNet-18 voor de kattenfoto, als balken; tabby, het label, heeft een vinkje als het erbij zit."}, ",", mark="tabby")
        return _html, _say(k)

    explore_k = L.explore("k", "k", _frame, start=1, values=C.KS, role="",
                          task="Zoek de kleinste k waarbij de kattenfoto als goed telt, met tabby als label.")
    explore_k
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Met tabby als label is de kattenfoto fout voor top-1 en top-2, en goed vanaf top-3. De drie klassen van huiskatten hebben samen 99,5 procent: het model weet zeker dat het een kat ziet, en twijfelt welke van de drie klassen van ImageNet het is. Top-5 accuracy vergeeft dat soort twijfel; top-1 accuracy niet. Vanaf k = 5 groeit het deel nauwelijks nog: de nieuwe balken zijn bijna leeg, omdat het laatste stukje van een procent dun verdeeld is over de andere 996 klassen.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een ImageNet-classifier `model` naar een foto te laten kijken:

    1. Maak de foto klaar zoals in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl): `x` met shape `(1, 3, 224, 224)`.
    2. De logits: `logits = model(x)`, binnen `with torch.no_grad():`, met shape `(1, 1000)`.
    3. De kansen: `z = logits[0].numpy()`, `e = np.exp(z - z.max())` en `p = e / e.sum()`.
    4. De top-1: `names[np.argmax(p)]`. De top-5: `sorted([(p[i], names[i]) for i in range(len(names))], reverse=True)[:5]`.
    5. Beoordeel hem: een foto is goed voor top-k als zijn label bij zijn k beste gokken zit, `label in guesses[:k]`: `in` werkt op een lijst net als op de keys van een dictionary. Top-k accuracy is het deel van de foto's dat goed is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest de namen van de 1000 klassen, één per regel in het bestand van de course, en print hoeveel het er zijn en de naam met index 967. Hij maakt de koffiefoto klaar, stap 1, en print de shape van de logits, stap 2. Hij maakt de kansen, stap 3, en print hun som. Dan print hij de top-1 met zijn kans, en de top-5, stap 4. `resnet` is ResNet-18, dat de cel hieronder laadt zodra het is gedownload.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="ResNet-18 downloaden, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 staat nog niet op je computer. De knop downloadt het één keer, 47 MB, van Hugging Face. Het uitgewerkte voorbeeld heeft het nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**ResNet-18 downloaden** · {done} van {total} MB",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 47 MB. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*ResNet-18 staat op je computer: 11,7 miljoen parameters, met de weights van torchvision, getraind op ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_coffee(resnet):
    import numpy as np
    import torch
    from PIL import Image

    with open("data/imagenet-classes.txt") as f:
        names = [line.strip() for line in f]
    print(len(names), names[967])
    # 1. de vijf stappen van de vorige les
    photo = Image.open("data/photos/coffee.png")
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    crop = small[16:240, 80:304]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)
    # 2. de logits: één per klasse
    with torch.no_grad():
        logits = resnet(x)
    print(logits.shape)
    # 3. de kansen, met softmax; .round(3): op drie decimalen
    z = logits[0].numpy()
    e = np.exp(z - z.max())
    p = e / e.sum()
    print(p.sum().round(3))
    # 4. de top-1, en de top-5: paren van kans en naam, van groot naar klein
    print(names[np.argmax(p)], p.max().round(3))
    pairs = [(p[i], names[i]) for i in range(len(names))]
    for prob, name in sorted(pairs, reverse=True)[:5]:
        print(name, prob.round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het bestand heeft 1000 namen, en die met index 967 is espresso. De logits hebben shape `(1, 1000)`: een batch van één foto, een logit per klasse. De kansen zijn samen 1. De top-1 is espresso, met 0,852, en dat klopt. De andere vier van de top-5 zijn chocolate sauce, consomme (heldere bouillon), soup bowl en cup: bruine dranken en gerechten, en een kopje, dat ook op de foto staat. Met het label espresso is de koffiefoto goed voor top-1, en dus ook voor top-5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Goed of fout

    Het staande stuk van de kat uit de vorige les heeft het label tabby. De vijf beste gokken van ResNet-18 ervoor zijn, in deze volgorde: tiger cat, tabby, Egyptian cat, lynx en tiger.
    """)
    return


@app.cell(hide_code=True)
def choice_rank_cell():
    choice_rank = L.choice(
        "rank",
        question="Hoe telt het staande stuk?",
        options={
            "right": "Fout voor top-1, goed voor top-5",
            "both": "Goed voor top-1 en voor top-5",
            "neither": "Fout voor top-1 en voor top-5",
            "flip": "Goed voor top-1, fout voor top-5",
        },
        correct="right",
        why={
            "right": "Voor top-1 telt alleen de beste gok, tiger cat: niet het label. Voor top-5 moet het label bij de vijf beste gokken zitten, en tabby staat op plaats 2.",
            "both": "Voor top-1 telt alleen de beste gok, en dat is tiger cat, niet tabby. Tabby staat op plaats 2, dus het telt pas vanaf top-2.",
            "neither": "Voor top-5 hoeft het label alleen bij de vijf beste gokken te zitten, en tabby staat op plaats 2: goed voor top-5.",
            "flip": "Een foto die goed is voor top-1, is altijd ook goed voor top-5: de beste gok is een van de vijf beste. Hier is het andersom: de beste gok is tiger cat, en tabby staat op plaats 2.",
        },
    )
    choice_rank
    return (choice_rank,)


@app.cell(hide_code=True)
def text_exercise_topk():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De k beste gokken

        Maak `top_k(p, names, k)` af, die de `k` waarschijnlijkste klassen teruggeeft als paren van kans en naam, de beste gok eerst: stap 4 van het recept, voor elke `k`. Vervang de twee `...`. De check roept hem aan met de kansen van vier klassen van katten:

        ```python
        top_k(np.array([0.22, 0.39, 0.38, 0.01]), ["tabby", "tiger cat", "Egyptian cat", "lynx"], 2)
        ```

        Dat moet `[(0.39, 'tiger cat'), (0.38, 'Egyptian cat')]` geven. De check probeert ook andere aantallen klassen, en andere waarden van `k`.
        """),
        L.task("Laat `top_k(p, names, k)` de `k` beste gokken teruggeven, de beste eerst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_topk():
    topk_code = L.code_box("topk", lines=3)
    topk_code
    return (topk_code,)


@app.cell(hide_code=True)
def check_topk(topk_code):
    result_topk = L.check("topk", topk_code.value)
    result_topk
    return (result_topk,)


@app.cell(hide_code=True)
def hints_topk():
    L.hints(
        "topk",
        [
            "Stap 4 van het recept, met `k` in plaats van 5, en `len(names)` voor het aantal klassen.",
            "`pairs` krijgt een paar `(p[i], names[i])` voor elke `i` van 0 tot en met `len(names) - 1`: een list comprehension. Sorteer de paren dan van groot naar klein, en pak de eerste `k`.",
            "`pairs = [(p[i], names[i]) for i in range(___)]` en `return sorted(pairs, reverse=___)[:___]`.",
        ],
        why="`sorted` vergelijkt paren op hun eerste waarde, de kans, dus met `reverse=True` komt de waarschijnlijkste klasse eerst: `[(0.39, 'tiger cat'), (0.38, 'Egyptian cat'), (0.22, 'tabby'), (0.01, 'lynx')]`. `[:k]` pakt daarvan de eerste `k`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Top-k accuracy

        Vier foto's, met bij elke foto de vijf beste gokken van ResNet-18: de koffiefoto, de kattenfoto, het staande stuk van de kat uit de vorige les, en de tekening van fruit uit [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl). En het label van elke foto; voor de tekening orange, een van zijn vruchten waar ImageNet een klasse voor heeft.

        ```python
        guesses = [["espresso", "chocolate sauce", "consomme", "soup bowl", "cup"],
                   ["Egyptian cat", "tiger cat", "tabby", "lynx", "Persian cat"],
                   ["tiger cat", "tabby", "Egyptian cat", "lynx", "tiger"],
                   ["balloon", "ping-pong ball", "Granny Smith", "punching bag", "croquet ball"]]
        labels = ["espresso", "tabby", "tabby", "orange"]
        ```

        Maak `top_k_accuracy(guesses, labels, k)` af, die het deel van de foto's teruggeeft waarbij het label bij de eerste `k` gokken zit. Vervang de twee `...`. `top_k_accuracy(guesses, labels, 1)` moet `0.25` geven: alleen de koffiefoto is goed met zijn beste gok. De check probeert ook andere waarden van `k`, en andere foto's.
        """),
        L.task("Laat `top_k_accuracy(guesses, labels, k)` het deel van de foto's teruggeven dat goed is voor top-k."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    accuracy_code = L.code_box("accuracy", lines=6)
    accuracy_code
    return (accuracy_code,)


@app.cell(hide_code=True)
def check_accuracy(accuracy_code):
    result_accuracy = L.check("accuracy", accuracy_code.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Stap 5 van het recept. In de loop loopt `i` over de foto's: `labels[i]` is het label van foto `i`, en `guesses[i]` zijn gokken, de beste eerst.",
            "De eerste `k` gokken van foto `i` zijn de slice `guesses[i][:k]`, en `in` zegt of het label erbij zit, net als bij de keys van een dictionary. Tel de foto's waarvoor dat zo is, en deel door het aantal foto's.",
            "`if labels[i] in guesses[i][:___]:` met `right = right + 1` eronder, en `return right / len(___)`.",
        ],
        why="Met `k` 1 is alleen de koffiefoto goed: 1 van de 4, dus 0,25. Met `k` 2 telt het staande stuk van de kat ook mee, met tabby op plaats 2: 0,5. Vanaf `k` 3 telt ook de hele kat mee, 0,75; de tekening nooit, want orange staat niet eens in zijn top 5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_balloon():
    _sep = ","
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Een ballon

        ResNet-18 kijkt naar de tekening van fruit, klaargemaakt zoals in de vorige les, dus het model ziet alleen het middelste vierkant: twee rode appels en een deel van een derde, de sinaasappel, de halve limoen en een klein stukje van de citroen, op een geruit tafelkleed. Zijn beste gok is balloon, met 0,73; ping-pong ball volgt, en orange komt pas op plaats 8, met 0,001.
        """),
        mo.Html(C.bars("fruit", 5, {"title": "De tekening van fruit", "alt": "Vijf balken, de beste gok eerst: " + ", ".join(f"{_n} {C.shown(_p, _sep)}" for _n, _p in C.REFERENCE["fruit"][:5]) + "."}, _sep)),
    ])
    return


@app.cell(hide_code=True)
def explain_balloon_cell():
    explain_balloon = L.explain(
        "balloon",
        question="Waarom is het model zo zeker van iets dat zo fout is? Denk aan wat softmax doet met 1000 klassen, aan waar het model van leerde, en aan wat er in het vierkant staat dat het krijgt.",
        model_answer="**Het model moet kiezen uit zijn 1000 klassen, en softmax geeft altijd kansen die samen 1 zijn,** dus er is geen antwoord als \"geen van deze\": 0,73 betekent dat balloon beter past dan de andere 999 klassen, niet dat het klopt. **Het model leerde van foto's,** en de tekening is geen foto: gladde, glanzende bollen met een lichte vlek, op een eenvoudig patroon, lijken meer op de ballonnen en pingpongballetjes van ImageNet dan op zijn foto's van fruit. **En een classifier geeft het hele beeld één klasse,** terwijl er zelfs in het middelste vierkant dat het krijgt zes stukken fruit staan, sommige maar voor een deel, en ImageNet niet voor allemaal een klasse heeft: er zit geen rode appel en geen limoen bij zijn 1000 klassen.",
        points=[
            "je zegt dat softmax altijd kansen geeft die samen 1 zijn, dus dat het model moet kiezen uit zijn klassen, ook als er geen past",
            "je zegt dat het model van foto's leerde, en dat de tekening meer lijkt op de ballonnen en ballen daarin dan op hun fruit",
            "je zegt dat er in het vierkant meerdere stukken fruit staan, sommige zonder eigen klasse, terwijl het model één klasse geeft",
        ],
    )
    explain_balloon
    return (explain_balloon,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een foto die is klaargemaakt zoals in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl), gaat ResNet-18 in, dat een logit geeft voor elk van de 1000 klassen van ImageNet; softmax, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), maakt er kansen van, en sorteren geeft de beste gokken. Top-1 accuracy is de accuracy uit [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl); top-5 accuracy vergeeft ook een label op plaats 2 tot en met 5, en dat telt als klassen op elkaar lijken en er op een foto meer dan één ding staat. En de kansen zijn altijd samen 1, ook voor een tekening van fruit die een ballon wordt. In de volgende les, *Wat een beeldmodel ziet: features en een linear probe*, bekijk je de getallen die een groot pretrained beeldmodel voor een foto berekent, zijn features, en leer je daarmee eigen klassen uit een paar voorbeelden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rank, explain_balloon, result_accuracy, result_topk):
    L.wrap_up(choice_rank, result_topk, result_accuracy, explain_balloon)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
