# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-pretrained-features. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-pretrained-features", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl) en over [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_freeze():
    quick_freeze = L.choice(
        "quick-freeze",
        question="Je bevriest elke parameter van de cijferlezer van de course met `p.requires_grad = False`, en geeft hem daarna een nieuwe laatste layer: `model.out = nn.Linear(64, 10)`. Wat leert er als je hem traint?",
        options={
            "right": "Alleen de nieuwe laatste layer: 650 getallen",
            "nothing": "Niets: je hebt elke parameter bevroren",
            "all": "Elke layer: de nieuwe layer brengt de gradients terug",
        },
        correct="right",
        why={
            "right": "Een nieuwe `nn.Linear` heeft `requires_grad` aan, en de oude layers blijven bevroren: alleen zijn 64 · 10 + 10 = 650 getallen leren. Vandaag is het bevroren deel een veel groter model.",
            "nothing": "Je bevroor elke parameter die het model op dat moment had. De nieuwe layer kwam daarna, dus die leert. Zie [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl).",
            "all": "De bevroren parameters krijgen nog steeds geen gradient, wat er ook na komt: alleen de nieuwe layer leert. Zie [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl).",
        },
    )
    quick_freeze
    return (quick_freeze,)


@app.cell(hide_code=True)
def quick_check_close():
    quick_close = L.choice(
        "quick-close",
        question="Een model geeft drie foto's, A, B en C, elk een embedding. A en B hebben een cosine similarity van 0,95, A en C een van −0,1. Wat zegt dat?",
        options={
            "right": "Voor het model lijken A en B op elkaar, en A en C niet",
            "same": "A en B zijn dezelfde foto",
            "length": "De embedding van A is langer dan die van C",
        },
        correct="right",
        why={
            "right": "Dicht bij 1 betekent dat de twee embeddings dezelfde kant op wijzen: voor het model lijken ze op elkaar. Vandaag komen de embeddings van een model dat van 142 miljoen foto's leerde.",
            "same": "0,95 betekent dat ze op elkaar lijken, niet dat ze hetzelfde zijn: twee plaatjes van hetzelfde cijfer hadden gemiddeld 0,95. Zie [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
            "length": "Cosine similarity kijkt alleen naar richtingen; de lengtes vallen weg. Zie [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
        },
    )
    quick_close
    return (quick_close,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je wilt een model dat jouw eigen soorten dingen uit elkaar houdt, zeg tien soorten voorwerpen, en je hebt van elke soort maar een paar foto's. ResNet-18 uit de vorige les kent 1000 klassen, maar dat zijn die van ImageNet: er zitten libel, zeester, nautilus, haan en schenkkan (onder de naam pitcher) bij, en geen zonnebloem, lotus, waterlelie, kroonluchter of windsorstoel. Een netwerk dat je vanaf nul traint op een paar foto's per klasse, leert ze uit het hoofd. Wat je nodig hebt, is een model dat al ziet wat er op een foto staat, wat het ook is, zodat alleen een klein laatste deel jouw klassen hoeft te leren.

    **Een voorbeeld uit het dagelijks leven.** Iemand die duizenden vogels heeft bekeken, ziet een nieuwe soort één keer, in een boek, en herkent hem daarna in de tuin. Die leert niet opnieuw kijken: die weet al waar je op let, de snavel, de vleugels, de kleuren, en leert alleen welke combinatie daarvan deze vogel heeft.
    """)
    return


@app.cell(hide_code=True)
def predict_one():
    guess_one = L.guess(
        "one",
        question="DINOv2, een groot beeldmodel, maakt van elke foto 384 getallen. Een lineaire layer leert daarop 10 klassen van maar 1 foto per klasse: 10 foto's in totaal. Hoeveel van 100 andere foto's van dezelfde klassen haalt hij goed?",
        options={
            "chance": "Ongeveer 10: evenveel als gokken",
            "third": "Ongeveer 35",
            "two-thirds": "Ongeveer 65",
            "almost": "Ongeveer 95",
        },
        reveal=f"**Ongeveer 95: {C.REFERENCE['features'][1]} van de 100,** en tussen {C.REFERENCE['seeds'][0]} en {C.REFERENCE['seeds'][1]} over 5 seeds. Dezelfde lineaire layer op de pixels van de foto's, {C.number(C.PIXELS, 'nl')} getallen per foto, haalt er {C.REFERENCE['pixels'][1]}. Van één zonnebloem valt weinig te leren; maar op 384 getallen die al beschrijven wat er op een foto staat, is één foto per klasse bijna genoeg. Vier van zijn vijf fouten zien een lotus aan voor een waterlelie: twee waterbloemen die op elkaar lijken.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een vision foundation model** is een groot netwerk dat getraind is op een enorme, gevarieerde verzameling foto's, niet voor één taak, maar zodat wat het vanbinnen berekent voor bijna elke taak bruikbaar is. DINOv2, dat Meta in 2023 uitbracht, is er een. Het leerde van 142 miljoen foto's, die zijn makers automatisch kozen uit 1,2 miljard foto's op het web, en zonder labels: met self-supervised learning, zoals in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl). Grof gezegd leerde het om twee verschillende uitsneden van dezelfde foto dezelfde features te geven. DINOv2-small, de kleinste versie, heeft 22 miljoen parameters en maakt van een foto van 224 bij 224 pixels 384 getallen: zijn features, een embedding van de foto zoals in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl). Het is een Vision Transformer, een soort netwerk dat latere lessen bouwen; hier is het een model dat je gebruikt.

        **Een linear probe** is één lineaire layer die getraind is op de bevroren features van een model: `nn.Linear(384, k)` voor k klassen. Het is de nieuwe laatste layer uit [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl), met alles ervoor bevroren. De naam zegt waar hij voor is: hij peilt wat er in de features zit. Een lineaire layer kan alleen platte vlakken tussen klassen leggen. Dus als een linear probe de klassen uit elkaar houdt, zit in de features wat ze onderscheidt, en wel zo eenvoudig: in de features zijn de klassen lineair scheidbaar, zoals in [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl).

        **Features één keer.** In DINOv2 leert niets, dus een foto krijgt altijd dezelfde 384 getallen. Je rekent ze één keer uit voor elke foto, `dino(prepare(photo))[0]`, en zet ze samen in een tensor met shape `(n, 384)`, een rij per foto. De probe traint op die tensor: een epoch is dan één kleine matrixvermenigvuldiging in plaats van 22 miljoen parameters die voor elke foto aan het werk gaan, en 100 epochs duren een fractie van een seconde. `prepare` komt uit `everything_ai.dinov2`: de vijf stappen van [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl), maar bicubisch verkleind, een gladdere manier dan bilineair, zoals de instellingen vragen die bij DINOv2 op Hugging Face horen.

        **Twee downloads.** DINOv2-small is 88 MB, van Hugging Face, open onder de licentie Apache 2.0. Caltech-101 is een verzameling foto's van voorwerpen in 101 klassen, in 2003 van het web verzameld: 137 MB, uit de eigen dataopslag van Caltech, onder de licentie CC BY 4.0, maar het auteursrecht op elke foto ligt bij de fotograaf. Daarom levert de course de foto's niet mee, en deelt hij geen model dat erop getraind is. Hij laat de twee klassen met gezichten weg, en gebruikt 10 klassen van 30 foto's die hij één voor één bekeken heeft, zonder mensen: kroonluchter, libel, schenkkan, lotus, nautilus, haan, zeester, zonnebloem, waterlelie en windsorstoel (een houten stoel met een rugleuning van dunne spijlen). De knop in het uitgewerkte voorbeeld downloadt ze allebei, één keer.
        """),
        L.watch_out("De makers van DINOv2 kozen zijn foto's zo dat ze lijken op die van bekende fotoverzamelingen, en ze testten het ook op Caltech-101: het kan dus best foto's gezien hebben die erg op deze lijken, zonder hun labels. Op foto's die op niets van het web lijken, zoals röntgenfoto's of foto's door een microscoop, kan een probe op zijn features veel slechter werken. Test hem op foto's van je eigen taak."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **vision foundation model** | een groot netwerk dat getraind is op een enorme, gevarieerde verzameling foto's, vaak zonder labels, en waarvan de features voor veel taken bruikbaar zijn | |
        | **linear probe** | één lineaire layer die getraind is op de bevroren features van een model; hoe goed hij het doet, laat zien wat er in de features zit | `nn.Linear(384, k)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_probe():
    mo.vstack([
        mo.Html(C.probe_picture({"title": "Een linear probe op de features van DINOv2", "photo": "een foto, klaargemaakt: 3 × 224 × 224 getallen",
                                 "dino": "DINOv2-small", "dino_note": "22 miljoen parameters, één keer per foto",
                                 "features": "384 features: de embedding van de foto",
                                 "probe": "linear probe: nn.Linear(384, 10)", "probe_note": "384 · 10 + 10 = 3850 getallen",
                                 "logits": "10 logits, één per klasse", "frozen": "bevroren", "learns": "leert",
                                 "alt": "Van boven naar beneden: een klaargemaakte foto gaat DINOv2-small in, dat bevroren is en één keer per foto draait; dat geeft 384 features, de embedding van de foto; de linear probe, het enige deel dat leert, 3850 getallen, maakt er 10 logits van, één per klasse."})),
        mo.md(r"""
        DINOv2 draait één keer per foto en leert niets; de probe leert van de features, en is het enige dat leert: 3850 getallen, tegen de 22 miljoen van DINOv2.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(f"""
    Met de schuif kies je van hoeveel foto's per klasse de probe leert, van 1 tot 20. Je ziet hoeveel van de 100 testfoto's hij dan goed haalt op de features van DINOv2, en hoeveel dezelfde probe goed haalt op de pixels van de klaargemaakte foto's, {C.number(C.PIXELS, 'nl')} getallen per foto.
    """)
    return


@app.cell(hide_code=True)
def explore_n_cell():
    def _say(n):
        _n = int(n)
        _f, _p, _mixed = C.REFERENCE["features"][_n], C.REFERENCE["pixels"][_n], C.REFERENCE["mixed"][_n]
        _photos = "1 foto" if _n == 1 else f"{_n} foto's"
        _wrong = 100 - _f
        _of = f"alle {_wrong}" if _mixed == _wrong else f"{_mixed} van de {_wrong}"
        return (f"Met {_photos} per klasse, {10 * _n} in totaal: op de features van DINOv2 haalt de probe {_f} van de 100 testfoto's goed, "
                f"op de pixels {_p}. Op de features halen {_of} fouten lotus en waterlelie door elkaar.")

    def _frame(n):
        _html = C.curve(int(n), {"title": "Testfoto's goed, van de 100", "x": "foto's per klasse om van te leren", "y": "goed",
                                 "features": "features van DINOv2", "pixels": "pixels", "guess": "gokken",
                                 "alt": "Twee lijnen over het aantal foto's per klasse, 1 tot 20: op de features van DINOv2 haalt de probe 94 tot 98 van de 100 testfoto's goed, op de pixels 20 tot 34; gokken zou 10 geven."}, "nl")
        return _html, _say(n)

    explore_n = L.explore("n", "foto's per klasse", _frame, start=1, values=C.PER_CLASS, role="",
                          task="Zoek het kleinste aantal foto's per klasse waarbij de probe op de pixels meer dan 33 testfoto's goed haalt.")
    explore_n
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Op de features van DINOv2 geeft één foto per klasse al 95, en 20 foto's geven 98: meer foto's helpen nauwelijks, want wat overblijft zijn vooral lotussen en waterlelies, die op elkaar lijken. De aantallen gaan een beetje op en neer, omdat elke nieuwe foto een makkelijke of een moeilijke kan zijn. Op de pixels heeft de probe 10 foto's per klasse nodig om er meer dan 33 goed te halen, en hij komt nooit boven de 34: beter dan gokken, 10, maar ver onder de features. Waarom de pixels zo ver achterblijven, is de vraag van oefening 3.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo leer je een bevroren beeldmodel nieuwe klassen met een linear probe:

    1. Laad het model, `dino`, in eval mode, en laat het rekenen binnen `with torch.no_grad():`.
    2. Kies foto's om van te leren en andere foto's om op te testen, en geef elke foto zijn klasse als heel getal van 0 tot en met k − 1: zijn label.
    3. Reken de features van elke foto één keer uit: `dino(prepare(photo))[0].numpy()`, 384 getallen, en zet de rijen samen in een tensor met shape `(n, 384)`.
    4. Train een linear probe op de features van de trainingsfoto's: `nn.Linear(384, k)`, `nn.CrossEntropyLoss()` en `torch.optim.SGD`, zoals in [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).
    5. Test hem op de features van de testfoto's: hoeveel haalt hij goed?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt de eerste 5 foto's van elk van de 10 klassen om van te leren, en foto 20 tot en met 29 van elke klasse om op te testen: 50 en 100 foto's. `C.CLASSES` bevat de namen van de klassen, en `C.photos(name)` de 30 foto's van een klasse; het label van een foto is de plaats van zijn klasse in `C.CLASSES`. `features(photos)` is stap 3 voor een lijst foto's: de features van elke foto als rij in NumPy, en alle rijen als één tensor. De cel print de shapes van de features, en hoeveel trainings- en testfoto's de probe na 100 epochs goed haalt. De features van 150 foto's kosten een paar seconden; de 100 epochs minder dan één. `dino` is DINOv2-small, dat de cel hieronder laadt zodra beide downloads klaar zijn.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="DINOv2-small en Caltech-101 downloaden, 225 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dino_cell(get_button):
    dino = C.dino_or_stop(get_button, {
        "ask": "*DINOv2-small en de foto's van Caltech-101 staan nog niet op je computer. De knop downloadt ze één keer: DINOv2-small van Hugging Face, 88 MB, en Caltech-101 van Caltech, 137 MB, waarvan hij daarna de foto's uitpakt. Het uitgewerkte voorbeeld en de demo hebben ze nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**{name} downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van Caltech-101 uitpakken** · {done} foto's",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face of Caltech was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 225 MB en de uitgepakte foto's. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*DINOv2-small en de foto's staan op je computer: 22 miljoen parameters, die van 142 miljoen foto's leerden zonder labels.*",
    })
    return (dino,)


@app.cell
def example_probe(dino):
    import numpy as np
    import torch
    from torch import nn
    from everything_ai.dinov2 import prepare

    def features(photos):
        rows = []
        with torch.no_grad():
            for photo in photos:
                rows.append(dino(prepare(photo))[0].numpy())
        return torch.tensor(np.array(rows))

    # 2. 5 foto's per klasse om van te leren, en 10 andere om op te testen, met hun labels
    train, y_train, test, y_test = [], [], [], []
    for label in range(10):
        photos = C.photos(C.CLASSES[label])
        for photo in photos[:5]:
            train.append(photo)
            y_train.append(label)
        for photo in photos[20:30]:
            test.append(photo)
            y_test.append(label)
    # 3. de features van elke foto, één keer
    X_train, X_test = features(train), features(test)
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
    print(X_train.shape, X_test.shape)
    # 4. een linear probe van de 384 features naar de 10 klassen, getraind op de features
    torch.manual_seed(0)
    probe = nn.Linear(384, 10)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(probe.parameters(), lr=0.1)
    for epoch in range(100):
        loss = loss_fn(probe(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 5. test hem
    with torch.no_grad():
        print((probe(X_train).argmax(dim=1) == y_train).sum().item(), "van de 50 trainingsfoto's goed")
        print((probe(X_test).argmax(dim=1) == y_test).sum().item(), "van de 100 testfoto's goed")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De features hebben shape `(50, 384)` en `(100, 384)`: een rij van 384 getallen per foto. Na 100 epochs haalt de probe alle 50 trainingsfoto's goed, en zijn loss is bijna 0. Dat alleen zegt weinig: een layer van 3850 getallen kan 50 foto's uit het hoofd leren. Waar het om gaat, zijn de 100 testfoto's die hij nooit zag, en daarvan haalt hij er 97 goed. De drie die hij fout heeft, zijn waterlelies die hij aanziet voor een lotus.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_params():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat leert er
    """)
    return


@app.cell(hide_code=True)
def choice_params_cell():
    choice_params = L.choice(
        "params",
        question="Een linear probe op de 384 features van DINOv2-small, voor alle 99 klassen die de course van Caltech-101 houdt. Hoeveel getallen leren er?",
        options={
            "right": "38.115",
            "no-bias": "38.016",
            "all": "22 miljoen en 38.115",
            "sum": "483",
        },
        correct="right",
        why={
            "right": "384 · 99 weights, één per feature en klasse, en 99 biases, één per klasse: 38.016 + 99 = 38.115. De 22 miljoen parameters van DINOv2 blijven bevroren.",
            "no-bias": "Dat zijn de weights, 384 · 99. Elke klasse heeft ook een bias: 38.016 + 99 = 38.115.",
            "all": "DINOv2 is bevroren: zijn 22 miljoen parameters krijgen geen gradient. Alleen de probe leert: 384 · 99 + 99 = 38.115.",
            "sum": "Een lineaire layer heeft een weight voor elk paar van een feature en een klasse, 384 · 99, en een bias per klasse: 38.115.",
        },
    )
    choice_params
    return (choice_params,)


@app.cell(hide_code=True)
def text_exercise_probe():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een linear probe trainen

        Maak `train_probe(X, y, k)` af: stap 4 van het recept, voor welke features en welk aantal klassen ook. `X` heeft een rij features per foto, `y` het label van elke foto, een heel getal van 0 tot en met `k` − 1, en `k` is het aantal klassen. Vervang de twee `...`; de functie geeft de getrainde probe terug. De check verzint features, 4 per foto voor 3 klassen, roept `train_probe(X, y, 3)` aan, en kijkt naar de vorm van de probe en naar de klassen die hij 8 testpunten geeft. Hij probeert ook andere aantallen features en klassen.
        """),
        L.task("Laat `train_probe(X, y, k)` een linear probe trainen van de features in `X` naar `k` klassen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probe():
    probe_code = L.code_box("probe", lines=14)
    probe_code
    return (probe_code,)


@app.cell(hide_code=True)
def check_probe(probe_code):
    result_probe = L.check("probe", probe_code.value)
    result_probe
    return (result_probe,)


@app.cell(hide_code=True)
def hints_probe():
    L.hints(
        "probe",
        [
            "Stap 4 van het uitgewerkte voorbeeld, met het aantal features en het aantal klassen uit `X` en `k`.",
            "Het aantal features is de lengte van een rij van `X`: `X.shape[1]`. De loss krijgt de logits van de probe voor alle foto's, en hun labels.",
            "`probe = nn.Linear(X.shape[___], k)` en `loss = loss_fn(probe(___), y)`.",
        ],
        why="`nn.Linear(X.shape[1], k)` neemt de features van één foto en geeft `k` logits. `loss_fn(probe(X), y)` is de cross-entropy over alle foto's tegelijk, en elke stap van SGD verschuift de weights en biases van de probe een beetje, terwijl de features blijven zoals ze zijn.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw probe op de foto's

    Zodra DINOv2 en de foto's gedownload zijn, in het uitgewerkte voorbeeld, geeft de cel hieronder de features van DINOv2 aan jouw `train_probe` zodra die klopt, en anders aan die van de course. Voor elk aantal foto's per klasse van de schuif traint hij een probe op de eerste foto's van elke klasse en test hij die op de 100 testfoto's. De features van alle 300 foto's kosten een paar seconden, één keer.
    """)
    return


@app.cell(hide_code=True)
def demo_probe(dino, probe_code, result_probe):
    _d = C.demo(dino, probe_code.value if result_probe.state == "correct" else None)
    _who = "jouw `train_probe`" if _d["yours"] else "de `train_probe` van de course"
    _rows = "\n".join(f"| {_n} | {_d['right'][_n]} | {C.REFERENCE['features'][_n]} |" for _n in C.PER_CLASS)
    mo.md(f"""Met {_who}, op de features van DINOv2:

| foto's per klasse | testfoto's goed, van de 100 | gemeten door de course |
|---|---|---|
{_rows}""")
    return


@app.cell(hide_code=True)
def text_exercise_pixels():
    mo.md(r"""
    ### Oefening 3. Een probe op de pixels
    """)
    return


@app.cell(hide_code=True)
def choice_pixels_cell():
    choice_pixels = L.choice(
        "pixels",
        question=f"Geef dezelfde probe de pixels van elke klaargemaakte foto in plaats van zijn features: 3 · 224 · 224 = {C.number(C.PIXELS, 'nl')} getallen, dus `nn.Linear({C.PIXELS}, 10)`. Met 5 foto's per klasse gaat ook zijn training loss naar 0: hij haalt alle 50 trainingsfoto's goed. Toch haalt hij maar {C.REFERENCE['pixels'][5]} van de 100 testfoto's goed, tegen {C.REFERENCE['features'][5]} op de features. Waarom?",
        options={
            "short": "Hij trainde te kort: met meer epochs haalt hij ook 97",
            "right": "Geen enkele pixel zegt wat er op een foto staat: de probe leert zijn 50 foto's uit het hoofd",
            "less": f"De pixels bevatten minder dan de features: {C.number(C.PIXELS, 'nl')} getallen zeggen minder dan 384",
            "small": "De layer is te klein voor zoveel getallen",
        },
        correct="right",
        why={
            "short": "Zijn training loss is al 0: hij haalt elke trainingsfoto goed. Meer epochs maken hem alleen zekerder van die 50, niet beter op nieuwe foto's: wat hij leerde, geldt alleen voor die 50, zoals in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "right": f"Een pixel zegt alleen hoe licht één plekje is. Of er een zonnebloem op de foto staat, zegt geen enkele pixel, en de zonnebloem kan overal in beeld staan, in elk licht. Met {C.number(C.PIXELS * 10 + 10, 'nl')} getallen voor 50 foto's vindt de probe in elke trainingsfoto wel iets wat die foto onderscheidt, en leert hij de 50 uit het hoofd, zoals in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl). De features van DINOv2 beschrijven al wat er op een foto staat, dus daar gelden de platte vlakken die de probe tussen de klassen legt ook voor nieuwe foto's.",
            "less": "DINOv2 berekent zijn 384 features uit precies deze pixels, dus alles wat in de features zit, zit ook in de pixels. Maar niet in een vorm die een lineaire layer kan gebruiken: geen enkele pixel zegt wat er op de foto staat, en het voorwerp kan overal staan. Dus leert de probe zijn 50 foto's uit het hoofd.",
            "small": f"Hij heeft {C.number(C.PIXELS * 10 + 10, 'nl')} getallen, tegen 3850 op de features, en hij haalt alle 50 trainingsfoto's goed. Hij heeft te veel ruimte, niet te weinig: hij leert de 50 uit het hoofd, zoals in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
        },
    )
    choice_pixels
    return (choice_pixels,)


@app.cell(hide_code=True)
def text_exercise_no_labels():
    mo.md(r"""
    ### Oefening 4. Klassen zonder labels
    """)
    return


@app.cell(hide_code=True)
def explain_no_labels_cell():
    explain_no_labels = L.explain(
        "no-labels",
        question="DINOv2 heeft nooit een label gezien, zelfs het woord zonnebloem niet. Toch houdt een linear probe op zijn features zonnebloemen, zeesterren en kroonluchters uit elkaar na één foto van elk. Hoe kunnen features die zonder labels geleerd zijn, toch klassen bevatten? Denk aan wat DINOv2 leerde doen, en aan wat twee foto's van hetzelfde soort ding gemeen hebben.",
        model_answer="**DINOv2 leerde twee verschillende uitsneden van dezelfde foto dezelfde features te geven,** dus zijn features kunnen niet afhangen van waar iets in beeld staat, hoe groot het is of hoe het belicht is: ze moeten beschrijven wat er op de foto staat, zijn vormen, structuren en onderdelen. **Twee foto's van hetzelfde soort ding hebben juist dat gemeen,** zoals de bloemblaadjes rond het donkere hart van een zonnebloem of de vijf armen van een zeester, dus hun features liggen dicht bij elkaar, en ver van die van andere soorten. **De klassen zitten als groepen in de features voordat iemand ze een naam geeft.** Een label zegt alleen welke groep welke naam krijgt, en daarom heeft de probe zo weinig foto's nodig om zijn platte vlakken ertussen te leggen.",
        points=[
            "je zegt dat DINOv2 leerde verschillende uitsneden van dezelfde foto dezelfde features te geven, zodat zijn features beschrijven wat er op een foto staat, niet waar of in welk licht",
            "je zegt dat foto's van hetzelfde soort ding gemeen hebben wat erop staat, zodat hun features dicht bij elkaar liggen",
            "je zegt dat de groepen er al zijn voor er een label is, zodat een paar foto's met een label genoeg zijn om de probe te vertellen welke groep welke is",
        ],
    )
    explain_no_labels
    return (explain_no_labels,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De bevroren feature extractor uit [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl) is nu een vision foundation model dat van 142 miljoen foto's leerde zonder labels, en zijn features zijn een embedding van een foto, zoals in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl). Een linear probe, één layer getraind met cross-entropy, laat zien wat er in die features zit: 95 van de 100 testfoto's goed na één foto per klasse, tegen 20 op de pixels. In de volgende les, *Foto's vinden die op elkaar lijken*, laat je de probe weg: de cosine similarity van de features van DINOv2 vindt de foto's die het meest op een gegeven foto lijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_params, choice_pixels, explain_no_labels, result_probe):
    L.wrap_up(choice_params, result_probe, choice_pixels, explain_no_labels)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
