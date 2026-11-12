# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-fine-tune-photos. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-fine-tune-photos", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl) en over [Data augmentation](/course/open/nn-data-augmentation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_phases():
    quick_phases = L.choice(
        "quick-phases",
        question="Bij fine-tuning laat fase 2 elke layer leren van de fouten van de laatste layer. Waarom komt fase 1, die alleen een nieuwe laatste layer traint, eerst?",
        options={
            "faster": "Een bevroren netwerk traint sneller, dus fase 1 bespaart tijd",
            "right": "Een nieuwe laatste layer begint met willekeurige weights: zijn eerste fouten zeggen niets over de taak, en zouden de geleende layers meetrekken",
            "small": "Zodat fase 2 een grotere learning rate kan gebruiken",
        },
        correct="right",
        why={
            "faster": "Fase 1 is snel, maar daarom komt hij niet eerst: zonder fase 1 las de cijferlezer 747 tot 844 van de 1.000 cijfers goed, en met fase 1 879 tot 893. Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
            "right": "Na fase 1 gaan de fouten van de laatste layer over de nieuwe taak, en daarvan leert fase 2. Vandaag zie je wat fase 2 nog te leren heeft als fase 1 al elke trainingsfoto goed heeft.",
            "small": "Andersom: fase 2 heeft een learning rate nodig die ongeveer tien keer zo klein is, zodat de geleende features maar een beetje veranderen. Zie [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
        },
    )
    quick_phases
    return (quick_phases,)


@app.cell(hide_code=True)
def quick_check_mirror():
    quick_mirror = L.choice(
        "quick-mirror",
        question="Welke verandering van een trainingsplaatje is een goede data augmentation voor een model dat handgeschreven cijfers leest?",
        options={
            "mirror": "Het links-rechts spiegelen",
            "right": "Het een of twee pixels verschuiven",
            "upside": "Het ondersteboven draaien",
        },
        correct="right",
        why={
            "mirror": "Een gespiegeld cijfer kan helemaal geen cijfer meer zijn, of een ander. Zie [Data augmentation](/course/open/nn-data-augmentation?lang=nl).",
            "right": "Een cijfer dat een pixel is verschoven, is nog steeds hetzelfde cijfer, dus het label blijft kloppen. Een links-rechts gespiegelde foto van een zonnebloem laat nog steeds een zonnebloem zien; vandaag laat je augmentation weg, om te zien wat fase 2 alleen doet.",
            "upside": "Ondersteboven wordt een 6 een 9. Zie [Data augmentation](/course/open/nn-data-augmentation?lang=nl).",
        },
    )
    quick_mirror
    return (quick_mirror,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt een paar honderd foto's van je eigen soorten dingen, en een classifier die op ImageNet is getraind. Fase 1, een nieuwe laatste layer op zijn bevroren features, is snel klaar, en heeft de meeste van je foto's al goed. De gebruikelijke volgende stap is elke layer laten leren: fase 2 van fine-tuning, op een netwerk met 11 miljoen parameters in plaats van de kleine cijferlezer. Dat kost echt rekenwerk: elke trainingsfoto gaat door elke layer, heen en terug, in elke epoch. Of het loont, kan alleen een meting zeggen op foto's waarop het model nooit trainde.

    **Een voorbeeld uit het dagelijks leven.** Stel, je bent een goede bloemenfotograaf en gaat werken voor een natuurtijdschrift. In een week leer je welke foto's de redactie wil. Daarna zou je ook anders kunnen gaan fotograferen. Voor bloemen in een tuin werkt je oude manier al, en levert oefenen weinig op. Voor vogels in de vlucht, waar je oog voor licht nog steeds helpt maar de timing nieuw is, loont oefenen wel.
    """)
    return


@app.cell(hide_code=True)
def predict_gain():
    _lo, _hi = C.low_high(C.REFERENCE["phase1"])
    _alo, _ahi = C.low_high(C.after(C.TUNE_LR))
    _glo, _ghi = C.low_high(C.gains())
    guess_gain = L.guess(
        "gain",
        question=f"Een nieuwe laatste layer op de bevroren features van ResNet-18 leert 10 soorten dingen uit 20 foto's per soort, en haalt {_lo} tot {_hi} van de 100 testfoto's goed, over 5 seeds. Daarna laat fase 2 alle 11 miljoen parameters van ResNet-18 leren, 2 epochs met een kleine learning rate. Hoeveel testfoto's haalt het model daarna goed?",
        options={
            "fewer": "Minder: 11 miljoen getallen leren 200 foto's uit het hoofd",
            "same": "Ongeveer evenveel: hooguit 2 meer",
            "more": "Duidelijk meer: ongeveer 5 meer",
            "all": "Bijna alle 100",
        },
        reveal=f"**Ongeveer evenveel: {_alo} tot {_ahi},** {C.gain_words(_glo, _ghi, 'nl')} dan na fase 1 bij elke seed. Fase 1 heeft alle 200 trainingsfoto's al goed, dus de loss waarvan fase 2 leert, ligt al dicht bij 0, en zijn gradients en stappen ook. Bijna alle fouten die overblijven, zijn een lotus die voor een waterlelie is aangezien, of andersom: twee bloemen die op elkaar lijken.",
    )
    guess_gain
    return (guess_gain,)


@app.cell(hide_code=True)
def text_idea():
    _tlo, _thi = C.low_high(C.REFERENCE["train-mode"])
    _alo, _ahi = C.low_high(C.after(C.TUNE_LR))
    mo.vstack([
        mo.md(rf"""
        ## Het idee

        **Full fine-tuning** is fine-tuning waarbij elke parameter van het model leert: fase 2 uit [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl), nu op ResNet-18. Met een nieuwe laatste layer voor 10 klassen zijn dat {C.number(C.PARAMS, 'nl')} getallen, tegen {C.number(C.HEAD, 'nl')} in fase 1. De naam zet het af tegen manieren om maar een deel van een model te finetunen; die komen in latere lessen. Het is op twee manieren duur. Bij elke stap krijgt elk van de 11 miljoen parameters een gradient, die de computer naast de parameter zelf bewaart: twee keer het geheugen van de parameters alleen, en meer met een optimizer als Adam, die voor elke parameter eigen getallen bijhoudt. En elke trainingsfoto gaat door elke layer, heen en terug, in elke epoch.

        **Fase 1 op features, één keer.** In fase 1 leert niets vóór de nieuwe laatste layer, zoals in [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl). Een foto krijgt dus altijd dezelfde 512 features, de getallen die de laatste layer van ResNet-18 krijgt. Met `model.fc = nn.Identity()` geeft het model die features in plaats van logits; je rekent ze één keer uit voor elke trainingsfoto, in batches van 50 binnen `with torch.no_grad():`, en traint de nieuwe laatste layer erop. Elk van de 100 epochs van die layer is dan een paar kleine matrixvermenigvuldigingen, in een oogwenk klaar.

        **Fase 2 in mini-batches.** Nu moet elke foto weer door het hele netwerk, in elke epoch, want elke layer verandert. Een `DataLoader` geeft de 200 trainingsfoto's in mini-batches van 20, in elke epoch in een nieuwe volgorde, zoals in [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl): de computer bewaart wat elke layer uitrekende voor 20 foto's tegelijk, voor de weg terug, in plaats van voor alle 200. De learning rate is 0,003. Tien keer zo klein als de 0,1 van fase 1 zou 0,01 zijn, maar dat ligt dicht bij waar fase 2 ontspoort, zoals de slider hieronder laat zien, dus de les blijft er ruim onder.

        **Eval mode blijft aan.** ResNet-18 heeft layers die in train mode anders rekenen, zoals dropout in [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl): batch normalization, die een latere les uitlegt. In eval mode houden ze de gemiddelden die ze op de foto's van ImageNet maten, in plaats van ze uit elke batch van 20 opnieuw te berekenen. De course houdt het model ook tijdens fase 2 in eval mode: in train mode haalde het {_tlo} tot {_thi} testfoto's goed, over 5 seeds, tegen {_alo} tot {_ahi}.
        """),
        L.watch_out("Beoordeel fase 2 op foto's waarop hij niet trainde: na fase 1 heeft ResNet-18 alle 200 trainingsfoto's al goed, dus die kunnen niet laten zien of fase 2 helpt. Maak keuzes zoals de learning rate op een validation set, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl); deze les kiest alleen op zijn 100 testfoto's omdat er bij 30 foto's per klasse weinig over is om apart te houden."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **full fine-tuning** | fine-tuning waarbij elke parameter van een model leert; daarvoor is voor elke parameter een gradient nodig, en moet elke foto heen en terug door elke layer | `for p in model.parameters(): p.requires_grad = True` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_phases():
    mo.vstack([
        mo.Html(C.phases_picture({"title": "Full fine-tuning van ResNet-18", "phase1": "fase 1", "rate1": "learning rate 0,1 · 100 epochs",
                                  "count1": f"{C.number(C.HEAD, 'nl')} getallen leren", "phase2": "fase 2",
                                  "rate2": "learning rate 0,003 · 2 epochs", "count2": f"{C.number(C.PARAMS, 'nl')} getallen leren",
                                  "fc": "fc (nieuw)", "learns": "leert", "frozen": "bevroren",
                                  "alt": f"Twee kolommen met de delen van ResNet-18 van boven naar beneden: conv1, layer1, layer2, layer3, layer4 en fc, de nieuwe laatste layer voor 10 klassen. In fase 1, met een learning rate van 0,1 en 100 epochs, is alles bevroren behalve fc: {C.number(C.HEAD, 'nl')} getallen leren. In fase 2, met een learning rate van 0,003 en 2 epochs, leert elk deel: {C.number(C.PARAMS, 'nl')} getallen."})),
        mo.md(r"""
        conv1 en layer1 tot en met layer4, groepen layers zoals ResNet-18 ze noemt, zijn het deel dat hij op ImageNet leerde; hoe ze rekenen, laten latere lessen zien. fc is nieuw. In fase 1 leert alleen fc, op features die één keer zijn uitgerekend; in fase 2 leert alles, en gaat elke foto er weer helemaal doorheen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Met de slider kies je de learning rate van fase 2, van 0,0003 tot 0,1. Voor elke seed van 0 tot en met 4 trainde de course fase 1, en daarna 2 epochs fase 2 met die learning rate. De balk loopt van de minste tot de meeste testfoto's goed over de 5 seeds; de grijze band is fase 1 alleen.
    """)
    return


@app.cell(hide_code=True)
def explore_lr_cell():
    def _say(lr):
        _lo, _hi = C.low_high(C.after(lr))
        _p1, _p2 = C.low_high(C.REFERENCE["phase1"])
        _text = (f"Met een learning rate van {C.rate(lr, 'nl')} haalt fase 2 na 2 epochs {_lo} tot {_hi} van de 100 testfoto's goed, "
                 f"over 5 seeds; fase 1 alleen haalde er {_p1} tot {_p2}.") if _lo < _hi else (
                 f"Met een learning rate van {C.rate(lr, 'nl')} haalt elke seed er na 2 epochs fase 2 {_lo} van de 100 testfoto's goed; "
                 f"fase 1 alleen haalde er {_p1} tot {_p2}.")
        if C.REFERENCE["classes"][lr] == 1:
            _text += " De getallen in het netwerk zijn `nan` geworden, en elke testfoto krijgt dezelfde klasse: evenveel goed als er testfoto's van die klasse zijn."
        return _text

    def _frame(lr):
        _p1, _p2 = C.low_high(C.REFERENCE["phase1"])
        _a, _b = C.low_high(C.after(0.01))
        _g1, _g2 = C.low_high(C.gains(0.01))
        _html = C.curve(lr, {"title": "Testfoto's goed na fase 2, van de 100", "x": "learning rate van fase 2", "y": "goed",
                             "phase1": "fase 1 alleen", "phase2": "na fase 2", "guess": "gokken",
                             "alt": f"Over de learning rate van fase 2, van {C.rate(C.RATES[0], 'nl')} tot {C.rate(C.RATES[-1], 'nl')}: tot en met 0,003 blijft fase 2 in of net boven de band van fase 1 alleen, {_p1} tot {_p2} van de 100 testfoto's goed; bij 0,01 haalt hij er {_a} tot {_b}, maar de ene seed {-_g1} minder dan fase 1 en een andere {_g2} meer; vanaf 0,03 haalt elke seed er 10, evenveel als gokken."}, "nl")
        return _html, _say(lr)

    explore_lr = L.explore("lr", "learning rate", _frame, start=C.TUNE_LR, values=C.RATES, role="hyper",
                           task="Zoek de kleinste learning rate waarbij fase 2 elke testfoto dezelfde klasse geeft.")
    explore_lr
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Vanaf 0,03 haalt elke seed er 10 van de 100: fase 2 geeft elke testfoto dezelfde klasse. De getallen in het netwerk zijn `nan` geworden, zoals `inf / inf` in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), en een model vol `nan` kan geen foto's meer uit elkaar houden; oefening 3 vraagt waarom fase 2 ontspoort. Tot en met 0,003 blijft fase 2 dicht bij fase 1: met 0,003 haalt elke seed er 0 tot 2 meer. Bij 0,01 komt fase 2 verder van fase 1 af: van 1 minder tot 3 meer, afhankelijk van de seed. De grens ligt dus ergens tussen 0,01 en 0,03, en de les neemt 0,003, ruim daaronder.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo finetune je elke layer van een pretrained classifier op je eigen foto's van k soorten dingen:

    1. Maak elke foto één keer klaar, zoals in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl), en verdeel ze in drieën: foto's om van te leren, een validation set om mee te kiezen, en foto's om op te testen, elk met zijn label, een heel getal van 0 tot en met k − 1.
    2. Fase 1: reken met het model bevroren en `model.fc = nn.Identity()` de features van elke trainingsfoto één keer uit; geef het model dan een nieuwe laatste layer, `model.fc = nn.Linear(512, k)`, met 512 het aantal features van ResNet-18, en train die op die features.
    3. Fase 2: ontdooi elke parameter, en train ze allemaal verder in mini-batches met een kleine learning rate en het model in eval mode.
    4. Vergelijk de twee fases op de validation set, en houd fase 2 alleen als die het daar beter doet; test daarna het model dat je houdt op de testfoto's, één keer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    **Twee downloads.** ResNet-18 is die uit [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl): 47 MB van Hugging Face, met de weights van torchvision, getraind op ImageNet. Caltech-101 bestaat uit foto's van voorwerpen in 101 klassen, in 2003 van het web gehaald: 137 MB, uit de eigen dataopslag van Caltech, onder de licentie CC BY 4.0, maar de rechten op elke foto liggen bij de fotograaf. Daarom levert de course de foto's niet mee, en deelt hij geen model dat erop getraind is. Hij gebruikt 10 klassen van 30 foto's, die hij een voor een heeft nagekeken, zodat er op geen ervan een mens staat: kroonluchter, libel, schenkkan (een hoge kan), lotus, nautilus, haan, zeester, zonnebloem, waterlelie en windsorstoel (een houten stoel met een rugleuning van dunne spijlen). Daarvan heeft ImageNet libel en zeester, en nog drie onder een andere naam: chambered nautilus, cock en pitcher. Lotus, waterlelie, zonnebloem, kroonluchter en windsorstoel heeft het niet. De knop hieronder downloadt ze allebei, één keer.

    De cel neemt de eerste 20 foto's van elk van de 10 klassen om van te leren, en foto 20 tot en met 29 van elke klasse om op te testen: 200 en 100 foto's. `C.CLASSES` bevat de namen van de klassen, en `C.photos(name)` de 30 foto's van een klasse; het label van een foto is de plaats van zijn klasse in `C.CLASSES`. `C.prepare(photo)` doet de vijf stappen uit [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl) voor een foto van elke grootte, met het gemiddelde en de standaardafwijking per kanaal van ImageNet, en geeft een array met shape `(3, 224, 224)`. `right(model, X, y)` telt hoeveel foto's het model goed heeft, 50 tegelijk. `pretrained()` geeft een nieuwe ResNet-18 met de weights van ImageNet, elke parameter bevroren, in eval mode: de cel hieronder geeft die door zodra beide downloads klaar zijn. De cel print de shapes van de foto's, en hoeveel testfoto's goed zijn na fase 1 en na fase 2. Op de computer van de course, met 4 cores, duurde fase 1 ongeveer 3 seconden, de features van de 200 foto's meegerekend, en elke epoch van fase 2 ongeveer 7: twee epochs van fase 2 kosten ongeveer vijf keer zoveel als de hele fase 1. Op een tragere computer duurt alles langer.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="ResNet-18 en Caltech-101 downloaden, 184 MB")
    return (get_button,)


@app.cell(hide_code=True)
def downloads_cell(get_button):
    pretrained = C.downloads_or_stop(get_button, {
        "ask": "*ResNet-18 en de foto's van Caltech-101 staan nog niet op je computer. De knop downloadt ze één keer: ResNet-18 van Hugging Face, 47 MB, en Caltech-101 van Caltech, 137 MB, waarvan hij daarna de foto's uitpakt. Het uitgewerkte voorbeeld en de demo hebben ze nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**{name} downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van Caltech-101 uitpakken** · {done} foto's",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face of Caltech was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 184 MB en de uitgepakte foto's. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*ResNet-18 en de foto's staan op je computer: 11,7 miljoen parameters, getraind op ImageNet, en de foto's van Caltech-101, waarvan de les 10 klassen gebruikt.*",
    })
    return (pretrained,)


@app.cell
def example_fine_tune(pretrained):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    def right(model, X, y):
        count = 0
        with torch.no_grad():
            for xb, yb in DataLoader(TensorDataset(X, y), batch_size=50):
                count += (model(xb).argmax(dim=1) == yb).sum().item()
        return count

    # 1. elke foto één keer klaargemaakt: 20 per klasse om van te leren, 10 om op te testen
    train, y_train, test, y_test = [], [], [], []
    for label in range(10):
        photos = C.photos(C.CLASSES[label])
        for photo in photos[:20]:
            train.append(C.prepare(photo))
            y_train.append(label)
        for photo in photos[20:30]:
            test.append(C.prepare(photo))
            y_test.append(label)
    X_train, X_test = torch.tensor(np.array(train)), torch.tensor(np.array(test))
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
    print(X_train.shape, X_test.shape)
    # 2. fase 1: de 512 features van elke trainingsfoto, één keer; een nieuwe laatste layer leert erop
    model = pretrained()
    model.fc = nn.Identity()
    rows = []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X_train, y_train), batch_size=50):
            for row in model(xb):
                rows.append(row.numpy())
    features = torch.tensor(np.array(rows))
    torch.manual_seed(0)
    model.fc = nn.Linear(512, 10)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.fc.parameters(), lr=0.1)
    for epoch in range(100):
        loss = loss_fn(model.fc(features), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("na fase 1:", right(model, X_test, y_test), "van de 100 testfoto's goed")
    # 3. fase 2: elke parameter leert, met een kleine learning rate, in mini-batches van 20
    torch.manual_seed(0)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=20, shuffle=True)
    for epoch in range(2):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # 4. test opnieuw
    print("na fase 2:", right(model, X_test, y_test), "van de 100 testfoto's goed")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De foto's hebben shape `(200, 3, 224, 224)` en `(100, 3, 224, 224)`. Na fase 1 zijn 90 van de 100 testfoto's goed, en alle 10 fouten halen lotus en waterlelie door elkaar. Fase 2 brengt dat op 92: 2 meer dan fase 1, en de 8 fouten die overblijven, zijn weer lotussen en waterlelies. Over 5 seeds halen 2 epochs fase 2 er bij elke seed 0 tot 2 meer, en 10 epochs tussen 1 minder en 3 meer. Fase 1 heeft alle 200 trainingsfoto's al goed, dus de loss waarvan fase 2 leert, is klein, en zijn stappen ook. Ter vergelijking: een ResNet-18 met willekeurige weights, getraind met de loop van fase 2 op dezelfde 200 foto's, maar in train mode, 10 epochs en met een learning rate van 0,01, haalt er 29 tot 38 van de 100 goed, over 3 seeds. Zonder wat hij op ImageNet leerde, zijn 200 foto's veel te weinig voor 11 miljoen getallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_params():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat leert er in fase 2
    """)
    return


@app.cell(hide_code=True)
def choice_params_cell():
    choice_params = L.choice(
        "params",
        question="Je finetunet ResNet-18 voor alle 99 klassen die de course van Caltech-101 bewaart (hij laat de twee klassen met gezichten weg): een nieuwe laatste layer van 512 features naar 99 klassen, en dan full fine-tuning. ResNet-18 zoals je hem downloadt, met zijn laatste layer voor de 1.000 klassen van ImageNet, heeft 11.689.512 parameters. Hoeveel getallen leren er in fase 2?",
        options={
            "right": "11.227.299",
            "head": "50.787",
            "imagenet": "11.689.512",
            "no-head": "11.176.512",
            "both": "11.740.299",
        },
        correct="right",
        why={
            "right": "Zonder zijn oude laatste layer, 512 · 1.000 + 1.000 = 513.000 getallen, heeft ResNet-18 er 11.176.512; de nieuwe voegt er 512 · 99 + 99 = 50.787 toe. Bij full fine-tuning leren ze allemaal: 11.227.299.",
            "head": "Dat zijn de getallen van de nieuwe laatste layer, 512 · 99 + 99: wat er in fase 1 leert. In fase 2 leert elke parameter: ook de 11.176.512 van de rest van ResNet-18, samen 11.227.299.",
            "imagenet": "De oude laatste layer, voor 1.000 klassen, is weg: `model.fc` is vervangen. Haal zijn 512 · 1.000 + 1.000 = 513.000 eraf, en tel de nieuwe erbij, 512 · 99 + 99 = 50.787: 11.227.299.",
            "no-head": "Dat is ResNet-18 zonder laatste layer. De nieuwe leert ook mee, in beide fases: 512 · 99 + 99 = 50.787 meer, samen 11.227.299.",
            "both": "Dan tel je de oude laatste layer, voor 1.000 klassen, naast de nieuwe mee. Maar `model.fc = nn.Linear(512, 99)` vervangt hem: haal zijn 512 · 1.000 + 1.000 = 513.000 eraf, en er blijven 11.227.299 over.",
        },
    )
    choice_params
    return (choice_params,)


@app.cell(hide_code=True)
def text_exercise_phase_two():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Fase 2 in een training loop

        Maak `phase_two(model, X, y, lr, epochs)` af: stap 3 van het recept, voor elk model na fase 1. `X` bevat de klaargemaakte trainingsfoto's, `y` hun labels, `lr` is de learning rate en `epochs` het aantal epochs. Vervang de drie `...`; de functie geeft het model terug, verder getraind. De check verzint een klein model zoals fase 1 het achterlaat, een bevroren eerste layer en een laatste layer die leert, en 50 verzonnen voorbeelden van elk 4 getallen, en roept `phase_two(model, X, y, 0.1, 2)` aan. Hij kijkt welke layers veranderd zijn, en wat de loss van het model op zijn voorbeelden daarna is. Hij probeert ook andere learning rates en aantallen epochs.
        """),
        L.task("Laat `phase_two` elke parameter van het model ontdooien en ze allemaal trainen in mini-batches, met learning rate `lr`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_phase_two():
    phase_two_code = L.code_box("phase-two", lines=20)
    phase_two_code
    return (phase_two_code,)


@app.cell(hide_code=True)
def check_phase_two(phase_two_code):
    result_phase_two = L.check("phase-two", phase_two_code.value)
    result_phase_two
    return (result_phase_two,)


@app.cell(hide_code=True)
def hints_phase_two():
    L.hints(
        "phase-two",
        [
            "Stap 3 van het uitgewerkte voorbeeld, met de learning rate uit `lr`.",
            "Ontdooien is bevriezen andersom: `requires_grad` weer aan. De optimizer krijgt elke parameter van het model, en de loss krijgt de logits van de batch, `model(xb)`, en de labels van de batch.",
            "`p.requires_grad = ___`, `opt = torch.optim.SGD(model.___(), lr=lr)` en `loss = loss_fn(model(xb), ___)`.",
        ],
        why="`p.requires_grad = True` laat elke parameter weer een gradient krijgen, en `torch.optim.SGD(model.parameters(), lr=lr)` verschuift ze allemaal, een kleine stap per batch. `loss_fn(model(xb), yb)` is de loss van één batch van 20 voorbeelden, dus het model zet 10 stappen per epoch op 200 foto's in plaats van één.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw fase 2 op de foto's

    Zodra je ResNet-18 en de foto's met de knop bij het uitgewerkte voorbeeld hebt gedownload, traint de knop hieronder fase 1 opnieuw en geeft hij het model aan jouw `phase_two` als die klopt, anders aan die van de course: 2 epochs met een learning rate van 0,003. Terwijl fase 2 traint, laat een grafiek de loss van elke batch van 20 zien. Ernaast of eronder staat wat ResNet-18 op dat moment gokt voor de 20 testfoto's van lotus en waterlelie, de twee bloemen die bij alle fouten na fase 1 door elkaar worden gehaald, elk met hoe zeker het is: de kans die softmax de gegokte klasse geeft, zoals in [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl). Daaronder staat wat `layer4.1`, het laatste deel van `layer4`, van de eerste lotus maakt: zijn **feature maps**, een raster van 7 bij 7 getallen voor elk van zijn channels, één getal voor elk stuk van de foto. Allebei worden ze opnieuw getekend terwijl fase 2 traint. Daarna kun je terugspoelen naar een paar stappen van fase 2, met allebei zoals ze toen waren, en laat een tabel per klasse zien hoeveel van de 10 testfoto's goed zijn na fase 1 en na fase 2. Dat duurt ongeveer even lang als het uitgewerkte voorbeeld.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Train fase 1 en fase 2")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_phase_two(pretrained, demo_button, phase_two_code, result_phase_two):
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om fase 1 en fase 2 te trainen.*")
    else:
        _now = {}
        _layer = "Het laatste deel van layer4 bij de eerste testfoto van een lotus: {name}"

        def _progress(_p):
            if "step" not in _p:   # fase 1 begint: een nieuwe grafiek voor fase 2
                _now["live"] = L.live({"title": "Fase 2: de loss van elke batch", "x": "stap", "y": "loss",
                                       "alt": "De loss van elke batch van 20 trainingsfoto's in fase 2 tot nu toe, stap voor stap."},
                                      C.TUNE_EPOCHS * C.STEPS, low=0.0, look_every=2.0)
                _now.pop("look", None)
                _now["live"].line("**Fase 1** · de features van de 200 trainingsfoto's, en een nieuwe laatste layer")
            else:
                if "look" not in _now:   # fase 2 begint: wat ResNet-18 bij elke stap maakt, naast de grafiek
                    _now["look"] = _p["look"]
                    _now["live"].watch(lambda: C.now_pictures(_now["look"](), _layer, L.lang, "nv-tune-now"))
                _now["live"].add(_p["step"], _p["loss"], f"**Fase 2** · stap {_p['step']} van {C.TUNE_EPOCHS * C.STEPS} · loss {C.loss_text(_p['loss'], 'nl')}")

        _d = C.demo(pretrained, phase_two_code.value if result_phase_two.state == "correct" else None, _progress)
        _who = "jouw `phase_two`" if _d["yours"] else "de `phase_two` van de course"
        _rows = "\n".join(f"| {C.NAMES['nl'][_c]} | {_b} | {_a} |" for _c, _b, _a in zip(C.CLASSES, _d["before"], _d["after"]))
        _labels = {"title": "Fase 2: de loss van elke batch", "x": "stap", "y": "loss",
                   "alt": "De loss van elke batch van 20 trainingsfoto's in fase 2, stap voor stap."}
        _when = sorted(_d["looked"])

        def _frame(_k):
            _k, _seen = int(_k), _d["looked"][int(_k)]
            _right = sum(_g == _t for _g, _t in zip(_seen["guesses"], _seen["truth"]))
            _at = "Na fase 1, voordat fase 2 een stap zet" if _k == 0 else f"Na stap {_k} van fase 2"
            return C.step_frame(_d, _k, _labels, _layer, L.lang), f"{_at}: {_right} van deze 20 foto's zijn goed."

        if len(_when) > 1:   # terugspoelen door fase 2, met heel fase 2 in het plaatje
            _pic = L.explore("rewind", "stap", _frame, start=_when[-1], values=_when, role="",
                             task="Ga terug naar stap 0 en dan weer vooruit: bij welke foto's verandert de gok, en hoe zeker is ResNet-18 daarvan?")
        else:
            _pic = mo.Html(C.losses_picture(_d["losses"], _labels, L.lang))
        _out = mo.vstack([_pic, mo.md(f"""Met {_who}, van de 10 testfoto's van elke klasse:

| klasse | goed na fase 1 | goed na fase 2 |
|---|---|---|
{_rows}
| **alle 100** | **{sum(_d['before'])}** | **{sum(_d['after'])}** |

Bij {_d['mixed']} van de {100 - sum(_d['after'])} fouten na fase 2 is een lotus voor een waterlelie aangezien, of andersom.""")])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_too_big():
    mo.md(r"""
    ### Oefening 3. Een learning rate die te groot is
    """)
    return


@app.cell(hide_code=True)
def choice_too_big_cell():
    choice_too_big = L.choice(
        "too-big",
        question="Met een learning rate van 0,03 groeit de loss van fase 2 van stap tot stap, van 0,01 tot 49 in zijn eerste 7 stappen, en twee stappen later is hij `nan`. Vanaf dan geeft ResNet-18 elke foto dezelfde klasse. Fase 1 trainde zijn nieuwe laatste layer met 0,1, meer dan drie keer zo groot, zonder problemen. Waarom gaat fase 2 met 0,03 mis?",
        options={
            "right": "In fase 2 bewegen alle layers tegelijk, en elke verandering verandert ook wat alle latere layers krijgen: één stap verandert de logits veel meer dan in fase 1",
            "by-heart": "11 miljoen getallen leren de 200 trainingsfoto's uit het hoofd",
            "short": "2 epochs zijn te weinig: met een paar meer zou fase 2 zich herstellen",
        },
        correct="right",
        why={
            "right": "In fase 1 verschuift een stap alleen de laatste layer, op features die hetzelfde blijven. In fase 2 verschuift een stap ook conv1 en elke layer daarna, en elke verandering werkt door in alle layers die later komen, dus samen veranderen ze de logits veel meer dan dezelfde stap op alleen de laatste layer. Stappen die te groot zijn, springen over het dal heen, de loss groeit, en de volgende gradients groeien mee: de training divergeert, zoals in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl), tot de getallen te groot worden voor de computer en `nan` worden, zoals in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl). Daarom heeft fase 2 een veel kleinere learning rate nodig dan fase 1.",
            "by-heart": "Uit het hoofd leren haalt de trainingsfoto's goed en nieuwe foto's fout. Hier doen de trainingsfoto's het niet beter: van de 200 zijn er maar 20 goed, allemaal van één klasse. De training is ontspoord, en de getallen in het netwerk zijn `nan` geworden. Zie [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
            "short": "De course probeerde 6 epochs met 0,03: nog steeds 10 van de 100, elke foto dezelfde klasse. Zodra een getal in het netwerk `nan` is, is alles wat ermee wordt uitgerekend ook `nan`, elke gradient incluis: er is geen weg terug. Zie [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl).",
        },
    )
    choice_too_big
    return (choice_too_big,)


@app.cell(hide_code=True)
def text_exercise_worth_it():
    mo.md(r"""
    ### Oefening 4. Wanneer loont het?
    """)
    return


@app.cell(hide_code=True)
def explain_worth_it_cell():
    explain_worth_it = L.explain(
        "worth-it",
        question="Over 5 seeds halen 2 epochs full fine-tuning 0 tot 2 van de 100 testfoto's meer goed dan fase 1 alleen, voor ongeveer vijf keer het rekenwerk. Waarom voegt het hier zo weinig toe, en wanneer zou het wel lonen? Denk aan wat fase 1 al goed heeft, aan de fouten die overblijven, en aan de kleding uit [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl).",
        model_answer="**Hier laat fase 1 weinig te leren over.** Die heeft alle 200 trainingsfoto's al goed, dus de loss waarvan fase 2 leert, ligt dicht bij 0, en zijn stappen ook. De features van ResNet-18 houden deze soorten al uit elkaar: over 5 seeds halen 30 van de 32 fouten na fase 1 lotus en waterlelie door elkaar, twee bloemen die op elkaar lijken, en sommige foto's met het label lotus lijken waterlelies te laten zien. Fase 2 haalt er een paar van goed, en 26 van de 27 fouten daarna zijn nog steeds lotus en waterlelie. **Full fine-tuning loont als de features maar half bij de foto's passen:** dichtbij genoeg om vanuit te beginnen, niet zo dichtbij dat fase 1 alles al goed heeft. Bij de kleding ging de cijferlezer van 506 tot 539 van de 1.000 testplaatjes goed met fase 1 alleen naar 566 tot 593 met fine-tuning, al deed een nieuw netwerk vanaf nul het daar nog beter. Meer foto's per klasse helpen ook, of meer versies van dezelfde foto's door data augmentation, zodat 11 miljoen getallen kunnen bewegen zonder hun foto's uit het hoofd te leren. **Of het loont, kunnen alleen foto's zeggen waarop hij niet trainde:** vergelijk fase 2 met fase 1 alleen op een validation set, en weeg wat het oplevert af tegen wat het kost.",
        points=[
            "je zegt dat fase 1 alle trainingsfoto's al goed heeft, zodat fase 2 weinig over heeft om van te leren",
            "je zegt dat de fouten die overblijven vooral lotus en waterlelie door elkaar halen, die op elkaar lijken, zodat de features al bij deze foto's passen",
            "je zegt dat full fine-tuning loont als de features maar half bij de foto's passen, zoals de features van de cijferlezer bij de kleding, en dat alleen foto's waarop hij niet trainde kunnen laten zien of het loont",
        ],
    )
    explain_worth_it
    return (explain_worth_it,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Fase 1 is de feature extractor uit [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl), fase 2 de fine-tuning uit [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl), nu op ResNet-18 uit [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl), in mini-batches uit een `DataLoader`. Full fine-tuning laat alle 11 miljoen parameters leren. Op foto's die de features van ResNet-18 al goed beschrijven, voegen 2 epochs ervan 0 tot 2 van de 100 toe aan fase 1, en een learning rate die tien keer te groot is, laat het ontsporen. Of het loont, kunnen alleen foto's zeggen waarop hij niet trainde. Batch normalization, de reden dat het model in eval mode blijft, komt in [Batch normalization](/course/open/nn-batchnorm?lang=nl); maar een deel van een model finetunen in [Project: LoRA, finetunen met een low-rank update](/course/open/llm-lora?lang=nl). In de volgende les, *Labels, boxes, masks en keypoints*, laat je hele foto's met één label achter je: wat een beeldmodel moet vinden, bepaalt hoe zijn foto's gelabeld worden, met een box per voorwerp, een mask per pixel of een punt per gewricht.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_params, result_phase_two, choice_too_big, explain_worth_it):
    L.wrap_up(choice_params, result_phase_two, choice_too_big, explain_worth_it)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
