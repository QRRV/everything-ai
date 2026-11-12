# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-project-resnet-grad-cam. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-project-resnet-grad-cam", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl) en over [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_layer():
    quick_layer = L.choice(
        "quick-layer",
        question="Uit welke layer van ResNet-18 hield Grad-CAM op de collage van de kat en de koffie de twee klassen nog uit elkaar?",
        options={
            "right": "`layer4`, de laatste convolutional layer: zijn maps komen vlak vóór global average pooling en de logits",
            "finer": "`layer3`: zijn maps hebben 14 bij 14 plaatsen in plaats van 7 bij 7, dus laten ze meer zien",
            "first": "`conv1`, de eerste layer: die ziet de foto zelf",
            "any": "Elke layer: Grad-CAM geeft uit elk ervan dezelfde map",
        },
        correct="right",
        why={
            "right": "Van de maps van de laatste convolutional layer worden de logits gemaakt, en ze hebben nog plaatsen. Vandaag neem je Grad-CAM uit het laatste block van je eigen ResNet: 7 bij 7 plaatsen voor een afbeelding van 28 bij 28.",
            "finer": "Ze zijn fijner, maar op de collage hielden de maps uit `layer3` de kat en de koffie niet meer uit elkaar: die layer kijkt nog naar onderdelen, nog niet naar klassen. Zie [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl).",
            "first": "De eerste layer zoekt naar randen en kleuren, overal in de foto, voor elke klasse hetzelfde. Grad-CAM houdt klassen uit elkaar waar de maps al voor klassen staan: in de laatste convolutional layer.",
            "any": "Grad-CAM werkt voor elke layer, maar geeft niet dezelfde map: uit `layer3` hielden de maps de kat en de koffie niet meer uit elkaar. Zie [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl).",
        },
    )
    quick_layer
    return (quick_layer,)


@app.cell(hide_code=True)
def quick_check_upside():
    quick_upside = L.choice(
        "quick-upside",
        question="Je ColorCNN leerde van foto's die ondersteboven gezet waren in plaats van gespiegeld, en las 479 tot 492 van de 1.000 testfoto's goed, tegen 541 tot 591 met spiegelbeelden. Waarom deed ondersteboven het slechter?",
        options={
            "right": "Echte foto's laten bijna nooit iets ondersteboven zien, dus leerde het netwerk van foto's die niet lijken op die waarop het getest wordt",
            "pixels": "Ondersteboven verplaatst meer pixels dan een spiegelbeeld",
            "fewer": "Ondersteboven geeft minder nieuwe foto's dan een spiegelbeeld",
            "colors": "Ondersteboven zet de kleuren van een foto in een andere volgorde",
        },
        correct="right",
        why={
            "right": "Een verandering helpt als het label blijft kloppen en de foto echt zou kunnen zijn: een gespiegelde auto is een auto op een weg, een auto op zijn kop is een foto die geen test set heeft. Vandaag vraag je hetzelfde over kleding: lijkt een gespiegelde schoen op de schoenen in de testafbeeldingen?",
            "pixels": "Allebei zetten ze elke pixel ergens anders neer: een spiegelbeeld de kolommen in de andere volgorde, ondersteboven de rijen. Het gaat erom of de foto die eruit komt echt zou kunnen zijn.",
            "fewer": "Allebei maken ze van elke foto precies één andere versie. Het gaat erom of die versie een echte foto zou kunnen zijn.",
            "colors": "Ondersteboven zet de rijen in de andere volgorde, `dims=[1]`; de drie channels, de kleuren, blijven waar ze zijn.",
        },
    )
    quick_upside
    return (quick_upside,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) lieten residual blocks een netwerk van 19 layers leren waar een gewoon netwerk dat niet deed. Sindsdien kwamen er in dit deel foto's in kleur bij, data augmentation, diepere netwerken, en manieren om te zien waar een netwerk naar kijkt. Vandaag komt dat samen. In 2015 trainden Kaiming He, Xiangyu Zhang, Shaoqing Ren en Jian Sun ResNets van 20 tot 1.202 layers op CIFAR-10, allemaal gemaakt van één block met **batch normalization** na elke convolutie. Jij bouwt de kleinste ervan, ResNet-20, van je eigen block, en traint het op kleding, met veranderingen die bij kleding passen. Je kiest zijn instellingen op een validation set, en test het één keer. En dan doe je wat er na een goede test hoort te komen: een **error analysis**. Je telt welke soorten je netwerk door elkaar haalt, bekijkt de fouten waarvan het het zekerst was, en ziet met je Grad-CAM waar het keek toen het ze maakte.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken", "Je CNN op kleurenfoto's", "Je eigen one-stage detector",
                            "Je eigen U-Net", "Eindproject: ResNet met Grad-CAM", "Eindproject: U-Net of detector"],
                           7, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_wrong():
    guess_wrong = L.guess(
        "most-wrong",
        question="Je ResNet zal de meeste van de 1.000 testafbeeldingen goed lezen. Welke soort leest het het vaakst fout?",
        options={
            "tshirt": "Het T-shirt",
            "coat": "De jas",
            "shirt": "Het overhemd",
            "sandal": "De sandaal",
        },
        reveal="Het overhemd. Met elk van de seeds 0 tot en met 4 las het ResNet van de course meer overhemden fout dan elke andere soort: 21 tot 29 van de 97, vaker als jas dan als iets anders. En 62% tot 68% van al zijn fouten zijn het ene kledingstuk voor het bovenlijf gelezen als een ander: T-shirt, trui, jas of overhemd. De error analysis na de demo kijkt naar die fouten.",
    )
    guess_wrong
    return (guess_wrong,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **ResNet-20.** Het netwerk dat He en zijn collega's op CIFAR-10 trainden, begint met één convolutie van 3 bij 3 naar 16 channels. Dan komen drie **stages** van residual blocks: de eerste met 16 channels op maps van 32 bij 32, de tweede met 32 channels op 16 bij 16, de derde met 64 channels op 8 bij 8. Elke stage halveert de hoogte en de breedte en verdubbelt de channels: het eerste block van stage 2 en van stage 3 doet dat met stride 2 in zijn eerste convolutie. Aan het eind komen global average pooling, het gemiddelde van elke map, zoals aan het eind van ResNet-18 in [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl), en een linear layer naar 10 logits. Met $n$ blocks per stage, elk met twee convoluties, zijn dat $6n + 2$ layers met weights achter elkaar; met $n = 3$ zijn het er 20, met 0,27 miljoen parameters. Op de afbeeldingen van kleding, 28 bij 28, zijn de maps van de drie stages 28, 14 en 7 plaatsen hoog en breed.

        **Het block.** In de paper komt batch normalization direct na elke convolutie, vóór de activatie. Dus is $f(x)$ een convolutie van 3 bij 3, batch normalization, een ReLU, een tweede convolutie en batch normalization; de residual connection telt $x$ erbij op, en na de som komt de tweede ReLU. Waar een block de channels of de maat verandert, krijgt de residual connection dezelfde verandering, zoals in [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl): een convolutie van 1 bij 1 met stride 2, hier ook gevolgd door batch normalization. Op CIFAR-10 vulde de paper de nieuwe channels in plaats daarvan met nullen; zijn netwerken voor ImageNet gebruiken de convolutie van 1 bij 1, die hier in totaal 2.848 parameters kost, 1% van de 272.970 van het hele netwerk.

        **Batch normalization van feature maps.** `nn.BatchNorm2d(c)` doet wat `nn.BatchNorm1d` deed in [Batch normalization](/course/open/nn-batchnorm?lang=nl), met een channel op de plaats van een neuron: in train mode standaardiseert hij elk channel over al zijn plaatsen in alle afbeeldingen van de mini-batch, en vermenigvuldigt dan met een geleerde $\gamma$ en telt een geleerde $\beta$ op, één van elk per channel. In eval mode gebruikt hij de voortschrijdende gemiddelden van het gemiddelde en de variantie die hij tijdens het trainen bijhield. Gemaakt van de blocks uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl), zonder batch normalization, leerde het ResNet-20 van de course slechter, en één keer liep het uit de hand: het las 822 en 834 van de 1.000 afbeeldingen van de validation set goed, met seed 0 en 1, tegen 892 en 891 met batch normalization; en toen het 40 epochs trainde in plaats van 20, sprong zijn validation loss na epoch 4 van 0,713 naar 2,315, en bleef daarna op de loss van gokken, $-\log 0{,}1 \approx 2{,}30$: het las er 87 van de 1.000 goed.

        **Veranderingen die bij kleding passen.** Elke afbeelding van Fashion-MNIST laat één kledingstuk zien, grijs, in het midden, op zwart. Een random crop met een zwarte rand van 2 pixels verschuift het tot 2 pixels in elke richting: $(2 \cdot 2 + 1)^2 = 25$ plekken. Bij een spiegelbeeld ligt dat anders. Een gespiegeld overhemd is nog steeds een overhemd. Maar de schoenen wijzen allemaal dezelfde kant op: de neus wijst naar links. In alle 602 enkellaarzen van de 6.000 trainingsafbeeldingen is de som van de pixels van de rechterhelft, met de hiel en de schacht, groter dan die van de linkerhelft, en zo ook in 610 van de 617 sneakers en in 553 van de 594 sandalen. Een gespiegelde schoen wijst naar rechts, zoals bijna geen schoen in de testafbeeldingen doet. Je `changed` kan allebei, en de course koos met de validation set, zoals de slider hieronder laat zien.

        **Kiezen op de validation set, één keer testen.** De 6.000 trainingsafbeeldingen worden gesplitst: het netwerk leert van de eerste 5.000, en de laatste 1.000 zijn zijn validation set, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl). Elke keuze wordt gemaakt op de validation set; hier welke veranderingen. Het aantal epochs en de learning rate blijven zoals de course ze zette: met de crop was de validation loss in de vijf seeds het laagst na epoch 16 tot 20, dus 20 epochs zijn niet te veel. De 1.000 testafbeeldingen worden één keer gemeten, aan het eind, zodat hun getal zegt hoe het netwerk het doet op afbeeldingen waarvoor niets gekozen is.

        **Error analysis.** Eén getal, hoeveel testafbeeldingen goed zijn, zegt niet wat er fout gaat. Een error analysis kijkt naar de fouten zelf. De **confusion matrix**, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), telt voor elke goede soort hoe vaak het netwerk elke soort gokte. De **zekerste fouten** zijn de foute gokken met de hoogste kans: `torch.softmax(logits, dim=1)` geeft de softmax van [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) voor elke rij van de logits tegelijk, en de kans van de gok is het grootste getal van zijn rij. Voor elk ervan laat Grad-CAM uit het laatste block zien waar het bewijs voor de gok ligt, en waar dat voor de goede soort ligt. Is het label fout? Lijken de twee soorten op elkaar? Keek het netwerk naar een detail dat het op het verkeerde been zette? Zulke vragen zeggen wat je daarna probeert: meer afbeeldingen van één soort, andere veranderingen, of niets, als zelfs een mens het niet kan zien. Wat je daarna ook probeert, kies het weer op de validation set: zodra er een keuze gemaakt is met de testafbeeldingen, zegt hun getal niet meer hoe het netwerk het doet op nieuwe afbeeldingen.
        """),
        L.watch_out("`torch.softmax(logits, dim=1)` neemt de softmax langs dimensie 1, over de soorten van elke afbeelding: elke rij telt dan op tot 1. Met `dim=0` zou hij de softmax nemen over de afbeeldingen van elke soort, een kolom tegelijk, en zouden de getallen in een rij niet meer de kansen van één afbeelding zijn."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **ResNet-20** | het ResNet van He en zijn collega's voor afbeeldingen van 32 bij 32: een convolutie, drie stages van 3 residual blocks met 16, 32 en 64 channels, global average pooling en een linear layer; 20 layers met weights | `ResNet(1, 16, 3, 10)` |
        | **stage** | de blocks van een ResNet die op maps van dezelfde maat werken; het eerste block van de volgende stage halveert de hoogte en de breedte en verdubbelt de channels | `stage` |
        | **error analysis** | kijken naar de fouten van een getraind netwerk, niet alleen ze tellen: welke soorten het door elkaar haalt, en waarom | |
        | **zekerste fouten** | de foute gokken van een netwerk met de hoogste kans, die waarvan het het zekerst was eerst | `sure_mistakes` |
        | `nn.BatchNorm2d(c)` | batch normalization van c channels: elk channel gestandaardiseerd over al zijn plaatsen in de mini-batch, dan keer $\gamma$ plus $\beta$ | |
        | `torch.softmax(logits, dim=1)` | de softmax van elke rij van de logits: voor elke afbeelding een kans voor elke soort | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_resnet():
    mo.vstack([
        mo.Html(C.network_picture(L.lang)),
        mo.md(r"""
        Het ResNet-20 van de course op een sneaker uit de testafbeeldingen, voordat het iets geleerd had, zoals PyTorch het uitvoert: een vak voor elke layer, met zijn aantal channels erboven; hoe hoger het vak, hoe groter zijn maps. De stem maakt 16 maps van 28 bij 28; de blocks van stage 1 houden ze; `blocks.3` en `blocks.6`, het zesde en het negende vak, halveren de hoogte en de breedte en verdubbelen de channels; global average pooling maakt één getal van elk van de 64 maps van 7 bij 7, en de linear layer 10 logits. Na de demo kun je door de layers van je eigen getrainde ResNet gaan en hun feature maps zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Met welke veranderingen moet het netwerk leren? De slider laat zien hoe de course koos: zijn ResNet-20 trainde 20 epochs lang met elk van drie instellingen, zonder veranderingen, met de random crop, en met de random crop en een spiegelbeeld voor de helft van de afbeeldingen. Telkens de loss op de 5.000 afbeeldingen waarvan het leerde, doorgetrokken, en op de validation set, gestreept, na elke epoch, voor seed 0; en daaronder wat de vijf seeds 0 tot en met 4 van de validation set goed lezen.
    """)
    return


@app.cell(hide_code=True)
def explore_changes_cell():
    def _frame(k):
        k = int(k)
        _name = ("geen veranderingen", "de random crop", "de random crop en een spiegelbeeld")[k]
        _html, _stop = C.settings_frame(k, {"title": f"Seed 0, met {_name}", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                                            "alt": f"De loss van het ResNet-20 van de course met {_name}, seed 0, na elke epoch: op de afbeeldingen waarvan het leerde doorgetrokken, op de validation set gestreept."}, L.lang)
        return _html, (f"Met {_name}: over de seeds 0 tot en met 4 leest het netwerk {_stop['low']} tot {_stop['high']} van de 1.000 afbeeldingen van de validation set goed, "
                       f"{_stop['middle']} in het midden; van de 5.000 afbeeldingen waarvan het leerde {_stop['fit_low']} tot {_stop['fit_high']}.")

    explore_changes = L.explore("changes-slider", "veranderingen", _frame, start=1, values=[0, 1, 2], role="",
                                task="Na welke epoch is de validation loss zonder veranderingen het laagst, en wat doet hij daarna? En welke instelling leest het meest van de validation set goed?")
    explore_changes
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Zonder veranderingen leert het netwerk zijn 5.000 afbeeldingen uit zijn hoofd: met seed 0 leest het er 4.999 goed, en zijn loss erop zakt tot 0,013. Zijn validation loss is het laagst na epoch 8, met 0,420; daarna gaat hij op en neer en eindigt op 0,517: het netwerk past steeds beter bij de afbeeldingen waarvan het leerde, terwijl zijn loss op de andere stijgt. Met de random crop leest het minder van de afbeeldingen waarvan het leerde goed, 4.689 tot 4.711 over de seeds, en meer van de validation set: 884 tot 894, 892 in het midden, tegen 871 zonder veranderingen. Met ook een spiegelbeeld las het netwerk met seed 0 er 900 van de validation set goed, meer dan de 892 met alleen de crop; maar over de vijf seeds 884 tot 900, 889 in het midden. Eén seed had het spiegelbeeld gekozen; vijf kozen alleen de crop, dus traint de demo met alleen de crop. De testafbeeldingen speelden bij dit alles geen rol.

    Na epoch 1 ligt de validation loss bij elke instelling boven de bovenkant van de grafiek: 8,59 met de crop en seed 0. In eval mode gebruikt batch normalization zijn voortschrijdende gemiddelden, en na één epoch lopen die nog achter op de weights, die dan snel veranderen. Met het gemiddelde en de variantie van de afbeeldingen van de validation set zelf was de loss na epoch 1 maar 0,88 geweest, met 643 van de 1.000 goed in plaats van 357.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Je eigen ResNet, getest en uitgelegd:

    1. Het block: $f(x)$ is een convolutie van 3×3 met stride 1 of 2, batch normalization, een ReLU, een convolutie van 3×3 en batch normalization. De residual connection is $x$ zelf, of, waar de channels of de maat veranderen, een convolutie van 1×1 met dezelfde stride en batch normalization. Het block geeft de ReLU van de som.
    2. Het netwerk: een convolutie van 3×3 naar 16 channels, batch normalization en een ReLU; drie stages van 3 blocks, met 16, 32 en 64 channels, het eerste block van stage 2 en 3 met stride 2; global average pooling; een linear layer naar 10 logits.
    3. Veranderingen die bij de afbeeldingen passen: hier een random crop met een zwarte rand van 2 pixels, telkens opnieuw gemaakt als de Dataset een afbeelding geeft.
    4. Leer van de eerste 5.000 trainingsafbeeldingen; kies veranderingen, epochs en learning rate op de laatste 1.000, de validation set.
    5. Train met de gekozen instellingen, en meet dan op de testafbeeldingen, één keer.
    6. De error analysis: de confusion matrix, de zekerste fouten, en voor elk ervan de Grad-CAM van de gok en van de goede soort, uit het laatste block.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel volgt stap 1 waar een stage verandert, met layers die nog niets geleerd hebben: 2 afbeeldingen die uit stage 1 komen, elk 16 maps van 28 bij 28, gaan door de eerste convolutie van stage 2 en door de convolutie van 1×1 van zijn residual connection, en één keer door een convolutie van 3×3 met stride 2 maar zonder padding. De tweede cel doet batch normalization op wat de eerste convolutie geeft, in train mode en in eval mode. De derde cel doet het eerste deel van stap 6 voor 6 verzonnen afbeeldingen en 3 soorten: de kansen, de gokken, de confusion matrix, en de kans van elke foute gok. Voer de cellen uit zoals ze zijn.
    """)
    return


@app.cell
def example_shapes():
    import torch
    from torch import nn

    # 1. waar een stage verandert: van 16 channels van 28 bij 28 naar 32 channels van 14 bij 14
    torch.manual_seed(0)
    x = torch.randn(2, 16, 28, 28)
    conv1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
    skip = nn.Conv2d(16, 32, 1, stride=2)
    print(conv1(x).shape)
    print(skip(x).shape)

    # dezelfde convolutie zonder padding: één plaats minder
    no_padding = nn.Conv2d(16, 32, 3, stride=2)
    print(no_padding(x).shape)
    return conv1, nn, torch, x


@app.cell
def example_norm(conv1, nn, torch, x):
    # batch normalization van 32 channels: elk channel over zijn plaatsen in beide afbeeldingen
    norm = nn.BatchNorm2d(32)
    with torch.no_grad():
        y = norm(conv1(x))
        # hoe ver het gemiddelde van elk channel van 0 ligt, en zijn standaardafwijking
        print(y.mean(dim=(0, 2, 3))[:4].abs().round(decimals=3))
        print(y.std(dim=(0, 2, 3))[:4].round(decimals=3))

        # in eval mode: de voortschrijdende gemiddelden die hij bijhield
        norm.eval()
        print(norm.running_mean[:4].round(decimals=3))
        print(norm(conv1(x)).std(dim=(0, 2, 3))[:4].round(decimals=3))
    return


@app.cell
def example_mistakes(torch):
    # 6. verzonnen logits van 6 afbeeldingen voor 3 soorten, en hun goede soorten
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 3.0, 0.1], [1.0, 1.2, 0.9],
                           [0.1, 0.3, 4.0], [2.5, 2.4, 0.0], [0.0, 0.2, 1.5]])
    labels = torch.tensor([0, 0, 2, 1, 1, 2])

    # de kansen van elke rij, en de gokken
    p = torch.softmax(logits, dim=1)
    guess = p.argmax(dim=1)
    print(p.round(decimals=3))
    print(guess)

    # de confusion matrix: een rij voor elke goede soort, een kolom voor elke gok
    counts = torch.zeros(3, 3, dtype=torch.int64)
    for i in range(len(labels)):
        counts[labels[i], guess[i]] += 1
    print(counts)

    # de kans van elke foute gok, met zijn plek
    for i in range(len(labels)):
        if guess[i] != labels[i]:
            print(i, round(p[i, guess[i]].item(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de eerste cel maken de 3×3-convolutie met stride 2 en padding 1 en de 1×1-convolutie met stride 2 allebei 32 maps van 14 bij 14 van de 16 maps van 28 bij 28, dus kan `x + f` ze optellen, plaats voor plaats. Zonder padding maakt de 3×3-convolutie 13 bij 13, `(28 - 3) // 2 + 1`, en dan zou de som stoppen met een fout. In de tweede cel geeft batch normalization in train mode elk channel een gemiddelde van 0 en een standaardafwijking van 1: `.std()` print 1,001 omdat hij deelt door 391, één minder dan de $2 \cdot 14 \cdot 14 = 392$ getallen van een channel, en batch normalization deelt door 392. Aan het begin is $\gamma$ 1 en $\beta$ 0. Het voortschrijdende gemiddelde van het gemiddelde was 0; na deze ene mini-batch is het $0{,}9 \cdot 0 + 0{,}1 \cdot 0{,}054 = 0{,}005$ voor het eerste channel. In eval mode gebruikt batch normalization de voortschrijdende gemiddelden, en wordt de standaardafwijking 0,584, niet 1: na één mini-batch zijn het nog niet die van de maps. De demo meet je netwerk pas in eval mode als het getraind heeft, en dan zijn ze dat wel. In de derde cel zijn de gokken de soorten 0, 1, 1, 2, 0 en 2, en 2 van de 6 zijn goed: de diagonaal van de confusion matrix. Van de 4 foute gokken is afbeelding 3 de zekerste, met 0,957, dan afbeelding 1 met 0,896: `sure_mistakes(logits, labels, 2)` geeft `[3, 1]`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het block

        Maak de class `Block(c_in, c_out, stride)` af, stap 1 van het recept. In `__init__`: `self.conv1`, een convolutie van 3×3 van `c_in` channels naar `c_out` met deze `stride` en padding 1, en `self.norm1`, zijn batch normalization; `self.conv2`, van `c_out` channels naar `c_out` met padding 1, en `self.norm2`. Waar de stride niet 1 is of de channels veranderen, `self.skip`, een convolutie van 1×1 van `c_in` channels naar `c_out` met deze `stride`, en `self.skip_norm`; elders blijft `self.skip` `None`. In `forward`: $f(x)$, de residual connection, en de ReLU van de som. De eerste cel van het uitgewerkte voorbeeld maakte zulke convoluties. De check probeert een block waar een stage verandert en een block binnen een stage op willekeurige maps, een oneven maat, de gradient, en telt de parameters. Als het klopt, bewaart hij `Block` in je project, als `work/projects/cnn/block.py`; je ResNet is ervan gemaakt.
        """),
        L.task("Maak van `Block` het residual block van ResNet, met batch normalization na elke convolutie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("block", lines=22)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "block",
        [
            "Maak in `__init__` de layers in deze volgorde: `conv1`, `norm1`, `conv2`, `norm2`, en in de `if` `skip` en `skip_norm`. Elke batch normalization komt na een convolutie naar `c_out` channels. In `forward` eerst $f(x)$, dan wat de residual connection van `x` maakt, dan de som.",
            "`self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)` en `self.norm1 = nn.BatchNorm2d(c_out)`; in de `if` `self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride)` en `self.skip_norm = nn.BatchNorm2d(c_out)`. In `forward`: `f = self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))`.",
            "Na `f`: `if self.skip is not None:` met `x = self.skip_norm(self.skip(x))`, en dan `return ___`, de ReLU van `x + f`.",
        ],
        why="Batch normalization direct na elke convolutie houdt wat elke ReLU in gaat bij het begin op een gemiddelde van ongeveer 0 en een standaardafwijking van ongeveer 1, wat de weights ook zijn. $f$ eindigt met batch normalization en zonder ReLU, zodat het ook getallen onder 0 kan maken en iets van $x$ kan afhalen; de ReLU komt na de som. Waar de shape verandert, moet $x$ op dezelfde manier veranderen, anders kan `+` de twee niet optellen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_resnet():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je ResNet

        Maak de class `ResNet(channels, width, blocks, kinds)` af, stap 2 van het recept: een ResNet uit je eigen `Block`. `__init__` heeft de stem al, `self.stem` en `self.stem_norm`, en na de loop de blocks in `self.blocks` en de output layer `self.out`. Vul de loop in: drie stages, met `width`, `2 * width` en `4 * width` channels, elk van `blocks` blocks; het eerste block van stage 2 en van stage 3 komt van de channels van de stage ervoor, met stride 2, en elk ander block houdt zijn channels, met stride 1. Dan `forward`: de stem, zijn batch normalization en een ReLU, elk block om de beurt, global average pooling en de output layer. De check probeert een klein ResNet op twee willekeurige afbeeldingen van kleding, kijkt welk block waar staat, naar de maps van het laatste block, en naar kleurenfoto's met 5 soorten, en telt de parameters van `ResNet(1, 16, 3, 10)`. Als het klopt, bewaart hij `ResNet` in je project, als `work/projects/cnn/deep_resnet.py`; de demo traint het.
        """),
        L.task("Maak van `ResNet` een stem, drie stages van je blocks, global average pooling en een linear layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resnet():
    box_resnet = L.code_box("resnet", lines=26)
    box_resnet
    return (box_resnet,)


@app.cell(hide_code=True)
def check_resnet(box_resnet):
    result_resnet = L.check("resnet", box_resnet.value)
    result_resnet
    return (result_resnet,)


@app.cell(hide_code=True)
def hints_resnet():
    L.hints(
        "resnet",
        [
            "Twee loops in elkaar: `for stage in range(3):` met de channels van die stage, `c = width * 2 ** stage`, en daarin `for b in range(blocks):`. Welk block is het enige dat de channels en de maat verandert, en van hoeveel channels komt het?",
            "In de binnenste loop: `if stage > 0 and b == 0:` dan `layers.append(Block(c // 2, c, 2))`, anders `layers.append(Block(c, c, 1))`. Begin in `forward` met `x = torch.relu(self.stem_norm(self.stem(x)))`, dan `for block in self.blocks:` met `x = block(x)`.",
            "Na de loop in `forward`: `return self.out(___)`, met het gemiddelde van elke map over zijn hoogte en breedte, dimensie 2 en 3.",
        ],
        why="Elke stage werkt op maps die half zo hoog en breed zijn als die van de stage ervoor, met twee keer zoveel channels, zodat elke layer ongeveer evenveel rekenwerk kost. Het eerste block van een stage maakt die verandering, met stride 2 en een convolutie van 1×1 op zijn residual connection; de andere blocks houden hun channels, en hun residual connection is $x$ zelf. Global average pooling maakt één getal van elke map, wat de maat van de afbeelding ook is, en Grad-CAM neemt zijn maps uit het laatste block.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Veranderingen voor elke afbeelding

        Maak `changed(x, u, pad, mirror)` af, stap 3 van het recept, voor een afbeelding `x` van elke shape (C, H, W). `u` bevat drie willekeurige getallen van 0 tot 1, die de demo elke keer opnieuw trekt. Als `mirror` True is en `u[0]` onder 0,5, wordt de afbeelding eerst gespiegeld. Dan gaat hij in een zwarte afbeelding met `pad` pixels aan elke kant, en geeft de functie de H bij W terug die begint bij rij `int(u[1] * (2 * pad + 1))` en kolom `int(u[2] * (2 * pad + 1))` daarvan: een van de $(2 \cdot pad + 1)^2$ plekken. Je `crop` uit [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl) deed dat voor foto's van 3 bij 32 bij 32 met een rand van 4; deze werkt voor elke maat. De check probeert een verzonnen grijze afbeelding van 4 bij 4, een kleurenafbeelding van 5 bij 6, spiegelen aan en uit, en randen van 0 tot 2. Als het klopt, bewaart hij `changed` in je project, als `work/projects/cnn/changes.py`; de demo verandert er elke trainingsafbeelding mee.
        """),
        L.task("Laat `changed` een afbeelding spiegelen als dat mag, en hem uit een zwarte rand snijden op de plek die `u` geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_changes():
    box_changes = L.code_box("changes", lines=14)
    box_changes
    return (box_changes,)


@app.cell(hide_code=True)
def check_changes(box_changes):
    result_changes = L.check("changes", box_changes.value)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def hints_changes():
    L.hints(
        "changes",
        [
            "Drie stappen, zoals in het commentaar. Een spiegelbeeld zet de kolommen in de andere volgorde: axis 2 van (C, H, W). De zwarte afbeelding is zo groot als `x` plus `pad` aan elke kant, en heeft evenveel channels als `x`.",
            "`if mirror and u[0] < 0.5:` met `x = x.flip(dims=[2])`. Dan `c, h, w = x.shape`, `padded = torch.zeros(c, h + 2 * pad, w + 2 * pad)` en `padded[:, pad:pad + h, pad:pad + w] = x`.",
            "`top, left = int(u[1] * (2 * pad + 1)), int(u[2] * (2 * pad + 1))`, en dan `return ___`, het deel van `padded` vanaf rij `top` en kolom `left`, `h` rijen hoog en `w` kolommen breed.",
        ],
        why="`u[1]` en `u[2]` zijn kleiner dan 1, dus is `int(u * (2 * pad + 1))` een heel getal van 0 tot en met 2 · pad: elk van de 2 · pad + 1 plekken heeft dezelfde kans. Met `pad = 0` is de uitsnede de afbeelding zelf, en blijft alleen het spiegelbeeld over.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. De zekerste fouten

        Maak `sure_mistakes(logits, labels, n)` af, het tweede deel van stap 6. `logits` heeft een rij voor elke afbeelding en een kolom voor elke soort; `labels` bevat de goede soort van elke afbeelding. De functie geeft een lijst terug van de plekken van de foute gokken, die met de hoogste kans eerst, hoogstens `n` ervan. De derde cel van het uitgewerkte voorbeeld printte de kans van elke foute gok: maak een lijst van paren `(kans, plek)`, zoals in [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl), zodat `sorted` ze op hun kans sorteert. De check probeert de verzonnen logits van het uitgewerkte voorbeeld, met `n` van 0 tot meer dan er fouten zijn, en logits zonder fouten. Als het klopt, bewaart hij `sure_mistakes` in je project, als `work/projects/cnn/mistakes.py`; de error analysis gebruikt hem.
        """),
        L.task("Laat `sure_mistakes` de plekken van de foute gokken geven, vanaf de zekerste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_sure_mistakes = L.code_box("sure-mistakes", lines=10)
    box_sure_mistakes
    return (box_sure_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_sure_mistakes):
    result_sure_mistakes = L.check("sure-mistakes", box_sure_mistakes.value)
    result_sure_mistakes
    return (result_sure_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "sure-mistakes",
        [
            "Eerst de kansen en de gokken, zoals in de derde cel van het uitgewerkte voorbeeld. Dan, voor elke plek `i` waar de gok niet het label is, een paar: eerst de kans van de gok, zodat `sorted` erop sorteert, en dan de plek.",
            "`pairs = [(p[i, guess[i]].item(), i) for i in range(len(labels)) if guess[i] != labels[i]]`. `sorted(pairs, reverse=True)` zet ze van de hoogste kans naar beneden.",
            "`return [pair[1] for pair in ___]`, met de gesorteerde paren ingekort tot de eerste `n`.",
        ],
        why="`sorted` sorteert paren op hun eerste getal, en kijkt alleen naar het tweede als twee eerste getallen gelijk zijn. `[:n]` neemt er hoogstens `n`: als er minder fouten zijn allemaal, en met `n = 0` geen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint je `ResNet(1, 16, 3, 10)`, gemaakt van je `Block`, op de eerste 5.000 trainingsafbeeldingen, telkens opnieuw veranderd door je `changed` als de Dataset er een geeft, met een rand van {C.PAD} en zonder spiegelbeeld, zoals de course op de validation set koos. Hij traint met je Trainer {C.EPOCHS} epochs lang in mini-batches van 64, met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0. Daarna meet hij je netwerk één keer op de 1.000 testafbeeldingen. Waar een module van jou ontbreekt of kapot is, springt die van de course in. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; op de computer van de course duurde dat 6 tot 8 minuten. Terwijl hij traint, laat de grafiek elke 10 batches de loss op de veranderde trainingsafbeeldingen zien, en gestreept de loss op de validation set na elke epoch. Ernaast staan de gokken van je netwerk op dat moment voor de eerste testafbeelding van elke soort, en daaronder de feature maps van zijn laatste block op een overhemd; allebei worden ze opnieuw getekend terwijl hij traint. Als hij klaar is, kun je door de epochs teruggaan en zie je ze allebei na elke epoch.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je ResNet")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_resnet(train_button, result_block, result_resnet, result_changes):
    result_block, result_resnet, result_changes  # opnieuw uitvoeren als een oefening je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je ResNet te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht
        _heads = {"guesses": "Wat je ResNet gokt", "layer": "Layer {name}, het laatste block van je ResNet, op een overhemd"}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je ResNet: de loss terwijl het traint", "x": "epoch", "y": "loss",
                                        "alt": "De loss van je ResNet tot nu toe: op de veranderde trainingsafbeeldingen elke 10 batches, en gestreept op de validation set na elke epoch."}, _p["epochs"], low=0.0, high=C.TOP_LOSS,
                                       key=[("loss", False, "veranderde trainingsafbeeldingen, elke 10 batches"), ("loss", True, "validation set, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _heads, L.lang, "nv-project-now"))
                _now["val"] = []
            if "val_loss" in _p:   # de validation set na elke epoch, gestreept
                _now["val"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["val"], "", role="loss", replace=True)
                _now["chart"].line(f"**Je ResNet** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # de trainingsafbeeldingen elke 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Je ResNet** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Je ResNet** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `ResNet` en je `changed` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan twintig minuten. Doet je `ResNet`, je `Block` of je `changed` meer werk dan nodig, zoals meer blocks of channels dan het recept? Volgt het precies het recept, dan is deze computer te traag voor de demo.*",
                "resnet": "*Je `ResNet` in `work/projects/cnn/deep_resnet.py` stopt met een fout, of geeft niet 10 logits voor elke afbeelding. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "changes": "*Je `changed` in `work/projects/cnn/changes.py` stopt met een fout, of geeft geen afbeelding met shape (1, 28, 28) terug. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check de oefening ervan opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course stond in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waar je project staat* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "alt": "De loss van je ResNet na elke epoch: op de veranderde trainingsafbeeldingen doorgetrokken, op de validation set gestreept, met een stip waar de validation loss het laagst is."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _heads, L.lang),
                        f"Na epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, en {_right} van deze 10 afbeeldingen goed gelezen.")

            if len(_when) > 1:   # terugspoelen door de run, met de hele run in beeld
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: welke soorten leest je netwerk het eerst goed?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _out = mo.vstack(_pic + [
                mo.md(f"""Je ResNet, met {C.grouped(_result['params'], L.lang)} parameters, leest **{C.grouped(_result['fit'], L.lang)}** van de 5.000 afbeeldingen waarvan het leerde goed, zoals ze zijn, en **{_result['val']}** van de 1.000 afbeeldingen van de validation set. Op de 1.000 testafbeeldingen, één keer gemeten: **{_result['test']}**. De run van de course met seed 0, die de demo volgt, las er 4.708, 892 en 909 goed; over de seeds 0 tot en met 4 waren het er 903 tot 914 van de testafbeeldingen.{_who}"""),
                mo.md("**Je ResNet, layer voor layer.** Het eerste vak is een overhemd uit de testafbeeldingen, en de vakken daarna zijn de layers van je getrainde ResNet, in de volgorde waarin het overhemd erdoor gaat; eronder staan de feature maps van de layer die je kiest."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Ga door je getrainde ResNet van het overhemd naar de logits: waar lijken de maps niet meer op het overhemd?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
                mo.md("In de stem en in stage 1 zijn de maps 28 bij 28, zo groot als de afbeelding, en de meeste volgen het overhemd nog: zijn omtrek, of de lichte en donkere stukken van de stof. Elke stage halveert de plaatsen: in stage 3 zijn er 7 bij 7 over, en in bijna elke map zijn maar een paar van de 49 donker, waar dat channel op het overhemd vond wat het zoekt. Global average pooling maakt één getal van elk van de 64 maps van `blocks.8`, en de linear layer, het laatste vak, maakt daar 10 logits van."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_analysis():
    mo.md(r"""
    ### De error analysis

    Als je ResNet getraind heeft, kijkt de knop hieronder naar zijn fouten op de testafbeeldingen. Eerst de confusion matrix: een rij voor elke goede soort, een kolom voor elke gok. Dan de 8 fouten die je `sure_mistakes` vooraan zet, elk twee keer, met je `grad_cam` uit [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl) uit het laatste block, `blocks[-1]`, eroverheen: links voor de soort die het gokte, rechts voor de goede soort. Waar de map laag is, ligt er een sluier over de afbeelding.
    """)
    return


@app.cell(hide_code=True)
def analysis_button_cell():
    analysis_button = mo.ui.run_button(label="Bekijk de fouten")
    analysis_button
    return (analysis_button,)


@app.cell(hide_code=True)
def analysis_cell(analysis_button, result_sure_mistakes):
    result_sure_mistakes  # opnieuw uitvoeren als oefening 4 je module bewaart
    _run = C.last_run()
    if not analysis_button.value:
        _out = mo.md("*Train eerst je ResNet hierboven, en druk dan op de knop.*")
    elif _run is None:
        _out = mo.md("*Er is nog geen getraind ResNet: druk eerst op de knop van de demo hierboven, en op deze opnieuw als hij klaar is.*")
    else:
        _found = C.analysis(_run)
        if _found is None:
            _out = mo.md("*Maak eerst oefening 4 af: dan staat je `sure_mistakes` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_found, str):
            _out = mo.md({
                "mistakes": "*Je `sure_mistakes` in `work/projects/cnn/mistakes.py` stopt met een fout, of geeft geen lijst van plekken van foute gokken. Heb je het bestand veranderd? Verwijder het, en check oefening 4 opnieuw.*",
                "grad_cam": "*Je `grad_cam` in `work/projects/cnn/gradcam.py` stopt met een fout, of geeft geen map van 7 bij 7 zonder getallen onder 0. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl) opnieuw.*",
            }[_found])
        else:
            _who = "" if not _found["stand_ins"] else " De versie van de course stond in voor " + ", ".join(f"`{m}`" for m in _found["stand_ins"]) + "."
            _out = mo.vstack([
                mo.Html(C.confusion_picture(_found["confusion"], {"guess": "gok", "right": "goede soort",
                                                                  "alt": "De confusion matrix van je ResNet op de 1.000 testafbeeldingen: een rij voor elke goede soort, een kolom voor elke gok, met het aantal in elke cel; de cellen met fouten zijn gekleurd, hoe meer fouten hoe donkerder."}, L.lang)),
                mo.Html(C.mistakes_picture(_found, {"title": "De fouten waarvan je ResNet het zekerst was",
                                                    "read": "{right}, gelezen als {guess}", "sure": "kans {p} · testafbeelding {place}",
                                                    "card": "Testafbeelding {place}, een {right}, die je ResNet las als {guess} met een kans van {p}: één keer met een sluier waar de Grad-CAM van {guess} laag is, één keer waar die van {right} laag is."}, L.lang)),
                mo.md(f"""Met seed 0 leest het ResNet van de course 909 van de 1.000 testafbeeldingen goed. Van zijn 91 fouten zijn er 62 het ene kledingstuk voor het bovenlijf gelezen als een ander, en 11 de ene schoen gelezen als een andere; de broeken en de tassen leest het samen maar twee keer fout. De overhemden leest het het vaakst fout: 21 van de 97, waarvan 10 als jas. Elk van de 8 fouten waarvan het het zekerst was, heeft een kans boven de 0,97. Op zes van hun kaartjes is de map van de goede soort hooguit ongeveer half zo sterk als die van de gok: het netwerk vond bewijs voor wat het gokte, en veel minder voor de goede soort. Alleen op de twee overhemden die als jas gelezen zijn, testafbeeldingen 44 en 444, zijn beide maps ongeveer even sterk: daar vond het allebei, en koos het jas.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 5. Waarom het misging
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-wrong",
        question="Testafbeelding 147 is een overhemd. Het ResNet van de course met seed 0 las het als jurk, met een kans van 0,999, en dat deed elk van de seeds 0 tot en met 4, met 0,997 of meer. Op zijn kaartje is de map van jurk het sterkst op het lange, smalle lijf van het kledingstuk onder de mouwen, en ligt er overal een sluier over de map van overhemd: zijn grootste getal is minder dan een tiende van dat van jurk. Wat zegt dat?",
        options={
            "right": "Voor het netwerk lijkt dit overhemd op een jurk: lang en smal onder de mouwen, zonder iets erin dat het voor een overhemd leerde",
            "more": "Het netwerk trainde te kort: met meer epochs zou het dit overhemd goed lezen",
            "mirror": "Spiegelbeelden hadden geholpen: het netwerk zag dit overhemd nooit andersom",
            "broken": "Grad-CAM werkt niet op deze afbeelding: een map met overal een sluier betekent dat het niets vond",
        },
        correct="right",
        why={
            "right": "Ja. Het bewijs voor jurk ligt op het lange, smalle lijf, 9 pixels breed in de 16 rijen onder de mouwen, en bijna niets spreekt voor overhemd. Het netwerk deed wat het van zijn afbeeldingen leerde; wat dat zou kunnen veranderen, zijn meer overhemden van deze vorm tussen de afbeeldingen waarvan het leert.",
            "more": "Elk van de vijf seeds las dit overhemd als jurk, met een kans van 0,997 of meer, na zijn 20 epochs. Een fout die elke run maakt, en zo zeker, komt van wat de afbeeldingen het leerden, niet van een training die te vroeg stopte.",
            "mirror": "De runs van de course met spiegelbeelden lazen dit overhemd ook als jurk: gespiegeld is een lang, smal overhemd nog steeds lang en smal.",
            "broken": "Grad-CAM werkt hier wel: uit dezelfde layer, op dezelfde afbeelding, is de map van jurk duidelijk. Een map met overal een sluier zegt dat niets in de afbeelding de logit van overhemd verhoogt: het netwerk vond er geen bewijs voor.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Oefening 6. Je verslag
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Schrijf een kort verslag over je ResNet voor iemand die deze les niet deed: hoe zijn zijn instellingen gekozen, hoe deed het op de testafbeeldingen, en wat laten zijn fouten zien? Gebruik de getallen van je eigen run, of die van de run van de course met seed 0.",
        model_answer="**Instellingen.** ResNet-20 met batch normalization na elke convolutie, 272.970 parameters, 20 epochs getraind op 5.000 afbeeldingen van kleding. Zijn veranderingen zijn gekozen op een validation set van 1.000 andere trainingsafbeeldingen, over vijf seeds: zonder veranderingen las het er in het midden 871 goed, met een random crop van hoogstens 2 pixels 892, en met ook spiegelbeelden 889. Dus traint het met alleen de crop. **Test.** Van de 1.000 testafbeeldingen, één keer gemeten, leest het er 909 goed. 62 van zijn 91 fouten zijn het ene kledingstuk voor het bovenlijf gelezen als een ander; de overhemden leest het het vaakst fout, 21 van de 97, waarvan 10 als jas. **Fouten.** Bij zes van de acht fouten waarvan het het zekerst was, laat Grad-CAM bewijs voor de gok zien en veel minder voor de goede soort: testafbeelding 147, een lang, smal overhemd, lijkt voor het netwerk een jurk. Bij twee overhemden die het als jas las, vond het bewijs voor allebei. Om hierna te proberen, en te kiezen op de validation set: meer overhemden met zulke vormen tussen de afbeeldingen waarvan het leert.",
        points=[
            "je zegt dat de veranderingen gekozen zijn op de validation set, met wat elke instelling daar goed las, en dat de testafbeeldingen één keer gemeten zijn, aan het eind",
            "je geeft hoeveel van de 1.000 testafbeeldingen het goed las, en welke soorten het het vaakst verwisselt, uit de confusion matrix",
            "je zegt wat de Grad-CAM van de gok en van de goede soort laat zien bij een fout waarvan het zeker was, en wat dat aanraadt om hierna te proberen, gekozen op de validation set",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het residual block uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) heeft nu de batch normalization uit [Batch normalization](/course/open/nn-batchnorm?lang=nl) na elke convolutie, zoals in de paper, en je ResNet bestaat uit drie stages ervan, zoals ResNet-20. De validation set uit [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) koos zijn veranderingen, waaronder de crop uit [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl); de testafbeeldingen werden één keer gemeten. En na de test kwam de error analysis: de confusion matrix uit [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), en je Grad-CAM uit [Grad-CAM: waar een CNN keek](/course/open/cv-grad-cam?lang=nl) over de fouten waarvan je netwerk het zekerst was. Je project `cnn` heeft nu `block.py`, `deep_resnet.py`, `changes.py` en `mistakes.py`. In de volgende les, *Eindproject: je eigen U-Net of detector*, kies je het andere eindproject: een netwerk dat niet alleen zegt wat er in een afbeelding is, maar ook waar.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_block, result_resnet, result_changes, result_sure_mistakes, choice_why, explain_report):
    L.wrap_up(result_block, result_resnet, result_changes, result_sure_mistakes, choice_why, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
