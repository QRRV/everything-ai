# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-alexnet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-alexnet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl) en [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_top5():
    quick_top5 = L.choice(
        "quick-top5",
        question="Een netwerk heeft een top-5 accuracy van 85% op de testfoto's van ImageNet. Wat zegt dat?",
        options={
            "top1": "Zijn beste gok is goed voor 85% van de foto's",
            "right": "Voor 85% van de foto's staat het label bij zijn vijf beste gokken",
            "five": "Van zijn vijf beste gokken voor een foto is gemiddeld 85% goed",
        },
        correct="right",
        why={
            "top1": "Dat is de top-1 accuracy. De top-5 accuracy is milder: een foto telt als goed als het label ergens op het lijstje van vijf staat. Zie [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl).",
            "five": "Van vijf gokken kan er hooguit één het label zijn, dus is er hooguit één van de vijf ooit \"goed\". De top-5 accuracy telt foto's: die waarvan het label bij de vijf staat. Zie [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl).",
            "right": "De wedstrijd van ImageNet gaf het andersom, als de top-5 error: 1 min de top-5 accuracy, hier 15%, de foto's waarvan het label niet bij de vijf staat. Hij werd zo beoordeeld omdat een foto vaak meer dan één ding laat zien, en maar één daarvan zijn label is.",
        },
    )
    quick_top5
    return (quick_top5,)


@app.cell(hide_code=True)
def quick_check_changes():
    quick_changes = L.choice(
        "quick-changes",
        question="In *Data augmentation voor foto's* veranderde de Dataset elke trainingsfoto in zijn `__getitem__`: gespiegeld of niet, op een willekeurige plek uitgesneden, met net andere kleuren. Waarom daar?",
        options={
            "memory": "Zodat de veranderde foto's geen eigen geheugen nodig hebben",
            "test": "Zodat de testfoto's op dezelfde manier veranderd worden",
            "right": "Zodat elke epoch elke foto in een andere versie laat zien",
        },
        correct="right",
        why={
            "memory": "Dat is mooi meegenomen, en het paper van AlexNet noemt het ook, maar het is niet de reden. Eén keer veranderd, vóór de training, zou elke epoch dezelfde veranderde foto laten zien, en zou het netwerk die uit zijn hoofd kunnen leren. Zie [Data augmentation voor foto's](/course/open/cv-augment-photos?lang=nl).",
            "test": "De testfoto's worden helemaal niet veranderd: die meten, zoals ze zijn. Alleen de trainingsfoto's gaan door die `__getitem__`. Zie [Data augmentation voor foto's](/course/open/cv-augment-photos?lang=nl).",
            "right": "`__getitem__` draait elke keer dat de DataLoader om een foto vraagt, dus kiest elke epoch nieuwe veranderingen. AlexNet deed in 2012 hetzelfde, en vandaag train je weer met die veranderingen, nu ook met dropout.",
        },
    )
    quick_changes
    return (quick_changes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Vanaf 2010 hield ImageNet elk jaar een wedstrijd: geef voor elke testfoto vijf gokken naar de klasse, uit 1000 klassen, beoordeeld op de top-5 error: het deel van de testfoto's waarvan de klasse niet bij de vijf gokken staat. In 2010 en 2011 maten de winnaars features die met de hand waren bedacht, zoals hoe sterk de randen in kleine vierkantjes elke kant op wijzen, en zetten ze daar een lineaire classifier op: een top-5 error van 28,2% in 2010, en 25,8% in 2011. In 2012 haalden Alex Krizhevsky, Ilya Sutskever en Geoffrey Hinton 15,3% met CNN's, en de tweede plaats, met features die met de hand bedacht waren, 26,2%. In 2013 waren de meeste inzendingen CNN's, en de error bleef dalen: VGG en GoogLeNet in 2014, en ResNet in 2015, met 3,6%. VGG en ResNet heb je al gebouwd. Vandaag ga je terug naar het CNN waarmee het begon, AlexNet: je telt waar zijn 61 miljoen parameters zitten, bouwt er een kleine versie van voor CIFAR-10, en meet wat zijn dropout en zijn veranderingen van de foto's nog toevoegen.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_params():
    guess_params = L.guess(
        "params-where",
        question="AlexNet heeft 61 miljoen parameters, in vijf convoluties en drie linear layers daarna. Waar zitten de meeste?",
        options={
            "conv": "In de vijf convoluties: die doen het kijkwerk",
            "even": "Ongeveer gelijk verdeeld over de acht layers",
            "right": "In de drie linear layers aan het eind",
        },
        reveal="**In de drie linear layers.** Van zijn 60.965.224 parameters zitten er 58.631.144 in zijn drie linear layers, 96,2%; `fc6` alleen, van de 9.216 getallen na de laatste pooling naar 4.096, heeft er 37.752.832, 61,9%. Zijn vijf convoluties hebben er samen 2.334.080, 3,8%. De slider na het uitgewerkte voorbeeld loopt ze layer voor layer langs.",
    )
    guess_params
    return (guess_params,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De wedstrijd.** De wedstrijd van ImageNet is ILSVRC, de ImageNet Large Scale Visual Recognition Challenge. De training set had 1,2 miljoen foto's, en de score was de top-5 error. Het plaatje laat de laagste top-5 error van elk jaar zien, op de testfoto's; de open stip is de tweede plaats van 2012. De laagste van 2012 en van 2013 kwamen van inzendingen die ook trainden op foto's van buiten de wedstrijd; zonder die foto's 16,4% en 11,7%.
        """),
        mo.Html(C.ilsvrc_picture({"title": "Laagste top-5 error in ILSVRC", "y": "top-5 error (%)", "second": "tweede, 2012", 2012: "AlexNet",
                                  "alt": "Staven van de laagste top-5 error in ILSVRC per jaar: 28,2% in 2010, 25,8% in 2011, 15,3% in 2012, met AlexNet erbij, 11,2% in 2013, 6,7% in 2014 en 3,6% in 2015. Een open stip op 26,2% voor de tweede plaats van 2012."}, L.lang)),
        mo.md(r"""
        **Acht layers met weights.** AlexNet stuurt een foto van 224 bij 224 door vijf convoluties en drie linear layers, met een ReLU na elke layer behalve de laatste. `conv1` heeft 96 filters van 11 bij 11 met stride 4, en geeft 96 maps van 55 bij 55. `conv2` heeft 256 filters van 5 bij 5; `conv3` en `conv4` 384 van 3 bij 3, en `conv5` 256 van 3 bij 3. Na `conv1`, `conv2` en `conv5` komt max pooling van 3 bij 3 met stride 2: de vierkanten overlappen, terwijl die van de pooling van 2 bij 2 van je SmallCNN dat niet doen. Na de laatste pooling zijn er 256 maps van 6 bij 6 over, 9.216 getallen. `fc6` en `fc7` maken daar elk 4.096 getallen van, en `fc8` geeft een logit voor elk van de 1000 klassen.

        **Twee GPU's.** Hun GPU, een GTX 580, had 3 GB geheugen, te weinig voor dit netwerk. Daarom werd het over twee GPU's verdeeld, de helft van de filters van elke layer op elke GPU, en kijken de filters van `conv2`, `conv4` en `conv5` alleen naar de maps op hun eigen GPU: elk filter van `conv2` ziet 48 van de 96 maps. De training duurde vijf tot zes dagen, voor ongeveer 90 rondes door de 1,2 miljoen foto's, met SGD met momentum 0,9, weight decay van 0,0005, mini-batches van 128 en een learning rate van 0,01, drie keer gedeeld door 10.

        **Wat het gebruikte, en wat elk volgens het paper opleverde.**
        - **ReLU** in plaats van tanh, de S-vormige curve van LeNet-5: een klein CNN van vier layers op CIFAR-10 haalde een training error van 25% zes keer zo snel met ReLU als met tanh. Het paper zegt erbij dat de grootte van dat effect van het netwerk afhangt.
        - **Local response normalization** na `conv1` en `conv2`: elk getal gedeeld door een getal dat groeit met de kwadraten van de getallen op dezelfde plek in naburige maps. Het verlaagde de top-1 en de top-5 error met 1,4 en 1,2 procentpunt.
        - **Pooling die overlapt**, 3 bij 3 met stride 2 in plaats van 2 bij 2 met stride 2: 0,4 en 0,3 procentpunt.
        - **Dropout** van 0,5 na `fc6` en `fc7`: zonder leert het netwerk volgens het paper zijn foto's uit zijn hoofd, en met dropout heeft het ongeveer twee keer zoveel stappen nodig om te convergeren.
        - **Veranderingen van de foto's**: een willekeurige uitsnede van 224 bij 224 uit een foto van 256 bij 256, in de helft van de gevallen gespiegeld, en een kleine verandering van kleur. Zonder de uitsneden en spiegelbeelden, zegt het paper, zou het netwerk zijn foto's uit zijn hoofd leren, en hadden alleen veel kleinere netwerken gewerkt; de verandering van kleur verlaagde de top-1 error met meer dan 1 procentpunt. Bij het testen middelde het zijn gokken over 10 uitsneden: de vier hoeken en het midden, en hun spiegelbeelden.

        De 15,3% kwam niet van één AlexNet. Eén CNN haalde 18,2% op de validatiefoto's; het gemiddelde van de gokken van vijf CNN's 16,4%; en dat van zeven, waarvan er twee eerst getraind waren op 15 miljoen foto's van 22.000 klassen, 15,4%, en 15,3% op de testfoto's.
        """),
        L.watch_out("Een score in een paper is de score van een opzet: één netwerk of het gemiddelde van meerdere, op welke foto's het getraind is, één uitsnede bij het testen of tien. Kijk voordat je twee getallen vergelijkt of ze uit dezelfde opzet komen. De 15,3% van AlexNet was van zeven netwerken; één netwerk haalde 18,2% op de validatiefoto's, die in het paper hooguit 0,1 procentpunt van de testfoto's verschillen: nog steeds ver onder de 26,2% van de tweede plaats."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **ILSVRC** | de jaarlijkse wedstrijd van ImageNet, van 2010 tot 2017: vijf gokken voor elke testfoto, beoordeeld op de top-5 error | |
        | **AlexNet** | het CNN van Krizhevsky, Sutskever en Hinton dat ILSVRC in 2012 won: vijf convoluties, drie linear layers, ReLU, dropout en veranderingen van de foto's | |
        | **local response normalization** | elk getal van een map gedeeld door een getal dat groeit met de kwadraten van de getallen op dezelfde plek in naburige maps | |

        **Een kleine AlexNet.** De demo traint de acht layers van AlexNet op de grootte van CIFAR-10, `SmallAlexNet` uit oefening 2: filters van 5 bij 5 in `conv1` en `conv2`, daarna drie van 3 bij 3, elk met padding zodat een convolutie de grootte houdt die hij krijgt: 32 in `conv1`, 15 in `conv2` en 7 in de andere drie; en dezelfde pooling van 3 bij 3 met stride 2, na `conv1`, `conv2` en `conv5`; dan drie linear layers, met dropout na `fc6` en `fc7`. Hij heeft 201.402 parameters, 17 keer zoveel als je ColorCNN, en 70,7% daarvan zit in zijn linear layers. Local response normalization laat hij weg.

        Het plaatje: hoeveel foto's SmallAlexNet goed leest na 40 epochs, van zijn trainingsfoto's zoals ze zijn, gevuld, en van de testfoto's, open, een stip voor elk van 5 seeds, met de code van de course; eronder, in grijs, je ColorCNN uit [Data augmentation voor foto's](/course/open/cv-augment-photos?lang=nl), ook na 40 epochs. Elk traint met een learning rate die voor hem werkt: 0,01 voor ColorCNN, en 0,001 voor SmallAlexNet, die bij 0,01 bijna niets leert. Zonder dropout en veranderingen leest SmallAlexNet 5.907 tot 5.977 van zijn 6.000 trainingsfoto's goed, en 541 tot 576 testfoto's; zijn validation loss is het laagst na epoch 11 tot 16, en stijgt daarna weer. Met dropout: 542 tot 575 testfoto's, ongeveer evenveel, en 5.482 tot 5.603 trainingsfoto's. Met de veranderingen: 594 tot 636 testfoto's en 4.076 tot 4.221 trainingsfoto's, met de laagste validation loss na epoch 29 tot 36. Met allebei, zoals AlexNet: 581 tot 604 testfoto's, en 3.871 tot 3.953 trainingsfoto's. Je ColorCNN las er met de veranderingen 583 tot 619 goed: de bereiken overlappen, dus op 6.000 foto's van 32 bij 32 leveren 17 keer zoveel parameters weinig op. Oefening 3 vraagt wat de getallen over dropout zeggen.
        """),
        mo.Html(C.runs_picture({"title": "Goed gelezen na 40 epochs", "x": "goed gelezen (%)", "none": "geen dropout, geen veranderingen", "dropout": "dropout",
                                  "changes": "veranderingen", "both": "dropout en veranderingen", "color_none": "ColorCNN, geen veranderingen",
                                  "color_all": "ColorCNN, veranderingen", "mine": "je run", "train": "training", "test": "test",
                                  "alt": "Zes rijen stippen tussen 40 en 100 procent goed gelezen, na 40 epochs. SmallAlexNet zonder dropout en veranderingen: trainingsfoto's ongeveer 99%, testfoto's ongeveer 56%. Met dropout: training ongeveer 93%, test ongeveer 56%. Met de veranderingen: training ongeveer 69%, test ongeveer 62%. Met dropout en veranderingen: training ongeveer 65%, test ongeveer 59%. In grijs ColorCNN zonder veranderingen: training ongeveer 100%, test ongeveer 50%; met de veranderingen: training ongeveer 66%, test ongeveer 60%."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een CNN bouwen naar een paper:

    1. Voor elke convolutie met filters van k bij k, stride s en padding p: n rijen en kolommen worden (n + 2p − k) // s + 1. Hij heeft (maps die hij ziet × k × k + 1) parameters per filter.
    2. Voor max pooling van 3 bij 3 met stride 2: n wordt (n − 3) // 2 + 1.
    3. De eerste linear layer krijgt maps × rijen × kolommen getallen: alles na de laatste pooling, als één rij per foto.
    4. Controleer het in PyTorch: stuur een batch nullen door de layers en print `.shape`; tel met `.numel()`.
    5. Zet dropout na de ReLU van elke linear layer behalve de laatste, train op veranderde foto's, en meet in eval mode.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel bouwt `conv1` van AlexNet en zijn pooling, stuurt één foto van nullen van 224 bij 224 erdoor, en print de shapes. Daarna bouwt hij `fc6`, en telt hij de parameters van `conv1` en van `fc6`. De regel van `conv1` klopt nog niet: de maps van 55 bij 55 uit het paper komen er niet uit. Voer de cel uit zoals hij is, verander hem dan waar het commentaar het zegt, en voer hem opnieuw uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_layers():
    import torch
    from torch import nn

    x = torch.zeros(1, 3, 224, 224)
    # 1. conv1: 96 filters van 11 bij 11 met stride 4
    conv1 = nn.Conv2d(3, 96, 11, stride=4)  # verander: voeg padding=2 toe
    # 2. max pooling van 3 bij 3 met stride 2: de vierkanten overlappen
    pool = nn.MaxPool2d(3, stride=2)
    h = conv1(x)
    print(h.shape)
    print(pool(h).shape)
    # 3. fc6: van 256 maps van 6 bij 6 naar 4.096 getallen
    fc6 = nn.Linear(256 * 6 * 6, 4096)
    # 4. de parameters: weights en biases
    print(conv1.weight.numel() + conv1.bias.numel())
    print(sum(q.numel() for q in fc6.parameters()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, print hij `torch.Size([1, 96, 54, 54])` en `torch.Size([1, 96, 26, 26])`: (224 − 11) // 4 + 1 = 54, en (54 − 3) // 2 + 1 = 26. Met `padding=2` print hij 55 en 27: (224 + 4 − 11) // 4 + 1 = 55, zoals in het paper. Het paper geeft zijn foto's als 224 bij 224 en de maps van `conv1` als 55 bij 55, en dat klopt met 2 pixels padding aan elke kant, zoals de AlexNet van torchvision heeft, of met foto's van 227 bij 227. `conv1` heeft 96 × (3 × 11 × 11 + 1) = 34.944 parameters, en `fc6` 9.216 × 4.096 + 4.096 = 37.752.832: meer dan duizend keer zoveel.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif door de acht layers van AlexNet. De staven zijn hun parameters, in miljoenen, met na elke staaf het deel van het geheel; de layers tot en met de layer die je kiest zijn oranje, en de staaf onderaan telt ze op, met een gestreepte lijn bij 90% van het geheel.
    """)
    return


@app.cell(hide_code=True)
def play_layers():
    _labels = {"title": "Parameters van AlexNet, layer voor layer", "x": "parameters (miljoenen)", "so_far": "tot hier", "line": "90%",
               "alt": "Staven van de parameters van de acht layers van AlexNet, in miljoenen: conv1 tot en met conv5 elk minder dan 1 miljoen, fc6 37,8, fc7 16,8 en fc8 4,1; de layers tot en met de gekozen layer in oranje, en daaronder een staaf van die layers samen, met een gestreepte lijn bij 90% van het geheel."}
    explore_layers = L.explore("layer", "layer", lambda _v: (C.layers_picture(int(_v), _labels, L.lang), C.layer_sentence(int(_v), L.lang)),
                               start=1, values=list(range(1, 9)), role="",
                               task="Na welke layer hebben de layers tot daar samen meer dan 90% van alle parameters?")
    explore_layers
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Na `fc7`: tot daar 56.868.224 parameters, 93,3%; na `fc6` 65,8%. De vijf convoluties hebben er samen 3,8%, minder dan `fc8` alleen, 6,7%. Maar voor één foto doen de convoluties 665.784.864 van de 724.406.816 vermenigvuldigingen, 91,9%: een filter van `conv1` heeft 363 weights, en wordt op elk van de 55 × 55 = 3.025 plekken van zijn map gebruikt, terwijl `fc6` elk van zijn weights één keer gebruikt. De parameters zitten aan het eind, het werk aan het begin. Latere netwerken lieten de grote linear layers vallen, maar niet meteen: VGG eindigde in 2014 nog met drie, met meer dan 100 miljoen parameters in de eerste.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_today():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat bleef?

    Zes dingen die AlexNet deed. ResNet-18, drie jaar later, is het netwerk waarin je keek in [In een CNN kijken: filters en feature maps](/course/open/cv-inside-cnn?lang=nl). Zet elk ding in zijn groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_today():
    sort_today = L.sort("alexnet-today")
    sort_today
    return (sort_today,)


@app.cell(hide_code=True)
def check_today(sort_today):
    result_today = L.check("alexnet-today", sort_today)
    result_today
    return (result_today,)


@app.cell(hide_code=True)
def hints_today():
    L.hints(
        "alexnet-today",
        [
            "Loop ResNet-18 langs zoals in *In een CNN kijken*: `conv1`, `bn1`, een ReLU, een max pooling, en vier fases van residual blocks. Welke van de zes zie je daar?",
            "Drie bleven en drie veranderden. De filters van `conv1` van ResNet-18 zijn 7 bij 7 met stride 2, en na `layer4` neemt hij het gemiddelde van elke map. Het ResNet-paper trainde op willekeurige uitsneden van zijn foto's en hun spiegelbeelden.",
        ],
        why="**ResNet-18 doet dit ook:** een ReLU na zijn layers met weights, max pooling van 3 bij 3 met stride 2 (met padding 1), en trainen op spiegelbeelden en willekeurige uitsneden van zijn foto's. **ResNet-18 doet dit anders:** filters van 7 bij 7 met stride 2 in `conv1`, de grootte waarmee Zeiler en Fergus AlexNet beter maakten; batch normalization in plaats van local response normalization; en het gemiddelde van elk van de 512 maps van `layer4`, dan één linear layer zonder dropout, in plaats van drie linear layers met dropout.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een kleine AlexNet

        Maak `SmallAlexNet` af: de acht layers met weights van AlexNet, op de grootte van CIFAR-10. Zijn vijf convoluties en zijn pooling staan al in het vak. Vul het aantal inputs van `fc6` in: hoeveel getallen elke foto heeft na de laatste pooling. En maak `forward` af: `fc6` en `fc7`, elk met een ReLU en dan dropout, en `fc8`, die de 10 logits geeft. `self.drop` is `nn.Dropout(p)` uit [Dropout](/course/open/nn-dropout?lang=nl): in training mode zet hij elk getal met kans p op 0, en in eval mode doet hij niets. De check geeft je netwerk verzonnen foto's, in eval mode en in training mode, en vergelijkt wat eruit komt met wat de course krijgt. Als het klopt, bewaart hij je class in je project, als `work/projects/cnn/alexnet.py`.
        """),
        L.task("Geef `fc6` het goede aantal inputs, en maak `forward` af met dropout na `fc6` en `fc7`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_small():
    box_small = L.code_box("small-alexnet", lines=28)
    box_small
    return (box_small,)


@app.cell(hide_code=True)
def check_small(box_small):
    result_small = L.check("small-alexnet", box_small.value)
    result_small
    return (result_small,)


@app.cell(hide_code=True)
def hints_small():
    L.hints(
        "small-alexnet",
        [
            "Volg de rijen van een foto: elke convolutie houdt de grootte die hij krijgt, dankzij zijn padding, en elke pooling maakt van n (n − 3) // 2 + 1. Hoeveel rijen zijn er over na drie poolings, en hoeveel maps geeft `conv5`?",
            "32, dan 15, dan 7, dan 3: `fc6` krijgt `32 * 3 * 3` getallen. In `forward` is `self.drop(torch.relu(self.fc6(x)))` `fc6` met zijn ReLU en zijn dropout.",
        ],
        why="`fc6` krijgt 32 maps van 3 bij 3, 288 getallen. Dropout komt na de ReLU van `fc6` en van `fc7`, niet op de logits: een logit die op 0 is gezet, zegt niets meer over zijn klasse. De AlexNet van torchvision zet zijn dropout juist vóór `fc6` en `fc7`: bij het trainen laat die getallen vallen van wat er `fc6` en `fc7` in gaat, en hier van wat er uit `fc6` en `fc7` komt. In eval mode geven ze allebei hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint je `SmallAlexNet` op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 3, L.lang)}, een warmup van één epoch en seed 0, en test hem op de eerste 1.000 testfoto's. Kies hoe: met de foto's zoals ze zijn en zonder dropout; met dropout van {C.number(C.P, 1, L.lang)}; met de veranderingen uit [Data augmentation voor foto's](/course/open/cv-augment-photos?lang=nl), je `crop` en je `jitter`, elke epoch opnieuw; of met allebei, zoals AlexNet. Waar een module van jou ontbreekt of kapot is, springt die van de course in. Hij heeft CIFAR-10 en oefening 2 nodig, en duurt een paar minuten. Tijdens het trainen laat een grafiek na elke epoch de validation loss zien. In het plaatje daarna is de grijze lijn de validation loss van de run van de course met seed 0 zonder dropout en veranderingen; als je zelf zo traint, zijn run met de veranderingen. Daarna kun je terugspoelen door je run, en zie je je run naast die van de course.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="CIFAR-10 downloaden, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. Heb je [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) op deze computer gedaan, dan wel: dan vindt deze les het. Anders downloadt de knop het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. De pagina van CIFAR-10 noemt geen licentie, dus levert de course het niet mee; waar de foto's vandaan komen, staat in die projectles. Alleen de demo heeft het nodig. Tijdens het downloaden wacht de hele les.*",
        "busy": "**CIFAR-10 downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van CIFAR-10 uitpakken** · dit duurt een paar seconden",
        "no_net": "*Het downloaden is niet gelukt: de University of Toronto was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "damaged": "*Het uitpakken is niet gelukt: het gedownloade bestand bleek beschadigd, dus de course heeft het verwijderd. Klik nog eens om het opnieuw te downloaden; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 170 MB en de uitgepakte foto's, 184 MB. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*CIFAR-10 staat op je computer: 50.000 trainingsfoto's en 10.000 testfoto's.*",
    })
    return (cifar,)


@app.cell(hide_code=True)
def variant_choice_cell():
    variant_choice = mo.ui.radio(options={"geen dropout, geen veranderingen": "none", "dropout": "dropout", "veranderingen": "changes", "dropout en veranderingen": "both"},
                                 value="dropout en veranderingen", label="Train met")
    train_button = mo.ui.run_button(label="Train je SmallAlexNet")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_alexnet(cifar, train_button, variant_choice, result_small):
    cifar, result_small  # opnieuw uitvoeren als CIFAR-10 binnen is, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je netwerk te trainen.*")
    else:
        _variant = variant_choice.value
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je SmallAlexNet op CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "De validation loss van je SmallAlexNet op de 1.000 testfoto's na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Je SmallAlexNet** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Je SmallAlexNet** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan staat je `SmallAlexNet` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `SmallAlexNet`, je `crop`, je `jitter` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "alexnet": "*Je `SmallAlexNet` in `work/projects/cnn/alexnet.py` stopt met een fout, of geeft geen 10 logits voor elke foto. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "changes": "*Je `crop` in `work/projects/cnn/crop.py` of je `jitter` in `work/projects/cnn/color.py` stopt met een fout, of geeft geen foto met shape (3, 32, 32) terug. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening in [Data augmentation voor foto's](/course/open/cv-augment-photos?lang=nl) opnieuw.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding onder *Waarom* zegt."
            _other = "changes" if _variant == "none" else "none"
            _names = {"none": "geen dropout, geen veranderingen", "dropout": "dropout", "changes": "veranderingen", "both": "dropout en veranderingen"}
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "none": "geen dropout, geen veranderingen", "changes": "veranderingen",
                       "alt": "De loss van je SmallAlexNet na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course met seed 0 " + ("met de veranderingen." if _other == "changes" else "zonder dropout en veranderingen, die daalt tot epoch 16 en daarna stijgt.")}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e, other=_other) + C.guesses_picture(_guesses[_e - 1], _result["truth"], L.lang),
                        f"Na epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, en {_right} van deze {len(C.SHOWN)} foto's goed gelezen.")

            if len(_when) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: wanneer is de validation loss het laagst, en wat gebeurt er daarna?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang, other=_other))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog steeds.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""Je netwerk leest **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, zoals ze zijn, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters.{_turn} Over 5 seeds, met de code van de course: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's, met de laagste validation loss na epoch {_be[0]} tot {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Goed gelezen na 40 epochs", "x": "goed gelezen (%)", "none": "geen dropout, geen veranderingen", "dropout": "dropout",
                                           "changes": "veranderingen", "both": "dropout en veranderingen", "color_none": "ColorCNN, geen veranderingen",
                                           "color_all": "ColorCNN, veranderingen", "mine": "je run: " + _names[_variant], "train": "training", "test": "test",
                                           "alt": "Rijen stippen tussen 40 en 100 procent goed gelezen: bovenaan je run, met je trainingsfoto's en testfoto's; daaronder de vier runs van SmallAlexNet van de course over 5 seeds, en in grijs ColorCNN zonder en met de veranderingen."}, L.lang, mine={"variant": _variant, "train": _result["train"], "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_dropout():
    mo.md(r"""
    ### Oefening 3. Wat dropout hier toevoegt
    """)
    return


@app.cell(hide_code=True)
def choice_dropout_cell():
    choice_dropout = L.choice(
        "dropout-here",
        question="Over 5 seeds, na 40 epochs. Zonder veranderingen leest SmallAlexNet 541 tot 576 testfoto's goed zonder dropout, en 542 tot 575 met dropout; zijn validation loss eindigt op 2,61 tot 3,45 zonder dropout, en op 1,87 tot 2,27 met. Met de veranderingen leest hij 594 tot 636 testfoto's goed zonder dropout, en 581 tot 604 met dropout. Wat zeggen de getallen hier over dropout?",
        options={
            "nothing": "Dropout doet hier niets: zonder veranderingen leest hij evenveel testfoto's goed",
            "both": "Dropout en veranderingen werken samen het best, zoals in AlexNet",
            "right": "Met de veranderingen leert SmallAlexNet zijn foto's niet meer uit zijn hoofd, en houdt dropout hem nog verder tegen: hij leest minder trainingsfoto's goed, en ook minder testfoto's",
        },
        correct="right",
        why={
            "nothing": "Evenveel foto's goed, ja, maar niet niets: zijn validation loss stijgt minder, tot 1,87 tot 2,27 in plaats van 2,61 tot 3,45, dus wordt hij minder zeker van zijn foute gokken. Hij krijgt er alleen niet meer goed. En met de veranderingen verlaagt dropout het aantal: 581 tot 604 in plaats van 594 tot 636.",
            "both": "Hier niet: met allebei 581 tot 604 testfoto's; met alleen de veranderingen 594 tot 636. Het netwerk uit het paper had linear layers van 4.096 getallen en 1,2 miljoen foto's; SmallAlexNet heeft er 256 en 6.000.",
            "right": "Met alleen de veranderingen leest hij 4.076 tot 4.221 van zijn trainingsfoto's goed, met aan het eind een training loss van 0,88 tot 0,94; met ook dropout 3.871 tot 3.953, en 1,02 tot 1,09. Volgens het paper heeft een netwerk met dropout ongeveer twee keer zoveel stappen nodig. Wat helpt, hangt af van het netwerk, de foto's en hoe lang het traint: meet het, zoals de course hier deed.",
        },
    )
    choice_dropout
    return (choice_dropout,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het CNN dat ILSVRC in 2012 won, met de ReLU uit [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl), de dropout uit [Dropout](/course/open/nn-dropout?lang=nl) en de veranderingen van de foto's uit [Data augmentation voor foto's](/course/open/cv-augment-photos?lang=nl), alle drie in één netwerk. Zijn convoluties volgen de regels uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl), en zijn parameters tel je zoals in [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl). VGG en ResNet kwamen erna, en ResNet-18, dat je gebruikte in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), hield zijn ReLU, zijn pooling en zijn veranderingen van de foto's. Je project `cnn` heeft nu `alexnet.py`. In de volgende les, *1×1 convolutions: de channels mengen*, leer je het kleinste filter dat er is kennen, dat ResNet al gebruikt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_today, result_small, choice_dropout):
    L.wrap_up(result_today, result_small, choice_dropout)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
