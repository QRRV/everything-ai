# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-color-cnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-color-cnn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) en [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="In je ResBlock is f(x) twee convoluties van 3 bij 3 van c naar c channels, elk met `padding=1`. Waarom die padding?",
        options={
            "speed": "Zodat het netwerk sneller traint",
            "right": "Zodat f(x) de hoogte en de breedte van x houdt, en x + f(x) opgeteld kan worden",
            "edges": "Zodat de pixels aan de rand meer meetellen dan die in het midden",
        },
        correct="right",
        why={
            "speed": "Padding kost eerder meer werk dan minder: een rand van nullen om elke feature map. Hij is er voor de shapes: zonder padding haalt elke convolutie van 3 bij 3 er 2 af van de hoogte en de breedte, en past x + f(x) niet meer. Zie [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).",
            "edges": "Door de nullen van de padding kan een filter ook op de pixels aan de rand staan, maar die tellen daardoor niet zwaarder mee. Hij is er voor de shapes: zonder padding haalt elke convolutie van 3 bij 3 er 2 af van de hoogte en de breedte, en past x + f(x) niet meer. Zie [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).",
            "right": "Zonder padding haalt elke convolutie van 3 bij 3 er 2 af van de hoogte en de breedte. Vandaag blijft het met dezelfde padding 32 bij 32 na elke convolutie van je CNN, en maken alleen de poolings het kleiner.",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def quick_check_norm():
    quick_norm = L.choice(
        "quick-norm",
        question="Voordat ResNet-18 een foto ziet, wordt elke kleur geschaald: een gemiddelde eraf, gedeeld door een standaardafwijking. Van wie zijn die getallen?",
        options={
            "own": "Het eigen gemiddelde en de eigen standaardafwijking van de foto",
            "fixed": "0,5 en 0,5, voor elk model en elke kleur",
            "right": "Die van de trainingsfoto's van ImageNet, de foto's waarvan hij leerde",
        },
        correct="right",
        why={
            "own": "Met zijn eigen getallen krijgt elke foto in elke kleur gemiddelde 0, en ziet een rode foto er niet meer rood uit. Zie [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl).",
            "fixed": "Sommige modellen gebruiken wel 0,5, maar ResNet-18 leerde zijn weights met de getallen van ImageNet, dus heeft een nieuwe foto die nodig. Zie [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl).",
            "right": "Een nieuwe foto krijgt de getallen waarmee het model zijn weights leerde. Vandaag train je zelf een netwerk: dan zijn het de getallen van je eigen trainingsfoto's.",
        },
    )
    quick_norm
    return (quick_norm,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je CNN's lazen tot nu toe Fashion-MNIST: grijze kledingstukken van 28 bij 28, elk in het midden, op zwart. Echte foto's zijn niet zo: ze hebben kleur en een achtergrond, en wat erop staat kan overal zitten, in elke grootte. **CIFAR-10** is daar de klassieke eerste stap: 60.000 kleurenfoto's van 32 bij 32 pixels in 10 soorten, van vliegtuigen tot vrachtwagens, in 2009 samengesteld door Alex Krizhevsky, Vinod Nair en Geoffrey Hinton aan de University of Toronto. Het ResNet-paper uit je vorige stap testte er netwerken met wel 1.202 layers op. In deze stap van je project geef je je CNN kleurenfoto's als input, en train je hem op CIFAR-10.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken", "Je CNN op kleurenfoto's", "Je eigen one-stage detector",
                            "Je eigen U-Net", "Eindproject: ResNet met Grad-CAM", "Eindproject: U-Net of detector"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_color():
    guess_color = L.guess(
        "color",
        question="Je SmallCNN las 864 tot 879 van de 1.000 testafbeeldingen van kleding goed, na 5 epochs op 6.000 trainingsafbeeldingen. Je geeft hem 3 channels als input, en traint hem op dezelfde manier op 6.000 kleurenfoto's van CIFAR-10. Hoeveel van zijn 1.000 testfoto's leest hij goed?",
        options={
            "same": "Ongeveer evenveel: 850 of meer",
            "right": "Ongeveer 550",
            "guess": "Ongeveer 100: niet beter dan gokken",
        },
        reveal="**Ongeveer 550.** Over 5 seeds, met de code van de course, leest hij 548 tot 561 testfoto's goed, tegen 864 tot 879 bij de kleding. Dat is veel beter dan gokken, waarmee je er 100 van de 1.000 goed hebt, maar ver onder de kleding: hetzelfde netwerk, dezelfde training, moeilijkere foto's. Van zijn eigen trainingsfoto's leest hij na 5 epochs 66 tot 70% goed, tegen 90 tot 91% bij de kleding.",
    )
    guess_color
    return (guess_color,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drie channels erin.** Een kleurenfoto is drie afbeeldingen op elkaar, rood, groen en blauw: shape (3, 32, 32). Een filter van de eerste convolutie heeft dan een 3 bij 3 voor elke kleur, shape (3, 3, 3). Op elke plek vermenigvuldigt het de 27 getallen eronder met zijn weights, telt ze allemaal op, de drie kleuren samen, en telt zijn bias erbij: één getal. Dus maakt `nn.Conv2d(3, 8, 3)` nog steeds 8 feature maps, uit filters van elk 27 weights. Alleen de eerste convolutie verandert; daarna rekent het netwerk verder met zijn 8 feature maps, net als voorheen.

        **Genormaliseerd met je eigen trainingsfoto's.** Net als ResNet-18 in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl) krijgt je netwerk elke kleur met een gemiddelde eraf en gedeeld door een standaardafwijking. Dit netwerk train je zelf, dus zijn het de getallen van je eigen trainingsfoto's: per kleur, over alle foto's, rijen en kolommen. Voor de eerste 6.000 foto's van CIFAR-10 zijn de gemiddelden `(0.492, 0.482, 0.445)` en de standaardafwijkingen `(0.248, 0.245, 0.263)`, voor rood, groen en blauw. De testfoto's krijgen dezelfde getallen, zoals elke nieuwe foto.

        **Waarom foto's moeilijker zijn.** De kledingstukken van Fashion-MNIST liggen in het midden, allemaal even groot, op zwart. In CIFAR-10 kan een kat dichtbij zijn of ver weg, links of rechts, op een bank of in het gras, en 32 bij 32 pixels laten weinig details zien. De soorten die het netwerk van deze les het vaakst verwart, over 5 seeds, zijn vogel en hert, vliegtuig en schip, kat en hond, en auto en vrachtwagen: soorten met een vergelijkbare vorm, of met een vergelijkbare achtergrond, zoals een blauwe lucht en een blauwe zee.

        **Uit je hoofd leren.** Met 6.000 foto's die zo verschillen, kan een netwerk zijn trainingsfoto's uit zijn hoofd leren: details van deze foto's die niet voor nieuwe gelden. Dat is de overfitting uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl): de training loss blijft dalen, terwijl de validation loss een laagste punt bereikt en weer stijgt.

        **Waar CIFAR-10 vandaan komt.** De foto's komen uit 80 Million Tiny Images, een veel grotere set kleine foto's die op het web gevonden zijn door op woorden te zoeken. In 2020 haalde MIT die set offline, omdat sommige labels beledigende woorden waren, en sommige foto's ook beledigend waren. CIFAR-10 kreeg met de hand opnieuw labels, voor zijn eigen 10 soorten, en wordt nog veel gebruikt. Op de pagina ervan staat geen licentie, dus de course levert CIFAR-10 niet mee en deelt geen model dat erop getraind is: de knop in het uitgewerkte voorbeeld downloadt het één keer, bij de makers.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **CIFAR-10** | sie-far tien | 60.000 kleurenfoto's van 32 bij 32 pixels in 10 soorten, 50.000 om op te trainen en 10.000 om mee te testen (Krizhevsky, 2009) | `everything_ai.cifar` |
        """),
        L.watch_out("Normaliseer de testfoto's met de getallen van de trainingsfoto's, niet met hun eigen: het netwerk leerde zijn weights met die van de trainingsfoto's. En een gemiddelde per kleur gaat over drie axes tegelijk, de foto's, de rijen en de kolommen: neem je het alleen over de rijen en de kolommen, dan krijg je een gemiddelde per foto."),
        mo.md(r"""
        Het plaatje: hoeveel afbeeldingen het netwerk goed leest, van zijn eigen trainingsafbeeldingen, gevuld, en van de testafbeeldingen, open, een stip voor elk van 5 seeds. Op kleding en op kleurenfoto's, 5 en 20 epochs getraind, als je SmallCNN met 1 of 3 channels als input. Bij kleding blijven ze binnen ongeveer 10 procentpunten van elkaar. Bij kleurenfoto's liggen ze na 20 epochs meer dan 40 procentpunten uit elkaar: door de langere training werd hij beter in zijn eigen foto's, en slechter in nieuwe.
        """),
        mo.Html(C.gap_picture({"title": "Goed gelezen: trainings- en testafbeeldingen", "x": "goed gelezen (%)",
                               "clothes": "kleding, {epochs} epochs", "photos": "kleurenfoto's, {epochs} epochs",
                               "train": "training", "test": "test", "lang": L.lang,
                               "alt": "Vier rijen stippen tussen 40 en 100 procent goed gelezen. Kleding na 5 epochs: trainingsafbeeldingen ongeveer 91%, testafbeeldingen ongeveer 87%. Kleding na 20 epochs: training ongeveer 98%, test ongeveer 87%. Kleurenfoto's na 5 epochs: training ongeveer 69%, test ongeveer 56%. Kleurenfoto's na 20 epochs: training ongeveer 95%, test ongeveer 53%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een CNN voor kleurenfoto's:

    1. De foto's als floats van 0 tot en met 1, met shape (N, 3, 32, 32): de bytes gedeeld door 255.
    2. Per kleur het gemiddelde en de standaardafwijking van de trainingsfoto's, over de foto's, de rijen en de kolommen.
    3. Normaliseer de trainingsfoto's en de testfoto's met die getallen: haal per kleur het gemiddelde eraf en deel door de standaardafwijking.
    4. De eerste convolutie krijgt 3 channels als input; elk van zijn filters heeft een 3 bij 3 voor elke kleur.
    5. Reken de maten uit tot aan de output head: met padding 1 halveert elke pooling van 2 de 32, tot 16 en dan tot 8.
    6. Train, en vergelijk de trainingsfoto's met de testfoto's: hoeveel hij van elk goed leest, en waar de validation loss weer gaat stijgen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel laadt CIFAR-10, en print de shapes van de foto's en de soorten van de eerste 8 testfoto's. Hij neemt één kikker, stap 1, en print het gemiddelde per kleur, over de rijen en de kolommen: stap 2, voor één foto. Hij stuurt de kikker door een convolutie van 3 naar 8 channels, stap 4, en print de shapes van de filters en van wat eruit komt. Dan rekent hij één getal van de eerste feature map met de hand uit, op rij 10 en kolom 20, uit de 3 bij 3 van elke kleur rond die plek: met padding 1 de rijen 9 tot en met 11 en de kolommen 19 tot en met 21. De regel met de hand is nog niet af: hij neemt alleen het rode channel. Voer de cel uit zoals hij is, maak hem dan af waar het commentaar zegt, en voer hem opnieuw uit. `cifar` is `everything_ai.cifar` van de course, die de cel hieronder geeft zodra CIFAR-10 gedownload is.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="CIFAR-10 downloaden, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. De knop downloadt het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. Het uitgewerkte voorbeeld, het plaatje van de foto's en de demo hebben het nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
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


@app.cell
def example_filter(cifar):
    import torch
    from torch import nn

    X_train, y_train, X_test, y_test = cifar.load()
    print(X_train.shape, X_train.dtype, X_test.shape)
    print([cifar.KINDS[k] for k in y_test[:8]])
    # 1. een kikker als floats van 0 tot en met 1, met voorop een axis voor de batch
    i = cifar.checked("frog", 1)[0]
    x = torch.tensor(X_test[i:i + 1], dtype=torch.float32) / 255
    print(x.shape)
    # 2. het gemiddelde per kleur, over de rijen en de kolommen
    print(x[0].mean(dim=(1, 2)))
    # 4. een convolutie van 3 naar 8 channels
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding=1)
    with torch.no_grad():
        maps = conv(x)
    print(conv.weight.shape, maps.shape)
    # met de hand: filter 0 op de 3 bij 3 rond rij 10, kolom 20
    patch = x[0, :, 9:12, 19:22]
    by_hand = (patch[0] * conv.weight[0, 0]).sum() + conv.bias[0]  # maak af: (patch * conv.weight[0]).sum() + conv.bias[0]
    print(round(maps[0, 0, 10, 20].item(), 4), round(by_hand.item(), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De foto's zijn bytes, `uint8`, met de kleuren voorop: 50.000 trainingsfoto's en 10.000 testfoto's met shape (3, 32, 32). De eerste 8 testfoto's zijn een kat, twee schepen, een vliegtuig, twee kikkers, een auto en een kikker; foto 4 is de eerste kikker. Het gemiddelde van de kikker is 0,423 voor rood, 0,479 voor groen en 0,362 voor blauw: een groenige kikker. Oefening 2 neemt hetzelfde gemiddelde over alle foto's tegelijk. De 8 filters samen hebben shape (8, 3, 3, 3): een 3 bij 3 voor elk van de 3 kleuren. Wat eruit komt, zijn 8 feature maps van 32 bij 32, want padding 1 houdt de grootte gelijk. Op rij 10 en kolom 20 staat in de eerste map −0,2426. Zoals de cel nu is, met alleen het rode channel, geeft de regel met de hand −0,1844. Afgemaakt, met alle drie de kleuren, geeft hij ook −0,2426: één filter telt op wat het in elke kleur ziet.
    """)
    return


@app.cell(hide_code=True)
def photos_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    mo.vstack([
        mo.md("De eerste 4 foto's van elke soort die de course een voor een bekeken heeft, vergroot zodat je elke pixel ziet. Sommige foto's van CIFAR-10 laten naast het dier of voertuig een mens zien, zoals een ruiter op een paard; de les laat alleen foto's zonder mensen zien."),
        mo.Html(C.photos_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De weights van een kleurenfilter
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "color-weights",
        question="Hoeveel parameters heeft `nn.Conv2d(3, 8, 3)`, de eerste convolutie van een CNN voor kleurenfoto's?",
        options={
            "gray": "80: 8 filters van 3 bij 3, en 8 biases",
            "right": "224: 8 filters van 3 bij 3 bij 3, en 8 biases",
            "apart": "240: 8 filters van 3 bij 3 voor elke kleur, 24 in totaal, en 24 biases",
        },
        correct="right",
        why={
            "gray": "Dat is `nn.Conv2d(1, 8, 3)`, voor grijze afbeeldingen. Met 3 channels als input heeft elk filter een 3 bij 3 voor elke kleur: 27 weights.",
            "apart": "Een filter kijkt niet naar één kleur: het kijkt naar alle drie tegelijk, en telt ze op tot één feature map. Dus maken 8 filters 8 feature maps, geen 24.",
            "right": "8 × 3 × 3 × 3 = 216 weights, in een tensor met shape (8, 3, 3, 3), en 8 biases: 224. Kleur kost de eerste convolutie 3 keer zoveel weights, en de rest van het netwerk niets.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_stats():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De getallen van elke kleur

        Maak `channel_stats(X)` af: geef voor foto's `X` met shape (N, 3, H, W) het gemiddelde en de standaardafwijking van elke kleur terug, over alle foto's, rijen en kolommen, stap 2 van het recept. Allebei zijn het tensors met shape (3,). `.mean` en `.std` van PyTorch nemen meerdere axes tegelijk, als tuple: `X.mean(dim=(1, 2))` neemt het gemiddelde over axis 1 en 2 samen. De check geeft je functie verzonnen foto's, en vergelijkt de shapes en de getallen. Als het goed is, bewaart hij je functie in je project, als `work/projects/cnn/channels.py`: de demo normaliseert de foto's ermee.
        """),
        L.task("Laat `channel_stats` het gemiddelde en de standaardafwijking van elke kleur teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stats():
    box_stats = L.code_box("channel-stats", lines=8)
    box_stats
    return (box_stats,)


@app.cell(hide_code=True)
def check_stats(box_stats):
    result_stats = L.check("channel-stats", box_stats.value)
    result_stats
    return (result_stats,)


@app.cell(hide_code=True)
def hints_stats():
    L.hints(
        "channel-stats",
        [
            "In `X` staan op axis 0 de foto's, op axis 1 de kleuren, en op axis 2 en 3 de rijen en de kolommen. Houd de kleuren: neem het gemiddelde over alle andere axes.",
            "De axes gaan als tuple in `dim`. De standaardafwijking gaat net zo als het gemiddelde, met `.std`. Geef ze allebei terug, met een komma ertussen.",
        ],
        why="`.std` van PyTorch deelt door N − 1, zoals de standaardafwijking van een steekproef; met `unbiased=False` deelt hij door N. Over 6.000 foto's van 32 bij 32 heeft elke kleur 6.144.000 getallen, en verschillen de twee pas in de achtste decimaal: de check rekent ze allebei goed.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cnn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je CNN in kleur

        `ColorCNN` is je SmallCNN, waarin twee getallen zijn weggelaten: vul ze in voor kleurenfoto's van 32 bij 32, stap 4 en 5 van het recept. De check maakt netwerken na `torch.manual_seed(0)` en vergelijkt de logits van 2 en van 3 willekeurige foto's, en hoeveel getallen zijn parameters bevatten. Als het goed is, bewaart de check hem in je project, als `work/projects/cnn/color_cnn.py`.
        """),
        L.task("Laat `ColorCNN` kleurenfoto's van 32 bij 32 als input krijgen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cnn():
    box_cnn = L.code_box("color-cnn", lines=17)
    box_cnn
    return (box_cnn,)


@app.cell(hide_code=True)
def check_cnn(box_cnn):
    result_cnn = L.check("color-cnn", box_cnn.value)
    result_cnn
    return (result_cnn,)


@app.cell(hide_code=True)
def hints_cnn():
    L.hints(
        "color-cnn",
        [
            "Het eerste getal van `nn.Conv2d` is het aantal channels dat binnenkomt. Hoeveel heeft een kleurenfoto er?",
            "Met padding 1 behouden de convoluties 32 bij 32, en elke pooling halveert het. Hoe groot zijn de 16 feature maps na de tweede pooling, en hoeveel getallen bevatten ze samen?",
        ],
        why="16 channels van 8 bij 8 zijn 1.024 getallen per foto, tegen 784 bij de kleding. De output head heeft dan 10.250 van de 11.642 parameters van het netwerk: bijna 9 op de 10.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij neemt de eerste 6.000 trainingsfoto's van CIFAR-10 en de eerste 1.000 testfoto's, evenveel als bij de kleding, en normaliseert ze allebei met de getallen die je `channel_stats` voor de trainingsfoto's geeft. Hij traint je `ColorCNN` met je Trainer: {C.EPOCHS} epochs, een learning rate van {C.number(C.LR, 2, L.lang)} met een warmup van één epoch, met seed 0, en telt zijn parameters met je `count_params`. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Hij heeft CIFAR-10 nodig, uit het uitgewerkte voorbeeld, en oefening 2 en 3, en duurt ongeveer een halve minuut. Tijdens het trainen laat een grafiek de validation loss na elke epoch zien, op de 1.000 testfoto's. Gokken uit 10 soorten geeft een loss van $\\ln 10 \\approx 2{{,}}30$. Bij epoch 20 kan de validation loss even hoog zijn, terwijl het netwerk nog ongeveer de helft van de testfoto's goed leest: bij de foto's die het fout heeft, is het erg zeker van een verkeerde soort, en dat kost meer dan een gok.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je ColorCNN")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_color(train_button, result_stats, result_cnn):
    result_stats, result_cnn  # run again when exercise 2 or 3 saves your module
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je netwerk te trainen.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je ColorCNN: de validation loss", "x": "epoch", "y": "validation loss",
                                        "alt": "De validation loss van je ColorCNN op de 1.000 testfoto's na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Je ColorCNN** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Je ColorCNN** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `channel_stats` en je `ColorCNN` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop in het uitgewerkte voorbeeld, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan vijf minuten om te trainen. Doet je `ColorCNN` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "stats": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet voor elk van de 3 kleuren een gemiddelde en een standaardafwijking boven 0. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `ColorCNN` in `work/projects/cnn/color_cnn.py` stopt met een fout, of geeft geen 10 logits per foto. Heb je zijn bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan weer.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
            _pic = [mo.Html(C.curves_picture(_result["curves"], {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                                                                 "alt": "De loss van je ColorCNN na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestippeld, met een stip waar de validation loss het laagst is."}, L.lang))] if _result["curves"]["val"] else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" De validation loss was het laagst na epoch **{_best}**; daarna steeg hij, terwijl de training loss bleef dalen." if _best < C.EPOCHS else f" De validation loss daalde na epoch {_best} nog steeds.")
            _numbers = "; ".join(C.number(_v, 3, L.lang) for _v in _result["mean"]), "; ".join(C.number(_v, 3, L.lang) for _v in _result["std"])
            _right = sum(_g == _k for _, _k, _g in _result["guesses"])
            _out = mo.vstack(_pic + [
                mo.md(f"""Je netwerk leest **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters, geteld met je `count_params`.{_turn} Over 5 seeds, met de code van de course: 5.514 tot 5.800 trainingsfoto's en 515 tot 543 testfoto's, met de laagste validation loss na epoch 3 tot 6. Je `channel_stats` gaf de gemiddelden ({_numbers[0]}) en de standaardafwijkingen ({_numbers[1]}).{_who}"""),
                mo.md(f"Hieronder {C.SHOWN} testfoto's van elke soort die de course bekeken heeft, met de gok van je netwerk: {_right} van de {len(_result['guesses'])} zijn goed. Waar het fout is, staat de goede soort tussen haakjes."),
                mo.Html(C.guesses_picture(_result["guesses"], L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.md(r"""
    ### Oefening 4. Wat de curves zeggen
    """)
    return


@app.cell(hide_code=True)
def choice_gap_cell():
    choice_gap = L.choice(
        "color-gap",
        question="Met de code van de course is de validation loss het laagst na epoch 3 tot 6, en stijgt hij daarna tot tussen 2,179 en 2,386 bij epoch 20, terwijl de training loss blijft dalen, tot tussen 0,164 en 0,305. Wat gebeurt er?",
        options={
            "rate": "De learning rate is te hoog, dus het netwerk raakt de weg kwijt",
            "harder": "De testfoto's zijn moeilijker dan de trainingsfoto's",
            "right": "Het netwerk leert zijn trainingsfoto's uit zijn hoofd: details die niet voor nieuwe foto's gelden",
        },
        correct="right",
        why={
            "rate": "Met een te hoge learning rate springt de training loss heen en weer, of stijgt hij ook. Hier blijft hij dalen, en na de warmup maakt de cosine decay van je Trainer de learning rate bij elke stap kleiner.",
            "harder": "Het zijn allebei foto's van dezelfde 10 soorten, uit dezelfde set; met seed 0 waren de twee losses na epoch 3 bijna gelijk, 1,34 en 1,38. Wat verandert, is het netwerk, niet de foto's.",
            "right": "Na epoch 3 tot 6 leert hij meer over zijn 6.000 foto's dan over de 10 soorten: hij leest meer van zijn trainingsfoto's goed en minder nieuwe. Meer foto's, of trainingsfoto's die elke epoch anders zijn, maken het moeilijker om ze uit het hoofd te leren.",
        },
    )
    choice_gap
    return (choice_gap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je SmallCNN uit [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl), nu met drie channels als input, en genormaliseerd met de getallen van zijn eigen trainingsfoto's, zoals ResNet-18 met die van ImageNet in [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl). Op kleurenfoto's groeit het gat tussen trainings- en testafbeeldingen uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) van ongeveer 10 procentpunten bij kleding tot meer dan 40. Je project `cnn` heeft nu `channels.py` en `color_cnn.py`. In de volgende les, *Data augmentation voor foto's: spiegelen, uitsnijden en kleur*, krijgt je `ColorCNN` elke epoch zijn trainingsfoto's een beetje anders te zien: gespiegeld, uitgesneden, met andere kleuren, zodat hij ze minder makkelijk uit zijn hoofd leert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_stats, result_cnn, choice_gap):
    L.wrap_up(choice_weights, result_stats, result_cnn, choice_gap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
