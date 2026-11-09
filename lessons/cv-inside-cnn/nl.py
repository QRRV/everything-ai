# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-inside-cnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-inside-cnn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) en [Een beeldclassifier uitproberen: top-1 en top-5](/course/open/cv-try-classifier?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_filter():
    quick_filter = L.choice(
        "quick-filter",
        question="Je ColorCNN begint met `nn.Conv2d(3, 8, 3, padding=1)`. Hoeveel weights heeft één van zijn 8 filters?",
        options={
            "nine": "9: een 3 bij 3",
            "right": "27: een 3 bij 3 voor elk van de 3 kleuren",
            "all": "72: 8 keer een 3 bij 3",
        },
        correct="right",
        why={
            "nine": "Een filter van een kleurenfoto kijkt naar alle 3 de kleuren tegelijk: een 3 bij 3 voor elke kleur, 3 × 3 × 3 = 27 weights. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "all": "8 × 9 telt de 8 filters met elk één 3 bij 3. Eén filter heeft een 3 bij 3 voor elk van de 3 kleuren, 27 weights, en de layer 8 × 27 = 216. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "right": "De filters van vandaag, die van de eerste layer van ResNet-18, zijn 7 bij 7: 3 × 7 × 7 = 147 weights per filter, genoeg om als een klein kleurenplaatje te tekenen.",
        },
    )
    quick_filter
    return (quick_filter,)


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="ResNet-18 zette espresso bovenaan voor de koffiefoto, met 0,85. Waar bewaart hij hoe espresso eruitziet?",
        options={
            "photos": "In espressofoto's uit ImageNet waarmee hij de foto vergelijkt",
            "right": "In zijn 11,7 miljoen weights, geleerd van de foto's van ImageNet",
            "names": "In het bestand met de namen van de 1000 klassen",
        },
        correct="right",
        why={
            "photos": "Het bestand van ResNet-18, 47 MB, bevat zijn weights en geen foto's. Training veranderde die weights tot ze de goede logits gaven voor de foto's van ImageNet. Zie [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl).",
            "names": "Dat bestand geeft alleen elk van de 1000 logits zijn naam, op volgorde. Hoe de klassen eruitzien, zit in de weights, geleerd van de foto's van ImageNet. Zie [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl).",
            "right": "Vandaag kijk je naar een deel van die weights, en naar wat ze voor een foto uitrekenen.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    ResNet-18 heeft een top-1 accuracy van zo'n 70% op de validatiefoto's van ImageNet, maar zijn 11,7 miljoen weights zeggen niet waar hij naar kijkt. Je kunt het wel bekijken. De filters van zijn eerste layer zijn kleine plaatjes, en voor elke foto geeft elke layer feature maps die laten zien waar in de foto hij vond waar hij naar zoekt. Zeiler en Fergus (2014) tekenden zo de eerste filters van AlexNet, en volgden op de foto terug waar zijn diepere layers op reageren. Ze zagen dat sommige van zijn eerste filters niets deden, dat andere alleen heel fijne en heel grove patronen hadden, en dat zijn stride van 4 een rommeltje maakte van de tweede layer. Met kleinere filters, 7 bij 7 in plaats van 11 bij 11, en een stride van 2 werd hun netwerk beter. De eerste layer van ResNet-18 heeft die maat. Vandaag teken je zijn filters en zijn feature maps, en zie je hoeveel van een foto één getal diep binnenin bekijkt.
    """)
    return


@app.cell(hide_code=True)
def predict_filters():
    guess_filters = L.guess(
        "filters-look",
        question="De eerste layer van ResNet-18 heeft 64 filters van 7 bij 7, voor de 3 kleuren. Hoe zien ze eruit als je ze tekent als kleine kleurenplaatjes, met grijs voor een weight van 0?",
        options={
            "noise": "Als ruis: training laat geen patroon achter dat je kunt zien",
            "right": "Vooral randen en lijnen onder elke hoek, en vlekken van één kleur",
            "things": "Als stukjes van de dingen in ImageNet: ogen, wielen, letters",
        },
        reveal="**Vooral randen en lijnen, en vlekken kleur.** Licht naast donker onder elke hoek en in verschillende breedtes; een rode vlek, een groene, een blauwe. De eerste layer van AlexNet zag er in 2012 ongeveer net zo uit. Stukjes van dingen, zoals ogen, komen dieper: oefening 4 gaat over een map van `layer3` die oplicht rond de ogen van de kat, en de demo laat die zien. En 8 van de 64 blijken helemaal niets te doen: het uitgewerkte voorbeeld laat er een zien.",
    )
    guess_filters
    return (guess_filters,)


@app.cell(hide_code=True)
def text_idea():
    _fields = C.field_frames({"alt": "De foto van de kat met voor elke fase een vierkant rond het midden van het oog links, het vierkant na {stage} dik, en dat oog in een gestippeld vak."})
    _field = L.explore("field", "receptive field", lambda _v: (_fields[int(_v)], C.field_sentence(int(_v), L.lang)),
                       start=7, values=list(C.FIELD_VIEWS), role="",
                       task="Schuif van 7 naar 435: vanaf welke fase ziet één getal het hele oog links, het gestippelde vak?")
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een filter is een klein plaatje.** De weights van de eerste layer van ResNet-18, `conv1.weight`, hebben shape (64, 3, 7, 7): 64 filters, elk een 7 bij 7 voor elk van de 3 kleuren, 147 weights. Elk getal van een feature map is een som van weight keer pixel over een stuk van 7 bij 7 van de foto, in alle 3 de kleuren, zoals in [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl). Het is het grootst waar de foto op het filter lijkt: licht waar de weights positief zijn, donker waar ze negatief zijn. Een filter dat je als plaatje tekent, laat dus zien waar het naar zoekt. Zijn weights liggen rond 0, positief en negatief, dus ze hebben een schaal nodig: deel door twee keer de grootste weight van het filter, + of -, en tel er 0,5 bij op. Een weight van 0 wordt dan 0,5, grijs; de grootste weight wordt 0 of 1; een positieve weight is lichter in zijn eigen kleur, een negatieve donkerder. `w.abs()` maakt elk getal positief, zoals `np.abs` doet. Een filter heeft zijn kleuren vooraan, (3, 7, 7), zoals een netwerk ze wil, en `plt.imshow` wil rijen, kolommen en kleuren: `.permute(1, 2, 0)` zet axis 1 vooraan, dan axis 2, dan axis 0, de stap van [Een foto klaarmaken voor een model](/course/open/cv-prepare-photo?lang=nl) andersom. De 64 filters naast elkaar heten een **filter grid**.

        **Feature maps, fase na fase.** ResNet-18 verwerkt een foto in fases. Eerst `conv1`, 64 filters van 7 bij 7 met stride 2, dan `bn1`, batch normalization, en een ReLU: 64 feature maps van 112 bij 112 voor een foto van 224 bij 224. Dan halveert max pooling van 3 bij 3 met stride 2 en padding 1 de rijen en de kolommen, en volgen er vier fases, `layer1` tot en met `layer4`, elk van twee residual blocks zoals in [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl). `layer1` geeft 64 maps van 56 bij 56. `layer2`, `layer3` en `layer4` beginnen elk met een stride van 2, dus halveert elk de rijen en de kolommen van de maps die hij krijgt, en heeft hij twee keer zoveel channels. Je kunt een map tekenen als een grijs plaatje: zwart voor 0, wit voor zijn grootste getal. De ReLU maakt elk negatief getal 0: op de kat is na `conv1` 29% van de getallen 0. De maps van een fase naast elkaar heten een **feature map grid**, de sterkste eerst: die met het hoogste gemiddelde.

        **Hoeveel van de foto één getal ziet.** Zoals in [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl) hangt elk getal van een map af van een stuk van de foto, zijn receptive field, en maakt elke layer dat stuk groter: een 3 bij 3 voegt 2 keer de afstand tussen buurgetallen toe. Na `conv1` 7 bij 7 pixels; na de pooling 11; na `layer1` 43; na `layer2` 99; na `layer3` 211; na `layer4` 435: meer dan de hele foto. De schuifregelaar laat dat stuk zien voor het getal van elke map dat het dichtst bij het midden van het oog links ligt: dik voor de fase die je kiest, dun voor de andere.
        """),
        _field,
        mo.md(r"""
        Vanaf `layer2` ziet één getal een heel oog. Hoe dieper, hoe meer een getal ziet, dus kan het op meer reageren dan een rand: op een deel, zoals een oog, en dan op een heel ding. De pixels dicht bij het midden van dat stuk tellen het meest. Vraag autograd hoe sterk elke pixel channel 120 van `layer3` op rij 7 en kolom 7 beweegt, een getal waarvan de 211 bij 211 pixels allemaal op de kat liggen: een derde daarvan komt uit de middelste 49 bij 49, 5% van het stuk, en 82% uit de middelste 97 bij 97, 21% van het stuk. Daarom laat een map van `layer3` nog steeds zien waar in de foto iets zit.
        """),
        L.watch_out("Als je elk filter schaalt met zijn eigen grootste weight, zie je het patroon van elk filter, ook van een filter dat niets doet. 8 van de 64 filters van ResNet-18 hebben weights kleiner dan 0,00001, tegen grootste weights van 0,06 tot 1,02 voor de andere, en `bn1` geeft hun maps ook een weight van bijna 0: op elke foto zijn hun maps 0. Kijk naar de grootte van de weights van een filter, `w.abs().max()`, en naar zijn map, voordat je zegt waar het naar zoekt."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **filter grid** | de filters van een layer naast elkaar, elk getekend als een klein plaatje | |
        | **feature map grid** | de feature maps van een fase naast elkaar, elk getekend als een grijs plaatje | |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kijk je in een getrainde CNN:

    1. Zijn eerste filters: `model.conv1.weight`, met shape (64, 3, 7, 7) in ResNet-18. Teken elk met grijs voor 0: `0.5 + w / (2 * w.abs().max())`, dan `.permute(1, 2, 0)`.
    2. Kijk naar de grootte van zijn weights, `w.abs().max()`: een filter waarvan alle weights dicht bij 0 liggen, doet niets, hoe duidelijk zijn plaatje er ook uitziet.
    3. Maak een foto klaar zoals voor het model, en haal hem fase voor fase door het model, binnen `with torch.no_grad():`, en bewaar wat elke fase geeft: zijn feature maps.
    4. Teken een map als een grijs plaatje, van zwart, 0, tot wit, zijn grootste getal: `cmap="gray", vmin=0`.
    5. Zoek waar een map het hoogst is, en het stuk van de foto dat dat getal ziet: zijn receptive field. Kijk naar meer foto's voordat je zegt waar een filter of een channel naar zoekt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel pakt de weights van `conv1` en print hun shape, maakt een plaatje van filter `k`, en print de shape ervan en het donkerste en het lichtste getal. Dan print hij de grootste weight van het filter, + of -, en tekent hij het. De tweede cel maakt de kat klaar zoals in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), haalt hem door de eerste fase, `conv1`, `bn1` en een ReLU, en print de shape van de maps en het deel van hun getallen dat 0 is. Dan tekent hij de map van filter `k`: `vmin=0` maakt 0 zwart, waar `plt.imshow` anders het kleinste getal van de map zwart maakt. Als laatste print hij waar die map het hoogst is: `argmax` telt de getallen van de map rij na rij, 112 per rij, dus geeft `// 112` de rij en `% 112` de kolom. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer ze uit zoals ze zijn, met filter 26; verander daarna `k` in de eerste cel waar het commentaar het zegt, en voer die opnieuw uit: de tweede cel draait dan vanzelf opnieuw, omdat hij `k` gebruikt. `resnet` is ResNet-18, die de cel hieronder laadt zodra hij gedownload is.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="ResNet-18 downloaden, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 staat nog niet op je computer. Als je [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl) op deze computer hebt gedaan, staat hij er wel: dan vindt deze les hem. Anders downloadt de knop hem één keer, 47 MB, van Hugging Face. Het uitgewerkte voorbeeld, de schuifregelaar langs de filters en de demo hebben hem nodig; de oefeningen niet. Terwijl hij downloadt, wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**ResNet-18 downloaden** · {done} van {total} MB",
        "no_net": "*De download lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus heeft de course het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 47 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
        "here": "*ResNet-18 staat op je computer: 11,7 miljoen parameters, met de weights van torchvision, getraind op ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_filters(resnet):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    # 1. de filters van de eerste layer: 64, elk een 7 bij 7 voor elk van de 3 kleuren;
    #    filter k als plaatje: grijs voor een weight van 0, de kleuren achteraan
    w = resnet.conv1.weight
    print(w.shape)
    k = 26  # verander: 9
    f = w[k]
    img = (0.5 + f / (2 * f.abs().max())).permute(1, 2, 0)
    print(img.shape, img.min(), img.max())
    # 2. de grootte van zijn weights: de grootste, + of -
    print(f.abs().max())
    plt.imshow(img)
    plt.show()
    return Image, k, np, plt, torch


@app.cell
def example_maps(Image, k, np, plt, resnet, torch):
    # 3. de kat, klaargemaakt zoals in Een beeldclassifier uitproberen, door de eerste fase: conv1, bn1 en een ReLU
    photo = Image.open("data/photos/chelsea.png")
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    crop = small[16:240, 80:304]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        maps = torch.relu(resnet.bn1(resnet.conv1(x)))
    print(maps.shape, (maps == 0).float().mean())
    # 4. de map van filter k op de kat: zwart is 0, wit zijn grootste getal
    plt.imshow(maps[0, k], cmap="gray", vmin=0)
    plt.show()
    # 5. waar hij het hoogst is: argmax telt rij na rij, 112 getallen per rij
    top = int(maps[0, k].argmax())
    print(top // 112, top % 112)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De weights van `conv1` hebben shape (64, 3, 7, 7). Filter 26 als plaatje heeft shape (7, 7, 3), en zijn getallen gaan van 0,0094 tot 1: zijn grootste weight, 0,2763, is positief, dus wordt die 1. Filter 26 is boven donker en onder licht: een horizontale rand. De maps hebben shape (1, 64, 112, 112), en 29% van hun getallen is 0, `tensor(0.2935)`. De map van filter 26 is het hoogst op rij 62 en kolom 96. Buurgetallen van een map van `conv1` liggen 2 pixels van de foto uit elkaar, dus ziet dat getal de 7 bij 7 pixels rond rij 124 en kolom 192 van de foto: vlak onder het oog rechts, waar het donkere oog overgaat in de lichtere vacht eronder. Met `k = 9` ziet het filter eruit als een blauwe vlek. Maar zijn grootste weight is `1.0849e-06`, 1,0849 keer 10 tot de macht -6: 0,0000011. Zijn map is zwart, overal 0, dus is zijn hoogste plek gewoon zijn eerste getal: rij 0, kolom 0. Filter 9 is een van de 8 die niets doen; de schaal maakte alleen zijn piepkleine weights zichtbaar.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Welk filter zoekt waarnaar? Het grid laat de 64 filters van `conv1` zien, 8 per rij: het getal links van een rij is het eerste filter ervan, en het getal boven een kolom zegt hoeveel je verder telt, dus filter 26 staat in rij 24, kolom 2. De schuifregelaar gaat ze langs: links het filter, rechts zijn map op de kat, van zwart, 0, tot wit, zijn grootste getal.
    """)
    return


@app.cell(hide_code=True)
def grid_cell(resnet):
    mo.Html(C.grid_picture(C.filter_images(resnet.conv1.weight), {
        "title": "De 64 filters van conv1",
        "alt": "Vierenzestig kleine gekleurde vierkantjes in 8 rijen van 8: de meeste laten lichte en donkere strepen of randen onder verschillende hoeken zien, een paar een vlek van één kleur, zoals rood, groen, blauw of magenta, en een paar een egaal kleurvlak.",
    }))
    return


@app.cell(hide_code=True)
def explore_filters_cell(resnet):
    _maps = C.course_maps("cat")[0]
    _frames = C.filter_frames(resnet.conv1.weight, _maps, {"filter": "filter {k}", "map": "zijn map op de kat",
                                                          "alt": "Filter {k} van conv1 als klein kleurenplaatje, en ernaast zijn feature map op de foto van de kat, in grijs."})

    def _frame(_k):
        _k = int(_k)
        return _frames[_k], C.filter_sentence(_maps, resnet.conv1.weight, _k, L.lang)

    explore_filters = L.explore("filter", "filter", _frame, start=26, values=list(range(64)), role="",
                                task="Zoek een filter waarvan de map het lichtst is op de roze neus van de kat, en een waarvan de map overal 0 is terwijl zijn grootste weight niet piepklein is.")
    explore_filters
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Filter 28, een rode vlek, is het hoogst op de roze neus, op rij 93 en kolom 73. Maar de oranje vacht laat bijna zijn hele map ook oplichten: geen getal ervan is 0. Daardoor is zijn map de sterkste van de 64 op de kat, en ook op de foto's van de koffie en het fruit. De maps van de filters 17 en 18 zijn overal 0 op de kat, terwijl hun grootste weights 0,08 en 0,28 zijn: `bn1` schuift hun maps omlaag, zodat alleen een stuk foto dat sterk op ze lijkt boven 0 komt. Filter 17 is een lichte vlek op rood, en die heeft de kat niet; op de foto's van de koffie en het fruit is zijn map niet 0. De meeste filters zijn randen onder een of andere hoek. De filters 26 en 57 zijn elkaars tegenpolen, donker boven licht en licht boven donker: 26 is het hoogst onder het oog rechts, 57 op de bovenrand van het oog links.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De shape na layer3
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "inside-shape",
        question="De foto van de kat, met shape (1, 3, 224, 224), gaat door ResNet-18. Na `conv1`, `bn1` en de ReLU is hij (1, 64, 112, 112); na de pooling en `layer1` (1, 64, 56, 56). Welke shape geeft `layer3`?",
        options={
            "wide": "(1, 256, 56, 56)",
            "right": "(1, 256, 14, 14)",
            "two": "(1, 128, 28, 28)",
        },
        correct="right",
        why={
            "wide": "De channels kloppen: 64, 128, 256. Maar `layer2` en `layer3` beginnen elk met een stride van 2, dus halveert elk de rijen en de kolommen: 56, 28, 14.",
            "two": "Dat geeft `layer2`: één keer twee keer zoveel channels en de helft van de rijen en kolommen. `layer3` doet dat nog een keer: (1, 256, 14, 14).",
            "right": "Twee fases, elk met twee keer zoveel channels en de helft van de rijen en kolommen: 64 → 128 → 256, en 56 → 28 → 14. `layer4` geeft dan (1, 512, 7, 7), en het gemiddelde van elk van zijn 512 maps is wat ResNet-18 aan zijn laatste layer geeft.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_filter():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een filter als plaatje

        Maak `filter_image(w)` af: maak van de weights van één filter, met shape (3, 7, 7), een plaatje met shape (7, 7, 3) met getallen van 0 tot 1, met grijs, 0,5, voor een weight van 0, en 0 of 1 voor de grootste weight van het filter, + of -. De check geeft je functie verzonnen filters, ook een van een andere maat en een waarvan de grootste weight negatief is. Als hij klopt, bewaart de check je functie in je project, als `work/projects/cnn/filters.py`.
        """),
        L.task("Laat `filter_image` van de weights van een filter een plaatje maken met grijs voor 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_filter():
    box_filter = L.code_box("filter-image", lines=7)
    box_filter
    return (box_filter,)


@app.cell(hide_code=True)
def check_filter(box_filter):
    result_filter = L.check("filter-image", box_filter.value)
    result_filter
    return (result_filter,)


@app.cell(hide_code=True)
def hints_filter():
    L.hints(
        "filter-image",
        [
            "Eerst de grootste weight, + of -: `w.abs().max()`. Noem die `m`. Dan gaat `w / m` hooguit van -1 tot 1; hoe laat je dat van 0 tot 1 gaan, met 0 in het midden?",
            "`w / (2 * m)` gaat van -0,5 tot 0,5; tel er 0,5 bij op. Zet dan de kleuren achteraan, met `.permute`: de nieuwe volgorde van de axes, de oude axis 1 eerst.",
        ],
        why="Grijs voor 0 houdt het teken van elke weight in beeld: lichter is een weight die zijn pixel optelt, donkerder een die hem aftrekt. Delen door de grootste weight van het filter zelf laat het patroon van elk filter zien, ook van een zwak filter; daarom kijk je ook naar `w.abs().max()`. Met één schaal voor de hele layer, zijn grootste weight, zouden de 8 filters die niets doen egaal grijs zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_maps():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De feature maps van elke fase

        Maak `feature_maps(model, x)` af: haal de foto's `x` fase voor fase door een ResNet-18 `model`, zoals *Het idee* zegt, en geef een lijst van 5 tensors terug: wat `conv1`, `bn1` en de ReLU geven, vóór de pooling, en wat elk van `layer1`, `layer2`, `layer3` en `layer4` geeft. De ResNet-18 van de course roept zijn ReLU en zijn pooling aan als functies; `torch.relu` en `nn.MaxPool2d(3, stride=2, padding=1)` doen hetzelfde. De check gebruikt een ResNet-18 met willekeurige weights en foto's van 64 bij 64, dus hij heeft geen download nodig, en vergelijkt je maps met die van de course. Als hij klopt, bewaart de check je functie in je project, als `work/projects/cnn/maps.py`.
        """),
        L.task("Laat `feature_maps` de maps van alle vijf de fases geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_maps():
    box_maps = L.code_box("feature-maps", lines=12)
    box_maps
    return (box_maps,)


@app.cell(hide_code=True)
def check_maps(box_maps):
    result_maps = L.check("feature-maps", box_maps.value)
    result_maps
    return (result_maps,)


@app.cell(hide_code=True)
def hints_maps():
    L.hints(
        "feature-maps",
        [
            "Begin zoals de eerste fase van het uitgewerkte voorbeeld: `x = torch.relu(model.bn1(model.conv1(x)))`, en bewaar dat in een lijst, `maps = [x]`. Pool het dan: `x = nn.MaxPool2d(3, stride=2, padding=1)(x)`.",
            "Dan een loop over `[model.layer1, model.layer2, model.layer3, model.layer4]`: elke layer krijgt wat de vorige gaf, `x = layer(x)`, en je bewaart elke `x` met `maps.append(x)`.",
        ],
        why="Een forward hook, uit [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl), kan bewaren wat elk van `layer1` tot en met `layer4` geeft zonder dat je de fases zelf schrijft; de eerste fase heeft dan een hook op `bn1` nodig en een ReLU erna, want de ReLU is hier geen module.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De knop gebruikt je project: je `filter_image` tekent de 64 filters van `conv1`, en je `feature_maps` haalt de foto die je kiest door de fases van ResNet-18. Dan gaat een schuifregelaar de fases langs. Links de foto met het vierkant pixels dat het sterkste getal van die fase ziet; rechts de 8 sterkste maps, de sterkste eerst, met een kader om zijn grootste getal. Hij heeft ResNet-18 nodig, uit het uitgewerkte voorbeeld, en oefening 2 en 3.
    """)
    return


@app.cell(hide_code=True)
def photo_choice_cell():
    photo_choice = mo.ui.radio(options={"de kat": "cat", "de koffie": "coffee", "het fruit": "fruit"}, value="de kat", label="Foto", inline=True)
    look_button = mo.ui.run_button(label="Kijk in ResNet-18 met je functies")
    mo.vstack([photo_choice, look_button])
    return look_button, photo_choice


@app.cell(hide_code=True)
def demo_inside(resnet, look_button, photo_choice, result_filter, result_maps):
    resnet, result_filter, result_maps  # opnieuw uitvoeren als oefening 2 of 3 je module bewaart
    if not look_button.value:
        _out = mo.md("*Druk op de knop om naar binnen te kijken.*")
    else:
        _result = C.demo(photo_choice.value)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `filter_image` en je `feature_maps` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "filters": "*Je `filter_image` in `work/projects/cnn/filters.py` stopt met een fout, of geeft geen plaatje met shape (7, 7, 3). Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "maps": "*Je `feature_maps` in `work/projects/cnn/maps.py` stopt met een fout, of geeft geen vijf maps met de goede shapes. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan opnieuw.*",
            }[_result])
        else:
            _maps = _result["maps"]
            _frames = C.stage_frames(_maps, _result["photo"], {"photo": "wat één getal ziet", "map": "channel {k}",
                                                              "alt": "De foto met een vierkant om de pixels die één getal van {stage} ziet, en de 8 sterkste feature maps van {stage}, in grijs."})

            def _frame(_i):
                return _frames[int(_i)], C.stage_sentence(_maps, int(_i), L.lang)

            _out = mo.vstack([
                mo.Html(C.grid_picture(_result["images"], {"title": "Getekend met je filter_image",
                                                           "alt": "De 64 filters van conv1 in 8 rijen van 8, getekend met je filter_image."})),
                L.explore("stage", "fase", _frame, start=0, values=list(range(5)), role="",
                          task="Schuif van fase 0, conv1, naar fase 4, layer4: vanaf welke fase ziet één getal de hele foto?"),
                mo.md("Fase 0 is `conv1`, met `bn1` en de ReLU; fase 1 tot en met 4 zijn `layer1` tot en met `layer4`."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_eyes():
    mo.md(r"""
    ### Oefening 4. Waar een channel naar zoekt
    """)
    return


@app.cell(hide_code=True)
def choice_eyes_cell():
    choice_eyes = L.choice(
        "eyes",
        question="Op de kat is de sterkste map van `layer3` channel 120, en die is hoog rond beide ogen. Op de koffiefoto staat channel 120 op plek 154 van de 256, naar gemiddelde, en op de fruitfoto op plek 63. Wat kun je zeggen?",
        options={
            "detector": "Channel 120 vindt kattenogen",
            "right": "Channel 120 reageert op iets rond de ogen van de kat; wat, moeten meer foto's uitwijzen",
            "useless": "Channel 120 heeft geen nut voor foto's zonder kat",
        },
        correct="right",
        why={
            "detector": "Drie foto's zijn te weinig om dat te zeggen. Het kan reageren op ronde vormen, op een donkere vlek in een lichte ring, of op de ogen van elk dier. Om het te weten, kijk je naar de foto's, uit heel veel, waarop het channel het sterkst is, zoals Zeiler en Fergus deden.",
            "useless": "De laatste layer weegt de gemiddelden van alle 512 maps van `layer4`, die gemaakt zijn van alle 256 van `layer3`. Een channel dat zwak is op een foto, vertelt de volgende layers ook iets: dat waar het op reageert er niet is.",
            "right": "Eén foto laat zien waar een channel hoog is, niet waar het op reageert. Zeiler en Fergus namen de 9 stukken foto's, uit de 50.000 van de validation set van ImageNet, die een channel het sterkst maken; voor één channel waren dat allemaal koppen van honden.",
        },
    )
    choice_eyes
    return (choice_eyes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De filters en feature maps van [Convolutie: één klein filter schuift over de afbeelding](/course/open/cnn-convolution-idea?lang=nl), nu van een getraind netwerk: de filters van de eerste layer van ResNet-18 zijn randen en vlekken kleur, en zijn diepere maps reageren op grotere stukken van een foto, zoals het receptive field van [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl) groeit. Je haalde een foto die klaargemaakt was zoals in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl) fase voor fase door [ResNet](/course/open/cnn-resnet?lang=nl). Je project `cnn` heeft nu `filters.py` en `maps.py`. In de volgende les, *ImageNet en AlexNet: waarom 2012 alles veranderde*, ga je terug naar de CNN waarvan Zeiler en Fergus de filters bekeken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_filter, result_maps, choice_eyes):
    L.wrap_up(choice_shape, result_filter, result_maps, choice_eyes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
