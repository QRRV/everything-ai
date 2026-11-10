# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-grad-cam. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-grad-cam", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) en [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_leaf():
    quick_leaf = L.choice(
        "quick-leaf",
        question="`a = torch.tensor(2.0, requires_grad=True)`, dan `c = a * 3`, `d = c * c` en `d.backward()`. Wat is `c.grad`?",
        options={
            "twelve": "`tensor(12.)`: de helling van `d` naar `c`, $2c$",
            "right": "`None`: `c` is geen leaf tensor, dus geeft hij zijn helling door en houdt hij er geen",
            "thirtysix": "`tensor(36.)`: de helling van `d` naar `a`",
        },
        correct="right",
        why={
            "twelve": "PyTorch rekent op de terugweg wel 12 uit, en geeft dat door aan `a`; maar `c` is berekend uit `a`, geen leaf tensor, dus houdt hij niets in `.grad`. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "thirtysix": "36 is de helling van $d = 9a^2$ naar `a`, $18a$: dat is `a.grad`, van de leaf tensor. `c` is berekend uit `a`, dus houdt hij geen helling. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "right": "Alleen leaf tensors houden hun hellingen. Vandaag heb je de hellingen nodig van een tensor ertussenin, de maps van een layer, en met één aanroep vraag je PyTorch die te bewaren.",
        },
    )
    quick_leaf
    return (quick_leaf,)


@app.cell(hide_code=True)
def quick_check_hook():
    quick_hook = L.choice(
        "quick-hook",
        question="Een forward hook `keep` aan `resnet.layer4` zet zijn output in een lijst `maps`. Wat is `maps[0].shape` na `resnet(x)` voor één foto van 224 bij 224?",
        options={
            "right": "(1, 512, 7, 7): de 512 feature maps van `layer4`, elk 7 bij 7",
            "pooled": "(1, 512): één getal per map, wat de laatste layer krijgt",
            "logits": "(1, 1000): een logit per klasse",
        },
        correct="right",
        why={
            "pooled": "De laatste layer krijgt één getal per map, maar de hook hangt aan `layer4`, en krijgt wat `layer4` geeft: de maps zelf. Zie [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl).",
            "logits": "De logits zijn wat het hele model geeft. Een hook krijgt de output van de layer waar hij aan hangt, hier `layer4`. Zie [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl).",
            "right": "Een hook krijgt de output van zijn layer: de 512 maps van 7 bij 7 uit [In een CNN kijken](/course/open/cv-inside-cnn?lang=nl). Vandaag maak je er voor een klasse één map van.",
        },
    )
    quick_hook
    return (quick_hook,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een saliency map, uit [Saliency: de pixels die het antwoord veranderen](/course/open/cv-saliency?lang=nl), is scherp maar gespikkeld: hij laat pixel voor pixel zien waar een score gevoelig is, en zijn spikkels liggen over de hele foto verspreid. Zhou en zijn collega's (2016) vonden een grovere map in de laatste feature maps van een CNN, de class activation map, die op de plekken van een klasse ligt. Hun netwerk leerde alleen welke klasse er op elke foto van ImageNet staat, en zag nooit een box. Toch vonden boxes rond de lichte delen van zijn maps het object op 62,9% van de testfoto's van ILSVRC 2014. Een foto telde als gevonden als een van de vijf gokken de goede klasse had, met een box die dicht genoeg bij de echte box van het object lag: een top-5-fout bij het lokaliseren van 37,1%, dicht bij de 34,2% van [AlexNet](/course/open/cv-alexnet?lang=nl), dat wel van de boxes had geleerd. Daarvoor moet een CNN eindigen met het gemiddelde van elke map en één linear layer. Selvaraju en zijn collega's (2017) maakten zo'n map voor elk CNN, uit de gradient: Grad-CAM, in hun woorden een manier om te lokaliseren die de klassen uit elkaar houdt. Vandaag maak je ze allebei voor ResNet-18, op een foto waar twee dingen op staan.
    """)
    return


@app.cell(hide_code=True)
def predict_espresso():
    guess_espresso = L.guess(
        "where-espresso",
        question="Een collage, 224 bij 224: links de middelste helft van de kat, rechts de middelste helft van de koffie. De beste gok van ResNet-18 is Egyptian cat, met 0,40; espresso is zijn vijfde gok, met 0,02. Waar ligt de Grad-CAM van espresso?",
        options={
            "cat": "Op de helft met de kat: ResNet-18 kijkt naar de kat, naar welke klasse je ook vraagt",
            "right": "Op de helft met de koffie: elke klasse krijgt zijn eigen map",
            "spread": "Verspreid over beide helften: met 0,02 ziet ResNet-18 geen espresso",
        },
        reveal="**Op de helft met de koffie.** 87% van de Grad-CAM van espresso ligt op de helft met de koffie, en 95% van die van Egyptian cat op de helft met de kat. Een map komt uit de logit van één klasse, dus maakt het hem niet uit hoe zeker het model van die klasse is: hij laat zien waar het bewijs voor die klasse ligt. De slider in *Het idee* laat de maps van alle 8 beste gokken zien.",
    )
    guess_espresso
    return (guess_espresso,)


@app.cell(hide_code=True)
def text_idea():
    _frames = C.guess_frames({
        "title": "Gok {rank}: {name}, {p}",
        "cat": "kattenhelft {share}%",
        "coffee": "koffiehelft {share}%",
        "alt": "De collage van de kat en de koffie, versluierd waar de Grad-CAM van {name} laag is: {cat}% van de map op de helft met de kat, {coffee}% op de helft met de koffie.",
    }, L.lang)
    _guesses = L.explore("guesses", "gok", lambda _v: _frames[int(_v)], start=1, values=C.RANKS, role="",
                         task="Schuif langs de 8 beste gokken van ResNet-18 voor de collage. Bij welke gok is de map verdeeld over de twee helften?")
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Global average pooling.** Voor een foto van 224 bij 224 geeft `layer4` van ResNet-18 512 feature maps van 7 bij 7. De laatste layer, `fc`, ziet ze niet als maps: `features` in de ResNet-18 van de course eindigt met `x.mean(dim=(2, 3))`, het gemiddelde van elke map, 512 getallen, en `fc`, een linear layer, maakt daar de 1.000 logits van. Het gemiddelde van elke hele map nemen heet **global average pooling**. Het houdt geen plekken: een kat in een hoek en een kat in het midden kunnen dezelfde gemiddelden geven.

        **De weights zetten de plekken terug.** Schrijf $A_k$ voor map $k$ van `layer4`, $A_k(i, j)$ voor zijn getal in rij $i$ en kolom $j$, $w_{ck}$ voor de weight in `fc` van map $k$ naar klasse $c$, en $b_c$ voor de bias van klasse $c$. Dan is de logit van klasse $c$

        $$z_c = \sum_k w_{ck} \cdot \frac{1}{49}\sum_{i,j} A_k(i, j) + b_c = \frac{1}{49}\sum_{i,j} \Big(\sum_k w_{ck} A_k(i, j)\Big) + b_c$$

        Het gemiddelde en de gewogen som mogen van plaats wisselen. Het deel tussen haakjes is een map van 7 bij 7, de **class activation map** van klasse $c$, kortweg CAM:

        $$M_c(i, j) = \sum_k w_{ck} A_k(i, j)$$

        Zijn gemiddelde plus de bias is precies de logit. Elk van zijn 49 getallen, gedeeld door 49, is dus wat die plek van de foto aan de logit bijdraagt: een groot getal, een plek die voor de klasse pleit; een negatief getal, een plek die ertegen pleit. Zhou en zijn collega's laten de bias weg: die is op elke plek hetzelfde.

        **Grad-CAM: de weights uit de gradient.** Niet elk CNN eindigt met global average pooling en één linear layer: LeNet-5, uit [LeNet-5, het CNN dat cheques las](/course/open/cnn-lenet?lang=nl), stopt zijn laatste maps zoals ze zijn in linear layers, getal voor getal. Daar is geen weight per map om te nemen. Selvaraju en zijn collega's (2017) vinden er een uit de hellingen van de logit naar de map, die autograd geeft: de gemiddelde helling over de plekken van de map,

        $$\alpha_{ck} = \frac{1}{Z}\sum_{i,j} \frac{\partial z_c}{\partial A_k(i, j)}$$

        voor een map van $Z$ getallen, 49 bij een map van 7 bij 7. Met die weights tellen ze de maps op, en houden ze alleen het positieve deel, want ze zijn alleen geïnteresseerd in de features die een positieve invloed op de klasse hebben:

        $$G_c = \text{ReLU}\Big(\sum_k \alpha_{ck} A_k\Big)$$

        Dat is **Grad-CAM**. De score is de logit, vóór softmax, net als bij een saliency map. Bij ResNet-18 komen de twee methoden samen, uit `layer4`: elk getal $A_k(i, j)$ bereikt de logit alleen via het gemiddelde, dus is zijn helling $w_{ck}/49$, op elke plek. Dan is $\alpha_{ck} = w_{ck}/49$, en is Grad-CAM het positieve deel van de class activation map, gedeeld door 49. In de woorden van de paper is Grad-CAM een strikte generalisatie van CAM.

        **De hellingen van een tensor ertussenin.** Een forward hook krijgt de maps, zoals in [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl). Maar ze zijn geen leaf tensor: PyTorch berekent ze uit de foto, en geeft hun hellingen door zonder ze te bewaren. Met `A.retain_grad()` vraag je PyTorch om ze toch te bewaren, in `A.grad`, bij de volgende `backward`. En PyTorch moet de foto opnemen: de weights van de ResNet-18 van de course houden geen hellingen bij, dus zonder `x.requires_grad = True` neemt PyTorch onderweg van de foto naar de logit niets op, en hebben de maps helemaal geen hellingen.

        **Grof.** Elk getal van een map van `layer4` staat voor 32 bij 32 pixels van de foto, 224 / 7. Het ziet veel meer: zijn receptive field is 435 bij 435 pixels, meer dan de hele foto. Maar zoals [In een CNN kijken](/course/open/cv-inside-cnn?lang=nl) liet zien, tellen de pixels dicht bij het midden van dat stuk het zwaarst, dus laat een map nog steeds zien waar iets is. De course tekent elk als een blok: een sluier in de kleur van de pagina over de foto, dikker waar de map lager is, en geen waar hij het grootst is. Selvaraju en zijn collega's vergroten de map tot de maat van de foto met bilineaire interpolatie, zoals in [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl), wat de blokken gladstrijkt. Ze nemen de laatste convolutional layer, omdat die volgens hen het beste evenwicht geeft tussen betekenis en ruimtelijk detail.

        De slider laat de collage van de vraag hierboven zien, met de Grad-CAM uit `layer4` van elk van de 8 beste gokken van ResNet-18 ervoor.
        """),
        _guesses,
        mo.md(r"""
        De vier kattenklassen hebben 95% of 96% van hun maps op de helft met de kat. Espresso, met maar 0,02, heeft 87% van zijn map op de helft met de koffie, en coffee mug en cup hebben 79%. Alleen bucket, met 0,01, is verdeeld: 44% op de helft met de kat en 56% op de helft met de koffie. Vóór de ReLU zakt de class activation map van Egyptian cat tot −2,54, op rij 3 en kolom 5, op de helft met de koffie; de ReLU maakt 15 van de 49 getallen van zijn Grad-CAM 0, allemaal vanaf kolom 4, op de helft met de koffie. De vier kattenklassen hebben ook bijna dezelfde map: de klassen uit elkaar houden betekent dat klassen op verschillende plekken maps op verschillende plekken krijgen, niet dat twee klassen die op elkaar lijken, op dezelfde plek, verschillende maps krijgen.
        """),
        L.watch_out("Grad-CAM uit een eerdere layer houdt de klassen minder goed uit elkaar; hier, uit `layer3`, helemaal niet. Met maps van 14 bij 14 hebben de 8 gokken 37% tot 65% van hun maps op de helft met de kat: Egyptian cat 37%, espresso 47%. Selvaraju en zijn collega's zagen hetzelfde: hoe eerder de convolutional layer, hoe slechter de maps van Grad-CAM. Neem de laatste convolutional layer; in de demo probeer je `layer3` met je eigen functie."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **global average pooling** | het gemiddelde van elke hele feature map: één getal per map | `x.mean(dim=(2, 3))` |
        | **class activation map** (CAM) | de maps van de laatste convolutional layer, elk keer de weight van een klasse in de linear layer na global average pooling, opgeteld; zijn gemiddelde plus de bias is de logit | `(A[0] * w.reshape(-1, 1, 1)).sum(dim=0)` |
        | **Grad-CAM** | de maps van een layer, elk keer de gemiddelde helling van de logit van een klasse over die map, opgeteld, en alleen het positieve deel | `torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))` |
        | `.retain_grad()` | bewaart de hellingen van een tensor die geen leaf tensor is, in zijn `.grad`, bij de volgende `backward` | `A.retain_grad()` |
        | $A_k(i, j)$ | het getal in rij $i$ en kolom $j$ van map $k$ | `A[0, k, i, j]` |
        | $w_{ck}$, $b_c$ | de weight in de linear layer van map $k$ naar klasse $c$, en de bias van klasse $c$ | `resnet.fc.weight[c, k]`, `resnet.fc.bias[c]` |
        | $M_c(i, j)$ | de class activation map van klasse $c$, in rij $i$ en kolom $j$ | `cam[i, j]` |
        | $\alpha_{ck}$ | de weight die Grad-CAM map $k$ geeft voor klasse $c$: het gemiddelde van zijn hellingen | `alpha[k]` |
        | $G_c$ | de Grad-CAM van klasse $c$ | `grad_cam` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zie je waar een CNN het bewijs voor een klasse vond:

    1. Maak de foto klaar zoals voor het model, een tensor `x` met shape (1, 3, 224, 224), en zet het opnemen aan: `x.requires_grad = True`.
    2. Hang een forward hook aan de layer waarvan je de maps wilt: de laatste convolutional layer, bij ResNet-18 `layer4`. Haal de foto door het model, en haal de hook eraf. De maps `A` uit de hook hebben shape (1, K, h, w). De score is de logit van de klasse, `model(x)[0, c]`, niet zijn kans.
    3. Bij een CNN dat eindigt met global average pooling en één linear layer, uit zijn laatste maps: de class activation map. De maps van foto 0, elk keer de weight van de klasse in die layer, opgeteld: `(A[0] * w.reshape(-1, 1, 1)).sum(dim=0)`. Zijn gemiddelde plus de bias is de logit.
    4. Bij elke layer van elk CNN: Grad-CAM. Roep `A.retain_grad()` aan, en ga dan terug vanaf de logit, `.backward()`: `A.grad` heeft een helling voor elk getal van elke map. Per map het gemiddelde van zijn hellingen, `alpha = A.grad[0].mean(dim=(1, 2))`; de maps van foto 0, elk keer zijn weight, opgeteld; en het positieve deel, `torch.relu`.
    5. Teken de map: als grijs plaatje, of over de foto, elk getal als een blok van de foto.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel onder de download volgt het recept voor de kat, klaargemaakt zoals in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), met de maps van `layer4`. Stap 3 neemt de weights van klasse 285, Egyptian cat, uit `fc`, en maakt de class activation map, zonder op te nemen: `w.reshape(-1, 1, 1)` geeft de 512 weights de shape (512, 1, 1), zodat broadcasting elke hele map met zijn weight vermenigvuldigt. Hij print het gemiddelde van de map plus de bias naast de logit, dan het grootste getal en waar dat staat: 7 getallen per rij, dus geeft `// 7` de rij en `% 7` de kolom. Stap 4 maakt de Grad-CAM. Hij print de eerste drie weights van Grad-CAM, keer 49, naast de eerste drie weights van `fc`, en of het positieve deel van de class activation map, gedeeld door 49, overal minder dan 0,000001 van de Grad-CAM afwijkt. Als laatste tekent hij de class activation map. De commentaren met `#` zeggen wat elke stap doet. Voer hem uit zoals hij is; verander dan `c` waar het commentaar dat zegt, in 967, espresso, en voer hem opnieuw uit. `resnet` is ResNet-18, die de cel hieronder laadt zodra hij gedownload is.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="ResNet-18 downloaden, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 staat nog niet op je computer. Als je [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl) op deze computer hebt gedaan, staat hij er wel: dan vindt deze les hem. Anders downloadt de knop hem één keer, 47 MB, van Hugging Face. Het uitgewerkte voorbeeld, de maps van de collage en de demo hebben hem nodig; de slider in *Het idee* en de oefeningen niet. Terwijl hij downloadt, wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**ResNet-18 downloaden** · {done} van {total} MB",
        "no_net": "*De download lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus heeft de course het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 47 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
        "here": "*ResNet-18 staat op je computer: 11,7 miljoen parameters, met de weights van torchvision, getraind op ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_cam(resnet):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    # 1. de kat, klaargemaakt zoals in Een beeldclassifier uitproberen, met het opnemen aan
    photo = Image.open("data/photos/chelsea.png")
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    crop = small[16:240, 80:304]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)
    x.requires_grad = True
    # 2. een forward hook bewaart de maps van layer4 terwijl ResNet-18 de logits berekent
    maps = []
    def keep(module, inputs, output):
        maps.append(output)
    handle = resnet.layer4.register_forward_hook(keep)
    logits = resnet(x)
    handle.remove()
    A = maps[0]
    print(A.shape)
    # 3. de class activation map van klasse c: elke map van foto 0 keer zijn weight in fc, opgeteld
    c = 285  # verander: 967
    w = resnet.fc.weight[c]
    print(w.shape)
    with torch.no_grad():
        cam = (A[0] * w.reshape(-1, 1, 1)).sum(dim=0)
    print(cam.shape)
    print(cam.mean() + resnet.fc.bias[c], logits[0, c])
    top = int(cam.argmax())
    print(top // 7, top % 7, cam.max(), cam.min())
    # 4. Grad-CAM: bewaar de hellingen van de maps, ga terug vanaf de logit, weeg elke map met het gemiddelde van zijn hellingen
    A.retain_grad()
    logits[0, c].backward()
    alpha = A.grad[0].mean(dim=(1, 2))
    with torch.no_grad():
        grad_cam = torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))
    print(alpha[:3] * 49, w[:3])
    print((torch.relu(cam) / 49 - grad_cam).abs().max() < 0.000001)
    # 5. teken de class activation map: zwart is zijn kleinste getal, wit zijn grootste
    plt.imshow(cam.numpy(), cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `A` heeft shape (1, 512, 7, 7): de 512 maps van `layer4` voor foto 0, de kat. `w` heeft shape (512,), een weight per map. De class activation map heeft shape (7, 7), en zijn gemiddelde plus de bias, `tensor(19.6482)`, is de logit van Egyptian cat, 19,6482: global average pooling en de linear layer, in de andere volgorde. De logit draagt een `grad_fn`: PyTorch heeft de weg ernaartoe opgenomen, voor stap 4. Het grootste getal van de map, 33,57, staat op rij 5 en kolom 1, het snorhaarkussen links van de neus, met zijn donkere stipjes; de ogen zijn ook licht. Alle 49 getallen zijn positief, het kleinste 11,32: het hele gezicht pleit voor Egyptian cat. In stap 4 zijn de weights van Grad-CAM keer 49 de weights van `fc`, en de laatste regel zegt `tensor(True)`: Grad-CAM is het positieve deel van de class activation map, gedeeld door 49, op afronding na. Met `c = 967`, espresso, is het gemiddelde plus de bias weer de logit, 0,0781, maar de map loopt van −5,42 tot 2,93, met het grootste getal op rij 3 en kolom 0, in de vacht aan de linkerrand: plekken die voor espresso pleiten en plekken die ertegen pleiten, die samen bijna niets opleveren. Zijn Grad-CAM is 0 overal waar de map negatief is.
    """)
    return


@app.cell(hide_code=True)
def text_before_compare():
    mo.md(r"""
    Op de collage de saliency map uit [Saliency: de pixels die het antwoord veranderen](/course/open/cv-saliency?lang=nl) en de Grad-CAM uit `layer4`, voor Egyptian cat en voor espresso. Een saliency map is wit vanaf zijn 1% grootste getallen, een Grad-CAM bij zijn grootste getal.
    """)
    return


@app.cell(hide_code=True)
def compare_cell(resnet):
    _x = C.input_of("collage")
    _maps = {"saliency": {_key: C.course_saliency(resnet, _x, _label) for _key, _label in (("egyptian", C.EGYPTIAN), ("espresso", C.ESPRESSO))},
             "gradcam": {_key: C.course_grad_cam(resnet, resnet.layer4, _x, _label).numpy() for _key, _label in (("egyptian", C.EGYPTIAN), ("espresso", C.ESPRESSO))}}
    mo.Html(C.compare_picture(_maps, {
        "saliency": "saliency map van {label}\n{share}% op de kattenhelft",
        "gradcam": "Grad-CAM van {label}\n{share}% op de kattenhelft",
        "alt": "Vier grijze maps van de collage. Links de saliency maps van Egyptian cat en van espresso: spikkels over beide helften, de lichtste op de kat bij Egyptian cat en op het kopje bij espresso. Rechts hun Grad-CAMs: blokken van 7 bij 7, op de helft met de kat voor Egyptian cat en op de helft met de koffie voor espresso.",
    }))
    return


@app.cell(hide_code=True)
def text_after_compare():
    mo.md(r"""
    De lichtste delen van de twee saliency maps liggen wel uit elkaar: van de 1% grootste getallen, wit getekend, ligt 98% op de helft met de kat bij Egyptian cat, en 89% op de helft met de koffie bij espresso. Maar de rest van een saliency map ligt over de hele foto verspreid: van de hele map ligt 74% op de helft met de kat bij Egyptian cat, en maar 56% op de helft met de koffie bij espresso. De Grad-CAMs liggen bijna helemaal op één helft: 95% van die van Egyptian cat op de helft met de kat, en 87% van die van espresso op de helft met de koffie. Een saliency map is scherp, pixel voor pixel, maar verspreid; een Grad-CAM is grof, 7 bij 7, en ligt op de plekken van zijn klasse.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logit():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Van de map naar de logit
    """)
    return


@app.cell(hide_code=True)
def choice_logit_cell():
    choice_logit = L.choice(
        "logit",
        question="Een klein CNN eindigt met twee maps van 2 bij 2, global average pooling en één linear layer. Map 1 is [[1, 0], [0, 0]] en map 2 is [[0, 0], [0, 2]]. Klasse c heeft weight 3 voor map 1, weight −1 voor map 2, en bias 0,5. Zijn class activation map is [[3, 0], [0, −2]]. Wat is de logit van c?",
        options={
            "right": "0,75: het gemiddelde van de class activation map, 0,25, plus de bias",
            "sum": "1,5: de som van de class activation map, 1, plus de bias",
            "nobias": "0,25: het gemiddelde van de class activation map",
        },
        correct="right",
        why={
            "sum": "Global average pooling neemt het gemiddelde van elke map, niet de som: map 1 geeft 0,25 en map 2 geeft 0,5, en 3 × 0,25 − 1 × 0,5 + 0,5 = 0,75. Het gemiddelde van de class activation map, 0,25, plus de bias is hetzelfde.",
            "nobias": "De linear layer telt ook zijn bias erbij: 3 × 0,25 − 1 × 0,5 + 0,5 = 0,75. In de class activation map zit geen bias, dus is zijn gemiddelde de logit zonder de bias.",
            "right": "De lange weg geeft hetzelfde: global average pooling maakt 0,25 en 0,5, en 3 × 0,25 − 1 × 0,5 + 0,5 = 0,75. De −2 in de map is de plek die tegen c pleit.",
        },
    )
    choice_logit
    return (choice_logit,)


@app.cell(hide_code=True)
def text_exercise_class_map():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De class activation map

        Maak `class_map(maps, weights)` af: de class activation map van de feature maps `maps` van één foto, een tensor met shape (K, H, W), met één weight per map, `weights`, een tensor met shape (K,): elke map keer zijn weight, alles opgeteld, een tensor met shape (H, W). De check geeft je functie de twee maps van oefening 1, en andere verzonnen maps. Als hij klopt, bewaart hij je functie in je project, als `work/projects/cnn/cam.py`.
        """),
        L.task("Laat `class_map` de class activation map van de maps van een foto geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_class_map():
    box_class_map = L.code_box("class-map", lines=5)
    box_class_map
    return (box_class_map,)


@app.cell(hide_code=True)
def check_class_map(box_class_map):
    result_class_map = L.check("class-map", box_class_map.value)
    result_class_map
    return (result_class_map,)


@app.cell(hide_code=True)
def hints_class_map():
    L.hints(
        "class-map",
        [
            "`weights` heeft shape (K,) en `maps` shape (K, H, W). Om elke hele map met zijn weight te vermenigvuldigen, hebben de weights de shape (K, 1, 1) nodig; broadcasting rekt elke weight dan uit over zijn map.",
            "`weights.reshape(-1, 1, 1)` geeft die shape. Daarna telt `.sum(dim=0)` op langs axis 0, de maps.",
        ],
        why="Broadcasting zet shapes van rechts tegen elkaar: (K,) tegen (K, H, W) komt bij W uit, de kolommen. Met (K, 1, 1) komen de K weights bij de K maps, en wordt elk uitgerekt over zijn H bij W getallen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_grad_cam():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Grad-CAM

        Maak `grad_cam(model, layer, x, label)` af: de Grad-CAM van de foto `x`, met shape (1, channels, H, W), voor de klasse `label`, uit de maps die `layer` geeft, als een tensor in de shape van één map. Het vak begint met `keep`, de forward hook uit [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl). De check gebruikt een ResNet-18 met willekeurige weights, met `layer4` en met `layer3`, en een klein CNN met willekeurige weights dat niet eindigt met global average pooling: op een grijs plaatje van 28 bij 28 gaan de maps van 14 bij 14 van zijn `conv2` door een ReLU en max pooling naar 7 bij 7, en dan, getal voor getal, een linear layer in. Als hij klopt, bewaart hij je functie in je project, als `work/projects/cnn/gradcam.py`.
        """),
        L.task("Laat `grad_cam` de Grad-CAM van een foto voor een klasse geven, uit elke layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grad_cam():
    box_grad_cam = L.code_box("grad-cam", lines=12)
    box_grad_cam
    return (box_grad_cam,)


@app.cell(hide_code=True)
def check_grad_cam(box_grad_cam):
    result_grad_cam = L.check("grad-cam", box_grad_cam.value)
    result_grad_cam
    return (result_grad_cam,)


@app.cell(hide_code=True)
def hints_grad_cam():
    L.hints(
        "grad-cam",
        [
            "Hang `keep` aan `layer` en bewaar de handle; zet het opnemen aan voor de foto; neem de logit van `label`, `model(x)[0, label]`; en haal de hook eraf. `maps[0]` bevat nu de maps, `A`, met shape (1, K, h, w).",
            "`A.retain_grad()` vóór `score.backward()`: dan heeft `A.grad` de hellingen, in de shape van `A`. De weight van map k is het gemiddelde van `A.grad[0, k]`, allemaal tegelijk met `.mean(dim=(1, 2))`. Weeg `A[0]` ermee zoals in oefening 2, en houd het positieve deel met `torch.relu`.",
        ],
        why="De hellingen van de logit naar een map zeggen hoeveel elk van zijn getallen bijdraagt; hun gemiddelde is één weight per map, zoals de weight van de linear layer bij ResNet-18. `A` is berekend uit de foto, dus geen leaf tensor: zonder `retain_grad` geeft PyTorch zijn hellingen door en houdt er geen, en blijft `A.grad` `None`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De knop gebruikt je project: je `grad_cam` maakt de maps van de foto die je kiest, uit de layer die je kiest, voor de drie beste gokken van ResNet-18, en de course tekent elk over de foto. Uit `layer4` maakt de course ook de class activation map van de beste gok, met je `class_map` en de weights van `fc`, en vergelijkt zijn positieve deel, gedeeld door 49, met je Grad-CAM. Hij heeft ResNet-18 nodig, uit het uitgewerkte voorbeeld, en oefening 2 en 3.
    """)
    return


@app.cell(hide_code=True)
def photo_choice_cell():
    photo_choice = mo.ui.radio(options={"de kat en de koffie": "collage", "de kat": "cat", "de koffie": "coffee", "de tekening van fruit": "fruit"},
                               value="de kat en de koffie", label="Foto", inline=True)
    layer_choice = mo.ui.radio(options={"layer4, maps van 7 bij 7": "layer4", "layer3, maps van 14 bij 14": "layer3"},
                               value="layer4, maps van 7 bij 7", label="Layer", inline=True)
    look_button = mo.ui.run_button(label="Maak de maps met je functies")
    mo.vstack([photo_choice, layer_choice, look_button])
    return layer_choice, look_button, photo_choice


@app.cell(hide_code=True)
def demo_grad_cam(resnet, look_button, photo_choice, layer_choice, result_class_map, result_grad_cam):
    resnet, result_class_map, result_grad_cam  # opnieuw uitvoeren als oefening 2 of 3 je module bewaart
    if not look_button.value:
        _out = mo.md("*Druk op de knop om de maps te maken.*")
    else:
        _result = C.demo(photo_choice.value, layer_choice.value)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3: dan staan je `class_map` en je `grad_cam` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "class_map": "*Je `class_map` in `work/projects/cnn/cam.py` stopt met een fout, of geeft geen map van 7 bij 7. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "grad_cam": "*Je `grad_cam` in `work/projects/cnn/gradcam.py` stopt met een fout, of geeft geen map in de shape van de maps van de layer, zonder negatieve getallen en niet overal 0. Heb je zijn bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan opnieuw.*",
            }[_result])
        else:
            _out = mo.vstack([
                mo.Html(C.demo_picture(_result, {"title": "{rank}. {name}, {p}", "cat": "kattenhelft {share}%", "coffee": "koffiehelft {share}%",
                                                 "alt": "De foto drie keer, elk versluierd waar je Grad-CAM van een van de drie beste gokken van ResNet-18 laag is."}, L.lang)),
                mo.md(C.demo_sentence(_result, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_evidence():
    mo.md(r"""
    ### Oefening 4. Zeker, of daar?
    """)
    return


@app.cell(hide_code=True)
def choice_evidence_cell():
    choice_evidence = L.choice(
        "evidence",
        question="Op de collage geeft ResNet-18 espresso maar 0,02, zijn vijfde gok. Toch ligt 87% van de Grad-CAM van espresso op de helft met de koffie, ongeveer even duidelijk als 95% van die van Egyptian cat, met 0,40, op de helft met de kat ligt. Wat zegt dat?",
        options={
            "noise": "De map van een klasse met een lage kans is ruis",
            "right": "Een Grad-CAM laat zien waar het bewijs voor een klasse in de foto ligt, niet hoe zeker het model van die klasse is",
            "wrong": "ResNet-18 ziet meer espresso dan hij zegt: zijn kans klopt niet",
        },
        correct="right",
        why={
            "noise": "Ruis zou niet voor 87% op de koffie landen en de kat links laten liggen. De map komt alleen uit de logit van espresso: de helft met de koffie verhoogt die, de helft met de kat niet.",
            "wrong": "De kans klopt voor wat ResNet-18 uitrekent: softmax weegt de logit van espresso af tegen die van alle 1.000 klassen. De logit van Egyptian cat, 11,20, is 3,02 groter dan die van espresso, 8,18, en softmax maakt daar ongeveer 20 keer zoveel van: 0,40 tegen 0,02. De map vergelijkt geen klassen; hij laat zien waar de logit van één klasse vandaan komt.",
            "right": "De map komt uit de logit van één klasse, vóór softmax. Waar het bewijs voor die klasse ligt, laat de map zien; welke klasse wint, beslist softmax, over alle 1.000. Een zeker antwoord kan dus een map op de verkeerde plek hebben, en een onwaarschijnlijke klasse een map op de goede.",
        },
    )
    choice_evidence
    return (choice_evidence,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Aan het eind van ResNet-18 neemt global average pooling het gemiddelde van elke map, zoals je in [In een CNN kijken](/course/open/cv-inside-cnn?lang=nl) zag; daardoor kunnen de weights van zijn laatste layer de plekken terugzetten: de class activation map. Grad-CAM vindt die weights voor elk CNN uit de gradient, met de forward hooks uit [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl) en de autograd uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl), plus `retain_grad` voor een tensor ertussenin. Waar de saliency map uit [Saliency: de pixels die het antwoord veranderen](/course/open/cv-saliency?lang=nl) scherp is maar over de hele foto verspreid ligt, is een Grad-CAM grof maar ligt hij op de plekken van zijn klasse. Je project `cnn` heeft nu `cam.py` en `gradcam.py`. Grad-CAM vindt waar een klasse is zonder ooit een box te hebben gezien; de volgende les, *Anchors: de boxes waar een detector mee begint*, begint aan de detectors die de boxes zelf leren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_logit, result_class_map, result_grad_cam, choice_evidence):
    L.wrap_up(choice_logit, result_class_map, result_grad_cam, choice_evidence)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
