# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-1x1-convolution. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-1x1-convolution", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="De eerste layer van je ColorCNN was `nn.Conv2d(3, 8, 3, padding=1)`. Hoeveel weights heeft één van zijn filters?",
        options={
            "nine": "9: een stuk van 3 bij 3",
            "right": "27: een stuk van 3 bij 3 van elk van de 3 kleuren",
            "all": "216: 8 filters van 27",
        },
        correct="right",
        why={
            "nine": "9 is een stuk van 3 bij 3 van één channel. Een filter van een kleurenfoto kijkt naar alle 3 de channels tegelijk: een stuk van 3 bij 3 van het rood, van het groen en van het blauw. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "all": "216 = 8 × 27 zijn de weights van alle 8 de filters samen. Eén filter heeft er 3 × 3 × 3 = 27, en een bias. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "right": "3 channels × 3 × 3 = 27 weights, en een bias. De filters van vandaag zijn de kleinste die er zijn, 1 bij 1: op een kleurenfoto 3 weights, één voor elk channel.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Een batch van één kleurenfoto van 32 bij 32, shape (1, 3, 32, 32), gaat door `nn.Conv2d(3, 8, 3, padding=1)`. Welke shape komt eruit?",
        options={
            "same": "(1, 3, 32, 32)",
            "no_pad": "(1, 8, 30, 30)",
            "right": "(1, 8, 32, 32)",
        },
        correct="right",
        why={
            "same": "Een convolutie geeft één channel voor elk filter: 8 filters, 8 channels, hoeveel er ook in gingen. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "no_pad": "Dat is zonder padding: een filter van 3 bij 3 past 30 keer in de breedte van 32. `padding=1` zet er een rand van één pixel omheen, zodat hij 32 keer past. Zie [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl).",
            "right": "8 filters geven 8 channels, en de padding houdt 32 bij 32. Een filter van 1 bij 1 houdt de grootte zonder padding: hij past op elke pixel.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) kreeg de residual connection een convolutie van 1 bij 1 waar de channels van $x$ en $f(x)$ niet pasten. Een filter van 1 bij 1 klinkt nutteloos: hij ziet één pixel, dus hij kan geen rand of vorm vinden. Maar een pixel diep in een CNN is niet één getal. Hij heeft een getal in elk channel, 64 of 512 stuks, en een filter van 1 bij 1 mengt die tot nieuwe. In 2013 zetten Min Lin, Qiang Chen en Shuicheng Yan zulke convoluties na de grotere in hun netwerk, *Network In Network*. In 2014 gebruikte GoogLeNet ze om de channels minder te maken vóór zijn dure convoluties van 3 bij 3 en 5 bij 5, zodat het met hetzelfde werk dieper en breder kon worden, en won het de wedstrijd van ImageNet. In 2015 bouwden ResNet-50 en de diepere ResNets elk block zo: een bottleneck. Vandaag meng je de kleuren van een foto met een 1×1-convolutie, tel je wat een bottleneck bespaart, bouw je er een, en zet je hem in een CNN op CIFAR-10.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_bottleneck():
    guess_bottleneck = L.guess(
        "bottleneck-params",
        question="In ResNet-50 krijgen en geven veel blocks 256 channels. Eén convolutie van 3 bij 3 van 256 channels naar 256 zou 590.080 parameters hebben. ResNet-50 gebruikt in plaats daarvan drie convoluties: 1 bij 1 van 256 channels naar 64, 3 bij 3 van 64 naar 64, en 1 bij 1 van 64 terug naar 256. Hoeveel parameters hebben die drie?",
        options={
            "more": "Meer: drie convoluties in plaats van één",
            "half": "Ongeveer de helft",
            "right": "Ongeveer een achtste",
        },
        reveal="**Ongeveer een achtste: 70.016.** De 1×1-convolutie van 256 naar 64 heeft er 64 × (256 + 1) = 16.448; de 3×3 van 64 naar 64 heeft er 64 × (64 × 9 + 1) = 36.928; en de 1×1 van 64 terug naar 256 heeft er 256 × (64 + 1) = 16.640. Samen 8,4 keer minder dan 590.080. Voor elke pixel doen ze 69.632 vermenigvuldigingen, tegen 589.824; en ongeveer evenveel als de 73.728 van het block van twee 3×3-convoluties van 64 channels in de kleinere ResNets, zoals ResNet-34: hetzelfde werk, op maps met 4 keer zoveel channels. Geteld met een bias voor elk filter, zoals `nn.Conv2d` ze maakt; ResNet laat de biases weg, omdat er na elke convolutie batch normalization komt.",
    )
    guess_bottleneck
    return (guess_bottleneck,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén pixel, al zijn channels.** Een 1×1-convolutie heeft filters van 1 bij 1. Op elke pixel neemt een filter het getal van die pixel in elk channel dat binnenkomt, vermenigvuldigt elk met zijn eigen weight, telt ze op met zijn bias, en dat is het getal van zijn channel op die pixel. Met $c_{in}$ channels erin en $c_{out}$ filters is dat een linear layer van $c_{in}$ getallen naar $c_{out}$, op elke pixel dezelfde. De rijen en kolommen blijven zoals ze zijn; alleen het aantal channels verandert.

        Het plaatje: het landschap uit [Een afbeelding is een rooster van getallen](/course/open/data-images-as-arrays?lang=nl), en wat twee 1×1-convoluties van 3 channels naar 1 ervan maken, getekend met het laagste getal zwart en het hoogste wit. De weights 0,299, 0,587 en 0,114 voor rood, groen en blauw geven een grijze foto: ze zeggen hoe licht elke kleur er voor ons uitziet, groen het meest en blauw het minst, en ze zijn een standaard voor televisie. De weights 1, 0 en −1 geven rood min blauw: licht waar een pixel meer rood dan blauw heeft, zoals in de gele zon, en donker waar hij meer blauw heeft, zoals in de lucht.
        """),
        mo.Html(C.mix_picture({"photo": "foto, 3 channels", "gray": "0,299; 0,587; 0,114", "red_blue": "1; 0; −1",
                               "alt": "Drie plaatjes van hetzelfde landschap van 24 bij 40 pixels: in kleur; in grijs, gemaakt met de weights 0,299, 0,587 en 0,114; en rood min blauw, gemaakt met de weights 1, 0 en −1, met de zon licht en de lucht donker."})),
        mo.md(r"""
        **Zijn parameters.** Een 1×1-convolutie van $c_{in}$ channels naar $c_{out}$ heeft $c_{out} \times (c_{in} + 1)$ parameters: per filter een weight voor elk channel dat binnenkomt, en een bias. Een 3×3-convolutie heeft er $c_{out} \times (9\,c_{in} + 1)$: negen keer zoveel weights. Allebei groeien ze met $c_{in} \times c_{out}$, dus een convolutie tussen veel channels is duur, en een 3×3-convolutie het meest.

        **Een bottleneck.** Zet de dure 3×3-convolutie dus op een smalle plek. Een bottleneck rond $c$ channels heeft drie convoluties, elk met een ReLU erna: een 1×1-convolutie van $c$ channels naar $m$, minder; de 3×3-convolutie van $m$ naar $m$; en een 1×1-convolutie van $m$ terug naar $c$. Wat eruit komt, heeft evenveel channels als wat erin ging, dus het past waar één 3×3-convolutie van $c$ naar $c$ zat.
        """),
        L.watch_out("Een 1×1-convolutie kijkt nooit naar een buur: wat hij op een pixel geeft, komt alleen uit die pixel. Een stapel van alleen 1×1-convoluties vindt geen randen, geen vormen, niets wat over twee pixels gaat. Daarom houdt een bottleneck een 3×3-convolutie in zijn midden: de 1×1-convoluties eromheen mengen alleen channels."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **1×1-convolutie** | een convolutie met filters van 1 bij 1: op elke pixel mengt hij de getallen van de channels van die pixel tot nieuwe channels, als een linear layer op elke pixel | `nn.Conv2d(c_in, c_out, 1)` |
        | **bottleneck** | een smal midden: een 1×1-convolutie naar minder channels, de dure convolutie op die paar, en een 1×1-convolutie terug naar evenveel als eerst | |

        **Het netwerk van de demo.** De demo traint een CNN zoals je ColorCNN, maar breder en met één stap meer: 32 en 64 filters van 3 bij 3, elk met max pooling van 2 bij 2, dan een block op de 64 maps van 8 bij 8, weer pooling, en één linear layer naar de 10 logits. Het block is één 3×3-convolutie van 64 naar 64 met een ReLU, 36.928 parameters, of je `Bottleneck` uit oefening 2, met 32 of 16 channels in het midden: 13.440 of 4.448 parameters. Het hele netwerk heeft er dan 66.570, 43.082 of 34.090: met 16 in het midden ongeveer de helft.

        Het plaatje: hoeveel foto's het netwerk goed leest na 20 epochs op de eerste 6.000 trainingsfoto's van CIFAR-10, van die trainingsfoto's, gevuld, en van de eerste 1.000 testfoto's, open, een stip voor elk van 5 seeds, met de code van de course. Met één 3×3-convolutie leest het 5.999 tot 6.000 van zijn trainingsfoto's goed, en 532 tot 572 testfoto's; zijn validation loss is het laagst na epoch 4 tot 6, en stijgt daarna weer. Met 32 in het midden: 5.333 tot 5.673 trainingsfoto's en 527 tot 553 testfoto's. Met 16 in het midden: 5.077 tot 5.510 trainingsfoto's en 521 tot 550 testfoto's, met de laagste validation loss na epoch 7 tot 12. Oefening 3 vraagt wat de getallen zeggen.
        """),
        mo.Html(C.runs_picture({"title": "Goed gelezen na 20 epochs", "x": "goed gelezen (%)", "plain": "één 3×3-convolutie",
                                "b32": "bottleneck, 32 in het midden", "b16": "bottleneck, 16 in het midden", "mine": "je run",
                                "train": "training", "test": "test",
                                "alt": "Drie rijen stippen tussen 40 en 100 procent goed gelezen, na 20 epochs. Met één 3×3-convolutie in het midden: trainingsfoto's ongeveer 100%, testfoto's ongeveer 55%. Een bottleneck met 32 in het midden: training ongeveer 92%, test ongeveer 54%. Een bottleneck met 16 in het midden: training ongeveer 89%, test ongeveer 54%."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De channels van een map mengen met een 1×1-convolutie:

    1. Maak hem met `nn.Conv2d(c_in, c_out, 1)`: een kernel size van 1, geen padding nodig. De rijen en kolommen blijven.
    2. Zijn `weight` heeft shape `(c_out, c_in, 1, 1)`: `weight[k, :, 0, 0]` bevat de weights van output channel `k`, één voor elk channel dat binnenkomt.
    3. Hij heeft $c_{out} \times (c_{in} + 1)$ parameters, tegen $c_{out} \times (9\,c_{in} + 1)$ voor een 3×3-convolutie.
    4. Voor een bottleneck rond $c$ channels met $m$ in het midden: een 1×1-convolutie van $c$ naar $m$, een 3×3 van $m$ naar $m$ met padding 1, en een 1×1 van $m$ terug naar $c$, elk met een ReLU erna.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest het landschap als een batch van één foto, maakt een 1×1-convolutie van 3 channels naar 1, geeft hem de weights van een grijswaarde, en print de shape van zijn weights, de shapes van de foto en van wat eruit komt, de drie kleuren van een pixel in de zon, rij 5 en kolom 32, en wat de convolutie van die pixel maakt. Daarna telt hij de parameters van de convolutie. Voer de cel uit zoals hij is, verander hem dan waar het commentaar het zegt, en voer hem opnieuw uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_mix():
    import matplotlib.pyplot as plt
    import torch
    from torch import nn

    img = plt.imread("lessons/data-images-as-arrays/landscape.png")
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    # 1. een 1×1-convolutie van 3 channels naar 1
    mix = nn.Conv2d(3, 1, 1)
    # 2. zijn weights: één voor rood, één voor groen en één voor blauw
    print(mix.weight.shape)
    with torch.no_grad():
        mix.weight[0, :, 0, 0] = torch.tensor([0.299, 0.587, 0.114])  # verander: 1.0, 0.0, -1.0
        mix.bias[0] = 0.0
        out = mix(x)
    print(x.shape, out.shape)
    print(x[0, :, 5, 32])
    print(out[0, 0, 5, 32])
    # 3. zijn parameters: 3 weights en een bias
    print(sum(q.numel() for q in mix.parameters()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, print hij eerst `torch.Size([1, 3, 1, 1])`: één output channel, met een weight voor elk van de 3 channels die binnenkomen, zoals stap 2 van het recept zegt. Daarna `torch.Size([1, 3, 24, 40]) torch.Size([1, 1, 24, 40])`: 3 channels erin, 1 eruit, en dezelfde 24 bij 40 pixels. De pixel in de zon is `tensor([0.9804, 0.8431, 0.2353])`: rood 0,9804, groen 0,8431 en blauw 0,2353. De convolutie maakt er `tensor(0.8149)` van: 0,299 × 0,9804 + 0,587 × 0,8431 + 0,114 × 0,2353 = 0,8149, elk channel keer zijn weight, opgeteld. Met `1.0, 0.0, -1.0` print hij `tensor(0.7451)`: rood min blauw, hoog in de gele zon. Als laatste `4`: 3 weights en een bias, die de convolutie op elk van de 24 × 40 = 960 pixels gebruikt. Dat is het plaatje onder *Het idee*.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif het aantal channels in het midden van een bottleneck rond 64 channels. De oranje staaf bestaat uit de drie convoluties van de bottleneck, in duizenden parameters, achter elkaar; de grijze staaf is één 3×3-convolutie van 64 naar 64, met een gestreepte lijn waar hij ophoudt.
    """)
    return


@app.cell(hide_code=True)
def play_middle():
    _labels = {"title": "Een bottleneck rond 64 channels", "x": "parameters (duizenden)", "plain": "één 3×3", "bottleneck": "bottleneck",
               "alt": "Twee staven parameters, in duizenden: één 3×3-convolutie van 64 naar 64 channels, 36.928, in grijs, met een gestreepte lijn waar hij ophoudt; en in oranje de drie convoluties van een bottleneck met het gekozen aantal channels in het midden, achter elkaar."}
    explore_middle = L.explore("middle", "channels in het midden", lambda _v: (C.bottleneck_picture(int(_v), _labels, L.lang), C.bottleneck_sentence(int(_v), L.lang)),
                               start=16, values=list(range(1, 65)), role="hyper",
                               task="Vanaf hoeveel channels in het midden heeft de bottleneck meer parameters dan die ene 3×3-convolutie?")
    explore_middle
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Vanaf 58: met 57 in het midden heeft hij 36.715 parameters, en met 58 heeft hij er 37.880. De 3×3-convolutie in het midden groeit met $m \times m$, dus de besparing groeit snel als $m$ kleiner wordt: met 32 in het midden zijn het er 13.440, 36,4% van de 36.928; met 16 zijn het er 4.448, 12,0%. ResNet-50 houdt een kwart van de channels in het midden: 64 van de 256.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_one_by_one():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén pixel of meer?

    Zes dingen die je een convolutie met een foto zou kunnen laten doen. Welke kan een 1×1-convolutie, met de goede weights, en voor welke is een groter filter nodig? Zet elk ding in zijn groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_one_by_one():
    sort_one_by_one = L.sort("one-by-one")
    sort_one_by_one
    return (sort_one_by_one,)


@app.cell(hide_code=True)
def check_one_by_one(sort_one_by_one):
    result_one_by_one = L.check("one-by-one", sort_one_by_one)
    result_one_by_one
    return (result_one_by_one,)


@app.cell(hide_code=True)
def hints_one_by_one():
    L.hints(
        "one-by-one",
        [
            "Vraag bij elk ding: heeft het nieuwe getal op een pixel iets anders nodig dan de oude getallen van diezelfde pixel?",
            "Drie ervan kan hij wel, de andere drie niet. Een 1×1-convolutie kan uit de channels van één pixel zoveel nieuwe channels maken als je wilt, met elke weights die je wilt; hij kan niet één pixel verder kijken.",
        ],
        why="**Een 1×1-convolutie kan het:** een grijze foto, met de weights 0,299, 0,587 en 0,114; minder channels, met 16 filters van elk 64 weights; en verwisselde kleuren, met een weight van 1 van het oude blauw naar het nieuwe rood en van het oude rood naar het nieuwe blauw, 1 van groen naar groen, en overal anders 0. **Daar is een groter filter voor nodig:** een rand, vervagen en verschuiven hebben allemaal de buren van een pixel nodig.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bottleneck():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een bottleneck

        Maak `Bottleneck` af, een block dat past waar een 3×3-convolutie van `channels` naar `channels` zat. Maak `reduce`, een 1×1-convolutie van `channels` naar `middle`, en `expand`, een 1×1-convolutie van `middle` terug naar `channels`; `conv`, de 3×3-convolutie ertussen, staat al in het vak. Stuur in `forward` `x` door `reduce`, `conv` en `expand`, in die volgorde, elk met een ReLU erna, en geef terug wat eruit komt. De check bouwt je block, geeft het verzonnen getallen, en vergelijkt wat eruit komt, en hoeveel parameters het heeft, met dat van de course. Als het klopt, bewaart hij je class in je project, als `work/projects/cnn/bottleneck.py`.
        """),
        L.task("Maak `reduce` en `expand`, en maak `forward` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bottleneck():
    box_bottleneck = L.code_box("bottleneck", lines=18)
    box_bottleneck
    return (box_bottleneck,)


@app.cell(hide_code=True)
def check_bottleneck(box_bottleneck):
    result_bottleneck = L.check("bottleneck", box_bottleneck.value)
    result_bottleneck
    return (result_bottleneck,)


@app.cell(hide_code=True)
def hints_bottleneck():
    L.hints(
        "bottleneck",
        [
            "Een 1×1-convolutie is `nn.Conv2d` met een kernel size van 1. Hoeveel channels gaan er in `reduce`, en hoeveel komen eruit? En bij `expand`?",
            "`self.reduce = nn.Conv2d(channels, middle, 1)`. In `forward` is `x = torch.relu(self.reduce(x))` de eerste stap; `conv` en `expand` gaan net zo, en de laatste regel geeft terug wat `expand` en zijn ReLU maken.",
        ],
        why="`reduce` is `nn.Conv2d(channels, middle, 1)` en `expand` is `nn.Conv2d(middle, channels, 1)`, en `forward` stuurt `x` door `reduce`, `conv` en `expand`, elk gevolgd door `torch.relu`. In ResNet-50 heeft elk van de drie convoluties batch normalization erna, en wordt de input van het block opgeteld bij wat `expand` geeft vóór de laatste ReLU, zoals in [ResNet](/course/open/cnn-resnet?lang=nl). Hier staat het block op de plek van één 3×3-convolutie, zonder die optelling.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint het netwerk van de demo op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test het op de eerste 1.000 testfoto's. Kies zijn block: één 3×3-convolutie van 64 naar 64, of je `Bottleneck` met 32 of met 16 channels in het midden. Waar een module van jou ontbreekt of kapot is, springt die van de course in. Hij heeft CIFAR-10 nodig, en voor een run met een bottleneck ook oefening 2, en duurt een paar minuten. Tijdens het trainen toont een grafiek de loss op de trainingsfoto's elke 10 batches, en gestreept op de testfoto's na elke epoch. Naast of onder de grafiek staat wat het netwerk op dat moment gokt bij één testfoto van elke soort, en daaronder de feature maps die zijn midden van een auto maakt; allebei worden ze opnieuw getekend terwijl het traint. In het plaatje daarna is de grijze lijn de validation loss van de run van de course met één 3×3-convolutie en seed 0; als je zelf zo traint, zijn run met 16 in het midden. Daarna kun je terugspoelen naar een paar epochs van je run, met beide plaatjes zoals ze toen waren, en zie je je run naast die van de course.
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
    variant_choice = mo.ui.radio(options={"één 3×3-convolutie": "plain", "bottleneck, 32 in het midden": "b32", "bottleneck, 16 in het midden": "b16"},
                                 value="bottleneck, 16 in het midden", label="In het midden")
    train_button = mo.ui.run_button(label="Train het netwerk")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_bottleneck(cifar, train_button, variant_choice, result_bottleneck):
    cifar, result_bottleneck  # opnieuw uitvoeren als CIFAR-10 binnen is, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het netwerk te trainen.*")
    else:
        _variant = variant_choice.value
        _names = {"plain": "één 3×3-convolutie", "b32": "bottleneck, 32 in het midden", "b16": "bottleneck, 16 in het midden"}
        _with = {"plain": "één 3×3-convolutie in het midden", "b32": "een bottleneck van 64 naar 32 channels en terug", "b16": "een bottleneck van 64 naar 16 channels en terug"}
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht
        _layer = "Het midden van het netwerk op een testfoto van een auto: layer {name}"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je netwerk op CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "De loss van het netwerk tot nu toe: op de trainingsfoto's elke 10 batches, en gestreept op de 1.000 testfoto's na elke epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "trainingsfoto's, elke 10 batches"), ("loss", True, "testfoto's, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-bottleneck-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**{_names[_variant].capitalize()}** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**{_names[_variant].capitalize()}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant].capitalize()}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan staat je `Bottleneck` in je project, en gebruikt deze cel hem. Of train met één 3×3-convolutie, daar is geen oefening voor nodig.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `Bottleneck` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "bottleneck": "*Je `Bottleneck` in `work/projects/cnn/bottleneck.py` stopt met een fout, of geeft geen 64 maps van 8 bij 8 terug. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding onder *Waarom* zegt."
            _other = "b16" if _variant == "plain" else "plain"
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "één 3×3-convolutie", "b16": "16 in het midden",
                       "alt": "De loss van het netwerk na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course met seed 0 " + ("met een bottleneck van 64 naar 16 channels en terug." if _other == "b16" else "met één 3×3-convolutie, die daalt tot epoch 6 en daarna stijgt.")}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _layer, L.lang, other=_other),
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
                mo.md(f"""Met {_with[_variant]} leest het netwerk **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters, waarvan {C.grouped(_result['block_params'], L.lang)} in zijn block.{_turn} Over 5 seeds, met de code van de course: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's, met de laagste validation loss na epoch {_be[0]} tot {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Goed gelezen na 20 epochs", "x": "goed gelezen (%)", "plain": "één 3×3-convolutie",
                                        "b32": "bottleneck, 32 in het midden", "b16": "bottleneck, 16 in het midden", "mine": "je run: " + _names[_variant],
                                        "train": "training", "test": "test",
                                        "alt": "Rijen stippen tussen 40 en 100 procent goed gelezen: bovenaan je run, met je trainingsfoto's en testfoto's; daaronder de drie runs van de course over 5 seeds, met één 3×3-convolutie en met een bottleneck van 64 naar 32 en naar 16 channels en terug."}, L.lang, mine={"variant": _variant, "train": _result["train"], "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Oefening 3. Wat de bottleneck hier doet
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "bottleneck-runs",
        question="Over 5 seeds, na 20 epochs. Met één 3×3-convolutie in het midden heeft het netwerk 66.570 parameters, en leest het 5.999 tot 6.000 van zijn trainingsfoto's goed en 532 tot 572 testfoto's. Met een bottleneck van 64 naar 16 channels en terug heeft het er 34.090, en leest het 5.077 tot 5.510 trainingsfoto's goed en 521 tot 550 testfoto's. Wat zeggen de getallen?",
        options={
            "worse": "De bottleneck is slechter: hij leest minder foto's goed, van allebei de soorten",
            "better": "De bottleneck is beter: hij leest meer testfoto's goed",
            "right": "Met ongeveer de helft van de parameters leest de bottleneck bijna evenveel testfoto's goed, en leert hij zijn trainingsfoto's minder uit zijn hoofd",
        },
        correct="right",
        why={
            "worse": "Minder trainingsfoto's, ja, maar het netwerk met één 3×3-convolutie leert zijn 6.000 uit zijn hoofd, 5.999 tot 6.000 goed, en zijn validation loss eindigt tussen 2,81 en 3,12. Op de testfoto's overlappen de bereiken: 521 tot 550 tegen 532 tot 572.",
            "better": "Niet meer: gemiddeld 537, tegen 549, en de bereiken overlappen. Zijn laagste validation loss is iets lager, 1,30 tot 1,35 tegen 1,37 tot 1,42, en aan het eind is hij 1,56 tot 1,86 tegen 2,81 tot 3,12: hij is minder zeker van zijn foute gokken, maar hij krijgt er niet meer goed.",
            "right": "Zijn block heeft 4.448 parameters in plaats van 36.928, 12,0%, en doet 278.528 vermenigvuldigingen voor een foto in plaats van 2.359.296. Zijn training loss eindigt tussen 0,30 en 0,49, terwijl die van het netwerk met één 3×3-convolutie eindigt tussen 0,006 en 0,018: met minder parameters heeft hij minder ruimte om elke foto uit zijn hoofd te leren. Op 6.000 foto's van 32 bij 32 doet hij het bijna even goed met veel minder. GoogLeNet stak zulke besparingen in een dieper en breder netwerk; ResNet-50 in vier keer zoveel channels tussen zijn blocks, en ResNet-101 en ResNet-152 in meer blocks.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een 1×1-convolutie is de linear layer uit [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl), gebruikt op elke pixel, met de shapes uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl): hij zat in de residual connection van [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl), en in een bottleneck laat hij de 3×3-convoluties uit [VGG: dieper met kleine filters](/course/open/cnn-vgg-blocks?lang=nl) op minder channels werken. Je telde zijn parameters zoals in [Project: je eerste CNN](/course/open/cnn-pytorch?lang=nl), en trainde hem met de Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) op de foto's uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl). Je project `cnn` heeft nu `bottleneck.py`. In de volgende les, *Inception: filters van verschillende groottes naast elkaar*, maken 1×1-convoluties de channels minder vóór filters van 3 bij 3 en 5 bij 5 die naast elkaar naar dezelfde maps kijken, zoals in GoogLeNet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_one_by_one, result_bottleneck, choice_runs):
    L.wrap_up(result_one_by_one, result_bottleneck, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
