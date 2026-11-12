# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-mobilenet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-mobilenet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Groepen van channels: ResNeXt](/course/open/cv-resnext?lang=nl) en [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_groups():
    quick_groups = L.choice(
        "quick-groups",
        question="`nn.Conv2d(64, 64, 3, padding=1, groups=4)` verdeelt zijn 64 channels in 4 groepen. Hoeveel weights heeft elk van zijn filters?",
        options={
            "right": "144: het ziet de 16 channels van zijn eigen groep, 16 × 3 × 3",
            "all": "576: het ziet alle 64 channels, 64 × 3 × 3",
            "four": "36: één filter van 3 × 3 voor elk van de 4 groepen",
        },
        correct="right",
        why={
            "right": "Met 4 groepen heeft elke groep 64 / 4 = 16 channels, en ziet elk filter alleen de channels van zijn groep: 16 × 9 = 144 weights. Vandaag worden de groepen zo klein als het kan: één channel per groep.",
            "all": "Dat is een convolutie zonder groepen. Met `groups=4` ziet een filter alleen de 16 channels van zijn eigen groep. Zie [ResNeXt](/course/open/cv-resnext?lang=nl).",
            "four": "Een filter is zo diep als de channels van zijn groep, en een groep heeft er 64 / 4 = 16. Zie [ResNeXt](/course/open/cv-resnext?lang=nl).",
        },
    )
    quick_groups
    return (quick_groups,)


@app.cell(hide_code=True)
def quick_check_pointwise():
    quick_pointwise = L.choice(
        "quick-pointwise",
        question="Een 1×1-convolutie van 64 channels naar 64 rekent één getal van zijn output uit, bij de pixel in rij 2, kolom 3. Welke getallen van zijn input gebruikt hij?",
        options={
            "right": "De 64 getallen van die pixel, één uit elk channel",
            "neighbors": "De 3 × 3 pixels eromheen, in elk channel",
            "one": "Alleen het getal van die pixel in één channel",
        },
        correct="right",
        why={
            "right": "Een 1×1-filter is 64 getallen diep en 1 pixel breed: het mengt de channels van één pixel, en kijkt niet naar zijn buren. Vandaag is de 1×1-convolutie een van de twee stappen die het werk van een 3×3-convolutie delen.",
            "neighbors": "Dat is een 3×3-convolutie. Een 1×1-filter beslaat één pixel, in al zijn channels. Zie [1×1-convoluties](/course/open/cv-1x1-convolution?lang=nl).",
            "one": "Een 1×1-filter is zo diep als de channels die het krijgt: het gebruikt alle 64 getallen van de pixel. Zie [1×1-convoluties](/course/open/cv-1x1-convolution?lang=nl).",
        },
    )
    quick_pointwise
    return (quick_pointwise,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een telefoon kan niet veel rekenen, en zijn batterij raakt leeg. In 2017 bouwden Andrew Howard en zijn collega's bij Google MobileNet voor zulke computers. Een 3×3-convolutie doet twee taken tegelijk: elk filter kijkt naar de buren van een pixel, en telt op wat het in alle channels vindt. MobileNet splitst die twee: eerst een convolutie die elk channel een eigen 3×3-filter geeft, de grouped convolution uit [Groepen van channels: ResNeXt](/course/open/cv-resnext?lang=nl) met evenveel groepen als channels; dan een 1×1-convolutie die de channels mengt, zoals in [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl). Op ImageNet had MobileNet 70,6% van de foto's goed, met 4,2 miljoen parameters en 569 miljoen vermenigvuldigingen per foto. Hetzelfde netwerk met gewone 3×3-convoluties had er 71,7% goed, met 29,3 miljoen parameters en 4.866 miljoen vermenigvuldigingen; VGG-16 71,5%, met 138 miljoen en 15.300 miljoen. Vandaag tel je wat de splitsing bespaart, bouw je hem, en test je in een CNN op CIFAR-10 wat elk van zijn twee stappen doet.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_mults():
    guess_mults = L.guess(
        "separable-mults",
        question="Een 3×3-convolutie van 64 channels naar 64, op maps van 8 bij 8, geeft 4.096 getallen: 64 channels van 64 pixels. Elk ervan kost 9 × 64 = 576 vermenigvuldigingen, samen 2.359.296. Een depthwise convolution, een 3×3-filter voor elk channel apart, met daarna een 1×1-convolutie van 64 channels naar 64, geeft maps van dezelfde shape. Hoeveel keer zo weinig vermenigvuldigingen heeft die nodig?",
        options={
            "two": "Ongeveer 2 keer zo weinig",
            "right": "Ongeveer 8 keer zo weinig",
            "many": "Ongeveer 64 keer zo weinig",
        },
        reveal="**Ongeveer 8 keer: 7,89.** De depthwise convolution heeft 9 vermenigvuldigingen nodig voor elk van zijn 4.096 getallen, 36.864; de 1×1-convolutie 64 voor elk, 262.144. Samen 299.008, tegen 2.359.296. Bijna alles zit in de 1×1-convolutie: de depthwise convolution kijkt bijna gratis naar de buren.",
    )
    guess_mults
    return (guess_mults,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De depthwise convolution.** Een depthwise convolution doet alleen de eerste taak van een 3×3-convolutie: hij geeft elk channel een eigen 3×3-filter, dat dat channel ziet en geen ander, en er één map van maakt. In PyTorch is het een grouped convolution met evenveel groepen als channels: `nn.Conv2d(64, 64, 3, padding=1, groups=64)` heeft 64 filters van 1 × 3 × 3. In zijn eentje mengt hij niets: wat in channel 5 zit, blijft in channel 5.

        **De depthwise separable convolution.** Een 1×1-convolutie erna doet de tweede taak: hij mengt de channels van elke pixel. De paper noemt hem de pointwise convolution, omdat hij naar één pixel tegelijk kijkt. Samen zijn ze een **depthwise separable convolution**: eerst elk channel filteren, dan mengen. Voor één pixel van de output, met $M$ channels erin, $N$ eruit en filters van $k$ bij $k$, heeft een gewone convolutie $k^2 M N$ vermenigvuldigingen nodig; de depthwise convolution $k^2 M$, en de 1×1-convolutie $M N$. Gedeeld door die van de gewone convolutie is dat

        $$\frac{k^2 M + M N}{k^2 M N} = \frac{1}{N} + \frac{1}{k^2}.$$

        Met 3×3-filters en veel channels is dat iets meer dan 1/9: 8 tot 9 keer zo weinig vermenigvuldigingen, zoals de paper zegt. Met stride 1 gebruikt een convolutie elk van zijn weights één keer voor elke pixel die hij geeft, dus neemt het aantal weights net zo sterk af, de biases niet meegerekend.

        Het plaatje: de weights van drie convoluties met 4 channels erin, een kolom voor elk channel en een rij voor elk filter. Een 3×3-convolutie naar 3 channels heeft op elke plek 3×3 weights; de depthwise convolution alleen op de diagonaal, omdat elk filter alleen zijn eigen channel ziet; de 1×1-convolutie naar 3 channels op elke plek één weight.
        """),
        C.filters_picture({"full": "3×3-convolutie", "depthwise": "depthwise convolution", "pointwise": "dan 1×1-convolutie",
                           "channels": "{n} channels erin", "filters": "filters", "weights": "{n} weights",
                           "alt": "Drie roosters van weights, elk met 4 kolommen voor de 4 channels die binnenkomen. De 3×3-convolutie: 3 rijen filters, op elke plek een rooster van 3×3 weights, 108 weights. De depthwise convolution: 4 rijen, alleen een rooster van 3×3 waar de rij en de kolom hetzelfde channel zijn, de andere plekken leeg, 36 weights. De 1×1-convolutie: 3 rijen met op elke plek één weight, 12 weights."}),
        mo.md(r"""
        **MobileNet** is één gewone 3×3-convolutie met daarna 13 depthwise separable convolutions, met batch normalization en een ReLU na elke depthwise en elke 1×1-convolutie. Een paar van zijn depthwise convolutions hebben stride 2, wat de maps kleiner maakt. Tabel 3 van de paper telt één layer ervan, met 512 channels erin en eruit op maps van 14 bij 14:

        | Eén layer | Vermenigvuldigingen | Weights |
        |---|---|---|
        | 3×3-convolutie | 462 miljoen | 2,36 miljoen |
        | depthwise separable convolution | 52,3 miljoen | 0,27 miljoen |

        En hele netwerken op ImageNet, uit de tabellen 4 en 8:

        | Netwerk | Goed op ImageNet | Vermenigvuldigingen per foto | Parameters |
        |---|---|---|---|
        | MobileNet met gewone 3×3-convoluties | 71,7% | 4.866 miljoen | 29,3 miljoen |
        | MobileNet | 70,6% | 569 miljoen | 4,2 miljoen |
        | GoogLeNet | 69,8% | 1.550 miljoen | 6,8 miljoen |
        | VGG-16 | 71,5% | 15.300 miljoen | 138 miljoen |
        """),
        L.watch_out("Minder vermenigvuldigingen maken een layer niet altijd evenveel sneller. Hoe snel een layer draait, hangt ook af van hoeveel werk hij doet voor elk getal dat hij leest: een depthwise convolution gebruikt elk getal in 9 vermenigvuldigingen, een 3×3-convolutie van 64 channels naar 64 in 576. De paper stopte 95% van de vermenigvuldigingen van MobileNet in zijn 1×1-convoluties, die een computer als één grote matrixvermenigvuldiging kan doen, iets waar hij snel in is. En de getallen uit de paper komen van ImageNet, met batch normalization en foto's van 224 bij 224: wat de splitsing doet in een klein netwerk op CIFAR-10, moet de demo laten zien."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **depthwise convolution** | een convolutie die elk channel een eigen filter geeft, dat alleen dat channel ziet: een grouped convolution met evenveel groepen als channels | `nn.Conv2d(64, 64, 3, padding=1, groups=64)` |
        | **depthwise separable convolution** | een depthwise convolution met een 1×1-convolutie erna: eerst elk channel filteren, dan de channels mengen | `SeparableConv(64, 64)` |
        | **MobileNet** | een CNN voor telefoons, gebouwd uit depthwise separable convolutions; uit 2017 | — |
        | $M$, $N$ | de channels die een convolutie in gaan, en de channels die eruit komen | `channels_in`, `channels_out` |

        **Het netwerk van de demo.** De demo traint het netwerk uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), met een depthwise separable convolution van 64 channels naar 64 in zijn midden, op de 64 maps van 8 bij 8: 4.800 parameters en 299.008 vermenigvuldigingen per foto, waar één 3×3-convolutie, zoals in die les, er 36.928 en 2.359.296 heeft. Om te zien wat elke stap doet, kan het midden ook twee depthwise separable convolutions na elkaar zijn, met 9.600 parameters, alleen de depthwise convolution, met 640, of alleen de 1×1-convolutie, met 4.160.

        Het plaatje: hoeveel van de eerste 1.000 testfoto's van CIFAR-10 het netwerk goed leest na 20 epochs op de eerste 6.000 trainingsfoto's, een stip voor elk van 5 seeds, met de code van de course. De bovenste rij komt uit die les: één 3×3-convolutie, 532 tot 572. De depthwise separable convolution: 548 tot 581. Twee na elkaar: 527 tot 541. Alleen de depthwise convolution: 557 tot 585; alleen de 1×1-convolutie: 557 tot 576. Oefening 3 vraagt wat de rijen zeggen.
        """),
        mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                "plain": "één 3×3-convolutie", "separable": "depthwise separable convolution",
                                "two": "twee depthwise separable convolutions", "depthwise": "alleen de depthwise convolution",
                                "pointwise": "alleen de 1×1-convolutie", "mine": "jouw run", "course": "de runs van de course",
                                "alt": "Vijf rijen open stippen, testfoto's goed gelezen van 1.000, een stip voor elke seed en een kort streepje bij hun gemiddelde. Eén 3×3-convolutie: 532 tot 572. Depthwise separable convolution: 548 tot 581. Twee daarvan na elkaar: 527 tot 541. Alleen de depthwise convolution: 557 tot 585. Alleen de 1×1-convolutie: 557 tot 576."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een depthwise separable convolution van `channels_in` channels naar `channels_out`:

    1. Maak in `__init__` de depthwise convolution: een 3×3-convolutie van `channels_in` channels naar `channels_in`, met `padding=1` en `groups=channels_in`, zodat elk channel een eigen filter krijgt.
    2. Maak de 1×1-convolutie, van `channels_in` channels naar `channels_out`.
    3. Stuur in `forward` `x` door de depthwise convolution en een ReLU, en dan door de 1×1-convolutie en een ReLU.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt een 3×3-convolutie van 64 channels naar 64, en de twee stappen van een depthwise separable convolution die zijn werk doen, en stuurt verzonnen maps van één foto, 64 channels van 8 bij 8, door allebei. Hij print de shapes van wat eruit komt, de shapes van de weights, en de vermenigvuldigingen van elk voor één foto: zijn weights keer de 8 × 8 pixels die hij geeft. Voer de cel uit zoals hij is, verander hem dan waar het commentaar het zegt, en voer hem opnieuw uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_separable():
    import torch
    from torch import nn

    x = torch.rand(1, 64, 8, 8)
    channels_out = 64  # verander: channels_out = 512
    full = nn.Conv2d(64, channels_out, 3, padding=1)
    # één 3×3-filter voor elk van de 64 channels, dan een 1×1-convolutie die ze mengt
    depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
    pointwise = nn.Conv2d(64, channels_out, 1)
    with torch.no_grad():
        print(list(full(x).shape), list(pointwise(depthwise(x)).shape))
    print(list(full.weight.shape), list(depthwise.weight.shape), list(pointwise.weight.shape))
    # elke weight wordt één keer gebruikt voor elk van de 8 × 8 pixels van de output
    full_mults = full.weight.numel() * 8 * 8
    separable_mults = (depthwise.weight.numel() + pointwise.weight.numel()) * 8 * 8
    print(full_mults, separable_mults, full_mults / separable_mults)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, print de cel `[1, 64, 8, 8] [1, 64, 8, 8]`: allebei geven ze 64 maps van 8 bij 8. Dan `[64, 64, 3, 3] [64, 1, 3, 3] [64, 64, 1, 1]`: de 3×3-convolutie heeft 64 filters van 64 × 3 × 3, de depthwise convolution 64 filters van 1 × 3 × 3, één voor elk channel, en de 1×1-convolutie 64 filters van 64 × 1 × 1. En `2359296 299008 7.890410958904109`: 7,89 keer zo weinig vermenigvuldigingen. Met `channels_out = 512` blijft de depthwise convolution gelijk, krijgen de andere twee 512 filters, en print de cel `18874368 2134016 8.844529750479847`: 8,84 keer zo weinig, zoals in de layer met 512 channels uit de paper.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif met de channels die eruit komen, van 4 tot 2.048, met evenveel erin. De bovenste balk zijn de vermenigvuldigingen van de 3×3-convolutie, 100%. Daaronder die van de depthwise separable convolution: licht die van de 1×1-convolutie, donker die van de depthwise convolution erachter. De gestreepte lijn staat op 1/9.
    """)
    return


@app.cell(hide_code=True)
def play_channels():
    _labels = {"title": "Depthwise separable tegen een 3×3-convolutie", "x": "vermenigvuldigingen, in % van die van de 3×3-convolutie",
               "full": "3×3-convolutie: 100%", "separable": "depthwise separable: {t}%", "floor": "de 1×1 alleen: 1/9",
               "alt": "Twee balken vermenigvuldigingen, in % van die van de 3×3-convolutie: de 3×3-convolutie op 100%, en daaronder de depthwise separable convolution, het deel van de 1×1-convolutie, 1/9, licht en het deel van de depthwise convolution donker erachter, met een gestreepte lijn op 1/9."}
    explore_channels = L.explore("channels-slider", "channels", lambda _v: (C.share_picture(int(_v), _labels, L.lang), C.share_sentence(int(_v), L.lang)),
                                 start=64, values=C.CHANNELS_OUT, role="hyper",
                                 task="Schuif de channels op tot 2.048. Hoeveel keer zo weinig vermenigvuldigingen benadert de depthwise separable convolution, zonder het ooit te halen?")
    explore_channels
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bijna 9, nooit 9: de 1×1-convolutie heeft in zijn eentje al 1/9 van de vermenigvuldigingen van de 3×3-convolutie nodig, hoeveel channels er ook zijn, want een 1×1-filter beslaat 1 pixel waar een 3×3-filter er 9 beslaat. De depthwise convolution voegt daar $1/N$ aan toe, en dat wordt kleiner naarmate er meer channels zijn. Met 4 channels zijn de twee samen maar 2,77 keer zo goedkoop; met 64, zoals in de demo, 7,89 keer; met 512, zoals in de layer uit de paper, 8,84 keer; en met 2.048 channels 8,96 keer. Met weinig channels is de depthwise convolution een groot deel van de kosten; met veel zit bijna alles in de 1×1-convolutie.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke stap doet wat

    Zes zinnen over de twee stappen van een depthwise separable convolution met 3×3-filters. Welke gaan over de depthwise convolution, en welke over de 1×1-convolutie erna? Deel elke zin in.
    """)
    return


@app.cell(hide_code=True)
def exercise_steps():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps",
        [
            "De depthwise convolution heeft 3×3-filters die elk één channel zien; de 1×1-convolutie heeft filters van 1 bij 1 die alle channels zien. Welke van de twee kan buren zien, en welke kan mengen?",
            "Bij elk horen drie zinnen. Voor de vermenigvuldigingen van één pixel: 9 × 64 voor de depthwise convolution, 64 × 64 voor de 1×1-convolutie.",
        ],
        why="**De depthwise convolution:** hij kijkt naar de 3×3 buren van elke pixel, elk van zijn filters ziet maar één channel, en zijn `groups` is het aantal channels dat hij krijgt. **De 1×1-convolutie:** hij mengt de channels van elke pixel, hij bepaalt hoeveel channels eruit komen, en met 64 channels erin en eruit doet hij de meeste vermenigvuldigingen, 4.096 per pixel tegen 576.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_separable():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een depthwise separable convolution

        Maak `SeparableConv` af. Zijn argumenten zijn de channels die binnenkomen, `channels_in`, en de channels die eruit gaan, `channels_out`. `forward` staat in het vak: die stuurt `x` door `self.depthwise` en een ReLU, en dan door `self.pointwise` en een ReLU. Maak de twee convoluties: de depthwise, een 3×3-filter voor elk channel apart, die de hoogte en breedte houdt, en de 1×1. De check bouwt je block, met 64 channels erin en eruit en met andere getallen, en vergelijkt wat eruit komt, de shapes van zijn weights en hoeveel parameters het heeft met die van de course. Als het klopt, bewaart hij je class in je project, als `work/projects/cnn/separable.py`.
        """),
        L.task("Maak de depthwise convolution en de 1×1-convolutie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_separable():
    box_separable = L.code_box("separable", lines=16)
    box_separable
    return (box_separable,)


@app.cell(hide_code=True)
def check_separable(box_separable):
    result_separable = L.check("separable", box_separable.value)
    result_separable
    return (result_separable,)


@app.cell(hide_code=True)
def hints_separable():
    L.hints(
        "separable",
        [
            "De depthwise convolution maakt één map uit elk channel dat hij krijgt: hoeveel filters heeft hij, en hoeveel groepen? Een 3×3-filter houdt de hoogte en breedte met `padding=1`.",
            "`self.depthwise = nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in)`. De 1×1-convolutie is stap 2 van het recept.",
        ],
        why="De depthwise convolution is `nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in)`, en de 1×1-convolutie `nn.Conv2d(channels_in, channels_out, 1)`. MobileNet zet batch normalization na elk van de twee, vóór zijn ReLU, en geeft een paar van zijn depthwise convolutions stride 2.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint het netwerk van de demo met het midden dat je kiest op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test het op de eerste 1.000 testfoto's. Kies het midden: je `SeparableConv`, twee daarvan na elkaar, of, van de course, alleen de depthwise convolution of alleen de 1×1-convolutie, die geen block van jou nodig hebben. Waar een module van jou uit een eerdere les ontbreekt of kapot is, springt die van de course in. Het heeft CIFAR-10 nodig, en voor je eigen block oefening 2, en duurt een paar minuten. Tijdens het trainen toont een grafiek de loss op de trainingsfoto's elke 10 batches, en gestreept op de testfoto's na elke epoch. Naast of onder de grafiek staat wat het netwerk op dat moment gokt bij één testfoto van elke soort, en daaronder de feature maps die zijn midden van een auto maakt; allebei worden ze opnieuw getekend terwijl het traint. In het plaatje daarna is de grijze lijn de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0. Daarna kun je terugspoelen naar een paar epochs van je run, met beide plaatjes zoals ze toen waren, en zie je je run naast die van de course.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. Heb je [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) op deze computer gedaan, dan wel: dan vindt deze les het. Anders downloadt de knop het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. De pagina van CIFAR-10 noemt geen licentie, dus levert de course het niet mee; waar de foto's vandaan komen, staat in die projectles. Alleen de demo heeft het nodig. Tijdens het downloaden wacht de hele les.*",
        "busy": "**CIFAR-10 downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van CIFAR-10 uitpakken** · dit duurt een paar seconden",
        "no_net": "*De download is niet gelukt: de University of Toronto was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download is niet gelukt: het bestand dat binnenkwam is niet het vastgelegde bestand van de course, dus heeft de course het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "damaged": "*Uitpakken is niet gelukt: het gedownloade bestand bleek beschadigd, dus heeft de course het verwijderd. Klik opnieuw om het opnieuw te downloaden; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download is niet gelukt: er is op je schijf geen ruimte voor 170 MB en de uitgepakte foto's, 184 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download is niet gelukt ({why}). Klik opnieuw.*",
        "here": "*CIFAR-10 staat op je computer: 50.000 trainingsfoto's en 10.000 testfoto's.*",
    })
    return (cifar,)


@app.cell(hide_code=True)
def variant_choice_cell():
    variant_choice = mo.ui.radio(options={"je depthwise separable convolution": "separable", "twee daarvan na elkaar": "two",
                                          "alleen de depthwise convolution": "depthwise", "alleen de 1×1-convolutie": "pointwise"},
                                 value="je depthwise separable convolution", label="Het midden van het netwerk")
    train_button = mo.ui.run_button(label="Train het netwerk")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_separable(cifar, train_button, variant_choice, result_separable):
    cifar, result_separable  # opnieuw uitvoeren als CIFAR-10 binnenkomt, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het netwerk te trainen.*")
    else:
        _variant = variant_choice.value
        _names = {"separable": "Depthwise separable", "two": "Twee depthwise separable", "depthwise": "Alleen depthwise", "pointwise": "Alleen 1×1"}
        _with = {"separable": "Met je depthwise separable convolution", "two": "Met twee van je depthwise separable convolutions na elkaar",
                 "depthwise": "Met alleen de depthwise convolution", "pointwise": "Met alleen de 1×1-convolutie"}
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht
        _layer = "Het midden van het netwerk op een testfoto van een auto: layer {name}"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je netwerk op CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "De loss van het netwerk tot nu toe: op de trainingsfoto's elke 10 batches, en gestreept op de 1.000 testfoto's na elke epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "trainingsfoto's, elke 10 batches"), ("loss", True, "testfoto's, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-mobilenet-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2: dan staat je `SeparableConv` in je project, en gebruikt deze cel hem. Alleen de depthwise of alleen de 1×1-convolutie heeft geen block van jou nodig.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `SeparableConv` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "separable": "*Je `SeparableConv` in `work/projects/cnn/separable.py` stopt met een fout, of geeft niet terug wat hij moet geven: met 64 channels erin en eruit, 64 maps van 8 bij 8. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validatie",
                       "plain": "één 3×3-convolutie",
                       "alt": "De loss van het netwerk na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0, die daalt tot epoch 6 en daarna stijgt."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _layer, L.lang),
                        f"Na epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, en {_right} van deze {len(C.SHOWN)} foto's goed gelezen.")

            if len(_when) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: wanneer is de validation loss het laagst, en wat gebeurt er daarna?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""{_with[_variant]} leest het netwerk **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters, waarvan {C.grouped(_result['middle_params'], L.lang)} in zijn midden.{_turn} Over 5 seeds, met de code van de course, dit midden: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's, met de laagste validation loss na epoch {_be[0]} tot {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                        "plain": "één 3×3-convolutie", "separable": "depthwise separable convolution",
                                        "two": "twee depthwise separable convolutions", "depthwise": "alleen de depthwise convolution",
                                        "pointwise": "alleen de 1×1-convolutie", "mine": "jouw run", "course": "de runs van de course",
                                        "alt": "Vijf rijen testfoto's goed gelezen, van 1.000: open stippen voor de runs van de course over 5 seeds, met één 3×3-convolutie, met de depthwise separable convolution, met twee daarvan, en met alleen de depthwise of alleen de 1×1-convolutie, met een kort streepje bij het gemiddelde van elke rij; en in de rij van jouw midden een dichte stip voor jouw run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Oefening 3. Wat de splitsing hier doet
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "separable-runs",
        question="Over 5 seeds, na 20 epochs. Met één 3×3-convolutie in zijn midden leest het netwerk 532 tot 572 testfoto's goed; met een depthwise separable convolution 548 tot 581, met 8 keer zo weinig vermenigvuldigingen en weights in zijn midden. Met alleen de depthwise convolution leest het er 557 tot 585, met alleen de 1×1-convolutie 557 tot 576, en met twee depthwise separable convolutions na elkaar 527 tot 541. Wat zeggen de getallen?",
        options={
            "both": "Elke stap heeft de andere nodig: in zijn eentje leest de depthwise of de 1×1-convolutie minder foto's goed dan de twee samen",
            "more": "Hoe meer depthwise separable convolutions, hoe beter: twee na elkaar lezen er de meeste",
            "right": "Hier kost de splitsing niets: de depthwise separable convolution houdt de 3×3-convolutie bij voor een achtste van de kosten, maar elke stap doet het in zijn eentje even goed, dus deze runs kunnen niet laten zien wat elke stap toevoegt",
        },
        correct="right",
        why={
            "both": "In hun eentje lezen ze er 557 tot 585 en 557 tot 576 goed: evenveel als de twee samen, 548 tot 581, of meer. In dit netwerk heeft het midden misschien weinig meer te doen: de twee 3×3-convoluties ervoor kijken al naar de buren en mengen de channels, en de linear layer erna mengt alles. Dat elke stap nodig is, kunnen deze runs niet laten zien.",
            "more": "Twee na elkaar lezen er het minst, 527 tot 541, en leren minder van hun 6.000 trainingsfoto's, 5.436 tot 5.735, tegen 5.829 tot 5.982 voor één. Meer layers zijn hier niet beter. MobileNet heeft batch normalization na elke convolutie; dit netwerk heeft die niet.",
            "right": "De depthwise separable convolution leest 548 tot 581 testfoto's goed, de 3×3-convolutie 532 tot 572: hun runs overlappen, met 299.008 vermenigvuldigingen in het midden per foto tegen 2.359.296, en 4.800 parameters tegen 36.928. Op ImageNet, in de paper, kostte de splitsing een beetje: 70,6% goed tegen 71,7%; hier niets wat je kunt zien. Maar alleen de depthwise convolution, met 640 parameters, leest er 557 tot 585, en alleen de 1×1-convolutie 557 tot 576: in dit kleine netwerk, met twee 3×3-convoluties vóór het midden, doet elke stap het in zijn eentje even goed. Wat elke stap toevoegt, kunnen deze runs niet laten zien: een uitkomst geldt waar ze gemeten is. En twee na elkaar lezen er minder, 527 tot 541.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een depthwise convolution is de grouped convolution uit [Groepen van channels: ResNeXt](/course/open/cv-resnext?lang=nl) met evenveel groepen als channels, en de 1×1-convolutie erna is die uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl). Je telde hun weights en vermenigvuldigingen met de shapes uit [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl), en trainde ze met de Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) op de foto's uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl). Je project `cnn` heeft nu `separable.py`. In de volgende les, *EfficientNet: breder, dieper of scherper*, groeit een netwerk van zulke blocks in diepte, in breedte en in de grootte van zijn foto's, en reken je uit wat elke keuze kost.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_steps, result_separable, choice_runs):
    L.wrap_up(result_steps, result_separable, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
