# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-resnext. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-resnext", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Inception: filters van verschillende groottes naast elkaar](/course/open/cv-inception?lang=nl) en [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cat():
    quick_cat = L.choice(
        "quick-cat",
        question="De vier takken van een Inception block geven 16, 32, 8 en 8 maps van 8 bij 8, voor een batch van 2 foto's. Welke shape geeft `torch.cat([y1, y3, y5, yp], dim=1)`?",
        options={
            "right": "(2, 64, 8, 8)",
            "stack": "(2, 4, 16, 8, 8): vier takken van 16",
            "batch": "(8, 16, 8, 8): de foto's van de vier takken na elkaar",
        },
        correct="right",
        why={
            "right": "Langs `dim=1`, de channels, komen de maps naast elkaar: 16 + 32 + 8 + 8 = 64. Vandaag doet een convolutie het omgekeerde: hij verdeelt zijn channels in groepen, en geeft elke groep eigen filters.",
            "stack": "`torch.cat` maakt geen nieuwe axis: hij zet de tensors aan elkaar langs een axis die ze al hebben, hier de channels. En de takken geven verschillende aantallen channels, dus stapelen kon niet eens. Zie [Inception](/course/open/cv-inception?lang=nl).",
            "batch": "Langs `dim=0`, de foto's, zou `torch.cat` stoppen: de takken geven verschillende aantallen channels. Langs `dim=1`, de channels, houdt de batch zijn 2 foto's, elk met 16 + 32 + 8 + 8 maps. Zie [Inception](/course/open/cv-inception?lang=nl).",
        },
    )
    quick_cat
    return (quick_cat,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Een residual block geeft $x + f(x)$ terug. Zijn input $x$ is 64 channels van 8 bij 8. Wat moet $f(x)$ geven?",
        options={
            "right": "64 channels van 8 bij 8: de shape van $x$",
            "any": "Elk aantal channels: `+` zet ze naast elkaar",
            "fewer": "32 channels van 8 bij 8: binnen het block worden de channels minder",
        },
        correct="right",
        why={
            "right": "In het block van ResNet heeft $f(x)$ de shape van $x$, zodat `+` bij elk getal van $x$ een eigen getal optelt. Het block van vandaag maakt de channels minder binnen $f$, en aan het eind weer meer, zodat het past.",
            "any": "Maps naast elkaar zetten is `torch.cat`. `+` telt ze getal voor getal op, en stopt als de shapes niet passen. Zie [ResNet](/course/open/cnn-resnet?lang=nl).",
            "fewer": "Dan stopt `+`: 32 channels passen getal voor getal niet op 64. Een block mag de channels binnen $f$ minder maken, als $f$ maar weer met 64 eindigt. Zie [ResNet](/course/open/cnn-resnet?lang=nl).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Inception: filters van verschillende groottes naast elkaar](/course/open/cv-inception?lang=nl) had elke tak een eigen ontwerp: filters van 1, 3 of 5 bij 5, met aantallen filters die voor elk van de 9 blocks van GoogLeNet met de hand gekozen waren. In 2016 behielden Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu en Kaiming He, een van de makers van [ResNet](/course/open/cnn-resnet?lang=nl), de vele paden naast elkaar, maar gaven ze allemaal dezelfde vorm: 32 dunne paden in één residual block. Omdat de paden dezelfde vorm hebben, is het hele block drie convoluties, de middelste een grouped convolution: een truc zo oud als AlexNet, dat zijn filters over twee GPU's verdeelde. Hun ResNeXt-50 heeft ongeveer evenveel parameters als ResNet-50, 25,0 miljoen tegen 25,5, en zijn eerste gok was fout voor 22,2% van de validatiefoto's van ImageNet, tegen 23,9%. ResNeXt werd tweede in de wedstrijd van ImageNet in 2016. Vandaag tel je wat groepen besparen, bouw je een ResNeXt block, en test je in een CNN op CIFAR-10 of meer groepen daar ook helpen.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "grouped-weights",
        question="Een 3×3-convolutie van 128 channels naar 128, `nn.Conv2d(128, 128, 3)`, heeft 147.456 weights: 128 filters, elk van 128 × 3 × 3. Met `groups=32` worden de channels en de filters in 32 groepen verdeeld, en ziet elk filter alleen de channels van zijn eigen groep. Hoeveel weights dan?",
        options={
            "same": "147.456: dezelfde 128 filters, alleen in groepen",
            "right": "4.608",
            "one": "144: 4 filters van 4 × 3 × 3",
        },
        reveal="**4.608, 32 keer zo weinig.** Elk van de 32 groepen krijgt 4 van de 128 channels en 4 van de 128 filters, dus is elk filter 4 × 3 × 3 = 36 weights diep in plaats van 128 × 3 × 3 = 1.152. 128 filters van 36: 4.608. Eén groep alleen heeft er 144, en het zijn er 32. Wat eruit komt, heeft nog steeds 128 channels van dezelfde grootte: alleen welke channels elk filter ziet, is veranderd.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Groepen.** In een gewone convolutie ziet elk filter alle channels die binnenkomen. `nn.Conv2d(c_in, c_out, k, groups=G)` verdeelt de channels die binnenkomen in $G$ groepen van $c_{in} / G$, en zijn filters ook: elke groep van $c_{out} / G$ filters ziet alleen de channels van zijn eigen groep. Zijn weights krijgen de shape $(c_{out}, c_{in} / G, k, k)$, $G$ keer zo weinig, en wat eruit komt zijn de groepen naast elkaar langs de channels, in hun volgorde. Een grouped convolution is dus $G$ kleine convoluties naast elkaar, hun maps aan elkaar gezet zoals `torch.cat` doet: het uitgewerkte voorbeeld laat het zien. Zowel $c_{in}$ als $c_{out}$ moet deelbaar zijn door $G$.

        Het plaatje: een convolutie van 8 channels naar 8, een rij voor elk filter en een kolom voor elk channel dat binnenkomt. Een oranje vakje staat voor de 3 × 3 weights waarmee dat filter naar dat channel kijkt. Met één groep ziet elk filter alle 8 channels; met 4 groepen ziet elk filter de 2 van zijn eigen groep: 16 vakjes in plaats van 64.
        """),
        C.groups_picture({"title": "Welke channels elk filter ziet", "in": "channels erin", "out": "filters",
                                  "sub": "groups={g}: {n} vakjes",
                                  "alt": "Twee roosters van 8 bij 8 vakjes, een rij voor elk filter en een kolom voor elk channel dat binnenkomt. Links, groups=1: alle 64 vakjes oranje, elk filter ziet alle 8 channels. Rechts, groups=4: vier oranje blokken van 2 bij 2 op de diagonaal, 16 vakjes: elk filter ziet alleen de 2 channels van zijn eigen groep."}),
        mo.md(r"""
        **Het ResNeXt block.** De paper van ResNeXt begint bij 32 paden van dezelfde vorm naast elkaar in een residual block: elk pad is een 1×1-convolutie van 256 channels naar 4, een 3×3-convolutie van 4 naar 4, en een 1×1-convolutie van 4 terug naar 256. Het block telt op wat de 32 paden geven, en telt daar zijn input $x$ bij op, zoals in [ResNet](/course/open/cnn-resnet?lang=nl). Omdat de paden dezelfde vorm hebben, passen ze in drie convoluties. De 32 eerste samen zijn één 1×1-convolutie van 256 channels naar 128. De 32 middelste, elk op zijn eigen 4 channels, zijn één 3×3-convolutie van 128 naar 128 in 32 groepen. En de 32 laatste optellen is één 1×1-convolutie van 128 terug naar 256. De paper noemt het aantal paden de **cardinality**, en schrijft een block als C×d, voor C paden van d channels: ResNeXt-50 heeft 32×4d.

        **Dezelfde kosten, twee keer zo breed.** Dit block, met 256 channels erin en eruit zoals in ResNeXt-50, heeft 70.656 parameters, geteld met een bias voor elk filter, zoals `nn.Conv2d` ze maakt. De bottleneck van ResNet-50 uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), van 256 channels naar 64, van 64 naar 64, en terug naar 256, heeft er 70.016. In het midden is het block van ResNeXt voor die kosten twee keer zo breed, 128 channels tegen 64; zonder groepen zou het 213.504 parameters hebben. Tabel 3 van de paper vergelijkt blocks van ongeveer dezelfde kosten op ImageNet, van 1 pad van 64 channels, ResNet-50 zelf, tot 32 paden van 4. De getallen zeggen voor hoeveel van de validatiefoto's van ImageNet de eerste gok van het netwerk fout was:

        | Block | Paden van channels | Fout op ImageNet |
        |---|---|---|
        | 1×64d, ResNet-50 | 1 van 64 | 23,9% |
        | 2×40d | 2 van 40 | 23,0% |
        | 4×24d | 4 van 24 | 22,6% |
        | 8×14d | 8 van 14 | 22,3% |
        | 32×4d, ResNeXt-50 | 32 van 4 | 22,2% |
        """),
        L.watch_out("De winst van 1,7 procentpunt is gemeten op ImageNet, 1,28 miljoen foto's van 1.000 soorten, met batch normalization na elke convolutie. Een uitkomst geldt waar hij gemeten is: op een klein netwerk en een paar duizend foto's kan het verschil kleiner zijn dan wat de seed verandert. Dat test de demo. En net als het eerste ResNet van 2015 zet de paper van ResNeXt een ReLU na het optellen; de course laat die weg, zoals in [ResNet](/course/open/cnn-resnet?lang=nl), zodat de input er onveranderd doorheen komt."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **grouped convolution** | een convolutie waarvan de channels en de filters in even grote groepen verdeeld zijn: elk filter ziet alleen de channels van zijn eigen groep, dus heeft de convolutie `groups` keer zo weinig weights | `nn.Conv2d(128, 128, 3, groups=32)` |
        | **ResNeXt** | een ResNet waarvan de blocks veel dunne paden van dezelfde vorm zijn, geschreven als een 1×1-convolutie, een grouped 3×3-convolutie en een 1×1-convolutie, met de input erbij opgeteld; uit 2016 | `ResNeXtBlock(256, 128, 32)` |
        | **cardinality** | het aantal paden in een ResNeXt block: de `groups` van zijn 3×3-convolutie. 32×4d is 32 paden van 4 channels | `groups=32` |

        **Het netwerk van de demo.** De demo traint het netwerk uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), met een ResNeXt block in het midden, op de 64 maps van 8 bij 8. Je kiest uit vier blocks: de 1×64d, 2×40d, 4×24d en 32×4d uit de paper met al hun channels gedeeld door 4, dus 1×16d, 2×10d, 4×6d en 32×1d. De 8×14d uit de paper zou 8 groepen van 3,5 channels geven, dus die valt af. Elk block heeft ongeveer 4.400 weights, 4.352, 4.360, 4.368 of 4.384, en het hele netwerk 34.090 tot 34.154 parameters. 1×16d heeft precies de drie convoluties van de bottleneck met 16 in het midden uit die les; het telt aan het eind de input erbij op, waar de bottleneck een ReLU heeft. In 32×1d ziet elk filter van 3 bij 3 maar één channel.

        Het plaatje: hoeveel van de eerste 1.000 testfoto's van CIFAR-10 het netwerk goed leest na 20 epochs op de eerste 6.000 trainingsfoto's, een stip voor elk van 5 seeds, met de code van de course. De twee bovenste rijen komen uit die les, zonder de input erbij: één 3×3-convolutie, 532 tot 572, en de bottleneck met 16 in het midden, 521 tot 550. De vier ResNeXt blocks: 562 tot 598, 565 tot 580, 572 tot 599 en 574 tot 584. Oefening 3 vraagt wat de rijen zeggen.
        """),
        mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                "plain": "één 3×3-convolutie, zonder x erbij", "b16": "bottleneck, 16 in het midden, zonder x erbij",
                                "1x16": "ResNeXt block 1×16d: 1 groep van 16 channels", "2x10": "2×10d: 2 groepen van 10",
                                "4x6": "4×6d: 4 groepen van 6", "32x1": "32×1d: 32 groepen van 1",
                                "mine": "je run", "course": "de runs van de course",
                                "alt": "Zes rijen open stippen, testfoto's goed gelezen van 1.000, één stip voor elke seed en een kort streepje bij hun gemiddelde. Eén 3×3-convolutie, zonder x erbij: 532 tot 572. Bottleneck met 16 in het midden, zonder x erbij: 521 tot 550. ResNeXt block 1×16d: 562 tot 598. 2×10d: 565 tot 580. 4×6d: 572 tot 599. 32×1d: 574 tot 584."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo bouw je een ResNeXt block voor `channels` channels, `width` in het midden, in `groups` groepen:

    1. Een 1×1-convolutie van `channels` naar `width`, met een ReLU.
    2. Een 3×3-convolutie van `width` naar `width` in `groups` groepen, met `padding=1` en een ReLU: elke groep heeft `width // groups` channels.
    3. Een 1×1-convolutie van `width` terug naar `channels`, zonder ReLU.
    4. Geef de input erbij opgeteld terug: `x + y`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt 32 kleine convoluties, elk van 4 channels naar 4, en één convolutie van 128 channels naar 128 in 32 groepen, en geeft die ene de weights en biases van de 32 kleine, na elkaar. Dan stuurt hij verzonnen maps van één foto, 128 channels van 14 bij 14, op twee manieren door: door de kleine convoluties, elk op zijn eigen 4 channels, naast elkaar, en in één keer door de grouped convolution. Hij print de shape en het aantal weights van de grouped convolution, en de eerste 4 channels op één pixel, van beide manieren. Voer de cel uit zoals hij is, verander hem dan waar het commentaar zegt, en voer hem nog eens uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_groups():
    import torch
    from torch import nn

    torch.manual_seed(0)
    x = torch.rand(1, 128, 14, 14)
    # 32 kleine convoluties, elk van 4 channels naar 4
    small = [nn.Conv2d(4, 4, 3, padding=1) for g in range(32)]
    # één convolutie van 128 channels naar 128, in 32 groepen
    conv = nn.Conv2d(128, 128, 3, padding=1, groups=32)  # verander: groups=1
    print(conv.weight.shape, conv.weight.numel())
    with torch.no_grad():
        # de weights en biases van de 32 kleine, na elkaar
        conv.weight[:] = torch.cat([s.weight for s in small])
        conv.bias[:] = torch.cat([s.bias for s in small])
        # elke kleine convolutie op zijn eigen 4 channels, naast elkaar
        parts = [small[g](x[:, 4 * g:4 * g + 4]) for g in range(32)]
        print(torch.cat(parts, dim=1)[0, :4, 0, 0])
        print(conv(x)[0, :4, 0, 0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel nu is, print hij `torch.Size([128, 4, 3, 3]) 4608`: 128 filters, elk 4 channels diep. `torch.cat` zet de weights van de 32 kleine convoluties, elk met de shape (4, 4, 3, 3), na elkaar langs de filters, en dat geeft precies die shape. Dan twee keer dezelfde 4 getallen, `tensor([ 0.4007, -0.6840, -0.4022,  0.1121])`: de grouped convolution is de 32 kleine naast elkaar. Met `groups=1` heeft de convolutie `torch.Size([128, 128, 3, 3]) 147456` weights, 32 keer zoveel: elk filter ziet alle 128 channels. Dan passen de weights van de kleine niet, en stopt PyTorch met `The expanded size of the tensor (128) must match the existing size (4) at non-singleton dimension 1`: weights van 4 channels diep, waar elk filter er nu 128 nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif het aantal groepen van de 3×3-convolutie in dit block van ResNeXt-50, met 256 channels, van 1 tot 128. De drie balken zijn de drie convoluties van het block; de as blijft dezelfde, in duizenden parameters.
    """)
    return


@app.cell(hide_code=True)
def play_groups():
    _labels = {"title": "Een block van ResNeXt-50, 256 channels", "x": "parameters (duizenden)",
               "reduce": "1×1-convolutie, 256 naar 128", "conv": "3×3-convolutie, 128 naar 128, in groepen",
               "expand": "1×1-convolutie, 128 terug naar 256",
               "alt": "Drie balken parameters, in duizenden, voor het gekozen aantal groepen, in oranje: de 1×1-convolutie van 256 channels naar 128, de 3×3-convolutie van 128 naar 128 in groepen, en de 1×1-convolutie van 128 terug naar 256."}
    explore_groups = L.explore("groups-slider", "groepen", lambda _v: (C.groups_bars(int(_v), _labels, L.lang), C.groups_sentence(int(_v), L.lang)),
                               start=32, values=C.GROUPS, role="hyper",
                               task="Vanaf hoeveel groepen heeft de 3×3-convolutie minder parameters dan de 1×1-convolutie ervoor?")
    explore_groups
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Vanaf 8 groepen: dan heeft de 3×3-convolutie 18.560 parameters, tegen 32.896 voor de 1×1-convolutie ervoor; met 4 groepen heeft hij er nog 36.992. Met 32 groepen, zoals in ResNeXt-50, heeft hij er 4.736, en zitten bijna alle 70.656 parameters van het block in zijn twee 1×1-convoluties: 65.920. Daarom kan ResNeXt het midden voor dezelfde kosten twee keer zo breed maken. Met 128 groepen ziet elk filter één channel: daar begint *Depthwise separable convolutions: MobileNet*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_groups():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke groepen werken

    Zes convoluties met `groups`. Welke kan PyTorch maken, en bij welke stopt hij? Deel elke convolutie in: hij werkt, of PyTorch stopt.
    """)
    return


@app.cell(hide_code=True)
def exercise_groups():
    sort_groups = L.sort("groups")
    sort_groups
    return (sort_groups,)


@app.cell(hide_code=True)
def check_groups(sort_groups):
    result_groups = L.check("groups", sort_groups)
    result_groups
    return (result_groups,)


@app.cell(hide_code=True)
def hints_groups():
    L.hints(
        "groups",
        [
            "In `nn.Conv2d(c_in, c_out, k, groups=G)` worden zowel de channels die binnenkomen als de filters in $G$ even grote groepen verdeeld. Reken voor elke convolutie $c_{in} / G$ en $c_{out} / G$ uit.",
            "Drie werken en drie stoppen. Een groep van één channel mag; een half channel of een half filter niet.",
        ],
        why="**Het werkt:** `nn.Conv2d(128, 128, 3, groups=32)`, 128 / 32 = 4 channels en 4 filters per groep; `nn.Conv2d(64, 64, 3, groups=64)`, 1 en 1; `nn.Conv2d(96, 64, 1, groups=8)`, 96 / 8 = 12 channels en 64 / 8 = 8 filters. **PyTorch stopt:** `nn.Conv2d(128, 100, 3, groups=32)`, 100 / 32 = 3,125 filters; `nn.Conv2d(128, 128, 3, groups=3)`, 128 / 3 is geen heel getal; `nn.Conv2d(64, 32, 3, groups=64)`, een half filter per groep.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_resnext():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een ResNeXt block

        Maak `ResNeXtBlock` af. Zijn argumenten zijn de channels die binnenkomen en eruit gaan, `channels`; de channels in het midden, `width`; en het aantal groepen, `groups`. `reduce` en `expand`, en de eerste regels van `forward`, staan in het vak. Maak `conv`, een 3×3-convolutie van `width` naar `width` in `groups` groepen die de hoogte en breedte houdt, en laat `forward` de input erbij opgeteld teruggeven. De check bouwt je block, met de getallen van een block van ResNeXt-50 met 256 channels en met verzonnen getallen, en vergelijkt wat eruit komt, en hoeveel parameters het heeft, met dat van de course. Als het klopt, bewaart hij je class in je project, als `work/projects/cnn/resnext.py`.
        """),
        L.task("Maak `conv`, en laat `forward` de input erbij opgeteld teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resnext():
    box_resnext = L.code_box("resnext", lines=19)
    box_resnext
    return (box_resnext,)


@app.cell(hide_code=True)
def check_resnext(box_resnext):
    result_resnext = L.check("resnext", box_resnext.value)
    result_resnext
    return (result_resnext,)


@app.cell(hide_code=True)
def hints_resnext():
    L.hints(
        "resnext",
        [
            "`nn.Conv2d` krijgt het aantal groepen als het argument `groups`. Welke padding houdt de hoogte en breedte bij 3 bij 3?",
            "`self.conv = nn.Conv2d(width, width, 3, padding=1, groups=groups)`. De laatste regel is stap 4 van het recept.",
        ],
        why="`conv` is `nn.Conv2d(width, width, 3, padding=1, groups=groups)`, en `forward` eindigt met `return x + y`. De blocks van ResNeXt zelf hebben batch normalization na elke convolutie, en een ReLU na het optellen, zoals het eerste ResNet had.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint het netwerk van de demo, met je `ResNeXtBlock` in het midden, op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test het op de eerste 1.000 testfoto's. Kies hoe zijn block het midden in groepen verdeelt: 1 groep van 16 channels, 2 van 10, 4 van 6 of 32 van 1. Waar een module van jou ontbreekt of kapot is, springt die van de course in. Hij heeft CIFAR-10 en oefening 2 nodig, en duurt een paar minuten. Tijdens het trainen toont een grafiek de loss op de trainingsfoto's elke 10 batches, en gestreept op de testfoto's na elke epoch. Naast of onder de grafiek staat wat het netwerk op dat moment gokt bij één testfoto van elke soort, en daaronder de feature maps die zijn midden van een auto maakt; allebei worden ze opnieuw getekend terwijl het traint. In het plaatje daarna is de grijze lijn de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0. Daarna kun je terugspoelen naar een paar epochs van je run, met beide plaatjes zoals ze toen waren, en zie je je run naast die van de course.
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
    variant_choice = mo.ui.radio(options={"1 groep van 16 channels, 1×16d": "1x16", "2 groepen van 10, 2×10d": "2x10",
                                          "4 groepen van 6, 4×6d": "4x6", "32 groepen van 1, 32×1d": "32x1"},
                                 value="32 groepen van 1, 32×1d", label="De 3×3-convolutie in het midden")
    train_button = mo.ui.run_button(label="Train het netwerk")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_resnext(cifar, train_button, variant_choice, result_resnext):
    cifar, result_resnext  # opnieuw uitvoeren als CIFAR-10 binnen is, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het netwerk te trainen.*")
    else:
        _variant = variant_choice.value
        _names = {"1x16": "1×16d", "2x10": "2×10d", "4x6": "4×6d", "32x1": "32×1d"}
        _with = {"1x16": "in 1 groep van 16 channels", "2x10": "in 2 groepen van 10 channels", "4x6": "in 4 groepen van 6 channels", "32x1": "in 32 groepen van 1 channel"}
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht
        _layer = "Het midden van het netwerk op een testfoto van een auto: layer {name}"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je netwerk op CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "De loss van het netwerk tot nu toe: op de trainingsfoto's elke 10 batches, en gestreept op de 1.000 testfoto's na elke epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "trainingsfoto's, elke 10 batches"), ("loss", True, "testfoto's, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-resnext-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**ResNeXt block {_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**ResNeXt block {_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**ResNeXt block {_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan staat je `ResNeXtBlock` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `ResNeXtBlock` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "resnext": "*Je `ResNeXtBlock` in `work/projects/cnn/resnext.py` stopt met een fout, of geeft geen 64 maps van 8 bij 8 terug. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "plain": "één 3×3-convolutie",
                       "alt": "De loss van het netwerk met het ResNeXt block na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course met één 3×3-convolutie in het midden en seed 0, die daalt tot epoch 6 en daarna stijgt."}
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
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog steeds.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""Met je ResNeXt block {_with[_variant]} leest het netwerk **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters, waarvan {C.grouped(_result['block_params'], L.lang)} in zijn block.{_turn} Over 5 seeds, met de code van de course, dit block: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's, met de laagste validation loss na epoch {_be[0]} tot {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                        "plain": "één 3×3-convolutie, zonder x erbij", "b16": "bottleneck, 16 in het midden, zonder x erbij",
                                        "1x16": "ResNeXt block 1×16d: 1 groep van 16 channels", "2x10": "2×10d: 2 groepen van 10",
                                        "4x6": "4×6d: 4 groepen van 6", "32x1": "32×1d: 32 groepen van 1",
                                        "mine": "je run", "course": "de runs van de course",
                                        "alt": "Zes rijen testfoto's goed gelezen, van 1.000: open stippen voor de runs van de course over 5 seeds, met één 3×3-convolutie en met de bottleneck met 16 in het midden, allebei zonder x erbij, en met de vier ResNeXt blocks, met een kort streepje bij het gemiddelde van elke rij; en in de rij van je block een dichte stip voor je run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Oefening 3. Wat de groepen hier doen
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "groups-runs",
        question="Over 5 seeds, na 20 epochs, elk netwerk met ongeveer 34.100 parameters. Met het ResNeXt block in 1 groep van 16 channels leest het 562 tot 598 testfoto's goed; in 2 groepen van 10, 565 tot 580; in 4 groepen van 6, 572 tot 599; in 32 groepen van 1, 574 tot 584. De bottleneck met 16 in het midden uit [1×1-convoluties](/course/open/cv-1x1-convolution?lang=nl), zonder de input erbij, leest 521 tot 550. Wat zeggen de getallen?",
        options={
            "more": "Meer groepen lezen meer foto's goed: 32 groepen van 1 is het best",
            "one": "Eén groep is het best: daar ziet elk filter alle 16 channels",
            "right": "De vier blocks overlappen, dus hier geven meer groepen geen duidelijke winst; het residual block wel: elk van de vier leest er meer goed dan de bottleneck zonder de input erbij",
        },
        correct="right",
        why={
            "more": "Het block met 32 groepen van 1 leest er 574 tot 584 goed, gemiddeld 580; met 1 groep van 16 leest het er gemiddeld 585, en met 4 groepen van 6 584. De hoogste run van allemaal, 599, is met 4 groepen. Met 5 seeds elk overlappen de vier bereiken: uit deze runs kun je de blocks niet uit elkaar houden.",
            "one": "Het block met 1 groep van 16 heeft het hoogste gemiddelde, 585, maar dat met 4 groepen van 6 heeft 584, en de vijf runs met 1 groep van 16 gaan van 562 tot 598: de seed alleen verschuift de telling van een block al 10 tot 36 foto's. De bereiken overlappen: uit deze runs kun je de blocks niet uit elkaar houden.",
            "right": "Met de input erbij lezen de vier blocks 562 tot 599 testfoto's goed; de bottleneck zonder, 521 tot 550: elke run van de vier verslaat elke run van de bottleneck. Voor 1×16d zijn het dezelfde drie convoluties, met de input aan het eind erbij en zonder de ReLU na `expand`. Met de input erbij leert het netwerk ook zijn trainingsfoto's, 5.938 tot 5.999 van de 6.000, tegen 5.077 tot 5.510. Tussen de vier blocks zijn de gemiddelden 585, 570, 584 en 580: ze stijgen of dalen niet met het aantal groepen. De winst uit de paper, 1,7 procentpunt op ImageNet, zou 17 van deze 1.000 foto's zijn; hier lezen 32 groepen er gemiddeld 5 minder goed dan één groep, en verschuift de seed de runs van één block al 10 tot 36 foto's: deze runs laten zo'n winst niet zien, in een netwerk zonder batch normalization op 6.000 kleine foto's. Wat de groepen hier wel geven: een breder midden voor dezelfde weights, 32 channels in 32 groepen tegen 16 in één.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een ResNeXt block is het residual block uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl) met de bottleneck uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl), in het midden twee keer zo breed gemaakt door een grouped convolution; zijn 32 paden naast elkaar zijn de takken uit [Inception: filters van verschillende groottes naast elkaar](/course/open/cv-inception?lang=nl), allemaal van één vorm. Je telde zijn weights met de shapes uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl), zette kleine convoluties naast elkaar met `torch.cat` uit [Tensors in PyTorch](/course/open/torch-tensors?lang=nl), en trainde het block met de Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) op de foto's uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl). Je project `cnn` heeft nu `resnext.py`. In de volgende les, *DenseNet: elke layer ziet alle layers ervoor*, telt een block zijn input niet op maar houdt hem: elke layer krijgt de maps van alle layers ervoor, gestapeld langs de channels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_groups, result_resnext, choice_runs):
    L.wrap_up(result_groups, result_resnext, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
