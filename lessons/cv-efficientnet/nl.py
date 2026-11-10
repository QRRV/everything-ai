# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-efficientnet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-efficientnet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Depthwise separable convolutions: MobileNet](/course/open/cv-mobilenet?lang=nl) en [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_separable():
    quick_separable = L.choice(
        "quick-separable",
        question="`SeparableConv(64, 64)` heeft 9 × 64 + 64 × 64 = 4.672 vermenigvuldigingen nodig voor elke pixel van zijn maps: 9 voor elk channel in zijn depthwise convolution, en 64 voor elk channel in zijn 1×1 convolution. Hoeveel heeft `SeparableConv(128, 128)` er nodig voor elke pixel?",
        options={
            "right": "17.536: bijna 4 keer zoveel, omdat de 1×1 convolution van 64 × 64 naar 128 × 128 gaat",
            "twice": "9.344: twee keer zoveel, omdat er twee keer zoveel channels zijn",
            "same": "4.672: zijn filters zijn nog steeds 3 bij 3 en 1 bij 1",
        },
        correct="right",
        why={
            "right": "9 × 128 + 128 × 128 = 1.152 + 16.384 = 17.536, 3,75 keer zoveel. De 1×1 convolution groeit met de channels die binnenkomen keer de channels die eruit gaan: twee keer zoveel van allebei is 4 keer zoveel. Vandaag tel je wat een breder netwerk kost, en een dieper, en een dat grotere foto's krijgt.",
            "twice": "De depthwise convolution verdubbelt, tot 9 × 128, maar de 1×1 convolution krijgt twee keer zoveel filters, elk twee keer zo diep: 128 × 128 = 16.384, 4 keer 64 × 64. Zie [MobileNet](/course/open/cv-mobilenet?lang=nl).",
            "same": "De filters houden hun grootte, maar het worden er meer, en elk 1×1-filter is zo diep als de channels die het krijgt. Zie [MobileNet](/course/open/cv-mobilenet?lang=nl).",
        },
    )
    quick_separable
    return (quick_separable,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Een block met een residual connection geeft `x + f(x)`, waarbij `f` de eigen layers van het block zijn. Wat moet `f(x)` teruggeven om dat te laten werken?",
        options={
            "right": "Evenveel channels als `x`, op maps van dezelfde grootte: `+` telt ze getal voor getal op",
            "half": "Half zoveel channels als `x`: het block maakt ze minder, zoals een bottleneck doet",
            "any": "Elk aantal channels: PyTorch telt op wat het kan",
        },
        correct="right",
        why={
            "right": "De som heeft de shape van `x`, dus moet `f(x)` die ook hebben. Vandaag is elk block een depthwise separable convolution van `width` channels naar `width`, waarvan de uitkomst bij `x` wordt opgeteld.",
            "half": "Dan telt `x + f(x)` maps van 32 channels op bij maps van 64, en stopt PyTorch. Een bottleneck maakt de channels minder in zijn midden, en aan zijn eind weer evenveel. Zie [ResNet](/course/open/cnn-resnet?lang=nl).",
            "any": "PyTorch stopt als de channels of de grootte van de maps niet passen: het block moet ze allebei houden. Zie [ResNet](/course/open/cnn-resnet?lang=nl).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een groter CNN leest meestal meer foto's goed, maar het moet ook meer rekenen. Als je computer meer aankan, waar steek je dat dan in? ResNet groeit van ResNet-18 tot ResNet-200 door er layers bij te zetten: dieper. Wide ResNets geven elke layer meer channels: breder. GPipe maakte een netwerk vier keer zo groot en gaf het foto's van 480 bij 480 pixels: scherper. In 2019 maten Mingxing Tan en Quoc Le bij Google wat elk van de drie oplevert voor wat het kost, en ze vonden dat alle drie tegelijk laten groeien, in een vaste verhouding, beter werkt dan er één laten groeien: **compound scaling**. Ze lieten een klein netwerk, EfficientNet-B0, stap voor stap groeien tot EfficientNet-B1 tot en met B7. EfficientNet-B7 had 84,3% van de foto's van ImageNet goed, evenveel als GPipe, met 66 miljoen parameters tegen 557 miljoen: 8,4 keer zo weinig. Vandaag tel je wat elke manier van groeien kost, bouw je een netwerk dat groeit, en test je op CIFAR-10 welke manier daar loont.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_grown():
    guess_grown = L.guess(
        "grown-imagenet",
        question="Tan en Le lieten EfficientNet-B0, dat 77,3% van de foto's van ImageNet goed had, groeien tot ongeveer 4,5 keer zijn vermenigvuldigingen, op vier manieren: alleen dieper, alleen breder, alleen scherpere foto's, of alle drie een beetje. Wat werkte het best?",
        options={
            "same": "Alle vier ongeveer even goed: dezelfde vermenigvuldigingen leveren ongeveer hetzelfde op",
            "sharper": "Alleen scherpere foto's, met afstand: grotere foto's laten meer zien",
            "right": "Alle drie een beetje, met ongeveer 2 punten voorsprong op elk van de andere",
        },
        reveal="**Alle drie een beetje: 81,1%.** Alleen dieper had 79,0% goed, alleen breder 78,9%, alleen scherper 79,1%, elk met 1,8 tot 1,9 miljard vermenigvuldigingen per foto, tegen 0,4 miljard voor B0. Alle drie samen hadden 81,1%, met 1,8 miljard (tabel 7 van de paper). Elke manier apart levert al snel steeds minder op; samen helpen ze elkaar.",
    )
    guess_grown
    return (guess_grown,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drie manieren van groeien.** Een CNN kan in drie richtingen groeien. Zijn **diepte** is hoeveel blocks het na elkaar heeft; zijn **breedte** is hoeveel channels elk block naast elkaar heeft; en de **resolutie** van zijn foto's is hoeveel pixels ze hebben, dus hoeveel rijen en kolommen elke map heeft. Een netwerk dat foto's met een hogere resolutie krijgt, noemen we hier scherper: het ziet fijnere details. Het plaatje: het basisnetwerk van de demo van vandaag, en vier manieren om het te laten groeien. Onderaan elk netwerk de foto die het krijgt, op schaal getekend: hoe meer pixels, hoe groter; erboven een balk voor elk block, zo breed als zijn channels. Eronder hoeveel keer de vermenigvuldigingen van het basisnetwerk het nodig heeft.
        """),
        C.networks_picture({"base": "basis", "deeper": "dieper", "wider": "breder", "sharper": "scherper", "compound": "alle drie",
                            "blocks": "{d} blocks", "channels": "{w} channels", "photo": "{s} × {s} pixels", "times": "× {t}",
                            "alt": "Vijf netwerken naast elkaar, elk een grijs vierkant voor zijn foto met erboven een balk voor elk block, zo breed als zijn channels. Basis: 2 blocks van 64 channels op foto's van 16 bij 16, × 1,00. Dieper: 9 blocks van 64 channels, 16 bij 16, × 3,95. Breder: 2 blocks van 136 channels, 16 bij 16, × 3,89. Scherper: 2 blocks van 64 channels op foto's van 32 bij 32, × 4,00. Alle drie: 3 blocks van 80 channels op foto's van 22 bij 22, × 4,01."}, L.lang),
        mo.md(r"""
        **Wat elke manier kost.** Een convolutie heeft één vermenigvuldiging nodig voor elke weight van een filter, voor elk getal dat hij geeft. Twee keer zoveel blocks: twee keer zoveel vermenigvuldigingen. Twee keer zoveel channels: elk filter ziet twee keer zoveel channels, en er zijn twee keer zoveel filters, dus 4 keer zoveel. Foto's die twee keer zo breed en twee keer zo hoog zijn: elke map heeft 4 keer zoveel pixels, dus ook 4 keer zoveel. De paper schrijft het als een regel: de vermenigvuldigingen groeien met

        $$d \cdot w^2 \cdot r^2,$$

        waarbij $d$, $w$ en $r$ zeggen hoeveel keer zo diep, zo breed en zo scherp het netwerk is geworden; $r$ vermenigvuldigt zowel de hoogte als de breedte van de foto, dus groeien zijn pixels met $r^2$. De parameters groeien anders: met de diepte, en met het kwadraat van de breedte, maar niet met de resolutie. Op een grotere foto schuiven dezelfde filters over meer pixels.

        **Compound scaling.** Tan en Le laten alle drie tegelijk groeien, met één getal $\varphi$ (phi) dat zegt hoeveel:

        $$d = \alpha^\varphi, \qquad w = \beta^\varphi, \qquad r = \gamma^\varphi, \qquad \text{met } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2.$$

        Volgens de regel groeien de vermenigvuldigingen dan met $(\alpha \cdot \beta^2 \cdot \gamma^2)^\varphi$: elke stap van $\varphi$ verdubbelt ze ongeveer. Met $\varphi = 1$ probeerden ze een paar waarden van $\alpha$, $\beta$ en $\gamma$ uit op EfficientNet-B0, en hielden ze de beste: $\alpha = 1{,}2$, $\beta = 1{,}1$ en $\gamma = 1{,}15$, met $\alpha \cdot \beta^2 \cdot \gamma^2 = 1{,}92$. Daarna hielden ze die drie vast en namen ze een grotere $\varphi$ voor elk van EfficientNet-B1 tot en met B7. Waarom alle drie tegelijk? Een grotere foto heeft meer layers nodig, zodat een filter laat in het netwerk er nog genoeg van ziet, zijn receptive field; en meer channels, voor de fijnere patronen die de extra pixels laten zien.

        **EfficientNet-B0** zelf is gevonden door een computer die zocht tussen veel kleine netwerken, gebouwd uit depthwise convolutions en 1×1 convolutions zoals in MobileNet, naar het netwerk dat de meeste foto's goed leest voor zijn vermenigvuldigingen. De paper telt in FLOPS, een vermenigvuldiging met haar optelling als één geteld; hier zijn dat de vermenigvuldigingen. Tabel 7 laat B0 op elk van de vier manieren groeien tot ongeveer 4,5 keer zijn vermenigvuldigingen:

        | EfficientNet-B0 | Vermenigvuldigingen per foto | Goed op ImageNet |
        |---|---|---|
        | zoals het is | 0,4 miljard | 77,3% |
        | dieper, $d = 4$ | 1,8 miljard | 79,0% |
        | breder, $w = 2$ | 1,8 miljard | 78,9% |
        | scherper, $r = 2$ | 1,9 miljard | 79,1% |
        | alle drie, $d = 1{,}4$, $w = 1{,}2$, $r = 1{,}3$ | 1,8 miljard | 81,1% |

        Hetzelfde geldt voor netwerken die er niet voor gemaakt zijn; tabel 3, elk gegroeid tot 3,6 tot 4,3 keer zijn vermenigvuldigingen:

        | Netwerk | Zoals het is | Dieper | Breder | Scherper | Alle drie |
        |---|---|---|---|---|---|
        | MobileNet | 70,6% | — | 74,2% | 72,7% | 75,6% |
        | MobileNetV2 | 72,0% | 76,8% | 76,4% | 74,8% | 77,4% |
        | ResNet-50 | 76,0% | 78,1% | 77,7% | 77,5% | 78,8% |

        En de familie, uit tabel 2, naast netwerken die ongeveer evenveel foto's goed hebben. B0 heeft daar 77,1%, iets minder dan in tabel 7:

        | Netwerk | Goed op ImageNet | Parameters | Vermenigvuldigingen per foto |
        |---|---|---|---|
        | EfficientNet-B0 | 77,1% | 5,3 miljoen | 0,39 miljard |
        | ResNet-50 | 76,0% | 26 miljoen | 4,1 miljard |
        | EfficientNet-B1 | 79,1% | 7,8 miljoen | 0,70 miljard |
        | ResNet-152 | 77,8% | 60 miljoen | 11 miljard |
        | EfficientNet-B4 | 82,9% | 19 miljoen | 4,2 miljard |
        | EfficientNet-B7 | 84,3% | 66 miljoen | 37 miljard |
        | GPipe | 84,3% | 557 miljoen | niet in de tabel |
        """),
        L.watch_out("De regel $d \\cdot w^2 \\cdot r^2$ klopt ongeveer, niet precies. De paper gebruikt hem omdat gewone convoluties, de 1×1 convolutions daaronder, het meeste werk doen in een CNN; maar layers die op een andere manier groeien, tellen ook mee, zoals de eerste convolutie en de depthwise convolutions, en afronden ook, zoals het uitgewerkte voorbeeld en de schuifregelaar laten zien. Hoeveel compound scaling oplevert, hangt ook af van het netwerk waarmee je begint, zegt de paper: daarom zochten Tan en Le eerst naar B0. En een hogere resolutie helpt alleen als de foto's de details hebben. De foto's van CIFAR-10 zijn 32 bij 32, dus krijgt het basisnetwerk van de demo ze verkleind tot 16 bij 16, en krijgen de scherpere netwerken weer meer pixels, tot 32 bij 32."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **diepte**, **breedte**, **resolutie** | hoeveel blocks een netwerk na elkaar heeft, hoeveel channels elk heeft, en hoeveel pixels zijn foto's hebben | `depth`, `width`, `side` |
        | **compound scaling** | een netwerk tegelijk laten groeien in diepte, breedte en resolutie, elk met een vaste factor tot de macht $\varphi$ | — |
        | **EfficientNet** | een familie van CNN's uit 2019: B0, gevonden met een zoektocht door een computer, en B1 tot en met B7, daaruit gegroeid met compound scaling | — |
        | $d$, $w$, $r$ | hoeveel keer zo diep, zo breed en zo scherp een gegroeid netwerk is als het netwerk waaruit het groeide | — |
        | $\varphi$; $\alpha$, $\beta$, $\gamma$ | het ene getal dat zegt hoeveel er gegroeid wordt; en de factoren voor diepte, breedte en resolutie in elke stap daarvan | `phi` |

        **Het netwerk van de demo.** `ScaledCNN(depth, width)` begint met een 3×3-convolutie met stride 2 van de 3 kleuren naar `width` channels, die de maps half zo breed en hoog maakt. Dan komen `depth` blocks, elk je depthwise separable convolution uit [MobileNet](/course/open/cv-mobilenet?lang=nl), `SeparableConv(width, width)`, met een residual connection zoals in [ResNet](/course/open/cnn-resnet?lang=nl). Aan het eind neemt het het gemiddelde van elke map, `width` getallen hoe groot de foto ook is, en een linear layer maakt daar de 10 logits van. Het basisnetwerk heeft 2 blocks van 64 channels, en krijgt de foto's van CIFAR-10 die de course 16 bij 16 heeft gemaakt, met de interpolatie uit [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl). Het heeft 12.042 parameters, en 709.248 vermenigvuldigingen per foto nodig. De vier gegroeide netwerken hebben er elk ongeveer 4 keer zoveel nodig. Dieper: 9 blocks. Breder: 136 channels. Scherper: de foto's van CIFAR-10 zoals ze zijn, 32 bij 32. Alle drie: de $\alpha$, $\beta$ en $\gamma$ uit de paper met $\varphi = 2$, afgerond: 3 blocks, 80 channels en foto's van 22 bij 22. Volgens de regel zijn 4 keer zoveel blocks er 8, en twee keer zoveel channels 128; precies geteld kosten die maar 3,53 en 3,48 keer zoveel, dus neemt de course er 9 en 136. Het uitgewerkte voorbeeld laat zien waarom.

        Het plaatje: hoeveel van de eerste 1.000 testfoto's van CIFAR-10 elk netwerk goed leest na 20 epochs op de eerste 6.000 trainingsfoto's, een stip voor elk van 5 seeds, met de code van de course. Het basisnetwerk: 506 tot 524. Dieper: 513 tot 536. Breder: 536 tot 555. Scherper: 514 tot 558. Alle drie samen: 543 tot 571. Oefening 3 vraagt wat de rijen zeggen.
        """),
        mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                "base": "basis: 2 blocks, 64 channels, 16 × 16", "deeper": "dieper: 9 blocks",
                                "wider": "breder: 136 channels", "sharper": "scherper: foto's van 32 × 32",
                                "compound": "alle drie: 3 blocks, 80 channels, 22 × 22", "mine": "jouw run", "course": "de runs van de course",
                                "alt": "Vijf rijen open stippen, testfoto's goed gelezen van 1.000, een stip voor elke seed en een kort streepje bij hun gemiddelde. Basisnetwerk: 506 tot 524. Dieper: 513 tot 536. Breder: 536 tot 555. Scherper: 514 tot 558. Alle drie samen: 543 tot 571."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een netwerk te bouwen dat groeit in diepte en breedte:

    1. Maak in `__init__` `depth` blocks `SeparableConv(width, width)`, een nieuw block voor elke plek, en bewaar ze in een `nn.ModuleList`, zodat PyTorch hun parameters ziet.
    2. Loop in `forward`, na de eerste convolutie en zijn ReLU, op volgorde door de blocks, en tel wat elk block geeft bij `x` op: `x = x + block(x)`.
    3. Het gemiddelde van elke map en de linear layer daarna maken de logits; die staan al in het vak.

    De resolutie zit niet in het netwerk: dat is de grootte van de foto's die je het geeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel telt de vermenigvuldigingen van `ScaledCNN(depth, width)` voor één foto van `side` bij `side` pixels, layer voor layer, zoals het netwerk van de demo ze heeft. Hij print die van het basisnetwerk, die van een gegroeid netwerk, en hoeveel keer zoveel dat is. Voer de cel uit zoals hij is, verander hem dan waar het commentaar het zegt, en voer hem opnieuw uit. Hij heeft geen download nodig.
    """)
    return


@app.cell
def example_cost():
    def multiplications(depth, width, side):
        half = side // 2  # de eerste convolutie heeft stride 2 (bij een even zijde)
        first = 3 * 3 * 3 * width * half * half
        # een block: 9 voor elk channel, dan width voor elk channel, bij elke pixel
        block = (3 * 3 * width + width * width) * half * half
        last = width * 10  # de linear layer, van de gemiddelden van de maps
        return first + depth * block + last

    base = multiplications(2, 64, 16)
    depth, width, side = 8, 64, 16  # verander: depth, width, side = 2, 128, 16
    grown = multiplications(depth, width, side)
    print(base, grown, grown / base)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, print de cel `709248 2503296 3.529507309149973`: 4 keer zoveel blocks kosten maar 3,53 keer zoveel. De blocks groeien wel 4 keer, van 598.016 naar 2.392.064, maar de eerste convolutie, 110.592, en de linear layer, 640, blijven zoals ze zijn. Met `depth, width, side = 2, 128, 16` print hij `709248 2467072 3.4784334957588885`: twee keer zoveel channels kosten 3,48 keer zoveel. De 1×1 convolutions in de blocks groeien 4 keer, maar de depthwise convolutions en de eerste convolutie maar 2 keer, omdat elk van hun filters één channel ziet, of de 3 kleuren. Probeer ook `2, 64, 32`: dan krijgt elke layer behalve de linear layer 4 keer zoveel pixels, en print de cel `709248 2835072 3.9972929074174335`, bijna precies 4.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Schuif $\varphi$ van 0 tot 5: het basisnetwerk, gegroeid met compound scaling en de $\alpha$, $\beta$ en $\gamma$ uit de paper, afgerond zoals in de demo: de diepte naar boven op hele blocks, de breedte op een veelvoud van 8, de zijde op een even getal. De balken zijn de vermenigvuldigingen, geteld, als veelvoud van die van het basisnetwerk, op een logaritmische schaal, waarop elke verdubbeling een even hoge stap is; donker de $\varphi$ die je koos. De gestreepte lijn is wat de regel zegt: 1,92 keer zoveel voor elke stap, dus een rechte lijn.
    """)
    return


@app.cell(hide_code=True)
def play_phi():
    _labels = {"title": "Het basisnetwerk gegroeid met compound scaling", "x": "φ", "y": "keer zoveel als de basis",
               "rule": "de regel: 1,92 per stap",
               "alt": "Zes balken, de vermenigvuldigingen van het basisnetwerk gegroeid met compound scaling voor φ van 0 tot 5, als veelvoud van die van het basisnetwerk, geteld, op een logaritmische schaal, met de balk van de gekozen φ donker, en een gestreepte lijn met open stippen voor wat de regel zegt, 1,92 tot de macht φ."}
    explore_phi = L.explore("phi-slider", "φ", lambda _v: (C.family_picture(int(_v), _labels, L.lang), C.family_sentence(int(_v), L.lang)),
                            start=2, values=C.PHIS, role="hyper",
                            task="Schuif φ van 0 tot 5. Bij welke φ komt het getelde aantal voor het eerst onder wat de regel zegt?")
    explore_phi
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij $\varphi = 5$: 22,22 keer de vermenigvuldigingen van het basisnetwerk, waar de regel 26,11 zegt. Twee dingen werken elkaar tegen. Het afronden, vooral dat van de diepte, maakt het netwerk groter dan de regel zegt: bij $\varphi = 1$ heeft het al 3 blocks, waar de regel er 2,4 heeft. En de eerste convolutie en de depthwise convolutions groeien maar met de breedte, niet met het kwadraat ervan, en dat maakt het netwerk kleiner, des te meer naarmate het breder wordt. Van $\varphi = 1$ tot en met 4 wint het afronden; bij $\varphi = 5$ wordt de diepte van $2 \cdot 1{,}2^5 = 4{,}98$ afgerond op 5 blocks, nauwelijks omhoog, en wint de breedte. Daar zijn de foto's ook zo groot als CIFAR-10 ze heeft: 32 bij 32. Een grotere $\varphi$ zou foto's nodig hebben die CIFAR-10 niet heeft. Elke stap verdubbelt de kosten ongeveer, zoals de regel zegt, maar nooit precies.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_costs():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat elke keuze kost

    Acht manieren om een netwerk te laten groeien. Welke geven het ongeveer 2 keer zoveel vermenigvuldigingen, volgens de regel $d \cdot w^2 \cdot r^2$, en welke ongeveer 4 keer zoveel? Deel ze allemaal in.
    """)
    return


@app.cell(hide_code=True)
def exercise_costs():
    sort_costs = L.sort("costs")
    sort_costs
    return (sort_costs,)


@app.cell(hide_code=True)
def check_costs(sort_costs):
    result_costs = L.check("costs", sort_costs)
    result_costs
    return (result_costs,)


@app.cell(hide_code=True)
def hints_costs():
    L.hints(
        "costs",
        [
            "De diepte telt één keer, de breedte en de resolutie twee keer: $d \\cdot w^2 \\cdot r^2$. Wat is 2 in het kwadraat, en wat is 1,41 in het kwadraat?",
            "In elke groep horen er vier. Eén stap van $\\varphi$ vermenigvuldigt met $\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 = 1{,}92$, twee stappen met $1{,}92^2 = 3{,}69$.",
        ],
        why="**Ongeveer 2 keer:** diepte × 2, breedte × 1,41, resolutie × 1,41, en $\\varphi$ één hoger (1,92). **Ongeveer 4 keer:** breedte × 2, resolutie × 2, diepte × 4, en $\\varphi$ twee hoger (3,69). Breedte en resolutie tellen in het kwadraat, de diepte één keer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scaled():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een netwerk dat groeit

        Maak `ScaledCNN` af. Zijn argumenten zijn het aantal blocks, `depth`, en de channels van elk, `width`. De eerste convolutie, de linear layer en het eind van `forward` staan in het vak. Maak de blocks: `depth` keer je `SeparableConv` van `width` channels naar `width`, in een `nn.ModuleList`. Stuur dan in `forward` `x` om de beurt door elk block, met een residual connection. De check bouwt je netwerk met 2 blocks van 64 channels en met andere getallen, en vergelijkt wat eruit komt, hoeveel blocks en parameters het heeft, en of elke parameter een gradient krijgt, met die van de course. Als het klopt, bewaart hij je class in je project, als `work/projects/cnn/scaled.py`.
        """),
        L.task("Maak de blocks in `__init__`, en de loop erover in `forward`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scaled():
    box_scaled = L.code_box("scaled", lines=20)
    box_scaled
    return (box_scaled,)


@app.cell(hide_code=True)
def check_scaled(box_scaled):
    result_scaled = L.check("scaled", box_scaled.value)
    result_scaled
    return (result_scaled,)


@app.cell(hide_code=True)
def hints_scaled():
    L.hints(
        "scaled",
        [
            "Stap 1 van het recept: een lijst van `depth` nieuwe blocks, één voor elke plek. Hoe maak je zo'n lijst in één regel, en wat moet eromheen, zodat PyTorch de blocks ziet?",
            "`self.blocks = nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])`. In `forward`: `for block in self.blocks:` en, daaronder ingesprongen, stap 2 van het recept.",
        ],
        why="De blocks zijn `nn.ModuleList([SeparableConv(width, width) for _ in range(depth)])`, en `forward` loopt erdoor met `for block in self.blocks:` en `x = x + block(x)`. De meeste blocks van EfficientNet maken de channels eerst 6 keer zoveel met een 1×1 convolution, vóór hun depthwise convolution, en ze gebruiken batch normalization; de diepte, breedte en resolutie van EfficientNet groeien net als hier.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint je `ScaledCNN` als het netwerk dat je kiest op de eerste 6.000 trainingsfoto's van CIFAR-10, zo groot gemaakt als dat netwerk nodig heeft, met je Trainer, {C.EPOCHS} epochs lang met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test het op de eerste 1.000 testfoto's. Zijn blocks zijn je `SeparableConv`. Waar een module van jou uit een eerdere les ontbreekt of kapot is, springt die van de course in. Het heeft CIFAR-10 en oefening 2 nodig, en duurt een paar minuten. Tijdens het trainen toont een grafiek de validation loss na elke epoch. In het plaatje daarna is de grijze lijn de validation loss van het basisnetwerk van de course met seed 0. Daarna kun je terugspoelen door je run, en zie je je run naast die van de course.
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
    variant_choice = mo.ui.radio(options={"alle drie samen: 3 blocks, 80 channels, 22 × 22": "compound",
                                          "dieper: 9 blocks": "deeper", "breder: 136 channels": "wider",
                                          "scherper: foto's van 32 × 32": "sharper",
                                          "het basisnetwerk: 2 blocks, 64 channels, 16 × 16": "base"},
                                 value="alle drie samen: 3 blocks, 80 channels, 22 × 22", label="Het netwerk")
    train_button = mo.ui.run_button(label="Train het netwerk")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_scaled(cifar, train_button, variant_choice, result_scaled):
    cifar, result_scaled  # opnieuw uitvoeren als CIFAR-10 binnenkomt, of als oefening 2 je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het netwerk te trainen.*")
    else:
        _variant = variant_choice.value
        _names = {"base": "Basis", "deeper": "Dieper", "wider": "Breder", "sharper": "Scherper", "compound": "Alle drie"}
        _with = {"base": "Het basisnetwerk", "deeper": "Het diepere netwerk, met 9 blocks,", "wider": "Het bredere netwerk, met 136 channels,",
                 "sharper": "Het scherpere netwerk, op foto's van 32 bij 32,", "compound": "Het netwerk dat op alle drie manieren gegroeid is,"}
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je netwerk op CIFAR-10", "x": "epoch", "y": "validation loss",
                                        "alt": "De validation loss van het netwerk op de 1.000 testfoto's na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**{_names[_variant]}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2: dan staat je `ScaledCNN` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop hierboven, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `ScaledCNN`, je `SeparableConv` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "scaled": "*Je `ScaledCNN` in `work/projects/cnn/scaled.py`, of de `SeparableConv` die hij gebruikt in `work/projects/cnn/separable.py`, stopt met een fout, of geeft niet terug wat hij moet geven: 10 logits voor elke foto. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `channel_stats` in `work/projects/cnn/channels.py` stopt met een fout, of geeft niet wat hij moet geven. Heb je zijn bestand veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "base": "basisnetwerk",
                       "alt": "De loss van het netwerk na elke epoch: op de trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van het basisnetwerk van de course met seed 0, die bij de laatste epoch gedaald is tot 1,363."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.guesses_picture(_guesses[_e - 1], _result["truth"], L.lang),
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
                mo.md(f"""{_with[_variant]} leest **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters en {C.grouped(_result['mults'], L.lang)} vermenigvuldigingen per foto.{_turn} Over 5 seeds, met de code van de course, dit netwerk: {C.grouped(_tr[0], L.lang)} tot {C.grouped(_tr[1], L.lang)} trainingsfoto's en {_te[0]} tot {_te[1]} testfoto's, met de laagste validation loss na epoch {_be[0]} tot {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Testfoto's goed gelezen na 20 epochs", "x": "goed gelezen, van 1.000",
                                        "base": "basis: 2 blocks, 64 channels, 16 × 16", "deeper": "dieper: 9 blocks",
                                        "wider": "breder: 136 channels", "sharper": "scherper: foto's van 32 × 32",
                                        "compound": "alle drie: 3 blocks, 80 channels, 22 × 22", "mine": "jouw run", "course": "de runs van de course",
                                        "alt": "Vijf rijen testfoto's goed gelezen, van 1.000: open stippen voor de runs van de course over 5 seeds, voor het basisnetwerk, dieper, breder, scherper en alle drie samen, met een kort streepje bij het gemiddelde van elke rij; en in de rij van jouw netwerk een dichte stip voor jouw run."}, L.lang, mine={"variant": _variant, "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_runs():
    mo.md(r"""
    ### Oefening 3. Welke manier hier loont
    """)
    return


@app.cell(hide_code=True)
def choice_runs_cell():
    choice_runs = L.choice(
        "scaling-runs",
        question="Over 5 seeds, na 20 epochs, testfoto's goed gelezen van 1.000. Het basisnetwerk leest er 506 tot 524 goed. Gegroeid tot ongeveer 4 keer zijn vermenigvuldigingen: dieper 513 tot 536; breder 536 tot 555; scherper 514 tot 558; alle drie samen 543 tot 571. Wat zeggen de runs?",
        options={
            "deeper": "Dieper is het best: het leert de meeste van zijn trainingsfoto's, 4.802 tot 4.979 van de 6.000",
            "sharper": "Scherper is het best: zijn beste run, 558, laat zien dat grotere foto's zijn wat dit netwerk mist",
            "right": "Alle drie samen lezen er de meeste goed: gemiddeld 558, tegen 543 voor breder, 534 voor scherper en 525 voor dieper, al overlappen zijn runs met die van breder en scherper; dieper verschilt nauwelijks van het basisnetwerk",
        },
        correct="right",
        why={
            "deeper": "Dieper leert zijn trainingsfoto's het best, 4.802 tot 4.979, maar leest van de vier gegroeide netwerken de minste testfoto's goed, 513 tot 536: zijn validation loss is het laagst na epoch 10 tot 13, en stijgt daarna. Het leert zijn trainingsfoto's uit het hoofd, overfitting; bij de andere is de validation loss het laagst na epoch 15 tot 20.",
            "sharper": "Eén run is niet de hele rij: scherper leest er 514 tot 558, de grootste spreiding, gemiddeld 534, onder breder en alle drie samen. Het heeft niet meer parameters dan het basisnetwerk, 12.042, en leert niet meer van zijn trainingsfoto's: 3.592 tot 3.720, tegen 3.590 tot 3.686. En met 2 blocks ziet een getal van de laatste maps, vóór het gemiddelde, 11 bij 11 pixels van een foto van 32 bij 32. Grotere foto's hebben meer layers en meer channels nodig, zegt de paper, en daarom laat compound scaling ze samen groeien; deze runs testen dat niet.",
            "right": "Alle drie samen lezen er 543 tot 571 goed, gemiddeld 558,4; breder 542,6, scherper 533,8, dieper 524,8, en het basisnetwerk 517,2. Elk gegroeid netwerk heeft ongeveer 4 keer de vermenigvuldigingen van het basisnetwerk nodig. Net als in de paper kwamen alle drie samen als eerste: daar met 81,1% goed, tegen 78,9 tot 79,1% voor één manier. Maar met 5 seeds is niet elk verschil zeker: zijn laagste run, 543, ligt nog onder de beste van breder, 555, en van scherper, 558, en dieper, 513 tot 536, verschilt nauwelijks van het basisnetwerk, 506 tot 524. Een uitkomst geldt waar hij gemeten is: op 6.000 foto's van CIFAR-10, met dit netwerk.",
        },
    )
    choice_runs
    return (choice_runs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De blocks van je netwerk zijn de depthwise separable convolutions uit [Depthwise separable convolutions: MobileNet](/course/open/cv-mobilenet?lang=nl), elk met de residual connection uit [ResNet: diepte die blijft werken](/course/open/cnn-resnet?lang=nl). Je telde hun vermenigvuldigingen met de shapes uit [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl), maakte de foto's kleiner met de interpolatie uit [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl), en trainde met de Trainer uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) op de foto's uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl). Je project `cnn` heeft nu `scaled.py`. De volgende les, *Saliency: de pixels die het antwoord veranderen*, stelt een CNN een andere vraag: niet hoe groot het moet zijn, maar welke pixels van een foto het zijn antwoord lieten kiezen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_costs, result_scaled, choice_runs):
    L.wrap_up(result_costs, result_scaled, choice_runs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
