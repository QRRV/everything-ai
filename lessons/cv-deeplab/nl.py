# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-deeplab. Route: course/route/lessen/beeld.yaml. Checks: checks.py.
# termen:allow dilation, dilations

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-deeplab", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stride():
    quick_stride = L.choice(
        "quick-stride",
        question="Het FCN van de course met output stride 16 heeft vier blokken, elk met een pooling van 2. Hoe groot is zijn laatste map voor een scène van 128 bij 128 pixels, en voor hoeveel pixels van de scène staat één plaats ervan?",
        options={
            "right": "8 bij 8 plaatsen, elk voor 16 bij 16 pixels",
            "eight": "16 bij 16 plaatsen, elk voor 8 bij 8 pixels",
            "four": "32 bij 32 plaatsen, elk voor 4 bij 4 pixels",
            "same": "128 bij 128 plaatsen: padding houdt de map zo groot als de scène",
        },
        correct="right",
        why={
            "right": "Elke pooling halveert de map: 128, 64, 32, 16, 8. Er blijven dus 8 bij 8 plaatsen over, en elk staat voor 128 / 8 = 16 bij 16 pixels: een output stride van 16. Zijn class maps waren grof, en na 8 epochs bleef zijn IoU onder die van stride 8. Vandaag haal je de laatste twee van die poolings eruit, en blijft de map 32 bij 32.",
            "eight": "Dat is het FCN met stride 8, met drie poolings. Een vierde halveert de map nog een keer: 8 bij 8 plaatsen, elk voor 16 bij 16 pixels.",
            "four": "Dat is een map na twee poolings. Vier poolings halveren de map vier keer: 128 / 16 = 8, dus 8 bij 8 plaatsen.",
            "same": "Padding houdt de map bij een convolutie zo groot als hij was, maar elke pooling van 2 halveert hem: vier poolings maken van 128 er 8.",
        },
    )
    quick_stride
    return (quick_stride,)


@app.cell(hide_code=True)
def quick_check_up():
    quick_up = L.choice(
        "quick-up",
        question="De upsampler uit de vorige les met stride 4 is `nn.ConvTranspose2d(4, 4, 8, stride=4, padding=2)`. Hij krijgt 4 maps van 32 bij 32. Hoe groot zijn de maps die hij geeft?",
        options={
            "right": "128 bij 128: (32 − 1) · 4 − 2 · 2 + 8",
            "no-padding": "132 bij 132: (32 − 1) · 4 + 8",
            "times": "136 bij 136: 32 · 4 + 8",
            "double": "64 bij 64: een transposed convolution maakt een map twee keer zo groot",
        },
        correct="right",
        why={
            "right": "Met $H$ plaatsen, stride $s$, padding $p$ en een kernel van $k$ heeft de output $(H - 1) \\cdot s - 2p + k = 31 \\cdot 4 - 4 + 8 = 128$ plaatsen: 4 keer zo groot. Het netwerk van vandaag eindigt met deze upsampler, op een map van 32 bij 32.",
            "no-padding": "Padding 2 haalt aan elk eind 2 plaatsen weg: $(H - 1) \\cdot s - 2p + k = 124 - 4 + 8 = 128$.",
            "times": "De laatste van de 32 waarden schildert zijn kopie vanaf plaats $(32 - 1) \\cdot 4 = 124$, niet vanaf $32 \\cdot 4$; en padding 2 haalt aan elk eind 2 plaatsen weg: 128.",
            "double": "Alleen met stride 2. Met stride 4 schildert elke waarde zijn kopie 4 plaatsen verder: ongeveer 4 keer zo groot, $(32 - 1) \\cdot 4 - 2 \\cdot 2 + 8 = 128$. Zie [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl).",
        },
    )
    quick_up
    return (quick_up,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een CNN dat foto's herkent, poolt keer op keer: zijn laatste map is 32 keer kleiner dan de foto, en elke plaats ervan ziet een groot deel van de foto. Een FCN dat van zo'n netwerk gemaakt is, maakt grove class maps, zoals je in de vorige les zag met stride 16. Minder poolings zouden een fijnere map geven, maar dan zien de layers erna minder van de foto dan waar ze naar leerden kijken. In 2015 vonden Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy en Alan Yuille een uitweg voor een VGG-16 dat op ImageNet getraind was: ze haalden zijn laatste twee poolings eruit, en spreidden de kernels van de layers daarna uit, met gaten ertussen. Elke layer zag nog steeds wat hij had leren zien, en de laatste map werd 8 keer kleiner dan de foto in plaats van 32. Hun **DeepLab** zette later een paar van die uitgespreide convoluties naast elkaar, om tegelijk een kleine en een grote omgeving te zien, en haalde een mean IoU van 79,7% op de testfoto's van PASCAL VOC 2012, waar het FCN-8s uit de vorige les 62,2% had. Vandaag bouw je een klein DeepLab uit de layers van het FCN van de course, en train je het op de scènes van de course.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_field():
    guess_field = L.guess(
        "dilated-field",
        question="Een 3×3-convolutie heeft zijn 9 taps naast elkaar, en ziet 3 bij 3 pixels. Spreid zijn taps uit tot ze 2 pixels uit elkaar liggen, met één pixel ertussen, en hij ziet 5 bij 5 pixels, met dezelfde 9 weights. Stapel drie 3×3-convoluties, de eerste met zijn taps 1 pixel uit elkaar, de tweede 2 en de derde 4. Hoe breed is het deel van het beeld dat één plaats na de derde ziet?",
        options={
            "seven": "7 pixels: elke 3×3-convolutie telt er 2 bij op",
            "nine": "9 pixels: de breedste van de drie",
            "right": "15 pixels",
            "twentyseven": "27 pixels: 3 · 3 · 3",
        },
        reveal="15: de eerste telt 2 op bij de 1 pixel van een plaats, de tweede 2 · 2 = 4, de derde 2 · 4 = 8, en 1 + 2 + 4 + 8 = 15. Drie gewone 3×3-convoluties zien er maar 7. De tweede cel van het uitgewerkte voorbeeld meet het, en oefening 1 telt het voor elke stapel.",
    )
    guess_field
    return (guess_field,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Gaten tussen de taps.** Een 3×3-convolutie vermenigvuldigt 9 pixels naast elkaar, zijn **taps**, met zijn 9 weights, en telt ze op. Een **dilated convolution** spreidt die taps uit: met **dilation** 2 liggen ze 2 pixels uit elkaar, en ziet hij 5 bij 5 pixels; met dilation 4 liggen ze 4 uit elkaar, en ziet hij 9 bij 9. Hij heeft nog steeds 9 weights, en slaat de pixels tussen zijn taps over. Een kernel van $k$ met dilation $d$ beslaat $k + (k - 1)(d - 1)$ pixels, voor een 3×3-kernel $2d + 1$. Hij doet hetzelfde als een 5×5-kernel met nullen tussen zijn taps, zoals de eerste cel van het uitgewerkte voorbeeld laat zien, maar vermenigvuldigt alleen de 9 taps. In PyTorch is het `nn.Conv2d(c_in, c, 3, padding=d, dilation=d)`: padding $d$ houdt de map zo groot als hij was. De DeepLab-papers noemen het een atrous convolution, naar het Franse *à trous*, met gaten. <!-- termen:ok -->

        **Gestapeld.** Een layer met dilation $d$ maakt het deel van het beeld dat één plaats ziet $(k - 1) \cdot d$ breder, keer de stap tussen buren van zijn input, zoals in [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl). Drie 3×3-convoluties met dilation 1, 2 en 4 zien 15 pixels, zoals in de gok; met nog een van dilation 8 erbij 31. Is elke dilation twee keer de vorige, dan verdubbelt wat één plaats ziet ongeveer met elke layer, terwijl elke layer er evenveel weights bij doet. Een pooling van 2 laat de layers erna ook twee keer zo ver zien, maar halveert de map.

        **Een pooling eruit.** Dat doet DeepLab met een CNN dat al geleerd heeft. Het zet de stride van een pooling op 1: de pooling neemt nog steeds de grootste van elke 2 bij 2 waarden, maar schuift 1 plaats per keer op, dus blijft de map twee keer zo groot. Elke layer daarna krijgt dilation 2. De layer die na de pooling kwam, krijgt nu een map waarvan de buren twee keer zo dicht bij elkaar liggen; met zijn taps 2 plaatsen uit elkaar bereikt hij precies de waarden die hij eerst bereikte. Op elke tweede plaats van de grotere map geeft hij dezelfde getallen als eerst, zoals de derde cel van het uitgewerkte voorbeeld laat zien, en daartussen geeft hij nieuwe. De layers na een tweede pooling die eruit is, krijgen dilation 4. De getrainde weights blijven werken, en de class map wordt fijner.

        **Atrous spatial pyramid pooling.** Voor sommige vormen is een kleine omgeving genoeg om de soort te zien, voor andere is een grote nodig. **ASPP**, atrous spatial pyramid pooling, zet een paar convoluties naast elkaar op dezelfde map: een 1×1-convolutie, en een 3×3-convolutie voor elke dilation, in die van de course met dilation 2, 4 en 8. Elke tak geeft `c` channels, gevolgd door ReLU. `torch.cat(maps, dim=1)` stapelt hun maps langs de channels, en een laatste 1×1-convolutie met ReLU maakt van die 4 · `c` channels weer `c`: die leert hoe hij combineert wat elke tak zag. De piramide is de verschillende groottes van de omgeving, zoals een beeld op een paar schalen; ondanks de naam poolt de ASPP van de course niets.

        **Het DeepLab van de course.** Het heeft de layers van het FCN van de course met stride 16: 3×3-convoluties met 16, 32, 64 en 64 channels. Het leert vanaf nul, dus het heeft geen getrainde weights te bewaren, en laat de derde en de vierde pooling helemaal weg: alleen na de eerste twee convoluties komt een pooling van 2, en zijn laatste map is 32 bij 32, 4 keer kleiner dan de scène. conv4 kwam na de derde pooling, dus krijgt hij dilation 2; het eind van het FCN, nog een 3×3-convolutie, kwam na de vierde, dus zou hij dilation 4 krijgen: daar zet de course een ASPP, met zijn dilations 2, 4 en 8 rond die 4. Dan een 1×1-convolutie naar een score voor elke klasse, en je upsampler uit de vorige les met stride 4. Oefening 3 bouwt het.

        **DeepLab in de papers.** Het eerste DeepLab, van 2015, begint bij een VGG-16 dat op ImageNet getraind is. Het zet de stride van de laatste twee van zijn vijf poolings op 1 en geeft de layers daarna dilation 2 en 4, zodat zijn laatste map 8 keer kleiner is dan de foto in plaats van 32. Het maakt zijn scores 8 keer zo groot met bilineaire interpolatie, die niets leert, en een aparte stap na het netwerk laat daarna de randen van de klassen de randen in de foto volgen. DeepLab v2, van 2016, voegt ASPP toe: vier takken naast elkaar, die elk beginnen met een 3×3-convolutie met dilation 6, 12, 18 of 24, en waarvan de scores worden opgeteld. Met een ResNet-101 en nog een paar stappen haalde het de 79,7% van *Waarom*. DeepLab v3, van 2017, heeft de ASPP waar die van de course op lijkt: een 1×1-convolutie en drie 3×3-convoluties met dilation 6, 12 en 18, elk met 256 channels en batch normalization, gestapeld. De makers zagen dat een 3×3-convolutie met een dilation die bijna zo groot is als de map, buiten zijn middelste tap vooral padding bereikt, en werkt als een 1×1-convolutie; daarom voegden ze een tak toe die de hele map middelt, voor wat de foto als geheel laat zien. Zonder de aparte stap aan het eind haalde v3 85,7%. De ASPP van de course heeft geen batch normalization en ook die tak niet.
        """),
        mo.Html(C.taps_picture({"title": "dilation {d}", "sees": "ziet {n} × {n}",
                                "alt": "Drie roosters van 9 bij 9 pixels, elk met de 9 taps van een 3×3-kernel gekleurd: met dilation 1 naast elkaar in het midden, met dilation 2 twee pixels uit elkaar, over 5 bij 5, en met dilation 4 vier pixels uit elkaar, over 9 bij 9. Een gestreept vierkant geeft aan welk deel elk ziet."})),
        mo.md(r"""
        De 9 taps van een 3×3-kernel met dilation 1, 2 en 4: dezelfde 9 weights, verspreid over 3 bij 3, 5 bij 5 en 9 bij 9 pixels. Het gestreepte vierkant is het deel van het beeld dat één plaats ziet; de pixels tussen de taps slaat hij over.
        """),
        L.watch_out("In [Vormen laten groeien en krimpen: erosie en dilatatie](/course/open/cv-morphology?lang=nl) liet dilatatie de witte vormen van een binair beeld groeien. De dilation van een convolutie is iets anders: hoe ver zijn taps uit elkaar liggen. Met `dilation=1`, de standaard, liggen ze naast elkaar."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **dilated convolution** | een convolutie waarvan de taps uitgespreid zijn met gaten ertussen: hij ziet een groter deel van de map met dezelfde weights | `nn.Conv2d(c_in, c, 3, padding=d, dilation=d)` |
        | **dilation** | hoeveel plaatsen de taps van een convolutie uit elkaar liggen; 1 is naast elkaar | `dilation=d` |
        | **tap** | één plaats van de input die een kernel met een van zijn weights vermenigvuldigt | |
        | **ASPP** | atrous spatial pyramid pooling: een 1×1-convolutie en 3×3-convoluties met verschillende dilations naast elkaar op één map, hun maps gestapeld en gecombineerd door een 1×1-convolutie | `ASPP` |
        | **DeepLab** | een netwerk voor semantic segmentation dat de laatste poolings uit een CNN haalt, de layers erna een dilation geeft, en eindigt met een ASPP | `DeepLab` |
        | `nn.MaxPool2d(2, stride=1)` | max pooling van 2 bij 2 die 1 plaats per keer opschuift: de map verliest maar één plaats | |
        | `x[:, :, ::2, ::2]` | elke tweede plaats van de maps, langs beide kanten, vanaf de eerste: `::2` is een slice met stap 2, zoals `range(0, n, 2)` | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_maps():
    mo.vstack([
        mo.Html(C.maps_picture({"scene": "De scène", "fcn": "FCN, stride 16", "deeplab": "DeepLab", "truth": "De echte klassen",
                                "kinds": ("cirkel", "vierkant", "driehoek"),
                                "alt": "Vier plaatjes naast elkaar: de scène van het uitgewerkte voorbeeld; de klasse die het FCN van de course met stride 16 elke pixel ervan geeft, grove vlekken; de klasse die het DeepLab van de course elke pixel ervan geeft, dicht bij de echte vormen; en de echte class map."})),
        mo.md(r"""
        De scène van het uitgewerkte voorbeeld, en de klassen die twee netwerken met dezelfde vier convoluties hem geven, elk net zo getraind als het FCN van de vorige les, seed 0. Het FCN met stride 16 poolt vier keer, en tekent zijn klassen vanuit een map van 8 bij 8. Het DeepLab van de course poolt twee keer, geeft conv4 en zijn ASPP een dilation, en tekent vanuit een map van 32 bij 32.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Spreid de taps te ver uit, en de layers slaan pixels over. De slider kiest de dilation van een tweede 3×3-convolutie na een eerste met dilation 1. Het rooster is het beeld rond één plaats, het omlijnde vierkantje in het midden: een pixel die de plaats ziet, is gekleurd, voller naarmate hij de plaats langs meer paden door de twee layers bereikt; een pixel die hij overslaat, blijft leeg. Het gestreepte vierkant is het deel dat hij beslaat.
    """)
    return


@app.cell(hide_code=True)
def explore_gaps_cell():
    def _frame(d):
        d = int(d)
        _html, _stop = C.gaps_frame(d, {"title": f"3×3, dan 3×3 met dilation {d}", "key": "hoe voller, hoe meer paden",
                                        "alt": "Een rooster van 15 bij 15 pixels rond één plaats, met de pixels die een 3×3-convolutie met dilation 1 gevolgd door een met dilation {d} ziet gekleurd, en de pixels die hij overslaat leeg."})
        _skips = "ziet hij elke pixel erin" if _stop["gaps"] == 0 else f"ziet hij er {_stop['seen']} van en slaat hij er {_stop['gaps']} over"
        return _html, f"Dilation {d}: de plaats beslaat {_stop['width']} bij {_stop['width']} pixels, {_stop['width'] ** 2} in totaal, en {_skips}."

    explore_gaps = L.explore("gaps-slider", "dilation", _frame, start=2, values=list(C.GAPS), role="",
                             task="Vanaf welke dilation slaat de plaats pixels over? En hoeveel pixels ziet hij vanaf daar?")
    explore_gaps
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Tot en met dilation 3 ziet de plaats elke pixel die hij beslaat: rond elke tap van de tweede layer beslaat de eerste 3 pixels op een rij, dus taps tot 3 uit elkaar laten geen gat. Vanaf dilation 4 liggen de taps verder uit elkaar dan dat, en ziet de plaats 81 pixels, de 3 bij 3 van de eerste layer rond elk van de 9 taps van de tweede, hoe ver ook uit elkaar: bij dilation 4 beslaat hij 11 bij 11 pixels en slaat hij er 40 over, bij 6 beslaat hij 15 bij 15 en slaat hij er 144 over. Daarom groeien de dilations van een stapel stap voor stap: 1, 2, 4 en 8 laten geen gat, zoals de tweede cel van het uitgewerkte voorbeeld laat zien. In het DeepLab van de course zijn conv3, conv4 en de tak met dilation 8 een stapel van 1, 2 en 8: die slaat een paar plaatsen over van de map die conv3 krijgt. De takken met dilation 2 en 4 zien elke plaats die ze beslaan.

        De tabel geeft de netwerken van de course over de seeds 0 tot 4, elk net zo getraind als het FCN van de vorige les: de zijde van de laatste map voor de upsampler, hoeveel pixels één plaats ervan ziet, bij een ASPP van zijn 1×1-tak tot zijn tak met de grootste dilation, en de IoU van de pixels van elke soort op de testscènes, van de laagste tot de hoogste over de seeds. *DeepLab, ASPP 1, 1, 1* is het DeepLab van de course met dilation 1 in elke tak van zijn ASPP; *met dilation, geen ASPP* heeft conv4 met dilation 2 en, in plaats van de ASPP, de laatste 3×3-convolutie van het FCN, met dilation 4; het netwerk eronder heeft dezelfde layers zonder dilation.
        """),
        mo.md(C.variants_table({"header": ("Netwerk", "Laatste map", "Ziet", "Cirkels", "Vierkanten", "Driehoeken"),
                                "names": {"deeplab": "DeepLab", "same-rates": "DeepLab, ASPP 1, 1, 1",
                                          "dilated": "met dilation, geen ASPP", "plain": "geen dilation, geen ASPP",
                                          "fcn-4": "FCN, stride 4", "fcn-8": "FCN, stride 8", "fcn-16": "FCN, stride 16"}}, L.lang)),
        mo.md(r"""
        Een fijnere map helpt zodra elke plaats genoeg ziet. Elk van de vier netwerken met een laatste map van 32 bij 32 waarvan de plaatsen 34 pixels of meer zien, heeft voor elke soort een hogere mediaan van de IoU dan de FCN's met stride 8 en 16; het FCN met stride 4 heeft ook een map van 32 bij 32, maar zijn plaatsen zien maar 18 pixels, en het scoort het laagst van allemaal. Een dilation alleen helpt op de scènes van de course niet: zonder ASPP heeft het netwerk met dilation voor elke soort een lagere mediaan dan dat zonder, en welke van de twee beter is, wisselt met de seed. Een netwerk dat vanaf nul leert, heeft geen getrainde weights te bewaren: welke afstand zijn taps ook hebben, het leert er weights voor. Met een ASPP na de conv4 met dilation komt het netwerk boven allebei uit. Maar met dilation 1, 1 en 1 doet die ASPP het ongeveer even goed als met 2, 4 en 8, dus het meeste van wat de ASPP hier toevoegt, komt van zijn extra takken, en daarmee bijna twee keer zoveel parameters: 193.188 tegen 98.724. De dilations 2, 4 en 8 geven de cirkels wel met elke seed een hogere IoU. Waar DeepLab een dilation eerst voor gebruikte, liet de derde cel van het uitgewerkte voorbeeld zien: een netwerk dat op foto's getraind is, houdt zijn weights als zijn poolings eruit gaan. De DeepLab-papers gebruiken het ook om een plaats op grote foto's ver om zich heen te laten zien zonder te poolen; op de scènes van de course leverde dat niets op.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Semantic segmentation met DeepLab:

    1. Begin bij een CNN met poolings, hier de layers van het FCN van de course met stride 16.
    2. Haal de laatste poolings eruit, hier de derde en de vierde: de laatste map blijft 4 keer kleiner dan de scène in plaats van 16.
    3. Geef elke layer na een pooling die eruit is een dilation: 2 na één, 4 na twee, met padding zo groot als de dilation.
    4. Aan het eind een ASPP: een 1×1-convolutie en 3×3-convoluties met verschillende dilations, naast elkaar op dezelfde map, elk gevolgd door ReLU; hun maps gestapeld langs de channels; een 1×1-convolutie met ReLU terug naar `c` channels.
    5. Een 1×1-convolutie naar een score voor elke klasse, en een transposed convolution die de scores zo groot maakt als de scène.
    6. Train met de cross-entropy van elke pixel, en meet de IoU van elke soort, zoals bij het FCN.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het uitgewerkte voorbeeld volgt het recept met layers die nog niets geleerd hebben. De eerste cel maakt één dilated convolution, en een 5×5-convolutie waarvan de kernel dezelfde 9 weights uitgespreid bevat, met nullen ertussen: de twee geven dezelfde map. De tweede cel stapelt 3×3-convoluties met dilation 1, 2, 4 en 8 op een beeld met één enkele 1 in het midden, elke weight 1, en telt na elke convolutie hoeveel pixels de 1 bereikt. De derde cel doet stap 2 en 3 voor conv4 op de scène uit [Labels, boxes, masks en keypoints](/course/open/cv-annotations?lang=nl), opnieuw gemaakt met `scene(18)`: een keer met de derde pooling, zoals in het FCN, en een keer met die pooling eruit en conv4 met dilation, met dezelfde weights. Stap 4 en 5 zijn oefening 2 en 3, en stap 6 is wat de demo doet. Voer de cellen uit zoals ze zijn.
    """)
    return


@app.cell
def example_spread():
    import torch
    from torch import nn

    # een 3x3-convolutie met dilation 2, en een 5x5-convolutie met dezelfde kernel uitgespreid,
    # nullen tussen zijn taps
    torch.manual_seed(0)
    dilated = nn.Conv2d(1, 1, 3, padding=2, dilation=2, bias=False)
    spread = nn.Conv2d(1, 1, 5, padding=2, bias=False)
    kernel = torch.zeros(5, 5)
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                kernel[2 * i, 2 * j] = dilated.weight[0, 0, i, j]
        spread.weight[0, 0] = kernel
    print(kernel)

    pic = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        a = dilated(pic)
        b = spread(pic)
    print(a.shape, b.shape)
    print(round(float((a - b).abs().max()), 5))
    return nn, torch


@app.cell
def example_seen(nn, torch):
    # een stapel 3x3-convoluties met dilation 1, 2, 4 en 8, elke weight 1, op een beeld met
    # één enkele 1 in het midden: de pixels die de 1 bereikt, zien hem
    dot = torch.zeros(1, 1, 41, 41)
    dot[0, 0, 20, 20] = 1
    for d in [1, 2, 4, 8]:
        conv = nn.Conv2d(1, 1, 3, padding=d, dilation=d, bias=False)
        nn.init.ones_(conv.weight)
        with torch.no_grad():
            dot = conv(dot)
        # de pixels boven 0 in de middelste rij, en in het hele beeld
        print(d, int((dot[0, 0, 20] > 0).sum()), int((dot[0, 0] > 0).sum()))
    return


@app.cell
def example_pooling(nn, torch):
    from everything_ai.scenes import scene

    # de layers van het FCN van de course met stride 16, tot en met conv4, en de scène tot en met conv3
    torch.manual_seed(0)
    conv1 = nn.Conv2d(3, 16, 3, padding=1)
    conv2 = nn.Conv2d(16, 32, 3, padding=1)
    conv3 = nn.Conv2d(32, 64, 3, padding=1)
    conv4 = nn.Conv2d(64, 64, 3, padding=1)
    pool = nn.MaxPool2d(2)
    s = scene(18)
    x = torch.tensor(s.image).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        y = pool(torch.relu(conv2(pool(torch.relu(conv1(x))))))
        y = torch.relu(conv3(y))

        # het FCN: de derde pooling, dan conv4
        fcn = torch.relu(conv4(pool(y)))

        # 2. en 3. de pooling eruit: een die 1 plaats per keer opschuift, en conv4
        #    met dilation 2, met dezelfde weights
        conv4d = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        conv4d.load_state_dict(conv4.state_dict())
        deeplab = torch.relu(conv4d(nn.MaxPool2d(2, stride=1)(y)))
    print(fcn.shape, deeplab.shape)
    # elke tweede plaats van de map van DeepLab, tegen de map van het FCN
    print(round(float((deeplab[:, :, ::2, ::2] - fcn).abs().max()), 5))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de eerste cel heeft de 5×5-kernel de 9 weights van de dilated convolution in elke tweede rij en kolom, en nullen daartussen. De twee maps hebben dezelfde shape, (1, 1, 8, 8), dankzij padding 2, en hun grootste verschil is 0,0: een dilated convolution is een grotere kernel met gaten. In de tweede cel bereikt de 1 na de eerste layer 3 pixels van de middelste rij, na de tweede 7, na de derde 15 en na de vierde 31: 9, 49, 225 en 961 pixels van het beeld, elke pixel van een vierkant van 3, 7, 15 en 31, zonder gat. In de derde cel is de map van het FCN na conv4 (1, 64, 16, 16). Met de pooling die 1 plaats per keer opschuift, verliest de map van 32 bij 32 één plaats, en is de map van DeepLab (1, 64, 31, 31). Zijn plaatsen 0, 2, 4 enzovoort tot en met 30, 16 langs elke kant, zijn de map van het FCN: het grootste verschil is 0,0. Daartussen heeft DeepLab langs elke kant 15 nieuwe plaatsen, die het FCN nooit uitrekende. Het DeepLab van de course in oefening 3 laat de pooling helemaal weg, dus zijn maps blijven 32 bij 32.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_field():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat één plaats ziet

        Maak `field(layers)` af: `layers` is een lijst met een tuple `(kernel, stride, dilation)` voor elke layer, van de scène omhoog. Een 3×3-convolutie is `(3, 1, 1)`, met dilation 2 `(3, 1, 2)`, een pooling van 2 `(2, 2, 1)`, een 1×1-convolutie `(1, 1, 1)`. De functie geeft terug hoeveel pixels van de scène één plaats na de laatste layer ziet, langs één kant. Begin bij één pixel, en een stap van 1 tussen buren; elke layer telt er `(k - 1) * d` keer de stap bij op, en vermenigvuldigt dan de stap met zijn stride, zoals in [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl). De check probeert het FCN van de course met stride 8, drie convoluties met dilation, elke tak van het DeepLab van de course, een stapel met stride 2 in allebei de layers, en helemaal geen layers.
        """),
        L.task("Laat `field` de pixels tellen die één plaats ziet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_field():
    box_field = L.code_box("field", lines=10)
    box_field
    return (box_field,)


@app.cell(hide_code=True)
def check_field(box_field):
    result_field = L.check("field", box_field.value)
    result_field
    return (result_field,)


@app.cell(hide_code=True)
def hints_field():
    L.hints(
        "field",
        [
            "Twee namen houden de telling bij: `seen`, de pixels die één plaats tot nu toe ziet, en `step`, hoeveel pixels van de scène twee buurplaatsen van de map tot nu toe uit elkaar liggen. Zonder layers zijn ze allebei 1. Dan verandert een loop over de layers ze allebei.",
            "`for k, stride, d in layers:` geeft de drie getallen van elke tuple een naam. In de loop eerst `seen += (k - 1) * d * step`, dan `step *= stride`.",
            "`seen = 1` en `step = 1`, de loop van hint 2, en na de loop `return ___`.",
        ],
        why="Een layer werkt op zijn input, waarvan de buren `step` pixels uit elkaar liggen, dus zijn de `(k - 1) * d` plaatsen die hij bij de middelste optelt `(k - 1) * d * step` pixels. Zijn stride telt pas voor de layers erna, dus verandert `step` na `seen`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_aspp():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ASPP

        Maak de class `ASPP(c_in, c, rates)` af, stap 4 van het recept. In `__init__`: `self.branches`, een `nn.ModuleList` met eerst een 1×1-convolutie van `c_in` channels naar `c`, en dan een 3×3-convolutie van `c_in` naar `c` voor elke dilation `r` in `rates`, met `padding=r` en `dilation=r`; en `self.project`, een 1×1-convolutie die de `c` channels van elke tak krijgt en er `c` geeft. In `forward`: elke tak op dezelfde `x`, elk gevolgd door ReLU; hun maps gestapeld langs de channels met `torch.cat(..., dim=1)`; en `self.project`, gevolgd door ReLU. De check probeert een ASPP met de dilations 2 en 4 op willekeurige maps, telt zijn parameters, en probeert een andere met drie dilations op een map die niet vierkant is. Klopt het, dan bewaart hij `ASPP` in je project, als `work/projects/cnn/aspp.py`; je DeepLab bouwt erop.
        """),
        L.task("Laat `ASPP` een 1×1-convolutie en 3×3-convoluties met dilation naast elkaar zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_aspp():
    box_aspp = L.code_box("aspp", lines=16)
    box_aspp
    return (box_aspp,)


@app.cell(hide_code=True)
def check_aspp(box_aspp):
    result_aspp = L.check("aspp", box_aspp.value)
    result_aspp
    return (result_aspp,)


@app.cell(hide_code=True)
def hints_aspp():
    L.hints(
        "aspp",
        [
            "`self.branches` houdt de convoluties naast elkaar, zoals een lijst, maar dan in een `nn.ModuleList`, zodat PyTorch hun weights ziet. Een list comprehension kan één 3×3-convolutie maken voor elke `r` in `rates`; een lijst met de 1×1-convolutie komt ervoor, met `+`.",
            "`nn.ModuleList([nn.Conv2d(c_in, c, 1)] + [nn.Conv2d(c_in, c, 3, padding=r, dilation=r) for r in rates])`. Met de 1×1-convolutie zijn er `len(rates) + 1` takken, dus `self.project = nn.Conv2d(c * (len(rates) + 1), c, 1)`.",
            "In `forward`: `maps = [torch.relu(branch(x)) for branch in self.branches]`, dan `return ___`, met de maps gestapeld langs `dim=1`, `self.project` en ReLU.",
        ],
        why="Elke tak krijgt dezelfde map, en `padding=r` houdt de map van de tak met dilation `r` zo groot, dus geeft elke tak een map van dezelfde grootte, zoals `torch.cat` nodig heeft. Gestapeld langs de channels krijgt `self.project` elke tak heel, en leert hij hoeveel hij van elke tak op elke plaats gebruikt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_deeplab():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je DeepLab

        Maak de class `DeepLab` af. `__init__` heeft al conv1, conv2 en conv3 van het FCN van de course met stride 16, de pooling, je `ASPP` met de dilations 2, 4 en 8, de 1×1-convolutie naar de scores, en je `upsampler` uit de vorige les met stride 4. Vul `self.conv4` in: een 3×3-convolutie van 64 channels naar 64. In het FCN kwam hij na de derde pooling, en die is weg. Dan `forward`: conv1 en conv2, elk gevolgd door ReLU en de pooling; conv3 en conv4, elk gevolgd door ReLU, zonder pooling; dan de ASPP, de scores en de upsampler. De check verandert één pixel van een scène van 128 bij 128 en telt de rijen van de scores die hij bereikt, probeert twee willekeurige beelden van 32 bij 32 en telt de parameters, en probeert een beeld van 24 bij 40. Klopt het, dan bewaart hij `DeepLab` in je project, als `work/projects/cnn/deeplab.py`; de demo traint het.
        """),
        L.task("Laat `DeepLab` twee keer poolen, en geef wat erna komt een dilation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_deeplab():
    box_deeplab = L.code_box("deeplab", lines=26)
    box_deeplab
    return (box_deeplab,)


@app.cell(hide_code=True)
def check_deeplab(box_deeplab):
    result_deeplab = L.check("deeplab", box_deeplab.value)
    result_deeplab
    return (result_deeplab,)


@app.cell(hide_code=True)
def hints_deeplab():
    L.hints(
        "deeplab",
        [
            "conv4 kwam na een pooling die weg is, dus krijgt hij dilation 2, zoals conv4d in de derde cel van het uitgewerkte voorbeeld, met padding 2 zodat de map zijn grootte houdt. Alleen na conv1 en conv2 komt `self.pool`.",
            "`self.conv4 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)`. In `forward`: `x = self.pool(torch.relu(self.conv1(x)))`, hetzelfde voor conv2, dan `x = torch.relu(self.conv3(x))` en hetzelfde voor conv4.",
            "Na conv4: `return ___`, met eerst de ASPP, dan de scores, dan de upsampler.",
        ],
        why="Zonder de derde en de vierde pooling blijft de map 4 keer kleiner dan de scène, en maakt de upsampler met stride 4 de scores zo groot als de scène. conv4 en de ASPP krijgen een dilation, zodat ze ongeveer zo ver reiken als ze na die poolings zouden doen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Druk op de knop: de demo traint je `DeepLab`, met je `ASPP` en je `upsampler`, op de class maps die je `class_map` van de 2.000 trainingsscènes maakt, 8 epochs lang in mini-batches van 32, net zoals het FCN van de course trainde. Na elke epoch meet hij je DeepLab op de 200 testscènes met je `kind_iou`. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; op de computer van de course duurde dat ongeveer 6 minuten. Terwijl hij traint, laat de grafiek elke 10 batches de loss op de trainingsscènes zien, en gestreept de loss op de 200 testscènes na elke epoch. Ernaast staan de klassen die je DeepLab op dat moment aan drie testscènes geeft, en daaronder de feature maps van je ASPP op de scène van het uitgewerkte voorbeeld; allebei worden ze opnieuw getekend terwijl hij traint. Is hij klaar, dan kun je teruggaan door de epochs en zie je ze allebei na elke epoch.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je DeepLab")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_deeplab(train_button, result_aspp, result_deeplab):
    result_aspp, result_deeplab  # opnieuw uitvoeren als een oefening je module bewaart
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je DeepLab te trainen.*")
    else:
        _now = {}   # de live grafiek, gemaakt bij het eerste bericht
        _words = {"scene": "scène {seed}", "kinds": ("cirkel", "vierkant", "driehoek"), "layer": "Layer {name} van je DeepLab op scène {seed}",
                  "alt": "Drie testscènes, elk met de klasse die je DeepLab na die epoch elke pixel gaf."}
        _words_now = _words | {"alt": "Drie testscènes, elk met de klasse die je DeepLab elke pixel nu geeft."}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je DeepLab: de loss tijdens het trainen", "x": "epoch", "y": "loss",
                                        "alt": "De loss van je DeepLab tot nu toe: op de trainingsscènes elke 10 batches, en gestreept op de 200 testscènes na elke epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "trainingsscènes, elke 10 batches"), ("loss", True, "testscènes, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _words_now, L.lang, "nv-deeplab-now"))
                _now["test"] = []
            if "val_loss" in _p:   # de testscènes na elke epoch, gestreept
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Je DeepLab** · epoch {_p['epoch']} van {_p['epochs']} · test loss {L.number(_p['val_loss'], 3)}")
            else:   # de trainingsscènes elke 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Je DeepLab** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 3 af: dan staat je `DeepLab` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Het trainen duurt hier te lang: meer dan twintig minuten. Doet je DeepLab meer werk dan nodig, zoals ook de tweede pooling eruit, of meer channels dan het recept? Volgt het precies het recept, dan is deze computer te traag voor de demo.*",
                "classes": "*Je `class_map` in `work/projects/cnn/classes.py` stopt met een fout, of geeft niet elke pixel van een scène een klasse van 0 tot 3. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) opnieuw.*",
                "deeplab": "*Je `DeepLab` in `work/projects/cnn/deeplab.py`, of de `ASPP` of de `upsampler` die het gebruikt, stopt met een fout, geeft niet 4 scores voor elke pixel van een scène, of traint naar een loss die geen getal is. Heb je een van hun bestanden veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan opnieuw.*",
                "kind_iou": "*Je `kind_iou` in `work/projects/cnn/kind_iou.py` stopt met een fout, of geeft geen 1 voor een class map vergeleken met zichzelf. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course stond in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de opmerking onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "test",
                       "alt": "De loss van je DeepLab na elke epoch: op de trainingsscènes doorgetrokken, op de testscènes gestreept."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"Na epoch {_e}: test loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # terugspoelen door de run, met de hele run in het plaatje
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: welke soort vindt je DeepLab als laatste?")]
            else:
                _pic = [mo.Html(C._FCN.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _ious = ", ".join(f"{_name} {L.number(_v, 2)}" for _name, _v in zip(("cirkels", "vierkanten", "driehoeken"), _result["ious"]))
            _out = mo.vstack(_pic + [
                mo.md(f"""Je DeepLab, met {C._FCN.whole(_result['params'], L.lang)} parameters, heeft {L.number(100 * _result['acc'], 1)}% van de pixels van de 200 testscènes goed, en de IoU van de pixels van elke soort, volgens je `kind_iou`, is: {_ious}. Met seed 0, de run die de demo volgt, heeft het DeepLab van de course 98,8% van de pixels goed, met een IoU van 0,91 voor de cirkels, 0,81 voor de vierkanten en 0,82 voor de driehoeken; over de seeds 0 tot 4 geeft de tabel na de slider de laagste en de hoogste.{_who}"""),
                mo.md("**Je DeepLab, layer voor layer.** Het eerste vak is de scène van het uitgewerkte voorbeeld, en de vakken daarna zijn de layers van je getrainde DeepLab, in de volgorde waarin de scène erdoor gaat; eronder staan de feature maps van de layer die je kiest."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Loop door je getrainde DeepLab, van de scène tot de scores: vanaf welke layer wordt de map niet meer kleiner?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
                mo.md("Na de tweede pooling, layer 4, blijft de map 32 bij 32: conv3, conv4 en je ASPP houden hem zo groot als hij was, met padding zo groot als hun dilation, en de scores ook, 4 channels voor de 4 klassen. In het FCN met stride 16 maakten een derde en een vierde pooling hem 16 bij 16 en daarna 8 bij 8. Je ASPP is één vak: zijn vier takken werken erin, naast elkaar. Alleen de upsampler, de laatste layer, maakt de map weer groter: 128 bij 128, zo groot als de scène."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 4. Waarom een dilation
    """)
    return


@app.cell(hide_code=True)
def explain_dilate_cell():
    explain_dilate = L.explain(
        "why-dilate",
        question="Het eerste DeepLab, van 2015, begint bij een VGG-16 dat op ImageNet getraind is, zet de stride van zijn laatste twee poolings op 1, en geeft de layers erna een dilation. Waarom die dilation? Wat zou er misgaan als het die layers liet zoals ze waren? En waarom hielp een dilation alleen de netwerken van de course niet, die vanaf nul leren?",
        model_answer="Elke layer na een pooling leerde zijn weights voor buren die in zijn input één plaats uit elkaar liggen, en dus twee plaatsen in de map van vóór die pooling. Haal de pooling eruit en laat de layer zoals hij was, en zijn taps komen op buren die twee keer zo dicht bij elkaar liggen: hij vermenigvuldigt andere waarden dan waarvoor hij leerde, en ziet half zo ver. Met dilation 2 liggen zijn taps weer twee plaatsen uit elkaar en bereiken ze precies de waarden die ze eerst bereikten: op elke tweede plaats geeft hij dezelfde getallen als eerst, zoals de derde cel van het uitgewerkte voorbeeld liet zien, dus blijven zijn getrainde weights werken terwijl de map twee keer zo groot blijft. De netwerken van de course leren vanaf nul, dus hebben ze geen getrainde weights te bewaren: welke afstand hun taps ook hebben, ze leren er weights voor. Een dilation laat ze dan alleen een groter deel van de scène zien, en op de scènes van de course hielp dat niet: zonder ASPP deed het netwerk met dilation het bij de mediaan iets slechter dan dat zonder.",
        points=[
            "Een layer die na een pooling leerde, heeft zijn weights voor buren die in de map van vóór die pooling 2 plaatsen uit elkaar liggen.",
            "Met de pooling eruit en dilation 2 komen zijn taps precies weer op die plaatsen: op elke tweede plaats geeft hij wat hij eerst gaf, dus blijven zijn weights werken.",
            "Vanaf nul getraind heeft een netwerk geen weights te bewaren: het leert weights voor de afstand die zijn taps hebben, dus laat een dilation het alleen verder zien, en op de scènes van de course hielp dat niet.",
        ],
    )
    explain_dilate
    return (explain_dilate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Wat één plaats ziet, uit [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl), telt nu ook de dilation mee. De layers van het FCN van de course uit [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) zijn nu een DeepLab met twee van hun poolings eruit, met aan het eind je `upsampler` uit die les, en gemeten met je `kind_iou`; de 1×1-convolutie uit [1×1-convoluties: de channels mengen](/course/open/cv-1x1-convolution?lang=nl) combineert nu de takken van een ASPP. Op de scènes van de course helpen een fijnere map waarvan de plaatsen genoeg zien, en de ASPP: over vijf seeds haalt het DeepLab van de course een IoU van 0,91 tot 0,93 voor de cirkels, waar het FCN met stride 8 0,78 tot 0,87 haalde. DeepLab gebruikte een dilation eerst om de weights te houden die een netwerk op foto's leerde. Je project `cnn` heeft nu `aspp.py` en `deeplab.py`. In de volgende les, *Een mask voor elk voorwerp: het idee van Mask R-CNN*, geeft een detector elke vorm die hij vindt een eigen mask, zodat twee vormen van dezelfde soort niet langer één vlek pixels zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_field, result_aspp, result_deeplab, explain_dilate):
    L.wrap_up(result_field, result_aspp, result_deeplab, explain_dilate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
