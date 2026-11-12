# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-project-detector-or-unet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-project-detector-or-unet", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Project: je eigen U-Net](/course/open/cv-unet?lang=nl) en over [Focal loss en RetinaNet](/course/open/cv-focal-loss?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_iou():
    quick_iou = L.choice(
        "quick-iou",
        question="Met seed 3 had het U-Net van de course na 8 epochs 97,3% van de pixels van de testscènes goed, en toch was zijn mean IoU maar 0,65. Hoe kan dat allebei kloppen?",
        options={
            "right": "Het noemde de driehoeken vierkanten; hun IoU is ongeveer 0, en in de mean IoU tellen de driehoeken even zwaar als de achtergrond, die de meeste pixels beslaat",
            "twice": "De mean IoU telt elke pixel twee keer, één keer voor elke klasse die hij zou kunnen hebben",
            "train": "97,3% gaat over de trainingsscènes, en 0,65 over de testscènes",
            "edges": "De mean IoU kijkt alleen naar de pixels op de randen van de vormen",
        },
        correct="right",
        why={
            "right": "De meeste pixels zijn achtergrond, en die had het U-Net goed: dus 97,3% van alle pixels. De mean IoU neemt het gemiddelde van de IoU van de 4 klassen, en elke klasse telt even zwaar: met de driehoeken rond 0 zakt hij naar 0,65. Vandaag wordt je U-Net weer zo gemeten, met de mean IoU over alle pixels van alle testscènes, of je detector met de mAP, die ook elke soort even zwaar laat tellen.",
            "twice": "Elke pixel heeft één klasse, en elke klasse krijgt haar eigen IoU: de pixels die ze kreeg en moest hebben, gedeeld door de pixels die ze kreeg of moest hebben. Dan het gemiddelde over de 4 klassen. Zie [Project: je eigen U-Net](/course/open/cv-unet?lang=nl).",
            "train": "Beide getallen gaan over de 200 testscènes. Het verschil zit in wat ze tellen: het aandeel van alle pixels, waarvan de meeste achtergrond zijn, of het gemiddelde van de IoU van elke klasse, die allemaal even zwaar tellen.",
            "edges": "De IoU van een klasse telt alle pixels van die klasse, ook binnen de vormen. Het verschil zit in hoe de twee getallen de klassen wegen: in de mean IoU tellen de paar pixels van de driehoeken even zwaar als de hele achtergrond.",
        },
    )
    quick_iou
    return (quick_iou,)


@app.cell(hide_code=True)
def quick_check_focal():
    quick_focal = L.choice(
        "quick-focal",
        question="Hoe zwaar telt in de focal loss met γ = 2 een anchor die de detector met een kans van 0,9 bij de goede klasse zet, vergeleken met zijn cross-entropy?",
        options={
            "right": "Een honderdste: zijn cross-entropy keer (1 − 0,9)² = 0,01",
            "tenth": "Een tiende: zijn cross-entropy keer 1 − 0,9",
            "same": "Even zwaar: de focal loss verandert alleen de anchors die de detector fout heeft",
            "more": "Zwaarder: een makkelijke anchor telt dubbel, zodat hij goed blijft",
        },
        correct="right",
        why={
            "right": "De focal loss weegt de cross-entropy van elke anchor met $(1 - p_t)^\\gamma$: $0{,}1^2 = 0{,}01$. Zo wegen de vele makkelijke anchors van de achtergrond weinig, en gaat het trainen naar de anchors die de detector nog fout heeft. De detector van vandaag is de one-stage detector uit [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl), met zijn eigen loss; de vraag van vandaag is wat er met hem gebeurt als de scènes veranderen.",
            "tenth": "Dat zou γ = 1 zijn. Met γ = 2 wordt de factor gekwadrateerd: $(1 - 0{,}9)^2 = 0{,}01$, een honderdste.",
            "same": "Hij verandert elke anchor, met $(1 - p_t)^\\gamma$: veel bij de makkelijke, dicht bij $p_t = 1$, en nauwelijks bij de moeilijke, dicht bij $p_t = 0$. Zie [Focal loss en RetinaNet](/course/open/cv-focal-loss?lang=nl).",
            "more": "Het is andersom: een makkelijke anchor krijgt een kleine factor, $(1 - p_t)^\\gamma$, dus telt hij minder, en tellen de moeilijke anchors naar verhouding zwaarder.",
        },
    )
    quick_focal
    return (quick_focal,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Elk netwerk uit dit deel leerde van de scènes van de course: heldere vormen op een egaal grijs, in goed licht. Op nieuwe scènes zoals die haalde het U-Net van de course een mean IoU van ongeveer 0,95, en zijn detector een mAP van ongeveer 0,91. Maar een camera in de mist, in de schaduw of in de schemering maakt zulke afbeeldingen niet. Een netwerk dat data tegenkomt die niet lijkt op de data waarvan het leerde, krijgt te maken met een **distribution shift**, en dan zegt zijn testscore weinig over wat het doet. Vandaag kies je een van je twee netwerken, je U-Net of je detector, en maak je de scènes zelf moeilijker: met mist, met licht dat naar één kant afneemt, en met ruis. De vormen blijven waar ze zijn; alleen hun pixels veranderen. Je ziet hoe netwerken die van makkelijke scènes leerden het op moeilijke doen, kiest op validatiescènes waarvan het jouwe leert, test het één keer, en kijkt dan waar het nog faalt.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken", "Je CNN op kleurenfoto's", "Je eigen one-stage detector",
                            "Je eigen U-Net", "Eindproject: ResNet met Grad-CAM", "Eindproject: U-Net of detector"],
                           8, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_easy():
    guess_easy = L.guess(
        "easy-on-hard",
        question="De detector van de course leerde van de 2.000 trainingsscènes zoals ze zijn, en vindt de vormen van de 200 validatiescènes met een mAP van ongeveer 0,91. Dan worden die 200 scènes moeilijker gemaakt, elk met haar eigen mist, licht en ruis. Welke mAP haalt dezelfde detector daarop?",
        options={
            "same": "Ongeveer 0,9: de vormen zijn dezelfde",
            "some": "Ongeveer 0,6",
            "third": "Ongeveer 0,3",
            "none": "Ongeveer 0,05: hij vindt bijna niets",
        },
        reveal="Ongeveer 0,3: van 0,22 tot 0,28 over de seeds 0 tot en met 4, tegen 0,90 tot 0,92 op de scènes zoals ze zijn. En het U-Net van de course, dat van dezelfde makkelijke scènes leerde, zakt van een mean IoU van 0,94 naar 0,23, in de mediaan van de vijf seeds. De vormen zijn dezelfde, op dezelfde plekken; de netwerken zagen nooit pixels zoals deze. De slider hieronder laat zien waar ze breken.",
    )
    guess_easy
    return (guess_easy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Dezelfde scènes, ander licht.** Een moeilijke scène is een scène van de course, met dezelfde vormen, masks en boxes; alleen haar pixels veranderen, in drie stappen, in deze volgorde. De afbeelding werkt met floats, van 0 tot 255.

        1. **Mist** trekt elk getal naar het gemiddelde $m$ van alle getallen van de afbeelding: $x' = m + (x - m)(1 - \text{fog})$. Elk verschil met het gemiddelde wordt $1 - \text{fog}$ keer zo groot: de afbeelding verliest contrast, zoals de doffe foto uit [Grijstinten spreiden: histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl). Met fog 1 is het één grijs.
        2. **Licht** van rechts: elk getal van kolom $j$ wordt vermenigvuldigd met $1 - \text{light} \cdot \left(1 - \frac{j}{W - 1}\right)$, waarin $W$ het aantal kolommen is. De rechterkolom blijft zoals hij is, de linker wordt $1 - \text{light}$ keer zo helder, en de kolommen ertussen iets daartussenin.
        3. **Ruis**: bij elk getal, voor elke pixel en elk channel, komt een getal uit een normale verdeling met een gemiddelde van 0 en een standaardafwijking $\sigma$, in grijswaarden.

        Aan het eind gaat elk getal terug binnen 0 en 255 met `np.clip(x, 0, 255)`, wordt het afgerond, en wordt het uint8. Elke moeilijke scène trekt haar eigen mist, licht en $\sigma$ uit haar seed, elk gelijkmatig van 0 tot 0,8, van 0 tot 0,8 en van 0 tot 24, en haar ruis ook: zo is een moeilijke scène op elke computer dezelfde.

        **Distribution shift.** Een netwerk leert van wat zijn training data laat zien. De netwerken van de course leerden van makkelijke scènes, waarin een vorm altijd duidelijk afsteekt tegen helder grijs. Op de moeilijke validatiescènes verliezen ze het meeste van wat ze konden: de detector zakt van een mAP van 0,91 naar 0,28, het U-Net van een mean IoU van 0,94 naar 0,23, elk de mediaan van de seeds 0 tot en met 4. De data die een netwerk na zijn training tegenkomt, kan verschillen van de data waarvan het leerde, en een test set uit dezelfde bron als de training data laat dat niet zien.

        **Waarvan het leert, gekozen op de validatiescènes.** De course probeerde voor elk netwerk drie instellingen: leren van de 2.000 trainingsscènes zoals ze zijn; van allemaal moeilijker gemaakt; en van de helft moeilijker gemaakt, die met een even seed. Elke instelling trainde 8 epochs, zoals in [Project: je eigen U-Net](/course/open/cv-unet?lang=nl) en [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl), met de seeds 0 tot en met 4, en werd gemeten op 200 validatiescènes, seeds 3000 tot en met 3199, moeilijker gemaakt: zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) wordt daar elke keuze gemaakt. De 200 testscènes, seeds 0 tot en met 199, moeilijker gemaakt, worden één keer gemeten, aan het eind, zodat hun getal zegt hoe het netwerk het doet op scènes waarvoor niets gekozen is.

        **Waar het faalt.** Eén getal, de mean IoU of de mAP, zegt niet wat er fout gaat. Dus kijk je naar elke vorm van de moeilijke validatiescènes: gevonden of niet. Op de validatiescènes, niet op de testscènes: wat je daar ziet, bepaalt wat je daarna probeert, en de testscènes blijven voor die ene meting aan het eind. Voor het U-Net is een vorm gevonden als meer dan de helft van zijn pixels in zicht zijn soort krijgt. Voor de detector als een van zijn boxes met een score van 0,5 of meer de soort van de vorm heeft en een IoU boven 0,5 ermee, zoals de mAP uit [Een detector beoordelen: mean average precision](/course/open/cv-map?lang=nl) hem telt. Dan zet je de vormen in **groepen** naar één eigenschap: hun pixels in zicht, het licht waar ze liggen, of de mist of de ruis van hun scène. Voor elke groep zegt het **aandeel gevonden** hoeveel van haar vormen het netwerk vond. Een groep met een laag aandeel zegt waar het netwerk faalt, en dus waar de scènes waarvan het leert meer van zouden kunnen bevatten; hoeveel een aandeel zegt, hangt af van hoeveel vormen de groep heeft. Zo'n error analysis kijkt naar de fouten zelf, niet alleen naar hoeveel het er zijn.
        """),
        L.watch_out("Een getal van uint8 is een heel getal van 0 tot 255, en ruis kan heldere en donkere pixels buiten dat bereik duwen. `astype(np.uint8)` houdt een getal boven 255 niet op 255, en een getal onder 0 niet op 0: op de computer van de course wordt 270,4 dan 14 en −3,2 wordt 253, en wat eruit komt kan van computer tot computer verschillen. En hij knipt de decimalen eraf, dus 99,6 wordt 99. Daarom komen `np.clip(x, 0, 255)` en `np.round` eerst."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **distribution shift** | een verschil tussen de data waarvan een netwerk leerde en de data die het later tegenkomt, zoals foto's in ander licht; zijn score op een test set van de oude data zegt dan weinig | |
        | **groep** | de vormen waarvan de eigenschap in één bin ligt, zoals de vormen met 250 tot 500 pixels in zicht | `inside` |
        | **aandeel gevonden** | voor een groep het aantal van haar vormen dat het netwerk vond, gedeeld door het aantal vormen erin | `shares` |
        | `np.clip(x, lo, hi)` | elk getal van x onder lo wordt lo, elk getal boven hi wordt hi | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_changes():
    mo.vstack([
        mo.Html(C.changes_picture({"as_is": "zoals ze is", "fog": f"mist {L.number(C.FOG, 1)}", "light": f"licht {L.number(C.LIGHT, 1)}", "noise": f"ruis σ {C.NOISE:g}",
                                   "all": "alle drie", "alt": f"Validatiescène {C.PLAY} zoals ze is, met alleen mist, alleen licht en alleen ruis op hun sterkst, en met alle drie zoals haar seed ze trok."})),
        mo.md(f"""
        Validatiescène {C.PLAY}, zoals ze is en moeilijker gemaakt door de course: met alleen mist {L.number(C.FOG, 1)} krimpt elk verschil met het gemiddelde tot een vijfde, dus blijven de vormen waar ze zijn, maar vervagen hun kleuren bijna in het grijs; met alleen licht {L.number(C.LIGHT, 1)} houdt de linkerrand een vijfde van zijn helderheid; met alleen ruis komt er bij elk getal een eigen willekeurig getal, met een standaardafwijking van {C.NOISE:g} grijswaarden; en met alle drie zoals haar seed ze trok: mist {L.number(C.conditions(C.PLAY)['fog'], 2)}, licht {L.number(C.conditions(C.PLAY)['light'], 2)} en ruis σ {L.number(C.conditions(C.PLAY)['sigma'], 1)}.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Hoe moeilijk kan een scène worden voordat een netwerk faalt? De slider maakt alle 200 validatiescènes tegelijk moeilijker: op niveau 0 zijn ze zoals ze zijn, op niveau 1 heeft elke scène mist 0,8, licht 0,8 en ruis σ 24, en daartussen is elke verandering dat deel van haar maximum. De grafiek laat voor elk niveau de mean IoU van het U-Net van de course zien en de mAP van zijn detector op die scènes, elk getraind met seed 0: één keer op de makkelijke trainingsscènes, en één keer zoals de course op de validatiescènes koos.
    """)
    return


@app.cell(hide_code=True)
def explore_levels_cell():
    def _frame(v):
        _labels = {"title": "Op de 200 validatiescènes", "x": "niveau", "y": "mean IoU of mAP", "scene": "validatiescène {seed}",
                   "unet_easy": "U-Net, leerde van makkelijke scènes", "unet_chosen": "U-Net, leerde van moeilijke scènes",
                   "detector_easy": "detector, leerde van makkelijke scènes", "detector_chosen": "detector, leerde van makkelijke en moeilijke scènes",
                   "alt": "Een validatiescène op dit niveau, naast de mean IoU van het U-Net van de course en de mAP van zijn detector op elk niveau, elk geleerd van makkelijke scènes en zoals de course koos, met een lijn op dit niveau."}
        _html, _stop = C.play_frame(v, _labels, L.lang)
        return _html, (f"Op niveau {L.number(_stop['level'], 2)}: mist {L.number(_stop['fog'], 2)}, licht {L.number(_stop['light'], 2)} en ruis {L.number(_stop['noise'], 1)}. "
                       f"Geleerd van makkelijke scènes haalt het U-Net een mean IoU van {L.number(_stop['unet_easy'], 2)} en de detector een mAP van {L.number(_stop['detector_easy'], 2)}; "
                       f"zoals de course koos {L.number(_stop['unet_chosen'], 2)} en {L.number(_stop['detector_chosen'], 2)}.")

    explore_levels = L.explore("levels-slider", "niveau", _frame, start=0.0, values=list(C.LEVELS), role="",
                               task="Ga van niveau 0 naar niveau 1: op welk niveau falen de netwerken die van makkelijke scènes leerden, en hoe ziet de scène er daar uit? En wat kost leren van moeilijke scènes op niveau 0?")
    explore_levels
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Geleerd van makkelijke scènes doen beide netwerken het op niveau 0,25 nog goed, 0,85 en 0,93, en vóór niveau 0,5 breken ze: daar haalt de detector 0,37 en het U-Net 0,32. Op niveau 1 vindt de detector bijna niets meer, 0,01. Geleerd zoals de course koos, houden ze het vol tot niveau 0,75, met 0,77 en 0,87, en op niveau 1 verliezen ook zij veel, tot 0,37 en 0,52. Op niveau 1 heeft elke scène alle drie de veranderingen op hun maximum, en een moeilijke trainingsscène trekt elk ervan gelijkmatig van 0 tot haar maximum: van de 2.000 trainingsscènes, moeilijker gemaakt, hebben er maar 33 alle drie boven niveau 0,75. En leren van moeilijke scènes kost op niveau 0 een beetje: de detector haalt 0,85 in plaats van 0,92, het U-Net 0,90 in plaats van 0,94.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Je eigen netwerk, op scènes die het moeilijker vindt:

    1. Maak een scène moeilijker: mist naar het gemiddelde, licht dat naar links afneemt, ruis; dan binnen 0 en 255, afgerond, als uint8.
    2. Maak de trainingsscènes zoals de instelling zegt, en de validatiescènes en de testscènes moeilijker, elk met de mist, het licht en de ruis van haar seed.
    3. Kies de instelling op de validatiescènes, over meerdere seeds: dat deed de course hier.
    4. Train je netwerk met de gekozen instelling, en meet het dan één keer op de moeilijke testscènes: de mean IoU voor het U-Net, de mAP voor de detector.
    5. Voor elke vorm van de moeilijke validatiescènes: gevonden of niet. Zet de vormen in groepen naar een eigenschap, en geef het aandeel gevonden van elke groep, met het aantal vormen erin.
    6. Schrijf op wat je ziet: waar het faalt, en wat je daarna probeert, weer gekozen op de validatiescènes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel doet stap 1 voor een verzonnen scène van 2 rijen van 4 pixels, grijs met één rode en één blauwe pixel: de mist met fog 0,5, en het licht met light 0,6. De tweede cel doet het eind van stap 1 voor vijf getallen die de ruis gemaakt zou kunnen hebben. De derde cel doet het eerste deel van stap 5 voor 7 verzonnen vormen: welke in de groep van 60 tot 250 pixels in zicht zitten, en welk aandeel daarvan gevonden is. Voer de cellen uit zoals ze zijn.
    """)
    return


@app.cell
def example_fog():
    import numpy as np

    # 1. een verzonnen scène van 2 rijen van 4 pixels, grijs, met één rode en één blauwe pixel
    image = np.array([[[220, 220, 220], [220, 220, 220], [200, 60, 50], [220, 220, 220]],
                      [[220, 220, 220], [30, 140, 200], [220, 220, 220], [220, 220, 220]]], dtype=np.uint8)
    x = image.astype(float)

    # mist 0,5: elk getal halverwege naar het gemiddelde van alle getallen
    m = x.mean()
    foggy = m + (x - m) * (1 - 0.5)
    print(round(m, 2))
    print(foggy[0, 2])

    # licht 0,6: één getal voor elke kolom, van 0,4 links tot 1 rechts
    w = x.shape[1]
    ramp = 1 - 0.6 * (1 - np.arange(w) / (w - 1))
    print(ramp)
    lit = foggy * ramp.reshape(1, w, 1)
    print(lit[0, 0], lit[0, 3])
    return image, np


@app.cell
def example_clip(np):
    # het eind van stap 1: getallen die de ruis gemaakt zou kunnen hebben
    noisy = np.array([270.4, -3.2, 99.6, 127.5, 128.5])
    print(np.clip(noisy, 0, 255))
    print(np.round(np.clip(noisy, 0, 255)))
    print(np.round(np.clip(noisy, 0, 255)).astype(np.uint8))

    # de ruis uit het voorbeeld van oefening 1: een getal voor elke pixel en elk channel van de verzonnen scène
    noise = np.random.default_rng(0).normal(0, 20, (2, 4, 3))
    print(noise.shape)
    return (noise,)


@app.cell
def example_group(np):
    # 5. verzonnen vormen: of het netwerk elk ervan vond, en hun pixels in zicht
    found = np.array([True, False, True, True, False, True, False])
    sizes = np.array([80, 120, 300, 450, 90, 700, 1200])

    # de groep van 60 tot 250 pixels: welke vormen erin zitten, en welk aandeel daarvan gevonden is
    inside = (sizes >= 60) & (sizes < 250)
    print(inside)
    print(found[inside], found[inside].mean())
    return found, sizes


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de eerste cel is het gemiddelde van alle 24 getallen 193,33. Met fog 0,5 gaat elk getal halverwege daarheen: de rode pixel, (200, 60, 50), wordt (196,67, 126,67, 121,67), veel dichter bij grijs, en het grijs 220 wordt 206,67. Het licht geeft de 4 kolommen 0,4, 0,6, 0,8 en 1: de grijze pixel in de linkerkolom houdt 0,4 van zijn helderheid, 82,67, en die in de rechterkolom blijft 206,67. In de tweede cel maakt `np.clip` van 270,4 255 en van −3,2 0, en maakt `np.round` van 99,6 100. 127,5 en 128,5 worden allebei 128: NumPy rondt een getal dat precies halverwege twee hele getallen ligt af naar het even getal. In de derde cel hebben 3 van de 7 vormen 60 tot 250 pixels in zicht, die van 80, 120 en 90 pixels, en 1 van de 3 is gevonden: een aandeel gevonden van 0,33.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_harder():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een moeilijkere scène

        Maak `harder(image, fog, light, noise)` af, stap 1 van het recept, voor een scène `image` met elke shape (H, W, 3) en getallen van uint8. `noise` bevat een getal voor elke pixel en elk channel, in grijswaarden. Eerst de mist, met het gemiddelde van de hele afbeelding; dan het licht, één getal voor elke kolom; dan de ruis; en aan het eind elk getal binnen 0 en 255, afgerond, als uint8. De eerste cel van het uitgewerkte voorbeeld deed de mist en het licht, de tweede het eind. De check probeert de verzonnen scène met de `noise` uit het uitgewerkte voorbeeld, fog en light van 0 en van 1, ruis die getallen boven 255 en onder 0 duwt, een scène met meer rijen dan kolommen, en een hele validatiescène. Als hij klopt, bewaart hij `harder` in je project, als `work/projects/cnn/harder.py`; de demo maakt er elke scène moeilijker mee.
        """),
        L.task("Laat `harder` mist, licht van rechts en ruis aan een scène toevoegen, en hele getallen van 0 tot 255 teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_harder():
    box_harder = L.code_box("harder", lines=14)
    box_harder
    return (box_harder,)


@app.cell(hide_code=True)
def check_harder(box_harder):
    result_harder = L.check("harder", box_harder.value)
    result_harder
    return (result_harder,)


@app.cell(hide_code=True)
def hints_harder():
    L.hints(
        "harder",
        [
            "Werk met floats, `x = image.astype(float)`, zodat er tijdens het rekenen niets overloopt. Dan drie stappen die elk op de hele array werken, in de volgorde van de commentaren, zoals in de eerste cel van het uitgewerkte voorbeeld, en het eind zoals in de tweede.",
            "`m = x.mean()` en `x = m + (x - m) * (1 - fog)`. Voor het licht: `w = x.shape[1]`, `ramp = 1 - light * (1 - np.arange(w) / (w - 1))` en `x = x * ramp.reshape(1, w, 1)`, zodat elke kolom zijn eigen getal krijgt.",
            "`x = x + noise`, en dan `return np.___(np.clip(x, 0, 255)).astype(np.uint8)`.",
        ],
        why="`ramp.reshape(1, w, 1)` heeft de shape (1, W, 1): broadcasting, zoals in [Broadcasting](/course/open/np-broadcasting?lang=nl), herhaalt hem over de rijen en de 3 channels, dus elke pixel van kolom `j` wordt met hetzelfde getal vermenigvuldigd. Het gemiddelde is één getal voor de hele afbeelding, dus de mist is grijs, geen kleur. En `np.clip` komt vóór `astype`, omdat uint8 geen getal onder 0 of boven 255 kan bevatten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het aandeel gevonden van elke groep

        Maak `shares(found, values, edges)` af, stap 5 van het recept. `found` zegt voor elke vorm of het netwerk hem vond, en `values` bevat een getal voor elke vorm, zoals zijn pixels in zicht. `edges` bevat k + 1 getallen van laag naar hoog, de randen van k bins: een waarde v zit in bin i als `edges[i] <= v < edges[i + 1]`. De functie geeft een lijst van k terug: voor elke bin het aandeel van zijn vormen dat gevonden is, en `None` voor een bin zonder vormen. De derde cel van het uitgewerkte voorbeeld deed het voor één bin. De check probeert de verzonnen vormen uit het uitgewerkte voorbeeld, waarden op een rand, lege bins, waarden buiten elke bin, en lijsten in plaats van arrays. Als hij klopt, bewaart hij `shares` in je project, als `work/projects/cnn/shares.py`; na de demo laat hij zien waar je netwerk faalt.
        """),
        L.task("Laat `shares` het aandeel gevonden van de vormen in elke bin geven, en `None` voor een lege bin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=12)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Een loop over de bins, `for i in range(len(edges) - 1):`, en daarin wat de derde cel van het uitgewerkte voorbeeld deed, met `edges[i]` en `edges[i + 1]` op de plaats van 60 en 250. Verzamel wat elke bin geeft in een lijst.",
            "`out = []` vóór de loop. Daarin: `inside = (values >= edges[i]) & (values < edges[i + 1])`, dan `if inside.sum() > 0:` met `out.append(float(found[inside].mean()))`.",
            "`else:` met `out.append(___)`, en na de loop `return out`.",
        ],
        why="`found[inside]` houdt de vormen van de bin, en het gemiddelde van True en False is het aandeel dat True is. Een bin bevat zijn onderste rand en niet zijn bovenste, dus een waarde op een rand zit in precies één bin. Een lege bin heeft geen aandeel: het gemiddelde van niets is `nan`, en 0 zou je lezen als een bin waarin het netwerk niets vond.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    Kies je netwerk, en druk op de knop. De demo maakt elke scène moeilijker met je `harder`: de 2.000 trainingsscènes zoals de course voor dat netwerk koos, allemaal voor het U-Net en die met een even seed voor de detector, en de 200 validatiescènes en de 200 testscènes allemaal. Dan traint hij je netwerk met seed 0, {C.EPOCHS} epochs lang, zoals in zijn eigen les, kijkt hij naar elke vorm van de moeilijke validatiescènes, en meet hij het één keer op de moeilijke testscènes: je U-Net met je `mean_iou`, je detector met de mAP. Waar een module van jou ontbreekt of niet werkt, springt die van de course in. Hij traint op één thread, zodat hij de run van de course met seed 0 zo goed volgt als je computer toelaat; op de computer van de course duurde dat 9 tot 10 minuten voor het U-Net en ongeveer 2 voor de detector. Terwijl hij traint, laat de grafiek elke 10 batches de loss op de trainingsscènes zien, en gestreept de loss op de validatiescènes na elke epoch. Ernaast staan drie validatiescènes, moeilijker gemaakt, met wat je netwerk er op dat moment van maakt, en daaronder de feature maps van de layer die hij volgt; allebei worden ze opnieuw getekend terwijl hij traint. Als hij klaar is, kun je door de epochs terugspoelen en zie je ze allebei na elke epoch.
    """)
    return


@app.cell(hide_code=True)
def model_choice_cell():
    model_choice = mo.ui.radio(options={"je U-Net": "unet", "je detector": "detector"}, value="je detector", label="Train")
    train_button = mo.ui.run_button(label="Train je netwerk op moeilijkere scènes")
    mo.vstack([model_choice, train_button])
    return model_choice, train_button


@app.cell(hide_code=True)
def demo_network(train_button, model_choice, result_harder):
    result_harder  # run again when exercise 1 saves your module
    if not train_button.value:
        _out = mo.md("*Kies je netwerk, en druk op de knop om het te trainen.*")
    else:
        _kind = model_choice.value
        _name = "U-Net" if _kind == "unet" else "detector"
        _now = {}   # the live chart, made at the first report
        _words = {"scene": "scène {seed}", "map": "", "kinds": ["cirkel", "vierkant", "driehoek"], "shapes": "de vormen", "boxes": "boxes van 0,5 of meer",
                  "layer": f"Layer {{name}} van je {_name} op validatiescène {{seed}}",
                  "alt": ("Drie validatiescènes, moeilijker gemaakt, elk met de class map die je U-Net haar na die epoch gaf." if _kind == "unet" else
                          "Drie validatiescènes, moeilijker gemaakt, elk met haar echte boxes en de gestreepte boxes van 0,5 of meer die je detector haar na die epoch gaf.")}
        _words_now = _words | {"alt": _words["alt"].replace("haar na die epoch gaf", "haar nu geeft")}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": f"Je {_name}: de loss tijdens het trainen", "x": "epoch", "y": "loss",
                                        "alt": f"De loss van je {_name} tot nu toe: op de trainingsscènes elke 10 batches, en gestreept op de 200 validatiescènes na elke epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "trainingsscènes, elke 10 batches"), ("loss", True, "validatiescènes, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _kind, _words_now, L.lang, "nv-scenes-project-now"))
                _now["val"] = []
            if "val_loss" in _p:   # de validatiescènes na elke epoch, gestreept
                _now["val"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["val"], "", role="loss", replace=True)
                _now["chart"].line(f"**Je {_name}** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {L.number(_p['val_loss'], 3)}")
            else:   # de trainingsscènes elke 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Je {_name}** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_kind, _progress)
        if _result is None:
            _out = mo.md(f"*Maak eerst oefening 1 af, en de oefeningen van je {_name} in {'[Project: je eigen U-Net](/course/open/cv-unet?lang=nl)' if _kind == 'unet' else '[Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl)'}: dan staan je `harder` en je {_name} in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Je project doet er hier te lang over: meer dan twee minuten om de scènes moeilijker te maken, meer dan vier voor de targets van de detector, of meer dan een half uur om het U-Net te trainen of tien minuten om de detector te trainen. Doet een module van jou meer werk dan nodig is, zoals een loop over de pixels of de anchors?*",
                "harder": "*Je `harder` in `work/projects/cnn/harder.py` stopt met een fout, of geeft geen afbeelding van 128 bij 128 bij 3 met hele getallen van 0 tot 255. Heb je het bestand veranderd? Verwijder het, en check oefening 1 opnieuw: de check bewaart je code dan weer.*",
                "classes": "*Je `class_map` in `work/projects/cnn/classes.py` stopt met een fout, of geeft niet voor elke scène een class map. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Een soort voor elke pixel: FCN en transposed convolutions](/course/open/cv-fcn?lang=nl) opnieuw.*",
                "unet": "*Je `UNet` in `work/projects/cnn/unet.py` stopt met een fout, of geeft niet 4 scores voor elke pixel. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Project: je eigen U-Net](/course/open/cv-unet?lang=nl) opnieuw.*",
                "mean_iou": "*Je `mean_iou` in `work/projects/cnn/mean_iou.py` stopt met een fout, of geeft geen 1 voor class maps vergeleken met zichzelf. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Project: je eigen U-Net](/course/open/cv-unet?lang=nl) opnieuw.*",
                "targets": "*Je `targets` in `work/projects/cnn/targets.py` stopt met een fout, of geeft niet voor elk van de 768 anchors van een scène een klasse en 4 offsets. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) opnieuw.*",
                "loss": "*Je `detection_loss` in `work/projects/cnn/detection_loss.py` stopt met een fout, of geeft niet één getal terug als tensor waar het trainen naar terug kan rekenen. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) opnieuw.*",
                "detector": "*Je `Detector` in `work/projects/cnn/detector.py` stopt met een fout, of geeft voor een scène geen 768 rijen van 8. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) opnieuw.*",
                "offsets": "*Je `boxes_from` in `work/projects/cnn/offsets.py` stopt met een fout op de offsets van je detector. Heb je het bestand veranderd? Verwijder het, en check de oefening ervan in [Anchors: de boxes waar een detector mee begint](/course/open/cv-anchors?lang=nl) opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validatie",
                       "alt": f"De loss van je {_name} na elke epoch: op de trainingsscènes doorgetrokken, op de validatiescènes gestreept."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"Na epoch {_e}: validation loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: welke vormen vindt je netwerk het eerst, en welke vindt het aan het eind nog steeds niet?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            if _kind == "unet":
                _said = (f"Je U-Net, met {C.whole(_result['params'], L.lang)} parameters, haalt een mean IoU van **{L.number(_result['score'], 3)}** op de 200 testscènes, moeilijker gemaakt en één keer gemeten, "
                         f"en vindt {_result['test_found']} van hun {_result['test_shapes']} vormen. Op dezelfde scènes zoals ze zijn: {L.number(_result['easy'], 3)}. Het U-Net van de course met seed 0, geleerd van moeilijke scènes: 0,908 op de moeilijke testscènes, met 764 van de 808 vormen gevonden, en 0,922 op die scènes zoals ze zijn. Zijn validation loss sprong na epoch 4 en 6 omhoog en zakte daarna telkens weer, tot 0,037 na epoch 8.")
            else:
                _said = (f"Je detector, met {C.whole(_result['params'], L.lang)} parameters, haalt een mAP van **{L.number(_result['score'], 3)}** op de 200 testscènes, moeilijker gemaakt en één keer gemeten, "
                         f"en vindt {_result['test_found']} van hun {_result['test_shapes']} vormen. Op dezelfde scènes zoals ze zijn: {L.number(_result['easy'], 3)}. De detector van de course met seed 0, geleerd van makkelijke en moeilijke scènes: 0,815 op de moeilijke testscènes, met 629 van de 808 vormen gevonden, en 0,852 op die scènes zoals ze zijn.")
            _out = mo.vstack(_pic + [
                mo.md(_said + _who),
                mo.md(f"**Je {_name}, layer voor layer.** Het eerste vak is validatiescène {C.EXAMPLE}, moeilijker gemaakt, en de vakken daarna zijn de layers van je getrainde {_name}, in de volgorde waarin de scène erdoor gaat; eronder staan de feature maps van de layer die je kiest."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Ga door je getrainde netwerk van de scène naar zijn scores: steken de vormen in het donkere linkerdeel af in de feature maps, en vanaf welke layer?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_analysis():
    mo.md(r"""
    ### Waar het faalt

    Als je netwerk getraind is, zet de knop hieronder de 802 vormen van de 200 moeilijke validatiescènes vier keer in groepen: naar hun pixels in zicht, naar het licht waar ze liggen (1: zo helder als in een makkelijke scène, 0,2: een vijfde daarvan), naar de mist van hun scène en naar haar ruis. Je `shares` geeft het aandeel gevonden van elke groep. De balken zijn van je netwerk; de gestreepte lijn over elke balk is het netwerk van de course van dezelfde soort, getraind met seed 0; boven elke balk staat het aantal vormen in zijn groep. Onder de grafieken staan de drie validatiescènes waarin je netwerk de meeste vormen miste: elke vorm die het miste in een dikke box, elke vorm die het vond in een dunne.
    """)
    return


@app.cell(hide_code=True)
def analysis_button_cell():
    analysis_button = mo.ui.run_button(label="Kijk waar het faalt")
    analysis_button
    return (analysis_button,)


@app.cell(hide_code=True)
def analysis_cell(analysis_button, result_shares):
    result_shares  # run again when exercise 2 saves your module
    _run = C.last_run()
    if not analysis_button.value:
        _out = mo.md("*Train eerst je netwerk hierboven, en druk dan op de knop.*")
    elif _run is None:
        _out = mo.md("*Er is nog geen getraind netwerk: druk eerst op de knop van de demo hierboven, en weer op deze als hij klaar is.*")
    else:
        _groups = C.analysis(_run)
        if _groups is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan staat je `shares` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_groups, str):
            _out = mo.md("*Je `shares` in `work/projects/cnn/shares.py` stopt met een fout, of geeft niet voor elke bin een aandeel van 0 tot 1 of `None`. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw.*")
        else:
            _name = "U-Net" if _run["model"] == "unet" else "detector"
            _out = mo.vstack([
                mo.Html(C.groups_figure(_groups, {"size": "Verdeeld naar pixels in zicht", "size_x": "pixels in zicht", "light": "Verdeeld naar licht", "light_x": "licht waar hij ligt",
                                                  "fog": "Verdeeld naar mist", "fog_x": "mist van zijn scène", "noise": "Verdeeld naar ruis", "noise_x": "ruis σ van zijn scène",
                                                  "mine": f"je {_name}", "course": {"unet": "het U-Net van de course, seed 0", "detector": "de detector van de course, seed 0"}[_run["model"]],
                                                  "alt": f"Het aandeel van de vormen van de validatiescènes dat je {_name} vond, {{what}}, met dat van de course als gestreepte lijn en het aantal vormen boven elke balk."}, L.lang)),
                mo.Html(C.worst_picture(_run, {"scene": ("scène {seed}: {n} gemist", "scène {seed}: {n} gemist"), "map": "", "kinds": ["cirkel", "vierkant", "driehoek"],
                                               "missed": "een vorm die het miste", "found": "een vorm die het vond", "boxes": f"boxes van je {_name}, 0,5 of meer",
                                               "alt": f"De drie validatiescènes waarin je {_name} de meeste vormen miste, moeilijker gemaakt: elke gemiste vorm in een dikke box, elke gevonden vorm in een dunne box."}, L.lang)),
                mo.md("Op de moeilijke validatiescènes vindt de detector van de course met seed 0 604 van de 802 vormen. Hij faalt het meest op kleine vormen, 40% van de 133 met 60 tot 250 pixels in zicht tegen 97% van de 79 grootste, en in zware mist, 60% van de 212 vormen bij mist 0,6 tot 0,8 tegen 86% bij mist 0 tot 0,2; over de groepen van ruis blijft zijn aandeel tussen 73% en 78%. Het U-Net van de course met seed 0 vindt er 746 van de 802; zijn laagste aandeel is dat van de kleinste vormen, 85%, en in de zwaarste mist vindt het 88%. De donkerste groep van licht heeft maar 17 vormen: één vorm meer of minder verandert haar aandeel met 6 procentpunten, dus zegt het minder dan het aandeel van een groep van 200."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Oefening 3. Wat je daarna probeert
    """)
    return


@app.cell(hide_code=True)
def choice_next_cell():
    choice_next = L.choice(
        "next-step",
        question="De detector van de course met seed 0 vindt 60% van de vormen van de moeilijke validatiescènes in de zwaarste mist, 0,6 tot 0,8, tegen 86% in de lichtste. Je wilt proberen of trainingsscènes met meer mist helpen. Waar meet je of ze dat doen?",
        options={
            "right": "Op de validatiescènes: het aandeel gevonden in zware mist, en de mean IoU of mAP van allemaal",
            "test": "Op de testscènes, moeilijker gemaakt: die zeggen hoe het het op nieuwe scènes doet",
            "train": "Op de trainingsscènes met zware mist: daar zie je of het ze geleerd heeft",
            "fog": "Alleen op de testscènes met zware mist: dat is de groep die beter moet worden",
        },
        correct="right",
        why={
            "right": "Ja. Elke keuze wordt gemaakt op de validatiescènes, zoals de instelling: het aandeel gevonden in zware mist zegt of de verandering hielp waar dat moest, en de score van allemaal of ze ergens anders iets kostte. De testscènes blijven voor het eind, één keer gemeten, zodat hun getal blijft zeggen hoe het netwerk het doet op scènes waarvoor niets gekozen is.",
            "test": "Zodra je met de testscènes kiest, zegt hun getal niet meer hoe het netwerk het doet op nieuwe scènes: je zou de verandering gekozen hebben die bij deze 200 past. Kies op de validatiescènes, en meet de testscènes één keer, aan het eind.",
            "train": "De trainingsscènes zeggen hoe goed het netwerk de scènes leerde die het zag, niet hoe het het op andere doet: een netwerk kan zijn trainingsscènes goed kennen en toch falen op nieuwe, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "fog": "Dat zijn nog steeds testscènes: ermee kiezen gebruikt de test set, net als met allemaal. En een verandering kan helpen in mist en elders iets kosten: kijk ook naar alle validatiescènes.",
        },
    )
    choice_next
    return (choice_next,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Oefening 4. Je verslag
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Schrijf een kort verslag over je netwerk voor iemand die deze les niet gevolgd heeft: wat werd er moeilijker gemaakt, en hoe deden netwerken die van makkelijke scènes leerden het daar? Hoe werd de instelling gekozen, hoe deed je netwerk het op de moeilijke testscènes, en in welke groepen van de moeilijke validatiescènes faalt het het meest? Gebruik de getallen van je eigen run, of die van de run van de course met seed 0.",
        model_answer="De moeilijke scènes zijn de scènes van de course met dezelfde vormen, moeilijker gemaakt met mist naar het gemiddelde, licht dat naar links afneemt, en ruis, elk per scène getrokken. Een detector die van de scènes zoals ze zijn leerde, haalt een mAP van 0,91 op de validatiescènes zoals ze zijn, en nog maar 0,28 op die scènes moeilijker gemaakt, de medianen van vijf seeds. Ik koos waarvan mijn detector leert op de 200 moeilijke validatiescènes, met de mediaan van vijf seeds: van de scènes zoals ze zijn 0,28, van allemaal moeilijker gemaakt 0,78, en van de helft moeilijker gemaakt 0,80; dus de helft. Eén keer gemeten op de 200 moeilijke testscènes haalt hij met seed 0 een mAP van 0,815 en vindt hij 629 van hun 808 vormen; op dezelfde scènes zoals ze zijn 0,852. Op de moeilijke validatiescènes faalt hij het meest op kleine vormen, 40% van de 133 met 60 tot 250 pixels in zicht, en in zware mist, 60% van de 212 vormen bij mist 0,6 tot 0,8. Daarna zou ik trainingsscènes met meer mist proberen, en op de validatiescènes kiezen of dat daar helpt zonder elders iets te kosten.",
        points=[
            "je zegt wat de moeilijke scènes zijn, dezelfde vormen met mist, licht van één kant en ruis, en hoeveel netwerken die van makkelijke scènes leerden daar verliezen",
            "je zegt dat de instelling gekozen is op de validatiescènes, met wat elke instelling daar haalde, en dat de moeilijke testscènes één keer gemeten zijn, aan het eind",
            "je geeft de score van je netwerk op de moeilijke testscènes, en de groepen van de validatiescènes met het laagste aandeel gevonden en hoeveel vormen ze hebben, met wat je daarom daarna zou proberen, gekozen op de validatiescènes",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het U-Net uit [Project: je eigen U-Net](/course/open/cv-unet?lang=nl) en de detector uit [Project: je eigen one-stage detector](/course/open/cv-one-stage-detector?lang=nl) kwamen scènes tegen die niet leken op die waarvan ze leerden: dezelfde vormen, in de mist, in de schaduw en met ruis. De mist neemt het contrast weg dat [Grijstinten spreiden: histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl) uitspreidde, en de scores zijn de mean IoU en de mAP uit [Een detector beoordelen: mean average precision](/course/open/cv-map?lang=nl). Zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) werd waarvan elk netwerk leert gekozen op de validatiescènes, en werden de testscènes één keer gemeten; de error analysis keek ook naar de validatiescènes, zodat wat je daarna probeert daar gekozen wordt. Je project `cnn` heeft nu `harder.py` en `shares.py`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_harder, result_shares, choice_next, explain_report):
    L.wrap_up(result_harder, result_shares, choice_next, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
