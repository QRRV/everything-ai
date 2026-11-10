# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-saliency. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-saliency", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) en [In een CNN kijken: filters en feature maps](/course/open/cv-inside-cnn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grad():
    quick_grad = L.choice(
        "quick-grad",
        question="`x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)`, dan `s = (x * x).sum()` en `s.backward()`. Wat is `x.grad`?",
        options={
            "sum": "`tensor(14.)`: de waarde van `s`",
            "right": "`tensor([2., 4., 6.])`: een helling voor elk getal van `x`",
            "square": "`tensor([1., 4., 9.])`: elk getal van `x` in het kwadraat",
        },
        correct="right",
        why={
            "sum": "14 is `s` zelf, 1 + 4 + 9. `x.grad` bevat hellingen, één voor elk getal van `x`, in zijn shape. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "square": "Dat zijn de getallen van `x * x`. De helling van $x^2$ is $2x$, dus is `x.grad` 2, 4 en 6. Zie [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl).",
            "right": "De helling van $x^2$ is $2x$, één voor elk getal van `x`, in zijn shape. Vandaag is `x` een foto, en bevat `x.grad` een helling voor elk van zijn 150.528 getallen.",
        },
    )
    quick_grad
    return (quick_grad,)


@app.cell(hide_code=True)
def quick_check_pixels():
    quick_pixels = L.choice(
        "quick-pixels",
        question="In ResNet-18 ziet één getal van een map van `layer4` 435 bij 435 pixels, meer dan de foto van 224 bij 224, en de laatste layer krijgt het gemiddelde van elke map van `layer4`. Welke pixels van de koffiefoto kunnen de logit van espresso veranderen?",
        options={
            "cup": "Alleen de pixels van het kopje",
            "right": "Elke pixel van de foto",
            "middle": "Alleen de middelste 7 bij 7 pixels, één voor elk getal van een map van `layer4`",
        },
        correct="right",
        why={
            "cup": "Het netwerk weet niet waar het kopje is voordat het kijkt: elk getal van `layer4` ziet de hele foto, en de logit wordt van al die getallen gemaakt. Zie [In een CNN kijken](/course/open/cv-inside-cnn?lang=nl).",
            "middle": "Een map van `layer4` is 7 bij 7, maar elk van zijn getallen ziet 435 bij 435 pixels, meer dan de hele foto. Zie [In een CNN kijken](/course/open/cv-inside-cnn?lang=nl).",
            "right": "Elke pixel bereikt de logit, maar niet allemaal even sterk. Vandaag zoek je uit welke pixels hem het sterkst bereiken.",
        },
    )
    quick_pixels
    return (quick_pixels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    ResNet-18 noemt de koffiefoto espresso, met 0,85. Maar een kans zegt niet waar hij op rust: de koffie, het kopje, of het rode schoteltje eronder. Een classifier kan gelijk hebben om de verkeerde reden. Lapuschkin en zijn collega's (2019) bekeken een classifier die paarden herkende op de foto's van PASCAL VOC 2007, en tekenden over elke foto, met een ander soort map dan die van vandaag, waar zijn antwoord op rustte. Hij keek naar een bronvermelding, een regel tekst in een hoek, die op ongeveer één op de vijf paardenfoto's stond. Op een foto van een auto geplakt maakte die bronvermelding de auto tot een paard. Simonyan, Vedaldi en Zisserman (2014) maakten de eenvoudigste map van dat soort voor een CNN, met één backward pass: de grootte van de helling van een klassescore naar elke pixel, die laat zien "which pixels need to be changed the least to affect the klasse score the most", welke pixels het minst hoeven te veranderen om de score het meest te veranderen. Vandaag maak je die map voor ResNet-18, en test je wat hij belooft.
    """)
    return


@app.cell(hide_code=True)
def predict_change():
    guess_change = L.guess(
        "change-answer",
        question="ResNet-18 noemt de tekening van fruit een balloon, met 0,73. Je verandert 5% van zijn pixels, 2.509 van de 50.176, elke kleur 1 op zijn schaal van 0 tot 255: te weinig om te zien. Kan dat het antwoord veranderen?",
        options={
            "no": "Nee: een verandering van 1 op 255 is veel te klein om iets uit te maken",
            "right": "Ja, als je de goede pixels kiest en elke kleur de goede kant op verandert",
            "any": "Ja, elke 5% van de pixels zou dat doen",
        },
        reveal="**Ja, met de goede pixels.** De 5% van de pixels met de grootste hellingen, elke kleur 1 tegen zijn helling in veranderd, brengen balloon omlaag naar 0,40, en ping-pong ball komt bovenaan, met 0,44. 5% willekeurige pixels, op dezelfde manier veranderd, laten balloon bovenaan, met 0,63. De hellingen komen uit één backward pass; hun grootste grootte per pixel, als plaatje getekend, is de map van vandaag. Met de slider in *Het idee* verander je ze meer.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    _frames = C.step_frames({
        "salient": "5% met de hoogste saliency, {step} veranderd",
        "random": "5% willekeurige pixels, {step} veranderd",
        "alt": "Twee groepen van drie balken. Na een verandering van {step} op de 5% pixels met de hoogste saliency: {salient}. Na een verandering van {step} op 5% willekeurige pixels: {random}.",
    }, L.lang)
    _steps = L.explore("steps", "verandering per kleur", lambda _v: (_frames[int(_v)], C.step_sentence(int(_v), L.lang)),
                       start=1, values=C.STEPS, role="",
                       task="Schuif van 0 naar 12: vanaf welke verandering is balloon niet meer de beste gok, op de 5% pixels met de hoogste saliency? En op 5% willekeurige pixels?")
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een helling voor elke pixel.** De logit van een klasse $c$, zijn score $z_c$, is één getal, berekend uit een foto $\mathbf{x}$ van 3 × 224 × 224 = 150.528 getallen. Net als in [Twee parameters: de gradient](/course/open/math-gradient?lang=nl) heeft zijn gradient een helling voor elk van die getallen, de partiële afgeleide $\partial z_c / \partial x_i$: hoeveel de score verandert als dat ene getal een beetje verandert. Dicht bij de foto is de score in elke richting bijna een rechte lijn:

        $$z_c(\text{veranderde foto}) \approx z_c(\mathbf{x}) + \sum_i \frac{\partial z_c}{\partial x_i} \times (\text{verandering van } x_i)$$

        Een kleine verandering van de getallen met de grootste hellingen beweegt de score dus het meest.

        **Autograd geeft alle hellingen tegelijk.** De foto is een tensor die PyTorch zonder opnemen maakte, dus een leaf tensor, en `x.requires_grad = True` zet het opnemen voor hem aan, zoals `requires_grad=True` deed in [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl). `resnet(x)[0, c]` is de logit van klasse `c` voor foto 0 van de batch, één getal, en `.backward()` vanaf dat getal vult `x.grad`, in de shape van `x`: (1, 3, 224, 224). Dat is één keer vooruit door het netwerk en één keer terug. De centrale differentie van [Gradient checking](/course/open/nn-gradient-check?lang=nl), één getal tegelijk, zou twee keer door het netwerk moeten voor elk van de 150.528 getallen: 301.056 keer. De weights van de ResNet-18 van de course houden geen hellingen bij, dus krijgt alleen `x` een `.grad`. Haal hem buiten `torch.no_grad()` door het model: daarbinnen neemt PyTorch niets op.

        **De logit, niet de kans.** De kans op een klasse kan stijgen doordat andere klassen dalen: softmax deelt door allemaal. De hellingen van een kans mengen er elke andere klasse in, dus namen Simonyan en zijn collega's de score zelf.

        **Een saliency map.** Voor een plaatje, één getal per pixel: de grootste grootte van zijn drie hellingen, één per kleur.

        $$M_{ij} = \max_k \left| \frac{\partial z_c}{\partial x_{kij}} \right|$$

        De grootte, zonder het teken, want een pixel die de score verlaagt als hij lichter wordt, telt even zwaar als een die hem verhoogt. Dat is de **saliency map**, 224 bij 224: `np.abs(x.grad[0].numpy()).max(axis=0)`. `x.grad[0]` heeft de hellingen van foto 0, met shape (3, 224, 224); `.numpy()` maakt er een NumPy-array van, zoals in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), en `.max(axis=0)` neemt de grootste langs axis 0, de kleuren. Een paar pixels hebben veel grotere hellingen dan de rest, dus is een getekende map wit vanaf de 1% grootste: `vmax=np.quantile(sal, 0.99)`, het kwantiel van [Data beschrijven](/course/open/stats-describe?lang=nl).

        **Houdt een map zijn belofte?** Verander alleen zijn lichtste pixels, een beetje: elke kleur 1 op zijn schaal van 0 tot 255, tegen zijn helling in, dus omlaag waar de helling positief is en omhoog waar hij negatief is. Klopt de map, dan verandert het antwoord eerder dan wanneer evenveel willekeurige pixels op dezelfde manier veranderen. De slider doet dat op de tekening van fruit, die ResNet-18 een balloon noemt: met de 5% pixels met de hoogste saliency in de map van balloon, 2.509 van de 50.176, of met 2.509 willekeurige, elke kleur zoveel veranderd als je kiest.
        """),
        _steps,
        mo.md(r"""
        Een verandering van 1 is te klein om te zien. Op de 5% pixels met de hoogste saliency brengt hij balloon van 0,73 naar 0,40, en komt ping-pong ball bovenaan, met 0,44. Op 5% willekeurige pixels, op dezelfde manier veranderd, blijft balloon bovenaan, met 0,63; pas een verandering van 4 neemt hem de eerste plaats af. De map houdt zijn belofte: zijn lichtste pixels veranderen het antwoord eerder dan willekeurige pixels. Maar de rechte lijn geldt alleen dicht bij de foto. Op de koffiefoto verlaagt een verandering van 1 op de 5% pixels met de hoogste saliency in de map van espresso de logit van espresso met 4,11, waar de rechte lijn 7,26 belooft. Een verandering van 2 brengt de logit van 15,62 naar 10,54, maar een verandering van 8 slechts naar 11,27: verder weg buigt de score af.
        """),
        L.watch_out("Een saliency map heeft geen teken. Een pixel leegmaken, hem de gemiddelde kleur van ImageNet geven, 0 na het klaarmaken, brengt elk van zijn kleuren naar 0: sommige omlaag, andere omhoog, en de map zegt niet of dat de score verhoogt of verlaagt. Maak de 5% pixels met de hoogste saliency in de map van de kat voor Egyptian cat leeg, en de logit van Egyptian cat daalt niet: hij stijgt, van 19,65 naar 20,83. De hellingen met hun teken zagen dat aankomen: de rechte lijn geeft een stijging van 1,66. Lees een saliency map niet als de pixels die een model nodig heeft."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **saliency map** | één getal per pixel: hoe sterk een klassescore verandert als die pixel een beetje verandert, de grootste grootte van zijn hellingen over de kleuren | `np.abs(x.grad[0].numpy()).max(axis=0)` |
        | $M_{ij}$ | de saliency van de pixel in rij $i$ en kolom $j$ | `sal[i, j]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zie je welke pixels het antwoord van een classifier veranderen:

    1. Maak de foto klaar zoals voor het model, een tensor `x` met shape (1, 3, 224, 224), en zet het opnemen aan: `x.requires_grad = True`.
    2. Haal hem door het model, niet binnen `torch.no_grad()`, en neem de logit van de klasse die je wilt uitleggen: `model(x)[0, c]`, niet zijn kans.
    3. Ga terug vanaf dat ene getal, `.backward()`: `x.grad` heeft een helling voor elke pixel en kleur. Zet vóór een tweede map van dezelfde `x` eerst `x.grad = None`: `.grad` telt op.
    4. De saliency map: per pixel de grootste grootte van zijn hellingen over de kleuren, `np.abs(x.grad[0].numpy()).max(axis=0)`, met shape (224, 224).
    5. Teken hem, wit vanaf de 1% grootste. Om hem te testen verander je zijn lichtste pixels een beetje en kijk je of het antwoord eerder verandert dan met willekeurige pixels: de demo doet dat met je functies.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel onder de download maakt de kat klaar zoals in [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), met het opnemen aan, en neemt de logit van klasse 285, Egyptian cat, de beste gok van ResNet-18 voor de kat. Hij print de score, gaat ervandaan terug, en print de shape van `x.grad`. Dan maakt hij de map, en print hij zijn shape, zijn grootste getal en waar dat staat: `argmax` telt de pixels rij voor rij, 224 per rij, dus geeft `// 224` de rij en `% 224` de kolom. Als laatste tekent hij de map. De commentaren met `#` zeggen welke stap van het recept elk deel is. Voer hem uit zoals hij is; verander dan `c` waar het commentaar dat zegt, in 967, espresso, en voer hem opnieuw uit. `resnet` is ResNet-18, die de cel hieronder laadt zodra hij gedownload is.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="ResNet-18 downloaden, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 staat nog niet op je computer. Als je [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl) op deze computer hebt gedaan, staat hij er wel: dan vindt deze les hem. Anders downloadt de knop hem één keer, 47 MB, van Hugging Face. Het uitgewerkte voorbeeld, de maps van de drie foto's en de demo hebben hem nodig; de slider in *Het idee* en de oefeningen niet. Terwijl hij downloadt, wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**ResNet-18 downloaden** · {done} van {total} MB",
        "no_net": "*De download lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus heeft de course het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 47 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
        "here": "*ResNet-18 staat op je computer: 11,7 miljoen parameters, met de weights van torchvision, getraind op ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_saliency(resnet):
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
    # 2. de logit van één klasse: 285 is Egyptian cat
    c = 285  # verander: 967
    score = resnet(x)[0, c]
    print(score)
    # 3. terug vanaf dat ene getal: een helling voor elke pixel en kleur
    score.backward()
    print(x.grad.shape)
    # 4. de map: per pixel de grootste grootte van zijn hellingen over de kleuren
    g = x.grad[0].numpy()
    sal = np.abs(g).max(axis=0)
    print(sal.shape, sal.max())
    top = int(sal.argmax())
    print(top // 224, top % 224)
    # 5. teken hem: zwart is 0, wit de 1% grootste
    plt.imshow(sal, cmap="gray", vmin=0, vmax=np.quantile(sal, 0.99))
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De score van Egyptian cat is 19,6482, en hij wordt geprint met `grad_fn=<SelectBackward0>`: PyTorch heeft opgenomen hoe hij uit `x` ontstond. `x.grad` heeft de shape van `x`, (1, 3, 224, 224). De map heeft shape (224, 224), en zijn grootste getal, 0,2192, staat op rij 163 en kolom 109, links van de neus. Met `c = 967`, espresso, is de score maar 0,0781: ResNet-18 ziet geen espresso in de kat. Toch heeft elke pixel ook een helling voor espresso, en de grootste, 0,1059, staat op rij 174 en kolom 134, net boven de roze neus.
    """)
    return


@app.cell(hide_code=True)
def text_before_maps():
    mo.md(r"""
    De drie foto's van [Een beeldclassifier uitproberen](/course/open/cv-try-classifier?lang=nl), elk met zijn map voor de beste gok van ResNet-18: Egyptian cat, espresso en balloon. Elke map is wit vanaf zijn eigen 1% grootste getallen.
    """)
    return


@app.cell(hide_code=True)
def maps_cell(resnet):
    _maps = {_photo: C.course_saliency(resnet, C.prepared(_photo), C.TOP[_photo]) for _photo in C.TOP}
    mo.Html(C.maps_picture(_maps, {
        "names": {"cat": "de kat", "coffee": "de koffie", "fruit": "de tekening van fruit"},
        "map": "map van {label}",
        "alt": "Bovenste rij: de kat, de koffie en de tekening van fruit. Onderste rij: hun saliency maps, grijze spikkels op zwart, lichter rond de ogen en de neus van de kat, in de koffie, en op het fruit.",
    }))
    return


@app.cell(hide_code=True)
def text_after_maps():
    mo.md(r"""
    Alle drie de maps zijn gespikkeld: naburige pixels kunnen sterk verschillen, en een map is niet een paar lichte vlekken: de 10% grootste getallen hebben maar 26% van de som van de map van de kat, 34% van die van de koffie en 29% van die van het fruit. Maar ze verzamelen zich wel: op de kat rond de ogen en de neus, op de koffiefoto in de koffie, en op de tekening op het fruit, waar de allerlichtste pixel het lichte plekje op de sinaasappel is. Het fruit beslaat 32% van het vierkant dat ResNet-18 ziet, en heeft 54% van de 5% lichtste pixels van de map van balloon; van de map van orange 75%. Een saliency map is een eerste uitleg, met veel ruis: hij laat zien waar de score gevoelig is, pixel voor pixel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_brighter():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke pixel is lichter?
    """)
    return


@app.cell(hide_code=True)
def choice_brighter_cell():
    choice_brighter = L.choice(
        "brighter",
        question="In de hellingen van een foto heeft pixel A 0,02, 0,01 en -0,01 voor rood, groen en blauw; pixel B heeft -0,05, 0,00 en 0,03. Welke is lichter in de saliency map, en met welk getal?",
        options={
            "a": "A, met 0,02: zijn grootste helling",
            "right": "B, met 0,05: de grootste van de groottes van zijn hellingen",
            "sum": "B, met 0,08: de groottes van zijn drie hellingen opgeteld",
        },
        correct="right",
        why={
            "a": "0,02 is de grootste van de hellingen van A met hun teken. De map neemt eerst de grootte: het rood van B, -0,05, verandert de score meer dan elke kleur van A, alleen de andere kant op, want lichter rood verlaagt hem.",
            "sum": "0,05 + 0,00 + 0,03 = 0,08 is ook een map, maar niet die van deze les: Simonyan en zijn collega's nemen de grootste grootte van de drie hellingen van een pixel, 0,05.",
            "right": "Zonder het teken, want een pixel die de score verlaagt als hij lichter wordt, telt even zwaar als een die hem verhoogt. Het rood van B verandert de score het meest, 0,05 per eenheid, tegen 0,02 voor het rood van A.",
        },
    )
    choice_brighter
    return (choice_brighter,)


@app.cell(hide_code=True)
def text_exercise_saliency():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De saliency map

        Maak `saliency(model, x, label)` af: de saliency map van de foto `x`, met shape (1, 3, H, W), voor de klasse `label`, als NumPy-array met shape (H, W): per pixel de grootste grootte van de hellingen van de logit van `label` over zijn 3 kleuren. De check gebruikt een ResNet-18 met willekeurige weights en kleine foto's, van 32 bij 32 en 40 bij 48, dus hij heeft geen download nodig. Hij vraagt ook twee maps van één foto, voor twee klassen, na elkaar: denk eraan dat `.grad` optelt. Als het klopt, bewaart hij je functie in je project, als `work/projects/cnn/saliency.py`.
        """),
        L.task("Laat `saliency` de map van een foto voor een klasse geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_saliency():
    box_saliency = L.code_box("saliency", lines=8)
    box_saliency
    return (box_saliency,)


@app.cell(hide_code=True)
def check_saliency(box_saliency):
    result_saliency = L.check("saliency", box_saliency.value)
    result_saliency
    return (result_saliency,)


@app.cell(hide_code=True)
def hints_saliency():
    L.hints(
        "saliency",
        [
            "Zet het opnemen aan voor de foto, `x.requires_grad = True`, en wis een oude helling, `x.grad = None`. Dan de logit: `model(x)` heeft shape (1, klassen); welk getal is de logit van `label` voor foto 0?",
            "`score.backward()` vult `x.grad`, met shape (1, 3, H, W). Neem de hellingen van foto 0, maak er een NumPy-array van zoals in *Het idee*, en neem per pixel de grootste grootte langs axis 0, de kleuren.",
        ],
        why="Eén backward pass vanaf één logit geeft een helling voor elke pixel en kleur, omdat de foto een leaf tensor is die wordt opgenomen. Zonder `x.grad = None` telt een tweede aanroep op dezelfde foto zijn hellingen op bij die van de eerste, zoals `.grad` dat bij de weights doet in een trainingsstap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_salient():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De pixels met de hoogste saliency

        Maak `most_salient(sal, share)` af: een array met True en False in de shape van de map, True voor het deel van de pixels met de hoogste saliency, zoals de 5% die de slider veranderde bij een deel van 0,05. `np.quantile` uit [Data beschrijven](/course/open/stats-describe?lang=nl) vindt het getal waar een deel van de map onder ligt. De check geeft je functie verzonnen maps. Als het klopt, bewaart hij je functie in je project, als `work/projects/cnn/salient.py`.
        """),
        L.task("Laat `most_salient` het lichtste deel van een map kiezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_salient():
    box_salient = L.code_box("most-salient", lines=5)
    box_salient
    return (box_salient,)


@app.cell(hide_code=True)
def check_salient(box_salient):
    result_salient = L.check("most-salient", box_salient.value)
    result_salient
    return (result_salient,)


@app.cell(hide_code=True)
def hints_salient():
    L.hints(
        "most-salient",
        [
            "De lichtste 5% zijn de pixels op of boven het getal waar 95% van de map onder ligt. Welk kwantiel is dat, voor elk deel?",
            "`np.quantile(sal, q)` is het getal waar een deel `q` van de map onder ligt. Een vergelijking van een array met één getal geeft True of False voor elke pixel: welke vergelijking houdt de lichte?",
        ],
        why="Een deel van de pixels, geen vast getal: maps verschillen in schaal. `sal >= 0.1` kiest geen enkele pixel bij een ResNet-18 met willekeurige weights, waarvan de hellingen tienduizendsten zijn, en minder dan 1% bij de getrainde: 0,27% van de map van de kat, 0,82% van die van de koffie en 0,12% van die van het fruit. Het kwantiel volgt elke map.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De knop gebruikt je project: je `saliency` maakt de map van de foto die je kiest, voor de beste gok van ResNet-18, en je `most_salient` kiest zijn 5% lichtste pixels. Dan verandert de course elke kleur van die pixels 1 tegen zijn helling in, en evenveel willekeurige pixels op dezelfde manier, en zegt wat dat met de beste gok doet. Hij heeft ResNet-18 nodig, uit het uitgewerkte voorbeeld, en oefening 2 en 3.
    """)
    return


@app.cell(hide_code=True)
def photo_choice_cell():
    photo_choice = mo.ui.radio(options={"de kat": "cat", "de koffie": "coffee", "de tekening van fruit": "fruit"}, value="de kat", label="Foto", inline=True)
    look_button = mo.ui.run_button(label="Maak de map met je functies")
    mo.vstack([photo_choice, look_button])
    return look_button, photo_choice


@app.cell(hide_code=True)
def demo_saliency(resnet, look_button, photo_choice, result_saliency, result_salient):
    resnet, result_saliency, result_salient  # opnieuw uitvoeren als oefening 2 of 3 je module bewaart
    if not look_button.value:
        _out = mo.md("*Druk op de knop om de map te maken.*")
    else:
        _result = C.demo(photo_choice.value)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3: dan staan je `saliency` en je `most_salient` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "saliency": "*Je `saliency` in `work/projects/cnn/saliency.py` stopt met een fout, of geeft geen map van 224 bij 224. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan opnieuw.*",
                "salient": "*Je `most_salient` in `work/projects/cnn/salient.py` stopt met een fout, of geeft niet voor elke pixel True of False. Heb je zijn bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan opnieuw.*",
            }[_result])
        else:
            _out = mo.vstack([
                mo.Html(C.demo_picture(_result, {"mask": "je 5% pixels met de hoogste saliency", "map": "je saliency map",
                                                 "alt": "De foto met zijn 5% pixels met de hoogste saliency in kleur en de rest vervaagd, en ernaast de saliency map in grijs."})),
                mo.md(C.demo_sentence(_result, photo_choice.value, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_alike():
    mo.md(r"""
    ### Oefening 4. Twee katten, één map?
    """)
    return


@app.cell(hide_code=True)
def choice_alike_cell():
    choice_alike = L.choice(
        "alike",
        question="Op de kat zijn 66% van de 5% lichtste pixels van de map van Egyptian cat ook bij de 5% lichtste van de map van tiger cat. Van de map van espresso 17%; twee willekeurige trekkingen van 5% delen er ongeveer 5%. Wat zegt dat?",
        options={
            "detector": "De map van Egyptian cat laat zien wat de kat tot een Egyptian cat maakt",
            "right": "Beide kattenklassen worden door ongeveer dezelfde pixels bewogen: de map laat zien waar een score gevoelig is, niet wat de twee onderscheidt",
            "noise": "De map van espresso is ruis, want er staat geen espresso op de foto",
        },
        correct="right",
        why={
            "detector": "Dan zou de map van tiger cat andere pixels aanwijzen, en twee derde van de lichtste pixels is hetzelfde: de ogen en de neus bewegen beide kattenklassen. Wat Egyptian cat van tiger cat scheidt, kan de map van één klasse niet laten zien.",
            "noise": "17% is meer dan de 5% die twee willekeurige trekkingen delen, en zijn lichtste pixel ligt ook bij de neus. Elke pixel heeft een helling voor elke klasse: de map van espresso wijst de pixels aan waarvan een verandering de score van espresso het meest beweegt, en sommige daarvan doen ook voor een kat ertoe.",
            "right": "Een saliency map beantwoordt één vraag: welke pixels deze score het meest veranderen, bij deze foto. Voor klassen die op elkaar lijken zijn dat ongeveer dezelfde pixels. De volgende les maakt een grovere map, uit de laatste feature maps van ResNet-18 in plaats van uit de pixels.",
        },
    )
    choice_alike
    return (choice_alike,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De gradient van [Twee parameters: de gradient](/course/open/math-gradient?lang=nl), nu naar de 150.528 getallen van een foto in plaats van naar twee parameters, en de autograd van [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl), die ze allemaal geeft met één backward pass vanaf één logit; ook hier telt zijn `.grad` op. Elke pixel bereikt de logit via de receptive fields van [In een CNN kijken](/course/open/cv-inside-cnn?lang=nl), en de map laat zien welke hem het sterkst bereiken, bij deze foto. Je project `cnn` heeft nu `saliency.py` en `salient.py`. In de volgende les, *Grad-CAM: waar een CNN keek*, maak je een grovere map, uit de laatste feature maps van ResNet-18.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_brighter, result_saliency, result_salient, choice_alike):
    L.wrap_up(choice_brighter, result_saliency, result_salient, choice_alike)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
