# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-prepare-photo. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-prepare-photo", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) en over [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Een model is getraind op een feature die is gestandaardiseerd met gemiddelde 50 en standaardafwijking 10. Een nieuw voorbeeld heeft de waarde 70. Welk getal gaat het model in?",
        options={
            "right": "2",
            "raw": "70",
            "own": "0",
            "seven": "7",
        },
        correct="right",
        why={
            "right": "(70 − 50) / 10 = 2: een nieuw voorbeeld standaardiseer je met het gemiddelde en de standaardafwijking van de training data, de getallen waarmee het model zijn weights leerde.",
            "raw": "Het model leerde op gestandaardiseerde getallen, die rond 0 liggen, en 70 ligt daar ver buiten. Standaardiseer met de getallen van de training data: (70 − 50) / 10.",
            "own": "Eén voorbeeld in zijn eentje is zijn eigen gemiddelde: als je zijn eigen gemiddelde eraf haalt, blijft er altijd 0 over, en dan zegt het niets meer. Gebruik het gemiddelde en de standaardafwijking van de training data: (70 − 50) / 10 = 2.",
            "seven": "70 / 10 deelt wel, maar vergeet eerst het gemiddelde eraf te halen: (70 − 50) / 10 = 2.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def quick_check_size():
    quick_size = L.choice(
        "quick-size",
        question="`img` heeft shape `(400, 600, 3)`. Welke shape heeft `np.array(Image.fromarray(img).resize((300, 200)))`?",
        options={
            "right": "`(200, 300, 3)`",
            "swapped": "`(300, 200, 3)`",
            "same": "`(400, 600, 3)`",
            "flat": "`(300, 200)`",
        },
        correct="right",
        why={
            "right": "Pillow schrijft een grootte als (breedte, hoogte), en NumPy een shape als (rijen, kolommen): breedte 300 en hoogte 200 geven 200 rijen van 300 kolommen, met de drie kleuren erbij.",
            "swapped": "Dan lees je `(300, 200)` als rijen en kolommen. Pillow schrijft de breedte eerst: 300 kolommen en 200 rijen, shape `(200, 300, 3)`.",
            "same": "`resize` geeft een nieuw beeld van de grootte waar je om vraagt: 300 bij 200, shape `(200, 300, 3)`.",
            "flat": "`resize` houdt de drie getallen per pixel. En Pillow schrijft de breedte eerst: shape `(200, 300, 3)`.",
        },
    )
    quick_size
    return (quick_size,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een pretrained beeldmodel leerde van meer dan een miljoen foto's, die allemaal op dezelfde manier waren klaargemaakt: even groot, uit het midden geknipt, en elke kleur geschaald met dezelfde getallen. Iets anders heeft het nooit als input gezien. Geef je het een foto van een andere grootte, of met kleuren van 0 tot en met 255, dan stopt het met een fout, of geeft het vol zelfvertrouwen een verkeerd antwoord. Dus voordat een pretrained model naar jouw foto kijkt, maak je die klaar zoals zijn trainingsfoto's werden klaargemaakt. De model card of de code bij het model zegt hoe; voor de meeste beeldmodellen die op ImageNet zijn getraind, een verzameling foto's waar de volgende les op terugkomt, is het het recept van deze les.

    **Een voorbeeld uit het dagelijks leven.** Een recept in grammen, afgewogen op een weegschaal in ounces: elk getal klopt op zijn eigen schaal, en toch mislukt de taart.
    """)
    return


@app.cell(hide_code=True)
def predict_squash():
    guess_squash = L.guess(
        "squash",
        question="De koffiefoto is 600 pixels breed en 400 hoog, en het model wil 224 bij 224. Pillow kan hem meteen naar die grootte platdrukken, met `resize((224, 224))`. Wat gebeurt er met het ronde kopje?",
        options={
            "round": "Het blijft rond",
            "narrow": "Het wordt hoger dan breed",
            "wide": "Het wordt breder dan hoog",
            "cut": "De helft wordt eraf geknipt",
        },
        reveal="**Het wordt hoger dan breed.** De breedte gaat van 600 naar 224, 0,37 keer zoveel, en de hoogte van 400 naar 224, 0,56 keer zoveel: alles wordt van opzij samengeknepen. Er wordt niets afgeknipt, maar elke vorm verandert, en het model leerde van foto's waarop kopjes rond zijn. Daarom houdt het recept hieronder de verhouding, en knipt het in plaats daarvan het midden eruit.",
    )
    guess_squash
    return (guess_squash,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De meeste beeldmodellen die op ImageNet zijn getraind, verwachten een foto die in dezelfde stappen is klaargemaakt. Eerst wordt de korte zijde 256 pixels, en de andere zijde groeit of krimpt mee, zodat er niets wordt platgedrukt: de koffiefoto van 600 bij 400 wordt 384 bij 256, want 600 × 256 / 400 = 384. Dan knipt een **center crop** het middelste vierkant van 224 bij 224 eruit. Van de 256 rijen passen er 32 niet: 16 gaan boven het vierkant weg en 16 eronder, dus het begint bij rij 16. Van de 384 kolommen passen er 160 niet, dus het begint bij kolom 80. Wat aan de randen van de foto ligt, raak je kwijt; de foto's waar het model van leerde, zijn op dezelfde manier geknipt, dus wat ze laten zien, ligt meestal in het midden.

        Dan de getallen. De kleuren worden gedeeld door 255, zodat ze van 0 tot en met 1 lopen, en dan komt **per-channel normalization**: haal per kleur het gemiddelde eraf en deel door de standaardafwijking van de trainingsfoto's van ImageNet, gemiddelde `(0.485, 0.456, 0.406)` en standaardafwijking `(0.229, 0.224, 0.225)` voor rood, groen en blauw. Dat is standaardisatie, zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl), met één gemiddelde en één standaardafwijking per kleur, en met de getallen van de training data, niet die van de nieuwe foto. Een pixel met de gemiddelde kleur van ImageNet wordt (0, 0, 0); een pixel die roder is dan gemiddeld, krijgt een rood boven 0.

        Tot slot de shape. Een beeld in NumPy heeft shape `(rijen, kolommen, kleuren)`, en de beeldmodellen van PyTorch verwachten `(batch, kleuren, rijen, kolommen)`: de kleuren vooraan, en een batch ervoor. `t.permute(2, 0, 1)` zet de assen van een tensor in een nieuwe volgorde: as 2, de kleuren, vooraan, dan as 0 en as 1. `t.unsqueeze(0)` zet er een nieuwe as van lengte 1 voor: een batch van één foto, shape `(1, 3, 224, 224)`. En de weights van het model zijn float32, dus de foto ook.
        """),
        L.watch_out("`reshape(1, 3, 224, 224)` geeft ook de goede shape, maar niet de goede foto: hij houdt de getallen in hun volgorde en knipt ze alleen opnieuw, dus het rood, groen en blauw van één pixel komen naast elkaar in één rij van de eerste kleur terecht, alsof het drie pixels zijn. `permute` verplaatst de assen zelf."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **center crop** | het middelste vierkant van een beeld, eruit geknipt met een slice | `small[16:240, 80:304]` |
        | **per-channel normalization** | per kleur een gemiddelde eraf halen en delen door een standaardafwijking: die van de trainingsfoto's | `(crop / 255 - mean) / std` |
        | **`permute`** | zet de assen van een tensor in een nieuwe volgorde | `t.permute(2, 0, 1)` |
        | **`unsqueeze`** | zet een nieuwe as van lengte 1 op een plek | `t.unsqueeze(0)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_crop():
    mo.vstack([
        mo.md("**Platgedrukt, of in verhouding verkleind en het midden eruit**"),
        mo.Html(C.picture({"squashed": "platgedrukt tot 224 × 224", "resized": "384 × 256", "crop": "het midden, 224 × 224",
                           "alt": "Drie versies van de koffiefoto op dezelfde schaal. Links platgedrukt tot 224 bij 224: het kopje is smal en hoog. In het midden 384 bij 256 met dezelfde verhouding, met links en rechts een brede donkerdere strook en boven en onder een smalle. Rechts het middelste vierkant van 224 bij 224: het kopje rond en bijna heel, het schoteltje aan de zijkanten afgesneden."})),
        mo.md(r"""
        Links de foto, meteen platgedrukt tot 224 bij 224: het kopje is smal en hoog. In het midden de foto op 384 bij 256, met dezelfde verhouding; de donkerdere stroken zijn wat de center crop wegknipt, breed links en rechts, smal boven en onder. Rechts het vierkant dat het model krijgt: het kopje rond en bijna heel, het schoteltje aan de zijkanten afgesneden.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Waarom eerst 256, en dan 224? Met de slider kies je de korte zijde van de eerste stap; de center crop knipt altijd 224 bij 224 uit het midden. Links zie je de foto op die grootte, met het deel dat wegvalt donkerder; rechts wat het model krijgt.
    """)
    return


@app.cell(hide_code=True)
def explore_side_cell():
    def _say(short):
        _w = C.size_after(600, 400, short)[0]
        _notes = {224: " Boven en onder valt niets weg, maar links en rechts een derde van de foto.",
                  256: " Boven en onder valt een smalle strook weg, en opzij een brede: de keuze van het recept. Alleen het allerbovenste stukje van de rand valt weg.",
                  320: " Er valt een duidelijk stuk van de bovenkant van de rand weg.",
                  384: " Ook de linkerkant van de rand valt weg.",
                  448: " De rand valt boven en aan beide kanten weg: alleen het midden van het kopje is over."}
        return (f"De foto wordt {_w} bij {short}; het vierkant bevat de middelste {C.percent(100 * 224 / short, ',')} procent van de "
                f"hoogte en {C.percent(100 * 224 / _w, ',')} procent van de breedte." + _notes.get(short, ""))

    def _frame(short):
        _html = C.side_frame(short, {"resized": "{w} × {h}", "crop": "het midden, 224 × 224",
                                     "alt": "Links de koffiefoto op {w} bij {h} pixels, bij elke stand even groot getekend, met alles buiten het middelste vierkant van 224 bij 224 donkerder. Rechts dat middelste vierkant: wat het model krijgt."})
        return _html, _say(short)

    explore_side = L.explore("side", "korte zijde", _frame, start=256, values=C.SIDES, role="",
                             task="Zoek de korte zijde waarbij het vierkant de hele hoogte van de foto bevat, en die waarbij de rand aan beide kanten wegvalt.")
    explore_side
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij 224 bevat het vierkant de hele hoogte van de foto; bij 256 de middelste 87,5 procent ervan, en komt het kopje wat dichterbij. Waarom dichterbij? Tijdens het leren zagen de ImageNet-modellen willekeurige stukken van hun trainingsfoto's, meestal kleiner dan de hele foto, opgeblazen tot 224 bij 224: wat erop stond, leek groter dan op een hele foto. De korte zijde op 256 en dan het middelste stuk van 224 zoomt op dezelfde manier een beetje in. Het is de gewoonte van de eerste grote ImageNet-modellen, die leerden van willekeurige vierkanten van 224 uit foto's van 256. Hoe groter de eerste stap de foto maakt, hoe dichterbij het model kijkt: bij 320 valt er een duidelijk stuk van de bovenkant van de rand weg, bij 384 ook de linkerkant, en bij 448 is alleen het midden van het kopje over.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een foto `photo`, geopend met Pillow, klaar te maken als input voor een ImageNet-model:

    1. Maak de korte zijde 256, met dezelfde verhouding: voor een foto met breedte `w` en hoogte `h` die breder is dan hoog, `small = np.array(photo.resize((int(256 * w / h), 256), Image.BILINEAR))`; voor een foto die hoger is dan breed, `(256, int(256 * h / w))`.
    2. Knip het middelste vierkant eruit: `top = int((small.shape[0] - 224) / 2)` en `left = int((small.shape[1] - 224) / 2)`, en dan `crop = small[top:top + 224, left:left + 224]`.
    3. Van 0 tot en met 1: `crop / 255`.
    4. Per kleur de getallen van ImageNet: `x = (crop / 255 - mean) / std`, met `mean = np.array([0.485, 0.456, 0.406])` en `std = np.array([0.229, 0.224, 0.225])`.
    5. Een tensor van float32, de kleuren vooraan, een batch van één: `torch.tensor(x).float().permute(2, 0, 1).unsqueeze(0)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel opent de koffiefoto en print zijn grootte. Hij maakt de korte zijde 256 en knipt het middelste vierkant eruit, stap 1 en 2, en print de shapes en waar het vierkant begint. Hij schaalt de kleuren, stap 3 en 4, en print één pixel ervoor en erna, en het gemiddelde van elke kleur over het hele vierkant. Dan maakt hij de tensor, stap 5, en print zijn shape en zijn dtype.
    """)
    return


@app.cell
def example_coffee():
    import numpy as np
    import torch
    from PIL import Image

    photo = Image.open("data/photos/coffee.png")
    print(photo.size)
    # 1. de korte zijde 256, dezelfde verhouding: 600 × 256 / 400 = 384
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    # 2. het midden, 224 bij 224
    top, left = int((256 - 224) / 2), int((384 - 224) / 2)
    crop = small[top:top + 224, left:left + 224]
    print(small.shape, crop.shape, top, left)
    # 3. en 4. van 0 tot en met 1, dan per kleur het gemiddelde en de standaardafwijking van ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (crop / 255 - mean) / std
    # .round(2): elk getal van de array op twee decimalen
    print(crop[0, 0], x[0, 0].round(2))
    # het gemiddelde over de rijen, dan over de kolommen: één per kleur
    print(x.mean(axis=0).mean(axis=0).round(2))
    # 5. een tensor van float32, de kleuren vooraan, een batch van één
    t = torch.tensor(x).float().permute(2, 0, 1).unsqueeze(0)
    print(t.shape, t.dtype)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Pillow geeft de grootte als (breedte, hoogte), 600 bij 400; NumPy de shape als (rijen, kolommen, kleuren). Na stap 1 heeft de foto 256 rijen van 384 kolommen, en het middelste vierkant begint bij rij 16 en kolom 80. De pixel linksboven, (158, 63, 24), is donker roodbruin: zijn rood komt boven het gemiddelde van ImageNet uit, 0,59, en zijn groen en blauw eronder, −0,93 en −1,39. Over het hele vierkant hebben de kleuren gemiddelde 0,54, −0,64 en −0,95, niet 0: de koffiefoto is roder en minder blauw dan de gemiddelde foto van ImageNet, en dat blijft na het schalen zo, omdat het de getallen van ImageNet gebruikt, niet die van de foto zelf. De tensor heeft de shape en het dtype die het model verwacht.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een staande foto

    Een foto is 300 pixels breed en 450 hoog: Pillow geeft zijn grootte als `(300, 450)`.
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape",
        question="Welke shape heeft de NumPy-array van deze foto na stap 1?",
        options={
            "right": "`(384, 256, 3)`",
            "landscape": "`(256, 384, 3)`",
            "squashed": "`(256, 256, 3)`",
            "crop": "`(224, 224, 3)`",
        },
        correct="right",
        why={
            "right": "De korte zijde is de breedte, 300: die wordt 256, en de hoogte 450 × 256 / 300 = 384. De grootte voor Pillow is `(256, 384)`, de breedte eerst; de shape van NumPy heeft de rijen eerst: 384 rijen van 256 kolommen, met drie kleuren.",
            "landscape": "Dat is de shape van de koffiefoto, die breder is dan hoog. Hier is de breedte, 300, de korte zijde: die wordt 256, en de hoogte 384. NumPy schrijft de rijen eerst: `(384, 256, 3)`.",
            "squashed": "Stap 1 houdt de verhouding: alleen de korte zijde wordt 256, en de andere zijde groeit mee, tot 450 × 256 / 300 = 384.",
            "crop": "Dat is na stap 2, het middelste vierkant. Na stap 1 is de breedte 256 en de hoogte 450 × 256 / 300 = 384.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_crop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het middelste vierkant

        Maak `center_crop(img, size)` af. Die geeft het middelste vierkant van `size` bij `size` van een beeld terug: van de rijen die niet passen, gaat de helft boven het vierkant weg en de helft eronder, en zo ook met de kolommen. Vervang de drie `...`. De check roept hem aan met een kleine array van 6 rijen bij 8 kolommen, met 0 tot en met 47, rij voor rij:

        ```python
        center_crop(np.arange(48).reshape(6, 8), 4)
        ```

        Dat moet `[[10, 11, 12, 13], [18, 19, 20, 21], [26, 27, 28, 29], [34, 35, 36, 37]]` geven. De check probeert ook een beeld dat hoger is dan breed, beelden met drie kleuren, en een beeld dat precies even groot is als het vierkant.
        """),
        L.task("Laat `center_crop(img, size)` het middelste vierkant van het beeld teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_crop():
    crop_code = L.code_box("crop", lines=4)
    crop_code
    return (crop_code,)


@app.cell(hide_code=True)
def check_crop(crop_code):
    result_crop = L.check("crop", crop_code.value)
    result_crop
    return (result_crop,)


@app.cell(hide_code=True)
def hints_crop():
    L.hints(
        "crop",
        [
            "Stap 2 van het recept: eerst waar het vierkant begint, dan de slice.",
            "`img.shape[0]` is het aantal rijen en `img.shape[1]` het aantal kolommen. Van de rijen passen er `img.shape[0] - size` niet; de helft daarvan gaat boven het vierkant weg, heel gemaakt met `int`.",
            "`top = int((img.shape[0] - ___) / 2)`, `left = int((img.shape[___] - size) / 2)` en `return img[top:top + ___, left:left + size]`.",
        ],
        why="Van de 6 rijen passen er 2 niet, dus het vierkant begint bij rij `int(2 / 2)`, 1; van de 8 kolommen passen er 4 niet, dus het begint bij kolom 2. `img[1:5, 2:6]` bevat de rijen 1 tot en met 4 en de kolommen 2 tot en met 5. Bij een beeld met kleuren laat de slice de derde as zoals hij is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_prepare():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De kat, klaar voor het model

        `data/photos/chelsea.png` is een foto van een kat, 451 bij 300 pixels. Het vak knipt er een staand stuk uit, 200 pixels breed en 300 hoog, alsof de foto rechtop is genomen, en doet stap 1 al: `small` heeft zijn korte zijde op 256, shape `(384, 256, 3)`, zoals in oefening 1. Het geeft je ook `mean` en `std`, de getallen van ImageNet. Maak van `crop` het middelste vierkant van `small`, stap 2, en van `x` de input van het model, met stap 3 tot en met 5: een tensor van float32 met shape `(1, 3, 224, 224)`.
        """),
        L.task("Maak van `crop` het middelste vierkant, en van `x` de kat zoals het model hem verwacht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prepare():
    prepare_code = L.code_box("prepare", lines=12)
    prepare_code
    return (prepare_code,)


@app.cell(hide_code=True)
def check_prepare(prepare_code):
    result_prepare = L.check("prepare", prepare_code.value)
    result_prepare
    return (result_prepare,)


@app.cell(hide_code=True)
def hints_prepare():
    L.hints(
        "prepare",
        [
            "Stap 2 tot en met 5 van het recept. De foto staat rechtop: waar zitten de rijen en de kolommen die niet passen?",
            "Van de 384 rijen van `small` passen er 160 niet, en van zijn 256 kolommen 32: van elk gaat de helft voor het vierkant weg. Dan de getallen: `crop / 255`, min `mean`, gedeeld door `std`. Dan een tensor ervan, float32 met `.float()`, de kleuren vooraan met `.permute` en een batch-as ervoor met `.unsqueeze`.",
            "`crop = small[80:___, ___:240]` en `x = torch.tensor((crop / 255 - ___) / ___).float().permute(2, ___, ___).unsqueeze(___)`.",
        ],
        why="De foto staat rechtop, dus de brede marge zit in de rijen: `int((384 - 224) / 2)` is 80 en `int((256 - 224) / 2)` is 16, dus `crop = small[80:304, 16:240]`. `(crop / 255 - mean) / std` schaalt elke kleur met de getallen van ImageNet: NumPy rekt `mean` en `std`, met shape `(3,)`, uit over de rijen en kolommen. `torch.tensor` maakt er een tensor van, float64 omdat hij uit NumPy komt, en `.float()` maakt hem float32. `.permute(2, 0, 1)` geeft shape `(3, 224, 224)`, en `.unsqueeze(0)` shape `(1, 3, 224, 224)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_own():
    mo.md(r"""
    ### Oefening 4. Zijn eigen getallen

    Iemand schaalt de kattenfoto met zijn eigen gemiddelde en standaardafwijking per kleur, in plaats van die van ImageNet. Elke kleur van de foto heeft dan gemiddelde 0 en standaardafwijking 1.
    """)
    return


@app.cell(hide_code=True)
def explain_own_cell():
    explain_own = L.explain(
        "own",
        question="Waarom is dat slechter voor het pretrained model? Denk aan waar het model zijn weights op leerde, en aan wat er gebeurt met een foto die roder is dan de meeste.",
        model_answer="**Het model leerde zijn weights op foto's die met de getallen van ImageNet zijn geschaald,** dus een nieuwe foto moet met dezelfde getallen worden geschaald, net als een nieuw voorbeeld in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl). **Met zijn eigen getallen krijgt elke foto in elke kleur gemiddelde 0:** een foto die roder, donkerder of blauwer is dan de meeste, ziet er niet meer zo uit. De koffiefoto had gemiddeld een rood van 0,54 en een blauw van −0,95; met zijn eigen getallen worden die allebei 0, en ziet het model niet meer dat de foto roodbruin is. **Dezelfde kleur krijgt dan in verschillende foto's andere getallen,** afhankelijk van wat er verder op de foto staat, dus kan het model er niet op rekenen dat een kleur betekent wat hij betekende toen het leerde.",
        points=[
            "je zegt dat het model leerde op foto's die met de getallen van ImageNet zijn geschaald, dus dat een nieuwe foto dezelfde getallen nodig heeft",
            "je zegt dat met zijn eigen getallen elke foto per kleur gemiddelde 0 krijgt, dus dat een rodere of donkerdere foto er niet meer zo uitziet",
            "je zegt dat dezelfde kleur dan in verschillende foto's andere getallen krijgt",
        ],
    )
    explain_own
    return (explain_own,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een pretrained model kent alleen de inputs waar het van leerde, dus een nieuwe foto maak je klaar zoals zijn trainingsfoto's: de korte zijde op maat met dezelfde verhouding, zoals in [Groter en kleiner: interpolatie](/course/open/cv-resize-interpolation?lang=nl), het middelste vierkant eruit, elke kleur gestandaardiseerd met de getallen van de trainingsfoto's, zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl), en het geheel als tensor van float32 met shape `(1, 3, 224, 224)`, klaar voor een model zoals in [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl). Andere modellen willen andere getallen, zoals 384 bij 384 pixels, of een gemiddelde van 0,5 voor elke kleur: de model card zegt welke, en dezelfde stappen doen het met die getallen. In de volgende les, *Een beeldclassifier uitproberen: top-1 en top-5*, kijkt een ResNet-18 die op ImageNet is getraind naar de koffiefoto en de kat, zo klaargemaakt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, explain_own, result_crop, result_prepare):
    L.wrap_up(choice_shape, result_crop, result_prepare, explain_own)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
