# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-augment-photos. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-augment-photos", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Data augmentation](/course/open/nn-data-augmentation?lang=nl) en [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mirror():
    quick_mirror = L.choice(
        "quick-mirror",
        question="In *Data augmentation* hoorde een spiegelbeeld bij de veranderingen die een cijfer kunnen veranderen of onleesbaar maken. Waarom?",
        options={
            "pixels": "Een spiegelbeeld verplaatst elke pixel, en daar kan een netwerk niet van leren",
            "right": "Een gespiegelde 2, 3 of 7 is geen cijfer dat iemand schrijft",
            "mlp": "Een spiegelbeeld helpt alleen een CNN, geen MLP",
        },
        correct="right",
        why={
            "pixels": "Eén pixel opzij verplaatst ook elke pixel, en daarbij bleef het cijfer hetzelfde. Waar het om gaat, is of het label blijft kloppen: een gespiegelde 3 is geen cijfer dat iemand schrijft. Zie [Data augmentation](/course/open/nn-data-augmentation?lang=nl).",
            "mlp": "Of het label bij een verandering blijft kloppen, hangt niet van het netwerk af: een gespiegelde 3 is geen cijfer dat iemand schrijft, voor een MLP en voor een CNN. Zie [Data augmentation](/course/open/nn-data-augmentation?lang=nl).",
            "right": "Handgeschreven cijfers hebben een links en een rechts. Foto's van auto's, paarden en schepen niet: gespiegeld zijn het foto's als alle andere, en vandaag is een spiegelbeeld een van de veranderingen.",
        },
    )
    quick_mirror
    return (quick_mirror,)


@app.cell(hide_code=True)
def quick_check_gap():
    quick_gap = L.choice(
        "quick-gap",
        question="Na 20 epochs las je ColorCNN 92 tot 97% van zijn 6.000 trainingsfoto's goed, en 52 tot 54% van de testfoto's. Na epoch 3 tot 6 steeg zijn validation loss weer. Wat gebeurde er?",
        options={
            "epochs": "Hij had te weinig epochs om de testfoto's te leren",
            "small": "Hij heeft te weinig parameters voor zulke moeilijke foto's",
            "right": "Hij leerde zijn trainingsfoto's uit zijn hoofd: details die niet voor nieuwe foto's gelden",
        },
        correct="right",
        why={
            "epochs": "Een netwerk leert nooit van de testfoto's: die meten alleen. En meer epochs maakten het daar slechter: na epoch 3 tot 6 steeg de validation loss. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "small": "Hij had genoeg parameters om 92 tot 97% van 6.000 foto's goed te lezen: hij kon veel details leren. Alleen golden die details niet voor nieuwe foto's. Zie [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl).",
            "right": "Vandaag laat elke epoch hem zijn trainingsfoto's een beetje anders zien, zodat hij ze minder makkelijk uit zijn hoofd leert.",
        },
    )
    quick_gap
    return (quick_gap,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je ColorCNN leerde zijn 6.000 trainingsfoto's uit zijn hoofd. In [Data augmentation](/course/open/nn-data-augmentation?lang=nl) zorgden willekeurige stapjes van een pixel ervoor dat een netwerk voor cijfers minder uit zijn hoofd leerde. Foto's laten meer veranderingen toe dan cijfers: een gespiegelde auto is nog steeds een auto, en een andere dag geeft ander licht. De ResNet-paper uit je project trainde op CIFAR-10 met twee veranderingen van vandaag, een spiegelbeeld en een random crop (He et al., 2015), en AlexNet, het CNN dat in 2012 ImageNet won, spiegelde zijn foto's, sneed er willekeurige stukken uit en veranderde hun kleuren een beetje. Vandaag schrijf je de crop en de kleurverandering zelf, en train je je ColorCNN ermee.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_train():
    guess_train = L.guess(
        "train-right",
        question="Zonder veranderingen las je ColorCNN na 20 epochs 92 tot 97% van zijn trainingsfoto's goed. Nu laat elke epoch hem elke foto zien, wel of niet gespiegeld, op een willekeurige plek uitgesneden, en met net iets andere kleuren, 40 epochs lang. Hoeveel van zijn 6.000 trainingsfoto's leest hij daarna goed, zoals ze zijn?",
        options={
            "more": "Bijna allemaal, net als zonder veranderingen",
            "right": "Ongeveer twee derde",
            "half": "Ongeveer de helft, evenveel als van de testfoto's",
        },
        reveal="**Ongeveer twee derde.** Over 5 seeds, met de code van de course, leest hij 3.917 tot 4.024 van zijn 6.000 trainingsfoto's goed, 65 tot 67%: veel minder dan zonder veranderingen. Maar van de 1.000 testfoto's leest hij er 583 tot 619 goed, tegen 515 tot 543 zonder veranderingen: hij leerde minder over zijn eigen foto's, en meer over de 10 soorten.",
    )
    guess_train
    return (guess_train,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Veranderingen die laten staan wat er op een foto staat.** Data augmentation verandert elke trainingsfoto willekeurig, opnieuw elke keer dat de Dataset hem geeft, op manieren waarbij de soort blijft kloppen. Het netwerk ziet dan nooit twee keer precies dezelfde foto, en details van de ene versie helpen niet bij de volgende. Voor foto's zijn er drie klassieke veranderingen.

        **Spiegelbeeld.** Links en rechts omgewisseld: een *horizontal flip*. Een auto die naar rechts rijdt, wordt een auto die naar links rijdt, en echte foto's laten allebei zien. Een foto met shape (3, 32, 32) heeft zijn kolommen op axis 2, dus geeft `x.flip(dims=[2])` het spiegelbeeld: de kolommen in de omgekeerde volgorde. Een willekeurige helft van de foto's wordt gespiegeld. `x.flip(dims=[1])` zet de rijen in de omgekeerde volgorde: ondersteboven. Dat is nog steeds een auto, maar een die echte foto's bijna nooit laten zien.

        **Random crop.** De foto gaat in een zwarte foto van 40 bij 40, 4 pixels van elke rand, en daaruit wordt op een willekeurige plek een 32 bij 32 gesneden: de foto tot 4 pixels opzij geschoven in elke richting, met zwart waar hij wegschoof. De hoek linksboven van de uitsnede kan op de rijen 0 tot en met 8 en de kolommen 0 tot en met 8 liggen: 9 × 9 = 81 plekken, en vanaf rij 4 en kolom 4 krijg je de foto zelf. `torch.zeros(3, 40, 40)` maakt de zwarte foto, zoals `np.zeros` een array met nullen maakt. Het is het opschuiven uit *Data augmentation*, met zwart waar `torch.roll` terugbracht wat er aan de andere kant afviel: rond een cijfer was dat leeg papier; rond een foto zou het een stuk van zijn andere kant zijn.

        **Color jitter.** Drie kleine veranderingen van de kleur, elk met een eigen factor, gekozen tussen 0,8 en 1,2. *Helderheid*: elk getal keer b, zodat zwart zwart blijft. *Contrast*, hoe ver de lichte en de donkere delen uit elkaar liggen, zoals in [Histogramegalisatie](/course/open/cv-histogram-equalization?lang=nl): de afstand van elk getal tot het gemiddelde van de hele foto, keer c. *Verzadiging*, hoe sterk de kleuren zijn: de afstand van elke kleur tot het grijs van zijn pixel, het gemiddelde van zijn 3 kleuren, keer s. Met s op 0 zou de foto grijs zijn; boven 1 worden zijn kleuren sterker. Na de veranderingen wordt elk getal tussen 0 en 1 gehouden, het bereik van een foto: `x.clamp(0, 1)` maakt van elk getal onder 0 een 0, en van elk getal boven 1 een 1.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **horizontal flip** | | een spiegelbeeld: links en rechts omgewisseld | `x.flip(dims=[2])` |
        | **random crop** | | een deel van de foto van een vaste grootte, op een willekeurige plek uitgesneden; hier 32 bij 32 uit de foto met een zwarte rand van 4 pixels | `padded[:, top:top + 32, left:left + 32]` |
        | **color jitter** | | een kleine willekeurige verandering van helderheid, contrast en verzadiging | `jitter(x, b, c, s)` |
        | `t.flip(dims)` | flip | de getallen van een tensor in de omgekeerde volgorde langs de axes `dims` | `x.flip(dims=[2])` |
        """),
        L.watch_out("Verander alleen de trainingsfoto's, en opnieuw elke keer dat de Dataset er een geeft: in zijn `__getitem__`, niet één keer voor het trainen. Meet op de testfoto's zoals ze zijn. En tel ook hoeveel trainingsfoto's hij goed leest op de trainingsfoto's zoals ze zijn: veranderd zeggen ze evenveel over de veranderingen als over het netwerk."),
        mo.md(r"""
        Het plaatje: hoeveel foto's je ColorCNN goed leest, van zijn trainingsfoto's zoals ze zijn, gevuld, en van de testfoto's, open, een stip voor elk van 5 seeds, met de code van de course. Zonder veranderingen maken 40 epochs hem slechter in nieuwe foto's dan 20: gemiddeld 503 testfoto's goed in plaats van 528, en 460 tot 545 over de 5 seeds, terwijl hij 99 tot 100% van zijn trainingsfoto's goed leest. Met alleen een spiegelbeeld leest hij er 541 tot 591 goed, maar zijn validation loss is het laagst na epoch 6 tot 11 en stijgt daarna. Met ook een random crop leest hij maar 65 tot 67% van zijn trainingsfoto's goed, en 590 tot 607 testfoto's: zijn validation loss blijft dalen tot epoch 36 tot 40. De kleur voegt hier weinig toe, 583 tot 619: de bereiken overlappen. Ondersteboven in plaats van gespiegeld, zonder iets anders, geeft 479 tot 492: veel minder dan alleen het spiegelbeeld, en zelfs minder dan zonder veranderingen na 20 epochs. Niet elke verandering waarbij de soort blijft kloppen helpt, en alleen meten laat het zien.
        """),
        mo.Html(C.gap_picture({"title": "Goed gelezen: trainings- en testfoto's", "x": "goed gelezen (%)",
                               "none": "geen veranderingen, {epochs} epochs", "flip": "spiegelen, {epochs} epochs",
                               "flip+crop": "spiegelen en crop, {epochs} epochs", "all": "spiegelen, crop en kleur, {epochs} epochs",
                               "upside": "ondersteboven in plaats van spiegelen, {epochs} epochs", "train": "training", "test": "test", "lang": L.lang,
                               "alt": "Zes rijen stippen tussen 40 en 100 procent goed gelezen. Geen veranderingen, 20 epochs: trainingsfoto's ongeveer 95%, testfoto's ongeveer 53%. Geen veranderingen, 40 epochs: training ongeveer 100%, test ongeveer 50%. Spiegelen, 40 epochs: training ongeveer 86%, test ongeveer 57%. Spiegelen en crop, 40 epochs: training ongeveer 66%, test ongeveer 60%. Spiegelen, crop en kleur, 40 epochs: training ongeveer 66%, test ongeveer 60%. Ondersteboven in plaats van spiegelen, 40 epochs: training ongeveer 78%, test ongeveer 49%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Data augmentation voor foto's:

    1. Schrijf elke verandering als een functie van één foto met shape (3, 32, 32): spiegelen, uitsnijden, kleur.
    2. In de `__getitem__` van een Dataset van de trainingsfoto's: kies willekeurig of je spiegelt, waar je snijdt, en de drie factoren van de kleur; verander de foto, en houd elk getal tussen 0 en 1.
    3. Normaliseer met de getallen van de trainingsfoto's zoals ze zijn; de testfoto's worden alleen genormaliseerd.
    4. Train langer: van veranderde foto's leert een netwerk langzamer, en leert het later uit zijn hoofd.
    5. Vergelijk met een run zonder veranderingen: hoeveel testfoto's hij goed leest, en hoeveel trainingsfoto's zoals ze zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel laadt CIFAR-10 en neemt testfoto 82, een auto, als floats van 0 tot en met 1. Hij spiegelt de auto en print de eerste 4 getallen van rij 10 in het rood, en de laatste 4 van rij 10 van het spiegelbeeld. Hij zet de auto in een zwarte foto van 40 bij 40 en snijdt daar 32 bij 32 uit vanaf rij 0 en kolom 8, en print beide shapes, en de pixel op rij 4 en kolom 0 van de uitsnede naast de pixel op rij 0 en kolom 4 van de auto. Daarna maakt hij de auto lichter, en print hij het grootste getal ervoor en erna, en hoeveel getallen boven 1 liggen. De regel van het spiegelbeeld klopt nog niet: die zet de auto ondersteboven. Voer de cel uit zoals hij is, verander hem dan waar het commentaar het zegt, en voer hem opnieuw uit. `cifar` is `everything_ai.cifar` van de course, die de cel hieronder geeft zodra CIFAR-10 op je computer staat.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="CIFAR-10 downloaden, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. Heb je [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) op deze computer gedaan, dan wel: dan vindt deze les het. Anders downloadt de knop het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. De pagina van CIFAR-10 noemt geen licentie, dus levert de course het niet mee; waar de foto's vandaan komen, staat in die projectles. Het uitgewerkte voorbeeld, het plaatje van de veranderingen en de demo hebben het nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
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


@app.cell
def example_changes(cifar):
    import torch

    X_train, y_train, X_test, y_test = cifar.load()
    x = torch.tensor(X_test[82], dtype=torch.float32) / 255
    print(cifar.KINDS[y_test[82]], x.shape)
    # een spiegelbeeld: de kolommen, axis 2, in de omgekeerde volgorde
    mirrored = x.flip(dims=[1])  # verander: dims=[2]
    print(x[0, 10, :4])
    print(mirrored[0, 10, -4:])
    # een crop: de auto in een zwarte foto van 40 bij 40, dan 32 bij 32 vanaf rij 0 en kolom 8
    padded = torch.zeros(3, 40, 40)
    padded[:, 4:36, 4:36] = x
    cut = padded[:, 0:32, 8:40]
    print(padded.shape, cut.shape)
    print(cut[0, 4, 0], x[0, 0, 4])
    # lichter: elk getal keer 1,2
    brighter = x * 1.2
    print(x.max(), brighter.max(), (brighter > 1).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste 4 getallen van rij 10 in het rood zijn 0,2667, 0,3098, 0,3451 en 0,3647. Zoals de cel nu is, print hij 0,3804, 0,5451, 0,6392 en 0,6392 voor het eind van rij 10 van het "spiegelbeeld": met `dims=[1]` staan de rijen in de omgekeerde volgorde, dus is zijn rij 10 rij 21 van de auto. Met `dims=[2]` print hij 0,3647, 0,3451, 0,3098 en 0,2667: de eerste 4, in de omgekeerde volgorde. De zwarte foto is 40 bij 40, en de uitsnede weer 32 bij 32. Zijn pixel op rij 4 en kolom 0 is de pixel van de auto op rij 0 en kolom 4, 0,2235: de auto schoof 4 pixels omlaag en 4 naar links. Het grootste getal is 0,9882, en 1,1859 in de lichtere auto, waar 57 van de 3.072 getallen boven 1 liggen, lichter dan wit. Daarom houdt de Dataset na de veranderingen elk getal tussen 0 en 1.
    """)
    return


@app.cell(hide_code=True)
def variants_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    mo.vstack([
        mo.md("De auto van het uitgewerkte voorbeeld, zoals hij is en veranderd: gespiegeld; lichter met meer contrast; donkerder met minder kleur; en ondersteboven, wat echte foto's bijna nooit laten zien."),
        mo.Html(C.variants_picture({"as_is": "zoals hij is", "mirrored": "gespiegeld", "brighter": "b 1,2 · c 1,2", "duller": "b 0,8 · s 0,8",
                                    "upside": "ondersteboven",
                                    "alt": "Vijf kleine foto's van een blauwe auto op gras. Zoals hij is, naar links gericht. Gespiegeld, naar rechts gericht. Lichter met meer contrast. Donkerder met flauwere kleuren. Ondersteboven."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_before_plane():
    mo.md(r"""
    Waar snijdt een random crop? Sleep de stip, of gebruik de pijltjestoetsen: hij verschuift de auto tot 4 pixels in elke richting. Links de auto in zijn zwarte foto van 40 bij 40, met een kader om de 32 bij 32 die wordt uitgesneden; rechts die uitsnede, zoals het netwerk hem krijgt.
    """)
    return


@app.cell(hide_code=True)
def plane_crop_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    _frames = C.crop_frames({"whole": "zwarte foto, 40 bij 40", "cut": "uitgesneden vanaf rij {top}, kolom {left}",
                             "alt": "De auto in een zwarte foto van 40 bij 40, met een kader om de 32 bij 32 die vanaf rij {top} en kolom {left} wordt uitgesneden; ernaast die uitsnede."})

    def _moved(_n, _way):
        return f"{abs(_n)} {'pixel' if abs(_n) == 1 else 'pixels'} {_way}"

    def _frame(_right, _up):
        _right, _up = int(_right), int(_up)
        _top, _left = 4 + _up, 4 - _right
        _ways = ([_moved(_up, "omhoog" if _up > 0 else "omlaag")] if _up else []) + ([_moved(_right, "naar rechts" if _right > 0 else "naar links")] if _right else [])
        _said = " en ".join(_ways) if _ways else "nergens heen: dit is de auto zelf"
        return _frames[(_right, _up)], f"Uitgesneden vanaf rij {_top} en kolom {_left}: de auto schoof {_said}."

    plane_crop = L.plane("crop-place", "pixels naar rechts", "pixels omhoog", _frame, C.SHIFTS, C.SHIFTS, start=(0, 0), role="",
                         task="Schuif de auto 4 pixels omlaag en 4 naar links, zoals in het uitgewerkte voorbeeld. Vanaf welke rij en kolom wordt hij uitgesneden?")
    plane_crop
    return


@app.cell(hide_code=True)
def text_after_plane():
    mo.md(r"""
    4 pixels omlaag en 4 naar links is de uitsnede vanaf rij 0 en kolom 8, die van het uitgewerkte voorbeeld: het kader zit rechtsboven in de zwarte foto, dus komt het zwart er boven en rechts in. Waar het kader ook zit, de uitsnede is 32 bij 32, en de zwarte band is nooit breder dan 4 pixels: de auto verliest hooguit 4 van zijn 32 rijen en 4 van zijn 32 kolommen. In het midden, vanaf rij 4 en kolom 4, is de uitsnede de auto zelf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke veranderingen helpen?

    Zes veranderingen die je bij een trainingsfoto van CIFAR-10 zou kunnen maken. Zet elke verandering in de goede groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("photo-changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("photo-changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def hints_changes():
    L.hints(
        "photo-changes",
        [
            "Vraag bij elke verandering twee dingen: is het nog dezelfde soort, en zou een echte foto er zo uit kunnen zien?",
            "Bij drie veranderingen blijft de soort kloppen en lijkt de foto echt. Een auto op zijn dak is nog steeds een auto: maar hoe vaak laat een foto er een zien?",
        ],
        why="**Blijft dezelfde soort, en lijkt op een echte foto:** een spiegelbeeld, 4 pixels verschoven, iets lichter. **Kan kwijtraken wat erop staat, of lijkt niet op een echte foto:** ondersteboven, een kwartslag gedraaid, één hoek vergroot. Welke veranderingen helpen, hangt van de foto's af: bij foto's van bovenaf, zoals van akkers vanuit een satelliet of van cellen onder een microscoop, lijken ondersteboven en een kwartslag even echt als de foto zelf.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_crop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een random crop

        Maak `crop(x, top, left)` af: zet de foto `x` in een zwarte foto van 40 bij 40, op de rijen en kolommen 4 tot en met 35, en geef de 32 bij 32 ervan terug die begint op rij `top` en kolom `left`. De Dataset van de demo roept hem aan met `top` en `left` willekeurig gekozen uit 0 tot en met 8. De check geeft je functie een verzonnen foto, en kijkt naar de shape van wat er terugkomt, naar de rijen en kolommen die zwart moeten zijn, en naar waar een pixel van de foto terechtkomt. Als hij klopt, bewaart de check je functie in je project, als `work/projects/cnn/crop.py`.
        """),
        L.task("Laat `crop` 32 bij 32 uit de foto snijden met een zwarte rand van 4 pixels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_crop():
    box_crop = L.code_box("crop", lines=8)
    box_crop
    return (box_crop,)


@app.cell(hide_code=True)
def check_crop(box_crop):
    result_crop = L.check("crop", box_crop.value)
    result_crop
    return (result_crop,)


@app.cell(hide_code=True)
def hints_crop():
    L.hints(
        "crop",
        [
            "Eerst de zwarte foto: `torch.zeros` met de shape (3, 40, 40). De foto gaat erin op 4 pixels van elke rand: de rijen 4 tot en met 35 en de kolommen 4 tot en met 35, voor alle 3 de kleuren.",
            "Snijd dan met slices: 32 rijen vanaf `top`, `top:top + 32`, en op dezelfde manier 32 kolommen vanaf `left`. Een `:` vooraan houdt alle 3 de kleuren.",
        ],
        why="`RandomCrop(32, padding=4)` van torchvision doet hetzelfde, met zwart als standaardvulling, en ook de ResNet-paper vulde op CIFAR-10 elke kant aan met 4 pixels en sneed er 32 bij 32 uit. Sommige versies vullen de rand met de pixels aan de rand van de foto, of met zijn spiegelbeeld, in plaats van met zwart.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_color():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Color jitter

        Maak `jitter(x, b, c, s)` af: verander de helderheid met factor `b`, dan het contrast met `c`, dan de verzadiging met `s`, zoals het commentaar in het vak zegt. Een foto heeft shape (3, H, W), dus is zijn gemiddelde over axis 0 het grijs van elke pixel, met shape (H, W): als je het van `x` aftrekt, gaat het van elk van de 3 kleuren af, net als bij NumPy. De Dataset van de demo houdt de getallen na je functie tussen 0 en 1, met `.clamp(0, 1)`, maar dat mag je ook zelf doen. De check geeft je functie verzonnen foto's, en vergelijkt wat er terugkomt met die van de course. Als hij klopt, bewaart de check je functie in je project, als `work/projects/cnn/color.py`.
        """),
        L.task("Laat `jitter` de helderheid, het contrast en de verzadiging van een foto veranderen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_color():
    box_color = L.code_box("color", lines=9)
    box_color
    return (box_color,)


@app.cell(hide_code=True)
def check_color(box_color):
    result_color = L.check("color", box_color.value)
    result_color
    return (result_color,)


@app.cell(hide_code=True)
def hints_color():
    L.hints(
        "color",
        [
            "Helderheid is één vermenigvuldiging. Contrast: neem het gemiddelde van de hele foto, `m = x.mean()`, en maak elke afstand daartoe `c` keer zo groot: `(x - m) * c + m`.",
            "Verzadiging gaat net als contrast, rond een ander getal: het grijs van elke pixel, het gemiddelde over de kleuren, axis 0.",
        ],
        why="De drie veranderingen geven in elke volgorde dezelfde foto, zolang niets de getallen tussendoor tussen 0 en 1 houdt: de check neemt elke volgorde. `ColorJitter` van torchvision maakt deze veranderingen ook, in een willekeurige volgorde, en houdt na elke verandering elk getal tussen 0 en 1; zijn grijs telt groen het zwaarst en blauw het lichtst, zoals ogen ze zien.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint je `ColorCNN` opnieuw op de eerste 6.000 trainingsfoto's van CIFAR-10, met je Trainer, maar nu uit een Dataset die elke foto opnieuw verandert, elke keer dat hij er een geeft: de helft gespiegeld, elke foto met je `crop` op een willekeurige plek uitgesneden, en met je `jitter` van kleur veranderd, met drie factoren gekozen tussen {C.number(C.LOW, 1, L.lang)} en {C.number(C.HIGH, 1, L.lang)}. Daarna wordt elk getal tussen 0 en 1 gehouden, en wordt de foto genormaliseerd met de getallen die je `channel_stats` voor de trainingsfoto's zoals ze zijn geeft. Zijn `__getitem__`:

    ```python
    def __getitem__(self, i):
        x = self.X[i].clone()                      # een kopie: de trainingsfoto zelf blijft zoals hij is
        u = torch.rand(6)                          # 6 willekeurige getallen van 0 tot 1
        if u[0] < 0.5:                             # de helft van de foto's
            x = x.flip(dims=[2])
        x = crop(x, int(u[1] * 9), int(u[2] * 9))  # top en left: hele getallen van 0 tot en met 8
        b, c, s = 0.8 + 0.4 * u[3:6]               # drie factoren van 0,8 tot 1,2
        x = jitter(x, b, c, s).clamp(0, 1)
        return (x - self.mean) / self.std, self.y[i]
    ```

    Hij traint {C.EPOCHS} epochs, twee keer zo lang als in de vorige les, met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test op de eerste 1.000 testfoto's, zoals ze zijn. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Hij heeft CIFAR-10 nodig, uit het uitgewerkte voorbeeld, en oefening 2 en 3, en duurt ongeveer een minuut. Tijdens het trainen laat een grafiek de validation loss na elke epoch zien. In het plaatje daarna is de grijze lijn de validation loss van hetzelfde netwerk zonder veranderingen, één run van de course met seed 0, 40 epochs lang: die verlaat het plaatje aan de bovenkant, op weg naar 4,4. Daarna kun je door je run terugspoelen: de schuif gaat terug naar een epoch, en laat zien wat je netwerk toen gokte bij één testfoto van elke soort.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je ColorCNN op veranderde foto's")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_augment(train_button, result_crop, result_color):
    result_crop, result_color  # run again when exercise 2 or 3 saves your module
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je netwerk te trainen.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je ColorCNN op veranderde foto's", "x": "epoch", "y": "validation loss",
                                        "alt": "De validation loss van je ColorCNN op de 1.000 testfoto's na elke epoch tot nu toe."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Je ColorCNN** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Je ColorCNN** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `crop` en je `jitter` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop in het uitgewerkte voorbeeld, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `crop`, je `jitter` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "crop": "*Je `crop` in `work/projects/cnn/crop.py` stopt met een fout, of geeft geen foto met shape (3, 32, 32) terug. Heb je zijn bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je code dan weer.*",
                "color": "*Je `jitter` in `work/projects/cnn/color.py` stopt met een fout, of geeft geen foto met shape (3, 32, 32) terug. Heb je zijn bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan weer.*",
                "cnn": "*Je `ColorCNN` in `work/projects/cnn/color_cnn.py`, of je `channel_stats` in `work/projects/cnn/channels.py`, stopt met een fout, of geeft niet wat hij moet geven. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening in [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation", "none": "zonder veranderingen",
                       "alt": "De loss van je ColorCNN na elke epoch: op de veranderde trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de run van de course zonder veranderingen, die het plaatje uit stijgt."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.guesses_picture(_guesses[_e - 1], _result["truth"], L.lang),
                        f"Na epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, en {_right} van deze {len(C.SHOWN)} foto's goed gelezen.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Ga terug naar epoch 1, en dan weer vooruit: welke foto's leest je netwerk het eerst goed?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog steeds.")
            _out = mo.vstack(_pic + [
                mo.md(f"""Je netwerk leest **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, zoals ze zijn, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters.{_turn} Over 5 seeds, met de code van de course: 3.917 tot 4.024 trainingsfoto's en 583 tot 619 testfoto's, met de laagste validation loss na epoch 33 tot 39; zonder veranderingen, na 40 epochs, 5.960 tot 6.000 trainingsfoto's en 460 tot 545 testfoto's.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.md(r"""
    ### Oefening 4. Wat de curves zeggen
    """)
    return


@app.cell(hide_code=True)
def choice_gap_cell():
    choice_gap = L.choice(
        "augment-gap",
        question="Zonder veranderingen eindigde de training loss na 40 epochs tussen 0,01 en 0,065, en de validation loss tussen 4,118 en 4,478. Met spiegelen, crop en kleur eindigen ze allebei rond 1,1: de training loss tussen 1,1 en 1,136, de validation loss tussen 1,072 en 1,176. Wat zeggen de curves?",
        options={
            "worse": "Het netwerk werd slechter: een training loss van 1,1 is veel hoger dan 0,01",
            "done": "Het netwerk is klaar met leren: beide losses blijven waar ze zijn",
            "right": "Het netwerk kan zijn trainingsfoto's nauwelijks nog uit zijn hoofd leren, dus geldt wat het leert ook voor de testfoto's",
        },
        correct="right",
        why={
            "worse": "Op zijn trainingsfoto's wel. Maar de loss die telt, is die op nieuwe foto's: 1,072 tot 1,176 tegen 4,118 tot 4,478, en het leest 583 tot 619 testfoto's goed, tegen 460 tot 545.",
            "done": "Zijn validation loss was het laagst na epoch 33 tot 39 van de 40: hij bleef dalen tot vlak voor het eind. Van veranderde foto's leert een netwerk langzamer, en daarom traint de demo twee keer zo lang als in de vorige les.",
            "right": "Elke epoch komt elke foto in een andere versie: wel of niet gespiegeld, verschoven, met andere kleuren. Details van de ene versie helpen niet bij de volgende, dus leert het wat alle versies gemeen hebben, zoals de vorm van een auto. Zijn training loss blijft hoog, en de validation loss blijft er dichtbij.",
        },
    )
    choice_gap
    return (choice_gap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De willekeurige veranderingen in een Dataset uit [Data augmentation](/course/open/nn-data-augmentation?lang=nl), nu voor foto's: een spiegelbeeld, het opschuiven uit die les als een random crop met een zwarte rand, en kleine willekeurige veranderingen van helderheid, contrast en verzadiging. Je `ColorCNN` uit [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) leest nu meer nieuwe foto's goed en leert zijn trainingsfoto's minder uit zijn hoofd, zoals de loss curves uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) laten zien. Je project `cnn` heeft nu `crop.py` en `color.py`. In de volgende les, *In een CNN kijken: filters en feature maps*, kijk je naar waar een getraind CNN op heeft leren letten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_changes, result_crop, result_color, choice_gap):
    L.wrap_up(result_changes, result_crop, result_color, choice_gap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
