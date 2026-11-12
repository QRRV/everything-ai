# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-vit-attention. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-vit-attention", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Een Vision Transformer vanaf nul](/course/open/cv-vit?lang=nl) en over [Wat een beeldmodel ziet: features en een linear probe](/course/open/cv-pretrained-features?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="In je ViT rekent head i van een block `A = torch.softmax(S, dim=2)` uit: voor foto's van 16 patches met het class token vooraan een tensor met shape `(n, 17, 17)`. Wat staat er in `A[0, 0]`, rij 0 van de eerste foto?",
        options={
            "right": "Hoe het class token zijn attention verdeelt over de 17 tokens, zichzelf meegerekend: 17 getallen die optellen tot 1",
            "column": "Hoeveel elk van de 17 tokens van het class token neemt",
            "features": "De 17 getallen die het class token na het block heeft",
        },
        correct="right",
        why={
            "right": "Rij i van de weights hoort bij token i, en de softmax over axis 2 laat elke rij optellen tot 1: een deel voor elk token waar het iets van neemt. Token 0 is het class token, dus rij 0 is waar het naar kijkt.",
            "column": "Dat is kolom 0, `A[0, :, 0]`: één getal uit elke rij. Een rij zegt hoe één token zijn eigen attention verdeelt; zie [Een Vision Transformer vanaf nul](/course/open/cv-vit?lang=nl).",
            "features": "De weights zeggen alleen hoeveel elk token van de andere neemt; wat het class token na het block heeft, zijn de gewogen values, met de residual connection en de MLP. Rij 0 van de weights is zijn attention: 17 delen die optellen tot 1.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_probe():
    quick_probe = L.choice(
        "quick-probe",
        question="Een linear probe leerde 10 klassen van de features die DINOv2 gaf voor 20 foto's per klasse. Wat krijgt hij van DINOv2 voor één foto?",
        options={
            "right": "384 getallen, de features van de foto: voor dezelfde foto altijd dezelfde, want DINOv2 is bevroren",
            "pixels": "De 150.528 getallen van de klaargemaakte foto, 3 · 224 · 224",
            "changing": "384 getallen die elke epoch een beetje veranderen, omdat DINOv2 meeleert",
        },
        correct="right",
        why={
            "right": "DINOv2 leert niets, dus een foto krijgt altijd dezelfde 384 getallen: daarom worden ze één keer uitgerekend, en leren alleen de 3850 getallen van de probe.",
            "pixels": "De pixels gaan DINOv2 in; de probe krijgt wat eruit komt, 384 getallen per foto. Op de pixels haalde dezelfde probe maar 33 van de 100 testfoto's goed; zie [Wat een beeldmodel ziet: features en een linear probe](/course/open/cv-pretrained-features?lang=nl).",
            "changing": "DINOv2 is bevroren: zijn 22 miljoen parameters krijgen geen gradient, dus zijn features blijven hetzelfde. Alleen de probe leert.",
        },
    )
    quick_probe
    return (quick_probe,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een linear probe op de features van DINOv2 haalt 98 van de 100 testfoto's van zeesterren, hanen, kroonluchters en de rest goed. Maar goed om welke reden? Een model kan een zeester herkennen aan het zand waar hij op ligt, of een waterlelie aan het water eromheen, en dan de mist ingaan bij een zeester op een rots. Om een model te vertrouwen, wil je zien uit welke delen van een foto zijn antwoord komt.

        Een Vision Transformer heeft iets wat een CNN niet heeft: getallen die zeggen hoeveel elk token van elk ander token neemt, zijn attention weights. DINOv2 is een Vision Transformer, en zijn 384 features zijn wat zijn class token na het laatste van zijn 12 blocks heeft. In elk block neemt het class token iets van de patches. Je kunt dus aflezen waar het naar kijkt, layer voor layer. Deze les leest die getallen, volgt ze door alle 12 layers, en test daarna of ze laten zien wat het model echt gebruikt.

        **Een voorbeeld uit het dagelijks leven.** Volg je de ogen van mensen die een kaart lezen, dan zie je waar ze kijken. Of ze dat stuk nodig hadden, merk je door het af te dekken: vinden ze de weg nog?
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_keep():
    guess_keep = L.guess(
        "keep",
        question="In de laatste layer van DINOv2 zet de attention van het class token de 256 patches van een foto op volgorde. Houd alleen het kwart over waar het het meest naar kijkt, 64 patches, en laat de rest weg: de probe krijgt wat het class token van die 64 maakt. Met alle 256 patches haalt hij 98 van de 100 testfoto's goed. Hoeveel nu?",
        options={
            "quarter": "Ongeveer 25",
            "half": "Ongeveer 50",
            "three-quarters": "Ongeveer 75",
            "almost-all": "Meer dan 90",
        },
        reveal="**94.** Met 64 willekeurig gekozen patches haalt hij er 81 tot 86, over 5 trekkingen; met de 64 waar het class token het minst naar kijkt 40. Het kwart van een foto waar het class token het meest naar kijkt, bevat bijna alles wat de probe nodig heeft. Oefening 4 en het tweede speelblok meten het met je eigen code.",
    )
    guess_keep
    return (guess_keep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De map van het class token.** DINOv2-small knipt een foto van 224 bij 224 pixels in 16 bij 16 patches van 14 bij 14 pixels: met het class token vooraan 257 tokens. Een layer is hier één block: layer 1 is het eerste block, layer 12 het laatste. In elk van zijn 12 blocks rekent elk van zijn 6 heads attention weights met shape (257, 257) uit, zoals een head van je ViT doet: rij i zegt hoe token i zijn attention verdeelt over alle 257 tokens, en telt op tot 1. Rij 0 is die van het class token. Het eerste getal is hoeveel het class token van zichzelf neemt; de andere 256 zeggen hoeveel het van elke patch neemt. Zet die 256 terug in 16 rijen van 16, en je hebt een map over de foto: waar het class token naar kijkt, in die head van die layer. De course stapelt de 6 heads van een block, dus een block geeft weights met shape (6, 257, 257).

        **Heads kijken naar verschillende dingen.** De figuur laat de koffie zien, de eigen foto van de course, met daaroverheen de map van het class token in elke head van layer 12: hoe helderder een patch, hoe meer het class token ervan neemt. Heads 1, 3 en 5 kijken vooral in het kopje, naar de koffie; head 4 is over de hele foto verspreid, net zo goed over de tafel als over het kopje. De DINO-paper (Caron en anderen, 2021) liet zulke maps zien van een Vision Transformer die zonder labels leerde, zoals DINOv2: in de laatste layer kijken verschillende heads naar verschillende voorwerpen of delen.
        """),
    ])
    return


@app.cell(hide_code=True)
def heads_picture_cell():
    mo.vstack([
        mo.Html(C.heads_picture({"head": "head {n}", "average": "gemiddelde van de heads", "rollout": "rollout, 12 layers",
                                 "alt": "Acht keer de koffie, met een map erover: de zes heads van layer 12, hun gemiddelde en de rollout van alle 12 layers. Heads 1, 3 en 5 kijken vooral in het kopje, naar de koffie; head 4 is over de hele foto verspreid; de rollout ook, met een paar helderdere patches."})),
        mo.md(r"""
        Over elke patch ligt een sluier in de kleur van de pagina, dunner naarmate het getal van de map groter is, en weg bij zijn drie grootste getallen. De laatste twee maps komen verderop in dit deel: het gemiddelde van de heads, en de rollout.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_idea_measure():
    mo.vstack([
        mo.md(r"""
        **Een map meten.** Een map kan overtuigend lijken en toch niet kloppen; daarom meet de course hem. Op elke foto van de 10 klassen van Caltech-101 tekenden de makers met de hand een omtrek om het voorwerp. Het deel van een map dat binnen de omtrek ligt, zegt hoeveel van de attention van het class token naar het voorwerp gaat: het getal van elke patch keer het deel van de patch binnen de omtrek, opgeteld, gedeeld door de som van de map. Een map die gelijk over de foto verspreid is, zou zoveel krijgen als het voorwerp beslaat: gemiddeld 51%, over de 300 foto's.

        **Layer voor layer.** In layer 12 legt het gemiddelde van de 6 heads 71% van de attention van het class token binnen de omtrek, en op 277 van de 300 foto's meer dan het voorwerp beslaat. Head 3 legt er 84% neer, head 4 maar 39%: die kijkt langs het voorwerp heen, zoals hij bij de koffie over de tafel verspreid is. In layer 1 tot en met 10 blijft het deel tussen 50% en 58%, dicht bij een gelijk verspreide map: daar neemt het class token nog overal iets van. Pas in layer 11 en 12 gaat het naar het voorwerp.
        """),
    ])
    return


@app.cell(hide_code=True)
def layers_picture_cell():
    mo.Html(C.layers_picture({"title": "Attention van het class token binnen de omtrek", "x": "layer", "y": "% binnen de omtrek",
                              "rollout": "rollout", "area": "wat het voorwerp beslaat: gelijk verspreid",
                              "alt": "Staven voor het deel van de attention van het class token binnen de omtrek, over 300 foto's: 50% tot 58% in layer 1 tot en met 10, 74% in layer 11, 71% in layer 12, en 57% voor de rollout van alle 12 layers; een gestreepte lijn bij 51%, het deel van de foto dat de voorwerpen beslaan."}, L.lang))
    return


@app.cell(hide_code=True)
def text_idea_rollout():
    mo.vstack([
        mo.md(r"""
        **Attention rollout.** In layer 12 neemt het class token iets van de 256 tokens van de patches, maar dat zijn de patches van de foto niet meer: in elke layer ervoor nam elk ervan iets van de andere. Om het class token terug te volgen naar de foto, vermenigvuldigt attention rollout (Abnar en Zuidema, 2020) de layers. Eerst maakt hij één matrix per layer, `W`, in twee stappen:

        1. **Het gemiddelde van de heads**, met shape (257, 257): één matrix in plaats van 6.
        2. **De residual connection.** Een block telt de uitkomst van attention op bij het token zelf, zoals in [Residual connections](/course/open/nn-residual?lang=nl): een token houdt ook zijn eigen getallen. Daarom telt rollout de eenheidsmatrix uit [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl) erbij op, attention van elk token naar zichzelf, en neemt van allebei de helft: `W = 0.5 · gemiddelde + 0.5 · I`. Elke rij van `W` telt nog steeds op tot 1. In PyTorch is `torch.eye(t)` de eenheidsmatrix van t bij t, zoals `np.eye`.

        Rij i van `W` zegt van welke tokens van vóór deze layer token i zijn getallen neemt. Dan begint `R` als de eenheidsmatrix, en doet elke layer, van de eerste tot de laatste, `R = W @ R`, een matrixvermenigvuldiging zoals in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl): rij i van `R` zegt hoeveel van token i uit elk token van de foto komt, vóór het eerste block. Na layer 12 is rij 0 van `R`, zonder zijn eerste getal, de map van de rollout. De ViT-paper tekende zijn maps van attention op deze manier.

        **Wat rollout hier vindt.** Op de 300 foto's legt de rollout 57% binnen de omtrek, tegen 71% voor layer 12 alleen, en op 228 van de 300 foto's meer dan het voorwerp beslaat. De informatie door elke layer volgen klinkt vollediger, maar hier komt de map meer verspreid uit dan die van layer 12 alleen. Oefening 3 vraagt waarom.
        """),
        L.watch_out("Een map van attention zegt waar het class token zijn getallen vandaan haalt, niet waarom, en niet voor welke klasse: DINOv2 leerde zonder klassen, en de probe komt erna. En in layer 12 heeft het token van een patch veel van andere patches in zich: een heldere patch zegt *dit token*, niet *deze pixels*. Dat een map er goed uitziet, laat niet zien dat het model die patches nodig heeft. Ze weglaten wel: het tweede speelblok meet het."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **attention rollout** | een map van waar het class token van een ViT zijn getallen vandaan haalt, gevolgd door elke layer: per layer het gemiddelde van de heads plus de eenheidsmatrix voor de residual connection, elk gehalveerd, en de layers met elkaar vermenigvuldigd | `R = W @ R` |
        | **`torch.sort(x, descending=True)`** | de getallen van x van groot naar klein; `.indices` bevat de plekken waar ze vandaan kwamen | `torch.sort(torch.tensor([0.2, 0.9, 0.5]), descending=True).indices` geeft `tensor([1, 2, 0])` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien waar het class token van een ViT naar kijkt, en het te testen:

    1. Stuur de foto door het model en bewaar de attention weights van elk block: per block een tensor met shape `(heads, T, T)`, met T = 1 + het aantal patches.
    2. **Eén layer:** rij 0 van elke head zonder zijn eerste getal, `weights[:, 0, 1:]`, in de shape van de foto: `(heads, rows, cols)`.
    3. **Alle layers:** begin met de eenheidsmatrix, `R = I = torch.eye(T)`; voor elk block, van het eerste tot het laatste, `W = 0.5 · (het gemiddelde van de heads) + 0.5 · I` en `R = W @ R`; aan het eind `R[0, 1:]` in de shape van de foto.
    4. **Test het:** laat de k patches weg die een map het hoogst zet, de eerste k van `torch.sort(m.reshape(-1), descending=True).indices`, geef het model het class token en de andere patches, en tel wat de probe nog goed haalt, tegen k willekeurig weggelaten patches.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel geeft DINOv2 de koffie en bewaart de attention weights van elk block, stap 1. `dino.looks(x)` draait de blocks en geeft de weights van elk, zoals `Block.weights` in `everything_ai/dinov2.py` ze uitrekent: voor een batch van n foto's per block een tensor met shape `(n, 6, 257, 257)`; `w[0]` is de ene foto. De cel print hoeveel blocks er zijn en de shape van het laatste, dan de sommen van de rijen van het class token in layer 12, het eerste getal van elke rij, en voor elke head de patch waar het het meest van neemt, als zijn rij en zijn kolom: patch j ligt in rij `j // 16` en kolom `j % 16`. `.argmax(dim=1)` werkt als `.argmax` met een axis in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl). `dino` is DINOv2-small, dat de cel hieronder laadt zodra beide downloads klaar zijn. Stap 2 tot en met 4 van het recept zijn de oefeningen.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="DINOv2-small en Caltech-101 downloaden, 225 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dino_cell(get_button):
    dino = C.dino_or_stop(get_button, {
        "ask": "*DINOv2-small en de foto's van Caltech-101 staan nog niet op je computer. De knop downloadt ze één keer: DINOv2-small van Hugging Face, 88 MB, en Caltech-101 van Caltech, 137 MB, waarvan hij daarna de foto's uitpakt. Deed je [Wat een beeldmodel ziet: features en een linear probe](/course/open/cv-pretrained-features?lang=nl), dan staan ze er al. Het uitgewerkte voorbeeld en de speelblokken hebben ze nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**{name} downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van Caltech-101 uitpakken** · {done} foto's",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face of Caltech was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 225 MB en de uitgepakte foto's. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*DINOv2-small en de foto's staan op je computer: 12 blocks van 6 heads, en 257 tokens voor een foto van 224 bij 224 pixels.*",
    })
    return (dino,)


@app.cell
def example_looks(dino):
    import torch
    from everything_ai.dinov2 import prepare

    # 1. de attention weights van elk block voor de koffie
    with torch.no_grad():
        layers = [w[0] for w in dino.looks(prepare("data/photos/coffee.png"))]
    print(len(layers), layers[11].shape)
    # de rij van het class token in elke head van layer 12
    row = layers[11][:, 0]
    print(row.sum(dim=1))
    print(row[:, 0])
    # waar elke head van layer 12 het meest naar kijkt: de patch met het grootste getal, als rij en kolom
    best = row[:, 1:].argmax(dim=1)
    print(best // 16, best % 16)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Er zijn 12 blocks, elk met weights met shape `(6, 257, 257)`. Elke rij van het class token telt op tot 1, zoals een softmax doet. Het eerste getal verschilt per head: head 3 neemt maar 0,02 van het class token zelf, heads 1 en 4 meer dan 0,17. Heads 1, 3 en 5 nemen het meest van een patch in rij 3 tot en met 5 en kolom 4 tot en met 10: in het kopje, zoals in de figuur van *Het idee*. Head 4 neemt het meest van patch (15, 15), de hoek rechtsonder van de foto, waar geen kopje is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_class():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De map van het class token

        Maak `class_attention(weights, rows, cols)` af: stap 2 van het recept. `weights` bevat de attention weights van één layer, shape `(heads, T, T)` met T = 1 + rows · cols; token 0 is het class token, en de patch in rij r en kolom c is token 1 + cols · r + c. Geef per head een map terug, shape `(heads, rows, cols)`. De check verzint weights waarin elke rij optelt tot 1, voor 2 tot 6 heads en foto's van 2 bij 2, 2 bij 3 en 3 bij 2 patches. Onder de check leest je functie de weights van de koffie in layer 12.
        """),
        L.task("Laat `class_attention` per head een map geven van waar het class token naar kijkt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_class():
    box_class = L.code_box("class-attention", lines=6)
    box_class
    return (box_class,)


@app.cell(hide_code=True)
def check_class(box_class):
    result_class = L.check("class-attention", box_class.value)
    result_class
    return (result_class,)


@app.cell(hide_code=True)
def hints_class():
    L.hints(
        "class-attention",
        hints=[
            "Je hebt één rij van elke head nodig, en in die rij alle getallen behalve het eerste. Welke index kiest de heads, welke de rij, en welke de getallen?",
            "`weights[:, 0, 1:]`: alle heads, rij 0, vanaf plek 1. Dat geeft shape `(heads, rows * cols)`.",
            "Geef het de shape van de foto: `return weights[:, 0, 1:].reshape(___, rows, cols)`, waarbij op de plek van `___` het aantal maps komt, één per head.",
        ],
        why="Rij 0 van elke head hoort bij het class token; het eerste getal is de attention van het class token naar zichzelf, dus `1:` houdt de patches over. De patches komen rij voor rij, dus `reshape(-1, rows, cols)` zet elk getal terug op zijn patch, en `-1` houdt één map per head.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rollout():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Attention rollout

        Maak `rollout(layers, rows, cols)` af: stap 3 van het recept. `layers` bevat de weights van elke layer, op volgorde, elk met shape `(heads, T, T)`. Vul de drie `...` in: `W` voor één layer, `R` na deze layer, en aan het eind de rij van het class token, zonder zichzelf, als een map van `rows` bij `cols`. De check verzint 1 tot 3 layers van 2 of 3 heads. Onder de check staat je map van het tweede geval naast de goede, waar ze verschillen.
        """),
        L.task("Laat `rollout` het class token door elke layer terugvolgen naar de patches."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rollout():
    box_rollout = L.code_box("rollout", lines=13)
    box_rollout
    return (box_rollout,)


@app.cell(hide_code=True)
def check_rollout(box_rollout):
    result_rollout = L.check("rollout", box_rollout.value)
    result_rollout
    return (result_rollout,)


@app.cell(hide_code=True)
def hints_rollout():
    L.hints(
        "rollout",
        hints=[
            "Drie dingen: één matrix voor de layer, gemaakt uit de heads en de residual connection; deze layer na de layers ervoor; en een rij van het resultaat.",
            "Het gemiddelde van de heads is `weights.mean(dim=0)`, met shape (T, T); de eenheidsmatrix is `I`, bovenaan de functie. Neem van allebei de helft. De nieuwe `R` is dan de `W` van deze layer keer de oude `R`.",
            "`W = 0.5 * weights.mean(dim=0) + 0.5 * I`, `R = W @ R`, en `return R[___, 1:].reshape(rows, cols)`, met de rij van het class token op de plek van `___`.",
        ],
        why="De heads zijn axis 0, dus `mean(dim=0)` middelt ze. De eenheidsmatrix staat voor de residual connection: elk token houdt ook zijn eigen getallen. Met van allebei de helft telt elke rij van `W` op tot 1. `W @ R` zet deze layer na de layers ervoor: rij 0 van het resultaat zegt hoeveel van het class token uit elk token van de foto komt, en `1:` laat het deel van het class token uit zichzelf weg.",
    )
    return


@app.cell(hide_code=True)
def text_look():
    mo.md(r"""
    ### Rondkijken

    Kies een foto, een layer en een map. De koffie is de eigen foto van de course; de andere zijn de eerste testfoto van elke klasse van Caltech-101, met de omtrek die de makers tekenden als gestreepte lijn. *head 1* tot en met *head 6* en *gemiddelde van de heads* zijn de maps van het class token in de layer die je kiest; *rollout* volgt het van layer 1 tot en met die layer. Onder de foto: hoeveel van de map binnen de omtrek ligt, tegen wat het voorwerp beslaat. Zodra oefening 1 en 2 goed zijn, komen de maps uit je `class_attention` en je `rollout`.
    """)
    return


@app.cell(hide_code=True)
def look_controls():
    look_photo = mo.ui.dropdown(options={"koffie": "coffee"} | {C.NAMES[L.lang][c]: c for c in C.CLASSES}, value="koffie", label="Foto")
    look_layer = mo.ui.slider(1, 12, value=12, label="Layer", show_value=True)
    look_map = mo.ui.radio(options={f"head {h}": str(h) for h in range(1, 7)} | {"gemiddelde van de heads": "average", "rollout": "rollout"},
                           value="gemiddelde van de heads", label="Map", inline=True)
    mo.vstack([look_photo, look_layer, look_map])
    return look_layer, look_map, look_photo


@app.cell(hide_code=True)
def look_cell(dino, look_photo, look_layer, look_map, box_class, result_class, box_rollout, result_rollout):
    _ca, _ca_yours = C.yours_or_course(box_class.value, "class_attention", result_class.state == "correct")
    _ro, _ro_yours = C.yours_or_course(box_rollout.value, "rollout", result_rollout.state == "correct")
    _key, _layer, _which = look_photo.value, look_layer.value, look_map.value
    try:
        _m = C.view(C.looks(dino, _key), _layer, _which, _ca, _ro)
        _failed = None
    except Exception:
        _m, _failed = C.view(C.looks(dino, _key), _layer, _which), True
    _title = (f"Rollout van layer 1 tot en met {_layer}" if _which == "rollout" else
              f"Layer {_layer}, gemiddelde van de heads" if _which == "average" else f"Layer {_layer}, head {_which}")
    _inner = C.inside(C.path_of(_key))
    if _inner is None:
        _said = "De koffie heeft geen omtrek: hier zie je alleen de map."
    else:
        _said = (f"**{C.percent(C.share(_m, _inner))}%** van deze map ligt binnen de omtrek; de {C.NAMES[L.lang][_key]} beslaat "
                 f"{C.percent(float(_inner.mean()))}% van de foto.")
    _who = ("" if (_ca_yours if _which != "rollout" else _ro_yours) else
            (" De map komt van de functie van de course: de jouwe stopt hier met een fout." if _failed else
             f" De map komt van de `{'rollout' if _which == 'rollout' else 'class_attention'}` van de course, tot je oefening {'2' if _which == 'rollout' else '1'} goed is."))
    mo.vstack([mo.Html(C.look_picture(_key, _m, {"title": _title, "alt": f"De foto met de map erover: {_title}."})), mo.md(_said + _who)])
    return


@app.cell(hide_code=True)
def text_after_look():
    mo.md(r"""
    In layer 12 ligt het gemiddelde van de heads op de zeester: 55% van de map, op een zeester die 10% van de foto beslaat. In elk van de tien foto's van Caltech-101 legt head 4 het minst binnen de omtrek: 9% bij de zeester. Kies een lagere layer: in layer 6 krijgt de zeester 18% van de map, een derde van wat hij in layer 12 krijgt. Niet elke foto werkt zo goed: bij de lotus legt zelfs layer 12 47% binnen een omtrek die 47% van de foto beslaat. Kies *rollout*: die is ook verspreid, met een paar heldere patches.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.md(r"""
    ### Oefening 3. Waarom rollout zich verspreidt
    """)
    return


@app.cell(hide_code=True)
def choice_spread_cell():
    choice_spread = L.choice(
        "spread",
        question="Over de 300 foto's legt de attention van het class token in layer 12 71% binnen de omtrek, en de rollout van alle 12 layers maar 57%. Waarom ligt de rollout minder op het voorwerp?",
        options={
            "right": "Hij vermenigvuldigt ook layer 1 tot en met 10 mee, waar het class token ongeveer overal iets van neemt",
            "no-residual": "Zijn helft eenheidsmatrix, voor de residual connection, verspreidt de attention over de foto",
            "one-head": "Hij gebruikt maar één head per layer, en één head kan langs het voorwerp kijken",
        },
        correct="right",
        why={
            "right": "De rollout van alleen layer 11 en 12 legt 72% binnen de omtrek; vanaf layer 9 65%; vanaf layer 1 57%. Elke vroege layer die hij meevermenigvuldigt, voegt een verspreide map toe: in layer 1 tot en met 10 legt de attention van het class token maar 50% tot 58% binnen de omtrek. En `R` eindigt met de `W` van layer 1, dus rij 0 van `R` is een gewogen gemiddelde van de rijen van die `W`: van waar de tokens in layer 1 naar kijken. Op 270 van de 300 foto's zitten minstens twee van de drie helderste patches van de rollout bij de drie waar het class token in layer 1 het meest naar kijkt.",
            "no-residual": "Zonder de eenheidsmatrix is `W` alleen het gemiddelde van de heads, en legt de rollout van alle 12 layers 56% binnen de omtrek: iets minder dan mét. De eenheidsmatrix houdt een deel van de eigen getallen van elk token vast; wat de map verspreidt, zijn de vroege layers.",
            "one-head": "Rollout middelt de heads, `weights.mean(dim=0)`, zoals het gemiddelde van layer 12 ook doet. Het verschil zit in de layers: rollout neemt alle 12 mee, en layer 1 tot en met 10 kijken ongeveer overal naar.",
        },
    )
    choice_spread
    return (choice_spread,)


@app.cell(hide_code=True)
def text_exercise_keep():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Patches weglaten

        Een map zegt waar het class token naar kijkt. Of DINOv2 die patches nodig heeft, test je door ze weg te laten: een ViT neemt elk aantal tokens, dus je geeft hem die patches gewoon niet. Elk token krijgt zijn positie voordat het de blocks in gaat, dus het houdt zijn plek in de foto; en zoals [Een Vision Transformer vanaf nul](/course/open/cv-vit?lang=nl) zei, maakt het attention niet uit in welke volgorde de tokens komen. Maak `keep(m, k)` af: stap 4 van het recept. `m` is een map met shape `(rows, cols)`, `k` het aantal patches om weg te laten. Geef de nummers van de tokens om te houden terug, als één tensor: het class token, 0, en elke patch behalve de k met de grootste getallen in `m`; patch j is token 1 + j. De volgorde maakt niet uit: de check sorteert wat je teruggeeft. Onder de check laat je `keep` de helft van de patches van de koffie weg.
        """),
        L.task("Laat `keep` het class token geven en elke patch behalve de k die een map het hoogst zet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_keep():
    box_keep = L.code_box("keep", lines=6)
    box_keep
    return (box_keep,)


@app.cell(hide_code=True)
def check_keep(box_keep):
    result_keep = L.check("keep", box_keep.value)
    result_keep
    return (result_keep,)


@app.cell(hide_code=True)
def hints_keep():
    L.hints(
        "keep",
        hints=[
            "Zet eerst de patches op volgorde, van het grootste getal in `m` naar het kleinste; sla dan de eerste k over, en zet het class token erbij.",
            "`torch.sort(m.reshape(-1), descending=True).indices` geeft de nummers van de patches in die volgorde. `[k:]` slaat de eerste k over. Patch j is token j + 1.",
            "`order = torch.sort(m.reshape(-1), descending=True).indices` en `return torch.cat([torch.tensor([0]), order[___] + 1])`, met de patches om te houden op de plek van `___`.",
        ],
        why="`reshape(-1)` maakt van de map één rij, zodat patch j nummer j is, net als bij de nummers van de tokens. `torch.sort` met `descending=True` zet de grootste vooraan, en `.indices` geeft waar elk vandaan kwam, dus `order[k:]` zijn de patches die over zijn na de k hoogste. Met 1 erbij wordt het nummer van een patch het nummer van zijn token, en `torch.tensor([0])` vooraan houdt het class token, waarvan de getallen de features van DINOv2 zijn.",
    )
    return


@app.cell(hide_code=True)
def text_leave():
    mo.md(r"""
    ### Patches weglaten

    Kies een foto, de map waarvan de grootste getallen eerst weggaan, en hoeveel patches je weglaat. *hoogste attention eerst* gebruikt het gemiddelde van de heads in layer 12; *laagste attention eerst* dezelfde map andersom; *willekeurig* willekeurige getallen; *hoogste rollout eerst* de rollout van alle 12 layers. Je ziet de foto met de weggelaten patches bedekt, en wat de probe uit [Wat een beeldmodel ziet: features en een linear probe](/course/open/cv-pretrained-features?lang=nl), getraind op 20 foto's per klasse, maakt van het class token van wat over is. De eerste keer dat het speelblok draait, rekent het voor de probe één keer de features van DINOv2 voor de 300 foto's uit: op de computer van de course ongeveer een halve minuut. De grafiek laat zien wat de course mat op de 100 testfoto's; de knop telt het met je `keep`, de eerste keer in ongeveer 15 seconden, omdat hij dan ook één keer de maps van de 100 testfoto's uitrekent.
    """)
    return


@app.cell(hide_code=True)
def leave_controls():
    leave_photo = mo.ui.dropdown(options={C.NAMES[L.lang][c]: c for c in C.CLASSES}, value=C.NAMES[L.lang]["starfish"], label="Foto")
    leave_map = mo.ui.radio(options={"hoogste attention eerst": "top", "laagste attention eerst": "low", "willekeurig": "random", "hoogste rollout eerst": "rollout"},
                            value="hoogste attention eerst", label="Weglaten", inline=True)
    leave_k = mo.ui.slider(0, 224, step=32, value=128, label="Weggelaten patches, van de 256", show_value=True)
    count_button = mo.ui.run_button(label="Met je keep tellen op de 100 testfoto's")
    mo.vstack([leave_photo, leave_map, leave_k, count_button])
    return count_button, leave_k, leave_map, leave_photo


@app.cell(hide_code=True)
def leave_cell(dino, leave_photo, leave_map, leave_k, box_keep, result_keep):
    _keep, _yours = C.yours_or_course(box_keep.value, "keep", result_keep.state == "correct")
    _key, _which, _k = leave_photo.value, leave_map.value, leave_k.value
    _out = C.leave_one(dino, _key, _which, _k, _keep)
    _name = C.NAMES[L.lang]
    _said = (f"Weggelaten: **{_k}** patches. De probe zegt nu **{_name[_out['now']]}**, met een kans van {C.percent(_out['p_now'])}%; "
             f"met alle 256 patches {_name[_out['all']]}, met {C.percent(_out['p_all'])}%.")
    _who = "" if _yours and not _out["stood_in"] else " De `keep` van de course koos de patches, tot je oefening 4 goed is."
    _chart = C.left_out_chart(_k, {"title": "Testfoto's goed, van de 100", "x": "weggelaten patches, van de 256", "y": "goed",
                                   "top": "hoogste attention eerst", "low": "laagste attention eerst", "random": "willekeurig", "rollout": "hoogste rollout eerst",
                                   "alt": "Vier lijnen over de weggelaten patches, van 0 tot 224 van de 256: het aantal testfoto's dat de probe goed haalt, daalt het snelst als de patches met de hoogste attention eerst gaan, tot 29 bij 224; het langzaamst als die met de laagste attention eerst gaan, 85 bij 224; willekeurige patches zitten ertussen, 59 bij 224; hoogste rollout eerst 42 bij 224."}, L.lang)
    mo.vstack([mo.Html(C.leave_picture(_key, _out["kept"], {"title": f"{_name[_key]}, {_k} patches weggelaten", "alt": f"De {_name[_key]} met {_k} bedekte patches."})),
               mo.md(_said + _who), mo.Html(_chart)])
    return


@app.cell(hide_code=True)
def count_cell(dino, count_button, leave_map, leave_k, box_keep, result_keep):
    if not count_button.value:
        _count = mo.md("*Klik op de knop om met je `keep` te tellen op de 100 testfoto's.*")
    else:
        _keep, _yours = C.yours_or_course(box_keep.value, "keep", result_keep.state == "correct")
        _which, _k = leave_map.value, leave_k.value
        mo.output.replace(mo.md("*De maps van de 100 testfoto's uitrekenen, één keer, en patches weglaten…*"))
        try:
            _right = C.right_without(dino, _keep, _which, _k, progress=lambda _n: mo.output.replace(mo.md(f"*Testfoto {_n} van 100…*")))
            _stood = False
        except Exception:
            _right, _stood = C.right_without(dino, C.course("keep"), _which, _k), True
        _ref = C.REFERENCE["left_out"][_which][C.KS.index(_k)]
        _whose = "je `keep`" if _yours and not _stood else "de `keep` van de course, omdat de jouwe nog niet goed is" if not _stood else "de `keep` van de course, omdat de jouwe hier met een fout stopt"
        _how = {"top": "hoogste attention eerst", "low": "laagste attention eerst", "random": "willekeurig", "rollout": "hoogste rollout eerst"}[_which]
        _count = mo.md(f"Met {_whose}, bij {_k} weggelaten patches, *{_how}*: de probe haalt **{_right}** van de 100 testfoto's goed. De course mat {_ref}.")
    _count
    return


@app.cell(hide_code=True)
def text_after_leave():
    mo.md(r"""
    De 64 patches weglaten waar het class token in layer 12 het meest naar kijkt, kost de probe 14 testfoto's: van 98 naar 84. 64 willekeurige patches kosten er 1, in alle 5 trekkingen. Laat de 192 met de laagste attention weg, zodat alleen de 64 met de hoogste overblijven, en hij haalt er nog 94 goed. De map wijst naar wat het model gebruikt: de patches die hij hoog zet, tellen veel zwaarder dan willekeurige, en die hij laag zet, tellen nauwelijks mee. De volgorde van de rollout zit ertussen: zijn 64 hoogste patches weglaten kost 10 testfoto's. Ook volgens deze test wijst layer 12 alleen beter dan de rollout van alle 12 layers, op dit model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_what():
    mo.md(r"""
    ### Oefening 5. Wat een map laat zien
    """)
    return


@app.cell(hide_code=True)
def explain_what_cell():
    explain_what = L.explain(
        "what-it-shows",
        question="Een collega laat je de map van het class token in layer 12 zien over een foto van een waterlelie, helder op het water rond de bloem, en zegt: \"Het model herkent waterlelies dus aan het water.\" Wat laat zo'n map zien, wat laat hij niet zien, en hoe zou je uitzoeken of de collega gelijk heeft?",
        model_answer="De map laat zien van welke tokens het class token in layer 12 zijn getallen neemt, als delen die optellen tot 1. Hij laat niet zien waarom, en ook niet voor welke klasse: DINOv2 heeft geen klassen, en in layer 12 heeft het token van een patch getallen van andere patches in zich, dus een heldere patch water gaat niet per se over de pixels van het water. Om het uit te zoeken, laat je de patches met water weg en kijk je of de probe nog waterlelie zegt, tegen evenveel willekeurig weggelaten patches; of je test de probe op waterlelies zonder water eromheen.",
        points=[
            "hij laat zien waar het class token zijn getallen vandaan haalt, in die layer",
            "hij laat niet zien waarom, of voor welke klasse",
            "een token in een late layer heeft getallen van andere patches in zich",
            "test het: laat die patches weg en vergelijk met willekeurige patches",
        ],
    )
    explain_what
    return (explain_what,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De attention weights van je ViT uit [Een Vision Transformer vanaf nul](/course/open/cv-vit?lang=nl) zijn getallen die je kunt aflezen: in DINOv2 is de rij van het class token in elke head een map van waar het naar kijkt, en in de laatste twee layers ligt die map op het voorwerp. Attention rollout volgt het class token met matrixvermenigvuldigingen door alle layers, maar op DINOv2-small telt hij vooral de eerste tien layers mee, die ongeveer overal naar kijken. Je `keep` maakte van de map een test: de patches weglaten waar het class token het meest naar kijkt, kost de probe uit [Wat een beeldmodel ziet: features en een linear probe](/course/open/cv-pretrained-features?lang=nl) veel meer dan willekeurige patches weglaten. Een plaatje van waar een model naar kijkt, is een bewering; delen weglaten test hem. Dit was de laatste les van de module over Vision Transformers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_class, result_rollout, choice_spread, result_keep, explain_what):
    L.wrap_up(result_class, result_rollout, choice_spread, result_keep, explain_what)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
