# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-vit. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-vit", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Het transformer block](/course/open/attn-transformer-block?lang=nl) en [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="Een tekst van 5 tokens gaat door je `CausalSelfAttention`. Van welke tokens neemt het derde token iets over?",
        options={
            "all": "Van alle 5: attention vergelijkt elk token met elk token",
            "right": "Van de eerste drie: van zichzelf en van de twee ervoor",
            "self": "Alleen van zichzelf",
            "after": "Van zichzelf en van de twee erna",
        },
        correct="right",
        why={
            "all": "Zonder de mask wel. De causal mask zet de scores van elk later token op min oneindig, vóór de softmax, dus neemt een token alleen iets over van zichzelf en van de tokens ervoor. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "self": "Dan was er helemaal geen attention. De mask verbergt alleen de tokens erna: het derde token neemt iets over van de eerste drie. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "after": "Andersom: de mask verbergt de toekomst, de tokens erna, omdat een model dat tekst schrijft niet mag zien wat er nog komt. Zie [De causal mask](/course/open/attn-causal-mask?lang=nl).",
            "right": "Een tekst heeft een ervoor en een erna, en een model dat schrijft, mag niet vooruitkijken. Een foto heeft dat allebei niet: vandaag gaat de mask eruit, en neemt elke patch iets over van elke patch.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def quick_check_changes():
    quick_changes = L.choice(
        "quick-changes",
        question="Met spiegelen, crop en kleur las je ColorCNN na 40 epochs 583 tot 619 van de 1.000 testfoto's goed; zonder veranderingen 460 tot 545. Wat deden de veranderingen?",
        options={
            "more": "Ze voegden foto's van nieuwe soorten toe",
            "right": "Ze maakten het moeilijker om de trainingsfoto's uit het hoofd te leren, dus leerde hij meer over de soorten",
            "easier": "Ze maakten de foto's makkelijker te lezen, met meer contrast",
        },
        correct="right",
        why={
            "more": "Het blijven dezelfde 10 soorten, en dezelfde 6.000 foto's: elke epoch laat ze elk een beetje anders zien. Zie [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl).",
            "easier": "De color jitter maakt sommige foto's ook doffer, en het netwerk wordt getest op de foto's zoals ze zijn. Wat veranderde, is dat hij zijn trainingsfoto's minder makkelijk uit zijn hoofd kon leren. Zie [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl).",
            "right": "Elke epoch komt elke foto in een andere versie, dus helpen details van de ene versie niet bij de volgende. Vandaag helpen dezelfde veranderingen een netwerk dat veel meer uit zijn foto's moet leren.",
        },
    )
    quick_changes
    return (quick_changes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een convolutie kijkt naar een paar naburige pixels tegelijk, met hetzelfde filter op elke plek van de foto: een oor is hetzelfde patroon linksboven als in het midden. Elk CNN heeft die twee ideeën al voordat het één foto heeft gezien. Attention heeft ze geen van beide: elk token kan iets overnemen van elk token, en een transformer weet alleen waar een token staat door de positie die erbij is opgeteld. In 2020 knipten Alexey Dosovitskiy en anderen bij Google foto's in vierkantjes van 16 bij 16 pixels, en gaven die als tokens aan een transformer, zonder één convolutie: een *Vision Transformer*, of ViT. Getraind op de 1,3 miljoen foto's van ImageNet, zonder sterke maatregelen tegen overfitting, eindigde hij een paar procentpunten onder ResNets van ongeveer zijn grootte. Gepretraind op 300 miljoen foto's, een eigen verzameling van Google, deed hij het even goed als de beste CNN's van toen, en kostte hij minder om te trainen. Vandaag bouw je er een uit je transformer block, en train je hem op de 6.000 foto's van je ColorCNN.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_vit():
    guess_vit = L.guess(
        "vit-or-cnn",
        question="Je ViT, met 11.738 parameters, en je ColorCNN, met 11.642, leren allebei 40 epochs lang van dezelfde 6.000 foto's, gespiegeld, uitgesneden en van kleur veranderd. Welke van de twee leest meer van de 1.000 testfoto's goed?",
        options={
            "right": "De ColorCNN, duidelijk",
            "same": "Ongeveer evenveel",
            "vit": "De ViT, duidelijk",
        },
        reveal="**De ColorCNN.** Over 5 seeds, met de code van de course, leest de ColorCNN 583 tot 619 testfoto's goed, en de ViT 489 tot 524. Uit 6.000 foto's moet de ViT leren wat de ColorCNN vanaf het begin heeft, en zover komt hij niet.",
    )
    guess_vit
    return (guess_vit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een foto als een rij tokens.** Een transformer leest een rij tokens. Een ViT maakt van een foto een rij: hij knipt de foto in vierkantjes van p bij p pixels, zijn **patches**, en neemt ze rij voor rij, zoals je een bladzijde leest. Een foto van 32 bij 32 met p = 4 geeft 8 rijen van 8 patches: 64 patches, elk van 3 · 4 · 4 = 48 getallen, zijn rood, zijn groen en zijn blauw. Het woord patch kwam je tegen in [Convolutie als één grote matrixvermenigvuldiging](/course/open/cnn-conv-numpy?lang=nl): daar was een patch het stuk van het beeld onder een filter; hier liggen de patches naast elkaar, zonder overlap.

        **Patch embedding.** Een tekstmodel zoekt elk token op in een tabel. Een ViT stuurt de 48 getallen van elke patch door één lineaire layer, dezelfde voor elke patch, naar d getallen: de **patch embedding**. Eén lineaire layer op elke patch is dezelfde d filters op elke plek: een convolutie met d output channels, een kernel size van 4 en een stride van 4, zoals in [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl), doet precies dit, en veel code voor ViTs schrijft het zo.

        **Posities.** Attention weet niet waar een patch vandaan kwam: hussel de 64 patches door elkaar, en elk token krijgt dezelfde getallen als ervoor, alleen in een andere volgorde. Daarom krijgt elke plek, zoals in [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl), een geleerde positie, een rij van een `nn.Embedding`, opgeteld bij zijn patch. Niets vertelt de ViT dat patch 9 naast patch 10 en 17 ligt: dat leert hij uit de foto's, of helemaal niet.

        **Het class token.** Welk token zegt wat de foto laat zien? Een ViT zet één extra token vóór de 64 patches: het **class token**, een parameter van d getallen, hetzelfde voor elke foto, die het netwerk leert. Het is geen deel van de foto, dus weet het alleen wat het van de patches overneemt. Na het laatste block gaan zijn d getallen door een layer normalization en een lineaire layer naar de 10 logits.

        **Geen mask.** In een model dat tekst schrijft, neemt een token alleen iets over van de tokens ervoor: het mag niet zien wat er nog komt. In een foto komt niets erna: elke patch neemt iets over van elke patch, en het class token, vooraan, van alle 64. Het block van een ViT is dus je block uit [Het transformer block](/course/open/attn-transformer-block?lang=nl), met de mask uit zijn attention gehaald.

        **Wat een convolutie inbouwt.** Een convolutie ziet alleen een paar naburige pixels, en gebruikt overal hetzelfde filter. Dat zijn aannames over foto's die al gelden voordat het netwerk iets leert: een **inductive bias**. Een ViT heeft daar weinig van: dat de pixels van één patch bij elkaar horen, en dat één layer elke patch leest. Dat patch 9 naast patch 10 ligt, en dat een oor in een hoek hetzelfde is als een in het midden, moet hij uit zijn foto's leren.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Betekenis | In code |
        |---|---|---|---|
        | **Vision Transformer** | | een transformer voor foto's: de patches van een foto zijn zijn tokens, met een class token vooraan; kort: ViT | `ViT(d, h, depth, p)` |
        | **patch embedding** | | de d getallen die een lineaire layer maakt van de getallen van één patch: het token van die patch | `nn.Linear(3 * p * p, d)` |
        | **class token** | | een extra token vóór de patches, geleerd, waaruit het netwerk leest wat de foto laat zien | `nn.Parameter(torch.zeros(1, 1, d))` |
        | **inductive bias** | | wat een netwerk over zijn data aanneemt voordat het iets leert, zoals dat naburige pixels bij elkaar horen | |
        | `t.expand(n, 1, -1)` | expand | dezelfde tensor n keer langs een axis van grootte 1, zonder te kopiëren; -1 laat een axis zoals hij is | `self.cls.expand(len(x), 1, -1)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een Vision Transformer voor foto's van 32 bij 32 met patches van p bij p:

    1. Knip elke axis van de foto's in tweeën met `reshape`: de rijen in `h // p` rijen patches van elk `p` rijen, de kolommen net zo: shape `(n, c, h // p, p, w // p, p)`.
    2. Zet de twee axes die zeggen welke patch het is vooraan, met `permute(0, 2, 4, 1, 3, 5)`, dan de kleur en de rijen en kolommen binnen de patch.
    3. `reshape` naar `(n, (h // p) * (w // p), c * p * p)`: één rij getallen per patch.
    4. Stuur elke patch door de patch embedding, zet het class token vóór de tokens van elke foto, en tel de geleerde posities van allemaal erbij op.
    5. Laat de tokens door de blocks gaan, elk met attention zonder mask.
    6. Lees het class token na het laatste block: een layer normalization, dan een lineaire layer naar de logits.
    7. Train hem zoals een CNN: dezelfde cross-entropy, dezelfde Trainer, dezelfde veranderde foto's.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel laadt CIFAR-10 en neemt testfoto 82, de auto, als een batch van één foto. Hij knipt patch (1, 2) eruit met slices, maakt er één rij getallen van, en print beide shapes, en dan de eerste 4 getallen van die rij naast de eerste rij rood van de patch. Daarna stuurt hij de rij door een patch embedding van 16 getallen, en print hij de shape daarvan. Voer de cel uit zoals hij is, verander dan `p` waar het commentaar het zegt, en voer hem opnieuw uit. `cifar` is `everything_ai.cifar` van de course, die de cel hieronder geeft zodra CIFAR-10 op je computer staat.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="CIFAR-10 downloaden, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 staat nog niet op je computer. Heb je [Project: je CNN op kleurenfoto's](/course/open/cv-color-cnn?lang=nl) op deze computer gedaan, dan wel: dan vindt deze les het. Anders downloadt de knop het één keer van de University of Toronto, 170 MB, pakt de foto's uit in één bestand van 184 MB, en verwijdert daarna de download. De pagina van CIFAR-10 noemt geen licentie, dus levert de course het niet mee; waar de foto's vandaan komen, staat in die projectles. Het uitgewerkte voorbeeld, het plaatje van de patches en de demo hebben het nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
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
def example_patch(cifar):
    import torch
    from torch import nn

    X_train, y_train, X_test, y_test = cifar.load()
    x = torch.tensor(X_test[82:83], dtype=torch.float32) / 255   # één foto, de auto, als een batch van 1
    p = 4  # verander: p = 8
    print(x.shape, (32 // p) ** 2)
    # patch (1, 2): rij 1 van de patches, kolom 2, geteld vanaf 0
    patch = x[:, :, 1 * p:2 * p, 2 * p:3 * p]
    flat = patch.reshape(1, 3 * p * p)
    print(patch.shape, flat.shape)
    print(flat[0, :4], patch[0, 0, 0])
    # de patch embedding: één lineaire layer, dezelfde voor elke patch
    torch.manual_seed(0)
    embed = nn.Linear(3 * p * p, 16)
    print(embed(flat).shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met p = 4 heeft de foto 64 patches. Patch (1, 2) is rij 4 tot en met 7 en kolom 8 tot en met 11 van de auto: shape `(1, 3, 4, 4)`, en als één rij `(1, 48)`. De eerste 4 getallen van die rij, 0,2588, 0,2510, 0,2510 en 0,2510, zijn de eerste rij van zijn rood: `reshape` leest de patch zoals hij staat, het rood rij voor rij, dan het groen, dan het blauw. De patch embedding maakt van de 48 getallen er 16: shape `(1, 16)`, het token van deze patch. Met p = 8 heeft de foto 16 patches, is patch (1, 2) rij 8 tot en met 15 en kolom 16 tot en met 23, en heeft één rij 3 · 8 · 8 = 192 getallen; de embedding maakt er nog steeds 16 van. Minder, grotere patches geven minder tokens, dus heeft attention minder te vergelijken, maar elk token vat meer van de foto samen. De paper nam 16 bij 16 voor foto's van 224 bij 224: 196 patches.
    """)
    return


@app.cell(hide_code=True)
def patches_picture_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    mo.vstack([mo.md(r"""
    Het plaatje: de auto uit het uitgewerkte voorbeeld, met lijnen tussen zijn 64 patches van 4 bij 4, en patch (1, 2), rij 1 en kolom 2, geteld vanaf 0, omlijst. Ernaast die patch, vergroot, en eronder zijn 48 getallen als één rij: eerst zijn 16 getallen rood, dan 16 groen, dan 16 blauw, elk rij voor rij, elk getal in zijn eigen kleur, zo sterk als het is. Die rij gaat door de patch embedding.
    """), mo.Html(C.patches_picture({"whole": "de auto in 64 patches", "patch": "patch (1, 2)", "numbers": "zijn 48 getallen: rood, groen, blauw",
                               "alt": "De auto geknipt in 8 bij 8 patches van 4 bij 4 pixels, met patch (1, 2) omlijst; ernaast die patch vergroot, en eronder zijn 48 getallen als een strook: eerst de roden, dan de groenen, dan de blauwen."}))])
    return


@app.cell(hide_code=True)
def text_exercise_patches():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. In patches knippen

        Maak `patches(x, p)` af: geef de patches van elke foto van `x`, met shape `(N, 3, H, W)`, terug als shape `(N, (H // p) * (W // p), 3 * p * p)`. De patches gaan rij voor rij, zoals in het uitgewerkte voorbeeld, en elke patch heeft zijn rood, zijn groen en zijn blauw, elk rij voor rij, zoals `reshape` daar patch (1, 2) las. Stap 1 tot en met 3 van het recept doen het voor alle patches tegelijk. De check geeft je functie verzonnen foto's, ook een die niet vierkant is, en onder de check zet hij de patches van een verzonnen foto terug op hun plek, met een zwarte lijn ertussen. Klopt het, dan bewaart hij je functie in je project, als `work/projects/cnn/patches.py`.
        """),
        L.task("Laat `patches` elke foto in zijn patches knippen, elk een rij getallen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_patches():
    box_patches = L.code_box("patches", lines=8)
    box_patches
    return (box_patches,)


@app.cell(hide_code=True)
def check_patches(box_patches):
    result_patches = L.check("patches", box_patches.value)
    result_patches
    return (result_patches,)


@app.cell(hide_code=True)
def hints_patches():
    L.hints(
        "patches",
        [
            "Een rij van 32 pixels is 8 stukken van 4. `reshape` kan dat zeggen: een axis van `h` wordt twee axes, `h // p` en `p`. Doe dat voor de rijen en voor de kolommen.",
            "Na `x.reshape(n, c, h // p, p, w // p, p)` zeggen de axes: foto, kleur, rij patches, rij binnen de patch, kolom patches, kolom binnen de patch. Welke twee axes zeggen welke patch het is? Zet die met `permute` direct na de foto.",
            "`x = x.permute(0, 2, 4, 1, 3, 5)` geeft shape `(n, h // p, w // p, c, p, p)`. Dan maakt één `reshape` van de patches één axis en van de getallen van een patch één axis: `x.reshape(n, ___, c * p * p)`.",
        ],
        why="`permute` verandert alleen de volgorde waarin de axes gelezen worden, dus kopieert de laatste `reshape` de getallen in hun nieuwe volgorde. De ViT van torchvision doet deze stap als een convolutie met een kernel size en een stride van p, gevolgd door een reshape en een permute: dezelfde patches, en de patch embedding in dezelfde stap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_attention():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Attention zonder mask

        Maak `SelfAttention` af. Hij bouwt voort op je `CausalSelfAttention` uit [Het transformer block](/course/open/attn-transformer-block?lang=nl), uit `work/projects/llm/attention.py`: `class SelfAttention(CausalSelfAttention)` houdt zijn `__init__`, met zijn vier layers `query`, `key`, `value` en `out`, en krijgt een eigen `forward`, zoals in [Voortbouwen op een andere class](/course/open/py-inheritance?lang=nl). Die forward doet wat de jouwe deed, voor een mini-batch met shape `(B, n, d)`, zonder de mask: elke patch neemt iets over van elke patch van zijn eigen foto. De check geeft hem verzonnen patches, en kijkt ook of de eerste patch iets overneemt van de laatste. Klopt het, dan bewaart hij je class in je project, als `work/projects/cnn/self_attention.py`.
        """),
        L.task("Laat in `SelfAttention` elke patch iets overnemen van elke patch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attention():
    box_attention = L.code_box("self-attention", lines=14)
    box_attention
    return (box_attention,)


@app.cell(hide_code=True)
def check_attention(box_attention):
    result_attention = L.check("self-attention", box_attention.value)
    result_attention
    return (result_attention,)


@app.cell(hide_code=True)
def hints_attention():
    L.hints(
        "self-attention",
        [
            "Open `work/projects/llm/attention.py`: je `forward` daar is bijna deze. Wat maakte hem causal?",
            "Per head: de scores `q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2)`, gedeeld door de wortel van `self.d_head`, dan de softmax langs axis 2, keer `v[:, :, a:b]`. Geen `future`, geen `masked_fill`.",
            "In de loop: `S = ___ / math.sqrt(self.d_head)` en `outputs.append(torch.softmax(S, dim=2) @ v[:, :, a:b])`; na de loop: `return self.out(torch.cat(outputs, dim=___))`.",
        ],
        why="Een encoder, een transformer die een hele input in één keer leest, heeft geen mask; een decoder, die schrijft, heeft er een. De ViT is een encoder, net als BERT, een taalmodel uit 2018 dat een hele tekst in één keer leest; het class token van de ViT komt van BERT.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De Vision Transformer

        Maak `ViT(d, h, depth, p)` af. In `__init__`: `patch`, de patch embedding; `cls`, het class token, een parameter met shape `(1, 1, d)` die met nullen begint; en `pos`, een `nn.Embedding` met een geleerde positie voor elk van de `n` patches en het class token. Maak ze in die volgorde: dan krijgen ze dezelfde willekeurige beginwaarden als in de oplossing van de course. De blocks zijn je `Block` uit [Het transformer block](/course/open/attn-transformer-block?lang=nl), uit `work/projects/llm/block.py`, elk met je `SelfAttention` in plaats van zijn `CausalSelfAttention`; die regels zijn gegeven, en vervangen de `attention` van elk block, zoals je `Block` hem noemt. In `forward`: de patches door de patch embedding, het class token vóór de tokens van elke foto, en de posities van alle `n + 1` tokens erbij opgeteld; de blocks zijn gegeven; dan het class token na het laatste block door `self.norm` en `self.out`. De check maakt ViTs van verschillende groottes, telt de parameters van de ViT uit de demo, en kijkt of een verandering rechtsonder in een foto het class token bereikt. Onder de check je netwerk: een vak per layer, met wat eruit komt. Klopt het, dan bewaart hij je class in je project, als `work/projects/cnn/vit.py`.
        """),
        L.task("Laat `ViT` uit het class token 10 logits per foto geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vit():
    box_vit = L.code_box("vit", lines=28)
    box_vit
    return (box_vit,)


@app.cell(hide_code=True)
def check_vit(box_vit, result_patches, result_attention):
    result_patches, result_attention  # check again when exercise 1 or 2 saves your module
    result_vit = L.check("vit", box_vit.value)
    result_vit
    return (result_vit,)


@app.cell(hide_code=True)
def hints_vit():
    L.hints(
        "vit",
        [
            "Drie layers in `__init__`: `nn.Linear` van de getallen van een patch naar d, `nn.Parameter` om een tensor met nullen, en `nn.Embedding` met een rij voor elk token, ook het class token.",
            "In `forward`, de tokens: `self.patch(patches(x, self.p))` heeft shape `(N, n, d)`. Het class token heeft shape `(1, 1, d)`: `self.cls.expand(len(x), 1, -1)` geeft elke foto er een, en `torch.cat` langs axis 1 zet het vooraan. Tel daarna `self.pos(torch.arange(x.shape[1]))` erbij op.",
            "Aan het eind: het eerste token van elke foto, `x[:, 0]`, door `self.norm`, dan `self.out`: `return self.out(self.norm(___))`.",
        ],
        why="De ViT uit de paper leest het class token zoals BERT dat doet; de auteurs probeerden ook het gemiddelde van alle patch tokens na het laatste block, en dat werkte even goed zodra de learning rate daarvoor was afgesteld (appendix D.3). Voor elke plek wordt een positie geleerd, ook in de paper, waar posities die van rijen en kolommen wisten het niet beter deden (appendix D.4).",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Hij traint je `ViT({C.D}, {C.H}, {C.DEPTH}, {C.P})`, met {C.grouped(C.PARAMS, L.lang)} parameters, op de eerste 6.000 trainingsfoto's van CIFAR-10, uit dezelfde Dataset als je ColorCNN in [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl): de helft gespiegeld, elke foto uitgesneden met je `crop` en van kleur veranderd met je `jitter`, elke keer opnieuw. Hij traint {C.EPOCHS} epochs met je Trainer, met een learning rate van {C.number(C.LR, 2, L.lang)}, een warmup van één epoch en seed 0, en test op de eerste 1.000 testfoto's, zoals ze zijn. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Hij heeft CIFAR-10 nodig, uit het uitgewerkte voorbeeld, en oefening 1 tot en met 3, en duurt ongeveer twee minuten. Tijdens het trainen toont een grafiek de loss op de veranderde trainingsfoto's elke 10 batches, en gestreept op de testfoto's na elke epoch. Naast of onder de grafiek staat wat je ViT gokt bij één testfoto van elke soort, en daaronder hoe sterk de positie van patch (3, 3) op die van elke patch lijkt, als de cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl): hoe sterker het oranje, hoe meer ze op elkaar lijken; de patch zelf is omlijst. In het plaatje daarna is de grijze lijn de validation loss van de ColorCNN die op dezelfde veranderde foto's getraind is, de run van de course met seed 0. Daarna kun je door je run terugspoelen, vanaf vóór de eerste epoch.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train je ViT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_vit(train_button, result_patches, result_attention, result_vit):
    result_patches, result_attention, result_vit  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je ViT te trainen.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Je ViT op veranderde foto's", "x": "epoch", "y": "loss",
                                        "alt": "De loss van je ViT tot nu toe: op de veranderde trainingsfoto's elke 10 batches, en gestreept op de 1.000 testfoto's na elke epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "veranderde trainingsfoto's, elke 10 batches"), ("loss", True, "testfoto's, na elke epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), L.lang))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Je ViT** · epoch {_p['epoch']} van {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Je ViT** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Je ViT** · epoch {_p['epoch']} van {_p['epochs']} · batch {_p['batch']} van {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Maak eerst oefening 1 tot en met 3 af: dan staan je `patches`, je `SelfAttention` en je `ViT` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 staat nog niet op je computer: download het met de knop in het uitgewerkte voorbeeld, en druk dan nog eens op deze knop.*",
                "slow": "*Je project doet er hier te lang over: meer dan tien minuten om te trainen. Doet je `patches`, je attention of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "vit": "*Je `ViT` in `work/projects/cnn/vit.py`, of een module die hij gebruikt, stopt met een fout, of geeft geen 10 logits per foto. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw: de check bewaart je code dan weer.*",
                "changes": "*Je `crop` of je `jitter` in `work/projects/cnn/` stopt met een fout, of geeft geen foto met shape (3, 32, 32) terug. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening in [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl) opnieuw.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, tellen met je `count_params` in `work/projects/cnn/count.py`, of je `channel_stats` in `work/projects/cnn/channels.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding onder *Waarom* zegt."
            _labels = {"title": "Loss na elke epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation", "cnn": "ColorCNN, validation",
                       "alt": "De loss van je ViT na elke epoch: op de veranderde trainingsfoto's doorgetrokken, op de testfoto's gestreept, met een stip waar de validation loss het laagst is; in grijs de validation loss van de ColorCNN van de course, getraind op dezelfde veranderde foto's."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) + 1 else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e], _result["truth"]))
                if _e == 0:
                    _said = f"Vóór de eerste epoch: {_right} van deze {len(C.SHOWN)} foto's goed gelezen, en de posities nog willekeurig."
                else:
                    _said = f"Na epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, en {_right} van deze {len(C.SHOWN)} foto's goed gelezen."
                return C.rewind_frame(_result, _e, _labels, L.lang), _said

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Ga terug naar vóór de eerste epoch, en dan weer vooruit: vanaf welke epoch springen de buren van patch (3, 3) eruit?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Zijn validation loss was het laagst na epoch **{_best}**." if _best < C.EPOCHS else f" Zijn validation loss daalde na epoch {_best} nog steeds.")
            _near = "" if _result["near"] is None else f" De posities van naburige patches hebben nu gemiddeld een cosine similarity van {C.number(_result['near'], 2, L.lang)}, die van patches die 5 of meer uit elkaar liggen {C.number(_result['far'], 2, L.lang)}."
            _out = mo.vstack(_pic + [
                mo.md(f"""Je ViT leest **{C.grouped(_result['train'], L.lang)}** van zijn 6.000 trainingsfoto's goed, zoals ze zijn, {C.percent(_result['train'], C.N_TRAIN)}%, en **{_result['test']}** van de 1.000 testfoto's, {C.percent(_result['test'], C.N_TEST)}%, met {C.grouped(_result['params'], L.lang)} parameters.{_turn}{_near} Over 5 seeds, met de code van de course: 3.243 tot 3.458 trainingsfoto's en 489 tot 524 testfoto's; de ColorCNN op dezelfde veranderde foto's 3.917 tot 4.024 en 583 tot 619.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Wat leerde de ViT over plekken? Vóór het trainen zijn de posities willekeurig: de positie van een patch lijkt evenveel op die van zijn buren als op die van patches ver weg. Na 40 epochs op veranderde foto's, met de code van de course, over 5 seeds, heeft de positie van een patch gemiddeld een cosine similarity van 0,42 tot 0,54 met die van zijn 4 buren, en van -0,20 tot -0,10 met die van patches 5 of meer verderop. De ViT leerde uit zijn foto's welke patches naast elkaar liggen: niemand vertelde het hem. Zonder veranderingen leerde hij daar minder van: 0,22 tot 0,33 met zijn buren. Het plaatje laat het zien voor de run van de course met seed 0: voor elke patch, op zijn eigen plek, een kaartje van hoe sterk zijn positie op die van alle 64 patches lijkt, zoals figuur 7 van de paper het laat zien voor een ViT die op 300 miljoen foto's getraind is.
        """),
        mo.Html(C.positions_picture(C.POSITIONS, {"title": "Hoe sterk de geleerde posities op elkaar lijken, na 40 epochs",
                                                  "under": "elk kaartje: één patch, omlijst, op zijn eigen plek",
                                                  "alt": "Een rooster van 8 bij 8 kaartjes, één per patch, elk omlijst op zijn eigen plek. In elk kaartje is de kleur het sterkst rond de omlijste patch en vervaagt hij verder weg: zijn geleerde positie lijkt het meest op die van de patches dichtbij."})),
        mo.md(r"""
        En hoeveel foto's leest hij goed? In het plaatje de ColorCNN en de ViT na 40 epochs, zonder veranderingen en met, een stip voor elk van 5 seeds: dicht voor de trainingsfoto's zoals ze zijn, open voor de testfoto's. Zonder veranderingen leest de ViT 442 tot 465 testfoto's goed, tegen 460 tot 545 voor de ColorCNN; met veranderingen 489 tot 524, tegen 583 tot 619. De veranderingen helpen de ViT gemiddeld met ongeveer 51 testfoto's, maar de ColorCNN blijft voor.
        """),
        mo.Html(C.gap_picture({"title": "Goed gelezen na 40 epochs", "x": "goed gelezen (%)",
                               "cnn_none": "ColorCNN, geen veranderingen", "cnn_all": "ColorCNN, spiegelen, crop en kleur",
                               "vit_none": "ViT, geen veranderingen", "vit_all": "ViT, spiegelen, crop en kleur",
                               "train": "training", "test": "test", "lang": L.lang,
                               "alt": "Vier rijen stippen tussen 40 en 100 procent goed gelezen. ColorCNN, geen veranderingen: training ongeveer 100%, test ongeveer 50%. ColorCNN met spiegelen, crop en kleur: training ongeveer 66%, test ongeveer 60%. ViT, geen veranderingen: training ongeveer 78%, test ongeveer 45%. ViT met spiegelen, crop en kleur: training ongeveer 56%, test ongeveer 50%."})),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_places():
    mo.md(r"""
    ### Oefening 4. Wat de posities zeggen
    """)
    return


@app.cell(hide_code=True)
def choice_places_cell():
    choice_places = L.choice(
        "learned-places",
        question="Vóór het trainen lijkt de positie van een patch evenveel op die van zijn buren als op die van patches ver weg. Na 40 epochs lijkt hij het meest op die van zijn buren: gemiddeld 0,42 tot 0,54, tegen -0,20 tot -0,10 voor patches die 5 of meer uit elkaar liggen. Wat zegt dat?",
        options={
            "given": "De course gaf de ViT de plekken van zijn patches in een rooster van 8 bij 8",
            "pixels": "Naburige patches hebben een deel van dezelfde pixels, dus gaan hun posities op elkaar lijken",
            "right": "De ViT leerde uit de foto's welke patches naast elkaar liggen: wat een convolutie vanaf het begin heeft",
        },
        correct="right",
        why={
            "given": "De posities beginnen als willekeurige rijen van een `nn.Embedding`, en niets in de ViT zegt dat patch 9 naast patch 10 ligt: elke patch is gewoon token 9 of token 10. Dat de posities van buren op elkaar gaan lijken, komt alleen uit de foto's.",
            "pixels": "De patches liggen naast elkaar, zonder overlap: geen pixel zit in twee patches. En de posities zien de pixels niet eens: ze worden geleerd voor de plekken, dezelfde in elke foto.",
            "right": "Niets in de ViT zegt welke patches buren zijn: dat hun posities op elkaar gingen lijken, kwam alleen uit de foto's. Een convolutie heeft zo'n les niet nodig: zijn filter ziet alleen buren. De ViT besteedt een deel van zijn foto's aan het leren ervan.",
        },
    )
    choice_places
    return (choice_places,)


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.md(r"""
    ### Oefening 5. Waarom de ViT meer foto's nodig heeft

    Antwoord in je eigen woorden, en vergelijk dan met het modelantwoord.
    """)
    return


@app.cell(hide_code=True)
def explain_bias_cell():
    explain_bias = L.explain(
        "inductive-bias",
        question="Op 6.000 foto's leest je ViT minder testfoto's goed dan je ColorCNN van dezelfde grootte. Gepretraind op 300 miljoen foto's deed de ViT uit de paper het even goed als de beste CNN's. Leg uit, met wat een convolutie inbouwt, waarom een ViT zoveel meer foto's nodig heeft.",
        model_answer="**Een convolutie bouwt twee aannames over foto's in**: een filter kijkt alleen naar naburige pixels, en hetzelfde filter wordt op elke plek gebruikt, dus werkt een patroon dat in de ene hoek geleerd is overal. Dat is zijn inductive bias, en die geldt voor foto's al voordat het netwerk er een heeft gezien. **Een ViT heeft daar weinig van**: zijn attention laat elke patch iets overnemen van elke patch, en zijn posities beginnen willekeurig. Dat buren bij elkaar horen, en dat een oor een oor is waar het ook staat, moet hij uit de foto's zelf leren, zoals zijn posities laten zien. **Met weinig foto's** kost hem dat foto's die het CNN aan de soorten kan besteden, dus eindigt hij lager. **Met heel veel foto's** leert de ViT die dingen, en meer dat een vast filter niet kan, zoals hoe een patch samenhangt met een patch ver weg, dus haalt hij het CNN in en kan hij het beter doen.",
        points=[
            "je noemt wat een convolutie inbouwt: hij kijkt alleen naar naburige pixels, en gebruikt op elke plek hetzelfde filter",
            "je zegt dat een ViT dat niet heeft: elke patch kan iets overnemen van elke patch, en zijn posities beginnen willekeurig",
            "je zegt dat de ViT die dingen uit zijn foto's moet leren, wat veel foto's kost, en dat hij met genoeg foto's het CNN inhaalt",
        ],
    )
    explain_bias
    return (explain_bias,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je block uit [Het transformer block](/course/open/attn-transformer-block?lang=nl), met je attention zonder zijn mask, leest nu foto's: in patches geknipt door je `patches`, elke patch een token via een patch embedding, met geleerde posities zoals in [Attention laten weten waar alles staat](/course/open/attn-positions?lang=nl), en een class token vooraan. Je ViT trainde met je Trainer op de veranderde foto's uit [Data augmentation voor foto's: spiegelen, uitsnijden en kleur](/course/open/cv-augment-photos?lang=nl), en eindigde onder je ColorCNN van dezelfde grootte: een convolutie bouwt in dat buren bij elkaar horen en dat een patroon overal hetzelfde betekent, en de ViT moet dat uit zijn foto's leren, zoals zijn posities lieten zien. Je project `cnn` heeft nu `patches.py`, `self_attention.py` en `vit.py`. In de volgende les, *Waar een Vision Transformer naar kijkt*, kijk je naar wat het class token van welke patches overneemt, in een ViT die van veel meer foto's leerde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_patches, result_attention, result_vit, choice_places, explain_bias):
    L.wrap_up(result_patches, result_attention, result_vit, choice_places, explain_bias)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
