# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-frozen-features. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-frozen-features", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl) en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="De model card van de cijferlezer van de course zegt: alleen cijfers zoals in QMNIST. Jouw cijfers komen van een oude scanner die op elke afbeelding grijze spikkels zet. Wat doe je eerst?",
        options={
            "trust": "Gebruik hem zoals hij is: gespikkelde cijfers zijn nog steeds cijfers",
            "right": "Test hem op een deel van je cijfers: dan zie je hoe goed hij leest",
            "never": "Laat hem liggen: de card zegt dat hij niet voor jouw cijfers is",
        },
        correct="right",
        why={
            "trust": "De card noemt dit als beperking: wat hij op andere afbeeldingen doet, weet je pas als je het probeert. Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "never": "Een beperking op de card zegt dat je moet controleren, niet dat het model niets waard is. Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "right": "Stap 4 van het recept. Van 1.000 gespikkelde testcijfers leest hij er 729 goed: minder dan de 929 van schone cijfers, maar verre van niets.",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Een netwerk met 52.000 parameters traint vele epochs op maar 100 afbeeldingen. Wat verwacht je van zijn twee loss curves?",
        options={
            "both": "Beide dalen tot ongeveer 0: 100 afbeeldingen zijn makkelijk te leren",
            "right": "Alleen de training loss daalt tot ongeveer 0: het leert de 100 uit het hoofd",
            "none": "Geen van beide daalt: 100 afbeeldingen zijn te weinig voor een netwerk om iets te leren",
        },
        correct="right",
        why={
            "both": "Op de 100 afbeeldingen wel. Maar op afbeeldingen die het niet zag, doet het het veel slechter: het leerde deze 100 uit het hoofd. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "none": "100 afbeeldingen zijn ruim genoeg om uit het hoofd te leren: de training loss daalt prima. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "Overfitting: met veel parameters en weinig voorbeelden leert het netwerk de voorbeelden, niet de taak.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        De cijferlezer leest gespikkelde cijfers van de oude scanner slechter: 729 van de 1.000. Gelabeld heb je er maar 100. Je kunt daarop een nieuw netwerk trainen. Of je houdt wat de cijferlezer al over cijfers weet, geleerd op 6.000 cijfers, en laat alleen een nieuwe laatste layer jouw taak leren.

        **Een voorbeeld uit het dagelijks leven.** Een ervaren kok begint in een nieuw restaurant. Opnieuw leren koken hoeft niet: messen, vuur en smaak kent de kok al. Alleen de kaart is nieuw, en die zit er in een paar dagen in.
        """),
        mo.Html(C.speckle_picture({"title": "Drie testcijfers", "clean": "zoals de set van de course ze heeft", "scanner": "zoals de oude scanner ze geeft",
                           "alt": "Drie handgeschreven cijfers, 7, 2 en 1, bovenaan schoon, en daaronder dezelfde drie met grijze spikkels over de hele afbeelding."})),
    ])
    return


@app.cell(hide_code=True)
def predict_frozen():
    guess_frozen = L.guess(
        "frozen",
        question="Met de 100 gelabelde cijfers van de oude scanner, twee manieren: een nieuw netwerk dat vanaf nul traint, of de cijferlezer met al zijn layers zoals ze zijn en alleen een nieuwe laatste layer getraind. Welke leest er meer van 1.000 gespikkelde testcijfers goed?",
        options={
            "scratch": "Vanaf nul: dat leert alles voor deze cijfers",
            "same": "Ongeveer evenveel: beide leren van dezelfde 100 cijfers",
            "right": "De cijferlezer met een nieuwe laatste layer: er hoeft minder te leren",
        },
        reveal=f"**De cijferlezer met een nieuwe laatste layer: {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]}**, over 5 seeds, tegen {C.REFERENCE['scratch'][0]} tot {C.REFERENCE['scratch'][1]} vanaf nul, en {C.REFERENCE['as-is'][0]} voor de cijferlezer zoals hij is. Vanaf nul leren 52.000 getallen van 100 afbeeldingen; met de geleende layers maar 650, op features die al cijfers beschrijven.",
    )
    guess_frozen
    return (guess_frozen,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Features en een laatste layer.** Alle layers van de cijferlezer voor zijn laatste maken van een afbeelding 64 getallen, zijn hidden layer: features die het cijfer beschrijven. De laatste layers, `embed` en `out`, maken van die 64 getallen 10 logits. Het netwerk is dus een **feature extractor** met een klein laatste deel erop.

        **Bevriezen.** `p.requires_grad = False` voor een parameter is het omgekeerde van `requires_grad=True` uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl): PyTorch berekent er geen gradient voor, dus de optimizer laat hem zoals hij is. Dat is **bevriezen**: de geleende layers houden wat ze leerden.

        **Een nieuwe laatste layer.** `model.embed = nn.Identity()` geeft de 64 verborgen getallen door zoals ze zijn, en `model.out = nn.Linear(64, 10)` is een nieuwe layer van die 64 getallen naar 10 logits. Een nieuwe layer heeft `requires_grad` aan: die is het enige deel dat leert, 64 · 10 + 10 = 650 getallen.

        Wat een model op de ene taak leerde opnieuw gebruiken voor een andere, is **transfer learning**. Het werkt als de nieuwe data lijken op de oude: gespikkelde cijfers, of een nieuwe vraag over cijfers. Afbeeldingen die helemaal niet op cijfers lijken, hebben andere features nodig.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **transfer learning** | | opnieuw gebruiken wat een model op de ene taak of dataset leerde, voor een andere | |
        | **feature extractor** | | een pretrained netwerk dat je alleen gebruikt om inputs om te zetten in features, waarop een nieuw klein model jouw taak leert | |
        | **bevriezen** | | parameters tijdens het trainen houden zoals ze zijn: geen gradient, dus de optimizer laat ze | `p.requires_grad = False` |
        | `nn.Identity()` | n n identity | een layer die zijn input onveranderd doorgeeft | `model.embed = nn.Identity()` |
        | `named_parameters()` | named parameters | elke parameter van een model, met zijn naam, zoals `out.weight` | `for name, p in model.named_parameters():` |
        """),
        L.watch_out("Eerst bevriezen, dan de nieuwe layer maken. Bevriezen zet `requires_grad = False` voor elke parameter die het model op dat moment heeft: bevries je nadat je de nieuwe layer maakte, dan is die ook bevroren, en leert er niets."),
        mo.md(r"""
        De figuur: de cijferlezer na de verandering. Zijn eerste drie layers zijn bevroren; de nieuwe `out` is de enige layer die leert.
        """),
        mo.Html(C.layers_picture({"title": "Bevroren layers en een nieuwe laatste layer", "conv1": "convolutie, 8 channels", "conv2": "convolutie, 16 channels",
                                  "hidden": "64 features van het cijfer", "out": "nieuw: van 64 naar 10 logits", "frozen": "bevroren",
                                  "learns": "leert", "gone": "embed geeft de 64 features door: nn.Identity()",
                                  "alt": "Vier layers van boven naar beneden: conv1, conv2 en hidden zijn bevroren en houden hun getallen; de nieuwe out, van 64 naar 10 logits, is de enige layer die leert. De oude embed geeft de 64 features onveranderd door."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een pretrained netwerk opnieuw te gebruiken voor een nieuwe taak met weinig gelabelde voorbeelden:

    1. Laad het pretrained netwerk, zoals in [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).
    2. Bevries elke parameter: `for p in model.parameters(): p.requires_grad = False`.
    3. Vervang zijn laatste deel: `model.embed = nn.Identity()`, en een nieuwe `model.out = nn.Linear(64, k)` voor $k$ antwoorden.
    4. Train alleen de nieuwe layer op je gelabelde voorbeelden, en test op andere.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vier stappen voor een nieuwe vraag over schone cijfers: is het cijfer oneven of even? Twee antwoorden, dus de nieuwe layer gaat van 64 naar 2. `digit_cnn()` geeft de cijferlezer van de course, geladen zoals in stap 3 van het recept van [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl), in eval mode. De 100 gelabelde cijfers zijn de eerste 100 trainingsafbeeldingen; het label van elk is de rest bij deling van zijn cijfer door 2, en die geeft `% 2`: 1 voor oneven, 0 voor even. `C.train(model, params, X, y, lr, epochs)` is de training loop van de course, zoals in [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), maar met de optimizer Adam, die een latere les uitlegt; `params` zijn de parameters die mogen leren.
    """)
    return


@app.cell
def example_odd():
    import numpy as np
    import torch
    from torch import nn
    from everything_ai.models import digit_cnn

    _data = np.load("data/qmnist-small.npz")
    _X = torch.tensor(_data["train_images"][:100], dtype=torch.float32) / 255
    _y = torch.tensor(_data["train_labels"][:100]) % 2
    _X_test = torch.tensor(_data["test_images"], dtype=torch.float32) / 255
    _y_test = torch.tensor(_data["test_labels"]) % 2
    # 1. de pretrained cijferlezer
    torch.manual_seed(0)
    _model = digit_cnn()
    # 2. bevries elke parameter
    for _p in _model.parameters():
        _p.requires_grad = False
    # 3. een nieuwe laatste layer: de 64 verborgen getallen naar 2 antwoorden, oneven of even
    _model.embed = nn.Identity()
    _model.out = nn.Linear(64, 2)
    for _name, _p in _model.named_parameters():
        print(_name, _p.requires_grad)
    # 4. train alleen de nieuwe layer, en test dan
    C.train(_model, _model.out.parameters(), _X, _y, 0.01, 30)
    print(C.right(_model, _X_test, _y_test), "van de", len(_y_test), "goed")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `named_parameters` geeft elke parameter met zijn naam: alleen `out.weight` en `out.bias` leren nog. Na 30 epochs op 100 cijfers zegt de nieuwe layer bij 944 van de 1.000 testcijfers goed of het cijfer oneven of even is, en over 5 seeds bij 926 tot 952. Een nieuw netwerk dat vanaf nul traint op dezelfde 100 cijfers, haalt er 802 tot 827, over 5 seeds: de geleende features onderscheiden de cijfers al, en uit het cijfer volgt of het oneven of even is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_head():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Bevriezen, en een nieuwe laatste layer

        Maak `new_head(model)` af: stap 2 en 3 van het recept, voor de 10 cijfers van de oude scanner. De functie krijgt de cijferlezer en geeft die terug met elke oude parameter bevroren en een nieuwe laatste layer van 64 naar 10. De check roept `new_head(reader)` aan en kijkt welke parameters nog leren, wat de shape van de weight van `out` is, en wat de logits voor het eerste testcijfer zijn.
        """),
        L.task("Laat `new_head` de cijferlezer bevriezen en hem een nieuwe laatste layer van 64 naar 10 geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_head():
    box_head = L.code_box("new-head", lines=10)
    box_head
    return (box_head,)


@app.cell(hide_code=True)
def check_head(box_head):
    result_head = L.check("new-head", box_head.value)
    result_head
    return (result_head,)


@app.cell(hide_code=True)
def hints_head():
    L.hints(
        "new-head",
        [
            "Stap 2 en 3 van het uitgewerkte voorbeeld, met 10 antwoorden in plaats van 2.",
            "In de loop: `p.requires_grad = False`.",
            "`nn.Identity()` voor `embed`, en `nn.Linear(64, 10)` voor `out`, na de loop.",
        ],
        why="De loop bevriest alle 10 parameters van de cijferlezer; de nieuwe `out` komt erna, dus die is niet bevroren. `nn.Identity()` geeft de 64 verborgen getallen zoals ze zijn door aan de nieuwe layer.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Jouw nieuwe laatste layer op de cijfers van de oude scanner

    De cel hieronder geeft de cijferlezer aan jouw `new_head` zodra die klopt, anders aan die van de course, en traint de nieuwe laatste layer op de 100 gelabelde cijfers van de oude scanner: {C.HEAD_EPOCHS} epochs met een learning rate van {L.number(C.HEAD_LR, 2)}. Ter vergelijking traint de cel een nieuw netwerk met dezelfde layers, die daar allemaal leren, vanaf nul op dezelfde 100 cijfers: {C.SCRATCH_EPOCHS} epochs met een learning rate van {L.number(C.SCRATCH_LR, 3)}. Dan test de cel beide, en de cijferlezer zoals hij is, op 1.000 gespikkelde testcijfers. Dat duurt een paar seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_frozen(box_head, result_head):
    _d = C.demo(box_head.value if result_head.state == "correct" else None)
    _who = "jouw `new_head`" if _d["yours"] else "de `new_head` van de course"
    mo.md(f"""Met {_who}, van de 1.000 gespikkelde testcijfers:

| | goed gelezen |
|---|---|
| de cijferlezer zoals hij is | {_d['as-is']} |
| een nieuw netwerk, vanaf nul op de 100 | {_d['scratch']} |
| bevroren, met een nieuwe laatste layer getraind op de 100 | {_d['frozen']} |

Over 5 seeds, met de code van de course: {C.REFERENCE['scratch'][0]} tot {C.REFERENCE['scratch'][1]} vanaf nul, en {C.REFERENCE['frozen'][0]} tot {C.REFERENCE['frozen'][1]} bevroren met een nieuwe laatste layer.""")
    return


@app.cell(hide_code=True)
def text_exercise_optimizer():
    mo.md(r"""
    ### Oefening 2. Welke parameters voor de optimizer?
    """)
    return


@app.cell(hide_code=True)
def choice_optimizer_cell():
    choice_optimizer = L.choice(
        "same-optimizer",
        question="In stap 4 van het uitgewerkte voorbeeld krijgt de optimizer `model.out.parameters()`, die van de nieuwe layer. Wat verandert er als je hem `model.parameters()` geeft, allemaal, ook de bevroren?",
        options={
            "learn": "De bevroren layers leren toch: de optimizer verandert elke parameter die hij krijgt",
            "right": "Niets: de bevroren parameters krijgen geen gradient, dus de optimizer laat ze",
            "error": "PyTorch stopt: een bevroren parameter kan niet in een optimizer",
        },
        correct="right",
        why={
            "learn": "De optimizer verandert een parameter met zijn gradient, en `backward()` berekent er geen voor een parameter met `requires_grad = False`: er is niets om hem mee te veranderen.",
            "error": "PyTorch neemt het zonder klagen aan; de optimizer slaat alleen een parameter zonder gradient over.",
            "right": "Bevriezen gebeurt in `requires_grad`, niet in de optimizer. Alleen de parameters van de nieuwe layer meegeven is toch duidelijker, en scheelt de optimizer wat werk.",
        },
    )
    choice_optimizer
    return (choice_optimizer,)


@app.cell(hide_code=True)
def text_exercise_clothes():
    mo.md(r"""
    ### Oefening 3. Een ander soort afbeelding
    """)
    return


@app.cell(hide_code=True)
def choice_clothes_cell():
    choice_clothes = L.choice(
        "clothes",
        question="Je hebt een netwerk nodig dat 10 soorten kleding herkent, met 100 gelabelde afbeeldingen van 28 bij 28 uit Fashion-MNIST: grijze plaatjes van kleding, schoenen en tassen, in de vorm van MNIST. Wint de bevroren cijferlezer met een nieuwe laatste layer ook hier van een nieuw netwerk dat vanaf nul traint?",
        options={
            "right": "Nee: zijn features beschrijven cijfers, en kleding vraagt om andere",
            "yes": "Ja: features van de ene soort afbeelding helpen bij elke andere soort",
            "more": "Ja, en ruimer: kleding bestaat uit grotere vormen dan cijfers",
        },
        correct="right",
        why={
            "yes": f"De course mat het, over 5 seeds: {C.REFERENCE['clothes-frozen'][0]} tot {C.REFERENCE['clothes-frozen'][1]} van de 1.000 bevroren, tegen {C.REFERENCE['clothes-scratch'][0]} tot {C.REFERENCE['clothes-scratch'][1]} vanaf nul. De 64 features zeggen hoe een cijfer geschreven is, niet of een afbeelding een schoen of een shirt toont.",
            "more": f"Grotere vormen vragen om eigen features. De course mat het, over 5 seeds: {C.REFERENCE['clothes-frozen'][0]} tot {C.REFERENCE['clothes-frozen'][1]} van de 1.000 bevroren, tegen {C.REFERENCE['clothes-scratch'][0]} tot {C.REFERENCE['clothes-scratch'][1]} vanaf nul.",
            "right": f"De course mat het, over 5 seeds: {C.REFERENCE['clothes-frozen'][0]} tot {C.REFERENCE['clothes-frozen'][1]} van de 1.000 bevroren, tegen {C.REFERENCE['clothes-scratch'][0]} tot {C.REFERENCE['clothes-scratch'][1]} vanaf nul. Transfer learning helpt als de nieuwe data lijken op de oude.",
        },
    )
    choice_clothes
    return (choice_clothes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De cijferlezer uit [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl), nu als feature extractor, en `requires_grad` uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl), nu om hem te bevriezen. Met weinig voorbeelden overfitten 650 getallen die leren veel minder dan 52.000, zoals de curves uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) zouden laten zien. Hierna laat [Een getraind netwerk hergebruiken: eerst bevriezen, dan finetunen](/course/open/torch-fine-tune?lang=nl) ook de bevroren layers een beetje leren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_head, choice_optimizer, choice_clothes):
    L.wrap_up(result_head, choice_optimizer, choice_clothes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
