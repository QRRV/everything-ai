# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-pretrained. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-pretrained", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl) en [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sizes():
    quick_sizes = L.choice(
        "quick-sizes",
        question="Je hebt de state_dict opgeslagen van een MLP met 64 hidden neuronen. Dan maak je een MLP van dezelfde class met 128 hidden neuronen, en laad je dat bestand erin met `load_state_dict`. Wat gebeurt er?",
        options={
            "half": "De eerste 64 neuronen krijgen de opgeslagen getallen, de andere 64 houden de hunne",
            "right": "PyTorch stopt met een fout: de shapes passen niet",
            "grow": "PyTorch maakt de opgeslagen layer groter, met nullen",
        },
        correct="right",
        why={
            "half": "`load_state_dict` kopieert hele tensors, en alleen naar een tensor met dezelfde shape. Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "grow": "PyTorch verandert geen maten voor je: het model moet van dezelfde class zijn, met dezelfde maten. Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "right": "Een weight met shape (128, 784) kan er geen van (64, 784) opnemen. Een bestand met getallen heb je dus alleen iets aan met de class waarvoor het gemaakt is, en zijn maten.",
        },
    )
    quick_sizes
    return (quick_sizes,)


@app.cell(hide_code=True)
def quick_check_nc():
    quick_nc = L.choice(
        "quick-nc",
        question="Een dataset met straatfoto's heeft de licentie CC BY-NC 4.0. Mag je er een model op trainen voor een app die je verkoopt?",
        options={
            "right": "Nee: NC betekent dat je er geen geld mee mag verdienen, en met een app die je verkoopt doe je dat wel",
            "credit": "Ja, als je de maker noemt en naar de licentie linkt",
            "free": "Ja: je gebruikt de foto's alleen, je deelt ze niet",
        },
        correct="right",
        why={
            "credit": "Dat is genoeg voor CC BY. NC voegt eraan toe: niet om geld te verdienen. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "free": "De licentie bepaalt wat je met de data mag, ook als je er alleen op traint. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "En wat voor data geldt, geldt ook voor een model: zijn licentie zegt wat je ermee mag, en de licentie van zijn training data telt ook mee.",
        },
    )
    quick_nc
    return (quick_nc,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een goed model trainen kost data, tijd en computers. Vaak heeft iemand al een model voor jouw taak getraind, en deelt dat: op een **model hub**, een website vol getrainde modellen die iedereen kan downloaden. De cijferlezer van de course staat er ook. Maar aan een bestand met getallen heb je alleen iets met de code van zijn class, en het is alleen veilig en toegestaan als je weet waar het vandaan komt. Dat vertelt zijn **model card** je.

    **Een voorbeeld uit het dagelijks leven.** Een tweedehands fiets: je kunt er vandaag op rijden, zonder er een te bouwen. Maar eerst kijk je naar zijn papieren: wie hem had, hoeveel hij gereden heeft, of hij niet gestolen is, en of hij de maat heeft die je nodig hebt.
    """)
    return


@app.cell(hide_code=True)
def predict_loaded():
    guess_loaded = L.guess(
        "loaded",
        question="Je maakt een nieuw `DigitCNN`, laadt er het bestand van de course in dat je van de hub haalt, en traint het helemaal niet. Hoeveel van de 1.000 testcijfers denk je dat het goed leest?",
        options={
            "guess": "Ongeveer 100: het heeft nog niet getraind",
            "half": "Ongeveer 500: een bestand bewaart maar een deel van wat het leerde",
            "right": "Ongeveer 930: wat het trainen van de course bereikte",
        },
        reveal=f"**Ongeveer 930: {C.RIGHT}.** Alles wat het model leerde bij het trainen van de course, zit in de 51.648 getallen van het bestand; ze laden is alles wat nodig is. Een nieuw `DigitCNN` zonder die getallen leest er 85 tot 129 van de 1.000 goed, bij 5 seeds: even goed als één van de 10 cijfers gokken.",
    )
    guess_loaded
    return (guess_loaded,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(rf"""
        ## Het idee

        **Een model hub** bewaart getrainde modellen, elk in een repository van bestanden met een eigen adres. De bekendste is Hugging Face; torchvision en timm, twee Python-libraries, bieden ook getrainde modellen voor afbeeldingen aan: één aanroep geeft een model van de class met zijn getallen. De cijferlezer van de course is de repository `{C.REPO}` op Hugging Face, op [{C.page()}]({C.page()}). Hij bevat `model.pt`, de state_dict, en `README.md`, zijn model card. Elk bestand kun je zonder account downloaden van een vast adres, zoals `{C.address("model.pt")}`.

        **Een model card** is voor een model wat een datasheet uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) voor data is: wat het model doet, zijn class en grootte, de data waarop het getraind is en hun licentie, de licentie van het model zelf, hoe goed het is, en wat het niet dekt. Je leest hem voordat je het model gebruikt.

        **Laad het veilig.** Een `.pt`-bestand van iemand anders kan bij het laden code op je computer uitvoeren, tenzij `torch.load` alleen tensors leest. Sinds PyTorch 2.6 doet het dat standaard; schrijf toch `torch.load(path, weights_only=True)` voor elk bestand dat je niet zelf gemaakt hebt: dan weigert PyTorch al het andere, wat de standaard ook is.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **pretrained model** | | een model dat iemand anders getraind heeft, gedeeld als zijn class en zijn parameters | |
        | **model hub** | | een website die getrainde modellen bewaart, elk in een repository met zijn bestanden | |
        | **model card** | | de beschrijving van een model: taak, class en grootte, training data, licentie, hoe goed het is, zijn grenzen | `README.md` |
        | `weights_only=True` | weights only | `torch.load` leest alleen tensors en eenvoudige waarden uit het bestand | `torch.load(path, weights_only=True)` |
        """),
        L.watch_out("Een `.pt`-bestand bevat alleen namen en getallen, niet de code van zijn class. Je hebt dus de class nodig die de card noemt, `DigitCNN`: zijn code staat in de card, en in de course als `everything_ai.models.DigitCNN`. Zijn eerste layers zijn convoluties, een soort layer voor afbeeldingen die een latere les uitlegt; om het model te laden en te gebruiken heb je alleen de class nodig, niet hoe het vanbinnen werkt."),
        mo.md(r"""
        De figuur: de repository op de hub, met het model en zijn card, en de weg van `model.pt` naar een eigen model.
        """),
        mo.Html(C.hub_picture({"title": "Van de hub naar je model", "hub": "Op Hugging Face:", "card": "de model card",
                               "weights": "de state_dict, 51.648 getallen",
                               "steps": [("torch.load(path, weights_only=True)", "download model.pt en lees het"),
                                         ("model = DigitCNN()", "een nieuw model van de class op de card"),
                                         ("model.load_state_dict(state)", "de getallen uit het bestand erin"),
                                         ("model.eval()", "eval mode, om cijfers te lezen")],
                               "alt": "De repository QRRV/everything-ai-qmnist-cnn op Hugging Face bevat README.md, de model card, en model.pt, de state_dict van 51.648 getallen. Na model.pt volgen vier stappen: torch.load met weights_only=True, een nieuw DigitCNN, load_state_dict, en eval mode."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een pretrained model te gebruiken:

    1. Zoek het op een hub, en lees zijn model card: wat het doet, zijn class en grootte, zijn training data, zijn licentie, zijn grenzen. Is het voor jouw taak, en mag je het zo gebruiken?
    2. Download zijn bestand.
    3. Laad het bestand met `torch.load(path, weights_only=True)` in een nieuw model van de class op de card, met `load_state_dict`, en zet het in eval mode.
    4. Probeer het op eigen data: doet het wat de card zegt?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vier stappen voor de cijferlezer van de course. Het commentaar met `#` geeft de stap van het recept. `C.card()` geeft de model card van de hub, in het Engels zoals de meeste op Hugging Face, en `C.hub_file("model.pt")` het bestand, met waar het vandaan kwam: `hub` als het gedownload is, of `course` als het downloaden niet lukt, bijvoorbeeld zonder verbinding: dan gebruikt de course zijn eigen kopie, hetzelfde bestand.
    """)
    return


@app.cell
def example_card():
    # 1. de model card
    mo.md(C.card())
    return


@app.cell
def example_load():
    import torch
    from everything_ai.models import DigitCNN

    # 2. het bestand
    _path, _where = C.hub_file("model.pt")
    print("model.pt van:", _where)
    # 3. laad het in een nieuw model van de class op de card
    _state = torch.load(_path, weights_only=True)
    print(len(_state), "tensors:", list(_state))
    _model = DigitCNN()
    print(_model.load_state_dict(_state))
    _model.eval()
    # 4. probeer het op de 1.000 testcijfers
    _X, _y = C.qmnist_test()
    with torch.no_grad():
        _pred = _model(_X).argmax(dim=1)
    print((_pred == _y).sum().item(), "van", len(_y), "goed")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    De card zegt dat de class `DigitCNN` is, dat het model getraind is op 6.000 cijfers van QMNIST, dat het de licentie CC BY 4.0 heeft, en dat het 929 van de 1.000 testcijfers goed leest. Het bestand bevat 10 tensors, twee voor elk van de 5 layers op de card: een weight en een bias; `load_state_dict` vindt voor elk een plek. En het model leest, zonder dat je zelf traint, {C.RIGHT} van de 1.000 testcijfers goed: wat de card zegt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welk model?
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick-model",
        question="Voor een app die je verkoopt heb je een model nodig dat cijfers leest die op papier geschreven zijn en met een telefoon gefotografeerd. Drie model cards. Welke kies je?",
        options={
            "course": "A: getraind op QMNIST, witte cijfers op zwart van 28 bij 28, CC BY 4.0, 929 van 1.000 goed",
            "nc": "B: getraind op 50.000 telefoonfoto's van cijfers op papier, CC BY-NC 4.0, 97 van 100 goed",
            "right": "C: getraind op 30.000 telefoonfoto's van cijfers op papier, CC BY 4.0, 94 van 100 goed",
        },
        correct="right",
        why={
            "course": "Zijn card zegt: alleen cijfers zoals in QMNIST, wit op zwart, in het midden, 28 bij 28 pixels. Een telefoonfoto van een cijfer op papier is donker op licht, en veel groter.",
            "nc": "Het past bij de taak, en scoort het best, maar NC betekent dat je er geen geld mee mag verdienen: niet in een app die je verkoopt.",
            "right": "Het is getraind op afbeeldingen zoals de jouwe, en met CC BY mag je je app verkopen als je de makers noemt. Probeer het dan op eigen foto's: stap 4.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_exercise_load():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Laad de cijferlezer

        Maak `load_reader(path)` af: stap 3 van het recept, voor een bestand met een state_dict van een `DigitCNN`. Hij geeft het geladen model terug, in eval mode. De check roept `load_reader(path)` aan met het bestand van de course, en met twee andere bestanden, en kijkt naar wat er terugkomt: zijn class, zijn modus, en de cijfers die het leest voor de eerste 8 testafbeeldingen.
        """),
        L.task("Laat `load_reader` een `DigitCNN` teruggeven met de getallen van het bestand, in eval mode."),
    ])
    return


@app.cell(hide_code=True)
def exercise_load():
    box_load = L.code_box("load-reader", lines=10)
    box_load
    return (box_load,)


@app.cell(hide_code=True)
def check_load(box_load):
    result_load = L.check("load-reader", box_load.value)
    result_load
    return (result_load,)


@app.cell(hide_code=True)
def hints_load():
    L.hints(
        "load-reader",
        [
            "Volg stap 3 van het uitgewerkte voorbeeld, met `path` voor het bestand.",
            "Eerst de getallen uit het bestand, `torch.load(path, weights_only=True)`; dan een nieuw model, `DigitCNN()`.",
            "`model.load_state_dict(state)` kopieert de getallen naar het model, en `model.eval()` zet het in eval mode.",
        ],
        why="Een nieuw `DigitCNN()` begint met willekeurige getallen; `load_state_dict` vervangt elk ervan door de getallen uit het bestand. Pas dan leest het cijfers zoals de card zegt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_safe():
    mo.md(r"""
    ### Oefening 3. Een bestand van een onbekende
    """)
    return


@app.cell(hide_code=True)
def choice_safe_cell():
    choice_safe = L.choice(
        "safe-load",
        question="Op een forum deelt een onbekende `reader.pt`, met de belofte dat het een betere cijferlezer is dan die van de course. Er is geen model card. Wat doe je?",
        options={
            "right": "Het niet gebruiken tot je weet waar het vandaan komt; en als je het laadt, alleen met `weights_only=True`",
            "try": "Het laden met `torch.load` en testen: een bestand met getallen kan geen kwaad",
            "never": "Nooit een model gebruiken dat je niet zelf getraind hebt",
        },
        correct="right",
        why={
            "try": "Een `.pt`-bestand kan meer bevatten dan getallen: tenzij `torch.load` alleen tensors leest, kan het bij het laden code op je computer uitvoeren. En zonder card ken je zijn data en licentie niet.",
            "never": "Dan kun je geen enkele hub gebruiken. Een model met een card van een maker die je kunt nagaan, veilig geladen, kun je prima gebruiken.",
            "right": "Zonder card ken je zijn training data en zijn licentie niet; en `weights_only=True` zorgt dat laden alleen tensors leest.",
        },
    )
    choice_safe
    return (choice_safe,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De state_dict en `load_state_dict` uit [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl), nu voor een bestand dat iemand anders maakte, en de datasheet en licenties uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl), nu voor een model: zijn model card. In de volgende les, [Een pretrained netwerk als feature extractor](/course/open/torch-frozen-features?lang=nl), houd je de layers van de cijferlezer zoals ze zijn, en train je alleen een nieuwe laatste layer voor een andere taak.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pick, result_load, choice_safe):
    L.wrap_up(choice_pick, result_load, choice_safe)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
