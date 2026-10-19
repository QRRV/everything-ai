# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cnn-vgg-blocks. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-vgg-blocks", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl) en [LeNet-5, het CNN dat cheques las](/course/open/cnn-lenet?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_receptive():
    quick_receptive = L.choice(
        "quick-receptive",
        question="Twee convoluties van 5 bij 5 met stride 1, na elkaar, zonder pooling. Hoe breed is het stuk van de afbeelding waarvan één getal van de tweede afhangt?",
        options={
            "ten": "10 pixels: twee keer 5",
            "right": "9 pixels: de tweede voegt er 4 toe",
            "five": "5 pixels: elk filter is 5 breed",
        },
        correct="right",
        why={
            "ten": "De stukken overlappen: de eerste layer ziet 5 pixels, en de tweede kijkt naar 5 getallen van de eerste, elk 1 pixel verder: 4 pixels meer. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "five": "Dat is de eerste layer. De tweede kijkt naar 5 getallen van de eerste, elk met een eigen stuk, één plek verschoven: 4 pixels meer. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "right": "5, dan 9: het receptive field. Een filter van k bij k voegt er k − 1 toe; vandaag stapel je filters van 3 bij 3, die er elk 2 toevoegen.",
        },
    )
    quick_receptive
    return (quick_receptive,)


@app.cell(hide_code=True)
def quick_check_padding():
    quick_padding = L.choice(
        "quick-padding",
        question="De eerste convolutie van LeNet-5 houdt 28 bij 28 met een filter van 5 bij 5 en padding 2. Welke padding houdt de grootte bij een filter van 3 bij 3?",
        options={
            "two": "2: zoals in LeNet-5",
            "none": "0: een klein filter heeft er geen nodig",
            "right": "1: één rand van nullen",
        },
        correct="right",
        why={
            "two": "Met padding 2 maakt een filter van 3 bij 3 de afbeelding 2 breder: 28 + 4 − 3 + 1 = 30. Zie [LeNet-5, het CNN dat cheques las](/course/open/cnn-lenet?lang=nl).",
            "none": "Zonder padding past een filter van 3 bij 3 op 28 − 3 + 1 = 26 plekken per zijde. Zie [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl).",
            "right": "De helft van het filter, naar beneden afgerond: 2 voor 5 bij 5, 1 voor 3 bij 3. Vandaag heeft elke convolutie die.",
        },
    )
    quick_padding
    return (quick_padding,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In 2014 bouwden Karen Simonyan en Andrew Zisserman van de Visual Geometry Group in Oxford **VGG**: CNN's van 11 tot 19 layers met weights, waar je LeNet-5 er 5 heeft, met bijna overal filters van 3 bij 3. Op ImageNet, meer dan een miljoen foto's van duizend soorten, maakten de diepere minder fouten, en VGG werd tweede in de classificatiewedstrijd van dat jaar. Zijn blok, een paar kleine convoluties en dan pooling, zit nog in veel CNN's. In deze stap van je project bouw je dat blok, en meet je wat diepte doet voor je eigen CNN.
        """),
        draw.project_steps(["Je eerste CNN", "LeNet-5, het CNN dat cheques las", "VGG: dieper met kleine filters",
                            "ResNet: diepte die blijft werken"],
                           2, here_label="← vandaag", title="Waar je project staat: je eigen CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_depth():
    guess_depth = L.guess(
        "depth",
        question="Je bouwt netwerken van twee VGG blocks, zoals je SmallCNN, met 1, 2, 4 of 8 convoluties van 3 bij 3 per blok, en traint elk 5 epochs met je Trainer op dezelfde 6.000 afbeeldingen. Welke leest de meeste van de 1.000 testafbeeldingen goed?",
        options={
            "eight": "8 per blok: het diepste ziet het meest",
            "four": "4 per blok: diep, maar niet te diep",
            "right": "1 of 2 per blok: kleine afbeeldingen hebben weinig layers nodig",
        },
        reveal="**1 of 2 per blok.** Over 5 seeds, met de code van de course, leest 1 per blok, je SmallCNN, er 864 tot 879 van de 1.000 goed, en 2 per blok 859 tot 883. Met 4 per blok is het 832 tot 869, bij elke seed onder 1 per blok. Met 8 per blok, 17 layers met weights, leert geen enkele run iets: 105 goed, één soort voor elke afbeelding. Op 6.000 kleine afbeeldingen helpt diepte hier niet; en een gewoon netwerk dat te diep is, traint helemaal niet.",
    )
    guess_depth
    return (guess_depth,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Kleine filters zien even ver.** Zoals in [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl) voegt elke convolutie van 3 bij 3 er 2 toe aan de breedte van het stuk waarvan een getal afhangt: twee ervan zien 5 bij 5, drie zien 7 bij 7, net als één filter van 7 bij 7.

        **Met minder weights.** Van $C$ naar $C$ channels heeft één convolutie van 7 bij 7 $49C^2$ weights: $C$ filters van $C \times 7 \times 7$. Drie van 3 bij 3 hebben er $3 \times 9C^2 = 27C^2$. De ene convolutie heeft er 81% meer nodig. En de stapel heeft een ReLU na elke convolutie: drie, waar de ene convolutie er één heeft.

        **Een VGG block** is $n$ convoluties van 3 bij 3 met padding 1, elk met een ReLU erna, en dan max pooling van 2 bij 2. De convoluties houden de hoogte en breedte; de pooling halveert ze. VGG verdubbelt de channels van blok tot blok: 64, 128, 256, 512, en 512 in het laatste.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **VGG block** | vee gee gee block | $n$ convoluties van 3 bij 3 met padding 1, elk met een ReLU, dan max pooling van 2 bij 2 | `VGGBlock(c_in, c_out, n)` |
        | **VGG** | vee gee gee | de CNN's van Simonyan en Zisserman (2014): vijf VGG blocks en drie linear layers, in totaal 11 tot 19 layers met weights | |
        """),
        L.watch_out("Dieper is moeilijker te trainen. VGG gaf zijn diepere netwerken de eerste en laatste layers van een ondieper netwerk dat al geleerd had, omdat slechte willekeurige beginwaarden een diep netwerk kunnen laten vastlopen. De eigen beginwaarden van PyTorch voor `nn.Conv2d` zijn kleiner dan He initialization vraagt: de verschillen tussen afbeeldingen krimpen in elke layer, en na 16 convoluties geeft elke afbeelding dezelfde logits. Met de He initialization uit [Initialisatie](/course/open/nn-initialization?lang=nl) leert 8 per blok wel: 847 tot 858 goed over 3 seeds, en nog steeds niet meer dan met 1."),
        mo.md(r"""
        De figuur: de weights van $C$ naar $C$ channels, in eenheden van $C^2$, tegen hoe breed het stuk is dat ze zien. Eén convolutie van $k$ bij $k$ groeit met het kwadraat van $k$; een stapel van 3 bij 3 groeit met 9 voor elke 2 pixels meer.
        """),
        mo.Html(C.weights_picture({"title": "Weights om even ver te zien", "x": "breedte van het geziene stuk, in pixels", "y": "weights, in C²",
                           "one": "één convolutie van k bij k", "stack": "een stapel van 3 bij 3",
                           "alt": "Twee lijnen van weights in eenheden van C kwadraat tegen de breedte van het geziene stuk, 3 tot 11 pixels. Eén convolutie van k bij k, gestippeld, stijgt van 9 via 25 en 49 naar 121. Een stapel van 3 bij 3 stijgt van 9 via 18 en 27 naar 45. Bij 7 pixels: 49 tegen 27."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een VGG block met $n$ convoluties, van $C_\text{in}$ naar $C_\text{out}$ channels:

    1. $n$ convoluties van 3 bij 3 met padding 1, elk met een ReLU erna: de eerste van $C_\text{in}$ naar $C_\text{out}$ channels, de andere van $C_\text{out}$ naar $C_\text{out}$. De hoogte en breedte blijven.
    2. Max pooling van 2 bij 2 aan het eind: de hoogte en breedte halveren.
    3. Het volgende blok begint bij de $C_\text{out}$ channels van dit blok, en heeft er meestal twee keer zoveel.
    4. Na het laatste blok: één rij per afbeelding, en de output head.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 van het recept, op een afbeelding van 11 bij 11 die overal 0 is behalve een 1 in het midden, met een convolutie waarvan alle weights 1 zijn, drie keer gebruikt. Hoeveel getallen bereikt die 1? Het commentaar met `#` zegt welke stap elk deel is. De derde convolutie is nog niet gedaan: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_reach():
    import torch
    from torch import nn

    _x = torch.zeros(1, 1, 11, 11)
    _x[0, 0, 5, 5] = 1.0
    # 1. convoluties van 3 bij 3 met padding 1: hier alle weights 1 en de bias 0
    _conv = nn.Conv2d(1, 1, 3, padding=1)
    nn.init.ones_(_conv.weight)
    nn.init.zeros_(_conv.bias)
    _h = _conv(_x)
    print(_h.shape, (_h != 0).sum().item())
    _h = _conv(_h)
    print(_h.shape, (_h != 0).sum().item())
    _h = _h  # maak af: _conv(_h)
    print(_h.shape, (_h != 0).sum().item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De grootte blijft 11 bij 11 met padding 1. Na één convolutie komt de 1 in 9 getallen terecht, een stuk van 3 bij 3; na twee 25, een stuk van 5 bij 5. Zoals de cel nu is, zegt de derde regel weer 25. Afgemaakt zegt hij 49: 7 bij 7. Andersom bekeken hangt één getal na drie convoluties af van een stuk van 7 bij 7 van de afbeelding, net als bij één filter van 7 bij 7. Hier is dat één convolutie, drie keer gebruikt; drie eigen convoluties van 3 bij 3 hebben samen 27 weights, één van 7 bij 7 heeft er 49.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Drie blokken
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "vgg-shape",
        question="Een netwerk voor afbeeldingen van 28 bij 28 heeft drie VGG blocks, met 16, 32 en 64 channels. Hoeveel getallen per afbeelding gaan zijn output head in?",
        options={
            "keep": "64 × 7 × 7 = 3.136: de convoluties houden de grootte",
            "right": "64 × 3 × 3 = 576: drie keer halveren, de laatste keer naar beneden afgerond",
            "up": "64 × 4 × 4 = 1.024: 7 gehalveerd is 3,5, naar boven afgerond",
        },
        correct="right",
        why={
            "keep": "De convoluties houden hem, maar elk blok eindigt met pooling: drie keer halveren, niet twee.",
            "up": "Max pooling van 2 bij 2 laat een laatste rij en kolom vallen die geen blok vullen: 7 wordt 3, naar beneden afgerond.",
            "right": "28, 14, 7, en dan 3: de laatste pooling laat de zevende rij en kolom vallen. 64 channels van 3 bij 3 zijn 576.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_params():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kleine filters tegen één groot

        Schrijf `stack_vs_one(c, n)`: het geeft twee aantallen parameters, weights en bias terms samen. Eerst dat van een stapel van `n` convoluties van 3 bij 3 van `c` naar `c` channels; dan dat van één convolutie van `c` naar `c` channels die hetzelfde stuk ziet, $2n + 1$ breed. De check roept `stack_vs_one(64, 3)` aan, wat `(110784, 200768)` moet geven, en andere aantallen channels en convoluties.
        """),
        L.task("Laat `stack_vs_one` de parameters tellen van een stapel kleine convoluties en van één grote convolutie die even ver ziet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_params():
    box_params = L.code_box("stack-params", lines=8)
    box_params
    return (box_params,)


@app.cell(hide_code=True)
def check_params(box_params):
    result_params = L.check("stack-params", box_params.value)
    result_params
    return (result_params,)


@app.cell(hide_code=True)
def hints_params():
    L.hints(
        "stack-params",
        [
            "Een convolutie van `c` naar `c` channels met een filter van `k` bij `k` heeft `c` filters van `c` bij `k` bij `k`, en `c` bias terms.",
            "De stapel is `n` keer een convolutie met `k = 3`. De ene convolutie heeft `k = 2 * n + 1`.",
        ],
        why="De ene convolutie groeit met het kwadraat van zijn breedte, de stapel alleen met zijn lengte: voor 3 convoluties bij 64 channels 200.768 tegen 110.784, 81% meer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het VGG block

        Maak `VGGBlock(c_in, c_out, n)` af: in `__init__` zijn `n` convoluties, in een `nn.ModuleList` zoals in [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl), en zijn pooling; in `forward` stap 1 en 2 van het recept. De check maakt blokken na `torch.manual_seed(0)` en vergelijkt wat eruit komt, dus maak de convoluties in de volgorde waarin de data erdoor gaat, met de eigen beginwaarden van PyTorch; hij telt ook de parameters die PyTorch ziet. Als het klopt, bewaart de check het in je project, als `work/projects/cnn/vgg.py`.
        """),
        L.task("Laat `VGGBlock` `n` convoluties van 3 bij 3 draaien, elk met een ReLU, en dan max pooling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("vgg-block", lines=16)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("vgg-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "vgg-block",
        [
            "Vul in `__init__` een `nn.ModuleList` in een `for`-loop die `n` keer draait. Alleen de eerste convolutie krijgt `c_in` channels; elke volgende krijgt de `c_out` van de vorige.",
            "Loop in `forward` met een `for`-loop over `self.convs` en zet elke convolutie binnen `torch.relu`; na de loop de pooling.",
        ],
        why="In een `nn.ModuleList` vindt PyTorch de convoluties: `model.parameters()` geeft hun weights, en de Trainer traint ze. In een gewone Python-lijst zou de check 0 parameters tellen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je project nu kan

    De knop gebruikt je project. Met je `VGGBlock` bouwt hij netwerken van twee blokken, van 1 naar {C.WIDTHS[0]} channels en van {C.WIDTHS[0]} naar {C.WIDTHS[1]}, zoals je SmallCNN, met {", ".join(str(n) for n in C.DEPTHS[:-1])} en {C.DEPTHS[-1]} convoluties per blok. Met 1 per blok is het je SmallCNN, met dezelfde beginwaarden. Hij traint elk met je Trainer: {C.EPOCHS} epochs op de 6.000 trainingsafbeeldingen, een learning rate van {C.number(C.LR, 2, L.lang)} met een warmup van één epoch, elk één run, met seed 0, en telt de parameters met je `count_params`. Waar een module van jou ontbreekt of stuk is, springt die van de course in. Het werkt zodra oefening 3 af is, en duurt een à twee minuten.
    """)
    return


@app.cell(hide_code=True)
def depth_button_cell():
    depth_button = mo.ui.run_button(label="Train de vier netwerken")
    depth_button
    return (depth_button,)


@app.cell(hide_code=True)
def demo_depth(depth_button, result_block):
    result_block  # run again when exercise 3 saves your module
    if not depth_button.value:
        _out = mo.md("*Druk op de knop om de vier netwerken te trainen.*")
    else:
        _result = C.demo()
        if _result is None:
            _out = mo.md("*Maak eerst oefening 3 af: dan staat je `VGGBlock` in je project, en gebruikt deze cel hem.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "slow": "*Je project doet er hier te lang over: meer dan vijf minuten om één netwerk te trainen, op een computer die niet met ander werk bezig is. Doet je `VGGBlock` of je Trainer meer werk dan nodig is, zoals een loop die te vaak draait?*",
                "vgg": "*Je `VGGBlock` in `work/projects/cnn/vgg.py` stopt met een fout, bij het maken van de netwerken. Heb je zijn bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan weer.*",
                "trainer": "*Trainen met je Trainer in `work/projects/nn/trainer.py`, of tellen met je `count_params` in `work/projects/cnn/count.py`, stopt met een fout. Heb je een van die bestanden veranderd? Verwijder het, en check zijn oefening opnieuw.*",
            }[_result])
        else:
            _rows = "\n".join(f"| {n} | {2 * n + 1} | {C.grouped(p, L.lang)} | {right} |" for n, p, right in _result["rows"])
            _who = "" if not _result["stand_ins"] else "\n\nDe versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
            _out = mo.md(f"""| convoluties per blok | layers met weights | parameters | goed |
|---|---|---|---|
{_rows}

Over 5 seeds, met de code van de course: 864 tot 879 goed met 1 per blok, 859 tot 883 met 2, 832 tot 869 met 4, en 105 in elke run met 8: één soort voor elke afbeelding.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het receptive field uit [Convolutie in detail: kernel size, stride en padding](/course/open/cnn-conv-formal?lang=nl), nu als reden om kleine filters te stapelen: ze zien even ver als één groot filter, met minder weights en meer ReLU's. Je project `cnn` heeft nu `vgg.py`. Dieper was hier niet beter: op 6.000 kleine afbeeldingen van 28 bij 28 deden één of twee convoluties per blok het best, en met acht per blok liep het trainen vast. De volgende extra les, *ResNet: diepte die blijft werken*, geeft een diep CNN de residual connections van je diepe MLP uit [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_params, result_block):
    L.wrap_up(choice_shape, result_params, result_block)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
