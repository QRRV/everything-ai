# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-gpu. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-gpu", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) en [Tensors in PyTorch](/course/open/torch-tensors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="Hoeveel weights heeft `nn.Linear(784, 64)`, zonder zijn bias terms?",
        options={
            "right": "50.176: 784 keer 64",
            "sum": "848: 784 + 64",
            "in": "784: één per pixel",
        },
        correct="right",
        why={
            "sum": "Elk van de 64 neuronen heeft zijn eigen weight voor elk van de 784 inputs. Zie *Het idee* in [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "in": "Dat zijn de weights van één neuron; de layer heeft er 64. Zie *Het idee* in [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "right": "784 · 64 = 50.176: één weight voor elk paar van een input en een neuron. Met de 64 bias terms heeft de layer 50.240 parameters.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="`t = torch.tensor([1.0, 2.0])`. Op welk device staat `t`?",
        options={
            "gpu": "Op de GPU, als de computer er een heeft",
            "none": "Nog op geen enkel: pas een berekening zet hem op een device",
            "right": "Op de CPU, tot je hem verplaatst",
        },
        correct="right",
        why={
            "gpu": "PyTorch maakt een nieuwe tensor op de CPU, ook als er een GPU is, en verplaatst hem nooit vanzelf. Zie *Waar een tensor staat* in [Tensors in PyTorch](/course/open/torch-tensors?lang=nl).",
            "none": "Elke tensor staat ergens vanaf het moment dat hij gemaakt wordt. Zie *Waar een tensor staat* in [Tensors in PyTorch](/course/open/torch-tensors?lang=nl).",
            "right": "Een nieuwe tensor wordt op de CPU gemaakt, tenzij je iets anders zegt; `t.device` zegt `cpu`.",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je MLP voor de cijfers trainde in een seconde of twee. De modellen achter chatbots en beeldgeneratoren hebben van ongeveer een miljard tot een biljoen parameters, tegen 50.890 in je MLP, en trainen op veel meer data. Ze worden getraind op GPU's, chips die voor graphics gemaakt zijn, omdat het werk van een neural network dezelfde eenvoudige berekening is, ontzettend vaak herhaald. Om te weten wat een model kost, tel je dat werk. En om vanuit PyTorch een GPU te gebruiken, heb je maar een paar regels nodig, die ook draaien op een computer zonder GPU.

    **Een voorbeeld uit het dagelijks leven.** Een restaurant heeft een paar ervaren koks, die elk gerecht kunnen maken. Voor een bruiloft moet de keuken ook een berg groente snijden: dan zijn honderd hulpjes die alleen kunnen snijden, maar dat allemaal tegelijk doen, veel eerder klaar dan de koks.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="Op een computer met een GPU train je het kleine `DigitsMLP(64)` uit de projectles op mini-batches van 64 cijfers, één keer op de CPU en één keer op de GPU. Wat denk je dat er gebeurt?",
        options={
            "much": "De GPU is veel sneller: die heeft duizenden cores",
            "right": "Ongeveer even snel, en soms zelfs langzamer dan de CPU",
            "never": "Het draait niet: een GPU neemt alleen grote netwerken",
        },
        reveal="**Ongeveer even snel, soms zelfs langzamer.** Een mini-batch van 64 cijfers kost dit netwerk ongeveer 20 miljoen FLOPs, te weinig om duizenden cores bezig te houden. En elke mini-batch moet eerst naar de GPU, en elke `.item()` weer terug. Een GPU loont als de matrixvermenigvuldigingen groot zijn: brede layers, grote mini-batches, grote modellen. Dan is hij vele keren sneller.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het werk tellen.** Zoals in [Te veel weights om één voor één op te schuiven](/course/open/nn-why-backprop?lang=nl) kost elke weight in een forward pass 2 FLOPs per voorbeeld: één vermenigvuldiging en één optelling. `nn.Linear(n_in, n_out)` heeft `n_in * n_out` weights, dus `DigitsMLP(64)`, met 784 · 64 + 64 · 10 = 50.816 weights, kost 101.632 FLOPs om één afbeelding te lezen. Een trainingsstap kost ongeveer drie keer zoveel, 6 FLOPs per weight per voorbeeld, omdat de backward pass elke weight twee keer gebruikt: één keer om zijn helling uit te rekenen en één keer om de hellingen door te geven aan de layer ervoor. Een epoch van 6.000 afbeeldingen kost dan ongeveer 6 · 6.000 · 50.816 = 1.829.376.000 FLOPs, bijna 2 miljard.

        **Waarom een GPU.** Die FLOPs staan los van elkaar: elk getal dat een layer teruggeeft, kun je apart uitrekenen. Een CPU heeft een paar snelle cores die alles kunnen, van 4 tot ongeveer 24 in een laptop. Een **GPU** heeft duizenden eenvoudige cores die allemaal dezelfde berekening doen, elk op andere getallen, tegelijk. Voor de grote matrixvermenigvuldigingen van een groot netwerk is dat vele keren sneller. Voor een klein netwerk kan het zelfs langzamer zijn, omdat het verplaatsen van de getallen naar de GPU ook tijd kost.

        **Dezelfde code op elke computer.** Zoals in [Tensors in PyTorch](/course/open/torch-tensors?lang=nl) moeten de tensors van één berekening op hetzelfde device staan, en PyTorch verplaatst ze nooit vanzelf. Dus:

        - kies het device één keer, met een `if`: `"cuda"`, een GPU van NVIDIA, als `torch.cuda.is_available()` `True` is, en anders `"cpu"`; op een Mac met een Apple-chip is de test `torch.backends.mps.is_available()` en het device `"mps"`;
        - verplaats het model, één keer, voordat je de optimizer maakt: `model.to(device)` verplaatst al zijn parameters;
        - verplaats elke mini-batch: `xb.to(device)` en `yb.to(device)`;
        - haal een resultaat terug als getal van Python met `.item()`.

        Op een computer zonder GPU is `device` `"cpu"`, en draait de code zoals altijd.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **GPU** | G P U | een chip met duizenden eenvoudige cores die dezelfde berekening op veel getallen tegelijk doen | `"cuda"`, `"mps"` |
        | `.to(device)` | to device | verplaatst een model of een tensor naar een device; voor een tensor geeft het een verplaatste kopie terug | `model.to(device)`, `xb.to(device)` |
        | `torch.cuda.is_available()` | | `True` als PyTorch een GPU van NVIDIA kan gebruiken | `if torch.cuda.is_available():` |
        """),
        L.watch_out("Op een computer zonder GPU geeft een vergeten `.to(device)` geen fout: alles staat toch al op de CPU. De fout komt pas op een computer met een GPU, misschien die van iemand anders. Daarom probeert de check van oefening 2 je code ook op het `\"meta\"`-device van PyTorch, een vervanger van een GPU die alleen de shapes van tensors kent, niet hun getallen, en net als een GPU stopt als een tensor op een ander device staat."),
        mo.md(r"""
        De figuur: wat waar staat terwijl een netwerk op een GPU traint.
        """),
        draw.flow([("CPU", ["je programma", "de DataLoader", "gewone getallen"]), ("GPU", ["het model", "elke mini-batch", "de loss"])],
                  [".to(device)"], [".item()"], title="Trainen op een GPU"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een training loop een GPU te laten gebruiken als die er is:

    1. Kies het device één keer, met een `if` en een `else`: `device = "cuda"` als `torch.cuda.is_available()`, anders `device = "cpu"`.
    2. Verplaats het model met `model.to(device)`, en maak pas daarna de optimizer.
    3. Verplaats in de loop elke mini-batch: `xb, yb = xb.to(device), yb.to(device)`.
    4. Haal resultaten terug als getal met `.item()`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het MLP uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), één epoch getraind met het recept. `weights` telt de weights van zijn twee `nn.Linear`-layers. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_device():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    # 1. kies het device één keer
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    torch.manual_seed(0)
    model = DigitsMLP(64)
    weights = 784 * 64 + 64 * 10
    print(2 * weights, 6 * 6000 * weights)
    # 2. verplaats het model, en maak dan de optimizer
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    print(model.out.weight.device)

    torch.manual_seed(2)
    total = 0
    for xb, yb in loader:
        # 3. verplaats elke mini-batch
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 4. terug als getal van Python
        total = total + loss.item()
    print(total / len(loader))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op een computer zonder GPU print de cel `cpu`, en staat ook `model.out.weight` op de CPU: er is niets verplaatst, en dat hoefde ook niet. Op een Mac met een Apple-chip print hij ook `cpu`, omdat deze code alleen naar een GPU van NVIDIA kijkt. Op een computer met een GPU van NVIDIA print hij `cuda`, en staan de weights op `cuda:0`, de eerste GPU. Hoe dan ook is de gemiddelde loss van de epoch ongeveer 0,76, het eerste punt in de demo van de projectles. De forward pass van één afbeelding kost 101.632 FLOPs, en een epoch trainen ongeveer 1,8 miljard: minder dan een seconde werk voor een CPU. Een netwerk met een miljard parameters heeft 2 miljard FLOPs nodig voor één enkele input.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nogpu():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Geen GPU
    """)
    return


@app.cell(hide_code=True)
def choice_nogpu_cell():
    choice_nogpu = L.choice(
        "no-gpu",
        question="Je laptop heeft geen GPU. Je draait de training loop uit het uitgewerkte voorbeeld. Wat gebeurt er?",
        options={
            "right": "Hij draait op de CPU, zoals eerst",
            "error": "Hij stopt met een fout: er is geen CUDA",
            "slower": "Hij draait, maar langzamer dan dezelfde loop zonder de regels met device",
        },
        correct="right",
        why={
            "error": "`torch.cuda.is_available()` is dan `False`, dus wordt `device` `\"cpu\"`. Zie *Dezelfde code op elke computer* in *Het idee*.",
            "slower": "Verplaatsen naar het device waar je al staat, kost niets. Zie *Dezelfde code op elke computer* in *Het idee*.",
            "right": "`device` wordt `\"cpu\"`, en `.to(\"cpu\")` laat model en data waar ze zijn: de code draait zoals eerst.",
        },
    )
    choice_nogpu
    return (choice_nogpu,)


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Voorspellen op een device

        `predict(model, X, device)` geeft het voorspelde cijfer voor elke afbeelding in `X`, uitgerekend op `device`. Vervang de twee `...`: verplaats eerst het model naar `device`; geef het daarna `X`, ook verplaatst naar `device`. De check maakt je `DigitsMLP(16)` en roept `predict` aan op vijf testcijfers, op de CPU en op het `"meta"`-device uit *Let op*. Op `"meta"` laat hij zien waar de voorspellingen staan, welke shape ze hebben en waar het model staat.
        """),
        L.course_versions(),
        L.task("Verplaats het model en de afbeeldingen naar `device` voordat het model voorspelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict-device", lines=12)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict-device", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict-device",
        [
            "Stap 2 en 3 van het recept, voor een voorspelling in plaats van een trainingsstap.",
            "Het model verhuist met één regel voordat het voorspelt. De afbeeldingen verhuizen in de regel waarin het model ze krijgt.",
            "`model.___(device)` op de eerste `...`, en `model(X.___(device))` op de tweede.",
        ],
        why="`model.to(device)` verplaatst alle parameters van het model; `X.to(device)` geeft een kopie van de afbeeldingen op dat device. Dan staan het model en zijn input op hetzelfde device, en de voorspellingen ook. Op `\"meta\"` zijn er geen getallen, alleen de shape: vijf voorspellingen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_moves():
    mo.md(r"""
    ### Oefening 3. Wat verhuist?

    Vijf dingen uit een training loop op een GPU. Zet elk in zijn groep: gaat naar het device, of hoeft niet te verhuizen.
    """)
    return


@app.cell(hide_code=True)
def exercise_moves():
    sort_moves = L.sort("what-moves")
    sort_moves
    return (sort_moves,)


@app.cell(hide_code=True)
def check_moves(sort_moves):
    result_moves = L.check("what-moves", sort_moves)
    result_moves
    return (result_moves,)


@app.cell(hide_code=True)
def hints_moves():
    L.hints(
        "what-moves",
        [
            "Vraag bij elk ding: is het, of bevat het, een tensor waarmee de GPU rekent?",
            "Twee dingen verhuizen, drie blijven. Gewone getallen van Python hoeven nooit te verhuizen.",
            "Het model en elke mini-batch verhuizen. Is een learning rate een tensor? En wat geeft `.item()` terug?",
        ],
        why="**Gaat naar het device:** het model, met zijn parameters, en elke mini-batch. **Hoeft niet te verhuizen:** de learning rate en het aantal epochs, gewone getallen van Python, en `loss.item()`, dat al een getal van Python is, terug op de CPU.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De devices uit [Tensors in PyTorch](/course/open/torch-tensors?lang=nl) draaien nu een hele training loop: het model gaat één keer naar het device, elke mini-batch elke keer. FLOPs tellen, bij het trainen ongeveer 6 per weight per voorbeeld, zegt wat de netwerken uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) en veel grotere kosten. In de volgende les, *Loss curves lezen*, kijk je wat het trainen epoch na epoch doet, op de training data en op data die het model nooit zag.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_nogpu, result_moves, result_predict):
    L.wrap_up(choice_nogpu, result_predict, result_moves)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
