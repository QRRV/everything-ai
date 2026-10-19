# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-mlp-digits. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-mlp-digits", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl) en [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_eval():
    quick_eval = L.choice(
        "quick-eval",
        question="Een netwerk met dropout gaat met `model.eval()` in eval mode, en voorspelt dan `model(X_test)`, zonder `torch.no_grad()`. Wat is het resultaat?",
        options={
            "random": "Voorspellingen die bij elke aanroep verschillen, omdat dropout nog aanstaat",
            "error": "Een fout: in eval mode vraagt PyTorch om `torch.no_grad()`",
            "right": "De juiste voorspellingen, en een graaf die niemand nodig heeft",
        },
        correct="right",
        why={
            "random": "`model.eval()` is de schakelaar voor dropout. Zie *Het idee* in [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl).",
            "error": "De twee schakelaars staan los van elkaar, en geen van beide heeft de ander nodig. Zie *Het idee* in [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl).",
            "right": "Eval mode zet dropout uit, dus de voorspellingen kloppen. Zonder `torch.no_grad()` neemt PyTorch er nog wel een graaf voor op, en dat kost geheugen en tijd.",
        },
    )
    quick_eval
    return (quick_eval,)


@app.cell(hide_code=True)
def quick_check_relu():
    quick_relu = L.choice(
        "quick-relu",
        question="Wat geeft ReLU voor de inputs −2, 0,5 en 3?",
        options={
            "right": "0, 0,5 en 3",
            "abs": "2, 0,5 en 3",
            "step": "0, 1 en 1",
        },
        correct="right",
        why={
            "abs": "ReLU maakt van een negatief getal geen positief getal. Zie [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).",
            "step": "Dat is een stap. ReLU laat een positief getal ongewijzigd door. Zie [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl).",
            "right": "ReLU geeft 0 voor een negatief getal en laat een positief getal door: max(0, z).",
        },
    )
    quick_relu
    return (quick_relu,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl) las je project 890 van de 1.000 testcijfers goed, met één gewogen som van de pixels per cijfer. De hellingen rekende je zelf uit. [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) liet zien dat een hidden layer een netwerk meer laat leren dan één gewogen som kan. En PyTorch geeft je nu alle onderdelen: een module, een optimizer, een DataLoader en eval mode. Vandaag zet je ze in elkaar tot een MLP dat dezelfde cijfers leest, en bewaar je het in je project. Dat is de laatste stap van dit deel van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           7, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Wie een handschrift leest, legt niet elk cijfer naast één vast voorbeeld, maar zoekt naar onderdelen, zoals een lusje bovenaan, een streep recht omlaag of een dwarsstreepje, en zet die samen: een lusje op een streep is een 9, een lusje op een lusje een 8.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share-right",
        question="Je lineaire classifier las 890 van de 1.000 testcijfers goed. Een MLP met een hidden layer van 64 neuronen traint op dezelfde 6.000 cijfers, 10 epochs lang. Hoeveel testcijfers denk je dat het goed leest?",
        options={
            "same": "Ongeveer 890: een hidden layer voegt weinig toe",
            "right": "Ongeveer 930",
            "all": "Ongeveer 995: bijna allemaal",
            "fewer": "Minder dan 890: met zoveel weights gokt hij",
        },
        reveal="**Ongeveer 930.** In de demo aan het eind van deze les leest het MLP 929 van de 1.000 testcijfers goed. Het leest er 71 fout, waar de lineaire classifier er 110 fout las: ongeveer een derde minder fouten. Niet allemaal goed: sommige cijfers zijn zo geschreven dat zelfs mensen twijfelen, en 6.000 voorbeelden zijn niet veel.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een MLP voor de cijfers heeft twee layers. De **hidden layer**, `nn.Linear(784, 64)` met ReLU, geeft elke afbeelding 64 getallen: elk hidden neuron heeft zijn eigen gewogen som van de 784 pixels, zijn eigen patroon, zoals in [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl). De **output layer**, `nn.Linear(64, 10)`, maakt van die 64 getallen de 10 logits, één logit per cijfer. Een cijfer heeft dus niet meer één vast patroon: zijn logit combineert er 64. Daar zijn meer getallen voor nodig: 784 · 64 + 64 = 50.240 voor de hidden layer en 64 · 10 + 10 = 650 voor de output layer, samen 50.890, tegen 7.850 voor je lineaire classifier.

        De loss is de cross-entropy uit [Cross-entropy-loss voor veel klassen](/course/open/ml-cross-entropy?lang=nl), als gemiddelde over de $n$ voorbeelden van een mini-batch:
        """),
        L.formula(r"[[loss:\mathcal{L}]] = -\frac{1}{n} \sum_{i=1}^{n} \log([[pred:\hat{y}_{i,c_i}]])"),
        mo.md(r"""
        Hier is $c_i$ het juiste cijfer van voorbeeld $i$, en $\hat{y}_{i,c_i}$ de kans die softmax aan dat cijfer geeft. In PyTorch is dat **`nn.CrossEntropyLoss()`**. Die krijgt de logits, shape $(n, 10)$, en de labels als cijfers, shape $(n,)$, als int64, en doet de softmax zelf. Het model geeft dus logits terug, zonder softmax. Het voorspelde cijfer is dat met de grootste logit, `logits.argmax(dim=1)`, zoals `argmax(axis=1)` in NumPy: softmax verandert niet welke logit de grootste is.

        | | Je lineaire classifier | Je MLP in PyTorch |
        |---|---|---|
        | het model | `X @ W.T + b` | `DigitsMLP`: `nn.Linear(784, 64)`, ReLU, `nn.Linear(64, 10)` |
        | getallen die het leert | 7.850 | 50.890 |
        | de loss | cross-entropy, uitgeschreven met NumPy | `nn.CrossEntropyLoss()` op de logits |
        | de hellingen | je eigen formule, `diff.T @ X / n` | `loss.backward()` |
        | de stap | `W = W - lr * grad_W` | `opt.step()` |
        | de data per stap | alle 6.000 voorbeelden | een mini-batch van 64 uit een DataLoader |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | `nn.CrossEntropyLoss()` | n n cross-entropy loss | de cross-entropy als PyTorch-module: softmax en het gemiddelde van min de log van de juiste kans, in één | `loss_fn = nn.CrossEntropyLoss()` |
        | $c_i$ | c i | de juiste klasse van voorbeeld $i$: hier zijn cijfer | `y[i]` |
        | $\hat{y}_{i,c_i}$ | y-dakje i c i | de kans die het model aan het juiste cijfer van voorbeeld $i$ geeft | |
        """),
        L.watch_out("`nn.CrossEntropyLoss` doet de softmax zelf. Bij een model dat eindigt met softmax gebeurt dat dus twee keer: de loss daalt nog wel, maar langzaam, en het model leert minder. Geef de loss de logits, en de labels als gehele getallen, int64, niet als floats."),
        mo.md(r"""
        De figuur: een mini-batch van $n$ afbeeldingen door het MLP, met de shape van wat elke layer geeft.
        """),
        draw.flow([("Mini-batch", ["n afbeeldingen", "van 784 pixels", "(n, 784)"]), ("Hidden layer", ["64 neuronen", "met ReLU", "(n, 64)"]),
                   ("Output layer", ["10 logits", "per afbeelding", "(n, 10)"])],
                  ["nn.Linear(784, 64)", "nn.Linear(64, 10)"], ["", ""], title="Een mini-batch van n afbeeldingen door DigitsMLP"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om in PyTorch een classifier voor de cijfers te trainen:

    1. Laad de cijfers als tensors: de pixels als float32 gedeeld door 255, de labels als int64. Zet de trainingscijfers in een DataLoader die shufflet.
    2. Maak het model: het geeft 10 logits per afbeelding terug, zonder softmax.
    3. Maak de loss en de optimizer: `loss_fn = nn.CrossEntropyLoss()` en `opt = torch.optim.SGD(model.parameters(), lr=...)`.
    4. Train een paar epochs. Voor elke mini-batch: `loss = loss_fn(model(xb), yb)`, dan `opt.zero_grad()`, `loss.backward()` en `opt.step()`.
    5. Meet op de testcijfers, in eval mode en zonder graaf: de voorspellingen zijn `model(X_test).argmax(dim=1)`, en de accuracy is het deel dat klopt, `(pred == y_test).float().mean()`: `.float()` maakt van True en False 1 en 0, omdat PyTorch alleen een gemiddelde van floats neemt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eerst het recept met het model dat je kent: je lineaire classifier, nu als één layer `nn.Linear(784, 10)` die 10 logits per afbeelding geeft. De cel traint hem 3 epochs met mini-batches van 64, met $\eta$ = 0,5, en print de loss over alle trainingscijfers vóór het trainen en na elke epoch. Daarna meet hij het model op de 1.000 testcijfers. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit; dat duurt een seconde.
    """)
    return


@app.cell
def example_linear():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. de cijfers als tensors, en een DataLoader die shufflet
    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(data["test_labels"], dtype=torch.int64)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    # 2. het model: één layer van 784 pixels naar 10 logits
    torch.manual_seed(0)
    model = nn.Linear(784, 10)

    # 3. de loss en de optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    print(loss_fn(model(X), y).item())

    # 4. drie epochs met mini-batches
    torch.manual_seed(2)
    for epoch in range(3):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(epoch, loss_fn(model(X), y).item())

    # 5. de accuracy op de testcijfers, in eval mode en zonder graaf
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
    model.train()
    print(pred[:10])
    print(y_test[:10])
    print((pred == y_test).float().mean().item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vóór het trainen is de loss ongeveer 2,35, dicht bij 2,30: de loss van een kans van 0,1 voor het juiste cijfer, omdat kleine willekeurige weights tien logits geven die dicht bij elkaar liggen. Na 3 epochs is hij ongeveer 0,28. Van de eerste tien testcijfers leest het model er negen goed: het negende is een 5, en het model zegt 6. Van alle 1.000 leest het er 887 goed, een accuracy van 0,887: ongeveer wat je eigen lineaire classifier las, 890 na 100 stappen over alle data. Het is hetzelfde soort model; wat het anders maakt, is een hidden layer: die komt in oefening 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_input():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat de loss krijgt
    """)
    return


@app.cell(hide_code=True)
def choice_input_cell():
    choice_input = L.choice(
        "loss-input",
        question="`loss_fn = nn.CrossEntropyLoss()`, en `xb` is een mini-batch van 64 afbeeldingen. Wat geef je `loss_fn`?",
        options={
            "probs": "De kansen na softmax, shape (64, 10), en de juiste cijfers, shape (64,)",
            "right": "De logits, shape (64, 10), en de juiste cijfers, shape (64,)",
            "argmax": "De voorspelde cijfers en de juiste cijfers, allebei met shape (64,)",
        },
        correct="right",
        why={
            "probs": "`nn.CrossEntropyLoss` doet de softmax zelf; met kansen zou hij dat twee keer doen. Zie *Het idee*.",
            "argmax": "Een voorspeld cijfer is een geheel getal: een kleine verandering in een weight verandert het bijna nooit, dus het geeft geen hellingen om van te leren. Zie *Het idee*.",
            "right": "De logits van het model, één rij per afbeelding, en het juiste cijfer van elke afbeelding als geheel getal: `nn.CrossEntropyLoss` doet de softmax en neemt min de log van de juiste kans.",
        },
    )
    choice_input
    return (choice_input,)


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je MLP als module

        `DigitsMLP(hidden)` is een MLP voor de cijfers met `hidden` neuronen in zijn hidden layer. `__init__` maakt de hidden layer al: `hidden` neuronen, elk met 784 weights. Vervang de twee `...`: de output layer, van `hidden` getallen naar 10 logits, en in `forward` de mini-batch `x` door de hidden layer, ReLU en de output layer. De check maakt `DigitsMLP(64)` en `DigitsMLP(16)`, elk na `torch.manual_seed(0)`, geeft ze drie testcijfers en vergelijkt de logits met die van het model van de course. Klopt je class, dan bewaart de check hem in je project, als `work/projects/nn/torch_mlp.py`.

        Dat bestand begint met `from torch import nn`: die `nn` is die van PyTorch, ook al heet je project ook `nn`. Een bestand uit je project importeer je met zijn volledige naam, zoals `from nn.torch_mlp import DigitsMLP`.
        """),
        L.task("Geef `DigitsMLP` zijn output layer, en een `forward` die 10 logits per afbeelding teruggeeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("digits-mlp", lines=14)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("digits-mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "digits-mlp",
        [
            "Stap 2 van het recept: de twee layers uit de figuur, met ReLU ertussen.",
            "De output layer krijgt de `hidden` getallen van de hidden layer en geeft er 10. `forward` werkt van binnen naar buiten: eerst de hidden layer op `x`, dan ReLU, dan de output layer.",
            "`self.out = nn.Linear(___, 10)`, en in `forward`: `return self.out(torch.relu(self.___(x)))`.",
        ],
        why="`self.out = nn.Linear(hidden, 10)` maakt van de `hidden` getallen van elke afbeelding 10 logits. `self.hidden(x)` geeft elke afbeelding `hidden` gewogen sommen, `torch.relu` zet de negatieve op 0, en `self.out` maakt de logits. Geen softmax: dat doet `nn.CrossEntropyLoss`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Trainen

        `fit(model, loader, epochs, lr)` traint een model op de mini-batches van een DataLoader, met `nn.CrossEntropyLoss` en SGD, en geeft de gemiddelde loss van elke epoch terug. De loss, de optimizer, de loops en de som van de losses staan er al. Vervang de `...` door stap 4 van het recept voor één mini-batch, en tel daarna de loss van de mini-batch als getal op bij `total`: de regel na de loop deelt die som door het aantal mini-batches. De check traint een kleine `nn.Linear(2, 3)` op vier voorbeelden, met `fit(model, loader, 3, 0.5)`, en ook met een andere batch size, een ander aantal epochs en een andere learning rate. Klopt je functie, dan bewaart de check hem in je project, als `work/projects/nn/torch_train.py`.
        """),
        L.task("Laat `fit` voor elke mini-batch een update-stap zetten, met de cross-entropy van zijn logits, en tel de losses van de epoch op."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-torch", lines=16)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit-torch", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-torch",
        [
            "Binnen de loop over de mini-batches: stap 4 van het recept, en daarna de loss bijhouden voor het gemiddelde van de epoch.",
            "Eerst de loss van deze mini-batch, uit zijn logits en zijn labels. Dan de drie regels van de optimizer: de hellingen wissen, ze uitrekenen, de stap zetten.",
            "`loss = loss_fn(model(___), ___)`, dan `opt.zero_grad()`, `loss.___()` en `opt.step()`, en als laatste `total = total + loss.___()`.",
        ],
        why="`loss_fn(model(xb), yb)` is de cross-entropy van deze mini-batch. `opt.zero_grad()` wist de hellingen van de vorige stap, `loss.backward()` rekent de nieuwe uit, en `opt.step()` beweegt elke parameter tegen zijn helling in. `total = total + loss.item()` telt de loss van de mini-batch als getal op, voor het gemiddelde van de epoch. Op de vier voorbeelden gaat de gemiddelde loss van ongeveer 1,11 in de eerste epoch naar 0,89 in de derde.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder maakt je eigen `DigitsMLP(64)`, traint het met je eigen `fit` op de 6.000 trainingscijfers, 10 epochs met mini-batches van 64 en $\eta$ = 0,5, en meet het op de 1.000 testcijfers, die het nooit zag. Dat duurt een seconde of twee. De cel gebruikt je eigen modules, dus hij werkt zodra oefening 2 en 3 af zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_digits(result_fit, result_mlp):
    result_mlp, result_fit  # run again when exercise 2 or 3 saves your module
    _result = C.trained()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `DigitsMLP` en je `fit` in je project, en gebruikt deze cel ze.*")
    elif _result == "slow":
        _out = mo.md("*Trainen met je `fit` in `work/projects/nn/torch_train.py` duurt op de 6.000 cijfers langer dan een minuut. Gaat hij in elke epoch één keer langs de mini-batches van `loader`?*")
    elif _result == "mlp":
        _out = mo.md("*De `DigitsMLP` in `work/projects/nn/torch_mlp.py` geeft niet meer de logits die de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en check oefening 2 opnieuw: de check bewaart je class dan opnieuw.*")
    elif _result == "fit":
        _out = mo.md("*De `fit` in `work/projects/nn/torch_train.py` traint niet meer zoals de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je functie dan opnieuw.*")
    else:
        _accuracy, _losses = _result
        _digit, _baseline = C.baseline()
        _right = round(_accuracy * 1000)
        _out = mo.vstack([
            mo.md(f"""Je MLP leest <span class="ea-num">{L.number(_right, 0)}</span> van de 1.000 testcijfers goed: een accuracy van <span class="ea-num">{L.number(_accuracy, 3)}</span>. Je lineaire classifier las er {L.number(C.LINEAR, 0)}, en altijd het vaakste cijfer zeggen, een {_digit}, haalt <span class="ea-num">{L.number(_baseline, 3)}</span>. Je MLP leest er dus {L.number(1000 - _right, 0)} fout, waar de lineaire classifier er {L.number(1000 - C.LINEAR, 0)} fout las."""),
            mo.Html(C.losses_picture(_losses, {"title": "De loss van je training, per epoch", "epoch": "epoch", "loss": "gemiddelde loss",
                                               "alt": f"Een lijn door 10 punten, één per epoch: de gemiddelde loss van de mini-batches daalt van {L.number(_losses[0], 2)} in de eerste epoch naar {L.number(_losses[-1], 2)} in de tiende, eerst snel en dan langzamer."})),
            mo.md(f"""Elk punt is de gemiddelde loss van de 94 mini-batches van één epoch, zoals je `fit` hem teruggeeft: van <span class="c-loss ea-num">{L.number(_losses[0], 2)}</span> in de eerste epoch naar <span class="c-loss ea-num">{L.number(_losses[-1], 2)}</span> in de tiende. Op de trainingscijfers blijft de loss dalen. Of dat ook geldt voor cijfers die het model nooit zag, ontdek je in *Loss curves lezen*, twee lessen verder."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De onderdelen uit de PyTorch-lessen passen hier in elkaar: een module zoals in [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl), een optimizer, een DataLoader die shufflet, en eval mode uit [Train mode en eval mode](/course/open/torch-eval-mode?lang=nl) om te meten. Met de hidden layer uit [Een hidden layer en ReLU](/course/open/nn-hidden-layer?lang=nl) maakt je MLP ongeveer een derde minder fouten dan de lineaire classifier uit [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl). Je project `nn` heeft nu ook `torch_mlp.py` en `torch_train.py`. In de volgende les, *Waarom AI GPU's gebruikt, en dezelfde code op de jouwe*, tel je het werk dat je MLP doet, en laat je dezelfde training loop een GPU gebruiken als die er is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_input, result_fit, result_mlp):
    L.wrap_up(choice_input, result_mlp, result_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
