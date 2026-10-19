# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-checkpoints. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("torch-checkpoints", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_best():
    quick_best = L.choice(
        "quick-best",
        question="Tijdens het trainen is de validation loss na elke epoch 0,90, 0,61, 0,50, 0,52, 0,55 en 0,60, terwijl de training loss blijft dalen. Na welke epoch was het model het best op nieuwe data?",
        options={
            "last": "Epoch 6: daar is de training loss het laagst",
            "right": "Epoch 3",
            "first": "Epoch 1: daarna leert het alleen nog uit zijn hoofd",
        },
        correct="right",
        why={
            "last": "De training loss zegt hoe goed het model past bij de voorbeelden waarvan het leert, niet bij nieuwe. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "first": "De validation loss daalt na epoch 1 nog, tot 0,50. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "De validation loss is het laagst na epoch 3, op 0,50, en stijgt daarna: vanaf daar overfit het model.",
        },
    )
    quick_best
    return (quick_best,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="Wat geeft je `fit(model, loader, epochs, lr)` uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) terug?",
        options={
            "model": "Een nieuw, getraind model",
            "nothing": "Niets: hij traint alleen",
            "right": "De gemiddelde loss van elke epoch",
        },
        correct="right",
        why={
            "model": "`fit` traint het model dat hij krijgt: `model` zelf verandert. Zie oefening 3 van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "nothing": "Kijk naar zijn laatste regel, `return losses`. Zie oefening 3 van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl).",
            "right": "Hij traint `model` zelf, en geeft een lijst terug met de gemiddelde loss van de mini-batches van elke epoch.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een getraind model leeft in het geheugen van het programma dat het trainde. Stop het programma, en het model is weg: de demo van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) traint je MLP elke keer opnieuw. En [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) liet zien dat het beste model vaak niet het laatste is, maar dat van een epoch halverwege. Allebei vragen ze om hetzelfde: de parameters van een model, bewaard in een bestand, om later weer te laden.

    **Een voorbeeld uit het dagelijks leven.** Bij een lang computerspel kun je opslaan. Je stopt, en morgen ga je verder waar je was, met alles wat je had. En voor een moeilijk stuk sla je ook op: gaat het mis, dan laad je het opgeslagen spel en ben je terug waar het nog goed ging.
    """)
    return


@app.cell(hide_code=True)
def predict_live():
    guess_live = L.guess(
        "live",
        question="Een MLP traint 30 epochs. Na epoch 8, de epoch met de laagste validation loss, schrijf je `best = model.state_dict()`, en het trainen gaat door tot epoch 30. Daarna krijgt een nieuw model `best` met `load_state_dict`. Welk model denk je dat je krijgt?",
        options={
            "eight": "Het model van epoch 8",
            "right": "Het model van epoch 30",
            "error": "Een fout: `best` is weg zodra het trainen doorgaat",
        },
        reveal="**Het model van epoch 30.** `model.state_dict()` kopieert niets: zijn dictionary bevat de eigen tensors van het model, en elke stap na epoch 8 verandert die. In de run van het uitgewerkte voorbeeld van [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) heeft het nieuwe model een validation loss van 0,48, die van epoch 30, niet de 0,41 van epoch 8. Een bestand dat je na epoch 8 met `torch.save` schrijft, houdt epoch 8 wel vast: dat doe je in oefening 2.",
    )
    guess_live
    return (guess_live,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De state_dict.** `model.state_dict()` is een dictionary met elke parameter van het model per naam. De namen komen van de attributes in `__init__`: `self.hidden` geeft `hidden.weight` en `hidden.bias`, `self.out` geeft `out.weight` en `out.bias`. Elke waarde is de tensor zelf.

        **Opslaan en laden.** `torch.save(obj, path)` schrijft tensors, en dictionaries en lijsten daarvan, naar een bestand; zo'n bestand eindigt meestal op `.pt`. `torch.load(path)` leest het terug. `model.load_state_dict(state)` kopieert de tensors van een state_dict naar een model. Dat model moet van dezelfde class zijn, met dezelfde grootte: elke naam moet erin staan, met dezelfde shape, anders stopt PyTorch met een fout. Het bestand bevat alleen namen en getallen; de code van de class blijft in je eigen bestanden, zoals `nn/torch_mlp.py` in je project.

        **Een checkpoint** is wat je opslaat tijdens het trainen: minstens de state_dict van het model. Wil je er later verder mee trainen, dan bevat het ook de epoch en de eigen state_dict van de optimizer, `opt.state_dict()`, alles in één dictionary. Bij gewone SGD staan in die state_dict alleen zijn instellingen, zoals de learning rate; optimizers die je later tegenkomt, bewaren er ook getallen per parameter in.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **state_dict** | state dict | de dictionary van PyTorch met elke parameter van een model per naam | `model.state_dict()` |
        | `torch.save(obj, path)` | torch save | schrijft tensors, en dictionaries en lijsten daarvan, naar een bestand | `torch.save(model.state_dict(), path)` |
        | `torch.load(path)` | torch load | leest zo'n bestand terug | `torch.load(path)` |
        | `load_state_dict(state)` | load state dict | kopieert de tensors van een state_dict naar een model van dezelfde class en grootte | `model.load_state_dict(state)` |
        | **checkpoint** | | een opgeslagen kopie van de parameters op één moment van het trainen, vaak met de epoch en de optimizer | `torch.save({"epoch": 10, ...}, path)` |
        """),
        L.watch_out("`model.state_dict()` kopieert niets: de tensors erin zijn die van het model zelf, en het trainen blijft ze veranderen. Om de parameters van één moment te bewaren, schrijf je ze met `torch.save` naar een bestand. En `torch.load` leest alleen tensors en eenvoudige waarden terug, zoals getallen en dictionaries. Dat is met opzet: een bestand van iemand anders zou anders bij het laden code op je computer kunnen uitvoeren."),
        mo.md(r"""
        De figuur: de state_dict van `DigitsMLP(64)`, de naam van elke parameter met de shape van zijn tensor.
        """),
        draw.dict_boxes(C.state_pairs(), key_label="naam", value_label="tensor", title="model.state_dict() van DigitsMLP(64)"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een model in een bestand te bewaren en terug te krijgen:

    1. Sla zijn parameters op: `torch.save(model.state_dict(), path)`.
    2. Laad ze in een nieuw model van dezelfde class en grootte: `model = DigitsMLP(64)`, dan `model.load_state_dict(torch.load(path))`. Zet het in eval mode om te voorspellen.
    3. Om later verder te trainen, sla je een checkpoint op: een dictionary met de epoch, `model.state_dict()` en `opt.state_dict()`. Laad elk deel in een nieuw model en een nieuwe optimizer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het MLP uit de demo van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl): 64 hidden neuronen, 10 epochs op de 6.000 trainingscijfers met $\eta$ = 0,5. De cel slaat het op, laadt het in een nieuw model, meet dat op de 1.000 testcijfers, en slaat daarna een checkpoint op en laadt het. `C.checkpoint_path("digits.pt")` geeft het pad van een bestand in je werkmap, `work/checkpoints/digits.pt`. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit; dat duurt een seconde of twee.
    """)
    return


@app.cell
def example_save():
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
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(data["test_labels"], dtype=torch.int64)
    torch.manual_seed(0)
    model = DigitsMLP(64)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    torch.manual_seed(2)
    for epoch in range(10):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 1. sla de parameters op
    state = model.state_dict()
    for name in state:
        print(name, state[name].shape)
    path = C.checkpoint_path("digits.pt")
    torch.save(model.state_dict(), path)

    # 2. laad ze in een nieuw model van dezelfde class en grootte
    loaded = DigitsMLP(64)
    print(loaded.load_state_dict(torch.load(path)))
    loaded.eval()
    with torch.no_grad():
        pred = loaded(X_test).argmax(dim=1)
    print((pred == y_test).float().mean().item())

    # 3. een checkpoint om later verder te trainen: de epoch, het model en de optimizer
    torch.save({"epoch": 10, "model": model.state_dict(), "opt": opt.state_dict()}, C.checkpoint_path("digits-epoch10.pt"))
    checkpoint = torch.load(C.checkpoint_path("digits-epoch10.pt"))
    again = DigitsMLP(64)
    again.load_state_dict(checkpoint["model"])
    opt_again = torch.optim.SGD(again.parameters(), lr=0.5)
    opt_again.load_state_dict(checkpoint["opt"])
    print(checkpoint["epoch"], checkpoint["opt"]["param_groups"][0]["lr"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De state_dict heeft vier tensors, met de shapes uit de figuur: 784 · 64 + 64 + 64 · 10 + 10 = 50.890 getallen. Het bestand is ongeveer 200 kB: een float32 neemt 32 bits in, oftewel 4 bytes, en de namen nemen nog wat ruimte in. `load_state_dict` meldt `<All keys matched successfully>`: elke naam uit het bestand stond in het nieuwe model, met de juiste shape. Het geladen model leest 929 van de 1.000 testcijfers goed, precies zoals in de demo van de projectles: dezelfde getallen geven hetzelfde model. Het checkpoint geeft de epoch terug, 10, en de instellingen van de optimizer, zoals zijn learning rate, 0,5; daarmee kan een loop `again` verder trainen vanaf epoch 11.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_saved():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat staat er in het bestand?
    """)
    return


@app.cell(hide_code=True)
def choice_saved_cell():
    choice_saved = L.choice(
        "what-saved",
        question="Wat zet `torch.save(model.state_dict(), path)` in het bestand?",
        options={
            "code": "De class `DigitsMLP` met zijn code, en zijn parameters",
            "right": "De parameters van elke layer, per naam",
            "preds": "De voorspellingen van het model voor de trainingscijfers",
        },
        correct="right",
        why={
            "code": "De code blijft in je eigen bestanden: om te laden maak je eerst een `DigitsMLP`. Zie *Opslaan en laden* in *Het idee*.",
            "preds": "Een state_dict bevat wat het model geleerd heeft, niet wat het voorspelde. Zie *De state_dict* in *Het idee*.",
            "right": "Een state_dict bevat elke parameter per naam: `hidden.weight`, `hidden.bias`, `out.weight` en `out.bias`. Om hem te laden heb je een model van dezelfde class en grootte nodig.",
        },
    )
    choice_saved
    return (choice_saved,)


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Bewaar de beste epoch

        De run uit oefening 1 van [Loss curves lezen](/course/open/torch-loss-curves?lang=nl): je `DigitsMLP` en `fit`, 300 trainingscijfers, 30 epochs. Na elke epoch komt de validation loss in `val_losses`, en de `if` is waar als die de laagste tot nu toe is. Vervang de twee `...`: sla binnen de `if` de parameters van het model op in `path`; laad na de loop het bestand in `best`. In dit vak geeft `checkpoint_path("best.pt")` een pad in de eigen map van de check, `work/checkpoints/check/`, en haalt het eerst het bestand van een eerdere run weg, zodat een oud bestand nooit in de weg zit. De laatste regel laat de validation loss zien van het model na epoch 30, en die van `best`.
        """),
        L.course_versions(),
        L.task("Sla de parameters op telkens als de validation loss de laagste tot nu toe is, en laad het laatst opgeslagen bestand in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-epoch", lines=36)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-epoch", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-epoch",
        [
            "Stap 1 en 2 van het recept: één regel binnen de `if`, één regel na de loop.",
            "Schrijf binnen de `if` de state_dict van `model` naar het bestand op `path`. Na de loop is `best` een nieuw `DigitsMLP(64)`: geef het wat `torch.load` uit dat bestand leest.",
            "`torch.save(model.___(), path)` binnen de `if`, en `best.___(torch.load(path))` na de loop.",
        ],
        why="Elke keer dat de validation loss de laagste tot nu toe is, schrijft `torch.save(model.state_dict(), path)` de parameters van dat moment over het oude bestand heen. De laatste keer is na epoch 9, dus het bestand houdt epoch 9 vast terwijl het trainen doorgaat. `best.load_state_dict(torch.load(path))` geeft `best` die parameters: een validation loss van 0,6424, tegen 0,7589 voor het model na epoch 30.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_checkpoint():
    mo.md(r"""
    ### Oefening 3. Wat hoort er in een checkpoint?

    Vijf dingen uit een training. Zet elk in zijn groep: in elk checkpoint, alleen als je later verder wilt trainen, of helemaal niet in het checkpoint.
    """)
    return


@app.cell(hide_code=True)
def exercise_checkpoint():
    sort_checkpoint = L.sort("checkpoint")
    sort_checkpoint
    return (sort_checkpoint,)


@app.cell(hide_code=True)
def check_checkpoint(sort_checkpoint):
    result_checkpoint = L.check("checkpoint", sort_checkpoint)
    result_checkpoint
    return (result_checkpoint,)


@app.cell(hide_code=True)
def hints_checkpoint():
    L.hints(
        "checkpoint",
        [
            "Vraag bij elk ding: heeft iemand die alleen voorspelt het nodig, heeft iemand die verder traint het nodig, of staat het al ergens anders?",
            "Een checkpoint bevat getallen die het trainen verandert. Wat verandert het trainen, en wat niet?",
            "De parameters heeft iedereen nodig. De optimizer en de epoch tellen alleen om verder te trainen. Welke twee dingen staan toch al in je bestanden?",
        ],
        why="**In elk checkpoint:** `model.state_dict()`, de parameters. **Alleen om later verder te trainen:** `opt.state_dict()` en het nummer van de epoch. **Niet in het checkpoint:** de code van `DigitsMLP`, die in je project staat, en de trainingscijfers, die het trainen niet verandert.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De parameters die [Layers als modules: nn.Linear](/course/open/torch-nn-module?lang=nl) voor elke layer maakte, hebben nu namen in een state_dict, en `torch.save` bewaart ze in een bestand. De beste epoch uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) kun je nu echt bewaren: schrijf de parameters weg telkens als de validation loss de laagste tot nu toe is. In de volgende les, *Early stopping*, stopt het trainen vanzelf zodra de validation loss een tijd niet meer verbetert, en bewaar je het beste checkpoint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_saved, result_best, result_checkpoint):
    L.wrap_up(choice_saved, result_best, result_checkpoint)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
