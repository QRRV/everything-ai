# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-early-stopping. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-early-stopping", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) en [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_jump():
    quick_jump = L.choice(
        "quick-jump",
        question="De validation loss van een run is in zes epochs na elkaar 0,43, 0,41, 0,44, 0,42, 0,40, 0,42. Wat zegt de 0,44 na de 0,41?",
        options={
            "right": "Op zich weinig: één epoch omhoog kan toeval zijn",
            "overfit": "Dat het model daar begint te overfitten",
            "stuck": "Dat het model vastzit",
        },
        correct="right",
        why={
            "overfit": "Twee epochs later is de loss lager dan ooit. Kijk naar de trend over meerdere epochs. Zie *Let op* in [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "stuck": "De loss daalt over deze epochs nog, van 0,43 naar 0,40. Zie *Zit vast* in [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "Een validation set van een paar honderd voorbeelden laat zijn loss door toeval een beetje springen. Hier daalt hij twee epochs later verder tot 0,40.",
        },
    )
    quick_jump
    return (quick_jump,)


@app.cell(hide_code=True)
def quick_check_copy():
    quick_copy = L.choice(
        "quick-copy",
        question="Wat doe je om de parameters van één epoch te bewaren terwijl het trainen doorgaat?",
        options={
            "name": "`best = model.state_dict()`",
            "right": "`torch.save(model.state_dict(), path)`",
            "model": "`best = model`",
        },
        correct="right",
        why={
            "name": "Die dictionary bevat de eigen tensors van het model, en die blijft het trainen veranderen. Zie *Let op* in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "model": "`best` is dan hetzelfde model onder een andere naam, en het verandert mee. Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "right": "Het bestand bevat de getallen van dat moment; latere stappen veranderen het model, niet het bestand.",
        },
    )
    quick_copy
    return (quick_copy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl) trainde je 30 epochs en bewaarde je de epoch met de laagste validation loss. Maar hoeveel epochs moet je trainen? Train je te kort, dan leert het model nog als je stopt; train je te lang, dan besteed je tijd aan epochs die het op nieuwe data alleen slechter maken. Je weet het aantal niet vooraf, en het verschilt per model en per dataset. Laat de validation loss dus beslissen terwijl het trainen loopt.

    **Een voorbeeld uit het dagelijks leven.** Je zoekt een parkeerplek in een lange straat. Elke plek die je passeert, ligt iets verder van waar je moet zijn, of iets dichterbij. Je kunt de hele straat niet in één keer overzien, dus rij je niet door tot het eind: heb je een paar plekken gepasseerd die niet beter zijn dan de beste tot nu toe, dan keer je terug naar die ene.
    """)
    return


@app.cell(hide_code=True)
def predict_stop():
    _end, _epoch, _loss = C.stop(1)
    guess_stop = L.guess(
        "patience-one",
        question="De run uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) heeft zijn laagste validation loss, 0,41, na epoch 8. Je stopt met trainen zodra één epoch niet beter is dan de beste tot nu toe, en bewaart de beste epoch tot dan. Welke epoch denk je dat je bewaart?",
        options={
            "eight": "Epoch 8, met 0,41",
            "thirty": "Epoch 30",
            "right": "Een eerdere epoch, met een hogere loss",
        },
        reveal=f"**Een eerdere epoch, met een hogere loss.** De validation loss daalt van 1,09 via 0,65 en 0,53 naar 0,45, en gaat dan door toeval omhoog naar 0,51 na epoch 5. Het trainen stopt dus na epoch {_end} en bewaart epoch {_epoch}, met {L.number(_loss, 2)}: het laagste punt na epoch 8 wordt nooit bereikt. Eén epoch omhoog is niet genoeg om op te stoppen. Wacht je een paar epochs, de *patience* uit *Het idee*, dan stop je niet op zo'n sprong.",
    )
    guess_stop
    return (guess_stop,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Early stopping** stopt het trainen zodra de validation loss een tijd niet meer is verbeterd, en bewaart het checkpoint van de beste epoch. Na elke epoch:

        - Is de validation loss de laagste tot nu toe? Sla dan een checkpoint op, onthoud de epoch, en zet de telling van epochs zonder verbetering terug op 0.
        - Zo niet, tel dan 1 op bij die telling. Bereikt die de **patience**, een getal dat je zelf kiest, dan stop je. Met `break`, zoals in [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl), houdt de loop meteen op.

        Daarna laad je het checkpoint: het model van de beste epoch, niet het model bij de stop. Dat is al `patience` epochs verder, en niet beter op de validation set.

        De patience is een hyperparameter, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl). Is hij te klein, dan stopt een sprong door toeval het trainen te vroeg, zoals in de gok; is hij te groot, dan train je veel epochs die niet helpen. Springt de validation loss veel, bijvoorbeeld bij een kleine validation set, dan heb je meer patience nodig.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **early stopping** | | stoppen met trainen zodra de validation loss `patience` epochs niet is verbeterd, en het checkpoint van de beste epoch bewaren | `break` |
        | **patience** | | hoeveel epochs na elkaar zonder nieuwe laagste validation loss je wacht voordat je stopt | `patience = 5` |
        """),
        L.watch_out("Early stopping kiest de epoch met de validation set. De validation loss van die epoch is dus een beetje te gunstig: het is de laagste van veel pogingen. Voor een eerlijk getal op nieuwe data meet je het bewaarde model nog één keer, op de test set."),
        mo.md(r"""
        De figuur: de run uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl), met early stopping bij een patience van 5. Eén run, de training loss doorgetrokken, de validation loss gestreept.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_stop():
    _end, _epoch, _loss = C.stop(5)
    mo.vstack([
        mo.Html(C.picture(5, {"title": "Early stopping met een patience van 5", "epoch": "epoch", "loss": "loss",
                              "kept": "bewaard", "stop": "stop",
                              "alt": f"De training loss, doorgetrokken, en de validation loss, gestippeld, op een logaritmische schaal, tot epoch {_end}. Een stip markeert de laagste validation loss, na epoch {_epoch}, op {L.number(_loss, 2)}; een dunne lijn markeert de stop, na epoch {_end}."})),
        mo.md(f"""
        De validation loss is het laagst na epoch {_epoch}, op <span class="c-loss ea-num">{L.number(_loss, 2)}</span>. In de vijf epochs daarna, tot en met epoch {_end}, kwam er geen lagere, dus stopt het trainen daar, en is het checkpoint van epoch {_epoch} het model dat je bewaart. De 30 epochs uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) waren er {30 - _end} meer dan nodig.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te trainen met early stopping:

    1. Kies de patience, en begin met `waited = 0`.
    2. Meet na elke epoch de validation loss.
    3. Is die de laagste tot nu toe? Sla een checkpoint op, onthoud de epoch, en zet `waited` terug op 0.
    4. Zo niet, tel 1 op bij `waited`; bereikt `waited` de patience, stop dan met `break`.
    5. Laad na de loop het checkpoint in een nieuw model: dat is het model dat je bewaart.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De run uit de figuur, met de kopie die de course zelf van je `DigitsMLP` heeft: 1.000 trainingscijfers, 500 validatiecijfers, $\eta$ = 0,5, hooguit 30 epochs, een patience van 5. Het checkpoint komt in je werkmap, zoals in de vorige les. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_stop():
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
    X_train, y_train = X[:1000], y[:1000]
    X_val, y_val = X[5000:5500], y[5000:5500]
    loss_fn = nn.CrossEntropyLoss()
    path = C.checkpoint_path("early.pt")

    torch.manual_seed(0)
    model = DigitsMLP(64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)
    torch.manual_seed(1)
    # 1. de patience, en nog geen epochs gewacht
    patience = 5
    waited = 0
    best_loss = 100.0   # hoger dan elke loss hier, dus de eerste epoch is een nieuwe laagste
    best_epoch = 0
    for epoch in range(30):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 2. de validation loss na de epoch
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        model.train()
        # 3. een nieuwe laagste: opslaan, onthouden, opnieuw beginnen met tellen
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), path)
            waited = 0
        # 4. anders tellen, en stoppen bij de patience
        else:
            waited = waited + 1
            if waited == patience:
                break
    print(epoch + 1, best_epoch, best_loss)

    # 5. het model dat je bewaart: het checkpoint van de beste epoch
    kept = DigitsMLP(64)
    kept.load_state_dict(torch.load(path))
    kept.eval()
    with torch.no_grad():
        print(loss_fn(kept(X_val), y_val).item(), loss_fn(model(X_val), y_val).item())
    return


@app.cell(hide_code=True)
def text_after_example():
    _end, _epoch, _loss = C.stop(5)
    _stop_loss = C.curves()[1][_end - 1]
    mo.md(f"""
    Het trainen stopt na epoch {_end}: `epoch` is dan {_end - 1}, geteld vanaf 0. De beste epoch is {_epoch}, met een validation loss van {L.number(_loss, 4)}, en het bewaarde model, geladen uit het bestand, heeft precies die loss. Het model bij de stop is op de validation set een beetje slechter, {L.number(_stop_loss, 4)}. `best_loss` begint op `100.0`, hoger dan elke loss hier, zodat de eerste epoch altijd een nieuwe laagste is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welk model?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-model",
        question="Early stopping met een patience van 5 stopt na epoch 13; de laagste validation loss was na epoch 8. Welk model gebruik je?",
        options={
            "stop": "Het model na epoch 13: dat heeft het langst getraind",
            "right": "Het checkpoint van epoch 8",
            "retrain": "Nog geen: eerst nog 13 epochs trainen",
        },
        correct="right",
        why={
            "stop": "Na epoch 8 werd de validation loss in vijf epochs niet lager: het model bij de stop is op de validation set niet beter. Zie *Het idee*.",
            "retrain": "Verder trainen gaat alleen verder voorbij de beste epoch. Zie *Het idee*.",
            "right": "Het checkpoint bevat de parameters van epoch 8, de beste op de validation set. De vijf epochs erna dienden alleen om zeker te zijn dat er geen betere kwam.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_stop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vanzelf stoppen

        Je `DigitsMLP` en `fit`, op 300 trainingscijfers, met een patience van 5 en hooguit 100 epochs. Na elke epoch komt de validation loss in `val_losses`. Vervang de `...` door stap 3 en 4 van het recept: sla bij een nieuwe laagste validation loss de parameters op in `path` en onthoud de epoch, geteld vanaf 1, in `best_epoch`; tel anders de epoch in `waited`, en stop met `break` als die `patience` bereikt. De test voor een nieuwe laagste is `val_losses[-1] == min(val_losses)`, zoals in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl). De laatste regel laat de epoch zien waarna het trainen stopte, de epoch die je bewaart, en de validation loss van dat model.
        """),
        L.course_versions(),
        L.task("Sla op bij elke nieuwe laagste validation loss, tel de epochs zonder, en stop als de telling `patience` bereikt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stop():
    box_stop = L.code_box("early-stop", lines=38)
    box_stop
    return (box_stop,)


@app.cell(hide_code=True)
def check_stop(box_stop):
    result_stop = L.check("early-stop", box_stop.value)
    result_stop
    return (result_stop,)


@app.cell(hide_code=True)
def hints_stop():
    L.hints(
        "early-stop",
        [
            "Stap 3 en 4 van het recept, zoals in het uitgewerkte voorbeeld; alleen de test op een nieuwe laagste is die uit de vorige les.",
            "Een `if` met de test op een nieuwe laagste, en een `else`. In de `if`: opslaan, de epoch onthouden, `waited` op 0. In de `else`: 1 bij `waited` optellen, en een `if` erin die de loop stopt.",
            "`if val_losses[-1] == min(val_losses):` met `torch.save(...)`, `best_epoch = epoch + 1` en `waited = 0`; `else:` met `waited = waited + 1` en `if waited == patience:` met `___`.",
        ],
        why="De validation loss is het laagst na epoch 9. Epoch 10 tot en met 14 zijn niet lager, dus `waited` gaat van 1 naar 5, en na epoch 14 beëindigt `break` de loop. Het bestand bevat epoch 9, dus het bewaarde model heeft 0,6424, terwijl het model bij de stop een hogere heeft, 0,6998.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_patience():
    mo.md(r"""
    ### Oefening 3. Hoeveel patience?
    """)
    return


@app.cell(hide_code=True)
def choice_patience_cell():
    choice_patience = L.choice(
        "patience",
        question="Je validation set heeft maar 100 cijfers, dus zijn loss springt veel van epoch tot epoch. Welke patience kies je, en wat kost die?",
        options={
            "small": "Een kleine, zoals 1: dan stopt een sprong het trainen voordat het model overfit",
            "none": "Het maakt niet uit: early stopping bewaart altijd de beste epoch",
            "right": "Een grotere, zoals 10: het trainen duurt na de beste epoch meer epochs",
        },
        correct="right",
        why={
            "small": "Een sprong door toeval is geen overfitting; met een patience van 1 stopt hij het trainen te vroeg, zoals in de gok. Zie *Het idee*.",
            "none": "Early stopping bewaart de beste epoch tot aan de stop. Stop je te vroeg, dan komt de beste epoch nooit. Zie de gok.",
            "right": "Met meer patience stopt een sprong door toeval het trainen niet. De prijs: na de beste epoch traint het nog `patience` epochs voordat het stopt.",
        },
    )
    choice_patience
    return (choice_patience,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Early stopping leest de curves uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) terwijl het trainen loopt, en gebruikt de checkpoints uit [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl) om de beste epoch te bewaren. Het is een eenvoudige rem op overfitting: het trainen stopt kort nadat het model zijn voorbeelden uit het hoofd begint te leren. In de volgende les, *Data augmentation*, pak je overfitting van de andere kant aan: met meer gevarieerde voorbeelden, gemaakt van de voorbeelden die je hebt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_patience, choice_which, result_stop):
    L.wrap_up(choice_which, result_stop, choice_patience)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
