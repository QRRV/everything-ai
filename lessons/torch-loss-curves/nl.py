# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-loss-curves. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-loss-curves", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl) en [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Een loss daalt in de eerste 5 epochs van 1 naar 0,1, en in de 5 daarna van 0,1 naar 0,01. Hoe zien de twee dalingen eruit in een grafiek met een logaritmische schaal op de y-as?",
        options={
            "smaller": "De tweede daling is maar een tiende zo lang",
            "right": "Beide dalingen zijn even lang",
            "larger": "De tweede daling is tien keer zo lang",
        },
        correct="right",
        why={
            "smaller": "Zo zien ze eruit op een gewone as. Op een logaritmische schaal zijn gelijke factoren gelijke stappen. Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "larger": "Op een logaritmische schaal zijn gelijke factoren gelijke stappen, en beide dalingen zijn een factor 10. Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "right": "Beide dalingen delen de loss door 10, en op een logaritmische schaal is elke factor 10 even ver. Zo laat een logaritmische schaal ook een loss zien die dicht bij 0 blijft dalen.",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def quick_check_train():
    quick_train = L.choice(
        "quick-train",
        question="De loss van een model op zijn trainingsvoorbeelden blijft dalen. Wat zegt dat over zijn loss op nieuwe voorbeelden?",
        options={
            "down": "Dat die ook daalt",
            "up": "Dat die stijgt",
            "right": "Nog niets: dat kunnen alleen voorbeelden vertellen waarop het nooit trainde",
        },
        correct="right",
        why={
            "down": "Een model kan zijn trainingsvoorbeelden ook uit zijn hoofd leren. Zie [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "up": "Niet per se: de training loss alleen kan het niet zeggen. Zie [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "right": "De training loss zegt hoe goed het model past bij de voorbeelden waarvan het leert. Hoe goed het model het doet op nieuwe voorbeelden, zie je alleen aan voorbeelden waarop het nooit trainde.",
        },
    )
    quick_train
    return (quick_train,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de demo van [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) daalde de loss van je MLP elke epoch. Dat was de loss op de cijfers waarop het traint. Om te weten hoe lang je traint, en of je iets moet veranderen, heb je er een tweede loss naast nodig, op cijfers waarop het model nooit traint. Twee curves samen vertellen meer dan elk afzonderlijk: of het model nog leert, zijn voorbeelden uit zijn hoofd leert, of vastzit.

    **Een voorbeeld uit het dagelijks leven.** Wie oefent met dezelfde 20 examenvragen, heeft er elke dag meer goed. Of iemand voor het examen zou slagen, blijkt pas uit vragen die nog nooit voorbij zijn gekomen.
    """)
    return


@app.cell(hide_code=True)
def predict_val():
    guess_val = L.guess(
        "val-loss",
        question="Een MLP traint 30 epochs op 1.000 handgeschreven cijfers, met een learning rate van 0,5. Zijn loss op die 1.000 cijfers daalt van 0,98 na de eerste epoch naar 0,0055. Wat denk je dat zijn loss op 500 andere cijfers doet?",
        options={
            "along": "Die daalt mee, tot dicht bij 0",
            "right": "Die daalt een paar epochs, en stijgt dan langzaam",
            "flat": "Die blijft waar hij begon",
        },
        reveal="**Die daalt een paar epochs, en stijgt dan langzaam.** In het uitgewerkte voorbeeld hieronder is de loss op de 500 andere cijfers het laagst na epoch 8, op 0,41, en stijgt hij tot 0,48 bij epoch 30, terwijl de loss op de trainingscijfers blijft dalen. Na epoch 8 leert het model zijn 1.000 cijfers uit zijn hoofd: het wordt er beter in, en een beetje slechter in nieuwe. Dat is de overfitting uit [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl), nu epoch na epoch.",
    )
    guess_val
    return (guess_val,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Meet na elke epoch twee losses, in eval mode en zonder graaf: de **training loss**, op alle voorbeelden waarop het model traint, en de **validation loss**, op een validation set zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl), voorbeelden waarop het nooit traint. Teken ze als twee curves in één grafiek: de training loss doorgetrokken, de validation loss gestippeld. De losses komen al snel dicht bij 0, dus een logaritmische schaal op de y-as, zoals in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl), laat zien wat daar gebeurt.

        Drie patronen vertellen wat er aan de hand is:

        - **Leert nog.** Beide curves dalen nog, dicht bij elkaar. Het model heeft nog niet geleerd wat het kan: train meer epochs.
        - **Overfitting.** De training loss blijft dalen, maar de validation loss bereikt een laagste punt en stijgt weer. Vanaf daar leert het model zijn voorbeelden uit zijn hoofd. Het beste model is dat van de epoch met de laagste validation loss; meer trainingsvoorbeelden maken uit het hoofd leren moeilijker.
        - **Zit vast.** Beide curves blijven hoog en bijna vlak. Voor tien cijfers is een loss van ongeveer 2,30 de loss van gokken, een kans van 0,1 voor elk cijfer. De stappen veranderen het model te weinig: de learning rate is te klein, of er klopt iets niet in de update-stap, zoals een vergeten `opt.step()`.

        Een learning rate die te groot is, zoals in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl), geeft een vierde patroon: een loss die op en neer springt, of groeit. Een learning rate die veel te groot is, kan beide curves ook laten vastzitten op de loss van gokken: vastzittende curves kunnen komen van een stap die te klein is of een die veel te groot is.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **training loss** | | de loss op de voorbeelden waarop het model traint, hier gemeten na elke epoch | `train_losses` |
        | **validation loss** | | de loss op de validation set, voorbeelden waarop het model nooit traint | `val_losses` |
        | **train- en validation-curves** | | de twee losses per epoch in één grafiek: training doorgetrokken, validation gestippeld | `plt.plot(train_losses)` |
        | `"--"` | | in `plt.plot` een gestippelde lijn | `plt.plot(val_losses, "--")` |
        """),
        L.watch_out("Beide curves springen een beetje van epoch tot epoch: de mini-batches van elke epoch duwen het model een kant op, en een paar honderd validatievoorbeelden zijn een kleine steekproef. Eén epoch omhoog is nog geen overfitting: kijk naar de trend over meerdere epochs. En de curves zijn één run: met een andere seed kan het laagste punt een aantal epochs eerder of later vallen: bij tien seeds van de middelste run hieronder ergens tussen epoch 5 en 13."),
        mo.md(r"""
        De figuur: drie runs van het MLP met 64 hidden neuronen, elk op dezelfde 1.000 trainingscijfers en 500 validatiecijfers, 30 epochs lang, met drie learning rates: de training loss doorgetrokken, de validation loss gestippeld. Elk is één run.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_runs():
    _epoch, _loss = C.best("overfitting")
    mo.vstack([
        mo.Html(C.picture({"learning": "η = 0,05: leert nog", "overfitting": "η = 0,5: overfitting", "stuck": "η = 0,001: zit vast",
                           "epoch": "epoch", "loss": "loss", "lowest": "laagste",
                           "alt": f"Drie grafieken van de loss per epoch op een logaritmische schaal, de training loss doorgetrokken en de validation loss gestippeld. Links, met een learning rate van 0,05, dalen ze allebei van ongeveer 2 tot onder 0,5, dicht bij elkaar. In het midden, met 0,5, daalt de training loss tot onder 0,01, terwijl de validation loss het laagst is na epoch {_epoch}, op {L.number(_loss, 2)}, en daarna stijgt. Rechts, met 0,001, dalen ze allebei maar van 2,29 naar 2,16."})),
        mo.md(f"""
        Links dalen beide curves na 30 epochs nog. In het midden is de validation loss het laagst na epoch {_epoch}, op <span class="c-loss ea-num">{L.number(_loss, 2)}</span>, terwijl de training loss verder daalt tot onder 0,01. Rechts dalen ze allebei maar van 2,29 naar 2,16, nauwelijks weg van 2,30, de loss van gokken.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te lezen hoe een training gaat:

    1. Houd een validation set apart: voorbeelden waarop het model nooit traint.
    2. Meet na elke epoch, in eval mode en zonder graaf, de loss op alle trainingsvoorbeelden en op de validation set, en voeg ze elk aan hun eigen lijst toe.
    3. Teken ze allebei in één grafiek, `plt.plot(train_losses)` en `plt.plot(val_losses, "--")`; met losses dicht bij 0 komt er `plt.yscale("log")` bij.
    4. Lees de curves: allebei nog dalend is leert nog; training dalend en validation stijgend is overfitting, en de beste epoch is die met de laagste validation loss; allebei hoog en vlak is zit vast.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De middelste run uit de figuur: een MLP met dezelfde layers als je `DigitsMLP`, getraind op de eerste 1.000 trainingscijfers met $\eta$ = 0,5, 30 epochs met mini-batches van 50. De validation set is 500 andere trainingscijfers, nummer 5000 tot en met 5499. De cel tekent de twee curves en print de beste epoch. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit; dat duurt een seconde.
    """)
    return


@app.cell
def example_curves():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt

    class Net(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))

    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    # 1. een validation set apart: cijfers waarop het model nooit traint
    X_train, y_train = X[:1000], y[:1000]
    X_val, y_val = X[5000:5500], y[5000:5500]

    torch.manual_seed(0)
    model = Net(64)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=50, shuffle=True)
    torch.manual_seed(1)
    train_losses = []
    val_losses = []
    for epoch in range(30):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 2. na elke epoch: beide losses, in eval mode en zonder graaf
        model.eval()
        with torch.no_grad():
            train_losses.append(loss_fn(model(X_train), y_train).item())
            val_losses.append(loss_fn(model(X_val), y_val).item())
        model.train()

    # 3. beide curves in één grafiek, op een logaritmische schaal
    plt.plot(range(1, 31), train_losses)
    plt.plot(range(1, 31), val_losses, "--")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    # 4. de beste epoch: de laagste validation loss
    best = np.argmin(val_losses)
    print(best + 1, val_losses[best], train_losses[best])
    print(train_losses[-1], val_losses[-1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De doorgetrokken curve, de training loss, daalt helemaal: van 0,98 na de eerste epoch tot 0,0055 na de laatste. De gestippelde curve, de validation loss, daalt tot zijn laagste punt na epoch 8, op 0,41, met dan een training loss van 0,08, en stijgt tot 0,48 bij epoch 30. `np.argmin` geeft de plek van de laagste validation loss, geteld vanaf 0, dus de epoch is `best + 1`; in de grafiek nummert `range(1, 31)` de epochs van 1 tot en met 30 langs de x-as. Na epoch 8 wordt het model beter in zijn 1.000 cijfers en een beetje slechter in nieuwe: overfitting. Had je het model van epoch 8 bewaard, dan was dat het beste geweest op nieuwe cijfers; de volgende lessen laten zien hoe.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_diagnose():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Lees de curves

    Hetzelfde MLP, nu getraind op maar 300 trainingscijfers, met $\eta$ = 0,5, 30 epochs lang, en gemeten op dezelfde 500 validatiecijfers. Eén run.
    """)
    return


@app.cell(hide_code=True)
def choice_diagnose_cell():
    _epoch, _loss = C.best(C.DIAGNOSE)
    _best_1000 = C.best("overfitting")[1]
    choice_diagnose = L.choice(
        "diagnose",
        question="Wat laten deze curves zien, en het model van welke epoch zou je bewaren?",
        options={
            "right": "Overfitting; het model van de epoch waar de gestippelde curve het laagst is",
            "last": "Leert nog; het model van epoch 30, waar de doorgetrokken curve het laagst is",
            "stuck": "Zit vast; elke epoch is ongeveer even goed als de andere",
        },
        correct="right",
        why={
            "last": "De doorgetrokken curve is de training loss: op de cijfers waarvan het leert, wordt het model steeds beter. Op nieuwe cijfers niet. Zie *Overfitting* in *Het idee*.",
            "stuck": "De training loss daalt van boven 1 tot dicht bij 0,01: de stappen veranderen het model wel. Zie *Zit vast* in *Het idee*.",
            "right": f"De training loss daalt tot dicht bij 0, terwijl de validation loss het laagst is na epoch {_epoch}, op {L.number(_loss, 2)}, en daarna stijgt. Met maar 300 cijfers ligt dat laagste punt hoger dan met de 1.000 cijfers van het uitgewerkte voorbeeld, {L.number(_best_1000, 2)}: minder voorbeelden zijn makkelijker uit het hoofd te leren.",
        },
    )
    mo.vstack([
        mo.Html(C.diagnose_picture({"title": "300 trainingscijfers, η = 0,5", "key": "training loss doorgetrokken, validation loss gestippeld",
                                    "epoch": "epoch", "loss": "loss",
                                    "alt": f"De loss per epoch op een logaritmische schaal. De training loss, doorgetrokken, daalt van boven 1 tot dicht bij 0,01. De validation loss, gestippeld, daalt tot zijn laagste punt na epoch {_epoch}, op {L.number(_loss, 2)}, en stijgt daarna langzaam tot ongeveer 0,76."})),
        choice_diagnose,
    ])
    return (choice_diagnose,)


@app.cell(hide_code=True)
def text_exercise_curves():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Meet beide curves

        De run uit oefening 1, met je eigen `DigitsMLP` en `fit` uit je project. Elke epoch traint `fit(model, loader, 1, 0.5)` één epoch. Vervang de `...` door stap 2 van het recept: meet na de epoch de loss op heel `X_train` en op heel `X_val`, in eval mode en zonder graaf, en voeg ze elk aan hun lijst toe. De laatste regel laat de laatste training loss, de laatste validation loss en de laagste validation loss zien.
        """),
        L.course_versions(),
        L.task("Voeg na elke epoch de training loss toe aan `train_losses` en de validation loss aan `val_losses`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_curves():
    box_curves = L.code_box("curves", lines=26)
    box_curves
    return (box_curves,)


@app.cell(hide_code=True)
def check_curves(box_curves):
    result_curves = L.check("curves", box_curves.value)
    result_curves
    return (result_curves,)


@app.cell(hide_code=True)
def hints_curves():
    L.hints(
        "curves",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld, ingesprongen onder `for epoch in range(30):`, na de regel met `fit`.",
            "Eval mode, dan een `with torch.no_grad():`-blok met twee `append`-regels: `loss_fn` op de logits van het model voor heel `X_train` met `y_train`, en hetzelfde voor `X_val` met `y_val`. Daarna terug naar train mode.",
            "`model.eval()`, dan `with torch.no_grad():` met `train_losses.append(loss_fn(model(___), ___).item())` en `val_losses.append(...)` voor `X_val`, dan `model.train()`.",
        ],
        why="Na elke epoch is `loss_fn(model(X_train), y_train)` de training loss en `loss_fn(model(X_val), y_val)` de validation loss; `.item()` maakt van elk een getal voor de lijst. De laatste training loss is 0,0098, dicht bij 0, terwijl de validation loss eindigt op 0,7589, boven zijn laagste punt, 0,6424.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_remedy():
    mo.md(r"""
    ### Oefening 3. Wat helpt?

    Vijf dingen die je kunt doen nadat je de curves hebt gelezen. Zet elk bij de curves met het probleem dat het verhelpt: leert nog, overfitting of zit vast.
    """)
    return


@app.cell(hide_code=True)
def exercise_remedy():
    sort_remedy = L.sort("remedy")
    sort_remedy
    return (sort_remedy,)


@app.cell(hide_code=True)
def check_remedy(sort_remedy):
    result_remedy = L.check("remedy", sort_remedy)
    result_remedy
    return (result_remedy,)


@app.cell(hide_code=True)
def hints_remedy():
    L.hints(
        "remedy",
        [
            "Vraag bij elke oplossing wat ze verandert: hoe lang het model traint, welk model je bewaart, waarvan het leert, of hoe groot zijn stappen zijn.",
            "*Overfitting* krijgt twee oplossingen, en *Zit vast* ook.",
            "Langer trainen is voor curves die al een eind gedaald waren en nog daalden. Welke twee oplossingen gaan uit het hoofd leren tegen, en welke twee zorgen dat de stappen iets doen?",
        ],
        why="**Leert nog:** train meer epochs. **Overfitting:** bewaar het model van de epoch met de laagste validation loss, en train op meer voorbeelden. **Zit vast:** verhoog de learning rate, en controleer of elke update-stap de parameters verandert.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De validation set uit [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) geeft nu een curve, epoch na epoch, naast de training loss die je `fit` uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) al gaf. De overfitting uit [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl) verschijnt daarin als een validation loss die weer stijgt, en een logaritmische schaal laat zien wat er dicht bij 0 gebeurt. Om het model van de beste epoch te bewaren, moet je het opslaan terwijl het trainen doorgaat: in de volgende les, *Een model opslaan en laden*, leer je hoe.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_diagnose, result_curves, result_remedy):
    L.wrap_up(choice_diagnose, result_curves, result_remedy)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
