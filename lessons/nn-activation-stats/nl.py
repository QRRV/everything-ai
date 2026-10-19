# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-activation-stats. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-activation-stats", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) en [Initialisatie](/course/open/nn-initialization?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_guessing():
    quick_guessing = L.choice(
        "quick-guessing",
        question="De training loss en de validation loss van een classifier voor de tien cijfers blijven allebei op ongeveer 2,30. Wat is 2,30 hier?",
        options={
            "right": "De loss van gokken, met een kans van 0,1 voor elk cijfer",
            "accuracy": "Het aantal cijfers van de 10 dat het model goed heeft",
            "overfitting": "De loss waarbij overfitting begint, voor elk model van tien cijfers",
        },
        correct="right",
        why={
            "accuracy": "Een cross-entropy-loss telt geen goede antwoorden. 2,30 is $-\\log 0{,}1$: de loss als het goede cijfer een kans van 0,1 krijgt. Zie *Zit vast* in [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "overfitting": "Overfitting zie je aan een validation loss die stijgt terwijl de training loss daalt; hier blijven ze allebei staan. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "$-\\log 0{,}1 \\approx 2{,}30$: het model geeft elk cijfer dezelfde kans. De curves zitten vast, en ze zeggen niet waarom.",
        },
    )
    quick_guessing
    return (quick_guessing,)


@app.cell(hide_code=True)
def quick_check_he():
    quick_he = L.choice(
        "quick-he",
        question="He initialization verdubbelt de variantie van de beginweights voor ReLU. Waarom?",
        options={
            "right": "ReLU zet ongeveer de helft van de z's op 0, en dat halveert het signaal",
            "pieces": "ReLU bestaat uit twee rechte stukken, dus elk stuk heeft zijn eigen deel van de spreiding nodig",
            "slope": "De helling van ReLU is 2 boven 0, dus het signaal verdubbelt in elke layer",
        },
        correct="right",
        why={
            "pieces": "De twee stukken krijgen niet elk een spreiding. Waar het om gaat: ReLU geeft 0 voor ongeveer de helft van zijn inputs. Zie [Initialisatie](/course/open/nn-initialization?lang=nl).",
            "slope": "De helling van ReLU is 1 boven 0, niet 2. Zie [Initialisatie](/course/open/nn-initialization?lang=nl).",
            "right": "Ongeveer de helft van de $z$'s ligt onder 0 en komt eruit als 0. Vandaag kijk je naar neuronen die bij elk voorbeeld 0 geven.",
        },
    )
    quick_he
    return (quick_he,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De loss is één getal voor het hele netwerk. Zit hij vast, zoals in [Loss curves lezen](/course/open/torch-loss-curves?lang=nl), dan zegt hij *dat* er iets mis is, niet *waar*. Een netwerk met drie hidden layers heeft drie plekken waar het mis kan gaan, en elke layer kan het laten zien: neuronen die nooit iets anders geven dan 0, of getallen die layer na layer groeien. Om dat te zien, kijk je naar binnen: naar wat elke layer geeft voor veel voorbeelden tegelijk, en naar de grootte van zijn gradients. PyTorch laat je dat doen zonder het model te veranderen.
    """)
    return


@app.cell(hide_code=True)
def predict_dead():
    guess_dead = L.guess(
        "how-many-dead",
        question="Een MLP met drie hidden layers van 128 ReLU-neuronen traint één epoch op de 6.000 trainingscijfers, met een learning rate van 2,0. Zijn loss eindigt op 2,30: het gokt. Hoeveel van de 128 neuronen van de derde hidden layer geven volgens jou 0 voor elk van de 1.000 testcijfers?",
        options={
            "tenth": "Ongeveer 13: een tiende ervan",
            "half": "Ongeveer 64: ReLU zet ongeveer de helft uit",
            "right": "Alle 128: de hele layer staat uit",
        },
        reveal="**Alle 128.** Elke $z$ van de derde layer is 0 of lager, voor elk testcijfer, dus na ReLU geeft de layer alleen nullen. De output-layer krijgt dan voor elk cijfer dezelfde input, en kan alleen gokken. En zo blijft het: onder 0 is de helling van ReLU 0, dus er komt geen gradient terug door deze neuronen, en hun weights veranderen nooit meer.",
    )
    guess_dead
    return (guess_dead,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een ReLU-neuron waarvan de $z$ bij elk voorbeeld 0 of lager is, is een **dead neuron**: het geeft altijd 0. Onder 0 is de helling van ReLU 0, dus elke gradient door het neuron is ook 0, voor zijn weights en zijn bias term:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial w_{jk}}]] = \sum_i [[grad:\frac{\partial \mathcal{L}}{\partial h_{ij}}]] \cdot \mathrm{ReLU}'(z_{ij}) \cdot x_{ik} = 0 \qquad \text{als } z_{ij} \le 0 \text{ voor elk voorbeeld } i"),
        mo.md(r"""
        Hier is $j$ het neuron, $k$ een van zijn inputs en $i$ het voorbeeld. Geen stap verschuift deze weights. Alleen een verandering in de layers ervoor kan zo'n neuron nog terugbrengen; in de eerste layer kan niets dat. Eén stap die veel te groot is, kan de bias terms en weights van veel neuronen zo ver omlaag duwen, en een hele layer kan doodgaan. Geeft een layer alleen nullen, dan komt er niets door hem heen, en krijgen de layers ervoor ook geen gradient.

        Om dead neurons te vinden, kijk je naar wat elke layer geeft. Een **forward hook** is een functie die PyTorch aanroept elke keer dat een layer zijn output heeft berekend. `layer.register_forward_hook(keep)` hangt de functie `keep` aan `layer`; die krijgt drie dingen, de layer, zijn inputs en zijn output, en kan de output in een lijst bewaren. Hooks draaien in de volgorde waarin het model zijn layers draait. `register_forward_hook` geeft een handle terug, en `handle.remove()` haalt de hook er weer af. Een hook op een `nn.Linear` krijgt zijn $z$'s, één rij per voorbeeld en één kolom per neuron. Een neuron is dead als zijn hele kolom 0 of lager is: `(Z <= 0).all(dim=0)` geeft `True` voor elke zo'n kolom.

        Een **activation histogram** is een histogram van wat één layer geeft, voor veel voorbeelden tegelijk. Hier laat het de $z$'s zien, vóór ReLU: na ReLU zou alles onder 0 zich op 0 ophopen, en zou je niet zien hoe ver eronder het ligt. Voor de $z$'s van een gezonde ReLU-layer ligt het aan beide kanten van 0; voor een layer die doodgaat, schuift het onder 0. En $z$'s die layer na layer groter worden, of gradients die in één layer veel groter zijn dan in de andere, wijzen op exploding gradients, zoals in [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **dead neuron** | | een ReLU-neuron waarvan de $z$ bij elk voorbeeld 0 of lager is: het geeft altijd 0, en krijgt geen gradient | `(Z <= 0).all(dim=0)` |
        | **forward hook** | | een functie die PyTorch aanroept elke keer dat een layer zijn output heeft berekend | `layer.register_forward_hook(keep)` |
        | `handle.remove()` | handle remove | haalt een hook weer van zijn layer af | `handle = layer.register_forward_hook(keep)` |
        | **activation histogram** | | een histogram van wat een layer geeft, voor veel voorbeelden tegelijk | |
        | `.all(dim=0)` | all, dim nul | per kolom: `True` als elke waarde erin `True` is | `(Z <= 0).all(dim=0)` |
        """),
        L.watch_out("Haal de hooks eraf als je klaar bent. Een hook blijft aan zijn layer hangen tot je hem verwijdert: elke latere run van het model roept hem opnieuw aan, ook tijdens het trainen, en een lijst die hij vult, blijft groeien. En draai het model onder `torch.no_grad()` terwijl je kijkt, zodat de bewaarde outputs niet aan een hele graaf vast blijven zitten."),
        mo.md(r"""
        De figuur: activation histograms van de derde hidden layer, zijn $z$'s voor alle 128 neuronen en de 1.000 testcijfers, na één epoch met $\eta = 0{,}3$ en met $\eta = 2{,}0$. Met 0,3 liggen de $z$'s aan beide kanten van 0, de meeste tussen −1 en 2, en zijn 2 van de 128 neuronen dead. Met 2,0 is elke $z$ 0 of lager, de meeste net eronder: ReLU maakt ze allemaal 0.
        """),
        mo.Html(C.z_picture({"good": "η = 0,3: {dead} van de 128 neuronen dead", "large": "η = 2,0: {dead} van de 128 neuronen dead",
                             "x": "z in hidden layer 3", "y": "deel van de z's", "zero": "← ReLU geeft 0",
                             "alt": "Twee histogrammen van de z's van hidden layer 3. Met een learning rate van 0,3 liggen ze tussen ongeveer −1 en 2, aan beide kanten van 0, het hoogst net onder 0. Met 2,0 is elke z 0 of lager: 70% ligt net onder 0, de rest tot ongeveer −1."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kijk je in een netwerk:

    1. Kies de layers waar je naar kijkt, en een vaste batch voorbeelden, zoals de testcijfers.
    2. Hang aan elke layer een forward hook die zijn output bewaart.
    3. Draai het model één keer op de batch, onder `torch.no_grad()`, en haal de hooks er weer af.
    4. Per layer: het deel dead neurons, en een histogram van zijn $z$'s.
    5. Voor de gradients: één backward pass op de batch, en dan de gemiddelde grootte van de gradient van de weights van elke layer. Een layer met gradients van 0, of veel kleinere of grotere dan de andere, is waar het misgaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het netwerk uit de gok, één epoch getraind met een learning rate van 2,0; de eerste cel traint het, in hooguit een paar seconden. De tweede cel kijkt naar binnen; in stap 5 doet `model.zero_grad()` wat `opt.zero_grad()` doet, voor elke weight en bias term van het model, zodat de gradients van de laatste trainingsstap niet meetellen, en `.abs()` en `.mean()` werken op een tensor zoals `np.abs` en `np.mean` op een array. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 5 is niet af: voer de cellen uit zoals ze zijn, maak stap 5 dan af waar het commentaar dat zegt, en voer hem opnieuw uit. Verander daarna `lr=2.0` in de eerste cel in `lr=0.3`: de tweede cel draait dan vanzelf opnieuw.
    """)
    return


@app.cell
def example_train():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.h1 = nn.Linear(784, 128)
            self.h2 = nn.Linear(128, 128)
            self.h3 = nn.Linear(128, 128)
            self.out = nn.Linear(128, 10)

        def forward(self, x):
            return self.out(torch.relu(self.h3(torch.relu(self.h2(torch.relu(self.h1(x)))))))

    _data = np.load("data/qmnist-small.npz")
    _X = torch.tensor(_data["train_images"], dtype=torch.float32) / 255
    _y = torch.tensor(_data["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(_data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(_data["test_labels"], dtype=torch.int64)
    # één epoch trainen met SGD
    torch.manual_seed(0)
    model = DeepMLP()
    _loader = DataLoader(TensorDataset(_X, _y), batch_size=64, shuffle=True)
    _opt = torch.optim.SGD(model.parameters(), lr=2.0)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(2)
    for _xb, _yb in _loader:
        _loss = loss_fn(model(_xb), _yb)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    print(_loss.item())
    return X_test, loss_fn, model, torch, y_test


@app.cell
def example_look(X_test, loss_fn, model, torch, y_test):
    # 1. de drie hidden layers, en de 1.000 testcijfers
    _layers = [model.h1, model.h2, model.h3]
    # 2. aan elke layer een forward hook die zijn output bewaart, de z's
    _zs = []

    def _keep(module, inputs, output):
        _zs.append(output)

    _handles = []
    for _layer in _layers:
        _handles.append(_layer.register_forward_hook(_keep))
    # 3. één run zonder gradients, dan de hooks er weer af
    with torch.no_grad():
        model(X_test)
    for _handle in _handles:
        _handle.remove()
    # 4. per layer: het deel van de neuronen waarvan de z bij elk testcijfer 0 of lager is
    for _z in _zs:
        print((_z <= 0).all(dim=0).float().mean().item())
    # 5. één backward pass op de testcijfers
    _test_loss = loss_fn(model(X_test), y_test)
    model.zero_grad()
    _test_loss.backward()
    print(_test_loss.item())
    # daarna: print voor elke layer in _layers de gemiddelde grootte van de gradient van zijn weights,
    # _layer.weight.grad.abs().mean().item(), en voer opnieuw uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met een learning rate van 2,0 heeft de laatste mini-batch een loss van 2,29, en de testcijfers 2,30: de loss van gokken. Binnenin is meer dan de helft van de neuronen van de eerste hidden layer dead, het grootste deel van de tweede, en van de derde allemaal. De course mat 59% en 80%; jouw computer kan voor de eerste twee iets meer of minder printen, omdat bij een learning rate die zo groot is het kleinste afrondingsverschil groeit. Welk cijfer er ook binnenkomt, de derde layer geeft alleen nullen. Stap 5 laat zien wat dat betekent voor het trainen: de gradient van de weights van elke hidden layer is precies 0. Geen stap verschuift ze nog, en meer epochs helpen niet. Alleen de bias terms van de output-layer leren nog, en die kunnen alleen leren hoe vaak elk cijfer voorkomt.

    Met 0,3 heeft de laatste mini-batch een loss van 0,79, en de testcijfers 0,83. Geen neuron van de eerste layer is dead, 1 van de tweede en 2 van de derde (de cel print hun delen, `0.0078125` en `0.015625`), en de gradients zijn ongeveer 0,001 tot 0,002 in elke layer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dead():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een dead neuron traint verder
    """)
    return


@app.cell(hide_code=True)
def choice_dead_cell():
    choice_dead = L.choice(
        "what-dead-means",
        question="Een neuron in de eerste hidden layer van een MLP geeft 0 voor elk trainingsvoorbeeld. Je traint verder met SGD. Wat gebeurt er met zijn weights?",
        options={
            "grow": "Ze groeien: de loss duwt het neuron stap voor stap weer omhoog, boven 0",
            "right": "Niets: elke gradient erdoorheen is 0, dus geen stap van SGD verschuift ze",
            "shrink": "Ze krimpen langzaam naar 0, zoals weights die het model niet nodig heeft",
        },
        correct="right",
        why={
            "grow": "De loss kan alleen duwen via de helling van ReLU, en onder 0 is die helling 0: er komt geen gradient aan bij de weights van dit neuron.",
            "shrink": "Krimpen kost ook stappen, en elke stap van SGD is de learning rate keer de gradient: hier 0.",
            "right": "Onder 0 is de helling van ReLU 0, dus bij elk voorbeeld is de gradient van de weights en de bias term van het neuron 0. In de eerste layer veranderen zijn inputs, de pixels, ook nooit: het blijft voorgoed dead. In een latere layer kan een verandering in de layers ervoor een neuron nog terugbrengen.",
        },
    )
    choice_dead
    return (choice_dead,)


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het deel dead neurons

        Schrijf `dead_share(Z)`: `Z` bevat de $z$'s van één layer, één rij per voorbeeld en één kolom per neuron, zoals een hook ze krijgt. Geef het deel dead neurons terug, als getal: het deel van de kolommen waarin elke $z$ 0 of lager is. De check roept `dead_share(torch.tensor([[1.0, -2.0, 0.0], [-1.0, -0.5, 2.0]]))` aan, en dat moet `0.3333` geven, omdat alleen het middelste neuron dead is; daarna probeert hij andere layers.
        """),
        L.task("Laat `dead_share` het deel van de neuronen geven waarvan de z bij elk voorbeeld 0 of lager is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("dead-share", lines=5)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("dead-share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "dead-share",
        [
            "`Z <= 0` geeft een tensor van `True` en `False` met de vorm van `Z`. `.all(dim=0)` vraagt per kolom of alles erin `True` is.",
            "`.float()` maakt van `True` 1.0 en van `False` 0.0, en `.mean()` geeft dan het deel `True`, zoals bij de accuracy in [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl). `.item()` maakt er een gewoon getal van.",
        ],
        why="`return (Z <= 0).all(dim=0).float().mean().item()`. In het voorbeeld zijn de kolommen $[1, -1]$, $[-2, -0{,}5]$ en $[0, 2]$; alleen de middelste is overal 0 of lager, dus 1 van de 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_layers():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Elke layer tegelijk, met hooks

        Schrijf `dead_per_layer(model, layers, x)`: `layers` is een lijst layers van `model`, in een willekeurige volgorde. Hang aan elke layer een forward hook, draai het model één keer op `x`, haal de hooks er weer af, en geef een lijst terug met het deel dead neurons van elke layer, in de volgorde van `layers`, met stap 2 tot en met 4 van het recept. De check maakt een klein netwerk `model` zoals dat uit het uitgewerkte voorbeeld, met twee layers, `model.h1 = nn.Linear(4, 6)` en `model.h2 = nn.Linear(6, 5)`, en ReLU ertussen, en 4 willekeurige voorbeelden. Hij roept `dead_per_layer(model, [model.h1, model.h2], x)` aan, en dat moet `[0.1667, 0.0]` geven. Daarna geeft hij de layers in de andere volgorde, `[model.h2, model.h1]`, en alleen `[model.h2]`.
        """),
        L.task("Laat `dead_per_layer` het deel dead neurons van elke layer geven, gemeten met forward hooks, in de volgorde van `layers`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_layers():
    box_layers = L.code_box("dead-per-layer", lines=18)
    box_layers
    return (box_layers,)


@app.cell(hide_code=True)
def check_layers(box_layers):
    result_layers = L.check("dead-per-layer", box_layers.value)
    result_layers
    return (result_layers,)


@app.cell(hide_code=True)
def hints_layers():
    L.hints(
        "dead-per-layer",
        [
            "De hooks draaien in de volgorde waarin het model zijn layers draait, niet in de volgorde van `layers`. Bewaar elke output dus onder zijn eigen layer, in een dict: in de hook `zs[module] = output`. Een layer kan een key van een dict zijn.",
            "Ga na de run door `layers` in hun volgorde, en reken het deel van elke layer uit met `zs[layer]`, zoals in oefening 2. Bewaar de handles in een lijst, en roep na de run `.remove()` aan op elke handle.",
        ],
        why="`zs = {}` met `def keep(module, inputs, output): zs[module] = output`; voor elke layer `handles.append(layer.register_forward_hook(keep))`; dan `with torch.no_grad(): model(x)`, `.remove()` op elke handle, en voor elke layer in `layers` het deel `(zs[layer] <= 0).all(dim=0).float().mean().item()` in een lijst.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Onder 0 geeft ReLU uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) 0, met een helling van 0. In [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) liet een verzadigde sigmoid bijna geen gradient door; een dead neuron laat er helemaal geen door, bij elk voorbeeld. [Initialisatie](/course/open/nn-initialization?lang=nl) geeft elke layer een gezonde start, maar een learning rate die veel te groot is, zoals in [Loss curves lezen](/course/open/torch-loss-curves?lang=nl), kan neuronen nog steeds voorgoed onder 0 duwen. De loss vertelde *dat* het netwerk vastzat; de hooks vertelden *waar*. In de volgende les, *Batch normalization*, standaardiseert een layer de $z$'s van elk neuron over de mini-batch, bij elke stap van het trainen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_dead, result_share, result_layers):
    L.wrap_up(choice_dead, result_share, result_layers)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
