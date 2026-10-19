# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-eval-mode. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-eval-mode", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_steps():
    quick_steps = L.choice(
        "quick-steps",
        question="`loader = DataLoader(ds, batch_size=50, shuffle=True)`, en `ds` bevat 200 voorbeelden. Hoeveel update-stappen tellen 3 epochs, met een stap voor elke mini-batch?",
        options={
            "epochs": "3: één per epoch",
            "examples": "600: één per voorbeeld, elke epoch",
            "right": "12: één per mini-batch, elke epoch",
        },
        correct="right",
        why={
            "epochs": "Elke epoch gaat langs alle mini-batches van de DataLoader, met voor elke mini-batch een stap. Zie [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).",
            "examples": "Een stap neemt een hele mini-batch van 50 tegelijk. Zie [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).",
            "right": "200 / 50 = 4 mini-batches per epoch, dus 3 epochs tellen 4 · 3 = 12 stappen.",
        },
    )
    quick_steps
    return (quick_steps,)


@app.cell(hide_code=True)
def quick_check_item():
    quick_item = L.choice(
        "quick-item",
        question="`ds` is een Dataset, zoals in [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl). Wat geeft `ds[3]`?",
        options={
            "right": "Voorbeeld 3: zijn input en zijn label",
            "batch": "De mini-batch met nummer 3",
            "first": "De eerste 3 voorbeelden",
        },
        correct="right",
        why={
            "batch": "Mini-batches komen uit de DataLoader, niet uit de Dataset. Zie *Het idee* in [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).",
            "first": "`ds[3]` voert `__getitem__` uit met `i` = 3. Zie *Het idee* in [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).",
            "right": "`ds[3]` voert `ds.__getitem__(3)` uit, en dat geeft de input en het label van voorbeeld 3 terug.",
        },
    )
    quick_item
    return (quick_item,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe werden de netwerken in deze course vooral getraind: een loss, een backward pass en een stap. Is een model eenmaal getraind, dan gebruik je het: het voorspelt voor nieuwe inputs, en er verandert niets aan. Dan zijn twee dingen anders. Sommige layers doen met opzet iets terwijl het netwerk leert, en moeten daarmee stoppen als het voorspelt. En PyTorch neemt voor elke berekening met een parameter een graaf op, wat geheugen en tijd kost, terwijl een voorspelling geen hellingen nodig heeft. PyTorch heeft voor allebei een schakelaar.

    **Een voorbeeld uit het dagelijks leven.** Bij een repetitie maakt de regisseur het met opzet moeilijker: een paar spelers doen een scène niet mee, zodat de anderen leren die zonder hen te spelen, en een assistent schrijft elke misser op voor de bespreking achteraf. Op de avond van de voorstelling speelt iedereen mee en schrijft niemand iets op: de spelers spelen gewoon wat ze geleerd hebben.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="Een getraind netwerk heeft een layer die dropout heet; *Het idee* hieronder beschrijft die. Het staat nog in de stand waarin het trainde. Je vraagt het drie keer om zijn voorspelling bij $x = 1$. Wat denk je dat je krijgt?",
        options={
            "same": "Drie keer hetzelfde getal",
            "right": "Drie verschillende getallen",
            "error": "Een fout: een netwerk dat nog traint, kan niet voorspellen",
        },
        reveal="**Drie verschillende getallen.** In de stand waarin het traint, zet dropout elk getal van de hidden layer op goed geluk op 0, bij elke aanroep opnieuw. Dus dezelfde input geeft elke keer een andere output. In het uitgewerkte voorbeeld hieronder geven de drie aanroepen 2,62; 2,33 en 2,04. In de stand om te voorspellen laat dropout alles door, en geeft elke aanroep 2,56.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een getraind model gebruiken om voor nieuwe inputs te voorspellen is **inference**, zoals in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl). Daarvoor heeft een netwerk twee schakelaars nodig, en die doen verschillende dingen.

        **Train mode en eval mode.** Elke module, van één `nn.Linear` tot een class die voortbouwt op `nn.Module`, heeft een schakelaar: `model.training`. Die begint op `True`: **train mode**. `model.eval()` zet hem op `False`, **eval mode**, voor het model en elke layer erin, en `model.train()` zet hem terug. De meeste layers trekken zich niets van de schakelaar aan: `nn.Linear` en ReLU rekenen in beide standen hetzelfde. Een paar layers lezen hem wel. **Dropout**, `nn.Dropout(0.5)`, is er één: in train mode zet het elk getal dat erdoorheen gaat met een kans van 0,5 op 0, en verdubbelt het de andere; in eval mode laat het alles ongewijzigd door. Een latere les legt uit waarom zulk toeval een netwerk helpt leren. Bij een voorspelling zit het alleen in de weg.

        **Geen graaf: `torch.no_grad()`.** Zoals in [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) neemt PyTorch binnen `with torch.no_grad():` niets op; daar hield het de update-stap buiten de graaf, en nu doet `opt.step()` dat voor je. Een voorspelling heeft ook geen hellingen nodig, dus kan haar forward pass de graaf ook overslaan, en het geheugen en de tijd besparen die het opnemen kost. De outputs zeggen dan `requires_grad=False`.

        De twee schakelaars staan los van elkaar: `model.eval()` stopt het opnemen niet, en `torch.no_grad()` zet dropout niet uit. Een voorspelling gebruikt ze allebei. En een netwerk dat tussen trainingsstappen door voorspelt, moet vóór de volgende stap terug naar train mode, met `model.train()`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **train mode** | | de stand waarin een netwerk traint: layers zoals dropout doen hun werk | `model.train()` |
        | **eval mode** | | de stand waarin een netwerk voorspelt: layers zoals dropout laten alles door | `model.eval()` |
        | `model.training` | | de schakelaar zelf: `True` in train mode, `False` in eval mode | `net.training` |
        | `nn.Dropout(p)` | n n dropout | een layer die in train mode elk getal met kans `p` op 0 zet; uitgelegd in een latere les | `nn.Dropout(0.5)` |
        """),
        L.watch_out("In de statistiek betekent inference iets anders: conclusies trekken over een hele groep uit een steekproef ervan, statistische inferentie. In machine learning is inference gewoon een getraind model gebruiken om te voorspellen. Lees je het woord in een tekst over statistiek, dan gaat het meestal om statistische inferentie."),
        mo.md(r"""
        De figuur: het netwerk uit het uitgewerkte voorbeeld, met dropout, getraind op de 200 voorbeelden uit [Mini-batches en epochs](/course/open/torch-batches?lang=nl). Zijn voorspellingen voor $x$ van −2 tot 2: drie keer in train mode, en één keer in eval mode.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_modes():
    _train_loss, _eval_loss = C.trained()[3], C.trained()[4]
    mo.vstack([
        mo.Html(C.picture({"title": "Eén netwerk, twee modes", "dashed": "streepjes: train mode, drie aanroepen", "solid": "doorgetrokken: eval mode",
                           "alt": f"200 punten rond de lijn y = 2x + 1. Drie streepjeslijnen, de voorspellingen van het netwerk in train mode, zigzaggen elk anders rond de punten; één doorgetrokken lijn, zijn voorspelling in eval mode, loopt er rustig doorheen. De loss over alle punten is ongeveer {L.number(_train_loss, 1)} in train mode, bij elke aanroep een beetje anders, en {L.number(_eval_loss, 2)} in eval mode."})),
        mo.md(f"""
        In train mode geeft elke aanroep een andere lijn, en is de loss over alle punten ongeveer <span class="c-loss ea-num">{L.number(_train_loss, 1)}</span>, bij elke aanroep een beetje anders. In eval mode is de lijn bij elke aanroep dezelfde, en is de loss ongeveer <span class="c-loss ea-num">{L.number(_eval_loss, 2)}</span>. Het netwerk is hetzelfde; alleen de schakelaar staat anders.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om met een netwerk te voorspellen, ook tussen trainingsstappen door:

    1. Zet het in eval mode: `model.eval()`.
    2. Reken de voorspellingen uit binnen `with torch.no_grad():`.
    3. Zet het terug in train mode, `model.train()`, voordat je verder traint.
    4. Zet stap 1 tot en met 3 in een functie `predict(model, X)`, zodat je er nooit een vergeet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het netwerk uit de figuur: 1 input, een hidden layer van 16 neuronen met ReLU, dropout, en 1 output, 15 epochs getraind met mini-batches van 20. Dan drie aanroepen in train mode, die elk voorspellen bij $x = 0$ en $x = 1$, en twee aanroepen met het recept. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_eval():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(1, 16)
            self.drop = nn.Dropout(0.5)
            self.out = nn.Linear(16, 1)

        def forward(self, x):
            return self.out(self.drop(torch.relu(self.hidden(x))))

    _X, _y = C.data()   # de 200 voorbeelden uit Mini-batches en epochs
    torch.manual_seed(1)
    _net = Net()
    _opt = torch.optim.SGD(_net.parameters(), lr=0.05)
    _loader = DataLoader(TensorDataset(_X, _y), batch_size=20, shuffle=True)
    for _epoch in range(15):
        for _xb, _yb in _loader:
            _loss = ((_net(_xb) - _yb) ** 2).mean()
            _opt.zero_grad()
            _loss.backward()
            _opt.step()

    _points = torch.tensor([[0.0], [1.0]])   # x = 0 en x = 1
    print(_net.training)
    for _k in range(3):
        _out = _net(_points)   # nog in train mode: elke keer een ander antwoord
        print(_out[0].item(), _out[1].item(), _out.requires_grad)

    # 4. het recept als functie
    def _predict(model, X):
        model.eval()               # 1. eval mode
        with torch.no_grad():      # 2. geen graaf
            out = model(X)
        model.train()              # 3. terug naar train mode
        return out

    for _k in range(2):
        _out = _predict(_net, _points)
        print(_out[0].item(), _out[1].item(), _out.requires_grad)
    print(_net.training)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Na het trainen is `_net.training` `True`: train mode, zoals elk netwerk begint. De drie aanroepen in train mode geven drie verschillende paren, ongeveer (0,99; 2,62), (1,02; 2,33) en (0,77; 2,04), en `requires_grad` is `True`: PyTorch nam voor elk een graaf op. `_predict` geeft twee keer ongeveer (0,87; 2,56), beide keren hetzelfde, en `requires_grad` is `False`. Na `_predict` is `_net.training` weer `True`. De voorspellingen liggen iets onder de lijn $y = 2x + 1$, die 1 en 3 geeft. Dat komt niet door te weinig epochs: een netwerk dat met dropout leert, komt uit op een lijn die iets vlakker loopt dan de data, zoals de doorgetrokken lijn in de figuur laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_switches():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee schakelaars
    """)
    return


@app.cell(hide_code=True)
def choice_switches_cell():
    choice_switches = L.choice(
        "two-switches",
        question="Wat doet `model.eval()`, en wat doet `torch.no_grad()`?",
        options={
            "swapped": "`model.eval()` stopt het opnemen; `torch.no_grad()` zet dropout uit",
            "right": "`model.eval()` zet dropout uit; `torch.no_grad()` stopt het opnemen",
            "same": "Allebei hetzelfde: ze zetten een netwerk om van leren naar voorspellen",
        },
        correct="right",
        why={
            "same": "Het zijn twee verschillende schakelaars, en een voorspelling heeft ze allebei nodig. Zie *Het idee*.",
            "swapped": "Het is andersom. Zie *Train mode en eval mode* in *Het idee*.",
            "right": "`model.eval()` zet `model.training` op `False`, en dat leest dropout; `torch.no_grad()` zorgt dat PyTorch geen graaf opneemt. Geen van beide doet het werk van de ander.",
        },
    )
    choice_switches
    return (choice_switches,)


@app.cell(hide_code=True)
def text_exercise_evaluate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Meet de loss in eval mode

        Om te zien hoe goed een netwerk het doet tussen trainingsstappen door, meet je zijn loss terwijl het voorspelt. De class `Net` uit het uitgewerkte voorbeeld staat er al. Maak `evaluate(model, X, y)` af: stap 1 tot en met 3 van het recept, met in stap 2 de mean squared error van de voorspellingen voor `X`, en geef die loss terug. Vervang de `...`. De laatste regels maken een netwerk en roepen `evaluate` aan op drie punten van de lijn $y = 2x + 1$; daarna laten ze zien of er een graaf is opgenomen, of het netwerk weer in train mode staat, en de loss.
        """),
        L.task("Laat `evaluate` de loss van het netwerk in eval mode teruggeven, uitgerekend zonder graaf, en zet het netwerk terug in train mode."),
    ])
    return


@app.cell(hide_code=True)
def exercise_evaluate():
    box_evaluate = L.code_box("evaluate", lines=24)
    box_evaluate
    return (box_evaluate,)


@app.cell(hide_code=True)
def check_evaluate(box_evaluate):
    result_evaluate = L.check("evaluate", box_evaluate.value)
    result_evaluate
    return (result_evaluate,)


@app.cell(hide_code=True)
def hints_evaluate():
    L.hints(
        "evaluate",
        [
            "Stap 1 tot en met 3 van het recept, zoals in `_predict` in het uitgewerkte voorbeeld; alleen wat je binnen het `with`-blok uitrekent, is anders.",
            "Binnen het `with`-blok: het gemiddelde van de gekwadrateerde verschillen tussen `model(X)` en `y`. Bewaar dat onder een naam, zet het netwerk terug, en geef het pas dan terug: een regel na `return` wordt nooit uitgevoerd.",
            "`model.___()`, dan `with torch.no_grad():` met `loss = ((model(X) - y) ** 2).mean()`, dan `model.___()` en `return loss`.",
        ],
        why="`model.eval()` zet dropout uit, dus de loss is die van het netwerk zoals het voorspelt; `torch.no_grad()` neemt geen graaf op; en met `model.train()` staat het netwerk klaar om verder te trainen. Dus laat de laatste regel `False`, `True` en de loss in eval mode zien.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Oefening 3. Trainen of voorspellen?

    Zes regels uit code die traint en code die voorspelt. Waarvoor dient elke regel: alleen voor het trainen, alleen voor het voorspellen, of voor allebei?
    """)
    return


@app.cell(hide_code=True)
def exercise_when():
    sort_when = L.sort("when")
    sort_when
    return (sort_when,)


@app.cell(hide_code=True)
def check_when(sort_when):
    result_when = L.check("when", sort_when)
    result_when
    return (result_when,)


@app.cell(hide_code=True)
def hints_when():
    L.hints(
        "when",
        [
            "Vraag bij elke regel: is hij nodig om te leren, om een voorspelling stabiel en goedkoop te maken, of bij elk gebruik van het netwerk?",
            "Hellingen zijn alleen voor het leren. En elke schakelaar uit *Het idee* heeft een stand voor het trainen en een voor het voorspellen.",
            "`opt.zero_grad()` en `loss.backward()` horen bij het trainen; `model.eval()` en `torch.no_grad()` bij het voorspellen. Bij welke kant hoort `model.train()`, en welke regel hebben ze allebei nodig?",
        ],
        why="**Alleen voor het trainen:** `model.train()`, `opt.zero_grad()` en `loss.backward()`. **Alleen voor het voorspellen:** `model.eval()`, en `with torch.no_grad():` om de forward pass heen. **Voor allebei:** de forward pass `y_pred = model(X)`. `model.train()` staat ook aan het eind van `evaluate`, maar daar is hij voor het trainen dat erna komt.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De training loop uit [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl) heeft nu een tegenhanger om het netwerk te gebruiken: eval mode voor layers zoals dropout, en `torch.no_grad()` uit [PyTorch-autograd doet wat jouw Value deed](/course/open/torch-autograd?lang=nl) voor de graaf. In de volgende les, *Project: een MLP dat handgeschreven cijfers leest*, zet je alles samen: een module, een optimizer, een DataLoader, en voorspellingen in eval mode om te zien hoeveel cijfers het goed leest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_switches, result_evaluate, result_when):
    L.wrap_up(choice_switches, result_evaluate, result_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
