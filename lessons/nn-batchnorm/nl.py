# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-batchnorm. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-batchnorm", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) en [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_standardized():
    quick_standardized = L.choice(
        "quick-standardized",
        question="Je standaardiseert een kolom getallen: min zijn gemiddelde, gedeeld door zijn standaardafwijking. Wat zijn daarna zijn gemiddelde en standaardafwijking?",
        options={
            "swapped": "Gemiddelde 1 en standaardafwijking 0",
            "right": "Gemiddelde 0 en standaardafwijking 1",
            "kept": "Gemiddelde 0, en de standaardafwijking die hij al had",
        },
        correct="right",
        why={
            "swapped": "Het gemiddelde eraf halen laat een gemiddelde van 0 over; delen door de standaardafwijking maakt die 1. Zie [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "kept": "Delen door de standaardafwijking verandert die ook: daarna is hij 1. Zie [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "right": "Het gemiddelde eraf halen geeft gemiddelde 0, en delen door de standaardafwijking geeft standaardafwijking 1. Vandaag doet een layer dat binnen in het netwerk.",
        },
    )
    quick_standardized
    return (quick_standardized,)


@app.cell(hide_code=True)
def quick_check_dead_layer():
    quick_dead_layer = L.choice(
        "quick-dead-layer",
        question="Elk neuron van één hidden layer van een MLP is dead. Wat krijgen de layers ervoor op de terugweg?",
        options={
            "same": "Dezelfde gradient als eerst: een dode layer geeft hem door",
            "right": "Een gradient van 0: er komt niets terug door de dode layer",
            "larger": "Een grotere gradient, om de dode layer goed te maken",
        },
        correct="right",
        why={
            "same": "Een dead neuron heeft bij elk voorbeeld een helling van 0, dus het vermenigvuldigt alles wat terugkomt met 0. Zie [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl).",
            "larger": "Niets maakt het goed: elke gradient door de dode layer wordt vermenigvuldigd met een helling van 0. Zie [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl).",
            "right": "Elk pad terug loopt door de dode layer, waar de helling van ReLU 0 is. De layers ervoor leren dan ook niet meer.",
        },
    )
    quick_dead_layer
    return (quick_dead_layer,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl) maakte een learning rate die veel te groot was een hele layer dood. He-beginweights uit [Initialisatie](/course/open/nn-initialization?lang=nl) geven elke layer een goede start, maar trainen verschuift de weights, en daarmee het gemiddelde en de spreiding van de $z$'s die elke layer geeft. Hoe dieper het netwerk, hoe meer layers kunnen afdrijven. Gradient descent houdt van features met gemiddelde 0 en standaardafwijking 1, zoals je zag in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl). Batch normalization geeft elke layer precies dat, bij elke stap van het trainen.
    """)
    return


@app.cell(hide_code=True)
def predict_deep():
    guess_deep = L.guess(
        "deep-with-bn",
        question="Een MLP met 10 hidden layers van 128 neuronen en He-beginweights traint één epoch op de 6.000 trainingscijfers, met een learning rate van 0,3. Het leest 16% van de 1.000 testcijfers goed, nauwelijks beter dan gokken. Nu komt er batch normalization tussen elke hidden layer en zijn ReLU; al het andere blijft hetzelfde. Hoeveel testcijfers leest het volgens jou nu goed?",
        options={
            "same": "Ongeveer evenveel: 16%",
            "some": "Wat meer: ongeveer 40%",
            "right": "De meeste: ongeveer 90%",
        },
        reveal="**Ongeveer 90%.** Zonder batch normalization blijft de loss de hele epoch op ongeveer 2,3; met batch normalization daalt hij tot ongeveer 0,43, en leest het netwerk 888 van de 1.000 testcijfers goed. Dezelfde layers, dezelfde beginweights, dezelfde learning rate: alleen het standaardiseren binnenin verschilt.",
    )
    guess_deep
    return (guess_deep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Batch normalization**, kort BatchNorm, doet met de $z$'s van een layer wat standaardiseren met features deed: per neuron haalt BatchNorm het gemiddelde eraf en deelt het door de standaardafwijking, over de voorbeelden in de mini-batch. Voor de $z$'s $z_1, \dots, z_m$ van één neuron in een mini-batch van $m$ voorbeelden:
        """),
        L.formula(r"\mu_B = \frac{1}{m} \sum_{i=1}^{m} z_i"),
        L.formula(r"\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (z_i - \mu_B)^2"),
        L.formula(r"\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + [[hyper:\varepsilon]]}} \qquad y_i = [[param:\gamma]] \, \hat{z}_i + [[param:\beta]]"),
        mo.md(r"""
        Daarna herschaalt het: $\gamma$ en $\beta$ worden geleerd, één van elk per neuron, zodat het netwerk een neuron een ander gemiddelde en een andere spreiding kan geven als dat helpt; met $\gamma = \sigma_B$ en $\beta = \mu_B$ kan het het standaardiseren zelfs ongedaan maken. Ze beginnen op 1 en 0. $\varepsilon$, meestal 0,00001, voorkomt dat je door 0 deelt. In PyTorch doet `nn.BatchNorm1d(128)` dit voor een layer van 128 neuronen; hij komt tussen de `nn.Linear` en de ReLU, en `bn.weight` is zijn $\gamma$, `bn.bias` zijn $\beta$.

        In een mini-batch hangt elk antwoord af van de andere voorbeelden erin. Na het trainen wil je één antwoord per voorbeeld, in welke batch het ook zit. Daarom houdt BatchNorm tijdens het trainen ook **voortschrijdende gemiddelden** bij van het gemiddelde en de variantie van elk neuron: na elke batch 0,9 keer het oude running average plus 0,1 keer dat van de batch zelf. Voor de variantie deelt het daar door $m - 1$ in plaats van door $m$, een correctie om de variantie van alle data te schatten uit een steekproef. In eval mode, `model.eval()`, gebruikt het die in plaats van die van de batch; `model.train()` zet het terug.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **batch normalization** | | per neuron: zijn $z$'s standaardiseren over de mini-batch, en dan keer $\gamma$ plus $\beta$ | `nn.BatchNorm1d(128)` |
        | $\mu_B$, $\sigma_B^2$ | mu B, sigma B kwadraat | het gemiddelde en de variantie van de $z$'s van een neuron over de mini-batch $B$, gedeeld door $m$ | `Z.mean(dim=0)`, `((Z - mean) ** 2).mean(dim=0)` |
        | $\hat{z}$ | z dakje | een gestandaardiseerde $z$: hier betekent het dakje gestandaardiseerd, niet geschat | `(Z - mean) / torch.sqrt(var + 1e-5)` |
        | $\gamma$, $\beta$ | gamma, beta | de geleerde schaal en bias term van elk neuron na het standaardiseren; ze beginnen op 1 en 0 | `bn.weight`, `bn.bias` |
        | $\varepsilon$ | epsilon | een klein getal dat voorkomt dat je door 0 deelt | `eps=1e-5` |
        | **voortschrijdende gemiddelden** | | het gemiddelde en de variantie per neuron die BatchNorm tijdens het trainen bijhoudt, voor eval mode | `bn.running_mean`, `bn.running_var` |
        """),
        L.watch_out("Roep `model.eval()` aan voordat je een netwerk met BatchNorm test of gebruikt, en `model.train()` voordat je het weer traint. In train mode wordt elke batch gestandaardiseerd met zijn eigen gemiddelde en variantie, en veranderen de voortschrijdende gemiddelden met alles wat je erin stopt. En houd batches groot genoeg: het gemiddelde en de variantie van 4 voorbeelden zijn ruwe schattingen. Waarom BatchNorm zo goed helpt, is nog onderwerp van discussie: de oorspronkelijke paper zei dat de inputs van elke layer tijdens het trainen blijven verschuiven; later werk wijst op een gladder loss-oppervlak. Dát het helpt, is duidelijk."),
        mo.md(r"""
        De figuur: de training loss van het netwerk uit de gok tijdens zijn ene epoch, het gemiddelde van de laatste 10 mini-batches. Zonder BatchNorm komt hij nauwelijks weg van 2,30, de loss van gokken. Met BatchNorm daalt hij tot ongeveer 0,43. Bij andere learning rates zie je hetzelfde: zonder BatchNorm traint alleen 0,1 goed: het netwerk leest dan 66% van de testcijfers goed; bij 0,3 zit het netwerk vast en bij 1,0 wordt de loss `nan`. Met BatchNorm halen 0,1, 0,3 en 1,0 allemaal 84% tot 89%.
        """),
        mo.Html(C.loss_picture({"title": "10 hidden layers, learning rate 0,3, één epoch", "x": "mini-batch", "y": "training loss",
                                "plain": "zonder BatchNorm", "normed": "met BatchNorm",
                                "alt": "Twee loss-curves over 94 mini-batches. Zonder BatchNorm, gestippeld, blijft de loss tussen ongeveer 2,1 en 2,4. Met BatchNorm daalt hij van ongeveer 2,3 naar ongeveer 0,4."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Batch normalization van een batch $z$'s, één rij per voorbeeld en één kolom per neuron:

    1. Per kolom: het gemiddelde en de variantie over de batch, gedeeld door het aantal rijen.
    2. Standaardiseer: min het gemiddelde, gedeeld door de wortel uit de variantie plus $\varepsilon$.
    3. Herschaal: keer $\gamma$, plus $\beta$.
    4. Werk in train mode de voortschrijdende gemiddelden bij: 0,9 keer de oude plus 0,1 keer het gemiddelde en de variantie van de batch, de variantie deze keer gedeeld door het aantal rijen min 1. Gebruik ze in eval mode in stap 2, in plaats van die van de batch.

    In een model gaat het Linear, dan BatchNorm, dan ReLU, met één `nn.BatchNorm1d` voor elke hidden layer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een batch van 4 voorbeelden, de $z$'s van een layer met 2 neuronen: de eerste tussen 1 en 6, de tweede tussen 40 en 70. `torch.sqrt` neemt de wortel uit elk getal in een tensor, zoals `** 0.5`, en `torch.ones(2)` en `torch.zeros(2)` maken een tensor met twee enen en een met twee nullen. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_batchnorm():
    import torch
    from torch import nn

    _Z = torch.tensor([[1.0, 50.0], [2.0, 70.0], [3.0, 40.0], [6.0, 60.0]])
    # 1. per kolom: het gemiddelde en de variantie over de batch, gedeeld door het aantal rijen
    _mean = _Z.mean(dim=0)
    _var = ((_Z - _mean) ** 2).mean(dim=0)
    print(_mean, _var)
    # 2. standaardiseer, met eps = 0.00001
    _Z_hat = (_Z - _mean) / torch.sqrt(_var + 1e-5)
    # 3. herschaal met gamma en beta, die op 1 en 0 beginnen
    _gamma = torch.ones(2)
    _beta = torch.zeros(2)
    print(_gamma * _Z_hat + _beta)
    # 4. nn.BatchNorm1d doet in train mode stap 1 tot en met 3, en werkt zijn voortschrijdende gemiddelden bij
    _bn = nn.BatchNorm1d(2)
    with torch.no_grad():
        print(_bn(_Z))
    # daarna: print ook _bn.running_mean, en vergelijk die met 0.1 * _mean; voer opnieuw uit
    # en dan: zet _bn in eval mode met _bn.eval(), print _bn(_Z) nog een keer, en voer opnieuw uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De gemiddelden zijn 3 en 55, de varianties 3,5 en 125. Na het standaardiseren hebben beide kolommen gemiddelde 0 en standaardafwijking 1: de $z$'s van het eerste neuron worden −1,07, −0,53, 0 en 1,60, die van het tweede −0,45, 1,34, −1,34 en 0,45, wat hun schaal ook was. Met $\gamma = 1$ en $\beta = 0$ verandert het herschalen nog niets. `nn.BatchNorm1d(2)` geeft dezelfde getallen. Zijn voortschrijdende gemiddelden waren aan het begin 0; na deze ene batch zijn ze $0{,}9 \cdot 0 + 0{,}1 \cdot 3 = 0{,}3$ en $0{,}1 \cdot 55 = 5{,}5$: nog ver van 3 en 55. Pas na veel batches liggen ze dicht bij het gemiddelde van alle data. In eval mode, na `_bn.eval()`, zou deze ene batch er dus verre van gestandaardiseerd uitkomen: zijn eerste kolom als 0,60, 1,45, 2,31 en 4,88.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_eval():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Testen zonder eval mode
    """)
    return


@app.cell(hide_code=True)
def choice_eval_cell():
    choice_eval = L.choice(
        "eval-mode",
        question="Een netwerk met BatchNorm is getraind. Je test het op de testcijfers in batches van 10, maar je vergeet `model.eval()`. Wat gebeurt er?",
        options={
            "nothing": "Er verandert niets: BatchNorm doet alleen iets terwijl het netwerk leert",
            "frozen": "BatchNorm gebruikt zijn voortschrijdende gemiddelden, maar werkt ze niet meer bij",
            "right": "Elk antwoord hangt dan af van de andere 9 cijfers in zijn batch",
        },
        correct="right",
        why={
            "nothing": "BatchNorm weet niet of je leert of test; het kent alleen zijn mode. In train mode standaardiseert het elke batch met het eigen gemiddelde en de eigen variantie van die batch.",
            "frozen": "Dat doet eval mode. Zonder `model.eval()` blijft BatchNorm in train mode: het gebruikt het eigen gemiddelde en de eigen variantie van elke batch, en werkt zelfs zijn voortschrijdende gemiddelden bij met je testcijfers.",
            "right": "In train mode wordt elke batch van 10 gestandaardiseerd met zijn eigen gemiddelde en variantie, dus het antwoord voor één cijfer hangt af van welke 9 andere erbij zitten, en de voortschrijdende gemiddelden veranderen ook met je testcijfers. `model.eval()` voor het testen, `model.train()` voordat je weer traint.",
        },
    )
    choice_eval
    return (choice_eval,)


@app.cell(hide_code=True)
def text_exercise_bn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Batch normalization met de hand

        Schrijf `batch_norm(Z, gamma, beta)`: batch normalization in train mode, volgens stap 1 tot en met 3 van het recept, met $\varepsilon$ = 0,00001. `Z` heeft één rij per voorbeeld en één kolom per neuron; `gamma` en `beta` hebben één getal per kolom. De check roept `batch_norm(torch.tensor([[1.0, 10.0], [2.0, 20.0], [6.0, 30.0]]), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]))` aan, en dat moet `[[-0.9258, -1.4495], [-0.4629, 1.0], [1.3887, 3.4495]]` geven; daarna probeert hij andere batches.
        """),
        L.task("Laat `batch_norm` elke kolom standaardiseren over de batch, en dan keer gamma, plus beta."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bn():
    box_bn = L.code_box("batch-norm", lines=6)
    box_bn
    return (box_bn,)


@app.cell(hide_code=True)
def check_bn(box_bn):
    result_bn = L.check("batch-norm", box_bn.value)
    result_bn
    return (result_bn,)


@app.cell(hide_code=True)
def hints_bn():
    L.hints(
        "batch-norm",
        [
            "Stap 1 zoals in het uitgewerkte voorbeeld: `Z.mean(dim=0)`, en voor de variantie het gemiddelde van de gekwadrateerde verschillen. Let op: in PyTorch delen `Z.var(dim=0)` en `Z.std(dim=0)` door één rij minder.",
            "`Z - mean` en delen door `torch.sqrt(var + 1e-5)` werken per kolom, zoals standaardiseren deed in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl). `gamma` en `beta` ook, omdat ze één getal per kolom hebben.",
        ],
        why="`mean = Z.mean(dim=0)`, `var = ((Z - mean) ** 2).mean(dim=0)`, en `return gamma * (Z - mean) / torch.sqrt(var + 1e-5) + beta`. In het voorbeeld heeft de eerste kolom gemiddelde 3 en variantie $14/3 \\approx 4{,}667$, dus 1 wordt $(1 - 3)/\\sqrt{4{,}667} \\approx -0{,}926$, en met $\\gamma = 1$ en $\\beta = 0$ blijft dat zo.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mlp():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een MLP met BatchNorm

        Schrijf een class `BNMLP(nn.Module)`: een MLP met `n_in` inputs, twee hidden layers van `n_hidden` neuronen en `n_out` logits, met een `nn.BatchNorm1d` tussen elke hidden `nn.Linear` en zijn ReLU, zoals in de laatste regel van het recept. Maak in `__init__` de layers in deze volgorde: `self.h1 = nn.Linear(n_in, n_hidden)`, `self.bn1`, `self.h2 = nn.Linear(n_hidden, n_hidden)`, `self.bn2` en `self.out = nn.Linear(n_hidden, n_out)`: dan krijgen de `nn.Linear`-layers dezelfde willekeurige weights als in de oplossing van de course. De check maakt `BNMLP(4, 6, 3)` na `torch.manual_seed(0)` en vergelijkt wat het geeft voor 8 willekeurige voorbeelden, eerst in train mode en dan in eval mode; daarna probeert hij een ander netwerk na twee batches. `.tolist()` maakt van een tensor een lijst getallen, voor de vergelijking.
        """),
        L.task("Maak van `BNMLP` een MLP met een BatchNorm1d tussen elke hidden layer en zijn ReLU."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mlp():
    box_mlp = L.code_box("bn-mlp", lines=16)
    box_mlp
    return (box_mlp,)


@app.cell(hide_code=True)
def check_mlp(box_mlp):
    result_mlp = L.check("bn-mlp", box_mlp.value)
    result_mlp
    return (result_mlp,)


@app.cell(hide_code=True)
def hints_mlp():
    L.hints(
        "bn-mlp",
        [
            "In `__init__` één `nn.BatchNorm1d(n_hidden)` voor elke hidden layer: elk houdt zijn eigen voortschrijdende gemiddelden bij en heeft zijn eigen $\\gamma$ en $\\beta$.",
            "Lees in `forward` elke hidden layer van binnen naar buiten: eerst de `nn.Linear`, dan zijn BatchNorm, dan `torch.relu`. `self.out` komt als laatste, zonder ReLU.",
        ],
        why="`__init__`: `super().__init__()`, dan `self.h1 = nn.Linear(n_in, n_hidden)`, `self.bn1 = nn.BatchNorm1d(n_hidden)`, `self.h2 = nn.Linear(n_hidden, n_hidden)`, `self.bn2 = nn.BatchNorm1d(n_hidden)` en `self.out = nn.Linear(n_hidden, n_out)`. `forward`: `h = torch.relu(self.bn1(self.h1(x)))`, `h = torch.relu(self.bn2(self.h2(h)))` en `return self.out(h)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het standaardiseren uit [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) gebeurt nu binnen in het netwerk, voor elke layer en elke mini-batch. [Initialisatie](/course/open/nn-initialization?lang=nl) geeft een goede start; BatchNorm houdt die goed tijdens het trainen. In elke batch heeft wat BatchNorm voor een neuron geeft gemiddelde $\beta$ en standaardafwijking $\gamma$, dus zolang $\beta$ 0 of meer is, kan dat niet bij elk voorbeeld 0 of lager zijn: de dode layer uit [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl) wordt veel minder waarschijnlijk. In de volgende les, *Residual connections*, telt elk block zijn input op bij zijn output, zodat de gradient een directe weg terug krijgt door een diep netwerk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_eval, result_bn, result_mlp):
    L.wrap_up(choice_eval, result_bn, result_mlp)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
