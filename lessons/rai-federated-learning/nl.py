# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-federated-learning. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-federated-learning", "nl")
    C = L.checks   # tien telefoons met QMNIST-cijfers, DigitsMLP(64), en FedAvg met de fedavg_states van de leerling


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl) en [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_taxonomy():
    quick_taxonomy = L.choice(
        "quick-taxonomy",
        question="Wat deelt de taxonomie van Solove (2006) in groepen in?",
        options={
            "right": "Activiteiten die privacyproblemen kunnen veroorzaken",
            "definition": "Definities van privacy, van de smalste tot de breedste",
            "data": "Soorten persoonsgegevens, van weinig tot heel gevoelig",
        },
        correct="right",
        why={
            "right": "Solove deelt de activiteiten die privacyproblemen kunnen veroorzaken in vier groepen en zestien soorten in: verzamelen, verwerken, verspreiden en binnendringen. Een definitie van privacy is het niet.",
            "definition": "De taxonomie van Solove definieert niets: ze deelt de activiteiten in die privacyproblemen kunnen veroorzaken. Zie *De taxonomie van Solove* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
            "data": "De taxonomie deelt in wat mensen en organisaties met informatie doen, zoals verzamelen en verspreiden, niet soorten data. Zie *De taxonomie van Solove* in *Het idee* van [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl).",
        },
    )
    quick_taxonomy
    return (quick_taxonomy,)


@app.cell(hide_code=True)
def quick_check_state():
    quick_state = L.choice(
        "quick-state",
        question="Wat geeft `model.state_dict()` van een `DigitsMLP(64)`?",
        options={
            "copy": "Een kopie van het model, code en getallen samen",
            "right": "Een dictionary met elke parameter per naam",
            "loss": "De loss na de laatste epoch",
        },
        correct="right",
        why={
            "right": "`model.state_dict()` is een dictionary met elke parameter per naam: `hidden.weight`, `hidden.bias`, `out.weight` en `out.bias`, elk de eigen tensor van het model.",
            "copy": "Een state_dict bevat alleen namen en tensors; de code van de class blijft in je eigen bestanden. Zie *De state_dict* in *Het idee* van [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "loss": "De state_dict bevat de parameters van het model, niet wat de training mat. Zie *De state_dict* in *Het idee* van [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
        },
    )
    quick_state
    return (quick_state,)


@app.cell(hide_code=True)
def quick_check_load():
    quick_load = L.choice(
        "quick-load",
        question="`model.load_state_dict(state)`, met `state` van een ander `DigitsMLP(64)`. Wat doet dat?",
        options={
            "right": "Het kopieert die tensors naar `model`",
            "train": "Het traint `model` tot het op `state` lijkt",
            "share": "Beide modellen delen daarna één set tensors",
        },
        correct="right",
        why={
            "right": "`load_state_dict` kopieert de tensors van een state_dict naar een model van dezelfde class en grootte. Elke telefoon in deze les doet dat met de weights van de server, voordat die gaat trainen.",
            "train": "Laden traint niets: het kopieert de getallen van `state` naar `model`, naam voor naam. Zie *Opslaan en laden* in *Het idee* van [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "share": "Het kopieert de getallen naar de eigen tensors van het model: als je `model` daarna traint, veranderen de tensors in `state` niet. Zie *Opslaan en laden* in *Het idee* van [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
        },
    )
    quick_load
    return (quick_load,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het toetsenbord van je telefoon raadt het volgende woord dat je gaat typen. Om zo'n model te trainen, kan een bedrijf verzamelen wat miljoenen mensen typen en dat op de eigen servers bewaren. H. Brendan McMahan en collega's bij Google stelden een andere manier voor, in een paper die in 2017 verscheen: elke client heeft eigen trainingsvoorbeelden die nooit naar de server gaan, rekent daarmee een update uit van het gedeelde model op de server, en stuurt alleen die update. Andrew Hard en collega's (2018) trainden zo een model van Gboard, het toetsenbord van Google, dat het volgende woord voorspelt, op 1,5 miljoen telefoons. In deze les verdeel je 6000 handgeschreven cijfers over tien telefoons, laat je een server één model trainen op al die cijfers zonder er ooit één te zien, en zie je wat er verandert als elke telefoon maar een paar soorten cijfers heeft.

    **Een voorbeeld uit het dagelijks leven.** Tien bakkers willen samen één broodrecept verbeteren, zonder iemand in hun keuken toe te laten. Elk probeert het recept thuis, past de hoeveelheden aan, en stuurt alleen het aangepaste recept terug. De club middelt de hoeveelheden, waarbij een bakker die meer broden bakte zwaarder meetelt, en stuurt het nieuwe recept weer rond.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="McMahan e.a. verdeelden MNIST over 100 clients. Toen elke client van elk cijfer een beetje had, had hun methode 18 rondes nodig om een CNN, een netwerk voor beelden dat een latere les behandelt, te trainen tot het 99% van de testcijfers goed las. Hoeveel rondes waren het toen de meeste clients maar twee cijfers hadden?",
        options={
            "same": "Ook zo'n 18 rondes",
            "right": "Ongeveer 170",
            "never": "Het haalde dat nooit",
        },
        reveal="**Ongeveer 170**: 173 rondes, tegen 18 toen elke client van elk cijfer een beetje had (McMahan e.a. 2017, tabel 2, met per ronde 20 epochs in mini-batches van 10 op elke client). Meer trainen op elke client spaarde op die verdeling nog steeds rondes uit, maar veel minder dan op de geschudde; de auteurs noemen die winst kleiner, maar nog altijd flink. In de demo zie je dezelfde vertraging op tien telefoons.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### De data blijven op de telefoon

    McMahan e.a. (2017) gaven de aanpak een naam: federated learning. Een telefoon, of een ander apparaat dat op de eigen data traint, is een **client**; de **server** heeft het gedeelde model. Peter Kairouz, McMahan en 57 andere auteurs (2021) omschrijven het zo: meerdere partijen, de clients, lossen samen een probleem met machine learning op, onder regie van een centrale server of dienst, en de ruwe data van elke client blijven bij de client zelf opgeslagen, zonder dat ze worden uitgewisseld of overgedragen. Volgens hen gebruikte Apple het in iOS 13 voor het QuickType-toetsenbord en om "Hey Siri" te herkennen.

    De data van clients zijn anders dan de data van een gewone training. McMahan e.a. noemen vier verschillen: de data van elke client wijken af van de data als geheel, clients hebben heel verschillende hoeveelheden, er zijn heel veel clients, en de communicatie met ze is beperkt. En de server is er nog steeds: McMahan e.a. schrijven zelf dat je de server die de training regelt nog altijd enigszins moet vertrouwen.

    ### FedAvg: middel de weights

    McMahan e.a. noemen hun methode Federated Averaging, **FedAvg**. Die heeft drie knoppen: C, het deel van de clients dat in elke ronde traint; E, het aantal epochs dat elke client in een ronde over de eigen data maakt; en B, de grootte van de mini-batches. Een **ronde** gaat zo. De server kiest willekeurig m = max(C · K, 1) van de K clients. Elk daarvan laadt de weights w<sub>t</sub> van de server, traint E epochs in mini-batches van B op de eigen data, en stuurt de nieuwe weights w<sup>k</sup><sub>t+1</sub> terug. De server neemt hun gemiddelde, waarin elke client meetelt naar het aantal voorbeelden n<sub>k</sub>:

    $$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{m_t}\, w^k_{t+1}, \qquad m_t = \sum_{k \in S_t} n_k$$

    met S<sub>t</sub> de clients die in ronde t gekozen zijn, en m<sub>t</sub> al hun voorbeelden samen: m telt clients, m<sub>t</sub> telt voorbeelden. Een client met 600 voorbeelden telt zes keer zo zwaar als een met 100. De som gaat alleen over de gekozen clients; de auteurs merken erbij op dat eerdere versies van hun paper hier ten onrechte over alle K clients optelden.

    Waarom kun je de weights van twee modellen eigenlijk middelen? Volgens McMahan e.a. hangt dat af van waar ze beginnen: als twee modellen vanaf dezelfde willekeurige start elk apart op een ander deel van de data trainen, werkt het simpelweg middelen van de parameters verrassend goed. Vanaf verschillende willekeurige starts mislukte middelen in hun test. Bij FedAvg begint elke client elke ronde bij de weights van de server. Meer trainen op elke client voordat je middelt, spaart rondes uit: over hun modellen en datasets heen hadden ze 10 tot 100 keer minder communicatierondes nodig dan met gesynchroniseerde stochastic gradient descent.

    ### Telefoons met verschillende cijfers

    Op echte telefoons lijken de data van elke client op de eigenaar, niet op iedereen. McMahan e.a. testten dat op MNIST met 100 clients van elk 600 cijfers, met een MLP en met een CNN. Geschud kreeg elke client van elk cijfer een beetje. Gesorteerd werden de cijfers in 200 stukken van 300 geknipt, en kreeg elke client twee stukken, een verdeling die ze pathologisch noemen, omdat de meeste clients dan alleen voorbeelden van twee cijfers hebben. Met 10 clients per ronde, E = 20 en B = 10 haalde de CNN 99% goed op de testcijfers in 18 rondes op geschudde clients, en in 173 op gesorteerde. Tegenover middelen na één stap op alle data van een client maakte meer trainen per client het aantal rondes 35 keer kleiner voor de CNN en 46 keer voor hun MLP op geschudde clients, en maar 2,8 tot 3,7 keer op gesorteerde. Hoe de data verschillen, maakt wel uit: op hun data van Shakespeare, met één client per sprekende rol in een toneelstuk, had FedAvg 41 rondes nodig om het doel te halen, waar de geschudde regels er 192 nodig hadden.

    ### Wat de weights nog verraden

    Federated learning houdt de ruwe data op de telefoon, maar de weights die een telefoon stuurt, zijn van die data geleerd. McMahan e.a. waarschuwen zelf dat bij een model voor woorden dat als invoer alleen krijgt welke woorden erin voorkomen, de gradient precies laat zien welke woorden iemand typte; bij een dicht model zoals een CNN is dat moeilijker, al blijven aanvallen mogelijk. Volgens Kairouz, McMahan e.a. geeft deze basisvorm van federated learning nog altijd geen formele garantie van privacy. Ligeng Zhu, Zhijian Liu en Song Han (2019) lieten zien dat je uit gedeelde gradients de private trainingsvoorbeelden kunt terughalen: uit de gradients van één trainingsstap bouwden ze de trainingsbeelden na, tot op de pixel. Hun test had voorwaarden: netwerken met sigmoid in plaats van ReLU, willekeurige weights en batches van hoogstens 8 beelden. FedAvg stuurt weights na veel lokale stappen, een geval dat hun paper niet test. Ruis op de gradients hield de aanval pas tegen toen die het model schaadde: in hun test op CIFAR-100 zakte de accuracy van 76,3% naar 45,3%.

    Hard e.a. (2018) trainden het model van Gboard met FedAvg, in 3000 rondes van elk 100 tot 500 telefoons, en schrijven dat gebruikers erop moeten vertrouwen dat de server niet naar de weights kijkt die elke telefoon apart uploadt. Op de logs van de server voorspelde hun model het volgende woord in 16,4% van de gevallen goed, tegen 16,5% voor hetzelfde model getraind op de server; op de eigen data van de telefoons was dat 15,8% tegen 15,0%. Voor sterkere garanties wijzen McMahan e.a. naar differential privacy, secure multi-party computation, of een combinatie van die twee.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **federated learning** | één gedeeld model trainen op data die op veel apparaten blijven: elk traint op de eigen data en stuurt alleen wat het leerde naar een server |
        | **client** | een telefoon of ander apparaat bij federated learning, met eigen data |
        | **server** | de computer die het gedeelde model heeft, de clients van een ronde kiest en middelt wat ze sturen |
        | **ronde** | één beurt: de server stuurt de eigen weights, de gekozen clients trainen en sturen die van hen terug, en de server middelt ze |
        | **FedAvg** | Federated Averaging: elke ronde trainen de gekozen clients E epochs vanaf de weights van de server, en de server middelt hun weights, elk naar het aantal voorbeelden |
        | **gewogen gemiddelde** | een gemiddelde waarin elke waarde zwaarder of lichter meetelt, hier naar het aantal voorbeelden: Σ n<sub>k</sub> w<sub>k</sub> / Σ n<sub>k</sub> |
        | **C, E, B** | het deel van de clients in een ronde, de epochs van elke client in een ronde, en de grootte van de mini-batches |
        """),
        L.watch_out("Federated learning houdt de ruwe data op de telefoon, niet alle privacy: de weights die een telefoon stuurt, kunnen iets over de data verraden, en je moet de server vertrouwen dat die niet naar de weights van elke telefoon kijkt. En FedAvg middelt alleen de clients die in die ronde gekozen zijn, elk naar het aantal voorbeelden."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om met FedAvg één model te trainen op data die op de telefoons blijven:

    1. De server maakt het model en bewaart de weights, als state_dict.
    2. Elke ronde kiest de server willekeurig een paar telefoons. Elke telefoon laadt de weights van de server met `load_state_dict`, traint op de eigen data, en stuurt alleen de nieuwe state_dict en het aantal voorbeelden terug.
    3. Voor elke naam in de state_dict middelt de server de tensors van de telefoons, elk naar het aantal voorbeelden: `fedavg` doet dat voor één getal, `fedavg_rows` voor elke kolom van een array tegelijk, en `fedavg_states` voor een hele state_dict.
    4. Het gemiddelde wordt de weights van de server, en de volgende ronde begint daar.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel verdeelt de 6000 trainingscijfers van QMNIST in willekeurige volgorde over 10 telefoons van elk 600, zodat elke telefoon van elk cijfer een beetje heeft. `train_on_phone(state, k)` is het deel van telefoon k in een ronde: de functie laadt de weights van de server, traint één epoch over de eigen 600 cijfers in mini-batches van 10, en geeft de nieuwe weights terug. Twee telefoons trainen vanaf dezelfde weights van de server, en de server neemt hun gemiddelde. De cel meet elk model op de 1000 testcijfers. Dat duurt een seconde.
    """)
    return


@app.cell
def example_round():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # 1. de 6000 trainingscijfers, verdeeld over 10 telefoons van elk 600
    data = np.load("data/qmnist-small.npz")
    X = torch.tensor(data["train_images"], dtype=torch.float32) / 255
    y = torch.tensor(data["train_labels"], dtype=torch.int64)
    X_test = torch.tensor(data["test_images"], dtype=torch.float32) / 255
    y_test = torch.tensor(data["test_labels"], dtype=torch.int64)
    phones = np.random.default_rng(0).permutation(6000).reshape(10, 600)


    class DigitsMLP(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Linear(784, hidden)
            self.out = nn.Linear(hidden, 10)

        def forward(self, x):
            return self.out(torch.relu(self.hidden(x)))


    def train_on_phone(state, k):
        # 2. op telefoon k: begin bij de weights van de server, één epoch over de eigen 600 cijfers
        model = DigitsMLP(64)
        model.load_state_dict(state)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X[phones[k]], y[phones[k]]), batch_size=10, shuffle=True)
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model.state_dict()   # 3. alleen de weights gaan naar de server


    def accuracy(state):
        model = DigitsMLP(64)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            return (model(X_test).argmax(dim=1) == y_test).float().mean().item()


    torch.manual_seed(0)
    server = DigitsMLP(64).state_dict()
    torch.manual_seed(2)
    a = train_on_phone(server, 0)
    b = train_on_phone(server, 1)
    # 4. de server middelt: beide telefoons hebben 600 cijfers, dus elk telt voor de helft
    new = {}
    for name in a:
        new[name] = (a[name] + b[name]) / 2
    print("server, before:", round(accuracy(server), 3))
    print("phone 0:", round(accuracy(a), 3))
    print("phone 1:", round(accuracy(b), 3))
    print("average:", round(accuracy(new), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Voor de training leest het model van de server 135 van de 1000 testcijfers goed: weinig meer dan gokken, dat er ongeveer 100 haalt. Na één epoch op de eigen 600 cijfers leest het model van telefoon 0 er 724 goed en dat van telefoon 1 er 772. Hun gemiddelde, een model dat niemand trainde, leest er 775 goed: iets meer dan de beste van de twee. Beide telefoons begonnen bij dezelfde weights, `server`, en juist dan werkt middelen volgens McMahan e.a. De server zag geen enkel cijfer: alleen `a` en `b`, twee state_dicts, kwamen bij de server aan. Elk komt van een nieuw model op de eigen telefoon, dus `server` blijft zoals het was: `load_state_dict` kopieert de getallen, en de state_dict van een model bestaat uit de eigen tensors van dat model, zoals in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl). Beide telefoons hebben 600 cijfers, dus elk telt voor de helft. In de oefeningen schrijf je het gemiddelde voor telefoons van elke grootte: eerst voor één getal, dan voor veel getallen tegelijk, dan voor hele state_dicts.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fedavg():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. fedavg: één weight

        Drie telefoons sturen elk één weight van hetzelfde model: 0,2, 0,8 en 0,6. Ze trainden op 100, 300 en 600 voorbeelden.

        `fedavg(weights, sizes)` krijgt twee lijsten van dezelfde lengte: `weights`, één weight van elke telefoon, en `sizes`, het aantal voorbeelden van elke telefoon. De functie geeft het gewogen gemiddelde terug: elke weight keer het aantal voorbeelden van de telefoon, opgeteld, en gedeeld door het aantal voorbeelden van alle telefoons samen. De eerste regels zetten `total` en `examples` op 0. Vervang de eerste `...` door een loop over de telefoons, en de tweede door het resultaat. De check roept `fedavg([0.2, 0.8, 0.6], [100, 300, 600])` aan, daarna andere telefoons.
        """),
        L.task("Geef het gemiddelde van weights terug, waarin elke weight meetelt naar het aantal voorbeelden in sizes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fedavg():
    box_fedavg = L.code_box("fedavg", lines=8)
    box_fedavg
    return (box_fedavg,)


@app.cell(hide_code=True)
def check_fedavg(box_fedavg):
    result_fedavg = L.check("fedavg", box_fedavg.value)
    result_fedavg
    return (result_fedavg,)


@app.cell(hide_code=True)
def hints_fedavg():
    L.hints(
        "fedavg",
        [
            "Stap 3 van het recept, voor één getal: de formule in *FedAvg: middel de weights*.",
            "Ga de telefoons langs met `for k in range(len(weights)):`, en tel in de loop twee dingen op: in `total` de weight van telefoon k, zo vaak meegeteld als telefoon k voorbeelden heeft; in `examples` het aantal voorbeelden van telefoon k. Deel na de loop het een door het ander.",
            "`        total = total + ___`",
        ],
        why="In de loop `total = total + sizes[k] * weights[k]` en `examples = examples + sizes[k]`; erna `return total / examples`. Dat geeft (100 · 0,2 + 300 · 0,8 + 600 · 0,6) / 1000 = 620 / 1000 = 0,62. Het gewone gemiddelde van de drie weights is 0,53: de telefoon met 600 voorbeelden trekt het gemiddelde naar de eigen 0,6.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fedavg_rows():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. fedavg_rows: alle weights tegelijk

        Een model heeft veel weights, en een loop over elk ervan is traag. `ROWS` bevat vier weights van elk van de drie telefoons, één rij per telefoon; de eerste kolom bevat de weights van oefening 1. `SIZES` bevat de aantallen voorbeelden van de telefoons als NumPy-array: 100, 300 en 600.

        `fedavg_rows(W, sizes)` krijgt een 2D-NumPy-array `W`, met één rij per telefoon en één kolom per weight, en een 1D-NumPy-array `sizes`, met het aantal voorbeelden van elke telefoon. De functie geeft het gewogen gemiddelde van elke kolom tegelijk terug, als 1D-array. Vervang de `...`, zonder loop. De check roept `fedavg_rows(ROWS, SIZES)` aan, daarna andere arrays.
        """),
        L.task("Geef het gewogen gemiddelde van elke kolom van W terug, waarin elke rij meetelt naar het aantal voorbeelden in sizes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fedavg_rows():
    box_fedavg_rows = L.code_box("fedavg-rows", lines=6)
    box_fedavg_rows
    return (box_fedavg_rows,)


@app.cell(hide_code=True)
def check_fedavg_rows(box_fedavg_rows):
    result_fedavg_rows = L.check("fedavg-rows", box_fedavg_rows.value)
    result_fedavg_rows
    return (result_fedavg_rows,)


@app.cell(hide_code=True)
def hints_fedavg_rows():
    L.hints(
        "fedavg-rows",
        [
            "Hetzelfde gewogen gemiddelde als in oefening 1, voor elke kolom tegelijk: met het matrixproduct `@`.",
            "`v @ W`, met `v` een 1D-array van lengte 3 en `W` een array van 3 rijen, telt de rijen op, rij k keer `v[k]`: één getal voor elke kolom. Dat is de teller van de breuk uit oefening 1, voor elke kolom tegelijk. De noemer is voor elke kolom dezelfde.",
            "`    return sizes @ W / ___`",
        ],
        why="`return sizes @ W / np.sum(sizes)` geeft `[0.62, 0.5, 3.0, 0.6]`: de eerste kolom is weer oefening 1, en de derde, waar elke telefoon 3,0 stuurt, blijft 3,0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fedavg_states():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. fedavg_states: een hele state_dict

        Een telefoon stuurt de weights als state_dict: een dictionary met een tensor voor elke naam, zoals `hidden.weight`. `STATES` is een lijst van drie state_dicts van een piepklein model met twee tensors, `w` en `b`: één voor elke telefoon van oefening 1.

        `fedavg_states(states, sizes)` krijgt `states`, een lijst met één state_dict per telefoon, en `sizes`, een lijst met het aantal voorbeelden van elke telefoon. De functie geeft een nieuwe dictionary `new` terug met dezelfde namen: voor elke naam het gewogen gemiddelde van de tensors van de telefoons met die naam. De eerste regels tellen alle voorbeelden op in `examples` en beginnen `new` leeg. Vervang de `...` door een loop over de namen in `states[0]`, met daarin een loop over de telefoons. De check roept `fedavg_states(STATES, [100, 300, 600])` aan, daarna state_dicts van `nn.Linear(3, 2)` en andere telefoons.
        """),
        L.task("Geef een dictionary terug met voor elke naam het gemiddelde van de tensors van de telefoons, elk naar het aantal voorbeelden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fedavg_states():
    box_fedavg_states = L.code_box("fedavg-states", lines=12)
    box_fedavg_states
    return (box_fedavg_states,)


@app.cell(hide_code=True)
def check_fedavg_states(box_fedavg_states):
    result_fedavg_states = L.check("fedavg-states", box_fedavg_states.value)
    result_fedavg_states
    return (result_fedavg_states,)


@app.cell(hide_code=True)
def hints_fedavg_states():
    L.hints(
        "fedavg-states",
        [
            "Stap 3 van het recept: het eind van het uitgewerkte voorbeeld, voor elk aantal telefoons, elk naar het aantal voorbeelden.",
            "Begin voor elke `name` in `states[0]` een `total` bij 0 en ga de telefoons langs met `for k in range(len(states)):`, waarbij je de tensor met die naam van telefoon k optelt, zo vaak meegeteld als telefoon k voorbeelden heeft. Een getal keer een tensor is een tensor, dus `total` wordt er een. Na de loop over de telefoons is `new[name]` gelijk aan `total` gedeeld door alle voorbeelden.",
            "`            total = total + ___`",
        ],
        why="Binnen `for name in states[0]:` eerst `total = 0`, dan `for k in range(len(states)):` met `total = total + sizes[k] * states[k][name]`, en daarna `new[name] = total / examples`. Op `STATES` wordt `w` gelijk aan `[[0.62, 0.5], [3.0, 0.6]]` en `b` aan `[0.6, 1.0]`: de getallen van `ROWS` uit oefening 2, nu naam voor naam. In het plaatje lezen drie telefoons met 600, 300 en 100 cijfers 735, 555 en 406 testcijfers goed; hun gewogen gemiddelde leest er 765 goed, meer dan elk van de drie. Een gewoon gemiddelde, waarin de telefoon met 100 cijfers even zwaar telt als die met 600, leest er 723 goed.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### FedAvg met jouw fedavg_states

    De knop draait FedAvg twee keer, elk 20 rondes, met jouw `fedavg_states` als stap van de server zodra oefening 3 goed is, en tot dan die van de course. Elke ronde kiest de server willekeurig 5 van de 10 telefoons; elk daarvan traint één epoch vanaf de weights van de server, zoals `train_on_phone` in het uitgewerkte voorbeeld, en alleen de weights komen terug. De eerste run gebruikt de telefoons van het uitgewerkte voorbeeld, met van elk cijfer een beetje. De tweede sorteert de 6000 cijfers eerst op cijfer, knipt ze in 20 stukken van 300 en geeft elke telefoon willekeurig twee stukken, zoals McMahan e.a. met MNIST deden: elke telefoon heeft dan twee of drie cijfers. Tijdens het trainen laat een grafiek na elke ronde zien welk deel van de 1000 testcijfers het model van de server fout leest, en ernaast wat dat model gokt voor één testcijfer van elke soort. Het duurt een paar seconden. Daarna kun je terug door de rondes.
    """)
    return


@app.cell(hide_code=True)
def fedavg_button_cell():
    fedavg_button = mo.ui.run_button(label="Draai FedAvg op beide soorten telefoons")
    fedavg_button
    return (fedavg_button,)


@app.cell(hide_code=True)
def demo_fedavg(box_fedavg_states, fedavg_button, result_fedavg_states):
    if not fedavg_button.value:
        _out = mo.md("*Druk op de knop om de twee runs te starten.*")
    else:
        _yours = C.mine(box_fedavg_states.value, result_fedavg_states.state == "correct")
        _kind = {"mixed": "van elk cijfer een beetje", "by_digit": "elk twee of drie cijfers"}
        _now = {}

        def _progress(_name, _p):
            if _now.get("name") != _name:   # een nieuwe run: een nieuwe grafiek, met de run ervoor gestreept
                _before = list(zip(_now["live"].xs, _now["live"].ys)) if "live" in _now else []
                _now["name"] = _name
                _now["live"] = L.live({"title": f"Telefoons met {_kind[_name]}", "x": "ronde", "y": "deel van de testcijfers fout",
                                       "alt": f"Het deel van de 1000 testcijfers dat het model van de server fout leest, na elke ronde tot nu toe, op telefoons met {_kind[_name]}{'; gestreept, de run op telefoons met van elk cijfer een beetje' if _before else ''}."},
                                      C.ROUNDS, low=0.0, high=1.0)
                if _before:
                    _now["live"].reference(_before, "elk cijfer", role="loss")
            _now["live"].watch(lambda: C.now_picture(_p["said"], f"Wat het model van de server gokt na ronde {_p['round']}", L.lang))
            _now["live"].add(_p["round"], _p["wrong"], f"**Telefoons met {_kind[_name]}** · ronde {_p['round']} van {_p['rounds']} · {round(1000 * (1 - _p['wrong']))} van de 1000 testcijfers goed")

        mo.output.replace(mo.md("*Ronde 1 · begint …*"))
        _runs = C.demo(_yours or C.fedavg_states, _progress)
        if isinstance(_runs, str):
            _note = {"average": "*Je `fedavg_states` stopt met een fout op de weights van de telefoons, of geeft weights die `DigitsMLP(64)` niet kan laden. Het plaatje onder oefening 3 zegt wat er misgaat.*",
                     "slow": "*De twee runs duren langer dan drie minuten. Gaat je `fedavg_states` de namen en de telefoons maar één keer langs?*"}[_runs]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _right = {_k: [round(1000 * (1 - _w)) for _w in _runs[_k][0]] for _k in _runs}
            _m, _d = _right["mixed"], _right["by_digit"]
            _best = max(range(len(_d)), key=lambda _i: _d[_i])
            _words = {"title": "Twee runs van FedAvg", "x": "ronde", "y": "deel van de testcijfers fout",
                      "mixed": "elk cijfer", "by_digit": "twee of drie cijfers",
                      "alt": f"Twee lijnen over 20 rondes: het deel van de testcijfers dat fout gelezen wordt, zakt naar {L.number(1 - _m[-1] / 1000, 3)} bij telefoons met van elk cijfer een beetje, gestreept, en springt op en neer op weg naar {L.number(1 - _d[-1] / 1000, 3)} bij telefoons met elk twee of drie cijfers.",
                      "guess_mixed": "Elk cijfer: de gokken na ronde {r}", "guess_by_digit": "Twee of drie cijfers: de gokken na ronde {r}"}
            _who = "jouw `fedavg_states`" if _yours else "de `fedavg_states` van de course, omdat oefening 3 nog niet goed is"
            _out = mo.vstack([
                mo.md(f"""Met {_who}: na 20 rondes leest het model van de server <span class="ea-num">{_m[-1]}</span> van de 1000 testcijfers goed bij telefoons met van elk cijfer een beetje, en <span class="ea-num">{_d[-1]}</span> bij telefoons met elk twee of drie cijfers. Op de gemengde telefoons leest het er na 5 rondes al {_m[4]} goed; op de gesorteerde telefoons springt de lijn van ronde tot ronde, omdat de 5 telefoons van elke ronde andere cijfers hebben, en in de beste ronde, ronde {_best + 1}, leest het er {_d[_best]} goed."""),
                L.explore("rewind", "ronde", lambda _r: (C.rewind_frame(_runs, int(_r), _words, L.lang),
                                                         f"Na ronde {int(_r)}: {_m[int(_r) - 1]} testcijfers goed met van elk cijfer een beetje op elke telefoon, {_d[int(_r) - 1]} met twee of drie cijfers per telefoon."),
                          start=C.ROUNDS, values=list(range(1, C.ROUNDS + 1)), role="",
                          task="Ga terug naar ronde 1, en dan weer vooruit: welke cijfers leest het model van de server bij de gesorteerde telefoons het eerst goed?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_two_digits():
    mo.md(r"""
    ### Oefening 4. Telefoons die verschillen

    Een toetsenbord leert met FedAvg van telefoons in veel landen. In elk land typen mensen vooral in hun eigen taal.
    """)
    return


@app.cell(hide_code=True)
def choice_two_digits_cell():
    choice_two_digits = L.choice(
        "two-digits",
        question="Wat doet de demo vermoeden over FedAvg op deze telefoons?",
        options={
            "same": "Even snel: het gemiddelde vlakt de landen uit",
            "right": "Langzamer dan op telefoons met van alles wat",
            "never": "Het leert niets: geen telefoon heeft alle talen",
        },
        correct="right",
        why={
            "right": "In de demo gaven telefoons met elk twee of drie cijfers een model op de server dat langzamer leerde en van ronde tot ronde sprong: in de run van de course 678 testcijfers goed na 20 rondes, tegen 904 met van elk cijfer een beetje op elke telefoon. McMahan e.a. zagen hetzelfde op MNIST: 173 rondes tot 99% in plaats van 18. Niet elk verschil maakt FedAvg trager: op hun data van Shakespeare, verdeeld per sprekende rol, waren er minder rondes nodig dan op geschudde regels.",
            "same": "In de demo leerde het gemiddelde van telefoons die elk een paar cijfers hebben langzamer dan met gemengde telefoons: in de run van de course 678 testcijfers goed na 20 rondes, tegen 904. Zie *Telefoons met verschillende cijfers* in *Het idee*.",
            "never": "Het leert wel, alleen langzamer: in de run van de course haalden de gesorteerde telefoons in hun beste ronde 804 testcijfers goed, en McMahan e.a. haalden 99% op gesorteerd MNIST in 173 rondes. Zie *Telefoons met verschillende cijfers* in *Het idee*.",
        },
    )
    choice_two_digits
    return (choice_two_digits,)


@app.cell(hide_code=True)
def text_exercise_server_learns():
    mo.md(r"""
    ### Oefening 5. Wat de server ziet

    In de demo kreeg de server geen enkel cijfer: alleen state_dicts.
    """)
    return


@app.cell(hide_code=True)
def choice_server_learns_cell():
    choice_server_learns = L.choice(
        "server-learns",
        question="Kan de server toch iets te weten komen over de cijfers van een telefoon?",
        options={
            "no": "Nee: de cijfers zelf zijn op de telefoons gebleven",
            "right": "Ja: de weights zijn van die cijfers geleerd",
            "labels": "Alleen als er ook labels meegaan",
        },
        correct="right",
        why={
            "right": "De weights zijn geleerd van de cijfers op de telefoon, dus ze dragen er iets van in zich. Zhu, Liu en Han bouwden trainingsbeelden na uit de gradients van één stap, en volgens Kairouz, McMahan e.a. geeft de basisvorm geen formele garantie van privacy. Hard e.a. vragen gebruikers erop te vertrouwen dat de server niet naar de weights van elke telefoon kijkt.",
            "no": "De cijfers bleven, maar de weights die een telefoon stuurt, zijn ervan geleerd: Zhu, Liu en Han bouwden trainingsbeelden na uit gradients. Zie *Wat de weights nog verraden* in *Het idee*.",
            "labels": "Wat een telefoon stuurt, kan genoeg zijn: uit de gradients van één stap bouwden Zhu, Liu en Han trainingsbeelden en hun labels na, en niemand stuurde ze een van beide. Zie *Wat de weights nog verraden* in *Het idee*.",
        },
    )
    choice_server_learns
    return (choice_server_learns,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat is privacy?](/course/open/rai-what-is-privacy?lang=nl) was verzamelen de eerste groep van de taxonomie van Solove. Federated learning traint zonder de data te verzamelen, al zegt wat de telefoons sturen er nog steeds iets over. Het bericht van elke telefoon is een state_dict, zoals in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl), en met `load_state_dict` begint elke telefoon bij de weights van de server. McMahan e.a. noemen secure multi-party computation als weg naar sterkere garanties: in de volgende les, *Rekenen met geheimen*, tellen een paar partijen hun getallen op zonder dat iemand de getallen van een ander ziet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_server_learns, choice_two_digits, result_fedavg, result_fedavg_rows, result_fedavg_states):
    L.wrap_up(result_fedavg, result_fedavg_rows, result_fedavg_states, choice_two_digits, choice_server_learns)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
