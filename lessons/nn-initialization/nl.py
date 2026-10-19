# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-initialization. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-initialization", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Veelgebruikte verdelingen](/course/open/prob-distributions?lang=nl) en [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_randn():
    quick_randn = L.choice(
        "quick-randn",
        question="`random.gauss(0, 0.1)` trekt één getal. Uit welke verdeling?",
        options={
            "right": "De normale verdeling met gemiddelde 0 en standaardafwijking 0,1: $\\mathcal{N}(0, 0{,}1^2)$",
            "variance": "De normale verdeling met gemiddelde 0 en variantie 0,1: $\\mathcal{N}(0, 0{,}1)$",
            "uniform": "Elk getal van −0,1 tot 0,1 even waarschijnlijk: een uniforme spreiding",
        },
        correct="right",
        why={
            "variance": "`random.gauss` neemt de standaardafwijking, niet de variantie; de variantie is dan $0{,}1^2 = 0{,}01$. Zie [Veelgebruikte verdelingen](/course/open/prob-distributions?lang=nl).",
            "uniform": "`gauss` trekt uit de normale verdeling, de klokvorm, niet gelijkmatig. Zie [Veelgebruikte verdelingen](/course/open/prob-distributions?lang=nl).",
            "right": "`random.gauss(mu, sigma)` neemt de standaardafwijking, en formules schrijven de variantie: $\\mathcal{N}(0, 0{,}1^2)$. Zo worden beginweights ook getrokken.",
        },
    )
    quick_randn
    return (quick_randn,)


@app.cell(hide_code=True)
def quick_check_saturated():
    quick_saturated = L.choice(
        "quick-saturated",
        question="Een neuron met een sigmoid krijgt $z = 8$. Waarom leert het bijna niets?",
        options={
            "zero": "Zijn output is 0: er komt niets door",
            "right": "Hij is verzadigd: zijn helling is daar bijna 0",
            "big": "Zijn gradient is te groot: hij explodeert",
        },
        correct="right",
        why={
            "zero": "Bij $z = 8$ geeft de sigmoid bijna 1, niet 0. Op de terugweg telt zijn helling. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "big": "Ver van 0 is de sigmoid vlak: zijn helling is piepklein, niet groot. Zie [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl).",
            "right": "$\\text{sigmoid}(8) \\approx 0{,}9997$, en de helling $\\text{sigmoid}(8)(1 - \\text{sigmoid}(8)) \\approx 0{,}0003$: alles wat terugkomt wordt daarmee vermenigvuldigd.",
        },
    )
    quick_saturated
    return (quick_saturated,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) besliste de spreiding van de beginweights of de gradient van een diep netwerk verdween, explodeerde of even groot bleef. Elk netwerk begint ergens: `nn.Linear` trekt kleine willekeurige getallen voor je. Welke spreiding goed is, hangt af van hoeveel inputs een neuron heeft en van zijn activation function. Met de verkeerde leert een diep netwerk nauwelijks iets, hoe lang je ook traint; met de goede begint het op een plek van waaruit gradient descent ergens heen kan.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "all-equal",
        question="Elke weight van een MLP met een hidden layer van 64 neuronen begint op hetzelfde getal, 0,1, en elke bias term op 0. Je traint het lang. Hoeveel van de 64 hidden neuronen gaan er volgens jou uiteindelijk iets anders doen dan de rest?",
        options={
            "all": "Alle 64: training trekt ze uit elkaar",
            "half": "Ongeveer de helft: ReLU zet er een deel van uit",
            "right": "Geen enkele: ze blijven kopieën van elkaar",
        },
        reveal="**Geen enkele.** Elk hidden neuron krijgt dezelfde inputs en heeft dezelfde weights, en geeft dus hetzelfde. Op de terugweg krijgt elk neuron dezelfde gradient, omdat de weights erna ook gelijk zijn, en dus dezelfde stap. Na de stap zijn ze weer gelijk, en zo na elke stap: 64 kopieën van één neuron. Willekeurige beginweights doorbreken die symmetrie.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Beginweights zijn willekeurig, zodat de neuronen van een layer verschillend beginnen en verschillend blijven. Maar hoe groot moet de spreiding zijn? Een neuron telt $n_\text{in}$ inputs op, elk keer een weight: $z = \sum_j w_j x_j$. Als de weights en de inputs onafhankelijk zijn, zodat het ene niets zegt over het andere, en gemiddelde 0 hebben, worden de varianties vermenigvuldigd en opgeteld; die regel uit de kansrekening neemt deze les aan zoals hij is:
        """),
        L.formula(r"\mathrm{Var}(z) = n_\text{in} \cdot \mathrm{Var}([[param:w]]) \cdot \mathrm{Var}(x)"),
        mo.md(r"""
        Om het signaal even groot te houden, moet $\mathrm{Var}(w)$ gelijk zijn aan $1 / n_\text{in}$. ReLU zet dan ongeveer de helft van de $z$'s op 0, en dat halveert wat erdoor komt; voor ReLU wordt de variantie dus verdubbeld. Dat is **He initialization**, naar Kaiming He. **Xavier initialization**, naar Xavier Glorot, telt ook de outputs $n_\text{out}$, zodat de gradient op de terugweg ook even groot blijft. Die is bedoeld voor activation functions die rond 0 het hele signaal doorlaten, zoals tanh: een S-vormige kromme van −1 tot 1, als een sigmoid met 0 in het midden, met helling 1 bij 0. Recurrent neural networks, later in de course, gebruiken hem.
        """),
        L.formula(r"\text{He: } \sigma = \sqrt{\frac{2}{n_\text{in}}}"),
        L.formula(r"\text{Xavier: } \sigma = \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}"),
        mo.md(r"""
        De weights worden getrokken uit $\mathcal{N}(0, \sigma^2)$ met die $\sigma$, en de bias terms beginnen op 0: de willekeurige weights zorgen al voor **symmetriebreking**. In PyTorch doet `nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")` He, en `nn.init.xavier_normal_(layer.weight)` Xavier; `nn.init.zeros_(layer.bias)` zet de bias terms op 0. De `_` aan het eind van hun namen betekent: ze veranderen de tensor zelf. `nn.Linear` trekt zijn eigen beginweights gelijkmatig uit een kleiner bereik, met een standaardafwijking van ongeveer $1/\sqrt{3 n_\text{in}}$: genoeg voor een paar layers, te klein voor veel ReLU-layers.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **initialisatie** | | de beginwaarden van de weights en bias terms kiezen, vóór het trainen | `nn.init` |
        | **symmetriebreking** | | willekeurige beginweights, zodat de neuronen van een layer verschillend beginnen en verschillende dingen kunnen leren | |
        | **He initialization** | | beginweights uit $\mathcal{N}(0, \sigma^2)$ met $\sigma = \sqrt{2 / n_\text{in}}$: voor ReLU | `nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")` |
        | **Xavier initialization** | | beginweights met $\sigma = \sqrt{2 / (n_\text{in} + n_\text{out})}$: voor tanh | `nn.init.xavier_normal_(layer.weight)` |
        | $n_\text{in}$, $n_\text{out}$ | n in, n uit | het aantal inputs en het aantal outputs van een layer | `layer.in_features`, `layer.out_features` |
        | `nn.init.zeros_` | n n init zeros | zet een tensor op 0, in de tensor zelf | `nn.init.zeros_(layer.bias)` |
        | `torch.randn` | torch rand n | een tensor met willekeurige getallen uit $\mathcal{N}(0, 1)$, in de shape die je opgeeft | `torch.randn(512, 256)` |
        | **activation** | | wat een neuron of layer geeft, na zijn activation function | `h` |
        """),
        L.watch_out("He en Kaiming zijn dezelfde methode, genoemd naar Kaiming He; Xavier en Glorot ook, naar Xavier Glorot. En de spreiding geldt alleen voor het begin: training verplaatst de weights waar de gradients ze heen sturen. Er bestaan ook andere manieren, zoals orthogonal initialization; He en Xavier kom je het meest tegen. Layers die het signaal normaliseren, later in de course, maken het begin minder belangrijk."),
        mo.md(r"""
        De figuur: de standaardafwijking van de activations na elk van 20 ReLU-layers van 256 neuronen, voor 512 willekeurige inputs, met He-beginweights en met Xavier-beginweights. Met He houdt het signaal zijn grootte; met Xavier, gemaakt voor tanh, verliest het in elke ReLU-layer ongeveer 30%.
        """),
        mo.Html(C.spread_picture({"title": "De spreiding van de activations door 20 ReLU-layers", "x": "layer", "y": "standaardafwijking",
                                  "he": "He", "xavier": "Xavier",
                                  "alt": "Twee lijnen op een logaritmische schaal over 20 layers. Met He blijft de standaardafwijking tussen ongeveer 0,55 en 0,87. Met Xavier, gestippeld, daalt hij van ongeveer 0,6 naar ongeveer 0,0006."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een netwerk beginweights te geven:

    1. Kies de spreiding bij de activation function: He voor ReLU, Xavier voor tanh.
    2. Tel de inputs van elke layer, bij Xavier ook zijn outputs, en reken $\sigma$ uit.
    3. Trek de weights uit $\mathcal{N}(0, \sigma^2)$, en zet de bias terms op 0.
    4. Controleer: stuur willekeurige inputs erdoor, en kijk naar de spreiding van de activations na elke layer. Die moet ongeveer gelijk blijven.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    20 ReLU-layers van 256 inputs en 256 outputs, met He-beginweights. Het commentaar met `#` zegt welke stap van het recept elk deel is. Het laatste commentaar is voor jou: voer de cel uit zoals hij is, doe dan wat er staat, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_init():
    import torch
    from torch import nn

    torch.manual_seed(0)
    _layers = []
    for _i in range(20):
        _layers.append(nn.Linear(256, 256))
    # 1. en 2. ReLU, dus He: sigma is de wortel uit 2 / 256
    print((2 / 256) ** 0.5)
    # 3. de weights uit N(0, sigma²), de bias terms 0
    for _layer in _layers:
        nn.init.kaiming_normal_(_layer.weight, nonlinearity="relu")
        nn.init.zeros_(_layer.bias)
    print(_layers[0].weight.std().item())
    # 4. 512 willekeurige inputs door alle layers, met de spreiding na elke layer; geprint na layer 1, 5, 10 en 20
    _h = torch.randn(512, 256)
    _spreads = []
    with torch.no_grad():
        for _layer in _layers:
            _h = torch.relu(_layer(_h))
            _spreads.append(_h.std().item())
    print(_spreads[0], _spreads[4], _spreads[9], _spreads[19])
    # daarna: vervang kaiming_normal_ in stap 3 door nn.init.xavier_normal_(_layer.weight), en voer opnieuw uit
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $\sqrt{2/256} \approx 0{,}0884$, en de getrokken weights van de eerste layer hebben ongeveer die spreiding: 0,0886. De activations blijven ongeveer even groot: 0,83 na layer 1, 0,87 na layer 5, 0,69 na layer 10 en 0,62 na layer 20. Met `xavier_normal_` is de spreiding $\sqrt{2/512} \approx 0{,}0625$, en hebben de getrokken weights 0,0627: te klein voor ReLU. De activations dalen van 0,59 na layer 1 naar 0,15 na layer 5, 0,021 na layer 10 en 0,0006 na layer 20.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_he():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. He voor 800 inputs
    """)
    return


@app.cell(hide_code=True)
def choice_he_cell():
    choice_he = L.choice(
        "he-512",
        question="Een ReLU-layer heeft 800 inputs en 200 outputs. Welke standaardafwijking geeft He initialization zijn weights?",
        options={
            "one": "Ongeveer 0,035: de wortel uit 1/800",
            "right": "Ongeveer 0,05: de wortel uit 2/800",
            "xavier": "Ongeveer 0,045: de wortel uit 2/(800 + 200)",
        },
        correct="right",
        why={
            "one": "Dat houdt het signaal bij een activation function die alles doorlaat. ReLU halveert het, dus He verdubbelt de variantie.",
            "xavier": "Dat is Xavier, die ook de outputs telt; die is bedoeld voor tanh. He telt alleen de inputs.",
            "right": "$\\sqrt{2/800} = \\sqrt{1/400} = 1/20 = 0{,}05$: He telt alleen de inputs, en verdubbelt de variantie voor ReLU.",
        },
    )
    choice_he
    return (choice_he,)


@app.cell(hide_code=True)
def text_exercise_std():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De spreiding als functie

        Schrijf `init_std(n_in, n_out, activation)`: de standaardafwijking van de beginweights van een layer met `n_in` inputs en `n_out` outputs, volgens He als `activation` gelijk is aan `"relu"` en volgens Xavier als die `"tanh"` is. De check roept `init_std(200, 100, "relu")` aan, en dat moet `0.1` geven; hij probeert ook andere layers en `"tanh"`.
        """),
        L.task("Laat `init_std` de spreiding van de beginweights geven volgens He of volgens Xavier."),
    ])
    return


@app.cell(hide_code=True)
def exercise_std():
    box_std = L.code_box("init-std", lines=6)
    box_std
    return (box_std,)


@app.cell(hide_code=True)
def check_std(box_std):
    result_std = L.check("init-std", box_std.value)
    result_std
    return (result_std,)


@app.cell(hide_code=True)
def hints_std():
    L.hints(
        "init-std",
        [
            "Twee gevallen: `if activation == \"relu\":` geeft de ene spreiding terug, en anders de andere.",
            "Beide spreidingen zijn een wortel, `** 0.5`, uit 2 gedeeld door een aantal: bij He de inputs, bij Xavier de inputs plus de outputs.",
        ],
        why="He voor ReLU: $\\sqrt{2/200} = \\sqrt{0{,}01} = 0{,}1$. Xavier voor tanh zou $\\sqrt{2/300} \\approx 0{,}0816$ geven.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_init():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. He-beginweights voor een heel netwerk

        Schrijf `he_init(layers)`: hij krijgt een lijst `nn.Linear`-layers en geeft elke layer He-beginweights en bias terms van 0, met stap 3 van het recept; hij verandert de layers zelf en geeft niets terug. De check maakt twee layers, `nn.Linear(400, 100)` en `nn.Linear(100, 50)`, na `torch.manual_seed(0)`, roept `he_init` erop aan, en kijkt naar de spreiding van de weights van elke layer, ongeveer 0,071 en 0,141, en naar zijn bias terms; dan naar drie andere layers. Willekeurige weights verschillen een beetje bij een andere manier van trekken, dus de check laat een paar procent ruimte.
        """),
        L.task("Laat `he_init` elke layer He-beginweights en bias terms van 0 geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_init():
    box_init = L.code_box("he-init", lines=8)
    box_init
    return (box_init,)


@app.cell(hide_code=True)
def check_init(box_init):
    result_init = L.check("he-init", box_init.value)
    result_init
    return (result_init,)


@app.cell(hide_code=True)
def hints_init():
    L.hints(
        "he-init",
        [
            "Een loop over de layers, zoals in stap 3 van het uitgewerkte voorbeeld.",
            "`nn.init.kaiming_normal_` haalt het aantal inputs uit de shape van de weights zelf, dus dezelfde regel werkt voor elke layer.",
        ],
        why="`for layer in layers:` met `nn.init.kaiming_normal_(layer.weight, nonlinearity=\"relu\")` en `nn.init.zeros_(layer.bias)`. De eerste layer krijgt $\\sigma = \\sqrt{2/400} \\approx 0{,}071$, de tweede $\\sqrt{2/100} \\approx 0{,}141$.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De variantie uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) en de normale verdeling uit [Veelgebruikte verdelingen](/course/open/prob-distributions?lang=nl) geven de spreiding die het signaal even groot houdt; He verdubbelt die voor ReLU, dat het signaal halveert. Zo blijven de gradients uit [Vanishing en exploding gradients](/course/open/nn-vanishing-exploding?lang=nl) ook even groot, in elk geval aan het begin. In de volgende les, *In een trainingsrun kijken*, kijk je in een trainingsrun, naar de activations en gradients van elke layer, en zie je waar het misgaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_he, result_std, result_init):
    L.wrap_up(choice_he, result_std, result_init)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
