# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les opt-momentum. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-momentum", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_valley():
    quick_valley = L.choice(
        "quick-valley",
        question="Gradient descent loopt in een lang, smal dal: steil dwars erop, flauw in de lengte. Waarom komt het maar langzaam bij de bodem?",
        options={
            "right": "De learning rate moet klein zijn voor de steile richting, en die kleine stappen brengen het in de flauwe richting nauwelijks verder",
            "far": "De bodem is ver weg, en elke stap is even groot",
            "minimum": "Een smal dal heeft geen laagste punt, dus het blijft doorgaan",
        },
        correct="right",
        why={
            "far": "De stappen zijn niet even groot: elke stap is de learning rate keer de helling, en die verschilt dwars op het dal en in de lengte ervan. Zie [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "minimum": "Het dal heeft een laagste punt; het probleem is daar komen. Zie [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "right": "Eén learning rate voor beide richtingen: klein genoeg om dwars op het dal niet te ontsporen, en daardoor veel te klein in de lengte. Standaardiseren maakte het dal rond; vandaag zie je een andere manier.",
        },
    )
    quick_valley
    return (quick_valley,)


@app.cell(hide_code=True)
def quick_check_plus_minus():
    quick_plus_minus = L.choice(
        "quick-plus-minus",
        question="Een getal is +1 of −1, elk met kans 0,5. Wat is zijn verwachtingswaarde?",
        options={
            "one": "1: de grootte van elke waarde",
            "half": "0,5: de kans op elke waarde",
            "right": "0: de twee waarden heffen elkaar op",
        },
        correct="right",
        why={
            "one": "De verwachtingswaarde telt elke waarde keer zijn kans op, met teken: $0{,}5 \\cdot 1 + 0{,}5 \\cdot (-1)$. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "half": "De kansen zijn de gewichten, niet het antwoord: $0{,}5 \\cdot 1 + 0{,}5 \\cdot (-1)$. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "$0{,}5 \\cdot 1 + 0{,}5 \\cdot (-1) = 0$. Gradients die steeds van teken wisselen, heffen elkaar zo ook grotendeels op als je ze optelt.",
        },
    )
    quick_plus_minus
    return (quick_plus_minus,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) maakte een lang, smal dal gradient descent traag, en standaardiseren van de features maakte het dal rond. Binnen in een diep netwerk kun je niet elke richting standaardiseren waarin de loss kan veranderen: de dalen blijven smal. De learning rate moet dan klein genoeg zijn voor de steilste richting, en in de flauwe richtingen zijn de stappen piepklein. Momentum laat gradient descent onthouden welke kant het op ging, en bijna elke optimizer die nu gebruikt wordt, heeft het.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "how-many-steps",
        question="Gradient descent op de loss $\\frac{1}{2}(w_1^2 + 100\\, w_2^2)$, een dal dat dwars 100 keer zo steil is als in de lengte, begint bij $(10, 1)$. Met een learning rate van 0,019, dicht bij zijn beste, heeft het 283 stappen nodig om de loss onder 0,001 te krijgen. Momentum, met de gebruikelijke instelling 0,9 en een learning rate van 0,01, ongeveer half zo groot. Hoeveel stappen heeft dat volgens jou nodig?",
        options={
            "more": "Meer dan 283: zijn stappen zijn ongeveer half zo groot",
            "same": "Ongeveer 283: momentum maakt alleen het pad gladder",
            "right": "Ongeveer 100: een derde daarvan",
        },
        reveal="**Ongeveer 100: 94 stappen.** In de lengte van het dal wijst de gradient steeds dezelfde kant op, en momentum telt hem op: zijn stappen groeien daar tot 10 keer de learning rate keer de gradient. Gewone gradient descent met dezelfde learning rate, 0,01, heeft 539 stappen nodig.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Gewone gradient descent stapt met de learning rate keer de gradient $g$ van alleen deze stap. **Momentum** houdt een som $v$ bij van de gradients tot nu toe, waarin elke gradient $\beta$ keer zo zwaar telt als die erna, en stapt daarmee:
        """),
        L.formula(r"[[grad:v]] \leftarrow [[hyper:\beta]] \, [[grad:v]] + [[grad:g]] \qquad [[param:w]] \leftarrow [[param:w]] - [[hyper:\eta]] \, [[grad:v]]"),
        mo.md(r"""
        $v$ begint op 0, en $\beta$ is meestal 0,9. Uitgeschreven is $v = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \dots$: de nieuwste gradient telt volledig mee, die ervoor 0,9 keer, die daarvoor 0,81 keer. Waar de gradient steeds dezelfde kant op wijst, zoals in de lengte van een dal, groeit $v$ tot een stap hem niet meer verandert: dan is $v = \beta v + g$, dus $v = g / (1 - \beta)$, 10 keer $g$, en worden de stappen tien keer zo groot. Waar hij steeds van teken wisselt, zoals dwars op een dal, heffen de termen elkaar grotendeels op, zoals $+1$ en $-1$ in een verwachtingswaarde: $v$ springt uiteindelijk heen en weer tussen $+g / (1 + \beta)$ en $-g / (1 + \beta)$, ongeveer de helft van $g$, want $\beta \cdot (-g / (1 + \beta)) + g = g / (1 + \beta)$. $(1 - \beta)\, v$ is een **exponentieel voortschrijdend gemiddelde** van de gradients: een gewogen gemiddelde waarin het gewicht bij elke stap terug met $\beta$ vermenigvuldigd wordt. In PyTorch doet `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` dit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **momentum** | | stappen met een som van de gradients tot nu toe, waarin elke gradient $\beta$ keer zo zwaar telt als die erna | `momentum=0.9` |
        | $\beta$ | bèta | hoeveel van $v$ elke stap bewaard blijft; meestal 0,9 | `beta` |
        | $v$ | v | de som van de gradients tot nu toe, waarin oudere minder tellen; hij begint op 0 | `v` |
        | **exponentieel voortschrijdend gemiddelde** | | een gewogen gemiddelde waarin elke stap terug met dezelfde factor minder zwaar telt | `(1 - beta) * v` |
        """),
        L.watch_out("Momentum kan doorschieten: het pad gaat voorbij het laagste punt en komt terug, zoals de figuur laat zien, en hoe groter $\\beta$, hoe langer het blijft zwaaien. In het dal uit de gok heeft momentum 707 stappen nodig met $\\beta = 0{,}99$, en 94 met 0,9. Daarom is $\\beta = 0{,}9$, en niet 0,99, de gebruikelijke keuze."),
        mo.md(r"""
        De figuur: 40 stappen in het dal uit de gok, vanaf $(10, 1)$. De <span class="c-loss">hoogtelijnen</span> verbinden de plekken met dezelfde loss, rond het laagste punt; de <span class="c-param">punten</span> zijn $(w_1, w_2)$ na elke stap. Zonder momentum, met een learning rate dicht bij zijn beste, springt elke stap naar de andere kant van het dal, en in de lengte komt $w_1$ maar van 10 tot 4,6. Met momentum zwaait het pad nog steeds heen en weer over het dal, maar trager, en die zwaaien doven langzamer uit; in de lengte komt het veel verder: het schiet zelfs voorbij het laagste punt.
        """),
        mo.Html(C.path_picture({"plain": "zonder momentum, η = 0,019", "momentum": "momentum 0,9, η = 0,01", "after": "loss na 40 stappen: {loss}",
                                "number": L.number,
                                "alt": "Twee contourplots van een smal dal met 40 stappen vanaf (10, 1). Zonder momentum zigzaggen de stappen over het dal en eindigen ze bij w1 = 4,6, loss 10,79. Met momentum zwaait het pad trager, gaat het voorbij het laagste punt en eindigt het bij (−1,4; 0,13), loss 1,73."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Gradient descent met momentum:

    1. Begin met $v = 0$, één voor elke parameter.
    2. Elke stap, na de gradient $g$: $v \leftarrow \beta v + g$.
    3. Stap: $w \leftarrow w - \eta v$.
    4. In PyTorch: `torch.optim.SGD(params, lr=..., momentum=0.9)`. Stappen op een gelijkmatige helling groeien tot hooguit $1 / (1 - \beta)$ keer zo groot, dus met momentum kan een kleinere learning rate toch sneller gaan, zoals in de gok.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het dal uit de gok, in NumPy: de gradient van $\frac{1}{2}(w_1^2 + 100\, w_2^2)$ is $(w_1, 100\, w_2)$. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit. Zoals hij is, is $v$ gewoon de gradient van deze stap: gewone gradient descent.
    """)
    return


@app.cell
def example_momentum():
    import numpy as np

    _lr = 0.01
    _beta = 0.9
    _w = np.array([10.0, 1.0])
    # 1. v begint op 0
    _v = np.zeros(2)
    _losses = []
    for _step in range(100):
        _g = np.array([_w[0], 100 * _w[1]])
        # 2. maak af: _beta keer de oude _v, plus _g
        _v = _g
        # 3. de stap
        _w = _w - _lr * _v
        _losses.append(0.5 * (_w[0] ** 2 + 100 * _w[1] ** 2))
    print(_losses[19], _losses[39], _losses[59], _losses[79], _losses[99])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, is de cel gewone gradient descent met learning rate 0,01: dwars op het dal is $100 \cdot 0{,}01 = 1$, dus één stap zet $w_2$ op 0; in de lengte haalt elke stap maar 1% van $w_1$ af. De loss is 33,4, 22,4, 15,0, 10,0 en 6,7 na 20, 40, 60, 80 en 100 stappen. Met momentum is hij 8,5, 1,7, 0,047, 0,014 en 0,00044: na 100 stappen meer dan 10.000 keer zo laag.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_zigzag():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Dwars en in de lengte
    """)
    return


@app.cell(hide_code=True)
def choice_zigzag_cell():
    choice_zigzag = L.choice(
        "zigzag",
        question="Met $\\beta = 0{,}9$ is de gradient van $w_1$ bij elke stap ongeveer 1, en die van $w_2$ om en om ongeveer +1, −1, +1, −1. Hoe groot is $v$ voor elk na veel stappen ongeveer?",
        options={
            "same": "Ongeveer 1 voor allebei: $v$ volgt de gradient",
            "right": "Ongeveer 10 voor $w_1$, en ongeveer 0,5 voor $w_2$, steeds van teken wisselend",
            "zero": "Ongeveer 10 voor $w_1$, en precies 0 voor $w_2$",
        },
        correct="right",
        why={
            "same": "$v$ telt de gradients op, en elke gradient telt 0,9 keer zo zwaar als die erna. Op een gelijkmatige helling groeit $v$ tot een stap hem niet meer verandert: $0{,}9 \\cdot 10 + 1 = 10$.",
            "zero": "De termen heffen elkaar grotendeels op, niet helemaal: de nieuwste gradient telt helemaal, die ervoor maar 0,9 keer. $v$ springt uiteindelijk heen en weer tussen ongeveer +0,53 en −0,53.",
            "right": "Voor $w_1$ groeit $v$ tot een stap hem niet meer verandert: $0{,}9 \\cdot 10 + 1 = 10$. Voor $w_2$ springt hij uiteindelijk heen en weer tussen $1 / 1{,}9 \\approx 0{,}53$ en $-0{,}53$: $0{,}9 \\cdot (-1 / 1{,}9) + 1 = 1 / 1{,}9$. Vergeleken met de gradient worden de stappen in de lengte van het dal dus ongeveer 19 keer zo groot als die dwars erop.",
        },
    )
    choice_zigzag
    return (choice_zigzag,)


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stappen van momentum

        Schrijf `momentum_steps(gs, lr, beta)`: `gs` is een lijst gradients van één parameter, één per stap. Geef de lijst stappen terug die momentum zet, $\eta v$, volgens stap 1 tot en met 3 van het recept. De check roept `momentum_steps([1.0, 1.0, 1.0, 1.0], 0.1, 0.9)` aan, en dat moet `[0.1, 0.19, 0.271, 0.3439]` geven; hij probeert ook gradients die van teken wisselen, andere waarden van `beta`, en één gradient.
        """),
        L.task("Laat `momentum_steps` voor elke gradient de stap van momentum geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_steps():
    box_steps = L.code_box("momentum-steps", lines=8)
    box_steps
    return (box_steps,)


@app.cell(hide_code=True)
def check_steps(box_steps):
    result_steps = L.check("momentum-steps", box_steps.value)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "momentum-steps",
        [
            "`v` begint op 0, vóór de loop over `gs`; in de loop eerst de nieuwe `v`, dan de stap.",
            "De nieuwe `v` is `beta` keer de oude `v`, plus de gradient van deze stap; de stap is `lr` keer die nieuwe `v`.",
        ],
        why="`v = 0.0`, `steps = []`, en voor elke `g` in `gs`: `v = beta * v + g` en `steps.append(lr * v)`; dan `return steps`. In het voorbeeld is $v$ 1, 1,9, 2,71 en 3,439, dus de stappen zijn 0,1, 0,19, 0,271 en 0,3439, op weg naar $0{,}1 \\cdot 10 = 1$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_optimizer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je eigen optimizer met momentum

        Schrijf een class `MomentumSGD` die doet wat `torch.optim.SGD(params, lr=lr, momentum=beta)` doet. `MomentumSGD(params, lr, beta)` krijgt een lijst tensors en bewaart die als `self.params`, met `self.lr`, `self.beta`, en een lijst `self.vs` met één $v$ van nullen voor elke parameter: `torch.zeros_like(p)` maakt een tensor met nullen in de vorm van `p`. `zero_grad` staat al in het vak. `step()` doet stap 2 en 3 van het recept voor elke parameter, met `p.grad` als zijn gradient, binnen `torch.no_grad()`, zoals in je training loops voordat je een optimizer had. De check traint er een klein lineair model mee en vergelijkt de weights met wat `torch.optim.SGD` geeft: na één stap, na vijf, en met andere getallen. `.tolist()` maakt van een tensor een lijst getallen, voor de vergelijking.
        """),
        L.task("Laat `MomentumSGD` dezelfde stappen zetten als torch.optim.SGD met momentum."),
    ])
    return


@app.cell(hide_code=True)
def exercise_optimizer():
    box_optimizer = L.code_box("momentum-sgd", lines=20)
    box_optimizer
    return (box_optimizer,)


@app.cell(hide_code=True)
def check_optimizer(box_optimizer):
    result_optimizer = L.check("momentum-sgd", box_optimizer.value)
    result_optimizer
    return (result_optimizer,)


@app.cell(hide_code=True)
def hints_optimizer():
    L.hints(
        "momentum-sgd",
        [
            "In `__init__` een loop over `params` die `torch.zeros_like(p)` aan `self.vs` toevoegt: één $v$ voor elke parameter, bewaard tussen de stappen.",
            "Ga in `step`, binnen `with torch.no_grad():`, langs de posities, `for i in range(len(self.params)):`, zodat `self.params[i]` en `self.vs[i]` bij elkaar horen.",
        ],
        why="`__init__`: `self.params = params`, `self.lr = lr`, `self.beta = beta`, `self.vs = []` en voor elke `p` in `params`: `self.vs.append(torch.zeros_like(p))`. `step`: `with torch.no_grad():`, en voor elke `i`: `p = self.params[i]`, `self.vs[i] = self.beta * self.vs[i] + p.grad` en `p -= self.lr * self.vs[i]`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het smalle dal uit [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) is nog steeds smal, maar momentum telt op wat steeds dezelfde kant op wijst en laat de rest tegen elkaar wegvallen, zoals $+1$ en $-1$ in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl). Je `MomentumSGD` heeft de vorm van elke optimizer in PyTorch uit [Laat de optimizer de stap zetten](/course/open/torch-optimizer?lang=nl): hij bewaart per parameter iets tussen de stappen. In de volgende les, *Adam*, houdt de optimizer ook een exponentieel voortschrijdend gemiddelde bij van elke gekwadrateerde gradient, zodat elke parameter stappen krijgt van een grootte die bij hem past.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_zigzag, result_steps, result_optimizer):
    L.wrap_up(choice_zigzag, result_steps, result_optimizer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
