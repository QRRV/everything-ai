# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les opt-adam. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-adam", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Momentum](/course/open/opt-momentum?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_old_gradient():
    quick_old_gradient = L.choice(
        "quick-old-gradient",
        question="Momentum met $\\beta = 0{,}9$. Hoe zwaar telt de gradient van drie stappen geleden nog mee in $v$, vergeleken met de nieuwste?",
        options={
            "right": "$0{,}9^3 \\approx 0{,}73$ keer zo zwaar: per stap terug nog een keer 0,9",
            "nothing": "Helemaal niet: $v$ bevat alleen de nieuwste gradient",
            "linear": "$1 - 3 \\cdot 0{,}1 = 0{,}7$ keer zo zwaar: 0,1 minder per stap terug",
        },
        correct="right",
        why={
            "nothing": "$v \\leftarrow \\beta v + g$ bewaart de oude gradients, elke stap verder terug nog een keer vermenigvuldigd met $\\beta$. Zie [Momentum](/course/open/opt-momentum?lang=nl).",
            "linear": "Elke stap terug vermenigvuldigt nog een keer met 0,9: hoe zwaar een gradient telt, krimpt met een factor, niet met een vast bedrag. Zie [Momentum](/course/open/opt-momentum?lang=nl).",
            "right": "$v = g_t + 0{,}9\\, g_{t-1} + 0{,}81\\, g_{t-2} + 0{,}729\\, g_{t-3} + \\dots$: een exponentieel voortschrijdend gemiddelde, op een factor na. Adam houdt er twee bij.",
        },
    )
    quick_old_gradient
    return (quick_old_gradient,)


@app.cell(hide_code=True)
def quick_check_root_mean_square():
    quick_root_mean_square = L.choice(
        "quick-root-mean-square",
        question="De gradients van een parameter waren 3, −3, 3 en −3. Wat is de wortel uit het gemiddelde van hun kwadraten?",
        options={
            "zero": "0: ze heffen elkaar op",
            "right": "3: de grootte van elke gradient",
            "nine": "9: het gemiddelde van de kwadraten",
        },
        correct="right",
        why={
            "zero": "In het gemiddelde heffen ze elkaar op, maar kwadraten zijn nooit negatief: het gemiddelde van de kwadraten is 9. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "nine": "9 is het gemiddelde van de kwadraten; de wortel daaruit is 3. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "De kwadraten zijn allemaal 9, hun gemiddelde is 9, en de wortel daaruit is 3: hoe groot de gradient doorgaans is, wat zijn teken ook is. *Het idee* hieronder gebruikt precies dit getal.",
        },
    )
    quick_root_mean_square
    return (quick_root_mean_square,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Met gradient descent en momentum stapt elke parameter met dezelfde learning rate keer zijn eigen gradient. In een netwerk krijgen sommige parameters gradients die honderd keer zo groot zijn als die van andere: de weights van andere layers, van inputs met andere eenheden, van features die zelden voorkomen. Eén learning rate past niet bij allemaal, en hij moet klein genoeg zijn voor de grootste. Met Adam, de optimizer van deze les, en zijn variant AdamW worden nu de meeste netwerken getraind, ook transformers.
    """)
    return


@app.cell(hide_code=True)
def predict_raw_pixels():
    guess_raw_pixels = L.guess(
        "raw-pixels",
        question="Het MLP uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl) traint één epoch, maar je vergeet de pixels door 255 te delen: ze lopen van 0 tot 255. Met SGD bij een learning rate van 0,1 las het 82% van de testcijfers goed toen de pixels van 0 tot 1 liepen; nu blijft het op 10%. Met Adam bij zijn gebruikelijke learning rate, 0,001, las het 84% goed. Hoeveel leest het nu met Adam volgens jou goed?",
        options={
            "ten": "Ongeveer 10%: hetzelfde probleem als SGD",
            "half": "Ongeveer 50%: het komt er wel, maar veel langzamer",
            "right": "Ongeveer 80%: maar een beetje minder dan eerst",
        },
        reveal="**Ongeveer 80%: 801 van de 1.000.** Aan het begin zijn de gradients van de eerste layer nu honderden keren zo groot, en Adam deelt elke stap door de grootte van de eigen gradients van zijn parameter, dus zijn stappen veranderen nauwelijks. SGD heeft een 100 keer zo kleine learning rate nodig, 0,001, om 77% goed te lezen.",
    )
    guess_raw_pixels
    return (guess_raw_pixels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Adam** houdt per parameter twee exponentieel voortschrijdende gemiddelden bij: $m$ van zijn gradients, zoals momentum, en $s$ van zijn gekwadrateerde gradients:
        """),
        L.formula(r"\begin{aligned} [[grad:m]] &\leftarrow [[hyper:\beta_1]] [[grad:m]] + (1 - [[hyper:\beta_1]]) [[grad:g]] \\ [[grad:s]] &\leftarrow [[hyper:\beta_2]] [[grad:s]] + (1 - [[hyper:\beta_2]]) [[grad:g]]^2 \end{aligned}"),
        mo.md(r"""
        $\sqrt{s}$ is hoe groot de gradient van deze parameter doorgaans is, wat zijn teken ook is, zoals in de instapcheck. Adam deelt de stap door dat getal:
        """),
        L.formula(r"\begin{aligned} [[param:w]] &\leftarrow [[param:w]] - [[hyper:\eta]] \, \frac{[[grad:\hat{m}]]}{\sqrt{[[grad:\hat{s}]]} + [[hyper:\varepsilon]]} \\ [[grad:\hat{m}]] &= \frac{[[grad:m]]}{1 - [[hyper:\beta_1]]^t} \qquad [[grad:\hat{s}]] = \frac{[[grad:s]]}{1 - [[hyper:\beta_2]]^t} \end{aligned}"),
        mo.md(r"""
        Waar de gradient steeds dezelfde kant op wijst, is $\hat{m} \approx g$ en $\sqrt{\hat{s}} \approx |g|$, dus elke stap is ongeveer $\eta$, of $g$ nu 0,01 is of 10. Waar hij steeds van teken wisselt, is $\hat{m}$ klein, en de stap ook. $m$ en $s$ beginnen op 0, dus in de eerste stappen zijn ze te klein; $s$ veel meer dan $m$, omdat $\beta_2$ dichter bij 1 ligt. Delen door $1 - \beta^t$, met $t$ het nummer van de stap, corrigeert dat: bij $t = 1$ is $\hat{m} = g$ en $\hat{s} = g^2$. De gebruikelijke getallen zijn $\eta = 0{,}001$, $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$ en $\varepsilon = 10^{-8}$, en `torch.optim.Adam(model.parameters(), lr=0.001)` gebruikt ze.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Adam** | | een optimizer die het momentum van elke parameter deelt door de wortel uit het gemiddelde van zijn gekwadrateerde gradients | `torch.optim.Adam` |
        | $\beta_1$, $\beta_2$ | bèta één, bèta twee | hoeveel van $m$ en van $s$ elke stap bewaard blijft: meestal 0,9 en 0,999 | `betas=(0.9, 0.999)` |
        | $m$, $s$ | m, s | de voortschrijdende gemiddelden van de gradient en van zijn kwadraat, per parameter | `m`, `s` |
        | $\hat{m}$, $\hat{s}$ | m-dakje, s-dakje | $m$ en $s$, gecorrigeerd voor hun begin op 0 | `m / (1 - beta1 ** t)` |
        | $t$ | t | het nummer van de stap: 1, 2, 3, … | `t` |
        | $\varepsilon$ | epsilon | een piepklein getal, $10^{-8}$, dat voorkomt dat de deling ontspoort waar $\hat{s}$ 0 is | `1e-8` |
        """),
        L.watch_out("De learning rate van Adam betekent iets anders dan die van SGD: hij gaat over de grootte van elke stap, niet over een factor op de gradient. Dus 0,1, goed voor SGD op de cijfers, is veel te groot voor Adam; begin bij 0,001. En met een straf op grote weights in de loss wordt zijn deel van de gradient net als de rest gedeeld door $\\sqrt{\\hat{s}}$, dus hij werkt veel minder op parameters met grote gradients. Daar gaat de volgende les over, *AdamW: weight decay zoals het hoort*."),
        mo.md(r"""
        De figuur: voor 1.500 weights van de hidden layer van het MLP voor de cijfers de grootte van de gradient en de grootte van de stap, bij stap 51 van het trainen, met SGD bij learning rate 0,1 en met Adam bij 0,001, op logaritmische schalen. Met SGD is elke stap 0,1 keer zijn gradient: een rechte lijn, en de weights met de kleinste gradients bewegen nauwelijks. Met Adam liggen de stappen tussen ongeveer 0,00001 en 0,001, voor gradients die duizend keer verschillen.
        """),
        mo.Html(C.step_picture({"sgd": "SGD, η = 0,1", "adam": "Adam, η = 0,001", "x": "grootte van de gradient", "y": "grootte van de stap",
                                "alt": "Twee spreidingsdiagrammen op logaritmische schalen van de grootte van de stap tegen die van de gradient, voor 1.500 weights. Met SGD liggen de punten op een rechte lijn: de stap is 0,1 keer de gradient. Met Adam vormen de punten een platte band tussen ongeveer 0,00001 en 0,001, voor gradients van ongeveer 0,00001 tot 0,01."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Adam, voor elke parameter:

    1. Begin met $m = 0$, $s = 0$ en de stapteller $t = 0$.
    2. Elke stap, na de gradient $g$: $t$ één omhoog, $m \leftarrow \beta_1 m + (1 - \beta_1) g$ en $s \leftarrow \beta_2 s + (1 - \beta_2) g^2$.
    3. Corrigeer allebei: $\hat{m} = m / (1 - \beta_1^t)$ en $\hat{s} = s / (1 - \beta_2^t)$.
    4. Stap: $w \leftarrow w - \eta \, \hat{m} / (\sqrt{\hat{s}} + \varepsilon)$. In PyTorch: `torch.optim.Adam(params, lr=0.001)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee parameters waarvan de gradients altijd 0,01 en 10 zijn: een factor duizend uit elkaar. Met SGD en learning rate 0,001 zouden hun stappen 0,00001 en 0,01 zijn. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit. Zoals hij is, ontbreken de correcties.
    """)
    return


@app.cell
def example_adam():
    import numpy as np

    _lr = 0.001
    _beta1 = 0.9
    _beta2 = 0.999
    # 1. m en s beginnen op 0, één voor elke parameter
    _m = np.zeros(2)
    _s = np.zeros(2)
    for _t in range(1, 6):
        _g = np.array([0.01, 10.0])
        # 2. de twee voortschrijdende gemiddelden
        _m = _beta1 * _m + (1 - _beta1) * _g
        _s = _beta2 * _s + (1 - _beta2) * _g**2
        # 3. maak af: deel _m door 1 - _beta1 ** _t, en _s door 1 - _beta2 ** _t
        _m_hat = _m
        _s_hat = _s
        # 4. de stap van elke parameter
        print(_t, _lr * _m_hat / (_s_hat**0.5 + 1e-8))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Voor allebei de parameters zijn de stappen gelijk, op de laatste cijfers na: de gradient van 10 is duizend keer zo groot, maar de wortel uit zijn $s$ ook. Zoals de cel is, zonder de correcties, is de eerste stap 0,0032, meer dan drie keer de learning rate, en de volgende groeien: 0,0042, 0,0050, 0,0054, 0,0058. $s$ begint verder onder zijn waarde dan $m$, dus de eerste stappen komen te groot uit. Met de correcties is elke stap precies 0,001 voor allebei: de learning rate, of de gradient nu 0,01 is of 10.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De eerste stap
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-step",
        question="Bij stap $t = 1$ krijgt een parameter de gradient $g = -0{,}004$. Hoeveel verandert Adam de parameter, met de correcties en $\\eta = 0{,}001$?",
        options={
            "sgd": "$-\\eta g = 0{,}000004$: de learning rate keer de gradient, zoals bij SGD",
            "zero": "Bijna 0: $m$ en $s$ zijn net begonnen",
            "right": "$+0{,}001$: de learning rate, tegen het teken van $g$ in",
        },
        correct="right",
        why={
            "sgd": "Adam deelt door de wortel uit $\\hat{s}$, de grootte van de gradient zelf, dus de grootte van $g$ valt weg.",
            "zero": "$m$ en $s$ zijn na één stap klein, maar de stap deelt het ene door de wortel uit het andere, dus hun grootte valt weg; met de correcties is na de eerste stap $\\hat{m} = g$ en $\\hat{s} = g^2$.",
            "right": "$\\hat{m} = -0{,}004$ en $\\sqrt{\\hat{s}} = 0{,}004$, dus de stap is $0{,}001 \\cdot (-0{,}004 / 0{,}004) = -0{,}001$, en $w$ min die stap gaat $\\eta$ omhoog: tegen het teken van de gradient in, zodat de loss daalt. In oefening 2 is een stap dat getal zelf: hier $-0{,}001$.",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stappen van Adam

        Schrijf `adam_steps(gs, lr, beta1, beta2)`: `gs` is een lijst gradients van één parameter, één per stap. Geef de lijst stappen terug die Adam zet, $\eta \hat{m} / (\sqrt{\hat{s}} + \varepsilon)$ met $\varepsilon = 10^{-8}$, volgens het recept. De check roept `adam_steps([4.0, 1.0, -2.0], 0.1, 0.9, 0.999)` aan, en dat moet ongeveer `[0.1, 0.08306, 0.02986]` geven; hij probeert ook piepkleine en enorme gradients, een gradient van 0, en andere betas.
        """),
        L.task("Laat `adam_steps` voor elke gradient de stap van Adam geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_steps():
    box_steps = L.code_box("adam-steps", lines=14)
    box_steps
    return (box_steps,)


@app.cell(hide_code=True)
def check_steps(box_steps):
    result_steps = L.check("adam-steps", box_steps.value)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "adam-steps",
        [
            "`m`, `s` en `t` beginnen op 0, vóór de loop; in de loop eerst `t` één omhoog, dan stap 2 tot en met 4 van het recept, zoals in het uitgewerkte voorbeeld.",
            "De correcties gebruiken het nummer van deze stap: `m / (1 - beta1 ** t)` en `s / (1 - beta2 ** t)`. De stap is `lr * m_hat / (s_hat ** 0.5 + 1e-8)`.",
        ],
        why="`m = s = 0.0`, `t = 0`, `steps = []`, en voor elke `g`: `t = t + 1`, `m = beta1 * m + (1 - beta1) * g`, `s = beta2 * s + (1 - beta2) * g ** 2`, `m_hat = m / (1 - beta1 ** t)`, `s_hat = s / (1 - beta2 ** t)` en `steps.append(lr * m_hat / (s_hat ** 0.5 + 1e-8))`; dan `return steps`. De eerste stap is $0{,}1 \\cdot 4 / 4 = 0{,}1$; bij de derde is de gradient van teken gewisseld, dus $\\hat{m}$ is klein, en de stap ook.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_optimizer():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Je eigen Adam

        Schrijf een class `AdamOptimizer` die doet wat `torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))` doet, in de vorm van je `MomentumSGD` uit [Momentum](/course/open/opt-momentum?lang=nl). `AdamOptimizer(params, lr, beta1, beta2)` bewaart `self.params`, `self.lr`, `self.beta1`, `self.beta2`, een stapteller `self.t = 0`, en twee lijsten, `self.ms` en `self.ss`, met een tensor met nullen voor elke parameter. `zero_grad` staat al in het vak. `step()` verhoogt `self.t` met één en doet dan, binnen `torch.no_grad()`, stap 2 tot en met 4 van het recept voor elke parameter, met $\varepsilon = 10^{-8}$. De check traint er een klein lineair model mee en vergelijkt de weights met wat `torch.optim.Adam` geeft: na één stap, na vijf, en met andere getallen.
        """),
        L.task("Laat `AdamOptimizer` dezelfde stappen zetten als torch.optim.Adam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_optimizer():
    box_optimizer = L.code_box("adam-optimizer", lines=26)
    box_optimizer
    return (box_optimizer,)


@app.cell(hide_code=True)
def check_optimizer(box_optimizer):
    result_optimizer = L.check("adam-optimizer", box_optimizer.value)
    result_optimizer
    return (result_optimizer,)


@app.cell(hide_code=True)
def hints_optimizer():
    L.hints(
        "adam-optimizer",
        [
            "`__init__` is die van je `MomentumSGD`, met twee lijsten in plaats van één, allebei gevuld met `torch.zeros_like(p)`, en `self.t = 0`.",
            "`self.t` gaat één keer per `step()` omhoog, niet één keer per parameter: vóór de loop over de parameters. In de loop horen `self.ms[i]` en `self.ss[i]` bij `self.params[i]`.",
        ],
        why="`__init__`: `self.params = params`, `self.lr = lr`, `self.beta1 = beta1`, `self.beta2 = beta2`, `self.t = 0`, `self.ms = []`, `self.ss = []`, en voor elke `p` in `params`: `self.ms.append(torch.zeros_like(p))` en `self.ss.append(torch.zeros_like(p))`. `step`: `self.t = self.t + 1`, dan `with torch.no_grad():` en voor elke `i`: `p = self.params[i]`, `self.ms[i] = self.beta1 * self.ms[i] + (1 - self.beta1) * p.grad`, `self.ss[i] = self.beta2 * self.ss[i] + (1 - self.beta2) * p.grad ** 2`, `m_hat = self.ms[i] / (1 - self.beta1 ** self.t)`, `s_hat = self.ss[i] / (1 - self.beta2 ** self.t)` en `p -= self.lr * m_hat / (s_hat ** 0.5 + 1e-8)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** $m$ is het momentum uit [Momentum](/course/open/opt-momentum?lang=nl), als gemiddelde; $\sqrt{s}$ is de wortel uit een gemiddelde van kwadraten, zoals een standaardafwijking in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl). Het ene door het andere delen geeft elke parameter stappen van ongeveer de learning rate, dus de schaal van de inputs, zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl), doet er veel minder toe. In de volgende les, *AdamW: weight decay zoals het hoort*, zie je wat dat doet met een straf op grote weights, en waarom transformers met AdamW getraind worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_steps, result_optimizer):
    L.wrap_up(choice_first, result_steps, result_optimizer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
