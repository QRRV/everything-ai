# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les opt-adamw. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow regularization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-adamw", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Adam](/course/open/opt-adam?lang=nl) en [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hundredfold():
    quick_hundredfold = L.choice(
        "quick-hundredfold",
        question="Elke gradient van één parameter wordt 100 keer zo groot. Waarom verandert de stap van Adam voor die parameter nauwelijks?",
        options={
            "lr": "Het verlaagt zijn eigen learning rate honderd keer om dat goed te maken",
            "right": "Het deelt door $\\sqrt{\\hat{s}}$, die ook honderd keer zo groot wordt",
            "clip": "Het kapt elke gradient af op 1 voordat het hem gebruikt",
        },
        correct="right",
        why={
            "lr": "De learning rate blijft zoals jij hem instelt. Wat verandert, is $\\sqrt{\\hat{s}}$, waardoor Adam deelt. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "clip": "Adam kapt niets af: het deelt $\\hat{m}$ door $\\sqrt{\\hat{s}}$. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "right": "$\\hat{m}$ en $\\sqrt{\\hat{s}}$ worden allebei honderd keer zo groot, dus hun verhouding blijft gelijk. Vandaag zie je wat dat doet met een straf die deel van de gradient is.",
        },
    )
    quick_hundredfold
    return (quick_hundredfold,)


@app.cell(hide_code=True)
def quick_check_l2():
    quick_l2 = L.choice(
        "quick-l2",
        question="L2-regularisatie telt $\\lambda \\sum_j w_j^2$ op bij de loss. Wat doet dat bij gewone gradient descent in elke update-stap met elke weight?",
        options={
            "zero": "Het zet de kleinste weights op precies 0 en laat de rest met rust",
            "lr": "Het maakt de learning rate bij elke stap een beetje kleiner",
            "right": "Het trekt elke weight naar 0, evenredig met zijn grootte",
        },
        correct="right",
        why={
            "zero": "L2 maakt weights kleiner, niet precies 0: zijn trekkracht, $2\\lambda w_j$, wordt kleiner naarmate de weight kleiner wordt. Zie [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).",
            "lr": "De learning rate blijft gelijk; de helling krijgt er $2\\lambda w_j$ bij. Zie [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).",
            "right": "De helling krijgt er $2 \\lambda w_j$ bij, dus elke stap laat $w_j$ met een fractie krimpen: weight decay. Met gradient descent zijn L2 en weight decay hetzelfde.",
        },
    )
    quick_l2
    return (quick_l2,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Met gradient descent zijn een L2-straf en weight decay hetzelfde: elke stap laat elke weight met dezelfde fractie krimpen. Met Adam niet, en veel code gebruikte de L2-straf met Adam zonder te merken wat die deed. In 2017 lieten Ilya Loshchilov en Frank Hutter het verschil zien, en hoe je het oplost. Hun oplossing, AdamW, is de optimizer waarmee de meeste transformers nu getraind worden.
    """)
    return


@app.cell(hide_code=True)
def predict_quiet():
    guess_quiet = L.guess(
        "quiet-weight",
        question="Een weight begint op 1 en krijgt geen gradient van de data: hij krijgt alleen de L2-straf, $\\lambda w$, als gradient. Adam traint hem met een learning rate van 0,01. Met $\\lambda = 0{,}01$ is de weight na 213 stappen onder 0,01. Met $\\lambda = 0{,}0001$, een straf die honderd keer zo zwak is, na hoeveel stappen volgens jou?",
        options={
            "hundred": "Na ongeveer 21.300: honderd keer zo zwak, honderd keer zo langzaam",
            "right": "Ook na 213 stappen: λ maakt hier geen verschil",
            "never": "Nooit: zo'n zwakke straf doet bijna niets",
        },
        reveal="**Ook na 213 stappen.** Adam deelt de gradient door de wortel uit zijn eigen gemiddelde kwadraat. Als de straf de hele gradient is, haalt die deling zijn grootte er helemaal uit: de stappen zijn hetzelfde wat $\\lambda$ ook is, de eerste ongeveer de learning rate, 0,01. De sterkte van de straf heeft op deze weight helemaal geen effect.",
    )
    guess_quiet
    return (guess_quiet,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een L2-straf $\frac{\lambda}{2} w^2$ telt $\lambda w$ op bij de gradient van elke weight. Adam behandelt die dan als de rest van de gradient: hij gaat in $m$ en $s$, en wordt gedeeld door $\sqrt{\hat{s}}$. Voor een weight waarvan de gradient vooral uit de straf bestaat, haalt die deling de grootte van de straf weg: de weight zet dezelfde stappen naar 0, de eerste van ongeveer $\eta$, wat $\lambda$ ook is. Voor een weight met grote gradients van de data is $\sqrt{\hat{s}}$ groot, en verdrinkt de straf: die weight wordt nauwelijks naar 0 getrokken. Hoe sterk een weight geregulariseerd wordt, hangt dan af van de grootte van zijn gradients, niet alleen van $\lambda$.

        **AdamW** haalt de straf uit de gradient. Adam stapt met alleen de gradient van de loss, en daarnaast krimpt elke weight met dezelfde fractie $\eta \lambda$:
        """),
        L.formula(r"\begin{aligned} [[param:w]] &\leftarrow (1 - [[hyper:\eta]] [[hyper:\lambda]]) \, [[param:w]] \\ [[param:w]] &\leftarrow [[param:w]] - [[hyper:\eta]] \, \frac{[[grad:\hat{m}]]}{\sqrt{[[grad:\hat{s}]]} + [[hyper:\varepsilon]]} \end{aligned}"),
        mo.md(r"""
        Dat is **decoupled weight decay**: het krimpen gaat niet meer door de deling van Adam, dus $\lambda$ betekent voor elke weight hetzelfde. In PyTorch telt `torch.optim.Adam(params, weight_decay=0.1)` de straf op bij de gradient, en laat `torch.optim.AdamW(params, weight_decay=0.1)` de weights los van de gradient krimpen; AdamW laat eerst krimpen en zet dan de stap van Adam. Weight decay krijgen meestal alleen de weight-matrices: bias terms en de $\gamma$ en $\beta$ van normalization layers krijgen niets, omdat die alleen iets optellen of herschalen, en geen verbinding tussen neuronen zijn. Daarvoor geef je AdamW twee **parameter groups**, elk met een eigen `weight_decay`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **AdamW** | Adam W | Adam met decoupled weight decay: elke weight krimpt per stap met de fractie $\eta\lambda$, los van de stap van Adam | `torch.optim.AdamW` |
        | **decoupled weight decay** | | elke weight direct laten krimpen, in plaats van een straf bij zijn gradient op te tellen | `w = w - lr * wd * w` |
        | **parameter groups** | | een lijst dicts, elk met eigen parameters en instellingen, die je aan een optimizer geeft | `[{"params": decay, "weight_decay": 0.1}, ...]` |
        | `p.dim()` | p dim | het aantal dimensies van een tensor: 2 voor een weight-matrix, 1 voor bias terms | `p.dim() >= 2` |
        """),
        L.watch_out("De `weight_decay` van PyTorch is de $\\lambda$ in $\\lambda w$, de gradient van $\\frac{\\lambda}{2} w^2$. [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl) schreef de straf als $\\lambda w^2$, met gradient $2\\lambda w$, dus die $\\lambda$ is de helft van de `weight_decay` van PyTorch. De standaardwaarde van PyTorch voor AdamW is 0,01; voor transformers is 0,1 gebruikelijk. Weight decay met AdamW heeft veel stappen nodig voordat het telt: met $\\eta = 0{,}001$ en $\\lambda = 0{,}1$ verliest een weight per stap een tienduizendste."),
        mo.md(r"""
        De figuur: de weight uit de gok, zonder gradient van de data, over 1.000 stappen met learning rate 0,01, voor $\lambda = 0{,}01$ en $\lambda = 0{,}0001$. Met de L2-straf binnen Adam liggen de twee lijnen op elkaar: de weight is in allebei de gevallen na ongeveer 200 stappen bijna 0. Met AdamW krimpt hij per stap met de fractie $\eta\lambda$: tot 0,905 na 1.000 stappen met $\lambda = 0{,}01$, en tot 0,999 met $\lambda = 0{,}0001$.
        """),
        mo.Html(C.quiet_picture({"l2": "L2-straf binnen Adam", "adamw": "AdamW", "x": "stap", "y": "w",
                                 "big": "λ = 0,01", "small": "λ = 0,0001", "both": "allebei de waarden van λ",
                                 "alt": "Twee panelen over 1.000 stappen. Met een L2-straf binnen Adam daalt de weight in ongeveer 200 stappen van 1 naar 0, voor allebei de waarden van lambda hetzelfde. Met AdamW daalt hij langzaam naar 0,905 voor lambda 0,01 en blijft hij op 0,999 voor lambda 0,0001."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    AdamW, voor elke parameter:

    1. Begin met $m = 0$, $s = 0$ en $t = 0$.
    2. Elke stap: de gradient $g$ van de loss zonder straf; $t$ één omhoog, $m$ en $s$ zoals bij Adam, en allebei gecorrigeerd.
    3. Laat de weight krimpen: $w \leftarrow (1 - \eta\lambda)\, w$.
    4. De stap van Adam: $w \leftarrow w - \eta\, \hat{m} / (\sqrt{\hat{s}} + \varepsilon)$.
    5. In PyTorch: `torch.optim.AdamW(groups, lr=...)`, met weight decay alleen voor de weight-matrices.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee weights die allebei op 1 beginnen: een rustige die geen gradient van de data krijgt, en een drukke waarvan de gradient willekeurige ruis is, uit een generator met een vaste seed: `_rng.normal()` geeft een getal uit de normale verdeling met gemiddelde 0 en standaardafwijking 1, zoals `torch.randn` doet. 1.000 stappen AdamW met $\eta = 0{,}01$ en $\lambda = 0{,}1$. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit. Zoals hij is, is de cel Adam zonder weight decay.
    """)
    return


@app.cell
def example_adamw():
    import numpy as np

    _rng = np.random.default_rng(0)
    _lr = 0.01
    _wd = 0.1
    _w = np.array([1.0, 1.0])
    # 1. m en s beginnen op 0
    _m = np.zeros(2)
    _s = np.zeros(2)
    for _t in range(1, 1001):
        # 2. de gradient van de data: geen voor de rustige weight, ruis voor de drukke; m en s, gecorrigeerd
        _g = np.array([0.0, _rng.normal()])
        _m = 0.9 * _m + 0.1 * _g
        _s = 0.999 * _s + 0.001 * _g**2
        _m_hat = _m / (1 - 0.9**_t)
        _s_hat = _s / (1 - 0.999**_t)
        # 3. maak af: laat _w krimpen met de fractie _lr * _wd
        # 4. de stap van Adam
        _w = _w - _lr * _m_hat / (_s_hat**0.5 + 1e-8)
    print(_w)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals de cel is, blijft de rustige weight precies op 1: Adam heeft geen gradient om mee te stappen, dus niets verschuift hem. De drukke dwaalt met zijn ruis af, naar 1,45. Met stap 3 krimpen ze allebei per stap met de fractie $0{,}01 \cdot 0{,}1 = 0{,}001$: de rustige tot $0{,}999^{1000} \approx 0{,}368$, en de drukke, die ook de stappen van Adam zet, tot 0,75. Het krimpen is voor allebei hetzelfde, wat hun gradients ook zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_arg():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee optimizers, één argument
    """)
    return


@app.cell(hide_code=True)
def choice_arg_cell():
    choice_arg = L.choice(
        "weight-decay-arg",
        question="In je code staat `torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)`. Wat doet `weight_decay` daar?",
        options={
            "adamw": "Wat AdamW doet: elke weight krimpt per stap met de fractie `lr * 0.1`",
            "right": "Het telt `0.1 * w` bij elke gradient op: ook dat deelt Adam door $\\sqrt{\\hat{s}}$",
            "ignored": "Niets: gewone Adam heeft geen weight decay, dus het negeert het argument",
        },
        correct="right",
        why={
            "adamw": "Alleen `torch.optim.AdamW` laat los van de gradient krimpen. `torch.optim.Adam` met `weight_decay` is een L2-straf binnen Adam.",
            "ignored": "Het doet wel iets, alleen niet wat de naam doet denken: het telt de straf op bij de gradient.",
            "right": "Het is een L2-straf binnen Adam, dus hij wordt met de rest gedeeld door $\\sqrt{\\hat{s}}$: weights met grote gradients merken er nauwelijks iets van. Gebruik voor decoupled weight decay `torch.optim.AdamW`.",
        },
    )
    choice_arg
    return (choice_arg,)


@app.cell(hide_code=True)
def text_exercise_path():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén weight onder AdamW

        Schrijf `adamw_path(gs, w, lr, wd)`: `gs` is een lijst gradients van de loss voor één weight, één per stap, en `w` zijn beginwaarde. Geef de lijst van `w` na elke stap van AdamW terug, volgens stap 1 tot en met 4 van het recept, met $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$ en $\varepsilon = 10^{-8}$: eerst krimpen, dan de stap van Adam. De check roept `adamw_path([0.3, 0.1, 0.0], 1.0, 0.1, 0.5)` aan, en dat moet ongeveer `[0.85, 0.72039, 0.61704]` geven; hij probeert ook een weight zonder gradients, een weight decay van 0, en andere getallen.
        """),
        L.task("Laat `adamw_path` geven waar de weight na elke stap van AdamW is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_path():
    box_path = L.code_box("adamw-path", lines=15)
    box_path
    return (box_path,)


@app.cell(hide_code=True)
def check_path(box_path):
    result_path = L.check("adamw-path", box_path.value)
    result_path
    return (result_path,)


@app.cell(hide_code=True)
def hints_path():
    L.hints(
        "adamw-path",
        [
            "Je `adam_steps` uit [Adam](/course/open/opt-adam?lang=nl) geeft de stappen; hier verandert `w` met elke stap, en de lijst bevat `w` zelf.",
            "Laat in elke stap eerst krimpen, met de `w` van de stap ervoor: `w = w - lr * wd * w`. Zet dan vanaf die `w` de stap van Adam.",
        ],
        why="`m = s = 0.0`, `t = 0`, `ws = []`, en voor elke `g`: `t = t + 1`, `m = 0.9 * m + 0.1 * g`, `s = 0.999 * s + 0.001 * g ** 2`, `m_hat = m / (1 - 0.9 ** t)`, `s_hat = s / (1 - 0.999 ** t)`, `w = w - lr * wd * w`, `w = w - lr * m_hat / (s_hat ** 0.5 + 1e-8)` en `ws.append(w)`; dan `return ws`. In het voorbeeld laat de eerste stap 1 krimpen tot $1 - 0{,}1 \\cdot 0{,}5 = 0{,}95$, en haalt de eerste stap van Adam er 0,1 af: 0,85.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_groups():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Weight decay alleen voor de weight-matrices

        Schrijf `param_groups(model, wd)`: ga langs `model.parameters()` en zet elke parameter met 2 of meer dimensies, de weight-matrices, in één lijst, en de andere, bias terms en de $\gamma$ en $\beta$ van normalization layers, in een andere. Geef een lijst van twee dicts terug: `{"params": ..., "weight_decay": wd}` voor de eerste lijst en `{"params": ..., "weight_decay": 0.0}` voor de tweede. `torch.optim.AdamW(param_groups(model, 0.1), lr=0.001)` laat dan alleen de weight-matrices krimpen. De check maakt een klein netwerk met twee `nn.Linear`-layers en een `nn.LayerNorm`, en telt per groep hoeveel getallen zijn parameters bevatten (`p.numel()` geeft dat voor één tensor) en welke weight decay hij heeft.
        """),
        L.task("Laat `param_groups` de weight-matrices weight decay geven, en de rest niet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_groups():
    box_groups = L.code_box("param-groups", lines=10)
    box_groups
    return (box_groups,)


@app.cell(hide_code=True)
def check_groups(box_groups):
    result_groups = L.check("param-groups", box_groups.value)
    result_groups
    return (result_groups,)


@app.cell(hide_code=True)
def hints_groups():
    L.hints(
        "param-groups",
        [
            "Twee lege lijsten, dan een loop over `model.parameters()` met een `if` op `p.dim() >= 2`.",
            "Een groep is een dict met de keys `\"params\"` en `\"weight_decay\"`; de functie geeft een lijst van de twee terug.",
        ],
        why="`decay = []`, `no_decay = []`; voor elke `p` in `model.parameters()`: `decay.append(p)` als `p.dim() >= 2`, anders `no_decay.append(p)`; dan `return [{\"params\": decay, \"weight_decay\": wd}, {\"params\": no_decay, \"weight_decay\": 0.0}]`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Met de gradient descent uit [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl) is een L2-straf weight decay. [Adam](/course/open/opt-adam?lang=nl) deelt elk deel van de gradient door $\sqrt{\hat{s}}$, de straf ook, en dat breekt die gelijkheid; AdamW zet het krimpen weer buiten de deling. In de volgende les, *Learning-rate schedules*, verandert de learning rate zelf tijdens het trainen: warmup aan het begin, langzaam afnemen tot het eind.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_arg, result_path, result_groups):
    L.wrap_up(choice_arg, result_path, result_groups)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
