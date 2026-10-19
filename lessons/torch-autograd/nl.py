# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les torch-autograd. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow variable

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-autograd", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tensors in PyTorch](/course/open/torch-tensors?lang=nl) en [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="`t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])`. Wat geeft `t.sum()` terug?",
        options={
            "number": "Het Python-getal 10.0",
            "rows": "`tensor([3., 7.])`: één som per rij",
            "right": "`tensor(10.)`: een tensor zonder axes",
        },
        correct="right",
        why={
            "number": "Een samenvatting van een tensor is weer een tensor. Wat haalt het getal eruit? Zie *Een tensor zonder axes* in [Tensors in PyTorch](/course/open/torch-tensors?lang=nl).",
            "rows": "Zonder `dim` telt `.sum()` alle vier de getallen op. Zie [Tensors in PyTorch](/course/open/torch-tensors?lang=nl).",
            "right": "Alle vier de getallen opgeteld, in een tensor zonder axes; `.item()` zou het Python-getal 10.0 geven.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_twice():
    quick_twice = L.choice(
        "quick-twice",
        question="Met je `Value` en `backward` uit [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl): `a = Value(3)` en `b = a * a`. Wat is `a.grad` na `backward(b)`?",
        options={
            "one": "3",
            "right": "6",
            "square": "9",
        },
        correct="right",
        why={
            "one": "`a` is allebei de ouders van `b`. Hoe vaak telt `backward_step()` op bij `a.grad`? Zie het uitgewerkte voorbeeld in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl).",
            "square": "9 is `b.data`, de waarde van `b`, geen helling. Zie het uitgewerkte voorbeeld in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl).",
            "right": "`a` is allebei de ouders van `b`, elk met local gradient 3, en `backward_step()` telt ze allebei op: 3 + 3 = 6, de helling van $a^2$ bij 3.",
        },
    )
    quick_twice
    return (quick_twice,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl) onthield je `Value` waar elk getal vandaan kwam, en vond je `backward` elke helling, voor een graaf van losse getallen. In [Tensors in PyTorch](/course/open/torch-tensors?lang=nl) bevat een tensor duizenden getallen tegelijk, maar hij onthoudt nog niets. **Autograd** van PyTorch brengt die twee samen: tensors die onthouden, en een `backward()` voor hele tensors tegelijk. Het doet wat je project doet, en geeft dezelfde hellingen. Vanaf nu geeft het de hellingen voor elk netwerk in deze course, tot en met het language model aan het eind.

    **Een voorbeeld uit het dagelijks leven.** Een kassabon zet elk gescand artikel op een rij, in volgorde. Daarmee kan de winkel achteraf uitrekenen hoeveel elk artikel aan het totaal bijdroeg, zonder iets opnieuw te scannen.
    """)
    return


@app.cell(hide_code=True)
def predict_between():
    guess_between = L.guess(
        "between",
        question="Met je `Value` kreeg elke waarde in de graaf een `grad`, ook die ertussen. In PyTorch: `w = torch.tensor(2.0, requires_grad=True)`, `z = w * 3`, `loss = z * z`, en dan `loss.backward()`. Wat denk je dat `z.grad` is?",
        options={
            "right": "`None`: PyTorch houdt alleen voor `w` een helling",
            "slope": "`tensor(12.)`: de helling van de loss naar `z`",
            "zero": "`tensor(0.)`",
        },
        reveal="**None.** PyTorch geeft de helling door via `z` op weg naar `w`, precies zoals je `backward` deed, maar houdt alleen een `.grad` voor de tensors die je zelf maakte met `requires_grad=True`: de parameters. Die verandert een trainingsstap; de hellingen ertussen worden weggegooid, om geheugen te sparen. `w.grad` is 36: $\\mathcal{L} = (3w)^2 = 9w^2$, met helling $18 \\cdot 2 = 36$ bij $w = 2$. Als je `z.grad` leest, laat PyTorch ook een waarschuwing zien.",
    )
    guess_between
    return (guess_between,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Autograd** is de automatic differentiation van PyTorch: wat je `Value` en `backward` deden, voor tensors. Het heeft drie delen.

        **Opnemen.** Een tensor gemaakt met `requires_grad=True` is een tensor waarvoor PyTorch hellingen bijhoudt, zoals een parameter. Elke berekening ermee geeft een uitkomst die onthoudt hoe hij gemaakt is, in zijn **`grad_fn`**: `c = a * b` heeft `grad_fn=<MulBackward0>`, de vermenigvuldiging met zijn local gradients. Dat deed `parents` in je `Value`: de graaf wordt opgenomen terwijl de forward pass loopt.

        **De backward pass.** `loss.backward()` doet wat je `backward(loss)` deed: hij loopt de opgenomen graaf terug vanaf de loss, in de goede volgorde, beginnend bij 1, en elke stap vermenigvuldigt met een local gradient. Hij heeft een loss met één getal erin nodig. De helling van elke **leaf tensor** met `requires_grad=True`, een tensor die je zelf maakte in plaats van de uitkomst van een berekening, komt in zijn `.grad`, met dezelfde shape: een `w` met twee weights krijgt twee hellingen. De tensors ertussen geven hun hellingen door, maar houden er geen.

        **Optellen, en niet opnemen.** Net als in je `Value` telt `.grad` elke nieuwe helling op bij de oude. Een trainingsstap zet daarom `w.grad = None` vóór `loss.backward()`: zo zet je in PyTorch de gradients op nul, waarbij `None` betekent: nog geen helling. En de stap zelf mag niet worden opgenomen: die verandert de parameters, hij hoort niet bij het model. Binnen **`with torch.no_grad():`** neemt PyTorch niets op, en daar verandert `w -= lr * w.grad` de tensor `w` zelf. Daarbuiten stopt PyTorch met "a leaf Variable that requires grad is being used in an in-place operation". `torch.no_grad()` werkt ook als decorator: met `@torch.no_grad()` boven een `def`, met de haakjes, wordt niets in die functie opgenomen, zoals bij een functie die alleen voorspelt.

        | In je project | In PyTorch |
        |---|---|
        | `a = Value(2)` | `a = torch.tensor(2.0, requires_grad=True)` |
        | `c.parents` | `c.grad_fn` |
        | `backward(loss)` | `loss.backward()` |
        | `a.grad` | `a.grad`, alleen voor een leaf tensor |
        | `p.grad = 0` | `p.grad = None` |
        | `p.data = p.data - lr * p.grad` | `p -= lr * p.grad`, binnen `with torch.no_grad():` |

        Het neuron van oefening 3 in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl), met zijn twee weights in één tensor $\mathbf{w}$ en zijn inputs in één tensor $\mathbf{x}$:
        """),
        L.formula(r"[[pred:\hat{y}]] = \mathrm{ReLU}([[param:\mathbf{w}]] \cdot \mathbf{x} + [[param:b]]), \qquad [[loss:\mathcal{L}]] = ([[pred:\hat{y}]] - y)^2"),
        mo.md(r"""
        `loss.backward()` geeft `w.grad` één helling per weight, in de shape van `w`, en `b.grad` één getal:
        """),
        L.formula(r"[[grad:\frac{\partial \mathcal{L}}{\partial \mathbf{w}}]] = \left([[grad:\frac{\partial \mathcal{L}}{\partial w_1}]],\ [[grad:\frac{\partial \mathcal{L}}{\partial w_2}]]\right)"),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **autograd** | auto-grad | de automatic differentiation van PyTorch: het neemt de graaf op en rekent elke helling uit, zoals je `Value` en `backward` deden | |
        | `requires_grad=True` | | "houd hellingen bij voor deze tensor": PyTorch neemt elke berekening ermee op | `torch.tensor(2.0, requires_grad=True)` |
        | `grad_fn` | grad f n | hoe een uitkomst gemaakt is, met zijn local gradients: zijn schakel terug in de graaf, zoals `parents` dat was | `c.grad_fn` |
        | **leaf tensor** | | een tensor die je zelf maakte, geen uitkomst van een berekening; met `requires_grad=True` krijgt hij een `.grad` | `w.is_leaf` |
        | `.backward()` | | de backward pass vanaf een loss met één getal erin: hij vult de `.grad` van elke leaf tensor | `loss.backward()` |
        | `.grad` | | de helling van de loss naar een leaf tensor, in zijn shape; `None` vóór de eerste backward pass | `w.grad` |
        | `-=` | min is | `w -= a` is kort voor `w = w - a`; bij een tensor verandert het de tensor zelf, ter plekke, in plaats van een nieuwe te maken | `w -= lr * w.grad` |
        | `torch.no_grad()` | | een blok, of een decorator, waarin PyTorch niets opneemt: voor de stap, en voor voorspellingen | `with torch.no_grad():` |
        | `torch.relu` | | ReLU op elk getal van een tensor | `torch.relu(z)` |
        """),
        L.watch_out("`w = w - lr * w.grad` lijkt dezelfde stap, maar maakt een nieuwe tensor: de uitkomst van een berekening, geen leaf tensor meer. De volgende `loss.backward()` geeft hem geen `.grad`, en de stap daarna stopt, omdat zijn `.grad` `None` is. Verander de tensor zelf, met `-=`, binnen `torch.no_grad()`."),
        mo.md(r"""
        De graaf die PyTorch opneemt voor $d = a \cdot b + a$, met $a = 2$ en $b = 3$, zoals in [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl). Naast elke pijl staat zijn local gradient; onder elke waarde de helling van $d$ naar die waarde.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_graph():
    mo.vstack([
        C.picture({"title": "De graaf die PyTorch opneemt voor d = a · b + a",
                   "alt": "a = 2 en b = 3 gaan in c = a · b = 6, en c en a gaan in d = c + a = 8. Local gradients: 3 van a naar c, 2 van b naar c, 1 van c naar d en 1 van a naar d. Hellingen van d: 4 voor a, 2 voor b, 1 voor c en 1 voor d."}),
        mo.md(r"""
        PyTorch neemt `c` op met `grad_fn=<MulBackward0>` en `d` met `grad_fn=<AddBackward0>`. `d.backward()` geeft de hellingen 4 voor $a$, 3 + 1 langs zijn twee paden, en 2 voor $b$: hetzelfde als met je `Value`. Alleen $a$ en $b$ zijn leaf tensors, dus alleen zij houden hun helling, in `a.grad` en `b.grad`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om hellingen te krijgen met autograd, en een stap te zetten:

    1. Maak de parameters met `requires_grad=True`.
    2. Forward pass: reken de loss uit met tensors. PyTorch neemt de graaf op.
    3. Zet elke `.grad` op `None`, en roep `loss.backward()` aan.
    4. Lees de hellingen in `.grad`, en zet de stap binnen `with torch.no_grad():`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eerst de graaf van de figuur, dan het lineaire model uit [Tensors in PyTorch](/course/open/torch-tensors?lang=nl), met één stap. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_autograd():
    import torch

    # 1. de parameters van d = a * b + a
    _a = torch.tensor(2.0, requires_grad=True)
    _b = torch.tensor(3.0, requires_grad=True)
    # 2. forward: PyTorch neemt de graaf op
    _c = _a * _b
    _d = _c + _a
    print(_c, _d)
    # 3. de backward pass vanaf d
    _d.backward()
    # 4. de hellingen van de leaf tensors
    print(_a.grad, _b.grad)

    # het lineaire model, met dezelfde vier stappen
    _X = torch.tensor([[1.0, 2.0], [3.0, 1.0], [0.0, 2.0], [2.0, 2.0]])
    _y = torch.tensor([5.0, 4.0, 3.0, 6.0])
    # 1.
    _w = torch.tensor([1.0, 1.5], requires_grad=True)
    _bias = torch.tensor(0.5, requires_grad=True)
    # 2.
    _loss = ((_X @ _w + _bias - _y) ** 2).mean()
    print(_loss)
    # 3.
    _w.grad = None
    _bias.grad = None
    _loss.backward()
    # 4.
    print(_w.grad, _bias.grad)
    with torch.no_grad():
        _w -= 0.1 * _w.grad
        _bias -= 0.1 * _bias.grad
    print(_w, _bias)
    print(((_X @ _w + _bias - _y) ** 2).mean().item())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `c` en `d` verschijnen met hun `grad_fn`: `MulBackward0` en `AddBackward0`. Na `d.backward()` is `a.grad` 4 en `b.grad` 2, zoals in de figuur. Het lineaire model begint met de loss 0,4375 uit [Tensors in PyTorch](/course/open/torch-tensors?lang=nl). `w.grad` bevat één helling per weight: 0,75 voor de eerste, en 0 voor de tweede, omdat hij sommige voorspellingen precies zoveel beter maakt als andere slechter. De stap zet `w` op (0,925; 1,5) en de bias term op 0,475, en de loss daalt naar ongeveer 0,4009. Na de stap zegt `w` nog steeds `requires_grad=True`: het is dezelfde leaf tensor, klaar voor de volgende stap.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_neuron():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het neuron opnieuw

        Het neuron van oefening 3 in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl): inputs 2 en 1, label 3, weights 0,5 en −1, en bias term 1. Nu vormen zijn twee weights samen één tensor `w`, en zijn inputs één tensor `x`. Vervang de `...` door de forward pass, die eindigt in `loss`, en dan `loss.backward()`. De laatste regel laat de hellingen zien. De check probeert ook andere weights en een andere bias term, één ervan met `z` onder 0.
        """),
        L.task("Laat autograd de hellingen van de loss naar `w` en `b` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neuron():
    box_neuron = L.code_box("neuron", lines=11)
    box_neuron
    return (box_neuron,)


@app.cell(hide_code=True)
def check_neuron(box_neuron):
    result_neuron = L.check("neuron", box_neuron.value)
    result_neuron
    return (result_neuron,)


@app.cell(hide_code=True)
def hints_neuron():
    L.hints(
        "neuron",
        [
            "Stap 2 en 3 van het recept. De formule staat in *Het idee*.",
            "`w @ x` is het inproduct van de weights en de inputs, en ReLU op een tensor is `torch.relu`. De `.grad`s zijn hier nog `None`, dus vóór deze ene backward pass hoeft er niets op `None`.",
            "`z = w @ x + ___`, `y_pred = torch.relu(z)`, `loss = (y_pred - y) ** 2` en `loss.___()`.",
        ],
        why="$z = 0{,}5 \\cdot 2 - 1 \\cdot 1 + 1 = 1$, dus $\\hat{y} = 1$ en $\\hat{y} - y = -2$. De helling van de loss naar $z$ is $2 \\cdot (-2) = -4$, en voor elke weight is dat keer zijn input: $-4 \\cdot 2 = -8$ en $-4 \\cdot 1 = -4$. Voor $b$ is het $-4$ zelf.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.md(r"""
    ### Oefening 2. Een stap die PyTorch weigert
    """)
    return


@app.cell(hide_code=True)
def choice_step_cell():
    choice_step = L.choice(
        "step",
        question="Een trainingsstap `w -= lr * w.grad` laat PyTorch stoppen met \"a leaf Variable that requires grad is being used in an in-place operation\". Wat lost het op?",
        options={
            "rebind": "`w = w - lr * w.grad` in plaats van `-=`",
            "backward": "Nog een keer `loss.backward()` aanroepen vóór de stap",
            "right": "Dezelfde regel, binnen `with torch.no_grad():`",
        },
        correct="right",
        why={
            "rebind": "Dat loopt één keer, maar dan is `w` de uitkomst van een berekening, geen leaf tensor: de volgende `loss.backward()` geeft hem geen `.grad`. Zie het let-op in *Het idee*.",
            "backward": "Een tweede backward pass over dezelfde graaf laat PyTorch ook stoppen, en de stap zou nog steeds worden opgenomen. Wat houdt de stap buiten de graaf? Zie *Het idee*.",
            "right": "Binnen `torch.no_grad()` neemt PyTorch niets op, dus mag je de parameter zelf veranderen. Hij blijft dezelfde leaf tensor, met `requires_grad=True`.",
        },
    )
    choice_step
    return (choice_step,)


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een training loop met autograd

        Het lineaire model van het uitgewerkte voorbeeld, getraind vanaf `w` en `b` op 0, in 100 stappen met de learning rate `lr`. De forward pass staat er al. Vervang de `...` door de rest van elke stap: stap 3 en 4 van het recept, voor `w` en `b`. De laatste regel is de loss van de laatste stap, als Python-getal. De check probeert ook een andere learning rate.
        """),
        L.task("Laat elke stap van de loop de gradients op nul zetten, de backward pass doen en een stap tegen de hellingen in zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train", lines=17)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train",
        [
            "Stap 3 en 4 van het recept, voor zowel `w` als `b`, zoals aan het eind van het uitgewerkte voorbeeld.",
            "Binnen de loop gaan de regels van het `with`-blok één niveau dieper. Gebruik `-=`, zodat `w` dezelfde leaf tensor blijft.",
            "`w.grad = None`, `b.grad = None` en `loss.___()`; dan `with torch.no_grad():` met `w -= lr * w.grad` en `b -= ___`.",
        ],
        why="Elke stap begint zonder oude hellingen, krijgt nieuwe van `loss.backward()`, en schuift `w` en `b` een klein stukje ertegenin. De loss gaat van 21,5, met alles op 0, naar ongeveer 0,33.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Autograd is je `Value` en je `backward` uit [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl), ingebouwd in de tensors van [Tensors in PyTorch](/course/open/torch-tensors?lang=nl): `requires_grad` neemt op, `backward()` loopt terug, `.grad` telt op, en de stap gebeurt binnen `torch.no_grad()`, dat ook een decorator is, zoals in [Decorators](/course/open/py-decorators?lang=nl). In de volgende les, *Layers als modules: nn.Linear*, houdt PyTorch ook de weights en de forward pass van een layer bij elkaar, zoals je `Layer` en `MLP` deden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_neuron, choice_step, result_train):
    L.wrap_up(result_neuron, choice_step, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
