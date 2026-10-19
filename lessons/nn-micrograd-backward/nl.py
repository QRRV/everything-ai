# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-micrograd-backward. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow recursion

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-backward", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl) en [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parents():
    quick_parents = L.choice(
        "quick-parents",
        question="Met je `Value`: `a = Value(2)`, `b = Value(3)` en `c = a * b`. Wat staat er in `c.parents`?",
        options={
            "own": "`[(a, 2), (b, 3)]`",
            "right": "`[(a, 3), (b, 2)]`",
            "plain": "`[a, b]`",
        },
        correct="right",
        why={
            "own": "Elke ouder staat in een paar met zijn local gradient. Hoe snel verandert $a \\cdot b$ als $a$ verandert? Zie *Het idee* in [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl).",
            "plain": "`parents` bevat paren: de waarde, en zijn local gradient. Zie *Het idee* in [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl).",
            "right": "`a` met het getal van `b`, 3, en `b` met het getal van `a`, 2: de local gradients van een product.",
        },
    )
    quick_parents
    return (quick_parents,)


@app.cell(hide_code=True)
def quick_check_base():
    quick_base = L.choice(
        "quick-base",
        question="Een recursieve functie roept zichzelf aan, maar heeft geen basisgeval. Wat gebeurt er als je hem aanroept?",
        options={
            "zero": "Hij geeft 0 terug",
            "forever": "Hij draait eindeloos door, en je moet hem zelf stoppen",
            "right": "Python stopt hem met een fout, na ongeveer duizend aanroepen",
        },
        correct="right",
        why={
            "zero": "Zonder basisgeval komt hij nooit bij een `return` met een waarde. Wat doet Python met aanroepen die nooit eindigen? Zie [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).",
            "forever": "Python houdt bij hoeveel aanroepen nog open staan, en geeft het bij een grens op. Zie *Het idee* in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).",
            "right": "Elke aanroep wacht op de volgende, en Python staat ongeveer duizend open aanroepen toe; daarna stopt Python met een fout.",
        },
    )
    quick_base
    return (quick_base,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl) riep je `backward_step()` zelf aan op elke waarde, in een volgorde die je uitwerkte, en de demo gebruikte een volgorde die de course schreef. Een netwerk met een hidden layer heeft honderden waarden, en elk nieuw model een andere graaf. Maar de volgorde zit al in de graaf: elke `Value` weet waar hij vandaan komt. In deze les krijgt je project een `backward()` die de volgorde zelf vindt, voor elke graaf. Dat is de vijfde stap van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           4, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        L.course_versions(),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Aankleden: sokken vóór schoenen, een shirt vóór een trui. Elke volgorde die zich aan die regels houdt, werkt. Om je uit te kleden neem je dezelfde volgorde achterstevoren: schoenen vóór sokken.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_order():
    guess_order = L.guess(
        "order",
        question="$c = a \\cdot b$ en $d = c + a$. Om een lijst te maken, begin je bij `d`: eerst zet je op dezelfde manier alles in de lijst waaruit `d` gemaakt is, en pas dan `d` zelf. Wat is de lijst volgens jou?",
        options={
            "start": "`[d, c, a, b]`",
            "twice": "`[a, b, c, a, d]`",
            "right": "`[a, b, c, d]`",
        },
        reveal="**`[a, b, c, d]`.** Elke waarde komt na de waarden waaruit hij gemaakt is, en elke waarde staat er één keer in: ook `a`, al zijn zowel `c` als `d` eruit gemaakt. In *Het idee* volg je stap voor stap hoe de lijst ontstaat.",
    )
    guess_order
    return (guess_order,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **topologische ordening** van een graaf is een lijst van zijn waarden waarin elke waarde komt na alle waarden waaruit hij gemaakt is. Een recursieve functie vindt er een, vanaf de loss:

        1. Staat de waarde al in de lijst, stop dan: dat is het basisgeval.
        2. Doe anders eerst hetzelfde voor elk van zijn ouders.
        3. Voeg dan de waarde zelf toe aan de lijst.

        Een waarde zonder ouders gaat er gewoon in. Om te onthouden welke waarden al bezocht zijn, houdt de functie een `set` bij, `seen`: `seen.add(v)` zet `v` erin, en `v in seen` vraagt of hij erin zit. Elke aanroep krijgt dezelfde `seen` en dezelfde `order`: elke aanroep heeft zijn eigen namen, zoals in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl), maar die wijzen allemaal naar één set en één lijst, dus wat de ene aanroep toevoegt, zien de andere ook.

        De backward pass neemt deze ordening achterstevoren: eerst de loss, en elke waarde pas na alle waarden waar hij in ging. Zo heeft een waarde zijn hele `grad` al binnen als hij die met `backward_step()` doorgeeft.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **topologische ordening** | | een lijst van de waarden van een graaf waarin elke waarde komt na de waarden waaruit hij gemaakt is | `order` |
        | `set()` | | een nieuwe, lege set | `build_order(loss, set(), order)` |
        | `seen.add(v)` | | zet `v` in de set `seen`; een waarde die er al in zit, blijft er één keer in | `seen.add(v)` |
        | `v in seen` | | `True` als `v` in de set `seen` zit | `if v in seen:` |
        | `reversed(order)` | | loopt een lijst door van het eind naar het begin, zonder hem te veranderen | `for v in reversed(order):` |
        """),
        L.watch_out("Elke waarde moet één keer in de lijst staan. Een waarde die in twee nodes gaat, zou twee keer bezocht worden; staat hij er twee keer in, dan draait zijn `backward_step()` twee keer en geeft hij zijn helling twee keer door. `seen` voorkomt dat, en in een grote graaf scheelt het ook de tijd om dezelfde waarden steeds opnieuw te bezoeken."),
        mo.md(r"""
        Schuif met de slider: hij volgt `build_order` vanaf `d` in $d = a \cdot b + a$. De titel van de figuur noemt elke waarde waarop `build_order` tot nu toe is aangeroepen, en een waarde krijgt zijn plek in de lijst zodra al zijn ouders er een hebben.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    _say = [
        "Begin bij `d`: `build_order(d)`. Die is gemaakt uit `c` en `a`, dus eerst `build_order(c)`; `c` is gemaakt uit `a` en `b`, dus eerst `build_order(a)`. De lijst is nog leeg.",
        "`a` is uit niets gemaakt: hij gaat erin, op plek 1. Terug in `c` is de volgende ouder `b`: `build_order(b)`.",
        "Ook `b` is uit niets gemaakt: plek 2. Nu staan beide ouders van `c` in de lijst.",
        "Dus `c` gaat erin, op plek 3. Terug in `d` is zijn andere ouder `a`: nog een keer `build_order(a)`. Maar `a` staat al in de lijst.",
        "Beide ouders van `d` staan in de lijst, dus `d` gaat erin, op plek 4. Achterstevoren is `d`, `c`, `b`, `a` de volgorde van de backward pass.",
    ]

    def _frame(step):
        step = int(step)
        return (C.picture(step, {"place": "plek", "title": f"build_order aangeroepen op {', '.join(C.CALLS[:C.CALLED[step]])}",
                                 "alt": f"De graaf met Values voor d = a · b + a: a en b gaan in c, c en a gaan in d. Na stap {step} van 4 is de lijst [{', '.join(C.ORDER[:step])}]."}).text,
                _say[step])

    explore_step = L.explore("step", "stap", _frame, start=0, low=0, high=4, step=1, role="", task="Hoe vaak wordt `build_order` aangeroepen, en welke aanroep stopt meteen, bij het basisgeval?")
    explore_step
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    `build_order` wordt vijf keer aangeroepen: voor `d`, `c`, `a`, `b`, en nog een keer voor `a`. Die laatste aanroep stopt meteen: `a` zit al in `seen`. Zonder `seen` zou `a` een tweede keer in de lijst komen: `[a, b, c, a, d]`.

    ### Recept

    Zo doe je de hele backward pass vanaf een loss:

    1. Maak een lege lijst `order` en een lege set `seen`.
    2. Vanaf de loss, recursief: sla een waarde over die in `seen` zit; zet hem anders in `seen`, doe eerst zijn ouders, en voeg hem dan toe aan `order`.
    3. Zet de `grad` van de loss op 1.
    4. Loop `order` achterstevoren door, en roep `backward_step()` aan op elke waarde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het hele recept op een kleine familie van waarden die hun `grad` doorgeven zoals hij is, zonder local gradients, zodat je de lijst ziet ontstaan en de `grad` ziet teruggaan. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_order():
    class _Node:
        def __init__(self, name, parents=()):
            self.name = name
            self.parents = parents
            self.grad = 0

        def backward_step(self):
            for _parent in self.parents:
                _parent.grad = _parent.grad + self.grad

    def _visit(v, seen, order):
        if v in seen:                 # 2. al bezocht: overslaan
            return
        seen.add(v)
        for _parent in v.parents:     # 2. eerst zijn ouders
            _visit(_parent, seen, order)
        order.append(v)               # 2. dan de waarde zelf

    _a = _Node("a")
    _b = _Node("b")
    _c = _Node("c", [_a, _b])
    _d = _Node("d", [_c, _a])
    # 1. een lege lijst en een lege set
    _order = []
    _visit(_d, set(), _order)
    _names = []
    for _v in _order:
        _names.append(_v.name)
    print(_names)
    # 3. de grad van de loss, d, is 1
    _d.grad = 1
    # 4. achterstevoren: elke waarde geeft zijn grad door
    for _v in reversed(_order):
        _v.backward_step()
        print(_v.name, _v.grad)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De lijst is `['a', 'b', 'c', 'd']`, zoals bij stap 4 hierboven: elke waarde na die waaruit hij gemaakt is. Achterstevoren geeft `d` zijn 1 door aan `c` en `a`, en pas daarna geeft `c` zijn 1 door aan `a` en `b`: `a` krijgt 1 langs elke weg terug vanaf `d`, samen 2. Deze waarden geven hun `grad` door zoals hij is; je `Value` vermenigvuldigt hem met de local gradient, en zijn ouders zijn paren, `(parent, local)`. Dus in je eigen functie haalt de loop ze uit elkaar, zoals `backward_step()` doet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke lijst is een topologische ordening?
    """)
    return


@app.cell(hide_code=True)
def choice_topo_cell():
    choice_topo = L.choice(
        "topo",
        question="De student van [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl): $m = w \\cdot x$, $\\hat{y} = m + b$, $r = 7 - \\hat{y}$ en $\\mathcal{L} = r^2$. Welke lijst is een topologische ordening van $w$, $x$, $b$, $m$, $\\hat{y}$, $r$ en $\\mathcal{L}$?",
        options={
            "right": "$b$, $w$, $x$, $m$, $\\hat{y}$, $r$, $\\mathcal{L}$",
            "loss": "$\\mathcal{L}$, $r$, $\\hat{y}$, $m$, $b$, $w$, $x$",
            "early": "$w$, $x$, $\\hat{y}$, $m$, $b$, $r$, $\\mathcal{L}$",
        },
        correct="right",
        why={
            "loss": "Dat is de volgorde van de backward pass: de ordening achterstevoren. Komt $\\mathcal{L}$ in een topologische ordening voor of na de waarden waaruit hij gemaakt is? Zie *Het idee*.",
            "early": "$\\hat{y}$ is gemaakt uit $m$ en $b$. Staan die allebei ervoor in deze lijst? Zie *Het idee*.",
            "right": "Elke waarde komt na die waaruit hij gemaakt is: $m$ na $w$ en $x$, $\\hat{y}$ na $m$ en $b$. Dat $b$ vooraan staat, maakt niet uit; meer dan één ordening is goed.",
        },
    )
    choice_topo
    return (choice_topo,)


@app.cell(hide_code=True)
def text_exercise_backward():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je backward()

        Maak `build_order` en `backward` af. Vervang in `build_order` de `...` door stap 2 van het recept voor de ouders van `v`: dat zijn paren, `(parent, local)`, zoals in `backward_step()`. Vervang in `backward` de `...` door stap 4. De check maakt grafen met je `Value` uit je project, roept je `backward` erop aan, en kijkt naar de hellingen, en naar de ordening die je `build_order` maakt. Klopt het, dan bewaart de check beide functies in je project, als `work/projects/nn/backward.py`.
        """),
        L.task("Laat `build_order` eerst de ouders in de lijst zetten, en laat `backward` `backward_step()` aanroepen in de omgekeerde volgorde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_backward():
    box_backward = L.code_box("backward", lines=16)
    box_backward
    return (box_backward,)


@app.cell(hide_code=True)
def check_backward(box_backward):
    result_backward = L.check("backward", box_backward.value)
    result_backward
    return (result_backward,)


@app.cell(hide_code=True)
def hints_backward():
    L.hints(
        "backward",
        [
            "Stap 2 van het recept, zoals `_visit` in het uitgewerkte voorbeeld, maar met paren; dan stap 4.",
            "De loop over `v.parents` geeft twee namen, zoals in `backward_step()`, en roept `build_order` aan op de eerste. Stap 4 loopt door `reversed(order)`.",
            "`for parent, local in v.parents:` met `build_order(___, seen, order)`; en `for v in reversed(___):` met `v.___()`.",
        ],
        why="Bij $d = a \\cdot b + a$ maakt `build_order` `[a, b, c, d]`, en achterstevoren geeft `d.backward_step()` `c` en `a` elk 1, daarna geeft `c.backward_step()` `a` er 3 bij en `b` 2. Dus `a.grad` is 4 en `b.grad` 2, de hellingen van $ab + a$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_neuron():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een neuron met twee inputs

        Een neuron met ReLU en twee inputs, voor één voorbeeld: $z = w_1 x_1 + w_2 x_2 + b$, $\hat{y} = \mathrm{ReLU}(z)$, en de loss $(\hat{y} - y)^2$. Het vak haalt je `Value` en je `backward` uit je project binnen. Vervang de `...` door de forward pass, met `+`, `*`, `relu()` en `Value(-1)` voor de min, en laat hem eindigen in `loss`. Dan doet één aanroep, `backward(loss)`, de rest. De laatste regel is `[w1.grad, w2.grad, b.grad]`, zodat de check het ziet. De check probeert ook andere getallen.
        """),
        L.task("Bouw de loss van het neuron met je Value, en krijg alle drie de hellingen met één aanroep van je backward."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neuron():
    box_neuron = L.code_box("neuron", lines=15)
    box_neuron
    return (box_neuron,)


@app.cell(hide_code=True)
def check_neuron(box_neuron, result_backward):
    result_backward  # check again when exercise 2 saves your module
    result_neuron = L.check("neuron", box_neuron.value)
    result_neuron
    return (result_neuron,)


@app.cell(hide_code=True)
def hints_neuron():
    L.hints(
        "neuron",
        [
            "Schrijf de forward pass bewerking voor bewerking, zoals in oefening 3 van de vorige les; je hebt niet meer voor elke waarde een naam nodig.",
            "$\\hat{y} - y$ is `y_pred + y * Value(-1)`, en het kwadraat is die waarde keer zichzelf.",
            "`z = w1 * x1 + w2 * x2 + b`, `y_pred = z.relu()`, `r = y_pred + y * Value(-1)`, `loss = ___ * ___`, dan `backward(___)`.",
        ],
        why="Met $x = (2, 1)$, $y = 3$, $w = (0{,}5; -1)$ en $b = 1$: $z = 1 - 1 + 1 = 1$, dus $\\hat{y} = 1$ en $\\hat{y} - y = -2$. Terug: $2 \\cdot (-2) = -4$ voor $\\hat{y}$, en $z$ ligt boven 0, dus $-4$ voor $z$; dan $-4 \\cdot 2 = -8$ voor $w_1$, $-4 \\cdot 1 = -4$ voor $w_2$, en $-4$ voor $b$.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder traint een netwerk met een hidden layer van 4 neuronen met ReLU op XOR, de taak die één neuron niet kon leren in [Wat één neuron niet kan leren](/course/open/nn-xor?lang=nl). Zijn 17 weights en bias terms zijn `Value`s van jou; elke stap bouwt de loss over de vier hoeken en roept je eigen `backward` aan, en schuift dan elke parameter tegen zijn helling in, 500 keer. De cel gebruikt je eigen modules, dus hij werkt zodra oefening 2 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_xor(result_backward):
    result_backward  # run again when exercise 2 saves your module
    _fit = C.trained()
    if _fit is None:
        _out = mo.md("*Maak eerst oefening 2 af: dan staat je `backward` in je project, en gebruikt deze cel hem.*")
    elif not _fit:
        _out = mo.md("*Je `backward` in `work/projects/nn/backward.py` geeft niet meer de goede hellingen, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart het dan opnieuw.*")
    else:
        _before, _after, _outputs, _size, _own = _fit
        _out = mo.md(f"""De loss over de vier hoeken ging van <span class="c-loss ea-num">{L.number(_before, 2)}</span> naar <span class="c-loss ea-num">{L.number(_after, 4)}</span>. Voor (0, 0), (0, 1), (1, 0) en (1, 1) geeft het netwerk nu <span class="c-pred ea-num">{"; ".join(L.number(_o, 2) for _o in _outputs)}</span>: XOR. Elke helling kwam uit {"je eigen `Value`" if _own else "de `Value` van de course"}, in een volgorde die je eigen `backward` vond, door een graaf van {_size} waarden per stap.""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Bouw je eigen autograd: een Value die onthoudt](/course/open/nn-micrograd-value?lang=nl) leerde elke `Value` zijn helling terug te geven; met de recursie van [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl) en een set uit [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl) vindt je `backward` nu zelf de volgorde van die stappen. Je project `nn` heeft nu ook `backward.py`. In de volgende les, *Neuronen, layers en een MLP bovenop je Value*, bouw je neuronen en layers als classes, zodat een netwerk van elke grootte een paar regels is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_topo, result_backward, result_neuron):
    L.wrap_up(choice_topo, result_backward, result_neuron)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
