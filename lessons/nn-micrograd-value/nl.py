# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-micrograd-value. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nn-micrograd-value", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl) en [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_local():
    quick_local = L.choice(
        "quick-local",
        question="In een backward pass is $m = w \\cdot x$ met $x = 3$, en $\\frac{\\partial \\mathcal{L}}{\\partial m} = 2$. Wat is $\\frac{\\partial \\mathcal{L}}{\\partial w}$?",
        options={
            "sum": "5",
            "same": "2",
            "right": "6",
        },
        correct="right",
        why={
            "sum": "De backward pass telt de local gradient niet op; wat doet hij ermee? Zie het recept in [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl).",
            "same": "Dat is de helling van $m$. Wat is de local gradient van $m = w \\cdot x$ voor $w$? Zie *Het idee* in [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl).",
            "right": "De helling erna, 2, keer de local gradient $\\frac{\\partial m}{\\partial w} = x = 3$: 6.",
        },
    )
    quick_local
    return (quick_local,)


@app.cell(hide_code=True)
def quick_check_add():
    quick_add = L.choice(
        "quick-add",
        question="`a` en `b` zijn objecten van je eigen class. Welke method voert Python uit voor `a + b`?",
        options={
            "right": "`a.__add__(b)`",
            "other": "`b.__add__(a)`",
            "str": "`a.__str__()`",
        },
        correct="right",
        why={
            "other": "Het object links van `+` is `self`. Welke is dat hier? Zie [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl).",
            "str": "`__str__` draait voor `print` en f-strings. Welke method draait voor `+`? Zie [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl).",
            "right": "`+` voert `__add__` uit van het object links ervan, met het object rechts als `other`.",
        },
    )
    quick_add
    return (quick_add,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl) schreef je elke waarde en elke helling zelf uit. Bij een netwerk met duizenden weights moet de code die boekhouding zelf bijhouden: welke waarde uit welke gemaakt is, en met welke local gradient. Dat is **autograd**, en elke library die netwerken traint, heeft er een. In deze les bouw je de kern ervan in je project, zoals het kleine *micrograd* van Andrej Karpathy dat doet: een `Value` die onthoudt waar hij vandaan komt. Dat is de vierde stap van je eigen neural network.
        """),
        draw.project_steps(["Lineaire regressie met veel features", "Logistische regressie trainen", "Een lineaire classifier voor handgeschreven cijfers",
                            "Je eigen autograd: een Value die onthoudt", "De volledige backward pass", "Neuronen, layers en een MLP",
                            "Je MLP trainen", "Een MLP in PyTorch dat handgeschreven cijfers leest", "Backprop door hele layers met de hand",
                            "Een herbruikbare training loop", "Een diep netwerk laten trainen"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen neural network"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Een cel met een formule in een spreadsheet, zoals `=A1*B1`, onthoudt uit welke cellen hij gemaakt is. Verander je `A1`, dan weet de spreadsheet welke cellen ervan afhangen.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_grad():
    guess_grad = L.guess(
        "grad",
        question="`a = Value(2)`, `b = Value(3)` en `c = a * b`. Je zet `c.grad = 1`, en `c` geeft zijn helling terug aan de waarden waar hij vandaan komt, elk keer zijn local gradient. Wat is `a.grad` dan volgens jou?",
        options={
            "one": "1",
            "product": "6",
            "right": "3",
        },
        reveal="**3.** Bij $c = a \\cdot b$ is de local gradient voor $a$ het getal van $b$, 3: als $a$ 1 omhoog gaat, gaat $c$ 3 omhoog. Dus $a$ krijgt $1 \\cdot 3 = 3$, en $b$ krijgt $1 \\cdot 2 = 2$. Dat kan `c` alleen als hij `a` en `b` onthoudt, en die local gradients.",
    )
    guess_grad
    return (guess_grad,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een `Value` bevat een getal en onthoudt drie dingen:

        - `data`: zijn getal, uit de forward pass;
        - `grad`: de helling van de loss naar hem, $\frac{\partial \mathcal{L}}{\partial \text{waarde}}$, die begint bij 0;
        - `parents`: de waarden waaruit hij gemaakt is, elk in een paar met zijn local gradient.

        Een bewerking op twee `Value`s maakt een nieuwe, en vult meteen zijn `parents` in, nu de getallen er zijn. `__add__` draait voor `+` en `__mul__` voor `*`; `relu` is een gewone method:
        """),
        L.formula(r"\begin{aligned} a + b &: \text{ouders } (a, 1),\ (b, 1) \\ a \cdot b &: \text{ouders } (a, \text{het getal van } b),\ (b, \text{het getal van } a) \\ \mathrm{ReLU}(a) &: \text{ouder } (a, 1) \text{ als } a > 0, \text{ anders } (a, 0) \end{aligned}"),
        mo.md(r"""
        `backward_step` is één stap van de backward pass: een waarde geeft zijn `grad` terug aan elke ouder, keer de local gradient, en *telt* het op bij wat die ouder al heeft. Zo bouwt de berekening zelf de computational graph, en komen de hellingen langs zijn pijlen terug. Dat is **autograd**: automatische gradients. micrograd bewaart voor deze stap geen paren, maar per waarde een kleine functie; de hellingen komen hetzelfde uit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **autograd** | auto-grad | code die de computational graph opbouwt terwijl hij rekent, en daarna zelf de hellingen uitrekent | |
        | `Value` | | een getal dat zijn helling onthoudt en waar het vandaan komt | `Value(2)` |
        | `data`, `grad` | | het getal van een `Value`, en de helling van de loss naar hem | `a.data`, `a.grad` |
        | `parents` | | de waarden waaruit een `Value` gemaakt is, elk in een paar met zijn local gradient | `[(a, 1), (b, 1)]` |
        | `__mul__` | | de special method die Python voor `*` uitvoert, zoals `__add__` voor `+` | `def __mul__(self, other):` |
        | `backward_step` | | één stap van de backward pass: geef `grad` terug aan elke ouder, keer zijn local gradient | `c.backward_step()` |
        """),
        L.watch_out("Tel op bij `parent.grad`, vervang hem niet. Een waarde die in twee nodes gaat, of twee keer in één, zoals `r` in `r * r`, krijgt langs elke pijl een helling terug, en pas hun som is zijn helling."),
        mo.md(r"""
        Schuif met de slider: hij loopt door $d = a \cdot b + a$, met $a = 2$ en $b = 3$. Elke node laat zijn `data` en zijn `grad` zien; de <span class="c-grad">getallen bij de pijlen</span> zijn de local gradients die zijn `parents` bewaren. Een tabel heeft de `grad` van elke waarde na elke stap, één rij per stap.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    _say = [
        "Stap 0: de forward pass heeft elke `Value` gemaakt. Elke `Value` kent zijn ouders en hun local gradients, en elke `grad` is 0.",
        "Stap 1: `d.grad = 1`, de helling van `d` naar zichzelf.",
        "Stap 2: `d.backward_step()` geeft 1 × 1 aan `c` en 1 × 1 aan `a`: allebei hebben ze `grad` 1.",
        "Stap 3: `c.backward_step()` geeft 1 × 3 aan `a`, die het optelt bij de 1 die hij had: 4; en 1 × 2 aan `b`: 2. Inderdaad heeft $d = ab + a$ de hellingen $b + 1 = 4$ en $a = 2$.",
    ]

    def _frame(step):
        step = int(step)
        g = C.grads(step)
        return (C.picture(step, {"title": "Een graaf van Values, stap voor stap",
                                 "alt": f"Vier Value-nodes: a = 2 en b = 3 gaan in c = a · b = 6, met de local gradients 3 en 2; c en a gaan in d = c + a = 8, met de local gradients 1 en 1. Bij stap {step} van 3 zijn de grads: a {g[0]}, b {g[1]}, c {g[2]}, d {g[3]}."}).text
                + C.steps_table(step, {"label": "De grads na elke stap",
                                       "alt": f"Een tabel met de grads van a, b, c en d na elke stap, één rij per stap: na stap 0 zijn ze allemaal 0; na stap 1 is d 1; na stap 2 zijn a, c en d 1; na stap 3 is a 4, b 2, c 1 en d 1. De rij van stap {step} is gemarkeerd."}).text,
                _say[step])

    explore_step = L.explore("step", "stap", _frame, start=0, low=0, high=3, step=1, role="", task="Welke waarde krijgt twee keer een helling terug? En wat zou zijn `grad` worden als `backward_step` de `grad` verving in plaats van erbij op te tellen?")
    explore_step
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    $a$ krijgt twee keer een helling terug: 1 van $d$ bij stap 2, en 3 van $c$ bij stap 3, dus zijn `grad` is 4. Als `backward_step` de `grad` verving in plaats van erbij op te tellen, bleef van $a$ alleen de 3 over die als laatste kwam, en was zijn helling fout: 3 in plaats van 4.

    ### Recept

    Zo krijg je hellingen met je eigen `Value`:

    1. Maak van elk getal dat een helling moet krijgen een `Value`.
    2. Reken met `+`, `*` en `relu()`: elke nieuwe `Value` vult zijn `parents` in, met hun local gradients.
    3. Zet de `grad` van de loss op 1.
    4. Roep `backward_step()` aan op elke waarde, vanaf de loss terug, zodat een waarde zijn `grad` pas doorgeeft als hij hem helemaal heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een kleine `Value` die alleen kan optellen, op $d = (a + b) + a$. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_value():
    class _Value:
        def __init__(self, data, parents=()):
            self.data = data
            self.grad = 0
            self.parents = parents      # paren: (de Value waar hij vandaan komt, zijn local gradient)

        def __add__(self, other):
            return _Value(self.data + other.data, [(self, 1), (other, 1)])

        def backward_step(self):
            for parent, local in self.parents:
                parent.grad = parent.grad + local * self.grad

    # 1. de getallen die een helling krijgen
    _a = _Value(2)
    _b = _Value(3)
    # 2. reken: elke nieuwe Value onthoudt waar hij vandaan komt
    _c = _a + _b
    _d = _c + _a
    print(_d.data)
    # 3. de helling van d naar zichzelf
    _d.grad = 1
    # 4. vanaf d terug: eerst d, dan c
    _d.backward_step()
    _c.backward_step()
    print(_a.grad, _b.grad, _c.grad)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    $d = 2a + b = 7$. `d.backward_step()` geeft `c` en `a` elk 1. Dan geeft `c.backward_step()` `a` er nog 1 bij, dus 2, en `b` 1. Dat zijn de hellingen van $2a + b$: 2 voor $a$ en 1 voor $b$.

    Twee stukjes Python in deze class zijn nieuw. `parents=()` in de `def`-regel geeft `parents` een waarde voor als je hem weglaat: `_Value(2)` krijgt de lege tuple `()`, dus geen ouders. En `for parent, local in self.parents:` haalt elk paar uit elkaar, zoals `x, y = point` in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl): `parent` krijgt de eerste waarde van het paar, `local` de tweede.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. In welke volgorde?
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="Wat gebeurt er in het uitgewerkte voorbeeld als je `c.backward_step()` aanroept vóór `d.backward_step()`?",
        options={
            "same": "Er verandert niets: de hellingen komen hetzelfde uit",
            "right": "`b` krijgt 0, en `a` maar 1: `c` geeft zijn `grad` door terwijl die nog 0 is",
            "error": "Python stopt met een fout, omdat `c` vóór `d` gemaakt is",
        },
        correct="right",
        why={
            "same": "Wat is `c.grad` op dat moment? Zie stap 4 van het recept.",
            "error": "`backward_step` werkt op elke `Value`; hij geeft door welke `grad` hij ook heeft. Wat is die van `c` voordat `d` er iets aan doorgaf? Zie stap 4 van het recept.",
            "right": "`c.grad` is nog 0, dus `c` geeft 0 door aan `a` en `b`. Daarna geeft `d` `c` zijn 1, maar `c` is dan al aan de beurt geweest. Daarom maakt de volgorde uit: de volgende les vindt hem voor je.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_exercise_value():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je Value

        Maak de class `Value` af: `__init__` en `__add__` zijn klaar, zoals in het uitgewerkte voorbeeld. Vervang de drie `...`: `__mul__` geeft een nieuwe `Value` terug met het product, en beide ouders met hun local gradients; `relu` geeft een nieuwe `Value` terug met $\mathrm{ReLU}$ van het getal, en zijn ene ouder met local gradient 1 of 0; `backward_step` geeft `grad` terug aan elke ouder. De check maakt `Value`s, rekent ermee en zet backward steps, in een paar kleine grafen. Klopt je class, dan bewaart de check hem in je project, als `work/projects/nn/value.py`.
        """),
        L.task("Geef `Value` een `__mul__`, een `relu` en een `backward_step`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value():
    box_value = L.code_box("value", lines=21)
    box_value
    return (box_value,)


@app.cell(hide_code=True)
def check_value(box_value):
    result_value = L.check("value", box_value.value)
    result_value
    return (result_value,)


@app.cell(hide_code=True)
def hints_value():
    L.hints(
        "value",
        [
            "`__mul__` lijkt op `__add__`, met het product en andere local gradients; `backward_step` is die van het uitgewerkte voorbeeld.",
            "Bij $a \\cdot b$ is de local gradient voor `self` het getal van `other`, en voor `other` het getal van `self`. ReLU geeft boven 0 zijn getal door, met local gradient 1, en geeft op andere plekken 0, met local gradient 0.",
            "`return Value(self.data * other.data, [(self, ___), (other, ___)])`; in `relu`: `if self.data > 0:` met `return Value(self.data, [(self, 1)])`, en anders `return Value(0, [(self, ___)])`.",
        ],
        why="`a = Value(2)`, `b = Value(3)`, `c = a * b` maakt `c.data` 6, met de ouders `(a, 3)` en `(b, 2)`. Na `c.grad = 1` en `c.backward_step()` is `a.grad` 3 en `b.grad` 2. Bij `r * r` is `r` allebei de ouders, dus krijgt hij twee keer $r$ keer de helling van het product terug: samen $2r$ keer die helling.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_student():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De student, met je Value

        De student van [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl): $x = 4$ uur, het cijfer $y = 7$, bij $w = 1$ en $b = 1$. Het vak haalt je `Value` uit je project binnen: `from nn.value import Value` haalt de class `Value` uit het bestand `value.py` in je package `nn`, zoals `from math import sqrt` `sqrt` uit `math` haalt. Daarna doet het de forward pass, met een naam voor elke waarde. `y + neg` is $y - \hat{y}$: je `Value` kan niet aftrekken, maar wel $\hat{y} \cdot (-1)$ optellen. Vervang de `...` door de backward pass: de `grad` van de loss, en dan één `backward_step()` per waarde, vanaf de loss terug. De laatste regel is `[w.grad, b.grad]`, zodat de check het ziet. De check probeert ook andere getallen voor `w` en `b`.
        """),
        L.task("Doe de backward pass met je eigen Value, zodat `w.grad` en `b.grad` de hellingen van de loss zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_student():
    box_student = L.code_box("student", lines=21)
    box_student
    return (box_student,)


@app.cell(hide_code=True)
def check_student(box_student, result_value):
    result_value  # check again when exercise 2 saves your module
    result_student = L.check("student", box_student.value)
    result_student
    return (result_student,)


@app.cell(hide_code=True)
def hints_student():
    L.hints(
        "student",
        [
            "Stap 3 en 4 van het recept: eerst de `grad` van `loss`, dan de waarden van de laatst gemaakte terug naar de eerste.",
            "Elke waarde geeft zijn `grad` door als hij hem helemaal heeft. Welke waarde is als laatste gemaakt, en welke vlak daarvoor?",
            "`loss.grad = 1`, dan `loss.backward_step()`, `r.backward_step()`, `neg.backward_step()`, `___.backward_step()` en `___.backward_step()`.",
        ],
        why="Vanaf de loss terug: `loss`, `r`, `neg`, `y_pred` en `m`. `loss = r * r` geeft `r` de helling $2 \\cdot 2 = 4$, `neg` geeft $4 \\cdot (-1) = -4$ door aan `y_pred`, en `y_pred` geeft `m` en `b` elk −4. `m` geeft `w` $-4 \\cdot 4 = -16$: de hellingen −16 en −4 van de vorige les, nu uit je eigen `Value`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder traint $w$ en $b$ voor vijf studenten met je eigen `Value`: voor elke student dezelfde graaf als in oefening 3, een backward pass in dezelfde volgorde, en dan één stap van gradient descent op de hellingen van alle vijf samen, 300 keer. De cel gebruikt je eigen module, dus hij werkt zodra oefening 2 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_train(result_value):
    result_value  # run again when exercise 2 saves your module
    _fit = C.trained()
    if _fit is None:
        _out = mo.md("*Maak eerst oefening 2 af: dan staat je `Value` in je project, en gebruikt deze cel hem.*")
    elif not _fit:
        _out = mo.md("*De `Value` in `work/projects/nn/value.py` geeft niet meer de goede hellingen, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je class dan opnieuw.*")
    else:
        _w, _b, _before, _after = _fit
        _out = mo.vstack([
            mo.Html(C.fit_picture(_w, _b, {"title": "Vijf studenten, en de lijn die je Value trainde", "x": "uren gestudeerd", "y": "cijfer",
                                           "alt": f"Vijf punten, de uren en cijfers van vijf studenten, en een rechte lijn erdoor met w = {L.number(_w, 2)} en b = {L.number(_b, 2)}."})),
            mo.md(f"""Je `Value` vond $w$ = <span class="c-param ea-num">{L.number(_w, 2)}</span> en $b$ = <span class="c-param ea-num">{L.number(_b, 2)}</span>, en de mean squared error ging van <span class="c-loss ea-num">{L.number(_before, 2)}</span> naar <span class="c-loss ea-num">{L.number(_after, 2)}</span>. Elke helling kwam uit je eigen `backward_step`; de volgorde van de stappen kwam nog van de course. In de volgende les vindt je project die volgorde zelf."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl) hield je de boekhouding van de graaf zelf bij; nu doet elke `Value` dat, met de special methods van [Special methods: + en print laten werken op je eigen objecten](/course/open/py-special-methods?lang=nl). Je project `nn` heeft nu ook `value.py`. In de volgende les, *De volledige backward pass, in de goede volgorde*, krijgt je project een `backward()` die de volgorde van de stappen zelf vindt, voor elke graaf.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_value, result_student):
    L.wrap_up(choice_order, result_value, result_student)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
