# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-slope-formulas. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-slope-formulas", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De afgeleide als helling](/course/open/math-slope?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="De loss is 10,000 bij $w = 2$, en 9,996 bij $w$ = 2,001. Welke helling meet je, met $h$ = 0,001?",
        options={
            "change": "−0,004",
            "right": "−4",
            "flipped": "4",
        },
        correct="right",
        why={
            "change": "Dat is de verandering van de loss zelf. Wat doe je daarmee in stap 3 van het recept? Zie [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "flipped": "De loss daalde. Welk teken krijgt de helling dan? Zie *Het idee* in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "right": "De verandering is 9,996 − 10,000 = −0,004, en gedeeld door 0,001 is dat −4.",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def quick_check_way():
    quick_way = L.choice(
        "quick-way",
        question="De helling van de loss bij jouw $w$ is −4. Wat verlaagt de loss?",
        options={
            "bigger": "$w$ groter maken",
            "smaller": "$w$ kleiner maken",
            "stay": "$w$ laten zoals hij is",
        },
        correct="bigger",
        why={
            "smaller": "Een negatieve helling: wat doet de loss als $w$ groter wordt? Zie het recept in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "stay": "Bij het laagste punt is de helling 0. Is hij hier 0? Zie *Het idee* in [De afgeleide als helling](/course/open/math-slope?lang=nl).",
            "bigger": "Een negatieve helling betekent dat de loss daalt als $w$ groter wordt.",
        },
    )
    quick_way
    return (quick_way,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een helling meten kost twee berekeningen, en de uitkomst klopt maar ongeveer: hij hangt een beetje af van $h$. Voor eenvoudige functies kan het beter: een korte formule die meteen de precieze helling geeft. Een model leert miljoenen keren van hellingen, dus het moet precies én snel.

    **Een voorbeeld uit het dagelijks leven.** Een auto trekt op vanuit stilstand: na $t$ seconden heeft hij $t^2$ meter afgelegd. Meet je bij $t = 3$ de volgende 0,1 seconde, dan legt hij 0,61 m af: je meet 6,1 m/s. De snelheidsmeter geeft precies 6 m/s, omdat de snelheid een formule volgt: $2t$.
    """)
    return


@app.cell(hide_code=True)
def predict_pattern():
    guess_pattern = L.guess(
        "pattern",
        question="Je meet de helling van $x^2$ bij $x = 1$, bij $x = 2$ en bij $x = 3$, met een heel kleine $h$. Wat denk je dat eruit komt?",
        options={
            "same": "Steeds 2",
            "double": "2, 4 en 6",
            "square": "1, 4 en 9",
        },
        reveal="**2, 4 en 6: twee keer $x$.** De helling van $x^2$ volgt een korte formule: $2x$. Controleer dat hieronder.",
    )
    guess_pattern
    return (guess_pattern,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Bij veel functies volgt de helling een korte formule: de hellingsformule. Voor een macht van $x$ geeft de machtsregel hem: de macht, $n$, komt naar voren, en de nieuwe macht is $n - 1$.
        """),
        L.formula(r"f(x) = x^n \quad\Longrightarrow\quad [[grad:f'(x)]] = n \cdot x^{n-1}"),
        mo.md(r"""
        Dus $x^2$ heeft de helling $2x$, en $x^3$ de helling $3x^2$. Een rechte lijn $a \cdot x$ heeft overal dezelfde helling: $a$, zoals de weight $w$ van een model de helling van zijn lijn is.

        | Functie | Hellingsformule | Helling bij $x = 3$ |
        |---|---|---|
        | $x^2$ | $2x$ | 6 |
        | $x^3$ | $3x^2$ | 27 |
        | $a \cdot x$, zoals $5x$ | $a$, zoals 5 | 5 |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **hellingsformule** | | een korte formule die de precieze helling van een functie bij elke $x$ geeft, zoals $2x$ bij $x^2$ | `2 * x` |
        | **machtsregel** | | de hellingsformule van $x$ tot de macht $n$: $n$ keer $x$ tot de macht $n - 1$ | bij $x^3$: `3 * x ** 2` |
        | $f'(x)$ | f-accent van x | de helling van de functie $f$ bij $x$ | |

        Verschuif de slider, of sleep over de grafiek. De kromme is $x^2$; het <span class="c-grad">groene lijntje</span> heeft de helling ervan bij de $x$ die je koos.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_x_cell():
    def _frame(x):
        return (C.picture(x, {"title": "De helling van x²",
                              "alt": f"De kromme van x in het kwadraat van −3 tot 3, een dal met het laagste punt bij x = 0. Bij x = {L.number(x, 1)} staat een punt, met een kort groen lijntje dat de helling daar heeft: {L.number(2 * x, 1)}."}),
                f"""Bij <i>x</i> = {L.number(x, 1)}: gemeten met <i>h</i> = 0,001 is de <span class="c-grad">helling</span> <span class="c-grad ea-num">{L.number(C.measured(x), 3)}</span>; de formule 2<i>x</i> geeft <span class="c-grad ea-num">{L.number(2 * x, 3)}</span>.""")

    explore_x = L.explore("slope-x", "x", _frame, start=1.5, low=-3, high=3, step=0.5, role="", task="Probeer een paar waarden van $x$ en vergelijk de twee hellingen onder de grafiek: hoeveel schelen ze?")
    explore_x
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij $x^2$ is de gemeten helling steeds precies $h$ te hoog, hier 0,001: dichtbij, maar niet precies. De formule is precies.

    ### Recept

    Zo vind je een helling met een formule:

    1. Schrijf de functie als een macht van $x$, zoals $x^2$ of $x^3$, of als een lijn $a \cdot x$.
    2. Gebruik de regel: $x^n$ heeft de helling $n \cdot x^{n-1}$; $a \cdot x$ heeft de helling $a$.
    3. Vul $x$ in.
    4. Controleer hem door te meten, met een heel kleine $h$: de twee horen dicht bij elkaar te liggen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De helling van $x^2$ bij $x = 3$, uit de formule en gemeten. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_formula():
    def _f(x):
        return x ** 2

    _x = 3
    # 1. en 2. x² is x tot de macht 2: de 2 komt naar voren, de macht wordt 1
    # 3. vul x in
    print(2 * _x)
    # 4. controleer hem door te meten, met een heel kleine h
    _h = 0.001
    print((_f(_x + _h) - _f(_x)) / _h)
    # een lijn, 5x, gemeten bij dezelfde x: de helling is 5
    print((5 * (_x + _h) - 5 * _x) / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De formule geeft precies 6, de meting 6,000999…: dichtbij, dus de formule doorstaat de controle. De meting is 0,001 te hoog, zoals bij de slider: bij $x^2$ meet een stap van $h$ steeds $2x + h$. De lijn $5x$ meet 4,99999…: bij een lijn is meten precies, op de laatste decimalen na, en de helling is 5 bij elke $x$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_x4():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De helling van x⁴
    """)
    return


@app.cell(hide_code=True)
def choice_x4_cell():
    choice_x4 = L.choice(
        "x4",
        question="Wat is de hellingsformule van $x^4$?",
        options={
            "4x": "$4x$",
            "x3": "$x^3$",
            "right": "$4x^3$",
        },
        correct="right",
        why={
            "4x": "Houdt $x$ een macht? Zie de machtsregel in *Het idee*: de macht gaat één omlaag, niet naar één.",
            "x3": "De macht ging één omlaag, maar wat komt er naar voren? Zie de machtsregel in *Het idee*.",
            "right": "De machtsregel: de 4 komt naar voren, en de macht gaat één omlaag: $4x^3$.",
        },
    )
    choice_x4
    return (choice_x4,)


@app.cell(hide_code=True)
def text_exercise_power():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De machtsregel als functie

        Maak `power_slope(n, x)` af. Hij geeft de helling van $x$ tot de macht $n$ bij $x$ terug, volgens de machtsregel. Vervang de `...`. De check roept `power_slope(3, 2)` aan: de helling van $x^3$ bij $x = 2$, en dat moet `12` geven. De check probeert ook andere machten en andere $x$.
        """),
        L.task("Laat `power_slope` de helling volgens de machtsregel teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_power():
    box_power = L.code_box("power", lines=4)
    box_power
    return (box_power,)


@app.cell(hide_code=True)
def check_power(box_power):
    result_power = L.check("power", box_power.value)
    result_power
    return (result_power,)


@app.cell(hide_code=True)
def hints_power():
    L.hints(
        "power",
        [
            "De machtsregel in *Het idee*: wat komt er naar voren, en wat gebeurt er met de macht?",
            "In Python is $x$ tot een macht `x ** ...`. Let op welk deel bij de nieuwe macht hoort.",
            "`return n * x ** (___)`.",
        ],
        why="`n * x ** (n - 1)`: de oude macht $n$ naar voren, en de macht één lager. Door de haakjes rekent Python eerst `n - 1` uit. `power_slope(3, 2)` is 3 · 2² = 12.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Controleer een bewering

        Iemand zegt: "De hellingsformule van $x^3$ is $3x$." Controleer dat door te meten. Het vak heeft de functie `f`, die $x^3$ geeft, het punt $x = 2$ en een heel klein stapje `h`. Zet de helling van `f` bij `x` in `measured`, en vergelijk hem met wat $3x$ geeft. Vervang de `...`. De laatste regel is `measured`, zodat de check hem ziet. De check probeert ook een andere $x$.
        """),
        L.task("Zet de gemeten helling van `f` bij `x` in `measured`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_claim():
    box_claim = L.code_box("claim", lines=9)
    box_claim
    return (box_claim,)


@app.cell(hide_code=True)
def check_claim(box_claim):
    result_claim = L.check("claim", box_claim.value)
    result_claim
    return (result_claim,)


@app.cell(hide_code=True)
def hints_claim():
    L.hints(
        "claim",
        [
            "Meet zoals in stap 4 van het uitgewerkte voorbeeld: `f` waar je staat, `f` een heel klein stapje verder, en dan delen.",
            "Hier is de functie `f` en het punt `x`. Waardoor deel je de verandering?",
            "`measured = (f(x + h) - ___) / h`.",
        ],
        why="`(f(x + h) - f(x)) / h` meet ongeveer 12,006 bij $x = 2$. De bewering $3x$ geeft 6, ver ernaast; de machtsregel geeft $3x^2$, en 3 · 2² = 12. De meting laat zien dat de bewering niet klopt.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De afgeleide als helling](/course/open/math-slope?lang=nl) mat je hellingen met een heel klein stapje. Nu weet je dat bij machten van $x$ een formule meteen de precieze helling geeft, en dat meten met zo'n stapje een goede manier is om een formule te controleren. Een loss is uit zulke stukken opgebouwd; latere lessen laten zien hoe je hun hellingen samenvoegt. In de volgende les, *Bergaf rollen: gradient descent*, vertelt de helling een model welke kant het zijn parameter op moet schuiven, stap na stap.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_x4, result_claim, result_power):
    L.wrap_up(choice_x4, result_power, result_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
