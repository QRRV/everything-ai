# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-slope. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-slope", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_valley():
    quick_valley = L.choice(
        "quick-valley",
        question="De loss voor elke $w$ is een dal, met het laagste punt bij $w$ = 3,1. Je staat bij $w$ = 2,5. Wat gebeurt er met de loss als je $w$ een beetje groter maakt?",
        options={
            "up": "Hij stijgt",
            "down": "Hij daalt",
            "same": "Hij blijft gelijk",
        },
        correct="down",
        why={
            "up": "Aan welke kant van het laagste punt sta je? Zie de grafiek met de slider in [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl).",
            "same": "Is het dal vlak bij $w$ = 2,5? Zie de grafiek met de slider in [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl).",
            "down": "Je staat links van het laagste punt, dus een stap naar rechts gaat bergaf: de loss daalt.",
        },
    )
    quick_valley
    return (quick_valley,)


@app.cell(hide_code=True)
def quick_check_lowest():
    quick_lowest = L.choice(
        "quick-lowest",
        question="Wat vertelt het laagste punt van dat dal je?",
        options={
            "best": "De $w$ met de laagste loss: de lijn die het best bij de huizen past",
            "zero": "Waar de loss 0 is",
            "steep": "Waar de lijn van het model het steilst is",
        },
        correct="best",
        why={
            "zero": "Is de loss daar 0? Kijk naar het getal op de verticale as bij de bodem van het dal in [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl).",
            "steep": "Het dal laat de loss zien, niet de lijn van het model. Zie *Het idee* in [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl).",
            "best": "Bij het laagste punt geeft geen andere $w$ een lagere loss: die $w$ geeft de lijn die het best bij de huizen past.",
        },
    )
    quick_lowest
    return (quick_lowest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Passen met de hand is proberen: een stap de ene kant op, een stap de andere kant op, en de losses vergelijken. Dat lukt nog bij één parameter. Een groot model heeft er miljoenen, en voor elk ervan steeds beide kanten proberen duurt eindeloos. Je wilt één getal dat precies waar je staat zegt welke kant de loss op daalt, en hoe steil.

    **Een voorbeeld uit het dagelijks leven.** Je loopt in dichte mist een heuvel af. Het dal zie je niet, maar onder je voeten voel je welke kant de grond op helt, en hoe steil.
    """)
    return


@app.cell(hide_code=True)
def predict_steep():
    guess_steep = L.guess(
        "steeper",
        question="De loss voor elke $w$ is een dal, met het laagste punt bij $w$ = 3,1. Waar denk je dat het dal steiler is: bij $w$ = 2,5 of bij $w$ = 3,0?",
        options={
            "near": "Bij $w$ = 3,0",
            "far": "Bij $w$ = 2,5",
            "same": "Even steil",
        },
        reveal="**Bij $w$ = 2,5.** Ver van het laagste punt is het dal steil: een kleine stap verandert de loss veel. Dichterbij wordt het dal vlakker, en bij het laagste punt zelf is het vlak. Verschuif de slider hieronder en kijk naar het groene lijntje.",
    )
    guess_steep
    return (guess_steep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De afgeleide van de loss bij een $w$ zegt hoe snel de loss verandert als je $w$ een heel klein beetje verandert: de helling van het dal precies daar. Je kunt hem meten: verander $w$ met een heel klein stapje $h$, en deel de verandering van de loss door $h$.
        """),
        L.formula(r"[[grad:\mathcal{L}'(w)]] \approx \frac{[[loss:\mathcal{L}(w + h)]] - [[loss:\mathcal{L}(w)]]}{h}"),
        mo.md(r"""
        Het teken zegt welke kant de loss op gaat. Een positieve <span class="c-grad">helling</span>: de loss stijgt als $w$ groter wordt, dus een kleinere $w$ is beter. Een negatieve helling: een grotere $w$ is beter. Een helling dicht bij 0: je staat dicht bij het laagste punt. De grootte zegt hoe steil: de verandering van de loss gedeeld door de verandering van $w$. Een kleine stap in $w$ verandert de loss met ongeveer de helling keer die stap.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **afgeleide**, $\mathcal{L}'(w)$ | L-accent van w | hoe snel de loss verandert als je $w$ een heel klein beetje verandert: de helling van de loss bij die $w$ | `slope` |
        | **eindige differentie** | | een schatting van de afgeleide uit twee punten dicht bij elkaar: verander $w$ met een heel kleine $h$, en deel de verandering van de loss door $h$ | `(loss(w + h) - loss(w)) / h` |
        | $h$ | h | het heel kleine stapje waarmee je $w$ verandert | `h` |
        | $\approx$ | is ongeveer | ongeveer gelijk aan | |

        Verschuif de slider, of sleep over de grafiek. De <span class="c-loss">kromme</span> is de loss voor elke $w$, met $b = 50$, zoals in de vorige les. Het <span class="c-grad">groene lijntje</span> heeft de helling van de loss bij de $w$ die je koos.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_slope_cell():
    def _frame(w):
        return (C.picture(w, {"title": "De helling van de loss", "loss": "loss",
                              "alt": f"De loss voor elke w van 2 tot 4,2, een dal met het laagste punt bij w = 3,1. Bij w = {L.number(w, 1)} staat een punt, met een kort groen lijntje dat de helling van het dal daar heeft: {L.number(C.slope(w), 0)}."}),
                f'Bij <i>w</i> = {L.number(w, 1)} is de <span class="c-grad">helling</span> <span class="c-grad ea-num">{L.number(C.slope(w), 0)}</span>.')

    explore_slope = L.explore("slope-w", "w", _frame, start=2.5, low=2.0, high=4.2, step=0.1, task="Van $w$ = 2 tot 4,2: waar is de helling negatief, waar positief, en waar ongeveer 0?")
    explore_slope
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Links van het laagste punt is de <span class="c-grad">helling</span> negatief, en rechts ervan positief. Bij $w$ = 3,1, op de bodem, is hij 31: vergeleken met de duizenden ernaast is dat ongeveer 0. Hoe verder van de bodem, hoe steiler het dal, en hoe verder de helling van 0 af.
        """),
        L.note("Die eindige differentie ken je van school misschien als het *differentiequotiënt*."),
        mo.md(r"""
        ### Recept

        Zo meet je de helling van de loss voor een parameter:

        1. Reken de loss uit waar je staat.
        2. Verander de parameter met een heel klein stapje $h$, zoals 0,001, en reken de loss opnieuw uit.
        3. Deel de verandering van de loss door $h$: nieuwe loss min oude loss, en dat gedeeld door $h$.
        4. Lees het teken: positief, maak de parameter kleiner; negatief, maak hem groter.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De helling van de loss bij $w$ = 2,5, met $b = 50$, gemeten met drie stapjes $h$, elk tien keer zo klein als het vorige. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_slope():
    import numpy as np

    _area = np.array([62, 85, 110, 48, 95, 130, 73, 58])
    _price = np.array([245, 310, 395, 198, 320, 465, 280, 232])

    def _loss(w):
        _residuals = _price - (w * _area + 50)
        return (_residuals ** 2).mean()

    _w = 2.5
    for _h in [0.1, 0.01, 0.001]:
        # 1. en 2. de loss waar je staat, en een heel klein stapje verder
        _change = _loss(_w + _h) - _loss(_w)
        # 3. de verandering gedeeld door het stapje
        print(_h, _change / _h)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met $h$ = 0,1 komt de helling uit op ongeveer −8239; met 0,01 op −8916; met 0,001 op −8983. Elke keer dat $h$ tien keer zo klein wordt, schuift het antwoord ongeveer tien keer zo weinig op: eerst 677, dan 67. Zo komt het uit bij ongeveer −8990: de afgeleide bij $w$ = 2,5. Stap 4: de helling is negatief, dus de loss daalt als $w$ groter wordt. Een grotere $w$ is beter, zoals het dal liet zien. De grootte betekent: een stap van 0,01 in $w$ verlaagt de loss met ongeveer 90.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_way():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke kant is bergaf?
    """)
    return


@app.cell(hide_code=True)
def choice_way_cell():
    choice_way = L.choice(
        "which-way",
        question="Bij $w$ = 3,5, met $b = 50$, is de helling van de loss ongeveer 6040. Welke kant schuif je $w$ op om de loss te verlagen?",
        options={
            "up": "Maak $w$ groter",
            "stay": "Laat $w$ zoals hij is: je staat bij het laagste punt",
            "down": "Maak $w$ kleiner",
        },
        correct="down",
        why={
            "up": "Een positieve helling: wat doet de loss als $w$ groter wordt? Zie het teken in *Het idee*.",
            "stay": "Wat is de helling bij het laagste punt? Zie *Het idee*: ligt 6040 daar dichtbij?",
            "down": "De helling is positief: de loss stijgt als $w$ groter wordt. Dus een kleinere $w$ verlaagt de loss.",
        },
    )
    choice_way
    return (choice_way,)


@app.cell(hide_code=True)
def text_exercise_b():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De helling voor b

        In het vak hangt de loss van beide parameters af: `loss(w, b)`. Meet met $w = 3$ en $b = 50$ de helling van de loss voor $b$: hoe snel de loss verandert als je $b$ een heel klein beetje verandert. Zet hem in `slope_b`. Vervang de `...`. De laatste regel is `slope_b`, zodat de check hem ziet. De check probeert ook een andere $b$.
        """),
        L.task("Zet de helling van de loss voor b in `slope_b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_b():
    box_b = L.code_box("slope-b", lines=16)
    box_b
    return (box_b,)


@app.cell(hide_code=True)
def check_b(box_b):
    result_b = L.check("slope-b", box_b.value)
    result_b
    return (result_b,)


@app.cell(hide_code=True)
def hints_b():
    L.hints(
        "slope-b",
        [
            "Volg het recept: de loss waar je staat, de loss een heel klein stapje verder, en dan delen.",
            "De loss heeft nu twee parameters. Welke van de twee verander je met `h`, en welke blijft zoals hij is?",
            "`slope_b = (loss(w, ___) - loss(w, b)) / h`.",
        ],
        why="`loss(w, b + h)` is de loss een heel klein stapje verder in $b$, en `loss(w, b)` de loss waar je staat. Hun verschil gedeeld door `h` is de helling: ongeveer −15,5. Negatief, dus een grotere $b$ verlaagt de loss.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Eén stap bergaf

        Maak `next_w(w, b, step)` af. Hij doet één stap van het passen met de hand, maar met de helling in plaats van beide kanten te proberen. Hij meet de helling van de loss voor $w$, en geeft $w$ terug, met `step` verschoven in de richting die de loss verlaagt. Vervang de `...`. De check roept `next_w(2.5, 50, 0.1)` aan: dat moet `2.6` geven. De check probeert ook andere plekken.
        """),
        L.task("Laat `next_w` de w teruggeven die één stap bergaf ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next():
    box_next = L.code_box("next-w", lines=16)
    box_next
    return (box_next,)


@app.cell(hide_code=True)
def check_next(box_next):
    result_next = L.check("next-w", box_next.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next-w",
        [
            "Twee stappen: eerst de helling voor $w$, zoals in het recept; dan een stap ertegenin.",
            "Stap 4 van het recept zegt bij elk teken van de helling welke kant je op gaat. Met `if` kies je tussen de twee kanten.",
            "`slope = ___`, dan `if slope > 0:` met `return w ___ step`, en anders de andere kant op.",
        ],
        why="`(loss(w + h, b) - loss(w, b)) / h` meet de helling voor $w$. Is die positief, dan stijgt de loss als $w$ groter wordt, dus `w - step` gaat bergaf; anders `w + step`. Bij $w$ = 2,5 is de helling ongeveer −8983, dus het antwoord is 2,6.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een lijn passen met de hand](/course/open/ml-fit-by-hand?lang=nl) vond je het laagste punt door naar beide kanten een stap te proberen. Nu vertelt één getal, de afgeleide, je precies waar je staat welke kant de loss op daalt, en hoe steil. Meten met een heel klein stapje kost nog steeds twee losses per parameter. In de volgende les, *Hellingsformules die je kunt vertrouwen*, zie je dat bij eenvoudige functies de helling een korte formule volgt, die hem meteen geeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_way, result_b, result_next):
    L.wrap_up(choice_way, result_b, result_next)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
