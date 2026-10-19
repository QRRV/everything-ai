# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nn-tuning-playbook. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nn-tuning-playbook", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Adam](/course/open/opt-adam?lang=nl) en [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_correction():
    quick_correction = L.choice(
        "quick-correction",
        question="Het gemiddelde $m$ van Adam begint bij 0, met $\\beta_1 = 0{,}9$. De eerste gradient is 2. Wat zijn $m$ en zijn correctie $\\hat{m}$ na die stap?",
        options={
            "right": "$m = 0{,}2$ en $\\hat{m} = 2$",
            "both": "$m = 0{,}2$ en $\\hat{m} = 0{,}2$",
            "full": "$m = 2$ en $\\hat{m} = 2$",
        },
        correct="right",
        why={
            "both": "$m$ is $0{,}1 \\times 2 = 0{,}2$, maar de correctie deelt het door $1 - 0{,}9^1 = 0{,}1$. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "full": "$m = 0{,}9 \\times 0 + 0{,}1 \\times 2 = 0{,}2$: de start bij 0 trekt het omlaag; $\\hat{m}$ corrigeert dat. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "right": "$0{,}2 / (1 - 0{,}9) = 2$. Deze les strijkt losses op dezelfde manier glad.",
        },
    )
    quick_correction
    return (quick_correction,)


@app.cell(hide_code=True)
def quick_check_validation():
    quick_validation = L.choice(
        "quick-validation",
        question="Je `Trainer` bewaart de parameters van de epoch met de laagste validation loss. Waarom de validation loss, en niet de training loss?",
        options={
            "speed": "De validation loss is sneller uit te rekenen",
            "right": "De training loss blijft dalen als het model begint te overfitten",
            "same": "Ze zijn hetzelfde, dus allebei is goed",
        },
        correct="right",
        why={
            "speed": "Snelheid is de reden niet: het gaat om wat de loss zegt. Zie [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).",
            "same": "Op data waarop het model traint, blijft de loss dalen; op nieuwe data kan hij weer stijgen. Zie [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).",
            "right": "Deze les kiest instellingen op dezelfde manier: op de validation loss.",
        },
    )
    quick_validation
    return (quick_validation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Van alle instellingen van een trainingsrun telt de learning rate het meest: te groot en de loss ontploft, te klein en het model leert nauwelijks iets in de tijd die je hebt. Elke waarde met een volledige run proberen duurt lang. Een **LR range test** vindt in één korte run een redelijk bereik, en een kleine sweep kiest daarin de beste waarde. Daarna ga je door naar de volgende instelling, één tegelijk.
    """)
    return


@app.cell(hide_code=True)
def predict_best():
    guess_best = L.guess(
        "best-rate",
        question="Een MLP met één hidden layer van 128 neuronen leert de handgeschreven cijfers, één epoch lang, met SGD en momentum 0,9. Met welke van deze learning rates eindigt de validation loss het laagst?",
        options={
            "small": "0,001",
            "right": "0,1",
            "large": "1",
        },
        reveal="**0,1.** Met 0,001 is de validation loss na de epoch nog 2,14: nauwelijks lager dan aan het begin, 2,30. Met 1 zijn de stappen te groot, en eindigt hij rond 1,9. Met 0,1 eindigt hij op 0,34, en 0,03 en 0,3 komen in de buurt, met ongeveer 0,45 en 0,47. De drie opties liggen een factor duizend uit elkaar, en alleen de middelste leert goed in één epoch.",
    )
    guess_best
    return (guess_best,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: één run, groeiende learning rates.** Een range test traint één korte run waarin de learning rate bij elke stap met dezelfde factor groeit, van veel te klein tot veel te groot. Met $n$ stappen van $\eta_{\min}$ tot $\eta_{\max}$ gebruikt stap $i$
        """),
        L.formula(r"[[hyper:\eta_i]] = [[hyper:\eta_{\min}]] \left( \frac{[[hyper:\eta_{\max}]]}{[[hyper:\eta_{\min}]]} \right)^{i / (n - 1)}"),
        mo.md(r"""
        De loss van elke stap wordt genoteerd. Eerst zijn de stappen te klein en beweegt de loss nauwelijks; dan daalt hij; dan worden de stappen te groot en ontploft hij. De test stopt zodra de loss 4 keer zo groot is als aan het begin.

        **Stap 2: gladstrijken.** De loss van één mini-batch is onrustig. Een lopend gemiddelde, gecorrigeerd voor zijn start bij 0 zoals in Adam, laat de trend zien. Het loopt een paar stappen achter, dus zijn laagste punt ligt iets rechts van waar de losses van afzonderlijke stappen het laagst zijn.

        **Stap 3: aflezen.** De laagste gladgestreken loss ligt ongeveer waar het trainen ophoudt stabiel te zijn, dus een goede learning rate ligt daar of eronder: van ongeveer een tiende ervan tot dat punt zelf.

        **Stap 4: sweep.** Train in dat bereik een paar korte runs met waarden die een gelijke factor uit elkaar liggen, en houd die met de laagste validation loss. Verander alleen de learning rate, leg hem dan vast en ga door naar de volgende instelling. Verander je er twee tegelijk, dan weet je van geen van beide wat hij deed.

        Het plaatje: een range test op de handgeschreven cijfers, van 0,0001 tot 10 in 100 stappen. De rondjes zijn de losses van afzonderlijke stappen, de lijn hun gladgestreken loss, en de stip zijn laagste punt; de band loopt van een tiende van dat punt tot dat punt.
        """),
        mo.Html(C.range_picture({"title": "Een LR range test", "x": "learning rate", "y": "loss", "lowest": "de stip: laagst bij {lr}",
                                 "band": "sweep", "lang": L.lang,
                                 "alt": "De loss tegen de learning rate op een logaritmische schaal van 0,0001 tot 10. De loss van losse stappen verspreidt zich rond een gladgestreken lijn die tot ongeveer 0,003 rond 2,3 blijft, bij een learning rate van ongeveer 0,3 tot ongeveer 0,9 daalt, en dan omhoogschiet, waar de test stopt. Een lichte band loopt van een tiende van dat laagste punt tot dat punt."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **LR range test** | één korte run met een learning rate die elke stap met dezelfde factor groeit, om te zien waar het trainen werkt en waar het ontploft | |
        | **hyperparameter sweep**, of sweep | korte runs die elk één waarde van een instelling proberen, met al het andere gelijk | |
        | $\eta_{\min}$, $\eta_{\max}$, $n$ | de eerste en de laatste learning rate van de test, en het aantal stappen | `lr_min`, `lr_max`, `steps` |
        """),
        L.watch_out("Een range test hoort bij zijn opzet: het model, de batch size, de optimizer. Verander er één en doe hem opnieuw. En het laagste punt is niet de beste learning rate: het ligt dicht bij waar het trainen misgaat, dus blijf eronder."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vind je een learning rate:

    1. Range test: één korte run, de learning rate groeit elke stap met dezelfde factor, van veel te klein tot veel te groot; noteer elke loss.
    2. Strijk de losses glad met een lopend gemiddelde, gecorrigeerd zoals in Adam, en zoek de learning rate bij het laagste punt.
    3. Sweep een paar waarden die een gelijke factor uit elkaar liggen, van een tiende van die learning rate tot die learning rate zelf, elk in een korte run.
    4. Houd de waarde met de laagste validation loss, leg hem vast, en ga door naar de volgende instelling.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Learning rates die een gelijke factor uit elkaar liggen, van 0,01 tot 1 in 5 stappen: elk is de vorige maal dezelfde factor. De factor is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_factor():
    _lr_min, _lr_max, _n = 0.01, 1.0, 5
    _factor = 1.0  # maak af: de factor tussen buren, (_lr_max / _lr_min) ** (1 / (_n - 1))
    _lr = _lr_min
    for _i in range(_n):
        print(round(_lr, 4))
        _lr = _lr * _factor
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zoals hij is, is de factor 1 en drukt elke regel `0.01` af. Afgemaakt is de factor $100^{1/4} \approx 3{,}16$, en zijn de regels `0.01`, `0.0316`, `0.1`, `0.3162` en `1.0`: van $\eta_{\min}$ tot $\eta_{\max}$ in 4 gelijke factoren, gelijkmatig verdeeld op een logaritmische schaal. $i$ keer met de factor vermenigvuldigen is hetzelfde als de formule van stap 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_range():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De learning rates van een range test

        Maak `lr_range(lr_min, lr_max, steps)` af: de lijst van `steps` learning rates van stap 1, van `lr_min` tot `lr_max`, elk dezelfde factor groter dan de vorige. De check roept `lr_range(0.001, 1.0, 4)` aan, wat `[0.001, 0.01, 0.1, 1.0]` moet geven, en probeert andere bereiken en aantallen stappen.
        """),
        L.task("Laat `lr_range` `steps` learning rates geven die een gelijke factor uit elkaar liggen, van `lr_min` tot `lr_max`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_range():
    box_range = L.code_box("lr-range", lines=6)
    box_range
    return (box_range,)


@app.cell(hide_code=True)
def check_range(box_range):
    result_range = L.check("lr-range", box_range.value)
    result_range
    return (result_range,)


@app.cell(hide_code=True)
def hints_range():
    L.hints(
        "lr-range",
        [
            "De formule van stap 1, voor elke `i` van de loop, met `lr_min`, `lr_max` en `steps`.",
            "In Python is `a ** b` `a` tot de macht `b`, en de n van de formule is `steps`. Controleer je regel: bij `i = 0` moet hij `lr_min` geven, bij `i = steps - 1` `lr_max`.",
        ],
        why="`lrs.append(lr_min * (lr_max / lr_min) ** (i / (steps - 1)))`: bij `lr_range(0.001, 1.0, 4)` zijn de factoren 1, 10, 100 en 1000.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Lees de test af

        Maak `range_pick(lrs, losses, beta)` af: strijk `losses` glad met een lopend gemiddelde `avg` dat bij 0 begint, `avg = beta * avg + (1 - beta) * loss`, corrigeer het voor zijn start zoals Adam doet, en geef de learning rate uit `lrs` terug waar het gecorrigeerde gemiddelde het laagst is. De check roept `range_pick([0.001, 0.01, 0.1, 1.0], [2.3, 2.0, 1.2, 5.0], 0.5)` aan, wat 0,1 moet geven, en probeert een toevallige dip, een andere `beta`, en `beta = 0`.
        """),
        L.task("Laat `range_pick` de learning rate geven bij het laagste gecorrigeerde lopende gemiddelde van de losses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("range-pick", lines=9)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("range-pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "range-pick",
        [
            "Drie regels in de loop: het lopende gemiddelde, de correctie ervan, en de beste tot nu toe bewaren, zoals je `Trainer` de beste validation loss bewaart.",
            "De correctie bij stap `i` deelt door `1 - beta ** (i + 1)`: `i` telt vanaf 0, de stappen van Adam vanaf 1.",
        ],
        why="`avg = beta * avg + (1 - beta) * losses[i]`, `smooth = avg / (1 - beta ** (i + 1))`, en `if best is None or smooth < best: best, best_lr = smooth, lrs[i]`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw range test, en een sweep

    De cel hieronder doet de range test van het plaatje met je `lr_range`, leest hem af met je `range_pick`, en traint dan drie korte runs, elk één epoch: op een tiende van de learning rate die hij vond, op een derde ervan, en op die learning rate zelf. Zolang je functies nog niet goed zijn, vallen die van de course in.
    """)
    return


@app.cell(hide_code=True)
def demo_sweep(box_pick, box_range, result_pick, result_range):
    _range, _pick, _r_yours, _p_yours = C.functions_of(box_range.value if result_range.state == "correct" else None,
                                                       box_pick.value if result_pick.state == "correct" else None)
    _lrs = list(_range(C.LO, C.HI, C.STEPS))
    _losses = C.range_test(_lrs, 0)
    _found = _pick(_lrs[:len(_losses)], _losses, C.BETA)
    _values = (_found / 10, _found / 3, _found)
    _val = C.sweep(_values)
    _best = min(range(3), key=lambda k: _val[k])
    _who = ("je `lr_range` en je `range_pick`" if _r_yours and _p_yours else
            "je `lr_range` en de `range_pick` van de course" if _r_yours else
            "de `lr_range` van de course en je `range_pick`" if _p_yours else "de functies van de course")
    _rows = [f"| {C.rate(_v, L.lang)} | {C.number(_l, 3, L.lang)}{' (laagst)' if _k == _best else ''} |" for _k, (_v, _l) in enumerate(zip(_values, _val))]
    mo.md("\n".join([
        f"Met {_who} stopte de test na {len(_losses)} stappen, en ligt zijn laagste gladgestreken loss bij een learning rate van {C.rate(_found, L.lang)}. De sweep, elk één epoch:",
        "",
        "| learning rate | validation loss |",
        "|---|---|",
        *_rows,
        "",
        "De test bracht een zoektocht over vijf factoren tien terug tot één; drie korte runs kozen daarna de waarde.",
    ]))
    return


@app.cell(hide_code=True)
def text_exercise_sweep():
    mo.md(r"""
    ### Oefening 3. Welke waarden je sweept
    """)
    return


@app.cell(hide_code=True)
def choice_sweep_cell():
    choice_sweep = L.choice(
        "sweep-values",
        question="Je range test heeft zijn laagste gladgestreken loss bij een learning rate van 0,3. Welke drie waarden probeer je in de sweep?",
        options={
            "close": "0,29; 0,3 en 0,31",
            "above": "0,3; 1 en 3",
            "right": "0,03; 0,1 en 0,3",
        },
        correct="right",
        why={
            "close": "Die verschillen ongeveer 3%: drie runs die je bijna niets vertellen, terwijl de goede waarde een factor 3 lager kan liggen.",
            "above": "Boven het laagste punt steeg de loss van de test al, en kort daarna ontplofte hij: daar is het trainen instabiel.",
            "right": "Van een tiende van het laagste punt tot dat punt, een gelijke factor uit elkaar: steeds ongeveer 3.",
        },
    )
    choice_sweep
    return (choice_sweep,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De learning rate van [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl), nu gevonden in één korte run, met het gecorrigeerde gemiddelde van [Adam](/course/open/opt-adam?lang=nl) en de validation loss die je `Trainer` uit [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl) bijhoudt. Train je later een eigen model, dan is een range test een goedkope eerste run vóór de lange.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sweep, result_pick, result_range):
    L.wrap_up(result_range, result_pick, choice_sweep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
