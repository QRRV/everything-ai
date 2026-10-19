# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les opt-lr-schedules. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow cosine annealing

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("opt-lr-schedules", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [AdamW](/course/open/opt-adamw?lang=nl) en [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_half():
    quick_half = L.choice(
        "quick-half",
        question="AdamW laat eerst elke weight krimpen met de fractie $\\eta\\lambda$, en zet dan de stap van Adam, die ongeveer $\\eta$ groot is. Je halveert de learning rate $\\eta$. Wat gebeurt er?",
        options={
            "step-only": "Alleen de stap van Adam halveert: het krimpen hangt alleen van $\\lambda$ af",
            "right": "Het krimpen en de stap van Adam halveren allebei: ze gaan mee met $\\eta$",
            "same": "Niets: Adam schaalt elke stap toch al naar de grootte van de gradient",
        },
        correct="right",
        why={
            "step-only": "De fractie is $\\eta\\lambda$, dus de helft van $\\eta$ is ook de helft van het krimpen. Zie [AdamW](/course/open/opt-adamw?lang=nl).",
            "same": "Adam deelt door $\\sqrt{\\hat{s}}$ om de grootte van de gradient eruit te halen, niet de grootte van $\\eta$: elke stap is $\\eta$ keer $\\hat{m} / \\sqrt{\\hat{s}}$. Zie [Adam](/course/open/opt-adam?lang=nl).",
            "right": "Allebei gaan ze mee met $\\eta$: het krimpen is $\\eta\\lambda w$, en de stap is $\\eta$ keer $\\hat{m} / (\\sqrt{\\hat{s}} + \\varepsilon)$. Wie $\\eta$ tijdens het trainen verandert, verandert dus meteen elke stap.",
        },
    )
    quick_half
    return (quick_half,)


@app.cell(hide_code=True)
def quick_check_jumping():
    quick_jumping = L.choice(
        "quick-jumping",
        question="Een training loss springt op en neer, en groeit. Wat is de waarschijnlijkste oorzaak?",
        options={
            "right": "Een te grote learning rate: de stappen schieten door",
            "overfitting": "Overfitting: het model leert zijn voorbeelden uit het hoofd",
            "small": "Een te kleine learning rate: het model zit vast",
        },
        correct="right",
        why={
            "overfitting": "Overfitting zie je aan de validation loss, die stijgt terwijl de training loss daalt; de training loss gaat er niet van springen. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "small": "Een te kleine learning rate geeft een loss die bijna niet beweegt, niet een die springt. Zie [Loss curves lezen](/course/open/torch-loss-curves?lang=nl).",
            "right": "Te grote stappen schieten voorbij de bodem, en de loss springt. Een veel te grote learning rate kan de loss zelfs laten vastlopen op de loss van gokken. Vandaag zie je hoe je de learning rate tijdens het trainen verandert.",
        },
    )
    quick_jumping
    return (quick_jumping,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Eén learning rate voor het hele trainen is een compromis. Aan het begin zijn de weights willekeurig, en de gradients kunnen groot zijn en vreemde kanten op wijzen: grote stappen kunnen het trainen daar verpesten. In het midden maken grote stappen de meeste voortgang. Aan het eind geeft elke mini-batch een net iets andere gradient, en grote stappen laten de weights heen en weer stuiteren rond de bodem van het dal, in plaats van er tot rust te komen. Een **learning-rate schedule** verandert de learning rate tijdens het trainen, en bijna elk groot model wordt met een schedule getraind.
    """)
    return


@app.cell(hide_code=True)
def predict_warmup():
    guess_warmup = L.guess(
        "with-warmup",
        question="Het MLP van 10 blokken met residual connections uit [Residual connections](/course/open/nn-residual?lang=nl) traint één epoch met AdamW en een learning rate van 0,03, 30 keer de gebruikelijke 0,001 van Adam. Zijn loss schiet in de eerste 20 stappen omhoog en eindigt op 2,35, ongeveer de loss van gokken: het leest 131 van de 1.000 testcijfers goed. Nu begint de learning rate bij bijna 0 en loopt hij in de eerste 30 stappen gelijkmatig op naar 0,03, en daarna blijft hij 0,03. Wat denk je dat er gebeurt?",
        options={
            "later": "Het loopt nog steeds vast, alleen wat stappen later: de piek is dezelfde",
            "right": "Het traint: na één epoch leest het meer dan 800 testcijfers goed",
            "stuck": "Het leert bijna niets: de stappen aan het begin zijn te klein",
        },
        reveal="**Het traint: 827 van de 1.000 testcijfers goed.** Zonder warmup duwen de eerste stappen, even groot als alle andere, bijna elke ReLU van de eerste layer onder 0 voor elk plaatje: het worden dead neurons, en vanaf dan geeft het netwerk voor elk cijfer bijna hetzelfde antwoord. Met warmup zijn de eerste stappen klein terwijl het netwerk uit zijn willekeurige begin komt, en tegen de tijd dat ze groot zijn, kunnen de weights ze aan. Met een cosine decay na de warmup leest het er 869 goed.",
    )
    guess_warmup
    return (guess_warmup,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een schedule geeft de learning rate als functie van de stap $t$, van 0 tot de laatste stap $T - 1$. De schedule die nu het meest gebruikt wordt, begint met een **warmup**: in de eerste $W$ stappen loopt de learning rate gelijkmatig op van bijna 0 naar zijn piek $\eta_{\max}$:
        """),
        L.formula(r"[[hyper:\eta_t]] = [[hyper:\eta_{\max}]] \, \frac{t + 1}{W} \qquad (t < W)"),
        mo.md(r"""
        Daarna komt **cosine decay**: de learning rate zakt van de piek naar een minimum $\eta_{\min}$ langs een halve cosinusgolf, met $r = (t - W)/(T - W)$:
        """),
        L.formula(r"[[hyper:\eta_t]] = [[hyper:\eta_{\min}]] + \tfrac{1}{2} ([[hyper:\eta_{\max}]] - [[hyper:\eta_{\min}]]) \bigl(1 + \cos(\pi r)\bigr)"),
        mo.md(r"""
        $r$ loopt van 0 direct na de warmup tot 1 bij stap $T$, net na de laatste stap. Van $\cos$ heb je alleen de vorm tussen 0 en $\pi$ nodig: hij zakt van 1 naar −1, langzaam aan beide uiteinden en het snelst in het midden. Dus $\tfrac{1}{2}(1 + \cos(\pi r))$ zakt op dezelfde manier van 1 naar 0, en de learning rate zakt van de piek naar het minimum: eerst langzaam, dan sneller, en aan het eind weer langzaam, waar de weights tot rust komen. In Python geven `math.cos` en `math.pi` uit de module `math` je $\cos$ en $\pi$.

        Oudere schedules zijn eenvoudiger: **step decay** vermenigvuldigt de learning rate om de paar epochs met bijvoorbeeld 0,1, en **exponential decay** vermenigvuldigt hem bij elke stap met een factor als 0,99. De transformer van 2017 gebruikte een warmup, gevolgd door een learning rate die daalt als één gedeeld door de wortel uit de stap. Een derde soort kijkt naar de validation loss en verlaagt de learning rate als die niet meer beter wordt: `torch.optim.lr_scheduler.ReduceLROnPlateau`. In PyTorch houdt een optimizer zijn instellingen per parameter group bij, zoals in [AdamW](/course/open/opt-adamw?lang=nl): elke groep is een dict in de lijst `opt.param_groups`, en zijn learning rate is `group["lr"]`. Een scheduler zet die: `torch.optim.lr_scheduler` heeft `LinearLR` voor een warmup en `CosineAnnealingLR` voor de daling (PyTorch noemt het cosine annealing; na zijn laatste stap stijgt hij weer), `SequentialLR` zet ze achter elkaar, en je roept `scheduler.step()` aan na elke `opt.step()`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **learning-rate schedule** | | de learning rate als functie van de stap | `lr_at(step)` |
        | **warmup** | | de eerste stappen, waarin de learning rate oploopt van bijna 0 naar zijn piek | `peak * (step + 1) / warmup` |
        | **cosine decay** | | de learning rate die van de piek naar een minimum zakt langs een halve cosinusgolf | `0.5 * (1 + math.cos(math.pi * r))` |
        | $\cos$ | cosinus | de cosinus: tussen 0 en $\pi$ zakt hij van 1 naar −1, langzaam aan beide uiteinden | `math.cos(x)` |
        | $\eta_t$ | èta t | de learning rate bij stap $t$ | `group["lr"]` |
        | **step decay**, **exponential decay** | | de learning rate om de paar epochs, of bij elke stap, met een factor vermenigvuldigen | `lr * 0.1`, `lr * 0.99` |
        """),
        L.watch_out("De schedule heeft het totale aantal stappen $T$ vanaf het begin nodig: train je langer dan gepland, dan zit de cosinus al op zijn minimum. En de warmup en de piek horen bij elkaar: een warmup redt een veel te grote piek niet. Bij het pretrainen van grote modellen is de warmup meestal 1% van alle stappen of minder, bij fine-tuning een paar procent; de 30 van 94 hier is lang, omdat één epoch kort is."),
        mo.md(r"""
        De figuur: links de drie schedules voor één epoch van 94 mini-batches; na zijn warmup blijft de schedule met alleen warmup op de piek, op de lijn van de constante. De twee zonder daling zijn gestippeld, die met cosine decay is doorgetrokken, hier en rechts. Rechts de training loss van het netwerk uit de gok met elke schedule, gemiddeld over de laatste 10 mini-batches, op een logaritmische schaal. Met een constante 0,03 schiet de loss omhoog en eindigt hij op 2,35, en leest het netwerk 131 testcijfers goed. Met 30 stappen warmup eindigt hij op 0,36 en leest het er 827 goed. Met warmup en cosine decay zijn de laatste stappen klein, eindigt de loss iets lager, op 0,34, en leest het er 869 goed.
        """),
        mo.Html(C.schedule_picture({"lr_title": "Learning rate", "loss_title": "Training loss", "x": "mini-batch", "lr": "learning rate", "loss": "loss",
                                    "constant": "constant", "warmup": "warmup", "cosine": "+ cosinus", "both": "warmup, en warmup + cosinus",
                                    "alt": "Links: drie learning-rate schedules over 94 mini-batches: constant op 0,03; warmup van 0 naar 0,03 in 30 stappen, daarna constant; dezelfde warmup, daarna een cosinus omlaag naar 0. Rechts, op een logaritmische schaal, de training loss met elk ervan. Met de constante learning rate schiet hij omhoog tot ongeveer 30 en eindigt hij op ongeveer 2,3; de twee schedules met warmup zakken naar ongeveer 0,35, dicht bij elkaar."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een schedule met warmup en cosinus:

    1. Kies de piek $\eta_{\max}$, het aantal warmup-stappen $W$, het totale aantal stappen $T$, en een minimum $\eta_{\min}$, vaak 0 of een tiende van de piek.
    2. Stap $0$ tot en met $W - 1$: $\eta_t = \eta_{\max}\,(t + 1)/W$.
    3. Stap $W$ tot en met $T - 1$: $r = (t - W)/(T - W)$ en $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi r))$. Vanaf $T$: $\eta_{\min}$.
    4. Zet vóór elke stap van de optimizer $\eta_t$ in elke parameter group: `group["lr"] = ...`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De schedule van de figuur (piek 0,03, 30 warmup-stappen, 94 stappen in totaal, minimum 0), uitgerekend bij een paar stappen en bij één stap na het eind. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar dat zegt, en voer hem opnieuw uit. Zoals hij is, blijft de learning rate na de warmup op de piek.
    """)
    return


@app.cell
def example_schedule():
    import math

    # 1. de getallen van de schedule
    _peak = 0.03
    _warmup = 30
    _total = 94
    _minimum = 0.0
    for _t in [0, 14, 29, 40, 62, 90, 100]:
        # 2. warmup
        if _t < _warmup:
            _lr = _peak * (_t + 1) / _warmup
        # 3. vanaf _total: het minimum
        elif _t >= _total:
            _lr = _minimum
        # 3. maak af: met _r = (_t - _warmup) / (_total - _warmup), de cosine decay van _peak naar _minimum
        else:
            _lr = _peak
        print(_t, round(_lr, 6))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De warmup geeft 0,001 bij stap 0, 0,015 bij stap 14 en de volle 0,03 bij stap 29; stap 100, na het eind, krijgt het minimum, 0. Zoals hij is, blijft de cel daartussen op 0,03. Met de cosinus heeft stap 40 $r = 10/64 \approx 0{,}16$ en een learning rate van 0,028229, nog dicht bij de piek; stap 62 is halverwege, $r = 0{,}5$, waar $\cos(\pi/2) = 0$ precies de halve piek geeft, 0,015; en stap 90 heeft 0,000288, bijna niets.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_decay():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom aan het eind omlaag
    """)
    return


@app.cell(hide_code=True)
def choice_decay_cell():
    choice_decay = L.choice(
        "why-decay",
        question="Waarom maken bijna alle schedules de learning rate klein tegen het eind van het trainen?",
        options={
            "larger": "De gradients worden tegen het eind van het trainen groter, en kleinere stappen maken dat goed",
            "right": "Met kleine stappen komen de weights tot rust, in plaats van op gradients met ruis rond de bodem te stuiteren",
            "overfit": "Kleine stappen aan het eind maken de overfitting door de grote stappen ervoor ongedaan",
        },
        correct="right",
        why={
            "larger": "Bij de bodem van de loss worden de gradients meestal kleiner, niet groter. Wat blijft, is hun ruis van de ene mini-batch op de volgende.",
            "overfit": "Kleine stappen maken niets ongedaan; overfitting gaat over wat het model leert, en daar helpen een validation set en regularisatie tegen.",
            "right": "Elke mini-batch geeft een net iets andere gradient. Met een grote learning rate gooit elke stap met ruis de weights een stukje naar de ene of de andere kant van de bodem; kleinere stappen gooien minder ver, en de loss zakt verder. In de figuur eindigt warmup met cosine decay lager dan alleen warmup, en leest het 869 cijfers goed in plaats van 827.",
        },
    )
    choice_decay
    return (choice_decay,)


@app.cell(hide_code=True)
def text_exercise_lr():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De learning rate bij een stap

        Schrijf `lr_at(step, peak, warmup, total, minimum)`: de learning rate van de schedule met warmup en cosinus bij stap `step`, volgens stap 2 en 3 van het recept, ook voor stappen vanaf `total`. De check roept `lr_at(30, 0.002, 10, 50, 0.0)` aan, wat `0.001` moet geven, halverwege de daling, en probeert warmup-stappen, de laatste stappen, stappen na het eind, en een minimum boven 0.
        """),
        L.task("Laat `lr_at` de learning rate van de schedule met warmup en cosinus bij een stap geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lr():
    box_lr = L.code_box("lr-at", lines=10)
    box_lr
    return (box_lr,)


@app.cell(hide_code=True)
def check_lr(box_lr):
    result_lr = L.check("lr-at", box_lr.value)
    result_lr
    return (result_lr,)


@app.cell(hide_code=True)
def hints_lr():
    L.hints(
        "lr-at",
        [
            "Drie delen, elk met een eigen `return`: `if step < warmup:`, dan `if step >= total:`, en de rest.",
            "Voor de daling eerst `r = (step - warmup) / (total - warmup)`; dan stap 3 van het recept, met `math.cos` en `math.pi`: alleen het deel boven het minimum daalt.",
        ],
        why="`if step < warmup: return peak * (step + 1) / warmup`; `if step >= total: return minimum`; `r = (step - warmup) / (total - warmup)`; `return minimum + 0.5 * (1 + math.cos(math.pi * r)) * (peak - minimum)`. In het voorbeeld is $r = 20/40 = 0{,}5$ en $\\cos(\\pi/2) = 0$, dus $0{,}5 \\cdot 1 \\cdot 0{,}002 = 0{,}001$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scheduler():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een scheduler voor een optimizer

        Schrijf een class `WarmupCosine`, een scheduler zoals PyTorch ze heeft, met minimum 0. `WarmupCosine(opt, peak, warmup, total)` bewaart de optimizer en de getallen, zet een stappenteller op 0, en zet meteen de learning rate van stap 0 in elke groep van `opt.param_groups`. `step()`, aangeroepen na elke `opt.step()`, verhoogt de teller met één en zet de learning rate van die volgende stap in elke groep. Je mag je `lr_at` in het vak kopiëren en gebruiken. De check maakt een SGD-optimizer met twee parameter groups, hangt je scheduler eraan, en schrijft vóór elke stap de learning rate op, met een korte schedule en met nog een andere.
        """),
        L.task("Laat `WarmupCosine` vóór elke stap de learning rate van de schedule zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scheduler():
    box_scheduler = L.code_box("warmup-cosine", lines=22)
    box_scheduler
    return (box_scheduler,)


@app.cell(hide_code=True)
def check_scheduler(box_scheduler):
    result_scheduler = L.check("warmup-cosine", box_scheduler.value)
    result_scheduler
    return (result_scheduler,)


@app.cell(hide_code=True)
def hints_scheduler():
    L.hints(
        "warmup-cosine",
        [
            "Het zetten van de learning rate is in `__init__` en in `step` hetzelfde: een lus `for group in self.opt.param_groups:` met `group[\"lr\"] = ...`. Een eigen method daarvoor scheelt het twee keer schrijven.",
            "`step()` komt na een stap van de optimizer, dus eerst de teller één omhoog, dan de learning rate voor die nieuwe stand.",
        ],
        why="`__init__`: `self.opt = opt`, de drie getallen, `self.t = 0`, en de learning rate zetten. `step`: `self.t = self.t + 1`, en hem opnieuw zetten. Het zetten: voor elke groep `group[\"lr\"] = lr_at(self.t, self.peak, self.warmup, self.total, 0.0)`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het springen uit [Loss curves lezen](/course/open/torch-loss-curves?lang=nl) komt van te grote stappen, en de eerste stappen vanaf een willekeurig begin zijn het gevaarlijkst: in de gok maakten ze van de ReLU's van de eerste layer dead neurons, zoals in [In een trainingsrun kijken](/course/open/nn-activation-stats?lang=nl). Warmup houdt die stappen klein. Omdat de stappen van [AdamW](/course/open/opt-adamw?lang=nl) meegaan met zijn learning rate, zet een schedule direct de grootte van elke stap; AdamW met warmup en cosine decay is hoe de meeste transformers getraind worden. In de volgende les, *Gradient clipping*, bescherm je het trainen tegen het andere gevaar: één gradient die ineens enorm is, op elk moment van het trainen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_decay, result_lr, result_scheduler):
    L.wrap_up(choice_decay, result_lr, result_scheduler)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
