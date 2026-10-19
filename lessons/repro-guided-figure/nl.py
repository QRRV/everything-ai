# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-guided-figure. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-figure", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Begeleide reproductie 3: sanity checks](/course/open/repro-guided-sanity?lang=nl) en [Figuren die iets laten zien](/course/open/res-figures?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_decay():
    quick_decay = L.choice(
        "quick-decay",
        question="Je sanity check leert één batch uit het hoofd zonder weight decay, en de runs van de paper gebruiken een weight decay van 1,0. Waarom laat de check hem weg?",
        options={
            "fast": "Weight decay maakt elke stap trager, en een sanity check moet in een minuut of twee klaar zijn",
            "right": "De check vraagt alleen of het model kan leren, en weight decay trekt de andere kant op",
            "paper": "Ook de paper laat weight decay weg in de eerste stappen van een run",
        },
        correct="right",
        why={
            "fast": "Weight decay kost bijna geen tijd: hij maakt elke weight bij elke stap een beetje kleiner. De check laat hem weg omdat hij ingaat tegen een batch uit het hoofd leren. Zie [Begeleide reproductie 3](/course/open/repro-guided-sanity?lang=nl).",
            "paper": "De runs van de paper gebruiken weight decay vanaf de eerste stap. De check laat hem weg omdat hij een andere vraag stelt dan de run. Zie [Begeleide reproductie 3](/course/open/repro-guided-sanity?lang=nl).",
            "right": "Weight decay houdt de weights klein, tegen wat één batch uit het hoofd leren nodig heeft. In de lange run van deze les is hij terug, op de 1,0 van de paper.",
        },
    )
    quick_decay
    return (quick_decay,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="Je hebt drie runs van één instelling, elk met een andere seed, en elk een lijn over de stappen. Wat tekent een figuur die iets laat zien?",
        options={
            "best": "De run die er het best uitziet, zodat de boodschap duidelijk is",
            "mean": "Het gemiddelde van de drie runs, als één lijn",
            "right": "Alle drie de runs, in één stijl, zodat de spreiding te zien is",
        },
        correct="right",
        why={
            "best": "Alleen de beste run verbergt hoeveel de runs verschillen, en laat het resultaat zekerder lijken dan het is. Zie [Figuren die iets laten zien](/course/open/res-figures?lang=nl).",
            "mean": "Een gemiddelde kan verbergen dat één run iets anders deed; teken bij drie runs ze allemaal. Zie [Figuren die iets laten zien](/course/open/res-figures?lang=nl).",
            "right": "Elke run, of de spreiding over seeds, hoort in de figuur: de lezer ziet meteen hoeveel van het resultaat de seed is.",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        De claim van de paper is een figuur: Figuur 1 van Power en anderen (2022), de accuracy op de trainings- en de validatiedata tegen de stappen op een logaritmische as, waarin validatie lang na training 99% haalt. Je plan zegt hoe je één run leest. Een reproductie is een eigen figuur, van dezelfde grootheden op dezelfde soort assen, getekend uit logboeken die je zelf hebt bijgehouden: dan kan iedereen hem naast die van de paper leggen en zien of hij hetzelfde laat zien. Drie dingen maken die figuur: een run die lang genoeg is om validatie te zien springen, logs op stappen die een logaritmische as gelijkmatig vullen, en meer dan één seed, want één run kan geluk zijn.

        **Een voorbeeld uit het dagelijks leven.** Een time-lapse van een plant die uit een zaadje groeit: in de eerste uren verandert er veel, later weinig per uur. Een camera die een foto maakt na 1, 2, 5, 10, 20, 50 en 100 uur legt elke fase vast met een paar foto's, en drie potten naast elkaar laten zien welke groei gewoon is en welke van één pot alleen.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de belangrijkste figuur",
                            "De ablation", "Vergelijk met de paper", "Het verslag"],
                           3, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_plateau():
    _r = C.read(C.course_runs()[0])
    guess_plateau = L.guess(
        "plateau-guess",
        question=f"In de run van de course met seed 0 haalt de training accuracy 99% bij stap {_r['t']}. Wat denk je dat de validation accuracy doet in de duizenden stappen voordat hij springt?",
        options={
            "chance": "Hij blijft op kansniveau: ongeveer 1%, één goed antwoord op 97",
            "flat": "Hij blijft vlak op ongeveer 30%",
            "climb": "Hij klimt langzaam en gestaag naar 99%",
        },
        reveal="**Hij blijft vlak op ongeveer 30%,** van ongeveer stap 300 tot ongeveer 5.000, en springt dan. In Figuur 1 van de paper blijft validatie juist laag; waarom, vraagt oefening 5.",
    )
    guess_plateau
    return (guess_plateau,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## Het idee

    Een **figuur opnieuw tekenen** is de figuur van de paper nog eens tekenen uit je eigen logboeken: dezelfde grootheden op dezelfde soort assen, zodat de twee naast elkaar kunnen liggen. Voor Figuur 1 is dat de accuracy op alle trainings- en alle validatievergelijkingen, tegen de stappen op een logaritmische as. Je kit krijgt er twee onderdelen voor.

    1. **Een logger**, `run_seed`: één run, die na gekozen stappen de accuracy op alle trainings- en alle validatievergelijkingen meet, en de stap en de twee accuracy's in een logboek bewaart. Alle 9.409 vergelijkingen meten duurt even lang als een aantal trainingsstappen, dus een run logt niet na elke stap. Op een logaritmische as nemen de stappen van 1 tot 10 evenveel ruimte in als die van 1.000 tot 10.000, dus liggen de stappen die hij logt gelijk verdeeld over die as: {C.PER_FACTOR} per factor 10, $10^{{k/{C.PER_FACTOR}}}$ afgerond, voor $k = 0, 1, 2, \dots$
    2. **Een multi-seed runner**, `run_seeds`: één run voor elke seed, elk als één regel toegevoegd aan een logboek. De seed legt de splitsing, de beginwaarden van de weights en de batches vast, dus is elke run een nieuwe trekking van alle drie. Een seed die al in het logboek staat, wordt niet opnieuw gedraaid: als je computer halverwege stopt, blijven de runs die klaar zijn klaar.

    Hieronder staat de run van de course met seed 0, met de instellingen van de paper bij {round(100 * C.FRACTION)}% van de tabel en {C.number(C.STEPS, 0, L.lang)} stappen. Hij laat de hele run zien. Speel je hem af, dan verschijnt het moment waar het plan naar zoekt zodra een lijn 99% haalt: $t$ voor training, $v$ voor validatie; aan het eind komt het oordeel van het plan.
    """)
    return


@app.cell(hide_code=True)
def play_replay_cell():
    _log = C.course_runs()[0]
    _r = C.read(_log)
    play_replay = play.Replay(
        steps=_log["step"], train=_log["train"], val=_log["val"], threshold=C.THRESHOLD, t_index=_r["t_index"], v_index=_r["v_index"],
        verdict=f"Het oordeel van het plan: <b>{_r['verdict']}</b>. Validatie haalt 99% {round(_r['times'])} keer zo laat als training, en het plan vraagt minstens {C.DELAY} keer.",
        decimal="," if L.lang == "nl" else ".",
        labels={"task_kick": L.w["task"], "task": "Speel de run af, of sleep erdoorheen. Bij welke stap haalt validatie 99%, en hoeveel keer zo laat als training is dat? Wat doet validatie daartussen?",
                "title": "Accuracy tijdens de run van de course, seed 0", "x_name": "stap", "train": "training", "val": "validatie",
                "play": "Afspelen", "pause": "Pauze", "back": "Terug", "fwd": "Vooruit", "step_name": "stap",
                "alt": f"De accuracy op de trainings- en de validatievergelijkingen tijdens de run van de course met seed 0, over {C.number(C.STEPS, 0, L.lang)} stappen op een logaritmische as. Training stijgt van ongeveer 1% naar 99% bij stap {_r['t']}; validatie blijft vlak op ongeveer 30% en haalt 99% bij stap {C.number(_r['v'], 0, L.lang)}."})
    mo.ui.anywidget(play_replay)
    return


@app.cell(hide_code=True)
def text_idea_after():
    _r = C.read(C.course_runs()[0])
    _low, _high = C.plateau(C.course_runs()[0])
    mo.vstack([
        mo.md(rf"""
        Training haalt 99% bij stap {_r['t']}, validatie bij stap {C.number(_r['v'], 0, L.lang)}: {round(_r['times'])} keer zo laat. Daartussen ligt validatie vlak, tussen {C.number(100 * _low, 0, L.lang)}% en {C.number(100 * _high, 0, L.lang)}% van stap 500 tot stap 4.000, terwijl training op 100% staat; daarna springt hij in een paar duizend stappen naar 100%. Dat is de claim van de paper, in een run met de instellingen van je eigen kit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **figuur opnieuw tekenen** | de figuur van de paper nog eens tekenen uit je eigen logboeken, met dezelfde grootheden op dezelfde soort assen | `plt.plot(log["step"], log["val"], "--")` |
        | **logger** | het deel van een run dat na gekozen stappen meet en bewaart wat de figuur nodig heeft | `log["val"].append(...)` |
        | logaritmisch verdeelde stappen | stappen die gelijk verdeeld op een logaritmische as liggen: evenveel van 10 tot 100 als van 1.000 tot 10.000 | `log_steps(12000, 20)` |
        | **multi-seed runner** | draait hetzelfde experiment één keer voor elke seed, en bewaart elke run | `run_seeds(settings, [0, 1, 2], steps, path)` |
        | $t$, $v$ | de eerste gelogde stappen waarop training en validatie 99% halen: de momenten van je plan | `first_step(...)` |
        """),
        L.watch_out(f"Een gelogd moment is een beetje te laat. Het plan leest $t$ en $v$ af op gelogde stappen, en het echte moment ligt tussen een gelogde stap en de stap die daarvoor gelogd is. Met {C.PER_FACTOR} gelogde stappen per factor 10 is elke stap $10^{{1/{C.PER_FACTOR}}} = 1{{,}}12$ keer de vorige: $v$ en $t$ zijn elk hooguit 12% te laat, dus kan $v / t$ ongeveer zoveel naar beide kanten afwijken. Bij {round(_r['times'])} tegen de {C.DELAY} van het plan verandert dat het oordeel niet; vlak bij {C.DELAY}: log vaker."),
        mo.md(r"""
        ### Recept

        Om een figuur van trainingslijnen opnieuw te tekenen uit je eigen runs:

        1. Kies de stappen om te loggen: gelijk verdeeld over de as die de figuur gebruikt, met de laatste stap erbij.
        2. Train voor elke seed, en meet na elk van die stappen wat de figuur laat zien, op alle data waar hij over gaat.
        3. Bewaar elke run in een logboek, één regel per run, zodat de runs die klaar zijn klaar blijven.
        4. Teken elke run op de assen van de figuur, met een eigen lijnstijl voor elke grootheid, zoals training doorgetrokken en validatie gestreept, en met zijn boodschap in de titel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Waarom niet om de 500 stappen loggen? De cel hieronder rekent uit wat dat geeft voor een run van 12.000 stappen, en hoeveel van een logaritmische as het leeg laat: stap 1 van het recept. Voer hem uit.
    """)
    return


@app.cell
def example_every():
    import math

    _every = list(range(500, 12_001, 500))   # 1. om de 500 stappen loggen
    print("gelogde stappen:", len(_every), "de eerste bij stap", _every[0])
    # het deel van een logaritmische as van stap 1 tot 12.000 dat vóór stap 500 ligt
    print("deel vóór stap 500:", round(math.log10(500) / math.log10(12_000), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Om de 500 stappen loggen geeft 24 logs, de eerste bij stap 500. Op een logaritmische as van stap 1 tot 12.000 ligt twee derde van de breedte, 0,66, vóór stap 500, en daar gaat training van 1% naar 100%: dat deel van de figuur zou leeg blijven. Logaritmisch verdeelde stappen vullen het, met evenveel logs van 1 tot 10 als van 1.000 tot 10.000.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.vstack([
        mo.md(rf"""
        ## Zelf doen

        ### Oefening 1. Stappen op een logaritmische as

        Maak `log_steps(budget, per_factor)` af: de stappen waarop een run logt. Voor $k = 0, 1, 2, \dots$ is de stap $10^{{k / \text{{per\_factor}}}}$, afgerond op een hele stap, zolang die niet boven `budget` ligt; in het begin ronden meerdere waarden van $k$ af naar dezelfde stap, en elke stap komt één keer in de lijst. De laatste stap van de run, `budget`, komt aan het eind, als hij er nog niet in staat. De check roept `log_steps(100, 5)` aan, die `[1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100]` moet geven, en andere budgetten. Als hij goed is, wordt hij in je project bewaard als `repro/log.py`.
        """),
        L.task("Maak `log_steps` af: een `while`-loop over `k` die elke nieuwe afgeronde stap toevoegt, en daarna het budget."),
    ])
    return


@app.cell(hide_code=True)
def exercise_log():
    box_log = L.code_box("log-steps", lines=14)
    box_log
    return (box_log,)


@app.cell(hide_code=True)
def check_log(box_log):
    result_log = L.check("log-steps", box_log.value)
    result_log
    return (result_log,)


@app.cell(hide_code=True)
def hints_log():
    L.hints(
        "log-steps",
        [
            "`10 ** (k / per_factor)` is een float, en `round` maakt er een hele stap van. Een `while`-loop gaat door zolang zijn voorwaarde klopt: welke voorwaarde laat hem stoppen bij het budget?",
            "Met `per_factor = 20` ronden `k = 0` tot en met `3` allemaal af naar 1. Kijk vóór `append` of de stap al in `steps` staat, met `not in`; en vergeet `k = k + 1` niet.",
        ],
        why="Met 20 gelogde stappen per factor 10 logt een run van 12.000 stappen 72 keer, waarvan 43 keer vóór stap 500: om de 500 stappen loggen zou 24 keer loggen, en daar geen enkele keer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De logger

        Maak `accuracy(model, x, y)` en het loggen in `run_seed(settings, seed, steps)` af. De rest van de run is gegeven: de splitsing, de tokens en het model zoals in [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl), AdamW met de instellingen van de paper en zijn warmup, en batches die een eigen generator trekt. `accuracy` geeft het aandeel van de vergelijkingen waarvan de gok op de laatste plek het antwoord is, als getal. Na elke stap in `at` voegt de loop aan `log` de stap toe, de accuracy op alle trainingsvergelijkingen en die op alle validatievergelijkingen. De check draait hem met kleine instellingen, optellen modulo 7, en vergelijkt de logboeken met die van de course. Als hij goed is, wordt hij in je project bewaard als `repro/run.py`.
        """),
        L.task("Maak `accuracy` en het loggen af: na elke stap in `at` de stap en beide accuracy's."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run-seed", lines=44)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run-seed", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run-seed",
        [
            "De gok is het token met de hoogste logit: `.argmax(-1)` van de logits op de laatste plek. `guess == y` is een tensor van `True` en `False`; als floats is hun gemiddelde het aandeel dat goed is.",
            "`log[\"step\"]`, `log[\"train\"]` en `log[\"val\"]` zijn lijsten: `append` aan elk, alleen als `step in at`. De trainingsvergelijkingen zijn `xt` en `yt`, de validatievergelijkingen `xv` en `yv`.",
        ],
        why="Gemeten op alle vergelijkingen is de accuracy van een stap elke keer dat je deze seed draait hetzelfde getal; gemeten op een willekeurige batch zou hij meespringen met de batch.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_seeds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De multi-seed runner

        Maak `run_seeds(settings, seeds, steps, path)` af: voor elke seed die niet in `done` staat, één run van je `run_seed`, met `device` en `progress` doorgegeven, als één regel JSON toegevoegd aan het bestand op `path`, zoals in [Een experimentlogboek bijhouden](/course/open/res-experiment-log?lang=nl). Het lezen van de seeds die klaar zijn, is gegeven: `done` bevat ze, en is leeg als het logboek nieuw is. De check draait hem met een snelle invaller voor `run_seed` die niet traint, in een eigen map. Als hij goed is, wordt hij in je project bewaard als `repro/seeds.py`.
        """),
        L.task("Maak `run_seeds` af: draai elke seed die niet in `done` staat, en voeg zijn logboek als één regel aan het bestand toe."),
    ])
    return


@app.cell(hide_code=True)
def exercise_seeds():
    box_seeds = L.code_box("run-seeds", lines=24)
    box_seeds
    return (box_seeds,)


@app.cell(hide_code=True)
def check_seeds(box_seeds):
    result_seeds = L.check("run-seeds", box_seeds.value)
    result_seeds
    return (result_seeds,)


@app.cell(hide_code=True)
def hints_seeds():
    L.hints(
        "run-seeds",
        [
            "`if seed not in done:` voert de ingesprongen regels alleen uit voor een seed die nog niet klaar is: de run, en zijn regel schrijven.",
            "Open het bestand met `\"a\"`, zodat elke run na de runs ervoor komt; één regel is `json.dumps(log)` met een regeleinde erachter.",
        ],
        why="Een run die halverwege stopt, omdat de computer slaapt of de les sluit, verliest alleen de seed waar hij mee bezig was: de volgende keer begint `run_seeds` met die seed.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_figure():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Teken Figuur 1 opnieuw

        Teken Figuur 1 nog eens, uit de logboeken van de course voor de seeds 0, 1 en 2 in `data/course-grokking.jsonl`: voor elke run zijn training accuracy doorgetrokken en zijn validation accuracy gestreept, tegen `log["step"]`, op een logaritmische as van stappen, met de ene boodschap van de figuur als titel en beide assen benoemd. De legenda is gegeven: hij benoemt de eerste twee lijnen. De check vergelijkt je lijnen met de figuur van de course. Zodra je eigen runs in je logboek staan, tekent dezelfde code die: alleen het bestand is een ander.
        """),
        L.task("Teken beide lijnen van elke run tegen zijn stappen, op een logaritmische x-as, met een titel die zegt wat alle drie laten zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_figure():
    box_figure = L.code_box("figure", lines=16)
    box_figure
    return (box_figure,)


@app.cell(hide_code=True)
def check_figure(box_figure):
    result_figure = L.check("figure", box_figure.value)
    result_figure
    return (result_figure,)


@app.cell(hide_code=True)
def hints_figure():
    L.hints(
        "figure",
        [
            "`plt.plot(xs, ys, \"--\")` tekent een gestreepte lijn, en `\"-\"` een doorgetrokken. In de loop is `log` één run: teken er twee lijnen voor, training eerst.",
            "De x-as van Figuur 1 is een logaritmische as: `plt.xscale(\"log\")`. Wat laten alle drie de runs zien, in één zin? Dat is de titel.",
        ],
        why="Met elke run in de figuur ziet een lezer meteen hoe weinig de sprong van de seed afhangt: in alle drie haalt validatie 99% 45 tot 50 keer zo laat als training.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_plateau():
    mo.md(r"""
    ### Oefening 5. Waarom 30%?

    Je figuur laat meer zien dan de claim. Twee van de runs zakken even weg, ook training, en herstellen bij de volgende gelogde stap: een korte terugval van de training, een regel in je verslag waard. En validatie ligt duizenden stappen vlak, op een hoogte die Figuur 1 niet heeft.
    """)
    return


@app.cell(hide_code=True)
def choice_plateau_cell():
    choice_plateau = L.choice(
        "plateau",
        question=f"In je figuur ligt validatie duizenden stappen vlak op ongeveer 30% terwijl training op 100% staat; in Figuur 1 van de paper, van delen, blijft hij laag. In de splitsing van seed 0 heeft {C.number(100 * C.MIRROR['share'], 1, L.lang)}% van de validatievergelijkingen zijn spiegelbeeld, b + a, in de training data. Waarom blijft validatie steken op ongeveer 30%?",
        options={
            "leak": "Ongeveer 30% van de validatievergelijkingen staat ook in de training data, dus het heeft ze gezien",
            "chance": "Bij 97 mogelijke antwoorden is 30% wat gokken oplevert",
            "part": "Het model heeft 30% van de regel geleerd, omdat het op 30% van de tabel traint",
            "right": "Het heeft de vergelijkingen goed waarvan het b + a in de training zag, en bijna geen andere",
        },
        correct="right",
        why={
            "leak": "Je `dataset` splitst de tabel: elke vergelijking staat in de training data of in de validatiedata, nooit in allebei. Wat ongeveer 30% van de validatievergelijkingen wel heeft, is zijn spiegelbeeld in de training data.",
            "chance": "Gokken tussen 97 antwoorden heeft er één op 97 goed, ongeveer 1%. 30% ligt ver boven kansniveau: het model weet iets over die vergelijkingen.",
            "part": f"Een model dat een deel van de regel kende, zou van elke soort vergelijking een deel goed hebben. Bij stap {C.number(C.MIRROR['step'], 0, L.lang)} van seed 0 heeft het {C.number(100 * C.MIRROR['mirrored'], 1, L.lang)}% van de gespiegelde validatievergelijkingen goed en {C.number(100 * C.MIRROR['others'], 1, L.lang)}% van de andere: het kent het spiegelbeeld, niet de regel.",
            "right": f"Bij stap {C.number(C.MIRROR['step'], 0, L.lang)} van seed 0 heeft het model {C.number(100 * C.MIRROR['mirrored'], 1, L.lang)}% van de validatievergelijkingen met hun spiegelbeeld in de training data goed, en {C.number(100 * C.MIRROR['others'], 1, L.lang)}% van de andere. Delen heeft zo'n spiegelbeeld niet, dus heeft Figuur 1 zo'n plateau niet: een verschil tussen jouw figuur en die van de paper, met zijn oorzaak, voor je verslag.",
        },
    )
    choice_plateau
    return (choice_plateau,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Wat je project nu kan

    De knop traint de seeds 0, 1 en 2 met je kit, na elkaar: je `run_seeds` met je `run_seed`, op je ModernGPT met je data en je instellingen, elk {C.number(C.STEPS, 0, L.lang)} stappen, in je logboek `work/projects/repro/runs.jsonl`. Je ziet elke run terwijl hij traint. Op de CPU van de course, met één thread, duurt een seed ongeveer {round(C.STEPS * C.SECONDS_PER_STEP / 60)} minuten, alle drie ongeveer {round(3 * C.STEPS * C.SECONDS_PER_STEP / 60)}; op een GPU een paar minuten. Daar bewijst je runner zich: sluit je de les halverwege, dan blijven de seeds die klaar zijn in je logboek staan; de volgende druk op de knop gaat verder met de andere. Terwijl hij traint, wacht de rest van deze les: je kunt verder lezen, of in een andere les werken. Waar een onderdeel van je kit ontbreekt of kapot is, springt de versie van de course in.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train de seeds 0, 1 en 2 met je kit")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_run, result_seeds):
    import time as _time

    result_run, result_seeds  # opnieuw uitvoeren als een oefening een onderdeel van de kit bewaart
    _ready = result_run.state == "correct" and result_seeds.state == "correct"
    _labels = {"x": "stap", "y": "accuracy", "train": "training", "val": "validatie"}
    trained = None
    if not train_button.value:
        _out = mo.md("*Druk op de knop om je seeds met je kit te trainen.*" if _ready else "*Maak eerst oefening 2 en 3 af, en druk dan op de knop: hij traint met je `run_seeds` en je `run_seed`.*")
    elif set(C.SEEDS) <= {_log["seed"] for _log in C.my_runs() or []}:
        _out = mo.md("*De seeds 0, 1 en 2 staan al in je logboek, dus slaat je `run_seeds` ze over. Wil je ze opnieuw trainen, verwijder dan `work/projects/repro/runs.jsonl`.*")
    else:
        _began = {}

        def _progress(_log):
            _step, _now = _log["step"][-1], _time.perf_counter()
            _spent = _now - _began.setdefault(_log["seed"], _now)
            _left = f", nog ongeveer {round(_spent / _step * (C.STEPS - _step) / 60)} te gaan" if _step >= 1000 else ""   # vóór stap 1.000 maken de vele logs van het begin een schatting te hoog
            mo.output.replace(mo.vstack([
                mo.md(f"**Seed {_log['seed']}, stap {C.number(_step, 0, L.lang)} van {C.number(C.STEPS, 0, L.lang)}** · {C.number(_spent / 60, 1, L.lang)} minuten{_left} · training {C.number(100 * _log['train'][-1], 1, L.lang)}%, validatie {C.number(100 * _log['val'][-1], 1, L.lang)}%"),
                mo.Html(C.figure_picture([_log], {**_labels, "title": f"Accuracy tijdens je run, seed {_log['seed']}", "alt": f"Je run van seed {_log['seed']} tot stap {_step}: training op {C.number(100 * _log['train'][-1], 1, L.lang)}%, validatie op {C.number(100 * _log['val'][-1], 1, L.lang)}%."})),
            ]))

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `run_seed` en je `run_seeds` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_r, str):
            _out = mo.md({"seeds": "*Je `repro/seeds.py` of `repro/run.py` stopt met een fout. Heb je hem na zijn check veranderd? Verwijder hem, en check zijn oefening opnieuw.*",
                          "model": "*Je ModernGPT, je data of je instellingen stoppen met een fout. Heb je `llm/modern_gpt.py`, `repro/data.py` of `repro/settings.py` na zijn check veranderd? Verwijder het bestand dat je veranderde, en check zijn oefening opnieuw.*"}[_r])
        else:
            trained = _r
            _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
            _out = mo.md(f"*Klaar: de seeds 0, 1 en 2 staan in je logboek, `work/projects/repro/runs.jsonl`.{_stand}*")
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_mine(trained):
    trained  # het logboek opnieuw lezen na een run
    _mine = C.my_runs()
    if not _mine:
        _out = mo.md("*Je runs spelen hier af zodra ze in je logboek staan.*")
    else:
        _log = _mine[0]
        _r = C.read(_log)
        _seen = [(_l["seed"], C.read(_l)) for _l in _mine]
        _when = lambda _s: C.number(_s, 0, L.lang) if _s is not None else "niet gehaald"
        _rows = "\n".join(f"| {_seed} | {_when(_x['t'])} | {_when(_x['v'])} | {C.number(_x['times'], 0, L.lang) if _x['times'] else '–'} | {_x['verdict']} |" for _seed, _x in _seen)
        _out = mo.vstack([
            mo.ui.anywidget(play.Replay(
                steps=_log["step"], train=_log["train"], val=_log["val"], threshold=C.THRESHOLD, t_index=_r["t_index"], v_index=_r["v_index"],
                verdict=f"Het oordeel van het plan: <b>{_r['verdict']}</b>.", decimal="," if L.lang == "nl" else ".",
                labels={"task_kick": L.w["task"], "task": f"Speel je run van seed {_log['seed']} af: laat hij zien wat de run van de course liet zien?",
                        "title": f"Accuracy tijdens je run, seed {_log['seed']}", "x_name": "stap", "train": "training", "val": "validatie",
                        "play": "Afspelen", "pause": "Pauze", "back": "Terug", "fwd": "Vooruit", "step_name": "stap",
                        "alt": f"De accuracy op de trainings- en de validatievergelijkingen tijdens je run van seed {_log['seed']}, over {C.number(_log['step'][-1], 0, L.lang)} stappen op een logaritmische as. Het oordeel van het plan: {_r['verdict']}."})),
            mo.md(f"""
| seed | $t$ | $v$ | $v / t$ | het oordeel van het plan |
|---|---|---|---|---|
{_rows}

Om ze te tekenen, kopieer je je figuurcode van oefening 4 naar een eigen cel en laat je hem `work/projects/repro/runs.jsonl` openen.
"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Je kit draait nu elke seed van een instelling in één logboek, en je figuurcode tekent wat dat logboek bevat.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het plan uit [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl), nu afgelezen in echte runs; elke run in één figuur, zoals in [Figuren die iets laten zien](/course/open/res-figures?lang=nl); één regel per run, zoals in [Een experimentlogboek bijhouden](/course/open/res-experiment-log?lang=nl); en de spiegelcellen van de tabel uit [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl), nu zichtbaar als een plateau. Je kit heeft nu `repro/log.py`, `repro/run.py` en `repro/seeds.py`. De volgende les, *Begeleide reproductie 5: de ablation*, zet weight decay uit en aan en varieert het deel van de tabel om op te trainen, om te vinden welk onderdeel de sprong nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_log, result_run, result_seeds, result_figure, choice_plateau):
    L.wrap_up(result_log, result_run, result_seeds, result_figure, choice_plateau)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
