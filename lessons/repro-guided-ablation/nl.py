# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-guided-ablation. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-ablation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl) en [Begeleide reproductie 4: reproduceer de belangrijkste figuur](/course/open/repro-guided-figure?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_two():
    quick_two = L.choice(
        "quick-two",
        question="Je reproductie gebruikt de warmup van de paper, 10 stappen. Je wilt weten of de sprong die nodig heeft. Welke runs geven daar antwoord op?",
        options={
            "right": "De run met de instellingen van je kit, en dezelfde run met de warmup eruit",
            "two": "Een run zonder warmup en zonder weight decay, tegen de run met de instellingen van je kit",
            "seeds": "Twee runs zonder warmup, één met seed 0 en één met seed 1",
        },
        correct="right",
        why={
            "two": "Er zijn twee delen tegelijk weg, dus een verschil kan van elk van beide komen. Haal één deel weg, en vergelijk met de control run. Zie [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl).",
            "seeds": "Beide runs missen de warmup, dus geen van beide laat zien wat de warmup doet; samen laten ze zien hoeveel de seed een run verandert. Zie [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl).",
            "right": "Een ablation haalt één deel weg en vergelijkt met de control run, verder in alles hetzelfde: een verschil komt dan van dat deel.",
        },
    )
    quick_two
    return (quick_two,)


@app.cell(hide_code=True)
def quick_check_close():
    quick_close = L.choice(
        "quick-close",
        question="Een run geeft t = 200 en v = 1.995, afgelezen op gelogde stappen, 20 per factor 10. Je plan vraagt een vertraging van minstens 10 keer. Wat zeg je?",
        options={
            "sure": "Geen vertraging, en dat staat vast: 9,98 ligt onder de 10 van het plan, wat de logs ook misten",
            "delay": "Een vertraging: validatie heeft ongeveer 1.800 stappen meer nodig dan training, en dat is lang",
            "right": "Geen vertraging, maar het scheelt weinig: elk moment kan tot 12% te laat zijn",
        },
        correct="right",
        why={
            "sure": "Elk gelogd moment kan tot 12% te laat zijn, dus 1.995 / 200 = 9,98 kan in werkelijkheid alles van ongeveer 8,9 tot 11,2 zijn. Volgens het plan is het geen vertraging, maar het scheelt weinig. Zie [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).",
            "delay": "Het plan telt hoeveel keer zo laat, niet hoeveel stappen meer: 1.995 is 9,98 keer 200, net onder de 10 van het plan. Zie [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).",
            "right": "Volgens het plan is het geen vertraging; maar met gelogde momenten die tot 12% te laat kunnen zijn, is een verhouding van 9,98 een nipte uitslag, en is die meer logs of meer seeds waard.",
        },
    )
    quick_close
    return (quick_close,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je figuur laat de claim zien in je eigen runs. Maar de paper zegt meer dan zijn belangrijkste figuur: Figuur 1 (midden) laat zien dat de tijd tot validatie 99% haalt snel groeit als het deel van de training data kleiner wordt, en Figuur 2 (links) dat weight decay de generalisatie "het meest verbetert". Welke van de twee heeft de sprong in jouw reproductie nodig? Een ablation zoekt het uit: verander één instelling, houd de rest zoals het was, en vergelijk met de control run. Hier twee instellingen, elk met een paar waarden, in elke combinatie: zodat je ook ziet of de ene alleen bij sommige waarden van de andere uitmaakt.

        **Een voorbeeld uit het dagelijks leven.** Rijst het brood door de gist, of door het warme plekje bij de verwarming? Bak met en zonder gist, op het warme plekje en op een koude plek: vier broden. Rijst alleen het brood met gist op het warme plekje, dan heeft het allebei nodig; rijzen beide broden met gist, dan is de gist genoeg.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de belangrijkste figuur",
                            "De ablation", "Vergelijk met de paper", "Het verslag"],
                           4, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_decay():
    _r = C.read(C.run(1.0, C.FRACTION))
    guess_decay = L.guess(
        "no-decay",
        question=f"In de run van de course bij {round(100 * C.FRACTION)}% van de tabel haalde validatie 99% bij stap {C.number(_r['v'], 0, L.lang)}. Wat denk je dat er binnen de {C.number(C.STEPS, 0, L.lang)} stappen gebeurt met weight decay uit, en verder alles hetzelfde?",
        options={
            "later": "Validatie haalt 99% later, maar binnen de 12.000 stappen",
            "same": "Validatie haalt 99% bij ongeveer dezelfde stap",
            "never": "Validatie haalt 99% niet binnen de 12.000 stappen",
        },
        reveal=f"**Hij haalt 99% niet binnen het budget:** na {C.number(C.STEPS, 0, L.lang)} stappen staat validatie op {C.number(100 * C.run(0.0, C.FRACTION)['val'][-1], 0, L.lang)}%. Wat \"binnen het budget\" betekent, en bij welke delen van de tabel hij 99% wel haalt, laat *Het idee* zien.",
    )
    guess_decay
    return (guess_decay,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## Het idee

    Een **ablation grid** draait elke combinatie van een paar waarden van elke instelling die je verandert: weight decay 1 of 0, maal de delen 30%, 50% en 90%, geeft 6 runs. Eén daarvan is de control run, met de instellingen van je kit: de weight decay van de paper, 1, bij {round(100 * C.FRACTION)}% van de tabel. Elke regel van het experimentlogboek bewaart de waarden die zijn run verandert, zodat je het grid als tabel kunt teruglezen.

    Elke run heeft een **budget**: hier {C.number(C.STEPS, 0, L.lang)} stappen. Haalt validatie daarbinnen geen 99%, dan zegt het plan "not reached": niet binnen dit budget, en dat is iets anders dan nooit. De run van Figuur 1 (links) van de paper zelf, delen modulo 97 bij 50% van de tabel, is getraind zonder weight decay, met Adam en een budget van $10^6$ stappen (bijlage A.1.2), en daar had validatie bijna $10^6$ stappen nodig om training in te halen.

    Hieronder staan de runs van de course, seed 0, met weight decay (doorgetrokken) en zonder (gestreept), bij delen van 30% tot 90%: de validation accuracy over de stappen, en een stip waar hij 99% haalt. Schuif om het deel van de tabel te kiezen.
    """)
    return


@app.cell(hide_code=True)
def explore_fraction_cell():
    def _frame(fraction):
        _f = round(fraction / 100, 1)
        _on, _off = C.read(C.run(1.0, _f)), C.read(C.run(0.0, _f))
        _say_on = (f"haalt 99% bij stap {C.number(_on['v'], 0, L.lang)}, {C.number(_on['times'], 1, L.lang)} keer zo laat als training" if _on["v"] is not None
                   else f"eindigt op {C.number(100 * C.run(1.0, _f)['val'][-1], 0, L.lang)}%")
        _say_off = (f"haalt 99% bij stap {C.number(_off['v'], 0, L.lang)}, {C.number(_off['times'], 1, L.lang)} keer zo laat als training" if _off["v"] is not None
                    else f"eindigt op {C.number(100 * C.run(0.0, _f)['val'][-1], 0, L.lang)}% na {C.number(C.STEPS, 0, L.lang)} stappen")
        return (C.val_picture(_f, {"title": f"Validatie bij {round(100 * _f)}% van de tabel", "x": "stap", "y": "validation accuracy",
                                   "with": "met weight decay", "without": "zonder weight decay",
                                   "alt": f"De validation accuracy over {C.number(C.STEPS, 0, L.lang)} stappen op een logaritmische as, bij {round(100 * _f)}% van de tabel. Met weight decay: validatie {_say_on}; zonder: validatie {_say_off}."}),
                f"Bij {round(100 * _f)}% met weight decay: validatie {_say_on}; zonder weight decay: validatie {_say_off}.")

    explore_fraction = L.explore("fraction", "% van de tabel", _frame, start=round(100 * C.FRACTION), low=30, high=90, step=10, role="hyper",
                                 task="Vanaf welk deel haalt validatie zonder weight decay 99% binnen het budget? En vanaf welk deel is hij met weight decay minder dan 10 keer zo laat als training?")
    explore_fraction
    return


@app.cell(hide_code=True)
def text_idea_after():
    _no = C.first_reached(0.0)
    _close = next(_f for _f in C.FRACTIONS if C.read(C.run(1.0, _f))["verdict"] == "no delay")
    mo.vstack([
        mo.md(rf"""
        Zonder weight decay haalt validatie de 99% binnen het budget pas vanaf {round(100 * _no)}% van de tabel, en zelfs daar laat: {C.number(C.read(C.run(0.0, _no))['times'], 1, L.lang)} keer zo laat als training. Daaronder eindigt hij ergens tussen {C.number(100 * C.run(0.0, C.FRACTIONS[0])['val'][-1], 0, L.lang)}% en {C.number(100 * C.run(0.0, round(_no - 0.1, 1))['val'][-1], 0, L.lang)}%. Met weight decay haalt hij bij elk deel 99%, en hoe kleiner het deel, hoe verder hij achterblijft bij training: {C.number(C.read(C.run(1.0, C.FRACTION))['times'], 1, L.lang)} keer zo laat bij {round(100 * C.FRACTION)}%, en vanaf {round(100 * _close)}% minder dan 10 keer. Binnen dit budget brengt weight decay de sprong dus bij elk deel eerder, en onder {round(100 * _no)}% maakt hij het verschil tussen een sprong en geen sprong. Bij {round(100 * _close)}% is de verhouding {C.number(C.read(C.run(1.0, _close))['times'], 1, L.lang)}: het scheelt weinig.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **ablation grid** | elke combinatie van een paar waarden van elke instelling die je verandert, elke run tegen dezelfde control run | `grid(settings, changes)` |
        | **budget** | het grootste aantal stappen dat een run krijgt; een resultaat zegt wat er daarbinnen gebeurde | `steps` |
        | not reached | validatie haalde binnen het budget geen 99%, en dat is iets anders dan nooit | `"not reached"` |
        """),
        L.watch_out("Eén seed per combinatie is één worp met de dobbelsteen. Een duidelijk verschil, zoals een sprong tegen geen sprong, overleeft dat; een nipte uitslag niet: bij het deel waar de vertraging dicht bij 10 komt, kan een andere seed het oordeel naar beide kanten laten omslaan. De volgende les herhaalt de runs over seeds."),
        mo.md(r"""
        ### Recept

        Om uit te zoeken welke instellingen een effect nodig heeft:

        1. Kies de instellingen om te veranderen, en een paar waarden van elk, rond die van de paper.
        2. Tel de runs: de aantallen waarden met elkaar vermenigvuldigd, maal de tijd van één run.
        3. Draai elke combinatie met hetzelfde budget, dezelfde seed en dezelfde code, en bewaar de waarden die elke run verandert bij zijn logboek.
        4. Lees elke run met je plan, en zet de oordelen in een tabel: de ene instelling in de rijen, de andere in de kolommen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Hoe lang duurt het grid? De cel hieronder telt de runs van weight decay 1 of 0 maal drie delen van de tabel, en van het grid van de course met zeven delen, bij de seconden per stap van de course op één thread: stap 2 van het recept. Voer hem uit.
    """)
    return


@app.cell
def example_count():
    _decays = [1.0, 0.0]
    _fractions = [0.3, 0.5, 0.9]
    _runs = len(_decays) * len(_fractions)   # 2. de aantallen waarden vermenigvuldigd
    _hours = _runs * 12_000 * 0.113 / 3600   # stappen per run, seconden per stap
    print("runs:", _runs, "uur:", round(_hours, 1))
    _runs = len(_decays) * 7                 # het grid van de course: zeven delen
    print("runs:", _runs, "uur:", round(_runs * 12_000 * 0.113 / 3600, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zes runs duren 2,3 uur op de CPU van de course, en met zeven delen, 14 runs, 5,3 uur: elke waarde die je aan de ene instelling toevoegt, voegt een run toe voor elke waarde van de andere. Op een GPU duurt elke run een paar minuten. Kies de waarden die je vraag beantwoorden, en niet meer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_grid():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elke combinatie

        Maak `grid(settings, changes)` af: elke combinatie van de waarden in `changes`, elk een kopie van `settings` met die waarden. `changes` bevat een lijst waarden voor elke sleutel die hij verandert. Begin met één kopie van `settings`; maak voor elke sleutel een nieuwe lijst uit elke run tot nu toe, één keer voor elke waarde van die sleutel, en laat `runs` die lijst worden. Zo verandert de eerste sleutel het langzaamst, zoals in een tabel die je rij voor rij leest. De check roept `grid({"lr": 0.001, "weight_decay": 1.0, "batch": 512}, {"lr": [0.001, 0.003], "batch": [256, 512]})` aan, die 4 instellingen moet geven, en andere grids. Als hij goed is, wordt hij in je project bewaard als `repro/grid.py`.
        """),
        L.task("Maak `grid` af: voor elke sleutel een kopie van elke run tot nu toe voor elke waarde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grid():
    box_grid = L.code_box("grid", lines=13)
    box_grid
    return (box_grid,)


@app.cell(hide_code=True)
def check_grid(box_grid):
    result_grid = L.check("grid", box_grid.value)
    result_grid
    return (result_grid,)


@app.cell(hide_code=True)
def hints_grid():
    L.hints(
        "grid",
        [
            "Twee loops binnen de loop over de sleutels: de buitenste over `runs`, de binnenste over `changes[key]`. Elke ronde van de binnenste loop voegt één kopie toe aan `new`.",
            "`dict(run)` maakt een kopie: verander je de kopie, dan blijft `run` zoals hij was; `one[key] = value` zet de waarde in de kopie. Na beide loops: `runs = new`.",
        ],
        why="Kopiëren is nodig: `one = run` zou dezelfde dictionary zijn, en elke run die ervan gemaakt is, zou meeveranderen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run_grid():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De ablation-runner

        Maak `run_grid(settings, changes, steps, path)` af: voor elke combinatie van `grid(settings, changes)` waarvan de waarden niet in `done` staan, één run van je `run_seed` met seed 0, met `device` en `progress` doorgegeven, met de waarden van de sleutels toegevoegd aan zijn logboek, en het logboek als één regel JSON toegevoegd aan het bestand op `path`. `values` is de lijst van zijn waarden van de sleutels, in de volgorde van `keys`, zoals elk item van `done`. De rest is gegeven, zoals in je `run_seeds`. De check draait hem met een snelle invaller voor `run_seed` die niet traint, in een eigen map. Als hij goed is, wordt hij in je project bewaard als `repro/ablation.py`.
        """),
        L.task("Maak `run_grid` af: draai elke combinatie die niet klaar is, voeg zijn waarden toe aan zijn logboek, en voeg het logboek toe aan het bestand."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run_grid():
    box_run_grid = L.code_box("run-grid", lines=24)
    box_run_grid
    return (box_run_grid,)


@app.cell(hide_code=True)
def check_run_grid(box_run_grid):
    result_run_grid = L.check("run-grid", box_run_grid.value)
    result_run_grid
    return (result_run_grid,)


@app.cell(hide_code=True)
def hints_run_grid():
    L.hints(
        "run-grid",
        [
            "`values = [one[key] for key in keys]` geeft de waarden van deze combinatie in de volgorde van `done`. Daarna `if values not in done:`, zoals in je `run_seeds`.",
            "De run traint met `one`, niet met `settings`: `run_seed(one, 0, steps, device, progress)`. Vóór het schrijven `log[key] = one[key]` voor elke sleutel, zodat de regel zegt welke combinatie hij is.",
        ],
        why="Zonder zijn waarden in het logboek kan een regel niet zeggen welke combinatie hij was, en kun je de tabel niet teruglezen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_figure():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De momenten tegen het deel van de tabel

        Teken de runs van de course met weight decay zoals Figuur 1 (midden) van de paper zijn runs tekent: voor elk deel van de tabel `t` en `v`, de eerste gelogde stappen waarop training en validatie 99% halen, met `first_step` en `THRESHOLD` van je plan. Teken `t` en `v` tegen de delen, elk een lijn met stippen, `t` eerst en doorgetrokken, `v` gestreept, op een logaritmische y-as, met de ene boodschap van de figuur als titel en een label bij beide assen. De legenda is gegeven. De check vergelijkt je lijnen met de figuur van de course.
        """),
        L.task("Verzamel het deel, t en v van elke run met weight decay, en teken t en v tegen het deel op een logaritmische y-as."),
    ])
    return


@app.cell(hide_code=True)
def exercise_figure():
    box_figure = L.code_box("figure", lines=20)
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
            "`first_step(log[\"step\"], log[\"train\"], THRESHOLD)` geeft `t` van één run; met `log[\"val\"]` geeft hij `v`. Voeg ze toe, en `log[\"fraction\"]`, aan de drie lijsten.",
            "`plt.plot(fractions, t, \"o-\")` tekent een lijn met stippen; `\"o--\"` een gestreepte. `plt.yscale(\"log\")` maakt van de y-as een logaritmische as.",
        ],
        why="Op een logaritmische y-as is de afstand tussen de lijnen hoeveel keer zo laat validatie is: het verschil dat je plan afleest.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shows():
    mo.md(r"""
    ### Oefening 4. Wat laat het zien?
    """)
    return


@app.cell(hide_code=True)
def choice_shows_cell():
    _no = C.first_reached(0.0)
    choice_shows = L.choice(
        "what-shows",
        question=f"De ablation van de course, seed 0, {C.number(C.STEPS, 0, L.lang)} stappen per run, delen van 30% tot 90%: met weight decay haalt validatie bij elk deel 99%; zonder weight decay pas vanaf {round(100 * _no)}%, en later dan met. Wat mag je verslag zeggen?",
        options={
            "right": "Binnen dit budget brengt weight decay de sprong eerder, en bepaalt hij onder " + f"{round(100 * _no)}% of er een sprong komt",
            "never": "Zonder weight decay generaliseert dit model nooit onder " + f"{round(100 * _no)}% van de tabel",
            "fraction": "Alleen het deel van de tabel doet ertoe voor de sprong; weight decay maakt elke run alleen wat sneller",
        },
        correct="right",
        why={
            "never": "\"Not reached\" betekent: niet binnen dit budget. De run van Figuur 1 (links) van de paper had ook geen weight decay, en validatie kwam er na bijna een miljoen stappen wel.",
            "fraction": f"Onder {round(100 * _no)}% halen de runs zonder weight decay binnen het budget helemaal geen 99%, en die met weight decay wel: binnen dit budget maakt weight decay het verschil tussen een sprong en geen sprong.",
            "right": "Dat laat het grid zien, en niet meer: één seed per combinatie, één budget, één taak. De volgende les herhaalt de runs over seeds en vergelijkt ze met de paper.",
        },
    )
    choice_shows
    return (choice_shows,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Wat je project nu kan

    De knop traint het grid met je kit: je `run_grid` met je `grid` en je `run_seed`, op je ModernGPT met je data en je instellingen, weight decay 1 en 0 bij 30%, 50% en 90%, elk {C.number(C.STEPS, 0, L.lang)} stappen, in je experimentlogboek `work/projects/repro/ablation.jsonl`. Een grafiek laat elke run zien terwijl hij traint: zijn validation accuracy, met zijn training accuracy gestreept, op een logaritmische as van de stappen zoals in de figuur hierboven, en in de titel de weight decay en het deel van de tabel van de run. Op een GPU duren de zes runs elk een paar minuten; op de CPU van de course, met één thread, samen ongeveer {C.number(6 * C.STEPS * C.SECONDS_PER_STEP / 3600, 1, L.lang)} uur. Sluit je de les halverwege, dan blijven de runs die klaar zijn in je logboek staan; de volgende druk op de knop gaat verder met de andere. Terwijl hij traint, wacht de rest van deze les: je kunt verder lezen, of in een andere les werken. Waar een onderdeel van je kit ontbreekt of kapot is, springt de versie van de course in.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train het grid met je kit")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(train_button, result_grid, result_run_grid):
    import time as _time

    result_grid, result_run_grid  # opnieuw uitvoeren als een oefening een onderdeel van de kit bewaart
    _ready = result_grid.state == "correct" and result_run_grid.state == "correct"
    trained = None
    if not train_button.value:
        _out = mo.md("*Druk op de knop om het grid met je kit te trainen.*" if _ready else "*Maak eerst oefening 1 en 2 af, en druk dan op de knop: hij traint met je `run_grid` en je `grid`.*")
    elif len(C.my_runs() or []) >= 6:
        _out = mo.md("*De zes runs van het grid staan al in je logboek, dus slaat je `run_grid` ze over. Wil je ze opnieuw trainen, verwijder dan `work/projects/repro/ablation.jsonl`.*")
    else:
        _kept = [(_l["weight_decay"], _l["fraction"]) for _l in C.my_runs() or []]
        _todo = [(_wd, _f) for _wd in C.CHANGES["weight_decay"] for _f in C.CHANGES["fraction"] if (_wd, _f) not in _kept]   # wat deze druk traint, in de volgorde van het grid
        _starts, _lives = [], {}   # when each run of this press started; its live chart

        def _progress(_log):
            _step, _now = _log["step"][-1], _time.perf_counter()
            if _step == 1:   # de eerste gelogde stap van een nieuwe run
                _starts.append(_now)
            _spent, _n = _now - _starts[-1], len(_starts)
            if _n not in _lives:   # zijn instellingen; geen alleen als je run_grid een run traint die al in het logboek staat
                _title = f"Weight decay {C.number(_todo[_n - 1][0], 0, L.lang)} bij {round(100 * _todo[_n - 1][1])}% van de tabel" if _n <= len(_todo) else f"Run {_n} van deze druk op de knop"
                _lives[_n] = L.live({"title": _title, "x": "stap", "y": "accuracy", "alt": f"{_title}: de validation accuracy tot nu toe, op een logaritmische as van de stappen; gestreept de training accuracy."},
                                    C.STEPS, role="ink", low=0.0, high=1.15, window=1, xlog=True)   # ruimte boven 1 voor de naam van de lijn van training
            _live, _of = _lives[_n], f" van {len(_todo)}" if _n <= len(_todo) else ""
            _live.reference(list(zip(_log["step"], _log["train"])), "training", role="ink")   # it grows with the run
            _live.extend(list(zip(_log["step"][len(_live.xs):], _log["val"][len(_live.xs):])), f"**Run {_n}{_of} bij deze druk op de knop, stap {C.number(_step, 0, L.lang)} van {C.number(C.STEPS, 0, L.lang)}** · {C.number(_spent / 60, 1, L.lang)} minuten · training {C.number(100 * _log['train'][-1], 1, L.lang)}%, validatie {C.number(100 * _log['val'][-1], 1, L.lang)}%")

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Maak eerst oefening 1 en 2 af: dan staan je `grid` en je `run_grid` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_r, str):
            _out = mo.md({"grid": "*Je `repro/ablation.py`, `repro/grid.py` of `repro/run.py` stopt met een fout. Heb je hem na zijn check veranderd? Verwijder hem, en check zijn oefening opnieuw.*",
                          "model": "*Je ModernGPT, je data of je instellingen stoppen met een fout. Heb je `llm/modern_gpt.py`, `repro/data.py` of `repro/settings.py` na zijn check veranderd? Verwijder het bestand dat je veranderde, en check zijn oefening opnieuw.*"}[_r])
        else:
            trained = _r
            _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
            _out = mo.md(f"*Klaar: de zes runs van het grid staan in je logboek, `work/projects/repro/ablation.jsonl`.{_stand}*")
    _out
    return (trained,)


@app.cell(hide_code=True)
def demo_table(trained):
    trained  # het logboek opnieuw lezen na een run
    _mine = C.my_runs()
    _runs = _mine or [C.run(_wd, _f) for _wd in (1.0, 0.0) for _f in (0.3, 0.5, 0.9)]
    _when = lambda _s: C.number(_s, 0, L.lang) if _s is not None else "–"
    _rows = "\n".join(f"| {C.number(_l['weight_decay'], 0, L.lang)} | {round(100 * _l['fraction'])}% | {_when(_x['t'])} | {_when(_x['v'])} | {C.number(_x['times'], 1, L.lang) if _x['times'] else '–'} | {_x['verdict']} |"
                      for _l, _x in ((_l, C.read(_l)) for _l in _runs))
    _whose = "Je grid" if _mine else "Het grid van de course, tot het jouwe in je logboek staat"
    mo.md(f"""
**{_whose}**, gelezen met het plan:

| weight decay | deel | $t$ | $v$ | $v / t$ | het oordeel van het plan |
|---|---|---|---|---|---|
{_rows}
""")
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De ablation uit [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl), nu met twee instellingen in elke combinatie; de runner en het plan uit [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl), nu over een grid; en Figuur 1 (midden) en Figuur 2 (links) van de paper, nu getest op je eigen taak. Je kit heeft nu `repro/grid.py` en `repro/ablation.py`. De volgende les, *Begeleide reproductie 6: klopt het met de paper?*, herhaalt de runs over seeds en vergelijkt je getallen met die van de paper.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_grid, result_run_grid, result_figure, choice_shows):
    L.wrap_up(result_grid, result_run_grid, result_figure, choice_shows)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
