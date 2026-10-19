# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-guided-sanity. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-sanity", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Begeleide reproductie 2: bouw het model en de data](/course/open/repro-guided-build?lang=nl) en [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chance():
    quick_chance = L.choice(
        "quick-chance",
        question="Vóór elke training lag de loss van je ModernGPT op de antwoorden dicht bij ln 99 = 4,60. Waarom dat getal?",
        options={
            "trained": "Een model van deze grootte haalt ongeveer 4,60 op modulair rekenen, vóór en na de training",
            "batch": "De batch heeft 512 vergelijkingen, en 4,60 is ongeveer ln 99, het aantal antwoorden erin",
            "right": "Hij geeft elk van de 99 tokens ongeveer dezelfde kans, 1/99, en −ln(1/99) = ln 99",
        },
        correct="right",
        why={
            "trained": "Training brengt de loss ver daaronder: een model dat de antwoorden geleerd heeft, geeft ze een kans dicht bij 1, en −ln 1 = 0. Zie [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl).",
            "batch": "De loss hangt niet af van de grootte van de batch: het is een gemiddelde over de vergelijkingen. De 99 is het aantal tokens waaruit het model kiest. Zie [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl).",
            "right": "Een model dat niets geleerd heeft, gokt gelijk verdeeld over zijn 99 tokens; de cross-entropy van het goede is dan −ln(1/99).",
        },
    )
    quick_chance
    return (quick_chance,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Je netwerk leert één mini-batch van 64 plaatjes uit het hoofd: na 300 stappen is zijn loss erop 0,03. Wat zegt dat?",
        options={
            "solved": "Het doet het ook goed op alle data: het heeft de taak geleerd",
            "right": "Het netwerk en zijn trainingsstap kunnen leren, en meer niet",
            "wrong": "Er is iets mis: een netwerk hoort geen 64 plaatjes uit het hoofd te leren",
        },
        correct="right",
        why={
            "solved": "Slagen voor de test betekent niet dat het netwerk het goed doet op alle data: een netwerk dat 64 plaatjes uit het hoofd leerde, kan nieuwe plaatjes slecht lezen. Zie [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).",
            "wrong": "Eén batch uit het hoofd leren is precies wat de test vraagt: een netwerk dat werkt, kan het, en dit netwerk laat zien dat het het kan. Zie [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl).",
            "right": "De overfit-test is een test, geen manier om te trainen: hij sluit een kapot netwerk, een kapotte start of een kapotte trainingsstap uit, en meer niet.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(rf"""
        ## Waarom

        Je reproductie traint lang. In de run van de course bij de fraction van je plan haalde validatie de 99% pas bij stap {C.number(C.RUN_STEPS, 0, L.lang)}: bij {C.number(C.SECONDS_PER_STEP, 2, L.lang)} seconde per stap op de CPU van de course, ongeveer {round(C.RUN_STEPS * C.SECONDS_PER_STEP / 60)} minuten, en het budget van de paper, {C.number(C.BUDGET, 0, L.lang)} stappen, is meer dan {int(C.BUDGET * C.SECONDS_PER_STEP / 3600)} uur. Een fout die je na die run vindt, kostte je de run, en een fout die je nooit vindt, kost je de reproductie: een curve die niet springt, zegt niets over de paper als je model het toch niet had kunnen leren. Dus vóór elke lange run een paar snelle tests waarvan je de uitkomst vooraf kent. Samen kosten ze ongeveer een minuut.

        **Een voorbeeld uit het dagelijks leven.** Voor een lange rit controleer je de brandstof en de banden. Dat kost een minuut, tegen halverwege stilstaan, en je weet wat je moet zien voordat je kijkt.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de belangrijkste figuur",
                            "De ablation", "Vergelijk met de paper", "Het verslag"],
                           2, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_place():
    guess_place = L.guess(
        "place",
        question="Je ModernGPT heeft een fout: hij leest zijn gok voor $c$ op de plek van $a$, `model(x)[:, 0]`, in plaats van op de plek van $=$. Welke van drie snelle tests vangt die volgens jou?",
        options={
            "shape": "De vorm van de logits: die is niet wat de instellingen geven",
            "first": "De loss vóór de training: die ligt ver van ln 99",
            "right": "Eén batch uit het hoofd leren: de loss gaat niet naar bijna 0",
        },
        reveal="**Eén batch uit het hoofd leren.** De logits hebben in beide gevallen dezelfde vorm, en vóór elke training gokt elke plek ongeveer gelijk verdeeld, dus de eerste loss is op beide plekken ongeveer ln 99. Wat verschilt, is wat elke plek gezien heeft; *Het idee* laat zien wat dat doet.",
    )
    guess_place
    return (guess_place,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **sanity check** is een snelle test die vanzelfsprekend moet slagen voordat je een lange run vertrouwt. Elk van de drie hieronder heeft een getal dat de opzet je vooraf geeft, dus je weet wat je moet zien voordat je kijkt.

    1. **De vormen.** Eén batch van 512 vergelijkingen van 4 tokens gaat erin: 512 × 4. Eruit komen de logits: 512 × 4 × 99, een rij van $p + 2$ scores op elke plek. Een ander getal hier betekent een model dat met andere instellingen gebouwd is dan je denkt.
    2. **De loss vóór de training.** Een model dat niets geleerd heeft, gokt ongeveer gelijk verdeeld over zijn 99 tokens, dus zijn loss op de antwoorden is ongeveer ln 99 = 4,60. Veel hoger: het begint zeker en fout, zoals met weights die te groot beginnen. Veel lager: het weet iets wat het nog niet kan weten, zoals een antwoord dat in zijn input lekt.
    3. **Eén batch uit het hoofd leren.** Train op die ene batch alleen, steeds opnieuw, zonder weight decay: de test vraagt alleen of het model kan leren, en weight decay trekt de andere kant op. Een model dat werkt, leert 512 vergelijkingen in een paar honderd stappen, en zijn loss gaat naar bijna 0. Lukt dat niet, dan is het model of zijn trainingsstap kapot.

    Hieronder staan twee runs van de ModernGPT van de course met de instellingen van de paper, op de eerste 512 vergelijkingen van je training data, met de AdamW van de paper zonder weight decay. Eén van de twee heeft de fout uit de gok: hij leest zijn gok op de plek van $a$. Schuif met de slider, of sleep over de grafiek, om het aantal stappen te zetten: de grafiek toont de loss van beide op de batch bij elke stap, de run met de fout gestippeld, en de grijze stippellijn is ln 99.
    """)
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    def _frame(step):
        right, wrong = C.RUNS["right"][step], C.RUNS["wrong"][step]
        return (C.picture(step, {"title": "Eén batch, uit het hoofd geleerd", "x": "stap", "y": "loss op de batch", "even": "gelijk verdeeld gokken: ln 99",
                                 "right": "leest bij =", "wrong": "leest bij a: de fout",
                                 "alt": f"De loss op één batch van 512 vergelijkingen over 150 stappen, voor twee ModernGPT's. Die op de plek van = leest, daalt van 4,76 naar 0,0041; die op de plek van a leest, daalt ook van 4,79, eerst zelfs sneller, maar blijft rond 1,78. Een stippellijn markeert ln 99 = 4,60. Na {step} stappen: {C.number(right, 4, L.lang)} en {C.number(wrong, 4, L.lang)}."}),
                f"""Na {step} stappen is de loss van het model dat bij = leest <span class="c-loss ea-num">{C.number(right, 4, L.lang)}</span>, en die van het model dat bij a leest <span class="c-loss ea-num">{C.number(wrong, 4, L.lang)}</span>.""")

    explore_steps = L.explore("steps", "stap", _frame, start=20, low=0, high=150, step=5, role="", task="Vanaf welke stap is de loss van het model dat bij = leest onder 0,01? Waar blijft het andere steken?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(rf"""
    Vanaf stap 90 is het model dat bij $=$ leest onder 0,01, op 0,0098: het heeft de 512 vergelijkingen uit het hoofd geleerd. Het model dat bij $a$ leest, daalt eerst zelfs sneller, maar blijft steken rond 1,78. Zijn vormen klopten, en zijn eerste loss, 4,79, lag dicht bij ln 99, net als die van het andere: alleen één batch uit het hoofd leren ving hem.

    **Waar een check blijft steken, zegt vaak waarom.** Op de plek van $a$ heeft het model alleen $a$ gezien. In de batch komen de meeste waarden van $a$ met verschillende $b$, dus met verschillende antwoorden, en het model kan die niet uit elkaar houden. Komt een $a$ $k$ keer voor, dan kan het op zijn best elk van die $k$ antwoorden $1/k$ geven, een loss van $\ln k$ voor elk van die vergelijkingen. Gemiddeld over de batch:
    """)
    return


@app.cell(hide_code=True)
def text_idea_formula():
    mo.vstack([
        L.formula(r"[[loss:\mathcal{L}_{\min}]] = \frac{1}{n} \sum_{a} k_a \ln k_a"),
        mo.md(rf"""
        met $k_a$ het aantal vergelijkingen met die $a$ onder de $n$ van de batch. Voor de eerste 512 vergelijkingen van de training data is dat {C.number(C.floor(), 2, L.lang)}: het model met de fout blijft net boven de laagste loss die het kan halen. Een loss die blijft steken op een getal dat je kunt uitrekenen uit wat het model ziet, wijst dus aan wat het niet kan zien.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **sanity check** | een snelle test die vanzelfsprekend moet slagen voordat je een lange run vertrouwt, met een uitkomst die je vooraf kent | `sanity(model, x, y, vocab, steps)` |
        | loss vóór de training | de loss van een model dat niets geleerd heeft: ongeveer ln van het aantal tokens waaruit het kiest | `math.log(vocab)` |
        | $k_a$ | het aantal vergelijkingen in de batch met deze $a$ | `counts[a]` |
        | $\mathcal{{L}}_{{\min}}$ | de laagste loss die een model kan halen met wat het ziet | `best_loss(rows)` |
        """),
        L.watch_out("Een check die slaagt, is geen bewijs. Zouden de antwoorden in je data fout zijn, bijvoorbeeld (a · b) mod p in plaats van (a + b) mod p, dan slagen alle drie de checks toch: een model kan elke batch uit het hoofd leren, ook een met foute antwoorden. Sanity checks vangen een kapot model en een kapotte trainingsstap; of het experiment dat van de paper is, zeggen de instellingen, elk met zijn plek in de paper."),
        mo.md(r"""
        ### Recept

        Zo controleer je een model vóór een lange run:

        1. Neem één batch van de training data, zo groot als die van de echte run.
        2. Vergelijk de vormen van de batch en van de logits met wat de instellingen geven.
        3. Vergelijk de loss vóór de training met ln van het aantal tokens.
        4. Train op die batch alleen, zonder weight decay, een paar honderd stappen lang: de loss hoort naar bijna 0 te gaan. Blijft hij hoger steken, zoek dan uit wat het model kan zien, en welke laagste loss dat toelaat.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De getallen die je mag verwachten bij een kleine batch van 4 vergelijkingen van optellen modulo 5, voordat je iets draait. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_numbers():
    import math

    _p = 5
    _rows = [(1, 2, 3), (1, 4, 0), (2, 2, 4), (1, 0, 1)]   # 1. één batch: vier vergelijkingen (a, b, c)
    # 2. de vormen: 4 vergelijkingen van 4 tokens erin, en p + 2 scores op elke plek eruit
    print("x:", (len(_rows), 4), "logits:", (len(_rows), 4, _p + 2))
    # 3. de loss vóór de training: gelijk verdeeld gokken over de p + 2 tokens
    print("eerste loss: ongeveer", round(math.log(_p + 2), 2))
    # 4. een model dat alleen a ziet: a = 1 komt 3 keer voor, a = 2 één keer
    print("laagste loss als het alleen a ziet:", round((3 * math.log(3) + 1 * math.log(1)) / 4, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `x` is 4 × 4 en de logits 4 × 4 × 7, en de eerste loss hoort ongeveer ln 7 = 1,95 te zijn. De batch uit het hoofd leren hoort de loss naar bijna 0 te brengen; een model dat alleen $a$ ziet, blijft op zijn best op 0,824 steken. De drie vergelijkingen met $a = 1$ hebben drie verschillende antwoorden, 3, 0 en 1, dus elk kost op zijn best ln 3, en die met $a = 2$ kost ln 1 = 0: $(3 \ln 3 + 1 \ln 1) / 4$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Waar hij blijft steken

        Maak `best_loss(rows)` af: de laagste loss die een model kan halen op de vergelijkingen `rows`, elk `(a, b, c)`, als het alleen `a` ziet. Tel hoe vaak elke `a` voorkomt, zoals in het uitgewerkte voorbeeld, en tel $k_a \ln k_a$ op over elke `a`, gedeeld door het aantal vergelijkingen. In deze data komt elk paar van `a` en `b` één keer voor, dus de vergelijkingen van één `a` hebben verschillende antwoorden. De check roept `best_loss([(3, 0, 3), (3, 1, 4), (0, 2, 2), (0, 0, 0), (3, 4, 2)])` aan, en andere batches.
        """),
        L.task("Maak `best_loss` af: tel de vergelijkingen van elke `a`, en dan het gemiddelde van $k_a \\ln k_a$."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-loss", lines=11)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-loss", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-loss",
        [
            "Tellen per `a` is tellen met een dictionary: is `a` al een key, dan gaat zijn aantal 1 omhoog; zo niet, dan begint het bij 1.",
            "In het uitgewerkte voorbeeld kwam `a = 1` 3 keer voor, en kostten zijn vergelijkingen samen `3 * math.log(3)`. Doe hetzelfde voor elke `a` in `counts`, met zijn eigen aantal.",
        ],
        why="Een `a` die één keer voorkomt, kost ln 1 = 0: het model kan zijn ene antwoord leren. Alleen een `a` met meer dan één antwoord houdt de loss boven 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_overfit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Leer één batch uit het hoofd

        Maak `overfit_one_batch(model, x, y, steps, lr)` af voor een GPT: `steps` stappen van AdamW, elke stap op dezelfde `x` en `y`, met de loss alleen op de antwoorden, op de laatste plek, die van $=$. Hij bewaart de loss van elke stap, gemeten vóór zijn update, in `losses`. De optimizer staat er al: de betas van de paper, zonder weight decay. De check traint kleine modellen op verzonnen batches en vergelijkt de losses en het model erna. Als hij goed is, wordt hij in je project bewaard als `repro/overfit.py`.
        """),
        L.task("Maak `overfit_one_batch` af: de loss op de laatste plek, bewaard, en een stap van AdamW, `steps` keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overfit():
    box_overfit = L.code_box("overfit", lines=13)
    box_overfit
    return (box_overfit,)


@app.cell(hide_code=True)
def check_overfit(box_overfit):
    result_overfit = L.check("overfit", box_overfit.value)
    result_overfit
    return (result_overfit,)


@app.cell(hide_code=True)
def hints_overfit():
    L.hints(
        "overfit",
        [
            "`model(x)` geeft één rij logits voor elke plek; het antwoord `c` wordt op de laatste plek gegokt. Hoe neem je alleen die rijen?",
            "`losses` bewaart getallen, geen tensors: welke methode van een tensor geeft het getal erin? Daarna komen de drie regels van elke trainingsstap, in hun volgorde.",
        ],
        why="Met de loss alleen op de laatste plek traint de test precies wat de echte run traint; een loss over elke plek zou het model vragen `c` te gokken waar het `b` nog niet gezien heeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bug():
    mo.md(r"""
    ### Oefening 3. Welke fout?
    """)
    return


@app.cell(hide_code=True)
def choice_bug_cell():
    choice_bug = L.choice(
        "which-bug",
        question=f"Je draait de drie checks op een model van een medeleerling, op de eerste 512 vergelijkingen van de training data. De vormen kloppen, de eerste loss is 4,81, en na {C.STEPS} stappen waarin het de batch uit het hoofd leert, is de loss 1,79 en daalt hij nauwelijks nog. `best_loss` van de batch geeft {C.number(C.floor(), 2, L.lang)}. Waar zit de fout het waarschijnlijkst?",
        options={
            "right": "Het model gokt op een plek die b nog niet gezien heeft",
            "lr": "De learning rate is te klein: het heeft meer stappen nodig",
            "answers": "De antwoorden in de data zijn fout",
        },
        correct="right",
        why={
            "lr": f"Met een te kleine learning rate daalt de loss langzaam, maar hij blijft dalen. Deze blijft steken, net boven {C.number(C.floor(), 2, L.lang)}: de laagste loss van een model dat alleen `a` ziet.",
            "answers": "Een model kan elke batch uit het hoofd leren, ook een met foute antwoorden: de loss zou toch naar bijna 0 gaan. Zie het let-op in *Het idee*.",
            "right": f"Hij blijft net boven {C.number(C.floor(), 2, L.lang)} steken, de laagste loss die een model kan halen dat alleen `a` ziet: het leest zijn gok op een plek vóór `b`, zoals die van `a`.",
        },
    )
    choice_bug
    return (choice_bug,)


@app.cell(hide_code=True)
def text_exercise_sanity():
    mo.vstack([
        mo.md(rf"""
        ### Oefening 4. De sanity check van je kit

        Maak `sanity(model, x, y, vocab, steps)` af, de drie checks in één functie, voor elk model dat een batch tokens neemt en op elke plek een rij logits geeft. Hij geeft een rapport terug: `shape`, of de logits `len(x)` bij het aantal plekken bij `vocab` zijn; `first`, of de loss vóór de training binnen {C.number(C.FIRST_TOLERANCE, 1, L.lang)} van ln `vocab` ligt; en `learns`, of de loss van de laatste stap onder {C.number(C.LEARNED, 2, L.lang)} ligt nadat het model de batch met je `overfit_one_batch` uit het hoofd heeft geleerd, bij de learning rate van de paper, `1e-3`. De twee losses komen erbij. De check draait hem op kleine modellen, en op modellen met een fout. Als hij goed is, wordt hij in je project bewaard als `repro/sanity.py`.
        """),
        L.task("Maak `sanity` af: de vorm, de eerste loss en de losses van de batch uit het hoofd leren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sanity():
    box_sanity = L.code_box("sanity", lines=18)
    box_sanity
    return (box_sanity,)


@app.cell(hide_code=True)
def check_sanity(box_sanity):
    result_sanity = L.check("sanity", box_sanity.value)
    result_sanity
    return (result_sanity,)


@app.cell(hide_code=True)
def hints_sanity():
    L.hints(
        "sanity",
        [
            "`tuple(logits.shape)` maakt van de vorm een tuple, die je met `==` met een andere tuple kunt vergelijken. Welke drie getallen hoort hij te hebben, uit `x` en `vocab`?",
            "De eerste loss komt uit de `logits` die je al hebt, op dezelfde plek en op dezelfde manier als de loss in je `overfit_one_batch`. Daarna traint je `overfit_one_batch`, bij de learning rate van de paper.",
        ],
        why="Meet de eerste loss vóór het leren van de batch: daarna heeft het model geleerd, en zegt zijn loss niets meer over hoe het begon.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Wat je project nu kan

    De knop draait je `sanity` uit je project twee keer, op de eerste batch van je training data: op je ModernGPT met je instellingen, en op hetzelfde model met de fout uit de gok erin gezet, zodat hij zijn gok op de plek van $a$ leest. Elk leert de batch {C.STEPS} stappen lang uit het hoofd; samen kost dat op een CPU ongeveer een halve minuut. Waar een deel van je kit ontbreekt of kapot is, springt de versie van de course in.
    """)
    return


@app.cell(hide_code=True)
def sanity_button_cell():
    sanity_button = mo.ui.run_button(label="Draai je sanity check")
    sanity_button
    return (sanity_button,)


@app.cell(hide_code=True)
def demo_sanity(sanity_button, result_overfit, result_sanity):
    result_overfit, result_sanity  # run again when an exercise saves a part of the kit
    _ready = result_overfit.state == "correct" and result_sanity.state == "correct"
    if not sanity_button.value:
        _out = mo.md("*Druk op de knop om je sanity check te draaien.*" if _ready else "*Maak eerst oefening 2 en 4 af, en druk dan op de knop: hij draait je `sanity` met je `overfit_one_batch`.*")
    else:
        with mo.status.spinner(title="Je sanity check leert de batch uit het hoofd, twee keer: ongeveer een halve minuut"):
            _r = C.demo()
        if _r is None:
            _out = mo.md("*Maak eerst oefening 2 en 4 af: dan staan je `overfit_one_batch` en je `sanity` in je project, en gebruikt deze cel ze.*")
        elif isinstance(_r, str):
            _out = mo.md({"slow": "*Je sanity check doet er hier langer dan drie minuten over voor één model. Doet je `overfit_one_batch` of je `sanity` meer werk dan nodig is, zoals meer stappen dan `steps`?*",
                          "sanity": "*Je `repro/sanity.py` of `repro/overfit.py` stopt met een fout. Heb je hem na zijn check veranderd? Verwijder hem, en check zijn oefening opnieuw.*",
                          "model": "*Je ModernGPT of je data stopt met een fout. Heb je `llm/modern_gpt.py`, `repro/data.py` of `repro/settings.py` na zijn check veranderd? Verwijder het bestand dat je veranderde, en check zijn oefening opnieuw.*"}[_r])
        else:
            _y, _b = _r["reports"]["yours"], _r["reports"]["planted"]
            _yes = lambda _v: "slaagt" if _v else "**zakt**"
            _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
            _verdict = ("Je model slaagt voor alle drie de checks, en je check vangt de fout die erin gezet is" if all(_y[_k] for _k in ("shape", "first", "learns")) and not _b["learns"]
                        else "Niet elke check kwam uit zoals verwacht: kijk naar de rij die afwijkt")
            _out = mo.md(f"""
| check | je ModernGPT | met de fout | wat je mag verwachten |
|---|---|---|---|
| vorm van de logits | {_yes(_y['shape'])} | {_yes(_b['shape'])} | {_r['batch']} × 4 × {_r['vocab']} |
| loss vóór de training | {C.number(_y['first_loss'], 2, L.lang)}: {_yes(_y['first'])} | {C.number(_b['first_loss'], 2, L.lang)}: {_yes(_b['first'])} | ongeveer ln {_r['vocab']} = {C.number(_r['even'], 2, L.lang)} |
| loss na {_r['steps']} stappen op de batch | {C.number(_y['last_loss'], 4, L.lang)}: {_yes(_y['learns'])} | {C.number(_b['last_loss'], 4, L.lang)}: {_yes(_b['learns'])} | onder {C.number(C.LEARNED, 2, L.lang)}; een model dat alleen a ziet, op zijn best {C.number(_r['floor'], 2, L.lang)} |

{_verdict}.{_stand}
""")
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Je kit heeft nu een check die je vóór een lange run op elk model kunt loslaten: je eigen, een veranderd model, of dat van een medeleerling. Draai hem opnieuw na elke verandering aan het model of de trainingsstap. In de volgende les komt de lange run, en die begint pas als deze drie slagen.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De overfit-test uit [Project: een diep netwerk laten trainen](/course/open/nn-deep-net-project?lang=nl), nu voor een GPT en met een loss alleen op het antwoord; de loss vóór de training uit [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl), nu een check met een getal om te halen; de cross-entropy van een model dat gelijk verdeeld gokt; en, zoals in [Gradient checking](/course/open/nn-gradient-check?lang=nl), een test waarvan je de goede uitkomst kent voordat je hem draait. Je kit heeft nu `repro/overfit.py` en `repro/sanity.py`. De volgende les, *Begeleide reproductie 4: reproduceer de belangrijkste figuur*, traint lang genoeg om validatie te zien springen, en tekent de figuur opnieuw uit je eigen logboek.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_overfit, result_best, choice_bug, result_sanity):
    L.wrap_up(result_overfit, result_best, choice_bug, result_sanity)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
