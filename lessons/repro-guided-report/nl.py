# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-guided-report. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-report", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Begeleide reproductie 6: klopt het met de paper?](/course/open/repro-guided-compare?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_unsaid():
    quick_unsaid = L.choice(
        "quick-unsaid",
        question="Welke van deze laat de grokking-paper aan jou over?",
        options={
            "batch": "De batch size, als de training data klein is",
            "lr": "De learning rate van AdamW",
            "right": "Op welke vergelijkingen elke seed traint",
        },
        correct="right",
        why={
            "batch": "Bijlage A.1.2 geeft hem: 512, of de helft van de training data als dat minder is. Je zette hem in je `SETTINGS` in [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl).",
            "lr": "Bijlage A.1.2 geeft hem: 10⁻³. Je zette hem in je `SETTINGS` in [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl).",
            "right": "De paper zegt dat de training data een willekeurig deel van de vergelijkingen is, maar niet hoe dat getrokken wordt; je kit trekt hem met de seed.",
        },
    )
    quick_unsaid
    return (quick_unsaid,)


@app.cell(hide_code=True)
def quick_check_two_pairs():
    _two = C.compare(C.runs_of(0.0, C.WIDE)[:2], C.runs_of(1.0, C.WIDE)[:2])
    quick_two_pairs = L.choice(
        "quick-two-pairs",
        question=f"Met twee paren loopt het interval van je `compare` van {C.number(10 ** _two['low'], 1, L.lang)} tot {C.number(10 ** _two['high'], 1, L.lang)} keer zo laat. Wat is dat interval waard?",
        options={
            "sure": "Veel: het ligt ruim boven 1, dus is het verschil echt",
            "right": "Weinig: een bootstrap van twee paren kan alleen uit die twee trekken",
            "none": "Helemaal niets: een interval heeft minstens tien seeds nodig om iets te zeggen",
        },
        correct="right",
        why={
            "sure": "Boven 1 met twee paren zegt weinig: het interval loopt van het ene paar naar het andere, hoe ver de seeds ook uit elkaar kunnen vallen. Zie [Begeleide reproductie 6](/course/open/repro-guided-compare?lang=nl).",
            "none": "Zo'n grens is er niet: elk paar maakt het interval meer waard, en met heel weinig is het te smal. Zie [Begeleide reproductie 6](/course/open/repro-guided-compare?lang=nl).",
            "right": "Met weinig paren lijkt het interval zekerder dan het is; meer seeds maken het de moeite waard.",
        },
    )
    quick_two_pairs
    return (quick_two_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je kit heeft nu resultaten: de runs van de belangrijkste figuur, het grid van de ablation, een vergelijking over vijf seeds. Ze staan in bestanden op je computer, waar niemand anders ze kan lezen, laat staan controleren. Een reproductie telt pas als iemand anders kan lezen wat je toetste, hoe je dat deed, wat eruit kwam en wat je niet kon controleren, en het opnieuw kan draaien. Daar is een reproductierapport voor. Tijdschriften zoals ReScience C publiceren ze, en een reproductie die niet standhield, is ook de moeite van opschrijven waard: dan hoeft de volgende niet hetzelfde te proberen.

        **Een voorbeeld uit het dagelijks leven.** Een review van een recept waar je iets aan hebt: "Twee keer gebakken. Hij rees zoals beloofd, maar had in mijn oven een kwartier langer nodig. Het recept zegt niet hoe groot de vorm is; ik nam 24 cm. De glutenvrije versie heb ik niet geprobeerd." Er staat wat geprobeerd is, hoe, wat eruit kwam, waar het afweek en wat nog open is.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de belangrijkste figuur",
                            "De ablation", "Vergelijk met de paper", "Het verslag"],
                           6, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_pick():
    _every = C.every()
    guess_pick = L.guess(
        "pick",
        question=f"Over alle vier de paren bij {round(100 * C.WIDE)}% geeft je `compare` zonder weight decay {C.number(_every, 1, L.lang)} keer zo laat. Een verslag dat alleen de ene seed gaf die het best bij de claim past: hoeveel keer zo laat zou dat zeggen?",
        options={
            "double": f"Bijna twee keer zoveel, ongeveer {round(C.reported(1))}",
            "same": "Ongeveer hetzelfde: de seeds zijn het eens",
            "less": "Minder, ongeveer 3",
        },
        reveal=f"**Ongeveer {round(C.reported(1))}:** seed {C.best_first(1)[0]} alleen geeft {C.number(C.reported(1), 1, L.lang)}, bijna twee keer de {C.number(_every, 1, L.lang)} van alle vier de paren. *Het idee* laat zien wat elke keuze geeft.",
    )
    guess_pick
    return (guess_pick,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## Het idee

    **Vijf delen, altijd in deze volgorde.** Een reproductierapport zegt wat je toetste, hoe, wat eruit kwam, waar het van de paper afwijkt en wat je niet kon controleren:

    1. **Claim:** de claim in de woorden van de paper, met waar hij staat, en de regel die je plan vóór elke run vastlegde.
    2. **Opzet:** elke instelling, de data, de seeds, het budget, en wat je bewust anders doet.
    3. **Resultaten:** elke run die je plande, met zijn getallen, zijn intervallen, en de runs die er niet kwamen.
    4. **Verschillen:** waar je opzet en je getallen afwijken van die van de paper, en de details die de paper niet geeft.
    5. **Niet gecontroleerd:** wat je niet kon toetsen, en wat daarvoor nodig is.

    **Elke run die je plande.** Welke runs in de resultaten komen, kies je niet achteraf. Hieronder staan de vijf seeds van je vergelijking bij {round(100 * C.WIDE)}%, zoals in de vorige les. Schuif om een verslag alleen de seeds te laten geven die het best bij de claim passen: de gevulde stippen staan in het verslag, de open zijn weggelaten. De dikke gestreepte lijn is het getal dat dat verslag zou geven, de dunne gestreepte het gemiddelde van alle vier de paren, en de doorgetrokken lijn 1, waar beide even snel zijn.
    """)
    return


@app.cell(hide_code=True)
def explore_pick_cell():
    def _frame(k):
        _k = int(k)
        _seeds = ", ".join(str(_s) for _s in C.best_first(_k))
        _lost = "niet gehaald"
        _every = C.every()
        return (C.chosen_picture(_k, {"title": "Een verslag van de seed die het best bij de claim past" if _k == 1 else f"Een verslag van de {_k} seeds die het best bij de claim passen", "x": "seed",
                                      "y": "keer zo laat", "none": _lost, "reported": "het verslag", "every": "alle paren",
                                      "alt": f"Hoeveel keer zo laat validatie zonder weight decay 99% haalt, seed voor seed; het verslag geeft de seeds {_seeds}: {C.number(C.reported(_k), 1, L.lang)} keer zo laat, tegen {C.number(_every, 1, L.lang)} over alle paren."}),
                f"Een verslag van seed{'' if _k == 1 else 's'} {_seeds} zegt **{C.number(C.reported(_k), 1, L.lang)} keer zo laat**; alle vier de paren geven {C.number(_every, 1, L.lang)}.")

    explore_pick = L.explore("pick", "seeds in het verslag", _frame, start=1, low=1, high=4, step=1, role="",
                             task="Hoe ver verschuift het getal als het verslag zijn seeds uitkiest? En welke seed krijgt geen getal, wat het verslag ook kiest?")
    explore_pick
    return


@app.cell(hide_code=True)
def text_idea_after():
    _every = C.every()
    _lost = [_s for _s, _x in enumerate(C.factors(len(C.SEEDS))) if _x is None][0]
    mo.vstack([
        mo.md(rf"""
        De ene seed die het best bij de claim past, geeft {C.number(C.reported(1), 1, L.lang)}, en elke seed die erbij komt, trekt het getal omlaag, tot {C.number(_every, 1, L.lang)} met alle vier de paren. Een verslag dat zijn runs uitkiest, kan elk van die getallen geven, en een lezer ziet niet welke runs het wegliet: dat is het **cherry-picking** uit [Kritisch lezen: claims tegen bewijs](/course/open/res-critical-reading?lang=nl), nu in je eigen verslag. Daarom geven de resultaten elke run die het plan noemde, beoordeeld met de regel die vooraf vastlag: alle vier de paren, en seed {_lost}, die geen getal heeft om te kiezen. Zonder weight decay haalde zijn validatie binnen 12.000 stappen geen 99%. Dat is een **negatief resultaat**, en het hoort net zo goed in het verslag: met zijn budget, en met wat het met het gemiddelde doet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **reproductierapport** | een kort verslag van een reproductie: de claim, de opzet, de resultaten, de verschillen met de paper en wat je niet kon controleren | `report(title, parts)` |
        | **negatief resultaat** | een resultaat dat laat zien dat iets niet werkte of dat een claim niet standhield; de moeite waard om te melden, omdat anderen het dan niet opnieuw hoeven te proberen | `f"not within {{budget}}"` |
        | **Markdown** | gewone tekst met een paar tekens die een pagina als opmaak toont: een regel die begint met `# ` is een titel, `## ` een kop, `*woorden*` cursief, en <code>&#124; a &#124; b &#124;</code> een rij van een tabel | `"## " + name` |
        """),
        L.watch_out("Een negatief resultaat zegt wat er niet gebeurde binnen wat je probeerde. \"Niet binnen 12.000 stappen\" is een resultaat; \"generaliseert niet\" is een claim die de run niet kan onderbouwen. Schrijf het met zijn budget, en zeg wat er nog meer voor nodig is."),
        mo.md(r"""
        ### Recept

        Zo schrijf je een reproductierapport:

        1. Schrijf de claim in de woorden van de paper, met waar hij staat, en de regel van je plan.
        2. Geef de opzet: elke instelling, de seeds, het budget, en wat je bewust anders doet.
        3. Geef elke geplande run in de resultaten, met zijn getallen en intervallen; een run die er niet kwam, met zijn budget.
        4. Noem de verschillen met de paper, en de details die hij niet geeft.
        5. Zeg wat je niet kon controleren, en geef het oordeel dat je resultaten steunen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 van het recept met de hand: een tabel met resultaten voor twee runs zonder weight decay bij 90%, seed 0, die er kwam, en seed 4, die er niet kwam. Voer de cel uit.
    """)
    return


@app.cell
def example_table():
    _budget = 12000
    _runs = [{"seed": 0, "t": 316, "v": 3548}, {"seed": 4, "t": 355, "v": None}]   # t en v van elke run
    _rows = []
    for _run in _runs:
        _v = _run["v"]
        if _v is None:   # 3. een run die er niet kwam: in de tabel, met zijn budget
            _v = f"not within {_budget}"
        _rows.append([_run["seed"], _run["t"], _v])
    print("| seed | t | v |")
    for _row in _rows:
        print(f"| {_row[0]} | {_row[1]} | {_row[2]} |")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Beide runs krijgen een rij. De `None` van seed 4 wordt "not within 12000": een lezer ziet dat de run bij zijn budget stopte, niet dat validatie er nooit zou komen, en geen `None` die hem niets zegt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een rij voor elke run

        Maak `result_rows(runs, budget)` af: voor elke run van het log, in zijn volgorde, een rij `[seed, t, v, verdict]`. `t` en `v` komen uit `first_step` van je plan, met `THRESHOLD`; waar er een `None` is, komt de tekst `f"not within {budget}"` op zijn plek; het oordeel is `verdict` van je plan. De check maakt kleine runs waarvan training en validatie 99% halen op een stap die je kunt aflezen, zoals in `run(0, 10, 1000)`, en vergelijkt de rijen. Als het goed is, wordt hij in je project bewaard als `repro/results.py`.
        """),
        L.task("Maak `result_rows` af: t en v van elke run, het budget voor een stap die niet gehaald wordt, en een rij met het oordeel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_rows = L.code_box("result-rows", lines=22)
    box_rows
    return (box_rows,)


@app.cell(hide_code=True)
def check_rows(box_rows):
    result_rows_check = L.check("result-rows", box_rows.value)
    result_rows_check
    return (result_rows_check,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "result-rows",
        [
            "Volg het uitgewerkte voorbeeld: `first_step` één keer met de training-accuracy en één keer met de validatie-accuracy, en voor elke die `None` is, de tekst op zijn plek.",
        ],
        why="Elke run krijgt een rij, wat er ook uitkwam: een tabel die de runs weglaat die er niet kwamen, vertelt alleen het deel van het verhaal dat werkte.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het sjabloon

        Maak `report(title, parts)` af: het verslag in Markdown. Eerst `"# "` en de titel; dan voor elke naam in `SECTIONS`, in die volgorde, `"## "` en de naam, en zijn tekst uit de dictionary `parts`, of `"*Not written yet.*"` als `parts` er geen tekst voor heeft; allemaal samengevoegd met een lege regel ertussen. De secties zijn in het Engels, zoals de meeste reproductierapporten, zodat iedereen het jouwe kan lezen. Als het goed is, wordt hij in je project bewaard als `repro/report.py`.
        """),
        L.task("Maak `report` af: elke sectie van `SECTIONS`, op volgorde, met zijn tekst of \"*Not written yet.*\"."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=18)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Loop over `SECTIONS`, niet over `parts`, en bepaal per naam of `parts` er een tekst voor heeft, ook als die tekst leeg is.",
        ],
        why="Met elke sectie op zijn plek, ook die nog niet geschreven zijn, ziet een lezer meteen wat het verslag nog niet zegt, en jij ook.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_not_checked():
    mo.md(r"""
    ### Oefening 3. Wat je niet kon controleren
    """)
    return


@app.cell(hide_code=True)
def explain_not_checked():
    explain_not_checked_cell = L.explain(
        "not-checked",
        question="Je reproductie toetste de grokking-claim en weight decay op optellen, met de kit van de course. Noem wat hij aan de claims van de paper niet kon controleren, en wat er nodig is om elk ervan te controleren.",
        model_answer="**De eigen taken en maat van de paper:** zijn claim over weight decay gaat over het product in S₅, gemeten als hoeveel data het kost, en zijn Figuur 1 (links) over delen; om die te controleren heb je die taken nodig, en runs bij veel fracties. **De budgetten en seeds van de paper:** 100.000 stappen voor de meeste runs, 500.000 voor de mediaan over zeven seeds van Figuur 1 (midden), en 1.000.000 voor Figuur 1 (links); daarvoor zijn 8 tot 80 keer zoveel stappen per run nodig, en meer seeds. **Wat de paper niet zegt:** op welke parameters weight decay werkt, en hoe vaak validatie gemeten werd; daarvoor heb je de code van de paper nodig, of een vraag aan de auteurs. **De run die stopte:** seed 4 zonder weight decay bij 90%, voorbij 12.000 stappen; daarvoor is een langere run nodig.",
        points=[
            "je noemt de eigen taak of maat van de paper (S₅, hoeveel data het kost, delen) als niet gecontroleerd",
            "je noemt het budget van de paper of zijn aantal seeds",
            "je noemt een detail dat de paper niet geeft, en dat zijn code of zijn auteurs daar uitsluitsel over kunnen geven",
            "bij elk zeg je wat er nodig is om het te controleren",
        ],
    )
    explain_not_checked_cell
    return (explain_not_checked_cell,)


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.md(r"""
    ### Oefening 4. Het oordeel
    """)
    return


@app.cell(hide_code=True)
def choice_verdict_cell():
    choice_verdict = L.choice(
        "verdict",
        question="Je figuur grokte in alle drie de seeds; je vergelijking vond over vier van de vijf seeds dat validatie zonder weight decay later 99% haalt, en in de vijfde niet binnen het budget. Welk oordeel komt aan het eind van je verslag?",
        options={
            "full": "Gereproduceerd: de vertraging en het effect van weight decay kloppen, dus de claims van de paper ook",
            "none": "Niet gereproduceerd: de getallen verschillen van die van de paper",
            "right": "Deels gereproduceerd: beide kloppen op optellen; de eigen opzet van de paper is niet gecontroleerd",
        },
        correct="right",
        why={
            "full": "Wat je runs laten zien, klopt op optellen, in stappen, binnen 12.000 stappen per run; de claim van de paper over weight decay gaat over S₅, in data. Een oordeel zegt niet meer dan de resultaten laten zien.",
            "none": "Andere getallen op een andere taak zijn geen mislukking: voor optellen geeft de paper alleen de beste validation accuracy na 100.000 stappen (Figuur 2, rechts), niet de stap waarop hij die haalt, met of zonder weight decay. Wat je runs toetsten, klopte.",
            "right": "Dat steunen de resultaten: wat klopte, waarop, en wat nog open is, dat de sectie *Not checked* noemt.",
        },
    )
    choice_verdict
    return (choice_verdict,)


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(r"""
        ### Je verslag

        De les schrijft je verslag met je kit: je `result_rows` en `report` zodra ze hun check halen, je `compare` op de runs van de course over vijf seeds bij 90%, je instellingen, en je runs van de belangrijkste figuur en de ablation, of die van de course waar de jouwe er nog niet zijn. De claim begint als die van de course; het deel *Not checked* schrijf je zelf, in het Engels, zoals het verslag is, en je sluit het af met je oordeel uit oefening 4. De knop onder het verslag bewaart het als `work/projects/repro/report.md`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def report_inputs():
    claim_box = mo.ui.text_area(value=C.CLAIM, label="Claim, in het Engels", full_width=True, rows=5)
    not_checked_box = mo.ui.text_area(value="", label="Not checked: wat je niet kon controleren, in het Engels", full_width=True, rows=5)
    mo.vstack([claim_box, not_checked_box])
    return claim_box, not_checked_box


@app.cell(hide_code=True)
def demo_report(claim_box, not_checked_box, result_rows_check, result_report):
    result_rows_check, result_report  # opnieuw uitvoeren als een oefening zijn module bewaart
    _r = C.demo(claim_box.value, not_checked_box.value)
    report_text = _r["text"]
    _whose = {"result_rows": "`result_rows`", "report": "`report`", "compare": "`compare`"}
    _mine = [_whose[_k] for _k in _r["theirs"]]
    _notes = [f"*Geschreven met je {' en '.join(_mine)}; die van de course springen in voor de rest.*" if _mine else "*Geschreven met de functies van de course: de jouwe nemen het over zodra ze hun check halen.*",
              "*Je runs van de belangrijkste figuur.*" if _r["my_figure"] else "*De runs van de course voor de belangrijkste figuur: de jouwe zijn er nog niet.*",
              "*Je grid van de ablation.*" if _r["my_grid"] else "*Het grid van de course voor de ablation: het jouwe is er nog niet.*"]
    _body = report_text.replace("\n## ", "\n#### ").replace("# ", "### ", 1)   # zijn koppen onder die van de les
    mo.vstack([
        mo.Html(f"<div style='border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.2rem 1rem; overflow-x: auto'>{mo.md(_body).text}</div>"),
        mo.md(" ".join(_notes)),
    ])
    return (report_text,)


@app.cell(hide_code=True)
def save_cell():
    save_button = mo.ui.run_button(label="Bewaar het verslag")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def demo_save(save_button, report_text):
    _msg = None
    if save_button.value:
        try:
            _msg = f"*Bewaard als `{C.save(report_text)}`. De volgende les deelt het, met je code, instellingen en seeds.*"
        except OSError:
            _msg = "*Het verslag kon niet bewaard worden: bestaat de map `work/projects/repro/`, en mag de course erin schrijven?*"
    mo.output.replace(mo.md(_msg)) if _msg else mo.output.clear()
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het plan uit [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl), nu de eerste sectie van je verslag; de runs uit [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl) en het grid uit [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl), nu zijn tabellen; en de vergelijking en de indeling uit [Begeleide reproductie 6](/course/open/repro-guided-compare?lang=nl), nu zijn resultaten en verschillen. Je kit heeft nu `repro/results.py` en `repro/report.py`, en je verslag. In de volgende les, *Je werk delen zodat anderen het kunnen herhalen*, publiceer je het met je code, instellingen en seeds, zodat iemand anders het opnieuw kan draaien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rows_check, result_report, explain_not_checked_cell, choice_verdict):
    L.wrap_up(result_rows_check, result_report, explain_not_checked_cell, choice_verdict)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
