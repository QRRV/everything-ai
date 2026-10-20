# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-compare. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-compare", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 6: de ablation](/course/open/repro-own-ablation?lang=nl) en [Is het verschil echt?](/course/open/res-significance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_keeps():
    quick_keeps = L.choice(
        "quick-keeps",
        question="Een ablation houdt 120% van de winst van de claim. Wat zegt de regel van de helft, en wat betekent het aandeel?",
        options={
            "broken": "Dat de check stuk is: geen ablation kan meer houden dan de hele winst",
            "right": "Dat het niet van het deel afhangt: zonder doet de methode het zelfs iets beter",
            "depends": "Dat het resultaat van het deel afhangt, want 120% ligt ver van de 100% van de eigen winst van de claim",
        },
        correct="right",
        why={
            "broken": "Een aandeel boven 1 betekent dat de winst zonder het deel, in de mediaan, groter is dan die van de claim: zonder het deel doet de methode het beter dan met. Dat kan, en het is het vermelden waard. Zie [Je reproductie 6](/course/open/repro-own-ablation?lang=nl).",
            "depends": "De regel kijkt naar minder dan de helft: 120% is meer dan de helft, dus hangt het resultaat niet van het deel af. Zie [Je reproductie 6](/course/open/repro-own-ablation?lang=nl).",
            "right": "Er is minstens de helft over, dus volgens de regel is het deel niet nodig; dat het zonder beter gaat, is een bevinding op zich.",
        },
    )
    quick_keeps
    return (quick_keeps,)


@app.cell(hide_code=True)
def quick_check_paired():
    quick_paired = L.choice(
        "quick-paired",
        question="Twee settings draaiden met dezelfde seeds. Waarom geeft een gepaarde bootstrap een smaller interval dan een die elke setting apart resamplet?",
        options={
            "twice": "Paren geven hem twee keer zoveel getallen om uit te trekken: twee runs per seed",
            "boot": "Een gepaarde bootstrap trekt minder resamples",
            "right": "Wat een seed met beide runs doet, valt weg uit hun verschil",
        },
        correct="right",
        why={
            "twice": "Paren geven hem half zoveel getallen: één verschil per seed, niet twee runs. Zie [Is het verschil echt?](/course/open/res-significance?lang=nl).",
            "boot": "Beide trekken zoveel resamples als je vraagt. Zie [Is het verschil echt?](/course/open/res-significance?lang=nl).",
            "right": "Een gelukkige seed verlaagt beide runs evenveel; hun verschil houdt wat de setting doet, en de spreiding tussen seeds valt weg.",
        },
    )
    quick_paired
    return (quick_paired,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je hoofdresultaat zegt wat je regel over je runs zegt. Het zegt nog niet of ze kloppen met de paper. De paper heeft zijn eigen getal, uit zijn eigen opzet, vaak één run. Verschilt het jouwe, dan zijn er twee vragen: is het verschil meer dan je seeds verklaren, en zo ja, waar komt het vandaan? Een verschil is geen mislukking: de reden ervan vinden is vaak het nuttigste wat een reproductie teruggeeft.

        **Een voorbeeld uit het dagelijks leven.** Je brood rijst minder dan op de foto bij het recept. Eén brood kan pech zijn; als het brood na brood tekortschiet, niet meer. Dan kijk je naar het meel, de warmte van de keuken en hoe lang het stond.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           6, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_real():
    guess_real = L.guess(
        "real",
        question="In het hoofdresultaat van de course verlaagt dropout de testfout met 0,4, 0,7 en 1,2 procentpunt voor zijn drie seeds, met een fout zonder dropout van 12,9% tot 13,6%. De paper: met dropout gaat de fout op MNIST van 1,60% naar 1,35%, een winst van 0,25 procentpunt. In procentpunten wint de course meer. En als aandeel van de fout zonder dropout?",
        options={
            "right": "Minder: zijn winst is klein naast zijn grote fout",
            "more": "Daar ook meer: een grotere daling van de fout is een groter deel ervan",
            "same": "Ongeveer evenveel: dezelfde methode doet op allebei ongeveer hetzelfde",
        },
        reveal="**Minder.** De 0,25 procentpunt van de paper is 16% van zijn 1,60%; de 0,4 tot 1,2 procentpunt van de course zijn maar 3% tot 9% van zijn 12,9% tot 13,6%, een fout van ongeveer acht keer die van de paper. In procentpunten wint de course meer, als aandeel minder: dezelfde runs verschillen in tegengestelde richting van de paper. Of drie seeds genoeg zijn om dat te zeggen, laat *Het idee* zien.",
    )
    guess_real
    return (guess_real,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Jouw winst tegen die van de paper.** De bootstrap van [Is het verschil echt?](/course/open/res-significance?lang=nl) geeft de gemiddelde winst van je seeds en een 95%-interval: de gemiddelden van veel resamples van je winsten, zonder de laagste en de hoogste 2,5%. De paper geeft één getal. Ligt dat buiten je interval, dan verklaren je seeds het verschil niet: het is echt. Met drie seeds is het interval wel alleen het bereik van je seeds, en dat noemde [Is het verschil echt?](/course/open/res-significance?lang=nl) veel te smal om op te vertrouwen: dan telt ook hoe ver het getal erbuiten ligt.

    **In de eenheid van de claim.** Een winst kun je tellen in punten van de metric, of als aandeel van de metric zonder de methode. Van 1,60% naar 1,35% is in de paper 0,25 procentpunt, of 16% van zijn fout. Als de data verschillen, kunnen die twee tegengestelde verhalen vertellen: zeg over welke eenheid de claim gaat, en geef beide.

    **Zoek dan de reden.** Leg je opzet naast die van de paper: de data, het netwerk, de training en de meting; wat de paper zegt, wat hij niet zegt, en wat jij met opzet anders deed, zoals in Begeleide reproductie 6. Een reden is een gok tot een run hem test, zoals een ablation.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | relatieve winst | de winst als aandeel van de metric zonder de methode, in % |

    Het plaatje: het hoofdresultaat van de course voor de dropout-paper bij 12.800 stappen tegen de eigen getallen van de paper, in twee panelen: in procentpunten testfout, en als aandeel van de fout zonder dropout. Elke stip is de winst van één seed, de grijze band zijn interval, de ring zijn gemiddelde; de gestreepte lijn is die van de paper. Beweeg de schuif om één, twee of drie seeds te gebruiken.
    """)
    return


@app.cell(hide_code=True)
def explore_seeds_cell():
    _labels = {"points": "Winst in procentpunten testfout", "share": "Winst als aandeel van de fout zonder dropout",
               "x_points": "procentpunten", "x_share": "%", "paper": "paper",
               "alt": "De winsten van de drie seeds van de course bij 12.800 stappen tegen die van de paper. In procentpunten: 0,4, 0,7 en 1,2, een interval van 0,4 tot 1,2; de 0,25 van de paper ligt eronder. Als aandeel van de fout zonder dropout: 3,1%, 5,1% en 8,8%, een interval van 3,1% tot 8,8%; de 15,6% van de paper ligt erboven."}
    _c, _o = C.course_results()

    def _frame(k):
        _k = int(k)
        _p = C.interval(C.gains(_c, _o)[:_k])
        _s = C.interval(C.shares(_c, _o)[:_k])
        _pg, _ps = C.paper_gain(C.PAPER), C.paper_share(C.PAPER)
        _seeds = "1 seed" if _k == 1 else f"{_k} seeds"
        _say = (f"{_seeds}. In procentpunten: een gemiddelde van {C.short(round(_p[0], 2), L.lang)}, interval {C.short(round(_p[1], 2), L.lang)} tot {C.short(round(_p[2], 2), L.lang)}; "
                f"de {C.short(_pg, L.lang)} van de paper ligt {'erbuiten' if not _p[1] <= _pg <= _p[2] else 'erbinnen'}. "
                f"Als aandeel: {C.short(round(_s[0], 1), L.lang)}%, interval {C.short(round(_s[1], 1), L.lang)}% tot {C.short(round(_s[2], 1), L.lang)}%; "
                f"de {C.short(round(_ps, 1), L.lang)}% van de paper ligt {'erbuiten' if not _s[1] <= _ps <= _s[2] else 'erbinnen'}.")
        return C.compare_picture(_k, _labels), _say

    explore_seeds = L.explore("seeds", "seeds", _frame, start=3, low=1, high=3, step=1, role="",
                              task="Met hoeveel seeds is het interval meer dan een punt? En aan welke kant ervan ligt de winst van de paper, in elke eenheid?")
    explore_seeds
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Met één seed is het interval een punt: een bootstrap van één winst kan geen spreiding zien, dus lijkt hij zeker terwijl hij niets weet. Met drie loopt het van de laagste seed tot de hoogste: in 1 op de 27 resamples vallen alle drie de trekkingen op de laagste seed, meer dan de 2,5% die het interval eraf haalt. Dat is het interval dat [Is het verschil echt?](/course/open/res-significance?lang=nl) veel te smal noemde om op te vertrouwen, dus kijk hoe ver het getal van de paper erbuiten ligt. Als aandeel van de fout ligt de 15,6% van de paper ver boven de hoogste seed, 8,8%, verder dan de seeds uiteenlopen: daar is de winst van de course duidelijk kleiner. In procentpunten ligt de 0,25 van de paper net onder de laagste seed, 0,4, terwijl de seeds 0,8 procentpunt uiteenlopen: de winst van de course lijkt groter, maar drie seeds laten dat niet sterk zien. Dropout verlaagt de testfout op beide datasets, en dat zegt de claim; hoeveel, verschilt tussen MNIST, met een fout van 1,60%, en Fashion-MNIST, met ongeveer 13,4%.
        """),
        L.watch_out("Met drie seeds is een getal van de paper net buiten je interval zwak bewijs; een getal dat er verder buiten ligt dan je seeds uiteenlopen, is sterk, en meer seeds maken het interval meer waard. Het getal van de paper komt zonder spreiding over seeds: zijn eigen ruis zit niet in je interval. En de 1,60% en 1,35% van de paper horen bij twee verschillende netwerken: het beste dat hij noemt zonder dropout, en een met dropout dat hij verder verbetert, tot onder 1%. Zeg waar elk getal van is."),
        mo.md(r"""
        ### Recept

        Zo vergelijk je je reproductie met de paper:

        1. Zoek de eigen getallen van de paper, met de methode en zonder, en schrijf op waar ze staan en waar ze van zijn.
        2. Geef de gemiddelde winst van je seeds met zijn bootstrap-interval, bij je budget.
        3. Zeg of de winst van de paper buiten het interval ligt, en hoe ver: ligt hij er verder buiten dan je seeds uiteenlopen, dan is het verschil echt; net erbuiten, met weinig seeds, is het zwak.
        4. Doe hetzelfde als aandeel van de metric zonder de methode, en zeg over welke eenheid de claim gaat.
        5. Leg de opzetten naast elkaar, en noem het verschil waarvan je denkt dat het dit verklaart; test het met een run als dat kan.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De winst van de paper en die van de course, in procentpunten en als aandeel van de fout zonder dropout. Voer de cel uit.
    """)
    return


@app.cell
def example_units():
    _with, _without = 1.35, 1.60       # de paper, zijn resultaten op MNIST
    _gain = round(_without - _with, 2)
    print("paper:", _gain, "procentpunt,", round(100 * _gain / _without, 1), "%")
    _claim = [12.5, 12.9, 12.4]        # de course: met dropout
    _other = [12.9, 13.6, 13.6]        # zonder
    _mine = [round(o - c, 1) for c, o in zip(_claim, _other)]
    _share = [round(100 * g / o, 1) for g, o in zip(_mine, _other)]
    print("course:", _mine, "procentpunt,", _share, "%")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De 0,25 procentpunt van de paper is 15,6% van zijn 1,60; de winsten van de course zijn groter in procentpunten en kleiner als aandeel, omdat zijn fout zonder dropout ruim acht keer zo groot is. Een winst als aandeel deelt door de metric van de andere setting met dezelfde seed, dus blijft hij gepaard.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vs():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Jouw winst tegen die van de paper

        Maak `vs_paper(gains, paper_gain, n_boot, seed)` af: de vergelijking van je kit. Hij geeft het gemiddelde van je winsten, het 0,025- en het 0,975-kwantiel van de gemiddelden van `n_boot` resamples ervan, en `real`: of de winst van de paper buiten dat interval ligt. De check probeert vier winsten tegen een 0,3 van een paper, de winsten van de course in procentpunten en als aandeel, een paper waarvan de winst erbinnen ligt, vijf seeds, en één.
        """),
        L.task("Maak `vs_paper` af: de resamples met teruglegging, de twee kwantielen, en `real`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vs():
    box_vs = L.code_box("vs-paper", lines=14)
    box_vs
    return (box_vs,)


@app.cell(hide_code=True)
def check_vs(box_vs):
    result_vs = L.check("vs-paper", box_vs.value)
    result_vs
    return (result_vs,)


@app.cell(hide_code=True)
def hints_vs():
    L.hints(
        "vs-paper",
        [
            "De loop van `paired_bootstrap` van [Is het verschil echt?](/course/open/res-significance?lang=nl), op je winsten: die zijn al de verschillen van de paren.",
            "`np.quantile(means, [0.025, 0.975])` geeft beide grenzen; dan is `real` waar als `paper_gain` onder de ene of boven de andere ligt.",
        ],
        why="De resamples laten zien hoeveel het gemiddelde van een paar seeds kan bewegen; een winst van de paper buiten dat bereik is er een die je seeds niet zouden geven.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_numbers():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De getallen van de paper

        Zoek de eigen getallen van de paper voor je claim op, waar je plan zegt dat de paper hem laat zien, of in de tekst of tabel die dat resultaat in getallen geeft: de metric met de methode en zonder, in de eenheid van je metric. Schrijf op waar ze staan, en waar ze van zijn: twee keer hetzelfde netwerk, of het beste van elke soort. De check bewaart ze in je kit, als `work/projects/own/paper_numbers.json`.
        """),
        L.task("Vul de twee getallen van de paper in, en in een zin waar ze staan en waar ze van zijn."),
    ])
    return


@app.cell(hide_code=True)
def numbers_inputs():
    _old = C.saved_numbers() or {}
    numbers_with = mo.ui.number(start=-1e9, stop=1e9, step=0.01, value=_old.get("with") or 0, label="Het getal van de paper met de methode")
    numbers_without = mo.ui.number(start=-1e9, stop=1e9, step=0.01, value=_old.get("without") or 0, label="Het getal van de paper zonder")
    numbers_where = mo.ui.text_area(label="Waar in de paper, en waar de getallen van zijn", full_width=True, rows=3, value=_old.get("where", ""))
    _plan = C.P.saved_plan()
    _head = (mo.md(f"*Je plan: {_plan.get('metric')}; de paper laat het zien in {_plan.get('shown_in')}.*") if _plan
             else mo.md("*Nog geen plan: schrijf het eerst in [Je reproductie 2](/course/open/repro-own-claim?lang=nl).*"))
    mo.vstack([_head, numbers_with, numbers_without, numbers_where])
    return numbers_where, numbers_with, numbers_without


@app.cell(hide_code=True)
def check_numbers(numbers_where, numbers_with, numbers_without):
    result_numbers = L.check("paper-numbers", {"with": numbers_with.value, "without": numbers_without.value, "where": numbers_where.value})
    result_numbers
    return (result_numbers,)


@app.cell(hide_code=True)
def hints_numbers():
    L.hints(
        "paper-numbers",
        [
            "Begin waar je plan zegt dat de paper de claim laat zien; de tekst rond een tabel zegt vaak in woorden wat de tabel in getallen laat zien.",
            "Controleer dat beide getallen in de eenheid van je metric zijn, zoals de testfout in %, en zeg of ze bij één netwerk horen of bij twee.",
        ],
        why="Voor de dropout-paper: **1,35 met dropout, 1,60 zonder.** " + C.PAPER_WHERE[L.lang],
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je hoofdresultaat bij je budget, tegen de getallen van de paper in je kit, gelezen met je `vs_paper`: in punten en als aandeel van de andere setting.
    """)
    return


@app.cell(hide_code=True)
def demo_compare(result_vs, result_numbers):
    result_vs, result_numbers  # opnieuw uitvoeren als een oefening een deel van de kit bewaart
    _r = C.my_comparison()
    if _r is None:
        _out = mo.md("*Je vergelijking verschijnt hier zodra je kit je plan heeft, je budget, de runs van je hoofdresultaat daarbij, je `vs_paper` van oefening 1 en de getallen van de paper van oefening 2.*")
    elif _r == "error":
        _out = mo.md("*Je `own/compare.py` stopt met een fout, of geeft iets anders dan zijn dictionary: check oefening 1 opnieuw.*")
    else:
        _row = lambda _name, _x, _p, _d: (f"| {_name} | {C.signed(round(_x['mean'], _d), L.lang)} | {C.signed(round(_x['low'], _d), L.lang)} tot {C.signed(round(_x['high'], _d), L.lang)} | "
                                           f"{C.signed(round(_p, _d), L.lang)} | {'echt' if _x['real'] else 'niet aangetoond'} |")
        _out = mo.md(f"""
| eenheid | jouw gemiddelde | jouw interval | die van de paper | het verschil |
|---|---|---|---|---|
{_row("punten", _r["points"], _r["paper_gain"], 2)}
{_row("aandeel, %", _r["share"], _r["paper_share"], 1) if _r["share"] is not None else "| aandeel, % | | | | geen: zonder de methode is de metric niet boven 0 |"}
""")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Waarom het verschilt

    Je vergelijking ligt voor je. Zeg wat hij laat zien, en wat het volgens jou verklaart.
    """)
    return


@app.cell(hide_code=True)
def explain_why_differs():
    explain_why_differs_cell = L.explain(
        "why-differs",
        question="Is het verschil tussen jouw resultaat en dat van de paper echt, en in welke eenheid? Wat verklaart het volgens jou, en hoe kan een run dat testen?",
        model_answer="**Duidelijk als aandeel, zwak in procentpunten:** als aandeel van de fout loopt mijn winst van 3% tot 9%, en de 16% van de paper ligt er ver boven. In procentpunten is mijn gemiddelde winst 0,77, met een interval van 0,4 tot 1,2, en de 0,25 van de paper ligt er net onder; maar met drie seeds is dat interval alleen het bereik van mijn seeds, veel te smal om op te vertrouwen. De claim, een duidelijk lagere testfout, klopt in allebei. **Mijn beste gok** voor de grootte: mijn data. Fashion-MNIST is moeilijker, dus is mijn fout ongeveer acht keer die van de paper, en met 6000 trainingsafbeeldingen in plaats van 60.000 kan een netwerk meer overfitten, wat dropout in procentpunten meer te doen geeft. Bovendien horen de twee getallen van de paper bij twee netwerken, en komen ze zonder spreiding over seeds. **Een test:** hetzelfde experiment op de 6000 cijfers van QMNIST in de course, afbeeldingen zoals die van MNIST: zakt mijn winst in procentpunten dan richting die van de paper, dan ligt het aan de soort data, niet aan het aantal afbeeldingen. Meer seeds zouden laten zien of mijn winst in procentpunten echt verschilt.",
        points=[
            "je zegt of de winst van de paper buiten je interval ligt, en hoe ver, in elke eenheid",
            "je noemt een verschil in de opzet dat het kan verklaren",
            "je zegt hoe een run dat kan testen",
        ],
    )
    explain_why_differs_cell
    return (explain_why_differs_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De bootstrap van [Is het verschil echt?](/course/open/res-significance?lang=nl) en het naast elkaar leggen van opzetten van [Begeleide reproductie 6](/course/open/repro-guided-compare?lang=nl), nu voor je eigen paper; en het hoofdresultaat van [Je reproductie 5](/course/open/repro-own-figure?lang=nl). In de volgende les, *Je reproductie 8: verslag en delen*, schrijf je alles op, publiceer je je code en resultaten, en vertel je de auteurs wat je vond.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_vs, result_numbers, explain_why_differs_cell):
    L.wrap_up(result_vs, result_numbers, explain_why_differs_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
