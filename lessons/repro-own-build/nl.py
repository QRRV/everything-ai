# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-build. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-build", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 2: claim en plan](/course/open/repro-own-claim?lang=nl) en [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_verdict():
    quick_verdict = L.choice(
        "quick-verdict",
        question="Een plan zegt 3 seeds, minstens 0,5 lager. De winsten van de seeds zijn 0,9, 0,2 en 0,6. Wat geeft `verdict` terug?",
        options={
            "one": "`\"not supported\"`: één seed zit onder 0,5",
            "few": "`\"too few runs\"`: de seed onder 0,5 telt niet mee",
            "right": "`\"supported\"`: de mediaan van de drie winsten is 0,6",
        },
        correct="right",
        why={
            "one": "De mediaan beslist, niet elke seed: één seed die misgaat, kan hem niet ver verschuiven. Zie [Je reproductie 2](/course/open/repro-own-claim?lang=nl).",
            "few": "Alle drie de seeds van het plan zijn gedraaid, en elke telt mee. Zie [Je reproductie 2](/course/open/repro-own-claim?lang=nl).",
            "right": "De middelste van 0,2, 0,6 en 0,9 is 0,6, en dat is minstens 0,5.",
        },
    )
    quick_verdict
    return (quick_verdict,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="In `run_seed` van Begeleide reproductie 4 legt de seed drie dingen vast. Welke hoort daar niet bij?",
        options={
            "right": "Hoeveel stappen hij draait",
            "weights": "De startwaarden van de weights",
            "batches": "De volgorde van de batches",
        },
        correct="right",
        why={
            "weights": "`torch.manual_seed(seed)` legt ze vast, vlak voordat het model gemaakt wordt. Zie [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).",
            "batches": "Die komen uit een eigen generator, `torch.Generator().manual_seed(seed)`. Zie [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl).",
            "right": "De stappen zijn een eigen argument; de seed legt de splitsing, de startwaarden en de batches vast.",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je plan zegt wat je draait; nu bouw je het. De lessen hierna draaien je experiment vaak: over seeds, op het budget van je plan, in de ablation. Dat kan alleen als elke run een paar beloftes houdt: hij geeft één getal terug, hetzelfde als je het nog eens vraagt, en zijn seed en zijn setting veranderen het echt. Een run die er een breekt, geeft nog steeds getallen, maar getallen waar je plan niets mee kan, en niets waarschuwt je. In deze les bouw je het experiment, en een contract dat die beloftes in een paar seconden controleert.

        **Een voorbeeld uit het dagelijks leven.** Een recept met "een snufje zout" en "bakken tot het gaar is" geeft in elke keuken een andere taart, en wat één verandering doet, proef je niet. Een recept met 5 gram en 25 minuten op 180 graden geeft twee keer dezelfde taart, en als je één ding verandert, proef je wat dat ene ding doet.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           2, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_short():
    guess_short = L.guess(
        "short",
        question=f"Het experiment van de course voor de dropout-paper, {C.STEPS} stappen gedraaid met seed 0: met dropout is de testfout {C.number(C.SHORT_RUN['claim'], 1, L.lang)}%, zonder {C.number(C.SHORT_RUN['other'], 1, L.lang)}%. Wat zegt dat over de claim van de paper?",
        options={
            "fails": "Hij reproduceert niet: met dropout is de testfout duidelijk hoger",
            "right": f"Nog niets: {C.STEPS} stappen laten zien dat de code draait, niet de claim",
            "seeds": f"Nog te weinig: met vijf seeds van {C.STEPS} stappen zou het blijken",
        },
        reveal=f"**Nog niets: {C.STEPS} stappen laten zien dat de code draait, niet de claim.** Na {C.STEPS} stappen heeft geen van beide netwerken veel geleerd, en dropout, dat willekeurig neuronen uitzet, leert in het begin langzamer. Meer seeds van zo'n korte run meten hetzelfde alleen preciezer. Het plaatje hieronder laat zien vanaf welk budget dropout loont.",
    )
    guess_short
    return (guess_short,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Eén functie, twee settings.** Elk experiment van je kit is één functie, `run(setting, seed, steps)`. `"claim"` is de setting van de claim van je plan, `"other"` die waarmee je hem vergelijkt; ze delen alle code behalve het ene ding dat je plan verandert, en de functie geeft de metric van je plan terug, één getal. Die twee namen blijven hetzelfde, wat je paper ook is, zodat de lessen hierna elk experiment kunnen draaien.

    **De seed legt alles vast wat willekeurig is.** `torch.manual_seed(seed)` aan het begin van de run legt de startwaarden van de weights en de dropout masks vast, en de volgorde van de batches krijgt een eigen generator, `torch.Generator().manual_seed(seed)`. Dan zien seed 0 van `"claim"` en seed 0 van `"other"` dezelfde batches, ook als maar een van de twee dropout masks trekt: het uitgewerkte voorbeeld laat zien waarom daar een eigen generator voor nodig is. Dezelfde aanroep geeft dan precies hetzelfde getal: dat is determinisme.

    **Een contract.** De beloftes van een run: hij geeft een getal terug, geen NaN; dezelfde aanroep geeft hetzelfde getal; en de stappen, de seed en de setting veranderen het elk. Een run die er een breekt, geeft nog steeds getallen: met een seed die niets doet, zijn je drie seeds drie keer dezelfde run. Een onbekende setting moet hem stoppen: zoek de setting op in een dict, dan stopt een typefout met een KeyError in plaats van iets anders te draaien.

    **Delen voor de sanity checks.** Naast `run` geeft je experiment de sanity checks van de volgende les wat ze bekijken, gemaakt door dezelfde code: `make_model(setting, seed)`, het model vóór elke training, dat `run` ook gebruikt; `one_batch()`, één kleine batch van de training data; `K`, hoeveel getallen het model per voorbeeld geeft; en `FROM_SCRATCH`, of het begint met willekeurige weights of met een getraind model.

    **Een korte run controleert de code.** Het contract draait 20 stappen: genoeg om in een paar seconden te zien dat de code zijn beloftes houdt. Over de claim zegt het niets.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **contract** | wat een functie belooft aan wie hem aanroept: wat hij aanneemt, wat hij teruggeeft, en wat hij doet met wat hij niet kent |
    | **determinisme** | precies dezelfde uitkomst krijgen als je dezelfde code op dezelfde data nog eens draait |

    Het plaatje: het experiment van de course voor de dropout-paper, de oplossing van oefening 2, over seed 0, 1 en 2. Voor elk budget aan stappen, elk een eigen run, een stip voor de winst van elke seed: hoeveel procentpunt lager de testfout is met dropout. De ringen zijn de medianen, en de stippellijn is het *minstens* van het plan van de course, 0,5. Beweeg de slider, of sleep over het plaatje: een dunne lijn markeert zijn budget, en de zin onder het plaatje geeft de runs van dat budget.
    """)
    return


@app.cell(hide_code=True)
def explore_gains_cell():
    _labels = {"title": "Wat dropout wint, per budget", "x": "stappen van de run", "y": "winst, procentpunt", "line": "minstens 0,5",
               "alt": "De winst van dropout per seed tegen het aantal stappen van een run, op een log-as van 100 tot 12.800 stappen. Tot 400 stappen is elke winst onder 0, tot -3,8; bij 800 en 1600 stappen zijn de medianen -1,0 en 0,3; vanaf 3200 stappen zijn ze 0,5, 0,4 en 0,7, rond een stippellijn bij 0,5."}

    def _frame(s):
        _s = int(s)
        _k = C.BUDGETS.index(_s)
        _r = C.RUNS[_s]
        _say = "de regel van het plan zegt gesteund" if C.supported(_s) else "de regel van het plan zegt niet gesteund"
        _with = ", ".join(C.number(_x, 1, L.lang) for _x in _r["claim"])
        _without = ", ".join(C.number(_x, 1, L.lang) for _x in _r["other"])
        return (C.gains_picture(_k, _labels),
                f"Na {C.whole(_s, L.lang)} stappen: met dropout {_with}% testfout, zonder {_without}%. Mediaan van de winst {C.number(C.median_gain(_s), 1, L.lang).replace('-', '−')}: {_say}.")

    explore_gains = L.explore("budget", "stappen", _frame, start=C.BUDGETS[0], values=list(C.BUDGETS), role="",
                              task="Vanaf welk budget ligt dropout voor? En hoe ver komt het boven de stippellijn?")
    explore_gains
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Tot 800 stappen ligt dropout achter, in de mediaan tot 3,7 procentpunt: willekeurig neuronen uitzetten maakt het leren in het begin trager. Bij 1600 stappen ligt het 0,3 voor, en vanaf 3200 stappen ongeveer een half procentpunt, dicht bij het *minstens* van het plan: een mediaan van 0,5 bij 3200 stappen, 0,4 bij 6400 en 0,7 bij 12.800. Een run van 20 stappen zegt daar niets over, en het contract dus ook niet. Wat het wel zegt: de getallen die je volgende lessen meten, komen uit een run die zijn beloftes houdt.
        """),
        L.watch_out("Kies de stappen van je hoofdresultaat voordat je naar winsten als deze kijkt: uit de opzet van de paper en de uren van je plan. Het plan van de course voor de dropout-paper noemt zelf geen aantal stappen, en het plaatje laat zien wat dat openlaat: volgens zijn regel steunen 3200 en 12.800 stappen de claim, 1600 en 6400 niet. Het budget kiezen waarop de winst er het mooist uitziet, is je regel kiezen na de resultaten."),
        mo.md(r"""
        ### Recept

        Zo bouw je het experiment van je plan:

        1. Schrijf in een dict op wat `"claim"` en `"other"` zijn: het ene ding dat je plan verandert.
        2. Bouw het deel dat beide settings delen, uit de lessen waarop je paper bouwt: de data, het model in `make_model`, één batch in `one_batch`, de training in `run`.
        3. Leg alles wat willekeurig is vast met de seed: `torch.manual_seed(seed)` in `make_model`, en een eigen generator voor de volgorde van de batches.
        4. Geef de metric van je plan terug, één getal.
        5. Houd de run aan het contract met 20 stappen, vóór elke lange run.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 van het recept, en waarom de batches een eigen generator nodig hebben. Seed 0 in beide settings: `"claim"` trekt eerst een dropout mask, `"other"` niet. Voer de cel uit.
    """)
    return


@app.cell
def example_generators():
    # "claim": eigen generator voor batches, en mask
    torch.manual_seed(0)
    _g = torch.Generator().manual_seed(0)
    torch.rand(512)
    _c = torch.randint(0, 6000, (3,), generator=_g).tolist()
    # "other": dezelfde generator, en geen mask
    torch.manual_seed(0)
    _g = torch.Generator().manual_seed(0)
    _o = torch.randint(0, 6000, (3,), generator=_g).tolist()
    print("eigen generator:", _c, _o)

    # hetzelfde, met de ene globale generator
    torch.manual_seed(0)
    torch.rand(512)
    _c = torch.randint(0, 6000, (3,)).tolist()
    torch.manual_seed(0)
    _o = torch.randint(0, 6000, (3,)).tolist()
    print("één generator:", _c, _o)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met een eigen generator is de eerste batch van seed 0 in beide settings dezelfde drie afbeeldingen, wat de run verder ook trekt. Met de ene generator die `torch.manual_seed` zet, gebruikt het dropout mask van `"claim"` willekeurige getallen op, en schuift alles wat daarna getrokken wordt op: de twee runs van seed 0 zien andere batches. Dan vergelijkt de winst van een seed runs die in meer dan dropout verschillen, en verliest het koppelen per seed waar het voor was.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_contract():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Het contract

        Maak `contract(run, steps)` af: hij geeft het eerste van de vijf antwoorden in zijn commentaar terug dat klopt voor `run`, in die volgorde, of `None` als de run zijn beloftes houdt. Roep de run aan zoals het commentaar zegt, te beginnen met `a = run("claim", 0, steps)`. Nieuw is `math.isnan(a)`: `True` als `a` NaN is, de float die geen getal is, en die een training teruggeeft die vastloopt. Een run die helemaal geen getal teruggeeft, zoals `None`, laat `math.isnan` al stoppen met een fout; de check van oefening 2 zegt je dat. De check probeert verzonnen runs, elk genoemd naar wat hij doet.
        """),
        L.task("Maak `contract` af: roep de run aan, en geef het eerste antwoord terug dat klopt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_contract():
    box_contract = L.code_box("contract", lines=18)
    box_contract
    return (box_contract,)


@app.cell(hide_code=True)
def check_contract(box_contract):
    result_contract = L.check("contract", box_contract.value)
    result_contract
    return (result_contract,)


@app.cell(hide_code=True)
def hints_contract():
    L.hints(
        "contract",
        [
            "Eerst `a = run(\"claim\", 0, steps)` en de controle op NaN, dan één `if` per antwoord, in de volgorde van het commentaar, elk met een eigen aanroep.",
            "\"steps do nothing\" vergelijkt `run(\"claim\", 0, 2 * steps)` met `a`, en \"seed does nothing\" vergelijkt `run(\"claim\", 1, steps)` ermee.",
        ],
        why="Elke `if` is één belofte van het contract, in de volgorde van het commentaar, zodat het eerste antwoord het eerste is om te repareren.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_experiment():
    mo.md(r"""
    ### Oefening 2. Je experiment

    Bouw het experiment van je plan in het vak: `run(setting, seed, steps)` en zijn delen, voor de paper die je koos. De hulp hieronder zegt wat `"claim"` en `"other"` voor jouw paper zijn, en wat je kunt hergebruiken. De check houdt je run aan het contract met 20 stappen, en noemt de eerste belofte die hij breekt; hij kijkt ook of een onbekende setting hem stopt. Houdt hij ze allemaal, dan staat hij in `work/projects/own/experiment.py`. Lees data met een pad vanaf de map van de course, zoals `np.load("data/fashion-mnist-small.npz")`. Op een NVIDIA-kaart zet je het model en de batches erop binnen `run`: de lessen hierna roepen `run` op elke computer op dezelfde manier aan. Voor de dropout-paper kost deze oefening ongeveer een half uur; voor een andere paper, die je zelf bouwt, reken op een avond.
    """)
    return


@app.cell(hide_code=True)
def help_for_your_paper():
    mo.vstack([
        mo.accordion({"Hulp voor je paper": mo.md(C.outline_text(C.saved_choice(), C.saved_plan(), L.lang))}),
        L.task("Bouw `run` voor je paper, en check hem: de check noemt de eerste belofte die hij breekt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_experiment():
    box_experiment = L.code_box("experiment", lines=30)
    box_experiment
    return (box_experiment,)


@app.cell(hide_code=True)
def check_experiment(box_experiment):
    result_experiment = L.check("experiment", box_experiment.value)
    result_experiment
    return (result_experiment,)


@app.cell(hide_code=True)
def hints_experiment():
    L.hints(
        "experiment",
        [
            "Begin met het deel dat beide settings delen, en draai het met één setting: geeft het één getal terug?",
            "Dan het ene ding dat verschilt, in de dict `SETTINGS`, en de seed: `torch.manual_seed(seed)` in `make_model`, `torch.Generator().manual_seed(seed)` voor de batches in `run`, die zijn model maakt met `make_model`.",
            "Voor de dropout-paper: een MLP met twee hidden layers van 512 op Fashion-MNIST, dropout van 0,2 op de input en 0,5 op de hidden layers, SGD met momentum, en een learning rate die over de stappen afloopt naar 0.",
        ],
        why="Dit is het experiment van de course voor de dropout-paper, waarvan het plaatje de runs laat zien. Voor een andere paper blijft de vorm hetzelfde: een dict met de settings, eerst de seed, het deel dat beide delen, en de metric als één getal.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je experiment uit je kit, seed 0, 1 en 2 in beide settings, 200 stappen, en wat de regel van je plan erover zegt. 200 stappen zijn niet het budget van je plan: dat draaien de volgende lessen.
    """)
    return


@app.cell(hide_code=True)
def demo_button():
    run_demo = mo.ui.run_button(label="Draai seed 0, 1 en 2 met 200 stappen")
    run_demo
    return (run_demo,)


@app.cell(hide_code=True)
def demo_experiment(run_demo, result_experiment):
    result_experiment  # opnieuw uitvoeren als oefening 2 je experiment bewaart
    _out = mo.md("*Druk op de knop: hij draait je `own/experiment.py` zes keer.*")
    if run_demo.value:
        _runs = C.demo_runs()
        if _runs is None:
            _out = mo.md("*Maak eerst oefening 2 af: dan zit je experiment in je kit, en draait deze cel het.*")
        elif _runs == "slow":
            _out = mo.md("*Eén run van 200 stappen van je experiment duurt langer dan twee minuten, dus de demo stopt. Het hoofdresultaat draait het budget van je plan zonder die grens.*")
        elif _runs == "error":
            _out = mo.md("*Je `own/experiment.py` stopt met een fout bij 200 stappen, meer dan de 20 en 40 van de check. Krijgt elke stap een batch, ook na één keer door de data? Of heb je het bestand na de check veranderd? Verwijder het dan, en check oefening 2 opnieuw.*")
        else:
            _said = C.demo_verdict(_runs, C.saved_plan())
            _rule = f"Je regel zegt `{_said}`: na 200 stappen gaat dat over de code, niet over de claim." if _said else "Met een plan en een regel in je kit zou je regel ook deze runs beoordelen."
            _out = mo.vstack([mo.Html(C.demo_table(_runs, {"seed": "seed"}, L.lang)), mo.md(f"*{_rule}*")])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Wat een korte run zegt

    Je experiment houdt het contract met 20 stappen. Zeg wat dat je vertelt, en wat nog niet.
    """)
    return


@app.cell(hide_code=True)
def explain_why_short():
    explain_why_short_cell = L.explain(
        "why-short",
        question="Je experiment houdt het contract met 20 stappen. Wat vertelt dat je erover, en wat nog niet?",
        model_answer="**Wat het me vertelt:** mijn run geeft één getal terug, hetzelfde bij dezelfde aanroep, en zijn stappen, zijn seed en zijn setting veranderen het elk: drie seeds zijn drie runs, en de twee settings verschillen echt. Een onbekende setting stopt hem, dus een typefout kan niet het verkeerde experiment draaien. **Wat nog niet:** of de claim klopt. Na 20 stappen hebben de runs van de course een testfout van 67% met dropout en 53% zonder: dropout leert in het begin langzamer, en ligt pas vanaf ongeveer 1600 stappen voor, vanaf 3200 met een half procentpunt. Of de mijne dat doet, is voor het hoofdresultaat, op het budget van mijn plan, dat ik kies voordat ik kijk.",
        points=[
            "je noemt wat het contract over je run laat zien",
            "je zegt waarom dat voor je plan telt, zoals seeds die echt verschillen",
            "je zegt dat een run die zo kort is niets over de claim zegt",
            "je zegt waar de claim getest wordt: in het hoofdresultaat, op het budget van je plan",
        ],
    )
    explain_why_short_cell
    return (explain_why_short_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het plan van [Je reproductie 2](/course/open/repro-own-claim?lang=nl), nu als code; de seed van [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl), die de splitsing, de startwaarden en de batches vastlegde; en [Dropout](/course/open/nn-dropout?lang=nl), dat het leren in het begin trager maakt. In de volgende les, *Je reproductie 4: sanity checks*, richt je de sanity checks van je kit op de delen van je experiment, vóór elke lange run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_contract, result_experiment, explain_why_short_cell):
    L.wrap_up(result_contract, result_experiment, explain_why_short_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
