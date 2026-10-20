# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-claim. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-claim", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 1: kies een paper](/course/open/repro-own-choose?lang=nl) en [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_margin():
    quick_margin = L.choice(
        "quick-margin",
        question="Een paper heeft een NVIDIA-kaart met 8 GB nodig, en de course schat 2 uur. Je computer heeft een NVIDIA-kaart met 12 GB, en je hebt één avond van 4 uur. Past hij, met een marge van 3?",
        options={
            "right": "Nee: 2 uur keer 3 is 6, meer dan de 4 die je hebt",
            "card": "Ja: 12 GB is meer dan de 8 die hij nodig heeft",
            "hours": "Ja: 2 uur past in 4",
        },
        correct="right",
        why={
            "card": "De kaart is genoeg, maar ook de tijd moet passen, met de marge. Zie [Je reproductie 1](/course/open/repro-own-choose?lang=nl).",
            "hours": "Zonder marge wel; met een marge van 3 telt de schatting als 6 uur. Zie [Je reproductie 1](/course/open/repro-own-choose?lang=nl).",
            "right": "De kaart is genoeg, de avond niet. Een nacht wel, of een kleinere paper.",
        },
    )
    quick_margin
    return (quick_margin,)


@app.cell(hide_code=True)
def quick_check_before():
    quick_before = L.choice(
        "quick-before",
        question="Het plan van Begeleide reproductie 1 noemt een run grokked als de validatie 99% haalt minstens 10 keer zo laat als de training. Een run haalt het 8 keer zo laat, en de curve ziet er voor jou uit als grokking. Wat zegt het plan?",
        options={
            "curve": "`\"grokked\"`: de curve laat het zien",
            "change": "Zet `DELAY` op 8: 8 zit dicht bij 10",
            "right": "`\"no delay\"`: 8 keer is minder dan de 10 van het plan",
        },
        correct="right",
        why={
            "curve": "De curve lijkt er misschien op, maar het plan zegt minstens 10 keer zo laat, en dat zei het vóór de run. Zie [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl).",
            "change": "`DELAY` veranderen nadat je een run hebt gezien, past de regel aan de uitkomst aan, en dan kan het plan niet meer falen. Zie [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl).",
            "right": "Het plan is vóór de run geschreven, dus 8 keer zo laat is ook een uitkomst, geen reden om de regel te veranderen.",
        },
    )
    quick_before
    return (quick_before,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In de begeleide reproductie koos de course de getallen van het plan voor je: 99%, en 10 keer zo laat. Voor je eigen paper schrijf je ze zelf. Vóór je eerste run beslis je wat je meet, waartegen, met hoeveel seeds, wat het kost, en welk resultaat zou betekenen dat de claim niet reproduceert. Een plan zonder dat laatste stuk kan niet falen, en een reproductie die niet kan falen, vertelt niemand iets, jou ook niet.

        **Een voorbeeld uit het dagelijks leven.** Een weddenschap die je na de wedstrijd beslecht, met een regel die je dan kiest, win je altijd. Vóór de wedstrijd kun je "ze winnen met twee doelpunten verschil of meer" verliezen, en juist daardoor betekent winnen iets.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           1, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_extra():
    guess_extra = L.guess(
        "extra",
        question="Je plan zegt 3 seeds, en dat de claim klopt als zijn setting in de mediaan over de seeds minstens 0,5 voorligt. Na 3 seeds is de mediaan 0,4. Teleurgesteld draai je er nog 2, en over alle 5 is de mediaan 0,6. Wat zegt je plan?",
        options={
            "yes": "Gesteund: over 5 seeds is de mediaan 0,6",
            "right": "Niet gesteund: de 3 seeds van het plan geven 0,4",
            "wait": "Nog geen oordeel: draai seeds tot het duidelijk is",
        },
        reveal="**Niet gesteund: de 3 seeds van het plan geven 0,4.** De 2 extra seeds draaide je omdat de uitkomst tegenviel, en anders had je ze niet gedraaid: ze meetellen is cherry-picking. Zet ze in je verslag, als extra; het oordeel komt uit het plan.",
    )
    guess_extra
    return (guess_extra,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een claim als vergelijking.** Elke claim op de lijst kun je testen als een vergelijking van twee settings die in één ding verschillen: met en zonder dropout, twee layers en één, willekeurige labels en de echte. Je plan noemt ze allebei, wat hetzelfde blijft, en één getal per run, de metric. Een claim van *beter* zegt welke kant de metric opgaat met zijn setting, hoger of lager.

        **Per seed, dan de mediaan.** Je draait beide settings met dezelfde seeds: seed 0 van de ene naast seed 0 van de andere, die hun geluk delen, zoals de volgorde van de data. Per seed is de *winst* hoe ver de setting van de claim voorligt, in de richting van de claim. De mediaan van de winsten beslist: één seed die misgaat, kan die niet ver verschuiven.

        **Hoe groot.** Het plan zegt hoe groot de mediaan van de winst minstens moet zijn om als de claim te tellen: een effectgrootte, in de eenheden van de metric. Te klein, en ruis tussen seeds gaat door voor de claim; te groot, en een echt effect faalt. Een claim van *even goed*, zoals een netwerk dat willekeurige labels ongeveer even goed leert als de echte, heeft een negatief getal: minstens -1 betekent hooguit 1 achter. Zo'n claim klopt ook als geen van beide settings werkt; daarom zegt het plan wat de andere moet halen, zoals ongeveer 100% training-accuracy met de echte labels.

        **Vóór de eerste run, met de seeds die je plant.** Runs die je toevoegt omdat de uitkomst tegenvalt, tellen niet. In de wetenschap heet het vooraf opschrijven een preregistratie.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **effectgrootte** | hoe groot een verschil is, in de eenheden van wat je meet: 1,2 procentpunt testfout, bijvoorbeeld |
        | **preregistratie** | je plan opschrijven, met de regel die beslist, voordat je een resultaat ziet, en je eraan houden |
        | **winst** | hier: voor één seed, hoe ver de setting van de claim voorligt op de andere, in de richting van de claim |

        Het plaatje: verzonnen uitkomsten voor de dropout-paper, de winst van elke seed als een stip, in procentpunten testfout, en de mediaan van elke rij als een ring. Bovenaan werkt dropout; onderaan maakt het niets uit, en verschillen alleen de seeds. De stippellijn door beide rijen is je *minstens*: schuif hem, of sleep hem.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_counts_cell():
    _labels = {"title": "Wat telt als de claim", "x": "winst per seed, procentpunt testfout",
               "works": "dropout werkt", "none": "geen effect", "yes": "gesteund", "no": "niet gesteund",
               "alt": "Twee rijen van drie stippen op een as van winsten van -1 tot 2,5. Bovenste rij, dropout werkt: winsten 0,9, 1,2 en 1,5, mediaan 1,2. Onderste rij, geen effect: winsten -0,3, 0,1 en 0,4, mediaan 0,1. Een stippellijn door beide rijen markeert de kleinste winst die telt."}

    def _frame(d):
        _d = round(d, 1)
        _works, _none = C.verdicts_at(_d)
        _say = ("allebei tellen ze als de claim, en ruis gaat door voor werkende dropout." if _works and _none
                else "het plan houdt ze uit elkaar." if _works else "geen van beide telt, en zelfs dropout die werkt, haalt je plan niet.")
        return C.counts_picture(_d, _labels), f"Minstens {C.number(_d, 1, L.lang).replace('-', '−')}: {_say}"

    explore_counts = L.explore("at-least", "minstens", _frame, start=0, low=-0.5, high=2, step=0.1, role="",
                               task="Zoek waar het plan de twee uitkomsten uit elkaar houdt. Waar in dat stuk zou jij hem zetten, vóór de eerste run?")
    explore_counts
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Van 0,2 tot 1,2 houdt het plan de twee uitkomsten uit elkaar; daaronder telt ruis als de claim, en daarboven telt zo weinig dat dropout die werkt toch faalt. Waar je hem zet, komt uit twee dingen die je vóór je runs kunt weten: hoeveel twee seeds van dezelfde setting verschillen, zoals in [Eén run is geen resultaat](/course/open/res-seeds-variance?lang=nl), en hoe groot het effect volgens de paper is. Zet hem boven de ruis, en niet hoger dan het kleinste effect dat je nog de claim van de paper zou noemen.

        **De kosten.** Eén seed van beide settings kost een aantal uur; het plan kost dat keer de seeds, keer je marge, en dat moet passen in de tijd die je hebt.
        """),
        L.watch_out("De regel veranderen nadat je je resultaten hebt gezien, is geen correctie maar een nieuw plan. Doe je het toch, zeg het dan in je verslag, met het oordeel van allebei."),
        mo.md(r"""
        ### Recept

        Zo schrijf je het plan van je reproductie:

        1. Schrijf de claim in één zin, en zoek de figuur of tabel van de paper die hem laat zien.
        2. Noem de twee settings, en alles wat in beide hetzelfde blijft.
        3. Kies de metric: één getal per run.
        4. Zeg welke kant de claim opgaat, en de kleinste mediaan van de winst die telt, uit de ruis tussen seeds en het effect in de paper.
        5. Plan 3 seeds of meer, en reken de kosten uit: seeds keer de uren van één seed, keer een marge van 2 of 3.
        6. Schrijf het op vóór de eerste run, en verander het daarna niet meer.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De regel met de hand, voor de bovenste rij van het plaatje: drie seeds van een testfout in %, met dropout en zonder, en een plan dat zegt: minstens 0,5 lager. Voer de cel uit.
    """)
    return


@app.cell
def example_rule():
    _with = [10.6, 10.3, 9.9]       # testfout met dropout, seed 0, 1 en 2
    _without = [11.5, 11.5, 11.4]   # dezelfde seeds, zonder dropout
    _gains = []
    for _i in range(3):
        _gains.append(_without[_i] - _with[_i])   # lager is de claim: hoeveel lager met dropout
    print("winsten:", [round(_g, 2) for _g in _gains], "mediaan:", round(np.median(_gains), 2), "telt:", np.median(_gains) >= 0.5)
    print("kosten:", round(3 * 0.3 * 2, 2), "uur")   # 3 seeds van 0,3 uur, met een marge van 2
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De winsten zijn 0,9, 1,2 en 1,5 procentpunt, de mediaan is 1,2, en dat is minstens 0,5: gesteund. Met dropout is de fout lager, dus de winst is *zonder min met*; bij een claim van hoger is het andersom. Het plan kost 1,8 uur, en dat past in een avond. `round` maakt de winsten alleen makkelijk leesbaar: `11.5 - 10.6` is in floating point 0.9000000000000004.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De regel van je plan

        Maak `verdict(plan, results)` af. `plan["settings"]` bevat de twee settings, die van de claim eerst; `plan["direction"]` is `"higher"` of `"lower"`; `plan["at_least"]` is de kleinste mediaan van de winst die telt; `plan["seeds"]` is het aantal seeds. `results[name]` is de metric van elke run van de setting `name`, seed 0 eerst. Geef `"too few runs"` terug als een setting minder runs heeft dan de seeds van het plan, anders `"supported"` als de mediaan van de winst over de seeds van het plan minstens `plan["at_least"]` is, en `"not supported"` als dat niet zo is. De check probeert beide richtingen, een seed te weinig, meer runs dan gepland, een mediaan van precies `at_least`, een negatieve `at_least` en vier seeds.
        """),
        L.task("Maak `verdict` af: tel de runs, maak de winst van elke seed van het plan, en vergelijk de mediaan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=16)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Eerst het tellen: `len(results[claim]) < n or len(results[other]) < n` betekent dat er een seed ontbreekt.",
            "Dan de loop van het uitgewerkte voorbeeld, over `range(n)`: alleen de seeds van het plan, hoeveel er ook meer zijn. Welke van de twee je van de andere aftrekt, hangt af van `plan[\"direction\"]`.",
            "Als laatste `np.median(gains) >= plan[\"at_least\"]`.",
        ],
        why="De winst per seed vergelijkt runs die hun geluk delen, de mediaan voorkomt dat één slechte seed beslist, en `range(n)` houdt extra runs buiten de deur.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.md(r"""
    ### Oefening 2. Je plan

    Vul het plan van je eigen reproductie in, voor de paper die je koos in [Je reproductie 1](/course/open/repro-own-choose?lang=nl). De check hieronder leest het zoals een reviewer dat zou doen, en stuurt het eerste terug dat ontbreekt of niet klopt. Is er niets meer, dan staat je plan in `work/projects/own/plan.json`, voor de lessen die het bouwen en draaien. Je kunt het later nog veranderen, vóór je eerste run.
    """)
    return


@app.cell(hide_code=True)
def plan_inputs():
    _choice = C.saved_choice()
    _old = C.saved_plan() or {}
    _old = _old if _choice and _old.get("paper") == _choice["id"] else {}
    _s = list(_old.get("settings") or ["", ""])
    _ways = {"hoger dan met de andere": "higher", "lager dan met de andere": "lower"}
    _levels = {"elke computer": "any", "NVIDIA, 8 GB": "gpu8", "aanbevolen specs": "rec"}
    plan_claim = mo.ui.text_area(label="De claim die je test, in één zin", full_width=True, rows=4, value=_old.get("claim") or C.claim_of(_choice, L.lang))
    plan_shown = mo.ui.text(label="Waar de paper hem laat zien: de figuur of tabel", full_width=True, value=_old.get("shown_in", ""))
    plan_a = mo.ui.text(label="De setting van de claim, zoals met dropout", full_width=True, value=_s[0])
    plan_b = mo.ui.text(label="De setting waarmee je hem vergelijkt, zoals zonder dropout", full_width=True, value=_s[1])
    plan_setup = mo.ui.text_area(label="Wat in beide hetzelfde blijft: data, netwerk, stappen", full_width=True, rows=2, value=_old.get("setup", ""))
    plan_metric = mo.ui.text(label="Wat je meet: één getal per run, zoals de testfout aan het eind, in %", full_width=True, value=_old.get("metric", ""))
    plan_seeds = mo.ui.number(start=1, stop=20, step=1, value=_old.get("seeds") or 3, label="Seeds")
    plan_direction = mo.ui.dropdown(options=_ways, value=next((_k for _k, _v in _ways.items() if _v == _old.get("direction")), None), label="Met de setting van de claim is de metric")
    plan_at_least = mo.ui.number(start=-1000, stop=1000, step=0.01, value=_old.get("at_least") or 0, label="met minstens, in de mediaan over de seeds (negatief: hooguit zoveel de andere kant op)")
    plan_per_seed = mo.ui.number(start=0, stop=100, step=0.1, value=_old.get("hours_per_seed") or 0, label="Uren voor één seed, beide settings")
    plan_margin = mo.ui.number(start=1, stop=5, step=0.5, value=_old.get("margin") or 2, label="Marge")
    plan_computer = mo.ui.dropdown(options=_levels, value=next(_k for _k, _v in _levels.items() if _v == (_old.get("computer") or C.machine())), label="Je computer")
    plan_hours = mo.ui.number(start=0, stop=200, step=0.5, value=_old.get("hours") or 4, label="Uren die je ervoor hebt")
    _head = mo.md(f"*Je paper: {C.plain(_choice['title'])}.*") if _choice else mo.md("*Nog geen paper gekozen: kies er eerst een in [Je reproductie 1](/course/open/repro-own-choose?lang=nl).*")
    mo.vstack([_head,
               mo.md("**De claim**"), plan_claim, plan_shown,
               mo.md("**Het experiment**"), plan_a, plan_b, plan_setup, plan_metric, plan_seeds,
               mo.md("**Wat telt**"), plan_direction, plan_at_least,
               mo.md("**De kosten**"), plan_per_seed, plan_margin, plan_computer, plan_hours])
    return plan_a, plan_at_least, plan_b, plan_claim, plan_computer, plan_direction, plan_hours, plan_margin, plan_metric, plan_per_seed, plan_seeds, plan_setup, plan_shown


@app.cell(hide_code=True)
def plan_value(plan_a, plan_at_least, plan_b, plan_claim, plan_computer, plan_direction, plan_hours, plan_margin, plan_metric, plan_per_seed, plan_seeds, plan_setup, plan_shown):
    _choice = C.saved_choice() or {}
    plan = {"paper": _choice.get("id"), "needs": _choice.get("needs"), "claim": plan_claim.value, "shown_in": plan_shown.value,
            "metric": plan_metric.value, "settings": [plan_a.value, plan_b.value], "setup": plan_setup.value,
            "direction": plan_direction.value, "at_least": plan_at_least.value, "seeds": plan_seeds.value,
            "hours_per_seed": plan_per_seed.value, "margin": plan_margin.value, "computer": plan_computer.value, "hours": plan_hours.value}
    return (plan,)


@app.cell(hide_code=True)
def check_plan(plan):
    result_plan = L.check("plan", plan)
    result_plan
    return (result_plan,)


@app.cell(hide_code=True)
def hints_plan():
    L.hints(
        "plan",
        [
            "Begin bij de kaart van je paper in [Je reproductie 1](/course/open/repro-own-choose?lang=nl): de claim noemt de twee settings, en *Je meet* de metric.",
            "Voor *minstens*: hoeveel verschillen twee seeds van dezelfde setting in je metric? Ga daarboven zitten, en niet hoger dan het kleinste effect dat je nog de claim van de paper zou noemen.",
            "Voor de uren: meet de tijd van één korte run, voor een paper van de lijst in de les waarop hij bouwt, vermenigvuldig met de stappen die je plant, en tel beide settings op.",
        ],
        why=C.course_plan_text(C.saved_choice(), L.lang),
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je plan zegt

    Je regel op drie verzonnen uitkomsten van je eigen plan: een waarin de winsten ruim voorbij wat telt liggen, een waarin ze er evenveel onder blijven, en de eerste, een seed te vroeg gestopt.
    """)
    return


@app.cell(hide_code=True)
def demo_plan(plan, result_verdict, result_plan):
    result_verdict, result_plan  # opnieuw uitvoeren als oefening 1 je regel bewaart
    _shown = C.demo(plan)
    if _shown is None:
        _out = mo.md("*Zodra een reviewer niets meer zou terugsturen, laat je regel hier zien wat hij zegt.*")
    else:
        _whose = {"yours": "het oordeel van je `own/plan.py`", "course": "het oordeel van de course, tot oefening 1 het jouwe bewaart"}[_shown[1]]
        _out = mo.vstack([mo.Html(C.outcomes_table(plan, _shown[0], {"holds": "ruim voorbij wat telt", "fails": "evenveel eronder",
                                                                     "early": "de eerste, een seed te vroeg", "outcome": "verzonnen uitkomst",
                                                                     "gains": "winst per seed", "verdict": "oordeel"}, L.lang)),
                          mo.md(f"*Met {_whose}.*")])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Waarom dat getal

    Het *minstens* van je plan is het getal waardoor je reproductie kan falen. Zeg waar het vandaan komt.
    """)
    return


@app.cell(hide_code=True)
def explain_why_number():
    explain_why_number_cell = L.explain(
        "why-number",
        question="Waarom koos je het getal achter *minstens* in je plan? En wat zou het doen als je het pas na je runs koos?",
        model_answer="**Het getal:** lager met minstens 0,5 procentpunt testfout. Als twee seeds van hetzelfde netwerk 0,3 tot 0,5 procentpunt verschillen, laat een kleiner getal ruis de claim steunen; de paper noemt een duidelijk lagere testfout zijn resultaat, en een half procentpunt op een fout van ongeveer 13% is het minste dat ik nog zo zou noemen. **Achteraf:** met mijn runs voor me kon ik een getal kiezen net onder de mediaan die ik kreeg, en dan steunde elke uitkomst de claim. Nu ik het vooraf opschrijf, kan mijn regel falen, en daardoor betekent het iets als hij dat niet doet.",
        points=[
            "je noemt je getal, in de eenheden van je metric",
            "je legt het naast hoeveel twee seeds verschillen",
            "je legt het naast wat de paper zijn effect noemt",
            "je zegt wat het zou doen als je het na de runs koos",
        ],
    )
    explain_why_number_cell
    return (explain_why_number_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het plan van [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl), nu met getallen die je zelf kiest; de winst per seed van [Begeleide reproductie 6](/course/open/repro-guided-compare?lang=nl), die runs met dezelfde seed vergeleek; en de ruis tussen seeds van [Eén run is geen resultaat](/course/open/res-seeds-variance?lang=nl). In de volgende les, *Je reproductie 3: bouw het*, bouw je de methode en de data van je paper, voor de twee settings van je plan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_verdict, result_plan, explain_why_number_cell):
    L.wrap_up(result_verdict, result_plan, explain_why_number_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
