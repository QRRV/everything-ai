# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-ablation. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-ablation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 5: het hoofdresultaat](/course/open/repro-own-figure?lang=nl) en [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="Je logboek heeft één regel: `{\"setting\": \"claim\", \"seed\": 0, \"steps\": 400, \"result\": 16.4}`. Je `run_plan` wordt aangeroepen met seeds `[0]` en budgetten `[400, 800]`. Welke runs draait hij?",
        options={
            "two": "Twee: beide settings bij 800 stappen, want seed 0 staat al in het logboek",
            "four": "Vier: een regel van eerder telt niet voor een nieuwe aanroep",
            "right": "Drie: `\"other\"` bij 400 stappen, en beide settings bij 800",
        },
        correct="right",
        why={
            "two": "De setting, de seed en de stappen wijzen samen een run aan: alleen `\"claim\"` van seed 0 bij 400 stappen is klaar. Zie [Je reproductie 5](/course/open/repro-own-figure?lang=nl).",
            "four": "`run_plan` leest eerst het logboek, en slaat elke run over waarvan de regel erin staat, welke aanroep hem ook schreef. Zie [Je reproductie 5](/course/open/repro-own-figure?lang=nl).",
            "right": "Alleen een run waarvan setting, seed en stappen samen op een regel van het logboek staan, wordt overgeslagen.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def quick_check_grid():
    quick_grid = L.choice(
        "quick-grid",
        question="Begeleide reproductie 5 draaide een ablation grid van weight decay en de fractie van de tabel. Een grid van weight decay 1, 0,1 of 0, maal de fracties 30%, 50%, 70% en 90%: hoeveel runs is dat, en waarom?",
        options={
            "right": "12: elke waarde van het ene, met elke waarde van het andere",
            "one": "6: de control run, en elke andere waarde één keer apart",
            "three": "4: één voor elke fractie, de control run daarbij",
        },
        correct="right",
        why={
            "one": "Dan verander je één ding tegelijk ten opzichte van de control run. Een grid draait elke combinatie, dus laat ook zien of weight decay bij de ene fractie meer uitmaakt dan bij de andere. Zie [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl).",
            "three": "Vier runs variëren alleen de fractie: ze kunnen niet laten zien wat weight decay doet. Zie [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl).",
            "right": "3 × 4 = 12: elke combinatie één keer, de control run daarbij.",
        },
    )
    quick_grid
    return (quick_grid,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je hoofdresultaat zegt dat de methode werkt in je runs. Het zegt niet welk deel van de methode het werk doet. Veel methodes zijn een paar veranderingen tegelijk: de dropout-paper zet neuronen uit bij de input, met een kans van 0,2, en in de hidden layers, met een kans van 0,5. Een ablation haalt één deel weg, houdt al het andere, en draait opnieuw bij het budget en over de seeds van je hoofdresultaat: wat er van de winst over is, is wat de methode zonder dat deel doet. Reviewers vragen erom, en de lezers van je verslag ook.

        **Een voorbeeld uit het dagelijks leven.** Je fiets is sneller na een nieuwe ketting en nieuwe banden. Rijd nog eens op de oude banden: wat er van de winst over is, is wat de ketting alleen deed.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           5, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_input():
    guess_input = L.guess(
        "no-input",
        question="In het hoofdresultaat van de course voor de dropout-paper verlaagt dropout de testfout bij 12.800 stappen met 0,7 procentpunt in de mediaan. Haal de dropout bij de input weg, en houd de 0,5 in de hidden layers. Wat denk je dat er van de winst over is?",
        options={
            "most": "Het meeste: de hidden layers, met hun grotere kans van 0,5, doen het werk",
            "half": "Ongeveer de helft: elk van de twee delen levert zijn eigen aandeel",
            "right": "Niets: het netwerk doet het zelfs iets slechter dan zonder dropout",
        },
        reveal="**Niets.** Zonder de dropout bij de input zijn de winsten ten opzichte van geen dropout −0,9, 0,4 en −0,5 voor seed 0, 1 en 2: een mediaan van −0,5. Dropout in de hidden layers alleen helpt dit kleine netwerk bij 12.800 stappen niet; met ook de dropout bij de input geven de twee 0,7. *Het idee* laat ook de andere ablation zien.",
    )
    guess_input
    return (guess_input,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Eén deel eruit.** De ablation is de setting van de claim met één deel van zijn methode weggehaald, en al het andere hetzelfde: een derde setting van je experiment, `"ablation"`. De runs van de claim zijn de control runs van de ablation; de andere setting is waarmee de claim vergeleken wordt. Haal je twee delen tegelijk weg, dan kan de ablation niet zeggen welk van de twee het werk deed. Is de methode één verandering, zoals een rate, dan kan een ablation er een deel van nemen: de halve rate.

    **Wat hij houdt.** Seed voor seed de winst van de ablation op de andere setting, tegen de winst van de claim: het aandeel dat hij houdt, is de mediaan van zijn winsten, gedeeld door de mediaan van die van de claim. Een aandeel van 1 houdt de hele winst, 0 niets ervan, en onder 0 doet de ablation het slechter dan de andere setting.

    **Eerst de regel.** Je resultaat hangt van het deel af als de ablation minder dan de helft van de winst houdt. Je legt het deel en de regel vast vóór de runs, net als de regel van je plan, en de ablation draait bij het budget en over de seeds van je hoofdresultaat, zodat zijn runs seed voor seed te vergelijken zijn met die van de claim.

    **Je experiment blijft zoals het is.** Je `own/ablation.py` importeert je experiment en voegt `"ablation"` toe aan zijn `SETTINGS` terwijl hij draait: het bestand van je experiment verandert niet, en je hoofdresultaat blijft wat het was.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | ablation-setting | de setting van de claim met één deel van zijn methode weggehaald |
    | aandeel dat hij houdt | de winst van de ablation op de andere setting in de mediaan, gedeeld door die van de claim |

    Het plaatje: de ablations die de course met zijn experiment voor de dropout-paper draaide bij 12.800 stappen, met de claim en de andere setting: de vier manieren om de dropout bij de input (0,2) en in de hidden layers (0,5) wel of niet te hebben. Elke stip is de testfout van één seed, de ring de mediaan van drie; de doorgetrokken lijn is de mediaan van de claim, de gestreepte die van de andere setting. Beweeg de schuif om elke rij te lezen.
    """)
    return


@app.cell(hide_code=True)
def explore_rows_cell():
    _names = ["allebei: de claim", "alleen hidden layers", "alleen input", "geen: de andere"]
    _labels = {"title": "Testfout bij 12.800 stappen, per dropout", "x": "testfout (%)",
               "alt": "De testfout van drie seeds bij 12.800 stappen voor vier settings. Beide dropouts, de claim: 12,5, 12,9 en 12,4. Alleen hidden layers: 13,8, 13,2 en 14,1. Alleen input: 13,5, 13,3 en 12,7. Geen van beide, de andere setting: 12,9, 13,6 en 13,6."}
    _rows = ["claim", "no-input", "no-hidden", "other"]

    def _frame(k):
        _k = int(k) - 1   # rij 1 tot 4, van boven
        _row = _rows[_k]
        _g = C.gains(C.course_runs("claim"), C.course_runs("other"))
        if _row == "claim":
            _say = f"Allebei, de claim: winsten van {', '.join(C.signed(g, L.lang) for g in _g)} op geen dropout, een mediaan van {C.signed(C.median(_g), L.lang)}: de winst waartegen de ablations gemeten worden."
        elif _row == "other":
            _say = "Geen van beide, de andere setting: de runs waar elke winst vanaf geteld wordt."
        else:
            _a = C.gains(C.course_runs(_row), C.course_runs("other"))
            _part = "de dropout bij de input" if _row == "no-input" else "de dropout in de hidden layers"
            _share = round(100 * C.course_kept(_row))
            _say = (f"Zonder {_part}: winsten van {', '.join(C.signed(g, L.lang) for g in _a)}, een mediaan van {C.signed(C.median(_a), L.lang)}. "
                    f"Hij houdt {C.signed(_share, L.lang)}% van de winst: minder dan de helft, dus volgens de regel hangt het resultaat af van {_part}.")
        return C.ablation_picture(_k, _names, _labels), _say

    explore_rows = L.explore("row", "rij", _frame, start=2, low=1, high=4, step=1, role="",
                             task="Van welk van de twee delen hangt de winst af? En tellen de aandelen van de twee ablations op tot de hele winst?")
    explore_rows
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Met elk deel alleen blijft minder dan de helft van de winst over: zonder de dropout bij de input niets ervan, −71%, omdat het netwerk het iets slechter doet dan zonder dropout; zonder de dropout in de hidden layers 43%. Volgens de regel hangt het resultaat van allebei af: wat dropout hier doet, doen de twee delen samen. En de aandelen tellen niet op: −71% en 43% zijn ver van de hele winst. Als delen samenwerken, zegt een ablation waarvoor een deel nodig is, niet hoeveel van de winst van hemzelf is. Met drie seeds is het eerste antwoord duidelijk, een mediaan van −0,5 tegen 0,7; het tweede ligt dicht bij de grens, 43% tegen 50%.
        """),
        L.watch_out("Een ablation van drie seeds ziet een grote verandering, geen kleine: een aandeel dicht bij de helft kan met één seed meer omslaan. Leg het deel en de regel vast vóór de runs, draai één ablation, en zeg in je verslag hoe dichtbij het was. Delen proberen tot er een het antwoord geeft waarop je hoopte, is je resultaat kiezen."),
        mo.md(r"""
        ### Recept

        Zo draai je de ablation van je reproductie:

        1. Noem de delen van de methode van je claim, en kies het deel dat je weghaalt, vóór elke run.
        2. Voeg de setting `"ablation"` toe aan je experiment: die van de claim, met dat ene deel eruit.
        3. Draai hem met de runner van je kit, over de seeds van je plan, bij je budget.
        4. Reken het aandeel uit dat hij houdt: zijn winsten op de andere setting, seed voor seed, tegen die van de claim.
        5. Lees het met de regel: onder de helft hangt je resultaat van dat deel af.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het aandeel dat de andere ablation houdt, zonder de dropout in de hidden layers, met de hand: de winst van elke seed op geen dropout, voor de claim en voor de ablation, en de mediaan van elk. Nieuw is `zip`: dat loopt tegelijk door een paar lijsten, per ronde één item van elk, zodat `c` en `o` altijd van dezelfde seed zijn. Voer de cel uit.
    """)
    return


@app.cell
def example_share():
    _claim = [12.5, 12.9, 12.4]      # beide dropouts: seed 0, 1, 2
    _other = [12.9, 13.6, 13.6]      # geen dropout
    _ablation = [13.5, 13.3, 12.7]   # alleen de dropout bij de input
    _g_claim = [round(o - c, 1) for c, o in zip(_claim, _other)]
    _g_ablation = [round(o - a, 1) for a, o in zip(_ablation, _other)]
    print("claim:", _g_claim, np.median(_g_claim))
    print("ablation:", _g_ablation, np.median(_g_ablation))
    print("houdt:", round(np.median(_g_ablation) / np.median(_g_claim), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De testfout hoort omlaag te gaan, dus een winst is de andere setting min de setting zelf. De claim wint 0,7 in de mediaan, de ablation 0,3, dus houdt hij 0,3 / 0,7 = 0,43 van de winst: minder dan de helft. Seed 0 van de ablation loopt zelfs achter op geen dropout, met 0,6, en seed 2 loopt 0,9 voor: de mediaan neemt de middelste winst, 0,3, dus beslist geen van beide uitersten het.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ablation():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. Je ablation

        Maak `own/ablation.py` af in het vak: de waarde van de setting `"ablation"`, die van de claim met één deel van zijn methode weggehaald, en in `PART` welk deel dat is. Het vak importeert je experiment uit je kit, en voegt de setting toe aan zijn `SETTINGS` terwijl het draait. De check vergelijkt de nieuwe setting met `"claim"` en `"other"`, en draait twee keer `run("ablation", 0, 20)`. Voor de dropout-paper is `"claim"` `(0.2, 0.5)`.
        """),
        L.task("Maak de ablation af: de setting van de claim zonder één deel, en de naam van dat deel."),
    ])
    return


@app.cell(hide_code=True)
def course_note():
    L.course_versions()
    return


@app.cell(hide_code=True)
def exercise_ablation():
    box_ablation = L.code_box("ablation", lines=9)
    box_ablation
    return (box_ablation,)


@app.cell(hide_code=True)
def check_ablation(box_ablation):
    result_ablation = L.check("ablation", box_ablation.value)
    result_ablation
    return (result_ablation,)


@app.cell(hide_code=True)
def hints_ablation():
    L.hints(
        "ablation",
        [
            "Kijk naar de waarde van `\"claim\"` in de `SETTINGS` van je experiment: welke delen ervan zijn de methode, en welk kan weg zonder de andere?",
            "Neem de waarde van `\"claim\"` over, en zet één deel ervan op wat `\"other\"` daar heeft: voor de dropout-paper een van zijn twee getallen op 0.0.",
        ],
        why="Eén deel eruit, de rest zoals de claim het heeft, zodat een verandering in het resultaat alleen van dat deel kan komen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kept():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat de ablation houdt

        Maak `kept(direction, claim, ablation, other)` af. Elke lijst heeft de metric van één setting, seed 0 eerst, bij je budget; `direction` is `"lower"` of `"higher"`, zoals de claim zegt dat de metric gaat. Hij geeft het aandeel van de winst van de claim terug dat de ablation houdt, zoals in het uitgewerkte voorbeeld, of `None` als de claim geen winst heeft om te houden: een winst van 0 of minder in de mediaan. De check probeert de ablation van de course, de andere, een accuracy die omhoog hoort te gaan, een claim zonder winst, en vier seeds.
        """),
        L.task("Maak `kept` af: de winsten van elke seed zoals `direction` zegt, dan `None` of het aandeel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kept():
    box_kept = L.code_box("kept", lines=18)
    box_kept
    return (box_kept,)


@app.cell(hide_code=True)
def check_kept(box_kept):
    result_kept = L.check("kept", box_kept.value)
    result_kept
    return (result_kept,)


@app.cell(hide_code=True)
def hints_kept():
    L.hints(
        "kept",
        [
            "Loop met `zip` tegelijk door de drie lijsten, één seed per keer, zoals het uitgewerkte voorbeeld doet; bij `\"higher\"` is een winst de setting min de andere.",
            "Na de loop: `None` als `np.median(claim_gains) <= 0`; anders de mediaan van de winsten van de ablation gedeeld door de mediaan van die van de claim.",
        ],
        why="Een winst van een seed vergelijkt twee runs die dezelfde startwaarden en batches zagen; de medianen zorgen dat één seed die wegloopt het niet beslist; en zonder winst is er niets te houden.",
    )
    return


@app.cell(hide_code=True)
def text_run():
    mo.md(rf"""
    ### Wat je project nu kan

    De knop draait je ablation met je kit: je `run_plan` met de `run` van je `own/ablation.py`, elke seed van je plan, bij je budget, in een eigen logboek, `work/projects/own/ablation.jsonl`. Daarna leest hij de runs van de claim en de andere setting bij dat budget uit je hoofdresultaat, en geeft alle drie aan je `kept`. Voor het experiment van de course voor de dropout-paper duren de drie runs ongeveer {C.ABLATION_MINUTES} minuten op één thread van de computer van de course. Net als bij je hoofdresultaat kost een stop de run die bezig was. Verander je je ablation na zijn runs, verwijder dan eerst `work/projects/own/ablation.jsonl`: de knop draait alleen wat nog geen regel heeft.
    """)
    return


@app.cell(hide_code=True)
def button_ablation():
    run_ablation = mo.ui.run_button(label="Draai mijn ablation")
    run_ablation
    return (run_ablation,)


@app.cell(hide_code=True)
def demo_ablation(run_ablation, result_ablation, result_kept):
    result_ablation, result_kept  # opnieuw uitvoeren als een oefening een deel van de kit bewaart
    ablation_lines = None
    if not run_ablation.value:
        _out = mo.md("*Druk op de knop om je ablation met je kit te draaien.*")
    else:
        def _progress(_line):
            _n = len(C.my_ablation_log() or [])
            mo.output.replace(mo.md(f"**{_n} {'run' if _n == 1 else 'runs'} van de ablation in zijn logboek** · de nieuwste: seed {_line['seed']}, {C.whole(_line['steps'], L.lang)} stappen: {C.short(_line['result'], L.lang)}"))

        _r = C.demo(_progress)
        if _r is None:
            _out = mo.md("*Je kit heeft eerst vier delen nodig, en een ervan is er nog niet: je ablation, oefening 1 hier; je runner, oefening 1 van [Je reproductie 5](/course/open/repro-own-figure?lang=nl); je experiment, oefening 2 van [Je reproductie 3](/course/open/repro-own-build?lang=nl); en je plan, van [Je reproductie 2](/course/open/repro-own-claim?lang=nl).*")
        elif _r == "budget":
            _out = mo.md("*Leg eerst je budget vast in oefening 3 van [Je reproductie 5](/course/open/repro-own-figure?lang=nl): de ablation draait daarbij.*")
        elif _r == "log":
            _out = mo.md("*Een regel van het logboek van je ablation, `work/projects/own/ablation.jsonl`, is geen run van de ablation zoals `run_plan` hem schrijft: `\"setting\"` `\"ablation\"`, een hele `\"seed\"` en `\"steps\"`, en een getal als `\"result\"`. Heb je je `own/runner.py` veranderd na zijn check, of het logboek met de hand?*")
        elif _r == "error":
            _out = mo.md("*Je `own/runner.py`, `own/ablation.py` of `own/experiment.py` stopt met een fout. Heb je er een veranderd na zijn check? Check hem dan opnieuw. De runs die klaar waren, staan in het logboek.*")
        else:
            ablation_lines = _r
            _out = mo.md(f"*Klaar: {len(_r)} runs van de ablation staan in zijn logboek, `work/projects/own/ablation.jsonl`.*")
    _out
    return (ablation_lines,)


@app.cell(hide_code=True)
def demo_reading(ablation_lines, result_kept):
    ablation_lines, result_kept  # de logboeken opnieuw lezen na een run
    _plan, _budget = C.P.saved_plan(), C.F.saved_budget()
    _r = C.my_reading(_plan, _budget["steps"]) if _plan and _budget else None
    if _r is None:
        _out = mo.md("*Je ablation verschijnt hier zodra zijn runs, en die van je hoofdresultaat bij je budget, in je logboeken staan.*")
    else:
        _a, _b = C.P._settings(_plan)
        _rows = "\n".join(f"| {_i} | {C.short(_c, L.lang)} | {C.short(_x, L.lang)} | {C.short(_o, L.lang)} |"
                          for _i, (_c, _x, _o) in enumerate(zip(_r["claim"], _r["ablation"], _r["other"])))
        if _r["kept"] == "error":
            _say = "Je `own/kept.py` stopt met een fout, of zit nog niet in je kit: check oefening 2."
        elif _r["kept"] is None:
            _say = "Je `kept` geeft `None`: de claim heeft geen winst bij je budget, dus is er niets voor de ablation om te houden."
        else:
            _share = round(100 * _r["kept"])
            _say = f"Je ablation houdt **{C.signed(_share, L.lang)}%** van de winst: " + ("minder dan de helft, dus volgens de regel hangt je resultaat af van het deel dat hij weghaalt." if C.depends(_r["kept"]) else "minstens de helft, dus volgens de regel hangt je resultaat niet af van het deel dat hij weghaalt.")
        _out = mo.md(f"""
| seed | {_a} | ablation | {_b} |
|---|---|---|---|
{_rows}

{_say}
""")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Wat je ablation zegt

    Je ablation staat in zijn logboek. Zeg waarvan je resultaat afhangt, volgens je regel, en wat je zekerder zou maken.
    """)
    return


@app.cell(hide_code=True)
def explain_why_ablation():
    explain_why_ablation_cell = L.explain(
        "why-ablation",
        question="Van welk deel van de methode hangt je resultaat af, volgens je regel? En wat zou je er zekerder van maken?",
        model_answer="**Volgens mijn regel:** zonder de dropout bij de input houdt mijn ablation −71% van de winst, ver onder de helft: mijn resultaat hangt af van de dropout bij de input. Zonder die dropout is het netwerk zelfs iets slechter dan zonder dropout, dus helpt de dropout in de hidden layers hier niet in zijn eentje. **Om zekerder te zijn:** meer seeds, want mijn seeds verschillen tot 0,9 procentpunt en drie zien geen kleine verandering; en de andere ablation, zonder de dropout in de hidden layers, die de course op 43% mat, dicht bij de grens. Ik zou beide in mijn verslag zetten, met de regel die ik vóór de runs vastlegde.",
        points=[
            "je geeft het aandeel dat je ablation houdt, en wat de regel zegt",
            "je zegt wat dat betekent voor het deel dat je weghaalde",
            "je noemt iets wat je zekerder zou maken, zoals meer seeds of een andere ablation",
        ],
    )
    explain_why_ablation_cell
    return (explain_why_ablation_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De ablation van [Ablation: wat gebeurt er als je een deel weghaalt?](/course/open/res-ablation?lang=nl) en het grid van [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl), nu voor de methode van je eigen paper; en de runner en het budget van [Je reproductie 5](/course/open/repro-own-figure?lang=nl). In de volgende les, *Je reproductie 7: vergelijk met de paper*, test je of het verschil tussen jouw getallen en die van de paper echt is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ablation, result_kept, explain_why_ablation_cell):
    L.wrap_up(result_ablation, result_kept, explain_why_ablation_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
