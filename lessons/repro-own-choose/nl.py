# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson repro-own-choose. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-choose", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Begeleide reproductie 7: schrijf het verslag](/course/open/repro-guided-report?lang=nl) en [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_section():
    quick_section = L.choice(
        "quick-section",
        question="Welke sectie van je reproductierapport zegt wat je niet kon toetsen, en wat daarvoor nodig is?",
        options={
            "differences": "Differences",
            "right": "Not checked",
            "results": "Results",
        },
        correct="right",
        why={
            "differences": "*Differences* zegt waar je opzet en je getallen afwijken van die van de paper, en welke details hij niet geeft. Zie [Begeleide reproductie 7](/course/open/repro-guided-report?lang=nl).",
            "results": "*Results* geeft elke run die je plande, met zijn getallen. Zie [Begeleide reproductie 7](/course/open/repro-guided-report?lang=nl).",
            "right": "En hij eindigt met het oordeel dat je resultaten steunen.",
        },
    )
    quick_section
    return (quick_section,)


@app.cell(hide_code=True)
def quick_check_sums():
    quick_sums = L.choice(
        "quick-sums",
        question="Iemand controleert de map die je deelde met `sha256sum -c SHA256SUMS`, en één regel zegt FAILED. Wat zegt dat?",
        options={
            "broken": "Dat je code een bug heeft in dat bestand",
            "right": "Dat zijn kopie van dat bestand niet het bestand is dat je deelde",
            "order": "Dat de regels van SHA256SUMS in een andere volgorde staan",
        },
        correct="right",
        why={
            "broken": "Een checksum zegt niets over of code klopt, alleen of een bestand hetzelfde is. Zie [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl).",
            "order": "Elke regel wordt op zichzelf gecontroleerd, tegen het bestand dat hij noemt. Zie [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl).",
            "right": "Eén veranderd teken in een bestand geeft een heel andere SHA-256, dus weet hij welk bestand hij opnieuw moet halen.",
        },
    )
    quick_sums
    return (quick_sums,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Vanaf hier is de reproductie van jou: jij kiest de paper, en de course helpt minder. Of het lukt, wordt grotendeels nu beslist. Een paper die meer nodig heeft dan je computer heeft, een claim die geen figuur kan weerleggen, of data die je niet mag gebruiken, loopt vast vóór de eerste run, hoe goed je hem ook bouwt. [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl) gaf je de checks; deze les past ze toe op een lijst die de course gemaakt heeft, en op je eigen computer.

        **Een voorbeeld uit het dagelijks leven.** Kiezen wat je bakt voor gasten: een taart die een oven nodig heeft die je niet hebt, of een dag die je niet hebt, of iets wat je niet kunt kopen, is geen keuze, hoe mooi hij er in het kookboek ook uitziet.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw hem", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           0, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_cpu():
    guess_cpu = L.guess(
        "cpu",
        question=f"Hoeveel van de {len(C.papers())} papers op de lijst van de course passen volgens de course op een computer zonder NVIDIA-kaart?",
        options={
            "none": "Geen: elke paper op de lijst heeft een GPU nodig",
            "right": f"{sum(_p['needs'] == 'any' for _p in C.papers())}",
            "all": "Allemaal, als je maar lang genoeg wacht",
        },
        reveal=f"**{sum(_p['needs'] == 'any' for _p in C.papers())}:** " + " en ".join(f"*{_p['title']}*" for _p in C.papers() if _p['needs'] == 'any') + ", allebei nagebouwd met een MLP op Fashion-MNIST, afbeeldingen van kleding. De andere zes, waaronder twee CNN's op dezelfde afbeeldingen, zijn gemaakt voor een NVIDIA-kaart met 8 GB, of de aanbevolen specs. *Het idee* laat de hele lijst zien.",
    )
    guess_cpu
    return (guess_cpu,)


@app.cell(hide_code=True)
def text_idea():
    _here = {"any": "een computer zonder NVIDIA-kaart", "gpu8": "een NVIDIA-kaart met 8 GB of meer", "rec": "de aanbevolen specs, een NVIDIA-kaart met 12 GB of meer"}[C.machine()]
    mo.md(rf"""
    ## Het idee

    **De lijst.** Elke paper erop haalde de checks uit [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl): één claim die een figuur of tabel kan weerleggen, data met een licentie waaronder je ze mag gebruiken, en een kernresultaat waarvan de course schat dat het in een avond of een nacht past op de aanbevolen specs. Die schattingen zijn van de course, nog niet gemeten op de testmachine: plan met een marge. Elke paper bouwt voort op lessen die je gedaan hebt, en de kaart noemt ze.

    **Je computer.** Wat een paper nodig heeft, is een van drie niveaus: elke computer, een NVIDIA-kaart met 8 GB of meer, of de aanbevolen specs, een NVIDIA-kaart met 12 GB. Een computer van een hoger niveau draait wat een lager draait. De course ziet hier {_here}; een Mac telt als een computer zonder NVIDIA-kaart, omdat de schattingen voor zo'n kaart zijn. De uren zijn die op de aanbevolen specs: op een computer van een lager niveau kan een paper die erop draait langer duren, dus meet de tijd voordat je kiest, zoals *Let op* hieronder zegt.

    Hieronder staat de lijst, met de schatting van elke paper en voor elk niveau van computer of hij in één avond van 4 uur past. Schuif naar de marge waarmee je plant: de schatting keer de marge moet in de avond passen. De kolommen zijn de drie niveaus: elke computer, een NVIDIA-kaart met 8 GB, en 12 GB. De kolom van je computer is gemarkeerd; de papers die erop passen staan bovenaan, de andere zijn bleek.
    """)
    return


@app.cell(hide_code=True)
def explore_margin_cell():
    _mine = C.machine()
    _yours = {"any": "een zonder NVIDIA-kaart", "gpu8": "een met een NVIDIA-kaart van 8 GB", "rec": "een met de aanbevolen specs"}[_mine]
    _labels = {"paper": "paper", "hours": "uren", "yes": "ja", "more": "meer nodig", "long": "te lang", "yours": "(de jouwe)",
               "any": "elke", "gpu8": "8 GB", "rec": "12 GB"}

    def _frame(m):
        _m = int(m)
        _n = sum(C.course_fits(_p, _mine, C.EVENING / _m) for _p in C.papers())
        return (C.levels_picture(_m, _mine, _labels),
                f"Met een marge van {_m} past in een avond van {C.EVENING} uur een schatting van hooguit {C.number(C.EVENING / _m, 1, L.lang)} uur. Op je computer, {_yours}: **{_n} van de {len(C.papers())} papers {'past' if _n == 1 else 'passen'}**.")

    explore_margin = L.explore("margin", "marge", _frame, start=2, low=1, high=3, step=1, role="",
                               task="Hoeveel papers passen op je computer in één avond met een marge van 2, en met 3? En wat zou een computer van een hoger niveau toevoegen?")
    explore_margin
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(rf"""
        Zonder marge passen in één avond op elke computer de twee papers met een MLP op afbeeldingen van kleding; een NVIDIA-kaart met 8 GB voegt de kleine transformer van de induction heads en de twee CNN's toe, en de aanbevolen specs de andere drie, over language models: de mirage van emergent abilities, de reversal curse en LoRA. Met een marge van 3 past alleen {" and ".join(_p["title"].split(":")[0] for _p in C.papers() if C.course_fits(_p, "rec", C.EVENING / 3))} nog in een avond, op elke computer. Een paper die niet in één avond past, past misschien in een nacht, of een weekend: dat bepaal jij.

        **Een eigen paper** is welkom, zoals in de ML Reproducibility Challenge, als hij dezelfde checks haalt: schrijf zijn kaart zelf, met zijn claim, wat je zou meten, zijn data en licentie, wat hij nodig heeft, en je schatting van de uren.

        Dit zijn de kaarten van de lijst:
        """),
        mo.md("\n\n".join(f"**{_p['title']}** ({_p['year']}), {_p['authors']}, {_p['ref']}. *Claim:* {C.CARDS[_p['id']]['claim']['nl']} *Je meet* {C.CARDS[_p['id']]['measure']['nl']}. *Data:* {C.data_text(_p, L.lang)}. *Bouwt voort op:* {C.builds_on_text(_p, L.lang)}."
                          for _p in C.papers())),
        L.watch_out("Een schatting is geen meting. Kies je een paper die net past, meet dan eerst de tijd van één korte run ervan in de les waarop hij voortbouwt, en vermenigvuldig: stappen, seeds en instellingen, met een marge van 2 of 3."),
        mo.md(r"""
        ### Recept

        Zo kies je de paper van je reproductie:

        1. Zoek het niveau van je computer, en de uren die je hebt.
        2. Houd de papers over die niet meer dan je niveau nodig hebben en in je uren passen.
        3. Lees van elk de claim, en in de paper de figuur of tabel waar hij op rust.
        4. Controleer dat je zijn data mag gebruiken, en op welke lessen hij voortbouwt.
        5. Kies er een, en zeg waarom: de claim, wat je meet, en wat het kost.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 van het recept voor één paper met de hand: een paper die een NVIDIA-kaart met 8 GB en ongeveer 3 uur nodig heeft, op een computer met de aanbevolen specs, met één avond van 4 uur. Nieuw is de methode `.index` van een lijst: `_levels.index("gpu8")` geeft de index van `"gpu8"`, zijn plaats in de lijst geteld vanaf 0, hier 1. Voer de cel uit.
    """)
    return


@app.cell
def example_fits():
    _levels = ["any", "gpu8", "rec"]   # van minst naar meest
    _paper = {"needs": "gpu8", "hours": 3}
    _machine, _hours = "rec", 4
    _enough_machine = _levels.index(_paper["needs"]) <= _levels.index(_machine)   # 2. niet meer dan je niveau
    _enough_time = _paper["hours"] <= _hours                                      # 2. binnen je uren
    print("machine:", _enough_machine, "tijd:", _enough_time, "past:", _enough_machine and _enough_time)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `"gpu8"` staat op index 1 van de niveaus en `"rec"` op index 2, dus de machine is genoeg; 3 uur past in 4. Vergelijk de indexen, niet de namen: `"gpu8" <= "rec"` vergelijkt hun spelling, letter voor letter, en geeft hier alleen het goede antwoord omdat a, g en r toevallig in die volgorde in het alfabet staan. De index zegt welke meer is, hoe de namen ook luiden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Past hij?

        Maak `fits(paper, machine, hours)` af: `True` als de paper niet meer nodig heeft dan `machine`, en zijn schatting `paper["hours"]` niet meer is dan `hours`; anders `False`. `paper["needs"]` en `machine` zijn namen uit `LEVELS`, van minst naar meest. De check probeert papers die meer, evenveel en minder nodig hebben dan de machine, met te weinig uren en precies genoeg.
        """),
        L.task("Maak `fits` af: vergelijk de plaatsen in `LEVELS`, en de schatting met je uren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fits():
    box_fits = L.code_box("fits", lines=8)
    box_fits
    return (box_fits,)


@app.cell(hide_code=True)
def check_fits(box_fits):
    result_fits = L.check("fits", box_fits.value)
    result_fits
    return (result_fits,)


@app.cell(hide_code=True)
def hints_fits():
    L.hints(
        "fits",
        [
            "Zet de twee regels van het uitgewerkte voorbeeld in één `return`: `LEVELS.index(...)` voor beide niveaus, en `<=` voor beide vergelijkingen, verbonden met `and`.",
        ],
        why="De plaats in `LEVELS` zegt hoeveel een niveau heeft; de naam zelf niet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.md(r"""
    ### Oefening 2. Je paper

    Kies hieronder je paper, van de lijst of een eigen, en bewaar hem: hij gaat in je project als `work/projects/own/paper.json`, en de lessen van je reproductie beginnen ermee. Je kunt hem later veranderen door opnieuw te bewaren. Zeg daarna waarom je hem koos.
    """)
    return


@app.cell(hide_code=True)
def choose_inputs():
    _options = {f"{_p['title']} ({_p['year']})": _p["id"] for _p in C.papers()}
    _options["Een eigen paper"] = "own"
    _saved = C.saved_choice()
    _start = next((_k for _k, _v in _options.items() if _saved and _v == _saved["id"]), None)
    paper_dropdown = mo.ui.dropdown(options=_options, value=_start, label="Je paper", full_width=True)
    _own = _saved if _saved and _saved["id"] == "own" else {}
    own_title = mo.ui.text(label="Titel", full_width=True, value=_own.get("title", ""))
    own_ref = mo.ui.text(label="Verwijzing, zoals arXiv:2201.02177", full_width=True, value=_own.get("ref", ""))
    own_claim = mo.ui.text_area(label="Zijn claim, in één of twee zinnen, in het Engels", full_width=True, rows=3, value=_own.get("claim", ""))
    own_measure = mo.ui.text_area(label="Wat je zou meten, en de figuur of tabel van de paper die je nadoet, in het Engels", full_width=True, rows=2, value=_own.get("measure", ""))
    own_data = mo.ui.text(label="Zijn data, en de licentie waaronder je ze mag gebruiken", full_width=True, value=_own.get("data", ""))
    _needs = {"elke computer": "any", "NVIDIA, 8 GB": "gpu8", "aanbevolen specs": "rec"}
    own_needs = mo.ui.dropdown(options=_needs, value=next((_k for _k, _v in _needs.items() if _v == _own.get("needs")), "aanbevolen specs"), label="Wat hij nodig heeft")
    own_hours = mo.ui.number(start=0, stop=200, step=0.5, value=_own.get("hours") or 4, label="Je schatting, in uren")
    save_choice_button = mo.ui.run_button(label="Bewaar mijn keuze")
    mo.vstack([paper_dropdown, mo.accordion({"Alleen voor een eigen paper": mo.vstack([own_title, own_ref, own_claim, own_measure, own_data, own_needs, own_hours])}), save_choice_button])
    return own_claim, own_data, own_hours, own_measure, own_needs, own_ref, own_title, paper_dropdown, save_choice_button


@app.cell(hide_code=True)
def demo_choice(paper_dropdown, own_title, own_ref, own_claim, own_measure, own_data, own_needs, own_hours, save_choice_button, box_fits, result_fits):
    _out = None
    if save_choice_button.value and paper_dropdown.value:
        _own = {"title": own_title.value, "ref": own_ref.value, "claim": own_claim.value, "measure": own_measure.value,
                "data": own_data.value, "needs": own_needs.value, "hours": own_hours.value}
        if paper_dropdown.value == "own" and not (own_title.value.strip() and own_claim.value.strip() and (own_hours.value or 0) > 0):
            _out = mo.md("*Schrijf voor een eigen paper minstens zijn titel, zijn claim en je schatting van de uren, en bewaar dan opnieuw.*")
        else:
            _c = C.save_choice(paper_dropdown.value, _own)
            _fn = C.fits_of(box_fits.value) if result_fits.state == "correct" else None
            _ok = (_fn or C.course_fits)(_c, C.machine(), 4)
            _say = "Hij past in één avond op je computer." if _ok else "Volgens zijn schatting past hij niet in één avond op je computer: plan een nacht of een weekend, of kies een kleinere." if C.level_fits(_c, C.machine()) else "Hij heeft meer nodig dan je computer heeft, en meer uren helpen niet: kies een paper die je computer kan draaien, of draai hem op een computer die heeft wat hij nodig heeft."
            _out = mo.md(f"*Bewaard als `work/projects/own/{C.CHOICE}`: {C.plain(_c['title'])}. {_say}*")
    _saved = C.saved_choice()
    _out = _out or (mo.md(f"*Je keuze tot nu toe: {C.plain(_saved['title'])}.*") if _saved else mo.md("*Nog geen keuze bewaard.*"))
    _out
    return


@app.cell(hide_code=True)
def explain_why_this():
    explain_why_this_cell = L.explain(
        "why-this",
        question="Waarom is de paper die je koos er een die je kunt reproduceren? Zeg wat je gaat toetsen, waarop, en wat het kost.",
        model_answer="**De claim:** een netwerk dat met dropout getraind is, heeft een duidelijk lagere testfout dan hetzelfde netwerk zonder; hij rust op de figuur van de paper die de testfout op MNIST van netwerken met en zonder dropout naast elkaar zet. **Wat ik meet:** de testfout van hetzelfde netwerk met en zonder dropout, over drie seeds, op Fashion-MNIST, dat de course onder de MIT-licentie host. **Wat het kost:** de course schat minder dan een uur op de aanbevolen specs, en hij draait op elke computer. De mijne heeft geen NVIDIA-kaart, dus het kan langer duren: ik meet eerst de tijd van één korte run. Ik heb een avond, dus zelfs bij drie keer de schatting past hij. **Wat ik hergebruik:** het netwerk uit *Project: een diep netwerk laten trainen*, en dropout uit zijn eigen les.",
        points=[
            "je noemt de claim, en de figuur of tabel van de paper waar hij op rust",
            "je zegt wat je gaat meten, op welke data, en dat je die mag gebruiken",
            "je vergelijkt de kosten met je computer en je uren, met een marge",
            "je noemt wat je uit de lessen of je kit kunt hergebruiken",
        ],
    )
    explain_why_this_cell
    return (explain_why_this_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De checks uit [Een paper kiezen die je kunt nabouwen](/course/open/res-choose-paper?lang=nl), nu op een lijst en op je eigen computer; en het plan uit [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl), dat hierna voor je eigen paper komt. In de volgende les, *Je reproductie 2: claim en plan*, schrijf je je claim op, het experiment, de kosten en wat als een mislukte reproductie telt, in de vorm van je kit, vóór elke run.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_fits, explain_why_this_cell):
    L.wrap_up(result_fits, explain_why_this_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
