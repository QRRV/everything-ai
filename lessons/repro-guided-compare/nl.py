# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-guided-compare. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-guided-compare", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Is het verschil echt?](/course/open/res-significance?lang=nl) en [Begeleide reproductie 5: de ablation](/course/open/repro-guided-ablation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_interval():
    quick_interval = L.choice(
        "quick-interval",
        question="Vijf seeds, gepaard: het 95%-interval van het gemiddelde verschil loopt van −0,05 tot 0,40, met het gemiddelde op 0,18. Wat zegt je verslag?",
        options={
            "zero": "Dat er geen verschil is tussen de twee instellingen",
            "right": "Geen duidelijk verschil: het interval bevat 0, dus het kan ruis zijn",
            "mean": "Een verschil van 0,18, want dat is het gemiddelde van de vijf paren",
        },
        correct="right",
        why={
            "zero": "Een interval dat 0 bevat, laat niet zien dat er geen verschil is: het bevat ook 0,3. Het zegt dat vijf seeds het niet kunnen onderscheiden. Zie [Is het verschil echt?](/course/open/res-significance?lang=nl).",
            "mean": "Het gemiddelde alleen verbergt hoe onzeker het is: met deze vijf paren kan het echte verschil net zo goed 0 zijn. Zie [Is het verschil echt?](/course/open/res-significance?lang=nl).",
            "right": "Een interval dat 0 bevat, betekent dat de data een verschil niet van ruis kan onderscheiden; meer seeds maken het interval smaller.",
        },
    )
    quick_interval
    return (quick_interval,)


@app.cell(hide_code=True)
def quick_check_one_seed():
    _times = C.A.read(C.A.run(1.0, 0.5))["times"]   # met weight decay bij 50%: hoeveel keer zo laat als training
    quick_one_seed = L.choice(
        "quick-one-seed",
        question=f"In je ablation haalde validatie met weight decay bij 50% de 99% {C.number(_times, 1, L.lang)} keer zo laat als training, op één seed; je plan noemt {C.DELAY} keer of meer \"grokked\". Wat mag je verslag over 50% zeggen?",
        options={
            "grok": f"Dat het bij 50% grokked is: {C.number(_times, 1, L.lang)} is zo goed als {C.DELAY}",
            "none": "Dat weight decay vanaf 50% de vertraging weghaalt",
            "right": f"Dat één seed net onder {C.DELAY} keer zo laat bleef, en een andere aan beide kanten kan uitkomen",
        },
        correct="right",
        why={
            "grok": f"Je plan legde de grens op {C.DELAY} vóór elke run, zodat geen uitkomst hem achteraf kon verschuiven. {C.number(_times, 1, L.lang)} ligt eronder, dus is het oordeel \"no delay\". Zie [Begeleide reproductie 1](/course/open/repro-guided-claim?lang=nl).",
            "none": f"Eén seed per combinatie kan een nipte uitkomst niet van een regel onderscheiden: bij 50% kwam het op {C.number(_times, 1, L.lang)}, dicht bij de grens. Zie [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl).",
            "right": "Met één seed per combinatie is een nipte uitkomst alleen dat: pas meer seeds laten zien aan welke kant hij meestal uitkomt.",
        },
    )
    quick_one_seed
    return (quick_one_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je ablation vond dat weight decay de sprong eerder brengt: met één seed per combinatie. De paper zegt dat weight decay de generalisatie "het meest verbetert": op een andere taak, het product in de groep $S_5$, gemeten als hoeveel data het kost. Voordat je verslag kan zeggen dat je reproductie met de paper overeenkomt, twee vragen. Is je verschil echt, of één worp met de dobbelsteen? Draai het over seeds, en vergelijk paren. En waar jouw getal en dat van de paper verschillen, waarom? Leg de opzetten naast elkaar, en zoek wat de paper niet zegt.

        **Een voorbeeld uit het dagelijks leven.** Jij en een vriend bakken volgens hetzelfde recept, en jouw brood is platter. Voordat je zegt dat het recept niet werkt, bak je er nog een paar, op verschillende dagen: is het elke keer, of was het die ene dag? En dan vergelijk je wat je deed: dezelfde bloem, dezelfde oven? Het recept zei nooit hoe lang je moet kneden.
        """),
        draw.project_steps(["Lees de paper en kies de claim", "Bouw het model en de data", "Sanity checks", "Reproduceer de belangrijkste figuur",
                            "De ablation", "Vergelijk met de paper", "Het verslag"],
                           5, here_label="← vandaag", title="Waar je project staat: je reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_seeds():
    _f = C.factors(len(C.SEEDS))
    _got, _none = [_x for _x in _f if _x is not None], [_s for _s, _x in enumerate(_f) if _x is None]
    guess_seeds = L.guess(
        "five-seeds",
        question=f"In seed 0 bij {round(100 * C.WIDE)}% van de tabel haalde validatie zonder weight decay 99% {C.number(C.factors(1)[0], 1, L.lang)} keer zo laat als met. Wat denk je dat er over vijf seeds gebeurt?",
        options={
            "half": "Zonder weight decay later in ongeveer de helft van de seeds, eerder in de andere",
            "all": "Zonder weight decay later in elke seed",
            "most": "Zonder weight decay eerder in de meeste seeds: seed 0 was de uitzondering",
        },
        reveal=f"**Later in elke seed:** in {len(_got)} seeds {C.number(min(_got), 1, L.lang)} tot {C.number(max(_got), 1, L.lang)} keer zo laat, en in seed {', '.join(str(_s) for _s in _none)} haalde validatie zonder weight decay binnen de 12.000 stappen geen 99%. Hoe zeker het gemiddelde daardoor is, laat *Het idee* zien.",
    )
    guess_seeds
    return (guess_seeds,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(rf"""
    ## Het idee

    **Koppel op seed.** In je kit legt de seed de splitsing, de beginwaarden van de weights en de batches vast. Twee runs met dezelfde seed en een andere weight decay verschillen alleen in die ene instelling, dus vergelijk ze als paar: wat de seed met allebei doet, valt binnen het paar weg.

    **Vergelijk factoren, als logaritmen.** Het moment $v$ waarop validatie 99% haalt, kan factoren 10 verschillen, dus vergelijk hoeveel keer zo laat, niet hoeveel stappen meer. Voor elk paar $\log_{{10}} v_a - \log_{{10}} v_b$: 0 als ze even snel zijn, 1 als $a$ 10 keer zo laat is. De gepaarde bootstrap uit [Is het verschil echt?](/course/open/res-significance?lang=nl) geeft het gemiddelde van die verschillen en een 95%-interval; $10$ tot de macht het gemiddelde zegt hoeveel keer zo laat het doorgaans is.

    Hieronder staan de runs van de course bij {round(100 * C.WIDE)}% van de tabel, met en zonder weight decay, voor de seeds 0 tot en met 4: een stip voor elke seed, hoeveel keer zo laat validatie zonder weight decay 99% haalt, of een pijl omhoog vanaf het minste dat het kan zijn als hij er binnen 12.000 stappen niet komt. De gestreepte lijn is het gemiddelde, de grijze band zijn interval, en de dunne lijn 1, waar ze even snel zijn. Schuif om te kiezen hoeveel seeds de vergelijking gebruikt.
    """)
    return


@app.cell(hide_code=True)
def explore_seeds_cell():
    def _frame(n):
        _n = int(n)
        _r = C.compare(C.runs_of(0.0, C.WIDE)[:_n], C.runs_of(1.0, C.WIDE)[:_n])
        _say = (f"{C.number(10 ** _r['mean'], 1, L.lang)} keer zo laat, met een interval van {C.number(10 ** _r['low'], 1, L.lang)} tot {C.number(10 ** _r['high'], 1, L.lang)}")
        _pairs = f"{_r['pairs']} {'paar' if _r['pairs'] == 1 else 'paren'}" + (f", {_r['left_out']} weggevallen" if _r["left_out"] else "")
        _lost = "niet gehaald"
        return (C.factors_picture(_n, {"title": f"Zonder weight decay tegen met, {_n} {'seed' if _n == 1 else 'seeds'}", "x": "seed", "y": "keer zo laat",
                                       "band": "95%-interval", "none": _lost,
                                       "alt": f"Voor de seeds 0 tot en met {_n - 1}, hoeveel keer zo laat validatie zonder weight decay 99% haalt als met, op een logaritmische as: {', '.join(C.number(_x, 1, L.lang) if _x is not None else _lost for _x in C.factors(_n))}. Het gemiddelde is {_say}."}),
                f"Met {_n} {'seed' if _n == 1 else 'seeds'} ({_pairs}): zonder weight decay haalt validatie 99% {_say}.")

    explore_seeds = L.explore("seeds", "seeds", _frame, start=5, low=1, high=5, step=1, role="",
                              task="Hoe verandert het interval als je seeds toevoegt? En wat voegt de laatste seed toe?")
    explore_seeds
    return


@app.cell(hide_code=True)
def text_idea_after():
    _two, _four, _five = (C.compare(C.runs_of(0.0, C.WIDE)[:_n], C.runs_of(1.0, C.WIDE)[:_n]) for _n in (2, 4, 5))
    _lost = [_s for _s, _x in enumerate(C.factors(len(C.SEEDS))) if _x is None][0]   # the seed that is left out
    mo.vstack([
        mo.md(rf"""
        Met één seed is het interval een lijn: een bootstrap van één paar ziet geen spreiding, dus lijkt hij zeker terwijl hij niets weet. Met twee loopt het van het ene paar naar het andere, {C.number(10 ** _two['low'], 1, L.lang)} tot {C.number(10 ** _two['high'], 1, L.lang)}: een bootstrap kan alleen trekken uit de paren die hij heeft, dus met weinig paren is het te smal, zoals [Is het verschil echt?](/course/open/res-significance?lang=nl) al zei. Met vier loopt het van {C.number(10 ** _four['low'], 1, L.lang)} tot {C.number(10 ** _four['high'], 1, L.lang)}, en elk paar ligt ruim boven 1: zonder weight decay haalt validatie later 99%, op een manier die de seeds niet verklaren.

        Seed {_lost} voegt geen paar toe. Zonder weight decay haalde validatie binnen 12.000 stappen geen 99%, dus laat `compare` hem weg, en telt hem. Het is de traagste seed van allemaal, meer dan {C.number(C.at_least(_lost), 1, L.lang)} keer zo laat, dus door hem weg te laten is het gemiddelde van {C.number(10 ** _five['mean'], 1, L.lang)} te laag, niet te hoog. Je verslag zegt dat: {_five['pairs']} paren, {_five['left_out']} weggevallen, en waarom.

        **Leg dan de opzetten naast elkaar.** De claim van de paper gaat over $S_5$ en hoeveel data het kost; de jouwe over optellen bij {round(100 * C.WIDE)}% en hoeveel stappen het kost. Een getal dat moet kloppen, vraagt dezelfde opzet: elk detail dat de paper geeft, en de details die hij niet geeft, gevonden in zijn code of gevraagd aan zijn auteurs.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **gekoppeld op seed** | twee runs met dezelfde seed en één andere instelling, als paar vergeleken | `by_seed[log["seed"]]` |
        | $\log_{{10}} v_a - \log_{{10}} v_b$ | hoeveel factoren 10 later validatie 99% haalt in $a$ dan in $b$ | `math.log10(va) - math.log10(vb)` |
        | **ongedocumenteerd detail** | een keuze die de paper maakte zonder hem op te schrijven, en die het resultaat kan veranderen | |
        """),
        L.watch_out("Een paar waarin een run binnen het budget geen 99% haalt, heeft geen factor. Hem weglaten is alleen eerlijk als je het zegt: het resultaat telt hem, en je verslag noemt hem. Vallen er veel paren weg, dan gaat de vergelijking over de runs die snel genoeg waren, niet over allemaal."),
        mo.md(r"""
        ### Recept

        Om je reproductie met een claim van de paper te vergelijken:

        1. Draai beide instellingen over meerdere seeds, en koppel de runs op seed.
        2. Neem voor elk paar het verschil van de logaritmen van de momenten; tel de paren die ze niet halen.
        3. Geef het gemiddelde met zijn interval uit de gepaarde bootstrap, en zeg hoeveel keer zo laat dat is.
        4. Leg je opzet naast die van de paper: wat hij zegt, wat hij niet zegt, en wat jij bewust anders doet.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 en 3 van het recept met de hand, met de seeds 0 en 1 bij 90%, en seed 4, die geen paar heeft. Voer de cel uit.
    """)
    return


@app.cell
def example_pairs():
    import math

    _without = {0: 3548, 1: 5012, 4: None}   # v zonder weight decay, per seed; None: niet binnen 12.000 stappen
    _with = {0: 1259, 1: 1000, 4: 891}       # v met weight decay
    _diffs, _left_out = [], 0
    for _s in _without:
        if _without[_s] is None or _with[_s] is None:   # 2. geen factor: het paar valt weg, en telt mee
            _left_out = _left_out + 1
        else:
            _diffs.append(math.log10(_without[_s]) - math.log10(_with[_s]))   # 2. een verschil van logaritmen per paar
    print("verschillen van de logaritmen:", [round(_d, 2) for _d in _diffs], "weggevallen:", _left_out)
    print("keer zo laat, doorgaans:", round(10 ** (sum(_diffs) / len(_diffs)), 2))   # 3. het gemiddelde, als factor
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Seed 0 geeft 0,45, een factor 2,8, en seed 1 geeft 0,70, een factor 5,0; seed 4 heeft geen factor, dus valt hij weg, en telt hij mee. Het gemiddelde van de twee, 0,575, is een factor 3,76: niet het gemiddelde van 2,8 en 5,0, dat 3,9 is, want een gemiddelde van logaritmen is de factor in het midden op een logaritmische as. Twee paren geven geen interval dat iets waard is; wat meer seeds doen, laat het blok hierboven zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De vergelijking

        Maak `compare(runs_a, runs_b, n_boot, seed)` af: voor elke run van `runs_a` waarvan de seed in `by_seed` staat, de run van `runs_b` met die seed, en in beide de eerste stap waarop validatie `THRESHOLD` haalt, met `first_step` van je plan. Een paar waarin een van beide `None` is, valt weg en telt in `left_out`; voor de andere gaat `math.log10` van elke stap in `a` en `b`. `paired_bootstrap` en de rest zijn gegeven. De check maakt kleine runs waarvan validatie 99% haalt op een stap die je kunt aflezen, zoals in `run(0, 1000)`, en vergelijkt de resultaten. Als hij goed is, wordt hij in je project bewaard als `repro/compare.py`.
        """),
        L.task("Maak `compare` af: koppel op seed, lees v af in beide runs, tel een paar zonder, en bewaar de logaritmen van de andere."),
    ])
    return


@app.cell(hide_code=True)
def exercise_compare():
    box_compare = L.code_box("compare", lines=30)
    box_compare
    return (box_compare,)


@app.cell(hide_code=True)
def check_compare(box_compare):
    result_compare = L.check("compare", box_compare.value)
    result_compare
    return (result_compare,)


@app.cell(hide_code=True)
def hints_compare():
    L.hints(
        "compare",
        [
            "`if log[\"seed\"] in by_seed:` vindt het paar; `by_seed[log[\"seed\"]]` is de run van `runs_b` met die seed. `first_step(log[\"step\"], log[\"val\"], THRESHOLD)` geeft `v` van een run.",
            "Als `va is None or vb is None`, tel 1 op bij `left_out`; anders `a.append(math.log10(va))` en `b.append(math.log10(vb))`.",
        ],
        why="Koppelen op seed, niet op plek in de lijst, houdt de vergelijking juist als de ene instelling een seed meer heeft of een andere volgorde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_details():
    mo.md(r"""
    ### Oefening 2. Wat zegt de paper?

    Zet elk detail van de opzet in een groep: wat de paper zegt, in de tekst of zijn bijlage; wat hij niet zegt, zodat je in zijn code moet kijken of het zijn auteurs moet vragen; en wat je reproductie bewust anders doet. Houd de tekst van de paper ernaast open: paragraaf 3 en bijlage A.1.
    """)
    return


@app.cell(hide_code=True)
def exercise_details():
    sort_details = L.sort("details")
    sort_details
    return (sort_details,)


@app.cell(hide_code=True)
def check_details(sort_details):
    result_details = L.check("details", sort_details)
    result_details
    return (result_details,)


@app.cell(hide_code=True)
def text_exercise_agrees():
    mo.md(r"""
    ### Oefening 3. Komt het overeen?
    """)
    return


@app.cell(hide_code=True)
def choice_agrees_cell():
    _r = C.compare(C.runs_of(0.0, C.WIDE), C.runs_of(1.0, C.WIDE))
    choice_agrees = L.choice(
        "agrees",
        question=f"Je compare over vijf seeds bij {round(100 * C.WIDE)}% geeft, zonder weight decay tegen met, {_r['pairs']} paren en {_r['left_out']} weggevallen, een gemiddelde van {C.number(_r['mean'], 2, L.lang)} en een interval van {C.number(_r['low'], 2, L.lang)} tot {C.number(_r['high'], 2, L.lang)}. De paper zegt dat weight decay de generalisatie \"het meest verbetert\", op S₅, gemeten als hoeveel data het kost. Wat zegt je verslag?",
        options={
            "right": "Het komt in richting overeen met de paper, op een andere taak en maat",
            "match": f"Het reproduceert het getal van de paper: weight decay maakt validatie {C.number(10 ** _r['mean'], 1, L.lang)} keer zo snel",
            "differ": "Het komt niet overeen, want de paper mat op een andere taak",
        },
        correct="right",
        why={
            "match": "De paper geeft zo'n getal niet voor optellen bij 90%: zijn claim gaat over hoeveel data S₅ kost. Je factor is je eigen resultaat; de claim van de paper is de richting die hij aanwijst.",
            "differ": "Een andere taak maakt de vergelijking zwakker, niet fout: allebei zeggen ze dat weight decay de generalisatie helpt, en je interval ligt ruim boven 0. Zeg wat er verschilt, en dat de richting overeenkomt.",
            "right": f"Zonder weight decay haalt validatie 99% {C.number(10 ** _r['mean'], 1, L.lang)} keer zo laat, met een interval van {C.number(10 ** _r['low'], 1, L.lang)} tot {C.number(10 ** _r['high'], 1, L.lang)}, over {_r['pairs']} van de {len(C.SEEDS)} seeds; in de andere kwam hij er binnen 12.000 stappen niet. Dat is dezelfde richting als de claim van de paper, op optellen en in stappen, en het is de zin voor je verslag.",
        },
    )
    choice_agrees
    return (choice_agrees,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je `compare` kan je reproductie seed voor seed naast een referentie leggen. De cel hieronder vergelijkt je runs uit [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl), in `work/projects/repro/runs.jsonl`, met de runs van de course met dezelfde seeds: doet je kit wat die van de course doet, dan haalt validatie bij ongeveer dezelfde stap 99%, en ligt het gemiddelde dicht bij 0.
    """)
    return


@app.cell(hide_code=True)
def demo_compare(result_compare):
    result_compare  # opnieuw uitvoeren als de oefening de vergelijking bewaart
    _r = C.demo()
    if _r is None:
        _out = mo.md("*Maak eerst oefening 1 af: dan staat je `compare` in je project, en gebruikt deze cel hem.*")
    elif _r == "runs":
        _out = mo.md("*Er zijn nog geen runs van jou: train ze met de knop van [Begeleide reproductie 4](/course/open/repro-guided-figure?lang=nl), en kom dan terug.*")
    elif _r == "compare":
        _out = mo.md("*Je `repro/compare.py` stopt met een fout. Heb je hem na zijn check veranderd? Verwijder hem, en check zijn oefening opnieuw.*")
    elif not _r["pairs"]:
        _out = mo.md(f"*Geen van je runs heeft een paar met die van de course dat in beide 99% haalt ({_r['left_out']} weggevallen).*")
    else:
        _same = max(abs(_r["low"]), abs(_r["high"])) < 0.06   # elk paar binnen één gelogde stap: 20 per factor 10 liggen 0,05 uit elkaar
        _verdict = ("Je reproductie komt seed voor seed overeen met die van de course." if _same
                    else "In minstens één seed verschillen je runs meer dan een gelogde stap van die van de course: een verschil om op te sporen, in het apparaat, het aantal threads, of een module die je na zijn check veranderde.")
        _stand = (" De versie van de course sprong in voor: " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + ".") if _r["stand_ins"] else ""
        _out = mo.md(f"""
Je runs tegen die van de course, seeds {', '.join(str(_s) for _s in _r['seeds'])}: {_r['pairs']} {'paar' if _r['pairs'] == 1 else 'paren'}, {_r['left_out']} weggevallen, een gemiddelde van **{L.number(_r['mean'], 3)}**, met een interval van {L.number(_r['low'], 3)} tot {L.number(_r['high'], 3)}. {_verdict}{_stand}
""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De gepaarde bootstrap uit [Is het verschil echt?](/course/open/res-significance?lang=nl), nu over seeds van je reproductie; het grid uit [Begeleide reproductie 5](/course/open/repro-guided-ablation?lang=nl), nu herhaald over seeds; en de bijlage van de paper die je las in [Begeleide reproductie 2](/course/open/repro-guided-build?lang=nl), nu als checklist. Je kit heeft nu `repro/compare.py`. In de volgende les, *Begeleide reproductie 7: schrijf het verslag*, schrijf je op wat je vond: de claim, de opzet, het resultaat, de verschillen, en wat je niet kon controleren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_compare, result_details, choice_agrees):
    L.wrap_up(result_compare, result_details, choice_agrees)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
