# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-significance. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-significance", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl) en [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_resample():
    quick_resample = L.choice(
        "quick-resample",
        question="Je bootstrapt het gemiddelde van 8 runs. Wat is één resample?",
        options={
            "right": "8 van je runs, getrokken met teruglegging",
            "half": "4 van je runs, getrokken zonder teruglegging",
            "new": "8 nieuwe runs, getraind met andere seeds",
        },
        correct="right",
        why={
            "half": "Een resample is even groot als je steekproef, en een run kan twee keer voorkomen. Zie [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl).",
            "new": "De bootstrap traint niets nieuws: hij trekt uit de runs die je hebt. Zie [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl).",
            "right": "Evenveel als je hebt, met teruglegging, dus sommige runs komen twee keer voor en andere helemaal niet.",
        },
    )
    quick_resample
    return (quick_resample,)


@app.cell(hide_code=True)
def quick_check_baseline_rate():
    quick_rate = L.choice(
        "quick-rate",
        question="Je vergelijkt je methode met een baseline. Welke learning rate krijgt de baseline?",
        options={
            "yours": "De rate die het best was voor jouw methode",
            "paper": "Die uit zijn eigen paper",
            "right": "Zijn eigen beste, uit dezelfde pogingen",
        },
        correct="right",
        why={
            "yours": "Een rate die bij jouw methode past, past misschien niet bij de baseline. Zie [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl).",
            "paper": "Zijn paper had een andere opzet, en misschien een andere tuning effort. Zie [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl).",
            "right": "Dezelfde learning rates geprobeerd voor allebei, en elk bij zijn eigen beste: dezelfde tuning effort.",
        },
    )
    quick_rate
    return (quick_rate,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl) eindigde de kleine GPT zonder warmup bij elk van de vijf seeds lager dan met warmup, en toch overlapten de twee intervallen flink. Die intervallen behandelden de runs als twee losse groepen. Maar de runs horen in paren bij elkaar: met dezelfde seed beginnen beide runs met dezelfde weights en zien ze de mini-batches in dezelfde volgorde. Eindigt de ene run van een seed hoog, dan doet de andere dat meestal ook. Vergelijk de runs paar voor paar, en dat gedeelde deel valt weg. <!-- termen:ok -->

    **Een voorbeeld uit het dagelijks leven.** Je wilt weten welke van twee schoenzolen minder slijt. Geef tien kinderen zool A en tien andere kinderen zool B, en de verschillen tussen de kinderen, hoeveel ze rennen en springen, overstemmen de zolen. Geef elk kind van elke soort één zool, één aan elke voet, en vergelijk de twee voeten van elk kind: de verschillen tussen kinderen vallen weg.
    """)
    return


@app.cell(hide_code=True)
def predict_chance():
    guess_chance = L.guess(
        "all-ten",
        question="Stel dat warmup helemaal geen verschil maakt. Dan is het bij elke seed kop of munt welke van zijn twee runs lager eindigt. In de runs van de course eindigden alle tien seeds lager zonder warmup. Hoe waarschijnlijk is dat door toeval alleen?",
        options={
            "ten": "Ongeveer 1 op 10",
            "hundred": "Ongeveer 1 op 100",
            "right": "Ongeveer 1 op 1000",
        },
        reveal="**Ongeveer 1 op 1000.** Alle tien lager is als tien keer achter elkaar kop gooien: $(1/2)^{10} = 1/1024$. Ook als je beide richtingen telt, allemaal lager of allemaal hoger, is het 2 op 1024, 0,2%. Toeval alleen verklaart het slecht. De course trainde seeds 5 tot en met 9 voor deze les; seeds 0 tot en met 4 zijn die uit Eén run is geen resultaat, en ze kwamen precies hetzelfde uit.",
    )
    guess_chance
    return (guess_chance,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Geef beide instellingen dezelfde seeds, en neem het **verschil** van elk paar: de run met warmup min de run zonder. Vraag dan of die verschillen door toeval alleen kunnen komen.

        De **nulhypothese** is dat de verandering geen verschil maakt. Klopt die, dan kunnen de twee runs van een seed net zo goed van plaats wisselen, en zo'n wissel draait het teken van hun verschil om. Een **permutatietoets** doet dat: hij draait het teken van elk verschil willekeurig om, heel vaak, en neemt elke keer het gemiddelde. Die gemiddelden zijn wat toeval alleen geeft. De **p-waarde** is het deel ervan dat minstens zo ver van 0 ligt als het gemiddelde dat jij mat, aan welke kant ook: voordat je mat, kon de verandering beide kanten op gaan. Is p klein, meestal onder 0,05, dan verklaart toeval alleen je resultaat slecht, en heet het verschil **statistisch significant**.

        Met weinig paren zijn er weinig patronen van tekens: $2^n$ voor $n$ paren. Met vijf paren zijn dat er 32, en het meest extreme patroon, allemaal positief of allemaal negatief, geeft nog steeds p = 2/32 = 0,0625. Met deze toets komen vijf seeds nooit onder 0,05, hoe duidelijk het verschil ook is.

        Een **gepaarde bootstrap** geeft in plaats daarvan een interval: hij resamplet de verschillen, met teruglegging, en neemt elke keer het gemiddelde; de middelste 95% van die gemiddelden is het interval van het verschil. Ligt 0 er niet in, dan is toeval alleen ook een slechte verklaring. Zoals bij elke bootstrap is het interval te smal als je maar een paar paren hebt. Lijken de twee runs van een seed op elkaar, zoals hier, en resample je de twee instellingen elk apart, dan haal je de paren uit elkaar: het interval wordt veel breder, omdat de verschillen tussen seeds dan als ruis tellen.

        Het eerste plaatje: elk van de tien seeds als een lijn van zijn run met warmup naar zijn run zonder. Het tweede: de gemiddelden van alle 1024 patronen van tekens van de tien verschillen, en het gemiddelde dat je mat: geen patroon ligt verder van 0, en alleen zijn spiegelbeeld, met alle tekens omgedraaid, ligt even ver.
        """),
        mo.Html(C.pairs_picture({"left": "Seed voor seed", "right": "Toeval alleen", "y": "validation loss", "with": "met warmup", "without": "zonder",
                                 "x": "gemiddeld verschil", "count": "patronen", "yours": "gemeten", "lang": L.lang,
                                 "alt_left": "Tien lijnen, één per seed, van de run met warmup naar de run zonder; alle tien gaan omlaag, met 0,002 tot 0,014.",
                                 "alt_right": "Een histogram van de gemiddelden van alle 1024 patronen van tekens van de tien verschillen, rond 0, van ongeveer −0,006 tot 0,006; het gemeten gemiddelde, 0,0059, ligt helemaal aan de rechterrand."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **nulhypothese** | de aanname dat een verandering geen verschil maakt, die een toets controleert |
        | **permutatietoets** | een toets die husselt wat volgens de nulhypothese niet uitmaakt, hier de tekens van de verschillen, om te zien wat toeval alleen geeft; papers over NLP noemen deze gepaarde vorm de approximate randomization test | <!-- termen:ok -->
        | **p-waarde** | het deel van de toevalsuitkomsten dat minstens zo extreem is als de jouwe, als de nulhypothese klopt |
        | **statistisch significant** | met een p-waarde onder een vooraf gekozen grens, meestal 0,05 |
        | **gepaarde bootstrap** | de bootstrap van de verschillen van paren, die elk paar bij elkaar houdt |
        """),
        L.watch_out("Een p-waarde is niet de kans dat de nulhypothese klopt, maar hoe vaak toeval alleen een resultaat zou geven dat minstens zo extreem is als het jouwe, als die klopt. Een p-waarde boven 0,05 laat niet zien dat er geen verschil is: met weinig seeds blijft zelfs een echt verschil verborgen. En een heel kleine p-waarde laat niet zien dat een verschil groot is: hier is het 0,006."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo toets je of een verschil tussen twee instellingen toeval kan zijn:

    1. Maak paren van de runs: dezelfde seeds voor beide instellingen, en het verschil per seed.
    2. Permutatietoets: draai de tekens van de verschillen willekeurig om, heel vaak, en neem elke keer het gemiddelde; p is het deel van die gemiddelden dat minstens zo ver van 0 ligt als het jouwe.
    3. Gepaarde bootstrap: resample de verschillen met teruglegging, heel vaak, en neem elke keer het gemiddelde; de middelste 95% is het interval van het verschil.
    4. Rapporteer het gemiddelde verschil, zijn interval, p en het aantal seeds.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en 2 met de verschillen van de course, met warmup min zonder. Het commentaar zegt welke stap elk deel is. Hij is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_permutation():
    import numpy as np

    _all = np.array([0.0036, 0.0083, 0.0059, 0.0022, 0.005, 0.0135, 0.0048, 0.0053, 0.0071, 0.0036])
    # 1. de verschillen van seeds 0 tot en met 4
    _diffs = _all[:5]  # maak af: de verschillen van alle tien seeds, _all
    _observed = np.mean(_diffs)
    # 2. de permutatietoets: willekeurige tekens, 10.000 keer, en hoe vaak het gemiddelde minstens zo ver van 0 ligt als het onze
    _rng = np.random.default_rng(0)
    _means = []
    for _ in range(10000):
        _signs = _rng.choice([-1, 1], size=len(_diffs))
        _means.append(np.mean(_signs * _diffs))
    _p = np.mean(np.abs(_means) >= abs(_observed))
    print("seeds:", len(_diffs), " gemiddeld verschil:", round(_observed, 4), " p:", _p)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met vijf seeds is p ongeveer 0,06: alle vijf positief is het meest extreme patroon dat er is, en toch komen 2 van de 32 patronen van tekens even ver. Met alle tien seeds is p ongeveer 0,002. Het verschil veranderde nauwelijks, van ongeveer 0,005 naar 0,006; wat veranderde, is hoeveel toeval kan verklaren.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bootstrap():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een gepaarde bootstrap

        Maak `paired_bootstrap(a, b, n_boot, seed)` af: stap 3 van het recept als functie. `a` en `b` zijn de runs van twee instellingen, met dezelfde seeds in dezelfde volgorde. Hij geeft drie getallen terug: het gemiddelde van de verschillen `a` min `b`, en de onder- en bovenkant van het 95%-bootstrapinterval van dat gemiddelde, uit `n_boot` resamples met een random number generator die met `seed` gemaakt is. De loop heeft de vorm van het uitgewerkte voorbeeld, met een resample van de verschillen in plaats van willekeurige tekens. Vervang de twee `...`. De check roept `paired_bootstrap(with_warmup, without_warmup, 10000, 0)` aan met de tien seeds van de course, en probeert andere runs. Een bootstrap is willekeurig, dus de check laat de kleine verschillen toe die een andere manier van trekken geeft.
        """),
        L.task("Laat `paired_bootstrap(a, b, n_boot, seed)` drie getallen teruggeven: het gemiddelde van de verschillen `a` min `b`, paar voor paar, en de twee kanten van het 95%-bootstrapinterval van dat gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bootstrap():
    box_bootstrap = L.code_box("paired-bootstrap", lines=10)
    box_bootstrap
    return (box_bootstrap,)


@app.cell(hide_code=True)
def check_bootstrap(box_bootstrap):
    result_bootstrap = L.check("paired-bootstrap", box_bootstrap.value)
    result_bootstrap
    return (result_bootstrap,)


@app.cell(hide_code=True)
def hints_bootstrap():
    L.hints(
        "paired-bootstrap",
        [
            "Eerst de verschillen: trek de twee lijsten van elkaar af als NumPy-arrays, elke run in `a` min de run in `b` met dezelfde seed. Resample die verschillen dan zoals je runs resamplede in [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
            "`diffs = np.array(___) - np.array(___)`, en in de loop `means.append(np.mean(rng.choice(___, size=len(diffs))))`.",
        ],
        why="```python\nimport numpy as np\n\n\ndef paired_bootstrap(a, b, n_boot, seed):\n    diffs = np.array(a) - np.array(b)\n    rng = np.random.default_rng(seed)\n    means = []\n    for _ in range(n_boot):\n        means.append(np.mean(rng.choice(diffs, size=len(diffs))))\n    low, high = np.quantile(means, [0.025, 0.975])\n    return [np.mean(diffs), low, high]\n```\n\nElke resample trekt hele paren, als hun verschillen, dus de twee runs van een seed blijven bij elkaar. De middelste 95% van de gemiddelden van de resamples is het interval van het gemiddelde verschil.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Gepaard of niet

    Jouw `paired_bootstrap` op de paren van de course, naast de bootstrap die elke instelling apart resamplet, en de exacte p-waarde van de permutatietoets over alle patronen van tekens. Zolang oefening 1 nog niet goed is, rekent de `paired_bootstrap` van de course de gepaarde intervallen uit.

    In de eerste kolom maken de paren het interval veel smaller: de twee runs van een seed lijken op elkaar. In de tweede niet: met een ander aantal layers geeft dezelfde seed niet meer dezelfde beginwaarden van de weights of dezelfde volgorde van de mini-batches, dus de twee runs van een seed hebben minder gemeen. En met drie paren loopt het gepaarde interval alleen van het kleinste verschil tot het grootste: veel te smal om op te vertrouwen.
    """)
    return


@app.cell(hide_code=True)
def demo_paired(box_bootstrap, result_bootstrap):
    _boot, _yours = C.paired_bootstrap_of(box_bootstrap.value if result_bootstrap.state == "correct" else None)
    _s = lambda x: C.signed(x, 4, L.lang)
    # één kolom per vergelijking, zodat de tabel op een telefoon past
    _cols = []
    for _a, _b in [(C.WARMUP, C.NO_WARMUP), (C.FOUR_LAYERS, C.TWO_LAYERS)]:
        _m, _lo, _hi = _boot(list(_a), list(_b), 10000, 0)
        _u = C.bootstrap(_a, _b, paired=False)
        _p = C.flip_p(C.differences(_a, _b))
        _cols.append([str(len(_a)), _s(_m), f"{_s(_lo)} tot {_s(_hi)}", f"{_s(_u[1])} tot {_s(_u[2])}", C.number(_p, 3, L.lang)])
    _names = ["seeds", "gemiddeld verschil", "gepaard interval", "ongepaard interval", "p"]
    _lines = [f"| {_names[_i]} | {_cols[0][_i]} | {_cols[1][_i]} |" for _i in range(len(_names))]
    _who = "" if _yours else "\n\nOefening 1 is nog niet goed, dus de `paired_bootstrap` van de course maakte de gepaarde intervallen."
    mo.md("| | met warmup min zonder | 4 layers min 2 layers, allebei bij 0,03 |\n|---|---|---|\n" + "\n".join(_lines) + _who)
    return


@app.cell(hide_code=True)
def text_exercise_three():
    mo.md(r"""
    ### Oefening 2. Drie seeds
    """)
    return


@app.cell(hide_code=True)
def choice_three_cell():
    choice_three = L.choice(
        "three-seeds",
        question="De GPT met 4 en die met 2 layers, elk bij zijn beste learning rate, met seeds 0 tot en met 2: de verschillen zijn −0,0114; +0,0003 en +0,0038, en de permutatietoets geeft p = 1. Wat concludeer je?",
        options={
            "same": "De GPT met 2 en die met 4 layers zijn hier even goed",
            "right": "Er is geen verschil te zien, maar er kan er een zijn",
            "better": "4 layers wint: zijn gemiddelde is 0,002 lager",
        },
        correct="right",
        why={
            "same": "Een grote p-waarde laat niet zien dat er geen verschil is. Met drie paren zijn er maar 8 patronen van tekens, dus p kan nooit onder 2/8 = 0,25 komen.",
            "better": "Zijn gemiddelde is 0,002 lager, maar één seed gaat de ene kant op en twee de andere: toeval verklaart dat makkelijk.",
            "right": "Deze runs geven geen reden om te denken dat 4 layers hier helpt. Maar er zijn maar 8 patronen van tekens, dus met drie seeds kan p niet onder 0,25 komen, hoe groot het verschil ook is: om een klein verschil te laten zien, heb je veel meer seeds nodig.",
        },
    )
    choice_three
    return (choice_three,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Oefening 3. Rapporteer het effect van warmup
    """)
    return


@app.cell(hide_code=True)
def explain_report():
    explain_report_cell = L.explain(
        "report",
        question="Schrijf het resultaat van warmup op zoals in een paper: de opzet, de seeds, het gemiddelde verschil met zijn interval, de p-waarde en de toets, en wat het betekent.",
        model_answer="**Opzet:** de kleine GPT uit Eén ding tegelijk veranderen, 6 epochs op het eerste kwart van Tiny Shakespeare, met een warmup van 100 stappen en zonder, seeds 0 tot en met 9 voor allebei. **Resultaat:** zonder warmup eindigde de validation loss bij alle tien seeds lager, gemiddeld 0,006 (gepaard 95%-bootstrapinterval 0,004 tot 0,008; permutatietoets over de tekens van de verschillen, p = 0,002). **Betekenis:** toeval alleen verklaart dit verschil slecht, maar het is klein: een learning rate van 0,003 in plaats van 0,001 verlaagt de loss met 0,3. Met alleen de eerste vijf seeds had dezelfde toets niet onder p = 0,0625 kunnen komen.",
        points=[
            "je noemt de opzet en dat beide instellingen dezelfde tien seeds hadden",
            "je geeft het gemiddelde verschil, zijn gepaarde interval, de p-waarde en welke toets die gaf",
            "je zegt wat het betekent: toeval alleen verklaart het slecht, en toch is het klein naast wat de learning rate doet",
        ],
    )
    explain_report_cell
    return (explain_report_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De bootstrap komt uit [Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap](/course/open/stats-bootstrap?lang=nl), de seeds uit [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl), en de runs met 2 en 4 layers uit [Eerlijke vergelijkingen](/course/open/res-fair-baselines?lang=nl). Paren vormen werkt hier omdat beide instellingen dezelfde seeds hadden; hetzelfde idee werkt voor twee modellen op dezelfde testvoorbeelden, paar voor paar. De volgende les, *Figuren die iets laten zien*, zet een resultaat als dit in een figuur. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_three, explain_report_cell, result_bootstrap):
    L.wrap_up(result_bootstrap, choice_three, explain_report_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
