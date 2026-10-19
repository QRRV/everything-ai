# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-choose-paper. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-choose-paper", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Kritisch lezen: claims tegen bewijs](/course/open/res-critical-reading?lang=nl) en [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_smaller():
    quick_smaller = L.choice(
        "quick-smaller",
        question="Kritisch lezen vindt overclaims in een paper. Wat blijft er meestal over?",
        options={
            "nothing": "Niets: één overclaim maakt de hele paper onwaar",
            "right": "Een kleinere claim: de claim die het bewijs wél draagt",
            "same": "Dezelfde claim: overclaims zijn alleen woorden",
        },
        correct="right",
        why={
            "nothing": "Een overclaim zegt meer dan het bewijs; het bewijs zelf kan nog steeds deugen. Zie [Kritisch lezen: claims tegen bewijs](/course/open/res-critical-reading?lang=nl).",
            "same": "Een overclaim zegt meer dan de experimenten laten zien; de claim om op voort te bouwen is die ze wél laten zien. Zie [Kritisch lezen: claims tegen bewijs](/course/open/res-critical-reading?lang=nl).",
            "right": "En een kleine, deugdelijke claim is precies wat deze les zoekt: een die je zelf kunt toetsen.",
        },
    )
    quick_smaller
    return (quick_smaller,)


@app.cell(hide_code=True)
def quick_check_flops():
    quick_flops = L.choice(
        "quick-6nd",
        question="Een model traint 10 epochs op een dataset van 100 miljoen tokens. Wat is $D$ in $6ND$?",
        options={
            "epoch": "100 miljoen",
            "right": "1 miljard",
            "six": "6 miljard",
        },
        correct="right",
        why={
            "epoch": "Dat is één epoch. $D$ telt elk token waarop het model traint, epochs meegeteld. Zie [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).",
            "six": "Dat is $6D$: de 6 zijn de FLOPs per parameter per token, geen deel van $D$. Zie [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl).",
            "right": "$10 \\times 10^8 = 10^9$: elk token telt, epochs meegeteld. Deze les telt een niveau hoger: runs maal seeds.",
        },
    )
    quick_flops
    return (quick_flops,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een paper nabouwen is een van de beste manieren om hem te begrijpen, en om te weten of hij klopt. Maar veel pogingen stranden vóór het eerste resultaat: het experiment vraagt maanden GPU-tijd, de data is niet te krijgen, of de claim is te vaag om te toetsen. Sinds 2018 vraagt de ML Reproducibility Challenge mensen om papers van de grote conferenties na te bouwen, en elke reproductie begint met een keuze: welke paper, en welke claim. In de volgende lessen bouw je zelf een paper na, eerst begeleid, dan zelfstandig.
    """)
    return


@app.cell(hide_code=True)
def predict_code():
    guess_code = L.guess(
        "shared-code",
        question="In 2018 bekeken Odd Erik Gundersen en Sigbjørn Kjensmo 400 papers van twee grote AI-conferenties. Hoeveel daarvan deelden de code van hun methode?",
        options={
            "most": "Ongeveer 90%",
            "half": "Ongeveer de helft",
            "right": "Ongeveer 6%",
        },
        reveal="**Ongeveer 6%.** Ongeveer een derde deelde de data waarop getest was, en ongeveer de helft gaf pseudocode, de stappen van de methode uitgeschreven in gewone woorden en formules. Sindsdien vragen conferenties auteurs een checklist over code en data in te vullen, en delen veel meer papers hun code. Maar niet allemaal, en code die op de machine van een ander draait is nog zeldzamer.",
    )
    guess_code
    return (guess_code,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Stap 1: past het in je compute?** Schat één run met 6ND, en de hele reproductie als runs × seeds. Vergelijk dan met wat je hebt: de CPU van een laptop haalt ongeveer $10^{11}$ FLOP/s als hij een klein model traint, dus op een dag ongeveer $10^{16}$ FLOPs; één GPU een week lang, op 40% van de piek van een A100, ongeveer $10^{20}$; een klein model haalt veel minder van die piek. Zoek de kleinste opzet uit de paper waarin de claim nog te zien is.
        """),
        mo.Html(C.budget_picture({"title": "Eén run van vier kandidaten", "x": "FLOPs van één run",
                                  "names": ["0,4M parameters, 0,25B tokens", "10M parameters, 1B tokens", "124M parameters, 10B tokens", "7B parameters, 2T tokens"],
                                  "laptop": "laptop, een dag", "gpu": "GPU, een week",
                                  "alt": "Vier staven op een logaritmische schaal van FLOPs. 0,4 miljoen parameters op 0,25 miljard tokens: ongeveer 6 keer 10 tot de 14e, ruim binnen een dag van een laptop. 10 miljoen op 1 miljard: 6 keer 10 tot de 16e, een week van een laptop, minder dan een uur van een GPU. 124 miljoen op 10 miljard: ongeveer 7 keer 10 tot de 18e, binnen een week van één GPU. 7 miljard op 2 biljoen: ongeveer 8 keer 10 tot de 22e, duizend keer meer dan een week van één GPU."})),
        mo.md(r"""
        **Stap 2: wat is er?** Code, data, checkpoints, en de instellingen in de appendix. Niet elk ontbrekend stuk kost evenveel: data die je niet kunt krijgen houdt je tegen, code die ontbreekt kun je schrijven, instellingen die ontbreken moet je raden of aan de auteurs vragen.

        **Stap 3: een kleine, toetsbare claim.** Eén zin, één figuur of tabel waar hij op rust, en een resultaat dat zou laten zien dat hij niet klopt. "Een stap naar begrip" noemt zo'n resultaat niet; "de methode verslaat de baseline met 2 procentpunt accuracy, in elk van 3 seeds" wel.

        **Stap 4: de kosten voordat je begint.** Runs × seeds × FLOPs per run, gedeeld door wat je machine doet, maal een marge van 2 of 3 voor de runs die misgaan. Past het niet, kies dan een kleinere opzet, of een andere claim.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **reproductie** | het experiment van een ander opnieuw doen, vanuit de paper, om te zien of het resultaat standhoudt |
        | **falsifieerbaar** | van een claim: er is een resultaat dat zou laten zien dat hij niet klopt |
        | **ML Reproducibility Challenge** | een jaarlijks evenement, sinds 2018, waarin mensen papers van grote ML-conferenties nabouwen en publiceren wat ze vonden |
        """),
        L.watch_out("Een reproductie die het resultaat van de paper niet vindt, is ook een resultaat, als je deed wat de paper zei. Schrijf elke keuze op die je zelf moest maken: daar zit vaak het verschil."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo kies je een paper die je kunt nabouwen:

    1. Compute: schat één run met 6ND, en zoek de kleinste opzet uit de paper waarin de claim te zien is.
    2. Beschikbaarheid: kijk of er code, data, checkpoints en de instellingen zijn; data die je niet kunt krijgen sluit een paper uit.
    3. Claim: schrijf één zin, de figuur waar hij op rust, en het resultaat dat zou laten zien dat hij niet klopt.
    4. Kosten: runs × seeds × FLOPs per run, gedeeld door de FLOP/s van je machine, maal een marge van 2 of 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Nieuw zijn de kosten van een hele reproductie, niet van één run: de tweede kandidaat uit het plaatje, met zijn baseline, elk met 3 seeds.
    """)
    return


@app.cell
def example_cost():
    _n_params, _n_tokens = 10e6, 1e9  # één run: een kleine GPT op 1 miljard tokens
    _runs, _seeds = 2, 3  # de methode en zijn baseline, elk met 3 seeds
    _flops = 6 * _n_params * _n_tokens * _runs * _seeds
    for _machine, _speed in [("laptop", 1e11), ("GPU", 312e12 * 0.1)]:  # een A100 op 10% van zijn piek: het model is klein
        _hours = _flops / _speed / 3600
        print(_machine, round(_hours, 1), "uur, met een marge van 2:", round(2 * _hours, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zes runs kosten $3{,}6 \times 10^{17}$ FLOPs: 1.000 uur op een laptop, zes weken zonder te stoppen, en ongeveer 3 uur op één GPU. Dezelfde reproductie is onhaalbaar op de ene machine en makkelijk op de andere; met een kleinere opzet, zoals 100 miljoen tokens per run, past hij in ongeveer vier dagen op een laptop, acht met een marge van 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Jouw beurt

    ### Oefening 1. Een goede eerste reproductie?

    Zes claims, uit echte en verzonnen papers. Zet elke claim in zijn groep, voor een reproductie op je eigen machine.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_first = L.sort("first-repro")
    sort_first
    return (sort_first,)


@app.cell(hide_code=True)
def check_sort(sort_first):
    result_first = L.check("first-repro", sort_first)
    result_first
    return (result_first,)


@app.cell(hide_code=True)
def hints_sort():
    L.hints(
        "first-repro",
        [
            "Stap 1 tot en met 3 van het recept, voor elke claim: past hij op je machine, kun je aan de data komen, en welk resultaat zou laten zien dat hij niet klopt?",
        ],
        why="**Goed:** de kleine transformer op optellen modulo een getal, dropout in een kleine GPT, batch normalization in een klein netwerk op de cijfers: kleine modellen, openbare of gegenereerde data, een getal om te meten. **Nu niet:** 70 miljard parameters op 15 biljoen tokens (ver voorbij één GPU), dossiers van één ziekenhuis (data die je niet krijgt), \"echt begrijpen\" (geen resultaat kan laten zien dat het niet klopt).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_falsifiable():
    mo.md(r"""
    ### Oefening 2. Een claim die je kunt toetsen
    """)
    return


@app.cell(hide_code=True)
def choice_falsifiable_cell():
    choice_falsifiable = L.choice(
        "falsifiable",
        question="Drie claims die iemand over batch normalization zou kunnen schrijven. Welke kan je reproductie onderuithalen?",
        options={
            "right": "Met batch normalization leest een MLP met 10 hidden layers na één epoch met een learning rate van 1,0 minstens 80% van de testcijfers goed",
            "light": "Onze resultaten werpen nieuw licht op waarom batch normalization diepe netwerken makkelijker laat trainen",
            "deep": "Batch normalization onthult een diep verband tussen de schaal van de layers en hoe goed ze leren",
        },
        correct="right",
        why={
            "light": "Geen resultaat kan laten zien dat er geen licht geworpen is: de zin noemt niets om te meten.",
            "deep": "\"Een diep verband\" noemt geen resultaat: wat zou je meten, en welk getal zou laten zien dat het niet klopt?",
            "right": "Eén getal, één moment in de training, één opzet: blijven je runs na die epoch onder 80%, dan klopt de claim niet.",
        },
    )
    choice_falsifiable
    return (choice_falsifiable,)


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.md(r"""
    ### Oefening 3. Je plan
    """)
    return


@app.cell(hide_code=True)
def explain_plan():
    explain_plan_cell = L.explain(
        "my-plan",
        question="Kies de claim over dropout of over batch normalization uit oefening 1, of een paper die je zelf kiest; de eerste claim, de late sprong die grokking heet, is voor de volgende les. Schrijf je plan: de claim in één zin, de figuur waar hij op rust, het resultaat dat zou laten zien dat hij niet klopt, welke code en data er zijn, en de kosten van de reproductie in FLOPs en in uren op je machine.",
        model_answer="**Claim:** in de kleine GPT van ongeveer een miljoen parameters die de course op een CPU traint, geeft dropout van 0,1 een lagere beste validation loss op Tiny Shakespeare dan geen dropout. **Figuur:** de validation loss na elke epoch, met en zonder dropout. **Klopt niet als:** met 3 seeds per opzet is de beste validation loss met dropout niet lager dan zonder, of is het verschil kleiner dan de spreiding tussen seeds. **Code en data:** Tiny Shakespeare zit in de course, mijn GPT en mijn Trainer staan in mijn project, en dropout is één layer meer, `nn.Dropout(0.1)`. **Kosten:** ongeveer $10^6$ parameters, en 10 epochs over een miljoen tekens is $10^7$ tokens per run: $6 \\times 10^6 \\times 10^7 = 6 \\times 10^{13}$ FLOPs, ongeveer 10 minuten op een laptop met $10^{11}$ FLOP/s; met en zonder dropout, elk met 3 seeds en een marge van 2, ongeveer 2 uur.",
        points=[
            "je schrijft de claim als één zin, met de figuur of tabel waar hij op rust",
            "je noemt een resultaat dat zou laten zien dat de claim niet klopt",
            "je zegt welke code en data er zijn, of hoe je eraan komt",
            "je schat de kosten met 6ND, maal runs en seeds, in uren op je machine",
        ],
    )
    explain_plan_cell
    return (explain_plan_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De kleinere, deugdelijke claims van [Kritisch lezen: claims tegen bewijs](/course/open/res-critical-reading?lang=nl), doorgerekend met [Trainingscompute schatten: de 6ND-vuistregel](/course/open/sys-compute-6nd?lang=nl). De volgende les, *Begeleide reproductie 1: lees de paper en kies de claim*, doet dit samen met jou voor de paper over grokking.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_falsifiable, explain_plan_cell, result_first):
    L.wrap_up(result_first, choice_falsifiable, explain_plan_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
