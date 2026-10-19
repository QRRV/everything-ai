# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-fair-baselines. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-fair-baselines", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) en [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl). <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="Een model haalt een accuracy van 91% op de test set. Waarmee vergelijk je dat eerst?",
        options={
            "train": "De accuracy op de training set",
            "right": "Een eenvoudige baseline op dezelfde test set",
            "perfect": "100%, de best mogelijke score",
        },
        correct="right",
        why={
            "train": "De training set zegt hoe goed het model past bij wat het gezien heeft, niet of 91% goed is. Zie [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "perfect": "100% is het plafond, geen vergelijking: het zegt niet of 91% op deze test set moeilijk of makkelijk te halen is. Zie [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "right": "Als altijd de meest voorkomende klasse gokken ook al 90% haalt, zegt 91% weinig.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_noise():
    quick_noise = L.choice(
        "quick-noise",
        question="Met alleen een andere seed eindigt de kleine GPT een paar honderdsten hoger of lager. Eén run met een learning rate van 0,003 eindigt 0,3 lager dan één run met 0,001. Wat zegt dat?",
        options={
            "nothing": "Niets: één run is nooit een resultaat",
            "seed": "Alleen iets over die ene seed",
            "right": "De learning rate doet er hier toe",
        },
        correct="right",
        why={
            "nothing": "Eén run kan geen verschil van een paar duizendsten laten zien, maar 0,3 is tien keer de ruis tussen seeds. Zie [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
            "seed": "Met een andere seed schuift elke run een paar honderdsten op, geen 0,3. Zie [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl).",
            "right": "Een verschil van tien keer de ruis tussen seeds komt niet van de seed. Voor een klein verschil heb je meerdere seeds nodig.",
        },
    )
    quick_noise
    return (quick_noise,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een nieuwe methode is maar zo overtuigend als de baseline die hij verslaat. In onderzoek is de baseline de methode die je wilt verbeteren, en dan in zijn beste versie, niet in een zwakke. Tune je je eigen methode zorgvuldig en draai je de baseline met de instellingen die hij toevallig had, dan wint je methode deels door het tunen, niet doordat hij beter is.

    Dat gebeurt in gepubliceerd werk. In 2018 tuneden Gábor Melis, Chris Dyer en Phil Blunsom een ouder, eenvoudiger soort language model en nieuwere modellen op dezelfde manier, met dezelfde grote automatische zoektocht naar hun instellingen. Daarna versloeg het oudere model de nieuwere ([Melis et al., 2018](https://arxiv.org/abs/1707.05589)).

    **Een voorbeeld uit het dagelijks leven.** Twee hardlopers doen een wedstrijd: de een op hardloopschoenen, na een warming-up, de ander op laarzen, net uit bed. De winnaar zegt weinig over wie harder loopt.
    """)
    return


@app.cell(hide_code=True)
def predict_gap():
    guess_gap = L.guess(
        "tuned-gap",
        question="In Eén ding tegelijk veranderen eindigde de GPT met 4 layers 0,07 lager dan die met 2, allebei met een learning rate van 0,001. Nu probeer je negen learning rates voor de GPT met 4 layers en neem je de beste, en laat je de GPT met 2 layers op 0,001. Hoe ver liggen ze uit elkaar?",
        options={
            "same": "Weer ongeveer 0,07",
            "right": "Veel verder uit elkaar",
            "closer": "Dichter bij elkaar",
        },
        reveal="**Veel verder uit elkaar: 0,62.** De GPT met 4 layers eindigt bij zijn beste rate, 0,03, gemiddeld over seeds 0 tot en met 2 op 1,777; de GPT met 2 layers bij 0,001 op 2,400. Bijna alles daarvan is tunen: geef de GPT met 2 layers dezelfde negen rates, en zijn beste, ook bij 0,03, eindigt op 1,779. Dan is het verschil 0,002, kleiner dan de spreiding tussen seeds.",
    )
    guess_gap
    return (guess_gap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **eerlijke vergelijking** geeft je methode en de baseline dezelfde kansen:

        - dezelfde data, op dezelfde manier gesplitst in training set en validation set;
        - dezelfde compute, de hoeveelheid rekenwerk die het trainen kost: in de praktijk evenveel epochs, of evenveel tijd op dezelfde computer;
        - dezelfde seeds;
        - dezelfde **tuning effort**: evenveel instellingen geprobeerd, op dezelfde manier gekozen, op de validation set.

        De tuning effort zie je het makkelijkst over het hoofd. De beste van negen pogingen verslaat één poging, zelfs bij twee methoden die even goed zijn: de beste poging vindt een betere instelling, en heeft ook wat geluk met de seeds. Dus lijkt de methode die je meer tunede beter dan hij is. Rapporteer hoe je elke methode tunede, de baseline ook.

        Het plaatje: de runs van de course met de kleine GPT met 2 layers (gestippeld) en met 4 layers, bij negen learning rates. Elke lijn gaat door het gemiddelde van seeds 0 tot en met 2, de kleine rondjes zijn de losse runs, en een ring markeert de beste rate van elk model. Bij 0,001 ligt de GPT met 4 layers duidelijk lager. Bij hun beste rates eindigen ze op dezelfde plek. En de GPT met 4 layers deed er per run ongeveer twee keer zo lang over: evenveel epochs gaf hem twee keer zoveel tijd. In dezelfde tijd had de GPT met 2 layers langer kunnen trainen. Zeg welke van de twee je gelijk hield.
        """),
        mo.Html(C.tuning_picture({"title": "Negen learning rates per model", "x": "learning rate", "y": "validation loss", 2: "2 layers", 4: "4 layers", "lang": L.lang,
                                  "alt": "Validation loss tegen de learning rate, van 0,001 tot 0,1, voor de GPT met 2 en met 4 layers. Beide dalen van ongeveer 2,4 en 2,33 bij 0,001 naar ongeveer 1,78 bij 0,03, en stijgen weer bij 0,05 en 0,1. Bij 0,03 liggen de twee op elkaar."})),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **eerlijke vergelijking** | een vergelijking waarin de baseline dezelfde data, compute, seeds en tuning effort krijgt als je eigen methode |
        | **tuning effort** | hoeveel instellingen je voor een methode probeert voordat je de beste neemt |
        """),
        L.watch_out("Tune op de validation set, nooit op de test set. Een instelling die je kiest omdat hij op de test set het best scoort, maakt die testscore te mooi."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vergelijk je je methode eerlijk met een baseline:

    1. Leg vast wat ze delen: de data en de splitsing ervan, het aantal epochs, de seeds.
    2. Geef ze dezelfde tuning effort: dezelfde learning rates, of evenveel instellingen, gekozen op de validation set.
    3. Neem elke methode bij zijn eigen beste instelling, op het gemiddelde over de seeds.
    4. Vergelijk die twee, en rapporteer hoe je elke methode tunede.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De runs van de course bij drie van de negen learning rates, elk met seeds 0 tot en met 2. Het commentaar zegt welke stap elk deel is. Hij is nog niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_fair():
    import numpy as np

    # 1. dezelfde data, epochs en seeds voor allebei
    _two = {0.001: [2.3902, 2.3924, 2.4174],
            0.003: [2.0827, 2.0987, 2.085],
            0.03: [1.7859, 1.7773, 1.7738]}
    _four = {0.001: [2.3335, 2.3245, 2.3374],
             0.003: [2.0117, 1.9931, 2.0099],
             0.03: [1.7745, 1.7776, 1.7776]}
    # 2. dezelfde learning rates voor allebei; 3. van elke het gemiddelde over de seeds
    for _lr in _two:
        print("rate", _lr, " 2 layers:", round(np.mean(_two[_lr]), 3), " 4 layers:", round(np.mean(_four[_lr]), 3))
    # 3-4. elk bij zijn eigen beste rate, en het verschil
    _gap = np.mean(_four[0.03]) - np.mean(_two[0.001])  # maak af: de GPT met 2 layers bij zijn eigen beste rate, niet bij 0.001
    print("4 layers min 2 layers:", round(_gap, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de GPT met 2 layers bij 0,001 is het verschil −0,623: de GPT met 4 layers lijkt veel beter. Bij zijn eigen beste rate, 0,03, is het verschil −0,002, kleiner dan de spreiding tussen de seeds van één rate. Bijna die hele 0,62 kwam van het tunen.

    Dat geldt voor deze kleine opzet: 6 epochs op het eerste kwart van Tiny Shakespeare, 3 seeds. Met langer trainen neemt de GPT met 4 layers misschien wel een voorsprong. Een eerlijke vergelijking zegt welke methode beter is onder de omstandigheden die je getest hebt, niet in het algemeen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De beste instelling van een methode

        Maak `best_setting(results)` af: stap 3 van het recept als functie. `results` is een dictionary met de learning rates als keys en de validation losses van de seeds van elke rate als waarden, zoals `{0.001: [2.3902, 2.3924, 2.4174], 0.003: [...]}`. De functie geeft de learning rate met het laagste gemiddelde over zijn seeds terug, en dat gemiddelde, als `[best_lr, best_mean]`. Vervang de twee `...`. De check roept `best_setting(two_layers)` aan, met alle negen rates van de GPT met 2 layers van de course, en probeert andere resultaten.
        """),
        L.task("Laat `best_setting(results)` de learning rate met de laagste gemiddelde validation loss over zijn seeds teruggeven, en dat gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-setting", lines=10)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-setting", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-setting",
        [
            "Stap 3 van het recept: voor elke learning rate het gemiddelde van zijn losses, zoals in het uitgewerkte voorbeeld. Een loss is beter als hij lager is, dus houd je een rate als zijn gemiddelde lager is dan het beste tot nu toe.",
            "`mean = np.mean(results[___])` en `if best_mean is None or mean ___ best_mean:`.",
        ],
        why="```python\nimport numpy as np\n\n\ndef best_setting(results):\n    best_lr = None\n    best_mean = None\n    for lr in results:\n        mean = np.mean(results[lr])\n        if best_mean is None or mean < best_mean:\n            best_lr = lr\n            best_mean = mean\n    return [best_lr, best_mean]\n```\n\nHet gemiddelde over de seeds voorkomt dat één gelukkige seed de rate kiest. De eerste rate is de beste tot nu toe, omdat er nog niets is om mee te vergelijken; elke volgende rate neemt zijn plaats in als zijn gemiddelde lager is.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Drie manieren om te vergelijken

    Jouw `best_setting` kiest de beste rate van elk model uit de negen van de course; zolang oefening 1 nog niet goed is, doet de `best_setting` van de course het. De tabel zet drie vergelijkingen naast elkaar.
    """)
    return


@app.cell(hide_code=True)
def demo_compare(box_best, result_best):
    _best, _yours = C.best_setting_of(box_best.value if result_best.state == "correct" else None)
    _lr2, _m2 = _best({_k: list(_v) for _k, _v in C.GRID[2].items()})
    _lr4, _m4 = _best({_k: list(_v) for _k, _v in C.GRID[4].items()})
    _d2, _d4 = C.mean_of(2, C.DEFAULT_LR), C.mean_of(4, C.DEFAULT_LR)
    _n, _s, _r = (lambda x: C.number(x, 3, L.lang)), (lambda x: C.signed(x, 3, L.lang)), (lambda x: C.rate(x, L.lang))
    _who = "Jouw `best_setting` koos" if _yours else "De `best_setting` van de course koos"
    mo.md(f"""
    | vergelijking | 4 layers | 2 layers | verschil |
    |---|---|---|---|
    | allebei bij {_r(C.DEFAULT_LR)} | {_n(_d4)} | {_n(_d2)} | {_s(_d4 - _d2)} |
    | alleen 4 layers getuned | {_n(_m4)} bij {_r(_lr4)} | {_n(_d2)} bij {_r(C.DEFAULT_LR)} | {_s(_m4 - _d2)} |
    | allebei op dezelfde manier getuned | {_n(_m4)} bij {_r(_lr4)} | {_n(_m2)} bij {_r(_lr2)} | {_s(_m4 - _m2)} |

    {_who} de rates. De eerste rij behandelt beide modellen gelijk, maar bij een rate die bij geen van beide past. Alleen de laatste rij is eerlijk: beide modellen kregen dezelfde negen pogingen, en elk staat bij zijn eigen beste rate.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Oefening 2. Welke vergelijking is eerlijk?
    """)
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair",
        question="Je geeft je GPT 4 layers in plaats van 2 en wilt laten zien dat dat helpt. Welke vergelijking geeft de GPT met 2 layers een eerlijke kans?",
        options={
            "yours": "De GPT met 4 layers getuned over negen rates, die met 2 layers op de 0,001 die hij altijd had",
            "right": "Allebei getuned over dezelfde negen rates, met dezelfde seeds en epochs",
            "longer": "Allebei bij 0,001, de GPT met 4 layers met twee keer zoveel epochs omdat hij groter is",
        },
        correct="right",
        why={
            "yours": "De GPT met 4 layers krijgt negen pogingen en die met 2 layers één: het grootste deel van het verschil is dan tunen. Zie het plaatje bij *Het idee*.",
            "longer": "Twee keer zoveel epochs is meer compute: de grotere GPT wint dan deels doordat hij langer trainde. En 0,001 past bij geen van beide.",
            "right": "Dezelfde tuning effort, dezelfde seeds, dezelfde epochs: wat er van het verschil overblijft, komt van de layers. Hier is dat bijna niets.",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_review():
    mo.md(r"""
    ### Oefening 3. Een claim lezen
    """)
    return


@app.cell(hide_code=True)
def explain_review():
    explain_review_cell = L.explain(
        "review-claim",
        question="Een paper zegt: ‘Ons model haalt een validation loss van 1,80, ver onder de 2,40 van de baseline. We probeerden 30 learning rates voor ons model en trainden het 20 epochs. Het getal van de baseline komt uit zijn oorspronkelijke paper, die hem 6 epochs trainde.’ Wat maakt deze vergelijking oneerlijk, en wat zou je de auteurs vragen te doen?",
        model_answer="**Oneerlijk:** de twee kregen niet dezelfde kansen. Hun model kreeg 30 learning rates, de baseline wat zijn eigen paper probeerde; hun model trainde 20 epochs, de baseline 6, dus het kreeg meer compute; en het getal van de baseline komt uit een andere paper, met andere code en misschien andere data of een andere splitsing, dus de twee getallen zijn niet eens op dezelfde manier gemeten. Elk getal kan ook één run zijn. En hun 1,80 is de beste van 30 pogingen op de validation set zelf, dus een beetje te mooi. **Vraag:** draai de baseline opnieuw in hun eigen opzet, op dezelfde data en splitsing, met dezelfde 20 epochs en dezelfde seeds, getuned over dezelfde 30 learning rates op de validation set. Rapporteer dan allebei als gemiddelde over meerdere seeds, met de spreiding, meet de gekozen instellingen op een test set, en zeg hoe elk getuned is.",
        points=[
            "je noemt de ongelijke tuning effort: 30 learning rates tegen die van de oorspronkelijke paper",
            "je noemt de ongelijke compute, 20 tegen 6 epochs, en dat het getal van de baseline uit een andere opzet komt",
            "je vraagt de baseline opnieuw te draaien in dezelfde opzet, met dezelfde tuning effort en seeds, en beide op dezelfde manier te rapporteren",
        ],
    )
    explain_review_cell
    return (explain_review_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een baseline kwam voor het eerst in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl); hier is het de methode die je wilt verslaan, en hij krijgt dezelfde zorg als je eigen methode. De runs breiden die van [Eén ding tegelijk veranderen](/course/open/res-hypothesis-control?lang=nl) uit, en gemiddelden over seeds vergelijken komt uit [Eén run is geen resultaat: seeds en variantie](/course/open/res-seeds-variance?lang=nl). Een verschil van 0,002 tussen twee getunede modellen: is dat een verschil of ruis? De volgende les, *Is het verschil echt?*, toetst het. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, explain_review_cell, result_best):
    L.wrap_up(result_best, choice_fair, explain_review_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
