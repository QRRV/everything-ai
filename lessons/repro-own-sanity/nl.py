# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-sanity. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-sanity", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 3: bouw het](/course/open/repro-own-build?lang=nl) en [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_same():
    quick_same = L.choice(
        "quick-same",
        question="Twee cellen roepen elk `make_model(\"claim\", 0)` aan uit je experiment van Je reproductie 3 en meten de eerste loss op `one_batch()`. Allebei krijgen ze 2,335. Waarom hetzelfde getal?",
        options={
            "right": "`make_model` begint met `torch.manual_seed(0)`, dus wat daarna willekeurig is, komt hetzelfde uit",
            "only-k": "Vóór elke training hangt de eerste loss alleen van K af, en K is 10 in beide cellen",
            "kept": "Python onthoudt het model van de eerste aanroep, en de tweede aanroep krijgt datzelfde model terug",
        },
        correct="right",
        why={
            "only-k": "Hij ligt dicht bij ln K, 2,30, maar niet precies: de willekeurige startwaarden, en dropout, maken er 2,335 van. Seed 1 geeft 2,322. Zie [Je reproductie 3](/course/open/repro-own-build?lang=nl).",
            "kept": "Elke aanroep bouwt een nieuw model. Het is hetzelfde model omdat zijn willekeurige startwaarden uit dezelfde seed komen. Zie [Je reproductie 3](/course/open/repro-own-build?lang=nl).",
            "right": "Daar is de seed voor: de startwaarden, en de dropout daarna, komen elke keer uit dezelfde willekeurige getallen.",
        },
    )
    quick_same
    return (quick_same,)


@app.cell(hide_code=True)
def quick_check_reshape():
    quick_reshape = L.choice(
        "quick-reshape",
        question="In Een herbruikbare training loop is de regel van de loss `F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))`. De logits van een model van tekst hebben de vorm (8, 16, 50): 8 teksten van 16 plekken, en 50 tokens. Welke vorm gaat de loss in?",
        options={
            "rows": "(8, 800): één rij voor elke tekst, met al zijn plekken naast elkaar",
            "right": "(128, 50): één rij van 50 voor elk van de 8 × 16 plekken",
            "same": "(8, 16, 50): `reshape(-1, ...)` laat de vorm zoals hij is",
        },
        correct="right",
        why={
            "rows": "Dat zou `reshape(len(logits), -1)` zijn. Hier staat de -1 eerst: zoveel rijen als nodig, elk van `logits.shape[-1]` = 50 getallen. Zie [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).",
            "same": "`reshape` houdt de getallen, niet de vorm: de -1 betekent zoveel rijen als nodig, dus 8 × 16 = 128 rijen van 50. Zie [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl).",
            "right": "Elke plek van elke tekst krijgt een eigen rij van 50 logits, en `yb.reshape(-1)` geeft zijn 128 targets in dezelfde volgorde.",
        },
    )
    quick_reshape
    return (quick_reshape,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Het hoofdresultaat draait je experiment urenlang, over elke seed van je plan. Een bug in het model of de training kost je die allemaal, en je ziet hem pas aan het eind, als een resultaat dat een beetje afwijkt, of helemaal niet. De sanity checks van je kit kosten een minuut en kijken naar de drie dingen die zo'n bug het eerst bederft: wat het model teruggeeft, wat het weet voordat het iets geleerd heeft, en of het überhaupt kan leren. In Begeleide reproductie 3 waren ze geschreven voor het grokking-model; hier werken ze voor elk model, en richt je ze op je eigen experiment.

        **Een voorbeeld uit het dagelijks leven.** Voor een lange autorit loop je even om de auto heen: banden, olie, lichten. Vijf minuten zeggen niet of de reis goed gaat, maar ze vinden wat je op de snelweg zou laten stranden.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_trained():
    guess_trained = L.guess(
        "trained",
        question="Een language model dat al eerder getraind is, met een vocabulary van 151.936 tokens, heeft op je batch een eerste loss van 2,9: ver onder ln 151.936, ongeveer 11,9. Wat zegt de sanity check over die eerste loss?",
        options={
            "fails": "Hij zakt: een eerste loss van 2,9 ligt ver van ln K, dus er is iets mis",
            "leak": "Hij waarschuwt dat het antwoord lekt in wat het model ziet",
            "right": "Niets: hij vergelijkt alleen een model vanaf nul met ln K",
        },
        reveal="**Niets: hij vergelijkt alleen een model vanaf nul met ln K.** ln K is de loss van een model dat nog niets weet. Een model dat al eerder getraind is, weet meer dan toeval, en daar was zijn training voor. Voor zo'n model tellen de vorm van de logits en één batch leren nog wel.",
    )
    guess_trained
    return (guess_trained,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Drie checks, elk model.** De checks van Begeleide reproductie 3 werken voor elk model waarvan de logits voor een batch `x` de vorm van zijn labels `y` hebben en dan K getallen. Een classifier geeft één rij van K voor elk voorbeeld; een model van tekst geeft één rij van K voor elke plek, met een label voor elke plek. Zet je ze met `reshape(-1, K)` onder elkaar in één lange lijst rijen, dan krijgen ze hun loss op dezelfde manier.

    **De vorm.** De logits moeten de vorm van `y` hebben en dan K. Is dat niet zo, dan hebben de andere twee checks geen zin, dus stopt de check daar.

    **De eerste loss.** Vóór elke training geeft een model vanaf nul elk antwoord ongeveer dezelfde kans, dus is zijn loss ongeveer ln K. Veel hoger betekent te grote startwaarden; veel lager betekent dat het antwoord lekt in wat het model ziet. Een model dat al eerder getraind is, geladen uit een checkpoint, begint lager, en zo hoort het ook: daarvoor zegt deze check niets.

    **Eén batch uit het hoofd.** 200 stappen AdamW bij een learning rate van `1e-3` op een paar tientallen voorbeelden moeten de loss onder 0,05 brengen. Een model dat zo weinig niet uit het hoofd kan leren, leert de data ook niet: misschien is een deel bevroren, losgekoppeld van de loss, of hangt zijn output niet van zijn weights af. Weight decay gaat eruit, net als in Begeleide reproductie 3: die hoort bij de training, en de check traint op zijn eigen manier. Dropout blijft aan: in `"claim"` hoort hij bij het model dat het plan test, en een netwerk dat groot genoeg is, leert één batch ook met dropout. Eén batch is niet uit het hoofd te leren als dezelfde input er met verschillende labels in zit, zoals bij een model van tekst dat alleen het token ervoor ziet: kies een batch waarin elke input zijn label onderscheidt.

    **De delen van je experiment.** De check haalt ze uit je experiment van Je reproductie 3: `make_model(setting, seed)`, `one_batch()`, `K` en `FROM_SCRATCH`, in beide settings.

    ### Nieuwe woorden en tekens

    | Teken | Wat het betekent |
    |---|---|
    | ln K | de loss van een model dat elk van K antwoorden dezelfde kans geeft, 1/K |

    Het plaatje: de eerste loss van het experiment van de course voor de dropout-paper, vóór elke training, op zijn batch van 32 afbeeldingen, in setting `"claim"` met zijn dropout aan, zoals de check hem meet, met elke startwaarde 1 tot 3 keer zo groot als PyTorch hem maakt. De band is ln 10 plus of min 0,5, waar de check slaagt. Schuif de slider.
    """)
    return


@app.cell(hide_code=True)
def explore_first_cell():
    _labels = {"title": "Eerste loss, per grootte van de startwaarden", "x": "startwaarden, keer zo groot", "y": "eerste loss",
               "band": "ln 10 ± 0,5",
               "alt": "De eerste loss van het model van de course voor de dropout-paper tegen hoeveel keer zo groot zijn startwaarden zijn, van 1 tot 3. Bij 1 is hij 2,34, binnen een band van 1,80 tot 2,80 rond ln 10; bij 1,75 is hij 2,59; bij 2 is hij 2,81, net boven de band; bij 3 is hij 5,27."}

    def _frame(s):
        _f = C.first_loss(s)
        _say = "dus de check slaagt" if C.first_ok(s) else "dus de check zakt"
        return C.first_picture(s, _labels), f"Startwaarden {C.number(s, 2, L.lang)} keer zo groot: eerste loss {C.number(_f, 2, L.lang)} tegen ln 10 = {C.number(math.log(10), 2, L.lang)}, {_say}."

    explore_first = L.explore("scale", "schaal", _frame, start=1, low=1, high=3, step=0.25, role="",
                              task="Vanaf welke grootte zakt de check? En waarom blijft de loss eerst bij ln 10, en schiet hij dan omhoog?")
    explore_first
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Tot 1,75 keer zo groot blijft de eerste loss binnen een half punt van ln 10; bij 2 keer is hij 2,81, net te veel, en zakt de check; bij 3 keer is hij 5,27. De loss blijft eerst bij ln 10 en schiet dan omhoog, omdat de logits veel sneller groeien dan de weights: elk van de drie layers vermenigvuldigt opnieuw met de schaal, dus 3 keer zo grote weights geven meer dan 20 keer zo grote logits. Zolang ze klein zijn, krijgt elke soort nog ongeveer 1/10; zijn ze groot, dan is het model zeker, en meestal fout. Dropout maakt het erger: die staat aan, zoals bij de training, en spreidt de logits bij de start nog verder. Zonder dropout, zoals in `"other"`, is 3 keer nog maar 2,95. Te grote startwaarden komen uit een eigen initialisatie, of een schaal die je verkeerd overnam uit een paper. Een eerste loss die veel te laag is, is de andere waarschuwing: een model dat het antwoord weet voordat het iets geleerd heeft, ziet het ergens.
        """),
        L.watch_out("Deze checks kijken naar het model en één batch, met een eigen training: niet naar de training in je `run`, en niet naar wat je meet. Een bug in die training, een test set die in de training data lekt, een verkeerde metric, of dropout die aan blijft tijdens de test komen door alle drie. Bewaak die in `run` en in je plan."),
        mo.md(r"""
        ### Recept

        Zo haal je een model door de sanity checks:

        1. Neem één kleine batch, een paar tientallen voorbeelden, en het model vóór elke training.
        2. Controleer dat zijn logits de vorm van de labels hebben en dan K.
        3. Controleer voor een model vanaf nul dat de eerste loss ongeveer ln K is.
        4. Leer de batch uit het hoofd: 200 stappen AdamW bij `1e-3`, en de loss moet onder 0,05 komen.
        5. Doe het in beide settings, vóór elke lange run.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 met de hand: logits die elk antwoord dezelfde score geven, voor 4 afbeeldingen en 10 soorten kleding, en voor 2 teksten van 3 plekken. `reshape(-1, 10)` is die van [Een herbruikbare training loop](/course/open/nn-trainer?lang=nl): rijen van 10, zoveel als er nodig zijn. Voer de cel uit.
    """)
    return


@app.cell
def example_ln():
    # 4 afbeeldingen, gelijke scores voor 10 soorten
    _logits = torch.zeros(4, 10)
    _labels = torch.tensor([3, 0, 7, 9])
    _loss = F.cross_entropy(_logits, _labels).item()
    _ln = round(math.log(10), 4)
    print("afbeeldingen:", round(_loss, 4), "ln 10:", _ln)
    # 2 teksten van 3 plekken: een rij van 10 per plek
    _text = torch.zeros(2, 3, 10)
    _next = torch.tensor([[1, 4, 2], [0, 0, 9]])
    _rows = _text.reshape(-1, 10)   # 6 rijen van 10
    _labels = _next.reshape(-1)     # en 6 labels
    _loss = F.cross_entropy(_rows, _labels).item()
    print("tekst:", tuple(_rows.shape), round(_loss, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Gelijke scores geven elke soort 1/10, dus de loss is -ln(1/10) = ln 10 = 2,3026, wat de labels ook zijn. `reshape(-1, 10)` maakt van de 2 keer 3 plekken 6 rijen van 10, en `reshape(-1)` van de labels 6; dan reken je de loss van een model van tekst net zo uit als die van een classifier, en is hij weer ln 10.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sanity():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De sanity checks, voor elk model

        Maak `sanity(model, x, y, k, steps, from_scratch)` af, na het commentaar. Hij geeft een rapport terug: `"shape"`, `True` als de logits van `model(x)` de vorm van `y` hebben en dan `k`; `"first"`, voor een model vanaf nul of de eerste loss binnen 0,5 van ln k ligt, en `None` voor een model dat al eerder getraind is; `"learns"`, of `steps` stappen AdamW bij `1e-3` de loss op de batch onder 0,05 brengen; en de eerste en de laatste loss; de eerste uit dezelfde logits als de vormcheck, met de dropout van het model aan. Klopt de vorm niet, dan geeft hij `False` en vier keer `None` terug. De check probeert een model dat werkt, een verkeerde `k`, te grote startwaarden, een bevroren model, een getraind model, een model van tekst en een met dropout.
        """),
        L.task("Maak `sanity` af: eerst de vorm, dan de eerste loss, dan de batch uit het hoofd."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sanity():
    box_sanity = L.code_box("sanity", lines=20)
    box_sanity
    return (box_sanity,)


@app.cell(hide_code=True)
def check_sanity(box_sanity):
    result_sanity = L.check("sanity", box_sanity.value)
    result_sanity
    return (result_sanity,)


@app.cell(hide_code=True)
def hints_sanity():
    L.hints(
        "sanity",
        [
            "De vorm: vergelijk `tuple(logits.shape)` met `tuple(y.shape) + (k,)`, en geef het rapport van `False` en vier keer `None` terug als ze verschillen.",
            "De eerste loss over elke rij: `F.cross_entropy(logits.reshape(-1, k), y.reshape(-1)).item()`, vóór de loop.",
            "De loop van `overfit_one_batch` van Begeleide reproductie 3, met dezelfde `reshape` in zijn loss, en de optimizer van het commentaar: AdamW bij `1e-3` met `weight_decay=0.0`, zonder de betas van de paper.",
        ],
        why="De vorm komt eerst, omdat de losses hem nodig hebben; de eerste loss vóór elke stap; en de loop bij `1e-3` test of het model überhaupt kan leren.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run():
    mo.md(r"""
    ### Oefening 2. Je experiment

    Druk op de knop: de check leest je experiment uit `work/projects/own/experiment.py`, en haalt zijn delen door de sanity checks van de course in beide settings, seed 0, met 200 stappen voor de batch. Zakt er een, dan noemt zijn melding de setting en waar je moet kijken. Repareer het in oefening 2 van [Je reproductie 3](/course/open/repro-own-build?lang=nl), waar je experiment staat, en druk opnieuw.
    """)
    return


@app.cell(hide_code=True)
def button_run():
    run_sanity = mo.ui.run_button(label="Haal mijn experiment door de sanity checks")
    run_sanity
    return (run_sanity,)


@app.cell(hide_code=True)
def check_run(run_sanity):
    result_run = L.check("sanity-run", C.experiment_text() if run_sanity.value else "")
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "sanity-run",
        [
            "Elke melding noemt de check die zakt en de setting. Kijk in `make_model` voor de vorm en de eerste loss, en, voor het leren van de batch, naar welke parameters kunnen leren.",
            "Probeer je delen eerst met een paar eigen regels: `model = make_model(\"claim\", 0)`, `x, y = one_batch()`, en dan `model(x).shape` en `y.shape`.",
        ],
        why="Dit is het experiment van de course voor de dropout-paper: zijn delen halen alle drie de checks in beide settings.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je eigen sanity check, `own/sanity.py`, op je eigen experiment, in beide settings: het rapport dat hij geeft. Hij draait als je op de knop hierboven drukt.
    """)
    return


@app.cell(hide_code=True)
def demo_sanity(run_sanity, result_sanity, result_run):
    result_sanity, result_run  # opnieuw uitvoeren als oefening 1 je check bewaart
    _out = mo.md("*Druk op de knop hierboven: dan draait deze cel je sanity check op je experiment.*")
    if run_sanity.value and result_run.state == "slow":
        _out = mo.md("*De check hierboven had te weinig tijd, dus de demo draait je sanity check niet ook nog: die zou net zo lang duren. Zorg eerst dat de check op tijd klaar is.*")
    elif run_sanity.value:
        _r = C.demo()
        if _r is None:
            _out = mo.md("*Maak eerst oefening 1 hier en oefening 2 van Je reproductie 3 af: dan staan beide bestanden in je kit, en draait deze cel ze.*")
        elif _r == "slow":
            _out = mo.md("*Je sanity check doet er voor één setting langer dan een minuut over, dus de demo stopt. Is je batch een paar tientallen voorbeelden?*")
        elif _r == "error":
            _out = mo.md("*Je `own/sanity.py` of `own/experiment.py` stopt met een fout. Heb je er een veranderd na zijn check? Check hem dan opnieuw.*")
        else:
            _out = mo.Html(C.demo_table(_r, {"True": "ja", "False": "nee", "None": "–"}, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 3. Wat de checks vangen

    Je experiment haalt de sanity checks. Zeg wat elk ervan gevangen zou hebben, en wat geen van alle kan vangen.
    """)
    return


@app.cell(hide_code=True)
def explain_why_sanity():
    explain_why_sanity_cell = L.explain(
        "why-sanity",
        question="Welke bug zou elk van de drie sanity checks in je experiment gevangen hebben? En welke bug kan geen van alle vangen?",
        model_answer="**De vorm** vangt een model met een ander aantal outputs dan K, zoals een laatste layer van 5 voor 10 soorten kleding, of labels met een andere vorm. **De eerste loss** vangt te grote startwaarden, zoals een schaal die ik verkeerd overnam, en een antwoord dat in de input lekt, waardoor hij veel te laag is. **Eén batch uit het hoofd** vangt een model dat niet kan leren: een deel dat ik bevroor, een deel dat onder `torch.no_grad()` berekend wordt, losgekoppeld van de loss, een output die niet van de weights afhangt. **Geen van alle** vangt een bug in de training van mijn `run`, want de check traint met een eigen AdamW, en ook niet wat ik na de training doe: dropout die aan blijft tijdens de test, een metric die het verkeerde telt, of testafbeeldingen die ook in de training data zitten. Die bewaak ik in `run`, en door te vergelijken met de paper.",
        points=[
            "je noemt een bug die de check van de vorm vangt",
            "je noemt een bug die de eerste loss vangt, te hoog of te laag",
            "je noemt een bug die een model één batch niet laat leren",
            "je noemt een bug die geen van de drie vangt, in de training van `run`, de test of de metric",
        ],
    )
    explain_why_sanity_cell
    return (explain_why_sanity_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De sanity checks van [Begeleide reproductie 3](/course/open/repro-guided-sanity?lang=nl), nu voor elk model; de delen van je experiment uit [Je reproductie 3](/course/open/repro-own-build?lang=nl); en de startwaarden van [Initialisatie](/course/open/nn-initialization?lang=nl). In de volgende les, *Je reproductie 5: het hoofdresultaat*, draai je je experiment op het budget van je plan, over zijn seeds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_sanity, result_run, explain_why_sanity_cell):
    L.wrap_up(result_sanity, result_run, explain_why_sanity_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
