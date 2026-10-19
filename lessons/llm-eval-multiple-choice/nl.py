# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-eval-multiple-choice. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-eval-multiple-choice", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) en [Je base model evalueren](/course/open/llm-eval-base?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Na een context geeft een model de drie tokens van een woord de log-kansen −1,2, −0,7 en −2,1, elk na de tokens ervoor. Wat is de log-kans op het hele woord na die context?",
        options={
            "mean": "−1,33, hun gemiddelde",
            "right": "−4,0, hun som",
            "last": "−2,1, die van het laatste token",
        },
        correct="right",
        why={
            "mean": "De kans op het woord is het product van de drie kansen, en de log van een product is de som van de logs. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "last": "Het woord komt alleen als alle drie de tokens komen: de kansen vermenigvuldigen, de logs tellen op. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "right": "Vandaag scoort een model zo hele antwoorden: de som van de log-kansen van hun tokens.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_windows():
    quick_windows = L.choice(
        "quick-windows",
        question="Je `token_losses` scoort een tekst van 1.000 tokens in vensters van 256. Hoeveel losses geeft hij?",
        options={
            "right": "999: elk token behalve het eerste",
            "thousand": "1.000: één per token",
            "windows": "4: één per venster van 256 tokens",
        },
        correct="right",
        why={
            "thousand": "Het eerste token heeft niets ervoor, dus niets voorspelt het. Zie [Je base model evalueren](/course/open/llm-eval-base?lang=nl).",
            "windows": "Met `reduction=\"none\"` krijgt elk voorspeld token een eigen loss, niet elk venster. Zie [Je base model evalueren](/course/open/llm-eval-base?lang=nl).",
            "right": "Vandaag lees je de log-kansen van maar een paar tokens: die van een antwoord, na een vraag.",
        },
    )
    quick_windows
    return (quick_windows,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel benchmarks voor base models zijn meerkeuze: HellaSwag bijvoorbeeld geeft het begin van een tekst en vier manieren waarop hij verder kan gaan (Zellers en anderen, 2019). Een base model kan geen vraag beantwoorden, maar het kan wel zeggen hoe waarschijnlijk elk antwoord is. GPT-3 werd zo getest: het vergeleek de likelihoods van de opties na de context (Brown en anderen, 2020). In deze les test je zo je eigen model, op 500 vragen over verhalen.

    **Een voorbeeld uit het dagelijks leven.** Je hoort het begin van een zin en drie mogelijke eindes. Je hoeft het einde niet zelf te zeggen om aan te wijzen welk einde het natuurlijkst klinkt.
    """)
    return


@app.cell(hide_code=True)
def predict_sum():
    _m = C.MEASURED["yours"]
    guess_sum = L.guess(
        "sum",
        question=f"Elk van de 500 vragen heeft drie zinnen van een verhaal en vier volgende zinnen: de echte, en drie uit andere verhalen. Gokken haalt er ongeveer 125 goed. De run van de course uit de capstone, een model van {C.number(_m['params'], 0, L.lang)} parameters getraind op TinyStories, neemt de optie met de hoogste log-kans na de context. Hoeveel haalt het er goed?",
        options={
            "right": "Nauwelijks meer dan gokken",
            "half": "Ongeveer de helft van de vragen",
            "most": "De meeste",
        },
        reveal=f"**Nauwelijks meer dan gokken: {_m['right']['sum']} van de 500.** De gewone log-kans zegt vooral hoe waarschijnlijk een zin op zich is: een korte, of een met gewone woorden, wint na elke context. Deze les meet wat de context toevoegt.",
    )
    guess_sum
    return (guess_sum,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Scoor elke optie.** Zet de optie na de context, en tel de log-kansen van haar tokens op, elk gelezen op de plek waar het model dat token voorspelt: dat is de log-kans op de hele optie na de context.

    **De som is niet eerlijk.** Elk token telt er een negatief getal bij, dus een optie van veel tokens krijgt een lage som, ook als ze goed past. En een zin met gewone woorden is na elke context waarschijnlijk. GPT-3 werd op twee eerlijkere manieren gescoord:

    - **Per token.** Deel de som door het aantal tokens van de optie: een lange optie wordt niet meer gestraft voor haar lengte. Per byte, dat de paper van GPT-3 niet noemt, werkt net zo en hangt niet van de tokenizer af.
    - **Vergeleken met de optie op zich.** Scoor de optie ook na een neutraal begin, voor de verhalen hier een end-of-text, het begin van elk verhaal. Haar log-kans daar zegt hoe waarschijnlijk ze sowieso is. De optie waarvan de log-kans door de context het meest stijgt, is die waar de context naar wijst:
    """)
    return


@app.cell(hide_code=True)
def formula_rise():
    L.formula(r"\begin{aligned} \text{stijging} &= \log P(\text{optie} \mid \text{context}) \\ &\quad - \log P(\text{optie} \mid \text{begin}) \end{aligned}")
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        GPT-3 deed dat voor drie van zijn benchmarks, ARC, OpenBookQA en RACE, met "Answer: " of "A: " als neutraal begin, en deelde de kansen door elkaar; het verschil van de logs is de log van die deling. Op de 500 vragen van deze les haalt de run van de course uit de capstone er per regel zoveel goed:
        """),
        C.rules_picture({"sum": "de gewone som", "token": "per token", "byte": "per byte", "alone": "vergeleken met de optie op zich",
                         "title": "Goede antwoorden van de 500", "note": "Gokken haalt er ongeveer 125.",
                         "alt": "Goede antwoorden van de 500 voor de vier regels: de gewone som 138, per token 151, per byte 156, vergeleken met de optie op zich 220; gokken haalt er ongeveer 125."}),
        mo.md(r"""
        ### Nieuwe woorden

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **log-likelihood scoring** | een model testen op een meerkeuzevraag met de log-kans die het elke optie na de vraag geeft, en de hoogste nemen | `option_logprob(model, context, option)` |

        ### Recept

        Om een base model te testen op een meerkeuzevraag:

        1. Knip de context en elke optie in de tokens van het model.
        2. Scoor elke optie: de som van de log-kansen van haar tokens na de context.
        3. Scoor elke optie ook op zich, na een neutraal begin.
        4. Kies de optie waarvan de log-kans het meest stijgt; of, eenvoudiger, de hoogste per token.
        5. Tel de goede antwoorden over veel vragen, en vergelijk ze met gokken.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4 voor één vraag met drie opties. De log-kansen van hun tokens zijn verzonnen.
    """)
    return


@app.cell
def example_rules():
    _after_context = {"short": [-2.0, -3.5], "right": [-1.5, -1.0, -2.0, -1.5, -0.5], "odd": [-4.0, -6.0, -3.0]}
    _on_its_own = {"short": [-2.5, -3.5], "right": [-3.0, -2.5, -3.0, -2.0, -1.5], "odd": [-4.5, -6.5, -3.0]}
    for _name in _after_context:
        _with = sum(_after_context[_name])
        _alone = sum(_on_its_own[_name])
        print(_name, _with, round(_with / len(_after_context[_name]), 2), _with - _alone)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De gewone som kiest "short", −5,5 tegen −6,5: twee tokens kosten minder dan vijf. Per token wint "right", −1,3 tegen −2,75. Tegen de optie op zich wint "right" ook, en ruimer: de context verhoogt haar met 5,5, "short" maar met 0,5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logprob():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De log-kans op een optie

        Maak `option_logprob(model, context, option)` af: stap 2. `context` en `option` zijn lijsten van token-ids. De functie zet ze achter elkaar, laat het model alles behalve de laatste id lezen, en maakt van de logits log-kansen: `torch.log_softmax(logits, dim=-1)` is de log van de softmax van elke rij in één stap, de rij min haar log-sum-exp, zodat geen piepkleine kans eerst op 0 afrondt. Tel voor elk token van de optie zijn log-kans op in de rij van het token ervoor. De check probeert je functie op een klein model, met `option_logprob(model, [3, 1, 4], [1, 5])`, en meer.
        """),
        L.task("Laat `option_logprob` de log-kans op `option` na `context` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logprob():
    box_logprob = L.code_box("option-logprob", lines=12)
    box_logprob
    return (box_logprob,)


@app.cell(hide_code=True)
def check_logprob(box_logprob):
    result_logprob = L.check("option-logprob", box_logprob.value)
    result_logprob
    return (result_logprob,)


@app.cell(hide_code=True)
def hints_logprob():
    L.hints(
        "option-logprob",
        [
            "Rij `j` van `logprobs` is de voorspelling na token `j` van `ids`. `option[0]` staat op plek `len(context)`, dus het model voorspelt het in rij `len(context) - 1`.",
            "`logprobs[row, token]` is één getal in een tensor; `.item()` maakt er een Python-getal van om bij `total` op te tellen.",
        ],
        why="De kans op de hele optie is het product van de kansen van haar tokens, elk na alle tokens ervoor: de som van hun logs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De optie waar de context naar wijst

        Maak `best_option(with_context, alone)` af: stap 4. `with_context[i]` is de log-kans op optie `i` na de context, `alone[i]` hetzelfde na een neutraal begin. Geef de index terug van de optie waarvan de log-kans het meest stijgt, de eerste als twee even veel stijgen. De check roept `best_option([-12.5, -9.0, -20.1, -15.0], [-14.0, -8.5, -26.0, -16.2])` aan, en andere.
        """),
        L.task("Laat `best_option` de index geven van de optie die de context het meest verhoogt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-option", lines=8)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-option", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-option",
        [
            "De stijging van optie `i` is `with_context[i] - alone[i]`. In het voorbeeld heeft optie 1 na de context de hoogste log-kans, maar op zich is ze zelfs iets waarschijnlijker.",
            "Houd de index van de beste optie tot nu toe bij, te beginnen bij 0, en vervang hem als een optie meer stijgt.",
        ],
        why="Een optie die sowieso waarschijnlijk is, krijgt daar geen punten voor; alleen wat de context toevoegt telt.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(f"""
        ### Je model op 500 vragen

        De 500 vragen van `data/story-cloze.json` die [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl) maakte uit verhalen van Cosmopedia: drie zinnen van een verhaal, de echte volgende zin, en drie zinnen uit andere verhalen, in een geschudde volgorde. Geen model van de course trainde op deze verhalen. De knop scoort elke optie met je `option_logprob` op je model uit [Capstone: je eigen kleine LLM pretrainen](/course/open/llm-pretrain-capstone?lang=nl), of op het model van de course als je er geen hebt, na de context en na alleen een end-of-text, en telt de goede antwoorden van de vier regels; de laatste gebruikt je `best_option`. Op de computer van de course, met één thread, duurde dat ongeveer een minuut voor de run van de course uit de capstone, en ongeveer twee minuten voor het model van de course, dat vijf keer zo groot is.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def score_button_cell():
    score_button = mo.ui.run_button(label="Scoor de 500 vragen")
    score_button
    return (score_button,)


@app.cell(hide_code=True)
def demo_score(box_best, box_logprob, result_best, result_logprob, score_button):
    scored = None
    _note = None
    if not (result_logprob.state == "correct" and result_best.state == "correct"):   # no mo.stop: the table shows the course's run
        _note = "*Maak eerst oefening 1 en 2 af: de demo scoort met je `option_logprob` en kiest met je `best_option`.*"
    elif score_button.value:
        _r = C.score_all(run_code(box_logprob.value).names["option_logprob"], run_code(box_best.value).names["best_option"],
                         lambda _p: mo.output.replace(mo.md(f"**Scoren** · {_p['done']} van {_p['total']} vragen · {round(_p['seconds'])} s")))
        if _r == "run":
            _note = "*Het scoren stopt met een fout. Komen je modules in `work/projects/llm/` nog door hun checks? Verwijder een bestand dat je veranderde, en check de oefening ervan opnieuw.*"
        elif _r == "learner":
            _note = "*Je `option_logprob` of `best_option` stopt met een fout op dit model. Werkt `option_logprob` voor een context van tientallen tokens en een optie van een hele zin?*"
        else:
            scored = _r
    mo.output.replace(mo.md(_note)) if _note else mo.output.clear()   # also when scoring is done: no stale progress line
    return (scored,)


@app.cell(hide_code=True)
def demo_table(scored):
    _m = C.MEASURED["yours"]
    _r = scored or _m
    _names = {"sum": "de gewone som", "token": "per token", "byte": "per byte", "alone": "vergeleken met de optie op zich"}
    _rows = "\n".join(f"| {_names[_rule]} | {_r['right'][_rule]} | {C.number(100 * _r['right'][_rule] / _r['total'], 0, L.lang)}% |" for _rule in C.RULES)
    _whose = ("Je model" if scored["source"] == "yours" else "Het model van de course, op Cosmopedia") if scored else "De run van de course uit de capstone, op TinyStories, tot je het jouwe scoort"
    mo.md(f"{_whose}: goede antwoorden van {_r['total']} vragen; gokken haalt ongeveer 25%.\n\n| regel | goed | aandeel |\n|---|---|---|\n{_rows}")
    return


@app.cell(hide_code=True)
def text_exercise_short():
    mo.md(r"""
    ### Oefening 3. Waarom korte opties winnen
    """)
    return


@app.cell(hide_code=True)
def choice_short_cell():
    choice_short = L.choice(
        "why-short",
        question="Met de gewone som van log-kansen kiest een model vaak een korte optie, ook als die slecht bij het verhaal past. Waarom?",
        options={
            "data": "Zijn trainingsverhalen waren kort, dus het leerde korte zinnen boven lange te verkiezen",
            "right": "Elk token telt er een negatieve log-kans bij, dus minder tokens geven een hogere som",
            "right-answers": "Bij vragen zoals deze zijn de korte opties vaker het goede antwoord",
        },
        correct="right",
        why={
            "data": "Ook een model dat op lange teksten trainde, geeft een korte zin vaak een hogere som: elk extra token kan hem alleen verlagen. Per token of vergeleken met de optie op zich verdwijnt die voorkeur.",
            "right": "Een kans is hooguit 1, dus haar log is hooguit 0: elk token verlaagt de som. Delen door de lengte, of de score van de optie op zich ervan aftrekken, neemt dat weg.",
            "right-answers": "In deze 500 vragen zijn de goede zinnen even lang als de andere, gemiddeld 87 tekens. De voorkeur zit in de som, niet in de antwoorden.",
        },
    )
    choice_short
    return (choice_short,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De log-likelihood uit [Hoe goed is een language model?](/course/open/lm-nll?lang=nl), nu van hele antwoorden; het scoren uit [Je base model evalueren](/course/open/llm-eval-base?lang=nl), nu op vragen; en de vragen uit [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl), gemaakt uit verhalen waarop geen model van de course trainde, zodat het model het goede antwoord niet al eerder zag. Latere lessen leren je model zelf instructies te beantwoorden, en testen dat met checks die een programma kan nagaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_logprob, result_best, choice_short):
    L.wrap_up(result_logprob, result_best, choice_short)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
