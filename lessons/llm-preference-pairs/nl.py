# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-preference-pairs. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-preference-pairs", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl) en [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mirror():
    quick_mirror = L.choice(
        "quick-mirror",
        question="De sigmoid van een logit $z$ is 0,8. Wat is de sigmoid van $-z$?",
        options={
            "right": "0,2",
            "minus": "−0,8",
            "same": "0,8",
        },
        correct="right",
        why={
            "minus": "Een sigmoid ligt altijd tussen 0 en 1. Een minteken voor de logit draait de kans om: $\\sigma(-z) = 1 - \\sigma(z)$. Zie [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "same": "De sigmoid groeit met $z$, dus geeft $-z$ een kleinere kans: $\\sigma(-z) = 1 - \\sigma(z)$. Zie [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl).",
            "right": "$\\sigma(-z) = 1 - \\sigma(z)$. Vandaag: de kans dat a de voorkeur krijgt boven b, en die dat b de voorkeur krijgt boven a, tellen op tot 1.",
        },
    )
    quick_mirror
    return (quick_mirror,)


@app.cell(hide_code=True)
def quick_check_both():
    quick_both = L.choice(
        "quick-both",
        question="Je model van voor SFT en dat van na SFT schreven allebei een verhaal voor *Write a story with the word: kite.*, en `words_used` zegt True voor allebei. Welk verhaal noemt die check beter?",
        options={
            "after": "Dat van na SFT: dat model deed meer instructies goed",
            "longer": "Het langste: daarin was meer ruimte voor het woord",
            "right": "Geen van beide: de check zegt alleen of elk het deed",
        },
        correct="right",
        why={
            "after": "Op deze instructie deden beide wat er gevraagd werd; de check kijkt naar één antwoord tegelijk. Zie [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).",
            "longer": "`words_used` kijkt alleen of de woorden erin staan, niet naar de lengte. Zie [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl).",
            "right": "Twee keer geslaagd zegt niets over welk beter is. Vandaag vergelijk je alleen antwoorden die de check uit elkaar houdt.",
        },
    )
    quick_both
    return (quick_both,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voor SFT moet iemand op elke instructie een goed antwoord schrijven. Vaak is het makkelijker om te zeggen welk van twee antwoorden beter is. De mensen die de data van InstructGPT maakten, deden dat: ze zetten een paar antwoorden van het model op één prompt op volgorde, van best naar slechtst (Ouyang en anderen, 2022). Twee antwoorden op dezelfde prompt, met een oordeel over welk beter is, zijn een **preference pair**. Ook een programma kan dat oordeel geven, als het kan nakijken wat er gevraagd werd. En om veel van zulke oordelen om te zetten in één score per antwoord of per model, is er een model uit 1952: het **Bradley-Terry model**. Chatbot Arena rangschikt chat models ermee, uit de stemmen van mensen die twee antwoorden vergeleken (Chiang en anderen, 2024).

    **Een voorbeeld uit het dagelijks leven.** Een voetbalcompetitie. Een ploeg die een sterke ploeg verslaat, laat meer zien dan een die een zwakke verslaat. Uit wie van wie won kun je elke ploeg een sterkte geven, en uit twee sterktes zeggen hoe waarschijnlijk het is dat de ene ploeg de andere verslaat.
    """)
    return


@app.cell(hide_code=True)
def predict_odds():
    guess_odds = L.guess(
        "odds",
        question="Antwoord A krijgt 3 van de 4 keer de voorkeur boven antwoord B. Antwoord B krijgt 3 van de 4 keer de voorkeur boven antwoord C. Als elk antwoord één score heeft, zoals in het Bradley-Terry model, hoe vaak krijgt A de voorkeur boven C?",
        options={
            "same": "3 van de 4 keer",
            "right": "9 van de 10 keer",
            "always": "Altijd",
        },
        reveal="**9 van de 10 keer.** Odds van 3 tegen 1, twee keer achter elkaar, geven odds van 3 × 3 = 9 tegen 1: in het Bradley-Terry model vermenigvuldigen de odds van twee stappen. Waarom, laat het volgende deel zien.",
    )
    guess_odds
    return (guess_odds,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Een preference pair** is een instructie, een antwoord dat **gekozen** is (chosen) en een dat **afgewezen** is (rejected). Een checker geeft er alleen een waar hij twee antwoorden uit elkaar houdt: het ene slaagt en het andere zakt. Twee keer geslaagd of twee keer gezakt zegt niets, en geeft dus geen paar.

    **Het Bradley-Terry model** geeft elk antwoord, of elk model, een **score** $s$. De kans dat $a$ de voorkeur krijgt boven $b$ is de sigmoid van het verschil van hun scores:
    """)
    return


@app.cell(hide_code=True)
def formula_bt():
    mo.vstack([L.formula(r"P(a \succ b) = \sigma(s_a - s_b)"), L.formula(r"\sigma(z) = \frac{1}{1 + e^{-z}}")])
    return


@app.cell(hide_code=True)
def picture_bt():
    mo.vstack([
        mo.md("Het beeld: de kans tegen het verschil van de scores, met de drie spelers van het uitgewerkte voorbeeld hieronder: A, B en C, met de scores ln 3, 0 en −ln 3."),
        mo.Html(C.curve_picture([(math.log(3), "A − B, B − C"), (2 * math.log(3), "A − C"), (-2 * math.log(3), "C − A")],
                                {"title": "Bradley-Terry: de sigmoid van een verschil", "x": "verschil van de scores", "y": "kans op de voorkeur", "lang": L.lang,
                                 "alt": "Een S-vormige kromme van 0 naar 1 die bij een verschil van 0 door 0,5 gaat; punten bij een verschil van 1,1 (0,75), 2,2 (0,9) en −2,2 (0,1)."})),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    Alleen het verschil telt: tel bij elke score hetzelfde getal op, en elke kans blijft zoals hij was. Gelijke scores geven 0,5. En $s_a - s_b$ is de logit van de kans, de log van de odds, dus tellen de verschillen langs een keten op en vermenigvuldigen de odds: dat zijn de 9 tegen 1 van de gok.

    **De scores fitten.** Geef elke speler een score van 0, en verander de scores zo dat wat er gebeurde waarschijnlijker wordt: het gemiddelde van $-\log P(\text{winnaar} \succ \text{verliezer})$ over alle vergelijkingen is de cross-entropy van de uitkomsten, een loss als elke andere. Gradient descent maakt hem kleiner. Een reward model, in de volgende les, wordt met dezelfde loss getraind.

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **preference pair** | een instructie met een gekozen en een afgewezen antwoord | `(chosen, rejected)` |
    | **gekozen**, **afgewezen** | het antwoord dat de voorkeur krijgt (chosen), en het antwoord dat die niet krijgt (rejected) | `answers[i]`, `answers[j]` |
    | **Bradley-Terry model** | een score per speler; de sigmoid van het verschil van twee scores is de kans dat de eerste de voorkeur krijgt | `bt_prob(s_a, s_b)` |
    | $a \succ b$ | "$a$ krijgt de voorkeur boven $b$" | |
    | $s_a$ | de score van $a$ | `scores["a"]` |

    ### Recept

    Om antwoorden of modellen te rangschikken uit vergelijkingen:

    1. Verzamel vergelijkingen: twee antwoorden op dezelfde instructie, en welk beter is, van mensen of van een check.
    2. Laat de gelijke spelen weg: een check die beide laat slagen of zakken zegt niets.
    3. Geef elke speler een score, en elke vergelijking de kans $\sigma(s_a - s_b)$.
    4. Fit de scores: maak het gemiddelde van $-\log P(\text{winnaar} \succ \text{verliezer})$ zo klein als je kunt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Drie spelers met de scores $\ln 3$, 0 en $-\ln 3$, ongeveer 1,1, 0 en −1,1.
    """)
    return


@app.cell
def example_bt():
    _s = {"A": math.log(3), "B": 0.0, "C": -math.log(3)}
    for _a, _b in (("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")):
        print(_a, _b, round(1 / (1 + math.exp(-(_s[_a] - _s[_b]))), 4))
    _shifted = {_k: _v + 5 for _k, _v in _s.items()}   # 5 meer voor elke speler
    print(round(1 / (1 + math.exp(-(_shifted["A"] - _shifted["C"]))), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A krijgt 3 van de 4 keer de voorkeur boven B, en B net zo boven C: hun scores liggen $\ln 3$ uit elkaar, odds van 3 tegen 1. A en C liggen $2 \ln 3$ uit elkaar, odds van 9 tegen 1: 0,9. C boven A is de andere kant, 0,1. Met 5 meer voor elke speler is A boven C nog steeds 0,9.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prob():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De kans op een voorkeur

        Maak `bt_prob(score_a, score_b)` af: stap 3. Geef de kans terug dat de speler met `score_a` de voorkeur krijgt boven die met `score_b`. De check roept `bt_prob(1.0, 0.0)` aan, en andere.
        """),
        L.task("Laat `bt_prob` de Bradley-Terry-kans van twee scores geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prob():
    box_prob = L.code_box("bt-prob", lines=5)
    box_prob
    return (box_prob,)


@app.cell(hide_code=True)
def check_prob(box_prob):
    result_prob = L.check("bt-prob", box_prob.value)
    result_prob
    return (result_prob,)


@app.cell(hide_code=True)
def hints_prob():
    L.hints(
        "bt-prob",
        [
            "De sigmoid uit [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl), met `math.exp` voor $e$ tot een macht. Welk getal gaat erin? Geen score, maar het verschil van de twee: de eerste min de tweede.",
            "`return 1 / (1 + math.exp(-(___)))`, met het verschil op de open plek.",
        ],
        why="Alleen het verschil van de scores telt, en een hogere score voor `a` geeft een hogere kans voor `a`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Paren uit een checker

        Maak `make_pairs(answers, hits)` af: stap 1 en 2. `answers` zijn antwoorden op één instructie, `hits` zegt van elk of het de check haalde. Geef een lijst van preference pairs `(chosen, rejected)` terug: één voor elk antwoord dat slaagde en elk antwoord dat zakte. De volgorde van de paren maakt niet uit. De check roept `make_pairs(["A", "B", "C"], [True, False, False])` aan, en andere.
        """),
        L.task("Laat `make_pairs` een paar geven voor elk antwoord dat slaagde en elk antwoord dat zakte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("make-pairs", lines=8)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("make-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "make-pairs",
        [
            "Antwoord `i` kan gekozen worden als het slaagde, als `hits[i]` True is; antwoord `j` kan afgewezen worden als het zakte. Allebei moet het kloppen, voor dezelfde `i` en `j`.",
            "`if ___ and ___:`, en daaronder `pairs.append(...)` met een tuple van de twee antwoorden: het gekozen antwoord eerst.",
        ],
        why="Een paar zegt welk van twee antwoorden beter is, dus heeft het er een nodig dat slaagde en een dat zakte. Twee keer geslaagd of twee keer gezakt zouden paren zijn die niets zeggen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De loss van de scores

        Maak `bt_loss(games, scores)` af: stap 4. `games` is een lijst van paren namen `(winner, loser)`, `scores` een dict van namen naar scores. Geef het gemiddelde terug, over de partijen, van min de log van de kans dat de winnaar de voorkeur krijgt. De check roept `bt_loss([("lora", "base"), ("sft", "base"), ("base", "sft"), ("lora", "sft")], {"base": 0.0, "sft": 1.0, "lora": 1.5})` aan, en andere.
        """),
        L.task("Laat `bt_loss` de gemiddelde loss van de scores op de partijen geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("bt-loss", lines=9)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("bt-loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "bt-loss",
        [
            "Zoek voor elke partij de twee scores op in `scores`; de sigmoid van de score van de winnaar min die van de verliezer is de kans `p` dat de winnaar de voorkeur krijgt.",
            "De loss van één partij is min de log van `p`, met `math.log`: 0 als `p` 1 is, en groot als de winnaar onwaarschijnlijk was. Tel hem op bij `total`; de laatste regel neemt het gemiddelde.",
        ],
        why="De loss is klein als de scores het eens zijn met wie won. Het is de cross-entropy van de uitkomsten, dus kun je hem fitten zoals elke loss.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _h = C.arena()
    mo.vstack([
        mo.md(f"""
        ### Een arena van drie modellen

        In [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl) en [Project: LoRA, finetunen met een low-rank update](/course/open/llm-lora?lang=nl) beantwoordden drie modellen dezelfde {len(_h['base'])} testinstructies: het base model uit de capstone, het na full fine-tuning, en het met LoRA. Waar precies één van twee modellen een instructie goed deed, zegt de check welk antwoord beter is: een vergelijking. De knop fit voor elk model een score met je `bt_prob`, met gradient descent op de gemiddelde loss, {C.number(C.STEPS, 0, L.lang)} stappen met een learning rate van {C.number(C.RATE, 1, L.lang)} vanaf scores van 0, met de score van het base model vast op 0; je `bt_loss` geeft de loss ervoor en erna. De oordelen zijn die van je eigen runs van beide demo's als je die hebt, op hetzelfde model uit de capstone; anders die van de runs van de course.
        """),
    ])
    return


@app.cell(hide_code=True)
def fit_button_cell():
    fit_button = mo.ui.run_button(label="Fit de scores")
    fit_button
    return (fit_button,)


@app.cell(hide_code=True)
def demo_fit(box_loss, box_prob, fit_button, result_loss, result_prob):
    fitted = None
    _note = None
    if result_prob.state != "correct" or result_loss.state != "correct":
        _note = "*Maak eerst oefening 1 en 3 af: de demo fit met je `bt_prob`, en meet met je `bt_loss`.*"
    elif fit_button.value:
        _hits = C.arena()
        _games = C.comparisons(_hits)
        _prob = run_code(box_prob.value).names["bt_prob"]
        _loss = run_code(box_loss.value).names["bt_loss"]
        _scores = C.fit(_games, _prob)
        if not _games:
            _note = "*In deze arena verschillen geen twee modellen op een instructie: er is niets te vergelijken, en dus niets te fitten.*"
        elif _scores == "prob":
            _note = "*Je `bt_prob` stopt met een fout tijdens het fitten, of geeft geen kans tussen 0 en 1. Geeft hij voor elke twee scores een getal tussen 0 en 1?*"
        else:
            try:
                fitted = {"hits": _hits, "games": _games, "scores": _scores, "start": _loss(_games, {_m: 0.0 for _m in C.MODELS}),
                          "end": _loss(_games, _scores)}
            except Exception:
                _note = "*Je `bt_loss` stopt met een fout op de partijen van de arena. Werkt hij voor een paar honderd partijen?*"
    mo.md(_note) if _note else None
    return (fitted,)


@app.cell(hide_code=True)
def demo_result(fitted):
    _fit = fitted if fitted is not None else C.course_fit()   # no mo.stop: the course's fit stands in meanwhile
    if not _fit["games"]:   # the learner's runs differ nowhere: the note of the cell before says so
        _fit = C.course_fit(standin=True)
    _names = {"base": "base model", "sft": "full fine-tuning", "lora": "LoRA"}
    _short = {"base": "base", "sft": "SFT", "lora": "LoRA"}   # short, so the labels of the picture fit
    _h, _s, _w = _fit["hits"], _fit["scores"], C.wins(_fit["games"])
    _rows = [f"| {_names[_m]} | {sum(_h[_m])} van {len(_h[_m])} | {C.number(_s[_m], 2, L.lang)} |" for _m in C.MODELS]
    _pairs, _gap = [], 0.0
    for _a, _b in (("sft", "base"), ("lora", "base"), ("lora", "sft")):
        _n = _w[(_a, _b)] + _w[(_b, _a)]
        _seen = _w[(_a, _b)] / _n if _n else None
        _gap = max(_gap, abs(_seen - C.sigmoid(_s[_a] - _s[_b])) if _n else 0.0)
        _pairs.append(f"| {_names[_a]} tegen {_names[_b]} | {_w[(_a, _b)]} tegen {_w[(_b, _a)]} | {C.number(_seen, 2, L.lang) if _n else '–'} | {C.number(C.sigmoid(_s[_a] - _s[_b]), 2, L.lang)} |")
    _whose = "je runs" if _h["source"] == "yours" else "de runs van de course, op haar model uit de demorun van de capstone"
    _stand = "" if fitted is not None else " De fit van de course, met haar eigen functies, staat er tot je met de jouwe fit."
    mo.vstack([
        mo.md(f"Uit {_whose}: {len(_fit['games'])} vergelijkingen. De loss ging van {C.number(_fit['start'], 3, L.lang)}, de log van 2 voor scores van 0, naar {C.number(_fit['end'], 3, L.lang)}.{_stand}\n\n"
              "| model | instructies goed | score |\n|---|---|---|\n" + "\n".join(_rows)),
        mo.md("| vergelijking | gewonnen | aandeel gewonnen | Bradley-Terry |\n|---|---|---|---|\n" + "\n".join(_pairs)
              + ("\n\nHier passen de drie aandelen toevallig bij één stel scores: de odds van de twee modellen tegen het base model, gedeeld, geven de odds tussen die twee." if _gap < 0.005 else
                 f"\n\nDe kansen van de scores wijken tot {C.number(_gap, 2, L.lang)} af van de aandelen: drie scores kunnen niet bij alle drie de aandelen tegelijk passen.")),
        mo.Html(C.curve_picture([(_s[_a] - _s[_b], f"{_short[_a]} − {_short[_b]}") for _a, _b in (("sft", "base"), ("lora", "base"), ("lora", "sft"))],
                                {"title": "De gefitte scores op de kromme", "x": "verschil van de scores", "y": "kans op de voorkeur", "lang": L.lang,
                                 "alt": "De sigmoid van het verschil van twee scores, met een punt voor elke twee modellen van de arena."})),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Drie scores moeten tegelijk bij drie paren modellen passen, en de gemeten aandelen hoeven het niet met elkaar eens te zijn: elk is gemeten op een paar dozijn vergelijkingen. Als ze het niet eens zijn, komt niet elk aandeel precies uit. Wat de gefitte scores altijd vasthouden, is het aantal overwinningen van elk model: de kansen van zijn vergelijkingen tellen op tot het aantal dat het echt won.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shift():
    mo.md(r"""
    ### Oefening 4. Alleen verschillen
    """)
    return


@app.cell(hide_code=True)
def choice_shift_cell():
    _lora = C.course_fit()["scores"]["lora"]
    choice_shift = L.choice(
        "shift",
        question=f"De fit van de course hield de score van het base model op 0, en gaf LoRA een score van {C.number(_lora, 2, L.lang)}. Als hij het base model op 10 had gehouden, welke score had LoRA dan gekregen?",
        options={
            "same": f"{C.number(_lora, 2, L.lang)}: dezelfde",
            "times": f"{C.number(10 * _lora, 2, L.lang)}: tien keer zoveel",
            "right": f"{C.number(10 + _lora, 2, L.lang)}",
        },
        correct="right",
        why={
            "same": "Dan kreeg LoRA de voorkeur boven het base model met de sigmoid van een verschil van bijna −8: bijna nooit. Alleen verschillen tellen, dus schuift elke score met dezelfde 10 op.",
            "times": "Scores vermenigvuldigen verandert hun verschillen, en dus elke kans. Hetzelfde getal bij elke score optellen verandert niets: LoRA schuift ook 10 op.",
            "right": "Elke score schuift met dezelfde 10 op, dus blijft elk verschil, en elke kans, zoals het was. Daarom mag de fit één score op elk getal houden.",
        },
    )
    choice_shift
    return (choice_shift,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De sigmoid uit [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl), nu van een verschil van twee scores; en de check uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), nu als scheidsrechter tussen twee antwoorden. Je project `llm` heeft nu `preference_pairs.py`. Hierna leert een reward model met dezelfde loss zo'n score voor elk antwoord, en gebruikt DPO de preference pairs van je `make_pairs` zonder reward model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_prob, result_pairs, result_loss, choice_shift):
    L.wrap_up(result_prob, result_pairs, result_loss, choice_shift)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
