# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-dpo-loss. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import math

    import marimo as mo
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-dpo-loss", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl) en [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_best():
    quick_best = L.choice(
        "quick-best",
        question="RLHF met een KL penalty van gewicht $\\beta$: wat is de beste policy voor één instructie?",
        options={
            "top": "Alles op het antwoord met de meeste reward, want de reward is wat telt",
            "right": "Het reference model, elk antwoord gewogen met $e^{r/\\beta}$ en gedeeld door $Z(x)$, want de reward kantelt het",
            "ref": "Het reference model zelf, want de straf houdt de policy waar hij begon",
        },
        correct="right",
        why={
            "top": "Alleen als $\\beta$ dicht bij 0 ligt. De KL penalty kost meer naarmate de policy verder van het reference model gaat. Zie [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl).",
            "ref": "Alleen als $\\beta$ heel groot is. Met een kleinere $\\beta$ is een antwoord met meer reward een beetje straf waard. Zie [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl).",
            "right": "Vandaag draai je deze formule om.",
        },
    )
    quick_best
    return (quick_best,)


@app.cell(hide_code=True)
def quick_check_shift():
    quick_shift = L.choice(
        "quick-shift",
        question="In een Bradley-Terry model heeft het ene paar antwoorden de scores 2 en 1, het andere paar 12 en 11. In welk paar is de kans groter dat het eerste antwoord de voorkeur krijgt?",
        options={
            "higher": "In het tweede: beide scores zijn daar hoger",
            "ratio": "In het eerste: 2 is twee keer 1, terwijl 12 geen twee keer 11 is",
            "right": "In geen van beide: alleen het verschil van de scores telt",
        },
        correct="right",
        why={
            "higher": "De kans is de sigmoid van het verschil van de twee scores, en dat is in beide paren 1. Zie [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl).",
            "ratio": "De sigmoid kijkt alleen naar het verschil, niet naar hoeveel keer de ene score de andere is. Zie [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl).",
            "right": "Vandaag valt hierdoor een hele term uit een loss weg.",
        },
    )
    quick_shift
    return (quick_shift,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    RLHF werkt, maar het is zwaar. PPO, de reinforcement learning van InstructGPT, houdt tijdens het trainen vier modellen bij de hand: de policy, het reference model, het reward model, en een value function, een model dat schat hoeveel reward er nog komt. En de policy moet steeds antwoorden schrijven, zodat het reward model ze een score kan geven. **DPO**, direct preference optimization (Rafailov en anderen, 2023), traint op de preference pairs zelf, met een loss, zoals SFT op voorbeelden traint: twee modellen, de policy en het bevroren reference model, en geen antwoorden om te schrijven tijdens het trainen. De truc is de formule van de beste policy van RLHF, omgedraaid.

    **Een voorbeeld uit het dagelijks leven.** Een proefpanel zegt van paren taarten welke het lekkerder vindt. Je kunt eerst een jurylid opleiden dat elke taart een cijfer geeft, en dan bakken voor dat jurylid. Of je bakt vanuit de paren zelf: meer zoals de taarten die ze kozen, minder zoals die ze afwezen, en niet te ver van je eigen recept.
    """)
    return


@app.cell(hide_code=True)
def predict_start():
    guess_start = L.guess(
        "start",
        question="DPO begint met de policy als kopie van het reference model. Een preference pair: het gekozen antwoord heeft onder allebei een log-kans van -135, het afgewezen antwoord onder allebei -139. Wat is de loss van DPO voor dit paar, met $\\beta = 0{,}1$?",
        options={
            "zero": "0: het model maakt het gekozen antwoord al waarschijnlijker",
            "right": "0,693, dat is ln 2: de policy is nog niet verschoven",
            "big": "Meer dan 1: de policy heeft nog niets geleerd",
        },
        reveal="**0,693, dat is ln 2.** DPO kijkt niet naar hoe waarschijnlijk de policy het gekozen antwoord maakt, maar naar hoeveel *waarschijnlijker* dan het reference model. Aan het begin is dat niets, voor beide antwoorden: de kans van Bradley-Terry is $\\sigma(0) = 0{,}5$, en de loss $-\\ln 0{,}5 = 0{,}693$. Elk paar begint daar.",
    )
    guess_start
    return (guess_start,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Van de beste policy terug naar een reward.** In [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl) was de beste policy voor een reward $r$ gelijk aan $\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \, e^{r(x, y) / \beta}$. Neem van beide kanten de log, en schrijf de reward vrij:
    """)
    return


@app.cell(hide_code=True)
def formula_reward():
    L.formula(r"r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)")
    return


@app.cell(hide_code=True)
def text_idea_pair():
    mo.md(r"""
    Een policy staat dus voor een reward: $\beta$ keer de log van hoeveel waarschijnlijker hij een antwoord maakt dan het reference model, plus een deel dat voor elk antwoord op $x$ hetzelfde is.

    **In Bradley-Terry.** In [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl) was de kans dat het gekozen antwoord $y_w$ de voorkeur krijgt boven het afgewezen antwoord $y_l$ de sigmoid van het verschil van hun scores. Neem deze rewards als de scores: $\beta \log Z(x)$ zit in allebei, en valt weg uit het verschil. Wat overblijft heeft geen reward model nodig, alleen de policy en het reference model. DPO zet de policy $\pi$ die leert op de plaats van $\pi^*$. De marge $m$ van een paar is hoeveel meer de policy op het reference model won met het gekozen antwoord dan met het afgewezen antwoord, alles gegeven $x$:
    """)
    return


@app.cell(hide_code=True)
def formula_margin():
    L.formula(r"m = \log \frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}")
    return


@app.cell(hide_code=True)
def text_idea_loss():
    mo.md(r"""
    De kans van Bradley-Terry dat het paar de goede kant op gaat is $\sigma(\beta m)$, en de **loss van DPO** voor het paar is min de log daarvan, zoals bij het fitten van het Bradley-Terry model; over een batch paren is de loss het gemiddelde daarvan. Voor één paar:
    """)
    return


@app.cell(hide_code=True)
def formula_loss():
    L.formula(r"L_{\text{DPO}} = -\log \sigma(\beta \, m)")
    return


@app.cell(hide_code=True)
def picture_loss():
    _labels = {"ref": "reference model", "right": "goede kant op", "wrong": "verkeerde kant op", "x": "marge m", "y": "loss",
           "title": "De loss van een paar, bij β = 0,1", "lang": "nl",
           "alt": "De loss van DPO tegen de marge bij beta 0,1: 0,693 bij een marge van 0, dalend naar 0 bij een grote marge en in een rechte lijn stijgend bij een negatieve, met de drie policies van het uitgewerkte voorbeeld erop."}
    mo.Html(C.loss_picture(0.1, [(0.0, _labels['ref']), (13.0, _labels['right']), (-5.0, _labels['wrong'])], _labels))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    **De log-kans van een antwoord** is de som van de log-kansen van zijn tokens, elk gegeven de instructie en de tokens ervoor, zoals de loss van SFT alleen de tokens van het antwoord telde. Een language model geeft logits; `torch.log_softmax(logits, dim=-1)` maakt van elke rij log-kansen. De rij op plaats $t$ is wat het model voorspelde na token $t$, dus een voorspelling voor het token op plaats $t + 1$.

    **Wat een stap doet.** Een update-stap trekt de log-kans van het gekozen antwoord omhoog en duwt die van het afgewezen antwoord omlaag, allebei met de factor $\beta \, \sigma(-\beta m)$: veel voor een paar dat de policy nog verkeerd heeft, weinig voor een paar dat hij goed heeft (Rafailov en anderen, 2023, paragraaf 4). Niets in de loss zegt welke van de twee moet bewegen: alleen de marge telt.

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **DPO** | direct preference optimization: een policy direct op preference pairs trainen, met een loss, zonder reward model en zonder reinforcement learning | `dpo_loss(...)` |
    | $y_w$, $y_l$ | het gekozen en het afgewezen antwoord van een paar: w van winner, l van loser | hun log-kansen: `policy_chosen`, `policy_rejected` |
    | $m$, de marge | hoeveel meer de policy op het reference model won met het gekozen antwoord dan met het afgewezen antwoord, in nats | `margin` |
    | `torch.log_softmax(x, dim=-1)` | log-kansen uit logits, langs de laatste dimensie | `logp = torch.log_softmax(logits, dim=-1)` |
    | `F.logsigmoid(x)` | $\log \sigma(x)$ in één stap: die blijft precies waar $\sigma(x)$ zelf op 0 zou afronden | `F.logsigmoid(beta * margin)` |

    ### Recept

    Om een model met DPO op preference pairs te finetunen:

    1. Neem het SFT-model twee keer: een policy die leert, en een bevroren reference model.
    2. Reken de log-kans van het gekozen en van het afgewezen antwoord van elk paar uit onder het reference model. Dat verandert niet, dus één keer is genoeg.
    3. Elke stap: dezelfde twee log-kansen onder de policy, de marge, en de loss $-\log \sigma(\beta m)$, als gemiddelde over een batch paren.
    4. Een update-stap op alleen de policy, en terug naar stap 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén paar, met de log-kansen uit [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl) onder het SFT-model van de course: het verhaal met *kite* en *park* is gekozen, -135,358; hetzelfde verhaal met *hill* in plaats van *park* is afgewezen, -139,358. Dat model is het reference model. Drie policies, met $\beta = 0{,}1$: het reference model zelf, één die de goede kant op is geschoven, en één die de verkeerde kant op is geschoven.
    """)
    return


@app.cell
def example_loss():
    _ref_chosen, _ref_rejected = -135.358, -139.358
    _beta = 0.1
    for _chosen, _rejected in ((-135.358, -139.358), (-133.0, -150.0), (-140.0, -139.0)):
        _margin = (_chosen - _ref_chosen) - (_rejected - _ref_rejected)
        _p = 1 / (1 + math.exp(-_beta * _margin))
        print(round(_margin, 3), round(_p, 3), round(-math.log(_p), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het reference model heeft een marge van 0, een kans van 0,5 en een loss van 0,693. De tweede policy maakte het gekozen antwoord 2,358 nats waarschijnlijker en het afgewezen antwoord 10,642 nats minder waarschijnlijk: een marge van 13, dus $\beta m = 1{,}3$: een kans van 0,786 en een loss van 0,241. De derde maakte het gekozen antwoord 4,642 nats minder waarschijnlijk en het afgewezen antwoord 0,358 nats meer: een marge van -5, en een loss van 0,974, hoger dan aan het begin. Onder die derde policy is het afgewezen antwoord waarschijnlijker dan het gekozen antwoord, -139 tegen -140, maar dat is niet de reden dat de loss hoog is: de loss kijkt alleen naar hoe ver elk antwoord van het reference model is verschoven.
    """)
    return


@app.cell(hide_code=True)
def text_example_logprob():
    mo.md(r"""
    En de log-kans van een antwoord, voor een vocabulary van 3 tokens: `ids` zijn 2, 0 en 1, het antwoord begint op plaats 1, en de logits zijn die van `ids[:-1]`, twee rijen.
    """)
    return


@app.cell
def example_logprob():
    _logits = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.5, 0.0]])   # de logits van ids[:-1]: twee plaatsen, een vocabulary van 3
    _ids = [2, 0, 1]
    _logp = torch.log_softmax(_logits, dim=-1)
    print(_logp)
    print(round(float(_logp[0, _ids[1]] + _logp[1, _ids[2]]), 4))   # plaats 1 uit rij 0, plaats 2 uit rij 1
    return


@app.cell(hide_code=True)
def text_after_logprob():
    mo.md(r"""
    Rij 0 is de voorspelling voor plaats 1, waar token 0 staat: -1,4076. Rij 1 is de voorspelling voor plaats 2, waar token 1 staat: -0,958. Samen -2,3656, de log-kans van het antwoord; de instructie, plaats 0, telt niet mee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logprob():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De log-kans van een antwoord

        Maak `answer_logprob(logits, ids, start)` af. `ids` zijn de token ID's van een instructie en zijn antwoord, het antwoord vanaf plaats `start`; `logits` is wat het model geeft voor `ids[:-1]`, één rij per plaats. Geef de som van de log-kansen van de tokens van het antwoord terug, als tensor. De check voert `answer_logprob(logits, [3, 1, 4, 1, 5, 2], 2)` uit, met logits van 5 plaatsen voor een vocabulary van 7, en meer.
        """),
        L.task("Laat `answer_logprob` de log-kansen van de tokens van het antwoord optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_logprob():
    box_logprob = L.code_box("answer-logprob", lines=9)
    box_logprob
    return (box_logprob,)


@app.cell(hide_code=True)
def check_logprob(box_logprob):
    result_logprob = L.check("answer-logprob", box_logprob.value)
    result_logprob
    return (result_logprob,)


@app.cell(hide_code=True)
def hints_logprob():
    L.hints(
        "answer-logprob",
        [
            "Elke rij van `logits` heeft een getal voor elk token van de vocabulary. Welke functie maakt daar log-kansen van, en langs welke dimensie?",
            "Rij 0 is wat het model voorspelde na `ids[0]`: een voorspelling voor `ids[1]`. Welke rij voorspelt `ids[t]`?",
        ],
        why="De log van een product is de som van de logs: de log-kansen van de tokens optellen geeft de log-kans van het hele antwoord, zonder getallen die zo klein zijn dat ze op 0 afronden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loss van DPO

        Maak `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)` af. Elk van de vier is een tensor met één log-kans per paar: van het gekozen en van het afgewezen antwoord, onder de policy en onder het reference model. Geef de gemiddelde loss van DPO over de paren terug, als tensor. De check voert `dpo_loss(torch.tensor([-134.0]), torch.tensor([-141.0]), torch.tensor([-135.358]), torch.tensor([-139.358]), 0.1)` uit, en meer.
        """),
        L.task("Laat `dpo_loss` de loss van DPO geven voor een batch paren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("dpo-loss", lines=6)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("dpo-loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "dpo-loss",
        [
            "De marge heeft vier termen: twee log-verhoudingen, elk de policy min het reference model. Welke van de twee gaat van welke af?",
            "De loss van een paar is min de log van een kans, zoals bij het fitten van het Bradley-Terry model. Welke kans, en waar hoort `beta`?",
        ],
        why="De marge vergelijkt de policy met het reference model; de sigmoid maakt van $\\beta m$ de kans dat het paar de goede kant op gaat; min de log daarvan is klein als die kans dicht bij 1 ligt.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Twintig stappen DPO

    Het SFT-model van de course uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), als de policy en als het reference model, en één preference pair voor *Write a story with the words: kite and park.*: het verhaal met beide woorden is gekozen, hetzelfde verhaal met *hill* in plaats van *park* is afgewezen. Voor de eerste stap de twee log-kansen van het reference model met je `answer_logprob`; elke stap de twee van de policy, de loss met je `dpo_loss`, en een stap AdamW op de hele policy. Kies $\\beta$: elke waarde traint opnieuw, in ongeveer {C.number(C.MEASURED['seconds'], 0, L.lang)} seconden op de CPU van de course met één thread.
    """)
    return


@app.cell(hide_code=True)
def beta_cell():
    beta_pick = mo.ui.slider(start=0, stop=len(C.BETAS) - 1, value=C.BETAS.index(0.1), label="β", show_value=False)
    beta_pick
    return (beta_pick,)


@app.cell(hide_code=True)
def demo_dpo(beta_pick, box_logprob, result_logprob, box_loss, result_loss):
    _beta = C.BETAS[beta_pick.value]
    _mine = {"logprob": run_code(box_logprob.value).names.get("answer_logprob") if result_logprob.state == "correct" else None,
             "loss": run_code(box_loss.value).names.get("dpo_loss") if result_loss.state == "correct" else None}
    _r, _failed = C.demo(_beta, _mine["logprob"], _mine["loss"])
    _names = {"logprob": "`answer_logprob`", "loss": "`dpo_loss`"}
    _notes = [f"*De {_names[_k]} van de course staat er tot de jouwe slaagt voor zijn check.*" for _k in _names if _mine[_k] is None]
    _notes += [f"*Je {_names[_k]} stopt met een fout op het model, of geeft geen tensor om mee te trainen: die van de course staat er.*" for _k in _failed]
    _b = C.number(_beta, 2 if _beta < 0.1 else 1, L.lang)
    if _r == "run":
        _out = mo.md("*Het SFT-model van de course kan hier niet gemaakt worden: een module van je project die het nodig heeft stopt met een fout. De notitie bovenaan de les zegt welke.*")
    else:
        _h = _r["history"]
        _pct = lambda _v: C.percent(_v, L.lang)
        _moved = lambda _v: ("+" if _v >= 0 else "") + C.number(_v, 1, L.lang)
        _out = mo.vstack([
            mo.md(f"**β = {_b}** · loss na {C.STEPS} stappen {C.number(_h[-1][0], 3, L.lang)} · gekozen antwoord {_moved(_h[-1][1])} nats, afgewezen antwoord {_moved(_h[-1][2])} nats, vergeleken met het reference model"),
            mo.Html(C.curves_picture(_h, {"loss_title": "Loss", "moved_title": "Hoe ver elk antwoord ging", "x": "stap", "loss": "loss",
                                          "moved": "nats", "chosen": "gekozen", "rejected": "afgewezen", "lang": L.lang,
                                          "alt": f"Twee grafieken over {C.STEPS} stappen bij beta {_b}: de loss daalt van 0,693 naar {C.number(_h[-1][0], 3, L.lang)}; het gekozen antwoord verschuift {_moved(_h[-1][1])} nats en het afgewezen antwoord {_moved(_h[-1][2])} nats ten opzichte van het reference model."})),
            mo.md(f"""
            | Na *{C.CONTEXT}* | SFT-model | na {C.STEPS} stappen |
            |---|---|---|
            | *park* | {_pct(_r['before']['park'])} | {_pct(_r['after']['park'])} |
            | *hill* | {_pct(_r['before']['hill'])} | {_pct(_r['after']['hill'])} |
            """),
        ])
    mo.vstack([_out] + [mo.md(_n) for _n in _notes])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _m = C.MEASURED["betas"]
    _gain = [_v["chosen"] for _v in _m.values()]
    _loss = [-_v["rejected"] for _v in _m.values()]
    _park = [_v["after"]["park"] for _v in _m.values()]
    mo.md(f"""
    Bij stap 0 is de loss 0,693 voor elke $\\beta$: de policy is nog het reference model. Daarna verliest het afgewezen antwoord veel meer dan het gekozen antwoord wint: na {C.STEPS} stappen heeft het gekozen antwoord tussen {C.number(min(_gain), 1, L.lang)} en {C.number(max(_gain), 1, L.lang)} nats gewonnen, het afgewezen antwoord tussen {C.number(min(_loss), 1, L.lang)} en {C.number(max(_loss), 1, L.lang)} verloren. DPO vraagt alleen om de marge, en het afgewezen antwoord omlaag duwen helpt de loss evenveel als het gekozen antwoord omhoog trekken.

    De twee antwoorden verschillen in één woord, en daar veranderde de policy het meest. Na *{C.CONTEXT}* gaf het SFT-model *park* een kans van {C.percent(C.MEASURED['before']['park'], L.lang)}; na {C.STEPS} stappen geeft het *park* tussen {C.percent(min(_park), L.lang)} en {C.percent(max(_park), L.lang)}, en *hill* bijna niets. Hoe kleiner $\\beta$, hoe verder het afgewezen antwoord zakt en hoe meer *park* wint: $\\beta = 0{{,}}05$ laat de policy het verst gaan, $\\beta = 1$ houdt hem het dichtst bij het reference model, zoals de KL penalty in RLHF deed. En een lage loss betekent niet dat de policy ver ging: bij $\\beta = 1$ eindigt de loss bij 0, terwijl de policy het minst bewoog. Eén paar en {C.STEPS} stappen is geen echte training; in het project train je op veel paren.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_z():
    mo.md(r"""
    ### Oefening 3. Waar Z bleef
    """)
    return


@app.cell(hide_code=True)
def choice_z_cell():
    choice_z = L.choice(
        "no-z",
        question="De beste policy van RLHF heeft een $Z(x)$, de loss van DPO niet. Waarom mag die weg?",
        options={
            "right": "Beide antwoorden delen de instructie $x$, dus $\\beta \\log Z(x)$ valt weg uit het verschil van hun rewards",
            "one": "Voor een language model is $Z(x)$ altijd 1, want zijn kansen tellen voor elke instructie op tot 1",
            "small": "$Z(x)$ is piepklein naast de rewards van de antwoorden, dus verandert hij de loss nauwelijks",
        },
        correct="right",
        why={
            "one": "De kansen van het reference model tellen op tot 1, maar gewogen met $e^{r/\\beta}$ niet: $Z(x)$ is hun som, en die kan elk getal boven 0 zijn.",
            "small": "$Z(x)$ kan groot of klein zijn. Hij gaat weg omdat hij voor beide antwoorden op $x$ hetzelfde is, niet om zijn grootte: in het Bradley-Terry model telt alleen het verschil.",
            "right": "Dat maakt DPO mogelijk: $Z(x)$ is een som over elk antwoord dat er bestaat, veel te veel om op te tellen.",
        },
    )
    choice_z
    return (choice_z,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De beste policy uit [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl), omgedraaid tot een reward; de loss van Bradley-Terry uit [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl), nu met die reward als score; en de log-kansen uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), nu van hele antwoorden. Hierna train je in het project DPO op je verhalenmodel met veel paren van een checker, en is het reference model je model met de LoRA-adapter uitgeschakeld.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_logprob, result_loss, choice_z):
    L.wrap_up(result_logprob, result_loss, choice_z)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
