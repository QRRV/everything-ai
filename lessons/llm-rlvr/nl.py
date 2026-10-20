# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-rlvr. Route: course/route/lessen/taal.yaml. Checks and the course's runs: checks.py.
# termen:allow iteration

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-rlvr", "nl")
    C = L.checks
    R = C.COURSE_RUN          # the course's run: GRPO on its model of llm-dpo, 25 steps
    K = C.COURSE_KEEP         # keeping the answers that pass, and fine-tuning on them
    FIX = C.COURSE_REMEDIES   # two remedies for what the run took down


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl) en [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_when():
    quick_when = L.choice(
        "quick-when",
        question="In het DPO-project maakte een checker van antwoorden van het SFT-model preference pairs. Wanneer keek hij naar de antwoorden?",
        options={
            "right": "Eén keer, voor de training, om de pairs te maken",
            "every": "Na elke stap, op de nieuwste antwoorden van het model",
            "never": "Nooit: DPO heeft geen checker nodig, alleen een reward model",
        },
        correct="right",
        why={
            "right": "De pairs werden één keer gemaakt, uit antwoorden van het SFT-model, en bleven de hele run hetzelfde.",
            "every": "DPO traint op pairs die voor de run gemaakt zijn; de antwoorden die het model geeft terwijl het leert, worden niet gecontroleerd. Zie [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).",
            "never": "DPO heeft geen reward model nodig, maar wel pairs, en in het project maakte een checker ze. Zie [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl).",
        },
    )
    quick_when
    return (quick_when,)


@app.cell(hide_code=True)
def quick_check_kl():
    quick_kl = L.choice(
        "quick-kl",
        question="Bij RLHF gaat er een KL penalty van de reward van elk antwoord af. Waar is die voor?",
        options={
            "accurate": "Hij maakt het reward model nauwkeuriger",
            "short": "Hij maakt de antwoorden korter, zodat ze minder kosten",
            "right": "Hij houdt de policy dicht bij het reference model",
        },
        correct="right",
        why={
            "right": "Hoe verder de log-kansen van de policy afdrijven van die van het bevroren reference model, hoe meer er afgaat: de policy blijft dicht bij het model waarmee hij begon.",
            "accurate": "Het reward model wordt vooraf getraind, op preference pairs; de KL penalty verandert het niet. Zie [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl).",
            "short": "Hij meet hoe ver de policy van het reference model is afgedreven, niet hoe lang een antwoord is. Zie [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl).",
        },
    )
    quick_kl
    return (quick_kl,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In het DPO-project keek een checker één keer naar de antwoorden, maakte pairs, en DPO leerde van die pairs. Maar een programma dat een antwoord kan controleren, kan meer: het kan elk nieuw antwoord controleren terwijl het model leert, en het meteen belonen. Dat is reinforcement learning met verifiable rewards, RLVR. Er is geen reward model voor nodig, dat door een policy bedot kan worden, en geen mensen die antwoorden vergelijken: alleen een controle die een programma kan doen, zoals of een woord is weggelaten, of code zijn tests haalt, of een som de goede uitkomst heeft. Zo worden reasoning models getraind: op wiskunde- en codeproblemen waarvan een programma de antwoorden kan controleren.

    **Een voorbeeld uit het dagelijks leven.** Vrije worpen oefenen. Je kijkt niet één keer een video van goede worpen en doet die na; je gooit, ziet meteen of de bal erin ging, en past de volgende worp aan. De basket is het programma dat controleert.
    """)
    return


@app.cell(hide_code=True)
def predict_side():
    guess_side = L.guess(
        "side",
        question=f"De course traint zijn model uit het DPO-project {len(R['rewards'])} stappen met RL, op instructies om een verhaal zonder een woord te schrijven, beloond door een programma. Bij woorden waarop de run nooit trainde, laat het het woord {R['without']['after']} van de {R['without']['total']} keer weg in plaats van {R['without']['before']}. Vraag je om een verhaal *met* woorden, het soort instructie waarop DPO het trainde: hoe vaak gebruikt het nu elk gevraagd woord?",
        options={
            "better": "Vaker dan eerst",
            "same": "Ongeveer even vaak als eerst",
            "worse": "Veel minder vaak",
        },
        reveal=f"**Veel minder vaak:** het model gebruikt elk gevraagd woord {R['with']['after']} van de {R['with']['total']} keer, waar het dat eerst {R['with']['before']} keer deed. De run verhoogde wat zijn reward controleerde, en niets hield vast wat hij niet controleerde. *Wat je programma beloont, krijg je; wat het niet meet, kun je kwijtraken.*",
    )
    guess_side
    return (guess_side,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een verifiable reward.** Een programma kijkt naar een antwoord en geeft het een reward: hier 1 als het verhaal het woord weglaat en minstens 3 zinnen heeft, anders 0. Hetzelfde programma voor elk antwoord, de hele run lang.

        **Groepen, en hoeveel beter.** Voor elke instructie schrijft het model een groep van $G$ antwoorden, hier 8. Elk krijgt zijn reward $r_i$, en zijn **advantage** is hoeveel beter het het deed dan zijn groep:
        """),
        L.formula(r"A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j"),
        mo.md(r"""
        Een antwoord dat slaagde in een groep waar de meeste zakten, krijgt een grote advantage; een dat zakte waar de meeste slaagden, een grote negatieve. Een groep waarvan alle antwoorden slagen, of alle zakken, heeft advantages van 0: daar leert het model niets van, zoals een instructie waarvan alle antwoorden slaagden geen DPO-pairs gaf.

        **De stap.** De loss maakt de antwoorden met een positieve advantage waarschijnlijker, en die met een negatieve advantage minder waarschijnlijk. Elk antwoord telt mee met de gemiddelde log-kans $\overline{\log \pi}(a_i)$ van zijn tokens, zodat een lang antwoord even zwaar telt als een kort:
        """),
        L.formula(r"\mathcal{L} = -\frac{1}{G}\sum_{i=1}^{G} A_i \,\overline{\log \pi}(a_i)"),
        mo.md(r"""
        Dit is GRPO, group relative policy optimization, in zijn eenvoudigste vorm: er is geen model voor nodig dat raadt hoe goed een antwoord zal zijn, want de groep zelf is de maatstaf. Volledige GRPO deelt ook nog door de spreiding van de rewards van de groep, begrenst hoe ver één stap de kans op een antwoord kan veranderen, zoals PPO doet, en voegt een KL penalty naar het reference model toe.

        **Een eenvoudigere weg, en waarom die hier niet genoeg was.** Een oudere, eenvoudigere manier om van een programma te leren is alleen de antwoorden te houden die slagen en daarop te finetunen, zoals STaR en expert iteration doen. De course probeerde dat op zijn model: van {tried} verhalen slaagden er {kept}, en na SFT daarop liet het het woord {after} van de {total} keer weg in plaats van {before}. Houden wat slaagt, duwt nooit omlaag wat zakt, en dat moest het hier juist leren.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **verifiable reward** | een reward die een programma geeft door het antwoord te controleren, met dezelfde controle voor elk antwoord |
        | **groep** ($G$ antwoorden) | de antwoorden die het model op één instructie schrijft, vergeleken met elkaar |
        | **advantage** ($A_i$) | hoeveel beter een antwoord het deed dan zijn groep: zijn reward min het gemiddelde van de groep |
        | **GRPO** | reinforcement learning die elk antwoord beloont naar zijn advantage binnen zijn groep |
        | **expert iteration** | alleen de antwoorden houden die door het programma komen, daarop finetunen, en dat opnieuw doen |
        | $\overline{\log \pi}(a_i)$ | de gemiddelde log-kans van de tokens van antwoord $a_i$ onder de policy |

        Elk punt is de gemiddelde reward van de 32 antwoorden van één stap: 4 instructies, elk 8 antwoorden.
        """.replace("{tried}", str(K["tried"])).replace("{kept}", str(K["kept"])).replace("{after}", str(K["without"]["after"]))
           .replace("{total}", str(K["without"]["total"])).replace("{before}", str(K["without"]["before"]))),
        mo.Html(C.rewards_picture(R["rewards"], {"title": "De course: gemiddelde reward per stap", "x": "stap", "y": "gemiddelde reward",
                                                 "course": "", "alt": "De gemiddelde reward van elke stap van de run van de course van 25 stappen: van ongeveer 0,3 in de eerste vijf stappen tot ongeveer 0,8 in de laatste vijf."})),
        L.watch_out("Een programma controleert precies wat het controleert, en de run vindt elke manier om erdoor te komen. Een leeg verhaal laat elk woord weg: daarom vraagt de reward van de course ook om minstens 3 zinnen. En wat het programma niet bekijkt, kan de run veranderen: meet meer dan je beloont."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén stap GRPO met een verifiable reward:

    1. Neem een paar trainingsinstructies, en laat het model op elk een groep antwoorden schrijven, bij zijn gewone temperatuur.
    2. Geef elk antwoord zijn reward met het programma.
    3. Geef elk antwoord zijn advantage: zijn reward min het gemiddelde van zijn groep.
    4. Neem de loss: min het gemiddelde van advantage maal de gemiddelde log-kans van elk antwoord; zet een stap.
    5. Meet na de run op instructies waarop het nooit trainde: het soort dat je beloonde, en de andere soorten ook.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén groep van 8 verhalen zonder het woord *park*: 3 kwamen door het programma, 5 niet. Voer de cel uit.
    """)
    return


@app.cell
def example_group():
    _rewards = [1, 0, 0, 1, 1, 0, 0, 0]
    _mean = sum(_rewards) / len(_rewards)
    print("mean:", _mean)
    print("advantages:", [r - _mean for r in _rewards])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het gemiddelde is $3/8 = 0{,}375$. Elk van de 3 verhalen die slaagden, krijgt $1 - 0{,}375 = 0{,}625$, elk van de 5 die zakten $-0{,}375$; samen tellen ze op tot 0. De stap maakt de 3 waarschijnlijker en de 5 minder waarschijnlijk, in totaal met evenveel gewicht: $3 \times 0{,}625 = 5 \times 0{,}375$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_advantages():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De advantages van een groep

        Maak `advantages(rewards)` af: voor elk antwoord van een groep zijn reward min de gemiddelde reward van de groep. De check probeert een groep van 4, een groep van 8 waarin er maar één slaagde, groepen die allemaal slagen of allemaal zakken, en rewards tussen 0 en 1.
        """),
        L.task("Maak `advantages` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_advantages():
    box_advantages = L.code_box("advantages", lines=5)
    box_advantages
    return (box_advantages,)


@app.cell(hide_code=True)
def check_advantages(box_advantages):
    result_advantages = L.check("advantages", box_advantages.value)
    result_advantages
    return (result_advantages,)


@app.cell(hide_code=True)
def hints_advantages():
    L.hints(
        "advantages",
        [
            "Het gemiddelde van een lijst is zijn som gedeeld door zijn lengte.",
            "Een list comprehension geeft elke reward min dat gemiddelde.",
        ],
        why="Een antwoord telt naar hoeveel beter het het deed dan de andere op dezelfde instructie, dus gaat het gemiddelde van de groep van elke reward af.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loss van een groep

        Maak `policy_loss(logprobs, advantages)` af. `logprobs` heeft voor elk antwoord van een groep een tensor met de log-kansen van zijn tokens onder het model; `advantages` heeft een getal voor elk antwoord. De loss is min het gemiddelde, over de antwoorden, van advantage maal de gemiddelde log-kans van de tokens van het antwoord. `logprobs[i].mean()` geeft het gemiddelde van de tensor van antwoord `i`, en `torch.stack(terms)` maakt één tensor van een lijst getallen die tensors zijn, zodat `.mean()` er ook op werkt.

        De check probeert een voorbeeld van twee antwoorden van 2 en 4 tokens, dezelfde twee met gelijke advantages, een groep van drie antwoorden, de gradient van de loss, en advantages van 0.
        """),
        L.task("Maak `policy_loss` af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("policy-loss", lines=12)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("policy-loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "policy-loss",
        [
            "Voor antwoord `i` is de term zijn advantage maal het gemiddelde van zijn log-kansen. `advantages[i]` is zijn advantage, en de tekst van de oefening zegt hoe je het gemiddelde van een tensor neemt.",
            "De loss is min het gemiddelde van de termen: een stap die hem verlaagt, verhoogt de termen met een positieve advantage.",
        ],
        why="Min advantage maal log-kans verlagen, verhoogt de log-kans van elk antwoord met een positieve advantage en verlaagt die van de andere; het gemiddelde over de tokens voorkomt dat een lang antwoord zwaarder telt dan een kort.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Train je model met GRPO

    De knop meet je model uit het DPO-project (of dat van de course, als je er geen hebt) op {C.MEASURE} instructies om een woord weg te laten waarop de run nooit traint, en op {C.MEASURE} van de testinstructies met woorden uit het SFT-project; daarna traint hij het met GRPO, met je `advantages` en `policy_loss`, groepen van {C.G} antwoorden op {C.PROMPTS} instructies per stap, {C.STEPS['cpu']} stappen op een CPU of {C.STEPS['gpu']} op een GPU; en dan meet hij opnieuw. Met het model van de course op een CPU duurt dat ongeveer zeven minuten, met een kleiner eigen model korter, en op een GPU veel korter. Tien stappen komen niet zo ver als de {len(R['rewards'])} van de course: zijn run staat onder de jouwe getekend.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Train mijn model met GRPO")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_train(box_advantages, box_loss, demo_button, result_advantages, result_loss):
    trained = None
    if result_advantages.state != "correct" or result_loss.state != "correct":
        _out = mo.md("*Maak eerst `advantages` en `policy_loss` goed in oefening 1 en 2: de run traint ermee.*")
    elif not demo_button.value:
        _out = mo.md("*Druk op de knop om je model te trainen.*")
    else:
        _live = L.live({"title": "Je run: gemiddelde reward per stap", "x": "stap", "y": "gemiddelde reward",
                        "alt": "De gemiddelde reward van elke stap van je run tot nu toe, als lijn in inkt met een stip bij elke stap; gestippeld de run van de course op zijn model, bij dezelfde stappen."}, C.steps(), role="ink", low=0.0, high=1.0)
        _live.reference(list(enumerate(R["rewards"][:C.steps()], 1)), "de run van de course", role="ink")

        def _show(_d):
            if _d["phase"] == "measure":
                _live.line(f"**Bezig met meten:** {_d['done']} van {_d['total']} …")
            elif _d["phase"] == "start":
                _live.line(f"**Bezig met trainen:** stap 1 van {_d['steps']} …")
            else:
                _live.extend(_d["fresh"], f"**Stap {_d['step']} van {_d['steps']}:** gemiddelde reward {C.number(_d['reward'], 2, 'nl')}, {_d['seconds']:.0f} s")

        _r = C.run(run_code(box_advantages.value).names["advantages"], run_code(box_loss.value).names["policy_loss"], _show)
        trained = None if isinstance(_r, str) else _r
        if _r == "run":
            _out = _live.stopped("*Het laden of draaien van je model stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks? Verwijder een bestand dat je veranderde, en check zijn oefening opnieuw.*")
        elif _r == "advantages":
            _out = _live.stopped("*Je `advantages` stopt met een fout op een groep van de run, of geeft niet voor elk antwoord een getal.*")
        elif _r == "loss":
            _out = _live.stopped("*Je `policy_loss` stopt met een fout op een groep van de run, of geeft niet één getal met een gradient.*")
        elif _r == "memory":
            _out = _live.stopped("*Het apparaat had niet genoeg geheugen.*")
        else:
            _b, _a = _r["before"], _r["after"]
            _out = mo.vstack([
                mo.Html(C.rewards_picture(_r["rewards"], {"title": "Je run: gemiddelde reward per stap", "x": "stap", "y": "gemiddelde reward",
                                                          "course": "gestippeld: de run van de course", "alt": "De gemiddelde reward van elke stap van je run, met de run van de course van 25 stappen als een gestippelde lijn."},
                                          course=R["rewards"])),
                mo.md(f"**{'Je model' if _r['source'] == 'yours' else 'Het model van de course'}**, {_r['steps']} stappen op {_r['device']}, {_r['seconds']:.0f} s:\n\n"
                      f"| instructies waarop de run nooit trainde | ervoor | erna |\n|---|---|---|\n"
                      f"| zonder het woord: weggelaten | {_b['without']} van {_b['total']} | {_a['without']} van {_a['total']} |\n"
                      f"| met woorden: elk woord gebruikt | {_b['with']} van {_b['total']} | {_a['with']} van {_a['total']} |"),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(rf"""
    **Je run lezen.** In {C.STEPS['cpu']} stappen verschuift *zonder* maar een beetje, en op {C.MEASURE} instructies kan een verschil van een of twee toeval zijn; *met woorden* kan al omlaaggaan. Met het model van de course op een CPU zijn je stappen de eerste {C.STEPS['cpu']} van de course, dus ligt je lijn op de gestippelde. De run slaat het model niet op: je model uit het DPO-project blijft zoals het was.
    """)
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(r"""
    ### Je run, stap voor stap

    Bij elke stap bewaart de run de groep antwoorden op de eerste instructie van die stap: welke reward elk antwoord kreeg, en wat je `advantages` ervan maakte. Hier ga je door de stappen: de gemiddelde reward van elke stap, met een lijn bij de stap die je kiest; ernaast de advantage van elk antwoord van de groep van die stap als een staaf omhoog of omlaag vanaf nul; en daaronder het begin van elk antwoord.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _run = trained if trained is not None else C.saved()
    if _run is None or not _run.get("groups"):
        _out = mo.md("*Als je je model met GRPO hebt getraind, kun je hier door zijn run gaan.*")
    else:
        _labels = {"run": "Je run: gemiddelde reward per stap", "run_x": "stap", "run_y": "gemiddelde reward",
                   "title": "Stap {step}: de groep van één instructie", "x": "antwoord", "y": "advantage", "lang": L.lang,
                   "alt": "De gemiddelde reward van elke stap van je run, met een gestippelde lijn bij de gekozen stap; en de advantage van elk antwoord van de groep van die stap, als een staaf omhoog of omlaag vanaf nul.",
                   "step": "Stap {step}", "mean": "gemiddelde reward van de groep", "reward": "reward", "advantage": "advantage"}
        _out = L.explore("replay", "stap", lambda _v: C.group_frame(_run, int(_v), _labels), start=C.replay_start(_run),
                         values=[_g["step"] for _g in _run["groups"]], role="",
                         task="Welke antwoorden van een groep krijgen een positieve advantage, en tot hoeveel tellen de advantages van een groep samen op?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(r"""
    De antwoorden met een reward boven het gemiddelde van de groep krijgen een positieve advantage, die eronder een negatieve, en in elke groep tellen de advantages samen op tot nul: de stap maakt die erboven waarschijnlijker en die eronder minder waarschijnlijk, sterker naarmate ze verder van het gemiddelde liggen. Een groep waarin elk antwoord dezelfde reward krijgt, heeft overal een advantage van nul: daar leert de stap niets van. GRPO leert alleen van groepen waarin sommige antwoorden het beter doen dan andere.
    """)
    return


@app.cell(hide_code=True)
def text_hack():
    mo.md(r"""
    ### Oefening 3. Waarom de extra check
    """)
    return


@app.cell(hide_code=True)
def choice_hack_cell():
    choice_hack = L.choice(
        "hack",
        question="De reward van de course is alleen 1 als het verhaal het woord weglaat *en* minstens 3 zinnen heeft. Waarom de tweede check?",
        options={
            "easier": "Verhalen van minstens 3 zinnen kan het programma makkelijker controleren",
            "right": "Zonder hem zou een leeg verhaal de volle reward krijgen",
            "model": "Het model kan geen verhalen van minder dan 3 zinnen schrijven",
        },
        correct="right",
        why={
            "right": "Een verhaal zonder woorden, of van één korte regel, haalt *laat het woord weg* voor elk woord. Een run die zulke antwoorden vindt, verhoogt zijn reward zonder betere verhalen te schrijven: reward hacking, met een programma in plaats van een reward model.",
            "easier": "Het programma controleert een verhaal van elke lengte even makkelijk. Het gaat erom wat de run zou kunnen doen om alleen door de eerste check te komen. Kijk nog eens naar het blok *Let op* in *Het idee*.",
            "model": "Dat kan het wel: van de verhalen zonder een woord van de course had ongeveer een op de zes er maar 2 zinnen, en een run kan het naar kortere duwen. Het gaat erom wat de run zou kunnen doen om alleen door de eerste check te komen. Kijk nog eens naar het blok *Let op* in *Het idee*.",
        },
    )
    choice_hack
    return (choice_hack,)


@app.cell(hide_code=True)
def text_exercise_trade():
    mo.md(rf"""
    ### Oefening 4. Wat de run liet zakken

    In de run van de course ging het woord weglaten van {R['without']['before']} naar {R['without']['after']} van de {R['without']['total']}, en elk gevraagd woord gebruiken van {R['with']['before']} naar {R['with']['after']} van de {R['with']['total']}. De course probeerde twee remedies, elk {len(R['rewards'])} stappen:

    | run | zonder het woord, van {R['without']['total']} | met woorden, van {R['with']['total']} |
    |---|---|---|
    | ervoor | {R['without']['before']} | {R['with']['before']} |
    | GRPO op *zonder* | {R['without']['after']} | {R['with']['after']} |
    | GRPO op *zonder*, met een KL penalty, $\beta = 0{{,}}2$ | {FIX['kl']['without']} | {FIX['kl']['with']} |
    | GRPO met de helft van de instructies *met* woorden, beloond met hun eigen check | {FIX['mix']['without']} | {FIX['mix']['with']} |
    """)
    return


@app.cell(hide_code=True)
def explain_trade():
    explain_trade_cell = L.explain(
        "trade",
        question="Waarom ging *met woorden* omlaag, en wat laten de twee remedies zien? Wat zou je hierna proberen?",
        model_answer=f"De reward controleerde alleen dat het woord was weggelaten, dus duwde de run het model waar die check naartoe wees, en niets hield vast dat het gevraagde woorden gebruikte: het lijkt in het algemeen geleerd te hebben gevraagde woorden vaker weg te laten. Een KL penalty van 0,2 hield de policy zo dicht bij het reference model dat hij nauwelijks leerde waarvoor hij beloond werd: *zonder* bleef op {FIX['kl']['without']}; en *met* ging toch omlaag, naar {FIX['kl']['with']}, want de penalty kijkt alleen naar de antwoorden op de instructies waarop de run traint. Instructies met woorden ertussen mengen, elk met een eigen check, hield meer van *met* vast ({FIX['mix']['with']}) terwijl *zonder* minder steeg ({FIX['mix']['without']}). Ik zou een kleinere beta proberen, tussen 0 en 0,2, samen met de mix, en beide soorten meten, en andere soorten instructies ook, na elke paar stappen.",
        points=[
            "de reward controleerde één soort instructie, en niets beschermde de andere",
            "een KL penalty die de policy dicht bij het reference model houdt, laat hem weinig leren, en hier beschermde hij de andere soort niet",
            "meerdere soorten belonen, elk met een eigen check, spreidt wat de run leert",
            "meet elke soort die ertoe doet, niet alleen de soort die je beloont",
        ],
    )
    explain_trade_cell
    return (explain_trade_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De checker van [Project: DPO, voorkeuren zonder reward model](/course/open/llm-dpo?lang=nl) maakte één keer pairs; hier controleert een programma elk nieuw antwoord en beloont het meteen. De KL penalty en reward hacking van [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl) kwamen terug, met een programma in plaats van een reward model. De volgende les, *Heeft post-training geholpen?*, meet je modellen op meer soorten instructies, en deze run liet zien dat dat nodig is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_advantages, result_loss, choice_hack, explain_trade_cell):
    L.wrap_up(result_advantages, result_loss, choice_hack, explain_trade_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
