# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-dpo. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("llm-dpo", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [DPO: voorkeuren zonder reward model](/course/open/llm-dpo-loss?lang=nl) en [LoRA: finetunen met een low-rank update](/course/open/llm-lora?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_margin():
    quick_margin = L.choice(
        "quick-margin",
        question="Na een paar stappen DPO maakt de policy het gekozen antwoord van een paar 2 nats minder waarschijnlijk dan het reference model, en het afgewezen antwoord 6 nats minder waarschijnlijk. Is de loss van dat paar nu lager dan ln 2, of hoger?",
        options={
            "above": "Hoger: het gekozen antwoord werd minder waarschijnlijk, en DPO wilde het omhoog",
            "right": "Lager: de marge is -2 - (-6) = 4, boven 0",
            "same": "Precies ln 2: beide antwoorden bewogen, dus er veranderde niets",
        },
        correct="right",
        why={
            "above": "De loss kijkt alleen naar de marge: wat het gekozen antwoord won op het reference model, min wat het afgewezen antwoord won. -2 min -6 is 4. Zie [DPO: voorkeuren zonder reward model](/course/open/llm-dpo-loss?lang=nl).",
            "same": "Beide bewogen, maar niet evenveel: de marge is 4, en de loss is alleen ln 2 bij een marge van 0. Zie [DPO: voorkeuren zonder reward model](/course/open/llm-dpo-loss?lang=nl).",
            "right": "Onthoud het: de loss ziet alleen de marge, niet welk van de twee antwoorden bewoog.",
        },
    )
    quick_margin
    return (quick_margin,)


@app.cell(hide_code=True)
def quick_check_off():
    quick_off = L.choice(
        "quick-off",
        question="Een `LoRALinear` met `on = False`: wat geeft die voor een input `x`?",
        options={
            "zero": "Nullen: zonder zijn adapter staat de layer uit",
            "update": "Alleen de update van zijn adapter, `x @ A.T @ B.T` keer de schaal",
            "right": "Wat de layer die hij omhult geeft, alsof er geen adapter is",
        },
        correct="right",
        why={
            "zero": "`on` schakelt alleen de adapter: `forward` begint altijd met de output van de layer die hij omhult. Zie [LoRA: finetunen met een low-rank update](/course/open/llm-lora?lang=nl).",
            "update": "De update is wat `on` weglaat; de output van de layer die hij omhult blijft. Zie [LoRA: finetunen met een low-rank update](/course/open/llm-lora?lang=nl).",
            "right": "Met elke adapter uit is het model dus het model dat het voor LoRA was.",
        },
    )
    quick_off
    return (quick_off,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De les over DPO trainde op één paar, twintig stappen lang. Een echte run traint op veel paren, en heeft de log-kansen van het reference model van al die paren nodig: van een tweede kopie van het model in het geheugen, of van een ronde over alle paren voordat de training begint. Met LoRA geeft één kopie ze allebei, onderweg. De policy is het SFT-model met adapters; zet de adapters uit, en het SFT-model is terug, als het reference model. En voor de paren zijn geen mensen nodig: bij instructies met gevraagde woorden zegt `words_used` welke antwoorden beter zijn.

    **Een voorbeeld uit het dagelijks leven.** Een schilder schildert op een glasplaat die over een voltooid schilderij ligt. Om te zien wat er veranderd is, tilt de schilder het glas op: daaronder is het schilderij zoals het was.
    """)
    return


@app.cell(hide_code=True)
def predict_moved():
    _y, _c = C.MEASURED["yours"]["moved"], C.MEASURED["course"]["moved"]
    guess_moved = L.guess(
        "moved",
        question="DPO op preference pairs van een checker, één ronde erover. Wat is daarna verder van het reference model geschoven: de gekozen antwoorden van de paren, of de afgewezen antwoorden?",
        options={
            "chosen": "De gekozen antwoorden: DPO trekt ze omhoog, en daar is het voor",
            "right": "De afgewezen antwoorden: die zakken veel verder dan de gekozen antwoorden bewegen",
            "same": "Allebei ongeveer even ver: het ene omhoog, het andere evenveel omlaag",
        },
        reveal=f"**De afgewezen antwoorden.** In de runs van de course, op de eerste {C.MOVED} paren: op haar model van de demorun bewogen de gekozen antwoorden gemiddeld {C.signed(_y['chosen'], 1, L.lang)} nats en de afgewezen antwoorden {C.signed(_y['rejected'], 1, L.lang)}; op haar eigen SFT-model {C.signed(_c['chosen'], 1, L.lang)} en {C.signed(_c['rejected'], 1, L.lang)}. De loss vraagt alleen om de marge, en de afgewezen antwoorden omlaag duwen levert die even goed op als de gekozen antwoorden omhoog trekken: op het tweede model werden zelfs de gekozen antwoorden minder waarschijnlijk. Toch schrijft het model nu vaker antwoorden die elk woord gebruiken: antwoorden op andere instructies dan die van de paren.",
    )
    guess_moved
    return (guess_moved,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Paren van een checker.** Het SFT-model van de course schreef vier antwoorden op elk van duizend instructies, zoals *Write a story with the words: mirror and sun.* Je `words_used` zegt welke antwoorden elk gevraagd woord gebruiken, en je `make_pairs` uit [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl) maakt een paar van elk antwoord dat slaagt en elk antwoord dat zakt. Twee keer geslaagd of twee keer gezakt zegt niets.

        **Het reference model cadeau.** Met LoRA is de policy het SFT-model met een adapter op elke linear layer van zijn blocks. Elke `LoRALinear` heeft zijn schakelaar `on`: met elke adapter uit is de policy weer het SFT-model, het reference model, zonder een tweede kopie in het geheugen. `B` begint bij nul, dus bij de eerste stap zijn de twee gelijk; daarna zijn alleen de adapters veranderd.

        **Eén stap DPO** neemt een batch paren: vier log-kansen per paar, met `answer_logprob` uit [DPO: voorkeuren zonder reward model](/course/open/llm-dpo-loss?lang=nl), die van de policy met de adapters aan en die van het reference model met de adapters uit; de loss van `dpo_loss`; en een stap AdamW op alleen de adapters.
        """),
        draw.flow(
            [("Antwoorden", ["4 op elk van", "1.000 instructies"]),
             ("Paren", ["gekozen, afgewezen", "door words_used"]),
             ("DPO", ["adapters aan:", "de policy", "adapters uit:", "het reference model"])],
            forward=["make_pairs", "dpo_step"],
            back=[],
            title="Van antwoorden naar een beter model"),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent | In code |
    |---|---|---|
    | **on-policy** | getraind op antwoorden die het model dat getraind wordt zelf schreef, voor de training | |
    | **off-policy** | getraind op antwoorden die een ander model schreef; DPO werkt met allebei | `data/course-dpo-answers.json` |
    | `layer.on` | de schakelaar van één adapter: `True` voor de policy, `False` voor het reference model | `layer.on = False` |
    | `torch.stack(xs)` | één tensor van een lijst tensors, met één dimensie meer; de gradient gaat terug naar elk ervan | `torch.stack(pc)` |

    De antwoorden komen van het SFT-model van de course. Is dat ook het jouwe, dan zijn de paren on-policy: het model dat je traint schreef zulke antwoorden zelf, voor de training. Heb je in [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl) je eigen model gefinetuned, dan zijn ze off-policy. DPO traint er net zo goed op, maar antwoorden die ver afstaan van wat je model zou schrijven leren het minder: daarom neemt het paper over DPO als reference model het SFT-model dat de antwoorden schreef, waar dat kan, om dat verschil klein te houden (Rafailov en anderen, 2023, paragraaf 4). De run van de course op haar model van de demorun is zo'n off-policy run.

    ### Recept

    Om je model met DPO te trainen:

    1. Laat een model een paar antwoorden op elke instructie schrijven, en laat een checker zeggen welke slagen: elk antwoord dat slaagt en elk antwoord dat zakt vormen een preference pair.
    2. Zet LoRA-adapters op je SFT-model: de policy.
    3. Elke stap, voor een batch paren: beide antwoorden onder de policy, beide onder het reference model met elke adapter uit, de loss van DPO, en een stap van de optimizer op de adapters.
    4. Eén ronde over de paren, en stel de testinstructies opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een van de duizend instructies, *Write a story with the words: mirror and sun.*, en het oordeel over elk van zijn vier antwoorden.
    """)
    return


@app.cell
def example_pairs():
    _item = C.items()[1]
    print(_item["words"])
    for _answer in _item["answers"]:
        _found = {_w.strip(".,!?;:\"'") for _w in _answer.lower().split()}
        print(all(_w in _found for _w in _item["words"]), _answer[:60] + " ...")
    return


@app.cell(hide_code=True)
def text_after_example():
    _c = C.MEASURED["course"]
    mo.md(f"""
    Het eerste en het derde antwoord gebruiken beide woorden, het tweede en het vierde niet: *Sunnyville* is geen *sun*. Deze instructie geeft dus 2 × 2 = 4 paren: het eerste antwoord tegen het tweede en tegen het vierde, en het derde tegen allebei. Een instructie waarvan alle vier de antwoorden slagen, of alle vier zakken, geeft er geen: van de duizend geven er {C.number(_c['mixed'], 0, L.lang)} paren, {C.number(_c['pairs'], 0, L.lang)} in totaal.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Paren van een checker

        Maak `dpo_pairs(items)` af. Elk item is een dict met de gevraagde `"words"` van een instructie en de `"answers"` erop. Geef een lijst terug van `(words, chosen, rejected)` voor elk paar dat je `make_pairs` maakt uit de antwoorden en de oordelen van je `words_used`. De check roept `dpo_pairs` aan met twee instructies van elk drie antwoorden, en meer.
        """),
        L.task("Laat `dpo_pairs` de oordelen van de checker omzetten in preference pairs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("dpo-pairs", lines=10)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("dpo-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "dpo-pairs",
        [
            "Welke functie van je project zegt of een antwoord elk gevraagd woord gebruikt, en wat heeft `make_pairs` daarvan nodig, voor elk antwoord?",
            "Elk paar van `make_pairs` is twee antwoorden. Wat heeft het paar nog nodig, zodat de instructie voor allebei geschreven kan worden?",
        ],
        why="De checker maakt van vier antwoorden oordelen, en `make_pairs` maakt van de oordelen paren; de woorden gaan mee, want DPO heeft de instructie nodig om de log-kans van elk antwoord erna uit te rekenen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_reference():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het reference model onder de adapters

        Maak `reference_logprob(model, layers, ids, start)` af. `model` is je SFT-model met adapters, `layers` de lijst `LoRALinear`-layers die je `add_lora` teruggaf, en `ids` en `start` zoals bij `answer_logprob`. Geef de log-kans van het antwoord onder het reference model terug: het model met elke adapter uit. Daarna staan de adapters weer aan. De check gebruikt een klein model waarvan de adapters al iets geleerd hebben.
        """),
        L.task("Laat `reference_logprob` de adapters uitzetten, de log-kans uitrekenen, en ze weer aanzetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reference():
    box_reference = L.code_box("reference-logprob", lines=12)
    box_reference
    return (box_reference,)


@app.cell(hide_code=True)
def check_reference(box_reference):
    result_reference = L.check("reference-logprob", box_reference.value)
    result_reference
    return (result_reference,)


@app.cell(hide_code=True)
def hints_reference():
    L.hints(
        "reference-logprob",
        [
            "Elke `LoRALinear` heeft een schakelaar waar zijn `forward` naar kijkt. Wat moet die zijn voor het reference model, en voor welke layers?",
            "Na deze functie moeten de adapters verder leren. Hoe moeten de schakelaars staan als hij klaar is?",
        ],
        why="Onder de adapters is het SFT-model er nog, onaangeroerd: de adapters zijn de enige getallen die leren. Ze uitzetten geeft het reference model, en er is geen tweede kopie van het model nodig.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Eén stap DPO

        Maak `dpo_step(model, layers, opt, batch, beta)` af. `batch` is een lijst van `(chosen, rejected, start)`: de ids van de instructie met elk antwoord, en waar de antwoorden beginnen. Reken de vier log-kansen van elk paar uit, de loss met je `dpo_loss`, en zet één stap met de optimizer `opt`. Geef de loss terug als getal. De check zet drie stappen op een klein model met twee paren, en kijkt naar de loss van elke stap en naar hoe ver `B` van nul af ging, en meer.
        """),
        L.task("Laat `dpo_step` één stap DPO zetten op een batch paren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("dpo-step", lines=16)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("dpo-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "dpo-step",
        [
            "Het afgewezen antwoord gaat door de policy zoals het gekozen antwoord. Welke van je functies geeft beide antwoorden onder het reference model, en waarom mogen dat losse getallen zijn?",
            "Elke training loop van de course sluit een stap af met dezelfde drie regels. Welke, en in welke volgorde?",
        ],
        why="Alleen de log-kansen van de policy hebben een gradient, terug naar de adapters; die van het reference model zijn vaste getallen om mee te vergelijken.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    _y, _c, _r = C.MEASURED["yours"], C.MEASURED["course"], C.RANK
    mo.vstack([
        mo.md(f"""
        ### Train je model met DPO

        De knop zet adapters van rang {_r} op je SFT-model uit [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl), of op dat van de course als je er geen hebt, maakt de paren met je `dpo_pairs`, en laat je `dpo_step` één keer over alle paren gaan, in batches van {C.BATCH}, bij $\\beta = {C.number(C.BETA, 1, L.lang).replace(',', '{,}')}$ en een learning rate van {C.number(C.LR, 3, L.lang)}. Daarna beantwoordt het model de {C.T.TESTS} testinstructies van die les opnieuw, en meet de demo hoe ver de antwoorden van de eerste {C.MOVED} paren bewogen. Op de CPU van de course, met één thread, duurde dat ongeveer {round((_y['seconds'] + _y['other_seconds']) / 60)} minuten voor haar model van de demorun, en ongeveer {round((_c['seconds'] + _c['other_seconds']) / 60)} minuten op twee threads voor het eigen SFT-model van de course, dat ongeveer vijf keer zo groot is. De adapters worden bewaard in `work/projects/llm/dpo.pt`.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def button_cell():
    dpo_button = mo.ui.run_button(label="Train met DPO")
    dpo_button
    return (dpo_button,)


@app.cell(hide_code=True)
def demo_dpo(dpo_button, result_pairs, result_reference, result_step):
    trained = None
    _note = None
    _live = None
    _k = C.RANK   # the rank of the adapters
    if not all(_r.state == "correct" for _r in (result_pairs, result_reference, result_step)):
        _note = "*Maak eerst oefening 1 tot en met 3 af: de demo traint met je `dpo_pairs`, `reference_logprob` en `dpo_step`.*"
    elif dpo_button.value:
        _live = L.live({"title": "Je DPO-run: de loss van elke stap", "x": "stap", "y": "loss",
                        "alt": "De loss van DPO van elke stap van je run tot nu toe, in grijs, en als dikkere lijn het gemiddelde over de laatste stappen; gestippeld ln 2, de loss aan het begin."}, C.MEASURED["yours"]["steps"], low=0.0)

        def _progress(_p):
            if _p["phase"] == "pairs":
                _live.steps = _p["steps"]   # known once the pairs are made
                _live.reference(C.ln2_line(_p["steps"]), "ln 2")
                _live.line(f"**{C.number(_p['pairs'], 0, L.lang)} paren** uit {C.number(_p['instructions'], 0, L.lang)} instructies · adapters van rang {_k} …")
            elif _p["phase"] == "after":
                _live.line(f"**Na DPO** · instructie {_p['done']} van {_p['total']} · elk woord gebruikt: {_p['used']}")
            else:
                _live.extend(_p["fresh"], f"**Stap {C.number(_p['step'], 0, L.lang)} van {C.number(_p['steps'], 0, L.lang)}** · loss {C.number(_p['loss'], 3, L.lang)} · {C.number(_p['seconds'] / 60, 1, L.lang)} minuten")
        _r = C.run(_progress)
        _note = {
            "run": "*De demo stopt met een fout. Slagen je modules in `work/projects/llm/` nog voor hun checks? Verwijder een bestand dat je veranderde, en check zijn oefening opnieuw.*",
            "pairs": "*Je `dpo_pairs` stopt met een fout op de antwoorden van de course, of geeft geen lijst van `(words, chosen, rejected)`. Slaagt hij nog voor zijn check?*",
            "lora": "*Je `add_lora` stopt met een fout op het model. Slaagt hij nog voor zijn check in de les over LoRA?*",
            "step": "*Je `dpo_step` of `reference_logprob` stopt met een fout op het model, of geeft geen getal. Slagen ze nog voor hun checks?*",
            "memory": "*Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop.*",
            "diverged": "*De loss is geen getal meer: de training is ontspoord. Druk opnieuw op de knop.*",
        }.get(_r) if isinstance(_r, str) else None
        trained = None if isinstance(_r, str) else _r
    if _note and _live is not None and _live.xs:
        _live.line(_note)   # a run that stops after its first step keeps its chart, with the note under it
    else:
        mo.output.replace(mo.md(_note)) if _note else mo.output.clear()
    return (trained,)


@app.cell(hide_code=True)
def demo_result(trained):
    _run = C.keep(trained) if trained is not None else C.saved() or C.MEASURED["yours"]
    _mine = trained is not None or C.saved() is not None
    _whose = ("je SFT-model" if _run["base"] == "yours" else "het SFT-model van de course") if _mine else "het model van de demorun van de course"
    _stand = "" if _mine else "\n\n*De run van de course staat er tot je zelf traint.*"
    _ex = _run["example"]
    _after = _ex["after"] if len(_ex["after"]) < 600 else _ex["after"][:600] + " ..."
    mo.vstack([
        mo.md(f"Hoeveel antwoorden van {_whose} op de {_run['after']['total']} testinstructies elk gevraagd woord gebruiken:\n\n"
              f"| | elk woord gebruikt |\n|---|---|\n| SFT | **{_run['before']['used']}** van {_run['before']['total']} |\n"
              f"| SFT en DPO | **{_run['after']['used']}** van {_run['after']['total']} |\n\n"
              f"{C.number(_run['pairs'], 0, L.lang)} paren, {C.number(_run['steps'], 0, L.lang)} stappen. Ten opzichte van het reference model bewogen de gekozen antwoorden van de eerste {C.MOVED} paren gemiddeld {C.signed(_run['moved']['chosen'], 1, L.lang)} nats, de afgewezen antwoorden {C.signed(_run['moved']['rejected'], 1, L.lang)}.{_stand}"),
        *([] if _run.get("snapshots") else [mo.Html(C.losses_picture(_run["losses"], _run["every"], {"title": "De loss van DPO tijdens de run", "x": "stap", "y": "loss", "lang": L.lang,
                                                  "alt": f"De loss van DPO tijdens de run: van 0,693 naar ongeveer {C.number(sum(_run['losses'][-5:]) / len(_run['losses'][-5:]), 2, L.lang)}."}))]),
        mo.md(("**" + ("Een testinstructie die het na DPO deed en ervoor niet" if _ex["better"] else "De eerste testinstructie") + f"**: *{C.T.instruction(_ex['words']).splitlines()[0]}*\n\n" + C.T.P.quote(_after))),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    _y, _c = C.MEASURED["yours"], C.MEASURED["course"]
    mo.md(f"""
    Met DPO ging het model van de demorun van de course van {_y['before']['used']} naar {_y['after']['used']} van de {C.T.TESTS} testinstructies waarvan het antwoord elk gevraagd woord gebruikte; het eigen SFT-model van de course van {_c['before']['used']} naar {_c['after']['used']}. De paren vragen nooit om deze testinstructies, en de checker is dezelfde die ze telt: DPO leerde het model wat `words_used` beloont. Dat moet het hier ook leren, maar daarom is een check die je kunt bespelen een risico, zoals reward hacking liet zien in [RLHF: leren van menselijke voorkeuren](/course/open/llm-rlhf-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(f"""
    ### Je DPO-run, stap voor stap

    Net als in het project over SFT laat de knop je model tijdens DPO zes keer de eerste testinstructie beantwoorden: voor de eerste stap, en daarna nog vijf keer, gelijkmatig verdeeld over de {C.number(C.MEASURED['yours']['steps'], 0, L.lang)} stappen van de run. Hier ga je door die momenten: de loss van DPO tijdens de run, met ln 2 en het moment dat je kiest als gestippelde lijnen, en wat je model toen schreef.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _run = C.keep(trained) if trained is not None else C.saved()
    if _run is None:
        _out = mo.md("*Als je je model met DPO hebt getraind, kun je hier door zijn run gaan.*")
    elif not _run.get("snapshots"):
        _out = mo.md("*Je bewaarde run heeft geen momenten om door te gaan: hij is gemaakt voordat de les ze bewaarde. Train opnieuw met DPO om hier door zijn run te gaan.*")
    else:
        _labels = {"title": "De loss van DPO tijdens je run", "x": "stap", "y": "loss", "lang": L.lang, "reference": "ln 2",
                   "alt": "De loss van DPO tijdens je run, het gemiddelde van vijf gemeten stappen, met ln 2 gestippeld en een gestippelde lijn op het gekozen moment.",
                   "start": "Voor de eerste stap", "after": "Na {step} stappen", "used": "elk woord gebruikt", "not_used": "niet elk woord gebruikt",
                   "stopped": "zelf gestopt", "not_stopped": "niet zelf gestopt"}
        _replay = C.replay_run(_run)
        _out = L.explore("replay", "stap", lambda _v: C.T.replay_frame(_replay, int(_v), _labels, C.ln2_line(_run["steps"])), start=_run["snapshots"][-1]["step"],
                         values=[_s["step"] for _s in _run["snapshots"]], role="",
                         task=f"Op welke momenten gebruikt het antwoord van je model op *{C.T.instruction(_run['snapshots'][0].get('words') or C.T.tests()[0]).splitlines()[0]}* elk gevraagd woord, en stopt het zelf?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    mo.md(f"""
    Voor de eerste stap is `B` nog nul, dus het eerste moment is het antwoord van het SFT-model zelf. De paren vragen nooit om deze instructie: ze lieten het model alleen bij andere instructies zien welk van twee antwoorden elk gevraagd woord gebruikt, volgens je `words_used`. Gebruikt een later moment het woord wel, dan zie je die voorkeur ook bij deze instructie doorwerken. Net als in het project over SFT blijft één antwoord één trekking: de tabel hierboven telt de antwoorden op {C.T.TESTS} instructies.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_off():
    mo.md(r"""
    ### Oefening 4. Waarom uit?
    """)
    return


@app.cell(hide_code=True)
def choice_off_cell():
    choice_off = L.choice(
        "why-off",
        question="`B` begint bij nul, dus bij de eerste stap geeft de policy met zijn adapters aan dezelfde log-kansen als met ze uit. Waarom zet je ze dan toch uit voor het reference model?",
        options={
            "right": "Na de eerste stap hebben de adapters geleerd, en het reference model moet het SFT-model blijven",
            "speed": "Met de adapters uit draait het model sneller, en het reference model is bij elke stap nodig",
            "grad": "Met de adapters aan zou het reference model een gradient krijgen en meeleren met de policy",
        },
        correct="right",
        why={
            "speed": "Een adapter is klein: hem uitzetten spaart weinig werk. De reden is waar het reference model voor staat: het model waarmee de policy begon.",
            "grad": "`dpo_step` maakt van de log-kansen van het reference model losse getallen, en `reference_logprob` werkt binnen `torch.no_grad()`: er gaat geen gradient doorheen, adapters aan of uit. De reden is waar het reference model voor staat: het model waarmee de policy begon.",
            "right": "Met de adapters aan zou het reference model meebewegen met de policy: de marge zou altijd 0 zijn en de loss ln 2, hoe ver de policy ook ging, en niets zou hem dicht bij het SFT-model houden.",
        },
    )
    choice_off
    return (choice_off,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De paren uit [Preference pairs en het Bradley-Terry model](/course/open/llm-preference-pairs?lang=nl), nu gemaakt uit duizend instructies; de loss uit [DPO: voorkeuren zonder reward model](/course/open/llm-dpo-loss?lang=nl), nu op veel paren; en de adapters uit [LoRA: finetunen met een low-rank update](/course/open/llm-lora?lang=nl), die nu ook het reference model geven. Hierna: als een programma het antwoord kan controleren, kan die controle zelf de reward zijn, zoals bij het trainen van reasoning models.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pairs, result_reference, result_step, choice_off):
    L.wrap_up(result_pairs, result_reference, result_step, choice_off)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
