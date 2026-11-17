# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-tracing. Route: course/route/lessen/bouwen.yaml. Checks, de traces van de keukenagent en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tracing", "nl")
    C = L.checks   # de twaalf runs van de keukenagent als traces, en de plaatjes
    TRACE = C.TRACE
    TRACES = C.TRACES
    NEEDED = C.Named(C.needed(C.EXAMPLE), "NEEDED")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [De agent loop](/course/open/build-agent-loop?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="Wanneer stopt de loop van een agent?",
        options={
            "thought": "Zodra het model in zijn antwoord een regel met Thought: schrijft",
            "calls": "Bij een antwoord zonder tool calls, of bij de stappenlimiet",
            "tool": "Als een tool zijn resultaat terugstuurt",
        },
        correct="calls",
        why={
            "calls": "Een antwoord zonder tool calls is het eindantwoord, en de stappenlimiet stopt een agent die tools blijft aanroepen. Elke ronde van die loop is een call naar het model: in deze les is elke ronde een span.",
            "thought": "Een gedachte is tekst die het model voor een tool call schrijft; de loop gaat door en voert de call uit. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
            "tool": "Na een resultaat vraagt de loop het model opnieuw, en het model beslist of het nog een tool nodig heeft. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    Een klant van de keukenapp vraagt wat er nodig is voor pasta met pesto, wacht {s} seconden, en krijgt een zin over wat de agent nog moet nakijken. Het antwoord alleen vertelt je niet waarom het zo lang duurde, en ook niet waar het misging. De agent vroeg het model {c} keer om een antwoord en riep daartussen {tn} keer een tool aan; wat elke stap kreeg en gaf, is weg zodra het request voorbij is, tenzij je code het opschreef.

    **Een voorbeeld uit het dagelijks leven.** Een pakketdienst schrijft elke stop van een pakket op: opgehaald, gesorteerd, in de bus, bezorgd, elk met zijn tijd. Is een pakket te laat, dan laat de lijst zien waar het bleef liggen; ging het naar het verkeerde adres, dan laat de lijst zien bij welke stop.
    """.format(s=C.sec(_f["ex_seconds"], L.lang), c=_f["ex_calls"], tn=len(_f["ex_tools"])))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **span** | één stap van een request, opgeschreven: zijn naam, wanneer hij begon en eindigde, wat erin ging en wat eruit kwam, en de span waarin hij liep |
    | **trace** | alle spans van één request, in elkaar: het request is de buitenste span, en elke call naar het model en elke tool call is een span daarin; ook wel request trace genoemd |
    | **observability** | achteraf kunnen zien wat je app deed en waarom, uit wat hij opschreef terwijl hij draaide, zoals traces |

    Tools voor traces schrijven de spans vaak in de vorm van OpenTelemetry, een gedeelde standaard, zodat elke tool de traces van elke app kan lezen; die heeft afgesproken namen voor de delen van een call naar een model, zoals zijn tokens. In deep learning frameworks betekent tracing iets anders: de bewerkingen van een model vastleggen om het sneller te laten draaien.

    Een trace beantwoordt drie vragen die het antwoord alleen niet beantwoordt: waar ging de tijd heen, wat kostte het request, en bij welke stap ging het mis. Je hebt hem alleen als je code de spans opschrijft terwijl hij draait, bij elk request: je weet niet van tevoren naar welk request een klant zal vragen.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een trace van een request

    1. **Span:** schrijf rond elke stap, een call naar het model of een tool, zijn naam op, zijn begin en eind, en wat erin ging en eruit kwam; bij een call naar het model ook zijn tokens.
    2. **Nesten:** geef elke span het id van de span waarin hij loopt. Het request is de buitenste span.
    3. **Bewaren:** sla de spans van elk request op, ook van de requests die goed gingen.
    4. **Lezen:** de langzaamste spans voor de tijd, de tokens van de calls voor de kosten, en wat erin ging en eruit kwam, span voor span, voor waar het misging.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course schreef de twaalf runs van de keukenagent met ReAct uit [De agent loop](/course/open/build-agent-loop?lang=nl) op als traces. Een span is een dictionary: `"id"`, `"parent"` (het id van de span waarin hij loopt, `None` voor het request), `"name"`, `"start"` en `"end"` in seconden sinds het request binnenkwam, bij een call naar het model `"prompt_tokens"` en `"output_tokens"`, bij een tool `"input"`, zijn argumenten, en `"output"`, wat eruit kwam. `TRACE` is run {i}, pasta met pesto, een lijst van {k} spans:
    """.format(i=C.EXAMPLE, k=len(C.TRACE)))
    return


@app.cell(hide_code=True)
def picture_spans():
    mo.Html(C.spans_table(C.EXAMPLE, L.lang))
    return


@app.cell(hide_code=True)
def text_read():
    _f = C.findings()
    _t = _f["ex_call_seconds"]
    mo.md(r"""
    1. **Tijd:** het request duurde {s} seconden. Zijn {c} calls naar het model duurden {t0}, {t1}, {t2} en {t3} seconden. Zijn {tn} tool calls duren in de trace {z} seconden: de opname mat alleen de calls naar het model, en de tools van de keuken zijn functies in Python die ruim binnen een milliseconde antwoorden.
    2. **Kosten:** elke call naar het model kreeg het hele gesprek opnieuw, dus zijn prompt groeide van {p0} naar {p3} tokens. Samen kregen de calls {ps} prompttokens en schreven ze er {os}: {t} tokens voor een vraag van {w} woorden.
    3. **Waar het misging:** het recept heeft vier ingrediënten. De agent vroeg de voorraadkast naar {a0} en {a1}, nooit naar {m0} en {m1}. Zijn laatste call schreef een gedachte en riep geen tool aan, dus stopte de loop, en kreeg de klant die gedachte als antwoord.
    """.format(s=C.sec(_f["ex_seconds"], L.lang), c=_f["ex_calls"], t0=C.sec(_t[0], L.lang), t1=C.sec(_t[1], L.lang), t2=C.sec(_t[2], L.lang), t3=C.sec(_t[3], L.lang),
               p0=_f["ex_prompts"][0], p3=_f["ex_prompts"][-1], ps=C.thousands(_f["ex_prompt_sum"], L.lang), os=sum(_f["ex_outputs"]),
               t=C.thousands(_f["ex_tokens"], L.lang), w=_f["ex_question_words"], a0=_f["ex_tools"][1][1]["item"], a1=_f["ex_tools"][2][1]["item"],
               m0=_f["ex_missed"][0], m1=_f["ex_missed"][1], tn=len(_f["ex_tools"]), z=C.sec(0.0, L.lang)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### Wat de course mat

    In de twaalf traces samen duurden {mc} calls naar het model {ms} seconden; de {tc} tool calls, elk ruim binnen een milliseconde, tellen nauwelijks mee. De langzaamste span was de eerste call van run {sr}: {ss} seconden, waarin het model {so} tokens schreef. Elke eerste call kreeg {fp0} tot {fp1} prompttokens, de system prompt, de tools en de vraag, en elke latere call meer. In {np} van de {n} runs vroeg de agent de voorraadkast nooit iets, en in elke run werd minstens één ingrediënt nooit opgezocht. Run 5 gaf het enige goede antwoord, en zocht milk, cheese en butter nooit op: er lag toevallig genoeg in de voorraadkast. Een goed antwoord kan een verkeerde weg ernaartoe verbergen; de trace laat de weg zien.
    """.format(mc=_f["model_calls"], ms=C.sec(_f["model_seconds"], L.lang), tc=_f["tool_calls"],
               sr=_f["slowest_run"], ss=C.sec(_f["slowest_seconds"], L.lang), so=_f["slowest_output"], fp0=_f["first_prompt"][0], fp1=_f["first_prompt"][1],
               np=len(_f["never_pantry"]), n=_f["n"]))
    return


@app.cell(hide_code=True)
def picture_measured():
    L.watch_out("Een trace bevat wat klanten schreven en wat het model antwoordde. Bewaar hem net zo veilig als de chats zelf, en niet langer dan je hem nodig hebt.")
    return


@app.cell(hide_code=True)
def text_slowest():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De langzaamste span

        `slowest(spans)` moet de span in het request geven die het langst duurde: van de spans met `span["parent"] == 0` de span waarvan `span["end"] - span["start"]` het grootst is. De startcode houdt de laatste span in het request, hoe lang die ook duurde. Het plaatje onder de check laat je span zien in elk van de twaalf traces, `TRACES`.
        """),
        L.task("Houd een span alleen als hij langer duurde dan `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slowest():
    box_slowest = L.code_box("slowest", lines=8)
    box_slowest
    return (box_slowest,)


@app.cell(hide_code=True)
def check_slowest(box_slowest):
    result_slowest = L.check("slowest", box_slowest.value)
    result_slowest
    return (result_slowest,)


@app.cell(hide_code=True)
def hints_slowest():
    L.hints(
        "slowest",
        [
            "Hoe lang een span duurde, is zijn eind min zijn begin. Vergelijk dat van elke span met dat van `best`, de langzaamste tot nu toe.",
            "Een tweede `if` in de eerste: als `span[\"end\"] - span[\"start\"]` groter is dan hetzelfde voor `best`, wordt de span `best`.",
            "Schrijf onder `if span[\"parent\"] == 0:` de regel `if span[\"end\"] - span[\"start\"] > ___:`, met `best = span` een stap verder in.",
        ],
        why="""```python
def slowest(spans):
    best = spans[1]
    for span in spans:
        if span["parent"] == 0:
            if span["end"] - span["start"] > best["end"] - best["start"]:
                best = span
    return best
```

De span van het request blijft erbuiten: die duurt zo lang als alles erin, dus zou altijd de langzaamste zijn.""",
    )
    return


@app.cell(hide_code=True)
def text_tokens():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De tokens van een request

        `tokens_of(spans)` moet de tokens van alle calls naar het model in een request geven: voor elke span met de naam `"model"` zijn `"prompt_tokens"` plus zijn `"output_tokens"`, bij elkaar opgeteld. De startcode telt alleen de outputtokens.
        """),
        L.task("Tel de prompttokens van elke call er ook bij op."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tokens():
    box_tokens = L.code_box("tokens_of", lines=7)
    box_tokens
    return (box_tokens,)


@app.cell(hide_code=True)
def check_tokens(box_tokens):
    result_tokens = L.check("tokens_of", box_tokens.value)
    result_tokens
    return (result_tokens,)


@app.cell(hide_code=True)
def hints_tokens():
    L.hints(
        "tokens_of",
        [
            "Een call naar het model kost de tokens die hij kreeg, de prompt, en de tokens die hij schreef.",
            "Tel in de regel binnen de `if` ook de prompttokens van de span op bij wat de regel al optelt.",
            "Maak van die regel `total = total + ___ + span[\"output_tokens\"]`.",
        ],
        why="""```python
def tokens_of(spans):
    total = 0
    for span in spans:
        if span["name"] == "model":
            total = total + span["prompt_tokens"] + span["output_tokens"]
    return total
```

Met een prijs per token is dit getal wat een request kost. De prompttokens zijn het grootste deel: elke call stuurt het hele gesprek opnieuw.""",
    )
    return


@app.cell(hide_code=True)
def text_looked():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wat de agent nooit opzocht

        `not_looked_up(spans, needed)` moet de items van `needed` geven, in hun volgorde, waar geen span van de voorraadkast naar vroeg. De startcode verzamelt in `asked` al het item van elke span van de voorraadkast, uit zijn input. `x not in a_list` is `True` als `x` niet in de lijst voorkomt. `NEEDED` bevat de ingrediënten van pasta met pesto.
        """),
        L.task("Loop door `needed`, en houd elk item dat niet in `asked` staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_looked():
    box_looked = L.code_box("not_looked_up", lines=10)
    box_looked
    return (box_looked,)


@app.cell(hide_code=True)
def check_looked(box_looked):
    result_looked = L.check("not_looked_up", box_looked.value)
    result_looked
    return (result_looked,)


@app.cell(hide_code=True)
def hints_looked():
    L.hints(
        "not_looked_up",
        [
            "Het antwoord is een lijst items uit `needed`: die waar de agent nooit naar vroeg.",
            "Een loop `for item in needed:` met daarin een `if` die vraagt of `item` in `asked` ontbreekt.",
            "Schrijf onder `for item in needed:` de regel `if ___:`, met `missing.append(item)` een stap verder in.",
        ],
        why="""```python
def not_looked_up(spans, needed):
    asked = []
    for span in spans:
        if span["name"] == "pantry":
            asked.append(span["input"]["item"])
    missing = []
    for item in needed:
        if item not in asked:
            missing.append(item)
    return missing
```

De stap waar een run misging, vertelt je wat je hierna controleert: een input als deze hoort in je eval set.""",
    )
    return


@app.cell(hide_code=True)
def text_faster():
    _f = C.findings()
    mo.md(r"""
    ### Oefening 4. Sneller maken

    Run {i} duurde {s} seconden.
    """.format(i=C.EXAMPLE, s=C.sec(_f["ex_seconds"], L.lang)))
    return


@app.cell(hide_code=True)
def choice_faster_cell():
    _f = C.findings()
    choice_faster = L.choice(
        "faster",
        question="Wat zou run {i} sneller maken?".format(i=C.EXAMPLE),
        options={
            "tool": "Een snellere tool om in de voorraadkast te kijken",
            "question": "Een kortere vraag van de klant",
            "calls": "Minder of kortere calls naar het model",
        },
        correct="calls",
        why={
            "calls": "Zijn {c} calls naar het model duurden alle {s} seconden; zijn tools, elk ruim binnen een milliseconde, tellen nauwelijks mee. Een call duurt langer naarmate hij meer schrijft: ongeveer {pt} seconden voor elk token dat hij schreef, op de computer van de course. Minder stappen, of kortere gedachten, maken het sneller.".format(
                c=_f["ex_calls"], s=C.sec(_f["ex_seconds"], L.lang), pt=C.sec(round(_f["per_output_token"], 2), L.lang)),
            "tool": "De tool van de voorraadkast antwoordt ruim binnen een milliseconde: de tools van de keuken zijn functies in Python. Alle {s} seconden gingen naar de {c} calls naar het model.".format(
                s=C.sec(_f["ex_seconds"], L.lang), c=_f["ex_calls"]),
            "question": "De vraag heeft {w} woorden. De eerste call kreeg {p} prompttokens, vooral de system prompt en de tools, en elke latere call kreeg dat allemaal opnieuw. De tijd ging naar de calls naar het model, niet naar de vraag.".format(
                w=_f["ex_question_words"], p=_f["ex_prompts"][0]),
        },
    )
    choice_faster
    return (choice_faster,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Lees de traces zelf

    Kies een run, en lees zijn trace: de waterval voor de tijd, en de spans voor wat erin ging en eruit kwam.
    """)
    return


@app.cell(hide_code=True)
def settings_run():
    run_pick = mo.ui.dropdown(options=C.run_options(L.lang), value=list(C.run_options(L.lang))[C.EXAMPLE], label="Run", full_width=True)
    run_pick
    return (run_pick,)


@app.cell(hide_code=True)
def demo_trace(run_pick):
    mo.vstack([mo.Html(C.trace_picture(run_pick.value, L.lang)), mo.Html(C.spans_table(run_pick.value, L.lang))])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Dit zijn de runs die je in [De agent loop](/course/open/build-agent-loop?lang=nl) las als gedachten, acties en observations, nu met hun tijd en tokens. Een eval set vertelt je dat een antwoord fout is; zijn trace vertelt je waar. In [Guardrails](/course/open/build-guardrails?lang=nl) krijgt elke controle voor en na het model een eigen span, en [Hoe agents de fout in gaan](/course/open/build-agent-failures?lang=nl) telt de soorten fouten in traces zoals deze.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_faster, result_looked, result_slowest, result_tokens):
    L.wrap_up(result_slowest, result_tokens, result_looked, choice_faster)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
