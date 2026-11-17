# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-human-in-loop. Route: course/route/lessen/bouwen.yaml. Checks, de tools van de bank, de dag met berichten, de mens en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-human-in-loop", "nl")
    C = L.checks   # de tools van de bank, de dag met berichten en hun stappen, de mens en de plaatjes
    TOOLS = C.TOOLS     # per tool: "money" (kost geld), "sends" (verstuurt iets), "can_undo" (terug te draaien)
    DAY = C.DAY         # tien berichten van één dag, elk met de "steps" die de course erbij uitschreef
    STEPS = C.STEPS     # alle stappen van de dag op een rij; een stap is {"id", "tool", "arguments"}
    PERSON = C.PERSON   # de mens bij de bank: PERSON(step) geeft "yes", "no" of de stap zoals die hem aanpaste


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Wat een tool mag](/course/open/build-tool-permissions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ask():
    quick_ask = L.choice(
        "quick-ask",
        question="Een policy zegt `ask` voor `delete_file`, en de agent roept `delete_file` aan. Wat gebeurt er?",
        options={
            "after": "Hij draait, en de gebruiker hoort het achteraf",
            "never": "Hij wordt geweigerd, wat de gebruiker ook zegt",
            "yes": "Hij draait pas als de gebruiker ja zegt",
        },
        correct="yes",
        why={
            "yes": "`ask` betekent: vraag eerst de gebruiker, en laat de call alleen bij ja draaien. `allow` draait meteen, en `deny` draait nooit.",
            "after": "Dan is nee zeggen al te laat. `ask` vraagt voordat de call draait, en laat hem alleen bij ja draaien. Zie [Wat een tool mag](/course/open/build-tool-permissions?lang=nl).",
            "never": "Dat is `deny`. Bij `ask` beslist de gebruiker, en draait de call bij ja. Zie [Wat een tool mag](/course/open/build-tool-permissions?lang=nl).",
        },
    )
    quick_ask
    return (quick_ask,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    In [Wat een tool mag](/course/open/build-tool-permissions?lang=nl) zei een policy per tool of hij mag draaien, eerst moet vragen, of niet mag draaien. Nu handelt de assistent van de bank voor zijn klanten: hij zoekt een rekening op, blokkeert een kaart, bestelt een nieuwe, betaalt geld terug, verstuurt een e-mail, sluit een rekening. Welke van die stappen horen op een mens te wachten?

    Vraag te weinig, en er gaat 250 euro terug waar de klant om 25 vroeg. Vraag bij elke stap, en op een dag met {tasks} berichten krijgt iemand bij de bank {every} keer een vraag, ook voordat er een saldo gelezen wordt. Elke vraag kost die persoon tijd, en de klant wachttijd.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe medewerker in een winkel mag zelf vragen beantwoorden, dingen opzoeken en een aankoop met de bon erbij terugnemen. Geld terug zonder bon gaat eerst langs de bedrijfsleider. Die wordt niet bij elke klant geroepen: dan blijft er geen tijd over voor de klanten waar het om gaat.
    """.format(tasks=_f["tasks"], every=_f["asks"]["every"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **human-in-the-loop** | een opzet waarin een mens meedoet aan wat een AI-systeem doet: hier iemand die een stap van de agent goedkeurt, aanpast of weigert voordat hij draait |
    | **approval step** | een punt waar de agent vóór een stap stopt en wacht op het antwoord van een mens: ja, een aangepaste stap, of nee |

    Vóór elke stap stelt de code drie vragen over de tool: kost hij geld, verstuurt hij iets buiten de bank, is hij niet terug te draaien? Is geen daarvan waar, dan draait de stap meteen: een fout kost daar weinig en is te herstellen. Is er één waar, dan stopt de agent en vraagt hij een mens. Die kan ja zeggen, de stap aanpassen, zoals het bedrag, of nee zeggen. Bij nee stopt de agent en laat hij de klant weten dat er een mens reageert, want de stappen na de geweigerde kunnen ervan afhangen.

    De vragen gaan over de tool, niet over het model, dus ze staan in de code, net als de policy van [Wat een tool mag](/course/open/build-tool-permissions?lang=nl). De antwoorden van de mens zijn het bewaren waard: elke aanpassing en elk nee is een stap die de agent verkeerd plande, een geval voor de eval set van [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een approval step

    1. **Per tool:** schrijf op wat hij met de wereld doet: of hij geld kost, iets verstuurt, en terug te draaien is.
    2. **In de code die de stappen uitvoert:** vraag vóór elke stap die geld kost, iets verstuurt of niet terug te draaien is; laat de rest meteen draaien.
    3. **Laat de hele stap zien:** de tool en zijn argumenten, zoals het bedrag en het adres, zoals hij gaat draaien.
    4. **Drie antwoorden:** ja laat de stap draaien, een aangepaste stap draait zoals hij is aangepast, nee stopt de agent, die de klant laat weten dat er een mens reageert.
    5. **Bewaar de antwoorden:** tel hoe vaak een mens een stap aanpaste of weigerde, en zet die stappen in je eval set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _task = DAY[2]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `TOOLS` beschrijft de {k} tools van de agent van de bank: bij elke tool of hij geld kost, iets verstuurt en terug te draaien is. `TOOLS["order_card"]` is `{order}`: een nieuwe kaart kost de bank geld, maar de bestelling kan worden geannuleerd.

    `DAY` bevat {tasks} berichten van één dag, elk met de stappen die de course erbij uitschreef, zoals een agent ze zou kunnen plannen; `STEPS` heeft alle {steps} op een rij. `DAY[2]` is *{message}*, en de stappen zijn `{s1}` en `{s2}` met een bedrag van {amount}: een 0 te veel.

    `PERSON` staat voor iemand bij de bank. `PERSON(step)` geeft `"yes"`, `"no"`, of de stap zoals die persoon hem aanpaste. Bij deze terugbetaling geeft `PERSON` dezelfde stap met een bedrag van {fixed}.
    """.format(k=len(TOOLS), order="{" + ", ".join(f'"{_k}": {_v}' for _k, _v in TOOLS["order_card"].items()) + "}", tasks=_f["tasks"], steps=_f["steps"], message=_task["message"],
               s1=_task["steps"][0]["tool"], s2=_task["steps"][1]["tool"], amount=L.number(_task["steps"][1]["arguments"]["amount"], 2),
               fixed=L.number(PERSON(_task["steps"][1])["arguments"]["amount"], 2)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _a = _f["answers"]["rule"]
    mo.md(r"""
    ### Wat de regel doet op de dag

    De course schreef {wrong} van de {steps} stappen expres verkeerd, en schreef op wat de mens antwoordt: 250 euro terugbetalen, een rekening sluiten zonder de klant eerst te bellen, een kaart blokkeren die de klant terugvond, en bonnetjes sturen naar een adres dat in het bericht stond.

    Volgens de regel werd de mens {rule} keer gevraagd: {yes} keer ja, {changed} keer aangepast en {no} keer nee. Drie verkeerde stappen werden opgevangen: de terugbetaling draaide zoals de mens hem aanpaste, en de andere twee draaiden niet. De vierde, de kaart blokkeren, draaide zonder dat iemand werd gevraagd, want een blokkade is terug te draaien: `unblock_card` herstelt het. Wordt bij elke stap gevraagd, dan vangt de mens ook die op, met {every} keer vragen in plaats van {rule}. Zonder vragen draaiden ze alle {wrong}.
    """.format(wrong=_f["wrong"], steps=_f["steps"], rule=_f["asks"]["rule"], yes=_a["yes"], changed=_a["changed"], no=_a["no"], every=_f["asks"]["every"]))
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.ways_picture(L.lang)),
        L.watch_out("Een aangepaste stap draait zoals de mens hem achterliet, dus laat de hele stap zien, met het bedrag en het adres, niet alleen de naam van de tool."),
    ])
    return


@app.cell(hide_code=True)
def text_needs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wanneer vragen

        `needs_approval(step, tools)` moet zeggen of de agent moet vragen vóór `step`: `True` als de tool geld kost, iets verstuurt, of niet terug te draaien is. `tools[step["tool"]]` is de dictionary van de tool, met `"money"`, `"sends"` en `"can_undo"`, elk `True` of `False`. `a or b` is `True` als minstens een van de twee `True` is, en `not x` is `True` als `x` `False` is. De startcode vraagt nooit. Het plaatje laat je functie los op elke tool.
        """),
        L.task("Geef `True` als een van de drie waar is: geld, verstuurt, of niet terug te draaien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_needs():
    box_needs = L.code_box("needs_approval", lines=4)
    box_needs
    return (box_needs,)


@app.cell(hide_code=True)
def check_needs(box_needs):
    result_needs = L.check("needs_approval", box_needs.value)
    result_needs
    return (result_needs,)


@app.cell(hide_code=True)
def hints_needs():
    L.hints(
        "needs_approval",
        [
            "Drie vragen, en één ja is genoeg om te vragen.",
            "Verbind de drie met `or`; de derde vraag is of een stap *niet* terug te draaien is.",
            "Maak er `return tool[\"money\"] or tool[\"sends\"] or ___` van.",
        ],
        why="""```python
def needs_approval(step, tools):
    tool = tools[step["tool"]]
    return tool["money"] or tool["sends"] or not tool["can_undo"]
```

Opzoeken en blokkeren draaien meteen: een fout is daar te herstellen. Een nieuwe kaart vraagt omdat hij de bank geld kost, ook al kan de bestelling worden geannuleerd.""",
    )
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vragen, en doorgaan zoals de mens zegt

        `run_steps(steps, ask)` moet de stappen van één bericht uitvoeren zoals de mens antwoordt. `ask(step)` vraagt het de mens en geeft `"yes"`, `"no"`, of de stap zoals die persoon hem aanpaste. Vraag alleen vóór een stap waarbij `needs_approval(step, TOOLS)` `True` is, en tel hoe vaak je vraagt. Bij `"no"` stop je: de stappen daarna draaien niet. Bij een aangepaste stap laat je die draaien. Geef de stappen terug die draaiden, en het aantal keer vragen. De startcode laat elke stap draaien en vraagt nooit. Het plaatje laat je functie los op elk bericht van de dag, met `PERSON`.
        """),
        L.task("Tel vóór een stap die goedkeuring nodig heeft 1 bij `asks` op en vraag; stop bij nee, en laat een aangepaste stap aangepast draaien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run_steps", lines=13)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run_steps", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run_steps",
        [
            "Drie antwoorden, drie uitkomsten: ja laat de stap draaien, een aangepaste stap draait zoals hij is aangepast, en nee stopt de agent.",
            "Tel binnen een `if needs_approval(step, TOOLS):` 1 bij `asks` op en vraag met `answer = ask(step)`; wat daarna komt, hangt af van `answer`.",
            "Beëindig bij nee meteen de loop: `if answer == \"no\":` met `___` eronder.",
        ],
        why="""```python
def run_steps(steps, ask):
    done = []
    asks = 0
    for step in steps:
        if needs_approval(step, TOOLS):
            asks = asks + 1
            answer = ask(step)
            if answer == "no":
                break
            if answer != "yes":
                step = answer
        done.append(step)
    return done, asks
```

Bij nee stopt de agent in plaats van door te gaan: de e-mail die zegt dat de rekening dicht is, mag niet weg als de rekening nog open is.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.md(r"""
    ### Oefening 3. Welke stap wacht?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-step",
        question="Welke van deze stappen van de agent van de bank hoort op een mens te wachten?",
        options={
            "unblock": "De kaart weer deblokkeren",
            "refund": "40 euro terugbetalen",
            "read": "De laatste tien transacties lezen",
        },
        correct="refund",
        why={
            "refund": "Terugbetalen kost geld en is niet terug te draaien: dat keurt een mens goed. Lezen verandert niets, en deblokkeren is terug te draaien met `block_card`.",
            "unblock": "Deblokkeren is terug te draaien: `block_card` blokkeert de kaart weer. 40 euro terugbetalen kost geld en is niet terug te draaien; bij die stap vraag je het.",
            "read": "Lezen verandert niets in de wereld. 40 euro terugbetalen kost geld en is niet terug te draaien; bij die stap vraag je het.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_every():
    mo.md(r"""
    ### Oefening 4. Bij elke stap vragen?
    """)
    return


@app.cell(hide_code=True)
def choice_every_cell():
    _f = C.findings()
    _every, _rule = _f["asks"]["every"], _f["asks"]["rule"]
    choice_every = L.choice(
        "every-step",
        question="Voor de zekerheid laat de bank zijn agent vóór elke stap een mens vragen, ook vóór het opzoeken. Wat is het risico?",
        options={
            "habit": "De mens zegt uit gewoonte ja",
            "slow": "De agent kan zijn tools niet meer gebruiken",
            "none": "Geen: meer vragen vangt altijd meer",
        },
        correct="habit",
        why={
            "habit": "Op de dag van de course is dat {e} keer vragen in plaats van {r}, en {x} daarvan gaan over stappen die niets kosten en terug te draaien zijn. Wie zoveel onschuldige stappen goedkeurt, kan eraan gewend raken ja te zeggen, ook bij de stap waar het om gaat. Vraag waar het telt.".format(
                e=_every, r=_rule, x=_every - _rule),
            "slow": "De tools draaien nog steeds, elk na een ja. Het risico is de mens: {e} keer vragen per dag in plaats van {r}, de meeste over onschuldige stappen, en wie er zoveel goedkeurt, kan eraan gewend raken ja te zeggen.".format(
                e=_every, r=_rule),
            "none": "Op de dag van de course ving vragen bij elke stap één verkeerde stap meer op, een blokkade die terug te draaien was, voor {x} keer vragen meer. Wie zoveel onschuldige stappen goedkeurt, kan eraan gewend raken ja te zeggen, ook bij de stap waar het om gaat.".format(
                x=_every - _rule),
        },
    )
    choice_every
    return (choice_every,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De regel is een policy zoals in [Wat een tool mag](/course/open/build-tool-permissions?lang=nl), nu op grond van wat een stap met de wereld doet. Elke aanpassing en elk nee is een geval voor de eval set van [Een eval set en regression tests](/course/open/build-eval-set?lang=nl). [Hoe agents ontsporen](/course/open/build-agent-failures?lang=nl) telt de manieren waarop een agent misgaat in opgenomen runs; een approval step is een van de dingen die daartegen helpen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_every, choice_which, result_needs, result_run):
    L.wrap_up(result_needs, result_run, choice_which, choice_every)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
