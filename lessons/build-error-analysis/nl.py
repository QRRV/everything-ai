# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-error-analysis. Route: course/route/lessen/bouwen.yaml. Checks, de opgenomen antwoorden, de notities van de course en de figuren: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-error-analysis", "nl")
    C = L.checks   # de assistent van de bank, zijn opgenomen antwoorden, de notities van de course erbij, de checks en de figuren
    NOTES = C.NOTES
    COUNTS = C.COUNTS


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_json():
    quick_json = L.choice(
        "quick-json",
        question="In JSON mode waren alle acht antwoorden van Qwen3 over een fiets geldige JSON. Wat vertelde dat je?",
        options={
            "fits": "Dat elk antwoord de goede keys en types had",
            "form": "Dat json.loads ze kon lezen, meer niet",
            "right": "Dat de gegevens in de antwoorden klopten",
        },
        correct="form",
        why={
            "form": "Geldige JSON is nog niet de JSON die je app nodig heeft: in vier van die antwoorden was de prijs nog een tekst. En of wat een antwoord zegt klopt, kan geen enkele check van de vorm je vertellen.",
            "fits": "JSON mode houdt alleen de tekst JSON: in vier van de acht antwoorden was de prijs nog een tekst. Keys en types kijkt een JSON schema na. Zie [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).",
            "right": "JSON mode kijkt naar de vorm, niet naar de betekenis: geldige JSON kan een foute prijs bevatten. Zie [JSON waar je op kunt bouwen](/course/open/build-structured-output?lang=nl).",
        },
    )
    quick_json
    return (quick_json,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een assistent beantwoordt de klanten van een onlinebank in de chat van haar app. Je probeert hem uit op 120 echte vragen, en elk antwoord is geldige JSON met een onderwerp en een reactie. Is de assistent goed? Een getal vertelt je alleen wat je bedacht had te meten. Wat er in deze app misgaat, zoals een reactie die de bank de schuld geeft of een stap die de app niet heeft, vind je alleen door de antwoorden te lezen.

    **Een voorbeeld uit het dagelijks leven.** Een docent kijkt een stapel opstellen na. Niet met een lijst mogelijke fouten die vooraf is gemaakt, maar door elk opstel te lezen en een woord in de kantlijn te schrijven. Achteraf laten de kantlijnen zien dat de halve klas dezelfde twee woorden door elkaar haalt: dat is de les die volgt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **error analysis** | de antwoorden van een app een voor een lezen, en bij elk in een paar woorden opschrijven wat er misgaat |
    | **failure taxonomy** | de soorten fouten die je vond, elk met een regel voor wat erbij hoort, en hoe vaak elke soort voorkwam |
    | **evaluation-driven development** | een app bouwen door voor en na elke wijziging te meten, en een wijziging alleen te houden als de metingen laten zien dat die helpt; wat je meet, komt uit de soorten die je vond |

    De soorten komen uit de antwoorden, niet uit een lijst die je vooraf maakt: zo'n lijst noemt fouten zoals "ongeldige JSON", en mist die van jouw app. Eén antwoord kan meer dan één soort hebben. Lezen kost tijd, dus je leest een steekproef van ongeveer honderd antwoorden; de grootste soorten zie je daar al in.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: error analysis

    1. **Verzamelen:** ongeveer honderd echte inputs, en de antwoorden van je app daarop.
    2. **Lezen en noteren:** schrijf bij elk antwoord in een paar woorden op wat er misgaat, of niets als het goed is.
    3. **Indelen:** zet notities die hetzelfde zeggen bij elkaar in één soort, en geef elke soort een naam en een regel.
    4. **Tellen:** voor elke soort het aantal antwoorden dat die heeft.
    5. **Kiezen:** pak eerst de grootste soort aan, en meet na de wijziging opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        1. **Verzamelen:** 120 vragen van echte klanten van een onlinebank, 20 voor elk van zes onderwerpen, uit de dataset BANKING77. De assistent is Qwen3 met deze system prompt, bij temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt:
        """),
        mo.Html(C.prompt_html(C.SYSTEM)),
        mo.md(r"""
        2. **Lezen en noteren:** drie van de antwoorden, met wat de course bij elk opschreef:
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.vstack([mo.Html(C.answer_card(_i, L.lang)) for _i in C.WORKED])
    return


@app.cell(hide_code=True)
def text_worked_sort():
    _f = C.findings()
    mo.md(r"""
    3. **Indelen:** de notitie bij antwoord 4 bevat drie soorten: het verkeerde onderwerp, de stappen van een ander probleem, en een fout van de bank die de klant nooit noemde. Antwoord 80 zegt iets wat niet in de feiten staat: *verzonnen*. Antwoord 2 heeft geen notitie.
    4. **Tellen:** de course las zo alle {n} antwoorden: {noted} hadden minstens één soort, {fine} geen enkele.
    5. **Kiezen:** de grootste soort is het verkeerde onderwerp, met {t} antwoorden. Oefening 4 vraagt wat je daaraan doet.
    """.format(n=_f["n"], noted=_f["noted"], fine=_f["fine"], t=_f["counts"]["wrong-topic"]))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### Wat de course mat

    Alle {n} antwoorden waren geldige JSON met een onderwerp en een reactie, en alle {n} hielden vanzelf op, met `finish_reason` `"stop"`. Gemeten naar de vorm is de assistent klaar. Gelezen hadden {noted} van de {n} antwoorden iets fout. De course deelde haar notities in zes soorten in:
    """.format(n=_f["n"], noted=_f["noted"]))
    return


@app.cell(hide_code=True)
def picture_kinds():
    mo.Html(C.kinds_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_kinds():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Samen tellen de soorten meer dan {noted}, omdat {multi} antwoorden er meer dan één hadden. Het verkeerde onderwerp en een ander probleem komen vaak samen voor: {both} antwoorden hebben ze allebei. {rr} van de {r} vragen over een terugbetaling, en {pr} van de {p} over een geblokkeerde pincode, kregen het onderwerp `card_payment_wrong_exchange_rate`, en {asks} van die {rate} antwoorden vroegen om de datum en het bedrag van een betaling. Voor een vraag die de feiten niet beantwoorden, zegt de system prompt dat de klant moet horen dat iemand van het team zal reageren: geen van de {n} antwoorden deed dat, ook niet bij *{police}*, een vraag die de feiten niet beantwoorden.
        """.format(noted=_f["noted"], multi=_f["noted"] - _f["one_kind"], both=_f["topic_and_other"], rr=_f["refunds_rate"], r=_f["refunds"],
                   pr=_f["pins_rate"], p=_f["pins"], asks=_f["rate_asks"], rate=_f["refunds_rate"] + _f["pins_rate"], n=_f["n"], police=C.QUESTIONS[99])),
        L.watch_out("Twee mensen die dezelfde antwoorden lezen, schrijven niet altijd dezelfde notities. Schrijf de regel van elke soort op, zoals in de tabel, en lees opnieuw met die regel als je twijfelt: een regel maakt het tellen herhaalbaar."),
    ])
    return


@app.cell(hide_code=True)
def text_count():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De soorten tellen

        `NOTES` bevat de notities van de course: een lijst voor elk antwoord, met de soorten die de course erbij opschreef, en `[]` voor een antwoord zonder fout. `NOTES[3]`, de notitie bij antwoord 4, is `["wrong-topic", "other-problem", "admits"]`. `count_kinds(notes)` moet een dictionary geven met voor elke soort het aantal antwoorden dat die heeft. De begincode zet voor elke soort 1 neer. De check probeert ook korte lijsten van zichzelf.
        """),
        L.task("Staat een soort al in `counts`, tel er dan 1 bij op; zet hem anders op 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_kinds", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_kinds", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_kinds",
        [
            "Elke keer dat een soort in de loop langskomt, moet zijn aantal één omhoog. Een soort die voor het eerst langskomt, staat nog niet in `counts`.",
            "`kind in counts` is `True` als de soort er al staat. Gebruik dat in een `if`, met een `else` voor een nieuwe soort.",
            "Vervang de regel `counts[kind] = 1` door `if kind in counts:` met `counts[kind] = ___` eronder, en dan `else:` met `counts[kind] = 1` eronder.",
        ],
        why="""```python
def count_kinds(notes):
    counts = {}
    for kinds in notes:
        for kind in kinds:
            if kind in counts:
                counts[kind] = counts[kind] + 1
            else:
                counts[kind] = 1
    return counts
```

Dit is de failure taxonomy in getallen. Een antwoord met twee soorten telt voor elk één keer, dus de aantallen komen samen hoger uit dan het aantal antwoorden met een fout.""",
    )
    return


@app.cell(hide_code=True)
def text_biggest():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De grootste soort

        `biggest(counts)` moet de soort met het hoogste aantal geven. `COUNTS` bevat de aantallen van de course, de uitkomst van oefening 1. Als twee soorten hetzelfde aantal hebben, moet de eerste in `counts` eruit komen. De begincode geeft de laatste soort, wat zijn aantal ook is.
        """),
        L.task("Laat een soort alleen `best` worden als zijn aantal hoger is dan `most`, en maak `most` dan zijn aantal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_biggest():
    box_biggest = L.code_box("biggest", lines=8)
    box_biggest
    return (box_biggest,)


@app.cell(hide_code=True)
def check_biggest(box_biggest):
    result_biggest = L.check("biggest", box_biggest.value)
    result_biggest
    return (result_biggest,)


@app.cell(hide_code=True)
def hints_biggest():
    L.hints(
        "biggest",
        [
            "Loop de soorten langs en onthoud de beste tot nu toe, met zijn aantal in `most`. Een soort verslaat hem alleen als zijn aantal hoger is.",
            "In de loop: `if counts[kind] > most:`, en daaronder de twee regels die deze soort de beste tot nu toe maken.",
            "Onder de `if`: `best = ___` en `most = ___`, allebei één stap verder in dan de `if`.",
        ],
        why="""```python
def biggest(counts):
    best = None
    most = 0
    for kind in counts:
        if counts[kind] > most:
            best = kind
            most = counts[kind]
    return best
```

Met `>` neemt een latere soort met hetzelfde aantal niet de plaats van de eerste in. Voor de notities van de course is de grootste soort `"wrong-topic"`.""",
    )
    return


@app.cell(hide_code=True)
def text_which():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Welke soort?

        Nog één antwoord, zonder de notitie van de course:
        """),
        mo.Html(C.answer_card(C.WHICH, L.lang, note=False)),
    ])
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-kind",
        question="Bij welke soort hoort dit antwoord?",
        options={
            "elsewhere": "stuurt door",
            "made-up": "verzonnen",
            "other-problem": "ander probleem",
        },
        correct="made-up",
        why={
            "made-up": "De feiten zeggen niets over een kaart terugvinden, en de app kan dat niet: het antwoord belooft wat de bank niet kan, en vraagt in een chat om een kaartnummer. De feiten beantwoorden deze vraag niet, dus het antwoord had moeten zeggen dat iemand van het team zal reageren.",
            "elsewhere": "Het stuurt de klant nergens heen: het belooft de kaart zelf te zoeken. Dat staat niet in de feiten: *verzonnen*.",
            "other-problem": "Het vraagt niet om wat bij een ander probleem hoort, zoals de datum en het bedrag van een betaling. Het belooft wat niet in de feiten staat: *verzonnen*.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_fix():
    _f = C.findings()
    mo.md(r"""
    ### Oefening 4. Wat eerst?

    De grootste soort is het verkeerde onderwerp, met {t} antwoorden; {both} daarvan vroegen ook om wat bij een ander probleem hoort.
    """.format(t=_f["counts"]["wrong-topic"], both=_f["topic_and_other"]))
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    _f = C.findings()
    choice_fix = L.choice(
        "fix-first",
        question="Welke wijziging probeer je eerst?",
        options={
            "admit": "Voeg een zin toe: zeg nooit dat een betaling fout was voordat het team die controleerde",
            "json": "Vraag om JSON mode, zodat elk antwoord geldige JSON is",
            "topics": "Zet elk feit naast zijn onderwerp, zodat het model de feiten van één onderwerp neemt",
        },
        correct="topics",
        why={
            "topics": "{rr} van de {r} vragen over een terugbetaling en {pr} van de {p} over een geblokkeerde pincode kregen het onderwerp van een foute wisselkoers, en hun antwoorden namen de feiten daarvan. In de system prompt koppelt niets een feit aan zijn onderwerp. Als je ze koppelt, pak je beide soorten tegelijk aan. Of het helpt, meet je: dat is de volgende les.".format(
                rr=_f["refunds_rate"], r=_f["refunds"], pr=_f["pins_rate"], p=_f["pins"]),
            "admit": "Dat pakt de soort *geeft een fout toe* aan, met {a} antwoorden: een echt probleem, maar een kleiner dan het verkeerde onderwerp, met {t}. Pak de grootste soort eerst aan.".format(
                a=_f["counts"]["admits"], t=_f["counts"]["wrong-topic"]),
            "json": "Alle {n} antwoorden waren al geldige JSON: JSON mode lost een probleem op dat de assistent niet heeft. Zonder te lezen had je misschien alleen dat gemeten, en de assistent klaar genoemd.".format(n=_f["n"]),
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_read():
    mo.md(r"""
    ### Lees de antwoorden zelf

    Kies een groep, en loop de antwoorden daarin een voor een langs, met de notitie van de course. Lees een paar van de verzonnen antwoorden: had jij elk bij dezelfde soort gezet?
    """)
    return


@app.cell(hide_code=True)
def settings_group():
    group_pick = mo.ui.dropdown(options=C.group_options(L.lang), value=list(C.group_options(L.lang))[0], label="Groep", full_width=True)
    group_pick
    return (group_pick,)


@app.cell(hide_code=True)
def settings_reply(group_pick):
    reply_pick = mo.ui.slider(start=1, stop=max(1, len(C.group(group_pick.value))), value=1, label="Antwoord", show_value=True)
    reply_pick
    return (reply_pick,)


@app.cell(hide_code=True)
def demo_read(group_pick, reply_pick):
    _numbers = C.group(group_pick.value)
    mo.Html(C.answer_card(_numbers[min(reply_pick.value, len(_numbers)) - 1], L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De soorten die je vond, zijn wat je vanaf nu meet. [Een eval set en regression tests](/course/open/build-eval-set?lang=nl) maakt van de soorten die code kan zien, zoals het verkeerde onderwerp, checks in code, en meet of een verbetering voor de grootste soort helpt. Een soort zoals *verzonnen* vraagt iemand die de betekenis leest: [Wie het antwoord beoordeelt](/course/open/build-judges?lang=nl) laat een model dat doen, en meet hoe vaak het het eens is met de notities van de course.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, choice_which, result_biggest, result_count):
    L.wrap_up(result_count, result_biggest, choice_which, choice_fix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
