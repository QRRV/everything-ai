# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-eval-set. Route: course/route/lessen/bouwen.yaml. Checks, the two versions, the course's checks and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-eval-set", "nl")
    C = L.checks   # the eval set, both versions of the bank's assistant, the course's checks and the pictures
    reply_of = C.reply_of
    RATE = C.RATE
    PASSED_V1 = C.passed("v1")
    PASSED_V2 = C.passed("v2")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Code nakijken die AI schreef](/course/open/build-check-ai-code?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="Alle acht count_vowels van Qwen3 telden alleen kleine letters. Welke test vangt die bug?",
        options={
            "banana": 'assert count_vowels("banana") == 3',
            "empty": 'assert count_vowels("") == 0',
            "apple": 'assert count_vowels("Apple") == 2',
        },
        correct="apple",
        why={
            "apple": "De bug wordt alleen zichtbaar met een hoofdletter: \"Apple\" heeft twee klinkers, en een functie die alleen kleine letters telt, geeft 1. Een test vangt een bug alleen als zijn geval de bug zichtbaar maakt.",
            "banana": "Al zijn letters zijn klein: een functie die alleen kleine letters telt, geeft ook 3, dus de test slaagt met de bug. Zie [Code nakijken die AI schreef](/course/open/build-check-ai-code?lang=nl).",
            "empty": "Een lege tekst heeft helemaal geen letters: de test slaagt met de bug. Zie [Code nakijken die AI schreef](/course/open/build-check-ai-code?lang=nl).",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl) was het verkeerde onderwerp de grootste soort fout van de assistent van de bank. Je verandert de system prompt om dat te verhelpen. Hielp de wijziging? En brak hij iets wat werkte? Alle 120 antwoorden opnieuw lezen kost een uur, en je verandert de prompt nog vaak. Checks in code antwoorden in een seconde, en elke keer op dezelfde manier.

    **Een voorbeeld uit het dagelijks leven.** Na een reparatie aan de remmen loopt een garage dezelfde lijst na als na elke reparatie: lampen, toeter, ruitenwissers, remmen. De lijst laat zien dat de remmen nu werken, en ook dat de reparatie niets anders kapotmaakte, zoals een kabel van de lampen die los is geraakt.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **eval set** | een vaste lijst inputs, elk met wat je verwacht, zoals het onderwerp waar ze over gaan, en checks die je na elke wijziging draait op de antwoorden van je app |
    | **assertion-based test** | een check in code die ja of nee zegt over één antwoord, zoals: is het geldige JSON, is het onderwerp het onderwerp dat de eval set verwacht, heeft het hoogstens drie zinnen |
    | **regression test** | de eval set draaien op de oude en de nieuwe versie van je app; hij faalt als een input die slaagde met de oude versie faalt met de nieuwe, een *regression* |

    Een check in code ziet alleen wat code kan zien: de vorm, een woord, de lengte, een onderwerp dat je van tevoren weet. Probeer een check uit op de antwoorden die je al las, voordat je hem vertrouwt: hij moet afkeuren wat je voor zijn soort noteerde, en verder niets. Een check die een goed antwoord afkeurt, geeft vals alarm; een check die een fout antwoord doorlaat, mist het.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een eval set

    1. **Inputs:** een vaste lijst inputs, elk met wat je verwacht. Houd de lijst van versie tot versie hetzelfde.
    2. **Checks:** voor elke soort fout die code kan zien, een functie die voor één antwoord `True` of `False` geeft.
    3. **De checks testen:** draai ze op de antwoorden die je las, en vergelijk met je notities.
    4. **Draaien:** de eval set op de oude versie en op de nieuwe.
    5. **Vergelijken:** wat de nieuwe versie verbeterde, en zijn regressions. Lees de regressions voordat je de nieuwe versie houdt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    1. **Inputs:** de {n} vragen van de error analysis, elk met het onderwerp waar ze in BANKING77 over gaan.
    2. **Checks:** één voor de vorm, en één voor elke soort die code kan zien: *verkeerd onderwerp*, *ander probleem* en *te lang*. Deze is voor *te lang*:

    ```python
    def short_ok(text, topic):
        answer = reply_of(text)
        if answer is None:
            return False
        return sentences(answer["reply"]) <= 3
    ```

    Elke check krijgt dezelfde twee dingen: `text`, het antwoord van de app, en `topic`, het onderwerp dat de eval set verwacht. `reply_of(text)` geeft het antwoord als dictionary met `"topic"` en `"reply"`, of `None` als de tekst geen JSON van die vorm is; `sentences` telt de zinnen van een tekst. `json_ok` geeft alleen `False` als `reply_of` `None` geeft.

    3. **De checks testen:** op de {n} antwoorden van versie 1, naast de notities van de course:
    """.format(n=C.findings()["n"]))
    return


@app.cell(hide_code=True)
def picture_tested():
    mo.Html(C.tested_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_tested():
    _f = C.findings()
    _a = _f["notes"]["admits_ok"]
    _s = _f["notes"]["steps_ok"]
    mo.md(r"""
    `topic_ok` en `short_ok` keuren precies de antwoorden af die de course voor hun soort noteerde. `steps_ok` keurt {sc} van de {sn} antwoorden af die als *ander probleem* genoteerd zijn, en geen ander antwoord. De {sm} die hij mist, vragen in andere woorden om de stappen van een ander probleem, zoals een kaart bevriezen die nog niet is aangekomen; een check die naar vaste woorden zoekt, ziet dat niet. Een check die er een paar mist maar geen vals alarm geeft, verdient toch zijn plek: wat hij afkeurt, is fout.

    De course probeerde ook een check voor de soort *geeft een fout toe*: een antwoord faalt als het het woord *incorrect*, *wrong* of *too low* bevat. Hij keurt alle {caught} genoteerde antwoorden af, en nog {alarms} andere: *"If the rate is incorrect, please provide the date and amount"* geeft niets toe, en *"blocked after 3 wrong attempts"* gaat over de pincode. Of een antwoord een fout toegeeft, vraagt iemand die de betekenis leest, dus blijft `admits_ok` uit de eval set.
    """.format(caught=len(_a["caught"]), alarms=len(_a["alarms"]), sc=len(_s["caught"]), sn=len(_s["noted"]), sm=len(_s["missed"])))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.vstack([
        mo.md(r"""
        4. **Draaien:** versie 2 van de system prompt zet elk feit naast zijn onderwerp, en vraagt het model eerst het onderwerp te kiezen:
        """),
        mo.Html(C.E.prompt_html(C.SYSTEM_V2)),
        mo.md(r"""
        De eval set op beide versies:
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_versions():
    mo.Html(C.versions_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_compare():
    _f = C.findings()
    mo.md(r"""
    5. **Vergelijken:** met versie 1 kwamen {all1} van de {n} inputs door elke check, met versie 2 {all2}. Wat de wijziging moest verbeteren, werd ook beter: {r2} van de {rn} vragen over een terugbetaling en {p2} van de {pn} over een geblokkeerde pincode kregen het goede onderwerp, tegen {r1} en {p1} met versie 1. Toch kwamen er met versie 1 {t1} inputs door `topic_ok` en met versie 2 {t2}: hetzelfde aantal, maar niet dezelfde inputs.
    """.format(n=_f["n"], all1=_f["v1"]["all"], all2=_f["v2"]["all"], r1=_f["right"]["v1"]["request_refund"], r2=_f["right"]["v2"]["request_refund"],
               p1=_f["right"]["v1"]["pin_blocked"], p2=_f["right"]["v2"]["pin_blocked"], rn=C.LABELS.count("request_refund"),
               pn=C.LABELS.count("pin_blocked"), t1=_f["v1"]["topic_ok"], t2=_f["v2"]["topic_ok"]))
    return


@app.cell(hide_code=True)
def text_topic():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De check voor het onderwerp

        `topic_ok(text, topic)` moet `True` geven als het antwoord JSON van de goede vorm is en het onderwerp `topic` is, het onderwerp dat de eval set verwacht. De startcode geeft `True` voor elk antwoord dat JSON is. Het plaatje onder de check laat zien wat je check over elk antwoord van versie 1 zegt.
        """),
        L.task("Geef of het onderwerp van `answer` hetzelfde is als `topic`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_topic():
    box_topic = L.code_box("topic_ok", lines=6)
    box_topic
    return (box_topic,)


@app.cell(hide_code=True)
def check_topic(box_topic):
    result_topic = L.check("topic_ok", box_topic.value)
    result_topic
    return (result_topic,)


@app.cell(hide_code=True)
def hints_topic():
    L.hints(
        "topic_ok",
        [
            "`answer` is een dictionary met de keys `\"topic\"` en `\"reply\"`. De check moet er één van vergelijken met `topic`.",
            "`answer[\"topic\"]` is het onderwerp dat het model koos. Vergelijk het met `==`.",
            "Vervang `return True` door `return ___ == topic`, met op de plek van het gat het onderwerp dat het model koos.",
        ],
        why="""```python
def topic_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return answer["topic"] == topic
```

De check kent het goede onderwerp omdat de eval set het voor elke input geeft: zonder wat je verwacht, kan code een goed onderwerp niet van een fout onderscheiden.""",
    )
    return


@app.cell(hide_code=True)
def text_steps():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stappen van een ander probleem

        Veel antwoorden van versie 1 vroegen om de datum en het bedrag van een betaling, de stappen voor een foute wisselkoers, terwijl de klant iets anders vroeg. `steps_ok(text, topic)` moet `False` geven voor een antwoord waarvan `"reply"` de tekst `"date and amount"` bevat, tenzij `topic` in `RATE` staat, de twee onderwerpen over wisselkoersen waar die stappen passen. Voor een tekst die geen JSON is, moet hij ook `False` geven. `a or b` is `True` als ten minste een van de twee `True` is, en `x not in text` is `True` als `x` niet in `text` voorkomt.
        """),
        L.task("Geef `True` als `\"reply\"` de tekst \"date and amount\" niet bevat, of als `topic` in `RATE` staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_steps():
    box_steps = L.code_box("steps_ok", lines=6)
    box_steps
    return (box_steps,)


@app.cell(hide_code=True)
def check_steps(box_steps):
    result_steps = L.check("steps_ok", box_steps.value)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps_ok",
        [
            "Twee dingen kunnen een antwoord laten slagen: de reply vraagt niet om de datum en het bedrag, of de vraag gaat over wisselkoersen. Eén van de twee is genoeg.",
            "`\"date and amount\" not in answer[\"reply\"]` is het eerste; `topic in RATE` is het tweede. Verbind ze met `or`.",
            "Vervang `return True` door `return \"date and amount\" ___ answer[\"reply\"] or topic ___ RATE`.",
        ],
        why="""```python
def steps_ok(text, topic):
    answer = reply_of(text)
    if answer is None:
        return False
    return "date and amount" not in answer["reply"] or topic in RATE
```

Gebruik `topic`, niet `answer["topic"]`: als het model voor een terugbetaling een foute wisselkoers koos, passen de stappen bij zijn eigen onderwerp, en alleen de eval set weet dat de vraag over een terugbetaling ging.""",
    )
    return


@app.cell(hide_code=True)
def text_regressions():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De regressions

        `PASSED_V1` en `PASSED_V2` bevatten voor elk van de 120 inputs `True` als het antwoord door elke check van de eval set komt, met versie 1 en met versie 2. `regressions(old, new)` moet de nummers geven van de inputs die slaagden met `old` en falen met `new`. De startcode geeft elke input die faalt met `new`: `not new[i]` is `True` als `new[i]` `False` is.
        """),
        L.task("Houd alleen de inputs over die ook slaagden met `old`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_regressions():
    box_regressions = L.code_box("regressions", lines=6)
    box_regressions
    return (box_regressions,)


@app.cell(hide_code=True)
def check_regressions(box_regressions):
    result_regressions = L.check("regressions", box_regressions.value)
    result_regressions
    return (result_regressions,)


@app.cell(hide_code=True)
def hints_regressions():
    L.hints(
        "regressions",
        [
            "Een input die met `old` al faalde, is geen regression: hij was nooit goed. Alleen een input die van slagen naar falen ging, telt.",
            "De `if` moet twee dingen tegelijk vragen: `old[i]` is `True`, en `new[i]` niet. Verbind ze met `and`.",
            "Maak van de `if`-regel `if old[i] ___ not new[i]:`.",
        ],
        why="""```python
def regressions(old, new):
    found = []
    for i in range(len(old)):
        if old[i] and not new[i]:
            found.append(i)
    return found
```

Een regression test faalt als deze lijst niet leeg is. Dan lees je die antwoorden voordat je de nieuwe versie houdt.""",
    )
    return


@app.cell(hide_code=True)
def text_keep():
    _f = C.findings()
    mo.md(r"""
    ### Oefening 4. Versie 2 houden?

    Versie 2 verbeterde {fixed} inputs en brak er {broke}, en elk van de {broke} regressions faalt bij `topic_ok`.
    """.format(fixed=len(_f["fixed"]), broke=len(_f["broke"])))
    return


@app.cell(hide_code=True)
def choice_keep_cell():
    _f = C.findings()
    _an = C.LABELS.count("card_arrival")
    choice_keep = L.choice(
        "keep-v2",
        question="Wat doe je met versie 2?",
        options={
            "keep": "Houden: {a2} inputs komen door elke check, niet {a1}".format(a1=_f["v1"]["all"], a2=_f["v2"]["all"]),
            "read": "Eerst de {b} regressions lezen, dan beslissen".format(b=len(_f["broke"])),
            "drop": "Weggooien: een nieuwe versie mag niets breken",
        },
        correct="read",
        why={
            "read": "De regressions zijn {ba} vragen over wanneer een kaart aankomt en {br} over een foute wisselkoers. {al} van de {an} vragen over de aankomst van een kaart krijgen nu het onderwerp van een verloren of gestolen kaart, en {af} van hun antwoorden zeggen de klant de kaart te bevriezen; van de {wn} over een foute wisselkoers kregen er {w2} het goede onderwerp, tegen {w1} met versie 1. De verbetering voor terugbetalingen maakte nieuwe fouten: de regressions laten ze zien, de totalen verbergen ze. Pak ze aan in een versie 3, en draai de eval set opnieuw.".format(
                ba=_f["broke_arrival"], br=_f["broke_rate"], al=_f["arrival_lost"], an=_an, af=_f["arrival_freeze"],
                wn=C.LABELS.count("card_payment_wrong_exchange_rate"), w2=_f["right"]["v2"]["card_payment_wrong_exchange_rate"],
                w1=_f["right"]["v1"]["card_payment_wrong_exchange_rate"]),
            "keep": "Het totaal ging {up} omhoog, maar {b} inputs die goed waren, zijn nu fout: {al} van de {an} vragen over wanneer een kaart aankomt, krijgen het onderwerp van een verloren of gestolen kaart. Een totaal kan verbergen wat er brak.".format(
                up=_f["v2"]["all"] - _f["v1"]["all"], b=len(_f["broke"]), al=_f["arrival_lost"], an=_an),
            "drop": "Versie 2 gaf {r2} van de {rn} vragen over een terugbetaling het goede onderwerp, tegen {r1}. Met weggooien raak je dat ook kwijt: bijna elke wijziging breekt iets, en de regression test zegt je wat. Lees wat er brak, en pak dat aan in de volgende versie.".format(
                r2=_f["right"]["v2"]["request_refund"], r1=_f["right"]["v1"]["request_refund"], rn=C.LABELS.count("request_refund")),
        },
    )
    choice_keep
    return (choice_keep,)


@app.cell(hide_code=True)
def text_read():
    mo.md(r"""
    ### Vergelijk de versies zelf

    Kies een groep, en loop de inputs een voor een door, met het antwoord van elke versie en wat elke check erover zegt.
    """)
    return


@app.cell(hide_code=True)
def settings_group():
    group_pick = mo.ui.dropdown(options=C.group_options(L.lang), value=list(C.group_options(L.lang))[0], label="Groep", full_width=True)
    group_pick
    return (group_pick,)


@app.cell(hide_code=True)
def settings_input(group_pick):
    input_pick = mo.ui.slider(start=1, stop=max(1, len(C.group(group_pick.value))), value=1, label="Input", show_value=True)
    input_pick
    return (input_pick,)


@app.cell(hide_code=True)
def demo_compare(group_pick, input_pick):
    _numbers = C.group(group_pick.value)
    mo.Html(C.compare_card(_numbers[min(input_pick.value, len(_numbers)) - 1], L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Checks in code dekken de soorten die code kan zien. Voor *verzonnen* en *geeft een fout toe* laat [Wie het antwoord beoordeelt](/course/open/build-judges?lang=nl) een model de antwoorden beoordelen, en meet het tegen de notities van de course, zoals jij je checks mat. [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl) voegt toe wat er met echte gebruikers gebeurt: wat zij melden, wordt nieuwe input voor de eval set.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keep, result_regressions, result_steps, result_topic):
    L.wrap_up(result_topic, result_steps, result_regressions, choice_keep)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
