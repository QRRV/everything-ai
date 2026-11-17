# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-judges. Route: course/route/lessen/bouwen.yaml. Checks, the judges' recorded verdicts and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-judges", "nl")
    C = L.checks   # the replies of version 1, what two judges wrote about them, the course's notes and the pictures
    words = C.words
    JUDGE_PLAIN = C.JUDGE_PLAIN
    JUDGE_RUBRIC = C.JUDGE_RUBRIC
    PEOPLE_GOOD = C.PEOPLE_GOOD
    PEOPLE_MADE_UP = C.PEOPLE_MADE_UP


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_alarm():
    quick_alarm = L.choice(
        "quick-alarm",
        question="admits_ok van de course keurde alle 18 antwoorden af die als *geeft een fout toe* waren genoteerd, en nog 21 andere. Waarom bleef hij uit de eval set?",
        options={
            "alarms": "Zijn woorden staan ook in antwoorden die niets toegeven",
            "missed": "Hij miste een aantal antwoorden die wel een fout toegaven",
            "slow": "Een check in code is te traag om na elke wijziging te draaien",
        },
        correct="alarms",
        why={
            "alarms": "*\"If the rate is incorrect, please provide the date and amount\"* geeft niets toe, en *\"blocked after 3 wrong attempts\"* gaat over de pincode. Of een antwoord een fout toegeeft, vraagt iemand die leest wat het betekent: daar gaat deze les over.",
            "missed": "Hij miste er geen: hij keurde alle 18 af. Zijn probleem was andersom, 21 keer vals alarm. Zie [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).",
            "slow": "Een check in code loopt in een oogwenk door alle 120 antwoorden. Zijn probleem was 21 keer vals alarm. Zie [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).",
        },
    )
    quick_alarm
    return (quick_alarm,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Van de zes soorten fouten die de course in de assistent van de bank vond, controleert de eval set er drie in code: *verkeerd onderwerp*, *ander probleem* en *te lang*. *Verzonnen*, in 23 antwoorden, en *geeft een fout toe*, in 18, vragen iemand die leest wat een antwoord betekent. De course las alle 120 antwoorden, maar niemand leest elk antwoord na elke wijziging, laat staan elke chat van echte klanten. Een model kan lezen, en snel: je geeft het het bericht van de klant, het antwoord en waar het op moet letten, en het schrijft een oordeel. Of dat oordeel klopt, weet je pas als je het meet.

    **Een voorbeeld uit het dagelijks leven.** Een docent laat een assistent de examens nakijken. Voordat de assistent alleen nakijkt, kijkt de docent dertig examens ook na en vergelijkt: keurt de assistent dezelfde af? Een assistent die elk examen afkeurt, is het bij elk slecht examen met de docent eens, en toch heb je er niets aan.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **LLM as a judge** | een model dat de antwoorden van een app beoordeelt: het krijgt de input, het antwoord en waar het op moet letten, en schrijft een oordeel, zoals pass of fail |
    | **rubric** | een beschrijving op papier van één ding om te beoordelen: wat telt als pass en wat als fail, soms met voorbeelden; mensen en judges beoordelen met dezelfde rubric |

    Drie kunnen een antwoord beoordelen: code, mensen en een judge. Een judge leest als een mens en is snel als code, maar zijn oordelen zijn ook een meting, en die controleer je tegen die van mensen, op antwoorden die mensen beoordeelden. Twee delen zeggen hoe een judge het doet: van de antwoorden die mensen afkeurden, het deel dat de judge ook afkeurde, en van de antwoorden die mensen goedkeurden, het deel dat de judge ook goedkeurde. Agreement, het deel van de antwoorden waarbij de twee hetzelfde oordeel geven, zet beide in één getal, en kan verbergen dat een van de twee nul is.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.graders_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een judge die je kunt vertrouwen

    1. **Criterium:** kies één ding om te beoordelen, en schrijf er een rubric voor.
    2. **Mensen:** beoordeel zelf een steekproef van antwoorden, met de rubric.
    3. **Judge:** laat het model dezelfde antwoorden beoordelen, met dezelfde rubric.
    4. **Lezen:** maak in code een oordeel van wat de judge schreef.
    5. **Meten:** het deel van de fails van mensen dat hij vangt en van de passes van mensen dat hij ook goedkeurt, naast de agreement. Gebruik de judge alleen als beide delen hoog zijn; verander anders de rubric of neem een groter model, en meet opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _p = _f["plain"]
    mo.vstack([
        mo.md(r"""
    ### Uitgewerkt voorbeeld

    De judge is Qwen3, het model van de assistent zelf, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt. Elk request bevat het bericht van de klant en het antwoord van versie 1.

    1. **Criterium:** eerst zonder rubric: is het antwoord goed?
    2. **Mensen:** een antwoord is goed als de notities van de course erbij leeg zijn; {pf} van de {n} antwoorden waren niet goed.
    3. **Judge:** deze system prompt:
    """.format(n=_f["n"], pf=_p["people_fail"])),
        mo.Html(C.E.prompt_html(C.PLAIN)),
        mo.md(r"""
    4. **Lezen:** hij antwoordde elke keer met één woord: {f} keer *fail*, {p} keer *pass*.
    5. **Meten:** hij keurde {c} van de {pf} antwoorden af die de course afkeurde, en keurde {pp} van de {ppass} goed die de course goedkeurde. Zijn agreement is {agree:.0%}: hij keurt bijna elk antwoord af, goed of fout.
    """.format(n=_f["n"], pf=_p["people_fail"], f=_p["fail"], p=_p["pass"], c=_p["caught"], pp=_p["passed"],
               ppass=_p["people_pass"], agree=_p["agree"])),
    ])
    return


@app.cell(hide_code=True)
def text_measured():
    _r = C.findings()["rubric"]
    mo.vstack([
        mo.md(r"""
    ### Wat de course mat

    Daarna één criterium met een rubric, voor de soort *verzonnen*:

    """),
        mo.Html(C.E.prompt_html(C.RUBRIC)),
        mo.md(r"""
    De rubric vraagt eerst een zin en het oordeel als laatste; in {nl} van de {n} oordelen kwam het oordeel eerst, vóór een eigen zin, en in de andere schreef hij alleen het oordeel. Hij schreef elke keer *pass*, ook bij de {pf} antwoorden die de course als *verzonnen* noteerde.
    """.format(nl=_r["not_last"], n=C.findings()["n"], pf=_r["people_fail"])),
    ])
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.measured_picture(L.lang)),
        L.watch_out("Geen van beide judges kon een goed antwoord van een fout onderscheiden: de een keurt bijna alles af, de ander keurt alles goed. Een model zo klein als Qwen3-0.6B is hier geen judge voor. Een groter model beoordeelt vaak beter, en dat meet je op dezelfde manier, op antwoorden die mensen beoordeelden; tot een judge die test doorstaat, beoordelen mensen."),
    ])
    return


@app.cell(hide_code=True)
def text_verdict():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het oordeel lezen

        Een judge schrijft niet altijd één woord: de judge met een rubric schreef *pass. All facts are included in the reply.* `verdict(text)` moet `True` geven voor pass en `False` voor fail, op grond van het eerste woord van de tekst dat pass of fail is, en `None` als geen van beide voorkomt. `words(text)` geeft de woorden van een tekst in kleine letters, zonder leestekens. De startcode leest alleen een tekst die precies `pass` of `fail` is.
        """),
        L.task("Loop door `words(text)`, en geef het oordeel van het eerste woord dat pass of fail is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=7)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Bekijk de woorden een voor een, en stop bij het eerste dat pass of fail is. Pas als geen enkel woord dat was, geeft de functie `None`.",
            "`for word in words(text):` met de twee `if`-regels van de startcode erin, die `word` vergelijken in plaats van `text`. De `return None` komt na de loop.",
            "Schrijf boven de twee `if`-regels `for word in ___:`, schuif de twee `if`-regels een stap in, en vergelijk in allebei `word` in plaats van `text`. Houd `return None` aan het eind, niet in de loop.",
        ],
        why="""```python
def verdict(text):
    for word in words(text):
        if word == "pass":
            return True
        if word == "fail":
            return False
    return None
```

`None` houdt apart wat de judge niet zei: een beoordeling zonder oordeel is geen pass en geen fail.

Dit leest de eerste pass of fail, en dat past bij deze judges: ze zetten het oordeel vooraan. Een judge die de rubric volgt, schrijft eerst zijn zin, en daarin kan het woord staan, zoals in *nothing in it would make it fail*; lees bij zo'n judge de laatste pass of fail. Welke past, zie je door te lezen wat je judge schrijft.""",
    )
    return


@app.cell(hide_code=True)
def text_agreement():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Agreement

        `JUDGE_PLAIN` bevat de oordelen van de eenvoudige judge, één voor elk van de 120 antwoorden, en `PEOPLE_GOOD` die van de course: `True` voor een antwoord zonder enige fout. `agreement(judge, people)` moet het deel van de antwoorden geven waarbij de twee oordelen hetzelfde zijn. De startcode telt elk antwoord.
        """),
        L.task("Tel een antwoord alleen als `judge[i]` hetzelfde is als `people[i]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agreement():
    box_agreement = L.code_box("agreement", lines=6)
    box_agreement
    return (box_agreement,)


@app.cell(hide_code=True)
def check_agreement(box_agreement):
    result_agreement = L.check("agreement", box_agreement.value)
    result_agreement
    return (result_agreement,)


@app.cell(hide_code=True)
def hints_agreement():
    L.hints(
        "agreement",
        [
            "De twee zijn het over een antwoord eens als ze het hetzelfde oordeel geven: allebei pass, of allebei fail.",
            "Zet een `if` om `same = same + 1` die `judge[i]` vergelijkt met `people[i]`.",
            "Schrijf boven `same = same + 1` de regel `if ___ == ___:`, en schuif `same = same + 1` een stap in.",
        ],
        why="""```python
def agreement(judge, people):
    same = 0
    for i in range(len(people)):
        if judge[i] == people[i]:
            same = same + 1
    return same / len(people)
```

De eenvoudige judge is het over meer dan de helft van de antwoorden eens, terwijl hij nooit een goed antwoord goedkeurt: agreement alleen zegt je dat niet.""",
    )
    return


@app.cell(hide_code=True)
def text_caught():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De fails die hij vangt

        `caught(judge, people)` moet van de antwoorden die mensen afkeurden het deel geven dat de judge ook afkeurde. Een oordeel `None` is geen fail. De startcode telt de antwoorden die mensen afkeurden in `failed`, maar telt `both` nooit. Het plaatje laat je deel zien naast het deel van de goede antwoorden dat elke judge goedkeurde.
        """),
        L.task("Tel het antwoord binnen de `if` bij `both` als de judge het ook afkeurde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_caught():
    box_caught = L.code_box("caught", lines=9)
    box_caught
    return (box_caught,)


@app.cell(hide_code=True)
def check_caught(box_caught):
    result_caught = L.check("caught", box_caught.value)
    result_caught
    return (result_caught,)


@app.cell(hide_code=True)
def hints_caught():
    L.hints(
        "caught",
        [
            "Hier tellen alleen de antwoorden die mensen afkeurden. Tel daarvan de antwoorden die de judge ook afkeurde.",
            "Een tweede `if` in de eerste: als `judge[i] == False`, tel 1 op bij `both`.",
            "Schrijf onder `failed = failed + 1`, even ver ingesprongen, `if ___:` met `both = both + 1` een stap verder in.",
        ],
        why="""```python
def caught(judge, people):
    failed = 0
    both = 0
    for i in range(len(people)):
        if people[i] == False:
            failed = failed + 1
            if judge[i] == False:
                both = both + 1
    return both / failed
```

`judge[i] == False` en niet `not judge[i]`: `not None` is `True`, en een beoordeling zonder oordeel is geen fail.""",
    )
    return


@app.cell(hide_code=True)
def text_what():
    _r = C.findings()["rubric"]
    mo.md(r"""
    ### Oefening 4. Wat zegt het je?

    De judge met een rubric is het bij {agree:.0%} van de antwoorden met de course eens.
    """.format(agree=_r["agree"]))
    return


@app.cell(hide_code=True)
def choice_what_cell():
    _f = C.findings()
    _r = _f["rubric"]
    choice_what = L.choice(
        "what-agreement",
        question="Wat zegt die agreement je?",
        options={
            "good": "Hij vangt de meeste antwoorden die iets verzinnen",
            "little": "Weinig: hij keurde alle 120 goed, ook de 23 verzonnen",
            "fifth": "Hij keurt ongeveer één op de vijf antwoorden af, net als mensen",
        },
        correct="little",
        why={
            "little": "{pp} van de {n} antwoorden verzinnen niets, en een judge die alles goedkeurt, is het over al die antwoorden eens: {agree:.0%} zonder een van de {pf} te vangen. Lees het deel dat hij vangt naast de agreement.".format(
                pp=_r["people_pass"], n=_f["n"], agree=_r["agree"], pf=_r["people_fail"]),
            "good": "Hij ving er {c} van de {pf}: hij schreef pass bij alle {n} antwoorden. De {agree:.0%} zijn de {pp} antwoorden die niets verzinnen, en bij die antwoorden zit een judge die altijd goedkeurt goed.".format(
                c=_r["caught"], pf=_r["people_fail"], n=_f["n"], agree=_r["agree"], pp=_r["people_pass"]),
            "fifth": "Hij keurde er geen af: hij schreef pass bij alle {n} antwoorden. Mensen keurden er {pf} af, en precies daar is hij het oneens.".format(
                n=_f["n"], pf=_r["people_fail"]),
        },
    )
    choice_what
    return (choice_what,)


@app.cell(hide_code=True)
def text_read():
    mo.md(r"""
    ### Lees de judges zelf

    Kies een judge, en loop de antwoorden een voor een door: het antwoord, de notities van de course, wat de judge schreef en het oordeel dat daaruit is gelezen.
    """)
    return


@app.cell(hide_code=True)
def settings_judge():
    judge_pick = mo.ui.dropdown(options={"eenvoudige judge": "plain", "judge met een rubric": "rubric"}, value="eenvoudige judge", label="Judge", full_width=True)
    judge_pick
    return (judge_pick,)


@app.cell(hide_code=True)
def settings_reply():
    reply_pick = mo.ui.slider(start=1, stop=120, value=1, label="Antwoord", show_value=True)
    reply_pick
    return (reply_pick,)


@app.cell(hide_code=True)
def demo_read(judge_pick, reply_pick):
    mo.Html(C.judged_card(judge_pick.value, reply_pick.value - 1, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De eval set beoordeelt wat code kan zien, en een judge kan, als hij gemeten is, beoordelen wat code niet ziet. [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl) voegt toe wat je van echte gebruikers leert: wat zij melden, wordt nieuwe input en nieuwe beoordeelde antwoorden. In [Guardrails](/course/open/build-guardrails?lang=nl) draaien zulke checks terwijl de app antwoordt, voordat een antwoord bij een klant komt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_what, result_agreement, result_caught, result_verdict):
    L.wrap_up(result_verdict, result_agreement, result_caught, choice_what)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
