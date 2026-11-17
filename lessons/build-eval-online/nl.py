# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-eval-online. Route: course/route/lessen/bouwen.yaml. Checks, de notities bij versie 2, de week chats en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-eval-online", "nl")
    C = L.checks   # versie 1 en 2 van de assistent van de bank, de notities van de course bij beide, een week chats, en de plaatjes
    WEEK = C.WEEK
    WEEK_2 = C.WEEK_2   # weeks with other coin flips, which the checks also try
    WEEK_3 = C.WEEK_3


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
def quick_check_regression():
    quick_regression = L.choice(
        "quick-regression",
        question="Wat laat een regression test tussen twee versies zien?",
        options={
            "total": "Hoeveel inputs er meer doorkomen dan met de oude versie",
            "broke": "De inputs die eerst goed waren en nu fout zijn",
            "both": "De inputs die met beide versies fout zijn",
        },
        correct="broke",
        why={
            "broke": "Een regressie is een input die van goed naar fout ging. Het totaal kan stijgen terwijl er inputs breken: de regression test laat zien welke.",
            "total": "Het totaal kan stijgen terwijl er inputs breken. Een regression test noemt de inputs die met de oude versie goed waren en met de nieuwe fout zijn. Zie [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).",
            "both": "Die waren al fout. Een regressie is een input die met de oude versie goed was en met de nieuwe fout is. Zie [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).",
        },
    )
    quick_regression
    return (quick_regression,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    In [Een eval set en regression tests](/course/open/build-eval-set?lang=nl) doorstond versie 2 van de assistent van de bank elke check bij {e2} van de {n} inputs, tegen {e1} bij versie 1. Daarna las de course ook de antwoorden van versie 2, met de regels van [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl): versie 2 verzon iets in {m2} antwoorden, versie 1 in {m1}. De eval set heeft geen check voor *verzonnen*, dus kon dat niet zien. Klanten merken het wel: naar een verzonnen termijn handelen ze.

    Geen eval set bevat elk bericht dat klanten zullen sturen, of weet wat ze met een antwoord doen. Dat merk je pas bij echte klanten, terwijl de app draait, en ook dan kun je er maar een deel van meten.

    **Een voorbeeld uit het dagelijks leven.** Een restaurant probeert een nieuw recept eerst in de eigen keuken: de koks proeven het. Daarna serveert het het nieuwe gerecht aan de helft van de tafels en het oude aan de rest, en kijkt wat er op de borden terugkomt. Een paar gasten vullen het kaartje op tafel in; de meeste zeggen niets.
    """.format(n=_f["n"], e1=_f["eval"]["v1"], e2=_f["eval"]["v2"], m1=_f["kinds"]["v1"]["made-up"], m2=_f["kinds"]["v2"]["made-up"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **offline evaluation** | een wijziging testen op een vaste verzameling voorbeelden voordat gebruikers haar krijgen, zoals met een eval set |
    | **online evaluation** | meten hoe de app het doet bij echte gebruikers terwijl hij draait: waar ze op klikken, of hun taak gedaan is, wat elke chat kostte |
    | **feedback van gebruikers** | wat gebruikers je over een antwoord vertellen: bewust, zoals een duim omhoog of omlaag of een opmerking, of door wat ze doen, zoals opnieuw vragen of om een mens vragen |
    | **A/B test** | elke gebruiker krijgt bij toeval versie A of versie B, zoals met een muntworp, en je vergelijkt hetzelfde getal voor de twee groepen |
    | **data flywheel** | de chats die misgingen worden gelezen en worden nieuwe inputs voor de eval set, zodat elke volgende versie erop getest wordt |

    Wat een versie waard is, zie je aan wat hij voor het bedrijf doet: hoeveel klanten hun taak gedaan kregen, en wat dat kostte. Een versie die per chat minder kost maar minder klanten helpt, kan per gedane taak meer kosten.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.loop_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: van offline naar online

    1. **Offline:** draai de eval set op de nieuwe versie. Met regressies die je niet kunt accepteren, gaat hij niet verder.
    2. **A/B test:** geef de nieuwe versie aan klanten die bij toeval gekozen zijn, en de oude aan de rest.
    3. **Meten:** voor elke versie het aandeel duimen omhoog, het aandeel gedane taken, en de kosten per gedane taak.
    4. **Terugvoeren:** lees de chats die misgingen, vooral die de eval set doorliet, en voeg ze toe aan de eval set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De bank heeft geen klanten, dus speelt de course ze. De {n} testberichten komen binnen als {n} chats in één week, en een muntworp geeft elke klant versie 1 of versie 2. Een klant is geholpen als de notitie van de course bij het antwoord aan die klant geen *ander probleem*, niets *verzonnen* en geen *stuurt door* heeft; een verkeerd onderwerpveld ziet de klant nooit, een te lang antwoord helpt ook, en dat de bank een fout toegeeft, vindt de klant niet erg. Een taak telt als gedaan als het antwoord hielp. Eén klant op de {t}, bij toeval gekozen, klikt op een duim: omhoog na een antwoord dat hielp, omlaag na een antwoord dat niet hielp.

    Deze gespeelde klanten kennen de feiten, dus merken ze elk verzonnen antwoord op, en ze klikken bij toeval. Echte klanten zien vaak niet dat een antwoord verzonnen is, en of ze klikken, kan afhangen van hoe de chat liep. De week laat zien hoe de getallen van een online evaluation werken, niet hoe echte klanten zich gedragen.

    `WEEK` bevat de chats, elk een dictionary: `"message"`, `"version"` (`"v1"` of `"v2"`), `"reply"`, `"prompt_tokens"` en `"output_tokens"`, `"seconds"`, `"thumb"` (`"up"`, `"down"` of `None` als de klant nergens op klikte), `"done"` (`True` als de taak gedaan is) en `"passed"` (`True` als het antwoord elke check van de eval set doorstond).
    """.format(n=len(WEEK), t=C.THUMB))
    return


@app.cell(hide_code=True)
def picture_week():
    mo.Html(C.week_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_read():
    _f = C.findings()
    mo.md(r"""
    1. **Wie wat kreeg:** de muntworpen gaven {w1} klanten versie 1 en {w2} versie 2.
    2. **Feedback:** {t} klanten klikten op een duim, {u} ervan omhoog: {up}% omhoog. De taak was gedaan in {d} van de {n} chats, {dp}%.
    3. **Kosten:** met de verzonnen prijzen van [Tokens en prijzen: tel voordat je verstuurt](/course/open/build-tokens-prices?lang=nl), {pi} en {po} dollar voor een miljoen tokens erin en eruit, kostten 1.000 chats van versie 1 {c1} dollar en 1.000 van versie 2 {c2}: de antwoorden van versie 2 zijn korter.
    """.format(w1=_f["week_n"]["v1"], w2=_f["week_n"]["v2"], t=_f["thumbs"], u=_f["ups"], up=C.pct(_f["up_share"]),
               d=sum(c["done"] for c in WEEK), n=_f["n"], dp=C.pct(_f["done_all"]), pi=C.PRICE_IN, po=C.PRICE_OUT,
               c1=C.money(_f["cost_1000"]["v1"], L.lang), c2=C.money(_f["cost_1000"]["v2"], L.lang)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _k = _f["kinds"]
    mo.md(r"""
    ### Wat de course mat

    De notities bij versie 2 wijzen twee kanten op: *ander probleem* ging van {o1} antwoorden naar {o2} en *geeft een fout toe* van {a1} naar {a2}, maar *verzonnen* ging van {m1} naar {m2}. Als elke klant versie 1 kreeg, zou die {h1} van de {n} helpen; versie 2 {h2}. Dat verschil rust op weinig antwoorden: {only1} antwoorden hielpen alleen met versie 1, {only2} alleen met versie 2. Op deze {n} berichten zijn de twee ongeveer gelijk, en hoe de course een paar antwoorden noteerde, kan het verschil omdraaien.

    In de week die de course speelde, was de taak gedaan voor {d1}% van de klanten met versie 1 en {d2}% met versie 2. Andere muntworpen geven andere weken: in {r} weken lag versie 1 {a} keer voor en versie 2 {b} keer, met versie 2 ergens tussen {lo} punten slechter en {hi} punten beter. Met {n} klanten beslissen de muntworpen meer dan de versies: een A/B test heeft veel meer klanten nodig voordat een paar punten iets betekenen.

    Wat het waard is: 1.000 chats kosten {c1} dollar met versie 1 en {c2} met versie 2, maar 1.000 gedane taken kosten {k1} tegen {k2}, bijna hetzelfde. Versie 2 is goedkoper per chat, maar helpt {dh} klanten minder, dus per gedane taak is de besparing bijna weg.
    """.format(o1=_k["v1"]["other-problem"], o2=_k["v2"]["other-problem"], a1=_k["v1"]["admits"], a2=_k["v2"]["admits"],
               m1=_k["v1"]["made-up"], m2=_k["v2"]["made-up"], h1=_f["helped"]["v1"], h2=_f["helped"]["v2"], n=_f["n"],
               e1=_f["eval"]["v1"], e2=_f["eval"]["v2"], d1=C.pct(_f["done"]["v1"]), d2=C.pct(_f["done"]["v2"]), r=C.RUNS,
               a=_f["v1_ahead"], b=_f["v2_ahead"], lo=-_f["spread"][0], hi=_f["spread"][1], only1=_f["only"]["v1"], only2=_f["only"]["v2"], dh=_f["helped"]["v1"] - _f["helped"]["v2"],
               c1=C.money(_f["cost_1000"]["v1"], L.lang), c2=C.money(_f["cost_1000"]["v2"], L.lang),
               k1=C.money(_f["cost_done"]["v1"], L.lang), k2=C.money(_f["cost_done"]["v2"], L.lang)))
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.notes_picture(L.lang)),
        L.watch_out("Feedback en chats bevatten wat klanten schreven. Vertel ze dat je het gebruikt, bewaar het veilig, en laat een mens controleren wat er terug in de eval set gaat."),
    ])
    return


@app.cell(hide_code=True)
def text_up():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het aandeel duimen omhoog

        `up_share(chats)` moet het aandeel duimen omhoog geven onder de chats met een duim: de duimen omhoog gedeeld door de duimen omhoog en omlaag samen. Chats waarin de klant nergens op klikte, tellen niet mee. De startcode deelt door alle chats. Het plaatje onder de check zet je aandeel naast het aandeel gedane taken, voor de hele week en voor elke versie.
        """),
        L.task("Deel door de chats met een duim."),
    ])
    return


@app.cell(hide_code=True)
def exercise_up():
    box_up = L.code_box("up_share", lines=10)
    box_up
    return (box_up,)


@app.cell(hide_code=True)
def check_up(box_up):
    result_up = L.check("up_share", box_up.value)
    result_up
    return (result_up,)


@app.cell(hide_code=True)
def hints_up():
    L.hints(
        "up_share",
        [
            "De meeste klanten klikken nergens op. Een aandeel duimen omhoog is van de duimen, niet van de chats.",
            "De startcode telt `ups` en `downs` al. Deel door het aantal chats met een duim.",
            "Maak van de laatste regel `return ups / (___)`.",
        ],
        why="""```python
def up_share(chats):
    ups = 0
    downs = 0
    for chat in chats:
        if chat["thumb"] == "up":
            ups = ups + 1
        if chat["thumb"] == "down":
            downs = downs + 1
    return ups / (ups + downs)
```

Maar een deel van de klanten klikt, dus rust dit aandeel op weinig chats. Vergelijk het met het aandeel gedane taken in het plaatje.""",
    )
    return


@app.cell(hide_code=True)
def text_done():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Gedane taken voor één versie

        `done_share(chats, version)` moet van de chats die `version` kregen het aandeel geven waarin de taak gedaan is: `chat["done"]` is `True`. Dat is het getal dat een A/B test vergelijkt. De startcode telt de chats van beide versies.
        """),
        L.task("Tel alleen de chats van `version`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_done():
    box_done = L.code_box("done_share", lines=10)
    box_done
    return (box_done,)


@app.cell(hide_code=True)
def check_done(box_done):
    result_done = L.check("done_share", box_done.value)
    result_done
    return (result_done,)


@app.cell(hide_code=True)
def hints_done():
    L.hints(
        "done_share",
        [
            "Twee tellingen, allebei alleen voor de chats van één versie: hoeveel chats, en bij hoeveel daarvan de taak gedaan is.",
            "Zet om de twee tellingen in de loop een `if` die vraagt of de chat de versie kreeg die je telt.",
            "In de loop: `if ___ == version:`, met de `if` op `chat[\"done\"]` en `count = count + 1` een stap verder in.",
        ],
        why="""```python
def done_share(chats, version):
    done = 0
    count = 0
    for chat in chats:
        if chat["version"] == version:
            if chat["done"]:
                done = done + 1
            count = count + 1
    return done / count
```

Elke klant zag één versie, dus heeft elke versie zijn eigen groep. De muntworp maakt de twee groepen gelijk in alles behalve de versie, voor zover het toeval dat toelaat.""",
    )
    return


@app.cell(hide_code=True)
def text_passed_down():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wat de eval set miste

        `passed_down(chats)` moet de berichten geven van de chats met een duim omlaag die de eval set doorliet, in hun volgorde. Deze inputs heeft de eval set het hardst nodig: hij keurde het antwoord goed, en de klant gaf het een duim omlaag. De startcode geeft elke chat met een duim omlaag.
        """),
        L.task("Houd een chat alleen als de eval set hem ook doorliet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_passed_down():
    box_passed_down = L.code_box("passed_down", lines=8)
    box_passed_down
    return (box_passed_down,)


@app.cell(hide_code=True)
def check_passed_down(box_passed_down):
    result_passed_down = L.check("passed_down", box_passed_down.value)
    result_passed_down
    return (result_passed_down,)


@app.cell(hide_code=True)
def hints_passed_down():
    L.hints(
        "passed_down",
        [
            "Twee voorwaarden: de duim is omlaag, en de eval set liet het antwoord door.",
            "Een tweede `if` in de eerste, die vraagt of de eval set de chat doorliet.",
            "Schrijf onder `if chat[\"thumb\"] == \"down\":` de regel `if ___:`, met de `append` een stap verder in.",
        ],
        why="""```python
def passed_down(chats):
    messages = []
    for chat in chats:
        if chat["thumb"] == "down":
            if chat["passed"]:
                messages.append(chat["message"])
    return messages
```

Deze berichten gaan de eval set in, met een notitie van wat een goed antwoord nodig heeft. Dat is het data flywheel: wat online misgaat, wordt vanaf dan offline getest.""",
    )
    return


@app.cell(hide_code=True)
def text_worth():
    mo.md(r"""
    ### Oefening 4. Wat een versie waard is
    """)
    return


@app.cell(hide_code=True)
def choice_worth_cell():
    _f = C.findings()
    _c = {v: C.money(_f["cost_1000"][v], L.lang) for v in ("v1", "v2")}
    _k = {v: C.money(_f["cost_done"][v], L.lang) for v in ("v1", "v2")}
    choice_worth = L.choice(
        "worth",
        question="Welk getal vertelt je het best wat een versie de bank waard is?",
        options={
            "chat": "Wat 1.000 chats kosten",
            "eval": "Hoeveel inputs de eval set doorstaan",
            "done": "Wat 1.000 gedane taken kosten",
        },
        correct="done",
        why={
            "done": "Een chat die niet helpt, kost ook tokens, en de klant is niet geholpen. Versie 2 kost {c2} dollar voor 1.000 chats tegen {c1}, maar voor 1.000 gedane taken {k2} tegen {k1}: bijna hetzelfde, omdat hij minder klanten helpt.".format(
                c1=_c["v1"], c2=_c["v2"], k1=_k["v1"], k2=_k["v2"]),
            "chat": "Versie 2 is goedkoper per chat, {c2} tegen {c1} dollar voor 1.000 chats, maar helpt minder klanten, {h2} tegen {h1} van de {n}. Voor 1.000 gedane taken kosten de twee bijna hetzelfde, {k2} tegen {k1}.".format(
                c1=_c["v1"], c2=_c["v2"], k1=_k["v1"], k2=_k["v2"], h1=_f["helped"]["v1"], h2=_f["helped"]["v2"], n=_f["n"]),
            "eval": "De eval set keurde meer antwoorden van versie 2 goed, {e2} tegen {e1}, maar heeft geen check voor verzonnen. Wat de bank krijgt, zijn gedane taken, en wat ze ervoor betaalt.".format(
                e1=_f["eval"]["v1"], e2=_f["eval"]["v2"]),
        },
    )
    choice_worth
    return (choice_worth,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer andere weken

    Elk getal is een week met andere muntworpen: andere klanten krijgen versie 1 en versie 2. Kijk hoeveel de uitkomst van de A/B test verandert.
    """)
    return


@app.cell(hide_code=True)
def settings_runs():
    runs_slider = mo.ui.slider(start=1, stop=C.RUNS, value=C.SEED, label="Week", show_value=True)
    runs_slider
    return (runs_slider,)


@app.cell(hide_code=True)
def demo_runs(runs_slider):
    mo.Html(C.runs_picture(runs_slider.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De eval set van [Een eval set en regression tests](/course/open/build-eval-set?lang=nl) is de offline helft van deze kringloop; de chats die uit de online helft terugkomen, zijn wat [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl) leest. Een check voor *verzonnen* vraagt iemand die de betekenis leest, zoals in [Wie het antwoord beoordeelt](/course/open/build-judges?lang=nl). [Zien wat er gebeurde](/course/open/build-tracing?lang=nl) laat zien wat er binnen één chat gebeurde, stap voor stap, en [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl) maakt elke chat goedkoper.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_worth, result_done, result_passed_down, result_up):
    L.wrap_up(result_up, result_done, result_passed_down, choice_worth)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
