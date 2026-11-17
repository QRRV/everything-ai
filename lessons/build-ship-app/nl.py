# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-ship-app. Route: course/route/lessen/bouwen.yaml. Checks, de instellingenbestanden van de bank, de log van twee dagen en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-ship-app", "nl")
    C = L.checks   # de instellingenbestanden van de bank, de log van twee dagen en de plaatjes
    DEFAULTS = C.DEFAULTS             # wat de app gebruikt voor een instelling die een instellingenbestand weglaat
    HISTORY = C.HISTORY               # de instellingenbestanden van de bank, oudste eerst, elk met zijn "version"
    SETTINGS_TEXT = C.SETTINGS_TEXT   # het instellingenbestand van 2026-11-12, als tekst
    LOG = C.LOG                       # de log van twee dagen: één regel JSON-tekst per request


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_threshold():
    _f = C.findings()
    _lo, _mid = _f["cache"][0.75], _f["cache"][0.85]
    quick_threshold = L.choice(
        "quick-threshold",
        question="Een semantic cache geeft een eerder antwoord vanaf een similarity van 0,85. Wat gebeurt er als je dat verlaagt naar 0,75?",
        options={
            "fewer": "Minder hits, en elke hit past beter",
            "more": "Meer hits, en meer verkeerde",
            "same": "Dezelfde hits, alleen sneller gevonden",
        },
        correct="more",
        why={
            "more": "Vanaf 0,75 beantwoordde de cache van de course {h} van de {n} vragen in plaats van {h0}, en kregen er {o} het antwoord voor een ander onderwerp in plaats van {o0}.".format(
                h=_lo[0], n=_f["n"], h0=_mid[0], o=_lo[1], o0=_mid[1]),
            "fewer": "Een lagere drempel laat meer eerdere antwoorden door, niet minder: vanaf 0,75 beantwoordde de cache van de course {h} vragen in plaats van {h0}. Zie [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl).".format(
                h=_lo[0], h0=_mid[0]),
            "same": "De drempel bepaalt welke eerdere antwoorden dichtbij genoeg zijn, dus de hits veranderen: vanaf 0,75 beantwoordde de cache van de course {h} vragen in plaats van {h0}. Zie [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl).".format(
                h=_lo[0], h0=_mid[0]),
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    _v1, _v2 = (s for s, _, _ in C.DAYS)
    mo.md(r"""
    ## Waarom

    In deze module bekeek je de assistent van de bank van alle kanten: een eval set, gebruikers, traces, guardrails, kosten en een cache. Elk daarvan verandert de app, en een app die klanten gebruiken, blijft veranderen: een nieuwe prompt, een nieuw model, een cache. Versie 2 van de prompt kwam bij {p2} van de {n} testvragen door de eval set, tegen {p1} voor versie 1, en verzon toch iets in {m2} antwoorden tegen {m1}, volgens de aantekeningen van de course in [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl). Daar heeft de eval set geen check voor.

    Staat zo'n versie eenmaal live, dan wil je weten welke versie welke klant antwoordde, en wil je in één stap terug, nog voordat je de oorzaak kent.

    **Een voorbeeld uit het dagelijks leven.** Een bakkerij probeert een nieuw recept voor brood. De oude receptkaart blijft in de la, en op elke bestelbon staat met welk recept het brood gebakken is. Als klanten klagen, bakt de bakkerij de volgende ochtend weer het oude recept, en zoekt pas daarna uit wat er mis was met het nieuwe.
    """.format(n=_f["n"], p1=_f["passed"][_v1], p2=_f["passed"][_v2], m1=_f["made_up"][_v1], m2=_f["made_up"][_v2]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **AI engineering** | een applicatie bouwen op een model dat iemand anders trainde: het werk zit in de prompts, de data, de evals en de code om het model heen, niet in het trainen |
    | **rollback** | teruggaan naar een eerdere versie die werkte; hier: het instellingenbestand daarvan opnieuw laden |

    Een LLM-app die klanten gebruiken, heeft vier delen: de instellingen, jouw code, het model en de log. De instellingen zijn alles wat je kunt veranderen zonder de code te veranderen: het model, de versie van de prompt, `max_tokens`, de temperature, de cache en de drempel daarvan. Ze staan in een bestand met een versie. Elke verandering is een nieuw bestand, en de oude blijven bewaard.

    Je code leest de instellingen bij het starten, en gebruikt een standaardwaarde voor alles wat het bestand weglaat. Elk request schrijft één regel naar de log: de tijd, de versie van de instellingen, de vraag, de seconden, de tokens en wat de checks zeiden. Een regel is JSON-tekst, dus een programma kan hem teruglezen en tellen.

    Een rollback laadt een ouder instellingenbestand. Er verandert geen code en er wordt geen model getraind; er verandert één bestand. Daarna zoek je de oorzaak, en breng je de oplossing uit als een nieuwe versie.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een verandering uitbrengen

    1. **Een bestand per versie:** zet elke instelling in een bestand met een versie: het model, de versie van de prompt, `max_tokens`, de temperature, de cache. Bewaar de oude bestanden.
    2. **Voor de klanten:** draai de eval set op de nieuwe versie en kijk naar de regressions, zoals in [Een eval set en regression tests](/course/open/build-eval-set?lang=nl).
    3. **Dan een deel van de klanten:** geef de nieuwe versie eerst aan een deel van de klanten, zoals in [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl), en log elk request met de versie van de instellingen.
    4. **Tel per versie:** tel uit de log voor elke versie wat de checks zeiden, wat mensen aangaven, de seconden en de tokens.
    5. **Eerst terug, dan de oorzaak:** doet een versie het slechter, ga dan terug naar de laatste goede, en zoek daarna de oorzaak. De oplossing is een nieuwe versie die dezelfde stappen doorloopt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _first = C.DAYS[0]
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `SETTINGS_TEXT` is het instellingenbestand van 2026-11-12 als tekst: `{text}`. `json.loads(SETTINGS_TEXT)` maakt er een dictionary van. Het bestand noemt alleen wat deze versie zelf instelt; voor de rest gebruikt de app `DEFAULTS`: {defaults}.

    `HISTORY` bevat de {k} instellingenbestanden van de bank, oudste eerst: versie 1 van de prompt op 2026-11-02, versie 2 op 2026-11-09, en op 2026-11-12 de semantic cache uit [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl), vanaf een similarity van 0,75.

    `LOG` bevat {lines} regels van de log, één per request. Op {d1} kwamen de {n} testvragen binnen onder de instellingen van {s1}, en op {d2} dezelfde {n} onder die van {s2}. `LOG[0]` is:

    ```json
    {line}
    ```

    De seconden en tokens zijn wat de course mat toen hij elke versie draaide; de tijden van de dag zijn van de course zelf. `passed` is wat de eval set uit [Een eval set en regression tests](/course/open/build-eval-set?lang=nl) van het antwoord zei, en `made_up` is de aantekening van de course erbij uit [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl). Een echte app weet dat niet meteen: hij leert het uit duimpjes en van mensen die een steekproef van de antwoorden lezen.
    """.format(text=SETTINGS_TEXT, defaults=", ".join(f"`{k}` {json.dumps(v)}" for k, v in DEFAULTS.items()), k=len(HISTORY),
               lines=_f["lines"], n=_f["n"], d1=_first[2], s1=_first[0], d2=C.DAYS[1][2], s2=C.DAYS[1][0], line=LOG[0]))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _v1, _v2 = (s for s, _, _ in C.DAYS)
    _h, _o, _cp, _fp = _f["cache"][0.75]
    mo.md(r"""
    ### Wat de course mat

    Geteld per versie van de instellingen zegt de log: onder {v1} kwamen {p1} van de {n} antwoorden door de eval set, en verzonnen er {m1} iets; onder {v2} kwamen er {p2} door, en verzonnen er {m2} iets. Versie 2 doet het beter op de eval set en verzint meer; volgens de aantekeningen van de course had hij ook bij {o2} antwoorden een ander probleem tegen {o1}, dus de twee versies hielpen ongeveer evenveel klanten, {h2} tegen {h1}. Van de vragen die versie 1 goed had, had versie 2 er {reg} fout. Snelheid is het verschil niet: de {n} antwoorden van elke versie duurden samen {s1} en {s2} seconden.

    De instellingen van 2026-11-12 zetten de cache aan vanaf 0,75. In [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl) beantwoordde die cache {h} van de {n} vragen, en kregen er {o} het antwoord voor een ander onderwerp; de eval set keurde {cp} van zijn antwoorden goed tegen {fp} van de eigen antwoorden van het model. Dus op 2026-11-12 gaat de bank eerst terug naar 2026-11-09. Of hij daarna nog verder teruggaat, naar 2026-11-02, hangt af van wat voor de bank zwaarder weegt: minder andere problemen en een betere eval set, of minder antwoorden die iets verzinnen.
    """.format(v1=_v1, v2=_v2, n=_f["n"], p1=_f["passed"][_v1], p2=_f["passed"][_v2], m1=_f["made_up"][_v1], m2=_f["made_up"][_v2],
               o1=_f["other"][_v1], o2=_f["other"][_v2], h1=_f["helped"]["v1"], h2=_f["helped"]["v2"],
               reg=_f["regressions"], s1=L.number(_f["seconds"][_v1], 1), s2=L.number(_f["seconds"][_v2], 1), h=_h, o=_o, cp=_cp, fp=_fp))
    return


@app.cell(hide_code=True)
def picture_measured():
    L.watch_out("Bewaar de oude instellingenbestanden, en schrijf de versie van de instellingen in elke regel van de log. Zonder de versie kan een log je niet vertellen welke verandering het slechter maakte; zonder het oude bestand is er niets om naar terug te gaan.")
    return


@app.cell(hide_code=True)
def text_settings():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Instellingen met standaardwaarden

        `settings_from(text)` moet de tekst van een instellingenbestand omzetten in een dictionary met elke instelling die de app nodig heeft: wat het bestand zegt, en voor elke key van `DEFAULTS` die het bestand weglaat, de standaardwaarde. De startcode leest het bestand met `json.loads`. `for key in DEFAULTS:` gaat de keys van `DEFAULTS` langs, en `key in a_dict` is `True` als de dictionary die key heeft. Het plaatje laat zien wat je functie geeft voor `SETTINGS_TEXT`.
        """),
        L.task("Voeg elke key van `DEFAULTS` toe die `settings` nog niet heeft, met de standaardwaarde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_settings():
    box_settings = L.code_box("settings_from", lines=7)
    box_settings
    return (box_settings,)


@app.cell(hide_code=True)
def check_settings(box_settings):
    result_settings = L.check("settings_from", box_settings.value)
    result_settings
    return (result_settings,)


@app.cell(hide_code=True)
def hints_settings():
    L.hints(
        "settings_from",
        [
            "Het bestand noemt alleen wat deze versie zelf instelt; de app heeft toch elke instelling nodig.",
            "Ga de keys van `DEFAULTS` langs met `for`, en voeg een key alleen toe als hij niet in `settings` zit.",
            "Schrijf onder die `if`: `settings[key] = ___`.",
        ],
        why="""```python
def settings_from(text):
    settings = json.loads(text)
    for key in DEFAULTS:
        if key not in settings:
            settings[key] = DEFAULTS[key]
    return settings
```

Het bestand blijft kort en zegt alleen wat deze versie instelt. Verschillen twee versies, dan is dat wat je vergelijkt.""",
    )
    return


@app.cell(hide_code=True)
def text_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Geteld per versie

        `count_by_settings(lines, field)` moet voor elke versie van de instellingen in de log de regels tellen waar `field` `True` is. Elke regel is JSON-tekst: de startcode leest hem met `json.loads` en haalt de versie uit `entry["settings"]`. Geef elke versie in de log een telling, ook als die 0 is. Het plaatje telt `made_up` en `passed` in `LOG`.
        """),
        L.task("Zet elke nieuwe versie op 0, en tel er 1 bij op als `entry[field]` `True` is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_by_settings", lines=11)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_by_settings", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_by_settings",
        [
            "`counts` koppelt een versie aan een telling: een versie krijgt zijn key de eerste keer dat hij langskomt, met 0.",
            "Twee keer `if` in de loop: één voor een versie die nog niet in `counts` zit, en één voor `entry[field]`.",
            "Schrijf onder de tweede `if`: `counts[version] = ___`.",
        ],
        why="""```python
def count_by_settings(lines, field):
    counts = {}
    for line in lines:
        entry = json.loads(line)
        version = entry["settings"]
        if version not in counts:
            counts[version] = 0
        if entry[field]:
            counts[version] = counts[version] + 1
    return counts
```

Dezelfde functie telt elk veld dat `True` of `False` is: de eval set, de aantekeningen, en met één veld erbij ook de duimpjes omlaag.""",
    )
    return


@app.cell(hide_code=True)
def text_good():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De weg terug

        `bad` is een set versies die het slechter deden. `last_good(history, bad)` moet de nieuwste instellingen in `history` geven waarvan de versie niet in `bad` zit, of `None` als die er niet zijn. `history` staat oudste eerst, zoals `HISTORY`. De startcode loopt hem langs en houdt elke dictionary met instellingen, dus hij geeft de nieuwste, slecht of niet. Het plaatje laat twee rollbacks zien: met 2026-11-12 slecht, en met 2026-11-09 ook slecht.
        """),
        L.task("Houd een dictionary met instellingen alleen als de versie ervan niet in `bad` zit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_good():
    box_good = L.code_box("last_good", lines=6)
    box_good
    return (box_good,)


@app.cell(hide_code=True)
def check_good(box_good):
    result_good = L.check("last_good", box_good.value)
    result_good
    return (result_good,)


@app.cell(hide_code=True)
def hints_good():
    L.hints(
        "last_good",
        [
            "De loop gaat van oud naar nieuw, dus wat hij als laatste houdt, is de nieuwste die hij hield.",
            "Zet vóór `good = settings` een `if` die naar de versie van `settings` kijkt.",
            "Schrijf `if ___ not in bad:`.",
        ],
        why="""```python
def last_good(history, bad):
    good = None
    for settings in history:
        if settings["version"] not in bad:
            good = settings
    return good
```

Ga zo weinig mogelijk terug: elke versie die je overslaat, neemt ook mee wat die goed deed.""",
    )
    return


@app.cell(hide_code=True)
def text_first_step():
    mo.md(r"""
    ### Oefening 4. De eerste stap
    """)
    return


@app.cell(hide_code=True)
def choice_first_step_cell():
    _f = C.findings()
    _h, _o = _f["cache"][0.75][:2]
    choice_first_step = L.choice(
        "first-step",
        question="De instellingen van 2026-11-12 staan live, en klanten krijgen antwoorden die voor een ander onderwerp bedoeld zijn. Wat doe je eerst?",
        options={
            "back": "De instellingen van 2026-11-09 opnieuw laden",
            "fix": "Met de eval set een betere drempel zoeken, en die uitbrengen",
            "wait": "Laten draaien en volgende week de log lezen",
        },
        correct="back",
        why={
            "back": "Een rollback verandert één bestand en stopt de verkeerde antwoorden nu. Zoek daarna een betere drempel, en breng die uit als een nieuwe versie die eerst door de eval set gaat.",
            "fix": "Dat is een goede volgende stap, maar het kost tijd, en intussen blijven klanten antwoorden voor een ander onderwerp krijgen. Ga eerst terug: laad de instellingen van 2026-11-09 opnieuw.",
            "wait": "Elke dag kost klanten verkeerde antwoorden: vanaf 0,75 waren {o} van de {h} antwoorden uit de cache voor een ander onderwerp. Ga eerst terug.".format(o=_o, h=_h),
        },
    )
    choice_first_step
    return (choice_first_step,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een regel van de log is een kleine trace, zoals in [Zien wat er gebeurde](/course/open/build-tracing?lang=nl), en die per versie tellen is de online evaluation van [Offline, online en wat het waard is](/course/open/build-eval-online?lang=nl). Het instellingenbestand bevat de versies van de prompt uit [Prompts als code: templates en versies](/course/open/build-prompt-templates?lang=nl) en de cache uit [Kosten, latency en antwoorden cachen](/course/open/build-cost-latency-cache?lang=nl). [Eerst vragen, dan doen](/course/open/build-human-in-loop?lang=nl) voegt een mens toe aan de app, voor de stappen die hij niet alleen hoort te zetten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first_step, result_count, result_good, result_settings):
    L.wrap_up(result_settings, result_count, result_good, choice_first_step)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
