# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-package-squatting. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-package-squatting", "nl")
    C = L.checks   # de run van de course: de packagenamen van Qwen3 voor tien taken, en wat PyPI had op de dag dat de course keek


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Is dit het model dat je bedoelde?](/course/open/rai-hub-integrity?lang=nl) en [Code uit modules gebruiken](/course/open/py-imports?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pin():
    quick_pin = L.choice(
        "quick-pin",
        question="Je pin bevat een commit id en een SHA-256, en een bestand klopt met allebei. Wat vertelt je dat?",
        options={
            "right": "Dat het het bestand is dat je vastlegde",
            "maker": "Dat de organisatie in de naam het heeft gemaakt",
            "safe": "Dat niemand een aanval in het bestand heeft gestopt",
        },
        correct="right",
        why={
            "right": "Een pin houdt je bij één bestand, goed of slecht, wie het ook maakte. Deze les stelt de vraag die vóór de pin komt: of de naam die je koos de naam is die je bedoelde.",
            "maker": "Een naam kan van eigenaar wisselen, en een SHA-256 van het bestand van de nieuwe eigenaar klopt met dat bestand. Zie *Wat een hash je niet vertelt* in [Is dit het model dat je bedoelde?](/course/open/rai-hub-integrity?lang=nl).",
            "safe": "Een SHA-256 laat een verandering na de pin zien, geen aanval die er al vóór de pin in zat, zoals in de releases van xz. Zie *Wat een hash je niet vertelt* in [Is dit het model dat je bedoelde?](/course/open/rai-hub-integrity?lang=nl).",
        },
    )
    quick_pin
    return (quick_pin,)


@app.cell(hide_code=True)
def quick_check_import():
    quick_import = L.choice(
        "quick-import",
        question="Wat doet `import math`?",
        options={
            "right": "Het haalt de module math binnen in je programma",
            "download": "Het downloadt math elke keer opnieuw van internet",
            "copy": "Het schrijft de code van math in je bestand",
        },
        correct="right",
        why={
            "right": "`import` haalt een module binnen die al op je computer staat: `math` hoort bij Python. Een package dat iemand anders schreef, zoals NumPy, is daar ooit op gezet, op naam. Bij die naam begint deze les.",
            "download": "`import` downloadt niets: `math` hoort bij Python, en `import` haalt binnen wat al op je computer staat. Zie *Het idee* in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "copy": "Je bestand blijft zoals het is: `import` maakt de namen van de module bruikbaar, met `math.` ervoor. Zie *Het idee* in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
        },
    )
    quick_import
    return (quick_import,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Spoke & Chain, de fietsenwinkel uit de vorige les, legt zijn model nu vast. Zijn scripts hebben ook Python-packages nodig, en de winkel vraagt een LLM welke. Voor de barcode op een foto van een fietsonderdeel antwoordt het LLM van de course, Qwen3-0.6B, `pip install barcode_reader` of `pip install barcode-reader`, bij alle tien pogingen van de course. pip, dat Python-packages installeert, vindt een package op naam op PyPI, de *Python Package Index*, waar iedereen een package kan publiceren. Op 19 november 2026 had PyPI geen project met die naam, dus pip zou niets hebben gevonden om te installeren. Zodra iemand de naam registreert, installeert hetzelfde commando wat die persoon erin heeft gestopt.

    Twee soorten namen nodigen daartoe uit. **Typosquatting** is een package publiceren onder een naam die op een populaire lijkt, zoals `urlib3` voor `urllib3`, in de hoop dat mensen het per ongeluk installeren. **Slopsquatting** is een naam registreren die language models verzinnen, zoals `barcode-reader`, zodat wie het model vertrouwt het package van de aanvaller installeert. Deze les controleert een naam voordat hij wordt geïnstalleerd; de namen die de les laat zien, zijn er om te controleren, nooit om te installeren of te registreren.

    **Een voorbeeld uit het dagelijks leven.** Een vriend zegt dat je een onderdeel moet kopen bij "Wielerwereld aan de Stationsstraat". Als die winkel er niet is, krijgt wie er onder die naam een winkel opent jouw bezoek, wat er ook verkocht wordt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Bij onderzoek naar namen van packages die LLM's verzinnen, kwam Bar Lanyado van het beveiligingsbedrijf Lasso Security `huggingface-cli` tegen, een Python-package dat niet bestond. Lanyado publiceerde een leeg package onder die naam. Hoe vaak werd het in drie maanden gedownload?",
        options={
            "small": "Ongeveer 300",
            "middle": "Ongeveer 3000",
            "right": "Ruim 30.000",
        },
        reveal="**Ruim 30.000 keer.** Volgens Lanyado kreeg het nep-package, dat leeg was, in drie maanden meer dan 30.000 echte downloads, en de teller liep nog. Instructies om het te installeren stonden in de README van een repository voor onderzoek door Alibaba. Downloads zijn geen slachtoffers: het package was leeg, en Lanyado publiceerde een tweede nep-package om te schatten hoeveel downloads van scanners kwamen en niet van mensen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Namen die op elkaar lijken

    De truc is ouder dan packages. CAPEC, de catalogus van aanvalspatronen van MITRE, beschrijft hem voor webadressen als CAPEC-630, *TypoSquatting*: een aanvaller registreert een domeinnaam die minstens één teken verschilt van een vertrouwd domein. Joseph Spracklen en vijf coauteurs omschrijven het voor packages als packages maken met namen die op populaire lijken, om gebruikers te misleiden.

    In september 2017 meldde de Nationale Veiligheidsautoriteit van Slowakije aan het *Python Security Response Team* dat PyPI kwaadaardige packages bevatte. De beveiligingsaankondiging van Python noemt er 11, waaronder `setup-tools` en `urlib3`, gepubliceerd in juni en ongeveer drie maanden online (Vu et al. tellen er tien). Na de melding waren ze volgens die aankondiging binnen 70 minuten allemaal weg, en de packages installeerden ook de module die je verwachtte, zodat de aanval niet makkelijk opviel. Mensen typen echt verkeerd: volgens dezelfde aankondiging hield een project dat de namen van de standaardbibliotheek bezette, de modules die bij Python horen, in 4 dagen ongeveer 7500 pogingen tegen om zo'n naam te installeren, zo'n 0,006% van de ruwweg 123 miljoen downloads van PyPI in die dagen.

    In 2020 bestudeerden Duc-Ly Vu en vier coauteurs 36 bekende aanvallen: aanvallers halen het liefst tekens weg uit de naam van het echte package. Andere vormen die ze noemen: een enkelvoud voor een meervoud (`setuptool`), een extra streepje (`setup-tools`), de Britse spelling (`colourama` voor `colorama`), een hoofdletter I voor een kleine l (`jeIlyfish`, 119 keer gedownload toen het werd gevonden), en *combosquatting*, dat de woorden van een naam omwisselt: `nmap-python` voor `python-nmap`. In maart 2024 vond Check Point meer dan 500 kwaadaardige packages op PyPI, elk van een eigen account, met namen van gewone verschrijvingen (`requestss` voor `requests`) tot namen die er inhoudelijk (`pandas-sdk` voor `pandas`) of qua uiterlijk (`request5` voor `requests`) op lijken. Op 28 maart 2024 stopte PyPI volgens zijn statuspagina ongeveer tien en een half uur met nieuwe projecten en nieuwe accounts, om een lopende golf van malware tegen te gaan.

    ### Als twee namen één zijn

    PyPI vergelijkt namen niet letter voor letter. De regel, in de *Python Packaging User Guide*: schrijf de naam in kleine letters, en vervang elke reeks van de tekens `.`, `-` en `_` door één `-`. Dus `setup_tools`, `Setup.Tools` en `setup-tools` zijn één naam, en `setuptools` is een andere. In oefening 1 schrijf je die regel.

    Je kunt PyPI vragen of het een project met een naam heeft: de JSON API antwoordt op `GET /pypi/<project>/json` met de gegevens van het project, of met de code 404, niet gevonden. Een 404 zegt alleen dat er nu geen project is, niet dat iedereen de naam mag nemen: de regels van PyPI, PEP 541, laten de beheerders sommige namen bij voorbaat onbeschikbaar verklaren om veiligheidsredenen, en sinds de aankondiging van 2017 weigert PyPI vanzelf elke naam uit de standaardbibliotheek.

    ### Hoe ver twee namen uit elkaar liggen

    Spracklen et al. meten hoeveel twee namen op elkaar lijken met de **Levenshtein-afstand**: hoeveel tekens je moet invoegen, weghalen of vervangen om de ene string gelijk te maken aan de andere. Van `urllib3` naar `urlib3` is één keer weghalen: afstand 1. Van `requests` naar `reqeusts` is twee keer vervangen: afstand 2. `C.distance(a, b)` van de course rekent hem uit, en oefening 2 en 3 gebruiken hem. Vu et al. legden hun grens voor hoeveel twee namen op elkaar lijken bij twee, omdat die de meeste bekende aanvallen vindt, 21 van de 36, met minder valse meldingen: de andere 15 lagen erbuiten. `pandas-sdk` ligt 4 bewerkingen van `pandas`.

    ### Namen die een model verzint

    Vraag een LLM om code, en het kan een package noemen dat niet bestaat. Spracklen et al. noemen dat *package hallucination*: een LLM maakt code die een package aanraadt of gebruikt dat niet echt bestaat. Ze lieten 16 LLM's 576.000 stukken code maken in Python en JavaScript, en vergeleken elke Python-naam met de lijst van PyPI en elke JavaScript-naam met die van npm, de index voor JavaScript, allebei van 10 januari 2024. Van 2,23 miljoen packagenamen stonden er 440.445 (19,7%) niet op de lijst van hun taal: 205.474 verschillende namen. Het gemiddelde was minstens 5,2% voor de commerciële modellen, de GPT-modellen van OpenAI, en 21,7% voor de open-source modellen; GPT-4 Turbo had het laagste percentage, 3,59%. Verzonnen namen komen terug: van 500 prompts die een verzonnen naam hadden opgeleverd, elk nog 10 keer gesteld, kwam 43% van de verzonnen packages alle 10 keer terug, en 39% geen enkele keer meer; en 81% van de verzonnen packages kwam van maar één model. Ze lijken zelden op een typfout: van de verzonnen Python-namen lag maar 13,4% (10.263 van 76.489) binnen een Levenshtein-afstand van 1 of 2 van een echt package, en 48,6% lag 6 of meer bewerkingen weg. Spracklen et al. publiceerden hun lijst met namen niet.

    Het woord slopsquatting kwam in april 2025. Andrew Nesbitt postte op 8 april 2025 op Mastodon een omschrijving: een LLM verzint een naam van een package dat niet bestaat, en iemand met kwade bedoelingen registreert hem; het AI-broertje van typosquatting. Nesbitt gaf de eer voor de naam aan Seth Larson, *security developer-in-residence* bij de *Python Software Foundation*, die het woord in een gesprek had voorgesteld. *Slop* is een gangbaar minachtend woord voor wat AI-modellen maken, schreef The Register. Larson zei daar dat wie code, packages en informatie van een LLM gebruikt, die tegen de werkelijkheid moet controleren voordat ze in gebruik gaan. Hoeveel installaties van verzonnen namen komen, is volgens Larson waarschijnlijk niet te tellen zonder meer openheid van de makers van LLM's; het bekende geval is het lege testpackage uit de schatvraag hierboven.

    ### Voordat je installeert

    Een naam op PyPI betekent alleen dat iemand hem registreerde. PyPI is volgens de beveiligingsaankondiging van Python uit 2017 een open platform met bewust weinig drempels om mee te doen: het haalt een project weg na een melding, als het project malware is, of als het een naam bezet zonder iets te doen, een leeg package, zo staat het in PEP 541, en het controleert projecten niet voordat ze gepubliceerd worden. Spracklen et al.: dat een package in een open-source repository staat, bevestigt nog niet dat het te vertrouwen is. Dus een naam van een model krijgt de controles van een naam van een vreemde: is het het package dat je bedoelde, bestaat het, en als het bestaat, wat zeggen zijn pagina en zijn code.

    ### Wat de course deed

    De course vroeg Qwen3-0.6B, dat op je eigen computer draait, naar de packages voor tien taken van Spoke & Chain, elk tien keer, met de seeds 0 tot en met 9, in het Engels: welke Python-packages heb ik nodig om de taak uit te voeren, antwoord met één pip install-commando en verder niets. Daarna vroeg de course de JSON API van PyPI, één keer, op 19 november 2026, of het een project met elke naam had. Er werd niets geïnstalleerd. Spracklen et al. hielden hun lijst met namen voor zich; de namen in deze les komen van een klein LLM dat iedereen met deze tien vragen kan draaien, en ze staan hier om ze te controleren. De course installeert zijn eigen packages uit `uv.lock`, een bestand in de course met 101 packages, van elk één versie en de SHA-256 van elk van hun bestanden, 323 in totaal: pins, zoals in de vorige les. Die 101 namen zijn `C.known()`.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **typosquatting** | een package publiceren onder een naam die op een populaire lijkt, in de hoop dat mensen het per ongeluk installeren |
        | **slopsquatting** | een naam registreren die language models verzinnen, zodat wie het model vertrouwt het package van de aanvaller installeert |
        | **PyPI** | de *Python Package Index*, waar pip en uv een package op naam vinden, en waar iedereen een package kan publiceren |
        | **genormaliseerde naam** | een naam zoals PyPI hem vergelijkt: kleine letters, en elke reeks van `.`, `-` en `_` als één `-` |
        | **Levenshtein-afstand** | hoeveel letters je moet invoegen, weghalen of veranderen om de ene naam in de andere te veranderen |
        | `C.distance(a, b)` | de Levenshtein-afstand tussen `a` en `b` |
        """),
        L.watch_out("Installeer nooit een naam uit deze les, of uit het antwoord van een LLM, om te zien wat het is, en registreer er nooit een. Een naam die PyPI niet had, kan iedereen registreren, met wat dan ook erin; en een naam die het wel had, is het package van iemand, goed of slecht."),
        mo.Html(C.names_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Normaliseer de naam, en vergelijk hem met de packages die je al gebruikt: dezelfde naam is het package dat je kent.
    2. Een naam die 1 of 2 bewerkingen van een bekende ligt, is waarschijnlijk een typfout: gebruik de naam die je kent.
    3. Een naam die PyPI niet heeft, is verzonnen, door een mens of een model: zoek het package voor de klus op een andere manier.
    4. Een naam die PyPI wel heeft: lees zijn pagina en zijn code voordat je hem installeert, en leg hem daarna vast, met een versie en de hashes van zijn bestanden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel telt de run van de course. `C.recorded_answers()` geeft per taak de teksten van de tien antwoorden van Qwen3; `C.names_in(text)` de packagenamen in de pip install-commando's van een antwoord, genormaliseerd; `C.recorded_names()` elke naam in de 100 antwoorden, elk één keer; en `C.pypi()["names"]` per naam of PyPI hem op 19 november 2026 had.
    """)
    return


@app.cell
def example_count():
    answers = C.recorded_answers()
    print(answers["barcode"][1], C.names_in(answers["barcode"][1]))

    names = C.recorded_names()
    on_pypi = C.pypi()["names"]   # True als PyPI de naam had
    missing = []
    for name in names:
        if not on_pypi[name]:
            missing.append(name)
    print(len(names), "different names,", len(missing), "not on PyPI")
    print(missing[:5])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.vstack([
        mo.md(r"""
        `barcode_reader` wordt `barcode-reader`, de naam die PyPI vergelijkt. 98 van de 100 antwoorden hadden een pip install-commando, en die gaven 33 verschillende namen; 15 van die namen stonden op 19 november 2026 niet op PyPI, en 31 antwoorden noemden er een van. Voor een hartslagmeter gaf Qwen3 zes namen in tien antwoorden, en PyPI had er geen van. Spracklen et al. telden verzonnen namen die bij alle tien pogingen terugkwamen, en deze run heeft er een: `barcode-reader` kwam elke keer.
        """),
        mo.Html(C.run_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_normalize():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. normalize: de naam die PyPI vergelijkt

        Schrijf `normalize(name)`: `name` zoals PyPI hem vergelijkt, in kleine letters, met elke reeks van `.`, `-` en `_` vervangen door één `-`. Het vak loopt door de letters van `name.lower()` en bouwt het resultaat op in `out`; `out.endswith("-")` is `True` als het laatste teken van `out` een `-` is. Vervang de `...`. De check roept `normalize("barcode_reader")` aan, en daarna de namen in het plaatje eronder.
        """),
        L.task("Voeg een letter toe zoals hij is; voeg voor een scheidingsteken alleen een \"-\" toe als out er nog niet op eindigt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_normalize():
    box_normalize = L.code_box("normalize", lines=10)
    box_normalize
    return (box_normalize,)


@app.cell(hide_code=True)
def check_normalize(box_normalize):
    result_normalize = L.check("normalize", box_normalize.value)
    result_normalize
    return (result_normalize,)


@app.cell(hide_code=True)
def hints_normalize():
    L.hints(
        "normalize",
        [
            "*Als twee namen één zijn* in *Het idee*: kleine letters, en een reeks scheidingstekens telt als één.",
            "Een `if` die kijkt of `ch` een van de drie scheidingstekens is; daaronder een tweede `if` die alleen een `\"-\"` toevoegt als `out` er nog niet op eindigt. Onder `else` voeg je `ch` zelf toe.",
            "`        if ch in ___:`",
        ],
        why="`if ch in \".-_\":` met `if not out.endswith(\"-\"):` en `out += \"-\"` eronder, en `else:` met `out += ch`. `normalize(\"barcode_reader\")` geeft `\"barcode-reader\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_normalize():
    mo.md(r"""
    `zope.interface` en `zope-interface` zijn voor PyPI één project, en `my-._package` laat zien waarom een reeks als één telt: het wordt `my-package`. Elke controle in de rest van deze les begint bij de genormaliseerde naam.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_near():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. near: namen die bijna hetzelfde zijn

        Schrijf `near(name, known)`: de namen in de lijst `known` die 1 of 2 bewerkingen van `name` liggen als die genormaliseerd is, in de volgorde van `known`. `known` bevat genormaliseerde namen, en `found` verzamelt de namen. Het vak mag `normalize` en `distance` van de course gebruiken. Python kan twee vergelijkingen aan elkaar rijgen: `a <= b <= c` is `True` als `b` minstens `a` en hoogstens `c` is. Vervang de `...`. De check roept `near("urlib3", KNOWN)` aan, waarin `KNOWN` de 101 namen uit de `uv.lock` van de course bevat, en daarna de namen in het plaatje eronder.
        """),
        L.task("Houd elke naam in known die 1 of 2 bewerkingen verderop ligt, en alleen die."),
    ])
    return


@app.cell(hide_code=True)
def exercise_near():
    box_near = L.code_box("near", lines=8)
    box_near
    return (box_near,)


@app.cell(hide_code=True)
def check_near(box_near):
    result_near = L.check("near", box_near.value)
    result_near
    return (result_near,)


@app.cell(hide_code=True)
def hints_near():
    L.hints(
        "near",
        [
            "*Hoe ver twee namen uit elkaar liggen* in *Het idee*: een typfout ligt op een kleine afstand van de echte naam, en een naam ligt op geen afstand van zichzelf.",
            "Eén `if` over `distance(name, k)`, met aan elke kant een vergelijking, zoals in `a <= b <= c`; daaronder `found.append(k)`.",
            "`        if ___ <= distance(name, k) <= 2:`",
        ],
        why="`if 1 <= distance(name, k) <= 2:` met `found.append(k)` eronder. `near(\"urlib3\", KNOWN)` geeft `[\"urllib3\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_near():
    mo.md(r"""
    `near` vindt `urlib3` en `setup-tools`, twee van de 11 namen uit 2017. Ook met `pandas` in `known` zou hij `pandas-sdk` niet vinden, 4 bewerkingen verderop: een afstand van 1 of 2 vangt typfouten, geen toegevoegde woorden. En `numpy` ligt dicht bij `sympy`: die twee namen, allebei in `uv.lock`, liggen 2 bewerkingen uit elkaar, dus een naam die dicht bij een bekende ligt, kan zelf een bekende zijn. Oefening 3 kijkt eerst in `known`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. verdict: wat je met een naam doet

        Schrijf `verdict(name, known, on_pypi)`: wat je met een naam doet, als een van vier teksten, in deze volgorde bekeken. `"known"` als de genormaliseerde naam in `known` staat; `"close to "` met de eerste naam die `near` geeft, als hij er een geeft; `"not on PyPI"` als de naam niet in de lijst `on_pypi` staat; en `"read its page first"` voor elke andere naam. Het vak mag `normalize`, `distance` en `near` van de course gebruiken; zet wat `near` geeft in `close`, met `close = near(name, known)`, en `if close:` is waar als de lijst `close` niet leeg is. Vervang de `...`. De check roept `verdict("barcode_reader", KNOWN, ON_PYPI)` aan, waarin `ON_PYPI` de namen uit de run van de course en uit de voorbeelden van deze les bevat die PyPI op 19 november 2026 had, en daarna de namen in het plaatje eronder.
        """),
        L.task("Eerst known, dan een naam die erop lijkt, dan PyPI; de laatste regel van het vak is voor elke andere naam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=11)
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
            "Stap 1 tot en met 4 van het recept, in die volgorde: elke stap die van toepassing is, beëindigt de functie met zijn tekst.",
            "Drie keer `if`, elk met een `return` eronder: of `name` in `known` staat; over `close`, na `close = near(name, known)`; en of `name` niet in `on_pypi` staat.",
            "`        return \"close to \" + ___`",
        ],
        why="`if name in known:` met `return \"known\"`, dan `close = near(name, known)` en `if close:` met `return \"close to \" + close[0]`, dan `if name not in on_pypi:` met `return \"not on PyPI\"`. `verdict(\"barcode_reader\", KNOWN, ON_PYPI)` geeft `\"not on PyPI\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_verdict():
    mo.md(r"""
    `urlib3` krijgt `"close to urllib3"` voordat PyPI ook maar gevraagd wordt: dat hij vandaag niet op PyPI staat, doet er minder toe dan waar hij op probeert te lijken. `pdfbox` krijgt `"read its page first"`: hij staat op PyPI en ligt bij niets in `uv.lock` in de buurt, en alleen zijn pagina en zijn code kunnen meer zeggen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De antwoorden van Qwen3, beoordeeld

    Kies een taak van Spoke & Chain. De tabel laat de tien antwoorden zien die de course opnam, de namen in elk antwoord, hun verdicts, en of PyPI elke naam op 19 november 2026 had. De verdicts komen van jouw `verdict` zodra oefening 3 klopt, en tot dan van die van de course.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    demo_task = mo.ui.dropdown(options=C.task_options(L.lang), value=C.task_text("barcode", L.lang), label="De taak")
    demo_task
    return (demo_task,)


@app.cell(hide_code=True)
def demo_look(box_verdict, demo_task, result_verdict):
    _verdict = C.mine(box_verdict.value, result_verdict.state == "correct", "verdict")
    _out = mo.Html(C.task_picture(demo_task.value, L.lang, _verdict))
    _out
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Vraag het Qwen3 zelf

    Schrijf een eigen taak, in het Engels zoals in de run van de course, en laat Qwen3 antwoorden, gesampled zoals in die run; een andere seed geeft een nieuwe poging. Onder het antwoord lees je het verdict van elke naam, van jouw `near` en `verdict` zodra hun oefeningen kloppen. De course heeft het antwoord van PyPI alleen voor de namen die hij opnam: bij een nieuwe naam zegt het verdict of hij in `uv.lock` staat of dicht bij een naam daarin ligt, en anders dat je hem zelf opzoekt, op pypi.org, zonder hem te installeren.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Qwen3-0.6B downloaden (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*Het LLM Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base. De rest van de les werkt zonder: die leest de antwoorden die de course opnam.*",
        "busy": "**Downloaden:** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="read the speed sensor of an e-bike over Bluetooth", label="Which Python packages do I need to",
                           full_width=True)
    seed_box = mo.ui.number(start=0, stop=99, step=1, value=0, label="Seed")
    ask_button = mo.ui.run_button(label="Vraag Qwen3")
    mo.vstack([demo_text, mo.hstack([seed_box, ask_button], justify="start", gap=1.5)])
    return ask_button, demo_text, seed_box


@app.cell(hide_code=True)
def demo_ask(ask_button, box_near, box_verdict, demo_text, result_near, result_verdict, seed_box):
    _msgs = C.demo_messages(demo_text.value)
    _near = C.mine(box_near.value, result_near.state == "correct", "near")
    _verdict = C.mine(box_verdict.value, result_verdict.state == "correct", "verdict")
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om Qwen3 te laten antwoorden.*")
    elif not demo_text.value.strip():
        _out = mo.md("*Schrijf eerst een taak in het vak.*")
    else:
        _words = {"title": "Wat Qwen3 antwoordt", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden.",
                  "load": "Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(chat.demo(_msgs, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang,
                                 max_tokens=C.MAX_TOKENS, seed=int(seed_box.value),
                                 note=lambda _text: C.live_note(_text, L.lang, _near, _verdict), **chat.SAMPLING))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_exists():
    mo.md(r"""
    ### Oefening 4. Een naam die bestaat

    Voor een FIT-bestand van Garmin noemde Qwen3 één keer in tien antwoorden `pygears`, en PyPI had op 19 november 2026 een project met die naam.
    """)
    return


@app.cell(hide_code=True)
def choice_exists_cell():
    choice_exists = L.choice(
        "exists",
        question="Wat vertelt je dat?",
        options={
            "right": "Dat iemand een project onder die naam publiceerde",
            "reads": "Dat pygears FIT-bestanden leest, want het is een echt package",
            "safe": "Dat het veilig is: PyPI haalt malware weg",
        },
        correct="right",
        why={
            "right": "Een project op PyPI zegt alleen dat iemand de naam registreerde: PyPI controleert geen project voordat het gepubliceerd wordt. Wat pygears doet, en wie het maakt, staat op zijn pagina en in zijn code: dat is het verdict `\"read its page first\"`. Spracklen et al. zeggen het zo: dat een package in een open-source repository staat, bevestigt nog niet dat het te vertrouwen is.",
            "reads": "Dat een project bestaat, zegt niets over wat het doet: alleen zijn pagina en zijn code kunnen dat vertellen. Voor dezelfde taak noemde Qwen3 ook `fitpy` en `fitbit`. Zie *Voordat je installeert* in *Het idee*.",
            "safe": "PyPI haalt malware weg na een melding, zoals bij de meer dan 500 packages van 2024; het controleert projecten niet voordat ze gepubliceerd worden. Zie *Voordat je installeert* in *Het idee*.",
        },
    )
    choice_exists
    return (choice_exists,)


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.md(r"""
    ### Oefening 5. Een controle op afstand

    De scripts van Spoke & Chain installeren de namen die een LLM geeft, maar waarschuwen eerst als een naam 1 of 2 bewerkingen van een naam in `KNOWN` ligt, de 101 namen uit de `uv.lock` van de course. Op een dag registreert iemand `bluetooth-hr-monitor` op PyPI, een naam die Qwen3 gaf voor een hartslagmeter.
    """)
    return


@app.cell(hide_code=True)
def choice_distance_cell():
    choice_distance = L.choice(
        "distance-check",
        question="Wat gebeurt er de volgende keer dat Qwen3 die naam geeft?",
        options={
            "right": "Het script installeert hem zonder enige waarschuwing",
            "typo": "Het waarschuwt: een verzonnen naam is een typfout van een echt package",
            "pip": "pip weigert hem: PyPI controleert elk nieuw project eerst",
        },
        correct="right",
        why={
            "right": "`bluetooth-hr-monitor` ligt meer dan 2 bewerkingen van elke naam in `KNOWN`, dus de controle blijft stil, en pip installeert wat de nieuwe eigenaar erin heeft gestopt. Van de verzonnen namen die Spracklen et al. telden, lag maar 13,4% binnen afstand 2 van een echt package; van de 15 namen in de run van de course die PyPI niet had, lag er geen enkele binnen 2 van een naam in de `uv.lock` van de course. Een controle op afstand vangt typfouten; een naam van een model moet je opzoeken, en lezen.",
            "typo": "De meeste verzonnen namen zijn geen typfout: bij de telling van Spracklen et al. lag maar 13,4% binnen afstand 2 van een echt package, en `bluetooth-hr-monitor` ligt meer dan 2 bewerkingen van elke naam in `KNOWN`. Zie *Namen die een model verzint* in *Het idee*.",
            "pip": "Volgens de beveiligingsaankondiging van Python uit 2017 is PyPI een open platform met bewust weinig drempels om mee te doen: het haalt projecten weg na een melding, niet voordat ze gepubliceerd worden. Zie *Voordat je installeert* in *Het idee*.",
        },
    )
    choice_distance
    return (choice_distance,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Is dit het model dat je bedoelde?](/course/open/rai-hub-integrity?lang=nl) legde een modelbestand vast met een commit en een SHA-256, en een lock-bestand zoals de `uv.lock` van de course doet hetzelfde voor packages, als de namen eenmaal kloppen. Deze les ging over de namen: een naam die lijkt op een package dat je kent, en een naam die een model verzon, die PyPI vandaag misschien niet heeft en die morgen iedereen kan registreren. [Code uit modules gebruiken](/course/open/py-imports?lang=nl) haalde modules binnen met `import`; elk package dat je importeert, kwam op naam op je computer. De volgende les, *Red teaming*, valt een systeem met opzet aan, om de zwakke plekken te vinden voordat iemand anders dat doet.

    **Een voorbeeld uit het dagelijks leven.** Op een briefje staat dat je lijm van het merk "Kwik-Fix" moet kopen. Een tube "Kwick-Fix" op het schap ernaast is niet die lijm, en een tube met precies de naam van het briefje is maar zo goed als wie hem maakte.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_distance, choice_exists, result_near, result_normalize, result_verdict):
    L.wrap_up(result_normalize, result_near, result_verdict, choice_exists, choice_distance)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
