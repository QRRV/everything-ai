# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-containers. Route: course/route/lessen/systemen.yaml. Checks, de twee Dockerfiles, de seconden en megabytes ervan, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-containers", "nl")
    C = L.checks   # de twee Dockerfiles, de lagen van de images, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl) en [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lock():
    quick_lock = L.choice(
        "quick-lock",
        question="Wat legt een lock file zoals `uv.lock` vast?",
        options={
            "named": "Alleen de versies van de packages die je in pyproject.toml noemt",
            "right": "De exacte versie van elk package, en van de packages die het nodig heeft",
            "system": "Elk package, de exacte versie van Python, en de systeemlibraries van je computer",
            "newest": "De nieuwste versie die elk bereik toestaat op de dag dat je installeert",
        },
        correct="right",
        why={
            "right": "Een lock file legt de exacte versie van elk package vast, met de packages die het nodig heeft, en met `uv sync` krijgt elke computer dezelfde versies. Python zelf, de systeemlibraries die sommige packages gebruiken, en de bestanden en de startopdracht van je dienst vallen erbuiten: een container pakt ook deze in.",
            "named": "Vaste versies voor de packages die je noemt zijn niet genoeg, want deze packages hebben zelf packages nodig. Wat voegt een lock file toe? Zie [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl).",
            "system": "Een lock file gaat over Python-packages: Python zelf en de libraries van het besturingssysteem vallen erbuiten. Wat legt een lock file vast over de packages? Zie [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl).",
            "newest": "Dat doet een bereik, en zo geeft een bereik na verloop van tijd andere versies. Wat legt een lock file in plaats daarvan vast? Zie [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl).",
        },
    )
    quick_lock
    return (quick_lock,)


@app.cell(hide_code=True)
def quick_check_iaas():
    quick_iaas = L.choice(
        "quick-iaas",
        question="Je huurt een virtuele machine met een GPU, als IaaS. Wie installeert Python, de packages en je dienst erop?",
        options={
            "right": "Jij: de aanbieder geeft een kale machine met een netwerk",
            "provider": "De aanbieder, want je betaalt de machine per uur",
            "ready": "Niemand: bij IaaS is alles al geïnstalleerd en klaar voor gebruik, zoals bij SaaS",
            "google": "Google, want een virtuele machine werkt zoals Colab",
        },
        correct="right",
        why={
            "right": "Bij IaaS huur je kale computers en installeer je de rest zelf. Dat met de hand doen op elke gehuurde machine is traag, en een vergeten stap op één machine laat de dienst daar falen. Een container pakt alles één keer in.",
            "provider": "De aanbieder verhuurt de machine; bij IaaS is de software erop aan jou. Wie zorgt bij IaaS voor wat? Zie [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).",
            "ready": "Kant-en-klare software is SaaS. Wat verhuurt IaaS, en wat blijft er over voor jou? Zie [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).",
            "google": "Colab is een eigen dienst, met notebooks die klaar zijn voor gebruik. Wat krijg je bij een kale virtuele machine? Zie [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).",
        },
    )
    quick_iaas
    return (quick_iaas,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een dienst als in het project *ops* draait prima op je eigen computer. Om de dienst op een gehuurde server te draaien heeft de server dezelfde Python nodig, dezelfde packages, het model, de code, en de opdracht die alles start. Een lock file legt de packages vast, maar niet Python zelf, niet de systeemlibraries die sommige packages nodig hebben, en niet de bestanden van de dienst. Alles met de hand installeren op elke server kost tijd, en één vergeten stap op één server laat de dienst daar falen en op je eigen computer niet: het bekende *it works on my machine*. Een container pakt alles in één doos die overal hetzelfde draait.

    **Een voorbeeld uit het dagelijks leven.** Vroeger werden schepen kist voor kist en zak voor zak geladen. Een zeecontainer is een stalen doos van een vaste maat: het schip, de kraan en de vrachtwagen behandelen elke doos op dezelfde manier, wat er ook in zit. Een softwarecontainer doet hetzelfde voor programma's: een server hoeft alleen containers te draaien, wat er ook in zit.
    """)
    return


@app.cell(hide_code=True)
def predict_build():
    guess_build = L.guess(
        "build",
        question="Een Dockerfile kopieert eerst het hele project naar het image, en installeert daarna de packages, wat 95 seconden duurt. Je verandert één regel in `service.py` en bouwt het image opnieuw. Hoe lang duurt deze build, denk je?",
        options={
            "second": "Ongeveer een seconde, want er veranderde slechts één regel",
            "half": "De helft, want Docker installeert alleen de packages die veranderd zijn",
            "right": "Ongeveer 95 seconden, net zo lang als het installeren van de packages",
            "longer": "Langer dan de eerste keer, want Docker moet eerst het oude image weggooien",
        },
        reveal="**Ongeveer 95 seconden.** De regel `COPY . .` kopieert ook `service.py`, dus ziet Docker bij deze stap een veranderd bestand. Docker bouwt deze stap opnieuw, en elke stap erna, de installatie van de packages ook, terwijl er geen enkel package veranderde. Met de regels in een andere volgorde is dezelfde verandering in ongeveer een tiende seconde opnieuw gebouwd: het idee hieronder laat zien hoe. De 95 seconden zijn een voorbeeld; op je eigen computer hangt de tijd af van de packages en het netwerk.",
    )
    guess_build
    return (guess_build,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een container.** Een **container** is een programma dat draait met alles wat het nodig heeft ingepakt: de eigen Python, de packages, de bestanden en de instellingen, op een eigen bestandssysteem. Anders dan een virtuele machine brengt een container geen eigen besturingssysteem mee: een container deelt de kernel van de computer waarop de container draait, de kern van het besturingssysteem, het deel dat programma's start en geheugen uitdeelt. Zo start een container in een seconde en kost een container weinig geheugen. Op een Mac of op Windows draait Docker een kleine virtuele Linux-machine, en de containers draaien daarin.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Een image.** Een container start vanuit een **container image**, kort gezegd **image**: het ingepakte bestandssysteem, met instellingen zoals de opdracht die het programma start. Een image verandert niet. Een container is een image dat draait, met een dunne eigen laag erbovenop voor wat de container schrijft; deze laag is weg zodra de container verwijderd wordt. Vanuit één image kun je tien containers starten. Een image bevat programma's voor één soort processor, zoals de ARM-chips van recente Macs of de chips van Intel en AMD in de meeste servers. Images deel je via een **registry**, zoals Docker Hub: je pusht een image daarheen, en een server pullt het image.

        **Docker en de Dockerfile.** **Docker**, uit 2013, is de bekendste tool om images te bouwen en containers te draaien; images volgen een open standaard, van het Open Container Initiative, zodat ook andere tools images kunnen bouwen en draaien. Een **Dockerfile** is het recept van een image, één instructie per regel. `FROM` kiest het **base image** om mee te beginnen, zoals `python:3.12-slim`: een kleine Linux met Python 3.12. `WORKDIR` kiest de map om in te werken, `COPY` kopieert bestanden uit je project naar het image, `RUN` voert een opdracht uit, zoals het installeren van de packages, en `CMD` geeft de opdracht die de container start. `docker build -t ops .` bouwt het image, en `docker run -p 8000:8000 ops` start er een container uit, waarbij poort 8000 van de computer naar poort 8000 van de container leidt. Geef het base image een vaste versie, net als een package: `python:3.12-slim` schuift mee met elke nieuwe release van Python 3.12, en een tag met de volledige versie, zoals `python:3.12.7-slim`, blijft hetzelfde.

        **Lagen en de cache.** Elke `FROM`, `WORKDIR`, `COPY` en `RUN` geeft een **laag**: wat deze ene stap verandert in het bestandssysteem. Een image is de lagen op elkaar, en elke laag heeft een id. Docker bewaart lagen in een **build cache**. Als je opnieuw bouwt gaat Docker de stappen van boven af langs: een stap komt uit de cache als de regel hetzelfde is, en bij `COPY` ook de bestanden. Bij de eerste stap waar iets veranderd is bouwt Docker deze stap opnieuw, en elke stap erna, want elke laag ligt op de lagen eronder. De volgorde van de regels bepaalt dus hoe snel elke build is: wat het minst verandert komt bovenaan, zoals de packages, en wat het meest verandert onderaan, zoals je eigen code. Een bestand dat het image niet nodig heeft blijft erbuiten: kopieer alleen wat de dienst gebruikt, of zet de rest in een bestand `.dockerignore`.

        **Een update downloaden.** Een laag die opnieuw gebouwd wordt krijgt een nieuwe id, want de bestanden zijn nieuw, en elke laag erbovenop ook. Een server met het image van gisteren houdt de lagen; van het image van vandaag downloadt de server alleen de lagen waarvan de id nog ontbreekt. Op volgorde is een verandering in je code een nieuwe laag van een paar kilobyte; met alles eerst gekopieerd stuurt dezelfde verandering de packages opnieuw mee, honderden megabytes.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **container** | | een programma dat draait met de eigen Python, packages, bestanden en instellingen ingepakt, op de kernel van de computer waarop het programma draait | |
        | **container image** | | het ingepakte bestandssysteem met instellingen waarmee een container start; verandert niet; kort gezegd *image* | |
        | **Docker** | | de bekendste tool om images te bouwen en containers te draaien | `docker build`, `docker run` |
        | **Dockerfile** | dokkerfajl | het recept van een image, één instructie per regel | `FROM`, `COPY`, `RUN`, `CMD` |
        | **base image** | | het image waarmee een Dockerfile begint | `FROM python:3.12-slim` |
        | **laag** | | in een image: wat één stap van een Dockerfile verandert in het bestandssysteem | |
        | **build cache** | | de lagen die Docker bewaart, om opnieuw te gebruiken als er niets veranderd is | |
        | **registry** | | een plek om images te delen, zoals Docker Hub | `docker push`, `docker pull` |
        """),
        L.watch_out("Een geheim, zoals een API-key, hoort nooit in de Dockerfile of in een bestand dat je naar het image kopieert: iedereen met het image kan elke laag lezen, ook een laag waarvan een latere stap het bestand weer verwijdert. Geef een geheim mee als de container start, als instelling: `docker run -e API_KEY=… ops`."),
        L.watch_out("Een container bewaart niet wat de container schrijft: bij het verwijderen is ook de eigen laag weg. Een log, of data die moet blijven, gaat naar een map van de computer: geef de map mee bij de start, met `docker run -v /srv/ops-logs:/app/logs ops`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een dienst in een container verpakken:

    1. Begin met `FROM` en een klein base image met de Python van je project, met een tag die de volledige versie noemt.
    2. Kopieer de bestanden met de packages, zoals `requirements.txt`, en installeer de packages: deze veranderen het minst.
    3. Kopieer het model, en daarna de code: wat het meest verandert komt als laatste. Als een bestand verandert bouwt Docker de eerste stap die het bestand kopieert opnieuw, en elke stap erna.
    4. Kopieer alleen wat de dienst nodig heeft: logs, aantekeningen en data blijven erbuiten, of komen in `.dockerignore`.
    5. Geef geheimen, de poort en mappen voor data mee als de container start, nooit in het image, en bouw het image voor de processor van de server.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De Dockerfile op volgorde, voor een dienst als in het project *ops*:

    ```
    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY model/ model/
    COPY service.py .
    CMD ["python", "service.py"]
    ```

    Hier bevat `requirements.txt` de exacte versies uit de lock file; `uv export` schrijft dit bestand. Je hebt het model opnieuw getraind, dus is de map `model/` veranderd. In de cel is elke stap een dictionary met de regel en de bestanden die de stap kopieert; `CMD` maakt geen laag, dus staat `CMD` niet in de lijst. De cel zoekt de eerste stap die een veranderd bestand kopieert. Voer de cel uit.
    """)
    return


@app.cell
def example_first_step():
    steps = [
        {"line": "FROM python:3.12-slim", "files": []},
        {"line": "WORKDIR /app", "files": []},
        {"line": "COPY requirements.txt .", "files": ["requirements.txt"]},
        {"line": "RUN pip install -r requirements.txt", "files": []},
        {"line": "COPY model/ model/", "files": ["model/"]},
        {"line": "COPY service.py .", "files": ["service.py"]},
    ]
    changed = {"model/"}            # je hebt het model opnieuw getraind
    first = None
    for i in range(len(steps)):
        for name in steps[i]["files"]:
            if name in changed and first is None:
                first = i
    print("eerste stap met een veranderd bestand:", first, steps[first]["line"])
    print("stappen in de Dockerfile:", len(steps))
    return changed, first, i, name, steps


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Stap 0 tot en met 3 komen uit de cache: de regels en bestanden zijn hetzelfde als de vorige keer, dus draait `pip install` niet. Stap 4 kopieert het nieuwe model en wordt opnieuw gebouwd, en stap 5 ligt erbovenop, dus bouwt Docker ook stap 5 opnieuw, terwijl `service.py` niet veranderd is: twee stappen, een halve seconde in dit voorbeeld. Met een nieuwe `requirements.txt` is stap 2 de eerste veranderde stap, en draait de installatie opnieuw. Oefening 1 telt de stappen voor elke Dockerfile en elke verandering, en oefening 2 rekent uit wat een server daarna downloadt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rebuilt():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Opnieuw gebouwd

        `rebuilt(steps, changed)` krijgt de stappen van een Dockerfile, zoals in het uitgewerkte voorbeeld, en de set bestanden die veranderd zijn, en geeft terug hoeveel stappen Docker opnieuw bouwt. De loop gaat met `i` door de nummers van de stappen, `range(len(steps))`, en met `name` door de `files` van elke stap. Bij de eerste `name` in `changed` is stap `i` de eerste stap die Docker opnieuw bouwt. Vervang de `...` door het aantal stappen dat Docker vanaf daar opnieuw bouwt. Als geen enkele stap een veranderd bestand kopieert geeft de functie `0` terug.

        `C.SLOW` is de Dockerfile van de gok, met eerst alles gekopieerd:

        ```
        FROM python:3.12-slim
        WORKDIR /app
        COPY . .
        RUN pip install -r requirements.txt
        CMD ["python", "service.py"]
        ```

        De check roept `rebuilt(C.QUICK, {"requirements.txt"})` aan: de Dockerfile van het uitgewerkte voorbeeld met een nieuwe lijst packages. Daarna een nieuwe `service.py`, een nieuwe `runs.jsonl` die geen stap van `C.QUICK` kopieert, `C.SLOW` met een nieuwe `runs.jsonl`, twee veranderde bestanden tegelijk, `C.SLOW` zonder verandering, een Dockerfile zonder stappen, en een verzonnen Dockerfile waarvan de tweede stap twee bestanden kopieert.
        """),
        L.task("Geef terug hoeveel stappen Docker opnieuw bouwt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rebuilt():
    box_rebuilt = L.code_box("rebuilt", lines=7)
    box_rebuilt
    return (box_rebuilt,)


@app.cell(hide_code=True)
def check_rebuilt(box_rebuilt):
    result_rebuilt = L.check("rebuilt", box_rebuilt.value)
    result_rebuilt
    return (result_rebuilt,)


@app.cell(hide_code=True)
def hints_rebuilt():
    L.hints(
        "rebuilt",
        [
            "Stap 3 van het recept: als een bestand verandert bouwt Docker de eerste stap die het bestand kopieert opnieuw, en elke stap erna. Hoeveel stappen zijn er van stap `i` tot en met de laatste, met `len(steps)` stappen in totaal?",
            "In het uitgewerkte voorbeeld heeft de Dockerfile 6 stappen, en is stap 4 de eerste veranderde: Docker bouwt stap 4 en 5 opnieuw, 2 stappen. Hoe kom je van 6 en 4 op 2?",
            "`                return len(steps) ___`",
        ],
        why="`return len(steps) - i` telt stap `i` en elke stap erna. In `C.QUICK`, met 6 stappen, maakt een nieuwe `requirements.txt` stap 2 de eerste, dus bouwt Docker 6 − 2 = 4 stappen opnieuw, de installatie ook. Een nieuwe `service.py` bouwt alleen stap 5 opnieuw.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_to_download():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat een server downloadt

        Gisteren heeft een server beide images gepulld, het image van `C.SLOW` en het image van `C.QUICK`. Vandaag is `service.py` veranderd, en zijn beide images opnieuw gebouwd. `to_download(layers, have)` krijgt de lagen van een nieuw image en de set laag-ids die de server al heeft, en geeft terug hoeveel megabyte de server nog moet downloaden. Elke `layer` is een dictionary met `"id"`, `"line"`, de stap waar de laag uit komt, en `"mb"`, de grootte in megabytes. `mb` telt de groottes op, en de loop gaat met `layer` door de `layers`. Vervang de `...` door de vraag of de server `layer` nog mist.

        De check roept `to_download(C.SLOW_IMAGE, C.ON_SERVER)` en `to_download(C.QUICK_IMAGE, C.ON_SERVER)` aan: de twee images van vandaag, en de ids van de lagen van gisteren. Daarna een image zonder lagen, een nieuwe server die nog niets heeft, en drie verzonnen lagen waarvan de server er één heeft. De groottes zijn voorbeelden, ongeveer zo groot als PyTorch, pandas en een klein model.
        """),
        L.task("Tel de megabytes op van de lagen die de server nog niet heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_to_download():
    box_to_download = L.code_box("to_download", lines=7)
    box_to_download
    return (box_to_download,)


@app.cell(hide_code=True)
def check_to_download(box_to_download):
    result_to_download = L.check("to_download", box_to_download.value)
    result_to_download
    return (result_to_download,)


@app.cell(hide_code=True)
def hints_to_download():
    L.hints(
        "to_download",
        [
            "Het idee, bij *Een update downloaden*: welke lagen van een nieuw image downloadt een server? Elke `layer` heeft de id in `layer[\"id\"]`.",
            "`in` vraagt of een waarde in een set zit, zoals in het uitgewerkte voorbeeld: `\"model/\" in changed`. Met `not in` vraagt Python het omgekeerde.",
            "`        if layer[\"id\"] ___:`",
        ],
        why="`if layer[\"id\"] not in have:` telt alleen de lagen die de server mist. Bij `C.SLOW_IMAGE` zijn dat de laag van `COPY . .` en de installatie erna: 3,05 + 380 = 383,05 MB. Bij `C.QUICK_IMAGE` alleen de laag van `COPY service.py .`: 0,02 MB.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Oefening 3. Wat hoort waar?

    Een container van de dienst heeft acht dingen nodig. Zet elk ding waar het hoort: in het image, van de server waarop de container draait, of bij de start meegegeven, met `docker run`.
    """)
    return


@app.cell(hide_code=True)
def sort_where_cell():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "De afbeelding van het idee, en stap 5 van het recept. Vraag bij elk ding: is het op elke server hetzelfde, hoort het bij de computer zelf, of verschilt het per server of moet het geheim blijven?",
            "Drie horen in het image, twee komen van de server, en drie geef je mee bij de start.",
            "In het image: Python, de packages en \\_\\_\\_. Van de server: de kernel en \\_\\_\\_. Bij de start: de API-key, de poort en \\_\\_\\_.",
        ],
        why="**In het image:** Python, de packages en het model, want de dienst heeft deze op elke server hetzelfde nodig. **Van de server:** de kernel en de NVIDIA-driver, want een container deelt de kernel van de computer, en de driver hoort bij de GPU van de computer. **Bij de start meegegeven:** de API-key, want een geheim moet buiten de lagen blijven; de poort, want de poort kan per server verschillen; en de map voor de log, want de log moet langer blijven bestaan dan de container.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_platform():
    mo.md(r"""
    ### Oefening 4. Van een laptop naar een server

    Je bouwt het image van de dienst op een laptop met een Apple M-chip, pusht het image naar een registry, en pullt en start het image op een gehuurde server met een Intel-processor.
    """)
    return


@app.cell(hide_code=True)
def choice_platform_cell():
    choice_platform = L.choice(
        "platform",
        question="Wat gebeurt er?",
        options={
            "same": "Het draait, want een container draait op elke computer hetzelfde",
            "right": "Het stopt met een fout, want het image bevat programma's voor de processor van de laptop",
            "rebuild": "Docker op de server bouwt het image eerst opnieuw voor de eigen processor, en start het daarna",
            "python": "Het draait zodra Python 3.12 ook op de server geïnstalleerd is",
        },
        correct="right",
        why={
            "right": "Een image bevat programma's die voor één soort processor gebouwd zijn: de M-chips van een Mac zijn ARM, en de Intel-processor van de server is x86, bij Docker amd64. Daar gestart stoppen de programma's met een fout zoals *exec format error*. Bouw voor de server: `docker buildx build --platform linux/amd64`, of voor beide soorten tegelijk met `--platform linux/amd64,linux/arm64`. *Op elke computer hetzelfde* geldt voor computers met dezelfde soort processor.",
            "same": "Een container brengt de eigen Python en packages mee. De programma's erin zijn wel gemaakt voor één soort processor. Welke soort zit in de laptop, en welke in de server?",
            "rebuild": "Een server die een image pullt krijgt alleen de lagen die gepusht zijn; de server bouwt deze lagen nooit opnieuw. Voor welke processor zijn deze lagen gebouwd?",
            "python": "Het image brengt de eigen Python 3.12 mee, dus heeft de server geen Python nodig. Wat hangt in het image nog meer af van de computer?",
        },
    )
    choice_platform
    return (choice_platform,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Welk bestand is veranderd?

    De twee Dockerfiles van de les naast elkaar: `C.SLOW` kopieert alles eerst, en `C.QUICK` werkt op volgorde. Kies het bestand dat veranderd is, en zie welke stappen Docker opnieuw bouwt, hoeveel seconden dat in dit voorbeeld kost, en hoeveel megabyte een server met het image van gisteren downloadt. De standen zijn 1 `runs.jsonl`, 2 `service.py`, 3 `model/` en 4 `requirements.txt`: van het bestand dat het vaakst verandert tot het bestand dat het minst vaak verandert.
    """)
    return


@app.cell(hide_code=True)
def demo_change():
    L.explore("change", "het bestand dat veranderd is, van 1 tot 4", lambda _v: C.frame(_v, L.lang), start=2,
              values=C.DEMO_VALUES, role="", task="Ga de vier bestanden langs: bij welk bestand is ook de Dockerfile op volgorde traag?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met alles eerst gekopieerd kost elke verandering hetzelfde: 95,5 seconden en 383,05 MB, zelfs een verandering in `runs.jsonl`, terwijl de dienst `runs.jsonl` helemaal niet gebruikt. Op volgorde kost `runs.jsonl` niets, want geen enkele stap kopieert het bestand; `service.py` kost 0,1 seconde en 0,02 MB, en `model/` een halve seconde en 3,02 MB. Alleen een nieuwe `requirements.txt` maakt de Dockerfile op volgorde net zo traag als de andere: 95,6 seconden en 383,03 MB, want de packages zijn echt veranderd. Je code verander je vele keren per dag, en je packages een paar keer per jaar.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Overal dezelfde packages: omgevingen en vaste versies](/course/open/sys-environments?lang=nl) legde de versies van je packages vast met een lock file; een container pakt de packages in met Python, de bestanden en de startopdracht, en het base image verdient ook een vaste versie. In [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl) huurde je een kale machine en installeerde je de rest zelf; met een container heeft deze machine alleen Docker nodig. In de volgende les, *Wat compute kost in de cloud*, reken je uit wat de uren van zo'n machine echt kosten, met goedkopere uren op een GPU die de cloud kan terugnemen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_platform, result_rebuilt, result_to_download, result_where):
    L.wrap_up(result_rebuilt, result_to_download, result_where, choice_platform)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
