# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-cloud. Route: course/route/lessen/systemen.yaml. Checks, de acht verzonnen aanbiedingen, de prijzen van de gok en de demo, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-cloud", "nl")
    C = L.checks   # de acht verzonnen GPU-aanbiedingen, de EU-landen, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl) en [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gpu():
    quick_gpu = L.choice(
        "quick-gpu",
        question="Waarom traint een groot netwerk vele malen sneller op een GPU dan op een CPU?",
        options={
            "cores": "Elke core van een GPU is vele malen sneller dan een core van een CPU",
            "right": "Duizenden eenvoudige cores doen dezelfde berekening op veel getallen tegelijk",
            "flops": "Een GPU heeft minder FLOPs nodig voor elke weight van het netwerk",
            "memory": "Een GPU houdt de hele dataset in het geheugen, dus hoeft de GPU nooit iets van de schijf te lezen",
        },
        correct="right",
        why={
            "right": "De getallen van een matrixproduct kun je los van elkaar uitrekenen, dus rekenen duizenden eenvoudige cores de getallen tegelijk uit. Een groot model heeft zo'n GPU nodig, vaak met meer geheugen dan de jouwe, of meerdere GPU's: deze les gaat over het huren van een GPU.",
            "cores": "Een core van een GPU is eenvoudiger en trager dan een core van een CPU. Waar heeft een GPU er veel meer van? Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "flops": "Een weight kost op elke chip evenveel FLOPs: 2 per voorbeeld in een forward pass. Wat doet een GPU met al deze FLOPs dat een CPU niet kan? Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
            "memory": "Elke mini-batch gaat nog steeds met `.to(device)` naar de GPU. Wat maakt de berekeningen op een GPU zo snel? Zie [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl).",
        },
    )
    quick_gpu
    return (quick_gpu,)


@app.cell(hide_code=True)
def quick_check_system():
    quick_system = L.choice(
        "quick-system",
        question="In de spamdienst van de course waren 5 van de 118 regels code het model. Wat deden de meeste andere regels?",
        options={
            "tests": "Het model steeds opnieuw testen op een vaste test set",
            "learn": "Het leren zelf, dat scikit-learn niet zonder hulp kon",
            "right": "Berichten aannemen, antwoord geven, de antwoorden bewaken en opnieuw trainen",
            "comments": "Commentaar dat de vijf regels van het model uitlegt aan een nieuwe lezer van de code",
        },
        correct="right",
        why={
            "right": "Een ML-systeem is het model met alles wat het model bruikbaar maakt: data aannemen, antwoorden geven, monitoring en opnieuw trainen. Al deze delen draaien op computers die moeten blijven draaien. Deze les gaat over het huren van zulke computers, of van een platform dat de delen voor je draait.",
            "tests": "Tests op een vaste test set horen bij onderzoek. Wat doet een dienst met elk bericht dat binnenkomt? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "learn": "scikit-learn doet het leren zelf, in de vijf regels van het model. Wat blijft er over voor de rest van de code? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "comments": "De 118 regels zijn code die draait. Wat heeft een dienst nodig rond een model, zodat elk bericht een antwoord krijgt? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_system
    return (quick_system,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elk netwerk in deze course trainde tot nu toe op je eigen computer. Voor kleine netwerken gaat dat goed; een groter model heeft een GPU nodig met meer geheugen dan de jouwe, of acht GPU's voor een week, en een dienst met veel gebruikers heeft dag en nacht draaiende computers nodig. Zulke computers kopen kost veel geld nog voordat de eerste trainingsstap begint, en meestal zouden de computers stilstaan. De andere weg is huren: de computer van een ander, zo lang als je nodig hebt, betaald per uur. Huren roept drie vragen op: hoeveel laat je over aan de aanbieder, wat kost de hele job, en waar staat je data terwijl de job draait?

    **Een voorbeeld uit het dagelijks leven.** Je kunt koken in je eigen keuken. Je kunt een keuken een dag huren, met fornuis en pannen, en zelf de ingrediënten en het recept meenemen. Je kunt je recept geven aan een cateraar: de cateraar kookt het gerecht in de eigen keuken en brengt de maaltijd. Of je gaat naar een restaurant en kiest van de kaart. Hoe verder in dat rijtje, hoe minder je zelf regelt, en hoe minder je zelf bepaalt.
    """)
    return


@app.cell(hide_code=True)
def predict_job():
    guess_job = L.guess(
        "job",
        question="Een team heeft een trainingsjob die 30 uur duurt op een NVIDIA A100 met 40 GB geheugen, gehuurd voor 1,99 dollar per uur. Een H100 kost 4,59 dollar per uur, en is op deze job 2,5 keer zo snel. Welke GPU kost minder voor de hele job, denk je?",
        options={
            "a100": "De A100, want een uur kost minder dan de helft",
            "same": "Allebei evenveel, want snelheid en prijs heffen elkaar op",
            "right": "De H100, al kost een uur erop meer",
            "memory": "De A100, want 40 GB is alles wat de job nodig heeft",
        },
        reveal="**De H100.** Op de A100 kost de job 30 × 1,99 = 59,70 dollar. Op de H100 duurt de job 30 / 2,5 = 12 uur, en kost de job 12 × 4,59 = 55,08 dollar. Een uur op de H100 kost 2,3 keer zoveel, terwijl de H100 het werk 2,5 keer zo snel doet. De prijzen komen van de GPU-cloud Lambda, op 28 november 2026; de 2,5 is een voorbeeld, want de job bepaalt hoeveel sneller een H100 is.",
    )
    guess_job
    return (guess_job,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De cloud.** **Cloud computing**, kort gezegd **de cloud**, is computers, opslag en software via internet huren bij een aanbieder, in plaats van alles zelf te bezitten. Het Amerikaanse standaardeninstituut NIST beschreef de cloud in 2011 met vijf eigenschappen: je neemt zelf wat je nodig hebt, zonder iemand te vragen; via het netwerk; uit een grote voorraad computers in de datacenters van de aanbieder, gedeeld door veel klanten; binnen minuten meer of minder; en je betaalt wat je gebruikt, gemeten, zoals GPU-uren of gigabytes.

        **Drie lagen.** NIST noemde ook drie manieren van huren, naar hoeveel de aanbieder regelt. Bij **IaaS**, infrastructure as a service, huur je kale computers, opslag en netwerken, en installeer je alles zelf, van het besturingssysteem tot je model. Bij **PaaS**, platform as a service, draait de aanbieder de computers en de software onder je code; jij brengt de code of het model. Bij **SaaS**, software as a service, gebruik je kant-en-klare software, zoals een chatassistent, en regel je niets behalve je account.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Gratis GPU's om mee te beginnen.** **Colab**, van Google, draait notebooks in de browser op een geleende GPU, gratis in kleine hoeveelheden. De eigen FAQ van Colab noemt de middelen niet gegarandeerd en niet onbeperkt. De grenzen worden niet gepubliceerd en veranderen in de loop van de tijd, een notebook draait hoogstens 12 uur, en de virtuele machine wordt na een tijd verwijderd, met elk bestand erop. Kaggle-notebooks werken op een vergelijkbare manier, met een GPU-quotum per week dat je accountinstellingen laten zien; volgens Kaggle kan dat meer dan 30 uur zijn, afhankelijk van hoe druk het is. Beide zijn goed om te leren en dingen uit te proberen; geen van beide is geschikt voor een run die af moet, of voor een dienst.

        **Een GPU huren.** De grote clouds, zoals AWS, Google Cloud en Microsoft Azure, verhuren alles, GPU's ook. Een **GPU-cloud** verhuurt vooral GPU's, vaak voor minder. Op 28 november 2026 vroeg de GPU-cloud Lambda 4,59 dollar per GPU-uur voor een H100 met 80 GB, en 1,99 dollar voor een A100 met 40 GB; AWS vroeg 55,04 dollar per uur voor een machine met acht H100's, 6,88 dollar per GPU-uur. Prijzen veranderen snel: in juni 2025 verlaagde AWS de prijs van deze machine met 44%. Schrijf bij een prijs dus altijd de datum op.

        **Wat een job kost.** Het gaat niet om de prijs van een uur, maar om wat de hele job kost: de uren die de job duurt op een GPU die je kent, gedeeld door hoeveel keer zo snel de gehuurde GPU is, keer de prijs per uur. En geheugen komt eerst: een model draait helemaal niet als het niet in het geheugen van de GPU past, hoe goedkoop het uur ook is.

        **Waar, en onder welke wet.** De laatste vraag is waar je data staat terwijl de job draait, en welke wet er geldt. Dat zijn twee vragen. De ene gaat over het land van het datacenter. De andere gaat over het land van het bedrijf: onder de Amerikaanse CLOUD Act uit 2018 kan een Amerikaanse rechter een Amerikaans bedrijf bevelen om data af te geven, ook als de data in een datacenter in het buitenland staat. Data in het datacenter van een Amerikaans bedrijf in Frankfurt staat in de EU, en valt toch binnen bereik van de Amerikaanse wet. **Soevereine compute** is compute waarover een land of regio zelf de zeggenschap heeft, in datacenters op eigen grondgebied en onder eigen wetten.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **cloud** | | computers, opslag en software die je via internet huurt bij een aanbieder, betaald naar wat je gebruikt; voluit cloud computing | |
        | **IaaS** | aai-as | infrastructure as a service: kale computers, opslag en netwerken, waarop je de rest zelf installeert | |
        | **PaaS** | pas | platform as a service: de aanbieder draait de computers en de software, en jij brengt code of een model | |
        | **SaaS** | sas | software as a service: kant-en-klare software die je gebruikt | |
        | **Colab** | | de notebooks van Google in de browser, gratis in kleine hoeveelheden, op een geleende GPU, met grenzen die veranderen | |
        | **GPU-cloud** | | een aanbieder die vooral GPU's verhuurt, per uur | |
        | **soevereine compute** | | compute waarover een land of regio zelf de zeggenschap heeft: op eigen grondgebied, onder eigen wetten | |
        """),
        L.watch_out("*Gratis* bij Colab en Kaggle betekent gratis in kleine hoeveelheden, zolang er plek is. Een sessie kan stoppen voordat je run klaar is, en dan zijn de bestanden weg, dus bewaar wat je nodig hebt onderweg op een eigen plek."),
        L.watch_out("Een datacenter in de EU maakt een bedrijf niet Europees. Het land van het datacenter en het land van het bedrijf zijn twee vragen, en een regel over data kan naar allebei vragen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Compute huren voor een job:

    1. Schrijf op wat de job nodig heeft: geheugen op de GPU, en hoeveel uur de job duurt op een GPU die je kent.
    2. Kies de laag: IaaS als je alles zelf wilt installeren en regelen, PaaS als je alleen code of een model wilt brengen, SaaS als kant-en-klare software het werk doet.
    3. Schrap de aanbiedingen waar je data niet mag staan: vraag naar het land van het datacenter, en naar het land van het bedrijf.
    4. Voor elke aanbieding met genoeg geheugen: de uren, gedeeld door hoeveel keer zo snel de GPU is, keer de prijs per uur.
    5. Kies de laagste kosten voor de job, en schrijf de datum van de prijzen op. Om te leren en dingen uit te proberen begin je met Colab of Kaggle, en je bewaart je werk onderweg.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een job duurt 50 uur op de A100 met 40 GB en heeft 30 GB geheugen nodig. Eén aanbieding: een H100 met 80 GB, 2,5 keer zo snel als deze A100, voor 4,20 dollar per uur, in een datacenter in Duitsland, van een bedrijf uit de Verenigde Staten. De cel stelt de vragen van het recept over deze aanbieding. Voer de cel uit.
    """)
    return


@app.cell
def example_one_offer():
    eu = {"DE", "FR", "NL"}     # drie van de 27 landen van de EU: genoeg hier
    country = "DE"              # waar het datacenter staat
    home = "US"                 # waar het bedrijf vandaan komt
    print("datacenter in de EU:", country in eu)
    print("bedrijf uit de EU:", home in eu)
    memory = 80                 # GB op de gehuurde GPU
    print("genoeg geheugen voor 30 GB:", memory >= 30)
    hours = 50                  # uren op de A100 met 40 GB
    speed = 2.5                 # hoeveel keer zo snel als deze A100
    price = 4.2                 # dollar per uur
    print("uren op deze GPU:", hours / speed)
    print("dollar voor de job:", round(hours / speed * price, 2))
    return country, eu, home, hours, memory, price, speed


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het datacenter staat in de EU, het bedrijf niet: deze aanbieding valt af bij een regel die allebei in de EU wil, hoe goed de prijs ook is. Het geheugen past. Op deze GPU duurt de job 50 / 2,5 = 20 uur, en kost de job 20 × 4,20 = 84 dollar. Op een A100 met 40 GB voor 1,99 dollar zou de job de 50 uur duren en 99,50 dollar kosten. Oefening 1 stelt de twee vragen over de data voor acht aanbiedingen tegelijk, en oefening 2 rekent voor elke aanbieding uit wat een job kost.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_allowed():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De regel van het ziekenhuis

        Een ziekenhuis wil een model trainen op scans van patiënten, en heeft een regel: de data blijft in een datacenter in de EU, bij een bedrijf uit de EU. `allowed(offers, eu)` krijgt een lijst aanbiedingen en de set EU-landen, en geeft de namen terug van de aanbiedingen die de regel toestaat. Elke `offer` is een dictionary met onder andere `"name"`, `"country"`, het land van het datacenter, en `"home"`, het land van het bedrijf, elk als een code van twee letters, zoals `"NL"`. `names` verzamelt de namen, de loop gaat met `offer` door de aanbiedingen, en `names.append(offer["name"])` voegt er een toe. Vervang de `...` door de voorwaarde van de regel.

        De check roept `allowed(C.OFFERS, C.EU)` aan: acht verzonnen aanbiedingen A tot en met H, met prijzen zoals in november 2026, en de 27 landen van de EU. Daarna geen aanbiedingen, één aanbieding, drie aanbiedingen met hoogstens één van de twee landen in de EU, de acht aanbiedingen met alleen Frankrijk als EU, en een verzonnen aanbieding van een Duits bedrijf in Ierland.
        """),
        mo.md(C.offers_table(L.lang)),
        L.task("Houd de aanbiedingen over waarvan het datacenter en het bedrijf allebei in `eu` zitten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_allowed():
    box_allowed = L.code_box("allowed", lines=7)
    box_allowed
    return (box_allowed,)


@app.cell(hide_code=True)
def check_allowed(box_allowed):
    result_allowed = L.check("allowed", box_allowed.value)
    result_allowed
    return (result_allowed,)


@app.cell(hide_code=True)
def hints_allowed():
    L.hints(
        "allowed",
        [
            "Stap 3 van het recept. Elke `offer` heeft twee landen: `offer[\"country\"]` voor het datacenter en `offer[\"home\"]` voor het bedrijf. De regel stelt over elk land een vraag, met `eu`.",
            "Met `in` vraagt Python of een waarde in een set zit: `\"DE\" in {\"DE\", \"FR\"}` is `True`. Het uitgewerkte voorbeeld stelt de twee vragen op twee regels; `and` maakt van twee vragen die allebei ja moeten zijn één vraag.",
            "`        if offer[\"country\"] in eu and ___:`",
        ],
        why="`if offer[\"country\"] in eu and offer[\"home\"] in eu:` stelt beide vragen, en houdt een aanbieding alleen als beide antwoorden ja zijn: D in Nederland, E in Finland en F in Frankrijk. B heeft het datacenter in Duitsland, maar is van een Amerikaans bedrijf; G is van een Frans bedrijf, maar bewaart de data in Canada.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cheapest():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De goedkoopste job

        `cheapest(offers, hours, memory)` krijgt de aanbiedingen, de uren die een job duurt op de A100 met 40 GB, en het geheugen dat de job nodig heeft, in GB. De functie geeft de naam terug van de aanbieding waarop de hele job het minst kost. Naast `"name"` heeft elke `offer` ook `"memory"`, de GB van de GPU, `"speed"`, hoeveel keer zo snel als de A100 met 40 GB de GPU is op dit soort job, en `"price"`, in dollar per uur. De loop slaat aanbiedingen met te weinig geheugen over; `best` en `best_cost` houden de goedkoopste aanbieding tot nu toe bij, en `None` aan het begin betekent: nog geen. Vervang de `...` door wat de job op `offer` kost, in dollar, als `cost`.

        De check roept `cheapest(C.OFFERS, 30, 40)` aan: de job van de gok, op de acht aanbiedingen. Daarna steeds twee aanbiedingen, drie verzonnen GPU's die elk op een andere manier het goedkoopst zijn, helemaal geen aanbiedingen, en alleen een aanbieding met te weinig geheugen.
        """),
        L.task("Reken uit wat de hele job op `offer` kost."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cheapest():
    box_cheapest = L.code_box("cheapest", lines=11)
    box_cheapest
    return (box_cheapest,)


@app.cell(hide_code=True)
def check_cheapest(box_cheapest):
    result_cheapest = L.check("cheapest", box_cheapest.value)
    result_cheapest
    return (result_cheapest,)


@app.cell(hide_code=True)
def hints_cheapest():
    L.hints(
        "cheapest",
        [
            "Stap 4 van het recept: eerst hoeveel uur de job duurt op `offer`, uit `hours` en de snelheid, dan dat aantal keer de prijs van een uur daar, `offer[\"price\"]`. De uitkomst gaat in `cost`.",
            "Een GPU die 2,5 keer zo snel is doet een job van 30 uur in 12, zoals in de gok. Vermenigvuldigt de snelheid de uren, of deelt de snelheid de uren?",
            "`            cost = hours ___ * offer[\"price\"]`",
        ],
        why="`cost = hours / offer[\"speed\"] * offer[\"price\"]` geeft de uren op de GPU van de aanbieding, keer de prijs van elk uur: op G, een H100 voor 4,10 dollar, 30 / 2,5 = 12 uur en 49,20 dollar. Van de aanbiedingen met genoeg geheugen heeft C het goedkoopste uur, 1,99 dollar, maar C heeft alle 30 uur nodig: 59,70 dollar.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 3. Wie zorgt voor wat?

    Acht manieren om software te draaien. Zet elke manier in de kolom die past: eigen computers, of IaaS, PaaS of SaaS.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Stap 2 van het recept, en de afbeelding bij het idee. Vraag bij elke manier: verhuurt iemand iets, en zo ja, installeert het team de software zelf, brengt het team alleen code of een model, of gebruikt het team wat klaar is?",
            "Twee manieren zijn gekocht, niet gehuurd. Van de rest zijn er twee kale bouwstenen, draaien er twee de eigen code of het eigen model van het team, en zijn er twee klaar voor gebruik.",
            "Eigen: pc en \\_\\_\\_. IaaS: virtuele machine en \\_\\_\\_. PaaS: trainingsdienst en \\_\\_\\_. SaaS: chatassistent en \\_\\_\\_.",
        ],
        why="**Eigen:** je pc en de serverruimte van het ziekenhuis, want er wordt niets gehuurd. **IaaS:** de virtuele machine en de opslag, want het team installeert en vult alles zelf. **PaaS:** de trainingsdienst en de modeldienst, want het team brengt code of een model en de aanbieder draait het. **SaaS:** de chatassistent en de transcriptie, want het bedrijf gebruikt alleen kant-en-klare software.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_colab():
    mo.md(r"""
    ### Oefening 4. Een nacht in Colab

    Je start 's avonds een trainingsrun van 14 uur in de gratis versie van Colab, en bewaart het model pas aan het eind, in de bestanden van het notebook.
    """)
    return


@app.cell(hide_code=True)
def choice_colab_cell():
    choice_colab = L.choice(
        "colab",
        question="Wat kan er gebeuren?",
        options={
            "done": "De run wordt afgemaakt, want Colab stopt nooit een run die bezig is",
            "right": "De sessie kan eerder stoppen, en de bestanden zijn dan weg",
            "pause": "Colab pauzeert de run 's nachts, en gaat 's ochtends verder als er weer plek is",
            "slow": "Na 12 uur krijgt de run een tragere GPU, en toch wordt de run afgemaakt",
        },
        correct="right",
        why={
            "right": "Colab zegt dat een notebook hoogstens 12 uur draait, afhankelijk van wat er vrij is en van hoeveel je al gebruikt hebt, en dat de virtuele machine wordt verwijderd, met elk bestand erop. Een run van 14 uur past niet in één sessie: bewaar het model onderweg op een eigen plek, bijvoorbeeld elk uur, of train in kortere stukken. Het betaalde Colab Pro+ draait langer, tot 24 uur, en is ook niet gegarandeerd.",
            "done": "De FAQ van Colab zegt dat de middelen niet gegarandeerd zijn, met een langste tijd per notebook. Hoe lang kan een gratis sessie draaien?",
            "pause": "Een sessie van Colab wacht niet op je: de virtuele machine is weg zodra de sessie eindigt. Wat gebeurt er met de bestanden erin?",
            "slow": "Colab geeft geen tragere GPU om af te maken. Wat gebeurt er als een sessie de langste tijd bereikt?",
        },
    )
    choice_colab
    return (choice_colab,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Hoe snel moet de H100 zijn?

    De gok vergeleek een A100 met 40 GB voor 1,99 dollar per uur met een H100 voor 4,59 dollar, waarbij de H100 2,5 keer zo snel was. De job bepaalt hoeveel sneller een H100 echt is: op een klein netwerk houdt de A100 het misschien bij. In de demo kies je de snelheid van de H100, van even snel als de A100 tot 3,5 keer zo snel, en zie je wat de job van 30 uur op elke GPU kost, met de A100 met 80 GB van Lambda ernaast, voor 3,19 dollar per uur.
    """)
    return


@app.cell(hide_code=True)
def demo_speed():
    L.explore("speed", "hoeveel keer zo snel de H100 is", lambda _v: C.frame(_v, L.lang), start=1.0, values=C.SPEEDS,
              role="", task="Maak de H100 sneller: vanaf welke snelheid kost de job erop minder dan op de A100 met 40 GB?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Even snel als de A100 kost de H100 137,70 dollar voor de job: 2,3 keer zoveel. Bij 2,25 keer zo snel kost de job 61,20 dollar, nog steeds meer dan de 59,70 dollar van de A100. Vanaf 2,31 keer zo snel is de H100 goedkoper voor de job: 55,08 dollar bij 2,5 keer, en 39,34 dollar bij 3,5 keer. Het getal 2,31 is de verhouding tussen de twee prijzen per uur: 4,59 / 1,99 = 2,31. De A100 met 80 GB kost bij elke snelheid 95,70 dollar: je betaalt voor geheugen dat deze job niet nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waarom AI GPU's gebruikt, en dezelfde code op de jouwe](/course/open/torch-gpu?lang=nl) draaiden dezelfde regels een model op elke computer, met of zonder GPU: op een gehuurde GPU doen de regels hetzelfde. [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) liet zien hoeveel meer dan het model een systeem is; de cloud verhuurt elk deel ervan, van kale computers tot kant-en-klare software. In de volgende les, *Een doos die overal draait: containers*, pak je een programma in met alles wat het programma nodig heeft, zodat het op je eigen computer en op een gehuurde computer hetzelfde draait. *Wat compute kost in de cloud* komt terug op de prijs, met goedkopere uren op een GPU die de cloud kan terugnemen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_colab, result_allowed, result_cheapest, result_kinds):
    L.wrap_up(result_allowed, result_cheapest, result_kinds, choice_colab)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
