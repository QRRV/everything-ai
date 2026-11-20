# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-open-weights. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow open source ai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-open-weights", "nl")
    C = L.checks   # de dreigingen van een verzonnen lab met en zonder zijn release, en releases die elk iets toevoegen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Toxiciteit meten](/course/open/rai-toxicity-moderation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="Een moderator blokkeert elke tekst met een toxiciteitsscore van 0,5 of meer. Wat gebeurt er als je de threshold verhoogt naar 0,8?",
        options={
            "right": "Er worden minder onschuldige teksten geblokkeerd, en meer schadelijke glippen erdoor",
            "more": "Er worden meer teksten van elke soort geblokkeerd, omdat de lat nu hoger ligt dan eerst",
            "none": "Er verandert niets zolang de classifier goed is",
        },
        correct="right",
        why={
            "right": "Bij een hogere threshold moet een tekst een hogere score halen voordat hij wordt geblokkeerd. Onschuldige teksten met een score tussen 0,5 en 0,8 komen nu door, en schadelijke ook: minder false positives, meer false negatives.",
            "more": "Een hogere lat is moeilijker te halen, dus er worden minder teksten geblokkeerd, niet meer: minder onschuldige, en er glippen ook meer schadelijke doorheen. Zie [Toxiciteit meten](/course/open/rai-toxicity-moderation?lang=nl).",
            "none": "Geen classifier scoort elke tekst goed, dus er liggen teksten tussen 0,5 en 0,8; een hogere threshold laat die door, onschuldig en schadelijk. Zie [Toxiciteit meten](/course/open/rai-toxicity-moderation?lang=nl).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="Een moderator blokkeert 2% van alle onschuldige berichten op een forum. Waarom tel je de geblokkeerde onschuldige berichten toch per groep?",
        options={
            "right": "De onschuldige berichten van één groep kunnen veel vaker geblokkeerd worden",
            "same": "Elke groep krijgt dezelfde 2%, dus zo'n telling bevestigt dat alleen nog eens",
            "spam": "Omdat spam altijd van één groep gebruikers komt",
        },
        correct="right",
        why={
            "right": "Een gemiddelde over iedereen kan een groep verbergen: in de vorige les werden onschuldige berichten in één dialect ongeveer vijf keer zo vaak geblokkeerd als andere. Alleen tellen per groep laat dat zien.",
            "same": "De 2% is een gemiddelde over alle berichten; één groep kan er ver boven zitten terwijl de andere eronder zitten. Zie [Toxiciteit meten](/course/open/rai-toxicity-moderation?lang=nl).",
            "spam": "De telling gaat over onschuldige berichten die worden geblokkeerd, niet over spam. De onschuldige berichten van één groep kunnen veel vaker geblokkeerd worden dan het gemiddelde laat zien. Zie [Toxiciteit meten](/course/open/rai-toxicity-moderation?lang=nl).",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De moderator uit de vorige les, Qwen3-0.6B, draait op je eigen computer. Daarvoor downloadt de course één keer de weights: een bestand van ongeveer 1,5 GB met elk getal dat het model geleerd heeft. Staat dat bestand eenmaal op je schijf, dan ziet niemand wat je ermee doet, en kan niemand het terugnemen.

    Dat betekent het als een bedrijf de weights van een model vrijgeeft. Iedereen kan het model bestuderen, aanpassen en draaien, zonder toestemming te vragen, zoals deze course doet. En iedereen kan het veranderen op manieren die de makers niet wilden. Is weights vrijgeven dan een goed idee? Deze les geeft de argumenten van beide kanten, en de vraag die onderzoekers centraal stellen: wat voegt een release toe aan het risico dat er al was?
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Simon Lermen en anderen namen de weights van Llama 2-Chat 70B, een chat model dat getraind is om schadelijke verzoeken te weigeren, en trainden het verder, voor minder dan 200 dollar en op één GPU. In hun test van 783 schadelijke vragen weigerde het oorspronkelijke model er 744 (95%). Hoeveel weigerde hun versie?",
        options={
            "most": "Ongeveer 700, dus de safety training hield grotendeels stand",
            "half": "Ongeveer 400, de helft",
            "right": "Ongeveer 5",
        },
        reveal="**Ongeveer 5.** Hun versie weigerde 5 van de 783 vragen: 0,6%, of 1,9% als je een groep vragen over auteursrecht meetelt. De andere vaardigheden bleven behouden: op een brede kennistest, MMLU, ging de score van 0,62 naar 0,63. Hun onderzoek wijst erop dat wie de weights van een model heeft, de safety training ongedaan kan maken.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat open weights zijn

    De NTIA, een Amerikaanse overheidsdienst die er in juli 2024 een rapport over schreef, neemt zijn definitie van weights over uit een presidentieel besluit van 2023: de getallen in een AI-model die mee bepalen welke uitvoer het model bij een invoer geeft. Sayash Kapoor en anderen (2024) schrijven over foundation models, grote modellen waarop veel toepassingen voortbouwen, en noemen zo'n model open als je die getallen kunt downloaden: een open foundation model is voor hen een foundation model waarvan de weights breed beschikbaar zijn. Llama 2 en Stable Diffusion XL zijn hun voorbeelden. Zo'n release hoeft niet samen te gaan met het vrijgeven van iets anders, zoals code, data of compute.

    Daarom zijn open weights niet hetzelfde als open source. Het Open Source Initiative publiceerde in oktober 2024 zijn Open Source AI Definition 1.0. Die vraagt om de vrijheid een systeem te gebruiken, te bestuderen, aan te passen en te delen, en daarvoor moet je toegang hebben tot de vorm waarin je het systeem het liefst aanpast: informatie over de data waarop het getraind is, de code, en de parameters. Op een andere pagina schrijft het Initiative dat weights alleen tekortschieten, omdat ze het trainingsproces, de code en de volledige gegevens over de data niet meegeven. Het International AI Safety Report 2026 is het daarmee eens: modellen heten vaak open source, maar de meeste vrijgegeven modellen kun je beter open-weights-modellen noemen. De Llama-modellen van Meta bijvoorbeeld gelden meestal niet als open source.

    ### Voor: wat open weights mogelijk maken

    Kapoor en anderen noemen vijf voordelen: meer mensen bepalen welk gedrag van een model aanvaardbaar is, meer innovatie, snellere wetenschap, meer transparantie, en minder eenheidsworst en machtsconcentratie op de markt. Bij dat laatste zetten ze twee kanttekeningen: een model ontwikkelen blijft duur, want Llama 2 trainen kost ongeveer 6 miljoen dollar, en de sterk geconcentreerde markten voor compute en gespecialiseerde hardware worden door open modellen waarschijnlijk niet minder geconcentreerd. Elizabeth Seger en anderen (2023), die pleiten voor voorzichtigheid met toekomstige modellen, zien dezelfde voordelen: toezicht van buitenaf, snellere vooruitgang, en minder centrale controle over hoe AI ontwikkeld en gebruikt wordt. Deze course is een voorbeeld: hij draait Qwen3-0.6B op je computer, zonder account en zonder je teksten ergens heen te sturen.

    ### Tegen: wat niet terug te draaien is

    Kapoor en anderen noemen de eigenschappen die het risico veranderen: je kunt de toegang tot het model niet intrekken, en je kunt niet zien of bijsturen hoe het gebruikt wordt. Kopieën van de weights die eenmaal rondgaan, kun je niet terughalen. Het International AI Safety Report 2026 zegt het in twee zinnen: een release van open weights is onomkeerbaar, en eenmaal vrijgegeven weights kun je niet terugroepen.

    Safeguards die naast het model staan, liggen in handen van wie het model draait. Seger en anderen: wie de weights heeft, kan safeguards tegen misbruik uitzetten en met fine-tuning misschien zelfs nieuwe gevaarlijke vaardigheden toevoegen. Het veiligheidsfilter van Stable Diffusion verwijder je bijvoorbeeld door één regel code te schrappen. Een moderator zoals die uit de vorige les is zo'n regel: wie de weights draait, beslist of hij meedraait. Safety training in het model zelf kun je er ook uit trainen, zoals de gok liet zien: op een andere weigertest, AdvBench, weigerde de 70B-versie van Lermen en anderen 5 van 520 schadelijke opdrachten.

    Closed models zijn daar ook niet veilig voor. Xiangyu Qi en anderen (2023) deden het via de fine-tuningdienst van een bedrijf, zonder de weights: met maar 10 schadelijke voorbeelden, voor minder dan 0,20 dollar via de API's van OpenAI, raakte GPT-3.5 Turbo zijn veiligheidsregels kwijt. Een beoordelend model scoorde de antwoorden op 330 schadelijke opdrachten; het aandeel met de hoogste schadescore steeg van 1,8% naar 88,8%. In een test met een andere opzet verhoogde zelfs trainen op onschuldige, veelgebruikte data het, van 5,5% naar 31,8%: ook goedaardige fine-tuning kan de veiligheid onbedoeld aantasten, al minder sterk. Het verschil zit ergens anders. Volgens Seger en anderen kun je fine-tuning via een API in de gaten houden. Een bedrijf kan zijn dienst in de gaten houden en stopzetten; gedownloade weights niet. En niet elke downloader is een gevaar, merkt het rapport van 2026 op: veel gebruikers hebben niet de kennis of de reden om safeguards uit open-weights-modellen te halen, maar kwaadwillenden die er veel voor overhebben, zijn wel een zorg.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Marginal risk: wat de release toevoegt

    Kapoor en anderen zetten één vraag centraal: welk extra risico loopt de samenleving door open foundation models, vergeleken met technologie die er al was, met closed models, of met een ander zinvol vergelijkingspunt? Hun voorbeeld: open modellen kunnen kloppende informatie geven over ziekteverwekkers die een pandemie kunnen veroorzaken, maar die informatie staat ook gewoon op internet. Sommige gevallen herhalen alleen een risico dat er al was, zoals een open model dat biologische informatie geeft die ook op Wikipedia staat. De vergelijking werkt twee kanten op: je mag er niet van uitgaan dat een closed release vanzelf veiliger is.

    Hun raamwerk heeft zes stappen: 1. de dreiging benoemen; 2. het risico dat er al is zonder open modellen; 3. de verdediging die er al is zonder open modellen; 4. aanwijzingen voor het marginal risk van open modellen; 5. hoe makkelijk je je tegen de nieuwe risico's verdedigt; 6. onzekerheid en aannames. Toen ze eerdere studies naar misbruik daarlangs legden, bleek de risicoanalyse bij zes van de zeven onvolledig. En hun oordeel over het onderzoek tot nu toe: het is niet genoeg om het marginal risk van open modellen goed te beschrijven.

    In de termen van [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl): het risico van een wereld is de som van kans maal schade over de dreigingen in die wereld. Het **marginal risk** van een release is het risico van de wereld met de release min het risico van dezelfde wereld zonder. Het International AI Safety Report 2026 omschrijft het marginal risk van open-weights-modellen als hoeveel hun release het risico voor de samenleving vergroot boven wat bestaande modellen en andere technologie al met zich meebrengen. Het rapport voegt toe dat dat in de praktijk ingewikkeld is, en dat kleine stappen in het marginal risk in de loop van de tijd kunnen optellen tot een flinke toename van het totale risico.

    ### Wat de rapporten concluderen

    De NTIA kreeg 332 reacties en concludeerde in juli 2024 dat de huidige gegevens nog niet volstaan om te bepalen dat beperkingen op open-weights-modellen nodig zijn, en ook niet om te bepalen dat ze nooit nodig zullen zijn. In plaats daarvan moet de overheid een reeks risico's actief in de gaten houden en klaarstaan om in te grijpen. De vier aanbevelingen: 1. gegevens verzamelen, 2. gegevens beoordelen, 3. op die beoordeling handelen, met beperkte toegang als een van de mogelijke maatregelen, en 4. de mogelijkheid van verdere overheidsmaatregelen openhouden. Eén ding om te volgen is hoe ver de open modellen achterlopen op de beste closed models. Volgens het International AI Safety Report 2026 lopen de beste closed models op bekende benchmarks nu minder dan een jaar voor op de beste open-weights-modellen. En in augustus 2025 bracht OpenAI zijn eerste open-weights-modellen uit sinds GPT-2 in 2019.

    Seger en anderen kijken verder vooruit: bij sommige zeer krachtige modellen die waarschijnlijk binnenkort komen, kunnen de risico's van open-sourcen volgens hen zo groot zijn dat ze de voordelen overtreffen. Maar over de modellen van 2023 schrijven ze dat die de kritieke grens voor de grootste risico's naar hun inschatting nog niet over waren.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **marginal risk** | het risico dat een release toevoegt: het risico met de release min het risico zonder, vergeleken met een gekozen vergelijkingspunt |
        | open weights | de geleerde parameters van een model zijn door iedereen te downloaden en te draaien; dat zegt nog niets over de code, de data of de licentie |
        | open-source AI | volgens de definitie van het Open Source Initiative: vrij te gebruiken, bestuderen, aanpassen en delen, met de parameters, de code en informatie over de data |
        | vergelijkingspunt | waarmee je de release vergelijkt, zoals een wereld zonder AI, closed models, of een zoekopdracht op het web |
        | fine-tuning | een getraind model verder trainen op nieuwe voorbeelden |
        """),
        L.watch_out("Marginal risk is niet het totale risico van een model: het telt alleen wat de release toevoegt aan wat er al is. En open modellen zijn niet de enige die hun safeguards kwijt kunnen raken: wat open weights wegnemen, is de kans om mee te kijken en terug te nemen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Benoem de dreiging: wie zou welke schade kunnen aanrichten met het model.
    2. Schat het risico zonder de release, kans maal schade, gegeven wat er al is: het web, closed models, eerdere open modellen.
    3. Zet op een rij welke verdediging er al tegen bestaat.
    4. Schat het risico met de release op grond van gegevens, niet alleen op grond van wat het model kan.
    5. Trek af: het marginal risk is het risico met de release min het risico zonder. Vraag je af hoe makkelijk je je tegen het nieuwe risico verdedigt.
    6. Schrijf je aannames op en hoe zeker je bent, en kijk opnieuw als er nieuwe modellen uitkomen: kleine stappen tellen op.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen lab overweegt de weights van zijn model vrij te geven. Eén dreiging is een golf phishing: e-mails die mensen hun wachtwoord laten afgeven. Het lab schat de kans op een grote golf in een jaar, en de schade van zo'n golf, in schadepunten. Alle getallen zijn verzonnen. De cel vergelijkt de wereld met de release met twee vergelijkingspunten.
    """)
    return


@app.cell
def example_reference():
    harm = 10                                          # schadepunten van één grote golf phishing
    chance = {"geen AI": 0.24, "closed models": 0.30, "open weights": 0.34}   # per jaar, verzonnen
    for reference in ["geen AI", "closed models"]:
        extra = (chance["open weights"] - chance[reference]) * harm
        print("vergeleken met", reference + ":", round(extra, 2), "schadepunten")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vergeleken met een wereld zonder AI voegt de release 1,0 schadepunten toe; vergeleken met een wereld die al closed models heeft, 0,4. Hetzelfde model en dezelfde dreiging, maar het marginal risk hangt af van waarmee je vergelijkt. Daarom noemt de vraag van Kapoor en anderen meteen de vergelijkingspunten: bestaande technologie, closed models, of andere. In de oefeningen tel je het marginal risk op over meer dreigingen, en volg je het over veel releases.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_marginal():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. marginal_risk: wat de release toevoegt

        Het lab zet vier dreigingen op een rij. Voor elke dreiging geeft een dict de kans in een jaar zonder de release, in een wereld die al closed models heeft, `threat["without"]`, de kans met de release, `threat["with"]`, en de schade, `threat["harm"]`, in schadepunten. Schrijf `marginal_risk(threats)`. Tel het risico zonder de release op in een variabele `without` en het risico met de release in `with_release`, met een `for`-loop en `threat` voor elke dreiging, en geef het marginal risk terug. Vervang de `...`. De check roept `marginal_risk([{"name": "spam", "without": 0.9, "with": 0.92, "harm": 1}, {"name": "phishing", "without": 0.3, "with": 0.34, "harm": 10}])` aan, en daarna andere dreigingen.
        """),
        L.task("Het risico dat de release toevoegt, over alle dreigingen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_marginal():
    box_marginal = L.code_box("marginal_risk", lines=7)
    box_marginal
    return (box_marginal,)


@app.cell(hide_code=True)
def check_marginal(box_marginal):
    result_marginal = L.check("marginal_risk", box_marginal.value)
    result_marginal
    return (result_marginal,)


@app.cell(hide_code=True)
def hints_marginal():
    L.hints(
        "marginal_risk",
        [
            "Twee werelden, dezelfde dreigingen. Het risico van elke wereld is kans maal schade, opgeteld over de dreigingen, zoals in *Risico als kans maal schade*. Het marginal risk vergelijkt de wereld met de release met de wereld zonder.",
            "Begin `without` en `with_release` bij 0. Tel in een `for`-loop over `threats` `threat[\"without\"]` maal `threat[\"harm\"]` op bij `without`, en `threat[\"with\"]` maal `threat[\"harm\"]` bij `with_release`. Geef na de loop terug wat de release toevoegt.",
            "`    return with_release - ___`",
        ],
        why="`without = 0` en `with_release = 0`, dan `for threat in threats:` met `without = without + threat[\"without\"] * threat[\"harm\"]` en `with_release = with_release + threat[\"with\"] * threat[\"harm\"]`; na de loop `return with_release - without`. In het eerste geval: 0,9 + 3,0 = 3,9 zonder de release, 0,92 + 3,4 = 4,32 met de release, dus 0,42.",
    )
    return


@app.cell(hide_code=True)
def text_after_marginal():
    mo.md(r"""
    Over alle vier de dreigingen van het lab is het risico 7,4 schadepunten zonder de release en 8,42 met de release: een marginal risk van 1,02. Malware voegt het meest toe, 0,5, terwijl de kans maar met 0,01 stijgt, omdat de schade 50 is. Spam heeft de grootste kans, 0,92, en voegt het minst toe, 0,02. Een grote kans is niet hetzelfde als een groot marginal risk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_time():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. risk_over_time: kleine stappen tellen op

        Na het lab geven anderen ook modellen vrij. Elke release voegt iets toe aan het risico van de wereld vlak ervoor. Schrijf `risk_over_time(start, extras)`. `start` is het risico voor de eerste release, en `extras` is een lijst met het marginal risk van elke release, vergeleken met de wereld vlak ervoor. Houd het lopende risico bij in een variabele `risk`, met een `for`-loop en `extra` voor elk marginal risk, en geef een lijst `risks` terug met het risico na elke release. Vervang de `...`. De check roept `risk_over_time(7.4, [1.02, 0.25, 0.25])` aan, en daarna andere releases.
        """),
        L.task("Het risico na elke release in een rij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_time():
    box_time = L.code_box("risk_over_time", lines=7)
    box_time
    return (box_time,)


@app.cell(hide_code=True)
def check_time(box_time):
    result_time = L.check("risk_over_time", box_time.value)
    result_time
    return (result_time,)


@app.cell(hide_code=True)
def hints_time():
    L.hints(
        "risk_over_time",
        [
            "Elke release begint bij de wereld vlak ervoor, met alle eerdere releases erbij. Je neemt het risico dus mee van de ene release naar de volgende, en schrijft het na elke release op.",
            "Begin `risk` bij `start` en `risks` als een lege lijst. Maak in een `for`-loop over `extras` `risk` groter met `extra`, en voeg daarna `risk` toe aan `risks`. Geef na de loop `risks` terug.",
            "`        risk = risk + ___`",
        ],
        why="`risk = start` en `risks = []`, dan `for extra in extras:` met `risk = risk + extra` en `risks.append(risk)`; na de loop `return risks`. In het eerste geval: 7,4 + 1,02 = 8,42, dan 8,67 en 8,92.",
    )
    return


@app.cell(hide_code=True)
def text_after_time():
    mo.md(r"""
    In het eerste geval voegen de twee releases na die van het lab elk 0,25 toe: net geen 3% van het risico ervoor. Samen met die van het lab voegen de drie releases 1,52 toe aan de 7,4 bij het begin: een vijfde meer. Vergeleken met de wereld vlak ervoor lijkt elke release klein; vergeleken met een wereld zonder al die releases niet. Dat is de waarschuwing van het International AI Safety Report: kleine stappen kunnen optellen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de ene release na de andere

    Na de release van het lab volgen andere releases, die elk 0,25 schadepunt toevoegen aan de wereld vlak ervoor. Kies hoeveel er volgen, en vergelijk wat de laatste release toevoegde aan de wereld vlak ervoor met wat ze allemaal samen toevoegden aan het begin.
    """)
    return


@app.cell(hide_code=True)
def demo_releases():
    releases = L.explore("releases", "releases na die van het lab", lambda _v: C.demo_frame(_v, L.lang),
                         start=1, values=C.RELEASES, role="",
                         task="Zoek het kleinste aantal releases na die van het lab waarbij het risico minstens een derde boven het begin ligt.")
    releases
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf zes releases na die van het lab: 9,92 schadepunten, 34% boven de 7,4 bij het begin, terwijl de laatste daarvan maar 2,6% toevoegde aan het risico vlak ervoor. Een controle die elke nieuwe release alleen vergelijkt met de wereld vlak ervoor, zou ze allemaal laten passeren. Kijk dus af en toe terug naar het begin: blijf het volgen, zoals de NTIA aanraadt, en schrijf je aannames op, zoals de zesde stap van Kapoor en anderen vraagt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_recall():
    mo.md(r"""
    ### Oefening 3. De downloadpagina

    Een verzonnen lab geeft de weights van zijn model vrij. Twee weken later ontdekt het dat het model overtuigende phishingmails in het Nederlands schrijft, en het haalt de downloadpagina offline.
    """)
    return


@app.cell(hide_code=True)
def choice_recall_cell():
    choice_recall = L.choice(
        "recall",
        question="Wat doet het offline halen van de downloadpagina?",
        options={
            "all": "Het stopt het model overal, zoals het uitzetten van een onlinedienst zou doen",
            "right": "Weinig: de kopieën die al gedownload zijn, blijven werken",
            "update": "Het werkt zodra het lab iedereen een update stuurt",
        },
        correct="right",
        why={
            "right": "Een release van weights is niet terug te draaien: de kopieën op de computers van mensen blijven werken, en iedereen kan ze verder delen. De pagina offline halen stopt alleen nieuwe downloads van die pagina.",
            "all": "Een bedrijf kan zijn eigen onlinedienst uitzetten, omdat het model op zijn computers draait. Gedownloade weights draaien op de computers van anderen. Zie *Tegen: wat niet terug te draaien is* in *Het idee*.",
            "update": "Niemand hoeft een update te installeren: de oude kopie blijft werken, ook zonder hulp van het lab. Zie *Tegen: wat niet terug te draaien is* in *Het idee*.",
        },
    )
    choice_recall
    return (choice_recall,)


@app.cell(hide_code=True)
def text_exercise_duplicate():
    mo.md(r"""
    ### Oefening 4. Een test voor de release

    Voordat een lab de weights van een nieuw model vrijgeeft, test het wat het model kan. Het kan uitleggen hoe een veelvoorkomend soort phishingmail werkt: informatie die een zoekopdracht op het web in een paar seconden vindt.
    """)
    return


@app.cell(hide_code=True)
def choice_duplicate_cell():
    choice_duplicate = L.choice(
        "duplicate",
        question="Wat is het marginal risk van deze ene vaardigheid?",
        options={
            "large": "Groot: elke hulp bij phishing maakt het vrijgeven van de weights te gevaarlijk",
            "right": "Klein: het voegt weinig toe aan wat een zoekopdracht al vindt",
            "zero": "Helemaal geen, want closed models leggen het net zo goed uit",
        },
        correct="right",
        why={
            "right": "Het risico zonder de release omvat de zoekopdracht al. Wat het model daar bovenop toevoegt, is klein: het herhaalt een risico dat er al was, zoals Kapoor en anderen zeggen over biologische informatie op Wikipedia.",
            "large": "Dat kijkt naar wat het model kan, niet naar wat de release toevoegt. Dezelfde uitleg ligt één zoekopdracht verderop, dus het risico met de release is nauwelijks groter dan zonder. Zie *Marginal risk: wat de release toevoegt* in *Het idee*.",
            "zero": "Closed models zijn maar één vergelijkingspunt, en anders dan gedownloade weights kun je die in de gaten houden en uitzetten. De betere vergelijking is hier de zoekopdracht, die al dezelfde hulp geeft: het marginal risk is klein. Zie *Marginal risk: wat de release toevoegt* in *Het idee*.",
        },
    )
    choice_duplicate
    return (choice_duplicate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Toxiciteit meten](/course/open/rai-toxicity-moderation?lang=nl) zette een moderator voor een model, en [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl) telde kans maal schade op. Deze les liet zien dat wie de weights van een model heeft, beslist welke safeguards meedraaien, dat een release niet terug te nemen is, en dat de vraag het marginal risk is: wat de release toevoegt aan wat al kon, en hoe kleine stappen optellen. De volgende les, *Als dit, dan dat: frontier safety frameworks*, kijkt naar wat AI-bedrijven vooraf beloven te doen als een test een gevaarlijke vaardigheid laat zien.

    **Een voorbeeld uit het dagelijks leven.** Een slotenmaker publiceert een boek over hoe sloten werken. Inbrekers kunnen het lezen, maar het meeste was onder inbrekers al bekend; slotenmakers en kopers leren eruit welke sloten ze moeten mijden. Of het boek een goed idee was, hangt af van wat het toevoegde, niet van wat erin staat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_duplicate, choice_recall, result_marginal, result_time):
    L.wrap_up(result_marginal, result_time, choice_recall, choice_duplicate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
