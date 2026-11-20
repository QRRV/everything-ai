# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-safe-design-principles. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-safe-design-principles", "nl")
    C = L.checks   # controleurs die over facturen stemmen, betalingen met een tweede persoon, en modellen die stemmen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_failures():
    quick_failures = L.choice(
        "quick-failures",
        question="Een dienst heeft drie negens van betrouwbaarheid en krijgt 100.000 verzoeken per dag. Hoeveel verzoeken mislukken er ongeveer per dag?",
        options={
            "right": "100: 0,1% van 100.000",
            "three": "3: één mislukt verzoek per negen",
            "percent": "1000: 1% van 100.000",
        },
        correct="right",
        why={
            "right": "Drie negens is 99,9%, dus de kans op falen is 0,001: 100.000 × 0,001 = 100 mislukte verzoeken per dag.",
            "three": "De negens tellen factoren 10 in de kans op falen, geen mislukte verzoeken: drie negens is een kans op falen van 0,001, en 100.000 × 0,001 = 100. Zie [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl).",
            "percent": "1% is de kans op falen bij twee negens, 99%. Drie negens laten 0,1% over: 100 van de 100.000. Zie [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl).",
        },
    )
    quick_failures
    return (quick_failures,)


@app.cell(hide_code=True)
def quick_check_steps():
    quick_steps = L.choice(
        "quick-steps",
        question="Een AI-agent doet een taak van 3 stappen. Elke stap slaagt met kans 0,9, onafhankelijk van de andere. Hoe groot is de kans dat de hele taak slaagt?",
        options={
            "right": "0,729: 0,9 × 0,9 × 0,9",
            "one": "0,9: elke stap heeft dezelfde kans",
            "sum": "0,7: één min de drie kansen op falen van 0,1",
        },
        correct="right",
        why={
            "right": "Alle drie de stappen moeten slagen, en bij onafhankelijke stappen worden de kansen vermenigvuldigd: 0,9 × 0,9 × 0,9 = 0,729.",
            "one": "Elke stap heeft kans 0,9, maar alle drie moeten slagen: de kansen worden vermenigvuldigd, 0,9 × 0,9 × 0,9 = 0,729. Zie [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl).",
            "sum": "De kansen op falen aftrekken klopt niet precies: een taak waarin twee stappen falen, tel je zo twee keer. De kansen op slagen worden vermenigvuldigd: 0,9 × 0,9 × 0,9 = 0,729. Zie [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl).",
        },
    )
    quick_steps
    return (quick_steps,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een verzonnen bedrijf laat een AI-agent zijn facturen afhandelen. De agent leest elke factuur, beslist of die echt is, en betaalt de factuur van de bankrekening van het bedrijf. Hij zit in 99,9% van de gevallen goed: drie negens. Maar één verkeerde beslissing kan €40.000 naar een oplichter sturen, en in dit ontwerp houdt niets dat tegen: één model beslist, zijn account kan elk bedrag betalen, en niemand anders kijkt mee.

    Ingenieurs liepen hier al lang voor AI tegenaan, bij liften, treinen, ruimtevaartuigen en banken. Hun antwoord was niet alleen elk onderdeel betrouwbaarder maken, maar het systeem zo ontwerpen dat één onderdeel dat faalt niet genoeg is voor schade. Deze les neemt hun principes en past ze toe op AI-systemen: redundantie, fail-safes, functiescheiding, least privilege en negatieve terugkoppeling.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De Space Shuttle vloog met vijf boordcomputers. Hoeveel daarvan draaiden dezelfde software?",
        options={
            "all": "Alle vijf",
            "right": "Vier van de vijf",
            "none": "Geen enkele: elk had eigen software",
        },
        reveal="**Vier van de vijf.** Tijdens de lancering en de landing vormden vier computers met dezelfde software de redundante set; over hun opdrachten werd gestemd, zodat een computer die faalde werd overstemd. De vijfde was een reserve met software die apart was ontwikkeld, zodat één fout in de gedeelde software niet alle vijf kon stilleggen. Redundantie werkt het best als de onderdelen niet om dezelfde reden falen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Principes voor het hele systeem

    Het leerboek *Introduction to AI Safety, Ethics, and Society* (2024) van Dan Hendrycks, dat in [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl) betrouwbaarheid in negens telde, geeft ook principes om veiligere systemen te ontwerpen. Wie ze bij het ontwerpen volgt, maakt de kans op falen kleiner en de gevolgen ervan minder ernstig, schrijft het leerboek. Ze horen bij het begin: je bouwt ze er vanaf het eerste ontwerp in, niet achteraf. En ze gaan over meer dan het model: ze helpen bij het bouwen van de AI zelf, maar ook bij het systeem eromheen. Het leerboek noemt er acht, waaronder de vijf van deze les. Veel ervan zijn ouder dan AI. In 1975 zetten Jerome H. Saltzer en Michael D. Schroeder acht principes op een rij om informatie in computersystemen te beschermen; least privilege staat in beide lijsten.

    ### Redundantie

    Redundantie betekent volgens het leerboek dat er voor elke kritieke functie meerdere reserveonderdelen zijn, zodat er geen single point of failure meer is: geen onderdeel waarvan het falen alleen al genoeg is voor een ongeluk. Een auto met een voetrem en een handrem kan nog afremmen als een van beide het niet doet. Dat is **redundantie**.

    Een goed gedocumenteerd voorbeeld is de Space Shuttle, beschreven in *Computers in Spaceflight* (1988), een geschiedenis door James E. Tomayko, uitgegeven door NASA. Tijdens de lancering en de landing vormden vier computers met dezelfde software de redundante set. Over hun opdrachten werd gestemd bij de actuatoren, de onderdelen die de opdrachten uitvoeren, zodat de opdrachten van een computer die faalde fysiek werden overstemd. Voor een stemming zijn er minstens drie nodig: met twee computers die het oneens zijn, weet je wel dat er een fout zit, maar niet welke. Meer computers maken een ramp zeldzamer: met drie computers verwachtte men drie verloren vliegtuigen per miljoen vluchten, met vier nog maar vier per miljard. Dat waren voorspellingen voor vliegtuigen, en de vierde computer maakt een ramp daarin 750 keer zo zeldzaam. Bij de eerste van de naderings- en landingstests van de Shuttle viel een computer uit op het moment dat de Enterprise werd losgelaten van de Boeing 747 waarop het toestel meevloog, en toch verliep de landing goed.

    Maar vier computers met dezelfde software kunnen op hetzelfde moment dezelfde fout maken. De vijfde computer was een reserve, en NASA liet de software daarvan apart ontwikkelen, los van die van de andere vier, om te voorkomen dat één softwarefout alle computers tegelijk zou treffen. Redundantie helpt als de onderdelen niet om dezelfde reden falen.

    ### Fail-safes

    Fail-safes zijn volgens het leerboek voorzieningen die een systeem veilig moeten houden, ook als het faalt. Het klassieke voorbeeld uit het leerboek zijn de remmen van een lift. Ze zitten aan de buitenkant van de liftkooi en worden alleen opengehouden door de spanning in de kabels waaraan de kooi hangt. Valt die spanning weg, dan klemmen de remmen zich vanzelf vast op de rails in de liftschacht. Bij de lift brengt het falen zelf de veilige toestand, zonder sensor of beslissing ertussen. Dat is een **fail-safe**.

    De Amerikaanse regels voor treinen kennen er ook een. Een alerter in de cabine let erop of de machinist de bediening gebruikt. Verandert er te lang niets en reageert niemand op het alarm, dan zet het apparaat de remmen in tot de trein stilstaat. Hoe lang het duurt voordat het alarm klinkt, in seconden, is 2400 gedeeld door de rijsnelheid in mijl per uur: 40 seconden bij 60 mijl per uur. Voor vertrek wordt de alerter getest door die tijd te laten verlopen.

    Voor AI noemt het leerboek mogelijke fail-safes, zoals een kill switch waarmee alle exemplaren van een AI-systeem uit te zetten zijn als dat nodig is, bijvoorbeeld bij een storing. Saltzer en Schroeder hadden een verwante regel voor toegang, fail-safe defaults: baseer toegang op toestemming, niet op uitsluiting. Zonder uitdrukkelijke toestemming is het antwoord nee, en een fout leidt dan meestal tot een weigering: een veilige toestand, die ook snel opvalt.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Functiescheiding

    Functiescheiding betekent volgens het leerboek dat niemand het hele systeem alleen kan besturen of misbruiken. Het voorbeeld uit het leerboek is een lab met twee chemicaliën die kunnen ontploffen als je grote hoeveelheden mengt: elk staat in een eigen kast, en elk wordt door een andere persoon uitgegeven, in kleine hoeveelheden. Zo heeft niemand toegang tot veel van allebei. Voor AI stelt het leerboek voor om meerdere smalle AI-modellen te gebruiken in plaats van één algemeen model. Dat is **functiescheiding**.

    NIST, het Amerikaanse instituut voor standaarden, geeft een alledaags voorbeeld: wie een salarisbetaling goedkeurt, mag die niet ook zelf klaarzetten. Een vorm ervan is het vier-ogenprincipe: twee mensen moeten samen handelen. Saltzer en Schroeder noemden het separation of privilege: een slot met twee sleutels, zodat één ongeluk, één misleiding of één geschonden vertrouwen niet genoeg is om de beschermde informatie in gevaar te brengen. Het wordt gebruikt bij huurkluizen in een bank, en bij het systeem dat een kernwapen alleen afvuurt als twee verschillende mensen allebei de juiste opdracht geven. In 2025 beschreef Anthropic zo'n regel voor de weights van zijn AI-modellen: een medewerker die toegang nodig heeft, moet op het moment van de aanvraag toestemming krijgen van een tweede persoon.

    ### Least privilege

    Elk programma en elke gebruiker van het systeem moet werken met zo weinig rechten als nodig is voor de taak, schreven Saltzer en Schroeder. Dit principe beperkt vooral de schade die een ongeluk of fout kan aanrichten. Het leerboek voegt eraan toe dat functiescheiding misschien alleen werkt als niemand toegang heeft tot delen van het systeem die niets met de eigen taak te maken hebben. Voor AI: misschien wil je vermijden dat AI's op het internet worden aangesloten of beheerdersrechten krijgen over vertrouwelijke informatie.

    OWASP, een stichting voor softwarebeveiliging, maakt een top 10 van risico's van toepassingen die op large language models zijn gebouwd. De lijst van 2025 noemt AI-agents die te veel kunnen, waardoor een onverwachte, dubbelzinnige of gemanipuleerde output van een LLM schadelijke handelingen kan veroorzaken, wat de oorzaak van die output ook is. Het voorbeeld van OWASP: een extensie die alleen gegevens hoeft te lezen, maakt verbinding met een database met een account dat ook gegevens kan wijzigen, toevoegen en verwijderen. OWASP vraagt om zo weinig mogelijk rechten, om een mens die handelingen met grote gevolgen goedkeurt, en om controles buiten het model: laat de systemen verderop beslissen of iets mag, niet het LLM. Volgens OWASP voorkomen logboeken en een grens aan hoe vaak een agent iets mag doen het risico niet, maar kunnen ze de schade wel beperken.

    ### Negatieve terugkoppeling

    Bij positieve terugkoppeling versterkt een verandering zichzelf: als ijs smelt, weerkaatst de aarde minder zonlicht, warmt die op, en smelt er meer ijs. Negatieve terugkoppeling remt een verandering af en houdt een systeem stabiel: een lichaam dat te warm wordt, gaat zweten, en een lichaam dat het koud krijgt, gaat rillen. Voor AI stelt het leerboek middelen voor die opmerken wanneer een AI te machtig wordt of zich gevaarlijk gedraagt, en dan een maatregel in gang zetten. Dat kan een drastische maatregel zijn, zoals de AI loskoppelen van het internet, terugzetten naar een eerdere versie, of andere AI's inzetten die getraind zijn om een te machtige AI zijn macht af te nemen.

    ### Elk principe heeft een prijs

    Deze principes zijn geen regels die overal gelden. Afhankelijk van het systeem kunnen sommige minder nuttig zijn of zelfs averechts werken, zegt het leerboek, en Saltzer en Schroeder zagen hun principes vooral als waarschuwingen. Veel principes voegen onderdelen, kosten en ingewikkeldheid toe. NASA vond de reservecomputer van de Shuttle zijn gewicht, stroom en ingewikkeldheid waard, als verzekering tegen een fout in de gedeelde software. Maar John R. Garman van het Johnson Space Center van NASA vond dat de reserve het systeem als geheel waarschijnlijk meer kwaad dan goed had gedaan. Volgens Tomayko vond Garman dat de reserve veel druk wegnam bij de ontwikkelaars van de hoofdsoftware.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **redundantie** | meer dan één onderdeel voor een kritieke functie, zodat één onderdeel dat faalt niet genoeg is voor een ongeluk |
        | **fail-safe** | een voorziening die een systeem veilig houdt als het faalt, zoals remmen die dichtklemmen als de kabel slap wordt |
        | **functiescheiding** | een kritieke taak verdelen over meerdere mensen of systemen, zodat niemand het systeem alleen kan misbruiken |
        | least privilege | elke gebruiker en elk programma alleen de toegang geven die de taak nodig heeft |
        | single point of failure | een onderdeel waarvan het falen alleen al genoeg is voor een ongeluk |
        | vier-ogenprincipe | een vorm van functiescheiding waarbij twee mensen samen moeten handelen |
        | meerderheid van stemmen | het antwoord dat meer dan de helft van de stemmen geeft |
        | negatieve terugkoppeling | een mechanisme dat een verandering tegenwerkt, zodat een systeem stabiel blijft |
        | `not in` | de test van Python of iets niet in een lijst staat: `"pay" not in needs` |
        """),
        L.watch_out("Kopieën van één onderdeel kunnen om dezelfde reden falen: redundantie helpt alleen als de onderdelen dat niet doen. En een fail-safe gaat over wat er gebeurt als een systeem faalt, niet over een belofte dat het nooit faalt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zoek de single points of failure: de onderdelen waarvan het falen alleen al schade kan veroorzaken.
    2. Geef elke kritieke functie meer dan één onderdeel, en zorg dat die onderdelen niet om dezelfde reden falen.
    3. Bepaal wat de veilige toestand is, en laat een storing daar vanzelf naartoe leiden.
    4. Verdeel kritieke taken, zodat niemand, mens of AI, alleen schade kan aanrichten.
    5. Geef elk onderdeel alleen de toegang die de taak nodig heeft.
    6. Weeg de prijs: een principe kan onderdelen en kosten toevoegen, en veranderen hoe mensen werken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De factuuragent uit *Waarom* moet facturen lezen en betalingen klaarzetten. De cel vergelijkt dat met wat zijn account mag, en drukt elk recht af dat meer is dan de taak nodig heeft.
    """)
    return


@app.cell
def example_privilege():
    needs = ["facturen lezen", "betalingen klaarzetten"]                          # wat de taak vraagt
    has = ["facturen lezen", "betalingen klaarzetten", "betalen", "verwijderen"]  # wat het account mag
    for right in has:
        if right not in needs:                                                    # meer dan nodig
            print("weghalen:", right)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het account mag twee dingen meer dan de taak nodig heeft: betalen en verwijderen. Met least privilege gaan die allebei weg. Vraagt een verkeerde of gemanipuleerde output de agent dan om te betalen, dan kan zijn account dat niet. In de oefeningen bouw je twee andere principes: een stemming van redundante controleurs die terugvalt op de veilige kant, en een controle die een tweede persoon nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. vote: een meerderheid, of tegenhouden

        Een aantal controleurs, bijvoorbeeld modellen, controleert elke factuur en antwoordt `"pay"` (betalen) of `"hold"` (tegenhouden); een controleur die faalt, geeft geen antwoord: `None`. Schrijf `vote(answers)`. De functie krijgt de lijst `answers` en geeft alleen `"pay"` terug als meer dan de helft van alle antwoorden `"pay"` is, waarbij ook de antwoorden die `None` zijn meetellen. Anders geeft ze `"hold"` terug, het veilige antwoord. Tel de keren pay in een variabele `pays`, met een `for`-loop over `answers` en `answer` voor elk antwoord. Vervang de `...`. De check roept `vote(["pay", "pay", "hold"])` aan, daarna andere lijsten.
        """),
        L.task("Een meerderheid van stemmen die terugvalt op het veilige antwoord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("vote", lines=8)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "vote",
        [
            "Een meerderheid van stemmen, zoals bij de Space Shuttle: meer dan de helft moet het eens zijn. Is dat niet zo, dan valt de functie terug op de veilige kant, zoals een fail-safe.",
            "Begin `pays` bij 0, en tel er in een `for`-loop 1 bij op voor elk antwoord dat `\"pay\"` is. Vergelijk na de loop `pays` met de helft van `len(answers)`: meer dan de helft geeft `\"pay\"`. Elk ander geval eindigt bij `return \"hold\"`.",
            "`    if pays > ___:`",
        ],
        why="`pays = 0`, een loop die 1 optelt voor elke `\"pay\"`, dan `if pays > len(answers) / 2:` met `return \"pay\"`, en daarna `return \"hold\"`. Bij `[\"pay\", None, None]`: één pay van de drie is niet meer dan de helft, dus tegenhouden.",
    )
    return


@app.cell(hide_code=True)
def text_after_vote():
    mo.md(r"""
    Twee controleurs overstemmen een derde die fout zit, en als twee van de drie gefaald hebben, kan de controleur die overblijft niet alleen betalen: de stemming houdt tegen. Dat is redundantie en een fail-safe samen: een controleur die faalt, leidt tot het veilige antwoord, niet tot een betaling.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pay():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. may_pay: niemand keurt de eigen aanvraag goed

        Betalingen tot en met `LIMIT`, €1000, mogen zonder goedkeuring door. Een grotere betaling heeft een goedkeurder nodig die de betaling niet zelf heeft aangevraagd: de agent mag een betaling klaarzetten, maar niet goedkeuren. Schrijf `may_pay(amount, requested_by, approvers)`. De functie geeft `True` terug als de betaling door mag en anders `False`; `approvers` is een lijst met de namen van wie de betaling goedkeurde, en die lijst kan leeg zijn. Loop er met een `for`-loop doorheen, met `name` voor elke goedkeurder. Vervang de `...`. De check roept `may_pay(5000, "agent", ["agent"])` aan, daarna andere betalingen.
        """),
        L.task("Een controle waarbij niemand de eigen betaling goedkeurt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pay():
    box_pay = L.code_box("may_pay", lines=10)
    box_pay
    return (box_pay,)


@app.cell(hide_code=True)
def check_pay(box_pay):
    result_pay = L.check("may_pay", box_pay.value)
    result_pay
    return (result_pay,)


@app.cell(hide_code=True)
def hints_pay():
    L.hints(
        "may_pay",
        [
            "Functiescheiding, zoals in het voorbeeld van NIST met de salarisbetaling: wie iets aanvraagt, keurt het niet goed. Tot en met de grens valt er niets te controleren.",
            "Eerst de grens: geef tot en met `LIMIT` `True` terug. Loop daarboven door `approvers`; geef bij een `name` die niet `requested_by` is `True` terug. Eindigt de loop zonder zo iemand, geef dan na de loop `False` terug.",
            "`        if name ___ requested_by:`",
        ],
        why="`if amount <= LIMIT:` met `return True`; dan `for name in approvers:` met daarin `if name != requested_by:` en `return True`; na de loop `return False`. De goedkeuring van de agent zelf telt niet.",
    )
    return


@app.cell(hide_code=True)
def text_after_pay():
    mo.md(r"""
    Een aanvraag van €5000 die alleen de agent goedkeurde, wordt tegengehouden; een die Sam ook goedkeurde, mag door. In een echt systeem draait zo'n controle buiten het model, in het betaalsysteem, zoals OWASP vraagt: dan kan een verkeerde output van het model die controle niet uitzetten.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: laat modellen stemmen

    Elk model dat een factuur controleert, zit fout met de kans op de schuif. Het plaatje laat zien hoe vaak de beslissing fout is met één model, met drie kopieën van één model die stemmen, en met drie of vijf verschillende modellen die stemmen en hun fouten onafhankelijk van elkaar maken, met de krommen voor alle kansen tot 60%.
    """)
    return


@app.cell(hide_code=True)
def demo_voting():
    voting = L.explore("error", "kans dat één model fout zit, in %", lambda _v: C.demo_frame(_v, L.lang),
                       start=10, values=C.ERRORS, role="",
                       task="Zoek de kleinste kans op de schuif waarbij drie verschillende modellen die stemmen vaker fout zitten dan één model alleen.")
    voting
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 60%. Bij 50% verandert stemmen niets, en daarboven zit de meerderheid vaker fout dan één model: stemmen helpt alleen als elk model vaker goed dan fout zit. Bij 10% zitten drie verschillende modellen fout bij 280 van de 10.000 facturen en vijf bij 86, tegen 1000 voor één model, als hun fouten onafhankelijk zijn. Kopieën van één model blijven op 1000: ze maken dezelfde fouten, zoals computers met dezelfde software. Delen verschillende modellen een deel van hun fouten, dan ligt de winst tussen die lijnen in.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fail_safe():
    mo.md(r"""
    ### Oefening 3. Een robot die faalt

    Een AI-systeem stuurt een robot die zware dozen door een magazijn draagt.
    """)
    return


@app.cell(hide_code=True)
def choice_fail_safe_cell():
    choice_fail_safe = L.choice(
        "fail_safe",
        question="Welke van deze voorzieningen maakt de robot fail-safe?",
        options={
            "backup": "Een tweede besturing die het overneemt als de eerste uitvalt",
            "right": "Remmen die de robot stilzetten zodra hij stroom of verbinding verliest",
            "camera": "Een camera die elke beweging filmt, zodat je elke storing achteraf kunt bestuderen",
        },
        correct="right",
        why={
            "right": "Bij een fail-safe leidt het falen zelf tot een veilige toestand, zoals bij de liftremmen die dichtklemmen als de kabel slap wordt: zonder stroom of verbinding staat de robot stil.",
            "backup": "Dat is redundantie: de robot blijft werken als één besturing uitvalt, maar het zegt niets over wat er gebeurt als ze allebei uitvallen. Zie *Fail-safes* in *Het idee*.",
            "camera": "Een camera helpt om achteraf van een storing te leren; hij maakt de robot niet veilig op het moment dat die faalt. Zie *Fail-safes* in *Het idee*.",
        },
    )
    choice_fail_safe
    return (choice_fail_safe,)


@app.cell(hide_code=True)
def text_exercise_privilege():
    mo.md(r"""
    ### Oefening 4. Een agent voor de agenda

    Een AI-agent plant vergaderingen in de agenda's van een team. Daarvoor kreeg hij een account dat ook ieders e-mail kan lezen en versturen.
    """)
    return


@app.cell(hide_code=True)
def choice_privilege_cell():
    choice_privilege = L.choice(
        "privilege",
        question="Wat vraagt least privilege hier?",
        options={
            "approve": "Een mens die elke e-mail goedkeurt voordat de agent die verstuurt",
            "right": "Een account voor de agent dat bij de agenda's kan en verder nergens bij",
            "log": "Een logboek van elke e-mail die de agent leest of verstuurt, dat iemand wekelijks bekijkt",
        },
        correct="right",
        why={
            "right": "Least privilege geeft de agent alleen de toegang die de taak nodig heeft: de agenda's. Dan kan een verkeerde of gemanipuleerde output helemaal geen e-mail lezen of versturen.",
            "approve": "Een goedkeuring beperkt wat de agent op eigen houtje doet, maar de agent kan nog steeds ieders e-mail lezen. Least privilege neemt de toegang weg die hij niet nodig heeft. Zie *Least privilege* in *Het idee*.",
            "log": "Een logboek helpt om achteraf uit te zoeken wat er gebeurde; OWASP rekent logboeken tot de maatregelen die het risico niet voorkomen, maar de schade wel kunnen beperken. Zie *Least privilege* in *Het idee*.",
        },
    )
    choice_privilege
    return (choice_privilege,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Negens van betrouwbaarheid](/course/open/rai-nines?lang=nl) telde hoe vaak een systeem faalt. Deze les gaf principes die voorkomen dat een storing schade aanricht: redundantie, met onderdelen die niet om dezelfde reden falen; fail-safes, die van een storing een veilige toestand maken; functiescheiding en least privilege, die voorkomen dat één partij, mens of AI, alleen schade aanricht; en negatieve terugkoppeling, die een verandering tegenwerkt voordat die groeit. De volgende les, *Safeguards achter elkaar*, stapelt safeguards en rekent uit hoe groot de kans is dat ze allemaal tegelijk falen.

    **Een voorbeeld uit het dagelijks leven.** In veel bedrijven keurt de collega die iets bestelt de factuur niet zelf goed. En veel grasmaaiers zetten hun mes stil als je de hendel loslaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fail_safe, choice_privilege, result_pay, result_vote):
    L.wrap_up(result_vote, result_pay, choice_fail_safe, choice_privilege)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
