# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-security-basics. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-security-basics", "nl")
    C = L.checks   # dreigingen en een aanvalsboom voor het ledenbestand van een verzonnen bibliotheek


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl) en [Dictionaries](/course/open/py-dicts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_recursion():
    quick_recursion = L.choice(
        "quick-recursion",
        question="Wat laat dit programma zien?\n\n```python\ndef f(n):\n    if n == 0:\n        return 1\n    return 2 * f(n - 1)\n\nprint(f(3))\n```",
        options={
            "six": "6",
            "right": "8",
            "never": "Niets: het stopt nooit",
        },
        correct="right",
        why={
            "right": "`f(3)` geeft `2 * f(2)`, `f(2)` geeft `2 * f(1)`, `f(1)` geeft `2 * f(0)`, en het basisgeval `f(0)` geeft 1. Op de weg terug naar boven: 2, 4, 8.",
            "six": "Elke aanroep vermenigvuldigt met 2, drie keer, en het basisgeval geeft 1: 2 × 2 × 2 × 1 = 8. Zie *Oefening 2. Volg de aanroepen* in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).",
            "never": "Elke aanroep krijgt een getal dat één kleiner is, en `f(0)` is het basisgeval: daar stoppen de aanroepen. Zie *Recept* in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).",
        },
    )
    quick_recursion
    return (quick_recursion,)


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="Wat laat dit programma zien?\n\n```python\nlevel = {\"desk\": 1, \"it\": 3}\nlevel[\"data\"] = 2\nprint(len(level), level[\"it\"])\n```",
        options={
            "two": "2 3",
            "error": "Een fout: \"data\" is geen key",
            "right": "3 3",
        },
        correct="right",
        why={
            "right": "De tweede regel voegt een derde paar toe, dus `len(level)` is 3, en de key `\"it\"` heeft nog steeds de waarde 3.",
            "two": "`level[\"data\"] = 2` voegt een nieuw paar toe als de key er nog niet in staat: de dictionary heeft nu drie keys. Zie *Nieuwe woorden en tekens* in [Dictionaries](/course/open/py-dicts?lang=nl).",
            "error": "Een waarde geven aan een key die er nog niet is, voegt die key toe; alleen een ontbrekende key opzoeken geeft een fout. Zie *Nieuwe woorden en tekens* in [Dictionaries](/course/open/py-dicts?lang=nl).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een verzonnen bibliotheek, die uit de module privacy, houdt een ledenbestand bij: de namen, e-mailadressen en telefoonnummers van de leden. Er minder van bewaren, en wissen wat niet meer nodig is, gaat over wat de bibliotheek zelf met de gegevens doet. Security stelt een andere vraag: wat zou iemand doen om met opzet bij de gegevens te komen? Niet "is het ledenbestand veilig?", maar "wat is de goedkoopste manier om het te lezen, en voor wie?". Denken als een aanvaller betekent opschrijven wat er mis kan gaan voordat iemand anders het vindt, en dan zoeken naar de weg naar binnen die het minst kost.

    **Een voorbeeld uit het dagelijks leven.** Een inbreker kraakt niet het dure slot op de voordeur als het raam aan de achterkant openstaat. Een sterkere voordeur verandert dan niets; het raam dichtdoen wel.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="In 1999 tekende Bruce Schneier de manieren om een kluis open te krijgen als een boom, met wat elke manier een aanvaller zou kosten. Welke manier vond die boom het goedkoopst?",
        options={
            "lock": "Het slot openpeuteren",
            "combo": "De combinatie te weten komen",
            "right": "De kluis opensnijden",
        },
        reveal="**De kluis opensnijden**, voor 10.000 dollar in de figuur van Schneier. De combinatie te weten komen kost op zijn goedkoopst 20.000 dollar, door de eigenaar van de kluis om te kopen, en het slot openpeuteren 30.000 dollar. *Aanvalsbomen* in *Het idee* laat zien hoe de boom dat uitrekent.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Drie dingen om te beschermen

    Security beschermt vooral drie dingen. FIPS 199, een standaard van NIST, het Amerikaanse instituut voor standaarden, zegt dat de Amerikaanse wet ze vastlegt als drie doelen van beveiliging, en beschrijft wat het betekent als je er een verliest:

    - **vertrouwelijkheid** (*confidentiality*): informatie komt terecht bij iemand die er geen recht op heeft. Iemand leest wat die niet mag lezen.
    - **integriteit** (*integrity*): informatie wordt zonder recht veranderd of vernietigd. Iemand verandert of wist wat die niet mag veranderen.
    - **beschikbaarheid** (*availability*): de toegang tot de informatie of het systeem, of het gebruik ervan, wordt verstoord. Wie het systeem mag gebruiken, kan dat niet.

    Samen heten ze vaak de **CIA-triade**, naar hun Engelse beginletters. Nieuw waren ze in 2004 niet: Saltzer en Schroeder schreven in 1975 al dat specialisten in security het handig vinden om mogelijke schendingen van beveiliging in drie groepen in te delen: informatie zonder recht vrijgeven, informatie zonder recht veranderen, en een bevoegde gebruiker het gebruik ervan onmogelijk maken. Ze schreven er een waarschuwing bij die nog steeds geldt: de grootste complicatie in een computersysteem voor algemeen gebruik dat op afstand te bereiken is, is dat de indringer ook een gewone, bevoegde gebruiker kan zijn. De aanvaller kan een medewerker zijn.

    ### Wat er mis kan gaan: STRIDE

    In 1999 schreven Loren Kohnfelder en Praerit Garg binnen Microsoft een notitie met zes soorten dreigingen waar een productteam tijdens het ontwerp naar moet zoeken. De Engelse beginletters vormen **STRIDE**: *spoofing* (je voordoen als iemand anders), *tampering* (gegevens veranderen), *repudiation* (ontkennen dat je iets deed; hun notitie schrijft *Repudiability*), *information disclosure* (informatie die uitlekt), *denial of service* (een dienst platleggen) en *elevation of privilege* (rechten krijgen die je niet hoort te hebben). Een publicatie in MSDN Magazine uit 2006, van Shawn Hernan, Scott Lambert, Tomasz Ostwald en Adam Shostack, koppelt elke soort dreiging aan de eigenschap die ertegen beschermt: elke soort gaat in tegen één eigenschap. Drie daarvan zijn de CIA-triade; de andere drie zijn authenticatie (je bent wie je zegt te zijn), onweerlegbaarheid (een handeling hangt vast aan wie die deed) en autorisatie (je doet alleen wat je mag). Dezelfde publicatie waarschuwt dat één dreiging onder twee letters kan passen, en raadt aan je niet blind te staren op de namen. STRIDE helpt je om geen soort dreiging te vergeten; het hoeft niet elke dreiging perfect in te delen.

    ### Een dreigingsmodel

    Een **dreigingsmodel** beschrijft een systeem zoals een aanvaller het ziet, zodat je het kunt verdedigen: wat is het waard om aan te vallen, wie zou aanvallen, en hoe. Het *Threat Modeling Manifesto*, in 2020 geschreven door een werkgroep van vijftien mensen, vat het samen in vier vragen: waar werken we aan? Wat kan er misgaan? Wat gaan we daaraan doen? Hebben we het goed genoeg gedaan? STRIDE helpt bij de tweede vraag, en aanvalsbomen ook.

    ### Aanvalsbomen

    Bruce Schneier beschreef **aanvalsbomen** (*attack trees*) in 1999: je zet de aanvallen op een systeem in een boom, met het doel van de aanvaller als wortel en de verschillende manieren om dat doel te bereiken als bladeren. Elke node boven de bladeren is een van twee soorten. Een OR-node geeft alternatieven, zoals de vier manieren om een kluis open te krijgen; een AND-node geeft stappen die allemaal nodig zijn voor hetzelfde doel. Geef elk blad een waarde, en de boom rekent de waarde uit van elke node erboven. Voor kosten is de regel: een OR-node kost zoveel als zijn goedkoopste kind, een AND-node de som van zijn kinderen. De waarde bij de wortel is dan wat de goedkoopste aanval kost. Hetzelfde werkt voor mogelijk of onmogelijk: een OR-node is mogelijk als een van zijn kinderen dat is, een AND-node alleen als ze dat allemaal zijn.

    ### De goedkoopste weg naar binnen

    Een aanvaller heeft niet de moeilijkste weg naar binnen nodig, alleen de goedkoopste die werkt. Het voorbeeld van Schneier is PGP, een programma voor versleutelde e-mail: de versleuteling ervan kraken is niet de aanval die het meest oplevert. Kohnfelder en Garg schreven in 1999 hetzelfde: een systeem als geheel is meestal niet veiliger dan zijn zwakste schakel. En een boom is nooit af. Schneier: er is altijd een kans dat je een aanval vergeten bent, maar je wordt er met de tijd beter in.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **CIA-triade** | vertrouwelijkheid, integriteit en beschikbaarheid: de drie dingen die security beschermt |
        | **beschikbaarheid** (*availability*) | wie een systeem of de gegevens erin mag gebruiken, kan dat, op het moment dat het nodig is |
        | **STRIDE** | zes soorten dreigingen om naar te zoeken: *spoofing*, *tampering*, *repudiation*, *information disclosure*, *denial of service* en *elevation of privilege* |
        | **dreigingsmodel** | een beschrijving van wat er in een systeem het aanvallen waard is, wie zou aanvallen en hoe, zodat je het kunt verdedigen |
        | **aanvalsboom** (*attack tree*) | een boom met het doel van een aanvaller als wortel en de manieren om er te komen daaronder; OR-nodes zijn alternatieven, AND-nodes stappen die allemaal nodig zijn |
        """),
        L.watch_out("STRIDE in hoofdletters is de lijst met dreigingen; de stride van een convolutie, uit [Convolutie in detail](/course/open/cnn-conv-formal?lang=nl), heeft er niets mee te maken. En in een aanvalsboom is de goedkoopste stap niet altijd de goedkoopste aanval: onder een AND-node moeten de andere stappen ook betaald worden."),
        mo.Html(C.tree_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo denk je als een aanvaller over een systeem:

    1. Zeg wat je beschermt, en om welke van de drie het gaat: vertrouwelijkheid, integriteit, beschikbaarheid.
    2. Loop STRIDE langs, en schrijf op wat er van elke soort mis kan gaan. Dat is `at_risk`.
    3. Teken een aanvalsboom voor één doel van de aanvaller: OR voor alternatieven, AND voor stappen die allemaal nodig zijn.
    4. Reken vanaf de bladeren naar boven uit wat de goedkoopste aanval kost, en welke stappen die zet. Dat is `cost` en `cheapest`.
    5. Maak die goedkoopste aanval duurder, en kijk opnieuw: de aanvaller gaat naar de volgende goedkoopste weg.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.TREE` is de aanvalsboom voor het lezen van het ledenbestand van de bibliotheek, zoals in het plaatje. Elke node is een dict met een `"name"` en een `"kind"`: `"leaf"` (blad), `"and"` of `"or"`. Een `"and"`- of `"or"`-node heeft een lijst `"children"`; een blad heeft een `"cost"` in euro's en `"possible"`, dat zegt of een aanvaller van buiten, met weinig geld, die stap kan zetten. De cel past de regel van Schneier voor mogelijk en onmogelijk toe met een functie die zichzelf aanroept: een blad geeft direct antwoord, en een node telt hoeveel van zijn kinderen mogelijk zijn.
    """)
    return


@app.cell
def example_possible():
    # mogelijk of niet, met de regel van Schneier: een OR-node heeft één mogelijk kind nodig, AND ze allemaal
    def _possible(node):
        if node["kind"] == "leaf":
            return node["possible"]
        _count = 0
        for _child in node["children"]:
            if _possible(_child):
                _count = _count + 1
        if node["kind"] == "and":
            return _count == len(node["children"])
        return _count > 0

    for _way in C.TREE["children"]:
        print(_way["name"] + ":", _possible(_way))
    print("the goal:", _possible(C.TREE))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Inloggen als medewerker (*Log in as a staff member*) is mogelijk: het is een AND-node, en zijn beide kinderen, een wachtwoord krijgen en de inlogcode krijgen, hebben een mogelijke stap. De back-up stelen (*Steal the backup*) niet: in de IT-ruimte komen is makkelijk, maar de versleuteling kraken kan deze aanvaller niet, en een AND-node heeft ze allebei nodig. Omkopen kan niet zonder geld. Met twee mogelijke wegen over is het doel ook mogelijk. De regel van Schneier zegt of er een weg naar binnen is; de oefeningen vragen wat de goedkoopste weg kost.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_risk():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. at_risk: welke eigenschappen gevaar lopen

        `THREATS` is een lijst met acht dreigingen tegen de bibliotheek, elk een dict met een `"what"` en onder `"stride"` de soort STRIDE-dreiging, zoals `"tampering"`. `PROPERTY` is een dictionary die voor elke soort de eigenschap geeft die erdoor breekt: `PROPERTY["tampering"]` is `"integrity"`. `at_risk(threats)` geeft een dictionary terug met voor elke eigenschap hoeveel van de dreigingen die breken. Binnenin begint `counts` leeg, en de loop gaat langs de dreigingen. Vervang de `...`. De check roept `at_risk(THREATS)` aan, en daarna kortere lijsten.
        """),
        L.task("Tel voor elke eigenschap hoeveel dreigingen die breken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_risk():
    box_risk = L.code_box("at-risk", lines=10)
    box_risk
    return (box_risk,)


@app.cell(hide_code=True)
def check_risk(box_risk):
    result_risk = L.check("at-risk", box_risk.value)
    result_risk
    return (result_risk,)


@app.cell(hide_code=True)
def hints_risk():
    L.hints(
        "at-risk",
        [
            "Stap 2 van het recept. In [Dictionaries](/course/open/py-dicts?lang=nl) telde je met een dictionary in een loop: tel er 1 bij op als de key er al in staat, en begin bij 1 als die er nog niet in staat.",
            "Zoek voor elke dreiging eerst in `PROPERTY` op welke eigenschap de soort breekt, en bewaar die onder een naam zoals `prop`. Tel daarna `prop` in `counts`.",
            '`prop = PROPERTY[___]`, dan `if prop in counts:` met `counts[prop] = counts[prop] + 1` eronder, en `else:` met `counts[prop] = 1`',
        ],
        why='`prop = PROPERTY[threat["stride"]]` geeft de eigenschap; `if prop in counts:` telt er 1 bij op, en `else:` begint bij 1. `at_risk(THREATS)` geeft `{"authentication": 1, "integrity": 2, "non-repudiation": 1, "confidentiality": 2, "availability": 1, "authorization": 1}`.',
    )
    return


@app.cell(hide_code=True)
def text_after_risk():
    mo.md(r"""
    Vijf van de acht dreigingen breken een van de drie eigenschappen van de CIA-triade: twee de vertrouwelijkheid, twee de integriteit, één de beschikbaarheid. Voor de andere drie heb je nodig wat STRIDE toevoegt: een gestolen wachtwoord breekt de authenticatie, een kwijtgescholden boete waar niemand verantwoordelijk voor wil zijn breekt de onweerlegbaarheid, en een balieaccount dat de bestanden van het datateam kan openen breekt de autorisatie. Zo'n lijst is het begin van een dreigingsmodel. Hij zegt niet welke dreiging het zwaarst weegt; daarvoor moet je ook weten hoe erg elke dreiging zou zijn, en wat elke aanval zou kosten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. cost: wat de goedkoopste aanval kost

        `cost(node)` geeft terug wat de goedkoopste aanval onder `node` kost: voor een blad zijn `"cost"`, voor een AND-node de som van wat zijn kinderen kosten, en voor een OR-node wat het goedkoopste kind kost. Een kind kan een node zijn met eigen kinderen, dus `cost` roept zichzelf aan op elk kind. De eerste regels staan er al: een blad geeft zijn eigen kosten terug. Vervang de `...`. De check roept `cost(TREE)` aan, en daarna delen van de boom zoals `LOGIN` (inloggen als medewerker), de boom van Schneier `SAFE` (in duizenden dollars) en eigen kleine bomen.
        """),
        L.task("Geef voor een AND-node de som van de kinderen terug, en voor een OR-node de kosten van het goedkoopste kind."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cost():
    box_cost = L.code_box("cost", lines=13)
    box_cost
    return (box_cost,)


@app.cell(hide_code=True)
def check_cost(box_cost):
    result_cost = L.check("cost", box_cost.value)
    result_cost
    return (result_cost,)


@app.cell(hide_code=True)
def hints_cost():
    L.hints(
        "cost",
        [
            "Stap 4 van het recept, met de vorm van het uitgewerkte voorbeeld: ga één keer langs de kinderen, en beslis daarna op grond van de soort node.",
            'Begin voor de loop een `total` op 0 en een `best` op de kosten van het eerste kind, `cost(node["children"][0])`. Tel in de loop de kosten van elk kind op bij `total`, en bewaar ze in `best` als ze lager zijn. Na de loop geeft een AND-node de een van de twee terug, een OR-node de ander.',
            '`if node["kind"] == "and":` met `return ___` eronder, en daarna `return best`',
        ],
        why='In de loop telt `total = total + cost(child)` de kinderen op, en houdt `if cost(child) < best:` met `best = cost(child)` het goedkoopste vast. Daarna geeft `if node["kind"] == "and":` `total` terug, en anders `return best`. `cost(TREE)` geeft `500`.',
    )
    return


@app.cell(hide_code=True)
def text_after_cost():
    mo.md(r"""
    Het ledenbestand lezen kost op zijn goedkoopst €500: inloggen, en daarvoor zijn een wachtwoord (op zijn goedkoopst €100) en de inlogcode (€400) nodig. In de IT-ruimte komen kost maar €50, de goedkoopste stap in de boom, maar voor de back-up stelen moet ook de versleuteling gekraakt worden: €20.050. De SQL injection in de zoekfunctie van de catalogus, het onderwerp van de volgende les, kost €600, vlak daarachter. En de kluis van Schneier komt uit op 10, de 10.000 dollar van het opensnijden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cheapest():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. cheapest: de stappen van de goedkoopste aanval

        `cheapest(node)` geeft de namen van de stappen van de goedkoopste aanval onder `node` terug, als lijst. Een blad geeft zijn eigen naam in een lijst terug, en een AND-node de stappen van al zijn kinderen: die regels staan er al. Volg bij een OR-node het kind dat het minst kost, en geef de stappen daaronder terug; bij twee even goedkope kinderen het eerste daarvan. `cost` uit Oefening 2 staat klaar, de versie van de course. Vervang de `...`. De check roept `cheapest(TREE)` aan, en daarna delen van de boom, `SAFE` en kleine bomen.
        """),
        L.task("Geef bij een OR-node de stappen van de goedkoopste aanval onder het goedkoopste kind terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cheapest():
    box_cheapest = L.code_box("cheapest", lines=15)
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
            "Stap 4 van het recept. Een OR-node is een keuze: de aanvaller neemt één kind, het goedkoopste. De stappen zijn de stappen van de goedkoopste aanval onder dat kind.",
            'Begin met `best` als het eerste kind, `node["children"][0]`. Ga langs de kinderen, en bewaar een kind in `best` als het minder kost dan `best`. Laat `cheapest` daarna de stappen onder `best` zoeken.',
            '`if ___ < cost(best):` met `best = child` eronder, en na de loop `return cheapest(best)`',
        ],
        why='`best = node["children"][0]`, dan in de loop `if cost(child) < cost(best):` met `best = child`, en na de loop `return cheapest(best)`. `cheapest(TREE)` geeft `["Guess the password", "Phish the code too"]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_cheapest():
    mo.md(r"""
    De goedkoopste aanval zet twee stappen: het wachtwoord raden, en daarna ook de inlogcode phishen. Samen kosten ze €500, evenveel als `cost(TREE)`. De boom zegt nu waar een verdediging het meest helpt: op deze weg. Sterkere versleuteling van de back-up zou niets veranderen, want niemand zou die kant op gaan. Voor de kluis van Schneier geeft `cheapest(SAFE)` `["Cut Open Safe"]`.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Verdedigingen

    Kies verdedigingen voor de bibliotheek. Elke verdediging maakt stappen duurder, of haalt een stap weg. Het plaatje laat zien wat elke node nu kost, met de stappen van de goedkoopste aanval gevuld: met jouw `cost` en `cheapest` zodra Oefening 2 en 3 goed zijn, en tot dan met die van de course. Voeg ze een voor een toe, in de volgorde van de lijst, en kijk waar de aanvaller heen gaat.
    """)
    return


@app.cell(hide_code=True)
def defenses_cell():
    defenses = mo.ui.multiselect(options=C.DEFENSE_LABELS[L.lang], label="Verdedigingen")
    defenses
    return (defenses,)


@app.cell(hide_code=True)
def demo_defenses(box_cheapest, box_cost, defenses, result_cheapest, result_cost):
    _my_cost = C.mine(box_cost.value, result_cost.state == "correct", "cost")
    _my_cheapest = C.mine(box_cheapest.value, result_cheapest.state == "correct", "cheapest", {"cost": C.cost})
    _out = mo.Html(C.defense_picture(defenses.value, _my_cost, _my_cheapest, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met lange wachtwoorden gaat de aanvaller kijken hoe een wachtwoord bij de balie wordt ingetikt: €550. Een training tegen phishing maakt daarna de SQL injection de goedkoopste weg naar binnen, voor €600. Placeholders in de zoekfunctie halen die stap weg, en dan komt inloggen terug, nu met een gestolen telefoon: €1150. Met de schermen bij de balie ook nog weggedraaid is de goedkoopste aanval een medewerker omkopen, voor €2000. Elke verdediging hielp, en elke keer nam de aanvaller gewoon de volgende weg naar binnen. Daarom kijk je naar de hele boom, en niet alleen naar de stap waar je je het eerst zorgen over maakte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_integrity():
    mo.md(r"""
    ### Oefening 4. Welke eigenschap?

    Iemand verandert de openingstijden op de website van de bibliotheek, zodat er staat dat de bibliotheek op zaterdag dicht is. De website zelf werkt zoals altijd.
    """)
    return


@app.cell(hide_code=True)
def choice_integrity_cell():
    choice_integrity = L.choice(
        "integrity",
        question="Welke van de drie breekt dat?",
        options={
            "avail": "Beschikbaarheid: op zaterdag komt er nu niemand meer",
            "right": "Integriteit: de tijden zijn onbevoegd veranderd",
            "conf": "Vertrouwelijkheid: iedereen ziet nu de veranderde pagina",
        },
        correct="right",
        why={
            "right": "FIPS 199: integriteit verlies je als informatie zonder recht veranderd of vernietigd wordt. Iemand veranderde de openingstijden zonder dat te mogen; in STRIDE is dat *tampering*.",
            "avail": "De website werkt nog, en iedereen kan de pagina nog lezen: er is niets afgesneden. Wat misging, is dat de informatie veranderd is. Zie *Drie dingen om te beschermen* in *Het idee*.",
            "conf": "Er is niets geheims gelezen: de openingstijden zijn openbaar. De informatie is veranderd. Zie *Drie dingen om te beschermen* in *Het idee*.",
        },
    )
    choice_integrity
    return (choice_integrity,)


@app.cell(hide_code=True)
def text_exercise_and():
    mo.md(r"""
    ### Oefening 5. De goedkoopste stap

    In de boom van de bibliotheek kost in de IT-ruimte komen €50: de goedkoopste stap van allemaal.
    """)
    return


@app.cell(hide_code=True)
def choice_and_cell():
    choice_and = L.choice(
        "and-node",
        question="Waarom is de back-up stelen toch niet de goedkoopste aanval?",
        options={
            "right": "Een AND-node: ook de versleuteling moet gekraakt worden",
            "outside": "De boom telt alleen wat iemand van buiten kan",
            "or": "Volgens Schneier kost een OR-node zoveel als het duurste kind",
        },
        correct="right",
        why={
            "right": "Een AND-node kost de som van zijn kinderen: in de IT-ruimte komen (€50) plus de versleuteling kraken (€20.000) is €20.050.",
            "outside": "`cost` telt kosten op; of een aanvaller van buiten een stap kan zetten, is de vraag van het uitgewerkte voorbeeld, niet van deze. Zie *Aanvalsbomen* in *Het idee*.",
            "or": "Een OR-node kost zoveel als zijn goedkoopste kind. En de back-up stelen is een AND-node. Zie *Aanvalsbomen* in *Het idee*.",
        },
    )
    choice_and
    return (choice_and,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl) riep een functie zichzelf aan op een kleiner getal of een kortere tekst; hier riep hij zichzelf aan op elk kind van een boom, en hetzelfde soort basisgeval, een blad, hield hem tegen. Met [Dictionaries](/course/open/py-dicts?lang=nl) telde je dreigingen per eigenschap. De module privacy vroeg wat een organisatie met persoonsgegevens doet; vanaf hier vraagt de module security wat een aanvaller ermee zou kunnen doen. *Injection en geheimen* neemt de stap van €600 uit deze boom, de SQL injection, en laat zien hoe die werkt en hoe je hem dichtzet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_and, choice_integrity, result_cheapest, result_cost, result_risk):
    L.wrap_up(result_risk, result_cost, result_cheapest, choice_integrity, choice_and)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
