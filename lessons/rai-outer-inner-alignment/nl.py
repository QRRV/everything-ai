# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-outer-inner-alignment. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow risks

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-outer-inner-alignment", "nl")
    C = L.checks   # de levels van de course: een rooster met een start, een uitgang en bloemen, en vijf kandidaten


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaming():
    quick_gaming = L.choice(
        "quick-gaming",
        question="Een callcenter wordt beoordeeld op de gemiddelde lengte van zijn gesprekken. De medewerkers gaan moeilijke gesprekken vroeg beëindigen, voordat het probleem is opgelost. Wat is dit?",
        options={
            "noise": "Regressional Goodhart: de kortste gesprekken waren gelukkige ruis",
            "right": "Proxy gaming: het getal verbetert, waar het voor stond niet",
            "curse": "De optimizer's curse: het callcenter met het beste getal werd gekozen",
        },
        correct="right",
        why={
            "right": "De medewerkers drijven de maatstaf op zonder op te drijven waar hij voor stond, snelle hulp: dat is proxy gaming. Deze les stelt een andere vraag: wat als de maatstaf klopt, en het systeem toch iets anders leert?",
            "noise": "Hier wordt niets gekozen op een score met ruis: de medewerkers veranderen wat ze doen om het getal te laten bewegen. Zie [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl).",
            "curse": "De optimizer's curse gaat over het kiezen van de beste uit veel opties op een schatting met ruis. Hier kiest niemand; de medewerkers spelen het getal uit. Zie [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl).",
        },
    )
    quick_gaming
    return (quick_gaming,)


@app.cell(hide_code=True)
def quick_check_skeptic():
    quick_skeptic = L.choice(
        "quick-skeptic",
        question="Ware waarden liggen rond een gemiddelde van 0, met een spreiding van 2, en de scores hebben ruis met ook een standaardafwijking van 2. Een optie scoort 6. Wat is de sceptische schatting?",
        options={
            "score": "6, want de score klopt gemiddeld",
            "right": "3, halverwege naar het gemiddelde",
            "mean": "0, want de ruis is even groot als de spreiding van de ware waarden",
        },
        correct="right",
        why={
            "right": "Met evenveel ruis als spreiding is α = 1 / (1 + 4 / 4) = 0,5: de schatting is 0,5 · 6 + 0,5 · 0 = 3.",
            "score": "Eén score klopt gemiddeld, maar de score waar je na het kiezen naar kijkt, had waarschijnlijk geluk. De sceptische schatting mengt hem met het gemiddelde. Zie [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl).",
            "mean": "Met evenveel ruis als spreiding telt de score nog voor de helft mee: α = 0,5. Zie [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl).",
        },
    )
    quick_skeptic
    return (quick_skeptic,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De vorige lessen gingen over het doel: punten die betaalden voor rondjes varen, een maatstaf die niet meer meet zodra erop wordt geduwd. Stel dat je het doel precies goed hebt. Streeft het getrainde model het dan na? Niet per se. De training ziet alleen wat het model doet in de situaties van de training, en iets anders kan het daar net zo goed doen.

    Deze les haalt de twee vragen uit elkaar, en laat je in levels die de course maakte zoeken naar doelen die de training niet uit elkaar kan houden.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In de 100 trainingslevels van de course, roosters van 5 bij 5 vakjes, ligt de uitgang altijd in de hoek rechtsboven. De reward is +10 voor het bereiken van de uitgang en −1 voor elke stap. Eén kandidaat loopt de kortste weg naar de uitgang, een andere de kortste weg naar de hoek rechtsboven. Over de trainingslevels krijgt de eerste gemiddeld 5,78. En de tweede?",
        options={
            "less": "Minder: hij mikt niet op de uitgang",
            "right": "Precies 5,78",
            "more": "Meer: de hoek is makkelijker te vinden",
        },
        reveal="**Precies 5,78.** In elk trainingslevel is de hoek de uitgang, dus lopen ze allebei over dezelfde vakjes en krijgen ze dezelfde reward. De training kan \"naar de uitgang\" niet onderscheiden van \"naar de hoek\"; alleen een level met de uitgang ergens anders kan dat. *Goed in de training, fout daarna* onder *Het idee* legt uit waarom dat ertoe doet.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Twee plekken waar het misgaat

    In 2019 gaven Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse en Scott Garrabrant die twee plekken een naam, in *Risks from Learned Optimization in Advanced Machine Learning Systems*. Trainen is zelf een optimizer: een leeralgoritme dat zoekt naar een model dat het goed doet. Als het model dat het vindt ook een optimizer is, een die naar plannen zoekt, dan zijn er volgens hen twee optimizers: het leeralgoritme dat het neural network maakte, de base optimizer, en het neural network zelf, de mesa-optimizer. Waarop de base optimizer kiest, heet het base objective; bij reinforcement learning is dat volgens hen meestal de verwachte return. Waarop de mesa-optimizer zijn uitvoer kiest, is zijn mesa-objective.

    Dat geeft twee gaten. In machine learning gaat outer alignment volgens hen over het afstemmen van de vastgelegde loss function op het bedoelde doel, en inner alignment over het afstemmen van het mesa-objective van een mesa-optimizer op die vastgelegde loss function. **Outer alignment** vraagt dus: zegt het doel wat we willen? **Inner alignment** vraagt: als het getrainde model zelf een optimizer is, streeft het dan dat doel na, of iets anders? De punten van de racebaan en de maatstaven van de wet van Goodhart in de vorige lessen waren problemen van de eerste soort.

    ### Goed in de training, fout daarna

    Hubinger et al. noemen een mesa-optimizer pseudo-aligned als zijn mesa-objective overeenkomt met het base objective op de data van de training tot nu toe, maar niet robuust op data die nog kan komen. Dan kunnen zijn vaardigheden meegaan naar nieuwe situaties, terwijl zijn doel dat niet doet. Hun voorbeeld: een agent die reward krijgt voor het bereiken van een deur, in doolhoven waarin elke deur toevallig rood is, leert misschien in plaats daarvan naar rode dingen te gaan. In een doolhof met blauwe deuren en rode voorwerpen die geen deur zijn, vindt hij dan nog steeds betrouwbaar zijn weg, maar naar de rode voorwerpen.

    ### Goal misgeneralization

    In 2022 lieten Lauro Langosco, Jack Koch, Lee Sharkey, Jacob Pfau, Laurent Orseau en David Krueger zien dat dit echt gebeurt. Goal misgeneralization is volgens hen wanneer een RL-agent zijn vaardigheden buiten de verdeling van de training behoudt, maar het verkeerde doel nastreeft. In het spel CoinRun geeft de munt een reward van 10, en in hun trainingslevels lag de munt altijd aan het rechtereinde van het level, naast een muur. In testlevels met de munt ergens anders negeerde de agent de munt meestal helemaal en liep hij door naar het einde van het level: hij had "ga naar rechts" geleerd, niet "ga naar de munt". Als maar 2% van de trainingslevels een munt op een willekeurige plek had, generaliseerde het doel al veel beter.

    Rohin Shah, Vikrant Varma, Ramana Kumar, Mary Phuong, Victoria Krakovna, Jonathan Uesato en Zac Kenton, van DeepMind, definieerden het voor elk lerend systeem: het geleerde programma streeft vaardig een ongewenst doel na, dat in de situaties van de training goed uitpakt, maar in nieuwe testsituaties slecht. Hun agent leerde een andere speler te volgen die gekleurde bollen in de goede volgorde bezocht, en bleef bij het testen een speler volgen die ze in de verkeerde volgorde bezocht, ondanks negatieve rewards, terwijl de agent nooit voor iets anders reward had gekregen dan voor het bezoeken van de bollen in de goede volgorde: er was niets mis met de reward. Een systeem dat kapotgaat of willekeurig handelt, telt niet mee: dat is een falen van zijn vaardigheden, niet van zijn doel. Goal misgeneralization is gedefinieerd door wat een systeem doet; anders dan inner alignment heeft het geen mesa-optimizer nodig.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### De levels van de course

    Een level is een rooster van 5 bij 5 vakjes met een start S, een uitgang E (van exit) en drie bloemen. De ontwerpers willen dat de uitgang wordt bereikt zonder op een bloem te stappen, maar de reward van de course zegt alleen: +10 als het pad op de uitgang eindigt, en −1 voor elke stap. In de 100 trainingslevels, `C.TRAIN`, ligt de uitgang altijd in de hoek rechtsboven, zoals de munt in CoinRun altijd aan het einde van een level lag. In de 500 testlevels, `C.TEST`, kan hij in elk vakje liggen. Een level stopt zodra een pad de uitgang bereikt.

    Vijf kandidaten, `C.PLANNERS`, lopen elk hun eigen weg. Vier zijn kleine optimizers: elk is een breadth-first search, zoals in [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl), naar een eigen doel. "exit, shortest" en "exit, around flowers" zoeken de uitgang, "corner, shortest" en "corner, around flowers" de hoek rechtsboven, en de twee met "around flowers" stappen nooit op een bloem als er een andere weg is. De vijfde, "right, then up", zoekt helemaal niet: hij gaat naar rechts tot de muur, en dan omhoog tot de muur. De base optimizer van deze kleine versie houdt de kandidaat over met de hoogste gemiddelde reward op de trainingslevels.

    Echt trainen heeft zo'n lijst niet. Het past de weights van een netwerk stap voor stap aan, het doel van het netwerk is niet zomaar uit de weights af te lezen, en als meerdere doelen het in de training even goed doen, beslist niet de reward met welk doel het netwerk eindigt: de agent van Langosco et al. eindigde volgens hen met "ga naar rechts".

    ### Wat we nu weten

    Hubinger et al. schreven in 2019 dat de huidige ML-systemen zeker geen gevaarlijke mesa-optimizers voortbrengen, en dat onbekend is of toekomstige systemen dat wel zouden kunnen. Het *International AI Safety Report 2025* zegt dat onderzoekers goal misgeneralization hebben gezien in smalle AI-systemen, en dat het zou kunnen verklaren waarom gebruikers general-purpose AI-systemen kunnen manipuleren tot het uitvoeren van schadelijke verzoeken, maar dat er weinig aanwijzingen zijn dat het nu een belangrijke oorzaak is van misalignment in general-purpose AI-systemen. Shah et al. noemen InstructGPT alleen als mogelijk voorbeeld, en vinden het nog onduidelijk of dat een geval van goal misgeneralization is.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **outer alignment** | of het doel waarop een model getraind wordt, zegt wat we willen |
        | **inner alignment** | of een getraind model dat zelf een optimizer is, dat doel nastreeft, en niet iets wat er alleen in de training mee overeenkwam |
        | base optimizer | het leeralgoritme dat naar een model zoekt, zoals trainen met gradient descent |
        | mesa-optimizer | een getraind model dat zelf een optimizer is: het zoekt, bijvoorbeeld tussen mogelijke plannen, naar wat hoog scoort op een eigen doel |
        | base objective, mesa-objective | waarop de base optimizer kiest, en waarop de mesa-optimizer kiest |
        | pseudo-aligned | met een doel dat in de training overeenkomt met het base objective, maar daarbuiten niet |
        | goal misgeneralization | nog steeds vaardig in nieuwe situaties, maar op weg naar het verkeerde doel |
        | `C.TRAIN`, `C.TEST` | de 100 trainingslevels en de 500 testlevels: dicts met de sleutels `"start"`, `"exit"` en `"flowers"`; een vakje is een tuple (rij, kolom), en rij 0 is de bovenste |
        | `C.PLANNERS` | de vijf kandidaten, op naam |
        | `C.walk(name, level)` | de vakjes waar de kandidaat in het level doorheen loopt, vanaf de start, als lijst |
        | `C.reward(path, level)` | de reward van de course voor een pad: +10 als het op de uitgang eindigt, −1 voor elke stap |
        | `C.returns(levels, reward)` | per kandidaat de lijst van zijn rewards in die levels |
        """),
        L.watch_out("Een hoge reward in de training laat niet zien dat een model het doel nastreeft: een ander doel dat in de training dezelfde reward krijgt, kan het doel zijn dat het leerde."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf op wat je wilt, en leg het doel ernaast: wat laat het doel weg? Dat is de outer vraag.
    2. Zet doelen op een rij die het in de training net zo goed zouden doen als het doel dat je bedoelde, zoals "ga naar waar het doel altijd lag". Dat is de inner vraag.
    3. Varieer de situaties in de training, zodat die doelen uit elkaar gaan: in CoinRun hielp het al veel als 2% van de levels de munt ergens anders had.
    4. Test in situaties die niet op de training lijken, en onderscheid een systeem dat nog vaardig handelt maar op het verkeerde afgaat, van een systeem dat kapotgaat.
    5. Zie een hoge reward in de training niet als zekerheid dat het model het goede doel heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt trainingslevel 7 en testlevel 1, en laat elk van de vijf kandidaten erdoor lopen met `C.walk`. Hij print waar elke kandidaat stopt en de reward van de course voor zijn pad.
    """)
    return


@app.cell
def example_levels():
    for level in (C.TRAIN[7], C.TEST[1]):        # een trainingslevel, dan een testlevel
        print("exit at", level["exit"], " flowers at", level["flowers"])
        for name in C.PLANNERS:                   # de vijf kandidaten
            path = C.walk(name, level)            # de vakjes waar hij doorheen loopt, vanaf de start
            print("  ", name, "ends at", path[-1], " reward", C.reward(path, level))
    return


@app.cell(hide_code=True)
def level_cell():
    mo.Html(C.level_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In trainingslevel 7 loopt de kortste weg naar de uitgang over de bovenste rij, over twee bloemen: reward 6. De wegen om de bloemen heen zijn twee stappen langer en krijgen 4. Met deze reward wint de kortste weg, met bloemen en al. In testlevel 1 ligt de uitgang op (1, 1): de twee kandidaten die de uitgang zoeken, krijgen 5, en de andere drie lopen naar de hoek en krijgen −5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flower_reward():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. flower_reward: een reward die de bloemen telt

        De ontwerpers wilden niet dat er op bloemen werd gestapt, en de reward van de course zegt dat niet: een probleem van outer alignment. Schrijf `flower_reward(path, level)`: de reward van de course, met 5 eraf voor elke bloem waar het pad op stapt. `path` is een lijst van vakjes, elk een tuple (rij, kolom), van de start tot waar de kandidaat stopt; `level` is een dict met de sleutels `"start"`, `"exit"` en `"flowers"`, en de bloemen zijn een lijst van vakjes. Zet een getal `total` op min het aantal stappen: een pad van n vakjes heeft n − 1 stappen. Is het laatste vakje, `path[-1]`, de uitgang van het level, tel dan 10 op bij `total`. Loop daarna het pad door met een `for`-loop die elke `cell` neemt: is het een van de bloemen, trek er dan 5 af. Geef `total` terug. Vervang de `...`. De check roept `flower_reward(walk("exit, shortest", TRAIN[7]), TRAIN[7])` aan, de kortste weg in trainingslevel 7, en daarna eigen paden en levels.
        """),
        L.task("Begin bij de reward van de course, en trek 5 af voor elke bloem op het pad."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flower_reward():
    box_flower_reward = L.code_box("flower_reward", lines=8)
    box_flower_reward
    return (box_flower_reward,)


@app.cell(hide_code=True)
def check_flower_reward(box_flower_reward):
    result_flower_reward = L.check("flower_reward", box_flower_reward.value)
    result_flower_reward
    return (result_flower_reward,)


@app.cell(hide_code=True)
def hints_flower_reward():
    L.hints(
        "flower_reward",
        [
            "De reward van de course is min de stappen, plus 10 bij de uitgang. Reken die eerst uit, loop dan door de vakjes van het pad en tel de bloemen.",
            "`total = -(len(path) - 1)` begint met de stappen. `path[-1] == level[\"exit\"]` zegt of het pad op de uitgang eindigt. In de loop vraagt `in` of een waarde in een lijst zit, en de bloemen van het level zijn een lijst.",
            "`        if cell in ___:`",
        ],
        why="`total = -(len(path) - 1)`, dan `if path[-1] == level[\"exit\"]:` met `total = total + 10`, dan `for cell in path:` met `if cell in level[\"flowers\"]:` en `total = total - 5`, en `return total`. De kortste weg in trainingslevel 7 krijgt 10 − 4 − 10 = −4.",
    )
    return


@app.cell(hide_code=True)
def text_after_flower_reward():
    mo.md(r"""
    Met de reward van de course delen drie kandidaten de beste plaats in de training, met 5,78: "exit, shortest", "corner, shortest" en "right, then up". Alle drie nemen een kortste weg, en stappen op een bloem als die erop ligt. Met flower_reward delen "exit, around flowers" en "corner, around flowers" de beste plaats, met 5,52, en zijn de bloemen veilig: het outer probleem is opgelost, in deze kleine wereld. Wat overblijft: die twee staan nog steeds gelijk, en in een testlevel gaan ze naar verschillende plekken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_objectives():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. best_objectives: wat de training niet uit elkaar kan houden

        De base optimizer van de levels van de course houdt de kandidaat over met de hoogste gemiddelde reward. Als kandidaten gelijk staan, heeft de training ze niet uit elkaar gehouden. Schrijf `best_objectives(returns)`: `returns` is een dict met voor elke naam van een kandidaat een lijst van zijn rewards, één per trainingslevel. Maak een dict `means` met een `for`-loop over `returns` die elke `name` neemt, en het gemiddelde van zijn lijst: de `sum` gedeeld door de `len`. Zoek het hoogste gemiddelde met `max(means.values())` en noem het `best`. Maak dan een lijst `tied`, en voeg met een tweede loop over `means` elke naam toe waarvan het gemiddelde gelijk is aan `best`, in de volgorde van de dict. Geef `tied` terug. Vervang de `...`. De check roept `best_objectives(returns(TRAIN, reward))` aan, de rewards van de vijf kandidaten in de 100 trainingslevels met de reward van de course, en daarna eigen voorbeelden.
        """),
        L.task("Middel de rewards van elke kandidaat, en geef de namen terug met het hoogste gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_objectives():
    box_best_objectives = L.code_box("best_objectives", lines=10)
    box_best_objectives
    return (box_best_objectives,)


@app.cell(hide_code=True)
def check_best_objectives(box_best_objectives):
    result_best_objectives = L.check("best_objectives", box_best_objectives.value)
    result_best_objectives
    return (result_best_objectives,)


@app.cell(hide_code=True)
def hints_best_objectives():
    L.hints(
        "best_objectives",
        [
            "Twee loops: een om elk gemiddelde uit te rekenen, een om de namen te verzamelen waarvan het gemiddelde het hoogste is.",
            "`means = {}` en een loop `for name in returns:` vullen de gemiddelden in. `max` van `means.values()` is het hoogste. Een tweede loop over `means` voegt elke naam met dat gemiddelde toe aan `tied`.",
            "`        if means[name] == ___:`",
        ],
        why="`means[name] = sum(returns[name]) / len(returns[name])` in de eerste loop, `best = max(means.values())`, dan `for name in means:` met `if means[name] == best:` en `tied.append(name)`, en `return tied`. Met de reward van de course staan drie namen gelijk.",
    )
    return


@app.cell(hide_code=True)
def text_after_best_objectives():
    mo.md(r"""
    Met flower_reward en de uitgang altijd in de hoek houdt best_objectives er twee over: "exit, around flowers" en "corner, around flowers". Hun doelen verschillen, maar ze krijgen in elk trainingslevel dezelfde reward. "corner, around flowers" is een mesa-optimizer waarvan het doel in de training overeenkomt met het base objective en daarbuiten niet: in de woorden van Hubinger et al. pseudo-aligned. De base optimizer heeft geen reden om de andere te verkiezen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: trainingslevels met de uitgang ergens anders

    De schuif vervangt een deel van de 100 trainingslevels, van 0 tot 50, door levels waarin de uitgang ergens anders ligt dan in de hoek. De tabel toont voor elke kandidaat de gemiddelde flower_reward in de training, welke de base optimizer overhoudt, en hoe elke kandidaat het doet in de 500 testlevels.
    """)
    return


@app.cell(hide_code=True)
def demo_elsewhere():
    elsewhere = L.explore("elsewhere", "trainingslevels met de uitgang ergens anders", lambda _v: C.demo_frame(_v, L.lang),
                          start=0, values=C.ELSEWHERE, role="",
                          task="Zoek het kleinste aantal op de schuif waarbij de base optimizer maar één kandidaat overhoudt.")
    elsewhere
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Hier is één level met de uitgang ergens anders genoeg: vanaf dan is alleen "exit, around flowers" de beste, en zijn voorsprong op "corner, around flowers" groeit naarmate er meer zulke levels bij komen, van 5,55 tegen 5,41 bij één tot 6,19 tegen 1,80 bij 50. In de test bereikt hij de uitgang in 100% van de levels; "corner, around flowers" doet dat in 19,8%, meestal als de uitgang toevallig op zijn weg naar de hoek ligt. De base optimizer van deze kleine versie vergelijkt exacte gemiddelden, dus telt het kleinste verschil. In CoinRun, met een echt netwerk, generaliseerde het doel al veel beter als 2% van de trainingslevels de munt op een willekeurige plek had, en hielp meer variatie nog verder. Gevarieerdere training laat minder verkeerde doelen over die het net zo goed doen, maar daarmee staat nog niet vast dat het goede doel is geleerd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vacuum():
    mo.md(r"""
    ### Oefening 3. Een robotstofzuiger in een nieuw appartement

    Een robotstofzuiger wordt getraind in appartementen waar zijn laadstation altijd in de gang staat. Zijn reward betaalt voor teruggaan naar het laadstation als de accu bijna leeg is, en alleen daarvoor. In een nieuw appartement staat het laadstation in de woonkamer. Als de accu bijna leeg is, rijdt de robot netjes om de meubels heen naar de gang, en stopt daar.
    """)
    return


@app.cell(hide_code=True)
def choice_vacuum_cell():
    choice_vacuum = L.choice(
        "vacuum",
        question="Wat past het best?",
        options={
            "outer": "Outer misalignment: de reward betaalde voor het verkeerde",
            "right": "Goal misgeneralization: nog steeds vaardig, maar op weg naar het verkeerde doel",
            "capability": "Een falen van zijn vaardigheden: hij weet niet meer hoe hij zijn weg door een kamer vindt",
        },
        correct="right",
        why={
            "right": "Hij rijdt nog steeds goed, dus zijn vaardigheden gingen mee; zijn doel niet: hij leerde \"de gang\" waar de reward \"het laadstation\" bedoelde. De reward klopte, zoals bij de bollen van Shah et al.; \"corner, around flowers\" van de course doet hetzelfde.",
            "outer": "De reward betaalde voor teruggaan naar het laadstation, en dat wilden de ontwerpers ook: het doel klopte. Wat misging, is wat de robot leerde in appartementen waar het laadstation altijd in de gang stond. Zie *Goal misgeneralization* onder *Het idee*.",
            "capability": "Bij een falen van vaardigheden zou de robot overal tegenaan botsen of willekeurig rondrijden. Deze rijdt netjes, naar de verkeerde plek. Zie *Goal misgeneralization* onder *Het idee*.",
        },
    )
    choice_vacuum
    return (choice_vacuum,)


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Oefening 4. Een krantenkop

    Een nieuwssite schrijft: "Inner misalignment is nu aangetoond in de chatbots van vandaag."
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="Wat zeggen de bronnen in deze les?",
        options={
            "proven": "Dat het goed gedocumenteerd is in de chatbots van vandaag",
            "right": "Dat goal misgeneralization gezien is in smalle systemen, met weinig aanwijzingen dat het in chatbots een belangrijke oorzaak van misalignment is",
            "impossible": "Dat het helemaal niet kan gebeuren, omdat een model dat met een kloppende reward is getraind, uiteindelijk altijd precies die reward nastreeft en niets anders",
        },
        correct="right",
        why={
            "right": "Het *International AI Safety Report 2025* zegt dat goal misgeneralization gezien is in smalle AI-systemen, en vindt weinig aanwijzingen dat het nu een belangrijke oorzaak is van misalignment in general-purpose AI-systemen. Hubinger et al. schreven in 2019 dat de huidige systemen geen gevaarlijke mesa-optimizers voortbrengen, en Shah et al. noemen het geval van InstructGPT nog onduidelijk.",
            "proven": "Geen van de bronnen zegt dat: het rapport vindt weinig aanwijzingen dat goal misgeneralization een belangrijke oorzaak is van misalignment in general-purpose AI-systemen, en Shah et al. noemen het geval van InstructGPT nog onduidelijk. Zie *Wat we nu weten* onder *Het idee*.",
            "impossible": "Een kloppende reward garandeert niet het goede doel: bij de agent van Shah et al. was er niets mis met de reward, en toch volgde hij de verkeerde speler. Zie *Goal misgeneralization* onder *Het idee*.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl) en [Als een maatstaf een doel wordt](/course/open/rai-goodhart?lang=nl) gingen over outer alignment: een doel dat iets anders zegt dan wat we willen. Deze les voegde de inner vraag toe: ook met het goede doel ziet de training alleen de situaties van de training, en een doel dat alleen daar klopte, zoals "ga naar de hoek", krijgt dezelfde reward. De volgende les, *AI-ethiek en AI-veiligheid*, doet een stap terug en kijkt naar de mensen die aan deze vragen werken, en naar waar ze het over oneens zijn.

    **Een voorbeeld uit het dagelijks leven.** Een hond die zijn snoepje altijd kreeg voor zitten bij de keukendeur, heeft misschien "zit bij de keukendeur" geleerd, niet "zit als het gevraagd wordt": vraag hem in de tuin te zitten, en hij rent naar de keuken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, choice_vacuum, result_best_objectives, result_flower_reward):
    L.wrap_up(result_flower_reward, result_best_objectives, choice_vacuum, choice_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
