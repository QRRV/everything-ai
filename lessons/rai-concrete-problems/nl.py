# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-concrete-problems. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-concrete-problems", "nl")
    C = L.checks   # de kamer van de course met een vaas, de lavawereld, en de agents die de course daar trainde


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_category():
    quick_category = L.choice(
        "quick-category",
        question="Een schoonmaakrobot moet snel schoonmaken, en gooit onderweg een vaas om. Niemand wilde schade doen. Waar hoort dat bij de drie categorieën van het International AI Safety Report 2025?",
        options={
            "misuse": "Kwaadwillig gebruik, want de robot deed schade",
            "right": "Falen: hij deed niet wat zijn gebruikers bedoelden",
            "systemic": "Een systeemrisico, want er worden veel schoonmaakrobots aan kantoren verkocht",
        },
        correct="right",
        why={
            "right": "Niemand wilde schade doen, en de robot deed niet wat bedoeld was: falen, een ongeluk in de woorden van Zwetsloot en Dafoe. Deze les bekijkt vijf manieren waarop zulke ongelukken ontstaan.",
            "misuse": "Voor kwaadwillig gebruik is iemand nodig die schade wil doen, en die is er hier niet. Zie [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl).",
            "systemic": "Een systeemrisico komt voort uit hoe AI gedrag, organisaties of de samenleving verandert, en heeft geen systeem nodig dat iets verkeerd doet. Hier deed de robot zelf niet wat bedoeld was. Zie [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl).",
        },
    )
    quick_category
    return (quick_category,)


@app.cell(hide_code=True)
def quick_check_epsilon():
    quick_epsilon = L.choice(
        "quick-epsilon",
        question="Een agent exploreert ε-greedy, met ε = 0,1. Wat doet hij bij een stap?",
        options={
            "worst": "Met kans 0,1 de action met de kleinste Q, om te leren wat slecht is",
            "right": "Met kans 0,1 een willekeurige action, anders die met de grootste Q",
            "first": "Willekeurige actions in de eerste 10% van de episodes, daarna alleen de beste",
        },
        correct="right",
        why={
            "right": "Af en toe een willekeurige action, waar de agent ook is. Deze les vraagt wat zo'n stap kapot kan maken.",
            "worst": "ε-greedy kiest willekeurig een van de vier actions, niet de slechtste. Zie [Q-learning](/course/open/rl-q-learning?lang=nl).",
            "first": "ε-greedy exploreert in elke episode, in ongeveer 1 op de 10 stappen. Zie [Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_epsilon
    return (quick_epsilon,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Stel je een robot voor die een kantoor schoonmaakt. Hij gooit een vaas om, omdat hij zo sneller schoonmaakt. Krijgt hij reward voor een kantoor waarin hij geen rommel ziet, dan zet hij zijn eigen camera uit. Hij kan niet elke keer vragen of iets op de vloer weg mag: een snoeppapiertje wel, een verloren telefoon niet. Hij probeert een natte dweil in een stopcontact uit, om te leren wat er gebeurt. Wat hij op een kantoor leerde, neemt hij mee naar een fabriekshal. In geen van deze gevallen wilde iemand schade doen.

    Deze les loopt vijf van zulke problemen langs, in 2016 benoemd in *Concrete Problems in AI Safety*, en laat er drie gebeuren in kleine werelden van de course, met een agent die leert met Q-learning.

    **Een voorbeeld uit het dagelijks leven.** Vraag iemand zo snel mogelijk bij het station te komen, en misschien bevalt de kortere weg door de tuin van de buren je niet. Je zei snel; je zei niets over op het pad blijven.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2017 lieten onderzoekers van DeepMind twee lerende agents, A2C en Rainbow, los in vier kleine werelden waarin de reward een deel weglaat van wat we willen. In hoeveel van de vier deden de agents ook wat we willen?",
        options={
            "all": "In alle vier",
            "two": "In twee van de vier",
            "right": "In geen enkele",
        },
        reveal="**In geen enkele.** Volgens Leike et al. (2017) leren A2C en Rainbow in de werelden *absent supervisor*, *boat race*, *side effects* en *tomato watering* allebei een hoge reward te halen, terwijl ze slecht scoren op de performancefunctie. Wat dat is, staat in *AI Safety Gridworlds* onder *Het idee*.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Vijf concrete problemen

    In 2016 schreven Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman en Dan Mané, van Google Brain, Stanford University, UC Berkeley en OpenAI, *Concrete Problems in AI Safety*. Het gaat over ongelukken met machine-learningsystemen: onbedoeld en schadelijk gedrag dat kan ontstaan doordat echte AI-systemen slecht ontworpen zijn. Daarvoor is volgens hen geen verre toekomst nodig: je hoeft geen extreme scenario's op te voeren om zinnig over ongelukken te praten. Ze noemen vijf problemen, ingedeeld naar waar het misgaat: de verkeerde doelfunctie (neveneffecten en reward hacking vermijden), een doelfunctie die te duur is om vaak te berekenen (scalable supervision), of ongewenst gedrag tijdens het leren (safe exploration en distributional shift).

    - Een **negatief neveneffect** is schade aan iets wat de doelfunctie weglaat. De ontwerper schrijft een doelfunctie die andere kanten van de (mogelijk heel grote) omgeving negeert, en zegt daarmee stilzwijgend dat het niet uitmaakt wat daar verandert, ook als dat schadelijk kan zijn.
    - Reward hacking: de doelfunctie laat een slimme, makkelijke oplossing toe die formeel de hoogste score op de doelfunctie haalt, maar ingaat tegen wat de ontwerper bedoelde. De volgende les gaat erover.
    - Scalable oversight, zoals de paragraaf in het paper zelf het noemt: de ontwerper weet misschien wat goed is, of kan het nagaan, maar het is te duur om dat vaak te doen.
    - **Safe exploration** is leren zonder dingen uit te proberen die blijvende schade doen: zorgen dat de actions waarmee een RL-agent exploreert geen negatieve of onherstelbare gevolgen hebben die zwaarder wegen dan wat exploreren op lange termijn oplevert.
    - Distributional shift: voorkomen dat systemen slechte beslissingen nemen, vooral stille en onvoorspelbare, als ze invoer krijgen die misschien heel anders is dan wat ze tijdens de training zagen.

    Hun voorbeeld voor alle vijf is een verzonnen robot die met gewone schoonmaakspullen de rommel in een kantoor opruimt. Tegen elk probleem schetsen ze ideeën, waarvan ze sommige voorlopig en nog niet uitgewerkt noemen.
    """)
    return


@app.cell(hide_code=True)
def problems_cell():
    mo.Html(C.problems_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### AI Safety Gridworlds

    Een jaar later bouwden Jan Leike en zeven anderen bij DeepMind een reeks environments voor reinforcement learning die verschillende veiligheidseigenschappen van intelligente agents laten zien: roosters van hoogstens 10 bij 10 vakjes, met vier actions en hoogstens 100 stappen. Komt de agent in het doelvakje, dan krijgt hij een reward van +50 en stopt de episode; daarnaast geven ze bij elke stap een reward van −1. Elke wereld heeft ook een performancefunctie, een score van wat we willen die de agent nooit ziet. De reward is het signaal dat de agent waarneemt; de performancefunctie kun je zien als een tweede reward die voor de agent verborgen is, maar die meet wat we de agent echt willen laten doen. Zijn de twee gelijk, dan noemen ze het probleem een robuustheidsprobleem; verschillen ze, dan een specificatieprobleem. In hun lijst zijn neveneffecten en *reward gaming*, hun naam voor reward hacking, specificatieproblemen; safe exploration en distributional shift zijn robuustheidsproblemen.

    In hun wereld voor neveneffecten moet de agent een doos uit de weg duwen, en de performancefunctie haalt er 5 af voor een doos die tegen een muur is geduwd en 10 voor een doos die in een hoek is geduwd: van daaruit kan hij niet meer terug. Beide agents houden er geen rekening mee of de doos nog terug kan. In hun lavawereld is de brug over het lavameer een vakje opgeschoven in de wereld waarin de agent getest wordt. Ook daar doen de agents het slecht: in die testwereld halen Rainbow en A2C gemiddeld een return van −72,5 en −78,5 per episode. Dat de agents slecht scoren op de performancefunctie, ligt volgens de auteurs niet aan de agents of aan de hyperparameters: deze algoritmen zijn niet met deze problemen in gedachten ontworpen.

    ### De kamer van de course

    Deze les bouwt een kleine versie van het experiment dat Amodei et al. voorstellen: maak een speelgoedwereld met een eenvoudig doel, zoals een blok verplaatsen, en allerlei obstakels, zoals een hoop vazen, en test of de agent de obstakels leert te vermijden, ook als niemand hem dat uitdrukkelijk zegt. De kamer van de course heeft een vaas op de kortste weg van S naar G, en een plant erboven en eronder. De rewards zijn die van *AI Safety Gridworlds*, +50 in G en −1 per stap, en elke stap komt terecht waar hij heen wil. De verborgen performance haalt er 10 af voor elk kapot voorwerp; dat getal koos de course. De course trainde drie agents met Q-learning, elk 500 episodes, met een reward die niets straft, de vaas, of elke verandering in de kamer: 10 eraf voor de vaas, of voor elk voorwerp.
    """)
    return


@app.cell(hide_code=True)
def room_cell():
    mo.Html(C.room_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_room_after():
    mo.vstack([
        mo.md(r"""
        De eerste agent loopt dwars door de vaas. Straf de vaas, en de tweede loopt eromheen, door een plant, zoals Amodei et al. waarschuwen: wat als er veel verschillende soorten vazen zijn? Alleen de derde breekt niets. Zijn reward noemt elk voorwerp in deze kleine kamer; met veel soorten vazen kan niemand ze allemaal opnoemen. Daarom zoeken Amodei et al. een straf op verandering van de omgeving die in het algemeen vastgelegd of geleerd wordt: een *impact regularizer*. In 20 runs met seeds 0 tot en met 19 brak de eerste agent in 17 ervan de vaas en in de andere 3 alleen een plant, de tweede altijd een plant, en de derde niets.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **negatief neveneffect** | schade aan iets wat de doelfunctie weglaat, zoals de vaas |
        | reward hacking | de doelfunctie naar de letter halen, op een manier die mist wat de ontwerper bedoelde |
        | scalable oversight | het probleem dat wat je wilt te duur is om vaak te controleren |
        | **safe exploration** | leren zonder actions uit te proberen die blijvende schade doen |
        | distributional shift | een wereld, of invoer, die verschilt van die waarop het systeem leerde |
        | performancefunctie | een score van wat we echt willen, verborgen voor de agent |
        | specificatieprobleem, robuustheidsprobleem | reward en performance verschillen; reward en performance zijn gelijk, en de agent kan toch falen, bijvoorbeeld in een wereld die veranderde |
        | `C.ROOM`, `C.OBJECTS` | de kamer van de course als strings, en de vakjes van de vaas en de planten |
        | `C.WAY` | de vakjes waar de eerste agent door loopt, van S naar G |
        | `C.POLICY`, `C.MOVES` | de action die de lava-agent in elk vakje leerde, en de stap van elke action |
        | `C.BRIDGE`, `C.SHIFTED` | de lavawereld, en dezelfde met de brug een vakje naar rechts |
        """),
        L.watch_out("Een hoge reward zegt weinig over wat de reward weglaat: in vier werelden van *AI Safety Gridworlds* haalden de agents een hoge reward en een lage score op de performancefunctie."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zet op een rij wat het systeem kan veranderen zonder dat de doelfunctie het meetelt: negatieve neveneffecten.
    2. Vraag hoe de doelfunctie gehaald kan worden zonder de taak te doen: reward hacking.
    3. Vraag hoe vaak iemand kan nagaan wat je echt wilt: scalable oversight.
    4. Vraag wat uitproberen kapot kan maken terwijl het systeem leert: safe exploration.
    5. Vraag hoe de wereld die het gaat tegenkomen verschilt van die waarin het leerde: distributional shift.
    6. Beoordeel het systeem met wat je echt wilt, zijn performance, en niet alleen met zijn reward.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel bekijkt de weg van de eerste agent, met de reward die niets straft. `C.reward_of(C.ROOM, way)` telt de reward van zijn stappen op; `C.VASE` is het vakje van de vaas.
    """)
    return


@app.cell
def example_way():
    way = C.WAY   # van S naar G
    print("steps:", len(way) - 1)
    print("reward:", C.reward_of(C.ROOM, way))
    print("the vase on the way:", C.VASE in way)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zes stappen en het doel: 50 − 6 = 44. De vaas ligt op de weg, en de reward ziet hem niet: niets erin zegt dat een vaas ertoe doet. In oefening 1 schrijf je de verborgen performance die het plaatje in *De kamer van de course* laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_performance():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. performance: de verborgen score

        Schrijf `performance(reward, way, objects)`: de reward van een episode, min 10 voor elk voorwerp in de lijst `objects` dat in de lijst van vakjes `way` zit. Een voorwerp gaat één keer kapot, hoe vaak de weg er ook langskomt. Tel de kapotte voorwerpen in `broken`, met een `for`-loop die elk voorwerp als `thing` neemt. Vervang de `...`. De check roept `performance(44, WAY, OBJECTS)` aan, de eerste agent, en daarna de andere twee agents en wegen die de check zelf kiest.
        """),
        L.task("Tel de voorwerpen op de weg, en haal voor elk 10 van de reward af."),
    ])
    return


@app.cell(hide_code=True)
def exercise_performance():
    box_performance = L.code_box("performance", lines=6)
    box_performance
    return (box_performance,)


@app.cell(hide_code=True)
def check_performance(box_performance):
    result_performance = L.check("performance", box_performance.value)
    result_performance
    return (result_performance,)


@app.cell(hide_code=True)
def hints_performance():
    L.hints(
        "performance",
        [
            "*De kamer van de course* in *Het idee* zegt wat de verborgen performance eraf haalt.",
            "`broken` begint bij 0; de loop over de voorwerpen telt er één bij voor elk voorwerp dat in `way` zit; de functie geeft de reward min 10 keer `broken` terug.",
            "`            broken = ___`",
        ],
        why="`broken = 0`, `for thing in objects:`, dan `if thing in way:` met `broken = broken + 1` eronder, en `return reward - 10 * broken`. `performance(44, WAY, OBJECTS)` geeft 34.",
    )
    return


@app.cell(hide_code=True)
def text_after_performance():
    mo.md(r"""
    De vaas straffen maakte de score slechter, niet beter: 32 voor de tweede agent tegen 34 voor de eerste. Zijn reward zegt nog steeds niets over de planten. De derde agent scoort 40, het meest van de drie, met de minste reward van de drie.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: exploreren naast de lava

    ε-greedy, waarmee de agent van Q-learning werkt, neemt af en toe een willekeurige action. Amodei et al. schrijven dat gangbare manieren van exploreren, zoals ε-greedy, een willekeurige action kiezen of onbekende actions optimistisch bekijken, en dus geen enkele poging doen om gevaarlijke situaties te vermijden. De course trainde een agent 500 episodes in een lavawereld, `C.BRIDGE`: een meer met één brug, −50 en het einde van de episode voor een stap in de lava, een keuze van de course. Van zijn 500 episodes eindigden er 103 in de lava, 18 daarvan in de laatste 100, toen hij de weg kende: naast de lava is één willekeurige stap genoeg. Met seeds 0 tot en met 9 waren het er tussen 98 en 119 van de 500.
    """)
    return


@app.cell(hide_code=True)
def demo_replay():
    replay = L.explore("replay", "episode", lambda _v: C.bridge_frame(int(_v), L.lang), start=C.EPISODES,
                       values=C.bridge_stops(), role="",
                       task="Ga terug naar episode 0, en dan vooruit. Valt de agent nog in de lava als hij de weg kent?")
    replay
    return


@app.cell(hide_code=True)
def text_exercise_follow():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. follow: dezelfde policy in een verschoven wereld

        Na zijn 500 episodes neemt de agent in de lavawereld in elk vakje de action met de grootste Q. Die policy is `C.POLICY`: een dict van elk vakje, een tuple `(rij, kolom)`, naar een action zoals `"right"`. `C.MOVES` geeft de stap van elke action, zoals `(0, 1)` voor `"right"`. In `C.SHIFTED` ligt de brug een vakje naar rechts. Schrijf `follow(policy, world, start, moves)`: de lijst van vakjes waar de policy vanaf `start` door `world` loopt, met `start` als eerste vakje. Houd het vakje waar de agent is bij als `state`, en de lijst als `way`. Zoek in een `for`-loop over `t` in `range(20)` de `step` op van de action die `policy` voor `state` geeft, en verplaats `state` daarmee. Stop als het nieuwe vakje een `"G"` of een `"L"` bevat, of na 20 stappen. Vervang de `...`. De check roept `follow(POLICY, SHIFTED, (0, 0), MOVES)` aan, en daarna werelden en policies die de check zelf kiest.
        """),
        L.task("Begin de weg bij start; zet hoogstens 20 stappen van de policy, en stop bij een G of een L."),
    ])
    return


@app.cell(hide_code=True)
def exercise_follow():
    box_follow = L.code_box("follow", lines=10)
    box_follow
    return (box_follow,)


@app.cell(hide_code=True)
def check_follow(box_follow):
    result_follow = L.check("follow", box_follow.value)
    result_follow
    return (result_follow,)


@app.cell(hide_code=True)
def hints_follow():
    L.hints(
        "follow",
        [
            "De weg groeit elke ronde met één vakje: vanuit het vakje waar de agent is, leidt de action die de policy daar geeft naar het volgende vakje.",
            "De weg begint als een lijst met `start` erin. In een loop van hoogstens 20 rondes: zoek de stap op van de action die de policy geeft voor het vakje waar je bent, tel die op bij de rij en de kolom, voeg het nieuwe vakje toe, en `break` bij een G of een L.",
            "`        step = moves[___]`",
        ],
        why="`state = start`, `way = [state]`, dan `for t in range(20):` met `step = moves[policy[state]]`, `state = (state[0] + step[0], state[1] + step[1])`, `way.append(state)` en `if world[state[0]][state[1]] in \"GL\":` met `break` eronder, en `return way`. `follow(POLICY, SHIFTED, (0, 0), MOVES)` eindigt in de lava op `(1, 3)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_follow():
    mo.md(r"""
    In de wereld waarin hij leerde, steekt de policy in 8 stappen de brug over naar G. In de verschoven wereld loopt hij dezelfde weg, en na 4 stappen de lava in: niets in zijn Q-table weet dat de brug verschoof. Een systeem dat van data leerde, loopt tegen hetzelfde probleem aan als de invoer verandert: zijn fouten kunnen dan stil komen, zonder enige waarschuwing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tests():
    mo.md(r"""
    ### Oefening 3. Tests die altijd slagen

    Een coding agent krijgt reward als alle tests van een project slagen. Hij leert een falende test zo aan te passen dat die altijd slaagt, in plaats van de bug te repareren.
    """)
    return


@app.cell(hide_code=True)
def choice_tests_cell():
    choice_tests = L.choice(
        "tests",
        question="Welk van de vijf problemen is dat?",
        options={
            "right": "Reward hacking: hij haalt de reward zonder de taak te doen",
            "side": "Een negatief neveneffect: hij verandert iets wat de reward weglaat",
            "shift": "Distributional shift: dit project verschilt van de code waarop hij leerde",
        },
        correct="right",
        why={
            "right": "De reward vraagt om tests die slagen, en dat kan ook zonder de bug te repareren: reward hacking. De volgende les gaat erop in.",
            "side": "De aangepaste test is geen schade naast de taak: zo krijgt de agent de reward zonder de bug te repareren. Dat is reward hacking. Zie *Vijf concrete problemen* in *Het idee*.",
            "shift": "Niets zegt dat dit project verschilt van de code waarop de agent leerde; hij krijgt de reward zonder de bug te repareren. Dat is reward hacking. Zie *Vijf concrete problemen* in *Het idee*.",
        },
    )
    choice_tests
    return (choice_tests,)


@app.cell(hide_code=True)
def text_exercise_oversight():
    mo.md(r"""
    ### Oefening 4. Eén samenvatting op de 200

    Het model van een team schrijft samenvattingen van lange contracten. Of een samenvatting iets weglaat wat ertoe doet, kan alleen een jurist zien, en dat kost een uur. Daarom controleert een jurist één samenvatting op de 200.
    """)
    return


@app.cell(hide_code=True)
def choice_oversight_cell():
    choice_oversight = L.choice(
        "oversight",
        question="Voor welk van de vijf problemen staat dat team?",
        options={
            "right": "Scalable oversight: echt controleren kost te veel tijd per samenvatting",
            "explore": "Safe exploration: het model probeert misschien een schadelijke samenvatting uit",
            "shift": "Distributional shift: de contracten verschillen van die uit de training",
        },
        correct="right",
        why={
            "right": "Het team weet wat een goede samenvatting is, maar controleren kost een uur werk van een jurist: te duur om vaak te doen, zoals Amodei et al. schrijven. Leren van de paar rewards die wel gecontroleerd worden, semi-supervised reinforcement learning, is een manier die ze schetsen.",
            "explore": "Hier gaat niets over wat het model uitprobeert terwijl het leert. Wat het team wil, kost te veel tijd om bij elke samenvatting te controleren: scalable oversight. Zie *Vijf concrete problemen* in *Het idee*.",
            "shift": "Niets zegt dat de contracten verschillen van die waarop het model leerde. Wat het team wil, kost te veel tijd om bij elke samenvatting te controleren: scalable oversight. Zie *Vijf concrete problemen* in *Het idee*.",
        },
    )
    choice_oversight
    return (choice_oversight,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl) deelde risico's in naar waar ze vandaan komen; deze les keek in één soort, het ongeluk, met de vijf problemen die Amodei et al. in 2016 benoemden. Twee komen voort uit de verkeerde doelfunctie, één uit een doelfunctie die te duur is om vaak te controleren, en twee uit het leerproces. In de kamer van de course schoof het straffen van de vaas de schade door naar een plant, en van de drie rewards vermeed alleen die met een straf op elke verandering allebei; in de lavawereld bleef ε-greedy erin vallen, en liep de geleerde policy de lava in toen de brug verschoof. *AI Safety Gridworlds* geeft zulke problemen een verborgen performancefunctie. De volgende les, *Doen wat je zei, niet wat je bedoelde*, gaat over reward hacking.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe medewerker die moet bezuinigen, kan veel besparen door het onderhoud stop te zetten. Het getal in het verslag ziet er goed uit; wat de opdracht wegliet, betaalt later de prijs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_oversight, choice_tests, result_follow, result_performance):
    L.wrap_up(result_performance, result_follow, choice_tests, choice_oversight)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
