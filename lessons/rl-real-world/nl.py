# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-real-world. Route: course/route/lessen/reinforcement-learning.yaml. Checks, het datacenter van de course in een simulator, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-real-world", "nl")
    C = L.checks   # het datacenter van de course in een simulator, de Q-learning-agent, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bins():
    quick_bins = L.choice(
        "quick-bins",
        question="In MountainCar worden de positie en de snelheid elk in 12 bins verdeeld, zodat een Q-table kan leren. Hoeveel rijen heeft de tabel nodig?",
        options={
            "bins": "24: één voor elke bin van de twee getallen",
            "right": "144: één voor elk paar bins",
            "two": "2: één voor elk getal",
            "speed": "12: de snelheid verandert te snel om eigen rijen nodig te hebben",
        },
        correct="right",
        why={
            "right": "Elke state is een paar, een bin van de positie en een bin van de snelheid: 12 × 12 = 144 rijen. In deze les is de state ook een paar: de temperatuur van een ruimte en het uur van de dag.",
            "bins": "Een rij hoort bij één state, en een state heeft tegelijk een bin van de positie en een bin van de snelheid. Hoeveel paren kun je maken uit 12 en 12 bins? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
            "two": "Een rij van een Q-table hoort bij een state, niet bij een getal. In hoeveel verschillende paren bins kan MountainCar zijn? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
            "speed": "Zonder de snelheid weet de auto niet of een zwaai de heuvel op al lukt. Met 12 bins voor elk getal, hoeveel states zijn er? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
        },
    )
    quick_bins
    return (quick_bins,)


@app.cell(hide_code=True)
def quick_check_explore():
    quick_explore = L.choice(
        "quick-explore",
        question="Een Q-learning-agent met ε = 0,1 zet 1.000 stappen. Ongeveer hoeveel van de actions worden willekeurig gekozen?",
        options={
            "none": "0: als de agent eenmaal geleerd heeft, kiest de agent altijd de beste action",
            "right": "Ongeveer 100: één op elke tien stappen",
            "few": "Ongeveer 10: ε is een klein getal",
            "all": "Ongeveer 1.000: in het begin is elke action een gok",
        },
        correct="right",
        why={
            "right": "Met kans ε kiest de agent een willekeurige action, dus zijn ongeveer 0,1 × 1.000 = 100 van de 1.000 stappen willekeurig, ook nadat de agent geleerd heeft. In deze les zijn zulke willekeurige actions de uren waarin een datacenter te heet wordt terwijl de agent leert.",
            "none": "ε-greedy blijft willekeurige actions proberen zolang ε boven 0 is, ook met een goede tabel. Wat is 0,1 × 1.000? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "few": "ε is bij elke stap de kans op een willekeurige action. Wat is 0,1 keer 1.000? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "all": "Met ε = 0,1 kiest de agent in 9 van elke 10 stappen de beste action, ook als de tabel nog leeg is. Hoeveel van 1.000 stappen zijn dan willekeurig? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_explore
    return (quick_explore,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    RL werd bekend door games, maar draait ook op plekken waar fouten iets kosten. In 2016 meldden DeepMind en Google dat neural networks de energie voor het koelen van een datacenter van Google met 40% hadden verlaagd, waarbij mensen uitvoerden wat het systeem adviseerde; in 2018 regelde een AI-systeem de koeling zelf, met gemiddeld zo'n 30% besparing. In een studie uit 2022 hield een RL-agent het plasma van een fusiereactor in Lausanne in vorm. RL plaatste ook de onderdelen van computerchips, en vond sorteerroutines die nu in een standaardbibliotheek van C++ draaien. In een game kost een fout een verloren partij, en een seconde later speelt de agent de volgende. In een datacenter betekent een fout servers die te heet worden. Deze les loopt vijf zulke problemen langs met dezelfde vragen, en traint dan een agent voor een datacenter in een simulator, om te zien wat er verandert als fouten een prijs hebben.

    **Een voorbeeld uit het dagelijks leven.** Niemand leert autorijden op de snelweg. Je begint op een leeg parkeerterrein, of zelfs in een simulator, waar een fout niets kost. Op de weg heeft de instructeur een tweede rempedaal en grijpt in bij een fout die te veel zou kosten. Hoe meer de weg van het parkeerterrein verschilt, hoe meer je dat pedaal nodig hebt.
    """)
    return


@app.cell(hide_code=True)
def predict_hot():
    guess_hot = L.guess(
        "hot",
        question="In de simulator van de les traint Q-learning 10.000 dagen met ε = 0,1: servers warmen een ruimte op, de agent koelt, en elk uur boven 27 graden kost een straf. Hoeveel uur is de ruimte tijdens de training in totaal te heet, denk je?",
        options={
            "none": "Geen: de straf leert de agent meteen onder 27 te blijven",
            "few": "Zo'n 200, in de eerste dagen",
            "some": "Zo'n 2.000",
            "right": "Meer dan 20.000",
        },
        reveal="**Meer dan 20.000.** In de run van de course zijn het 21.528 uur, ongeveer tweeënhalf jaar servers die te heet zijn. Alleen al op de eerste dag is de ruimte 17 uur te heet, want met de lege tabel staat de koeling overal uit; in de eerste 1.000 dagen 7.926 uur; en zelfs aan het eind, met een goede tabel, zo'n 220 uur per 1.000 dagen, omdat ε-greedy in één op de tien stappen een willekeurige action blijft proberen. In een simulator is elk van deze uren gratis. In een echt datacenter zou elk uur de hardware in gevaar brengen.",
    )
    guess_hot
    return (guess_hot,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vijf problemen, dezelfde vragen.** Elke toepassing van RL beantwoordt dezelfde vragen: wat ziet de agent (de state), wat kan de agent veranderen (de actions), wat levert dat op (de reward), en waar kan de agent dingen uitproberen? En buiten games nog één: wat kost een fout?

        | Probleem | State | Actions | Reward | Een fout kost |
        |---|---|---|---|---|
        | video's aanraden (YouTube, 2019) | wat iemand eerder keek | welke video's tonen, uit miljoenen | hoeveel mensen kijken; een video waar niemand op klikt geeft 0 | een saaie suggestie, miljoenen keren |
        | een datacenter koelen (Google, 2018; Trane, 2022) | temperaturen en andere sensoren | de instellingen van de koeling | bespaarde stroom, binnen grenzen | servers die te heet worden |
        | plasma vasthouden in een fusiereactor (TCV, 2022) | metingen aan het plasma | spanningen op 19 spoelen, 10.000 keer per seconde | hoe dicht positie, stroomsterkte en vorm van het plasma bij het doel zitten | plasma dat de wand raakt: afkoeling, en mogelijk schade |
        | de onderdelen van een chip plaatsen (Google, 2021) | de chip tot nu toe | waar het volgende blok komt | korte draden, weinig drukte | een tragere of grotere chip |
        | sneller sorteren (AlphaDev, 2023) | het programma tot nu toe, en wat het programma met testinvoer doet | nog één instructie | correct, en snel | niets: een fout programma wordt weggegooid |

        **Waar de agent kan oefenen.** AlphaDev en de chipplaatsingen worden in een computer uitgeprobeerd, dus kost een fout alleen rekentijd; AlphaDev vond routines om drie, vier en vijf getallen te sorteren, tot 70% sneller voor vijf, en deze routines zitten nu in de standaardsortering van de C++-bibliotheek van LLVM. Het is omstreden of de chipplaatsingen de beste oudere methoden verslaan: andere onderzoekers konden de resultaten niet reproduceren, en volgens Google lieten deze onderzoekers stappen weg waar de methode niet zonder kan. Op het plasma van de TCV-tokamak in Lausanne kun je niet oefenen, dus trainden Jonas Degrave en collega's een agent in een simulator van de tokamak, en lieten de agent zonder verdere training de echte machine sturen. Het aanbevelingssysteem van YouTube leerde van wat eerdere systemen hadden vastgelegd, en hield er rekening mee dat de keuzes van deze eerdere systemen kwamen. En de koelsystemen leerden op echte gebouwen, met grenzen op elke action, regels om op terug te vallen, en mensen die de AI op elk moment konden uitzetten; in twee gebouwen van het bedrijf Trane bespaarde een RL-agent zo'n 9% en 13% van de energie.

        **Een datacenter in een simulator.** Het probleem van deze les is een klein datacenter, steeds één dag van 24 uur, vanaf 20 graden. De servers warmen de ruimte 's nachts 1 graad per uur op, overdag 3 en 's avonds 2. Elk uur kiest de agent een van drie actions: uit; laag, wat 2 graden wegneemt voor 1 eenheid stroom; of hoog, wat 4 graden wegneemt voor 3 eenheden. Een eenheid kost 's nachts 1, overdag 4 en 's avonds 2, en elk uur boven 27 graden kost nog 10 extra, omdat de servers te heet worden. De state is de temperatuur en het uur; de reward is min de rekening, min 10 voor elk uur te heet. De kosten van een dag zijn de rekening plus deze straffen.
        """),
        mo.Html(C.days_picture(L.lang)),
        mo.md(r"""
        De thermostaat is een vaste regel: hoog koelen vanaf 25 graden en laag vanaf 23. Een dag met de thermostaat kost 103. Na 10.000 dagen leren in de simulator kost de agent 63, 39% minder. De agent koelt vanaf de nacht bijna elk uur laag, als stroom goedkoop is, tot 15 graden in de ochtend, en laat de ruimte daarna langzaam opwarmen tot 27 in de avond. Hoog koelen doet de agent nooit: hoog kost drie eenheden voor wat twee uur laag koelen met twee doet. De thermostaat wacht tot de ruimte warm is, en moet dan in de dure uren hard koelen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **simulator** | | een programma dat een deel van de wereld nabootst, zodat een agent daar actions kan uitproberen en fouten kan maken, sneller en goedkoper dan in werkelijkheid | `step` |
        """),
        L.watch_out("Een simulator is nooit de wereld zelf. Na het leren in een simulator kent een agent alleen de states die de simulator liet zien: in een state die nooit langskwam, staan in de Q-table nog de nullen van het begin."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een echt probleem in RL gieten:

    1. De state: wat ziet de agent op elk moment?
    2. De actions: wat kan de agent veranderen?
    3. De reward: één getal per stap voor wat je wilt, met een kostenpost voor wat niet mag gebeuren.
    4. Waar de agent kan oefenen: in een simulator, in data die eerdere systemen vastlegden, of in de wereld zelf.
    5. Wat een fout kost tijdens het leren en daarna, en wat een fout opvangt: grenzen op elke action, een regel om op terug te vallen, een mens die het kan overnemen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel laat de simulator van de les met de hand de eerste tien uur van een dag draaien, met de thermostaat: hoog koelen vanaf 25 graden, laag vanaf 23, en uit daaronder. `[1] * 8` is een lijst van acht enen, en `+` zet lijsten achter elkaar, dus bevat `HEAT` de warmte van alle 24 uur. Voer de cel uit.
    """)
    return


@app.cell
def example_thermostat():
    HEAT = [1] * 8 + [3] * 12 + [2] * 4   # graden die de servers in elk uur van de dag toevoegen
    COOL = [0, 2, 4]                       # graden die elke action wegneemt: uit, laag, hoog
    temp = 20
    for hour in range(10):
        if temp >= 25:
            action = 2
        elif temp >= 23:
            action = 1
        else:
            action = 0
        temp = temp + HEAT[hour] - COOL[action]
        print("hour", hour, "action", action, "temperature", temp)
    return COOL, HEAT, action, hour, temp


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    's Nachts houdt de thermostaat de ruimte tussen 22 en 23 graden, met om het andere uur laag koelen. Bij uur 8 begint de drukke dag en springt de ruimte naar 25, dus koelt de thermostaat bij uur 9 hoog: 3 eenheden stroom tegen de dagprijs van 4, een rekening van 12 voor één uur. Dat zijn de uren die de agent vermijdt door 's nachts te koelen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De simulator

        `step(temp, hour, action)` is de simulator van de les: uit de temperatuur `temp` nu, het uur `hour` en de action `action` geeft de functie `new`, de temperatuur een uur later, en de reward. De eerste regel telt de warmte van de servers op, `HEAT[hour]`, en trekt af wat de action koelt, `COOL[action]`, zoals in het voorbeeld; de tweede regel houdt `new` met `min` en `max` tussen `COLDEST`, 15, en `HOTTEST`, 35. Dan is `reward` min de rekening van dit uur: stroom kost `PRICE[hour]` per eenheid, en de action gebruikt `ENERGY[action]` eenheden, 0, 1 of 3. Boven `LIMIT`, 27 graden, kost het uur nog `PENALTY`, 10, extra. Vervang de `...` door min de rekening.

        De check roept `step(24, 10, 1)` aan: laag koelen bij 24 graden om 10 uur, in de dure dag. Dan zes andere gevallen: een uur te heet overdag, hoog koelen dat de ruimte onder 15 graden zou brengen, laag koelen precies op de grens, een ruimte die heter dan 35 zou worden, en twee gewone uren.
        """),
        L.task("Maak `reward` min de stroomrekening van dit uur."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("step", lines=7)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "step",
        [
            "Stap 3 van het recept: `reward` is één getal voor wat je wilt, en een rekening is een kostenpost. De rekening is de prijs van dit `hour`, uit `PRICE`, keer de eenheden stroom van de `action`, uit `ENERGY`.",
            "In het geval van de check gebruikt laag koelen om 10 uur 1 eenheid tegen de dagprijs van 4: een rekening van 4, dus een reward van −4. De ruimte gaat van 24 naar 25 graden, onder de grens, dus komt er geen straf bij.",
            "`    reward = ___`",
        ],
        why="`reward = -PRICE[hour] * ENERGY[action]` maakt van de rekening een kostenpost: hoe meer stroom, hoe lager de reward, en dezelfde eenheid kost overdag vier keer zoveel als 's nachts. Met de straf voor een uur boven 27 graden zijn deze paar regels alles wat de agent ooit leert over wat wij willen.",
    )
    return


@app.cell(hide_code=True)
def text_after_step():
    mo.md(r"""
    De afbeelding laat de thermostaat één dag door je simulator lopen. Met de getallen van de course kost deze dag 103: alleen de rekening, want de thermostaat laat de ruimte nooit boven 27 graden komen. Q-learning weet niets van het datacenter behalve wat `step` vertelt, dus zou een fout in de reward net zo trouw geleerd worden als de goede reward.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_act():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een veiligheidsregel

        Na de training kiest de agent met `int(np.argmax(q))`: de action met de hoogste van de drie Q's van de state, waarbij `q` een lijst van drie getallen is. `count` zegt hoe vaak de simulator deze state tijdens de training liet zien. Als dat nooit gebeurde, heeft de tabel niets over de state geleerd: de drie Q's zijn nog 0, en `np.argmax` kiest de eerste: action 0, de koeling uit. In zo'n state moet `act(q, count, temp, hour)` de thermostaat laten beslissen, met `thermostat(temp, hour)`. Vervang de `...` door de test voor een state die de simulator nooit liet zien.

        De check roept `act([0.0, 0.0, 0.0], 0, 26, 3)` aan: een state die nooit langskwam, 26 graden om 3 uur. Dan vijf andere gevallen: states die 40, 7 en 1 keer langskwamen, en twee states die nooit langskwamen, bij 33 en bij 15 graden.
        """),
        L.task("Laat de thermostaat beslissen in een state die de simulator nooit liet zien."),
    ])
    return


@app.cell(hide_code=True)
def exercise_act():
    box_act = L.code_box("act", lines=4)
    box_act
    return (box_act,)


@app.cell(hide_code=True)
def check_act(box_act):
    result_act = L.check("act", box_act.value)
    result_act
    return (result_act,)


@app.cell(hide_code=True)
def hints_act():
    L.hints(
        "act",
        [
            "Stap 5 van het recept: een regel om op terug te vallen. De `if` moet precies waar zijn als `count`, het aantal keer dat de simulator deze state liet zien, 0 is.",
            "In het eerste geval van de check kwam de state nooit langs, dus beslist de thermostaat: bij 26 graden koelt de thermostaat hoog, action 2. In een state die 40 keer langskwam, beslist de tabel.",
            "`    if ___:`",
        ],
        why="`if count == 0:` geeft elke state die de simulator nooit liet zien aan de thermostaat. Echte systemen doen iets vergelijkbaars: in 2018 viel de AI van Google voor de koeling van datacenters zo nodig terug op de eigen regels van het gebouw, en konden mensen het op elk moment overnemen.",
    )
    return


@app.cell(hide_code=True)
def text_after_act():
    mo.md(r"""
    De afbeelding zet de agent van de course in een warme nacht: de servers blijven druk, 2 graden per uur in plaats van 1, en zo'n nacht liet de simulator nooit zien. Zonder de regel laat de tabel, nog vol nullen in deze states, de koeling uit: na vier uur is de ruimte boven 27 graden, en tegen de tijd dat de ruimte in states komt waar de tabel iets van weet, is het er al 35 graden, en zijn de meeste uren te heet niet meer te vermijden. Zo wordt het 21 uur te heet. Met de regel grijpt de thermostaat in, tot de ruimte terug is in states die de tabel kent.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Trainen in de simulator, dan een dag in werkelijkheid

    De demo traint Q-learning 10.000 dagen in een simulator: je eigen `step` als je `step` de check haalt, en anders de `step` van de course, met η 0,1, ε 0,1 en γ 1, en de temperatuur en het uur als state. De grafiek toont de kosten van elke dag training, met gestreept de 103 van de thermostaat; ernaast wat de agent op dat moment zou doen. Na de training leeft de agent één dag in werkelijkheid: zoals in de simulator, met een warmere middag, vier uur één graad warmer, of met een warme nacht. Kies of je `act` als veiligheidsregel mag ingrijpen; als je `act` de check nog niet haalt, grijpt de `act` van de course in. Ga daarna terug door de training.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    reality_choice = mo.ui.dropdown(options=C.reality_options(L.lang), value=C.DEMO_TEXT[L.lang]["realities"]["sim"], label=C.DEMO_TEXT[L.lang]["reality"])
    safety_choice = mo.ui.dropdown(options=C.safety_options(L.lang), value=C.DEMO_TEXT[L.lang]["safeties"]["off"], label=C.DEMO_TEXT[L.lang]["safety"])
    train_button = mo.ui.run_button(label="Train in de simulator")
    mo.hstack([reality_choice, safety_choice, train_button], justify="start", gap=1.5, wrap=True)
    return reality_choice, safety_choice, train_button


@app.cell(hide_code=True)
def demo_train(box_act, box_step, reality_choice, result_act, result_step, safety_choice, train_button):
    if not train_button.value:
        _out = mo.md("*Kies de dag in werkelijkheid en de veiligheidsregel, en druk dan op de knop om te trainen.*")
    else:
        _mine = result_step.state == "correct"
        _step = run_code(box_step.value, C.S_NAMES).names["step"] if _mine else C.step
        _act_mine = result_act.state == "correct"
        _act = run_code(box_act.value, C.A_NAMES).names["act"] if _act_mine else C.act
        _reality, _safety = reality_choice.value, safety_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), C.DAYS, role="loss", low=0, high=200, window=100, xticks=C.demo_xticks(),
                       key=C.chart_key(L.lang))
        _live.reference(C.thermostat_line(), "", role="ink-2")
        _shown = [0]

        def _progress(_run):
            _live.watch(lambda: C.now_picture(_run, L.lang))
            _new = [(_d + 1, _run.costs[_d]) for _d in range(_shown[0], _run.days)]
            _shown[0] = _run.days
            _live.extend(_new, _w["learning"].format(d=C.number(_run.days, 0, L.lang), n=C.number(C.DAYS, 0, L.lang),
                                                     hot=C.number(sum(_run.hots), 0, L.lang)))

        _run = C.demo_run(_step, show=_progress, pause=0.02)
        if _run == "error":
            _out = _live.stopped("*Je `step` gaf een fout, of iets anders dan een temperatuur en een getal. Haalt je `step` de check nog?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "day", lambda _v: C.rewind_frame(_run, int(_v), L.lang), start=C.DAYS,
                          values=list(range(0, C.DAYS + 1, C.SNAP)), role="",
                          task="Ga terug naar dag 0, en dan vooruit: vanaf welke dag wordt de eigen dag van de agent niet meer te heet?"),
                mo.Html(C.test_picture(_run, _reality, _safety, L.lang, _act)),
                mo.md(C.verdict(_run, _reality, _safety, _mine, _act_mine, L.lang, _act)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Zoals in de simulator kost de agent 63, tegen 103 voor de thermostaat. Met de warmere middag gaat het nog goed, 79 tegen 119, omdat zulke states in de training langskwamen: door ε-greedy was de ruimte af en toe warmer dan gewoonlijk. In de warme nacht kost de agent zonder regel 214, met 21 uur te heet, terwijl de thermostaat op 110 blijft, zonder enig uur te heet; met de veiligheidsregel kost de agent 104, zonder uur te heet. Dat is het patroon van echte systemen: hoe meer de wereld van de simulator afwijkt, hoe meer een geleerde policy grenzen en een regel om op terug te vallen nodig heeft. En een count van 0 is het minste wat een regel moet opvangen: in andere trainingsruns, met een andere seed, misleiden states die slechts een paar keer langskwamen de agent nog steeds in de warme nacht.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Oefening 3. Waar leren

    Een magazijn wil een RL-agent de robots tussen de schappen laten besturen. Een botsing kost een robot.
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "learn_where",
        question="Hoe moet de agent leren?",
        options={
            "right": "Eerst in een simulator van het magazijn, dan op de vloer met een veiligheidsregel",
            "floor": "Meteen op de echte vloer: echte ervaring leert het best",
            "logs": "Alleen uit de logs van de robots die de agent vervangt: daarin staat elke botsing die de agent kan maken",
            "sim": "Alleen in de simulator: wat daar werkt, werkt ook op de vloer",
        },
        correct="right",
        why={
            "right": "Zo deden de tokamak en de datacenters het: de vele eerste fouten gebeuren waar een fout niets kost, en op de echte machine vangen grenzen en een regel om op terug te vallen op wat de simulator mis had, want een simulator is nooit precies de wereld.",
            "floor": "In de demo was de ruimte tijdens het leren meer dan 20.000 uur te heet. Wat zouden zoveel botsingen op een echte vloer kosten?",
            "logs": "Logs laten alleen zien wat de oude robots deden, en de oude robots botsten bijna nooit. Kan een agent leren wat een botsing kost uit data met bijna geen botsingen?",
            "sim": "In de demo was de warme nacht een situatie die de simulator nooit liet zien. Wat doet de agent op de vloer in een situatie die in de simulator ontbrak?",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_exercise_zero_shot():
    mo.md(r"""
    ### Oefening 4. Rechtstreeks uit de simulator

    Jonas Degrave en collega's (2022) trainden een agent in een simulator van de TCV-tokamak, en lieten de agent zonder verdere training de echte machine sturen.
    """)
    return


@app.cell(hide_code=True)
def choice_zero_shot_cell():
    choice_zero_shot = L.choice(
        "zero_shot",
        question="Wat had de simulator van de tokamak, en miste de simulator van de demo in de warme nacht?",
        options={
            "days": "Meer dagen training in dezelfde simulator",
            "right": "Een simulator die de states van de echte machine goed nabootst",
            "eta": "Een grotere learning rate η, zodat de agent sneller leert",
            "reward": "Een reward zonder straf, zodat fouten minder meetellen in wat de agent leert",
        },
        correct="right",
        why={
            "right": "De agent kent alleen de states waarin de agent oefende, dus moet de simulator elke state laten zien waar de echte machine komt, en zich daar gedragen zoals de machine. De simulator van de demo had geen warme nacht, en meer training daarin had niet geholpen: deze states komen er nooit langs.",
            "days": "In de warme nacht kwam de agent in states die de simulator nooit liet zien. Zouden meer dagen in dezelfde simulator deze states ooit laten zien?",
            "eta": "η bepaalt hoe groot elke update is. Helpt een grotere stap in een state waar de agent nooit kwam?",
            "reward": "Zonder de straf zouden de uren te heet de agent niets kunnen schelen. Zou het plasma dan wegblijven van de wand?",
        },
    )
    choice_zero_shot
    return (choice_zero_shot,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl) waren de environments simulators, gemaakt om te leren, waar een fout niets kostte. Buiten games komen dezelfde vragen terug, de state, de actions en de reward, met één erbij: wat kost een fout, en wat vangt een fout op? De ε-greedy exploration van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) maakt leren in de wereld juist duur, en dus leren echte systemen in simulators, uit vastgelegde data, of achter grenzen en regels om op terug te vallen. De volgende les, *Van checkers tot Go: mijlpalen van RL*, gaat terug naar de games, waar RL bekend door werd.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, choice_zero_shot, result_act, result_step):
    L.wrap_up(result_step, result_act, choice_where, choice_zero_shot)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
