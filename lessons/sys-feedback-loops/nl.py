# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-feedback-loops. Route: course/route/lessen/systemen.yaml. Checks, de routes, de eerste week, de runs en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-feedback-loops", "nl")
    C = L.checks   # de acht routes en de kans van elke route, de eerste week van de course, de runs van de app en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="De fietsservice logt elk verzoek en elk antwoord van het model. Welke drift kan de service pas zien als bekend is wat er echt gebeurde?",
        options={
            "data": "Data drift: een verandering in de inputs, zoals de temperatuur",
            "pred": "Prediction drift: een verandering in de antwoorden van het model",
            "right": "Label drift: een verandering in het aantal verhuurde fietsen",
            "none": "Geen van de drie: de service ziet alle drie bij elk verzoek",
        },
        correct="right",
        why={
            "right": "De verhuurde fietsen komen pas binnen als het uur voorbij is; de inputs en de antwoorden staan meteen in de log. In deze les komt alleen binnen wat er echt gebeurde bij de keuze van het model: een bezoeker kan alleen een route nemen die de app toont.",
            "data": "De inputs zitten in elk verzoek, dus staan de inputs meteen in de log. Welke waarden komen pas later binnen? Zie [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).",
            "pred": "De service logt elk antwoord op het moment van antwoorden. Welke waarden komen pas later binnen? Zie [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).",
            "none": "De inputs en de antwoorden staan meteen in de log, en de derde soort niet. Welke waarden komen pas later binnen? Zie [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl).",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_explore():
    quick_explore = L.choice(
        "quick-explore",
        question="Een agent met ε-greedy en $\\varepsilon$ = 0,1 neemt in een tiende van de stappen een willekeurige action. Waarom neemt de agent niet altijd de action met de grootste $Q$?",
        options={
            "faster": "Willekeurige actions kosten minder rekenwerk, dus haalt de agent meer episodes per seconde",
            "right": "De Q-waarden komen alleen van actions die de agent nam: de andere blijven onbekend",
            "slip": "De wereld laat de agent soms uitglijden, en willekeurige actions bootsen dat na",
            "forget": "Zonder willekeurige actions zou de Q-table vergeten wat de agent eerder leerde",
        },
        correct="right",
        why={
            "right": "De $Q$ van een action verandert alleen als de agent de action neemt. Zonder willekeurige actions wordt een action die vroeg slecht leek nooit meer geprobeerd, ook als juist deze action de beste is. De app van deze les heeft hetzelfde probleem: de app leert alleen over de routes die de app toont.",
            "faster": "Een willekeurige action duurt even lang als elke andere, en de agent neemt willekeurige actions om een andere reden. Wat zou de agent zonder willekeurige actions nooit te weten komen? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "slip": "Het uitglijden komt van de wereld, wat de agent ook kiest. Wat zou de agent zonder willekeurige actions nooit te weten komen? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "forget": "Een $Q$ verandert alleen als de agent de action neemt, dus wordt er zonder willekeurige actions niets vergeten. Wat wordt er dan nooit geleerd? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_explore
    return (quick_explore,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model in productie geeft niet alleen antwoord: wat het model antwoordt bepaalt vaak wat er daarna gebeurt, en wat er daarna gebeurt wordt de data van het model. Een winkel die een product bovenaan zet verkoopt meer van dat product; een model dat van deze verkopen leert vindt het product dan nog beter, en zet het product nog vaker bovenaan. Wat het model geeft komt terug in de eigen data. Zo'n loop kan een model vastzetten op een keuze die alleen door toeval een beetje voorlag, en de data laten nooit zien wat de andere keuzes gedaan zouden hebben. Een vergelijking met wat er echt gebeurde vangt het probleem ook niet: wat er echt gebeurde komt alleen binnen voor de keuze van het model.

    **Een voorbeeld uit het dagelijks leven.** Een pad over een grasveld begint met een paar mensen die een hoek afsnijden. De stappen drukken het gras plat, dus nemen de volgende mensen dezelfde lijn, en het pad wordt elke week duidelijker. Niemand komt te weten of een andere lijn korter was geweest: iedereen loopt over het uitgesleten pad, omdat iedereen daar loopt.
    """)
    return


@app.cell(hide_code=True)
def predict_loop():
    guess_loop = L.guess(
        "loop",
        question="De app van de fietsservice stelt op het startscherm een van acht routes voor. In een eerste week toont de app elke bezoeker een willekeurige route; daarna altijd de route met tot nu toe de meeste ritten. Route E is de route die het vaakst genomen wordt als de app route E toont. De course liet deze app draaien op 1.000 verschillende eerste weken, elk gevolgd door 20.000 bezoekers. In hoeveel daarvan toont de app uiteindelijk route E?",
        options={
            "all": "In bijna alle: de beste route krijgt uiteindelijk de meeste ritten",
            "right": "In ongeveer de helft: welke route er ook voorlag na de eerste week",
            "most": "In ongeveer negen van de tien: een eerste week zet zelden een slechtere route voorop",
            "few": "In bijna geen: route E heeft te weinig ritten om ooit voor te komen",
        },
        reveal="**In ongeveer de helft: 549 van de 1.000.** In elke run wordt de route met de meeste ritten na de eerste week aan elke latere bezoeker getoond, dus krijgt deze route elke latere rit en blijft de route voorgoed bovenaan. Route E lag na de eerste week voor in 549 van de 1.000 weken, en precies in deze weken eindigde de app op E. In de andere 451 had een route die minder bezoekers nemen een gelukkige week, meestal route C, en hield deze route het scherm.",
    )
    guess_loop
    return (guess_loop,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De app.** De app van de fietsservice stelt op het startscherm een van acht routes voor, A tot en met H. Een bezoeker die een route ziet neemt de route of niet: route E wordt genomen door 32 van elke 100 bezoekers met route E op het scherm, route F door 4 van de 100. De app kent deze getallen niet; de app weet alleen hoe vaak elke route getoond werd en hoeveel ritten elke route kreeg. In de eerste week toonde de app een willekeurige route aan elk van 400 bezoekers. Vanaf dan toont de app de route met tot nu toe de meeste ritten.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Feedback loop.** Een *feedback loop* is een loop waarin wat een model geeft de data verandert waarvan het model later leert. Hier bepaalt de keuze van de app welke route gezien wordt, en alleen een route die gezien wordt kan ritten krijgen.

        **Degenerate feedback loop.** In een **degenerate feedback loop** maakt de loop steeds kleiner wat het model toont, tot het model vrijwel alleen nog één ding toont. Na de eerste week lag route G voor met 15 ritten tegen 13 voor route E. Dus wordt G aan elke bezoeker getoond, krijgt G elke rit, en loopt G met elke rit verder uit; E wordt nooit meer getoond, en de 13 ritten van E blijven 13. De tellingen van ritten zeggen dat G veruit de beste route is, en zeggen dat omdat de app de tellingen zo maakte.

        **Popularity bias.** **Popularity bias** is de gewoonte van een model om wat al populair is steeds vaker te tonen, omdat het populaire de meeste klikken of ritten krijgt, of het nu het beste is of niet. Een telling van ritten meet net zo goed hoe vaak een route getoond werd als hoe goed de route is.

        **Ritten per keer getoond.** Een betere maat deelt de ritten van elke route door het aantal keer dat de route getoond werd: route E kreeg 13 ritten en werd 44 keer getoond, 0,30 per keer getoond, en route G kreeg 15 ritten bij 53 keer, 0,28. Alleen vaker getoond worden laat een route in deze rangschikking niet stijgen.

        **Exploration.** Ook een rangschikking op ritten per keer getoond kan vastlopen: een route met een slechte eerste week wordt nooit meer getoond, dus krijgt het getal van deze route nooit de kans om te stijgen. Net als bij Q-learning houdt een aandeel $\varepsilon$ willekeurige routes de app aan het leren over elke route: exploration. Elke willekeurige route kost ritten, want de meeste routes zijn slechter dan de beste; dat is de prijs van het uitzoeken.

        **De spreiding meten.** Tel hoe vaak het model elke keuze toonde, in de loop van de tijd. In een degenerate feedback loop toont het model steeds minder verschillende dingen, en kruipt het aandeel van de bovenste keuze naar 1.

        **In de praktijk.** Teams met zulke systemen loggen wat er getoond werd naast wat er geklikt of gereden werd, zodat elke telling later door het aantal keer getoond te delen is. Veel teams houden een klein deel van het verkeer apart waarin een willekeurige keuze getoond wordt, om te meten wat de andere keuzes zouden doen. Chip Huyen beschrijft degenerate feedback loops, en willekeurige keuzes als middel ertegen, in *Designing Machine Learning Systems*, in het hoofdstuk over monitoring.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | feedback loop | | een loop waarin wat een model geeft de data verandert waarvan het model later leert | |
        | **degenerate feedback loop** | | een feedback loop die steeds kleiner maakt wat een model toont, tot het model vrijwel alleen nog één ding toont | |
        | **popularity bias** | | de gewoonte van een model om wat al populair is steeds vaker te tonen, omdat het populaire de meeste klikken of ritten krijgt | |
        | ritten per keer getoond | | de ritten van een route gedeeld door het aantal keer dat de route getoond werd | `rides / shown` |
        | keer getoond | | één keer dat de app een route op het scherm van een bezoeker zette | `shown[route]` |
        """),
        L.watch_out("Meer data lossen een degenerate feedback loop niet op. Na 20.000 bezoekers heeft route G 4.858 ritten en route E nog steeds 13: de data groeien alleen waar de app kijkt."),
        L.watch_out("Een test set uit de logs bevat de loop ook: na de eerste week bevat de test set geen enkele keer dat route E getoond werd, dus kan de test set nooit laten zien dat E beter was geweest."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een feedback loop vinden en doorbreken:

    1. Vraag of wat het model geeft bepaalt welke data er terugkomen: wat er getoond wordt, wie een lening krijgt, waar de fietsen heen gaan.
    2. Log wat het model toonde naast wat er terugkwam, zodat elke telling te delen is door het aantal keer getoond.
    3. Rangschik op ritten per keer getoond, niet op ritten.
    4. Toon een klein aandeel $\varepsilon$ van de bezoekers een willekeurige keuze, en houd dat aandeel.
    5. Houd de spreiding in de gaten: hoe vaak elke keuze getoond wordt, in de loop van de tijd.
    6. Weeg de kosten: elke willekeurige keuze kost nu een beetje, en vertelt de app wat de andere keuzes zouden doen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.ROUTES` is de string `"ABCDEFGH"`, de namen van de acht routes. `C.SHOWN` bevat hoe vaak de eerste week van de course elke route toonde, en `C.RIDES` de ritten van elke route. `C.APPEAL` bevat de kans dat een bezoeker die een route ziet de route neemt; de app kent deze kansen niet. De cel print het aantal keer getoond, de ritten, de ritten per keer getoond afgerond op twee decimalen, en de route met de meeste ritten. Voer de cel uit.
    """)
    return


@app.cell
def example_week():
    import numpy as np

    print(C.SHOWN)
    print(C.RIDES)
    print(np.round(C.RIDES / C.SHOWN, 2))   # ritten per keer getoond
    print(C.ROUTES[np.argmax(C.RIDES)])     # de route met de meeste ritten
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Route G kreeg de meeste ritten, 15, en route E 13. Maar E werd 44 keer getoond en G 53 keer: per keer getoond heeft E 0,30 en G 0,28. De app die ritten telt toont vanaf nu G, terwijl de eerste week al naar E wijst. In Oefening 1 laat je deze app draaien.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_simulate():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De app die ritten telt

        `simulate(appeal, rides, shown, visitors, seed)` laat de app na de eerste week `visitors` bezoekers bedienen. `rides` en `shown` bevatten de ritten en het aantal keer getoond van elke route tot nu toe; `appeal` bevat de kans van elke route, en alleen de simulatie kent deze kansen. `rng = np.random.default_rng(seed)` maakt de random number generator. Voor elke bezoeker `v` in `range(visitors)` kiest de app `route = np.argmax(rides)`, de route met tot nu toe de meeste ritten, en telt 1 op bij `shown[route]`. Dan neemt de bezoeker de route met kans `appeal[route]`, en zo ja, dan krijgt de route een rit. De functie geeft `rides` terug. Vervang de `...` door een test die waar is met kans `appeal[route]`.

        De check roept `simulate(C.APPEAL, C.RIDES, C.SHOWN, 20000, 1)` aan: de eerste week van de course, dan 20.000 bezoekers. Dan 3.000 bezoekers met een andere seed, twee routes waarvan de route die voorligt nooit genomen wordt, en drie andere routes.
        """),
        L.task("Laat de bezoeker de getoonde route nemen met kans `appeal[route]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_simulate():
    box_simulate = L.code_box("simulate", lines=8)
    box_simulate
    return (box_simulate,)


@app.cell(hide_code=True)
def check_simulate(box_simulate):
    result_simulate = L.check("simulate", box_simulate.value)
    result_simulate
    return (result_simulate,)


@app.cell(hide_code=True)
def hints_simulate():
    L.hints(
        "simulate",
        [
            "`route` is de route die de app toont, en `appeal[route]` de kans van deze route. Net als in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) geeft `rng.random()` een float vanaf 0 tot 1, waarbij 1 zelf nooit voorkomt.",
            "Een willekeurige float vanaf 0 tot 1 ligt in ongeveer 30 van de 100 trekkingen onder 0,3. Welke vergelijking is waar met kans `appeal[route]`?",
            "`        if rng.random() ___ appeal[route]:`",
        ],
        why="`if rng.random() < appeal[route]:`: de willekeurige float ligt bij dat aandeel van de bezoekers onder de kans van de route, dus wordt route G, met 0,24, door ongeveer 24 van elke 100 genomen. Na de eerste week wordt G alle 20.000 keer getoond en krijgt G 4.843 ritten; route E wordt nooit meer getoond.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Ritten per keer getoond, en een paar willekeurige routes

        `choose(rides, shown, epsilon, rng)` geeft de route terug die de app aan één bezoeker toont. De functie geeft met kans `epsilon` een willekeurige route, met `rng.integers(0, len(rides))`: dat gebeurt als `rng.random() < epsilon` waar is, zoals bij ε-greedy van Q-learning. Vervang de `...` door de route met de meeste ritten per keer getoond.

        De check roept `choose(C.RIDES, C.SHOWN, 0.0, rng)` 200 keer aan met één random number generator, en telt hoe vaak de functie elke route geeft. Dan `C.RIDES_AFTER` en `C.SHOWN_AFTER`, de ritten en het aantal keer getoond nadat de app van Oefening 1 de 20.000 bezoekers bediende, dan twee en drie routes, en `epsilon` gelijk aan `0.1` en `1.0`.
        """),
        L.task("Laat `choose` buiten de willekeurige routes de route met de meeste ritten per keer getoond teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=4)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose",
        [
            "Stap 3 van het recept: rangschik op ritten per keer getoond, niet op ritten. `np.argmax` geeft de plek van de grootste waarde van een array, en hier is een plek een route.",
            "In het uitgewerkte voorbeeld gaf `C.RIDES / C.SHOWN` de ritten per keer getoond van elke route. Welke twee arrays van `choose` bevatten de ritten en het aantal keer getoond?",
            "`    return np.argmax(___)`",
        ],
        why="`return np.argmax(rides / shown)`: de ritten van elke route gedeeld door het aantal keer getoond, en de plek van de grootste. Na de eerste week is dat route E, met 0,30 per keer getoond. Met 10% willekeurige routes toont de app E aan 18.024 van de 20.000 bezoekers en krijgt de app 30,5 ritten per 100 bezoekers, tegen 24,2 voor de app van Oefening 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_count_explore():
    mo.md(r"""
    ### Oefening 3. Willekeurige routes voor de app die telt

    De app van Oefening 1 rangschikt op ritten. Een team houdt deze rangschikking, en voegt alleen 10% willekeurige routes toe, zoals in `choose`. Over de 1.000 eerste weken van de course eindigt de app zonder willekeurige routes in 549 ervan op route E.
    """)
    return


@app.cell(hide_code=True)
def choice_count_explore_cell():
    choice_count_explore = L.choice(
        "count-explore",
        question="In hoeveel van de 1.000 eindigt de app op route E met 10% willekeurige routes, als de app nog steeds op ritten rangschikt?",
        options={
            "most": "Bijna altijd: de willekeurige routes tonen E vaak genoeg om de ritten van E te laten inlopen",
            "right": "Ongeveer even vaak: de koploper krijgt nog 9 van de 10 bezoekers, en blijft ver voor",
            "more": "In ongeveer 700: met elke willekeurige route krijgt route E een kleine kans om voor te komen",
            "never": "Nooit: willekeurige routes halen ook bij route E bezoekers weg, waar E ook staat",
        },
        correct="right",
        why={
            "right": "552 van de 1.000, tegen 549 zonder. Elke route komt ongeveer eens per 80 bezoekers willekeurig op het scherm, dus wordt route E over 20.000 bezoekers ongeveer 250 keer getoond en krijgt E 80 ritten, terwijl de koploper ongeveer 18.000 keer getoond wordt, en duizenden ritten krijgt. Willekeurige routes helpen alleen een app die op ritten per keer getoond rangschikt; in *Laat de app draaien* kun je allebei proberen.",
            "most": "Route E wordt over 20.000 bezoekers ongeveer 250 keer willekeurig getoond, en de koploper ongeveer 18.000 keer. Kunnen de ritten van E dat inhalen?",
            "more": "Elke keer dat route E willekeurig getoond wordt krijgt E een rit met kans 0,32; route E komt ongeveer eens per 80 bezoekers op het scherm, de koploper bij 9 van de 10. Van welke route groeien de ritten sneller?",
            "never": "Willekeurige routes zetten route E op het scherm, ook als de app route E anders niet toont. Welke route wordt nog steeds het vaakst getoond, en krijgt zo de meeste ritten?",
        },
    )
    choice_count_explore
    return (choice_count_explore,)


@app.cell(hide_code=True)
def text_exercise_loops():
    mo.md(r"""
    ### Oefening 4. Loop of geen loop?

    Zes modellen aan het werk. Zet elk model bij wel of geen feedback loop.
    """)
    return


@app.cell(hide_code=True)
def sort_loops_cell():
    sort_loops = L.sort("loops")
    sort_loops
    return (sort_loops,)


@app.cell(hide_code=True)
def check_loops(sort_loops):
    result_loops = L.check("loops", sort_loops)
    result_loops
    return (result_loops,)


@app.cell(hide_code=True)
def hints_loops():
    L.hints(
        "loops",
        [
            "Vraag bij elk model: verandert wat het model geeft wat er daarna gebeurt, en komt dat in de data van het model terecht?",
            "Drie van de zes beslissen wat mensen zien of krijgen; de andere drie bekijken alleen wat hoe dan ook gebeurt.",
            "Loop: bestsellers bovenaan, \\_\\_\\_, \\_\\_\\_. Geen loop: een regenvoorspelling, \\_\\_\\_, \\_\\_\\_.",
        ],
        why="**Loop:** bestsellers bovenaan, fietsen waar het model ritten verwacht, en leningen voor sommigen: wat het model geeft bepaalt welke verkopen, ritten of terugbetalingen er terugkomen. **Geen loop:** een regenvoorspelling, de aankomst van een trein en kapotte onderdelen: het model kijkt alleen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Laat de app draaien

    Kies hoe de app de routes rangschikt en hoeveel willekeurige routes de app toont, en druk op de knop. De app bedient na de eerste week van de course 20.000 bezoekers. De grafiek toont de ritten per 100 bezoekers over de laatste 1.000 bezoekers; de lijn met streepjes staat op 32, wat de app zou krijgen als de app elke keer route E toonde. Ernaast zie je hoe vaak de app elke route toonde sinds de eerste week. Rangschikken op ritten per keer getoond gebruikt je `choose` zodra je functie de check haalt, en tot dan de `choose` van de course. Het duurt een paar seconden; daarna kun je terug door de run.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_rank = mo.ui.dropdown(
        options={"de meeste ritten": "rides", "de meeste ritten per keer getoond": "rate"},
        value="de meeste ritten",
        label="Rangschik op",
    )
    demo_eps = mo.ui.dropdown(
        options={"0%": 0.0, "5%": 0.05, "10%": 0.1, "20%": 0.2},
        value="0%",
        label="Willekeurige routes",
    )
    run_button = mo.ui.run_button(label="Laat de app draaien")
    mo.vstack([
        mo.hstack([demo_rank, demo_eps, run_button], justify="start", wrap=True),
        L.task("Begin met de meeste ritten en 0%: de app van Oefening 1. Rangschik dan op ritten per keer getoond, met 0% en met 10%. Welke route toont elke app aan het eind het vaakst, en hoe ver blijft de app onder de lijn met streepjes?"),
    ])
    return demo_eps, demo_rank, run_button


@app.cell(hide_code=True)
def demo_run(box_choose, demo_eps, demo_rank, result_choose, run_button):
    if not run_button.value:
        _out = mo.md("*Druk op de knop om de app te laten draaien.*")
    else:
        _yours = C.mine(box_choose.value, result_choose.state == "correct")
        _w = C.live_words(L.lang)
        _now = {}   # de live grafiek, en hoe vaak de app elke route toonde na de laatste bezoeker tot nu toe

        def _progress(_t, _per_100, _after):
            if "after" not in _now:   # vanaf het eerste punt het aantal keer getoond naast de grafiek
                _now["chart"].watch(lambda: C.now_picture(_now["after"], _now["t"], L.lang))
            _now["after"], _now["t"] = _after, _t
            _now["chart"].add(_t, _per_100, C.progress_line(_t, demo_rank.value, demo_eps.value, L.lang))

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                               C.VISITORS, role="ink", low=0, high=40, window=1, xticks=C.TICKS)
        _now["chart"].reference([(0, C.BEST_PER_100), (C.VISITORS, C.BEST_PER_100)], _w["best"], role="ink-2")
        _now["chart"].line("**De app draait** …")
        _run = C.demo(demo_rank.value, demo_eps.value, _yours, show=_progress)
        if _run == "choose":
            _out = _now["chart"].stopped("*Je `choose` gaf tijdens de run een fout, of geen route. Haalt je functie de check nog?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*De run duurde langer dan twee minuten. Doet je `choose` meer werk dan nodig?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "bezoeker", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.VISITORS, values=C.LOOKS, role="",
                          task="Ga terug naar bezoeker 200, en dan weer vooruit: wanneer blijft de app bij één route, en bij welke?"),
                mo.md(C.result_text(_run, demo_rank.value, demo_eps.value, _yours is not None, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De course liet elk van de acht apps draaien op de 1.000 eerste weken, elk gevolgd door 20.000 bezoekers:

    | Rangschik op | Willekeurige routes | Eindigt op route E | Ritten per 100 bezoekers |
    |---|---|---|---|
    | de meeste ritten | 0% | 549 | 29,3 |
    | de meeste ritten | 5% | 550 | 28,7 |
    | de meeste ritten | 10% | 552 | 28,2 |
    | de meeste ritten | 20% | 548 | 27,0 |
    | de meeste ritten per keer getoond | 0% | 780 | 31,0 |
    | de meeste ritten per keer getoond | 5% | 928 | 30,8 |
    | de meeste ritten per keer getoond | 10% | 970 | 30,3 |
    | de meeste ritten per keer getoond | 20% | 990 | 29,0 |

    Bij rangschikken op ritten veranderen willekeurige routes bijna niets: de koploper na de eerste week houdt het scherm. Rangschikken op ritten per keer getoond doet het meeste werk, met route E aan het eind in 780 van de 1.000 weken; in 201 van de andere 220 eindigt de app op route C, en route C wordt door 28 van de 100 bezoekers genomen. Willekeurige routes brengen de 780 naar 970 met 10% en 990 met 20%, maar elke willekeurige route kost nu ritten: 31,0 per 100 bezoekers zonder, 29,0 met 20%. In deze wereld verandert niets tijdens de 20.000 bezoekers. Een team houdt toch een klein willekeurig aandeel: zonder dat aandeel wordt een route die onder de koploper zakt nooit meer getoond, dus kan de app niet merken dat zo'n route beter wordt, bijvoorbeeld doordat een nieuwe brug route C korter maakt.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl) bewoog de wereld vanzelf, en de echte ritten lieten dat zien. Hier verandert het model de eigen data, en komt alleen binnen wat er echt gebeurde bij de keuze van het model. De willekeurige routes zijn de ε-greedy van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl), en elke bezoeker is een trekking van een random number generator, zoals in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl): met dezelfde seed dezelfde bezoekers. In de volgende les, *Project: je model als service, bewaakt op drift*, controleert je service de data van 2012 maand voor maand.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count_explore, result_choose, result_loops, result_simulate):
    L.wrap_up(result_simulate, result_choose, choice_count_explore, result_loops)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
