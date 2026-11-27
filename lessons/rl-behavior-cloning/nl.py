# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-behavior-cloning. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de brug, de expert, de klonen van de course en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-behavior-cloning", "nl")
    C = L.checks   # de brug, de expert, de klonen van de course en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) en [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_knn():
    quick_knn = L.choice(
        "quick-knn",
        question="Drie voorbeelden uit de training set: (0, 0) met klasse A, (0, 3) met klasse B en (4, 0) met klasse B. Welke klasse geeft k-NN met k = 1 aan het nieuwe voorbeeld (1, 1)?",
        options={
            "majority": "B, omdat twee van de drie voorbeelden uit de training set klasse B hebben",
            "right": "A, de klasse van het dichtstbijzijnde voorbeeld",
            "mean": "Halverwege tussen A en B",
            "tie": "Geen, want één buur is te weinig voor een stemming",
        },
        correct="right",
        why={
            "right": "De afstanden vanaf (1, 1) zijn $\\sqrt{2} \\approx 1{,}41$ tot (0, 0), $\\sqrt{5} \\approx 2{,}24$ tot (0, 3) en $\\sqrt{10} \\approx 3{,}16$ tot (4, 0). Met k = 1 beslist het dichtstbijzijnde voorbeeld alleen: A. Met k = 3 zou B winnen. In deze les doet k-NN met k = 1 een expert na.",
            "majority": "Met k = 3 stemmen alle drie, en wint B: hoeveel buren stemmen er met k = 1? Zie [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "mean": "k-NN geeft een klasse die de buren hebben, niet iets tussen twee klassen in: welke buur beslist hier? Zie [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "tie": "Met k = 1 heeft de stemming één stemmer, het dichtstbijzijnde voorbeeld, en dit voorbeeld beslist alleen: welk voorbeeld ligt het dichtst bij (1, 1)? Zie [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
        },
    )
    quick_knn
    return (quick_knn,)


@app.cell(hide_code=True)
def quick_check_slip():
    quick_slip = L.choice(
        "quick-slip",
        question="Op een grid gaat een action met kans 0,8 waar de action heen wijst, en glijdt de stap met kans 0,1 uit naar elke kant, zoals bij value iteration. Een agent staat in de rij vlak boven een rij putten en gaat naar rechts. Hoe groot is de kans dat deze stap in een put eindigt?",
        options={
            "zero": "0, want rechts gaat niet richting de putten",
            "both": "0,2, de kans dat de stap überhaupt uitglijdt",
            "right": "0,1, het uitglijden naar de kant van de putten",
            "half": "0,05, want het uitglijden van 0,1 kan naar twee kanten, en slechts één kant heeft de putten",
        },
        correct="right",
        why={
            "right": "Als rechts uitglijdt, gaat de stap omhoog of omlaag, elk met 0,1, en omlaag ligt de put. Op de brug van deze les gaat de expert in deze rij omhoog, terug naar het midden: dan glijdt een stap uit naar links of rechts, langs de putten, nooit erin.",
            "zero": "Rechts gaat niet richting de putten, maar uitglijden gaat naar een kant van de action: welke kanten heeft rechts? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "both": "0,2 is de kans op uitglijden naar een van beide kanten, en slechts één kant heeft de putten: welke? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "half": "De 0,1 is al de kans voor één kant, en de andere kant heeft een eigen 0,1: hoe groot is de kans voor de kant van de putten? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
        },
    )
    quick_slip
    return (quick_slip,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe leerden je agents van rewards, door te proberen, honderden episodes lang, en betaalden je agents onderweg voor elke fout. Vaak is er al iemand die de taak kan: een chauffeur, een piloot, iemand die een robotarm met de hand bestuurt. Dan kun je het proberen overslaan: neem op wat de expert doet, en leer uit deze opnames een policy, zoals een classifier leert uit voorbeelden. Zo leerde ALVINN, een van de eerste auto's die zelf stuurden, rond 1990 aan Carnegie Mellon University: uit camerabeelden van de weg, elk met de stuurhoek van de menselijke bestuurder. Eén probleem dook meteen op. Een goede bestuurder houdt de auto in het midden van de rijstrook, dus in de opnames zaten bijna geen beelden van een auto die naar de kant was afgedreven, en zodra ALVINN afdreef, had niemand de weg terug laten zien. Dean Pomerleau loste het op met extra beelden, opzij geschoven, elk met de stuurhoek die de auto terugbrengt. Ross en Bagnell lieten in 2010 zien dat dit geen pech is: in het ergste geval groeien de kosten van nadoen met het kwadraat van de lengte van de taak. Deze les laat zien waarom.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe bestuurder kijkt alleen mee terwijl een instructeur rijdt, en ziet zo hoe je in het midden van de rijstrook rijdt, maar niet hoe je terugkomt van de rand van de weg: de instructeur komt daar nooit. Drijft de nieuwe bestuurder voor het eerst naar de rand af, dan is dat een situatie die niemand liet zien, en een verkeerde beweging daar brengt de auto nog verder van alles wat de instructeur deed.
    """)
    return


@app.cell(hide_code=True)
def predict_falls():
    guess_falls = L.guess(
        "falls",
        question="De brug in de afbeelding is drie vakjes breed, tussen twee rijen putten. Een stap gaat met kans 0,8 waar de stap heen wijst, en glijdt met kans 0,1 uit naar elke kant, zoals bij value iteration. De expert is de beste policy uit value iteration: in de middelste rij gaat de expert naar rechts, en in de twee buitenste rijen stapt de expert terug naar het midden, dus uitglijden brengt de expert nooit in een put. De expert loopt één keer van S naar G, en k-NN met k = 1 leert uit deze ene demonstratie een kloon: het vakje is het voorbeeld, de action van de expert de klasse. In hoeveel episodes valt de kloon in een put, denk je, gemiddeld over 20 van zulke demonstraties, elk met een eigen seed?",
        options={
            "never": "Bijna nooit: in elk vakje dat de expert liet zien, doet de kloon wat de expert deed",
            "right": "In ongeveer 2 van de 5 episodes",
            "rare": "In ongeveer 1 van de 20 episodes",
            "always": "In bijna elke episode",
        },
        reveal="**In ongeveer 2 van de 5 episodes.** Over de 20 demonstraties vallen de klonen in 43,8% van de episodes, terwijl de expert in geen van de 2.000 episodes valt. Waar de expert stond, doet een kloon precies wat de expert deed. Eén demonstratie laat echter slechts ongeveer 15 van de 35 vakjes zien, en uitglijden brengt de kloon al snel op een vakje dat niemand de kloon liet zien.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_falls])
    return (guess_falls,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Imitation learning.** Leert een agent van een expert in plaats van van rewards, dan heet dat **imitation learning**. De eenvoudigste manier is **behavior cloning**: neem op wat de expert doet, en behandel het als een taak voor een classifier, met supervised learning. Elke stap van de expert geeft één voorbeeld: de state is het voorbeeld, met de features, en de action van de expert is de klasse. Een classifier leert van deze voorbeelden, en de voorspelling in een state wordt de action van de agent in deze state: de agent is een *kloon* van de expert. De opgenomen episodes van de expert zijn de **demonstraties**. De kloon ziet nooit een reward.

        **De brug.** Een brug van drie rijen en twaalf kolommen, met een rij putten erboven en eronder. De agent begint op S, in de middelste rij links, en de episode eindigt op G, rechts, met een reward van +1. Een stap in een put geeft −1 en beëindigt de episode ook; elke andere stap geeft 0. Net als bij value iteration gaat een action met kans 0,8 waar de action heen wijst, en glijdt de stap met kans 0,1 uit naar elke kant; na een stap tegen de rand van de brug blijft de agent staan. In code is een state (rij, kolom), geteld vanaf linksboven, met de putten in rij 0 en 4: S is (2, 0) en G is (2, 11). `CELLS` is de lijst met de 35 vakjes waar de agent kan staan.

        **De expert.** De expert is de beste policy uit value iteration, met $\gamma$ = 0,9: in code `EXPERT`, een dict met een action voor elk vakje van `CELLS`. In de middelste rij gaat de expert naar rechts; in de twee buitenste rijen stapt de expert terug naar het midden, dus glijdt een stap uit naar links of rechts, langs de putten, nooit erin. De expert valt nooit: geen enkele keer in 2.000 episodes, van gemiddeld 17 stappen.

        **De kloon.** k-NN, zoals in k-nearest neighbors, met het vakje als features, de rij en de kolom, en de action van de expert als klasse. Met k = 1 doet de kloon in elk vakje wat de expert deed in het dichtstbijzijnde vakje waar de expert stond.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        In de afbeelding seed 0. Boven staat de ene demonstratie van de expert: 21 stappen op 17 van de 35 vakjes, licht ingekleurd. Vier keer brengt uitglijden de expert naar een buitenste rij, en elke keer gaat de expert terug naar het midden. Onder staat de kloon van deze demonstratie. Op elk ingekleurd vakje doet de kloon wat de expert deed; in 16 van de andere 18 vakjes doet de kloon iets anders, omlijnd: in de buitenste rijen neemt de kloon meestal het "rechts" van de middelste rij over, en loopt de kloon langs de putten. De lijn is de eerste episode van de kloon. In (3, 11), vlak onder G, zou de expert omhoog op G stappen; de kloon gaat naar rechts, tegen de rand, blijft staan, en bij de 21e stap glijdt de kloon uit, de put in.

        **Compounding errors.** Op de vakjes van de demonstratie maakt een kloon met k = 1 helemaal geen fouten. Maar de kloon blijft niet op deze vakjes. Uitglijden brengt de kloon op een vakje waar de expert nooit stond, waar de kloon iets anders kan doen dan de expert, en met dat andere, zoals langs de putten lopen in plaats van terugstappen, blijft de kloon op vakjes die de expert nooit liet zien. Over de 20 klonen van één demonstratie is gemiddeld 32,4% van de stappen op de vakjes waar de klonen zelf staan een fout, en na een fout is ook de volgende stap een fout in 83,6% van de gevallen. Elke val komt vlak na een fout: geen enkele action van de expert zelf kan in een put uitglijden. Zulke fouten heten **compounding errors**: fouten die de agent naar states brengen waar de agent nog meer fouten maakt. De kloon leerde op de vakjes van de expert en wordt getest op eigen vakjes: een **distribution shift**, zoals wanneer een model data tegenkomt die anders is dan de training set.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **imitation learning** | | een policy leren van een expert in plaats van van rewards | |
        | **demonstratie** | | een episode van de expert, opgenomen als paren van een state en de action van de expert in deze state | `demonstrate` |
        | **behavior cloning** | | imitation learning als supervised learning: de state is het voorbeeld, de action van de expert de klasse | `clone` |
        | **compounding errors** | | fouten die de agent naar states brengen waar de expert nooit was, waar de agent nog meer fouten maakt | `mistakes` |
        """),
        L.watch_out("De kloon weet niet dat een put slecht is: de kloon zag nooit een reward, alleen wat de expert deed. Waar de expert nooit stond, heeft de kloon niets anders dan het dichtstbijzijnde vakje waar de expert wel stond."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Behavior cloning, met k-NN:

    1. Laat de expert episodes spelen, en schrijf elke stap op als een paar: het vakje waar de expert staat, en de action die de expert daar neemt. Dit zijn de demonstraties.
    2. Zet de vakjes van de paren in `X`, de features, en de actions in `y`, de klassen.
    3. Laat k-NN voor elk vakje van de brug de action voorspellen uit `X` en `y`: dat is de policy van de kloon.
    4. Laat de kloon spelen, en tel waar de kloon iets anders doet dan de expert: op de vakjes van de demonstraties, en op de vakjes waar de kloon zelf staat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De kloon van seed 0, in vakje (1, 2): de rij boven het midden, waar de expert nooit stond. De dichtstbijzijnde vakjes van de demonstratie:

    | Vakje van de demonstratie | Action van de expert daar | Afstand tot (1, 2) |
    |---|---|---|
    | (2, 2) | rechts | $\sqrt{1^2 + 0^2} = 1$ |
    | (2, 1) | rechts | $\sqrt{1^2 + 1^2} \approx 1{,}41$ |
    | (2, 3) | rechts | $\sqrt{1^2 + 1^2} \approx 1{,}41$ |
    | (1, 4) | omlaag | $\sqrt{0^2 + 2^2} = 2$ |

    Met k = 1 beslist het dichtstbijzijnde vakje: de kloon gaat in (1, 2) naar rechts, waar de expert omlaag zou gaan, terug naar de middelste rij. Naar rechts blijft de kloon naast de putten: met kans 0,1 glijdt de stap omhoog uit, een put in. Met kans 0,8 komt de kloon in (1, 3), weer een vakje waar de expert nooit stond, op afstand 1 van zowel (2, 3) als (1, 4). Bij een gelijke stand neemt `knn_predict` het paar dat het eerst in de demonstratie staat, (2, 3): weer rechts. Pas in (1, 4), een vakje van de demonstratie, gaat de kloon omlaag, net als de expert.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_demonstrate():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De expert opnemen

        `demonstrate(expert, episodes, seed)` laat `expert` vanaf `START` `episodes` episodes spelen, en geeft `pairs` terug: één paar voor elke stap, van het vakje en de action die de expert daar neemt, in de volgorde van de stappen. `expert` is een policy als dict, zoals `EXPERT`: `expert[state]` is de action in `state`. `step(state, action, rng)` is één stap op de brug: `step` geeft het nieuwe vakje terug, de reward, en `over`, dat `True` is zodra de agent op G of in een put staat. `rng = np.random.default_rng(seed)` is de generator van willekeurige getallen voor het uitglijden, en het vak begint dus met `import numpy as np`. In `for _ in range(episodes):` staat de naam `_` voor een getal dat de loop niet nodig heeft, en `while not over:` herhaalt het blok tot de episode voorbij is. Vervang de `...` door drie regels: de action van de expert, het paar in `pairs`, en de stap.

        De check roept `demonstrate(EXPERT, 1, 0)` aan, de demonstratie van de afbeelding bij *Het idee*, en drie andere gevallen, waaronder een expert die alleen naar rechts gaat.
        """),
        L.task("Laat `demonstrate` voor elke stap een paar van het vakje en de action van de expert opschrijven, en dan de stap zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_demonstrate():
    box_demonstrate = L.code_box("demonstrate", lines=12)
    box_demonstrate
    return (box_demonstrate,)


@app.cell(hide_code=True)
def check_demonstrate(box_demonstrate):
    result_demonstrate = L.check("demonstrate", box_demonstrate.value)
    result_demonstrate
    return (result_demonstrate,)


@app.cell(hide_code=True)
def hints_demonstrate():
    L.hints(
        "demonstrate",
        [
            "Stap 1 van het recept, binnen de `while`-loop: eerst `action`, de action van de expert in `state`, uit de dict `expert`; dan het paar in `pairs`, met `pairs.append`; dan `step`, dat drie dingen teruggeeft: de nieuwe `state`, de reward en `over`.",
            "De volgorde telt: het paar bevat het vakje waar de expert de action neemt, dus gaat het paar in `pairs` voordat `step` de agent verplaatst. `append` neemt één waarde, dus gaan het vakje en de action er samen in, als één tuple tussen haakjes.",
            "`            pairs.append(___)`",
        ],
        why="`pairs.append((state, action))` schrijft het vakje en de action van de expert daar op als één tuple, voordat `step` de agent verder brengt. De drie regels zijn `action = expert[state]`, `pairs.append((state, action))` en `state, reward, over = step(state, action, rng)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clone():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De kloon

        `clone(pairs, k)` leert een kloon uit `pairs`, zoals `demonstrate` de paren geeft, en geeft `policy` terug: een dict met een action voor elk vakje van `CELLS`. Het vak begint weer met `import numpy as np`. De eerste loop, `for state, action in pairs:`, haalt elk paar uit elkaar in het vakje en de action: zet het vakje in `X` en de action in `y`, met `append`. Daarna maakt `np.array` van beide lijsten arrays: `X` krijgt één rij per paar, de rij en de kolom van het vakje. De tweede loop gaat over elk vakje van `CELLS`, ook de vakjes waar de expert nooit stond, en geeft `policy[state]` wat `knn_predict` voorspelt. Dat is je functie uit k-nearest neighbors, `knn_predict(X_train, y_train, x, k)`: de functie zoekt de `k` voorbeelden uit de training set met de kleinste afstand tot `x`, en geeft de klasse terug die de meeste van deze voorbeelden hebben, hier een action. Vervang de twee `...`.

        De check roept `clone(demonstrate(EXPERT, 1, 0), 1)` aan, de kloon van de afbeelding bij *Het idee*, en vier andere gevallen, met k = 1, 3 en 9.
        """),
        L.task("Laat `clone` de vakjes in `X` zetten en de actions in `y`, en geef elk vakje van `CELLS` de action die k-NN voor dit vakje voorspelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clone():
    box_clone = L.code_box("clone", lines=14)
    box_clone
    return (box_clone,)


@app.cell(hide_code=True)
def check_clone(box_clone):
    result_clone = L.check("clone", box_clone.value)
    result_clone
    return (result_clone,)


@app.cell(hide_code=True)
def hints_clone():
    L.hints(
        "clone",
        [
            "Stap 2 en 3 van het recept. In de eerste loop `X.append(state)` en `y.append(action)`. In de tweede loop krijgt `policy[state]` de action die `knn_predict` voorspelt voor `state`.",
            "`knn_predict` neemt vier dingen, in deze volgorde: de voorbeelden uit de training set, de klassen van deze voorbeelden, het nieuwe voorbeeld en k. Welke namen bevatten deze dingen hier?",
            "`        policy[state] = ___`",
        ],
        why="`policy[state] = knn_predict(X, y, state, k)` vraagt k-NN om de action in elk vakje, uit de vakjes en actions van de expert. Met k = 1 neemt de kloon de action over van het dichtstbijzijnde vakje waar de expert stond.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Fouten tellen

        `mistakes(policy, expert, states)` geeft het deel van de states in `states` terug waar `policy` iets anders doet dan `expert`: een getal van 0 tot en met 1. `states` is een lijst met vakjes, en hetzelfde vakje kan er vaker in staan; dan telt het vakje elke keer mee, als een stap die de agent daar zet. Tel in de loop, `for state in states:`, 1 op bij `wrong` voor elke state waar de twee policies verschillende actions geven; geef na de loop `wrong` gedeeld door het aantal states terug. Vervang de twee `...`.

        De check roept `mistakes(clone(demonstrate(EXPERT, 1, 0), 1), EXPERT, CELLS)` aan: de kloon van oefening 2 op alle 35 vakjes. Daarna dezelfde kloon op `demo_cells`, de 21 vakjes van de demonstratie, één per stap, en op `clone_cells`, de 238 vakjes waar de kloon staat in 20 eigen episodes; dan `STRAIGHT`, een expert die alleen naar rechts gaat; en de expert tegen zichzelf.
        """),
        L.task("Laat `mistakes` het deel van de states teruggeven waar de policy en de expert verschillende actions geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_mistakes = L.code_box("mistakes", lines=5)
    box_mistakes
    return (box_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_mistakes):
    result_mistakes = L.check("mistakes", box_mistakes.value)
    result_mistakes
    return (result_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "mistakes",
        [
            "Stap 4 van het recept. In de loop vergelijkt een `if` de action van `policy` in `state` met de action van `expert` daar, met `!=`, de test op ongelijkheid; eronder `wrong = wrong + 1`. Deel na de loop `wrong` door `len(states)`.",
            "Beide policies zijn dicts met een vakje als key. In het voorbeeld verschillen 16 van de 35 vakjes, dus hoort de functie 16 / 35 terug te geven, ongeveer 0,4571.",
            "`        if ___:`",
        ],
        why="`if policy[state] != expert[state]:` telt een fout overal waar de kloon en de expert verschillende actions geven; `return wrong / len(states)` maakt er een deel van, zodat je lijsten van verschillende lengte kunt vergelijken.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Van 1 tot en met 50 demonstraties

    Meer demonstraties laten meer vakjes zien. Het blok hieronder gaat langs 1, 2, 5, 10, 20 en 50 demonstraties, elke keer met 20 seeds, en laat elke kloon 500 episodes spelen. Links het deel van de episodes waarin de klonen vallen, met een verticale lijn bij het getoonde aantal demonstraties; de gestreepte lijn bij 0 is de expert. Rechts de kloon van seed 0.
    """)
    return


@app.cell(hide_code=True)
def explore_demos_cell():
    explore_demos = L.explore("demos", "demonstraties", lambda _v: C.demo_frame(int(_v), L.lang), start=1,
                              values=C.NS, role="",
                              task="Ga van 1 naar 50: vanaf welk aantal doet de kloon van seed 0 in elk vakje wat de expert doet, en hoeveel vallen blijven er dan over, alle 20 seeds samen?")
    explore_demos
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met één demonstratie vallen de klonen in 43,8% van de episodes, met 5 in 22,3%, met 20 in 2,3%, en met 50 in geen van de 10.000. Op de vakjes van de demonstraties maken de klonen nooit een fout, bij elk aantal demonstraties. Het deel van de fouten op de vakjes waar de klonen zelf staan daalt wel: van 32,4% met één demonstratie tot 1,2% met 20. Eén demonstratie laat ongeveer 15 van de 35 vakjes zien, 20 demonstraties ongeveer 32. De seeds verschillen veel: met één demonstratie valt de kloon van de ene seed in 22,4% van de episodes, en de kloon van een andere seed in 57,2%.

    ### De vakjes van één demonstratie met NumPy

    `C.course_runs()[1][0]` geeft de run van seed 0 met één demonstratie: een dict met onder meer `"pairs"` en `"policy"`. De cel hieronder maakt twee arrays met de vorm van de wereld, 5 rijen bij 12 kolommen. `np.zeros((5, 12), dtype=int)` zit vol nullen, gehele getallen, en `np.zeros((5, 12), dtype=bool)` vol `False`. `_seen[_state] += 1` telt 1 op bij het vakje `_state`: een tuple (rij, kolom) kiest één getal van een 2D-array. Zo telt `_seen` hoe vaak de expert op elk vakje stond, en is `_wrong` `True` waar de kloon iets anders doet dan de expert. `_seen[1:4]` neemt rij 1 tot en met 3, de brug zonder de putten. `.astype(int)` maakt van `True` en `False` 1 en 0, `.sum(axis=1)` telt elke rij op, en `&` is *en* voor arrays van `True` en `False`: `_wrong & (_seen > 0)` is `True` waar de kloon een fout maakt op een vakje waar de expert stond. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _run = C.course_runs()[1][0]              # seed 0, één demonstratie
    _seen = np.zeros((5, 12), dtype=int)      # hoe vaak de expert op elk vakje stond
    for _state, _action in _run["pairs"]:
        _seen[_state] += 1
    _wrong = np.zeros((5, 12), dtype=bool)    # True waar de kloon iets anders doet dan de expert
    for _state in C.CELLS:
        _wrong[_state] = _run["policy"][_state] != C.EXPERT[_state]
    print("hoe vaak de expert op elk vakje stond, rij 1 tot en met 3:")
    print(_seen[1:4])
    print("fouten van de kloon, rij 1 tot en met 3:")
    print(_wrong[1:4].astype(int))
    print("fouten per rij:", _wrong[1:4].sum(axis=1))
    print("fouten op vakjes waar de expert stond:", int((_wrong & (_seen > 0)).sum()))
    print("vakjes waar de expert stond:", int((_seen > 0).sum()))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    De expert stond op elk vakje van de middelste rij behalve G, en op 6 vakjes van de buitenste rijen: 17 vakjes. De kloon maakt 7 fouten in de rij boven het midden, 9 in de rij eronder, en geen in de middelste rij. Op een vakje waar de expert stond, doet de kloon met k = 1 wat de expert deed, dus de regel met de fouten op deze vakjes zegt 0. Alle 16 fouten liggen op vakjes naast de putten, waar één verkeerde stap de laatste kan zijn.

    ### De expert opnieuw vragen: DAgger

    De kloon gaat de mist in op vakjes die de expert nooit liet zien. Laat deze vakjes dan zien: laat de kloon spelen, vraag de expert wat de expert gedaan zou hebben in elk vakje waar de kloon stond, voeg deze paren toe aan de demonstratie, en leer de kloon opnieuw. Dat is het idee van **DAgger**, van Ross, Gordon en Bagnell, uit 2011; hier gebeurt het één keer. Eén demonstratie en 5 episodes van de kloon, met de actions van de expert, geven ongeveer 88 paren, evenveel als 5 demonstraties. Deze klonen vallen in 13,7% van de episodes, tegen 22,3% voor de klonen van 5 demonstraties: de extra paren liggen waar de kloon komt, niet waar de expert komt. DAgger heeft wel een expert nodig die voor elke state een antwoord kan geven, niet alleen opnames: iemand die naar de robot kijkt en zegt wat de robot had moeten doen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_k():
    mo.md(r"""
    ### Oefening 4. Meer buren

    Bij k-nearest neighbors hielp een grotere k vaak: er stemmen meer buren, dus telt één vreemd voorbeeld uit de training set minder zwaar. De klonen van deze les gebruiken allemaal k = 1. Een collega wil k = 9 gebruiken voor de klonen van 5 demonstraties, zodat losse vakjes de kloon minder van de wijs brengen.
    """)
    return


@app.cell(hide_code=True)
def choice_k_cell():
    choice_k = L.choice(
        "bigger-k",
        question="Wat gebeurt er met de vallen van de klonen van 5 demonstraties met k = 9 in plaats van k = 1?",
        options={
            "right": "Meer vallen: de negen dichtstbijzijnde vakjes liggen meestal in de middelste rij, dus gaat de kloon in de buitenste rijen naar rechts",
            "fewer": "Minder vallen: negen stemmen strijken de vreemde vakjes glad, zoals bij de pinguïns",
            "same": "Evenveel vallen: op de vakjes van de demonstraties maakt k niets uit",
            "none": "Helemaal geen vallen: met negen buren heeft elk vakje een buur waar de expert stond, en vindt de kloon altijd de action van de expert terug",
        },
        correct="right",
        why={
            "right": "Met k = 9 gaan de vallen van 22,3% naar 55,3%; met k = 3 naar 37,4%. Een buitenste rij heeft weinig vakjes van de demonstraties, de middelste rij veel, allemaal met rechts. De negen buren van een vakje in een buitenste rij zijn meestal vakjes uit het midden, dus wint rechts de stemming, ook in vakjes waar de expert stond en terugstapte naar het midden: op de vakjes van de demonstraties maken de klonen nu 16,2% fouten, tegen geen met k = 1. Een stemming helpt tegen klassen die toevallig fout zijn; de expert zit nooit fout, en de zeldzame klasse is juist de klasse die telt.",
            "fewer": "Een stemming helpt als losse voorbeelden uit de training set toevallig fout zijn, en de actions van de expert zijn nooit fout. Welke action hebben de meeste van de negen dichtstbijzijnde vakjes van een vakje in een buitenste rij?",
            "same": "Met k = 1 doet een kloon de expert na op de vakjes van de demonstraties; beslist het vakje zelf nog met negen stemmers?",
            "none": "Ook met k = 1 heeft elk vakje een dichtstbijzijnde vakje waar de expert stond: de vraag is waarop de negen dichtstbijzijnde vakjes stemmen. Welke action hebben de meeste, voor een vakje in een buitenste rij?",
        },
    )
    choice_k
    return (choice_k,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) leerde de agent van rewards, door te proberen, en betaalde de agent tijdens het leren voor de eigen fouten. Behavior cloning leert zonder enige reward, van een expert, met de classifier van [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl): elke state een voorbeeld, elke action van de expert de klasse. Dat werkt op de states die de expert liet zien, en de eigen fouten van de agent brengen de agent ergens anders heen. Veel robots leren eerst zo, van mensen die de robot met de hand besturen, met veel meer demonstraties; DAgger of RL achteraf herstellen dan wat nadoen mist. De volgende les, *Je krijgt wat je beloont: reward misspecification*, gaat terug naar rewards: een agent doet wat de reward zegt, ook als dat niet is wat je bedoelde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_k, result_clone, result_demonstrate, result_mistakes):
    L.wrap_up(result_demonstrate, result_clone, result_mistakes, choice_k)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
