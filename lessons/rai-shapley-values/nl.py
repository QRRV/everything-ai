# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-shapley-values. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-shapley-values", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weight():
    quick_weight = L.choice(
        "quick-weight",
        question="In de vorige les gaf LIME education_num een weight van 0,144 voor de persoon van 41. Wat is die weight?",
        options={
            "right": "De helling van een lijn bij de persoon",
            "share": "Het deel van de kans dat van de opleiding komt",
            "count": "Hoe vaak het forest naar de opleiding vraagt",
        },
        correct="right",
        why={
            "right": "Goed: een weight van LIME is de helling van een rechte lijn die dicht bij de persoon gefit is, per standaardafwijking van de kolom. Die weight verdeelt de kans niet over de features. De Shapley-waarden van deze les doen dat wel.",
            "share": "Dat geven de Shapley-waarden van deze les. Een weight van LIME is de helling van een rechte lijn dicht bij de persoon: hoeveel de kans van de lijn meebeweegt bij één standaardafwijking meer. Zie [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl).",
            "count": "LIME kijkt nooit in het forest: het vraagt alleen om antwoorden op verzonnen rijen, en de weights zijn de hellingen van een rechte lijn daardoorheen. Zie [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl).",
        },
    )
    quick_weight
    return (quick_weight,)


@app.cell(hide_code=True)
def quick_check_small():
    quick_small = L.choice(
        "quick-small",
        question="Met kernel width 0,1 was de weight van education_num in drie runs 0,081, 0,199 en 0,242. Waarom sprong hij zo?",
        options={
            "right": "Samen telden de 5000 rijen als zo'n 2",
            "trained": "Het forest werd elke run opnieuw getraind",
            "moved": "De willekeurige rijen verschoven de persoon elke keer",
        },
        correct="right",
        why={
            "right": "Goed: met width 0,1 tellen de 5000 rijen even zwaar als maar ongeveer 2 rijen op de plek van de persoon, dus de lijn van elke run rust op een handvol rijen. De Shapley-waarden van deze les worden voor vier kolommen precies uitgerekend: bij dezelfde achtergrond komen ze elke keer hetzelfde uit.",
            "trained": "Het forest groeit één keer, met `random_state=0`; alleen de verzonnen rijen van LIME veranderen van run tot run. Met width 0,1 tellen ze even zwaar als maar ongeveer 2 rijen op de plek van de persoon. Zie [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl).",
            "moved": "De persoon blijft op dezelfde plek; de verzonnen rijen veranderen van run tot run, en met width 0,1 tellen ze even zwaar als maar ongeveer 2 rijen op de plek van de persoon. Zie [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl).",
        },
    )
    quick_small
    return (quick_small,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De toezichthouder heeft het antwoord van LIME voor de persoon van 41 met een bachelordiploma die nee kreeg: met de eigen kernel width van lime weegt education_num het zwaarst, met een kleinere width hours_per_week. De persoon vraagt iets anders: hoeveel van het verschil tussen de eigen kans en die van andere mensen komt van elke feature? Met een vierde kolom, capital_gain, geeft het forest deze persoon een kans van 0,451, en 200 mensen uit de helft om op te trainen krijgen gemiddeld 0,225. Een Shapley-waarde verdeelt dat verschil van 0,226 zo over de vier features dat de vier delen er precies op uitkomen, met een regel uit de speltheorie die Lloyd Shapley in 1952 opschreef. Met vier features kun je hem precies uitrekenen. Maar "andere mensen" is een keuze: vergelijk met een andere groep, en de delen veranderen.

    **Een voorbeeld uit het dagelijks leven.** Twee vrienden doen als team mee aan een pubquiz en halen 10 punten. In hun eentje had Ash er 2 gehaald en Bo 4. Hoe verdeel je de 10? Laat ze in elke volgorde binnenkomen, en geef elk wat het team erbij krijgt als die persoon aanschuift. Komt Ash eerst, dan voegt Ash 2 toe en daarna Bo 8; komt Bo eerst, dan voegt Bo 4 toe en daarna Ash 6. Gemiddeld over de twee volgordes krijgt Ash (2 + 6) / 2 = 4 en Bo (8 + 4) / 2 = 6: samen 10.
    """)
    return


@app.cell(hide_code=True)
def predict_largest():
    guess_largest = L.guess(
        "largest",
        question="De vier features verdelen het verschil van 0,226 tussen de kans van deze persoon en het gemiddelde van de achtergrond. Welke krijgt het grootste deel?",
        options={
            "age": "age",
            "education": "education_num",
            "hours": "hours_per_week",
            "gain": "capital_gain",
        },
        reveal="**education_num, met +0,181.** Daarna age, met +0,087. hours_per_week krijgt −0,010 en capital_gain −0,032, onder nul: de persoon heeft geen vermogenswinst, en als je die 0 in elke rij van de achtergrond zet, zakt het gemiddelde van het forest van 0,225 naar 0,190; 22 van de 200 mensen hebben er wel een. Samen 0,226, precies het verschil. LIME zette education_num bij de eigen width van lime ook bovenaan, maar zijn weights zijn hellingen; dit zijn delen van een kans.",
    )
    guess_largest
    return (guess_largest,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een deel voor elke speler.** In het stuk van Lloyd Shapley voor de RAND Corporation uit 1952 geeft een spel een waarde aan elke coalitie, elke groep spelers, met 0 voor de lege, en is de vraag hoe je de waarde van alle spelers samen verdeelt. Het antwoord is een verhaal: de coalitie begint met één speler en neemt er telkens één bij, tot iedereen erin zit. Het toeval bepaalt de volgorde, waarbij elke volgorde even waarschijnlijk is, en elke speler krijgt wat de coalitie erbij krijgt als die speler aanschuift. Shapley vroeg om drie regels: symmetrie, de waarde hangt alleen af van het spel zelf en niet van de namen van de spelers; efficiëntie, de waarde verdeelt de volledige opbrengst van het spel; en een regel voor optellen, de waarde van twee spellen samen is de som van hun waarden. Shapley vond het opmerkelijk dat er verder niets nodig is om de waarde vast te leggen: er is er precies één. Een speler die aan geen enkele coalitie iets toevoegt, krijgt dan niets.

        **De formule.** Erik Štrumbelj en Igor Kononenko schrijven hem als een som over de coalities $S$ waar speler $i$ niet in zit, met $n$ spelers en $s$ spelers in $S$:

        $$\phi_i = \sum_{S\,:\,i \notin S} \frac{s!\,(n-s-1)!}{n!}\,\bigl(v(S \cup \{i\}) - v(S)\bigr)$$

        Het gewicht is het deel van de $n!$ volgordes waarin precies de spelers van $S$ vóór $i$ komen. Met vier spelers zijn er 24 volgordes. In 6 daarvan komt $i$ eerst, dus de lege coalitie telt $6/24 = 1/4$; één bepaalde speler staat in 2 volgordes als enige vóór $i$, $1/12$; twee bepaalde spelers ook in 2, $1/12$; alle drie de anderen in 6, $1/4$. Elk van de vier spelers heeft 8 coalities waar die niet in zit, en alle vier de waarden samen hebben de 16 coalities nodig.

        **Van spelers naar features.** Štrumbelj en Kononenko (2010) maakten de features van een model tot de spelers, met als doel een algemene methode om losse voorspellingen van classificatiemodellen uit te leggen. Hun eerste voorbeeld: vooraf heeft een passagier van de Titanic 32% kans om het te overleven, en het model voorspelt voor één passagier 67%. De waarde van een coalitie is dan hoeveel de verwachte voorspelling verandert als alleen de features daarin bekend zijn. De onbekende features namen ze aan als gelijk verdeeld over al hun mogelijke waarden. Deze les doet in plaats daarvan wat Christoph Molnar beschrijft: vul elke onbekende feature in uit rijen van de data, hier 200 mensen uit de helft om op te trainen, de achtergrond, en neem het gemiddelde. Het gemiddelde zelf is hier de waarde van een coalitie, dus de lege coalitie is 0,225 waard in plaats van 0; hetzelfde getal van elke coalitie aftrekken verandert geen enkel verschil, dus de Shapley-waarden blijven gelijk. Dan tellen de waarden op tot de voorspelling min de gemiddelde voorspelling over diezelfde achtergrond. Het voorbeeld van Molnar is een appartement: het model voorspelt €300.000, de gemiddelde voorspelling voor alle appartementen is €310.000, en de vier bijdragen tellen op tot −€10.000, de voorspelling min de gemiddelde voorspelde prijs.

        **SHAP.** Scott Lundberg en Su-In Lee (2017) zetten LIME en Shapley-waarden in één familie: een uitleg als een som, een basiswaarde plus één deel voor elke feature. Ze vroegen om drie eigenschappen: local accuracy, de basiswaarde en de delen tellen samen op tot de voorspelling; missingness, een feature die er niet is krijgt niets; en consistency. Ze lieten zien dat maar één uitleg van die vorm alle drie heeft: de Shapley-waarden. Ze noemden ze SHAP-waarden, de Shapley-waarden van een voorwaardelijke verwachting van het oorspronkelijke model, en om ze uit te rekenen stonden ze toe dat je de features onafhankelijk van elkaar neemt, een aanname die je mag maken maar niet hoeft, en dat geeft de soort die deze les uitrekent. Met de eigen keuzes van LIME, lieten ze zien, vindt de lijn van LIME de Shapley-waarden niet; met een andere kernel, gefit op welke features bekend zijn (0 of 1), wel. De volgende les laat de tools zien.

        **Welk "onbekend"?** Invullen uit rijen van de achtergrond is één keuze van spel, en er zijn er meer. Elizabeth Kumar, Suresh Venkatasubramanian, Carlos Scheidegger en Sorelle Friedler noemen de keuze een catch-22. Hun value function is het spel zelf, de $v$ die elke coalitie haar waarde geeft, niet de value function uit reinforcement learning. Vul je in uit andere mensen zoals ze zijn, zoals hier, dan vraag je het model naar rijen die misschien niet bestaan: zulke methoden hangen ervan af dat het model antwoordt op rijen die niet lijken op de data waarop het getraind is. Vul je alleen waarden in die bij de bekende passen (de voorwaardelijke soort), dan kan een feature zonder enig effect toch invloed krijgen: een feature die het model nooit gebruikt, kan een deel krijgen. Mukund Sundararajan en Amir Najmi vonden dat de verschillende manieren heel verschillende uitkomsten geven, waardoor de uniciteit niets meer betekent. De regels van Shapley maken het antwoord uniek voor één spel; welk spel is jouw keuze.

        **Wat een Shapley-waarde niet zegt.** Molnar, met nadruk: de Shapley-waarde is NIET het verschil in voorspelling als je de feature uit het model haalt. Een Shapley-waarde geeft ook geen richting: een positieve Shapley-waarde betekent niet dat de voorspelling stijgt als de feature groter wordt. Kumar en anderen geven een voorbeeld: een model $f(x) = 2 - (x - 1)^2$, met $x$ getrokken rond 0 met spreiding 1. Iemand met $x = 1$ krijgt een Shapley-waarde van 2, het meeste dat iemand kan krijgen, en kan dan denken dat een nog grotere $x$ de score verder verhoogt, maar dat is niet zo. En een deel is altijd tegenover één achtergrond: Molnar wijst erop dat je een Shapley-waarde altijd leest tegenover de data waaruit je de ontbrekende spelers invulde. Vermenigvuldigt een model onafhankelijke features rond 0 met elkaar, dan krijgt elke feature hetzelfde deel, wat zijn eigen waarde ook is (Kumar en anderen). En SHAP is net als LIME te misleiden, al minder vaak: met één betekenisloze kolom werkte de aanval van Slack en anderen bij 84% van de mensen in COMPAS.

        **Wat het kost.** Molnar: de rekentijd groeit exponentieel met het aantal features. Vier features vragen 16 coalities; twintig er meer dan een miljoen. Štrumbelj en Kononenko namen daarom al een steekproef van volgordes, om de exponentiële rekentijd van de methode te omzeilen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **Shapley-waarde** | het deel van een feature in het verschil tussen één voorspelling en de gemiddelde voorspelling: wat hij aan een coalitie toevoegt, gemiddeld over elke volgorde waarin de features kunnen aanschuiven |
        | **coalitie** | de features die bekend zijn; bij vier features zijn er 16, van geen tot alle vier |
        | **achtergrond** | de rijen die de features invullen die niet bekend zijn; de Shapley-waarden tellen op tot de voorspelling min het gemiddelde van het model daarover |
        | **SHAP** | de naam van Lundberg en Lee voor de Shapley-waarden van de voorspelling van een model, en de library die ze uitrekent |
        | $\phi_i$ | de Shapley-waarde van feature $i$ (phi) |
        | $s!$ | $1 \times 2 \times \dots \times s$, met $0! = 1$; in Python `math.factorial(s)` |
        """),
        L.watch_out("Een Shapley-waarde is een deel van het verschil tussen deze voorspelling en het gemiddelde van het model over de achtergrond die jij koos. Die is niet wat de voorspelling doet zonder de feature, die zegt niet in welke richting je de feature moet veranderen, en met een andere achtergrond verandert die. Zeg welke achtergrond je gebruikte, en lees de waarde als een deel, niet als een oorzaak."),
    ])
    return


@app.cell(hide_code=True)
def explore_background_cell():
    explore_background = L.explore("background-top", "hoogste education_num in de achtergrond",
                                   lambda v: C.background_picture(v, "nl"), start=16, values=C.TOPS, role="hyper",
                                   task="Schuif van 16 omlaag naar 8. Wat gebeurt er met het deel van education_num, en waar tellen de vier waarden elke keer op uit?")
    mo.vstack([
        mo.md("**Eén persoon, andere mensen om mee te vergelijken**"),
        explore_background,
        mo.md(r"""
        De Shapley-waarden van de course voor de persoon van 41 met education_num 13, gemaakt zoals in het uitgewerkte voorbeeld verderop, met 200 rijen voor de achtergrond, getrokken uit de mensen van de helft om op te trainen met een education_num van hoogstens de waarde op de schuifbalk. Elke balk is de waarde van één feature en begint waar de balk erboven eindigt: van het gemiddelde van de achtergrond bovenaan naar de kans van de persoon, 0,451, onderaan.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om één voorspelling over zijn features te verdelen:

    1. Neem de rij die je uitlegt, en een achtergrond: rijen uit de data om in te vullen wat niet bekend is.
    2. Neem voor elke coalitie, de features die je kent, elke rij van de achtergrond, zet de bekende features van de rij die je uitlegt erin, vraag het model, en neem het gemiddelde van de antwoorden: dat is de waarde van de coalitie.
    3. Neem voor elke feature elke coalitie zonder die feature, en kijk wat het kennen van de feature eraan toevoegt.
    4. Weeg elke toevoeging met het deel van de volgordes waarin precies die coalitie vóór de feature komt, en tel ze op: dat is de Shapley-waarde van de feature.
    5. Controleer dat de waarden optellen tot het antwoord van het model voor de rij min de waarde als niets bekend is.
    6. Lees elke waarde als een deel van dat verschil, tegenover deze achtergrond: niet als wat er gebeurt als je de feature verandert of weghaalt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het forest uit de vorige lessen met een vierde kolom, capital_gain, en dezelfde persoon. `np.random.default_rng(0).choice(n, 200, replace=False)` trekt 200 verschillende plaatsen uit `n`. `itertools.product([0, 1], repeat=4)` geeft de 16 tuples van vier nullen en enen, van `(0, 0, 0, 0)` tot `(1, 1, 1, 1)`: de coalities. `np.where(condition, a, b)` neemt `a` waar de voorwaarde klopt en `b` op de andere plaatsen: hier de waarde van de persoon in de bekende kolommen en de eigen waarde van de rij uit de achtergrond in de andere, voor alle 200 rijen tegelijk. `itertools.permutations(range(4))` geeft de 24 volgordes van 0, 1, 2 en 3. Stap 4 laat de kolommen in elke volgorde een voor een aanschuiven, en geeft elke kolom wat zijn komst toevoegt, gedeeld door 24. `f"{x:+.3f}"` schrijft `x` met zijn teken en drie decimalen. Voer de cel uit. Dat duurt een paar seconden.
    """)
    return


@app.cell
def example_shapley():
    import itertools

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # 1. het forest uit de vorige lessen, nu met een vierde kolom, en de persoon die je uitlegt
    _people = pd.read_csv("data/adult.csv.gz")
    _columns = ["age", "education_num", "hours_per_week", "capital_gain"]
    _X = _people[_columns].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _person = _X[_test][5]
    print("persoon:", _person, " kans op ja:", _forest.predict_proba([_person])[0, 1].round(3))

    # 2. de achtergrond: 200 mensen uit de helft om op te trainen, om de onbekende kolommen in te vullen
    _background = _X[_train][np.random.default_rng(0).choice(len(_train), 200, replace=False)]

    # 3. de waarde van elk van de 16 coalities: de gemiddelde kans op ja van het forest over de achtergrond,
    #    met de bekende kolommen (een 1 in de coalitie) uit de persoon
    _v = {}
    for _known in itertools.product([0, 1], repeat=4):
        _rows = np.where(np.array(_known) == 1, _person, _background)
        _v[_known] = _forest.predict_proba(_rows)[:, 1].mean()
    print("niets bekend:", _v[(0, 0, 0, 0)].round(3), " alleen education_num:", _v[(0, 1, 0, 0)].round(3),
          " alle vier:", _v[(1, 1, 1, 1)].round(3))

    # 4. wat elke kolom toevoegt als hij aanschuift, gemiddeld over de 24 volgordes waarin dat kan
    _orders = list(itertools.permutations(range(4)))
    _values = [0, 0, 0, 0]
    for _join in _orders:
        _known = [0, 0, 0, 0]
        for _i in _join:
            _before = _v[tuple(_known)]
            _known[_i] = 1
            _values[_i] = _values[_i] + (_v[tuple(_known)] - _before) / len(_orders)
    for _i in range(4):
        print("Shapley-waarde van", _columns[_i], f"{_values[_i]:+.3f}")
    print("samen:", round(sum(_values), 3),
          " kans min gemiddelde:", (_v[(1, 1, 1, 1)] - _v[(0, 0, 0, 0)]).round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Als niets bekend is, is het gemiddelde van het forest over de achtergrond 0,225; als alleen education_num bekend is 0,366, en als alle vier bekend zijn 0,451, de eigen kans van de persoon. education_num krijgt +0,181 en age +0,087; hours_per_week krijgt −0,010 en capital_gain −0,032. Samen 0,226, precies het verschil: dat is de efficiëntie van Shapley. Voer de cel nog eens uit en er verandert niets, anders dan bij LIME: als de achtergrond eenmaal getrokken is, zit er geen toeval meer in. De formule, met een gewicht per coalitie, geeft dezelfde vier getallen als de 24 volgordes; dat is oefening 2. (De print schrijft de getallen met een punt, zoals Python dat doet.)
    """)
    return


@app.cell(hide_code=True)
def text_exercise_value():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De waarde van een coalitie

        Schrijf `value(model, person, background, known)`: `model(rows)` geeft een lijst met het antwoord van het model voor elke rij, `person` is de rij die je uitlegt, `background` is een lijst met rijen, en `known` is een tuple met een 0 of 1 voor elke kolom. Maak voor elke rij van `background` een nieuwe rij `filled` met `person[j]` waar `known[j]` 1 is, en de eigen `row[j]` van de rij uit de achtergrond waar hij 0 is. Vraag `model` naar alle nieuwe rijen, en geef het gemiddelde van de antwoorden terug. De check roept eerst `value(add_up, [4, 0, 1], [[1, 2, 3], [3, 1, 0]], (1, 0, 1))` aan, waarbij `add_up` voor elke rij de som teruggeeft; daarna andere modellen, zoals `product` en `biggest`. De tekening voert je functie uit met het forest, voor alle 16 coalities van de persoon.
        """),
        L.task("Schrijf `value(model, person, background, known)`: bouw een lijst `rows` met één ingevulde rij voor elke rij van de achtergrond, en geef het gemiddelde van `model(rows)` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value():
    box_value = L.code_box("value", lines=14)
    box_value
    return (box_value,)


@app.cell(hide_code=True)
def check_value(box_value):
    result_value = L.check("value", box_value.value)
    result_value
    return (result_value,)


@app.cell(hide_code=True)
def hints_value():
    L.hints(
        "value",
        [
            "Een bekende kolom komt uit de persoon, een onbekende uit de rij van de achtergrond. Bouw één nieuwe rij voor elke rij van de achtergrond, en vraag het model daarna naar allemaal tegelijk.",
            "Begin met `rows = []` en `for row in background:`, met `filled = []` en een binnenste `for j in range(len(person)):` die `person[j]` of `row[j]` toevoegt. Na de loop: `chances = model(rows)`.",
            "`return sum(chances) / ___`",
        ],
        why="`rows = []`, dan `for row in background:` met `filled = []`, `for j in range(len(person)):`, en `filled.append(person[j])` als `known[j] == 1`, anders `filled.append(row[j])`; dan `rows.append(filled)`. Na de loop `chances = model(rows)` en `return sum(chances) / len(chances)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shapley():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De Shapley-waarde van een kolom

        Het vak geeft `weight(s, n)`, het gewicht $s!\,(n-s-1)!/n!$ van een coalitie van $s$ kolommen uit $n$. Maak `shapley(v, i)` af: `v` is een dict van elke coalitie, een tuple met nullen en enen, naar zijn waarde, en `i` is de kolom. Loop langs de coalities `known` in `v`. Maak voor elke coalitie waarin `known[i]` 0 is dezelfde coalitie met kolom `i` erbij: `list(known)`, zet plaats `i` op 1, en weer `tuple(...)` om hem in `v` op te zoeken. Tel zijn gewicht keer het verschil van de twee waarden op bij `total`. Een coalitie heeft `sum(known)` bekende kolommen, uit `len(known)`. De check roept eerst `shapley({(0, 0): 0, (1, 0): 2, (0, 1): 4, (1, 1): 10}, 0)` aan, de pubquiz uit *Waarom*: Ash hoort 4 te krijgen. Daarna spellen met drie en vier kolommen. De tekening voert je functie uit op de 16 waarden uit het uitgewerkte voorbeeld.
        """),
        L.task("Vervang de `...`: begin met `total = 0`, tel voor elke coalitie zonder kolom `i` zijn gewicht keer wat `i` eraan toevoegt op, en geef `total` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shapley():
    box_shapley = L.code_box("shapley", lines=17)
    box_shapley
    return (box_shapley,)


@app.cell(hide_code=True)
def check_shapley(box_shapley):
    result_shapley = L.check("shapley", box_shapley.value)
    result_shapley
    return (result_shapley,)


@app.cell(hide_code=True)
def hints_shapley():
    L.hints(
        "shapley",
        [
            "Kolom `i` voegt aan elke coalitie zonder `i` de waarde mét `i` min de waarde zonder `i` toe, en elk van die verschillen telt mee met het gewicht van de grootte van de coalitie: `sum(known)` kolommen uit `len(known)`.",
            "Begin met `total = 0` en `for known in v:`. Maak binnen `if known[i] == 0:` de lijst `with_i = list(known)`, zet `with_i[i] = 1`, en tel op bij `total`; geef na de loop `total` terug.",
            "`total = total + weight(sum(known), len(known)) * (___ - v[known])`",
        ],
        why="`total = 0`, dan `for known in v:` met `if known[i] == 0:`, `with_i = list(known)`, `with_i[i] = 1` en `total = total + weight(sum(known), len(known)) * (v[tuple(with_i)] - v[known])`. Na de loop `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_meaning():
    mo.md(r"""
    ### Oefening 3. Wat het deel zegt
    """)
    return


@app.cell(hide_code=True)
def choice_meaning_cell():
    choice_meaning = L.choice(
        "meaning",
        question="Voor deze persoon krijgt education_num een Shapley-waarde van +0,181. Wat zegt dat?",
        options={
            "right": "Zijn deel van het verschil met het gemiddelde",
            "removed": "Zonder deze feature zou de kans 0,181 lager zijn",
            "more": "Meer opleiding verhoogt de kans met 0,181",
        },
        correct="right",
        why={
            "right": "Goed. De vier waarden verdelen het verschil tussen de kans van de persoon, 0,451, en het gemiddelde van het forest over de achtergrond, 0,225, en het deel van education_num is +0,181: wat het kennen ervan toevoegt, gemiddeld over elke volgorde waarin de kolommen kunnen aanschuiven. Molnar zegt het met nadruk: de Shapley-waarde is NIET het verschil in voorspelling als je de feature uit het model haalt.",
            "removed": "Molnar zegt het met nadruk: de Shapley-waarde is NIET het verschil in voorspelling als je de feature uit het model haalt. Het is een gemiddelde over de coalities: weten wat education_num is, voegt tussen 0,139 en 0,226 toe, afhankelijk van wat er verder bekend is. Zie *Wat een Shapley-waarde niet zegt* in *Het idee*.",
            "more": "Een Shapley-waarde heeft geen richting: volgens Molnar betekent een positieve Shapley-waarde niet dat de voorspelling stijgt als de feature groter wordt. De waarde zegt hoe het verschil met het gemiddelde verdeeld wordt, voor deze persoon zoals die is. Zie *Wat een Shapley-waarde niet zegt* in *Het idee*.",
        },
    )
    choice_meaning
    return (choice_meaning,)


@app.cell(hide_code=True)
def text_exercise_background():
    mo.md(r"""
    ### Oefening 4. Een andere achtergrond
    """)
    return


@app.cell(hide_code=True)
def choice_background_cell():
    choice_background = L.choice(
        "background",
        question="Je legt dezelfde persoon opnieuw uit, met een achtergrond van 200 mensen met een education_num van 13 of meer. Wat gebeurt er?",
        options={
            "right": "De waarden veranderen, en hun som ook",
            "unique": "Niets: Shapley-waarden zijn uniek, dus ze blijven gelijk",
            "only": "Alleen education_num krijgt een nieuwe waarde",
        },
        correct="right",
        why={
            "right": "Goed. De gemiddelde kans op ja van die mensen is 0,444, dus het verschil met de 0,451 van deze persoon is maar 0,008 (0,4513 min 0,4436, vóór het afronden), en education_num krijgt −0,027: onder mensen met 13 of meer is 13 het laagst. Dezelfde persoon en hetzelfde forest, vergeleken met andere mensen. Molnar wijst erop dat je een Shapley-waarde altijd leest tegenover de data waaruit je de ontbrekende spelers invulde.",
            "unique": "De regels van Shapley maken het antwoord uniek voor één spel, en de achtergrond hoort bij het spel: met mensen met een education_num van 13 of meer is het gemiddelde 0,444 en krijgt education_num −0,027. Sundararajan en Najmi vonden dat de verschillende manieren heel verschillende uitkomsten geven, waardoor de uniciteit niets meer betekent. Zie *Welk \"onbekend\"?* in *Het idee*.",
            "only": "Het deel van elke feature kan veranderen, en ook het verschil dat ze verdelen: met mensen met een education_num van 13 of meer is het gemiddelde 0,444, tellen de vier waarden op tot het verschil, 0,008 vóór het afronden, en krijgt age +0,112 in plaats van +0,087. Zie de schuifbalk in *Het idee*.",
        },
    )
    choice_background
    return (choice_background,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl) legde één voorspelling uit met de hellingen van een lijn, die veranderen met de kernel width; Shapley-waarden verdelen de voorspelling, en veranderen in plaats daarvan met de achtergrond. Net als LIME zijn ze een local explanation, en model-agnostic: ze hebben alleen de antwoorden van het model nodig, in de indeling van [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl). De volgende les, [Tools voor uitleg](/course/open/rai-xai-tools?lang=nl), laat de libraries zien die dit voor veel features doen, en welke van hun keuzes het antwoord veranderen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_background, choice_meaning, result_shapley, result_value):
    L.wrap_up(result_value, result_shapley, choice_meaning, choice_background)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
