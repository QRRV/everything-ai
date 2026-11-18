# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-rashomon. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-rashomon", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een korte vraag over [Glass-box models](/course/open/rai-glass-box-models?lang=nl) en een over [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_glass():
    quick_glass = L.choice(
        "quick-glass",
        question="Is elke beslisboom een glass box?",
        options={
            "right": "Alleen een kleine: het hangt af van de grootte",
            "yes": "Ja: je kunt zijn vragen altijd volgen",
            "no": "Nee: een boom is een black box, net als een neural network",
        },
        correct="right",
        why={
            "right": "Goed. Geen soort model is uit zichzelf een glass box: een boom met diepte 2 heeft hoogstens vier bladeren, die je met de hand kunt volgen, maar een boom met diepte 9 kan er 512 hebben. In deze les doen bomen van elke diepte van 1 tot 10 mee.",
            "yes": "Elk pad van een diepe boom kun je volgen, maar het zijn er te veel om te overzien: een boom met diepte 9 kan 512 bladeren hebben. Of een mens een model kan volgen, hangt af van de grootte. Zie [Glass-box models](/course/open/rai-glass-box-models?lang=nl).",
            "no": "Een korte boom is het schoolvoorbeeld van een glass box: een boom met diepte 2 heeft hoogstens vier bladeren, en die kun je met de hand volgen. Het hangt af van de grootte. Zie [Glass-box models](/course/open/rai-glass-box-models?lang=nl).",
        },
    )
    quick_glass
    return (quick_glass,)


@app.cell(hide_code=True)
def quick_check_bootstrap():
    quick_bootstrap = L.choice(
        "quick-bootstrap",
        question="Elke boom van een random forest leert van een bootstrap-steekproef uit de n voorbeelden van de training set. Wat is dat?",
        options={
            "right": "n trekkingen, met teruglegging",
            "order": "Alle n voorbeelden, in een nieuwe volgorde geschud",
            "half": "Een willekeurige helft, elk voorbeeld hoogstens één keer",
        },
        correct="right",
        why={
            "right": "Goed: n plaatsen, willekeurig getrokken, elke keer uit alle n, dus sommige voorbeelden komen twee keer voor en andere helemaal niet. In deze les krijgt elke boom zijn eigen bootstrap-steekproef: elke keer een beetje andere data.",
            "order": "Dezelfde voorbeelden in een andere volgorde geven dezelfde boom. Een bootstrap-steekproef trekt n keer met teruglegging, dus sommige voorbeelden komen twee keer voor en andere helemaal niet. Zie [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
            "half": "Een bootstrap-steekproef heeft evenveel trekkingen als er voorbeelden zijn, n, met teruglegging: sommige komen twee keer voor, andere helemaal niet. Zie [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
        },
    )
    quick_bootstrap
    return (quick_bootstrap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een bank maakt vijftig modellen om over leningen te beslissen, en kiest het model dat het vaakst goed zit; net als in [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl) lezen we een voorspelling van meer dan 50.000 dollar verdienen als het ja van de bank. Vijfendertig van de modellen zitten hoogstens één procentpunt onder het beste. Neem iemand van 36 met een bachelordiploma die 40 uur per week werkt. In het uitgewerkte voorbeeld van deze les zegt het beste model nee, en zeggen 17 van die 35 ja. Of deze persoon een ja krijgt, hangt dan af van welk van de even goede modellen de bank toevallig koos. Emily Black, Manish Raghavan en Solon Barocas (2022) vragen waarom iemand een ongunstige uitkomst van een model zou moeten krijgen, als er een even goed model bestaat dat gunstiger voor die persoon uitvalt.

    **Een voorbeeld uit het dagelijks leven.** Vier mensen die dezelfde gebeurtenis zagen, vertellen vier verschillende verhalen, en elk verhaal is geloofwaardig. De Japanse film *Rashomon* (1950) is daarop gebouwd, en de statisticus Leo Breiman noemde het effect van deze les ernaar: data waar veel verschillende modellen ongeveer even goed bij passen.
    """)
    return


@app.cell(hide_code=True)
def predict_fifty():
    guess_fifty = L.guess(
        "fifty",
        question="In het uitgewerkte voorbeeld zitten 35 bomen hoogstens één procentpunt onder de beste boom. Bij welk deel van de 24.421 mensen waarop ze getest worden, geeft minstens één van de 35 een ander antwoord dan de beste boom?",
        options={
            "few": "Ongeveer 1 op de 100",
            "some": "Ongeveer 1 op de 20",
            "many": "Ongeveer 1 op de 5",
        },
        reveal="**Ongeveer 1 op de 5: 20,6 procent**, 5035 mensen. Toch zit elk van de 35 bomen bij 78,9 tot 79,8 procent van hen goed. Bomen die ongeveer even goed zijn, kunnen het toch over veel mensen oneens zijn: elke boom maakt zijn fouten deels bij andere mensen.",
    )
    guess_fifty
    return (guess_fifty,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het Rashomon effect.** Leo Breiman (2001) haalt drie lessen uit zijn werk, en de eerste noemt hij Rashomon: dat er veel goede modellen zijn. Hij legt uit dat er vaak een hele reeks verschillende beschrijvingen is, verschillende vergelijkingen f(x) uit dezelfde soort, die ongeveer dezelfde kleinste fout geven; dat noemt hij het Rashomon effect. In een van zijn voorbeelden passen veel vergelijkingen met vijf variabelen ongeveer even goed. Welke is beter? Het probleem, schrijft hij, is dat elke vergelijking een ander verhaal vertelt over welke variabelen ertoe doen. Bomen doen het ook: als hij een willekeurige 2 tot 3 procent van de training set weghaalt, krijgt hij een boom die flink van de eerste verschilt, met bijna dezelfde fout op de test set. Zijn conclusie: dit probleem van veel modellen, en wat het doet met de conclusies die mensen uit modellen trekken, verdient serieuze aandacht.

        **Andere antwoorden voor dezelfde mensen.** Het gaat Breiman om verschillende verhalen. Charles Marx, Flavio du Pin Calmon en Berk Ustun (2020) kijken naar de mensen. Predictive multiplicity is bij hen dat een voorspelprobleem concurrerende modellen toelaat die verschillende voorspellingen doen. De concurrerende modellen zijn alle modellen van één soort met een fout van hoogstens ε boven die van het beste model; zij noemen dat de ε-level set, anderen de Rashomon set. Ze meten twee getallen, op de training set. Ambiguity: hoeveel mensen van minstens één concurrerend model een andere voorspelling krijgen. Discrepancy: hoeveel voorspellingen hoogstens veranderen als je het model dat je gebruikt vervangt door een concurrerend model. Op COMPAS-data over wie opnieuw werd gearresteerd, vonden ze dat een model dat maar 1 procent minder goed voorspelt dan het beste, bij meer dan 17 procent van de mensen een andere voorspelling doet, en dat de keuze van het model de voorspelling van 44 procent kan veranderen. Dat geldt niet voor iedereen even sterk: in hun data over arrestaties voor een geweldsmisdrijf kon 72,9 procent van de Afro-Amerikaanse mensen binnen die 1 procent een andere voorspelling krijgen, tegen 37,2 procent van de witte mensen.

        **Breder zoeken vindt meer.** Die getallen komen uit een zoektocht door alle lineaire modellen binnen de marge. Veel modellen opnieuw trainen, zoals deze les doet, vindt er minder: op dezelfde data gaf zo'n losse aanpak een ambiguity van 10 procent en een discrepancy van 7 procent, veel minder dan de 44 en 17 procent van hun eigen methode. Zulke aanpakken kunnen laten zien dat er predictive multiplicity is, schrijven ze, maar je kunt er niet mee aantonen dat die er niet is.

        **Model multiplicity.** Black, Raghavan en Barocas noemen het geheel model multiplicity: even goede modellen kunnen verschillen in hoe ze vanbinnen werken, procedural multiplicity, en in wat ze voorspellen, predictive multiplicity. Zij meten op de data met labels die een ontwikkelaar heeft, en dat kan een test set zijn, zoals in deze les. Ze zien een probleem: wie een model alleen kiest op hoe goed het voorspelt, kiest zo goed als willekeurig tussen gunstiger en ongunstiger voorspellingen voor sommige mensen. En een kans: tussen de even goede modellen kan een ontwikkelaar zoeken naar modellen die minder leunen op features die mensen moeilijk of helemaal niet kunnen veranderen.

        **Rudin: kies er een die je kunt lezen.** Cynthia Rudin (2019) maakt van de vele goede modellen een argument. Als de data een grote set redelijk goede modellen toelaten, zit daar vaak minstens één model tussen dat je kunt lezen, en dat model is dan zowel te lezen als goed. Experts in het vak hebben dan misschien meer vrijheid om een model te kiezen dat zij kunnen lezen. Lesia Semenova, Cynthia Rudin en Ronald Parr (2022) brengen het als hypothese: als veel echte datasets grote Rashomon sets hebben, zoals zij vermoeden, dan kun je voor beslissingen waar veel van afhangt misschien vaak eenvoudige of leesbare modellen gebruiken zonder dat ze slechter voorspellen. Let op het "vaak" en het "misschien": Rudin heeft het over gestructureerde data met features die iets betekenen, waar algoritmes voor machine learning meestal ongeveer even goed presteren.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **Rashomon effect** | er zijn vaak veel verschillende modellen die ongeveer even goed voorspellen, en elk vertelt zijn eigen verhaal, bijvoorbeeld over welke features ertoe doen (Breiman) |
        | **Rashomon set** | alle modellen van één soort die hoogstens ε slechter zijn dan het beste; Marx en anderen noemen het de ε-level set |
        | **predictive multiplicity** | dat modellen uit de Rashomon set voor dezelfde mensen verschillende antwoorden geven |
        | **ambiguity** | het deel van de mensen bij wie minstens één model uit de set een ander antwoord geeft dan het gekozen model |
        | **discrepancy** | het grootste deel van de mensen van wie het antwoord verandert als je overstapt op één ander model uit de set |
        | ε | hoeveel slechter dan het beste een model mag zijn om nog mee te tellen; in deze les 0,01, één procentpunt |
        """),
        L.watch_out("Veel goede modellen hebben is niet voor iedereen goed nieuws. Rudin ziet er ruimte in om een model te kiezen dat je kunt lezen, merkt op dat het betekent dat een leermethode misschien niet stabiel is, en voegt eraan toe dat niet alle onderzoekers naar uitleg van modellen dit standpunt over de voordelen van instabiliteit delen. En hoeveel mensen een ander antwoord kunnen krijgen, is geen vast getal: het hangt af van de soort model, de data, ε, en hoe breed je zoekt."),
    ])
    return


@app.cell(hide_code=True)
def explore_eps_cell():
    explore_eps = L.explore("eps", "ε in procentpunten", lambda e: C.eps_picture(e, "nl"), start=1, values=C.STOPS,
                            role="hyper",
                            task="Schuif ε van 0 naar 5 procentpunten. Hoeveel bomen komen er bij elke stap bij in de set, en wat gebeurt er met het deel van de mensen dat een ander antwoord kan krijgen?")
    mo.vstack([
        mo.md("**De bomen binnen ε van de beste**"),
        explore_eps,
        mo.md(r"""
        De 50 bomen van het uitgewerkte voorbeeld, elk gegroeid op een eigen bootstrap-steekproef uit de ene helft van UCI Adult, en getest op de andere helft. ε is hoe ver een boom onder de beste mag liggen om nog als goed te tellen. Elke boom die erbij komt, kan sommige mensen een ander antwoord geven dan de beste boom.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo meet je hoeveel even goede modellen het oneens zijn:

    1. Verdeel de data in een deel om op te trainen en een deel om op te testen.
    2. Maak veel modellen van één soort, elk op een eigen bootstrap-steekproef uit het deel om op te trainen, en bewaar wat elk model zegt voor elke testpersoon en hoe vaak het goed zit.
    3. Neem het beste model, en de set modellen die hoogstens ε onder het beste liggen.
    4. Ambiguity: het deel van de testpersonen bij wie minstens één model uit de set iets anders zegt dan het beste. Discrepancy: het grootste deel dat verandert als je overstapt op één model uit de set.
    5. Zoek in de set een model dat je kunt lezen, en vergelijk het met het beste voordat je kiest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De drie kolommen uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), net als in de lessen hiervoor. `np.random.default_rng(0).permutation(n)` zet de plaatsen 0 tot en met n − 1 in een willekeurige volgorde; de eerste helft wordt de helft om op te trainen, de rest de helft om op te testen; `//` deelt en houdt het gehele deel, dus `len(_y) // 2` is de helft van het aantal mensen, naar beneden afgerond. Dan komen er bomen met diepte 1 tot en met 10, vijf van elke diepte, elk op een eigen bootstrap-steekproef uit de helft om op te trainen. `np.array` maakt van de lijst antwoorden een tabel met een rij per boom en een kolom per testpersoon. `_answers[_good]` houdt de rijen van de bomen in de set, en `!=` vergelijkt elk ervan met de rij van de beste boom. `.any(axis=0)` vraagt per kolom of er een rij `True` is: of minstens één boom deze persoon een ander antwoord geeft. `.mean(axis=1)` geeft het deel per boom. `np.argmax` op een rij met `True` en `False` geeft de plaats van de eerste `True`: de kleinste boom in de set, omdat de bomen van ondiep naar diep gemaakt worden; `export_text` print hem, zoals in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl). Als laatste een random forest op dezelfde helft om op te trainen, om te vergelijken: `n_estimators=100` is het aantal bomen, en `min_samples_leaf=20` zorgt dat elk blad minstens 20 mensen bevat. Voer de cel uit. Dat duurt een paar seconden.
    """)
    return


@app.cell
def example_trees():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 1. 48.842 mensen uit een Amerikaanse enquête van 1994 (UCI Adult), in twee helften
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]

    # 2. vijftig bomen, diepte 1 tot en met 10, vijf per diepte, elk op een eigen bootstrap-steekproef
    _rng = np.random.default_rng(1)
    _trees, _answers, _right, _depths = [], [], [], []
    for _depth in range(1, 11):
        for _k in range(5):
            _rows = _train[_rng.integers(0, len(_train), size=len(_train))]
            _tree = DecisionTreeClassifier(max_depth=_depth, random_state=0).fit(_X[_rows], _y[_rows])
            _trees.append(_tree)
            _answers.append(_tree.predict(_X[_test]))
            _right.append(np.mean(_answers[-1] == _y[_test]))
            _depths.append(_depth)
    _answers, _right, _depths = np.array(_answers), np.array(_right), np.array(_depths)

    # 3. de beste boom, en de set bomen die hoogstens één procentpunt onder hem liggen
    _best = np.argmax(_right)
    _good = _right >= _right[_best] - 0.01
    print("beste: diepte", _depths[_best], "goed", _right[_best].round(4))
    print("in de set:", _good.sum(), "bomen, diepte", _depths[_good].min(), "tot", _depths[_good].max())

    # 4. wie van een boom in de set een ander antwoord krijgt dan van de beste boom
    _differ = _answers[_good] != _answers[_best]
    print("ander antwoord van minstens één:", _differ.any(axis=0).mean().round(3))
    print("hoogstens veranderd door één boom:", _differ.mean(axis=1).max().round(3))

    # 5. de kleinste boom in de set
    print(export_text(_trees[np.argmax(_good)], feature_names=_names))

    # ter vergelijking: een random forest van 100 bomen op dezelfde helft om op te trainen
    _forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0)
    _forest.fit(_X[_train], _y[_train])
    print("forest goed", np.mean(_forest.predict(_X[_test]) == _y[_test]).round(4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De beste boom heeft diepte 5 en zit bij 79,8 procent van de testhelft goed; altijd nee zeggen zit bij 75,9 procent goed. 35 van de 50 bomen liggen hoogstens een procentpunt onder de beste, van diepte 2 tot diepte 9. Toch geeft minstens één ervan bij 20,6 procent van de 24.421 testpersonen een ander antwoord dan de beste boom, en verandert overstappen op één ervan het antwoord bij hoogstens 6,9 procent. De bomen vertellen ook verschillende verhalen, zoals Breiman zag: de drie bomen met diepte 2 in de set vragen alleen naar opleiding en leeftijd, nooit naar de uren, terwijl de andere 32 ook naar de uren vragen. De kleinste, hierboven geprint, zegt alleen ja tegen mensen met een opleiding boven 12,5 die ouder zijn dan 33,5, en zit bij 78,9 procent goed. Het random forest van 100 bomen zit bij 79,8 procent goed: niet beter dan de beste losse boom.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ambiguity():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Ambiguity met de hand

        Schrijf `ambiguity(base, others)`: het deel van de mensen bij wie minstens één van de andere modellen iets anders zegt dan het gekozen model. `base` bevat wat het gekozen model voor elke persoon zegt, en `others` bevat zo'n lijst voor elk ander model. De check roept eerst `ambiguity([1, 0, 0, 1, 0], [[1, 1, 0, 1, 0], [0, 1, 0, 1, 0]])` aan: vijf mensen en twee andere modellen. Daarna probeert hij andere mensen en modellen; de tekening voert je functie uit op de bomen van het uitgewerkte voorbeeld.
        """),
        L.task("Schrijf `ambiguity(base, others)`: tel elke persoon één keer als minstens één lijst in `others` daar van `base` verschilt, en geef het deel terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ambiguity():
    box_ambiguity = L.code_box("ambiguity", lines=12)
    box_ambiguity
    return (box_ambiguity,)


@app.cell(hide_code=True)
def check_ambiguity(box_ambiguity):
    result_ambiguity = L.check("ambiguity", box_ambiguity.value)
    result_ambiguity
    return (result_ambiguity,)


@app.cell(hide_code=True)
def hints_ambiguity():
    L.hints(
        "ambiguity",
        [
            "Ga de mensen langs. Zoek voor elke persoon uit of minstens één model iets anders zegt dan `base`, en tel de persoon dan één keer. Deel aan het eind door het aantal mensen.",
            "Loop over de mensen met `for i in range(len(base)):`. Begin voor elke persoon met `differs = False`, loop over de modellen met `for answers in others:`, en zet `differs = True` als `answers[i] != base[i]`. Tel na die binnenste loop 1 op bij `conflicts`, dat bij 0 begint, als `differs` `True` is.",
            "`return conflicts / ___`",
        ],
        why="`conflicts = 0`, dan `for i in range(len(base)):` met `differs = False`, een binnenste loop `for answers in others:` die `differs = True` zet als `answers[i] != base[i]`, en daarna `if differs:` met `conflicts = conflicts + 1`. Na beide loops `return conflicts / len(base)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_good():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De Rashomon set

        Het vak bevat hoe vaak elk van de 50 bomen van het uitgewerkte voorbeeld goed zit op de testhelft, afgerond, en de diepte van elke boom. Vul de twee `...` in: `best`, het hoogste aandeel in `right`, en binnen de loop een regel die de plaats `k` aan `good` toevoegt als boom `k` hoogstens `eps` onder de beste ligt. De check probeert ook een andere `eps`; de tekening laat je set zien.
        """),
        L.task("Vervang de twee `...`: het hoogste aandeel als `best`, en `good.append(k)` voor elke boom die hoogstens `eps` daaronder ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_good():
    box_good = L.code_box("good", lines=15)
    box_good
    return (box_good,)


@app.cell(hide_code=True)
def check_good(box_good):
    result_good = L.check("good", box_good.value)
    result_good
    return (result_good,)


@app.cell(hide_code=True)
def hints_good():
    L.hints(
        "good",
        [
            "Een boom hoort bij de set als hij hoogstens `eps` minder vaak goed zit dan de beste. Zoek eerst het hoogste aandeel, ga dan de bomen langs en houd de plaatsen van de bomen die slagen.",
            "`max(right)` geeft het hoogste aandeel. Een boom slaagt als zijn aandeel minstens `best` min `eps` is; voeg zijn plaats toe met `good.append(k)`.",
            "`if right[k] >= best - ___:`",
        ],
        why="`best = max(right)`, en binnen de loop `if right[k] >= best - eps:` met `good.append(k)` eronder.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ### Oefening 3. Welke boom?
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick",
        question="Je set bevat 35 bomen, van diepte 2 tot diepte 9, allemaal binnen een procentpunt van de beste. Welke boom zou Rudin kiezen voor beslissingen over leningen?",
        options={
            "best": "De beste, met diepte 5, omdat die het vaakst goed zit",
            "right": "Een boom met diepte 2, die je helemaal kunt lezen",
            "vote": "Alle 35, met een stemming",
        },
        correct="right",
        why={
            "best": "Zijn voorsprong is minder dan een procentpunt op de testhelft, en hij heeft 32 bladeren. Het argument van Rudin: als de set goede modellen groot is, zit er vaak een model in dat je kunt lezen, en bij beslissingen waar veel van afhangt neem je dat. Zie *Rudin: kies er een die je kunt lezen* in *Het idee*.",
            "right": "Goed. De bomen met diepte 2 zitten in de set: binnen een procentpunt van de beste, en je kunt er een helemaal lezen, vier bladeren, zoals in het uitgewerkte voorbeeld. Dat is het argument van Rudin in de praktijk. Of hij goed genoeg is, blijft een keuze van de mensen die hem gebruiken.",
            "vote": "Een stemming van bomen op bootstrap-steekproeven is bagging, de kern van een random forest: het forest uit het uitgewerkte voorbeeld voorspelt ongeveer even goed als de beste boom, en is veel moeilijker te volgen. Bij beslissingen waar veel van afhangt, pleit Rudin voor het model uit de set dat je kunt lezen. Zie *Rudin: kies er een die je kunt lezen* in *Het idee*.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_exercise_sure():
    mo.md(r"""
    ### Oefening 4. Wat het getal zegt
    """)
    return


@app.cell(hide_code=True)
def choice_sure_cell():
    choice_sure = L.choice(
        "sure",
        question="Met de 35 bomen is de ambiguity 20,6 procent: zoveel testpersonen kunnen van een ervan een ander antwoord krijgen dan van de beste boom. Wat zegt dat getal?",
        options={
            "right": "Minstens zoveel; breder zoeken kan er meer vinden",
            "exact": "Precies hoeveel mensen het raakt",
            "most": "Dat niet meer mensen dan dat een ander antwoord kunnen krijgen",
        },
        correct="right",
        why={
            "right": "Goed. Het is wat deze 35 bomen laten zien: een ondergrens. Marx en anderen vonden 10 procent door opnieuw te trainen, waar hun zoektocht door alle goede lineaire modellen 44 procent vond, en ze waarschuwen dat je met zulke aanpakken niet kunt aantonen dat er geen multiplicity is. Andere soorten modellen of andere steekproeven kunnen binnen dezelfde ε nog meer mensen een ander antwoord geven.",
            "exact": "Het telt alleen de 35 bomen die hier gemaakt zijn. Andere goede modellen, van een andere soort of uit andere steekproeven, kunnen nog meer mensen een ander antwoord geven. Zie *Breder zoeken vindt meer* in *Het idee*.",
            "most": "Andersom: het is een ondergrens. In het werk van Marx en anderen vond opnieuw trainen 10 procent, waar een zoektocht door alle goede lineaire modellen 44 procent vond. Zie *Breder zoeken vindt meer* in *Het idee*.",
        },
    )
    choice_sure
    return (choice_sure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Glass-box models](/course/open/rai-glass-box-models?lang=nl) was een boom met diepte 2 een model dat je met de hand kunt volgen; hier ligt zo'n boom binnen een procentpunt van de beste van 50 bomen, en dat laat het antwoord van Rudin zien op de vraag van die les: waarom een eenvoudig model vaak ongeveer even goed voorspelt als een ingewikkeld model. De bomen groeiden op bootstrap-steekproeven, zoals in [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl), waar de stemming van veel verschillende bomen het antwoord stabieler maakte. De volgende les, [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl), meet hoeveel een model op elke feature leunt; denk daar aan de waarschuwing van Breiman: even goede modellen kunnen verschillende verhalen vertellen over welke features ertoe doen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pick, choice_sure, result_ambiguity, result_good):
    L.wrap_up(result_ambiguity, result_good, choice_pick, choice_sure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
