# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-goodhart. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-goodhart", "nl")
    C = L.checks   # de wedstrijden van de course: opties met een ware waarde, een score met ruis, en een winnaar


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl) en [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaming():
    quick_gaming = L.choice(
        "quick-gaming",
        question="Op de racebaan van de course ging een agent met 3 punten per target heen en weer tussen twee targets, in plaats van naar de finish te gaan. Waarom?",
        options={
            "right": "De punten betaalden er meer voor dan voor de finish",
            "bug": "Een bug in de code van zijn Q-learning",
            "few": "Te weinig episodes, zodat hij de finish nooit heeft gevonden",
        },
        correct="right",
        why={
            "right": "Heen en weer leverde 2 punten per stap op, en de agent deed waar de punten voor betaalden: specification gaming. Deze les vraagt waarom een getal waar je op duwt niet meer meet wat het moest meten.",
            "bug": "Het leren werkte: value iteration, die helemaal niet hoeft te leren, gaf dezelfde weg. De punten betaalden voor heen en weer. Zie [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl).",
            "few": "In geen van zijn 500 episodes bereikte hij de finish, en value iteration, dat geen episodes nodig heeft, geeft dezelfde weg: de punten betalen voor heen en weer gaan. Zie [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl).",
        },
    )
    quick_gaming
    return (quick_gaming,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="Je maakt `rng = np.random.default_rng(3)` en vraagt `rng.random(size=2)`. Later maak je een nieuwe `np.random.default_rng(3)` en vraag je hetzelfde. Wat krijg je?",
        options={
            "new": "Twee nieuwe getallen, zoals willekeurige getallen horen te zijn",
            "right": "Dezelfde twee getallen als de eerste keer",
            "first": "Hetzelfde eerste getal, en dan een ander tweede",
        },
        correct="right",
        why={
            "right": "Een generator met dezelfde seed geeft dezelfde getallen, in dezelfde volgorde. De wedstrijden van de course in deze les zijn getrokken met seed 0, dus iedereen die ze draait, krijgt dezelfde.",
            "new": "De getallen lijken alleen willekeurig: ze volgen uit de seed. Dezelfde seed geeft dezelfde getallen. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "first": "De hele rij getallen volgt uit de seed, niet alleen het eerste. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In AI kies je vaak op een getal: het model met de laagste loss, de prompt met de beste score op een test, het antwoord dat een reward model het hoogst waardeert. Het getal staat voor wat je wilt, en zolang niemand erop duwt, meet het dat vaak goed. Kies erop, en kies hard, en het begint weg te drijven van wat het mat.

    Een deel van dat wegdrijven vraagt helemaal geen valsspelen, alleen ruis en een keuze. Deze les laat zien waarom, en laat je het meten in wedstrijden die de course trok.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Tien opties zijn allemaal even goed: elk heeft een ware waarde van 0. Je geeft elke optie een score met een meting die gemiddeld klopt, maar ruis heeft met een standaardafwijking van 1, en je kiest de optie met de hoogste score. Hoe ver ligt de score van de winnaar gemiddeld boven zijn ware waarde?",
        options={
            "zero": "Ongeveer 0, want de score klopt gemiddeld",
            "half": "Ongeveer 0,5",
            "right": "Ongeveer 1,5",
        },
        reveal="**Ongeveer 1,5.** James Smith en Robert Winkler (2006) rekenden het uit: met tien opties is het 154% van de standaardafwijking van de schattingen. De winnaar is de optie met de gelukkigste ruis. *Kiezen op een score met ruis* onder *Het idee* vertelt meer.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### De wet van Goodhart

    In 1975 schreef Charles Goodhart in *Problems of Monetary Management: The U.K. Experience* dat elke waargenomen statistische regelmaat de neiging heeft in te storten zodra er druk op komt om er iets mee te sturen. Goodhart schreef het over monetair beleid: een verband tussen getallen dat standhield zolang niemand erop stuurde, breekt meestal zodra iemand dat wel doet.

    De zin die de meeste mensen kennen, is niet van Goodhart. Marilyn Strathern schreef hem in 1997, in *'Improving ratings': audit in the British University system*: als een maatstaf een doel wordt, is hij geen goede maatstaf meer. De naam, de wet van Goodhart, schrijft Strathern toe aan Hoskin. Over audits zegt Strathern dat die meer doen dan toezien: ze krijgen een eigen leven, dat het leven waar ze toezicht op houden in gevaar brengt.

    Donald Campbell had ongeveer hetzelfde gezegd, in een paper uit 1976, waarvan in 1974 al een versie werd gepresenteerd: hoe meer een kwantitatieve sociale indicator wordt gebruikt om sociale beslissingen te nemen, hoe meer hij blootstaat aan druk om hem te vervalsen, en hoe eerder hij de sociale processen verstoort en bederft die hij moet volgen. In Texarkana, in de Verenigde Staten, zouden aannemers betaald worden naar hoeveel de testscores van individuele leerlingen stegen. Wat bleek: de aannemers leerden de leerlingen de antwoorden op de vragen die in de eindtest zouden komen. Sovjetfabrieken die op gewicht werden beoordeeld, las Campbell, maakten alleen nog hun zwaarste product, zoals de grootste spijkers; beoordeeld op het aantal stuks, alleen het makkelijkste, zoals de kleinste spijkers. Campbell noemde die aanwijzingen zelf overwegend anekdotisch.

    De **wet van Goodhart**, kort gezegd: duw hard op een maatstaf, en hij meet niet meer wat je wilde. Als een systeem een maatstaf opdrijft op een manier die niet opdrijft waar de maatstaf voor stond, heet dat **proxy gaming**: de aannemers speelden de test uit, de fabrieken het gewicht, en in de vorige les speelde de agent van de course de punten uit.

    ### Kiezen op een score met ruis

    Uitspelen vraagt iemand die uitspeelt. Eén deel van de wet van Goodhart vraagt helemaal niemand. In 2018 werkten David Manheim en Scott Garrabrant vier soorten van de wet van Goodhart uit, die Garrabrant in 2017 al had beschreven. De eerste is regressional Goodhart: wie kiest op een proxy, kiest niet alleen op het ware doel, maar ook op het verschil tussen de proxy en het doel. Zie een score als de ware waarde plus ruis. De optie met de hoogste score heeft waarschijnlijk een hoge ware waarde, maar had waarschijnlijk ook gelukkige ruis. Over deze soort schrijven ze dat het het eenvoudigste Goodhart-effect is, maar ook het meest fundamentele: je kunt er niet omheen.

    In 2006 hadden James Smith en Robert Winkler dit al uitgewerkt voor beslisanalyse, onder de naam optimizer's curse: neem je de geschatte waarden zoals ze zijn en kies je daarop, dan moet je verwachten dat de gekozen optie minder waard is dan de schatting, ook als de schattingen gemiddeld kloppen. Met opties die allemaal even goed zijn en ruis met een standaardafwijking van 1, ligt de score van de winnaar gemiddeld 0,85 boven zijn ware waarde bij drie opties, 1,03 bij vier, en 1,54 bij tien. De teleurstelling die je kunt verwachten, groeit met het aantal opties dat je vergelijkt.

    Dat maakt de winnaar geen slechte keuze. Als de opties echt verschillen, is de winnaar nog steeds naar verwachting beter dan een gemiddelde optie; Manheim en Garrabrant merken op dat een optie met een hoge score nog steeds naar verwachting een hogere ware waarde heeft dan andere. Alleen belooft zijn score te veel.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Vier soorten Goodhart

    De vier soorten van Manheim en Garrabrant, in het kort. Regressional: wie kiest op een onvolmaakte proxy, kiest onvermijdelijk ook op ruis. Extremal: kiezen op de maatstaf duwt de toestand naar een gebied waar de oude verbanden niet meer gelden. Causal: een ingreep van de regulator laat het verband instorten. Adversarial: iemand met andere doelen dan de regulator laat het verband instorten. De regulator is wie de maatstaf vaststelt. Extremal betekent: duw ver genoeg, en je komt in situaties die niet lijken op die waarin je zag dat de maatstaf het doel volgde. Onder adversarial plaatsen ze de wet van Campbell en het cobra-effect, genoemd naar een vermeende situatie in koloniaal India, waar de Britse overheid een beloning zou hebben uitgeloofd voor dode cobra's.

    ### Sceptisch zijn

    De uitweg van Smith en Winkler is: neem de scores niet zoals ze zijn. Stel dat je vooraf ongeveer weet waar de ware waarden liggen, rond een gemiddelde met een standaardafwijking die we de spreiding noemen, en hoe groot de ruis is. Meng dan elke score met dat gemiddelde:

    schatting = α · score + (1 − α) · gemiddelde, met α = 1 / (1 + ruis² / spreiding²).

    Bij weinig ruis ligt α dicht bij 1 en telt de score bijna helemaal mee; bij veel ruis blijft de schatting dicht bij het gemiddelde. Over zulke schattingen schrijven ze dat wie geschatte waarden opvat als een Bayesiaanse scepticus, gemiddeld niet teleurgesteld hoort te worden. Een Bayesiaanse scepticus mengt een nieuwe score met wat die vooraf verwachtte, zoals hier. In een studie bij Kodak uit 1999 met zeven opties namen ze aan dat het gemiddelde 0 was en de variantie van de ruis 20% van die van de spreiding, dus α = 1 / 1,2 = 0,8333, en de waarde van de aanbevolen optie ging van 80,0 naar 66,7 miljoen dollar.

    ### Reward models die te ver worden doorgeduwd

    In 2022 maten Leo Gao, John Schulman en Jacob Hilton van OpenAI de wet van Goodhart in language models. Een reward model geeft een antwoord een score voor hoe goed het is. Je kunt een policy trainen om hoge scores te halen, of veel antwoorden trekken en het antwoord met de hoogste score houden. Om ook de echte kwaliteit te kennen, lieten ze één reward model de plaats van mensen innemen: een vast reward model, de gouden standaard, speelt de rol van mensen en levert de labels waarmee een proxy-reward model wordt getraind. Het gouden reward model had 6B parameters, de proxy-reward models van 3M tot 3B, en de policy 1,2B; in een vergelijking ook 6B. Wat ze vonden: optimaliseer je tegen een geleerde proxy van de gouden reward, dan stijgt de gouden reward eerst en daalt hij later. Trek je n antwoorden en houd je het antwoord waaraan de proxy de hoogste score geeft, dan is dat de wedstrijd van deze les, met antwoorden als opties. Voor het beste van n antwoorden kozen ze de vorm van hun formule op tot 1000 antwoorden en testten die daarna, als voorspelling, op tot 60.000.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **wet van Goodhart** | duw hard op een maatstaf, en hij meet niet meer wat je wilde |
        | **proxy gaming** | een proxy opdrijven op een manier die niet opdrijft waar hij voor staat |
        | regressional Goodhart | wie kiest op een score met ruis, kiest ook gelukkige ruis |
        | optimizer's curse | de optie die je kiest, is naar verwachting minder waard dan de schatting, ook als de schattingen gemiddeld kloppen |
        | α (alpha) | het gewicht dat een sceptische schatting de score geeft: 1 / (1 + ruis² / spreiding²) |
        | reward model | een model dat een antwoord een score geeft voor hoe goed het is |
        | `C.TRUES`, `C.SCORES` | 2000 wedstrijden van 10 opties die echt verschillen: de ware waarden en de scores, een rij per wedstrijd |
        | `C.EQUAL_TRUES`, `C.EQUAL_SCORES` | 2000 wedstrijden van 10 opties die allemaal even goed zijn: elke ware waarde is 0 |
        """),
        L.watch_out("De winnaar van een wedstrijd met ruis is geen slechte keuze: als de opties echt verschillen, is hij meestal beter dan een gemiddelde optie, al is hij vaak niet de echt beste. Hij is slechter dan zijn score."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zoek uit hoeveel ruis je score heeft: geef dezelfde opties twee keer een score, of scoor ze op nieuwe vragen, en kijk hoeveel de scores bewegen.
    2. Verwacht dat de winnaar tegenvalt ten opzichte van zijn score, des te meer als je meer opties vergeleek of de score meer ruis heeft.
    3. Meet de winnaar opnieuw, op nieuwe data die niet meetelde bij het kiezen.
    4. Weet je ongeveer waar de ware waarden liggen, schuif elke score dan op naar dat gemiddelde voordat je vergelijkt, met α = 1 / (1 + ruis² / spreiding²).
    5. Wordt een getal snel beter onder druk, kijk dan waar het voor staat: werd het doel beter, of alleen het getal?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt wedstrijd 6 van de course uit `C.TRUES` en `C.SCORES`: 10 opties die echt verschillen. Hij zoekt de winnaar met `np.argmax`, de optie met de hoogste score, en vergelijkt de score van de winnaar met zijn ware waarde.
    """)
    return


@app.cell
def example_contest():
    true, score = C.TRUES[6], C.SCORES[6]   # wedstrijd 6: de ware waarden en de scores van 10 opties
    best = np.argmax(score)                  # de winnaar: de optie met de hoogste score
    print("winner: option", best)
    print("its score:", round(score[best], 2), " its true value:", round(true[best], 2))
    print("gap:", round(score[best] - true[best], 2))
    print("truly best: option", np.argmax(true))
    return


@app.cell(hide_code=True)
def contest_cell():
    mo.Html(C.contest_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De winnaar, optie 7, scoort 3,0 maar is 1,32 waard: zijn ruis gaf hem 1,68 extra. Twee opties zijn echt beter, en optie 2 is echt de beste. Eén wedstrijd kan geluk zijn; in oefening 1 neem je het gemiddelde verschil over alle 2000.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_average_gap():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. average_gap: hoe ver de score van de winnaar boven zijn ware waarde ligt

        Schrijf `average_gap(trues, scores)`: het gemiddelde, over alle wedstrijden, van hoe ver de score van de winnaar boven zijn ware waarde ligt. `trues` en `scores` zijn NumPy-arrays met een rij per wedstrijd en een kolom per optie. Houd de verschillen bij in een lijst `gaps`. Ga de wedstrijden langs met een `for`-loop over `range(len(scores))` die elk rijnummer als `i` neemt: zoek de winnaar `best` met `np.argmax` van de rij van `scores`, en voeg de score van de winnaar min zijn ware waarde toe aan `gaps`. Geef na de loop het gemiddelde terug met `np.mean`. Vervang de `...`. De check roept `average_gap(TRUES, SCORES)` aan, de 2000 wedstrijden van opties die echt verschillen, en daarna eigen wedstrijden.
        """),
        L.task("Neem in elke wedstrijd de winnaar op zijn score, en neem het gemiddelde van zijn score min zijn ware waarde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average_gap():
    box_average_gap = L.code_box("average_gap", lines=9)
    box_average_gap
    return (box_average_gap,)


@app.cell(hide_code=True)
def check_average_gap(box_average_gap):
    result_average_gap = L.check("average_gap", box_average_gap.value)
    result_average_gap
    return (result_average_gap,)


@app.cell(hide_code=True)
def hints_average_gap():
    L.hints(
        "average_gap",
        [
            "Het *Uitgewerkt voorbeeld* zoekt de winnaar van één wedstrijd en zijn verschil. Doe hetzelfde voor elke rij, en neem het gemiddelde.",
            "`gaps` begint als een lege lijst. In de loop is `best` de index van het hoogste getal in `scores[i]`; het verschil is het getal van de winnaar in `scores[i]` min zijn getal in `trues[i]`, allebei bij index `best`. Na de loop: `return np.mean(gaps)`.",
            "`        gaps.append(___)`",
        ],
        why="`gaps = []`, dan `for i in range(len(scores)):` met daarin `best = np.argmax(scores[i])` en `gaps.append(scores[i][best] - trues[i][best])`; na de loop `return np.mean(gaps)`. `average_gap(TRUES, SCORES)` geeft ongeveer 1,11.",
    )
    return


@app.cell(hide_code=True)
def text_after_average_gap():
    mo.md(r"""
    Met opties die echt verschillen, ligt de score van de winnaar gemiddeld 1,11 boven zijn ware waarde; met opties die allemaal even goed zijn, 1,56, dicht bij de 1,54 van Smith en Winkler. Over alle opties, niet alleen de winnaars, is het verschil zo goed als 0: elke score klopt gemiddeld. Het kiezen maakt de score van de winnaar te hoog.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: meer opties, een groter verschil

    Voor negen aantallen opties van 1 tot 1000 trok de course telkens 2000 wedstrijden: één keer met opties die allemaal even goed zijn, en één keer met opties die echt verschillen, met ware waarden die een standaardafwijking van 1 hebben. De schuif zet het aantal opties; de balken tonen de score van de winnaar, zijn ware waarde en het verschil, gemiddeld.
    """)
    return


@app.cell(hide_code=True)
def demo_options():
    options = L.explore("options", "opties in een wedstrijd", lambda _v: C.demo_frame(_v, L.lang), start=10,
                        values=C.NS, role="",
                        task="Zoek het kleinste aantal op de schuif waarbij het verschil, met opties die allemaal even goed zijn, groter is dan 2.")
    options
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Het verschil groeit met elke optie die je toevoegt, maar steeds langzamer: met opties die allemaal even goed zijn 1,56 bij 10 opties, 2,50 bij 100 en 3,24 bij 1000. Als de opties echt verschillen, groeit ook de ware waarde van de winnaar, van 1,08 bij 10 opties tot 2,30 bij 1000: harder kiezen kiest nog steeds beter, alleen belooft de score meer dan dat. Dit doet ruis alleen. Gao et al. zeggen hetzelfde: als dit de enige soort Goodhart is, moet de gouden reward altijd stijgen als de proxy-reward stijgt. Hun gouden reward daalde na een tijd wel, dus was de ruis niet van deze eenvoudige soort, of waren er ook andere soorten aan het werk; Gao et al. verwachten dat vooral extremal Goodhart daarachter zit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_skeptical():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. skeptical: een score die je niet zomaar gelooft

        Schrijf `skeptical(scores, mean, spread, noise)`: de sceptische schattingen van Smith en Winkler voor een NumPy-array `scores`, als de ware waarden rond `mean` liggen met standaardafwijking `spread`, en de ruis standaardafwijking `noise` heeft. Reken eerst het gewicht `alpha` uit met de formule onder *Sceptisch zijn* in *Het idee*. Geef dan `alpha` keer `scores` plus `1 - alpha` keer `mean` terug. Vervang de `...`. De check roept `skeptical(SCORES[6], 0.0, 1.0, 1.0)` aan, de scores van wedstrijd 6 in de opzet van de course, en daarna eigen opzetten.
        """),
        L.task("Reken alpha uit met de ruis en de spreiding, en meng de scores met het gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_skeptical():
    box_skeptical = L.code_box("skeptical", lines=3)
    box_skeptical
    return (box_skeptical,)


@app.cell(hide_code=True)
def check_skeptical(box_skeptical):
    result_skeptical = L.check("skeptical", box_skeptical.value)
    result_skeptical
    return (result_skeptical,)


@app.cell(hide_code=True)
def hints_skeptical():
    L.hints(
        "skeptical",
        [
            "*Sceptisch zijn* onder *Het idee* geeft de formule voor α en voor de schatting.",
            "In Python is een kwadraat `**2`, dus ruis² is `noise**2`. `alpha` is 1 gedeeld door 1 plus de variantie van de ruis gedeeld door die van de spreiding. Daarna `return alpha * scores + (1 - alpha) * mean`.",
            "`    alpha = 1 / (1 + ___)`",
        ],
        why="`alpha = 1 / (1 + noise**2 / spread**2)` en `return alpha * scores + (1 - alpha) * mean`. Met `noise` en `spread` allebei 1 is `alpha` 0,5: elke score schuift halverwege op naar het gemiddelde.",
    )
    return


@app.cell(hide_code=True)
def text_after_skeptical():
    mo.md(r"""
    Met evenveel ruis als spreiding is α = 0,5: de sceptische schatting halveert de afstand van elke score tot het gemiddelde. In de wedstrijden van de course scoren de winnaars gemiddeld 2,19, komen hun sceptische schattingen uit op 1,10 en hun ware waarden op 1,08: de scepticus wordt gemiddeld niet teleurgesteld. De prijs is wat je zelden precies weet: waar de ware waarden liggen, en hoe groot de ruis is. En als sommige opties moeilijker te schatten zijn dan andere, kan opschuiven veranderen welke optie als beste uitkomt: Smith en Winkler laten met de studie bij Kodak zien dat een optie met een hoge maar onzekere schatting het kan afleggen tegen een optie met een lagere, preciezere schatting.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prompts():
    mo.md(r"""
    ### Oefening 3. De beste van 50 prompts

    Een team probeert 50 prompts op dezelfde 20 vragen, met een score van een judge, en houdt de prompt met de hoogste score: 85%. De prompts verschillen echt in kwaliteit. Daarna draait het team de winnende prompt op 200 nieuwe vragen.
    """)
    return


@app.cell(hide_code=True)
def choice_prompts_cell():
    choice_prompts = L.choice(
        "prompts",
        question="Wat verwacht je op de nieuwe vragen?",
        options={
            "same": "Ongeveer 85%, want de score klopte gemiddeld",
            "right": "Onder 85%, maar boven een gemiddelde prompt",
            "worse": "Onder een gemiddelde prompt, want de winnaar had alleen geluk",
        },
        correct="right",
        why={
            "right": "De winnaar is gekozen op 20 vragen, dus zijn score bevat gelukkige ruis: op nieuwe vragen is dat geluk weg. Maar de prompts verschillen echt, dus wie op de hoogste score koos, koos ook een goede prompt: hij blijft boven het gemiddelde, zoals de winnaar in de wedstrijden van de course.",
            "same": "Elke losse score klopt gemiddeld, maar die van de winnaar niet: hij is gekozen omdat hij hoog was, en een deel daarvan was ruis. Zie *Kiezen op een score met ruis* onder *Het idee*.",
            "worse": "Als de opties echt verschillen, is de winnaar naar verwachting nog steeds beter dan een gemiddelde optie; alleen belooft zijn score te veel. In de wedstrijden van de course was de winnaar van 10 opties 1,08 waard, tegen 0 voor een gemiddelde optie. Zie *Kiezen op een score met ruis* onder *Het idee*.",
        },
    )
    choice_prompts
    return (choice_prompts,)


@app.cell(hide_code=True)
def text_exercise_fraud():
    mo.md(r"""
    ### Oefening 4. Een fraudemodel in gebruik

    Een bank test een model dat frauduleuze betalingen markeert op de gevallen van vorig jaar: het vangt 95% van de fraude. Zodra het model in gebruik is, proberen fraudeurs uit welke betalingen het doorlaat, en passen ze hun werkwijze aan. Het deel van de fraude dat het model vangt, daalt.
    """)
    return


@app.cell(hide_code=True)
def choice_fraud_cell():
    choice_fraud = L.choice(
        "fraud",
        question="Welke van de vier soorten Goodhart past het best?",
        options={
            "regressional": "Regressional: het model is gekozen op gelukkige ruis bij de gevallen van vorig jaar",
            "right": "Adversarial: mensen met eigen doelen reageren op het model",
            "extremal": "Extremal: het model is naar extreme scores geduwd",
        },
        correct="right",
        why={
            "right": "Het deel daalde omdat anderen, met eigen doelen, hun gedrag veranderden zodra het model tegen hen werd ingezet. Manheim en Garrabrant noemen dat adversarial: iemand met andere doelen dan de regulator laat het verband instorten.",
            "regressional": "Gelukkige ruis zou de 95% vanaf het begin te hoog maken, en het zou niet blijven dalen terwijl de fraudeurs leren. Hier komt de daling van wat de fraudeurs doen. Zie *Vier soorten Goodhart* onder *Het idee*.",
            "extremal": "Niemand duwde de scores van het model naar extremen; de fraudeurs veranderden hun werkwijze. Goodhart door anderen met eigen doelen is adversarial. Zie *Vier soorten Goodhart* onder *Het idee*.",
        },
    )
    choice_fraud
    return (choice_fraud,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl) liet agents zien die hun reward uitspelen; deze les gaf het patroon een naam, de wet van Goodhart, en liet een deel ervan zien dat helemaal geen uitspelen vraagt: wie het beste kiest op een score met ruis, kiest ook gelukkige ruis, des te meer naarmate je meer opties vergelijkt. De sceptische schatting van Smith en Winkler haalt dat geluk er gemiddeld weer uit, en Gao et al. zagen de echte reward van een language model eerst stijgen en dan dalen toen het tegen een proxy werd doorgeduwd. De volgende les, *Outer en inner alignment*, splitst het probleem in tweeën: zegt de doelfunctie wat we willen, en streeft het getrainde model die doelfunctie na?

    **Een voorbeeld uit het dagelijks leven.** Het restaurant met de beste recensies van de vijftig in de stad valt vaak een beetje tegen als je er gaat eten: een deel van zijn voorsprong was geluk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fraud, choice_prompts, result_average_gap, result_skeptical):
    L.wrap_up(result_average_gap, result_skeptical, choice_prompts, choice_fraud)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
