# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ml-threat-models. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-ml-threat-models", "nl")
    C = L.checks   # een klein model dat de course zelf traint: stopborden en snelheidsborden van 8 bij 8 pixels


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) en [De workflow van machine learning](/course/open/ml-workflow?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cia():
    quick_cia = L.choice(
        "quick-cia",
        question="Iemand breekt in op de database van een navigatie-app en zet de maximumsnelheid van een paar wegen hoger. Welke eigenschap gaat daarmee verloren?",
        options={
            "right": "Integriteit: iemand verandert gegevens zonder dat te mogen",
            "conf": "Vertrouwelijkheid: iemand leest wat die niet mag lezen",
            "avail": "Beschikbaarheid: weggebruikers kunnen de app niet meer gebruiken",
        },
        correct="right",
        why={
            "right": "Iets veranderen waar je geen recht op hebt, is een verlies van integriteit. Dat de indringer überhaupt binnenkwam, is ook een probleem voor de security, maar wat er misgaat voor de weggebruikers, is dat de gegevens niet meer kloppen.",
            "conf": "De maximumsnelheden zijn niet geheim. Wat er misgaat, is dat ze veranderd zijn. Zie *Drie dingen om te beschermen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
            "avail": "De app blijft voor iedereen werken, met de verkeerde snelheden. Zie *Drie dingen om te beschermen* in [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl).",
        },
    )
    quick_cia
    return (quick_cia,)


@app.cell(hide_code=True)
def quick_check_workflow():
    quick_workflow = L.choice(
        "quick-workflow",
        question="Wanneer meet je in de workflow van machine learning het model op de test set?",
        options={
            "right": "Eén keer, aan het eind, als alles gekozen is",
            "round": "In elke ronde, om te kiezen wat je nu verandert",
            "start": "Aan het begin, om het doel te bepalen",
        },
        correct="right",
        why={
            "right": "De test set wacht tot het eind: kies je ermee, dan buigt de score ernaartoe.",
            "round": "Kiezen met de test set maakt de score te mooi; in de rondes gebruik je de validation set. Zie *Recept* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "start": "Het doel bepaal je voordat je uitkomsten ziet, en zonder de test set. Zie *Recept* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
        },
    )
    quick_workflow
    return (quick_workflow,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe liepen de aanvallen in deze module via wat er rond een programma zit: de accounts, de database, de code. Ook een model kun je aanvallen, en wel via waar het voor gemaakt is: antwoorden geven. Een aanvaller kan een input een beetje veranderen, zodat het model zich vergist, of vraag na vraag stellen, tot de antwoorden verraden waar het model van leerde. Deze les deelt die aanvallen in, zoals het Amerikaanse instituut voor standaarden NIST dat doet, en voert er dan zelf een uit: je maakt van een stopbord een snelheidsbord, voor een klein model, eerst met de weights en daarna zonder.

    **Een voorbeeld uit het dagelijks leven.** Een spammail schrijft *V1agra* in plaats van *Viagra*. Wie de mail leest, ziet hetzelfde woord; een filter dat op het woord let, ziet een ander woord. De mail is een beetje veranderd, en alleen voor het filter.
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Goodfellow en collega's lieten GoogLeNet, een neural network, een foto van een panda zien. Dat is een veel groter model dan die tot nu toe, van een soort die latere lessen bouwen. Panda, zei het netwerk, met 57,7% zekerheid. Daarna schoven ze elke pixel omhoog of omlaag, met de kleinste stap die een 8-bitsafbeelding kent, in een richting die ze uit het netwerk afleidden. Wat zei het netwerk over de nieuwe foto?",
        options={
            "same": "Panda, met iets minder zekerheid",
            "right": "Gibbon, met 99,3% zekerheid",
            "unsure": "Twijfel: geen klasse boven 50%",
        },
        reveal="**Gibbon, met 99,3% zekerheid.** Volgens de auteurs is hun ε van 0,007 zo groot als de kleinste stap in een 8-bitsafbeelding, nadat GoogLeNet de pixels in reële getallen heeft omgezet. Kreeg het netwerk alleen het patroon van tekens te zien, dan zei het nematode, met 8,2% zekerheid. Een jaar eerder hadden Szegedy en collega's zulke inputs een naam gegeven: *adversarial examples*.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat aanvallers van een model willen

    Het rapport *Adversarial Machine Learning* van NIST (NIST AI 100-2 E2025, maart 2025) deelt aanvallen op machine learning in langs vier assen: de leermethode en de fase van het leren waarin de aanval plaatsvindt, de doelen van de aanvaller, wat de aanvaller kan, en wat de aanvaller weet van het leren. Voor modellen die voorspellen, zoals een classifier, noemt het drie doelen: beschikbaarheid laten uitvallen (*availability breakdown*), integriteit schenden (*integrity violation*) en privacy aantasten (*privacy compromise*). Dat zijn de drie eigenschappen uit [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl), met privacy op de plek van vertrouwelijkheid. NIST merkt op dat in de gewone security *confidentiality* het gangbare woord is, maar dat het vakgebied AML meestal *privacy* zegt. AML is *adversarial machine learning*, het onderzoek naar aanvallen op machine learning en verdedigingen ertegen.

    - **Beschikbaarheid laten uitvallen**: de aanvaller zorgt dat andere gebruikers of processen het model niet meer op tijd en betrouwbaar kunnen gebruiken.
    - **Integriteit schenden**: de aanvaller dwingt het model om af te wijken van waar het voor bedoeld is, en voorspellingen te geven die de aanvaller goed uitkomen.
    - **Privacy aantasten**: afgeschermde of bedrijfseigen informatie lekt onbedoeld uit het model, zoals details over de training data, de weights of de architectuur (hoe het model in elkaar zit).

    Voor generatieve AI voegt NIST een vierde doel toe: misbruik mogelijk maken (*misuse enablement*). De aanvaller probeert met opzet de technische beperkingen te omzeilen die de eigenaar van het systeem aan het gebruik ervan heeft gesteld. Daarover gaat *Jailbreaks*, verderop in deze module.

    ### Wanneer: terwijl het leert, of als het gebruikt wordt

    Aan de fase herken je twee families. Een poisoning-aanval slaat toe terwijl een model leert: de aanvaller smokkelt iets in de training data. Een **evasion attack** slaat toe als het model gebruikt wordt: de aanvaller verandert een input. NIST zet evasion onder integriteit. Poisoning kan twee van die doelen dienen: data poisoning die het model als geheel bederft, staat onder beschikbaarheid, en *targeted* en *backdoor poisoning*, waardoor het model zich vergist bij inputs die de aanvaller kiest, onder integriteit. Over de eerste familie gaat *De training data vergiftigen*, verderop in deze module; deze les gaat over de tweede.

    ### Wat de aanvaller weet

    Bij een **white-box**-aanval weet de aanvaller volgens NIST alles van het ML-systeem, ook de training data, de architectuur van het model en de hyperparameters. Bij een **black-box**-aanval weet de aanvaller weinig, en soms helemaal niets. Daartussen liggen gray-box-aanvallen. Black box betekent niet machteloos: NIST noemt die aanvallen juist het meest praktisch, omdat ze geen kennis van het systeem nodig hebben en gebruikmaken van wat er voor gewoon gebruik al openstaat. Iedereen die het model een vraag kan stellen, kan het proberen.

    ### Adversarial examples

    Bij een evasion attack wil de aanvaller volgens NIST *adversarial examples* maken: inputs waarvan de aanvaller de indeling kan veranderen in elke klasse die de aanvaller wil, vaak met maar een heel kleine verandering. Het idee is ouder dan deep learning: al in 1988 waren er volgens NIST evasion attacks, en in 2004 adversarial examples voor lineaire classifiers in spamfilters. Szegedy en collega's (2014) ontdekten dat je bij neural networks met een onzichtbare, niet-willekeurige verandering van een testafbeelding elke voorspelling van het netwerk kunt afdwingen die je wilt.

    ### De fast gradient sign method

    Goodfellow, Shlens en Szegedy (2015) gaven een snelle manier om er een te maken: schuif elke waarde van de input even ver op, een klein stukje ε, in de richting waarin de loss van het juiste antwoord stijgt. In hun notatie is de verandering η = ε · sign(∇ₓJ(θ, x, y)): ∇ₓJ is de gradient van de loss J naar de input x, niet naar de parameters θ, y is het juiste antwoord, en sign houdt van elke helling alleen het teken over, +1 of −1. Ze noemden het de *fast gradient sign method*. Bij een logistische regressie is het eenvoudig. Maak je pixel j één eenheid hoger, dan verandert de logit met de weight wⱼ, en bij een stopbord betekent een hogere logit een hogere loss. De richting van pixel j is dus het teken van wⱼ, en de logit stijgt met ε keer de som van de groottes van alle weights, ε · Σ|wⱼ|. Goodfellow en collega's wijzen hetzelfde gedrag aan als oorzaak bij neural networks: dat ze zo makkelijk te misleiden zijn, komt volgens hen vooral doordat ze zich lineair gedragen.

    ### Buiten de computer

    Eykholt en collega's (2018) vielen een echt stopbord aan, met alleen zwarte en witte stickers. Het model dat ze aanvielen, las het bord als een snelheidsbord in 100% van de foto's in het lab, en in 84,8% van de videobeelden vanuit een rijdende auto. Dat model kenden ze door en door: ze werkten in de white-box-situatie. Echte aanvallers kunnen zich met veel minder redden. Bij een detector van phishingpagina's, meldt NIST, vonden onderzoekers 100 adversarial examples tussen 4600 voorbeelden waarover de detector twijfelde, gemaakt met vrij eenvoudige trucs: afbeeldingen bijsnijden, deels afdekken of vervagen. MITRE houdt een openbare verzameling van zulke aanvallen bij, ATLAS.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **evasion attack** | een aanval terwijl een model gebruikt wordt: een veranderde input laat het een verkeerd antwoord geven |
        | **adversarial example** | een input die met opzet is veranderd, vaak heel weinig, zodat een model zich vergist |
        | **white box** | de aanvaller weet alles van het model: weights, architectuur, training data |
        | **black box** | de aanvaller weet weinig of niets van het model, en kan het meestal alleen vragen stellen |
        | **FGSM** | de fast gradient sign method: elke waarde van de input schuift ε op, in de richting van het teken van de gradient |
        | **ε** | hier: hoe ver FGSM elke waarde van de input opschuift; `eps` in code |
        | `np.sign(a)` | het teken van elk getal in de array `a`: 1, −1, of 0 bij 0 |
        """),
        L.watch_out("Black box betekent niet dat de aanvaller niets kan: NIST noemt black-box-aanvallen het meest praktisch. En de stickers misleidden het model niet altijd: 100% was in het lab; vanuit een rijdende auto 84,8% van de videobeelden, bij één model dat de onderzoekers volledig kenden."),
        mo.Html(C.idea_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo misleid je een classifier met een kleine verandering van een input, langs de gradient:

    1. Kies de input, en het antwoord dat het model moet geven.
    2. Zoek voor elke waarde van de input de richting die het model naar dat antwoord duwt: het teken van de gradient van de loss van het juiste antwoord. Bij onze logistische regressie, waar we een snelheidsbord willen (een hogere logit), is dat het teken van elke weight.
    3. Schuif elke waarde van de input even ver op, een kleine ε, in die richting.
    4. Reken de kleinste ε uit die het antwoord laat omslaan, en controleer dat het model nu het antwoord geeft dat je wilde.
    5. Zonder de weights vind je de richtingen met vragen: verander één waarde van de input een beetje, en kijk welke kant het antwoord op gaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course trainde een logistische regressie op verzonnen plaatjes van 8 bij 8 pixels, elke pixel een getal tussen 0 en 1 (in de plaatjes steekt een cel meer af tegen de pagina naarmate het getal hoger is): 100 stopborden en 100 snelheidsborden. Het model heeft 64 weights `C.W`, één voor elke pixel, en een bias term `C.B`; de kans die het geeft, is die op een snelheidsbord. Op 200 nieuwe plaatjes vergist het zich nooit. `C.STOP` is een stopbord dat het model niet eerder zag. De cel schuift elke pixel van `C.STOP` 0,1 op, één keer willekeurig omhoog of omlaag (`C.RANDOM_SIGNS` bevat 64 keer 1 of −1, willekeurig gekozen) en één keer langs de tekens van de weights, en rekent elke keer de logit en de kans uit.
    """)
    return


@app.cell
def example_steps():
    # het model van de course: 64 weights en een bias term, voor plaatjes van 8 bij 8 pixels
    w, b = C.W, C.B
    z = np.dot(w, C.STOP) + b
    print("STOP: logit", round(z, 3), "P(speed limit)", round(1 / (1 + np.exp(-z)), 3))
    # elke pixel 0,1 omhoog of omlaag: willekeurig, of langs de tekens van de weights
    for name, step in [("random", 0.1 * C.RANDOM_SIGNS), ("along the signs", 0.1 * np.sign(w))]:
        z = np.dot(w, C.STOP + step) + b
        print(name + ": logit", round(z, 3), "P(speed limit)", round(1 / (1 + np.exp(-z)), 3))
    print("sum of the sizes of the weights:", round(np.sum(abs(w)), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    STOP heeft een logit van −3,845: een snelheidsbord met kans 0,021. Met willekeurige richtingen heffen de stappen van 0,1 elkaar grotendeels op; hier gaat de logit zelfs iets omlaag, naar −4,209. Langs de tekens duwt elke stap de logit omhoog: met 0,1 keer de grootte van de weight, samen 0,1 × 18,18 = 1,818, naar −2,027. Nog steeds een stopbord, met kans 0,116 op een snelheidsbord. Stappen van 0,1 zijn niet genoeg.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fgsm():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. fgsm: één stap langs de tekens

        Schrijf `fgsm(image, w, eps)`: het plaatje `image` met elke pixel `eps` opgeschoven, omhoog waar de weight in `w` positief is en omlaag waar die negatief is. Vervang de `...`. De check roept `fgsm(STOP, W, 0.1)` aan, met het plaatje en de weights uit het uitgewerkte voorbeeld, en daarna andere plaatjes, andere stappen en een tweede model. fgsm kan een pixel onder 0 of boven 1 duwen; een echte aanval zou elke pixel ook tussen 0 en 1 houden, en de check keurt allebei goed.
        """),
        L.task("Schuif elke pixel van image eps op, in de richting van het teken van de weight van die pixel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fgsm():
    box_fgsm = L.code_box("fgsm", lines=3)
    box_fgsm
    return (box_fgsm,)


@app.cell(hide_code=True)
def check_fgsm(box_fgsm):
    result_fgsm = L.check("fgsm", box_fgsm.value)
    result_fgsm
    return (result_fgsm,)


@app.cell(hide_code=True)
def hints_fgsm():
    L.hints(
        "fgsm",
        [
            "Stap 3 van het recept, zoals in de laatste regels van het uitgewerkte voorbeeld.",
            "Neem de tekens van de weights met `np.sign`, vermenigvuldig ze met `eps`, en tel de uitkomst bij het plaatje op.",
            "`return image + eps * ___`",
        ],
        why="`return image + eps * np.sign(w)`. `fgsm(STOP, W, 0.1)` schuift elke pixel 0,1 op langs de tekens: het model leest nog steeds een stopbord, met kans 0,116 op een snelheidsbord.",
    )
    return


@app.cell(hide_code=True)
def text_after_fgsm():
    mo.md(r"""
    Met eps = 0,25 leest het model een snelheidsbord, met kans 0,668 (0,623 als je fgsm elke pixel tussen 0 en 1 houdt). Je fgsm vroeg nergens wat er op het plaatje staat: de richting komt alleen van het model, en is voor elk stopbord dezelfde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_eps():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. smallest_eps: hoe ver precies?

        Het model leest een snelheidsbord zodra de logit boven 0 komt: de decision boundary uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl). Eén stap fgsm met eps verhoogt de logit met eps keer de som van de groottes van de weights. Schrijf `smallest_eps(image, w, b)`: de eps waarbij de logit van `fgsm(image, w, eps)` precies 0 is. `abs` werkt ook op een array: het geeft de grootte van elk getal erin. Vervang de `...`. De check roept `smallest_eps(STOP, W, B)` aan, en daarna een ander plaatje en het tweede model.
        """),
        L.task("Geef de eps terug die de logit van het model precies op 0 brengt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_eps():
    box_eps = L.code_box("smallest-eps", lines=4)
    box_eps
    return (box_eps,)


@app.cell(hide_code=True)
def check_eps(box_eps):
    result_eps = L.check("smallest-eps", box_eps.value)
    result_eps
    return (result_eps,)


@app.cell(hide_code=True)
def hints_eps():
    L.hints(
        "smallest-eps",
        [
            "Stap 4 van het recept. In het uitgewerkte voorbeeld verhoogden stappen van 0,1 de logit met 0,1 keer de som van de groottes van de weights.",
            "Reken de logit `z` van `image` uit met `np.dot`, `w` en `b`. Een stap met eps verhoogt die met eps keer de som van de groottes van de weights. Bij welke eps komt `z` precies op 0?",
            "`z = np.dot(w, image) + b`, en dan `return -z / ___`",
        ],
        why="`z = np.dot(w, image) + b` en `return -z / np.sum(abs(w))`. Voor STOP: 3,845 / 18,18, dus 0,212 per pixel.",
    )
    return


@app.cell(hide_code=True)
def text_after_eps():
    mo.md(r"""
    STOP heeft 0,212 per pixel nodig, en de andere 99 stopborden uit de test set tussen 0,172 en 0,257. Op een schaal van 0 tot 1 is dat veel: in het plaatje bij *Het idee*, met 0,22, zie je de verandering. De foto van de panda had maar de kleinste stap van een 8-bitsafbeelding nodig. Ons model telt maar 64 pixels op; een foto voor GoogLeNet heeft veel meer waarden in de input, dus kunnen veel meer kleine stappen optellen. Goodfellow en collega's zagen de belangrijkste oorzaak van deze zwakte van neural networks in hun lineaire gedrag.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Langs de tekens, of willekeurig

    Schuif eps van 0 naar 0,3. Het plaatje schuift elke pixel van STOP twee keer eps op: langs de tekens van de weights, zoals je fgsm doet, en willekeurig omhoog of omlaag.
    """)
    return


@app.cell(hide_code=True)
def try_eps_cell():
    try_eps = mo.ui.slider(start=0, stop=0.3, step=0.01, value=0.15, label="eps")
    try_eps
    return (try_eps,)


@app.cell(hide_code=True)
def demo_eps(try_eps):
    _out = mo.Html(C.demo_picture(try_eps.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Langs de tekens slaat het model om tussen 0,21 en 0,22. Met willekeurige richtingen slaat het zelfs bij 0,3 niet om: daar is het juist zekerder van het stopbord, met kans 0,007 op een snelheidsbord. Even veel verandering, in een andere richting, levert de aanvaller niets op.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_signs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. guess_signs: dezelfde richting, zonder de weights

        Tot nu toe kende je de weights: een white box. Achter een website ziet een aanvaller alleen de antwoorden: een black box. `ask(picture)` antwoordt zoals zo'n model: de kans op een snelheidsbord, verder niets. Schrijf `guess_signs(ask, image, delta)`: vraag één keer naar `image`, en dan voor elke pixel naar een kopie van `image` waarin alleen die pixel `delta` hoger is. Bewaar 1 als het antwoord omhoog gaat en −1 als dat niet zo is, en geef de 64 tekens terug met `np.array`. Een kopie van het plaatje is `np.array(image)`. Vervang de `...`. De check roept `guess_signs(ask, STOP, 0.01)` aan, en daarna een ander plaatje en een tweede model.
        """),
        L.task("Vind het teken van elke weight uit alleen de antwoorden van ask."),
    ])
    return


@app.cell(hide_code=True)
def exercise_signs():
    box_signs = L.code_box("signs", lines=12)
    box_signs
    return (box_signs,)


@app.cell(hide_code=True)
def check_signs(box_signs):
    result_signs = L.check("signs", box_signs.value)
    result_signs
    return (result_signs,)


@app.cell(hide_code=True)
def hints_signs():
    L.hints(
        "signs",
        [
            "Stap 5 van het recept: één vraag over het plaatje, en dan één per pixel, met een loop zoals in [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "Bewaar het antwoord voor `image` zelf. Maak dan voor elke `i` in `range(len(image))` een kopie met `np.array(image)`, tel `delta` op bij pixel `i` van die kopie, vraag naar de kopie, en voeg 1 of −1 toe aan een lijst.",
            "`if ask(bumped) > ___:`, met `bumped = np.array(image)` en `bumped[i] = bumped[i] + delta` erboven, in de loop",
        ],
        why="`base = ask(image)` en `signs = []`; dan voor elke `i` in `range(len(image))`: `bumped = np.array(image)`, `bumped[i] = bumped[i] + delta`, en `signs.append(1)` als `ask(bumped) > base`, anders `signs.append(-1)`; aan het eind `return np.array(signs)`. Voor STOP kloppen alle 64 tekens na 65 vragen.",
    )
    return


@app.cell(hide_code=True)
def text_after_signs():
    mo.md(r"""
    Met 65 vragen heeft de aanvaller dezelfde richting als met de weights, en werkt fgsm net als eerst. Onze `ask` geeft een kans, dus zelfs een kleine verandering is te zien in het antwoord. NIST noemt black-box-aanvallen die zuinig zijn met hun vragen en er doorgaans minder dan 1000 nodig hebben. Een aanvaller die niet vraag na vraag kan stellen, heeft nog een weg: Szegedy en collega's ontdekten dat dezelfde verandering ook een ander netwerk kan misleiden, dat op een ander deel van de data getraind is. Een aanvaller kan de verandering dus maken op een eigen model, en die op het echte model uitproberen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_goal():
    mo.md(r"""
    ### Oefening 4. Welk doel?

    Een aanvaller stuurt een model van een ziekenhuis duizenden vragen, en leidt uit de antwoorden af welke patiënten in de training data zaten.
    """)
    return


@app.cell(hide_code=True)
def choice_goal_cell():
    choice_goal = L.choice(
        "goal",
        question="Welk doel van NIST heeft de aanvaller?",
        options={
            "right": "Privacy aantasten: data lekt uit het model",
            "integ": "Integriteit schenden: foute antwoorden",
            "avail": "Beschikbaarheid laten uitvallen: anderen kunnen het niet meer gebruiken",
        },
        correct="right",
        why={
            "right": "Volgens NIST lekt bij privacy aantasten (*privacy compromise*) onbedoeld afgeschermde of bedrijfseigen informatie uit het model, zoals details over de training data, de weights of de architectuur. In de woorden van [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) is het een verlies van vertrouwelijkheid.",
            "integ": "De antwoorden kunnen allemaal kloppen: de aanvaller leest ze alleen, en leert ervan. Bij integriteit schenden dwingt de aanvaller het model tot voorspellingen die de aanvaller goed uitkomen. Zie *Wat aanvallers van een model willen* in *Het idee*.",
            "avail": "Iedereen kan het model nog gebruiken. Wat er misgaat, is wat de antwoorden verraden. Zie *Wat aanvallers van een model willen* in *Het idee*.",
        },
    )
    choice_goal
    return (choice_goal,)


@app.cell(hide_code=True)
def text_exercise_knowledge():
    mo.md(r"""
    ### Oefening 5. Wat wisten ze?

    Eykholt en collega's maakten zwarte en witte stickers voor een echt stopbord. In het lab las het model dat ze aanvielen elke foto ervan als een snelheidsbord.
    """)
    return


@app.cell(hide_code=True)
def choice_knowledge_cell():
    choice_knowledge = L.choice(
        "knowledge",
        question="Wat wisten ze van dat model?",
        options={
            "right": "Alles, ook de weights: white box",
            "black": "Alleen de antwoorden op hun foto's: black box",
            "none": "Niets: stickers misleiden elk model",
        },
        correct="right",
        why={
            "right": "De paper zegt dat ze zich op de white-box-situatie richten. Met het model in handen konden ze de gradient volgen, zoals je fgsm de tekens van de weights volgt.",
            "black": "Ze hadden meer dan antwoorden: de paper zegt dat ze zich op de white-box-situatie richten. Een black-box-aanval kan ook, zoals in oefening 3, maar dit was er geen. Zie *Wat de aanvaller weet* in *Het idee*.",
            "none": "De stickers waren voor één model gemaakt, en volgens de paper in de white-box-situatie. Vanuit een rijdende auto werkten ze in 84,8% van de videobeelden, bij dat model. Zie *Buiten de computer* in *Het idee*.",
        },
    )
    choice_knowledge
    return (choice_knowledge,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Denken als een aanvaller](/course/open/rai-security-basics?lang=nl) beschermde security vertrouwelijkheid, integriteit en beschikbaarheid; NIST deelt aanvallen op modellen in naar dezelfde drie doelen, met privacy voor vertrouwelijkheid. In [De workflow van machine learning](/course/open/ml-workflow?lang=nl) mat de test set het model één keer, aan het eind. Ons model heeft alle 200 testplaatjes goed, en toch slaat STOP om met 0,212 per pixel: een test set meet het model op inputs zoals die waar het van leerde, niet op inputs die gemaakt zijn om het te misleiden. In [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) zei het model ja aan één kant van een lijn; `smallest_eps` is hoe ver een plaatje van die lijn ligt, in stappen langs de tekens. *De OWASP-lijsten voor LLM-apps* gaat over applicaties die op language models draaien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_goal, choice_knowledge, result_eps, result_fgsm, result_signs):
    L.wrap_up(result_fgsm, result_eps, result_signs, choice_goal, choice_knowledge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
