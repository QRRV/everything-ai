# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-reliability. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Risk, Risk Management, SPLIT

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-reliability", "nl")
    C = L.checks   # de softmax-regressie van de course voor cijfers, de 1000 testcijfers, drie verzonnen agents


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl) en [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_category():
    quick_category = L.choice(
        "quick-category",
        question="Een systeem dat op een postkantoor cijfers leest, leest vlekkerige postcodes verkeerd. Niemand wilde schade doen, en het moest ze juist goed lezen. Waar zet het International AI Safety Report van 2025 dit?",
        options={
            "right": "Onder falen, bij onbetrouwbaarheid",
            "misuse": "Onder kwaadwillig gebruik, want de schade treft de afzenders",
            "systemic": "Onder systeemrisico's, want niemand wilde schade doen",
        },
        correct="right",
        why={
            "right": "Niemand wilde schade doen, en het systeem deed niet wat bedoeld was: falen. Het rapport van 2025 rekent onbetrouwbaarheid daartoe, naast bias en verlies van controle.",
            "misuse": "Kwaadwillig gebruik vraagt iemand die schade wil doen, en die is er hier niet. Zie [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl).",
            "systemic": "Dat niemand schade wilde doen, sluit kwaadwillig gebruik uit; vraag dan of het systeem deed wat bedoeld was. Dat deed het niet, dus is het falen. Zie [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl).",
        },
    )
    quick_category
    return (quick_category,)


@app.cell(hide_code=True)
def quick_check_logits():
    quick_logits = L.choice(
        "quick-logits",
        question="Softmax-regressie krijgt een beeld als een rij `x` van 784 pixels. Hoe kiest het model een cijfer?",
        options={
            "common": "Het cijfer dat het vaakst voorkomt in de training set",
            "right": "Het cijfer met de grootste van de tien logits `x @ W.T + b`",
            "bias": "Het cijfer waarvan de eigen bias term in `b` de grootste van de tien is",
        },
        correct="right",
        why={
            "right": "Elk cijfer heeft een eigen rij weights in `W` en een eigen bias term in `b`; de grootste logit wint, en softmax maakt van de logits alleen kansen.",
            "common": "Dat is de baseline, die altijd hetzelfde cijfer zegt en er 126 van de 1000 goed leest. Zie [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl).",
            "bias": "De bias terms zijn maar een deel van elke logit; de pixels maal de weights bepalen het meeste. Zie [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl).",
        },
    )
    quick_logits
    return (quick_logits,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de projectles las softmax-regressie 890 van 1000 nieuwe cijfers goed. Die cijfers waren op dezelfde manier gescand als de 6000 waarvan het model leerde. In de praktijk kan er een grijze waas over een scan liggen of ruis van een goedkope scanner, of staat een cijfer een beetje opzij. Leest het model ze dan nog?

    Blijven werken vraagt meer dan een goede score op een test set. Deze les geeft er vier woorden voor, meet er twee, bij de classifier voor cijfers en bij verzonnen agents, en laat zien hoe een wet en een contract ze vastleggen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De beeldclassifier AlexNet benoemt 43,5% van de schone testfoto's van ImageNet verkeerd, dus 56,5% goed. Een studie voegde Gauss-ruis toe aan de foto's, de korrel van een foto die bij weinig licht is genomen, in vijf sterktes. Welk deel benoemt AlexNet goed, gemiddeld over de vijf?",
        options={
            "half": "Ongeveer 50%: ruis maakt weinig uit",
            "right": "Ongeveer 11%: één foto op negen",
            "third": "Ongeveer 35%: een derde van de treffers kwijt",
        },
        reveal="**Ongeveer 11%: één foto op negen.** De fout van AlexNet ging van 43,5% op schone foto's naar 88,6% met Gauss-ruis, gemiddeld over de vijf sterktes, dus het benoemde 11,4% goed. Mensen raken door veel van zulke beschadigingen niet in de war, schrijven de auteurs: *Ruis, onscherpte en opschuiven* in *Het idee* komt erop terug.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Vier woorden voor blijven werken

    In januari 2023 publiceerde het Amerikaanse normeninstituut NIST zijn AI Risk Management Framework. Bij de eigenschappen van AI-systemen die mensen kunnen vertrouwen, noemt het onder meer geldig en betrouwbaar zijn, en beveiligd en veerkrachtig. Zijn definities baseert het op de internationale norm ISO/IEC TS 5723:2022:

    - betrouwbaarheid: het vermogen van een onderdeel om te werken zoals vereist, zonder te falen, gedurende een bepaalde tijd en onder bepaalde omstandigheden;
    - **robuustheid**: het vermogen van een systeem om zijn prestatieniveau te houden in allerlei omstandigheden. NIST voegt eraan toe dat een robuust systeem, als het in een onverwachte situatie werkt, ook zo moet werken dat het mensen zo weinig mogelijk schade doet;
    - veerkracht: in de kortere woorden van NIST het vermogen om na een onverwachte nadelige gebeurtenis weer normaal te werken; een veerkrachtig systeem mag ook veilig en geleidelijk minder gaan doen als dat nodig is.

    Beschikbaarheid staat in het framework bij beveiliging: een beveiligd systeem bewaart vertrouwelijkheid, integriteit en beschikbaarheid. Het Engelse woord voor voorspelbaarheid komt in het framework helemaal niet voor. Deze les verstaat eronder: elke keer hetzelfde doen als dezelfde taak terugkomt. Hieronder meet de les dat met pass^k.

    NIST waarschuwt ook dat accuracy en robuustheid met elkaar in spanning kunnen staan. De accuracy op een test set zegt hoe een model het doet op inputs zoals die in de test set, en niet meer.

    ### Ruis, onscherpte en opschuiven

    In 2019 publiceerden Dan Hendrycks en Thomas Dietterich ImageNet-C: de testfoto's van ImageNet met 15 soorten beschadiging die een algoritme aanbrengt, uit de groepen ruis, onscherpte, weer en digitaal, elk in vijf sterktes, 75 in totaal. De eerste is Gauss-ruis, die kan ontstaan bij weinig licht. Modellen worden op deze foto's getest en horen er niet op te trainen. Het gaat om beschadigingen zoals die vanzelf ontstaan, niet om de ergste aanvallen die een tegenstander kan bedenken: robuustheid tegen aanvallen is een onderwerp apart, dat NIST en de AI Act onder beveiliging rekenen. Mensen raken door veel beschadigingen niet in de war, zoals sneeuw, onscherpte, grove pixels en nieuwe combinaties daarvan, schrijven ze, terwijl bij de netwerken van AlexNet tot ResNet de robuustheid tegen beschadiging zelf nauwelijks veranderde. Hun conclusie: de classifiers die we bovenmenselijk noemen, zijn duidelijk ondermenselijk. In een tweede test uit hetzelfde paper, met foto's die stap voor stap veranderen, bijvoorbeeld door verschuiven, draaien of inzoomen, veranderde een ResNet-18 bij inzoomen in 15,6% van de gevallen zijn beste antwoord tussen twee opeenvolgende beelden.

    ### Wat de AI Act vraagt

    De AI Act van de EU vraagt in zijn tekst van 2024, in artikel 15, dat AI-systemen met een hoog risico een passend niveau van accuracy, robuustheid en cyberbeveiliging halen, en dat ze daarin hun hele levensduur consequent presteren. Ze moeten zo veerkrachtig mogelijk zijn tegen fouten, storingen en inconsistenties, en robuustheid kan bereikt worden met technische redundantie, zoals een back-up of een fail-safe: een tweede systeem dat het overneemt, of een veilige manier om het systeem stil te leggen. Aanvallen krijgen een eigen lid: systemen moeten bestand zijn tegen pogingen van onbevoegden om hun gebruik of output te veranderen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Elke keer hetzelfde

    Een AI-agent die bestellingen wijzigt of vluchten boekt, hoort niet de ene dag te slagen en de volgende dag te falen op hetzelfde verzoek. In 2024 bouwden Shunyu Yao en collega's bij het bedrijf Sierra τ-bench, waarin een agent een gesimuleerde klant helpt van een winkel (τ-retail) of van een luchtvaartmaatschappij (τ-airline). Ze stelden **pass^k** voor (spreek uit: pass hat k): de kans dat alle k pogingen bij een taak slagen, gemiddeld over de taken, met pogingen die i.i.d. zijn: elke poging staat los van de andere en gebeurt onder dezelfde omstandigheden. Een taak die slaagt met kans p, slaagt k keer achter elkaar met kans pᵏ: p, k keer met zichzelf vermenigvuldigd. Tussen de pogingen veranderde alleen wat de language models schreven: de prompt van de gebruiker en de veranderingen in de database waren steeds dezelfde. gpt-4o loste ongeveer 61% van de taken van τ-retail op in één poging. Naarmate k groter wordt, daalt de kans dat een taak elke keer lukt snel, voor hetzelfde model tot maar ongeveer 25% bij pass^8 op τ-retail.

    ### Er zijn als het nodig is

    Beschikbaarheid wordt op twee manieren gemeten, legt het boek van Google over site reliability engineering uit: als het deel van de tijd dat een systeem werkt, of, zoals Google doet, als het deel van de requests dat slaagt. Een cloudcontract voor AI-modellen, dat van Amazon Bedrock, kiest het tweede: de beschikbaarheid is per interval van vijf minuten het percentage van de requests aan Amazon Bedrock dat niet met een Error mislukt, en een Error, zoals het contract het noemt, is elk request dat foutcode 500 teruggeeft. De latere les *Negens van betrouwbaarheid* rekent met zulke percentages.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **robuustheid** | het prestatieniveau houden als de input of de situatie een beetje anders is dan bij de tests, zoals bij ruis, onscherpte of cijfers die opschuiven |
        | betrouwbaarheid | werken zoals vereist, zonder te falen, gedurende een bepaalde tijd en onder bepaalde omstandigheden |
        | veerkracht | een onverwachte gebeurtenis of verandering doorstaan en weer normaal werken, of veilig minder gaan doen |
        | beschikbaarheid | er zijn als het nodig is: het deel van de tijd, of van de requests, waarin een systeem werkt |
        | voorspelbaarheid | elke keer hetzelfde doen als dezelfde taak komt |
        | **pass^k** | de kans dat alle k pogingen bij een taak slagen, gemiddeld over de taken; pass^1 is gewoon het slagingspercentage |
        | pᵏ | p tot de macht k: p, k keer met zichzelf vermenigvuldigd; Python schrijft een macht met `**`, dus `0.5 ** 3` is 0,125 |
        | `rng.normal(0, sigma, size=X.shape)` | willekeurige getallen met gemiddelde 0 en standaardafwijking `sigma`, één voor elke pixel van `X`: Gauss-ruis |
        | `C.model()`, `C.test_set()` | de softmax-regressie van de course, `W` en `b`, getraind zoals in de projectles (100 stappen, learning rate 1, dezelfde 6000 cijfers), en de 1000 testcijfers `X` met hun labels `y` |
        | `C.AGENT`, `C.STEADY`, `C.SPLIT` | drie verzonnen agents: elk een lijst met het slagingspercentage van elk van zijn tien taken |
        """),
        L.watch_out("Een model dat schone testcijfers goed leest, kan nog falen op cijfers met een waas of ruis, of op cijfers die opzij staan. Test het op de veranderingen die je in gebruik verwacht; robuustheid tegen aanvallen vraagt eigen tests."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zeg welke soort blijven werken je bedoelt: robuust tegen veranderde input, veerkrachtig na een storing, beschikbaar als het nodig is, of voorspelbaar van poging tot poging.
    2. Kies voor robuustheid veranderingen die in gebruik vanzelf gebeuren, zoals ruis, een waas of cijfers die opschuiven, en houd ze buiten het trainen.
    3. Meet de accuracy met en zonder elke verandering, in een paar sterktes.
    4. Laat voor voorspelbaarheid elke taak een paar keer lopen en reken pass^k uit, niet alleen pass^1.
    5. Lees voor beschikbaarheid wat als falen telt: een foutcode, een timeout of een fout antwoord.
    6. Test robuustheid tegen aanvallen apart: NIST en de AI Act rekenen die onder beveiliging.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel legt een grijze waas over de 1000 testcijfers door bij elke pixel 0,3 op te tellen, en meet welk deel het model van de course dan nog goed leest, naast de schone cijfers.
    """)
    return


@app.cell
def example_haze():
    W, b = C.model()                                         # de softmax-regressie van de course
    X, y = C.test_set()                                      # de 1000 testcijfers en hun labels
    for change in [0, 0.3]:
        predicted = np.argmax((X + change) @ W.T + b, axis=1)   # het cijfer met de grootste logit
        print(change, np.mean(predicted == y))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Schoon leest het model 890 van de 1000 goed; met de waas 741. De waas verplaatst geen enkele streep: elk cijfer staat waar het stond, op een grijze achtergrond. In oefening 1 tel je in plaats daarvan ruis op, van elke sterkte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_noisy():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. noisy_accuracy: de accuracy onder ruis

        Schrijf `noisy_accuracy(W, b, X, y, sigma)`. Maak een generator `rng` met `np.random.default_rng(0)`, zodat de ruis bij elke run hetzelfde is. Maak `noisy`: `X` plus ruis uit `rng.normal`, met gemiddelde 0, standaardafwijking `sigma` en `size=X.shape`. Laat `predicted` voor elke rij van `noisy` het cijfer met de grootste logit zijn, zoals in het uitgewerkte voorbeeld, en geef het deel van `predicted` terug dat gelijk is aan `y`. Vervang de `...`. De check roept `noisy_accuracy(W, b, X, y, 0.5)` aan met het model en de testcijfers van de course, daarna andere sterktes en een eigen klein model.
        """),
        L.task("Het deel van de cijfers met ruis dat het model nog goed leest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_noisy():
    box_noisy = L.code_box("noisy_accuracy", lines=8)
    box_noisy
    return (box_noisy,)


@app.cell(hide_code=True)
def check_noisy(box_noisy):
    result_noisy = L.check("noisy_accuracy", box_noisy.value)
    result_noisy
    return (result_noisy,)


@app.cell(hide_code=True)
def hints_noisy():
    L.hints(
        "noisy_accuracy",
        [
            "Drie stappen: maak de ruis en tel die op, laat het model voor elk beeld met ruis een cijfer kiezen, en tel hoe vaak dat klopt.",
            "`rng.normal(0, sigma, size=X.shape)` geeft één getal per pixel. De logits zijn de beelden met ruis maal de getransponeerde weights, plus de bias terms; `np.argmax` met `axis=1` neemt de grootste in elke rij, en `np.mean` van `predicted == y` geeft het deel.",
            "`    predicted = np.argmax(___, axis=1)`",
        ],
        why="`rng = np.random.default_rng(0)`, `noisy = X + rng.normal(0, sigma, size=X.shape)`, `predicted = np.argmax(noisy @ W.T + b, axis=1)` en `return np.mean(predicted == y)`. Met ruis van 0,5 leest het model 820 van de 1000 testcijfers goed.",
    )
    return


@app.cell(hide_code=True)
def text_after_noisy():
    mo.md(r"""
    Met ruis van 0,25 leest het model 874 van de 1000 goed, bij 0,5 is dat 820, en bij 1,0 nog 637, terwijl ruis met standaardafwijking 1,0 zo groot is als het hele bereik van een pixel. Een waas van 0,3 kostte meer dan ruis van 0,5. Elke logit is een som over de 784 pixels. De waas telt daar bij elk beeld hetzelfde bij op: 0,3 maal de som van de weights van dat cijfer: +1,67 voor een 2, maar −3,41 voor een 1, het laagste van de tien, dus zegt het model maar 22 keer een 1 in plaats van 127 keer. Ruis rond 0 verandert elke logit bij elk beeld met ongeveer 1, maar even vaak omhoog als omlaag: geen cijfer verliest bij elk beeld. Dat geldt voor dit model op deze cijfers; AlexNet verloor veel meer door de Gauss-ruis van ImageNet-C, bij een andere taak. Welke verandering pijn doet, hangt af van het model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. pass_hat: alle k pogingen

        `C.AGENT` is een verzonnen agent: een lijst met het slagingspercentage van elk van zijn tien taken, van 1,0 tot 0,0. Schrijf `pass_hat(rates, k)`. Voor elk percentage `p` in `rates` is de kans dat alle `k` pogingen slagen p tot de macht `k`. Tel die op in `total`, dat bij 0 begint, en geef `total` gedeeld door het aantal taken terug. Vervang de `...`. De check roept `pass_hat(AGENT, 8)` aan, daarna andere agents en waarden van `k`.
        """),
        L.task("De kans dat alle k pogingen slagen, gemiddeld over de taken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pass():
    box_pass = L.code_box("pass_hat", lines=5)
    box_pass
    return (box_pass,)


@app.cell(hide_code=True)
def check_pass(box_pass):
    result_pass = L.check("pass_hat", box_pass.value)
    result_pass
    return (result_pass,)


@app.cell(hide_code=True)
def hints_pass():
    L.hints(
        "pass_hat",
        [
            "Eén taak tegelijk: de kans dat die `k` keer achter elkaar slaagt. Daarna het gemiddelde over de taken.",
            "Python schrijft een macht met `**`. Tel op in een `for`-loop over `rates`, en deel door `len(rates)`.",
            "`        total = total + ___`",
        ],
        why="`total = 0`, `for p in rates:` met daarin `total = total + p ** k`, en dan `return total / len(rates)`. Voor AGENT met k = 8: 0,268.",
    )
    return


@app.cell(hide_code=True)
def text_after_pass():
    mo.md(r"""
    Alle drie de agents slagen in 60% van hun pogingen: pass^1 is voor elk 0,6. Bij k = 8 zakt STEADY, met elke taak op 60%, naar 0,017; AGENT zakt naar 0,268; SPLIT, die zes taken altijd oplost en de andere vier nooit, blijft op 0,6. Hetzelfde slagingspercentage kan heel verschillende consistentie verbergen. De course verzon AGENT zo dat hij ongeveer even snel zakt als gpt-4o op τ-retail, van ongeveer 61% naar ongeveer 25%.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: opschuiven naar rechts

    De schuif zet de 1000 testcijfers een aantal pixels naar rechts; de kolommen die links binnenkomen, zijn zwart. Het plaatje toont de eerste acht testcijfers met wat het model van de course bij elk zegt, en hoeveel van de 1000 het bij elk aantal pixels goed leest.
    """)
    return


@app.cell(hide_code=True)
def demo_shift():
    shift = L.explore("shift", "pixels naar rechts", lambda _v: C.demo_frame(_v, L.lang),
                      start=0, values=C.SHIFTS, role="",
                      task="Zoek het kleinste aantal pixels op de schuif waarbij het model minder dan de helft van de testcijfers goed leest.")
    shift
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 3 pixels leest het model minder dan de helft goed: 452 van de 1000. Eén pixel kost 46 cijfers, twee pixels 251, ongeveer evenveel als ruis van 1,0. Elke weight van softmax-regressie hoort bij één pixel, dus een streep die opschuift, komt op andere weights terecht: schuif de weights ook drie pixels naar rechts, en het model leest er weer 890. Voor een mens is een cijfer dat twee pixels opzij staat hetzelfde cijfer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_test():
    mo.md(r"""
    ### Oefening 3. Welke test?

    De classifier voor cijfers gaat postcodes op brieven lezen. Een team wil weten hoe robuust hij is tegen veranderingen die vanzelf gebeuren, zoals Hendrycks en Dietterich dat testen.
    """)
    return


@app.cell(hide_code=True)
def choice_test_cell():
    choice_test = L.choice(
        "test",
        question="Welke test meet dat?",
        options={
            "right": "Voeg ruis of een waas toe, of schuif de cijfers iets op, en meet de accuracy opnieuw",
            "attack": "Verander pixels expres, stap voor stap, tot het model een fout cijfer leest",
            "repeat": "Laat dezelfde schone testcijfers nog eens zien en kijk of elk antwoord hetzelfde blijft",
        },
        correct="right",
        why={
            "right": "Veranderingen die vanzelf gebeuren, in een paar sterktes, gemeten tegen de schone accuracy: zo testen Hendrycks en Dietterich, met 15 soorten beschadiging en in hun tweede test met verschuiven en inzoomen.",
            "attack": "Dat is een aanval. ImageNet-C laat de ergste aanvallen die een tegenstander kan bedenken juist weg, en NIST en de AI Act rekenen aanvallen onder beveiliging. Zie *Ruis, onscherpte en opschuiven* in *Het idee*.",
            "repeat": "Dat meet of het model elke keer hetzelfde doet, en dat is voorspelbaarheid; softmax-regressie geeft voor hetzelfde beeld altijd hetzelfde antwoord. Robuustheid gaat over inputs die afwijken. Zie *Vier woorden voor blijven werken* in *Het idee*.",
        },
    )
    choice_test
    return (choice_test,)


@app.cell(hide_code=True)
def text_exercise_uptime():
    mo.md(r"""
    ### Oefening 4. Beschikbaar, maar goed?

    Een hele maand beantwoordt een clouddienst met een AI-model elk request binnen een seconde, nooit met foutcode 500. Eén antwoord op de tien is fout.
    """)
    return


@app.cell(hide_code=True)
def choice_uptime_cell():
    choice_uptime = L.choice(
        "uptime",
        question="Wat is de beschikbaarheid van de dienst die maand, in een contract zoals dat van Amazon Bedrock?",
        options={
            "ninety": "90%: elk fout antwoord telt als een mislukt request",
            "right": "100%: een fout antwoord dat aankomt, is daar geen Error",
            "servers": "Dat valt niet te zeggen: daarvoor is ook de uptime van de servers nodig",
        },
        correct="right",
        why={
            "right": "Het contract telt de requests die niet met een Error mislukken, en een Error is een request dat foutcode 500 teruggeeft. Beschikbaarheid zegt dat de dienst er was, niet dat hij goed was: daarvoor is een eigen maat nodig, zoals accuracy.",
            "ninety": "Een fout antwoord geeft geen foutcode 500 terug, dus voor het contract mislukte het niet. Zie *Er zijn als het nodig is* in *Het idee*.",
            "servers": "Het contract meet beschikbaarheid per request, voor elk interval van vijf minuten, niet als de uptime van servers. Zie *Er zijn als het nodig is* in *Het idee*.",
        },
    )
    choice_uptime
    return (choice_uptime,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl) rekende onbetrouwbaarheid tot falen. Deze les splitste blijven werken op in robuustheid, veerkracht, beschikbaarheid en voorspelbaarheid, en mat er twee, met het model van [Project: een lineaire classifier voor handgeschreven cijfers](/course/open/ml-digits-linear?lang=nl). De volgende les, *Risico als kans maal schade*, vraagt hoe erg falen is: hoe waarschijnlijk het is, en hoeveel schade het doet.

    **Een voorbeeld uit het dagelijks leven.** Een buslijn kan op droge dagen op tijd rijden. Robuust is een bus die ook in de regen op tijd rijdt; veerkrachtig een busbedrijf dat bij pech snel een reservebus laat rijden; beschikbaar een lijn die rijdt als mensen hem nodig hebben; voorspelbaar een bus die elke dag op dezelfde minuut vertrekt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test, choice_uptime, result_noisy, result_pass):
    L.wrap_up(result_noisy, result_pass, choice_test, choice_uptime)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
