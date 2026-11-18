# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-unawareness-proxies. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-unawareness-proxies", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) en [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ratio():
    quick_ratio = L.choice(
        "quick-ratio",
        question="Een model geeft 7 procent van de vrouwen en 25 procent van de mannen een ja. Wat zegt de 80%-regel?",
        options={
            "right": "Een signaal: 0,28 is minder dan 0,8",
            "points": "Nog niets: het verschil is maar 18 procentpunten",
            "column": "Alleen een signaal als het model de kolom sex ziet",
        },
        correct="right",
        why={
            "right": "7 gedeeld door 25 is 0,28, ver onder vier vijfde. De regel kijkt naar wie een ja krijgt, niet naar hoe het model daar kwam.",
            "points": "De regel deelt de percentages, hij trekt ze niet van elkaar af: 7 / 25 = 0,28. Zie *Het idee* in [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl).",
            "column": "De regel kijkt alleen naar wie een ja krijgt, niet naar de kolommen die het model gebruikte. Zie *Het idee* in [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl).",
        },
    )
    quick_ratio
    return (quick_ratio,)


@app.cell(hide_code=True)
def quick_check_class():
    quick_class = L.choice(
        "quick-class",
        question="Een logistische regressie geeft een leerling een kans van 0,73 om te slagen. Welke klasse voorspelt het model voor die leerling?",
        options={
            "fail": "Gezakt, want 0,73 is minder dan 1",
            "right": "Geslaagd, want 0,73 is meer dan 0,5",
            "none": "Geen: het model geeft alleen een kans",
        },
        correct="right",
        why={
            "fail": "Het model voorspelt klasse 1 overal waar de kans boven 0,5 ligt, niet alleen bij 1. Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "right": "Boven 0,5 is de logit groter dan 0: de leerling ligt aan de kant van geslaagd van de decision boundary.",
            "none": "Logistische regressie geeft een kans, en voorspelt klasse 1 waar die boven 0,5 ligt. Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
        },
    )
    quick_class
    return (quick_class,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2014 begon Amazon aan een model dat sollicitanten één tot vijf sterren gaf. Het leerde van de cv's die in tien jaar naar het bedrijf waren gestuurd, de meeste van mannen. Rond 2015 zag het bedrijf dat het model sollicitanten niet genderneutraal beoordeelde: het waardeerde cv's met het Engelse woord *women's* (van vrouwen) lager, zoals bij de aanvoerder van een schaakclub voor vrouwen. Het team liet het model zulke woorden negeren, maar kon niet zeker weten dat het geen andere manier zou vinden om de sollicitanten te sorteren die kan discrimineren, en het team werd opgeheven. De les [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl) vertelt het hele verhaal. Een woord op een cv was genoeg om voor geslacht te staan. Deze les laat zien dat een model vrouwen en mannen nog steeds verschillend kan behandelen als je de kolom voor geslacht weglaat.

    **Een voorbeeld uit het dagelijks leven.** Een vacature zegt dat er niet naar je leeftijd gevraagd wordt. Maar het formulier vraagt in welk jaar je je diploma haalde. Wie de formulieren leest, kan nog steeds ongeveer uitrekenen hoe oud je bent.
    """)
    return


@app.cell(hide_code=True)
def predict_unaware():
    guess_unaware = L.guess(
        "unaware",
        question="In de tabel UCI Adult uit [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) geeft een model dat voorspelt wie meer dan 50.000 dollar verdient, 7 procent van de vrouwen en 25 procent van de mannen een ja. Het wordt opnieuw getraind zonder de kolom `sex`. Wat denk je dat de vrouwen dan krijgen?",
        options={
            "men": "Ongeveer 25 procent, evenveel als de mannen",
            "right": "Ongeveer 7 procent, bijna geen verschil",
            "half": "Ongeveer 16 procent",
        },
        reveal="**Bijna geen verschil: nog steeds 7 procent.** De andere kolommen vertellen het model veel over wie een vrouw is, dus maakt het weglaten van `sex` bijna niets uit. De demo hieronder laat het zien, en hoe goed de andere kolommen het geslacht verraden.",
    )
    guess_unaware
    return (guess_unaware,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **De kolom weglaten.** Het eerste idee van veel teams: geef het model geen kolom voor geslacht of afkomst, dan kan het die niet gebruiken. Matt Kusner en anderen (2017) noemen dat fairness through unawareness: volgens die definitie is een algoritme in orde zolang het een protected attribute A niet uitdrukkelijk gebruikt bij zijn beslissingen. A is hier een kolom zoals sex. De zwakke plek, in hun woorden: in de andere features kan informatie zitten die hetzelfde verraadt als A, zonder dat dat meteen opvalt.

    **Proxies.** Het Bureau van de Europese Unie voor de grondrechten (FRA) schreef in 2022 dat een proxy een ogenschijnlijk neutraal gegeven is dat toch sterk samenhangt met een discriminatiegrond, zoals schoenmaat met geslacht, of namen met afkomst. Er zijn er mogelijk eindeloos veel, en daarom is discriminatie via proxies moeilijker te voorkomen. De *Handreiking non-discriminatie by design* (2021), geschreven voor het ministerie van Binnenlandse Zaken, zegt: "postcode kan een proxy zijn voor etniciteit." Ze noemt ook een regel die alleen sollicitanten doorlaat die langer zijn dan 1,70 meter, omdat iedereen in het huidige personeel dat is: "Vrouwen zijn gemiddeld kleiner en zullen dus minder snel door het algoritme worden geselecteerd."

    **Veel kleine aanwijzingen.** Een proxy hoeft niet één duidelijke kolom te zijn. Het boek *Fairness and Machine Learning* merkt op dat een bezoek aan pinterest.com in de Verenigde Staten een kleine statistische samenhang had met vrouw zijn: te klein om er iemands geslacht goed mee te raden. Maar uit een aantal features die elk een beetje over het geslacht zeggen, kan een model het geslacht toch goed voorspellen. Laat je de kolom weg uit een model dat hem gebruikte, dan vindt het model hem terug in de andere features: een redundant encoding.

    **Redlining.** Tussen 1935 en 1940 tekende de Amerikaanse Home Owners' Loan Corporation gekleurde kaarten van 239 steden, met een letter voor elke wijk: van A, *Best*, in groen, tot D, *Hazardous* (gevaarlijk), in rood. Onderzoek dat de Federal Reserve Bank van St. Louis aanhaalt, vond de grootste concentraties Afro-Amerikanen in de wijken met de slechtste letter. Leningen weigeren of beperken voor hele wijken heet redlining, naar dat rood; Cynthia Dwork en anderen (2012) noemen het een bekende vorm van discriminatie via redundant encoding. In de VS is het nu verboden. In 2021 schikte Trustmark National Bank een zaak met het Amerikaanse ministerie van Justitie, dat de bank verweet zwarte en Latijns-Amerikaanse wijken van Memphis te hebben uitgesloten, met een fonds van 3,85 miljoen dollar om leningen goedkoper te maken en een boete van 5 miljoen.

    **Probeer het.** De cel hieronder gebruikt de tabel UCI Adult uit [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl): 48.842 mensen uit een Amerikaans onderzoek van 1994. Hij traint drie keer een logistische regressie op de training set om te voorspellen wie meer dan 50.000 dollar verdient: met de kolom `sex`, zonder, en ook zonder `relationship` en `marital_status`. `LogisticRegression` uit scikit-learn werkt als `LinearRegression` in [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl): `fit`, dan `predict`, dat `True` geeft waar de kans boven 0,5 ligt. `pd.get_dummies` maakt van elke kolom met woorden one-hot-kolommen, één per waarde, zoals in [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl). Voor elk model laat de cel zien welk deel van de vrouwen en van de mannen in de test set een ja krijgt, en hoe vaak het model daar gelijk heeft. Daarna traint hij een model dat uit de andere kolommen raadt wie een man is. Voer de cel uit; het duurt een paar seconden.
    """)
    return


@app.cell
def demo_models():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"
    _tested = _people[_test]
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]


    def _features(words):
        # elke kolom met woorden wordt one-hot-kolommen; daarna loopt elke kolom van 0 tot 1
        columns = ["age", "education_num", "hours_per_week"] + words
        X = pd.get_dummies(_people[columns], columns=words, dtype=float)
        return X / X.max()


    def _model_says(name, words):
        X = _features(words)
        rich = _people["income"] == ">50K"
        model = LogisticRegression(max_iter=1000)   # meer stappen dan standaard, zodat het klaar is met leren
        model.fit(X[_train], rich[_train])
        yes = model.predict(X[_test])
        picked = _tested[yes]
        rates = picked.groupby("sex").size() / _tested.groupby("sex").size()
        right = (yes == rich[_test]).mean()
        print(name, "vrouwen", round(rates["Female"], 2), "mannen", round(rates["Male"], 2),
              "goed", round(right, 2))


    _model_says("met sex:   ", _words + ["sex"])
    _model_says("zonder sex:", _words)
    _model_says("ook zonder relationship en marital_status:", ["workclass", "occupation", "race"])

    # kunnen de andere kolommen zien wie een man is?
    _man = _people["sex"] == "Male"
    _X = _features(_words)
    _guesser = LogisticRegression(max_iter=1000)
    _guesser.fit(_X[_train], _man[_train])
    _guessed = _guesser.predict(_X[_test])
    print("geslacht raden, goed:", round((_guessed == _man[_test]).mean(), 2))
    print("altijd 'man', goed:", round(_man[_test].mean(), 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture({"title": "Hoe vaak vrouwen en mannen een ja krijgen",
                           "rows": ["In de data: de labels", "Model met de kolom sex", "Model zonder de kolom sex",
                                    "Ook zonder relationship en marital status"],
                           "xlabel": "deel met een ja", "key": "open: vrouwen, gevuld: mannen", "decimal": ",",
                           "alt": "Vier paren staven van het deel van de vrouwen en van de mannen met een ja in de test set. In de data: 0,11 en 0,30. Het model met de kolom sex: 0,07 en 0,25. Zonder: 0,07 en 0,25. Ook zonder relationship en marital status: 0,08 en 0,17."})),
        mo.md(r"""
        Met de kolom `sex` geeft het model 0,07 van de vrouwen en 0,25 van de mannen een ja; zonder die kolom dezelfde 0,07 en 0,25, en het heeft ongeveer even vaak gelijk. De andere kolommen weten wie een vrouw is: een model dat het geslacht daaruit raadt, heeft in 0,84 van de gevallen gelijk, terwijl altijd "man" raden in 0,67 gelijk heeft. Het duidelijkst is `relationship`: iemands plek in het huishouden, zoals Husband (echtgenoot) of Wife (echtgenote). In Oefening 1 meet je hoeveel die verraadt.

        Laat je ook `relationship` en `marital_status` weg, dan wordt het verschil kleiner, 0,08 tegen 0,17, maar heeft het model minder vaak gelijk, 0,8, en krijgen vrouwen nog steeds half zo vaak een ja als mannen. Een deel van het verschil zit in de labels zelf: in de data verdient 0,11 van de vrouwen en 0,30 van de mannen meer dan 50.000 dollar. Toon Calders en Sicco Verwer (2010) vonden op dezelfde tabel dat een ander soort model, getraind zonder de kolom voor geslacht, mannen zelfs meer bevoordeelde dan de data: een ja voor 38 procent van de mannen en 10 procent van de vrouwen, terwijl de data 30 en 11 had. *Fairness and Machine Learning* zegt het algemener: met rijke datasets heeft het weglaten van zulke kolommen meestal een verwaarloosbaar effect op een model. Proxies een voor een najagen kan zelfs averechts werken: toen sommige Amerikaanse wetten bij sollicitaties vragen over een strafblad verboden, gingen werkgevers meer discrimineren op ras, vonden Agan en Starr.

        Dwork en anderen noemden redundant encoding een geslaagde aanval op fairness through blindness, en gaven hun eigen paper de titel *Fairness Through Awareness*: hun methode beslist met kennis van wat twee mensen voor de taak op elkaar doet lijken. De cel hierboven laat een eenvoudiger soort awareness zien: om dit allemaal te zien, had hij de kolom `sex` nodig. Hij liet die weg uit het model, maar hield hem apart om te controleren.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **fairness through unawareness** | een kolom zoals geslacht of afkomst weglaten uit de features van een model, in de hoop dat het de groepen dan gelijk behandelt; op zich werkt dat niet |
        | **proxy** | een feature die neutraal lijkt, maar informatie bevat over zo'n kolom, zoals een postcode over afkomst |
        | **redundant encoding** | als de andere features samen een weggelaten kolom verraden |
        | **redlining** | leningen weigeren of beperken voor hele wijken, wat mensen van kleur trof: de wijk stond voor ras |
        """),
        L.watch_out("Een kolom weglaten uit de features is iets anders dan hem weggooien. Zonder die kolom kan niemand tellen welk deel van elke groep een ja krijgt, zoals de cel hierboven deed, dus kan niemand zien dat het model de groepen nog steeds verschillend behandelt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een model te controleren dat een kolom zoals geslacht weglaat:

    1. Houd de kolom apart: niet in de features, wel in de data waarmee je het model test.
    2. Tel per groep welk deel een ja krijgt, en deel het kleinste deel door het grootste, zoals bij de 80%-regel.
    3. Zoek proxies: per waarde van een kolom het deel van elke groep. Een waarde met een deel dicht bij 0 of 1 verraadt de groep.
    4. Laat een model de kolom raden uit de andere features. Doet het veel beter dan altijd de grootste groep raden, dan zit de kolom nog in de data.
    5. Reken er niet op dat je proxies een voor een kunt schrappen. De volgende lessen meten fairness en passen modellen bewust aan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Weer de tabel UCI Adult. `_column` is de kolom `relationship`: iemands plek in het huishouden, zoals Husband, Wife of Own-child (kind in het huishouden). De cel kiest de mensen die meer dan 50.000 dollar verdienen, en deelt hoeveel van hen elke waarde hebben door hoeveel mensen die waarde hebben. Voer de cel uit.
    """)
    return


@app.cell
def example_rich(pd):
    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _column = "relationship"

    _rich = _people[_people["income"] == ">50K"]
    _share = _rich.groupby(_column).size() / _people.groupby(_column).size()
    print(_share.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Mensen met Husband en met Wife verdienen ongeveer even vaak meer dan 50.000 dollar, 0,45 en 0,47; wie als kind in het huishouden woont bijna nooit, 0,01. De kolom helpt het model dus om het inkomen te voorspellen. Maar wie hebben Husband, en wie Wife? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_proxy():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een proxy voor geslacht

        `column` is de kolom `relationship`. Vervang de `...` door de twee regels van het uitgewerkte voorbeeld, met de namen zonder hun liggend streepje: `men` en `share`. Zet in `men` de rijen waarin `sex` gelijk is aan `"Male"`, en in `share` de mannen per waarde van `column` gedeeld door alle mensen per waarde. De laatste regel is `share`, zodat de check het ziet. De check probeert ook een andere kolom.
        """),
        L.task("Reken `share` uit: per waarde van `relationship` het deel mannen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_proxy():
    box_proxy = L.code_box("proxy", lines=10)
    box_proxy
    return (box_proxy,)


@app.cell(hide_code=True)
def check_proxy(box_proxy):
    result_proxy = L.check("proxy", box_proxy.value)
    result_proxy
    return (result_proxy,)


@app.cell(hide_code=True)
def hints_proxy():
    L.hints(
        "proxy",
        [
            "Gebruik de twee regels van het uitgewerkte voorbeeld. Alleen de rijen die je kiest, veranderen: niet de mensen die meer dan 50.000 dollar verdienen, maar de mannen.",
            "De mannen zijn de rijen waarin de kolom `sex` gelijk is aan `\"Male\"`: `people[people[\"sex\"] == \"Male\"]`.",
            "`share = men.groupby(column).size() / ___`.",
        ],
        why="`men = people[people[\"sex\"] == \"Male\"]`, en dan `share = men.groupby(column).size() / people.groupby(column).size()`. Dat geeft ongeveer `1.0` voor Husband en ongeveer `0.0` voor Wife.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proxies():
    mo.md(r"""
    ### Oefening 2. Wat kan waarvoor staan?

    Zes gegevens die een model voor werving zou kunnen krijgen. Geen ervan noemt afkomst of geslacht. Deel elk in bij waar het voor kan staan.
    """)
    return


@app.cell(hide_code=True)
def sort_proxies_cell():
    sort_proxies = L.sort("proxies")
    sort_proxies
    return (sort_proxies,)


@app.cell(hide_code=True)
def check_proxies(sort_proxies):
    result_proxies = L.check("proxies", sort_proxies)
    result_proxies
    return (result_proxies,)


@app.cell(hide_code=True)
def hints_proxies():
    L.hints(
        "proxies",
        [
            "Vraag bij elk gegeven: welke groep heeft dit vaker, of minder vaak?",
            "Een postcode zegt waar iemand woont, en wijken zijn vaak niet gemengd. Welk ander gegeven zegt iets over waar een familie vandaan komt?",
            "Kan staan voor afkomst: postcode en \\_\\_\\_. Kan staan voor geslacht: lengte, \\_\\_\\_, het gat en de schoenmaat.",
        ],
        why="**Kan staan voor afkomst:** postcode en achternaam. **Kan staan voor geslacht:** lengte, de club, het gat in het cv en de schoenmaat.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_check():
    mo.md(r"""
    ### Oefening 3. Houd hem apart om te controleren
    """)
    return


@app.cell(hide_code=True)
def choice_check_cell():
    choice_check = L.choice(
        "check",
        question="Een team haalt de kolom `sex` helemaal uit zijn data, ook uit de data waarmee het het model test. Kan het dan nog laten zien dat zijn model vrouwen en mannen ongeveer even vaak een ja geeft?",
        options={
            "never": "Ja, het model zag nooit geslacht",
            "right": "Nee, het kan niet per geslacht tellen",
            "accuracy": "Ja, als het model even vaak gelijk heeft als eerst",
        },
        correct="right",
        why={
            "never": "Het model zonder `sex` gaf vrouwen en mannen nog steeds heel verschillend vaak een ja. Zie *Probeer het* in *Het idee*.",
            "right": "Wie per groep wil tellen hoe vaak er een ja komt, moet weten wie in welke groep zit. Laat de kolom weg uit de features, maar houd hem apart om te controleren.",
            "accuracy": "Zonder `sex` had het model ongeveer even vaak gelijk, en gaf het mannen toch meer dan drie keer zo vaak een ja. Zie *Probeer het* in *Het idee*.",
        },
    )
    choice_check
    return (choice_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) telde je per groep hoe vaak er een ja kwam; hier zag je dat een model groepen heel verschillend vaak een ja kan geven zonder de groep ooit te zien. In [Feedback loops](/course/open/rai-feedback-loops?lang=nl) vormde een model zijn eigen data; hier vormt de data het model, via kolommen die voor geslacht staan. Jouw deel per waarde is een eerste test voor een proxy, en die werkt alleen omdat de kolom `sex` bewaard bleef. De volgende les, *Evenveel ja per groep*, maakt evenveel ja per groep tot een eigen criterium voor fairness, en laat zien wat dat mist.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_check, result_proxies, result_proxy):
    L.wrap_up(result_proxy, result_proxies, choice_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
