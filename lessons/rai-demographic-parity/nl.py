# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-demographic-parity. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-demographic-parity", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl) en [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_proxy():
    quick_proxy = L.choice(
        "quick-proxy",
        question="Een model laat de kolom `sex` weg, maar houdt `relationship`, waarin veel mensen Husband of Wife zijn. Wat kan er gebeuren met het deel van de vrouwen en van de mannen dat een ja krijgt?",
        options={
            "equal": "Die worden gelijk, nu sex weg is",
            "right": "Die kunnen ver uit elkaar blijven",
            "swap": "Die draaien om: vrouwen krijgen vaker ja",
        },
        correct="right",
        why={
            "equal": "Zonder `sex` veranderden de delen met een ja nauwelijks: andere kolommen verraadden het geslacht. Zie *Probeer het* in [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl).",
            "right": "`relationship` verraadt het geslacht van de meeste mensen, dus het model kan vrouwen en mannen nog steeds verschillend behandelen zonder ooit `sex` te zien.",
            "swap": "Niets laat ze omdraaien: zonder `sex` gaf het model vrouwen en mannen dezelfde delen als ervoor, 0,07 en 0,25. Zie *Probeer het* in [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl).",
        },
    )
    quick_proxy
    return (quick_proxy,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="Een model zegt ja tegen 200 mensen, en 150 van hen verdienen echt meer dan 50.000 dollar. Wat is de precision?",
        options={
            "wrong": "0,25, het deel dat fout is",
            "right": "0,75, want 150 / 200",
            "count": "150 mensen",
        },
        correct="right",
        why={
            "wrong": "Precision telt de ja's die kloppen, niet de ja's die fout zijn: 150 / 200 = 0,75. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "right": "Van de 200 ja's kloppen er 150: een precision van 0,75.",
            "count": "Precision is een deel, geen aantal: 150 van de 200, dus 0,75. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl) gaf een model zonder de kolom `sex` 0,07 van de vrouwen en 0,25 van de mannen een ja. De eerste eis die bij je opkomt: geef elke groep even vaak een ja. Die eis heeft een naam, demographic parity, en het is het eenvoudigste criterium van group fairness, dat een model beoordeelt door getallen tussen groepen te vergelijken. Deze les meet het, bereikt het, en laat zien wat het niet kan zien.

    **Een voorbeeld uit het dagelijks leven.** Een school heeft plaatsen voor een uitstapje, en laat elke klas hetzelfde deel van de leerlingen sturen: een op de vijf. Het deel is in elke klas hetzelfde. Of een klas de leerlingen stuurt die het graagst willen, of namen uit een hoed trekt, zie je niet aan dat deel. <!-- termen:ok -->
    """)
    return


@app.cell(hide_code=True)
def predict_limits():
    guess_limits = L.guess(
        "limits",
        question="Het model zonder `sex` geeft 0,07 van de vrouwen en 0,25 van de mannen een ja; van de vrouwen met een ja verdient 0,67 echt meer dan 50.000 dollar. Nu krijgt elke groep een eigen threshold, zodat beide in 0,19 van de gevallen een ja krijgen. Hoeveel van de vrouwen met een ja verdienen dan echt meer, denk je?",
        options={
            "same": "Ongeveer evenveel: 0,67",
            "right": "Minder: zo'n 0,43",
            "more": "Meer: zo'n 0,85",
        },
        reveal="**Minder: zo'n 0,43.** Om meer vrouwen een ja te geven, gaat de threshold voor vrouwen omlaag, en de vrouwen die erbij komen, zijn vrouwen over wie het model minder zeker is. Het deel met een ja is nu gelijk, maar een ja betekent iets anders voor vrouwen dan voor mannen. De demo hieronder laat het zien.",
    )
    guess_limits
    return (guess_limits,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Onafhankelijkheid.** Barocas, Hardt en Narayanan noemen het eenvoudigste criterium in *Fairness and Machine Learning* independence, onafhankelijkheid: de beslissing hangt niet af van de groep. Bij een model dat ja of nee zegt, heeft het veel namen: demographic parity, statistical parity, group fairness, disparate impact en meer. De voorwaarde is dat elke groep even vaak een ja krijgt. Deze course noemt het demographic parity. Cynthia Dwork en anderen (2012) zeggen het zo: de mensen die een positieve (of negatieve) uitkomst krijgen, zijn qua samenstelling precies zoals de hele bevolking. Moritz Hardt en anderen (2016) in één zin: of je bij een beschermde groep hoort, mag niet samenhangen met de beslissing. In deze course is group fairness de hele familie van criteria die getallen tussen groepen vergelijken; demographic parity is het eerste lid, en de volgende les voegt er twee aan toe. <!-- termen:ok -->

    **Verschil of verhouding.** Precies even vaak komt zelden voor, dus meet je hoe ver de delen uit elkaar liggen. De library Fairlearn noemt het deel met een ja het selectiepercentage, zoals [de 80%-regel](/course/open/rai-80-percent-rule?lang=nl) deed, en geeft twee getallen. Het demographic parity difference is het verschil tussen het grootste en het kleinste selectiepercentage: 0 betekent parity. De demographic parity ratio is het kleinste selectiepercentage gedeeld door het grootste, dus een uitkomst van 1 betekent demographic parity. Volgens sommigen, schrijft *Fairness and Machine Learning*, hangt een verhouding van minstens 0,8 samen met de 80%-regel. Toon Calders en Sicco Verwer (2010) gebruikten het verschil; Sainyam Galhotra en anderen (2017) geven een voorbeeld: krijgt 30 procent van de mensen onder de 40 een lening en 40 procent van de mensen boven de 40, dan is het verschil 10 procent.

    **Waarom mensen het willen.** Volgens *Fairness and Machine Learning* zien sommigen er een aanname van gelijkheid in: alle groepen hebben evenveel recht op een ja, dus verdeel je wat er te verdelen is naar verhouding. Het is ook vaak het criterium waar je wiskundig en in code het makkelijkst mee werkt. Alexandra Chouldechova (2017) vindt het geschikt voor werk en toelating tot een opleiding, waar het wenselijk of zelfs wettelijk verplicht kan zijn om mensen in gelijke verhouding aan te nemen of toe te laten over groepen naar afkomst, geslacht of woonplaats. En het neemt het probleem van de vorige les weg: volgens Dwork en anderen maakt het redundant encodings onschadelijk. Een model mag het geslacht via elke proxy vinden die het wil; als elke groep even vaak een ja moet krijgen, levert dat geen groep meer ja's op.

    **Hoe je het bereikt.** Hardt en anderen bereikten het op kredietscores met een threshold per groep, zodat van elke groep hetzelfde deel een ja krijgt. Het model blijft hetzelfde; alleen de **decision threshold** verandert, de kans vanaf waar het ja zegt, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl). Voor elke groep kies je de threshold waarboven een gekozen deel van de kansen ligt: een kwantiel, zoals in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).

    **Probeer het.** De cel hieronder traint het model zonder de kolom `sex` uit [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl). `predict_proba` geeft de kans zelf in plaats van `True` of `False`; `[:, 1]` pakt de kans op een ja. Daarna deelt de cel op drie manieren een ja uit: één threshold van 0,5 voor iedereen; een threshold per groep, zodat elke groep even vaak een ja krijgt als alle mensen samen; en de threshold van 0,5 voor de mannen, terwijl elke vrouw door toeval een ja krijgt, even vaak als de mannen. Voor elke manier laat de cel zien welk deel van de vrouwen en van de mannen een ja krijgt, hun precision (van de mensen met een ja het deel dat echt meer dan 50.000 dollar verdient), en hoe vaak het model gelijk heeft. Voer de cel uit; het duurt een paar seconden.
    """)
    return


@app.cell
def demo_ways():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"
    _tested = _people[_test]
    _rich = _tested["income"] == ">50K"
    _women = _tested["sex"] == "Female"
    _men = _tested["sex"] == "Male"

    # het model zonder de kolom sex, zoals in de vorige les:
    # elke kolom met woorden wordt one-hot-kolommen, daarna loopt elke kolom van 0 tot 1
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")
    _score = _model.predict_proba(_X[_test])[:, 1]   # voor elke persoon de kans op een ja


    def _show(name, yes):
        # per geslacht: het deel met een ja, en de precision: van wie een ja kreeg het deel dat meer verdient
        picked = _tested[yes]
        rates = picked.groupby("sex").size() / _tested.groupby("sex").size()
        rich_picked = picked[picked["income"] == ">50K"]
        precision = rich_picked.groupby("sex").size() / picked.groupby("sex").size()
        right = (yes == _rich).mean()
        print(name, "ja: vrouwen", round(rates["Female"], 2), "mannen", round(rates["Male"], 2),
              "| precision: vrouwen", round(precision["Female"], 2), "mannen", round(precision["Male"], 2),
              "| goed", round(right, 2))


    # 1. één threshold voor iedereen: een ja waar de kans boven 0.5 ligt
    _one = _score > 0.5
    _show("één threshold:          ", _one)

    # 2. een threshold per groep: daarboven ligt in elke groep hetzelfde deel als het deel met een ja in 1
    _rate = _one.mean()
    _threshold_women = np.quantile(_score[_women], 1 - _rate)
    _threshold_men = np.quantile(_score[_men], 1 - _rate)
    # elke vrouw krijgt de threshold van de vrouwen, elke man die van de mannen
    _threshold = _threshold_women * _women + _threshold_men * _men
    _show("een threshold per groep:", _score > _threshold)
    print("de thresholds: vrouwen", round(_threshold_women, 2), "mannen", round(_threshold_men, 2))

    # 3. de mannen als in 1; elke vrouw krijgt door toeval een ja, even vaak als de mannen er een krijgen
    _rng = np.random.default_rng(0)
    _chance = _rng.random(len(_tested)) < _one[_men].mean()
    _show("vrouwen door toeval:    ", _chance * _women + _one * _men)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture({"title": "Drie manieren om een ja uit te delen",
                           "rows": ["Eén threshold", "Een threshold per groep", "Vrouwen door toeval"],
                           "xlabels": ["deel met een ja", "precision"], "key": "open: vrouwen, gevuld: mannen",
                           "decimal": ",",
                           "alt": "Twee panelen met elk drie rijen. Links het deel van de vrouwen en van de mannen met een ja: één threshold 0,07 en 0,25, een threshold per groep 0,19 en 0,19, vrouwen door toeval 0,25 en 0,25. Rechts de precision van vrouwen en mannen: één threshold 0,67 en 0,69, een threshold per groep 0,43 en 0,73, vrouwen door toeval 0,11 en 0,69."})),
        mo.md(r"""
        Met één threshold krijgt 0,07 van de vrouwen een ja en 0,25 van de mannen. Met een threshold per groep krijgen beide 0,19: demographic parity. De threshold voor vrouwen gaat omlaag naar 0,2 en die voor mannen omhoog naar 0,6, en het model heeft iets minder vaak gelijk, 0,81 in plaats van 0,83. Maar een ja betekent nu iets anders: van de vrouwen met een ja verdient 0,43 echt meer dan 50.000 dollar, van de mannen 0,73. Een threshold per groep heeft bij de beslissing ook de kolom `sex` nodig, precies de kolom die het model wegliet; of dat mag, is een vraag voor het recht, zoals in [Directe en indirecte discriminatie](/course/open/rai-discrimination-law?lang=nl). En om de demo kort te houden, kiest hij de thresholds op de test set; in de praktijk kies je ze op een validation set.

        De derde manier voldoet ook aan demographic parity, 0,25 voor beide. Maar de vrouwen worden door toeval gekozen: van hen met een ja verdient maar 0,11 meer, net zoveel als onder alle vrouwen in de test set. *Fairness and Machine Learning* beschrijft precies dit geval: een bedrijf neemt in de ene groep zorgvuldig gekozen sollicitanten aan en in de andere slordig gekozen sollicitanten, even vaak, en achteraf lijkt het dan alsof de tweede groep slechter presteerde. Dwork en anderen wijzen erop dat dit expres kan, om later discriminatie te rechtvaardigen, of zonder kwade bedoeling, door willekeurig te kiezen; hoe dan ook krijgt de groep zo een slechte reputatie.

        Demographic parity kijkt alleen naar wie een ja krijgt, niet naar wie die verdiende. Dat werkt ook andersom. In de data verdient 0,11 van de vrouwen en 0,30 van de mannen meer dan 50.000 dollar, dus een perfect model, dat over iedereen gelijk heeft, voldoet er niet aan: Hardt en anderen merken op dat demographic parity het ideale model niet toestaat. Op 301.536 kredietscores uit 2003 vonden ze dat een geldverstrekker in hun opzet met demographic parity 69,8 procent verdiende van wat er maximaal te verdienen viel, en met één threshold voor iedereen 99,3 procent. En gelijke delen voor vrouwen en mannen zeggen niets over kleinere groepen: dat een groep als geheel aan demographic parity voldoet, betekent niet dat delen van die groep dat doen, schreven Dwork en anderen. Een model kan vrouwen als geheel hun deel van de ja's geven, en oudere vrouwen bijna geen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **group fairness** | criteria die een model beoordelen door getallen tussen groepen te vergelijken, zoals hoe vaak elke groep een ja krijgt |
        | **demographic parity** | elke groep krijgt even vaak een ja, los van de labels; ook onafhankelijkheid (independence) of statistical parity genoemd <!-- termen:ok --> |
        | **demographic parity difference** | het grootste deel met een ja van de groepen min het kleinste; 0 betekent parity |
        | **demographic parity ratio** | het kleinste deel met een ja van de groepen gedeeld door het grootste; 1 betekent parity |
        """),
        L.watch_out("Evenveel ja per groep betekent niet dat de ja's bij de juiste mensen terechtkwamen. Kijk altijd ook naar wie de ja kreeg, bijvoorbeeld naar de precision per groep."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om demographic parity van een model of een regel te meten:

    1. Tel per groep welk deel een ja krijgt.
    2. Neem het verschil: het grootste deel min het kleinste, waarbij 0 parity betekent. En neem de verhouding: het kleinste gedeeld door het grootste, waarbij 1 parity betekent.
    3. Een verhouding onder 0,8 hangt samen met de 80%-regel: een signaal om beter te kijken, geen oordeel.
    4. Kijk naar wie de ja kreeg: per groep de precision. Evenveel ja per groep kan een slordige keuze in één groep verbergen.
    5. Vraag je af of evenveel ja per groep past bij de beslissing. Waar de echte uitkomst per groep verschilt, sluit demographic parity zelfs een perfect model uit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tabel UCI Adult, en een perfect model: het zegt precies ja tegen de mensen die meer dan 50.000 dollar verdienen, dus zijn ja's zijn de rijen waarin `income` gelijk is aan `">50K"`. `_column` is de kolom `sex`. De cel rekent per groep het deel met een ja uit, zoals in [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl), en daarna het verschil en de verhouding met `max()` en `min()`. Voer de cel uit.
    """)
    return


@app.cell
def example_perfect(pd):
    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _column = "sex"

    _yes = _people[_people["income"] == ">50K"]
    _rates = _yes.groupby(_column).size() / _people.groupby(_column).size()
    _difference = _rates.max() - _rates.min()
    _ratio = _rates.min() / _rates.max()
    print(_rates.round(2))
    print("verschil", round(_difference, 2), "verhouding", round(_ratio, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het perfecte model geeft 0,11 van de vrouwen en 0,30 van de mannen een ja: een verschil van 0,19 en een verhouding van 0,36, ver onder de 80%. Een model dat over iedereen gelijk heeft, voldoet dus niet aan demographic parity, omdat de uitkomst zelf tussen de groepen verschilt. Ligt dat aan het model, aan de wereld waar de data vandaan komt, of aan het criterium? De volgende lessen komen op die vraag terug. Eerst een regel van jezelf: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parity():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een regel op opleiding

        Een verzonnen regel voor werving geeft iedereen met minstens een bachelor een ja: een `education_num` van 13 of meer. `column` is de kolom `race`, met de vijf groepen zoals het onderzoek van 1994 ze noemde. Vervang de `...` door de vier regels van het uitgewerkte voorbeeld, met de namen zonder hun liggend streepje: `yes`, `rates`, `difference` en `ratio`. Zet in `yes` de rijen van de regel in plaats van de mensen die meer verdienen. De laatste regel is `difference, ratio`, zodat de check ze allebei ziet. De check probeert ook een andere kolom.
        """),
        L.task("Reken voor de regel `difference` en `ratio` uit van het deel met een ja per waarde van `race`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parity():
    box_parity = L.code_box("parity", lines=12)
    box_parity
    return (box_parity,)


@app.cell(hide_code=True)
def check_parity(box_parity):
    result_parity = L.check("parity", box_parity.value)
    result_parity
    return (result_parity,)


@app.cell(hide_code=True)
def hints_parity():
    L.hints(
        "parity",
        [
            "Gebruik de vier regels van het uitgewerkte voorbeeld. Alleen de rijen die je kiest, veranderen: niet de mensen die meer dan 50.000 dollar verdienen, maar de mensen met een `education_num` van 13 of meer.",
            "De regel is `people[people[\"education_num\"] >= 13]`. Het verschil is `rates.max() - rates.min()`.",
            "`ratio = ___ / rates.max()`.",
        ],
        why="`yes = people[people[\"education_num\"] >= 13]`, `rates = yes.groupby(column).size() / people.groupby(column).size()`, `difference = rates.max() - rates.min()` en `ratio = rates.min() / rates.max()`. Dat geeft een verschil van 0,33 en een verhouding van 0,23.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_care():
    mo.md(r"""
    ### Oefening 2. Wat demographic parity ziet

    Zes manieren om een ja uit te delen. Deel elke manier in bij wat demographic parity erin kan zien.
    """)
    return


@app.cell(hide_code=True)
def sort_care_cell():
    sort_care = L.sort("care")
    sort_care
    return (sort_care,)


@app.cell(hide_code=True)
def check_care(sort_care):
    result_care = L.check("care", sort_care)
    result_care
    return (result_care,)


@app.cell(hide_code=True)
def hints_care():
    L.hints(
        "care",
        [
            "Vraag je bij elke manier twee dingen af: krijgt elke groep even vaak een ja, en worden de mensen met een ja in elke groep op dezelfde manier gekozen?",
            "Kiezen door toeval of op wie het eerst kwam, is niet dezelfde zorg als kiezen op spaargeld of na gesprekken. Een perfect model geeft elke groep het deel van haar labels.",
            "Slordig in één groep: spaargeld en toeval, en \\_\\_\\_. Niet evenveel: het perfecte model en \\_\\_\\_.",
        ],
        why="**Dezelfde zorg:** spaargeld, en een op de vijf. **Slordig in één groep:** spaargeld en toeval, en wie het eerst komt. **Niet evenveel:** het perfecte model, en het model zonder de kolom.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.md(r"""
    ### Oefening 3. Waar het past
    """)
    return


@app.cell(hide_code=True)
def choice_fit_cell():
    choice_fit = L.choice(
        "fit",
        question="Bij welke beslissing past *evenveel ja per groep* het best als doel?",
        options={
            "crime": "Voorspellen wie opnieuw een misdrijf pleegt",
            "right": "Sollicitanten uitnodigen voor een eerste gesprek",
            "disease": "Een ziekte voorspellen die de ene groep vaker krijgt",
        },
        correct="right",
        why={
            "crime": "Chouldechova vindt demographic parity moeilijk te verdedigen bij het voorspellen van wie opnieuw een misdrijf pleegt: daar gaat de discussie over de vraag of de fouten of de betekenis van een risicoscore gelijk moeten zijn. Daar gaat de volgende les over.",
            "right": "Chouldechova vindt het criterium geschikt voor werk en toelating, waar het wenselijk of zelfs wettelijk verplicht kan zijn om mensen uit elke groep in gelijke verhouding aan te nemen. Een eerste gesprek is zo'n beslissing.",
            "disease": "Komt de ziekte echt vaker voor in de ene groep, dan sluit evenveel ja per groep zelfs een perfecte test uit, net als bij het perfecte model uit het uitgewerkte voorbeeld.",
        },
    )
    choice_fit
    return (choice_fit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) deelde je de delen met een ja door elkaar als signaal van disparate impact; hier werd evenveel ja per groep een criterium op zich, demographic parity, met een verschil en een verhouding om het te meten. In [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl) hielden proxies de delen uit elkaar; een threshold per groep sluit het gat, wat de proxies ook zeggen. Maar demographic parity telt alleen de ja's: je indeling liet zien dat het zorg niet van toeval kan onderscheiden. De volgende les, *Gelijke fouten of gelijke betekenis*, kijkt naar wie de ja verdiende: per groep even vaak een fout, en een score die in elke groep hetzelfde betekent.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fit, result_care, result_parity):
    L.wrap_up(result_parity, result_care, choice_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
