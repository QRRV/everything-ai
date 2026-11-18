# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-reweighing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-reweighing", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ratio():
    quick_ratio = L.choice(
        "quick-ratio",
        question="Een model geeft een ja aan 0,07 van de vrouwen en 0,25 van de mannen. Wat is de demographic parity ratio?",
        options={
            "difference": "0,18, het verschil tussen beide",
            "right": "0,28, want 0,07 / 0,25",
            "upside": "3,57, want 0,25 / 0,07",
        },
        correct="right",
        why={
            "difference": "0,18 is de demographic parity difference. De ratio deelt het kleinste deel met een ja door het grootste: 0,07 / 0,25 = 0,28. Zie *Het idee* in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).",
            "right": "Het kleinste deel met een ja gedeeld door het grootste: 0,07 / 0,25 = 0,28. Een ratio van 1 zou demographic parity betekenen.",
            "upside": "De ratio deelt het kleinste deel met een ja door het grootste, dus komt hij nooit boven 1: 0,07 / 0,25 = 0,28. Zie *Het idee* in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).",
        },
    )
    quick_ratio
    return (quick_ratio,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="Een threshold per groep gaf vrouwen en mannen even vaak een ja. Wat moet die weten als hij over iemand beslist?",
        options={
            "right": "Het geslacht van die persoon",
            "nothing": "Niets meer dan de eigen kolommen van het model",
            "label": "Of die persoon meer verdient",
        },
        correct="right",
        why={
            "right": "Vrouwen en mannen hebben elk een eigen threshold, dus heeft de beslissing het geslacht van de persoon nodig om er een te kiezen.",
            "nothing": "Het model laat de kolom `sex` weg, maar de threshold niet: vrouwen en mannen hebben elk een eigen threshold. Zie *Het idee* in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).",
            "label": "Of iemand meer verdient, is wat het model raadt; op het moment van de beslissing weet niemand dat. Wat de threshold nodig heeft, is het geslacht van de persoon. Zie *Het idee* in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) gaf het model zonder de kolom `sex` een ja aan 0,07 van de vrouwen en 0,25 van de mannen, en bracht een threshold per groep allebei op 0,19. Dat paste de beslissing aan na het trainen, en het had het geslacht nodig van iedereen over wie het model beslist. Je kunt ook ergens anders beginnen: bij de data waar het model van leert. Van de vrouwen in de data om op te trainen verdient 0,11 meer dan 50.000 dollar, van de mannen 0,31, en dat patroon leert het model. Deze les verandert hoe zwaar elke persoon meetelt terwijl het model leert, zodat het patroon uit de data verdwijnt, en meet hoeveel ervan in het model overblijft.

    **Een voorbeeld uit het dagelijks leven.** Een peiling vraagt 1000 mensen wat ze gaan stemmen. Jongeren antwoorden minder vaak: ze geven 10 procent van de antwoorden, maar zijn 20 procent van de kiezers. Dus telt het peilingbureau het antwoord van elke jongere dubbel, en elk ander antwoord iets minder, tot elke leeftijdsgroep even zwaar weegt als onder de kiezers. Niemands antwoord verandert, alleen hoe zwaar het meetelt.
    """)
    return


@app.cell(hide_code=True)
def predict_ratio():
    guess_ratio = L.guess(
        "ratio",
        question="Zonder gewichten geeft het model vrouwen 0,28 keer zo vaak een ja als mannen. Hoe vaak denk je, als het traint op data waarin elke persoon zo meetelt dat vrouwen en mannen hetzelfde deel hebben dat meer verdient?",
        options={
            "same": "Net als eerst: zo'n 0,28 keer",
            "right": "Dichterbij: zo'n 0,57 keer",
            "equal": "Even vaak: 1 keer",
        },
        reveal="**Dichterbij: zo'n 0,57 keer.** De gewichten halen het verschil uit de labels waar het model van leert, niet uit zijn beslissingen. Vrouwen en mannen verschillen nog in andere kolommen die het model gebruikt, dus komen de delen met een ja dichter bij elkaar, maar worden ze niet gelijk. De demo hieronder laat het zien.",
    )
    guess_ratio
    return (guess_ratio,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Drie plekken om in te grijpen.** Methoden die een model groepen meer gelijk laten behandelen, heten **bias mitigation**, en Barocas, Hardt en Narayanan delen ze in *Fairness and Machine Learning* in naar de plek waar ze ingrijpen. Pre-processing verandert de data, zodat de invoer niet meer samenhangt met het protected attribute. In-training, wat de library AIF360 in-processing noemt, bouwt de fairness-eis in het optimaliseren in dat van de trainingsvoorbeelden een classifier maakt. Post-processing verandert de beslissingen van een getraind model, zodat die niet meer samenhangen met het protected attribute. De threshold per groep uit [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) was post-processing. Elke plek heeft zijn prijs. In-training kan volgens het boek het meeste opleveren, maar dan heb je de ruwe data en de hele training nodig; post-processing werkt voor elke classifier, hoe die vanbinnen ook werkt, zonder opnieuw te trainen. Het team achter AIF360 formuleert de keus als wat je mag veranderen: mag je de data veranderen, dan kan pre-processing.

    **Gewichten in plaats van nieuwe labels.** Faisal Kamiran en Toon Calders (2012) vergeleken manieren om een tabel voor het trainen op te schonen, voor één gevoelige kolom met twee groepen en een label ja of nee. De kolom weglaten haalde het verschil niet weg, zoals in [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl). *Massaging* verandert het label van sommige mensen van wie de kans dicht bij de threshold ligt: een paar nee worden ja in de ene groep, en een paar ja worden nee in de andere. **Reweighing** verandert geen enkel label. Het geeft elke persoon een gewicht: voor hoeveel mensen die persoon meetelt terwijl het model leert, geen weight binnen het model. Het gewicht hangt alleen af van de groep en het label van de persoon. Neem het aantal mensen dat je met die groep en dat label zou verwachten als het label niet van de groep afhing, en deel het door het aantal dat er is. Komt een combinatie minder vaak voor dan verwacht, dan tellen de mensen ervan zwaarder; komt ze vaker voor, dan lichter. In de gewogen data heeft elke groep dan hetzelfde deel van elk label: de onafhankelijkheid uit [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl), nu in de labels in plaats van in de beslissingen.

    **Op UCI Adult.** Van de 32.561 mensen om op te trainen zijn er 10.771 vrouw en 21.790 man, en 7841 verdienen meer dan 50.000 dollar: 0,24 van iedereen. Als het geslacht niets zei over het inkomen, zou 0,24 van de vrouwen meer verdienen: 10.771 × 7841 / 32.561 = 2594 vrouwen. Het zijn er 1179. Dus telt elk van hen als 2594 / 1179 = 2,2 mensen. Dezelfde som geeft 0,85 voor de vrouwen die minder verdienen, 0,79 voor de mannen die meer verdienen en 1,09 voor de mannen die minder verdienen. De gewichten laten de grootte van elke groep gelijk: de vrouwen tellen nog steeds als ongeveer 10.771 mensen, maar nu verdient 0,24 van hen meer, net als van de mannen. Een model in scikit-learn krijgt de gewichten mee in `fit`, als `sample_weight`: volgens de documentatie een gewicht per voorbeeld, en zonder gewichten telt elk voorbeeld als 1.

    **Wat het nodig heeft, en wat het oplevert.** De gewichten hebben het geslacht nodig van iedereen om op te trainen, maar niet van de mensen over wie het model beslist: het model zelf kan de kolom nog steeds weglaten. In de tests van Kamiran en Calders hield reweighing de accuracy hoog, al deed het het minder goed dan massaging. Het team achter AIF360 ziet bij reweighing op UCI Adult, met afkomst als groep, een goede verbetering van de fairness, zonder dat de accuracy er veel onder lijdt.

    **Probeer het.** De cel hieronder rekent de vier gewichten uit op de mensen om op te trainen, en traint het model uit [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) twee keer, zonder de kolom `sex`: een keer zonder gewichten, een keer met. `_cell` markeert de mensen met één geslacht en één label, en `_weight` verzamelt voor elke persoon het gewicht van dat vakje. Voor elk model drukt de cel af hoe vaak het goed zit op de test set, en per geslacht het deel met een ja en de recall: van de mensen die echt meer verdienen, het deel met een ja. Voer de cel uit; het duurt een paar seconden.
    """)
    return


@app.cell
def demo_reweighing():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaanse enquête van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"

    # het model zonder de kolom sex, zoals in de vorige lessen
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _y = _people["income"] == ">50K"

    # 1. de gewichten, op de mensen om op te trainen: per geslacht en label verwacht gedeeld door echt
    _learn = _people[_train]
    _weight = 0
    for _sex in ["Female", "Male"]:
        for _label in [">50K", "<=50K"]:
            _cell = (_learn["sex"] == _sex) * (_learn["income"] == _label)
            _in_group = (_learn["sex"] == _sex).sum()
            _with_label = (_learn["income"] == _label).sum()
            _expected = _in_group * _with_label / len(_learn)
            _w = _expected / _cell.sum()
            _weight = _weight + _w * _cell
            print("gewicht", _sex, _label, round(_w, 2))

    # 2. per geslacht, op de test set: wie het heeft, en wie echt meer verdient
    _tested = _people[_test]
    _rich = _tested["income"] == ">50K"
    _women = _tested["sex"] == "Female"
    _men = _tested["sex"] == "Male"


    def _show(name, weight):
        # train met of zonder gewichten, en meet op de test set
        model = LogisticRegression(max_iter=1000)
        model.fit(_X[_train], _y[_train], sample_weight=weight)
        yes = model.predict_proba(_X[_test])[:, 1] > 0.5
        print(name, "| goed", round((yes == _rich).mean(), 3))
        for sex, group in {"vrouwen": _women, "mannen": _men}.items():
            print("  ", sex, "| deel met een ja", round(yes[group].mean(), 2),
                  "| recall", round(yes[group * _rich].mean(), 2))
        print("   vrouwen krijgen", round(yes[_women].mean() / yes[_men].mean(), 2),
              "keer zo vaak een ja als mannen")


    _show("zonder gewichten", None)
    _show("reweighing", _weight)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_effect({"title": "Het model zonder de kolom sex, zonder en met gewichten",
                                  "rows": ["Zonder gewichten, accuracy 0,834", "Reweighing, accuracy 0,829"],
                                  "xlabels": ["deel met een ja", "recall"],
                                  "key": "open: vrouwen, gevuld: mannen", "decimal": ",",
                                  "alt": "Twee panelen met elk twee rijen. Links het deel met een ja van vrouwen en mannen: zonder gewichten 0,07 en 0,25, met reweighing 0,12 en 0,20. Rechts de recall: zonder gewichten 0,44 en 0,57, met reweighing 0,58 en 0,50. Het model zit zonder gewichten 0,834 van de keren goed en met gewichten 0,829."})),
        mo.md(r"""
        Zonder gewichten kregen vrouwen 0,28 keer zo vaak een ja als mannen; met de gewichten 0,57 keer, 0,116 tegen 0,204. Het model zat iets minder vaak goed: 0,829 in plaats van 0,834 van de keren. De gewichten maakten geslacht en label onafhankelijk in de data waar het model van leert, niet in zijn beslissingen: vrouwen en mannen verschillen nog in andere kolommen die het model wel gebruikt, zoals `relationship` en `occupation`. Kijk ook naar de recall. Van de vrouwen die meer verdienen krijgt nu 0,58 een ja, van zulke mannen 0,50: de vrouwen gingen van onder de mannen naar erboven. Reweighing mikt op een even groot deel met een ja, niet op gelijke fouten; of gelijke fouten belangrijker zijn, is de vraag van [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl).

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **bias mitigation** | methoden die een model groepen meer gelijk laten behandelen: voor, tijdens of na het trainen |
        | **pre-processing** | bias mitigation die de data verandert voor het trainen, zoals reweighing |
        | **in-processing** | bias mitigation die een fairness-eis in het trainen zelf inbouwt; ook in-training genoemd |
        | **post-processing** | bias mitigation die de beslissingen van een getraind model verandert, zoals een threshold per groep |
        | **reweighing** | een gewicht per persoon naar groep en label, het verwachte aantal gedeeld door het echte aantal, zodat elke groep in de gewogen data hetzelfde deel van elk label heeft |
        | **gewicht**, `sample_weight` | voor hoeveel mensen één persoon meetelt terwijl een model leert, geen weight binnen het model; scikit-learn krijgt het mee in `fit(X, y, sample_weight=...)` |
        """),
        L.watch_out("Fairlearn zegt het over zijn eigen mitigation-methoden: ze kunnen een maat voor unfairness dicht bij nul brengen, maar dat betekent niet dat de uitkomsten fair zijn. Reweighing verandert alleen hoe zwaar elk label meetelt. Meet een label het verkeerde, zoals in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl), dan leert het model dat nog steeds, alleen met andere gewichten."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo pas je de data aan met reweighing:

    1. Tel in de data om op te trainen hoeveel mensen er zijn per groep en label, per groep, per label en in totaal.
    2. Reken per groep en label het aantal uit dat je zou verwachten: de grootte van de groep keer de grootte van het label, gedeeld door het totaal. Het gewicht is dat aantal gedeeld door het aantal dat er is.
    3. Geef elke persoon het gewicht van de eigen groep en het eigen label, en geef de gewichten aan `fit` mee als `sample_weight`. De groep is hier nodig, niet als het model beslist.
    4. Meet opnieuw op de test set: per groep het deel met een ja, de fouten en hoe vaak het model goed zit. De gewichten veranderen wat het model leert, niet wat het doet met wat het leerde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen tabel van 10 mensen die dezelfde vier gewichten geeft als het voorbeeld van Kamiran en Calders: 5 vrouwen, van wie 2 het label ja hebben, en 5 mannen, van wie 4 het hebben. 6 van de 10 hebben een ja. De cel rekent per geslacht en label het aantal uit dat je zou verwachten, de grootte van het geslacht keer de grootte van het label gedeeld door iedereen, en het gewicht: dat aantal gedeeld door het aantal dat er is.
    """)
    return


@app.cell
def example_toy():
    # een verzonnen tabel van 10 mensen: per geslacht hoeveel het label ja hebben en hoeveel nee
    _counts = {"vrouwen": {"ja": 2, "nee": 3}, "mannen": {"ja": 4, "nee": 1}}
    _in_group = {"vrouwen": 5, "mannen": 5}
    _with_label = {"ja": 6, "nee": 4}
    _total = 10
    for _group in _counts:
        for _label in _counts[_group]:
            _expected = _in_group[_group] * _with_label[_label] / _total
            print(_group, _label, "| verwacht", _expected, "| er zijn", _counts[_group][_label],
                  "| gewicht", round(_expected / _counts[_group][_label], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vrouwen met een ja tellen 1,5 keer: je zou er 3 verwachten, en het zijn er 2. Met de gewichten hebben de vrouwen 2 × 1,5 = 3 ja op 3 + 3 × 0,67 = 5, en de mannen 4 × 0,75 = 3 op 3 + 1 × 2 = 5: in allebei een deel van 0,6, net als onder alle 10 mensen. Oefening 1 doet hetzelfde op UCI Adult.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De vier gewichten van UCI Adult

        `counts` bevat per geslacht hoeveel van de mensen om op te trainen meer dan 50.000 dollar verdienen (`">50K"`) en hoeveel niet (`"<=50K"`); `"women"` zijn de vrouwen en `"men"` de mannen. `in_group` bevat de grootte van elk geslacht, `with_label` de grootte van elk label, en `total` iedereen. De buitenste loop gaat langs de geslachten met `group` en maakt een lege dict `weight[group]`; de binnenste loop gaat langs de labels met `label`. Vervang de `...` door twee regels, zoals in het uitgewerkte voorbeeld: `expected`, het aantal dat je zou verwachten, en `weight[group][label]`, `expected` gedeeld door het aantal dat er is. De check probeert ook een verzonnen tabel.
        """),
        L.task("Reken `weight[group][label]` uit voor elk geslacht en label in `counts`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weights():
    box_weights = L.code_box("weights", lines=13)
    box_weights
    return (box_weights,)


@app.cell(hide_code=True)
def check_weights(box_weights):
    result_weights = L.check("weights", box_weights.value)
    result_weights
    return (result_weights,)


@app.cell(hide_code=True)
def hints_weights():
    L.hints(
        "weights",
        [
            "Doe wat het uitgewerkte voorbeeld deed, met de namen uit het vak: het aantal dat je zou verwachten is de grootte van het geslacht keer de grootte van het label, gedeeld door iedereen.",
            "`expected = in_group[group] * with_label[label] / total`. Het gewicht deelt dat door het aantal dat er is, `counts[group][label]`.",
            "`weight[group][label] = expected / ___`.",
        ],
        why="`expected = in_group[group] * with_label[label] / total` en `weight[group][label] = expected / counts[group][label]`. Dat geeft 2,20 en 0,85 voor de vrouwen, 0,79 en 1,09 voor de mannen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het deel dat meer verdient, met de gewichten

        `counts` is de tabel uit Oefening 1, en `weight` bevat de gewichten ervan, afgerond. De loop gaat langs de geslachten met `group`; de laatste regel ervan deelt `above` door `everyone` tot `share[group]`. Vervang de `...` door twee regels: `above`, het aantal mensen dat meer verdient keer hun gewicht, en `everyone`, `above` plus het aantal dat minder verdient keer hun gewicht. De check probeert ook andere gewichten.
        """),
        L.task("Reken voor elk geslacht `share[group]` uit, het deel dat meer verdient als de gewichten meetellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=11)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Iemand met een gewicht van 2,2 telt als 2,2 mensen. Vermenigvuldig dus elk aantal met het gewicht van het eigen geslacht en label, zoals het uitgewerkte voorbeeld deed met 2 × 1,5.",
            "`above = counts[group][\">50K\"] * weight[group][\">50K\"]`. `everyone` telt de mensen die minder verdienen er op dezelfde manier bij op.",
            "`everyone = above + counts[group][\"<=50K\"] * ___`.",
        ],
        why="`above = counts[group][\">50K\"] * weight[group][\">50K\"]` en `everyone = above + counts[group][\"<=50K\"] * weight[group][\"<=50K\"]`. Dat geeft 0,24 voor de vrouwen en 0,24 voor de mannen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_needs():
    mo.md(r"""
    ### Oefening 3. Wat reweighing nodig heeft
    """)
    return


@app.cell(hide_code=True)
def choice_needs_cell():
    choice_needs = L.choice(
        "needs",
        question="Je past het model uit de demo aan met reweighing. Wat heb je daarvoor nodig?",
        options={
            "decide": "Het geslacht van de mensen die het beoordeelt",
            "right": "Het geslacht van de mensen om op te trainen",
            "labels": "Nieuwe labels voor een deel van de vrouwen",
        },
        correct="right",
        why={
            "decide": "Dat heeft een threshold per groep nodig, zoals in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl). Reweighing verandert alleen hoe zwaar elke persoon meetelt terwijl het model leert; het model beslist zonder `sex`.",
            "right": "De gewichten komen uit aantallen per geslacht en label, dus heeft iedereen om op te trainen een geslacht nodig. Als het model beslist, niet: het model uit de demo zag de kolom `sex` nooit, alleen de gewichten.",
            "labels": "Labels veranderen is massaging, een andere methode van Kamiran en Calders. Reweighing laat elk label zoals het is en verandert alleen hoe zwaar elke persoon meetelt.",
        },
    )
    choice_needs
    return (choice_needs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) maakte een threshold per groep de beslissingen na het trainen gelijk, en had die het geslacht nodig van iedereen over wie hij besliste. Reweighing begint voor het trainen: het verandert hoe zwaar elke persoon meetelt, zodat in de data elk geslacht hetzelfde deel heeft dat meer verdient. Het heeft het geslacht nodig van de mensen om op te trainen, niet van de mensen over wie het model beslist, en het bracht de delen met een ja dichter bij elkaar, niet gelijk, voor een klein verlies aan accuracy. De drie plekken om in te grijpen, voor, tijdens en na het trainen, kom je bij elke methode weer tegen. De volgende les, *De beslissing aanpassen: een threshold per groep*, gaat terug naar de beslissing, en laat zien wat elk fairness-criterium kost.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_needs, result_shares, result_weights):
    L.wrap_up(result_weights, result_shares, choice_needs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
