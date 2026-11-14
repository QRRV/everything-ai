# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-competitions. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-competitions", "nl")
    C = L.checks   # de verzonnen wedstrijd van munten, de publieke en private scores, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) en één over [De standaard voor tabellen: gradient-boosted trees](/course/open/ml-gbdt-default?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_luck():
    quick_luck = L.choice(
        "quick-luck",
        question="Tien modellen krijgen een score op dezelfde validation set, en je houdt de beste. Wat zegt de validation-score van dat model?",
        options={
            "right": "Iets te rooskleurig: de winnaar is deels de gelukkigste op die voorbeelden",
            "fair": "Eerlijk: zo doet hij het op nieuwe voorbeelden",
            "low": "Te somber: de validation set is klein",
        },
        correct="right",
        why={
            "fair": "Eén model op een validation set geeft een eerlijke schatting. De beste van tien niet: wie uit een paar modellen de winnaar kiest, kiest deels het model dat op die voorbeelden het meeste geluk had. Zie de melding *Let op* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "low": "Een kleine validation set maakt de score onrustig, naar beide kanten. Wie de beste van tien houdt, kiest de ruis die omhoog viel. Zie de melding *Let op* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "right": "Daarom meet je aan het eind op de test set. Vandaag zie je wat er gebeurt als iedereen steeds opnieuw naar de test set kijkt.",
        },
    )
    quick_luck
    return (quick_luck,)


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="`HistGradientBoostingRegressor(early_stopping=True)` krijgt `X_val` en `y_val` in `fit`. Wat kiest hij daarmee?",
        options={
            "right": "Hoeveel bomen hij houdt",
            "bins": "Hoeveel bins elke feature krijgt",
            "features": "Welke features hij gebruikt",
        },
        correct="right",
        why={
            "bins": "De bins worden gemaakt uit de trainingsrijen, vóór de eerste boom, met `max_bins`. Zie *Early stopping* in *Het idee* in [De standaard voor tabellen](/course/open/ml-gbdt-default?lang=nl).",
            "features": "Hij gebruikt elke feature die hij krijgt. Zie *Early stopping* in *Het idee* in [De standaard voor tabellen](/course/open/ml-gbdt-default?lang=nl).",
            "right": "Hij stopt zodra 10 bomen op rij de fout op de validation set niet lieten dalen. Op 7.837 uren van 2016 is die keuze veilig; vandaag zie je wanneer kiezen op dezelfde voorbeelden dat niet is.",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In een wedstrijd op een site als Kaggle krijgen teams een training set met labels en een test set zonder. Ze sturen voor elk testvoorbeeld een voorspelling in, een inzending, en een leaderboard toont hun score op een deel van de test set: het **public leaderboard**. De rest van de test set blijft tot het eind verborgen, en de score daarop, het **private leaderboard**, beslist wie wint. Teams mogen meerdere inzendingen per dag doen. Sommigen klimmen op het public leaderboard door het steeds opnieuw te proberen en te houden wat het best scoort, en zakken als het private leaderboard verschijnt. Wat hun overkomt, overkomt iedereen die vaak kiest op dezelfde voorbeelden: in een wedstrijd, op het werk of in een paper.

    **Een voorbeeld uit het dagelijks leven.** Een leerling oefent voor een examen met dezelfde 20 vragen van vorig jaar, steeds opnieuw, tot die elk antwoord kent. De score op die 20 vragen zegt weinig over het examen, dat nieuwe vragen heeft.
    """)
    return


@app.cell(hide_code=True)
def predict_coins_cell():
    guess_coins = L.guess(
        "coins-guess",
        question="200 teams kijken niet eens naar de data: elk stuurt één inzending in met een muntworp voor elk testvoorbeeld, 1 of 0. Het public leaderboard scoort 300 van de 1.000 testvoorbeelden. Wat is de beste publieke score?",
        options={
            "coin": "Ongeveer 50%: een munt is een munt",
            "right": "Ongeveer 57%",
            "high": "Ongeveer 75%",
        },
        reveal="**57,0%.** De beste van 200 keer munten gooien heeft 57,0% van de 300 publieke voorbeelden goed. Op de andere 700 voorbeelden, het private deel, haalt dezelfde inzending 48,6%. Het public leaderboard beloonde geluk.",
    )
    guess_coins
    return (guess_coins,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het leaderboard.** De test set wordt willekeurig verdeeld in een publiek deel en een privaat deel. Elke inzending krijgt op allebei een score: hier het aandeel goed, op 300 publieke en 700 private voorbeelden. Tijdens de wedstrijd toont het **leaderboard** alleen de publieke scores.

        **Elke inzending is een blik.** Een publieke score is een score op een validation set, een set waar je naar kijkt telkens als je iets instuurt. Eén blik geeft een eerlijke schatting. De beste van veel blikken niet: net als in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) is de winnaar deels de gelukkigste. Een muntworp per voorbeeld heeft gemiddeld 50% goed, maar op 300 voorbeelden schommelt zijn score, met een standaardafwijking van ongeveer 2,9 procentpunt. Hoe meer inzendingen, hoe hoger de beste ervan klimt alleen door geluk, terwijl de private score van diezelfde inzending blijft waar het echte kunnen ligt. Kiezen op het publieke deel tot de score daar te mooi is, heet **leaderboard overfitting**.

        **Geluk dat zich opstapelt.** Inzendingen kun je ook combineren. Houd de muntworpen die toevallig boven 50% scoorden op het publieke deel, en laat ze per voorbeeld stemmen. Op het publieke deel had elk van hen iets vaker gelijk dan ongelijk, dus hun stemming heeft daar veel vaker gelijk. Op het private deel weet geen van hen iets, en de stemming dus ook niet. Die stemming bouw je in oefening 2.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **public leaderboard** | de scores op het deel van de test set dat tijdens een wedstrijd zichtbaar is | `right[:, public].mean(axis=1)` |
        | **private leaderboard** | de scores op de rest van de test set, aan het eind getoond; die beslissen wie wint | `right[:, private].mean(axis=1)` |
        | **leaderboard overfitting** | kiezen op het publieke deel tot de score daar te mooi is, zodat de private score lager uitvalt | |
        | `np.argmax(a)` | de plek van de grootste waarde van `a` | `np.argmax([0.2, 0.7, 0.5])` geeft `1` |
        """),
        L.watch_out("De publieke score van je beste inzending is niet je score. Het is de beste van veel blikken op dezelfde paar voorbeelden. Beoordeel je werk op je eigen validatie, zoals cross-validation op de training set, en gebruik het public leaderboard als nog een blik met ruis."),
        mo.md(r"""
        De slider zet hoeveel van de muntworp-inzendingen er ingestuurd worden. Elke stip is een inzending: de publieke score op de bovenste rij, de private score op de onderste. De donkere lijn verbindt de publieke en de private score van de inzending die het best is op het publieke deel. De gestippelde lijn is 50%. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_submissions_cell():
    def _frame(k):
        said = C.strip_said(k, L.lang)
        alt = ("Twee rijen stippen tussen 40% en 65% goed: boven de publieke scores van de eerste inzendingen, onder hun "
               "private scores; een lijn verbindt de twee scores van de beste op het publieke deel. " + said)
        return (C.strip_picture(k, {"title": "Munten op het leaderboard", "x": "aandeel goed", "public": "publieke deel",
                                    "private": "private deel", "alt": alt}), said)

    explore_submissions = L.explore("submissions", "inzendingen", _frame, start=1, values=C.SLIDER, role="hyper",
                                    task="Ga van 1 inzending naar 1.000. Wat doet de beste publieke score, en de private score van diezelfde inzending?")
    explore_submissions
    return


@app.cell(hide_code=True)
def text_after_submissions():
    mo.md(r"""
    Met 10 inzendingen is de beste publieke score 53,7%; met 100 56,7%; met 200 57,0%; met 1.000 61,3%. De private score van de winnaar blijft elke keer rond de 50%: 48,9%, 51,7%, 48,6% en 54,4%. Ook 54,4% is geluk: op 700 voorbeelden schommelt een munt met ongeveer 1,9 procentpunt. Geen enkele inzending weet iets; alleen het geluk op 300 voorbeelden groeit met het aantal pogingen.

    **Ook echte modellen.** Met echte modellen gebeurt het ook, zachter, omdat hun scores echt verschillen. De course nam 24 instellingen van `HistGradientBoostingRegressor` op het verkeer uit [De standaard voor tabellen](/course/open/ml-gbdt-default?lang=nl), en verdeelde de 15.246 testuren willekeurig in een publiek en een privaat deel, 200 keer voor elke grootte. Met 100 publieke uren was de instelling die het best was op het publieke deel in 10% van de verdelingen ook de beste op het private deel; met 300 uren in 32%, met 1.000 in 55%, met 4.500 in 75%. En met 100 publieke uren was de private MAE van de winnaar gemiddeld 6,5 voertuigen per uur hoger dan zijn publieke MAE; met 1.000 1,4 hoger. Hoe kleiner het deel waarop je kiest, en hoe vaker je kiest, hoe meer de winnaar geluk is.

    ### Recept

    Zo houd je een leaderboard eerlijk:

    1. Zet je eigen validatie apart, of gebruik cross-validation op de training set, en kies daarop.
    2. Stuur weinig in, en lees elke publieke score als nog een blik met ruis.
    3. Wantrouw een sprong op het public leaderboard die je eigen validatie niet laat zien.
    4. Kies je laatste inzendingen op je eigen validatie, niet op de publieke score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt de wedstrijd van de slider: 1.000 verborgen labels, 0 of 1; een willekeurige volgorde van de 1.000 voorbeelden, waarvan de eerste 300 het publieke deel zijn en de andere 700 het private deel; en 1.000 inzendingen met een muntworp voor elk voorbeeld. `submissions == y` vergelijkt elke rij met de labels, door broadcasting, en het gemiddelde van een rij True en False is het aandeel goed. De cel print de gemiddelde publieke en private score van alle inzendingen, en daarna, van de eerste 200, welke het best is op het publieke deel, zijn publieke score en zijn private score. Voer de cel uit.
    """)
    return


@app.cell
def example_competition():
    import numpy as np

    _rng = np.random.default_rng(0)
    _y = _rng.integers(0, 2, 1000)                      # de verborgen labels van de 1.000 testvoorbeelden
    _order = _rng.permutation(1000)
    _public = _order[:300]                              # de 300 voorbeelden van het public leaderboard
    _private = _order[300:]                             # de 700 van het private leaderboard
    _submissions = _rng.integers(0, 2, (1000, 1000))    # 1.000 inzendingen: een muntworp voor elk voorbeeld

    _right = _submissions == _y                         # True waar een inzending goed is
    _public_scores = _right[:, _public].mean(axis=1)
    _private_scores = _right[:, _private].mean(axis=1)
    print(round(_public_scores.mean(), 3), round(_private_scores.mean(), 3))

    # de eerste 200 inzendingen: de beste op het publieke deel, en dezelfde op het private deel
    _first = _public_scores[:200]
    _best = np.argmax(_first)
    print(_best, _first[_best], round(_private_scores[_best], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Gemiddeld hebben de inzendingen 49,9% goed op het publieke deel en 50,0% op het private deel, zoals munten horen te doen. Van de eerste 200 is inzending 115 het best op het publieke deel, met 57,0%, en die haalt 48,6% op het private deel. Op een leaderboard van deze 200 had die tijdens de wedstrijd bovenaan gestaan, en aan het eind op plek 159, gedeeld met twee andere.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_of():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De winnaar van het public leaderboard

        `best_of(public_scores, private_scores)` krijgt de publieke en de private score van elke inzending, in dezelfde volgorde, en moet de private score geven van de inzending met de hoogste publieke score: wat de leider van het public leaderboard aan het eind haalt. Vervang de twee `...`. De check roept `best_of(public_scores, private_scores)` aan met de scores van de eerste 200 inzendingen uit het uitgewerkte voorbeeld, en probeert drie korte arrays.
        """),
        L.task("Laat `best_of` de leider van het public leaderboard vinden en de private score daarvan geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_of():
    box_best_of = L.code_box("best-of", lines=6)
    box_best_of
    return (box_best_of,)


@app.cell(hide_code=True)
def check_best_of(box_best_of):
    result_best_of = L.check("best-of", box_best_of.value)
    result_best_of
    return (result_best_of,)


@app.cell(hide_code=True)
def hints_best_of():
    L.hints(
        "best-of",
        [
            "Twee stappen: zoek waar de hoogste publieke score staat, en kijk dan op diezelfde plek naar de private score.",
            "Je hebt de plek van de hoogste waarde nodig, niet de waarde zelf; NumPy heeft daar een functie voor, zoals in het uitgewerkte voorbeeld. De private score van die inzending staat op dezelfde plek in `private_scores`.",
            "`best = np.___(public_scores)` en `return private_scores[___]`.",
        ],
        why="De leider van het public leaderboard van 200 keer munten gooien scoort daar 57,0% en op het private deel 48,6%. De beste private score van de 200 is 56,4%, maar niemand kon zien welke dat was.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stemming van wie geluk had

        `vote(submissions, public_scores)` krijgt de inzendingen, een rij per inzending en een kolom per voorbeeld, en hun publieke scores. Hij moet de inzendingen houden met een publieke score boven 0,5, en voor elk voorbeeld geven of meer dan de helft van hen 1 zegt. Vervang de twee `...`. De check roept `vote(submissions, public_scores)` aan met de eerste 200 inzendingen uit het uitgewerkte voorbeeld, en probeert drie verzonnen sets, elk met één publieke score van precies 0,5.
        """),
        L.task("Laat `vote` de inzendingen combineren die geluk hadden op het publieke deel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("vote", lines=5)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "vote",
        [
            "Twee stappen: houd de rijen van de inzendingen met geluk, en neem dan voor elke kolom, elk voorbeeld, het aandeel van hen dat 1 zegt.",
            "Een mask tussen de blokhaken houdt de rijen waar hij True is, zoals in [Delen van een array kiezen](/course/open/np-indexing?lang=nl). Het gemiddelde van nullen en enen is het aandeel enen, en `axis` zegt of het langs de kolommen omlaag loopt of langs de rijen.",
            "`lucky = submissions[public_scores ___ 0.5]` en `votes = lucky.mean(axis=___)`.",
        ],
        why="Van de eerste 200 inzendingen scoren er 100 boven 50% op het publieke deel. Hun stemming heeft daar 67,0% goed, en op het private deel 50,9%. Met alle 1.000 inzendingen haalt de stemming 84,0% op het publieke deel en 49,9% op het private deel: hoe hoger de publieke score, hoe meer geluk erin zit.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_final():
    mo.md(r"""
    ### Oefening 3. De laatste inzendingen
    """)
    return


@app.cell(hide_code=True)
def choice_final_cell():
    choice_final = L.choice(
        "final-pick",
        question="Een wedstrijd eindigt morgen, en je mag twee van je inzendingen kiezen voor het private leaderboard. Welke twee kies je?",
        options={
            "right": "De twee met de beste score in je eigen cross-validation op de training set",
            "public": "De twee met de beste publieke score",
            "vote": "Een stemming van je inzendingen die het best deden op het publieke deel",
        },
        correct="right",
        why={
            "public": "De publieke score van je beste inzending is de beste van veel blikken op dezelfde paar voorbeelden: de leider van 200 keer munten gooien had daar 57,0% en op het private deel 48,6%.",
            "vote": "Dat is de stemming uit oefening 2: 67,0% op het publieke deel en 50,9% op het private deel. Die telt het geluk op het publieke deel op, verder niets.",
            "right": "Je cross-validation gebruikt je hele training set, meer voorbeelden dan het publieke deel, en die heb je niet beklommen door in te sturen. Na veel keuzes kan ook die iets te mooi uitvallen, en daarom heeft het private leaderboard, de test set, het laatste woord.",
        },
    )
    choice_final
    return (choice_final,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het public leaderboard is een validation set waar iedereen steeds opnieuw naar kijkt, dus de waarschuwing uit [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) groeit met elke inzending. De stemming uit oefening 2 is de majority voting uit [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl), van modellen die niets weten. Early stopping in [De standaard voor tabellen](/course/open/ml-gbdt-default?lang=nl) kiest ook op een validation set, maar het kiest één getal, hoeveel bomen, op 7.837 uren. In de volgende les, *Als één klasse zeldzaam is*, kan een model dat 97% goed heeft nutteloos zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_final, result_best_of, result_vote):
    L.wrap_up(result_best_of, result_vote, choice_final)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
