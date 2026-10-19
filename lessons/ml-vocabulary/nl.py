# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-vocabulary. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-vocabulary", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl) en [Vragen stellen aan data](/course/open/data-explore?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_coat():
    quick_coat = L.choice(
        "quick-coat",
        question="In de les over regels schrijven en leren van voorbeelden vond de computer de regel voor wanneer je een jas aantrekt. Waar haalde hij die regel uit?",
        options={
            "person": "Uit een regel die een mens eerst opschreef",
            "temps": "Alleen uit de temperaturen van de ochtenden",
            "examples": "Uit ochtenden met hun temperatuur, en of er een jas gedragen werd",
        },
        correct="examples",
        why={
            "person": "Dan zou niemand iets hoeven te leren. Wat kreeg de computer in plaats daarvan? Zie de tabel met nieuwe woorden in [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl).",
            "temps": "Hoe kon hij met alleen de temperaturen de fouten van een regel tellen? Zie het recept in [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl).",
            "examples": "Elke ochtend had een temperatuur en het antwoord, jas of geen jas. De computer probeerde regels en hield de regel met de minste fouten op die voorbeelden.",
        },
    )
    quick_coat
    return (quick_coat,)


@app.cell(hide_code=True)
def quick_check_isna():
    quick_isna = L.choice(
        "quick-isna",
        question="`df` heeft een kolom `price`, waarin een paar prijzen ontbreken. Welke regel houdt de rijen over waarin de prijs ontbreekt?",
        options={
            "rows": "`df[df[\"price\"].isna()]`",
            "count": "`df[\"price\"].isna().sum()`",
            "mask": "`df[\"price\"].isna()`",
        },
        correct="rows",
        why={
            "count": "Dat telt de ontbrekende prijzen: één getal. Hoe houd je rijen over met een mask? Zie het recept in [Tabellen met pandas](/course/open/data-pandas?lang=nl).",
            "mask": "Dat is het mask: `True` of `False` voor elke rij. Waar moet een mask staan om rijen over te houden? Zie het recept in [Tabellen met pandas](/course/open/data-pandas?lang=nl).",
            "rows": "`df[\"price\"].isna()` is `True` waar de prijs ontbreekt, en tussen de vierkante haken van `df` houdt het die rijen over.",
        },
    )
    quick_isna
    return (quick_isna,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De meeste taken in machine learning, van een spamfilter tot een language model, beschrijf je met dezelfde paar woorden. Welke voorbeelden zijn er? Wat weet je van elk voorbeeld? Wat moet het model voorspellen? Van welke voorbeelden leert het, en welke voorspelt het? Met deze woorden kun je voor zo'n taak zeggen wat erin gaat, wat eruit komt, en waar de goede antwoorden vandaan komen.

    **Een voorbeeld uit het dagelijks leven.** Vogels leren herkennen: je bekijkt veel vogels en let op hun grootte, kleur en snavel, en een gids vertelt je de naam van elke vogel. Later zie je een nieuwe vogel, en weet je zelf hoe hij heet.
    """)
    return


@app.cell(hide_code=True)
def predict_houses():
    guess_houses = L.guess(
        "house-features",
        question="Een tabel met huizen heeft de kolommen `area`, `rooms`, `garden` en `price`. Een model moet de prijs voorspellen van huizen die te koop staan. Welke kolommen kan het model volgens jou gebruiken om de prijs te voorspellen van een huis dat te koop staat?",
        options={
            "three": "`area`, `rooms` en `garden`",
            "price": "Alleen `price`",
            "all": "Alle vier de kolommen",
        },
        reveal="**`area`, `rooms` en `garden`.** Die zijn van elk huis bekend, ook van een huis dat te koop staat. De prijs is wat het model moet voorspellen: van een huis dat te koop staat, is die nog niet bekend. Het model leert dus van de eerste drie, en van de prijzen van huizen die verkocht zijn.",
    )
    guess_houses
    return (guess_houses,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## Het idee

    Een dataset voor machine learning is hier een tabel met voorbeelden, één per rij. De kolommen die je van tevoren weet, zijn de features: wat het model krijgt. De kolom die je wilt voorspellen, is het label: het goede antwoord. Training is hier leren van voorbeelden met een label, en inference is het getrainde model gebruiken op nieuwe voorbeelden, die wel features hebben maar nog geen label.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **voorbeeld** | één geval in de data, één rij van de tabel: hier één huis | `df.iloc[0]` |
    | **feature** | één waarde die je van tevoren van een voorbeeld weet, één kolom: hier de oppervlakte | `df["area"]` |
    | **label** | het goede antwoord dat bij een voorbeeld hoort, de kolom die je wilt voorspellen: hier de prijs | `df["price"]` |
    | **training** | een model leert van voorbeelden waarvan het label bekend is | |
    | **inference** | een getraind model voorspelt het label van nieuwe voorbeelden, uit hun features | |

    `df.iloc[0]` is de eerste rij, het eerste voorbeeld: `iloc` pakt rijen op hun plek, vanaf 0, zoals een index in een lijst, niet op hun rijlabel. Dezelfde tabel kun je voor een andere taak met een ander label gebruiken: welke kolom het label is, hangt af van wat je wilt voorspellen.
    """),
    L.watch_out("Een *label* is hier het goede antwoord van een voorbeeld. In [Tabellen met pandas](/course/open/data-pandas?lang=nl) waren de rijlabels de getallen links van elke rij, en in een grafiek is een label de tekst langs een as: dat zijn andere dingen. Een *feature* is geen kenmerk van een product, en *inference* betekent hier alleen een getraind model gebruiken."),
    ])
    return


@app.cell(hide_code=True)
def picture_houses():
    mo.vstack([
        draw.data_tables(C.picture(["de features", "het label"]), title="Features en label",
                         alt="Twee keer de tabel van 10 huizen. Links zijn de kolommen area, rooms en garden gemarkeerd: de features. Rechts is de kolom price gemarkeerd: het label. De laatste twee huizen hebben een vraagteken als prijs."),
        mo.md(r"""
        Dezelfde 10 huizen, twee keer. Links zijn de features gemarkeerd: de oppervlakte in vierkante meters, het aantal kamers, en of er een tuin is (1) of niet (0). Rechts is het label gemarkeerd: de prijs, in duizenden euro's. De huizen 0 tot en met 7 zijn verkocht, dus hun prijs is bekend: die zijn voor training. De huizen 8 en 9 staan te koop: hun prijs is een vraagteken, en die voorspellen is inference.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Een feature moet bekend zijn op het moment dat je voorspelt. De prijs waarvoor een huis verkocht is, kan geen feature zijn om die prijs te voorspellen, en ook niets anders wat pas achteraf bekend is: een model dat daarvan leert, heeft er bij nieuwe voorbeelden niets aan."),
        mo.md(r"""
        ### Recept

        Zo beschrijf je een taak in machine learning met een tabel:

        1. Zeg wat je wilt voorspellen: die kolom is het label.
        2. Kies de features: de kolommen die bekend zijn op het moment dat je voorspelt, en die kunnen helpen.
        3. Haal ze uit de tabel: de features met een lijst namen, het label met één naam.
        4. De rijen met een label zijn voor training; de rijen zonder label zijn voor inference.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De 10 huizen, in `lessons/ml-vocabulary/houses.csv`. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_houses():
    import pandas as pd

    _houses = pd.read_csv("lessons/ml-vocabulary/houses.csv")
    # 1. het label: wat we willen voorspellen
    _labels = _houses["price"]
    # 2. en 3. de features: wat we van tevoren weten
    _features = _houses[["area", "rooms", "garden"]]
    print(_features.shape)
    print(_labels.shape)
    # 4. welke huizen een label hebben, en welke niet
    print(_labels.isna())
    # één voorbeeld: het eerste huis, met zijn features en zijn label
    print(_houses.iloc[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De features zijn een tabel van 10 rijen en 3 kolommen, `(10, 3)`; de labels zijn één kolom van 10 waarden, `(10,)`. Het mask is `False` voor de huizen 0 tot en met 7, die een prijs hebben, en `True` voor de huizen 8 en 9: dat zijn de huizen om te voorspellen. Het eerste voorbeeld is een huis van 62 vierkante meter, met 3 kamers en geen tuin, verkocht voor 245 duizend euro. Het staat er als 62.0, 3.0 en 0.0, omdat één rij één dtype heeft, en de ontbrekende prijzen de tabel floats laten gebruiken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spam():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Het label van een spamfilter
    """)
    return


@app.cell(hide_code=True)
def choice_spam_cell():
    choice_spam = L.choice(
        "spam-label",
        question="Een e-mailprogramma moet nieuwe e-mails sorteren in spam en geen spam. Het leert van oude e-mails die mensen als spam of geen spam markeerden. Wat is in deze taak het label?",
        options={
            "words": "De woorden in de e-mail",
            "marked": "Spam of geen spam, zoals mensen het markeerden",
            "sender": "Wie de e-mail stuurde",
        },
        correct="marked",
        why={
            "words": "De woorden zijn van elke e-mail bekend, ook van een nieuwe. Is dat wat het programma moet voorspellen, of waaruit het voorspelt?",
            "sender": "De afzender is van elke e-mail bekend, ook van een nieuwe. Is dat wat het programma moet voorspellen, of waaruit het voorspelt?",
            "marked": "Spam of niet is wat het programma voor een nieuwe e-mail moet voorspellen, en mensen gaven dat antwoord voor de oude: dat is het label. De woorden en de afzender zijn features.",
        },
    )
    choice_spam
    return (choice_spam,)


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vóór het examen

        Een groep leerlingen wil op de ochtend van een examen voorspellen wie er slaagt. Het vak leest hun tabel als `df`, uit `lessons/ml-vocabulary/students.csv`: voor elke leerling het aantal uur dat die leerde (`hours`), het aantal uur slaap de nacht ervoor (`sleep`), het cijfer dat die leerling voor het examen kreeg, van 1 tot 10 (`grade`), en of die leerling slaagde (`passed`). Maak `features`: de kolommen die features kunnen zijn. Vervang de `...`. De laatste regel is `features`, zodat de check het ziet.
        """),
        L.task("Maak `features`: de kolommen van `df` die vóór het examen bekend zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    box_features = L.code_box("features", lines=6)
    box_features
    return (box_features,)


@app.cell(hide_code=True)
def check_features(box_features):
    result_features = L.check("features", box_features.value)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def hints_features():
    L.hints(
        "features",
        [
            "Stap 2 van het recept: een feature moet bekend zijn op het moment dat je voorspelt, hier vóór het examen.",
            "`passed` is het label. Welke van de andere drie weet een leerling op de ochtend van het examen, en welke pas erna?",
            "`features = df[[___, ___]]`.",
        ],
        why="`hours` en `sleep` zijn vóór het examen bekend. `passed` is het label, het antwoord om te voorspellen, en `grade` is pas na het examen bekend, dus geen van beide kan een feature zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Welke voorbeelden voorspellen?

        Maak `to_predict(path, label)` af. Hij leest het CSV-bestand op `path` als `df`, en geeft de rijen terug waarvan het label ontbreekt, met al hun kolommen: de voorbeelden die een getraind model moet voorspellen. `label` is de naam van de kolom met het label. Vervang de `...`. De check roept `to_predict("lessons/ml-vocabulary/houses.csv", "price")` aan: dat moet de twee huizen geven die te koop staan. De check probeert ook andere tabellen en labels.
        """),
        L.task("Laat `to_predict(path, label)` de rijen van de tabel teruggeven waarvan het label ontbreekt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("to-predict", lines=7)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("to-predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "to-predict",
        [
            "Stap 4 van het recept: de rijen zonder label zijn voor inference. Houd ze over met een mask, zoals in de tweede vraag van de instapcheck.",
            "`.isna()` op de kolom waarvan de naam in `label` zit, geeft het mask: `True` waar het label ontbreekt.",
            "`return df[___]`, met het mask tussen de haken.",
        ],
        why="`df[label].isna()` is `True` voor elke rij waarvan het label ontbreekt, en `df[...]` houdt die rijen over, met al hun kolommen: de features waaruit een model voorspelt.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl) waren de ochtenden de voorbeelden, de temperatuur de feature, en jas of geen jas het label; nu heb je er de woorden voor. In [Vragen stellen aan data](/course/open/data-explore?lang=nl) zocht je ontbrekende waarden; een ontbrekend label wijst een voorbeeld aan om te voorspellen. In de volgende les, *Een model is een functie met parameters*, zie je wat een model met de features van een voorbeeld doet: het maakt er een voorspelling van.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spam, result_features, result_predict):
    L.wrap_up(choice_spam, result_features, result_predict)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
