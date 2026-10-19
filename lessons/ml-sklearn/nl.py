# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-sklearn. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow regression, intercept

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-sklearn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) en [Tabellen met pandas](/course/open/data-pandas?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="Je `fit(X, y, lr, steps)` uit [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl): wat geeft die terug?",
        options={
            "predictions": "De voorspellingen voor elk voorbeeld",
            "right": "De weights `w` en de bias term `b`",
            "losses": "De loss na elke stap",
        },
        correct="right",
        why={
            "predictions": "Voorspellen is een aparte stap, met `X @ w + b`. Wat vindt het trainen? Zie oefening 3 van [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "losses": "De loss daalt onderweg, maar wat komt er aan het eind uit de functie? Zie oefening 3 van [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "right": "Trainen vindt de parameters: `return w, b`. Daarmee voorspelt `X @ w + b`.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def quick_check_columns():
    quick_columns = L.choice(
        "quick-columns",
        question="Wat geeft een tabel met alleen de kolommen `area` en `garden` van `df`?",
        options={
            "right": "`df[[\"area\", \"garden\"]]`",
            "one-pair": "`df[\"area\", \"garden\"]`",
            "plus": "`df[\"area\"] + df[\"garden\"]`",
        },
        correct="right",
        why={
            "one-pair": "Met één paar haken zoekt pandas één kolom met beide namen erin. Zie de waarschuwing in [Tabellen met pandas](/course/open/data-pandas?lang=nl).",
            "plus": "Dat telt de twee kolommen getal voor getal op: één kolom. Zie de tabel in [Tabellen met pandas](/course/open/data-pandas?lang=nl).",
            "right": "De namen in een lijst, binnen de haken: twee paar vierkante haken, en je krijgt een nieuwe tabel met die twee kolommen.",
        },
    )
    quick_columns
    return (quick_columns,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je schreef lineaire regressie zelf: de hellingen, een loop, een learning rate, een aantal stappen. Zo leer je hoe het werkt. Maar voor elk nieuw model zou je een nieuwe loop schrijven, en opnieuw een learning rate en een aantal stappen kiezen. Om een model te gebruiken, neem je meestal een library zoals scikit-learn. Die heeft tientallen modellen, en elk ervan werkt met dezelfde twee stappen: `fit` om van de data te leren, en `predict` om te voorspellen. Ken je die, dan wissel je het ene model voor het andere door één regel te veranderen.

        **Een voorbeeld uit het dagelijks leven.** Elke auto heeft een stuur en pedalen op dezelfde plekken. Kun je in de ene auto rijden, dan kun je ook in een andere rijden, al verschillen de motoren.
        """),
        draw.marked_lines([("Je eigen code", [("w = np.zeros(X.shape[1])", "1"), ("b = 0.0", ""), ("for step in range(steps):", "2"),
                                              ("    grad_w, grad_b = gradients.mse_gradients(X, y, w, b)", ""),
                                              ("    w = w - lr * grad_w", ""), ("    b = b - lr * grad_b", ""), ("y_pred = X @ w + b", "3")]),
                           ("scikit-learn", [("model = LinearRegression()", "1"), ("model.fit(X, y)", "2"), ("y_pred = model.predict(X)", "3")])],
                          {"1": "maak het model", "2": "leer van de data", "3": "voorspel"},
                          title="Maken, fit, predict"),
    ])
    return


@app.cell(hide_code=True)
def predict_weights():
    guess_weights = L.guess(
        "weights",
        question="`LinearRegression` van scikit-learn leert van de acht huizen uit [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl). Wat denk je: hoe verhouden zijn weights zich tot die van je eigen `fit`?",
        options={
            "same": "Ongeveer hetzelfde",
            "different": "Heel anders",
            "lr": "Hij heeft eerst een learning rate nodig",
        },
        reveal="**Ongeveer hetzelfde:** 204,1 voor de oppervlakte, 16,6 voor een tuin, −38,8 voor de afstand, en 77,3 voor de bias term. Het is hetzelfde model met dezelfde loss, dus met hetzelfde laagste punt. Alleen de weg ernaartoe verschilt: scikit-learn rekent het laagste punt rechtstreeks uit, zonder stappen en zonder learning rate.",
    )
    guess_weights
    return (guess_weights,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        In scikit-learn maak je een model, laat je het leren, en vraag je het daarna om voorspellingen. `LinearRegression()` maakt een nieuw model dat nog niets weet. `model.fit(X, y)` laat het leren van de features `X`, een tabel met één kolom per feature, en de labels `y`. Daarna geeft `model.predict(X_new)` één voorspelling per rij van `X_new`, en staan de geleerde parameters in `model.coef_`, de weights, en `model.intercept_`, de bias term. Een ander model uit dezelfde library werkt met precies dezelfde stappen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **scikit-learn** | | een Python-library met veel modellen voor machine learning, allemaal met dezelfde stappen | `import sklearn` |
        | `fit` | | een model zijn parameters laten leren uit de features en de labels | `model.fit(X, y)` |
        | `predict` | | een model dat geleerd heeft laten voorspellen, één getal per rij | `model.predict(X_new)` |
        | `coef_`, `intercept_` | coef, intercept | de weights en de bias term die een lineair model leerde; de `_` aan het eind betekent "pas na `fit` aanwezig" | `model.coef_` |
        """),
        L.watch_out("`fit` wil de features als tabel, ook bij één feature: `df[[\"area\"]]`, met twee paar haken. `df[\"area\"]` is één kolom getallen, en dan stopt `fit` met een fout."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo train en gebruik je een model met scikit-learn:

    1. Zet de features in een tabel `X`, één kolom per feature, en de labels in `y`.
    2. Maak het model: `model = LinearRegression()`.
    3. Laat het leren: `model.fit(X, y)`.
    4. Voorspel met `model.predict(X_new)`; de weights staan in `model.coef_` en de bias term in `model.intercept_`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De acht huizen als tabel. `pd.DataFrame` maakt een tabel van een dictionary: elke key is een kolomnaam, en zijn lijst is die kolom. De kolommen: oppervlakte in honderden m², tuin, afstand tot het station in km, en de prijs in duizenden euro's. Daarna een voorspelling voor een nieuw huis: 100 m², een tuin, 500 m van het station. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_sklearn():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    _df = pd.DataFrame(C.HOUSES)
    # 1. de features als tabel, en de labels
    _X = _df[["area", "garden", "distance"]]
    _y = _df["price"]
    # 2. en 3. maak het model, en laat het leren
    _model = LinearRegression()
    _model.fit(_X, _y)
    print(_model.coef_, _model.intercept_)
    # 4. voorspel voor een nieuw huis, ook als tabel
    _new = pd.DataFrame({"area": [1.0], "garden": [1], "distance": [0.5]})
    print(_model.predict(_new))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De weights zijn 204,1; 16,6 en −38,8, en de bias term 77,3: hetzelfde als waar je eigen `fit` na 1000 stappen uitkwam. Voor het nieuwe huis voorspelt het model ongeveer 278,6 duizend euro. `predict` geeft een array, met één voorspelling per rij van `_new`: hier één.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hoeveel voorspellingen?
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "predict-shape",
        question="`X` heeft 20 rijen en 3 kolommen. Wat geeft `model.predict(X)` na `model.fit(X, y)`?",
        options={
            "weights": "3 getallen: de weights",
            "loss": "Eén getal: de loss",
            "right": "20 getallen: één voorspelling per rij",
        },
        correct="right",
        why={
            "weights": "Die staan in `model.coef_`. Wat geeft `predict` voor elke rij? Zie *Het idee*.",
            "loss": "`predict` vergelijkt niet met de labels. Wat geeft hij voor elke rij? Zie *Het idee*.",
            "right": "Eén voorspelling voor elk van de 20 rijen; de 3 weights staan in `model.coef_`.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Leerlingen met scikit-learn

        `df` bevat zes leerlingen: uren leren, uren slaap, en het cijfer dat ze kregen. `y` is de kolom met cijfers. Zet de features `hours` en `sleep` in `X`, als tabel, en vervang de tweede `...` door de twee regels die een `LinearRegression` maken en laten leren van `X` en `y`. De laatste regel is `model.coef_`, zodat de check de weights ziet. De check probeert ook andere cijfers.
        """),
        L.task("Train een `LinearRegression` op de leerlingen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit-students", lines=10)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit-students", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit-students",
        [
            "Stap 1 tot en met 3 van het recept.",
            "Twee kolommen van een tabel: hoeveel paar vierkante haken? En het model moet bestaan voordat het kan leren.",
            "`X = df[[\"hours\", ___]]`, dan `model = ___()` en `model.___(X, y)`.",
        ],
        why="`df[[\"hours\", \"sleep\"]]` is de tabel met features. `model = LinearRegression()` maakt het model, en `model.fit(X, y)` laat het leren: ongeveer 0,42 per uur leren en 0,26 per uur slaap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wissel het model

        Is het model goed? Vergelijk het met een model dat de features negeert en altijd het gemiddelde cijfer voorspelt: `DummyRegressor` uit scikit-learn. Het vak traint het lineaire model en zet zijn MSE in `mse_model`. Het maakt ook `mean_model`. Vervang de `...` door dezelfde stappen voor `mean_model`: laat het leren, en zet de MSE van zijn voorspellingen in `mse_mean`. De laatste regel laat beide zien. De check probeert ook andere cijfers.
        """),
        L.task("Zet de MSE van het gemiddelde-model in `mse_mean`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-model", lines=16)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-model", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-model",
        [
            "Kijk naar de regels voor `model` er net boven: dezelfde stappen, met het andere model.",
            "Welk model moet nu leren, en wiens voorspellingen gaan in `mse_mean`?",
            "`mean_model.fit(___, ___)`, dan `mse_mean = ((y - mean_model.___(X)) ** 2).mean()`.",
        ],
        why="`mean_model.fit(X, y)` leert het gemiddelde cijfer, 6,33, en `mean_model.predict(X)` voorspelt dat voor elke leerling. Zijn MSE is ongeveer 0,96, tegen 0,016 voor het lineaire model: de features helpen flink.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) bouwde je het model zelf; scikit-learn geeft hetzelfde model, en vele andere, achter dezelfde twee stappen, `fit` en `predict`. De features komen in een tabel, zoals in [Tabellen met pandas](/course/open/data-pandas?lang=nl). Vergelijken met een model dat altijd het gemiddelde voorspelt, vertelt je of een model überhaupt iets geleerd heeft; *Versla altijd een eenvoudige baseline* komt daar later op terug, en *Project: iets echts voorspellen, van begin tot eind* gebruikt scikit-learn voor een heel project. In de volgende les, *Kans: tellen hoe vaak*, gaat het over kans: hoe vaak iets gebeurt, van alle pogingen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_fit, result_mean):
    L.wrap_up(choice_shape, result_fit, result_mean)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
