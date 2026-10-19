# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-train-test-split. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-train-test-split", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) en [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parameters():
    quick_parameters = L.choice(
        "quick-parameters",
        question="`X` heeft de shape (50, 3): 50 huizen met elk 3 features. Hoeveel getallen leert lineaire regressie?",
        options={
            "weights": "3",
            "right": "4",
            "all": "150",
        },
        correct="right",
        why={
            "weights": "Dat is één weight per feature. Wat telt het model er nog bij op? Zie *Het idee* in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "all": "Het model leert per feature, niet per huis. Zie *Het idee* in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl).",
            "right": "Eén weight per feature, 3, plus de bias term $b$.",
        },
    )
    quick_parameters
    return (quick_parameters,)


@app.cell(hide_code=True)
def quick_check_permutation():
    quick_permutation = L.choice(
        "quick-permutation",
        question="`rng = np.random.default_rng(0)`. Wat geeft `rng.permutation([120, 250, 310])`?",
        options={
            "one": "Eén van de drie getallen, willekeurig gekozen",
            "new": "Drie willekeurige getallen tussen 0 en 1",
            "right": "De drie getallen in een willekeurige volgorde, elk één keer",
        },
        correct="right",
        why={
            "one": "`permutation` houdt elk item; het verandert alleen de volgorde. Zie de tabel in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "new": "Dat geeft `rng.random(size=3)`. Zie de tabel in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "right": "`permutation` geeft elk item één keer terug, in een willekeurige volgorde, als een nieuwe array.",
        },
    )
    quick_permutation
    return (quick_permutation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je traint een model op voorbeelden waarvan je de antwoorden kent, maar je hebt het nodig voor voorbeelden waarvan je de antwoorden nog niet kent: de prijs van een huis dat morgen te koop komt. Een lage loss op de voorbeelden waarvan het leerde, zegt niet hoe goed het daarop werkt. Om dat te weten, houd je een paar voorbeelden met bekende antwoorden achter, train je op de rest, en meet je op de voorbeelden die het model nooit zag.

    **Een voorbeeld uit het dagelijks leven.** Een kok die de soep alleen laat proeven door de mensen die meekookten, hoort dat de soep perfect is. Om te weten of de soep goed is, vraag je iemand die niet in de keuken stond.
    """)
    return


@app.cell(hide_code=True)
def predict_look_up():
    guess_look_up = L.guess(
        "look-up",
        question="Twee modellen leren van dezelfde 32 huizen. Een lijn krijgt daarop een mean squared error van 482. Een opzoekmodel, dat elk huis onthoudt waarvan het leerde, krijgt 0. Welk model voorspelt volgens jou 8 andere huizen beter?",
        options={
            "look-up": "Het opzoekmodel: op zijn huizen zit het er nooit naast",
            "line": "De lijn",
            "same": "Allebei ongeveer even goed",
        },
        reveal="**De lijn.** Op de 8 andere huizen krijgt de lijn een MSE van 648 en het opzoekmodel 1863. Het opzoekmodel geeft een nieuw huis de prijs van het huis uit de training set met de dichtstbijzijnde oppervlakte, dus het zit precies goed op huizen die het al zag, en neemt hun toevallige afwijkingen over bij elk nieuw huis. Zijn 0 zegt niets over nieuwe huizen.",
    )
    guess_look_up
    return (guess_look_up,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een model is nuttig als het werkt op voorbeelden die het niet zag: dat heet **generalisatie**. Dat kun je niet meten op de voorbeelden waarvan het model leerde, dus splits je de voorbeelden waarvan je de antwoorden kent vóór het trainen in tweeën. De **training set**, vaak ongeveer 80%, is waar het model van leert. De **test set**, de rest, blijft tot het eind uit zicht. Dan meet je de loss één keer op de test set: dat is je schatting van hoe het model het doet op nieuwe data.

        Schud voordat je splitst. Data staat vaak gesorteerd, op oppervlakte, op prijs of op klasse, en dan lijkt de laatste 20% niet op de rest. Met `rng.permutation(n)` schud je de posities 0 tot en met $n - 1$; de eerste 80% daarvan gaat naar het trainen, de rest naar het testen. Een array met posities tussen vierkante haken kiest de elementen op die posities, in die volgorde: `sizes[train]`.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **generalisatie** | | hoe goed een getraind model werkt op nieuwe data die het tijdens het trainen niet zag | |
        | **training set** | | de voorbeelden waarop het model getraind wordt | `x_train`, `y_train` |
        | **test set** | | voorbeelden die het model tijdens het trainen nooit zag, en die je aan het eind één keer gebruikt om te meten hoe goed het werkt op nieuwe data | `x_test`, `y_test` |
        | **`rng.permutation(n)`** | | met een geheel getal $n$: de posities 0 tot en met $n - 1$ in een willekeurige volgorde, elk één keer; hetzelfde als `rng.permutation(np.arange(n))` | `rng.permutation(4)` |
        | **elementen op positie** | | een array met posities tussen vierkante haken kiest die elementen, in die volgorde | `sizes[np.array([2, 0])]` |
        """),
        L.watch_out("Gebruik de test set één keer, aan het eind. Probeer je een paar modellen en houd je het model met de beste testscore, dan heeft de test set je helpen kiezen, en is zijn score geen eerlijke schatting meer voor nieuwe data. Een latere les laat zien hoe je kiest zonder hem."),
        mo.md(r"""
        De figuur toont de veertig huizen uit het uitgewerkte voorbeeld hieronder. De gevulde stippen zijn de training set, de open stippen de test set. De ene grafiek heeft een <span class="c-pred">lijn</span>, geleerd van de training set; de andere het <span class="c-pred">opzoekmodel</span>, dat elke oppervlakte de prijs geeft van het huis uit de training set dat er het dichtst bij ligt. Schuif met de slider voor een andere schudbeurt.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_shuffle_cell():
    def _frame(k):
        k = int(k)
        e = {key: L.number(v, 0) for key, v in C.errors(k).items()}
        return (C.picture(k, {"line": "Een lijn", "look": "Een opzoekmodel", "x": "oppervlakte (honderden m²)", "y": "prijs (duizend euro)",
                              "alt": f"Twee grafieken met veertig huizen naar oppervlakte en prijs, 32 gevulde stippen voor de training set en 8 open stippen voor de test set. De ene heeft een rechte lijn erdoor, de andere een trap die naar de prijs van elk huis uit de training set springt. De lijn heeft een test-MSE van {e['line_test']}, het opzoekmodel {e['look_test']}."}),
                f"""Schudbeurt {k}. De lijn: een MSE van <span class="c-loss ea-num">{e['line_train']}</span> op de training set en <span class="c-loss ea-num">{e['line_test']}</span> op de test set. Het opzoekmodel: <span class="c-loss ea-num">0</span> op de training set en <span class="c-loss ea-num">{e['look_test']}</span> op de test set.""")

    explore_shuffle = L.explore("shuffle", "schudbeurt", _frame, start=1, low=1, high=8, step=1, role="", task="Ga alle acht schudbeurten langs: welk model doet het beter op de test set, en verandert dat ooit?")
    explore_shuffle
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij elke schudbeurt doet de lijn het beter op de test set: het opzoekmodel is daar veel slechter, hoe perfect het er op de training set ook uitziet. De test-MSE van de lijn verandert met de schudbeurt, en ligt soms zelfs onder zijn training-MSE: 8 huizen is een kleine test set, en welke erin terechtkomen, is deels toeval.

    ### Recept

    Zo zoek je uit hoe een model het doet op nieuwe data:

    1. Schud de posities van de voorbeelden: `order = rng.permutation(n)`.
    2. Neem ongeveer 80% van de posities om te trainen en de rest om te testen: `train = order[:32]`, `test = order[32:]`.
    3. Kies de voorbeelden op positie: `sizes[train]`, `prices[train]`, en hetzelfde met `test`.
    4. Train alleen op de training set.
    5. Meet aan het eind één keer de loss op de test set: dat is je schatting voor nieuwe data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Veertig verzonnen huizen: de oppervlakte in honderden m², en een prijs in duizenden euro's van 60 plus 150 per honderd m², door toeval hooguit 40 hoger of lager. De lijn wordt getraind met gradient descent, zoals in [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl), met $\eta$ = 0,1. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_split():
    import numpy as np

    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)

    # 1. schud de posities
    _order = _rng.permutation(40)
    print(_order[:10])
    # 2. 32 posities om te trainen, 8 om te testen
    _train = _order[:32]
    _test = _order[32:]
    # 3. de huizen op positie
    _x_train = _sizes[_train]
    _y_train = _prices[_train]
    _x_test = _sizes[_test]
    _y_test = _prices[_test]
    print(len(_x_train), len(_x_test))
    # 4. train alleen op de training set
    _w = 0.0
    _b = 0.0
    for _step in range(2000):
        _r = _y_train - (_w * _x_train + _b)
        _w = _w - 0.1 * (-2 * (_r * _x_train).mean())
        _b = _b - 0.1 * (-2 * _r.mean())
    print(_w, _b)
    # 5. de MSE op de training set, en één keer op de test set
    print(np.mean((_y_train - (_w * _x_train + _b)) ** 2))
    print(np.mean((_y_test - (_w * _x_test + _b)) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het schudden zet huis 2 vooraan, dan huis 33, 6 enzovoort; de eerste 32 posities gaan naar het trainen. De lijn leert ongeveer 153 per honderd m² en 52 erbovenop, dicht bij de 150 en 60 waarmee de huizen gemaakt zijn. Zijn MSE is ongeveer 482 op de training set en ongeveer 648 op de test set: op huizen waarvan hij niet leerde, zit hij er hier verder naast. Dit is schudbeurt 1 in de figuur.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sorted():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Gesorteerde huizen
    """)
    return


@app.cell(hide_code=True)
def choice_sorted_cell():
    choice_sorted = L.choice(
        "sorted",
        question="Een bestand bevat 100 huizen, gesorteerd op oppervlakte, de kleinste eerst. Je traint op de eerste 80 rijen en test op de laatste 20, zonder te schudden. Wat gaat er mis?",
        options={
            "right": "De test set bevat alleen de grootste huizen, groter dan alle huizen waarop het model trainde",
            "nothing": "Niets: 80 om te trainen en 20 om te testen is een goede splitsing",
            "small": "De test set is te klein om iets te meten",
        },
        correct="right",
        why={
            "nothing": "De aantallen kloppen, maar welke huizen komen in de test set? Zie *Het idee*.",
            "small": "Twintig huizen is genoeg om te meten; het probleem is welke twintig. Zie *Het idee*.",
            "right": "De test set moet lijken op de nieuwe huizen die het model tegenkomt. Schud eerst, en beide sets krijgen kleine en grote huizen.",
        },
    )
    choice_sorted
    return (choice_sorted,)


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tien huizen splitsen

        Tien huizen, met hun oppervlaktes in `sizes` en hun prijzen in `prices`. Schud hun posities met `rng` in `order`, zet de eerste 8 posities in `train` en de laatste 2 in `test`. Vervang de drie `...`. De laatste regels kiezen de oppervlaktes van beide sets en laten ze zien, zodat de check ze ziet. De check probeert ook andere huizen.
        """),
        L.task("Splits de tien huizen in een training set van 8 en een test set van 2."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=11)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "Stap 1 en 2 van het recept, met 10 huizen in plaats van 40.",
            "`rng.permutation` met een geheel getal schudt de posities: welk getal hoort bij tien huizen? En hoe neem je de eerste 8 van een array, en de rest?",
            "`order = rng.permutation(10)`, `train = order[___]` en `test = order[___]`.",
        ],
        why="Met seed 1 is de geschudde volgorde `[8 4 7 0 1 2 5 9 6 3]`. De eerste 8 posities gaan naar het trainen; posities 6 en 3 naar het testen: de huizen met een oppervlakte van 1,5 en 1,8.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two_mse():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Twee keer de MSE

        De splitsing uit oefening 2: `x_train` en `y_train` bevatten de 8 huizen van de training set, `x_test` en `y_test` de 2 huizen van de test set. En een lijn met $w = 160$ en $b = 50$ die goed past bij de training set. Zet zijn mean squared error op de training set in `mse_train`, en op de test set in `mse_test`. Vervang de twee `...`. De laatste regel laat ze allebei zien, zodat de check ze ziet. De check probeert ook een andere lijn.
        """),
        L.task("Zet de MSE van de lijn op de training set in `mse_train` en op de test set in `mse_test`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_two_mse():
    box_two_mse = L.code_box("two-mse", lines=12)
    box_two_mse
    return (box_two_mse,)


@app.cell(hide_code=True)
def check_two_mse(box_two_mse):
    result_two_mse = L.check("two-mse", box_two_mse.value)
    result_two_mse
    return (result_two_mse,)


@app.cell(hide_code=True)
def hints_two_mse():
    L.hints(
        "two-mse",
        [
            "Stap 5 van het recept, voor beide sets: het gemiddelde van de gekwadrateerde residuen.",
            "De voorspelling voor een oppervlakte `x` is `w * x + b`. Welke `x` en `y` horen bij elke set?",
            "`mse_train = np.mean((y_train - ___) ** 2)`, en `mse_test` hetzelfde met `x_test` en `y_test`.",
        ],
        why="Op de 8 huizen van de training set liggen de residuen tussen −4 en 6, en hun kwadraten hebben een gemiddelde van 13,5. Op de 2 huizen van de test set voorspelt de lijn 290 en 338 voor 290 en 330: residuen 0 en −8, kwadraten 0 en 64, gemiddelde 32.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl) werd de loss van een model gemeten op de huizen waarvan het leerde; nu meet je hem ook op huizen die het nooit zag, en alleen dat zegt hoe het op nieuwe huizen gaat. In [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) schudde `rng.permutation` een lijst; hier schudt het de posities van de voorbeelden, en de seed maakt de splitsing bij elke run hetzelfde. In de volgende les, *Overfitting en underfitting*, zie je wat er gebeurt als een model te flexibel is, zoals het opzoekmodel, of te eenvoudig.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sorted, result_split, result_two_mse):
    L.wrap_up(choice_sorted, result_split, result_two_mse)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
