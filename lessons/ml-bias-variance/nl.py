# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-bias-variance. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-bias-variance", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_luck():
    quick_luck = L.choice(
        "quick-luck",
        question="Met één validation set van 68 pinguïns kozen acht schudbeurten vijf verschillende waarden van k. Waarom?",
        options={
            "learns": "k-NN leert van dezelfde pinguïns elke keer iets anders",
            "right": "Welke pinguïns in de validation set komen, is toeval, en bij 68 pinguïns beslist één fout meer of minder",
            "test": "De test set was in elke schudbeurt anders",
        },
        correct="right",
        why={
            "learns": "Met dezelfde pinguïns en dezelfde k voorspelt k-NN altijd hetzelfde. Wat veranderde er tussen de schudbeurten? Zie *Waarom* in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "test": "De test set bleef dezelfde 68 pinguïns. Zie het uitgewerkte voorbeeld in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "right": "Een paar pinguïns met geluk of pech geven de doorslag; cross-validation telt over allemaal.",
        },
    )
    quick_luck
    return (quick_luck,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Twee weddenschappen betalen allebei gemiddeld 5 uit. Weddenschap A betaalt 4 of 6, weddenschap B 0 of 10, elk met kans 0,5. Welke heeft de grootste variantie?",
        options={
            "a": "Weddenschap A",
            "right": "Weddenschap B",
            "same": "Geen van beide: allebei hebben ze een verwachtingswaarde van 5",
        },
        correct="right",
        why={
            "a": "Weddenschap A betaalt 1 meer of minder dan 5 uit, weddenschap B 5 meer of minder. Bij welke liggen de uitkomsten verder uit elkaar? Zie *Het idee* in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "same": "De variantie meet hoe ver de uitkomsten van de verwachtingswaarde af liggen. Zie *Het idee* in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "Weddenschap B: 5² × 0,5 + 5² × 0,5 = 25, tegen 1 voor weddenschap A.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je model zit op de validation set verder ernaast dan je zou willen. Meer trainingsvoorbeelden verzamelen, wat tijd en geld kost, of het model veranderen? Dat hangt ervan af waarom het ernaast zit: omdat het te eenvoudig is voor het patroon, of omdat het de toevallige details van zijn eigen trainingsvoorbeelden volgt. Deze les geeft die twee een naam, en laat een grafiek zien die ze uit elkaar houdt.

    **Een voorbeeld uit het dagelijks leven.** Twee weegschalen. De ene geeft altijd 2 kilo te weinig aan: tien metingen zijn het eens, maar zitten allemaal op dezelfde manier ernaast. De andere klopt gemiddeld, maar springt elke keer een kilo omhoog of omlaag. Tien keer wegen en het gemiddelde nemen helpt bij de tweede weegschaal, nooit bij de eerste.
    """)
    return


@app.cell(hide_code=True)
def predict_extra():
    guess_extra = L.guess(
        "extra-temperatures",
        question="Je traint een rechte lijn en een curve van graad 9 op 12 temperaturen van een dag zoals die in *Overfitting en underfitting*, en daarna op 200. Bij welke van de twee denk je dat de extra temperaturen veel uitmaken voor nieuwe temperaturen?",
        options={
            "line": "De rechte lijn",
            "nine": "De curve van graad 9",
            "both": "Bij allebei ongeveer evenveel",
        },
        reveal="**De curve van graad 9.** Met 12 temperaturen volgt hij ook hun ruis, en meer temperaturen nemen dat weg. Geen aantal temperaturen leert een rechte lijn stijgen en dalen. Het uitgewerkte voorbeeld hieronder zet er getallen bij.",
    )
    guess_extra
    return (guess_extra,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Stel dat je hetzelfde soort dag steeds opnieuw meet, en hetzelfde model op elke nieuwe training set traint. Zijn fits, de lijnen of curves die het leert, kunnen dan op twee manieren missen:

        - **Bias**, voluit statistische bias: ze missen op elke training set op dezelfde manier, zoals een rechte lijn die midden op de dag elke keer te laag zit. Dat is underfitting: meer voorbeelden helpen niet, meer capaciteit wel.
        - **Variantie**, de variantie van hun voorspellingen over veel training sets, zoals in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl): ze verschillen veel van de ene training set tot de andere, omdat het model ook de ruis volgt van de voorbeelden die het kreeg. Dat is overfitting: meer voorbeelden helpen, en minder capaciteit ook.

        Meer capaciteit verlaagt de bias en verhoogt de variantie: de **bias-variance trade-off**. Op nieuwe data zit een model ernaast door de ruis, zijn bias en zijn variantie samen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **bias**, statistische bias | | een misser van een model die op elke training set dezelfde is | |
        | **variantie** van een model | | de variantie van zijn voorspellingen over veel training sets: hoeveel ze veranderen van de ene training set tot de andere | |
        | **bias-variance trade-off** | | meer capaciteit geeft minder bias en meer variantie | |
        | **learning curve** | | de training- en validation-MSE tegen het aantal trainingsvoorbeelden | |
        """),
        L.watch_out("Bias betekent in deze course meer dan één ding. Hier komen er drie samen: de **bias term** $b$ uit [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl), een parameter; **statistische bias**, een misser die op elke training set dezelfde is, zoals in deze les; en **data bias**, een scheefheid in een dataset die een model overneemt, zoals groepen die ontbreken of te weinig voorkomen, in de latere les *Wie staat er in de data, en wie ontbreekt?*"),
        mo.md(r"""
        Twintig training sets, gemaakt met de regel uit [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl), elk met eigen uren en eigen ruis. Op elke set worden een rechte lijn en een curve van graad 9 getraind: de twintig <span class="c-pred">fits</span> van allebei in blauw, de regel gestippeld, met een stip om drie uur 's middags, waar hij 20 °C geeft. Verschuif de slider om elke set meer temperaturen te geven.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_sets_cell():
    def _frame(n):
        n = int(n)
        lo1, hi1 = C.spread(1, n)
        lo9, hi9 = C.spread(9, n)
        off = C.far_off(9, n)
        values = {"n": n, "lo1": L.number(lo1, 1), "hi1": L.number(hi1, 1), "lo9": L.number(lo9, 1), "hi9": L.number(hi9, 1), "off": off}
        if off == 20:
            nine = "ver uit elkaar, en ze gaan allemaal buiten de grafiek"
        elif off == 1:
            nine = "dichter rond de boog, maar één ervan gaat nog buiten de grafiek"
        elif off > 0:
            nine = "dichter rond de boog, maar {off} ervan gaan nog buiten de grafiek".format(**values)
        else:
            nine = "dicht rond de boog"
        alt = ("Twee grafieken van de temperatuur over een dag, met de regel als gestippelde boog die om drie uur 's middags 20 °C geeft. "
               "Links twintig rechte lijnen op training sets van {n} temperaturen: ze verschillen van set tot set, en elke lijn ligt midden op de dag onder de boog, van {lo1} tot {hi1} °C om drie uur. "
               "Rechts twintig curves van graad 9: ").format(**values) + nine + ", van {lo9} tot {hi9} °C om drie uur.".format(**values)
        return (C.picture(n, {"line": "Graad 1", "nine": "Graad 9", "hour": "uur van de dag", "temp": "temperatuur (°C)", "alt": alt}),
                """{n} temperaturen per training set. Om drie uur zeggen de twintig <span class="c-pred">rechte lijnen</span> {lo1} tot {hi1} °C, de twintig <span class="c-pred">curves van graad 9</span> {lo9} tot {hi9} °C; de regel geeft 20.""".format(**values))

    explore_sets = L.explore("sets", "n", _frame, start=12, values=C.SIZES, role="",
                             task="Vanaf hoeveel temperaturen per training set liggen de twintig curves van graad 9 dicht bij de gestippelde regel? En wat doen de rechte lijnen intussen?")
    explore_sets
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.vstack([
        mo.md(r"""
        Met 12 temperaturen verschillen ook de rechte lijnen van set tot set, maar ze missen allemaal op dezelfde manier: te laag midden op de dag, te hoog 's ochtends vroeg en 's avonds laat. Dat is bias. De curves van graad 9 gaan alle kanten op, van 13,0 tot 27,3 °C om drie uur, en ze gaan allemaal buiten de grafiek: variantie. Vanaf 100 temperaturen liggen ze dicht rond de regel. Ook de lijnen komen dichter bij elkaar, maar met 400 temperaturen zeggen ze om drie uur nog steeds 17,7 tot 18,3 °C in plaats van 20: meer voorbeelden namen hun variantie weg, niet hun bias.

        ### Eén training set: de learning curve

        Met één training set zie je geen twintig fits, maar je kunt wel op een steeds groter deel ervan trainen, de eerste 12 voorbeelden, de eerste 25, enzovoort, en elke keer de MSE meten op die voorbeelden en op de validation set. Dat is de **learning curve**:

        - **Hoge bias**: beide MSE's vlakken af, dicht bij elkaar en ver boven de ruis. Meer voorbeelden helpen niet; meer capaciteit wel.
        - **Hoge variantie**: de validation-MSE ligt ver boven de training-MSE, en blijft dalen als het aantal voorbeelden groeit. Meer voorbeelden helpen; minder capaciteit ook.
        """),
        L.watch_out("Een learning curve is iets anders dan de loss curve uit [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl), de loss na elke trainingsstap. Sommige bronnen noemen ze allebei een learning curve. En het is geen leercurve: hoe snel iemand iets leert."),
        mo.md(r"""
        De learning curves van het uitgewerkte voorbeeld hieronder, beide assen op een logaritmische schaal: de <span class="c-loss">training-MSE</span> doorgetrokken met gevulde stippen, de <span class="c-loss">validation-MSE</span> gestippeld met open stippen, en de ruis, grijs gestippeld: een MSE van 0,75, die zelfs de regel zelf gemiddeld heeft.
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_curves():
    mo.Html(C.curves_picture({"line": "Graad 1", "nine": "Graad 9", "n": "trainingstemperaturen", "mse": "MSE", "noise": "ruis", "train": "training set", "val": "validation set",
                              "alt": "Twee learning curves, de MSE tegen het aantal trainingstemperaturen van 12 tot 200, beide assen op een logaritmische schaal. Graad 1: de training-MSE blijft tussen 5,1 en 5,7, de validation-MSE daalt van 6,9 naar 5,6; de twee eindigen dicht bij elkaar, ver boven de ruis van 0,75. Graad 9: de training-MSE stijgt van 0,45 naar 0,65, de validation-MSE daalt van 20,9 naar 0,75, tot op de ruis."}))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een learning curve te maken en te lezen:

    1. Houd een validation set apart, en schud de trainingsvoorbeelden.
    2. Voor een groeiend aantal $n$: train op de eerste $n$ trainingsvoorbeelden.
    3. Meet de MSE op die $n$ voorbeelden, en op de validation set.
    4. Vergelijk de twee tegen $n$. Dicht bij elkaar en hoog: hoge bias, dus geef het model meer capaciteit. Ver uit elkaar, met een validation-MSE die nog daalt: hoge variantie, dus meer voorbeelden helpen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Driehonderd verzonnen temperaturen met dezelfde regel, afgelezen op willekeurige uren, dus al geschud: de eerste 200 om op te trainen, de laatste 100 om te valideren. Voor graad 1 en voor graad 9 de learning curve bij 12, 25, 50, 100 en 200 trainingstemperaturen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_learning_curve():
    import numpy as np

    _rng = np.random.default_rng(75)
    _hours = 24 * _rng.random(size=300)
    _temps = 20 - 0.05 * (_hours - 15) ** 2 + 3 * (_rng.random(size=300) - 0.5)
    # 1. 200 om op te trainen, 100 om te valideren; de uren zijn willekeurig, dus al geschud
    _x_train, _y_train = _hours[:200], _temps[:200]
    _x_val, _y_val = _hours[200:], _temps[200:]

    for _degree in [1, 9]:
        for _n in [12, 25, 50, 100, 200]:
            # 2. train op de eerste _n trainingstemperaturen
            _w = np.polyfit(_x_train[:_n], _y_train[:_n], _degree)
            # 3. de MSE op die _n, en op de validation set
            _mse_train = np.mean((_y_train[:_n] - np.polyval(_w, _x_train[:_n])) ** 2)
            _mse_val = np.mean((_y_val - np.polyval(_w, _x_val)) ** 2)
            print(_degree, _n, _mse_train, _mse_val)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke regel geeft de graad, $n$, de training-MSE en de validation-MSE. Graad 1: de training-MSE blijft tussen ongeveer 5,1 en 5,7, en de validation-MSE daalt van ongeveer 6,9 naar 5,6: dicht bij elkaar, ver boven de ruis. Hoge bias, dus stap 4 zegt meer capaciteit. Graad 9: ongeveer 0,45 tegen 20,9 met 12 temperaturen, 0,65 tegen 0,75 met 200: het gat is dicht, want meer voorbeelden namen de variantie weg. Zijn training-MSE gaat een beetje omhoog, omdat het moeilijker is om door meer temperaturen heen te buigen.

    Een andere set temperaturen geeft andere getallen, vooral bij graad 9 met weinig temperaturen; cross-validation bij elke $n$, zoals in de vorige les, geeft een rustiger curve.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Bias of variantie?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "bias-or-variance",
        question="Twintig training sets met huizenprijzen, en op elke set hetzelfde soort model. De twintig fits liggen bijna op elkaar, en elke fit voorspelt grote huizen zo'n 40 duizend euro te goedkoop. Waar heeft dit model het meest last van?",
        options={
            "right": "Hoge bias",
            "variance": "Hoge variantie",
            "neither": "Van geen van beide: de fits zijn het eens, dus het model klopt",
        },
        correct="right",
        why={
            "variance": "Variantie zie je aan fits die verschillen van de ene training set tot de andere. Is dat hier zo? Zie *Het idee*.",
            "neither": "Het eens zijn is nog geen kloppen: alle twintig missen op dezelfde manier, zoals de eerste weegschaal.",
            "right": "Ze missen op elke training set op dezelfde manier: bias. Meer huizen veranderen dat niet; meer capaciteit wel.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_curve():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een learning curve

        Honderdvijftig temperaturen met dezelfde regel: de eerste 100 in `x_train` (uren) en `y_train` (temperaturen), de laatste 50 in `x_val` en `y_val`. Laat in de lus een curve van graad `degree` leren van de eerste `n` trainingstemperaturen, en voeg zijn MSE op die `n` toe aan `train_mses`; zijn validation-MSE voegt het vak al toe aan `val_mses`. Vervang de twee `...`. De laatste regel is `train_mses, val_mses`, zodat de check ze allebei ziet. De check probeert ook graad 1.
        """),
        L.task("Zet voor elke `n` de training-MSE van een curve, geleerd van de eerste `n` trainingstemperaturen, in `train_mses`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_curve():
    box_curve = L.code_box("learning-curve", lines=16)
    box_curve
    return (box_curve,)


@app.cell(hide_code=True)
def check_curve(box_curve):
    result_curve = L.check("learning-curve", box_curve.value)
    result_curve
    return (result_curve,)


@app.cell(hide_code=True)
def hints_curve():
    L.hints(
        "learning-curve",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld, voor één graad.",
            "`np.polyfit` krijgt de eerste `n` trainingsuren, de eerste `n` trainingstemperaturen en de graad: `[:n]` neemt de eerste `n`. De training-MSE meet je op diezelfde `n`.",
            "`w = np.polyfit(x_train[:n], y_train[:n], degree)` en `train_mses.append(np.mean((y_train[:n] - np.polyval(w, ___)) ** 2))`.",
        ],
        why="Met graad 6 zijn de training-MSE's ongeveer 0,08, 0,37, 0,71 en 0,76, en de validation-MSE's ongeveer 1,99, 1,35, 0,83 en 0,78: het gat wordt kleiner als `n` groeit.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_more():
    mo.md(r"""
    ### Oefening 3. De extra data aannemen?
    """)
    return


@app.cell(hide_code=True)
def choice_more_cell():
    choice_more = L.choice(
        "more-data",
        question="Een learning curve: met 250, 500 en 1000 trainingsvoorbeelden is de training-MSE 0,2, 0,3 en 0,4, de validation-MSE 6,0, 4,2 en 3,1. De ruis is ongeveer 0,5. Iemand biedt je 3000 voorbeelden extra aan. Neem je ze aan?",
        options={
            "right": "Ja: hoge variantie, want de validation-MSE ligt ver boven de training-MSE en daalt nog",
            "worse": "Nee: de training-MSE stijgt met meer voorbeelden, dus het model wordt slechter",
            "capacity": "Nee: het model is te eenvoudig en heeft meer capaciteit nodig",
        },
        correct="right",
        why={
            "worse": "Meer voorbeelden zijn moeilijker te volgen, dus de training-MSE stijgt; het gaat om de validation-MSE. Zie het uitgewerkte voorbeeld.",
            "capacity": "Een model dat te eenvoudig is, heeft ook een hoge training-MSE. Is die hier hoog? Zie *Eén training set: de learning curve*.",
            "right": "Een groot gat dat steeds kleiner wordt: meer voorbeelden verlagen de validation-MSE hoogstwaarschijnlijk nog verder.",
        },
    )
    choice_more
    return (choice_more,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl) hielden de training- en test-MSE underfitting en overfitting uit elkaar; bias en variantie zeggen wat er bij elk misgaat, en de learning curve zegt of meer voorbeelden zouden helpen. In [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl) hing het van de pinguïns in de validation set af welke k won; hier hangt het model zelf af van de temperaturen in de training set. In de volgende les, *Regularisatie en weight decay*, verlaag je de variantie van een flexibel model op een andere manier: door grote weights iets te laten kosten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_curve, choice_more):
    L.wrap_up(choice_which, result_curve, choice_more)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
