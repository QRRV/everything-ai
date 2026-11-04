# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-kmeans. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-kmeans", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl) en [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Je hebt de metingen van 342 pinguïns, maar niemand heeft hun soort opgeschreven. Je wilt dat een model er groepen in vindt. Welke soort leren is dat?",
        options={
            "supervised": "Supervised learning",
            "right": "Unsupervised learning",
            "rl": "Reinforcement learning",
        },
        correct="right",
        why={
            "supervised": "Supervised learning heeft voor elk voorbeeld een label nodig, en hier is er geen. Zie [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "rl": "Reinforcement learning leert van beloningen voor acties; hier zijn er alleen metingen. Zie [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "right": "Er zijn helemaal geen labels: het model heeft alleen de metingen om op af te gaan.",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="De snavellengte verschilt tussen pinguïns ongeveer 5 mm, de flipperlengte ongeveer 14 mm. Wat gebeurt er met de afstand tussen twee pinguïns als je ze niet standaardiseert?",
        options={
            "same": "Niets: millimeters zijn millimeters",
            "right": "De flipperlengte telt zwaarder mee in de afstand dan de snavellengte",
            "bill": "De snavellengte telt zwaarder mee, omdat die de eerste feature is",
        },
        correct="right",
        why={
            "same": "Elke millimeter telt even zwaar, maar flipperlengtes verschillen met meer millimeters, dus maken die het grootste deel van de afstand. Zie het uitgewerkte voorbeeld in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "bill": "De volgorde van de features maakt voor een afstand niet uit; hoeveel elke feature varieert wel. Zie het uitgewerkte voorbeeld in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "Standaardiseren geeft elke feature een standaardafwijking van 1, zodat ze allemaal even zwaar meetellen in de afstand.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe kwam elk voorbeeld met zijn label: de soort van een pinguïn, de prijs van een huis. Vaak heb je alleen de metingen. Een winkel weet wat zijn klanten kopen, maar niet wat voor soorten klanten hij heeft; een bioloog heeft dieren gemeten, maar weet nog niet hoeveel soorten het zijn. Dan wil je dat de data zelf laat zien welke voorbeelden bij elkaar horen. Dat doet deze les met de pinguïns, zonder het model hun soort te vertellen, en pas aan het eind kijk je hoe goed de groepen bij de soorten passen.

    **Een voorbeeld uit het dagelijks leven.** Je kiepert een pot kralen om en sorteert ze in drie stapels, zonder dat iemand je vertelt welke soorten er zijn. Je begint met drie kralen die er verschillend uitzien, legt elke kraal op de stapel waar hij het meest op lijkt, en kijkt dan opnieuw wat elke stapel geworden is: een stapel met vooral grote kralen staat nu voor "groot". Je verlegt een paar kralen, kijkt weer, en stopt als er geen kraal meer verhuist.
    """)
    return


@app.cell(hide_code=True)
def predict_found():
    guess_found = L.guess(
        "species-found",
        question="k-means verdeelt de 342 pinguïns in 3 groepen op snavellengte en flipperlengte, gestandaardiseerd, en ziet hun soort nooit. Daarna tel je voor elke groep de pinguïns van de soort die er het vaakst in zit. Hoeveel van de 342 zijn dat volgens jou?",
        options={
            "third": "Ongeveer een derde, zoals bij toeval",
            "twothirds": "Ongeveer twee derde",
            "right": "Bijna allemaal",
        },
        reveal="**327 van de 342**, 96%. Eén groep heeft 122 Gentoos, één 146 Adelies, één 59 Chinstraps, en maar 15 pinguïns zitten in de groep van een andere soort. Zonder standaardiseren, dus met flipperlengtes die ongeveer tweeënhalf keer zo ver uiteenlopen als snavellengtes, zijn het er 287.",
    )
    guess_found
    return (guess_found,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Clustering** verdeelt voorbeelden zonder labels in groepen, zo dat voorbeelden in een groep op elkaar lijken en voorbeelden in verschillende groepen niet. Zo'n groep heet een **cluster**. Je kunt een cluster beschrijven met zijn **centroid**: het gemiddelde van zijn voorbeelden, feature voor feature, zijn middelpunt.

        **k-means** is de bekendste manier om te clusteren. Je kiest het aantal clusters, k, en k centroids om mee te beginnen, bijvoorbeeld k willekeurig gekozen voorbeelden. Dan herhaalt k-means twee stappen:

        1. **Toewijzen**: elk voorbeeld gaat naar het cluster van zijn dichtstbijzijnde centroid.
        2. **Verschuiven**: elke centroid gaat naar het gemiddelde van de voorbeelden in zijn cluster.

        Laat een stap elk voorbeeld in het cluster waar het al zat, dan verschuiven de centroids ook niet meer, en is k-means klaar. Dichtstbij betekent de afstand uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), dus je standaardiseert eerst. k-means leert de centroids; wat een cluster *is*, leert het nooit. Welk cluster de Adelies zijn, ontdek je door te kijken.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **clustering** | voorbeelden zonder labels verdelen in groepen van voorbeelden die op elkaar lijken | |
        | **cluster** | één zo'n groep | `labels == j` |
        | **centroid** | het gemiddelde van de voorbeelden in een cluster, feature voor feature | `X[labels == j].mean(axis=0)` |
        | **k-means** | clusteren door twee stappen te herhalen: elk voorbeeld naar zijn dichtstbijzijnde centroid, elke centroid naar het gemiddelde van zijn voorbeelden | |
        | `KMeans(n_clusters=3, n_init=10, random_state=0)` | k-means in scikit-learn: 3 clusters, vanaf 10 starts, waarvan het de beste houdt; `random_state=0` maakt de willekeurige starts elke keer hetzelfde. `.fit(X)` voert het uit en geeft het model terug, dus `km = KMeans(...).fit(X)` is het gefitte model | |
        | `km.labels_`, `km.cluster_centers_` | na `fit`: het cluster van elk voorbeeld, en de centroids | |
        """),
        L.watch_out("De nummers van de clusters betekenen niets. Cluster 0 is niet \"de eerste soort\": een andere start, of scikit-learn, kan dezelfde groepen andere nummers geven."),
        mo.md(r"""
        Het plaatje toont k-means met de hand op de 342 pinguïns, op snavellengte en flipperlengte, gestandaardiseerd. Het begint bij drie willekeurig gekozen pinguïns, twee Gentoos en een Adelie. Links elke pinguïn in de kleur en vorm van zijn cluster, en de drie centroids in oranje, met een pijl vanaf waar ze een stap eerder lagen; elk cluster heeft de kleur van de soort die de meeste van zijn pinguïns aan het eind hebben. Rechts de soorten, die k-means nooit zag: een rondje is een Adelie, een driehoek een Chinstrap en een vierkant een Gentoo. Hier betekenen kleur en vorm een groep. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    def _frame(step):
        step = int(step)
        centers, assigned, changed = C.steps()[step]
        counts = [int((assigned == j).sum()) for j in range(3)]
        if step == 0:
            said = f"""De start: elke pinguïn gaat naar zijn dichtstbijzijnde startcentroid. De clusters hebben {counts[0]}, {counts[1]} en {counts[2]} pinguïns: twee centroids delen de meeste Gentoos, en één krijgt alle Adelies en de meeste Chinstraps."""
        elif changed > 0:
            said = f"""Stap {step}: de centroids zijn naar de gemiddelden van hun clusters verschoven, en daarna veranderden {changed} pinguïns van cluster. De clusters hebben {counts[0]}, {counts[1]} en {counts[2]} pinguïns."""
        else:
            said = f"""Stap {step}: de centroids zijn nog één keer verschoven, en geen enkele pinguïn veranderde van cluster: k-means is klaar. De clusters hebben {counts[0]}, {counts[1]} en {counts[2]} pinguïns, waarvan {C.matched(C.penguins()["species"], assigned)} in het cluster van hun eigen soort."""
        alt = (f"De eerste grafiek: de 342 pinguïns op snavellengte, 32 tot 60 mm, en flipperlengte, 172 tot 231 mm, in drie clusters na stap {step} van k-means, met hun drie centroids. "
               + "De tweede grafiek: dezelfde pinguïns per soort, met de centroids waarop k-means eindigt; Adelies hebben korte snavels en flippers, Chinstraps lange snavels en korte flippers, Gentoos lange flippers. "
               + said.replace("**", ""))
        return (C.picture(step, {"title": f"De clusters, stap {step}", "species": "De soorten, die k-means niet ziet",
                                 "x": "snavellengte (mm)", "y": "flipperlengte (mm)", "alt": alt}), said)

    explore_steps = L.explore("steps", "stap", _frame, start=0, values=C.STEPS, role="",
                              task="Ga langs de stappen. Waar gaat de derde centroid heen, en wanneer verandert er niets meer?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Stap voor stap komt de derde centroid van de Gentoos omlaag naar de lange snavels en korte flippers van de Chinstraps, en geeft de tweede zijn Chinstraps aan hem af. Er veranderen 50, 24, 16, 20, 11, 4 en dan 0 pinguïns van cluster: na stap 7 verschuift er niets meer. De drie centroids eindigen op 47,6 en 217,0 mm, 38,7 en 189,6 mm, en 49,0 en 195,7 mm, dicht bij de gemiddelden van de Gentoos, de Adelies en de Chinstraps.

    ### Recept

    Om te clusteren met k-means:

    1. Kies de features, en standaardiseer ze.
    2. Kies k, en neem k voorbeelden als startcentroids.
    3. Toewijzen: geef elk voorbeeld het cluster van zijn dichtstbijzijnde centroid.
    4. Verschuiven: zet elke centroid op het gemiddelde van de voorbeelden in zijn cluster.
    5. Herhaal 3 en 4 tot geen enkel voorbeeld meer van cluster verandert.
    6. Kijk naar de clusters: wat hebben de voorbeelden in elk cluster gemeen? Met labels die je achterhield, tel je hoe goed ze passen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 tot en met 5 van het recept op zes pinguïns, weinig genoeg om met het oog te volgen: in de tabel van 342 zijn de pinguïns in rij 0, 1 en 2 Adelies, die in rij 151, 152 en 153 Gentoos. Gestandaardiseerd met het gemiddelde en de standaardafwijking van alle 342, met k = 2, en de eerste twee pinguïns, allebei Adelies, als startcentroids. Voor elke ronde het cluster van elk van de zes; na drie rondes de centroids. `np.array(dists)` heeft een rij voor elke centroid en een kolom voor elke pinguïn, dus `np.argmin(..., axis=0)` geeft elke pinguïn de rij van zijn dichtstbijzijnde centroid. Aan het eind doet scikit-learn hetzelfde met `KMeans`. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_kmeans():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["flipper_length_mm"].isna() == False]
    _raw = np.array(_df[["bill_length_mm", "flipper_length_mm"]])
    # 1. standaardiseer, en neem zes pinguïns
    _X = (_raw - _raw.mean(axis=0)) / _raw.std(axis=0)
    _six = _X[[0, 1, 2, 151, 152, 153]]
    # 2. k = 2, en de eerste twee pinguïns als startcentroids
    _centers = _six[[0, 1]]
    # 5. herhaal 3 en 4: in ronde 2 verandert er niets meer
    for _round in range(3):
        # 3. elke pinguïn naar zijn dichtstbijzijnde centroid
        _dists = []
        for _c in _centers:
            _dists.append(np.linalg.norm(_six - _c, axis=1))
        _labels = np.argmin(np.array(_dists), axis=0)
        print(_round, _labels)
        # 4. elke centroid naar het gemiddelde van zijn pinguïns
        _means = []
        for _j in range(2):
            _means.append(_six[_labels == _j].mean(axis=0))
        _centers = np.array(_means)
    print(np.round(_centers, 2))
    # hetzelfde met scikit-learn
    _km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_six)
    print(_km.labels_)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In ronde 0 ligt alleen de eerste pinguïn het dichtst bij de eerste centroid; de andere vijf, ook de Gentoos, gaan naar de tweede. Die centroid verschuift naar het gemiddelde van die vijf, richting de Gentoos, dus in ronde 1 liggen de twee andere Adelies dichter bij de eerste centroid. In ronde 2 verandert er niets: de Adelies zijn cluster 0, de Gentoos cluster 1. scikit-learn vindt dezelfde twee groepen, maar noemt de Adelies cluster 1: de nummers betekenen niets.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén stap met de hand

        Het vak leest alle 342 pinguïns in en standaardiseert ze in `X`, en neemt rij 216, 54 en 240 als startcentroids: de start van het plaatje. Vervang de drie `...`: de afstand van elke pinguïn tot centroid `c`, de dichtstbijzijnde centroid van elke pinguïn, en het gemiddelde van de pinguïns van cluster `j`. De laatste regel maakt van de nieuwe centroids weer millimeters.
        """),
        L.task("Doe met de hand één keer toewijzen en één keer verschuiven op alle pinguïns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("one-step", lines=19)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("one-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "one-step",
        [
            "Stap 3 en 4 van het uitgewerkte voorbeeld, nu met `X` in plaats van `_six` en 3 clusters in plaats van 2.",
            "De afstand is `np.linalg.norm` van het verschil tussen alle pinguïns en één centroid, langs `axis=1`; de dichtstbijzijnde is `np.argmin` langs `axis=0`; een centroid is het gemiddelde van een deel van `X`.",
            "`dists.append(np.linalg.norm(X - c, axis=___))`, `labels = np.argmin(np.array(dists), axis=___)` en `new.append(X[labels == j].mean(axis=0))`.",
        ],
        why="`np.array(dists)` heeft een rij voor elk van de 3 centroids en een kolom voor elk van de 342 pinguïns, dus `np.argmin` langs `axis=0` geeft elke pinguïn zijn dichtstbijzijnde centroid. `labels == j` kiest dan de pinguïns van cluster `j`, en hun gemiddelde is zijn nieuwe centroid.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stable():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tot er niets meer verandert

        Stap 5 van het recept. Het vak heeft dezelfde start, en een loop met beide stappen. `labels` begint op −1 voor elke pinguïn, nog geen cluster; `new` is het cluster dat elke pinguïn in deze stap krijgt. Vervang de twee `...`: hoeveel pinguïns van cluster veranderden, en wanneer de loop stopt. De laatste regel geeft de stap waarbij hij stopte, en hoeveel pinguïns elk cluster heeft.
        """),
        L.task("Herhaal de twee stappen tot geen enkele pinguïn meer van cluster verandert."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stable():
    box_stable = L.code_box("until-stable", lines=26)
    box_stable
    return (box_stable,)


@app.cell(hide_code=True)
def check_stable(box_stable):
    result_stable = L.check("until-stable", box_stable.value)
    result_stable
    return (result_stable,)


@app.cell(hide_code=True)
def hints_stable():
    L.hints(
        "until-stable",
        [
            "Een pinguïn veranderde van cluster als zijn cluster in `new` anders is dan dat in `labels`. Hoe tel je hoe vaak een vergelijking `True` is?",
            "`new != labels` geeft `True` of `False` voor elke pinguïn, en `np.sum` telt de `True`'s. De loop stopt met `break` als er niets veranderde.",
            "`changed = np.sum(new ___ labels)` en `if changed == ___:`.",
        ],
        why="Verandert geen enkele pinguïn van cluster, dan blijven de centroids, de gemiddelden van dezelfde pinguïns, ook waar ze zijn, en zou elke volgende stap hetzelfde zijn. Dus daar is k-means klaar; door de −1 aan het begin telt stap 0 altijd alle 342 pinguïns als veranderd.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_four():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Alle vier de metingen, met scikit-learn

        Nu k-means uit scikit-learn, op alle vier de metingen: snavellengte, snavelhoogte, flipperlengte en lichaamsgewicht, gestandaardiseerd in `X4`. Vervang de twee `...`: `KMeans` met 3 clusters, `n_init=10` en `random_state=0`, gefit op `X4`, en voor cluster `j` het aantal pinguïns van de soort die er het vaakst in zit. `right` telt die dan op, zoals in de gok aan het begin.
        """),
        L.task("Cluster de pinguïns op vier metingen, en tel hoeveel er in het cluster van hun eigen soort zitten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_four():
    box_four = L.code_box("four", lines=16)
    box_four
    return (box_four,)


@app.cell(hide_code=True)
def check_four(box_four):
    result_four = L.check("four", box_four.value)
    result_four
    return (result_four,)


@app.cell(hide_code=True)
def hints_four():
    L.hints(
        "four",
        [
            "De nieuwe woorden bij *Het idee* zeggen hoe je `KMeans` maakt en uitvoert. `np.unique(..., return_counts=True)` geeft voor één cluster elke soort erin en hoeveel pinguïns van elke soort.",
            "`km` is het gefitte model, dus `KMeans(...)` met daarachter `.fit(X4)`. Van de aantallen van één cluster wil je het grootste.",
            "`km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(___)` en `right = right + counts.___()`.",
        ],
        why="`km.labels_ == j` kiest de pinguïns van cluster `j`, `np.unique` telt hun soorten, en het grootste aantal zijn de pinguïns van de soort die het vaakst in dat cluster zit. Opgeteld over de drie clusters is dat hoe goed de clusters bij de soorten passen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Oefening 4. Waarom minder met meer metingen?
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-four",
        question="Met snavellengte en flipperlengte zitten 327 pinguïns in het cluster van hun eigen soort; met alle vier de metingen 313, en 24 Adelies zitten in het cluster van de Chinstraps. Gestandaardiseerd verschillen Adelies en Chinstraps bijna 2 in snavellengte, en minder dan 0,5 in elk van de andere drie metingen. Waarom passen vier metingen hier minder goed bij de soorten?",
        options={
            "always": "Meer metingen maken clusters altijd slechter",
            "right": "De snavellengte is nu één van de vier metingen in de afstand, en de andere drie houden Adelies en Chinstraps nauwelijks uit elkaar",
            "two": "k-means kan maar twee metingen tegelijk gebruiken",
        },
        correct="right",
        why={
            "always": "Niet altijd: een meting waarin de groepen verschillen, helpt. Hier verschillen snavelhoogte en lichaamsgewicht nauwelijks tussen Adelies en Chinstraps, dus voegen ze afstand toe die niets met de soort te maken heeft.",
            "two": "k-means gebruikt elke feature die je het geeft: de afstand gaat over alle vier. Vraag je af in welke ervan Adelies en Chinstraps verschillen.",
            "right": "k-means kent alleen afstanden. Een feature waarin twee soorten nauwelijks verschillen, telt toch mee in de afstand, en dan wegen verschillen binnen een soort zwaarder: van de 24 Adelies in het cluster van de Chinstraps zijn er 23 mannetjes: zwaarder, met hogere en langere snavels. Welke features je kiest, bepaalt welke groepen k-means kan vinden.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** k-means heeft geen labels, zoals in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl), en meet nabijheid met de gestandaardiseerde afstand uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl); een centroid is het gemiddelde langs een as dat je in NumPy nam. Hier koos je k = 3 omdat je wist dat er drie soorten zijn. In de volgende les, *Hoeveel clusters? k kiezen*, kies je k als je dat niet weet, en zie je waarom je k-means vanaf meerdere starts uitvoert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why, result_four, result_stable, result_step):
    L.wrap_up(result_step, result_stable, result_four, choice_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
