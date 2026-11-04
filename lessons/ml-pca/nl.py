# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-pca. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.
# termen:allow hoofdcomponentenanalyse

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-pca", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) en [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Vier schaduwen zijn −2, −1, 1 en 2, allemaal even vaak, dus hun gemiddelde is 0. Wat is hun variantie?",
        options={
            "zero": "0",
            "right": "2,5",
            "abs": "1,5",
        },
        correct="right",
        why={
            "zero": "0 is hun gemiddelde: de schaduwen onder 0 heffen die erboven op. De variantie is de gemiddelde gekwadrateerde afstand tot het gemiddelde, en kwadraten heffen elkaar niet op. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "abs": "1,5 is de gemiddelde afstand tot het gemiddelde, zonder kwadrateren. De variantie neemt het gemiddelde van de gekwadrateerde afstanden. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "(4 + 1 + 1 + 4) / 4 = 2,5. Met het gemiddelde op 0 is de variantie gewoon het gemiddelde van de kwadraten, zoals straks bij de schaduwen in deze les.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Waarom kan de kleinste within-cluster sum of squares k niet kiezen?",
        options={
            "labels": "Omdat hij labels nodig heeft",
            "right": "Omdat de beste som alleen maar kleiner wordt als k groeit",
            "negative": "Omdat hij onder 0 kan komen",
        },
        correct="right",
        why={
            "labels": "Hij heeft geen labels nodig: alleen de voorbeelden en hun centroids. Zie [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).",
            "negative": "Een som van kwadraten is nooit kleiner dan 0. Zie [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).",
            "right": "De kleinste is dus altijd de grootste k die je probeert, en daarom zoek je de elleboog of de silhouette score.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een pinguïn met vier metingen is vier getallen, en vier getallen passen niet in één plaatje. Een foto van 28 bij 28 pixels is 784 getallen. Om naar zulke data te kijken, of om een model minder getallen te geven, wil je minder getallen die zo veel mogelijk van de data bewaren. Deze les zoekt ze voor de pinguïns, zonder hun soort, zoals in de vorige twee lessen.

    **Een voorbeeld uit het dagelijks leven.** Een lamp werpt de schaduw van een potlood op de muur: drie dimensies worden er twee. Houd je het potlood met zijn punt naar de lamp, dan is de schaduw een stip; houd je het dwars, dan laat de schaduw de hele lengte zien. De ene schaduw laat meer zien dan de andere, en je kunt het potlood draaien tot zijn schaduw het meeste laat zien.
    """)
    return


@app.cell(hide_code=True)
def predict_kept():
    guess_kept = L.guess(
        "kept",
        question="De 342 pinguïns op snavellengte en flipperlengte, in millimeters. Je trekt één lijn door hun midden en vervangt elke pinguïn door zijn schaduw op die lijn: één getal in plaats van twee. Hoeveel van de spreiding van de pinguïns kan de beste lijn bewaren?",
        options={
            "half": "Ongeveer de helft",
            "threequarters": "Ongeveer driekwart",
            "right": "Meer dan 90%",
        },
        reveal="**93,0%.** De beste lijn loopt vooral langs de flipperlengte: een lijn alleen langs de snavellengte bewaart 13,1%, een lijn alleen langs de flipperlengte 86,9%. Flipperlengtes lopen meer millimeters uiteen, en pinguïns met lange flippers hebben meestal ook lange snavels, dus één getal bevat het meeste van wat twee zeggen.",
    )
    guess_kept
    return (guess_kept,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Dimensiereductie** beschrijft elk voorbeeld met minder getallen, en bewaart daarbij zo veel als het kan. De eenvoudigste manier is een schaduw. Trek eerst van elke feature het gemiddelde af, zodat de voorbeelden rond 0 liggen. Neem dan een richting $\mathbf{w}$ met lengte 1. De schaduw van een voorbeeld $\mathbf{x}$ op de lijn door 0 in die richting is het inproduct $\mathbf{x} \cdot \mathbf{w}$: hoe ver het voorbeeld langs de lijn ligt. Schaduw is beeldspraak: in de wiskunde is dit getal de projectie van $\mathbf{x}$ op $\mathbf{w}$, en het wijst het punt van de lijn aan dat het dichtst bij $\mathbf{x}$ ligt. De schaduwen liggen ook rond 0, dus hun spreiding, hun variantie, is het gemiddelde van hun kwadraten.

        Van de schaduw naar het voorbeeld is een rechte hoek, dus volgens Pythagoras is de gekwadrateerde lengte van elk voorbeeld zijn gekwadrateerde schaduw plus zijn gekwadrateerde afstand tot de lijn:

        $$\mathbf{x} \cdot \mathbf{x} = (\mathbf{x} \cdot \mathbf{w})^2 + d^2$$

        Neem het gemiddelde over alle voorbeelden: de spreiding van alle voorbeelden is de spreiding die de schaduwen bewaren plus de gemiddelde gekwadrateerde afstand tot de lijn. Wat de schaduwen bewaren, verliest de lijn niet. De richting waarvan de schaduwen het meeste bewaren, is dus ook de lijn die het dichtst bij de voorbeelden ligt. Anders dan bij lineaire regressie meet je de afstand loodrecht op de lijn, niet recht omhoog: geen enkele feature is degene die je voorspelt.

        **Principal component analysis**, PCA, in de Nederlandse statistiek ook hoofdcomponentenanalyse, vindt die richting: de eerste **principal component**. De tweede is de richting loodrecht op de eerste waarvan de schaduwen het meeste bewaren van wat er over is, en zo verder, zoveel als er features zijn. De **verklaarde variantie** van een component is het deel van de spreiding dat zijn schaduwen bewaren; over alle componenten samen is dat 1. Houd je alleen de eerste paar, dan beschrijf je elk voorbeeld met zijn schaduwen daarop.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **dimensiereductie** | elk voorbeeld met minder getallen beschrijven, en daarbij zo veel mogelijk bewaren | |
        | schaduw | voor een voorbeeld $\mathbf{x}$ min het gemiddelde, en een richting $\mathbf{w}$ met lengte 1: $\mathbf{x} \cdot \mathbf{w}$ | `M @ w` |
        | **principal component analysis**, PCA | de richtingen zoeken waarvan de schaduwen de meeste spreiding bewaren, de een na de ander, elk loodrecht op de eerdere | `PCA(n_components=2).fit(X)` |
        | **principal component** | een van die richtingen, een vector met lengte 1 | `pca.components_[0]` |
        | **verklaarde variantie** | het deel van de spreiding dat de schaduwen op een component bewaren | `pca.explained_variance_ratio_` |
        | `pca.transform(X)` | de schaduwen van elk voorbeeld op de componenten, een kolom voor elke component | |
        | `pca.explained_variance_` | zonder `_ratio`: de spreiding van de schaduwen op elke component zelf, niet het deel | |
        """),
        L.watch_out("PCA ziet nooit een label: het bewaart de meeste spreiding, en de meeste spreiding is niet altijd wat groepen uit elkaar houdt. En het teken van een component betekent niets: (0,267, 0,964) en (−0,267, −0,964) zijn dezelfde lijn, en de schaduwen veranderen alleen van teken."),
        mo.md(r"""
        Het plaatje toont de 342 pinguïns op snavellengte en flipperlengte, in millimeters vanaf hun gemiddelde, met beide assen op dezelfde schaal. De oranje lijn gaat door het gemiddelde onder de hoek van de slider: 0° loopt langs de snavellengte, 90° langs de flipperlengte. De schaduw van elke pinguïn op de lijn is blauw, en de dunne rode lijntjes zijn de afstanden tot de lijn. Rechts voor elke hoek het gemiddelde van de gekwadrateerde rode lijntjes, in paars. Schuif met de slider, of sleep over het rechterplaatje.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_angle_cell():
    def _frame(angle):
        angle = int(angle)
        f = C.shadow(angle)
        said = f"""Bij {angle}°: de schaduwen bewaren {L.number(100 * f["share"], 1)}% van de spreiding, {L.number(f["kept"], 1)} van de {L.number(f["total"], 1)} mm². De gemiddelde gekwadrateerde afstand tot de lijn is de rest, {L.number(f["lost"], 1)} mm²."""
        alt = (f"Het eerste plaatje: de 342 pinguïns op snavellengte en flipperlengte, in millimeters vanaf hun gemiddelde, met een lijn door het gemiddelde onder {angle} graden, de schaduw van elke pinguïn erop, en de afstand van elke pinguïn tot de lijn. "
               + "Het tweede plaatje: de gemiddelde gekwadrateerde afstand tot de lijn voor elke hoek van 0 tot 175 graden; die is het kleinst bij 75 graden en het grootst bij 165. "
               + said)
        return (C.picture(angle, {"title": f"De lijn bij {angle}°", "every": "Elke hoek", "angle": "hoek (°)", "lost": "gemiddelde gekwadrateerde afstand (mm²)",
                                  "x": "snavellengte, vanaf het gemiddelde (mm)", "y": "flipperlengte, vanaf het gemiddelde (mm)", "alt": alt}), said)

    explore_angle = L.explore("angle", "hoek", _frame, start=0, values=C.ANGLES, role="param",
                              task="Draai de lijn. Bij welke hoek zijn de rode lijntjes het kortst, en hoeveel van de spreiding bewaren de schaduwen daar?")
    explore_angle
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Langs de snavellengte, bij 0°, bewaren de schaduwen maar 13,1%: snavellengtes verschillen maar ongeveer 5 mm. Langs de flipperlengte, bij 90°, bewaren ze 86,9%. De beste hoek is 75°: de schaduwen bewaren 93,0%, en de rode lijntjes zijn het kortst, met een gemiddeld kwadraat van 15,8 mm². Loodrecht daarop, bij 165°, bewaren de schaduwen maar 7,0%: dat is de richting van de tweede principal component.

    ### Recept

    Om voorbeelden met minder getallen te beschrijven met PCA:

    1. Kies de features, en standaardiseer ze als hun eenheden verschillen.
    2. Trek van elke feature het gemiddelde af, zodat de voorbeelden rond 0 liggen.
    3. Voor een richting $\mathbf{w}$ met lengte 1 zijn de schaduwen de inproducten `M @ w`; hun spreiding is het gemiddelde van hun kwadraten.
    4. De eerste principal component is de richting waarvan de schaduwen de meeste spreiding bewaren; elke volgende bewaart het meeste van wat er over is, loodrecht op de eerdere. `PCA` vindt ze.
    5. De verklaarde variantie van een component is de spreiding van zijn schaduwen gedeeld door de spreiding van alles. Houd zoveel componenten als je nodig hebt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het recept op snavellengte en flipperlengte. `_total` is de spreiding van alle pinguïns: het gemiddelde van hun gekwadrateerde lengtes, `np.sum(_M ** 2, axis=1)` voor elke pinguïn. Dan drie richtingen met lengte 1, gemaakt met de norm: langs de snavellengte, langs de flipperlengte, en precies ertussenin. Voor elke richting de richting zelf, de spreiding van zijn schaduwen in mm², en het deel van `_total`. Aan het eind hetzelfde voor de eerste principal component van scikit-learn, en zijn `explained_variance_ratio_`. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_pca():
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["flipper_length_mm"].isna() == False]
    # 1. beide features zijn in millimeters, dus niet standaardiseren
    _raw = np.array(_df[["bill_length_mm", "flipper_length_mm"]])
    # 2. trek het gemiddelde af
    _M = _raw - _raw.mean(axis=0)
    _total = np.mean(np.sum(_M ** 2, axis=1))
    print(np.round(_total, 1))
    # 3. drie richtingen met lengte 1, en de spreiding van hun schaduwen
    for _d in [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]:
        _w = _d / np.linalg.norm(_d)
        _shadow = _M @ _w
        print(np.round(_w, 3), np.round(np.mean(_shadow ** 2), 1), np.round(np.mean(_shadow ** 2) / _total, 3))
    # 4. de eerste principal component, uit scikit-learn
    _pca = PCA(n_components=2).fit(_M)
    _w = _pca.components_[0]
    _shadow = _M @ _w
    # 5. het deel van de spreiding dat zijn schaduwen bewaren
    print(np.round(_w, 3), np.round(np.mean(_shadow ** 2), 1), np.round(np.mean(_shadow ** 2) / _total, 3))
    print(np.round(_pca.explained_variance_ratio_, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De spreiding van alle pinguïns is 226,9 mm². Langs de snavellengte bewaren de schaduwen daarvan 29,7, langs de flipperlengte 197,2: dat zijn de varianties van de twee features, en 29,7 + 197,2 = 226,9, dus de spreiding van alles is de variantie van elke feature bij elkaar opgeteld. Precies ertussenin, (0,707, 0,707), bewaren de schaduwen 163,7. De eerste principal component is (0,267, 0,964), bij de 75° van het plaatje, en zijn schaduwen bewaren 211,1 mm²: 0,93 van de spreiding, hetzelfde als `explained_variance_ratio_[0]`. De tweede bewaart de andere 0,07.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het deel van de eerste component

        Stap 3 tot en met 5 van het uitgewerkte voorbeeld, nu op snavellengte en snavelhoogte, allebei in millimeters. Het vak trekt het gemiddelde af in `M`, rekent `total` uit en fit `PCA`; `w` is de eerste principal component. Vervang de twee `...`: de schaduw van elke pinguïn op `w`, en het deel van de spreiding dat die schaduwen bewaren. De laatste regel zet jouw deel naast `pca.explained_variance_ratio_[0]`.
        """),
        L.task("Reken het deel van de spreiding uit dat de eerste principal component bewaart."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("first-share", lines=15)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("first-share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "first-share",
        [
            "Stap 3 en 5 van het uitgewerkte voorbeeld, met `M`, `w` en `total` in plaats van `_M`, `_w` en `_total`.",
            "De schaduw van elke pinguïn is het inproduct met `w`, voor alle pinguïns tegelijk met `@`. Hun spreiding is het gemiddelde van hun kwadraten, en het deel is dat gedeeld door `total`.",
            "`shadow = M @ ___` en `share = np.mean(shadow ** ___) / total`.",
        ],
        why="`M @ w` geeft één getal voor elke pinguïn: hoe ver hij langs `w` ligt. De schaduwen liggen rond 0, omdat `M` dat doet, dus het gemiddelde van hun kwadraten is hun spreiding; gedeeld door de spreiding van alles is dat het deel dat de lijn bewaart.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_many():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel componenten?

        Nu alle vier de metingen, gestandaardiseerd in `X4`, omdat je grammen en millimeters niet kunt vergelijken. Het vak fit `PCA` met 4 componenten. De loop telt de verklaarde variantie van de componenten een voor een op in `kept`, en stopt zodra hij minstens 0,9 heeft. Vervang de twee `...`: wat er voor component `n` bij komt, en wanneer de loop stopt. De laatste regel geeft hoeveel componenten dat kostte, en wat ze bewaren.
        """),
        L.task("Zoek uit hoeveel principal components minstens 90% van de spreiding bewaren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_many():
    box_many = L.code_box("how-many", lines=16)
    box_many
    return (box_many,)


@app.cell(hide_code=True)
def check_many(box_many):
    result_many = L.check("how-many", box_many.value)
    result_many
    return (result_many,)


@app.cell(hide_code=True)
def hints_many():
    L.hints(
        "how-many",
        [
            "`pca.explained_variance_ratio_` heeft een deel voor elk van de 4 componenten. Welk hoort bij `n`, en wanneer heb je genoeg gehouden?",
            "Tel `pca.explained_variance_ratio_[n]` op bij `kept`, en stop met `break` zodra `kept` 0,9 of meer is.",
            "`kept = kept + pca.explained_variance_ratio_[___]` en `if kept >= ___:`.",
        ],
        why="De componenten komen op volgorde: de eerste bewaart het meest, elke volgende minder. Door ze een voor een op te tellen, stopt de loop bij het kleinste aantal componenten dat 90% bewaart; `n` telt vanaf 0, dus dat zijn `n + 1` componenten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shadows():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. k-means op twee schaduwen

        Het vak fit `PCA` met 2 componenten op `X4`. Vervang de twee `...`: de schaduwen van elke pinguïn op de twee componenten, met `pca.transform`, en waarop `KMeans` wordt gefit. De loop telt dan, zoals in de les over k-means, de pinguïns van de soort die het vaakst in elk cluster zit.
        """),
        L.task("Cluster de pinguïns op hun schaduwen op twee componenten, en tel hoeveel er in het cluster van hun eigen soort zitten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shadows():
    box_shadows = L.code_box("two-shadows", lines=19)
    box_shadows
    return (box_shadows,)


@app.cell(hide_code=True)
def check_shadows(box_shadows):
    result_shadows = L.check("two-shadows", box_shadows.value)
    result_shadows
    return (result_shadows,)


@app.cell(hide_code=True)
def hints_shadows():
    L.hints(
        "two-shadows",
        [
            "De nieuwe woorden van *Het idee* zeggen wat de schaduwen van elk voorbeeld op de componenten geeft.",
            "`Z` is `pca.transform(X4)`: twee getallen voor elke pinguïn. k-means clustert dan `Z` in plaats van `X4`.",
            "`Z = pca.transform(___)` en `km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(___)`.",
        ],
        why="`pca.transform(X4)` trekt het gemiddelde af en neemt het inproduct van elke pinguïn met elke component, dus `Z` beschrijft elke pinguïn met twee getallen in plaats van vier. k-means ziet alleen die twee, en vindt de groepen die zij bewaren.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scale():
    mo.md(r"""
    ### Oefening 4. Waarom eerst standaardiseren?
    """)
    return


@app.cell(hide_code=True)
def choice_scale_cell():
    choice_scale = L.choice(
        "why-scale",
        question="Zonder standaardiseren vindt PCA op de vier metingen een eerste component van ongeveer (0,004, −0,001, 0,015, 1,000): snavellengte, snavelhoogte, flipperlengte en lichaamsgewicht. Zijn verklaarde variantie is 0,9999. Waarom?",
        options={
            "species": "Het lichaamsgewicht houdt de soorten het best uit elkaar",
            "right": "Het lichaamsgewicht is in grammen: het varieert zo'n 800 g, de flipperlengte zo'n 14 mm, dus in kwadraten maakt het lichaamsgewicht bijna alle spreiding",
            "last": "PCA legt het meeste gewicht op de laatste feature",
        },
        correct="right",
        why={
            "species": "PCA ziet de soorten nooit: het kijkt alleen naar de spreiding. En het lichaamsgewicht alleen houdt Adelies en Chinstraps niet uit elkaar.",
            "last": "De volgorde van de features maakt voor PCA niet uit; hoeveel elke feature varieert wel. Kijk naar hun eenheden.",
            "right": "De spreiding telt in kwadraten: ongeveer 641.251 g² voor het lichaamsgewicht tegen 197 mm² voor de flipperlengte. Elke richting weg van het lichaamsgewicht verliest veel meer dan hij wint, dus is de eerste component het lichaamsgewicht. Gestandaardiseerd heeft elke feature een spreiding van 1, en vindt PCA wat de vier gemeen hebben.",
        },
    )
    choice_scale
    return (choice_scale,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De schaduw is het inproduct uit [Veel features: het inproduct](/course/open/math-dot-product?lang=nl), voor alle pinguïns tegelijk met de `@` uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl), en de spreiding is de variantie uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl). Net als [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl) ziet PCA nooit een label, dus bewaart het wat het meest varieert, niet wat jij zoekt: de derde component van de vier metingen bewaart maar 9% van de spreiding, en toch verschillen Adelies en Chinstraps daarop ongeveer even veel als op elk van de eerste twee. Hiermee is de module *Zonder labels* klaar. In de volgende les, *Een neuron*, beginnen de neural networks.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scale, result_many, result_shadows, result_share):
    L.wrap_up(result_share, result_many, result_shadows, choice_scale)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
