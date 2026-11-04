# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-choosing-k. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-choosing-k", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="Wanneer is k-means klaar?",
        options={
            "k": "Na k stappen, één voor elk cluster",
            "right": "Als geen enkel voorbeeld meer van cluster verandert",
            "equal": "Als elk cluster evenveel voorbeelden heeft als de andere",
        },
        correct="right",
        why={
            "k": "Het aantal stappen heeft niets met k te maken: op de pinguïns, met k = 3, waren het er 7. Zie het plaatje in [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "equal": "Clusters kunnen verschillend groot zijn: op de pinguïns hadden ze er 127, 151 en 64. Zie [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "right": "Dan blijven de centroids, de gemiddelden van dezelfde voorbeelden, ook waar ze zijn, en zou elke volgende stap hetzelfde zijn.",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def quick_check_numbers():
    quick_numbers = L.choice(
        "quick-numbers",
        question="Je draait k-means twee keer en krijgt dezelfde drie groepen, maar de Adelies zijn de eerste keer cluster 0 en de tweede keer cluster 2. Wat zegt dat?",
        options={
            "wrong": "De tweede keer is fout",
            "right": "Niets: de nummers van de clusters betekenen niets",
            "moved": "Een paar Adelies zijn van cluster veranderd",
        },
        correct="right",
        why={
            "wrong": "Allebei vonden ze dezelfde groepen; alleen hun namen verschillen. Zie het uitgewerkte voorbeeld in [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "moved": "De groepen zijn hetzelfde, dus geen pinguïn veranderde van groep; alleen het nummer van de groep veranderde. Zie het uitgewerkte voorbeeld in [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "right": "k-means zegt alleen welke voorbeelden bij elkaar horen. Welk cluster de Adelies zijn, ontdek je door te kijken.",
        },
    )
    quick_numbers
    return (quick_numbers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les koos je k = 3, omdat je wist dat er drie soorten pinguïns zijn. Zonder labels weet je dat niet: een winkel weet niet hoeveel soorten klanten hij heeft, en dat is precies wat hij wil uitzoeken. Je hebt dus een manier nodig om k uit de data te kiezen. En er is nog een gat: k-means eindigt altijd, maar waar het eindigt, hangt af van waar het begint.

    **Een voorbeeld uit het dagelijks leven.** Je sorteert je vakantiefoto's in albums. Eén album is een rommeltje; een album voor elke foto heb je ook niets aan. Ergens daartussen maakt een nieuw album de andere niet veel netter meer: ongeveer zoveel wil je er. En een foto zit in een goed album als hij meer lijkt op de foto's in zijn eigen album dan op die in het album ernaast.
    """)
    return


@app.cell(hide_code=True)
def predict_smallest():
    guess_smallest = L.guess(
        "smallest-sum",
        question="k-means maakt de gekwadrateerde afstanden van elke pinguïn tot de centroid van zijn cluster klein. Je draait het op de pinguïns voor elke k van 1 tot en met 8 en telt die gekwadrateerde afstanden op. Voor welke k is de som volgens jou het kleinst?",
        options={
            "three": "k = 3, het aantal soorten",
            "two": "k = 2",
            "right": "k = 8, de grootste k die je probeert",
        },
        reveal="**k = 8.** De som is 684,0 met één cluster, 247,0 met twee, 157,8 met drie en 57,8 met acht: met elke centroid erbij liggen de pinguïns dichter bij een centroid. Met 342 clusters, één voor elke pinguïn, zou hij 0 zijn. De kleinste som alleen kan k dus niet kiezen.",
    )
    guess_smallest
    return (guess_smallest,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **within-cluster sum of squares** telt voor elk voorbeeld de gekwadrateerde afstand tot de centroid van zijn cluster op. Die maakt k-means klein: toewijzen geeft elk voorbeeld zijn dichtstbijzijnde centroid, en verschuiven zet elke centroid op het gemiddelde van zijn voorbeelden, het punt met de kleinste som van gekwadrateerde afstanden tot hen. scikit-learn noemt hem `inertia_`. De kleinste som die er is, wordt alleen maar kleiner als k groeit, dus je zoekt niet de kleinste, maar de **elleboog**: de k waarna een cluster erbij hem nog maar een beetje kleiner maakt. Zo k kiezen heet de elleboogmethode.

        Het **silhouette** van één voorbeeld vraagt hoe duidelijk het in zijn eigen cluster ligt. Neem $a$, de gemiddelde afstand tot de andere voorbeelden in zijn eigen cluster, en $b$, de gemiddelde afstand tot de voorbeelden van het dichtstbijzijnde andere cluster. Dan ligt

        $$s = \frac{b - a}{\max(a, b)}$$

        tussen −1 en 1: dicht bij 1 als het voorbeeld veel dichter bij zijn eigen cluster ligt dan bij elk ander, dicht bij 0 als het op de grens ligt, en onder 0 als een ander cluster dichterbij ligt. De **silhouette score** is het gemiddelde silhouette van alle voorbeelden. Hij heeft minstens twee clusters nodig, en hoger is beter.

        Een voorbeeld: de Adelie in rij 94 van de tabel heeft een snavel van 40,8 mm en lange flippers van 208 mm. In de drie clusters van k = 3, gestandaardiseerd, ligt hij gemiddeld 1,446 van de andere pinguïns van zijn cluster, de Adelies, en 1,433 van het dichtstbijzijnde andere cluster, de Gentoos. Dus $a$ = 1,446, $b$ = 1,433, en zijn silhouette is (1,433 − 1,446) / 1,446 = −0,009: net onder 0. k-means zette hem bij de Adelies, maar gemiddeld liggen de Gentoos een beetje dichterbij; hij is de enige van de 342 onder 0. De Adelie in rij 0, met een snavel van 39,1 mm en flippers van 181 mm, heeft $a$ = 0,809 en $b$ = 2,135, tot de Chinstraps: zijn silhouette is (2,135 − 0,809) / 2,135 = 0,621, duidelijk een Adelie.

        Nog één ding: k-means stopt als geen enkel voorbeeld meer van cluster verandert, maar dat is niet altijd de kleinste som die er is. Vanaf een slechte start kan het eindigen in een **lokaal minimum**: een oplossing die geen van beide stappen nog beter maakt, terwijl een andere een kleinere som heeft. Daarom draait `KMeans` vanaf `n_init` starts en houdt het de start met de kleinste som.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **within-cluster sum of squares** | voor elk voorbeeld de gekwadrateerde afstand tot de centroid van zijn cluster, opgeteld | `km.inertia_` |
        | **elleboog** | de k waarna een cluster erbij de within-cluster sum of squares nog maar een beetje kleiner maakt | |
        | **silhouette** | voor één voorbeeld $(b - a) / \max(a, b)$: zijn eigen cluster tegen het dichtstbijzijnde andere | `silhouette_samples(X, labels)` |
        | $a$, $b$ | de gemiddelde afstand tot de andere voorbeelden van zijn eigen cluster, en tot die van het dichtstbijzijnde andere cluster | |
        | **silhouette score** | het gemiddelde silhouette van alle voorbeelden, van −1 tot 1 | `silhouette_score(X, labels)` |
        | **lokaal minimum** | een oplossing die geen stap beter maakt, terwijl er een betere is | |
        | `n_init=10` | k-means vanaf 10 starts draaien en de start met de kleinste within-cluster sum of squares houden | |
        """),
        L.watch_out("De silhouette score zegt hoe duidelijk de clusters uit elkaar liggen, niet of het de groepen zijn die je zoekt. Twee clusters kunnen verder uit elkaar liggen dan de drie soorten."),
        mo.md(r"""
        Het plaatje toont `KMeans` met `n_init=10` op de 342 pinguïns, op snavellengte en flipperlengte, gestandaardiseerd, voor elke k van 1 tot en met 8. Links elke pinguïn, met een dun lijntje verbonden met de centroid van zijn cluster; de centroids zijn oranje. De within-cluster sum of squares is de som van de gekwadrateerde lengtes van al die lijntjes. Rechts die som voor elke k, in paars, met deze k ingekleurd. Schuif met de slider, of sleep over het rechterplaatje.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        k = int(k)
        f = C.fit(k)
        if k == 1:
            said = f"""k = 1: één centroid, op het gemiddelde van alle pinguïns. De within-cluster sum of squares is {L.number(f["wcss"], 1)}: 342 voor elk van de twee gestandaardiseerde features. Er is geen silhouette score, omdat er geen ander cluster is."""
        else:
            said = f"""k = {k}: de within-cluster sum of squares is {L.number(f["wcss"], 1)}, {L.number(C.fit(k - 1)["wcss"] - f["wcss"], 1)} minder dan met één cluster minder. De silhouette score is {L.number(f["silhouette"], 3)}."""
        alt = (f"Het eerste plaatje: de 342 pinguïns op snavellengte en flipperlengte, gestandaardiseerd, elk met een lijntje verbonden met de dichtstbijzijnde van de {k} centroids van k-means. "
               + "Het tweede plaatje: de within-cluster sum of squares voor k van 1 tot en met 8; hij daalt van 684,0 naar 247,0, 157,8, 118,8, 91,5, 78,5, 66,7 en 57,8. "
               + said)
        return (C.picture(k, {"title": f"{k} clusters", "every": "Elke k", "k": "k", "wcss": "within-cluster sum of squares",
                              "x": "snavellengte, gestandaardiseerd", "y": "flipperlengte, gestandaardiseerd", "alt": alt}), said)

    explore_k = L.explore("k", "k", _frame, start=1, values=C.KS, role="hyper",
                          task="Ga van k = 1 naar 8. Na welke k daalt de som niet snel meer, en welke k heeft de hoogste silhouette score?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    De som daalt met 437,0 van één cluster naar twee, met 89,2 naar drie, en daarna met 39,0 en minder. De elleboog is hier niet scherp: je kunt hem bij 2 of bij 3 zien. De silhouette score is het hoogst bij k = 2, met 0,543, dan 0,522 bij k = 3 en 0,507 bij k = 4, en vanaf k = 5 is hij 0,435 of lager.

    ### Recept

    Om k te kiezen voor k-means:

    1. Standaardiseer de features.
    2. Draai `KMeans` voor elke k die je overweegt, vanaf meerdere starts met `n_init`.
    3. Noteer de within-cluster sum of squares van elke k, en zoek de elleboog.
    4. Reken de silhouette score van elke k vanaf 2 uit: hoger betekent clusters die duidelijker uit elkaar liggen.
    5. Bekijk de clusters van de beste paar k, en kies met waar je ze voor wilt gebruiken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het recept op de pinguïns, op snavellengte en flipperlengte. Voor elke k van 1 tot en met 8 print de cel k, de within-cluster sum of squares met de hand, dezelfde uit scikit-learn, en vanaf k = 2 de silhouette score. `_km.cluster_centers_[_km.labels_]` geeft, in een rij voor elke pinguïn, de centroid van zijn cluster, dus `_X` min dat is het verschil van elke pinguïn met zijn eigen centroid. Aan het eind, voor stap 5, de soorten in elk van de twee clusters van k = 2, zoals in oefening 3 van de vorige les. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_choose_k():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["flipper_length_mm"].isna() == False]
    _raw = np.array(_df[["bill_length_mm", "flipper_length_mm"]])
    # 1. standaardiseer de features
    _X = (_raw - _raw.mean(axis=0)) / _raw.std(axis=0)
    for _k in range(1, 9):
        # 2. KMeans voor deze k, vanaf 10 starts
        _km = KMeans(n_clusters=_k, n_init=10, random_state=0).fit(_X)
        # 3. de within-cluster sum of squares, met de hand en uit scikit-learn
        _wcss = np.sum((_X - _km.cluster_centers_[_km.labels_]) ** 2)
        if _k == 1:
            print(_k, np.round(_wcss, 1), np.round(_km.inertia_, 1))
        else:
            # 4. de silhouette score, vanaf k = 2
            print(_k, np.round(_wcss, 1), np.round(_km.inertia_, 1), np.round(silhouette_score(_X, _km.labels_), 3))
    # 5. bekijk de clusters van k = 2: de soorten in elk
    _km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_X)
    _species = np.array(_df["species"])
    for _j in range(2):
        _names, _counts = np.unique(_species[_km.labels_ == _j], return_counts=True)
        print(_j, _names, _counts)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De som met de hand is voor elke k hetzelfde als `inertia_`. Stap 5: de elleboog bij 2 of 3 en de hoogste silhouette score bij 2 laten k = 2 en k = 3 over. De twee clusters van k = 2 zijn vooral de korte snavels en flippers tegen de lange: 149 Adelies en 23 Chinstraps, tegen 2 Adelies, 45 Chinstraps en 123 Gentoos. De drie van k = 3 liggen dicht bij de soorten, zoals in de vorige les. Welke van de twee goed is, vraagt oefening 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_wcss():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De within-cluster sum of squares met de hand

        Stap 3 van het uitgewerkte voorbeeld, nu op alle vier de metingen: snavellengte, snavelhoogte, flipperlengte en lichaamsgewicht, gestandaardiseerd in `X4`. Het vak clustert ze met `KMeans` in 3 clusters. Vervang de `...`: de within-cluster sum of squares, in NumPy. De laatste regel zet jouw som naast `km.inertia_`.
        """),
        L.task("Reken de within-cluster sum of squares van de drie clusters op vier metingen uit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wcss():
    box_wcss = L.code_box("wcss", lines=12)
    box_wcss
    return (box_wcss,)


@app.cell(hide_code=True)
def check_wcss(box_wcss):
    result_wcss = L.check("wcss", box_wcss.value)
    result_wcss
    return (result_wcss,)


@app.cell(hide_code=True)
def hints_wcss():
    L.hints(
        "wcss",
        [
            "Stap 3 van het uitgewerkte voorbeeld, met `X4` in plaats van `_X` en `km` in plaats van `_km`.",
            "`X4 - km.cluster_centers_[km.labels_]` geeft elke pinguïn zijn verschil met zijn eigen centroid, in alle vier de features. Kwadrateer dat met `** 2`, en tel alles op met `np.sum`.",
            "`wcss = np.sum((X4 - km.cluster_centers_[km.labels_]) ** ___)`.",
        ],
        why="De gekwadrateerde afstand van een pinguïn tot zijn centroid is de som van zijn gekwadrateerde verschillen in alle vier de features, dus als je alle gekwadrateerde verschillen optelt, van elke pinguïn en elke feature, krijg je in één keer de som van al die gekwadrateerde afstanden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_starts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tien starts

        `by_hand` is k-means met de hand, zoals in de vorige les: vanaf de rijen `start` als startcentroids, tot geen enkele pinguïn meer van cluster verandert, en dan geeft het de within-cluster sum of squares terug. Vervang de twee `...`: drie willekeurige rijen voor deze `seed`, en de uitkomst van `by_hand` vanaf die rijen. `np.random.default_rng(seed).permutation(len(X))` zet de rijnummers 0 tot en met 341 in een willekeurige volgorde, voor elke `seed` een andere. De laatste regel geeft de kleinste en de grootste van de tien uitkomsten.
        """),
        L.task("Draai k-means met de hand vanaf tien starts, en vergelijk waar ze eindigen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_starts():
    box_starts = L.code_box("starts", lines=31)
    box_starts
    return (box_starts,)


@app.cell(hide_code=True)
def check_starts(box_starts):
    result_starts = L.check("starts", box_starts.value)
    result_starts
    return (result_starts,)


@app.cell(hide_code=True)
def hints_starts():
    L.hints(
        "starts",
        [
            "Drie willekeurige rijen zijn de eerste drie rijnummers van een willekeurige volgorde. En `by_hand` heeft de pinguïns en de startrijen nodig.",
            "Neem de eerste drie van `np.random.default_rng(seed).permutation(len(X))` met een slice, en voeg `by_hand(X, start)` toe aan `results`.",
            "`start = np.random.default_rng(seed).permutation(len(X))[:___]` en `results.append(by_hand(___, start))`.",
        ],
        why="Elke seed geeft andere startrijen, en k-means eindigt waar zijn twee stappen het vanaf daar brengen. Als twee uitkomsten verschillen, is minstens één ervan een lokaal minimum: geen van beide stappen maakt het beter, en toch vond een andere start een kleinere som.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. k kiezen met de silhouette score

        De loop draait `KMeans` voor elke k van 2 tot en met 8, en houdt in `best` de k met de hoogste score tot nu toe bij, en in `top` die score. Vervang de `...`: de silhouette score van de clusters van deze `km`. De laatste regel geeft de beste k en zijn score.
        """),
        L.task("Kies k met de silhouette score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-k", lines=19)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-k", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-k",
        [
            "De nieuwe woorden van *Het idee* zeggen welke functie de silhouette score geeft, en wat die nodig heeft.",
            "`silhouette_score` heeft de pinguïns nodig en het cluster van elke pinguïn, dat in `km.labels_` staat.",
            "`score = silhouette_score(___, km.labels_)`.",
        ],
        why="De silhouette score kijkt naar de clusters, niet naar de centroids of de som: voor elke pinguïn vergelijkt hij zijn eigen cluster met het dichtstbijzijnde andere. Een hogere score betekent clusters die duidelijker uit elkaar liggen, dus houdt de loop de hoogste.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 4. Welke k?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-k",
        question="Op de pinguïns is de silhouette score het hoogst voor k = 2, ligt de elleboog bij 2 of 3, en zijn er drie soorten. De soorten zelf, als clusters, hebben een silhouette score van 0,483: lager dan de 0,522 van de drie clusters van k-means. Welke k is goed?",
        options={
            "two": "k = 2, omdat zijn silhouette score het hoogst is",
            "three": "k = 3, omdat er drie soorten zijn",
            "right": "Dat hangt af van waar je de clusters voor wilt: geen getal beslist het in zijn eentje",
        },
        correct="right",
        why={
            "two": "De twee clusters van k = 2 liggen het duidelijkst uit elkaar, maar ze zetten de Chinstraps deels bij de Adelies en deels bij de Gentoos. Zoek je soorten, dan is dat niet wat je wilt. Het silhouette zegt hoe duidelijk clusters uit elkaar liggen, niet waar ze voor zijn.",
            "three": "Hier weet je dat er drie soorten zijn, maar zonder labels zou je dat niet weten. En zelfs de soorten liggen minder duidelijk uit elkaar dan twee clusters: op de getallen alleen zou niemand ze kiezen.",
            "right": "k = 2 is de duidelijkste splitsing, korte snavels en flippers tegen lange; k = 3 vindt de soorten, en die wil een bioloog. De within-cluster sum of squares en de silhouette score beperken de keus; waarvoor je de clusters wilt, beslist.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De within-cluster sum of squares is wat de twee stappen van [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl) klein maken, en het silhouette gebruikt dezelfde gestandaardiseerde afstand als [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl). k is een hyperparameter, zoals de k van k-nearest neighbors, maar zonder labels is er geen validation set om hem mee te kiezen, dus kies je met de som, het silhouette en wat je van de data weet. In de volgende les, *De beste schaduw: PCA*, houden de pinguïns hun vier metingen, en zoek je minder getallen die er zo veel mogelijk van bewaren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_best, result_starts, result_wcss):
    L.wrap_up(result_wcss, result_starts, result_best, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
