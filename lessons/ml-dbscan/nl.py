# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-dbscan. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-dbscan", "nl")
    C = L.checks   # de 400 punten, DBSCAN voor elke eps, k-means en single linkage erop, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Clusters in clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=nl) en één over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_single():
    quick_single = L.choice(
        "quick-single",
        question="Single linkage vindt twee clusters dichtbij zodra…",
        options={
            "right": "één voorbeeld van het ene dicht bij één voorbeeld van het andere ligt",
            "means": "hun gemiddelden dicht bij elkaar liggen",
            "all": "elk voorbeeld van het ene dicht bij elk voorbeeld van het andere ligt",
        },
        correct="right",
        why={
            "means": "Single linkage kijkt naar paren voorbeelden, niet naar gemiddelden: het neemt de afstand van het dichtstbijzijnde paar. Zie *Het idee* in [Clusters in clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=nl).",
            "all": "Dat is complete linkage, dat het verste paar neemt. Single linkage neemt het dichtstbijzijnde paar. Zie *Het idee* in [Clusters in clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=nl).",
            "right": "Zo verbindt een ketting van voorbeelden die dicht bij elkaar liggen twee groepen, hoe ver de rest ook uit elkaar ligt. Vandaag een methode die ook vormen volgt, maar groepen niet door een paar losse voorbeelden aaneen laat rijgen.",
        },
    )
    quick_single
    return (quick_single,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="k-NN meet de afstand tussen pinguïns op snavellengte, tientallen millimeters, en lichaamsgewicht, duizenden grammen. Wat gebeurt er zonder standaardiseren?",
        options={
            "error": "k-NN kan millimeters niet met grammen vergelijken en stopt met een fout",
            "right": "De afstand is bijna alleen het lichaamsgewicht",
            "nothing": "Niets: k-NN vraagt alleen welke voorbeelden het dichtstbij liggen, niet hoe ver weg ze zijn",
        },
        correct="right",
        why={
            "error": "NumPy trekt de getallen af en kwadrateert ze, wat hun eenheid ook is. Het probleem is dat een verschil van honderden grammen zwaarder weegt dan een van een paar millimeter. Zie de waarschuwing in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "nothing": "Welke voorbeelden het dichtstbij liggen, hangt af van de afstand, en die is bijna alleen het lichaamsgewicht. Zie de waarschuwing in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "Standaardiseer dus eerst. De punten van vandaag hebben twee features op dezelfde schaal, dus die kunnen blijven zoals ze zijn.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    k-means zet elk voorbeeld bij zijn dichtstbijzijnde centroid, dus zijn clusters zijn ronde klonten rond een middelpunt, en Ward zoekt hetzelfde. Single linkage kan een vorm van elke soort volgen, maar elk voorbeeld moet ergens bij horen: een los voorbeeld wordt een eigen cluster, en een paar losse voorbeelden tussen twee groepen rijgen die aaneen. Echte groepen zijn niet altijd rond, en echte data heeft voorbeelden die bij geen enkele groep horen, zoals een verkeerd ingetikte meting of een zeldzame gebeurtenis. Een cluster kan ook een gebied zijn waar voorbeelden dicht bij elkaar liggen, welke vorm het ook heeft, met de voorbeelden buiten al zulke gebieden als ruis over.

    **Een voorbeeld uit het dagelijks leven.** Vlieg 's nachts over een land. Een stad is waar lichtjes dicht bij elkaar liggen, of die nu langs een rivier ligt of rond een meer, en één lichtje van een boerderij ver van alles is geen stad.
    """)
    return


@app.cell(hide_code=True)
def predict_shapes():
    guess_shapes = L.guess(
        "shapes-guess",
        question="De course maakte 400 punten met twee features: een ring van 200, een klont van 80 in het midden ervan, een gebogen band van 100 ernaast, en 20 losse punten overal. Je draait k-means met k = 3 erop. Wat vindt het?",
        options={
            "shapes": "De ring, de klont en de band",
            "right": "De band, en de ring in twee helften gesneden, elk met een deel van de klont",
            "strays": "Drie groepen losse punten",
        },
        reveal="**De band, en twee helften van de ring.** k-means maakt ronde clusters rond drie centroids: 99 punten van de ring met 63 van de klont, de andere 101 punten van de ring met de andere 17, en de band met 10 losse punten. Single linkage, in drie clusters gesneden, volgt de vormen wel, maar de losse punten slaan er bruggen tussen: 398 punten in één cluster, en twee losse punten alleen.",
    )
    guess_shapes
    return (guess_shapes,)


@app.cell(hide_code=True)
def picture_others_cell():
    mo.vstack([
        mo.md(r"""
        De 400 punten, links in de drie clusters van k-means, rechts in die van single linkage. Kleur en vorm betekenen hier een cluster.
        """),
        mo.Html(C.picture_others({"kmeans": "k-means, k = 3", "single": "Single linkage, 3 clusters", "x": "feature 1", "y": "feature 2",
                                  "alt": "Twee grafieken van de 400 punten: een ring, een klont in het midden, een gebogen band rechts en losse punten. Links k-means met k = 3: de band is één cluster, en de ring is in twee helften gesneden, elk met een deel van de klont. Rechts single linkage, in 3 clusters gesneden: 398 punten in één cluster en twee losse punten alleen."})),
    ])
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Dichte gebieden.** **DBSCAN** is een vorm van **density-based clustering**: een cluster is een gebied waar voorbeelden dicht bij elkaar liggen, van elke vorm, en een voorbeeld in geen enkel zo'n gebied is ruis. De naam is kort voor density-based spatial clustering of applications with noise. Twee instellingen zeggen wat dicht bij elkaar betekent: een straal, `eps`, en een aantal, `min_samples`.

        - De buurt van een voorbeeld is elk voorbeeld binnen `eps` ervan, zichzelf meegeteld.
        - Een **core point** heeft minstens `min_samples` voorbeelden in zijn buurt.
        - Een cluster begint bij een core point en neemt zijn hele buurt op. Elk core point dat het opneemt, voegt zijn eigen buurt toe, enzovoort, tot er niets nieuws meer bijkomt. Zo kan een cluster langs een ring of een band groeien, zolang er core points op elkaar volgen.
        - Een voorbeeld in een cluster dat zelf geen core point is, ligt aan de rand ervan: een randpunt. Het hoort bij het cluster, maar voegt er niets aan toe.
        - Een **noise point** ligt binnen `eps` van geen enkel core point. Het hoort bij geen cluster: scikit-learn geeft het het label −1.

        Er is geen k: het aantal clusters komt eruit. En waar single linkage twee clusters verbindt via elk paar voorbeelden dat dicht bij elkaar ligt, groeit DBSCAN alleen via core points, dus een los voorbeeld tussen twee vormen verbindt ze niet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **density-based clustering** | clustering waarin een cluster een gebied is waar voorbeelden dicht bij elkaar liggen, van elke vorm | |
        | **DBSCAN** | density-based clustering die clusters laat groeien vanuit core points en de rest als ruis laat liggen | `DBSCAN(eps=0.4, min_samples=5)` |
        | `eps` | de straal van een buurt | `eps=0.4` |
        | `min_samples` | hoeveel voorbeelden een buurt nodig heeft, zichzelf meegeteld, voor een core point | `min_samples=5` |
        | **core point** | een voorbeeld met minstens `min_samples` voorbeelden binnen `eps` | `db.core_sample_indices_` |
        | **noise point** | een voorbeeld binnen `eps` van geen enkel core point: in geen cluster | het label `-1` in `db.labels_` |
        """),
        L.watch_out("Eén `eps` geldt voor alle data. Waar het ene cluster minder dicht is dan het andere, past misschien geen enkele `eps` bij allebei: bij `eps` = 0,3 zijn de klont en de band heel, terwijl de ring, waarvan de punten verder uit elkaar liggen, in vier stukken uiteenvalt."),
        mo.md(r"""
        Het plaatje toont DBSCAN met `min_samples` = 5 op de 400 punten. Elk cluster heeft een eigen kleur en vorm: de drie grootste elk hun eigen, alle andere de vierde. Core points zijn groot, randpunten klein, en noise points zijn open grijze rondjes. De <span class="c-hyper">gestreepte cirkel</span> is de buurt van één punt van de ring, met straal `eps`. Kleur en vorm betekenen hier een cluster. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_eps_cell():
    def _frame(eps):
        f = C.fit(eps)
        sizes = (f"van {C.listed(f['sizes'], L.lang)} punten" if f["clusters"] <= 3
                 else f"de grootste van {f['sizes'][0]} punten")
        said = f"`eps` = {L.number(eps, 1)}: {f['clusters']} clusters, {sizes}, en {f['noise']} noise points."
        alt = ("De 400 punten: een ring, een klont in het midden, een gebogen band rechts en losse punten, in de clusters van DBSCAN "
               f"met min_samples 5 en eps {L.number(eps, 1)}, met een gestreepte cirkel met die straal om één punt van de ring. "
               + said.replace("`", ""))
        return (C.picture(eps, {"title": f"DBSCAN, eps = {L.number(eps, 1)}", "x": "feature 1", "y": "feature 2", "alt": alt}), said)

    explore_eps = L.explore("eps", "eps", _frame, start=0.1, values=C.EPS, role="hyper",
                            task="Ga van `eps` = 0,1 naar 1,0. Vanaf welke `eps` zijn de ring, de klont en de band elk één cluster, en bij welke `eps` worden twee ervan één?")
    explore_eps
    return


@app.cell(hide_code=True)
def text_after_eps():
    mo.md(r"""
    Bij `eps` = 0,1 hebben de meeste punten te weinig buren: 264 noise points, en 15 kleine clusters. Bij 0,2 zijn de klont en de band bijna heel, maar ligt de ring nog in stukken. Bij 0,3 valt de ring uiteen in vier stukken, van 72, 52, 50 en 27 punten: zijn punten liggen verder uit elkaar dan die van de klont en de band. Van 0,4 tot 0,7 vindt DBSCAN de drie vormen, en zijn 15 tot 13 noise points zijn allemaal losse punten; de andere losse punten liggen dicht genoeg bij een vorm om erbij te horen. Bij 1,0 worden de ring en de klont één cluster van 288 punten. Drie clusters houden stand over een breed stuk van `eps`, van 0,4 tot 0,7: een goed teken dat ze echt zijn.

    ### Recept

    Zo vind je clusters van elke vorm, en ruis:

    1. Standaardiseer de features, tenzij ze op dezelfde schaal liggen, zoals deze twee.
    2. Kies `min_samples`: 5, de standaardwaarde van scikit-learn, is een goed begin; meer voor grote data met veel ruis.
    3. Probeer een reeks waarden van `eps`, en kijk naar het aantal clusters, het aantal noise points en de clusters zelf. Een aantal clusters dat standhoudt over een stuk van `eps` is een goed teken.
    4. `DBSCAN(eps=..., min_samples=...).fit(X).labels_` geeft elk voorbeeld zijn cluster, en −1 voor ruis.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Weer de zes getallen uit de vorige les: 1, 2, 4, 8, 9 en 15, elk een voorbeeld met één feature, met `eps` = 2. De cel draait DBSCAN met `min_samples` 2 en 3, en print voor elk: `min_samples`, het cluster van elk voorbeeld en de plaatsen van de core points. De commentaren met `#` zeggen welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_dbscan():
    import numpy as np
    from sklearn.cluster import DBSCAN

    # 1. zes voorbeelden met één feature: niets te standaardiseren
    _points = np.array([[1.0], [2.0], [4.0], [8.0], [9.0], [15.0]])

    # 2. en 4. DBSCAN met eps 2, voor twee waarden van min_samples: het cluster van elk voorbeeld, en de core points
    for _m in [2, 3]:
        _db = DBSCAN(eps=2, min_samples=_m).fit(_points)
        print(_m, _db.labels_, _db.core_sample_indices_)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Binnen 2 van elkaar liggen 1 en 2, 2 en 4, en 8 en 9; 15 heeft geen buur. Met `min_samples` = 2 is elk getal met een buur een core point, de vijf op de plaatsen 0 tot en met 4: 1, 2 en 4 groeien uit tot cluster 0, 8 en 9 tot cluster 1, en 15 is ruis, −1. Met `min_samples` = 3 heeft alleen 2 er genoeg: 1, 2 en 4, zichzelf meegeteld, dus de core points zijn alleen `[1]`, de plaats van de 2. Zijn cluster neemt 1 en 4 nog steeds op, als randpunten, maar 8 en 9 hebben geen core point tussen zich en worden ruis, met 15.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_core():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Core points

        `core_points(X, eps, min_samples)` moet voor elke rij van `X` teruggeven of het een core point is: `True` of `False`. De loop rekent al voor elk voorbeeld `x` de afstand uit tot elk voorbeeld, zichzelf meegeteld. Vervang de twee `...`. De check roept `core_points(X, 0.6, 3)` aan met acht punten, en probeert er nog vijf: een rooster waarvan de buren precies `eps` uit elkaar liggen, de getallen uit het uitgewerkte voorbeeld met `min_samples` 2 en 3, `min_samples` 1, en drie features.
        """),
        L.task("Laat `core_points` zeggen welke voorbeelden core points zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_core():
    box_core = L.code_box("core-points", lines=10)
    box_core
    return (box_core,)


@app.cell(hide_code=True)
def check_core(box_core):
    result_core = L.check("core-points", box_core.value)
    result_core
    return (result_core,)


@app.cell(hide_code=True)
def hints_core():
    L.hints(
        "core-points",
        [
            "Een core point heeft minstens `min_samples` voorbeelden binnen `eps`, zichzelf meegeteld. `distances` bevat de afstand van `x` tot elk voorbeeld: hoeveel daarvan zijn hoogstens `eps`?",
            "Een vergelijking op een array geeft een array van True en False, en `np.sum` telt de Trues. Vergelijk dat aantal dan met `min_samples`.",
            "`count = np.sum(distances ___ eps)` en `core.append(count ___ min_samples)`.",
        ],
        why="Van de acht punten heeft (0, 0) zichzelf, (0,5, 0) en (0, 0,5) binnen 0,6: 3, dus het is een core point. (1, 0) heeft zichzelf en (0,5, 0): 2, één te weinig.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_samples():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vijf waarden van min_samples

        In dit vak staat `X` klaar: de 400 punten van het plaatje. De loop draait DBSCAN met `eps` = 0,4 voor vijf waarden van `min_samples`. Vervang de twee `...`: hoeveel noise points er zijn, en hoeveel clusters. De laatste regel laat `found` zien.
        """),
        L.task("Zoek voor elke `min_samples` uit hoeveel clusters en noise points DBSCAN geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_samples():
    box_samples = L.code_box("min-samples", lines=10)
    box_samples
    return (box_samples,)


@app.cell(hide_code=True)
def check_samples(box_samples):
    result_samples = L.check("min-samples", box_samples.value)
    result_samples
    return (result_samples,)


@app.cell(hide_code=True)
def hints_samples():
    L.hints(
        "min-samples",
        [
            "Noise points hebben het label −1; de clusters hebben de labels 0, 1, 2 en hoger.",
            "`labels == -1` geeft een True of False voor elk punt, en `np.sum` telt de Trues. Tel voor de clusters de verschillende labels, maar laat −1 weg.",
            "`n_noise = int(np.sum(labels ___ -1))` en `n_clusters = len(np.unique(labels[labels ___ 0]))`.",
        ],
        why="Met `min_samples` 5, zoals op de slider bij `eps` = 0,4: 3 clusters en 15 noise points.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_single():
    mo.md(r"""
    ### Oefening 3. Eén voorbeeld is genoeg
    """)
    return


@app.cell(hide_code=True)
def choice_single_cell():
    choice_single = L.choice(
        "single",
        question="Met `min_samples` = 1 is elk voorbeeld een core point, omdat zijn buurt zichzelf bevat. Welke clusters geeft DBSCAN dan?",
        options={
            "kmeans": "De clusters van k-means",
            "right": "De clusters van single linkage, met de boom gesneden op hoogte `eps`",
            "noise": "Geen: elk voorbeeld is ruis, omdat geen enkel genoeg buren heeft",
        },
        correct="right",
        why={
            "kmeans": "k-means heeft k en centroids nodig, en DBSCAN heeft geen van beide. Met `min_samples` = 1 komt elk voorbeeld binnen `eps` van een ander bij dat andere. Zie *Oefening 2*, waar `min_samples` 1 16 clusters geeft.",
            "noise": "Een voorbeeld ligt in zijn eigen buurt, dus met `min_samples` = 1 is elk voorbeeld een core point en is geen enkel ruis. Zie *Oefening 2*: met `min_samples` 1 is er helemaal geen ruis.",
            "right": "Twee voorbeelden binnen `eps` van elkaar komen in één cluster, en elke ketting van zulke stapjes ook: precies de clusters die single linkage heeft als elke samenvoeging boven hoogte `eps` ongedaan is gemaakt. Op de 400 punten met `eps` = 0,4 geven ze allebei dezelfde 16 clusters. `min_samples` is dus wat het aaneenrijgen tegenhoudt: een voorbeeld ertussenin verbindt twee clusters alleen als het zelf genoeg buren heeft.",
        },
    )
    choice_single
    return (choice_single,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** DBSCAN vraagt over elk voorbeeld wat [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) over een nieuw voorbeeld vraagt, welke voorbeelden dichtbij liggen, maar binnen een straal `eps` in plaats van de dichtstbijzijnde k. Met `min_samples` = 1 is het de single linkage uit [Clusters in clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=nl), gesneden op `eps`; een grotere `min_samples` houdt het aaneenrijgen tegen, en laat losse voorbeelden als ruis liggen in plaats van als eigen clusters. In de volgende les, *Een kaart van veel dimensies: t-SNE*, teken je voorbeelden met veel features op een platte kaart, waar groepen zoals deze zichtbaar worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_single, result_core, result_samples):
    L.wrap_up(result_core, result_samples, choice_single)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
