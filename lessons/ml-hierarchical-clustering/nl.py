# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-hierarchical-clustering. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-hierarchical-clustering", "nl")
    C = L.checks   # de pinguïns uit ml-choosing-k, de boom van elke linkage, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Op de pinguïns daalt de beste within-cluster sum of squares van 684,0 met één cluster tot 57,8 met acht. Waarom kun je k niet kiezen op de kleinste som?",
        options={
            "right": "Hij daalt alleen maar als k groeit: met een cluster voor elke pinguïn is hij 0",
            "local": "k-means kan in een lokaal minimum eindigen, dus sommen van verschillende k kun je niet vergelijken",
            "scale": "De features zijn gestandaardiseerd, dus de som betekent niets",
        },
        correct="right",
        why={
            "local": "Met `n_init` starts vindt k-means voor elke k de beste som, en die kun je vergelijken: ze dalen als k groeit. Dat is juist het probleem. Zie *Het idee* in [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).",
            "scale": "Standaardiseren geeft elke feature evenveel invloed; de som zegt nog steeds hoe dicht de voorbeelden bij hun centroids liggen. Hij daalt alleen steeds als k groeit. Zie *Het idee* in [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).",
            "right": "Daarom zoek je de elleboog, of gebruik je de silhouette score. Vandaag een methode die je elke k tegelijk geeft.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_starts():
    quick_starts = L.choice(
        "quick-starts",
        question="Waarom draait `KMeans` vanaf `n_init` starts?",
        options={
            "k": "Om vanaf elke start een ander aantal clusters te proberen",
            "right": "Vanaf één start kan het in een lokaal minimum eindigen, met een grotere som dan de beste",
            "faster": "Zodat elke start minder stappen nodig heeft",
        },
        correct="right",
        why={
            "k": "Elke start gebruikt dezelfde k; ze verschillen alleen in waar de centroids beginnen. Zie *Het idee* in [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).",
            "faster": "Elke start draait tot geen enkel voorbeeld meer van cluster verandert; meer starts kosten meer tijd, niet minder. Zie *Het idee* in [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl).",
            "right": "En het houdt de start met de kleinste som. De methode van vandaag heeft geen start: hij geeft altijd dezelfde boom.",
        },
    )
    quick_starts
    return (quick_starts,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    k-means vraagt om k voordat het begint, en elke k is een nieuwe clustering: de clusters van k = 3 hoeven niet binnen die van k = 2 te liggen. Bij de pinguïns doen ze dat ook niet: een van de drie clusters van k-means, 64 pinguïns, is verdeeld over de twee clusters van k = 2, 21 en 43. Toch zitten er vaak groepen in groepen: soorten, en binnen een soort mannetjes en vrouwtjes. Hierarchical clustering bouwt één boom die elke k tegelijk bevat, elke clustering binnen die erboven, en je kiest k achteraf, door naar de boom te kijken.

    **Een voorbeeld uit het dagelijks leven.** Een bibliotheek ordent haar boeken per vak, elk vak per onderwerp en elk onderwerp per schrijver. Vraag om drie afdelingen of om dertig: elk van de dertig ligt binnen een van de drie, omdat ze allebei uit dezelfde ordening komen.
    """)
    return


@app.cell(hide_code=True)
def predict_chain():
    guess_chain = L.guess(
        "chain-guess",
        question="Neem de pinguïns op snavellengte en flipperlengte. Begin met elke pinguïn als zijn eigen cluster, en voeg steeds de twee dichtstbijzijnde clusters samen, waarbij de afstand tussen twee clusters die van hun dichtstbijzijnde paar pinguïns is. Stop als er twee clusters over zijn. Welke zijn dat?",
        options={
            "species": "De Adelies, en de Chinstraps met de Gentoos",
            "halves": "Twee groepen van ongeveer 170 pinguïns",
            "right": "341 pinguïns, en 1",
        },
        reveal="**341 en 1.** Die ene is een Chinstrap met een snavel van 58,0 mm, de langste van zijn soort, en korte flippers, 181 mm. Tussen de soorten liggen genoeg pinguïns dat een ketting van korte stapjes ze allemaal verbindt, lang voordat die ene pinguïn dichter bij een ander ligt.",
    )
    guess_chain
    return (guess_chain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Van onderop.** **Hierarchical clustering** begint in zijn gewone vorm met elk voorbeeld als zijn eigen cluster: 342 clusters bij de pinguïns. Het voegt de twee clusters samen die het dichtst bij elkaar liggen, en herhaalt dat tot er één cluster over is: 341 keer samenvoegen. Elke samenvoeging verbindt twee clusters die eerder gemaakt zijn, dus de clusters vormen een boom. Er is geen start om te kiezen en geen k: dezelfde data geven altijd dezelfde boom.

        **Linkage.** Welke twee clusters het dichtst bij elkaar liggen, hangt af van hoe je de afstand tussen twee clusters meet. Die regel is de **linkage**:

        - single linkage: de afstand van het dichtstbijzijnde paar, één voorbeeld uit elk cluster;
        - complete linkage: de afstand van het verste paar;
        - average linkage: de gemiddelde afstand over alle paren;
        - Ward: hoeveel de samenvoeging de within-cluster sum of squares zou laten stijgen, de som die k-means klein maakt; Ward voegt de twee clusters samen waarvoor die stijging het kleinst is. scipy tekent een samenvoeging van Ward op de wortel van twee keer die stijging: dezelfde volgorde, een andere schaal.

        **Het dendrogram.** Het **dendrogram** tekent de boom. De voorbeelden staan naast elkaar onderaan, en elke samenvoeging is een brug tussen twee takken, op de hoogte van hun afstand: hoe hoger de brug, hoe verder de twee clusters uit elkaar lagen. Voor k clusters snijd je de boom met een horizontale lijn onder zijn bovenste k − 1 samenvoegingen: de takken die de lijn kruist, zijn de clusters. Snijd je lager, dan krijg je meer clusters, elk binnen een van de clusters erboven.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **hierarchical clustering** | clustering die een boom van clusters in clusters bouwt, hier door steeds de twee dichtstbijzijnde clusters samen te voegen | |
        | **linkage** | de regel voor de afstand tussen twee clusters: single, complete, average of Ward | `linkage(X, "ward")` |
        | **dendrogram** | de boom als plaatje: elke samenvoeging een brug op de hoogte van haar afstand | |
        | snede | een horizontale lijn door het dendrogram: de takken die hij kruist, zijn de clusters | `fcluster(Z, 3, criterion="maxclust")` |
        | `Z` | de boom van scipy: een rij voor elke samenvoeging, met de twee clusters, hun afstand en de grootte van het nieuwe cluster | `Z = linkage(X, "ward")` |
        """),
        L.watch_out("Hierarchical clustering vergelijkt elk paar voorbeelden: voor de 342 pinguïns zijn dat 58.311 afstanden, voor een miljoen voorbeelden ongeveer 500 miljard. Voor grote datasets is k-means veel goedkoper."),
        mo.md(r"""
        Het plaatje toont Ward op de 342 pinguïns, op snavellengte en flipperlengte, gestandaardiseerd, zoals in de vorige les. Links het dendrogram, met de snede als <span class="c-hyper">gestreepte lijn</span>; de takken van elk cluster in zijn eigen kleur. Rechts de pinguïns in de kleur en vorm van hun cluster. Kleur en vorm betekenen hier een cluster. Schuif van 1 naar 4 clusters.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_cut_cell():
    def _frame(k):
        k = int(k)
        sizes = C.sizes("ward", k)
        if k == 1:
            said = "1 cluster: alle 342 pinguïns. Er is geen snede: elke samenvoeging is gedaan."
        else:
            said = (f"{k} clusters: {C.listed(sizes, L.lang)} pinguïns. De snede ligt op hoogte "
                    f"{L.number(C.cut('ward', k), 1)}, tussen de samenvoegingen op "
                    f"{L.number(float(C.tree('ward')[-k][2]), 1)} en {L.number(float(C.tree('ward')[-k + 1][2]), 1)}.")
        alt = ("Links: het dendrogram van Ward op de 342 pinguïns, met zijn bovenste samenvoeging op hoogte 29,2. "
               "Rechts: de pinguïns op snavellengte en flipperlengte, gestandaardiseerd, in de kleur van hun cluster. " + said)
        return (C.picture("ward", k, {"tree": "Ward: de boom", "leaves": "de 342 pinguïns, in de volgorde van de boom",
                                      "height": "hoogte van samenvoegen", "title": f"{k} cluster" + ("s" if k > 1 else ""),
                                      "x": "snavellengte, gestandaardiseerd", "y": "flipperlengte, gestandaardiseerd", "alt": alt}), said)

    explore_cut = L.explore("clusters", "clusters", _frame, start=1, values=C.KS, role="hyper",
                            task="Ga van 1 naar 4 clusters. Welk cluster splitst bij elke stap, en welke blijven zoals ze waren?")
    explore_cut
    return


@app.cell(hide_code=True)
def text_after_cut():
    mo.md(r"""
    Met twee clusters zet Ward 146 Adelies en 6 Chinstraps aan de ene kant, en de rest aan de andere: 123 Gentoos, 62 Chinstraps en 5 Adelies. Met drie splitst dat tweede cluster: 122 Gentoos met 2 Adelies en 2 Chinstraps, en 60 Chinstraps met 3 Adelies en 1 Gentoo. Met vier splitsen de Gentoos weer: 46 pinguïns, waarvan 43 mannetjes, tegen een cluster met vooral vrouwtjes. Clusters in clusters: soorten, en dan het geslacht. Het langste stuk zonder samenvoeging ligt tussen 14,1 en de bovenste samenvoeging op 29,2, waar je snijdt voor twee clusters; het volgende, tussen 8,7 en 14,1, geeft er drie. Een lang stuk zegt dat de samenvoeging erboven clusters verbindt die ver uit elkaar liggen: zo laat het dendrogram een elleboog zien. Met drie clusters is de within-cluster sum of squares van Ward 159,3, dicht bij de 157,8 van k-means, al zocht Ward nooit naar de beste drie: het bouwde de hele boom.

    Het tweede plaatje houdt drie clusters en verandert de linkage: 1 single, 2 complete, 3 average, 4 Ward. De hoogtes verschillen per linkage, omdat elke linkage de afstand tussen clusters op haar eigen manier meet.
    """)
    return


@app.cell(hide_code=True)
def explore_linkage_cell():
    def _frame(v):
        method = C.METHODS[int(v) - 1]
        name = C.NAMES[method][L.lang]
        said = f"{name[0].upper() + name[1:]}, 3 clusters: {C.listed(C.sizes(method, 3), L.lang)} pinguïns."
        alt = (f"Links: het dendrogram van {name} op de 342 pinguïns, gesneden in 3 clusters. "
               "Rechts: de pinguïns op snavellengte en flipperlengte, gestandaardiseerd, in de kleur van hun cluster. " + said)
        return (C.picture(method, 3, {"tree": f"{name[0].upper() + name[1:]}: de boom", "leaves": "de 342 pinguïns, in de volgorde van de boom",
                                      "height": "hoogte van samenvoegen", "title": "3 clusters",
                                      "x": "snavellengte, gestandaardiseerd", "y": "flipperlengte, gestandaardiseerd", "alt": alt}), said)

    explore_linkage = L.explore("linkage", "linkage", _frame, start=1, values=[1, 2, 3, 4], role="hyper",
                                task="Welke linkages vinden de drie soorten, en welke laten een cluster van één pinguïn over?")
    explore_linkage
    return


@app.cell(hide_code=True)
def text_after_linkage():
    mo.md(r"""
    Single linkage geeft 340, 1 en 1: de Chinstrap van de gok, en een Gentoo met de langste snavel van allemaal, 59,6 mm. Bij single linkage is het dendrogram een trap: één groot cluster dat de ene pinguïn na de andere opneemt. Average linkage geeft 176, 165 en 1: de Adelies met 26 Chinstraps, de Gentoos met 41 Chinstraps, en diezelfde Chinstrap alleen. Complete linkage, met 151, 134 en 57, en Ward, met 152, 126 en 64, vinden allebei de drie soorten: Ward met 14 pinguïns in het cluster van een andere soort, complete linkage met 23. Welke linkage goed is, hangt af van wat je een cluster noemt: single linkage volgt kettingen van voorbeelden die dicht bij elkaar liggen, complete linkage en Ward zoeken compacte klonten.

    ### Recept

    Zo cluster je zonder eerst k te kiezen:

    1. Standaardiseer de features, zoals voor k-means.
    2. Bouw de boom: `Z = linkage(X, method)`. Ward als je compacte klonten verwacht, zoals k-means.
    3. Kijk naar het dendrogram: een lang stuk zonder samenvoegingen is een goede plek om te snijden.
    4. Snijd in k clusters met `fcluster(Z, k, criterion="maxclust")`, en kijk naar hun grootte: een cluster van één of twee voorbeelden is een uitschieter, geen groep.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Zes getallen op een lijn, elk een voorbeeld met één feature: 1, 2, 4, 8, 9 en 15. De cel bouwt de boom met single linkage en met complete linkage, en snijdt elk in twee clusters. scipy nummert de zes voorbeelden 0 tot en met 5, en het cluster dat rij $i$ van `Z` maakt, geteld vanaf 0, krijgt nummer $6 + i$. De commentaren met `#` zeggen welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_tree():
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

    # 1. zes voorbeelden met elk één feature, zoals ze zijn, dus de hoogtes zijn gaten tussen de getallen
    _points = np.array([[1.0], [2.0], [4.0], [8.0], [9.0], [15.0]])

    # 2. de boom, met het dichtstbijzijnde paar en met het verste paar: per rij de twee clusters, hun afstand, de nieuwe grootte
    _single = linkage(_points, "single")
    _complete = linkage(_points, "complete")
    print(_single)
    print(_complete)

    # 4. snijd elke boom in twee clusters: het cluster van elk voorbeeld
    print(fcluster(_single, 2, criterion="maxclust"))
    print(fcluster(_complete, 2, criterion="maxclust"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Beide bomen beginnen hetzelfde: 1 en 2 gaan op afstand 1 samen tot cluster 6, en 8 en 9 op afstand 1 tot cluster 7. Dan voegt single linkage 4 bij cluster 6 op 2, het gat tussen 2 en 4; complete linkage op 3, de afstand van 4 tot 1. Daarna gaan ze uiteen. Single linkage voegt {1, 2, 4} samen met {8, 9} op 4, het gat tussen 4 en 8, en voegt 15 als laatste toe, op 6. Complete linkage voegt eerst 15 bij {8, 9}, op 7, de afstand van 15 tot 8, en voegt de twee helften als laatste samen, op 14, de afstand van 1 tot 15. In twee gesneden geeft single linkage `[1 1 1 1 1 2]`: 15 alleen. Complete linkage geeft `[1 1 1 2 2 2]`: {1, 2, 4} en {8, 9, 15}. Dezelfde zes getallen, twee verschillende bomen, omdat de linkage bepaalt wat dichtbij betekent.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_link():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het dichtstbijzijnde paar

        `single_link(A, B)` moet de afstand van single linkage tussen twee clusters teruggeven: `A` en `B` zijn arrays met een rij per voorbeeld. De twee loops gaan al langs elk paar, één voorbeeld uit elk cluster. Vervang de twee `...`. De check roept `single_link(A, B)` aan met een cluster van twee voorbeelden en een van drie, en probeert er nog vier: de getallen 1, 2, 4 en 8, 9 uit het uitgewerkte voorbeeld, drie features, één voorbeeld per cluster, en een lang cluster naast een kort.
        """),
        L.task("Laat `single_link` de afstand van het dichtstbijzijnde paar teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_link():
    box_link = L.code_box("single-link", lines=9)
    box_link
    return (box_link,)


@app.cell(hide_code=True)
def check_link(box_link):
    result_link = L.check("single-link", box_link.value)
    result_link
    return (result_link,)


@app.cell(hide_code=True)
def hints_link():
    L.hints(
        "single-link",
        [
            "Single linkage: de afstand tussen twee clusters is die van hun dichtstbijzijnde paar. De loops bezoeken al elk paar; verzamel hun afstanden en neem dan de kleinste.",
            "De afstand tussen twee voorbeelden is de lengte van hun verschil, zoals bij k-NN. Welke ingebouwde functie geeft het kleinste getal van een lijst?",
            "`distances.append(np.linalg.norm(___))` en `return ___(distances)`.",
        ],
        why="Voor `A` en `B` is het dichtstbijzijnde paar (1, 0) in `A` en (4, 0) in `B`: 3 uit elkaar. Het verste paar, (0, 0) en (5, 1), ligt 5,099 uit elkaar: dat zou complete linkage zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_largest():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Elke linkage

        De code leest de pinguïns zoals in de vorige les en gaat langs de vier linkages. Voor elke linkage moet hij de boom van `X` bouwen, die in drie clusters snijden en de grootte van het grootste cluster bewaren. Vervang de twee `...`. De laatste regel laat `largest` zien.
        """),
        L.task("Zoek voor elke linkage uit hoeveel pinguïns het grootste van drie clusters bevat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_largest():
    box_largest = L.code_box("largest", lines=16)
    box_largest
    return (box_largest,)


@app.cell(hide_code=True)
def check_largest(box_largest):
    result_largest = L.check("largest", box_largest.value)
    result_largest
    return (result_largest,)


@app.cell(hide_code=True)
def hints_largest():
    L.hints(
        "largest",
        [
            "Stap 2 en 4 van het recept, zoals in het uitgewerkte voorbeeld: `linkage` bouwt de boom, `fcluster` snijdt hem.",
            "De linkage van deze ronde van de loop staat in `method`. Om in een aantal clusters te snijden, heeft `fcluster` `criterion=\"maxclust\"` nodig.",
            "`Z = linkage(X, ___)` en `labels = fcluster(Z, ___, criterion=\"maxclust\")`.",
        ],
        why="Met single linkage bevatten de drie clusters 340, 1 en 1 pinguïns, dus het grootste is 340.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.md(r"""
    ### Oefening 3. Een lange ketting
    """)
    return


@app.cell(hide_code=True)
def choice_chain_cell():
    choice_chain = L.choice(
        "chain",
        question="Met single linkage bevatten drie clusters van de pinguïns 340, 1 en 1 pinguïns; met Ward 152, 126 en 64. Waarom zet single linkage bijna elke pinguïn in één cluster?",
        options={
            "outliers": "De pinguïns hebben twee uitschieters die zo ver weg liggen dat geen linkage de rest kan splitsen",
            "right": "Tussen de soorten liggen pinguïns die er half tussenin zitten, en single linkage kijkt alleen naar het dichtstbijzijnde paar, dus een ketting van korte stapjes verbindt de soorten",
            "scale": "De features zijn niet gestandaardiseerd, dus één feature bepaalt de afstand",
        },
        correct="right",
        why={
            "outliers": "Ward en complete linkage splitsen dezelfde pinguïns in drie echte groepen; de twee eenlingen houden ze niet tegen. Zie het tweede plaatje.",
            "scale": "De code standaardiseert beide features, zoals in de vorige les; de andere linkages gebruiken dezelfde afstanden. Zie *Oefening 2*.",
            "right": "Twee clusters tellen als dichtbij zodra één paar dichtbij ligt, hoe ver de rest ook uit elkaar ligt. Dat heet chaining: goed voor clusters in de vorm van lange banden, slecht voor de compacte klonten van de soorten.",
        },
    )
    choice_chain
    return (choice_chain,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Ward maakt dezelfde within-cluster sum of squares klein als [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl), één samenvoeging tegelijk, en de lange stukken van het dendrogram doen het werk van de elleboog uit [Hoeveel clusters? k kiezen](/course/open/ml-choosing-k?lang=nl). Single linkage volgt kettingen van voorbeelden die dicht bij elkaar liggen, ten goede en ten kwade. In de volgende les, *Clusters van elke vorm: DBSCAN*, is een cluster een gebied waar voorbeelden dicht bij elkaar liggen, van elke vorm, en is een voorbeeld ver van elk gebied ruis in plaats van een eigen cluster.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chain, result_largest, result_link):
    L.wrap_up(result_link, result_largest, choice_chain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
