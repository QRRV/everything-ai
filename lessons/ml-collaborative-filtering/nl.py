# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-collaborative-filtering. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-collaborative-filtering", "nl")
    C = L.checks   # de user-item matrix, de similarities van de producten, de drie recommenders, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Meer van wat je kocht: content-based filtering](/course/open/ml-content-based?lang=nl) en één over [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_profile():
    quick_profile = L.choice(
        "quick-profile",
        question="Content-based filtering raadde klant 13960 tien hartjes aan, waarvan negen rieten. Wat moest het daarvoor weten?",
        options={
            "right": "De woorden van de productnamen, en wat de klant kocht",
            "others": "Wat andere klanten kochten",
            "stars": "Hoeveel sterren de klant elk product gaf",
        },
        correct="right",
        why={
            "others": "Content-based filtering kijkt alleen naar de eigen producten van één klant en naar de namen van alle producten: andere klanten spelen geen rol. Zie *Het gebruikersprofiel* in [Meer van wat je kocht: content-based filtering](/course/open/ml-content-based?lang=nl).",
            "stars": "De winkel heeft geen sterren: een aankoop is implicit feedback. Zie *Implicit feedback* in [Wat laat je hierna zien: het aanbevelingsprobleem](/course/open/ml-recommend-problem?lang=nl).",
            "right": "De namen geven de item features, de aankopen het profiel. Vandaag spelen de namen geen rol: alleen wie wat kocht.",
        },
    )
    quick_profile
    return (quick_profile,)


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`R` heeft vorm (2829, 3576): een rij voor elke klant, een kolom voor elk product. `R[:, j]` is kolom `j`, met vorm (2829,). Welke vorm heeft `R.T @ R[:, j]`?",
        options={
            "right": "(3576,): een getal voor elk product",
            "customers": "(2829,): een getal voor elke klant",
            "error": "Een fout: de vormen passen niet",
        },
        correct="right",
        why={
            "customers": "`R.T` heeft vorm (3576, 2829): een rij voor elk product. Keer een vector van 2.829 getallen geeft dat één getal per rij van `R.T`, dus per product. Zie *Het idee* in [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl).",
            "error": "`R.T` heeft vorm (3576, 2829), en zijn 2.829 kolommen passen bij de 2.829 getallen van `R[:, j]`. Zie de shape-regel in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl).",
            "right": "Elk getal is het inproduct van een kolom van `R` met kolom `j`. Vandaag is dat inproduct het aantal klanten dat beide producten kocht.",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Content-based filtering moet weten wat een product is, en een naam zegt weinig: PARTY BUNTING en REGENCY CAKESTAND 3 TIER hebben geen woord gemeen, en toch kopen veel klanten ze voor hetzelfde feest. De winkel weet iets beters dan de namen: wie wat kocht, voor duizenden klanten. Een product dat veel van dezelfde klanten kopen, past waarschijnlijk goed, wat zijn naam ook is.

    **Een voorbeeld uit het dagelijks leven.** Een platenwinkelier ziet dat de klanten die het ene album kopen vaak ook een bepaald ander album kopen, en zet de twee naast elkaar. Hoe die albums klinken, hoeft de winkelier niet te weten.
    """)
    return


@app.cell(hide_code=True)
def predict_collaborative():
    guess_collaborative = L.guess(
        "collaborative-guess",
        question="Raad elke klant de tien producten aan die klanten met dezelfde aankopen ook kochten, zonder naar de namen te kijken. Bij hoeveel van de 2.498 klanten zit minstens één van de tien in de laatste bestelling?",
        options={
            "fewer": "Minder dan met populariteit, 21,7%: zonder namen weet het minder",
            "same": "Ongeveer evenveel als met de namen, 23,8%",
            "right": "Meer dan met allebei: ongeveer een derde",
        },
        reveal="**Ongeveer een derde.** 800 van de 2.498 klanten, 32,0%: meer dan de namen, 595, en meer dan populariteit, 541. Wie wat kocht, zegt meer over een product dan zijn naam. En samen vinden de drie iets voor 1.195 klanten, 47,8%.",
    )
    guess_collaborative
    return (guess_collaborative,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een product is wie het kocht.** Neem de user-item matrix van de bestellingen vóór de laatste van elke klant: 2.829 klanten en 3.576 producten, met 195.264 enen. Noem hem `R`. Een rij is een klant; een kolom is een product, beschreven door wie het kocht: de kolom van HEART OF WICKER SMALL heeft een 1 voor elk van zijn 423 kopers. De namen spelen geen rol.

        **Verwant als dezelfde klanten ze kopen.** Het inproduct van twee kolommen telt de klanten die beide producten kochten. `R.T @ R[:, j]` geeft het voor kolom `j` en elke kolom tegelijk. Maar een telling bevoordeelt wat populair is: van de 423 kopers van het kleine rieten hartje kochten er 216 ook de WHITE HANGING HEART T-LIGHT HOLDER, maar die houder heeft 642 kopers, en veel van hen hadden hem toch wel gekocht. De cosine similarity van de twee kolommen deelt door beide lengtes. Een kolom van nullen en enen is zo lang als de wortel van zijn aantal kopers: 216 / √(423 × 642) = 0,414 voor de houder, tegen 254 / √(423 × 368) = 0,644 voor HEART OF WICKER LARGE, met 368 kopers. Dat is de **item-item similarity**: hoe sterk twee producten op elkaar lijken volgens wie ze kocht.

        **Collaborative filtering.** Geef elk product een score: zijn similarities met de producten die een klant kocht, opgeteld. Raad de `k` met de hoogste scores aan die de klant nog niet kocht, gelijke scores in de volgorde van populariteit. Dat is **collaborative filtering**: de klanten samen bepalen wat op elkaar lijkt, en de recommender hoeft niets van de producten te weten.

        **Cold start.** Een nieuw product dat nog niemand kocht, heeft een kolom van nullen: het lijkt op niets, en niets brengt het naar boven. Een nieuwe klant heeft nog niets gekocht, dus valt er niets op te tellen. Dat is het **cold start**-probleem. In deze winkel plaatsten 1.505 van de 4.334 klanten één bestelling: voor die bestelling hadden collaborative filtering en content-based filtering niets om op af te gaan. Nieuwe producten zijn hier zeldzaam: van de 37.914 producten die in de laatste bestelling van een klant nieuw voor die klant waren, hebben er 26 geen kolom in `R`: geen klant kocht ze vóór de eigen laatste bestelling.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **item-item similarity** | hoe sterk twee items op elkaar lijken volgens de gebruikers die ze hadden: de cosine similarity van hun kolommen in de user-item matrix | |
        | **collaborative filtering** | aanraden op basis van wat veel gebruikers deden, zonder te weten wat de items zijn | |
        | **cold start** | een nieuwe gebruiker of een nieuw item waar een recommender nog niets van weet | |
        """),
        mo.md(r"""
        Het plaatje laat allebei zien voor HEART OF WICKER SMALL. Links de vijf producten die het vaakst samen ermee gekocht werden; rechts de vijf die er volgens de cosine similarity het meest op lijken. Bij elk staat hoeveel klanten het kochten.
        """),
        mo.Html(C.pairs_picture(C.PRODUCT, {"together": "Het vaakst samen gekocht", "alike": "Lijken het meest (cosine similarity)",
                                            "buyers": "{n} kopers",
                                            "alt": "Twee lijsten van vijf producten. Het vaakst samen gekocht met HEART OF WICKER SMALL: HEART OF WICKER LARGE, 254 klanten; WHITE HANGING HEART T-LIGHT HOLDER, 216; NATURAL SLATE HEART CHALKBOARD, 157; SMALL WHITE HEART OF WICKER, 155; REGENCY CAKESTAND 3 TIER, 138. Lijken het meest volgens de cosine similarity: HEART OF WICKER LARGE, 0,644; SMALL WHITE HEART OF WICKER, 0,482; LARGE WHITE HEART OF WICKER, 0,438; WHITE HANGING HEART T-LIGHT HOLDER, 0,414; WICKER STAR, 0,369."}, L.lang)),
        mo.md(r"""
        De tellingen halen de producten binnen die iedereen koopt: de REGENCY CAKESTAND 3 TIER, het populairste product van de winkel, komt vijfde met 138 klanten, maar met 663 kopers is zijn cosine similarity maar 0,261. De cosine similarity zet de witte rieten hartjes en een WICKER STAR op hun plek: wat kopers van dit hartje vaker kopen dan andere klanten.
        """),
        L.watch_out("Tel je tellingen op in plaats van similarities, dan winnen voor iedereen de populaire producten. Met als score hoeveel klanten beide kochten, vinden de tien producten iets voor 657 klanten, 26,3%; met de cosine similarities voor 800, 32,0%."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo raad je aan met item-item collaborative filtering:

    1. Maak de user-item matrix `R` van wat de gebruikers vóór de splitsing hadden: een rij voor elke gebruiker, een kolom voor elk item.
    2. Tel voor een item `j` bij elk item hoeveel gebruikers ze allebei hadden: `R.T @ R[:, j]`.
    3. Deel door de lengtes van beide kolommen: de item-item similarity.
    4. Tel voor een gebruiker de similarities op met elk item dat de gebruiker had, en raad de `k` hoogste aan die de gebruiker nog niet had.
    5. Meet de hit rate, en vergelijk met de andere recommenders op dezelfde gebruikers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel volgt stap 1 en 2. Ze splitst de bestellingen op tijd zoals eerder, geeft elk product een kolom, de populairste eerst, en elke klant met een bestelling vóór de laatste een rij. Er komt een 1 waar de klant het product kocht. Daarna telt ze voor HEART OF WICKER SMALL hoeveel van zijn kopers elk product kochten, en print ze de zes hoogste tellingen: de telling, hoeveel klanten dat product kochten, en de naam. Voer de cel uit.
    """)
    return


@app.cell
def example_together():
    import numpy as np
    import pandas as pd

    # wat elke klant vóór de laatste bestelling kocht, en de producten, de populairste eerst
    _rows = pd.read_csv("data/online-retail-baskets.csv.gz").dropna(subset=["customer"])
    _rows["last"] = _rows.groupby("customer")["invoice"].transform("max")
    _train = _rows[_rows["invoice"] < _rows["last"]]
    _seen = _train.groupby("customer")["product"].agg(set)
    _names = list(_train.drop_duplicates(subset=["customer", "product"])["product"].value_counts().index)

    # 1. de user-item matrix: een rij voor elke klant, een kolom voor elk product
    _col_of = {}
    for _i in range(len(_names)):
        _col_of[_names[_i]] = _i
    _customers = list(_seen.index)
    _R = np.zeros((len(_customers), len(_names)))
    for _r in range(len(_customers)):
        for _product in _seen[_customers[_r]]:
            _R[_r, _col_of[_product]] = 1
    print(_R.shape, _R.sum())

    # 2. voor elk product: hoeveel van de kopers van HEART OF WICKER SMALL het ook kochten
    _j = _col_of["HEART OF WICKER SMALL"]
    _together = _R.T @ _R[:, _j]
    for _i in np.argsort(-_together, kind="stable")[:6]:
        print(int(_together[_i]), int(_R[:, _i].sum()), _names[_i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    2.829 rijen en 3.576 kolommen, met 195.264 enen. De eerste regel van de tellingen is het hartje zelf: alle 423 kopers kochten het. Dan het grote rieten hartje, 254 van 368 kopers, en het witte hangende hartje, 216 van 642. De tellingen zijn de inproducten van de kolommen; de lengtes die ze tot cosine similarities maken, zijn de wortels van de tweede getallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_similar():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Verwant volgens wie ze kocht

        `similar_to(R, j)` moet de cosine similarity van kolom `j` van `R` met elke kolom geven: een array met een getal voor elk product. In `together` staat al hoeveel klanten elk product samen met product `j` kochten, en in `lengths` de lengte van elke kolom. Vervang de `...`. De check roept `similar_to(R, j)` aan met de matrix van het uitgewerkte voorbeeld en `j` de kolom van HEART OF WICKER SMALL, en probeert drie kleine matrices.
        """),
        L.task("Laat `similar_to` de cosine similarity van kolom `j` met elke kolom geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_similar():
    box_similar = L.code_box("similar-to", lines=7)
    box_similar
    return (box_similar,)


@app.cell(hide_code=True)
def check_similar(box_similar):
    result_similar = L.check("similar-to", box_similar.value)
    result_similar
    return (result_similar,)


@app.cell(hide_code=True)
def hints_similar():
    L.hints(
        "similar-to",
        [
            "Het is dezelfde cosine similarity als in `cosine_scores` uit de vorige les, nu met de kolommen van `R`. Door welke twee lengtes deel je de telling?",
            "`lengths` heeft een lengte voor elke kolom, dus delen door alleen `lengths` dekt maar één van de twee kolommen. Waar in `lengths` staat de lengte van kolom `j`?",
            "`return together / (___ * lengths[___])`.",
        ],
        why="Voor twee kolommen van nullen en enen telt het inproduct de klanten die beide kochten, en elke lengte is de wortel van een aantal kopers. Door beide te delen lijkt een product dat iedereen koopt niet meer op het ene product dan een product dat alleen dezelfde klanten kopen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De scores voor één klant

        `cf_scores(R, bought)` moet de score van elk product voor één klant geven: zijn similarities met elk product in `bought`, de kolommen van wat de klant kocht, opgeteld. `similar_to` staat al in het vak, de versie van de course. Vervang de `...`: wat `scores` wordt in elke ronde van de loop. De check roept `cf_scores(R, bought)` aan met de vier producten die klant 13960 vóór de laatste bestelling kocht, en probeert drie kleine matrices. Om aan te raden neem je de hoogste scores die de klant nog niet kocht, met `recommend` uit de vorige les.
        """),
        L.task("Laat `cf_scores` de similarities met elk gekocht product optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("cf-scores", lines=14)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("cf-scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "cf-scores",
        [
            "In elke ronde telt één gekocht product zijn similarities op bij de scores tot dan toe, zoals in de vorige les de rijen van de producten tot een profiel werden opgeteld.",
            "`similar_to(R, j)` geeft een array met een getal voor elk product, even lang als `scores`. Twee arrays van dezelfde lengte tel je waarde voor waarde op.",
            "`scores = scores + ___(R, ___)`.",
        ],
        why="Een product dat op meerdere producten van de klant lijkt, krijgt van elk een beetje. Voor klant 13960 staat de WHITE HANGING HEART T-LIGHT HOLDER bovenaan: hij lijkt redelijk op alle vier producten van deze klant, zonder op één ervan het meest te lijken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_new():
    mo.md(r"""
    ### Oefening 3. Een eerste bestelling
    """)
    return


@app.cell(hide_code=True)
def choice_new_cell():
    choice_new = L.choice(
        "new-customer",
        question="Een klant plaatst vandaag de allereerste bestelling bij de winkel. Welke van de drie recommenders kan tien producten voor deze klant kiezen?",
        options={
            "right": "Alleen de popularity baseline",
            "collaborative": "Alleen collaborative filtering",
            "content": "Content-based filtering en collaborative filtering",
            "all": "Alle drie",
        },
        correct="right",
        why={
            "collaborative": "Collaborative filtering telt de similarities op met wat de klant kocht, en die kocht nog niets: elke score is 0. Zie *Cold start* in *Het idee*.",
            "content": "Allebei hebben ze nodig wat de klant kocht: content-based filtering voor het profiel, collaborative filtering voor de similarities om op te tellen. Een nieuwe klant heeft geen van beide. Zie *Cold start* in *Het idee*.",
            "all": "Content-based filtering en collaborative filtering beginnen allebei bij wat de klant kocht, en een nieuwe klant kocht nog niets. Zie *Cold start* in *Het idee*.",
            "right": "Populariteit heeft niets van de klant nodig: iedereen krijgt de populairste producten. Daarom blijft het nuttig naast de andere twee, hier voor 1.505 van de 4.334 klanten, die maar één bestelling plaatsten.",
        },
    )
    choice_new
    return (choice_new,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De user-item matrix en de hit rate komen uit [Wat laat je hierna zien: het aanbevelingsprobleem](/course/open/ml-recommend-problem?lang=nl), de cosine similarity en het optellen uit [Meer van wat je kocht: content-based filtering](/course/open/ml-content-based?lang=nl), nu met de kolommen van de matrix in plaats van de woorden van de namen, en `R.T` uit [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl). Collaborative filtering vindt hier de meeste klanten, 32,0%, maar weet niets van een nieuwe klant of een nieuw product; populariteit en de namen vullen dat gat. Wie wat kocht, is ook een netwerk: klanten en producten als nodes, elke aankoop een edge. In de volgende les, *Netwerken als data: graden en hubs*, kijk je naar netwerken zelf.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new, result_scores, result_similar):
    L.wrap_up(result_similar, result_scores, choice_new)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
