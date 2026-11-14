# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-content-based. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-content-based", "nl")
    C = L.checks   # de woorden van de productnamen, de profielen, beide recommenders voor elke klant, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl) en één over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Twee teksten, als tellingen van drie woorden: `a = [2, 0, 1]` en `b = [4, 0, 2]`. `b` gebruikt dezelfde woorden in dezelfde verhouding, twee keer zo vaak. Wat is hun cosine similarity?",
        options={
            "right": "1",
            "half": "0,5, omdat `b` twee keer zo lang is",
            "two": "2",
        },
        correct="right",
        why={
            "half": "De cosine similarity deelt door beide lengtes, dus de lengte valt weg: alleen de richting telt, en `b` wijst dezelfde kant op als `a`. Zie *Het idee* in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "two": "De cosine similarity is nooit meer dan 1: het inproduct wordt gedeeld door beide lengtes. Zie *Het idee* in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "right": "Dezelfde richting, dus 1, wat de lengtes ook zijn. Vandaag zijn de teksten de namen van producten.",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_argsort():
    quick_argsort = L.choice(
        "quick-argsort",
        question="Wat geeft `np.argsort(np.array([0.3, 0.9, 0.1]), kind=\"stable\")`?",
        options={
            "right": "`[2, 0, 1]`",
            "values": "`[0.1, 0.3, 0.9]`",
            "largest": "`[1, 0, 2]`",
        },
        correct="right",
        why={
            "values": "`np.argsort` geeft posities, geen waarden: waar de kleinste waarde staat, dan de volgende. Zie de tabel met nieuwe woorden in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "largest": "`np.argsort` gaat van de kleinste waarde omhoog, niet van de grootste omlaag. Zie de tabel met nieuwe woorden in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "De kleinste waarde, 0,1, staat op positie 2, dan 0,3 op 0 en 0,9 op 1. Vandaag heb je de grootste eerst nodig.",
        },
    )
    quick_argsort
    return (quick_argsort,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vóór de laatste bestelling kocht klant 13960 van de webwinkel uit de vorige les vier producten: een kleine en een grote HEART OF WICKER, een HANGING METAL HEART LANTERN en een deurmat. De popularity baseline geeft deze klant de tien producten die hij bijna iedereen geeft, en daarvan zit er geen in de laatste bestelling. Een verkoper die die vier producten zag, zou meer hartjes laten zien. Daarvoor moet een recommender weten wat een product is, en niet alleen hoe vaak het gekocht werd.

    **Een voorbeeld uit het dagelijks leven.** Een vriend die weet dat je detectives leest, raadt je een andere detective aan, ook een die gisteren uitkwam en die nog niemand gelezen heeft.
    """)
    return


@app.cell(hide_code=True)
def predict_content():
    guess_content = L.guess(
        "content-guess",
        question="Geef elke klant de tien producten waarvan de namen het meest lijken op de namen van wat de klant kocht. Voor hoeveel van de 2.498 klanten zit er dan minstens één van de tien in de laatste bestelling, vergeleken met de tien populairste producten?",
        options={
            "fewer": "Voor veel minder klanten",
            "right": "Voor ongeveer evenveel, maar niet dezelfde klanten",
            "double": "Voor twee keer zoveel",
        },
        reveal="**Ongeveer evenveel, maar niet dezelfde klanten.** De namen vinden iets voor 595 klanten, 23,8%; populariteit voor 541, 21,7%. Maar slechts 188 klanten krijgen een hit van allebei. De namen vinden 407 klanten die populariteit mist, en populariteit 353 die de namen missen.",
    )
    guess_content
    return (guess_content,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Item features.** Beschrijf elk item met wat het is: zijn **item features**. Hier zijn dat de woorden van de productnamen. De 3.576 producten die vóór een laatste bestelling gekocht werden, hebben samen 2.163 verschillende woorden in hun namen, gemiddeld 4,36 verschillende woorden per naam. Geef elk woord een kolom en elk product een rij, met een 1 waar het woord in de naam staat: een matrix van 3.576 rijen en 2.163 kolommen, zoals de woordtellingen in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl). De rij van HEART OF WICKER SMALL heeft een 1 in de kolommen van HEART, OF, WICKER en SMALL, en een 0 in de 2.159 andere.

        **Het gebruikersprofiel.** Tel de rijen op van alles wat een klant kocht: het **gebruikersprofiel**, een vector met voor elk woord in hoeveel van de producten van die klant het in de naam staat. De vier producten van klant 13960 hebben samen 18 woorden, waarvan 14 verschillende: HEART drie keer, OF en WICKER twee keer, en de andere elf één keer.

        **Het dichtst bij het profiel.** Geef elk product een score: de cosine similarity van zijn rij met het profiel. Raad de `k` producten aan met de hoogste scores die de klant nog niet kocht. Veel producten krijgen dezelfde score: hun namen zijn even lang, en delen evenveel met het profiel. Gelijke scores houden de volgorde van de producten, de populairste eerst. Dat is **content-based filtering**: de producten die het dichtst liggen bij wat een klant kocht, gemeten aan wat de producten zijn.

        Waarom de cosine similarity, en niet alleen het inproduct? Het inproduct van een rij en het profiel telt de woorden die de naam met het profiel deelt, elk zo vaak als het in het profiel staat. SET OF 3 HEART COOKIE CUTTERS deelt OF en HEART met het profiel van klant 13960, en FINE WICKER HEART deelt WICKER en HEART: allebei krijgen ze een inproduct van 5. Maar de koekjesvormpjes hebben zes woorden en het hartje drie. De cosine similarity deelt door beide lengtes, dus telt ze welk deel van de naam overeenkomt: 0,386 voor de koekjesvormpjes, 0,546 voor het rieten hartje.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **item features** | wat een recommender over een item zelf weet, zoals de woorden van zijn naam | een rij van `features` |
        | **gebruikersprofiel** | de item features van alles wat een gebruiker had, bij elkaar opgeteld | `profile = profile + features[row]` |
        | **content-based filtering** | de items aanraden waarvan de features het dichtst bij het profiel van een gebruiker liggen | |
        | `text.split()` | knipt een tekst bij de spaties in zijn woorden | `"HEART OF WICKER".split()` geeft `['HEART', 'OF', 'WICKER']` |
        """),
        L.watch_out("Content-based filtering kent alleen wat een klant kocht, dus raadt het meer van hetzelfde aan. Van de negen producten die klant 14905 kocht, zijn er twee houders voor T-lights. Het tiental van deze klant bestaat helemaal uit houders, negen ervan voor T-lights, en er zit er geen in de laatste bestelling; de tien populairste producten vinden er twee. Kijk naar de klanten die een recommender mist, niet alleen naar zijn hit rate."),
        mo.md(r"""
        De slider gaat vijf klanten langs. Links de woorden die het vaakst in hun profiel staan; rechts de tien producten van elke recommender, een vierkantje elk: gevuld als het product in de laatste bestelling zit, open als het er niet in zit. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_customer_cell():
    def _frame(i):
        customer = C.CUSTOMERS[i - 1]
        said = C.said(customer, L.lang)
        alt = ("Twee grafieken. Links: staven voor de woorden die het vaakst in het profiel van de klant staan, de langste "
               "bovenaan. Rechts: twee rijen van tien vierkantjes, content-based filtering boven en populariteit onder, "
               "gevuld als het product in de laatste bestelling van de klant zit. " + said)
        return (C.picture(customer, {"words": "Woorden van het profiel", "times": "producten met het woord",
                                     "ten": "De tien aanbevolen", "place": "plek in de lijst", "content": "content-based",
                                     "popular": "populariteit", "alt": alt}), said)

    explore_customer = L.explore("customer-slider", "klant", _frame, start=1, values=[1, 2, 3, 4, 5], role="",
                                 task="Ga de vijf klanten langs. Bij welke vinden de namen meer dan populariteit, en bij welke minder?")
    explore_customer
    return


@app.cell(hide_code=True)
def text_after_slider():
    mo.md(r"""
    Bij klanten die steeds één soort product kopen, werken de namen. Klant 13242 kocht vijf jumbo bags, en van de tien jumbo bags die deze klant krijgt, zitten er zes in de laatste bestelling; klant 16356 kocht metalen bordjes, en drie van de tien zitten in de laatste bestelling. Populariteit vindt voor geen van beiden iets. Bij klant 14905 en klant 14875 is het andersom: wat ze daarna kochten is populair, maar niet waar hun profiel naartoe wijst. Over alle 2.498 testklanten vinden de twee ongeveer evenveel, 595 tegen 541, maar andere klanten: minstens één van de twee vindt iets voor 948 klanten, 38,0%.

    ### Recept

    Zo raad je aan met content-based filtering:

    1. Geef elk item een vector van zijn features: hier een kolom voor elk woord van de namen, met een 1 waar het woord in de naam staat.
    2. Tel voor elke gebruiker de rijen op van de items die de gebruiker had: het profiel.
    3. Geef elk item als score de cosine similarity van zijn rij en het profiel.
    4. Raad de `k` items aan met de hoogste scores die de gebruiker nog niet had; gelijke scores in de volgorde van populariteit.
    5. Meet de hit rate op de verborgen bestellingen, en vergelijk met de popularity baseline op dezelfde gebruikers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel volgt stap 1 en 2. Ze splitst de bestellingen op tijd zoals in de vorige les, en zet de producten op een rij, de populairste eerst. Daarna geeft ze elk woord van hun namen een kolom: `_words[_word]` is het nummer van de kolom van een woord, en `_row_of[_product]` het nummer van de rij van een product. `np.zeros` maakt de matrix, en er komt een 1 waar een woord in een naam staat. De klantnummers zijn floats, zoals `13960.0`, net als in de vorige les. Tot slot telt ze de rijen op van wat klant 13960 vóór de laatste bestelling kocht, en print ze die producten, en de woorden die in meer dan één ervan staan. Voer de cel uit.
    """)
    return


@app.cell
def example_profile():
    import numpy as np
    import pandas as pd

    # wat elke klant vóór de laatste bestelling kocht, en de producten, de populairste eerst
    _rows = pd.read_csv("data/online-retail-baskets.csv.gz").dropna(subset=["customer"])
    _rows["last"] = _rows.groupby("customer")["invoice"].transform("max")
    _train = _rows[_rows["invoice"] < _rows["last"]]
    _seen = _train.groupby("customer")["product"].agg(set)
    _names = list(_train.drop_duplicates(subset=["customer", "product"])["product"].value_counts().index)

    # 1. een kolom voor elk woord van de namen, een rij voor elk product, en een 1 waar het woord in de naam staat
    _words = {}
    _row_of = {}
    for _i in range(len(_names)):
        _row_of[_names[_i]] = _i
        for _word in _names[_i].split():
            if _word not in _words:
                _words[_word] = len(_words)
    _features = np.zeros((len(_names), len(_words)))
    for _i in range(len(_names)):
        for _word in _names[_i].split():
            _features[_i, _words[_word]] = 1
    print(_features.shape, round(_features.sum() / len(_names), 2))

    # 2. het profiel van klant 13960: de rijen van wat de klant kocht, opgeteld
    _profile = np.zeros(len(_words))
    for _product in sorted(_seen[13960.0]):
        print(_product)
        _profile = _profile + _features[_row_of[_product]]
    for _word in _words:
        if _profile[_words[_word]] > 1:
            print(_word, _profile[_words[_word]])
    print(_profile.sum(), round(np.linalg.norm(_profile), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    3.576 rijen en 2.163 kolommen, met gemiddeld 4,36 enen in een rij: het grootste deel van de matrix is 0, net als de user-item matrix. Klant 13960 kocht de vier producten hierboven. HEART staat in drie ervan, OF en WICKER in twee; het profiel telt op tot 18, en zijn lengte is 5,29. De deurmat brengt zes woorden mee die geen enkel hartje heeft: DOORMAT, KEEP, CALM, AND, COME en IN.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elk product tegen het profiel

        `cosine_scores(features, profile)` moet voor elke rij van `features` de cosine similarity met `profile` geven: een array met een score voor elk product. In `dots` staat al het inproduct van elke rij met het profiel, en in `lengths` de lengte van elke rij, zoals `np.linalg.norm(..., axis=1)` de afstanden gaf in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl). Vervang de `...`. De check roept `cosine_scores(features, profile)` aan met de features van het uitgewerkte voorbeeld en het profiel van klant 13960, en probeert drie kleine arrays.
        """),
        L.task("Laat `cosine_scores` de cosine similarity van elke rij met het profiel geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("cosine-scores", lines=7)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("cosine-scores", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "cosine-scores",
        [
            "De cosine similarity deelt het inproduct door twee lengtes. Welke twee?",
            "`dots` en `lengths` hebben een waarde voor elk product; de lengte van het profiel is één getal. Arrays van dezelfde lengte delen waarde voor waarde, en een getal deelt elke waarde.",
            "`return dots / (___ * np.linalg.norm(___))`.",
        ],
        why="Door de lengte van elke rij telt een lange naam niet zwaarder dan een korte met dezelfde overeenkomende woorden; delen door de lengte van het profiel verandert voor één klant geen enkele volgorde, maar houdt elke score tussen 0 en 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_recommend():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De tien dichtstbijzijnde, nog niet gekocht

        `recommend(scores, names, seen, k)` moet de `k` producten geven met de hoogste scores die niet in `seen` zitten, de hoogste eerst; gelijke scores houden de volgorde van `names`, de populairste eerst. `scores[i]` is de score van `names[i]`. Vervang de twee `...`: de posities die je langsgaat, van de hoogste score omlaag, en wanneer een product in `top` komt. De check roept `recommend(scores, names, seen, 10)` aan met de scores van klant 13960 en de producten die deze klant vóór de laatste bestelling kocht, en probeert drie kleine voorbeelden.
        """),
        L.task("Laat `recommend` de k producten geven met de hoogste scores die een klant nog niet kocht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recommend():
    box_recommend = L.code_box("recommend", lines=9)
    box_recommend
    return (box_recommend,)


@app.cell(hide_code=True)
def check_recommend(box_recommend):
    result_recommend = L.check("recommend", box_recommend.value)
    result_recommend
    return (result_recommend,)


@app.cell(hide_code=True)
def hints_recommend():
    L.hints(
        "recommend",
        [
            "`np.argsort` geeft de posities van de kleinste waarde omhoog, en je wilt de hoogste score eerst. Een product komt in `top` onder dezelfde twee voorwaarden als in `popular` uit de vorige les.",
            "Een minteken draait de volgorde om: de hoogste score wordt de kleinste. Houd gelijke scores in hun volgorde, zoals in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "`for i in np.argsort(___, kind=\"stable\"):` en `if names[i] ___ seen and len(top) ___ k:`.",
        ],
        why="De twee producten met de hoogste scores delen vier woorden met het profiel: HEART, OF, WICKER, en SMALL of LARGE. De volgende vier delen er drie. Gelijke scores blijven in de volgorde van `names`, dus van twee gelijke producten komt het populairste eerst.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_new():
    mo.md(r"""
    ### Oefening 3. Een nieuw product
    """)
    return


@app.cell(hide_code=True)
def choice_new_cell():
    choice_new = L.choice(
        "new-product",
        question="De winkel neemt een nieuw product op, GREEN HANGING WICKER HEART, dat nog niemand gekocht heeft. Welke recommender kan het aan klant 13960 laten zien?",
        options={
            "right": "Alleen content-based filtering",
            "popular": "Alleen de popularity baseline",
            "both": "Allebei",
            "neither": "Geen van beide: niemand heeft het gekocht",
        },
        correct="right",
        why={
            "popular": "De popularity baseline telt hoeveel klanten een product eerder kochten, en voor een nieuw product is dat 0: het komt als laatste. Zie *De popularity baseline* in [Wat laat je hierna zien: het aanbevelingsprobleem](/course/open/ml-recommend-problem?lang=nl).",
            "both": "Content-based filtering kan het, maar de popularity baseline telt klanten, en 0 klanten kochten het nieuwe product: het komt als laatste. Zie *De popularity baseline* in [Wat laat je hierna zien: het aanbevelingsprobleem](/course/open/ml-recommend-problem?lang=nl).",
            "neither": "Content-based filtering heeft geen kopers nodig: het vergelijkt de woorden van de naam met het profiel. Zie *Item features* in *Het idee*.",
            "right": "Meer dan de naam heeft content-based filtering niet nodig: GREEN HANGING WICKER HEART scoort 0,567 tegen het profiel van klant 13960, even hoog als de producten op de 3e tot en met 6e plek van de lijst van deze klant. Populariteit telt kopers, en die heeft het niet.",
        },
    )
    choice_new
    return (choice_new,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De score is de cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), hier tussen een naam en een profiel, en de producten die het dichtst bij het profiel liggen vind je met de `np.argsort` uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl). De hit rate en de popularity baseline uit [Wat laat je hierna zien: het aanbevelingsprobleem](/course/open/ml-recommend-problem?lang=nl) laten zien dat de namen ongeveer evenveel klanten vinden, maar andere. Content-based filtering moet weten wat een product is. In de volgende les, *Klanten zoals jij: collaborative filtering*, kijkt de recommender helemaal niet naar de namen, alleen naar wie wat kocht.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new, result_cosine, result_recommend):
    L.wrap_up(result_cosine, result_recommend, choice_new)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
