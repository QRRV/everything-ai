# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-image-search. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-image-search", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl) en over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Wat geeft `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` voor `a = np.array([1, 2])` en `b = np.array([2, 4])`?",
        options={
            "right": "1,0",
            "dot": "10,0",
            "ratio": "2,0",
            "half": "0,5",
        },
        correct="right",
        why={
            "right": "`b` is `a` keer 2: dezelfde richting, dus de cosine similarity is 1, wat de lengtes ook zijn. Vandaag vergelijkt hij foto's.",
            "dot": "10 is het inproduct, 1 · 2 + 2 · 4. De cosine similarity deelt het door de twee normen, √5 en √20, en hun product is 10. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "ratio": "`b` is twee keer zo lang als `a`, maar de cosine similarity kijkt alleen naar de richting, en die is gelijk. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "half": "De lengtes vallen weg uit een cosine similarity, en `a` en `b` wijzen dezelfde kant op. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
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
            "ranks": "`[1, 2, 0]`",
        },
        correct="right",
        why={
            "right": "De posities van klein naar groot: 0,1 staat op positie 2, dan 0,3 op 0 en 0,9 op 1. Vandaag heb je de grootste eerst nodig.",
            "values": "Dat zijn de waarden zelf, gesorteerd, zoals `sorted` ze geeft. `np.argsort` geeft hun posities. Zie [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "largest": "Dat is van groot naar klein. `np.argsort` begint bij de kleinste, zoals voor de k dichtstbijzijnde bij k-NN. Zie [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "ranks": "Dat is de plek die elke waarde in de gesorteerde volgorde krijgt. `np.argsort` geeft de posities in de gesorteerde volgorde: eerst de positie van de kleinste. Zie [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
        },
    )
    quick_argsort
    return (quick_argsort,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt één foto, en je wilt de foto's die erop lijken uit een grote verzameling waarin niemand heeft opgeschreven wat er op elke foto staat: je eigen foto's, de catalogus van een winkel, een archief waarin dezelfde foto misschien twee keer zit. Een classifier helpt niet: die kent alleen de klassen die hij leerde, en iemand moet er foto's voor labelen. Wat wel helpt, is de foto vergelijken met elke foto in de verzameling, en de foto's houden die er het meest op lijken.

    **Een voorbeeld uit het dagelijks leven.** Iemand laat je een foto zien van een paddenstoel uit het bos, en jij bladert in een gids naar de plaatjes die er het meest op lijken. Je vergelijkt hoe ze eruitzien, niet hun namen; de namen komen daarna, van de plaatjes die je vond.
    """)
    return


@app.cell(hide_code=True)
def predict_five():
    guess_five = L.guess(
        "five",
        question="Voor elk van de 300 foto's uit de vorige les zoek je de 5 foto's die er het meest op lijken tussen de 299 andere, op de cosine similarity van hun features van DINOv2. Hoeveel van die 5 laten gemiddeld hetzelfde soort voorwerp zien?",
        options={
            "one": "Ongeveer 1",
            "two": "Ongeveer 2",
            "three": "Ongeveer 3,5",
            "almost": "Bijna 5",
        },
        reveal=f"**Bijna 5: gemiddeld {L.number(C.REFERENCE['of5']['features'], 2)},** zonder één label. Voor 8 van de 10 klassen is het 5 of bijna; lotus komt op {L.number(C.REFERENCE['class5']['features'][3], 2)} en waterlelie op {L.number(C.REFERENCE['class5']['features'][8], 2)}, omdat ze elkaar vinden. Hetzelfde zoeken op de pixels vindt er {L.number(C.REFERENCE['of5']['pixels'], 2)}.",
    )
    guess_five
    return (guess_five,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Image retrieval** is de foto's in een verzameling vinden die het meest op een gegeven foto lijken. Met een vision foundation model gaat dat in vier stappen. Reken de features van elke foto in de verzameling één keer uit: een matrix `F` met een rij van 384 getallen per foto. Reken de features `q` uit van de foto waarmee je zoekt. Bereken de cosine similarity van `q` met elke rij van `F`. En houd de rijen met de grootste. Niemand hoeft iets te labelen: de course gebruikt de klassen alleen om te tellen hoe goed het zoeken gaat. Op de features van DINOv2 hebben twee foto's van dezelfde klasse gemiddeld een cosine similarity van 0,60, twee foto's van verschillende klassen 0,05.

        **Alle rijen tegelijk.** `F @ q` is het inproduct van `q` met elke rij van `F`, zoals in [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl): één getal per foto. `np.linalg.norm(F, axis=1)` geeft de norm van elke rij, zoals het in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) de afstanden gaf. Deel het ene door het andere en door de norm van `q`, en je hebt de cosine similarity van `q` met elke foto uit de verzameling, in één regel:

        `sims = (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`

        **Meest gelijkend eerst.** `np.argsort` sorteert van klein naar groot, en voor k-NN was dat goed: een kleine afstand betekent dichtbij. Bij een cosine similarity is het andersom: groot betekent dat ze op elkaar lijken. Dus sorteer je de similarities met een minteken ervoor, `np.argsort(-sims, kind="stable")`, en houd je de eerste k posities. Image retrieval is de eerste helft van k-NN: k-NN laat de k gevonden foto's stemmen over een label, en image retrieval laat ze zien.
        """),
        L.watch_out("Zit de foto waarmee je zoekt zelf in de verzameling, dan komt hij als eerste, met een cosine similarity van 1: niets lijkt meer op een foto dan de foto zelf. Sla hem over."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **image retrieval** | de foto's in een verzameling vinden die het meest op een gegeven foto lijken, meestal door hun features te vergelijken | `np.argsort(-sims, kind="stable")[:k]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_search():
    mo.vstack([
        mo.Html(C.search_picture({"title": "300 foto's doorzoeken op de meest gelijkende", "collection": "verzameling: 300 foto's",
                                  "query": "de foto waarmee je zoekt", "dino": "DINOv2-small", "once": "één keer per foto",
                                  "F": "F: 300 rijen van 384", "q": "q: 384 features",
                                  "sims": "(F @ q) / (normen van de rijen · norm van q)", "sims_note": "300 cosine similarities, één per foto",
                                  "top": "np.argsort(-sims, kind=\"stable\")[:5]", "top_note": "de posities van de 5 meest gelijkende foto's",
                                  "alt": "Links gaan de 300 foto's van de verzameling elk één keer door DINOv2-small, naar F, een matrix van 300 rijen van 384 features. Rechts gaat de foto waarmee je zoekt door DINOv2-small naar q, 384 features. Samen geven ze 300 cosine similarities, één per foto, en np.argsort geeft de posities van de 5 meest gelijkende foto's."})),
        mo.md(r"""
        De linkerkolom draait één keer, hoe vaak je ook zoekt; de rechterkolom draait bij elke zoekopdracht, en dan is het één matrix keer een vector en één keer sorteren.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Elk van de 300 foto's doorzoekt de 299 andere. Met de slider kies je hoeveel van de meest gelijkende foto's elke zoekopdracht houdt, van 1 tot 29: elke klasse heeft 29 andere foto's. Je ziet welk percentage daarvan van dezelfde klasse is, op de features van DINOv2 en op de pixels van de klaargemaakte foto's.
    """)
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _say(k):
        _k = int(k)
        _f, _p = C.percent(C.REFERENCE["features"][_k], "nl"), C.percent(C.REFERENCE["pixels"][_k], "nl")
        if _k == 1:
            return f"De meest gelijkende foto is in {_f} procent van de zoekopdrachten van dezelfde klasse op de features van DINOv2, en in {_p} procent op de pixels."
        return f"Van de {_k} meest gelijkende foto's is gemiddeld {_f} procent van dezelfde klasse op de features van DINOv2, en {_p} procent op de pixels."

    def _frame(k):
        _html = C.curve(int(k), {"title": "Van dezelfde klasse, in procenten", "x": "gehouden foto's, meest gelijkende eerst", "y": "procent",
                                 "features": "features van DINOv2", "pixels": "pixels", "chance": "willekeurig gekozen",
                                 "alt": "Twee lijnen over het aantal gehouden foto's, meest gelijkende eerst, 1 tot 29: op de features van DINOv2 is 96 tot 91 procent daarvan van dezelfde klasse, op de pixels 39 tot 19 procent; willekeurig gekozen foto's zouden ongeveer 10 geven."}, "nl")
        return _html, _say(k)

    explore_k = L.explore("k", "meest gelijkende foto's", _frame, start=1, values=C.K_VALUES, role="",
                          task="Zoek het grootste aantal foto's dat je kunt houden terwijl op de features van DINOv2 minstens 95 procent ervan van dezelfde klasse is.")
    explore_k
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Op de features van DINOv2 kun je tot 13 meest gelijkende foto's houden met nog minstens 95 procent ervan van dezelfde klasse: precies 95,0 bij 13, en 94,9 bij 14. Zelfs met 29, alle andere foto's die een klasse heeft, is nog 90,9 procent van dezelfde klasse: het zoeken vindt bijna alle foto's van een klasse, zonder één label. Op de pixels is de meest gelijkende foto in 38,7 procent van de zoekopdrachten van dezelfde klasse, en van 29 foto's is dat 18,8 procent: beter dan willekeurig gekozen foto's, ongeveer 10, maar ver onder de features.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een verzameling foto's te doorzoeken op de foto's die het meest op een gegeven foto lijken:

    1. Reken de features van elke foto in de verzameling één keer uit, met een vision foundation model: een rij per foto in `F`, met shape `(n, 384)`.
    2. Reken op dezelfde manier de features `q` uit van de foto waarmee je zoekt.
    3. Bereken de cosine similarity van `q` met elke rij: `sims = (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`.
    4. Neem de posities van de k grootste, de grootste eerst: `np.argsort(-sims, kind="stable")[:k]`.
    5. Zoek de foto's op die op die posities staan. Zit de foto waarmee je zoekt in de verzameling, sla hem dan over.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel zoekt met de eerste lotus, foto 90 van de 300. `C.collection(dino)` is stap 1: de features van de 300 foto's, als NumPy-array `F` met een rij per foto in de volgorde van `C.CLASSES`, met hun labels, de plek van de klasse van elke foto in `C.CLASSES`, en hun bestanden. Hij rekent ze één keer uit, zoals `features(photos)` in de vorige les deed, en dat duurt ongeveer een kwart minuut, op een trage computer langer. De cel print de shape van `F`, de posities van de 6 meest gelijkende foto's en hun cosine similarities, en laat die foto's daarna zien met hun klasse. `dino` is DINOv2-small, dat de cel hieronder laadt zodra beide downloads klaar zijn.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="DINOv2-small en Caltech-101 downloaden, 225 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dino_cell(get_button):
    dino = C.dino_or_stop(get_button, {
        "ask": "*DINOv2-small en de foto's van Caltech-101 staan nog niet op je computer. De knop downloadt ze één keer: DINOv2-small van Hugging Face, 88 MB, en Caltech-101 van Caltech, 137 MB, waarvan hij daarna de foto's uitpakt. Het uitgewerkte voorbeeld en de demo hebben ze nodig; de oefeningen niet. Tijdens het downloaden wacht de hele les: doe bij een trage verbinding eerst de oefeningen.*",
        "busy": "**{name} downloaden** · {done} van {total} MB",
        "unpacking": "**De foto's van Caltech-101 uitpakken** · {done} foto's",
        "no_net": "*Het downloaden is niet gelukt: Hugging Face of Caltech was niet bereikbaar, of de verbinding brak af. Controleer je verbinding, en klik nog eens.*",
        "not_pinned": "*Het downloaden is niet gelukt: een bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik nog eens; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden is niet gelukt: er is geen ruimte op je schijf voor 225 MB en de uitgepakte foto's. Maak ruimte, en klik nog eens.*",
        "failed": "*Het downloaden is niet gelukt ({why}). Klik nog eens.*",
        "here": "*DINOv2-small en de foto's staan op je computer: 300 foto's om te doorzoeken, 30 van elk van 10 klassen.*",
    })
    return (dino,)


@app.cell
def example_search(dino):
    import numpy as np

    # 1. de features van de 300 foto's, één keer: een rij van F per foto
    F, labels, photos = C.collection(dino)
    print(F.shape)
    # 2. de foto waarmee je zoekt: de eerste lotus, foto 90
    q = F[90]
    # 3. zijn cosine similarity met elke foto
    sims = (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))
    # 4. de posities van meest naar minst gelijkend
    order = np.argsort(-sims, kind="stable")
    print(order[:6])
    print(sims[order[:6]])
    # 5. de foto zelf en de 5 foto's die er het meest op lijken, met hun klasse
    C.results(photos, labels, sims, order[:6], "nl")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `F` heeft een rij van 384 features voor elk van de 300 foto's. De eerste positie is 90: de lotus zelf, met een cosine similarity van 1, hier geprint als `0.9999999`, op jouw computer misschien als `1.` of `1.0000001`, omdat de computer een beetje afrondt. Dan komen drie lotussen, met 0,776, 0,776 en 0,774, een waterlelie met 0,749, en nog een lotus met 0,742. Alle 300 similarities kostten één regel, en het sorteren nog één.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sims():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Alle rijen tegelijk
    """)
    return


@app.cell(hide_code=True)
def choice_sims_cell():
    choice_sims = L.choice(
        "sims",
        question="`F` bevat de features van 1.000 foto's, een rij van 768 getallen per foto, en `q` de 768 features van één foto. Wat is `(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))`?",
        options={
            "one": "Eén getal: de cosine similarity van `q` met alle foto's samen",
            "right": "1.000 getallen: de cosine similarity van `q` met elke foto",
            "features": "768 getallen: één voor elke feature",
            "distance": "1.000 getallen: de afstand van `q` tot elke foto",
        },
        correct="right",
        why={
            "one": "`F @ q` geeft een getal voor elke rij van `F`: 1.000 inproducten. Gedeeld door de 1.000 normen van de rijen en de norm van `q` worden het 1.000 cosine similarities, één per foto.",
            "right": "`F @ q` is het inproduct van `q` met elke rij, `np.linalg.norm(F, axis=1)` de norm van elke rij: delen geeft de cosine similarity van `q` met elke foto, 1.000 getallen, hoeveel features een rij ook heeft.",
            "features": "De 768 features verdwijnen in het inproduct: elke rij van `F` en `q` geven samen één getal. Het zijn er dus 1.000, één per foto, elk een cosine similarity.",
            "distance": "Een afstand is de norm van een verschil, `np.linalg.norm(F - q, axis=1)`. Dit is het inproduct gedeeld door de normen: de cosine similarity van `q` met elke foto, die alleen naar de richting kijkt.",
        },
    )
    choice_sims
    return (choice_sims,)


@app.cell(hide_code=True)
def text_exercise_search():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een verzameling doorzoeken

        Maak `search(F, q, k)` af: stap 3 en 4 van het recept. `F` heeft een rij features per foto, `q` de features van de foto waarmee je zoekt, en `k` is hoeveel posities de functie teruggeeft, de meest gelijkende eerst. Vervang de twee `...`. De check verzint een verzameling van 6 rijen van 3 features, van heel verschillende lengte, roept `search(F, q, 3)` aan, en kijkt naar de posities die hij teruggeeft. Hij probeert ook andere verzamelingen en andere `k`; daarin is `q` nooit een van de rijen.
        """),
        L.task("Laat `search(F, q, k)` de posities geven van de `k` rijen van `F` die het meest op `q` lijken, de meest gelijkende eerst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    search_code = L.code_box("search", lines=8)
    search_code
    return (search_code,)


@app.cell(hide_code=True)
def check_search(search_code):
    result_search = L.check("search", search_code.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Stap 3 en 4 van het uitgewerkte voorbeeld, met `k` in plaats van 6.",
            "`F @ q` geeft het inproduct van `q` met elke rij; deel het door de normen van de rijen en van `q`. Sorteer de similarities daarna zo dat de grootste vooraan komen, en houd de eerste `k`.",
            "`sims = (F @ q) / (np.linalg.norm(F, axis=___) * np.linalg.norm(q))` en `return np.argsort(___sims, kind=\"stable\")[:k]`.",
        ],
        why="`(F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q))` is de cosine similarity van `q` met elke rij tegelijk. `np.argsort(-sims, kind=\"stable\")` geeft de posities van de grootste similarity naar de kleinste, en `[:k]` houdt de eerste `k`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw zoekfunctie op de foto's

    Als DINOv2 en de foto's gedownload zijn (de knop staat in het uitgewerkte voorbeeld), laat de cel hieronder jouw `search` los op de 300 foto's als hij klopt, en anders die van de course. Elke foto zoekt de 6 rijen die er het meest op lijken; de 5 na de foto zelf tellen. Per klasse zie je hoeveel van die 5 gemiddeld van dezelfde klasse zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_search(dino, search_code, result_search):
    _d = C.demo(dino, search_code.value if result_search.state == "correct" else None)
    _who = "jouw `search`" if _d["yours"] else "de `search` van de course"
    _rows = "\n".join(f"| {C.NAMES['nl'][_c]} | {L.number(_d['class5'][_i], 2)} | {L.number(C.REFERENCE['class5']['features'][_i], 2)} |"
                      for _i, _c in enumerate(C.CLASSES))
    mo.md(f"""Met {_who}, op de features van DINOv2:

| klasse | van de 5 meest gelijkende, van dezelfde klasse | gemeten door de course |
|---|---|---|
{_rows}""")
    return


@app.cell(hide_code=True)
def text_exercise_pixels():
    mo.md(r"""
    ### Oefening 3. Zoeken op de pixels
    """)
    return


@app.cell(hide_code=True)
def choice_pixels_cell():
    choice_pixels = L.choice(
        "pixels",
        question=f"Hetzelfde zoeken op de pixels van de klaargemaakte foto's vindt gemiddeld {L.number(C.REFERENCE['class5']['pixels'][7], 2)} zonnebloemen tussen de 5 foto's die het meest op een zonnebloem lijken, en maar {L.number(C.REFERENCE['class5']['pixels'][4], 1)} nautilussen bij een nautilus. Waarom doen zonnebloemen het het best?",
        options={
            "middle": "Zonnebloemen vullen het midden van de foto, en de pixels vergelijken wat in het midden ligt",
            "more": "Er zitten meer zonnebloemen tussen de 300 foto's",
            "right": "Foto's van zonnebloemen hebben veel geel gemeen, en pixels vergelijken kleuren plek voor plek",
            "shape": "De pixels beschrijven de vorm van de bloemblaadjes beter dan de features van DINOv2",
        },
        correct="right",
        why={
            "middle": f"Veel foto's hebben hun voorwerp in het midden, niet alleen zonnebloemen. Haal de kleur weg en houd de plekken: op grijze pixels, het gemiddelde van de drie kleurkanalen, vinden zonnebloemen er nog maar {L.number(C.REFERENCE['gray']['sunflower'], 1)} van de 5. Het is de kleur: gemiddeld is ongeveer {C.REFERENCE['yellow']['sunflower']} procent van de pixels van een foto van een zonnebloem sterk geel, telde de course, en ongeveer {C.REFERENCE['yellow']['others'][0]} tot {C.REFERENCE['yellow']['others'][1]} procent in elke andere klasse.",
            "more": f"Elke klasse heeft 30 foto's. Wat foto's van zonnebloemen gemeen hebben, is hun kleur: gemiddeld is ongeveer {C.REFERENCE['yellow']['sunflower']} procent van hun pixels sterk geel, telde de course, en ongeveer {C.REFERENCE['yellow']['others'][0]} tot {C.REFERENCE['yellow']['others'][1]} procent in elke andere klasse.",
            "right": f"Gemiddeld is ongeveer {C.REFERENCE['yellow']['sunflower']} procent van de pixels van een foto van een zonnebloem sterk geel, telde de course, en ongeveer {C.REFERENCE['yellow']['others'][0]} tot {C.REFERENCE['yellow']['others'][1]} procent in elke andere klasse. De cosine similarity van de pixels van twee foto's is groot als dezelfde kleuren op dezelfde plekken liggen, wat het voorwerp ook is: kleur is wat pixels rechtstreeks zeggen. Op grijze pixels, zonder de kleur, vinden zonnebloemen er nog maar {L.number(C.REFERENCE['gray']['sunflower'], 1)} van de 5. Een nautilus is wit, bruin of grijs, net als veel andere dingen. Op de features van DINOv2 vinden zonnebloemen er {L.number(C.REFERENCE['class5']['features'][7], 2)} van de 5.",
            "shape": f"Op de features van DINOv2 vinden zonnebloemen er {L.number(C.REFERENCE['class5']['features'][7], 2)} van de 5, op de pixels {L.number(C.REFERENCE['class5']['pixels'][7], 2)}. Een pixel zegt alleen welke kleur één plekje heeft; van bloemblaadjes weet hij niets. Wat foto's van zonnebloemen op de pixels gemeen hebben, is hun kleur: gemiddeld is ongeveer {C.REFERENCE['yellow']['sunflower']} procent van hun pixels sterk geel.",
        },
    )
    choice_pixels
    return (choice_pixels,)


@app.cell(hide_code=True)
def text_exercise_look_alike():
    mo.md(r"""
    ### Oefening 4. Een waterlelie bij een lotus
    """)
    return


@app.cell(hide_code=True)
def explain_look_alike_cell():
    explain_look_alike = L.explain(
        "look-alike",
        question="In het uitgewerkte voorbeeld zit een waterlelie tussen de 5 foto's die het meest op de lotus lijken, en gemiddeld vinden lotussen maar 3,7 lotussen tussen hun 5. Is dat een fout van het zoeken? Zeg waar het zoeken naar kijkt, en wat ervoor nodig is om lotussen en waterlelies uit elkaar te houden.",
        model_answer="**Het zoeken vindt de foto's waarvan de features het meest dezelfde kant op wijzen, en verder niets:** het kent geen klassen. **Een lotus en een waterlelie lijken echt op elkaar,** twee grote bloemen met puntige bloemblaadjes, vaak tussen ronde bladeren op het water, en hun features liggen dicht bij elkaar: een lotus en een waterlelie hebben gemiddeld een cosine similarity van 0,50, bijna net zoveel als twee lotussen, 0,52, tegen 0,05 voor twee foto's van verschillende klassen in het algemeen. Wil je foto's die op deze lijken, dan is de waterlelie een goed antwoord; alleen als je hetzelfde soort bloem wilt, is het een fout, en dat zeggen alleen de labels. Ook labels kunnen fout zijn: sommige foto's die in Caltech-101 lotus heten, lijken waterlelies te laten zien. **Om ze uit elkaar te houden, heb je gelabelde foto's nodig:** een probe die daaraan leert welke verschillen tellen, of features waarin die kleine verschillen duidelijker zitten. k-NN helpt weinig: dat leert niets, en het stemt met dezelfde buren die het zoeken vindt. De probe uit de vorige les haalde dezelfde twee klassen door elkaar, dus zelfs met labels blijft het lastig.",
        points=[
            "je zegt dat het zoeken alleen kijkt naar hoeveel de features op elkaar lijken, en geen klassen kent",
            "je zegt dat lotus en waterlelie op elkaar lijken, zodat een waterlelie een goed antwoord is op 'foto's die hierop lijken', en alleen een fout bij 'hetzelfde soort'",
            "je zegt dat je labels nodig hebt om ze uit elkaar te houden, zoals bij een probe die ervan leert, of features waarin het verschil duidelijker zit",
        ],
    )
    explain_look_alike
    return (explain_look_alike,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), die in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl) het plaatje van een cijfer vond dat het meest op een gegeven plaatje lijkt, doorzoekt nu 300 foto's tegelijk, op de features uit [Wat een beeldmodel ziet: features en een linear probe](/course/open/cv-pretrained-features?lang=nl). Het is de eerste helft van [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), met een minteken, omdat een grote similarity dichtbij betekent. In de volgende les, *Een classifier finetunen op je eigen soorten foto's*, leert het model zelf ook een beetje: elke layer van een pretrained classifier, op een paar honderd foto's van nieuwe soorten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pixels, choice_sims, explain_look_alike, result_search):
    L.wrap_up(choice_sims, result_search, choice_pixels, explain_look_alike)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
