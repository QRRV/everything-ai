# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-recommend-problem. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-recommend-problem", "nl")
    C = L.checks   # de bestellingen gesplitst op tijd, de popularity baseline en willekeurig kiezen voor elke k, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) en één over [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="In een test set zijn 98 van de 100 patiënten gezond. Een model dat een zeldzame ziekte opspoort, heeft het bij 97 van de 100 goed. Wat zeg je?",
        options={
            "right": "Het doet het slechter dan altijd gezond zeggen, wat er 98 goed heeft",
            "good": "Het is goed: 97% goed",
            "train": "Meet het liever op de training set",
        },
        correct="right",
        why={
            "good": "97% klinkt hoog, maar tegen iedereen gezond zeggen, zonder te kijken, heeft er 98 goed. Een score betekent pas iets naast zijn baseline. Zie *Het idee* in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "train": "Een model meet je op de test set, en zijn baseline ook: dezelfde set, dezelfde metric. Zie *Recept* in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "right": "Het model verslaat de baseline van de meest voorkomende klasse niet. Vandaag raadt de baseline aan wat iedereen koopt.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_duplicates():
    quick_duplicates = L.choice(
        "quick-duplicates",
        question="`df` heeft een rij voor elk product van elke bestelling, met de kolommen `invoice`, `customer` en `product`. Een klant die dezelfde mok in drie bestellingen kocht, heeft er drie rijen voor. Wat geeft één rij per klant en product?",
        options={
            "right": "`df.drop_duplicates(subset=[\"customer\", \"product\"])`",
            "all": "`df.drop_duplicates()`",
            "product": "`df.drop_duplicates(subset=\"product\")`",
        },
        correct="right",
        why={
            "all": "Zonder kolom laat `drop_duplicates` alleen rijen vallen die in elke kolom gelijk zijn, en de drie rijen van de mok hebben drie verschillende invoices. Zie *Het idee* in [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl).",
            "product": "Dat houdt één rij per product: de mok één keer, voor maar één klant. Zie de tabel met nieuwe woorden in [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl).",
            "right": "Van de rijen met dezelfde klant en hetzelfde product blijft alleen de eerste. Vandaag zijn die rijen de vakjes van een matrix.",
        },
    )
    quick_duplicates
    return (quick_duplicates,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een Britse webwinkel in cadeaus, voor een groot deel aan andere winkels, had 3.633 verschillende producten en 4.334 klanten van wie hij de bestellingen kent, van december 2010 tot december 2011. Op een pagina of in een mail passen tien producten. Welke tien moet elke klant zien? Een winkel, een streamingdienst of een nieuwssite beantwoordt die vraag voor elke bezoeker, vele keren per dag. En hij kan de klant niet vragen of het antwoord goed was: hij kan alleen afwachten wat de klant daarna doet.

    **Een voorbeeld uit het dagelijks leven.** Een boekhandelaar die je kent, legt drie boeken voor je apart voor je volgende bezoek. Zijn het de drie bestsellers van de week, dan heeft hij het vaak goed, en heb je ze misschien al gelezen.
    """)
    return


@app.cell(hide_code=True)
def predict_popular():
    guess_popular = L.guess(
        "popular-guess",
        question="Geef elke klant de 10 populairste producten die de klant nog niet kocht. Kijk dan naar de volgende bestelling. Bij hoeveel klanten zit minstens één van de tien erin?",
        options={
            "few": "Ongeveer 1 op de 25",
            "right": "Ongeveer 1 op de 5",
            "half": "Ongeveer de helft",
        },
        reveal="**Ongeveer 1 op de 5.** 541 van de 2.498 klanten: 21,7%. Tien producten willekeurig gekozen uit wat een klant nog niet kocht, vinden iets bij 4,2%. Populariteit doet het vijf keer zo goed, en vindt nog steeds niets bij vier van de vijf klanten. Dat is de baseline: een recommender die hem niet verslaat, heeft niets over de klant geleerd.",
    )
    guess_popular
    return (guess_popular,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Gebruikers, items en interacties.** Een **recommender system** kiest voor elke gebruiker welke items die te zien krijgt: hier klanten en producten. Eén klant die één product koopt, is een **interactie**. Zet ze in een **user-item matrix**: een rij voor elke klant, een kolom voor elk product, en een 1 waar de klant het product kocht. Deze heeft 4.334 × 3.633 = 15.745.422 vakjes, en 266.035 daarvan zijn 1: 1,69%. Een matrix die bijna helemaal 0 is, is **sparse**, en hoe leeg hij is, zijn **sparsity**. Een recommender raadt welke lege vakjes straks gevuld worden. Een set producten per klant is dezelfde matrix met alleen zijn enen: 266.035 waarden in plaats van bijna 16 miljoen.

        **Implicit feedback.** Een aankoop is **implicit feedback**: de klant zei niet wat die van een product vindt, maar kocht het. Een 1 zegt iets; een 0 zegt weinig. De klant kent het product misschien niet, kocht het ergens anders, of heeft er geen nodig. Sterren die een klant aan een product geeft, zeggen meer, maar weinig klanten geven ze.

        **Test op wat later komt.** Splits op tijd: de laatste bestelling van elke klant is de test, en de bestellingen ervoor zijn alles wat een recommender mag gebruiken. Raad `k` producten aan die de klant nog niet eerder kocht, en kijk naar de producten van de laatste bestelling die nieuw zijn voor die klant. De **hit rate** bij `k` is het deel van de klanten voor wie minstens één van de `k` producten daarbij zit. 2.829 klanten hebben een bestelling vóór hun laatste, en bij 2.498 van hen bevat de laatste bestelling minstens één nieuw product: zij zijn de test.

        **De popularity baseline.** De eenvoudigste recommender telt voor elk product hoeveel klanten het vóór hun laatste bestelling kochten, en geeft elke klant de populairste producten die de klant nog niet kocht. Iedereen krijgt bijna dezelfde lijst. Dat is de **popularity baseline**: een recommender die iets van de klant weet, hoort hem te verslaan.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **recommender system** | een systeem dat kiest welke items elke gebruiker te zien krijgt | |
        | **interactie** | één ding dat een gebruiker met een item deed, hier een aankoop | een rij van de tabel |
        | **user-item matrix** | een rij per gebruiker, een kolom per item, en een 1 voor elke interactie | |
        | **sparsity** | hoe leeg een matrix is: hier is 98,31% van de vakjes 0 | `1 - len(pairs) / (customers * products)` |
        | **implicit feedback** | wat gebruikers doen, zoals kopen, in plaats van wat ze zeggen goed te vinden | |
        | **hit rate** bij `k` | het deel van de gebruikers voor wie minstens één van `k` aanbevolen items later terugkomt | |
        | **popularity baseline** | de populairste items die een gebruiker nog niet had, in dezelfde volgorde voor iedereen | |
        | `transform("max")` | geeft na `groupby` elke rij het maximum van zijn groep, zodat de tabel zijn rijen houdt | `df.groupby("customer")["invoice"].transform("max")` |
        | `counts.index` | van wat `value_counts` geeft: de waarden die geteld zijn, de vaakste eerst | `counts.index[0]` |
        """),
        L.watch_out("Tel nooit op de bestellingen waarop je test. Tel je de populariteit op alle bestellingen, de laatste meegerekend, dan zegt de hit rate bij 10 26,1% in plaats van 21,7%: de tellingen bevatten dan precies de bestellingen die de test verbergt, en de aanbevelingen lijken alleen maar beter. Splits eerst, en tel daarna op de bestellingen van vóór de splitsing."),
        mo.md(r"""
        Het plaatje toont een hoek van de matrix: 60 van de testklanten, elk een rij, en de 120 populairste producten, elk een kolom, de populairste links. Een gevulde stip is een product dat de klant vóór de laatste bestelling kocht. Een open stip is een product uit de laatste bestelling dat de klant nog niet eerder kocht: wat de test verbergt. Van deze 7.200 vakjes zijn er 647 gevuld, 9,0%, tegen 1,69% in de hele matrix: dit zijn de producten die het vaakst gekocht worden. 132 zijn open.
        """),
        mo.Html(C.matrix({"title": "60 klanten, 120 producten", "xlabel": "product, populairste eerst", "ylabel": "klant",
                          "alt": "Een rooster van stippen: 60 klanten als rijen tegen de 120 populairste producten als kolommen. Gevulde stippen, 647, zijn producten die een klant vóór de laatste bestelling kocht; open stippen, 132, zijn nieuwe producten uit de laatste bestelling. De linkerkolommen, de populairste producten, hebben de meeste stippen."})),
    ])
    return


@app.cell(hide_code=True)
def text_before_slider():
    mo.md(r"""
    De slider zet `k`, hoeveel producten elke klant krijgt. De lijn met stippen is de hit rate van de popularity baseline over de 2.498 testklanten; de gestreepte lijn met open stippen is `k` producten willekeurig kiezen uit wat een klant nog niet kocht, als gemiddelde over elke manier om ze te kiezen. `k` loopt op een logaritmische schaal: 1, 5, 10, 20, 50. Schuif met de slider.
    """)
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        said = C.said(k, L.lang)
        alt = ("Een lijngrafiek van de hit rate in procenten tegen k op een logaritmische schaal, van 1 tot 50: de popularity baseline "
               "als gevulde stippen op een lijn, willekeurig kiezen als open stippen op een gestreepte lijn, met een gestreepte streep bij de k van de slider. " + said)
        return (C.picture(k, {"title": "Hit rate over 2.498 klanten", "ylabel": "hit rate (%)", "popular": "populariteit",
                              "random": "willekeurig", "alt": alt}), said)

    explore_k = L.explore("recommend-k", "k", _frame, start=10, values=C.KS, role="hyper",
                          task="Ga van k = 1 naar k = 50. Hoeveel keer zo vaak als willekeurig kiezen vindt populariteit iets bij k = 1, en bij k = 50?")
    explore_k
    return


@app.cell(hide_code=True)
def text_after_slider():
    mo.md(r"""
    Met één product per klant vindt populariteit iets bij 95 klanten, 3,8%: ongeveer negen keer zo vaak als willekeurig kiezen, 0,4%. Met 10 bij 21,7%; met 50 bij 52,0%, drie keer zo vaak als willekeurig, 17,7%. Een langere lijst vindt meer, maar een pagina toont maar een paar producten, en een klant leest er minder. Populariteit is moeilijk te verslaan: de 120 populairste producten zijn 3,4% van de 3.576, maar 14,2% van de nieuwe producten in de laatste bestellingen. Toch geeft hij iedereen bijna dezelfde lijst, en daar kan een recommender winnen die naar de klant kijkt.

    ### Recept

    Zo bouw je een test voor aanbevelingen en zijn baseline:

    1. Maak een tabel met een rij voor elke interactie: hier elk product van elke bestelling, met zijn klant.
    2. Splits op tijd: de laatste bestelling van elke gebruiker is de test, de rest is wat een recommender mag gebruiken. Houd de gebruikers met iets nieuws in hun laatste bestelling.
    3. Tel hoeveel gebruikers elk item vóór de splitsing hadden: `drop_duplicates` op gebruiker en item, dan `value_counts`.
    4. Neem voor elke gebruiker de `k` populairste items die die gebruiker nog niet had.
    5. Meet de hit rate bij `k`, en vergelijk elke recommender met deze baseline op dezelfde gebruikers en dezelfde `k`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel volgt stap 1 tot en met 3. Ze leest de bestellingen, houdt de rijen met een klant, en telt de klanten, de producten en de vakjes met een 1. `transform("max")` zet het nummer van de laatste bestelling van een klant op elke rij van die klant; de invoices lopen op met de tijd, dus de grootste is de laatste. `agg(set)` geeft een set per klant: `seen`, wat de klant vóór de laatste bestelling kocht, en de producten van de laatste bestelling. Net als een dictionary heeft wat het geeft `items()`, en `seen[customer]` is de set van één klant. Daarna houdt de cel per klant de producten van de laatste bestelling die niet in `seen` zitten, en telt ze de populariteit van elk product. De klantnummers zijn floats, zoals `14905.0`, omdat de kolom lege vakjes had vóór `dropna`, en pandas zo'n kolom als floats bewaart. Tot slot print ze voor klant 14905 bij elk van de 12 populairste producten of de klant het eerder kocht en of het in de laatste bestelling zit. Voer de cel uit.
    """)
    return


@app.cell
def example_split():
    import pandas as pd

    # 1. elk product van elke bestelling met een bekende klant, en de grootte van de user-item matrix
    _rows = pd.read_csv("data/online-retail-baskets.csv.gz").dropna(subset=["customer"])
    _pairs = _rows.drop_duplicates(subset=["customer", "product"])
    _customers = len(set(_pairs["customer"]))
    _products = len(set(_pairs["product"]))
    print(_customers, _products, len(_pairs), round(len(_pairs) / (_customers * _products), 4))

    # 2. splits op tijd: de laatste bestelling van elke klant is de test, de bestellingen ervoor mag een recommender gebruiken
    _rows["last"] = _rows.groupby("customer")["invoice"].transform("max")
    _train = _rows[_rows["invoice"] < _rows["last"]]
    _test = _rows[_rows["invoice"] == _rows["last"]]
    _seen = _train.groupby("customer")["product"].agg(set)
    _last = _test.groupby("customer")["product"].agg(set)
    _hidden = {}
    for _customer, _bought in _seen.items():
        _new = []
        for _product in _last[_customer]:
            if _product not in _bought:
                _new.append(_product)
        if len(_new) > 0:
            _hidden[_customer] = set(_new)
    print(len(_seen), len(_hidden))

    # 3. de populariteit van elk product: hoeveel klanten het vóór hun laatste bestelling kochten
    _counts = _train.drop_duplicates(subset=["customer", "product"])["product"].value_counts()
    print(len(_counts), _counts.index[0], _counts.iloc[0])

    # klant 14905: van de 12 populairste producten, eerder gekocht? in de laatste bestelling?
    for _product in _counts.index[:12]:
        print(_product in _seen[14905.0], _product in _hidden[14905.0], _product)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    4.334 klanten en 3.633 producten; 266.035 vakjes bevatten een 1, een aandeel van 0,0169. 2.829 klanten hebben een bestelling vóór hun laatste, en 2.498 van hen kochten daarin iets nieuws. 3.576 producten werden vóór een laatste bestelling gekocht; de REGENCY CAKESTAND 3 TIER door de meeste klanten, 663. Klant 14905 plaatste twee bestellingen. Van de 12 populairste producten kocht deze klant de tweede en de derde al eerder, de WHITE HANGING HEART T-LIGHT HOLDER en de PARTY BUNTING, dus loopt het tiental door tot het 12e. Twee van de tien zitten in de laatste bestelling: de JUMBO BAG RED RETROSPOT, 6e, en de HEART OF WICKER SMALL, 12e. Bij deze klant is de baseline raak.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_popular():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het populairst, nog niet gekocht

        `popular(counts, seen, k)` moet de lijst geven van de `k` populairste producten die niet in `seen` zitten, de populairste eerst. `counts` is wat `value_counts` geeft: `counts.index` bevat de producten, de meest gekochte eerst. Vervang de `...`: wanneer een product in `top` komt. De check roept `popular(counts, seen, 10)` aan met de tellingen van het uitgewerkte voorbeeld en `seen` de producten die klant 14905 vóór de laatste bestelling kocht, en probeert er nog vier op een winkel met acht producten.
        """),
        L.task("Laat `popular` de k populairste producten geven die een klant nog niet kocht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_popular():
    box_popular = L.code_box("popular", lines=6)
    box_popular
    return (box_popular,)


@app.cell(hide_code=True)
def check_popular(box_popular):
    result_popular = L.check("popular", box_popular.value)
    result_popular
    return (result_popular,)


@app.cell(hide_code=True)
def hints_popular():
    L.hints(
        "popular",
        [
            "Een product komt in `top` als twee dingen allebei waar zijn: het ene gaat over de klant, het andere over hoe lang `top` al is.",
            "De klant mag het niet gekocht hebben, en in `top` moet nog plek zijn voor één. `and` verbindt twee voorwaarden.",
            "`if product ___ seen and len(top) ___ k:`.",
        ],
        why="`counts.index` loopt van het populairste product omlaag. Het 2e en het 3e zitten in `seen`, dus die worden overgeslagen, en de tien zijn het 1e en het 4e tot en met het 12e.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hit_rate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De hit rate

        `hit_rate(recommended, hidden)` moet het deel van de klanten geven voor wie minstens één aanbevolen product bij hun verborgen producten zit. Allebei zijn het dictionaries met een klant als key: `recommended[customer]` is een lijst producten, `hidden[customer]` de set nieuwe producten van de laatste bestelling. Vervang de twee `...`: wanneer een aanbevolen product raak is, en wat de functie teruggeeft. De check roept `hit_rate(recommended, hidden)` aan met de tien producten van de popularity baseline voor alle 2.498 testklanten, en probeert drie kleine dictionaries.
        """),
        L.task("Laat `hit_rate` het deel van de klanten geven dat minstens één product terugvindt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hit_rate():
    box_hit_rate = L.code_box("hit-rate", lines=10)
    box_hit_rate
    return (box_hit_rate,)


@app.cell(hide_code=True)
def check_hit_rate(box_hit_rate):
    result_hit_rate = L.check("hit-rate", box_hit_rate.value)
    result_hit_rate
    return (result_hit_rate,)


@app.cell(hide_code=True)
def hints_hit_rate():
    L.hints(
        "hit-rate",
        [
            "Een product is raak als de klant het in de laatste bestelling kocht. De hit rate is een deel: van alle klanten, hoeveel vonden iets?",
            "Zoek een product op in de set van die ene klant, niet in heel `hidden`. En een deel is een aantal gedeeld door een totaal: welk totaal?",
            "`if product in ___:` en `return hits / len(___)`.",
        ],
        why="`found` wordt `True` bij de eerste hit en blijft `True`, dus een klant met twee hits telt één keer. Van de 2.498 klanten tellen er 541: 0,217.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_implicit():
    mo.md(r"""
    ### Oefening 3. Wat een 0 zegt
    """)
    return


@app.cell(hide_code=True)
def choice_implicit_cell():
    choice_implicit = L.choice(
        "implicit",
        question="Klant 14905 kocht nooit de REGENCY CAKESTAND 3 TIER, het populairste product. Wat zegt de 0 in de rij van deze klant in de user-item matrix?",
        options={
            "right": "Weinig: de klant kent hem misschien niet, kocht er ergens anders een, of heeft er geen nodig",
            "dislike": "Dat de klant hem niet mooi vindt",
            "never": "Dat de klant hem niet gaat kopen",
        },
        correct="right",
        why={
            "dislike": "Een aankoop is implicit feedback: hij zegt wat een klant deed, niet wat de klant ervan vindt. Veel klanten kennen de meeste van de 3.633 producten niet. Zie *Implicit feedback* in *Het idee*.",
            "never": "Een 0 gaat over het verleden. De open stippen in het plaatje van de matrix zijn producten die vóór de laatste bestelling een 0 waren en erin een 1. Zie *Implicit feedback* in *Het idee*.",
            "right": "Bij implicit feedback zegt een 1 iets en een 0 weinig. Daarom test je een recommender op wat klanten later kopen, en niet op de nullen.",
        },
    )
    choice_implicit
    return (choice_implicit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De popularity baseline is de baseline van [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl), en de hit rate de metric om op te vergelijken: dezelfde klanten, dezelfde verborgen bestellingen, dezelfde `k`. De tabel wordt schoongemaakt met de `drop_duplicates` van [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl), en de splitsing op tijd houdt de test eerlijk. Populariteit geeft iedereen bijna dezelfde lijst. In de volgende les, *Meer van wat je kocht: content-based filtering*, kijkt de lijst naar wat een klant kocht en naar wat elk product is: de woorden van zijn naam.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_implicit, result_hit_rate, result_popular):
    L.wrap_up(result_popular, result_hit_rate, choice_implicit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
