# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-network-data. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-network-data", "nl")
    C = L.checks   # de gradenverdeling, de hubs, de vliegvelden met hubs dicht, de download, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl) en één over [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_degrees():
    quick_degrees = L.choice(
        "quick-degree-sum",
        question="Een graaf zonder richtingen heeft 5 edges. Wat is de som van de graden van al zijn nodes?",
        options={
            "right": "10",
            "five": "5",
            "depends": "Dat hangt af van het aantal nodes",
        },
        correct="right",
        why={
            "five": "Elke edge heeft twee uiteinden, en telt 1 op bij de graad van de node aan elk uiteinde. Zie de adjacency list in [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl): daarin staan twee namen voor elke edge.",
            "depends": "Elke edge telt 1 op bij de graad van de node aan elk van zijn twee uiteinden, hoeveel nodes er ook zijn. Zie de adjacency list in [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl): daarin staan twee namen voor elke edge.",
            "right": "Twee keer het aantal edges: elke edge telt 1 op bij de graad aan elk van zijn twee uiteinden. Vandaag geeft dat de gemiddelde graad van een netwerk met 19.256 edges.",
        },
    )
    quick_degrees
    return (quick_degrees,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="Op een log-log plot gaat een rechte lijn door (1, 1000) en (100, 0,1). Wat is de helling?",
        options={
            "right": "−2",
            "four": "−4",
            "half": "−0,5",
        },
        correct="right",
        why={
            "four": "Van 1000 omlaag naar 0,1 is vier factoren 10, maar opzij, van 1 naar 100, gaat de lijn ook twee factoren 10. De helling is omhoog gedeeld door opzij. Zie *Het idee* in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "half": "De helling is omhoog gedeeld door opzij, niet opzij gedeeld door omhoog: vier factoren 10 omlaag op twee opzij. Zie *Het idee* in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "right": "Vier factoren 10 omlaag op twee opzij: −4 / 2 = −2, een machtswet met exponent −2. Vandaag lees je zo'n helling af aan de graden van een netwerk.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel van de wereld is een netwerk: vliegvelden en de vluchten ertussen, webpagina's en hun links, mensen en wie wie volgt. Zulke netwerken zijn geen roosters, waarin elke node evenveel buren heeft, en ze zijn ook niet willekeurig. Een paar nodes hebben heel veel edges, en de meeste maar een paar. Dat bepaalt wat er op zo'n netwerk gebeurt: hoe snel een vertraging of een virus zich verspreidt, wat er stukgaat als één node uitvalt, en waarom een paar items veel vaker gezien worden dan alle andere. Voordat je kunt zeggen wie ertoe doet in een netwerk, moet je zien hoe de edges over de nodes verdeeld zijn.

    **Een voorbeeld uit het dagelijks leven.** Vlieg je van een kleine stad naar een kleine stad op een ander continent, dan stap je bijna altijd over op een groot vliegveld, zoals Amsterdam of Atlanta. Twee vluchten of drie, nooit twintig.
    """)
    return


@app.cell(hide_code=True)
def predict_hubs():
    guess_hubs = L.guess(
        "hubs-guess",
        question="In het netwerk van vluchten van deze les heeft een vliegveld gemiddeld 11,2 buren: vliegvelden waar je zonder overstappen heen vliegt. Hoeveel buren heeft het vliegveld met de meeste?",
        options={
            "small": "Ongeveer 25",
            "middle": "Ongeveer 60",
            "right": "Ongeveer 250",
        },
        reveal="**Ongeveer 250.** Amsterdam heeft 248 buren, Frankfurt 244, Parijs 240. De helft van alle vliegvelden heeft er 3 of minder. In een willekeurig netwerk met evenveel vliegvelden en edges zou het vliegveld met de meeste er 27 hebben.",
    )
    guess_hubs
    return (guess_hubs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het netwerk van vluchten.** OpenFlights verzamelt de routes van luchtvaartmaatschappijen: 67.663 stuks, zoals ze in juni 2014 waren. Een route is één maatschappij die van het ene vliegveld naar het andere vliegt. Maak van elk vliegveld een node, en verbind twee vliegvelden met een edge als er minstens één route tussen vliegt, in welke richting ook: een graaf zonder richtingen met 3.425 nodes en 19.256 edges. De graad van een vliegveld is zijn aantal buren, de vliegvelden waar je zonder overstappen heen vliegt.

        **De gradenverdeling.** De graden tellen op tot twee keer het aantal edges, dus de gemiddelde graad is 2 × 19.256 / 3.425 = 11,2. Tel nu voor elke graad hoeveel vliegvelden die hebben: de **gradenverdeling**. 768 vliegvelden hebben graad 1 en 744 graad 2; de helft van alle vliegvelden heeft er 3 of minder. Een paar hebben er veel meer: Amsterdam 248, Frankfurt 244, Parijs 240, Istanbul 236 en Atlanta 217. Dat zijn **hubs**, nodes met veel meer edges dan bijna alle andere. De 34 grootste hubs, 1% van de vliegvelden, zitten aan een uiteinde van 5.495 van de 19.256 edges: 28,5%.
        """),
        mo.Html(C.degrees_picture({"normal": "Gewone assen", "loglog": "Log-log plot", "degree": "graad", "airports": "vliegvelden",
                                   "alt": "Twee grafieken van het aantal vliegvelden met elke graad. Op gewone assen vormen de stippen een piek bij graad 1 tot 3, tot 768 vliegvelden, en liggen ze plat langs de onderkant tot graad 248. Op een log-log plot liggen de gevulde stippen langs een gestippelde rechte lijn van graad 2 tot ongeveer 100 en dunnen ze daarboven uit; de open stippen van een willekeurig netwerk vormen een smalle bult tussen graad 2 en 27."})),
        mo.md(r"""
        Op gewone assen, links, is de gradenverdeling een piek aan het begin en een lange platte staart: de hubs verdwijnen langs de onderkant. Op de log-log plot, rechts, liggen de gevulde stippen, één voor elke graad, langs een rechte lijn van graad 2 tot ongeveer 100, met een helling van ongeveer −1,5: een vliegveld met 10 keer zoveel buren is ongeveer 35 keer zo zeldzaam. Een rechte lijn op een log-log plot is een machtswet, zoals in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl). Een netwerk waarvan de gradenverdeling een machtswet volgt, is een **schaalvrij netwerk**: het heeft geen typische graad, geen schaal. Boven ongeveer 100 dunnen de stippen uit: bij 99 van de 148 graden van 101 tot 248 is er geen enkel vliegveld. De lijn zou ongeveer 100 vliegvelden geven met 101 tot 248 buren, en ongeveer 150 met meer; het zijn er 64, en geen: geen vliegveld heeft plek voor duizend bestemmingen. Hoeveel echte netwerken echt schaalvrij zijn, is een punt van discussie, maar veel kleine nodes en een paar enorme hubs kom je in veel netwerken tegen, zoals webpagina's en hun links.

        De open stippen zijn een willekeurig netwerk met evenveel nodes en edges, elke edge tussen twee willekeurig gekozen vliegvelden. De graden liggen tussen 2 en 27, de meeste dicht bij 11: een bult rond het gemiddelde, en geen hubs.

        **Een kleine wereld.** Breadth-first search vanaf een vliegveld, met de stappen geteld, geeft het kleinste aantal vluchten naar elk ander vliegveld. 3.397 vliegvelden hangen samen; de andere 28 hebben alleen routes onderling, in zeven kleine groepjes. Twee vliegvelden die elkaar kunnen bereiken, liggen gemiddeld 4,10 vluchten uit elkaar, en van alle paren vliegvelden ligt 67,6% hoogstens 4 vluchten uit elkaar. Vanuit Abilene in Texas, met Dallas-Fort Worth als enige buur, bereik je met hoogstens 3 vluchten 1.361 andere vliegvelden. Dat is een **small-world network**: de meeste nodes liggen een paar stappen uit elkaar, hoewel elke node maar een paar buren heeft. De hubs maken de kortere wegen: vanaf een klein vliegveld vlieg je naar een hub, van daar naar een andere hub, en dan door naar waar je heen wilt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **gradenverdeling** | voor elke graad hoeveel nodes die hebben | `counts[3]`: het aantal nodes met 3 buren |
        | **hub** | een node met veel meer edges dan bijna alle andere | Amsterdam, met 248 buren |
        | **schaalvrij netwerk** | een netwerk waarvan de gradenverdeling een machtswet volgt: een rechte lijn op een log-log plot, en geen typische graad | |
        | **small-world network** | een netwerk waarin de meeste nodes een paar stappen uit elkaar liggen, hoewel elke node maar een paar buren heeft | |
        """),
        L.watch_out("De gemiddelde graad, 11,2, past bij bijna geen enkel vliegveld: 56% heeft 3 buren of minder, en de hubs trekken het gemiddelde omhoog. Kijk in een netwerk met hubs naar de hele gradenverdeling, niet alleen naar het gemiddelde."),
        mo.md(r"""
        Wat doen de hubs voor de kleine wereld? De slider sluit de grootste hubs, met al hun routes: 10, 20, 50, 100 of 200 stuks. De doorgetrokken lijn is het deel van alle paren open vliegvelden dat hoogstens zoveel vluchten uit elkaar ligt; de gestippelde lijn hetzelfde als er evenveel willekeurig gekozen vliegvelden dichtgaan; de grijze lijn met elk vliegveld open. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_closed_cell():
    def _frame(h):
        h = int(h)
        said = C.closed_said(h, L.lang)
        alt = ("Een grafiek met, van 1 tot 10 vluchten, het percentage paren vliegvelden dat hoogstens zoveel vluchten uit elkaar ligt: "
               "met de grootste hubs dicht, doorgetrokken; met evenveel willekeurige vliegvelden dicht, gestippeld; met elk vliegveld open, grijs. " + said)
        return C.closed_picture(h, {"title": "Paren vliegvelden binnen zoveel vluchten", "flights": "vluchten",
                                    "pairs": "% van de paren", "alt": alt}), said

    explore_closed = L.explore("close-hubs", "hubs dicht", _frame, start=50, values=C.STOPS, role="",
                               task="Schuif van 0 naar 200. Hoe ver komen de vliegvelden uit elkaar te liggen met de hubs dicht, en met evenveel willekeurige vliegvelden dicht?")
    explore_closed
    return


@app.cell(hide_code=True)
def text_after_slider():
    mo.md(r"""
    Sluit 100 willekeurige vliegvelden, en van de 3.325 die over zijn, hangen er nog 3.223 samen, gemiddeld 4,15 vluchten uit elkaar: bijna zoals eerst. Sluit in plaats daarvan de 100 grootste hubs, 3% van de vliegvelden, en dan doen dat er nog maar 2.903, gemiddeld 5,74 vluchten uit elkaar; nog maar 17,4% van de paren ligt hoogstens 4 vluchten uit elkaar, tegen 67,6% met elk vliegveld open. Een netwerk met hubs merkt bijna niets van een willekeurig vliegveld dat dichtgaat, maar verliest zijn kleine wereld als zijn hubs dichtgaan.

    ### Recept

    Zo bekijk je een netwerk als data:

    1. Maak van elk ding een node, en van elke verbinding tussen twee dingen een edge; zonder richtingen bewaar je voor elke node een verzameling buren, zodat twee verbindingen tussen dezelfde nodes één keer tellen.
    2. De graad van een node is de grootte van zijn verzameling. Tel hoeveel nodes elke graad hebben: de gradenverdeling.
    3. Teken die op een log-log plot. Een rechte lijn is een machtswet: een schaalvrij netwerk, met hubs. Een bult rond het gemiddelde betekent geen hubs.
    4. Benoem de hubs: de nodes met de hoogste graden.
    5. Tel met breadth-first search de stappen van een node naar alle andere, om te zien hoe klein de wereld is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel volgt stap 1 en 2. `flights.routes()` geeft de routes als paren codes van vliegvelden, van en naar. De cel geeft elk vliegveld een lege verzameling, en zet elk vliegveld van een route in de verzameling van het andere, zodat een route terug, of een tweede maatschappij op dezelfde route, niets toevoegt. Dan telt ze de graden op, samen twee keer het aantal edges, en print het aantal vliegvelden, het aantal edges en de gemiddelde graad. Als laatste print ze de graad van Amsterdam, AMS, en van Abilene, ABI, met de ene buur van Abilene, en telt ze de vliegvelden met minstens 100 buren. `flights` is `everything_ai.flights` van de course, die de cel hieronder geeft zodra de routes gedownload zijn. Voer de cel uit.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="De routes downloaden, 2,4 MB")
    return (get_button,)


@app.cell(hide_code=True)
def flights_cell(get_button):
    flights = C.flights_or_stop(get_button, {
        "ask": "*De routes van OpenFlights staan nog niet op je computer. De knop downloadt ze één keer uit de repository van OpenFlights op GitHub, 2,4 MB. Het uitgewerkte voorbeeld heeft ze nodig, en met de routes laten de plaatjes onder je checks alle 3.425 vliegvelden zien; de rest van de les werkt zonder. De course host ze niet: hun licentie, de Open Database License (ODbL), vraagt dat een database die ervan gemaakt is onder dezelfde licentie gedeeld wordt.*",
        "busy": "**De routes worden gedownload** · {done} van {total} MB",
        "no_net": "*Het downloaden lukte niet: GitHub was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen plek op je schijf voor 2,4 MB. Maak plek, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
        "here": "*De routes van OpenFlights staan op je computer: 67.663 routes tussen 3.425 vliegvelden. Ze bevatten informatie van OpenFlights, beschikbaar gesteld onder de ODbL.*",
    })
    return (flights,)


@app.cell
def example_network(flights):
    # de routes als paren codes van vliegvelden: van, naar
    _routes = flights.routes()
    print(len(_routes), _routes[:3])

    # 1. elk vliegveld en de verzameling van zijn buren, zonder richtingen
    _graph = {}
    for _a, _b in _routes:
        if _a != _b:   # één route gaat van een vliegveld naar zichzelf
            if _a not in _graph:
                _graph[_a] = set()
            if _b not in _graph:
                _graph[_b] = set()
            _graph[_a].add(_b)
            _graph[_b].add(_a)

    # 2. de graden tellen op tot twee keer het aantal edges
    _total = 0
    for _airport in _graph:
        _total = _total + len(_graph[_airport])
    print(len(_graph), _total / 2, round(_total / len(_graph), 2))

    # de graad van een hub en van een klein vliegveld, en de vliegvelden met minstens 100 buren
    print(len(_graph["AMS"]), len(_graph["ABI"]), _graph["ABI"])
    _big = 0
    for _airport in _graph:
        if len(_graph[_airport]) >= 100:
            _big = _big + 1
    print(_big)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    67.663 routes worden 3.425 vliegvelden en 19.256 edges: veel routes vliegen tussen hetzelfde paar vliegvelden, in beide richtingen of met meer dan één maatschappij. De gemiddelde graad is 11,24. Amsterdam heeft 248 buren, en Abilene één, Dallas-Fort Worth, geschreven als `{'DFW'}`: een verzameling met één code. Maar 64 vliegvelden, minder dan 2%, hebben 100 buren of meer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De gradenverdeling

        `degree_counts(graph)` moet de gradenverdeling van `graph` geven: een dictionary met voor elke graad het aantal nodes dat die heeft. `graph` geeft elke node zijn buren, zoals het uitgewerkte voorbeeld hem maakte. Vervang de twee `...`: de graad van `node`, en de nieuwe telling van die graad. De check roept `degree_counts(small)` aan met het kleine netwerk hieronder, en probeert vier andere netwerken.

        ```python
        small = {"H": {"A", "B", "C", "D", "K"}, "K": {"H", "E", "F", "G"}, "A": {"H", "B"}, "B": {"H", "A"},
                 "C": {"H"}, "D": {"H"}, "E": {"K"}, "F": {"K"}, "G": {"K"}}
        ```

        H en K zijn de hubs. A en B vliegen naar H en naar elkaar; C en D alleen naar H, en E, F en G alleen naar K.
        """),
        L.task("Laat `degree_counts` voor elke graad geven hoeveel nodes die hebben."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("degree-counts", lines=9)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("degree-counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "degree-counts",
        [
            "De graad van een node is zijn aantal buren, en die staan in `graph[node]`. Elke node telt daarna één op bij de telling van zijn graad.",
            "De lengte van een verzameling is het aantal elementen erin. De regels `if degree not in counts:` en `counts[degree] = 0` zorgen dat `counts[degree]` er is, en bij 0 begint.",
            "`degree = len(___)` en `counts[degree] = counts[degree] + ___`.",
        ],
        why="Elke node wordt één keer geteld, bij zijn graad, dus de tellingen tellen op tot het aantal nodes: 9 voor het kleine netwerk, 3.425 voor het netwerk van vluchten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flights():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel vluchten naar elk vliegveld

        `flights_from(graph, start)` moet voor elk vliegveld dat vanaf `start` te bereiken is het kleinste aantal vluchten erheen geven: een dictionary met `start` op 0, zijn buren op 1, hun buren op 2, enzovoort. Het is de breadth-first search van [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl), maar die houdt voor elk vliegveld een getal bij in plaats van een pad. Vervang de twee `...`: wanneer een buur `nxt` een getal krijgt, en welk getal. De check roept `flights_from(small, "C")` aan met het kleine netwerk van oefening 1, en probeert vijf andere gevallen, waaronder een ring.
        """),
        L.task("Laat `flights_from` het kleinste aantal vluchten geven van `start` naar elk vliegveld dat het kan bereiken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flights():
    box_flights = L.code_box("flights-from", lines=13)
    box_flights
    return (box_flights,)


@app.cell(hide_code=True)
def check_flights(box_flights):
    result_flights = L.check("flights-from", box_flights.value)
    result_flights
    return (result_flights,)


@app.cell(hide_code=True)
def hints_flights():
    L.hints(
        "flights-from",
        [
            "Breadth-first search neemt de vliegvelden in de volgorde waarin het ze vond, dus de eerste keer dat het een vliegveld tegenkomt, is het via de kortste weg gekomen. Wat moet er de tweede keer gebeuren?",
            "Met `in` zie je of `nxt` al in `flights` staat. En het getal van een buur volgt uit het getal van het vliegveld waar hij vandaan gevonden is.",
            "`if nxt ___ flights:` en `flights[nxt] = flights[___] + 1`.",
        ],
        why="De frontier is een wachtrij, dus elk vliegveld op 1 vlucht komt eruit vóór elk vliegveld op 2 vluchten. Het eerste getal dat een vliegveld krijgt, is het kleinste, en een vliegveld met een getal gaat nooit meer de frontier in: dat voorkomt ook dat de search eindeloos rond een ring gaat.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fragile():
    mo.md(r"""
    ### Oefening 3. Welk netwerk valt uit elkaar?
    """)
    return


@app.cell(hide_code=True)
def choice_fragile_cell():
    choice_fragile = L.choice(
        "fragile",
        question="Neem het netwerk van vluchten en het willekeurige netwerk met evenveel vliegvelden en edges, waarvan de grootste node 27 buren heeft. Sluit in elk de 100 nodes met de hoogste graad. In welk netwerk komen de nodes die over zijn veel verder uit elkaar te liggen?",
        options={
            "right": "In het netwerk van vluchten",
            "random": "In het willekeurige netwerk",
            "same": "In allebei evenveel: 100 nodes is 3% van elk",
        },
        correct="right",
        why={
            "random": "Het willekeurige netwerk heeft geen hubs: zijn 100 grootste nodes hebben elk ongeveer 20 buren, en de andere vinden er een weg omheen. Zie de slider in *Het idee* voor wat het sluiten van de hubs met het netwerk van vluchten doet.",
            "same": "De 3% is hetzelfde, maar de edges die ze meenemen niet: de 100 grootste hubs van het netwerk van vluchten hebben duizenden edges, en de kortere wegen ertussen. Zie de slider in *Het idee*.",
            "right": "Met de 100 grootste hubs dicht gaat het netwerk van vluchten van gemiddeld 4,10 naar 5,74 vluchten, en hangen nog maar 2.903 van de 3.325 vliegvelden die over zijn samen, tegen 3.397 van de 3.425 eerst. Het willekeurige netwerk gaat van gemiddeld 3,63 naar 3,72 stappen, en elke node die over is, hangt nog samen.",
        },
    )
    choice_fragile
    return (choice_fragile,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De adjacency list en breadth-first search van [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl), nu op een netwerk van 3.425 echte vliegvelden; de log-log plot en de machtswet van [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl), nu in de gradenverdeling. De graad zegt hoeveel buren een node heeft, maar niet of hij tussen andere in ligt. In de volgende les, *Wie doet ertoe in een netwerk? Centraliteit en PageRank*, meet je op andere manieren hoe belangrijk een node is, en bereken je PageRank, het idee waarmee Google begon.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fragile, result_counts, result_flights):
    L.wrap_up(result_counts, result_flights, choice_fragile)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
