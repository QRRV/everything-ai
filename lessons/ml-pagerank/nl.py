# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-pagerank. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-pagerank", "nl")
    C = L.checks   # de karateclub, zijn vier centraliteiten, het kleine web, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hubs():
    quick_hubs = L.choice(
        "quick-hubs",
        question="In het netwerk van vluchten gaan de 100 grootste hubs dicht, met al hun routes: 3% van de vliegvelden. Wat gebeurt er met de vliegvelden die overblijven?",
        options={
            "right": "Ze komen veel verder uit elkaar te liggen: gemiddeld 5,74 vluchten in plaats van 4,10",
            "nothing": "Bijna niets: 3% van de vliegvelden is te weinig om uit te maken",
            "apart": "Geen van hen kan nog een ander bereiken",
        },
        correct="right",
        why={
            "nothing": "Dat gebeurt als er 100 willekeurige vliegvelden dichtgaan. De hubs houden de kortere wegen: zie de slider in [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl).",
            "apart": "De meeste hangen nog samen, 2.903 van de 3.325 die overblijven, maar ze liggen verder uit elkaar. Zie de slider in [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl).",
            "right": "En maar 2.903 van de 3.325 vliegvelden die overblijven, hangen nog samen. De graad vertelde je welke vliegvelden dat zijn. Vandaag kom je nodes tegen die ertoe doen terwijl hun graad niet hoog is.",
        },
    )
    quick_hubs
    return (quick_hubs,)


@app.cell(hide_code=True)
def quick_check_bfs():
    quick_bfs = L.choice(
        "quick-bfs",
        question="Breadth-first search geeft een vliegveld zijn aantal vluchten de eerste keer dat het hem tegenkomt, en verandert dat nooit meer. Waarom is dat eerste getal het kleinste?",
        options={
            "right": "De frontier is een wachtrij: elk vliegveld op 1 vlucht komt eruit vóór elk vliegveld op 2 vluchten",
            "fewest": "Het neemt altijd eerst de buur met de minste edges",
            "close": "Het is niet altijd het kleinste, maar wel dichtbij genoeg",
        },
        correct="right",
        why={
            "fewest": "Breadth-first search kijkt niet naar de graden: het neemt de vliegvelden in de volgorde waarin het ze vond. Zie oefening 2 van [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl).",
            "close": "Het is altijd het kleinste, en daardoor is het het kleinste aantal vluchten. Zie het *waarom* onder de hints van oefening 2 in [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl).",
            "right": "Dus één breadth-first search geeft het kleinste aantal stappen van één node naar alle andere. Vandaag tel je die stappen op, voor elke node.",
        },
    )
    quick_bfs
    return (quick_bfs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De graad telt de buren van een node, en in het netwerk van vluchten vond hij de hubs. Maar een node kan op andere manieren belangrijk zijn. Hij kan dicht bij alle andere liggen, zodat nieuws vanaf hem zich snel verspreidt. Hij kan tussen twee groepen in liggen, zodat wat er tussen hen gaat door hem heen gaat. Of er kunnen nodes naar hem linken die zelf belangrijk zijn. Elke manier geeft een ander getal, een centraliteit, en een andere ranglijst. De laatste is PageRank, het idee waarmee Google in 1998 begon om webpagina's te rangschikken op hun links. Een willekeurige wandeling zoals die erachter rangschikt ook producten: op een netwerk van klanten en producten komt een wandeling die steeds terugspringt naar één klant het vaakst uit bij de producten die dicht bij die klant liggen.

    **Een voorbeeld uit het dagelijks leven.** In een vriendengroep is degene die de meeste mensen kent niet altijd degene via wie nieuws de andere helft van de groep bereikt.
    """)
    return


@app.cell(hide_code=True)
def predict_between():
    guess_between = L.guess(
        "between-guess",
        question="In de karateclub van deze les heeft lid 1 9 vrienden, en lid 31 6. Wie van de twee ligt op meer kortste paden tussen twee andere leden?",
        options={
            "one": "Lid 1",
            "right": "Lid 31",
            "same": "Ongeveer evenveel",
        },
        reveal="**Lid 31.** Door lid 31 lopen kortste paden ter waarde van 73,0 van de 528 paren andere leden, door lid 1 maar 28,5. Van de vrienden van lid 1 naast lid 0 zijn er op één na allemaal ook vrienden van lid 0, dus de meeste paden kunnen om lid 1 heen. Lid 31 is bevriend met leden uit beide helften van de club.",
    )
    guess_between
    return (guess_between,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De karateclub.** Van 1970 tot 1972 volgde Wayne Zachary een karateclub aan een Amerikaanse universiteit. Van de 34 leden, hier genummerd van 0 tot 33, schreef hij op welke paren elkaar ook buiten de club zagen: 78 paren vrienden, een graaf zonder richtingen met 34 nodes en 78 edges. In die jaren kregen de instructeur, lid 0, en de beheerder, lid 33, ruzie, en de club viel in tweeën uiteen: met elk van hen gingen 17 leden mee. In de plaatjes zijn de leden die met de instructeur meegingen gevuld, en de andere open.

        **Centraliteit.** Een **centraliteit** is een getal voor elke node dat zegt hoe belangrijk hij in het netwerk is. Welke centraliteit je neemt, hangt af van wat je met belangrijk bedoelt. Vier ervan:

        1. **De graad**: het aantal vrienden. Lid 33 heeft er 17, lid 0 16.
        2. **Closeness centrality**: het aantal andere nodes gedeeld door de som van de stappen naar hen, het kleinste aantal stappen uit breadth-first search. Dat is 1 gedeeld door het gemiddelde aantal stappen, dus hoe dichter een node bij alle andere ligt, hoe hoger. Lid 0 bereikt de 33 anderen in samen 58 stappen: 33 / 58 = 0,569, gemiddeld 1,76 stappen naar elk.
        3. **Betweenness centrality**: voor elk paar andere nodes het deel van hun kortste paden dat door de node loopt, opgeteld over alle paren. Hebben twee leden 3 kortste paden en loopt er 1 door lid 0, dan telt dat paar voor 1/3. De andere 33 leden vormen 33 × 32 / 2 = 528 paren. Lid 0 krijgt 231,1, lid 33 160,6; 12 leden krijgen 0: geen enkel kortste pad heeft ze nodig.
        4. **PageRank**, via een **willekeurige wandeling**: een wandelaar begint bij een node, en volgt bij elke stap één edge van waar hij is, willekeurig gekozen. Af en toe, met een kans van 0,15, springt hij in plaats daarvan naar een willekeurige node. De **PageRank** van een node is het deel van alle stappen dat de wandelaar er op den duur doorbrengt. Een node krijgt een hoge PageRank als veel wandelingen naar hem leiden: als er nodes naar hem linken die zelf vaak bezocht worden.

        De slider laat één centraliteit tegelijk zien: 1 is de graad, 2 closeness, 3 betweenness en 4 PageRank. Hoe hoger de centraliteit van een lid, hoe groter zijn cirkel; de hoogste, en de leden 1 en 31, hebben hun nummer ernaast. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_club_cell():
    def _frame(i):
        i = int(i)
        return C.picture(i, L.lang), C.said(i, L.lang)

    explore_club = L.explore("club-centrality", "centraliteit", _frame, start=1, values=[1, 2, 3, 4], role="",
                             task="Schuif van 1 naar 4. Welke leden blijven bij de hoogste, en waar gaan de leden 1 en 31 heen?")
    explore_club
    return


@app.cell(hide_code=True)
def text_power():
    mo.vstack([
        mo.md(r"""
        De leden 0 en 33, de instructeur en de beheerder, horen bij alle vier bij de hoogste. Daaronder verschillen de ranglijsten. Lid 1, met 9 vrienden, staat 5e op graad maar 9e op closeness. Lid 31, met maar 6, staat 4e op closeness en 5e op betweenness: door dat lid lopen de paden tussen de twee helften van de club.

        **De machtsmethode.** Een wandelaar heeft veel stappen nodig voordat zijn delen vast komen te liggen. Er is een snellere weg naar dezelfde getallen: verspreid de kans zelf. Geef elke node eerst $1/n$, hier $1/34$. Elke ronde geeft elke node $d$ keer zijn score door aan zijn buren, in gelijke delen; $d = 0{,}85$ is de kans dat de wandelaar een edge volgt. Elke node krijgt daarnaast $(1 - d)/n$ voor de sprongen. Voor een node $v$:

        $$\text{PR}_\text{nieuw}(v) = \frac{1 - d}{n} + d \sum_{u} \frac{\text{PR}(u)}{\text{graad}(u)}$$

        De som loopt over de buren $u$ van $v$. Herhaal dat ronde na ronde: de **machtsmethode**. Na elke ronde tellen de scores op tot 1, want wat de ene node doorgeeft, krijgt een andere. Op de club veranderen de scores op den duur niet meer: vanaf ronde 20 verandert geen score meer dan 0,00001. Lid 33 eindigt op 0,101 en lid 0 op 0,097.

        **Richtingen.** Zonder richtingen blijft PageRank dicht bij het deel van de graden van elke node: lid 33 heeft 17 van de 156 uiteinden van edges, 0,109. Met richtingen kan het heel anders uitvallen. Sergey Brin en Larry Page gebruikten het voor het web, waar een link één kant op gaat: een pagina geeft zijn score alleen door aan de pagina's waar hij naar linkt. Een link van een pagina met een hoge PageRank en weinig links is dan veel meer waard dan veel links van pagina's waar niemand naar linkt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **centraliteit** | een getal voor elke node dat zegt hoe belangrijk hij in het netwerk is | |
        | **closeness centrality** | het aantal andere nodes gedeeld door de som van de stappen naar hen: 1 gedeeld door het gemiddelde aantal stappen | `(len(steps) - 1) / total` |
        | **betweenness centrality** | voor elk paar andere nodes het deel van hun kortste paden dat door de node loopt, opgeteld | |
        | **willekeurige wandeling** | een wandeling die bij elke stap langs een willekeurig gekozen edge gaat | `random.choice(club[node])` |
        | **PageRank** | het deel van de stappen dat een willekeurige wandelaar bij een node doorbrengt, met af en toe een sprong naar een willekeurige node | `scores[node]` |
        | $d$ | de kans dat de wandelaar een edge volgt in plaats van springt, hier 0,85 | `d` |
        | **machtsmethode** | scores berekenen door dezelfde ronde van doorgeven te herhalen tot ze niet meer veranderen | `for _ in range(rounds):` |
        """),
        L.watch_out("Er is niet één lid dat het belangrijkst is. Lid 33 staat bovenaan op graad en PageRank, lid 0 op closeness en betweenness, en lid 31, met maar 6 vrienden, staat 4e op closeness. Kies de centraliteit die bij je vraag past: wie bereikt iedereen snel, wie staat tussen groepen in, naar wie linken degenen die ertoe doen."),
        mo.md(r"""
        ### Recept

        Om uit te zoeken wie er in een netwerk toe doet:

        1. Zeg wat je met belangrijk bedoelt: veel buren, dicht bij iedereen, tussen anderen in, of gelinkt door nodes die ertoe doen.
        2. Voor de graad: tel de buren.
        3. Voor closeness: doe breadth-first search vanaf de node, tel de stappen op, en deel het aantal andere nodes door die som.
        4. Voor PageRank: geef elke node $1/n$, en herhaal de ronde van doorgeven tot de scores niet meer veranderen.
        5. Vergelijk de ranglijsten: een node die op de ene hoog staat en op de andere laag, vertelt je wat zijn plek in het netwerk is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel schrijft de club op als zijn 78 paren vrienden, en geeft elk lid een lijst van vrienden. Elk paar staat één keer in de lijst, dus geen vriend komt twee keer in een lijst. Ze print het aantal leden, het aantal edges, en de graden van de leden 0 en 33. Dan loopt een willekeurige wandelaar 100.000 stappen door de club, vanaf lid 0. Bij elke stap kiest hij een getal van 1 tot 100: bij 85 of lager gaat hij naar een willekeurige vriend, en anders springt hij naar een willekeurig lid. De cel telt hoe vaak de wandelaar bij elk lid is, en print de delen van de leden 33, 0 en 11. `random.seed(0)` maakt elke run gelijk. Voer de cel uit.
    """)
    return


@app.cell
def example_walk():
    import random

    # de 78 paren leden die elkaar ook buiten de club zagen (Zachary 1977)
    _pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13),
              (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
              (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6),
              (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32),
              (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
              (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 29),
              (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33),
              (32, 33)]

    # elk lid en de lijst van zijn vrienden
    _club = {}
    for _m in range(34):
        _club[_m] = []
    for _a, _b in _pairs:
        _club[_a].append(_b)
        _club[_b].append(_a)
    print(len(_club), len(_pairs), len(_club[0]), len(_club[33]))

    # een willekeurige wandelaar: volg een vriend met kans 85 op 100, spring anders naar een willekeurig lid
    random.seed(0)
    _node = 0
    _visits = {}
    for _m in _club:
        _visits[_m] = 0
    for _ in range(100000):
        if random.randint(1, 100) <= 85:
            _node = random.choice(_club[_node])
        else:
            _node = random.randint(0, 33)
        _visits[_node] = _visits[_node] + 1
    print(_visits[33] / 100000, _visits[0] / 100000, _visits[11] / 100000)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    34 leden, 78 edges; lid 0 heeft 16 vrienden en lid 33 17. De wandelaar brengt 9,97% van zijn stappen door bij lid 33 en 9,80% bij lid 0, dicht bij hun PageRank uit de machtsmethode, 0,101 en 0,097. Lid 11 heeft één vriend, lid 0, en krijgt 0,89% van de stappen. Met meer stappen komen de delen dichter bij de PageRank; de machtsmethode komt er in 20 rondes, zonder toeval.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_closeness():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Closeness centrality

        `closeness(graph, node)` moet de closeness centrality van `node` geven: het aantal andere nodes dat hij bereikt, gedeeld door de som van de stappen naar hen. `graph` geeft elke node zijn buren, zoals het uitgewerkte voorbeeld de club maakte. Het vak begint met `steps_from`, de breadth-first search uit [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl): die geeft voor elke node die `start` bereikt het kleinste aantal stappen ernaartoe, met `start` zelf op 0. Vervang de twee `...`: de nieuwe `total`, en wat de functie teruggeeft. De check roept `closeness(club, 0)` aan met de club uit het uitgewerkte voorbeeld, en probeert zeven andere gevallen, waaronder een graaf in twee delen.
        """),
        L.task("Laat `closeness` het aantal andere nodes geven, gedeeld door de som van de stappen naar hen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_closeness():
    box_closeness = L.code_box("closeness", lines=21)
    box_closeness
    return (box_closeness,)


@app.cell(hide_code=True)
def check_closeness(box_closeness):
    result_closeness = L.check("closeness", box_closeness.value)
    result_closeness
    return (result_closeness,)


@app.cell(hide_code=True)
def hints_closeness():
    L.hints(
        "closeness",
        [
            "`steps` heeft een getal voor elke node die `node` bereikt, `node` zelf ook, op 0 stappen. Tel die getallen op in `total`.",
            "`node` zelf staat ook in `steps`, dus het aantal andere nodes is één minder dan het aantal nodes in `steps`.",
            "`total = total + steps[___]` en `return (len(___) - 1) / total`.",
        ],
        why="Het aantal anderen delen door de som is hetzelfde als 1 gedeeld door het gemiddelde aantal stappen: gemiddeld weinig stappen geeft een hoge closeness. `node` zelf telt 0 op bij de som, dus mag in de loop blijven, maar mag niet als een van de anderen meetellen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pagerank():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. PageRank met de machtsmethode

        `pagerank(graph, d, rounds)` moet de PageRank van elke node geven na `rounds` rondes van de machtsmethode. `graph` geeft elke node de nodes waaraan hij zijn score doorgeeft: zijn buren, of, met richting, de pagina's waar hij naar linkt. Elke node begint op `1 / n`. Elke ronde begint `new` voor elke node op `(1 - d) / n`, voor de sprongen; daarna geeft elke node `d` keer zijn score door, in gelijke delen, aan de nodes in `graph[node]`. Vervang de twee `...`: het deel dat `node` aan elk van hen doorgeeft, en de nieuwe score van `nxt` met dat deel. De check roept `pagerank(club, 0.85, 100)` aan, en probeert vijf andere gevallen, waaronder het kleine web van oefening 3.
        """),
        L.task("Laat `pagerank` de score van elke node ronde na ronde doorgeven, en de scores teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pagerank():
    box_pagerank = L.code_box("pagerank", lines=15)
    box_pagerank
    return (box_pagerank,)


@app.cell(hide_code=True)
def check_pagerank(box_pagerank):
    result_pagerank = L.check("pagerank", box_pagerank.value)
    result_pagerank
    return (result_pagerank,)


@app.cell(hide_code=True)
def hints_pagerank():
    L.hints(
        "pagerank",
        [
            "Een node geeft `d` keer zijn score door, niet alles: de rest is wat de sprongen uitdelen. En dat deel wordt in gelijke delen verdeeld over de nodes in `graph[node]`.",
            "`new[nxt]` heeft al wat `nxt` kreeg van de sprongen en van de nodes ervoor. Het deel van deze node komt daarbovenop.",
            "`share = d * scores[node] / len(___)` en `new[nxt] = new[nxt] + ___`.",
        ],
        why="Elke node geeft `d` keer zijn score door en elke node krijgt `(1 - d) / n` van de sprongen, dus de scores blijven ronde na ronde optellen tot 1. Een node met veel buren geeft kleinere delen door: daarom telt PageRank niet gewoon de links naar een node.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_web():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Welke pagina krijgt meer?

        In dit kleine web linkt elke andere pagina naar A. A linkt alleen naar X. D en E, waar geen pagina naar linkt, linken ook naar Y.

        ```python
        web = {"A": {"X"}, "B": {"A"}, "C": {"A"}, "D": {"A", "Y"}, "E": {"A", "Y"},
               "F": {"A"}, "X": {"A"}, "Y": {"A"}}
        ```
        """),
        C.web_picture(L.lang),
    ])
    return


@app.cell(hide_code=True)
def choice_web_cell():
    choice_web = L.choice(
        "web",
        question="Met d = 0,85: welke pagina krijgt de hoogste PageRank, X, met één link ernaartoe, van A, of Y, met twee links, van D en E?",
        options={
            "right": "X",
            "y": "Y: daar gaan meer links naartoe",
            "same": "Allebei evenveel: elk krijgt zijn links van pagina's met weinig links",
        },
        correct="right",
        why={
            "y": "Links tellen is juist wat PageRank niet doet. Het gaat om de score van de pagina waar een link vandaan komt, en over hoeveel links die zijn score verdeelt. Hoe hoog is de score van A, en die van D en E?",
            "same": "A krijgt een link van elke andere pagina, dus zijn score is hoog, en hij geeft 0,85 daarvan aan X alleen. Naar D en E linkt geen pagina: ze hebben alleen wat de sprongen ze geven, en verdelen 0,85 daarvan over twee links.",
            "right": "X krijgt 0,411 en Y 0,035. A krijgt links van alle 7 andere pagina's en heeft de hoogste PageRank, 0,461; hij geeft 0,85 daarvan aan X alleen. D en E krijgen alleen wat de sprongen geven, elk 0,019, en verdelen dat over A en Y.",
        },
    )
    choice_web
    return (choice_web,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De grafen en breadth-first search uit [Netwerken als data: graden en hubs](/course/open/ml-network-data?lang=nl) geven closeness: breadth-first search vanaf elke node, met zijn stappen opgeteld. De graad uit die les is één centraliteit van de vier. PageRank voegt een wandeling met toeval toe, en de machtsmethode maakt daar scores van die optellen tot 1. Met een wandeling die steeds terugspringt naar één klant rangschikken dezelfde stappen producten op een netwerk van klanten en producten, zoals [Klanten zoals jij: collaborative filtering](/course/open/ml-collaborative-filtering?lang=nl) dat uit bestellingen maakt. Hiermee eindigt de module over aanbevelen. De volgende module gaat over tabellen in de praktijk, en begint met het model dat je als eerste probeert op een tabel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_web, result_closeness, result_pagerank):
    L.wrap_up(result_closeness, result_pagerank, choice_web)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
