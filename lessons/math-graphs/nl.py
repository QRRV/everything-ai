# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les math-graphs. Route: course/route/lessen/klassiek-ml.yaml. Checks, de grafen en het speelblok: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-graphs", "nl")
    C = L.checks   # de grafen van de les, en de eigen adjacency list en matrix van de course die het speelblok tekenen

    def names_in(names):
        """Namen in woorden: B, E en F."""
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" en {names[-1]}"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl) en [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bfs():
    quick_bfs = L.choice(
        "quick-bfs",
        question="Breadth-first search vindt een pad van S naar G in een doolhof. Wat weet je van dat pad?",
        options={
            "first": "Het is de eerste weg die het doolhof biedt; er kan een kortere zijn",
            "right": "Geen pad van S naar G heeft minder stappen",
            "least": "Geen andere search breidt minder states uit om het te vinden",
        },
        correct="right",
        why={
            "right": "Breadth-first search pakt het oudste pad eerst uit zijn wachtrij, dus elk korter pad kwam er eerder uit, en geen daarvan bereikte G.",
            "first": "Breadth-first search probeert elk pad van 1 stap, dan van 2 stappen, enzovoort, dus het eerste pad dat G bereikt, is een kortste. Zie [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl).",
            "least": "Breadth-first search breidt elke state uit die dichter bij S ligt dan G, ook aan de kant die van G af ligt; zijn pad is een kortste, maar hij breidt veel states uit om het te vinden. Zie [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl).",
        },
    )
    quick_bfs
    return (quick_bfs,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`m = np.array([[0, 1, 1], [1, 0, 0]])`. Wat is `m.shape`?",
        options={
            "flat": "`(6,)`",
            "swapped": "`(3, 2)`",
            "right": "`(2, 3)`",
        },
        correct="right",
        why={
            "right": "Twee binnenste lijsten maken 2 rijen, van elk 3 getallen: eerst de rijen, dan de kolommen.",
            "swapped": "De shape geeft eerst de rijen: twee binnenste lijsten zijn 2 rijen, van elk 3 getallen. Zie [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "flat": "Een geneste lijst maakt een array met twee richtingen, rijen en kolommen, dus zijn shape heeft twee getallen. Zie [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel van de wereld bestaat uit dingen en de verbindingen ertussen: mensen en wie wie kent, vliegvelden en de vluchten ertussen, webpagina's en hun links, steden en wegen. Opgeschreven is elk daarvan een **graaf**: punten, en lijnen ertussen. Je kwam er al twee tegen zonder de naam. Het doolhof van [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl) is er een, met een vakje als punt en een stap als lijn, en de berekening in [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl) ook. Zodra een netwerk een graaf is, werkt één stuk code voor allemaal: breadth-first search vindt een kortste weg tussen twee mensen, net zoals in het doolhof.

    **Een voorbeeld uit het dagelijks leven.** Een metrokaart laat niet zien hoe lang de sporen zijn of hoe ze buigen. Hij laat de stations zien, en welk station op welk volgt: een graaf. Meer heb je niet nodig om een reis te plannen.
    """)
    return


@app.cell(hide_code=True)
def predict_pairs():
    guess_pairs = L.guess(
        "pairs",
        question="Zeven mensen tellen elk hoeveel van de anderen ze kennen: 2, 3, 3, 3, 3, 3 en 1. Kennen gaat twee kanten op. Hoeveel tweetallen kennen elkaar?",
        options={
            "sum": "18, de som van de tellingen",
            "half": "9",
            "unknown": "Dat kun je uit deze tellingen niet halen",
        },
        reveal="**9.** Elk tweetal is twee keer geteld, één keer door elk van de twee. Dus de tellingen komen samen op 18, twee keer het aantal tweetallen. In de woorden van deze les: de graden van een graaf zijn samen twee keer zijn aantal edges.",
    )
    guess_pairs
    return (guess_pairs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **graaf** is een verzameling **nodes** die verbonden zijn door **edges**. Een edge verbindt twee nodes, en die twee zijn elkaars buren. In deze les zijn de nodes zeven mensen, A tot en met G, en een edge betekent dat twee van hen elkaar kennen. De **graad** van een node is zijn aantal edges: A kent B en C, dus A heeft graad 2. Elke edge heeft twee uiteinden, dus de graden zijn samen twee keer het aantal edges, zoals bij de voorspelling.

        Iemand kennen gaat twee kanten op: een edge zonder richting. In een **gerichte graaf** heeft elke edge een richting, getekend als pijl: A volgt B op een sociaal netwerk, maar B hoeft A niet te volgen. In een **gewogen graaf** draagt elke edge een getal, zijn gewicht, zoals de lengte van een weg in km.

        Een **pad** is een rij nodes waarin elke node met een edge verbonden is met de volgende, zoals A, B, D: van A naar D in twee stappen. Zijn lengte is zijn aantal edges; in een gewogen graaf kun je ook de getallen langs het pad optellen. Breadth-first search vindt een pad met zo min mogelijk edges.

        Om met een graaf te rekenen, sla je hem op, op een van twee manieren. Een **adjacency list** geeft elke node de lijst van zijn buren; in Python een dictionary, waarin `graph["A"]` de lijst `["B", "C"]` geeft. Een **adjacency matrix** geeft elke node een nummer, en daarmee een rij en een kolom: A is 0, B is 1, enzovoort. Element `adj[i, j]` is 1 als er een edge van node `i` naar node `j` loopt, en 0 als dat niet zo is; in een gewogen graaf staat er het gewicht. Zonder richtingen staat elke edge er twee keer in, op `[i, j]` en op `[j, i]`, dus de matrix is gelijk aan zijn getransponeerde, zoals in [Een matrix omklappen: de getransponeerde](/course/open/math-transpose?lang=nl). Rij `i` telt op tot de graad van node `i`, dus `adj.sum(axis=1)` geeft alle graden tegelijk.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **graaf** | nodes die verbonden zijn door edges | |
        | **node** | een van de punten van een graaf, zoals een persoon | `"A"` |
        | **edge** | een verbinding tussen twee nodes, die ze tot buren maakt | `("A", "B")` |
        | **graad** | het aantal edges aan een node | `len(graph["A"])` |
        | **gerichte graaf** | een graaf waarvan de edges een richting hebben, getekend als pijlen | |
        | **gewogen graaf** | een graaf waarvan elke edge een getal draagt, zijn gewicht | `("P", "Q", 4)` |
        | **pad** | nodes op een rij, elk met een edge verbonden met de volgende | `["A", "B", "D"]` |
        | **adjacency list** | voor elke node de lijst van zijn buren | `graph["A"]` geeft `["B", "C"]` |
        | **adjacency matrix** | een matrix met een rij en een kolom per node, en op rij `i`, kolom `j` een 1, of het gewicht, voor een edge van node `i` naar node `j` | `adj[0, 1]` geeft `1.0` |
        | `np.zeros((n, n))` | een matrix van `n` rijen en `n` kolommen, allemaal nullen: zijn shape als tuple, zoals `.shape` hem schrijft | `np.zeros((2, 3))` |
        """),
        L.watch_out("Een graaf is geen grafiek. Een grafiek is een tekening van getallen; een graaf bestaat in deze course altijd uit nodes en edges."),
    ])
    return


@app.cell(hide_code=True)
def figure_kinds():
    C.kinds({"undirected": "Zonder richting: wie kent wie", "directed": "Gericht: wie volgt wie", "weighted": "Gewogen: wegen in km"},
            alt="Drie kleine grafen. Zonder richting: A, B, C en D, met vijf lijnen. Gericht: dezelfde vier mensen, met vijf pijlen; "
                "tussen B en D gaat de pijl twee kanten op. Gewogen: de steden P, Q, R en S, met wegen van 4 km van P naar Q, "
                "3 km van Q naar R, 9 km van P naar R en 2 km van R naar S.")
    return


@app.cell(hide_code=True)
def text_after_kinds():
    mo.md(r"""
    In de gewogen graaf is de weg van P rechtstreeks naar R één edge van 9 km; de weg via Q heeft twee edges, maar is samen 4 + 3 = 7 km. Breadth-first search telt edges, dus hij neemt de weg van 9 km.
    """)
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    De graaf van de zeven mensen drie keer: als tekening, als adjacency list en als adjacency matrix, waarvan de rijen en kolommen het nummer en de naam van elke node dragen. Schuif om een node te kiezen: zijn ring en zijn edges in de tekening, zijn regel in de lijst en zijn rij in de matrix zijn gemarkeerd, en zijn buren zijn grijs gevuld.
    """)
    return


@app.cell(hide_code=True)
def explore_node_cell():
    def _frame(k):
        k = int(k)
        _name = C.NAMES[k]
        _nbrs = C.GRAPH[_name]
        _d = len(_nbrs)
        _alt = (f"De graaf van de zeven mensen, met node {_name} gemarkeerd, en ernaast zijn adjacency list en adjacency matrix. "
                f"{_name} heeft de buren {names_in(_nbrs)}.")
        _said = (f"Node {_name}, nummer {k}: graad {_d}. Zijn lijst is `{C.GRAPH[_name]}`, en rij {k} van de matrix heeft "
                 f"{'één 1' if _d == 1 else f'{_d} keer een 1'}, in de {'kolom' if _d == 1 else 'kolommen'} van {names_in(_nbrs)}. "
                 f"Kolom {k} is hetzelfde als rij {k}: de edges hebben geen richting.")
        return C.views(k, {"graph": "De graaf", "list": "graph, de adjacency list", "matrix": "adj, de adjacency matrix"}, _alt).text, _said

    explore_node = L.explore("node", "nodenummer", _frame, start=3, low=0, high=len(C.NAMES) - 1, step=1, role="",
                             task="Schuif langs alle zeven. Welke node heeft de laagste graad, en waar zie je dat in elk van de drie?")
    explore_node
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    De lijst bevat 18 namen, twee voor elk van de 9 edges; de matrix bevat 49 getallen, waarvan er 18 een 1 zijn. Bij 1.000 mensen die elk 10 anderen kennen, zou de lijst 10.000 namen bevatten en de matrix 1.000.000 getallen, waarvan 99 van de 100 een 0. De matrix zegt je in één blik of twee nodes buren zijn, `adj[i, j]`; de lijst is veel korter, en geeft je meteen de buren van een node.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Van een lijst edges naar een graaf waarmee je kunt rekenen:

    1. Schrijf de nodes op, en elke edge als paar nodes, of met zijn gewicht als derde waarde.
    2. Adjacency list: geef elke node een lege lijst. Zet bij elke edge van `a` naar `b` de node `b` in de lijst van `a`; zonder richtingen zet je ook `a` in de lijst van `b`.
    3. Adjacency matrix: geef elke node een nummer van 0 tot en met `n - 1`, en begin met `np.zeros((n, n))`. Zet bij elke edge van node `i` naar node `j` het element `adj[i, j]` op 1, of op zijn gewicht; zonder richtingen zet je ook `adj[j, i]`.
    4. De graad van een node is de lengte van zijn lijst, of de som van zijn rij.
    5. Een search op de graaf vraagt de lijst om de buren van een state: `graph[state]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De zeven mensen, met hun edges. De cel bouwt de adjacency list en de adjacency matrix, laat de graad van elke node op beide manieren zien, en draait dan de breadth-first search van [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl) op de lijst: `_neighbors(maze, state)` is `graph[state]` geworden. `for _a, _b in _edges:` haalt elk paar uit elkaar, zoals `for key, value in d.items():` dat doet. Voer de cel uit.
    """)
    return


@app.cell
def example_graphs():
    import numpy as np
    from collections import deque

    # 1. de nodes, en de edges als paren: wie kent wie
    _names = ["A", "B", "C", "D", "E", "F", "G"]
    _edges = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"),
              ("D", "E"), ("D", "F"), ("E", "F"), ("F", "G")]

    # 2. de adjacency list: elke node begint met een lege lijst buren
    _graph = {}
    for _name in _names:
        _graph[_name] = []
    for _a, _b in _edges:
        _graph[_a].append(_b)
        _graph[_b].append(_a)   # geen richting: b is een buur van a, en a van b
    print(_graph)

    # 3. de adjacency matrix: node nummer i heeft rij i en kolom i
    _number = {}
    for _i in range(len(_names)):
        _number[_names[_i]] = _i
    _adj = np.zeros((len(_names), len(_names)))
    for _a, _b in _edges:
        _adj[_number[_a], _number[_b]] = 1
        _adj[_number[_b], _number[_a]] = 1
    print(_adj)

    # 4. de graad van elke node, uit de lijst en uit zijn rij van de matrix
    for _name in _names:
        print(_name, len(_graph[_name]), _adj[_number[_name]].sum())
    print(_adj.sum(axis=1))

    # 5. breadth-first search op de adjacency list
    def _bfs(graph, start, goal):
        frontier = deque([[start]])
        done = set()
        while len(frontier) > 0:
            path = frontier.popleft()
            state = path[-1]
            if state == goal:
                return path
            if state not in done:
                done.add(state)
                for nxt in graph[state]:
                    frontier.append(path + [nxt])
        return None

    print(_bfs(_graph, "A", "G"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De graden uit de lijst en uit de matrix zijn gelijk: 2, 3, 3, 3, 3, 3 en 1, samen 18, twee keer de 9 edges. De matrix laat `1.` zien omdat `np.zeros` floats maakt. De search vindt een pad van 4 stappen van A naar G, via B, D en F. De weg via C, E en F is even kort; breadth-first search geeft de weg die hij het eerst bereikt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_list():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je eigen adjacency list

        `to_list(names, edges)` bouwt de adjacency list van een graaf zonder richtingen, zoals in stap 2 van het recept. Vervang de `...` door de regels die elke edge in de lijsten zetten. De check roept `to_list(NAMES, EDGES)` aan, met de zeven mensen van deze les, en probeert drie andere grafen, waaronder een met een naam zonder edges. Elke volgorde van de buren telt als goed.
        """),
        L.task("Laat `to_list` bij elke edge `b` in de lijst van `a` zetten, en `a` in de lijst van `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_list():
    box_list = L.code_box("to-list", lines=8)
    box_list
    return (box_list,)


@app.cell(hide_code=True)
def check_list(box_list):
    result_list = L.check("to-list", box_list.value)
    result_list
    return (result_list,)


@app.cell(hide_code=True)
def hints_list():
    L.hints(
        "to-list",
        [
            "Stap 2 van het recept. Een edge zonder richting maakt twee mensen elkaars buren: hoeveel lijsten veranderen er bij elke edge?",
            "`graph[a]` is de lijst van `a`; `append` zet een naam achteraan.",
            "`graph[a].append(___)` en `graph[___].append(a)`.",
        ],
        why="`graph[a].append(b)` en `graph[b].append(a)` voegen toe aan de lijst die er al staat, dus elke naam verzamelt al zijn buren, edge voor edge.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_matrix():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De wegen als matrix

        De vier steden van de gewogen graaf hierboven, nu met nummers: P is 0, Q is 1, R is 2 en S is 3. De lijst `ROADS` van de check bevat elke weg als `(i, j, w)`, een weg van `w` km tussen stad `i` en stad `j`:

        ```python
        ROADS = [(0, 1, 4), (1, 2, 3), (0, 2, 9), (2, 3, 2)]
        ```

        `to_matrix(n, edges)` moet de adjacency matrix van `n` nodes geven, met de lengte van elke weg erin, en 0 waar geen weg is. Vervang de twee `...`. De check roept `to_matrix(4, ROADS)` aan, en probeert een graaf met een node zonder edges en een graaf zonder één edge.
        """),
        L.task("Begin met een matrix met nullen met een rij en een kolom voor elke node, en zet de lengte van elke weg op beide plekken waar hij hoort."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matrix():
    box_matrix = L.code_box("to-matrix", lines=8)
    box_matrix
    return (box_matrix,)


@app.cell(hide_code=True)
def check_matrix(box_matrix):
    result_matrix = L.check("to-matrix", box_matrix.value)
    result_matrix
    return (result_matrix,)


@app.cell(hide_code=True)
def hints_matrix():
    L.hints(
        "to-matrix",
        [
            "Stap 3 van het recept, met het gewicht op de plek van de 1. Hoe groot is de matrix, en op hoeveel plekken komt elke weg?",
            "`np.zeros((n, n))` maakt de matrix; `adj[i, j] = w` zet een getal op rij `i`, kolom `j`.",
            "`adj = np.zeros((___, ___))`, dan `adj[i, j] = w` en `adj[___, ___] = w`.",
        ],
        why="`np.zeros((n, n))` geeft elke node een rij en een kolom. Een weg gaat twee kanten op, dus zijn lengte staat op `[i, j]` en op `[j, i]`, en de matrix is gelijk aan zijn getransponeerde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_column():
    mo.md(r"""
    ### Oefening 3. Rijen en kolommen met richtingen
    """)
    return


@app.cell(hide_code=True)
def choice_column_cell():
    choice_column = L.choice(
        "column",
        question="Een gerichte graaf van wie wie volgt: `adj[i, j]` is 1 als persoon `i` persoon `j` volgt. Rij `i` laat zien wie `i` zelf volgt. Wat laat kolom `i` zien?",
        options={
            "same": "Weer wie `i` zelf volgt: een kolom is hetzelfde als zijn rij",
            "followers": "Door wie `i` gevolgd wordt",
            "nothing": "Niets: in een gerichte graaf tellen alleen de rijen",
        },
        correct="followers",
        why={
            "followers": "Kolom `i` bevat `adj[j, i]` voor elke `j`, en dat is 1 als `j` persoon `i` volgt. Zonder richtingen is de matrix gelijk aan zijn getransponeerde, dus is een kolom hetzelfde als zijn rij; met richtingen hoeft dat niet.",
            "same": "Dat klopt in een graaf zonder richtingen, waarvan de matrix gelijk is aan zijn getransponeerde. Hier is `adj[j, i]` 1 als `j` persoon `i` volgt, en dat zegt niets over of `i` persoon `j` volgt. Zie *Het idee*.",
            "nothing": "Elk element telt mee: `adj[j, i]`, in kolom `i`, is 1 als `j` persoon `i` volgt. Zie *Het idee*.",
        },
    )
    choice_column
    return (choice_column,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het doolhof van [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl) was al die tijd een graaf, met een vakje als node en een stap als edge; nu draait dezelfde search op elke graaf die als adjacency list is opgeslagen. De adjacency matrix is een array zoals in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl), en de sommen van zijn rijen zijn de graden. De berekening van [Backpropagation met de hand op een kleine graaf](/course/open/nn-backward-by-hand?lang=nl) was een gerichte graaf, met een pijl van elke waarde naar de waarden die ervan gemaakt worden. Grafen komen terug in de domeinen: netwerken van vluchten, aanbevelingen uit wat mensen samen kochten, en neural networks die op grafen leren. In de volgende les, *Een language model voorspelt het volgende token*, leert een model uit alleen tekst welk token er volgt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_column, result_list, result_matrix):
    L.wrap_up(result_list, result_matrix, choice_column)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
