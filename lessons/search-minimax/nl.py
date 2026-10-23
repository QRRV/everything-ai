# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les search-minimax. Route: course/route/lessen/overige-takken.yaml. Checks, de posities en het speelblok: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("search-minimax", "nl")
    C = L.checks   # de posities, en de eigen minimax van de course die het speelblok tekent


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl) en [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_state():
    quick_state = L.choice(
        "quick-state",
        question="Wat is in het doolhof van *A\\*: zoeken met richtinggevoel* een state?",
        options={
            "maze": "Het hele doolhof, met zijn muren",
            "right": "Het vakje waar je staat, als `(row, column)`",
            "path": "Het pad dat je tot nu toe liep",
        },
        correct="right",
        why={
            "right": "Een state is één situatie van de taak; in het doolhof: waar je staat. De actions leiden van de ene state naar de volgende.",
            "maze": "Het doolhof blijft tijdens de search hetzelfde; wat verandert, is waar je staat. Zie [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl).",
            "path": "Een pad is een lijst states, één action per stap; een state is er één van. Zie [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl).",
        },
    )
    quick_state
    return (quick_state,)


@app.cell(hide_code=True)
def quick_check_recursion():
    quick_recursion = L.choice(
        "quick-recursion",
        question="Wat laat dit programma zien?\n\n```python\ndef total(xs):\n    if len(xs) == 0:\n        return 0\n    return xs[0] + total(xs[1:])\n\nprint(total([4, 1, 2]))\n```",
        options={
            "first": "`4`",
            "right": "`7`",
            "base": "`0`",
        },
        correct="right",
        why={
            "right": "`total([4, 1, 2])` is 4 + `total([1, 2])`, dus 4 + 1 + 2 + 0: de lege lijst is het basisgeval.",
            "first": "`total` stopt niet bij het eerste getal: hij telt er `total(xs[1:])` bij op, dezelfde vraag voor de rest van de lijst. Zie [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).",
            "base": "0 is wat het basisgeval teruggeeft, voor de lege lijst; op de weg terug omhoog telt elke aanroep zijn eigen eerste getal erbij op. Zie [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl).",
        },
    )
    quick_recursion
    return (quick_recursion,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In een doolhof zet alleen jij een stap, dus een pad naar het doel is een plan. In een spel zet een tegenstander ook, en die wil het tegenovergestelde van wat jij wilt: na elke zet van jou kiest hij. Hoe kies je een zet als de ander bepaalt wat er daarna gebeurt?

        **Een voorbeeld uit het dagelijks leven.** Je speelt boter-kaas-en-eieren met een vriendin. Voordat je je X neerzet, denk je: als ik hier zet, waar zet zij dan? En ik daarna? Je rekent niet op een fout: je verwacht haar beste antwoord.

        Hier zijn de vakjes genummerd van 0 tot 8, rij voor rij, en X is aan zet.
        """),
        C.squares_and_game(("De vakjes", "X aan zet"), "Twee borden van 3 bij 3 vakjes. Het eerste laat het nummer van elk vakje zien, 0 tot 8 rij voor rij: 0, 1 en 2 bovenaan, 6, 7 en 8 onderaan. Het tweede is een spel waarin X aan zet is: bovenste rij X, O, X; middelste rij O en twee lege vakjes, 4 en 5; onderste rij O, X en het lege vakje 8."),
    ])
    return


@app.cell(hide_code=True)
def predict_end():
    guess_end = L.guess(
        "end",
        question="Hoe loopt het spel af als beide spelers vanaf deze positie hun beste zetten doen?",
        options={
            "win": "X wint",
            "draw": "Gelijkspel",
            "loss": "O wint",
        },
        reveal="**X wint**, met vakje 8; met vakje 4, in het midden, wordt het gelijkspel. *Minimax*, de search van deze les, rekent dat voor elke positie uit, en het speelblok hieronder laat zien hoe.",
    )
    guess_end
    return (guess_end,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een spel waarin twee spelers om de beurt een zet doen, is een search problem met een tegenstander. Een state is een positie: de tekens op het bord, en wie er aan zet is. Een action is een zet: hier een teken in een leeg vakje. Een positie waarin iemand drie op een rij heeft, of waarin het bord vol is, beëindigt het spel.

        Alle posities die je vanuit één positie kunt bereiken, zet voor zet tot elk spel afgelopen is, vormen de **game tree**: die positie bovenaan, en elk niveau lager één beurt. Elk afgelopen spel heeft een waarde: +1 als X won, −1 als O won, 0 bij gelijkspel. X wil de waarde zo groot mogelijk, O zo klein mogelijk.

        **Minimax** geeft elke positie de waarde die de speler aan zet kan afdwingen, als de ander ook zijn best speelt. Is X aan zet, dan is dat de grootste waarde van de posities na de zetten van X; is O aan zet, dan de kleinste. Elk van die waarden is dezelfde vraag één zet verder, dus minimax roept zichzelf aan, en de afgelopen spellen zijn het basisgeval.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **game tree** | alle posities die je vanuit één positie kunt bereiken, zet voor zet; elk niveau is de beurt van één speler | |
        | **minimax** | een search die elke positie de waarde geeft die de speler aan zet kan afdwingen: voor X de grootste waarde na zijn zetten, voor O de kleinste | `minimax(board, player)` |
        """),
        L.watch_out("*Waarde* is hier een gewoon woord: de waarde van een positie is hoe het spel vanaf daar afloopt als beide spelers hun best spelen, +1, 0 of −1."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Minimax op de positie van de gok, aanroep voor aanroep. Onder elke positie staat haar waarde, een `?` zolang minimax die niet weet, en het vakje van de laatste zet is grijs. Een gestippelde ring is een aanroep die wacht op de waarden eronder, de dikke ring de positie waarvan de waarde net terugkwam, en een dikke lijn de zet die de speler daar doet. Het blok begint bij de laatste stap.
    """)
    return


@app.cell(hide_code=True)
def explore_tree_cell():
    _rows = ["X aan zet", "O aan zet", "X aan zet", "Afgelopen"]

    def _listed(items):
        return ", ".join(items[:-1]) + " en " + items[-1]

    def _say(k):
        if k == 0:
            return "Stap 0: minimax wordt aangeroepen op de positie bovenaan. Voordat hij een waarde weet, roept hij zichzelf aan op de positie na elke zet, tot aan de spellen die afgelopen zijn."
        _i = C.BACK[k - 1]
        _n = C.NODES[_i]
        _value = C.shown(_n["value"])
        if not _n["kids"]:
            _w = C.winner(_n["board"])
            if _w is None:
                return f"Stap {k}: het bord is vol zonder drie op een rij. Gelijkspel: waarde {_value}."
            return f"Stap {k}: {_w} heeft drie op een rij. Het spel is afgelopen: waarde {_value}."
        if len(_n["kids"]) == 1:
            return f"Stap {k}: {_n['player']} heeft nog één leeg vakje, dus deze positie is evenveel waard als die zet: {_value}."
        _values = [C.shown(C.NODES[_c]["value"]) for _c in _n["kids"]]
        _square = C.NODES[C.chosen(_i)]["square"]
        if _n["player"] == "X":
            _said = f"Stap {k}: X is aan zet, en neemt de grootste van {_listed(_values)}: {_value}, met vakje {_square}."
        else:
            _said = f"Stap {k}: O is aan zet, en neemt de kleinste van {_listed(_values)}: {_value}, met vakje {_square}."
        if _n["parent"] is None:
            _said += " Dat is de waarde van de positie bovenaan: X wint."
        return _said

    def _frame(k):
        k = int(k)
        _known = len(C.step(k)["known"])
        _alt = (f"De game tree van de positie van de gok, {len(C.NODES)} posities op 4 niveaus, na stap {k}: "
                f"van {_known} is de waarde bekend.")
        if k == C.STEPS:
            _alt += " De positie bovenaan is +1 waard, met X in vakje 8."
        return C.tree_picture(k, _rows, _alt).text, _say(k)

    explore_tree = L.explore("tree", "stap", _frame, start=C.STEPS, low=0, high=C.STEPS, step=1, role="",
                             task="Schuif terug naar stap 0, en volg de waarden omhoog. Waarom is X in vakje 4 maar 0 waard?")
    explore_tree
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Met vakje 4 maakt X één lijn met twee X'en, en die blokkeert O met vakje 8: O neemt de kleinste waarde, 0, en niet de +1 van het spel waarin hij niet blokkeert. Met vakje 8 maakt X twee zulke lijnen, en O kan er maar één blokkeren: zijn beide antwoorden zijn +1 waard. X neemt de grootste van 0, 0 en +1.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Minimax, de waarde van een positie waarin `player` aan zet is:

    1. Is het spel afgelopen? Geef dan zijn waarde terug: 1 als X drie op een rij heeft, −1 als O dat heeft, 0 als het bord vol is.
    2. Zo niet, probeer dan elk leeg vakje als zet, en vraag minimax naar de waarde van de positie na die zet, met de andere speler aan zet.
    3. Is X aan zet, geef dan de grootste van die waarden terug; is O aan zet, de kleinste.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een bord is een tekst van 9 tekens, rij voor rij, met `.` voor een leeg vakje: `"XOXO..OX."` is de positie van de gok. `_winner` kijkt naar de 8 lijnen van drie vakjes; `for a, b, c in _lines` geeft de drie vakjes van elke lijn een eigen naam, zoals `x, y = point` dat doet. `board[:i] + player + board[i + 1:]` is het bord met `player` in vakje `i`: de tekst ervoor, het teken, en de tekst erna. `"." not in board` is `True` als er geen leeg vakje meer is. `max(values)` geeft het grootste getal van de lijst `values`, en `min(values)` het kleinste. `_minimax` volgt het recept; de loop aan het eind laat de waarde van elke zet van X zien. Voer de cel uit.
    """)
    return


@app.cell
def example_minimax():
    # de 8 lijnen van drie vakjes: de rijen, de kolommen en de twee diagonalen
    _lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def _winner(board):
        # de speler met drie op een rij, of None
        for a, b, c in _lines:
            three = board[a] + board[b] + board[c]
            if three == "XXX":
                return "X"
            if three == "OOO":
                return "O"
        return None

    def _minimax(board, player):
        # 1. een spel dat afgelopen is: zijn waarde
        w = _winner(board)
        if w == "X":
            return 1
        if w == "O":
            return -1
        if "." not in board:
            return 0
        # 2. de waarde na elke zet, met de andere speler aan zet
        if player == "X":
            other = "O"
        else:
            other = "X"
        values = []
        for i in range(9):
            if board[i] == ".":
                after = board[:i] + player + board[i + 1:]
                values.append(_minimax(after, other))
        # 3. X neemt de grootste waarde, O de kleinste
        if player == "X":
            return max(values)
        return min(values)

    _board = "XOXO..OX."
    for _i in range(9):
        if _board[_i] == ".":
            _after = _board[:_i] + "X" + _board[_i + 1:]
            print("X in vakje", _i, "is waard:", _minimax(_after, "O"))
    print("de positie is waard:", _minimax(_board, "X"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    X in vakje 4 of 5 is 0 waard, en X in vakje 8 is 1 waard: de grootste, dus de positie is 1 waard, zoals in het speelblok. Voor die laatste regel riep `_minimax` zichzelf 15 keer aan: één keer voor elke positie onder de top van de game tree.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minimax():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je eigen minimax

        `minimax(board, player)` hieronder is `_minimax` uit het uitgewerkte voorbeeld, met `LINES` en `winner` erboven. Vervang de drie `...`: in stap 2 van het recept de waarde van de positie na de zet, en in stap 3 de waarde die X neemt en de waarde die O neemt. De check roept eerst `minimax("XOXO..OX.", "X")` aan, dat 1 moet geven, en daarna negen andere posities, met X of O aan zet, waarvan drie spellen die al afgelopen zijn. Geen ervan heeft meer dan 6 lege vakjes, zodat de check snel klaar is: op het lege bord doet minimax 549.946 aanroepen.
        """),
        L.task("Laat `minimax` elke positie de waarde geven die de speler aan zet kan afdwingen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minimax():
    box_minimax = L.code_box("minimax", lines=30)
    box_minimax
    return (box_minimax,)


@app.cell(hide_code=True)
def check_minimax(box_minimax):
    result_minimax = L.check("minimax", box_minimax.value)
    result_minimax
    return (result_minimax,)


@app.cell(hide_code=True)
def hints_minimax():
    L.hints(
        "minimax",
        [
            "Stap 2 en 3 van het recept. Wie is er aan zet in de positie `after`?",
            "De waarde van `after` is een vraag voor `minimax` zelf, met `other` aan zet. X wil de grootste waarde, O de kleinste.",
            "`values.append(minimax(___, ___))`, dan `return max(___)` en `return ___(values)`.",
        ],
        why="`minimax(after, other)` is de waarde van de positie na de zet, met de andere speler aan zet: dezelfde vraag één zet verder, tot een spel afgelopen is. Daarna neemt X de grootste van die waarden, `max(values)`, en O de kleinste, `min(values)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een game tree met de hand

        In deze game tree kiest X a, b of c, en daarna kiest O een van drie antwoorden. Elk kader onderaan is een spel dat afgelopen is, met zijn waarde.
        """),
        C.small_tree(["X aan zet", "O aan zet", "Afgelopen"], "Een game tree met X aan zet bovenaan. Daaronder drie posities, a, b en c, met O aan zet, en onder elk drie spellen die afgelopen zijn: na a +1, +1 en −1; na b 0, +1 en 0; na c +1, −1 en 0. De waarden van a, b, c en de top zijn vraagtekens."),
    ])
    return


@app.cell(hide_code=True)
def choice_small_cell():
    choice_small = L.choice(
        "small-tree",
        question="Wat is de waarde van de positie bovenaan?",
        options={
            "plus": "+1",
            "right": "0",
            "minus": "−1",
        },
        correct="right",
        why={
            "right": "O neemt onder elk van zijn posities de kleinste waarde: −1 na a, 0 na b, −1 na c. X neemt daarvan de grootste: 0, met b. Onder a en c ligt een +1, maar O laat X daar nooit komen.",
            "plus": "Onder a en onder c ligt een +1, maar daar kiest O, en O neemt de kleinste waarde. Reken eerst de waarde van a, b en c uit, zoals in stap 3 van het recept.",
            "minus": "−1 is wat O neemt na a en na c. Maar X kiest eerst, en neemt de zet met de grootste waarde: b, waar O niet beter kan dan 0.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_exercise_chess():
    mo.md(r"""
    ### Oefening 3. En schaken?
    """)
    return


@app.cell(hide_code=True)
def choice_chess_cell():
    choice_chess = L.choice(
        "chess",
        question="Op het lege bord geeft minimax 0, na 549.946 aanroepen: met de beste zetten eindigt boter-kaas-en-eieren in gelijkspel. Waarom rekent een schaakprogramma de waarde van een schaakpositie niet op dezelfde manier uit?",
        options={
            "outcomes": "Een schaakpartij heeft meer uitkomsten dan winst, gelijkspel en verlies",
            "right": "De game tree is veel te groot: het programma kijkt maar een paar zetten vooruit, en schat hoe goed de posities daar zijn",
            "board": "Minimax werkt alleen op een bord van 3 bij 3 vakjes",
        },
        correct="right",
        why={
            "right": "In een schaakpositie heeft een speler zo'n 35 zetten, dus 10 zetten verder heeft de game tree onderaan al $35^{10}$ posities, bijna 3 miljoen miljard. Een schaakprogramma stopt na een paar zetten, geeft elke positie daar een getal uit een *evaluation function*, een schatting van wie er voorstaat, en doet minimax op die getallen.",
            "outcomes": "Een schaakpartij eindigt ook in winst, gelijkspel of verlies, dus +1, 0 en −1 zouden volstaan. Het probleem is de grootte van de game tree.",
            "board": "Minimax heeft alleen de zetten in elke positie nodig en de waarde van een afgelopen spel; die heeft schaken ook. Het probleem is de grootte van de game tree.",
        },
    )
    choice_chess
    return (choice_chess,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Minimax doorzoekt een game tree zoals [breadth-first search](/course/open/search-bfs?lang=nl) een doolhof doorzocht, maar in een andere volgorde: hij volgt elke zet tot het eind van het spel voordat hij de volgende probeert, zoals in het speelblok. Hij geeft een waarde terug in plaats van een pad, en hij roept zichzelf aan, zoals in [Een functie die zichzelf aanroept](/course/open/py-recursion?lang=nl). Met een waarde voor elke positie is de beste zet die waarvan de waarde het best is voor de speler aan zet. Zoeken in spellen als schaken komt terug in een tak die nog komt, *Zoeken, logica, planning en optimalisatie*. In de volgende les, *Als stappen kunnen uitglijden: value iteration*, antwoordt er geen tegenstander op je action, maar het toeval, en krijgt elke state weer een waarde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chess, choice_small, result_minimax):
    L.wrap_up(result_minimax, choice_small, choice_chess)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
