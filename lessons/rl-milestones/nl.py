# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-milestones. Route: course/route/lessen/reinforcement-learning.yaml. Checks, het spel met 21 stokjes van de course met de self-play, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-milestones", "nl")
    C = L.checks   # het spel met 21 stokjes van de course, de self-play, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [RL buiten games](/course/open/rl-real-world?lang=nl) en [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_unseen():
    quick_unseen = L.choice(
        "quick-unseen",
        question="Een agent heeft alleen in een simulator geleerd, en komt in de echte wereld terecht in een state die de simulator nooit liet zien. Wat staat er in de Q-table bij deze state?",
        options={
            "similar": "De waarden van de state die er het meest op lijkt",
            "right": "De nullen waarmee de tabel begon",
            "ask": "Waarden die de agent aan de simulator vraagt",
            "none": "Geen rij: de agent stopt",
        },
        correct="right",
        why={
            "right": "Een rij van de tabel verandert alleen als de eigen state langskomt, dus houdt een state die nooit langskwam de waarden van het begin. De tabel in deze les begint ook met nullen, en leert de waarden door tegen zichzelf te spelen.",
            "similar": "Een Q-table heeft één rij per state, en elke rij verandert alleen als de eigen state langskomt. Waarmee begon de rij van een state die nooit langskwam? Zie [RL buiten games](/course/open/rl-real-world?lang=nl).",
            "ask": "Na de training kijkt de agent alleen in de tabel, en doet de simulator niet meer mee. Wat stond er voor de training in deze rij? Zie [RL buiten games](/course/open/rl-real-world?lang=nl).",
            "none": "Elke state heeft vanaf het begin een rij, en de agent kiest daar nog steeds een action. Met welke waarden begint de rij van een state die nooit langskwam? Zie [RL buiten games](/course/open/rl-real-world?lang=nl).",
        },
    )
    quick_unseen
    return (quick_unseen,)


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="De waarde van een state is 0,3, het target is 0,4 en de learning rate η is 0,5. Wat is de waarde na één update van TD(0)?",
        options={
            "target": "0,4: de waarde springt helemaal naar het target",
            "plus": "0,7: de waarde plus het target",
            "eta": "0,5: de learning rate zelf",
            "right": "0,35: halverwege van 0,3 naar het target",
        },
        correct="right",
        why={
            "right": "De update brengt de waarde een deel η van de weg naar het target: 0,3 + 0,5 × (0,4 − 0,3) = 0,35. In deze les komt het target van een positie uit de positie die de tegenstander krijgt.",
            "target": "Met η = 0,5 gaat de waarde slechts de helft van de weg. Hoeveel is de helft van het verschil tussen 0,3 en 0,4? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "plus": "De update telt η keer het verschil tussen het target en de waarde op, niet het target zelf. Hoe groot is dat verschil hier? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "eta": "η zegt welk deel van het verschil de waarde opschuift, niet wat de waarde wordt. Hoeveel is 0,3 + 0,5 × (0,4 − 0,3)? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Al meer dan zestig jaar laten games zien hoe ver RL is. Een game heeft duidelijke regels en een duidelijke winnaar, een computer kan zoveel partijen spelen als de tijd toelaat, en iedereen weet hoe sterk de beste menselijke spelers zijn. In 1959 beschreef Arthur Samuel een programma dat checkers leerde door tegen een kopie van zichzelf te spelen. In 2016 versloeg AlphaGo Lee Sedol, een van de sterkste Go-spelers ter wereld, met 4 tegen 1. Daartussen en daarna kwamen backgammon, Atari, schaken, StarCraft, Dota 2, racen en een robothand, en veel van deze programma's leerden van partijen tegen zichzelf. Deze les loopt de mijlpalen langs, en laat daarna één tabel een klein spel leren door beide kanten te spelen, om te zien hoeveel een programma ontdekt zonder dat iemand het iets leert.

    **Een voorbeeld uit het dagelijks leven.** Je kunt een bordspel leren door zelf beide kanten te spelen: je doet een zet voor de ene kant, draait het bord om en zoekt het beste antwoord voor de andere kant. Elke partij leert je iets over beide kanten, en je tegenstander is altijd precies even goed als jij, dus wordt je tegenstander beter als jij beter wordt. Zo leer je nooit een zet die geen van beide kanten ooit probeert.
    """)
    return


@app.cell(hide_code=True)
def predict_lost():
    guess_lost = L.guess(
        "lost",
        question="In het spel van de les liggen 21 stokjes op tafel; twee spelers pakken om de beurt 1, 2 of 3 stokjes, en winnaar is wie het laatste stokje pakt. Eén tabel, met een waarde voor elk aantal stokjes, speelt 200 partijen lang beide kanten. Bij welke aantallen stokjes komt de tabel onder 0 uit, een slechte positie voor de speler aan zet, denk je?",
        options={
            "none": "Nergens: met de goede zet is elke positie te winnen",
            "few": "Bij 1, 2 en 3: het eind is dichtbij, en de tegenstander kan winnen",
            "right": "Bij 4, 8, 12, 16 en 20",
            "far": "Bij de grote aantallen: ver van het eind kan alles nog",
        },
        reveal="**Bij 4, 8, 12, 16 en 20.** Na 200 partijen staat in de tabel −0,89 bij 4 stokjes, −0,56 bij 8, −0,52 bij 12, −0,48 bij 16 en −0,31 bij 20, en overal anders meer dan 0. De reden: bij 4 stokjes pakt de speler aan zet 1, 2 of 3, en pakt de tegenstander de rest van de 4 en wint. Bij 8 doet de tegenstander dat twee keer. Een veelvoud van 4 is verloren tegen een speler die dit weet, en elk ander aantal win je door de tegenstander een veelvoud van 4 te laten. Niemand gaf de tabel deze regel: de tabel vond de regel door tegen zichzelf te spelen.",
    )
    guess_lost
    return (guess_lost,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Zestig jaar mijlpalen.** Dit zijn de mijlpalen van RL in games, met waar elk programma van leerde.

        | Jaar | Programma | Taak | Leerde van | Schaal en resultaat |
        |---|---|---|---|---|
        | 1959 | het checkersprogramma van Arthur Samuel | checkers | partijen tegen een kopie van zichzelf, en een paar zetten vooruitkijken | draaide op een IBM 704; won in 1962 een partij van een sterke speler, die de revanche won |
        | 1992 | TD-Gammon (Gerald Tesauro) | backgammon | TD learning in een neural network, uit partijen tegen zichzelf | 1,5 miljoen partijen in de bekendste versie; dicht bij de beste menselijke spelers, niet beter |
        | 2015 | DQN (DeepMind) | 49 games van de Atari 2600 | alleen de pixels van het scherm en de score, met hetzelfde netwerk voor elke game | meer dan 75% van de score van een professionele gametester in 29 van de 49 games |
        | 2016 | AlphaGo (DeepMind) | Go | eerst partijen van menselijke experts, daarna partijen tegen zichzelf, met zoeken | versloeg Fan Hui, de Europees kampioen, met 5–0, en Lee Sedol met 4–1 |
        | 2017 | AlphaGo Zero | Go | alleen partijen tegen zichzelf, vanaf willekeurig spel | versloeg na 3 dagen de versie die Lee Sedol versloeg, met 100–0 |
        | 2018 | AlphaZero | schaken, shogi en Go | alleen partijen tegen zichzelf, met dezelfde methode voor alle drie de spellen | 5.000 TPU's, chips gemaakt voor neural networks, speelden de partijen; bij schaken was het programma Stockfish na 4 uur ingehaald |
        | 2019 | AlphaStar (DeepMind) | StarCraft II | eerst menselijke partijen, daarna een league van agents die tegen elkaar speelden | Grandmaster, beter dan 99,8% van de gerangschikte menselijke spelers |
        | 2019 | OpenAI Five | Dota 2, vijf tegen vijf | partijen tegen zichzelf | 250 jaar aan partijen per dag; versloeg OG, de wereldkampioen, met 2–0 |
        | 2019 | een robothand (OpenAI) | een Rubiks kubus oplossen | een simulator die steeds de grootte van de kubus, de wrijving en meer veranderde | loste 60% op van de kubussen die 15 draaien nodig hadden; een vaste oplosser koos de draaien |
        | 2022 | GT Sophy (Sony AI) | racen in Gran Turismo | races in de game, met een reward die onsportieve botsingen bestrafte | verloor in juli 2021 een teamwedstrijd van topcoureurs met 70–86, en won de revanche in oktober met 104–52 |

        **Wat de mijlpalen gemeen hebben.** Elke mijlpaal combineerde leren van de eigen partijen met iets wat de partijen liet tellen: vooruitkijken naar zetten voordat het programma er een kiest (Samuel, AlphaGo, AlphaZero), een neural network dat ook nooit geziene posities beoordeelt (vanaf TD-Gammon), of een enorm aantal partijen (OpenAI Five). Later in de course krijgt zoeken eigen lessen: zetten van tevoren uitproberen. En de meeste programma's leerden van partijen tegen zichzelf: **self-play**. Self-play heeft geen menselijke partijen nodig en raakt nooit door de tegenstanders heen, en de tegenstander is altijd precies even sterk als de speler, dus wordt de tegenstander sterker als de speler sterker wordt.

        **Self-play in het klein.** Het spel van de les is zo klein als een spel voor twee kan zijn: 21 stokjes, elke speler pakt om de beurt 1, 2 of 3, en winnaar is wie het laatste stokje pakt. Eén tabel `V` bevat voor elk aantal stokjes van 0 tot en met 21 hoe goed de positie is voor de speler aan zet: 1 is een zekere overwinning, −1 een zekere nederlaag. `V[0]` is −1, want als er geen stokjes meer liggen, heeft de tegenstander net het laatste gepakt. Dezelfde tabel speelt beide kanten:

        - **Kiezen.** Met `sticks` stokjes laat elke zet de tegenstander een positie, en de beste zet laat de positie met de laagste waarde, want deze positie is het slechtst voor de tegenstander. Met kans ε een willekeurige zet.
        - **Leren.** Na een zet van `sticks` naar `left` stokjes doet `V[sticks]` een TD-stap richting de waarde van de positie van de tegenstander, met het teken omgedraaid: goed voor de tegenstander betekent slecht voor jou.
        """),
        mo.Html(C.tables_picture(L.lang)),
        mo.md(r"""
        Voor de eerste partij is elke waarde 0 behalve `V[0]`. De tabel pakt bij 2 en 3 stokjes alles, want dat wint meteen, en overal anders 1 stokje: bij slechts 8 van de 16 winbare posities goed. Na 200 partijen tegen zichzelf, met η 0,1 en ε 0,1, staan de waarden onder 0 bij precies de veelvouden van 4, en pakt de tabel bij elke andere positie wat de tegenstander een veelvoud laat.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **self-play** | | een agent trainen door de agent tegen kopieën van zichzelf te laten spelen, zodat de tegenstander even hard sterker wordt | één `V` voor beide spelers |
        | **AlphaZero** | | een programma van DeepMind (2018) dat schaken, shogi en Go leerde door alleen self-play, met zoeken en een neural network dat zetten voorstelt en posities beoordeelt | |
        """),
        L.watch_out("De tabel kon dit spel leren omdat het spel piepklein is: 22 waarden, en elke positie komt keer op keer langs. Schaken heeft meer posities dan er ooit langs kunnen komen, dus gebruikte AlphaZero een neural network dat ook nooit geziene posities beoordeelt, en keek het programma voor elke zet vooruit met zoeken."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Self-play in een spel voor twee:

    1. Eén tabel, of één netwerk, met voor elke positie hoe goed de positie is voor de speler aan zet.
    2. Kies de zet die de tegenstander de slechtste positie laat, en met kans ε een willekeurige zet.
    3. Laat na elke zet de waarde van je eigen positie een stap zetten richting de waarde van de tegenstander, met het teken omgedraaid.
    4. Laat dezelfde tabel beide kanten spelen, partij na partij.
    5. Test de tabel tegen een speler van buiten de eigen partijen: een perfecte speler, een oudere versie, of mensen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt de tabel van de course na 200 partijen self-play, voor 0 tot en met 7 stokjes, en bekijkt de drie zetten vanaf 7 stokjes: hoeveel stokjes elke zet overlaat, en hoe goed deze positie is voor de tegenstander. Voer de cel uit.
    """)
    return


@app.cell
def example_moves():
    # de tabel van de course na 200 partijen self-play, voor 0 tot en met 7 stokjes
    V = [-1.0, 0.99, 0.95, 0.95, -0.89, 0.5, 0.86, 0.85]
    sticks = 7
    for take in (1, 2, 3):
        left = sticks - take
        print("take", take, "leaves", left, "sticks, worth", V[left], "to the opponent")
    return V, left, sticks, take


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met 1 of 2 pakken krijgt de tegenstander een positie met waarde 0,86 of 0,5: allebei goed voor de tegenstander. Met 3 pakken blijven er 4 stokjes over, met waarde −0,89 voor de tegenstander, een verloren positie, dus is dat de zet. En `V[7]` zelf, 0,85, ligt dicht bij wat deze zet waard is voor de speler aan zet, 0,89: de −0,89 van de tegenstander met het teken omgedraaid.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een zet kiezen

        `choose(V, sticks)` geeft de zet voor de speler aan zet bij `sticks` stokjes, met de tabel `V`. `best` begint op 1, en de loop probeert `take` = 1, 2 en 3; de test `take <= sticks` zorgt dat een zet nooit meer stokjes pakt dan er liggen. Daarna wordt `best` gelijk aan `take` als deze zet de tegenstander een positie laat met een lagere waarde dan de positie die `best` laat. Vervang de `...` door deze test.

        De check roept `choose` aan met een tabel voor 0 tot en met 7 stokjes, `[-1.0, 0.9, 0.7, 0.5, -0.4, 0.3, 0.1, 0.2]`, en 6 stokjes: 1, 2 of 3 pakken laat de tegenstander 5, 4 of 3 stokjes, met waarde 0,3, −0,4 en 0,5 voor de tegenstander. Daarna vijf andere gevallen: 1, 2, 3 en 7 stokjes met dezelfde tabel, en 7 stokjes met een tabel met nullen, waar alle drie de zetten even goed lijken en `best` 1 moet blijven.
        """),
        L.task("Kies de zet die de tegenstander de laagste waarde laat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=6)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose",
        [
            "Stap 2 van het recept. De regel begint met `if` en de test dat `take` hooguit `sticks` is; na `and` komt de test die twee waarden van `V` vergelijkt: van de positie die deze zet de tegenstander laat, `sticks - take` stokjes, en van de positie die `best` laat, `sticks - best` stokjes.",
            "In het geval van de check, bij 6 stokjes, begint `best` op 1, wat 5 stokjes overlaat, met waarde 0,3. 2 pakken laat 4 over, met waarde −0,4: lager, dus wordt `best` 2. 3 pakken laat 3 over, met waarde 0,5: hoger dan −0,4, dus blijft `best` 2.",
            "`        if take <= sticks and ___:`",
        ],
        why="`if take <= sticks and V[sticks - take] < V[sticks - best]:` houdt de zet met de positie die het slechtst is voor de tegenstander. De strikte `<` verandert `best` alleen bij een echt betere zet, dus blijft bij zetten die even goed lijken de eerste staan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_learn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Leren van de positie van de tegenstander

        `learn(V, sticks, left, eta)` geeft de nieuwe waarde van `V[sticks]` na een zet van `sticks` naar `left` stokjes, met learning rate `eta`, η. De `return`-regel is de stap van TD(0): `V[sticks]` schuift een deel `eta` van de weg op richting `target`. Het target is de waarde van de positie die de tegenstander krijgt, gezien vanaf jouw kant: goed voor de tegenstander betekent slecht voor jou. Vervang de `...` door het target.

        De check roept `learn` aan met dezelfde tabel: vanaf 5 stokjes pakte de zet 1 en liet de tegenstander 4, met η 0,5. Daarna vijf andere gevallen: de laatste 3 stokjes pakken, zodat `left` 0 is; twee zetten die 4 overlaten, vanaf 7 en vanaf 6 stokjes; het laatste stokje met η 1; en een zet vanaf 4 stokjes met een tabel met nullen.
        """),
        L.task("Maak van `target` de waarde van de tegenstander, gezien vanaf jouw kant."),
    ])
    return


@app.cell(hide_code=True)
def exercise_learn():
    box_learn = L.code_box("learn", lines=3)
    box_learn
    return (box_learn,)


@app.cell(hide_code=True)
def check_learn(box_learn):
    result_learn = L.check("learn", box_learn.value)
    result_learn
    return (result_learn,)


@app.cell(hide_code=True)
def hints_learn():
    L.hints(
        "learn",
        [
            "Stap 3 van het recept: `target` is de waarde van de positie die de tegenstander krijgt, bij `left` stokjes, gezien vanaf jouw kant, waar een goede positie voor de tegenstander als slecht telt.",
            "In het geval van de check zijn 4 stokjes −0,4 waard voor de tegenstander: een slechte positie voor de tegenstander, dus 0,4 waard voor jou. `V[5]` is 0,3, en met η 0,5 schuift de waarde halverwege op naar 0,4: 0,35.",
            "`    target = ___`",
        ],
        why="`target = -V[left]` maakt van de waarde van de tegenstander je eigen waarde. Met dat ene minteken kan één tabel beide kanten spelen: elke partij leert de tabel tegelijk iets vanaf de kant van de winnaar en van de speler die verloor.",
    )
    return


@app.cell(hide_code=True)
def text_after_learn():
    mo.md(r"""
    In het begin leren alleen de zetten aan het eind van een partij iets, want `V[0]` is de enige waarde die niet 0 is: bij het pakken van de laatste stokjes schuift de waarde op richting 1. Partij na partij verspreiden de waarden zich terug vanaf het eind, zoals in [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl), en elke partij leert de tabel tegelijk iets over beide kanten.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Self-play, en daarna een partij tegen een perfecte speler

    De demo laat één tabel beide kanten van het spel spelen: je eigen `learn` en `choose` als elke functie de check haalt, en anders de versie van de course, met η 0,1. Kies hoeveel partijen de tabel tegen zichzelf speelt, en ε, de kans op een willekeurige zet. De grafiek telt na elke partij hoeveel van de 16 winbare posities de tabel nog fout speelt; ernaast de tabel op dat moment. Na de partijen speelt de tabel één partij tegen een perfecte speler, en begint bij 21 stokjes. Ga daarna terug door de partijen.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    games_choice = mo.ui.dropdown(options=C.games_options(L.lang), value="200", label=C.DEMO_TEXT[L.lang]["games"])
    eps_choice = mo.ui.dropdown(options=C.eps_options(L.lang), value=C.number(C.EPSILON, 1, L.lang),
                                label=C.DEMO_TEXT[L.lang]["eps"])
    play_button = mo.ui.run_button(label="Speel tegen zichzelf")
    mo.hstack([games_choice, eps_choice, play_button], justify="start", gap=1.5, wrap=True)
    return eps_choice, games_choice, play_button


@app.cell(hide_code=True)
def demo_play(box_choose, box_learn, eps_choice, games_choice, play_button, result_choose, result_learn):
    if not play_button.value:
        _out = mo.md("*Kies de partijen en ε, en druk dan op de knop om de tabel tegen zichzelf te laten spelen.*")
    else:
        _mine_learn = result_learn.state == "correct"
        _learn = run_code(box_learn.value, {}).names["learn"] if _mine_learn else C.learn
        _mine_choose = result_choose.state == "correct"
        _choose = run_code(box_choose.value, {}).names["choose"] if _mine_choose else C.choose
        _games, _eps = games_choice.value, eps_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), _games, role="loss", low=0, high=16, window=1,
                       xticks=C.demo_xticks(_games), key=C.chart_key(L.lang))
        _shown = [-1]

        def _progress(_run):
            _live.watch(lambda: C.now_picture(_run, L.lang, _choose))
            _new = [(_g, _run.wrong[_g]) for _g in range(_shown[0] + 1, _run.games + 1)]
            _shown[0] = _run.games
            _live.extend(_new, _w["playing"].format(g=C.number(_run.games, 0, L.lang), n=C.number(_games, 0, L.lang),
                                                    wrong=_run.wrong[-1]))

        _run = C.demo_run(_learn, _choose, _games, _eps, show=_progress, pause=0.02)
        if _run == "error":
            _out = _live.stopped("*Je `learn` of `choose` gaf een fout, of iets anders dan een getal of een zet. Halen beide functies nog de check?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "partij", lambda _v: C.rewind_frame(_run, int(_v), L.lang, _choose), start=_games,
                          values=list(range(0, _games + 1)), role="",
                          task="Ga terug naar partij 0, en dan vooruit: na welke partij speelt de tabel elke winbare positie goed?"),
                mo.Html(C.game_picture(_run, L.lang, _choose)),
                mo.md(C.verdict(_run, _mine_learn, _mine_choose, _eps, L.lang, _choose)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met ε 0,1 en 200 partijen speelt de tabel vanaf partij 98 elke winbare positie goed, en verslaat de tabel de perfecte speler: de tabel pakt 1 van de 21 stokjes, en pakt na elk antwoord wat de tegenstander weer op een veelvoud van 4 zet. Na 25 of 50 partijen speelt de tabel nog 8 of 4 posities fout, en wint de perfecte speler. Twee dingen zijn misschien verrassend. Met ε 0, dus zonder willekeurige zetten, speelt de tabel vanaf partij 16 alles goed: in dit piepkleine spel verlaagt een verloren partij de waarde van de zetten waarmee de tabel verloor, dus probeert de tabel de volgende keer vanzelf een andere zet, en voegen willekeurige zetten alleen ruis toe. Met ε 0,3 speelt de tabel bij partij 92 alles goed, zakt daarna terug, en blijft pas vanaf partij 118 goed, want 3 van elke 10 zetten zijn willekeurig en duwen de waarden steeds heen en weer. Runs met een andere seed en ε 0,1 hebben tussen 21 en 66 partijen nodig, en elke run eindigt met de waarden onder 0 bij precies de veelvouden van 4. In een groot spel ziet een programma de meeste posities nooit, en kan het programma steeds dezelfde lijn blijven spelen; AlphaZero voegde tijdens self-play willekeurige ruis toe aan de zetkeuzes van het zoeken, zodat ook zetten die het programma anders oversloeg een kans kregen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_from_rules():
    mo.md(r"""
    ### Oefening 3. Een nieuw spel

    Een team bedenkt een nieuw bordspel voor twee. Niemand speelt het nog goed, dus kan een programma niet leren van de partijen van sterke spelers.
    """)
    return


@app.cell(hide_code=True)
def choice_from_rules_cell():
    choice_from_rules = L.choice(
        "from_rules",
        question="Welke mijlpaal laat zien dat een programma het spel toch goed kan leren spelen?",
        options={
            "alphago": "AlphaGo, dat eerst menselijke experts nadeed en daarna tegen zichzelf speelde",
            "dqn": "DQN, dat 49 games leerde vanaf pixels",
            "right": "AlphaGo Zero en AlphaZero, die alleen tegen zichzelf speelden",
            "alphastar": "AlphaStar, dat Grandmaster werd",
        },
        correct="right",
        why={
            "right": "AlphaGo Zero en AlphaZero begonnen met willekeurig spel en leerden alleen van partijen tegen zichzelf, dus heeft een nieuw spel de regels en veel rekenkracht nodig, en verder niets. De tabel van deze les leerde ook van niets anders dan de regels.",
            "alphago": "AlphaGo leerde eerst de zetten van menselijke experts na te doen, en speelde pas daarna tegen zichzelf. Waar komen zulke partijen van experts vandaan in een spel dat niemand goed speelt?",
            "dqn": "DQN leerde games voor één speler, met een score op het scherm als maat voor hoe goed het ging. Wie zou de tegenstander zijn in een bordspel voor twee?",
            "alphastar": "AlphaStar begon ook met het nadoen van menselijke partijen van StarCraft II. Wat zou AlphaStar nadoen in een spel zonder sterke spelers?",
        },
    )
    choice_from_rules
    return (choice_from_rules,)


@app.cell(hide_code=True)
def text_exercise_league():
    mo.md(r"""
    ### Oefening 4. Een league van tegenstanders

    In StarCraft II is elke strategie te verslaan door een andere, zoals bij steen, papier, schaar. AlphaStar speelde niet alleen tegen de nieuwste versie van zichzelf: AlphaStar trainde in een league van eerdere versies en andere agents.
    """)
    return


@app.cell(hide_code=True)
def choice_league_cell():
    choice_league = L.choice(
        "league",
        question="Waarom de league?",
        options={
            "faster": "Tegen veel agents tegelijk spelen kost minder rekenkracht",
            "people": "De league bestond uit menselijke spelers, en mensen zijn moeilijker te verslaan",
            "right": "Zodat AlphaStar ook oude strategieën blijft verslaan",
            "rules": "StarCraft II staat niet toe dat een agent tegen zichzelf speelt",
        },
        correct="right",
        why={
            "right": "Tegen alleen de nieuwste versie van zichzelf kan een agent wisselen van steen naar papier naar schaar en terug, en onderweg vergeten hoe je wint van wat de agent niet meer tegenkomt. Een league houdt de oude strategieën als tegenstanders in de buurt, dus moet elke nieuwe versie ook van de oude strategieën blijven winnen.",
            "faster": "Een league betekent meer partijen, niet minder. Wat kan er misgaan als een agent alleen de nieuwste versie van zichzelf tegenkomt, in een spel waarin strategieën elkaar in een kring verslaan?",
            "people": "In de league zaten agents, geen mensen: eerdere versies van AlphaStar en andere agents. Wat riskeert een agent die alleen de nieuwste versie van zichzelf tegenkomt?",
            "rules": "Voor self-play zijn twee kopieën van hetzelfde programma genoeg, en elk spel voor twee laat dat toe. Waarom zou je dan eerdere versies als tegenstanders bewaren?",
        },
    )
    choice_league
    return (choice_league,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [RL buiten games](/course/open/rl-real-world?lang=nl) had een fout een prijs, dus leerde de agent in een simulator. In een game kost een fout een verloren partij en verder niets, en kan een programma miljoenen partijen tegen zichzelf spelen: zo droegen games RL van het checkersprogramma van Samuel tot AlphaZero. De tabel van deze les leerde zoals in [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl), met één verandering: het target komt uit de positie van de tegenstander, met het teken omgedraaid. Dit is de laatste les van de basis van RL. In maart 2025 werden Andrew Barto en Richard Sutton bekendgemaakt als winnaars van de Turing Award van 2024, de bekendste prijs in de informatica, voor de conceptuele en algoritmische basis van reinforcement learning. Het boek van Barto en Sutton, *Reinforcement Learning: An Introduction*, gaat dieper in op alles uit dit domein. Voorbij de basis vervangt RL de tabel door een neural network, zoals DQN en AlphaZero deden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_from_rules, choice_league, result_choose, result_learn):
    L.wrap_up(result_choose, result_learn, choice_from_rules, choice_league)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
