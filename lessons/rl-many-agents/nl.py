# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-many-agents. Route: course/route/lessen/reinforcement-learning.yaml. Checks, het spel, de runs van de course en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-many-agents", "nl")
    C = L.checks   # het spel, de runs van 20 paren van de course en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_transition():
    quick_transition = L.choice(
        "quick-transition",
        question="Wat zegt de transition function van een Markov decision process voor een state s en een action a?",
        options={
            "best": "Welke action het beste is in s",
            "right": "Wat er na a in s kan gebeuren, elke uitkomst met de kans en de reward",
            "value": "Hoeveel reward de agent vanaf s kan verwachten, als de agent vanaf daar goed handelt",
            "path": "De weg van s naar het doel",
        },
        correct="right",
        why={
            "right": "De transition function is de regel van de wereld: wat er op een action volgt, en hoe waarschijnlijk elke uitkomst is. In deze les zit er in de wereld van een agent een tweede agent: wat er op een action van A volgt, hangt af van wat B doet, en B blijft leren. Zo verandert de transition function van de wereld van A doordat B leert.",
            "best": "Welke action het beste is, rekent de agent uit, met value iteration of Q-learning; de transition function hoort bij de wereld. Wat zegt de wereld over een action? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "value": "Dat is de value van een state, en value iteration berekent deze uit de transition function. Wat zegt de transition function zelf? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "path": "De transition function kijkt één stap vooruit, niet helemaal tot een doel: wat zegt de transition function over deze ene stap? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
        },
    )
    quick_transition
    return (quick_transition,)


@app.cell(hide_code=True)
def quick_check_epsilon():
    quick_epsilon = L.choice(
        "quick-epsilon",
        question="Een agent met twee actions kiest ε-greedy, met ε = 0,05: met kans ε een willekeurige action van de twee, en anders de action met de grootste Q. In welk aandeel van de stappen neemt de agent de action met de kleinere Q?",
        options={
            "never": "Nooit: ε-greedy neemt altijd de grootste Q",
            "right": "In 2,5% van de stappen, ongeveer 1 op 40",
            "five": "In 5% van de stappen",
            "half": "In de helft van de stappen, want er zijn twee actions",
        },
        correct="right",
        why={
            "right": "In 5% van de stappen neemt de agent een willekeurige action, en de helft daarvan is toevallig toch de action met de grootste Q: 0,05 × ½ = 0,025. In deze les kiezen twee agents zo, dus verraadt elk van de twee de ander af en toe bij toeval, en moet de ander daarmee omgaan.",
            "never": "Met kans ε neemt de agent een willekeurige action van de twee, wat de Q's ook zeggen: hoe vaak is dat de action met de kleinere Q? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "five": "In 5% van de stappen is de action willekeurig, en zo'n willekeurige action van de twee kan ook de action met de grootste Q zijn: in welk aandeel van de stappen is het de andere? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "half": "Alleen de willekeurige stappen kunnen de kleinere Q nemen, en dat is 5% van alle stappen: welk aandeel daarvan is de action met de kleinere Q? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_epsilon
    return (quick_epsilon,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2019 lieten Bowen Baker en collega's van OpenAI agents verstoppertje spelen in een kleine wereld met muren, dozen en hellingen. De verstoppers kregen bij elke stap +1 als geen enkele verstopper te zien was voor de zoekers, en −1 als de zoekers er één zagen; de zoekers kregen het omgekeerde. Niemand beloonde het gebruik van een doos of een helling. Eerst was het alleen wegrennen en achternazitten. Na ongeveer 25 miljoen episodes leerden de verstoppers dozen tegen elkaar te schuiven tot een schuilplaats. Ongeveer 75 miljoen episodes later leerden de zoekers een helling naar de schuilplaats te brengen en over de muur te klimmen, en al snel leerden de verstoppers eerst de hellingen weg te halen. Nog later leerden de zoekers op een doos te klimmen en ermee naar de schuilplaats te rijden, en leerden de verstoppers elke doos vast te zetten voordat de zoekers kwamen. Niemand programmeerde een van deze strategieën: elke strategie ontstond doordat het andere team iets geleerd had, en elke keer dat het ene team leerde, veranderde de wereld van het andere team. In deze les zie je hetzelfde op een schaal die je kunt volgen: twee agents, twee actions, en een spel waarin beide er baat bij hebben om samen te werken.

    **Een voorbeeld uit het dagelijks leven.** In een gedeelde keuken op het werk kost je eigen kopje omspoelen een minuut. Wat de anderen ook doen, je kopje in de gootsteen laten staan scheelt je deze minuut. Laat iedereen het eigen kopje staan, dan is de keuken voor iedereen een rommel, en dat kost elk van de collega's meer dan de minuut die elk bespaarde.
    """)
    return


@app.cell(hide_code=True)
def predict_pairs():
    guess_pairs = L.guess(
        "pairs",
        question="Twee agents, A en B, spelen het spel van de afbeelding: in elke ronde kiezen beide tegelijk, samenwerken of verraden. Een game is 100 rondes, en Q-learning traint beide agents 300 games lang, elk met een eigen Q-table. Elke agent onthoudt de laatste ronde: de state is wat beide in deze ronde deden. Na de training nemen beide de action met de grootste Q. In hoeveel van 20 zulke paren, elk getraind met eigen willekeurige getallen, werken beide agents in elke ronde van een game samen, denk je?",
        options={
            "none": "Geen enkel paar: verraden levert meer op, wat de ander ook doet",
            "five": "Ongeveer 5 van de 20",
            "right": "14 van de 20",
            "all": "Alle 20: beide leren dat samenwerken meer oplevert",
        },
        reveal="**14 van de 20.** Zonder geheugen doet geen enkel paar het: elke agent heeft dan slechts één state, en beide leren verraden. Met geheugen werken 14 paren in elke ronde samen, en in 13 daarvan leerden beide agents dezelfde regel, zonder dat iemand de regel opschreef: samenwerken na een ronde waarin beide hetzelfde deden, en verraden na een ronde waarin de twee verschillende zetten deden.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_pairs])
    return (guess_pairs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het spel.** In het **prisoner's dilemma** kiezen twee spelers tegelijk, zonder te weten wat de ander kiest: samenwerken of verraden. In code heten de zetten `"cooperate"`, samenwerken, en `"betray"`, verraden. De naam komt van een verhaal over twee verdachten die apart verhoord worden en elk kunnen zwijgen of de ander verraden. In de tabel van deze les, `PAYOFF`, krijgen beide 3 als beide samenwerken, en 1 als beide verraden; verraadt de een en werkt de ander samen, dan krijgt wie verraadt 5 en de ander 0. `PAYOFF` heeft een paar zetten als key en geeft de reward van de speler met de eerste zet in het paar: `PAYOFF[("betray", "cooperate")]` is 5, en `PAYOFF[("cooperate", "betray")]` is 0.

        **Verraden loont, wat de ander ook doet.** Werkt B samen, dan krijgt A 5 voor verraden en 3 voor samenwerken. Verraadt B, dan krijgt A 1 voor verraden en 0 voor samenwerken. In beide gevallen levert verraden A meer op, en voor B geldt hetzelfde. Toch krijgen beide 1 als beide verraden, terwijl allebei samenwerken elk 3 zou geven. Pakt de keuze die voor elke agent apart het beste is slechter uit voor alle agents samen, dan is dat een **sociaal dilemma**; het prisoner's dilemma is het bekendste, voor twee spelers.

        **Twee lerende agents in één wereld.** **Multi-agent reinforcement learning**, kortweg MARL, is reinforcement learning met meerdere agents in één environment, waarbij elke agent leert terwijl de andere agents ook leren. De eenvoudigste aanpak, en de aanpak van deze les: maak van elke agent een eigen Q-learner, zoals in *Leren van je eigen stappen: Q-learning*, met een eigen Q-table, eigen rewards, en de andere agent als gewoon een deel van de wereld. Dat deel blijft niet hetzelfde. Voor A hangt wat een verraad oplevert af van wat B doet, en B verandert door te leren: dezelfde action in dezelfde state kan vandaag 5 opleveren en honderd games later 1. In elke eerdere les bleef de wereld hetzelfde terwijl de agent leerde; hier verandert de wereld doordat de andere agent leert.

        **Rondes en games.** In een ronde kiezen beide agents ε-greedy, en krijgt elke agent de reward uit `PAYOFF`. Een game is `ROUNDS` rondes, 100. Na elke ronde schuift elke agent de eigen Q op naar het target r + γ keer de grootste Q van de volgende state, zoals bij Q-learning. Voor zover de agents weten gaat de game door, dus heeft het target altijd een volgende state. De course traint elk paar 300 games lang, met η = 0,1, γ = 0,9 en ε = 0,05.

        **Zonder geheugen en met geheugen.** Zonder geheugen zit een agent in elke ronde in dezelfde state, `START`. De agent kan niet zien wat er eerder gebeurde, en kan dus alleen leren welke zet op zichzelf meer oplevert: verraden. Met geheugen is de state van een agent de laatste ronde zoals de agent deze zag: een paar zetten, de eigen zet eerst en de zet van de ander als tweede. Voor A is dat `(a, b)`, voor B `(b, a)`; in de eerste ronde van een game zitten beide in `START`. Nu kan een agent leren dat de zet van deze ronde de zet van de ander in de volgende ronde verandert, en met γ = 0,9 tellen de rewards van latere rondes mee.

        **Emergent behavior.** **Emergent behavior** is gedrag van een groep agents dat niemand programmeerde en dat ontstaat doordat de agents samen leren: de schuilplaatsen en hellingen bij verstoppertje, en de regel in de afbeelding hieronder.

        De afbeelding toont run 1 na 300 games: wat A en B met de grootste Q doen, in elke state.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Zonder geheugen verraden beide agents van run 1. Met geheugen werken beide in de eerste ronde samen, en daarna volgen beide dezelfde regel: samenwerken na een ronde waarin beide hetzelfde deden, en verraden na een ronde waarin de twee verschillende zetten deden. De speltheorie kent deze regel als win-stay, lose-shift: houd je zet na een ronde die veel opleverde, 3 of 5, en wissel na een ronde die weinig opleverde, 0 of 1. Niemand gaf de agents deze regel; elke agent leerde de regel terwijl de ander ook leerde.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **multi-agent reinforcement learning**, MARL | | reinforcement learning met meerdere agents in één environment, waarbij elke agent leert terwijl de anderen ook leren | `QA`, `QB` |
        | **prisoner's dilemma** | | een spel voor twee waarin elke speler beter af is door te verraden, wat de ander ook doet, terwijl beide slechter uitkomen dan als beide hadden samengewerkt | `PAYOFF`, `is_dilemma` |
        | **sociaal dilemma** | | een situatie waarin elke agent er baat bij heeft om voor zichzelf te kiezen, terwijl iedereen slechter uitkomt dan als iedereen had samengewerkt | |
        | **emergent behavior** | | gedrag van een groep agents dat niemand programmeerde, en dat ontstaat doordat de agents samen leren | |
        """),
        L.watch_out("Q-learning is gemaakt voor een wereld die hetzelfde blijft. Met een tweede lerende agent erin is er geen garantie dat de twee agents tot rust komen: dat kan, zoals bij de meeste paren met geheugen hier, maar de agents kunnen ook de wereld van de ander blijven veranderen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Twee agents trainen in één wereld:

    1. Geef elke agent een eigen Q-table en een eigen reward: hier allebei uit `PAYOFF`, elk gezien vanaf de eigen kant.
    2. Laat in elke ronde beide agents tegelijk kiezen, ε-greedy, elk vanuit de eigen state.
    3. Geef elke agent de reward en de volgende state: de ronde zoals de agent deze zag, de eigen zet eerst.
    4. Schuif de Q van elke agent op zoals bij Q-learning; voor elke agent hoort de ander bij de wereld.
    5. Laat na de training beide de action met de grootste Q nemen, en tel wat het paar doet: hier de rondes waarin beide samenwerken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Volgen twee agents allebei de regel van de afbeelding, dan werken de twee in elke ronde samen. Maar in ronde 10 verraadt A bij toeval, in een van de willekeurige stappen:

    | Ronde | A doet | B doet | Reward van A | Reward van B |
    |---|---|---|---|---|
    | 9 | samenwerken | samenwerken | 3 | 3 |
    | 10 | verraden, bij toeval | samenwerken | 5 | 0 |
    | 11 | verraden | verraden | 1 | 1 |
    | 12 | samenwerken | samenwerken | 3 | 3 |

    In ronde 11 zit A in state `("betray", "cooperate")` en B in `("cooperate", "betray")`: de twee deden verschillende zetten, dus verraden beide. In ronde 12 zitten beide in `("betray", "betray")`: beide deden hetzelfde, dus werken beide samen, en vanaf daar werken de twee weer samen. Over ronde 10 en 11 kreeg A 5 + 1 = 6, net zoveel als 3 + 3 voor twee keer samenwerken, terwijl B 0 + 1 = 1 kreeg in plaats van 6. Met γ = 0,9 is het verraad voor A iets meer waard: 5 + 0,9 × 1 = 5,9, tegen 3 + 0,9 × 3 = 5,7. De misstap kostte het paar twee rondes en leverde A bijna niets op: de regel beantwoordt een verraad één keer, en gaat dan terug naar samenwerken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rewards():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De rewards van een ronde

        `rewards(a, b)` geeft de rewards van één ronde waarin A `a` speelt en B `b`, elk `"cooperate"` of `"betray"`. `reward_a = PAYOFF[(a, b)]` zoekt de reward van A al op: de zet van A staat als eerste in de key. Vervang de `...` door de reward van B, uit dezelfde tabel. `return reward_a, reward_b` geeft beide terug als paar.

        De check roept `rewards("betray", "cooperate")` aan: A verraadt en B werkt samen. Daarna de andere drie paren zetten. De afbeelding toont de tabel van het spel zoals je functie de tabel geeft.
        """),
        L.task("Laat `rewards` de reward van A en de reward van B voor één ronde geven, allebei uit `PAYOFF`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rewards():
    box_rewards = L.code_box("rewards", lines=4)
    box_rewards
    return (box_rewards,)


@app.cell(hide_code=True)
def check_rewards(box_rewards):
    result_rewards = L.check("rewards", box_rewards.value)
    result_rewards
    return (result_rewards,)


@app.cell(hide_code=True)
def hints_rewards():
    L.hints(
        "rewards",
        [
            "Stap 1 van het recept: elke agent krijgt de reward vanaf de eigen kant. `reward_b` zoekt in dezelfde tabel, `PAYOFF`, met een key waarin de zet van B als eerste staat.",
            "In het voorbeeld werkt B samen terwijl A verraadt, dus moet B 0 krijgen, en `PAYOFF[(\"cooperate\", \"betray\")]` is 0. Welke van de twee namen, `a` of `b`, bevat de zet van B?",
            "`    reward_b = ___`",
        ],
        why="`reward_b = PAYOFF[(b, a)]` zoekt de ronde op vanaf de kant van B, met de eigen zet eerst: 0 voor B als A verraadt, en 5 voor B als B verraadt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dilemma():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Is het een dilemma?

        `is_dilemma(payoff)` zegt of een spel een prisoner's dilemma is: verraden levert meer op, wat de ander ook doet, terwijl allebei samenwerken meer geeft dan allebei verraden. `payoff` is een tabel zoals `PAYOFF`. De eerste vier regels halen de rewards eruit: `both_cooperate` en `both_betray`; `tempted`, wat ik krijg als ik verraad terwijl de ander samenwerkt; en `fooled`, wat ik krijg als ik samenwerk terwijl de ander verraadt. Vervang de twee `...`: `betray_pays`, `True` als verraden in beide gevallen meer oplevert, en na `return` de hele test.

        De check roept `is_dilemma(PAYOFF)` aan, het spel van deze les. Daarna vijf andere spellen, `GAME_2` tot en met `GAME_6`, elk met eigen rewards; de afbeelding toont alle zes.
        """),
        L.task("Laat `is_dilemma` `True` geven voor een spel waarin verraden meer oplevert wat de ander ook doet, terwijl allebei samenwerken meer geeft dan allebei verraden, en anders `False`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dilemma():
    box_dilemma = L.code_box("is_dilemma", lines=7)
    box_dilemma
    return (box_dilemma,)


@app.cell(hide_code=True)
def check_dilemma(box_dilemma):
    result_dilemma = L.check("is_dilemma", box_dilemma.value)
    result_dilemma
    return (result_dilemma,)


@app.cell(hide_code=True)
def hints_dilemma():
    L.hints(
        "is_dilemma",
        [
            "`betray_pays` vergelijkt verraden twee keer met samenwerken: één keer voor als de ander samenwerkt, en één keer voor als de ander verraadt. Welke twee van de vier namen zijn mijn rewards als de ander samenwerkt, en welke twee als de ander verraadt?",
            "Beide vergelijkingen moeten kloppen, dus verbind de vergelijkingen met `and`, elk met `>`; voor `PAYOFF` is dat 5 > 3 en 1 > 0. Na `return` komt `betray_pays`, en ook dat allebei samenwerken meer geeft dan allebei verraden.",
            "`    betray_pays = ___`",
        ],
        why="`betray_pays = tempted > both_cooperate and both_betray > fooled` is waar als verraden meer oplevert, wat de ander ook doet. `return betray_pays and both_cooperate > both_betray` voegt toe dat allebei verraden voor beide slechter uitpakt dan allebei samenwerken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_together():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een paar bekijken

        `together(QA, QB)` laat een getraind paar één game van `ROUNDS` rondes (100) spelen, met de Q-table `QA` van A en `QB` van B, en telt de rondes waarin beide samenwerken. `sa` en `sb` zijn de states van A en B, allebei `START` in de eerste ronde, en `count` begint bij 0; `count += 1` telt er 1 bij op. `greedy(Q, s)` geeft de action met de grootste Q in state `s`, `"cooperate"` bij een gelijke stand of een nieuwe state, dus zijn `a` en `b` de zetten van A en B in deze ronde. In `for _ in range(ROUNDS):` staat de naam `_` voor een getal dat de loop niet nodig heeft. Vervang de twee `...`: de test voor een ronde waarin beide samenwerken, en de states voor de volgende ronde, elk met de eigen zet van de agent eerst.

        De check roept `together(QA_1, QB_1)` aan: A en B van run 1 na 300 games, het paar van de afbeelding in *Het idee*. Daarna runs 4, 7 en 12 na 300 games, run 14 na 100 games, en run 16 na 50. In de afbeelding telt je functie alle 20 paren na 300 games.
        """),
        L.task("Laat `together` de rondes van een game tellen waarin beide agents samenwerken, met de grootste Q en geheugen van de laatste ronde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_together():
    box_together = L.code_box("together", lines=10)
    box_together
    return (box_together,)


@app.cell(hide_code=True)
def check_together(box_together):
    result_together = L.check("together", box_together.value)
    result_together
    return (result_together,)


@app.cell(hide_code=True)
def hints_together():
    L.hints(
        "together",
        [
            "Stap 3 en 5 van het recept. De test na `if`: `a` is `\"cooperate\"` en `b` ook, twee tests met `==`, verbonden met `and`. Daarna krijgen `sa, sb` de ronde zoals elke agent deze zag: `sa` de state van A, `sb` de state van B.",
            "Een state is een paar zetten tussen haakjes, met de eigen zet van de agent eerst. Welke van `a` en `b` komt voor A als eerste, en welke voor B? Twee paren met een komma ertussen geven `sa` en `sb` in één keer.",
            "`        sa, sb = ___`",
        ],
        why="`sa, sb = (a, b), (b, a)` geeft elke agent de ronde zoals de agent deze zag: A de eigen zet `a` eerst, B de eigen zet `b` eerst. Met hetzelfde paar voor beide zou B een verraad van A onthouden als een eigen verraad.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Van 0 tot en met 300 games

    De course trainde 20 paren met geheugen en 20 paren zonder geheugen, elk met eigen willekeurige getallen, 300 games lang. Het blok hieronder gaat door de training, met de games op een logaritmische schaal. De grafiek toont de rondes van 100 waarin beide samenwerken: het gemiddelde van de 20 runs over de laatste 10 games, doorgetrokken met geheugen en gestreept zonder. Ernaast staat wat A en B van run 1 na zoveel games doen, met de grootste Q.
    """)
    return


@app.cell(hide_code=True)
def explore_games_cell():
    explore_games = L.explore("games", "games", lambda _v: C.demo_frame(int(_v), L.lang), start=0,
                              values=C.LOOKS, role="",
                              task="Ga van 0 naar 300: wanneer houden de paren zonder geheugen op met samenwerken, en wat doen de paren met geheugen tussen 20 en 300 games?")
    explore_games
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Beide soorten paren beginnen met samenwerken, en dat hebben de agents niet geleerd: elke Q begint bij 0, en bij een gelijke stand van de twee Q's van een state neemt een agent de eerste action, `"cooperate"`. Daarna ontdekken de agents dat een verraad loont. Zonder geheugen daalt het aantal rondes waarin beide samenwerken binnen 80 games van ongeveer 90 naar bijna 0, en na 200 games werkt geen enkel paar met de grootste Q nog in elke ronde samen. Met geheugen daalt het aantal ook, rond game 80 naar ongeveer 23 van de 100, als nog slechts enkele paren in elke ronde samenwerken. Daarna stijgt het aantal weer, doordat de agents leren dat een verraad in de volgende ronde beantwoord wordt. Na 300 games werken beide in ongeveer 70 van de 100 rondes samen, en werken 14 van de 20 paren in elke ronde samen. Doorgetraind tot 1.000 games doen nog steeds 14 paren dat.

    Het samenwerken van de paren met geheugen hangt af van het begin. Met `"betray"` als eerste action, zodat een gelijke stand een verraad geeft, werkt na dezelfde training geen enkel paar van de 20 in elke ronde samen. Geheugen maakt samenwerken mogelijk, maar niet zeker.

    ### Alle games met NumPy

    `C.history("memory")` geeft een array met shape (20, 300), een rij per run en een kolom per game: de rondes van 100 waarin beide samenwerkten. `C.history("none")` geeft hetzelfde zonder geheugen, en `C.history("betray_first")` met geheugen en `"betray"` als eerste action; met `"reward_a"` als tweede argument geeft `C.history` in plaats daarvan de reward van A in elke game. `_both[:, -50:]` neemt de laatste 50 games van elke run. `.mean(axis=1)` neemt het gemiddelde van elke rij, één getal per run, terwijl `.mean()` zonder `axis` het gemiddelde van alle getallen neemt. `np.round` rondt elk getal af, en `.astype(int)` maakt van de afgeronde getallen gehele getallen, zodat de getallen zonder decimale punt geprint worden. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    for _kind in ("memory", "none", "betray_first"):
        _both = C.history(_kind)   # shape (20, 300): de rondes van 100 waarin beide samenwerkten
        _reward = C.history(_kind, "reward_a")   # dezelfde shape: de reward van A in elke game
        _runs = _both[:, -50:].mean(axis=1)   # één getal per run: het gemiddelde over de laatste 50 games
        print(_kind, _both.shape)
        print("  beide samenwerken, rondes van 100, laatste 50 games:", round(float(_both[:, -50:].mean()), 2))
        print("  reward van A per ronde, laatste 50 games:", round(float(_reward[:, -50:].mean()) / C.ROUNDS, 2))
        print("  per run:", np.round(_runs).astype(int))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Over de laatste 50 games werken de paren met geheugen in 66,78 van de 100 rondes samen, en krijgt A 2,46 per ronde. Zonder geheugen is dat 2,6 rondes, en 1,14 per ronde: weinig meer dan de 1 van twee agents die altijd verraden. Met geheugen zitten 15 van de 20 runs boven 70, niet op 100: met ε = 0,05 neemt elke agent nog af en toe een willekeurige stap, en elke misstap kost het paar een ronde of twee, zoals in het uitgewerkte voorbeeld. De andere vijf runs blijven onder 40. Een gemiddelde over de laatste 50 games is wel iets anders dan het eind van de training: runs 9 en 17 beginnen pas in de laatste paar games met samenwerken, en run 5 blijft verraden na een ronde waarin beide verraadden, zodat één misstap de rest van de game kan kosten. Met de grootste Q werken alle drie na 300 games in elke ronde samen; runs 6 en 12, hier allebei boven 70, doen dat niet, zoals de afbeelding bij oefening 3 laat zien. Zonder geheugen zitten 19 runs op 0; in één run werken de agents een reeks games samen, verraden, en werken weer samen, een paar keer achter elkaar, en na 300 games verraden de agents met de grootste Q. Met `"betray"` als eerste action zitten ook 19 runs op 0, en krijgt A 1,16 per ronde; de ene andere run werkt samen in 83 van de 100 rondes, en met de grootste Q na 300 games in 99 van de 100.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pricing():
    mo.md(r"""
    ### Oefening 4. Prijzen die leren

    In 2020 lieten Emilio Calvano en drie collega's twee Q-learners de prijzen van twee concurrerende winkels bepalen, ronde na ronde. Elke agent kreeg de winst van de eigen winkel als reward, en onthield de prijzen van beide winkels in de laatste ronde, net als de agents met geheugen in deze les. De agents waren niet gebouwd om samen te werken, en konden niet met elkaar praten.
    """)
    return


@app.cell(hide_code=True)
def choice_pricing_cell():
    choice_pricing = L.choice(
        "pricing",
        question="Wat leerden de twee agents?",
        options={
            "low": "Elkaar onderbieden tot beide prijzen zo laag mogelijk waren, want de goedkoopste winkel verkoopt meer",
            "right": "Beide prijzen hoog houden, en een prijsverlaging beantwoorden met een paar rondes lage prijzen",
            "chaos": "Niets stabiels: elke agent bleef de prijs veranderen omdat de ander dat deed",
            "one": "Eén winkel leerde goedkoop te zijn en won alle klanten; de andere gaf het op",
        },
        correct="right",
        why={
            "right": "De agents leerden prijzen ruim boven het niveau waar concurrentie de prijzen naartoe duwt, zonder te praten en zonder dat iemand dit had ingebouwd. Verlaagde de ene agent de prijs, dan antwoordde de andere een paar rondes met lage prijzen, en daarna gingen beide stap voor stap weer omhoog. Dit doen de agents met geheugen in deze les ook: een verraad wordt beantwoord, dus levert verraden bijna niets op. Voor de kopers is dat slecht nieuws. Het mededingingsrecht verbiedt winkels om prijsafspraken te maken, maar deze agents maakten nooit een afspraak, en de onderzoekers zien hier een uitdaging voor wie dit recht moet handhaven.",
            "low": "Dat zou één losse ronde leren: de goedkoopste winkel verkoopt meer. Maar de agents kwamen elkaar steeds weer tegen, en onthielden de laatste ronde. Wat leverde een prijsverlaging op in de volgende ronde?",
            "chaos": "Twee lerende agents kunnen elkaars wereld blijven veranderen, maar deze agents kwamen tot rust, net als de meeste paren met geheugen in deze les. Bij welke prijzen?",
            "one": "Beide agents leerden met dezelfde methode en dezelfde soort reward, en beide onthielden de laatste ronde. Wat leverde een prijsverlaging door een van de twee op in de volgende ronde?",
        },
    )
    choice_pricing
    return (choice_pricing,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) leerde één agent in een wereld die hetzelfde bleef. Hier was elke agent een deel van de wereld van de ander, en deze wereld veranderde terwijl beide leerden. Zonder geheugen leerden twee Q-learners in het prisoner's dilemma elkaar te verraden. Met geheugen van één ronde vonden de meeste paren een regel die niemand de agents gaf, en werkten samen; het begin maakte nog steeds veel uit. Hetzelfde gebeurt op grotere schaal: bij verstoppertje leidde elke nieuwe strategie van het ene team tot de volgende strategie van het andere, en twee prijsagents leerden de prijzen hoog te houden zonder ooit te praten. De volgende les, *Eén interface voor elke environment: Gymnasium*, begint het deel over reinforcement learning in de praktijk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pricing, result_dilemma, result_rewards, result_together):
    L.wrap_up(result_rewards, result_dilemma, result_together, choice_pricing)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
