# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-what-is-rl. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rl-what-is-rl", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl) en [Herhalen: for-loops](/course/open/py-for?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Een robot leert lopen. Niemand doet het voor, maar hij krijgt een punt voor elke meter die hij vooruitkomt zonder te vallen. Welke soort leren is dat?",
        options={
            "supervised": "Supervised learning: de punten zijn de labels",
            "right": "Reinforcement learning: de punten zijn rewards",
            "unsupervised": "Unsupervised learning: er zijn geen goede antwoorden",
        },
        correct="right",
        why={
            "supervised": "Een label is het goede antwoord bij een voorbeeld, vooraf opgeschreven. Niemand schreef de goede stap op; de punten komen nadat de robot iets doet. Zie [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "unsupervised": "Unsupervised learning vindt groepen of patronen zonder enige terugkoppeling; hier doet de robot iets en krijgt hij er punten voor. Zie [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
            "right": "De robot doet iets, en de punten die hij daarvoor krijgt, zijn rewards.",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="Wat laat dit programma zien?\n\n```python\nw = 1\nfor k in range(3):\n    w = w * 0.5\nprint(w)\n```",
        options={
            "right": "0,125",
            "plus": "2,5",
            "once": "0,5",
        },
        correct="right",
        why={
            "plus": "`w * 0.5` halveert `w`; het telt er geen 0,5 bij op. Zie [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "once": "De loop draait drie keer, voor `k` = 0, 1 en 2, en halveert `w` elke keer opnieuw. Zie [Herhalen: for-loops](/course/open/py-for?lang=nl).",
            "right": "Drie keer gehalveerd: 1, dan 0,5, 0,25 en 0,125.",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Bij supervised learning schrijven mensen bij elk voorbeeld het goede antwoord op. Bij veel taken kan dat niet: welke zet is goed in deze schaakstelling, hoe hard moet een robotbeen nu duwen, hoeveel moet een verwarming stoken om 7:14 uur? Achteraf kun je wel zeggen hoe goed het ging: een gewonnen partij, een gelopen meter, een warm huis voor weinig energie. Reinforcement learning leert precies daarvan, van een getal na elke stap dat zegt hoe goed die was.

    **Een voorbeeld uit het dagelijks leven.** Een hond leert zitten. Je kunt hem niet in woorden vertellen wat je wilt. Maar als hij gaat zitten, krijgt hij een snoepje. Na een tijdje gaat hij vaker zitten, omdat zitten loont.
    """)
    return


@app.cell(hide_code=True)
def predict_credit():
    guess_credit = L.guess(
        "credit",
        question="Een programma speelt schaak. Het krijgt alleen aan het eind een reward: +1 voor een gewonnen partij en −1 voor een verloren partij, en 0 na elke zet daarvoor. Wat denk je dat het over zijn eerste zet kan leren?",
        options={
            "nothing": "Niets: de eerste zet kreeg altijd 0",
            "right": "Iets, via de reward aan het eind",
            "all": "Alles: de eerste zet beslist de partij",
        },
        reveal="**Iets, via de reward aan het eind.** Het programma kijkt niet alleen naar de reward vlak na een zet, maar naar alle rewards die erop volgen: de return uit *Het idee*. De winst aan het eind telt dus ook mee voor de eerste zet. Maar welke van zijn 40 zetten won de partij? Dat is het moeilijke deel van reinforcement learning, *credit assignment* genoemd. Over veel partijen ontdekt het programma welke zetten vaak voor een winst komen.",
    )
    guess_credit
    return (guess_credit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Bij reinforcement learning leert een **agent** door te handelen. Alles buiten de agent is de **environment**. Ze zijn om de beurt aan zet, stap na stap:

        1. de agent ziet de situatie waarin hij zit en kiest een **action**;
        2. de environment verandert, en geeft de nieuwe situatie terug en een **reward**: een getal dat zegt hoe goed die stap was.

        De rewards zijn de enige terugkoppeling: niemand zegt welke action goed was geweest. De *reward-hypothese* zegt dat elk doel zo te schrijven is: zoveel mogelijk reward halen, bij elkaar opgeteld.

        De agent probeert niet de grootste volgende reward te halen, maar de grootste **return**: de som van alle rewards vanaf nu. Latere rewards tellen minder. Voor een spel van $T$ stappen, met reward $r_1$ na de eerste stap, $r_2$ na de tweede, enzovoort:
        """),
        L.formula(r"[[loss:G_0]] = [[data:r_1]] + [[hyper:\gamma]] [[data:r_2]] + [[hyper:\gamma]]^2 [[data:r_3]] + \dots + [[hyper:\gamma]]^{T-1} [[data:r_T]]"),
        mo.md(r"""
        De **discount factor** $\gamma$ is een getal van 0 tot en met 1. Elke stap verder weg vermenigvuldigt een reward nog een keer met $\gamma$: met $\gamma$ = 0,5 telt een reward één stap later voor de helft, twee stappen later voor een kwart. Met $\gamma$ dicht bij 1 is de agent geduldig: rewards ver weg tellen nog mee. Met $\gamma$ = 0 telt alleen de volgende reward. Verdisconteren past ook bij het idee dat een reward nu zekerder is dan een reward ver vooruit, en met $\gamma$ onder 1 houdt het de return eindig bij een taak die nooit stopt, zoals de verwarming.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **agent** | | wat in reinforcement learning leert en kiest: kiest actions en krijgt rewards | |
        | **environment** | | alles buiten de agent: krijgt de action, en geeft de nieuwe situatie en een reward terug | |
        | **action** | | één keuze van de agent bij een stap, zoals een zet in een spel | |
        | **reward** | | een getal van de environment na elke stap: hoe goed die stap was | `r` |
        | **return** | | de som van de rewards vanaf nu, latere verdisconteerd: wat de agent zo groot mogelijk probeert te maken | `G` |
        | **discount factor** | | een getal van 0 tot en met 1: een reward één stap later telt $\gamma$ keer zo zwaar | `gamma` |
        | $G_0$ | G nul | de return vanaf het begin | `G` |
        | $r_1, r_2, \dots$ | r één, r twee | de rewards na stap 1, stap 2, enzovoort | `rewards` |
        | $\gamma$ | gamma | de discount factor | `gamma` |
        """),
        L.watch_out("*Agent* en *return* betekenen elders iets anders. Hier is een agent wat in reinforcement learning leert en kiest; een *LLM agent* is een language model dat tools aanroept, en dat is iets anders. En de return is de verdisconteerde som van de rewards; het `return`-statement van Python, in codeletters, beëindigt een functie en geeft zijn uitkomst terug."),
        mo.md(r"""
        De figuur: één stap van de lus tussen agent en environment. Die herhaalt zich tot de taak klaar is.
        """),
        draw.flow(
            [("Agent", ["ziet de situatie,", "kiest een action"]),
             ("Environment", ["verandert, en geeft", "een reward terug"])],
            forward=["action"],
            back=["reward en\nnieuwe situatie"],
            title="Eén stap, steeds opnieuw"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de return van een spel uit te rekenen:

    1. Schrijf de rewards van het spel op, één per stap, op volgorde.
    2. Begin met een return `G` van 0, en een factor van 1 voor de eerste reward.
    3. Voor elke reward: tel de reward keer de factor op bij `G`, en vermenigvuldig dan de factor met `gamma` voor de volgende reward.
    4. Vergelijk returns: de agent kiest liever de grootste.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee spellen van vier stappen: het ene geeft meteen een punt, het andere 4 punten bij de laatste stap. Welk spel heeft de grootste return, met $\gamma$ = 0,5 en met $\gamma$ = 0,75? `_discounted` rekent de return van een lijst rewards uit. Elke `print` laat $\gamma$ zien, dan de return van het punt meteen, en dan die van de 4 punten aan het eind. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_return():
    # 1. de rewards van twee spellen, één per stap: meteen een punt, of 4 punten aan het eind
    _now = [1, 0, 0, 0]
    _late = [0, 0, 0, 4]

    def _discounted(rewards, gamma):
        # 2. een return van 0, en een factor van 1 voor de eerste reward
        G = 0
        factor = 1
        # 3. elke reward keer zijn factor; de volgende factor is gamma keer zo groot
        for r in rewards:
            G = G + factor * r
            factor = factor * gamma
        return G

    # 4. vergelijk de returns: gamma, het punt meteen, de 4 punten aan het eind
    print(0.5, _discounted(_now, 0.5), _discounted(_late, 0.5))
    print(0.75, _discounted(_now, 0.75), _discounted(_late, 0.75))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met $\gamma$ = 0,5 telt de 4 bij stap 4 maar 0,5 · 0,5 · 0,5 = 0,125 keer zo zwaar: een return van 0,5, minder dan het punt meteen. Met $\gamma$ = 0,75 telt hij 0,75 · 0,75 · 0,75 ≈ 0,42 keer zo zwaar: een return van 1,6875, meer dan het punt meteen. Dezelfde rewards, een andere $\gamma$, een andere keuze: de discount factor bepaalt hoe geduldig de agent is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parts():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welk deel is wat?

    Twee taken: een robotstofzuiger die leert schoonmaken, en een programma dat leert schaken. Zet elk onderdeel in zijn groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_parts():
    sort_parts = L.sort("parts")
    sort_parts
    return (sort_parts,)


@app.cell(hide_code=True)
def check_parts(sort_parts):
    result_parts = L.check("parts", sort_parts)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "parts",
        [
            "Vraag bij elk onderdeel: kiest het, wordt het gekozen, wordt erop gewerkt, of is het een getal dat zegt hoe goed het ging?",
            "Elke taak heeft één agent, één environment, hier één action en één reward. De agent is wat leert; de rewards zijn getallen.",
            "De robotstofzuiger en het schaakprogramma zijn de agents: die kiezen. Waarop werken ze, wat doen ze, en welke getallen zeggen hoe goed het ging?",
        ],
        why="**Agent:** de robotstofzuiger, het schaakprogramma. **Environment:** de kamers met hun stof en meubels, het bord en de tegenstander. **Action:** naar links draaien, een paard verzetten. **Reward:** +1 voor elke vierkante meter die schoon wordt, +1 voor een gewonnen partij en −1 voor een verloren partij.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_return():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De return van elk spel

        `discounted_return(rewards, gamma)` geeft de return van een lijst rewards. Vervang de twee `...` door stap 3 van het recept. De check roept `discounted_return([0, 0, 0, 1], 0.9)` aan: een reward van 1, drie stappen na de eerste, dus dat moet 0,9 · 0,9 · 0,9 = `0.729` geven. Hij probeert ook andere rewards, ook negatieve, en andere discount factors.
        """),
        L.task("Laat `discounted_return` elke reward keer zijn factor optellen, en de factor bij elke stap `gamma` keer zo groot maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_return():
    box_return = L.code_box("discounted-return", lines=8)
    box_return
    return (box_return,)


@app.cell(hide_code=True)
def check_return(box_return):
    result_return = L.check("discounted-return", box_return.value)
    result_return
    return (result_return,)


@app.cell(hide_code=True)
def hints_return():
    L.hints(
        "discounted-return",
        [
            "Stap 3 van het recept: twee regels, in die volgorde.",
            "De eerste regel telt deze reward op bij `G`, maar niet volledig: keer `factor`. De tweede regel maakt `factor` klaar voor de volgende reward, die `gamma` keer zo zwaar telt.",
            "`G = G + factor * ___` en `factor = factor * ___`.",
        ],
        why="Met `[0, 0, 0, 1]` en 0,9 zijn de factoren 1, 0,9, 0,81 en 0,729. De eerste drie rewards zijn 0, dus alleen de laatste telt iets op: 0,729 · 1 = 0,729.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Oefening 3. Helemaal geen geduld
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "gamma-zero",
        question="Een agent met $\\gamma$ = 0 kan kiezen tussen twee actions. De ene geeft een reward van 1 bij de volgende stap en daarna niets; de andere geeft 0 bij de volgende stap en 100 bij de stap daarna. Welke kiest hij?",
        options={
            "later": "De 100: dat is honderd keer zoveel",
            "either": "Maakt niet uit: het zijn allebei rewards",
            "right": "De 1: alleen de volgende reward telt",
        },
        correct="right",
        why={
            "later": "De 100 komt een stap later, dus telt hij $\\gamma$ keer zo zwaar, en $\\gamma$ is hier 0. Zie *Het idee*.",
            "either": "De twee returns verschillen. Reken ze uit met de formule uit *Het idee*.",
            "right": "Met $\\gamma$ = 0 wordt elke reward na de volgende met 0 vermenigvuldigd: de return is 1 tegen 0. Zo'n agent ziet alleen de volgende reward.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl) noemde reinforcement learning leren van rewards; nu ken je de onderdelen. Een agent handelt in een environment, en niet de volgende reward telt, maar de return: alle rewards vanaf nu, de latere verdisconteerd met $\gamma$. Welke action tot een grote return leidt, moet de agent leren, door te proberen. In de volgende les, *Een doolhof doorzoeken: breadth-first search*, kent een agent zijn environment, en plant hij zijn actions voordat hij ze uitvoert: hij zoekt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_zero, result_parts, result_return):
    L.wrap_up(result_parts, result_return, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
