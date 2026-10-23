# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-smoothing. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-smoothing", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="Op dezelfde tekst haalt model A een gemiddelde NLL van 2,3 en model B 2,6. Welk model voorspelde de tekst beter?",
        options={
            "right": "A: het was minder verrast door de tekst",
            "b": "B: zijn getal is groter",
            "neither": "Geen van beide: de NLL zegt niets over hoe goed een model is",
        },
        correct="right",
        why={
            "b": "De NLL is de gemiddelde verrassing: hoe groter, hoe slechter het model de tekst voorspelde. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "neither": "De gemiddelde NLL meet precies dat: hoe goed het model de echte tekst voorspelde. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "right": "Lager is beter: A gaf de tekens die echt kwamen gemiddeld hogere kansen.",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def quick_check_zero():
    quick_zero = L.choice(
        "quick-zero",
        question="Een tekst heeft 1.000 paren. Een model geeft 999 ervan de kans 0,1, en één ervan de kans 0. Wat is de gemiddelde NLL?",
        options={
            "near": "Ongeveer 2,3: één paar op de 1.000 telt nauwelijks mee",
            "more": "Iets meer dan 2,3: de nul trekt het wat omhoog",
            "right": "Oneindig: één paar met kans 0 is genoeg",
        },
        correct="right",
        why={
            "near": "De verrassing bij kans 0 is min log 0: oneindig, en een gemiddelde met een oneindige term is oneindig. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "more": "Niet een beetje meer: de verrassing van dat ene paar is oneindig. Zie [Hoe goed is een language model?](/course/open/lm-nll?lang=nl).",
            "right": "Min log 0 is oneindig, en één oneindige verrassing maakt het totaal, en dus het gemiddelde, oneindig.",
        },
    )
    quick_zero
    return (quick_zero,)


@app.cell(hide_code=True)
def text_why():
    mo.md(f"""
    ## Waarom

    In de vorige les kreeg "Zebra" een oneindige NLL van je model van de toneelstukken, alleen omdat daar na een "Z" nooit een "e" kwam. Het model noemt zo'n paar onmogelijk, terwijl het alleen zeldzaam is. Van de {C.grouped(65 * 65, L.lang)} vakken van je tabel staat er in {C.grouped(C.zero_cells(), L.lang)} een 0. Elke nieuwe tekst met één zo'n paar krijgt een oneindige NLL, dus de NLL kan je niet vertellen hoe goed het model is op tekst die het niet gezien heeft, en voor die tekst is een language model juist bedoeld.

    **Een voorbeeld uit het dagelijks leven.** Een winkel die in een droge maand geen enkele paraplu verkocht en daarom voor de volgende maand niets bestelt, heeft niets te verkopen op de eerste regendag: nog niet is niet hetzelfde als nooit.
    """)
    return


@app.cell(hide_code=True)
def predict_last():
    guess_last = L.guess(
        "last-part",
        question=f"Tel de paren van de eerste 90% van de toneelstukken, en kijk naar de laatste 10%, tekst die niet in de tellingen zat. Hoeveel van de {C.grouped(C.unseen_in_last()[1], L.lang)} paren daarvan kwamen volgens jou in de eerste 90% nooit voor?",
        options={
            "none": "Geen: het is meer van dezelfde toneelstukken",
            "right": "Enkele: minder dan 1 op de 100",
            "many": "Veel: meer dan 1 op de 10, want de meeste vakken van de tabel zijn 0",
        },
        reveal=f"**Enkele: {C.grouped(C.unseen_in_last()[0], L.lang)} van de {C.grouped(C.unseen_in_last()[1], L.lang)}**, minder dan 1 op de 500. De meeste nullen van de tabel zijn paren die bijna nooit voorkomen. Maar enkele zijn genoeg: het model geeft elk ervan kans 0, dus de gemiddelde NLL op de laatste 10% is oneindig, terwijl die op de eerste 90% {C.number(C.nll_split(0)[0], 2, L.lang)} is. De tekst is niet vreemder dan de rest van de toneelstukken; het model is er alleen zeker van dat wat het nooit zag, niet kan gebeuren.",
    )
    guess_last
    return (guess_last,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Smoothing** geeft elk paar een beetje kans, ook de paren die nooit voorkwamen. De eenvoudigste manier is **add-k smoothing**: tel bij elke telling een klein getal k op, voordat je door de rijsommen deelt. Een paar dat nooit voorkwam, krijgt dan als kans k gedeeld door de nieuwe rijsom: klein, maar geen 0. Met k = 1 heet het Laplace smoothing. Voor het paar van a en b, met $N$ de telmatrix en 65 tekens:
        """),
        L.formula(r"[[pred:P(b \mid a)]] = \frac{[[data:N]][a, b] + [[hyper:k]]}{(\text{som van rij } a \text{ van } [[data:N]]) + 65 [[hyper:k]]}"),
        mo.md(f"""
        De prijs: de paren die wel voorkwamen, staan een beetje van hun kans af, dus de NLL op de getelde tekst gaat iets omhoog. De winst zit in tekst die het model niet telde, waar de nullen de NLL oneindig maakten. k is een getal dat je zelf kiest. De tabel: de eerste 90% van de toneelstukken geteld, met een paar waarden van k, en de gemiddelde NLL op dat getelde deel en op de laatste 10%.

        | k | NLL op de getelde 90% | NLL op de laatste 10% |
        |---|---|---|
        | 0 | {C.number(C.nll_split(0)[0], 4, L.lang)} | {C.number(C.nll_split(0)[1], 4, L.lang)} |
        | 0,01 | {C.number(C.nll_split(0.01)[0], 4, L.lang)} | {C.number(C.nll_split(0.01)[1], 4, L.lang)} |
        | 0,1 | {C.number(C.nll_split(0.1)[0], 4, L.lang)} | {C.number(C.nll_split(0.1)[1], 4, L.lang)} |
        | 1 | {C.number(C.nll_split(1)[0], 4, L.lang)} | {C.number(C.nll_split(1)[1], 4, L.lang)} |
        | 10 | {C.number(C.nll_split(10)[0], 4, L.lang)} | {C.number(C.nll_split(10)[1], 4, L.lang)} |
        | 100 | {C.number(C.nll_split(100)[0], 4, L.lang)} | {C.number(C.nll_split(100)[1], 4, L.lang)} |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **smoothing** | | elk mogelijk paar minstens een beetje kans geven, ook de paren die nooit voorkwamen | |
        | **add-k smoothing** | add k | hetzelfde kleine getal k bij elke telling optellen voordat je door de rijsommen deelt; met k = 1 Laplace smoothing | `N + k` |
        """),
        L.watch_out("Smoothing maakt de NLL op de getelde tekst slechter, niet beter: daar lijkt k = 0 altijd het best. De winst zie je alleen op tekst die het model niet telde, dus kies k op zulke tekst."),
        mo.md(r"""
        De figuur: de 65 kansen na "Z", in de volgorde van de tabel, zonder smoothing en met k = 1.
        """),
        mo.Html(C.row_bars("Z", {"title": "Na \"Z\": zonder en met smoothing", "before": "zonder smoothing: {n} van de 65 zijn 0",
                                 "after": "met k = 1: geen enkele is 0", "axis": "het volgende teken, in de volgorde van de tabel",
                                 "alt": "Twee rijen van 65 staven, de kansen van elk teken na Z. Zonder smoothing zijn er maar 4 staven te zien, waarvan 2 heel klein, en zijn er 61 nul. Met k = 1 zijn alle 65 staven te zien: de 2 grote zijn ongeveer een kwart lager, en de andere 63 zijn klein maar geen 0."})),
        mo.md(f"""
        Een "Z" kwam {C.row_count("Z")} keer voor in de toneelstukken, gevolgd door maar vier verschillende tekens. Met k = 1 wordt de rijsom {C.row_count("Z")} + 65 = {C.row_count("Z") + 65}, en elk teken dat nooit na "Z" kwam, krijgt 1/{C.row_count("Z") + 65}. Samen krijgen die 61 tekens 61/{C.row_count("Z") + 65}, bijna een kwart van de rij, en dat leveren de vier die wel kwamen in: in een rij met weinig tellingen is k = 1 veel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om add-k smoothing toe te passen op een telmatrix `N`, en k te kiezen:

    1. Tel k bij elke telling op: `M = N + k`.
    2. Deel elke rij door haar nieuwe som: `P = M / M.sum(axis=1, keepdims=True)`.
    3. Om k te kiezen: meet de NLL op tekst die niet geteld is, voor een paar waarden van k, en neem de k met de laagste NLL.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tellingen van "abab ba", en een nieuwe tekst, "ba ab", die niet geteld is: de paren "a " en " a" kwamen in "abab ba" nooit voor. Zonder smoothing zou de NLL ervan oneindig zijn. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_smooth():
    import numpy as np

    _N = np.array([[0, 0, 1], [0, 0, 2], [1, 2, 0]])   # de tellingen van "abab ba"
    _row_of = {" ": 0, "a": 1, "b": 2}
    _new = "ba ab"   # een tekst die niet geteld is
    for _k in [0.01, 0.1, 1, 10]:
        # 1. k bij elke telling opgeteld
        _M = _N + _k
        # 2. elke rij gedeeld door haar nieuwe som
        _P = _M / _M.sum(axis=1, keepdims=True)
        # 3. de NLL op de tekst die niet geteld is
        _total = 0
        for _i in range(len(_new) - 1):
            _total = _total - np.log(_P[_row_of[_new[_i]], _row_of[_new[_i + 1]]])
        print(_k, _total / (len(_new) - 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met k = 0,01 krijgen de twee paren die nooit voorkwamen zo weinig dat de NLL nog hoog is, ongeveer 2,59. Met k = 1 is hij het laagst, ongeveer 1,05. Met k = 10 is elke rij bijna gelijk verdeeld, dus de paren die wel voorkwamen, leveren te veel in, en de NLL gaat weer omhoog, naar ongeveer 1,07. Op deze tekst is k = 1 de beste van de vier.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unseen():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een paar dat nooit voorkwam
    """)
    return


@app.cell(hide_code=True)
def choice_unseen_cell():
    choice_unseen = L.choice(
        "unseen-prob",
        question="De tellingen in één rij van een tabel met 65 tekens tellen op tot 935. Welke kans geeft add-k smoothing met k = 1 aan een paar uit die rij dat nooit voorkwam?",
        options={
            "zero": "0: het kwam nooit voor",
            "right": "0,001: 1 op 935 + 65",
            "row": "1 op 935: één telling meer",
            "even": "1 op 65: alsof elk teken gelijk is",
        },
        correct="right",
        why={
            "zero": "Dat is zonder smoothing. Met k = 1 wordt de telling ervan 0 + 1.",
            "row": "De telling ervan wordt 1, maar de rijsom groeit ook: elk van de 65 tellingen krijgt er 1 bij.",
            "even": "Dat zou een rij zonder tellingen zijn. Deze rij heeft er 935, en die tellen nog steeds mee.",
            "right": "De telling ervan wordt 1, en de rijsom 935 + 65 = 1.000, dus 1/1.000 = 0,001.",
        },
    )
    choice_unseen
    return (choice_unseen,)


@app.cell(hide_code=True)
def text_exercise_smooth():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Smoothing van een telmatrix

        `smooth(N, k)` geeft de kansen van de telmatrix `N` met add-k smoothing. Vervang de `...` door stap 1 en 2 van het recept. De check roept `smooth(np.array([[3, 1, 0], [0, 0, 2]]), 1)` aan, en dat moet ongeveer `[[0.571, 0.286, 0.143], [0.2, 0.2, 0.6]]` geven; hij probeert ook andere tellingen en andere waarden van `k`.
        """),
        L.task("Laat `smooth` `k` bij elke telling optellen en elke rij door haar nieuwe som delen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_smooth():
    box_smooth = L.code_box("smooth", lines=5)
    box_smooth
    return (box_smooth,)


@app.cell(hide_code=True)
def check_smooth(box_smooth):
    result_smooth = L.check("smooth", box_smooth.value)
    result_smooth
    return (result_smooth,)


@app.cell(hide_code=True)
def hints_smooth():
    L.hints(
        "smooth",
        [
            "Stap 1 en 2 van het recept, zoals in het uitgewerkte voorbeeld.",
            "Eerst een nieuwe matrix met `k` bij elke telling. Dan de som van elk van haar rijen, als kolom bewaard met `keepdims=True`, zodat broadcasting elke rij door haar eigen som deelt.",
            "`M = N + ___` en `return M / M.sum(axis=___, keepdims=True)`.",
        ],
        why="`[[3, 1, 0], [0, 0, 2]]` plus 1 is `[[4, 2, 1], [1, 1, 3]]`, met de rijsommen 7 en 5. Gedeeld: `[[4/7, 2/7, 1/7], [1/5, 1/5, 3/5]]`, ongeveer `[[0.571, 0.286, 0.143], [0.2, 0.2, 0.6]]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ### Oefening 3. Kies k
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick-k",
        question="Kijk naar de tabel in *Het idee*. Welke k zou je kiezen voor je model van de toneelstukken, om te gebruiken op tekst die het niet gezien heeft?",
        options={
            "zero": "0: de NLL op de getelde 90% is daar het laagst",
            "small": "0,01: zo weinig mogelijk verandering aan de tellingen",
            "right": "1: de NLL op de laatste 10% is daar het laagst",
            "big": "100: geen enkel paar komt nog in de buurt van 0",
        },
        correct="right",
        why={
            "zero": "Op de getelde tekst lijkt k = 0 altijd het best; op nieuwe tekst geeft het een oneindige NLL. Zie het blok *Let op* in *Het idee*.",
            "small": "Beter dan 0, maar de NLL op de laatste 10% is hoger dan met k = 1: de paren die nooit voorkwamen, krijgen nog te weinig.",
            "big": "Met k = 100 worden de rijen met weinig tellingen bijna gelijk verdeeld, dus de paren die wel voorkwamen, leveren daar te veel in: de NLL op de laatste 10% is de hoogste van de tabel, na die van 0.",
            "right": "Van de waarden in de tabel geeft k = 1 de laagste NLL op tekst die niet in de tellingen zat, en daar is het model voor.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De NLL uit [Hoe goed is een language model?](/course/open/lm-nll?lang=nl) werd oneindig voor elke tekst met een paar dat het model nooit zag; add-k smoothing geeft elk paar een beetje kans, en de NLL op nieuwe tekst laat zien welke k het best werkt. Met *Toon verdieping* aan komt eerst *Spam of niet: naive Bayes*: daar maken woordtellingen met Laplace smoothing en de regel van Bayes samen een spamfilter. In de volgende les van de hoofdroute, *Project: hetzelfde bigram-model, geleerd met gradient descent*, leert je model deze tabel in plaats van haar te tellen, en zie je wat daar de plaats van k inneemt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pick, choice_unseen, result_smooth):
    L.wrap_up(choice_unseen, result_smooth, choice_pick)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
