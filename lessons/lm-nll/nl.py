# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-nll. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow log-loss

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-nll", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl) en [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_certain():
    quick_certain = L.choice(
        "quick-certain",
        question="Een model gaf de uitkomst die echt gebeurde de kans 1. Wat is de verrassing, min de log van die kans?",
        options={
            "right": "0: het model was zeker, en had gelijk",
            "one": "1: de kans zelf",
            "inf": "Oneindig groot: niets is ooit zeker",
        },
        correct="right",
        why={
            "one": "De verrassing is min de log van de kans, en log 1 is 0. Zie [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "inf": "Oneindig groot is de verrassing bij kans 0, een uitkomst die het model onmogelijk noemde. Zie [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "right": "log 1 is 0, dus min log 1 is ook 0: waar het model zeker van was, kost niets als het gebeurt.",
        },
    )
    quick_certain
    return (quick_certain,)


@app.cell(hide_code=True)
def quick_check_cell():
    quick_cell = L.choice(
        "quick-cell",
        question="Wat is `P[i, j]` in je tabel met bigram-kansen?",
        options={
            "count": "Hoe vaak het paar `chars[i] + chars[j]` in de tekst voorkwam",
            "right": "Hoe waarschijnlijk het is dat `chars[j]` volgt op `chars[i]`",
            "reverse": "Hoe waarschijnlijk het is dat `chars[i]` vóór `chars[j]` komt",
        },
        correct="right",
        why={
            "count": "Dat zijn de tellingen van de telmatrix `N`; `P` bevat elke rij daarvan gedeeld door haar som. Zie [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).",
            "reverse": "Rij `i` is het teken ervoor, en de getallen erin tellen op tot 1 over wat erna komt, niet over wat er vóór `chars[j]` komt. Zie [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).",
            "right": "Rij `i` is de verdeling van wat er na `chars[i]` komt, en kolom `j` kiest daaruit `chars[j]`.",
        },
    )
    quick_cell
    return (quick_cell,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je bigram-model geeft een kans aan elk teken dat kan volgen. Maar hoe goed is het, in een getal? Om het te vergelijken met de betere modellen van latere lessen, heb je één maat nodig die zegt hoe goed een model echte tekst voorspelt. Elk language model wordt zo gemeten, tot de grootste LLM's toe.

    **Een voorbeeld uit het dagelijks leven.** Een weervoorspeller die 90% kans op regen gaf op de dagen dat het regende, en 10% op de droge dagen, deed het goed; wie elke dag 50% zei, niet. Je beoordeelt een voorspeller op de kansen die de voorspeller gaf aan wat er echt gebeurde.
    """)
    return


@app.cell(hide_code=True)
def predict_plays():
    guess_plays = L.guess(
        "plays-nll",
        question=f"Een model dat niets weet, geeft elk van de 65 tekens de kans 1/65. Zijn gemiddelde verrassing per teken is log 65, ongeveer {L.number(C.uniform_nll(), 2)}. Wat denk je dat je bigram-model van de toneelstukken haalt, op de toneelstukken zelf?",
        options={
            "barely": "Ongeveer 4: één teken ervoor zegt weinig",
            "right": "Ongeveer 2,5: minder verrast, maar nog ver van zeker",
            "tiny": "Ongeveer 0,5: het is bijna nooit verrast",
        },
        reveal=f"**Ongeveer 2,5: {L.number(C.plays_nll(), 2)}**, gemiddeld over alle {C.grouped(C.pair_count(), L.lang)} paren van de toneelstukken. Als kans: e tot de macht min {L.number(C.plays_nll(), 2)} is ongeveer {L.number(2.718281828459045 ** -C.plays_nll(), 3)}, dus bij een gewoon teken gaf het model wat er echt volgde een kans van ongeveer 1 op {round(2.718281828459045 ** C.plays_nll())}, tegen 1 op 65 voor een model dat niets weet. Beter, maar verre van zeker: één teken is niet veel om op af te gaan.",
    )
    guess_plays
    return (guess_plays,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Geef het model een echte tekst, en kijk naar de kans die het gaf aan elk volgend teken dat echt kwam. De **likelihood** van de tekst is al die kansen met elkaar vermenigvuldigd: voor "the" is dat $P(\text{h} \mid \text{t}) \cdot P(\text{e} \mid \text{h})$. Hoe hoger, hoe beter het model de tekst voorspelde.

        Voor een lange tekst is dat product piepklein: een miljoen kansen van ongeveer 0,1 vermenigvuldigd geven een getal met bijna een miljoen nullen achter de komma, dat een computer als 0 bewaart. Dus neem je de log van elke kans en tel je de logs op, want de log van een product is de som van de logs. Met een minteken ervoor, zodat het getal positief is, is dat de **negative log-likelihood**, kortweg NLL: de totale verrassing van de tekst. Gedeeld door het aantal voorspellingen is het de gemiddelde verrassing per teken:
        """),
        L.formula(r"[[loss:\text{gemiddelde NLL}]] = -\frac{1}{n} \sum_{t=1}^{n} \log [[pred:P(c_t \mid c_{t-1})]]"),
        mo.md(r"""
        Hier zijn $c_0, c_1, \dots, c_n$ de tekens van de tekst, dus een tekst van $n + 1$ tekens heeft $n$ paren. Lager is beter. Het is dezelfde gemiddelde verrassing van wat er echt gebeurde als bij de log loss uit [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl), nu met 65 tekens om uit te kiezen bij elke stap in plaats van 2 klassen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **likelihood** | | de kans die een model aan een hele tekst geeft: de kansen van alle volgende tekens die echt kwamen, vermenigvuldigd | |
        | **negative log-likelihood** (NLL) | | min de log van de likelihood: de som van de verrassingen; gedeeld door het aantal paren, de gemiddelde NLL | `total - np.log(p)` |
        | $P(c_t \mid c_{t-1})$ | P van c t gegeven c t min 1 | de kans die het model geeft aan teken $c_t$ na het teken ervoor | `P[row_of["t"], row_of["h"]]` |
        """),
        L.watch_out("Een paar dat het model nooit zag, heeft kans 0, en de verrassing ervan, min log 0, is oneindig. Eén zo'n paar maakt de NLL van de hele tekst oneindig. Voor het model van de toneelstukken heeft \"Zebra\" er een: na een \"Z\" kwam nooit een \"e\". De volgende les lost dat op."),
        mo.md(r"""
        De figuur: de verrassing van elk paar van "To be, or not to be", voor het model van de toneelstukken, met hun gemiddelde als stippellijn.
        """),
        mo.Html(C.surprise_bars("To be, or not to be", {"title": "Verrassing per paar", "mean": f"gemiddeld {L.number(C.text_nll('To be, or not to be'), 2)}",
                                                         "alt": "18 staven, één per paar van de regel To be, or not to be. De hoogste zijn de n na een spatie, ongeveer 3,5, de komma na be en de o na een spatie, allebei ongeveer 3,1 tot 3,2. De laagste is de spatie na de komma, ongeveer 0,3. Een stippellijn geeft het gemiddelde aan, ongeveer 2,2."})),
        mo.md(r"""
        De hoogste staven zijn tekens die het model niet zag aankomen: de "n" van " not" na een spatie, en de komma na "be". De laagste is de spatie na de komma: na een komma komt meestal een spatie.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de gemiddelde NLL van een tekst te vinden met een tabel van bigram-kansen `P`:

    1. Noteer de rij van elk teken: `row_of[chars[k]] = k`, voor elke `k`.
    2. Zoek voor elk paar van de tekst, `text[i]` en `text[i + 1]`, de kans op van het tweede teken na het eerste: `p = P[row_of[text[i]], row_of[text[i + 1]]]`.
    3. Tel de verrassing ervan op bij het totaal: `total = total - np.log(p)`.
    4. Deel het totaal door het aantal paren, `len(text) - 1`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tabel van "abab ba" uit [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl), en de tekst "ab ba". Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_nll():
    import numpy as np

    _P = np.array([[0, 0, 1], [0, 0, 1], [1/3, 2/3, 0]])   # de tabel van "abab ba"
    _chars = [" ", "a", "b"]
    _text = "ab ba"
    # 1. de rij van elk teken
    _row_of = {}
    for _k in range(len(_chars)):
        _row_of[_chars[_k]] = _k
    _total = 0
    for _i in range(len(_text) - 1):
        # 2. de kans op het volgende teken na dit teken
        _p = _P[_row_of[_text[_i]], _row_of[_text[_i + 1]]]
        print(_text[_i] + _text[_i + 1], _p)
        # 3. de verrassing ervan, opgeteld bij het totaal
        _total = _total - np.log(_p)
    # 4. gedeeld door het aantal paren
    print(_total / (len(_text) - 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De paren "ab" en " b" hadden kans 1, dus die kostten niets. "b " had 1/3 en kostte ongeveer 1,10, "ba" had 2/3 en kostte ongeveer 0,41. Samen ongeveer 1,50, over 4 paren: een gemiddelde NLL van ongeveer 0,376.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_by_hand():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Met de hand
    """)
    return


@app.cell(hide_code=True)
def choice_by_hand_cell():
    choice_by_hand = L.choice(
        "by-hand",
        question="Een model geeft de twee paren van de tekst \"abc\" de kansen 0,5 en 0,25. Wat is de gemiddelde negative log-likelihood? (log 0,5 is ongeveer −0,69, en log 0,25 ongeveer −1,39.)",
        options={
            "mean-p": "Ongeveer 0,375",
            "right": "Ongeveer 1,04",
            "sum": "Ongeveer 2,08",
            "minus": "Ongeveer −1,04",
        },
        correct="right",
        why={
            "mean-p": "Dat is het gemiddelde van de kansen zelf. De NLL middelt hun verrassingen, min hun logs.",
            "sum": "Dat is de totale verrassing van de twee paren. Het gemiddelde deelt die door het aantal paren, 2.",
            "minus": "Dat is het gemiddelde van de logs, zonder het minteken. De verrassing is min de log, dus de NLL is positief.",
            "right": "De verrassingen zijn 0,69 en 1,39; samen 2,08, en over 2 paren is dat 1,04.",
        },
    )
    choice_by_hand
    return (choice_by_hand,)


@app.cell(hide_code=True)
def text_exercise_nll():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De NLL van een tekst

        `nll(P, chars, text)` geeft de gemiddelde NLL van `text` met de tabel `P`. Het woordenboek `row_of`, de loop over de paren en de laatste regel staan er al. Vervang de `...` door stap 2 en 3 van het recept. De check roept `nll(P, ["a", "b", "c"], "abca")` aan, met `P = np.array([[0.1, 0.6, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]])`, en dat moet ongeveer `0.415` geven; hij probeert ook andere tabellen en teksten, ook een met een paar met kans 0.
        """),
        L.task("Laat `nll` de kans van elk paar opzoeken en de verrassing ervan bij `total` optellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nll():
    box_nll = L.code_box("nll", lines=12)
    box_nll
    return (box_nll,)


@app.cell(hide_code=True)
def check_nll(box_nll):
    result_nll = L.check("nll", box_nll.value)
    result_nll
    return (result_nll,)


@app.cell(hide_code=True)
def hints_nll():
    L.hints(
        "nll",
        [
            "Stap 2 en 3 van het recept, zoals in het uitgewerkte voorbeeld.",
            "De rij is het teken op `i`, de kolom dat op `i + 1`. Daarna gaat de verrassing van die kans in `total`.",
            "`p = P[row_of[text[___]], row_of[text[___]]]` en `total = total - np.log(___)`.",
        ],
        why="In \"abca\" heeft \"ab\" 0,6, \"bc\" 0,6 en \"ca\" 0,8. Hun verrassingen zijn ongeveer 0,51, 0,51 en 0,22; samen 1,24, over 3 paren ongeveer 0,415.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_most():
    mo.md(r"""
    ### Oefening 3. De meest verrassende tekst
    """)
    return


@app.cell(hide_code=True)
def choice_most_cell():
    choice_most = L.choice(
        "most-surprising",
        question="Welke van deze teksten verrast het model van de toneelstukken het meest, per teken?",
        options={
            "king": "\"the king\"",
            "hamlet": "\"To be, or not to be\"",
            "right": "\"jazz\"",
            "caps": "\"THE KING\"",
        },
        correct="right",
        why={
            "king": f"\"the king\" is gewone tekst voor de toneelstukken: de gemiddelde NLL is {L.number(C.text_nll('the king'), 2)}, onder de {L.number(C.plays_nll(), 2)} van de toneelstukken als geheel.",
            "hamlet": f"\"To be, or not to be\" staat niet in deze toneelstukken, maar de paren erin zijn gewoon Engels: de gemiddelde NLL is {L.number(C.text_nll('To be, or not to be'), 2)}.",
            "caps": f"De toneelstukken hebben namen in hoofdletters, dus hoofdletters na hoofdletters komen voor: de gemiddelde NLL van \"THE KING\" is {L.number(C.text_nll('THE KING'), 2)}.",
            "right": f"Na \"j\" is een \"a\" zeldzaam in de toneelstukken, na \"a\" een \"z\" nog zeldzamer, en \"zz\" ook: de gemiddelde NLL van \"jazz\" is {L.number(C.text_nll('jazz'), 2)}, zelfs hoger dan de {L.number(C.uniform_nll(), 2)} van een model dat niets weet.",
        },
    )
    choice_most
    return (choice_most,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De verrassing uit [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl), gemiddeld over elk paar van een tekst, meet een language model, met de kansen uit [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl). Je model van de toneelstukken haalt ongeveer 2,45 op de toneelstukken zelf, tegen 4,17 voor een model dat niets weet. Maar een tekst met een paar dat de toneelstukken nooit hadden, krijgt een oneindige NLL. In de volgende les, *Paren die je nooit zag: smoothing*, los je dat op.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_by_hand, choice_most, result_nll):
    L.wrap_up(choice_by_hand, result_nll, choice_most)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
