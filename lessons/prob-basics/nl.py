# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les prob-basics. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("prob-basics", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Dictionaries](/course/open/py-dicts?lang=nl) en [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Een programma telt de woorden van \"to be or not to be\" in een dictionary `counts`. Wat is `counts[\"to\"]`?",
        options={
            "once": "1",
            "right": "2",
            "words": "6",
        },
        correct="right",
        why={
            "once": "Elke key staat maar één keer in een dictionary, maar wat is zijn waarde? Zie oefening 3 van [Dictionaries](/course/open/py-dicts?lang=nl).",
            "words": "Dat is het aantal woorden. Wat telt de waarde onder de key \"to\"? Zie oefening 3 van [Dictionaries](/course/open/py-dicts?lang=nl).",
            "right": "\"to\" komt twee keer langs, dus zijn telling gaat naar 2.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="`A` heeft shape (3, 4). Wat is de shape van `A.sum(axis=0)`?",
        options={
            "right": "(4,)",
            "rows": "(3,)",
            "one": "Eén getal",
        },
        correct="right",
        why={
            "rows": "Dat is `axis=1`. Welke axis verdwijnt bij `axis=0`? Zie *Het idee* in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "one": "Dat is `A.sum()` zonder axis. Zie *Het idee* in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "right": "`axis=0` telt op langs de rijen omlaag: één som per kolom, dus 4.",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een language model zegt niet welk woord er volgt: het zegt hoe waarschijnlijk elk woord is. Een spamfilter zegt hoe waarschijnlijk het is dat een mail spam is. Modellen spreken in kansen, en een kans kun je tellen: hoe vaak iets gebeurt, op het totale aantal pogingen.

    **Een voorbeeld uit het dagelijks leven.** "70% kans op regen" betekent: op dagen zoals deze regende het op ongeveer 7 van de 10.
    """)
    return


@app.cell(hide_code=True)
def predict_sixes():
    guess_sixes = L.guess(
        "sixes",
        question="Je gooit 60 keer met een eerlijke dobbelsteen. Hoe vaak gooi je volgens jou een 6?",
        options={
            "exactly": "Precies 10 keer",
            "about": "Ongeveer 10 keer, een paar meer of minder",
            "never-more": "Nooit meer dan 10 keer",
        },
        reveal="**Ongeveer 10, een paar meer of minder.** Gemiddeld is één worp op de zes een 6, maar elke worp is toeval: precies 10 zessen gooi je maar in ongeveer 1 op de 7 reeksen van 60 worpen, en meer dan 10 in ongeveer 4 op de 10. Kijk hieronder wat meer worpen doen.",
    )
    guess_sixes
    return (guess_sixes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Tel hoe vaak iets gebeurde, en deel door het aantal pogingen: dat deel is de kans erop. Hij ligt tussen 0, nooit, en 1, altijd.
        """),
        L.formula(r"P(A) = \frac{\text{hoe vaak } A \text{ gebeurde}}{\text{aantal pogingen}}"),
        mo.md(r"""
        Een kansverdeling zet elke uitkomst op een rij met zijn kans. Elke poging heeft precies één uitkomst, dus de kansen op alle uitkomsten zijn samen 1. Als elke uitkomst even waarschijnlijk is, kun je ook tellen zonder te gooien: een eerlijke dobbelsteen heeft zes kanten, dus elke kant heeft kans 1/6, ongeveer 0,167.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **kans** | | hoe waarschijnlijk iets is, als getal van 0 tot en met 1: hoe vaak het gebeurt, op het totale aantal pogingen | `count / total` |
        | **kansverdeling**, of **verdeling** | | elke uitkomst met zijn kans; samen zijn ze 1 | `{"heads": 0.7, "tails": 0.3}` |
        | $P(A)$ | P van A | de kans op $A$ | |
        """),
        L.watch_out("Deel door het aantal pogingen, niet door het aantal verschillende uitkomsten. 7 keer kop in 10 worpen is 7 / 10 = 0,7, niet 7 / 2."),
        mo.md(r"""
        2000 worpen met een dobbelsteen, steeds dezelfde. De staven tonen het deel van elke kant in de eerste worpen, zoveel als de slider zegt: eerst 60, zoals bij je gok. De andere grafiek toont het kleinste en het grootste van de zes delen voor elk aantal worpen. Beweeg de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_n_cell():
    def _frame(n):
        n = int(n)
        shares = C.shares(n)
        return (C.picture(n, {"title": "Deel van elke kant", "x": "kant", "y": "deel van de worpen", "title_spread": "Kleinste en grootste deel", "x_spread": "worpen",
                              "alt": f"Zes staven, één per kant van de dobbelsteen, elk zo hoog als het deel van die kant in de eerste {n} worpen: " + ", ".join(f"{_f}: {L.number(_s, 3)}" for _f, _s in zip(range(1, 7), shares)) + f". Een stippellijn bij 1/6, ongeveer 0,167. Een tweede grafiek toont het kleinste en het grootste deel voor elk aantal worpen van 10 tot 2000: na 10 worpen 0 en 0,3, daarna komen ze dicht bij de stippellijn; bij {n} worpen zijn ze {L.number(min(shares), 3)} en {L.number(max(shares), 3)}."}),
                f"""Na {n} worpen lopen de zes delen van <span class="ea-num">{L.number(min(shares), 3)}</span> tot <span class="ea-num">{L.number(max(shares), 3)}</span>; de stippellijn is 1/6. Samen zijn ze precies 1.""")

    explore_n = L.explore("throws", "worpen", _frame, start=60, low=10, high=2000, step=10, role="", task="Ga van 10 worpen naar 2000: wat gebeurt er met de zes delen?")
    explore_n
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij 10 worpen springen de delen alle kanten op, van 0 tot 0,3; bij 2000 liggen ze alle zes dicht bij 1/6, tussen 0,158 en 0,172. Een kans die je uit pogingen telt, is een schatting, en meer pogingen maken hem beter.

    ### Recept

    Zo maak je een kansverdeling uit pogingen:

    1. Tel in een dictionary hoe vaak elke uitkomst gebeurde.
    2. Tel alle tellingen op: het aantal pogingen.
    3. Deel elke telling door dat aantal: de kans op elke uitkomst.
    4. Controleer dat de kansen samen 1 zijn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De verdeling van de letters in "to be or not to be": hoe waarschijnlijk elke letter is, als je zomaar een letter aanwijst. Spaties tellen niet mee. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_letters():
    _text = "to be or not to be"
    # 1. tel elke letter
    _counts = {}
    for _ch in _text:
        if _ch != " ":
            if _ch in _counts:
                _counts[_ch] = _counts[_ch] + 1
            else:
                _counts[_ch] = 1
    print(_counts)
    # 2. het aantal pogingen: alle tellingen opgeteld
    _total = 0
    for _ch in _counts:
        _total = _total + _counts[_ch]
    print(_total)
    # 3. elke telling gedeeld door het totaal
    _probs = {}
    for _ch in _counts:
        _probs[_ch] = _counts[_ch] / _total
    print(_probs)
    # 4. samen zijn ze 1
    _check = 0
    for _ch in _probs:
        _check = _check + _probs[_ch]
    print(_check)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Er zijn 13 letters. `o` komt 4 keer voor, dus P(o) = 4 / 13, ongeveer 0,31; `r` en `n` maar één keer, elk ongeveer 0,08. De zes kansen zijn samen `1.0`. De eenvoudigste language models beginnen precies zo: door te tellen hoe vaak elke letter voorkomt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_marbles():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Knikkers
    """)
    return


@app.cell(hide_code=True)
def choice_marbles_cell():
    choice_marbles = L.choice(
        "marbles",
        question="In een zak zitten 3 rode, 5 blauwe en 2 groene knikkers. Je pakt er zonder te kijken één. Wat is P(blauw)?",
        options={
            "count": "5",
            "colours": "1/3",
            "right": "0,5",
        },
        correct="right",
        why={
            "count": "Zoveel blauwe knikkers zijn er. Een kans is een deel: waarvan? Zie de formule in *Het idee*.",
            "colours": "Dat deelt door het aantal kleuren, maar de kleuren zijn niet even waarschijnlijk. Zie het blok *Let op* in *Het idee*.",
            "right": "5 van de 10 knikkers zijn blauw: 5 / 10 = 0,5.",
        },
    )
    choice_marbles
    return (choice_marbles,)


@app.cell(hide_code=True)
def text_exercise_distribution():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Van tellingen naar een kansverdeling

        Maak `distribution(counts)` af. Hij krijgt een dictionary met hoe vaak elke uitkomst gebeurde, en geeft een nieuwe dictionary terug met de kans op elke uitkomst. Vervang de `...`. De check roept `distribution({"heads": 7, "tails": 3})` aan, met 7 keer kop en 3 keer munt: dat moet `{"heads": 0.7, "tails": 0.3}` geven. Hij probeert ook andere tellingen.
        """),
        L.task("Laat `distribution` de kans op elke uitkomst teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_distribution():
    box_distribution = L.code_box("distribution", lines=9)
    box_distribution
    return (box_distribution,)


@app.cell(hide_code=True)
def check_distribution(box_distribution):
    result_distribution = L.check("distribution", box_distribution.value)
    result_distribution
    return (result_distribution,)


@app.cell(hide_code=True)
def hints_distribution():
    L.hints(
        "distribution",
        [
            "Stap 2 en 3 van het recept: eerst het aantal pogingen, dan elke telling daardoor gedeeld.",
            "Een loop over de keys van `counts` telt het totaal op. Een tweede loop vult een nieuwe dictionary. Wat komt er onder elke key?",
            "`total = total + ___` in de eerste loop, en `probs[outcome] = ___ / total` in de tweede.",
        ],
        why="De eerste loop telt alle tellingen op: 10 pogingen in het voorbeeld. De tweede loop zet `counts[outcome] / total` onder elke key van een nieuwe dictionary: 7 / 10 = 0,7 en 3 / 10 = 0,3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_estimate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Schatten uit worpen

        Het vak gooit 500 keer met een dobbelsteen met `random.randint(1, 6)`, vanaf een seed, en bewaart de worpen in de lijst `throws`. Zet het deel zessen in `p_six`: een schatting van P(6). Vervang de `...`. De laatste regel is `p_six`, zodat de check hem ziet. De check probeert ook een andere seed.
        """),
        L.task("Zet het deel zessen in `throws` in `p_six`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_estimate():
    box_estimate = L.code_box("estimate", lines=14)
    box_estimate
    return (box_estimate,)


@app.cell(hide_code=True)
def check_estimate(box_estimate):
    result_estimate = L.check("estimate", box_estimate.value)
    result_estimate
    return (result_estimate,)


@app.cell(hide_code=True)
def hints_estimate():
    L.hints(
        "estimate",
        [
            "De formule in *Het idee*: hoe vaak een 6 gebeurde, gedeeld door het aantal pogingen.",
            "Tel de zessen met een loop over `throws` en een `if`. Hoeveel pogingen waren er?",
            "`if t == 6:` en dan `sixes = sixes + 1`, en na de loop `p_six = sixes / ___`.",
        ],
        why="Een loop over `throws` telt de worpen die een 6 zijn: 80 stuks. Gedeeld door `len(throws)`, 500, geeft dat 0,16: dicht bij 1/6, maar niet precies, want 500 worpen zijn veel, maar niet eindeloos veel.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Dictionaries](/course/open/py-dicts?lang=nl) telde je woorden; gedeeld door het totaal worden tellingen kansen, en samen vormen ze een kansverdeling. Later in de course geeft een model zelf een kansverdeling: een kans voor elk volgend woord, of voor elke klasse. In de volgende les, *Voorwaardelijke kans*, tel je alleen onder de gevallen waarin iets anders al gebeurd is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_marbles, result_distribution, result_estimate):
    L.wrap_up(choice_marbles, result_distribution, result_estimate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
