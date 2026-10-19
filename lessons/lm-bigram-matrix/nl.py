# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-bigram-matrix. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-bigram-matrix", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) en [Broadcasting](/course/open/np-broadcasting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_probs():
    quick_probs = L.choice(
        "quick-probs",
        question="Je `bigram_probs(counts, \"t\")` geeft een dictionary. Wat staat erin?",
        options={
            "right": "Elk teken dat ooit na \"t\" komt, met zijn kans",
            "counts": "Elk paar dat met \"t\" begint, met hoe vaak het voorkomt",
            "one": "Het ene teken dat het vaakst na \"t\" komt",
        },
        correct="right",
        why={
            "counts": "Dat zijn de tellingen die `count_bigrams` geeft; `bigram_probs` deelt ze door hun totaal. Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
            "one": "Een language model geeft een kans voor elk mogelijk volgend token, niet één antwoord. Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
            "right": "De tellingen van de paren die met \"t\" beginnen, elk gedeeld door hun totaal: samen tellen ze op tot 1.",
        },
    )
    quick_probs
    return (quick_probs,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`N` heeft de shape (65, 65). Welke shape heeft `N.sum(axis=1, keepdims=True)`?",
        options={
            "flat": "(65,): één som per rij, en de opgetelde axis is weg",
            "cols": "(1, 65): één som per kolom",
            "right": "(65, 1): één som per rij, als kolom",
        },
        correct="right",
        why={
            "flat": "Dat is `N.sum(axis=1)` zonder `keepdims`: de axis verdwijnt. Zie [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "cols": "`axis=1` telt op langs elke rij, dus er is één som per rij, niet per kolom. Zie [Broadcasting](/course/open/np-broadcasting?lang=nl).",
            "right": "Eén som per rij, en `keepdims=True` houdt de opgetelde axis met lengte 1, dus `N / sums` deelt elke rij door haar eigen som.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je `bigram_probs` rekent de kansen na één teken uit, en gaat daarvoor alle 1.403 paren langs. Om je model te laten schrijven, zoals in de lessen hierna, heeft het voor elk teken dat het schrijft opnieuw de kansen na het laatste teken nodig. Je kunt ze beter één keer allemaal uitrekenen en in één tabel bewaren: 65 rijen en 65 kolommen, één rij voor elk teken. Eén regel NumPy maakt dan van elke rij van die tabel een kansverdeling. Dit is nog steeds de tweede stap van je eigen LLM.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           1, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        L.course_versions(),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** De afstandstabel achter in een wegenatlas: een rij en een kolom voor elke stad, en in elk vak de afstand tussen de twee. Je zoekt elk paar in één keer op, in plaats van het elke keer uit te rekenen. Die tabel leest in beide richtingen hetzelfde; de tabel van deze les niet.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_row():
    guess_row = L.guess(
        "space-row",
        question="In de tabel met kansen bevat de rij van een teken de kansen op wat er na dat teken komt, voor alle 65 tekens. Na een spatie begint een nieuw woord. Hoe groot denk je dat het grootste getal in de rij van de spatie is?",
        options={
            "right": "Ongeveer 0,15: veel verschillende letters beginnen vaak een woord",
            "half": "Ongeveer 0,4: één letter begint bijna de helft van alle woorden",
            "high": "Ongeveer 0,8: bijna elk woord begint met dezelfde letter",
        },
        reveal=f"**Ongeveer 0,15: {L.number(C.top_after(C.course_probs(), ' ', 1)[0][1], 2)}, voor \"t\".** Woorden beginnen met veel verschillende letters, dus de rij van de spatie is verspreid: \"a\", \"s\" en \"h\" komen elk ongeveer half zo vaak na een spatie als \"t\". De rij van \"q\" is het omgekeerde: na een \"q\" komt elke keer een \"u\", dus die rij heeft één vak met 1 en 64 vakken met 0. Een rij laat zien hoe zeker het model is van wat er volgt: onzeker na een spatie, zeker na een \"q\".",
    )
    guess_row
    return (guess_row,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Zet de 65 tekens in een vaste volgorde, `chars = sorted(set(text))`: het teken `chars[i]` krijgt rij `i` en kolom `i`. In rij $i$ en kolom $j$ van de **telmatrix** $N$ staat hoe vaak `chars[j]` na `chars[i]` komt. Een paar dat nooit voorkomt, krijgt een 0. Rij $i$ bevat alle tellingen na `chars[i]`.

        Deel elk getal door de som van zijn rij, en rij $i$ wordt de kansverdeling van het volgende teken na `chars[i]`: haar getallen tellen op tot 1.
        """),
        L.formula(r"[[pred:P]][i, j] = \frac{[[data:N]][i, j]}{\text{de som van rij } i \text{ van } [[data:N]]}"),
        mo.md(r"""
        In NumPy is dat één regel, met de broadcasting uit [Broadcasting](/course/open/np-broadcasting?lang=nl): `N.sum(axis=1, keepdims=True)` geeft de som van elke rij, met de shape (65, 1), en `N / sums` deelt elke rij door haar eigen som. De kansen na een teken zijn dan een rij van de matrix: `P[i]`, zonder de paren opnieuw langs te gaan. Daarvoor heeft elke rij minstens één telling nodig: een teken waar nooit iets na komt, zoals een laatste teken dat verder nergens in de tekst staat, krijgt een rij met alleen nullen, en 0 / 0 geeft `nan`, "not a number": geen getal. In de toneelstukken komt na elk teken minstens één keer iets.

        Een **heatmap** laat een matrix zien als een rooster van vakjes, feller gekleurd waar het getal groter is, zodat je het patroon in één oogopslag ziet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **telmatrix** | | de tellingen van elk paar als tabel: rij $i$, kolom $j$ is hoe vaak `chars[j]` na `chars[i]` komt | `N[i, j]` |
        | $P[i, j]$ | P i j | de kans dat `chars[j]` na `chars[i]` komt: rij $i$ van $N$ gedeeld door haar som | `P[i, j]` |
        | **heatmap** | | een beeld van een matrix: een vakje per getal, feller gekleurd waar het groter is | |
        """),
        L.watch_out("Rijen en kolommen zijn hier niet uitwisselbaar. Rij `i` is wat er *na* `chars[i]` komt, en telt op tot 1. De som van een kolom kan van alles zijn: kolom `j` bevat hoe waarschijnlijk `chars[j]` is na elk teken, en die getallen horen bij verschillende verdelingen."),
        mo.md(r"""
        De figuur: de heatmap van de rijen en kolommen van de spatie en de 26 kleine letters, uit alle toneelstukken. De volledige matrix heeft 65 rijen en 65 kolommen; hoofdletters, leestekens en het regeleinde zijn hier weggelaten, zodat de letters te lezen zijn.
        """),
        mo.Html(C.heatmap(C.course_probs(), {"title": "Wat komt na wat", "next": "het volgende teken", "rows": "Elke rij: het teken ervoor",
                                              "alt": "Een rooster van 27 bij 27 vakjes voor de spatie en de letters a tot en met z, feller naarmate het waarschijnlijker is. De rij van q heeft één fel vakje, onder u. De rij van de spatie is zwak gekleurd en verspreid over veel letters, met haar felste vakje onder t."})),
        mo.md(r"""
        De rij van "q" heeft één fel vakje, onder "u": na een "q" komt altijd een "u". De rij van de spatie is zwak gekleurd en verspreid: een woord kan met bijna elke letter beginnen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om van de bigram-tellingen een matrix van kansen te maken:

    1. Maak de telmatrix: voor elk teken `a` uit `chars` een rij, en daarin, voor elk teken `b`, de telling van het paar `a + b`, of 0 als het nooit voorkomt.
    2. Maak van de lijst rijen een NumPy-array: `np.array(rows)`.
    3. Tel elke rij op en houd de axis: `sums = N.sum(axis=1, keepdims=True)`.
    4. Deel: `P = N / sums`. Rij `i` van `P` is de verdeling van wat er na `chars[i]` komt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De matrix van de korte tekst "abab ba", met zijn drie tekens, " ", "a" en "b". Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_matrix():
    import numpy as np

    _counts = {"ab": 2, "ba": 2, "b ": 1, " b": 1}   # de paren van "abab ba"
    _chars = [" ", "a", "b"]
    # 1. een rij voor elk teken, met de telling van elk paar, of 0
    _rows = []
    for _a in _chars:
        _row = []
        for _b in _chars:
            if _a + _b in _counts:
                _row.append(_counts[_a + _b])
            else:
                _row.append(0)
        _rows.append(_row)
    # 2. de rijen als array
    _N = np.array(_rows)
    print(_N)
    # 3. de som van elke rij, als kolom
    _sums = _N.sum(axis=1, keepdims=True)
    print(_sums)
    # 4. elke rij gedeeld door haar eigen som
    _P = _N / _sums
    print(_P)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De loop over `_b` staat binnen de loop over `_a`: voor elk teken `_a` gaat hij alle drie de tekens `_b` langs en vult hij één rij, dus `_rows` krijgt drie rijen van drie getallen. De rij van "b" bevat 1, 2 en 0: na "b" kwam één keer een spatie en twee keer "a". Haar som is 3, dus ze wordt 0,33, 0,67 en 0. De rij van "a" heeft alleen het paar "ab", twee keer, dus ze wordt 0, 0, 1. Elke rij van `_P` telt op tot 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_q():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Lees de heatmap
    """)
    return


@app.cell(hide_code=True)
def choice_q_cell():
    choice_q = L.choice(
        "space-column",
        question="In de heatmap uit het idee is de kolom van de spatie in veel rijen fel gekleurd. Wat zegt dat?",
        options={
            "starts": "Woorden beginnen vaak met die letters",
            "right": "Die letters staan vaak aan het eind van een woord",
            "space": "Na een spatie zijn die letters waarschijnlijker dan de andere",
        },
        correct="right",
        why={
            "starts": "Een kolom is wat er volgt: een fel vakje in de kolom van de spatie betekent dat er na de letter van die rij een spatie komt. Zie het blok *Let op* in *Het idee*.",
            "space": "Wat er na een spatie komt, staat in de rij van de spatie, niet in haar kolom. Zie *Het idee*.",
            "right": "Een fel vakje in rij \"e\", kolom ␣, betekent dat er na een \"e\" vaak een spatie komt: de \"e\" sluit vaak een woord af.",
        },
    )
    choice_q
    return (choice_q,)


@app.cell(hide_code=True)
def text_exercise_matrix():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De telmatrix

        `bigram_matrix(counts, chars)` maakt de telmatrix: een rij voor elk teken uit `chars`, met de telling van elk paar. De loops staan er al. Vervang de `...` door stap 1 van het recept. De check roept `bigram_matrix({"ab": 3, "ac": 1, "ba": 2}, ["a", "b", "c"])` aan, en dat moet `[[0, 3, 1], [2, 0, 0], [0, 0, 0]]` geven; hij probeert ook andere tellingen en een andere volgorde van `chars`. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/bigram_matrix.py`.
        """),
        L.task("Laat `bigram_matrix` de telling van het paar `a + b` in de rij zetten, of 0 als het paar nooit voorkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matrix():
    box_matrix = L.code_box("bigram-matrix", lines=12)
    box_matrix
    return (box_matrix,)


@app.cell(hide_code=True)
def check_matrix(box_matrix):
    result_matrix = L.check("bigram-matrix", box_matrix.value)
    result_matrix
    return (result_matrix,)


@app.cell(hide_code=True)
def hints_matrix():
    L.hints(
        "bigram-matrix",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld.",
            "De meeste paren staan niet in `counts`. Kijk eerst of `a + b` erin staat: dan gaat zijn telling de rij in, en anders een 0.",
            "`if a + b in counts:`, dan `row.append(counts[___])`, en na `else:` de regel `row.append(___)`.",
        ],
        why="Met `[\"a\", \"b\", \"c\"]` krijgt de rij van \"a\" de tellingen van \"aa\", \"ab\" en \"ac\": 0, 3 en 1. De rij van \"b\" krijgt 2 voor \"ba\" en twee keer 0. Geen paar begint met \"c\", dus haar rij is helemaal 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Elke rij een verdeling

        `to_probs(N)` deelt elke rij van een telmatrix door haar som. Vervang de `...` door stap 3 van het recept. De check roept `to_probs(np.array([[3, 1], [0, 2]]))` aan, en dat moet `[[0.75, 0.25], [0, 1]]` geven; hij probeert ook andere matrices. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/bigram_rows.py`.
        """),
        L.task("Laat `to_probs` elke rij van `N` door haar eigen som delen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("to-probs", lines=5)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("to-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "to-probs",
        [
            "Stap 3 van het recept: de som van elke rij, in een shape waarmee broadcasting elke rij door haar eigen som deelt.",
            "Een rij loopt langs axis 1. Zonder `keepdims` hebben de sommen de shape (rijen,), en koppelt broadcasting ze in plaats daarvan aan de kolommen.",
            "`sums = N.sum(axis=___, keepdims=___)`.",
        ],
        why="`[[3, 1], [0, 2]]` heeft de rijsommen 4 en 2, als kolom `[[4], [2]]`. Delen geeft `[[3/4, 1/4], [0/2, 2/2]]`: `[[0.75, 0.25], [0, 1]]`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder telt de paren van de toneelstukken met je `count_bigrams` uit [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl), of met die van de course als de jouwe ontbreekt of stuk is, maakt de telmatrix met je `bigram_matrix`, en maakt er kansen van met je `to_probs`. Hij werkt zodra oefening 2 en 3 af zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_table(result_matrix, result_probs):
    result_matrix, result_probs  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `bigram_matrix` en je `to_probs` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*De tabel maken met je functies duurt langer dan 30 seconden. Gaat elke functie één keer over de tekst of de tekens?*",
            "counts": "*Je `count_bigrams` in `work/projects/llm/bigrams.py` stopt met een fout, of telt de paren van de toneelstukken niet meer zoals die van de course. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 van [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
            "matrix": "*De `bigram_matrix` in `work/projects/llm/bigram_matrix.py` geeft niet meer de tellingen die de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
            "rows": "*De `to_probs` in `work/projects/llm/bigram_rows.py` geeft niet meer de kansen die de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
        }[_result])
    else:
        _table, _yours = _result
        _out = mo.vstack([
            mo.md(f"""Je tabel heeft de shape {_table.shape}, en elke rij telt op tot 1. De kansen na een "t", afgelezen uit rij "t" van je tabel:"""),
            draw.next_word_bars("t", C.top_after(_table, "t"), picked=0, picked_label="waarschijnlijkst", title="Je tabel, rij \"t\""),
            mo.md("""Dezelfde getallen als je `bigram_probs` in de vorige les gaf, maar nu liggen ze allemaal klaar, voor elk teken tegelijk. In de volgende les schrijft je model ermee."""
                  if _yours else """De tellingen komen van de `count_bigrams` van de course: de jouwe ontbreekt of is stuk, zoals de melding onder *Waar je project staat* zegt. De tabel maken je `bigram_matrix` en je `to_probs` nog steeds zelf. In de volgende les schrijft je model ermee."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tellingen uit [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) zijn nu één matrix, en de broadcasting uit [Broadcasting](/course/open/np-broadcasting?lang=nl) maakt van al haar rijen in één keer kansverdelingen. Je project `llm` heeft nu ook `bigram_matrix.py` en `bigram_rows.py`. In de volgende les, *Sampling: laat het model schrijven*, trekt je model het volgende teken uit een rij van deze tabel, steeds opnieuw, en schrijft het zelf tekst.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_q, result_matrix, result_probs):
    L.wrap_up(choice_q, result_matrix, result_probs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
