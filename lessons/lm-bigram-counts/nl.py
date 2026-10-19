# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-bigram-counts. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-bigram-counts", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) en [Project: welke letters gebruikt een boek?](/course/open/py-project-letter-counts?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gives():
    quick_gives = L.choice(
        "quick-gives",
        question="Wat geeft een language model terug voor de tekst tot nu toe?",
        options={
            "one": "Het ene juiste volgende token",
            "right": "Een kans voor elk mogelijk volgend token",
            "label": "Een label dat zegt waar de tekst over gaat",
        },
        correct="right",
        why={
            "one": "Een model weet niet welk token goed is; het zegt hoe waarschijnlijk elk token is. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "label": "Een language model voorspelt wat er volgt, niet waar een tekst over gaat. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "Een kans voor elk token dat het kent, en samen tellen die op tot 1.",
        },
    )
    quick_gives
    return (quick_gives,)


@app.cell(hide_code=True)
def quick_check_pairs():
    quick_pairs = L.choice(
        "quick-pairs",
        question="Wat geeft jouw `count_pairs(\"Hi, ho\")`?",
        options={
            "all": "`{\"hi\": 1, \"i,\": 1, \", \": 1, \" h\": 1, \"ho\": 1}`",
            "capital": "`{\"Hi\": 1, \"ho\": 1}`",
            "right": "`{\"hi\": 1, \"ho\": 1}`",
        },
        correct="right",
        why={
            "all": "`count_pairs` telt een paar alleen als beide tekens letters zijn. Zie [Project: welke letters gebruikt een boek?](/course/open/py-project-letter-counts?lang=nl).",
            "capital": "`count_pairs` zet de tekst eerst in kleine letters, met `text.lower()`. Zie [Project: welke letters gebruikt een boek?](/course/open/py-project-letter-counts?lang=nl).",
            "right": "De tekst wordt \"hi, ho\", en alleen \"hi\" en \"ho\" zijn twee letters naast elkaar.",
        },
    )
    quick_pairs
    return (quick_pairs,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je project `llm` heeft `count_pairs`: die telt welke letter op welke volgt. Dat is bijna een language model, zoals in [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl): uit de paren die met een t beginnen, kun je zeggen wat er waarschijnlijk na een t komt. Bijna, want hij slaat spaties en leestekens over en maakt hoofdletters klein, dus een model dat daaruit gemaakt is, kan nooit een woord afsluiten of een zin beginnen. Vandaag tel je elk paar tekens in de toneelstukken van Shakespeare, en maak je van de tellingen kansen: de tweede stap van je eigen LLM.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           1, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Een vuistregel voor het weer: na een droge dag is de volgende dag meestal ook droog. Dat weet je door dagen te tellen, niet doordat je het weer begrijpt, en je kijkt alleen naar gisteren.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_pairs():
    guess_pairs = L.guess(
        "pairs",
        question="De toneelstukken van Shakespeare in de course gebruiken 65 verschillende tekens, dus er zijn 65 · 65 = 4.225 mogelijke paren tekens. Hoeveel daarvan denk je dat er minstens één keer voorkomen in de 1,1 miljoen tekens?",
        options={
            "all": "Bijna alle 4.225: in zoveel tekst komt elk paar langs",
            "few": "Ongeveer 100",
            "right": "Ongeveer een derde ervan",
        },
        reveal=f"**Ongeveer een derde: {C.grouped(C.distinct(), L.lang)}.** De meeste paren komen nooit voor: geen woord heeft \"qz\" of \"Xk\", en na een \"q\" komt elke keer een \"u\", alle {C.grouped(C.after_q(), L.lang)} keer. Een model dat uit deze tellingen gemaakt is, geeft elk paar dat het nooit zag een kans van 0.",
    )
    guess_pairs
    return (guess_pairs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **bigram** is een paar tokens naast elkaar; hier twee tekens, zoals "th". Tel elk bigram van een tekst, zoals `count_pairs` deed, maar nu telt elk teken mee: spaties, hoofdletters, leestekens en regeleinden ook. Dan geven de tellingen de kans dat $b$ na $a$ komt: van alle paren die met $a$ beginnen, het deel dat $ab$ is.
        """),
        L.formula(r"[[pred:P]](b \mid a) = \frac{[[data:\mathrm{count}(ab)]]}{[[data:\text{hoeveel paren met } a \text{ beginnen}]]}"),
        mo.md(r"""
        Dat is een **bigram-model**: een language model dat alleen naar het laatste token kijkt. De tekst tot nu toe kan lang zijn, maar voor het volgende teken gebruikt het alleen het laatste. Die vereenvoudiging is de **Markov-aanname**:
        """),
        L.formula(r"P(x_{t+1} \mid x_1, \dots, x_t) \approx P(x_{t+1} \mid x_t)"),
        mo.md(r"""
        Een **n-gram** is een reeks van $n$ tokens achter elkaar. Een n-gram-model kijkt naar de laatste $n - 1$ tokens: een bigram-model is een 2-gram-model. Het model uit [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) keek naar de laatste drie tekens, dus dat was een 4-gram-model. Meer tokens terugkijken geeft betere voorspellingen, maar ook veel meer reeksen die nooit in de tekst voorkomen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **bigram** | | twee tokens naast elkaar, zoals "th" | `text[i:i + 2]` |
        | **n-gram** | | een reeks van $n$ tokens achter elkaar; een n-gram-model kijkt naar de laatste $n - 1$ | |
        | **Markov-aanname** | | het volgende token hangt alleen af van de laatste paar tokens, niet van alles ervoor | |
        | $\mathrm{count}(ab)$ | count van a b | hoe vaak het paar $ab$ in de tekst voorkomt | `counts[a + b]` |
        | hoeveel paren met $a$ beginnen | | de tellingen van alle paren die met $a$ beginnen bij elkaar opgeteld, wat er ook als tweede komt | `total` |
        | $P(b \mid a)$ | P van b gegeven a | de kans dat $b$ na $a$ komt | `probs[b]` |
        | $\approx$ | is ongeveer | ongeveer gelijk aan | |
        """),
        L.watch_out("Een paar dat nooit in de tekst voorkomt, krijgt een kans van 0: het model zegt dan dat het nooit kan gebeuren, ook als het wel kan, alleen niet in deze toneelstukken. Een latere les, *Paren die je nooit zag: smoothing*, lost dat op."),
        mo.md(r"""
        De figuur: de kansen na een "t", uit de tellingen van alle toneelstukken van Shakespeare in de course. Alleen de vijf waarschijnlijkste tekens staan erin; ␣ is een spatie.
        """),
        draw.next_word_bars("t", C.top(C.course_probs("t")), picked=0, picked_label="waarschijnlijkst", title="Na \"t\""),
        mo.md(r"""
        Na een "t" komt ongeveer een derde van de keren een "h", zoals in "the", "that" en "thou", en ongeveer een kwart van de keren een spatie, aan het eind van een woord.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om van een tekst een bigram-model te maken, en het te gebruiken na een teken `a`:

    1. Tel elk bigram: voor elke plek `i` het paar `text[i:i + 2]`, één erbij elke keer dat het langskomt.
    2. Neem het laatste teken `a` van de tekst tot nu toe.
    3. Tel de tellingen op van alle paren die met `a` beginnen: dat is `total`.
    4. Deel de telling van elk van die paren door `total`: de kans dat zijn tweede teken na `a` komt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een bigram-model van de tekst "to be or not to be", en zijn kansen na een "o". Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_bigrams():
    _text = "to be or not to be"
    # 1. tel elk bigram
    _counts = {}
    for _i in range(len(_text) - 1):
        _pair = _text[_i:_i + 2]
        if _pair in _counts:
            _counts[_pair] = _counts[_pair] + 1
        else:
            _counts[_pair] = 1
    print(_counts)
    # 2. het laatste teken van de tekst tot nu toe
    _so_far = "to be or no"
    _a = _so_far[-1]
    # 3. hoeveel paren ermee beginnen
    _total = 0
    for _pair in _counts:
        if _pair[0] == _a:
            _total = _total + _counts[_pair]
    print(_total)
    # 4. elk volgend teken, met zijn telling gedeeld door het totaal
    for _pair in _counts:
        if _pair[0] == _a:
            print([_pair[1], _counts[_pair] / _total])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tekst tot nu toe is "to be or no", en het model kijkt alleen naar het laatste teken daarvan, een "o". Vier paren beginnen met "o": twee keer "o ", in "to be", en "or" en "ot" elk één keer. Na een "o" geeft het model een spatie dus 2 / 4 = 0,5, en "r" en "t" elk 0,25. Elk ander teken krijgt 0: in deze korte tekst kwam er na een "o" nooit iets anders. Elke regel wordt als lijst geprint, zodat de spatie tussen zijn aanhalingstekens te zien is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_markov():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Alleen het laatste teken
    """)
    return


@app.cell(hide_code=True)
def choice_markov_cell():
    choice_markov = L.choice(
        "markov",
        question="Je bigram-model krijgt de tekst \"the k\", en later de tekst \"a k\". Wat geeft het na elk van de twee?",
        options={
            "different": "Andere kansen: na \"the k\" komt vaker een \"i\", zoals in \"the king\"",
            "right": "Dezelfde kansen: die na \"k\"",
            "nothing": "Niets: deze teksten zijn langer dan één paar",
        },
        correct="right",
        why={
            "different": "Dat klopt voor echte tekst, maar een bigram-model ziet het niet: het kijkt alleen naar het laatste teken. Zie de Markov-aanname in *Het idee*.",
            "nothing": "De tekst tot nu toe mag elke lengte hebben; het model gebruikt het laatste teken. Zie *Het idee*.",
            "right": "Beide teksten eindigen op \"k\", en een bigram-model kijkt alleen naar het laatste teken: het geeft beide keren de kansen na \"k\".",
        },
    )
    choice_markov
    return (choice_markov,)


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tel elk paar

        `count_bigrams(text)` telt elk paar tekens naast elkaar, zoals je `count_pairs` deed, maar zonder iets over te slaan: spaties, hoofdletters, leestekens en regeleinden tellen ook mee. De loop en het paar staan er al. Vervang de `...` door het tellen, stap 1 van het recept. De check roept `count_bigrams("To be, or not")` aan, en probeert ook andere teksten. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/bigrams.py`.
        """),
        L.task("Laat `count_bigrams` elk paar tekens tellen, wat het ook is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-bigrams", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-bigrams", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-bigrams",
        [
            "Stap 1 van het recept: hetzelfde tellen als in je `count_pairs`, met een `if` en een `else`.",
            "Staat `pair` al in `counts`, dan krijgt het er 1 bij; staat het er niet in, dan begint het bij 1. Laat de controle op letters weg, en laat de tekst zoals hij is.",
            "`if pair in counts:`, dan `counts[pair] = counts[pair] + ___`, en na `else:` de regel `counts[pair] = ___`.",
        ],
        why="\"To be, or not\" heeft 13 tekens, dus 12 paren, allemaal verschillend: \"To\", \"o \", \" b\", \"be\", \"e,\", \", \", \" o\", \"or\", \"r \", \" n\", \"no\" en \"ot\", elk één keer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Van tellingen naar kansen

        `bigram_probs(counts, a)` geeft de kansen na het teken `a`, als dictionary: voor elk teken dat ooit na `a` komt, de kans dat het dat doet. De twee loops staan er al. Vervang de twee `...` door stap 3 en 4 van het recept. De check roept `bigram_probs({"ab": 3, "ac": 1, "ba": 2}, "a")` aan, en dat moet `{"b": 0.75, "c": 0.25}` geven; hij probeert ook andere tellingen. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/bigram_model.py`.
        """),
        L.task("Laat `bigram_probs` de telling van elk paar dat met `a` begint delen door `total`: de tellingen van al die paren bij elkaar opgeteld."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("bigram-probs", lines=11)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("bigram-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "bigram-probs",
        [
            "Stap 3 en 4 van het recept. De eerste loop telt alleen op; de tweede deelt.",
            "In de eerste loop groeit `total` met de telling van elk paar dat met `a` begint. In de tweede loop is de key het tweede teken van het paar, en de waarde zijn telling als deel van `total`.",
            "`total = total + counts[___]`, en `probs[pair[1]] = counts[pair] / ___`.",
        ],
        why="Met `{\"ab\": 3, \"ac\": 1, \"ba\": 2}` en `\"a\"` beginnen de paren \"ab\" en \"ac\" met \"a\": `total` is 3 + 1 = 4. Dan krijgt \"b\" 3 / 4 = 0,75 en \"c\" 1 / 4 = 0,25; \"ba\" begint met \"b\" en telt niet mee.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder telt elk paar in de toneelstukken van Shakespeare met je eigen `count_bigrams`, en vraagt je eigen `bigram_probs` wat er na een "t" en na een "q" komt. Hij gebruikt je eigen modules, dus hij werkt zodra oefening 2 en 3 af zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_model(result_count, result_probs):
    result_count, result_probs  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `count_bigrams` en je `bigram_probs` in je project, en gebruikt deze cel ze.*")
    elif _result == "slow":
        _out = mo.md("*Het tellen van de toneelstukken met je `count_bigrams` in `work/projects/llm/bigrams.py` duurt langer dan 30 seconden. Gaat hij één keer over de tekst?*")
    elif _result == "counts":
        _out = mo.md("*De `count_bigrams` in `work/projects/llm/bigrams.py` telt niet meer zoals de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw.*")
    elif _result == "probs":
        _out = mo.md("*De `bigram_probs` in `work/projects/llm/bigram_model.py` geeft niet meer de kansen die de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*")
    else:
        _pairs, _after_t, _after_q = _result
        _out = mo.vstack([
            mo.md(f"""Je `count_bigrams` vond <span class="ea-num">{C.grouped(_pairs, L.lang)}</span> verschillende paren in de toneelstukken. Je `bigram_probs` zegt wat er na een "t" komt:"""),
            draw.next_word_bars("t", C.top(_after_t), picked=0, picked_label="waarschijnlijkst", title="Je model, na \"t\""),
            mo.md(f"""En na een "q" geeft het een "u" een kans van <span class="ea-num">{L.number(_after_q.get("u", 0), 2)}</span>: elke "q" in de toneelstukken wordt gevolgd door een "u". Dit is je eerste language model. In de lessen na de volgende schrijft het zelf tekst."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De paartellingen uit [Project: welke letters gebruikt een boek?](/course/open/py-project-letter-counts?lang=nl) werden een language model, zoals in [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl): een model dat alleen naar het laatste teken kijkt, volgens de Markov-aanname. Je project `llm` heeft nu ook `bigrams.py` en `bigram_model.py`. In de volgende les, *De bigramtabel als matrix van kansen*, zet je alle tellingen in één tabel van 65 bij 65, en maak je van elke rij in één keer een kansverdeling.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_markov, result_count, result_probs):
    L.wrap_up(choice_markov, result_count, result_probs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
