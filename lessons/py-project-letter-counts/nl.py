# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-project-letter-counts. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.
# termen:allow functions

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, plots
    from everything_ai.lesson import lesson

    L = lesson("py-project-letter-counts", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl) en [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_comprehension():
    quick_comprehension = L.choice(
        "quick-comprehension",
        question="Wat laat dit programma zien?\n\n```python\nscores = {\"a\": 3, \"b\": 5}\nnames = [\"b\", \"a\"]\nprint([scores[n] for n in names])\n```",
        options={
            "keys": "`[\"b\", \"a\"]`",
            "order": "`[3, 5]`",
            "values": "`[5, 3]`",
        },
        correct="values",
        why={
            "keys": "De comprehension maakt een lijst van `scores[n]`, de waarden, niet van de namen. Zie het recept in [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).",
            "order": "De nieuwe lijst volgt de volgorde van `names`, niet die van `scores`. Welke naam komt eerst?",
            "values": "Voor elke naam in `names`, in die volgorde, pakt de comprehension zijn waarde: eerst `scores[\"b\"]`, dan `scores[\"a\"]`.",
        },
    )
    quick_comprehension
    return (quick_comprehension,)


@app.cell(hide_code=True)
def quick_check_bar():
    quick_bar = L.choice(
        "quick-bar",
        question="Je wilt een staaf voor elk van de dagen `ma`, `di` en `wo`, zo hoog als de regen van die dag. Welke regel tekent dat?",
        options={
            "bar": "`plt.bar([\"ma\", \"di\", \"wo\"], [3, 5, 2])`",
            "plot": "`plt.plot([\"ma\", \"di\", \"wo\"], [3, 5, 2])`",
            "swapped": "`plt.bar([3, 5, 2], [\"ma\", \"di\", \"wo\"])`",
        },
        correct="bar",
        why={
            "bar": "`plt.bar` krijgt eerst de namen, langs de onderkant, en dan de hoogtes.",
            "plot": "`plt.plot` verbindt punten met een lijn. Welke functie tekent staven? Zie de tabel met nieuwe woorden in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
            "swapped": "De volgorde van de twee lijsten doet ertoe. Welke lijst gaat langs de onderkant? Zie het recept in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl).",
        },
    )
    quick_bar
    return (quick_bar,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Dit is de eerste stap van je eigen language model. Een language model raadt wat er in een tekst hierna komt. Het eenvoudigste dat er is, hoeft alleen te weten welke letter meestal op welke volgt: na `t` komt veel vaker `h` dan `x`. Dus begin je met tellen, in een echt boek: elke letter, en elk paar letters naast elkaar. Beide telfuncties bewaar je in je project; in een latere les worden de paartellingen een model dat schrijft.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           0, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Het toetsenbord van je telefoon dat het volgende woord voorstelt. Het heeft gezien welke woorden mensen na welke typen, en heeft dat geteld.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_letter():
    guess_letter = L.guess(
        "letter",
        question="Welke letter komt volgens jou het vaakst voor in toneelstukken van Shakespeare, als je spaties en tekens weglaat?",
        options={
            "e": "`e`",
            "t": "`t`",
            "a": "`a`",
        },
        reveal="**`e`.** In het Engels is `e` de meest voorkomende letter, en Shakespeare is geen uitzondering: in het boek van deze les komt hij meer dan 100.000 keer voor. Je telt het zelf in oefening 3.",
    )
    guess_letter
    return (guess_letter,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Tellen is een loop met een dictionary: elke letter is een key, en de waarde ervan is hoe vaak hij voorkomt. Een paar is twee tekens naast elkaar, dus een slice van twee: `text[i:i + 2]`. Twee nieuwe stukken gereedschap: `.lower()`, zodat `T` en `t` als dezelfde letter tellen, en `and`, zodat een paar alleen telt als beide tekens letters zijn.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **`.lower()`** | geeft de tekst in kleine letters | `"To Be".lower()` geeft `"to be"` |
    | **`in` bij een tekst** | `True` als het teken links ervan in de tekst rechts ervan voorkomt | `"e" in "aeiou"` geeft `True` |
    | **`and`** | alleen `True` als de voorwaarde links ervan en die rechts ervan allebei `True` zijn | `"e" in "aeiou" and "x" in "aeiou"` geeft `False` |
    | **paar** | twee tekens naast elkaar in een tekst | `text[i:i + 2]` |
    | **telling** | hoe vaak een key voorkomt: de waarde onder die key | `counts["e"]` |
    | **project** | je eigen code die les na les groeit, in `work/projects/`; dit project heet `llm` | `from llm import letters` |

    Het boek is *Tiny Shakespeare*: zo'n 40.000 regels uit toneelstukken van William Shakespeare, die vrij van rechten zijn (public domain). Het staat in de map van de course, als `data/tiny-shakespeare.txt`.
    """)
    return


@app.cell(hide_code=True)
def picture_pair():
    mo.vstack([
        draw.list_boxes(list("to be"), index_label="index", span=(1, 3), span_label="text[1:3]", title="Een paar is een slice van twee"),
        mo.md(r"""
        Boven elk teken staat zijn index; het lege vakje op index 2 is de spatie. De slice `text[1:3]` pakt de tekens op index 1 en 2: `"o "`. In dat paar zit een spatie, dus het telt niet mee. `text[3:5]` is `"be"`: allebei letters, dus dat paar telt wel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("De eerste keer dat een letter langskomt, staat hij nog niet in `counts`: `counts[ch] + 1` stopt dan. Kijk eerst met `if ch in counts:`, en begin een nieuwe letter bij 1."),
        mo.md(r"""
        ### Recept

        Zo tel je wat er in een tekst staat:

        1. Lees de tekst: `with open(path) as f:`, met daaronder `text = f.read()`.
        2. Zet alles in kleine letters: `text.lower()`, zodat hoofdletters als kleine letters tellen.
        3. Tel met een dictionary: loop over de tekst, houd alleen over wat je wilt tellen (met `and` als twee dingen allebei waar moeten zijn), en tel 1 op bij de telling ervan, of begin hem bij 1.
        4. Teken de tellingen: de keys op volgorde met `sorted`, hun tellingen met een list comprehension, en `plt.bar`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De klinkers in het gedicht uit [Een tekstbestand lezen](/course/open/py-files?lang=nl). Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_vowels():
    import matplotlib.pyplot as plt

    # 1. lees de tekst
    with open("lessons/py-files/poem.txt") as _f:
        _text = _f.read()
    # 2. zet alles in kleine letters
    _text = _text.lower()
    # 3. tel met een dictionary
    _vowels = {}
    for _ch in _text:
        if _ch in "aeiou":
            if _ch in _vowels:
                _vowels[_ch] = _vowels[_ch] + 1
            else:
                _vowels[_ch] = 1
    print(_vowels)
    # 4. teken: de keys op volgorde, hun tellingen in een comprehension
    _names = sorted(_vowels)
    _heights = [_vowels[_v] for _v in _names]
    plt.bar(_names, _heights)
    plt.title("Klinkers in het gedicht")
    plt.ylabel("hoe vaak")
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen de tellingen, als dictionary, en een staafdiagram met een staaf voor `a`, `e`, `i`, `o` en `u`. `sorted(_vowels)` geeft de keys van a tot u, en de comprehension pakt hun tellingen in diezelfde volgorde, dus krijgt elke staaf zijn eigen telling.
    """)
    return


@app.cell
def example_pairs():
    # stap 3 voor paren: elke slice van twee, geteld zoals een letter
    _word = "banana"
    _pairs = {}
    for _i in range(len(_word) - 1):
        _pair = _word[_i:_i + 2]
        if _pair in _pairs:
            _pairs[_pair] = _pairs[_pair] + 1
        else:
            _pairs[_pair] = 1
    print(_pairs)
    return


@app.cell(hide_code=True)
def text_after_pairs():
    mo.md(r"""
    Paren tel je op dezelfde manier, met een slice van twee als key. `"banana"` heeft 6 letters, en 5 paren: `ba`, `an`, `na`, `an` en `na`. Met de laatste letter begint geen eigen paar meer, dus stopt de loop één plek eerder: `range(len(_word) - 1)` geeft `_i` van 0 tot en met 4. De cel print `{'ba': 1, 'an': 2, 'na': 2}`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_letters():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tel de letters

        Maak `count_letters(text)` af: die geeft een dictionary terug met elke letter van `text` en hoe vaak hij voorkomt, alleen voor de letters a tot en met z, met hoofdletters geteld als kleine letters. Het vak begint met `LETTERS`, de 26 kleine letters, en een loop die over de tekst in kleine letters gaat, één teken `ch` per keer; `counts` is de dictionary die hij vult. Vervang de `...`. De check roept `count_letters("Hello, World")` aan: dat moet `{'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1}` geven. Klopt je functie, dan bewaart de check hem in je project, als `work/projects/llm/letters.py`.
        """),
        L.task("Laat `count_letters(text)` elke letter a tot en met z, klein of hoofdletter, tellen in een dictionary."),
    ])
    return


@app.cell(hide_code=True)
def exercise_letters():
    box_letters = L.code_box("count-letters", lines=12)
    box_letters
    return (box_letters,)


@app.cell(hide_code=True)
def check_letters(box_letters):
    result_letters = L.check("count-letters", box_letters.value)
    result_letters
    return (result_letters,)


@app.cell(hide_code=True)
def hints_letters():
    L.hints(
        "count-letters",
        [
            "Stap 3 van het recept, en de klinkers in het uitgewerkte voorbeeld: dezelfde loop, met `LETTERS` in plaats van `\"aeiou\"`.",
            "Tel `ch` alleen als hij in `LETTERS` staat. Staat hij al in `counts`, tel dan 1 op bij zijn telling; zo niet, begin hem bij 1.",
            "`if ch in LETTERS:`, met daaronder, nog een stap ingesprongen, `if ch in counts:` met ___ eronder; dan `else:` met `counts[ch] = 1`.",
        ],
        why="`text.lower()` maakt elke hoofdletter klein. `if ch in LETTERS:` slaat spaties en tekens over. Een letter die al in `counts` staat, krijgt er 1 bij; een nieuwe letter begint bij 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tel de paren

        Maak `count_pairs(text)` af: die geeft een dictionary terug met elk paar van twee letters naast elkaar en hoe vaak het voorkomt. Een paar met een spatie of een teken erin telt niet mee. De loop pakt al elke slice van twee, `pair`. Vervang de `...`. De check roept `count_pairs("Hello")` aan: dat moet `{'he': 1, 'el': 1, 'll': 1, 'lo': 1}` geven. Klopt je functie, dan bewaart de check hem als `work/projects/llm/pairs.py`.
        """),
        L.task("Laat `count_pairs(text)` elk paar van twee letters naast elkaar tellen in een dictionary."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("count-pairs", lines=13)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("count-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "count-pairs",
        [
            "Hetzelfde tellen als in oefening 1, maar nu voor `pair` in plaats van `ch`.",
            "Een paar telt alleen als beide tekens letters zijn: `pair[0]` en `pair[1]` moeten allebei in `LETTERS` staan.",
            "`if pair[0] in LETTERS and ___:`, met daaronder hetzelfde tellen als in oefening 1, met `pair` als key.",
        ],
        why="`text[i:i + 2]` pakt twee tekens naast elkaar. `if pair[0] in LETTERS and pair[1] in LETTERS:` slaat paren met een spatie of een teken over. Daarna wordt elk paar geteld zoals een letter in oefening 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_chart():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De letters van een boek

        Nu gaat je project aan het werk. Het vak leest het hele boek, telt de letters met je eigen `count_letters`, binnengehaald uit je project, en zet de letters op volgorde in `names`. Vervang de `...` door de regels die een lijst `heights` maken, één telling per letter in `names`, en teken daar een staafdiagram van, met een titel en een label bij de y-as. De check vergelijkt je grafiek met de eigen telling van de course, en laat je grafiek onder zijn melding zien.
        """),
        L.task("Teken een staaf per letter van het boek, zo hoog als zijn telling, met je eigen `count_letters`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chart():
    box_chart = L.code_box("letter-chart", lines=14)
    box_chart
    return (box_chart,)


@app.cell(hide_code=True)
def check_chart(box_chart, result_letters):
    result_letters  # check again when exercise 1 saves your module
    result_chart = L.check("letter-chart", box_chart.value)
    result_chart
    return (result_chart,)


@app.cell(hide_code=True)
def hints_chart():
    L.hints(
        "letter-chart",
        [
            "Stap 4 van het recept, zoals aan het eind van het uitgewerkte voorbeeld.",
            "Maak `heights` met een list comprehension over `names`: de telling van elke letter. Teken dan `names` en `heights` als staven, met een titel en een label bij de y-as.",
            "`heights = [___ for ch in names]`, dan `plt.bar(names, heights)`, `plt.title(\"...\")` en `plt.ylabel(\"...\")`.",
        ],
        why="`[counts[ch] for ch in names]` pakt de telling van elke letter, van a tot z. `plt.bar(names, heights)` tekent een staaf per letter. De hoogste staaf is `e`, dan `t` en `o`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je `count_pairs` weet welke letter op welke volgt. De cel hieronder vraagt hem, voor het hele boek, welke letters na `t` komen, en tekent de vijf die het vaakst voorkomen. Dat is al de kern van een language model: na `t` raad je `h`. De cel gebruikt je eigen module, dus werkt hij zodra oefening 2 af is.
    """)
    return


@app.cell(hide_code=True)
def demo_after_t(result_pairs):
    result_pairs  # run again when exercise 2 saves your module
    _ok = C.pairs_ok()
    if _ok is None:
        _out = mo.md("*Maak eerst oefening 2 af: dan staat je `count_pairs` in je project, en gebruikt deze cel hem.*")
    elif not _ok:
        _out = mo.md("*De `count_pairs` in `work/projects/llm/pairs.py` telt niet meer goed, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw.*")
    else:
        import matplotlib.pyplot as _plt
        from llm import pairs as _pairs
        with open("data/tiny-shakespeare.txt") as _f:
            _book = _f.read()
        _counts = _pairs.count_pairs(_book)
        _after_t = sorted(((n, p[1]) for p, n in _counts.items() if p[0] == "t"), reverse=True)[:5]
        _fig = _plt.figure(figsize=(6.4, 2.8))
        _plt.bar([b for _, b in _after_t], [n for n, _ in _after_t])
        _plt.title("Wat er na t komt bij Shakespeare")
        _plt.ylabel("hoe vaak")
        _out = mo.Html(plots.render(_fig))
        _plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Dictionaries](/course/open/py-dicts?lang=nl) telde je woorden; nu heb je een heel boek geteld. In [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl) bewaarde de check een functie in je project; nu heeft je project `llm` zijn eerste twee bestanden. In [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl) tekende je staven; nu laten ze zien wat Shakespeare schreef. In een latere les, *Een bigram-model door te tellen*, worden je paartellingen een model dat tekst schrijft, letter voor letter. Daar tel je elk teken, ook spaties en leestekens, zodat het model die ook kan schrijven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chart, result_letters, result_pairs):
    L.wrap_up(result_letters, result_pairs, result_chart)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
