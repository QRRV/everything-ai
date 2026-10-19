# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-sampling. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-sampling", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl) en [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="`P` is je tabel met kansen, en `chars[i]` is \"e\". Waar staan de kansen op het teken dat na \"e\" komt?",
        options={
            "column": "In kolom `i`: `P[:, i]`",
            "right": "In rij `i`: `P[i]`",
            "cell": "In het ene getal `P[i, i]`",
        },
        correct="right",
        why={
            "column": "Kolom `i` bevat hoe waarschijnlijk \"e\" is na elk teken. Zie het blok *Let op* in [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).",
            "cell": "`P[i, i]` is alleen de kans dat er na \"e\" nog een \"e\" komt. Zie [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl).",
            "right": "Rij `i` is de verdeling van het volgende teken na `chars[i]`: de getallen erin tellen op tot 1.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_while():
    quick_while = L.choice(
        "quick-while",
        question="Hoe vaak voegt deze loop een \"x\" toe?\n\n```python\ntext = \"ab\"\nwhile len(text) < 5:\n    text = text + \"x\"\n```",
        options={
            "five": "5 keer",
            "two": "2 keer",
            "right": "3 keer",
        },
        correct="right",
        why={
            "five": "De tekst heeft al 2 tekens, en de loop stopt zodra hij er 5 heeft. Zie [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl).",
            "two": "Na 2 keer is de tekst \"abxx\", 4 tekens, nog korter dan 5, dus de loop gaat nog een keer rond. Zie [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl).",
            "right": "\"ab\" wordt \"abx\", \"abxx\" en \"abxxx\". Dan is `len(text) < 5` `False`, en de loop stopt.",
        },
    )
    quick_while
    return (quick_while,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je tabel bevat voor elk teken de kansen op het volgende. Maar een tabel schrijft niet. Om te schrijven heeft je model een regel nodig: zoek de rij van het laatste teken op, kies daaruit een volgend teken, voeg het aan de tekst toe, en doe dan hetzelfde opnieuw. Welk teken je kiest, is de vraag van deze les: elke keer het waarschijnlijkste, of een willekeurig getrokken teken, waarbij waarschijnlijke tekens vaker getrokken worden? Elk language model dat schrijft, tot het grootste LLM, maakt die keuze bij elk token. Dit is nog steeds de tweede stap van je eigen LLM: aan het eind van deze les schrijft je bigram-model zijn eerste tekst.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           1, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        L.course_versions(),
        mo.md(r"""
        **Een voorbeeld uit het dagelijks leven.** Een verloting: wie drie van de vier loten heeft, wint in drie van de vier keer, maar niet altijd. Af en toe valt de prijs op dat ene andere lot; als de meeste loten altijd wonnen, gebeurde dat nooit.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_greedy():
    guess_greedy = L.guess(
        "greedy-loop",
        question="Je model begint met \"t\" en neemt steeds het waarschijnlijkste volgende teken: na \"t\" is dat \"h\". Wat denk je dat het schrijft, 30 tekens lang?",
        options={
            "english": "Tekst die steeds verandert, zoals Engels",
            "right": "Eén kort woord, steeds opnieuw",
            "random": "Letters in een willekeurige volgorde",
        },
        reveal=f"**Eén kort woord, steeds opnieuw:** `{C.greedy_text('t', 30)}`. Na \"t\" is \"h\" het waarschijnlijkste teken, na \"h\" is dat \"e\", na \"e\" een spatie, en na een spatie weer \"t\". Elke keer dat het model terug is bij \"t\", doet het precies hetzelfde: het draait in een kringetje rond.",
    )
    guess_greedy
    return (guess_greedy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Je model schrijft één teken tegelijk. Het zoekt de rij van het laatste teken op, kiest uit die rij het volgende teken, en voegt dat aan de tekst toe. Dan is dat nieuwe teken het laatste, en doet het hetzelfde opnieuw. Zo schrijven, waarbij je elk nieuw token kiest uit de voorspelling voor de tekst tot nu toe, heet **autoregressive generation**; hier is elk token één teken. Je bigram-model kijkt alleen naar het laatste teken; de modellen van latere lessen kijken naar veel meer van de tekst, maar ze schrijven op dezelfde manier.

        Hoe kies je? Steeds het waarschijnlijkste teken nemen is **greedy decoding**: `np.argmax(P[i])`, de plaats van het grootste getal in rij `i`. Dat is `P[i].argmax()` uit [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl), geschreven als functie. Greedy decoding schrijft elke keer dezelfde tekst, en zoals je bij de gok zag, kan het in een kringetje ronddraaien.

        **Sampling** trekt het volgende teken in plaats daarvan, met de kansen van de rij: een teken met kans 0,43 wordt iets minder dan de helft van de keren getrokken, en een teken met kans 0,01 ongeveer één keer op de honderd. De random number generator uit [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) doet dat met `rng.choice`: `rng.choice(len(chars), p=P[i])` trekt het nummer van een kolom: kolom `j` met kans `P[i, j]`. Met dezelfde seed trekt hij dezelfde getallen, dus schrijft je model dezelfde tekst opnieuw; met een andere seed schrijft het een andere tekst.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **autoregressive generation** | | één token tegelijk schrijven: elk nieuw token wordt gekozen uit de voorspelling voor de tekst tot nu toe, en eraan toegevoegd | `text = text + chars[j]` |
        | **greedy decoding** | | steeds het waarschijnlijkste volgende token nemen | `j = np.argmax(P[i])` |
        | **sampling** | | het volgende token trekken met de kansen die het model geeft: waarschijnlijke tokens komen vaak, onwaarschijnlijke af en toe | `j = rng.choice(len(chars), p=P[i])` |
        | `rng.choice(n, p=probs)` | | een van de getallen 0 tot en met n − 1, getrokken met de kansen in `probs`, die optellen tot 1 | `rng.choice(3, p=[0.2, 0.5, 0.3])` |
        """),
        L.watch_out("`rng.choice(len(chars), p=P[i])` geeft een getal, de plaats van een kolom, geen teken: het teken is `chars[j]`. En `p` moet optellen tot 1. Een rij van `P` doet dat; een kolom niet, en dan stopt NumPy met een fout."),
        mo.md(r"""
        De figuur: vijf stappen van sampling met de tabel van de toneelstukken, beginnend met "T", met seed 23.
        """),
        mo.Html(C.steps_picture("T", 23, 5, {"title": "Vijf stappen van sampling", "text": "de tekst tot nu toe",
                                             "row": "de 4 waarschijnlijkste in de rij van het laatste teken", "drawn": "getrokken",
                                             "alt": "Vijf rijen, één per stap. De tekst tot nu toe is T, Th, The, The met een spatie, en The a, elk met het laatste teken vet. Ernaast staan de vier waarschijnlijkste tekens in de rij van dat laatste teken, met hun kansen, en het getrokken teken heeft een rand: h na T, e na h, een spatie na e, a na de spatie terwijl t waarschijnlijker was, en r na a terwijl n waarschijnlijker was."})),
        mo.md(r"""
        Elke stap kijkt alleen naar het laatste teken, vetgedrukt. De eerste drie keer werd het waarschijnlijkste teken getrokken, en werd de tekst "The ". Daarna trok het model "a" na de spatie, niet "t", het waarschijnlijkste, en "r" na "a", niet "n". Greedy decoding was verdergegaan met "The the"; sampling schreef "The ar", en gaat van daaruit verder.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om je model `n` tekens te laten schrijven, beginnend met `start`:

    1. Maak één random number generator, met een seed: `rng = np.random.default_rng(seed)`.
    2. Noteer de rij van elk teken: `row_of[chars[k]] = k`, voor elke `k`.
    3. Begin de tekst: `text = start`.
    4. Zolang `len(text) < n`: zoek de rij van het laatste teken, `i = row_of[text[-1]]`; trek het volgende, `j = rng.choice(len(chars), p=P[i])`; en voeg het toe, `text = text + chars[j]`.
    5. Voor greedy decoding neem je het waarschijnlijkste teken in plaats van er een te trekken: `j = np.argmax(P[i])`. Dan heb je geen random number generator nodig.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tabel van "abab ba" uit het uitgewerkte voorbeeld van [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl), met zijn drie tekens " ", "a" en "b". De cel schrijft er 12 tekens mee, beginnend met "a": eerst door te samplen, dan met greedy decoding. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_write():
    import numpy as np

    _P = np.array([[0, 0, 1], [0, 0, 1], [1/3, 2/3, 0]])   # de tabel van "abab ba"
    _chars = [" ", "a", "b"]
    # 1. één random number generator, met een seed
    _rng = np.random.default_rng(4)
    # 2. de rij van elk teken
    _row_of = {}
    for _k in range(len(_chars)):
        _row_of[_chars[_k]] = _k
    # 3. het begin
    _text = "a"
    # 4. trek het volgende teken uit de rij van het laatste, tot er 12 zijn
    while len(_text) < 12:
        _i = _row_of[_text[-1]]
        _j = _rng.choice(len(_chars), p=_P[_i])
        _text = _text + _chars[_j]
    print(_text)
    # 5. greedy decoding: steeds het waarschijnlijkste teken
    _greedy = "a"
    while len(_greedy) < 12:
        _i = _row_of[_greedy[-1]]
        _j = np.argmax(_P[_i])
        _greedy = _greedy + _chars[_j]
    print(_greedy)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In "abab ba" kwam na "a" en na een spatie altijd "b", dus beide teksten schrijven daar "b". Na "b" trok sampling drie keer "a" en twee keer een spatie. Greedy decoding neemt na "b" altijd "a", de waarschijnlijkste, dus schrijft het `abababababab`. Voer de cel opnieuw uit: met seed 4 blijft ook de tekst van sampling hetzelfde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_draw():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De regel die trekt
    """)
    return


@app.cell(hide_code=True)
def choice_draw_cell():
    choice_draw = L.choice(
        "draw-line",
        question="`i` is de rij van het laatste teken. Welke regel trekt het volgende teken met de kansen van je model?",
        options={
            "argmax": "`j = np.argmax(P[i])`",
            "even": "`j = rng.integers(0, len(chars))`",
            "right": "`j = rng.choice(len(chars), p=P[i])`",
            "column": "`j = rng.choice(len(chars), p=P[:, i])`",
        },
        correct="right",
        why={
            "argmax": "Die regel neemt altijd het waarschijnlijkste teken: greedy decoding. Hij trekt niet.",
            "even": "Die regel trekt wel, maar geeft elk teken dezelfde kans, wat je model ook zegt.",
            "column": "Kolom `i` bevat hoe waarschijnlijk `chars[i]` is na elk teken: niet wat er na `chars[i]` komt, en de getallen erin tellen niet op tot 1. Zie het blok *Let op* in *Het idee*.",
            "right": "Hij trekt het nummer van een kolom: kolom `j` met kans `P[i, j]`. Waarschijnlijke tekens komen vaak, onwaarschijnlijke af en toe.",
        },
    )
    choice_draw
    return (choice_draw,)


@app.cell(hide_code=True)
def text_exercise_greedy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Greedy decoding

        `greedy(P, chars, start, n)` schrijft `n` tekens met greedy decoding. De loop staat er al. Vervang de `...` door stap 5 van het recept. De check roept `greedy(P, ["a", "b", "c"], "a", 6)` aan, met `P = np.array([[0.1, 0.6, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]])`, en dat moet `"abcabc"` geven; hij probeert ook andere tabellen en begintekens. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/greedy.py`.
        """),
        L.task("Laat `greedy` als `j` de plaats van het grootste getal in rij `i` nemen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_greedy():
    box_greedy = L.code_box("greedy", lines=13)
    box_greedy
    return (box_greedy,)


@app.cell(hide_code=True)
def check_greedy(box_greedy):
    result_greedy = L.check("greedy", box_greedy.value)
    result_greedy
    return (result_greedy,)


@app.cell(hide_code=True)
def hints_greedy():
    L.hints(
        "greedy",
        [
            "Stap 5 van het recept, zoals in het uitgewerkte voorbeeld.",
            "`np.argmax` geeft de plaats van het grootste getal in een array. De rij van het laatste teken is `P[i]`.",
            "`j = np.argmax(___)`.",
        ],
        why="In rij \"a\" van `P` is het grootste getal 0,6, in kolom 1: \"b\". In rij \"b\" is dat 0,6, in kolom 2: \"c\", en in rij \"c\" 0,8, in kolom 0: \"a\". Zo gaat de tekst steeds rond: `abcabc`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_generate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Laat je model schrijven

        `generate(P, chars, start, n, seed)` schrijft `n` tekens door te samplen. De random number generator, de rijen en het begin staan er al. Vervang de `...` door stap 4 van het recept: de `while`-loop. De check roept `generate(P, ["a", "b", "c"], "a", 8, 1)` aan, met de `P` van oefening 2, en dat moet `"abcacabc"` geven; hij probeert ook andere tabellen, begintekens, lengtes en seeds. Is je functie goed, dan bewaart de check hem in je project, als `work/projects/llm/sampling.py`.
        """),
        L.task("Laat `generate` een teken trekken uit de rij van het laatste teken en het toevoegen, tot de tekst `n` tekens lang is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_generate():
    box_generate = L.code_box("generate", lines=14)
    box_generate
    return (box_generate,)


@app.cell(hide_code=True)
def check_generate(box_generate):
    result_generate = L.check("generate", box_generate.value)
    result_generate
    return (result_generate,)


@app.cell(hide_code=True)
def hints_generate():
    L.hints(
        "generate",
        [
            "Stap 4 van het recept, zoals in het uitgewerkte voorbeeld: drie regels in een `while`-loop.",
            "Elke ronde: de rij van het laatste teken, `text[-1]`; dan het nummer van een kolom, getrokken met `rng.choice` en `p=`; dan het teken met dat nummer, toegevoegd aan `text`.",
            "`while len(text) < n:`, en daarin `i = row_of[___]`, `j = rng.choice(len(chars), p=___)` en `text = text + chars[___]`.",
        ],
        why="Met seed 1 trekt `rng.choice` elke keer dezelfde getallen, dus weet de check wat je functie moet geven. Vanaf \"a\" trok hij \"b\", dan \"c\" en \"a\", en dan \"c\": na \"a\" heeft \"c\" kans 0,3, en greedy decoding had \"b\" genomen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder maakt de tabel van de toneelstukken met de `count_bigrams`, `bigram_matrix` en `to_probs` in je project, of die van de course waar de jouwe ontbreekt of stuk is (de notitie onder *Waar je project staat* zegt welke), en laat je model schrijven, beginnend met "T": 60 tekens met je `greedy`, en 200 tekens met je `generate`, één keer met seed 1 en één keer met seed 2. Hij werkt zodra oefening 2 en 3 af zijn.
    """)
    return


@app.cell(hide_code=True)
def demo_write(result_generate, result_greedy):
    result_greedy, result_generate  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `greedy` en je `generate` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*De tabel maken of schrijven met je functies duurt langer dan 30 seconden. Gaat elke functie één keer over zijn tekst of tekens, en stopt elke loop zodra de tekst lang genoeg is?*",
            "table": "*De tabel van de toneelstukken komt niet goed uit met de `count_bigrams`, `bigram_matrix` en `to_probs` in `work/projects/llm/`: niet elke rij telt op tot 1. Heb je een van die bestanden veranderd? Verwijder het, en laat de oefening ervan opnieuw nakijken, in [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) of [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl): de check bewaart je functie dan opnieuw.*",
            "greedy": "*De `greedy` in `work/projects/llm/greedy.py` geeft niet meer wat de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
            "sampling": "*De `generate` in `work/projects/llm/sampling.py` geeft niet meer wat de check verwacht, of stopt met een fout. Heb je dat bestand veranderd? Verwijder het, en laat oefening 3 opnieuw nakijken: die check bewaart je functie dan opnieuw.*",
        }[_result])
    else:
        _out = mo.vstack([
            mo.md("Greedy decoding, met je `greedy`:"),
            mo.Html(C.text_block(_result['greedy'])),
            mo.md("Sampling, met je `generate` en seed 1:"),
            mo.Html(C.text_block(_result[1])),
            mo.md("En met seed 2:"),
            mo.Html(C.text_block(_result[2])),
            mo.md("""Greedy decoding blijft na vier tekens hangen. De teksten van sampling zijn geen Engels, maar van een afstand lijken ze erop: korte woorden tussen spaties, regels die met een hoofdletter beginnen, en af en toe een naam met een dubbele punt, zoals in de toneelstukken. Veel meer kan een model dat alleen naar het laatste teken kijkt, niet. Met dezelfde seed schrijft je model elke keer dezelfde tekst; met een andere seed een andere."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tabel uit [De bigram-tabel als matrix van kansen](/course/open/lm-bigram-matrix?lang=nl) geeft de kansen, de random number generator uit [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) trekt eruit, en de `while`-loop uit [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl) gaat door tot de tekst lang genoeg is. Je project `llm` heeft nu ook `greedy.py` en `sampling.py`. In de volgende les, *Temperature: gedurfder of veiliger tekst*, maak je de keuze tussen veilig en gedurfd zelf: scherpere kansen, dichter bij greedy decoding, of vlakkere, dichter bij willekeurig trekken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_draw, result_generate, result_greedy):
    L.wrap_up(choice_draw, result_greedy, result_generate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
