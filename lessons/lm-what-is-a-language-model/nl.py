# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-what-is-a-language-model. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow function

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-what-is-a-language-model", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl) en [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_given():
    quick_given = L.choice(
        "quick-given",
        question="In een lange tekst komt de letter q 20 keer voor, en 19 van die keren volgt er een u. Wat is P(u volgt | q)?",
        options={
            "count": "19",
            "right": "0,95",
            "rest": "0,05",
        },
        correct="right",
        why={
            "count": "19 is hoe vaak de u volgt. Een kans is een aandeel, tussen 0 en 1: tel alleen onder de 20 gevallen met een q. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "rest": "Dat is de kans dat er geen u volgt. Zie [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl).",
            "right": "Van de 20 keer dat er een q staat, volgt er 19 keer een u: 19 / 20 = 0,95.",
        },
    )
    quick_given
    return (quick_given,)


@app.cell(hide_code=True)
def quick_check_function():
    quick_function = L.choice(
        "quick-function",
        question="Een model is een functie. Wat doet die functie?",
        options={
            "right": "Die neemt de features en geeft een voorspelling terug",
            "store": "Die bewaart elk trainingsvoorbeeld, om het later op te zoeken",
            "labels": "Die schrijft bij elk voorbeeld het goede antwoord",
        },
        correct="right",
        why={
            "store": "Een model bewaart zijn parameters, niet de voorbeelden: het rekent een voorspelling uit de features uit. Zie [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "labels": "De labels horen bij de voorbeelden; de output van het model is een voorspelling. Zie [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl).",
            "right": "Features erin, een voorspelling eruit, zoals ŷ = w · x + b, met zijn parameters w en b.",
        },
    )
    quick_function
    return (quick_function,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een chatprogramma schrijft zijn antwoord stukje voor stukje, en elke keer raadt het model erachter wat er volgt. Om goed te worden in raden, moet een model leren van veel meer tekst dan mensen ooit met de hand kunnen labelen: miljarden woorden. Waar komen zijn labels dan vandaan?

    **Een voorbeeld uit het dagelijks leven.** Als iemand zegt "En ze leefden nog lang en …", weet je het volgende woord al voordat je het hoort. Niemand heeft je dat als regel geleerd: je hebt het opgepikt door veel verhalen te horen.
    """)
    return


@app.cell(hide_code=True)
def predict_examples():
    guess_examples = L.guess(
        "examples",
        question="Een model leert het volgende teken van een tekst te voorspellen. Hoeveel trainingsvoorbeelden denk je dat de tekst \"the cat sat\", 11 tekens lang, geeft?",
        options={
            "all": "11: één voor elk teken",
            "right": "10: één voor elk teken na het eerste",
            "one": "1: de hele tekst is één voorbeeld",
        },
        reveal="**10.** Elk teken na het eerste is het label van één voorbeeld, en de tekst ervoor is de input: \"t\" → \"h\", \"th\" → \"e\", \"the\" → een spatie, enzovoort tot \"the cat sa\" → \"t\". Niet 11: voor het eerste teken staat niets, dus dat is nooit een label. De tekst geeft zijn eigen labels, dus een model kan van elke tekst leren, zonder dat iemand hem labelt: self-supervised learning, zoals in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl).",
    )
    guess_examples
    return (guess_examples,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **token** is een stukje tekst dat een language model leest en schrijft. In deze les is een token één teken; echte modellen gebruiken grotere stukken, zoals de volgende les laat zien. Schrijf de tokens van een tekst als $x_1, x_2, \dots$: $x_1$ is het eerste token, $x_t$ het $t$-de. Hier begint het tellen bij 1, zoals gebruikelijk in de wiskunde; Python telt de plekken vanaf 0, dus $x_t$ is `text[t - 1]`.

        Een **language model** geeft, voor de tekst tot nu toe, de kans op elk mogelijk volgend token:
        """),
        L.formula(r"[[pred:P]](x_{t+1} \mid x_1, \dots, x_t)"),
        mo.md(r"""
        Dat is een voorwaardelijke kans, zoals in [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl): de kans op het volgende token, gegeven de tekst tot nu toe. Samen tellen de kansen van alle mogelijke volgende tokens op tot 1. En een language model is een functie, zoals in [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl): de tekst tot nu toe gaat erin, een kans voor elk token komt eruit.

        **Next-token prediction** is hoe een language model leert: op echte tekst voorspelt het elk volgend token, en die voorspelling wordt vergeleken met het token dat echt volgt. De tekst geeft de labels zelf: een tekst van $n$ tokens geeft $n - 1$ voorbeelden. In code is de tekst tot nu toe een slice: een slice werkt op een tekst net als op een lijst, zoals in [Lijsten](/course/open/py-lists?lang=nl), dus `text[:i]` is de tekst voor plek `i`, en `"my lord"[:2]` is `"my"`. Om te schrijven kiest een model een volgend token uit zijn kansen, plakt het achter de tekst, en doet dat opnieuw.

        Het language model van de course, `C.next_probs(text)`, telt in Tiny Shakespeare, zo'n 40.000 regels uit toneelstukken van Shakespeare, welk teken er volgt na de laatste drie tekens van `text`; is de tekst korter, of komen die drie nergens voor, dan kijkt het naar minder tekens. Het kent 65 verschillende tekens. Tellen is genoeg voor een language model: daar is geen neural network voor nodig. Echte language models kijken naar duizenden tokens tegelijk en leren hun kansen met een neural network, maar ze geven hetzelfde: een kans voor elk mogelijk volgend token.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **token** | | een stukje tekst dat een language model leest en schrijft; hier één teken | `text[i]` |
        | **language model** | | een model dat voor de tekst tot nu toe de kans op elk mogelijk volgend token geeft | `C.next_probs(text)` |
        | **next-token prediction** | | leren door elk volgend token van een echte tekst te voorspellen: de tekst geeft de labels | |
        | $x_t$ | x t | het $t$-de token van een tekst, geteld vanaf 1 | `text[t - 1]` |
        | $P(x_{t+1} \mid x_1, \dots, x_t)$ | de kans op x t plus 1 gegeven x 1 tot en met x t | de kans op het volgende token, gegeven de tekst tot nu toe | `C.next_probs(text[:t])` |
        """),
        L.watch_out("*Token* betekent ook iets anders: een login-token is een sleutel die toegang geeft tot een dienst, en een cryptotoken is een soort digitale munt. In deze course is een token altijd een stukje tekst dat een language model leest; voor de sleutel zegt de course *API key*."),
        mo.md(r"""
        De figuur: de kansen die het model van de course geeft aan het volgende teken na "my lo". Alleen de vijf waarschijnlijkste van de 65 staan erin.
        """),
        draw.next_word_bars("my lo", C.top("my lo"), picked=0, picked_label="waarschijnlijkst",
                            title="Na \"my lo\""),
        mo.md(r"""
        Het model ziet alleen de laatste drie tekens: een spatie, "l" en "o", het begin van een woord. "lord" en "love" komen allebei veel voor in de toneelstukken, dus "r" en "v" zijn bijna even waarschijnlijk. Een model dat alleen "r" voorspelt, zit er vaak naast; de kansen zeggen hoe zeker het is. Een model dat ook "my" zag, zou het beter doen: in de toneelstukken staat "my lord" zo'n tien keer zo vaak als "my love".
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een language model te laten leren en te gebruiken:

    1. Hak de tekst in tokens: hier is elk teken een token.
    2. Maak de voorbeelden: voor elke plek `i` vanaf 1 is de input de tekst tot nu toe, `text[:i]`, en het label het volgende token, `text[i]`.
    3. Geef het model de tekst tot nu toe: het geeft een kans voor elk mogelijk volgend token.
    4. Zoek de kans op van het token dat echt volgt: een goed model geeft dat een hoge kans.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De voorbeelden van de tekst "my lord", en het model van de course op plek 5, na "my lo". Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_model():
    _text = "my lord"
    # 1. de tokens: elk teken, dus 7 tokens
    print(len(_text))
    # 2. elke plek vanaf 1 geeft een voorbeeld: de tekst tot nu toe, en het token dat volgt
    _examples = []
    for _i in range(1, len(_text)):
        _examples.append((_text[:_i], _text[_i]))
    print(len(_examples), _examples)
    # 3. het model op plek 5: de tekst tot nu toe, "my lo", erin; een kans voor elk mogelijk volgend token eruit
    _probs = C.next_probs(_text[:5])
    print(len(_probs))
    # 4. de kans op het token dat echt volgt, _text[5], dat is "r"; en op "v"
    print(_probs[_text[5]], _probs["v"])
    return


@app.cell(hide_code=True)
def text_after_example():
    _probs = C.next_probs("my lo")
    mo.md(f"""
    Zeven tokens geven zes voorbeelden, van ("m", "y") tot ("my lor", "d"). Na "my lo" geeft het model alle 65 tekens een kans; de "r" die echt volgt, krijgt {L.number(_probs["r"], 2)}, en "v" krijgt {L.number(_probs["v"], 2)}. Bij het trainen zou een neural network zijn parameters zo veranderen dat het token dat echt volgt een hogere kans krijgt, voorbeeld na voorbeeld.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_addup():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Alle kansen samen
    """)
    return


@app.cell(hide_code=True)
def choice_addup_cell():
    choice_addup = L.choice(
        "add-up",
        question="Na \"my lo\" geeft het model van de course elk van de 65 tekens die het kent een kans. Hoeveel zijn die 65 kansen samen?",
        options={
            "right": "1: precies één ervan volgt",
            "many": "65: één voor elk teken",
            "depends": "Dat hangt af van de tekst tot nu toe",
        },
        correct="right",
        why={
            "many": "Elke kans is een aandeel tussen 0 en 1, en samen verdelen ze alle gevallen. Zie *Het idee*.",
            "depends": "De kansen zelf hangen van de tekst af, maar hun som niet. Zie *Het idee*.",
            "right": "Er volgt precies één teken, dus de kansen verdelen alle gevallen: een kansverdeling over de 65 tekens.",
        },
    )
    choice_addup
    return (choice_addup,)


@app.cell(hide_code=True)
def text_exercise_examples():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De voorbeelden in een tekst

        `examples(text)` geeft de trainingsvoorbeelden van een tekst voor next-token prediction, elk als paar: de tekst tot nu toe, en het token dat volgt. Vervang de drie `...` door stap 2 van het recept. De check roept `examples("my lord")` aan, en dat moet de zes voorbeelden uit het uitgewerkte voorbeeld geven. Hij probeert ook andere teksten, ook korte.
        """),
        L.task("Laat `examples` voor elke plek vanaf 1 de tekst tot nu toe en het token dat volgt geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_examples():
    box_examples = L.code_box("examples", lines=7)
    box_examples
    return (box_examples,)


@app.cell(hide_code=True)
def check_examples(box_examples):
    result_examples = L.check("examples", box_examples.value)
    result_examples
    return (result_examples,)


@app.cell(hide_code=True)
def hints_examples():
    L.hints(
        "examples",
        [
            "Stap 2 van het recept: welke plekken `i`, en wat zijn de input en het label op elke plek?",
            "De plekken lopen van 1 tot en met de laatste plek van de tekst, `len(text) - 1`. De input stopt net voor plek `i`; het label is het token op plek `i`.",
            "`range(1, len(___))`, en het paar `(text[:___], text[___])`.",
        ],
        why="Voor \"my lord\" gaat `i` van 1 tot en met 6. Bij `i` = 1 is het paar (\"m\", \"y\"), bij `i` = 6 is het (\"my lor\", \"d\"): zes voorbeelden, één voor elk token na het eerste.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_real():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Hoe zeker is het model?

        Zoek voor elke plek in de tekst "to be" de kans die het model van de course geeft aan het token dat echt volgt. In het vak heet het model `next_probs`, net als `C.next_probs` in het uitgewerkte voorbeeld. Vervang de twee `...` door stap 3 en 4 van het recept. De laatste regel laat de lijst `real` zien, met één kans voor elk van de vier voorbeelden. De check probeert je code ook op andere teksten.
        """),
        L.task("Vul `real` met de kans die het model geeft aan elk token dat echt volgt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_real():
    box_real = L.code_box("real-probs", lines=7)
    box_real
    return (box_real,)


@app.cell(hide_code=True)
def check_real(box_real):
    result_real = L.check("real-probs", box_real.value)
    result_real
    return (result_real,)


@app.cell(hide_code=True)
def hints_real():
    L.hints(
        "real-probs",
        [
            "Stap 3 en 4 van het recept, op elke plek `i`: het model krijgt de tekst tot nu toe, en jij zoekt één kans op in wat het teruggeeft.",
            "`next_probs` krijgt de tekst tot nu toe, zoals in stap 2 van het recept. Het geeft een dictionary terug met een kans voor elk teken: zoek het token dat echt volgt op als key, zoals `_probs[\"r\"]` in het uitgewerkte voorbeeld.",
            "`probs = next_probs(text[:___])` en `real.append(probs[text[___]])`.",
        ],
        why="Bij `i` = 1 krijgt het model \"t\" en is het token dat echt volgt \"o\"; bij `i` = 4 krijgt het \"to b\" en is het token dat echt volgt \"e\", waaraan het een kans van ongeveer 0,66 geeft.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een language model is een functie, zoals in [Een model is een functie met parameters](/course/open/ml-model-is-a-function?lang=nl), die een voorwaardelijke kans geeft, zoals in [Voorwaardelijke kans](/course/open/prob-conditional?lang=nl), voor elk mogelijk volgend token. Het leert alleen van tekst, omdat de tekst zijn eigen labels geeft: self-supervised learning, zoals in [Soorten leren](/course/open/ml-kinds-of-learning?lang=nl). Hier was een token een teken, zoals in [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl). In de volgende les, *Tokens: hoe echte LLM's tekst in stukken hakken*, zie je de stukken die echte language models gebruiken; daarna bouw je in *Een bigram-model door te tellen* zelf een model dat telt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_addup, result_examples, result_real):
    L.wrap_up(choice_addup, result_examples, result_real)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
