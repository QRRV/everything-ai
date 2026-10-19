# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-model-families. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-model-families", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl) en [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_attention():
    quick_attention = L.choice(
        "quick-attention",
        question="Welke woorden neemt een woord met attention op om zijn nieuwe vector te maken?",
        options={
            "right": "Alle woorden, ook zichzelf, elk met zijn attention weight",
            "before": "Alleen het woord vlak ervoor",
            "same": "Alleen de woorden die letter voor letter gelijk zijn aan dat woord",
        },
        correct="right",
        why={
            "before": "Dat lijkt meer op een recurrent neural network, dat dingen woord voor woord doorgeeft. Zie [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl).",
            "same": "Elk woord krijgt een weight, ook een woord dat maar een beetje past. Zie [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl).",
            "right": "Elk woord kijkt naar alle woorden, ook naar zichzelf. Vandaag zie je de families van transformers, en de ene regel die ze onderscheidt: naar welke tokens een token mag kijken.",
        },
    )
    quick_attention
    return (quick_attention,)


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="Bij next-token prediction krijgt het model de tekst tot nu toe, om daarmee het volgende token te voorspellen. Zit dat volgende token in zijn input?",
        options={
            "yes": "Ja: het model heeft het token nodig om van te leren",
            "end": "Ja, maar alleen aan het eind van de tekst",
            "right": "Nee: de input stopt er net voor",
        },
        correct="right",
        why={
            "yes": "Het token is het label, waarmee de voorspelling van het model wordt vergeleken; in de input zou het het antwoord verklappen. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "end": "Op elke plek is de input alleen de tekst vóór het label. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "De input is de tekst tot nu toe, zonder het token dat voorspeld moet worden: anders kon het model het antwoord gewoon afkijken. Een model met attention, dat naar alle tokens kijkt, heeft daar een regel voor nodig: vandaag zie je hem.",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    BERT, GPT en T5 zijn drie bekende modellen uit 2018 en 2019, en alle drie zijn het transformers: layer na layer attention, elk gevolgd door een klein MLP. Toch doen ze ander werk. BERT leest een tekst en zegt er iets over, GPT schrijft tekst, en T5 maakt van de ene tekst een andere. Kom je een nieuw model tegen, dan vertelt zijn familie wat het kan, en waar het niet voor gemaakt is. Het verschil komt uit één regel in de attention: naar welke tokens een token mag kijken.

    **Een voorbeeld uit het dagelijks leven.** Als je een brief leest om te weten waar hij over gaat, lees je hem helemaal, en elk woord helpt je de andere te begrijpen. Als je een brief schrijft, kun je alleen terugkijken naar wat je al geschreven hebt: de volgende woorden zijn er nog niet. Een tolk doet allebei: eerst luisteren naar de hele zin, en dan de vertaling woord voor woord uitspreken, met een blik terug op wat er gezegd is en op wat de tolk zelf al gezegd heeft.
    """)
    return


@app.cell(hide_code=True)
def predict_look():
    guess_look = L.guess(
        "third-token",
        question="GPT is getraind om elk volgend token van een tekst te voorspellen: op de plek van het derde token voorspelt GPT het vierde. Naar welke tokens mag het derde token met attention kijken?",
        options={
            "before": "De eerste twee: alleen de tokens ervoor",
            "right": "De eerste drie: de twee ervoor, en zichzelf",
            "all": "Alle tokens van de tekst: ook die na het derde",
        },
        reveal="**De eerste drie.** Op zijn plek is het derde token het laatste token van de tekst tot nu toe, en zoals elk token bij attention mag het naar zichzelf en de tokens ervoor kijken. Naar het vierde mag het niet kijken: dat is het token dat het moet voorspellen, en met een blik vooruit kon het dat gewoon afkijken en niets leren. Als GPT schrijft, zijn de tokens na het derde er nog niet eens. In BERT bestaat zo'n regel niet: elk token kijkt naar alle tokens.",
    )
    guess_look
    return (guess_look,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **encoder** is een transformer waarin elk token naar alle tokens kijkt, ervoor en erna. Hij leest een hele tekst, en geeft elk token een vector die de hele context kent. De bekendste is BERT, van Google uit 2018: die leerde door woorden te raden die in een tekst verborgen waren, uit de woorden aan beide kanten. Encoders zijn gemaakt om te lezen: om te zeggen of een recensie positief is, om de namen in een tekst te vinden, of om de teksten te vinden die bij een zoekvraag passen. Ze zijn niet gemaakt om tekst te schrijven.

        Een **decoder** is een transformer waarin elk token alleen naar zichzelf en de tokens ervoor kijkt. Daardoor is hij een language model zoals in [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl): hij voorspelt het volgende token, en schrijft door er één token tegelijk aan toe te voegen. Dat is de GPT-lijn: GPT-1 uit 2018, met 117 miljoen parameters, GPT-2 uit 2019 met 1,5 miljard, en GPT-3 uit 2020 met 175 miljard; steeds hetzelfde soort decoder, maar groter, en getraind op meer tekst. Vanaf GPT-3 was een taak in woorden beschrijven, vaak met een paar voorbeelden in de tekst, genoeg: geen extra training per taak. De meeste grote language models van nu zijn decoders.

        Een **encoder-decoder** heeft ze allebei: een encoder leest de inputtekst, en een decoder schrijft de outputtekst, één token tegelijk. Elk token van de decoder kijkt naar alle tokens van de input, en in de output naar zichzelf en de tokens ervoor. De eerste transformer, uit 2017, was er een, gemaakt om te vertalen; T5, van Google uit 2019, doet elke taak zo, van een tekst naar een tekst.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **encoder** | | een transformer waarin elk token naar alle tokens kijkt; hij leest een tekst | |
        | **decoder** | | een transformer waarin elk token alleen naar zichzelf en de tokens ervoor kijkt; hij schrijft tekst, één token tegelijk | |
        | **encoder-decoder** | | een encoder die de input leest, en een decoder die de output schrijft en ook naar de hele input kijkt | |
        """),
        L.watch_out("\"Decoder\" en \"encoder\" betekenen hier iets anders dan in [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl), waar encoderen tekst in bytes omzette, en decoderen bytes weer in tekst. Hier zijn het twee soorten transformer. En GPT is de naam van de modellen van OpenAI, maar mensen gebruiken het ook voor andere decoders die zo gebouwd zijn."),
        mo.md(r"""
        De figuur: wie naar wie kijkt, in een encoder en in een decoder, voor een tekst van vijf tokens. Rij voor rij een token; een gevuld vakje is een token waar het naar kijkt.
        """),
        mo.Html(C.look_picture({"title": "Wie kijkt naar wie", "encoder": "Encoder, zoals BERT", "decoder": "Decoder, zoals GPT",
                                "tokens": ["de", "kat", "ging", "zitten", "."], "caption": "Rij: het token dat kijkt. Kolom: een token waar het naar kijkt.",
                                "alt": "Twee roosters van vijf bij vijf vakjes, voor de tokens de, kat, ging, zitten en punt. In de encoder zijn alle 25 vakjes gevuld: elk token kijkt naar alle tokens. In de decoder zijn alleen de vakjes op en onder de diagonaal gevuld: het eerste token kijkt naar zichzelf, het laatste naar alle vijf."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Naar welke tokens token $i$ kijkt, met de plekken geteld vanaf 0:

    1. In een encoder: alle tokens, ervoor en erna.
    2. In een decoder: zichzelf en de tokens ervoor, de plekken 0 tot en met $i$.
    3. In een encoder-decoder: een token van de decoder kijkt naar alle tokens van de input, en naar de output zoals in stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Voor elk token van "the cat sat down .", de tokens waar het naar kijkt, in een encoder en in een decoder; en bij een vertaling het tweede token dat de decoder geschreven heeft. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_families():
    _tokens = ["the", "cat", "sat", "down", "."]
    # 1. een encoder: elk token kijkt naar alle tokens
    print("encoder")
    for _i in range(len(_tokens)):
        print(_tokens[_i], "->", _tokens)
    # 2. een decoder: elk token kijkt naar zichzelf en de tokens ervoor
    print("decoder")
    for _i in range(len(_tokens)):
        print(_tokens[_i], "->", _tokens[:_i + 1])
    # 3. een encoder-decoder die vertaalt: het tweede outputtoken kijkt naar de hele input, en naar de outputplekken 0 en 1
    _input = ["the", "cat"]
    _output = ["de", "kat"]
    print(_output[1], "->", _input, _output[:2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de encoder krijgt elk token dezelfde lijst: alle vijf de tokens. In de decoder groeit de lijst: "the" kijkt alleen naar zichzelf, "cat" naar "the" en zichzelf, en de punt naar alle vijf. `_tokens[:_i + 1]` stopt net na plek `_i`, dus het token zelf zit erin. Bij de vertaling kijkt "kat", het tweede token dat de decoder schreef, naar de hele Engelse input, en in de Nederlandse output naar "de" en zichzelf, om het volgende Nederlandse token te voorspellen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_family():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De goede familie
    """)
    return


@app.cell(hide_code=True)
def choice_family_cell():
    choice_family = L.choice(
        "family",
        question="Een programma leest een hele productrecensie en zegt of die positief of negatief is. Welke familie is precies hiervoor gemaakt?",
        options={
            "right": "Een encoder: elk woord kijkt naar de hele recensie",
            "decoder": "Een decoder: dat is de nieuwste familie, dus de beste in alles",
            "encdec": "Een encoder-decoder: die maakt van de recensie een andere tekst",
        },
        correct="right",
        why={
            "decoder": "Een decoder kan het wel, door \"positief\" of \"negatief\" te schrijven, maar hij is gemaakt om te schrijven, en elk token ziet alleen de woorden ervoor. Nieuwer is niet hetzelfde als gemaakt voor deze taak.",
            "encdec": "Het antwoord is een label, geen nieuwe tekst: er valt niets te schrijven. Een decoder erbovenop is werk zonder nut.",
            "right": "Om een recensie te beoordelen heb je hem helemaal nodig, en in een encoder kijkt elk token naar alle tokens. Classificatie van tekst is precies waar BERT en zijn familie voor gemaakt zijn.",
        },
    )
    choice_family
    return (choice_family,)


@app.cell(hide_code=True)
def text_exercise_can_see():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Mag token i naar token j kijken?

        Laat `can_see(kind, i, j)` `True` geven als in een model van soort `kind` het token op plek `i` naar het token op plek `j` mag kijken, en anders `False`. `kind` is `"encoder"` of `"decoder"`. Vervang de twee `...`, volgens stap 1 en 2 van het recept: de eerste is één waarde, de tweede een vergelijking van `i` en `j`. De check roept `can_see("decoder", 1, 3)` aan, wat `False` moet geven, en probeert beide soorten met andere plekken.
        """),
        L.task("Laat `can_see` zeggen of een token naar een ander mag kijken, in een encoder en in een decoder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_can_see():
    box_can_see = L.code_box("can-see", lines=6)
    box_can_see
    return (box_can_see,)


@app.cell(hide_code=True)
def check_can_see(box_can_see):
    result_can_see = L.check("can-see", box_can_see.value)
    result_can_see
    return (result_can_see,)


@app.cell(hide_code=True)
def hints_can_see():
    L.hints(
        "can-see",
        [
            "In een encoder hangt het antwoord niet af van `i` en `j`. In een decoder wel: ligt plek `j` op of vóór plek `i`?",
            "Een vergelijking zoals `a < b` geeft zelf `True` of `False`. In een decoder mag `j` kleiner zijn dan `i`, of eraan gelijk.",
            "`return True` voor de encoder, en `return j ___ i` voor de decoder.",
        ],
        why="In `can_see(\"decoder\", 1, 3)` vraagt token 1 naar token 3, dat erna komt: een decoder mag niet vooruitkijken, dus `False`. `can_see(\"decoder\", 2, 2)` is `True`: een token mag altijd naar zichzelf kijken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_translate():
    mo.md(r"""
    ### Oefening 3. Een vertaler schrijft
    """)
    return


@app.cell(hide_code=True)
def choice_translate_cell():
    choice_translate = L.choice(
        "translate",
        question="Een encoder-decoder vertaalt \"the cat sat down .\" naar het Nederlands. Zijn decoder heeft drie Nederlandse tokens geschreven, en op de plek van het derde voorspelt hij het vierde. Naar welke tokens mag het derde Nederlandse token kijken?",
        options={
            "input": "Alleen naar de vijf Engelse tokens: de vertaling komt uit de input",
            "all": "Naar alle Engelse en alle Nederlandse tokens, ook die erna",
            "right": "Naar alle vijf de Engelse en de eerste drie Nederlandse tokens",
        },
        correct="right",
        why={
            "input": "De decoder kijkt ook terug naar wat hij al geschreven heeft: het volgende Nederlandse woord hangt ook af van de woorden ervoor.",
            "all": "De Nederlandse tokens na het derde zijn nog niet geschreven: de decoder mag in zijn eigen tekst alleen terugkijken.",
            "right": "De decoder kijkt naar de hele input, zoals stap 3 van het recept zegt, en in zijn eigen tekst, zoals elke decoder, alleen naar zichzelf en wat hij ervoor schreef.",
        },
    )
    choice_translate
    return (choice_translate,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Met de attention uit [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl) kijkt elk woord naar alle woorden; een decoder beperkt dat tot zichzelf en de woorden ervoor, zodat hij next-token prediction kan leren zoals in [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) zonder het antwoord te zien. In het gevorderde deel bouw je die regel zelf in attention, en daarna een eigen decoder: een kleine GPT. In de volgende les, *Kijk in een echt LLM: de kansen op het volgende token*, laad je een echte decoder, GPT-2, en zie je zijn kansen voor het volgende token.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_family, result_can_see, choice_translate):
    L.wrap_up(choice_family, result_can_see, choice_translate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
