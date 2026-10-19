# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-kinds-of-learning. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-kinds-of-learning", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="Een tabel met huizen heeft de kolommen oppervlakte, kamers, tuin en prijs. Je wilt de prijs voorspellen. Wat is het label?",
        options={
            "area": "De oppervlakte",
            "all": "Alle vier de kolommen",
            "right": "De prijs",
        },
        correct="right",
        why={
            "area": "De oppervlakte weet je van tevoren. Wat wil je voorspellen? Zie *Het idee* in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).",
            "all": "Sommige kolommen krijgt het model, één kolom voorspelt het. Zie *Het idee* in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).",
            "right": "Het label is de kolom die je wilt voorspellen: het goede antwoord. Oppervlakte, kamers en tuin zijn de features.",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_inference():
    quick_inference = L.choice(
        "quick-inference",
        question="Een getraind model voorspelt de prijs van een huis dat te koop staat. Hoe heet dat?",
        options={
            "right": "Inference",
            "training": "Training",
            "label": "Labelen",
        },
        correct="right",
        why={
            "training": "Training is leren van voorbeelden waarvan het label bekend is. Is de prijs van dit huis bekend? Zie de tabel in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).",
            "label": "Het model schrijft geen goed antwoord op; het voorspelt er een. Zie de tabel in [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl).",
            "right": "Een getraind model gebruiken op een nieuw voorbeeld, waarvan het label nog niet bekend is, heet inference.",
        },
    )
    quick_inference
    return (quick_inference,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe had elk voorbeeld om van te leren een label: een goed antwoord dat iemand had opgeschreven, zoals de prijs waarvoor een huis verkocht is. Maar een large language model leerde het meeste van wat het weet van miljarden pagina's tekst, en niemand heeft die gelabeld. Een robot die leert lopen, heeft ook geen tabel met goede antwoorden. De soorten leren verschillen in één ding: waar de goede antwoorden vandaan komen.

    **Een voorbeeld uit het dagelijks leven.** Een taal kun je leren van een leraar die elke zin verbetert, door boeken te lezen en te merken welke woorden na welke komen, of door in het buitenland eten te bestellen en te zien of je krijgt wat je wilde.
    """)
    return


@app.cell(hide_code=True)
def predict_labels():
    guess_labels = L.guess(
        "labels",
        question="Een large language model leerde van miljarden pagina's tekst. Waar kwamen zijn labels volgens jou vandaan?",
        options={
            "people": "Mensen schreven ze, een voor een",
            "text": "Uit de tekst zelf: het volgende woord is het label",
            "none": "Het heeft helemaal geen labels nodig",
        },
        reveal="**Uit de tekst zelf.** Neem een willekeurig stuk tekst, verberg het woord dat erna komt, en dat woord is het label. Elke pagina geeft honderden voorbeelden, en niemand hoeft ook maar één label te schrijven. Dit is self-supervised learning.",
    )
    guess_labels
    return (guess_labels,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Er zijn vier belangrijke soorten leren, en het verschil zit in waar de goede antwoorden vandaan komen.

        - Bij **supervised learning** heeft elk voorbeeld een label dat iemand eraan gaf, en het model leert het label te voorspellen uit de features: huizen met hun prijs.
        - Bij **unsupervised learning** zijn er helemaal geen labels. Het model zoekt structuur in de voorbeelden, zoals groepen klanten die hetzelfde kopen.
        - Bij **self-supervised learning** komen de labels uit de data zelf: verberg een deel, zoals het volgende woord, en voorspel het. Language models leren zo.
        - Bij **reinforcement learning** zijn er geen voorbeelden met antwoorden, maar acties en rewards (beloningen): het model probeert iets, krijgt een reward of niet, en leert welke acties lonen, zoals in een spel.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **supervised learning** | leren van voorbeelden die een label hebben |
        | **unsupervised learning** | leren van voorbeelden zonder labels, om er structuur in te vinden |
        | **self-supervised learning** | leren van data zonder labels door de labels uit de data zelf te maken, zoals het volgende woord |
        | **reinforcement learning** | leren door acties te proberen en er rewards voor te krijgen |
        """),
        L.watch_out("Self-supervised learning is niet unsupervised learning: er zijn labels, en het model leert ze te voorspellen. Alleen schreef niemand ze: de data geeft ze. Sommige bronnen noemen het toch unsupervised, omdat niemand met de hand iets labelt; deze course houdt de twee uit elkaar."),
    ])
    return


@app.cell(hide_code=True)
def picture_kinds():
    mo.vstack([
        draw.data_tables([
            ("Supervised", ["oppervlakte", "prijs"], [["62", "245"], ["85", "310"], ["110", "395"]], {(0, 1), (1, 1), (2, 1)}, ""),
            ("Unsupervised", ["brood", "melk"], [["2", "0"], ["0", "3"], ["2", "1"]], set(), ""),
            ("Self-supervised", ["input", "label"], [["de", "kat"], ["kat", "zat"], ["zat", "stil"]], {(0, 1), (1, 1), (2, 1)}, ""),
            ("Reinforcement", ["actie", "reward"], [["links", "0"], ["op", "0"], ["op", "1"]], {(0, 1), (1, 1), (2, 1)}, ""),
        ], title="Waar de antwoorden vandaan komen",
            alt="Vier kleine tabellen, één voor elke soort leren. Supervised: huizen met hun oppervlakte en hun prijs; de kolom prijs is gemarkeerd. Unsupervised: klanten met hoeveel brood en melk ze kochten; niets is gemarkeerd. Self-supervised: paren uit de tekst de kat zat stil, een woord als input en het volgende woord als label; de kolom label is gemarkeerd. Reinforcement: acties, links, op en op, elk met een reward van 0, 0 en 1; de kolom reward is gemarkeerd."),
        mo.md(r"""
        Vier kleine tabellen met voorbeelden. Een dikke rand markeert de goede antwoorden; bij reinforcement learning nemen de rewards hun plaats in. Supervised: de prijzen zijn opgeschreven toen de huizen verkocht werden. Unsupervised: alleen wat elke klant kocht, en geen kolom met antwoorden. Self-supervised: het label van elke rij is de input van de volgende rij, dus de labels komen uit de tekst *de kat zat stil* zelf. Reinforcement: geen goede antwoorden, alleen een reward na elke actie.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien welke soort leren een taak is, vraag je waar de goede antwoorden vandaan komen:

    1. Gaven mensen elk voorbeeld zijn label? Supervised learning.
    2. Geeft de data zelf het label, zoals het volgende woord? Self-supervised learning.
    3. Zijn er helemaal geen antwoorden, alleen voorbeelden om structuur in te vinden? Unsupervised learning.
    4. Doet het model iets, en krijgt het rewards voor wat het doet? Reinforcement learning.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier taken door het recept. Stel de vragen op volgorde: het eerste *ja* zegt de soort.

    **Een foto-app** leert katten en honden uit elkaar houden, van 50.000 foto's die mensen het label "kat" of "hond" gaven. Stap 1, gaven mensen elk voorbeeld zijn label? Ja: elke foto heeft het label dat mensen eraan gaven. **Supervised learning.**

    **Een toetsenbord op je telefoon** leert je volgende woord voorstellen, van berichten die mensen typten. Stap 1: nee, mensen typten de berichten, maar niemand zette er een label bij. Stap 2, geeft de data zelf het label? Ja: na elk woord in een bericht staat het volgende woord er al. **Self-supervised learning.**

    **Een muziekapp** zet nummers die op elkaar lijken in groepen, en niemand zegt welke groepen er zijn. Stap 1 en 2: geen labels van mensen, en niets verborgen om te voorspellen. Stap 3, helemaal geen antwoorden, alleen voorbeelden om structuur in te vinden? Ja: alleen nummers, om in groepen te verdelen. **Unsupervised learning.**

    **Een programma leert een auto parkeren** in een simulator: het stuurt, en krijgt punten als de auto in het parkeervak eindigt. Stap 1 tot en met 3: geen voorbeelden met antwoorden, en geen groepen om te vinden. Stap 4, doet het model iets, en krijgt het rewards voor wat het doet? Ja: het stuurt, en die punten zijn de rewards. **Reinforcement learning.**

    **Stap 2 in code.** Hoe geeft de data de labels? Trainingsparen gemaakt van de letters van een woord, elke letter met de letter erna als label. Het commentaar met `#` zegt bij welke stap van het recept de code hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_pairs():
    _text = "hello"
    # 2. de data geeft de labels: elke letter, met de volgende letter als label
    _pairs = []
    for _i in range(len(_text) - 1):
        _pairs.append((_text[_i], _text[_i + 1]))
    print(_pairs)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vijf letters geven vier paren: de laatste letter heeft geen letter erna. Elk paar is een voorbeeld, de eerste letter de input en de tweede het label, en niemand schreef een label. Het eerste language model dat je in deze course bouwt, leert van precies zulke paren; een large language model kijkt naar alle woorden ervoor, niet alleen naar het laatste.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spam():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een spamfilter
    """)
    return


@app.cell(hide_code=True)
def choice_spam_cell():
    choice_spam = L.choice(
        "spam-kind",
        question="Een spamfilter leert van 10.000 mails die mensen als spam of geen spam markeerden. Welke soort leren is dat?",
        options={
            "right": "Supervised learning",
            "unsupervised": "Unsupervised learning",
            "self": "Self-supervised learning",
            "reinforcement": "Reinforcement learning",
        },
        correct="right",
        why={
            "unsupervised": "Hier zijn wel antwoorden: mensen markeerden elke mail. Zie stap 1 van het recept.",
            "self": "De markeringen komen niet uit de mails zelf: mensen voegden ze toe. Zie stap 1 en 2 van het recept.",
            "reinforcement": "Het filter doet niets en krijgt geen rewards; het leert van voorbeelden met antwoorden. Zie het recept.",
            "right": "Elke mail heeft een label dat mensen eraan gaven: spam of geen spam.",
        },
    )
    choice_spam
    return (choice_spam,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vijf taken, vier soorten

        Elke taak hieronder krijgt één soort leren. Gebruik het recept, zoals in het uitgewerkte voorbeeld: de vragen op volgorde.
        """),
        L.task("Zet elke taak bij zijn soort leren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Neem één taak tegelijk door het recept, zoals in het uitgewerkte voorbeeld: het eerste *ja* zegt de soort.",
            "Stap 1 bij elke taak: wie gaf de goede antwoorden? Dan stap 2: verbergt een taak een deel van zijn eigen data en voorspelt hij dat?",
            "De fraudecheck en het vertalen hebben antwoorden van mensen. De fotopuzzel voorspelt een afgedekt vierkantje van zijn eigen foto's. Welke taak zoekt groepen, en welke krijgt punten?",
        ],
        why="**Supervised:** de fraudecheck en het vertalen; mensen markeerden elke betaling en schreven elke vertaling. **Self-supervised:** de fotopuzzel; het afgedekte vierkantje is het label, en dat komt uit de foto zelf. **Unsupervised:** de nieuwsonderwerpen; er is geen lijst met onderwerpen, alleen artikelen om in groepen te verdelen. **Reinforcement:** schaken; het programma speelt, en winnen of verliezen geeft zijn reward.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Labels uit tekst

        Maak `word_pairs(words)` af. Hij krijgt een lijst woorden en geeft een lijst trainingsparen terug: elk woord waar nog een woord na komt, met dat volgende woord als label. Vervang de `...`. De check roept `word_pairs(["the", "cat", "sat", "down"])` aan: dat moet `[("the", "cat"), ("cat", "sat"), ("sat", "down")]` geven. Hij probeert ook andere lijsten, ook een met één woord en een lege lijst.
        """),
        L.task("Laat `word_pairs` elk woord teruggeven met het volgende woord als label."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("word-pairs", lines=7)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("word-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "word-pairs",
        [
            "Kijk naar het uitgewerkte voorbeeld: hetzelfde, met woorden in plaats van letters.",
            "Een loop over de plekken `i` van de woorden. Welke plekken hebben een woord erna, en welk woord is dat?",
            "`for i in range(len(words) - ___):` en daarin `pairs.append((words[i], words[___]))`.",
        ],
        why="`range(len(words) - 1)` gaat langs elk woord met nog een woord erna, en `(words[i], words[i + 1])` koppelt het aan dat volgende woord. Vier woorden geven drie paren.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Voorbeelden, features en labels](/course/open/ml-vocabulary?lang=nl) had elk voorbeeld om van te leren een label; nu weet je dat het label van mensen kan komen, uit de data zelf, of helemaal niet bestaat, en dat een model ook kan leren van rewards. Het language model dat je later in de course bouwt, leert self-supervised, van paren zoals die in oefening 3. Reinforcement learning krijgt na de hoofdroute een eigen tak in de course. In de volgende les, *Classificatie: een klasse voorspellen*, kijk je naar supervised learning waarbij het label een klasse is in plaats van een getal.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spam, result_kinds, result_pairs):
    L.wrap_up(choice_spam, result_kinds, result_pairs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
