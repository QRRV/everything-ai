# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-embeddings-idea. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-embeddings-idea", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl) en [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_onehot():
    quick_onehot = L.choice(
        "quick-onehot",
        question="De klassen kat, hond en vogel worden geschreven als de one-hot-vectoren $(1, 0, 0)$, $(0, 1, 0)$ en $(0, 0, 1)$. Welke twee klassen liggen het dichtst bij elkaar?",
        options={
            "pets": "Kat en hond: dat zijn de twee huisdieren",
            "right": "Geen: ze liggen alle drie even ver uit elkaar",
            "order": "Kat en vogel: hun enen staan het verst uit elkaar, dus hun vectoren lijken het meest op elkaar",
        },
        correct="right",
        why={
            "pets": "De vectoren weten niets van huisdieren: kat en hond verschillen op twee plekken, net als kat en vogel. Zie [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).",
            "order": "Elk paar vectoren verschilt op precies twee plekken; waar de 1 staat, maakt niet uit. Zie [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl).",
            "right": "Elk paar verschilt op twee plekken, dus elke afstand is $\\sqrt{2}$, en elke cosine similarity is 0. One-hot-vectoren zeggen welke klasse iets is, en niets over welke klassen op elkaar lijken. Vandaag leer je vectoren kennen die dat wel doen.",
        },
    )
    quick_onehot
    return (quick_onehot,)


@app.cell(hide_code=True)
def quick_check_angle():
    quick_angle = L.choice(
        "quick-angle",
        question="Wat is de cosine similarity van $(1, 0)$ en $(0, 2)$?",
        options={
            "right": "0: de twee pijlen staan haaks op elkaar",
            "half": "0,5: de ene is twee keer zo lang als de andere",
            "two": "2: het inproduct van de twee",
        },
        correct="right",
        why={
            "half": "De lengtes vallen weg uit een cosine similarity: die kijkt alleen naar de richtingen. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "two": "Het inproduct is $1 \\cdot 0 + 0 \\cdot 2 = 0$, en de cosine similarity deelt het door de twee normen. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "right": "Het inproduct is 0, dus de cosine similarity is 0, wat de lengtes ook zijn: de ene pijl wijst naar rechts, de andere omhoog. Vandaag meet je precies zo hoe sterk twee plaatjes op elkaar lijken.",
        },
    )
    quick_angle
    return (quick_angle,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om dingen te vergelijken, heeft een computer er getallen voor nodig. One-hot-vectoren zetten elke klasse even ver van elke andere. Pixels helpen maar een klein beetje: twee plaatjes van hetzelfde cijfer, door twee mensen geschreven, hebben slechts een deel van hun pixels gemeen. Je wilt getallen waarin dingen die op elkaar lijken dicht bij elkaar liggen, en dingen die verschillen ver uit elkaar.

    **Een voorbeeld uit het dagelijks leven.** Op een kaart zeggen twee getallen waar een plaats ligt: hoe ver naar het oosten, en hoe ver naar het noorden. Plaatsen die op de kaart dicht bij elkaar liggen, liggen in het echt dicht bij elkaar, en op de kaart zie je in één oogopslag welke plaatsen naast elkaar liggen. Een model kan zo'n kaart leren, maar dan van betekenis.
    """)
    return


@app.cell(hide_code=True)
def predict_ones():
    guess_ones = L.guess(
        "where-ones",
        question="Het cijfermodel van de course maakt van elk plaatje 2 getallen, een punt op een kaart, vlak voordat het beslist welk cijfer het is. Waar denk je dat de punten van de 126 testplaatjes van een 1 terechtkomen?",
        options={
            "spread": "Over de hele kaart: elke 1 is anders geschreven",
            "right": "Langs één lijn vanuit het midden: sommige dichtbij, sommige ver weg, allemaal dezelfde kant op",
            "point": "Allemaal op één punt: voor het model zijn ze allemaal hetzelfde cijfer",
        },
        reveal="**Langs één lijn vanuit het midden.** Negen op de tien enen liggen rechtsboven, in een smalle wig, sommige dicht bij het midden en sommige ver ervandaan. Elk cijfer krijgt zijn eigen richting. Hoe ver een plaatje van het midden ligt, zegt hoe zeker het model is, niet welk cijfer het is.",
    )
    guess_ones
    return (guess_ones,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **embedding** is een lijst getallen die een model aan iets geeft, zoals een woord, een plaatje of een geluid, zodat dingen die op elkaar lijken getallen krijgen die op elkaar lijken. Het cijfermodel van de course maakt er een voor elk plaatje. Het is een klein netwerk zoals het MLP uit [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl), met aan het begin andere layers die een latere les uitlegt, en de course heeft het getraind op de 6.000 trainingscijfers. Vlak voordat het beslist welk cijfer het ziet, maakt het van het plaatje 2 getallen: zijn embedding. Echte modellen gebruiken honderden of duizenden getallen; dit model gebruikt er 2, zodat je ze kunt tekenen. Elke embedding is dan een punt, en samen vormen ze een **embedding map**.

        Op de kaart krijgt elk cijfer zijn eigen richting vanuit het midden. Hoe sterk twee embeddings op elkaar lijken, meet je met de hoek ertussen: hun cosine similarity. Twee plaatjes van hetzelfde cijfer hebben gemiddeld een cosine similarity van 0,95, twee plaatjes van verschillende cijfers −0,11. Op hun 784 pixels krijgen dezelfde paren 0,53 en 0,37, een veel kleiner verschil.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **embedding** | | een lijst getallen die een model aan iets geeft, zodat dingen die op elkaar lijken getallen krijgen die op elkaar lijken | `E[0]`, de 2 getallen van plaatje 0 |
        | **embedding map** | | de embeddings getekend als punten, zodat je ziet welke dingen dicht bij elkaar liggen | `plt.scatter(E[:, 0], E[:, 1])` |
        """),
        L.watch_out("Bij gewoon computergebruik betekent een video embedden dat je hem in een webpagina zet. In AI is een embedding een lijst getallen. En de getallen betekenen op zichzelf niets: wat telt, is hoe de embeddings zich tot elkaar verhouden. Opnieuw getraind met een andere seed tekent het model een andere kaart, met de cijfers in andere richtingen; met maar 2 getallen delen twee cijfers soms zelfs één straal. De course koos een training waarvan de kaart tien duidelijke stralen laat zien."),
        mo.md(r"""
        De figuur: de embeddings van de eerste 300 testcijfers, elk getekend als zijn eigen cijfer, op zijn 2 getallen. Uit het midden komen tien stralen, één voor elk cijfer.
        """),
        mo.Html(C.map_picture({"title": "De embeddings van 300 testcijfers", "x": "getal 1", "y": "getal 2",
                               "alt": "Cijfers getekend op hun 2 getallen. Ze vormen tien stralen vanuit het midden, één per cijfer: enen rechtsboven, vieren naar rechts, zevens naar links, nullen rechtsonder, enzovoort; vieren en negens liggen naast elkaar."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Het plaatje vinden dat het meest op een gegeven plaatje lijkt, met embeddings:

    1. Haal de embedding van elk plaatje op: hier laat `C.embeddings()` het cijfermodel van de course los op de 1.000 testcijfers, en geeft het `E` terug, één rij van 2 getallen per plaatje, en hun `labels`.
    2. Vergelijk twee embeddings met hun cosine similarity: dicht bij 1 betekent dat ze op elkaar lijken, 0 of minder dat ze dat niet doen.
    3. Vergelijk de embedding van het gegeven plaatje met die van elk ander plaatje, en kies het plaatje met de grootste cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Testplaatje 0, een 7, vergeleken met de volgende vier, en daarna met alle andere. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_embeddings():
    import numpy as np

    # 1. de embeddings van de 1.000 testcijfers, van het model van de course
    _E, _labels = C.embeddings()
    print(_E.shape, _labels[:5])
    print(round(_E[0, 0], 1), round(_E[0, 1], 1), _labels[0])
    # 2. de cosine similarity van plaatje 0 met plaatje 1 tot en met 4
    for _j in range(1, 5):
        _cos = np.dot(_E[0], _E[_j]) / (np.linalg.norm(_E[0]) * np.linalg.norm(_E[_j]))
        print(_j, _labels[_j], round(_cos, 3))
    # 3. het plaatje dat het meest op plaatje 0 lijkt: de grootste cosine similarity van alle andere
    _best = -2
    _best_j = -1
    for _j in range(1, len(_E)):
        _cos = np.dot(_E[0], _E[_j]) / (np.linalg.norm(_E[0]) * np.linalg.norm(_E[_j]))
        if _cos > _best:
            _best = _cos
            _best_j = _j
    print(_best_j, _labels[_best_j], round(_best, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `_E` heeft 1.000 rijen van 2 getallen, en de eerste vijf labels zijn 7, 2, 1, 0 en 4. Plaatje 0, de 7, heeft de embedding $(-168{,}9;\ 23{,}0)$: ver naar links, waar de zevens op de kaart liggen. Met de 2 is de cosine similarity 0,319; met de 1 −0,384; met de 0 −0,662; en met de 4 −0,999, de tegenovergestelde richting. Het plaatje dat er het meest op lijkt, is plaatje 26, nog een 7, met een cosine similarity van 1,0, afgerond op vier decimalen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Dicht bij het midden, ver ervandaan
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "same-direction",
        question="Op de kaart ligt één 4 ver van het midden, een andere er dichtbij, in dezelfde richting. Wat is de cosine similarity van hun embeddings?",
        options={
            "half": "Ongeveer 0,5: de ene is veel langer dan de andere",
            "zero": "Ongeveer 0: ze liggen ver uit elkaar",
            "right": "Ongeveer 1: ze wijzen dezelfde kant op",
        },
        correct="right",
        why={
            "half": "De lengtes vallen weg uit de cosine similarity: die deelt door beide normen.",
            "zero": "Ver uit elkaar, dat is de afstand. De cosine similarity kijkt alleen naar de richtingen, en die zijn hetzelfde.",
            "right": "Dezelfde richting geeft een cosine similarity dicht bij 1, wat de lengtes ook zijn. Daarom past die bij deze kaart: de richting zegt welk cijfer, de lengte alleen hoe zeker het model is.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_similar():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De rij die er het meest op lijkt

        Laat `most_similar(E, i)` het nummer `j` geven van de rij van `E` die het meest op rij `i` lijkt, volgens stap 3 van het recept: de grootste cosine similarity, voor een `j` die niet `i` is. `E` is een NumPy-array met één embedding per rij. De loop staat er al: vul in wat er voor elke `j` gebeurt. De check roept `most_similar(np.array([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), 0)` aan, wat `1` moet geven: rij 1 wijst bijna dezelfde kant op als rij 0, al is hij tien keer zo lang. Hij probeert ook andere tabellen, ook met 3 getallen per rij.
        """),
        L.task("Laat `most_similar` de rij geven die het meest op rij `i` lijkt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_similar():
    box_similar = L.code_box("most-similar", lines=13)
    box_similar
    return (box_similar,)


@app.cell(hide_code=True)
def check_similar(box_similar):
    result_similar = L.check("most-similar", box_similar.value)
    result_similar
    return (result_similar,)


@app.cell(hide_code=True)
def hints_similar():
    L.hints(
        "most-similar",
        [
            "In de loop hetzelfde als in stap 3 van het uitgewerkte voorbeeld, met `E` en `i` in plaats van `_E` en `0`.",
            "Eerst `if j != i:`, zodat rij `i` niet met zichzelf wordt vergeleken; daarbinnen de cosine similarity van `E[i]` en `E[j]`.",
            "`cos = np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))`, en dan, `if cos > ___:`, zet je `best = cos` en `best_j = ___`.",
        ],
        why="In de loop: `if j != i:`, dan `cos = np.dot(E[i], E[j]) / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))`, en `if cos > best:` met `best = cos` en `best_j = j`. In het voorbeeld heeft rij 1 een cosine similarity van 0,995 met rij 0, rij 2 0,707, en rij 3 −1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_map():
    mo.md(r"""
    ### Oefening 3. De kaart lezen
    """)
    return


@app.cell(hide_code=True)
def choice_map_cell():
    choice_map = L.choice(
        "most-like-4",
        question="Kijk naar de kaart. De embeddings van welk cijfer lijken het meest op die van de vieren?",
        options={
            "right": "De negens",
            "zero": "De nullen",
            "seven": "De zevens",
        },
        correct="right",
        why={
            "zero": "De nullen liggen aan de andere kant van de vieren, verder weg: een 4 en een 0 hebben gemiddeld een cosine similarity van 0,66.",
            "seven": "De zevens wijzen bijna de andere kant op: een 4 en een 7 hebben gemiddeld een cosine similarity van −0,93.",
            "right": "De negens liggen vlak naast de vieren: een 4 en een 9 hebben gemiddeld een cosine similarity van 0,82, twee vieren 0,95. En ze lijken op elkaar: allebei hebben ze een dichte of bijna dichte bovenkant en een streep omlaag aan de rechterkant. Niemand heeft het model verteld dat vieren en negens op elkaar lijken; dat leerde het uit de plaatjes.",
        },
    )
    choice_map
    return (choice_map,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De vectoren uit [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl) zetten elke klasse even ver van elke andere; een embedding wordt geleerd, zodat de richtingen iets betekenen. Om embeddings te vergelijken gebruik je de cosine similarity uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), en de kaart is een scatter plot zoals in [Getallen tekenen: lijngrafiek, scatter plot en staafdiagram](/course/open/py-plotting?lang=nl). In de volgende les, *Convolutie: één klein filter schuift over de afbeelding*, zie je hoe netwerken zoals het cijfermodel van de course naar een plaatje kijken. Later krijgen woorden ook embeddings: zo weet een language model welke woorden op elkaar lijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, result_similar, choice_map):
    L.wrap_up(choice_direction, result_similar, choice_map)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
