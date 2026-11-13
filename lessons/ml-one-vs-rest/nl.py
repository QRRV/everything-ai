# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-one-vs-rest. Route: course/route/lessen/klassiek-ml.yaml. Checks en beelden: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-one-vs-rest", "nl")
    C = L.checks   # de pinguïns uit ml-knn, de straten van de course voor one-vs-rest en one-vs-one, en de beelden


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De breedste straat: de maximum-margin classifier](/course/open/ml-max-margin?lang=nl) en [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="Een straat met een soft margin tussen de Chinstraps ($y = -1$) en de Gentoos ($y = +1$) geeft een nieuwe pinguïn $w \\cdot x + b = -0{,}4$. Wat zegt hij?",
        options={
            "right": "Chinstrap: de pinguïn ligt op de straat, maar aan de kant van de Chinstraps",
            "none": "Niets: een pinguïn op de straat krijgt geen klasse",
            "gentoo": "Gentoo: hij ligt dichter bij de lijn dan bij de rand van de Chinstraps",
        },
        correct="right",
        why={
            "none": "De straat is er voor de training: hij zegt hoeveel ruimte de lijn heeft. Bij het classificeren telt alleen de kant van de lijn, en $-0{,}4$ ligt onder 0. Zie *Het idee* in [De breedste straat: de maximum-margin classifier](/course/open/ml-max-margin?lang=nl).",
            "gentoo": "Dichter bij de lijn betekent minder zeker, niet de andere klasse: onder 0 is de kant van de Chinstraps, hoe dichtbij ook. Zie *Het idee* in [De breedste straat: de maximum-margin classifier](/course/open/ml-max-margin?lang=nl).",
            "right": "Onder 0 is de kant van de Chinstraps. De score zegt ook hoe zeker het model is: $-0{,}4$ is niet erg zeker. Vandaag vergelijk je zulke scores tussen modellen.",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Een model met softmax geeft drie klassen de logits $-1{,}5$, $-0{,}2$ en $-0{,}9$. Welke klasse voorspelt hij?",
        options={
            "none": "Geen: elke logit ligt onder 0",
            "right": "De tweede: de hoogste logit krijgt de hoogste kans",
            "first": "De eerste: de logit die het verst van 0 ligt",
        },
        correct="right",
        why={
            "none": "Softmax maakt van alle logits kansen die samen 1 zijn, ook van negatieve; de hoogste logit krijgt altijd de hoogste kans. Zie *Het idee* in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "first": "Het verst van 0 is hier de laagste, en de laagste logit krijgt de laagste kans. Zie *Het idee* in [Veel klassen: softmax](/course/open/ml-softmax?lang=nl).",
            "right": "$-0{,}2$ is de hoogste, dus krijgt de tweede klasse de hoogste kans, al ligt elke logit onder 0. Vandaag wint de hoogste score op dezelfde manier.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De breedste straat scheidt twee klassen, de Chinstraps van de Gentoos. Maar er zijn drie soorten pinguïns, handgeschreven cijfers zijn er tien, en de maximum-margin classifier antwoordt alleen "deze of die". Veel modellen doen dat. Een model met softmax geeft elke klasse een eigen logit, allemaal in één model, maar niet elk model kun je zo ombouwen. Wat altijd kan: een paar modellen voor twee klassen trainen en hun antwoorden combineren. Daar zijn twee gangbare manieren voor, en ze zijn het niet altijd eens.

    **Een voorbeeld uit het dagelijks leven.** In een halve competitie speelt elk paar teams één keer tegen elkaar, en het team met de meeste overwinningen wint de beker. In geen enkele wedstrijd staan meer dan twee teams.
    """)
    return


@app.cell(hide_code=True)
def predict_models():
    guess_models = L.guess(
        "models",
        question="Eén manier traint een model voor elk paar klassen. Hoeveel modellen zijn dat voor de tien cijfers 0 tot en met 9?",
        options={
            "ten": "10: één voor elk cijfer",
            "twenty": "20: twee voor elk cijfer",
            "right": "45",
        },
        reveal="**45**: 0 tegen 1 tot en met 9 zijn 9 paren, 1 tegen 2 tot en met 9 zijn er 8 meer, en zo verder: 9 + 8 + … + 1 = 45, of 10 · 9 / 2. Eén model per klasse zou 10 zijn. Voor 100 klassen is het 4.950 tegen 100; maar een model voor een paar traint alleen op de voorbeelden van zijn twee klassen.",
    )
    guess_models
    return (guess_models,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **One-vs-rest.** Train één model voor elke klasse, op alle voorbeelden: die klasse $+1$, elke andere klasse $-1$. Voor de pinguïns: Adelie of niet, Chinstrap of niet, Gentoo of niet. Elk model geeft een score $w_c \cdot x + b_c$, en boven 0 zegt het ja. Op de tekens alleen kan een pinguïn helemaal geen ja krijgen, of meer dan één. Daarom kijkt **one-vs-rest** niet naar de tekens: het neemt de klasse waarvan het model de hoogste score geeft, het model dat het zekerst is, ook als elke score onder 0 ligt. Zet de $w_c$ naast elkaar als de kolommen van een matrix $W$ en de $b_c$ in een vector $b$, en $x\,W + b$ geeft alle scores tegelijk:
        """),
        L.formula(r"\hat{y} = \text{de klasse } c \text{ met de hoogste } (x\,[[param:W]] + [[param:b]])_c"),
        mo.md(r"""
        Met $k$ klassen zijn dat $k$ modellen, elk getraind op alle voorbeelden.

        **One-vs-one.** Train één model voor elk paar klassen, alleen op de voorbeelden van die twee: Adelie tegen Chinstrap, Adelie tegen Gentoo, Chinstrap tegen Gentoo. Elk model geeft één stem, aan een van zijn twee klassen: de klasse die het kiest. Bij **one-vs-one** wint de klasse met de meeste stemmen. Met $k$ klassen zijn dat $k(k-1)/2$ modellen: 3 voor drie soorten, 45 voor tien cijfers; maar elk ziet maar een deel van de data.

        **Output code.** Een derde manier trekt one-vs-rest verder door. Geef elke klasse een rij van $+1$ en $-1$, haar code, en train één model voor elke kolom: het leert $+1$ voor de klassen met $+1$ in zijn kolom. Een nieuw voorbeeld krijgt een rij antwoorden van de modellen, en krijgt de klasse waarvan de code er op de minste plaatsen van verschilt. One-vs-rest is de code met één $+1$ in elke rij, afgelezen aan de scores in plaats van aan de tekens; one-vs-one past er ook in als een code ook $0$ mag bevatten, voor een klasse die een model nooit ziet. Met meer kolommen dan nodig kunnen de andere modellen een fout van één model overstemmen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **one-vs-rest** | een model voor elke klasse, die klasse tegen alle andere; de klasse met de hoogste score wint | `ovr_predict(names, W, b, x)` |
        | **one-vs-one** | een model voor elk paar klassen, alleen getraind op die twee; de klasse met de meeste stemmen wint | `ovo_votes(x, names, pairs)` |
        | **output code** | een rij van $+1$ en $-1$ voor elke klasse, een model voor elke kolom; de klasse waarvan de rij op de minste plaatsen van de antwoorden verschilt, wint | |
        | $W$ | de weights van de one-vs-rest-modellen, één kolom voor elke klasse | `W` |
        """),
        L.watch_out("De drie one-vs-rest-modellen worden los van elkaar getraind, dus hun scores hebben geen gedeelde schaal zoals de kansen van softmax. De hoogste nemen werkt in de praktijk goed, maar niets garandeert het. En one-vs-one kan eindigen in een gelijkspel: met drie klassen kunnen de drie modellen elk op een andere klasse stemmen. Hier wint dan de eerste klasse in `names`, zoals bij `np.argmax`; in het beeld gebeurt dat op 0,02% van het vlak."),
        mo.md(r"""
        Het beeld: de pinguïns uit de training set naar snavellengte en lichaamsgewicht, Adelies (rondjes), Chinstraps (driehoekjes) en Gentoos (vierkantjes), en de straten van deze les, elk met $C$ = 1. Links one-vs-rest: gestippeld de lijn van elk van zijn drie modellen, waar zijn score 0 is; <span class="c-pred">in blauw</span> waar de hoogste score van soort wisselt. Rechts one-vs-one: gestippeld de lijn van het model van elk paar; in blauw waar de meeste stemmen van soort wisselen.
        """),
    ])
    return


@app.cell(hide_code=True)
def figure_ways():
    mo.Html(C.ways_picture({"ovr": "One-vs-rest", "ovo": "One-vs-one", "x": "snavellengte (mm)", "y": "lichaamsgewicht (g)",
                            "alt": "Twee grafieken van de pinguïns uit de training set naar snavellengte en lichaamsgewicht: Adelies linksonder, Chinstraps rechtsonder, Gentoos bovenaan. In elk drie gestippelde lijnen, de lijnen van de drie modellen, en drie blauwe grenzen die samenkomen bij ongeveer 45 mm en 4.200 g en het vlak in een deel voor elke soort verdelen. Links, bij one-vs-rest, volgen de blauwe grenzen de gestippelde lijnen niet; rechts, bij one-vs-one, zijn ze er stukken van. De twee sets blauwe grenzen liggen dicht bij elkaar."}))
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Een eigen pinguïn: sleep de stip over het vlak, of verplaats hem met de pijltjestoetsen. De grafiek ernaast toont de pinguïns uit de training set, de grenzen van one-vs-rest in blauw en die van one-vs-one gestippeld; de tekst geeft de score van elk one-vs-rest-model en de stemmen van one-vs-one.
    """)
    return


@app.cell(hide_code=True)
def plane_penguin_cell():
    _names = ["Adelie", "Chinstrap", "Gentoo"]
    _yes = ["geen enkel model zegt ja", "één model zegt ja", "twee modellen zeggen ja", "alle drie zeggen ja"]

    def _frame(_b, _m):
        _s, _v = C.scores(_b, _m), C.votes(_b, _m)
        _n = sum(1 for _x in _s if _x > 0)
        _r, _o = C.ovr_says(_b, _m), C.ovo_says(_b, _m)
        _said = (f"Een pinguïn van {_b} mm en {C.grouped(_m, L.lang)} g. De scores van one-vs-rest: "
                 + ", ".join(f"{_a} {L.number(_x, 2)}" for _a, _x in zip(_names, _s))
                 + f"; {_yes[_n]}. One-vs-rest zegt **{_r}**. De stemmen van one-vs-one: "
                 + ", ".join(f"{_a} {_x}" for _a, _x in zip(_names, _v))
                 + f". One-vs-one zegt **{_o}**." + (" Ze zijn het oneens." if _r != _o else ""))
        _alt = (f"De pinguïns uit de training set met de grenzen van one-vs-rest en one-vs-one, en een nieuwe pinguïn op {_b} mm en "
                f"{C.grouped(_m, L.lang)} g: one-vs-rest zegt {_r}, one-vs-one zegt {_o}.")
        return C.plane_picture(_b, _m, {"title": "Jouw pinguïn en de twee manieren", "x": "snavellengte (mm)",
                                         "y": "lichaamsgewicht (g)", "alt": _alt}), _said

    _plane = L.plane("penguin", "snavellengte (mm)", "gewicht (g)", _frame, C.BILLS, C.MASSES, start=(40, 3600), role="",
                     task="Zoek een pinguïn waarvoor geen enkel one-vs-rest-model ja zegt, en een waarover de twee manieren het oneens zijn.")
    _plane
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In het midden, tussen de drie soorten, zegt geen enkel one-vs-rest-model ja: bij 44 mm van 3.600 tot 4.400 g, bij 46 mm en 4.000 of 4.400 g, en bij 48 mm en 4.400 g is een pinguïn voor alle drie de modellen "de rest". One-vs-rest antwoordt toch, met de hoogste score, en op elk van deze punten geeft one-vs-one hetzelfde antwoord. Aan de randen zeggen vaak twee modellen ja: bij 34 mm en 5.200 g zowel het Adelie-model (4,15) als het Gentoo-model (1,68). Op deze punten zijn de twee manieren het maar twee keer oneens, bij 38 mm en 5.200 g (one-vs-rest Adelie, one-vs-one Gentoo) en bij 54 mm en 4.800 g (Gentoo tegen Chinstrap), allebei buiten de wolken pinguïns in de grafiek. Waar de modellen weinig voorbeelden zagen, vertelde niets ze wat ze moesten zeggen.

    ### Recept

    Zo classificeer je veel klassen met modellen voor twee:

    1. Train een model voor elke klasse tegen de rest (one-vs-rest), of voor elk paar klassen op de voorbeelden van die twee (one-vs-one).
    2. One-vs-rest: de score van elk model, `x @ W + b`, en de klasse van de hoogste, `np.argmax`.
    3. One-vs-one: het model van elk paar geeft één stem, aan zijn tweede klasse bij een score boven 0 en anders aan zijn eerste; de klasse met de meeste stemmen wint.
    4. Vergelijk de manieren op de validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een Gentoo uit de validation set: 45,8 mm en 4.200 g, gestandaardiseerd zoals in de les over k-NN. De cel bevat de drie one-vs-rest-modellen en de drie one-vs-one-modellen van de course, elk een straat met een soft margin en $C$ = 1, en vraagt beide manieren naar deze pinguïn. De opmerkingen met `#` zeggen welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_ways():
    import numpy as np

    # de Gentoo van 45,8 mm en 4.200 g, gestandaardiseerd zoals in de les over k-NN
    x = np.array([0.361, 0.058])
    names = ["Adelie", "Chinstrap", "Gentoo"]

    # 1. one-vs-rest: een kolom van W en een getal van b voor elke soort, die soort tegen de andere twee
    W = np.array([[-2.509, 2.648, 0.392], [-0.146, -2.337, 2.709]])
    b = np.array([-0.264, -1.743, -1.229])

    # 2. de score van elk model, en de soort van de hoogste
    scores = x @ W + b
    print(scores)
    print(names[np.argmax(scores)])

    # 1. one-vs-one: een model voor elk paar, alleen getraind op die twee soorten; boven 0 is een stem voor de tweede
    pairs = [("Adelie", "Chinstrap", np.array([2.335, -1.301]), -0.804),
             ("Adelie", "Gentoo", np.array([1.759, 1.813]), -0.433),
             ("Chinstrap", "Gentoo", np.array([-1.416, 2.293]), 0.765)]

    # 3. de score van het model van elk paar
    for first, second, w, b_pair in pairs:
        print(first, second, round(x @ w + b_pair, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Alle drie de scores liggen onder 0: geen enkel model zegt dat deze pinguïn zijn soort is. De hoogste, $-0{,}923$, is die van het Chinstrap-model, nog geen 0,01 boven die van het Gentoo-model: one-vs-rest zegt Chinstrap, en zit fout. De paren: Adelie tegen Chinstrap geeft $-0{,}037$, een stem voor Adelie; Adelie tegen Gentoo geeft 0,307 en Chinstrap tegen Gentoo 0,387, twee stemmen voor Gentoo. One-vs-one zegt Gentoo, en zit goed. Op alle 68 pinguïns uit de validation set heeft one-vs-rest er 4 fout en one-vs-one 3; ze zijn het over 3 pinguïns oneens, en op de training set hebben ze er allebei 14 van de 206 fout.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ovr():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De hoogste score

        `ovr_predict(names, W, b, x)` moet de soort geven die one-vs-rest zegt voor één pinguïn `x`, zoals stap 2 van het recept. Vervang de twee `...`. De check roept `ovr_predict(names, W, b, np.array([-1.0, -0.4]))` aan met de `names`, `W` en `b` uit het uitgewerkte voorbeeld, en probeert er nog vijf: pinguïns waarvoor geen enkel model of twee modellen ja zeggen, vier verzonnen klassen, en drie metingen. Het beeld onder de check voert je functie uit op alle 68 pinguïns uit de validation set.
        """),
        L.task("Laat `ovr_predict` de soort teruggeven van het model met de hoogste score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ovr():
    box_ovr = L.code_box("ovr-predict", lines=6)
    box_ovr
    return (box_ovr,)


@app.cell(hide_code=True)
def check_ovr(box_ovr):
    result_ovr = L.check("ovr-predict", box_ovr.value)
    result_ovr
    return (result_ovr,)


@app.cell(hide_code=True)
def hints_ovr():
    L.hints(
        "ovr-predict",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld: eerst een score voor elk model, dan de soort van de hoogste.",
            "`x @ W` geeft één getal voor elke kolom van `W`, één voor elk model; tel `b` erbij op. `np.argmax` geeft de positie van de hoogste, en `names[...]` de soort op die positie.",
            "`scores = x @ W + b` en `return names[np.___(scores)]`.",
        ],
        why="Voor `np.array([-1.0, -0.4])` zijn de scores 2,303, −3,456 en −2,705: alleen het Adelie-model zegt ja, en het heeft de hoogste score.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ovo():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tel de stemmen

        `ovo_votes(x, names, pairs)` moet de stemmen tellen van de one-vs-one-modellen voor één pinguïn `x`, zoals stap 3 van het recept. `pairs` is een lijst zoals die in het uitgewerkte voorbeeld: `(first, second, w, b)` voor elk paar klassen. De eerste regels geven elke klasse 0 stemmen; vervang de `...` in de lus. De check roept `ovo_votes(np.array([0.361, 0.058]), names, pairs)` aan met de Gentoo en de `pairs` uit het uitgewerkte voorbeeld, en probeert er nog vijf: drie andere pinguïns, vier verzonnen klassen met zes paren, en drie metingen.
        """),
        L.task("Laat het model van elk paar één stem geven, en `ovo_votes` de stemmen van elke klasse teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ovo():
    box_ovo = L.code_box("ovo-votes", lines=10)
    box_ovo
    return (box_ovo,)


@app.cell(hide_code=True)
def check_ovo(box_ovo):
    result_ovo = L.check("ovo-votes", box_ovo.value)
    result_ovo
    return (result_ovo,)


@app.cell(hide_code=True)
def hints_ovo():
    L.hints(
        "ovo-votes",
        [
            "Stap 3 van het recept. Het model van elk paar geeft precies één stem: bereken zijn score, en kijk of die boven 0 ligt.",
            "Een `if` met de score, `x @ w + b > 0`, en een `else`. Een stem voor `second` is `votes[second] = votes[second] + 1`.",
            "`if x @ w + b > 0:` en dan `votes[second] = votes[second] + 1`, en `else:` en dan `votes[___] = votes[___] + 1`, elk op een eigen regel, ingesprongen onder de `for`.",
        ],
        why="Adelie tegen Chinstrap geeft deze pinguïn −0,037: een stem voor Adelie. De andere twee paren geven 0,307 en 0,387: twee stemmen voor Gentoo.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_code():
    mo.md(r"""
    ### Oefening 3. Een fout model overstemd

    Een output code voor de drie soorten met vijf modellen. Elke rij is de code van een soort: het antwoord dat elk model ervoor leert. Model 1 leert $+1$ voor de Adelies en $-1$ voor de andere twee; model 4 leert $+1$ voor de Adelies en de Gentoos.

    | | model 1 | model 2 | model 3 | model 4 | model 5 |
    |---|---|---|---|---|---|
    | Adelie | $+1$ | $-1$ | $-1$ | $+1$ | $+1$ |
    | Chinstrap | $-1$ | $+1$ | $-1$ | $-1$ | $+1$ |
    | Gentoo | $-1$ | $-1$ | $+1$ | $+1$ | $-1$ |

    Voor een nieuwe pinguïn antwoorden de vijf modellen $-1$, $-1$, $-1$, $+1$, $+1$.
    """)
    return


@app.cell(hide_code=True)
def choice_code_cell():
    choice_code = L.choice(
        "output-code",
        question="Welke soort geeft de output code deze pinguïn?",
        options={
            "right": "Adelie: zijn rij verschilt maar op één plaats van de antwoorden",
            "not-adelie": "Chinstrap of Gentoo: model 1, dat alleen voor de Adelies $+1$ leert, zegt $-1$",
            "none": "Geen: geen enkele rij komt precies overeen met de antwoorden",
        },
        correct="right",
        why={
            "not-adelie": "Een output code vertrouwt nooit op één model: hij vergelijkt de hele rij. Tel op hoeveel plaatsen elke rij van de antwoorden verschilt. Zie *Output code* in *Het idee*.",
            "none": "Met vijf modellen komt een rij vaak niet precies overeen; je neemt de dichtstbijzijnde rij, die op de minste plaatsen verschilt. Zie *Output code* in *Het idee*.",
            "right": "Adelie verschilt alleen bij model 1, Chinstrap bij model 2 en 4, Gentoo bij model 3 en 5. Model 1 zat fout, en de andere vier overstemden het. Elke twee codes verschillen hier op minstens drie plaatsen, dus met één fout model is de goede rij nog steeds de dichtstbijzijnde: een error-correcting output code.",
        },
    )
    choice_code
    return (choice_code,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De modellen voor twee klassen zijn de straten uit [De breedste straat: de maximum-margin classifier](/course/open/ml-max-margin?lang=nl). De hoogste score nemen is de keuze uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl), maar daar leert één model alle scores samen, dus delen ze één schaal. Een model dat zelf met veel klassen overweg kan, zoals k-NN uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), heeft geen van beide manieren nodig. In de volgende les, *Geen model wint overal*, zet je de modellen van deze module naast elkaar: elk wint waar klopt wat het over de data aanneemt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_code, result_ovo, result_ovr):
    L.wrap_up(result_ovr, result_ovo, choice_code)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
