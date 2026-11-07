# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-alpha-compositing. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-alpha-compositing", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen, over [Broadcasting](/course/open/np-broadcasting?lang=nl) en over [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`m` heeft shape `(2, 3)` en `col` heeft shape `(2, 1)`. Wat geeft `m * col`?",
        options={
            "fits": "Een array met shape `(2, 3)`",
            "error": "Een fout: 3 en 1 verschillen",
            "small": "Een array met shape `(2, 1)`",
            "more": "Een array met shape `(2, 3, 1)`",
        },
        correct="fits",
        why={
            "fits": "Rechts uitgelijnd: 3 en 1, en een van de twee is 1, dus rekt NumPy de kolom uit over de 3 kolommen; dan 2 en 2, gelijk. Elk getal in een rij wordt vermenigvuldigd met het getal van `col` in die rij. In deze les krijgt de alpha zo'n lengte 1, zodat hij op de drie kleuren past.",
            "error": "3 en 1 verschillen, maar een van de twee is 1, en NumPy rekt een lengte 1 uit tot de andere lengte. De shapes passen: de uitkomst heeft shape `(2, 3)`.",
            "small": "NumPy rekt de lengte 1 van `col` uit tot 3; van `m` gooit hij niets weg. De uitkomst heeft shape `(2, 3)`.",
            "more": "Beide shapes hebben twee lengtes, dus de uitkomst ook: rechts uitgelijnd geven 3 en 1 samen 3, en 2 en 2 samen 2. De uitkomst heeft shape `(2, 3)`.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="`h` is een array met de kleurtoon van elke pixel. Welke regel geeft True voor de pixels waarvan de kleurtoon tussen 0,3 en 0,45 ligt?",
        options={
            "amp": "`(h > 0.3) & (h < 0.45)`",
            "and": "`h > 0.3 and h < 0.45`",
            "bar": "`(h > 0.3) | (h < 0.45)`",
            "bare": "`h > 0.3 & h < 0.45`",
        },
        correct="amp",
        why={
            "amp": "`&` verbindt twee maskers pixel voor pixel: True waar ze allebei True zijn. Elke vergelijking krijgt haar eigen haakjes, omdat Python `&` vóór `<` en `>` doet.",
            "and": "`and` wil één True of False, en een array heeft er één per pixel: Python stopt met een fout. Verbind maskers met `&`.",
            "bar": "`|` geeft True waar minstens een van de twee True is, en elke kleurtoon ligt boven 0,3 of onder 0,45: alle pixels dus. Je wilt ze allebei: `&`.",
            "bare": "Python doet `&` vóór `<` en `>`, dus probeert het eerst `0.3 & h`, en stopt met een fout. Zet elke vergelijking tussen haakjes.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een weerpresentator staat voor een groene wand, en jij ziet een kaart achter hem of haar. Een film zet acteurs op een planeet die niet bestaat. Een logo of sticker op een website ligt over wat erachter staat, met zachte randen. Elke keer ligt het ene beeld over het andere, en per pixel zegt een getal hoeveel het voorste beeld van het achterste bedekt. In AI kom je dezelfde mix tegen: een model dat een voorwerp uit een foto knipt, geeft zo'n getal per pixel, en uitgeknipte voorwerpen op veel achtergronden plakken is een goedkope manier om meer trainingsfoto's te maken.

    **Een voorbeeld uit het dagelijks leven.** Een weerpresentator draagt nooit groen: alles wat groen is voor de wand, zou ook verdwijnen, en dan schijnt de kaart door de kleren heen.
    """)
    return


@app.cell(hide_code=True)
def predict_mix():
    guess_mix = L.guess(
        "mix",
        question="Een oranje pixel van een sticker, rood 240, groen 120 en blauw 0, ligt over een blauwe pixel van de achtergrond, (0, 0, 240). De sticker bedekt er een kwart van: de achtergrond schijnt voor driekwart door. Welke kleur komt eruit?",
        options={
            "orange": "(240, 120, 0)",
            "half": "(120, 60, 120)",
            "quarter": "(60, 30, 180)",
            "three": "(180, 90, 60)",
        },
        reveal="**(60, 30, 180)**: vooral blauw, met een vleugje oranje. Elke kleur is een kwart van de sticker plus driekwart van de achtergrond: 0,25 × 240 + 0,75 × 0 = 60 rood, 0,25 × 120 + 0,75 × 0 = 30 groen, en 0,25 × 0 + 0,75 × 240 = 180 blauw. (240, 120, 0) is de sticker alleen, alsof hij alles bedekt; (120, 60, 120) de helft van elk; en (180, 90, 60) driekwart van de sticker, met de gewichten omgekeerd.",
    )
    guess_mix
    return (guess_mix,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een PNG kan per pixel een vierde getal opslaan, naast rood, groen en blauw: **alpha**, hoeveel de pixel bedekt van wat erachter ligt. 255 bedekt het helemaal, 0 helemaal niet, en 102 voor 40 procent. Een beeld met een **alpha channel** heeft shape `(rijen, kolommen, 4)`: rood, groen, blauw en alpha, kortweg RGBA. Een sticker heeft alpha 255 op de tekening, 0 eromheen, en waarden daartussen op zijn zachte rand. Een schaduw is zwart met een alpha daartussen: hij maakt wat erachter ligt donkerder, zonder het te verbergen.

        **Alpha compositing** legt een voorste beeld over een achterste, pixel voor pixel. Met `a` de alpha van 0 tot en met 1, het alpha channel gedeeld door 255:

        `out = a * front + (1 - a) * back`

        voor elk van de drie kleuren. Bij `a = 1` zie je alleen het voorste beeld, bij `a = 0` alleen het achterste, en daartussen een mix. Een schaduwpixel van de sticker in deze les met alpha 102, dus `a = 0.4`, maakt de tafel erachter 0,6 keer zo helder.

        De shapes moeten passen. `sticker[:, :, :3]` bevat de kleuren, shape `(400, 600, 3)`. `sticker[:, :, 3]` geeft de alpha met shape `(400, 600)`: één as minder, en rechts uitgelijnd past hij niet op de kleuren. `sticker[:, :, 3:]` geeft dezelfde getallen met de as erbij, shape `(400, 600, 1)`, en die rekt NumPy uit over de drie kleuren, zoals in [Broadcasting](/course/open/np-broadcasting?lang=nl).

        Een foto voor een green screen heeft geen alpha channel: de alpha moet uit de kleuren komen. Dat is een **chroma key**. Elke pixel waarvan de kleurtoon het groen van het scherm is en de verzadiging sterk, hoort bij het scherm en krijgt alpha 0; elke andere pixel krijgt alpha 1. Een schermpixel in deze les is rood 47, groen 195 en blauw 92: kleurtoon 0,38 en verzadiging 0,76, dus scherm. Een pixel van de limoen is (100, 187, 47): ook groen, maar geelgroen, met kleurtoon 0,27. Een key waarvan het groen op 0,3 begint, houdt de limoen; een die op 0,2 begint, zoals de regel voor groen in [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl), laat hem verdwijnen.
        """),
        L.watch_out("Deel het alpha channel door 255 voordat je mengt. In uint8 geeft `1 - a` met `a` op 255 geen -254: het loopt rond naar 2."),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **alpha channel** | een vierde getal per pixel: hoeveel de pixel bedekt van wat erachter ligt, van 0 tot en met 255 in een PNG | `sticker[:, :, 3:]` |
        | **alpha compositing** | het ene beeld over het andere leggen, met per pixel `a` keer het voorste plus `1 - a` keer het achterste | `a * front + (1 - a) * back` |
        | **chroma key** | de alpha van een foto uit zijn kleuren maken: 0 waar de kleur van het scherm is, 1 overal anders | `1 - screen` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_alpha():
    mo.vstack([
        mo.md("**Een sticker en zijn alpha channel**"),
        mo.Html(C.alpha_picture({"alpha": "alpha channel", "over": "sticker over de koffiefoto",
                                 "alt": "Links het alpha channel van de sticker als grijs beeld: zwart waar het 0 is, wit waar het 255 is. Vier witte schijven, het fruit, elk met een zachte grijze vlek linksonder, de schaduw. Rechts de sticker over de koffiefoto: een sinaasappel rechtsboven, een limoen en een citroen rechtsonder, en een appel linksonder, elk met een zachte schaduw op de tafel."})),
        mo.md(r"""
        Links het alpha channel als grijs beeld: wit waar de sticker bedekt, zwart waar hij alles doorlaat. Het fruit is wit, met een dunne grijze rand; de schaduwen zijn donkergrijs, omdat ze een beetje bedekken. Rechts de sticker over de koffiefoto, gemengd met `a * front + (1 - a) * back`: het fruit ligt op de tafel, en de schaduwen maken het hout eronder donkerder.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    De course legde dezelfde sticker over een green screen en sloeg de foto op. Met de slider kies je waar het groen van de key begint: elke pixel met een kleurtoon vanaf daar tot 0,45 en een verzadiging boven 0,5 telt als scherm. Links zie je de alpha die de key maakt: wit waar de foto blijft, zwart waar de key scherm ziet; rechts de foto over de koffiefoto gelegd met die alpha.
    """)
    return


@app.cell(hide_code=True)
def explore_key_cell():
    def _say(low):
        _out = C.key_outcome(C.key(low))
        if _out["left"]:
            return "Het scherm ligt op 0,38 tot 0,40, onder het begin van het bereik, dus er verdwijnt niets van."
        _names = {"orange": "de sinaasappel (0,075)", "lemon": "de citroen (0,14)", "lime": "de limoen (0,27)"}
        _gone = [_names[name] for name in ("orange", "lemon", "lime") if _out["lost"][name]]
        if not _gone:
            return "Alle vier de stukken fruit blijven, en het hele scherm verdwijnt."
        if len(_gone) == 1:
            return (f"De limoen, op 0,27, telt ook als scherm: {C.grouped(_out['lost']['lime'], '.')} van zijn "
                    f"{C.grouped(C.fruit_parts()['lime'].sum(), '.')} pixels verdwijnen, op de witte glans na.")
        _stay = "De sinaasappel, op 0,075, en de appel, op 0, blijven." if not _out["lost"]["orange"] else "De appel, op kleurtoon 0, blijft."
        return (f"{', '.join(_gone[:-1]).capitalize()} en {_gone[-1]} tellen ook als scherm, en verdwijnen; alleen hun "
                "witte glans blijft, omdat die bijna geen verzadiging heeft. " + _stay)

    def _frame(low):
        _html = C.key_frame(low, {"alpha": "alpha van de key", "over": "over de koffiefoto",
                                  "alt": "Links de alpha van een key waarvan het groen begint bij een kleurtoon van {low:.2f}: wit waar de foto blijft, zwart waar de key scherm ziet. Rechts de foto voor het green screen over de koffiefoto gelegd met die alpha."})
        return _html, _say(low)

    explore_key = L.explore("key", "groen begint bij", _frame, start=0.2, values=C.BOUNDS, role="",
                            task="Zoek waar het groen kan beginnen, zodat alle vier de stukken fruit blijven en er geen groen overblijft.")
    explore_key
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij 0,30 blijft elk stuk fruit staan en verdwijnt het hele scherm. Bij 0,20, waar de regel voor groen van [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) begint, en bij 0,25 verdwijnt de limoen ook: die regel was gemaakt om de limoen te vinden. Bij 0,40 mist het bereik het scherm, dat op 0,38 tot 0,40 ligt. Kijk goed naar de randen van het fruit, en naar waar de schaduwen gebleven zijn: oefening 4 komt erop terug.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`.astype(np.uint8)` laat weg wat na de komma komt: 67,9 wordt 67. Tel er eerst 0,5 bij op, dan rondt hij af: `(out + 0.5).astype(np.uint8)`."),
        mo.md(r"""
        ### Recept

        Om een sticker `sticker` met een alpha channel over een even grote achtergrond `back` te leggen:

        1. Neem de kleuren: `front = sticker[:, :, :3]`.
        2. Neem de alpha van 0 tot en met 1, met de as erbij: `a = sticker[:, :, 3:] / 255`.
        3. Meng: `out = a * front + (1 - a) * back`.
        4. Rond af en maak er weer uint8 van: `(out + 0.5).astype(np.uint8)`.

        Bij een foto voor een green screen komt de alpha uit de kleuren, in plaats van stap 1 en 2:

        5. `hsv = rgb_to_hsv(photo / 255)`, met `h = hsv[:, :, 0:1]` en `s = hsv[:, :, 1:2]`: de kleurtoon en de verzadiging, elk met de as erbij.
        6. `screen` is True waar `h` in het groen van het scherm ligt en `s` sterk is, verbonden met `&`.
        7. `a = 1 - screen`, die de shape `(rijen, kolommen, 1)` van `h` heeft, en meng de foto zelf als voorste beeld.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest de sticker en de koffiefoto, en print hun shapes. Hij neemt de kleuren en de alpha, stap 1 en 2 van het recept, en print de shape van de alpha en hoeveel pixels alpha 1 en alpha 0 hebben. Hij legt de sticker over de koffiefoto, stap 3 en 4, print één pixel van een schaduw ervoor en erna, en laat de uitkomst zien.
    """)
    return


@app.cell
def example_sticker():
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    sticker = np.array(Image.open("data/photos/fruit-sticker.png"))
    coffee = np.array(Image.open("data/photos/coffee.png"))
    print(sticker.shape, coffee.shape)
    # 1. en 2. de kleuren, en de alpha van 0 tot en met 1 met de as erbij
    front = sticker[:, :, :3]
    a = sticker[:, :, 3:] / 255
    print(a.shape, (a == 1).sum(), (a == 0).sum())
    # 3. de sticker over de koffiefoto
    out = a * front + (1 - a) * coffee
    # 4. weer uint8, afgerond
    out = (out + 0.5).astype(np.uint8)
    # een pixel van de schaduw onder de sinaasappel
    print(sticker[92, 495], coffee[92, 495], out[92, 495])
    plt.imshow(out)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De sticker heeft vier getallen per pixel, de koffiefoto drie, en `a` heeft shape `(400, 600, 1)`: één getal per pixel, dat NumPy uitrekt over de drie kleuren. 15.544 pixels hebben alpha 1, het fruit binnen zijn randen, en 220.176 alpha 0; de 4280 daartussen zijn de randen en de schaduwen. De schaduwpixel is zwart met alpha 102, dus `a = 0.4`: het hout erachter, (177, 99, 58), wordt (106, 59, 35), 0,6 keer zo helder en afgerond.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_axis():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De as van de alpha

    `sticker` is de sticker uit het uitgewerkte voorbeeld, shape `(400, 600, 4)`, en `coffee` de koffiefoto, shape `(400, 600, 3)`. Iemand neemt de alpha zonder zijn as: `a = sticker[:, :, 3] / 255`.
    """)
    return


@app.cell(hide_code=True)
def choice_axis_cell():
    choice_axis = L.choice(
        "axis",
        question="Wat geeft `a * sticker[:, :, :3]` dan?",
        options={
            "error": "Een fout: de shapes `(400, 600)` en `(400, 600, 3)` passen niet",
            "mix": "De kleuren van de sticker, elk keer zijn alpha",
            "alpha": "De alpha, drie keer",
        },
        correct="error",
        why={
            "error": "Rechts uitgelijnd staat 600 tegenover 3: die verschillen, en geen van beide is 1. `sticker[:, :, 3:]` houdt de as, met lengte 1: shape `(400, 600, 1)`, en die rekt NumPy uit over de drie kleuren.",
            "mix": "Dat geeft `sticker[:, :, 3:] / 255`, met de as erbij. Zonder as heeft `a` shape `(400, 600)`, en rechts uitgelijnd staat 600 tegenover 3.",
            "alpha": "NumPy kopieert geen array om hem passend te maken; hij rekt alleen een as van lengte 1 uit. `(400, 600)` tegenover `(400, 600, 3)`: rechts uitgelijnd staat 600 tegenover 3.",
        },
    )
    choice_axis
    return (choice_axis,)


@app.cell(hide_code=True)
def text_exercise_over():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het ene beeld over het andere

        Maak `over(sticker, back)` af. Die legt een sticker met een alpha channel over een even grote achtergrond, en geeft de mix terug als floats, nog niet afgerond. Vervang de twee `...`: `a` is de alpha van 0 tot en met 1, met de as erbij, en de functie geeft de mix van stap 1 tot en met 3 van het recept terug. De check roept hem aan met twee pixels: een oranje met alpha 51, een vijfde, over blauw, en een schaduwpixel met alpha 102 over het hout van de tafel:

        ```python
        sticker = np.array([[[240, 120, 0, 51], [0, 0, 0, 102]]], dtype=np.uint8)
        back = np.array([[[0, 0, 240], [180, 120, 80]]], dtype=np.uint8)
        over(sticker, back)
        ```

        Dat moet `[[[48, 24, 192], [108, 72, 48]]]` geven. De check probeert ook pixels met alpha 0 en 255, en een sticker die donkerder is dan zijn achtergrond.
        """),
        L.task("Laat `over(sticker, back)` de sticker met zijn alpha over de achtergrond mengen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_over():
    over_code = L.code_box("over", lines=4)
    over_code
    return (over_code,)


@app.cell(hide_code=True)
def check_over(over_code):
    result_over = L.check("over", over_code.value)
    result_over
    return (result_over,)


@app.cell(hide_code=True)
def hints_over():
    L.hints(
        "over",
        [
            "Stap 2 en 3 van het recept: eerst de alpha van 0 tot en met 1, dan de mix.",
            "`a` is het laatste getal van elke pixel, met `3:` zodat de as blijft, gedeeld door 255. De mix geeft de kleuren `a` en de achtergrond `1 - a`.",
            "`a = sticker[:, :, ___] / 255` en `return a * sticker[:, :, ___] + (1 - a) * back`.",
        ],
        why="Voor de oranje pixel: `a = 51 / 255 = 0.2`, dus 0,2 × 240 + 0,8 × 0 = 48 rood, 0,2 × 120 = 24 groen, en 0,8 × 240 = 192 blauw. Voor de schaduw: `a = 0.4` en de kleur is zwart, dus 0,6 × (180, 120, 80) = (108, 72, 48).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_key():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een key voor het green screen

        `data/photos/green-screen.png` is de foto van de slider: de sticker voor een green screen, 400 rijen bij 600 kolommen, zonder alpha channel. Het vak leest hem in en geeft je de kleurtoon `h` en de verzadiging `s` van elke pixel, elk met de as erbij, zoals `3:` de as van de alpha hield: shape `(400, 600, 1)`. Maak van `screen` een masker dat True is op het scherm en False op het fruit, en van `alpha` de alpha van de foto: 0 op het scherm, 1 overal anders, met dezelfde shape `(400, 600, 1)`, zodat hij past op de drie kleuren van de foto. Het groen van het scherm ligt op een kleurtoon van 0,38 tot 0,40; de limoen op 0,27.
        """),
        L.task("Laat `alpha` alle vier de stukken fruit houden en het green screen weglaten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_key():
    key_code = L.code_box("key", lines=11)
    key_code
    return (key_code,)


@app.cell(hide_code=True)
def check_key(key_code):
    result_key = L.check("key", key_code.value)
    result_key
    return (result_key,)


@app.cell(hide_code=True)
def hints_key():
    L.hints(
        "key",
        [
            "Stap 6 en 7 van het recept. Met de slider vond je waar het groen kan beginnen.",
            "`screen` verbindt drie vergelijkingen met `&`: de kleurtoon boven het begin, de kleurtoon onder 0,45, en de verzadiging boven 0,5. `1 - screen` maakt van True 0 en van False 1.",
            "`screen = (h > ___) & (h < 0.45) & (s > 0.5)` en `alpha = 1 - ___`.",
        ],
        why="Het scherm ligt op een kleurtoon van 0,38 tot 0,40, de limoen op 0,27, dus het groen kan overal daartussen beginnen, bijvoorbeeld op 0,3: `screen = (h > 0.3) & (h < 0.45) & (s > 0.5)`. `1 - screen` is 0 op het scherm en 1 overal anders, met de shape van `h`, `(400, 600, 1)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_edges():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Van dichtbij

        De course legde de foto voor het green screen twee keer over de koffiefoto: met de key waarvan het groen op 0,3 begint, en met de eigen alpha van de sticker. De sinaasappel, van dichtbij:
        """),
        mo.Html(C.edge_picture({"key": "met de key", "sticker": "met de alpha van de sticker",
                                "alt": "De sinaasappel van dichtbij, twee keer. Links over de koffiefoto gelegd met de key: zijn rand heeft trapjes van hele pixels, hier en daar met een dun groen randje, en er is geen schaduw. Rechts met de eigen alpha van de sticker: een gladde rand, en een zachte schaduw linksonder."})),
    ])
    return


@app.cell(hide_code=True)
def explain_edges_cell():
    explain_edges = L.explain(
        "edges",
        question="Met de key heeft de rand trapjes en een dun groen randje, en de schaduw is weg. Waar komen die vandaan? Denk aan wat een pixel op de rand van de sinaasappel in de foto voor het green screen bevatte, en welke alpha de key hem geeft. Welke alpha zou zo'n pixel eigenlijk moeten krijgen?",
        model_answer="**Een pixel op de rand is deels sinaasappel en deels scherm:** de foto is de sticker over het scherm, dus hij bevat een mix van de twee. **De key geeft elke pixel 0 of 1.** Een gemengde pixel met genoeg sinaasappel om zijn kleurtoon onder het begin van de key te trekken, krijgt 1, en neemt zijn deel groen mee: het randje. Een met minder sinaasappel krijgt 0, en zijn deel sinaasappel gaat verloren: elke randpixel is alles of niets, dus de rand krijgt trapjes. **De schaduw is donkerder groen:** hij heeft de kleurtoon en de verzadiging van het scherm, dus de key houdt hem voor scherm. **Een randpixel heeft een alpha tussen 0 en 1 nodig,** zoals de eigen alpha van de sticker op zijn zachte rand, zodat hij zich met de koffiefoto mengt zoals de sticker dat deed. Een zachtere key maakt zo'n alpha afhankelijk van hoe groen een pixel is, en haalt dan ook het groen van het scherm uit zijn kleur.",
        points=[
            "je zegt dat een pixel op de rand een mix van sinaasappel en scherm bevat",
            "je zegt dat de key alleen 0 of 1 geeft, dus dat een gemengde pixel zijn groen houdt of zijn oranje verliest",
            "je zegt dat de schaduw de kleurtoon en verzadiging van het scherm heeft, dus als scherm telt",
            "je zegt dat een randpixel een alpha tussen 0 en 1 nodig heeft, zodat hij zacht mengt",
        ],
    )
    explain_edges
    return (explain_edges,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het ene beeld over het andere is één mix per pixel: `a` keer het voorste plus `1 - a` keer het achterste, met `a` uit het alpha channel, dat een PNG als vierde getal kan opslaan. Een chroma key maakt die alpha uit de kleuren: met een masker op kleurtoon en verzadiging, zoals in [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl), is hij 0 of 1. Dat houdt het fruit en laat het scherm weg, maar geeft de randen trapjes en een groen randje. Filmstudio's gebruiken een zachte key: die geeft een alpha tussen 0 en 1, afhankelijk van hoe groen een pixel is, en haalt het groen uit de randen. Nieuwere programma's laten een neural network de alpha van elke pixel bepalen, zelfs zonder green screen. Dezelfde mix laat de ene scène overgaan in de andere, met `a` die in de tijd van 0 naar 1 gaat, en legt ondertitels over een film. Deze les sluit de module over beelden bewerken af; de volgende module, over een getraind model gebruiken, begint met *Een foto klaarmaken voor een model*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_axis, explain_edges, result_key, result_over):
    L.wrap_up(choice_axis, result_over, result_key, explain_edges)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
