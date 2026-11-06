# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-color-spaces. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-color-spaces", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Een kleurenfoto is 600 pixels breed en 400 hoog. Wat is de shape van `np.array(Image.open(path))`?",
        options={
            "rows": "`(400, 600, 3)`",
            "width": "`(600, 400, 3)`",
            "flat": "`(600, 400)`",
        },
        correct="rows",
        why={
            "rows": "De shape van een array begint met de rijen, de hoogte, dan de kolommen, de breedte, en dan de 3 kleuren.",
            "width": "Dat is de volgorde die Pillow voor `.size` gebruikt, de breedte eerst. Wat komt er eerst in de shape van een array? Zie de tweede let-op in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
            "flat": "Een kleurenfoto heeft drie getallen per pixel. Waar staan die in de shape, en wat komt er eerst, de rijen of de kolommen? Zie [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_wrap():
    quick_wrap = L.choice(
        "quick-wrap",
        question="`a` en `b` zijn arrays met dtype uint8. Een getal in `a` is 10, en hetzelfde getal in `b` is 12. Wat is dat getal in `a - b`?",
        options={
            "minus": "−2",
            "plus": "2",
            "wrap": "254",
        },
        correct="wrap",
        why={
            "minus": "Dat geven gewone gehele getallen. Getallen met het dtype uint8 blijven tussen 0 en 255: waar komt −2 dan uit? Zie de eerste let-op in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
            "plus": "Dat is de grootte van het verschil, `abs`. Maar `a - b` met uint8 blijft tussen 0 en 255: wat wordt −2 dan? Zie de eerste let-op in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).",
            "wrap": "uint8 blijft tussen 0 en 255, dus −2 wordt 254. Zet je ze eerst om met `.astype(int)`, dan is het −2.",
        },
    )
    quick_wrap
    return (quick_wrap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om de rode appels op een foto te vinden, lijkt een regel als "rood boven 150" genoeg, maar een appel in de schaduw heeft van alles minder, en zijn rood zakt onder 150 terwijl hij net zo rood is. HSV schrijft een kleur als drie andere getallen: welke kleur, hoe sterk en hoe helder. Schaduw verandert dan alleen het laatste, en een regel voor "rood" blijft eenvoudig. Zo werkt ook het uitkiezen van één kleur, een green screen, en het veranderen van de kleuren van foto's om een model meer variatie te geven.

    **Een voorbeeld uit het dagelijks leven.** In een verfwinkel vraag je niet om hoeveelheden rood, groen en blauw. Je wijst een kleur aan op een kleurencirkel, en zegt dan hoe fel hij moet zijn, en hoe licht of donker.
    """)
    return


@app.cell(hide_code=True)
def predict_shade():
    guess_shade = L.guess(
        "shade",
        question="Een rode appel in de tekening van de course heeft in het licht de pixel `(171, 26, 26)`, en zo'n appel in de schaduw `(108, 16, 16)`. Je schrijft ze allebei als kleurtoon, verzadiging en helderheid: welke van de drie getallen verschillen?",
        options={
            "all": "Alle drie",
            "hue": "De kleurtoon: in de schaduw is het een ander rood",
            "value": "Alleen de helderheid",
        },
        reveal="**Alleen de helderheid**: 0,67 in het licht, 0,42 in de schaduw. Allebei hebben kleurtoon 0, rood, en verzadiging 0,85. De schaduw maakt rood, groen en blauw elk ongeveer 0,6 keer zo groot, en dat verandert hoe helder de pixel is, niet zijn kleur.",
    )
    guess_shade
    return (guess_shade,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    RGB zegt hoeveel rood, groen en blauw licht een pixel heeft: de drie getallen van een afbeeldingsbestand. HSV schrijft dezelfde kleur als drie andere getallen, elk van 0 tot en met 1: de kleurtoon, de verzadiging en de helderheid; de letters H, S en V komen van hun Engelse namen. De helderheid $v$ is de sterkste van de drie, op een schaal van 0 tot en met 1. De verzadiging is hoeveel daarvan kleur is en geen grijs: $(v - \text{min}) / v$, de sterkste min de zwakste, gedeeld door de sterkste, en 0 bij zwart, waar $v$ gelijk is aan 0. De kleurtoon zegt welke kleur: een plek op een cirkel van kleuren. De formule voor de kleurtoon heeft een geval voor elk van rood, groen en blauw, al naargelang welke de sterkste is, dus dat laat je `rgb_to_hsv` van matplotlib doen. Veel andere programma's schrijven de kleurtoon in graden, van 0 tot 360; matplotlib schrijft hem van 0 tot 1.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **kleurruimte** | een manier om een kleur als getallen op te schrijven | RGB, HSV |
    | **RGB** | een kleur als hoeveel rood, groen en blauw licht: de getallen van een afbeeldingsbestand | `img[:, :, 0]` is het rood |
    | **HSV** | een kleur als kleurtoon, verzadiging en helderheid, elk van 0 tot en met 1 | `rgb_to_hsv(img / 255)` |
    | **kleurtoon** | welke kleur: een plek op de kleurencirkel, met rood bij 0 en weer bij 1 | `hsv[:, :, 0]` |
    | **verzadiging** | hoe sterk de kleur is: 0 heeft geen kleur, zoals grijs en wit, 1 is de zuivere kleur | `hsv[:, :, 1]` |
    | **helderheid** | hoe helder: de sterkste van rood, groen en blauw | `hsv[:, :, 2]` |
    | **`&`**, **`|`** | en, of, voor twee maskers: pixel voor pixel | `(h < 0.05) | (h > 0.95)` |

    `rgb_to_hsv` wil floats van 0 tot en met 1, dus je deelt de afbeelding eerst door 255; je krijgt een array met dezelfde shape, met kleurtoon, verzadiging en helderheid in plaats van rood, groen en blauw. Python doet `&` en `|` vóór `<` en `>`, dus elke vergelijking krijgt haar eigen haakjes.
    """)
    return


@app.cell(hide_code=True)
def picture_strips():
    mo.vstack([
        mo.md("**De drie getallen van HSV**"),
        mo.Html(C.strips({"hue": "Kleurtoon: welke kleur", "saturation": "Verzadiging van rood: van wit, via roze, tot het zuivere rood",
                          "value": "Helderheid van rood: van zwart tot het helderste rood",
                          "hue_ticks": ["0\nrood", "1/6\ngeel", "1/3\ngroen", "1/2\ncyaan", "2/3\nblauw", "5/6\nmagenta", "1\nrood"],
                          "saturation_ticks": ["0", "0,5", "1"], "value_ticks": ["0", "0,5", "1"],
                          "alt": "Drie stroken van 0 tot 1. De kleurtoon gaat de kleuren rond: rood bij 0, geel bij 1/6, groen bij 1/3, cyaan bij 1/2, blauw bij 2/3, magenta bij 5/6, en weer rood bij 1. De verzadiging van rood gaat van wit via roze naar felrood; de helderheid van rood van zwart naar felrood."})),
        mo.md(r"""
        Elke strook verandert één van de drie getallen, van 0 tot 1. De kleurtoon gaat één keer de kleuren rond en eindigt waar hij begon: bij 0 en bij 1 is het hetzelfde rood. De verzadiging van rood gaat van wit, zonder kleur, via roze naar het zuivere rood; de helderheid van zwart, zonder licht, naar het helderste rood.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Rood ligt aan beide einden van de kleurtoon: net boven 0 en net onder 1, zoals 23:59 en 0:01 vlak bij elkaar liggen op een klok. Een regel voor rood heeft beide einden nodig: `(h < 0.05) | (h > 0.95)`."),
        L.watch_out("Grijs, wit en zwart hebben geen kleur: verzadiging 0, en hun kleurtoon betekent niets. `rgb_to_hsv` geeft ze kleurtoon 0, de kleurtoon van rood, dus een regel op alleen de kleurtoon pakt ook grijs. Voeg een voorwaarde op de verzadiging toe."),
        L.watch_out("Het woord verzadiging heeft bij neural networks ook een andere betekenis, die in een latere les komt. In deze les betekent het altijd hoe sterk een kleur is."),
        mo.md(r"""
        ### Recept

        Zo kies je de pixels van één kleur uit:

        1. Maak floats van 0 tot en met 1 van de afbeelding: `img / 255`.
        2. Schrijf elke pixel in HSV: `hsv = rgb_to_hsv(img / 255)`.
        3. Haal de getallen uit elkaar: `h = hsv[:, :, 0]` en `s = hsv[:, :, 1]`; de helderheid is `hsv[:, :, 2]`.
        4. Schrijf de regel als masker: een venster van kleurtoon voor de kleur, en een verzadiging boven een drempel, zodat grijs erbuiten blijft. Verbind de vergelijkingen met `&` en `|`, elk tussen haakjes.
        5. Tel wat hij pakt: met een label voor elke pixel is `mask[labels == 4].mean()` het deel van de pixels met label 4 dat hij vindt, zoals `(diff > 0).mean()` een deel was in [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl).
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course tekende fruit op een roze geruit tafelkleed, onder een lamp aan de linkerkant: `data/photos/fruit.png`. Omdat de course hem zelf maakte, weet hij ook wat elke pixel is: `data/photos/fruit-labels.png` heeft 0 voor het tafelkleed, 1 voor een appel, 2 voor de sinaasappel, 3 voor de citroen en 4 voor de limoen. Het voorbeeld zoekt de limoen, met een regel in RGB en een in HSV. Voer de twee cellen uit.
    """)
    return


@app.cell
def example_hsv():
    import numpy as np
    from PIL import Image
    from matplotlib.colors import rgb_to_hsv

    img = np.array(Image.open("data/photos/fruit.png"))
    labels = np.array(Image.open("data/photos/fruit-labels.png"))
    # 1. en 2. floats van 0 tot en met 1, en elke pixel in HSV
    hsv = rgb_to_hsv(img / 255)
    print(img.shape, hsv.shape)
    # het midden van een appel in het licht, en van een in de schaduw
    print(img[70, 70], hsv[70, 70])
    print(img[175, 260], hsv[175, 260])
    return hsv, img, labels


@app.cell
def example_lime(hsv, img, labels):
    # 3. de kleurtoon en de verzadiging apart
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # 4. een regel voor groen: in RGB op alleen het groen, in HSV op kleurtoon en verzadiging
    green_rgb = img[:, :, 1] > 150
    green_hsv = (h > 0.2) & (h < 0.45) & (s > 0.5)
    # 5. het deel van de limoen dat elke regel vindt, en hoeveel andere pixels hij pakt
    lime = labels == 4
    print(green_rgb[lime].mean(), green_rgb[labels != 4].sum())
    print(green_hsv[lime].mean(), green_hsv[labels != 4].sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tekening heeft shape `(240, 360, 3)`, en `hsv` ook: drie getallen per pixel, nu kleurtoon, verzadiging en helderheid. De twee appels hebben allebei kleurtoon 0 en, afgerond, verzadiging 0,85, maar helderheid 0,67 in het licht en 0,42 in de schaduw. De regel in RGB vindt niets van de limoen: die ligt rechts, waar de lamp minder licht geeft, en zijn groen blijft daar onder 150. Wel pakt hij 6.163 andere pixels: van de citroen, waarvan het geel veel groen heeft, van het tafelkleed in het licht, en van de witte glans op het andere fruit. De regel in HSV vindt 0,988 van de limoen, 98,8%, en verder niets. Wat hij mist, is de glans van de lamp, waar de kleur vervaagt en de verzadiging onder 0,5 zakt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stronger():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Sterk of helder
    """)
    return


@app.cell(hide_code=True)
def choice_stronger_cell():
    choice_stronger = L.choice(
        "stronger",
        question="Welke van deze twee kleuren heeft de hoogste verzadiging: donkerrood `(100, 0, 0)` of roze `(255, 200, 200)`?",
        options={
            "dark": "Het donkerrood",
            "pink": "Het roze",
            "same": "Geen van beide: allebei rood, dus dezelfde verzadiging",
        },
        correct="dark",
        why={
            "dark": "Het donkerrood heeft helemaal geen groen en geen blauw: verzadiging (100 − 0) / 100 = 1, de zuivere kleur, alleen donker. Het roze is zo helder als het kan, helderheid 1, maar er zit veel groen en blauw in: verzadiging (255 − 200) / 255 = 0,22.",
            "pink": "Het roze is helderder: dat is zijn helderheid, 1. Verzadiging zegt hoe sterk de kleur is: hoeveel groen en blauw zit er door het rood? Reken het uit met de formule onder Het idee.",
            "same": "Allebei hebben kleurtoon 0, rood. Maar verzadiging zegt hoeveel er van de andere twee door zit. Reken het uit: de sterkste min de zwakste, gedeeld door de sterkste.",
        },
    )
    choice_stronger
    return (choice_stronger,)


@app.cell(hide_code=True)
def text_exercise_value_saturation():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Helderheid en verzadiging met de hand

        Maak `value_saturation(pixel)` af. Hij krijgt één pixel, drie getallen van 0 tot en met 255, en geeft zijn helderheid en zijn verzadiging terug, elk van 0 tot en met 1. De eerste regel maakt er al floats van 0 tot en met 1 van, in `rgb`. Vervang de `...`. De check roept hem aan met een bruinig oranje:

        ```python
        value_saturation([204, 102, 51])
        ```

        Dat moet `(0.8, 0.75)` geven. Hij probeert ook andere pixels, waaronder grijs en zwart.
        """),
        L.task("Laat `value_saturation(pixel)` de helderheid en de verzadiging van de pixel teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value_saturation():
    box_value_saturation = L.code_box("value-saturation", lines=9)
    box_value_saturation
    return (box_value_saturation,)


@app.cell(hide_code=True)
def check_value_saturation(box_value_saturation):
    result_value_saturation = L.check("value-saturation", box_value_saturation.value)
    result_value_saturation
    return (result_value_saturation,)


@app.cell(hide_code=True)
def hints_value_saturation():
    L.hints(
        "value-saturation",
        [
            "De helderheid is de sterkste van de drie: `rgb.max()`. De verzadiging is de sterkste min de zwakste, gedeeld door de sterkste.",
            "Zwart is `[0, 0, 0]`: dan is de helderheid 0, en door 0 kun je niet delen. Welke verzadiging heeft een pixel zonder licht?",
            "`v = rgb.max()`, dan `if v == 0:` met `return 0.0, 0.0`, dan `s = (v - rgb.___()) / ___` en `return v, s`.",
        ],
        why="Bij `[204, 102, 51]` is `rgb` gelijk aan `[0.8, 0.4, 0.2]`. De helderheid is de sterkste, 0,8; de verzadiging is (0,8 − 0,2) / 0,8 = 0,75. Bij zwart is de helderheid 0: geen licht, dus geen kleur, en verzadiging 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.md(r"""
    ### Oefening 3. Laat het tafelkleed weg

    De regel voor rood pakt een kleurtoon onder 0,05 of boven 0,95. Roze is ook een rood, met weinig verzadiging, dus het tafelkleed valt ook in het venster. Met de slider zet je de drempel: het masker houdt alleen pixels met een verzadiging erboven. Wat het masker pakt, houdt zijn kleuren; de rest wordt grijs.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _html, _f = C.threshold_frame(t, {"alt": "De tekening met fruit, met in kleur wat het masker pakt bij verzadigingsdrempel {t}, en de rest grijs."})
        return (_html,
                f'Drempel {L.number(t, 1)}: het masker vindt {L.number(100 * _f["apple"], 1)}% van de appelpixels, en <span class="c-loss ea-num">{C.grouped(_f["cloth"], ".")}</span> pixels van het tafelkleed.')

    explore_threshold = L.explore("threshold", "drempel", _frame, start=0.0, values=C.THRESHOLDS, role="hyper",
                                  task="Zoek de drempel die de meeste appelpixels vindt zonder één pixel van het tafelkleed.")
    explore_threshold
    return (explore_threshold,)


@app.cell(hide_code=True)
def check_threshold(explore_threshold):
    result_threshold = L.check("threshold", explore_threshold)
    result_threshold
    return (result_threshold,)


@app.cell(hide_code=True)
def hints_threshold():
    L.hints(
        "threshold",
        [
            "Schuif de slider stap voor stap omhoog, en kijk naar de ruiten van het tafelkleed: hebben ze nog kleur?",
            "De zin onder het plaatje telt de pixels van het tafelkleed die het masker pakt. Je wilt 0, met zoveel mogelijk appelpixels.",
            "Hoe hoger de drempel, hoe minder van het tafelkleed, maar bovenaan verdwijnen ook de appels. Ga omhoog tot het tafelkleed weg is, en stop daar.",
        ],
        why="Bij 0,4 pakt het masker nog 26.038 pixels van het tafelkleed; bij 0,5 geen, met 99,1% van de appelpixels. Het roze van het tafelkleed heeft een verzadiging tot 0,47, de appels ongeveer 0,85.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_red():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Een masker voor rood

        Maak `red_mask(hsv)` af. Hij krijgt een afbeelding in HSV, zoals `rgb_to_hsv` hem maakt, en geeft een masker terug: `True` voor elke rode pixel. Rood is een kleurtoon onder 0,05 of boven 0,95, met een verzadiging boven 0,5, de drempel van oefening 3. De eerste regels halen `h` en `s` al uit elkaar. Vervang de `...`. De check roept hem aan met één rij van vijf pixels, in HSV:

        ```python
        red_mask([[[0.02, 0.9, 0.8], [0.98, 0.7, 0.5], [0.08, 0.9, 0.9], [0.0, 0.0, 0.6], [0.01, 0.3, 0.9]]])
        ```

        Dat moet `[[True, True, False, False, False]]` geven: rood, rood aan het andere eind van de cirkel, oranje, grijs en roze. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `red_mask(hsv)` `True` geven voor elke pixel met een kleurtoon onder 0,05 of boven 0,95 en een verzadiging boven 0,5."),
    ])
    return


@app.cell(hide_code=True)
def exercise_red():
    box_red = L.code_box("red-mask", lines=9)
    box_red
    return (box_red,)


@app.cell(hide_code=True)
def check_red(box_red):
    result_red = L.check("red-mask", box_red.value)
    result_red
    return (result_red,)


@app.cell(hide_code=True)
def hints_red():
    L.hints(
        "red-mask",
        [
            "Lees de twee let-ops boven het recept: waar ligt rood op de kleurtoon, en welke kleurtoon krijgt een grijze pixel?",
            "Twee voorwaarden op de kleurtoon, verbonden met `|`, samen tussen haakjes; dan `&` met de voorwaarde op de verzadiging.",
            "`(h < 0.05) | (h > 0.95) & (s > 0.5)` is het niet: Python doet eerst de `&`, dus de verzadiging telt maar voor één eind. Waar komen haakjes, zodat de twee voorwaarden op de kleurtoon samen tellen?",
        ],
        why="`(h < 0.05) | (h > 0.95)` is rood aan beide einden van de cirkel; de haakjes eromheen laten Python dat vóór de `&` doen. `& (s > 0.5)` laat grijs en roze weg. Voor de rij van het voorbeeld geeft dat `True, True, False, False, False`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl) was een pixel drie getallen voor rood, groen en blauw; nu schrijf je dezelfde kleur als kleurtoon, verzadiging en helderheid, en houdt een regel voor een kleur stand in het licht en in de schaduw. In [Delen van een array kiezen](/course/open/np-indexing?lang=nl) kwam een masker uit één vergelijking; nu verbind je vergelijkingen met `&` en `|`. Later in deze route krijgt een green screen zijn masker op dezelfde manier, uit hoe groen elke pixel is. In de volgende les, *Grijstinten spreiden: histogramegalisatie*, spreid je de helderheid van een hele afbeelding over de hele schaal.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stronger, result_red, result_threshold, result_value_saturation):
    L.wrap_up(choice_stronger, result_value_saturation, result_threshold, result_red)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
