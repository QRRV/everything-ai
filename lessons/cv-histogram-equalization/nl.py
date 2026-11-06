# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les cv-histogram-equalization. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-histogram-equalization", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_value():
    quick_value = L.choice(
        "quick-value",
        question="Wat is de helderheid van de pixel `(60, 180, 120)` in HSV?",
        options={
            "max": "180 / 255, ongeveer 0,71",
            "mean": "120 / 255, ongeveer 0,47",
            "saturation": "(180 − 60) / 180, ongeveer 0,67",
        },
        correct="max",
        why={
            "max": "De helderheid is de sterkste van rood, groen en blauw, op een schaal van 0 tot en met 1: 180 / 255, ongeveer 0,71.",
            "mean": "Dat is het gemiddelde van de drie. De helderheid van HSV neemt er één van de drie: welke? Zie Het idee in [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl).",
            "saturation": "Dat is de formule van de verzadiging: hoe sterk de kleur is. De helderheid zegt hoe helder: wat neemt die van de drie? Zie [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl).",
        },
    )
    quick_value
    return (quick_value,)


@app.cell(hide_code=True)
def quick_check_quantile():
    quick_quantile = L.choice(
        "quick-quantile",
        question="Voor een array met getallen `x` geeft `np.quantile(x, 0.75)` de waarde 40. Welk deel van de getallen is 40 of minder?",
        options={
            "three": "Ongeveer 3 van elke 4",
            "one": "Ongeveer 1 van elke 4",
            "forty": "Ongeveer 40%",
        },
        correct="three",
        why={
            "three": "Het 0,75-kwantiel is de waarde waaronder 0,75 van de waarden ligt: 3 van elke 4 getallen zijn 40 of minder.",
            "one": "Dat zou het 0,25-kwantiel zijn. Welke fractie zegt `0.75`? Zie de nieuwe woorden in [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "forty": "40 is de waarde zelf, geen aandeel. Welk deel van de waarden ligt onder het 0,75-kwantiel? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
        },
    )
    quick_quantile
    return (quick_quantile,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een foto die in de mist is genomen, of van een bladzijde bij slecht licht, ziet er vlak en grauw uit: de donkerste delen zijn niet zwart, en de lichtste niet wit. De pixels gebruiken maar een smalle band van de 256 grijstinten, en de details zijn slecht te zien, voor jou en voor een programma. Histogramegalisatie spreidt die grijstinten over de hele schaal, van 0 tot en met 255, met één regel die alleen kijkt naar hoeveel pixels elke grijstint heeft. Het is een van de oudste gereedschappen van de beeldbewerking: scikit-image en OpenCV hebben het allebei, en een variant ervan wordt vaak gebruikt op röntgenfoto's en microscoopbeelden.

    **Een voorbeeld uit het dagelijks leven.** Twintig hardlopers komen binnen één minuut na elkaar over de finish. Geef je ze hun plaats, van 1 tot en met 20, in plaats van hun tijd, dan worden de kleine verschillen grote. Egalisatie doet hetzelfde met grijstinten: elke grijstint krijgt zijn plaats in de rij van donker naar licht.
    """)
    return


@app.cell(hide_code=True)
def predict_grays():
    guess_grays = L.guess(
        "grays",
        question="Het uitgewerkte voorbeeld van deze les maakt een foto expres dof: daarna gebruikt hij 65 van de 256 grijstinten. Hoeveel verschillende grijstinten gebruikt de foto na histogramegalisatie?",
        options={
            "all": "Alle 256",
            "same": "65, net als eerst",
            "fewer": "Minder dan 65",
        },
        reveal="**Minder dan 65**: hij gebruikt er 49. Egalisatie geeft elke grijstint een nieuwe plek op de schaal, maar alle pixels met dezelfde grijstint krijgen dezelfde nieuwe grijstint, dus er kunnen nooit meer grijstinten komen dan er waren. Een paar grijstinten naast elkaar die weinig pixels hebben, kunnen zelfs samen één worden. De 49 grijstinten liggen nu verspreid van 0 tot en met 255, met gaten ertussen.",
    )
    guess_grays
    return (guess_grays,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Het contrast van een foto is hoe ver zijn grijstinten uit elkaar liggen: een doffe foto heeft weinig contrast. Zijn histogram laat dat zien: een smalle band staven, en links en rechts ervan niets. Histogramegalisatie geeft elke grijstint een nieuwe grijstint met één regel: neem het aandeel van de pixels met deze grijstint of een donkerdere, en vermenigvuldig dat met 255. Een grijstint met aandeel 0,02 wordt 5; een grijstint waarbij de helft van de pixels die grijstint of een donkerdere heeft, wordt ongeveer 128, vlak bij het midden van de schaal; de lichtste grijstint wordt 255. Waar veel pixels bijna dezelfde grijstint hebben, stijgt het aandeel snel van de ene grijstint naar de volgende, dus die grijstinten komen ver uit elkaar te liggen. Waar weinig pixels zijn, verandert het aandeel bijna niet, en schuiven die grijstinten naar elkaar toe. De regel hoeft niet te weten hoe de foto dof is geworden: het histogram is genoeg.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **contrast** | hoe ver de grijstinten van een afbeelding uit elkaar liggen; een doffe afbeelding heeft weinig contrast | één maat: `img.std()` |
    | **cumulatief histogram** | voor elke grijstint hoeveel pixels die grijstint of een donkerdere hebben; gedeeld door het aantal pixels is het een aandeel van 0 tot en met 1 | `np.cumsum(counts)` |
    | **histogramegalisatie** | elke grijstint krijgt de nieuwe grijstint 255 × zijn aandeel in het cumulatieve histogram, zodat de grijstinten zich over de hele schaal spreiden | `table[img]` |
    | **`np.cumsum(x)`** | het lopende totaal: elk getal plus alle getallen ervoor | `np.cumsum([2, 0, 3, 1])` geeft `array([2, 2, 5, 6])` |
    | **`table[img]`** | een array met gehele getallen tussen vierkante haken: voor elk getal erin het element van `table` op die plek, in dezelfde shape | `t[np.array([2, 0, 0])]` geeft `array([30, 10, 10])` als `t` gelijk is aan `np.array([10, 20, 30])` |

    De nieuwe grijstinten van alle 256 grijstinten gaan in één array, een tabel: `table[g]` is de nieuwe grijstint van grijstint `g`. `table[img]` zoekt dan elke pixel van de afbeelding in één keer op, en geeft een afbeelding met dezelfde shape.
    """)
    return


@app.cell(hide_code=True)
def picture_before_after():
    mo.vstack([
        mo.md("**Een doffe foto, voor en na egalisatie**"),
        mo.Html(C.before_after({"dull": "Dof", "after": "Na egalisatie", "histogram": "Histogram",
                                "cumulative": "Cumulatief histogram", "gray": "grijstint",
                                "alt": "Twee rijen van drie plaatjes. Bovenaan de doffe foto van een kat, helemaal in middengrijs; zijn histogram is een smalle band staven tussen 101 en 169; zijn cumulatieve histogram is 0 onder grijstint 101, stijgt steil rond grijstint 145 en bereikt 1 bij grijstint 169. Onderaan de foto na egalisatie, met zwart en wit erin; zijn histogram heeft staven verspreid van 0 tot en met 255, met gaten ertussen; zijn cumulatieve histogram ligt dicht bij de rechte lijn van 0 naar 1."})),
        mo.md(r"""
        Bovenaan: de doffe foto; zijn histogram is een smalle band tussen grijstint 101 en grijstint 169, en zijn cumulatieve histogram stijgt binnen die band van 0 naar 1. Onderaan: dezelfde foto na egalisatie; zijn staven liggen verspreid over de hele schaal, met gaten, en zijn cumulatieve histogram ligt dicht bij de gestippelde rechte lijn. Elk kwart van de schaal bevat nu ongeveer een kwart van de pixels.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`plt.imshow` met `cmap=\"gray\"` rekt de grijstinten zelf uit: hij laat de donkerste grijstint van de array zien als zwart, en de lichtste als wit. Een doffe foto ziet er dan niet dof uit. Wil je de grijstinten zien zoals ze zijn, zet er dan `vmin=0, vmax=255` bij."),
        L.watch_out("Egalisatie maakt geen nieuwe grijstinten: alle pixels met dezelfde grijstint krijgen dezelfde nieuwe grijstint. Het histogram heeft daarna evenveel staven als eerst, of minder, met gaten ertussen."),
        mo.md(r"""
        ### Recept

        Zo egaliseer je een grijswaardenbeeld `img` van uint8:

        1. Tel de pixels van elke grijstint: `counts = np.array([(img == g).sum() for g in range(256)])`. Dat is het histogram als getallen, één voor elke grijstint van 0 tot en met 255.
        2. Tel ze op van donker naar licht en deel door het aantal pixels: `share = np.cumsum(counts) / counts.sum()`. Dat is het cumulatieve histogram als aandelen; de lichtste grijstint heeft aandeel 1.
        3. Maak de tabel met de nieuwe grijstint van elke grijstint: `table = (255 * share).astype(np.uint8)`. `astype` hakt de decimalen eraf: 212,5 wordt 212.
        4. Zoek elke pixel op in de tabel: `new = table[img]`.
        5. Bekijk het resultaat met `plt.imshow(new, cmap="gray", vmin=0, vmax=255)`, en vergelijk het contrast: `img.std()` en `new.std()`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De foto is Chelsea de kat uit [Van licht tot een bestand met getallen](/course/open/cv-image-files?lang=nl): `data/photos/chelsea.png`. Het voorbeeld maakt hem eerst grijs, met de helderheid van HSV van elke pixel: `img.max(axis=2)`, de sterkste van rood, groen en blauw; `axis=2` is de derde richting van de array, de drie kleuren. Die grijze foto heeft een goed contrast, dus daarna maakt het voorbeeld hem dof, zoals mist dat zou doen: elke grijstint g wordt 100 + 0,3 g. Egalisatie gebruikt die formule niet en kijkt alleen naar het histogram. Voer de twee cellen uit.
    """)
    return


@app.cell
def example_count():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/chelsea.png"))
    # een foto in grijstinten: de helderheid van HSV van elke pixel
    photo = img.max(axis=2)
    # mist, met opzet: elke grijstint g wordt 100 + 0,3 g
    dull = (100 + 0.3 * photo).astype(np.uint8)
    # 1. de pixels van elke grijstint, en hoeveel grijstinten de foto gebruikt
    counts = np.array([(dull == g).sum() for g in range(256)])
    print(dull.min(), dull.max(), (counts > 0).sum())
    # 2. het cumulatieve histogram als aandelen
    share = np.cumsum(counts) / counts.sum()
    print(share[130], share[145], share[169])
    return dull, np, plt, share


@app.cell
def example_equalize(dull, np, plt, share):
    # 3. de tabel: de nieuwe grijstint van elke grijstint
    table = (255 * share).astype(np.uint8)
    print(table[101], table[145], table[169])
    # 4. elke pixel opgezocht in de tabel
    new = table[dull]
    # 5. het contrast voor en na, en de foto na
    print(dull.std(), new.std())
    plt.imshow(new, cmap="gray", vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De grijstinten van de doffe foto liggen tussen 101 en 169, en hij gebruikt er 65 van de 256: een paar lichte grijstinten, 165 tot en met 168, komen niet voor. Grijstint 130 heeft aandeel 0,081: maar 8% van de pixels heeft die grijstint of een donkerdere, de ogen, de neus en de donkere strepen op de kop. Grijstint 145 heeft aandeel 0,521, dus net iets meer dan de helft van de pixels heeft die grijstint of een donkerdere, en de lichtste grijstint, 169, heeft aandeel 1. De tabel maakt van 101, de donkerste grijstint, een 0, van grijstint 145 een 132, en van 169 een 255. De standaardafwijking van de grijstinten, één maat voor het contrast, gaat van ongeveer 10 naar ongeveer 74. Onder de cel is de kat weer duidelijk, met donkerdere strepen en lichtere vacht dan in de grijze foto van voor de mist.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_snow():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Sneeuw
    """)
    return


@app.cell(hide_code=True)
def choice_snow_cell():
    choice_snow = L.choice(
        "snow",
        question="Een foto van sneeuw: 9 van elke 10 pixels zijn lichte grijstinten van 200 tot en met 250, de sneeuw, en de andere pixels zijn donkere grijstinten van 10 tot en met 40, een paar bomen. Welke grijstinten krijgt de sneeuw na histogramegalisatie?",
        options={
            "most": "Alles boven ongeveer 25: negen tiende van de schaal",
            "same": "Nog steeds 200 tot en met 250: egalisatie houdt lichte grijstinten licht",
            "half": "Van 128 tot en met 255: de lichte helft van de schaal",
        },
        correct="most",
        why={
            "most": "De bomen zijn 1 van elke 10 pixels: zij nemen de aandelen tot 0,1, dus de nieuwe grijstinten tot ongeveer 25. De sneeuw neemt de aandelen van 0,1 tot en met 1: de nieuwe grijstinten van net boven 25 tot en met 255, negen tiende van de schaal. Kleine verschillen in de sneeuw worden goed zichtbaar, en de bomen komen allemaal tussen 0 en 25 terecht.",
            "same": "Egalisatie houdt de grijstint zelf niet: ze kijkt naar het aandeel van de pixels met deze grijstint of een donkerdere. Wat is dat aandeel voor de donkerste grijstint van de sneeuw?",
            "half": "De sneeuw blijft wel lichter dan de bomen. Maar hoe ver naar beneden gaat hij? De bomen zijn maar 1 van elke 10 pixels: welk aandeel heeft de donkerste grijstint van de sneeuw, en wat is 255 keer dat?",
        },
    )
    choice_snow
    return (choice_snow,)


@app.cell(hide_code=True)
def text_exercise_cumulative():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het cumulatieve histogram

        Maak `cumulative(counts)` af. Hij krijgt het aantal pixels van elke grijstint, op volgorde van donker naar licht, en geeft voor elke grijstint het aandeel van de pixels met die grijstint of een donkerdere terug. Vervang de `...`. De check roept hem aan met de aantallen van een afbeelding van 6 pixels: 2 met grijstint 0, geen met grijstint 1, 3 met grijstint 2 en 1 met grijstint 3.

        ```python
        cumulative([2, 0, 3, 1])
        ```

        Dat moet 1/3, 1/3, 5/6 en 1 geven; de check laat ze zien met zes decimalen, als `[0.333333, 0.333333, 0.833333, 1]`. Hij probeert ook andere lijsten met aantallen.
        """),
        L.task("Laat `cumulative(counts)` voor elke grijstint het aandeel geven van de pixels met die grijstint of een donkerdere."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cumulative():
    box_cumulative = L.code_box("cumulative", lines=7)
    box_cumulative
    return (box_cumulative,)


@app.cell(hide_code=True)
def check_cumulative(box_cumulative):
    result_cumulative = L.check("cumulative", box_cumulative.value)
    result_cumulative
    return (result_cumulative,)


@app.cell(hide_code=True)
def hints_cumulative():
    L.hints(
        "cumulative",
        [
            "Twee stappen: eerst het lopende totaal van `counts`, dan deel je dat door het aantal pixels.",
            "Het lopende totaal is `np.cumsum(counts)`. Hoeveel pixels zijn er in totaal? Dat is de som van de aantallen.",
            "`return np.cumsum(counts) / ___`, met op de open plek het aantal pixels.",
        ],
        why="`np.cumsum([2, 0, 3, 1])` is `[2, 2, 5, 6]`: 2 pixels hebben grijstint 0 of donkerder, 2 hebben grijstint 1 of donkerder, 5 hebben grijstint 2 of donkerder, en alle 6 hebben grijstint 3 of donkerder. Gedeeld door de 6 pixels is dat 1/3, 1/3, 5/6 en 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_middle():
    mo.md(r"""
    ### Oefening 3. Het midden van de schaal

    De slider kiest een grijstint van de doffe foto uit het uitgewerkte voorbeeld. Het plaatje laat zijn cumulatieve histogram zien, met die grijstint gemarkeerd: links het aandeel van de pixels met die grijstint of een donkerdere, rechts hetzelfde aandeel keer 255, de nieuwe grijstint.
    """)
    return


@app.cell(hide_code=True)
def explore_middle_cell():
    def _frame(g):
        _html, _f = C.gray_frame(g, {"gray": "grijstint van de doffe foto", "share": "aandeel", "new": "nieuwe grijstint",
                                     "alt": "Het cumulatieve histogram van de doffe foto, van grijstint 96 tot 174, met grijstint {g} gemarkeerd met een stip."})
        return (_html,
                f'Grijstint {g}: {L.number(100 * _f["share"], 1)}% van de pixels heeft deze grijstint of een donkerdere, dus hij wordt grijstint {_f["new"]}.')

    explore_middle = L.explore("middle", "grijstint", _frame, start=130, values=C.GRAYS, role="",
                               task="Zoek de donkerste grijstint van de doffe foto die grijstint 128 of lichter wordt, ongeveer het midden van de schaal.")
    explore_middle
    return (explore_middle,)


@app.cell(hide_code=True)
def check_middle(explore_middle):
    result_middle = L.check("middle", explore_middle)
    result_middle
    return (result_middle,)


@app.cell(hide_code=True)
def hints_middle():
    L.hints(
        "middle",
        [
            "Schuif de slider één stap tegelijk, en lees de nieuwe grijstint in de zin onder het plaatje.",
            "De nieuwe grijstint is 255 keer het aandeel. Welk aandeel geeft ongeveer 128, halverwege 0 en 255?",
            "Net iets meer dan de helft: zoek de eerste grijstint waar het aandeel voorbij 50% gaat.",
        ],
        why="Grijstint 144 heeft aandeel 0,467 en wordt 119; grijstint 145 heeft aandeel 0,521 en wordt 132. Dus 145 is de eerste grijstint waarbij de helft van de pixels die grijstint of een donkerdere heeft: de mediaan van de doffe foto, `np.median(dull)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_equalize():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. Egaliseren

        Maak `equalize(img)` af. Hij krijgt een grijswaardenbeeld met gehele getallen van 0 tot en met 255, en geeft de afbeelding na histogramegalisatie terug, als uint8. De eerste regels tellen al de pixels van elke grijstint, in `counts`. Vervang de `...`: het aandeel van elke grijstint, de tabel met nieuwe grijstinten, en elke pixel opgezocht in de tabel. De check roept hem aan met een afbeelding van 6 pixels:

        ```python
        equalize([[100, 100, 120], [120, 120, 200]])
        ```

        Dat moet `[[85, 85, 212], [212, 212, 255]]` geven. Hij probeert ook andere afbeeldingen.
        """),
        L.task("Laat `equalize(img)` de afbeelding na histogramegalisatie teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_equalize():
    box_equalize = L.code_box("equalize", lines=9)
    box_equalize
    return (box_equalize,)


@app.cell(hide_code=True)
def check_equalize(box_equalize):
    result_equalize = L.check("equalize", box_equalize.value)
    result_equalize
    return (result_equalize,)


@app.cell(hide_code=True)
def hints_equalize():
    L.hints(
        "equalize",
        [
            "Volg stap 2, 3 en 4 van het recept: het aandeel, de tabel, en dan elke pixel opzoeken.",
            "Het aandeel is het lopende totaal van `counts` gedeeld door het aantal pixels, zoals in oefening 2. De tabel is 255 keer dat, als uint8.",
            "`share = np.cumsum(counts) / counts.sum()`, dan `table = (___ * share).astype(np.uint8)`, dan `return table[___]`.",
        ],
        why="De 6 pixels zijn 2 keer grijstint 100, 3 keer grijstint 120 en 1 keer grijstint 200. De aandelen zijn 2/6, 5/6 en 6/6, en 255 keer dat is 85, 212,5 en 255; `astype` hakt 212,5 af tot 212. `table[img]` geeft elke pixel de nieuwe grijstint van zijn grijstint.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) was een kwantiel de waarde waaronder een bepaald deel van de waarden ligt; het cumulatieve histogram werkt andersom, en zegt voor elke grijstint welk deel van de pixels erop of eronder ligt. De mediaan van de doffe foto kwam vlak bij het midden van de schaal. In [Een kleur anders opschrijven: HSV](/course/open/cv-color-spaces?lang=nl) zei de helderheid hoe helder een pixel is: om een kleurenfoto te egaliseren, kun je alleen zijn helderheid egaliseren, dan blijven de kleuren hetzelfde. In de volgende les, *Zwart of wit: een drempel en de methode van Otsu*, zegt het histogram zelf waar je de donkere pixels van de lichte scheidt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_snow, result_cumulative, result_equalize, result_middle):
    L.wrap_up(choice_snow, result_cumulative, result_middle, result_equalize)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
