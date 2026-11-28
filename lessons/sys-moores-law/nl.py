# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-moores-law. Route: course/route/lessen/systemen.yaml. Checks, the chips and trainings, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-moores-law", "nl")
    C = L.checks   # de transistors van vijftien chips, de compute van vier trainingen, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl) en [Waarom trainen miljoenen kost](/course/open/sys-why-training-costs-millions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Een grafiek heeft een logaritmische schaal naar boven, met streepjes bij 1, 10, 100 en 1.000, op gelijke afstand. Een rechte lijn daarop stijgt elke 5 jaar één streepje. Wat zegt dat?",
        options={
            "add": "De waarde groeit elke 5 jaar met 10",
            "right": "De waarde wordt elke 5 jaar 10 keer zo groot",
            "double": "De waarde verdubbelt elke 5 jaar",
            "slow": "De groei vertraagt, want elk streepje is meer waard dan het streepje eronder",
        },
        correct="right",
        why={
            "right": "Elk streepje is tien keer het streepje eronder, dus is een rechte lijn op een logaritmische schaal dezelfde factor in dezelfde tijd. In deze les liggen de transistors op een chip vijftig jaar lang op zo'n lijn.",
            "add": "Elke 5 jaar 10 erbij is een rechte lijn op een gewone schaal. Hoeveel is elk streepje waard, vergeleken met het streepje eronder? Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "double": "Bij verdubbelen zou elk streepje twee keer het streepje eronder zijn. Hoeveel is elk streepje hier waard? Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "slow": "De lijn stijgt elke 5 jaar één streepje, en elk streepje is dezelfde factor hoger. Verandert de factor? Zie [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def quick_check_gpus():
    quick_gpus = L.choice(
        "quick-gpus",
        question="Een training heeft 1.000.000 GPU-uren nodig, tegen 2 dollar per GPU-uur. Je draait de training op twee keer zoveel GPU's van hetzelfde type, tegen dezelfde prijs. Wat gebeurt er met de rekening?",
        options={
            "double": "De rekening verdubbelt naar 4.000.000 dollar",
            "right": "De rekening blijft 2.000.000 dollar, en de run duurt half zo lang",
            "half": "De rekening halveert naar 1.000.000 dollar, want de run duurt half zo lang",
            "hours": "De rekening verdubbelt, want twee keer zoveel GPU's gebruiken twee keer de GPU-uren",
        },
        correct="right",
        why={
            "right": "De rekening is de GPU-uren keer de prijs, en het werk heeft 1.000.000 GPU-uren nodig, hoeveel GPU's het werk ook delen. In deze les zie je waarom elke nieuwe generatie GPU's meer werk per uur doet, zodat hetzelfde werk minder GPU-uren nodig heeft.",
            "double": "Twee keer zoveel GPU's voor half zo lang geven dezelfde GPU-uren. Wat telt de rekening? Zie [Waarom trainen miljoenen kost](/course/open/sys-why-training-costs-millions?lang=nl).",
            "half": "Half zo lang, op twee keer zoveel GPU's: de GPU-uren blijven 1.000.000. Wat telt de rekening? Zie [Waarom trainen miljoenen kost](/course/open/sys-why-training-costs-millions?lang=nl).",
            "hours": "Elke GPU draait half zo lang, dus gebruiken twee keer zoveel GPU's dezelfde 1.000.000 GPU-uren. Wat geven GPU's keer uren? Zie [Waarom trainen miljoenen kost](/course/open/sys-why-training-costs-millions?lang=nl).",
        },
    )
    quick_gpus
    return (quick_gpus,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 1971 had de eerste processor van Intel, de 4004, 2.300 transistors. De H100 van NVIDIA uit 2022 heeft er 80 miljard: 35 miljoen keer zoveel, 51 jaar later. Door deze groei doet een uur van een nieuwe GPU meer werk dan een uur van een oude, en duren trainingen waar twintig jaar geleden geen computer aan kon beginnen nu maanden. De compute van de grootste trainingen groeide nog sneller. In deze les meet je zulke groei met één getal, de verdubbelingstijd, en zie je waar de groei vertraagde en waar de groei tegen een muur liep.

    **Een voorbeeld uit het dagelijks leven.** Vouw een vel papier van 0,1 millimeter dubbel, en het vel is twee keer zo dik. Na 10 keer vouwen is het vel ongeveer 10 centimeter dik; na 20 keer zou het vel ongeveer 105 meter dik zijn, als je het nog kon vouwen. Elke vouw voegt evenveel toe als alle eerdere vouwen samen: zo werkt verdubbelen.
    """)
    return


@app.cell(hide_code=True)
def predict_times():
    guess_times = L.guess(
        "times",
        question="OpenAI trainde GPT-3 in 2020 met 3,14 × 10²³ FLOPs. Meta trainde Llama 3.1 405B in 2024. Hoeveel keer zoveel compute kostte de training van Llama 3.1 405B, denk je?",
        options={
            "chip": "Ongeveer 4 keer: zo snel als de transistors op een chip groeien",
            "twelve": "Ongeveer 12 keer",
            "right": "Ongeveer 120 keer",
            "thousand": "Ongeveer 1.000 keer",
        },
        reveal="**Ongeveer 120 keer.** Meta meldt 3,8 × 10²⁵ FLOPs, 121 keer de 3,14 × 10²³ van GPT-3: ongeveer 7 verdubbelingen in 4 jaar, één per 7 maanden. De transistors op een chip verdubbelen ongeveer elke twee jaar, dus zijn er na 4 jaar 4 keer zoveel. De rest kwam van meer GPU's die langer werkten: Meta trainde het 405B-model op maximaal 16.384 H100-GPU's.",
    )
    guess_times
    return (guess_times,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De wet van Moore.** In 1965 zag Gordon Moore, later een van de oprichters van Intel, dat het aantal onderdelen op een chip elk jaar was verdubbeld. Moore voorspelde dat de verdubbeling zou doorgaan, en maakte er in 1975 elke twee jaar van. De **wet van Moore** is deze voorspelling: het aantal transistors op een chip verdubbelt ongeveer elke twee jaar. Het is geen natuurwet, zoals de zwaartekracht. Het is een trend die chipmakers vijftig jaar volhielden, door transistors steeds kleiner te maken.

        Op een logaritmische schaal is een verdubbeling per twee jaar een rechte lijn. Vijftien chips, van de Intel 4004 uit 1971 tot de NVIDIA B200 uit 2024, liggen dicht bij zo'n lijn:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        De beste rechte lijn door de vijftien chips stijgt één verdubbeling per 2,04 jaar. De B200 zet twee chips in één behuizing, met samen 208 miljard transistors.

        **Verdubbelingstijd.** De **verdubbelingstijd** van een groei is de tijd die de groei nodig heeft om te verdubbelen. Tel van een begin tot een eind hoeveel verdubbelingen erin passen, en deel het aantal jaren door dat aantal. Het aantal verdubbelingen is de log2 van hoeveel keer zo groot het eind is. Van de 4004 in 1971 tot de Pentium 4 in 2000 groeide het aantal transistors 18.261 keer, en de log2 van 18.261 is 14,16: 14,16 verdubbelingen in 29 jaar, één per 2,05 jaar.

        **Dennard scaling.** In 1974 beschreven Robert Dennard en collega's bij IBM waarom kleinere transistors ook beter waren: een kleinere transistor heeft minder spanning nodig en schakelt sneller. Een chip vol kleinere transistors draait sneller bij dezelfde stroom per vierkante millimeter. Dat is **Dennard scaling**, en rond 2005 hield het op. Transistors werden zo klein dat er stroom weglekt, ook als een transistor uit staat, en een hogere kloksnelheid maakte chips te heet. In november 2004 bracht Intel een Pentium 4 van 3,8 GHz uit en schrapte Intel de versie van 4 GHz. Sindsdien groeide het aantal transistors door, maar gingen de transistors naar meer cores in plaats van naar een hogere kloksnelheid, en later naar GPU's: chips met duizenden cores die tegelijk rekenen.

        **De memory wall.** Een GPU rekent met getallen uit het geheugen. De bandbreedte van dat geheugen, hoeveel bytes per seconde het geheugen kan leveren, groeide langzamer dan het rekenen. Amir Gholami en anderen maten over de 20 jaar tot 2024: de hoogste FLOP/s van serverhardware groeiden elke twee jaar 3,0 keer, de bandbreedte van het geheugen 1,6 keer, en de bandbreedte van de verbindingen tussen chips 1,4 keer. Elke nieuwe GPU kan dus per FLOP minder bytes lezen, en veel berekeningen zijn langer bezig met wachten op data dan met rekenen. Dat is de **memory wall**. Een latere les laat zien hoe geheugen in elkaar zit, en waarom data dichtbij snel is.

        **De compute van training.** De grootste trainingen groeiden veel sneller dan de chips, omdat trainingen ook meer chips gebruiken, en langer. Epoch AI, een onderzoeksgroep, houdt een database bij met de compute van de training van opvallende modellen: van 2010 tot mei 2024 groeide de compute 4,1 keer per jaar, en de compute van de tien grootste op elk moment 5,3 keer per jaar. De pagina met trends van Epoch, bijgewerkt op 5 november 2026, zet de grootste trainingen van language models op 5 keer per jaar sinds 2020: één verdubbeling per 5,2 maanden. [Waarom trainen miljoenen kost](/course/open/sys-why-training-costs-millions?lang=nl) liet zien wat elk GPU-uur van zulke trainingen kost.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **wet van Moore** | | de voorspelling, uit 1975, dat het aantal transistors op een chip ongeveer elke twee jaar verdubbelt; een trend, geen natuurwet | |
        | **Dennard scaling** | | kleinere transistors hebben minder spanning nodig, dus wordt een chip sneller bij dezelfde stroom per vierkante millimeter; rond 2005 hield dat op | |
        | **memory wall** | | het rekenen van chips groeit sneller dan de bandbreedte van het geheugen, dus wacht het rekenen steeds vaker op data | |
        | **verdubbelingstijd** | | de tijd die een groei nodig heeft om te verdubbelen; na `t` jaar geeft een verdubbelingstijd `d` `2 ** (t / d)` keer zoveel | `2 ** (t / d)` |
        | `3.8e25` | "3 punt 8 e 25" | zo schrijft Python 3,8 × 10²⁵: 38 met 24 nullen erachter | `3.8e25` |
        """),
        L.watch_out("Meer transistors maken een chip niet vanzelf sneller. Sinds het eind van Dennard scaling gaan de extra transistors naar meer cores, en een programma wordt alleen sneller als het werk over de cores verdeeld wordt."),
        L.watch_out("Een rechte lijn op een logaritmische schaal nodigt uit om de lijn door te trekken. De wet van Moore hield vijftig jaar stand doordat chipmakers er hard aan werkten, en een trend zegt niet zeker wat de komende tien jaar brengen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De verdubbelingstijd van een groei:

    1. Neem de waarde aan het begin, de waarde aan het eind, en de jaren ertussen.
    2. Deel het eind door het begin: hoeveel keer zo groot is de waarde geworden?
    3. Neem daar de log2 van: het aantal verdubbelingen.
    4. Deel de jaren door het aantal verdubbelingen: de jaren per verdubbeling.
    5. Bij veel punten: pas een rechte lijn door de jaren en de log2 van de waarden. De helling is het aantal verdubbelingen per jaar, en één gedeeld door de helling de jaren per verdubbeling.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Met één verdubbeling per 2 jaar wordt een getal in `years` jaar `2 ** (years / 2)` keer zo groot. De cel begint bij de 2.300 transistors van de 4004 en voorspelt het aantal om de tien jaar, en in 2022, het jaar van de H100. Voer de cel uit.
    """)
    return


@app.cell
def example_moore():
    _start = 2300   # de transistors van de Intel 4004, 1971
    for _years in [0, 10, 20, 30, 40, 51]:
        print(1971 + _years, round(_start * 2 ** (_years / 2)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke tien jaar groeit het aantal 32 keer: vijf verdubbelingen. De voorspelling voor 2022 is 109 miljard, en de H100 uit dat jaar heeft 80 miljard, ongeveer drie kwart ervan, na meer dan 25 verdubbelingen. Op een logaritmische schaal is dat gat klein. Oefening 1 draait het om: uit twee aantallen en de jaren ertussen vindt de functie de verdubbelingstijd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_time():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De verdubbelingstijd

        `doubling_time(start, end, years)` krijgt een waarde aan het begin, een waarde aan het eind en de jaren ertussen, en geeft de jaren per verdubbeling terug. Vervang de `...` door de jaren gedeeld door het aantal verdubbelingen: de log2 van hoeveel keer zo groot `end` is als `start`. NumPy staat klaar als `np`.

        De check roept `doubling_time(2300, 80000000000, 51)` aan: de transistors van de 4004 en van de H100, 51 jaar uit elkaar. Daarna de compute van de training van GPT-3 tot Llama 3.1 405B in 4 jaar, en eigen kleine getallen.
        """),
        L.task("Geef de jaren per verdubbeling van start tot end terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_time():
    box_time = L.code_box("time", lines=3)
    box_time
    return (box_time,)


@app.cell(hide_code=True)
def check_time(box_time):
    result_time = L.check("time", box_time.value)
    result_time
    return (result_time,)


@app.cell(hide_code=True)
def hints_time():
    L.hints(
        "time",
        [
            "`end / start` zegt hoeveel keer zo groot het eind is, en `np.log2` van een getal telt de verdubbelingen erin. `years` gedeeld door het aantal verdubbelingen geeft de jaren per verdubbeling. Wat komt er na de `/`?",
            "`np.log2(1024)` geeft `10.0`: 1.024 is tien verdubbelingen van 1. Van 1 tot 1.024 in 20 jaar is één verdubbeling per 2 jaar.",
            "`    return years / ___`",
        ],
        why="`return years / np.log2(end / start)` deelt de jaren door het aantal verdubbelingen. Van de 4004 tot de H100: 34,8 miljoen keer zoveel transistors, 25,05 verdubbelingen in 51 jaar, één per 2,04 jaar.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een lijn door alle punten

        Twee chips geven een verdubbelingstijd, en vijftien chips geven veertien verdubbelingstijden met kleine verschillen, één voor elk paar chips naast elkaar. Een rechte lijn door alle punten geeft er één. Op een schaal van log2 is de helling het aantal verdubbelingen per jaar, en één gedeeld door de helling de jaren per verdubbeling.

        `np.polyfit(x, y, 1)` past de beste rechte lijn door punten en geeft twee getallen terug: eerst de helling, dan de bias term. `fit_doubling(years, counts)` krijgt twee NumPy-arrays. Vervang de `...` zodat `slope` de helling bevat van de rechte lijn door `years` en de log2 van `counts`; de functie geeft dan `1 / slope` terug.

        De check roept `fit_doubling(C.YEARS, C.COUNTS)` aan: de vijftien chips van de afbeelding. Daarna `C.TRAIN_YEARS` en `C.TRAIN_FLOP`, de compute van de training van BERT-Large, T5-11B, GPT-3 en Llama 3.1 405B, en eigen kleine lijsten.
        """),
        L.task("Maak slope de helling van de lijn door de jaren en de log2 van de aantallen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fit():
    box_fit = L.code_box("fit", lines=4)
    box_fit
    return (box_fit,)


@app.cell(hide_code=True)
def check_fit(box_fit):
    result_fit = L.check("fit", box_fit.value)
    result_fit
    return (result_fit,)


@app.cell(hide_code=True)
def hints_fit():
    L.hints(
        "fit",
        [
            "`np.polyfit` krijgt wat horizontaal gaat, wat verticaal gaat, en `1` voor een rechte lijn, en `[0]` haalt de helling uit wat `np.polyfit` teruggeeft. Horizontaal gaan de `years`. Verticaal gaat de log2 van elk aantal, en `np.log2` werkt in één keer op een hele array. Wat gaat er verticaal, zodat `slope` verdubbelingen telt?",
            "Aantallen 1, 2 en 4 in de jaren 0, 1 en 2 hebben de log2 0, 1 en 2: een lijn met helling 1, één verdubbeling per jaar.",
            "`    slope = np.polyfit(years, ___, 1)[0]`",
        ],
        why="`slope = np.polyfit(years, np.log2(counts), 1)[0]` past de lijn door de jaren en de verdubbelingen. De vijftien chips stijgen 0,491 verdubbelingen per jaar: één per 2,04 jaar. De vier trainingen stijgen 2,44 verdubbelingen per jaar: één per 4,9 maanden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_laws():
    mo.md(r"""
    ### Oefening 3. Drie ideeën

    Acht feiten uit deze les. Zet elk feit onder het idee waar het feit bij hoort: *De wet van Moore*, *Dennard scaling en het eind daarvan* of *De memory wall*.
    """)
    return


@app.cell(hide_code=True)
def sort_laws_cell():
    sort_laws = L.sort("laws")
    sort_laws
    return (sort_laws,)


@app.cell(hide_code=True)
def check_laws(sort_laws):
    result_laws = L.check("laws", sort_laws)
    result_laws
    return (result_laws,)


@app.cell(hide_code=True)
def hints_laws():
    L.hints(
        "laws",
        [
            "De wet van Moore telt transistors. Dennard scaling gaat over stroom en kloksnelheid. De memory wall gaat over rekenen tegenover geheugen.",
            "Twee feiten horen bij de wet van Moore, drie bij Dennard scaling en het eind daarvan, en drie bij de memory wall.",
            "De wet van Moore: 1965 en 1975, \\_\\_\\_. Dennard scaling en het eind daarvan: kleiner, zelfde stroom, \\_\\_\\_, \\_\\_\\_. De memory wall: 3,0 tegen 1,6, \\_\\_\\_, \\_\\_\\_.",
        ],
        why="**De wet van Moore:** 1965 en 1975, 2.300 en 80 miljard. **Dennard scaling en het eind daarvan:** kleiner, zelfde stroom, 3,8 GHz, meer cores. **De memory wall:** 3,0 tegen 1,6, wachten op data, minder bytes per FLOP.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_wall():
    mo.md(r"""
    ### Oefening 4. Twintig jaar memory wall

    Over 20 jaar groeiden de hoogste FLOP/s van een server elke twee jaar 3,0 keer, en de bandbreedte van het geheugen elke twee jaar 1,6 keer.
    """)
    return


@app.cell(hide_code=True)
def choice_wall_cell():
    choice_wall = L.choice(
        "wall",
        question="Hoeveel keer zoveel FLOPs kan een server na deze 20 jaar doen per byte die de server uit het geheugen leest?",
        options={
            "ratio": "Ongeveer 1,9 keer: 3,0 gedeeld door 1,6",
            "fourteen": "Ongeveer 14 keer",
            "right": "Ongeveer 540 keer",
            "flops": "Ongeveer 59.000 keer, want 3,0 tot de macht 10 is 59.049 en het geheugen telt niet mee",
        },
        correct="right",
        why={
            "right": "20 jaar zijn tien perioden van twee jaar. De FLOP/s groeien 3,0¹⁰ = 59.049 keer, de bandbreedte 1,6¹⁰ = 110 keer, en 59.049 gedeeld door 110 is ongeveer 540. Elke byte uit het geheugen moet 540 keer zoveel FLOPs voeden, dus wacht een GPU van nu veel vaker op het geheugen dan een GPU van 20 jaar geleden.",
            "ratio": "1,9 is het gat na één periode van twee jaar. Hoeveel perioden passen er in 20 jaar, en wat doet elke periode met het gat?",
            "fourteen": "De groei vermenigvuldigt, en telt niet op. Wat geven tien perioden van 3,0 en tien van 1,6?",
            "flops": "59.049 is alleen de groei van de FLOP/s, en het geheugen groeide ook, 1,6 keer per twee jaar. Hoeveel van de 59.049 blijft er per byte over?",
        },
    )
    choice_wall
    return (choice_wall,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Vijf soorten groei

    Hoeveel keer elke soort groei is vermenigvuldigd na een aantal jaren, op een logaritmische schaal: de compute van de grootste trainingen met 5,3 keer per jaar, de hoogste FLOP/s met 3,0 keer in twee jaar, transistors met 2 keer, de bandbreedte van het geheugen met 1,6 keer, en de bandbreedte tussen chips met 1,4 keer. Kies een aantal jaren.
    """)
    return


@app.cell(hide_code=True)
def demo_years():
    L.explore("years", "jaren", lambda _v: C.frame(_v, L.lang), start=4, values=C.DEMO_VALUES, role="",
              task="Ga van 0 naar 14 jaar: welke balk loopt weg, en wat gebeurt er met de FLOPs per gelezen byte?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Na 14 jaar zijn de grootste trainingen 13,8 miljard keer gegroeid, de hoogste FLOP/s 2.187 keer en de transistors 128 keer. De bandbreedte van het geheugen is 27 keer gegroeid, dus moet elke gelezen byte 81 keer zoveel FLOPs voeden. De trainingen groeien harder dan de chips door meer chips te gebruiken, en het gat tussen de bovenste balk en de transistors wordt betaald in GPU's, stroom en geld. De groeicijfers zijn gemiddelden over de jaren waarin de cijfers gemeten zijn, en de balken trekken de cijfers door: niemand weet of de groei nog 14 jaar zo doorgaat.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl) was groei met dezelfde factor per stap een rechte lijn op een logaritmische schaal; hier is zo'n lijn vijftig jaar chips, en de helling is een verdubbelingstijd. In [Waarom trainen miljoenen kost](/course/open/sys-why-training-costs-millions?lang=nl) kostte een training GPU-uren: de wet van Moore is de reden dat een uur van een nieuwere GPU meer werk doet, en de memory wall de reden dat de GPU vaak wacht. De volgende les is *Dichtbij en snel, ver weg en traag: de geheugenhiërarchie*: hoe geheugen in elkaar zit, en waarom data dichtbij snel is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_wall, result_fit, result_laws, result_time):
    L.wrap_up(result_time, result_fit, result_laws, choice_wall)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
