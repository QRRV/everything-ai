# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-case-compas-amazon. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-case-compas-amazon", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) en [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_harm():
    quick_harm = L.choice(
        "quick-harm",
        question="Het model van een bank zegt vaker nee tegen een lening voor mensen uit de ene wijk dan voor mensen met hetzelfde inkomen uit een andere. Welke soort schade is dat?",
        options={
            "representational": "Representatieve schade",
            "right": "Allocatieve schade",
            "none": "Geen schade, zolang de meeste antwoorden goed zijn",
        },
        correct="right",
        why={
            "representational": "Representatieve schade gaat over hoe een systeem groepen laat zien of beschrijft. Dit model beslist wie een lening krijgt. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Het model deelt leningen uit, en de ene groep krijgt er minder dan zou moeten: allocatieve schade.",
            "none": "Een model kan meestal gelijk hebben en toch vaker nee zeggen tegen één groep. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_harm
    return (quick_harm,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="Een spamfilter wordt getest op 100 mails. Welk deel is de precision?",
        options={
            "right": "Van de mails die het spam noemt, het deel dat spam is",
            "recall": "Van de mails die echt spam zijn, het deel dat het spam noemt",
            "accuracy": "Van alle mails, het deel dat het goed heeft",
        },
        correct="right",
        why={
            "right": "Precision is $TP / (TP + FP)$: van wat het filter positief noemt, het deel dat echt positief is.",
            "recall": "Dat is recall, $TP / (TP + FN)$. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "accuracy": "Dat is accuracy, die elke mail meetelt. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2016 en 2018 lieten twee onderzoeken zien wat je ontdekt als je goed naar de fouten van een model kijkt. Het ene ging over een risicoscore die Amerikaanse rechtbanken gebruiken, het andere over een model van Amazon dat sollicitanten beoordeelde. Het zijn nog steeds de voorbeelden die mensen als eerste noemen, en deze les bekijkt ze allebei, met de cijfers erbij.

    **Een voorbeeld uit het dagelijks leven.** Een winkeldetective controleert de tassen van klanten. Vraag je *welk deel van de controles iets oplevert*, dan lijkt de detective voor elke groep even eerlijk. Vraag je *welk deel van de klanten die niets hadden gestolen gecontroleerd werd*, dan kan de ene groep veel vaker zijn gecontroleerd. De twee vragen kunnen over dezelfde detective een ander antwoord geven.
    """)
    return


@app.cell(hide_code=True)
def predict_again():
    guess_again = L.guess(
        "again",
        question="ProPublica volgde 7.214 verdachten in Broward County, Florida, twee jaar lang. COMPAS had ongeveer 3.300 van hen een gemiddeld of hoog risico gegeven. Welk deel daarvan werd volgens jou binnen twee jaar opnieuw opgepakt?",
        options={
            "most": "Ongeveer 9 op 10",
            "right": "Ongeveer 6 op 10",
            "few": "Ongeveer 3 op 10",
        },
        reveal="**Ongeveer 6 op 10**: van de 3.317 mensen met een gemiddeld of hoog risico werden er 2.035 opnieuw opgepakt, 0,61. Ongeveer vier op de tien mensen die de score markeerde, werden dus niet opnieuw opgepakt.",
    )
    guess_again
    return (guess_again,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **COMPAS** is een programma van het bedrijf Northpointe, nu Equivant, dat een verdachte een **risicoscore** van 1 tot 10 geeft. Broward County in Florida gebruikt het vooral om te beslissen of iemand thuis op het proces mag wachten of vast blijft zitten. Scores 1 tot 4 gelden als laag, 5 tot 7 als gemiddeld en 8 tot 10 als hoog.

        In 2016 verzamelden journalisten van ProPublica de scores van de mensen die in 2013 en 2014 in Broward County een score kregen, en keken ze wie binnen twee jaar opnieuw werd opgepakt. Noem *gemiddeld of hoog risico* positief, of *gemarkeerd*, en elke groep krijgt een confusion matrix, zoals in de afbeelding. Dan geven twee vragen twee antwoorden.

        - *Welk deel van de mensen die de score markeerde, werd opnieuw opgepakt?* Dat is de precision van elke groep. Die is voor zwarte en witte verdachten bijna gelijk, zoals het uitgewerkte voorbeeld laat zien, en Northpointe zei dat de score daarom eerlijk is: een hoge score betekent in beide groepen hetzelfde.
        - *Welk deel van de mensen die niet opnieuw werden opgepakt, markeerde de score toch?* Dat reken je uit in Oefening 1. ProPublica vond bij zwarte verdachten een bijna twee keer zo groot deel, en noemde de score bevooroordeeld.

        Beide antwoorden kloppen tegelijk. In deze data werd 0,51 van de zwarte verdachten opnieuw opgepakt en 0,39 van de witte: opgepakt, niet bewezen schuldig aan een misdrijf, en wie er wordt opgepakt, hangt ook af van waar de politie zoekt. Als die delen verschillen, kan een score die fouten maakt niet in beide groepen dezelfde precision, hetzelfde deel onterechte markeringen en hetzelfde deel gemiste gevallen hebben; [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) laat zien waarom.

        **Amazon** begon in 2014 aan een model dat sollicitanten één tot vijf sterren gaf. Het leerde van patronen in de cv's die in tien jaar naar het bedrijf waren gestuurd, en de meeste daarvan kwamen van mannen: in feite, schreef Reuters, leerde het model zichzelf dat mannelijke kandidaten de voorkeur hadden. In 2015 had het team gezien dat het model cv's lager waardeerde als het Engelse woord *women's* erin stond, zoals in *women's chess club captain*, en ook mensen die van twee hogescholen alleen voor vrouwen kwamen. Het team liet het model die woorden negeren, maar kon niet zeker zijn dat het geen andere manier zou vinden om sollicitanten in te delen die discriminerend kon uitpakken, en begin 2017 werd het team opgeheven. Reuters bracht het in 2018 naar buiten; volgens Reuters keken recruiters wel naar het advies van het model, maar vertrouwden ze er nooit alleen op.

        In beide gevallen is de schade allocatief, of zou ze dat geweest zijn, over vrijheid en over werk, en in beide kwam ze uit de data en uit wat er gemeten werd, niet uit een bug in de code.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **risicoscore** | een getal dat een model aan een persoon of dossier geeft om aan te geven hoe groot de kans op iets ongewensts is, zoals een nieuwe arrestatie of een gemiste betaling; vaak bepaalt het wie als eerste gecontroleerd wordt |
        """),
        L.watch_out("*Eerlijk* heeft een maat nodig. Dezelfde scores zijn eerlijk volgens de ene vraag en oneerlijk volgens de andere. Noemt iemand een model eerlijk, vraag dan: volgens welke cijfers, en voor wie?"),
        mo.Html(C.picture({"title": "COMPAS in Broward County: een confusion matrix voor elke groep",
                           "groups": ["Zwarte verdachten", "Witte verdachten"],
                           "rows": ["niet opnieuw opgepakt", "opnieuw opgepakt"],
                           "columns": ["laag risico", "gemiddeld of hoog risico"],
                           "wrong_flag": "gemarkeerd, maar niet opnieuw opgepakt",
                           "missed": "laag risico, maar opnieuw opgepakt",
                           "thousands": ".",
                           "alt": "Twee confusion matrices. Zwarte verdachten: van wie niet opnieuw werd opgepakt, 990 met laag risico en 805 met gemiddeld of hoog risico; van wie opnieuw werd opgepakt, 532 met laag risico en 1.369 met gemiddeld of hoog risico. Witte verdachten: van wie niet opnieuw werd opgepakt, 1.139 met laag risico en 349 met gemiddeld of hoog risico; van wie opnieuw werd opgepakt, 461 met laag risico en 505 met gemiddeld of hoog risico."})),
        mo.md(r"""
        De aantallen die ProPublica publiceerde voor de 6.150 zwarte en witte verdachten. In elke matrix staan in het vakje rechtsboven de mensen die de score markeerde en die niet opnieuw werden opgepakt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo controleer je een risicoscore groep voor groep:

    1. Kies wat als positief telt: hier een gemiddeld of hoog risico.
    2. Tel de confusion matrix van elke groep.
    3. Reken per groep uit welk deel van de gemarkeerde mensen de uitkomst had: de precision.
    4. Reken per groep uit welk deel van de mensen zonder de uitkomst toch gemarkeerd werd.
    5. Vergelijk de groepen op allebei, en vraag je af wat het label echt meet.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 3 voor COMPAS. `_tp` geeft per groep hoeveel mensen de score markeerde die opnieuw werden opgepakt, `_fp` hoeveel mensen hij markeerde die dat niet werden. Voer de cel uit.
    """)
    return


@app.cell
def example_precision():
    _tp = {"black": 1369, "white": 505}
    _fp = {"black": 805, "white": 349}

    for _group in _tp:
        print(_group, _tp[_group] / (_tp[_group] + _fp[_group]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Van de zwarte verdachten die de score markeerde, werd 0,63 opnieuw opgepakt; van de witte verdachten 0,59. Bijna gelijk: dat was het antwoord van Northpointe aan ProPublica. Oefening 1 doet stap 4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flagged():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Gemarkeerd, maar niet opnieuw opgepakt

        Stap 4 van het recept. `fp` geeft per groep hoeveel mensen die niet opnieuw werden opgepakt toch een gemiddeld of hoog risico kregen, `tn` hoeveel van hen een laag risico kregen, en de loop gaat met `group` langs de groepen. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld, maar zet elk deel in `flagged` in plaats van het te printen. De laatste regel is `flagged`, zodat de check het ziet. De check probeert ook andere aantallen.
        """),
        L.task("Zet in `flagged`, onder elke groep, het deel van de mensen die niet opnieuw werden opgepakt dat de score een gemiddeld of hoog risico gaf."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flagged():
    box_flagged = L.code_box("flagged", lines=8)
    box_flagged
    return (box_flagged,)


@app.cell(hide_code=True)
def check_flagged(box_flagged):
    result_flagged = L.check("flagged", box_flagged.value)
    result_flagged
    return (result_flagged,)


@app.cell(hide_code=True)
def hints_flagged():
    L.hints(
        "flagged",
        [
            "Dezelfde vorm als in het uitgewerkte voorbeeld: een aantal gedeeld door zichzelf plus een ander aantal. Alleen welke aantallen je gebruikt, verandert, en het antwoord komt onder de groep.",
            "Boven de streep: de mensen die gemarkeerd werden. Eronder: alle mensen die niet opnieuw werden opgepakt, dus twee aantallen bij elkaar opgeteld. En hoe zet je een waarde onder een key in een dictionary?",
            "`flagged[group] = ___[group] / (___[group] + tn[group])`",
        ],
        why="`flagged[group] = fp[group] / (fp[group] + tn[group])` geeft ongeveer `{'black': 0.45, 'white': 0.23}`: van de zwarte verdachten die niet opnieuw werden opgepakt, werd bijna de helft gemarkeerd, tegen minder dan een kwart van de witte verdachten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sources():
    mo.md(r"""
    ### Oefening 2. Waar komt het probleem vandaan?

    Zes situaties: de cv's en de arrestatie komen uit de twee gevallen, de andere vier zijn verzonnen door de course. Zet elke situatie bij waar het probleem vandaan komt.
    """)
    return


@app.cell(hide_code=True)
def sort_sources_cell():
    sort_sources = L.sort("sources")
    sort_sources
    return (sort_sources,)


@app.cell(hide_code=True)
def check_sources(sort_sources):
    result_sources = L.check("sources", sort_sources)
    result_sources
    return (result_sources,)


@app.cell(hide_code=True)
def hints_sources():
    L.hints(
        "sources",
        [
            "Vraag bij elke situatie: gaat het over van wie het model leerde, over wat het moest voorspellen, of over wat mensen met het antwoord doen?",
            "Twee situaties gaan over mensen die in de data ontbreken, of die er het grootste deel van vormen. Twee andere gaan over wat mensen met de score doen als die er eenmaal is.",
            "Data: de cv's en \\_\\_\\_. Label: \\_\\_\\_ en de promotie. Gebruik: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Wie er in de data zit:** de cv's en de enquête. **Wat het label meet:** de arrestatie en de promotie. **Hoe de score gebruikt wordt:** de straf en de top vijf.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Oefening 3. Eerlijk volgens welke maat?
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="Een bedrijf zegt: *Onze fraudescore is eerlijk: in elke groep heeft hetzelfde deel van de mensen die hij markeert echt fraude gepleegd.* Wat laat die bewering open?",
        options={
            "meaning": "Of een markering in elke groep hetzelfde betekent",
            "total": "Hoeveel mensen de score per jaar bekijkt, in alle groepen samen",
            "right": "Hoe vaak mensen zonder fraude in elke groep gemarkeerd worden",
        },
        correct="right",
        why={
            "meaning": "Dat is precies wat de bewering zegt: hetzelfde deel van de gemarkeerden pleegde echt fraude, dus een markering betekent hetzelfde. Wat zegt ze niet? Zie *Het idee*.",
            "total": "Het aantal mensen telt mee voor de kosten, maar niet voor de vraag of de score de groepen gelijk behandelt. Zie *Het idee*.",
            "right": "De bewering gaat over precision. Net als bij COMPAS kunnen de mensen zonder fraude in één groep toch veel vaker gemarkeerd worden, dus vraag ook naar dat deel.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) zag je dat een systeem dat de ene groep meer van het slechte geeft, allocatieve schade doet; COMPAS en het model van Amazon zijn twee echte gevallen, over vrijheid en over werk. In [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) telde je één confusion matrix; hier vergeleek je er een voor elke groep, en zag je dat één getal per groep niet beslist of een score eerlijk is. In de volgende les, *De toeslagenaffaire en SyRI*, bekijk je twee Nederlandse gevallen waarin de overheid risicoscores gebruikte om fraude op te sporen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, result_flagged, result_sources):
    L.wrap_up(result_flagged, result_sources, choice_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
