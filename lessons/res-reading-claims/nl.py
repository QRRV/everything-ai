# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-reading-claims. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-reading-claims", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl) en [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_strong():
    quick_strong = L.choice(
        "quick-strong",
        question="Wat ondersteunt een resultaat het sterkst?",
        options={
            "reviewed": "Een paper erover die door peer review kwam",
            "press": "Een persbericht van het lab dat het vond, met citaten van het hoofd van het lab",
            "right": "Hetzelfde resultaat, opnieuw gevonden door andere onderzoekers",
        },
        correct="right",
        why={
            "reviewed": "Peer review helpt, maar reviewers voeren de experimenten zelden opnieuw uit. Zie [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).",
            "press": "Een persbericht is de eigen samenvatting van het lab, geen controle door iemand anders. Zie [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl).",
            "right": "Een resultaat dat in andere handen standhield, is het sterkst: dat is de laatste stap van preprint naar stevig resultaat.",
        },
    )
    quick_strong
    return (quick_strong,)


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="Een model haalt een accuracy van 0,85 op zijn test set. Altijd de meest voorkomende klasse voorspellen haalt 0,80, gemeten op de training set. Wat kun je zeggen?",
        options={
            "right": "Nog niets: meet de baseline op dezelfde test set",
            "beats": "Het model verslaat zijn baseline: 0,85 is hoger dan 0,80",
            "close": "Het model heeft weinig geleerd: 0,85 ligt dicht bij 0,80",
        },
        correct="right",
        why={
            "beats": "De twee getallen zijn op verschillende sets gemeten, dus je kunt ze niet vergelijken. Zie [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "close": "Of het verschil klein is, weet je pas als beide op dezelfde test set gemeten zijn. Zie [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "right": "Een vergelijking telt pas als beide op dezelfde test set gemeten zijn, met dezelfde metric.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    "Onze AI verslaat artsen." "Het nieuwe model begrijpt wiskunde." Zulke claims bereiken je via persberichten, nieuwsberichten en posts, en elke stap maakt ze korter en stelliger. Erachter zit meestal één getal op één test. Om te weten wat een claim waard is, ga je terug naar wat er echt gemeten is.

    **Een voorbeeld uit het dagelijks leven.** Een reclame zegt dat een tandpasta je tanden "tot 40% witter" maakt. De kleine lettertjes zeggen: in één test, met twaalf mensen, na acht weken, vergeleken met helemaal niet poetsen.
    """)
    return


@app.cell(hide_code=True)
def predict_new_set():
    guess_new_set = L.guess(
        "new-set",
        question="Veel language models scoren hoog op een bekende set verhaaltjessommen. Iemand schrijft een helemaal nieuwe set, met opgaven van dezelfde soort en even moeilijk. Hoe denk je dat de modellen op de nieuwe set scoren?",
        options={
            "same": "Ongeveer hetzelfde: de opgaven zijn even moeilijk",
            "higher": "Hoger: ze hebben op dit soort opgaven geoefend",
            "right": "Sommige duidelijk lager: ze kennen de oude vragen misschien",
        },
        reveal="**Sommige duidelijk lager.** Toen onderzoekers dit in 2024 deden met een populaire set verhaaltjessommen van de basisschool, scoorden meerdere modellen lager op de nieuwe opgaven, tot 8 procentpunten, terwijl andere nauwelijks zakten. De modellen die zakten, hadden de oude vragen, of vragen die er sterk op leken, waarschijnlijk tijdens het trainen gezien: de vragen stonden al jaren op internet. Een score op zo'n set laat zien hoe een model het doet op die vragen, niet op elke vraag van die soort.",
    )
    guess_new_set
    return (guess_new_set,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **benchmark** is een vaste set testtaken met een vaste manier van scoren, zodat verschillende modellen op dezelfde manier gemeten worden en hun scores te vergelijken zijn. De meeste claims over AI komen neer op een score op een benchmark, of op een eigen test set van de onderzoekers.

        Drie soorten tekst zeggen elk iets anders over hetzelfde resultaat:

        - **De benchmarkscore**: één getal, op die taken, op die manier gescoord. Dat is wat er gemeten is.
        - **De paper**: hoe de score gemeten is, op welke data, waartegen, en met welke beperkingen. De beperkingen staan vaak in een eigen stuk aan het eind.
        - **Het persbericht of nieuwsbericht**: wat het resultaat zou moeten betekenen. Het is het kortst en het stelligst van de drie, en het verst van de meting.

        Vraag dus bij elke claim: **wat is er precies gemeten, en waartegen?** Een score betekent pas iets naast een vergelijking: een baseline zoals in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl), eerdere modellen op dezelfde benchmark, of mensen onder dezelfde omstandigheden. En dan: **wat zegt de claim dat de meting niet laat zien?** In dat gat wordt een claim te groot.

        Een benchmark slijt ook. Zijn vragen komen op internet, en zo in de training data van latere modellen. Een model kan dan hoog scoren omdat het de vragen gezien heeft, niet omdat het de taak kan: dat heet contaminatie, en een latere les laat zien hoe makers van modellen dat proberen te voorkomen. Een nieuwe set vergelijkbare vragen laat het verschil zien.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **benchmark** | | een vaste set testtaken met een vaste manier van scoren, zodat modellen te vergelijken zijn | |
        """),
        L.watch_out("Een model met een hogere score op een benchmark is niet in het algemeen \"beter\": het is beter op die taken, op die manier gescoord. Of dat ook geldt voor waarvoor jij het model wilt gebruiken, zegt de benchmark niet."),
        mo.md(r"""
        De figuur: hoe een claim groeit van de meting naar de kop, met het voorbeeld uit het uitgewerkte voorbeeld hieronder.
        """),
        mo.Html(C.ladder({"title": "Van meting tot kop",
                          "rows": [("Gemeten", "86% goed op 200 foto's uit één ziekenhuis"),
                                   ("Paper", "hoger dan 16 huidartsen die alleen dezelfde foto's zagen"),
                                   ("Persbericht", "\"Onze AI stelt huidkanker beter vast dan artsen\"")],
                          "alt": "Drie vakken van onder naar boven. Gemeten: 86% goed op 200 foto's uit één ziekenhuis. Paper: hoger dan 16 huidartsen die alleen dezelfde foto's zagen. Persbericht: onze AI stelt huidkanker beter vast dan artsen. Elk vak is breder dan het vak eronder."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een claim over AI te controleren:

    1. Schrijf precies op wat er beweerd wordt, in één zin.
    2. Zoek wat er gemeten is: welke taak of benchmark, welke testdata, en welk getal.
    3. Zoek waarmee het vergeleken is: een baseline, eerdere modellen of mensen, en onder welke omstandigheden.
    4. Leg de claim naast de meting: wat zegt de claim dat de meting niet laat zien?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen voorbeeld, gebouwd zoals veel echte. Een persbericht zegt: **"Onze AI stelt huidkanker beter vast dan artsen."** De paper erachter zegt: het model beoordeelde 200 foto's van huidplekjes, allemaal uit één ziekenhuis, en had er 86% van goed. Op de helft van de foto's stond huidkanker, dus altijd "geen kanker" zeggen zou 50% goed hebben. Zestien huidartsen beoordeelden dezelfde 200 foto's, met niets anders om op af te gaan, en hadden gemiddeld 79% goed.

    1. **De claim:** de AI stelt huidkanker beter vast dan artsen.
    2. **Gemeten:** de accuracy op 200 foto's uit één ziekenhuis: 86% voor het model.
    3. **Vergeleken met:** de baseline van altijd "geen kanker" zeggen, 50%, en 16 huidartsen, gemiddeld 79%, die alleen de foto's zagen. In hun praktijk zien ze ook de patiënt, stellen ze vragen en kunnen ze de plek aanraken.
    4. **Claim naast meting:** "stelt vast" en "beter dan artsen" gaan verder dan "meer goede antwoorden op 200 foto's, tegen artsen die alleen de foto's hadden". Of het ook geldt voor foto's uit andere ziekenhuizen, van andere camera's of van andere huidskleuren, laat deze test niet zien.

    De meting is echt en interessant. De claim is groter dan de meting.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measured():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Gemeten, of meer?

        Zes uitspraken over een model dat op wiskundevragen getest is. Zet elke uitspraak in de goede groep.
        """),
    ])
    return


@app.cell(hide_code=True)
def exercise_measured():
    sort_measured = L.sort("measured")
    sort_measured
    return (sort_measured,)


@app.cell(hide_code=True)
def check_measured(sort_measured):
    result_measured = L.check("measured", sort_measured)
    result_measured
    return (result_measured,)


@app.cell(hide_code=True)
def hints_measured():
    L.hints(
        "measured",
        [
            "Vraag bij elke uitspraak: zegt ze wat er op een test gebeurde, of meer dan die test kon laten zien?",
            "Kijk terug naar stap 4 van het uitgewerkte voorbeeld: \"meer goede antwoorden op 200 foto's\" is gemeten; \"stelt vast\" en \"beter dan artsen\" gaan verder. Welke uitspraken lijken op het eerste, en welke op het tweede?",
            "Drie zijn metingen: een score op een set vragen, of een vergelijking van scores op dezelfde vragen. De andere drie zeggen wat het model begrijpt, wat het gaat doen, of hoe het het doet op vragen waarop het nooit getest is.",
        ],
        why="**Gemeten:** 92% van 1.000 testvragen goed; 4 punten hoger dan het beste model van vorig jaar op dezelfde benchmark; 9 punten lager op nieuwe vergelijkbare vragen dan op de benchmark. **Gaat verder:** begrijpt wiskunde; vervangt binnenkort wiskundeleraren; 92% van alle verhaaltjessommen van de basisschool, terwijl het op één set ervan getest is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_contamination():
    mo.md(r"""
    ### Oefening 2. Een score die omhoogsprong
    """)
    return


@app.cell(hide_code=True)
def choice_contamination_cell():
    choice_contamination = L.choice(
        "contamination",
        question="Een nieuw language model haalt 95% op een bekende benchmark, waar het beste model van vorig jaar 70% haalde. De vragen en antwoorden van de benchmark staan al jaren op internet. Welke vraag stel je eerst?",
        options={
            "size": "Hoeveel parameters het nieuwe model heeft",
            "right": "Of de vragen van de benchmark in zijn training data zaten",
            "press": "Of het persbericht het een doorbraak noemt",
        },
        correct="right",
        why={
            "size": "Grootte kan een beter model verklaren, maar niet of de score de taak meet. Zie contaminatie in *Het idee*.",
            "press": "Het persbericht staat het verst van de meting. Zie *Het idee*.",
            "right": "Als het model de vragen tijdens het trainen zag, laat de score geheugen zien, geen vaardigheid: contaminatie. Een nieuwe set vergelijkbare vragen zou het uitwijzen.",
        },
    )
    choice_contamination
    return (choice_contamination,)


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Oefening 3. Controleer zelf een claim

    Weer een verzonnen voorbeeld. Een nieuwsbericht: **"AI schrijft nu betere code dan programmeurs."** De paper erachter: op een benchmark van 164 korte programmeertaken, elk met tests, slaagde de code van het model voor de tests bij 90% van de taken. De paper vergelijkt het met eerdere modellen op dezelfde benchmark, waarvan de beste 81% haalde. Programmeurs worden niet getest.
    """)
    return


@app.cell(hide_code=True)
def explain_claim():
    explain_check = L.explain(
        "check-claim",
        question="Controleer de claim met de vier stappen van het recept. Schrijf op wat er gemeten is, waarmee het vergeleken is, en wat de claim zegt dat de meting niet laat zien.",
        model_answer="**Claim:** AI schrijft betere code dan programmeurs. **Gemeten:** slagen voor de tests bij 90% van 164 korte taken van één benchmark. **Vergeleken met:** eerdere modellen op dezelfde benchmark, hooguit 81%; helemaal niet met programmeurs. **Claim naast meting:** \"beter dan programmeurs\" is nooit gemeten, en korte taken met kant-en-klare tests zijn niet de code die programmeurs elke dag schrijven.",
        points=[
            "je noemt de meting: 90% van 164 korte taken van één benchmark",
            "je noemt de vergelijking: eerdere modellen, geen programmeurs",
            "je zegt dat \"beter dan programmeurs\" verder gaat dan wat gemeten is",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl) liet zien hoe ver een paper gekomen is; nu zie je hoe ver een claim afdwaalt van wat zijn paper mat. Net als in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) betekent een score pas iets naast een vergelijking. In de volgende les, *Een abstract, een figuur en een resultatentabel lezen*, lees je de delen van een paper waar de metingen staan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_contamination, explain_check, result_measured):
    L.wrap_up(result_measured, choice_contamination, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
