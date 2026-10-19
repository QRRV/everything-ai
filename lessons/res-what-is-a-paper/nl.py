# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-what-is-a-paper. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.
# termen:allow genai

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("res-what-is-a-paper", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_translate():
    quick_translate = L.choice(
        "quick-translate",
        question="Een programma vertaalt tekst naar andere talen. Het leerde dat van miljoenen vertaalde zinnen, met een neural network van veel layers. Wat is de kleinste cirkel waar het in past?",
        options={
            "dl": "Deep learning: het is een neural network met veel layers",
            "right": "Generatieve AI: het maakt nieuwe tekst",
            "ml": "Machine learning: het leerde van voorbeelden",
        },
        correct="right",
        why={
            "dl": "Het is deep learning, maar de derde vraag van het recept gaat één cirkel verder naar binnen: maakt het iets nieuws? Zie [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "ml": "Het leerde van voorbeelden, dus het is machine learning; maar stel ook de volgende twee vragen. Zie [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "right": "Geleerd van voorbeelden, een neural network met veel layers, en het schrijft nieuwe zinnen: generatieve AI.",
        },
    )
    quick_translate
    return (quick_translate,)


@app.cell(hide_code=True)
def quick_check_tax():
    quick_tax = L.choice(
        "quick-tax",
        question="Een programma zoekt uit wat er mis is met een auto door je vragen te stellen, met regels die monteurs hebben opgeschreven. Waar past het?",
        options={
            "right": "AI, maar geen machine learning: mensen schreven de regels",
            "ml": "Machine learning: het weet wat monteurs weten",
            "none": "Nergens: een programma met regels is geen AI",
        },
        correct="right",
        why={
            "ml": "Machine learning vindt zijn regels in voorbeelden; hier schreven monteurs ze op. Zie [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "none": "Sommige AI volgt regels die mensen schreven, zoals het schaakprogramma: een fout in een auto vinden is werk voor een expert. Zie [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl).",
            "right": "De eerste vraag van het recept: mensen schreven de regels, dus het blijft in de cirkel AI.",
        },
    )
    quick_tax
    return (quick_tax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Nieuws over AI begint vaak met "uit nieuw onderzoek blijkt". Achter zo'n bericht zit meestal een paper. Of je het bericht kunt vertrouwen, hangt af van wat voor paper het is, waar hij verscheen, en wie hem controleerde. Later in deze course lees je zelf papers, en aan het eind van het onderzoeksdeel bouw je zelf een paper na.

    **Een voorbeeld uit het dagelijks leven.** Twee links naar een recept voor dezelfde taart: een naar de website van een kookboek, waar een redacteur elk recept las en verbeterde voordat het online kwam, en een naar een recept dat iemand dezelfde avond online zette. Op je scherm zien ze er hetzelfde uit.
    """)
    return


@app.cell(hide_code=True)
def predict_review():
    guess_review = L.guess(
        "review",
        question="Een paper kwam door de peer review van een grote AI-conferentie: voordat de conferentie hem accepteerde, hebben een paar andere onderzoekers hem gelezen en beoordeeld. Hoe zeker kun je zijn dat het belangrijkste resultaat klopt?",
        options={
            "certain": "Zeker: experts hebben alles erin gecontroleerd",
            "right": "Waarschijnlijker dan daarvoor, maar niet zeker",
            "same": "Niet zekerder dan daarvoor: review verandert niets",
        },
        reveal="**Waarschijnlijker dan daarvoor, maar niet zeker.** Een paar experts lezen de paper, vragen om aanpassingen en vangen veel fouten af. Maar net als de redacteur van het kookboek, die een recept leest zonder de taart te bakken, voeren ze de experimenten meestal niet opnieuw uit, ze hebben weinig tijd, en ze zijn het vaak oneens. In 2014 liet de grote AI-conferentie NeurIPS, toen nog NIPS, twee aparte groepen reviewers dezelfde papers beoordelen, en de twee groepen waren het over veel ervan oneens. Een resultaat wordt pas echt stevig als andere onderzoekers het herhalen.",
    )
    guess_review
    return (guess_review,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **paper** is een geschreven verslag van een onderzoek: de vraag, wat de onderzoekers deden, wat eruit kwam, en hoe het zich verhoudt tot eerder werk. Onderzoekers delen hun werk als papers, zodat anderen het kunnen lezen, controleren en erop voortbouwen.

        Een **preprint** is een paper die de auteurs zelf online zetten, vóór of zonder peer review. Veel AI-papers verschijnen eerst op **arXiv**, uitgesproken als "archive", een gratis website voor preprints. Dat is snel en voor iedereen open, maar niemand heeft gecontroleerd of de paper klopt: arXiv kijkt alleen of het eruitziet als onderzoek. Een preprint kan nieuwe versies krijgen, v1, v2 enzovoort.

        **Peer review** is de controle door andere onderzoekers uit het vakgebied, de peers. Een conferentie of een tijdschrift legt een ingediende paper voor aan een paar experts, die meestal anoniem blijven en adviseren of hij gepubliceerd moet worden en wat er beter moet. Pas dan neemt een conferentie de paper op in haar proceedings, de verzameling papers die ze accepteerde, of een tijdschrift in een nummer. Peer review vangt veel fouten af, maar is geen garantie: reviewers lezen, ze voeren de experimenten zelden opnieuw uit.

        Een claim wordt dus stap voor stap sterker: een preprint, dan een paper die door peer review kwam, dan een resultaat dat andere onderzoekers herhalen. Een persbericht of een nieuwsbericht is iemands samenvatting van een paper, vaak korter en stelliger dan de paper zelf.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **paper** | een geschreven verslag van een onderzoek: vraag, methode, resultaten, en hoe het zich verhoudt tot eerder werk |
        | **preprint** | een paper die de auteurs zelf online zetten, vóór of zonder peer review: nog door niemand gecontroleerd |
        | **arXiv** (zeg "archive") | de gratis website waar de meeste AI-preprints het eerst verschijnen |
        | **peer review** | de controle door een paar andere onderzoekers voordat een conferentie of tijdschrift een paper publiceert; vangt veel fouten af, maar is geen garantie |
        """),
        L.watch_out("\"Gepubliceerd\" betekent twee dingen. \"Gepubliceerd op arXiv\" is een preprint, door de auteurs zelf online gezet. \"Gepubliceerd op een conferentie\" of \"in een tijdschrift\" betekent dat hij door peer review kwam. Nieuwsberichten maken dat onderscheid vaak niet."),
        mo.md(r"""
        De figuur: de gebruikelijke weg van een paper in AI.
        """),
        draw.flow(
            [("Preprint", ["de auteurs zetten", "hem op arXiv"]),
             ("Peer review", ["een paar onderzoekers", "lezen en beoordelen"]),
             ("Gepubliceerd", ["in de proceedings", "van een conferentie,", "of in een tijdschrift"])],
            forward=["ingestuurd", "geaccepteerd"],
            back=["aanpassingen\ngevraagd, of\nafgewezen"],
            title="Van preprint tot gepubliceerde paper"),
        mo.md(r"""
        Niet elke paper gaat de hele weg: sommige blijven een preprint, en een afgewezen paper gaat vaak, na aanpassingen, naar een andere conferentie.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zoek je uit hoeveel je een onderzoek kunt vertrouwen waarover je leest:

    1. Zoek de paper zelf, niet alleen het bericht erover: een goed nieuwsbericht linkt ernaar; anders zoek je op de titel.
    2. Kijk waar hij verscheen: alleen op arXiv of een website, dus een preprint; of op een conferentie of in een tijdschrift, na peer review.
    3. Kijk naar de datum en de versie: een preprint kan nog veranderen.
    4. Vraag of anderen het resultaat herhaalden: een resultaat dat ook bij anderen standhield, is het sterkst.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    **"Attention Is All You Need"**, van Vaswani en zeven coauteurs, de paper die de transformer introduceerde: het soort neural network achter de chatassistenten van nu, dat je later in de course zelf bouwt.

    1. **De paper zelf.** Hij staat op arXiv, met het nummer 1706.03762: 17 voor 2017 en 06 voor juni.
    2. **Waar hij verscheen.** Eerst als preprint, in juni 2017. Daarna kwam hij door de peer review van de conferentie NeurIPS, toen nog NIPS, en verscheen hij in december 2017 in haar proceedings.
    3. **Datum en versie.** arXiv laat zeven versies zien: vier in juni 2017, een in december 2017 en twee in 2023. De auteurs veranderden onderweg details; lees de laatste.
    4. **Door anderen herhaald?** Ja, duizenden keren: de transformer is de basis van de chatassistenten van nu, en talloze latere papers bouwden erop voort en bevestigden dat hij werkt.

    Deze claim ging de hele weg, van preprint via peer review tot een resultaat dat anderen herhaalden. Veel claims in het nieuws hebben alleen de eerste stap gezet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat voor tekst?

        Zes teksten over onderzoek. Zet elke tekst in zijn groep.
        """),
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
            "Vraag bij elke tekst: is het het verslag van het onderzoek zelf, en zo ja, controleerden andere onderzoekers het voordat het verscheen?",
            "Een tekst die alleen over het onderzoek vertelt, zonder de methode en alle resultaten, is niet de paper. En een paper kwam alleen door peer review als een conferentie of een tijdschrift hem publiceerde.",
            "De blogpost en het persbericht zijn niet de paper. arXiv en de eigen website van de auteurs hebben geen peer review. Twee teksten zijn dus preprints, twee kwamen door peer review, en twee zijn geen paper.",
        ],
        why="**Preprint:** de paper op arXiv, versie 1, en de paper op de eigen website van de auteurs. **Paper die door peer review kwam:** de paper in de proceedings van NeurIPS en de paper die JMLR accepteerde. **Geen paper:** de blogpost en het persbericht.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_arxiv():
    mo.md(r"""
    ### Oefening 2. Gepubliceerd op arXiv
    """)
    return


@app.cell(hide_code=True)
def choice_arxiv_cell():
    choice_arxiv = L.choice(
        "on-arxiv",
        question="Een nieuwssite schrijft over een nieuw model: \"het onderzoek is vandaag gepubliceerd op arXiv\". Wat weet je over dat onderzoek?",
        options={
            "reviewed": "Reviewers controleerden of de resultaten kloppen",
            "rejected": "Een tijdschrift wees het af, dus zetten de auteurs het op arXiv",
            "right": "De auteurs zetten het zelf online; arXiv deed geen review",
        },
        correct="right",
        why={
            "reviewed": "arXiv kijkt alleen of een paper eruitziet als onderzoek; het heeft geen peer review. Zie *Het idee*.",
            "rejected": "Auteurs zetten een paper vaak op arXiv vóór of tijdens de review; het zegt niet dat iemand hem afwees. Zie *Het idee*.",
            "right": "Het is een preprint: snel en open, maar niemand deed er een review van. \"Gepubliceerd\" op arXiv betekent niet dat er peer review was.",
        },
    )
    choice_arxiv
    return (choice_arxiv,)


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Oefening 3. Voordat je het gelooft
    """)
    return


@app.cell(hide_code=True)
def explain_claim():
    explain_check = L.explain(
        "check-claim",
        question="Iemand deelt een nieuwsbericht met je: \"Wetenschappers tonen aan: nieuwe AI leest scans beter dan artsen.\" Welke vragen zou je stellen voordat je het gelooft? Schrijf ze op.",
        model_answer="**Is er een paper, en kan ik hem vinden?** **Is het een preprint, of kwam hij door peer review, en waar?** **Heeft iemand anders het resultaat herhaald?** En: zegt de paper zelf wat het nieuwsbericht zegt?",
        points=[
            "je vraagt naar de paper zelf, niet alleen naar het nieuwsbericht",
            "je vraagt of het een preprint is of door peer review kwam",
            "je vraagt of anderen het resultaat herhaalden",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [AI, machine learning, deep learning en generatieve AI](/course/open/start-ai-ml-genai?lang=nl) liet zien hoe je een systeem plaatst; nu kun je een claim erover plaatsen: een preprint die nog niemand controleerde, een paper die door peer review kwam, of een resultaat dat anderen herhaalden. In de volgende les, *AI-claims kritisch lezen*, vergelijk je wat een persbericht, een paper en een benchmarkscore elk zeggen, en vraag je wat er gemeten is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_arxiv, explain_check, result_kinds):
    L.wrap_up(result_kinds, choice_arxiv, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
