# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-paper-anatomy. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-paper-anatomy", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Papers vinden](/course/open/res-find-papers?lang=nl) en [Een abstract, een figuur en een resultatentabel lezen](/course/open/res-figure-and-table?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_invented():
    quick_invented = L.choice(
        "quick-invented",
        question="Een chatbot geeft je een referentie die compleet lijkt: drie auteurs, een titel en een jaartal. Wat doe je voordat je hem gebruikt?",
        options={
            "cite": "Hem gebruiken zoals hij is: de auteurs, de titel en het jaartal staan er allemaal",
            "right": "Hem opzoeken op arXiv, Semantic Scholar of Google Scholar",
            "ask": "De chatbot vragen of de paper echt bestaat",
        },
        correct="right",
        why={
            "cite": "Chatbots kunnen referenties verzinnen die compleet lijken. Zie [Papers vinden](/course/open/res-find-papers?lang=nl).",
            "ask": "De chatbot die een referentie verzon, kan ook zeggen dat hij bestaat. Controleer hem op een plek die echte papers bijhoudt. Zie [Papers vinden](/course/open/res-find-papers?lang=nl).",
            "right": "Een paper die je zelf gevonden hebt, in een zoekmachine voor papers, bestaat echt, en dan kun je hem ook lezen.",
        },
    )
    quick_invented
    return (quick_invented,)


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="In een resultatentabel staat de 91,8 van model A vetgedrukt, boven de 91,2 van model B. Het bijschrift zegt dat het getal van B uit een eerdere paper is overgenomen. Wat controleer je eerst?",
        options={
            "newer": "Of de paper van A nieuwer is dan die van B",
            "bold": "Of vetgedrukt betekent dat A op elke test beter is",
            "right": "Of beide op dezelfde manier gemeten zijn",
        },
        correct="right",
        why={
            "newer": "Nieuwer maakt een getal niet vergelijkbaar. Zie [Een abstract, een figuur en een resultatentabel lezen](/course/open/res-figure-and-table?lang=nl).",
            "bold": "Vetgedrukt markeert alleen het beste getal in zijn kolom. Zie [Een abstract, een figuur en een resultatentabel lezen](/course/open/res-figure-and-table?lang=nl).",
            "right": "Een getal uit een andere paper kan van een andere test set, metric of hoeveelheid training data komen. Alleen rijen die op dezelfde manier gemeten zijn, kun je vergelijken.",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt een paper gevonden. Hij telt dertig pagina's: formules, tabellen, figuren, en na de referenties een appendix. Van voor naar achter gelezen, als een boek, kost hij je uren, en dan mis je misschien nog wat je zocht. Maar elk deel van een paper beantwoordt zijn eigen vraag. Weet je welk deel welke vraag beantwoordt, dan ga je recht op je doel af.

    **Een voorbeeld uit het dagelijks leven.** De handleiding van een wasmachine: de eerste pagina zegt wat de machine kan, een hoofdstuk zegt hoe je hem gebruikt, en achterin staat in kleine letters een tabel met elk programma, zijn temperatuur en hoe lang het duurt. Voor het wolprogramma ga je meteen naar die tabel.
    """)
    return


@app.cell(hide_code=True)
def predict_proof():
    guess_proof = L.guess(
        "adam-proof",
        question="Adam, een optimizer waarmee veel modellen getraind worden, werd geïntroduceerd in een paper uit 2014. In zijn appendix bewijst de paper dat Adam convergeert. De paper werd een van de meest geciteerde in machine learning. Heeft iemand ooit een fout in dat bewijs gevonden?",
        options={
            "no": "Nee, het bewijs houdt stand",
            "right": "Ja, een paar jaar later",
            "never": "Niemand kon het controleren: het bewijs is nooit gepubliceerd",
        },
        reveal="**Ja, een paar jaar later.** In 2018 lieten Reddi, Kale en Kumar zien dat het bewijs een fout bevat, en gaven ze een eenvoudig probleem waarop Adam de beste oplossing niet vindt. Adam werkt in de praktijk nog steeds goed; alleen klopte de stelling niet zoals hij er stond. Een appendix krijgt vaak minder aandacht dan de hoofdtekst, terwijl er veel van de details in staan die je nodig hebt.",
    )
    guess_proof
    return (guess_proof,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De meeste papers in machine learning zijn uit dezelfde delen opgebouwd, en elk deel beantwoordt zijn eigen vraag. Het plaatje laat ze zien in de gebruikelijke volgorde.
        """),
        mo.Html(C.anatomy({"title": "De delen van een paper, en de vraag die elk beantwoordt",
                           "main": "Hoofdtekst: een paginalimiet, op de grote conferenties zo'n 8 tot 10 pagina's (2026)",
                           "main_parts": [("Titel en abstract", "Wat wordt er beweerd?"),
                                          ("Introductie", "Waarom doet het ertoe, en wat is er nieuw? Eindigt vaak met een lijst bijdragen."),
                                          ("Related work", "Waarin verschilt het van eerder werk?"),
                                          ("Methode", "Wat hebben ze precies gedaan of gebouwd?"),
                                          ("Experimenten", "Hoe is het gemeten, waartegen, en wat kwam eruit?"),
                                          ("Conclusie en beperkingen", "Wat betekent het, en waar houdt het op?")],
                           "after": "Na de hoofdtekst: geen paginalimiet",
                           "after_parts": [("Referenties", "Waarop bouwt het voort?"),
                                           ("Appendix A, B, C…", "De details: instellingen, extra resultaten, bewijzen.")],
                           "alt": "Een paper als twee blokken. De hoofdtekst, met een paginalimiet: titel en abstract, wat er beweerd wordt; introductie, waarom het ertoe doet en wat er nieuw is, vaak met aan het eind een lijst bijdragen; related work, waarin het verschilt van eerder werk; methode, wat ze precies gedaan of gebouwd hebben; experimenten, hoe het gemeten is, waartegen, en wat eruit kwam; conclusie en beperkingen, wat het betekent en waar het ophoudt. Na de hoofdtekst, zonder paginalimiet: referenties, waarop het voortbouwt; appendix A, B, C, de details: instellingen, extra resultaten, bewijzen."})),
        mo.md(r"""
        De namen verschillen per paper. De methode heet soms *Approach*, *Model* of naar de methode zelf; de experimenten *Results* of *Evaluation*; related work staat soms direct na de introductie, soms vlak voor de conclusie. De vragen blijven hetzelfde.

        De **lijst met bijdragen** aan het eind van de introductie zegt wat de paper beweert toe te voegen, vaak als (1), (2), (3). Samen met het abstract is dat wat de rest van de paper moet onderbouwen.

        De hoofdtekst heeft een paginalimiet, dus de details verhuizen naar de **appendix** (de bijlage), na de referenties, met paragrafen die letters hebben: A, B, C. De hoofdtekst verwijst ernaar: "see Appendix B". Om een experiment na te bouwen heb je juist die details nodig: de learning rate, de batch size, het aantal epochs, de seeds, hoe de data gesplitst is. Ze staan vaak alleen in de appendix, of in de code, als de auteurs die gepubliceerd hebben.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **opbouw van een paper** | de delen die de meeste papers hebben, elk met een eigen vraag: abstract, introductie, related work, methode, experimenten, conclusie, referenties, appendix |
        | **lijst met bijdragen** | de lijst aan het eind van de introductie met wat de paper beweert toe te voegen |
        | **related work** | het deel dat eerder werk aan hetzelfde probleem beschrijft en zegt waarin deze paper daarvan verschilt |
        | **beperkingen** | wat de paper niet getest heeft of niet kan beweren, vaak vlak bij de conclusie |
        | **appendix** | het deel na de referenties, zonder paginalimiet, met de details: instellingen, extra resultaten, bewijzen |
        """),
        L.watch_out("*Methode* betekent meer dan één ding. In een paper is de methode de nieuwe manier om iets te doen die de auteurs voorstellen, zoals in een paper over een methode, en het deel Methode zegt precies wat ze gedaan of gebouwd hebben. In Python is een methode een functie die bij een object hoort, zoals `append` in `scores.append(0.9)`."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om iets in een paper te vinden:

    1. Zeg wat je zoekt: de claim, wat er gedaan is, hoe het gemeten is, de instellingen om het na te bouwen, of waar het ophoudt.
    2. Ga naar het deel dat die vraag beantwoordt: het abstract en de lijst met bijdragen voor de claim, de methode voor wat er gedaan is, de experimenten voor hoe het gemeten is, de appendix voor de instellingen, de conclusie en de beperkingen voor waar het ophoudt.
    3. Staat het er niet? Zoek in de pdf naar het woord, zoals *learning rate*, *batch* of *seed*, en volg elke "see Appendix …". <!-- termen:ok -->
    4. Schrijf op wat je vond, met het deel waarin het staat, en wat je nergens kon vinden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen paper, gebouwd zoals veel echte: **How wide is wide enough? MLPs for handwritten digits.** Het abstract, vertaald: *We trainen MLP's met één hidden layer van 16 tot 4.096 neuronen op handgeschreven cijfers. De accuracy op de test set stijgt met de breedte tot 512 neuronen, en blijft daarna rond 98%.* Je wilt die curve nabouwen.

    1. **Wat je zoekt:** de claim, het model, de data en hoe het gemeten is, en de instellingen.
    2. **De claim** staat in het abstract: de accuracy stijgt tot 512 neuronen en blijft daarna rond 98%. De introductie eindigt met de bijdragen: (1) de curve van 16 tot 4.096 neuronen, (2) code om hem na te bouwen. **Het model** staat in paragraaf 3, de methode: één hidden layer met ReLU, softmax aan de uitgang. **De data en hoe het gemeten is** staan in paragraaf 4, de experimenten: 60.000 plaatjes om op te trainen en 10.000 om op te testen, en figuur 2 laat de accuracy op de test set zien tegen de breedte, het gemiddelde van vijf seeds met hun spreiding. **De instellingen** staan niet in paragraaf 4; die zegt "training details in Appendix B".
    3. In appendix B.1: SGD, een learning rate van 0,1, een batch size van 128, 20 epochs, seeds 0 tot en met 4. Zoeken op *validation* levert niets op: de paper zegt niet hoe de learning rate en de 20 epochs gekozen zijn.
    4. **Opgeschreven:** claim (abstract), model (paragraaf 3), data en curve (paragraaf 4, figuur 2), instellingen (appendix B.1). Niet gevonden: hoe de learning rate en het aantal epochs gekozen zijn. Kijk in de code, of vraag het de auteurs.

    Vier plekken in een paper van twintig pagina's, en één vraag die de paper openlaat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welk deel?

    Zes stukjes uit een andere verzonnen paper, die één brede hidden layer vergelijkt met drie smalle bij hetzelfde aantal parameters. Zet elk stukje bij het deel van de paper waar het waarschijnlijk uit komt.
    """)
    return


@app.cell(hide_code=True)
def exercise_where():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "Vraag bij elk stukje welke vraag het beantwoordt: wat er nieuw is, waarin het van anderen verschilt, wat het model doet, wat eruit kwam, waar het ophoudt, of hoe het precies is opgezet.",
            "Kijk naar het plaatje in *Het idee*. Eén stukje begint met een paragraafnummer met een letter erin; één somt op wat de paper toevoegt, als (1) en (2); één noemt een tabel.",
        ],
        why="**Introductie:** \"Onze bijdragen zijn…\". **Related work:** \"Lee et al. (2019)… Park et al. (2021)…; de onze niet\". **Methode:** hoe een plaatje van 28 bij 28 pixels de eerste hidden layer in gaat. **Experimenten:** tabel 2, de accuracy voor elke breedte over vijf seeds. **Conclusie en beperkingen:** alleen handgeschreven cijfers zijn getest. **Appendix:** B.2, de trainingsdetails.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_baselines():
    mo.md(r"""
    ### Oefening 2. Was de vergelijking eerlijk?
    """)
    return


@app.cell(hide_code=True)
def choice_baselines_cell():
    choice_baselines = L.choice(
        "fair-baselines",
        question="Het abstract van een paper zegt dat zijn methode elke baseline verslaat. Je wilt weten of de baselines even goed afgesteld zijn als de nieuwe methode. Waar staat dat waarschijnlijk?",
        options={
            "intro": "In de introductie, in de lijst met bijdragen",
            "right": "In de experimenten, of in de appendix bij de instellingen",
            "related": "In het deel related work, waar de papers van de baselines genoemd en vergeleken worden",
        },
        correct="right",
        why={
            "intro": "De bijdragen zeggen wat de paper beweert, niet hoe het gemeten is. Zie *Het idee*.",
            "related": "Related work zegt waarin de baselines in hun idee verschillen, niet hoe ze voor deze paper getraind zijn. Zie *Het idee*.",
            "right": "Hoe elke methode is afgesteld, hoort bij hoe er gemeten is: in de experimenten, of, vaak, in de appendix. Staat het nergens, schrijf dat dan op: dat is het weten waard.",
        },
    )
    choice_baselines
    return (choice_baselines,)


@app.cell(hide_code=True)
def text_exercise_repro():
    mo.md(r"""
    ### Oefening 3. Wat heb je nodig om het na te bouwen?

    Aan het eind van deze course reproduceer je een paper. Voordat je begint, wil je weten waar je alles vindt wat je nodig hebt.
    """)
    return


@app.cell(hide_code=True)
def explain_repro():
    explain_repro_list = L.explain(
        "repro-list",
        question="Maak een lijst van wat je nodig hebt om het hoofdresultaat van een paper na te bouwen, en zet bij elk punt het deel van de paper waar je eerst kijkt.",
        model_answer="**De claim:** het abstract en de lijst met bijdragen. **Het model:** de methode. **De data, de baseline, de metric en het resultaat dat je wilt halen:** de experimenten, in de tabel of figuur met het hoofdresultaat. **De instellingen:** de learning rate, de batch size, het aantal epochs, de seeds en de splitsing van de data: de appendix, of de code als de auteurs die gepubliceerd hebben. **Wat ontbreekt:** zoek in de pdf, en schrijf op wat je niet kon vinden.",
        points=[
            "je noemt instellingen zoals de learning rate, de batch size en de seeds, en zoekt ze in de appendix",
            "je zoekt de claim in het abstract of de bijdragen, en het resultaat dat je wilt halen in de experimenten",
            "je schrijft op wat je niet kon vinden",
        ],
    )
    explain_repro_list
    return (explain_repro_list,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een abstract, een figuur en een resultatentabel lezen](/course/open/res-figure-and-table?lang=nl) las je de delen met de metingen; nu weet je waar elk ander deel staat, en welke vraag het beantwoordt. [Papers vinden](/course/open/res-find-papers?lang=nl) gaf je de papers; nu vind je je weg erbinnen. In de volgende les, *Een paper lezen in drie rondes*, lees je de paper over de transformer in drie rondes, van de claim tot de details.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_baselines, explain_repro_list, result_where):
    L.wrap_up(result_where, choice_baselines, explain_repro_list)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
