# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les res-how-research-is-published. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-how-research-is-published", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl) en [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_experiment():
    quick_experiment = L.choice(
        "quick-experiment",
        question="Een team traint hetzelfde model vijf keer met een nieuwe learning rate en vijf keer met de oude, houdt al het andere gelijk en vergelijkt de validation losses. Welke stap van de onderzoekscyclus is dit?",
        options={
            "question": "De vraag",
            "right": "Het experiment",
            "hypothesis": "De hypothese",
        },
        correct="right",
        why={
            "question": "De vraag komt eerst: wat wil je weten? Dit team is al aan het toetsen. Zie [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl).",
            "hypothesis": "De hypothese is de precieze verwachting die ze toetsen; trainen en de losses vergelijken is de toets zelf. Zie [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl).",
            "right": "Een experiment verandert bewust één ding, houdt de rest gelijk en meet wat er gebeurt.",
        },
    )
    quick_experiment
    return (quick_experiment,)


@app.cell(hide_code=True)
def quick_check_benchmark():
    quick_benchmark = L.choice(
        "quick-benchmark",
        question="Een model haalt 88% op een benchmark van 500 rekenvragen. Wat laat dat zien?",
        options={
            "right": "Hoe het het doet op die 500 vragen, op die manier gescoord",
            "all": "Dat het 88% van alle rekenvragen van die soort oplost",
            "beats": "Dat het een model met 80% elders verslaat",
        },
        correct="right",
        why={
            "all": "Het is getest op 500 vragen, niet op alle. Zie [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).",
            "beats": "Scores op verschillende tests kun je niet vergelijken. Zie [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).",
            "right": "Een benchmarkscore zegt hoe een model het doet op die taken, op die manier gescoord, niet in het algemeen.",
        },
    )
    quick_benchmark
    return (quick_benchmark,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Als je een paper zoekt, voor een eigen project of voor de reproductie aan het eind van deze course, vind je er honderden: in 2026 verschijnen er elke maand duizenden nieuwe papers over AI op arXiv. Voordat je er een avond aan besteedt, vertellen twee vragen je wat je in handen hebt. Wat voegt deze paper toe? En waar verscheen hij? Het eerste zegt naar welk bewijs je zoekt, het tweede hoeveel controle de paper al kreeg.

    **Een voorbeeld uit het dagelijks leven.** Een krant heeft nieuwsberichten, opiniestukken, recensies en rectificaties. Een opiniestuk lees je anders dan een nieuwsbericht: het ene verdedigt een standpunt, het andere vertelt wat er gebeurde. Papers zijn er ook in soorten.
    """)
    return


@app.cell(hide_code=True)
def predict_venue():
    guess_venue = L.guess(
        "where-papers-count",
        question="In de biologie of de natuurkunde verschijnen de papers die tellen in tijdschriften, na maanden beoordeling. Waar gaan de meeste papers die tellen in machine learning door peer review?",
        options={
            "journal": "In tijdschriften, net als in andere wetenschappen",
            "right": "Op grote conferenties, zoals NeurIPS",
            "arxiv": "Nergens: ze komen alleen op arXiv",
        },
        reveal="**Op conferenties.** In machine learning tellen de grote conferenties even zwaar als tijdschriften in andere wetenschappen: een paar reviewers beoordelen elke paper binnen een paar weken, en de geaccepteerde papers komen in de proceedings. De paper over de transformer uit [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl) ging zo: in juni 2017 op arXiv, in december op NeurIPS, dat toen nog NIPS heette. Tijdschriften zoals JMLR zijn er ook, maar daar gaan veel minder papers heen. En niet elke bekende paper gaat door een review: het verslag uit 2019 over GPT-2, een bekend language model, verscheen alleen op de website van het lab dat het model maakte.",
    )
    guess_venue
    return (guess_venue,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Elke paper zegt wat hij toevoegt aan het vakgebied: zijn **bijdrage**. De meeste papers over AI leveren een van zes soorten bijdragen. Een paper over een **methode** brengt een nieuwe manier om iets te doen; een **empirische analyse** meet hoe iets wat al bestaat zich gedraagt; een paper met **theorie** bewijst iets met wiskunde; een paper over een **benchmark of dataset** geeft anderen nieuwe taken of data om te gebruiken; een **position paper** verdedigt een standpunt; een **reproductie** doet de experimenten van iemand anders opnieuw. Sommige papers leveren er twee, zoals een nieuwe benchmark en een analyse van modellen daarop. Een **survey** is een soort paper apart: hij vat de papers van een heel vakgebied samen en vergelijkt ze, en de volgende les begint bij zo'n survey.

        De soort vertelt je wat je controleert. Een nieuwe methode zegt weinig zonder een vergelijking op dezelfde data, zoals in [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl). Een benchmark is alleen iets waard als hij meet wat hij zegt, zoals je zag in [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl). Een bewijs geldt maar zo ver als zijn aannames.
        """),
        mo.Html(C.kinds({"title": "Zes soorten bijdragen", "adds": "Voegt toe:", "check": "Controleer:",
                         "kinds": [("Methode", "een nieuwe manier om iets te doen: een nieuw soort model, een trainingstruc, een algoritme", "verslaat hij sterke baselines op dezelfde data, bij meer dan één taak?", "Voorbeeld: de transformer, achter de chatbots van nu (2017)"),
                                   ("Empirische analyse", "metingen van hoe iets zich gedraagt, zonder nieuwe methode", "hoeveel instellingen en seeds, en geldt het patroon ook daarbuiten?", "Voorbeeld: Scaling Laws for Neural Language Models, hoe de loss daalt als modellen groeien (2020)"),
                                   ("Theorie", "resultaten die met wiskunde bewezen zijn", "welke aannames, en gelden die voor echte modellen?", "Voorbeeld: A Theory of the Learnable, hoeveel voorbeelden genoeg zijn om een regel te leren (1984)"),
                                   ("Benchmark of dataset", "nieuwe testtaken of data voor anderen", "meet hij wat hij zegt, kloppen de antwoorden, stonden de vragen al online?", "Voorbeeld: ImageNet, miljoenen foto's met een label (2009)"),
                                   ("Position paper", "een betoog over waar het vakgebied heen moet", "welk bewijs steunt het, en wat zou de andere kant zeggen?", "Voorbeeld: Troubling Trends in Machine Learning Scholarship (2018)"),
                                   ("Reproductie", "de experimenten van een ander opnieuw gedaan", "wat was hetzelfde, wat was anders, en wat kon niet gecontroleerd worden?", "Voorbeeld: de verslagen van de ML Reproducibility Challenge")],
                         "alt": "Zes kaartjes. Methode: voegt een nieuwe manier toe om iets te doen; controleer of hij sterke baselines verslaat op dezelfde data. Empirische analyse: voegt metingen toe van hoe iets zich gedraagt; controleer hoeveel instellingen en seeds. Theorie: voegt een met wiskunde bewezen uitspraak toe; controleer de aannames. Benchmark of dataset: voegt nieuwe taken of data toe; controleer of hij meet wat hij zegt. Position paper: voegt een betoog toe; controleer het bewijs en de andere kant. Reproductie: doet de experimenten van een ander opnieuw; controleer wat hetzelfde was en wat anders."})),
        mo.md(r"""
        Een paper verschijnt ook ergens: zijn **venue**. De venue vertelt je hoeveel controle de paper al kreeg voordat jij hem leest.

        | Venue | Wie keek hem na | Zoals in 2026 |
        |---|---|---|
        | arXiv | nog niemand: de auteurs zetten hem online | de meeste papers over AI, vaak vóór elke review |
        | workshop | een paar reviewers, vaak minder streng | de workshops naast NeurIPS en ICML |
        | conferentie | een paar reviewers, binnen een paar weken | NeurIPS, ICML, ICLR; ACL voor taal, CVPR voor beeld |
        | tijdschrift | reviewers, vaak in meerdere rondes | JMLR, TMLR |

        In de meeste wetenschappen telt het tijdschrift. In machine learning tellen de grote conferenties even zwaar, en de meeste papers met peer review verschijnen daar, in de proceedings. Een workshop is een kleinere bijeenkomst naast een conferentie, vaak voor werk dat nog in een vroeg stadium is. Namen van venues komen en gaan, dus kijk naar het jaartal erbij.

        Je vindt de venue op de eerste pagina (`Published as a conference paper at ICLR 2025`), in het veld Comments op arXiv (`Accepted at ICML 2025`), of in een verwijzing: *Advances in Neural Information Processing Systems 30* is deel 30 van de proceedings van NeurIPS, uit 2017. Staat er op geen van die plekken een venue? Behandel hem dan als preprint tot je er een vindt: auteurs zetten de venue niet altijd op arXiv.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **soort bijdrage** | wat een paper toevoegt aan het vakgebied: een methode, een empirische analyse, een theorie, een benchmark of dataset, een standpunt of een reproductie |
        | **venue** | de plek waar een paper verschijnt: een conferentie, een workshop, een tijdschrift, of arXiv voor preprints |
        | **workshop** | een kleinere bijeenkomst naast een conferentie, vaak voor vroeg werk, met een lichtere review |
        | **tijdschrift** | een wetenschappelijk blad dat papers publiceert na peer review, vaak in meerdere rondes |
        | **theorie** | hier: resultaten die met wiskunde bewezen zijn, geen vermoeden zoals in gewone taal |
        | **position paper** | een paper die een standpunt verdedigt over waar het vakgebied heen moet, in plaats van een nieuwe methode of een nieuw experiment te presenteren |
        | **reproductie** | de experimenten van een paper opnieuw doen, met de code van de auteurs of met je eigen code, om te zien of je dezelfde resultaten krijgt |
        """),
        L.watch_out("\"Accepted at NeurIPS\" betekent dat een paar reviewers de paper goed genoeg vonden, niet dat het resultaat klopt: peer review is geen garantie, zoals in [Wat een paper, een preprint en peer review zijn](/course/open/res-what-is-a-paper?lang=nl). En een paper die alleen op arXiv staat, is daarom nog niet fout. De venue zegt hoeveel er gecontroleerd is; het bewijs voor de claim controleer je nog altijd zelf."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een paper te plaatsen voordat je hem goed leest:

    1. Lees de titel en het abstract, de korte samenvatting bovenaan de paper, en zoek de zinnen die zeggen wat de paper toevoegt. Papers zijn bijna altijd in het Engels, dus zoek naar "we propose", "we show", "we prove", "we introduce", "we argue" of "we reproduce". Die woorden zijn een aanwijzing, geen bewijs: lees wat erop volgt.
    2. Benoem de soort bijdrage. Zijn het er twee, noteer ze allebei: de hoofdclaim rust meestal op allebei.
    3. Zoek de venue: op de eerste pagina, in het veld Comments op arXiv, of in een verwijzing. Geen venue: behandel hem als preprint tot je er een vindt.
    4. Beslis wat je eerst controleert: het bewijs dat zijn soort nodig heeft, en hoeveel controle zijn venue al deed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen paper, gebouwd zoals het echte onderzoek uit de gok van [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl). Op arXiv: **Fresh sums: 1,200 new maths word problems to test language models.** Uit het abstract, vertaald: *We schreven met de hand 1.200 verhaaltjessommen, in de stijl van een bekende benchmark, en zetten ze nergens online. We testten 30 language models op de oude benchmark en op onze sommen. Een aantal scoorde op onze sommen tot 10 procentpunten lager, wat erop wijst dat ze de oude vragen tijdens het trainen gezien hadden.* Het is versie 1, online gezet in maart 2026, en het veld Comments is leeg.

    1. **Wat hij toevoegt:** "We schreven 1.200 verhaaltjessommen" en "We testten 30 language models". Twee zinnen, twee bijdragen.
    2. **Soort:** een benchmark, met een empirische analyse erop. De hoofdclaim, dat sommige modellen de oude vragen gezien hadden, rust op allebei.
    3. **Venue:** alleen op arXiv, versie 1, geen venue in het veld Comments: een preprint. Niemand heeft hem nog gecontroleerd.
    4. **Wat je eerst controleert:** bij de benchmark of de nieuwe sommen echt even moeilijk zijn als de oude en de antwoorden kloppen; zijn de nieuwe moeilijker, dan kan de daling ook daardoor komen. Bij de analyse welke modellen daalden en hoeveel, en of iets anders dan het zien van de vragen dat kan verklaren. En omdat het een preprint is, ben jij een van de eersten die het controleert.

    Vier stappen, een paar minuten, en je weet wat deze paper moet laten zien voordat je zijn claim gebruikt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke soort?

    Zes verzonnen papers, elk in één zin. Zet elke paper bij de soort bijdrage die hij levert.
    """)
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
            "Vraag bij elke paper: wat kunnen anderen eruit halen? Iets nieuws om te gebruiken, een meting, een bewijs, nieuwe testdata, een betoog, of een controle van eerder werk?",
            "Kijk naar de kaartjes in *Het idee*. Maar één paper werkt met de experimenten van iemand anders, en maar één pleit ergens voor. Van de andere vier: welke brengt een nieuwe manier en vergelijkt die met de oude, en welke meet alleen hoe iets wat al bestaat zich gedraagt?",
        ],
        why="**Methode:** een nieuwe manier om ontbrekende waarden in te vullen. **Empirische analyse:** accuracy bij 1.000 tot 100.000 voorbeelden, met vijf seeds per hoeveelheid. **Theorie:** bewijst dat een trainingsmethode bij de laagste loss eindigt. **Benchmark of dataset:** 2.000 vragen over het Nederlandse recht. **Position paper:** pleit ervoor dat papers hun code publiceren. **Reproductie:** de experimenten van een paper uit 2023, opnieuw gedaan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_venue():
    mo.md(r"""
    ### Oefening 2. Waar verscheen hij?
    """)
    return


@app.cell(hide_code=True)
def choice_venue_cell():
    choice_venue = L.choice(
        "venue",
        question="Een paper waarop je wilt voortbouwen, staat op arXiv. In het veld Comments staat: \"Accepted at a workshop of ICML 2026.\" Wat vertelt dat je over de controle die hij kreeg?",
        options={
            "main": "Dezelfde peer review als de papers van de hoofdconferentie, ICML zelf",
            "none": "Geen: een paper op arXiv is een preprint die niemand gecontroleerd heeft",
            "right": "Enige review, vaak lichter dan bij de hoofdconferentie",
        },
        correct="right",
        why={
            "main": "Een workshop is een kleinere bijeenkomst naast de conferentie, met een eigen review, vaak een lichtere. Zie de venues in *Het idee*.",
            "none": "Veel papers met peer review staan ook op arXiv; het veld Comments zegt waar hij geaccepteerd is. Zie de venues in *Het idee*.",
            "right": "Een workshop is vaak voor vroeg werk: de review is echt, maar meestal lichter dan bij de hoofdconferentie. Reden te meer om het bewijs zelf te controleren.",
        },
    )
    choice_venue
    return (choice_venue,)


@app.cell(hide_code=True)
def text_exercise_place():
    mo.md(r"""
    ### Oefening 3. Plaats zelf een paper

    Nog een verzonnen paper. **Half the labels: training a classifier on fewer labeled examples.** Uit het abstract, vertaald: *We stellen een trainingsmethode voor die ook leert van voorbeelden zonder label. Op twee datasets met foto's haalt een model dat met onze methode op de helft van de voorbeelden met label getraind is, dezelfde accuracy op de test set als een gewoon model dat op alle voorbeelden getraind is.* Op de eerste pagina: `Published as a conference paper at ICLR 2026.` <!-- termen:ok -->

    Stap 1 van het recept is gedaan: wat hij toevoegt, is "We stellen een trainingsmethode voor die ook leert van voorbeelden zonder label".
    """)
    return


@app.cell(hide_code=True)
def explain_place():
    explain_place_paper = L.explain(
        "place-paper",
        question="Doe stap 2 tot en met 4 van het recept: benoem de soort bijdrage, de venue en wat die je vertelt, en wat je eerst zou controleren.",
        model_answer="**Soort:** een methode, een nieuwe manier om een classifier te trainen. **Venue:** een conferentie, ICLR 2026, dus een paar reviewers hebben hem gelezen en beoordeeld. **Eerst controleren:** of het gewone model dezelfde zorg kreeg, met even goed afgestelde hyperparameters, en over hoeveel seeds; en dat alleen twee datasets met foto's getest zijn, dus of het ook voor andere soorten data geldt, laat de paper niet zien.",
        points=[
            "je noemt de soort: een methode",
            "je zegt dat hij door de peer review van een conferentie kwam",
            "je noemt een controle die een methode nodig heeft: een eerlijke vergelijking met het gewone model, of hoe ver het resultaat reikt buiten twee datasets met foto's",
        ],
    )
    explain_place_paper
    return (explain_place_paper,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Hoe onderzoek werkt: vraag, hypothese, experiment](/course/open/res-research-cycle?lang=nl) liet de cyclus zien die in een paper eindigt, en [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl) hoe ver een claim kan afdwalen van zijn meting. Nu zie je, voordat je een paper goed leest, wat voor soort het is en hoeveel controle hij kreeg. Aan het eind van deze course maak je zelf een van de zes soorten: een reproductie. In de volgende les, *Papers vinden*, zoek je papers en volg je hun verwijzingen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_venue, explain_place_paper, result_kinds):
    L.wrap_up(result_kinds, choice_venue, explain_place_paper)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
