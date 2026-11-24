# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-qa-kinds. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-qa-kinds", "nl")
    C = L.checks   # de zes teksten en twaalf vragen van de course, de run van Qwen3-0.6B, de beelden en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl), en één over [Lijsten](/course/open/py-lists?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_nlu():
    quick_nlu = L.choice(
        "quick-nlu",
        question="In *Wat NLP is* gaat natural language understanding (NLU) van tekst naar betekenis, en natural language generation (NLG) van betekenis naar tekst. Een systeem leest een vraag, bepaalt wat er gevraagd wordt, en schrijft het antwoord in een eigen zin. Wat doet het systeem?",
        options={
            "nlu": "Alleen NLU: het systeem leest",
            "nlg": "Alleen NLG: het systeem schrijft",
            "right": "Allebei: NLU voor de vraag, NLG voor het antwoord",
            "cl": "Computational linguistics: het systeem bestudeert hoe mensen vragen stellen",
        },
        correct="right",
        why={
            "right": "De vraag lezen is NLU, en het antwoord schrijven is NLG. Een systeem dat het antwoord alleen in een tekst aanwijst, heeft geen NLG nodig: deze les laat beide soorten zien.",
            "nlu": "De vraag lezen is NLU. En wat doet het systeem daarna, met het antwoord? Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "nlg": "Het antwoord schrijven is NLG. En wat doet het systeem eerst, met de vraag? Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "cl": "Computational linguistics bestudeert taal zelf met computers. Dit systeem voert een taak met taal uit: welke delen van NLP gebruikt het systeem? Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_nlu
    return (quick_nlu,)


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question='Wat geeft `words[2:4]`, voor `words = ["het", "zwembad", "gaat", "om", "9.00"]`?',
        options={
            "early": '`["zwembad", "gaat"]`',
            "right": '`["gaat", "om"]`',
            "three": '`["gaat", "om", "9.00"]`: van index 2 tot en met index 4',
            "start": '`["het", "zwembad", "gaat"]`',
        },
        correct="right",
        why={
            "right": 'Index 2 is `"gaat"`, en de slice stopt net bij index 4, zonder index 4 zelf. Een tekst kun je op dezelfde manier in slices knippen, teken voor teken: deze les wijst een antwoord in een tekst aan met twee van zulke getallen.',
            "early": "Indexen beginnen bij 0. Welk item heeft index 2? Zie [Lijsten](/course/open/py-lists?lang=nl).",
            "three": "Een slice stopt net bij het tweede getal, zonder dat getal zelf. Welke index is de laatste in de slice? Zie [Lijsten](/course/open/py-lists?lang=nl).",
            "start": "Een slice begint bij het eerste getal, niet bij het begin van de lijst. Welk item heeft index 2? Zie [Lijsten](/course/open/py-lists?lang=nl).",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je typt een vraag in een zoekmachine, en boven de links staat het antwoord, met de zin waar het vandaan komt. Je stelt een chatassistent dezelfde vraag, en de assistent schrijft een antwoord in eigen woorden, zonder bron. Allebei zijn het vormen van **question answering**: vragen beantwoorden die in gewone taal gesteld zijn. Maar het antwoord komt van een andere plek, en heeft een andere vorm: de zoekmachine wijst een stuk aan van een pagina die de zoekmachine vond, de assistent schrijft uit wat het model leerde. Welke soort een systeem is, bepaalt wat je kunt controleren, en wat er mis kan gaan.

    In deze les beantwoordt Qwen3-0.6B, het kleine chat model van de course, twaalf vragen over zes korte teksten die de course schreef: één keer zonder tekst, en één keer met elk van de zes teksten.

    **Een voorbeeld uit het dagelijks leven.** Op een station leest de een je de vertrektijd voor van het bord, en een ander zegt de tijd uit het hoofd. De eerste zit er alleen naast als het bord fout is; de tweede kan ernaast zitten zonder dat iemand het merkt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_town = L.guess(
        "guess",
        question="Drie van de zes teksten gaan over de wereld: spinnen, Mercurius en de Eiffeltoren. Drie gaan over een dorp dat de course verzon: de bibliotheek, het repaircafé en het openluchtzwembad. Qwen3 krijgt de zes vragen over het dorp zonder tekst, met de seeds 0 tot en met 3, dus 24 keer in totaal. In hoeveel van deze 24 antwoorden klopt het antwoord?",
        options={
            "none": "Geen, of bijna geen",
            "quarter": "Ongeveer een kwart",
            "half": "Ongeveer de helft",
            "most": "De meeste: de vragen zijn makkelijk",
        },
        reveal="Geen enkele van de 24. Qwen3 schrijft dat de bibliotheek op vrijdag of op dinsdag gesloten is, dat het zwembad 14 of zelfs 100 maanden per jaar open is, en dat een dagkaartje *ongeveer €100* kost. Soms blijft het vaag, zoals *een bepaalde prijs*, maar het model zegt nergens dat het dat niet kan weten.",
    )
    guess_town
    return (guess_town,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Systemen voor **question answering** (QA) verschillen op twee punten: waar het antwoord vandaan komt, en welke vorm het antwoord heeft.

    **Waar het antwoord vandaan komt.** Bij **reading comprehension** komt er een tekst mee met de vraag, en het antwoord staat erin. Bij **open-domain question answering** krijgt het systeem alleen de vraag, en moet het eerst in een grote verzameling teksten vinden waar het antwoord in kan staan: een **retriever** vindt de teksten, en een **reader** beantwoordt de vraag met de gevonden teksten. Bij **closed-book question answering** zoekt het systeem niets op: een model antwoordt uit wat het leerde tijdens de training. Adam Roberts, Colin Raffel en Noam Shazeer, die de taak in 2020 de naam gaven, vergelijken het met een tentamen: een model dat zo antwoordt, lijkt op een student bij een tentamen met het boek dicht, die alles wat ertoe doet vooraf moet leren en onthouden. Open-domain question answering doet volgens Roberts en anderen alsof het een tentamen met het boek open maakt, waarbij het informatie uit een bron buiten het model mag opzoeken en gebruiken.

    **De vorm van het antwoord.** Bij **extractive question answering** is het antwoord een **span**: een stuk van de tekst dat er letterlijk in staat, aangegeven met de plaats waar het begint en de plaats waar het eindigt. Bij **abstractive question answering** schrijft het systeem het antwoord zelf: in eigen woorden, of na een kleine som. Roberts en anderen schrijven dat je een model kunt trainen om de span van de context met het antwoord aan te wijzen, of om de tekst van het antwoord zelf te geven. Closed-book question answering is altijd abstractive: zonder tekst kan het systeem nergens naar wijzen.

    Het beeld zet de soorten naast elkaar, elk met een systeem of een dataset die de soort bekend maakte.
    """)
    return


@app.cell(hide_code=True)
def kinds_cell():
    mo.Html(C.kinds_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **question answering** (QA) | vragen beantwoorden die in gewone taal gesteld zijn, met een gegeven tekst, met een verzameling teksten, of met wat een model leerde |
        | **reading comprehension** | question answering waarbij er een tekst met de vraag meekomt waar het antwoord in staat |
        | **open-domain question answering** | question answering zonder gegeven tekst: het systeem vindt eerst teksten in een verzameling, en leest deze teksten dan |
        | **retriever**, **reader** | de twee stappen van open-domain question answering: de retriever vindt teksten waar het antwoord in kan staan, de reader beantwoordt de vraag met de gevonden teksten |
        | **closed-book question answering** | question answering met alleen wat een model leerde, zonder iets op te zoeken |
        | **extractive question answering** | question answering waarbij het antwoord een span van een tekst is: het systeem wijst alleen aan waar de span begint en eindigt |
        | **span** | een stuk van een tekst, aangegeven met de plaats waar het begint en de plaats waar het eindigt; in de course een tuple `(start, end)`, zodat `text[start:end]` het antwoord is |
        | **abstractive question answering** | question answering waarbij het systeem het antwoord zelf schrijft |
        | `text[a:b]` | een slice van een tekst: de tekens vanaf plaats `a` tot plaats `b`, zonder plaats `b` zelf, net als bij een lijst |
        """),
        L.watch_out("In andere lessen van de course, over stap voor stap volgen wat een AI-toepassing doet, is een *span* één stap van een trace, zoals één aanroep van een model. Hier is een span een stuk van een tekst: hetzelfde woord voor iets anders."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zie je welke soort een systeem is:

    1. Komt er een tekst mee met de vraag? Dan is het reading comprehension.
    2. Vindt het systeem zelf teksten, in een verzameling? Dan is het open-domain: eerst een retriever, dan een reader.
    3. Zoekt het systeem niets op? Dan is het closed-book.
    4. Is het antwoord een span van een tekst, aangegeven met waar de span begint en eindigt? Dan is het extractive. Schrijft het systeem het antwoord? Dan is het abstractive.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tekst over de bibliotheek van het dorp beantwoordt *Hoe laat gaat de bibliotheek van het dorp op zaterdag dicht?* met *14.30 uur*. Neem één korte zin zoals in de tekst: *Zaterdags sluit de bibliotheek om 14.30.* Als span is het antwoord twee getallen: waar het in de zin begint, en waar het eindigt. Een slice van de tekst met deze twee getallen geeft het antwoord terug, net als een slice van een lijst:
    """)
    return


@app.cell
def example_cell():
    text = "Zaterdags sluit de bibliotheek om 14.30."
    start = 34
    end = 39
    print(text[start:end])
    print(end - start, len("14.30"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De span `(34, 39)` geeft *14.30*: er gaan 34 tekens aan vooraf, spaties meegeteld. Het einde is de plaats net na het laatste teken, dus `end - start` is de lengte van het antwoord, 5. Alleen een antwoord dat letterlijk in de tekst staat, heeft een span. In de hele tekst over de bibliotheek is het antwoord *14.30 uur* de span `(129, 138)`. Het beeld laat de vraag zien in alle vier de soorten, met de antwoorden van Qwen3 uit de run van de course.
    """)
    return


@app.cell(hide_code=True)
def four_cell():
    mo.Html(C.four_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_four():
    mo.md(r"""
    Met de tekst schrijft Qwen3 *De bibliotheek van het dorp is op zaterdag dicht van 14.30 uur.*: een abstractive antwoord met de span erin, al loopt de zin wat stroef. Zonder tekst schrijft het model *19:00*. Het dorp is verzonnen, dus Qwen3 kan niet weten wanneer de bibliotheek dichtgaat, maar het model antwoordt toch, en klinkt net zo zeker als met de tekst. Open-domain question answering moet de tekst eerst vinden: de tekst over de bibliotheek deelt negen verschillende woorden met de vraag, meer dan elke andere tekst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_span():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. find_span: het antwoord als span

        Schrijf `find_span(text, answer)`. Geef de tuple `(start, end)` terug van de eerste plaats waar `answer` letterlijk in `text` staat, zodat `text[start:end] == answer`; geef `None` terug als het antwoord nergens in de tekst staat. De loop probeert elke plaats waar het antwoord kan beginnen: de laatste is `len(text) - len(answer)`. Vervang de `...`.

        De check roept `find_span("De bibliotheek gaat om 10.00 uur open.", "10.00 uur")` aan, wat `(23, 32)` geeft, en hetzelfde in het Engels. Daarna eigen gevallen: een antwoord aan het begin en helemaal aan het eind van een tekst, een antwoord dat er niet in staat, een antwoord dat er twee keer in staat, en antwoorden in de teksten van de verzameling (in de les: `C.TEXTS`).
        """),
        L.task("Vind waar een antwoord in een tekst staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_span():
    box_span = L.code_box("find_span", lines=7)
    box_span
    return (box_span,)


@app.cell(hide_code=True)
def check_span(box_span):
    result_span = L.check("find_span", box_span.value)
    result_span
    return (result_span,)


@app.cell(hide_code=True)
def hints_span():
    L.hints(
        "find_span",
        [
            "In de loop is `start` de plaats om te proberen. Het stuk van `text` vanaf `start`, zo lang als het antwoord, eindigt bij `start + len(answer)`; als dat stuk gelijk is aan `answer`, geeft `return` de twee plaatsen terug.",
            "Een slice van een tekst werkt als een slice van een lijst: `text[3:8]` zijn de tekens vanaf plaats 3 tot plaats 8, zonder plaats 8 zelf. Welke slice is zo lang als het antwoord en begint bij `start`?",
            "`        if text[___] == answer:`",
        ],
        why="In de loop: `if text[start:start + len(answer)] == answer:` met daaronder `return (start, start + len(answer))`. Na de loop: `return None`.",
    )
    return


@app.cell(hide_code=True)
def text_after_span():
    mo.md(r"""
    Zeven van de twaalf antwoorden hebben een span. De andere vijf staan niet letterlijk in de tekst, zoals *twee* voor hoeveel poten een spin meer heeft dan een insect: de tekst zegt *acht* en *zes*, en het antwoord vraagt een som. Een extractive systeem kan zo'n antwoord niet geven; het kan alleen naar *acht* of naar *zes* wijzen. Antwoorden die mensen schrijven, zijn vaak zo. NarrativeQA, een dataset met vragen over boeken en filmscripts, heeft antwoorden die mensen in eigen woorden schreven, en Tomáš Kočiský en anderen telden hoeveel antwoorden als span in de samenvattingen en in de verhalen staan: 44,05% en 29,57%. Ongeveer 70% van de antwoorden is dus geen span van het verhaal.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. best_text: de retriever

        Schrijf `best_text`, dat een `question` krijgt en `texts`, een lijst teksten. Geef de plaats in de lijst terug van de tekst die de meeste verschillende woorden met de vraag deelt; bij gelijkspel de eerste daarvan. Het vak kent `words(text)`, dat de woorden van een tekst geeft in kleine letters, zonder leestekens: `words("Op zondag is het dicht.")` geeft `["op", "zondag", "is", "het", "dicht"]`. In de startcode bevat `asked` de verschillende woorden van de vraag, `best` de plaats van de beste tekst tot nu toe, en `most` het aantal woorden dat de beste tekst deelt. Vervang de `...`.

        De check roept `best_text("Wanneer gaat het zwembad open?", ["De bibliotheek gaat om 10.00 uur open.", "Het zwembad gaat om 9.00 uur open."])` aan, wat `1` geeft, en hetzelfde in het Engels. Daarna eigen gevallen: een gelijkspel, een woord van de vraag binnen een langer woord, een tekst die woorden van de vraag herhaalt, en vragen van de les met de zes teksten (in de les: `C.TEXTS`).
        """),
        L.task("Kies de tekst die de meeste woorden met de vraag deelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("best_text", lines=10)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("best_text", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "best_text",
        [
            "In de loop kan `shared` de woorden `w` van `asked` tellen die ook voorkomen in wat `words` van tekst `k` maakt: `len` van een list comprehension. Als `shared` groter is dan `most`, wordt `best` gelijk aan `k` en `most` gelijk aan `shared`.",
            "Tel elk verschillend woord van de vraag één keer: loop over `asked`, niet over de woorden van de tekst. En welke vergelijking houdt bij gelijkspel de eerste tekst, `>` of `>=`?",
            "`        shared = len([w for w in asked if w in ___])`",
        ],
        why="In de loop: `shared = len([w for w in asked if w in words(texts[k])])`, dan `if shared > most:` met daaronder `best = k` en `most = shared`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    Je retriever kiest bij elf van de twaalf vragen de tekst die de vraag beantwoordt. Bij *Hoeveel maanden per jaar is het zwembad open?* kiest de retriever de tekst over Mercurius: Mercurius, de bibliotheek en het zwembad delen elk drie woorden met de vraag, en Mercurius komt eerst. Bij Mercurius zijn dat *jaar*, *is* en *het*: *is* en *het* staan in de meeste teksten, en *jaar* staat erin omdat de tekst over een jaar op Mercurius gaat. Het ene woord dat ertoe doet, *zwembad*, telt niet zwaarder mee dan *het*. Een echte retriever laat zeldzame woorden zwaarder meetellen dan gewone, zoals latere lessen over zoeken laten zien.

    En een echte verzameling is veel groter. DrQA, het systeem van Danqi Chen en anderen, zocht in heel Wikipedia: om een vraag te beantwoorden, moet het systeem eerst de paar pagina's vinden die ertoe doen, tussen meer dan 5 miljoen, en deze pagina's dan zorgvuldig lezen om het antwoord aan te wijzen. Bij 77,8% van de vragen van SQuAD, een dataset met vragen over alinea's van Wikipedia, stond het antwoord in een van de vijf pagina's die de retriever bovenaan zette.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_open():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. open_domain: vind een tekst, en laat Qwen3 de tekst lezen

        Schrijf `open_domain(question, texts)`. Kies een tekst met `best_text`, en geef terug wat Qwen3 met deze tekst op de vraag antwoordt. Het vak kent de `best_text` van de course, dus je antwoord op oefening 2 doet hier niet mee, en `ask(question, text)`, dat het antwoord van Qwen3 geeft uit de run van de course, met seed 0, op een van de vragen van de les met een van de teksten; `ask(question)`, zonder tekst, geeft het closed-book antwoord. Vervang de `...`.

        De check roept `open_domain("Hoe laat gaat de bibliotheek van het dorp op zaterdag dicht?", C.TEXTS["nl"])` aan, wat `"De bibliotheek van het dorp is op zaterdag dicht van 14.30 uur."` geeft, en hetzelfde in het Engels. Daarna eigen gevallen: andere vragen, en een lijst met een deel van de teksten.
        """),
        L.task("Beantwoord een vraag met de tekst die je retriever vindt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_open():
    box_open = L.code_box("open_domain", lines=4)
    box_open
    return (box_open,)


@app.cell(hide_code=True)
def check_open(box_open):
    result_open = L.check("open_domain", box_open.value)
    result_open
    return (result_open,)


@app.cell(hide_code=True)
def hints_open():
    L.hints(
        "open_domain",
        [
            "`best_text` geeft de plaats van een tekst in `texts`, voor de vraag `question`. Met `texts[...]` neem je de tekst op deze plaats, en `ask(question, ...)` geeft wat Qwen3 met deze tekst antwoordde.",
            "Twee stappen in één regel: eerst de plaats, dan de tekst op deze plaats. Wat komt er tussen de vierkante haken?",
            "`    return ask(question, texts[___])`",
        ],
        why="`return ask(question, texts[best_text(question, texts)])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_open():
    mo.md(r"""
    Met seed 0 heeft Qwen3 zonder tekst geen van de twaalf goed: het model geeft een spin 6 poten, laat de Eiffeltoren in 1903 klaar zijn, en geeft op elke vraag over het dorp een verzonnen of vaag antwoord. Met de tekst die je open_domain kiest, zijn er 7 van de 12 goed. Vier van de vijf foute antwoorden gaan mis op een som of een telling: Qwen3 schrijft dat een spin *meer poten* heeft dan een insect, zonder te zeggen hoeveel; dat een dag op Mercurius 176 aardse dagen duurt, zonder de jaren te tellen; dat de Eiffeltoren in 1989 *ongeveer 40 jaar oud* was; en dat het repaircafé *elke eerste zaterdag van de maand* samenkomt. Bij de vijfde, het zwembad, gaf je retriever de tekst over Mercurius, en Qwen3 verzon er een antwoord bij: *Het zwembad is open op 176 aardse dagen per jaar.* Met een tekst die de vraag niet beantwoordt, zegt Qwen3 in slechts 7 van de 240 antwoorden dat de tekst het antwoord niet geeft; in het Engels deed het model dat 49 keer, en zonder tekst nooit.
    """)
    return


@app.cell(hide_code=True)
def table_cell():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.md(r"""
    Over de vier seeds zijn zonder tekst 1 van de 24 antwoorden over de wereld goed, en 12 met de tekst; over het dorp 0 zonder tekst en 16 met de tekst. In het Engels waren dat er zonder tekst 12 over de wereld: in het Nederlands weet het kleine model veel minder uit het hoofd. De antwoorden die met de tekst fout blijven, horen bij de vijf vragen die een som of een telling vragen. Hier doet open-domain question answering het even goed als reading comprehension: je retriever mist alleen de vraag over het zwembad, en bij deze vraag gaat Qwen3 ook met de tekst over het zwembad de mist in.

    Roberts en anderen trainden T5, met ongeveer 11 miljard parameters, voor closed-book question answering: het model had 32,6% van de vragen van Natural Questions goed, en het beste systeem uit de vergelijking dat dingen opzoekt 41,5%. De auteurs noemen ook de prijs: het model verdeelt kennis op een onverklaarbare manier over de parameters, en hallucineert antwoorden die echt lijken als het onzeker is. Een systeem dat dingen opzoekt, kan meestal de tekst laten zien die het gebruikte; een closed-book model kan dat niet.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: elk antwoord uit de run van de course

    Kies een vraag en een seed, en zie wat Qwen3 schreef zonder tekst en met elk van de zes teksten. Zoek een tekst die de vraag niet beantwoordt, en kijk wat Qwen3 ermee doet.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_question = mo.ui.dropdown(options=C.question_options(L.lang), value="7. Hoe laat gaat de bibliotheek van het dorp op zaterdag dicht?", label="Vraag")
    try_seed = mo.ui.slider(0, 3, value=0, label="Seed", show_value=True)
    mo.hstack([try_question, try_seed], justify="start", gap=1.5, wrap=True)
    return try_question, try_seed


@app.cell(hide_code=True)
def try_look(try_question, try_seed):
    mo.Html(C.try_picture(L.lang, try_question.value, try_seed.value))
    return


@app.cell(hide_code=True)
def text_exercise_kind():
    mo.md(r"""
    ### Oefening 4. Welke soort?

    Een fietsenwinkel heeft een helpdeskbot. De bot krijgt een vraag van een klant, zoekt de pagina van het handboek van de winkel op die de meeste woorden met de vraag deelt, en laat de zin van deze pagina zien waar het antwoord in staat, met het antwoord gemarkeerd.
    """)
    return


@app.cell(hide_code=True)
def choice_kind_cell():
    which_kind = L.choice(
        "which_kind",
        question="Welke soort question answering doet de bot?",
        options={
            "closed": "Closed-book: de bot antwoordt uit wat het model leerde",
            "right": "Open-domain en extractive",
            "reading": "Reading comprehension: de klant geeft de bot de pagina",
            "abstractive": "Open-domain en abstractive: de bot vindt de pagina, en schrijft dan een antwoord in eigen woorden",
        },
        correct="right",
        why={
            "right": "De bot vindt de pagina zelf, in een verzameling: open-domain, met een retriever. De bot laat letterlijk een stuk van de pagina zien: extractive. Een klant kan het antwoord naast de pagina leggen, maar een verkeerde pagina geeft een verkeerde zin.",
            "closed": "De bot zoekt een pagina van het handboek op. Zoekt een closed-book systeem iets op?",
            "reading": "De klant stelt alleen de vraag. Wie vindt de pagina?",
            "abstractive": "De bot laat een zin van de pagina zien, met het antwoord gemarkeerd. Schrijft de bot zelf iets?",
        },
    )
    which_kind
    return (which_kind,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl) deelde je systemen in naar wat een systeem met taal doet; hier deelde je question answering in naar waar het antwoord vandaan komt en welke vorm het heeft. Waarom een model een antwoord schrijft dat het niet kan weten, legt [Waarom language models dingen verzinnen](/course/open/lm-hallucination?lang=nl) uit. De volgende les, *Het antwoord vinden in een passage*, gaat verder met reading comprehension: een dataset met vragen over alinea's van Wikipedia, en een eerste reader die de zin kiest met de meeste woorden van de vraag.

    **Een voorbeeld uit het dagelijks leven.** Een hulppagina die de alinea uit de handleiding laat zien waar het antwoord vandaan komt, laat je het antwoord in een paar seconden controleren; een antwoord zonder bron kun je alleen geloven, of niet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_open, result_pick, result_span, which_kind):
    L.wrap_up(result_span, result_pick, result_open, which_kind)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
