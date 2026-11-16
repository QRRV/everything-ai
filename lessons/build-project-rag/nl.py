# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-project-rag. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("build-project-rag", "nl")
    C = L.checks   # de runs van de course over het handboek van Spoke & Chain, de stand-ins, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl), en één over [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="Je assistent moet een ander model op de server gebruiken. Welk van zijn bestanden verander je?",
        options={
            "prompts": "`app/prompts.py`, waar de woorden van je assistent staan",
            "client": "`app/client.py`, die met de server praat",
            "answers": "`app/answers.py`, van waaruit `ask_json` vraagt",
        },
        correct="client",
        why={
            "client": "Je `chat` zet de naam van het model in elk request; de andere bestanden geven hem alleen berichten.",
            "prompts": "`app/prompts.py` bevat de woorden: `SYSTEM` en `TEMPLATES`. Welk bestand maakt het request dat het model noemt? Zie [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl).",
            "answers": "`ask_json` vraagt via `chat`, en noemt nooit een model. Zie [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def quick_check_grounded():
    quick_grounded = L.choice(
        "quick-grounded",
        question="Wat vraagt de grounded prompt van het model, naast alleen antwoorden uit de genummerde bronnen?",
        options={
            "copy": "De bron woord voor woord over te nemen, zodat niets verzonnen is",
            "all": "Elke bron minstens één keer te gebruiken, zodat er geen ontbreekt",
            "cite": "Het nummer van zijn bron te noemen, of I don't know te antwoorden",
        },
        correct="cite",
        why={
            "cite": "Met het nummer vindt je code de bron waar het antwoord vandaan komt; I don't know is het antwoord als geen bron het zegt.",
            "copy": "Hij vraagt om één zin, in de eigen woorden van het model. Zie [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).",
            "all": "De meeste vragen worden door één bron beantwoord; de andere zijn er alleen omdat de zoekfunctie ze vond. Zie [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl).",
        },
    )
    quick_grounded
    return (quick_grounded,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je assistent weet wat Qwen3 in de training leerde, en niets over je eigen documenten: het handboek van een winkel, je aantekeningen, de regels van je club. Sinds [Project: je assistent, goed geprompt](/course/open/build-project-prompts?lang=nl) heb je alles gebouwd wat nodig is om dat te veranderen: bronnen vóór de vraag in [Eerst opzoeken: wat RAG is](/course/open/build-rag-idea?lang=nl), chunks in [Documenten in stukken knippen](/course/open/build-chunking?lang=nl), embeddings in [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl), een zoekfunctie in [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl), en een prompt die het model bij zijn bronnen houdt en ze laat noemen in [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl). Vandaag komt het samen in je project: je assistent leest een map met documenten, antwoordt eruit, en zegt uit welk stuk van welk document het antwoord komt.

        **Een voorbeeld uit het dagelijks leven.** Een nieuwe collega bij een helpdesk kent de regels niet uit het hoofd. Bij elke vraag zoekt die de goede bladzijde op in de map, antwoordt daaruit, en zegt welke bladzijde het was, zodat de klant het ook kan lezen. Je assistent krijgt hetzelfde: de library is de map met zijn register, de zoekfunctie vindt de bladzijden, en het antwoord noemt zijn bladzijde. Dat is alleen een vergelijking; hieronder zie je wat Qwen3 ermee doet.
        """),
        draw.project_steps(["Je assistent antwoordt", "Je assistent, goed geprompt", "Je assistent leest je documenten"], 2,
                           here_label="← vandaag", title="Waar je project staat: je assistent"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_without():
    _f = C.findings()
    guess_without = L.guess(
        "without",
        question="Spoke & Chain is een fietsenwinkel die de course verzon, met een handboek van {words} woorden. De course stelde Qwen3-0.6B {k} vragen die het handboek beantwoordt, elk twee keer, zonder het handboek: alleen een system prompt die zegt dat het vragen over de winkel beantwoordt. Hoeveel van de {runs} antwoorden waren goed?".format(
            words=_f["words"], k=_f["questions"], runs=_f["runs"]),
        options={
            "none": "Geen enkel antwoord",
            "some": "Ongeveer een kwart: de prijzen van fietsenwinkels lijken op elkaar",
            "most": "De meeste: een model heeft veel over fietsenwinkels gelezen",
        },
        reveal="**Geen enkel antwoord: {plain} van de {runs}.** Qwen3 verzon prijzen in dollars, zoals een borg van *around $25 to $50*, en bij de twee vragen die het handboek niet beantwoordt een telefoonnummer, *555-1234*, en een straat. Geen enkele keer zei het dat het het niet wist. Geen wonder: de winkel en zijn handboek zijn van de course zelf, dus ze zaten in niets waar Qwen3 van leerde.".format(
            plain=_f["plain"], runs=_f["runs"]),
    )
    guess_without
    return (guess_without,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Twee nieuwe bestanden, bovenop zes die je maakte

        Na vandaag heeft je package `app` twee bestanden meer, elk met één taak:

        - `app/library.py` heeft `make_library(url, chunks, about)`. Die maakt met je `embed` de embeddings van de chunks van je documenten, en geeft je library terug: een dictionary met `"about"`, een paar woorden over waar de documenten over gaan; `"chunks"`, de chunks als teksten; en `"vectors"`, een NumPy-array met één rij per chunk, zodat rij `i` de embedding van `chunks[i]` is.
        - `app/rag.py` heeft `ask(url, library, question, k=3)`. Die maakt de embedding van de vraag, vindt de `k` chunks met je `meaning_search`, vraagt met de grounded prompt van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) en de bronnen van je `with_sources`, via je `chat`, en houdt met je `cited` de chunks die het antwoord noemt. Hij geeft het antwoord terug, de gevonden chunks als `"sources"`, en de chunks die het antwoord noemt als `"named"`.

        De andere zes komen uit eerdere lessen, waarvan de checks je code als bestand bewaarden. Waar er een ontbreekt of kapot is, springt de versie van de course in, en de melding bij *Waar je project staat* zegt welke.
        """),
        mo.Html(C.files_picture(L.lang)),
        mo.md(r"""
        De grounded prompt is een template met één veld, `{{about}}`: de prompt van de course ging over de course, en die van jou gaat over wat je library zegt dat erin staat. En `make_library` stuurt niet alle chunks in één request. De server van de course neemt hooguit 2.048 teksten aan in één request, net als die van OpenAI, en antwoordt op meer met status 400, het getal dat een server terugstuurt bij een request dat hij weigert; een map met een paar boeken heeft meer chunks dan dat. Dus stuurt hij `BATCH` chunks tegelijk, 64, zoals de course deed toen hij de embeddings maakte van de 1.266 punten van zijn samenvattingen: {seconds} seconden in totaal, op de computer van de course.
        """.format(seconds=C.number(C.batch_seconds(), L.lang, 1))),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: je assistent leest je documenten

    1. **Knippen:** elk document in chunks met je `chunk` van [Documenten in stukken knippen](/course/open/build-chunking?lang=nl): 80 woorden die er 20 overlappen.
    2. **De library:** `make_library(url, chunks, about)` stuurt de chunks `BATCH` tegelijk naar `embed`, voegt de embeddings van elke batch toe aan één lijst, en geeft `"about"`, `"chunks"` en de embeddings als array, `"vectors"`, terug.
    3. **Zoeken:** `ask(url, library, question, k=3)` maakt de embedding van de vraag, en vindt de `k` chunks waarvan de embedding het hoogste inproduct ermee heeft.
    4. **Vragen:** twee berichten via `chat`: de system prompt `GROUNDED`, ingevuld met de `"about"` van de library, en de vraag met de chunks als genummerde bronnen, van `with_sources`.
    5. **Noemen:** `cited` vindt de nummers in het antwoord; elk nummer `n` van 1 tot en met het aantal bronnen is de chunk `passages[n - 1]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    _f = C.findings()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De run van de course met seed 0, op het handboek van Spoke & Chain: `ask(url, library, "{question}")`.

    1. **Knippen:** het handboek, {words} woorden, wordt {chunks} chunks van 80 woorden, de laatste van {last}.
    2. **De library:** {chunks} chunks passen in één request, dus `make_library` stuurt er één, en krijgt {chunks} embeddings terug.
    3. **Zoeken:** de chunk met het hoogste inproduct begint met *{start}*, en bevat *cash up to 500 euros*. Die wordt bron 1; bron 2 en 3 zijn de twee chunks die er in het handboek vlak vóór staan.
    4. **Vragen:** de system prompt zegt dat het model vragen beantwoordt over de fietsenwinkel Spoke & Chain, alleen met de genummerde bronnen; het bericht van de gebruiker bevat de drie chunks, en dan *Question: {question}*
    5. **Noemen:** Qwen3 antwoordde *{answer}* `cited` vindt `{cited}`, dus `named` bevat bron 1: de chunk met het antwoord, die je app ernaast kan laten zien. Zonder het handboek antwoordde Qwen3 *{plain}*
    """.format(question=_w["question"], words=_f["words"], chunks=_f["chunks"], last=len(C.CHUNKS[-1].split()),
               start=" ".join(_w["passages"][0].split()[:12]) + " …", answer=_w["answer"], cited=_w["cited"], plain=_w["plain"]))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        ### Wat de course mat

        De course stelde alle {k} vragen over het handboek, en {none} die het niet beantwoordt, elk twee keer: één keer met alleen de system prompt over de winkel, en één keer zoals je `ask` het doet, met de library van het handboek.
        """.format(k=_f["questions"], none=_f["none"])),
        mo.Html(C.runs_picture(L.lang)),
        mo.md(r"""
        Met je `ask` waren {yours} van de {runs} antwoorden goed, tegen {plain} zonder het handboek, en op de vragen die het handboek niet beantwoordt, zei het in {yours_none} van de {none_runs} runs *I don't know*. Elk goed antwoord noemde een chunk met het antwoord erin.

        Waar het misging, zegt meer. Zoeken op betekenis vond voor {hits} van de {k} vragen een chunk met het antwoord. Het miste de chunk met de volledige onderhoudsbeurt bij beide vragen erover, ook bij *How much does a full service cost?*, en dan zei het model geen *I don't know*: het nam een andere prijs uit de chunks die het kreeg, zoals *120 euros*, de onderhoudsbeurt van elektrische fietsen. Zoeken op woorden, de BM25 van [Zoeken op woorden: BM25](/course/open/build-bm25?lang=nl), zet bij die vraag de chunk met de volledige onderhoudsbeurt wél bovenaan. Waar de zoekfunctie de chunk wel vond, waren {right_found} van de {runs_found} antwoorden goed: het model haalde getallen binnen één chunk door elkaar, zoals *21:00* van de reparatieavond voor de sluitingstijd op zaterdag, en zei {refused} keer *I don't know* terwijl de chunk het antwoord had. De {other_named} foute antwoorden daar noemden allemaal de chunk met het goede antwoord, dus een lezer die de genoemde bron opent, ziet de fout.
        """.format(yours=_f["yours"], runs=_f["runs"], plain=_f["plain"], yours_none=_f["yours_none"], none_runs=_f["none_runs"],
                   hits=_f["hits"], k=_f["questions"], right_found=_f["right_found"], runs_found=_f["runs_found"],
                   refused=_f["refused"], other_named=_f["other_named"])),
    ])
    return


@app.cell(hide_code=True)
def text_library():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je library

        `make_library(url, chunks, about)` moet je library teruggeven, met de embeddings van alle chunks op hun volgorde. Het vak stuurt alle chunks in één request. Stuur ze in plaats daarvan `BATCH` tegelijk, één request per batch, en voeg de embeddings van elke batch toe aan één lijst. De check roept hem aan met een stand-in van de server, die antwoordt met verzonnen embeddings, met 150, 64 en 65 chunks. Als hij goed is, bewaart de check hem in je project, als `work/projects/app/library.py`.
        """),
        L.task("Stuur de chunks `BATCH` tegelijk, en voeg de embeddings van elke batch toe aan `vectors`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_library():
    box_library = L.code_box("make_library", lines=13)
    box_library
    return (box_library,)


@app.cell(hide_code=True)
def check_library(box_library):
    result_library = L.check("make_library", box_library.value)
    result_library
    return (result_library,)


@app.cell(hide_code=True)
def hints_library():
    L.hints(
        "make_library",
        [
            "Begin met een lege lijst, en vul die in een loop: één request per batch van `BATCH` chunks.",
            "`for start in range(0, len(chunks), BATCH):` geeft 0, 64, 128, …; de batch is `chunks[start:start + BATCH]`.",
            "In de loop: `vectors = vectors + embed(url, ...)`, met de batch als de teksten. `+` maakt van twee lijsten één: `[a, b] + [c]` is `[a, b, c]`.",
        ],
        why="""```python
def make_library(url, chunks, about):
    vectors = []
    for start in range(0, len(chunks), BATCH):
        vectors = vectors + embed(url, chunks[start:start + BATCH])
    return {"about": about, "chunks": chunks, "vectors": np.array(vectors)}
```

Het laatste stuk kan korter zijn: `chunks[128:192]` van 150 chunks heeft er 22.""",
    )
    return


@app.cell(hide_code=True)
def text_ask():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je assistent antwoordt uit de library

        `ask(url, library, question, k=3)` maakt al de embedding van de vraag en vindt de passages. Maar hij stuurt het model alleen de vraag, en noemt geen bron. Maak de twee berichten: de system prompt `GROUNDED` ingevuld met `library["about"]`, en als bericht van de gebruiker wat `with_sources` maakt van de vraag en de passages. Houd daarna in `named` de passages die het antwoord noemt: voor elk nummer `n` dat `cited(answer)` vindt `passages[n - 1]`, maar alleen als `n` van 1 tot en met `len(passages)` is. De check roept hem aan met een stand-in die met een vaste tekst antwoordt, eerst met de library van het handboek en de vraag over contant betalen. Als hij goed is, bewaart de check hem in je project, als `work/projects/app/rag.py`.
        """),
        L.task("Stuur de system prompt en de vraag met zijn bronnen via `chat`, en houd in `named` de passages die het antwoord noemt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ask():
    box_ask = L.code_box("ask", lines=24)
    box_ask
    return (box_ask,)


@app.cell(hide_code=True)
def check_ask(box_ask):
    result_ask = L.check("ask", box_ask.value)
    result_ask
    return (result_ask,)


@app.cell(hide_code=True)
def hints_ask():
    L.hints(
        "ask",
        [
            "Het is het request van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl), met `{about}` ingevuld, en daarna een loop over de nummers die het antwoord noemt.",
            "Twee berichten in een lijst: eerst de system prompt, `GROUNDED` met `.format` en `library[\"about\"]` voor zijn veld, dan dat van de gebruiker, dat `with_sources(question, passages)` maakt. Stuur de lijst met `chat`.",
            "Na `answer = chat(url, messages)`: `for n in cited(answer):`, en `named.append(passages[n - 1])` als `n >= 1 and n <= len(passages)`.",
        ],
        why="""```python
    messages = [{"role": "system", "content": GROUNDED.format(about=library["about"])},
                {"role": "user", "content": with_sources(question, passages)}]
    answer = chat(url, messages)
    named = []
    for n in cited(answer):
        if n >= 1 and n <= len(passages):
            named.append(passages[n - 1])
```

Een antwoord kan een nummer noemen dat geen bron is, zoals `[4]` bij drie bronnen: `passages[3]` zou stoppen met een IndexError, en `passages[-1]`, bij `[0]`, zou de laatste bron zijn: een negatieve index telt vanaf het eind.""",
    )
    return


@app.cell(hide_code=True)
def text_new_document():
    mo.md(r"""
    ### Oefening 3. Een nieuw document
    """)
    return


@app.cell(hide_code=True)
def choice_new_document_cell():
    choice_new_document = L.choice(
        "new-document",
        question="Je zet een nieuw document in de map van je assistent. Wat moet er een embedding krijgen voordat `ask` zijn chunks kan vinden?",
        options={
            "new": "Alleen de chunks van het nieuwe document",
            "all": "Alle chunks opnieuw, omdat elke embedding van de andere afhangt",
            "nothing": "Niets: `ask` maakt de embedding van de vraag, en dat is genoeg om ze te vinden",
        },
        correct="new",
        why={
            "new": "De embedding van een chunk hangt alleen af van zijn eigen tekst, dus de oude rijen blijven zoals ze zijn: voeg de nieuwe chunks en hun rijen toe. De demo hieronder maakt gewoon de hele library opnieuw als er een document verandert, en dat is prima voor een paar documenten.",
            "all": "Verandert de embedding van een chunk als er een andere chunk bij komt? Kijk wat `embed` krijgt: een lijst met teksten, en één embedding per tekst terug.",
            "nothing": "De zoekfunctie vergelijkt de vraag met de rijen van de library. Een chunk zonder rij wordt nooit vergeleken, dus hoe zou die gevonden worden?",
        },
    )
    choice_new_document
    return (choice_new_document,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Wat je project nu kan

    De cel neemt `make_library` en `ask` uit je project, en leest de documenten in `work/projects/app/documents/`: de bestanden die eindigen op `.md` of `.txt`. De eerste keer zet de course er zijn handboek van Spoke & Chain in. Zet er je eigen bestanden bij, of haal het handboek eruit. De cel knipt elk document met je `chunk`, maakt de library met je `make_library`, en bewaart die tot de documenten veranderen. Daarna stelt je `ask` Qwen3 je vraag. Alles draait op je eigen computer: je documenten gaan naar niemand. Schrijf in het vak waar je documenten over gaan, in het Engels, omdat het in de system prompt komt.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B en Granite Embedding (1,7 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*De demo heeft twee modellen nodig die nog niet allebei op je computer staan: het chat model Qwen3-0.6B, en Granite Embedding, het embedding model. De knop downloadt wat ontbreekt één keer van Hugging Face, op vaste versies: `Qwen/Qwen3-0.6B`, 1,5 GB, met de tokenizer van Qwen3-0.6B-Base, en `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, allebei onder de Apache-2.0-licentie.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B en Granite Embedding zijn gedownload.*")
    return


@app.cell(hide_code=True)
def question_cell():
    about_box = mo.ui.text(value=C.ABOUT, label="Waar je documenten over gaan, in het Engels", full_width=True)
    question_box = mo.ui.text_area(value="Tot welk bedrag kan ik contant betalen?", label="Je vraag", full_width=True, rows=2)
    ask_button = mo.ui.run_button(label="Vraag het je assistent")
    mo.vstack([about_box, question_box, ask_button])
    return about_box, ask_button, question_box


@app.cell(hide_code=True)
def demo_answer(about_box, ask_button, question_box, result_ask, result_library):
    _docs = C.documents()
    _names = ", ".join("`{name}` ({n} woorden)".format(name=_name, n=len(_text.split())) for _name, _text in _docs)
    _folder = mo.md("*In `work/projects/app/documents/`: {names}.*".format(names=_names) if _docs else
                    "*Er staan geen `.md`- of `.txt`-bestanden in `work/projects/app/documents/`. Zet er een document in, en druk opnieuw op de knop.*")
    if not C.downloaded():
        _out = mo.md("*Download eerst de twee modellen, met de knop hierboven.*")
    elif result_library.state != "correct" or result_ask.state != "correct":
        _out = mo.md("*Maak eerst oefening 1 en 2 af: dan staan je `make_library` en `ask` in je project, en gebruikt deze cel ze.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    elif not _docs:
        _out = mo.md("")
    else:
        _make, _why = C.learner("library", "make_library")
        _ask, _why_ask = C.learner("rag", "ask")
        _chunk, _why_chunk = C.learner("chunks", "chunk")
        if _make is None or _ask is None or _chunk is None:
            _out = mo.md("*Je project draait niet zoals het nu is: {why}*".format(why=_why or _why_ask or _why_chunk))
        else:
            _words = {"title": "Wat je assistent antwoordde: {question}", "embedding": "Je make_library maakt de embeddings van {n} chunks van {d} documenten …",
                      "asking": "Je ask zoekt, en Qwen3 schrijft …",
                      "library": "Library: {n} chunks van {d} documenten, gemaakt in {seconds} s; geantwoord in {asked} s.",
                      "answer": "Het antwoord:", "named": "Het noemt: {names}", "none": "Het noemt geen bron.",
                      "sources": "De bronnen die het vond, de bronnen die het noemt gemarkeerd:", "yours": "Alle modules waren van jou.",
                      "course": "De versie van de course viel in voor: {files}.", "failed": "Je assistent stopte met een fout tegen de echte server: {why}",
                      "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
            mo.output.replace(mo.md("*Je assistent leest je documenten …*"))
            _out = mo.Html(C.demo(_make, _ask, _chunk, C.documents(), about_box.value.strip() or C.ABOUT, question_box.value.strip(),
                                  lambda _t: mo.output.replace(mo.md("*" + _t + "*")), _words, L.lang))
    mo.vstack([_folder, _out])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De bronnen van [Eerst opzoeken: wat RAG is](/course/open/build-rag-idea?lang=nl), de chunks van [Documenten in stukken knippen](/course/open/build-chunking?lang=nl), de embeddings van [Embeddings en documenten via een API](/course/open/build-embeddings-api?lang=nl), de zoekfunctie van [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl) en de grounded prompt van [Antwoorden uit de bronnen](/course/open/build-grounded-answers?lang=nl) werken nu samen in je project, via de client van [Project: je assistent antwoordt](/course/open/build-project-call?lang=nl). De metingen lieten zien waar het tekortschiet: een antwoord kan maar zo goed zijn als de chunks die de zoekfunctie vindt. De volgende lessen van deze module maken het zoeken beter: sneller met een index, met filters, op woorden en betekenis samen, en met een tweede blik op de gevonden chunks; een latere les meet een RAG-systeem als geheel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new_document, result_ask, result_library):
    L.wrap_up(result_library, result_ask, choice_new_document)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
