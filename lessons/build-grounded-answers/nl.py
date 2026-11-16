# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-grounded-answers. Route: course/route/lessen/bouwen.yaml. Checks, the recorded runs and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-grounded-answers", "nl")
    C = L.checks   # de knowledge base, de opgenomen antwoorden van Qwen3, de checks, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_once():
    quick_once = L.choice(
        "quick-once",
        question="Een RAG-systeem doorzoekt 10.000 chunks op betekenis, en krijgt 100 vragen. Van hoeveel teksten maakt het in totaal een embedding?",
        options={
            "again": "1.000.100: de vraag en alle chunks, bij elke vraag",
            "once": "10.100: elke chunk één keer, en elke vraag",
            "questions": "100: alleen de vragen, want de chunks zijn tekst",
        },
        correct="once",
        why={
            "once": "De embeddings van de chunks hangen niet af van de vraag, dus worden ze één keer gemaakt: 10.000. Elke vraag heeft dan één eigen embedding nodig: 100 erbij.",
            "again": "Dat werkt, maar waarom de embeddings van de chunks opnieuw maken als ze niet veranderd zijn? Zie [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl).",
            "questions": "Zoeken op betekenis vergelijkt de embedding van de vraag met de embeddings van de chunks, dus de chunks hebben er ook een nodig. Zie [Zoeken op betekenis](/course/open/build-retrieval-embeddings?lang=nl).",
        },
    )
    quick_once
    return (quick_once,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    In [Je eerste RAG](/course/open/build-rag-first?lang=nl) kreeg Qwen3 bronnen in zijn prompt, maar niets zei dat het ze moest gebruiken. Met de drie punten die zoeken op betekenis vindt en de system prompt van die les waren {plain} van zijn {runs} antwoorden goed. *{sam}* Een van de bronnen zegt: in 8 van de 8 runs met de eerdere beurt, in 0 van de 8 zonder. Qwen3 antwoordde: *{plain_sam}* En op een vraag die geen punt beantwoordt, verzon het iets: *{city}* Aan zo'n antwoord zie je niet of het uit een bron kwam, of uit welke.

    **Een voorbeeld uit het dagelijks leven.** Bij een tentamen met open boek vraagt de docent bij elk antwoord de bladzijde. Wie de bladzijde moet opschrijven, zoekt het op in plaats van te gokken, de docent kijkt het antwoord in een oogwenk na, en een vraag waar het boek geen antwoord op geeft, krijgt *staat niet in het boek*.
    """.format(plain=_f["plain"], runs=_f["runs"], sam=C.question("q05")["text"]["en"], plain_sam=C.answers("q05", "plain")[0],
               city=C.answers("q27", "plain")[1]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **grounded generation** | een antwoord alleen schrijven uit de bronnen die het model kreeg, met het nummer van de bron, en het zeggen als ze de vraag niet beantwoorden |
    | `text.split("[")` | de stukken van `text` tussen de `[`'s: `"a [1] b".split("[")` geeft `["a ", "1] b"]` |
    | `inside.isdigit()` | `True` als `inside` alleen cijfers heeft, minstens één: `"12".isdigit()` is `True`, `"a".isdigit()` en `"".isdigit()` zijn `False` |
    | `a or b` | `True` als minstens een van de twee `True` is |

    In [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl) antwoordde Qwen3 uit één tekst die je het gaf, met een citatie: grounding. In een RAG-systeem zijn de teksten de bronnen die de zoekfunctie vond, genummerd zoals `with_sources` uit [Eerst opzoeken: wat RAG is](/course/open/build-rag-idea?lang=nl) ze in de prompt zet. Een grounded system prompt vraagt drie dingen: antwoord alleen uit de genummerde bronnen; eindig met het nummer van de bron, tussen vierkante haken; en als de bronnen geen antwoord geven, zeg dan alleen *I don't know*. Dit is die van de course:

    > {grounded}

    Door het nummer kun je een antwoord nakijken: je zoekt de bron op die het noemt, en kijkt of die dat zegt. Een programma kan een deel ervan doen: de nummers tussen haken vinden, en kijken of elk getal in het antwoord in de bronnen staat die het noemt.
    """.format(grounded=C.GROUNDED))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: antwoorden uit de bronnen

    1. **Zoek:** de `k` chunks die het best bij de vraag passen.
    2. **Nummer:** zet ze in de prompt als [1], [2], [3], met de vraag erachter.
    3. **Instrueer:** een system prompt die vraagt om een antwoord alleen uit de bronnen, met het nummer van zijn bron, en alleen *I don't know* als ze geen antwoord geven.
    4. **Kijk na:** vind de nummers tussen haken in het antwoord, en kijk of elk getal erin in de bronnen staat die het noemt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    *{question}*

    1. **Zoek:** op betekenis komen de drie beste punten uit *{t1}*, *{t2}* en *{t3}*.
    2. **Nummer:** bron [{n}] is het punt van *{tn}*: *{source}*
    3. **Instrueer:** met de gewone system prompt antwoordde Qwen3 *{plain}* Met de grounded prompt: *{grounded}*
    4. **Kijk na:** het antwoord noemt bron {n}. Zijn woorden met een cijfer erin zijn {numbers}, en alle {k} staan in bron {n}: de controle zegt `True`.

    Bron {n} heeft twee getallen voor de runs, 8 van de 8 en 0 van de 8, en het gewone antwoord pakte het verkeerde. Had het bron {n} genoemd, dan had een controle die alleen naar de getallen kijkt, er ook `True` op gezegd.
    """.format(question=_w["question"], t1=C.lesson_of(_w["passages"][0])["title"][L.lang], t2=C.lesson_of(_w["passages"][1])["title"][L.lang],
               t3=C.lesson_of(_w["passages"][2])["title"][L.lang], n=_w["cited"][0], tn=C.lesson_of(_w["source"])["title"][L.lang],
               source=_w["source"], plain=_w["plain"], grounded=_w["grounded"],
               numbers=", ".join("*" + x + "*" for x in _w["numbers"][:-1]) + " en *" + _w["numbers"][-1] + "*", k=len(_w["numbers"])))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course stelde Qwen3 alle {all} vragen over de course, elk met de drie punten die zoeken op betekenis vindt: twee keer met de gewone system prompt, en twee keer met de grounded prompt. De bronnen zijn dezelfde; alleen de system prompt verschilt.
    """.format(all=len(C.QUESTIONS)))
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.Html(C.runs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Met de grounded prompt waren {grounded} van de {runs} antwoorden op de {questions} vragen die een punt beantwoordt goed; met de gewone prompt {plain}. Elk grounded antwoord noemde een bron; van de gewone antwoorden {plain_cites}. Op de {none} vragen die geen punt beantwoordt, antwoordde de grounded prompt in {gnone} van de {nruns} runs *I don't know*. Met de gewone prompt zeiden {pnone} van de {nruns} dat ze niet konden antwoorden; de andere antwoordden toch.
        """.format(grounded=_f["grounded"], runs=_f["runs"], questions=_f["questions"], plain=_f["plain"], plain_cites=_f["plain_cites"],
                   none=_f["none"], gnone=_f["grounded_none"], nruns=_f["none_runs"], pnone=_f["plain_none"])),
        L.watch_out("Een nummer tussen haken is geen bewijs. *{q}* Bron 1 zegt dat een zin in het Engels 6 tokens is, en dezelfde zin in het Nederlands 15. Beide grounded runs antwoordden met [1]: *{a}*".format(
            q=C.question("q23")["text"]["en"], a=C.answers("q23", "grounded")[0].strip())),
    ])
    return


@app.cell(hide_code=True)
def text_cited():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De bronnen die een antwoord noemt

        `cited(answer)` moet de nummers geven van de bronnen die een antwoord noemt: de hele getallen tussen vierkante haken, in de volgorde waarin ze komen, elk één keer. Voor *{example}* is dat `[2]`. Haken met iets anders erin, zoals `[a]` of `[1, 2]`, noemen geen bron: de grounded prompt vraagt om één nummer. Het vak knipt het antwoord bij elke `[`, en houdt van elk stuk wat er vóór de volgende `]` staat. Als hij goed is, bewaart de check hem ook als bestand, `work/projects/app/citations.py`: [Project: je assistent leest je documenten](/course/open/build-project-rag?lang=nl) gebruikt hem.
        """.format(example=C.CITED_EXAMPLE)),
        L.task("Houd een stuk alleen als het een getal is, als heel getal, en maar één keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cited():
    box_cited = L.code_box("cited", lines=6)
    box_cited
    return (box_cited,)


@app.cell(hide_code=True)
def check_cited(box_cited):
    result_cited = L.check("cited", box_cited.value)
    result_cited
    return (result_cited,)


@app.cell(hide_code=True)
def hints_cited():
    L.hints(
        "cited",
        [
            "`inside.isdigit()` zegt of een stuk alleen cijfers heeft; `int(inside)` maakt er een heel getal van.",
            "Voeg een getal alleen toe als het nog niet in `found` zit: `int(inside) not in found`.",
            "`if inside.isdigit() and int(inside) not in found:`, en daaronder `found.append(int(inside))`.",
        ],
        why="""```python
def cited(answer):
    found = []
    for part in answer.split("[")[1:]:
        inside = part.split("]")[0]
        if inside.isdigit() and int(inside) not in found:
            found.append(int(inside))
    return found
```

`[1:]` slaat het stuk vóór de eerste `[` over. En `and` stopt bij de eerste `False`: als `inside` geen getal is, komt Python nooit bij `int(inside)`, dat met een fout zou stoppen.""",
    )
    return


@app.cell(hide_code=True)
def text_supported():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Staan zijn getallen in de bronnen die het noemt?

        `supported(answer, passages)` moet alleen `True` geven als het antwoord minstens één bron noemt, elke bron die het noemt er is, en elk getal in het antwoord in de bronnen staat die het noemt. `passages` zijn de bronnen in de volgorde van de prompt, dus bron 1 is `passages[0]`. `cited` en `numbers` zijn gegeven: `numbers(text)` geeft de woorden van een tekst met een cijfer erin, zonder de citaties, zoals `{{'0', '8', 'qwen3'}}` voor het antwoord hierboven. De check gebruikt vier bronnen van een fietsenwinkel, `SHOP`:

        {shop}

        Het vak zet de bronnen die het antwoord noemt samen in `text`, maar zegt daarna `True`, wat er ook in staat.
        """.format(shop="\n".join("{i}. *{t}*".format(i=i, t=t) for i, t in enumerate(C.SHOP, start=1)))),
        L.task("Geef `False` terug als het antwoord een bron noemt die er niet is, of een getal heeft dat niet bij de getallen van `text` zit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_supported():
    box_supported = L.code_box("supported", lines=12)
    box_supported
    return (box_supported,)


@app.cell(hide_code=True)
def check_supported(box_supported):
    result_supported = L.check("supported", box_supported.value)
    result_supported
    return (result_supported,)


@app.cell(hide_code=True)
def hints_supported():
    L.hints(
        "supported",
        [
            "Twee controles: in de loop, vóór `passages[n - 1]`, of bron `n` er is; na de loop elk getal van het antwoord tegen `numbers(text)`.",
            "Bron `n` is er als `n` van 1 tot en met `len(passages)` is, dus `n < 1 or n > len(passages)` betekent dat hij er niet is. Ga na de loop door `numbers(answer)`.",
            "In de loop: `if n < 1 or n > len(passages):` en `return False`. Erna: `for number in numbers(answer):`, `if number not in numbers(text):`, `return False`, en `return True` aan het eind.",
        ],
        why="""```python
def supported(answer, passages):
    found = cited(answer)
    if len(found) == 0:
        return False
    text = ""
    for n in found:
        if n < 1 or n > len(passages):
            return False
        text = text + " " + passages[n - 1]
    for number in numbers(answer):
        if number not in numbers(text):
            return False
    return True
```

Vergelijk getallen met getallen, niet met de tekst: als stukje tekst zit *1* in *15*. Het plaatje draait je functie op de grounded antwoorden van Qwen3.""",
    )
    return


@app.cell(hide_code=True)
def text_named_source():
    mo.md(r"""
    ### Oefening 3. Een getal uit de bron

    Elk getal in dit antwoord staat in de bron die het noemt.
    """)
    return


@app.cell(hide_code=True)
def choice_named_source_cell():
    _f = C.findings()
    choice_named_source = L.choice(
        "named-source",
        question="Op *{q}* antwoordde Qwen3 met de grounded prompt: *{a}* Bron 3 is: *{s}* Dus zegt `supported` `True`. Wat laat dat zien?".format(
            q=C.question("q01")["text"]["en"], a=C.answers("q01", "grounded")[0].strip(), s=C.found(C.question("q01"))[2]),
        options={
            "right": "Dat het antwoord goed is: het getal staat in de bron die het noemt",
            "search": "Dat de zoekfunctie het goede punt vond, als bron 3",
            "about": "Alleen dat 34.090 in bron 3 staat, die over een ander CNN gaat",
        },
        correct="about",
        why={
            "about": "Bron 3 is een punt van *{t}*: een ander CNN, met een bottleneck, van 34.090 parameters. Het kleine CNN van de course heeft er 9.098, en zijn punt zat niet bij de drie. Een controle op getallen ziet of een getal uit de bronnen komt, niet of het de vraag beantwoordt: van de {other} grounded antwoorden die niet goed waren, liet hij alle {other} door.".format(
                t=C.lesson_of(C.found(C.question("q01"))[2])["title"][L.lang], other=_f["other_supported"]),
            "right": "Lees bron 3 nog eens: welk CNN heeft daar 34.090 parameters, en is dat het kleine CNN van de course?",
            "search": "Bij deze vraag miste zoeken op betekenis het punt dat hem beantwoordt: het zit niet bij de drie. Over welk CNN gaat bron 3?",
        },
    )
    choice_named_source
    return (choice_named_source,)


@app.cell(hide_code=True)
def text_refused():
    mo.md(r"""
    ### Oefening 4. I don't know

    De grounded prompt vraagt om *I don't know* als de bronnen geen antwoord geven.
    """)
    return


@app.cell(hide_code=True)
def choice_refused_cell():
    _f = C.findings()
    choice_refused = L.choice(
        "refused",
        question="Bij *{q}* zegt bron 1: *{s}* Een van de twee grounded runs antwoordde: *{a}* Wat laat dat zien?".format(
            q=C.question("q08")["text"]["en"], s=C.found(C.question("q08"))[0], a=C.answers("q08", "grounded")[1].strip()),
        options={
            "missed": "Dat de zoekfunctie het punt met het antwoord miste, dus Qwen3 had niets",
            "percent": "Dat Qwen3 een deel niet kan lezen als het een percentage is, zoals 21,7%",
            "price": "Dat de strenge prompt soms weigert waar een bron wel antwoordt",
        },
        correct="price",
        why={
            "price": "Bron 1 heeft het antwoord, en kwam als eerste. Een prompt die *I don't know* een veilig antwoord maakt, krijgt het af en toe waar het niet hoort: in {refused} van de {runs} grounded antwoorden op de vragen die een punt beantwoordt. Daartegenover zei hij *I don't know* in {gnone} van de {nruns} runs waar geen punt antwoordt, waar de gewone prompt er in {pnone} zei dat hij niet kon antwoorden. Of één weigering te veel dat waard is, hangt af van wat een fout antwoord kost.".format(
                refused=_f["refused"], runs=_f["runs"], gnone=_f["grounded_none"], nruns=_f["none_runs"], pnone=_f["plain_none"]),
            "missed": "Bron 1 is het punt dat hem beantwoordt: 21,7% staat erin. Waarom zou Qwen3 dan toch zeggen dat het het niet weet?",
            "percent": "Qwen3 las 95% goed in allebei zijn antwoorden op een andere vraag. Wat in de system prompt maakt *I don't know* een antwoord dat het mag geven?",
        },
    )
    choice_refused
    return (choice_refused,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Vraag Qwen3 met je eigen system prompt

    Zodra oefening 1 en 2 goed zijn, kun je Qwen3 een van de vragen van de course stellen, met de drie punten die zoeken op betekenis vindt, na een system prompt van jezelf. De vraag gaat in het Engels naar Qwen3, net als de bronnen. Verander de grounded prompt, of laat een regel weg, en kijk wat je `cited` en `supported` van het antwoord zeggen.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de licentie Apache-2.0, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het binnengekomen bestand is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options={q["text"][L.lang]: q["id"] for q in C.QUESTIONS}, value=C.question("q05")["text"][L.lang],
                                   label="Vraag", full_width=True)
    system_text = mo.ui.text_area(value=C.GROUNDED, label="System prompt", full_width=True, rows=4)
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([question_pick, system_text, ask_button])
    return ask_button, question_pick, system_text


@app.cell(hide_code=True)
def demo_answer(ask_button, box_cited, box_supported, question_pick, system_text):
    _checks = C.learner_checks(box_cited.value, box_supported.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _checks is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo kijkt het antwoord na met je eigen `cited` en `supported`.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    else:
        _words = {"title": "Qwen3 met je system prompt: {question}", "sources": "De drie bronnen in de prompt, met de les van elk:",
                  "answer": "Het antwoord van Qwen3:", "loading": "Qwen3 wordt geladen …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden.", "checked": "Je cited: {cited}. Je supported: {supported}.",
                  "failed": "Je functies stopten met een fout op dit antwoord: {why}",
                  "load": "Qwen3 kon niet geladen worden. Download het opnieuw, met de knop hierboven.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_checks, question_pick.value, system_text.value, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Project: je assistent leest je documenten](/course/open/build-project-rag?lang=nl) doet je assistent het allemaal: hij knipt documenten in chunks, doorzoekt ze, en antwoordt uit de bronnen met het nummer van zijn bron. Een controle op getallen is een eerste zeef. Of een bron zegt wat het antwoord beweert, ziet een mens, of een tweede model dat het moet beoordelen, in een latere les.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_named_source, choice_refused, result_cited, result_supported):
    L.wrap_up(result_cited, result_supported, choice_named_source, choice_refused)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
