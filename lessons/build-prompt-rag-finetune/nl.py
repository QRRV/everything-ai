# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-prompt-rag-finetune. Route: course/route/lessen/bouwen.yaml. Controles, de antwoorden van beide versies van de bank en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-prompt-rag-finetune", "nl")
    C = L.checks              # de antwoorden van beide versies, wat de eval set en de notities van de course zeggen, en de plaatjes
    SYSTEM = C.SYSTEM         # de system prompt van versie 2: de feiten van de zes onderwerpen en de vorm van het antwoord
    QUESTIONS = C.QUESTIONS   # de 120 testvragen van de bank
    V1 = C.V1                 # per vraag, versie 1: {"reply", "passed" (de eval set), "notes" (van de course, met de hand)}
    V2 = C.V2                 # hetzelfde voor versie 2


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Waar een model draait](/course/open/build-where-models-run?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="Een open-weight model dat je op je eigen computer draaide, draait nu bij een bedrijf dat het voor je host. Wat verandert er in je code?",
        options={
            "api": "De vorm van elk request en elk antwoord",
            "weights": "Je downloadt eerst de weights van het model",
            "address": "De base URL en de naam van het model",
        },
        correct="address",
        why={
            "address": "Het request houdt zijn vorm: alleen de base URL en de naam van het model veranderen, plus de key die een aanbieder vraagt. Een model dat een aanbieder voor je finetunet, roep je op dezelfde manier aan, onder een eigen naam.",
            "api": "Lokale runtimes en aanbieders spreken dezelfde API, dus requests en antwoorden houden hun vorm. Alleen de base URL en de naam van het model veranderen. Zie [Waar een model draait](/course/open/build-where-models-run?lang=nl).",
            "weights": "De weights blijven bij de aanbieder; je code stuurt alleen requests. Alleen de base URL en de naam van het model veranderen. Zie [Waar een model draait](/course/open/build-where-models-run?lang=nl).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Waarom

    Versie 2 van de assistent van de bank kwam bij {p2} van de {n} testvragen door de eval set van [Een eval set en regression tests](/course/open/build-eval-set?lang=nl), en volgens de notities van de course verzon hij iets in {m2} antwoorden. Wat verander je nu? Er zijn drie manieren om te veranderen wat een model in je app doet. Je verandert de prompt. Je laat de app feiten opzoeken en in de prompt zetten: RAG, zoals in [Eerst opzoeken: wat RAG is](/course/open/build-rag-idea?lang=nl). Of je traint het model verder op je eigen voorbeelden: fine-tuning.

    Ze lossen verschillende dingen op en kosten verschillend veel. Een prompt verander je in een minuut, en de eval set zegt je binnen het uur of het hielp. RAG vraagt om een zoekfunctie over je teksten. Fine-tuning vraagt om honderden goede voorbeelden, een trainingsrun, en een nieuw model dat je bij moet houden. Kies je verkeerd, dan ben je weken bezig met een model dat nieuwe kosten nog steeds niet kent.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe medewerker geeft klanten steeds verkeerde antwoorden. Mist die informatie, dan geef je een briefje met de feiten, of het handboek om dingen in op te zoeken. Mist die een gewoonte, dan zeg je eerst duidelijk wat je verwacht. Pas als dat niet helpt, train je de medewerker, met voorbeelden van goed werk.
    """.format(p2=_f["passed"]["v2"], n=_f["questions"], m2=_f["made_up"]["v2"]))
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Betekenis |
    |---|---|
    | **fine-tuning** | een model dat al getraind is een stukje verder trainen op je eigen voorbeelden, zodat de weights veranderen |
    | **hosted fine-tuning** | fine-tuning die een aanbieder voor je doet: je uploadt je data, de aanbieder traint, en je roept je eigen versie van het model aan via dezelfde API, onder een nieuwe naam |

    Probeer de drie in een vaste volgorde, en meet elke stap op je eval set:

    1. **Eerst een betere prompt:** instructies, de vorm van het antwoord, een paar voorbeelden, en de feiten als het er weinig zijn. Proberen kost niets, en het is in een minuut terug te draaien.
    2. **RAG** als het model feiten mist: te veel om in de prompt te passen, of feiten die veranderen. De feiten van de bank zijn {sentences} zinnen over {topics} onderwerpen, dus ze passen in de prompt; alle voorwaarden van de bank zouden dat niet doen.
    3. **Fine-tuning** als het model een gedrag mist dat een goede prompt niet betrouwbaar krijgt, en je goede voorbeelden van dat gedrag hebt. Fine-tuning leert een manier van antwoorden, geen feiten: een feit dat zo geleerd is, verander je moeilijk, en het model kan het nog steeds verzinnen.

    De drie gaan samen: een gefinetuned model krijgt nog steeds een prompt, en kan nog steeds dingen opzoeken.

    Een aanbieder finetunet op een van drie manieren, elk vanuit een bestand met één JSON-object per regel:

    - **voorbeelden:** de berichten van een chat, met aan het eind het antwoord dat je wilt; het model leert zulke antwoorden schrijven.
    - **voorkeuren:** een vraag met een beter en een slechter antwoord; het model leert het betere te verkiezen.
    - **een grader:** het bestand bevat vragen, en code of een model geeft elk antwoord dat het model schrijft een score, de reward; het model leert antwoorden schrijven die hoog scoren. Controles zoals die van je eval set kunnen een grader zijn, op andere vragen dan die van de eval set.
    """.format(topics=_f["topics"], sentences=_f["fact_sentences"]))
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: de wijziging kiezen

    1. **Meet** op je eval set, en lees de fouten zoals in [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl).
    2. **Vraag bij elke soort fout:** mist het model feiten, of een gedrag?
    3. **Eerst de prompt:** schrijf op wat je wilt, met de feiten als ze passen, en meet opnieuw.
    4. **Feiten die niet passen, of die veranderen:** RAG.
    5. **Een gedrag dat de prompt niet betrouwbaar krijgt:** verzamel goede voorbeelden uit andere vragen dan die van de eval set, finetune, en meet op dezelfde eval set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `QUESTIONS` bevat de {n} testvragen van de bank. `V1` en `V2` bevatten voor elke vraag het antwoord van versie 1 en van versie 2 van de prompt, elk als dictionary met `"reply"`, `"passed"`, wat de eval set zei, en `"notes"`, de soorten fouten die de course met de hand opschreef, zoals `"made-up"`. `SYSTEM` is de system prompt van versie 2: de feiten van de {topics} onderwerpen en de vorm van het antwoord, {chars} tekens.

    Een trainingsvoorbeeld voor een aanbieder is de chat zoals de app hem stuurt, met aan het eind het antwoord dat je wilt. Voor `QUESTIONS[0]`, *{q0}*, is het het voorbeeld hieronder: de system prompt, de vraag met de rol `user`, en het antwoord van versie 2 met de rol `assistant`, dat door de eval set kwam en geen notitie heeft. Het antwoord is JSON omdat de app die vorm wil: het model leert de vorm die het ziet.
    """.format(n=_f["questions"], topics=_f["topics"], chars=C.number(_f["system_chars"], L.lang), q0=QUESTIONS[0]))
    return


@app.cell(hide_code=True)
def picture_worked_example():
    mo.Html(C.example_picture(0, L.lang))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _text = mo.md(r"""
    ### Wat de antwoorden van de bank zouden opleveren

    De eval set liet {p2} antwoorden van versie 2 door. Als trainingsvoorbeelden zouden die {p2} leren wat ze laten zien, en volgens de notities van de course verzonnen er {pm} iets: daar heeft de eval set geen controle voor. Houd je ze alleen als ze erdoor kwamen en geen notitie hebben, dan blijven er {ex} over.

    Voor een voorkeur is een vraag nodig waarbij precies één van de twee versies erdoor kwam. Dat zijn {pairs} van de {n} vragen: bij {v2b} is het antwoord van versie 2 het betere, bij {v1b} het antwoord van versie 1. Bij {both} vragen kwamen ze er allebei door en bij {neither} geen van beide, dus is er geen paar: de eval set zegt niet welk antwoord beter is. En bij {bm} van de {pairs} paren verzon het betere antwoord ook iets. Een voorkeur leert *meer zoals dit*, dus kijk ook naar het betere antwoord.

    De course finetunet hier niet, en deze voorbeelden mogen niet zoals ze zijn naar een aanbieder: ze beantwoorden de vragen van de eval set. Een model dat daarop getraind is, kan de antwoorden van de eval set uit het hoofd leren zonder beter te worden op nieuwe vragen. Echte voorbeelden komen uit andere vragen, zoals nieuwe berichten van klanten met antwoorden die iemand controleerde.
    """.format(p2=_f["passed"]["v2"], pm=_f["passed_made_up"], ex=_f["examples"], pairs=_f["pairs"], n=_f["questions"], v2b=_f["v2_better"],
               v1b=_f["v1_better"], both=_f["both"], neither=_f["neither"], bm=_f["better_made_up"]))
    mo.vstack([
        _text,
        L.watch_out("Houd de vragen van je eval set buiten alles waarop je traint. Anders meet de eval set wat het model gezien heeft, niet wat het kan."),
    ])
    return


@app.cell(hide_code=True)
def text_examples():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Trainingsvoorbeelden

        `examples(questions, replies)` moet de trainingsvoorbeelden geven die een aanbieder aanneemt: voor elk antwoord dat door de eval set kwam en geen notitie heeft, een dictionary `{"messages": [...]}` met drie berichten: `{"role": "system", "content": SYSTEM}`, de vraag met de rol `"user"`, en het antwoord met de rol `"assistant"`. `replies[i]` is de dictionary van vraag `i`, zoals `V2[i]`, en `r["notes"] == []` is `True` als de course geen notitie schreef. De begincode geeft elke vraag, zonder system prompt en zonder antwoord. Het plaatje laat je functie los op alle 120 antwoorden van versie 2.
        """),
        L.task("Houd een antwoord alleen als het erdoor kwam en geen notitie heeft, en zet `SYSTEM` voor de vraag en het antwoord erna."),
    ])
    return


@app.cell(hide_code=True)
def exercise_examples():
    box_examples = L.code_box("examples", lines=12)
    box_examples
    return (box_examples,)


@app.cell(hide_code=True)
def check_examples(box_examples):
    result_examples = L.check("examples", box_examples.value)
    result_examples
    return (result_examples,)


@app.cell(hide_code=True)
def hints_examples():
    _f = C.findings()
    L.hints(
        "examples",
        [
            "Twee dingen te doen: kiezen welke antwoorden je houdt, en de drie berichten van elk voorbeeld bouwen.",
            "Zet een `if` voor `out.append(...)`; in de lijst met berichten komt het system-bericht eerst en het assistant-bericht als laatste.",
            "Maak er `if r[\"passed\"] and r[\"notes\"] == []:` van, en sluit de lijst af met `{\"role\": \"assistant\", \"content\": ___}`.",
        ],
        why="""```python
def examples(questions, replies):
    out = []
    for i in range(len(questions)):
        r = replies[i]
        if r["passed"] and r["notes"] == []:
            out.append({{"messages": [
                {{"role": "system", "content": SYSTEM}},
                {{"role": "user", "content": questions[i]}},
                {{"role": "assistant", "content": r["reply"]}},
            ]}})
    return out
```

{pm} van de {p2} antwoorden die door de eval set kwamen, verzonnen iets. Als voorbeelden zouden ze het model leren om ook dingen te verzinnen.""".format(pm=_f["passed_made_up"], p2=_f["passed"]["v2"]),
    )
    return


@app.cell(hide_code=True)
def text_pair():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een voorkeur

        `pair(question, a, b)` moet van twee antwoorden op één vraag een voorkeur maken, zoals een aanbieder die aanneemt: `(question, better, worse)`, met de teksten van de antwoorden. `a` en `b` zijn dictionaries zoals `V1[i]` en `V2[i]`. Kwam precies één van de twee door de eval set, dan is dat het betere antwoord. Kwamen ze er allebei of geen van beide door, geef dan `None`. `not x` is `True` als `x` `False` is. De begincode geeft altijd `None`. Het plaatje laat je functie los op alle 120 vragen, met `a = V1[i]` en `b = V2[i]`.
        """),
        L.task("Geef de tekst van het betere antwoord vóór die van het slechtere, en `None` als de eval set ze niet uit elkaar houdt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pair():
    box_pair = L.code_box("pair", lines=7)
    box_pair
    return (box_pair,)


@app.cell(hide_code=True)
def check_pair(box_pair):
    result_pair = L.check("pair", box_pair.value)
    result_pair
    return (result_pair,)


@app.cell(hide_code=True)
def hints_pair():
    L.hints(
        "pair",
        [
            "Vier gevallen: alleen `a` kwam erdoor, alleen `b`, allebei, of geen van beide. Alleen de eerste twee maken een paar.",
            "Vraag `if a[\"passed\"] and not b[\"passed\"]:` voor het eerste geval en een tweede `if` andersom; na allebei `return None`.",
            "In de tweede `if`: `return (question, ___, a[\"reply\"])`.",
        ],
        why="""```python
def pair(question, a, b):
    if a["passed"] and not b["passed"]:
        return (question, a["reply"], b["reply"])
    if b["passed"] and not a["passed"]:
        return (question, b["reply"], a["reply"])
    return None
```

Kwamen ze er allebei door, dan zegt de eval set niet welk antwoord beter is, dus zou een paar daar een voorkeur leren die niemand mat.""",
    )
    return


@app.cell(hide_code=True)
def text_first():
    mo.md(r"""
    ### Oefening 3. Wat eerst?
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-change",
        question="De bank gaat 2,50 euro rekenen voor een kaartbetaling in het buitenland, en de assistent moet dat zeggen. Wat verander je eerst?",
        options={
            "tune": "Finetunen op antwoorden die de kosten noemen",
            "facts": "Het toevoegen aan de feiten in de prompt",
            "rag": "RAG bouwen over alle pagina's van de bank",
        },
        correct="facts",
        why={
            "facts": "De feiten staan al in de prompt, een paar zinnen per onderwerp: één zin erbij, en het volgende antwoord kan de kosten noemen. Meet dezelfde dag op de eval set. RAG komt pas als de feiten niet meer in de prompt passen; fine-tuning is voor een gedrag, niet voor een feit dat kan veranderen.",
            "tune": "Een gefinetuned model leert een feit pas uit veel voorbeelden, kan het nog steeds verzinnen, en houdt de oude kosten als de kosten weer veranderen. Een feit hoort in de prompt of in een zoekfunctie: hier één zin bij de feiten in de prompt.",
            "rag": "De feiten van {topics} onderwerpen, elk een paar zinnen, passen makkelijk in de prompt. RAG loont pas als de feiten niet meer passen, zoals alle voorwaarden; voeg hier één zin toe aan de feiten.".format(topics=C.findings()["topics"]),
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_leak():
    mo.md(r"""
    ### Oefening 4. Welke voorbeelden?
    """)
    return


@app.cell(hide_code=True)
def choice_leak_cell():
    choice_leak = L.choice(
        "eval-leak",
        question="Waarom moet de bank niet finetunen op de antwoorden op de 120 vragen uit de eval set?",
        options={
            "leak": "De eval set zou toetsen wat hij aanleerde",
            "few": "120 antwoorden zijn te weinig",
            "json": "Training kan niets met antwoorden die in JSON staan",
        },
        correct="leak",
        why={
            "leak": "Een model dat getraind is op de vragen en antwoorden van de eval set, kan die uit het hoofd leren zonder beter te worden op nieuwe vragen. Houd de eval set apart, en neem voorbeelden uit andere vragen.",
            "few": "Het aantal is hier het probleem niet: dit zijn de vragen van de eval set zelf. Een model dat daarop getraind is, kan de antwoorden van de eval set uit het hoofd leren zonder beter te worden op nieuwe vragen.",
            "json": "De app wil JSON, dus de voorbeelden moeten ook JSON zijn: het model leert de vorm die het ziet. Het probleem is dat dit de vragen van de eval set zelf zijn: daarop getraind leert het model de antwoorden van de eval set uit het hoofd.",
        },
    )
    choice_leak
    return (choice_leak,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elke stap meet je met de eval set van [Een eval set en regression tests](/course/open/build-eval-set?lang=nl), op de fouten die [Eerst de fouten bekijken](/course/open/build-error-analysis?lang=nl) vond. RAG begint bij [Eerst opzoeken: wat RAG is](/course/open/build-rag-idea?lang=nl), en [Klopt het RAG-systeem?](/course/open/build-rag-eval?lang=nl) meet het. Een gefinetuned model roep je aan zoals elk ander model, zoals in [Waar een model draait](/course/open/build-where-models-run?lang=nl). Wat fine-tuning in het model doet, zie je in [Project: leer je model instructies volgen (SFT)](/course/open/llm-sft?lang=nl) en [DPO: voorkeuren zonder reward model](/course/open/llm-dpo-loss?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, choice_leak, result_examples, result_pair):
    L.wrap_up(result_examples, result_pair, choice_first, choice_leak)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
