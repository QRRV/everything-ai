# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-agent-memory. Route: course/route/lessen/bouwen.yaml. Checks, de runs van de course, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-agent-memory", "nl")
    C = L.checks   # de tool remember, de runs van Qwen3 met en zonder geheugen, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [De agent loop](/course/open/build-agent-loop?lang=nl), en een over [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_request():
    quick_request = L.choice(
        "quick-request",
        question="Wat weet het model in `run_agent` van het gesprek als het het vierde request krijgt?",
        options={
            "all": "Wat in dat request staat: de system prompt, de taak, en elk antwoord en resultaat tot nu toe",
            "last": "Alleen het laatste resultaat van een tool, want de server bewaart de rest van het gesprek ervoor",
            "server": "Wat de server bewaarde van de drie requests ervoor, samen met de nieuwe berichten",
        },
        correct="all",
        why={
            "all": "Daarom stuurt `run_agent` elke keer het hele gesprek: het model weet alleen wat in het request staat.",
            "last": "De server bewaart niets tussen requests; `run_agent` stuurt elk bericht opnieuw. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
            "server": "De server bewaart niets tussen requests: wat het model weet, is wat in het request staat. Zie [De agent loop](/course/open/build-agent-loop?lang=nl).",
        },
    )
    quick_request
    return (quick_request,)


@app.cell(hide_code=True)
def quick_check_note():
    quick_note = L.choice(
        "quick-note",
        question="Een feit mag nooit verloren gaan in een lang gesprek, zoals een allergie. Waar hoorde het?",
        options={
            "summary": "In de samenvatting van de oude berichten, die het model schrijft zodra het venster vol is",
            "window": "In het venster van de laatste vier berichten, waarin altijd de nieuwste feiten staan",
            "note": "In een notitie die je code bewaart en elke keer in de system prompt meestuurt",
        },
        correct="note",
        why={
            "note": "Een notitie die je code bewaart, gaat mee in elk request, hoe lang het gesprek ook wordt.",
            "summary": "Een samenvatting houdt alleen wat het model dat hem schreef belangrijk vond. Zie [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl).",
            "window": "Een feit uit het eerste bericht valt uit een venster van de laatste vier. Zie [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl).",
        },
    )
    quick_note
    return (quick_note,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model onthoudt niets tussen requests. In [De agent loop](/course/open/build-agent-loop?lang=nl) stuurde elk request het hele gesprek opnieuw: zo wist de agent wat zijn tools hadden gevonden. Als het gesprek voorbij is, is dat allemaal weg, en het volgende gesprek begint leeg. In [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl) bewaarde je code een notitie en stuurde die elke keer mee. In deze les beslist de agent zelf wat hij bewaart, met een tool, en komen de notities in een bestand dat langer meegaat dan het gesprek.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **kortetermijngeheugen** | wat een agent weet binnen één gesprek: de berichten in zijn request |
        | **langetermijngeheugen** | wat een agent weet over gesprekken heen: notities die buiten het model bewaard worden, bijvoorbeeld in een bestand, en die je code in de context van een nieuw gesprek zet |
        | **`open(path, "a")`** | opent een bestand om aan toe te voegen: wat je schrijft komt aan het eind, en een bestand dat er nog niet is, wordt gemaakt |
        | **`f.write(text)`** | schrijft de tekst in het open bestand zoals hij is, zonder nieuwe regel aan het eind |
        | **`FileNotFoundError`** | de fout die `open` geeft als het bestand om te lezen er niet is |

        Langetermijngeheugen heeft hier twee helften. In een gesprek heeft de agent een tool `remember`, en de system prompt zegt wat de moeite van het bewaren waard is:

        > {told}

        Je code voert elke call van `remember` uit door de notitie aan een bestand toe te voegen, en stuurt `"Saved."` terug. Aan het begin van een nieuw gesprek leest je code het bestand en zet de notities in de system prompt, onder de regel *{head}*. Het model weet in het nieuwe gesprek dan wat de gebruiker het in een eerder gesprek vertelde.
        """.format(told=C.TOLD, head=C.NOTES_HEAD)),
        mo.accordion({"De tool remember zoals het request hem meegeeft, als je hem wilt lezen": mo.md(
            "```json\n" + C.REMEMBER_JSON + "\n```")}),
    ])
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.idea_picture("nl"))
    return


@app.cell(hide_code=True)
def text_measured():
    mo.md(r"""
    ### Wat de course mat

    De course vertelde Qwen3-0.6B {n} dingen over een gebruiker, elk in een eigen gesprek met de tool `remember`, met temperature 0, zodat hetzelfde request altijd hetzelfde antwoord krijgt. Elk ging twee keer: met `tool_choice` op `"auto"`, zodat het model kiest of het de tool aanroept, en met `tool_choice` die `remember` noemt, zodat het moet. Daarna stelde de course {n} vragen die ze nodig hebben: in een nieuw gesprek zonder geheugen, in een gesprek waarin de gebruiker het net had gezegd, en in een nieuw gesprek met in de system prompt elke notitie die de {n} gesprekken bewaarden.
    """.format(n=F["facts"]))
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture("nl"))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.vstack([
        mo.md(C.measured_text("nl")),
        L.watch_out("Een notitie is wat het model schreef, niet wat de gebruiker zei: lees de notities die je agent bewaart. Een notitie die niet meer klopt, stuurt elk nieuw gesprek de verkeerde kant op, en notities over een persoon gaan over die persoon: vertel gebruikers wat je agent bewaart, en laat ze het inzien en weghalen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: langetermijngeheugen voor een agent

    1. **Geef de agent een tool om een notitie te bewaren,** en zeg in de system prompt wat de moeite van het bewaren waard is.
    2. **Voeg elke notitie toe aan het eind van een bestand,** op een eigen regel: `open(path, "a")`.
    3. **Begin een nieuw gesprek met de notities:** lees het bestand, en zet de notities in de system prompt onder een regel die zegt wat ze zijn. Nog geen bestand? Dan zijn er geen notities.
    4. **Houd de notities kloppend:** een notitie die niet meer klopt, wordt vervangen, niet aangevuld.
    5. **Stuur bij veel notities alleen de notities mee die passen,** zodat de system prompt niet met elke notitie groeit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(C.worked_text("nl"))
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(C.WORKED, "nl"))
    return


@app.cell(hide_code=True)
def text_remember():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De tool remember

        `remember(arguments)` is wat je code uitvoert voor een call van de tool. De call van het model geeft zijn argumenten als dictionary, zoals `{"note": "The user has no oven."}`. Voeg de notitie toe aan het bestand met notities, op een eigen regel, en houd de notities die er al stonden. `NOTES` is het pad van het bestand; het vak kent het. De check roept je functie twee keer aan op een nieuw bestand, één keer op een bestand waarin al twee notities staan, en nog één keer op een nieuw bestand; het plaatje laat het bestand zien vóór en na je twee calls.
        """),
        L.task("Vervang `...` door een `with`-blok dat `NOTES` opent om aan toe te voegen, en de notitie schrijft met een nieuwe regel erachter."),
    ])
    return


@app.cell(hide_code=True)
def exercise_remember():
    box_remember = L.code_box("remember", lines=7)
    box_remember
    return (box_remember,)


@app.cell(hide_code=True)
def check_remember(box_remember):
    result_remember = L.check("remember", box_remember.value)
    result_remember
    return (result_remember,)


@app.cell(hide_code=True)
def hints_remember():
    L.hints(
        "remember",
        [
            "Een bestand lezen was `open(path)` in een `with`-blok. Aan een bestand toevoegen gaat net zo, met een tweede waarde die zegt hoe je het opent.",
            "De tweede waarde van `open` zegt hoe: `\"r\"` om te lezen, `\"w\"` om een nieuw bestand over het oude heen te schrijven, `\"a\"` om aan het eind toe te voegen. `f.write` schrijft precies wat je het geeft, dus een notitie heeft `\"\\n\"` erachter nodig om haar regel af te sluiten.",
            "`with open(NOTES, ___, encoding=\"utf-8\") as f:`, en de `f.write` ingesprongen eronder.",
        ],
        why="""```python
def remember(arguments):
    note = arguments["note"]
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(note + "\\n")
    return "Saved."
```

`"a"` houdt wat in het bestand staat en maakt een bestand dat er nog niet is; `"w"` zou het elke keer leegmaken. `"Saved."` gaat terug naar het model als het resultaat van de tool.""",
    )
    return


@app.cell(hide_code=True)
def text_first_messages():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een nieuw gesprek met de notities

        `first_messages(question)` maakt de eerste berichten van een nieuw gesprek. Lees het bestand met notities; staan er notities in, dan wordt de system prompt `SYSTEM`, een lege regel, `NOTES_HEAD`, een nieuwe regel en de notities. Is er nog geen bestand, of een leeg, dan blijft de system prompt `SYSTEM`. `f.read()` geeft het hele bestand als één string, en `.strip()` haalt de nieuwe regel aan het eind weg. Het vak kent `NOTES`, `SYSTEM` en `NOTES_HEAD`. De check probeert een bestand met twee notities, geen bestand, een leeg bestand en een bestand met één notitie; het plaatje laat de berichten zien die je functie voor het eerste maakt.
        """),
        L.task("Vervang `...` door code die de notities leest, of geen notities als het bestand er niet is, en `NOTES_HEAD` en de notities aan `system` toevoegt als die er zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first_messages():
    box_first_messages = L.code_box("first-messages", lines=12)
    box_first_messages
    return (box_first_messages,)


@app.cell(hide_code=True)
def check_first_messages(box_first_messages):
    result_first_messages = L.check("first-messages", box_first_messages.value)
    result_first_messages
    return (result_first_messages,)


@app.cell(hide_code=True)
def hints_first_messages():
    L.hints(
        "first-messages",
        [
            "Twee delen: lees de notities, waarbij een ontbrekend bestand geen notities betekent; verander daarna `system` alleen als er notities zijn.",
            "Zet het blok `with open(NOTES, encoding=\"utf-8\") as f:` onder `try:`, en geef `notes` de lege tekst `\"\"` onder `except FileNotFoundError:`. Een lege tekst telt als onwaar, dus `if notes:` klopt alleen als er notities zijn.",
            "Onder `if notes:`: `system = SYSTEM + ___ + NOTES_HEAD + \"\\n\" + notes`.",
        ],
        why="""```python
def first_messages(question):
    system = SYSTEM
    try:
        with open(NOTES, encoding="utf-8") as f:
            notes = f.read().strip()
    except FileNotFoundError:
        notes = ""
    if notes:
        system = SYSTEM + "\\n\\n" + NOTES_HEAD + "\\n" + notes
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]
```

Een nieuwe gebruiker heeft nog geen bestand met notities: `except FileNotFoundError` maakt dat een gewoon geval in plaats van een fout.""",
    )
    return


@app.cell(hide_code=True)
def text_stale_note():
    mo.md(r"""
    ### Oefening 3. Een notitie die niet meer klopt
    """)
    return


@app.cell(hide_code=True)
def choice_stale_note_cell():
    choice_stale_note = L.choice(
        "stale-note",
        question="In je bestand met notities staat *The user has no oven.* Vandaag zegt de gebruiker: *I bought an oven.* Wat moet er met de notitie gebeuren?",
        options={
            "keep": "Laat hem staan, en zet *The user has an oven.* eronder: het model leest toch beide notities",
            "all": "Maak het hele bestand met notities leeg, zodat niets ouds de agent nog op het verkeerde been zet",
            "replace": "Vervang hem door het nieuwe feit, zodat elke notitie in het bestand nog klopt",
        },
        correct="replace",
        why={
            "replace": "Elk nieuw gesprek gelooft de notities. Daarvoor is een manier nodig om een notitie te veranderen, zoals een tool die er een vervangt, of code die het bestand herschrijft.",
            "keep": "Dan zegt de system prompt zowel *no oven* als *an oven*, en niets vertelt het model welke van de twee nu klopt.",
            "all": "Dan vergeet de agent ook elke andere notitie, en die kloppen nog wel.",
        },
    )
    choice_stale_note
    return (choice_stale_note,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zijn oefening 1 en 2 goed, dan kun je Qwen3 een geheugen geven: de course start een eigen server op je computer, met Qwen3-0.6B erachter. Vertel het model iets over jezelf: het request noemt `remember` in `tool_choice`, zoals in de tweede run van de course, dus het model roept hem aan, en je functie bewaart de notitie. Stel daarna een vraag in een nieuw gesprek, dat begint met je `first_messages`. De notities blijven in een bestand in je werkmap tot je ze vergeet.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet te bereiken, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def tell_box_cell():
    tell_box = mo.ui.text(value="I am allergic to nuts.", label="Vertel de agent iets over jezelf", full_width=True)
    tell_button = mo.ui.run_button(label="Verstuur")
    mo.vstack([tell_box, tell_button])
    return tell_box, tell_button


@app.cell(hide_code=True)
def demo_tell(box_remember, tell_box, tell_button):
    _remember = C.learner_fn("remember", box_remember.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _remember is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo bewaart notities met je eigen `remember`.*")
    elif not tell_button.value:
        _out = mo.md("*Druk op de knop om te versturen.*")
    elif not tell_box.value.strip():
        _out = mo.md("*Schrijf eerst iets in het vak.*")
    else:
        mo.output.replace(mo.md("*Het bericht is verstuurd; de server laadt Qwen3 …*"))
        _words = {"title": "Je bericht, en wat je remember bewaarde", "no_call": "Het model riep geen tool aan: er is niets bewaard.",
                  "file": "Je bestand met notities nu:", "no_file": "(nog geen bestand)",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.tell(_remember, tell_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def ask_box_cell():
    ask_box = mo.ui.text(value=C.FACTS[0][1], label="Vraag in een nieuw gesprek", full_width=True)
    ask_button = mo.ui.run_button(label="Vraag")
    forget_button = mo.ui.run_button(label="Vergeet alle notities")
    mo.vstack([ask_box, mo.hstack([ask_button, forget_button], justify="start")])
    return ask_box, ask_button, forget_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button, box_first_messages, forget_button):
    _first = C.learner_fn("first-messages", box_first_messages.value)
    if forget_button.value:
        C.forget()
        _out = mo.md("*Het bestand met notities is weg: de agent begint weer zonder notities.*")
    elif not C.downloaded():
        _out = mo.md("*Download eerst Qwen3, met de knop hierboven.*")
    elif _first is None:
        _out = mo.md("*Maak eerst oefening 2 af: het nieuwe gesprek begint met je eigen `first_messages`.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 …*"))
        _words = {"title": "Een nieuw gesprek, begonnen door je first_messages",
                  "error": "Je first_messages stopte met {e}.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.ask_new(_first, ask_box.value, _words))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De agent loop](/course/open/build-agent-loop?lang=nl) was het geheugen van de agent het gesprek dat hij met elk request opnieuw stuurde: kortetermijngeheugen. [Wat er in het venster gaat](/course/open/build-context-engineering?lang=nl) zette een notitie van je code in de system prompt; hier bewaart de agent zijn eigen notities met een tool, in een bestand, en een nieuw gesprek begint ermee: langetermijngeheugen. De volgende les laat code draaien die een agent schreef, in een sandbox.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stale_note, result_first_messages, result_remember):
    L.wrap_up(result_remember, result_first_messages, choice_stale_note)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
