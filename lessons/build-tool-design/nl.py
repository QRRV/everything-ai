# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-tool-design. Route: course/route/lessen/bouwen.yaml. Checks, the course's runs, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-tool-design", "nl")
    C = L.checks   # drie beschrijvingen van convert, twee soorten fouten, de runs van Qwen3, de checks en de demo
    F = C.findings()


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl), één over [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_arguments():
    quick_arguments = L.choice(
        "quick-arguments",
        question="Een tool call in het antwoord heeft `\"arguments\": \"{\\\"amount\\\": 37, \\\"from_unit\\\": \\\"km\\\", \\\"to_unit\\\": \\\"mi\\\"}\"`. Wat geeft je programma aan de functie `convert`?",
        options={
            "text": "Die tekst zelf, zoals hij kwam: convert kan hem zo lezen",
            "loads": "Wat `json.loads` van die tekst maakt: een dictionary",
            "id": "Het `id` van de call, zodat convert weet op welke call hij antwoordt",
        },
        correct="loads",
        why={
            "loads": "De argumenten komen als JSON-tekst; `json.loads` maakt de dictionary die je functie leest.",
            "text": "Het is een tekst, nog geen dictionary: `arguments[\"amount\"]` zou erop vastlopen. Zie [Tools als JSON](/course/open/build-tool-calling?lang=nl).",
            "id": "Het `id` gaat terug met het resultaat, als `tool_call_id`; de functie heeft de argumenten nodig. Zie [Tools als JSON](/course/open/build-tool-calling?lang=nl).",
        },
    )
    quick_arguments
    return (quick_arguments,)


@app.cell(hide_code=True)
def quick_check_hosted():
    quick_hosted = L.choice(
        "quick-hosted",
        question="Wie voert `detect_objects` uit, de hosted tool van de server van de course?",
        options={
            "server": "De server, bij elk beeld, voordat het model de vraag leest",
            "program": "Je programma, zodra het model hem in zijn antwoord aanroept, met het beeld als argument",
            "model": "Qwen3 zelf: het leest het beeld en voert de detector in het model uit",
        },
        correct="server",
        why={
            "server": "Een hosted tool draait bij de aanbieder: je noemt hem in het request, en de server voert hem uit.",
            "program": "Zo werkt een tool die je eigen programma uitvoert. Een hosted tool noem je alleen. Zie [Meer dan tekst](/course/open/build-multimodal-apis?lang=nl).",
            "model": "Qwen3 leest alleen tekst; de detector van de server, DETR, is een model op zich. Zie [Meer dan tekst](/course/open/build-multimodal-apis?lang=nl).",
        },
    )
    quick_hosted
    return (quick_hosted,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Tools als JSON](/course/open/build-tool-calling?lang=nl) riep Qwen3 `convert` alle {n} keer dat het moest aan met de goede argumenten. Dat was met de volledige beschrijving. Een model kent een tool alleen van wat het in de prompt leest: de naam, de beschrijving, de argumenten. Het kan je code niet lezen, de tool niet uitproberen, en geen collega vragen wat `b` betekent. De beschrijving is dus geen papierwerk: ze bepaalt of het model de tool überhaupt kan gebruiken. De course mat wat er gebeurt met minder, en met foutmeldingen die minder zeggen.
    """.format(n=F["questions"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Alles wat een agent van een tool ziet, is zijn interface: de naam, de beschrijving en de argumenten die hij vóór een call leest, en het resultaat of de fout die hij erna leest. Een scherm voor mensen wordt ontworpen, getest en verbeterd; de interface voor een model verdient dezelfde zorg.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **agent-computer interface** | alles wat een agent van zijn tools ziet: namen, beschrijvingen, argumenten, en wat er terugkomt, resultaten en fouten; een gebruikersinterface, maar dan voor een model |
    | **client-side tool** | een tool die je eigen programma uitvoert, nadat het model hem aanroept, zoals `convert`; het tegenovergestelde van een hosted tool, die de aanbieder uitvoert |

    ### Drie beschrijvingen van convert

    De course beschreef dezelfde functie op drie manieren. Kaal: een korte naam, drie letters als argumenten, geen beschrijvingen.

    ```python
    {"name": "conv", "description": "",
     "parameters": {"type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "string"}, "c": {"type": "string"}},
                    "required": ["a", "b", "c"]}}
    ```

    Alleen namen: goede namen, maar geen beschrijvingen van de argumenten, en geen `enum` met de toegestane eenheden.

    ```python
    {"name": "convert", "description": "Convert an amount.",
     "parameters": {"type": "object",
                    "properties": {"amount": {"type": "number"}, "from_unit": {"type": "string"},
                                   "to_unit": {"type": "string"}},
                    "required": ["amount", "from_unit", "to_unit"]}}
    ```

    En volledig, zoals in [Tools als JSON](/course/open/build-tool-calling?lang=nl). Met elk ervan stelde de course dezelfde {n} vragen, en noemde het de tool in `tool_choice`, zodat elk eerste antwoord een call is.
    """.replace("{n}", str(F["questions"])))
    return


@app.cell(hide_code=True)
def picture_described():
    mo.Html(C.described_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_described():
    mo.vstack([
        mo.md(r"""
        Kaal waren {bare} van de {n} eerste calls goed; met alleen namen {names}; volledig alle {full}. Met alleen namen schreef Qwen3 de eenheden als woorden: {words}. `convert` kent daar geen van. De kale tool deed het slechter op een manier die moeilijker te zien is: bij *{q}* riep het `conv` aan met `{call}`, andersom. Dat kan `convert`, dus gaf hij `{result}` terug, zonder fout: een verkeerd getal dat er net zo goed uitziet als een goed. Met `from_unit` en `to_unit` als namen zette het model de eenheden elke keer in de goede volgorde.

        De volledige beschrijving kost tokens, in elk request: {p_full} prompt tokens, tegen {p_names} met alleen namen en {p_bare} kaal. Zo'n tachtig tokens voor acht goede calls in plaats van drie: dat is het waard.
        """.format(bare=len(F["right_bare"]), names=len(F["right_names"]), full=len(F["right_full"]), n=F["questions"],
                   words=", ".join("`{}`".format(x) for x in F["words"]), q=C.QUESTIONS[F["wrong_way"][0]][0],
                   call=C.short(C.as_convert(C.steps_of("bare", F["wrong_way"][0])[0]["tool_calls"][0])),
                   result=C.steps_of("bare", F["wrong_way"][0])[0]["results"][0],
                   p_full=F["prompt_full"], p_names=F["prompt_names"], p_bare=F["prompt_bare"])),
        L.watch_out("De ergste call is niet de call die mislukt, maar een verkeerde die de tool kan uitvoeren. Maak zulke calls moeilijk: namen die de volgorde zeggen, een vaste lijst waarden, en elk argument dat nodig is verplicht."),
    ])
    return


@app.cell(hide_code=True)
def text_errors():
    mo.md(r"""
    ### Fouten waar het model iets mee kan

    Met alleen namen kregen {k} eerste calls een fout: een onbekende eenheid. De course liet die {k} gesprekken verder lopen, één keer met elk van twee soorten fout uit `convert`: alleen `Error`, of een fout die zegt wat er misging en wat het anders moet, zoals `Error: unknown unit 'miles'. Use km, mi, kg or lb.` Elke run kreeg hooguit drie requests.
    """.format(k=len(F["failed"])))
    return


@app.cell(hide_code=True)
def picture_errors():
    mo.Html(C.errors_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_errors_after():
    mo.vstack([
        mo.md(r"""
        Met alleen `Error` gaf Qwen3 het alle {k} keer op, met een excuus zoals *{sorry}* Met de fout die zegt wat het moet doen, riep het in {again} van de {k} opnieuw aan; bij {fixed} daarvan vond het de goede call, en één keer antwoordde het goed binnen drie requests. Twee keer gaf het de fout door als antwoord. Een klein model gebruikt een goede fout dus maar af en toe; de beschrijving hielp veel meer. Maak ze allebei goed: de beschrijving voorkomt de meeste verkeerde calls, en de fout geeft het model een uitweg uit de rest.
        """.format(k=len(F["failed"]), sorry=C.steps_of("names-terse", F["failed"][0])[-1]["text"].split(". ")[0] + ".",
                   again=len(F["helpful_again"]), fixed=len(F["helpful_fixed"]))),
        L.watch_out("Een fout gaat als tekst naar het model, en kost tokens zoals elk bericht. Een hele traceback van Python zegt een model weinig: geef een of twee zinnen terug, met wat het nu kan doen."),
    ])
    return


@app.cell(hide_code=True)
def text_who_runs():
    mo.md(r"""
    ### Wie de tool uitvoert

    `convert` is een client-side tool: het model schrijft de call, je programma voert hem uit, en het resultaat gaat terug in het volgende request. De `detect_objects` van [Meer dan tekst](/course/open/build-multimodal-apis?lang=nl) was hosted: je noemde hem alleen, de server voerde hem uit, en je programma zag nooit een call. Hosted tools, zoals zoeken op het web of code uitvoeren bij een aanbieder, schelen je werk, maar ze draaien waar de aanbieder beslist, met wat die toestaat. Een client-side tool draait in je programma, op je data, onder je regels: over die regels gaat de volgende les.

    De woorden *agent-computer interface* komen van SWE-agent, een coding agent uit 2024, waarvan de makers merkten dat een model het beter deed met commando's die voor een model gemaakt waren dan met een gewone terminal. Anthropic raadt in *Building effective agents* (december 2024) aan om net zoveel zorg aan die interface te besteden als aan een scherm voor mensen.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een tool die een agent goed kan gebruiken

    1. **Een naam die zegt wat hij doet**: `convert`, niet `conv`.
    2. **Een beschrijving** van wat de tool doet, wanneer je hem gebruikt, en waar hij ophoudt: *kilometres and miles, or kilograms and pounds*.
    3. **Argumenten met duidelijke namen**, elk met een beschrijving en een voorbeeld, en een `enum` waar maar een paar waarden mogen.
    4. **Maak verkeerde calls moeilijk**: namen die de volgorde zeggen, verplichte argumenten, vaste waarden.
    5. **Fouten die zeggen wat er misging en wat het anders moet**, beginnend met `Error`.
    6. **Probeer het op het model**: stuur echte vragen, en lees de calls die het schrijft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _steps = C.steps_of("names-helpful", C.WORKED)
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    *{q}*, met alleen namen en de fout die zegt wat het moet doen.

    1. **De eerste call**: `{c1}`. De eenheid `{u}` kent `convert` niet.
    2. **De fout** gaat terug als resultaat van de call: `{r1}`. Hij noemt de eenheid, en de vier die het model kan gebruiken.
    3. **De tweede call**: `{c2}`, en `convert` geeft `{r2}` terug.
    4. **Het antwoord**: *{a}*

    Met alleen `Error` eindigde dezelfde eerste call in een excuus, en helemaal geen getal.
    """.format(q=C.QUESTIONS[C.WORKED][0], c1=C.short(C.as_convert(_steps[0]["tool_calls"][0])),
               u=C.as_convert(_steps[0]["tool_calls"][0])["from_unit"], r1=_steps[0]["results"][0],
               c2=C.short(C.as_convert(_steps[1]["tool_calls"][0])), r2=_steps[1]["results"][0], a=_steps[2]["text"]))
    return


@app.cell(hide_code=True)
def text_helpful():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Fouten waar het model iets mee kan

        Het vak bevat `convert` als functie die je programma uitvoert: hij krijgt de argumenten als dictionary, zoals `json.loads` ze van een call maakt. Elke fout zegt nog alleen `"Error"`. Laat elke fout zeggen wat er misging en wat het anders moet: bij een onbekende eenheid de eenheid die het model schreef en de vier eenheden die `convert` kent; bij twee eenheden die niet bij elkaar passen, welke wel. Begin elke fout met `Error`. De check roept je `convert` aan met de argumenten van {k} calls, zoals een model ze zou kunnen schrijven; het plaatje onder de check laat zien wat elke call terugkrijgt.
        """.format(k=len(C.VISIBLE) + len(C.HIDDEN))),
        L.task("Vervang elke `\"Error\"` door een tekst die begint met `Error`, zegt wat er misging, en de eenheden `km`, `mi`, `kg` en `lb` noemt; zet bij een onbekende eenheid ook die eenheid in de tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_helpful():
    box_helpful = L.code_box("helpful", lines=15)
    box_helpful
    return (box_helpful,)


@app.cell(hide_code=True)
def check_helpful(box_helpful):
    result_helpful = L.check("helpful", box_helpful.value)
    result_helpful
    return (result_helpful,)


@app.cell(hide_code=True)
def hints_helpful():
    L.hints(
        "helpful",
        [
            "Een fout wordt gelezen door een model dat het opnieuw wil proberen. Wat moet het weten? Welk woord van zijn call niet klopte, en wat het in plaats daarvan mag schrijven.",
            "Plak de eenheid die het model schreef in de tekst met `+`: `\"Error: unknown unit '\" + a + ...`. De vier eenheden kunnen gewoon in de tekst staan.",
            "`return \"Error: unknown unit '\" + a + \"'. Use km, mi, kg or lb.\"`, hetzelfde met `b`, en bij de laatste: `\"Error: cannot convert \" + a + \" to \" + b + ...`, met welke eenheden bij elkaar horen.",
        ],
        why="""```python
def convert(arguments):
    amount = arguments["amount"]
    a = arguments["from_unit"]
    b = arguments["to_unit"]
    if a not in FACTOR:
        return "Error: unknown unit '" + a + "'. Use km, mi, kg or lb."
    if b not in FACTOR:
        return "Error: unknown unit '" + b + "'. Use km, mi, kg or lb."
    if KIND[a] != KIND[b]:
        pairs = "Convert km and mi into each other, or kg and lb."
        return "Error: cannot convert " + a + " to " + b + ". " + pairs
    return str(round(amount * FACTOR[a] / FACTOR[b], 2)) + " " + b
```

Dit zijn de fouten van de runs van de course. Je woorden mogen anders zijn: de check kijkt naar wat erin staat, niet naar hoe je het zegt. Schrijf ze in het Engels, zoals de rest van de tool: Qwen3 leest Engels het best.""",
    )
    return


@app.cell(hide_code=True)
def text_describe_better():
    mo.md(r"""
    ### Oefening 2. Een betere beschrijving
    """)
    return


@app.cell(hide_code=True)
def choice_describe_better_cell():
    choice_describe_better = L.choice(
        "describe-better",
        question="Met alleen namen schreef Qwen3 eenheden zoals `miles` en `pounds`. Wat maakte in de runs van de course alle {n} eerste calls goed?".format(n=F["questions"]),
        options={
            "name": "Een langere naam voor de tool, `convert_between_kilometers_miles_kilograms_and_pounds`, die alles zegt",
            "enum": "De vier eenheden als `enum`, en een beschrijving van elk argument",
            "prompt": "Een system prompt die het model zegt elke eenheid kort te schrijven, zoals mi en lb",
        },
        correct="enum",
        why={
            "enum": "Met `\"enum\": [\"km\", \"mi\", \"kg\", \"lb\"]` leest het model welke vier waarden er zijn, en schrijft het die. Dat is de volledige beschrijving van de tabel hierboven.",
            "name": "Een naam zegt wat een tool doet, maar niet welke waarden een argument mag hebben. Waar in een beschrijving hoort dat?",
            "prompt": "Dat heeft de course niet gemeten; en de regel hoort bij de tool, zodat hij meegaat waar de tool heen gaat. Waar in de beschrijving kun je de toegestane eenheden noemen?",
        },
    )
    choice_describe_better
    return (choice_describe_better,)


@app.cell(hide_code=True)
def text_client_side():
    mo.md(r"""
    ### Oefening 3. Wie voert hem uit
    """)
    return


@app.cell(hide_code=True)
def choice_client_side_cell():
    choice_client_side = L.choice(
        "client-side",
        question="Een app geeft een model drie tools: het zoeken op het web van de aanbieder, een hosted tool, en `convert` en `send_email`, die je programma uitvoert. Voor welke tools krijgt je programma tool calls om uit te voeren?",
        options={
            "all": "Alle drie: elke tool call van het model komt terug bij je programma om uit te voeren",
            "search": "Alleen het zoeken op het web, omdat dat buiten de app reikt",
            "two": "`convert` en `send_email`",
        },
        correct="two",
        why={
            "two": "Die twee zijn client-side tools: je programma voert ze uit. Het zoeken op het web voert de aanbieder zelf uit, en het antwoord komt terug met wat het vond.",
            "all": "Een hosted tool draait bij de aanbieder: je noemt hem, en krijgt nooit een call om uit te voeren. Welke van de drie is hosted?",
            "search": "Het is andersom: het zoeken, een hosted tool, voert de aanbieder zelf uit. Welke tools draaien in je programma?",
        },
    )
    choice_client_side
    return (choice_client_side,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het op Qwen3

    Zodra oefening 1 goed is, kun je je `convert` achter de beschrijving met alleen namen zetten, en zien wat Qwen3 met je fouten doet: de course start zijn eigen server op je computer, met Qwen3-0.6B erachter. De demo stelt de vraag in het vak, maakt van het eerste antwoord een call, voert elke call uit met je `convert`, en vraagt opnieuw met het resultaat, hooguit drie requests. Probeer een vraag met miles of pounds in woorden. Schrijf hem in het Engels, omdat Qwen3 Engels het best schrijft.
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
def question_box_cell():
    question_box = mo.ui.text(value=C.QUESTIONS[1][0], label="Je vraag", full_width=True)
    send_button = mo.ui.run_button(label="Vraag, met je convert")
    mo.vstack([question_box, send_button])
    return question_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_helpful, question_box, send_button):
    _convert = C.learner_convert(box_helpful.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _convert is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo voert de calls uit met je eigen `convert`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om te vragen.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*De vraag is verstuurd; de server laadt Qwen3 en antwoordt …*"))
        _words = {"title": "Wat Qwen3 aanriep, en wat je convert teruggaf", "request": "Request {n}:",
                  "open": "Geen antwoord na {n} requests.", "failed": "Het stopte met een fout: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_convert, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De beschrijving die je in [Tools als JSON: function calling](/course/open/build-tool-calling?lang=nl) afmaakte, is de reden dat Qwen3 `convert` elke keer goed aanriep: met minder riep het twee of drie van de acht keer goed aan. De hosted tool van [Meer dan tekst](/course/open/build-multimodal-apis?lang=nl) heeft nu een naam voor zijn tegenovergestelde: een client-side tool, die je programma uitvoert. In de volgende les, [Wat een tool mag](/course/open/build-tool-permissions?lang=nl), beslis je welke van die calls je programma doorlaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_client_side, choice_describe_better, result_helpful):
    L.wrap_up(result_helpful, choice_describe_better, choice_client_side)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
