# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les start-prompt-basics. Route: course/route/lessen/bouwen.yaml. Checks, de vragen en de run van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("start-prompt-basics", "nl")
    C = L.checks   # de drie vragen, de zeven manieren om ze te stellen, en de run van Qwen3 van de course


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_window():
    quick_window = L.choice(
        "quick-window",
        question="Je chat al lang met een assistent, en nu vergeet hij iets wat je hem aan het begin vertelde. Wat is de waarschijnlijke reden?",
        options={
            "last": "Hij leest altijd alleen je laatste bericht",
            "window": "Het begin van de chat past niet meer in zijn context window",
            "stored": "Hij bewaart de chat, maar leest hem nooit terug",
        },
        correct="window",
        why={
            "last": "Hij leest veel meer dan je laatste bericht: alles wat in zijn context window past. Zie [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl).",
            "window": "Het model ziet alleen wat in zijn context window past; in een lange chat vallen de oudste berichten eruit.",
            "stored": "Wat in zijn context window past, leest hij elke keer dat hij antwoordt. Zie [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl).",
        },
    )
    quick_window
    return (quick_window,)


@app.cell(hide_code=True)
def quick_check_paste():
    quick_paste = L.choice(
        "quick-paste",
        question="Je wilt een chatassistent op het web een klacht van een klant laten samenvatten. Wat doe je eerst?",
        options={
            "nothing": "Niets: ik plak hem zoals hij is",
            "summary": "De samenvatting zelf schrijven, zodat de assistent hem nooit ziet",
            "leave-out": "De naam en contactgegevens van de klant weglaten of vervangen",
        },
        correct="leave-out",
        why={
            "nothing": "Wat je plakt, gaat naar de computers van de aanbieder. Staat er iets in over iemand anders? Zie [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl).",
            "summary": "Je kunt je best laten helpen: alleen niet met gegevens die hij niet nodig heeft. Zie [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl).",
            "leave-out": "Voor de samenvatting maakt het niet uit wie de klacht schreef; zonder die gegevens kan hij de chat in.",
        },
    )
    quick_paste
    return (quick_paste,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Hetzelfde model, dezelfde vraag, en toch een goed of een fout antwoord: het hangt ervan af hoe je vraagt. Een model schrijft zijn antwoord woord voor woord, en elk woord dat het schrijft, hangt af van alles wat ervoor staat, ook van jouw woorden. Wat je erin stopt, en in welke vorm, verandert dus wat eruit komt. Deze les laat zien hoeveel, met één klein chat model en drie schoolsommen, elk op zeven manieren gesteld, acht keer per manier.
    """)
    return


@app.cell(hide_code=True)
def predict_number():
    guess_number = L.guess(
        "only-number",
        question=f"De course stelde Qwen3-0.6B, een klein chat model, deze vraag {C.SEEDS} keer: *{C.QUESTIONS['bakery'][0]}* En {C.SEEDS} keer dezelfde, met *Reply with only the final number, nothing else.* erachter: geef alleen het getal. Bij welke manier kwam het goede antwoord, {C.QUESTIONS['bakery'][1]}, vaker?",
        options={
            "plain": "De vraag zoals hij is",
            "same": "Bij allebei ongeveer even vaak",
            "number": "Met alleen het getal",
        },
        reveal=C.number_reveal(L.lang),
    )
    guess_number
    return (guess_number,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **prompt** is alles wat een model krijgt voordat het schrijft. In een chat is dat niet alleen je laatste bericht, maar alle berichten tot nu toe, en elk bericht heeft een **rol**: `system` voor de instructies van wie de assistent draait, `user` voor jou, en `assistant` voor het model. Een goede prompt heeft tot vier delen: een *instructie*, wat het model moet doen; *context*, wat het moet weten, zoals de tekst waar het mee werkt; *voorbeelden* van wat je wilt; en de *vorm* van het antwoord. Een tekst die je erin plakt, zet je tussen **scheidingstekens**, zoals `<review>` en `</review>`, zodat het model je instructie en de tekst uit elkaar kan houden.

    Drie manieren van vragen hebben een naam. Vragen zonder voorbeelden heet **zero-shot**; met een paar voorbeelden in de prompt **few-shot**. Het model zeggen wie het moet zijn, zoals *Je bent een geduldige rekenleraar*, heet **role prompting**. En het vragen om eerst zijn stappen te schrijven en dan pas het antwoord, heet **chain of thought**.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | Voorbeeld |
    |---|---|---|
    | **prompt** | alles wat een model krijgt voordat het schrijft; in een chat alle berichten tot nu toe | |
    | **rol** van een bericht | van wie een bericht komt: `system` (wie de assistent draait), `user` (jij) of `assistant` (het model) | `{"role": "user", "content": "Hoi"}` |
    | **scheidingsteken** | een teken dat een deel van een prompt afbakent, zoals een tag voor en na een tekst | `<review>` … `</review>` |
    | **zero-shot prompting** | vragen zonder voorbeelden | *Vertaal dit naar het Engels: …* |
    | **few-shot prompting** | vragen met een paar voorbeelden van vragen en antwoorden in de prompt | *Nederlands: Goedemorgen. Engels: Good morning.* |
    | **role prompting** | het model zeggen wie het moet zijn | *Je bent een geduldige rekenleraar.* |
    | **chain of thought** | de stappen die een model vóór zijn antwoord schrijft, en het vragen daarom | *Werk het stap voor stap uit.* |
    """)
    return


@app.cell(hide_code=True)
def watch_out_role():
    L.watch_out("*Rol* betekent in deze les twee dingen. De *rol van een bericht* zegt wie het schreef: `system`, `user` of `assistant`. *Role prompting* zegt het model wie het moet zijn. De twee kunnen samenkomen: een role prompt als *Je bent een geduldige rekenleraar* staat vaak in het bericht met de rol `system`.")
    return


@app.cell(hide_code=True)
def picture_parts():
    mo.vstack([
        draw.marked_lines([("Een prompt met alle vier de delen", [
            ("Deel de e-mail tussen de <email>-tags in als klacht, vraag of compliment.", "1"),
            ("<email>", "2"),
            ("We hebben veertig minuten op onze pannenkoeken gewacht, en ze waren koud.", "2"),
            ("</email>", "2"),
            ("Voorbeeld: \"Hebben jullie vrijdag een tafel voor zes?\" is een vraag.", "3"),
            ("Antwoord met één woord: klacht, vraag of compliment.", "4"),
        ])], {"1": "de instructie", "2": "de context, tussen scheidingstekens", "3": "een voorbeeld", "4": "de vorm van het antwoord"},
            title="Vier delen van een prompt"),
        mo.md(r"""
        Niet elke prompt heeft alle vier nodig. Hier is één woord de goede vorm: er valt niets uit te rekenen, er is alleen een keuze. Bij een som ligt dat anders, zoals de zeven manieren hieronder laten zien.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_ways():
    mo.md(rf"""
    **Eén vraag, zeven manieren.** De course stelde Qwen3-0.6B elk van drie sommen op de zeven manieren hieronder, {C.SEEDS} keer per manier. Een model kiest elk volgend woord met een beetje toeval, dus dezelfde prompt kan een ander antwoord krijgen; een *seed* legt dat toeval vast, zodat je een run precies kunt herhalen, en {C.SEEDS} seeds laten zien wat geluk is en wat een patroon. De lengte van een antwoord telt in *tokens*, de stukjes woorden waarin [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl) het context window telde. Qwen3-0.6B is klein genoeg voor een laptop, en het schrijft het best Engels, dus de prompts zijn Engels. De les leest het laatste getal in elk antwoord. Schuif door de manieren voor de vraag over de bakker.
    """)
    return


@app.cell(hide_code=True)
def ways_cell():
    L.explore("ways", "manier", lambda _v: C.way_frame(int(_v), "bakery", L.lang), start=C.WAYS.index("number") + 1, low=1,
              high=len(C.WAYS), step=1, role="", task="Zoek de drie manieren die het minst vaak goed zijn. Wat hebben hun antwoorden gemeen?")
    return


@app.cell(hide_code=True)
def text_ways_after():
    mo.md(C.table(L.lang) + "\n\n" + C.ways_after(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vraag je een model iets goed:

    1. Zeg wat je wilt, als instructie.
    2. Geef de context die het nodig heeft, en zet een geplakte tekst tussen scheidingstekens.
    3. Heeft het antwoord een vaste vorm, laat die dan zien met een of twee voorbeelden, met hun stappen als er iets uit te rekenen valt.
    4. Valt er iets uit te rekenen, laat het model dan eerst zijn stappen schrijven, en het antwoord als laatste.
    5. Vraag aan het eind om de vorm die je nodig hebt, zoals *Answer: …* op de laatste regel, niet in plaats van de stappen.
    6. Controleer het antwoord: ook een goed gestelde vraag krijgt soms een fout antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een café wil dat een model de e-mails die binnenkomen indeelt als klacht, vraag of compliment. Een eerste poging: *Wat is deze e-mail?* met de e-mail eronder. Het model moet raden wat je wilt; misschien vat het de e-mail samen, of beantwoordt het hem. Het recept, stap voor stap:

    1. De instructie: *Deel de e-mail in als klacht, vraag of compliment.*
    2. De context: de e-mail, tussen `<email>` en `</email>`, zodat een zin in de e-mail als *Antwoord alsjeblieft snel* niet als instructie aan het model wordt gelezen.
    3. Eén voorbeeld van elke soort, zodat het model ziet waar de grens ligt: *"De pannenkoeken waren heerlijk" is een compliment.*
    4. Stappen? Er valt niets uit te rekenen, dus een kort antwoord is prima.
    5. De vorm: *Antwoord met één woord: klacht, vraag of compliment.* Nu kan een ander programma het antwoord lezen.
    6. Controleer: probeer het op e-mails waarvan je de soort al weet, zoals tien van vorige week, voordat je het de rest toevertrouwt.

    Bij de som van de bakker gaat stap 4 de andere kant op: daar moet het antwoord uitgerekend worden, en dat gebeurt in de stappen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De delen van een prompt

        Iemand vraagt een model een bericht van de verhuurder te vertalen. Deel de zes regels van die prompt in: welk deel van een prompt is elke regel?
        """),
        L.task("Zet elk van de zes regels in zijn groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parts():
    sort_parts = L.sort("parts")
    sort_parts
    return (sort_parts,)


@app.cell(hide_code=True)
def check_parts(sort_parts):
    result_parts = L.check("parts", sort_parts)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "parts",
        [
            "Vraag bij elke regel: zegt hij wat het model moet doen, wat het moet weten, hoe een goed antwoord eruitziet, of hoe het antwoord geschreven moet worden?",
            "Twee regels vertellen het model iets wat het moet weten, en één laat een antwoord zien dat al af is. Zoek eerst die drie.",
            "De instructie is de regel die zegt dat het model moet vertalen; de twee regels die het antwoord begrenzen, zijn de vorm.",
        ],
        why="*Vertaal het bericht tussen de tags naar het Engels* is de instructie. Het bericht tussen de tags en *Het bericht is van mijn verhuurder* zijn de context. *Nederlands: Goedemorgen! Engels: Good morning!* is een voorbeeld. *Geef alleen de vertaling* en *Gebruik hooguit twee zinnen* zeggen hoe het antwoord geschreven moet worden: de vorm.",
    )
    return


@app.cell(hide_code=True)
def text_sum():
    mo.md(r"""
    ### Oefening 2. Een som voor een klein model
    """)
    return


@app.cell(hide_code=True)
def choice_sum_cell():
    choice_sum = L.choice(
        "sum-prompt",
        question="Je wilt dat Qwen3-0.6B uitrekent hoeveel minuten er in drieënhalf uur zitten, en een ander programma leest het getal. Welke prompt geeft het vaakst het goede getal, als je afgaat op de tabel van de zeven manieren?",
        options={
            "professor": "*Je bent een wiskundeprofessor. Geef alleen het getal.*",
            "steps": "*Werk het stap voor stap uit. Geef daarna het antwoord op de laatste regel, als: Answer: &lt;getal&gt;*",
            "number": "*Geef alleen het getal, verder niets.*",
        },
        correct="steps",
        why={
            "professor": "Een rol verandert hoe het model schrijft, niet hoe goed het rekent, en *alleen het getal* laat het weinig ruimte om de som uit te rekenen. Kijk nog eens naar de manieren met een rol en met alleen het getal in de tabel.",
            "steps": "De stappen geven het model ruimte om het uit te rekenen, en *Answer:* op de laatste regel geeft het andere programma een vaste plek om het getal te lezen.",
            "number": "Met alleen het getal schrijft het model weinig vóór zijn antwoord, en in de tabel waren zulke korte antwoorden vaker fout. Kijk nog eens naar de manier met alleen het getal.",
        },
    )
    choice_sum
    return (choice_sum,)


@app.cell(hide_code=True)
def text_role_of():
    mo.md(r"""
    ### Oefening 3. Wie schreef het?
    """)
    return


@app.cell(hide_code=True)
def choice_role_cell():
    choice_role = L.choice(
        "message-role",
        question="De makers van een kookapp zetten *Antwoord altijd in het Nederlands, en alleen over koken.* vóór elke chat van hun gebruikers. Welke rol heeft dat bericht?",
        options={
            "user": "`user`",
            "assistant": "`assistant`",
            "system": "`system`",
        },
        correct="system",
        why={
            "user": "`user` is wie er chat. Dit bericht komt van de makers van de app, voordat de chat begint. Kijk nog eens naar de tabel met nieuwe woorden.",
            "assistant": "`assistant` is wat het model schreef. Dit bericht is voor het model geschreven, door de makers van de app. Kijk nog eens naar de tabel met nieuwe woorden.",
            "system": "Instructies van wie de assistent draait, staan in het bericht met de rol `system`, aan het begin van elke chat.",
        },
    )
    choice_role
    return (choice_role,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Vraag het Qwen3 zelf

    Schrijf je eigen prompt voor Qwen3-0.6B, en kijk wat het antwoordt: een van de drie sommen op jouw manier, of iets heel anders. Het antwoord wordt gesampled met seed 0, net als de eerste poging van de course bij elke manier. Qwen3 draait op je eigen computer, dus niets van wat je typt gaat de deur uit.
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
def prompt_box_cell():
    prompt_box = mo.ui.text_area(value=C.messages("steps", C.QUESTIONS["bakery"][0])[0]["content"],
                                 label="Je prompt", rows=5, full_width=True)
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([prompt_box, ask_button])
    return ask_button, prompt_box


@app.cell(hide_code=True)
def demo_answer(ask_button, prompt_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Schrijf eerst een prompt in het vak.*")
    else:
        _words = {"title": "Wat Qwen3 antwoordt", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden; het laatste getal erin: {read}", "nothing": "geen",
                  "load": "Het laden van Qwen3 stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.ask_live(prompt_box.value, lambda _html: mo.output.replace(mo.Html(_html)), _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Welke AI-tool waarvoor](/course/open/start-ai-tools-landscape?lang=nl) zag je dat een model alles leest wat in zijn context window staat; een prompt is wat jij daar neerzet, en de vorm ervan verandert het antwoord. De volgende les, [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl), knipt een grote taak op in kleine prompts achter elkaar, en laat een model zijn eigen antwoord controleren. In het domein *Large language models* zie je waarom een chat model überhaupt rollen heeft, en hoe chain of thought in een model getraind kan worden in plaats van erom te vragen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_role, choice_sum, result_parts):
    L.wrap_up(result_parts, choice_sum, choice_role)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
