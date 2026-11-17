# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-ai-services. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-ai-services", "nl")
    C = L.checks   # de run van de course met Qwen3 op 120 vragen van klanten, de stand-in van de checks, en de demo met je label


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_endpoint():
    quick_endpoint = L.choice(
        "quick-endpoint",
        question="Naar welk endpoint van een OpenAI-compatible server stuur je de berichten van een chat?",
        options={
            "chat": "`/v1/chat/completions`",
            "models": "`/v1/models`",
            "answers": "`/v1/answers`",
        },
        correct="chat",
        why={
            "chat": "Dat endpoint neemt de berichten aan en geeft het antwoord; hetzelfde adres bij elke OpenAI-compatible server.",
            "models": "Dat geeft de modellen die een server heeft, met een GET. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "answers": "Zo'n endpoint bestaat niet in de API van OpenAI. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_endpoint
    return (quick_endpoint,)


@app.cell(hide_code=True)
def quick_check_finish():
    quick_finish = L.choice(
        "quick-finish",
        question="Je vraagt met `\"max_tokens\": 20`, en het antwoord heeft `\"finish_reason\": \"stop\"`. Wat zegt dat?",
        options={
            "cut": "De server knipte het antwoord af bij 20 tokens",
            "error": "Er ging iets mis bij de server",
            "ended": "Het model beëindigde het antwoord zelf, vóór 20 tokens",
        },
        correct="ended",
        why={
            "ended": "`\"stop\"` betekent dat het model zelf zijn einde schreef; `\"length\"` zou betekenen dat het `max_tokens` bereikte.",
            "cut": "Dan zou er `\"length\"` staan. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "error": "Een fout komt terug als foutstatus, zonder antwoord. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_finish
    return (quick_finish,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel van het werk dat een app aan AI vraagt, is één vaste taak, steeds opnieuw: een opgenomen gesprek omzetten in tekst, een handleiding vertalen, het bedrag op een bonnetje lezen, de berichten van klanten in de goede wachtrij zetten. Voor veel van die taken zijn er kant-en-klare diensten achter een API, gebouwd voor die ene taak. En een LLM kan de meeste ook, met een prompt. Deze les laat allebei zien, en meet er één: hoe goed Qwen3-0.6B 120 vragen van klanten van een bank sorteert.

    **Een voorbeeld uit het dagelijks leven.** Een sorteermachine bij de post leest de postcode op elke envelop en legt hem in de goede bak: één taak, heel snel, en verder niets. Een medewerker die de hele brief leest, kan veel meer, ook sorteren, maar langzamer. Dat is alleen een vergelijking; hieronder zie je de kant-en-klare diensten, en een LLM dat sorteert.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **AI-dienst** | een kant-en-klaar model achter een API voor één taak, zoals de tekst op een foto lezen of vertalen: je stuurt een foto, geluid of tekst, en krijgt de uitkomst terug, zonder prompt |
    | **spraak naar tekst** | geluid erin, de woorden die gezegd werden eruit; *tekst naar spraak* is andersom |
    | **OCR** | *optical character recognition*: de tekst in een beeld lezen, zoals een gescande pagina of een bonnetje |
    | **klassieke NLP-taak** | een taak op taal die al lang vóór LLM's bestond: *sentiment* (is een tekst positief of negatief), *named entities* (de namen van mensen, plaatsen, bedrijven en datums in een tekst), *classificatie* (een tekst in labels sorteren), samenvatten, vertalen |
    | **label** | een van een vaste reeks antwoorden die een classificatie kan geven; hier waar het bericht van een klant over gaat, zoals `pin_blocked` |
    | `"temperature": 0` | in de body van een request: het model neemt elke keer zijn waarschijnlijkste volgende token, dus hetzelfde bericht krijgt hetzelfde antwoord |
    | `text.lower()` | de tekst in kleine letters: `"Exchange_Rate".lower()` is `"exchange_rate"` |

    ### Kant-en-klare diensten

    | Taak | Wat erin gaat, en wat eruit komt | Voorbeelden (november 2026) |
    |---|---|---|
    | spraak naar tekst | geluid; de woorden | het transcriptie-endpoint van OpenAI, Google Cloud Speech-to-Text, Amazon Transcribe |
    | tekst naar spraak | tekst; geluid | Google Cloud Text-to-Speech, Amazon Polly, ElevenLabs |
    | vertalen | tekst; dezelfde tekst in een andere taal | DeepL, Google Cloud Translation, Azure AI Translator |
    | zien en OCR | een beeld; de voorwerpen erop, of de tekst erin | Google Cloud Vision, Azure AI Vision, Amazon Rekognition en Textract |
    | klassieke NLP | tekst; het sentiment, de named entities, de labels | Google Cloud Natural Language, Azure AI Language, Amazon Comprehend |

    Elke dienst heeft zijn eigen endpoint en JSON, maar de manier van aanroepen ken je: een HTTP-request met de invoer, en JSON terug. Een dienst geeft elke keer dezelfde soort antwoord, wat een programma makkelijk kan lezen, en is meestal snel en goedkoop per request. Maar hij doet alleen zijn eigen taak, met zijn eigen labels.

    ### Of een LLM met een prompt

    Een LLM doet deze taken met een prompt: "Vertaal dit naar het Nederlands", "Wat is het sentiment van deze review?", of een lijst met je eigen labels. De course probeerde het: BANKING77 is een verzameling vragen van klanten van een online bank, elk met zijn label (Casanueva en anderen, 2020). De course nam 20 vragen voor elk van zes labels, 120 in totaal, en stuurde ze een voor een naar zijn server met deze system prompt, met `"temperature": 0`:

    ```text
    Sort the customer's message into one of these labels, and answer with the label only.
    card_arrival: the new card has not arrived yet
    lost_or_stolen_card: the card is lost or stolen
    exchange_rate: a question about exchange rates
    card_payment_wrong_exchange_rate: a card payment used the wrong exchange rate
    pin_blocked: the PIN is blocked
    request_refund: the customer wants money back for something
    ```

    Dit is een prompt zonder voorbeelden, *zero-shot*, zoals in [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl). De tabel laat zien wat Qwen3 antwoordde: elke rij is een label waar de vragen over gaan, elke kolom een label dat Qwen3 gaf. De cellen op de diagonaal, waar die twee gelijk zijn, zijn de goede antwoorden.
    """)
    return


@app.cell(hide_code=True)
def picture_table():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_table():
    mo.vstack([
        mo.md(r"""
        Qwen3 had er 99 van de 120 goed, 82,5%, en elk antwoord dat het schreef, was precies een van de zes labels. De fouten zitten vooral op één plek: van de 20 vragen over een kaartbetaling met de verkeerde wisselkoers zette het er 9 onder `exchange_rate`, het label dat een stuk van het langere is. Elk antwoord duurde ongeveer 1,0 seconde op de computer van de course, met ongeveer 116 tokens erin, vooral de system prompt, en 3 tokens eruit.
        """),
        L.watch_out("Een model zit niet vast aan je lijst: een ander model, of een andere prompt, schrijft `Label: pin_blocked.` of `Pin_Blocked`, of een label dat niet op de lijst staat. Lees het label zorgvuldig uit het antwoord, en tel het als geen label als er geen in staat."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een tekst in labels sorteren met een LLM

    1. **Schrijf een system prompt** die de labels opsomt, elk met één regel over wat het betekent, en die alleen om het label vraagt.
    2. **Verstuur hem** als eerste bericht, met de tekst als bericht van de gebruiker, en `"temperature": 0`.
    3. **Lees het label uit het antwoord:** neem van de labels die in het antwoord staan, in kleine letters, het langste; `None` als er geen in staat.
    4. **Meet het** op teksten waarvan je het label kent, voordat je erop vertrouwt, zoals de course deed met zijn 120 vragen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een van de 120 vragen: *I believe my card payment exchange rate is incorrect.* Het label is `card_payment_wrong_exchange_rate`.

    1. **Het request:** de system prompt, en dan de vraag als bericht van de gebruiker: er gingen 113 tokens in.
    2. **Het antwoord:** Qwen3 schreef `card_payment_wrong_exchange_rate`, in 1,3 seconde.
    3. **Het label:** er staan twee labels in dat antwoord, `exchange_rate` en `card_payment_wrong_exchange_rate`, omdat het eerste een stuk van het tweede is. Een functie die stopt bij het eerste label uit de lijst dat hij vindt, zou `exchange_rate` zeggen, en dat is fout. Het langste is goed.
    """)
    return


@app.cell(hide_code=True)
def text_parse():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het label in een antwoord

        Maak `parse_label(answer, labels)` af: hij geeft het langste label uit `labels` terug dat in `answer` staat, ook als het antwoord hoofdletters of woorden eromheen heeft, en `None` als er geen label in staat. De check probeert antwoorden met hoofdletters, met woorden om het label, en zonder label. De tabel onder de check leest de 120 antwoorden van Qwen3 met je functie.
        """),
        L.task("Vul de loop in: als `label` in kleine letters in het antwoord staat, en `best` nog `None` of korter is, wordt `label` de nieuwe `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parse():
    box_parse = L.code_box("parse_label", lines=6)
    box_parse
    return (box_parse,)


@app.cell(hide_code=True)
def check_parse(box_parse):
    result_parse = L.check("parse_label", box_parse.value)
    result_parse
    return (result_parse,)


@app.cell(hide_code=True)
def hints_parse():
    L.hints(
        "parse_label",
        [
            "Voor elk label twee vragen: staat het in het antwoord, en is het langer dan het beste tot nu toe?",
            "`label in answer.lower()` zegt of het label in het antwoord staat, wat de hoofdletters ook zijn; `len(label) > len(best)` of het langer is, en dat werkt pas als `best` geen `None` meer is.",
            "`if label in answer.lower() and (___ or len(label) > len(best)):`, met `best = label` eronder, verder ingesprongen.",
        ],
        why="""```python
def parse_label(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            best = label
    return best
```

`best is None or ...` laat het eerste gevonden label erin, en daarna alleen een langer; Python kijkt niet eens naar `len(best)` zolang `best` nog `None` is.""",
    )
    return


@app.cell(hide_code=True)
def text_label():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een bericht sorteren via de API

        `label(url, message)` verstuurt al een request en leest het antwoord. Vul de messages in, eerst de system prompt `SYSTEM` en dan het bericht van de klant, en geef het label terug, dat je `parse_label` uit oefening 1 uit de tekst van het antwoord leest. `SYSTEM` en `LABELS` staan klaar om te gebruiken. De check roept je functie aan met een stand-in van de server, die één keer antwoordt met alleen een label en één keer met woorden eromheen.
        """),
        L.task("Maak van `messages` de system prompt en het bericht, en geef het label terug dat je `parse_label` uit het antwoord leest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label", lines=12)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label, box_parse):
    result_label = L.check("label", box_label.value, C.label_inputs(box_parse.value))
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label",
        [
            "Stap 2 en 3 van het recept: twee berichten, en dan het label uit de tekst van het antwoord.",
            "`messages` is een lijst met twee dictionaries: een met de rol `\"system\"` en `SYSTEM`, een met de rol `\"user\"` en `message`. De tekst van het antwoord zit onder `\"choices\"`: in de eerste keuze, in zijn `\"message\"`, en daarin onder `\"content\"`.",
            "De laatste regel: `return parse_label(___, LABELS)`, met de tekst van het antwoord in het gat.",
        ],
        why="""```python
def label(url, message):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    body = {"model": "qwen3-0.6b", "messages": messages, "max_tokens": 20, "temperature": 0}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return parse_label(answer["choices"][0]["message"]["content"], LABELS)
```

De system prompt vertelt het model de labels; je `parse_label` maakt van wat het ook schrijft een van die labels, of `None`.""",
    )
    return


@app.cell(hide_code=True)
def text_own_labels():
    mo.md(r"""
    ### Oefening 3. Een dienst of een LLM?
    """)
    return


@app.cell(hide_code=True)
def choice_own_labels_cell():
    choice_own_labels = L.choice(
        "own-labels",
        question="Een bank wil elk bericht van een klant sorteren in zes labels die hij zelf bedacht, zoals `pin_blocked`. Hij heeft nog geen gesorteerde voorbeelden. Wat past?",
        options={
            "sentiment": "Een kant-en-klare sentimentdienst, die zegt of een tekst positief of negatief is",
            "llm": "Een LLM met een system prompt die de zes labels opsomt",
            "translate": "Een kant-en-klare vertaaldienst",
        },
        correct="llm",
        why={
            "llm": "Een prompt kan elk label noemen, ook labels die niemand eerder maakte; een kant-en-klare dienst kent alleen zijn eigen. Met gesorteerde voorbeelden kan ook een eigen klein model, zoals een latere les laat zien.",
            "sentiment": "Die dienst geeft zijn eigen labels, positief en negatief. Kent hij `pin_blocked`?",
            "translate": "Die zet een bericht om in een andere taal, maar sorteert hij het?",
        },
    )
    choice_own_labels
    return (choice_own_labels,)


@app.cell(hide_code=True)
def text_mixed():
    mo.md(r"""
    ### Oefening 4. Lees de tabel
    """)
    return


@app.cell(hide_code=True)
def choice_mixed_cell():
    choice_mixed = L.choice(
        "mixed",
        question="Welke fout maakte Qwen3 het vaakst, in de tabel van de run van de course?",
        options={
            "rate": "Het antwoordde `exchange_rate` bij een kaartbetaling met de verkeerde wisselkoers",
            "refund": "Het antwoordde `request_refund` bij een verloren of gestolen kaart",
            "pin": "Het antwoordde `pin_blocked` bij een kaart die nog niet aangekomen is",
        },
        correct="rate",
        why={
            "rate": "9 van die 20 vragen kregen het kortere label; de twee liggen dicht bij elkaar in betekenis, en het ene is een stuk van het andere.",
            "refund": "Kijk naar de rij van `lost_or_stolen_card`: hoeveel vragen staan er in de kolom van `request_refund`?",
            "pin": "Kijk naar de rij van `card_arrival`: hoeveel vragen staan er in de kolom van `pin_blocked`?",
        },
    )
    choice_mixed
    return (choice_mixed,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Sorteer je eigen bericht met je label

    Zodra oefening 1 en 2 goed zijn, kun je `label` met de echte server laten praten: de course start hem op je computer, met Qwen3-0.6B erachter. Schrijf een bericht zoals een klant van de bank het zou schrijven, in het Engels zoals de vragen van BANKING77, en kijk welk label terugkomt. Probeer ook een bericht dat bij geen van de zes past.
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
def message_box_cell():
    message_box = mo.ui.text(value="I paid in euros in Spain, and the rate on my card was way off.", label="Een bericht van een klant", full_width=True)
    send_button = mo.ui.run_button(label="Sorteer het met je label")
    mo.vstack([message_box, send_button])
    return message_box, send_button


@app.cell(hide_code=True)
def demo_answer(box_label, box_parse, message_box, send_button):
    _label = C.learner_label(box_label.value, box_parse.value)
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif _label is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo sorteert met je eigen `label`.*")
    elif not send_button.value:
        _out = mo.md("*Druk op de knop om het bericht te sorteren.*")
    elif not message_box.value.strip():
        _out = mo.md("*Schrijf eerst een bericht in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `label` heeft het bericht verstuurd; de server laadt Qwen3, dat zijn antwoord schrijft …*"))
        _words = {"title": "Het label dat je label las",
                  "label": "{label}", "no_label": "None: geen label in het antwoord",
                  "wrote": "Qwen3 schreef: {text} ({seconds} seconden)",
                  "failed": "Je label stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_label, message_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het request is dat van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), met een system prompt zoals in [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl). In [Een groot model of een klein?](/course/open/build-llm-or-small-model?lang=nl) neemt een klein model, getraind op 50 gesorteerde vragen per label, het op tegen Qwen3 op dezelfde 120 vragen, in kwaliteit, snelheid en prijs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_mixed, choice_own_labels, result_label, result_parse):
    L.wrap_up(result_parse, result_label, choice_own_labels, choice_mixed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
