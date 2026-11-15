# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-generation-hubs. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-generation-hubs", "nl")
    C = L.checks   # de tekenserver van de checks, de cijferlezer, de plaatjes, en de demo met je image_bytes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl), en één over [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_base64():
    quick_base64 = L.choice(
        "quick-base64",
        question="Een afbeelding gaat met een request mee als `data:image/png;base64,iVBORw0KGgo…`. Wat is de tekst na de komma?",
        options={
            "link": "Een adres waar de server de afbeelding vindt",
            "bytes": "De bytes van de PNG, als tekst geschreven",
            "words": "Een beschrijving van de afbeelding, in code",
        },
        correct="bytes",
        why={
            "bytes": "Base64 schrijft elke 3 bytes als 4 tekens, dus elk bestand past in de tekst van een JSON-request. In deze les komt een afbeelding op dezelfde manier naar je terug.",
            "link": "Een data-URL bevat het bestand zelf, niet waar het staat. Zie [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl).",
            "words": "Het is het bestand zelf, elke byte ervan, geschreven met 64 tekens. Zie [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl).",
        },
    )
    quick_base64
    return (quick_base64,)


@app.cell(hide_code=True)
def quick_check_hub():
    quick_hub = L.choice(
        "quick-hub",
        question="Je hebt een model op Hugging Face gevonden. Waar lees je van welke data het leerde, wat de licentie is en wat het niet kan?",
        options={
            "name": "In de naam van de repository",
            "weights": "In de state_dict, het bestand model.pt",
            "card": "In de model card, README.md",
        },
        correct="card",
        why={
            "card": "De model card zegt wat het model doet, zijn klasse, de data waarop het getraind is en de licentie daarvan, de licentie van het model zelf, hoe goed het werkt, en wat het niet dekt. Die lees je voordat je het model gebruikt.",
            "name": "Een naam zegt weinig, en iedereen kan hem kiezen. Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
            "weights": "De state_dict bevat alleen de getallen van de layers. Zie [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl).",
        },
    )
    quick_hub
    return (quick_hub,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een webwinkel wil een plaatje bij elk van zijn 300 producten; een leerkracht wil een tekening bij elk verhaal in een leesboek. Eén request met een beschrijving geeft een afbeelding. En als geen dienst jouw taak doet, heeft iemand er misschien een model voor getraind en het gedeeld: op een model hub staan duizenden modellen klaar, elk met zijn model card. Om er een te gebruiken, heb je ook de stappen eromheen nodig: wat erin gaat, en wat de output betekent.

    **Een voorbeeld uit het dagelijks leven.** Je kunt een taart bestellen door hem aan een bakker te beschrijven: twee lagen, rode bessen erop. Je kunt hem ook bakken met een recept uit een kookboek: dan doe je elke stap zelf, van afwegen tot de oven. Een API voor afbeeldingen is de bakker; een model van een hub met de stappen eromheen is het recept. Dat is alleen een vergelijking; hieronder zie je de requests.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **beeldgeneratie-API** | een endpoint dat afbeeldingen maakt bij een beschrijving: in het formaat van OpenAI een POST naar `/v1/images/generations` met `"prompt"`; elke afbeelding komt terug onder `"data"`, als base64 onder `"b64_json"` |
    | **videogeneratie-API** | een endpoint dat een video maakt bij een beschrijving, als job: de POST start hem en geeft een id, je vraagt de status op tot hij klaar is, en dan download je de video |
    | **pipeline** | hier: één functie om een model van een hub heen; hij bereidt de input voor zoals het model hem leerde, laat het model rekenen, en maakt van de output een antwoord |
    | `base64.b64decode(text)` | de weg terug van `base64.b64encode`: geeft de bytes die de base64-tekst `text` bevat |

    ### Een afbeelding bij een beschrijving

    Je vraagt een afbeelding zoals bij de API van OpenAI: een POST naar `/v1/images/generations` met een `"model"`, de beschrijving als `"prompt"`, en een `"size"`, zoals `"1024x1024"`. Met `"n"` vraag je om meer dan één afbeelding. Het antwoord heeft onder `"data"` één item per afbeelding, met de afbeelding als base64 onder `"b64_json"`: zoals in [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl), maar nu de andere kant op. `base64.b64decode` maakt van de tekst weer de bytes van een PNG, klaar om op te slaan of te laten zien.

    De server van de course antwoordt ook op dat endpoint, maar hij heeft geen beeldmodel: dat zou een download van gigabytes zijn. Zijn `course-shapes` is een klein tekenprogramma. Het leest cirkels, vierkanten en driehoeken in de prompt, in rood, blauw, groen, geel, paars of turquoise, tot 8 vormen, in het Engels of in het Nederlands, en tekent ze; onder `"revised_prompt"` zegt het wat het tekende. Elke afbeelding van één request heeft de vormen op andere plekken. Een echt beeldmodel tekent alles wat je beschrijft; hoe het dat doet, komt in het domein over generatieve AI.

    Echte API's voor afbeeldingen werken zo, zoals die van OpenAI, met zijn GPT Image-modellen (voorbeelden in november 2026); die sturen geen `"revised_prompt"`. Je betaalt per afbeelding, meer voor een groter formaat of een hogere kwaliteit. En een aanbieder weigert een beschrijving die zijn regels verbieden, met status 400.

    ### Een video: een job, geen antwoord

    Een video maken duurt minuten, langer dan een request kan wachten. Daarom geeft een API voor video's je een job. De POST met de beschrijving antwoordt meteen, met een id en een status zoals `queued`. Daarna vraag je om de paar seconden de status van dat id op, tot hij `succeeded` zegt, of `failed`; dan download je de video. De API van Runway werkt zo: `POST /v1/text_to_video` geeft het id van een taak, en `GET /v1/tasks/{id}` de status ervan, tot die `SUCCEEDED` of `FAILED` is (een voorbeeld in november 2026). De server van de course maakt geen video's; het plaatje toont de requests van zo'n job, in een verzonnen vorm:
    """)
    return


@app.cell(hide_code=True)
def picture_video():
    mo.Html(C.video_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_hubs():
    mo.md(r"""
    ### Model hubs en pipelines

    Op een model hub zoals Hugging Face heeft elk model een taak, die de hub laat zien; een model card kan die noemen in het blok bovenaan, onder `pipeline_tag`. De card van de cijferlezer van de course, uit [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl), begint zo:

    ```yaml
    ---
    license: cc-by-4.0
    pipeline_tag: image-classification
    ```

    Op de hub kun je zoeken op taak: `text-generation` voor chat models, `image-classification` voor modellen die een afbeelding een label geven, `text-to-image` voor beeldmodellen, en nog veel meer. Achter de meeste taken staan duizenden modellen.

    Een model alleen neemt een tensor en geeft logits. Een **pipeline** doet de stappen eromheen: hij bereidt de input voor zoals het model hem leerde, laat het model rekenen, en maakt van de output een antwoord. Voor de cijferlezer zegt de card wat erin gaat: pixels van 0 tot 1. De softmax uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) maakt van zijn 10 logits kansen:
    """)
    return


@app.cell(hide_code=True)
def picture_pipeline():
    mo.Html(C.pipeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_pipeline():
    mo.vstack([
        mo.md(r"""
        Met transformers, de library van Hugging Face, maakt één regel zo'n pipeline voor een model op de hub; de library leest op de hub hoe de input voorbereid moet worden, en wat de labels zijn:

        ```python
        from transformers import pipeline

        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        classifier("leaf.jpg")   # een lijst labels, elk met zijn score
        ```

        De course installeert transformers niet. In oefening 2 bouw je de pipeline van de cijferlezer zelf.
        """),
        L.watch_out("Een pipeline moet de input precies zo voorbereiden als het model hem leerde, en de card zegt hoe. De course gaf zijn cijferlezer de pixels van 0 tot 255, zonder door 255 te delen. Hij leest dan nog steeds 927 van de 1.000 testcijfers goed, tegen 929. Maar nu geeft hij elk cijfer een kans van minstens 0,9997, ook de 73 die hij fout leest: zijn kansen zeggen niets meer. Met de goede voorbereiding krijgen 64 van de 1.000 cijfers minder dan 0,9. Het woord pipeline heeft ook andere betekenissen: in scikit-learn is een Pipeline een keten van stappen die je samen fit, en in het dagelijks leven vervoert een pipeline olie of gas."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: een afbeelding via een API

    1. **Schrijf het request:** `{"model": ..., "prompt": ..., "size": "256x256"}`, met de beschrijving als prompt.
    2. **Stuur het:** een POST naar `url + "/images/generations"`, zoals je een chatrequest stuurt.
    3. **Haal de afbeelding eruit:** die staat in het eerste item onder `"data"`, onder `"b64_json"`, als base64-tekst.
    4. **Maak er bytes van:** `base64.b64decode` geeft de bytes van de PNG; sla ze op met `open(path, "wb")`, of laat ze zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De course vroeg een stand-in van zijn server om "two red circles and a blue square". Het request dat zijn code stuurde, wat er terugkwam, en de afbeelding:
    """)
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_worked_steps():
    mo.md(r"""
    1. **Het request:** de beschrijving onder `"prompt"`, en het formaat. Zonder `"n"` tekent de server één afbeelding.
    2. **Het antwoord:** onder `"data"` één item. Zijn `"b64_json"` heeft 2.492 tekens; het plaatje toont de eerste 24. Onder `"revised_prompt"` zegt de server wat hij tekende: 2 red circles and 1 blue square.
    3. **De bytes:** `base64.b64decode` geeft 1.868 bytes, 3 voor elke 4 tekens. Ze beginnen met de 8 bytes `\x89PNG\r\n\x1a\n`, het begin van elk PNG-bestand.
    4. **De afbeelding:** 256 bij 256 pixels, zoals het request vroeg.
    """)
    return


@app.cell(hide_code=True)
def text_image():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een afbeelding via de API

        `image_bytes(url, prompt)` stuurt het request en leest het antwoord al. Vul de drie `...` in: de beschrijving, het endpoint voor afbeeldingen, en de bytes van de eerste afbeelding. De check roept hem aan met een stand-in van de server, die tekent zoals de server van de course. Het plaatje onder de check laat zien wat je functie stuurde, wat er terugkwam, en de afbeelding die je functie teruggaf.
        """),
        L.task("Maak `image_bytes` af: stuur `prompt` naar het endpoint voor afbeeldingen, en geef de bytes van de eerste afbeelding in het antwoord terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_image():
    box_image = L.code_box("image", lines=12)
    box_image
    return (box_image,)


@app.cell(hide_code=True)
def check_image(box_image):
    result_image = L.check("image", box_image.value)
    result_image
    return (result_image,)


@app.cell(hide_code=True)
def hints_image():
    L.hints(
        "image",
        [
            "Stap 1 tot 4 van het recept: wat er onder `\"prompt\"` komt, waar het request heen gaat, en wat je uit het antwoord haalt.",
            "De beschrijving is de `prompt` die de functie krijgt; het endpoint is `/images/generations`, na `url`. De afbeelding staat in het eerste item onder `\"data\"`, onder `\"b64_json\"`, als base64-tekst.",
            "`\"prompt\": prompt`, `url + \"/images/generations\"`, en `return base64.b64decode(...)`, met de base64-tekst van de eerste afbeelding op de plaats van de puntjes.",
        ],
        why="""```python
def image_bytes(url, prompt):
    body = {"model": "course-shapes", "prompt": prompt, "size": "256x256"}
    request = urllib.request.Request(url + "/images/generations", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.loads(response.read())
    return base64.b64decode(answer["data"][0]["b64_json"])
```

`answer["data"]` is de lijst afbeeldingen, `[0]` de eerste, en `["b64_json"]` zijn base64-tekst; `b64decode` maakt daar de bytes van de PNG van.""",
    )
    return


@app.cell(hide_code=True)
def text_digit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een eigen pipeline

        Nu een model van een hub, met je eigen pipeline eromheen: de cijferlezer van de course. `read_digit(model, pixels)` krijgt de lezer en de 784 pixels van een testcijfer, als een lijst gehele getallen van 0 tot 255, rij na rij. Hij geeft het cijfer en zijn kans terug. Vul de twee `...` in: de pixels voorbereid zoals de model card zegt, en de 10 kansen van het cijfer. Twee dingen helpen:

        - `torch.tensor(pixels, dtype=torch.float32)` maakt van de lijst een tensor met floats. De lezer maakt van de 784 pixels zelf een afbeelding van 28 bij 28, dus hij geeft één rij van 10 logits: vorm `(1, 10)`.
        - `torch.softmax(t, dim=1)` is de softmax uit [Veel klassen: softmax](/course/open/ml-softmax?lang=nl) in PyTorch, langs as 1 van `t`: over de 10 logits van elke rij.

        De check roept hem aan met `reader`, de cijferlezer van de course, en `digits[36]`, de pixels van testcijfer 36, en daarna met nog drie cijfers. Het plaatje onder de check toont elk cijfer met wat je `read_digit` zegt.
        """),
        L.task("Maak `read_digit` af: bereid de pixels voor zoals het model ze leerde, en maak van de 10 logits van de lezer kansen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_digit():
    box_digit = L.code_box("read_digit", lines=10)
    box_digit
    return (box_digit,)


@app.cell(hide_code=True)
def check_digit(box_digit):
    result_digit = L.check("read_digit", box_digit.value)
    result_digit
    return (result_digit,)


@app.cell(hide_code=True)
def hints_digit():
    L.hints(
        "read_digit",
        [
            "De card zegt wat het model neemt: een tensor met pixels van 0 tot 1. En het model geeft logits: welke functie maakt van 10 logits kansen?",
            "Delen door 255 brengt pixels van 0 tot 255 naar 0 tot 1. `model(x)` heeft vorm `(1, 10)`: neem de softmax langs as 1, en dan rij 0.",
            "`x = torch.tensor(pixels, dtype=torch.float32) / ...` en `p = torch.softmax(model(x), dim=1)[...]`, met de goede getallen op de plaats van de puntjes.",
        ],
        why="""```python
def read_digit(model, pixels):
    x = torch.tensor(pixels, dtype=torch.float32) / 255
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    digit = int(p.argmax())
    return digit, float(p[digit])
```

Voorbereiden, het model, en het antwoord: de drie stappen van een pipeline. Voor testcijfer 36, een 7, geeft hij `(7, 0.897)`; voor testcijfer 62, een 9, geeft hij `(1, 0.693)`: fout, en minder zeker.""",
    )
    return


@app.cell(hide_code=True)
def text_video_job():
    mo.md(r"""
    ### Oefening 3. Er komt een video aan
    """)
    return


@app.cell(hide_code=True)
def choice_video_cell():
    choice_video = L.choice(
        "video-job",
        question="Een API voor video's antwoordt op je POST met `{\"id\": \"video_abc\", \"status\": \"queued\"}`. Wat doet je programma nu?",
        options={
            "poll": "Af en toe de status opvragen, tot hij klaar is",
            "decode": "De video uit dit antwoord decoderen, zoals bij een afbeelding",
            "again": "De POST opnieuw sturen tot de video zelf terugkomt",
        },
        correct="poll",
        why={
            "poll": "Het antwoord zegt alleen dat de job begonnen is: `queued`, in de rij. Een video maken duurt minuten, dus je vraagt om de paar seconden de status van `video_abc` op. Die gaat van `queued` naar `running` naar `succeeded`, of `failed`; dan download je de video. Tien keer per seconde vragen kost requests en levert niets op.",
            "decode": "Zit er een video in dit antwoord? Kijk wat erin staat: een id en een status.",
            "again": "Elke POST start een nieuwe job, en je betaalt voor elke video. Hoe hoor je hoe het gaat met de job die je al gestart hebt?",
        },
    )
    choice_video
    return (choice_video,)


@app.cell(hide_code=True)
def text_hub_task():
    mo.md(r"""
    ### Oefening 4. Welke taak?
    """)
    return


@app.cell(hide_code=True)
def choice_task_cell():
    choice_task = L.choice(
        "hub-task",
        question="Je wilt een model dat zegt welke van 20 plantenziektes een foto van een blad laat zien. Op welke taak zoek je op de hub?",
        options={
            "t2i": "text-to-image: een beschrijving erin, een afbeelding eruit",
            "cls": "image-classification: een afbeelding erin, een label eruit",
            "i2t": "image-to-text: een afbeelding erin, een tekst erover eruit",
        },
        correct="cls",
        why={
            "cls": "Twintig ziektes zijn twintig labels, en er gaat een foto in: image classification, de taak van de cijferlezer van de course, met `pipeline_tag: image-classification` in zijn card. Op de hub zoek je dan een model dat op bladeren getraind is, en lees je zijn card.",
            "t2i": "Die taak maakt afbeeldingen bij een beschrijving. Wat gaat er hier in, en wat moet eruit komen?",
            "i2t": "Zo'n model schrijft over een afbeelding. Heb je een tekst nodig, of één van 20 labels?",
        },
    )
    choice_task
    return (choice_task,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Teken met je eigen image_bytes

    Zodra oefening 1 goed is, kun je met je eigen `image_bytes` de server van de course om een afbeelding vragen, op je computer. Beschrijf tot 8 cirkels, vierkanten en driehoeken, in rood, blauw, groen, geel, paars of turquoise, in het Nederlands of in het Engels. Er wordt niets gedownload: de server tekent.
    """)
    return


@app.cell(hide_code=True)
def prompt_box_cell():
    prompt_box = mo.ui.text(value="drie gele driehoeken en een paars vierkant", label="Je beschrijving", full_width=True)
    draw_button = mo.ui.run_button(label="Teken met je image_bytes")
    mo.vstack([prompt_box, draw_button])
    return draw_button, prompt_box


@app.cell(hide_code=True)
def demo_image(box_image, draw_button, prompt_box):
    _draw = C.learner_image(box_image.value)
    if _draw is None:
        _out = mo.md("*Maak eerst oefening 1 af: de demo vraagt het met je eigen `image_bytes`.*")
    elif not draw_button.value:
        _out = mo.md("*Druk op de knop om te tekenen.*")
    elif not prompt_box.value.strip():
        _out = mo.md("*Schrijf eerst een beschrijving in het vak.*")
    else:
        _words = {"title": "Wat je image_bytes terugkreeg",
                  "drawn": "De server las: {drawn}.",
                  "done": "{seconds} seconden.",
                  "failed": "Je image_bytes stopte met een fout toen hij de server van de course vroeg: {why}",
                  "not_png": "Je image_bytes gaf iets anders terug dan de bytes van een PNG."}
        _out = mo.Html(C.demo(_draw, prompt_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het request is dat van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), naar een ander endpoint, en base64 is dat uit [Meer dan tekst: beelden, audio en realtime](/course/open/build-multimodal-apis?lang=nl), nu op de weg terug. De cijferlezer en zijn model card komen uit [Een pretrained model gebruiken](/course/open/torch-pretrained?lang=nl). De volgende les, [Een groot model of een klein?](/course/open/build-llm-or-small-model?lang=nl), zet een eigen klein model tegenover een LLM, voor één vaste taak. Hoe een beeldmodel een afbeelding maakt bij een beschrijving, komt in het domein over generatieve AI.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_task, choice_video, result_digit, result_image):
    L.wrap_up(result_image, result_digit, choice_video, choice_task)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
