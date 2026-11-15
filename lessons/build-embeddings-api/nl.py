# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-embeddings-api. Route: course/route/lessen/bouwen.yaml. Checks, de plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-embeddings-api", "nl")
    C = L.checks   # de opgenomen runs van Granite Embedding, de stand-in van de checks, en de demo met je best_match


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), en één over [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_usage():
    quick_usage = L.choice(
        "quick-usage",
        question="Een antwoord van de server van de course zegt `\"usage\": {\"prompt_tokens\": 19, \"completion_tokens\": 13, \"total_tokens\": 32}`. Wat zijn die 19?",
        options={
            "sent": "De tokens die je request meestuurde",
            "written": "De tokens die het model schreef",
            "words": "De woorden van je vraag",
        },
        correct="sent",
        why={
            "sent": "`prompt_tokens` telt wat erin ging, `completion_tokens` wat eruit kwam. Een antwoord met embeddings heeft alleen het eerste: er wordt niets geschreven.",
            "written": "Dat zijn de 13 onder `completion_tokens`. Wat gaat er in een prompt? Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
            "words": "Een server telt tokens, geen woorden, en een token is vaak een stuk van een woord. Zie [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl).",
        },
    )
    quick_usage
    return (quick_usage,)


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Twee embeddings hebben een cosine similarity van 0,98. Wat zegt dat?",
        options={
            "length": "Ze zijn bijna even lang",
            "far": "Ze liggen ver uit elkaar",
            "same": "Ze wijzen bijna dezelfde kant op",
        },
        correct="same",
        why={
            "same": "Cosine similarity kijkt alleen naar de richting: 1 is precies dezelfde kant op, 0 haaks, −1 de andere kant op. Bijna dezelfde kant op betekent dat het model de twee dingen als gelijksoortig ziet.",
            "length": "De cosine similarity deelt de lengtes weg: een lange en een korte pijl in dezelfde richting geven ook 1. Zie [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
            "far": "Dicht bij 1 betekent dicht bij elkaar in richting. Zie [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een klant van een webwinkel typt "How do I change my password?". De hulppagina met het antwoord zegt "Je wachtwoord wijzig je onder Instellingen, bij Account." Geen woord is hetzelfde, zelfs de taal niet, dus zoeken op woorden vindt niets. Zoeken op betekenis vindt die pagina meteen. Daarvoor heeft elke tekst een embedding nodig, en een embeddings-API geeft je die: veel teksten in één request. En soms gaat een vraag over een heel document, een contract of een handleiding; dan moet het document met het request mee.

    **Een voorbeeld uit het dagelijks leven.** In een bibliotheek staan de boeken op onderwerp, niet op de kleur van hun kaft: een boek over fietsen staat tussen andere boeken over fietsen, wat de titel ook zegt, en in welke taal het ook geschreven is. Een embedding geeft elke tekst zo'n plek op een plank. Dat is alleen een vergelijking; hieronder zie je de getallen.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **embedding model** | een model dat geen tekst schrijft: het maakt van een tekst één embedding, altijd evenveel getallen, zodat teksten met dezelfde betekenis embeddings krijgen die dezelfde kant op wijzen |
    | **embeddings-API** | het endpoint `/v1/embeddings`: je stuurt `"input"`, een tekst of een lijst teksten, en krijgt onder `"data"` één `"embedding"` per tekst terug, in dezelfde volgorde, elk met zijn `"index"` |
    | **bestandsupload** | een bestand één keer naar een aanbieder sturen, los van een request: je krijgt een id terug, zoals `file-abc123`, en latere requests noemen het bestand bij dat id |

    ### Veel teksten, één request

    Naast Qwen3 heeft de server van de course een embedding model: Granite Embedding 107M multilingual, van IBM, een download van 223 MB. Het geeft 384 getallen per tekst, en kent twaalf talen, waaronder Engels en Nederlands. De course stuurde het vier vragen van klanten in het Engels en dezelfde vier in het Nederlands, alle acht in één request, en rekende het inproduct uit van elke Engelse vraag met elke Nederlandse:
    """)
    return


@app.cell(hide_code=True)
def picture_pairs():
    mo.Html(C.pairs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_pairs():
    mo.vstack([
        mo.md(r"""
        Granite geeft elke embedding lengte 1, dus het inproduct is al de cosine similarity uit [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl): er zijn geen lengtes om door te delen. Een vraag en haar vertaling scoren 0,87 tot 0,96; een vraag en een andere vraag 0,51 tot 0,64. Niet 0: bij Granite scoort zelfs "The sun is a star." nog 0,473 tegen de vraag over het wachtwoord. Een score alleen zegt dus weinig; het gaat erom welke tekst het hoogst scoort.
        """),
        L.watch_out("Embeddings van twee modellen passen niet bij elkaar: elk model heeft zijn eigen coördinaten. Stap je over op een ander embedding model, dan maak je opnieuw de embeddings van al je teksten. En een model leest maar zoveel tokens van een tekst, Granite 512: de rest van een langere tekst valt weg, of een andere API weigert hem."),
    ])
    return


@app.cell(hide_code=True)
def text_documents():
    mo.md(r"""
    ### Documenten mee met een request

    Een vraag over een contract, een handleiding of een rapport heeft het document zelf nodig. Het document kan op twee manieren met het request mee.

    **Als tekst in het bericht.** Je programma leest het bestand, en zet de tekst in de prompt tussen delimiters, zoals in [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl). Elke chat-API neemt dat aan, ook die van de course. De tokens van het document tellen in elk request mee, en het moet in het context window passen.

    **Als bestandsupload.** Je programma stuurt het bestand één keer naar de aanbieder, naar zijn endpoint voor bestanden, en krijgt een id terug. Elk later request noemt het bestand bij dat id; in de chat-API van OpenAI is de content van het bericht dan een lijst, met het stuk `{"type": "file", "file": {"file_id": "file-abc123"}}` naast de vraag. Van een pdf lezen veel aanbieders de tekst en kijken ze ook naar elke pagina als beeld, zodat tabellen en grafieken ook meetellen. OpenAI, Anthropic en Google hebben elk zo'n API voor bestanden (voorbeelden in november 2026). De server van de course heeft er geen: een upload is iets voor de servers van een aanbieder.

    Houd er rekening mee dat een geüpload bestand bij de aanbieder blijft tot je het verwijdert, of tot de aanbieder het na een vaste tijd weghaalt.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: de tekst vinden die het best bij een vraag past

    1. **Zet alle teksten in één lijst,** de vraag vooraan: `[question] + texts`.
    2. **Vraag hun embeddings op:** een POST naar `url + "/embeddings"` met de body `{"model": ..., "input": die_lijst}`, zoals je een chatrequest verstuurt.
    3. **Haal de embeddings eruit:** elk item onder `"data"` bevat de `"embedding"` van één tekst, in de volgorde waarin je ze stuurde.
    4. **Geef elke tekst een score:** het inproduct van de embedding van de vraag, `vectors[0]`, met die van de tekst; voor embeddings met lengte 1 is dat de cosine similarity.
    5. **Neem de tekst met de hoogste score.**
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De vraag van de course over het wachtwoord, met twee hulppagina's: een Nederlandse over het wachtwoord, een Engelse over de bezorging. Het request dat de code stuurde, en wat er terugkwam:
    """)
    return


@app.cell(hide_code=True)
def picture_worked():
    mo.Html(C.worked_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_worked_steps():
    mo.md(r"""
    1. **De input:** één lijst met drie teksten, dus één request geeft alle drie de embeddings.
    2. **Het antwoord:** onder `"data"` drie items, met `"index"` 0, 1 en 2, in de volgorde van de teksten. Elke embedding heeft 384 getallen; het plaatje laat de eerste vier zien.
    3. **De tokens:** `"prompt_tokens": 35`: 9, 16 en 10 voor de drie teksten. Er wordt niets geschreven, dus er zijn geen `completion_tokens`.
    4. **De scores:** de vraag met de Nederlandse pagina 0,686, met de Engelse pagina over de bezorging 0,489. De Nederlandse pagina wint, zonder één woord van de vraag erin.
    5. **Eén request:** de vraag met alle zes hulppagina's duurde 0,025 seconde in één request op de computer van de course, en 0,072 seconde in 7 requests van één tekst elk, 2,9 keer zo lang. Bij een gehoste API reist elk request ook nog over het internet.
    """)
    return


@app.cell(hide_code=True)
def text_search():
    mo.md(r"""
    De course stelde zo drie vragen tegen de zes hulppagina's van een verzonnen webwinkel, drie pagina's in het Engels en drie in het Nederlands. Onder elke vraag elke pagina met haar score:
    """)
    return


@app.cell(hide_code=True)
def picture_search():
    mo.Html(C.search_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_embed():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Embeddings via de API

        `embed(url, texts)` stuurt al een request en leest het antwoord, zoals je `ask` deed in [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl). Vul de drie `...` in: de teksten als input, het endpoint voor embeddings, en de lijst met embeddings, één per tekst. De check roept hem aan met een stand-in van de server, die meteen antwoordt met verzonnen embeddings: 384 getallen per tekst, willekeurig getrokken, dus ze betekenen niets. Het plaatje onder de check laat zien wat je functie stuurde, en wat er terugkwam.
        """),
        L.task("Maak `embed` af: stuur `texts` als input naar het endpoint voor embeddings, en geef de lijst met hun embeddings terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_embed():
    box_embed = L.code_box("embed", lines=11)
    box_embed
    return (box_embed,)


@app.cell(hide_code=True)
def check_embed(box_embed):
    result_embed = L.check("embed", box_embed.value)
    result_embed
    return (result_embed,)


@app.cell(hide_code=True)
def hints_embed():
    L.hints(
        "embed",
        [
            "Stap 2 en 3 van het recept: wat erin gaat, waar het heen gaat, en wat je uit het antwoord haalt.",
            "De teksten gaan onder `\"input\"` als de lijst zelf; het endpoint is `/embeddings`, na `url`. In het antwoord heeft elk item onder `\"data\"` zijn getallen onder `\"embedding\"`.",
            "`\"input\": texts`, `url + \"/embeddings\"`, en `return [item[...] for item in answer[...]]`, met de goede keys op de plaats van de puntjes.",
        ],
        why="""```python
def embed(url, texts):
    body = {"model": "granite-embedding-107m-multilingual", "input": texts}
    request = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.loads(response.read())
    return [item["embedding"] for item in answer["data"]]
```

Eén request geeft de embeddings van alle teksten; de list comprehension haalt de getallen uit elk item, in de volgorde van de teksten.""",
    )
    return


@app.cell(hide_code=True)
def text_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De beste match

        Zoek nu de tekst die het best bij een vraag past, zoals in het recept. `best_match(url, question, texts)` heeft de loop al. Vul de twee `...` in: de embeddings van de vraag en alle teksten in één request, met je `embed` uit oefening 1, en de score van `texts[i]`. De check roept hem twee keer aan met een stand-in, waarvan de embeddings voor deze teksten zo gemaakt zijn dat elke tekst ongeveer scoort wat Granite hem geeft. Het plaatje onder de check laat de scores zien, en de tekst die je functie koos.
        """),
        L.task("Maak `best_match` af: haal de embeddings van de vraag en de teksten in één request op, en geef elke tekst als score het inproduct met de vraag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best_match", lines=11)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best, box_embed):
    result_best = L.check("best_match", box_best.value, C.best_inputs(box_embed.value))
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best_match",
        [
            "Stap 1 en 4 van het recept: eerst één lijst met de vraag vóór de teksten, dan voor elke tekst het inproduct met de vraag.",
            "`[question] + texts` is die lijst, en je `embed` geeft de embeddings ervan. `vectors[0]` is dan die van de vraag; op welke plek staat de embedding van `texts[i]`?",
            "`vectors = embed(url, [question] + texts)`, en `score = np.dot(q, vectors[...])`, met de plek van `texts[i]` op de plaats van de puntjes.",
        ],
        why="""```python
def best_match(url, question, texts):
    vectors = embed(url, [question] + texts)
    q = np.array(vectors[0])
    best, best_score = None, -2
    for i in range(len(texts)):
        score = np.dot(q, vectors[i + 1])
        if score > best_score:
            best, best_score = texts[i], score
    return best
```

De vraag staat vooraan, dus de embedding van `texts[i]` staat op plek `i + 1`. De embeddings van Granite hebben lengte 1, dus het inproduct is de cosine similarity; delen door de twee lengtes, zoals in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl), kiest dezelfde tekst.""",
    )
    return


@app.cell(hide_code=True)
def text_no_clear():
    mo.md(r"""
    ### Oefening 3. Geen duidelijke winnaar

    Kijk nog eens naar de derde vraag van de course, in het plaatje met de zes hulppagina's.
    """)
    return


@app.cell(hide_code=True)
def choice_no_clear_cell():
    choice_no_clear = L.choice(
        "no-clear-match",
        question="Bij `Kan ik iemand bellen?` kwam de pagina over terugsturen als beste match uit de bus. Wat vertellen de zes scores je?",
        options={
            "dutch": "Granite begrijpt geen Nederlandse vragen",
            "close": "Geen pagina springt eruit: de scores liggen dicht bij elkaar",
            "returns": "Je stuurt een bestelling telefonisch terug, dus de pagina over terugsturen is het antwoord",
        },
        correct="close",
        why={
            "close": "De zes scores liggen dicht bij elkaar, 0,522 tot 0,600, dus de beste pagina is geen zeker antwoord. Een duidelijke match springt eruit, zoals de pagina over het wachtwoord met 0,686 tegen 0,519 voor de volgende pagina. Hier wint de beste pagina met 0,015 van de volgende. Een zoekfunctie kan dan meerdere pagina's laten zien, of zeggen dat hij niets vond. De pagina over de klantenservice zegt wanneer die bereikbaar is, niet dat je kunt bellen.",
            "dutch": "Kijk naar de tweede vraag, ook in het Nederlands: welke pagina vond die, en in welke taal is die pagina geschreven?",
            "returns": "Lees de pagina over terugsturen nog eens: staat daar iets over bellen?",
        },
    )
    choice_no_clear
    return (choice_no_clear,)


@app.cell(hide_code=True)
def text_upload():
    mo.md(r"""
    ### Oefening 4. Eén pdf, veertig vragen
    """)
    return


@app.cell(hide_code=True)
def choice_upload_cell():
    choice_upload = L.choice(
        "upload",
        question="Je stelt een model 40 vragen over één pdf van 30 pagina's, in 40 requests. Je uploadt de pdf één keer, en noemt hem in elk request bij zijn id. Waarvoor betaal je?",
        options={
            "each": "Voor de tokens van de pdf in elk van de 40 requests",
            "once": "Eén keer voor de tokens van de pdf, bij de upload, en daarna alleen voor de vragen",
            "nothing": "Niets voor de pdf: een geüpload bestand gebruik je gratis",
        },
        correct="each",
        why={
            "each": "De upload bespaart dat je het bestand elke keer verstuurt, niet het lezen: het model leest het document in elk request opnieuw, en elk request betaalt voor de tokens ervan. Een latere les laat zien hoe je minder betaalt als elk request met dezelfde lange tekst begint.",
            "once": "Onthoudt het model de pdf van het ene request tot het volgende, of leest het hem opnieuw om vraag 37 te beantwoorden?",
            "nothing": "Sommige aanbieders bewaren een bestand een tijdje gratis. Maar wat gebeurt er als het model het leest, om een vraag te beantwoorden?",
        },
    )
    choice_upload
    return (choice_upload,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Zoek in de hulppagina's met je eigen best_match

    Zodra oefening 1 en 2 goed zijn, kunnen je functies echt in de zes hulppagina's zoeken: de course start zijn server op je computer, met Granite erachter. Stel een vraag in het Nederlands, in het Engels, of in een andere van de twaalf talen van Granite, en kijk welke pagina je `best_match` kiest.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*De demo heeft Granite Embedding nodig, en dat staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, in een vaste versie: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, onder de licentie Apache-2.0.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte vrij, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Granite Embedding is gedownload.*")
    return


@app.cell(hide_code=True)
def question_box_cell():
    question_box = mo.ui.text(value="Kan ik mijn bestelling nog annuleren?", label="Je vraag", full_width=True)
    search_button = mo.ui.run_button(label="Zoek met je best_match")
    mo.vstack([question_box, search_button])
    return question_box, search_button


@app.cell(hide_code=True)
def demo_answer(box_best, box_embed, question_box, search_button):
    _match = C.learner_best_match(box_best.value, box_embed.value)
    if not C.downloaded():
        _out = mo.md("*Download eerst Granite Embedding, met de knop hierboven.*")
    elif _match is None:
        _out = mo.md("*Maak eerst oefening 1 en 2 af: de demo zoekt met je eigen `embed` en `best_match`.*")
    elif not search_button.value:
        _out = mo.md("*Druk op de knop om te zoeken.*")
    elif not question_box.value.strip():
        _out = mo.md("*Schrijf eerst een vraag in het vak.*")
    else:
        mo.output.replace(mo.md("*Je `best_match` heeft de vraag en de zes pagina's verstuurd; de server laadt Granite en rekent hun embeddings uit …*"))
        _words = {"title": "Wat je best_match vond",
                  "mark": "← je best_match koos deze",
                  "done": "{seconds} seconden, met het laden van Granite de eerste keer.",
                  "failed": "Je best_match stopte met een fout tegen de echte server: {why}",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(C.demo(_match, question_box.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het request is dat van [Een model aanroepen over HTTP](/course/open/build-call-api?lang=nl), naar een ander endpoint; het inproduct en de cosine similarity zijn die uit [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl). De volgende les, [Kant-en-klare AI: zien, spraak, vertalen en klassieke NLP](/course/open/build-ai-services?lang=nl), kijkt naar kant-en-klare diensten voor één taak. Later zoekt de module over RAG, antwoorden uit je eigen documenten, zo in de samenvattingen van de course, met hetzelfde model, nadat langere teksten in stukken zijn geknipt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_no_clear, choice_upload, result_best, result_embed):
    L.wrap_up(result_embed, result_best, choice_no_clear, choice_upload)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
