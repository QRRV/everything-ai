# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-pretraining-data. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-pretraining-data", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) en [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="Een website biedt een grote verzameling teksten aan, gratis te downloaden, zonder licentie. Wat neem je aan?",
        options={
            "free": "Dat je er alles mee mag: het is gratis, en niemand claimt het",
            "name": "Dat je het mag gebruiken, zolang je de website noemt",
            "right": "Dat je het niet mag kopiëren of delen: vraag het de maker",
        },
        correct="right",
        why={
            "free": "Gratis te downloaden is niet gratis te gebruiken. Zonder licentie houdt de maker zijn rechten: zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "name": "De maker noemen is wat CC BY vraagt, en alleen een licentie als CC BY geeft je daar het recht voor terug. Zonder licentie is er zo'n recht niet: zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Deze les vraagt hetzelfde over de biljoenen tokens waarvan een groot language model leert.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def quick_check_stories():
    quick_stories = L.choice(
        "quick-stories",
        question="Je GPT op tokens trainde alleen op korte verhalen voor kinderen. Je laat hem beginnen met \"The capital of France is\". Wat verwacht je dat hij schrijft?",
        options={
            "right": "Iets uit een kinderverhaal: zulke feiten las hij nooit",
            "paris": "Paris: de tokenizer kent het woord, dus het model kan het schrijven",
            "nothing": "Niets: hij schrijft alleen na \"Once upon a time\"",
        },
        correct="right",
        why={
            "paris": "De tokenizer knipt tekst in stukken; wat ze betekenen weet hij niet. Je GPT leerde alleen wat zijn verhalen bevatten: zie [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl).",
            "nothing": "Hij schrijft verder na elk begin dat je hem geeft, met wat hij leerde: de woorden en wendingen van kinderverhalen. Zie [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl).",
            "right": "Een model weet wat het gelezen heeft. Deze les kijkt naar wat grote modellen lezen.",
        },
    )
    quick_stories
    return (quick_stories,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je GPT op tokens schrijft kinderverhalen, omdat hij niets anders gelezen heeft: 5 miljoen tekens TinyStories. Ook een groot language model weet wat het gelezen heeft, alleen las het biljoenen tokens: webpagina's, code, papers, boeken. Wat er in die stapel zit, bepaalt wat het model weet, welke talen het spreekt en of het code kan schrijven. En het bepaalt nog iets: of je het model mag delen. De course moest dat zelf beslissen: je GPT traint op TinyStories, en de GPT op tokens die de course deelt, is getraind op andere verhalen, uit Cosmopedia. Deze les laat zien waarom.

    **Een voorbeeld uit het dagelijks leven.** Iemand die alleen kookboeken gelezen heeft, kan je vertellen hoe je soep maakt, niet hoe een motor werkt. En een kookboek dat zonder te vragen van andermans plank is gekopieerd, mag diegene niet uitdelen.
    """)
    return


@app.cell(hide_code=True)
def predict_web():
    guess_web = L.guess(
        "web",
        question="Dolma, een open corpus van 3 biljoen tokens, bevat webpagina's, code, berichten op sociale media, wetenschappelijke papers, boeken en een encyclopedie. Hoeveel ervan zijn volgens jou webpagina's?",
        options={
            "quarter": "Ongeveer 25%",
            "half": "Ongeveer 50%",
            "most": "Ongeveer 80%",
        },
        reveal=f"**Ongeveer 80%: {C.number(C.DOLMA['web pages (Common Crawl)'], 0, L.lang)} van zijn {C.number(C.DOLMA_TOTAL, 0, L.lang)} miljard tokens zijn webpagina's, {C.number(C.share(C.DOLMA['web pages (Common Crawl)']), 0, L.lang)}%.** Code is {C.number(C.share(C.DOLMA['code (GitHub)']), 0, L.lang)}%, en de boeken en de encyclopedie samen zijn minder dan een half procent. Geen andere bron heeft bij lange na zoveel tekst als het web.",
    )
    guess_web
    return (guess_web,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Veel bronnen, één stapel.** De tekst waarop een model pretraint, komt uit veel bronnen, samengevoegd tot één corpus:

        - **Webpagina's**, verreweg het meest. Common Crawl, een non-profit, verzamelt sinds 2008 miljarden webpagina's en publiceert ze gratis; corpora als C4, FineWeb en Dolma schonen ze op voor het trainen.
        - **Code**, uit openbare repositories zoals die op GitHub.
        - **Wetenschappelijke papers**, zoals die op arXiv; **boeken**, zoals de boeken van Project Gutenberg, in het publieke domein; **encyclopedieën**, zoals Wikipedia; en **forums**, zoals Reddit en Stack Exchange.
        - **Tekst geschreven door modellen**, zoals de verhalen van TinyStories en Cosmopedia: synthetische data.

        **Wat het model weet, volgt uit de stapel.** Een model kan alleen leren wat erin zit. Een corpus zonder code geeft een model dat geen code kan schrijven; een corpus in het Engels een model dat weinig Nederlands schrijft; een corpus dat in 2024 ophoudt een model met een knowledge cutoff in 2024: het weet niets van wat daarna kwam.

        **Wat je ermee mag, volgt uit de bronnen.** De course stelt vier vragen aan elke tekst die in een model gaat dat ze deelt, en houdt de antwoorden bij in `data/README.md` en `course/route/bronnen.yaml`:

        1. **Is de tekst zelf open?** Tekst in het publieke domein, of onder een licentie als CC BY, CC BY-SA, Apache-2.0 of MIT, is dat. Met SA geeft de course een model dat ze deelt, voor de zekerheid dezelfde licentie: of SA ook geldt voor een model dat op de tekst getraind is, staat niet vast.
        2. **Wie schreef hem?** Tekst die een model schreef, kan de voorwaarden van dat model meedragen. De voorwaarden van OpenAI verbieden zijn output te gebruiken voor modellen die met OpenAI concurreren. De course wil niet discussiëren over wat concurreert, dus deelt ze geen model dat op zulke tekst getraind is; de licentie van het model Mixtral, Apache-2.0, stelt zo'n voorwaarde niet.
        3. **Is hij rechtmatig gekopieerd?** Nooit uit een piraterijbron, ook niet om zelf op te trainen: de uitzondering van vraag 4 geldt alleen voor tekst die je rechtmatig mag lezen. Books3, bijna 200.000 boeken van een piraterijsite, zat in een bekend open corpus, de Pile, tot een verzoek van de Deense Rights Alliance het offline haalde.
        4. **Heeft de maker een opt-out gegeven?** In de EU is tekst- en datamining van tekst die je rechtmatig mag lezen toegestaan, ook voor het trainen van AI, tenzij de rechthebbende het heeft voorbehouden; voor tekst online op een machineleesbare manier, zoals met regels in robots.txt, het bestand waarin een website crawlers vertelt wat ze mogen ophalen (de Europese auteursrechtrichtlijn van 2019, artikel 4).

        **Het model heeft ook een licentie.** Van veel modellen zijn de weights openbaar: open weights. Maar open weights alleen zijn nog geen open source. De weights van Llama, het language model van Meta, kan iedereen downloaden, maar zijn licentie beperkt wie ze mag gebruiken en hoe. Volgens de definitie van het Open Source Initiative (2024) komt open-source AI met zijn weights en zijn code, en met genoeg informatie over zijn training data om een vergelijkbaar model te bouwen, allemaal onder voorwaarden die iedereen laten gebruiken, bestuderen, aanpassen en delen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **pretraining corpus** | alle tekst waarvan een model leert tijdens het pretrainen, samengevoegd uit veel bronnen, vaak biljoenen tokens | `{"webpagina's": 2479, "code": 411}` |
        | **Common Crawl** | een non-profitproject dat sinds 2008 miljarden webpagina's verzamelt en gratis publiceert: de belangrijkste bron van webtekst voor pretraining | |
        | **synthetische data** | data die een programma maakt, in plaats van data die in de wereld verzameld is; hier tekst die een language model schreef | |
        | **tekst- en datamining** | grote hoeveelheden tekst en data door een computer laten analyseren om patronen te vinden, zoals het trainen van een model doet | |
        | **opt-out** | een manier waarop de eigenaar van content aangeeft dat die niet gebruikt mag worden, bijvoorbeeld voor het trainen van AI: met regels in robots.txt, metadata of een verzoek | |
        | **open weights** | de getrainde parameters van een model, gepubliceerd zodat iedereen ze kan downloaden en draaien; zijn data en code kunnen geheim blijven, en zijn licentie kan nog voorwaarden stellen | |
        | **open-source AI** | een model dat iedereen mag gebruiken, bestuderen, aanpassen en delen: met zijn weights, zijn code en genoeg informatie over zijn data om een vergelijkbaar model te bouwen | |
        """),
        L.watch_out("De licentie van een corpus kan de verzameling dekken en niet de teksten erin. De licentie van FineWeb, ODC-By 1.0, laat je de verzameling gebruiken en delen, als je de makers noemt; de webpagina's erin houden hun eigen rechten, dus de course traint er geen model op dat ze deelt."),
        mo.md(r"""
        Het beeld: de zes bronnen van Dolma, in miljarden tokens van de tokenizer van Llama, met het aandeel van elk.
        """),
        mo.Html(C.sources_picture({"title": "Waar Dolma uit bestaat", "note": "miljarden tokens · aandeel van alle 3.059",
                                   "alt": "Balken voor de zes bronnen van Dolma: webpagina's uit Common Crawl 2.479 miljard tokens, 81%; code van GitHub 411 miljard, 13,4%; sociale media van Reddit 89 miljard, 2,9%; wetenschappelijke papers van Semantic Scholar 70 miljard, 2,3%; boeken van Project Gutenberg 6 miljard, 0,2%; de encyclopedie, Wikipedia en Wikibooks, 4,3 miljard, 0,1%."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Voordat je een model op een corpus pretraint, en het deelt:

    1. Zet elke bron op een rij, met hoeveel tokens hij toevoegt: dat zegt wat het model zal weten.
    2. Zoek voor elke bron de licentie op, en kijk wat die dekt: de verzameling, of de teksten zelf.
    3. Zoek uit wie de teksten schreef: mensen, of een model waarvan de voorwaarden meekomen.
    4. Laat piraterijbronnen weg, en teksten waarvan de makers een opt-out gaven.
    5. Schrijf de bronnen, hun aandelen en hun licenties op naast het model, en geef het model een licentie die daarbij past.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 voor Dolma: zijn bronnen, in miljarden tokens, als dictionary, en het aandeel van elk, uit de tabel in zijn paper.
    """)
    return


@app.cell
def example_shares():
    # 1. elke bron van Dolma, met zijn tokens in miljarden
    _dolma = {"webpagina's": 2479, "code": 411, "sociale media": 89, "wetenschappelijke papers": 70, "boeken": 6.0, "encyclopedie": 4.3}
    _total = 0
    for _source in _dolma:
        _total = _total + _dolma[_source]
    print("alle tokens:", round(_total, 1), "miljard")
    # het aandeel van elke bron, in procenten
    for _source in _dolma:
        print(_source, round(100 * _dolma[_source] / _total, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De zes bronnen tellen op tot 3.059,3 miljard tokens; het paper rondt dat af op 3.059. Webpagina's zijn 81,0% en code 13,4%: een model dat op Dolma traint, leert vooral van het web, en een flink deel code. Boeken zijn er 0,2% van. Stap 2 tot en met 4 vragen meer dan getallen: de webpagina's van Dolma komen uit Common Crawl, en de licentie van Dolma, ODC-By net als die van FineWeb, dekt de verzameling, niet de pagina's.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Waar mag het heen?

        Zes datasets met tekst, zoals de course ze tegenkomt. Zet elke dataset waar de vier vragen van het idee hem zetten: het mag in de modellen die de course deelt; het is alleen om zelf op te trainen, en de course deelt geen model dat erop getraind is; of helemaal niet.
        """),
        L.task("Geef elke dataset zijn plek."),
    ])
    return


@app.cell(hide_code=True)
def exercise_where():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "Alleen een tekst die door alle vier de vragen komt, mag in de modellen die de course deelt. Een tekst die niet open is, of waarvan de voorwaarden van de maker een gedeeld model uitsluiten, kan nog prima zijn om op je eigen computer van te leren; een tekst die niet rechtmatig gekopieerd is, niet.",
            "Twee datasets zijn door modellen geschreven, en één komt van een piraterijsite. Voor de andere: dekt de licentie de teksten, of alleen de verzameling?",
        ],
        why="Tiny Shakespeare is publiek domein, Wikipedia is CC BY-SA, dus de course zou een model dat erop getraind is voor de zekerheid ook CC BY-SA geven, en Cosmopedia is geschreven door een model onder Apache-2.0: ze mogen in de modellen die de course deelt. TinyStories draagt de voorwaarden van OpenAI, en de licentie van FineWeb dekt alleen de verzameling: de course laat je er zelf op trainen, maar deelt geen model dat erop getraind is. Books3 komt van een piraterijsite: helemaal niet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_license():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel staat onder welke licentie?

        Stap 2 van het recept, in getallen. Maak `by_license(sources)` af: elke bron is een dict met een `name`, zijn `tokens` en zijn `license`, en de functie geeft een dict terug met elke licentie en het aandeel van alle tokens dat zijn bronnen hebben, in procenten, afgerond op 1 decimaal. De check roept `by_license` aan met een voorbeeldcorpus van vijf bronnen, in miljoenen tokens: Nederlandse Wikipedia, 900 onder CC BY-SA 4.0; Project Gutenberg, 300 in het publieke domein; code op GitHub, 250 onder MIT en 60 onder Apache-2.0; en Nederlandse webpagina's, 3.500 onder ODC-By 1.0. ODC-By 1.0 moet dan op 69,9 uitkomen. Daarna probeert hij andere.
        """),
        L.task("Laat `by_license` de tokens van elke licentie optellen, en geef elk als aandeel van alles."),
    ])
    return


@app.cell(hide_code=True)
def exercise_license():
    box_license = L.code_box("by-license", lines=14)
    box_license
    return (box_license,)


@app.cell(hide_code=True)
def check_license(box_license):
    result_license = L.check("by-license", box_license.value)
    result_license
    return (result_license,)


@app.cell(hide_code=True)
def hints_license():
    L.hints(
        "by-license",
        [
            "Eerst het totaal, met een loop zoals in het uitgewerkte voorbeeld; dan een tweede loop over `sources` die de tokens per licentie optelt, met `s[\"license\"]` als key; dan een loop over de licenties voor hun aandelen.",
            "Een licentie die je voor het eerst tegenkomt, begint bij de tokens van zijn bron; een licentie die je al eerder zag, krijgt ze erbij. Rond pas aan het eind af, één keer per licentie.",
        ],
        why="Eerst per licentie optellen, en één keer afronden, geeft elke licentie precies zijn deel van het corpus. Elke bron afronden voordat je ze optelt, zou de afrondingen ook optellen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_open():
    mo.md(r"""
    ### Oefening 3. Open weights of open source?
    """)
    return


@app.cell(hide_code=True)
def choice_open_cell():
    choice_open = L.choice(
        "open-weights",
        question="Iedereen kan de weights van Llama 3 downloaden. Zijn licentie verbiedt sommige toepassingen, en bedrijven met meer dan 700 miljoen gebruikers per maand moeten eerst Meta om toestemming vragen. Zijn training data is niet gepubliceerd. Is Llama 3 open-source AI, volgens de definitie van het Open Source Initiative?",
        options={
            "yes": "Ja: iedereen kan zijn weights downloaden, draaien, aanpassen en er zelf op voortbouwen",
            "right": "Nee: open weights, maar zijn voorwaarden beperken het en zijn data is onbekend",
            "free": "Ja, voor iedereen met minder dan 700 miljoen gebruikers per maand",
        },
        correct="right",
        why={
            "yes": "Dat maakt zijn weights open. Open-source AI vraagt meer: voorwaarden die iedereen laten gebruiken, bestuderen, aanpassen en delen, zijn code, en genoeg over zijn data om een vergelijkbaar model te bouwen.",
            "free": "Voorwaarden die afhangen van wie je bent, zijn geen open-source-voorwaarden: die laten iedereen het gebruiken, voor elk doel.",
            "right": "Open weights, geen open source. Een model dat je op Llama bouwt, erft zijn licentie; een model dat je zelf traint op open data, met je eigen code, kan open source zijn.",
        },
    )
    choice_open
    return (choice_open,)


@app.cell(hide_code=True)
def text_exercise_optout():
    mo.md(r"""
    ### Oefening 4. Een opt-out
    """)
    return


@app.cell(hide_code=True)
def choice_optout_cell():
    choice_optout = L.choice(
        "opt-out",
        question="Een Nederlandse nieuwssite behoudt tekst- en datamining van zijn artikelen voor, op een machineleesbare manier in zijn robots.txt. Een bedrijf in de EU crawlt de site toch, om een model te trainen dat het verkoopt. Staat de Europese auteursrechtrichtlijn dat toe?",
        options={
            "right": "Nee: de site gaf een opt-out voor tekst- en datamining",
            "public": "Ja: de artikelen staan online, gratis voor iedereen te lezen",
            "learn": "Ja: een model leert van de tekst; het kopieert hem niet",
        },
        correct="right",
        why={
            "public": "Gratis te lezen is niet vrij voor tekst- en datamining. Artikel 4 staat tekst- en datamining toe van tekst die je rechtmatig mag lezen, tenzij de rechthebbende het heeft voorbehouden, en deze site deed dat.",
            "learn": "Trainen begint met de pagina's naar een computer kopiëren, en die kopie heeft toestemming of een uitzondering nodig. De uitzondering van artikel 4 vervalt als de rechthebbende een opt-out geeft.",
            "right": "Met een machineleesbaar voorbehoud dekt artikel 4 de tekst- en datamining van het bedrijf niet meer. Of een voorbehoud telt, in gewone woorden op een pagina of alleen voor sommige crawlers, beslissen rechters, en daar zijn ze nog mee bezig.",
        },
    )
    choice_optout
    return (choice_optout,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De datasheet uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl), nu voor elke bron van een corpus van biljoenen tokens; en de verhalen uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl), waarvan de bron bepaalde welke modellen de course mag delen. De volgende lessen schonen zo'n corpus op: eerst slechte tekst eruit, dan dezelfde tekst die twee keer voorkomt, dan de vragen van tests.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_where, result_license, choice_open, choice_optout):
    L.wrap_up(result_where, result_license, choice_open, choice_optout)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
