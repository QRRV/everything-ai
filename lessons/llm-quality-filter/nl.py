# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-quality-filter. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow regression

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-quality-filter", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar de data voor pretraining vandaan komt](/course/open/llm-pretraining-data?lang=nl) en [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_web():
    quick_web = L.choice(
        "quick-web",
        question="De licentie van FineWeb, ODC-By 1.0, laat iedereen de verzameling gebruiken en delen, als ze de makers noemen. Wat zegt die over de webpagina's erin?",
        options={
            "right": "Niets, want elke pagina houdt zijn eigen rechten",
            "free": "Dat iedereen elke pagina erin mag kopiëren en delen",
            "public": "Dat de pagina's in het publieke domein zijn",
        },
        correct="right",
        why={
            "free": "De licentie dekt de verzameling als geheel, niet de teksten erin. Zie [Waar de data voor pretraining vandaan komt](/course/open/llm-pretraining-data?lang=nl).",
            "public": "Dat iedereen een pagina op het web kan lezen, zet hem niet in het publieke domein: hij houdt de rechten van zijn maker. Zie [Waar de data voor pretraining vandaan komt](/course/open/llm-pretraining-data?lang=nl).",
            "right": "Daarom host de course geen van de ruwe webpagina's van deze les: je downloadt ze zelf.",
        },
    )
    quick_web
    return (quick_web,)


@app.cell(hide_code=True)
def quick_check_probability():
    quick_probability = L.choice(
        "quick-probability",
        question="Een logistische regressie geeft een pagina het getal 0,8. Wat zegt dat getal?",
        options={
            "share": "Dat 80% van de pagina goed is",
            "class": "Dat de pagina zeker in klasse 1 zit",
            "right": "De kans, volgens het model, dat de pagina in klasse 1 zit",
        },
        correct="right",
        why={
            "share": "Het zegt niets over delen van de pagina: het is één kans voor de hele pagina. Zie [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "class": "De sigmoid haalt nooit 1: 0,8 laat een kans van 0,2 over voor klasse 0. Zie [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "right": "De sigmoid van `w · x + b`. In deze les is klasse 1 tekst die leest als een goed voorbeeld.",
        },
    )
    quick_probability
    return (quick_probability,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een webpagina zoals Common Crawl hem bewaart, is niet alleen zijn tekst. Het is ook het menu, de knoppen, de voettekst, de melding dat je JavaScript moet aanzetten, een lijst met tags, en soms een pagina met alleen maar trefwoorden om zoekmachines te misleiden. Een model leert van dat alles, token voor token, dus elk token van een menu is een token dat het niet aan een zin besteedt. Voor het pretrainen gooit elk open corpus een groot deel van een crawl weg, met regels die iedereen kan lezen en, steeds vaker, met een kleine classifier.

    **Een voorbeeld uit het dagelijks leven.** Je post sorteren: de folders gaan ongelezen bij het oud papier, en de brieven maak je open. Een paar regels sorteren het meeste; voor de rest kijk je naar een envelop en oordeel je.
    """)
    return


@app.cell(hide_code=True)
def predict_lines():
    _m = C.MEASURED
    guess_lines = L.guess(
        "lines",
        question=f"De eerste 20 MB van een shard van C4 van vóór het opschonen bevatten {C.number(_m['pages'], 0, L.lang)} ruwe webpagina's, {C.number(_m['chars'] / 1e6, 1, L.lang)} miljoen tekens. C4 houdt alleen de regels die een zin afsluiten en minstens 5 woorden hebben. Hoeveel van de tekst blijft er volgens jou over?",
        options={
            "most": "Ongeveer 90%",
            "two": "Ongeveer twee derde",
            "right": "Ongeveer een derde",
        },
        reveal=f"**Ongeveer een derde: {C.number(_m['chars_lines'] / 1e6, 1, L.lang)} miljoen tekens, {C.number(100 * _m['chars_lines'] / _m['chars'], 0, L.lang)}%.** De andere twee derde zijn regels zonder het einde van een zin, of met minder dan 5 woorden: menu's, knoppen, titels, lijsten, datums en links.",
    )
    guess_lines
    return (guess_lines,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eisen per regel.** C4, een corpus dat Google in 2019 uit Common Crawl maakte om T5 te trainen, houdt een regel alleen als hij een zin afsluit, met `.`, `!`, `?` of `"`, minstens 5 woorden heeft, en JavaScript niet noemt. De makers gooiden ook elke pagina weg met minder dan 3 zinnen, met "lorem ipsum" of een accolade `{` erin, en elke pagina met een woord uit een lijst van slechte woorden.

        **Eisen per pagina.** Gopher, een model van DeepMind uit 2021, gooide onder meer een pagina weg met minder dan 50 of meer dan 100.000 woorden, met een gemiddelde woordlengte onder 3 of boven 10, met minder dan twee van de stop words "the", "be", "to", "of", "and", "that", "have" en "with", of met minder dan 80% woorden die een letter bevatten. Zulke regels zijn goedkoop en iedereen kan ze lezen, maar ze zijn grof: een goede tabel met getallen zakt erdoor, en een pagina spam in nette zinnen komt erdoor.

        **Een classifier.** Daarom trainen corpora ook een classifier: een paar getallen per pagina, zijn features, gaan een model in dat de kans geeft dat de pagina leest als goede tekst. Om dat te leren, heeft het voorbeelden van goede tekst nodig. De classifier van GPT-3 was een logistische regressie op tellingen van woorden, met WebText, boeken en Wikipedia als goede voorbeelden en ruwe webpagina's als de andere. FineWeb-Edu liet een groot language model zo'n 450.000 pagina's beoordelen op hoeveel je ervan leert, trainde een classifier op die cijfers, en hield de pagina's die hij minstens 3 van 5 gaf: een threshold. De classifier leert hoe zijn goede voorbeelden eruitzien, en verder niets: dat is zijn kracht en zijn zwakte.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **quality filter** | een regel of een kleine classifier die tekst van lage kwaliteit, zoals spam, boilerplate of kapotte pagina's, weghaalt voor het trainen | `keep_page(text)` |
        | **stop word** | een heel gewoon woord dat op zichzelf weinig betekent, zoals "the", "of" of "and"; gewone zinnen staan er vol mee, lijsten en spam niet | `{"the", "of", "and"}` |
        | **eindigt op** | of een tekst eindigt op een stuk tekst, of op een van de stukken in een tuple | `line.endswith((".", "!"))` |
        | **is een letter** | of een teken een letter is, van welk alfabet ook; `isspace` vraagt hetzelfde over spaties, tabs en nieuwe regels | `c.isalpha()` |
        """),
        L.watch_out("Een filter haalt tekst weg, dus het beslist wat een model nooit ziet. De lijst van slechte woorden van C4 gooide ook veel pagina's van en over sommige groepen mensen vaker weg dan andere, zoals pagina's in Afro-Amerikaans Engels en pagina's die homo's of lesbiennes noemen (Dodge en anderen, 2021). Kijk naar wat een filter weggooit, niet alleen naar wat hij houdt."),
        mo.md(r"""
        Het beeld: de tekens van de 8.690 ruwe pagina's van de download van deze les, na elke stap, met de versies van de course van de oefeningen.
        """),
        mo.Html(C.steps_picture([("ruwe webpagina's", round(_m["chars"] / 1e6, 1)), ("na de regelfilters van C4", round(_m["chars_lines"] / 1e6, 1)),
                    ("na de paginafilters van Gopher", round(_m["chars_kept"] / 1e6, 1))],
                                {"title": "Miljoenen tekens, stap voor stap", "lang": L.lang,
                                 "note": f"{C.number(_m['kept'], 0, L.lang)} van de {C.number(_m['pages'], 0, L.lang)} pagina's blijven over",
                                 "alt": f"Balken met miljoenen tekens: de ruwe webpagina's {C.number(_m['chars'] / 1e6, 1, L.lang)}; na de regelfilters van C4 {C.number(_m['chars_lines'] / 1e6, 1, L.lang)}; na de paginafilters van Gopher {C.number(_m['chars_kept'] / 1e6, 1, L.lang)}. {C.number(_m['kept'], 0, L.lang)} van de {C.number(_m['pages'], 0, L.lang)} pagina's blijven over."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om ruwe webpagina's te filteren voor pretraining:

    1. Schoon elke pagina regel voor regel op: houd de regels die als zinnen lezen.
    2. Gooi de pagina's weg die een paginafilter niet doorstaan: te kort, te lang, vreemde woorden, geen stop words, te weinig letters.
    3. Beschrijf elke pagina die overblijft met een paar getallen, zijn features.
    4. Train een classifier op de features van goede voorbeelden en van gewone pagina's, en geef elke pagina een score.
    5. Houd de pagina's boven een threshold, en lees een steekproef van wat je weggooide.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 voor drie regels, één eis tegelijk: sluit de regel een zin af, en hoeveel woorden heeft hij?
    """)
    return


@app.cell
def example_lines():
    _end = (".", "!", "?", '"')
    _lines = ["  Read more  ", "Our shop is open every day from nine to five.", "Free shipping!"]
    for _line in _lines:
        # 1. de regel zonder spaties aan de kanten, dan de twee eisen
        _line = _line.strip()
        print(f'"{_line}"', "sluit een zin af:", _line.endswith(_end), "woorden:", len(_line.split()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    "Read more" is een knop: geen einde van een zin, 2 woorden. "Free shipping!" eindigt op `!`, maar met 2 woorden is het een slogan, geen zin. Alleen de openingstijden voldoen aan beide eisen. `strip()` doet ertoe: zonder zouden de spaties na "Read more" bij de regel horen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lines():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De regelfilters van C4

        Maak `clean_lines(page)` af: stap 1 van het recept. Hij geeft de regels van `page` terug die eindigen op een van `END`, minstens 5 woorden hebben en javascript niet noemen, met of zonder hoofdletters, elk zonder spaties aan de kanten, aan elkaar gezet met nieuwe regels. De check roept hem aan met een pagina van zes regels, van "Home" tot "We walked along the beach until the sun went down.", en met andere.
        """),
        L.task("Laat `clean_lines` alleen de regels houden die aan alle drie de eisen voldoen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lines():
    box_lines = L.code_box("clean-lines", lines=10)
    box_lines
    return (box_lines,)


@app.cell(hide_code=True)
def check_lines(box_lines):
    result_lines = L.check("clean-lines", box_lines.value)
    result_lines
    return (result_lines,)


@app.cell(hide_code=True)
def hints_lines():
    L.hints(
        "clean-lines",
        [
            "Het uitgewerkte voorbeeld laat de eerste twee eisen zien voor één regel. Voeg de derde toe, en houd de regel alleen als alle drie gelden.",
            "`line.lower()` geeft de regel in kleine letters, dus één test met `in` vindt JavaScript in elke spelling.",
        ],
        why="Een regel die een zin afsluit, met minstens 5 woorden, hoort bijna altijd bij de tekst die iemand schreef; de menu's, knoppen en meldingen van een pagina niet. De bewaarde regels aan elkaar zetten met nieuwe regels houdt de pagina een pagina.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_page():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De paginafilters van Gopher

        Maak `keep_page(text)` af: stap 2 van het recept, voor een pagina die `clean_lines` al opschoonde. Hij geeft `False` als de pagina een van de vier eisen van Gopher uit het idee niet haalt, en anders `True`. De check roept hem aan met een pagina van 60 woorden, steeds weer "the cat sat on a warm mat and looked at the birds with care", en met andere, zoals een pagina van 49 woorden.
        """),
        L.task("Laat `keep_page` een pagina weggooien die een van de vier eisen niet haalt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_page():
    box_page = L.code_box("keep-page", lines=18)
    box_page
    return (box_page,)


@app.cell(hide_code=True)
def check_page(box_page):
    result_page = L.check("keep-page", box_page.value)
    result_page
    return (result_page,)


@app.cell(hide_code=True)
def hints_page():
    L.hints(
        "keep-page",
        [
            "Elke eis is een `if` die eindigt met `return False`; een pagina die alle vier voorbijkomt, eindigt bij `return True`. De gemiddelde woordlengte is het totaal van `len(w)` gedeeld door het aantal woorden.",
            "Verzamel voor eis 3 de stop words van de pagina in een set: dan telt drie keer \"the\" één keer. Tel voor eis 4 de woorden met minstens één teken waarvoor `c.isalpha()` waar is.",
        ],
        why="Elke eis vangt een andere soort pagina die geen gewone tekst is: te weinig ervan, woorden die geen woorden zijn, geen zinnen, of vooral getallen en tekens. De volgorde maakt voor de uitkomst niet uit; goedkope eisen eerst besparen alleen tijd.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Vier getallen per pagina

        Maak `features(page)` af: stap 3 van het recept. Hij geeft vier getallen tussen 0 en 1 terug: het aandeel regels dat een zin afsluit, het aandeel woorden dat in kleine letters een van `WORDS` is, het aandeel tekens dat een letter of een spatie is, en het aantal woorden per regel gedeeld door 20, hoogstens 1. Regels die na `strip()` leeg zijn, tellen niet mee. De check roept hem aan met een pagina van vijf regels, van "Home" tot een e-mailadres, en met andere.
        """),
        L.task("Laat `features` de vier getallen van een pagina geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    box_features = L.code_box("features", lines=26)
    box_features
    return (box_features,)


@app.cell(hide_code=True)
def check_features(box_features):
    result_features = L.check("features", box_features.value)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def hints_features():
    L.hints(
        "features",
        [
            "Elk getal is een telling gedeeld door een totaal: regels, woorden of tekens. Tel met een loop en een `if`, zoals in oefening 2.",
            "Feature 4 is `len(words) / len(lines) / 20`, en `min(..., 1.0)` houdt hem op hoogstens 1.",
        ],
        why="Aandelen in plaats van tellingen maken een korte en een lange pagina vergelijkbaar, en vier getallen tussen 0 en 1 passen bij een logistische regressie zonder te schalen.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_lines, result_page, result_features):
    result_lines, result_page, result_features  # the note changes when an exercise is done
    mo.vstack([
        mo.md(f"""
        ### Wat je functies met ruwe webpagina's doen

        De knop hieronder downloadt de eerste {round(C.hub.PINNED[C.C4_SAMPLE]['size'] / 1e6)} MB van een shard van C4 van vóór het opschonen, één keer, van Hugging Face: {C.number(C.MEASURED['pages'], 0, L.lang)} ruwe webpagina's. De course host er geen: hun teksten houden hun eigen rechten, dus je downloadt ze voor jezelf, en de course deelt geen model dat erop getraind is. Daarna draaien je drie functies op elke pagina, en traint je `fit` uit [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl) de classifier: {C.number(C.N_TRAIN, 0, L.lang)} korte verhalen uit TinyStories als goede voorbeelden, klasse 1, en {C.number(C.N_TRAIN, 0, L.lang)} van de pagina's die door je filters komen als de andere, klasse 0. Dat duurt een paar seconden.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def download_button_cell():
    download_button = mo.ui.run_button(label="Download de webpagina's")
    download_button
    return (download_button,)


@app.cell(hide_code=True)
def demo_pipeline(download_button, box_lines, box_page, box_features, result_lines, result_page, result_features):
    _ready = result_lines.state == "correct" and result_page.state == "correct" and result_features.state == "correct"
    mo.stop(not _ready, mo.md("*Maak eerst oefening 1 tot en met 3 af: de demo draait je drie functies.*"))
    C.sample_or_stop(download_button, {
        "ask": "*Klik op de knop om de webpagina's te downloaden, 20 MB.*",
        "busy": "**Bezig met downloaden** · {done} van {total} MB",
        "no_net": "*Er is geen verbinding met Hugging Face. Probeer het opnieuw als je online bent.*",
        "not_pinned": "*Het bestand van Hugging Face is niet het bestand dat de course vastlegde, dus het is weggegooid. Meld het op GitHub, met de link onderaan de les.*",
        "no_room": "*Er is geen ruimte meer op de schijf voor de download. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download werkte niet: {why}.*",
    })
    _fns = {}
    for _box, _name in ((box_lines, "clean_lines"), (box_page, "keep_page"), (box_features, "features")):
        _fns[_name] = run_code(_box.value).names[_name]
    _r = C.pipeline(_fns["clean_lines"], _fns["keep_page"], _fns["features"])
    if "error" in _r:
        _out = mo.md("*Je `fit` in `work/projects/nn/logistic.py` stopt met een fout. Check zijn oefening opnieuw, of verwijder het bestand om de versie van de course te gebruiken.*")
    else:
        _above = sum(1 for _s in _r["scores"] if _s > 0.5)
        _rows = []
        for _page, _what in ((3715, "een pagina met trefwoorden voor T-shirts"), (22, "een pagina die films te downloaden aanbiedt"),
                             (368, "een helppagina over zoekresultaten exporteren"), (7149, "een vraag met antwoord op een forum")):
            _s = C.score_of(_r, _page)
            _rows.append(f"| {_what} | {'weggegooid door je filters' if _s is None else ('onder 0,001' if _s < 0.001 else C.number(_s, 3, L.lang))} |")
        _who = "" if not _r["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + "."
        _out = mo.vstack([
            mo.md(f"""Van {C.number(_r['pages'], 0, L.lang)} pagina's, {C.number(_r['chars'] / 1e6, 1, L.lang)} miljoen tekens, houden je regelfilters {C.number(_r['chars_lines'] / 1e6, 1, L.lang)} miljoen over, {C.number(100 * _r['chars_lines'] / _r['chars'], 0, L.lang)}%. Je paginafilters houden {C.number(_r['kept'], 0, L.lang)} pagina's, met {C.number(_r['chars_kept'] / 1e6, 1, L.lang)} miljoen tekens. Je classifier geeft {C.number(_above, 0, L.lang)} van die pagina's een score boven 0,5. Je eerste getal is hier op bijna elke pagina 1: je regelfilters hielden alleen regels die een zin afsluiten. Een feature die een filter al vastlegde, vertelt de classifier niets, dus hij leunt op de andere drie.{_who}"""),
            mo.Html(C.histogram_picture(_r["scores"], {"title": "De scores van de pagina's die overblijven", "x": "score van je classifier", "lang": L.lang,
                                                       "alt": f"Een histogram van de scores van de {C.number(_r['kept'], 0, L.lang)} pagina's, van 0 tot 1, met een lijn bij 0,5; {C.number(_above, 0, L.lang)} pagina's scoren daarboven."})),
            mo.md("Vier van de pagina's, en de score die je classifier elk geeft:\n\n| pagina | score |\n|---|---|\n" + "\n".join(_rows) +
                  "\n\nDe trefwoorden en de filmpagina komen door de filters, want hun regels zijn lang genoeg en eindigen op een leesteken, maar de classifier ziet dat ze niet als zinnen lezen. De helppagina en het antwoord op het forum wel."),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.md(r"""
    ### Oefening 4. Wat leert de classifier?
    """)
    return


@app.cell(hide_code=True)
def choice_bias_cell():
    choice_bias = L.choice(
        "bias",
        question="De classifier van de demo leerde van korte verhalen voor kinderen hoe goede tekst eruitziet. Welke goede pagina geeft hij waarschijnlijk een lage score?",
        options={
            "spam": "Een blogpost over een dag op het strand: lange, eenvoudige zinnen",
            "right": "Een helder recept: korte regels met getallen",
            "none": "Geen: een classifier geeft elke goede pagina een goede score",
        },
        correct="right",
        why={
            "spam": "Die leest vrij veel als zijn voorbeelden: lange zinnen, vol stop words en letters. Hij krijgt een hoge score.",
            "none": "Een classifier weet alleen hoe zijn goede voorbeelden eruitzien. Een goede pagina in een andere stijl lijkt voor hem op de pagina's die hij leerde afwijzen.",
            "right": "Korte regels, weinig stop words, veel getallen: anders dan een verhaal, dus een lage score, hoe goed het recept ook is. Elke classifier verkiest wat op zijn goede voorbeelden lijkt: die van FineWeb-Edu verkiest pagina's die lezen als schoolteksten.",
        },
    )
    choice_bias
    return (choice_bias,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De webpagina's uit [Waar de data voor pretraining vandaan komt](/course/open/llm-pretraining-data?lang=nl), nu opgeschoond; de logistische regressie uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), getraind met je `fit` uit [Logistische regressie trainen](/course/open/ml-train-logistic?lang=nl), nu als quality filter. De volgende les vindt de pagina's die bijna hetzelfde zijn als een andere, wat geen filter hier vangt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_lines, result_page, result_features, choice_bias):
    L.wrap_up(result_lines, result_page, result_features, choice_bias)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
