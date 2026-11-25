# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-summarize. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-summarize", "nl")
    C = L.checks   # de zes abstracts van de course, de samenvattingen van Qwen3-0.6B, de idf, SciTLDR, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl), en één over [Van data naar tekst](/course/open/nlp-what-is-nlg?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_idf():
    quick_idf = L.choice(
        "quick-idf",
        question="In *TF-IDF: woorden wegen* is de idf van een woord log10(n / df): n documenten, waarvan er df het woord bevatten. Neem de zes abstracts van deze les als de documenten. Wat is de idf van *the*, dat in alle zes staat?",
        options={
            "one": "1, omdat het woord in elk document staat",
            "right": "0",
            "six": "log10(6), ongeveer 0,78",
            "none": "Niets: idf bestaat alleen voor een woord dat in sommige documenten ontbreekt",
        },
        correct="right",
        why={
            "right": "log10(6 / 6) is log10(1), en dat is 0. Een woord in elk document helpt niet om de documenten uit elkaar te houden, dus het weegt niets: deze les gebruikt dat, zodat *the* en *we* niets aan een score toevoegen.",
            "one": "1 is n / df, zonder de log10. Wat is log10 van 1? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "six": "log10(6) is de idf van een woord in één van de zes documenten. Wat is df voor een woord in alle zes? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "none": "De formule werkt voor elk woord dat in minstens één document staat. Wat geeft log10(6 / 6)? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
        },
    )
    quick_idf
    return (quick_idf,)


@app.cell(hide_code=True)
def quick_check_selection():
    quick_selection = L.choice(
        "quick-selection",
        question="In *Van data naar tekst* bepaalt content selection welke feiten uit de data een tekst noemt. Een samenvatting is een kortere tekst, gemaakt van een langere. Welk deel van het maken van een samenvatting is content selection?",
        options={
            "words": "Een woord kiezen bij elk feit, zoals *winderig* bij windkracht 6",
            "right": "Kiezen welke zinnen van de tekst de samenvatting houdt",
            "commas": "Een komma tussen namen zetten, en *en* tussen de laatste twee",
            "file": "Eerst de tekst uit een bestand lezen, zodat het programma alle woorden heeft",
        },
        correct="right",
        why={
            "right": "Content selection bepaalt wat je zegt, en een samenvatting van gekozen zinnen uit de tekst maakt precies deze keuze: welke zinnen blijven. Deze les bouwt er een.",
            "words": "Woorden kiezen is microplanning, de middelste stap. Welke stap bepaalt wat er überhaupt gezegd wordt? Zie [Van data naar tekst](/course/open/nlp-what-is-nlg?lang=nl).",
            "commas": "Komma's en *en* horen bij surface realization, de laatste stap. Welke stap bepaalt wat er gezegd wordt? Zie [Van data naar tekst](/course/open/nlp-what-is-nlg?lang=nl).",
            "file": "Een bestand lezen gebeurt nog buiten de drie stappen en bepaalt niets over de tekst. Welke stap kiest wat de tekst zegt? Zie [Van data naar tekst](/course/open/nlp-what-is-nlg?lang=nl).",
        },
    )
    quick_selection
    return (quick_selection,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 1958 liet Hans Peter Luhn van IBM een IBM 704-computer zinnen kiezen uit teksten in vakbladen. Volgens Luhn is een abstract in technische literatuur bedoeld om snel en goed te laten zien waarover een gepubliceerd stuk gaat, zodat een lezer tijd en moeite bespaart bij het zoeken naar nuttige informatie. Het programma telde woorden: met statistiek over hoe vaak en waar woorden voorkomen, berekende de machine hoe belangrijk elk woord was, en daarna elke zin. De zinnen met de hoogste score kwamen op papier, en vormden samen het auto-abstract, zoals Luhn het noemde.

    Tegenwoordig zijn zelfs de abstracts te veel om te lezen. In 2020 stelden Isabel Cachola, Kyle Lo, Arman Cohan en Daniel S. Weld van het Allen Institute for AI een samenvatting van één zin per paper voor, de *TLDR*, kort voor *too long; didn't read* (te lang; niet gelezen). Zo'n TLDR is volgens de vier onderzoekers een alternatief voor het abstract: de TLDR richt zich op de kern van de paper, zoals wat de paper vooral bijdraagt, en laat achtergrond en details van de methode weg.

    In deze les schrijf je een samenvatter die zelf zinnen kiest, test je de samenvatter op 618 echte abstracts, en meet je wat een samenvatter toevoegt die zelf schrijft.

    **Een voorbeeld uit het dagelijks leven.** Onder elk resultaat toont een zoekmachine een paar regels van de pagina, vaak met je zoekwoorden vetgedrukt: een samenvatting die uit de pagina zelf is gekozen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_first = L.guess(
        "guess",
        question="SciTLDR, de set van Cachola en collega's, bevat 618 abstracts van papers in de informatica met de TLDR's erbij, en markeert in elk abstract de ene zin die het dichtst bij de TLDR's komt. In het nieuws zeggen de eerste zinnen meestal wat het belangrijkst is. Hoe vaak is de dichtstbijzijnde zin van een abstract de eerste zin?",
        options={
            "most": "In ongeveer 3 van elke 4 abstracts",
            "half": "In ongeveer de helft",
            "right": "In ongeveer 1 van elke 5",
            "rarely": "In minder dan 1 van elke 20",
        },
        reveal="In ongeveer 1 van elke 5: in 118 van de 618 abstracts, 19,1%. Een willekeurig gekozen zin zou in 14,8% van de abstracts de dichtstbijzijnde zijn, dus de eerste zin doet het maar een beetje beter dan toeval. Over nieuws schrijven Abigail See en collega's dat nieuwsberichten meestal met de belangrijkste informatie beginnen, en dat dit deels verklaart waarom de lead-3 baseline zo sterk is: de samenvatting van de eerste drie zinnen. Abstracts beginnen vaak juist met achtergrond, zoals het abstract over FearNet: *Incremental class learning involves sequentially learning classes in bursts of examples from the same class.* De dichtstbijzijnde zin is daar de vijfde: *FearNet is a generative model that does not store previous examples, making it memory efficient.*",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Abigail See, Peter Liu en Christopher Manning beschreven in 2017 samenvatten als een tekst terugbrengen tot een kortere versie met de belangrijkste informatie van het origineel. Er zijn volgens See en collega's twee brede aanpakken, extractief en abstractief. Extractieve methoden bouwen een samenvatting alleen uit stukken die letterlijk uit de brontekst komen, meestal hele zinnen; abstractieve methoden kunnen nieuwe woorden en zinsdelen maken die niet in de bron staan, zoals iemand doet die een abstract schrijft.

    1. Een **extractieve samenvatting** kiest zinnen: het is content selection op een tekst, zoals in *Van data naar tekst*. Een samenvatter geeft elke zin een **score** en houdt de hoogste. Luhn gaf zinnen een score op grond van de woorden die vaak voorkwamen. In 1969 probeerde H. P. Edmundson drie aanwijzingen erbij: signaalwoorden zoals *significant*, de plaats van een zin in de tekst, en de woorden van de titel. Over de titel ging Edmundson ervan uit dat een auteur de titel bedenkt als een afbakening van waar het stuk over gaat. En de uitkomst: de drie nieuwe aanwijzingen leverden betere extracten op dan woordfrequentie. Deze les bouwt de titelmethode, met elk titelwoord gewogen met de idf.
    2. Een **abstractieve samenvatting** schrijft eigen zinnen, zoals een language model doet. Zo'n samenvatting kan feiten samenvoegen en details weglaten, maar kan ook iets zeggen wat niet in de tekst staat. Joshua Maynez en collega's vonden in 2020 dat zulke samenvatters heel vaak inhoud verzinnen die niet trouw is aan de tekst. Maynez en collega's onderscheiden twee soorten: een **intrinsieke hallucinatie** verdraait wat de tekst zegt, en een **extrinsieke hallucinatie** voegt iets toe wat de tekst niet zegt.

    Hoe abstractief is een samenvatting? Cachola en collega's meten de mate van abstractie, de novelty, als het percentage woorden van de samenvatting dat niet in de brontekst voorkomt.

    En hoe goed is een keuze? SciTLDR markeert in elk abstract de **orakelzin**, de ene zin die een methode het best zou kiezen. Cachola en collega's zoeken voor elke TLDR van een paper de zin met de meeste ROUGE-overlap, en nemen van deze zinnen de zin met de hoogste ROUGE over alle TLDR's. ROUGE is het onderwerp van de volgende les; hier is genoeg dat de orakelzin de meeste woorden met de TLDR's deelt. Bij de zes abstracts die de course schreef, neemt de course de zin met het grootste aandeel van de verschillende woorden van de TLDR.

    De zes abstracts zijn verzonnen in de stijl van de informatica, elk met een titel en een TLDR die de course schreef. De abstracts zijn in het Engels, zoals bijna alle abstracts in de informatica. Qwen3-0.6B, het chat model van de course, een klein language model dat verder is getraind om in een chat te antwoorden, vatte elk abstract samen in één zin; de course legde de antwoorden vast. Het plaatje toont de twee soorten samenvatting bij het abstract over bijen.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **samenvatten** (summarization) | een tekst korter maken en de belangrijkste inhoud houden |
    | **extractieve samenvatting** | een samenvatting van zinnen die ongewijzigd uit de tekst zijn gekozen |
    | **abstractieve samenvatting** | een samenvatting in eigen zinnen, met misschien woorden die niet in de tekst staan |
    | **score** | een getal voor elke zin dat zegt hoezeer de zin in de samenvatting hoort |
    | **titelmethode** (title method) | een zin een score geven op grond van de woorden die de zin met de titel deelt, naar Edmundson |
    | **TLDR** | een samenvatting van één zin; kort voor *too long; didn't read* |
    | **orakelzin** (oracle sentence) | de zin van een tekst die het dichtst bij de referentiesamenvattingen ligt: de beste ene zin die een methode kan kiezen |
    | **novelty** | het aandeel van de verschillende woorden van een samenvatting dat niet in de tekst staat |
    | **hallucinatie** | wat een samenvatting zegt zonder steun in de tekst: intrinsiek als de tekst verdraaid wordt, extrinsiek als er iets bij komt |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo vat je een tekst samen met de titelmethode:

    1. Splits de titel en elke zin in woorden, in kleine letters.
    2. Geef elke zin een score: tel bij elk verschillend woord dat ook in de titel staat de idf op. Een woord in elk document, zoals *the*, voegt 0 toe.
    3. Kies de k zinnen met de hoogste scores; bij twee gelijke scores de eerdere zin.
    4. Geef de gekozen zinnen in de volgorde van de tekst.

    Zo meet je hoe abstractief een samenvatting is: tel de verschillende woorden die niet in de tekst staan, en deel door het aantal verschillende woorden van de samenvatting.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem zin 2 van het abstract over bijen, geteld vanaf 0. `C.ABSTRACTS` bevat de zes abstracts, elk een dictionary met `"title"`, `"sentences"` en `"tldr"`. `C.words` splitst een tekst in woorden in kleine letters, zoals in *TF-IDF: woorden wegen*, en `C.IDF` geeft de idf van elk woord over de zes abstracts, met elk abstract en de titel ervan als één document. De cel print de woorden van de titel, dan elk woord van de zin dat ook in de titel staat met de idf, en als laatste de score:
    """)
    return


@app.cell
def example_cell():
    _abstract = C.ABSTRACTS["bees"]
    _title = C.words(_abstract["title"])
    print(_title)
    _sentence = C.words(_abstract["sentences"][2])
    _shared = [_w for _w in sorted(set(_sentence)) if _w in _title]
    _total = 0
    for _w in _shared:
        print(_w, round(C.IDF[_w], 3))
        _total = _total + C.IDF[_w]
    print("score:", round(_total, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Negen van de elf titelwoorden staan in zin 2. *the* en *a* staan in alle zes abstracts en voegen 0 toe, en *network*, in vijf van de zes, voegt slechts 0,079 toe. *at* voegt 0,301 toe en *small* 0,176, en *bees*, *entrance*, *hive* en *neural* staan alleen in dit abstract en voegen elk 0,778 toe. De score is 3,67, de hoogste van de zeven zinnen: de eerste zin scoort 1,56, met *bees* en *hive*, zoals het plaatje bij *Het idee* laat zien. *counting* in de titel is niet *counts* in de zin: voor de methode zijn twee vormen van een woord twee woorden. De les *Stam en lemma* laat zien hoe je zulke vormen samenbrengt; de titelmethode hier doet dat niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_title():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. title_score: de titelwoorden van een zin

        Schrijf `title_score`, die `sentence` en `title` krijgt, elk een lijst met woorden, en `idf`, een dictionary van elk woord naar de idf, zoals `C.IDF`. Geef de score van de zin terug: de idf van elk verschillend woord van de zin dat ook in de titel staat, opgeteld. In de startcode begint `score` op 0, en de loop gaat langs de verschillende woorden van de zin, elk één keer. Vervang de twee `...`.

        De check roept `title_score(["bees", "count", "bees", "at", "the", "hive"], ["counting", "bees", "at", "the", "hive"], {"bees": 0.8, "count": 0.3, "at": 0.2, "the": 0.0, "hive": 0.8, "counting": 0.6})` aan, wat `1.8` geeft: *bees*, *at*, *the* en *hive* staan in de titel, en *bees* telt één keer. Daarna `title_score(["het", "netwerk", "telt", "bijen"], ["bijen", "tellen", "met", "een", "netwerk"], {"het": 0.0, "netwerk": 0.1, "telt": 0.8, "bijen": 0.8, "tellen": 0.8, "met": 0.0, "een": 0.0})`, met Nederlandse woorden, en eigen gevallen: een zin zonder titelwoorden, een woord dat drie keer voorkomt, en zinnen van een paar van de zes abstracts met `C.IDF`.
        """),
        L.task("Geef een zin een score met de idf van de woorden die de zin met de titel deelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_title():
    box_title = L.code_box("title_score", lines=7)
    box_title
    return (box_title,)


@app.cell(hide_code=True)
def check_title(box_title):
    result_title = L.check("title_score", box_title.value)
    result_title
    return (result_title,)


@app.cell(hide_code=True)
def hints_title():
    L.hints(
        "title_score",
        [
            "`word` is één verschillend woord van de zin. `in` vraagt of een lijst een waarde bevat, zoals in *Lijsten*, en `idf` is een dictionary: een key geeft de waarde, zoals in *Dictionaries*. `score` groeit met één waarde tegelijk, zoals een lopend totaal.",
            "De `if` vraagt of het woord in de titel staat. Daarbinnen komt de idf van dat ene woord bovenop wat `score` al bevat: `score` links van `=`, en rechts `score` plus deze waarde. Zoek het woord op in de dictionary, niet de hele zin.",
            "`            score = score + ___`",
        ],
        why="`if word in title:` en `score = score + idf[word]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_title():
    mo.md(r"""
    In 3 van de zes abstracts is de zin met de hoogste score de zin die het dichtst bij de TLDR ligt: bij bijen, rivieren en windstroom. In het abstract over tomatenbladeren is de dichtstbijzijnde zin de zin met de uitkomst, *the model finds 89% of the diseased leaves*, en de titelmethode kiest de zin met de methode. In het abstract over lekken zegt de titel *Leaks*, en de zin die zegt wat de paper doet, heeft *a leak*: de methode kiest de eerste zin, over waterbedrijven, met *leaks*, *water* en *pipes*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. pick: de beste zinnen, in de volgorde van de tekst

        Schrijf `pick`, die `scores` krijgt, de score van elke zin in de volgorde van de tekst, en `k`, hoeveel zinnen je kiest. Geef de plaatsen van de k hoogste scores terug, geteld vanaf 0, in de volgorde van de tekst; bij twee gelijke scores gaat de eerdere plaats eerst. In de startcode zet de eerste loop voor elke plaats `i` een tuple in `pairs`; `sorted(pairs)` zet de tuples op volgorde, en de tweede loop haalt uit elk van de eerste k de plaats. `sorted` zet tuples op volgorde van de eerste waarde, en tuples met dezelfde eerste waarde op volgorde van de tweede. Vervang de twee `...`.

        De check roept `pick([0.8, 2.6, 1.1, 2.6, 1.1], 3)` aan, wat `[1, 2, 3]` geeft: de twee keer `2.6`, en van de twee keer `1.1` de eerdere, op plaats 2. Daarna `pick([0.5, 0.0, 1.9], 1)`, wat `[2]` geeft, en eigen gevallen: k gelijk aan het aantal zinnen, k gelijk aan 0, scores die allemaal gelijk zijn, en de scores van drie van de zes abstracts.
        """),
        L.task("Kies de plaatsen van de k hoogste scores, in de volgorde van de tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick", lines=9)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick",
        [
            "`pairs.append` voegt voor plaats `i` één tuple aan `pairs` toe, en `scores[i]` is de score op deze plaats, zoals in *Lijsten*. De plaats `i` komt als tweede in de tuple, omdat de tweede loop de plaats eruit haalt met `pair[1]`. `sorted` gaat van klein naar groot, zoals in *Tuples, sets en sorteren*.",
            "Wil je de hoogste score eerst bij een sortering van klein naar groot, zet de score dan met een min in de tuple: `2.6` wordt `-2.6`, en dat is kleiner dan `-1.1`. Bij een gelijke stand kijkt `sorted` naar de plaats, dus de eerdere plaats komt eerst. De laatste regel geeft de gekozen plaatsen van klein naar groot, en dat is de volgorde van de tekst.",
            "`        pairs.append((___, i))`",
        ],
        why="`pairs.append((-scores[i], i))` en `return sorted(chosen)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    Met k = 2 bevat de samenvatting van het abstract over bijen de zinnen 0 en 2: zin 0 en zin 1 scoren allebei 1,56, en de eerdere gaat mee. Dezelfde regel beslist bij het abstract over plastic met k = 1: zin 1 en zin 2 scoren allebei 2,03, dus de titelmethode kiest zin 1, over donker plastic, en niet zin 2, de dichtstbijzijnde. Edmundson had ook last van gelijke scores: met de eerste waarden voor de woorden van de titelmethode kregen veel zinnen precies dezelfde score.

    ### Probeer het: 618 echte abstracts

    SciTLDR bevat 3.229 papers in de informatica, met TLDR's van de auteurs van inzendingen op OpenReview en TLDR's die experts schreven op grond van de peer reviews: wat andere onderzoekers over elke paper schreven toen de paper werd ingediend. De testset heeft 618 papers met 1.967 TLDR's, de TLDR van de auteurs eerst. De course gebruikt de versie met het abstract als tekst, SciTLDR-A, onder de licentie Apache-2.0. De demo kiest uit elk abstract één zin, op drie manieren: op titelwoorden, zoals je `title_score` doet, met de idf over de 618 abstracts; op gewogen woorden, naar Luhn: de TF-IDF-waarden van de verschillende woorden van een zin, elk met het aantal in het hele abstract, zoals in *TF-IDF: woorden wegen*; en de eerste zin. Kies een paper met de slider om de zinnen en de TLDR's te zien. De abstracts en de TLDR's zijn in het Engels.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciTLDR, 1,2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scitldr_cell(get_button):
    sci_papers = C.scitldr_or_stop(get_button, {
        "ask": "*De testset van SciTLDR staat nog niet op je computer. De knop downloadt de set één keer van de repository op GitHub, 1,2 MB, onder de licentie Apache-2.0. De rest van de les werkt zonder.*",
        "busy": "**Bezig met downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: GitHub was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen plek voor 1,2 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    sci_paper = mo.ui.slider(1, len(sci_papers), value=C.SCI_START, label="Paper", show_value=True)
    sci_paper
    return sci_paper, sci_papers


@app.cell(hide_code=True)
def scitldr_look(sci_paper, sci_papers):
    mo.Html(C.scitldr_picture(L.lang, sci_papers, sci_paper.value))
    return


@app.cell(hide_code=True)
def text_after_scitldr():
    mo.md(r"""
    De titelmethode kiest de dichtstbijzijnde zin in 222 van de 618 abstracts, 35,9%: ongeveer twee keer zo vaak als gewogen woorden, met 104, of 16,8%, wat slechter is dan de eerste zin. Edmundson zag in 1969 dezelfde volgorde: de combinatie van signaalwoorden, titel en plaats had de hoogste gemiddelde overlap met de extracten van mensen, en de methode met alleen frequente woorden, de Key-methode, de laagste van de automatische methoden. Abstracts in de informatica zeggen vaak met de woorden van de titel wat de paper doet: *We propose Aggregated Momentum (AggMo)* staat in de paper *Aggregated Momentum: Stability Through Passive Damping*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_novel():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. novel_share: hoeveel van een samenvatting nieuw is

        Schrijf `novel_share`, die `summary` en `source` krijgt, elk een lijst met woorden; de samenvatting heeft minstens één woord. Geef de novelty terug: het aandeel van de verschillende woorden van de samenvatting dat niet in de bron staat, een getal van 0 tot 1. In de startcode bevat `words` de verschillende woorden van de samenvatting, en `new` telt de woorden die niet in de bron staan. Vervang de twee `...`.

        De check roept `novel_share(["tiny", "network", "counts", "bees", "bees"], ["we", "present", "a", "small", "network", "that", "counts", "bees"])` aan, wat `0.25` geeft: van de vier verschillende woorden is alleen *tiny* nieuw. Daarna `novel_share(["het", "netwerk", "telt", "bijen"], ["we", "tellen", "bijen", "met", "een", "netwerk"])`, met Nederlandse woorden, wat `0.5` geeft, en eigen gevallen: een samenvatting van alleen overgenomen woorden, een samenvatting van alleen nieuwe woorden, en samenvattingen en TLDR's van een paar van de zes abstracts.
        """),
        L.task("Geef het aandeel van de verschillende woorden van een samenvatting dat niet in de bron staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_novel():
    box_novel = L.code_box("novel_share", lines=7)
    box_novel
    return (box_novel,)


@app.cell(hide_code=True)
def check_novel(box_novel):
    result_novel = L.check("novel_share", box_novel.value)
    result_novel
    return (result_novel,)


@app.cell(hide_code=True)
def hints_novel():
    L.hints(
        "novel_share",
        [
            "`word` is één verschillend woord van de samenvatting, en `new` telt de woorden die niet in de bron staan. `not in` vraagt of een waarde in een lijst ontbreekt. Een aandeel is een deel gedeeld door het geheel, met `/`.",
            "Het geheel is het aantal verschillende woorden van de samenvatting, niet van alle woorden: *bees* staat in het eerste geval twee keer, en `new` telt elk verschillend woord één keer. `len` geeft het aantal waarden in een lijst of een set.",
            "`    return new / ___`",
        ],
        why="`if word not in source:` en `return new / len(words)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_novel():
    mo.md(r"""
    De zes samenvattingen van Qwen3-0.6B hebben gemiddeld 33,2% nieuwe woorden, en de zes TLDR's die de course schreef 17,2%. Qwen3 voegt woorden toe als *improving*, *accuracy* en *outperforming*, en in deze woorden zitten de beweringen die de abstracts niet ondersteunen: in vier van de zes samenvattingen, zoals de course het leest. De samenvatting over tomatenbladeren noemt het model dat slecht presteert een model voor een telefooncamera, en laat weg dat het netwerk van het abstract, verder getraind op telefoonfoto's, 89% van de zieke bladeren vindt; het model dat slecht presteert, is op foto's uit een lab getraind: een intrinsieke hallucinatie. De samenvatting over lekken zegt dat het netwerk beter is dan traditionele methoden, terwijl het abstract het netwerk met geen enkele andere methode vergelijkt: een extrinsieke. Nieuwe woorden zijn op zich niet fout: *reducing manual labor* zegt in andere woorden dat tellen met de hand uren kost.

    Op SciTLDR hebben de TLDR's van de auteurs 23,3% nieuwe woorden, gemeten tegen de titel en het abstract. Cachola en collega's meten tegen de hele paper en vinden 9,6%, omdat een TLDR ook woorden uit de rest van de paper kan halen. Maynez en collega's lieten mensen samenvattingen van één zin van BBC-berichten controleren, gemaakt door de samenvatters van 2020: intrinsieke en extrinsieke hallucinaties kwamen vaak voor, in meer dan 70% van de samenvattingen.

    ### Probeer het: kiezen of schrijven

    Kies een abstract, een manier om de zinnen een score te geven, en hoeveel zinnen je kiest. Onder de gekozen zinnen staan de samenvatting van Qwen3-0.6B en de TLDR die de course schreef. Probeer het abstract over lekken met gewogen woorden: zelfs vier zinnen laten de dichtstbijzijnde zin weg. Probeer daarna het abstract over tomatenbladeren met titelwoorden: hoeveel zinnen heb je nodig tot de dichtstbijzijnde zin erbij is?
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_abstract = mo.ui.dropdown(options={C.ABSTRACT_NAMES[L.lang][_key]: _key for _key in C.ABSTRACTS},
                                  value=C.ABSTRACT_NAMES[L.lang]["leaks"], label="Abstract")
    try_method = mo.ui.dropdown(options={C.METHOD_NAMES[L.lang][_m]: _m for _m in C.METHODS},
                                value=C.METHOD_NAMES[L.lang]["title"], label="Score op")
    try_k = mo.ui.slider(1, 4, step=1, value=1, label="Zinnen", show_value=True)
    mo.hstack([try_abstract, try_method, try_k], justify="start", gap=1.5, wrap=True)
    return try_abstract, try_k, try_method


@app.cell(hide_code=True)
def try_look(try_abstract, try_k, try_method):
    mo.Html(C.try_picture(L.lang, try_abstract.value, try_method.value, try_k.value))
    return


@app.cell(hide_code=True)
def text_exercise_risks():
    mo.md(r"""
    ### Oefening 4. Welke samenvatting loopt dit risico?

    Elk probleem hieronder maakt een samenvatting slechter. Zet elk probleem bij de soort samenvatting die het risico loopt.
    """)
    return


@app.cell(hide_code=True)
def exercise_risks():
    sort_risks = L.sort("risks")
    sort_risks
    return (sort_risks,)


@app.cell(hide_code=True)
def check_risks(sort_risks):
    result_risks = L.check("risks", sort_risks)
    result_risks
    return (result_risks,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl) gaf elk woord een waarde naar hoe weinig documenten het woord bevatten; hier laat deze waarde de woorden van de titel meetellen en *the* niet. [Van data naar tekst](/course/open/nlp-what-is-nlg?lang=nl) zette drie stappen van data naar tekst; een extractieve samenvatting doet alleen de eerste stap, content selection, en dan op een tekst, en laat de zinnen ongewijzigd. Een samenvatter die schrijft, zoals een language model doet in [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl), leest beter en kan zeggen wat niet in de tekst staat. De volgende les, *Een samenvatting scoren: ROUGE*, bouwt de maat waarmee SciTLDR de dichtstbijzijnde zin vond, en geeft een samenvatting een score tegen samenvattingen die mensen schreven.

    **Een voorbeeld uit het dagelijks leven.** De preview van een mail in je inbox toont de eerste regel: de eenvoudigste extractieve samenvatting, en, zoals deze les bij abstracts zag, niet altijd de regel die het meest telt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_title, result_pick, result_novel, result_risks):
    L.wrap_up(result_title, result_pick, result_novel, result_risks)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
