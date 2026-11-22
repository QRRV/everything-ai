# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-normalize. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-normalize", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de controles


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl), en een over [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="Wat geeft `re.findall(r\"\\w+|[^\\w\\s]\", \"Café!\")`?",
        options={
            "split": "`[\"Caf\", \"é\", \"!\"]`",
            "right": "`[\"Café\", \"!\"]`",
            "whole": "`[\"Café!\"]`",
        },
        correct="right",
        why={
            "right": "`\\w` matcht ook een letter met een accent, zoals é, dus *Café* is één token, en `!` is een eigen token. Voor een programma zijn het token *Café* en het token *cafe* nog steeds twee verschillende strings; deze les maakt er één van.",
            "split": "`\\w` matcht ook letters met een accent, dus é hoort bij de reeks letters. Zie *Meer stukken van een patroon* in [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "whole": "`!` is geen letter en geen cijfer, dus de reeks letters houdt daar op, en de tweede keuze neemt `!` als een eigen token. Zie *Meer stukken van een patroon* in [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def quick_check_ids():
    quick_ids = L.choice(
        "quick-ids",
        question="De tokenizer van GPT-2 maakt van `\" cafe\"`, met de spatie ervoor, één token. Wat doet de tokenizer met `\" Cafe\"`?",
        options={
            "same": "Hetzelfde token-ID: GPT-2 maakt elke letter klein voordat de tokenizer een token opzoekt",
            "right": "Een ander token-ID: voor GPT-2 zijn *Cafe* en *cafe* verschillende tokens",
            "letters": "Twee token-ID's: een voor de hoofdletter C en een voor de rest",
        },
        correct="right",
        why={
            "right": "GPT-2 verandert de tekst niet: `\" cafe\"` is token 26725 en `\" Cafe\"` token 26965, en `\" café\"` en `\" Café\"` zijn nog twee andere tokens, 40304 en 42151. Zo worden het vier tokens voor wat een lezer als één woord ziet. Deze les voegt zulke vormen samen tot één, en laat zien wat dat kost.",
            "same": "De tokenizer van GPT-2 verandert de tekst niet voordat het knippen begint: `\"the\"` en `\" the\"` zijn verschillende tokens, en een woord met een hoofdletter en hetzelfde woord zonder hoofdletter ook. Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "letters": "Een veelgebruikt woord is één token, vaak met de spatie ervoor, ook als het woord met een hoofdletter begint: `\" Cafe\"` is één token, 26965. Zie [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
        },
    )
    quick_ids
    return (quick_ids,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een programma vergelijkt strings teken voor teken, op hun code points: voor Python is `"Café" == "cafe"` gelijk aan `False`. Een zoekopdracht naar *cafe* mist dus een tekst die *Café* schrijft, en een programma dat woorden telt, ziet twee woorden waar een lezer er één ziet. **Text normalization** zet tekst om naar één standaardvorm voordat een programma ermee werkt. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze beschrijven dat in *Introduction to Information Retrieval* als tokens gelijk maken, zodat tokens ook matchen als de tekens oppervlakkig verschillen. Deze les neemt drie veelgebruikte stappen: alle letters klein, de accenten eraf, en de meest voorkomende woorden eruit. De eerste twee stappen voegen vormen samen die bij elkaar horen, en elke stap voegt ook iets samen of gooit iets weg dat ertoe deed: *US*, de Verenigde Staten, wordt *us* (ons), *één* wordt *een*, en *niet* verdwijnt.

    **Een voorbeeld uit het dagelijks leven.** Iemand zoekt *Zoe* in de contacten van een telefoon en hoopt *Zoë* te vinden: alleen een zoekfunctie die de twee puntjes op de e negeert, vindt deze naam.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Snowball, een project dat voor veel talen hulpmiddelen voor zoeken maakt, houdt een lijst bij van 101 heel gewone Nederlandse woorden, zoals *de*, *en* en *van*. Welk deel van alle woorden in de 199.970 Nederlandse zinnen van Tatoeba is een van deze 101?",
        options={
            "few": "Slechts ongeveer 10%",
            "quarter": "Ongeveer 25%",
            "right": "51,1%",
            "most": "Ongeveer 75%",
        },
        reveal="**51,1%**: 664.197 van de 1.301.039 woorden. In de vijf Engelse boeken van de course beslaat de Engelse lijst van Snowball, met 174 woorden, tussen 48,0% van de woorden (*Moby-Dick*) en 54,9% (*The Adventures of Sherlock Holmes*). Een kleine set heel gewone woorden maakt ongeveer de helft van een tekst uit. Wie deze woorden weglaat, halveert de tekst, en de stukken hieronder laten zien wat er met deze woorden verdwijnt.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Hoofdletters

    De eenvoudigste stap is **case folding**: een veelgebruikte aanpak is volgens Manning en collega's om alle letters klein te maken. In Python doet `.lower()` dat. Of dat een goed idee is, hangt af van wat een hoofdletter vertelt. Jurafsky en Martin vragen zich in het hoofdstuk *Words and Tokens* van *Speech and Language Processing* af of een string met een hoofdletter, zoals *They*, en een string zonder hoofdletter, zoals *they*, als hetzelfde woord moeten tellen. Hun antwoord: dat hangt af van de taak. In het Nederlands en het Engels krijgt het eerste woord van een zin een hoofdletter vanwege de plaats, en die hoofdletter vertelt niets. Maar een hoofdletter kan ook twee woorden uit elkaar houden. Manning en collega's waarschuwen dat case folding woorden gelijk kan maken die je beter apart kunt houden. *Wenen* is de stad, en *wenen* het werkwoord, huilen. En de Engelse lijst van Snowball met heel gewone woorden, de stop words waar een stuk hieronder over gaat, laat het woord *us* (ons) weg, met als toelichting dat je hier moet oppassen omdat US ook de Verenigde Staten is, en dat het woord meestal veilig weg kan als het in kleine letters staat.

    Tussen alles en niets in kan een programma ook alleen sommige woorden klein maken. De eenvoudigste vuistregel is volgens Manning en collega's: maak de woorden aan het begin van een zin klein, en alle woorden in een titel die volledig in hoofdletters staat of waarin de meeste of alle woorden met een hoofdletter beginnen. Een hoofdletter midden in een zin blijft staan, en dat klopt meestal, voegen ze eraan toe. Een model voor machine learning kan dezelfde keuze preciezer maken, en dat noemen ze **truecasing**: voor elke hoofdletter beslissen of de hoofdletter blijft. Aan het begin van een zin kan een programma niet zien of een woord een naam is. Oefening 2 zoekt dat op in een lijst met namen die de course uit Tatoeba maakte: de woorden die midden in een zin vaker met een hoofdletter staan dan in kleine letters. Maar als de mensen die zoeken toch kleine letters typen, helpt het waarschijnlijk weinig om de hoofdletters goed te zetten, schrijven Manning en collega's, en alles klein maken blijft dus vaak de meest praktische oplossing. Python heeft ook `str.casefold()`, dat volgens de documentatie lijkt op klein maken, maar agressiever is: het maakt van de Duitse *ß* een *ss*.

    ### Accenten

    In *Tekst is een rij getallen* is é het code point 233. Unicode kan é ook als twee code points schrijven: e (101), gevolgd door code point 769, een **combinerend teken**: een code point dat iets op het teken ervoor zet, hier een accent aigu. De twee zien er hetzelfde uit. De Unicode-standaard noemt zulke strings in de bijlage *Unicode Normalization Forms* canoniek equivalent: de twee strings staan voor hetzelfde teken, en horen er hetzelfde uit te zien en zich hetzelfde te gedragen. De documentatie van Python laat dat in de *Unicode HOWTO* zien met ê, geschreven als één code point of als e met een combinerend accent circonflexe: geprint ziet de tekst er hetzelfde uit, maar de ene string heeft lengte 1 en de andere lengte 2.

    `unicodedata.normalize` zet een string in een van de **normalisatievormen** van Unicode. In de vorm **NFD** wordt een letter met tekens erop, zoals é, uit elkaar gehaald in de letter en de combinerende tekens. In de vorm **NFC** worden de letter en de tekens weer samengevoegd waar Unicode één code point voor het geheel heeft. De meeste tekst met accenten staat al in NFC, merkt de bijlage op, en de Nederlandse zinnen van Tatoeba staan er allemaal in. Over `unicodedata.combining(chr)` zegt de documentatie dat de functie de canonical combining class van het teken chr geeft, een geheel getal, en 0 als het teken er geen heeft. Voor het accent aigu geeft de functie 230. Accenten weghalen gaat dus in twee stappen: zet de tekst in de vorm NFD, en houd alleen de tekens waarvoor `unicodedata.combining` 0 geeft.

    Accenten, de twee puntjes van het trema in *ë* en het haakje van de cedille in *ç* zijn allemaal **diakritische tekens**: tekens op een letter. Diakritische tekens spelen in het Engels volgens Manning en collega's een vrij kleine rol, en je wilt waarschijnlijk dat *cliché* en *cliche* matchen, of *naive* en *naïve*. Soms verschillen woorden alleen in hun accenten, schrijven ze. Hun voorbeeld is Spaans: *peña* is een rots, en *pena* verdriet. Het Nederlands heeft ook zulke paren. *één* is het getal: Taaladvies.net legt uit dat de accenten alleen op *een* komen als het om het getal gaat en je het woord anders als lidwoord zou kunnen lezen. *vóór* en *voor* kunnen een zin iets anders laten betekenen: Taaladvies.net geeft *Ik voelde me vóór de yogatherapie erg zenuwachtig*, zenuwachtig voordat de therapie begon, tegenover *voor de yogatherapie*, zenuwachtig om naar de therapie te gaan. Een trema laat zien dat een klinker los van de klinker ervoor wordt uitgesproken, zoals in *ruïne* en *poëzie*, zegt de Leidraad, de spellingregels bij de Woordenlijst Nederlandse Taal.

    ### Stop words

    Volgens Manning en collega's laat een zoeksysteem soms heel gewone woorden volledig weg, omdat deze woorden weinig lijken te helpen bij het vinden van documenten die passen bij wat iemand zoekt. Zulke woorden heten **stop words**, en een lijst ervan is een **stop list**. De lijst van Snowball voor het Engels heeft 174 woorden, en de lijst voor het Nederlands 101. Over de Nederlandse lijst zegt het bestand dat de woorden op volgorde staan, van het vaakst naar het minst vaak, en dat de lijst uit een grote steekproef Nederlandse tekst komt. De Engelse lijst laat gewone woorden met een tweede betekenis weg, zoals *can* en *may*, met voorbeelden in de toelichting: een oud blik (*can*), de maand mei (*May*). De Nederlandse lijst houdt zulke woorden, en wijst in de toelichting op de tweede betekenis: *meer* kan ook het meer zijn, het water, en *bij* ook de bij, het insect.

    Een stop list gooit woorden weg die de betekenis kunnen dragen. Jurafsky en Martin geven in het hoofdstuk *Retrieval-based Models* van hetzelfde boek als voorbeeld dat een gewone stop list van *to be or not to be* alleen *not* overhoudt. Volgens Manning en collega's raakt de betekenis van *flights to London* waarschijnlijk kwijt als *to* wegvalt. De Nederlandse lijst bevat *niet*, *geen* en *niets*. De meeste zoekmachines zijn van stop lists afgestapt: volgens Manning en collega's gingen zoeksystemen in de loop van de tijd van vrij grote stop lists (200 tot 300 woorden) naar heel kleine (7 tot 12 woorden), en daarna naar geen stop list meer, en gebruiken zoekmachines op het web meestal geen stop list. Jurafsky en Martin schrijven dat moderne zoeksystemen veel minder vaak een stop list gebruiken, deels omdat zoeksystemen efficiënter zijn geworden, en deels omdat een manier om woorden te wegen, die de les *TF-IDF: de woorden wegen* later behandelt, al veel van hetzelfde werk doet. Toch blijft het weglaten van stop words volgens hen soms nuttig bij taken met taal, en is het de moeite waard om te onthouden.

    ### En language models?

    BERT, een language model dat Google in 2018 publiceerde, is er in een uncased en een cased versie. De README zegt dat `Uncased` betekent dat de tekst klein is gemaakt vóór de tokenization met WordPiece, de tokenizer van BERT, zodat `John Smith` `john smith` wordt, en dat het `Uncased` model ook de accenten weghaalt. Meestal is het `Uncased` model volgens de README beter, tenzij je weet dat de hoofdletters belangrijk zijn voor je taak, zoals bij het vinden van de namen in een tekst (Named Entity Recognition) of van de woordsoort van elk woord (Part-of-Speech tagging). GPT-2 doet niets van dit alles. Alec Radford en collega's schrijven in de paper van GPT-2, *Language Models are Unsupervised Multitask Learners*, dat hun model met bytes werkt en geen voorbewerking nodig heeft die informatie weggooit, zodat ze het model op elke benchmark voor language models kunnen testen. Voor GPT-2 is *café* in de vorm NFD dus een andere tekst: `" café"` met één code point voor é is token 40304, en met e en het combinerende accent worden het drie tokens: het token van `" cafe"`, en twee tokens met elk één byte van het accent.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **text normalization** | tekst omzetten naar één standaardvorm voordat een programma ermee werkt |
        | **case folding** | elke letter klein maken, zodat *Café* en *café* één string worden |
        | **truecasing** | voor elke hoofdletter beslissen of de hoofdletter blijft: een hoofdletter die een woord door de plaats heeft, gaat weg, de hoofdletter van een naam blijft |
        | **combinerend teken** | een code point dat iets op het teken ervoor zet, zoals het accent aigu 769 |
        | **diakritisch teken** | een teken op een letter: een accent zoals in é of è, een trema zoals in ë, een cedille zoals in ç |
        | **normalisatievorm**: **NFC**, **NFD** | een vaste manier om de tekens van Unicode te schrijven: elke letter met de tekens erop als één code point waar dat kan; letters zoals é los van de tekens erop |
        | **stop word**, **stop list** | een heel gewoon woord dat een programma weglaat, zoals *de* of *the*; de lijst van zulke woorden |
        | `import unicodedata` | de module van Python met wat Unicode over elk teken zegt |
        | `unicodedata.normalize("NFD", s)` | de string `s` in de vorm NFD: `len(unicodedata.normalize("NFD", "é"))` is 2 |
        | `unicodedata.combining(c)` | 0 voor een teken dat geen combinerend teken is, en voor de meeste combinerende tekens een geheel getal boven 0, zoals 230 voor het accent aigu |
        | `"\u0301"` | `\u` en een code point in vier hexadecimale cijfers, zoals Unicode code points schrijft: `0301` is 769, het combinerende accent aigu, en `ord("\u0301")` geeft 769 |
        | `s.casefold()` | zoals `s.lower()`, maar agressiever: `"Straße".casefold()` is `"strasse"` |
        | `words[1:]` | de lijst vanaf het tweede item: de lijst zonder het eerste item |
        """),
        L.watch_out("Een woord klein maken kan een vorm geven die de spelling niet kent. Het Nederlands zet geen accent op een hoofdletter aan het begin van een zin: je schrijft *Eén*, niet *Één*, zegt Taaladvies.net. Maar `\"Eén\".lower()` is *eén*, en dat is *één* (het getal) niet en *een* ook niet, en Tatoeba begint 75 zinnen met *Eén*. Fold, in oefening 1, haalt de accenten eraf na het klein maken, en maakt er *een* van."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Vraag wat de taak nodig heeft: houdt een hoofdletter, een accent of een klein woord hier twee dingen uit elkaar?
    2. Knip de tekst in tokens.
    3. Maak de letters klein, of alleen het eerste woord van elke zin, en houd de namen.
    4. Haal de accenten eraf: de vorm NFD, en laat dan de combinerende tekens weg.
    5. Laat de stop words alleen weg als de taak zonder deze woorden kan, en tel wat er verdwijnt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel zet *Café* in de vorm NFD, en laat de code points ervoor en erna zien, en wat `unicodedata.combining` voor elk teken geeft.
    """)
    return


@app.cell
def example_nfd():
    import unicodedata

    W = "Café"
    decomposed = unicodedata.normalize("NFD", W)
    print(len(W), len(decomposed))
    print([ord(c) for c in W])
    print([ord(c) for c in decomposed])
    print([unicodedata.combining(c) for c in decomposed])
    print(unicodedata.normalize("NFC", decomposed) == W)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In de vorm NFD wordt é (233) een e (101) en het combinerende accent aigu (769), dus de string wordt één teken langer. `unicodedata.combining` geeft 0 voor de vier letters en 230 voor het accent, en de vorm NFC voegt de twee weer samen. Oefening 1 laat de combinerende tekens weg, en maakt het woord klein.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fold():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. fold: kleine letters, geen accenten

        Schrijf `fold(word)`: `word` in kleine letters, zonder accenten of andere tekens op de letters. Maak het woord klein met `.lower()`, haal het woord uit elkaar met `unicodedata.normalize` in de vorm NFD, en plak de tekens die geen combinerend teken zijn weer aan elkaar met `"".join` om een list comprehension. Voor `fold("Café")` is het resultaat `"cafe"`. Vervang de `...`. De controle roept `fold("Café")` aan, en daarna eigen gevallen: het getal *één*, hoofdletters met een trema, twee trema's, de hoofdletter IJ, een apostrof, *café* geschreven met e en het combinerende accent, `"cafe\u0301"`, een cedille, en een lege string.
        """),
        L.task("Maak een woord klein en haal de accenten eraf: de vorm NFD, en laat dan de combinerende tekens weg."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fold():
    box_fold = L.code_box("fold", lines=5)
    box_fold
    return (box_fold,)


@app.cell(hide_code=True)
def check_fold(box_fold):
    result_fold = L.check("fold", box_fold.value)
    result_fold
    return (result_fold,)


@app.cell(hide_code=True)
def hints_fold():
    L.hints(
        "fold",
        [
            "Maak het woord klein en zet het in de vorm NFD in één regel, en geef het resultaat een naam, zoals `decomposed`. Geef daarna de tekens zonder de combinerende tekens terug, aan elkaar geplakt met `\"\".join` om een list comprehension met een `if`.",
            "Wat geeft `unicodedata.combining` voor een teken dat geen combinerend teken is? Welke test is waar voor precies deze tekens?",
            r'`    return "".join([c for c in decomposed if ___])`',
        ],
        why=r'`decomposed = unicodedata.normalize("NFD", word.lower())`, en dan `return "".join([c for c in decomposed if unicodedata.combining(c) == 0])`.',
    )
    return


@app.cell(hide_code=True)
def text_after_fold():
    mo.md(r"""
    Met je fold worden de 42.673 verschillende woordvormen van Tatoeba er 38.591. Kleine letters alleen maken er 38.682 van, dus het meeste samenvoegen komt van de hoofdletters. In Tatoeba hebben 791 vormen een accent of een ander teken, en samen komen deze vormen 5.065 keer voor. De vorm die het vaakst voorkomt, is *Australië* (852 keer). Daarna komt *één* (671 keer), dat net als *Eén* (77 keer) *een* wordt, en dan *geïnteresseerd* (169) en *skiën* (137). Bij *Australië* en *australie* gaat er niets verloren. Maar *één* en *een* zijn twee woorden, en fold maakt er één van. De Leidraad telt het trema niet als accent; fold haalt het trema er toch af. Manning en collega's telden wat case folding doet met het Engelse nieuws van Reuters-RCV1, na het weghalen van de getallen: 17% minder verschillende termen. In Tatoeba geven kleine letters 9,4% minder.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_truecase():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. truecase: alleen de hoofdletter van het eerste woord

        `NAMES` hieronder bevat de lijst met namen van de course: de 3.951 woorden die midden in een zin van Tatoeba vaker met een hoofdletter staan dan in kleine letters, zoals *Tom*, *Boston* en *IJsland*. Schrijf `truecase(words, names)`, voor de woorden van één zin: als `words` leeg is, of als het eerste woord in de set `names` staat, geef je `words` onveranderd terug; anders een nieuwe lijst met het eerste woord in kleine letters, gevolgd door de andere woorden, onveranderd. Voor `truecase(["Het", "regent", "in", "Wenen"], NAMES)` is het resultaat `["het", "regent", "in", "Wenen"]`. Vervang de `...`. De controle roept dat aan, en daarna eigen gevallen: een naam aan het begin, een zin zonder woorden, een naam in het midden, *Eén* aan het begin, de hoofdletter IJ, een woord in hoofdletters, een eigen lijst met namen, en een lege lijst met namen.
        """),
        L.task("Maak het eerste woord van een zin klein, behalve als het woord een naam is."),
    ])
    return


@app.cell
def names_cell():
    NAMES = C.names()
    print(len(NAMES), "Tom" in NAMES, "Het" in NAMES)
    return


@app.cell(hide_code=True)
def exercise_truecase():
    box_truecase = L.code_box("truecase", lines=4)
    box_truecase
    return (box_truecase,)


@app.cell(hide_code=True)
def check_truecase(box_truecase):
    result_truecase = L.check("truecase", box_truecase.value)
    result_truecase
    return (result_truecase,)


@app.cell(hide_code=True)
def hints_truecase():
    L.hints(
        "truecase",
        [
            "Eerst het geval waarin niets verandert, met één `if` en `or`: een lijst zonder woorden, of een zin die met een naam begint. Geef daarna een nieuwe lijst terug: het eerste woord klein, gevolgd door de andere woorden.",
            "Welk item van `words` is het eerste woord, en hoe vraag je of een waarde in een set staat?",
            r'`    if len(words) == 0 or ___:`',
        ],
        why=r'`if len(words) == 0 or words[0] in names:` geeft `words` terug; anders `return [words[0].lower()] + words[1:]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_truecase():
    mo.md(r"""
    Met je truecase vormen de woorden van Tatoeba 39.788 verschillende woorden, tegenover 40.603 als alleen het eerste woord klein wordt gemaakt, zonder lijst met namen: de 815 extra woorden zijn namen die klein zijn gemaakt, zoals *mary* en *boston*, en die de zinnen verder nergens in kleine letters schrijven. *Tom*, dat 17.626 zinnen begint, houdt de hoofdletter. Maar de lijst kent alleen wat het midden van een zin laat zien. Vijf zinnen beginnen met *Wenen*: drie gebruiken het werkwoord, zoals *Wenen zal niet helpen*, en twee de stad, zoals *Wenen is de hoofdstad van Oostenrijk*. Midden in een zin schrijft Tatoeba 42 keer *wenen* en 10 keer *Wenen*, dus *Wenen* staat niet op de lijst, en je functie maakt de stad klein. Een naam die alleen zinnen begint, zoals *Esther*, staat er ook niet op. En een regel van Tatoeba kan meer dan één zin bevatten, dus de lijst bevat ook woorden als *Ja*, *Nee* en *Eén*: midden in zo'n regel staan deze woorden met een hoofdletter. Een model dat truecasing leert, kan naar meer kijken dan het woord zelf, zoals de woorden erna.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_drop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. drop_stop_words: laat de gewone woorden weg

        `STOP_WORDS` hieronder bevat de 101 Nederlandse stop words van Snowball, en `STOP_WORDS_EN` de 174 Engelse, allemaal in kleine letters. Schrijf `drop_stop_words(words, stop_words)`: de woorden van `words`, in volgorde en zoals geschreven, zonder de woorden waarvan de vorm in kleine letters in de set `stop_words` staat. Gebruik een list comprehension met een `if`. Voor `drop_stop_words(["Tom", "is", "niet", "thuis"], STOP_WORDS)` is het resultaat `["Tom", "thuis"]`. Vervang de `...`. De controle roept dat aan, en daarna eigen gevallen: een stop word met een hoofdletter, een lege lijst, een woord dat een stop word is met een andere betekenis, een regel Engels met `STOP_WORDS_EN`, *US* en *us*, een leesteken, en een eigen lijst met stop words.
        """),
        L.task("Laat de woorden van een lijst weg waarvan de vorm in kleine letters een stop word is."),
    ])
    return


@app.cell
def stop_words_cell():
    STOP_WORDS = C.stop_words("nl")
    STOP_WORDS_EN = C.stop_words("en")
    print(len(STOP_WORDS), len(STOP_WORDS_EN), "niet" in STOP_WORDS, "us" in STOP_WORDS_EN)
    return


@app.cell(hide_code=True)
def exercise_drop():
    box_drop = L.code_box("drop-stop-words", lines=2)
    box_drop
    return (box_drop,)


@app.cell(hide_code=True)
def check_drop(box_drop):
    result_drop = L.check("drop-stop-words", box_drop.value)
    result_drop
    return (result_drop,)


@app.cell(hide_code=True)
def hints_drop():
    L.hints(
        "drop-stop-words",
        [
            "Bouw de nieuwe lijst met een list comprehension over `words`, met een `if` die een woord alleen houdt als het woord geen stop word is.",
            "De lijsten bevatten kleine letters. Welke vorm van het woord zoek je op, en welke vorm houd je?",
            r'`    return [word for word in words if ___]`',
        ],
        why=r'`return [word for word in words if word.lower() not in stop_words]`: de vorm in kleine letters om op te zoeken, het woord zoals geschreven om te houden.',
    )
    return


@app.cell(hide_code=True)
def text_after_drop():
    mo.md(r"""
    Je functie laat 51,1% van de woorden van Tatoeba weg, en tussen 44,6% en 54,9% van de woorden van elk van de acht boeken van de course, vijf Engelse en drie Nederlandse. Alle 29.667 zinnen met *niet*, *geen* of *niets* raken dat woord kwijt, en 844 zinnen raken elk woord kwijt, zoals *Ik kan niet meer!* Het voorbeeld van Jurafsky en Martin komt er nog leger uit: met de Engelse lijst van Snowball blijft er van *to be or not to be* niets over, omdat *not* ook op deze lijst staat. En in de Engelse boeken zit de apostrof in de weg: de boeken schrijven *don’t* met een gekrulde apostrof, en de lijst schrijft *don't* met een rechte, dus 1.744 woorden zoals *don’t* en *it’s* blijven staan. De apostrof normaliseren zou nog een stap zijn.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: je eigen tekst, stap voor stap

    Typ een tekst en kies een lijst met stop words. De afbeelding laat zien wat elke stap van deze les met de tekst doet: tokens, kleine letters, accenten eraf, stop words eruit. Probeer *naïef*, *Eén*, *IJsland*, een zin met *niet*, of *to be or not to be* met de Engelse lijst.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="Is Tom niet in het café in Wenen?", label="Tekst", full_width=True)
    demo_list = mo.ui.dropdown(options={"Nederlands": "nl", "Engels": "en"}, value="Nederlands", label="Stop words")
    mo.vstack([demo_text, demo_list])
    return demo_list, demo_text


@app.cell(hide_code=True)
def demo_look(demo_list, demo_text):
    mo.Html(C.demo_picture(demo_text.value, demo_list.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    *Is Tom niet in het café in Wenen?* wordt *tom cafe wenen ?*: de vraag is er nog, maar niet wat er gevraagd wordt. De stappen zijn voor elke tekst hetzelfde, en wat de stappen weggooien, verschilt per tekst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Wat heeft de taak nodig?

    Elke vetgedrukte taak vraagt iets van een programma dat zoekt of woorden telt. Zet elke taak bij wat de taak nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_keep = L.sort("what-to-keep")
    sort_keep
    return (sort_keep,)


@app.cell(hide_code=True)
def check_sort(sort_keep):
    result_sort = L.check("what-to-keep", sort_keep)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl) knipte een tekst in tokens. Hier werden de tokens minder: met case folding zijn *Café* en *café* één; met NFD en zonder de combinerende tekens zijn *café* en *cafe* één; en zonder de stop words is ongeveer de helft van elke tekst weg. Elke stap gooide ook iets weg: *Wenen* de stad en *wenen* het werkwoord, *één* en *een*, en de *niet* die een zin omdraait. Manning en collega's noemen case folding, het weglaten van stop words en stemming, dat de volgende les behandelt, vormen van compressie die informatie kwijtraakt: wat zulke stappen weghalen, komt niet terug. [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) liet een tokenizer zien die elke vorm onveranderd laat; het `Uncased` model van BERT maakt alles klein en haalt de accenten weg, het `Cased` model houdt de hoofdletters en de accenten. De volgende les van het domein, *Stam en lemma*, gaat een stap verder, en voegt de vormen van één woord samen, zoals *wandelt*, *wandelde* en *wandelen*.

    **Een voorbeeld uit het dagelijks leven.** De zoekfunctie van een webwinkel die *Café Noir* vindt als je *cafe noir* typt, maakt alles klein en haalt de accenten weg, zowel in wat je typt als in de namen van de producten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_drop, result_fold, result_sort, result_truecase):
    L.wrap_up(result_fold, result_truecase, result_drop, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
