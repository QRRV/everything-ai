# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-zipf-heaps. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-zipf-heaps", "nl")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl), en één over [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="Op een log-log-plot loopt een rechte lijn van $x$ = 1, $y$ = 1.000 omlaag naar $x$ = 100, $y$ = 10. Wat is de helling?",
        options={
            "normal": "−10",
            "gridlines": "−2",
            "right": "−1",
        },
        correct="right",
        why={
            "right": "Twee factoren 10 omlaag, van 1.000 naar 10, over twee factoren 10 opzij, van 1 naar 100: −2 gedeeld door 2 is −1. Een helling van ongeveer −1 komt in deze les terug, als je uitzet hoe vaak woorden voorkomen tegen hun rang.",
            "normal": "Dat is de helling op gewone assen: (10 − 1.000) / (100 − 1). Op een log-log-plot deel je de verandering van de logaritmen. Zie *Het idee* in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
            "gridlines": "De lijn gaat twee factoren 10 omlaag, maar ook twee opzij, en de helling is omhoog gedeeld door opzij. Zie *Het idee* in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question="De tokenizer van GPT-2 komt een woord tegen dat niet tussen de 50.257 tokens van de lijst staat. Wat gebeurt er met het woord?",
        options={
            "unknown": "Het woord krijgt één token voor onbekend",
            "right": "Het woord valt uiteen in tokens",
            "skip": "Het woord wordt weggelaten",
        },
        correct="right",
        why={
            "right": "Een zeldzamer woord valt uiteen in subwords, stukken die wel in de lijst staan. Deze les laat zien waarom een tokenizer die nodig heeft: hoe lang een lijst hele woorden ook is, er komen steeds nieuwe woorden bij.",
            "unknown": "Zo'n token heeft de tokenizer van GPT-2 niet: elke tekst wordt tokens uit de lijst. Wat gebeurt er met een woord dat niet als geheel in de lijst staat? Zie de tabel *Nieuwe woorden en tekens* in [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
            "skip": "Een tokenizer zet elke tekst om in tokens en weer terug, dus er raakt niets kwijt. Wat gebeurt er met een woord dat niet als geheel in de lijst staat? Zie de tabel *Nieuwe woorden en tekens* in [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tel de woorden van een willekeurige tekst, en hetzelfde patroon duikt op: een handvol woorden neemt een groot deel ervan in, en de meeste verschillende woorden komen één of twee keer voor. Lees verder, en er komen steeds nieuwe woorden bij, hoeveel je ook al gelezen hebt. Beide patronen doen ertoe voor elk model van taal. Een model ziet de meeste woorden maar een paar keer, te weinig om over elk woord veel te leren, en een lijst woorden uit de ene tekst mist altijd woorden van de volgende. Jurafsky en Martin schrijven in hun hoofdstuk over woorden en tokens dat geen vocabulary, hoe groot ook, alle woorden kan bevatten die zouden kunnen voorkomen. Dat is een van de redenen waarom de tokenizers van [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) zeldzame woorden in subwords knippen. Zoekmachines hebben met dezelfde groei te maken: in *Introduction to Information Retrieval* wijzen Manning, Raghavan en Schütze erop dat volgens de wet van Heaps, een van de twee wetten van deze les, de lijst woorden die een zoekmachine kan opzoeken blijft groeien met elk document dat erbij komt, zonder ooit een maximum te bereiken.

    **Een voorbeeld uit het dagelijks leven.** Denk aan de berichten die je in een week stuurt: *ik*, *je*, *de* en *oké* komen steeds terug, terwijl de meeste andere woorden één keer opduiken. En een reis naar een nieuwe stad brengt woorden die je nog nooit typte: de namen van de straten en van de gerechten.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="De vijf Engelse boeken van de course bevatten samen 551.664 woorden. Hoeveel van de woorden die het vaakst voorkomen, maken samen de helft daarvan uit?",
        options={
            "right": "Ongeveer 75",
            "mid": "Ongeveer 750",
            "many": "Ongeveer 7.500",
        },
        reveal="**Ongeveer 75**: de 76 woorden die het vaakst voorkomen, van *the*, *and* en *of* tot en met *other*, maken de helft van de 551.664 woorden uit, van 23.131 verschillende. Aan de andere kant komen 8.659 verschillende woorden maar één keer voor. *Alice* alleen heeft 58 woorden nodig voor de helft van de tekst, en in *Tom Sawyer* van Mark Twain vonden Manning en Schütze dat de 100 woorden die het vaakst voorkomen, 50,9% van de tekst uitmaken.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een paar woorden, heel vaak: de wet van Zipf

    Tel elk woord van een tekst, en zet de verschillende woorden op volgorde, van het vaakst naar het minst vaak. De plaats van een woord in die lijst is de **rang**: het woord dat het vaakst voorkomt, heeft rang 1, het volgende rang 2, enzovoort. In *Human Behavior and the Principle of Least Effort* uit 1949 schreef George Kingsley Zipf op hoe het aantal $f$ van een woord samenhangt met de rang $r$: rang keer aantal blijft ongeveer gelijk, $r \times f = C$. Het aantal is dus ongeveer een vast getal gedeeld door de rang, $f \approx C / r$: het woord op rang 2 komt ongeveer half zo vaak voor als het woord op rang 1, en het woord op rang 10 ongeveer een tiende zo vaak. Op een log-log-plot, zoals in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl), is dat een rechte lijn met helling −1. Het belangrijkste voorbeeld van Zipf was de roman *Ulysses* van James Joyce: 260.430 woorden, waarvan 29.899 verschillende, geteld voor een woordindex van het boek uit 1937. Het woord op rang 10 komt 2.653 keer voor en het woord op rang 100 nog 265 keer: tien keer verder in de lijst, een tiende van het aantal.

    Zipf zag het niet als eerste. Vanaf de vierde druk uit 1916 stonden in *Gammes sténographiques*, een boek over steno van de Franse stenograaf Jean-Baptiste Estoup, tellingen van het Frans: in een tekst van 20.000 woorden maakten zo'n twaalf woorden samen 40% van de tekst uit. Zipf noemde Estoup de eerste, voor zover Zipf wist. In 1928 zette Edward Condon, toen bij Bell Telephone Laboratories, de tellingen van twee Engelse woordenlijsten, elk uit 100.000 woorden, op logaritmische schalen, en de punten lagen dicht bij een rechte lijn met helling −1. Zipf maakte het patroon bekend, en de wet is naar Zipf genoemd. In 1953 en 1954 voegde Benoit Mandelbrot twee getallen aan de formule toe, zodat de formule beter past bij de woorden die het vaakst voorkomen.

    Waarom zouden woorden zo'n wet volgen? Zipf zag een evenwicht tussen twee soorten moeite: wie spreekt, gebruikt liever een paar woorden voor alles, en wie luistert, heeft liever voor elke betekenis een ander woord. Zipf noemde dat evenwicht zelf een hypothese, en een verklaring waar iedereen het over eens is, is er sindsdien niet gekomen.

    De afbeelding toont de tien woorden die het vaakst voorkomen in twee corpora: de vijf Engelse boeken van de course, en 1.301.039 woorden Nederlands van nu uit Tatoeba, een verzameling korte zinnen die vrijwilligers schrijven en vertalen, uitgebracht onder CC BY 2.0 FR. De course nam de Nederlandse zinnen op 21 november 2026 over, zonder de zinnen met een grof, seksueel of gewelddadig woord. De open punten zijn wat de wet van Zipf voorspelt vanuit het eerste woord: het aantal van het eerste woord gedeeld door de rang.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_tail():
    mo.md(r"""
    ### De meeste woorden, zelden

    Het andere eind van de lijst is lang. Een woord dat maar één keer voorkomt in een tekst of een corpus, is een **hapax legomenon**, Grieks voor eenmaal gezegd; het meervoud is *hapax legomena*. Het gaat niet om een paar vreemde woorden. In *Ulysses* komen 16.432 van de 29.899 verschillende woorden één keer voor, 55%; in *Tom Sawyer* telden Manning en Schütze er 3.993 van de 8.018, 49,8%. Een groter corpus lost dat niet op: het brengt nieuwe zeldzame woorden mee. Een model ziet de meeste verschillende woorden van het corpus dus maar één of twee keer, te weinig om over elk ervan veel te leren.

    En de wet geldt maar ongeveer. In een overzicht van het onderzoek uit 2014 schrijft Steven Piantadosi dat de tellingen van taal de wet van Zipf maar bij benadering volgen: op een log-log-plot is het ongeveer een rechte lijn, maar zelfs de woorden die het vaakst voorkomen, volgen de wet niet precies, en de zeldzaamste woorden buigen af van de lijn. Een kromme die sterk op die van Zipf lijkt, komt zelfs uit willekeurig typen, met letters en spaties op goed geluk aangeslagen, zoals Miller in 1957 opmerkte en Li in 1992 uitwerkte. Een rechte lijn alleen zegt dus weinig over taal; of willekeurige tekst ook in de details op echte tellingen lijkt, is omstreden. Toch zijn zulke tellingen volgens Piantadosi gevonden in elke taal die onderzocht is, zelfs in het Meroïtisch, een uitgestorven taal die nog niemand kan vertalen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_heaps():
    mo.md(r"""
    ### Er blijven nieuwe woorden komen: de wet van Heaps

    Lees nu een tekst vanaf het begin, en houd bij hoeveel verschillende woorden je al bent tegengekomen. Eerst is bijna elk woord nieuw; later heb je steeds meer woorden al eerder gezien, maar er blijven nieuwe bijkomen. Noem het aantal gelezen woorden $N$, en het aantal verschillende woorden daaronder $V$. In het hoofdstuk *Words and Tokens* van *Speech and Language Processing* schrijven Jurafsky en Martin het patroon als $V = k \cdot N^\beta$, met $k$ en $\beta$ positieve getallen en $\beta$ tussen 0 en 1. Ze noemen het de wet van Herdan of de wet van Heaps, naar Gustav Herdan, die het in 1960 in de taalkunde beschreef, en Harold Heaps, die het in 1978 voor zoeken beschreef. Deze course zegt de wet van Heaps; die Nederlandse naam is van de course zelf, naar het voorbeeld van de wet van Zipf. Op een log-log-plot is het weer een rechte lijn, nu met helling $\beta$. Voor $\beta$ zijn volgens Jurafsky en Martin vaak waarden van 0,44 tot 0,56 gemeld, en soms nog hoger: het aantal verschillende woorden groeit iets sneller dan de wortel van de lengte. Hoe goed dat kan passen, staat in *Introduction to Information Retrieval*: voor nieuwsberichten van Reuters vinden Manning, Raghavan en Schütze $k$ = 44 en $\beta$ = 0,49, wat 38.323 verschillende woorden voorspelt in de eerste 1.000.020 woorden. Het echte aantal is 38.365.

    De twee wetten horen bij elkaar. Als de meeste woorden zeldzaam zijn, zoals de wet van Zipf zegt, komt een langere tekst steeds zeldzame woorden tegen die er nog niet in stonden, dus het aantal verschillende woorden blijft groeien, maar steeds langzamer. Lü, Zhang en Zhou lieten in 2010 zien hoe de wet van Heaps volgt uit de wet van Zipf; hoe snel de verschillende woorden groeien, hangt af van de exponent van de wet van Zipf en van de lengte van de tekst. De groei is ook niet vanaf het begin één rechte lijn: in boeken van Project Gutenberg vonden Tria, Loreto en Servedio dat de verschillende woorden eerst bijna even snel groeien als de tekst, en pas later langzamer, dicht bij de wortel. Jurafsky en Martin verklaren het eerste stuk met woorden die vaak voorkomen en dan nog voor het eerst opduiken.

    Voor een model van taal is dit het probleem in één formule. Een lijst van hele woorden, hoe lang ook, mist woorden die de volgende tekst meebrengt: namen, typefouten, nieuwe woorden, en in het Nederlands de samenstellingen die als één woord geschreven worden. Sennrich, Haddow en Birch noemden vertalen in 2016 een probleem met een open woordenschat, vooral voor talen die nieuwe woorden maken door woorden aan elkaar te zetten, en lieten een model voor vertalen zeldzame woorden in subwords knippen. De tokenizers van nu doen hetzelfde met een lijst van vaste lengte: 50.257 tokens voor GPT-2, 151.669 voor Qwen3.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **rang** | de plaats van een woord in een lijst van het vaakst naar het minst vaak voorkomende woord: het woord dat het vaakst voorkomt, heeft rang 1 |
        | **wet van Zipf** | het aantal van een woord is ongeveer omgekeerd evenredig met de rang: $f \approx C / r$, dus rang keer aantal, $r \cdot f$, blijft ongeveer gelijk |
        | **hapax legomenon** | een woord dat maar één keer voorkomt in een tekst of een corpus; meervoud *hapax legomena* |
        | **wet van Heaps** | het aantal verschillende woorden in een tekst groeit ongeveer als een macht van de lengte: $V \approx k \cdot N^\beta$, met $\beta$ tussen 0 en 1 |
        | $N$, $V$ | het aantal woorden van een tekst, waarbij elke keer dat een woord voorkomt telt, en het aantal verschillende woorden erin |
        | `sorted(xs, reverse=True)` | de waarden van `xs` van groot naar klein; `reverse=True` is een keyword argument |
        | `np.arange(a, b)` | een array van de gehele getallen van `a` tot `b`, zonder `b` zelf, zoals `range(a, b)` telt |
        | `d.values()` | de waarden van de dictionary `d`, zonder de keys |
        | `seen.add(x)` | stopt `x` in de set `seen`; een set bevat elke waarde één keer, dus een waarde toevoegen die er al in zit, verandert niets |
        | `n % step` | de rest van `n` gedeeld door `step`: 0 als `n` een heel aantal stappen is |
        """),
        L.watch_out("De wet van Zipf en de wet van Heaps zijn patronen die tellingen ongeveer volgen, geen regels waar elke telling zich aan houdt. De getallen hangen ook af van wat je als woord telt: met *De* en *de* apart, of met *zo'n* in twee stukken, verandert elke telling."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip de tekst in woorden, en beslis wat als hetzelfde woord telt: hier dezelfde letters in kleine letters.
    2. Tel hoe vaak elk woord voorkomt, en sorteer de aantallen van hoog naar laag. De plaats in die lijst is de rang.
    3. Zet het aantal tegen de rang uit op een log-log-plot. Een lijn die ongeveer recht loopt, met een helling van ongeveer −1, is de wet van Zipf; rang keer aantal blijft dan ongeveer gelijk.
    4. Voor de wet van Heaps ga je één keer door de tekst, met een set van de woorden die je tot nu toe zag, en schrijf je om de zoveel woorden de grootte van de set op.
    5. Zet de verschillende woorden uit tegen de gelezen woorden op een log-log-plot. De helling tussen twee punten is de exponent $\beta$, berekend zoals in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel geeft je de woorden van *Alice's Adventures in Wonderland*, zoals in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl), en, uit `C.counts_by_rank`, de eigen functie van de course, de aantallen van de verschillende woorden, van het vaakst naar het minst vaak: wat je eigen functie van oefening 1 straks geeft. De cel print voor een paar rangen het aantal en rang keer aantal. De tweede cel telt de verschillende woorden in de eerste 1.000 woorden, de eerste 10.000 en het hele boek, telkens met een nieuwe set van de woorden in kleine letters, en berekent de exponent $\beta$ tussen de eerste twee punten, zoals stap 5 van het recept zegt.
    """)
    return


@app.cell
def book_cell():
    import numpy as np

    ALICE = C.book_words("alice")
    _counts = C.counts_by_rank(ALICE)
    print(len(ALICE), len(_counts))
    print(_counts[:5])
    for _rank in [1, 2, 3, 10, 100, 1000]:
        print(_rank, _counts[_rank - 1], _rank * _counts[_rank - 1])
    return ALICE, np


@app.cell
def example_growth(ALICE, np):
    _v_a = len(set([w.lower() for w in ALICE[:1000]]))
    _v_b = len(set([w.lower() for w in ALICE[:10000]]))
    print(_v_a, _v_b, len(set([w.lower() for w in ALICE])))
    _beta = (np.log10(_v_b) - np.log10(_v_a)) / (np.log10(10000) - np.log10(1000))
    print(round(_beta, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Alice* heeft 26.776 woorden, waarvan 2.636 verschillende. Het woord dat het vaakst voorkomt, *the*, staat er 1.651 keer in. De wet van Zipf voorspelt op rang 2 ongeveer de helft daarvan, 826, en het boek heeft er 874; op rang 10 een tiende, 165, en het boek heeft er 386. Rang keer aantal gaat van 1.651 op rang 1 naar 3.860 op rang 10 en 4.900 op rang 100, en terug naar 3.000 op rang 1.000: het verandert met een factor 3, terwijl het aantal zelf daalt van 1.651 naar 3. Zo ongeveer geldt de wet. De tweede cel vindt 389 verschillende woorden in de eerste 1.000 woorden, 1.552 in de eerste 10.000 en 2.636 in het hele boek: tien keer zoveel tekst, ongeveer vier keer zoveel verschillende woorden, dus $\beta$ is 0,6, iets meer dan een wortel, die 0,5 zou geven.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rank_counts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. rank_counts: woorden op rang

        Schrijf `rank_counts(words)`: tel hoe vaak elk woord van `words` voorkomt, in kleine letters, zodat *The* en *the* één woord zijn, en geef de aantallen terug in een lijst, van hoog naar laag. De woorden zelf blijven buiten de lijst: plaats 0 bevat het aantal van het woord op rang 1, plaats 1 het aantal op rang 2, enzovoort. Gebruik een dictionary van woord naar aantal, en `sorted(..., reverse=True)` voor de volgorde. Voor `rank_counts(ALICE)` begint de lijst met `[1651, 874, 729]`. Vervang de `...`. De check roept `rank_counts(ALICE)` aan, en daarna eigen gevallen: een paar korte lijsten, een met hoofdletters en een met woorden die even vaak voorkomen, een lijst van één woord en een lege lijst.
        """),
        L.task("Tel de woorden van een tekst, en geef de aantallen van hoog naar laag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rank_counts():
    box_rank_counts = L.code_box("rank-counts", lines=9)
    box_rank_counts
    return (box_rank_counts,)


@app.cell(hide_code=True)
def check_rank_counts(box_rank_counts):
    result_rank_counts = L.check("rank-counts", box_rank_counts.value)
    result_rank_counts
    return (result_rank_counts,)


@app.cell(hide_code=True)
def hints_rank_counts():
    L.hints(
        "rank-counts",
        [
            "Begin met een lege dictionary `counts`. Ga de woorden langs, zet elk woord in kleine letters, en werk het aantal van het woord bij, of geef het woord het aantal 1 als het nog niet in de dictionary staat. Sorteer na de loop de waarden van de dictionary van hoog naar laag.",
            "Een woord staat al in `counts`. Wat is het nieuwe aantal?",
            "`            counts[w] = ___`",
        ],
        why="`counts = {}`, dan `for w in words:` met daaronder `w = w.lower()`, en `if w in counts:` met `counts[w] = counts[w] + 1`, `else:` met `counts[w] = 1`. Na de loop: `return sorted(counts.values(), reverse=True)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rank_counts():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op twee hele corpora: de vijf Engelse boeken, 551.664 woorden, en de Nederlandse zinnen van Tatoeba, 1.301.039 woorden. Op logaritmische schalen lopen beide lijnen vanaf rang 10 ongeveer recht, dicht bij de gestippelde lijn van 1 gedeeld door de rang, en beide eindigen in treden: duizenden woorden aan het eind van de lijst delen het aantal 2, en dan nog duizenden het aantal 1. Die laatste zijn de hapax legomena.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_product():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. rank_times_count: het product van Zipf, met NumPy

        `COUNTS` hieronder bevat de aantallen van *Alice* op rang, als NumPy-array. Schrijf `rank_times_count(counts)`: geef een array terug met voor elke rang de rang keer het aantal. De rangen gaan 1, 2, 3 enzovoort, één voor elk aantal; `np.arange` maakt die rangen, en `*` vermenigvuldigt twee even lange arrays plaats voor plaats. Voor `COUNTS` begint de array met 1.651, 1.748 en 2.187. Vervang de `...`. De check roept `rank_times_count(COUNTS)` aan, en daarna eigen gevallen: korte arrays, waarvan één leeg.
        """),
        L.task("Vermenigvuldig elk aantal met de rang, met NumPy."),
    ])
    return


@app.cell
def counts_cell(ALICE, np):
    COUNTS = np.array(C.counts_by_rank(ALICE))
    print(COUNTS[:5])
    return (COUNTS,)


@app.cell(hide_code=True)
def exercise_product():
    box_product = L.code_box("rank-times-count", lines=6)
    box_product
    return (box_product,)


@app.cell(hide_code=True)
def check_product(box_product):
    result_product = L.check("rank-times-count", box_product.value)
    result_product
    return (result_product,)


@app.cell(hide_code=True)
def hints_product():
    L.hints(
        "rank-times-count",
        [
            "Maak een array `ranks` met één rang voor elk aantal, en geef `ranks` keer `counts` terug.",
            "Waar begint `np.arange(a, b)`, en waar stopt de reeks?",
            "`    ranks = np.arange(___)`",
        ],
        why="`ranks = np.arange(1, len(counts) + 1)` geeft 1, 2, 3 tot en met `len(counts)`; dan `return ranks * counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_product():
    mo.md(r"""
    Rang keer aantal is de manier waarop Zipf zelf de wet schreef: als het gelijk blijft, is het aantal een constante gedeeld door de rang. In de afbeelding daalt het aantal van de Engelse boeken de lijst af met een factor van meer dan 30.000, en verandert rang keer aantal met een factor 6: veel vlakker, al is het niet vlak. Het stijgt over de eerste rangen, en zakt aan het eind in treden, waar duizenden woorden één aantal delen en hun volgorde onderling alleen een plaats is in een lijst van gelijken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_growth():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. growth: er komen steeds nieuwe woorden bij

        Schrijf `growth(words, step)`: ga `words` één keer langs, met een set van de woorden die je tot nu toe zag, in kleine letters. Voeg na elke `step` woorden een paar toe aan een lijst: hoeveel woorden je hebt gelezen, en hoeveel verschillende woorden de set bevat. Een paar woorden aan het eind, minder dan `step`, geven geen punt. Geef de lijst terug. Het uitgewerkte voorbeeld maakte voor elk punt een nieuwe set; op een corpus van een miljoen woorden duurt dat te lang, dus hier groeit één set terwijl je verder gaat. `n % step` geeft de rest van `n` gedeeld door `step`. Voor `growth(ALICE, 5000)` zijn de eerste twee punten `(5000, 1066)` en `(10000, 1552)`. Vervang de `...`. De check roept `growth(ALICE, 5000)` aan, en daarna eigen gevallen: korte lijsten, een met hoofdletters, een stap die niet precies in de lengte past, een stap die langer is dan de lijst, en een lege lijst.
        """),
        L.task("Schrijf om de paar woorden op hoeveel verschillende woorden een tekst heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_growth():
    box_growth = L.code_box("growth", lines=8)
    box_growth
    return (box_growth,)


@app.cell(hide_code=True)
def check_growth(box_growth):
    result_growth = L.check("growth", box_growth.value)
    result_growth
    return (result_growth,)


@app.cell(hide_code=True)
def hints_growth():
    L.hints(
        "growth",
        [
            "Begin met een lege set `seen` en een lege lijst `points`. Ga de plaatsen `i` langs met `range(len(words))`, en stop elk woord in kleine letters in `seen`. Is het aantal tot nu toe gelezen woorden een heel aantal stappen, voeg dan het paar van dat aantal en `len(seen)` toe aan `points`.",
            "Hoeveel woorden heb je gelezen na het woord op plaats `i`, als `i` vanaf 0 telt?",
            "`        if ___ % step == 0:`",
        ],
        why="`seen = set()` en `points = []`, dan `for i in range(len(words)):` met daaronder `seen.add(words[i].lower())`, en `if (i + 1) % step == 0:` met `points.append((i + 1, len(seen)))`. Na de loop: `return points`.",
    )
    return


@app.cell(hide_code=True)
def text_after_growth():
    mo.md(r"""
    Op gewone assen buigen beide lijnen af: de verschillende woorden komen steeds langzamer. Op log-log-assen lopen de lijnen bijna recht, een machtswet, met $\beta$ ongeveer 0,64 voor de Engelse boeken en 0,65 voor de Nederlandse zinnen tussen 10.000 en 100.000 woorden: iets boven de 0,44 tot 0,56 die Jurafsky en Martin noemen, al melden ze ook hogere waarden. De Engelse lijn maakt een sprong waar een nieuw boek begint, zoals *Pride and Prejudice* na de 26.776 woorden van *Alice*: elk boek brengt eigen namen en woorden mee.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de woorden van een corpus

    Kies een corpus: een van de acht boeken, de vijf Engelse boeken samen, de drie Nederlandse boeken samen, of de Nederlandse zinnen van Tatoeba. De afbeelding geeft de getallen, de aantallen op rang en de groei op log-log-assen, en de woorden op een paar rangen. Vergelijk *Alice* met *Moby-Dick*, de Engelse boeken met de Nederlandse, en de oude spelling van de Nederlandse boeken met het Nederlands van nu in Tatoeba.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    _names = C.corpus_names(L.lang)
    demo_corpus = mo.ui.dropdown(options=_names, value=list(_names)[0], label="Corpus")
    demo_corpus
    return (demo_corpus,)


@app.cell(hide_code=True)
def demo_look(demo_corpus):
    mo.Html(C.demo_picture(demo_corpus.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Elk corpus laat beide wetten zien, en elk corpus heeft eigen getallen. In de drie Nederlandse boeken komt 51,0% van de verschillende woorden maar één keer voor, tegen 37,4% in de Engelse. Een van de redenen: het Nederlands schrijft een samenstelling als één woord, dus *treinkaartje* is één zeldzaam woord waar het Engels *train ticket* schrijft, twee woorden die vaak voorkomen. In Tatoeba is *tom* het zevende woord van de lijst, omdat 12,7% van de Nederlandse zinnen de naam Tom gebruikt, en *ik* en *je* staan hoog, zoals in een gesprek; in de boeken staan *en* en *de* vooraan. Een telling uit een corpus is een telling van dat corpus, zoals [Corpora en treebanks](/course/open/nlp-corpora?lang=nl) al zei.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_laws():
    mo.md(r"""
    ### Oefening 4. Welke wet?

    Elke vetgedrukte uitspraak gaat over een van de twee wetten van deze les. Zet de uitspraak bij die wet.
    """)
    return


@app.cell(hide_code=True)
def exercise_laws():
    sort_laws = L.sort("laws")
    sort_laws
    return (sort_laws,)


@app.cell(hide_code=True)
def check_laws(sort_laws):
    result_laws = L.check("laws", sort_laws)
    result_laws
    return (result_laws,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Corpora en treebanks](/course/open/nlp-corpora?lang=nl) stelde verzamelingen tekst samen; deze les telde de woorden ervan. In elk corpus komen een paar woorden heel vaak voor en de meeste zelden, ongeveer zoals de wet van Zipf zegt, en er komen steeds nieuwe woorden bij naarmate de tekst verdergaat, ongeveer zoals de wet van Heaps zegt. Je telde woorden op rang, vermenigvuldigde rang met aantal met NumPy, volgde de verschillende woorden door een tekst, en hield de twee wetten uit elkaar. Omdat er nooit een eind komt aan nieuwe woorden, knippen de tokenizers van [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) zeldzame woorden in subwords; in een latere les bouw je zelf zo'n tokenizer. De volgende les van het domein gaat over de talen zelf: *Zo'n 7.000 talen*.

    **Een voorbeeld uit het dagelijks leven.** De volgende keer dat je een woord tegenkomt dat je nog nooit zag, in een boek, een spel of op een menukaart, denk dan aan Heaps: hoeveel je ook leest, er is altijd nog een woord meer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_growth, result_laws, result_product, result_rank_counts):
    L.wrap_up(result_rank_counts, result_product, result_growth, result_laws)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
