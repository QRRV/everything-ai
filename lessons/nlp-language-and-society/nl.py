# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-language-and-society. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-language-and-society", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_family():
    quick_family = L.choice(
        "quick-family",
        question="Glottolog zet het Nederlands, het Engels en het Fries in één tak van een familie: het West-Germaans. Wat zegt dat over de drie talen?",
        options={
            "order": "De drie zetten onderwerp, werkwoord en lijdend voorwerp in dezelfde volgorde",
            "right": "De drie zijn voortgekomen uit één oudere taal",
            "understand": "De sprekers verstaan elkaar",
        },
        correct="right",
        why={
            "right": "Een familie gaat over afkomst: het Nederlands, het Engels en het Fries zijn voortgekomen uit één oudere taal. Of de sprekers elkaar verstaan, is een andere vraag, en daar komt deze les op terug.",
            "order": "Dat zou een type zijn, geen familie: verwante talen kunnen verschillen in woordvolgorde, zoals het Nederlands en het Engels. Waar staat een tak van een familie voor? Zie *Families* in [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
            "understand": "Verwante talen kunnen ver uit elkaar liggen: wie Engels spreekt, volgt geen Fries zonder het te leren. Waar staat een tak van een familie voor? Zie *Families* in [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
        },
    )
    quick_family
    return (quick_family,)


@app.cell(hide_code=True)
def quick_check_dialect():
    quick_dialect = L.choice(
        "quick-dialect",
        question="Wanneer telt Glottolog twee manieren van spreken als twee talen, en niet als twee dialecten van één taal?",
        options={
            "countries": "Als de twee aan weerszijden van een landsgrens worden gesproken",
            "spelling": "Als elk een eigen spelling heeft",
            "right": "Als de sprekers elkaar niet verstaan",
        },
        correct="right",
        why={
            "right": "Glottolog kijkt of de sprekers elkaar verstaan. Anderen beslissen met andere maatstaven, zoals een wet of een spelling, en deze les laat zien waar die maatstaven uiteenlopen.",
            "countries": "Een grens splitst voor Glottolog geen taal: het Nederlands wordt in Nederland en in België gesproken. Waar kijkt Glottolog naar? Zie *Hoeveel talen?* in [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
            "spelling": "Een spelling is een beslissing van mensen, en Glottolog kijkt naar de sprekers zelf. Waar kijkt Glottolog naar? Zie *Hoeveel talen?* in [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl).",
        },
    )
    quick_dialect
    return (quick_dialect,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In november 1962 vroeg William Labov verkopers in drie warenhuizen in New York naar een afdeling op de vierde verdieping, en luisterde hoe ze *fourth floor* zeiden: met of zonder de *r*. De warenhuizen waren gerangschikt van hoog naar laag: Saks Fifth Avenue, Macy's en S. Klein. Bij Saks sprak 62 procent van de verkopers de *r* minstens één keer uit, bij Macy's 51 procent en bij S. Klein 20 procent: dezelfde stad en dezelfde twee woorden, en toch hing de manier van spreken samen met de rang van het warenhuis. Zulke patronen bestudeert de sociolinguïstiek, en ook in tekst voor NLP zitten zulke patronen. Jurafsky en Martin schrijven in het hoofdstuk *Words and Tokens* van *Speech and Language Processing* dat de meeste talen ook meerdere variëteiten hebben, vaak gesproken in verschillende regio's of door verschillende sociale groepen. Een tool leert de variëteiten van de data, en werkt minder goed voor de andere. In 2016 lieten Su Lin Blodgett, Lisa Green en Brendan O'Connor langid.py, een tool die zegt in welke taal een tekst is, los op tweets. Hun model verbond de woorden van gebruikers, via volkstellingscijfers van de plaatsen waar de tweets vandaan kwamen, met Afro-Amerikanen of met witte Amerikanen, en de onderzoekers gingen na dat de eerste groep tweets bekende trekken van Afro-Amerikaans Engels heeft, een variëteit van het Amerikaans Engels die miljoenen mensen spreken. Van die tweets zag langid.py 13,2% aan voor een andere taal dan Engels, tegen 7,6% van de tweets die het model met witte sprekers verbond. In 2020 testten Allison Koenecke en anderen de systemen van vijf bedrijven die spraak omzetten in tekst, op interviews met 42 witte en 73 zwarte sprekers: bij elk systeem lag het foutpercentage voor de zwarte sprekers bijna twee keer zo hoog. De witte sprekers waren opgenomen in Californië en de zwarte in het oosten van de Verenigde Staten, dus een deel van het verschil kan komen van waar de sprekers wonen en niet van hun etnische groep, voegen de onderzoekers eraan toe. Deze les maakt dezelfde soort telling voor het Nederlands, met woorden die je kunt zien: wat een woordenlijst van standaardnederlands mist in het Fries, het Gronings en het Limburgs.

    **Een voorbeeld uit het dagelijks leven.** Typ een woord uit je eigen dialect of uit je stad op je telefoon, en kijk of het een rode streep krijgt, of in een ander woord verandert.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Tatoeba, een verzameling zinnen die vrijwilligers schrijven en vertalen, heeft 13.206 zinnen in het Gronings, dat in de provincie Groningen gesproken wordt, met een Nederlandse vertaling. Van de woorden van de Nederlandse vertalingen ontbreekt ongeveer 1 op de 16 in een woordenlijst uit de Nederlandse treebank van de course. En van de woorden van de Groningse zinnen?",
        options={
            "eight": "Ongeveer 1 op de 8: twee keer zoveel",
            "four": "Ongeveer 1 op de 4",
            "right": "Bijna 1 op de 2",
        },
        reveal="**Bijna 1 op de 2**: 32.492 van de 68.803 woorden, 47,2%, tegen 4.539 van de 71.152, 6,4%, in de Nederlandse vertalingen. De zinnen zeggen hetzelfde, dus het verschil komt van de variëteit, niet van waar de zinnen over gaan. Bij de tien woorden die de lijst het vaakst mist, horen kleine, veelvoorkomende woorden zoals *n*, *nait*, *veur* en *noar*, waar het Nederlands *een*, *niet*, *voor* en *naar* heeft.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Variëteiten

    Niemand spreekt een taal op maar één manier. In de eerste regels van hun inleiding schrijven Blodgett, Green en O'Connor dat er in talen over de hele wereld regionale en sociale dialecten bestaan, en dat die variëteiten of dialecten van de standaardvariëteit verschillen in zinsbouw, in klanken en in woordenschat. Zo'n vorm van een taal heet een **taalvariëteit**, kortweg **variëteit**; het woord zegt niet welke vorm de echte taal is. Een variëteit die bij een regio hoort, is een **dialect**, een variëteit die bij een sociale groep hoort een **sociolect**, ook wel sociaal dialect, en de manier van spreken of schrijven die bij een situatie hoort een **register**: je schrijft anders aan een vriend dan aan de gemeenteraad. Een **accent** is alleen een manier van uitspreken. Eén persoon kan meer dan één variëteit gebruiken, van het ene moment op het andere. De **sociolinguïstiek** bestudeert hoe taal samenhangt met regio's, groepen en situaties, en hoe mensen met taal laten zien wie ze zijn.

    ### Standaardnederlands in twee landen

    Eén variëteit heeft een bijzondere plaats: de **standaardtaal**. Taaladvies.net, het taaladvies van de Nederlandse Taalunie, beschrijft die als het Nederlands dat je overal in het openbare leven kunt gebruiken, zoals in het bestuur, de administratie, de rechtspraak, het onderwijs en de media. Ook de standaardtaal is een variëteit, en niet overal precies gelijk: volgens Taaladvies is de standaardtaal in Nederland en België grotendeels dezelfde, maar sommige woorden zijn alleen in een van de twee landen standaardtaal, met het label *standaardtaal in België* of *standaardtaal in Nederland*. Een Nederlander heeft het over een *pinpas*, een Belg over een *bankkaart*, en beide woorden zijn standaardtaal. Wat in Nederland *burgemeester en wethouders* heet, heet in België *burgemeester en schepenen*. Het Nederlands is volgens de website van de Taalunie de thuistaal van meer dan 25 miljoen mensen. De Taalunie gaat terug op een verdrag tussen België en Nederland van 1980, en sloot in 2003 een associatieovereenkomst met Suriname.

    Tussen dialect en standaardtaal heeft Vlaanderen de **tussentaal**. Team Taaladvies van de Vlaamse overheid omschrijft tussentaal als twee dingen tegelijk: de verzorgde taal van Vlamingen die dialect spreken en geen standaardtaal kunnen of willen spreken, en de ontspannen taal van Vlamingen die standaardtaal spreken en op persoonlijker of informelere momenten geen dialect kunnen of willen spreken. Team Taaladvies laat de stappen zien met één vraag, in standaardtaal, tussentaal en een dialect: *Heb je nog een kop koffie voor me?*, *'Ebde nog een tas koffie voor mij?* en *Édde nog een zjat kaffe veu maai?* De stappen lopen in elkaar over. Tussentaal heeft een slechte naam: volgens Robertus de Louw (2016) is de heersende houding tegenover tussentaal negatief, en Geert van Istendael bedacht er in 1989 de naam *Verkavelingsvlaams* voor.

    ### Talen naast het Nederlands

    Nederland heeft ook eigen talen. Volgens de Wet gebruik Friese taal, die op 1 januari 2014 in werking trad, zijn het Nederlands en het **Fries** de officiële talen van de provincie Fryslân; in twee onderzoeken uit 2025 gaf 53% van de inwoners van Fryslân het Fries op als eerste taal. In de indeling van Glottolog is het Fries geen tak van het Nederlands: het Fries deelt de tak Anglo-Fries met het Engels. Nederland erkende in 1996 het Nedersaksisch en in 1997 het Limburgs als **regionale taal**, onder het Europees Handvest voor regionale talen of talen van minderheden. Het Gronings is een van de variëteiten van het Nedersaksisch, zeggen de Nederlandse overheid en Glottolog allebei. Volgens de definitie van het Handvest zelf is een regionale taal geen dialect van de officiële taal: dialecten van de officiële talen en de talen van migranten vallen erbuiten. Dat het Limburgs een regionale taal heet, en geen groep dialecten van het Nederlands, is dus een beslissing van mensen, en de woorden blijven hoe dan ook dezelfde. Max Weinreich maakte er een Jiddisch gezegde over beroemd: een taal is een dialect met een leger en een vloot. In 1945 schreef Weinreich dat een toehoorder bij een reeks lezingen van Weinreich het had gezegd.

    ### Wat een tool kent

    Een tool voor tekst kent de woorden van de tekst waarop die gebouwd is. In deze les is dat een **woordenlijst**: elk woord van een treebank in standaardnederlands, in kleine letters, met hoe vaak het voorkomt. Een woord dat niet in de lijst staat, is een **onbekend woord**: een tool die de woorden opzoekt in de lijst, vindt het niet. Hoe minder woorden van een variëteit de lijst heeft, hoe meer van zulke woorden een tool tegenkomt die op de lijst gebouwd is. De afbeelding zet twee zinnen van Tatoeba, een verzameling zinnen die vrijwilligers schrijven en vertalen, naast de Nederlandse vertaling, met de onbekende woorden onderstreept, zoals een spellingcontrole dat doet.
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
        | **sociolinguïstiek** (sociolinguistics) | de studie van hoe taal samenhangt met regio's, sociale groepen en situaties, en hoe mensen met taal laten zien wie ze zijn |
        | **taalvariëteit** (language variety) | elke vorm van een taal die sprekers met elkaar delen, zoals een dialect, een sociolect of de standaardtaal |
        | **dialect** | een variëteit die bij een regio hoort: een plaatselijke vorm van een taal |
        | **sociolect** | een variëteit die bij een sociale groep hoort |
        | **register** | de manier van spreken of schrijven die bij een situatie hoort, zoals een gesprek met vrienden of een brief aan de gemeenteraad |
        | **accent** | een manier van uitspreken, typisch voor een regio, een groep of een persoon |
        | **standaardtaal** (standard language) | de variëteit voor het hele openbare leven, zoals het bestuur, de rechtspraak, het onderwijs en de media; in het Nederlands zijn sommige woorden alleen in België of alleen in Nederland standaardtaal |
        | **tussentaal** | in Vlaanderen: de taal tussen dialect en standaardtaal |
        | **regionale taal** (regional language) | een taal die de staat in een regio erkent en die geen dialect is van de officiële taal, zoals het Nedersaksisch en het Limburgs in Nederland |
        | **woordenlijst** (word list) | de woorden die een tool kent: hier elk woord van een treebank, in kleine letters, met hoe vaak het voorkomt |
        | **onbekend woord** (unknown word) | een woord dat niet in de woordenlijst staat |
        | `w in counts` | `True` als `w` een sleutel is van de dictionary `counts`: als het woord in de lijst staat |
        | `x not in y` | `True` als `x` niet in `y` zit: het omgekeerde van `x in y` |
        """),
        L.watch_out("Een variëteit is geen kapotte vorm van de standaardtaal: elke variëteit heeft eigen regels, en de standaardtaal is één variëteit naast andere, met een bijzondere plaats in het openbare leven. Een onbekend woord zegt iets over de woordenlijst, niet over de spreker. En de zinnen van Tatoeba laten een variëteit zien zoals vrijwilligers die schrijven, niet zoals de sprekers met elkaar praten."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Maak een woordenlijst van tekst in de standaardtaal: elk woord, in kleine letters, met hoe vaak het voorkomt.
    2. Neem zinnen in een variëteit, en dezelfde zinnen in de standaardtaal, elk als een lijst woorden.
    3. Tel de woorden van elke kant die niet in de lijst staan, en deel door alle woorden van die kant.
    4. Vergelijk de twee delen: de zinnen zeggen hetzelfde, dus het verschil komt van de variëteit.
    5. Tel welke woorden de lijst het vaakst mist, om te zien waar een tool struikelt die op de lijst gebouwd is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel bevat een piepkleine woordenlijst van de course, `SMALL`, als set, en een Fries spreekwoord van Tatoeba met de Nederlandse vertaling, elk als een lijst woorden: *De appel falt net fier fan de beam* en *De appel valt niet ver van de boom*. De tweede cel geeft de woorden van een zin terug die niet in de lijst staan, opgezocht in kleine letters, zodat *De* aan het begin telt als *de*.
    """)
    return


@app.cell
def sentences_cell():
    SMALL = {"de", "appel", "valt", "niet", "ver", "van", "boom"}
    FRISIAN_EXAMPLE = ["De", "appel", "falt", "net", "fier", "fan", "de", "beam"]
    DUTCH_EXAMPLE = ["De", "appel", "valt", "niet", "ver", "van", "de", "boom"]
    return DUTCH_EXAMPLE, FRISIAN_EXAMPLE, SMALL


@app.cell
def example_missing(DUTCH_EXAMPLE, FRISIAN_EXAMPLE, SMALL):
    def missing(words, known):
        found = []
        for word in words:
            if word.lower() not in known:
                found.append(word)
        return found

    print(missing(FRISIAN_EXAMPLE, SMALL))
    print(missing(DUTCH_EXAMPLE, SMALL))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In het Friese spreekwoord mist de piepkleine lijst vijf van de acht woorden, en geen enkel woord van de Nederlandse vertaling; *De* wordt gevonden, omdat de lijst in kleine letters wordt bevraagd. Met de woordenlijst van de hele treebank, in de afbeelding hierboven, houdt de Friese zin maar drie onbekende woorden over: *falt*, *fan* en *beam*. Die lijst kent *net* en *fier*, omdat het Nederlands die woorden ook heeft, met een andere betekenis: in het Fries is *net* hetzelfde als *niet*, en *fier* hetzelfde als *ver*; in het Nederlands betekent *net* verzorgd of zojuist, en *fier* trots. Een woordenlijst kent vormen, geen betekenissen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. word_counts: de woordenlijst van een treebank

        `ALPINO` hieronder bevat de 13.603 zinnen van de Nederlandse treebank van [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl), elk een lijst van de woorden zoals de treebank die schrijft, met hoofdletters en leestekens. Schrijf `word_counts(sentences)`: tel hoe vaak elk woord voorkomt in alle zinnen samen, in kleine letters, zodat *De* en *de* één woord zijn, in een dictionary van het woord naar het aantal. De sleutels van de dictionary zijn de woordenlijst: `w in counts` vraagt of het woord `w` erin staat. Voor `word_counts(ALPINO)` heeft de dictionary 27.169 woorden, en komt `"de"` 13.035 keer voor. Vervang de `...`. De check roept `word_counts(ALPINO)` aan, en daarna eigen gevallen: korte lijsten zinnen, één woord in hoofdletters en in kleine letters, een lege zin, en een lege lijst.
        """),
        L.task("Tel de woorden van een treebank in standaardnederlands: de woordenlijst van een tool."),
    ])
    return


@app.cell
def treebank_cell():
    ALPINO = C.treebank()
    print(len(ALPINO), ALPINO[0])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("word-counts", lines=10)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("word-counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "word-counts",
        [
            "Begin met een lege dictionary `counts`. Ga de zinnen langs, en binnen die loop de woorden van elke zin. Zet elk woord in kleine letters, `w`, en werk het aantal bij, of geef het het aantal 1 als het nog niet in de dictionary staat. Geef na beide loops de dictionary terug.",
            "Het woord `w` staat al in `counts`. Wat is het nieuwe aantal?",
            "`                counts[w] = ___`",
        ],
        why="`counts = {}`, dan `for sentence in sentences:` en daarbinnen `for word in sentence:`, met `w = word.lower()` en daaronder `if w in counts:`, met `counts[w] = counts[w] + 1`, en `else:` met `counts[w] = 1`. Na beide loops: `return counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    Elk Belgisch woord in de afbeelding is standaardtaal in België, en toch heeft de treebank geen enkele keer *hospitaal*, *plezant*, *proper* of *bankkaart*, terwijl *ziekenhuis* er 27 keer in staat en *leuk* 19 keer. Alleen *schepen* komt voor, 11 keer, en elke keer als meervoud van *schip*, en niet één keer als een Belgische schepen: ook hier kent de lijst de vorm en niet de betekenis. In deze paren kent een tool die op deze treebank gebouwd is de woorden van het hele taalgebied en van Nederland, en niet de woorden die alleen in België standaardtaal zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. share_unknown: hoeveel de lijst mist

        Hieronder bevat `COUNTS` de telling van de treebank, zoals `word_counts(ALPINO)` die geeft: de woordenlijst. `FRISIAN` bevat de 506 Friese zinnen van Tatoeba met een Nederlandse vertaling, en `FRISIAN_NL` die vertalingen, in dezelfde volgorde; `GRONINGS` en `GRONINGS_NL` hetzelfde voor het Gronings. Elke zin is een lijst van de woorden. Schrijf `share_unknown(sentences, known)`: het deel van alle woorden van alle zinnen samen dat niet in `known` staat, opgezocht in kleine letters, als een getal tussen 0 en 1. Een lange zin weegt zwaarder dan een korte, want die heeft meer woorden. Zonder woorden geef je 0 terug. Voor `share_unknown(FRISIAN, COUNTS)` is de uitkomst 0,5877. Vervang de `...`. De check roept `share_unknown(FRISIAN, COUNTS)` aan, en daarna eigen gevallen: de Nederlandse vertalingen, korte zinnen met een eigen woordenlijst, woorden met een hoofdletter, een lange zin en een korte, een lege zin, en geen zinnen.
        """),
        L.task("Bereken welk deel van de woorden van een variëteit een woordenlijst van de standaardtaal niet kent."),
    ])
    return


@app.cell
def pairs_cell():
    COUNTS = C.counts()
    FRISIAN, FRISIAN_NL = C.sentences("fry")
    GRONINGS, GRONINGS_NL = C.sentences("gos")
    print(len(COUNTS), FRISIAN[3], FRISIAN_NL[3])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share-unknown", lines=11)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share-unknown", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share-unknown",
        [
            "Houd twee aantallen bij, allebei beginnend bij 0: `total`, voor alle woorden, en `unknown`, voor de woorden die in kleine letters niet in `known` staan. Ga de zinnen langs, werk `total` bij bij elke zin, en ga de woorden van de zin langs om `unknown` bij te werken. Geef na de loops 0 terug als er geen woorden waren, en anders het deel.",
            "Hoeveel woorden voegt één zin toe aan `total`?",
            "`        total = total + ___`",
        ],
        why="`total = 0` en `unknown = 0`, dan `for sentence in sentences:` met daaronder `total = total + len(sentence)`, en `for word in sentence:` met `if word.lower() not in known:` en `unknown = unknown + 1`. Na de loops: `if total == 0:` met `return 0`, en dan `return unknown / total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    De zinnen van een paar zeggen hetzelfde, dus wat de lijst aan de ene kant mist en aan de andere niet, komt van de variëteit, niet van het onderwerp. De lijst mist 58,8% van de Friese woorden en 47,2% van de Groningse, tegen 6,1% en 6,4% in de Nederlandse vertalingen. Zelfs in standaardnederlands mist de lijst ongeveer 1 op de 16 woorden: namen, en alledaagse woorden die de 13.603 zinnen van een treebank toevallig nooit tegenkwamen, zoals *fles* en *soep*. Het Limburgs heeft maar 18 paren, korte zinnen zoals groeten, dus de staven van het Limburgs zeggen weinig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unknown():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. unknown_counts: de woorden die de lijst mist

        Schrijf `unknown_counts(sentences, known)`: tel de woorden van de zinnen met `word_counts`, en houd in een nieuwe dictionary alleen de woorden die niet in `known` staan, elk met hoe vaak het in de zinnen voorkomt. `word_counts` staat klaar voor je functie: de check geeft de functie van de course mee. Voor `unknown_counts(FRISIAN, COUNTS)` heeft de dictionary 745 woorden, en komt `"it"` 68 keer voor. Vervang de `...`. De check roept `unknown_counts(FRISIAN, COUNTS)` aan, en daarna eigen gevallen: de Groningse zinnen, korte zinnen met een eigen woordenlijst, woorden met een hoofdletter, een woord dat terugkomt, alleen bekende woorden, en geen zinnen.
        """),
        L.task("Tel welke woorden van een variëteit een woordenlijst van de standaardtaal mist."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unknown():
    box_unknown = L.code_box("unknown-counts", lines=7)
    box_unknown
    return (box_unknown,)


@app.cell(hide_code=True)
def check_unknown(box_unknown):
    result_unknown = L.check("unknown-counts", box_unknown.value)
    result_unknown
    return (result_unknown,)


@app.cell(hide_code=True)
def hints_unknown():
    L.hints(
        "unknown-counts",
        [
            "Vraag `word_counts(sentences)` om de telling. Maak een nieuwe, lege dictionary `unknown`, en ga de woorden `w` van de telling langs: als `w` niet in `known` staat, geef het een plaats in `unknown`. Geef na de loop `unknown` terug.",
            "Welk getal hoort bij een woord dat je houdt?",
            "`            unknown[w] = ___`",
        ],
        why="`counts = word_counts(sentences)` en `unknown = {}`, dan `for w in counts:` met daaronder `if w not in known:` en `unknown[w] = counts[w]`. Na de loop: `return unknown`.",
    )
    return


@app.cell(hide_code=True)
def text_after_unknown():
    mo.md(r"""
    In het Fries en het Gronings zijn de woorden die de lijst het vaakst mist klein en veelvoorkomend: Fries *it*, *fan* en *yn*, waar het Nederlands *het*, *van* en *in* heeft, en in het Gronings onder meer *n*, *nait* en *veur*, voor *een*, *niet* en *voor*. In het Gronings maken de 20 onbekende woorden die het vaakst voorkomen 24,8% uit van alle onbekende woorden in de zinnen: met die 20 woorden erbij zou de lijst een kwart minder woorden missen. Aan de Nederlandse kant mist de lijst namen zoals *Sami*, en alledaagse woorden zoals *lawaai*, *zussen* en *fles*, en 71% van die woorden komt maar één keer voor. De treebank kent zelfs de woorden *gronings* en *dialect* niet.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: zin voor zin

    Kies een variëteit, en waar je begint. De afbeelding toont zes zinnen van Tatoeba met de Nederlandse vertaling, met de woorden die de woordenlijst van de treebank niet kent onderstreept, en telt wat de lijst mist in die zes en in alle paren. Kijk welke woorden een streep krijgen in de Nederlandse vertalingen, en welke in de variëteit.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_variety = mo.ui.dropdown(options=C.variety_options(L.lang), value=list(C.variety_options(L.lang))[0], label="Variëteit")
    demo_variety
    return (demo_variety,)


@app.cell(hide_code=True)
def demo_slider(demo_variety):
    demo_start = mo.ui.slider(start=1, stop=C.last_start(demo_variety.value), step=1, value=1, label="Begin bij paar", full_width=True)
    demo_start
    return (demo_start,)


@app.cell(hide_code=True)
def demo_look(demo_start, demo_variety):
    mo.Html(C.demo_picture(demo_variety.value, demo_start.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In het Gronings komen dezelfde kleine woorden in paar na paar terug, terwijl de Nederlandse vertalingen maar af en toe een streep krijgen, bij een naam of een woord dat toevallig niet in de treebank staat. Sommige Friese paren zijn spreekwoorden met een ander spreekwoord als vertaling, zoals *In keap is in keap* naast *Gedane zaken nemen geen keer*: een paar van Tatoeba zegt hetzelfde, maar niet altijd met dezelfde woorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Regio, groep of situatie?

    Elke vetgedrukte uitspraak gaat over een verschil in hoe mensen spreken. Zet die bij waar het verschil mee samengaat.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_variation = L.sort("region-group-situation")
    sort_variation
    return (sort_variation,)


@app.cell(hide_code=True)
def check_sort(sort_variation):
    result_sort = L.check("region-group-situation", sort_variation)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Zo'n 7.000 talen](/course/open/nlp-languages-of-the-world?lang=nl) telde talen en families; deze les keek binnen één taal. Niemand spreekt een taal op maar één manier: variëteiten horen bij een regio, een sociale groep of een situatie, en de standaardtaal is er één van, met een bijzondere plaats in het openbare leven. De standaardtaal verschilt een beetje tussen Nederland en België, Vlaanderen heeft tussentaal tussen dialect en standaardtaal, en het Fries, het Nedersaksisch en het Limburgs ernaast; of iets als taal of als dialect telt, is een beslissing van mensen. Je maakte een woordenlijst van een treebank in standaardnederlands, rekende uit hoeveel die lijst mist van drie variëteiten tegenover de eigen vertalingen, en telde welke woorden de lijst mist: in het Fries en het Gronings het vaakst kleine, veelvoorkomende woorden, in het Nederlands namen en alledaagse woorden die de treebank nooit tegenkwam. Een tool die op één variëteit gebouwd is, werkt minder goed op de andere, dus schrijf op in welke variëteit je data is: een van de vragen die Jurafsky en Martin noemen om een corpus te beschrijven. De volgende les van het domein gaat over hoe talen van elkaar lenen en door de tijd veranderen: *Talen ontmoeten elkaar en veranderen*.

    **Een voorbeeld uit het dagelijks leven.** Als een spraakassistent of een vertaalapp je woorden verkeerd begrijpt, vraag je dan af of de data genoeg tekst bevatte van mensen die spreken zoals jij.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_counts, result_share, result_sort, result_unknown):
    L.wrap_up(result_counts, result_share, result_unknown, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
