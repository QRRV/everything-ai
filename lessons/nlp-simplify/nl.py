# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-simplify. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-simplify", "nl")
    C = L.checks   # de manier van knippen van de course, de brieven over de groene container, de boeken, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl), en één over [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sentences():
    quick_sentences = L.choice(
        "quick-sentences",
        question="In *Tekst in zinnen en woorden knippen* zegt `ends_sentence(word, abbreviations)` of een woord, geknipt bij spaties, een zin afsluit. De tekst *Dhr. Jansen lachte. Els knikte.* wordt bij spaties geknipt, en `\"dhr.\"` staat in de lijst. Bij hoeveel woorden van de tekst geeft `ends_sentence` `True`?",
        options={
            "three": "3: *Dhr.*, *lachte.* en *knikte.* eindigen elk op een punt",
            "right": "2: *lachte.* en *knikte.*",
            "one": "1: alleen *knikte.*, het laatste woord van de tekst",
            "none": "Geen: de functie heeft het volgende woord nodig om te beslissen",
        },
        correct="right",
        why={
            "right": "*Dhr.* staat in de lijst, dus `ends_sentence` geeft `False`; *lachte.* en *knikte.* eindigen op een punt en zijn geen afkortingen. Deze les knipt zinnen op dezelfde manier, en de lijst doet ertoe: zonder de lijst zou elke *Mr.* in *Pride and Prejudice* een zin afsluiten.",
            "three": "*Dhr.* eindigt op een punt, maar `ends_sentence` zoekt het woord eerst op in de lijst met afkortingen. Wat geeft de functie bij een woord uit de lijst? Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "one": "*lachte.* eindigt ook op een punt, en staat niet in de lijst. Kijkt `ends_sentence` naar de plaats van een woord in de tekst? Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
            "none": "`ends_sentence` krijgt één woord en de lijst, verder niets, en beslist op grond daarvan. Wat geeft de functie bij *knikte.*? Zie [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl).",
        },
    )
    quick_sentences
    return (quick_sentences,)


@app.cell(hide_code=True)
def quick_check_findall():
    quick_findall = L.choice(
        "quick-findall",
        question='In *Patronen vinden: reguliere expressies* zocht je met `re.findall`. Wat geeft `re.findall(r"\\d+", "Kamer 12 op verdieping 3")`?',
        options={
            "digits": '`["1", "2", "3"]`: één match voor elk cijfer van de tekst',
            "right": '`["12", "3"]`',
            "numbers": "`[12, 3]`, twee gehele getallen",
            "one": '`["12 op verdieping 3"]`: van het eerste cijfer tot het laatste',
        },
        correct="right",
        why={
            "right": "`\\d+` pakt een reeks cijfers zo lang mogelijk: *12*, dan *3*, en `re.findall` geeft elke match als string. Oefening 1 van deze les vindt op dezelfde manier reeksen klinkers.",
            "digits": "Zonder `+` zou elk cijfer een eigen match zijn. Wat voegt `+` toe aan `\\d`? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "numbers": "`re.findall` geeft de matches als stukken van de tekst. Zijn dat getallen of strings? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
            "one": "`\\d` matcht alleen cijfers, en *op verdieping* heeft er geen. Kan één match de letters tussen 12 en 3 bevatten? Zie [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl).",
        },
    )
    quick_findall
    return (quick_findall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tekst kan kloppen en toch te moeilijk zijn voor de mensen voor wie de tekst bedoeld is. In 1948 publiceerde Rudolf Flesch een manier om in een getal uit te drukken hoe moeilijk een tekst te lezen is: RE = 206,835 − 0,846 wl − 1,015 sl. Daarin is *wl* de lengte van de woorden, in lettergrepen per 100 woorden, en *sl* de lengte van de zinnen, in woorden; de score ligt volgens Flesch tussen 0, vrijwel onleesbaar, en 100, makkelijk voor iedereen die kan lezen. Flesch noemt de formule Formula A, een maat voor lengte: hoe langer de woorden en de zinnen, hoe moeilijker de tekst te lezen is.

    Ook overheden letten op leesbare teksten. De richtlijnen voor de websites van de Rijksoverheid zeggen: "Nieuwe teksten voor de website Rijksoverheid.nl schrijft de redactie op B1-niveau." En: "De overgrote meerderheid van de bevolking begrijpt teksten op taalniveau B1." In 2025 meldde het Centraal Bureau voor de Statistiek (CBS) dat ongeveer drie miljoen volwassenen van 16 tot 75 jaar in Nederland laaggeletterd zijn.

    In deze les bouw je de formule van Flesch uit de onderdelen: je telt lettergrepen, woorden en zinnen, en je scoort acht boeken, de Nederlandse zinnen van nu en twee versies van een brief. Daarna ontdek je wat zo'n formule niet kan zien.

    **Een voorbeeld uit het dagelijks leven.** Een brief van de gemeente legt in één zin van 53 woorden uit wanneer de groene container geleegd wordt. De ambtenaar die de brief schreef, vindt de brief duidelijk; een buur die nog Nederlands leert, leest de brief drie keer en zet de container op de verkeerde dag buiten.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_harder = L.guess(
        "guess",
        question="*Pride and Prejudice* (1813) en *Moby-Dick* (1851) zijn twee van de acht boeken van de course. Tel van elk heel boek de woorden per zin en de lettergrepen per woord, en vul de twee gemiddelden in de formule van Flesch in. Welk van de twee boeken krijgt de laagste score, en is dus het moeilijkst te lezen?",
        options={
            "moby-far": "*Moby-Dick*, met grote afstand: de hoofdstukken over walvissen en schepen staan vol lange, zeldzame woorden",
            "moby": "*Moby-Dick*, met een paar punten",
            "right": "*Pride and Prejudice*",
            "same": "Allebei precies dezelfde score",
        },
        reveal="*Pride and Prejudice*: 53,1, tegen 56,9 voor *Moby-Dick*, allebei *vrij moeilijk* in de tabel van Flesch. De zinnen van *Pride and Prejudice* zijn iets korter, gemiddeld 20,8 woorden tegen 21,2, maar de woorden zijn langer: 1,57 lettergrepen per woord tegen 1,52, in woorden als *acquaintance* en *civility*. In de formule kost 0,05 lettergreep per woord meer 84,6 × 0,05, ongeveer 4 punten, terwijl 0,4 woord per zin meer slechts 1,015 × 0,4 kost, ongeveer 0,4 punt.",
    )
    guess_harder
    return (guess_harder,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **leesbaarheidsformule** schat hoe moeilijk een tekst te lezen is uit dingen die een programma kan tellen. Het **leesgemak** van Flesch gebruikt twee gemiddelden: de woorden per zin en de lettergrepen per woord. Per woord geschreven in plaats van per 100 woorden is de formule

    $$\text{leesgemak} = 206{,}835 - 1{,}015 \cdot \frac{\text{woorden}}{\text{zinnen}} - 84{,}6 \cdot \frac{\text{lettergrepen}}{\text{woorden}}$$

    Hoe langer de zinnen en de woorden, hoe lager de score. Flesch gaf een tabel met wat de scores betekenen, met het soort tijdschrift dat meestal zo scoort:

    | Score | Leesgemak | Typisch tijdschrift |
    |---|---|---|
    | 90 tot 100 | zeer gemakkelijk | stripbladen |
    | 80 tot 90 | gemakkelijk | goedkope romanbladen |
    | 70 tot 80 | vrij gemakkelijk | glossy bladen met verhalen |
    | 60 tot 70 | standaard | digests, bladen met ingekorte stukken |
    | 50 tot 60 | vrij moeilijk | kwaliteitsbladen |
    | 30 tot 50 | moeilijk | academische bladen |
    | 0 tot 30 | zeer moeilijk | wetenschappelijke bladen |

    Niets houdt een score tussen 0 en 100: een tekst met woorden van één lettergreep in zinnen van één woord scoort 206,835 − 1,015 − 84,6 = 121,22, en een heel moeilijke tekst scoort onder 0.

    1. **Nederlands.** Flesch maakte de formule voor het Engels. In 1960 paste W. H. Douma van de Landbouwhogeschool in Wageningen, nu Wageningen University, de formule aan voor Nederlandse landbouwbladen. Douma vergeleek Engelse teksten met de Nederlandse vertaling ervan, vond het Nederlands op alle niveaus langer, en ging ervan uit dat Nederlandse woorden en zinnen allebei ongeveer 10% langer zijn: "Laten we echter stellen, dat deze voor beide elementen 10% bedraagt." De factoren van Flesch, afgerond op 0,85 en 1,02 en gedeeld door 1,1, geven de factoren van Douma: "RE = 206,84 - 0,77 x woordlengte - 0,93 x zinslengte." Per woord geschreven, met 77 in plaats van 0,77, is dat de formule **Flesch-Douma**. De tabel van Douma heeft dezelfde zeven niveaus als de tabel van Flesch, alleen loopt *vrij moeilijk* er van 45 tot 60 en *moeilijk* van 30 tot 45; Douma schrijft *tamelijk* waar deze les *vrij* schrijft.

    $$\text{Flesch-Douma} = 206{,}84 - 0{,}93 \cdot \frac{\text{woorden}}{\text{zinnen}} - 77 \cdot \frac{\text{lettergrepen}}{\text{woorden}}$$

    2. **Lettergrepen.** Een programma hoort geen lettergrepen, dus deze les telt lettergrepen met een regel: elke groep klinkers op een rij, a, e, i, o, u of y, is één lettergreep, en elk woord heeft er minstens één. *Simple* heeft twee groepen, *i* en *e*; *leesbaarheid* heeft er drie, *ee*, *aa* en *ei*. De regel mist een stomme *e*, zoals in het Engelse *make*; twee klinkers op een rij die bij twee lettergrepen horen, zoals in *museum*; en de *ë* van *ideeën*, die een nieuwe lettergreep begint maar niet bij de zes letters hoort.
    3. **Zinnen en woorden.** De course knipt een tekst in zinnen zoals `ends_sentence` van *Tekst in zinnen en woorden knippen* het doet: een woord, geknipt bij spaties, sluit een zin af als het woord op `.`, `?` of `!` eindigt, tenzij het een afkorting of een initiaal is. De course slaat een afsluitend aanhalingsteken of haakje over, en voegt Engelse afkortingen zoals *Mrs.* aan de lijst toe. De woorden van een zin zijn de reeksen letters, dus *Mr.* wordt het woord *Mr*.
    4. **Vereenvoudigen.** Advaith Siddharthan, 2014, noemt **text simplification** in enge zin het verminderen van de taalkundige complexiteit van een tekst, met behoud van de oorspronkelijke informatie en betekenis. Met de hand betekent dat: kortere zinnen, gewone woorden in plaats van zeldzame, en de hoofdzaak eerst. Het Europees Referentiekader voor talen (ERK, in het Engels CEFR) van de Raad van Europa beschrijft zes niveaus, van A1 tot C2, en wat een lezer op niveau B1 kan: de hoofdpunten begrijpen van duidelijke standaardtaal over vertrouwde zaken die regelmatig voorkomen op het werk, op school en in de vrije tijd.
    5. **Wat een formule niet ziet.** Een formule ziet de lengte van woorden en zinnen, verder niets. Bruce, Rubin en Starr schreven in 1981, in een rapport voor het Center for the Study of Reading van de University of Illinois, dat de meeste formules alleen de lengte van zinnen en de moeilijkheid van woorden meetellen, en andere dingen die een tekst moeilijk maken, zoals de samenhang van de tekst, hoogstens indirect zien. Een tekst die herschreven is om beter te scoren, met de lange zinnen geknipt en zonder de woorden die de zinnen verbonden, kan moeilijker worden: de aangepaste tekst kan volgens Bruce, Rubin en Starr zelfs moeilijker zijn dan het origineel. Een eenvoudigere tekst kan ook informatie kwijt zijn. En de voorbeelden waarvan programma's leren vereenvoudigen, zijn vaak niet eenvoudiger: Wei Xu, Chris Callison-Burch en Courtney Napoles bekeken paren zinnen uit de Engelse Wikipedia en Simple English Wikipedia waarmee onderzoekers zulke programma's trainden, en schreven in 2015 dat ongeveer 50% van de paren in het corpus PWKP geen eenvoudigere versie van de zin is. In het Nederlands kijkt LiNT, gemaakt aan de Universiteit Utrecht, naar meer dan lengte: hoe gewoon de woorden zijn, hoeveel inhoudswoorden er bij elk werkwoord komen, hoe concreet de zelfstandige naamwoorden zijn, en hoe ver woorden die bij elkaar horen uit elkaar staan.

    Het plaatje laat een brief over de groene container twee keer zien: in ambtelijke taal en in eenvoudige taal, elk gescoord met de formule van Douma.
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
    | **leesbaarheid** | hoe makkelijk een tekst te lezen en te begrijpen is |
    | **leesbaarheidsformule** | een formule die de leesbaarheid schat uit dingen die een programma kan tellen, zoals de lengte van zinnen en woorden |
    | **leesgemak** | de leesbaarheidsformule van Flesch (1948): 206,835, min 1,015 keer de woorden per zin, min 84,6 keer de lettergrepen per woord |
    | **Flesch-Douma** | de versie van Douma voor het Nederlands (1960), met kleinere factoren: 0,93 en 77 |
    | **lettergreep** | een deel van een woord dat je in één tel zegt, met één klinkerklank als kern: *lees* en *baar* in *leesbaar* |
    | **text simplification** | een tekst herschrijven zodat de tekst makkelijker te lezen is, met behoud van de informatie en de betekenis |
    | **taalniveau B1** | het derde van de zes niveaus van het ERK, van A1 tot C2: het niveau van een lezer die de hoofdpunten begrijpt van duidelijke standaardtaal over vertrouwde zaken |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te scoren hoe leesbaar een tekst is:

    1. Knip de tekst in zinnen, en elke zin in woorden.
    2. Tel de lettergrepen van elk woord: de groepen klinkers, minstens één.
    3. Tel de woorden, de zinnen en de lettergrepen van de hele tekst.
    4. Deel: de woorden per zin, en de lettergrepen per woord.
    5. Vul de formule in: de formule van Flesch voor Engels, de formule van Douma voor Nederlands. Zoek de score op in de tabel.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem de brief in eenvoudige taal, in de taal van deze les: `C.LETTERS` bevat beide brieven in beide talen. `C.sentences` knipt een tekst in zinnen, elk een lijst met de woorden, en `C.syllables` telt de lettergrepen van één woord volgens de regel van deze les. De cel print voor elke zin het aantal woorden en de lettergrepen van elk woord:
    """)
    return


@app.cell
def example_cell():
    _letter = C.LETTERS[L.lang]["plain"]
    for _sentence in C.sentences(_letter):
        print(len(_sentence), [C.syllables(_w) for _w in _sentence])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De brief heeft 13 + 9 + 4 + 7 = 33 woorden in 4 zinnen, en de lettergrepen van de woorden tellen op tot 51: 8,25 woorden per zin en ongeveer 1,55 lettergrepen per woord. Flesch-Douma is 206,84 − 0,93 × 33 / 4 − 77 × 51 / 33, ongeveer 80,2: gemakkelijk. De regel geeft *servicebalie* vijf lettergrepen, waar een lezer er vier hoort: de stomme *e* van het Engelse leenwoord *service*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_syllables():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. syllables: groepen klinkers

        Schrijf `syllables`, die `word` krijgt, één woord, en het aantal lettergrepen teruggeeft volgens de regel van deze les: de groepen klinkers op een rij, a, e, i, o, u of y, in het woord in kleine letters, en minstens 1. In de startcode moet `groups` de groepen bevatten die `re.findall` vindt. Vervang de twee `...`.

        De check roept `syllables("simple")` aan, met een woord van de Engelse les, wat `2` geeft, en `syllables("leesbaarheid")`, wat `3` geeft. Daarna eigen gevallen: een woord zonder klinkers, een woord met een hoofdletter, een woord met een y, twee klinkers op een rij, en woorden waarbij de regel en een lezer het oneens zijn, zoals *make*.
        """),
        L.task("Tel de groepen klinkers in een woord, en geef minstens 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_syllables():
    box_syllables = L.code_box("syllables", lines=4)
    box_syllables
    return (box_syllables,)


@app.cell(hide_code=True)
def check_syllables(box_syllables):
    result_syllables = L.check("syllables", box_syllables.value)
    result_syllables
    return (result_syllables,)


@app.cell(hide_code=True)
def hints_syllables():
    L.hints(
        "syllables",
        [
            "`re.findall(pattern, text)` geeft elke match van een patroon in een tekst als lijst, en de tekenset `[aeiouy]` matcht één van de zes klinkers. `word.lower()` geeft het woord in kleine letters. `max` geeft het grootste van de getallen die de functie krijgt: `max(1, 0)` is 1.",
            "Met `+` na de tekenset is één match een hele groep klinkers op een rij, dus de *oo* van *boot* is één groep. Het aantal lettergrepen is hoeveel groepen de lijst bevat, maar nooit minder dan 1.",
            "`    return max(1, ___)`",
        ],
        why='`groups = re.findall(r"[aeiouy]+", word.lower())` en `return max(1, len(groups))`.',
    )
    return


@app.cell(hide_code=True)
def text_after_syllables():
    mo.md(r"""
    De regel telt de meeste woorden van het plaatje goed. De regel mist een stomme *e* (*make*, *service*), twee klinkers op een rij die bij twee lettergrepen horen (*created*, *museum*, *theater*, *bioscoop*), een lettergreep zonder klinkerletter (*rhythm*), en de *ë* van *ideeën*. De formule gebruikt alleen het gemiddelde over alle woorden van een tekst, en lange woorden met veel groepen klinkers krijgen nog steeds veel lettergrepen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. counts: woorden, zinnen en lettergrepen

        Schrijf `counts`, die `sentences` krijgt, een tekst die in zinnen is geknipt, elk een lijst met woorden zoals `C.sentences` geeft, en drie getallen teruggeeft: hoeveel woorden, hoeveel zinnen en hoeveel lettergrepen de tekst heeft, in deze volgorde. De check geeft je functie de `syllables` van de course. Vervang de twee `...`: de eerste door twee regels die het woord en de lettergrepen ervan tellen, en de `return` door de drie getallen.

        De check roept `counts([["We", "start", "on", "Monday"], ["Call", "us", "today"]])` aan, met de woorden van de Engelse les, wat `(7, 2, 9)` geeft, en `counts([["Bel", "ons"], ["We", "beginnen", "op", "maandag"]])`, wat `(6, 2, 9)` geeft. Daarna eigen gevallen: geen zinnen, een lege zin, een woord zonder klinkers, en de vier brieven over de groene container.
        """),
        L.task("Tel de woorden, de zinnen en de lettergrepen van een tekst die in zinnen is geknipt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("counts", lines=9)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "counts",
        [
            "`n_words` en `n_syllables` beginnen bij 0 en groeien met één woord tegelijk, zoals een lopend totaal. `syllables` is de functie van de course: `syllables(\"maandag\")` is 2. `len(sentences)` is het aantal zinnen.",
            "In de binnenste loop telt elk woord 1 op bij de woorden en de lettergrepen ervan bij de lettergrepen. Geef na allebei de loops de drie getallen terug, met komma's ertussen: de woorden, de zinnen, de lettergrepen.",
            "`            n_syllables = n_syllables + ___`",
        ],
        why="`n_words = n_words + 1`, `n_syllables = n_syllables + syllables(word)` en `return n_words, len(sentences), n_syllables`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    De eerste zin van de brief in ambtelijke taal heeft 53 woorden; de langste zin van de eenvoudige brief heeft er 13. Zinnen tellen is minder simpel dan het lijkt. Volgens de telling van de course heeft *Pride and Prejudice* 127.858 woorden in 6.156 zinnen. Zonder de lijst met afkortingen zou de course 7.310 zinnen vinden: de afkortingen van de lijst komen 1.154 keer voor in het boek, *Mr.* en *Mrs.* alleen al 1.135 keer, en elke keer zou er een zin eindigen. Het aantal woorden per zin zou dalen van 20,8 naar 17,5, en de score van Flesch zou stijgen van 53,1 naar 56,4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flesch():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. flesch: het leesgemak

        Schrijf `flesch`, die `words`, `sentences` en `syllables` krijgt, de drie getallen die `counts` geeft, en het leesgemak van Flesch teruggeeft, zonder af te ronden. Vervang de `...`.

        De check roept `flesch(31, 4, 41)` aan, met de tellingen van de eenvoudige brief in het Engels, wat ongeveer `87.08` geeft, en `flesch(33, 4, 51)`, met de tellingen van de eenvoudige brief in het Nederlands, wat ongeveer `67.72` geeft. Daarna eigen gevallen: een tekst met woorden van één lettergreep in zinnen van één woord, de twee brieven in ambtelijke taal, en drie van de boeken.
        """),
        L.task("Vul de formule van Flesch in met de woorden, de zinnen en de lettergrepen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flesch():
    box_flesch = L.code_box("flesch", lines=3)
    box_flesch
    return (box_flesch,)


@app.cell(hide_code=True)
def check_flesch(box_flesch):
    result_flesch = L.check("flesch", box_flesch.value)
    result_flesch
    return (result_flesch,)


@app.cell(hide_code=True)
def hints_flesch():
    L.hints(
        "flesch",
        [
            "`words / sentences` is het aantal woorden per zin. De formule begint bij 206,835 en haalt er twee stukken af: 1,015 keer de woorden per zin, en 84,6 keer het andere gemiddelde. `return` geeft de score terug.",
            "Het andere gemiddelde is het aantal lettergrepen per woord: het aantal lettergrepen gedeeld door het aantal woorden. Python doet `*` en `/` eerder dan `-`, dus de formule heeft geen haakjes nodig.",
            "`    return 206.835 - 1.015 * words / sentences - 84.6 * ___`",
        ],
        why="`return 206.835 - 1.015 * words / sentences - 84.6 * syllables / words`.",
    )
    return


@app.cell(hide_code=True)
def text_after_flesch():
    mo.md(r"""
    Volgens de tabel van Flesch is *Alice's Adventures in Wonderland* vrij gemakkelijk, met 70,9, en is *Frankenstein* het moeilijkste van de Engelse boeken, met 50,3: de zinnen zijn het langst, gemiddeld 22,6 woorden. Bij deze acht boeken verschillen de boeken onderling meer dan de talen: *Eline Vere* heeft 14,0 woorden per zin, minder dan elk Engels boek, en zelfs de formule van Flesch geeft het boek 64,7, meer dan *Pride and Prejudice*, *Frankenstein* of *Moby-Dick*. Met de kleinere factoren van Douma krijgt elke tekst een hogere score dan met de factoren van Flesch: 77,4 voor *Eline Vere*. De Nederlandse zinnen van Tatoeba, door vrijwilligers geschreven in het Nederlands van nu, hebben gemiddeld 6,3 woorden en scoren 88,9 met de formule van Douma: gemakkelijk.

    ### Probeer het: een eigen tekst

    Het vak begint met de brief in ambtelijke taal. Herschrijf de brief in eenvoudige taal, of plak een eigen tekst, en kijk hoe de scores veranderen. Probeer eerst de brief moeilijker te volgen te maken zonder dat de score daalt. Schrijf daarna een versie die een buur begrijpt die nog Nederlands leert, en laat niets weg.
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.LETTERS[L.lang]["hard"], label="Je tekst", rows=6, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ### Oefening 4. Wat ziet de formule?

    De brief *Vanaf volgende maand legen we de groene container eens in de twee weken, en we beginnen op de eerste maandag. Op feestdagen legen we de container niet. Heeft u een vraag? Bel dan de servicebalie van de gemeente.* scoort 74,4 met de formule van Douma. Elke verandering hieronder maakt een nieuwe brief. Zet elke verandering bij wat de verandering met de score doet.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Tekst in zinnen en woorden knippen](/course/open/nlp-tokenize?lang=nl) vond waar zinnen eindigen, ondanks afkortingen, en deze les rekent erop: zonder de lijst krijgt *Pride and Prejudice* meer dan duizend zinnen te veel. [Patronen vinden: reguliere expressies](/course/open/nlp-regex?lang=nl) gaf de tekenset met `+` die de groepen klinkers vindt. [Een samenvatting scoren: ROUGE](/course/open/nlp-rouge?lang=nl) scoorde een tekst tegen referenties die mensen schreven; een leesbaarheidsformule heeft geen enkele referentie nodig, en ziet nog minder: de lengte van woorden en zinnen. De volgende les bouwt een programma dat mensen antwoord geeft met niets dan regels.

    **Een voorbeeld uit het dagelijks leven.** Sommige tekstverwerkers laten de score van Flesch van een document zien. De score kan de schrijver vertellen wanneer de zinnen lang zijn; de score kan niet zien of het document nog zegt wat de schrijver bedoelt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_syllables, result_counts, result_flesch, result_changes):
    L.wrap_up(result_syllables, result_counts, result_flesch, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
