# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-language-id. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-language-id", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl), en een over [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bigrams():
    quick_bigrams = L.choice(
        "quick-bigrams",
        question="Hoeveel bigrams telt `count_bigrams` in *banana*, herhalingen meegeteld?",
        options={
            "three": "3: *ba*, *an* en *na*",
            "right": "5",
            "six": "6: één voor elke letter van het woord",
        },
        correct="right",
        why={
            "right": "*banana* heeft zes letters, en `count_bigrams` neemt een paar op elke plaats waar nog een volgende letter komt: *ba*, *an*, *na*, *an*, *na*. Twee daarvan komen twee keer voor, en de aantallen houden beide bij. Deze les knipt een tekst op dezelfde manier in rijtjes van drie tekens, met een spatie voor en achter de tekst, zodat ook de eerste en de laatste letter eigen rijtjes krijgen.",
            "three": "Dat zijn de verschillende bigrams. Met herhalingen zijn het er vijf: *ba*, *an*, *na*, *an*, *na*, en `count_bigrams` telt *an* en *na* twee keer. Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
            "six": "Een bigram heeft een volgende letter nodig, en na de laatste *a* komt niets meer: zes letters geven vijf bigrams. Zie [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl).",
        },
    )
    quick_bigrams
    return (quick_bigrams,)


@app.cell(hide_code=True)
def quick_check_lower():
    quick_lower = L.choice(
        "quick-lower",
        question="*Huis* en *huis* zijn hetzelfde woord. Wat gebeurt er als je een tekst in kleine letters zet voordat je de woorden telt?",
        options={
            "drop": "Elk woord met een hoofdletter valt weg",
            "nothing": "Niets: dat doet Python al",
            "right": "*Huis* en *huis* tellen als één woord",
        },
        correct="right",
        why={
            "right": "Voor Python zijn *H* en *h* verschillende tekens, dus zonder kleine letters hebben de aantallen twee keys voor één woord. Deze les zet tekst ook in kleine letters: een hoofdletter aan het begin van een zin hoort geen eigen rijtjes letters te maken.",
            "drop": "Kleine letters veranderen letters, niet welke woorden blijven: *Huis* wordt *huis*. Woorden weglaten doet een lijst met stop words. Zie [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
            "nothing": "Voor Python geeft `\"Huis\" == \"huis\"` de waarde `False`: een hoofdletter is een ander teken. Zie [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
        },
    )
    quick_lower
    return (quick_lower,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voordat een programma woorden kan opzoeken, spelling kan verbeteren of een tekst kan vertalen, moet het programma weten in welke taal de tekst staat. Tommi Jauhiainen, Marco Lui, Marcos Zampieri, Timothy Baldwin en Krister Lindén beginnen hun overzicht van het vakgebied, uit 2019, met een definitie: language identification is bepalen in welke natuurlijke taal een document, of een deel ervan, geschreven is. Zoekmachines gebruiken language identification om pagina's in te delen, vertaaldiensten om een richting te kiezen, en wie tekst van het web verzamelt om language models te trainen, gebruikt language identification om elke pagina op de stapel van de juiste taal te leggen. Deze les bouwt zo'n programma in eigen code, naar de methode die William Cavnar en John Trenkle in 1994 publiceerden: tel de korte lettervolgordes van een tekst, en vergelijk de aantallen met de lettervolgordes van elke taal. Dezelfde aantallen laten ook tekst zien die helemaal geen taal is, zoals een regel HTML.

    **Een voorbeeld uit het dagelijks leven.** Een browser die aanbiedt een pagina uit het Italiaans te vertalen, heeft de taal van de pagina uit de tekst geraden.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Het programma van deze les leert twaalf talen uit korte zinnen van Tatoeba, een verzameling zinnen die vrijwilligers schrijven en vertalen: Nederlands, Afrikaans, Fries, Duits, Engels, Deens, Noors Bokmål (een van de twee schrijftalen van het Noors), Zweeds, Frans, Spaans, Portugees en Italiaans. Daarna krijgt het programma 1.190 zinnen die het niet eerder zag. Welke twee talen haalt het programma het vaakst door elkaar?",
        options={
            "afrikaans": "Nederlands en Afrikaans",
            "right": "Deens en Bokmål",
            "romance": "Spaans en Portugees",
            "frisian": "Nederlands en Fries",
        },
        reveal="**Deens en Bokmål**: 57 van de 192 zinnen die een verkeerde taal krijgen. 31 Deense zinnen komen eruit als Bokmål en 26 zinnen in het Bokmål als Deens, dus maar 54 van de 100 Deense zinnen en 52 van de 90 zinnen in het Bokmål komen er goed uit. Op papier lijken de twee talen veel op elkaar: beide hebben *jeg* voor *ik* en *er* voor *is*. Spaans en Portugees haalt het programma 11 keer door elkaar, Nederlands en Afrikaans 8 keer. Oefening 3 laat je eigen programma los op dezelfde 1.190 zinnen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Lettervolgordes verraden een taal

    Cavnar en Trenkle definiëren in de paragraaf *N-Grams* van hun paper een n-gram als een stukje van n tekens uit een langere string. Een **character n-gram** is een rijtje van n tekens uit een tekst, spaties en leestekens meegeteld, en een **trigram** is een rijtje van drie tekens. *de kat* bevat het trigram *kat*, maar ook *de␣* en *e␣k*, waarin ␣ voor een spatie staat. Cavnar en Trenkle zetten ook spaties om de tekst heen, zodat de n-grams laten zien waar een woord begint en waar een woord eindigt. Hun paper doet dat om elk woord, en gooit cijfers en leestekens weg; deze les houdt het eenvoudiger, met één spatie voor de hele tekst en één erachter, en elk teken blijft staan. *de kat* geeft dan zes trigrams, van *␣de* tot *at␣*: *␣de* zegt dat een woord met *de* begint, en *at␣* dat een woord op *at* eindigt.

    Sommige trigrams komen in de ene taal veel vaker voor dan in de andere. De afbeelding laat de tien vaakste trigrams van vier talen zien, geteld in de zinnen van de course. Het Nederlands heeft *␣de* en *en␣*, het Afrikaans *␣di* en *ie␣*, het begin en het eind van *die*, het Afrikaanse woord voor *de* en *het*. Het Duits heeft *ich* en *sch*, en het Engels *␣th*, *the* en *he␣*: het begin, het hele woord en het eind van *the*.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Een profiel

    Tel elk trigram in de zinnen van een taal, en zet de trigrams op volgorde, de vaakste bovenaan. Het **profiel** van de taal is het begin van die lijst: elk trigram met de **rang**, 0 voor de vaakste, dan 1, 2, enzovoort. Het profiel van een tekst maak je op dezelfde manier, uit alleen die tekst. Hoe lang moet een profiel zijn? Cavnar en Trenkle zagen dat de ongeveer 300 vaakste n-grams bijna altijd sterk met de taal samenhangen, en voegen eraan toe dat rang 300 niets bijzonders is: dat getal vonden ze vooral door te kijken. Deze les houdt 300 trigrams per taal over, uit 400 zinnen per taal (364 voor het Bokmål, dat er minder heeft). Deze les plakt die zinnen aan elkaar tot één tekst, met een spatie ertussen, dus een paar trigrams bestaan uit het eind van de ene zin en het begin van de volgende, zoals *.␣i* in het Nederlands.

    ### Uit de plaats

    Om twee profielen te vergelijken, gebruiken Cavnar en Trenkle wat ze de out-of-place-maat noemen: hoe ver een n-gram in het ene profiel uit de plaats staat ten opzichte van de plaats in het andere profiel. Hun voorbeeld: het n-gram *ING* staat op rang 2 in het document, maar op rang 5 in de taal, dus 3 rangen uit de plaats. De **out-of-place-maat** telt die getallen op voor elk n-gram van het profiel van de tekst. Een n-gram dat de taal mist, krijgt een straf: een maximale waarde, schrijven Cavnar en Trenkle, zonder te zeggen welke. Deze les neemt de grootte van het profiel van de taal, 300: meer dan een trigram in het profiel ooit uit de plaats kan staan, want de rangen lopen van 0 tot 299.

    Een lange tekst heeft meer trigrams dan een korte, dus een grotere som. Deze les deelt de som door het hoogste dat de som kan zijn, het aantal trigrams van de tekst keer 300, en noemt dat de **afstand**: 0 als elk trigram van de tekst op dezelfde rang staat als in de taal, 1 als de taal alle trigrams mist.

    ### De dichtstbijzijnde taal, of geen

    Het programma meet de afstand tot elk van de twaalf talen en neemt de dichtstbijzijnde. Maar de dichtstbijzijnde taal ligt niet altijd dichtbij: voor een regel HTML liggen alle talen ver weg, en toch zou het programma er een noemen. Cavnar en Trenkle zagen dat in hun conclusie: met een genormaliseerde score kan het systeem een document afwijzen waarvoor geen enkel profiel goed past. Het overzicht van Jauhiainen en collega's zegt het in één regel: de eenvoudigste manier om een onbekende taal te herkennen, is een drempel. Deze les zet een **grens** van 0,9: een taal telt alleen als de afstand onder de grens ligt, en als geen taal dat doet, antwoordt het programma *?*.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **language identification** | bepalen in welke taal een tekst geschreven is |
        | **character n-gram** | een rijtje van n tekens uit een tekst, spaties en leestekens meegeteld |
        | **trigram** | een character n-gram van drie tekens, zoals *␣de*; ␣ is een spatie |
        | **profiel** | de vaakste n-grams van een taal of een tekst, elk met de rang |
        | **rang** | de plaats van een n-gram in een profiel: 0 voor de vaakste, dan 1, enzovoort |
        | **out-of-place-maat** | voor elk n-gram van de tekst hoe ver de rang ligt van de rang in de taal, of een straf als de taal het n-gram mist, alles bij elkaar opgeteld |
        | **afstand** | in deze les: die som gedeeld door het hoogste dat de som kan zijn, tussen 0 en 1 |
        | **grens** | de afstand vanaf waar het programma *?* antwoordt in plaats van een taal |
        | **mojibake** | tekst waarvan de letters verkeerd uitkwamen doordat de bytes met de verkeerde encoding gelezen zijn, zoals *cafÃ©* voor *café* |
        | `max(a, b)` | het grootste van twee getallen; `min(a, b)` geeft het kleinste |
        """),
        L.watch_out("Een afstand is geen kans. Een afstand van 0,67 tot het Nederlands betekent niet dat het programma voor 33% zeker is: de afstand zegt alleen hoe ver de trigrams van de tekst liggen van waar het Nederlands die trigrams zet. Het programma vergelijkt de afstanden met elkaar en met de grens, meer niet."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Maak één keer het profiel van elke taal: zet de zinnen in kleine letters, plak de zinnen aan elkaar met een spatie ertussen, knip die tekst met een spatie ervoor en erachter in trigrams, tel de trigrams, en houd de 300 vaakste over, met de rang.
    2. Maak op dezelfde manier het profiel van de tekst.
    3. Tel voor elk trigram van het profiel van de tekst op hoe ver de rang ligt van de rang in het profiel van de taal, of 300 als de taal het trigram mist.
    4. Deel de som door het aantal trigrams van de tekst keer 300: dat is de afstand tot de taal.
    5. Geef de dichtstbijzijnde taal terug, of *?* als geen afstand onder de grens ligt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder maakt met `profile` van de course het profiel van *Ik fiets naar huis.*, neemt het profiel van het Nederlands uit `language_profiles` van de course, dat de profielen van alle twaalf talen geeft, en zoekt een paar trigrams in allebei op. `profile` zet de tekst eerst in kleine letters.
    """)
    return


@app.cell
def example_cell():
    text = "Ik fiets naar huis."
    doc = C.profile(text)
    lang = C.language_profiles()["nl"]
    print(len(doc), len(lang))
    print(list(doc)[:6])
    print(doc["ik "], lang["ik "])
    print(doc["aar"], lang["aar"])
    print("uis" in lang)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De zin heeft 19 trigrams, elk één keer, dus het profiel zet de trigrams op alfabet: *␣fi* eerst, op rang 0. *ik␣*, het woord *ik*, heeft rang 10 in de zin en rang 8 in het Nederlands: 2 uit de plaats. *aar* heeft rang 4 in de zin en 19 in het Nederlands: 15 uit de plaats. *uis*, uit *huis*, staat niet in het profiel van het Nederlands en telt dus 300 op. Tien van de 19 trigrams ontbreken, samen 3.000; de andere negen tellen samen 841 op, dus de som is 3.841. Gedeeld door 19 × 300 = 5.700 geeft dat een afstand van 0,674 tot het Nederlands, de dichtstbijzijnde van de twaalf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ngrams():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. ngrams: rijtjes tekens

        Schrijf `ngrams(text, n)`: zet een spatie voor `text` en een erachter, en geef elk rijtje van `n` tekens daarvan terug, als een lijst, van links naar rechts, met herhalingen. Voor `ngrams("de kat", 3)` is de uitkomst `[" de", "de ", "e k", " ka", "kat", "at "]`. Vervang de `...`. De check roept dat aan en `ngrams("the cat", 3)`, en daarna eigen gevallen: teksten van één of twee letters, een lege tekst, een `n` van 1 en van 2, een letter die zich herhaalt, hoofdletters, een `n` van 4, en een `n` die langer is dan de tekst met de spaties.
        """),
        L.task("Knip een tekst in de character n-grams, met een spatie ervoor en erachter."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ngrams():
    box_ngrams = L.code_box("ngrams", lines=6)
    box_ngrams
    return (box_ngrams,)


@app.cell(hide_code=True)
def check_ngrams(box_ngrams):
    result_ngrams = L.check("ngrams", box_ngrams.value)
    result_ngrams
    return (result_ngrams,)


@app.cell(hide_code=True)
def hints_ngrams():
    L.hints(
        "ngrams",
        [
            "Begin met `padded = \" \" + text + \" \"` en `grams = []`. Ga langs de plaatsen met `for i in range(...)`, voeg op elke plaats de slice `padded[i:i + n]` toe, en geef na de loop `grams` terug.",
            "*de kat* heeft met de twee spaties 8 tekens. Op welke plaats begint het laatste rijtje van 3 tekens, en hoeveel plaatsen zijn dat, als je vanaf 0 telt?",
            r"`    for i in range(___):`",
        ],
        why=r"`padded = \" \" + text + \" \"` en `grams = []`; `for i in range(len(padded) - n + 1):` met daarin `grams.append(padded[i:i + n])`; na de loop `return grams`.",
    )
    return


@app.cell(hide_code=True)
def text_after_ngrams():
    mo.md(r"""
    Een `n` die langer is dan de tekst met de spaties, geeft een lege lijst: `range` krijgt dan een getal onder 1. In de afbeelding telde je functie ook de trigrams van 400 Nederlandse zinnen, en het begin van die telling is het begin van het profiel van het Nederlands. `profile` van de course doet hetzelfde met een eigen `ngrams`, nadat de tekst in kleine letters staat en elk rijtje spaties, tabs en regeleinden één spatie is geworden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_oop():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. out_of_place: hoe ver uit elkaar

        `profiles` bevat het profiel van elk van de twaalf talen, onder de code van de taal, zoals `"nl"` voor het Nederlands en `"nb"` voor het Bokmål; elk profiel is een dictionary van trigram naar rang. Schrijf `out_of_place(doc, lang)`, waarin `doc` het profiel van een tekst is en `lang` het profiel van een taal. Ga langs de trigrams van `doc`. Tel voor een trigram dat `lang` heeft op hoe ver de twee rangen uit elkaar liggen: de grootste min de kleinste. Tel voor een trigram dat `lang` mist `len(lang)` op. Geef de som terug.

        Voor de kleine profielen `{" de": 0, "de ": 1, "en ": 2, " ka": 3}` en `{"en ": 0, " de": 1, "de ": 2, "et ": 3, " he": 4}` is de uitkomst 9: *␣de* staat 1 uit de plaats, *de␣* 1 en *en␣* 2, en *␣ka* ontbreekt, wat 5 optelt. Vervang de `...`. De check roept dat aan en `out_of_place(profile("Ik fiets naar huis."), profiles["nl"])`, en daarna eigen gevallen: een Engelse zin tegen het Engels, een lege `doc`, twee keer hetzelfde profiel, profielen zonder iets gemeen, rangen die naar beide kanten uit de plaats staan, profielen van verschillende grootte, en een Nederlandse zin tegen het Italiaans.
        """),
        L.task("Meet hoe ver het profiel van een tekst van het profiel van een taal ligt."),
    ])
    return


@app.cell
def profiles_cell():
    profiles = C.language_profiles()
    profile = C.profile
    print(sorted(profiles))
    print(len(profiles["nl"]), profiles["nl"]["en "], profiles["en"][" th"])
    print(profile("de kat"))
    return profile, profiles


@app.cell(hide_code=True)
def text_after_profiles():
    mo.md(r"""
    De codes zijn de codes van twee letters uit ISO 639-1: *af* voor het Afrikaans, *fy* voor het Fries, *sv* voor het Zweeds, enzovoort. *en␣* heeft rang 0 in het Nederlands en *␣th* rang 0 in het Engels. Het profiel van *de kat* bevat de zes trigrams van *de kat*; elk trigram komt één keer voor, dus de rangen volgen het alfabet.
    """)
    return


@app.cell(hide_code=True)
def exercise_oop():
    box_oop = L.code_box("out-of-place", lines=8)
    box_oop
    return (box_oop,)


@app.cell(hide_code=True)
def check_oop(box_oop):
    result_oop = L.check("out-of-place", box_oop.value)
    result_oop
    return (result_oop,)


@app.cell(hide_code=True)
def hints_oop():
    L.hints(
        "out-of-place",
        [
            "Begin met `total = 0` en ga langs de trigrams met `for gram in doc:`. Vraag met `if gram in lang:` of de taal het trigram heeft; tel dan de grootste van de rangen `doc[gram]` en `lang[gram]` min de kleinste op, en tel anders `len(lang)` op. Geef na de loop `total` terug.",
            "Rang 3 en rang 5 liggen 2 uit elkaar, en rang 5 en rang 3 ook. Welke aanroep geeft het kleinste van twee getallen?",
            r"`            total = total + max(doc[gram], lang[gram]) - ___`",
        ],
        why=r"`total = 0`; `for gram in doc:` met daarin `if gram in lang:` met `total = total + max(doc[gram], lang[gram]) - min(doc[gram], lang[gram])`, en `else:` met `total = total + len(lang)`; na de loop `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_oop():
    mo.md(r"""
    Tegen het Italiaans is de som voor *Ik fiets naar huis.* 5.697: alleen *␣fi* staat in het profiel van het Italiaans, en de andere 18 trigrams tellen elk 300 op. De afbeelding laat zien wat elk trigram van de zin met je functie optelt, en deelt je som voor elk van de twaalf talen door het hoogste dat de som kan zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_identify():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. identify: de dichtstbijzijnde taal, of "?"

        Schrijf `identify(text, profiles, limit)`: maak met `profile` het profiel van `text`, meet de `distance` tot het profiel van elke taal in `profiles`, en geef de code van de dichtstbijzijnde taal terug, zoals `"nl"`. Als geen afstand onder `limit` ligt, geef dan `"?"` terug. De check geeft je functie `profile`, `distance` en `out_of_place` van de course; `distance(doc, lang)` deelt de som van `out_of_place` door `len(doc) * len(lang)`, en geeft `1.0` als `doc` leeg is. Voor `identify("Ik fiets naar huis.", profiles, 0.9)` is de uitkomst `"nl"`. Vervang de `...`. De check roept dat aan en `identify('<div class="nav"><a href="/home">Home</a></div>', profiles, 0.9)`, een regel HTML, die `"?"` moet geven. Daarna komen eigen gevallen: een Engelse zin, een lagere grens, een lege tekst, een Russische zin, een Nederlandse zin met kapotte letters, eigen kleine profielen waarin de dichtstbijzijnde taal als tweede komt of geen taal onder de grens ligt, en een grens van 1.
        """),
        L.task("Geef de dichtstbijzijnde taal van een tekst, of \"?\" als geen taal dichtbij genoeg komt."),
    ])
    return


@app.cell
def distance_cell(profile, profiles):
    distance = C.distance
    print(distance(profile("Ik fiets naar huis."), profiles["nl"]))
    print(distance(profile("Ik fiets naar huis."), profiles["it"]))
    print(distance(profile("Я люблю читать книги."), profiles["nl"]))
    return


@app.cell(hide_code=True)
def text_after_distance():
    mo.md(r"""
    De Nederlandse zin ligt op 0,674 van het Nederlands en op 0,999 van het Italiaans. Python schrijft getallen met een punt: 0.674. De Russische zin, *Ik lees graag boeken.*, ligt op precies 1 van het Nederlands: geen enkel trigram van de zin staat in het profiel, want geen van de twaalf talen wordt met Cyrillische letters geschreven.
    """)
    return


@app.cell(hide_code=True)
def exercise_identify():
    box_identify = L.code_box("identify", lines=10)
    box_identify
    return (box_identify,)


@app.cell(hide_code=True)
def check_identify(box_identify):
    result_identify = L.check("identify", box_identify.value)
    result_identify
    return (result_identify,)


@app.cell(hide_code=True)
def hints_identify():
    L.hints(
        "identify",
        [
            "Maak `doc = profile(text)`, begin met `best = \"?\"` en met `best_distance` op een waarde die alleen een taal onder de grens kan verslaan, en ga langs de codes met `for lang in profiles:`. Meet `d = distance(doc, profiles[lang])`; als `d` kleiner is dan `best_distance`, zet dan `lang` in `best` en `d` in `best_distance`. Geef na de loop `best` terug.",
            "Een taal op 0,95 mag nooit blijven als de grens 0,9 is, en een taal op 0,62 nooit als de grens 0,6 is. Bij welke beginwaarde van `best_distance` is `d < best_distance` onwaar voor beide, terwijl een taal net onder de grens er nog wel in komt?",
            r"`    best_distance = ___`",
        ],
        why=r"`doc = profile(text)`, `best = \"?\"` en `best_distance = limit`; `for lang in profiles:` met `d = distance(doc, profiles[lang])`, en `if d < best_distance:` met `best = lang` en `best_distance = d`; na de loop `return best`.",
    )
    return


@app.cell(hide_code=True)
def text_after_identify():
    mo.md(r"""
    ### Hoe goed is het programma?

    In de afbeelding geeft je programma de goede taal voor 992 van de 1.190 zinnen die het niet eerder zag: 83,4%. Cavnar en Trenkle deden het veel beter: met profielen van 400 n-grams, van lengte 1 tot en met 5 samen, had hun programma ongeveer 99,7% van 3.478 Usenet-berichten in acht talen goed (het paper zegt 99,8%). Maar hun berichten waren gemiddeld ongeveer 1.700 bytes lang, en de zinnen van Tatoeba in de course gemiddeld ongeveer 30 tekens. Volgens het overzicht van Jauhiainen en collega's is language identification van korte strings lastig voor de bestaande technieken. Het overzicht meldt dat Mandl en collega's in 2006 zagen dat elk programma dat ze probeerden het veel slechter deed op teksten van 25 tekens dan op teksten van 125. De eerste tabel hieronder voegt de achtergehouden zinnen samen, per twee, drie of vijf: met vijf zinnen in één tekst heeft het programma 95,4% goed.

    Nauw verwante talen zijn het andere probleem, en een bekend probleem voor bestaande programma's, schrijft het overzicht. De tweede tabel laat het Deens, het Bokmål en het Zweeds zien. René Haas en Leon Derczynski testten in 2021 langid.py, een bekend programma voor 97 talen, op Deens, Zweeds, Noors, Faeröers en IJslands uit Wikipedia: langid.py had 78,3% goed, en zag ongeveer een kwart van de zinnen in het Bokmål aan voor Deens. Op zinnen van Tatoeba had langid.py 80,9% goed. In 2008 schreef het leerboek *Introduction to Information Retrieval* van Christopher Manning, Prabhakar Raghavan en Hinrich Schütze, in paragraaf 2.5, nog dat language identification van geschreven tekst als een vrij eenvoudig probleem geldt, en language identification van gesproken taal als moeilijker. Dat klopt voor lange, schone teksten in een paar talen die veel van elkaar verschillen.
    """)
    return


@app.cell(hide_code=True)
def length_picture():
    mo.Html(C.length_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_junk():
    mo.md(r"""
    ### Tekst die geen taal is

    Het web staat vol tekst die helemaal geen taal is: markup, code, getallen, en letters die kapot zijn gegaan. Isaac Caswell, Theresa Breiner, Daan van Esch en Ankur Bapna van Google werkten in 2020 aan een webcorpus voor duizend talen, en zagen dat de corpora die ze met language identification van het web haalden, voor veel kleinere talen maar voor ongeveer 5% uit de goede taal bestonden, gemeten door mensen. In hun corpus voor de taal Dimli stond vooral ongeldige HTML, en dat voor Fula zat vol URL's. Veel van die fouten volgen volgens hen direct uit modellen die n-grams tellen. In 2022 controleerden Julia Kreutzer en veel medeauteurs met de hand steekproeven van 100 regels uit 205 webcorpora: 15 van de 205 steekproeven, 7,3%, bevatten geen enkele goede zin. In één verzameling, CCAligned, was gemiddeld 31,42% van de regels geen taal, over de corpora die ze controleerden; de auteurs noemen hun getallen zelf ruwe schattingen.

    **Mojibake** is een geval apart. Volgens de documentatie van ftfy, een programma van Robyn Speer dat mojibake herstelt, ontstaat mojibake meestal doordat goede UTF-8 gelezen wordt alsof het een andere encoding is. De letter *é* neemt in UTF-8 twee bytes in, en een encoding met één byte per letter leest die twee bytes als twee letters: *café* wordt *cafÃ©*. Een Nederlandse zin met een paar kapotte letters houdt de meeste trigrams, dus ligt nog steeds het dichtst bij het Nederlands. De afbeelding laat zeven teksten buiten de twaalf talen zien, met de afstand tot de dichtstbijzijnde taal en de grens van 0,9.
    """)
    return


@app.cell(hide_code=True)
def junk_picture():
    mo.Html(C.junk_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_junk():
    mo.md(r"""
    De grens vangt de HTML, de data, de datum en beide versies van de Russische zin. De regel code en de Turkse zin glippen eronder door: beide liggen op 0,88 van het Bokmål, en het programma kan geen *Turks* antwoorden, want het Turks heeft geen profiel. Het overzicht spreekt van *unseen languages*: talen zonder trainingsdata die een programma voor language identification in echte data toch kan tegenkomen. Een lagere grens vangt meer: bij 0,85 krijgen de regel code en de Turkse zin ook *?*, maar dan ook 12 van de 1.190 achtergehouden zinnen in plaats van 6. Waar de grens ligt, is een keuze tussen rommel doorlaten en echte zinnen afwijzen.

    Gemengde tekst is een probleem apart. *Introduction to Information Retrieval* geeft het voorbeeld van een Franse e-mail die zinnen uit een Engels contract citeert. Het overzicht meldt dat Stupar en collega's in 2011 in 4% van de documenten die ze verzamelden tekst in meer dan één taal vonden. Een programma dat voor elke tekst één taal geeft, geeft ook voor een mengsel één antwoord.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: je eigen tekst

    Typ een tekst, en zet de grens. De afbeelding laat de afstand tot elk van de twaalf talen zien, het antwoord, en welke trigrams van je tekst in het profiel van de dichtstbijzijnde taal staan. De tekst begint als een mengsel van Nederlands en Engels. Probeer een zin in een dertiende taal, een naam, een webadres, of een tekst met kapotte letters.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text_area(value=C.DEMO_TEXT, label="Je tekst", full_width=True)
    demo_limit = L.slider("demo-limit", "grens", start=0.9, low=0.5, high=1.0, step=0.01, role="hyper")
    mo.vstack([demo_text, demo_limit])
    return demo_limit, demo_text


@app.cell(hide_code=True)
def demo_look(demo_limit, demo_text):
    mo.Html(C.demo_picture(demo_text.value, demo_limit.widget.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Het mengsel van Nederlands en Engels komt eruit als Engels, op 0,684, met het Nederlands als tweede op 0,751: vijf van de negen woorden zijn Engels, en het programma heeft één antwoord voor de hele tekst. Jurafsky en Martin schrijven in *Speech and Language Processing* dat sprekers en schrijvers vaak meer talen in één uiting gebruiken, wat code switching heet. Om elk deel van zo'n tekst een taal te geven, moet een programma naar kleinere stukken kijken, zoals elk woord met de buren.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: andere profielen

    Deze les koos trigrams en profielen van 300. Kies een andere grootte voor de profielen en een andere n, en kijk hoe het programma het doet op de 1.190 achtergehouden zinnen, met dezelfde grens van 0,9.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_size = mo.ui.dropdown(options=[str(s) for s in C.SIZES], value="300", label="N-grams in een profiel, hoogstens")
    explore_n = mo.ui.slider(1, 5, value=3, label="n", show_value=True)
    mo.vstack([explore_size, explore_n])
    return explore_n, explore_size


@app.cell(hide_code=True)
def explore_look(explore_n, explore_size):
    mo.Html(C.explore_picture(int(explore_size.value), explore_n.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Losse tekens geven maar de helft van de zinnen goed: elk van de twaalf talen gebruikt veel *e*, *a* en *n*. Bigrams halen al 81,6%. Langere n-grams zeggen meer, maar een korte zin deelt er minder met de profielen, dus komen meer zinnen boven de grens uit: met 5-grams krijgen 520 van de 1.190 zinnen *?*. Grotere profielen helpen: met 3.000 trigrams komt 89,5% er goed uit. Het overzicht waarschuwt wie zulke getallen vergelijkt: hoe goed een lengte van de n-grams het doet, hangt sterk af van de opzet waarin die lengte getest is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Wat komt eruit?

    Elke vetgedrukte regel is een tekst. Zet de tekst bij wat je `identify` ervoor geeft, met de twaalf profielen van deze les en een grens van 0,9.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_comes_out = L.sort("what-comes-out")
    sort_comes_out
    return (sort_comes_out,)


@app.cell(hide_code=True)
def check_sort(sort_comes_out):
    result_sort = L.check("what-comes-out", sort_comes_out)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een bigram-model door te tellen](/course/open/lm-bigram-counts?lang=nl) telde paren tekens om het volgende teken te voorspellen; deze les telde rijtjes van drie tekens om talen uit elkaar te houden, nadat de tekst in kleine letters stond, zoals in [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl). Een profiel houdt alleen de rangen van de vaakste trigrams over, de out-of-place-maat vergelijkt twee profielen, en een grens laat het programma zeggen dat het de taal niet weet. Korte teksten, nauw verwante talen, talen zonder profiel en tekst die helemaal geen taal is, blijven lastig. Programma's zoals langid.py maken van de aantallen kansen: volgens Marco Lui en Timothy Baldwin is langid.py een naive Bayes classifier over n-grams van 1 tot en met 4 bytes. De volgende les van het domein, *Spam of niet: naive Bayes*, bouwt zo'n classifier, voor spam.

    **Een voorbeeld uit het dagelijks leven.** Een online vertaler die boven je tekst *Nederlands, gedetecteerd* laat zien, heeft hetzelfde werk gedaan als je `identify`: de vertaler heeft de taal uit alleen de tekst afgeleid.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_identify, result_ngrams, result_oop, result_sort):
    L.wrap_up(result_ngrams, result_oop, result_identify, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
