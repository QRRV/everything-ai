# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-word-classes. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-word-classes", "nl")
    C = L.checks   # de zinnen van twee treebanks, Engels en Nederlands, met de tag van elk woord


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_inflect():
    quick_inflect = L.choice(
        "quick-inflect",
        question="Welk paar woorden is gemaakt met inflectie?",
        options={
            "ful": "*schoon* en *schoonheid*, omdat *-heid* een achtervoegsel is",
            "right": "*werkt* en *werkte*",
            "dog": "*deur* en *voordeur*",
        },
        correct="right",
        why={
            "right": "*-te* geeft de verleden tijd, en *werkte* is nog steeds een werkwoord, net als *werkt*: inflectie maakt de vormen van één woord en houdt het soort woord gelijk. Over die soorten woorden gaat deze les.",
            "ful": "*-heid* is een achtervoegsel, maar het maakt een nieuw woord: het bijvoeglijk naamwoord *schoon* wordt het zelfstandig naamwoord *schoonheid*. Dat is derivatie. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "dog": "*voordeur* bestaat uit twee woorden, *voor* en *deur*: een samenstelling. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_inflect
    return (quick_inflect,)


@app.cell(hide_code=True)
def quick_check_cut():
    quick_cut = L.choice(
        "quick-cut",
        question="In *Woorden en hun delen* knipt `cut_suffix` de eerste uitgang van de lijst af waarop het woord eindigt, met minstens 3 letters over, en geeft beide delen terug, of het woord als één deel. Wat geeft `cut_suffix(\"kunst\", [\"te\", \"en\", \"t\"])`?",
        options={
            "word": "`['kunst']`, omdat *kuns* geen Nederlands woord is",
            "both": "`['kunst', 't']`",
            "right": "`['kuns', 't']`",
        },
        correct="right",
        why={
            "right": "*kunst* eindigt op *-t* en houdt 4 letters over, dus `cut_suffix` knipt daar, al is *kunst* geen vorm van een werkwoord. De functie kijkt alleen naar de letters van één woord; in deze les helpen ook de woorden eromheen.",
            "word": "`cut_suffix` kijkt niet of wat overblijft een woord is: alleen naar de uitgang en de 3 letters. Zie oefening 1 in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "both": "`cut_suffix` geeft terug wat overblijft en de uitgang, `[rest, suffix]`, en van *kunst* blijft *kuns* over. Zie oefening 1 in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_cut
    return (quick_cut,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    *Woorden en hun delen* knipte woorden in stukken. Maar een zin is meer dan een rij woorden: elk woord heeft een rol, en woorden die bij elkaar horen, vormen groepen. **Syntaxis** is de manier waarop woorden samen groepen en zinnen vormen. Daniel Jurafsky en James H. Martin schrijven in hun leerboek *Speech and Language Processing* (versie van 19 augustus 2026) dat het woord van het Griekse *sýntaxis* komt, dat samen opstellen of ordening betekent.

    Een programma dat weet wat voor woord het ziet, weet meer dan de letters. Volgens Jurafsky en Martin zijn de soorten woorden nuttige aanwijzingen voor de bouw en de betekenis van een zin: of een woord een zelfstandig naamwoord of een werkwoord is, zegt iets over de woorden die er waarschijnlijk naast staan. In het Engels komen zelfstandige naamwoorden na woorden als *the* en na bijvoeglijke naamwoorden, en werkwoorden na zelfstandige naamwoorden. Toch kunnen dezelfde letters verschillende soorten woorden zijn. In *ik zet koffie* is *zet* iets wat je doet, maar in *een slimme zet* is het een ding.

    Deze les geeft de soorten woorden een naam, telt ze in duizenden echte zinnen die mensen woord voor woord gelabeld hebben, in het Engels en in het Nederlands, en zoekt de woorden die meer dan één soort zijn.

    **Een voorbeeld uit het dagelijks leven.** De stukken bij schaken zijn van een paar soorten, en de soort van een stuk zegt hoe het mag bewegen, waar het ook op het bord staat.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Jurafsky en Martin geven een tabel voor twee grote verzamelingen Engelse tekst waarin elk woord een label voor de soort heeft. Noem een woord dubbelzinnig als het ergens in zo'n verzameling meer dan één label heeft. Welk deel van de woorden in lopende tekst is dubbelzinnig, als je elke keer telt dat een woord voorkomt?",
        options={
            "few": "Minder dan 1 op de 10",
            "third": "Ongeveer een derde",
            "right": "Meer dan de helft",
        },
        reveal="**Meer dan de helft.** In hun tabel kunnen 55 van de 100 woorden in de teksten uit de *Wall Street Journal* meer dan één label hebben, en 67 van de 100 in de teksten van het Brown-corpus, met een set van 45 labels. Tel je verschillende woorden, dan zijn het er maar 14 à 15 op de 100: de dubbelzinnige woorden zijn weinig in aantal, maar komen heel vaak voor, zoals *that*, *back*, *down*, *put* en *set*. De afbeelding onder oefening 2 telt dit in de twee verzamelingen van deze les.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Soorten woorden

    In *De eerste helft boeide weinig* is *helft* een zelfstandig naamwoord, *boeide* een werkwoord en *eerste* een bijvoeglijk naamwoord. Zo'n soort woord heet een **woordsoort** (in het Engels part of speech, afgekort POS). Jurafsky en Martin beginnen bij een Griekse grammatica van rond 100 v.Chr., van Dionysius Thrax of misschien iemand anders. Daarin staan acht woordsoorten, waaronder zelfstandig naamwoord, werkwoord, voornaamwoord en voorzetsel. Eerdere geleerden, zoals Aristoteles, hadden eigen lijsten, maar deze acht werden de volgende 2000 jaar de basis voor beschrijvingen van Europese talen.

    Vaak zie je de woordsoort van een woord aan hoe het zich in een zin gedraagt. In de Engelse voorbeelden van Jurafsky en Martin kan een zelfstandig naamwoord na *a* of *this* komen (*a goat*, *this bandwidth*) en een meervoud krijgen (*goats*); een werkwoord heeft vormen als *eat*, *eats*, *eating* en *eaten*. In het Nederlands is dat net zo: *een geit*, *geiten*, en *loop*, *loopt*, *liep*. Die vormen komen van inflectie, en die houdt, zoals *Woorden en hun delen* liet zien, het soort woord gelijk.

    ### Open en gesloten woordsoorten

    Jurafsky en Martin verdelen de woordsoorten in twee groepen. Zelfstandige naamwoorden en werkwoorden zijn **open woordsoorten**: er komen steeds nieuwe bij, zoals *iPhone* of *faxen*. **Gesloten woordsoorten** krijgen bijna nooit nieuwe woorden: voorzetsels als *van* en *in* vormen een gesloten woordsoort, en er komt zelden een nieuw voorzetsel bij. De woorden van gesloten woordsoorten zijn meestal korte woorden die vaak voorkomen en een taak in de grammatica hebben, zoals *van*, *het*, *en* en *jij*.

    Niet elke taal heeft dezelfde woordsoorten, schrijven Jurafsky en Martin. Sommige talen hebben geen bijvoeglijke naamwoorden: in het Koreaans gedragen de woorden voor Engelse bijvoeglijke naamwoorden als *beautiful* (mooi) zich als werkwoorden. Veel taalkundigen denken dat elke menselijke taal zelfstandige naamwoorden en werkwoorden heeft; anderen stellen dat sommige talen, zoals het Tongaans en het Riau-Indonesisch, dat onderscheid niet eens maken.

    ### Zeventien tags voor elke taal

    Een **treebank** is een verzameling zinnen die mensen woord voor woord ontleed hebben: elk woord heeft een label voor de woordsoort, een **tag**, en de bouw van elke zin staat erin als een boom; die laat een latere les zien. Lang had elke treebank eigen tags. De Penn Treebank voor het Engels (Marcus, Santorini en Marcinkiewicz, 1993) had 36 tags voor woordsoorten, plus tags voor leestekens en symbolen.

    **Universal Dependencies** (UD) is een project dat zinnen van veel talen op dezelfde manier labelt: de website spreekt van meer dan 600 mensen die eraan bijdragen, en meer dan 200 treebanks in meer dan 150 talen. Het heeft in elke taal dezelfde 17 tags voor woordsoorten, in drie groepen:

    | Groep | Tag | Woordsoort | Engels | Nederlands |
    |---|---|---|---|---|
    | open | `NOUN` | zelfstandig naamwoord | *time*, *people* | *jaar*, *helft* |
    | | `PROPN` | eigennaam | *Iraq*, *China* | *Nederland*, *Amsterdam* |
    | | `VERB` | werkwoord | *have*, *get*, *know* | *komt*, *heet* |
    | | `ADJ` | bijvoeglijk naamwoord | *good*, *new* | *eerste*, *nieuwe* |
    | | `ADV` | bijwoord | *just*, *very*, *now* | *niet*, *ook* |
    | | `INTJ` | tussenwerpsel | *please*, *yes*, *wow* | *ja*, *nee* |
    | gesloten | `ADP` | voorzetsel | *of*, *in*, *for* | *van*, *op* |
    | | `AUX` | hulpwerkwoord | *is*, *will*, *can* | *is*, *heeft*, *zal* |
    | | `CCONJ` | nevenschikkend voegwoord | *and*, *but*, *or* | *en*, *maar* |
    | | `DET` | lidwoord en woorden als *deze* en *geen* | *the*, *a*, *this* | *de*, *het*, *een* |
    | | `NUM` | telwoord | *one*, *two*, `3` | *twee*, *drie* |
    | | `PART` | partikel | *to* in *to go*, *not* | (geen in deze treebank) |
    | | `PRON` | voornaamwoord | *I*, *it*, *you* | *ik*, *wat* |
    | | `SCONJ` | onderschikkend voegwoord | *that*, *if*, *because* | *dat*, *omdat* |
    | overig | `PUNCT` | leesteken | `.` `,` `?` | `.` `,` `?` |
    | | `SYM` | symbool | `$` `%` | `=` `km` |
    | | `X` | al het andere | stukjes van woorden, woorden uit een andere taal | *pct.*, *the* in een Nederlandse zin |

    Deze les gebruikt twee treebanks van UD-versie 2.19 (november 2026), en elk voorbeeld in de tabel komt daaruit. *English-EWT* heeft zinnen van het web: blogs, nieuwsgroepen, recensies en vragen met hun antwoorden; de course liet de e-mails weg, die echte namen en adressen bevatten. *Dutch-Alpino* heeft zinnen uit Nederlandse kranten en ander geschreven Nederlands.

    ### Woordgroepen

    Woorden die bij elkaar horen, vormen een **woordgroep**. Een **naamwoordgroep** is een zelfstandig naamwoord met de woorden die erbij horen: *de eerste helft*, met het lidwoord *de* en het bijvoeglijk naamwoord *eerste* vóór het zelfstandig naamwoord *helft*. Jurafsky en Martin beschreven die volgorde voor het Engels: zelfstandige naamwoorden komen na lidwoorden en bijvoeglijke naamwoorden, zoals in *a fantastic time* (een fantastische tijd). De volgende les laat zien hoe woordgroepen zijn opgebouwd, de ene in de andere.
    """)
    return


@app.cell(hide_code=True)
def tagged_cell():
    mo.Html(C.tagged_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **syntaxis** | de manier waarop woorden samen woordgroepen en zinnen vormen |
        | **woordsoort** | de soort van een woord, zoals zelfstandig naamwoord of werkwoord, naar hoe het zich in een zin gedraagt; in het Engels part of speech (POS) |
        | **open woordsoort**, **gesloten woordsoort** | woordsoorten die steeds nieuwe woorden krijgen, zoals zelfstandige naamwoorden, en woordsoorten die dat bijna nooit doen, zoals voorzetsels |
        | **treebank** | een verzameling zinnen die mensen woord voor woord ontleed hebben, met een boom voor elke zin |
        | **tag** | het label voor de woordsoort van een woord, zoals `NOUN` |
        | **Universal Dependencies** | een project dat zinnen van veel talen op dezelfde manier labelt, met 17 tags voor woordsoorten |
        | **naamwoordgroep** | een zelfstandig naamwoord met de woorden die erbij horen, zoals *de eerste helft* |
        | `C.EWT`, `C.ALPINO` | de zinnen van de twee treebanks, 11.723 Engelse en 13.603 Nederlandse: een tuple van zinnen, elk een tuple van paren `(word, tag)` |
        | `for word, tag in sentence:` | gaat de paren langs, en geeft de twee delen van elk paar een eigen naam, zoals `for key, value in a_dict.items():` dat doet |
        | `key in a_dict` | `True` als `key` een van de keys van de dict is |
        | `x not in a_list` | `True` als `x` niet een van de items van de lijst is |
        """),
        L.watch_out("Mensen gaven de tags, volgens de richtlijnen van hun treebank, en treebanks kunnen voor dezelfde soort woord anders kiezen. Waar het Engels het bijwoord *well* gebruikt, geeft Alpino *goed* de tag `ADJ`, ook in *het gaat goed*: 153 van de 155 keer dat *goed* voorkomt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Ga elke zin langs, en elk paar `(word, tag)` van elke zin.
    2. Om te tellen, houd je een dict bij: de eerste keer dat je een tag tegenkomt, krijgt die de telling 1, en daarna elke keer één meer.
    3. Om de tags van een woord te vinden, houd je een dict bij met het woord in kleine letters als key en een lijst als waarde, en voeg je elke tag die het woord krijgt één keer toe.
    4. Kijk naar de getallen: welke soorten komen het vaakst voor, en hoe vaak heeft één woord meer dan één tag?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel gaat de paren langs van `C.EXAMPLE`, de Engelse zin van de afbeelding, print elk woord met de tag ervan, en telt de zelfstandige naamwoorden.
    """)
    return


@app.cell
def example_count():
    nouns = 0
    for word, tag in C.EXAMPLE:
        print(word, tag)
        if tag == "NOUN":
            nouns = nouns + 1
    print("nouns:", nouns)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *time* (tijd) is het enige zelfstandig naamwoord, en ook de punt heeft een tag: `PUNCT`. `C.EXAMPLE` is één zin; `C.EWT` bevat 11.723 van zulke zinnen, en `C.ALPINO` 13.603. Oefening 1 telt elke tag, in elke zin.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. count_tags: hoeveel woorden van elke soort

        Schrijf `count_tags(sentences)`: geef een dict terug met elke tag die in `sentences` voorkomt als key, en als waarde het aantal woorden met die tag. Begin met een lege dict, en ga elk paar van elke zin langs. Is de tag al een key, dan gaat de telling ervan één omhoog; anders krijgt de tag de telling 1. Geef de dict aan het eind terug. Vervang de `...`. De check roept `count_tags(C.SAMPLE)` aan, twee korte Engelse zinnen naar voorbeelden van Jurafsky en Martin, *Book that flight.* (boek die vlucht) en *Hand me that book.* (geef me dat boek), met tags van de course; daarna eigen zinnen, ook in het Nederlands, en helemaal geen zinnen.
        """),
        L.task("Ga elk woord van elke zin langs, en tel hoeveel woorden elke tag hebben."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_tags", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_tags", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_tags",
        [
            "Twee loops, de ene in de andere: over de zinnen, en over de paren van een zin, zoals in het uitgewerkte voorbeeld. Binnenin vraagt een `if` of de tag al een key is.",
            "Wat is de telling van een tag die je voor het eerst tegenkomt? En waarom moet de lege dict vóór de loops gemaakt worden, en niet erin?",
            "`                counts[tag] = ___`",
        ],
        why="`counts = {}`, dan `for sentence in sentences:` en daarin `for word, tag in sentence:`, met `if tag in counts:` en daaronder `counts[tag] = counts[tag] + 1`, en `else:` met daaronder `counts[tag] = 1`; na beide loops `return counts`. `count_tags(C.SAMPLE)` geeft `{'VERB': 2, 'DET': 2, 'NOUN': 2, 'PUNCT': 2, 'PRON': 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    De afbeelding onder de check laat je functie zien op beide treebanks, per 100 woorden. In beide talen is `NOUN` de grootste groep, ongeveer 17 van de 100 woorden. Het Nederlands heeft meer `ADP` en `DET` (*van*, *de*, *een*), het Engels meer `PRON`. De paar woorden van de gesloten woordsoorten, zoals *de*, *van* en *en*, zijn samen goed voor ongeveer 4 van de 10 woorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tags():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. tags_per_word: één woord, meer dan één tag

        Schrijf `tags_per_word(sentences)`: geef een dict terug met elk woord, in kleine letters, als key, en als waarde de lijst met de tags die het in `sentences` heeft, elke tag één keer, in de volgorde waarin ze voor het eerst komen. Begin met een lege dict. Maak bij elk paar het woord klein met `lower()`. Is het kleine woord nog geen key, geef het dan een lege lijst. Staat de tag daarna nog niet in de lijst ervan, voeg de tag dan toe met `append`. Geef de dict aan het eind terug. Vervang de `...`. De check roept `tags_per_word(C.SAMPLE)` aan, waarin *Book* een `VERB` is en *book* een `NOUN`; daarna eigen zinnen.
        """),
        L.task("Geef elk woord in kleine letters de lijst met de tags ervan, elke tag één keer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tags():
    box_tags = L.code_box("tags_per_word", lines=10)
    box_tags
    return (box_tags,)


@app.cell(hide_code=True)
def check_tags(box_tags):
    result_tags = L.check("tags_per_word", box_tags.value)
    result_tags
    return (result_tags,)


@app.cell(hide_code=True)
def hints_tags():
    L.hints(
        "tags_per_word",
        [
            "Twee loops zoals in oefening 1. Binnenin twee keer `if`, na elkaar: de eerste maakt de lijst voor een nieuw woord, de tweede voegt de tag toe.",
            "Waarom moet de eerste `if` vóór de tweede komen? Wat zou er misgaan bij een woord dat je voor het eerst tegenkomt?",
            "`            if tag not in ___:`",
        ],
        why="`tags = {}`, dan `for sentence in sentences:` en daarin `for word, tag in sentence:`, met `small = word.lower()`, dan `if small not in tags:` met daaronder `tags[small] = []`, en `if tag not in tags[small]:` met daaronder `tags[small].append(tag)`; na beide loops `return tags`. `tags_per_word(C.SAMPLE)` geeft `{'book': ['VERB', 'NOUN'], 'that': ['DET'], 'flight': ['NOUN'], '.': ['PUNCT'], 'hand': ['VERB'], 'me': ['PRON']}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tags():
    mo.md(r"""
    In de afbeelding onder de check heeft *back* zes tags in EWT, en is *loop* in Alpino een zelfstandig naamwoord en een werkwoord. Net als in de gok heeft in beide treebanks meer dan de helft van de woorden in lopende tekst ergens meer dan één tag. Tel je verschillende woorden, dan zijn het er in het Engels ongeveer 13 van de 100, en in het Nederlands minder dan 5 van de 100.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de tags van een woord

    Typ een woord, en zie hoe vaak het elke tag heeft in de twee treebanks. Probeer *dat*, *een* of *weinig*; Engelse woorden als *that*, *to* of *well*; of *de*: bijna elke keer een lidwoord.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="loop", label="Een woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 3. Dezelfde letters, een andere tag

    Het vetgedrukte woord heeft elke keer dezelfde letters. Zet elk woord onder de tag die het in de zin heeft. De zinnen met *back* zijn voorbeelden van Jurafsky en Martin; de andere zijn van de course zelf.
    """)
    return


@app.cell(hide_code=True)
def exercise_which():
    sort_which = L.sort("which")
    sort_which
    return (sort_which,)


@app.cell(hide_code=True)
def check_which(sort_which):
    result_which = L.check("which", sort_which)
    result_which
    return (result_which,)


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Oefening 4. Wat komt er na een DET?

    Een programma leest een `DET`, zoals het lidwoord *de* of, in het Engels, *the*, en moet de tag van het volgende woord raden voordat het dat woord ziet.
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after",
        question="Welke tag is de beste gok?",
        options={
            "right": "NOUN, zoals in *de vlucht* en *the flight*",
            "adj": "ADJ, omdat een DET even vaak vóór een bijvoeglijk als vóór een zelfstandig naamwoord staat",
            "propn": "PROPN, zoals in *de Veluwe*",
        },
        correct="right",
        why={
            "right": "Na een `DET` zijn in Alpino 63 van de 100 woorden een `NOUN`, en in EWT 57 van de 100; `ADJ` komt op de tweede plaats, met 21 en 25 van de 100. Dat bedoelden Jurafsky en Martin: de tag van een woord zegt veel over de buren. Een latere les bouwt een programma dat elk woord de tag geeft.",
            "adj": "Een bijvoeglijk naamwoord komt vaak na een DET, zoals in *de eerste helft*, maar een zelfstandig naamwoord bijna drie keer zo vaak: na een `DET` zijn in Alpino 63 van de 100 woorden een `NOUN` en 21 van de 100 een `ADJ`. Zie *Woordgroepen* in *Het idee*.",
            "propn": "Namen komen af en toe na een DET, zoals in *de Veluwe*, maar na een `DET` zijn in Alpino slechts ruim 7 van de 100 woorden een `PROPN`. Zie *Woordgroepen* in *Het idee*.",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl) knipte woorden in morfemen, en liet zien dat inflectie de soort van een woord gelijk houdt. Die soort is de woordsoort. Universal Dependencies geeft elke taal dezelfde 17 tags, in open woordsoorten die steeds groeien en gesloten woordsoorten van korte woorden die vaak voorkomen. Je telde de tags van een Engelse en een Nederlandse treebank, en zette de tags van elk woord op een rij: dezelfde letters kunnen een zelfstandig naamwoord, een werkwoord of een bijwoord zijn, en de woorden eromheen bepalen welke. Woorden vormen ook woordgroepen, zoals de naamwoordgroep *de eerste helft*. De volgende les laat zien hoe woordgroepen zijn opgebouwd, en hoe een paar regels eindeloos veel zinnen kunnen maken.

    **Een voorbeeld uit het dagelijks leven.** In een band speelt dezelfde muzikant soms in het ene nummer gitaar en in het volgende drums: welke rol het is, weet je pas als je het nummer hoort.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_after, result_count, result_tags, result_which):
    L.wrap_up(result_count, result_tags, result_which, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
