# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-distributional-hypothesis. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-distributional-hypothesis", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Een document als vector](/course/open/nlp-document-vectors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dot():
    quick_dot = L.choice(
        "quick-dot",
        question="Twee sparse rijen bevatten aantallen per woord: `{\"cup\": 2, \"tea\": 1}` en `{\"cup\": 3, \"milk\": 4}`. Wat is het inproduct?",
        options={
            "pairs": "10",
            "right": "6",
            "add": "5",
        },
        correct="right",
        why={
            "right": "Alleen *cup* staat in beide rijen, dus alleen *cup* telt mee voor het inproduct: 2 × 3 = 6. *tea* en *milk* hebben elk een 0 in de andere rij. Deze les vergelijkt zulke rijen, met woorden als sleutels.",
            "pairs": "10 koppelt het eerste aantal aan het eerste en het tweede aan het tweede: 2 × 3 + 1 × 4. Een sparse rij koppelt aantallen via de sleutel, en *tea* en *milk* zijn verschillende sleutels. Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "add": "5 telt de twee aantallen van *cup* op; het inproduct vermenigvuldigt de twee aantallen. Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
        },
    )
    quick_dot
    return (quick_dot,)


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="In die les zocht de achtergehouden zin *The delivery was on time.* de meest gelijkende trainingszin met de cosine similarity van kale aantallen. Welke woorden hadden de twee zinnen gemeen?",
        options={
            "topic": "delivery, time",
            "right": "the, was, on",
            "tone": "great, rude",
        },
        correct="right",
        why={
            "right": "De meest gelijkende zin was *The bus boy on the other hand was so rude.*: geen woord over bezorging of tijd. Korte, gewone woorden staan in zoveel zinnen dat deze woorden een groot deel van de cosine similarity van kale aantallen bepalen. In deze les telt elk woord bij een woord mee, ook *the*: let op wat dat met de rijen doet.",
            "topic": "De zin over hetzelfde onderwerp, *The shipping time was also very quick!*, kwam pas op de derde plaats. Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "tone": "In *The delivery was on time.* staat geen woord van lof of klacht. Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een woordenboek legt een woord uit met andere woorden. Een computer heeft geen woordenboek in het hoofd, maar kan wel tellen. Deze les geeft elk woord een lijst getallen, een vector, door te tellen welke woorden er in een paar boeken bij staan, en woorden met een vergelijkbare betekenis blijken vergelijkbare vectoren te krijgen. Het idee is ouder dan de computers waarmee je het kon testen. De taalkundige Zellig Harris schreef in 1954 in *Distributional Structure* dat verschil in betekenis samengaat met verschil in distributie. Met de distributie van een woord bedoelde Harris de som van alle omgevingen van het woord: elke plaats waar het voorkomt, met de woorden eromheen. Het voorbeeld van Harris: *oculist* en *eye-doctor*, twee Engelse woorden voor oogarts, komen in bijna dezelfde omgevingen voor, *oculist* en *lawyer* (advocaat) maar in een deel daarvan. Dat tweede paar verschilt volgens Harris in betekenis, en ruwweg zoveel als de omgevingen verschillen.

    In 1957 schreef de taalkundige J. R. Firth de zin die nog steeds wordt aangehaald, en waar de titel van deze les op teruggaat: aan het gezelschap waarin een woord verkeert, herken je het (in het Engels: *You shall know a word by the company it keeps!*). Firth bedoelde iets beperkters dan deze les: de woorden die vaste metgezellen van een woord zijn, de collocaties, zoals *silly* bij *ass* in de Engelse uitroep *You silly ass!*. Voor Firth was dat gezelschap een van de betekenissen van een woord. Het leerboek *Speech and Language Processing* van Daniel Jurafsky en James H. Martin beschrijft het idee en de naam zo: woorden die in vergelijkbare contexten voorkomen, hebben vaak een vergelijkbare betekenis, en dat verband tussen waar woorden staan en wat woorden betekenen, heet de distributionele hypothese (in het Engels distributional hypothesis). De word embeddings van latere lessen, en de language models die daarop voortbouwen, steunen op hetzelfde idee.

    **Een voorbeeld uit het dagelijks leven.** In de trein praten twee mensen over een padelbaan, rackets, een wedstrijd op zondag, en een partner die te hard serveert. Je hebt nog nooit gepadeld, maar je weet al dat padel een spel is met rackets, in tweetallen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In de drie Nederlandse boeken van de course krijgt elk woord dat minstens 20 keer voorkomt en geen stop word is, een rij: hoe vaak elk woord binnen twee plaatsen ervan staat. Welk woord heeft de rij die het meest lijkt op de rij van *warm*?",
        options={
            "warm": "heet",
            "fire": "vuur",
            "right": "koud",
            "summer": "zomer",
        },
        reveal="**koud**: van de 1.950 andere woorden met een rij heeft *koud* de rij die het meest lijkt op de rij van *warm*, met een cosine similarity van 0,699. *vuur* komt op plaats 94, *heet* op plaats 433 en *zomer* op plaats 1.445. *warm* en *koud* betekenen het tegenovergestelde, maar verkeren in hetzelfde gezelschap: de woorden die de twee rijen het meest delen, zijn *en*, *te*, *het*, *een* en *was*, en beide woorden staan bij *geworden*. Een rij legt vast waar een woord staat, niet wat het zegt.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Context en venster

    De **context** van een woord is de tekst eromheen. Deze les neemt daar een klein stuk van: het **venster**, de woorden binnen een vast aantal plaatsen voor en na het woord. Dat aantal is de grootte van het venster. Met een venster van grootte 2 bevat het venster van *thee* in *we drinken thee met melk* de woorden *we*, *drinken*, *met* en *melk*. In het voorbeeld van Jurafsky en Martin is het venster groter: vier woorden links en vier woorden rechts van het woord. Het woord waarvan je het venster neemt, is het **doelwoord**, en de woorden in het venster zijn de **contextwoorden**. Een doelwoord en een woord in het venster ervan **komen samen voor**.

    ### De co-occurrence matrix

    Tel in een hele tekst hoe vaak elk woord in een venster van elk doelwoord staat, en je krijgt een tabel met één rij per doelwoord en één kolom per contextwoord: de **co-occurrence matrix**. Jurafsky en Martin beschrijven een soort daarvan: elke rij hoort bij een woord van de vocabulary, en elke kolom zegt hoe vaak een ander woord van de vocabulary in de buurt staat. Elke rij is de vector van een woord. Net als bij de documentvectoren van [Een document als vector](/course/open/nlp-document-vectors?lang=nl) is bijna elk aantal 0, dus deze les bewaart elke rij als sparse rij: een dictionary van contextwoord naar aantal, met het contextwoord als sleutel. De afbeelding toont de matrix van vier doelwoorden in zes korte zinnen van de course zelf.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Hetzelfde gezelschap, vergelijkbare rijen

    *thee* en *koffie* staan tussen dezelfde woorden: *we drinken … met*, *een kop …*. De rijen delen vier van de zes kolommen, en de cosine similarity van de twee rijen is 0,67. *thee* en *auto* delen geen kolom, dus daar is het 0. Over een paar boeken krijgen de rijen duizenden kolommen, en de **dichtstbijzijnde buren** van een woord zijn de woorden waarvan de rij de hoogste cosine similarity heeft met de rij van dat woord. De betekenis van woorden beschrijven via de contexten waarin de woorden voorkomen, met vectoren zoals deze, heet **distributionele semantiek**.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **distributionele hypothese** | het idee dat woorden die in vergelijkbare contexten voorkomen, vaak een vergelijkbare betekenis hebben; in het Engels distributional hypothesis |
        | **distributionele semantiek** | de betekenis van woorden beschrijven via de contexten waarin de woorden voorkomen, meestal met vectoren |
        | **context** | de tekst rond een woord |
        | **venster** | de woorden binnen een vast aantal plaatsen voor en na een woord; met grootte 2 twee woorden aan elke kant; in het Engels window |
        | **doelwoord** | een woord waarvan de vensters worden geteld; het krijgt een rij |
        | **contextwoord** | een woord in een venster van een doelwoord; het krijgt een kolom |
        | **samen voorkomen** | van een doelwoord en een contextwoord: het ene woord staat in een venster van het andere; in het Engels co-occur |
        | **co-occurrence matrix** | een tabel met een rij per doelwoord en een kolom per contextwoord, met hoe vaak de twee samen voorkomen |
        | **dichtstbijzijnde buren** | de woorden waarvan de rij de hoogste cosine similarity heeft met de rij van een woord |
        | `max(a, b)` | het grootste van twee getallen: `max(3, 5)` geeft 5; `min(a, b)` geeft het kleinste |
        | `a + b` | bij twee lijsten: één lijst met eerst de elementen van `a` en dan de elementen van `b`: `["x"] + ["y", "z"]` geeft `["x", "y", "z"]` |
        """),
        L.watch_out("Vergelijkbare rijen betekenen dat twee woorden in hetzelfde gezelschap verkeren, niet dat de twee woorden hetzelfde betekenen. Woorden die op dezelfde plaatsen staan, krijgen vergelijkbare rijen, wat de woorden ook betekenen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip de tekst in woorden, in kleine letters.
    2. Kies de doelwoorden: hier elk woord dat minstens 20 keer voorkomt en geen stop word is.
    3. Loop langs de plaatsen van de tekst. Neem bij een doelwoord het venster: de woorden binnen `size` plaatsen ervoor en erna.
    4. Tel in de rij van dat doelwoord 1 op bij het aantal van elk woord in het venster.
    5. Vergelijk twee rijen met de cosine similarity. De dichtstbijzijnde buren van een woord zijn de woorden waarvan de rij de hoogste cosine similarity heeft met de rij van dat woord.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder neemt de zes zinnen uit de afbeelding, uit `toy_sentences` van de course, en de vier doelwoorden, uit `toy_targets` van de course. Voor elke plaats waar *thee* staat, laat de code de plaats zien, de twee woorden ervoor en de twee woorden erna. Daarna komt de rij van *thee* uit `toy_rows` van de course, en een vergelijking met twee andere rijen met `cosine` van de course, die hetzelfde doet als `sparse_cosine` uit [Een document als vector](/course/open/nlp-document-vectors?lang=nl).
    """)
    return


@app.cell
def example_cell():
    sentences = C.toy_sentences(L.lang)
    small_targets = C.toy_targets(L.lang)   # thee, koffie, auto, fiets
    first = small_targets[0]
    for sentence in sentences:
        for i in range(len(sentence)):
            if sentence[i] == first:
                print(i, sentence[i - 2:i], sentence[i + 1:i + 3])
    toy = C.toy_rows(L.lang)
    print(toy[first])
    print(C.cosine(toy[first], toy[small_targets[1]]), C.cosine(toy[first], toy[small_targets[2]]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *thee* staat op plaats 2 van de eerste zin en op plaats 3 van de derde. Daar is `sentence[4:6]` leeg: een slice die voorbij het eind van een lijst loopt, stopt bij het eind. De rij van *thee* telt beide vensters op: zes woorden, elk één keer. De cosine similarity met de rij van *koffie* is 0,667, en met de rij van *auto* 0,0. In dit voorbeeld staat *thee* nooit op plaats 0 of 1, dus `i - 2` is nooit kleiner dan 0. Oefening 1 maakt een venster voor elke plaats en elke grootte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_window():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. window: de woorden rond een plaats

        Schrijf `window(words, i, size)`: geef de woorden binnen `size` plaatsen van plaats `i` van de lijst `words` terug, zonder het woord op plaats `i` zelf, in de volgorde van de lijst: eerst de woorden ervoor, dan de woorden erna. Vlak bij het begin of het eind van de lijst bevat het venster minder woorden.

        Voor `window(["a", "b", "c", "d", "e", "f", "g"], 3, 2)` is de uitkomst `["b", "c", "e", "f"]`: *d* staat op plaats 3, met *b* en *c* ervoor en *e* en *f* erna. Vervang de `...`. De check roept dat aan en `window(["a", "b", "c", "d", "e", "f", "g"], 0, 2)`, en daarna eigen gevallen: plaats 1, de laatste plaats, een venster dat precies bij plaats 0 begint, de groottes 1, 0 en 3, een venster dat aan beide kanten breder is dan de lijst, een lijst van één woord, een woord dat twee keer voorkomt, een lijst van twaalf woorden, en de eerste woorden van twee boeken. Let op vlak na het begin: in een slice telt een negatief getal vanaf het eind van de lijst, dus `words[-1:1]` is leeg.
        """),
        L.task("Geef de woorden binnen size plaatsen van plaats i terug, zonder het woord op plaats i zelf."),
    ])
    return


@app.cell(hide_code=True)
def exercise_window():
    box_window = L.code_box("window", lines=4)
    box_window
    return (box_window,)


@app.cell(hide_code=True)
def check_window(box_window):
    result_window = L.check("window", box_window.value)
    result_window
    return (result_window,)


@app.cell(hide_code=True)
def hints_window():
    L.hints(
        "window",
        [
            "Neem twee slices van `words` en voeg de twee samen met `+`: de slice met de woorden vóór plaats `i` eindigt bij `i`, en de slice met de woorden erna begint bij `i + 1`. Elke slice bevat hoogstens `size` woorden. Geef de twee samen terug, na `return`.",
            "Met `i` gelijk aan 1 en `size` gelijk aan 2 is `i - size` gelijk aan −1, en `words[-1:1]` is leeg, terwijl het venster het woord op plaats 0 moet bevatten. Bij welke plaats moet de eerste slice beginnen, voor elke `i` en `size`?",
            r"`    return words[___:i] + words[i + 1:i + size + 1]`",
        ],
        why=r"`return words[max(0, i - size):i] + words[i + 1:i + size + 1]`: `max(0, i - size)` is `i - size`, of 0 als dat kleiner is dan 0. Een slice die voorbij het eind van de lijst loopt, stopt bij het eind.",
    )
    return


@app.cell(hide_code=True)
def text_after_window():
    mo.md(r"""
    In de afbeelding nam je functie het venster op vier plaatsen van de eerste woorden van het eerste hoofdstuk van *Max Havelaar*, de roman van Multatuli uit 1860: *Ik ben makelaar in koffi, en woon op de Lauriergracht*, in de spelling van toen. Op plaats 0 bevat het venster alleen *ben* en *makelaar*, op plaats 9 alleen *op* en *de*. Met een venster van grootte 2 krijgt elk woord van de boeken vier contextwoorden, behalve de eerste twee en de laatste twee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rows():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. cooccurrence: een rij voor elk doelwoord

        `words` bevat elk woord van de drie Nederlandse boeken van de course, in kleine letters, het ene boek na het andere, uit `book_words` van de course. `targets` bevat de doelwoorden, uit `targets` van de course: de 1.951 woorden die minstens 20 keer voorkomen en geen stop word zijn, als set. Schrijf `cooccurrence(words, size, targets)`: geef een dictionary terug met een rij voor elk doelwoord. Een rij is een sparse rij: een dictionary van elk woord dat in een venster van het doelwoord staat, naar hoe vaak het woord daar staat. Loop langs de plaatsen van `words`; tel bij een doelwoord 1 op bij het aantal van elk woord in het venster, in de rij van dat doelwoord. De startcode maakt al een lege rij voor elk doelwoord, dus een doelwoord dat niet voorkomt, houdt een lege rij. De check geeft je functie `window` van de course, die hetzelfde doet als jouw `window`.

        Voor `cooccurrence(["a", "b", "c", "a", "b"], 2, ["a", "c"])` is de uitkomst `{"a": {"b": 3, "c": 2}, "c": {"a": 2, "b": 2}}`: de eerste *a* heeft *b* en *c* in het venster, de tweede *a* heeft *b*, *c* en *b*, en *c* heeft *a*, *b*, *a* en *b*. Vervang de `...`. De check roept dat aan en `cooccurrence(["a", "b", "a"], 1, ["a", "z"])`, en daarna eigen gevallen: een woord naast zichzelf, geen woorden, geen doelwoorden, de groottes 1 en 0, de doelwoorden als set, een woord twee keer met een venster dat breder is dan de lijst, de eerste woorden van twee boeken, en de eerste 500 woorden van *Alice's Adventures in Wonderland*.
        """),
        L.task("Geef elk doelwoord een sparse rij: hoe vaak elk woord in de vensters van het doelwoord staat."),
    ])
    return


@app.cell
def books_cell():
    window = C.window
    words = C.book_words(L.lang)
    targets = C.targets(L.lang)
    print(len(words), len(targets))
    print(words[:12])
    print(sorted(targets)[:8])
    return


@app.cell(hide_code=True)
def text_after_books():
    mo.md(r"""
    De boeken hebben 469.251 woorden, en beginnen met de titelpagina van *Eline Vere*, de roman van Louis Couperus uit 1889. De andere twee zijn *Max Havelaar* (1860) en *Camera Obscura* van Hildebrand (1839), en alle drie hebben de spelling van toen: *zoo* voor zo, *oogen* voor ogen, *zeide* voor zei, en in *Max Havelaar* een *y* voor ij, zoals in *hy* en *zyn*. De lijst met stop words is van nu, dus *zoo*, *hy* en *zyn* zijn hier doelwoorden. Het meest voorkomende doelwoord is *den*, een oude vorm van *de*.
    """)
    return


@app.cell(hide_code=True)
def exercise_rows():
    box_rows = L.code_box("cooccurrence", lines=13)
    box_rows
    return (box_rows,)


@app.cell(hide_code=True)
def check_rows(box_rows):
    result_rows = L.check("cooccurrence", box_rows.value)
    result_rows
    return (result_rows,)


@app.cell(hide_code=True)
def hints_rows():
    L.hints(
        "cooccurrence",
        [
            "Loop langs de plaatsen met `for i in range(len(words)):` en neem `word = words[i]`. Als `word in targets`, loop dan langs de woorden van het venster met `for context in window(words, i, size):`. Als `context in rows[word]`, tel dan 1 op bij het aantal in die rij; begin het anders bij 1. Geef na de loop `rows` terug.",
            "In `cooccurrence([\"a\", \"b\", \"a\"], 1, [\"a\", \"z\"])` staat *b* in de vensters van beide *a*'s. Als de tweede *a* komt, is het aantal van *b* in de rij van *a* al 1: wat moet het worden, en hoe schrijf je dat voor elke `word` en `context`?",
            r"`                    rows[word][context] = ___`",
        ],
        why=r"Houd de startcode, die een lege rij maakt voor elk doelwoord. Dan `for i in range(len(words)):` met `word = words[i]`, en `if word in targets:` met `for context in window(words, i, size):` eronder; daar `if context in rows[word]:` met `rows[word][context] = rows[word][context] + 1`, en `else:` met `rows[word][context] = 1`. Na de loop `return rows`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rows():
    mo.md(r"""
    Je functie telde voor elk van de 1.951 doelwoorden elk woord binnen twee plaatsen ervan: 300.532 aantallen die geen 0 zijn. Kijk naar de buren in de afbeelding. *zeide* vindt *meende*, *dacht*, *vroeg*, *wenschte*, *zag* en *antwoordde*: allemaal werkwoorden in de verleden tijd, de meeste voor wat iemand zegt of denkt. *oogen* vindt *handen*, *hand*, *tranen* en *armen*, *liep* vindt *ging*, *sloeg* en *kwam*, en *drie* vindt *vier* en *twee*. *vader* vindt eerst *moeder*, dan *dochter*, *vrouw* en *zoon*. Niemand vertelde de rijen wat deze woorden betekenen; de woorden staan op dezelfde plaatsen. Sommige buren zijn vreemd: *drie* vindt ook *kinderen* en *kracht*, en *vader* vindt *liefde*.

    De grootste aantallen in de rij van *deur* zijn *de*, 175 keer, dan *en*, *van*, *open*, *in* en *voor*. Steeds dezelfde gewone woorden vullen elke rij, en daardoor is elke cosine similarity in de afbeelding hoger dan 0,7.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: een ander woord, een ander venster

    Typ een woord, en kies de grootte van het venster. De afbeelding laat zien hoe vaak het woord in de boeken voorkomt, de grootste aantallen in de rij, en de acht dichtstbijzijnde buren. Probeer *deur* met elke grootte van het venster: vanaf welke grootte staat *open* bij de acht dichtstbijzijnde buren? Probeer daarna *zeide*, *hand*, of een eigen woord, in de spelling van de boeken.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_word = mo.ui.text(value=C.EXPLORE_WORD[L.lang], label="Woord")
    explore_size = mo.ui.dropdown(options=["1", "2", "5", "10"], value="2", label="Grootte van het venster")
    mo.hstack([explore_word, explore_size], justify="start", gap=1.5)
    return explore_size, explore_word


@app.cell(hide_code=True)
def explore_look(explore_size, explore_word):
    mo.Html(C.explore_picture(explore_word.value, explore_size.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Met een venster van 2 zijn de dichtstbijzijnde buren van *deur* de woorden *zon*, *eerste*, *bevolking*, *wereld* en *regeering*: woorden die, net als *deur*, het vaakst bij *de* staan. *open* staat 18 keer in de rij van *deur*, maar komt op plaats 159 bij de buren: *open* staat naast *deur*, niet op dezelfde plaatsen. Met een venster van 5 komt *open* op plaats 38, en met een venster van 10 op plaats 4, na *dames*, *heeren* en *hand*: een woord dat naast *deur* staat, komt bij een lang venster wel dichtbij. Jurafsky en Martin beschrijven hetzelfde: kortere vensters geven vectoren die wat meer over de grammatica zeggen, omdat de informatie uit de woorden er direct naast komt. Bij vectoren uit lange vensters zijn de woorden met de hoogste cosine similarity vaak woorden over hetzelfde onderwerp, die niet op het doelwoord lijken. De grootte ligt volgens Jurafsky en Martin meestal tussen 1 en 10 woorden aan elke kant van het doelwoord. Met een venster van 10 lopen de rijen nog voller met gewone woorden: de acht dichtstbijzijnde buren van *deur* liggen allemaal tussen 0,935 en 0,945.

    ### Wat de rijen missen

    Woorden met een tegenovergestelde betekenis verkeren in hetzelfde gezelschap. Van alle rijen lijkt de rij van *koud* het meest op de rij van *warm*; bij *wit* komt *zwart* op plaats 2, en bij *gelukkig* komt *ongelukkig* op plaats 7. Een antoniem is een woord met de tegenovergestelde betekenis. Peter Turney en Patrick Pantel schrijven in een overzicht uit 2010 dat antoniemen meestal niet als gelijkend gelden, maar in een ander opzicht sterk op elkaar lijken: warm en koud zijn allebei een soort temperatuur, zwart en wit allebei een soort kleur, hard en zacht allebei een soort geluid. Nikola Mrkšić en collega's wijzen er in 2016 op dat zelfs betere methoden synoniemen, woorden met dezelfde betekenis, meestal niet van antoniemen kunnen onderscheiden: woorden als *oost* en *west*, of *duur* en *goedkoop*, staan in bijna dezelfde contexten, en daardoor geven distributionele modellen zulke woorden sterk gelijkende vectoren.

    Woorden kunnen ook bij elkaar horen zonder op elkaar te lijken. Koffie lijkt volgens Jurafsky en Martin niet op een kop: de twee hebben vrijwel niets gemeen, en toch horen koffie en kop duidelijk bij elkaar. Een kort venster vindt vaker woorden van dezelfde soort, een lang venster vaker woorden over hetzelfde onderwerp, zoals *deur* en *open* lieten zien.

    De rijen kennen alleen deze boeken. De dichtstbijzijnde buur van *regent* is *javaan*, en daarna komt *kontroleur*: woorden uit *Max Havelaar*, dat op Java speelt, waar een regent een Javaanse bestuurder is.

    En kale aantallen laten de meest voorkomende woorden beslissen. Kies willekeurig 3.000 paren doelwoorden, en meer dan de helft van de paren heeft een cosine similarity boven 0,35: door *de*, *en* en *van* hebben zelfs woorden die niets met elkaar te maken hebben een groot deel van de rij gemeen. Turney en Pantel beschrijven de uitweg, weging: laat verrassende gebeurtenissen zwaarder meetellen en verwachte gebeurtenissen lichter. In het voorbeeld van Turney en Pantel zeggen de contexten *ontleden* en *uitroeien* meer over hoe sterk *muis* en *rat* op elkaar lijken dan de contexten *hebben* en *houden van*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 3. Dichtbij of ver weg?

    Elke regel noemt twee woorden uit de Nederlandse boeken. Zet de regel onder het kopje dat zegt waar het eerste woord staat tussen de buren van het tweede, met de rijen van deze les: een venster van 2 en kale aantallen.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_neighbors = L.sort("neighbors")
    sort_neighbors
    return (sort_neighbors,)


@app.cell(hide_code=True)
def check_sort(sort_neighbors):
    result_sort = L.check("neighbors", sort_neighbors)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    Jurafsky en Martin geven de twee soorten gezelschap een naam. Twee woorden komen in de eerste orde samen voor (in het Engels first-order co-occurrence, of syntagmatic association) als de twee meestal dicht bij elkaar staan: het Engelse *wrote* bij *book* of *poem*. Twee woorden komen in de tweede orde samen voor (second-order co-occurrence, of paradigmatic association) als de twee vergelijkbare buren hebben: *wrote* bij *said* of *remarked*. De rijen van deze les vinden de tweede soort: woorden met vergelijkbare buren.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een document als vector](/course/open/nlp-document-vectors?lang=nl) gaf elke zin een sparse rij met aantallen woorden, en vergeleek twee rijen met de cosine similarity. Deze les gaf elk woord zo'n rij, met de woorden eromheen als kolommen: een co-occurrence matrix, gebouwd op de distributionele hypothese. Woorden van dezelfde soort krijgen vergelijkbare rijen, en woorden met een tegenovergestelde betekenis ook; kale aantallen laten *de*, *en* en *van* te veel bepalen. De volgende les van het domein, *Word embeddings: buren en analogieën*, werkt met word embeddings: korte vectoren van enkele tientallen tot enkele honderden getallen, geleerd uit veel meer tekst.

    **Een voorbeeld uit het dagelijks leven.** Een zoekmachine die ook pagina's over een *bank* vindt als je *sofa* typt, kan uit tekst alleen leren dat de twee woorden in hetzelfde gezelschap verkeren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rows, result_sort, result_window):
    L.wrap_up(result_window, result_rows, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
