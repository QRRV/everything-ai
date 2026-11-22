# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-edit-distance. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("nlp-edit-distance", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl), en een over [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="Wat is de shape van `np.array([[0, 1, 2, 3], [1, 1, 2, 3]])`?",
        options={
            "flip": "`(4, 2)`",
            "right": "`(2, 4)`",
            "flat": "`(8,)`: één lange rij",
        },
        correct="right",
        why={
            "right": "Twee binnenste lijsten geven twee rijen, en vier getallen per lijst geven vier kolommen: de shape noemt eerst de rijen. Deze les bewaart de tabel met afstanden in zo'n array, met een rij voor elk beginstuk van het ene woord en een kolom voor elk beginstuk van het andere.",
            "flip": "De shape noemt eerst de rijen, dan de kolommen: twee binnenste lijsten zijn twee rijen, met elk vier getallen. Zie [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
            "flat": "Een lijst van lijsten wordt een rooster met rijen en kolommen, geen lange rij. Zie [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl).",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_case():
    quick_case = L.choice(
        "quick-case",
        question="Van welk paar maakt `lower()` alleen al één string?",
        options={
            "accent": "*Résumé* en *resume*",
            "right": "*Paris* en *paris*",
            "plural": "*Boeken* en *boek*",
        },
        correct="right",
        why={
            "right": "`lower()` maakt *paris* van *Paris*. Deze les vergelijkt de letters zelf, zonder iets te veranderen: zonder die stap verschillen *Paris* en *paris* in één letter.",
            "accent": "`lower()` maakt *résumé* van *Résumé*: de accenten blijven, dus het is nog steeds niet *resume*. De accenten eraf halen is een aparte stap; zie *Accenten* in [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
            "plural": "`lower()` verandert letters, geen woorden: *boeken* houdt de uitgang *-en*, dus het is nog steeds niet *boek*. Zie [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
        },
    )
    quick_case
    return (quick_case,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Iedereen maakt tikfouten: wie *resturant* typt, bedoelt *restaurant*. Een programma dat hele strings vergelijkt, ziet twee verschillende woorden. Om het bedoelde woord voor te stellen, heeft het programma een getal nodig dat zegt hoe ver twee strings uit elkaar liggen. Daniel Jurafsky en James Martin geven er een in het hoofdstuk *Words and Tokens* van *Speech and Language Processing*: de minimale edit distance tussen twee strings is het kleinste aantal bewerkingen, zoals invoegen, weghalen en vervangen, dat nodig is om de ene string in de andere te veranderen. Van *resturant* naar *restaurant* is één bewerking: er komt een *a* bij. Spraakherkenning wordt met dezelfde telling gemeten, over woorden in plaats van letters: hoeveel woorden van wat een programma hoorde moeten veranderen om te krijgen wat er gezegd werd. Deze les berekent de **edit distance** met een tabel waarin elke cel gebouwd wordt uit de cellen ernaast, een manier van werken die **dynamic programming** heet.

    **Een voorbeeld uit het dagelijks leven.** Een zoekvak dat *Bedoelde je restaurant?* vraagt als je *resturant* typt, heeft een woord gevonden dat één bewerking verwijderd is van wat je typte.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Lijn twee woorden van negen letters uit, zoals *intention* en *execution*: zet bij elke stap een letter van het ene woord tegenover een letter van het andere, of zet een letter van één woord tegenover een gat. Hoeveel verschillende manieren zijn er om de twee woorden uit te lijnen?",
        options={
            "few": "Ongeveer 1.000",
            "some": "Ongeveer 40.000",
            "right": "Ongeveer 1.500.000",
            "lots": "Ongeveer 300.000.000",
        },
        reveal="**1.462.563**: ongeveer anderhalf miljoen. Voor *phone* en *power*, met elk vijf letters, zijn het er al 1.683, en voor twee woorden van tien letters 8.097.453. Jurafsky en Martin schrijven dat de ruimte van alle mogelijke bewerkingen enorm is, zodat je niet zomaar alles kunt doorzoeken. De tabel van deze les heeft voor *intention* en *execution* maar 10 × 10 = 100 cellen nodig, één voor elk paar beginstukken, van geen letters tot alle negen, en vindt de goedkoopste van al die manieren.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Drie soorten bewerkingen

    Een **bewerking** verandert een string met één letter: invoegen zet er een letter bij, weghalen haalt er een weg, en vervangen zet een letter op de plaats van een andere. Van *phone* maak je *power* in drie bewerkingen: haal de *h* weg, vervang de *n* door een *w*, en voeg aan het eind een *r* in. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze geven een voorbeeld in het hoofdstuk *Dictionaries and tolerant retrieval* van *Introduction to Information Retrieval*: de edit distance tussen *cat* en *dog* is drie. Elke letter van *cat* wordt vervangen door een letter van *dog*.

    Als elke bewerking 1 kost, heet de edit distance ook de **Levenshtein distance**, naar Vladimir Levenshtein, die de afstand in 1965 beschreef in een paper over binaire codes: reeksen nullen en enen die over een verbinding gaan die een teken kan kwijtraken, er een kan toevoegen, of een 0 in een 1 of een 1 in een 0 kan veranderen. Niet elk boek telt op dezelfde manier. Jurafsky en Martin laten eerst elke bewerking 1 kosten, en dan is de Levenshtein distance tussen *intention* en *execution* 5. Ze schrijven ook dat Levenshtein een tweede versie voorstelde, waarin invoegen en weghalen 1 kosten en vervangen niet mag. Vervangen kost dan weghalen plus invoegen, dus 2, en de afstand tussen *intention* en *execution* wordt 8. In deze les kost elke bewerking 1; in de demo kun je 2 proberen.

    ### Twee woorden uitlijnen

    Een **alignment** zet het ene woord boven het andere, zodat je ziet welke letter bij welke hoort. Volgens Jurafsky en Martin is een alignment een koppeling tussen stukken van twee reeksen. De les schrijft een gat als `-`:

    ```
    p h o n e -
    p - o w e r
    ```

    Waar beide regels dezelfde letter hebben, verandert er niets. Een `-` in de onderste regel is weghalen: de *h* gaat eruit. Een `-` in de bovenste regel is invoegen: de *r* komt erbij. Twee verschillende letters zijn vervangen: *w* in plaats van *n*. Elke alignment van twee woorden heeft een aantal bewerkingen, en de edit distance is het kleinste van die aantallen. Alle vijf de letters van *phone* weghalen en dan *power* typen is ook een alignment, met tien bewerkingen.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_table():
    mo.md(r"""
    ### Een tabel met kleinere problemen

    Elke alignment proberen duurt veel te lang, zoals de schatting liet zien. De uitweg: los eerst kleinere problemen op en bewaar de antwoorden. Schrijf $D[i, j]$ voor de edit distance tussen de eerste $i$ letters van het ene woord en de eerste $j$ letters van het andere: voor *phone* en *power* is $D[2, 3]$ de afstand van *ph* naar *pow*, en $D[5, 5]$ het antwoord. De kleinste problemen zijn makkelijk. Van $i$ letters naar geen letters kost $i$ keer weghalen, schrijven Jurafsky en Martin, en van geen letters naar $j$ letters kost $j$ keer invoegen.

    Elke andere cel volgt uit drie cellen ernaast. Kijk naar de laatste plek van de goedkoopste alignment van de eerste $i$ letters met de eerste $j$ letters. Daar staat een van drie dingen:

    - de $i$-de letter van het eerste woord boven een gat: weghalen, dus de cel erboven plus 1, $D[i-1, j] + 1$;
    - een gat boven de $j$-de letter van het tweede woord: invoegen, dus de cel links plus 1, $D[i, j-1] + 1$;
    - de twee letters boven elkaar: de cel linksboven, $D[i-1, j-1]$, plus 0 bij twee gelijke letters en plus 1 bij twee verschillende.

    De cel krijgt de kleinste van de drie. Vul de tabel rij na rij, vanaf linksboven, en elke cel die je invult, heeft alleen cellen nodig die al gevuld zijn. De cel rechtsonder bevat de edit distance.

    Jurafsky en Martin noemen deze manier van werken dynamic programming: een familie algoritmen, voor het eerst beschreven door Bellman (1957), die een probleem oplossen met een tabel, door de oplossingen van deelproblemen te combineren. Richard Bellman schreef later in de autobiografie *Eye of the Hurricane* (1984) dat de naam langs een minister van Defensie moest die het woord *research* (onderzoek) haatte en vreesde. Het woord *dynamic* kun je volgens Bellman niet in een slechte betekenis gebruiken, en zelfs een congreslid zou niets tegen de naam *dynamic programming* kunnen hebben. Stuart Dreyfus citeert die passages in *Richard Bellman on the birth of dynamic programming* (2002).

    ### De weg terug

    De tabel geeft de afstand, niet de bewerkingen. De bewerkingen vind je door in de laatste cel te beginnen en terug te gaan langs de weg waarlangs de kleinste waarde kwam: naar de cel erboven, links, of linksboven. Jurafsky en Martin noemen dat een **backtrace**: je begint in de laatste cel, in de laatste rij en kolom, en volgt de wijzers terug door de tabel. Een wijzer zegt uit welke cel een waarde kwam. Een stap omhoog is weghalen, een stap naar links invoegen, en een stap linksboven houdt een letter of vervangt een letter. Elke volledige weg van de laatste cel naar de eerste is een goedkoopste alignment. Twee woorden kunnen meer dan één goedkoopste alignment hebben: *phone* en *power* hebben er één, *intention* en *execution* zeven, als elke bewerking 1 kost.
    """)
    return


@app.cell(hide_code=True)
def table_picture():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Eén idee, vaak gevonden

    Jurafsky en Martin schrijven dat Wagner en Fischer (1974) het algoritme een naam gaven, maar dat veel mensen het los van elkaar ontdekten. Manning en collega's schrijven hun edit distance toe aan Levenshtein (1965), en het algoritme aan Wagner en Fischer (1974). Robert Wagner en Michael Fischer schreven in de samenvatting van hun paper *The String-to-String Correction Problem* dat hun algoritme tijd kost in verhouding tot het product van de lengtes van de twee strings. Dat product is ongeveer de grootte van de tabel. Gonzalo Navarro noemt in een overzicht uit 2001 anderen die dezelfde methode vonden, onder wie Vintsyuk in 1968, voor spraak, en Needleman en Wunsch in 1970, die de aminozuurvolgorde van eiwitten vergeleken.

    ### Andere kosten en verwisselingen

    Niet elke bewerking is even waarschijnlijk. Manning en collega's geven als voorbeeld dat je *s* vervangen door *p* zwaarder kunt laten wegen dan *s* vervangen door *a*, omdat de *a* op het toetsenbord dichter bij de *s* ligt. De tabel werkt met andere kosten precies zo: alleen wat elke stap optelt, verandert.

    Mensen verwisselen ook twee letters, zoals in *acress* voor *caress*. Met invoegen, weghalen en vervangen kost een verwisseling twee bewerkingen. De methode van Fred Damerau, in *A technique for computer detection and correction of spelling errors* (1964), gaat ervan uit dat een woord dat niet in het woordenboek staat, hoogstens één fout heeft: een verkeerde, ontbrekende of extra letter, of één verwisseling. Een **verwisseling** van twee buurletters heet in het Engels een *transposition*, en een edit distance die zo'n verwisseling als één bewerking telt, heet de Damerau-Levenshtein distance. Jurafsky en Martin schrijven in de bijlage *Spelling Correction and the Noisy Channel* dat het merendeel van de spelfouten uit een verandering van één letter bestaat. Hun voorbeeld *acress* ligt op één bewerking van *actress*, *across* en *acres*, en met een verwisseling ook op één van *caress*.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **bewerking** | een verandering van één letter: invoegen, weghalen of vervangen |
        | **edit distance** | het kleinste aantal bewerkingen dat de ene string in de andere verandert: *phone* en *power* liggen 3 uit elkaar |
        | **Levenshtein distance** | de edit distance als elke bewerking 1 kost; sommige boeken laten vervangen 2 kosten |
        | **alignment** | twee strings boven elkaar, met `-` voor een gat, zodat je ziet welke letter bij welke hoort |
        | **dynamic programming** | een groot probleem oplossen door eerst kleinere op te lossen en de antwoorden in een tabel te bewaren |
        | **backtrace** | terug door de tabel gaan, van de laatste cel naar de eerste, om de bewerkingen van een goedkoopste alignment te vinden |
        | **verwisseling** | twee buurletters van plaats laten wisselen, zoals *hte* voor *het*; in het Engels transposition; de Damerau-Levenshtein distance telt een verwisseling als één bewerking |
        | **word error rate** | de bewerkingen tussen wat er gezegd werd en wat een programma hoorde, geteld in woorden, als deel van het aantal gezegde woorden |
        | `np.zeros((3, 4), dtype=int)` | een array van 3 rijen en 4 kolommen met gehele getallen, allemaal 0 |
        | `D[i, j] = v` | zet de waarde `v` in rij `i` en kolom `j` van de array `D` |
        | `min(a, b, c)` | de kleinste van de waarden tussen de haakjes: `min(3, 1, 2)` is `1` |
        """),
        L.watch_out("In andere lessen van de course betekent *alignment* iets anders: zorgen dat een AI-systeem de doelen nastreeft die mensen bedoelen. Hier is een alignment twee strings boven elkaar. En zeg bij een edit distance wat vervangen kost: de tabel van Jurafsky en Martin gebruikt 2, deze les 1."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Maak een tabel met een rij voor elk beginstuk van het eerste woord, van geen letters tot allemaal, en een kolom voor elk beginstuk van het tweede woord.
    2. Vul de eerste kolom met 0, 1, 2, ... (weghalen) en de eerste rij met 0, 1, 2, ... (invoegen).
    3. Vul elke andere cel, rij na rij, met de kleinste van drie: de cel erboven plus 1, de cel links plus 1, en de cel linksboven plus 0 bij dezelfde letter of 1 bij een andere.
    4. Lees de edit distance af in de laatste cel, en ga langs de weg terug om de bewerkingen te zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder rekent één cel van de tabel van *phone* en *power* uit: rij 4 en kolom 3, de afstand van *phon* naar *pow*. De drie cellen eromheen komen uit de tabel hierboven.
    """)
    return


@app.cell
def example_cell():
    a = "phone"
    b = "power"
    print(a[:4], b[:3])
    print(a[3], b[2], a[3] == b[2])
    up = 2         # rij 3, kolom 3: van pho naar pow
    left = 2       # rij 4, kolom 2: van phon naar po
    diagonal = 1   # rij 3, kolom 2: van pho naar po
    print(min(up + 1, left + 1, diagonal + 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vierde letter van *phone* is `a[3]`, omdat Python vanaf 0 telt, en de derde letter van *power* is `b[2]`. *n* en *w* verschillen, dus de stap linksboven telt ook 1 op. De goedkoopste van de drie wegen is vervangen: 1 + 1 = 2, het getal in de tabel. Oefening 3 doet dit voor elke cel; eerst komen oefening 1 en 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. count_edits: de bewerkingen van een alignment

        Een alignment bestaat uit twee strings van dezelfde lengte, `top` en `bottom`, met `"-"` voor een gat. Schrijf `count_edits(top, bottom)`: ga de plekken langs, en tel invoegen waar `top` een `"-"` heeft, weghalen waar `bottom` er een heeft, en vervangen waar twee verschillende letters boven elkaar staan. Geef een dictionary terug met de drie aantallen onder de keys `"insert"`, `"delete"` en `"substitute"`. Voor `count_edits("phone-", "p-ower")` is de uitkomst `{"insert": 1, "delete": 1, "substitute": 1}`. Vervang de `...`. De check roept dat aan, en daarna eigen gevallen: de alignment van *intention* en *execution* die Jurafsky en Martin laten zien, een zonder bewerkingen, een met alleen invoegen, twee verwisselde letters, een gat aan elke kant, twee lege strings, *kitten* en *sitting*, en een alignment van woorden, als lijsten.
        """),
        L.task("Tel hoe vaak een alignment met `-` voor een gat invoegt, weghaalt en vervangt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-edits", lines=10)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-edits", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-edits",
        [
            r'Begin met `counts = {"insert": 0, "delete": 0, "substitute": 0}` en ga de plekken langs met `for i in range(len(top))`. Vraag op elke plek met `if`, `elif` en `elif` welke soort bewerking het is, tel één op bij dat aantal, zoals in `counts["insert"] = counts["insert"] + 1`, en geef na de loop `counts` terug.',
            r'Vraag naar de gaten voordat je vraagt of de letters verschillen: een gat en een letter verschillen ook. Hoe krijg je op plek `i` wat de bovenste regel daar heeft?',
            r'`        if ___ == "-":`',
        ],
        why=r'`counts` begint met 0 voor elke key; in de loop telt `if top[i] == "-":` invoegen, `elif bottom[i] == "-":` weghalen, en `elif top[i] != bottom[i]:` vervangen; na de loop `return counts`.',
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    Je count_edits werkt net zo goed op lijsten woorden: de laatste alignment in de afbeelding bevat wat iemand zei, boven, en wat een programma hoorde, onder. Geteld in woorden geven de bewerkingen de word error rate van spraakherkenning. Volgens Jurafsky en Martin is de word error rate de standaardmaat om spraakherkenning mee te beoordelen. De word error rate is alle keren invoegen, vervangen en weghalen samen, als deel van het aantal woorden dat gezegd werd: 4 bewerkingen op 7 woorden is 57,1%. Hun eigen voorbeeld, uit een opgenomen telefoongesprek, heeft zes keer vervangen, drie keer invoegen en één keer weghalen op 13 woorden: 76,9%. Omdat invoegen ook telt, merken Jurafsky en Martin op, kan de word error rate boven de 100% uitkomen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_start():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. start_table: de eerste rij en kolom

        Schrijf `start_table(a, b)`: maak met `np.zeros` een tabel `D` met gehele getallen, met een rij voor elk beginstuk van `a`, van geen letters tot allemaal, en een kolom voor elk beginstuk van `b`. Vul de eerste kolom met 0, 1, 2, ..., want `D[i, 0]` is hoe vaak je moet weghalen om de eerste `i` letters van `a` in niets te veranderen, en vul de eerste rij op dezelfde manier, met invoegen. Geef `D` terug. Voor `start_table("phone", "power")` heeft de uitkomst de shape `(6, 6)`, met 0 tot en met 5 in de eerste kolom en in de eerste rij, en verder overal nullen. Vervang de `...`. De check roept dat aan, en daarna eigen gevallen: woorden van verschillende lengte, in beide volgordes, een leeg woord aan elke kant, twee lege woorden, en lijsten woorden.
        """),
        L.task("Maak een tabel met nullen, en vul de eerste kolom en de eerste rij met de afstanden vanaf en naar het lege begin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_start():
    box_start = L.code_box("start-table", lines=7)
    box_start
    return (box_start,)


@app.cell(hide_code=True)
def check_start(box_start):
    result_start = L.check("start-table", box_start.value)
    result_start
    return (result_start,)


@app.cell(hide_code=True)
def hints_start():
    L.hints(
        "start-table",
        [
            "Maak de tabel met `D = np.zeros((rows, columns), dtype=int)`, met het goede aantal rijen en kolommen. Daarna zet één `for` over de rijen `D[i, 0] = i`, en één over de kolommen `D[0, j] = j`. Geef `D` terug.",
            "*phone* heeft vijf letters. Hoeveel beginstukken heeft het woord, als helemaal geen letters er ook een is?",
            r'`    D = np.zeros((___, len(b) + 1), dtype=int)`',
        ],
        why=r'`D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)`, dan `for i in range(len(a) + 1):` met `D[i, 0] = i`, `for j in range(len(b) + 1):` met `D[0, j] = j`, en `return D`.',
    )
    return


@app.cell(hide_code=True)
def text_after_start():
    mo.md(r"""
    De start bevat al een deel van het antwoord: van *phone* naar geen letters kost vijf keer weghalen, onderaan de eerste kolom, en van geen letters naar *power* vijf keer invoegen, aan het eind van de eerste rij. $D[0, 0]$ is 0, want geen letters vragen geen bewerkingen. De grijze cellen zijn voor oefening 3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. edit_distance: vul de tabel

        Schrijf `edit_distance(a, b)`: begin met `D = start_table(a, b)`, de versie van de course, die de check aan je functie geeft. Vul daarna elke andere cel, rij na rij: `i` loopt van 1 tot en met de lengte van `a`, en voor elke `i` loopt `j` van 1 tot en met de lengte van `b`. Elke cel krijgt de kleinste van de drie wegen ernaartoe, zoals in het recept; de letters van rij `i` en kolom `j` zijn de `i`-de letter van `a` en de `j`-de letter van `b`. Geef de laatste cel terug, `D[len(a), len(b)]`. Voor `edit_distance("phone", "power")` is de uitkomst `3`. Vervang de `...`. De check roept dat aan, en daarna eigen gevallen: *intention* en *execution*, *cats* en *fast*, een ontbrekende letter, twee verwisselde letters, hetzelfde woord, een leeg woord aan elke kant, twee lege woorden, twee verschillende letters, en lijsten woorden, waarvoor dezelfde code werkt.
        """),
        L.task("Vul elke cel van de tabel met de goedkoopste van de drie wegen ernaartoe, en geef de laatste cel terug."),
    ])
    return


@app.cell
def start_cell():
    start_table = C.start_table
    print(start_table("pho", "po"))
    return


@app.cell(hide_code=True)
def exercise_distance():
    box_distance = L.code_box("edit-distance", lines=9)
    box_distance
    return (box_distance,)


@app.cell(hide_code=True)
def check_distance(box_distance):
    result_distance = L.check("edit-distance", box_distance.value)
    result_distance
    return (result_distance,)


@app.cell(hide_code=True)
def hints_distance():
    L.hints(
        "edit-distance",
        [
            "Twee `for`-loops, de ene in de andere: `for i in range(1, len(a) + 1)` en `for j in range(1, len(b) + 1)`. Zet in de binnenste loop `cost` op 0 als de letters `a[i - 1]` en `b[j - 1]` gelijk zijn en anders op 1, en vul dan `D[i, j]` met `min` van de drie wegen. Geef na beide loops `D[len(a), len(b)]` terug.",
            "Twee van de drie wegen tellen 1 op. Uit welke cel komt de derde weg, en wat telt de derde weg op?",
            r'`            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, ___)`',
        ],
        why=r'`D = start_table(a, b)`; `for i in range(1, len(a) + 1):` en daarbinnen `for j in range(1, len(b) + 1):`; `cost = 1`, en `cost = 0` als `a[i - 1] == b[j - 1]`; `D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)`; na beide loops `return D[len(a), len(b)]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_distance():
    mo.md(r"""
    Je edit_distance op elk beginstuk van *phone* en *power* maakt dezelfde tabel als de tabel hierboven, en op lijsten woorden geeft je functie de 4 bewerkingen van de word error rate. De edit distance vergelijkt letters, geen betekenissen: *car* (auto) en *cat* (kat) liggen één bewerking uit elkaar, *car* en *automobile* tien. En de berekening kost tijd: de tabel heeft een cel voor elk paar beginstukken, en groeit dus met het product van de twee lengtes, zoals Wagner en Fischer schreven. Eén woord vergelijken met elk woord van een lange woordenlijst wordt zo veel werk. Manning en collega's noemen het buitensporig duur om de edit distance uit te rekenen van een zoekopdracht naar elke term die een zoekmachine kent, en geven trucs om minder woorden te proberen, zoals alleen de termen die met dezelfde letter beginnen als de zoekopdracht, in de hoop dat de eerste letter geen spelfout bevat.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: de tabel van je eigen woorden

    Typ twee woorden en kies wat vervangen kost. De afbeelding vult de tabel, kleurt de weg terug vanuit de laatste cel, en laat de alignment zien die bij die weg hoort. Probeer *intention* en *execution* met beide kosten, een eigen tikfout, of twee woorden die alleen in een hoofdletter verschillen.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_a = mo.ui.text(value="intention", label="Eerste woord")
    demo_b = mo.ui.text(value="execution", label="Tweede woord")
    demo_cost = mo.ui.dropdown(options={"1": "1", "2": "2"}, value="1", label="Vervangen kost")
    mo.vstack([demo_a, demo_b, demo_cost])
    return demo_a, demo_b, demo_cost


@app.cell(hide_code=True)
def demo_look(demo_a, demo_b, demo_cost):
    mo.Html(C.demo_picture(demo_a.value, demo_b.value, demo_cost.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Als vervangen 1 kost, liggen *intention* en *execution* 5 uit elkaar, en zeven alignments komen op die afstand uit. Als vervangen 2 kost, is de afstand 8, zoals in de tabel van Jurafsky en Martin, en 134 alignments komen op die afstand uit: vervangen kost dan evenveel als weghalen plus invoegen, dus de tabel heeft vaak meer goedkoopste wegen naar een cel. De weg in de afbeelding neemt er een van.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Hoeveel bewerkingen?

    Elk vetgedrukt paar bestaat uit twee strings. Reken de edit distance uit in je hoofd, als elke bewerking 1 kost, en zet het paar erbij.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_edits = L.sort("how-many-edits")
    sort_edits
    return (sort_edits,)


@app.cell(hide_code=True)
def check_sort(sort_edits):
    result_sort = L.check("how-many-edits", sort_edits)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl) maakte van *Café* en *cafe* één string; deze les mat hoe ver twee strings uit elkaar liggen die niet gelijk zijn. Je telde de bewerkingen van een alignment, maakte het begin van de tabel in een NumPy-array, zoals in [Van lijsten naar arrays: shape en dtype](/course/open/np-arrays?lang=nl), en vulde de rest met dynamic programming: elke cel is de goedkoopste van drie wegen ernaartoe, gebouwd uit cellen die al gevuld zijn. Dezelfde tabel werkt voor de woorden van een zin, en dat geeft de word error rate van spraakherkenning. De volgende les van het domein, *Een spellingcorrector*, maakt elke string die één of twee bewerkingen van een tikfout af ligt, houdt daarvan de echte woorden over en kiest het woord dat het vaakst voorkomt in een verzameling boeken.

    **Een voorbeeld uit het dagelijks leven.** Een toetsenbord op je telefoon dat *hte* in *het* verandert, heeft een woord gekozen dat dicht bij wat je typte ligt: twee bewerkingen ervandaan, of één als een verwisseling van twee letters als één bewerking telt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_count, result_distance, result_sort, result_start):
    L.wrap_up(result_count, result_start, result_distance, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
