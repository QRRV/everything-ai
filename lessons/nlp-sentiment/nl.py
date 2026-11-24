# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-sentiment. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sentiment", "nl")
    C = L.checks   # de zinnen uit recensies in het Engels en het Nederlands, de classifier, de recensies van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_not():
    quick_not = L.choice(
        "quick-not",
        question="In *Logistische regressie op woorden* heeft *I did not expect this to be so good!* (Ik had niet verwacht dat dit zo goed zou zijn!) het label positief, en krijgt de zin een kans op positief van 0,423. Waar komt dat door?",
        options={
            "unknown": "*good* staat in geen enkele trainingszin, dus het woord telt niet mee",
            "right": "*not* duwt omlaag, wat er ook naast staat",
            "start": "De classifier leest alleen de eerste paar woorden van een zin, en de eerste woorden hier zijn neutraal",
            "mark": "Een uitroepteken verlaagt de kans van elke zin",
        },
        correct="right",
        why={
            "right": "*not* heeft een weight van −1,78, en een weight hoort bij één woord, waar het woord ook staat: *not* telt hier −1,852 op, meer dan de +1,505 van *good*. De classifier ziet nooit dat *not* bij *expect* hoort. Deze les kijkt naar een woord dat wel verandert wat de woorden eromheen betekenen: *but*, in het Nederlands *maar*.",
            "unknown": "*good* staat in veel trainingszinnen, en telt hier +1,505 op. Welk woord trekt de zin de andere kant op? Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "start": "De classifier telt voor elk woord van de zin een getal op, waar het woord ook staat. Welk woord telt het meest op richting negatief? Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "mark": "De woorden zijn reeksen letters, dus de classifier ziet een uitroepteken nooit, en het teken telt niet mee. Welk woord trekt de zin omlaag? Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
        },
    )
    quick_not
    return (quick_not,)


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="De TF-IDF-waarde van een woord in een tekst is het aantal keer dat het woord in de tekst staat, keer $\\log_{10}$ van 2.400 gedeeld door het aantal trainingszinnen met het woord. Voor *great* is de logaritme 1,15. Wat is de TF-IDF-waarde van *great* in *Great food, great service.* (Lekker eten, goede service.)?",
        options={
            "one": "1,15, zoals in elke tekst met het woord",
            "half": "0,575, omdat de tekst uit twee delen bestaat",
            "right": "2,30",
            "square": "1,32, dat is 1,15 keer 1,15, voor twee keer hetzelfde woord",
        },
        correct="right",
        why={
            "right": "*great* staat er twee keer in, dus de waarde is 2 × 1,15 = 2,30. In deze les wordt een hele recensie één tekst, en een woord dat in drie zinnen staat, telt drie keer.",
            "one": "De waarde telt hoe vaak het woord in de tekst staat. Hoe vaak staat *great* hier? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "half": "De waarde hangt niet af van de delen van een tekst, alleen van hoe vaak het woord erin staat. Hoe vaak staat *great* hier? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
            "square": "Het aantal vermenigvuldigt de logaritme één keer. Hoe vaak staat *great* in deze tekst? Zie [TF-IDF: woorden wegen](/course/open/nlp-tf-idf?lang=nl).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voordat je een waterkoker koopt of een tafel reserveert, lees je wat anderen erover schreven. Een winkel die wil laten zien wat duizenden kopers vinden, kan niet elke recensie met de hand lezen. Jurafsky en Martin noemen deze taak in *Speech and Language Processing* **sentiment analysis**: uit een tekst halen of de schrijver positief of negatief is over iets. De eenvoudigste vorm is volgens Jurafsky en Martin een keuze uit twee labels, en de woorden van een recensie geven daarbij al veel prijs.

    Je bouwde zo'n classifier in [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl): op de Engelse zinnen geeft de classifier 491 van de 600 achtergehouden zinnen het eigen label. Maar een recensie is meer dan één zin, en niet alle woorden wijzen dezelfde kant op. Pang, Lee en Vaithyanathan, die in 2002 filmrecensies met de computer sorteerden, beschrijven recensies met een bewust gebroken verwachting: de schrijver bouwt eerst iets op en zet er dan iets tegenover. Volgens Pang en de anderen staan in zulke recensies veel woorden die het tegendeel zeggen van de hele recensie.

    Deze les gaat van de labels van de zinnen naar het label van een recensie, en kijkt naar een woord dat een zin kan omdraaien: *maar*. De les doet dat in het Nederlands: de course vertaalde de 3.000 zinnen naar het Nederlands en trainde dezelfde classifier op de vertaling. Op de Nederlandse zinnen krijgt de classifier er 480 van de 600 goed.

    **Een voorbeeld uit het dagelijks leven.** Een kennis komt terug van vakantie en vertelt: het hotel was prachtig, het strand was mooi en het eten was goed, maar het heeft de hele week geregend. Drie dingen waren goed en één niet, en toch weet je dat de vakantie tegenviel.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_but = L.guess(
        "guess",
        question="65 van de 600 achtergehouden Nederlandse zinnen hebben het woord *maar*, en de classifier geeft 57 van deze 65 het eigen label. Wat gebeurt er als de classifier alleen de woorden na de laatste *maar* telt, en de woorden ervóór weglaat?",
        options={
            "all": "Alle 65 goed",
            "more": "Meer zinnen goed: ongeveer 60",
            "same": "Niets verandert: 57",
            "right": "Minder zinnen goed: 55",
        },
        reveal="**Minder zinnen goed: 55.** Twee zinnen gaan goed die fout waren, zoals *Ik ben echt dol op sushi, maar ik vond Kabuki te duur, te hip en te weinig service.* Maar vier zinnen die goed waren, gaan fout, zoals *Alle andere hoesjes die ik heb geprobeerd vallen meestal na een paar maanden uit elkaar, maar deze lijkt het lang vol te houden.*: de lof zit in de vergelijking met de andere hoesjes, en zonder het eerste deel valt de vergelijking weg. Een regel die een halve zin weggooit, gooit ook weg wat de eerste helft zegt.",
    )
    guess_but
    return (guess_but,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Je kunt sentiment op meer dan één niveau lezen. Bing Liu onderscheidt in *Sentiment Analysis and Opinion Mining* (2012) drie niveaus waarop sentiment analysis vooral onderzocht is. Op het **documentniveau** krijgt een hele tekst, zoals een recensie, één label: positief of negatief. Op het **zinsniveau** krijgt elke zin een label: positief, negatief of neutraal, en neutraal betekent volgens Liu meestal dat de zin geen mening geeft. Het derde niveau kijkt naar onderdelen van een product, zoals de accu of het scherm; deze les blijft bij de eerste twee.

    De 3.000 zinnen van de course hebben twee labels en geen neutraal label. Kotzias, Denil, de Freitas en Smyth, die de zinnen in 2015 verzamelden, kozen voor elk van drie websites 1.000 zinnen en labelden de zinnen met de hand, de helft positief en de helft negatief. Volgens de pagina van de set was het de bedoeling dat er geen neutrale zinnen tussen zaten. De set bewaart de zinnen, niet de recensies waar de zinnen uit kwamen. Dus schreef de course 18 korte recensies, over producten, films en restaurants, van drie tot vijf zinnen, en gaf elke zin en elke recensie een label.

    Van de labels van de zinnen naar het label van de recensie, op twee eenvoudige manieren:

    - **Majority voting**: geef elke zin een label, en geef de recensie het label dat de meeste zinnen hebben. Bij een gelijke stand laat deze les de laatste zin beslissen. Becker en Aharonson (2010) lieten lezers 16 recensies beoordelen, en ook de laatste en de tweede zin van elke recensie los. Voorzichtig, omdat de steekproef klein is, concluderen Becker en Aharonson dat een laatste zin het oordeel beter laat zien dan een tweede zin.
    - **Hele tekst**: zet de zinnen achter elkaar tot één tekst, en geef de hele tekst aan de classifier. Elk woord telt dan de TF-IDF-waarde keer de weight op bij $z$, in welke zin het woord ook staat, dus $z$ van de hele tekst is de bias plus wat de woorden van elke zin optellen: $z = b + \sum_s (z_s - b)$, met $z_s$ de $z$ van zin $s$ alleen.

    De afbeelding laat beide zien voor de tweede recensie van de course, over een waterkoker.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Drie zinnen met lof en één klacht: de meerderheid zegt positief, en de recensie is negatief. De hele tekst krijgt het goed, maar net, omdat de laatste zin harder omlaag trekt dan de andere drie samen omhoog duwen; het uitgewerkte voorbeeld hieronder laat de getallen zien. Over alle 18 recensies geeft de meerderheid er 15 het eigen label, en de hele tekst 16.

    **Wat *maar* doet.** De klacht in de recensie van de waterkoker begint met *Maar*. Volgens Polanyi en Zaenen (2004) kiest een schrijver met *but* (maar) ervoor om het tweede standpunt zwaarder te laten wegen. Socher en anderen (2013) zagen dat het model RNTN uit de voorbeelden leerde dat het sentiment na *but* de doorslag geeft. En Liu waarschuwt dat niet elke *but* een tegenstelling maakt, net zoals niet elke ontkenning iets omdraait: denk aan *not only … but also* (niet alleen … maar ook).

    Een eenvoudige manier om de woorden na *maar* meer te laten zeggen: geef elk woord vóór de laatste *maar* de halve weight. In deze les heet dat een **scale** voor elk woord: 0,5 vóór de laatste *maar*, 1 vanaf de *maar*, en de course vermenigvuldigt wat elk woord bij $z$ optelt met de scale van het woord. Van de 65 achtergehouden zinnen met *maar* krijgt de classifier er dan 58 goed in plaats van 57, en van alle 600 zinnen 481 in plaats van 480: één zin gaat goed die fout was, en geen enkele zin gaat fout. Eén zin van de 65 is te weinig om te zeggen dat de regel beter is. Bij de recensies van de course helpt de regel niet. Bij de steak, drie klachten en dan *Maar de steak was perfect, en daar kwamen we voor.*, gaat de kans op positief van 0,074 naar 0,295, en de recensie blijft fout. De hele tekst met scales krijgt 16 van de 18 recensies goed, net als zonder.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **sentiment analysis** | bepalen of een tekst positief of negatief (soms ook neutraal) is over het onderwerp |
        | **documentniveau** | één label voor een hele tekst, zoals een recensie |
        | **zinsniveau** | één label voor elke zin |
        | **majority voting** | een tekst het label geven dat de meeste zinnen hebben; in deze les beslist de laatste zin bij een gelijke stand |
        | **scale** | een getal voor elk woord waarmee de course vermenigvuldigt wat het woord bij $z$ optelt: 0,5 vóór de laatste *maar*, elders 1 |
        | `C.reviews(lang)` | de 18 recensies van de course in het Engels (`"en"`) of het Nederlands (`"nl"`), als lijst van (label, zinnen), met elke zin als (label, tekst) |
        | `C.probability(text, lang)` | de kans op positief die de classifier van de gekozen taal een tekst geeft |
        | `C.whole(sentences)` | de tekst van een recensie: de zinnen achter elkaar, met een spatie ertussen |
        """),
        L.watch_out("De 18 recensies en de labels zijn van de course zelf, geschreven om te laten zien hoe zinnen en een recensie het oneens kunnen zijn. Wat de recensies tellen, laat zien wat er kan gebeuren, niet hoe vaak het gebeurt. De aantallen op de 600 achtergehouden zinnen komen uit echte recensies."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Geef elke zin van een recensie de kans op positief, en het label: positief als de kans 0,5 of meer is.
    2. Majority voting: tel de positieve en de negatieve zinnen. Meer positieve zinnen maken de recensie positief, meer negatieve negatief; bij een gelijke stand beslist de laatste zin.
    3. Hele tekst: zet de zinnen achter elkaar tot één tekst, en geef de hele tekst een kans. Een woord dat in meer zinnen staat, telt elke keer mee.
    4. Met *maar*: geef elk woord een scale, 0,5 vóór de laatste *maar* en 1 vanaf de *maar*, en vermenigvuldig wat elk woord bij $z$ optelt met de scale.
    5. Vergelijk de labels met het label van de recensie, en tel hoeveel recensies elke manier goed krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code neemt de tweede recensie van de course, over een waterkoker, uit `reviews` van de course. Voor elke zin drukt de code de kans op positief af uit `probability` van de course, het label dat de course de zin gaf en de zin zelf. Daarna de kans van de hele recensie als één tekst, gemaakt met `whole` van de course, en het label van de recensie.
    """)
    return


@app.cell
def example_cell():
    label, sentences = C.reviews(L.lang)[1]
    for sentence_label, text in sentences:
        print(round(C.probability(text, L.lang), 3), sentence_label, text)
    print(round(C.probability(C.whole(sentences), L.lang), 3), label)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De classifier geeft elke zin het label dat de course de zin gaf, en toch zegt de meerderheid positief. Elke zin telt $z_s - b$ op bij het geheel: 1,167, 0,498 en 0,686 voor de drie zinnen met lof, samen ongeveer 2,35, en −2,431 voor de laatste. Met de bias van −0,117 is $z$ van de hele tekst −0,197, en de kans op positief 0,451: negatief, zoals de course de recensie labelde, maar net.

    De classifier kent *lekte* niet eens: geen trainingszin heeft dat woord, dus het telt niet mee. De laatste zin trekt omlaag via *ding* (−0,56), *teruggestuurd* (−0,51), *na* (−0,50), *maar* (−0,33), *dus* (−0,31) en *weken* (−0,31): woorden die vaker in negatieve trainingszinnen stonden dan in positieve. Ook *maar* zelf telt hier al negatief mee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_majority():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. majority: één label uit de zinnen

        Schrijf `majority(labels)`, waarbij `labels` het label van elke zin van een recensie bevat, op volgorde: 1 voor positief en 0 voor negatief. Geef 1 terug als meer zinnen positief dan negatief zijn, 0 als meer zinnen negatief zijn, en bij een gelijke stand het label van de laatste zin. Vervang de `...`. De check roept `majority([1, 1, 1, 0])` aan, de labels die de classifier de vier zinnen van de waterkoker geeft, en dat geeft `1`, en `majority([0, 1])`, een gelijke stand, en dat geeft `1`. Daarna eigen gevallen: de gelijke stand andersom, `[1, 0]`; één positieve en één negatieve zin los; twee en twee in beide volgordes; vijf zinnen met meer negatieve; zes met meer positieve; en drie negatieve zinnen vóór een positieve.
        """),
        L.task("Geef een recensie het label van de meeste zinnen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_majority():
    box_majority = L.code_box("majority", lines=12)
    box_majority
    return (box_majority,)


@app.cell(hide_code=True)
def check_majority(box_majority):
    result_majority = L.check("majority", box_majority.value)
    result_majority
    return (result_majority,)


@app.cell(hide_code=True)
def hints_majority():
    L.hints(
        "majority",
        [
            "Twee tellers, `positive` en `negative`, allebei 0 aan het begin. Een loop `for label in labels:` met een `if` en een `else`, zodat elk label 1 optelt bij een van de twee. Na de loop een `if` die 1 teruggeeft, een `if` die 0 teruggeeft, en nog één `return` voor een gelijke stand.",
            "Bij een gelijke stand beslist de laatste zin. Welke positie wijst altijd naar het laatste item van een lijst, hoe lang de lijst ook is?",
            "`    return labels[___]`",
        ],
        why="`positive = 0` en `negative = 0`; `for label in labels:` met `if label == 1:` en daaronder `positive = positive + 1`, en `else:` met `negative = negative + 1`. Na de loop: `if positive > negative:` met `return 1`, `if negative > positive:` met `return 0`, en als laatste `return labels[-1]`. De voorbeelden geven `1` en `1`.",
    )
    return


@app.cell(hide_code=True)
def text_after_majority():
    mo.md(r"""
    De meerderheid gaat mis waar een recensie omslaat. Bij de waterkoker stemmen drie zinnen met lof de klacht weg, en bij de steak winnen drie klachten het van *Maar de steak was perfect, en daar kwamen we voor.* Bij de film met goedkope effecten geeft de classifier *Maar de personages zijn zo sympathiek dat het me niets kon schelen.* een kans op positief van 0,188, en dan krijgt de meerderheid, twee tegen één, de recensie niet goed.

    Kotzias en de anderen beschrijven een recensie met een 8 van de 10, terwijl de meeste woorden negatief zijn: wie gewoon woorden of zinnen telt, zit er dan naast.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scales():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. scales: halve weight vóór de laatste *maar*

        Schrijf `scales(words, word)`, waarbij `words` een lijst is met de woorden van een tekst, in kleine letters, en `word` het woord dat een zin omdraait: `"maar"` in het Nederlands, `"but"` in het Engels. Geef een lijst terug met één getal voor elk woord: `0.5` voor elk woord vóór het laatste `word`, en `1.0` voor dat `word` en elk woord erna. Als `word` niet in de lijst staat, krijgt elk woord `1.0`. De course vermenigvuldigt dan wat elk woord bij $z$ optelt met het getal van het woord. Vervang de `...`. De check roept `scales(["good", "food", "but", "slow", "service"], "but")` aan, en dat geeft `[0.5, 0.5, 1.0, 1.0, 1.0]`. Daarna eigen gevallen: twee woorden zonder *but*; *but* als eerste woord; twee keer *but*; *but* als laatste woord; vijf Nederlandse woorden met *maar*; drie woorden met *but* en het woord `"maar"`; een lege lijst; en de elf woorden van de achtergehouden zin *This item worked great, but it broke after 6 months of use.* (Dit product werkte geweldig, maar na 6 maanden gebruik ging het kapot.)
        """),
        L.task("Geef elk woord de scale: half vóór de laatste maar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scales():
    box_scales = L.code_box("scales", lines=12)
    box_scales
    return (box_scales,)


@app.cell(hide_code=True)
def check_scales(box_scales):
    result_scales = L.check("scales", box_scales.value)
    result_scales
    return (result_scales,)


@app.cell(hide_code=True)
def hints_scales():
    L.hints(
        "scales",
        [
            "Eerst de positie van het laatste `word`: begin met `last = -1`, ga de posities langs met `for i in range(len(words)):`, en zet `last = i` telkens als `words[i] == word`. Dan een nieuwe lijst `out`, een tweede loop over de posities, en een `if` en een `else` die elk `out.append(...)` doen met `0.5` of `1.0`.",
            "Het woord zelf en elk woord erna houden `1.0`. Welke vergelijking van een positie `i` met `last` zegt dat het woord op `i` vóór het laatste `word` komt?",
            "`        if i ___ last:`",
        ],
        why="`last = -1`; `for i in range(len(words)):` met `if words[i] == word:` en daaronder `last = i`. Dan `out = []`; `for i in range(len(words)):` met `if i < last:` en daaronder `out.append(0.5)`, en `else:` met `out.append(1.0)`; na de loop `return out`. Zonder `word` blijft `last` −1, dus krijgt elk woord `1.0`. Het voorbeeld geeft `[0.5, 0.5, 1.0, 1.0, 1.0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_scales():
    mo.md(r"""
    Met de scales van de course gaat één achtergehouden zin goed die fout was: *Ik ben echt dol op sushi, maar ik vond Kabuki te duur, te hip en te weinig service.* (van 0,759 naar 0,448). Geen enkele zin gaat fout. Bij de recensies van de course blijven de steak en de film met goedkope effecten fout: bij de film leest de classifier de zin na *Maar* al verkeerd.

    *Maar* doet in het Nederlands meer dan een zin omdraaien: in *De bezorging duurde maar twee dagen.* betekent het woord slechts. In 16 van de 65 achtergehouden zinnen met *maar* staat in de Engelse zin geen *but*: daar betekent *maar* meestal slechts, alleen of zelfs, zoals in *zegt niemand zelfs maar hallo*. Een regel die op het woord *maar* let, geeft dan woorden de halve weight waar niets omslaat. In het Engels gaan met de scales drie zinnen goed en één fout (38 goed in plaats van 36 van de 50), en daar gaat de steak wel goed (van 0,447 naar 0,569).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 3. Waar staat het oordeel?

    Acht achtergehouden zinnen met *maar*, elk met het label. Zet elke zin waar het oordeel, positief of negatief, staat: na de *maar*, ervóór, of in beide delen, waar *maar* niets omdraait.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_judgment = L.sort("where-judgment")
    sort_judgment
    return (sort_judgment,)


@app.cell(hide_code=True)
def check_sort(sort_judgment):
    result_sort = L.check("where-judgment", sort_judgment)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: een eigen recensie

    Schrijf een recensie van een paar zinnen in het Nederlands, over een product, een film of een maaltijd. De afbeelding geeft elke zin de kans op positief, dan de meerderheid, de hele recensie als één tekst, en de hele recensie met halve weight vóór de laatste *maar*. Probeer een recensie die drie dingen prijst en eindigt met *Maar na een week was het kapot.*, en dan een recensie die drie keer klaagt en eindigt met *Maar het eten was het waard.*
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.text_area(value=C.TRY_START[L.lang], label="Jouw recensie", rows=4, full_width=True)
    try_box
    return (try_box,)


@app.cell(hide_code=True)
def try_look(try_box):
    mo.Html(C.try_picture(try_box.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De classifier van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) geeft één zin tegelijk een label. Een recensie heeft meer zinnen: een meerderheid telt de labels, en de hele tekst laat elk woord optellen bij één $z$, zodat één zin met zware woorden meer kan wegen dan een paar lichte zinnen. Op de 18 recensies van de course krijgt de meerderheid er 15 goed en de hele tekst 16. Een woord als *maar* verandert wat de woorden eromheen waard zijn. Scales die de woorden vóór de laatste *maar* halveren, helpen waar het oordeel erna komt, en trekken de verkeerde kant op waar het oordeel ervóór komt of waar *maar* niets omdraait, en in het Nederlands betekent *maar* ook vaak slechts: op de achtergehouden zinnen met *maar* komt de regel uit op 58 goed tegen 57. De volgende les, *Een lijst met goede en slechte woorden*, geeft elk woord een score uit hoe vaak het woord in positieve en in negatieve recensies voorkomt, en draait een score om na een ontkenning.

    **Een voorbeeld uit het dagelijks leven.** Een referentie die iemand *een prettige collega, maar vaak te laat* noemt, lees je als een brief over te laat komen, hoeveel vriendelijke zinnen er ook aan voorafgingen. Wie zo'n brief leest, weegt de zinnen in plaats van de zinnen te tellen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_majority, result_scales, result_sort):
    L.wrap_up(result_majority, result_scales, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
