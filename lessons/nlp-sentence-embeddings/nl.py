# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-sentence-embeddings. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sentence-embeddings", "nl")
    C = L.checks   # GloVe, de paren zinnen van de course met de vectoren van BERT-Tiny en Granite, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl) en [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_contextual():
    quick_contextual = L.choice(
        "quick-contextual",
        question="In *Eén vector per woord, of één per gebruik* las BERT-Tiny de 44 zinnen van de course met *bank*. Hoeveel verschillende vectoren kreeg *bank* na het tweede block?",
        options={
            "one": "1, dezelfde in elke zin",
            "two": "2, één per betekenis",
            "right": "44, één per zin",
            "numbers": "128, één per getal",
        },
        correct="right",
        why={
            "right": "Een model met contextual embeddings geeft elk gebruik een eigen vector, gemaakt uit de woorden eromheen: 44 zinnen, 44 vectoren. Deze les gaat een stap verder, naar één vector voor een hele zin.",
            "one": "Dat doet een static embedding zoals GloVe: één vector voor *bank* in elke zin. BERT-Tiny geeft elk gebruik een eigen vector. Zie [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl).",
            "two": "BERT-Tiny kent de betekenissen niet vooraf: het model geeft elk gebruik een eigen vector, gemaakt uit de woorden eromheen, 44 in totaal. De vectoren van de rivierzinnen liggen dichter bij elkaar dan bij de vectoren van de geldzinnen, maar elke vector staat op zichzelf.",
            "numbers": "Elke vector heeft 128 getallen, maar dat is de lengte, niet het aantal vectoren: elk gebruik van de 44 krijgt een eigen vector.",
        },
    )
    quick_contextual
    return (quick_contextual,)


@app.cell(hide_code=True)
def quick_check_attention():
    quick_attention = L.choice(
        "quick-attention",
        question="Wat geeft attention elke plek, in *Attention: elk woord kijkt naar de andere*?",
        options={
            "own": "De eigen vector, onveranderd",
            "right": "Een gewogen gemiddelde van de vectoren van alle plekken",
            "before": "De som van de vectoren van de plekken ervoor, op volgorde",
            "max": "Het grootste getal van elke kolom",
        },
        correct="right",
        why={
            "right": "De weights hangen af van hoe goed twee plekken bij elkaar passen, en tellen op tot 1. Zo bevat elke vector na een block iets van de hele zin. Deze les neemt nog één keer het gemiddelde van die vectoren, voor één vector per zin.",
            "own": "Dan zou een block niets veranderen. Attention mengt de vectoren van de andere plekken erdoor, elk met een weight. Zie [Attention: elk woord kijkt naar de andere](/course/open/attn-idea?lang=nl).",
            "before": "Attention neemt een gewogen gemiddelde, met weights die optellen tot 1, en geen som. In die les keek elk woord naar alle andere.",
            "max": "Attention kiest niet één getal: het neemt een gewogen gemiddelde van de vectoren van alle plekken, met weights die afhangen van hoe goed de plekken bij elkaar passen.",
        },
    )
    quick_attention
    return (quick_attention,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Zoeken op betekenis vergelijkt teksten, en een vraag en de tekst met het antwoord kunnen weinig woorden delen. Met één vector per zin vergelijkt een computer twee teksten met één cosine similarity, zoals twee woorden in [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl). Nils Reimers en Iryna Gurevych, de makers van Sentence-BERT, plakten er in 2019 een getal op. BERT leest twee zinnen samen en geeft het paar dan een score, dus het meest gelijkende paar vinden tussen 10.000 zinnen kost 49.995.000 keer BERT draaien: ongeveer 65 uur op een V100, een snelle grafische kaart uit die tijd. Met één vector per zin kost het ongeveer 5 seconden om de 10.000 vectoren te berekenen, en ongeveer 0,01 seconde om de vectoren te vergelijken.

    De eenvoudigste vector voor een zin is het gemiddelde van de vectoren van de woorden. Deze les laat zien wat zo'n gemiddelde kwijtraakt, en wat een model bewaart dat voor zinnen getraind is.

    **Een voorbeeld uit het dagelijks leven.** Een smoothie van een banaan en een aardbei smaakt hetzelfde, welke vrucht ook als eerste in de blender ging. Een gemiddelde van woordvectoren is een smoothie van de woorden.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="Granite Embedding is een model dat getraind is om elke zin één vector te geven. Welke zin zet het dichter bij *the film was good*: *i enjoyed the movie*, dezelfde betekenis in andere woorden, of *the film was not good*, de tegenovergestelde betekenis?",
        options={
            "far": "*i enjoyed the movie*, met afstand",
            "little": "*i enjoyed the movie*, een klein beetje",
            "same": "Allebei even dicht",
            "right": "*the film was not good*",
        },
        reveal="**De zin met *not*:** *the film was not good* krijgt een cosine similarity van 0,925 met *the film was good*, en *i enjoyed the movie* 0,830. Het paar met vier woorden gemeen komt dichter uit, hoewel *not* de betekenis omdraait. Het idee laat zien waar dat vandaan komt, en het *Let op* aan het eind ervan komt terug op *not*.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Eén vector voor een zin

    Een **sentence embedding** is één vector voor een hele zin: evenveel getallen voor een zin van drie woorden als voor een zin van dertig, zodat elke twee zinnen te vergelijken zijn met de cosine similarity, zoals in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl). Zinnen met dezelfde betekenis horen vectoren te krijgen die bijna dezelfde kant op wijzen. Deze les bekijkt drie manieren om daar te komen: het gemiddelde van de woordvectoren, een transformer met pooling, en een transformer die op paren zinnen getraind is.

    ### Het gemiddelde van de woordvectoren

    Neem de GloVe-vector van elk woord van de zin, en neem het gemiddelde: tel voor elk van de 50 plaatsen de getallen van alle woorden op en deel door het aantal woorden. *the dog bit the man* en *the man bit the dog* hebben dezelfde vijf woorden, dus de gemiddelden zijn gelijk, getal voor getal, en de cosine similarity is 1: het gemiddelde is de volgorde van de woorden kwijt. Een woord dat GloVe niet kent, blijft buiten het gemiddelde; een zin waarvan GloVe geen enkel woord kent, heeft geen gemiddelde.
    """)
    return


@app.cell(hide_code=True)
def idea_mean():
    mo.Html(C.idea_mean_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_pairs():
    mo.md(r"""
    *not* is één woord van de vijf in *the film was not good*, en de vector van *not* verschuift het gemiddelde maar een beetje: de cosine similarity met *the film was good* is 0,990. *i enjoyed the movie*, dezelfde betekenis in andere woorden, krijgt 0,949. Woorden als *the* en *was* tellen in een gemiddelde even zwaar als elk ander woord, en de meeste zinnen hebben zulke woorden: zelfs *the train leaves at nine*, over iets anders, krijgt 0,813.
    """)
    return


@app.cell(hide_code=True)
def idea_pairs():
    mo.Html(C.idea_pairs_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_pooling():
    mo.md(r"""
    ### Een transformer en pooling

    Een transformer zoals BERT-Tiny geeft elk stukje van een zin een eigen vector, gemaakt met attention uit alle andere stukjes, zoals in *Eén vector per woord, of één per gebruik*. Voor één vector per zin voegt een model die vectoren samen tot één: **pooling**. **Mean pooling** neemt het gemiddelde; **CLS pooling** neemt alleen de vector van *[CLS]*, het stukje voor de zin.

    Een model leest zinnen in een **batch**: een groep zinnen in één array, zoals een model in [Mini-batches en epochs](/course/open/torch-batches?lang=nl) van een paar voorbeelden tegelijk leerde. In die array heeft elke zin evenveel plekken nodig, dus een kortere zin wordt aangevuld met het stukje *[PAD]* tot de lengte van de langste. Een **mask** zegt welke plekken een echt stukje bevatten: 1 voor *[CLS]*, de stukjes van de zin en *[SEP]*, 0 voor *[PAD]*. De mask houdt de *[PAD]*-plekken buiten de attention, zodat de echte stukjes er niet door veranderen. De *[PAD]*-plekken krijgen wel een eigen vector, en die vectoren moeten buiten het gemiddelde blijven. De course stuurde de 32 zinnen in één batch door BERT-Tiny: elke zin heeft 10 plekken, waarvan 6 tot 10 echt, dus BERT-Tiny geeft elke zin een tabel van 10 rijen.

    Reimers en Gurevych probeerden beide manieren op BERT zelf. Ze schrijven dat de meest gebruikte aanpak het gemiddelde van de output-layer van BERT is, of de vector van het eerste stukje, *[CLS]*, en dat die gewoonte vrij slechte sentence embeddings geeft, vaak slechter dan het gemiddelde van GloVe-vectoren. Ze gebruikten zeven datasets met paren zinnen die mensen een score hadden gegeven voor hoeveel de twee zinnen op elkaar lijken. Een score van 100 betekent dat de cosine similarities de paren in precies dezelfde volgorde zetten als de mensen. Het gemiddelde van de vectoren van BERT kreeg gemiddeld 54,81, CLS pooling 29,19, en het gemiddelde van de GloVe-vectoren 61,32.

    ### Getraind op paren

    Wat helpt, is trainen op paren zinnen. Sentence-BERT stuurt elke zin van een paar door dezelfde BERT, met dezelfde weights, één zin tegelijk, maakt met pooling voor elke zin één vector, en leert van het paar. Reimers en Gurevych trainden het model op **NLI**, natural language inference: paren zinnen waarbij mensen aangaven of de tweede zin volgt uit de eerste, ermee in tegenspraak is, of geen van beide, 570.000 paren in SNLI en 430.000 in MultiNLI. Na één ronde over die paren kreeg mean pooling van het getrainde model 74,89 op dezelfde zeven datasets. In die training leerde het model elk paar bij een van de drie labels te zetten, en toen maakte de manier van pooling weinig uit: volgens de twee onderzoekers heeft de keuze van pooling maar een vrij kleine invloed als het model op NLI-data leert de paren in te delen. Uit het werk van Reimers en Gurevych groeide **Sentence Transformers**, een Python-library om sentence embeddings te berekenen en te trainen, met een familie modellen.

    Granite Embedding, het embedding-model van de course, leerde ook van paren, met **contrastive training**. Het team van IBM dat het model maakte, beschrijft een training die de embeddings van een zoekvraag dichter bij de embeddings van relevante passages brengt, en verder weg duwt van de embeddings van niet-relevante passages. Onder de paren zitten titels met de tekst erbij, vragen met het antwoord, en vragen die hetzelfde vragen. Granite gebruikt CLS pooling; volgens hetzelfde team werkt CLS pooling beter dan de mean pooling die in ander werk vaak gebruikt wordt. Granite geeft elke zin 384 getallen, in 12 talen, waaronder het Nederlands.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **sentence embedding** | één vector voor een hele zin, even lang voor elke zin, zodat zinnen te vergelijken zijn met de cosine similarity |
        | **pooling** | de vectoren van de stukjes van een zin samenvoegen tot één vector |
        | **mean pooling** | pooling met het gemiddelde van de vectoren van de echte stukjes |
        | **CLS pooling** | pooling met alleen de vector van *[CLS]* |
        | **batch** | een groep zinnen die een model samen leest, in één array |
        | ***[PAD]*** | het stukje dat een kortere zin aanvult tot de lengte van de langste in een batch |
        | **mask** | 1 voor elk echt stukje van een zin in een batch, 0 voor elke *[PAD]*-plek |
        | **NLI** | natural language inference: paren zinnen met een label of de tweede zin volgt uit de eerste, ermee in tegenspraak is, of geen van beide |
        | **contrastive training** | training die de vectoren van teksten die bij elkaar horen naar elkaar toe trekt, en de vectoren van andere teksten wegduwt |
        | **Sentence Transformers** | een Python-library, met een familie modellen, om sentence embeddings te berekenen en te trainen |
        | `np.zeros(n)` | een array met `n` nullen: `np.zeros(3)` geeft `array([0., 0., 0.])` |
        | `E.shape[1]` | het aantal kolommen van de tabel `E`: 50 voor GloVe |
        | `tokens * mask` | met `mask` een kolom met één getal per rij: elke rij van `tokens` keer het getal van `mask` in die rij |
        """),
        L.watch_out("Trainen op gewone paren lost *not* niet op. Granite zet *the film was not good* dichter bij *the film was good*, 0,925, dan *i enjoyed the movie*, 0,830. Miriam Anschütz, Diego Miguel Lozano en Georg Groh, die in 2023 maten hoe meerdere modellen met ontkenning omgaan, schrijven dat large language models onderschatten hoeveel een ontkenning de betekenis van een zin verandert. Ze trainden een model voor sentence embeddings verder op paren zinnen met en zonder ontkenning, en dat model deed het daarop beter."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip de zin in woorden, of in de stukjes van een model.
    2. Voor het gemiddelde van de woordvectoren: tel de vectoren op van de woorden die de tabel heeft, en deel door hoeveel het er zijn. Zonder zo'n woord heeft de zin geen gemiddelde.
    3. Voor een transformer: stuur de zinnen in een batch, aangevuld met *[PAD]*, met de mask erbij. Neem voor mean pooling het gemiddelde van de rijen waar de mask 1 is; neem voor CLS pooling de rij van *[CLS]*.
    4. Vergelijk twee sentence embeddings met de cosine similarity. Een model dat op paren getraind is, zet zinnen met dezelfde betekenis dichter bij elkaar dan een gemiddelde; probeer *not* op je eigen zinnen voordat je erop vertrouwt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code neemt GloVe, zoals in *Word embeddings: buren en analogieën*: `C.glove()` geeft de lijst `words` en de tabel `E`, met een vector van 50 getallen voor elk van de 50.000 woorden, en `C.rows(words)` geeft de dict `row` van elk woord naar de rij ervan. `mean_of` neemt het gemiddelde van de vectoren van de woorden van een zin, langs axis 0, zoals in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl): één getal voor elk van de 50 plaatsen. `cosine` is de cosine similarity van twee vectoren. De code print de lengte van het gemiddelde van *the film was good*, en de cosine similarity met twee andere zinnen.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    words, E = C.glove()
    row = C.rows(words)

    def mean_of(sentence):
        return np.mean([E[row[w]] for w in sentence], axis=0)

    def cosine(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    good = mean_of(["the", "film", "was", "good"])
    print(good.shape)
    print(cosine(good, mean_of(["the", "film", "was", "not", "good"])))
    print(cosine(good, mean_of(["i", "enjoyed", "the", "movie"])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het gemiddelde heeft 50 getallen, net als elke vector van GloVe. *not* verschuift het weinig: 0,990, tegen 0,949 voor dezelfde betekenis in andere woorden. `mean_of` zoekt elk woord op met `row[w]`, dus een woord dat GloVe niet kent, zoals *o'clock*, laat de functie stoppen met een KeyError. De `mean_vector` van oefening 2 laat zulke woorden weg.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Hetzelfde gemiddelde, of een ander?

    Elke regel bevat twee zinnen, en GloVe kent elk woord ervan. Zet het paar onder *Precies hetzelfde gemiddelde* als de gemiddelden van de GloVe-vectoren van de woorden gelijk uitkomen, getal voor getal, en onder *Een ander gemiddelde* als de gemiddelden verschillen, al is het maar een beetje.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_same = L.sort("same-mean")
    sort_same
    return (sort_same,)


@app.cell(hide_code=True)
def check_sort(sort_same):
    result_sort = L.check("same-mean", sort_same)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    De volgorde telt nooit mee: *the bus hit the van* en *the van hit the bus* krijgen hetzelfde gemiddelde, en *not cold but warm* en *not warm but cold* ook, al zeggen de twee het tegenovergestelde. Een zin die twee keer gezegd wordt, houdt het aandeel van elk woord: *rain* en *rain rain*. Maar *hot* twee keer tussen vijf woorden weegt zwaarder dan één keer tussen vier, en *a* is een ander woord dan *the*: die gemiddelden verschillen, al is het maar een beetje, met cosine similarities van 0,990 en 0,991. *the soup is not hot* krijgt nog 0,983 met *the soup is hot*, en *the soup is warm* 0,972.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. mean_vector: het gemiddelde van de woordvectoren van een zin

        Schrijf `mean_vector(words, E, row)`: `words` is de lijst woorden van een zin, `E` een tabel met woordvectoren, en `row` een dict van elk woord naar de rij ervan in `E`. Geef het gemiddelde terug van de vectoren van de woorden met een rij, elk woord zo vaak meegeteld als het voorkomt, of `None` als geen enkel woord een rij heeft.

        In het vak staat `total` al op `np.zeros(E.shape[1])`, een vector met nullen zo lang als een rij van `E`, en `count` op 0; vervang de `...`. De check gebruikt eerst een kleine tabel, `small_E = np.array([[1, 0], [0, 1], [2, 2]])` met `small_row = {"a": 0, "b": 1, "c": 2}`: `mean_vector(["a", "b"], small_E, small_row)` moet `[0.5, 0.5]` geven, en `mean_vector(["a", "zz"], small_E, small_row)` moet `[1, 0]` geven, omdat *zz* geen rij heeft. Daarna gevallen van de check zelf: een zin zonder woord met een rij, een zin die begint met een woord zonder rij, een woord dat twee keer voorkomt, en twee zinnen met de `E` en `row` van GloVe.
        """),
        L.task("Geef het gemiddelde terug van de vectoren van de woorden met een rij, of None zonder zo'n woord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean_vector", lines=13)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean_vector", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean_vector",
        [
            "Ga de woorden langs met `for w in words:`. Tel voor een woord met een rij de vector `E[row[w]]` op bij `total`, en tel 1 op bij `count`. Na de loop: als `count` 0 is, geef dan `None` terug; geef anders het gemiddelde terug.",
            "*zz* heeft geen rij, en `row[w]` stopt daarbij met een KeyError. Wat kun je de dict `row` vragen over `w` voordat je `w` opzoekt?",
            r"`        if ___:`",
        ],
        why="`for w in words:` met `if w in row:` eronder, en daaronder `total = total + E[row[w]]` en `count = count + 1`; na de loop `if count == 0:` met `return None`, en als laatste `return total / count`. Voor `[\"a\", \"zz\"]`: *a* telt `[1, 0]` op en telt 1, *zz* wordt overgeslagen, dus het gemiddelde is `[1, 0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_mean():
    mo.md(r"""
    In de afbeelding nam je functie het gemiddelde van elke zin van de 16 paren van de course, in vier soorten. Dezelfde woorden in een andere volgorde krijgen 1,000, en de paren met *not* erbij gemiddeld 0,991. De paren met dezelfde betekenis in andere woorden, gemiddeld 0,954, komen nauwelijks boven de paren met dezelfde woorden en een andere betekenis, 0,942: *we sat on the bank of the river* en *we put money in the bank* krijgen 0,949, evenveel als *the film was good* en *i enjoyed the movie*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pooling():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. mean_pooling: het gemiddelde van de echte rijen

        Schrijf `mean_pooling(tokens, mask)`: `tokens` is de tabel die een transformer een zin in een batch geeft, één rij per plek, en `mask` een kolom met 1 voor elk echt stukje en 0 voor elke *[PAD]*-plek. Geef het gemiddelde terug van de rijen van `tokens` waar `mask` 1 is. Eén regel is genoeg, zonder loop: `tokens * mask` houdt elke echte rij en maakt elke *[PAD]*-rij 0.

        Voor `mean_pooling(np.array([[1, 2], [3, 4], [9, 9]]), np.array([[1], [1], [0]]))` is de uitkomst `[2, 3]`, het gemiddelde van de eerste twee rijen: de derde is een *[PAD]*-plek. De check roept dat aan en een tabel zonder *[PAD]*, daarna gevallen van de check zelf: meer *[PAD]*-plekken, een 0 tussen twee keer 1, bredere rijen, en twee zinnen van de course in BERT-Tiny, met `C.tiny_tokens(i)` en `C.tiny_mask(i)`: de tabel van zin `i` van de 32, na het tweede block, en de mask ervan.
        """),
        L.task("Geef het gemiddelde terug van de rijen van tokens waar mask 1 is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pooling():
    box_pooling = L.code_box("mean_pooling", lines=6)
    box_pooling
    return (box_pooling,)


@app.cell(hide_code=True)
def check_pooling(box_pooling):
    result_pooling = L.check("mean_pooling", box_pooling.value)
    result_pooling
    return (result_pooling,)


@app.cell(hide_code=True)
def hints_pooling():
    L.hints(
        "mean_pooling",
        [
            "`tokens * mask` houdt de echte rijen en maakt elke *[PAD]*-rij 0. Tel de rijen van die tabel op met `.sum(axis=0)`, en deel door het aantal echte stukjes.",
            "Elk echt stukje heeft een 1 in `mask`, en elke *[PAD]*-plek een 0. Welk getal, berekend uit `mask`, is het aantal echte stukjes?",
            r"`    return (tokens * mask).sum(axis=0) / ___`",
        ],
        why="`return (tokens * mask).sum(axis=0) / mask.sum()`. Voor het voorbeeld: `tokens * mask` is `[[1, 2], [3, 4], [0, 0]]`, de som langs axis 0 is `[4, 6]`, en `mask.sum()` is 2, dus het gemiddelde is `[2, 3]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pooling():
    mo.md(r"""
    In de afbeelding zetten alle drie de kolommen de vier soorten in dezelfde volgorde: dezelfde woorden in een andere volgorde het hoogst, dan *not* erbij, dan dezelfde betekenis in andere woorden, en dezelfde woorden met een andere betekenis het laagst. Het verschil zit in hoe ver de laatste twee soorten uit elkaar komen. Met het gemiddelde van GloVe krijgt dezelfde betekenis in andere woorden gemiddeld 0,954 en krijgen dezelfde woorden met een andere betekenis 0,942; met BERT-Tiny en jouw mean pooling 0,897 en 0,857; met Granite, getraind op paren, 0,899 en 0,777. Niet elk paar past in dat beeld: in Granite krijgt *the film was long* nog 0,851 met *the film was good*, meer dan *the engine failed to turn over* met *the car would not start*, 0,818. En ook Granite geeft de paren met *not* erbij een hoge cosine similarity, gemiddeld 0,949.
    """)
    return


@app.cell(hide_code=True)
def text_explore_glove():
    mo.md(r"""
    ### Probeer het: twee zinnen met GloVe

    Typ twee Engelse zinnen. De afbeelding laat het gemiddelde van de GloVe-vectoren van elke zin zien, met de woorden die erin gingen en de woorden die GloVe niet kent, en de cosine similarity van de twee gemiddelden. Probeer dezelfde woorden in een andere volgorde, zet er *not* bij, of zeg hetzelfde in andere woorden.
    """)
    return


@app.cell(hide_code=True)
def explore_glove_controls():
    explore_first = mo.ui.text(value="the shop opens at nine o'clock", label="Eerste zin", full_width=True)
    explore_second = mo.ui.text(value="the store is open at nine", label="Tweede zin", full_width=True)
    mo.vstack([explore_first, explore_second])
    return explore_first, explore_second


@app.cell(hide_code=True)
def explore_glove(explore_first, explore_second):
    mo.Html(C.glove_picture(explore_first.value, explore_second.value, L.lang))
    return


@app.cell(hide_code=True)
def text_explore_granite():
    mo.md(r"""
    ### Probeer het: je zinnen in Granite

    Granite Embedding draait op je computer zodra het gedownload is. Typ tot zes zinnen, één per regel, in het Nederlands, het Engels of een andere taal van het model: de afbeelding vergelijkt elke twee, de meest gelijkende eerst. Probeer een zin en de vertaling ervan, een zin met *niet*, en dezelfde betekenis in andere woorden.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Granite Embedding downloaden (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def granite_download(get_button):
    C.granite_or_stop(get_button, {
        "ask": "*Dit onderdeel heeft Granite Embedding nodig, en dat staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, in een vaste versie: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, onder de licentie Apache-2.0. De oefeningen en de rest van de les werken zonder.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte vrij, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    granite_here = True
    mo.md("*Granite Embedding is gedownload.*")
    return (granite_here,)


@app.cell(hide_code=True)
def explore_granite_controls():
    explore_lines = mo.ui.text_area(value="de film was goed\nik vond de film leuk\nde film was niet goed\nthe film was good", label="Je zinnen, één per regel", rows=6, full_width=True)
    explore_lines
    return (explore_lines,)


@app.cell(hide_code=True)
def explore_granite(explore_lines, granite_here):
    _ = granite_here
    mo.Html(C.granite_picture(explore_lines.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl) gaf elk gebruik van een woord een eigen vector; hier krijgt een hele zin er één. Het gemiddelde van de woordvectoren van [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl) is de volgorde van de woorden helemaal kwijt, en *not* bijna. Mean pooling van de vectoren van een transformer, met een mask die *[PAD]* buiten houdt, doet het op zichzelf niet veel beter; een model dat op paren zinnen getraind is, zoals Granite, zet zinnen met dezelfde betekenis dichter bij elkaar, al blijft *not* lastig. De volgende les van het domein, *Dezelfde betekenis? Gelijkenis en parafrases*, meet hoe goed de cosine similarity volgt wat mensen vinden, en waar een parafrase begint.

    **Een voorbeeld uit het dagelijks leven.** Brood en pannenkoeken zijn allebei van bloem, eieren en melk. Een lijst ingrediënten is een gemiddelde: de bereiding, de volgorde waarin alles gebeurt, maakt het gerecht.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_mean, result_pooling, result_sort):
    L.wrap_up(result_sort, result_mean, result_pooling)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
