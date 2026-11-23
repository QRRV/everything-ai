# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-word-embeddings. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-word-embeddings", "nl")
    C = L.checks   # de vectoren van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl) en [Een woord ken je aan het gezelschap waarin het verkeert](/course/open/nlp-distributional-hypothesis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Wat is de cosine similarity van $(3, 4)$ en $(6, 8)$?",
        options={
            "dot": "50",
            "right": "1",
            "half": "0,5",
        },
        correct="right",
        why={
            "right": "$(6, 8)$ is $(3, 4)$ twee keer zo lang, dus de twee wijzen precies dezelfde kant op, en de lengtes vallen weg: $50 / (5 \\cdot 10) = 1$. In deze les krijgt elk woord zo'n vector, en de cosine similarity zegt hoeveel twee woorden op elkaar lijken.",
            "dot": "50 is het dot product, $3 \\cdot 6 + 4 \\cdot 8$. De cosine similarity deelt dat door de twee normen, 5 en 10. Zie [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
            "half": "De ene vector is twee keer zo lang als de andere, maar de cosine similarity kijkt alleen naar de richting, en de richting is dezelfde. Zie [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="In *Een woord ken je aan het gezelschap waarin het verkeert* gaven rijen met kale aantallen zelfs doelwoorden die niets met elkaar te maken hebben een groot deel van de rij gemeen. Welke woorden deden dat?",
        options={
            "pair": "warm, koud",
            "book": "regent, javaan, kontroleur",
            "right": "de, en, van",
        },
        correct="right",
        why={
            "right": "De meest voorkomende woorden staan bij bijna elk woord in de buurt, dus die woorden vullen elke rij: meer dan de helft van 3.000 willekeurige paren doelwoorden had een cosine similarity boven 0,35. De word embeddings van deze les zijn geleerd uit zulke aantallen over miljarden woorden, op een manier die heel vaak voorkomende paren van woorden minder zeggenschap geeft.",
            "pair": "*warm* en *koud* hadden vergelijkbare rijen omdat de twee woorden in hetzelfde gezelschap verkeren, maar dat zijn twee woorden, niet alle rijen. Zie [Een woord ken je aan het gezelschap waarin het verkeert](/course/open/nlp-distributional-hypothesis?lang=nl).",
            "book": "*regent* vond *javaan* en *kontroleur* omdat de rijen alleen drie boeken kenden, waaronder *Max Havelaar*. Dat is één rij, niet alle rijen. Zie [Een woord ken je aan het gezelschap waarin het verkeert](/course/open/nlp-distributional-hypothesis?lang=nl).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De rijen van de vorige les hadden duizenden kolommen, bijna allemaal 0, en de meest voorkomende woorden bepaalden een groot deel van elke cosine similarity. Rond 2013 kwamen er snelle methoden die elk woord in plaats daarvan een korte lijst getallen geven, geleerd uit miljarden woorden. Die vectoren vinden woorden die bij elkaar horen, en nog iets meer. Tomas Mikolov, Wen-tau Yih en Geoffrey Zweig schreven in 2013 dat de relatie tussen mannelijke en vrouwelijke woorden vanzelf wordt geleerd: met de geleerde vectoren komt King − Man + Woman heel dicht bij Queen uit. Dat voorbeeld werd beroemd. Veel zoekmachines en vertaalsystemen, en elk language model van nu, beginnen met het omzetten van woorden, of stukjes van woorden, in vectoren. Deze les werkt met echte word embeddings, zoekt de dichtstbijzijnde buren van woorden, en controleert hoe goed zulke sommen echt werken.

    **Een voorbeeld uit het dagelijks leven.** Wie weet dat Parijs bij Frankrijk hoort zoals Rome bij Italië, kan *Madrid hoort bij …?* beantwoorden zonder dat paar ooit geleerd te hebben.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_king = L.guess(
        "guess",
        question="De course heeft vectoren van 50 getallen voor de 50.000 meest voorkomende Engelse woorden. Neem de vector van *king* (koning), trek de vector van *man* af en tel de vector van *woman* (vrouw) erbij op. Welk van de 50.000 woorden heeft de vector met de hoogste cosine similarity met de uitkomst?",
        options={
            "queen": "queen",
            "right": "king",
            "princess": "princess",
            "woman": "woman",
        },
        reveal="**king**: met een cosine similarity van 0,886 is de vector van *king* zelf de dichtstbijzijnde. *queen* (koningin) komt op de tweede plaats, met 0,861, daarna *daughter*, *prince* en *throne*; *princess* komt op de zesde plaats, en *woman* pas op de 118e. Het beroemde resultaat laat de drie woorden van de vraag weg, en dan wint *queen*. Deze les laat zien waarom die stap nodig is, en wat dat zegt over de sommen.",
    )
    guess_king
    return (guess_king,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Word embeddings

    Een **word embedding** is een [embedding](/course/open/ml-embeddings-idea?lang=nl) van een woord: een korte lijst getallen, geleerd uit een grote hoeveelheid tekst, zodat woorden die in vergelijkbare contexten voorkomen vergelijkbare vectoren krijgen. De rijen van [Een woord ken je aan het gezelschap waarin het verkeert](/course/open/nlp-distributional-hypothesis?lang=nl) waren sparse: een kolom voor elk contextwoord, bijna elk aantal 0. Een word embedding is **dense**: enkele tientallen tot enkele honderden getallen, bijna geen ervan 0, en geen enkel getal staat voor één contextwoord. Daniel Jurafsky en James H. Martin schrijven in *Speech and Language Processing* dat die getallen geen duidelijke betekenis hebben. Alleen hoe de vectoren zich tot elkaar verhouden telt. Ook werken dense vectoren volgens het boek bij elke taak in NLP beter dan sparse vectoren.

    ### GloVe

    **GloVe**, kort voor Global Vectors, kwam in 2014 van Jeffrey Pennington, Richard Socher en Christopher Manning. Het begint nog steeds bij een co-occurrence matrix van een heel corpus, zoals in de vorige les: GloVe traint op de aantallen van paren woorden in het hele corpus. De makers trainden het op Wikipedia van 2014 en Gigaword 5, een grote verzameling nieuws, samen 6 miljard tokens in kleine letters, met een venster van tien woorden aan elke kant, en publiceerden vectoren voor 400.000 woorden. Deze les gebruikt de kleinste vectoren, van 50 getallen per woord, voor de 50.000 meest voorkomende woorden die alleen uit de letters a tot en met z bestaan. De andere beroemde methode uit die jaren is *word2vec*, van Mikolov en collega's in 2013, die de vectoren leert door een woord te voorspellen uit de woorden eromheen, of de woorden eromheen uit het woord.

    De afbeelding laat de 50 getallen van drie woorden zien. Geen enkel getal betekent op zichzelf iets, maar de meeste getallen van *frog* (kikker) en *toad* (pad) hebben hetzelfde teken.
    """)
    return


@app.cell(hide_code=True)
def idea_numbers():
    mo.Html(C.numbers_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Buren en analogieën

    De **dichtstbijzijnde buren** van een woord zijn, net als eerder, de woorden waarvan de vector de hoogste cosine similarity heeft met de vector van dat woord. Een **analogie** is een vraag van de vorm *a staat tot b als c staat tot wat?*; deze les schrijft die vraag als a : b = c : ?, bijvoorbeeld *apple : tree = grape : ?* (appel, boom, druif), waar het antwoord *vine* (wijnstok) is. Het **parallellogrammodel** beantwoordt de vraag met de vectoren: neem de stap van *a* naar *b*, b − a, tel de stap op bij *c*, en zoek het woord dat het dichtst bij b − a + c ligt. Jurafsky en Martin voeren het idee terug op de psychologen Rumelhart en Abrahamson in 1973, en geven het voorbeeld van deze les: king − man + woman geeft een vector dicht bij queen.
    """)
    return


@app.cell(hide_code=True)
def idea_parallelogram():
    mo.Html(C.parallelogram_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Zoals de gok liet zien, is het woord dat het dichtst bij b − a + c ligt vaak een woord van de vraag zelf. Jurafsky en Martin schrijven dat het parallellogram in de word embeddings van word2vec en GloVe meestal niet het gezochte woord teruggeeft, maar een van de drie woorden van de vraag of een andere vorm daarvan, en dat je die woorden dus uitdrukkelijk moet weglaten. Mikolov en collega's deden dat al in de publicatie uit 2013 waarin word2vec verscheen: bij het zoeken naar het antwoord lieten de makers de woorden van de vraag weg.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **word embedding** | een korte, dense vector van getallen voor een woord, geleerd uit een grote hoeveelheid tekst, zodat woorden in vergelijkbare contexten vergelijkbare vectoren krijgen |
        | **dense** | van een vector: enkele tientallen tot enkele honderden getallen, bijna geen ervan 0; het tegenovergestelde van sparse |
        | **GloVe** | Global Vectors (2014): word embeddings geleerd uit de aantallen van co-occurrence in een heel corpus |
        | **word2vec** | word embeddings geleerd door de woorden rond een woord te voorspellen, of een woord uit de woorden eromheen (2013) |
        | **analogie** | een vraag van de vorm a : b = c : ?, *a staat tot b als c staat tot wat?* |
        | **parallellogrammodel** | een analogie beantwoorden met het woord dat het dichtst bij b − a + c ligt |
        | `E @ v` | voor een tabel `E` en een vector `v`: het dot product van elke rij van `E` met `v`, één getal per rij |
        | `np.linalg.norm(A, axis=...)` | de norm langs één axis van een tabel: één norm per rij of één per kolom, afhankelijk van de axis, zoals bij `np.sum(A, axis=...)` |
        """),
        L.watch_out("Vergelijkbare vectoren betekenen nog steeds hetzelfde gezelschap, niet dezelfde betekenis: de dichtstbijzijnde buren van *hot* (heet) zijn *cool* en *cold* (koud). En de vectoren leerden wat de teksten zeggen, nieuws en Wikipedia, ook over mensen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Laad de vectoren: een lijst woorden en een tabel `E` met één rij per woord, en een dictionary van elk woord naar het nummer van de rij.
    2. Voor de buren van een woord: bereken de cosine similarity van de rij van dat woord met elke rij, en neem de hoogste, zonder het woord zelf.
    3. Voor een analogie a : b = c : ?: bereken v = b − a + c uit de rijen van de drie woorden.
    4. Bereken de cosine similarity van elke rij met v.
    5. Laat a, b en c weg, en neem het woord met de hoogste cosine similarity.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder laadt de vectoren met `glove` van de course, die de 50.000 woorden als lijst geeft en de vectoren als de tabel `E`, rij `i` voor woord `i`. `rows` van de course maakt de dictionary `index` van elk woord naar het nummer van de rij. De code laat de eerste vijf getallen van *frog* zien, en vergelijkt *frog* dan met *toad* en met *car* (auto), zoals in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl), één paar tegelijk.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    words, E = C.glove()
    index = C.rows(words)
    print(len(words), E.shape)
    print(words[:8])
    frog = E[index["frog"]]
    print(np.round(frog[:5], 3))
    for other in ["toad", "car"]:
        row = E[index[other]]
        print(other, round(float(np.dot(frog, row) / (np.linalg.norm(frog) * np.linalg.norm(row))), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `E` heeft 50.000 rijen van 50 getallen. De woorden gaan van het meest naar het minst voorkomende, dus de lijst begint met *the*, *of*, *to* en *and*. De eerste vijf getallen van *frog* zijn 0,61, −0,208, −0,72, 0,893 en 0,325. De cosine similarity met *toad* is 0,791, met *car* 0,042. Om de dichtstbijzijnde buren van *frog* te vinden, zou deze loop *frog* met alle 50.000 rijen vergelijken, één tegelijk. Oefening 1 doet alle rijen tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cosines():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. cosines: elke rij tegelijk

        Schrijf `cosines(E, v)`: geef de cosine similarity van elke rij van de tabel `E` met de vector `v` terug, één getal per rij, als één array. `E @ v` geeft het dot product van elke rij met `v`. Deel dat door de norm van elke rij en door de norm van `v`.

        Voor `cosines(np.array([[1.0, 0.0], [10.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]), np.array([1.0, 0.0]))` is de uitkomst `[1, 0.995, 0.707, -1]`, afgerond: de tabel van oefening 2 in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl), nu alle rijen in één keer. Vervang de `...`. De check roept dat aan en `cosines(np.array([[3.0, 4.0], [0.0, 2.0], [-6.0, -8.0]]), np.array([2.0, 0.0]))`, en daarna eigen gevallen: tabellen met 2 en 3 getallen per rij, een tabel van één rij, een `v` met norm 5, en twee vectoren van GloVe tegen de eerste 2.000 rijen van GloVe.
        """),
        L.task("Geef de cosine similarity van elke rij van E met v terug, als één array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosines():
    box_cosines = L.code_box("cosines", lines=5)
    box_cosines
    return (box_cosines,)


@app.cell(hide_code=True)
def check_cosines(box_cosines):
    result_cosines = L.check("cosines", box_cosines.value)
    result_cosines
    return (result_cosines,)


@app.cell(hide_code=True)
def hints_cosines():
    L.hints(
        "cosines",
        [
            "Drie stukken: `E @ v` geeft het dot product van elke rij met `v`, `np.linalg.norm(v)` de norm van `v`, en `np.linalg.norm(E, axis=...)` een norm voor elke rij, met de goede axis. Deel na `return` de dot products door de normen van de rijen keer de norm van `v`, met haakjes om dat product.",
            "`E` heeft één rij per woord en één kolom per getal. In *Sommen en gemiddelden langs een axis* telde `np.sum` op langs de ene of de andere axis. Welke axis geeft één uitkomst per rij, zodat elk woord een eigen norm krijgt?",
            r"`    return E @ v / (np.linalg.norm(E, ___) * np.linalg.norm(v))`",
        ],
        why=r"`return E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))`: `axis=1` loopt langs elke rij, dus elke rij krijgt een eigen norm. In het voorbeeld zijn de vier cosine similarities 1, 0,995, 0,707 en −1.",
    )
    return


@app.cell(hide_code=True)
def text_after_cosines():
    mo.md(r"""
    In de afbeelding vond je functie de dichtstbijzijnde buren van vijf woorden tussen alle 50.000. *frog* vindt *snake* (slang), *ape* (mensaap), *toad*, *monkey* (aap), *spider* (spin) en *lizard* (hagedis): dieren. *coffee* vindt *drink*, *drinks*, *wine*, *tea* en *beer*, en *computer* vindt *computers*, *software* en *technology*. In de rijen van de vorige les vond *zeide* woorden als *meende* en *dacht*, woorden van dezelfde soort; hier vindt *said* (zei) *told*, *says* en *asked*, maar ook *spokesman* (woordvoerder), een woord dat er in het nieuws bij hoort. De buren zijn woorden die op allerlei manieren bij het woord horen: woorden van dezelfde soort, andere vormen van hetzelfde woord, en woorden over hetzelfde onderwerp.

    De vectoren gaan nog steeds over gezelschap, niet over betekenis. De dichtstbijzijnde buren van *hot* (heet) zijn *cool* en *cold* (koud), net zoals *warm* en *koud* in de rijen van de vorige les op elkaar leken. En een woord krijgt één vector voor alle betekenissen, vooral geleerd uit nieuws: de dichtstbijzijnde buren van *apple* (appel) zijn *blackberry*, *chips*, *iphone* en *microsoft*, en daar zit geen fruit tussen. De les *Eén vector per woord, of één per gebruik* gaat daarop verder.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_analogy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. analogy: a staat tot b als c staat tot wat?

        Schrijf `analogy(E, words, index, a, b, c)`: geef het antwoord op a : b = c : ? terug, met het parallellogrammodel. Bereken `v = E[index[b]] - E[index[a]] + E[index[c]]`, en `sims`, de cosine similarity van elke rij met `v`, met `cosines`. De check geeft je functie de `cosines` van de course, die hetzelfde doet als de jouwe. Laat daarna de drie woorden van de vraag weg: geef elk van de drie in `sims` een getal dat zo laag is dat `np.argmax` het overslaat. Geef ten slotte het woord met de hoogste cosine similarity terug.

        Voor `analogy(E, words, index, "man", "king", "woman")` is de uitkomst `"queen"`, met de vectoren van GloVe. Vervang de `...`. De check roept dat aan en `analogy(E, words, index, "france", "paris", "italy")`, dat `"rome"` moet geven, en daarna eigen gevallen: drie kleine tabellen van de course zelf, en nog zes vragen van GloVe.
        """),
        L.task("Geef het woord dat het dichtst bij b − a + c ligt, met a, b en c weggelaten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_analogy():
    box_analogy = L.code_box("analogy", lines=9)
    box_analogy
    return (box_analogy,)


@app.cell(hide_code=True)
def check_analogy(box_analogy):
    result_analogy = L.check("analogy", box_analogy.value)
    result_analogy
    return (result_analogy,)


@app.cell(hide_code=True)
def hints_analogy():
    L.hints(
        "analogy",
        [
            "Eerst `v = E[index[b]] - E[index[a]] + E[index[c]]`, dan `sims = cosines(E, v)`. Dan `for word in [a, b, c]:`, met daarin een nieuw getal voor `sims[index[word]]`, zodat `np.argmax` dat woord niet kan kiezen. Ten slotte `return words[np.argmax(sims)]`.",
            "Een cosine similarity ligt tussen −1 en 1. Welk getal voor de drie woorden van de vraag zorgt dat elk ander woord hoger uitkomt, wat de cosine similarity van dat woord ook is?",
            r"`        sims[index[word]] = ___`",
        ],
        why=r"`sims[index[word]] = -2`, of elk ander getal onder −1: dan wint elk ander woord van de drie woorden van de vraag (−1 werkt hier ook, maar een woord dat precies de andere kant op wijst, zou dan gelijk eindigen). Voluit: `v = E[index[b]] - E[index[a]] + E[index[c]]`, `sims = cosines(E, v)`, de loop `for word in [a, b, c]:` met `sims[index[word]] = -2` eronder, en `return words[np.argmax(sims)]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_analogy():
    mo.md(r"""
    In de afbeelding staan in de laatste kolom de woorden die het dichtst bij b − a + c liggen, zonder iets weg te laten. In drie van de zes vragen komt een woord van de vraag zelf op de eerste plaats: *king*, *grape* en *bad*. Met die woorden weggelaten vindt je functie *queen*, *rome*, *vine* en *worse*. *big : bigger = small : ?* geeft *larger*, met een cosine similarity van 0,904; *smaller* komt op de derde plaats, met 0,865, na *large*. *netherlands : amsterdam = germany : ?* geeft *frankfurt*, 0,880, vóór *berlin*, 0,856. Mikolov en collega's rekenden alleen precies het bedoelde woord goed, dus een synoniem telde als fout.

    *man : doctor = woman : ?* geeft *nurse* (verpleegkundige). Zonder iets weg te laten is het dichtstbijzijnde woord *doctor* zelf, 0,896 tegen 0,842. Malvina Nissim, Rik van Noord en Rob van der Goot wezen er in 2020 op dat het weglaten van de woorden van de vraag een ander antwoord dan *doctor* afdwingt. Het is volgens de drie onderzoekers niet helemaal terecht om de word embeddings bevooroordeeld te noemen omdat de som *doctor* niet teruggeeft. Dat *nurse* direct na *doctor* komt, laat toch iets zien van wat de teksten over vrouwen en mannen zeggen. Dat goed meten is het onderwerp van *Bias in word embeddings*.
    """)
    return


@app.cell(hide_code=True)
def text_groups():
    mo.md(r"""
    ### De vragen van word2vec

    Mikolov en collega's maakten een set van 19.544 analogievragen om word embeddings mee te meten: vijf soorten vragen over betekenis en negen soorten over grammatica. De veertien groepen gaan over betekenis, zoals hoofdsteden en landen, en over grammatica, zoals meervouden en de verleden tijd. Een vraag telt alleen als goed wanneer het antwoord precies het bedoelde woord is. De course liet een analogy die hetzelfde doet als de jouwe los op elke vraag waarvan de vier woorden bij de 50.000 woorden van GloVe horen: 17.692 vragen.

    ### Oefening 3. Welke groepen werken?

    Elke regel noemt een groep van de vragen van word2vec, met het eerste paar van de groep. Zet de regel onder het kopje dat zegt hoe vaak de analogy van de course de vragen van die groep goed heeft met de vectoren van GloVe.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_groups = L.sort("groups")
    sort_groups
    return (sort_groups,)


@app.cell(hide_code=True)
def check_sort(sort_groups):
    result_sort = L.check("groups", sort_groups)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    De afbeelding laat alle veertien groepen zien, van de beste naar de slechtste.
    """)
    return


@app.cell(hide_code=True)
def groups_picture():
    mo.Html(C.groups_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_groups():
    mo.md(r"""
    Van alle 17.692 vragen heeft de analogy van de course 47,9% goed. Pennington en collega's melden 71,7% voor de vectoren van 300 getallen die de makers uit hetzelfde corpus leerden; de vectoren van deze les hebben er maar 50. Zonder iets weg te laten is maar 32,5% goed: in 46,1% van de vragen is het dichtstbijzijnde woord *b* of *c* zelf, en *a* geen enkele keer. Tal Linzen wees er in 2016 op dat het weglaten van de drie woorden zelf een truc is die in de oplosser is ingebouwd: zo'n truc maakt de score waarschijnlijk beter, maar maakt het ook twijfelachtig of een goede score iets zegt over de vorm van de ruimte van de vectoren.

    Linzen stelde ook een controle voor, en de grijze balken laten die zien: neem de dichtstbijzijnde buur van alleen *c*, en gebruik *a* en *b* alleen om die woorden weg te laten. Dat alleen heeft al 14,8% van de vragen goed, en voor de -ing-vormen, zoals *code : coding = dance : ?*, 40,6%, tegen 41,3% voor het parallellogram. *dancing* is gewoon de dichtstbijzijnde buur van *dance*; bij deze vraag antwoordt het parallellogram zelfs *electronica*. Bij de -ing-vormen voegt de stap van *a* naar *b* bijna niets toe.

    De groepen die werken, hebben veelvoorkomende woorden en een relatie die de teksten keer op keer noemen: landen en nationaliteiten, hoofdsteden, *boy* en *girl*. Anna Rogers, Aleksandr Drozd en Bofang Li schreven in 2017 dat de meeste modellen goed scoren op verbuigingen en vervoegingen, omdat de vormen van hetzelfde woord in vergelijkbare contexten staan en waarschijnlijk dicht bij elkaar liggen. Namen van munten zijn zeldzame woorden. En Rogers en collega's vonden dat de stappen tussen paren woorden niet zo regelmatig lijken te zijn, ook niet bij eenvoudige relaties. Ook in de vectoren van deze les wijzen de stappen van verschillende paren nogal verschillende kanten op bij woorden en het tegendeel, en bij bijvoeglijke naamwoorden en het bijwoord met *-ly*, veel meer dan bij landen en nationaliteiten. Jurafsky en Martin vatten het zo samen: word embeddings werken goed bij veelvoorkomende woorden, kleine afstanden en bepaalde relaties, zoals landen en hoofdsteden of woorden en de verbogen vormen ervan. Op een moeilijkere set van 99.200 vragen in 40 groepen vonden Anna Gladkova, Aleksandr Drozd en Satoshi Matsuoka in 2016 dat GloVe gemiddeld 28,5% goed had; afgeleide woorden en relaties uit het woordenboek blijven volgens de drie onderzoekers een groot probleem.
    """)
    return


@app.cell(hide_code=True)
def text_dutch():
    mo.md(r"""
    ### Hetzelfde in het Nederlands

    Voor het Nederlands gebruikt de course de vectoren van *fastText*, die Edouard Grave en collega's in 2018 publiceerden voor 157 talen, geleerd uit Common Crawl, een grote verzameling webpagina's, en uit Wikipedia: 300 getallen per woord, met een venster van vijf woorden. fastText bouwt de vector van een woord op uit stukjes van vijf letters, dus woorden met dezelfde stukjes krijgen verwante vectoren. Het bestand voor het Nederlands bevat twee miljoen woorden, van het meest naar het minst voorkomende, en is 1,3 GB groot. De course downloadt alleen de eerste 20 MB: 30.355 volledige regels, en daarvan bewaart de course de 28.209 woorden die alleen uit letters bestaan. fastText laat hoofdletters staan, dus *koning* en *Koning* zijn twee woorden met twee vectoren. De Nederlandse vragen maakte de course zelf: 185 paren woorden in tien van de groepen van word2vec. Elke twee verschillende paren van een groep, in beide volgordes, maken een vraag, zoals in de meeste groepen van word2vec: 3.584 vragen.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download de Nederlandse vectoren, 20 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dutch_cell(get_button):
    C.download_or_stop(get_button, {
        "ask": "*De Nederlandse vectoren van fastText staan nog niet op je computer. De knop downloadt de eerste 20 MB van het bestand één keer van de site van fastText, onder de licentie CC BY-SA 3.0. De rest van de les werkt ook zonder die vectoren.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*De download lukte niet: de site van fastText was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is geen ruimte op je schijf voor 20 MB. Maak ruimte en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.Html(C.dutch_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_dutch():
    mo.md(r"""
    De Nederlandse buren lijken veel op de Engelse: *kikker* vindt *kikkers*, *schildpad* en *hagedis*, *koffie* vindt *thee*, en *zei* vindt *antwoordde*, *zegt* en *sprak*. Van de 3.584 Nederlandse vragen is 70,2% goed met de woorden van de vraag weggelaten. Zonder iets weg te laten is 24,2% goed: in 72,9% van de vragen is het dichtstbijzijnde woord *b* of *c* zelf. *man : koning = vrouw : ?* geeft *koningen*, het meervoud, vóór *koningin*; *Frankrijk : Parijs = Italië : ?* geeft *Milaan* vóór *Rome*. Woorden en het tegendeel werken ook hier het slechtst, 21,4%; ook Stéphan Tulkens, Chris Emmery en Walter Daelemans zagen in 2016 bij eigen Nederlandse word embeddings steeds lage scores voor tegenstellingen, en voor munten.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: elk woord, of elke drie

    Typ één woord om de acht dichtstbijzijnde buren te zien, of drie woorden, a b c, voor de analogie a : b = c : ?. De afbeelding laat het antwoord zien met de drie woorden weggelaten, en de dichtstbijzijnde woorden zonder iets weg te laten. Probeer *slow slower fast*, *japan tokyo france* of *france french germany*, en dan eigen woorden. Kies fastText voor het Nederlands, zodra de vectoren gedownload zijn; daar tellen hoofdletters, zoals in *Frankrijk Parijs Duitsland*.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_text = mo.ui.text(value="frog", label="Woord, of drie woorden")
    explore_kind = mo.ui.dropdown(options={"GloVe (Engels)": "glove", "fastText (Nederlands)": "fasttext"},
                                  value="GloVe (Engels)", label="Vectoren")
    mo.hstack([explore_text, explore_kind], justify="start", gap=1.5)
    return explore_kind, explore_text


@app.cell(hide_code=True)
def explore_look(explore_kind, explore_text):
    mo.Html(C.explore_picture(explore_text.value, explore_kind.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een woord ken je aan het gezelschap waarin het verkeert](/course/open/nlp-distributional-hypothesis?lang=nl) telde het gezelschap van elk woord in lange, sparse rijen. GloVe leert korte, dense word embeddings uit zulke aantallen over miljarden woorden, en je vergeleek alle vectoren tegelijk met de cosine similarity van [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl). De dichtstbijzijnde buren zijn woorden die op allerlei manieren bij een woord horen. Het parallellogram heeft ongeveer de helft van de vragen van word2vec goed, maar alleen met de drie woorden van de vraag weggelaten, en het best bij veelvoorkomende woorden en relaties dichtbij. De volgende les van het domein, *Eén vector per woord, of één per gebruik*, vraagt wat je moet doen met een woord als *bank*, dat één vector krijgt voor alle betekenissen.

    **Een voorbeeld uit het dagelijks leven.** Een zoekvak in een webwinkel dat ook *sneakers* vindt als je *gympen* typt, kan word embeddings gebruiken: de twee woorden verkeren in miljoenen teksten in hetzelfde gezelschap.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_analogy, result_cosines, result_sort):
    L.wrap_up(result_cosines, result_analogy, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
