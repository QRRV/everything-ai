# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-wordnet. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-wordnet", "nl")
    C = L.checks   # WordNet als graaf, het keukenlexicon, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Betekenissen en relaties tussen woorden](/course/open/nlp-word-relations?lang=nl) en [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_meet():
    quick_meet = L.choice(
        "quick-meet",
        question="In *Betekenissen en relaties tussen woorden* klom je functie `chain` langs de ladders van eerste hyperoniemen van `poodle.n.01` en `cat.n.01`. Bij welke betekenis kwamen de twee ladders samen?",
        options={
            "entity": "`entity.n.01`, pas helemaal bovenaan",
            "animal": "`animal.n.01`",
            "right": "`carnivore.n.01`",
            "dog": "`dog.n.01`",
        },
        correct="right",
        why={
            "right": "Vanaf `carnivore.n.01` zijn de twee ladders hetzelfde. Hoe laag twee betekenissen samenkomen, zegt hoeveel de twee op elkaar lijken: deze les maakt daar een getal van.",
            "entity": "Dat waren de rivieroever en de bank voor geld, die onder de top niets delen. Een poedel en een kat zijn allebei carnivoren. Zie [Betekenissen en relaties tussen woorden](/course/open/nlp-word-relations?lang=nl).",
            "animal": "Allebei de ladders komen langs `animal.n.01`, maar komen al lager samen: een poedel en een kat zijn allebei carnivoren, en vanaf `carnivore.n.01` zijn de ladders hetzelfde.",
            "dog": "Een kat is geen soort hond, dus `dog.n.01` staat alleen op de ladder van `poodle.n.01`. De twee komen hoger samen, bij `carnivore.n.01`.",
        },
    )
    quick_meet
    return (quick_meet,)


@app.cell(hide_code=True)
def quick_check_bfs():
    quick_bfs = L.choice(
        "quick-bfs",
        question="Een gerichte graaf als adjacency list, met een edge van elke node naar elke node in de lijst ervan:\n\n```python\ngraph = {\"A\": [\"B\", \"E\"], \"B\": [\"C\"], \"C\": [\"D\"], \"E\": [\"D\"], \"D\": []}\n```\n\nBreadth-first search begint bij A. Welk pad naar D vindt het zoeken?",
        options={
            "first": "A, B, C, D, omdat B vooraan in de lijst van A staat",
            "right": "A, E, D",
            "direct": "A, D",
            "none": "Geen: de lijst van D is leeg",
        },
        correct="right",
        why={
            "right": "Breadth-first search probeert elk pad van 1 edge, dan elk pad van 2 edges, dus het eerste pad dat D bereikt, heeft de minste edges: A, E, D. In deze les klimt hetzelfde zoeken van een betekenis naar alle hyperoniemen erboven.",
            "first": "Breadth-first search volgt B niet eerst tot het eind: het zoeken probeert elk pad van 1 edge, dan van 2, en bereikt D via E na 2 edges, vóór het pad via B en C. Zie [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl).",
            "direct": "A heeft geen edge naar D: de lijst van A bevat B en E. Het kortste pad telt 2 edges.",
            "none": "De lege lijst zegt dat er geen edge uit D vertrekt. Er komen wel edges bij D aan, uit C en uit E, dus D is te bereiken.",
        },
    )
    quick_bfs
    return (quick_bfs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Lijkt een kat op een hond? Lijkt koffie op een kop? Daniel Jurafsky en James H. Martin houden de twee vragen in *Speech and Language Processing* uit elkaar. Een kat is geen synoniem van een hond, schrijven ze, maar katten en honden zijn zeker woorden die op elkaar lijken. Koffie lijkt niet op een kop: de twee hebben vrijwel niets gemeen, want koffie is een plant of een drank, en een kop een gemaakt voorwerp met een bepaalde vorm. Toch zijn koffie en een kop duidelijk verwant: de twee horen bij één alledaagse gebeurtenis, koffie drinken uit een kop. Word embeddings, zoals in *Word embeddings: buren en analogieën*, halen de twee door elkaar. GloVe, een verzameling word embeddings die geleerd is uit een grote hoeveelheid tekst, geeft elk woord een vector, en de cosine similarity van twee vectoren, hoogstens 1, zegt hoeveel de twee dezelfde kant op wijzen: *car* krijgt 0,92 met *truck*, en 0,85 met *driver*, terwijl een bestuurder een mens is en een vrachtwagen een voertuig. Een winkel die thee aanbiedt als de koffie op is, heeft woorden nodig die op elkaar lijken; met een kop heeft de klant niets.

    WordNet legt vast waar iets een soort van is. Deze les voegt de betekenissen van WordNet samen tot één netwerk, en meet hoe ver twee betekenissen daarin uit elkaar liggen.

    **Een voorbeeld uit het dagelijks leven.** Een supermarkt zet thee naast koffie, in het schap met dranken, en kopjes in een ander gangpad, hoewel je koffie uit een kopje drinkt. De schappen volgen wat iets is; een boodschappenlijstje volgt wat bij elkaar hoort.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="In WordNet heeft elke betekenis van een zelfstandig naamwoord een algemenere, tot één top. Neem twee zulke betekenissen op goed geluk, en loop van de ene omhoog naar de dichtstbijzijnde betekenis boven allebei, en omlaag naar de andere. Hoeveel stappen telt die wandeling gemiddeld?",
        options={
            "four": "Ongeveer 4",
            "eight": "Ongeveer 8",
            "right": "Ongeveer 14",
            "many": "Meer dan 30 stappen",
        },
        reveal="**Ongeveer 14**: bij 2.000 paren betekenissen van zelfstandige naamwoorden, op goed geluk gekozen, telt de kortste wandeling gemiddeld 13,66 stappen, van 1 tot 24. De meeste paren komen pas hoog samen, bij of vlak onder de top. Betekenissen die op elkaar lijken, komen veel lager samen: `coffee.n.01` en `tea.n.01` liggen 2 stappen uit elkaar. Oefening 3 maakt van het aantal stappen een gelijkenis.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### WordNet, een lexicon

    Volgens Jurafsky en Martin is de lexicale database WordNet (Fellbaum, 1998) de bron die het meest gebruikt wordt voor relaties tussen betekenissen, in het Engels en in veel andere talen. WordNet is een **lexicon**: een lijst woorden met informatie over elk woord, een woordenboek voor computers. Voor elk woord geeft WordNet de betekenissen, en voor elke betekenis de woorden, een definitie en de relaties met andere betekenissen. De groep bijna-synoniemen van een betekenis heet in WordNet een **synset**, van *synonym set*, en synsets zijn volgens Jurafsky en Martin een van de bouwstenen van WordNet. `sofa.n.01` uit *Betekenissen en relaties tussen woorden* is een synset met drie woorden, *sofa*, *couch* en *lounge*. De relaties van WordNet, zoals hyponymie en meronymie, verbinden synsets. De uitgave van Open English WordNet die de course gebruikt, heeft 107.425 synsets.

    ### Een graaf van betekenissen

    Neem elke synset als node, en elke verwijzing naar een hyperoniem als edge, met een richting: van de specifiekere betekenis naar de algemenere. Dan is WordNet een gerichte graaf, zoals in [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl), met 87.970 zulke edges. De course bewaart de graaf als adjacency list: `C.HYPERNYMS` is een dict van elk van de 107.425 synsets naar de lijst met hyperoniemen. Elke betekenis van een zelfstandig naamwoord klimt naar `entity.n.01`, de enige betekenis van een zelfstandig naamwoord zonder hyperoniem. De werkwoorden klimmen naar een van 8 toppen: 12.898 van de 13.811 betekenissen van werkwoorden klimmen naar `act.v.01`, andere bijvoorbeeld naar `exist.v.01` of `have.v.01`. De 21.830 betekenissen van bijvoeglijke naamwoorden en bijwoorden hebben helemaal geen hyperoniemen: de lijsten ervan zijn leeg.

    Een betekenis kan meer dan één hyperoniem hebben: dat geldt voor 2.298 betekenissen van zelfstandige naamwoorden en 38 van werkwoorden. `dog.n.01` heeft er twee, `canine.n.02` en `domestic_animal.n.01`. Boven een betekenis vormen de hyperoniemen, de hyperoniemen daarvan enzovoort dus een kleine graaf, en één ladder mist een deel ervan. Langs eerste hyperoniemen, zoals `chain` klom in *Betekenissen en relaties tussen woorden*, ligt `animal.n.01` 7 stappen boven `dog.n.01`; via `domestic_animal.n.01` zijn het er 2. Breadth-first search vindt elke betekenis boven een betekenis, elk met het kleinste aantal stappen.

    ### Path similarity

    Twee betekenissen die op elkaar lijken, komen laag samen. Om dat te meten, neem je het kortste pad tussen de twee via een hyperoniem dat de twee delen: omhoog van de ene betekenis naar dat hyperoniem, en omlaag naar de andere. Een van de twee kan zelf dat hyperoniem zijn: een betekenis en het eigen hyperoniem ervan liggen 1 stap uit elkaar. Voor `coffee.n.01` en `tea.n.01` loopt het pad via `beverage.n.01` en telt het 2 stappen; voor `coffee.n.01` en `cup.n.01` loopt het via `physical_entity.n.01` en telt het er 11, 5 omhoog vanaf koffie en 6 omhoog vanaf de kop. NLTK, een Python-library voor taal, maakt van het aantal stappen een **path similarity**:

    $$\text{path similarity} = \frac{1}{\text{stappen} + 1}$$

    Een betekenis en zichzelf liggen 0 stappen uit elkaar en krijgen 1; de documentatie van NLTK zegt dat een score van 1 voor dezelfde betekenis staat. Koffie en thee krijgen 1/3, koffie en een kop 1/12. Twee betekenissen die geen hyperoniem delen, zoals twee werkwoorden onder verschillende toppen, hebben geen pad tussen elkaar: dan is de path similarity in deze les `None`. NLTK zelf zet standaard een verzonnen top boven alle toppen van werkwoorden, en geeft zulke werkwoorden dan een kleine gelijkenis; met die verzonnen top uitgezet geeft NLTK ook `None`. Het idee is oud. Jurafsky en Martin beschrijven hoe M. Ross Quillian in 1968 en 1969 voorstelde om de betekenissen van woorden als netwerk op te slaan, en tussen de betekenissen van een woord te kiezen door het kortste pad tussen betekenissen in die graaf te zoeken.
    """)
    return


@app.cell(hide_code=True)
def idea_paths():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_lexicon():
    mo.md(r"""
    ### Een eigen lexicon

    Een klein eigen lexicon bouw je op dezelfde manier: schrijf van elk woord de hyperoniemen op, in een dict. Het keukenlexicon van deze les heeft 12 woorden: één top, *ding*, en 11 woorden daaronder, elk met één hyperoniem.
    """)
    return


@app.cell(hide_code=True)
def idea_kitchen():
    mo.Html(C.kitchen_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **WordNet** | een groot lexicon van het Engels: betekenissen met de woorden, definities en relaties ervan |
        | **lexicon** | een lijst woorden met informatie over elk woord, zoals de betekenissen of de hyperoniemen; een woordenboek voor computers |
        | **synset** | een groep woorden met één gedeelde betekenis in WordNet, met een naam zoals `sofa.n.01`; de relaties van WordNet verbinden synsets |
        | **path similarity** | 1 / (stappen + 1), voor het kortste pad tussen twee betekenissen via een hyperoniem dat de twee delen; `None` als er geen is |
        | **cosine similarity** | hoeveel twee vectoren dezelfde kant op wijzen, hoogstens 1; word embeddings geven woorden die vaak dezelfde woorden om zich heen hebben een hoge |
        | `C.HYPERNYMS` | de adjacency list van WordNet: een dict van elke synset naar de lijst met hyperoniemen, `[]` voor een betekenis zonder hyperoniem |
        | `wordnet.synset(name).hypernyms` | de hyperoniemen van één synset, zoals in *Betekenissen en relaties tussen woorden* |
        | `min(numbers)` | het kleinste getal in een lijst: `min([5, 3, 4])` geeft `3`; bij een lege lijst stopt Python met een ValueError |
        """),
        L.watch_out("Elke stap telt even zwaar, maar WordNet deelt sommige delen fijner in dan andere. `cat.n.01` ligt 7 stappen onder `animal.n.01`: feline, carnivore, placental, mammal, vertebrate, chordate, animal. `coffee.n.01` ligt 2 stappen onder `food.n.01`. Biologen delen dieren in veel niveaus in, en WordNet volgt dat; de path similarity telt elke stap even zwaar, waar die stap ook zit."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Bewaar een lexicon als adjacency list: een dict van elke betekenis naar de lijst met hyperoniemen, met `[]` voor een top.
    2. Om elke betekenis boven een betekenis te vinden, met het kleinste aantal stappen, gebruik je breadth-first search omhoog: de betekenis zelf ligt 0 stappen hoog, en een hyperoniem van een betekenis die *k* stappen hoog ligt, ligt *k* + 1 stappen hoog, tenzij het zoeken het hyperoniem al eerder bereikte.
    3. Neem voor twee betekenissen de betekenissen die vanaf allebei te bereiken zijn, waarbij elke betekenis zichzelf op 0 stappen omhoog telt. Tel voor elk de stappen omhoog vanaf de ene betekenis en vanaf de andere op: de kleinste som is de lengte van het kortste pad.
    4. De path similarity is 1 / (stappen + 1). Zonder gedeeld hyperoniem is de uitkomst `None`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code bewaart het keukenlexicon als adjacency list. `ladder` klimt van een woord langs het eerste hyperoniem, zoals `chain` deed, en houdt in een dict bij hoeveel stappen hoog elk woord ligt dat de functie passeert. Elk woord onder de top heeft één hyperoniem, dus één ladder vindt elk hyperoniem van een woord. Als laatste drukt de code de hyperoniemen af van `beverage.n.01` in WordNet, een betekenis van *drank*: elke vloeistof die je kunt drinken.
    """)
    return


@app.cell
def example_cell():
    from everything_ai import wordnet

    kitchen = {
        "ding": [],
        "keukengerei": ["ding"],
        "eten": ["ding"],
        "bestek": ["keukengerei"],
        "pan": ["keukengerei"],
        "drank": ["eten"],
        "soep": ["eten"],
        "vork": ["bestek"],
        "lepel": ["bestek"],
        "mes": ["bestek"],
        "thee": ["drank"],
        "koffie": ["drank"],
    }

    def ladder(word, hypernyms):
        steps = {word: 0}
        while len(hypernyms[word]) > 0:
            up = hypernyms[word][0]
            steps[up] = steps[word] + 1
            word = up
        return steps

    print(ladder("koffie", kitchen))
    print(ladder("soep", kitchen))
    print(wordnet.synset("beverage.n.01").hypernyms)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *koffie* klimt naar *drank*, *eten* en *ding*; *soep* naar *eten* en *ding*. De twee delen twee hyperoniemen: *eten*, 2 stappen boven *koffie* en 1 boven *soep*, samen 3, en *ding*, 3 en 2, samen 5. Het kortste pad, koffie, drank, eten, soep, telt 3 stappen, dus de path similarity is 1/(3 + 1) = 0,25. WordNet geeft een drank twee hyperoniemen: `beverage.n.01` is zowel `food.n.01` als `liquid.n.01`, eten en vloeistof. Een ladder van eerste hyperoniemen mist het tweede; oefening 2 vindt alle hyperoniemen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke path similarity?

    Elke regel bevat twee woorden uit het keukenlexicon. Tel de stappen van het kortste pad ertussen, en zet het paar onder de path similarity.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_similarity = L.sort("similarity")
    sort_similarity
    return (sort_similarity,)


@app.cell(hide_code=True)
def check_sort(sort_similarity):
    result_sort = L.check("similarity", sort_similarity)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    *bestek* is zelf het hyperoniem van *vork*: 1 stap, 1/2. *lepel* en *mes* hangen onder *bestek*, 1 stap omhoog en 1 omlaag: 1/3. *mes* en *pan* komen samen bij *keukengerei*, 2 stappen boven *mes* en 1 boven *pan*: 1/4. *pan* en *soep* komen pas samen bij de top, *ding*, 2 stappen boven elk: 1/5. Hoe lager twee woorden samenkomen, hoe hoger de path similarity.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ancestors():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ancestors: elke betekenis boven een betekenis

        Schrijf `ancestors(name, hypernyms)`. `hypernyms` is een adjacency list: een dict van elke betekenis naar de lijst met hyperoniemen, `[]` voor een top. Geef een dict terug van `name` en elke betekenis erboven naar het kleinste aantal stappen omhoog vanaf `name`.

        Voor `ancestors("poodle", {"poodle": ["dog"], "dog": ["animal"], "cat": ["animal"], "animal": []})` is de uitkomst `{"poodle": 0, "dog": 1, "animal": 2}`. Het vak begint de dict `steps` met `name` op 0 stappen en de `deque` `queue`, de wachtrij, met `name`, en neemt in elke ronde de volgende betekenis van voren uit de wachtrij, zoals breadth-first search doet; vervang de `...`. De check roept dat aan en `ancestors("animal", ...)` met dezelfde dict, wat `{"animal": 0}` moet geven, en daarna eigen gevallen: een betekenis met twee hyperoniemen, een betekenis die twee paden van verschillende lengte bereiken, en drie betekenissen uit WordNet met `C.HYPERNYMS`.
        """),
        L.task("Geef een dict terug van name en elke betekenis erboven naar het kleinste aantal stappen omhoog."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ancestors():
    box_ancestors = L.code_box("ancestors", lines=13)
    box_ancestors
    return (box_ancestors,)


@app.cell(hide_code=True)
def check_ancestors(box_ancestors):
    result_ancestors = L.check("ancestors", box_ancestors.value)
    result_ancestors
    return (result_ancestors,)


@app.cell(hide_code=True)
def hints_ancestors():
    L.hints(
        "ancestors",
        [
            "Ga in de loop de hyperoniemen van `node` langs, met `for up in hypernyms[node]:`. Een hyperoniem van `node` ligt één stap hoger dan `node`: geef het de stappen in `steps`, en zet het in de wachtrij, zodat het zoeken vandaar verdergaat.",
            "Breadth-first search bereikt een betekenis eerst langs een pad met de minste stappen. Bereikt het zoeken dezelfde betekenis nog een keer, langs een langer pad, dan moet het getal blijven zoals het is. Wat kun je aan `steps` vragen voordat je `up` een getal geeft?",
            r"`            if ___:`",
        ],
        why="`for up in hypernyms[node]:` met daaronder `if up not in steps:`, en daar weer onder `steps[up] = steps[node] + 1` en `queue.append(up)`. Voor het voorbeeld: *poodle* is 0; *dog* krijgt 1 en gaat de wachtrij in; dan krijgt *animal* 2. *animal* heeft geen hyperoniemen, dus de wachtrij raakt leeg en de loop stopt.",
    )
    return


@app.cell(hide_code=True)
def text_after_ancestors():
    mo.md(r"""
    In de afbeelding klom je functie vanaf `dog.n.01` en vanaf `domestic_cat.n.01`, de huiskat. Elk van de twee heeft twee hyperoniemen, en bij allebei hoort `domestic_animal.n.01`: daar komen de hond en de huiskat samen, 1 stap boven elk, samen 2 stappen. Langs eerste hyperoniemen zouden de twee pas samenkomen bij `carnivore.n.01`, net als de poedel en de kat. Boven `dog.n.01` liggen 15 betekenissen, van `canine.n.02` en `domestic_animal.n.01` op 1 stap tot `entity.n.01` op 6.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_similarity():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. path_similarity: hoeveel lijken twee betekenissen op elkaar?

        Schrijf `path_similarity(a, b, hypernyms)`: de path similarity van de betekenissen `a` en `b`, 1 / (stappen + 1) voor het kortste pad ertussen via een hyperoniem dat de twee delen, of `None` als de twee er geen delen. Het vak mag `ancestors` gebruiken: de versie van de course, die werkt zoals de jouwe hoort te werken. Het vak begint `up_a` en `up_b`, de dicts van `ancestors` voor allebei de betekenissen, en een lege lijst `steps`. Vervang de `...` door een loop die voor elk gedeeld hyperoniem het aantal stappen van het pad erlangs in `steps` zet, en daarna de twee regels met `return`: één voor geen gedeeld hyperoniem, één voor de path similarity.

        Voor `path_similarity("poodle", "cat", {"poodle": ["dog"], "dog": ["animal"], "cat": ["animal"], "animal": []})` is de uitkomst `0.25`: poodle, dog, animal, cat is 3 stappen. De check roept dat aan en `path_similarity("cat", "cat", ...)` met dezelfde dict, wat `1.0` moet geven, en daarna eigen gevallen: twee paren die meer dan één hyperoniem delen, twee betekenissen zonder gemeenschappelijk hyperoniem, en vier paren uit WordNet met `C.HYPERNYMS`.
        """),
        L.task("Geef 1 / (stappen + 1) terug voor het kortste pad via een gedeeld hyperoniem, of None als a en b er geen delen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_similarity():
    box_similarity = L.code_box("path_similarity", lines=11)
    box_similarity
    return (box_similarity,)


@app.cell(hide_code=True)
def check_similarity(box_similarity):
    result_similarity = L.check("path_similarity", box_similarity.value)
    result_similarity
    return (result_similarity,)


@app.cell(hide_code=True)
def hints_similarity():
    L.hints(
        "path_similarity",
        [
            "Ga de betekenissen van `up_a` langs met een `for`-loop, en vraag voor elk of het ook in `up_b` staat: dan delen de twee het. Na de loop: als `steps` leeg is, `return None`; anders geef je de path similarity van het kortste pad terug.",
            "Een pad via een gedeeld hyperoniem gaat omhoog van `a` naar dat hyperoniem, en vandaar omlaag naar `b`. Welke twee dicts zeggen hoeveel stappen elk deel telt?",
            r"`            steps.append(___)`",
        ],
        why="`for name in up_a:` met daaronder `if name in up_b:` en `steps.append(up_a[name] + up_b[name])`; dan `if len(steps) == 0:` met `return None`, en als laatste `return 1 / (min(steps) + 1)`. Voor het voorbeeld: *poodle* en *cat* delen *animal*, 2 stappen boven *poodle* en 1 boven *cat*, dus `steps` is `[3]` en de path similarity 1/4 = 0,25.",
    )
    return


@app.cell(hide_code=True)
def text_after_similarity():
    mo.md(r"""
    In de afbeelding mat je functie acht paren betekenissen. De vier die op elkaar lijken, krijgen een path similarity van 0,25 of 0,33; de vier die alleen verwant zijn, krijgen 0,08 of minder, omdat de paden ervan pas bij of vlak onder de top samenkomen: een leraar en een school delen niets onder `entity.n.01`. GloVe geeft allebei de soorten een hoge cosine similarity: *car* en *driver* krijgen 0,85, bijna zoveel als *car* en *truck*. Word embeddings leren van de woorden rond een woord, en een bestuurder komt in dezelfde teksten voor als een auto. *coffee* en *cup* krijgen in GloVe maar 0,36. GloVe heeft één vector voor alle betekenissen van een woord samen, en de dichtstbijzijnde woorden van *cup* in GloVe zijn *champions*, *championship* en *tournament*: de bekers van de sport.
    """)
    return


@app.cell(hide_code=True)
def text_explore_lexicon():
    mo.md(r"""
    ### Probeer het: een eigen lexicon

    Verander het lexicon in de cel hieronder: voeg woorden toe, zoals *bord* onder *keukengerei* of *brood* onder *eten*, geef een woord een tweede hyperoniem, zoals *drank* ook onder een nieuw woord *vloeistof*, of begin een lexicon over iets anders. Kies daarna twee woorden: de afbeelding tekent je lexicon van de toppen omlaag, met het kortste pad tussen de twee.
    """)
    return


@app.cell
def explore_lexicon():
    my_lexicon = {
        "ding": [],
        "keukengerei": ["ding"],
        "eten": ["ding"],
        "bestek": ["keukengerei"],
        "pan": ["keukengerei"],
        "drank": ["eten"],
        "soep": ["eten"],
        "vork": ["bestek"],
        "lepel": ["bestek"],
        "mes": ["bestek"],
        "thee": ["drank"],
        "koffie": ["drank"],
    }
    return (my_lexicon,)


@app.cell(hide_code=True)
def explore_menus(my_lexicon):
    _words = C.lexicon_words(my_lexicon)
    explore_first = mo.ui.dropdown(options=_words, value=C.first_choice(_words, L.lang), label="Eerste woord")
    explore_second = mo.ui.dropdown(options=_words, value=C.second_choice(_words, L.lang), label="Tweede woord")
    mo.hstack([explore_first, explore_second], justify="start", wrap=True)
    return explore_first, explore_second


@app.cell(hide_code=True)
def explore_tree(explore_first, explore_second, my_lexicon):
    mo.Html(C.lexicon_picture(my_lexicon, explore_first.value, explore_second.value, L.lang))
    return


@app.cell(hide_code=True)
def text_explore_words():
    mo.md(r"""
    ### Probeer het: twee woorden in WordNet

    Typ twee Engelse woorden. De afbeelding probeert elke betekenis van het eerste woord als zelfstandig naamwoord met elke betekenis van het tweede woord als zelfstandig naamwoord, en elke betekenis als werkwoord met elke betekenis als werkwoord. Je ziet de paren betekenissen die het meest op elkaar lijken, met het pad van het beste paar, en de cosine similarity van de twee woorden in GloVe. Het beste paar kan een betekenis zijn die je niet bedoelde: voor *coffee* en *cup* is het `cup.n.06`, punch die in een kan wordt geserveerd in plaats van in een punchkom: ook een drank, net als koffie. Probeer *doctor* en *nurse*, *doctor* en *hospital*, of *run* en *walk*.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_a = mo.ui.text(value="coffee", label="Eerste woord")
    explore_b = mo.ui.text(value="cup", label="Tweede woord")
    mo.hstack([explore_a, explore_b], justify="start", wrap=True)
    return explore_a, explore_b


@app.cell(hide_code=True)
def explore_look(explore_a, explore_b):
    mo.Html(C.words_picture(explore_a.value, explore_b.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Betekenissen en relaties tussen woorden](/course/open/nlp-word-relations?lang=nl) gaf de betekenissen van WordNet een naam en beklom één ladder van eerste hyperoniemen. Hier vormen de synsets en de hyperoniemen ervan een gerichte graaf, bewaard als adjacency list zoals in [Grafen: nodes, edges en paden](/course/open/math-graphs?lang=nl), en breadth-first search vindt elke betekenis boven een betekenis met het kleinste aantal stappen. Het kortste pad via een gedeeld hyperoniem geeft een path similarity: hoog voor betekenissen die op elkaar lijken, zoals koffie en thee, en laag voor betekenissen die alleen bij elkaar horen, zoals een auto en de bestuurder, terwijl GloVe juist die twee dicht bij elkaar zet. Een eigen lexicon werkt op dezelfde manier; een latere les, *Een lijst met goede en slechte woorden*, bouwt een lexicon van woorden met het sentiment ervan. De volgende les van het domein, *Een vector voor een hele zin*, gaat terug naar vectoren, nu voor zinnen.

    **Een voorbeeld uit het dagelijks leven.** Een bibliotheek ordent boeken naar onderwerp, met onderwerpen binnen onderwerpen. Twee boeken op één plank lijken op elkaar; twee boeken in verschillende zalen komen pas bovenaan samen: boeken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ancestors, result_similarity, result_sort):
    L.wrap_up(result_sort, result_ancestors, result_similarity)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
