# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-tf-idf. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("nlp-tf-idf", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Een document als vector](/course/open/nlp-document-vectors?lang=nl), en een over [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="Van 2.400 trainingszinnen bleek *The bus boy on the other hand was so rude.* het meest te lijken op *The delivery was on time.* Wat hebben die twee zinnen gemeen?",
        options={
            "tone": "Een negatieve toon, iets wat de cosine similarity kan opmerken",
            "right": "*the*, *was* en *on*, woorden die in veel zinnen staan",
            "topic": "Het onderwerp: beide gaan over service",
        },
        correct="right",
        why={
            "right": "De twee zinnen delen *the*, *was* en *on*, en geen enkel woord over een levering of over tijd. Zulke woorden staan in een groot deel van alle zinnen en zeggen dus weinig over één zin, maar kale aantallen laten zulke woorden toch een groot deel van de cosine similarity bepalen. Deze les geeft elk woord een waarde die laag is als het woord in veel documenten staat.",
            "tone": "De cosine similarity van aantallen ziet alleen welke woorden twee zinnen delen, niet de toon, en *The delivery was on time.* is een positieve zin. Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
            "topic": "De ene zin gaat over een levering, de andere over een bus boy, en de zinnen delen geen woord over een van beide. Zie [Een document als vector](/course/open/nlp-document-vectors?lang=nl).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="Wat geeft `np.log10(1000)`?",
        options={
            "e": "6,908, de macht van e die 1000 geeft",
            "hundred": "100,0, want dat is 1000 gedeeld door 10",
            "right": "3,0, want 10 × 10 × 10 = 1000",
            "small": "0,001, één gedeeld door 1000",
        },
        correct="right",
        why={
            "right": "`np.log10` vraagt welke macht van 10 het getal geeft: $10^3 = 1000$, dus de uitkomst is `3.0`. Elke factor 10 telt 1 op: `np.log10(100)` is 2 en `np.log10(10)` is 1. Deze les gebruikt dat om van *in tien keer zoveel documenten* *1 lager* te maken.",
            "e": "6,908 is `np.log(1000)`, de natuurlijke logaritme, met grondtal $e$; op school schrijf je daarvoor ln. `np.log10` heeft grondtal 10. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "hundred": "Een logaritme geeft een exponent, geen quotiënt. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
            "small": "0,001 is $10^{-3}$, één gedeeld door 1000; de logaritme van 1000 is de exponent zelf. Zie [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Een document als vector](/course/open/nlp-document-vectors?lang=nl) bepaalden gewone woorden zoals *the* en *was* een groot deel van de cosine similarity. Bij zoeken wordt het probleem nog duidelijker. Knip de vijf Engelse boeken van de course, *Alice's Adventures in Wonderland*, *Pride and Prejudice*, *The Adventures of Sherlock Holmes*, *Frankenstein* en *Moby-Dick*, in de 251 hoofdstukken, en zoek naar *the cat* door op te tellen hoe vaak elk hoofdstuk de twee woorden heeft. Bovenaan komt *The Adventure of the Speckled Band*, een verhaal over Sherlock Holmes van 9.882 woorden, met 598 keer *the* en 3 keer *cat*. *Pig and Pepper*, het hoofdstuk van *Alice* waarin de Cheshire Cat opduikt, heeft 22 keer *cat*, maar *the* alleen 150 keer, en komt op plaats 39. *the* staat in elk hoofdstuk, dus het woord zegt niets over welk hoofdstuk je zoekt, en toch telt het even zwaar als *cat*.

    Deze les geeft elk woord een waarde: hoog als het woord in weinig hoofdstukken staat, laag als het in veel hoofdstukken staat, en 0 als het in alle hoofdstukken staat. Jurafsky en Martin noemen in *Speech and Language Processing* twee gangbare manieren om termen zo te wegen: TF-IDF, en BM25, een variant daarvan. Voor het tweede deel van die weging verwijzen Jurafsky en Martin naar een paper van Karen Spärck Jones uit 1972. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze doen dat ook, in *Introduction to Information Retrieval*, en beschrijven het paper als gedetailleerde experimenten die laten zien wat de inverse document frequency oplevert bij het wegen van termen. Volgens hen kun je het werk van Hans Peter Luhn uit 1957 en 1958 zien als een voorloper van TF-IDF en verwante wegingen.

    **Een voorbeeld uit het dagelijks leven.** In een kookboek staat zout in bijna elk recept, en saffraan maar in een paar. Een recept met saffraan vertelt je veel meer over dat recept dan een recept met zout, ook al komt zout veel vaker voor.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Samen gebruiken de 251 hoofdstukken 22.858 verschillende woorden. Hoeveel van die woorden komen in alle 251 hoofdstukken voor?",
        options={
            "right": "Ongeveer 5",
            "fifty": "Ongeveer 50",
            "hundreds": "Ongeveer 500",
            "thousands": "Ongeveer 5.000",
        },
        reveal="**5**: *a*, *it*, *of*, *that* en *the*. *and* mist één hoofdstuk: hoofdstuk 122 van *Moby-Dick*, *Midnight Aloft.—Thunder and Lightning*, heeft *and* alleen in de titel, en de course laat de koppen weg. Het hoofdstuk is een kort toneel van 47 woorden: Tashtego, hoog op een ra van het schip in een onweer, wil rum in plaats van donder. Aan de andere kant komen 9.400 van de 22.858 woorden, 41,1%, in maar één hoofdstuk voor.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Term frequency en document frequency

    De **term frequency** van een woord in een document is hoe vaak het woord erin voorkomt: het aantal uit [Een document als vector](/course/open/nlp-document-vectors?lang=nl). Manning en collega's noemen dat aantal de term frequency. De **document frequency** van een woord is het aantal documenten waarin het woord voorkomt, hoe vaak het ook in elk document staat; Jurafsky en Martin geven dezelfde definitie, met *term* waar deze les *woord* zegt. Hier is een document één hoofdstuk, dus de *document frequency* van *hatter* is 3: de Hatter zit aan tafel in hoofdstuk 7 van *Alice* en is getuige in hoofdstuk 11, en hoofdstuk 6 noemt de Hatter alleen: de Cheshire Cat vertelt Alice waar een Hatter woont.

    Waarom documenten tellen, en niet elke keer dat een woord voorkomt? Manning en collega's vergelijken twee woorden in een verzameling nieuwsberichten, Reuters-RCV1: *try* komt 10.422 keer voor en *insurance* 10.440 keer, ongeveer even vaak, maar *try* is verspreid over 8.760 documenten en *insurance* over 3.997. De boeken van de course hebben ook zo'n paar: *alice* komt 397 keer voor, in 15 hoofdstukken, en *life* 399 keer, in 157 hoofdstukken. Een hoofdstuk met *alice* erin gaat waarschijnlijk over Alice; een hoofdstuk met *life* erin kan over alles gaan. Het totale aantal, dat Manning en collega's de **collection frequency** noemen, ziet geen verschil tussen de twee woorden; de *document frequency* wel.

    ### Inverse document frequency

    De **inverse document frequency**, kort IDF, maakt van de *document frequency* een waarde. Met $N$ documenten krijgt een woord $w$ dat in $\text{df}(w)$ daarvan voorkomt

    $$\text{idf}(w) = \log_{10} \frac{N}{\text{df}(w)}$$

    Jurafsky en Martin geven deze formule en zeggen wat de formule doet: hoe minder documenten de term bevatten, hoe hoger de waarde, en een term die in elk document voorkomt, krijgt de laagste waarde, 0. De logaritme houdt de waarden klein: een woord in één hoofdstuk van de 251 krijgt $\log_{10} 251 = 2{,}40$, niet 251, en een woord in tien keer zoveel hoofdstukken krijgt 1 minder. Manning en collega's nemen in hun voorbeeld ook grondtal 10, en voegen eraan toe dat het precieze grondtal voor de volgorde niet uitmaakt: een ander grondtal vermenigvuldigt elke waarde met hetzelfde getal. De afbeelding laat de IDF van een paar woorden uit de boeken zien.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### TF-IDF

    **TF-IDF** vermenigvuldigt de twee. De waarde van een woord $w$ in een document $d$ is de term frequency maal de inverse document frequency:

    $$\text{tf-idf}(w, d) = \text{tf}(w, d) \times \log_{10} \frac{N}{\text{df}(w)}$$

    Zo definiëren Manning en collega's het, met het aantal zelf als term frequency. Volgens hen is de waarde

    1. het hoogst als een term vaak voorkomt in een klein aantal documenten, want dan onderscheidt de term die documenten goed van de rest;
    2. lager als de term minder vaak in een document voorkomt, of in veel documenten staat, want dan zegt de term minder over wat relevant is;
    3. het laagst als de term in vrijwel alle documenten voorkomt.

    Een woord dat niet in een document staat, heeft aantal 0, dus ook waarde 0 in dat document.

    ### Een zoekopdracht, met een score

    Om te zoeken geef je elk document een **score**: tel de TF-IDF-waarden van de zoekwoorden in dat document op, en zet de documenten met de hoogste score bovenaan. Een zoekwoord dat niet in een document staat, telt niets op. Manning en collega's bouwen de score zo op: in plaats van de aantallen van de zoekwoorden tellen ze de TF-IDF-waarden op. Bij *the cat* telt *the* 0 op bij elk hoofdstuk, dus *cat* beslist: in *Pig and Pepper* geeft 22 keer *cat* met IDF 1,50 samen 32,9, de hoogste score van alle 251 hoofdstukken.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **term frequency** (TF) | hoe vaak een woord in één document voorkomt |
        | **document frequency** (df) | het aantal documenten waarin een woord voorkomt |
        | **collection frequency** | hoe vaak een woord in alle documenten samen voorkomt |
        | **inverse document frequency** (IDF) | $\log_{10}(N / \text{df})$: hoog voor een woord in weinig documenten, 0 voor een woord in alle documenten |
        | **TF-IDF** | de waarde van een woord in een document: de term frequency maal de inverse document frequency |
        | **score** | in deze les: de som van de TF-IDF-waarden van de zoekwoorden in één document |
        | $N$ | het aantal documenten |
        """),
        L.watch_out("De IDF hangt af van wat een document is. Neem de vijf boeken als vijf documenten, en *life*, dat in elk boek staat, krijgt 0, net als *the*; met hoofdstukken als documenten krijgt het woord 0,20. En een woord dat per ongeluk één keer voorkomt, zoals een tikfout, krijgt de hoogste IDF van allemaal."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip elk document in woorden, in kleine letters.
    2. Tel voor elk woord het aantal documenten waarin het voorkomt: de *document frequency*.
    3. Maak van elke *document frequency* een inverse document frequency: $\log_{10}$ van het quotiënt $N / \text{df}$.
    4. Vermenigvuldig het aantal van elk woord in een document met de IDF van het woord: de TF-IDF-waarde in dat document.
    5. Tel bij een zoekopdracht voor elk document de waarden van de zoekwoorden op, en zet de documenten met de hoogste score bovenaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code hieronder maakt van vijf *document frequencies* inverse document frequencies, voor $N = 251$: de code print elke *document frequency*, het quotiënt $N / \text{df}$, en de logaritme daarvan met grondtal 10.
    """)
    return


@app.cell
def example_cell():
    n = 251
    for frequency in [251, 176, 15, 3, 1]:
        print(frequency, round(n / frequency, 2), round(np.log10(n / frequency), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Een woord in alle 251 hoofdstukken krijgt 0, want $251 / 251 = 1$ en $\log_{10} 1 = 0$. *said*, in 176 hoofdstukken, krijgt 0,154; *alice*, in 15, krijgt 1,224; *hatter*, in 3, krijgt 1,923; en een woord in één hoofdstuk, zoals *steelkilt*, de naam van een zeeman in een verhaal dat in *Moby-Dick* verteld wordt, krijgt het meest: 2,4. Van 251 hoofdstukken naar 1 groeit het quotiënt 251 keer, en de logaritme met grondtal 10 alleen van 0 tot 2,4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_df():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. document_frequency: in hoeveel documenten?

        `small_docs`, de `SMALL_DOCS` van de course, bevat vier piepkleine documenten, elk een lijst woorden, en `chapters` de 251 hoofdstukken, uit `chapter_words` van de course. `chapter_label` geeft de naam van een hoofdstuk bij de plaats. Schrijf `document_frequency(docs)`: geef een dictionary terug van elk woord naar het aantal documenten van `docs` waarin het woord voorkomt. Een woord dat drie keer in één document staat, telt één keer voor dat document.

        Voor `document_frequency(small_docs)` krijgt *the* 4, krijgen *hatter* en *tea* 2, en krijgt elk ander woord 1: *tea* komt vier keer voor, maar in twee documenten. Vervang de `...`. De check roept dat aan en `document_frequency([["whale", "whale", "whale"], ["white", "whale"], ["ship"]])`, en daarna eigen gevallen: helemaal geen documenten, een leeg document, een woord in elk document, twee keer hetzelfde document, en de 251 hoofdstukken.
        """),
        L.task("Tel voor elk woord het aantal documenten waarin het voorkomt."),
    ])
    return


@app.cell
def chapters_cell():
    small_docs = C.SMALL_DOCS
    chapters = C.chapter_words()
    for doc in small_docs:
        print(doc)
    print(len(chapters), len(chapters[6]), chapters[6][:10])
    print(C.chapter_label(6, L.lang))
    return


@app.cell(hide_code=True)
def exercise_df():
    box_df = L.code_box("document-frequency", lines=9)
    box_df
    return (box_df,)


@app.cell(hide_code=True)
def check_df(box_df):
    result_df = L.check("document-frequency", box_df.value)
    result_df
    return (result_df,)


@app.cell(hide_code=True)
def hints_df():
    L.hints(
        "document-frequency",
        [
            "Begin met `df = {}`, en ga de documenten langs met `for doc in docs:`. Ga daarbinnen de verschillende woorden van het document langs, elk één keer. Tel voor elk zo'n `word` 1 op bij `df[word]`, of zet er 1 neer als het woord nog niet in `df` staat. Geef na de loops `df` terug.",
            "*tea* komt drie keer voor in het vierde document van `small_docs`. Hoeveel moet dat document optellen bij de *document frequency* van *tea*, en welke soort verzameling, gemaakt van een lijst, bevat elk item maar één keer?",
            r"`        for word in ___:`",
        ],
        why=r"`df = {}`; `for doc in docs:` en daarin `for word in set(doc):`, met `if word in df:` en `df[word] = df[word] + 1`, en `else:` met `df[word] = 1`; na de loops `return df`.",
    )
    return


@app.cell(hide_code=True)
def text_after_df():
    mo.md(r"""
    In de afbeelding telde je functie de *document frequency* van elk woord van de 251 hoofdstukken. Veel woorden zijn zeldzaam: 9.400 van de 22.858 komen in maar één hoofdstuk voor, en 14.871 in hoogstens drie. Hier zijn de *document frequencies* van de course voor de kleine documenten, `SMALL_DF`, en voor de hoofdstukken, uit `chapter_df`, en de aantallen van hoofdstuk 7 van *Alice*, *A Mad Tea-Party*, uit `chapter_counts`. De laatste twee zetten de meest voorkomende woorden vooraan.
    """)
    return


@app.cell
def counts_cell():
    small_df = C.SMALL_DF
    df = C.chapter_df()
    counts = C.chapter_counts(6)
    print(small_df)
    print(df["the"], df["hatter"], df["tea"])
    print(counts["the"], counts["hatter"], counts["tea"])
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. tf_idf: de waarden van één document

        Schrijf `tf_idf(counts, df, n)`, waarbij `counts` een dictionary is van elk woord van één document naar het aantal, `df` de *document frequencies* van alle woorden, en `n` het aantal documenten. Geef een dictionary terug van elk woord van `counts` naar de waarde: het aantal maal `np.log10(n / df[word])`.

        Voor `tf_idf({"the": 2, "hatter": 1, "poured": 1, "tea": 1}, small_df, 4)`, de aantallen van het eerste kleine document, is de uitkomst `{"the": 0.0, "hatter": 0.30103, "poured": 0.60206, "tea": 0.30103}`: *the* staat in alle vier documenten, *poured* in één, dus dat woord krijgt $\log_{10} 4 = 0{,}602$, en *hatter* en *tea* staan in twee. Vervang de `...`. De check roept dat aan en `tf_idf({"tea": 3, "said": 1, "the": 1, "hatter": 1}, small_df, 4)`, en daarna eigen gevallen: helemaal geen woorden, een woord in elk document, een andere `n`, en *A Mad Tea-Party* over de 251 hoofdstukken.
        """),
        L.task("Weeg elk aantal van één document met de inverse document frequency van het woord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weights():
    box_weights = L.code_box("tf-idf", lines=5)
    box_weights
    return (box_weights,)


@app.cell(hide_code=True)
def check_weights(box_weights):
    result_weights = L.check("tf-idf", box_weights.value)
    result_weights
    return (result_weights,)


@app.cell(hide_code=True)
def hints_weights():
    L.hints(
        "tf-idf",
        [
            "Begin met `weights = {}`, en ga de woorden langs met `for word in counts:`. Zet voor elk `word` in `weights[word]` het aantal van het woord maal de IDF, `np.log10(n / df[word])`. Geef na de loop `weights` terug.",
            "*tea* komt 3 keer voor in het vierde kleine document, en staat in 2 van de 4 documenten, dus de IDF is $\\log_{10}(4 / 2) = 0{,}301$. Waarmee vermenigvuldig je dat, en hoe schrijf je het aantal van een willekeurig woord van `counts`?",
            r"`        weights[word] = ___ * np.log10(n / df[word])`",
        ],
        why=r"`weights = {}`; `for word in counts:` met `weights[word] = counts[word] * np.log10(n / df[word])`; na de loop `return weights`.",
    )
    return


@app.cell(hide_code=True)
def text_after_weights():
    mo.md(r"""
    In de afbeelding woog je functie *A Mad Tea-Party*. De tien woorden met het hoogste aantal zijn *the*, *said*, *and*, *i*, *it*, *alice*, *you*, *a*, *to* en *of*: woorden van elk hoofdstuk, behalve *alice*. Met je waarden zijn het *hatter*, *alice*, *dormouse*, *hare*, *march*, *twinkle*, *treacle*, *tea*, *said* en *muchness*. De Hatter, de Dormouse en de March Hare zitten aan tafel, de Hatter zingt *Twinkle, twinkle, little bat*, de Dormouse vertelt over drie zussen die onderin een stroopput wonen, en *muchness* komt uit *much of a muchness*. *the*, 158 keer in het hoofdstuk, krijgt 0.

    Hier zijn de waarden van de course voor de vier kleine documenten, `SMALL_WEIGHTS`, en voor alle 251 hoofdstukken, uit `chapter_weights`, in de volgorde van `chapters`. Beide zijn een lijst dictionaries, één dictionary per document.
    """)
    return


@app.cell
def weights_cell():
    small_weights = C.SMALL_WEIGHTS
    weights = C.chapter_weights()
    print(small_weights[3])
    print(len(weights), round(weights[6]["hatter"], 2), weights[6]["the"])
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. score: een zoekopdracht

        Schrijf `score(query, weights)`, waarbij `query` een lijst zoekwoorden is en `weights` de waarden van één document: tel de waarden van de zoekwoorden op, en geef de som terug. Een zoekwoord dat niet in `weights` staat, telt niets op.

        Voor `score(["hatter", "tea"], small_weights[3])` is de uitkomst ongeveer 1,204: *tea* heeft waarde 0,903 in het vierde document en *hatter* 0,301. Vervang de `...`. De check roept dat aan en `score(["the", "queen"], small_weights[1])`, en daarna eigen gevallen: een zoekwoord dat niet in het document staat, helemaal geen zoekwoorden, een woord met waarde 0, een mengsel daarvan, en *the cheshire cat* tegen hoofdstuk 6 van *Alice*, *Pig and Pepper*.
        """),
        L.task("Tel de waarden van de zoekwoorden in één document op."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=6)
    box_score
    return (box_score,)


@app.cell(hide_code=True)
def check_score(box_score):
    result_score = L.check("score", box_score.value)
    result_score
    return (result_score,)


@app.cell(hide_code=True)
def hints_score():
    L.hints(
        "score",
        [
            "Begin met `total = 0`, en ga de zoekwoorden langs met `for word in query:`. Als het document een waarde heeft voor het `word`, tel die waarde dan op bij `total`. Geef na de loop `total` terug.",
            "*cat* staat niet in het eerste kleine document. Wat gebeurt er bij `small_weights[0][\"cat\"]`, en hoe vraag je eerst aan een dictionary of een key erin staat?",
            r"`        if word in ___:`",
        ],
        why=r"`total = 0`; `for word in query:` met `if word in weights:` en `total = total + weights[word]`; na de loop `return total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    In de afbeelding gaf je functie de 251 hoofdstukken een score voor drie zoekopdrachten. Bij *the cheshire cat* zetten kale aantallen *The Adventure of the Speckled Band* bovenaan, met 598 keer *the*; je scores zetten *Pig and Pepper* bovenaan, en *The Queen’s Croquet-Ground*, waar de Cat weer opduikt, op twee. *who stole the tarts* vindt hoofdstuk 11 van *Alice*, *Who Stole the Tarts?*, en *the red headed league* vindt het verhaal over Sherlock Holmes *The Red-Headed League*: de course knipt *Red-Headed* in de woorden *red* en *headed*.

    ### Hoe goed is het?

    TF-IDF heeft veel varianten. Manning en collega's betwijfelen of een term die twintig keer in een document staat, echt twintig keer zoveel betekent als een term die er één keer staat. Jurafsky en Martin nemen ook de logaritme met grondtal 10 van het aantal. Hun term frequency is $1 + \log_{10}$ van het aantal, en 0 bij een aantal van 0: een woord dat 10 keer voorkomt, krijgt 2, en 100 keer 3. `TfidfVectorizer` van scikit-learn gebruikt standaard het aantal zelf, maar neemt de natuurlijke logaritme en geeft elk woord $\log \frac{1 + N}{1 + \text{df}} + 1$, zodat een woord in elk document een kleine waarde houdt; daarna deelt de vectorizer de vector van elk document door de lengte. Het is allemaal TF-IDF; deze les gebruikt de vorm van Manning en collega's, met grondtal 10.

    TF-IDF doet ook veel van wat een lijst stop words doet. Volgens Jurafsky en Martin gebruiken moderne zoeksystemen veel minder vaak een lijst stop words, deels omdat zoeksystemen efficiënter zijn geworden en deels omdat de IDF al veel van dat werk doet: de IDF geeft functiewoorden die in elk document staan vanzelf een lage waarde.

    De score van deze les telt waarden op, dus een lang hoofdstuk dat een zoekwoord vaak heeft, kan een kort hoofdstuk verslaan dat helemaal over het zoekwoord gaat. Twee latere lessen van het domein pakken dat aan: *Rangschikken met de cosine similarity: het vector space model* deelt door de lengte van elk document, en bij *BM25: de klassieke rangschikking* telt elke extra keer dat een woord voorkomt minder mee.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: zoek in de boeken

    Typ een paar Engelse zoekwoorden; een woord dat je twee keer typt, telt één keer. De afbeelding laat de IDF van elk zoekwoord zien, de vijf hoofdstukken met de hoogste score, de waarde van elk zoekwoord in elk van die hoofdstukken, en het hoofdstuk dat kale aantallen bovenaan zouden zetten. Probeer *white whale*, een naam uit een van de boeken, alleen *the*, of een woord dat in geen enkel hoofdstuk staat.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_query = mo.ui.text(value=C.DEMO_QUERY, label="Zoekwoorden", full_width=True)
    demo_query
    return (demo_query,)


@app.cell(hide_code=True)
def demo_look(demo_query):
    mo.Html(C.demo_picture(demo_query.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij *a ship in the ice* tellen *a* en *the* 0 op en *in* bijna niets, dus *ice* en *ship* beslissen. Hoofdstuk 24 van *Frankenstein*, waar het verhaal eindigt in het ijs van het hoge noorden, heeft 18 keer *ice* en komt bovenaan, met 18,84; brief 4 uit hetzelfde boek, ook in het ijs, komt op twee. Hoofdstuk 135 van *Moby-Dick*, *The Chase.—Third Day*, komt op drie, met 19 keer *ship* en 3 keer *ice*. Kale aantallen zouden *The Adventure of the Speckled Band* bovenaan zetten: 1.030 keer *a*, *in* en *the*, en geen enkele keer *ship* of *ice*.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: andere waarden

    Kies een hoofdstuk, en bekijk de tien woorden met de hoogste waarde: met of zonder de IDF, en met het aantal zelf of met $1 + \log_{10}$ van het aantal, de term frequency van Jurafsky en Martin.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_chapter = mo.ui.dropdown(options=C.chapter_options(L.lang), value=C.chapter_label(6, L.lang), label="Hoofdstuk", full_width=True)
    explore_idf = mo.ui.checkbox(value=True, label="Maal de IDF")
    explore_log = mo.ui.checkbox(value=False, label="1 + log10 van het aantal, in plaats van het aantal")
    mo.vstack([explore_chapter, explore_idf, explore_log])
    return explore_chapter, explore_idf, explore_log


@app.cell(hide_code=True)
def explore_look(explore_chapter, explore_idf, explore_log):
    mo.Html(C.explore_picture(explore_chapter.value, explore_idf.value, explore_log.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Zonder de IDF zijn de tien woorden met de hoogste waarde in *A Mad Tea-Party* de woorden met de hoogste aantallen, vanaf *the*, 158 keer. Met de IDF en het aantal zelf komen de personages bovenaan. Met $1 + \log_{10}$ van het aantal weegt een woord dat 32 keer voorkomt maar $1 + \log_{10} 32 = 2{,}5$ keer zo zwaar als een woord dat één keer voorkomt, dus zeldzame woorden stijgen: *dormouse* gaat naar boven, en *muchness*, 3 keer in dit hoofdstuk en in geen ander, komt op zes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Welke waarde?

    Elke regel geeft een woord, hoe vaak het in een hoofdstuk voorkomt, en in hoeveel van de 251 hoofdstukken het voorkomt; elke regel gaat over een eigen hoofdstuk. Zet elk woord bij de TF-IDF-waarde in dat hoofdstuk.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_weight = L.sort("what-weight")
    sort_weight
    return (sort_weight,)


@app.cell(hide_code=True)
def check_sort(sort_weight):
    result_sort = L.check("what-weight", sort_weight)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een document als vector](/course/open/nlp-document-vectors?lang=nl) telde de woorden van elk document; deze les woog elk aantal met de inverse document frequency van het woord, een logaritme zoals in [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl), zodat een woord in elk document niets telt en een zeldzaam woord het meest. Als je die waarden optelt voor de woorden van een zoekopdracht, komen de hoofdstukken die over de zoekopdracht gaan bovenaan. De volgende les van het domein, *Logistic regression op woorden*, geeft TF-IDF-vectoren aan een classifier die uit voorbeelden met labels een waarde per woord leert.

    **Een voorbeeld uit het dagelijks leven.** Een zoekvak op een website dat de resultaten sorteert op hoe goed elk resultaat past, kan zo werken: de pagina's die de zeldzame woorden van je zoekopdracht vaak bevatten, komen bovenaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_df, result_score, result_sort, result_weights):
    L.wrap_up(result_df, result_weights, result_score, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
