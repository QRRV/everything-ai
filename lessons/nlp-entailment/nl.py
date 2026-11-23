# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-entailment. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-entailment", "nl")
    C = L.checks   # de paren van SNLI, drie modellen van naive Bayes, de getallen van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl) en [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_every():
    quick_every = L.choice(
        "quick-every",
        question="In de kleine wereld van *Betekenis: wanneer is een zin waar?* zijn Tom en Mia katten, en slapen Mia en Rex. Is *elke kat slaapt* waar in die wereld?",
        options={
            "mia": "Ja: Mia is een kat, en Mia slaapt",
            "two": "Ja: er slapen twee dieren, en er zijn twee katten",
            "right": "Nee: Tom is een kat en staat niet bij de slapers",
            "rex": "Dat valt niet na te gaan: Rex slaapt en is geen kat",
        },
        correct="right",
        why={
            "right": "*Elke* vraagt dat iedere kat bij de slapers staat, en Tom staat er niet bij. Die les toetste één zin aan één wereld. Deze les vergelijkt twee zinnen: als de eerste waar is, moet de tweede dan ook waar zijn?",
            "mia": "Dan is *sommige katten slapen* waar: minstens één kat slaapt. Wat vraagt *elke* van iedere kat? Zie [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
            "two": "*Elke kat slaapt* gaat niet over hoeveel dieren slapen, maar over welke. Staat iedere kat bij de slapers? Zie [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
            "rex": "De zin gaat over de katten; een dier dat geen kat is, verandert er niets aan. Staat iedere kat bij de slapers? Zie [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
        },
    )
    quick_every
    return (quick_every,)


@app.cell(hide_code=True)
def quick_check_not():
    quick_not = L.choice(
        "quick-not",
        question="In *Logistische regressie op woorden* geeft het model *Not bad at all.* (*Helemaal niet slecht.*) een kans van 0,008 op positief. Waarom zit het model er zo ver naast?",
        options={
            "short": "De zin is te kort om met TF-IDF te tellen",
            "right": "Elk woord heeft een eigen weight, dus *not* kan *bad* niet omdraaien",
            "unknown": "Het model zag *bad* nooit tijdens de training",
            "bias": "De bias is zo negatief dat de bias in elke zin alleen al de doorslag geeft",
        },
        correct="right",
        why={
            "right": "De logit telt op wat elk woord op zichzelf zegt: *not* en *bad* trekken allebei naar negatief. Een weight ziet één woord en niet de woorden eromheen. Deze les geeft zo'n model twee zinnen, en vraagt of de ene uit de andere volgt.",
            "short": "TF-IDF geeft ook elk woord van een korte zin een waarde. Wat doet het model met de waarde van *not* en de waarde van *bad*? Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "unknown": "*bad* heeft een sterk negatieve weight, dus het model kent het woord goed. Wat kan een weight van één woord niet zien? Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "bias": "De bias is één getal voor elke zin, en andere zinnen krijgen wel een hoge kans. Wat dragen *not* en *bad* elk bij? Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
        },
    )
    quick_not
    return (quick_not,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Volgt een antwoord uit een tekst, zegt een samenvatting wat de bron zegt, spreekt een bewering een verslag tegen? Elke keer is de vraag of de ene zin uit de andere volgt. Samuel Bowman, Gabor Angeli, Christopher Potts en Christopher Manning, de makers van de dataset van deze les, noemen het begrijpen van wat volgt en wat tegenspreekt een kern van taalbegrip, en een goede proef voor modellen die betekenis willen vastleggen. Een model dat zo'n proef goed doet, lijkt beide zinnen te begrijpen. Deze les kijkt hoe goed een model kan scoren als het maar één van de twee leest.

    **Een voorbeeld uit het dagelijks leven.** Een vriend appt *Ik ben met de hond op het strand.* Dan volgt daaruit *mijn vriend is buiten*. *De hond zwemt* kan waar zijn, of niet. *Mijn vriend zit thuis op de bank* kan niet tegelijk waar zijn.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="De course trainde naive Bayes, zoals in *Spam of niet*, op 36.788 paren van zinnen, elk met het label volgt, spreekt tegen of geen van beide. Het model kreeg van elk paar alleen de tweede zin te zien, nooit de eerste. Daarna labelde het model 9.824 nieuwe paren, waarin de drie labels ongeveer even vaak voorkomen. Hoeveel van elke 100 had het goed?",
        options={
            "chance": "Ongeveer 33 van elke 100, zoals bij gokken",
            "some": "Ongeveer 45 van elke 100",
            "right": "Ongeveer 61 van elke 100",
            "most": "Ongeveer 90 van elke 100",
        },
        reveal="**Ongeveer 61:** 5.972 van de 9.824 testparen, 0,608, zonder één eerste zin te lezen. Altijd het vaakste label geven, *volgt*, heeft er 3.368 goed, 0,343.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Premisse, hypothese, label

    **Natural language inference**, kortweg NLI, neemt twee zinnen: een **premisse** en een **hypothese**. De vraag is of de hypothese uit de premisse volgt (**entailment**), er niet tegelijk mee waar kan zijn (**contradiction**), of geen van beide (**neutral**). Het Stanford Natural Language Inference corpus, SNLI, bevat 570.152 van zulke paren, met hypothesen van ongeveer 2.500 schrijvers. Elke premisse is het bijschrift van een foto. Wie de hypothesen schreef, kreeg het bijschrift te zien en de foto niet, en moest met alleen het bijschrift en eigen kennis van de wereld drie nieuwe bijschriften schrijven: één dat zeker klopt bij de foto, één dat misschien klopt, en één dat zeker niet klopt. De instructies gaven één voorbeeld, in de afbeelding.
    """)
    return


@app.cell(hide_code=True)
def guide_cell():
    mo.Html(C.guide_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_world():
    mo.md(r"""
    *There are animals outdoors* (*Er zijn dieren buiten*) volgt alleen met wat je van de wereld weet: honden zijn dieren, en een veld is buiten. [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl) toetste één zin aan één kleine wereld. NLI vraagt iets ruimers: of iemand die de premisse leest, de hypothese zou concluderen.

    ### Eén scène

    Wanneer spreken twee zinnen elkaar tegen? Bowman en anderen geven het paar *A boat sank in the Pacific Ocean* en *A boat sank in the Atlantic Ocean* (*Een boot zonk in de Grote Oceaan* en *Een boot zonk in de Atlantische Oceaan*). Als je aanneemt dat de twee zinnen over één en dezelfde gebeurtenis gaan, kun je het paar een contradiction noemen, schrijven de makers; zonder die aanname is neutral ook redelijk. SNLI lost het grotendeels op met de scène: volgens Bowman en anderen moesten premisse en hypothese dezelfde situatie beschrijven, vanuit hetzelfde gezichtspunt. In SNLI gelden een boot in de premisse en een boot in de hypothese dus meestal als dezelfde boot. In een voetnoot schrijven de makers dat problemen met coreferentie, welke woorden naar hetzelfde verwijzen, zo niet helemaal opgelost zijn, maar wel veel kleiner worden.

    ### Wat de schrijvers achterlieten

    Suchin Gururangan en anderen keken in 2018 naar SNLI. In een flink deel van de data laat de manier van verzamelen aanwijzingen achter, schrijven de onderzoekers, waarmee je het label kunt vinden door alleen naar de hypothese te kijken, zonder de premisse. Een mogelijke verklaring volgens de onderzoekers: wie de hypothesen schreef, gebruikte vuistregels om snel en efficiënt te werken. *Nobody* in een hypothese wijst naar contradiction, net als *sleeping* en *cat*; *outdoors* en *animal* wijzen naar entailment, *tall* en *because* naar neutral. De afbeelding telt zeven van zulke woorden in de trainingsparen van deze les.
    """)
    return


@app.cell(hide_code=True)
def cue_cell():
    mo.Html(C.cue_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_shortcut():
    mo.md(r"""
    Over het voorbeeld uit de instructies schrijven de onderzoekers dat het de schrijvers waarschijnlijk op een spoor zette: *dog* vervangen door *animal*, en een beschrijving van de omgeving door *outdoors*. Een model dat op zulke paren traint, kan deze woorden leren in plaats van het verband tussen de twee zinnen. Dat is **shortcut learning**: een model leert een aanwijzing die in de trainingsdata bij het label hoort, in plaats van de vaardigheid waar de taak om vraagt. Op testparen die op dezelfde manier geschreven zijn, werkt de aanwijzing ook, dus de score ziet er goed uit.

    De afbeelding vergelijkt vier manieren om de 9.824 testparen van deze les te labelen. De drie modellen zijn naive Bayes, zoals in [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl), getraind op de 36.788 trainingsparen; de modellen verschillen alleen in de woorden die elk model leest. Het eerste leest alleen de hypothese. Het tweede leest de woorden van beide zinnen in één zak, met bij elk woord de zin waar het woord uit komt. Het derde leest de hypothese, met elk woord gemarkeerd als oud als de premisse het woord heeft, en als nieuw als dat niet zo is.
    """)
    return


@app.cell(hide_code=True)
def models_cell():
    mo.Html(C.models_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        De premisse erbij lezen levert niets op: 0,606 tegen 0,608. Naive Bayes scoort elk woord op zichzelf, en een woord van de premisse zegt op zichzelf weinig over het label. Het gaat erom hoe de woorden van de twee zinnen samenhangen, en het derde model vangt daar iets van: 0,693.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **natural language inference** (NLI) | bepalen of een hypothese uit een premisse volgt, de premisse tegenspreekt, of geen van beide |
        | **premisse** | de zin die gegeven is; in SNLI het bijschrift van een foto; in het Engels premise |
        | **hypothese** | de zin waarover je beslist; in het Engels hypothesis |
        | **entailment**, **contradiction**, **neutral** | de drie labels: de hypothese volgt eruit; de hypothese kan niet tegelijk waar zijn; de hypothese kan waar zijn |
        | **shortcut learning** | een model leert een aanwijzing die in de trainingsdata bij het label hoort, in plaats van de vaardigheid waar de taak om vraagt |
        | **overlap** | in deze les: het deel van de woorden van de hypothese dat in de premisse staat |
        | `C.TRAIN`, `C.TEST` | de 36.788 trainingsparen en de 9.824 testparen, elk een tuple `(premise, hypothesis, label)`, met premisse en hypothese als tuples van woorden in kleine letters |
        | `C.LABELS` | `("entailment", "neutral", "contradiction")` |
        | `C.LOG_PRIOR` | een dict met voor elk label de log van het deel van de trainingsparen met dat label |
        | `C.WORD_SCORES` | een dict met voor elk label een dict met de log-kans van elk woord uit de hypothesen van de training set, met Laplace smoothing, zoals in *Spam of niet* |
        """),
        L.watch_out("Een **hypothese** in NLI is een zin om aan de premisse te toetsen. Het is geen hypothese zoals in onderzoek, een bewering die een experiment test. En een hoge score op SNLI laat niet zien dat een model de twee zinnen met elkaar in verband brengt: in deze les haalt een model dat er maar één leest 0,608."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Labels van een woord: loop langs de paren, en tel voor elke hypothese met het woord 1 op bij het aantal van het label en bij het totaal. Deel elk aantal door het totaal.
    2. Een label op de hypothese alleen: begin voor elk label bij de log prior en tel de score op van elk woord van de hypothese dat in de dict van dat label staat. Het label met de hoogste score wint.
    3. Overlap: tel de woorden van de hypothese die in de premisse staan, en deel door het aantal woorden van de hypothese.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel telt de labels van de trainingsparen en print de aantallen. Daarna print de cel het eerste trainingspaar.
    """)
    return


@app.cell
def example_count():
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    for premise, hypothesis, label in C.TRAIN:
        counts[label] = counts[label] + 1
    print(counts)
    print(C.TRAIN[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De drie labels komen ongeveer even vaak voor, dus een model dat gokt, heeft ongeveer een derde goed. Het eerste paar heeft het label neutral: de premisse zegt niets over een park. Oefening 1 telt de labels van alleen de hypothesen waarin één bepaald woord staat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. label_shares: welke labels bij een woord horen

        Schrijf `label_shares(pairs, word)`: `pairs` bevat paren `(premise, hypothesis, label)`, met premisse en hypothese als lijsten of tuples van woorden. Geef van de hypothesen waarin `word` staat het deel van elk label terug, als dict met de keys `"entailment"`, `"neutral"` en `"contradiction"`. Tel elke hypothese één keer, ook als het woord er twee keer in staat. Staat het woord in geen enkele hypothese, dan is elk deel `0.0`. Vervang de `...`. De check roept `label_shares([(["a", "dog", "runs"], ["an", "animal", "runs"], "entailment"), (["a", "dog", "runs"], ["nobody", "runs"], "contradiction"), (["two", "kids", "play"], ["the", "kids", "know", "nobody"], "neutral"), (["a", "man", "sings"], ["nobody", "sings"], "contradiction")], "nobody")` aan: drie hypothesen hebben *nobody*, één neutral en twee contradiction, en dat geeft `{"entailment": 0.0, "neutral": 0.3333333333333333, "contradiction": 0.6666666666666666}`. Daarna eigen gevallen: een woord dat in geen enkele zin staat, een woord dat alleen in de premissen staat, een woord twee keer in één hypothese, en `C.TRAIN` met `"nobody"` en met `"outdoors"`.
        """),
        L.task("Geef het deel van elk label onder de hypothesen met een woord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("label_shares", lines=14)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("label_shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "label_shares",
        [
            "Twee tellers: `counts`, dat de start al geeft, en `total`, eerst 0. Een loop `for premise, hypothesis, label in pairs:` met een `if` die vraagt of `word` in `hypothesis` staat. Dan een nieuwe dict `shares`, en een loop `for label in counts:` die `shares[label]` vult.",
            "Tel binnen de `if` 1 op bij het aantal van het label van het paar, en 1 bij `total`. Een deel is een aantal gedeeld door het totaal; als `total` 0 is, is elk deel `0.0`.",
            "De regel onder `if total > 0:`: `            shares[label] = ___`",
        ],
        why="`total = 0`, dan `for premise, hypothesis, label in pairs:` met `if word in hypothesis:` en daaronder `counts[label] = counts[label] + 1` en `total = total + 1`. Na de loop `shares = {}` en `for label in counts:` met `if total > 0:` en daaronder `shares[label] = counts[label] / total`, en `else:` met `shares[label] = 0.0`; dan `return shares`. Het voorbeeld geeft `{\"entailment\": 0.0, \"neutral\": 0.3333333333333333, \"contradiction\": 0.6666666666666666}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_shares():
    mo.md(r"""
    De afbeelding onder de check laat je functie los op de trainingsparen, voor de zeven woorden uit het idee. Volgens Gururangan en anderen zijn ontkenningen als *nobody*, *no*, *never* en *nothing* sterke aanwijzingen voor contradiction, en spreekt *sleeping* elke bezigheid tegen. Voor de vele katten hebben de onderzoekers een eigen verklaring: waarschijnlijk komen de katten voort uit de vele foto's van honden in de oorspronkelijke dataset. Oefening 2 gebruikt alle woorden van een hypothese tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_guess():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. guess: een label op de hypothese alleen

        Schrijf `guess(hypothesis, log_prior, word_scores)`: naive Bayes zoals in [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl), met drie labels in plaats van twee, op de woorden van de hypothese alleen. `log_prior` bevat de log prior van elk label, en `word_scores` voor elk label een dict met de score van elk woord dat het label kent. De score van een label begint bij de log prior. Tel de score op van elk woord van de hypothese dat in de dict van dat label staat, zo vaak als het woord voorkomt; sla een woord over dat er niet in staat. De regels onder de `...` onthouden het label met de hoogste score. Vervang de `...`. De check roept `guess(["nobody", "is", "outdoors"], {"entailment": -1.0, "neutral": -1.2, "contradiction": -1.1}, {"entailment": {"nobody": -9.0, "outdoors": -4.0, "dog": -6.0}, "neutral": {"nobody": -9.5, "outdoors": -6.0, "dog": -6.1}, "contradiction": {"nobody": -5.0, "outdoors": -6.5, "dog": -6.2}})` aan. *is* staat niet in de dicts; contradiction krijgt −1,1 − 5,0 − 6,5 = −12,6, de hoogste van de drie, dus het antwoord is `"contradiction"`. Daarna eigen gevallen: een hypothese zonder woorden en een hypothese met alleen woorden die niet in de dicts staan, allebei met een prior waarin neutral het hoogst is; een woord twee keer; twee bekende woorden; één bekend woord met een prior waarin neutral een heel punt hoger is dan de rest; één bekend woord drie keer; en de drie hypothesen uit het voorbeeld van SNLI met `C.LOG_PRIOR` en `C.WORD_SCORES`.
        """),
        L.task("Kies een label op de woorden van de hypothese alleen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_guess():
    box_guess = L.code_box("guess", lines=14)
    box_guess
    return (box_guess,)


@app.cell(hide_code=True)
def check_guess(box_guess):
    result_guess = L.check("guess", box_guess.value)
    result_guess
    return (result_guess,)


@app.cell(hide_code=True)
def hints_guess():
    L.hints(
        "guess",
        [
            "Laat `score` beginnen bij de log prior van `label`. Loop dan langs de woorden van `hypothesis`, en tel voor elk woord dat in de dict van `label` staat de score van dat woord op bij `score`.",
            "De dict van een label is `word_scores[label]`, en `word in word_scores[label]` vraagt of het woord erin staat. De loop over de woorden komt direct onder de regel met de log prior, even ver ingesprongen. Een woord dat twee keer voorkomt, gaat twee keer door de loop.",
            "`                score = score + ___`",
        ],
        why="Vervang `score = ...` door `score = log_prior[label]`, dan `for word in hypothesis:` met `if word in word_scores[label]:` en daaronder `score = score + word_scores[label][word]`. De regels eronder blijven ongewijzigd. Het voorbeeld geeft `\"contradiction\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_guess():
    mo.md(r"""
    Op de hele test set heeft je functie 0,608 goed: het antwoord op de gok aan het begin. Gururangan en anderen melden ongeveer 67% voor een eigen model van de hypothese alleen, getraind op heel SNLI, tegen 34,3% voor het vaakste label. Adam Poliak en anderen melden 69,00% tegen 34,28% op de test set van SNLI. Maar het tweede model uit het idee, dat beide zinnen in één zak leest, kwam op 0,606: een woord van de premisse zegt op zichzelf niet veel. Oefening 3 bekijkt de twee zinnen samen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. overlap: hoeveel van de hypothese in de premisse staat

        Schrijf `overlap(premise, hypothesis)`: het deel van de woorden van de hypothese dat in de premisse staat. Tel elk woord van de hypothese, dus een woord dat twee keer voorkomt, telt twee keer. Een hypothese zonder woorden geeft `0.0`; dat doet de start al. Vervang de `...`. De check roept `overlap(["two", "dogs", "are", "running", "through", "a", "field"], ["there", "are", "animals", "outdoors"])` aan: alleen *are* staat in de premisse, en dat geeft `0.25`. Daarna eigen gevallen: elk woord erin, een hypothese zonder woorden, een woord twee keer in de hypothese, een woord twee keer in de premisse, het paar *The doctor was paid by the actor.* en *The doctor paid the actor.*, en geen enkel woord erin.
        """),
        L.task("Geef het deel van de woorden van de hypothese dat in de premisse staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overlap():
    box_overlap = L.code_box("overlap", lines=9)
    box_overlap
    return (box_overlap,)


@app.cell(hide_code=True)
def check_overlap(box_overlap):
    result_overlap = L.check("overlap", box_overlap.value)
    result_overlap
    return (result_overlap,)


@app.cell(hide_code=True)
def hints_overlap():
    L.hints(
        "overlap",
        [
            "Maak `in_premise`, de woorden van `premise` als set, en een teller `inside`, eerst 0. Loop langs de woorden van `hypothesis`, en tel de woorden die in `in_premise` staan.",
            "Een set houdt elk woord één keer: goed voor de premisse, waar je alleen iets aan vraagt. Tel de hypothese woord voor woord, en deel na de loop `inside` door het aantal woorden van de hypothese.",
            "`    in_premise = ___`",
        ],
        why="`in_premise = set(premise)` en `inside = 0`, dan `for word in hypothesis:` met `if word in in_premise:` en daaronder `inside = inside + 1`; na de loop `return inside / len(hypothesis)`. Het voorbeeld geeft `0.25`.",
    )
    return


@app.cell(hide_code=True)
def text_after_overlap():
    mo.md(r"""
    Op de testparen heeft entailment de hoogste overlap, gemiddeld 0,598, tegen 0,423 voor neutral en 0,402 voor contradiction. Gururangan en anderen vonden in SNLI dat 8,8% van de hypothesen met entailment helemaal in de premisse zit, als je elke zin als zak woorden neemt, tegen maar 0,2% van de hypothesen met neutral of contradiction. In de testparen van deze les hebben 330 van de 3.368 hypothesen met entailment overlap 1, tegen 3 met neutral en 3 met contradiction. Markeren welke woorden van de hypothese oud zijn en welke nieuw, zoals het derde model uit het idee doet, brengt de score van 0,608 naar 0,693.

    Gururangan en anderen splitsten de test set van SNLI ook in tweeën: de paren die het model van de hypothese alleen goed labelt, en de rest, die de onderzoekers Hard noemen. Drie bekende modellen uit 2016 tot 2018, die wel beide zinnen lezen, hebben 84,7 tot 86,5% van de hele test set goed, en 69,4 tot 72,7% van het Hard-deel. Volgens de onderzoekers betekent dat dat NLI-modellen entailment minder goed herkennen dan gedacht, en sterk leunen op de aanwijzingen in de hypothese. De onderzoekers voegen er wel aan toe dat zulke aanwijzingen geen enkel paar fout maken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_doctor():
    mo.md(r"""
    ### Oefening 4. Elk woord komt terug

    Tom McCoy, Ellie Pavlick en Tal Linzen beschrijven een regel die een model uit zulke data kan leren, de lexical overlap heuristic: neem aan dat uit een premisse elke hypothese volgt die gemaakt is van woorden uit de premisse. In de testparen van deze les werkt de regel goed: 330 van de 336 paren met overlap 1 zijn entailment.
    """)
    return


@app.cell(hide_code=True)
def choice_doctor_cell():
    choice_doctor = L.choice(
        "doctor",
        question="Wat zegt de regel over de premisse *The doctor was paid by the actor.* (*De dokter werd betaald door de acteur.*) en de hypothese *The doctor paid the actor.* (*De dokter betaalde de acteur.*), en klopt dat?",
        options={
            "neutral": "Neutral: de hypothese voegt niets nieuws toe",
            "right": "Entailment, en dat klopt niet: in de premisse betaalt de acteur",
            "fine": "Entailment, en dat klopt: elk woord van de hypothese staat in de premisse",
            "short": "Contradiction, omdat de hypothese korter is dan de premisse",
        },
        correct="right",
        why={
            "right": "Elk woord van de hypothese staat in de premisse, dus de overlap is 1, en de regel zegt entailment. Maar de volgorde van de woorden zegt wie wie betaalt, en een zak woorden gooit de volgorde weg. McCoy, Pavlick en Linzen bouwden een test set vol met zulke paren, HANS. De modellen die de onderzoekers testten, getraind op een andere grote dataset voor NLI, zeiden bijna altijd entailment, en hadden dus bijna elk paar fout waarvan het label geen entailment was.",
            "neutral": "De regel vraagt alleen of de woorden van de hypothese in de premisse staan, en hier staan alle woorden erin. Welk label geeft de regel dan?",
            "fine": "De regel zegt inderdaad entailment. Maar lees beide zinnen: wie geeft in elke zin het geld aan wie?",
            "short": "De regel kijkt niet naar lengte; de regel vraagt of de woorden van de hypothese in de premisse staan. Wat geeft de regel als alle woorden erin staan?",
        },
    )
    choice_doctor
    return (choice_doctor,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: drie modellen op je eigen paar

    Typ een premisse en een hypothese in het Engels, en kijk wat de drie modellen uit het idee zeggen. Verander alleen de premisse: het model van de hypothese alleen houdt hetzelfde label. Probeer *A cat is sleeping on a couch.* (*Een kat slaapt op een bank.*) met *A cat is sleeping.* (*Een kat slaapt.*): de hypothese volgt eruit, en dat model zegt contradiction, om *cat* en *sleeping*. Probeer daarna de dokter en de acteur.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_premise = mo.ui.text(value=C.DEMO[0], label="Premisse", full_width=True)
    demo_hypothesis = mo.ui.text(value=C.DEMO[1], label="Hypothese", full_width=True)
    mo.vstack([demo_premise, demo_hypothesis])
    return demo_hypothesis, demo_premise


@app.cell(hide_code=True)
def demo_look(demo_hypothesis, demo_premise):
    mo.Html(C.demo_picture(demo_premise.value, demo_hypothesis.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl) toetste één zin aan een wereld; NLI vraagt of de ene zin uit de andere volgt, met wat mensen van de wereld weten. De naive Bayes van [Spam of niet: naive Bayes](/course/open/lm-naive-bayes?lang=nl) scoort, net als de weights van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl), elk woord op zichzelf. Op SNLI levert dat 0,608 op met de hypothese alleen, tegen 0,343 voor altijd entailment, omdat de schrijvers van de hypothesen aanwijzingen achterlieten zoals *nobody*, *outdoors* en *tall*. Dat is shortcut learning: een goede score zonder de vaardigheid. Een test set waar de shortcut niet op past, zoals het Hard-deel van SNLI of de paren van HANS, laat het verschil zien. SNLI gebruikte één scène om grotendeels te bepalen of twee zinnen over dezelfde boot gaan; de volgende les, *Waar verwijst 'het' naar? Coreferentie*, zoekt uit welke woorden van een tekst naar hetzelfde verwijzen.

    **Een voorbeeld uit het dagelijks leven.** Bij een meerkeuzetoets waar het langste antwoord bijna altijd goed is, scoort een leerling die dat doorheeft goed zonder de stof te kennen. Een nieuwe toets, zonder dat patroon, laat zien wat de leerling weet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_doctor, result_guess, result_overlap, result_shares):
    L.wrap_up(result_shares, result_guess, result_overlap, choice_doctor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
