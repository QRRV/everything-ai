# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-rouge. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-rouge", "nl")
    C = L.checks   # de zes abstracts van de course en de samenvattingen van Qwen3-0.6B, ROUGE, SciTLDR, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl), en één over [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bleu():
    quick_bleu = L.choice(
        "quick-bleu",
        question="In *Een vertaling scoren: BLEU en chrF* telt `clipped` de n-grams van een vertaling die ook in de referentie staan, elk hoogstens zo vaak als in de referentie. Bij de vertaling *the cat the cat* hoort de referentie *the cat sat*: de afgetopte telling van de unigrams is 2, één *the* en één *cat*. Wat is bij BLEU de precision van de unigrams, $p_1$?",
        options={
            "three": "2 / 3: de referentie heeft 3 unigrams",
            "right": "2 / 4",
            "four": "4 / 4: elk woord van de vertaling staat ook in de referentie",
            "seven": "2 / 7: de woorden van de twee zinnen samen",
        },
        correct="right",
        why={
            "right": "De vertaling heeft 4 unigrams, en 2 daarvan tellen: $p_1$ = 2 / 4 = 0,5, een precision, het deel van de vertaling dat ook in de referentie staat. ROUGE deelt in deze les dezelfde telling door de unigrams van de referentie: 2 / 3, een recall.",
            "three": "2 / 3 deelt door de referentie: het deel van de referentie dat terugkomt in de vertaling, een recall. De ROUGE van deze les doet precies dat. BLEU deelt door de unigrams van de vertaling: hoeveel heeft de vertaling er? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
            "four": "4 / 4 telt elk woord zo vaak als de vertaling het heeft, zonder aftoppen: *the* twee keer, terwijl de referentie het één keer heeft. Hoe vaak mag *the* meetellen? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
            "seven": "BLEU telt de twee zinnen nooit bij elkaar op. BLEU deelt de afgetopte telling door de unigrams van een van de twee: welke? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
        },
    )
    quick_bleu
    return (quick_bleu,)


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="In *Hoe ver liggen twee woorden uit elkaar? Edit distance* heeft de tabel `D` een rij voor elk beginstuk van `a` en een kolom voor elk beginstuk van `b`, van geen letters tot alle letters. Wat staat er in de cel `D[i, j]`?",
        options={
            "letters": "Of letter `i` van `a` gelijk is aan letter `j` van `b`",
            "right": "De edit distance tussen de eerste `i` letters van `a` en de eerste `j` letters van `b`",
            "whole": "De edit distance tussen heel `a` en heel `b`, in elke cel van de tabel opnieuw opgeschreven",
            "plus": "`i + j`, het aantal letters van de twee beginstukken samen",
        },
        correct="right",
        why={
            "right": "Elke cel beantwoordt de vraag voor twee beginstukken, en elke nieuwe cel bouwt voort op cellen die al gevuld zijn: het antwoord voor heel `a` en heel `b` komt in de laatste cel terecht. De langste gemeenschappelijke deelrij van deze les vult een tabel op dezelfde manier.",
            "letters": "Of twee letters gelijk zijn, bepaalt alleen wat vervangen in deze cel kost. Wat bevat de cel zelf, voor de beginstukken van `a` en `b`? Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
            "whole": "De edit distance van heel `a` en heel `b` staat in slechts één cel: de laatste. Wat bevatten de cellen op weg daarheen? Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
            "plus": "`i + j` is een bovengrens van de cel: alle `i` letters weghalen en alle `j` letters invoegen. Wat bevat de cel als er letters gedeeld worden? Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    *Samenvatten: kiezen of schrijven* liet één vraag open: hoe scoor je een samenvatting? Mensen kunnen elke samenvatting lezen, maar dat kost tijd, bij elke samenvatting en bij elke nieuwe versie van een samenvatter. In 2004 publiceerde Chin-Yew Lin van het Information Sciences Institute van de University of Southern California een pakket maten dat helemaal geen mens nodig heeft, zodra mensen een paar referentiesamenvattingen hebben geschreven. De naam ROUGE staat volgens Lin voor *Recall-Oriented Understudy for Gisting Evaluation*. De maten tellen, schrijft Lin, hoeveel eenheden een samenvatting van de computer deelt met de ideale samenvattingen die mensen schreven: n-grams, reeksen woorden en paren woorden. SciTLDR koos de orakelzinnen met ROUGE.

    In deze les bouw je de drie belangrijkste delen van ROUGE, scoor je de samenvattingen van Qwen3-0.6B en vier soorten samenvatting van 618 echte papers, en ontdek je wat ROUGE niet kan zien.

    **Een voorbeeld uit het dagelijks leven.** Een docent kijkt een kort antwoord na door de kernwoorden van een modelantwoord af te vinken. Een antwoord met alle kernwoorden krijgt alle punten, ook als het antwoord iets anders zegt; een antwoord dat hetzelfde in andere woorden zegt, krijgt minder punten.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_longer = L.guess(
        "guess",
        question="SciTLDR heeft voor elk van de 618 papers de TLDR van de auteurs en één of meer TLDR's die experts schreven op grond van de peer reviews. Scoor de TLDR van de auteurs tegen elke TLDR van de experts: welk deel van de woorden van een TLDR van de experts komt erin terug? Houd per paper het beste deel. Doe hetzelfde met de ene zin die de titelmethode van *Samenvatten: kiezen of schrijven* uit het abstract kiest. Welke van de twee haalt gemiddeld het grootste deel?",
        options={
            "far": "De TLDR van de auteurs, met ruime afstand: deze TLDR is als TLDR geschreven, net als de TLDR's van de experts",
            "little": "De TLDR van de auteurs, met een paar punten",
            "right": "De zin die uit het abstract is gekozen",
            "same": "Allebei precies hetzelfde deel",
        },
        reveal="De gekozen zin: gemiddeld komt 31,7% van de woorden van de best passende TLDR van de experts erin terug, tegen 28,7% bij de TLDR van de auteurs, over de 618 papers. De gekozen zin is langer: gemiddeld 25,5 woorden, tegen 19,6 voor de TLDR van de auteurs en 22,4 voor een TLDR van de experts. Een langere samenvatting heeft meer kans om een woord van de referentie te bevatten, en een score die alleen telt hoeveel van de referentie terugkomt, let niet op wat de samenvatting toevoegt: de langste zin van elk abstract, gemiddeld 35,6 woorden, haalt 31,1%. De demo verderop in deze les laat een score zien die daar wel op let.",
    )
    guess_longer
    return (guess_longer,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Lin definieert ROUGE-N formeel als een recall van n-grams tussen een kandidaatsamenvatting en een verzameling referentiesamenvattingen. De **kandidaatsamenvatting** is de samenvatting die je scoort; een **referentiesamenvatting** is een samenvatting die mensen schreven, zoals de TLDR's van SciTLDR.

    1. **ROUGE-N** telt de n-grams van de referentie die terugkomen in de kandidaat, afgetopt zoals bij BLEU: een n-gram telt hoogstens zo vaak mee als het in allebei staat. ROUGE-N deelt door het aantal n-grams van de referentie, waar BLEU deelt door de n-grams van de vertaling: juist deze deling maakt er een recall van, zoals Lin opmerkt. ROUGE-1 telt woorden, ROUGE-2 bigrams.
    2. **Meer dan één referentie.** Lin scoort de kandidaat dan tegen elke referentie apart, en neemt het maximum van deze scores als de uiteindelijke score.
    3. **ROUGE-L** kijkt naar de volgorde van de woorden. De langste gemeenschappelijke deelrij (LCS, *longest common subsequence*) van twee rijen is volgens Lin een gemeenschappelijke deelrij met de grootste lengte. Een **deelrij** houdt de volgorde van een zin aan maar mag woorden weglaten: *the bees hive* is een deelrij van *the bees at the hive*. Het voordeel is volgens Lin dat de woorden niet direct op elkaar hoeven te volgen, als de volgorde maar klopt, zodat de maat iets van de woordvolgorde van de hele zin ziet. ROUGE-L-recall is de lengte van de LCS gedeeld door de lengte van de referentie, en de precision dezelfde lengte gedeeld door de lengte van de kandidaat. Lin combineert de twee in een F-score met een β, zoals chrF doet, en schrijft dat β bij DUC, destijds de jaarlijkse vergelijking van samenvatters, zo groot was dat alleen recall telde. De LCS komt uit een tabel, net als de edit distance.
    4. **ROUGE-S** telt **skip-bigrams**: volgens Lin elk paar woorden in de volgorde van de zin, met willekeurig veel woorden ertussen. Een zin van $n$ woorden heeft er $n(n - 1) / 2$. In het voorbeeld van Lin is de referentie S1, *police killed the gunman* (de politie doodde de schutter), en worden drie zinnen ertegen gescoord: S2, *police kill the gunman*; S3, *the gunman kill police*; en S4, *the gunman police killed*. Elke zin heeft 6 skip-bigrams; S2 deelt er 3 met S1, S3 één en S4 twee. Lin geeft S2 daarmee een ROUGE-S van 0,5, S3 0,167 en S4 0,333. Lin berekent deze scores als F-score met β = 1 uit de recall en de precision van de skip-bigrams, net als bij ROUGE-L; omdat elke zin 6 skip-bigrams heeft, zijn recall, precision en F-score hier gelijk. Anders dan bij BLEU hoeven de woorden volgens Lin niet direct op elkaar te volgen, terwijl de maat toch op de woordvolgorde let.

    Alle drie vergelijken woorden, en alleen woorden. In 2019 schreven Wojciech Kryściński en collega's van Salesforce Research dat ROUGE-scores alleen exact gelijke tokens tellen, zodat de overlap tussen zinsdelen met dezelfde betekenis niet meetelt. Over de manier waarop samenvattingen werden gescoord, met ROUGE als middelpunt, schreef het team dat deze manier zwak samenhangt met het oordeel van mensen, en geen rekening houdt met belangrijke eigenschappen zoals feitelijke juistheid.

    Het plaatje scoort de samenvatting van Qwen3-0.6B van het abstract over bijen, uit *Samenvatten: kiezen of schrijven*, tegen de TLDR die de course schreef, met ROUGE-1, ROUGE-2 en ROUGE-L. Net als daar is een woord een reeks letters, in kleine letters: *4%* bevat geen woord, en *count* en *counts* zijn twee verschillende woorden.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **ROUGE** | een familie maten voor hoeveel van een referentiesamenvatting terugkomt in een samenvatting (Lin, 2004) |
    | **kandidaatsamenvatting** | de samenvatting die gescoord wordt |
    | **referentiesamenvatting** | een samenvatting die mensen schreven, waarmee je de kandidaat vergelijkt |
    | **ROUGE-N** | de recall van de n-grams van de referentie: hoeveel er terugkomen in de kandidaat, afgetopt, gedeeld door hoeveel de referentie er heeft |
    | **deelrij** | woorden van een zin in de volgorde van de zin, met willekeurig veel woorden ertussen weggelaten |
    | **langste gemeenschappelijke deelrij** (LCS) | de langste deelrij die in allebei de zinnen staat |
    | **ROUGE-L** | de lengte van de LCS gedeeld door de lengte van de referentie (recall), of door de lengte van de kandidaat (precision) |
    | **skip-bigram** | een paar woorden van een zin in de volgorde van de zin, met willekeurig veel woorden ertussen |
    | **ROUGE-S** | een score uit de skip-bigrams die de kandidaat met de referentie deelt: Lin combineert de recall en de precision ervan in een F-score |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo scoor je een samenvatting tegen een referentie met ROUGE:

    1. Splits allebei in woorden, in kleine letters.
    2. ROUGE-N: tel de n-grams van allebei. Tel elk n-gram van de referentie zo vaak mee als het in allebei staat: de kleinste van de twee tellingen. Deel de uitkomst door het aantal n-grams van de referentie.
    3. ROUGE-L: vul een tabel met een rij voor elk beginstuk van de samenvatting en een kolom voor elk beginstuk van de referentie. Een cel krijgt de cel linksboven plus 1 als de twee woorden gelijk zijn, en anders de grootste van de cel erboven en de cel links ervan. Deel de laatste cel door de lengte van de referentie.
    4. ROUGE-S: maak van allebei elk paar woorden in de volgorde van de zin, tel de paren van de referentie die terugkomen zoals in stap 2, en deel door het aantal paren van de referentie: de recall. Lin deelt ook door het aantal paren van de samenvatting, voor de precision, en combineert de twee in een F-score.
    5. Bij meer dan één referentie scoor je tegen elke referentie, en houd je de hoogste score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem S1 van Lin als referentie en S2 als samenvatting. `C.ngram_counts` telt de n-grams van een lijst woorden, elk n-gram geschreven als de woorden met een spatie ertussen, zoals in *Een vertaling scoren: BLEU en chrF*. De cel print de unigrams en de bigrams van de referentie, elke regel gevolgd door de n-grams van de samenvatting:
    """)
    return


@app.cell
def example_cell():
    _reference = "police killed the gunman".split()
    _summary = "police kill the gunman".split()
    for _n in (1, 2):
        print(C.ngram_counts(_reference, _n))
        print(C.ngram_counts(_summary, _n))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Drie van de vier woorden van de referentie komen terug: *police*, *the* en *gunman*, dus ROUGE-1-recall is 3 / 4 = 0,75; *kill* is niet *killed*. Van de drie bigrams van de referentie komt alleen *the gunman* terug: ROUGE-2-recall is 1 / 3, ongeveer 0,33. De LCS is *police the gunman*, 3 woorden, dus ROUGE-L-recall is ook 3 / 4. Allebei de zinnen hebben 6 skip-bigrams, en S2 deelt er 3 met S1: *police the*, *police gunman* en *the gunman*. ROUGE-S-recall is 3 / 6 = 0,5; S2 heeft ook 6 paren, dus de precision en de F-score van Lin zijn ook 0,5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rouge():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. rouge_n: hoeveel van de referentie terugkomt

        Schrijf `rouge_n`, die `mine` krijgt, de tellingen van de n-grams van een samenvatting, en `theirs`, de tellingen van een referentie, allebei dictionaries zoals `C.ngram_counts` geeft; de referentie heeft minstens één n-gram. Geef ROUGE-N-recall terug: tel elk n-gram van de referentie zo vaak mee als allebei de dictionaries het hebben, en deel door het aantal n-grams van de referentie. In de startcode gaat de loop langs de n-grams van de referentie: `total` telt op hoe vaak elk n-gram voorkomt, en `matched` groeit bij elk n-gram dat de samenvatting ook heeft. Vervang de twee `...`.

        De check roept `rouge_n({"the": 3, "cat": 1}, {"the": 2, "cat": 1, "sat": 1, "down": 1})` aan, wat `0.6` geeft: *the* telt twee keer mee, omdat de referentie het twee keer heeft, en *cat* één keer, dus 3 van de 5 unigrams van de referentie komen terug. Daarna `rouge_n({"een": 1, "klein": 1, "netwerk": 1}, {"een": 1, "netwerk": 1, "telt": 1, "bijen": 1})`, met Nederlandse woorden, wat `0.5` geeft, en eigen gevallen: een samenvatting zonder n-grams, een woord dat de samenvatting minder vaak heeft dan de referentie, bigrams, en de samenvattingen van Qwen3-0.6B tegen de TLDR's van de course.
        """),
        L.task("Geef het deel van de n-grams van een referentie dat terugkomt in een samenvatting, afgetopt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rouge():
    box_rouge = L.code_box("rouge_n", lines=8)
    box_rouge
    return (box_rouge,)


@app.cell(hide_code=True)
def check_rouge(box_rouge):
    result_rouge = L.check("rouge_n", box_rouge.value)
    result_rouge
    return (result_rouge,)


@app.cell(hide_code=True)
def hints_rouge():
    L.hints(
        "rouge_n",
        [
            "`gram` is één n-gram van de referentie, en `theirs[gram]` is hoe vaak de referentie het heeft; `mine` bevat de tellingen van de samenvatting. `matched` groeit met één n-gram tegelijk, zoals een lopend totaal. `min` geeft het kleinste van de getallen die de functie krijgt: `min(3, 2)` is 2.",
            "Een n-gram telt zo vaak mee als de samenvatting het heeft, maar nooit vaker dan de referentie het heeft: de kleinste van de twee tellingen. Zoek de telling van de samenvatting op in de dictionary van de samenvatting. Recall deelt wat gevonden is door alle n-grams van de referentie, en `total` telt deze n-grams op.",
            "`            matched = matched + min(___, theirs[gram])`",
        ],
        why="`matched = matched + min(mine[gram], theirs[gram])` en `return matched / total`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rouge():
    mo.md(r"""
    De samenvatting van Qwen3-0.6B heeft bij 3 van de zes abstracts een hogere ROUGE-1-recall dan de zin die de titelmethode kiest. Bij het abstract over lekken haalt de geschreven samenvatting 57%, meer dan elke zin van het abstract: de dichtstbijzijnde zin haalt 48%. De samenvatting zegt ook dat het netwerk beter presteert dan traditionele methoden, en dat zegt het abstract niet; voor ROUGE-1-recall kosten woorden die niet in de TLDR staan, niets. In alle zes abstracts is de dichtstbijzijnde zin van *Samenvatten: kiezen of schrijven* de zin met de hoogste ROUGE-1-recall.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lcs():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. lcs: de langste gemeenschappelijke deelrij

        Schrijf `lcs`, die `a` en `b` krijgt, twee lijsten met woorden, en de lengte van de langste gemeenschappelijke deelrij teruggeeft. De tabel werkt net als de tabel van de edit distance: `L[i, j]` is de lengte van de LCS van de eerste `i` woorden van `a` en de eerste `j` woorden van `b`, en de eerste rij en de eerste kolom blijven 0, omdat een stuk zonder woorden niets deelt. De startcode maakt de tabel met `np.zeros`, vult de tabel rij voor rij, en geeft de laatste cel terug. Als woord `i` van `a` en woord `j` van `b`, geteld vanaf 1, gelijk zijn, kan de LCS van de twee stukken op dat woord eindigen; als de woorden verschillen, laat de LCS het laatste woord van `a` of het laatste woord van `b` weg. Vervang de twee `...`.

        De check roept `lcs(["police", "kill", "the", "gunman"], ["police", "killed", "the", "gunman"])` aan, wat `3` geeft, voor *police the gunman*. Daarna `lcs(["de", "bijen", "tellen"], ["het", "netwerk", "telt", "de", "bijen"])`, met Nederlandse woorden, wat `2` geeft, en eigen gevallen: een lege lijst aan elke kant, S3 en S4 van Lin tegen S1, een woord dat twee keer voorkomt, en de samenvattingen van Qwen3-0.6B tegen de TLDR's van de course.
        """),
        L.task("Vul de tabel met langste gemeenschappelijke deelrijen, en geef de laatste cel terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lcs():
    box_lcs = L.code_box("lcs", lines=9)
    box_lcs
    return (box_lcs,)


@app.cell(hide_code=True)
def check_lcs(box_lcs):
    result_lcs = L.check("lcs", box_lcs.value)
    result_lcs
    return (result_lcs,)


@app.cell(hide_code=True)
def hints_lcs():
    L.hints(
        "lcs",
        [
            "`L[i, j]` is de lengte van de LCS van de eerste `i` woorden van `a` en de eerste `j` woorden van `b`, zoals `D[i, j]` een edit distance bevatte. `max` geeft het grootste van de getallen die de functie krijgt, zoals `min` het kleinste geeft.",
            "Als de twee woorden gelijk zijn, wordt de LCS van de stukken zonder deze twee woorden één woord langer: de cel één rij hoger en één kolom naar links, plus 1. Als de woorden verschillen, staat de LCS zonder het woord van `a` in de cel erboven, en de LCS zonder het woord van `b` in de cel links ervan; de cel houdt de grootste van de twee.",
            "`                L[i, j] = max(L[i - 1, j], ___)`",
        ],
        why="`L[i, j] = L[i - 1, j - 1] + 1` en `L[i, j] = max(L[i - 1, j], L[i, j - 1])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_lcs():
    mo.md(r"""
    De samenvatting van het abstract over lekken haalt een ROUGE-L-recall van 24%, tegen een ROUGE-1-recall van 57%: 12 van de 21 woorden van de TLDR komen terug, maar slechts 5 daarvan in de volgorde van de TLDR. De samenvatting begint met de woorden van de titel, *Finding leaks in water pipes by listening*, terwijl de TLDR eerst het netwerk noemt. De samenvatting van het abstract over rivieren houdt de volgorde van de TLDR aan: ROUGE-L-recall en ROUGE-1-recall zijn allebei 52%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_skip():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. skip_bigrams: elk paar in de volgorde van de zin

        Schrijf `skip_bigrams`, die `words` krijgt, een lijst met woorden, en elk paar woorden in de volgorde van de lijst teruggeeft, elk paar als een tuple van twee: het woord op plaats `i` met elk woord op een latere plaats. In de startcode neemt de buitenste loop elke plaats `i`, de binnenste loop moet elke latere plaats `j` nemen, en `pairs.append` voegt één paar toe. Vervang de twee `...`.

        De check roept `skip_bigrams(["police", "killed", "the", "gunman"])` aan, wat de 6 paren `[("police", "killed"), ("police", "the"), ("police", "gunman"), ("killed", "the"), ("killed", "gunman"), ("the", "gunman")]` geeft. Daarna `skip_bigrams(["de", "politie", "schoot"])`, met Nederlandse woorden, wat 3 paren geeft, en eigen gevallen: geen woorden, één woord, een woord dat twee keer voorkomt, vijf letters, en de TLDR over plastic. De paren mogen in elke volgorde staan.
        """),
        L.task("Maak elk paar woorden van een lijst, in de volgorde van de lijst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_skip():
    box_skip = L.code_box("skip_bigrams", lines=6)
    box_skip
    return (box_skip,)


@app.cell(hide_code=True)
def check_skip(box_skip):
    result_skip = L.check("skip_bigrams", box_skip.value)
    result_skip
    return (result_skip,)


@app.cell(hide_code=True)
def hints_skip():
    L.hints(
        "skip_bigrams",
        [
            "`pairs.append` voegt één item toe aan `pairs`, zoals in *Lijsten*, en `words[i]` is het woord op plaats `i`. Een tuple van twee schrijf je met haakjes: `(\"police\", \"the\")`. `range(start, stop)` geeft de gehele getallen van `start` tot `stop`, zonder `stop`.",
            "Het tweede woord van een paar staat op een latere plaats dan `i`: de eerste latere plaats is de plaats direct na `i`, en de laatste is de laatste plaats van de lijst. Het paar bevat eerst het woord op `i`, en dan het woord op de latere plaats.",
            "`            pairs.append((words[i], ___))`",
        ],
        why="`for j in range(i + 1, len(words)):` en `pairs.append((words[i], words[j]))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_skip():
    mo.md(r"""
    S4, *the gunman police killed*, heeft elk woord van S1: ROUGE-1-recall is 1. Maar slechts 2 van de 6 paren houden de volgorde van S1 aan, *the gunman* en *police killed*, dus ROUGE-S is 0,333, zoals Lin uitrekende. S3, *the gunman kill police*, zegt het omgekeerde van S1 (de schutter doodt de politie) en deelt 1 paar met S1. Lin waarschuwt ook dat zonder grens aan de afstand tussen de woorden paren van kleine woorden die niets met elkaar te maken hebben, zoals *the the* of *of in*, als geldige overeenkomsten kunnen meetellen.

    ### Probeer het: 618 papers, elk met vier samenvattingen

    SciTLDR, uit *Samenvatten: kiezen of schrijven*, heeft voor elke paper de TLDR van de auteurs en één of meer TLDR's die experts schreven op grond van de peer reviews. De demo scoort vier samenvattingen van elke paper tegen de TLDR's van de experts: de TLDR van de auteurs, de zin die de titelmethode kiest, de eerste zin van het abstract en de langste zin. Van elke samenvatting telt de beste score over deze TLDR's. Naast recall laat de demo de **F-score** van ROUGE-1 met β = 1 zien, zoals in *Een vertaling scoren: BLEU en chrF*: precision en recall tellen even zwaar. Kies een paper met de slider.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciTLDR, 1,2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scitldr_cell(get_button):
    sci_papers = C.scitldr_or_stop(get_button, {
        "ask": "*De testset van SciTLDR staat nog niet op je computer. De knop downloadt de set één keer van de repository op GitHub, 1,2 MB, onder de licentie Apache-2.0. De rest van de les werkt zonder.*",
        "busy": "**Bezig met downloaden** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: GitHub was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is op je schijf geen plek voor 1,2 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    sci_paper = mo.ui.slider(1, len(sci_papers), value=C.SCI_START, label="Paper", show_value=True)
    sci_paper
    return sci_paper, sci_papers


@app.cell(hide_code=True)
def scitldr_look(sci_paper, sci_papers):
    mo.Html(C.scitldr_picture(L.lang, sci_papers, sci_paper.value))
    return


@app.cell(hide_code=True)
def text_after_scitldr():
    mo.md(r"""
    Op ROUGE-1-recall verslaat de zin van de titelmethode de TLDR van de auteurs bij 323 van de 618 papers, en de TLDR van de auteurs de zin bij 229. Met de F-score staat de TLDR van de auteurs bovenaan, met 28,5%, net boven de zin van de titelmethode met 27,8%, en de langste zin, die op recall bijna zo goed was als de beste, zakt naar 23,3%, omdat de meeste woorden van deze zin niet in een TLDR van de experts staan. Recall alleen beloont een lange samenvatting; precision neemt deze beloning weer weg.

    ### Probeer het: je eigen samenvatting

    Kies een abstract, en schrijf er een samenvatting van in het vak; het vak begint met de samenvatting van Qwen3-0.6B. De course scoort de jouwe tegen de TLDR, naast de samenvatting van Qwen3-0.6B en de dichtstbijzijnde zin. Probeer eerst een hoge ROUGE-1-recall te halen met een samenvatting die niet klopt: zet de woorden van de TLDR in een andere volgorde, of voeg *not* toe. Schrijf daarna een goede samenvatting in eigen woorden, en kijk wat je ermee scoort.
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    try_abstract = mo.ui.dropdown(options={C.ABSTRACT_NAMES[L.lang][_key]: _key for _key in C.ABSTRACTS},
                                  value=C.ABSTRACT_NAMES[L.lang]["leaks"], label="Abstract")
    try_abstract
    return (try_abstract,)


@app.cell(hide_code=True)
def try_write(try_abstract):
    try_text = mo.ui.text_area(value=C.QWEN[try_abstract.value], label="Je samenvatting", rows=3, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_abstract, try_text):
    mo.Html(C.try_picture(L.lang, try_abstract.value, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ### Oefening 4. Wat ziet ROUGE-1?

    De referentie is *A small network on a cheap board counts the bees at a hive.* (een klein netwerk op een goedkoop bordje telt de bijen bij een bijenkast), en de samenvatting *A small network counts the bees at a hive.*: 9 van de 13 woorden van de referentie komen terug, dus ROUGE-1-recall is ongeveer 0,69. Elke verandering hieronder maakt een nieuwe samenvatting. Zet elke verandering bij het effect op ROUGE-1-recall.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl) topte de n-grams van een vertaling af en deelde door de vertaling, een precision; ROUGE topt op dezelfde manier af en deelt door de referentie, een recall, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl). [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl) vulde een tabel met beginstukken met een minimum; de langste gemeenschappelijke deelrij vult er een met een maximum. [Samenvatten: kiezen of schrijven](/course/open/nlp-summarize?lang=nl) zocht de zin die het dichtst bij de TLDR's ligt, en bij de zes abstracts van de course blijkt het de zin met de hoogste ROUGE-1-recall te zijn. De volgende les, *Eenvoudige taal en leesbaarheid*, scoort een tekst zonder enige referentie, op grond van de lengte van de zinnen en de woorden.

    **Een voorbeeld uit het dagelijks leven.** Software die naar overgeschreven huiswerk zoekt, werkt net als ROUGE: de software telt de n-grams die in allebei de teksten staan, en markeert een tekst met veel gedeelde n-grams, wat de tekst ook zegt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_rouge, result_lcs, result_skip, result_changes):
    L.wrap_up(result_rouge, result_lcs, result_skip, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
