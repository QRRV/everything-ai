# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-sentiment-lexicon. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sentiment-lexicon", "nl")
    C = L.checks   # de zinnen uit recensies in het Engels en het Nederlands, de twee lexicons, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Sentiment: positief of negatief?](/course/open/nlp-sentiment?lang=nl), en een over [WordNet: een netwerk van betekenissen](/course/open/nlp-wordnet?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scales():
    quick_scales = L.choice(
        "quick-scales",
        question="In *Sentiment: positief of negatief?* krijgt elk woord een scale: 0,5 vóór de laatste *maar*, en 1 vanaf deze *maar*. Welke scales geeft *Mooi maar luid, maar wel goed.* aan de zes woorden?",
        options={
            "first": "0,5; 1; 1; 1; 1; 1",
            "right": "0,5; 0,5; 0,5; 1; 1; 1",
            "but": "0,5; 0,5; 0,5; 0,5; 1; 1",
            "after": "1; 1; 1; 1; 0,5; 0,5",
        },
        correct="right",
        why={
            "right": "De laatste *maar* is het vierde woord, dus *mooi*, de eerste *maar* en *luid* krijgen 0,5, en de laatste *maar*, *wel* en *goed* krijgen 1. Een scale verandert hoe zwaar een woord telt. In deze les krijgt een woord de score uit een lijst, en worden de scores van de twee woorden na *niet* omgedraaid.",
            "first": "De halve weight houdt op bij de laatste *maar*, niet bij de eerste. Waar staat de laatste *maar* in deze zin? Zie [Sentiment: positief of negatief?](/course/open/nlp-sentiment?lang=nl).",
            "but": "De laatste *maar* zelf telt al helemaal mee. Welke woorden komen ervóór? Zie [Sentiment: positief of negatief?](/course/open/nlp-sentiment?lang=nl).",
            "after": "De woorden na *maar* dragen het oordeel en houden dus de volle weight. Welke woorden krijgen de helft? Zie [Sentiment: positief of negatief?](/course/open/nlp-sentiment?lang=nl).",
        },
    )
    quick_scales
    return (quick_scales,)


@app.cell(hide_code=True)
def quick_check_senses():
    quick_senses = L.choice(
        "quick-senses",
        question="In *WordNet: een netwerk van betekenissen* staat het bijvoeglijk naamwoord *ridiculous* (belachelijk) in drie synsets: `pathetic.s.03`, `absurd.s.02` en `farcical.s.01`. Wat zegt dat?",
        options={
            "hyper": "*ridiculous* heeft drie hyperoniemen, elk een stap algemener",
            "same": "Drie woorden betekenen precies hetzelfde als *ridiculous*",
            "right": "*ridiculous* heeft drie betekenissen",
            "count": "*ridiculous* komt drie keer voor in de teksten waaruit WordNet is gemaakt",
        },
        correct="right",
        why={
            "right": "Een synset bevat de woorden die één betekenis delen, dus een woord in drie synsets heeft drie betekenissen: zielig, absurd en kluchtig. Een lijst met goede en slechte woorden moet dan kiezen. De Engelse lijst van Pattern, die deze les gebruikt, geeft *ridiculous* een score per betekenis: −1,0 voor wat minachtend medelijden wekt, +1,0 voor uitbundig grappig en −1,0 voor ongerijmd.",
            "hyper": "Bijvoeglijke naamwoorden hebben in WordNet helemaal geen hyperoniemen, en deze drie zijn de synsets waar *ridiculous* zelf in staat. Waar staat elke synset van een woord voor? Zie [WordNet: een netwerk van betekenissen](/course/open/nlp-wordnet?lang=nl).",
            "same": "Elk van de drie synsets bevat ook andere woorden, zoals *absurd* en *ludicrous*, en elke synset is anders. Waar staat elk van de drie voor? Zie [WordNet: een netwerk van betekenissen](/course/open/nlp-wordnet?lang=nl).",
            "count": "WordNet telt niet hoe vaak een woord voorkomt: WordNet geeft de betekenissen van woorden, en de woorden die elke betekenis delen. Wat heeft een woord in drie synsets? Zie [WordNet: een netwerk van betekenissen](/course/open/nlp-wordnet?lang=nl).",
        },
    )
    quick_senses
    return (quick_senses,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De classifier van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) leerde een weight voor elk woord uit 2.400 gelabelde zinnen. Lang vóór zulke classifiers schreven mensen met de hand lijsten met woorden. Jurafsky en Martin beschrijven in *Speech and Language Processing* een ander soort model: in plaats van elk woord te gebruiken, kijk je alleen naar woorden die sterk wijzen op een gevoel of een oordeel. Zulke lijsten heten bij Jurafsky en Martin *affective lexicons* of *sentiment lexicons*. De oudste lijst die Jurafsky en Martin noemen, de General Inquirer uit 1966, heeft 1.915 positieve en 2.291 negatieve woorden.

    Zulke lijsten worden nog steeds gebruikt. Tausczik en Pennebaker (2010) beschrijven LIWC als een doorzichtig programma dat woorden telt in groepen met een psychologische betekenis. Het eerste doel was volgens Tausczik en Pennebaker bescheiden: de computer moest alleen het percentage positieve en negatieve emotiewoorden in een tekst uitrekenen. LIWC wordt verkocht, met een licentie voor academisch onderzoek en een voor alle anderen. Voor VADER (Hutto en Gilbert, 2014), gemaakt voor sociale media, gaven tien mensen elk woord een cijfer van −4 tot +4, en er bleven iets meer dan 7.500 woorden en tekens over, zoals emoticons. VADER is gratis, onder de MIT-licentie. Pattern, een Python-pakket van de Universiteit Antwerpen, heeft een lijst voor het Engels en een voor het Nederlands, met een score voor elke betekenis van een woord; deze les gebruikt de Nederlandse.

    Een lijst heeft geen eigen gelabelde zinnen nodig, en je kunt de lijst lezen: je ziet waarom een zin een label krijgt. Volgens Jurafsky en Martin kan een lijst beter werken als er weinig trainingsdata is, of als de trainingsdata niet lijkt op de teksten die je wilt labelen. Deze les bouwt een lijst uit de gelabelde zinnen van de recensies, legt het resultaat naast de lijst van Pattern, en kijkt naar een woord dat andere woorden omdraait: *niet*.

    **Een voorbeeld uit het dagelijks leven.** Als je naar een tweedehands auto kijkt, scan je de advertentie op woorden: *goed onderhouden* en *eerste eigenaar* zijn goede tekens, *opknapper* en *start slecht* slechte. Je hebt een lijst met goede en slechte woorden in je hoofd, en je telt op wat je vindt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_lexicon = L.guess(
        "guess",
        question="Tel voor elk woord hoe vaak het in de positieve en in de negatieve trainingszinnen voorkomt, en geef het woord een score uit de twee aantallen. Een zin krijgt dan de som van de scores van de woorden: boven 0 positief, onder 0 negatief. De classifier van *Logistische regressie op woorden* geeft 480 van de 600 achtergehouden zinnen het eigen label. Hoeveel krijgt zo'n lijst goed?",
        options={
            "half": "Ongeveer 300, nauwelijks beter dan een munt",
            "fewer": "Ongeveer 430: duidelijk minder, want er wordt niets getraind",
            "right": "Ongeveer 480: evenveel als de classifier",
            "all": "Ongeveer 560, want de woorden komen uit dezelfde recensies",
        },
        reveal="**482: twee meer dan de classifier.** Hier worden geen weights samen geleerd: elk woord krijgt de score uit de eigen twee aantallen. De classifier weegt TF-IDF-waarden en leert alle weights samen, en toch komen de twee dicht bij elkaar uit. De Nederlandse lijst van Pattern, deels met de hand gemaakt uit boekrecensies, krijgt er 313 goed, en zegt niets over 174 zinnen, waar de som 0 is: in 156 daarvan staat geen enkel woord op de lijst.",
    )
    guess_lexicon
    return (guess_lexicon,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    Een **sentiment lexicon** is een lijst woorden met elk een score: boven 0 voor positief, onder 0 voor negatief. Hutto en Gilbert omschrijven het als een lijst woorden die meestal een label positief of negatief hebben, naar de kant waar elk woord naar wijst. Er zijn twee manieren om er een te maken.

    - **Met de hand.** Mensen geven elk woord een score. Voor de Nederlandse lijst van Pattern namen De Smedt en Daelemans (2012) de bijvoeglijke naamwoorden uit 14.000 Nederlandse boekrecensies: de 1.100 die het vaakst voorkwamen, alle bijvoeglijke naamwoorden die meer dan vier keer voorkwamen. Zeven mensen gaven elke betekenis van elk bijvoeglijk naamwoord een score van −1 tot +1, en de lijst houdt het gemiddelde. Pattern noemt deze score de **polarity**. Later kwamen er bijvoeglijke naamwoorden bij waarvan een programma de score afleidde: het bestand markeert 1.595 van de 3.918 betekenissen als met de hand gescoord en 2.113 als afgeleid. De course geeft een woord het gemiddelde over de betekenissen.
    - **Uit gelabelde recensies.** Tel hoe vaak elk woord in de positieve en in de negatieve trainingszinnen voorkomt. Volgens Jurafsky en Martin staan positieve woorden vaker in recensies met vijf sterren, en negatieve woorden in recensies met één ster. Deze les geeft een woord de score $\log_{10}\frac{p + 1}{n + 1}$, met $p$ het aantal keer in positieve en $n$ het aantal keer in negatieve zinnen. Een woord dat aan beide kanten even vaak voorkomt, krijgt 0; een woord dat vaker in positieve zinnen staat, krijgt meer dan 0. De +1 voorkomt dat een woord dat maar aan één kant voorkomt een score zonder einde krijgt.

    Een zin krijgt de som van de scores van de woorden; een woord dat de lijst niet heeft, telt niet mee. Boven 0 is positief, onder 0 negatief, en bij precies 0 zegt de lijst niets.

    Dan **ontkenning**: *niet goed* is niet goed. Pang, Lee en Vaithyanathan (2002) vonden het duidelijk dat *good* en *not very good* (goed en niet erg goed) een tegengesteld oordeel geven. De oplossing van Pang en collega's: elk woord tussen een ontkenning, zoals *not*, *isn't* of *didn't*, en het eerstvolgende leesteken krijgt het voorvoegsel NOT_. VADER kijkt vanaf elk woord drie woorden terug, en vangt zo volgens Hutto en Gilbert bijna 90% van de gevallen waarin een ontkenning het oordeel van de tekst omdraait. Voor het Nederlands zochten De Smedt en Daelemans naar *niet*, *nooit* en *geen*: de polarity van het bijvoeglijk naamwoord erna omdraaien kan de precision 2 tot 3% hoger maken.

    In deze les draait een ontkenning de score van de volgende twee woorden om: de twee woorden tellen met het omgekeerde teken. Het ontkenningswoord zelf houdt de eigen score. De afbeelding laat één achtergehouden zin zien in beide lijsten.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    De lijst van Pattern kent één woord van de zin, *goed*, en zegt positief; met de score van *goed* omgedraaid zegt de lijst negatief, net als het label van de zin. De lijst uit de recensies zegt zonder regel al negatief: *niet* zelf heeft een score van −0,62, want *niet* staat 348 keer in negatieve zinnen en 82 keer in positieve, en *geld* heeft −0,83. Met de regel tellen *goed* en *voor* andersom, en de som gaat van −0,88 naar −2,15.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **sentiment lexicon** | een lijst woorden met elk een score: boven 0 voor positief, onder 0 voor negatief |
        | **polarity** | de naam die Pattern gebruikt voor de score van een woord, van −1 (negatief) tot +1 (positief) |
        | **ontkenning** | een woord zoals *niet*, *geen* of *nooit*, dat omdraait wat de woorden erna zeggen |
        | $\log_{10}\frac{p + 1}{n + 1}$ | de score van een woord in de lijst uit de recensies, met $p$ het aantal keer in positieve en $n$ het aantal keer in negatieve trainingszinnen |
        | `C.training(lang)` | de 2.400 trainingszinnen in het Engels (`"en"`) of het Nederlands (`"nl"`), als lijst van `[label, words]`: het label 1 voor positief en 0 voor negatief, de woorden in kleine letters |
        | `C.counts(lang)` | voor elk woord van de trainingszinnen een lijst met de twee aantallen: eerst in negatieve zinnen, dan in positieve |
        | `C.lexicon(lang)` | de lijst van de course uit de trainingszinnen: een dictionary met een score voor elk woord |
        | `C.pattern(lang)` | de lijst van Pattern: een dictionary met een polarity voor elk woord, het gemiddelde over de betekenissen, eerst per woordsoort, zoals Pattern doet |
        | `C.NEGATIONS[lang]` | de ontkenningswoorden van deze les: in het Nederlands *niet*, *geen*, *nooit*, *niets*, *niemand* en *nergens*; in het Engels *not*, *no*, *never* en woorden zoals *didn't* |
        """),
        L.watch_out("De code van Pattern zelf doet meer dan deze les. Bij het laden van de Nederlandse lijst voegt Pattern van elk bijvoeglijk naamwoord de vorm toe die vóór een zelfstandig naamwoord staat, zoals *slechte* naast *slecht*; Pattern neemt een score na een ontkenning keer −0,5 in plaats van de score om te draaien; en Pattern laat woorden zoals *ontzettend* de volgende score sterker maken. De course zoekt woorden op zoals de woorden geschreven staan, dus de getallen voor de lijsten van Pattern zijn hier anders dan bij Pattern zelf."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Tel voor elk woord hoe vaak het in negatieve en in positieve trainingszinnen voorkomt: elke keer, ook twee keer in één zin.
    2. Geef elk woord de score $\log_{10}\frac{p + 1}{n + 1}$.
    3. Geef een zin de som van de scores van de woorden die de lijst heeft.
    4. Met ontkenning: de twee woorden vlak na een ontkenning tellen met het omgekeerde teken.
    5. Boven 0 positief, onder 0 negatief; bij 0 zegt de lijst niets. Tel hoeveel achtergehouden zinnen het eigen label krijgen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code neemt de vier woorden in `C.EXAMPLE_WORDS` voor de taal van de les, en print elk woord met de twee aantallen uit `counts` van de course, negatief eerst, en de score in `lexicon` van de course, afgerond op drie decimalen.
    """)
    return


@app.cell
def example_cell():
    counts = C.counts(L.lang)
    scores = C.lexicon(L.lang)
    for word in C.EXAMPLE_WORDS[L.lang]:
        print(word, counts[word], round(scores[word], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *uitstekend* staat nooit in een negatieve zin. Zonder de +1 zou de breuk door 0 delen; met de +1 is de score $\log_{10}\frac{28 + 1}{0 + 1} = \log_{10} 29 = 1{,}462$. *het* staat ongeveer even vaak aan beide kanten, 463 en 401 keer, dus de score ligt dicht bij 0: $\log_{10}\frac{402}{464} = -0{,}062$. *niet* en *geld* neigen naar negatief: $\log_{10}\frac{83}{349} = -0{,}624$ en $\log_{10}\frac{4}{27} = -0{,}829$.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lexicon():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. lexicon: een lijst uit gelabelde zinnen

        Schrijf `lexicon(sentences)`, waarin `sentences` een lijst zinnen is, elk een paar `[label, words]`: het label, 1 voor positief en 0 voor negatief, en de lijst woorden van de zin in kleine letters. Geef een dictionary terug met een score voor elk woord: $\log_{10}\frac{p + 1}{n + 1}$, met $p$ het aantal keer in positieve en $n$ het aantal keer in negatieve zinnen. Een woord telt elke keer dat het voorkomt. Je kunt `np` gebruiken. Vervang de `...`. De check roept `lexicon([[1, ["great", "food"]], [0, ["bad", "food"]], [1, ["great", "service"]]])` aan, dat `{"great": 0.477, "food": 0, "bad": -0.301, "service": 0.301}` geeft (hier afgerond), en `lexicon([[0, ["bad", "bad", "food"]], [1, ["good", "food"]]])`, dat `{"bad": -0.477, "food": 0, "good": 0.301}` geeft. Daarna eigen gevallen: helemaal geen zinnen; één positieve zin zonder woorden; drie negatieve zinnen van één woord, *slow*; één positieve zin met drie keer *nice*; en drie Nederlandse zinnen die met een negatieve beginnen.
        """),
        L.task("Bouw uit gelabelde zinnen een lijst met een score voor elk woord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lexicon():
    box_lexicon = L.code_box("lexicon", lines=16)
    box_lexicon
    return (box_lexicon,)


@app.cell(hide_code=True)
def check_lexicon(box_lexicon):
    result_lexicon = L.check("lexicon", box_lexicon.value)
    result_lexicon
    return (result_lexicon,)


@app.cell(hide_code=True)
def hints_lexicon():
    L.hints(
        "lexicon",
        [
            "Twee delen. Eerst de aantallen: een dictionary `counts`, leeg bij het begin, een loop `for label, words in sentences:` en daarin `for word in words:`. Elk woord krijgt een lijst met twee aantallen, `[0, 0]` de eerste keer, en het label van de zin is de plek in de lijst waar 1 bij komt. Dan de scores: een dictionary `scores`, een loop `for word in counts:`, het aantal in positieve zinnen als `positive = counts[word][1]`, en `np.log10` van een breuk voor `scores[word]`.",
            "Plek 0 in de lijst van een woord telt de zinnen met het andere label. Welk aantal hoort onder de deelstreep, en wat telt het recept erbij op?",
            "`        scores[word] = np.log10((positive + 1) / (___ + 1))`",
        ],
        why="`counts = {}`; `for label, words in sentences:` met `for word in words:` eronder, en daarin `if word in counts:` met `counts[word][label] = counts[word][label] + 1`, en `else:` met `counts[word] = [0, 0]` en `counts[word][label] = 1`. Dan `scores = {}`; `for word in counts:` met `negative = counts[word][0]`, `positive = counts[word][1]` en `scores[word] = np.log10((positive + 1) / (negative + 1))`; na de loop `return scores`. De voorbeelden geven `{\"great\": 0.477, \"food\": 0, \"bad\": -0.301, \"service\": 0.301}` en `{\"bad\": -0.477, \"food\": 0, \"good\": 0.301}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_lexicon():
    mo.md(r"""
    De woorden aan de twee uiteinden zou een mens ook op een lijst zetten: *geweldige* en *uitstekend* bovenaan, *slecht* en *slechte* onderaan. Daartussen staan woorden die niemand met de hand op een lijst zet: *werkt* krijgt +0,49, uit zinnen als *Werkt geweldig.*, en *uur* −0,93, want *uur* staat 16 keer in negatieve zinnen en één keer in een positieve. De lijst weet wat in deze recensies lof krijgt en waarover geklaagd wordt. Op de 600 achtergehouden zinnen krijgt de lijst er 482 goed, en de classifier van *Logistische regressie op woorden* 480.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. score: de som, met ontkenning

        Schrijf `score(words, lexicon, negations)`, waarin `words` de lijst woorden van een zin is, `lexicon` een dictionary met een score voor elk woord, en `negations` de lijst ontkenningswoorden. Geef de som terug van de scores van de woorden die het lexicon heeft, waarbij elk van de twee woorden vlak na een ontkenning met het omgekeerde teken telt: trek de score af in plaats van de score op te tellen. Een ontkenningswoord dat het lexicon heeft, houdt de eigen score. Vervang de `...`. De check roept `score(["not", "good", "for", "the", "money"], {"not": -0.6, "good": 0.5, "money": -0.7}, ["not", "no", "never"])` aan, dat `-1.8` geeft: *not* telt −0,6 op, *good* komt er vlak na en trekt 0,5 af, *for* is het tweede woord na *not* maar staat niet in het lexicon, en *money* telt −0,7 op. En `score(["never", "bad", "and", "not", "slow"], {"bad": -0.7, "slow": -0.5, "good": 0.5}, ["not", "never"])`, dat `1.2` geeft. Daarna eigen gevallen: een lege lijst; één woord zonder ontkenning; een woord twee plaatsen na een ontkenning; een woord drie plaatsen erna; een ontkenning met een eigen score; twee ontkenningen achter elkaar; Nederlandse ontkenningen; en een ontkenning als laatste woord.
        """),
        L.task("Tel de scores op, en draai de twee woorden na een ontkenning om."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=14)
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
            "Een `total`, 0 bij het begin, en een teller `left`, ook 0: hoeveel woorden nog met het omgekeerde teken tellen. Een loop `for word in words:`; voor een woord dat het lexicon heeft, een `if` op `left` die de score van `total` aftrekt of erbij optelt. Daarna gaat `left` 1 omlaag, en een `if word in negations:` zet `left` opnieuw. Na de loop `return total`.",
            "Het woord vlak na een ontkenning moet `left` boven 0 zien, en het woord daarna ook, maar het derde niet. `left` gaat na elk woord 1 omlaag, en een ontkenning zet `left` daarna. Op welk getal moet een ontkenning `left` zetten?",
            "`            left = ___`",
        ],
        why="`total = 0` en `left = 0`; `for word in words:` met `if word in lexicon:` eronder, en daarin `if left > 0:` met `total = total - lexicon[word]`, en `else:` met `total = total + lexicon[word]`. Dan, nog in de loop, `left = left - 1`, en `if word in negations:` met `left = 2`; na de loop `return total`. De voorbeelden geven `-1.8` en `1.2`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    Met ontkenning gaat de lijst van Pattern van 313 naar 330 goed: de zin die begint met *Geen goed product.* komt nu bijvoorbeeld negatief uit. De lijst uit de recensies gaat van 482 naar 481: de regel herstelt 8 zinnen en breekt er 9. Een zin die de regel breekt, is *Maak niet dezelfde fout als ik.* Daar hoort *niet* bij *maak*, en de regel draait *dezelfde* en *fout* om, zodat *fout* positief telt. Een lijst die uit de recensies is geleerd, heeft de ontkenning al in zich: *niet* zelf telt −0,62. Een lijst die met de hand is gemaakt, heeft dat niet.

    En een venster van twee woorden weet niet waar een ontkenning bij hoort. In *Niet de moeite waard.* hoort *niet* bij *waard*, het derde woord erna, en de regel draait *de* en *moeite* om. De lijst uit de recensies krijgt de zin zo goed, maar om de verkeerde reden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 3. Wat zeggen de twee lijsten?

    Acht woorden uit de zinnen van de recensies. Zet elk woord waar het hoort: geven de lijst van Pattern en de lijst uit de recensies het woord een score met hetzelfde teken, heeft maar één van de twee lijsten het woord, of geven de twee lijsten het woord scores met een tegengesteld teken?
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_lists = L.sort("words-lists")
    sort_lists
    return (sort_lists,)


@app.cell(hide_code=True)
def check_sort(sort_lists):
    result_sort = L.check("words-lists", sort_lists)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    De lijsten van Pattern bevatten bijvoeglijke naamwoorden: in de Nederlandse lijst zijn 3.802 van de 3.918 betekenissen gelabeld als bijvoeglijk naamwoord, in de Engelse 2.643 van de 2.918. Een lijst uit de recensies bevat ook de zelfstandige naamwoorden en werkwoorden waarover mensen lovend zijn of klagen, maar alleen voor het soort recensies waaruit de lijst leerde. Turney (2002) gaf een voorbeeld: het Engelse *unpredictable* (onvoorspelbaar) kan in een recensie van een auto negatief zijn, bij een onvoorspelbare besturing, en in een filmrecensie positief, bij een onvoorspelbaar plot. Jurafsky en Martin merken op dat de betekenis van een woord in een lexicon meestal vastligt, los van de zin waarin het woord staat. Zo krijgt *goedkoop* −0,10 in de lijst van Pattern, het gemiddelde van voordelig (+0,4) en ordinair (−0,6), en −0,38 uit de recensies, waar *goedkoop* 11 keer in negatieve zinnen staat en 4 keer in positieve.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: een eigen zin

    Schrijf een zin in het Nederlands. De afbeelding geeft elk woord de score in beide lijsten, en de som zonder en met ontkenning. Het vak begint met *Niet slecht, en het scherm is geweldig.*: zonder ontkenning noemt de lijst uit de recensies de zin negatief, omdat *niet* en *slecht* allebei omlaag trekken. Probeer een andere ontkenning, zoals *nooit*, en een zin waarin *niet* bij een woord verderop hoort, zoals *Ik zou niet zeggen dat het geluid slecht is.*
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.text_area(value=C.TRY_START[L.lang], label="Jouw zin", rows=2, full_width=True)
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

    **Samenhang.** [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) leerde alle weights samen. Een sentiment lexicon geeft elk woord een eigen score: met de hand, zoals in de lijsten van Pattern, of uit aantallen in gelabelde recensies, zoals jouw `lexicon` doet. Een som van scores, elk uit de twee aantallen van één woord, doet het op deze recensies even goed als de classifier, en de lijst van Pattern, gemaakt zonder deze zinnen, doet het veel slechter. Een ontkenning draait de woorden erna om: een venster van twee woorden helpt een lijst die met de hand is gemaakt, en maakt nauwelijks uit voor een lijst die *niet* uit de recensies leerde. De volgende les, *Project: sentiment in het Engels en het Nederlands*, zet je eigen classifier en je eigen lexicon naast elkaar, in beide talen, met macro F1 en de recensies die elk van de twee fout heeft.

    **Een voorbeeld uit het dagelijks leven.** *Koud* is lof voor een biertje en een klacht over soep. Een lijst met goede en slechte woorden voor een café heeft er twee nodig: een voor de drankjes en een voor de keuken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_lexicon, result_score, result_sort):
    L.wrap_up(result_lexicon, result_score, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
