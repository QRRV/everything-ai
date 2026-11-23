# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-static-vs-contextual. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-static-vs-contextual", "nl")
    C = L.checks   # de zinnen en vectoren van de course, BERT-Tiny, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_apple():
    quick_apple = L.choice(
        "quick-apple",
        question="Hoeveel vectoren heeft GloVe in *Word embeddings: buren en analogieën* voor het woord *apple* (appel)?",
        options={
            "senses": "Twee: één voor de vrucht, één voor het bedrijf",
            "right": "Eén",
            "uses": "Eén voor elke zin waarin het woord staat",
        },
        correct="right",
        why={
            "right": "GloVe geeft elk woord één vector, geleerd uit alle zinnen waarin het woord staat. De dichtstbijzijnde buren van *apple* waren *blackberry*, *chips*, *iphone* en *microsoft*: geen vrucht te bekennen. Deze les kijkt naar een model dat elk gebruik van een woord een eigen vector geeft.",
            "senses": "GloVe weet niets van betekenissen: het leert één vector per woord, uit alle zinnen samen. De dichtstbijzijnde buren van *apple* waren *blackberry*, *chips*, *iphone* en *microsoft*. Zie [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl).",
            "uses": "Dat doen de modellen van deze les. GloVe leert één vector per woord, uit alle zinnen samen, en zoekt die vector op in een tabel. Zie [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl).",
        },
    )
    quick_apple
    return (quick_apple,)


@app.cell(hide_code=True)
def quick_check_matmul():
    quick_matmul = L.choice(
        "quick-matmul",
        question="`E` is een tabel met één rij per woord, `v` een vector met evenveel getallen als een rij. Wat geeft `E @ v`?",
        options={
            "one": "Eén getal: het dot product van E en v",
            "cos": "De cosine similarity van elke rij met v, één per rij",
            "right": "Het dot product van elke rij van E met v",
        },
        correct="right",
        why={
            "right": "`E @ v` geeft één dot product per rij. Gedeeld door de norm van elke rij en door de norm van `v`, zoals in je `cosines`, worden dat cosine similarities. Deze les vergelijkt zo de vectoren van een woord in verschillende zinnen.",
            "one": "`@` van een tabel en een vector geeft één getal voor elke rij van de tabel, niet één getal in totaal. Zie [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl).",
            "cos": "Cosine similarities komen pas na het delen door de normen. `E @ v` alleen geeft de dot products, en dot products groeien mee met de lengte van de rijen. Zie [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl).",
        },
    )
    quick_matmul
    return (quick_matmul,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    GloVe geeft een woord één vector, wat het woord in een zin ook betekent. De dichtstbijzijnde buren van het Engelse *bank* zijn *banks*, *securities* (effecten), *banking*, *investment* (belegging) en *exchange* (beurs): allemaal geld. *money* is de 98e buur, en *river* (rivier) pas de 2.181e. De oever van een rivier, in het Engels ook *bank*, moet het doen met dezelfde vector als de bank waar je geld staat.

    In 2018 presenteerden Matthew Peters en anderen ELMo, een model dat een woord elke keer dat het gebruikt wordt een nieuwe vector geeft, uitgerekend uit de zin eromheen. Volgens Peters en anderen krijgt elk token in dit model een representatie die afhangt van de hele zin. De language models van nu werken vanbinnen ook zo. Deze les neemt een klein model van die soort, BERT-Tiny, en meet of de *bank* van een rivier en de *bank* van je geld echt verschillende vectoren krijgen.

    **Een voorbeeld uit het dagelijks leven.** Wie *de bank was nat van de regen* hoort, denkt geen moment aan geld: de rest van de zin bepaalt wat *bank* betekent.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_across = L.guess(
        "guess",
        question="De course schreef 44 Engelse zinnen met *bank*: 22 over een rivier (de oever) en 22 over geld. Aan het eind van de berekening geeft BERT-Tiny *bank* in elke zin een vector van 128 getallen. Neem de cosine similarity van *bank* in een rivierzin met *bank* in een geldzin, voor alle 22 keer 22 paren, en neem het gemiddelde. Wat komt er ongeveer uit?",
        options={
            "low": "0,12",
            "right": "0,83",
            "neg": "−0,40",
            "half": "0,45",
        },
        reveal="**0,83**: *bank* in een zin over een rivier en *bank* in een zin over geld wijzen nog bijna dezelfde kant op. Binnen de rivierzinnen is het gemiddelde 0,87, binnen de geldzinnen 0,84: iets hoger, maar niet veel. Deze les laat zien dat zo'n klein verschil toch genoeg kan zijn om de twee betekenissen uit elkaar te houden.",
    )
    guess_across
    return (guess_across,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Static en contextual embeddings

    Een **static embedding** geeft een woord één vaste vector, dezelfde in elke zin. GloVe en word2vec geven static embeddings. Daniel Jurafsky en James H. Martin schrijven in *Speech and Language Processing* dat word2vec static embeddings maakt: één vaste embedding voor elk woord van de vocabulary. Een **contextual embedding** is een vector voor één gebruik van een woord, uitgerekend uit de zin waarin het woord staat, dus hetzelfde woord krijgt in elke zin een andere vector. Jurafsky en Martin zeggen het zo: met contextual embeddings krijgt een woord elke keer dat het in een andere context staat een andere vector.

    ### BERT-Tiny

    BERT is een model dat Jacob Devlin en anderen bij Google in 2019 beschreven. De naam staat voor Bidirectional Encoder Representations from Transformers: het model kijkt naar de woorden aan beide kanten van elk woord. BERT leerde van boeken en van de Engelse Wikipedia, samen 3,3 miljard woorden, door woorden in te vullen die verborgen waren; een latere les, *BERT: het ontbrekende woord invullen*, laat zien hoe. **BERT-Tiny** is de kleinste van 24 kleine versies die Iulia Turc en anderen in 2019 beschreven en in 2020 uitbrachten: 4,4 miljoen parameters, waarvan 3,9 miljoen de vectoren van de 30.522 stukjes van het model zijn. Het kent alleen Engels, in kleine letters.

    BERT-Tiny knipt een zin eerst in stukjes, de tokens van het model, zoals de tokenizers van [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl): *kayak* wordt *kay* en *##ak*, waarbij *##* betekent dat het stukje een woord voortzet. Vóór de zin zet het model het stukje *[CLS]* en erachter *[SEP]*. Elk stukje krijgt een vector van 128 getallen: de vector van het stukje zelf, plus een vector voor de plek in de zin, plus nog één die voor elk stukje van één zin dezelfde is. Dan komen er twee **blocks**. Een block rekent voor elk stukje een nieuwe vector uit, uit de vectoren van alle stukjes van de zin, met attention, zoals in *Attention: elk woord kijkt naar de andere*. Vóór het eerste block hangt de vector van *bank* dus alleen af van het stukje *bank* en de plek ervan; na elk block van de hele zin.

    De afbeelding laat zien wat dat met de 44 zinnen van de course doet.
    """)
    return


@app.cell(hide_code=True)
def idea_uses():
    mo.Html(C.uses_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        ### Self-similarity

        Hoe sterk lijkt een woord in de ene zin op hetzelfde woord in een andere zin? Kawin Ethayarajh mat dat in 2019 met de **self-similarity**: de gemiddelde cosine similarity van een woord in elk paar van twee verschillende zinnen. Ethayarajh omschrijft de self-similarity van een woord in een layer, hier de vectoren na één block, als de gemiddelde cosine similarity van de contextual embeddings van het woord over alle verschillende contexten waarin het woord staat. Bij een static embedding is de self-similarity 1: elk gebruik heeft dezelfde vector. Hoe lager, hoe meer de zin meetelt.

        ### De betekenis raden uit voorbeelden

        Als de vectoren van *bank* in rivierzinnen dichter bij elkaar liggen dan bij de vectoren van *bank* in geldzinnen, kunnen de vectoren de betekenis van *bank* in een nieuwe zin raden. Neem zinnen waarvan je de betekenis kent, neem per betekenis het gemiddelde van de vectoren van *bank*, en kies voor de nieuwe zin de betekenis waarvan de gemiddelde vector de hoogste cosine similarity met de vector van de nieuwe zin heeft. Jurafsky en Martin beschrijven die gok, met grote verzamelingen zinnen waarbij de betekenis erbij staat; Peters en anderen gebruikten de gok voor ELMo.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **static embedding** | één vaste vector per woord, dezelfde in elke zin, zoals bij GloVe en word2vec |
        | **contextual embedding** | een vector voor één gebruik van een woord, uitgerekend uit de hele zin waarin het woord staat |
        | **BERT-Tiny** | een klein model (2019, uitgebracht in 2020) dat elk stukje van een Engelse zin na twee blocks een contextual embedding van 128 getallen geeft |
        | **block** | een stap van zo'n model: een nieuwe vector voor elk stukje, uitgerekend uit de vectoren van alle stukjes van de zin |
        | **[CLS]**, **[SEP]** | de stukjes die BERT vóór en na een zin zet |
        | **self-similarity** | de gemiddelde cosine similarity van een woord in elk paar van twee verschillende zinnen; 1 als elk gebruik dezelfde vector heeft |
        | `pieces.index(x)` | de plek van de eerste `x` in de lijst `pieces`, geteld vanaf 0 |
        | `np.mean(A, axis=...)` | het gemiddelde langs één axis van een tabel: één gemiddelde per kolom of één per rij, afhankelijk van de axis, zoals bij `np.sum(A, axis=...)` |
        """),
        L.watch_out("Een eigen vector voor elk gebruik betekent niet dat het model betekenissen kent. Het zag nooit een lijst met betekenissen: het leerde verborgen woorden invullen, en elke vector draagt iets van de hele zin mee."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip de zin in stukjes, en zoek de plek van het woord daartussen.
    2. Laat het model rekenen, en neem de vector op die plek: vóór het eerste block, en na elk block.
    3. Vergelijk het woord in twee zinnen met de cosine similarity. Neem voor alle zinnen met het woord het gemiddelde over elk paar verschillende zinnen: de self-similarity.
    4. Om de betekenis van een nieuw gebruik te raden: neem het gemiddelde van de vectoren van de voorbeelden van elke betekenis, en neem de betekenis waarvan het gemiddelde de hoogste cosine similarity met de nieuwe vector heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De knop hieronder downloadt BERT-Tiny. De code neemt dan *bank* in twee zinnen van de course: één over een rivier, één over geld. `bert.pieces` van de course geeft de stukjes van een zin, met *[CLS]* en *[SEP]*. `bert.layers` geeft drie tabellen, met één rij van 128 getallen per stukje: vóór het eerste block, na het eerste block en na het tweede. `bank_vectors` zoekt *bank* tussen de stukjes en geeft de drie rijen van *bank* terug.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download BERT-Tiny, 18 MB")
    return (get_button,)


@app.cell(hide_code=True)
def bert_cell(get_button):
    bert = C.bert_or_stop(get_button, {
        "ask": "*BERT-Tiny staat nog niet op je computer. De knop downloadt het model één keer van Hugging Face, in een vaste versie: `google/bert_uncased_L-2_H-128_A-2`, 18 MB, onder de Apache-2.0-licentie. Het uitgewerkte voorbeeld en het onderdeel om zelf te proberen hebben het model nodig; de oefeningen niet.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte, en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*BERT-Tiny is gedownload.*")
    return (bert,)


@app.cell
def example_cell(bert):
    import numpy as np


    def bank_vectors(sentence):
        pieces = bert.pieces(sentence)
        i = pieces.index("bank")
        return [layer[i] for layer in bert.layers(sentence)]


    river = "We had a picnic on the bank of the river."
    money = "I opened an account at the bank downtown."
    print(bert.pieces(river))
    print(bert.pieces(money))
    river_vectors = bank_vectors(river)
    money_vectors = bank_vectors(money)
    for k in range(3):
        a, b = river_vectors[k], money_vectors[k]
        print(k, round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste zin (*we picknickten op de oever van de rivier*) heeft 13 stukjes, de tweede (*ik opende een rekening bij de bank in het centrum*) 11; *bank* staat in allebei op plek 7, geteld vanaf 0, met *[CLS]* op plek 0. De laatste drie regels geven, na het nummer van de stap (0 vóór het eerste block, 1 en 2 na elk block), de cosine similarity van de twee vectoren van *bank*: 1,0 bij stap 0, 0,968 bij stap 1 en 0,9 bij stap 2. Vóór het eerste block zijn de twee precies gelijk. Na elk block verschillen de twee meer: de zinnen zijn in de vectoren terechtgekomen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_same():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom dezelfde vector?
    """)
    return


@app.cell(hide_code=True)
def choice_same_cell():
    choice_same = L.choice(
        "same-start",
        question="Vóór het eerste block heeft *bank* precies dezelfde vector in *We had a picnic on the bank of the river.* en in *I opened an account at the bank downtown.*: een cosine similarity van 1. Waarom?",
        options={
            "length": "De twee zinnen hebben evenveel stukjes",
            "right": "Het is hetzelfde stukje op dezelfde plek",
            "place": "Vóór de blocks heeft elk stukje één vector, op welke plek het ook staat",
        },
        correct="right",
        why={
            "right": "Vóór het eerste block is de vector van een stukje gemaakt uit het stukje en de plek ervan, en verder niets. *bank* staat in beide zinnen op plek 7, dus de twee vectoren zijn gelijk. De course schreef de 44 zinnen in zulke paren, één over een rivier en één over geld met *bank* op dezelfde plek, zodat alleen de blocks de twee betekenissen uit elkaar kunnen trekken.",
            "length": "De eerste zin heeft 13 stukjes, de tweede 11. De lengte telt vóór het eerste block niet mee: alleen het stukje en de plek ervan.",
            "place": "De plek telt ook mee: hetzelfde stukje op een andere plek krijgt een andere vector. Hier staat *bank* in beide zinnen op plek 7.",
        },
    )
    choice_same
    return (choice_same,)


@app.cell(hide_code=True)
def text_exercise_self():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. self_similarity: hoe sterk lijkt een woord op zichzelf?

        Schrijf `self_similarity(rows)`: geef de gemiddelde cosine similarity van elke twee verschillende rijen van de tabel `rows` terug. Neem elk paar van twee verschillende rijen één keer, reken de cosine similarity van de twee uit, en deel de som door het aantal paren.

        Voor `self_similarity(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))` is de uitkomst 0,4714, afgerond: de drie paren hebben cosine similarities van 0, 0,707 en 0,707. Vervang de `...`. De check roept dat aan en `self_similarity(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]))`, wat 1 moet geven, en daarna eigen gevallen: twee rijen, vier rijen van drie getallen, drie rijen waarvan twee tegengesteld wijzen, en de vectoren van *bank* in zinnen van de course, na het tweede block en vóór het eerste. `C.bank_rows(2, "river", "example")` geeft de tabel van de 14 rivierzinnen om van te leren, na het tweede block: één rij per zin.
        """),
        L.task("Geef de gemiddelde cosine similarity van elke twee verschillende rijen van rows terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_self():
    box_self = L.code_box("self_similarity", lines=10)
    box_self
    return (box_self,)


@app.cell(hide_code=True)
def check_self(box_self):
    result_self = L.check("self_similarity", box_self.value)
    result_self
    return (result_self,)


@app.cell(hide_code=True)
def hints_self():
    L.hints(
        "self_similarity",
        [
            "Begin met `total = 0` en `count = 0`. Dan twee loops: `for i in range(len(rows)):`, en daarin een loop over `j`, zodat elk paar van twee verschillende rijen één keer langskomt. Tel binnen beide loops de cosine similarity van `rows[i]` en `rows[j]` op bij `total`, en 1 bij `count`. Na de loops: `return total / count`.",
            "Rij 0 met rij 1 is hetzelfde paar als rij 1 met rij 0, en rij 1 met rij 1 is helemaal geen paar. Als `i` 1 is, wat is dan de eerste `j` die een nieuw paar geeft?",
            r"`        for j in range(___, len(rows)):`",
        ],
        why=r"`for j in range(i + 1, len(rows)):`: `j` begint net na `i`, dus elk paar komt één keer langs en geen rij komt zichzelf tegen. Helemaal: `total = 0` en `count = 0`; `for i in range(len(rows)):` met die loop erin; binnen beide `a, b = rows[i], rows[j]`, `total += a @ b / (np.linalg.norm(a) * np.linalg.norm(b))` en `count += 1`; dan `return total / count`. In het voorbeeld geven de drie paren 0, 0,707 en 0,707, en het gemiddelde is 0,4714.",
    )
    return


@app.cell(hide_code=True)
def text_after_self():
    mo.md(r"""
    In de afbeelding mat je functie *bank* in de 44 zinnen. In GloVe is de self-similarity 1: elk gebruik heeft dezelfde vector. In BERT-Tiny is de self-similarity 0,90 vóór het eerste block, 0,92 na het eerste en 0,84 na het tweede. Vóór het eerste block ligt het getal alleen onder 1 omdat *bank* in verschillende paren op verschillende plekken staat.

    Dat lijkt hoog, maar kijk naar de grijze balken: twee verschillende woorden in twee verschillende zinnen, zoals *picnic* en *account*, hebben vóór het eerste block al een gemiddelde cosine similarity van 0,24, na het eerste 0,53 en na het tweede 0,57. Na de blocks wijzen alle vectoren ongeveer dezelfde kant op. Ethayarajh vond dat ook in grotere modellen: de vectoren zitten samen in een smalle kegel van de ruimte, in plaats van alle kanten op te wijzen. Een self-similarity zegt op zichzelf dus weinig, en Ethayarajh vergelijkt het getal met twee willekeurige woorden uit verschillende zinnen. Hier ligt *bank* vóór het eerste block 0,66 boven de grijze balk, na het eerste 0,39 en na het tweede 0,27: na elk block telt de zin meer mee. Ethayarajh schrijft over BERT, ELMo en GPT-2 dat hetzelfde woord in verschillende contexten nog steeds een hogere cosine similarity heeft dan twee verschillende woorden, maar dat die self-similarity in de hogere layers veel lager is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_guess():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. guess_sense: rivier of geld?

        Schrijf `guess_sense(v, river, money)`. `v` is de vector van *bank* in een nieuwe zin; `river` en `money` zijn tabellen met de vectoren van *bank* in voorbeeldzinnen van elke betekenis, één rij per zin. Maak de gemiddelde rij van elke tabel, met één getal per kolom: het gemiddelde van die kolom. Geef `"river"` terug als `v` een hogere cosine similarity heeft met de gemiddelde rij van `river` dan met de gemiddelde rij van `money`, en anders `"money"`.

        Voor `guess_sense(np.array([1.0, 0.0]), np.array([[1.0, 0.2], [1.0, -0.2], [1.0, 0.0]]), np.array([[0.0, 1.0], [0.2, 1.0], [-0.1, 1.0]]))` is de uitkomst `"river"`. Vervang de `...`. De check roept dat aan en hetzelfde met `np.array([0.1, 1.0])` als `v`, wat `"money"` moet geven, en daarna eigen gevallen: kleine tabellen waarbij het dot product, de afstand, het ene dichtstbijzijnde voorbeeld of het gemiddelde van de cosine similarities de andere betekenis zou geven, en vier nieuwe zinnen van de course na het tweede block, uit `C.bank_rows(2, None, "new")`, met de 14 voorbeeldzinnen van elke betekenis.
        """),
        L.task("Geef de betekenis terug waarvan de gemiddelde rij de hoogste cosine similarity met v heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_guess():
    box_guess = L.code_box("guess_sense", lines=9)
    box_guess
    return (box_guess,)


@app.cell(hide_code=True)
def check_guess(box_guess):
    result_guess = L.check("guess_sense", box_guess.value)
    result_guess
    return (result_guess,)


@app.cell(hide_code=True)
def hints_guess():
    L.hints(
        "guess_sense",
        [
            "Vier stappen: `r = np.mean(river, axis=...)`, met de goede axis, en `m` net zo voor `money`; dan de cosine similarity van `v` met `r` en van `v` met `m`, zoals in je `self_similarity`; als laatste een `if` die `\"river\"` teruggeeft als de eerste hoger is, en anders `\"money\"`.",
            "`river` heeft één rij per zin en één kolom per getal, 128 bij BERT-Tiny. De gemiddelde rij heeft één getal per kolom nodig, zoals `v`. In *Sommen en gemiddelden langs een axis* bepaalde de axis welke kant de som of het gemiddelde op loopt. Langs welke axis loopt `np.mean` dan: omlaag langs de rijen, of langs elke rij?",
            r"`    r = np.mean(river, axis=___)`",
        ],
        why=r"`r = np.mean(river, axis=0)`: `axis=0` loopt omlaag langs de rijen, dus elke kolom krijgt een eigen gemiddelde, en `r` heeft evenveel getallen als `v`. Helemaal: `r` en `m` zo; `to_river = v @ r / (np.linalg.norm(v) * np.linalg.norm(r))` en `to_money` net zo met `m`; dan `if to_river > to_money:` met `return \"river\"` eronder, en `return \"money\"`. In het voorbeeld is `r` gelijk aan `[1, 0]` en `m` ongeveer `[0.033, 1]`, dus `v` ligt het dichtst bij de rivier.",
    )
    return


@app.cell(hide_code=True)
def text_after_guess():
    mo.md(r"""
    In de afbeelding raadde je functie *bank* in de 16 nieuwe zinnen, met de 14 voorbeeldparen. Bij GloVe heeft *bank* één vector, dus de twee gemiddelde rijen zijn gelijk, en je functie geeft 16 keer hetzelfde antwoord: 8 goed, zoveel als een munt gemiddeld goed heeft. Vóór het eerste block zijn het er ook 8: elke nieuwe zin heeft een paar met *bank* op dezelfde plek, dus beide zinnen krijgen dezelfde vector, en de voorbeelden van de twee betekenissen hebben hetzelfde gemiddelde. Na het eerste block en na het tweede zijn er 15 van de 16 goed. De ene misser is *We walked along the bank until sunset.* (we liepen langs de oever tot zonsondergang): er zit geen water in, en de zin gaat naar geld.

    Neem het gemiddelde van de vectoren van een woord in alle zinnen, welke betekenis ook, en je hebt weer één vector per woord: een static embedding, gemaakt uit een contextual model. Rishi Bommasani, Kelly Davis en Claire Cardie deden dat in 2020 met BERT en de Engelse Wikipedia: het gemiddelde over veel zinnen werkte het best, en hoe meer zinnen, hoe beter. Met genoeg zinnen deden de vectoren die zo ontstonden het beter dan GloVe en word2vec op tests van hoe sterk twee woorden op elkaar lijken. Ook de static vector van GloVe draagt de betekenissen van een woord: Sanjeev Arora en anderen lieten in 2018 zien dat zo'n vector, binnen een model van hoe tekst ontstaat dat Arora en anderen zelf opstelden, dicht bij een gewogen som ligt van een vector per betekenis, elk naar hoe vaak die betekenis voorkomt.

    BERT-Tiny leerde alleen Engels. Een Nederlandse zin knipt het model in vreemde stukjes: *De bank staat bij het water.* wordt *de*, *bank*, *st*, *##aa*, *##t*, *bi*, *##j*, *het*, *water* en de punt. Het Nederlandse *bank* heeft eigen betekenissen: een bank om op te zitten, en een bank voor geld. Voor het Nederlands zijn er modellen van dezelfde soort: BERTje (2019), geleerd uit 2,4 miljard Nederlandse tokens, en RobBERT (2020), met 117 miljoen parameters tegen de 4,4 miljoen van BERT-Tiny. Een latere les, *Language models voor het Nederlands*, komt daarop terug.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: één woord in twee zinnen

    Typ twee Engelse zinnen en een woord dat in allebei staat. De afbeelding toont de stukjes, en de cosine similarity van de vectoren van het woord in de twee zinnen, vóór het eerste block en na elk block. Peters en anderen lieten het woord *play* zien: in GloVe gaan de dichtstbijzijnde buren vooral over sport, terwijl ELMo een zin over sport vond bij *play* in een wedstrijd (een goede zet in het spel) en een zin over toneel bij *play* op Broadway (een toneelstuk). De twee zinnen hieronder beginnen daarmee. Probeer ook *bank*, *bat* (knuppel of vleermuis) of *light* (licht), en zet het woord in beide zinnen op dezelfde plek, om te zien wat de blocks alleen doen.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_first = mo.ui.text(value="The team made a brilliant play in the last minute.", label="Zin 1", full_width=True)
    explore_second = mo.ui.text(value="We saw a play by Shakespeare at the theater.", label="Zin 2", full_width=True)
    explore_word = mo.ui.text(value="play", label="Woord")
    mo.vstack([explore_first, explore_second, explore_word])
    return explore_first, explore_second, explore_word


@app.cell(hide_code=True)
def explore_look(bert, explore_first, explore_second, explore_word):
    bert   # BERT-Tiny is gedownload: deze cel wacht op de knop hierboven
    mo.Html(C.explore_picture(explore_first.value, explore_second.value, explore_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Word embeddings: buren en analogieën](/course/open/nlp-word-embeddings?lang=nl) gaf GloVe elk woord één vector: een static embedding. BERT-Tiny geeft elk gebruik een eigen vector, een contextual embedding, in twee blocks uitgerekend uit de hele zin. Vóór het eerste block hangt *bank* alleen af van het stukje en de plek; na de blocks liggen de vectoren van *bank* over een rivier en over geld ver genoeg uit elkaar dat het gemiddelde van 14 voorbeelden per betekenis in 15 van de 16 nieuwe zinnen de betekenis raadt. De voorsprong van *bank* op twee verschillende woorden wordt na elk block kleiner. De volgende les van het domein, *Betekenissen en relaties tussen woorden*, vraagt wat de betekenissen van een woord zijn, en hoe woorden met elkaar samenhangen.

    **Een voorbeeld uit het dagelijks leven.** Een vertaalapp die het Engelse *bank* naar het Nederlands vertaalt, moet kiezen tussen *oever* en *bank*: alleen de rest van de zin kan zeggen welke van de twee.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_same, result_guess, result_self):
    L.wrap_up(choice_same, result_self, result_guess)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
