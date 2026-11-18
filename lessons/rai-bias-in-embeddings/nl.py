# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-bias-in-embeddings. Route: course/route/lessen/verantwoord.yaml. Checks en de run van Granite van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-in-embeddings", "nl")
    C = L.checks   # de woordlijsten van drie tests van WEAT, en de embedding van Granite van elk woord


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl) en [Bias in language models](/course/open/rai-bias-in-llms?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Twee vectoren wijzen precies dezelfde kant op, en de ene is twee keer zo lang als de andere. Wat is hun cosine similarity?",
        options={
            "two": "2, want de ene is twee keer zo lang",
            "right": "1: de richting is dezelfde",
            "half": "0,5, de verhouding van de twee lengtes",
        },
        correct="right",
        why={
            "two": "Cosine similarity deelt het inproduct door beide lengtes, dus de lengtes vallen weg en alleen de richting telt. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "right": "Cosine similarity deelt het inproduct door beide lengtes, dus alleen de richting telt. Zo meet deze les hoe dicht twee woorden bij elkaar liggen.",
            "half": "Cosine similarity deelt het inproduct door beide lengtes, dus de lengtes vallen weg en alleen de richting telt. Zie [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_close():
    quick_close = L.choice(
        "quick-close",
        question="In een embedding krijgen twee woorden vectoren die bijna dezelfde kant op wijzen. Wat zegt dat?",
        options={
            "same": "Dat de twee woorden precies hetzelfde betekenen",
            "right": "Dat het model ze als gelijksoortig behandelt",
            "letters": "Dat ze bijna hetzelfde gespeld worden",
        },
        correct="right",
        why={
            "same": "Dichtbij is niet hetzelfde: een embedding legt dingen die op elkaar lijken dicht bij elkaar, en op elkaar lijken is niet gelijk zijn. Zie *Het idee* in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
            "right": "Een embedding legt dingen die het model als gelijksoortig behandelt dicht bij elkaar. Voor woorden leert een model dat uit hoe woorden in teksten gebruikt worden: deze les vraagt wat het zo nog meer leert.",
            "letters": "Een embedding gaat over betekenis, niet over spelling: hij legt dingen die het model als gelijksoortig behandelt dicht bij elkaar. Zie *Het idee* in [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl).",
        },
    )
    quick_close
    return (quick_close,)


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="Qwen3-0.6B had een bias score van 0,008 op de vragen in de vorm van BBQ van de course. Wat laat dat zien?",
        options={
            "none": "Dat Qwen3 geen genderbias en geen leeftijdsbias heeft",
            "right": "Hoe Qwen3 het deed op die vragen, meer niet",
            "small": "Dat kleine modellen minder bias hebben dan grote",
        },
        correct="right",
        why={
            "none": "Een lage score vindt geen probleem; hij bewijst niet dat er geen is. Met de feiten erbij had Qwen3 het vaker goed als ze bij het stereotype pasten. Zie [Bias in language models](/course/open/rai-bias-in-llms?lang=nl).",
            "right": "Een score geldt voor de vragen, de taal en de prompt die getest zijn. De test van deze les, WEAT, heeft dezelfde grens, en nog een paar.",
            "small": "De course testte één klein model, dus hij kan geen groottes vergelijken. En een lage score vindt geen probleem; hij bewijst niet dat er geen is. Zie [Bias in language models](/course/open/rai-bias-in-llms?lang=nl).",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voordat een model met woorden kan werken, maakt het er vectoren van: embeddings. Zoeken op betekenis, aanbevelingen en language models beginnen allemaal daar. Een embedding model leert welke woorden bij elkaar horen uit de teksten die het leest, en in die teksten zitten de stereotypen van de mensen die ze schreven. Aylin Caliskan en collega's (2017) lieten zien dat een word embedding zulke verbanden leerde: hoeveel dichter de embeddings van 50 beroepen bij vrouwelijke woorden lagen dan bij mannelijke, ging samen met het aandeel vrouwen in die beroepen in de VS: een correlatie van 0,90, op een schaal waar 1 betekent dat de twee perfect samen stijgen en 0 dat ze niets met elkaar te maken hebben. In deze les meet je hoe dicht woorden bij elkaar liggen in Granite, het embedding model van de course, met WEAT: een test die voor embeddings doet wat een bias-benchmark doet voor language models. Je haalt ook de genderrichting uit woorden, zoals een bekende methode doet, en meet wat er overblijft.

    **Een voorbeeld uit het dagelijks leven.** Bij een woordspel zegt iemand "salaris", en je hoort "kantoor" of "baas" in je hoofd voordat je nadenkt. Wat het eerst opkomt, is wat je het vaakst samen hoorde. Een embedding model leerde op dezelfde manier, uit welke woorden samen in teksten staan: ook dat sommige namen vaker naast "carrière" staan en andere naast "gezin".
    """)
    return


@app.cell(hide_code=True)
def predict_glove():
    guess_glove = L.guess(
        "glove",
        question="Caliskan en collega's vonden een effectgrootte van 1,81 in GloVe, de word embedding die ze testten, voor namen van mannen en vrouwen met woorden over carrière en over gezin. Op een schaal van −2 tot 2 betekent 0 geen verschil. Granite kwam uit in december 2024; het model waarop het gebouwd is, leerde van tekst die gefilterd was om haat, misbruik en grof taalgebruik eruit te halen. Wat verwacht je voor Granite?",
        options={
            "zero": "Rond 0, want het filter haalde het eruit",
            "right": "Nog steeds groot, boven 1",
            "flip": "Onder 0, andersom",
        },
        reveal="**Nog steeds groot**: 1,49, met elk woord los ingebed. Het filter haalde haat, misbruik en grof taalgebruik eruit; dat mannennamen dichter bij woorden over carrière liggen dan vrouwennamen, is geen van die drie. De rest van de les laat zien hoe dat getal ontstaat, en wat het wel en niet betekent.",
    )
    guess_glove
    return (guess_glove,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Word embeddings.** Een **word embedding** is een embedding van één woord, geleerd uit de woorden waar het vaak bij staat, zodat woorden die op dezelfde manier gebruikt worden vectoren krijgen die dezelfde kant op wijzen. GloVe en word2vec geven één vaste vector per woord. Granite is gemaakt voor hele teksten: het bedt een woord in als een tekst van één woord, als een vector van 384 getallen, met lengte 1, zodat het inproduct van twee embeddings hun cosine similarity is.

    **WEAT.** Caliskan en collega's bouwden de **WEAT**, de *Word Embedding Association Test*, naar het voorbeeld van een test voor mensen, de Implicit Association Test. WEAT neemt twee even lange lijsten **doelwoorden**, X en Y, zoals acht mannennamen en acht vrouwennamen, en twee lijsten **attribuutwoorden**, A en B, zoals acht woorden over carrière en acht over gezin. Eerst de **associatie** van één woord w: zijn gemiddelde cosine similarity met de woorden van A, min die met de woorden van B.

    $$s(w, A, B) = \text{gem}_{a \in A} \cos(w, a) - \text{gem}_{b \in B} \cos(w, b)$$

    Een positieve associatie betekent: dichter bij A dan bij B. Dan de effectgrootte: de gemiddelde associatie van X min die van Y, gedeeld door de standaardafwijking van de associaties van alle doelwoorden, X en Y samen.

    $$d = \frac{\text{gem}_{x \in X}\, s(x, A, B) - \text{gem}_{y \in Y}\, s(y, A, B)}{\text{standaardafwijking van } s(w, A, B) \text{ over alle doelwoorden}}$$

    Bij 0 liggen de twee lijsten even dicht bij A en B. Een positieve d betekent dat X dichter bij A ligt, en Y dichter bij B. Met X en Y even lang kan hij niet boven 2 komen: de standaardafwijking is dan minstens de helft van het verschil tussen de twee gemiddelden. Caliskan en collega's schrijven erbij dat hun getallen niet hetzelfde betekenen als bij de IAT, omdat de "proefpersonen" in hun experimenten woorden zijn, geen mensen. In GloVe vonden ze 1,81 voor namen van mannen en vrouwen met carrière en gezin, 1,06 voor woorden over wiskunde en kunst met mannelijke en vrouwelijke woorden, en 1,24 voor woorden over wetenschap en kunst met mannelijke en vrouwelijke woorden. Ze geven ook een p-waarde: hoe vaak de doelwoorden, willekeurig in twee lijsten verdeeld, een even groot verschil geven.

    **De genderrichting.** Tolga Bolukbasi en collega's (2016) vonden in word2vec dat "man staat tot computerprogrammeur zoals vrouw staat tot x" als x *homemaker* geeft: iemand die het huishouden doet. Hun bevinding: genderbias zit in een richting in de embedding. De verschillen *she* − *he*, *her* − *his*, *woman* − *man* en zeven paren meer wijzen ongeveer dezelfde kant op; deze les neemt hun gemiddelde, geschaald naar lengte 1, als de **genderrichting** g. Bolukbasi en collega's rekenen hem anders uit, met principal components, die een latere les behandelt; op Granite wijst hun richting bijna dezelfde kant op: de cosine similarity van de twee is 0,998. Hun methode, hard debiasing, heeft dan een stap **Neutralize**: voor elk woord dat genderneutraal hoort te zijn, zoals een beroep, haalt die het deel langs g eruit en schaalt de rest terug naar lengte 1. Met g van lengte 1 is dat deel (w · g) g. Een tweede stap, Equalize, verschuift paren zoals *grandmother* en *grandfather*, zodat elk neutraal woord even ver van allebei ligt. Van de bovenste 150 analogieën die de embedding gaf, vonden crowdworkers er ervoor 19% stereotiep, en erna maar 6%.

    **Verborgen, niet weg.** Hila Gonen en Yoav Goldberg (2019) keken opnieuw, en concludeerden dat het effect vooral is dat de bias verborgen wordt, niet weggehaald. Na hard debiasing vielen de woorden die het meest mannelijk of vrouwelijk waren geweest nog steeds in twee groepen die voor 92,5% bij hun oude gender pasten, tegen 99,9% ervoor. Hun uitleg: de genderrichting is een manier om de genderassociatie van een woord te meten, maar ze bepaalt die associatie niet.

    **Wat WEAT niet kan laten zien.** Chandler May en collega's (2019) bouwden SEAT, WEAT voor modellen van hele zinnen zoals BERT: elk woord gaat in zinnen zoals `This is <woord>.`. Hun waarschuwing: SEAT kan laten zien dat er bias is, maar een negatieve uitkomst betekent niet dat het model vrij van bias is. Voor BERT, met elk woord los, gaf de test met carrière en gezin 0,21, waar een eenvoudiger embedding, CBoW, 1,81 gaf. En Kawin Ethayarajh en collega's (2019) lieten zien dat WEAT bias stelselmatig overschat, dat de effectgrootte met één woord in elke lijst doelwoorden altijd 2 is de ene kant op en −2 de andere, en dat je de attribuutwoorden zo kunt kiezen dat bijna elk woord mannelijker uitkomt dan een ander. De model card van Granite zegt dat het model waarop het gebouwd is, leerde van tekst die gefilterd was om haat, misbruik en grof taalgebruik eruit te halen, en meldt geen test op sociale bias.

    **De run van de course.** De course bedde elk woord van deze les in met Granite, elk als een tekst van één woord, en elk woord van de drie tests ook in de zin `This is <woord>.`. De woordlijsten zijn Engels, zoals in de papers. De oefeningen lezen die embeddings; met de download hieronder bed je je eigen woorden in.

    **Probeer het.** Hieronder bedt Granite vier lijsten woorden van jezelf in, en de demo geeft de associatie van elk doelwoord en de effectgrootte van WEAT. Hij begint met één woord in elke lijst, zoals Ethayarajh en collega's deden: *door* (deur) en *curtain* (gordijn), met *masculine* en *feminine*. Probeer daarna *man* en *woman* als A en B. Of schrijf een eigen test, met meer woorden in elke lijst.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Granite Embedding staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, in een vaste versie: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, onder de Apache-2.0-licentie. De rest van de les werkt zonder: die leest de embeddings die de course heeft opgeslagen.*",
        "busy": "**Downloaden** {done} van {total} MB …",
        "no_net": "*De download lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding en klik opnieuw.*",
        "not_pinned": "*De download lukte niet: een bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*De download lukte niet: er is niet genoeg ruimte op je schijf. Maak ruimte en klik opnieuw.*",
        "failed": "*De download lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Granite Embedding is gedownload.*")
    return


@app.cell(hide_code=True)
def words_box_cell():
    x_box = mo.ui.text(value="door", label="Doelwoorden X", full_width=True)
    y_box = mo.ui.text(value="curtain", label="Doelwoorden Y", full_width=True)
    a_box = mo.ui.text(value="masculine", label="Attribuutwoorden A", full_width=True)
    b_box = mo.ui.text(value="feminine", label="Attribuutwoorden B", full_width=True)
    measure_button = mo.ui.run_button(label="Meet met Granite")
    mo.vstack([mo.md("*Woorden gescheiden door komma's, hooguit 12 in elke lijst.*"), x_box, y_box, a_box, b_box, measure_button])
    return a_box, b_box, measure_button, x_box, y_box


@app.cell(hide_code=True)
def demo_measure(a_box, b_box, measure_button, x_box, y_box):
    _boxes = {"X": x_box.value, "Y": y_box.value, "A": a_box.value, "B": b_box.value}
    if not C.downloaded():
        _out = mo.md("*Download eerst Granite Embedding, met de knop hierboven.*")
    elif not measure_button.value:
        _out = mo.md("*Druk op de knop om te meten.*")
    elif not C.filled(_boxes):
        _out = mo.md("*Schrijf minstens één woord in elk van de vier vakken.*")
    else:
        mo.output.replace(mo.md("*Granite laadt en bedt je woorden in …*"))
        _words = {"title": "Jouw test op Granite", "xlabel": "associatie s(woord, A, B)",
                  "rows": {"X": "doelwoorden X", "Y": "doelwoorden Y"},
                  "every": "Elk woord met zijn associatie: {words}.",
                  "effect": "De effectgrootte van WEAT: {d}.",
                  "no_spread": "Alle doelwoorden hebben dezelfde associatie, dus de effectgrootte is niet uit te rekenen: er is niets om door te delen."}
        _out = mo.Html(C.demo(_boxes, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **word embedding** | een embedding van één woord, geleerd uit de woorden waar het vaak bij staat |
        | **WEAT** | de *Word Embedding Association Test*: of twee lijsten doelwoorden verschillen in hoe dicht ze bij twee lijsten attribuutwoorden liggen, als een effectgrootte van −2 tot 2 |
        | **doelwoorden, attribuutwoorden** | de woorden die WEAT vergelijkt (X en Y, zoals namen), en de woorden waarmee het ze vergelijkt (A en B, zoals carrière en gezin) |
        | **associatie** s(w, A, B) | de gemiddelde cosine similarity van w met A, min die met B |
        | **genderrichting** g | de richting van *he* naar *she* in een embedding, met lengte 1 |
        | **Neutralize** | de stap van hard debiasing die het deel langs g uit een genderneutraal woord haalt: w − (w · g) g, geschaald naar lengte 1 |
        """),
        L.watch_out("Een getal van WEAT geldt voor de woordlijsten die je koos, ingebed zoals jij ze inbedde. Een groot getal laat een associatie in de embedding zien; een klein getal laat niet zien dat er geen is. En met weinig woorden kunnen andere lijsten voor hetzelfde idee een ander getal geven: kies de lijsten voordat je kijkt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een embedding testen met WEAT:

    1. Kies de doelwoorden X en Y en de attribuutwoorden A en B, met een reden voor elk woord, voordat je naar een uitkomst kijkt.
    2. Bed elk woord in met het model dat je test, zoals het gebruikt wordt: als los woord, of in een zin.
    3. Reken de associatie uit van elk doelwoord met A en B. Dat is `association`.
    4. Reken de effectgrootte uit. Dat is `effect_size`. Probeer ook andere lijsten voor hetzelfde idee.
    5. Test je een stap tegen bias, zoals Neutralize, meet dan daarna met meer dan de maat waarop die stap gebouwd is: andere lijsten, namen, en welke woorden nog dicht bij elkaar liggen.
    6. Schrijf de lijsten op, het model, hoe je de woorden inbedde, en dat een klein getal niet betekent dat er geen bias is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier doelwoorden en twee attribuutwoorden, in twee dimensies, elk met lengte 1: A heeft één woord (1; 0), B heeft één woord (0; 1). X heeft x₁ = (0,8; 0,6) en x₂ = (1; 0); Y heeft y₁ = (0,6; 0,8) en y₂ = (0; 1). Bij vectoren met lengte 1 is de cosine similarity het inproduct.

    | Woord | cos(w, a) | cos(w, b) | s(w, A, B) |
    |---|---|---|---|
    | x₁ = (0,8; 0,6) | 0,8 | 0,6 | 0,2 |
    | x₂ = (1; 0) | 1 | 0 | 1 |
    | y₁ = (0,6; 0,8) | 0,6 | 0,8 | −0,2 |
    | y₂ = (0; 1) | 0 | 1 | −1 |

    De cel rekent de effectgrootte uit de vier associaties uit.
    """)
    return


@app.cell
def example_effect():
    _s = [0.2, 1.0, -0.2, -1.0]   # de associaties van x1, x2, y1 en y2

    print("gemiddelde van X:", np.mean(_s[:2]))
    print("gemiddelde van Y:", np.mean(_s[2:]))
    print("standaardafwijking:", round(np.std(_s), 4))
    print("effectgrootte:", round((np.mean(_s[:2]) - np.mean(_s[2:])) / np.std(_s), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De gemiddelden zijn 0,6 en −0,6, een verschil van 1,2; de standaardafwijking van alle vier is 0,7211, dus de effectgrootte is 1,2 / 0,7211 = 1,66. Nu Neutralize, in drie dimensies: w = (0,6; 0; 0,8) en g = (0; 0; 1). Het deel langs g is w · g = 0,8 keer g, dus w − 0,8 g = (0,6; 0; 0). Dat heeft lengte 0,6; terug geschaald naar lengte 1 is het (1; 0; 0). Langs g ligt het woord nu op 0; in de andere richtingen wijst het nog waar het heen wees.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_association():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. association: dichter bij A of bij B

        `association(w, A, B)` krijgt één embedding `w` en twee lijsten embeddings, `A` en `B`. In het vak staan `cosine(u, v)` uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), en de loop die de cosine similarity van `w` met elk woord van `A` in de lijst `to_a` zet. Vervang de `...` door hetzelfde voor `B`, in een lijst `to_b`, en geef het gemiddelde van `to_a` min het gemiddelde van `to_b` terug. De check roept `association(vec["John"], CAREER, FAMILY)` aan, met `vec["John"]` de embedding van Granite van John en `CAREER` en `FAMILY` de embeddings van de acht woorden over carrière en over gezin; daarna vijf andere gevallen. Het plaatje onder de check zet dan alle zestien namen van de test neer met jouw functie.
        """),
        L.task("Geef de gemiddelde cosine similarity van `w` met `A` terug, min die met `B`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_association():
    box_association = L.code_box("association", lines=16)
    box_association
    return (box_association,)


@app.cell(hide_code=True)
def check_association(box_association):
    result_association = L.check("association", box_association.value)
    result_association
    return (result_association,)


@app.cell(hide_code=True)
def hints_association():
    L.hints(
        "association",
        [
            "De loop over `A` staat er al. Schrijf dezelfde loop over `B`, en trek dan de twee gemiddelden van elkaar af.",
            "`to_b = []`, dan `for b in B:` met daarin `to_b.append(cosine(w, b))`. Na de loop geeft `np.mean` het gemiddelde van een lijst.",
            "`return np.mean(to_a) - ___`",
        ],
        why="`to_b = []`, `for b in B:` met `to_b.append(cosine(w, b))`, en `return np.mean(to_a) - np.mean(to_b)`. Voor John geeft dat 0,0084: een beetje dichter bij de woorden over carrière dan bij die over gezin.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_effect():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. effect_size: het getal van WEAT

        `effect_size(X, Y, A, B)` krijgt vier lijsten embeddings. De functie zet de associatie van elk woord van `X` al in `s_x`, en van elk woord van `Y` in `s_y`, met de eigen `association` van de course, dus deze oefening werkt ook als oefening 1 nog niet af is. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld: het gemiddelde van `s_x` min het gemiddelde van `s_y`, gedeeld door de standaardafwijking van alle associaties. `s_x + s_y` zet de twee lijsten achter elkaar, in één lijst. De check roept `effect_size(MEN, WOMEN, CAREER, FAMILY)` aan, dan de lijsten van de twee andere tests, dezelfde test met de namen omgewisseld, en het uitgewerkte voorbeeld.
        """),
        L.task("Geef de effectgrootte van WEAT terug van `X` en `Y` met `A` en `B`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_effect():
    box_effect = L.code_box("effect", lines=14)
    box_effect
    return (box_effect,)


@app.cell(hide_code=True)
def check_effect(box_effect):
    result_effect = L.check("effect", box_effect.value)
    result_effect
    return (result_effect,)


@app.cell(hide_code=True)
def hints_effect():
    L.hints(
        "effect",
        [
            "Drie delen, zoals in het uitgewerkte voorbeeld: het gemiddelde van `s_x`, het gemiddelde van `s_y`, en de standaardafwijking van beide lijsten samen.",
            "`np.mean` geeft het gemiddelde van een lijst en `np.std` de standaardafwijking; `s_x + s_y` is één lijst met beide. Zet het verschil van de twee gemiddelden tussen haakjes voordat je deelt, anders deelt Python alleen het tweede gemiddelde.",
            "`return (np.mean(s_x) - np.mean(s_y)) / ___`",
        ],
        why="`return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x + s_y)`. Voor de namen met carrière en gezin geeft dat 1,49 op Granite.",
    )
    return


@app.cell(hide_code=True)
def text_granite():
    mo.md(r"""
    ### Wat de embeddings van Granite laten zien

    Met elk woord los ingebed geeft Granite 1,49 voor namen van mannen en vrouwen met carrière en gezin, 1,36 voor woorden over wiskunde en kunst met mannelijke en vrouwelijke woorden, en 0,96 voor woorden over wetenschap en kunst met mannelijke en vrouwelijke woorden. In GloVe vonden Caliskan en collega's 1,81, 1,06 en 1,24. In de zin `This is <woord>.` geeft Granite 1,28, 1,30 en 0,87: lager, maar nog steeds dezelfde kant op. Elk getal rust op acht woorden per lijst, en op deze acht; andere namen of andere woorden over carrière kunnen andere getallen geven.

    Het voorbeeld van Ethayarajh en collega's, gemeten op Granite: *door* tegen *curtain*, met één attribuutwoord per kant. Met *masculine* en *feminine* is de effectgrootte 2,00: *door* is het mannelijkere woord. Met *man* en *woman* is hij −2,00: nu is *curtain* het. De associaties achter die getallen zijn −0,0117 voor *door* en −0,0128 voor *curtain* met *masculine* en *feminine*, en −0,0043 voor *door* en −0,0016 voor *curtain* met *man* en *woman*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_door():
    mo.md(r"""
    ### Oefening 3. Een deur en een gordijn
    """)
    return


@app.cell(hide_code=True)
def choice_door_cell():
    choice_door = L.choice(
        "door",
        question="Op Granite geeft *door* tegen *curtain*, met *masculine* en *feminine* als attribuutwoorden, een effectgrootte van 2,00. Met *man* en *woman* geeft het −2,00. Wat laat dat zien?",
        options={
            "doors": "Dat Granite deuren met mannen verbindt",
            "right": "Eén woord per kant geeft altijd 2 of −2",
            "broken": "Dat de embeddings van losse woorden van Granite kapot zijn",
            "large": "Een groot effect, want 2 is het meeste dat WEAT kan geven",
        },
        correct="right",
        why={
            "doors": "Met *man* en *woman* is *curtain* het mannelijkere woord. Met één woord per kant bepalen de woorden die je kiest welke kant wint, en de effectgrootte is hoe dan ook 2 of −2. Zie *Wat de embeddings van Granite laten zien*.",
            "right": "De associaties verschillen ongeveer 0,001 met *masculine* en *feminine*, en ongeveer 0,003 met *man* en *woman*. Met één woord in X en één in Y is de standaardafwijking precies de helft van het verschil, dus de effectgrootte is 2 of −2, hoe klein het verschil ook is. Ethayarajh en collega's concluderen dat je met zelfgekozen attribuutwoorden bijna elk woord een genderbias kunt geven ten opzichte van een ander.",
            "broken": "Er is niets kapot: de twee associaties verschillen ongeveer 0,001 en 0,003, en met één woord per kant geeft elk verschil 2 of −2. Zie *Wat de embeddings van Granite laten zien*.",
            "large": "2 is het meeste dat WEAT kan geven, en hier komt het uit verschillen van ongeveer 0,001 en 0,003. Met één woord per kant is de effectgrootte altijd 2 of −2, dus hij zegt niets over hoe groot het verschil is.",
        },
    )
    choice_door
    return (choice_door,)


@app.cell(hide_code=True)
def text_exercise_neutralize():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. neutralize: haal de genderrichting eruit

        `neutralize(w, g)` krijgt één embedding `w` en de genderrichting `g`, met lengte 1. De eerste regel rekent `along = np.dot(w, g)` uit: hoe ver `w` langs `g` ligt. Vervang de `...` door twee regels, zoals in het uitgewerkte voorbeeld: haal `along * g` uit `w`, en geef `w` terug geschaald naar lengte 1, gedeeld door `np.linalg.norm(w)`. De check roept `neutralize(vec["nurse"], g)` aan, met `g` de genderrichting van de tien paren op Granite, en vier andere gevallen. Het plaatje onder de check stuurt dan elk genderneutraal woord van de les door jouw functie, en meet WEAT opnieuw.
        """),
        L.task("Geef `w` terug zonder zijn deel langs `g`, geschaald naar lengte 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neutralize():
    box_neutralize = L.code_box("neutralize", lines=10)
    box_neutralize
    return (box_neutralize,)


@app.cell(hide_code=True)
def check_neutralize(box_neutralize):
    result_neutralize = L.check("neutralize", box_neutralize.value)
    result_neutralize
    return (result_neutralize,)


@app.cell(hide_code=True)
def hints_neutralize():
    L.hints(
        "neutralize",
        [
            "Twee stappen, zoals in het uitgewerkte voorbeeld: trek het deel langs `g` eraf, en schaal dan naar lengte 1.",
            "Het deel langs `g` is `along * g`, een vector. `np.linalg.norm(w)` is de lengte van `w`; erdoor delen geeft lengte 1.",
            "`w = w - ___` en dan `return w / np.linalg.norm(w)`.",
        ],
        why="`w = w - along * g` en `return w / np.linalg.norm(w)`. Voor w = (0,6; 0; 0,8) en g = (0; 0; 1) geeft dat (1; 0; 0).",
    )
    return


@app.cell(hide_code=True)
def text_after_neutralize():
    mo.md(r"""
    ### Wat Neutralize veranderde

    De course stuurde elk genderneutraal woord van de les door Neutralize: de woorden over carrière, gezin, wiskunde, wetenschap en kunst, en twintig beroepen, de tien die Bolukbasi en collega's in word2vec het dichtst bij *she* vonden en de tien het dichtst bij *he*. De namen van mannen en vrouwen, en woorden zoals *he* of *sister*, blijven zoals ze zijn, omdat ze over gender gaan. Daarna ligt elk neutraal woord op 0 langs g. Wiskunde en kunst met mannelijke en vrouwelijke woorden zakt van 1,36 naar −0,33, en wetenschap en kunst van 0,96 naar 0,13. Maar de test met namen, carrière en gezin zakt alleen van 1,49 naar 1,15: de namen bleven onaangeraakt. De tien beroepen aan de kant van *she* tegen de tien aan de kant van *he*, met mannelijke en vrouwelijke woorden, gaan van 1,11 naar 0,96. En 15 van de 20 beroepen hebben nog steeds als dichtstbijzijnde beroep een van hun eigen kant van de lijst, tegen 16 ervoor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hidden():
    mo.md(r"""
    ### Oefening 5. Vrij van genderbias?
    """)
    return


@app.cell(hide_code=True)
def choice_hidden_cell():
    choice_hidden = L.choice(
        "hidden",
        question="Een team stuurt elk genderneutraal woord van zijn embedding door Neutralize. WEAT met woorden over wiskunde en kunst zakt dan van 1,36 naar −0,33, en het team schrijft: \"Onze embedding is nu vrij van genderbias.\" Wat zouden Gonen en Goldberg zeggen?",
        options={
            "right": "Dat het de bias meer verbergt dan weghaalt",
            "done": "Dat het werkt: −0,33 ligt dicht genoeg bij 0",
            "worse": "Dat het de bias erger maakte, nu tegen mannen",
            "more": "Dat ze twee richtingen moeten weghalen in plaats van één",
        },
        correct="right",
        why={
            "right": "Volgens hen verbergt zo'n methode de bias vooral, in plaats van hem weg te halen. Op Granite ook: namen met carrière en gezin geven nog 1,15, de beroepen verschillen nog 0,96 met mannelijke en vrouwelijke woorden, en 15 van de 20 beroepen houden een dichtstbijzijnde beroep van hun oude kant. De richting meet de genderassociatie van een woord; ze bepaalt die niet.",
            "done": "Bij deze ene test wel. Maar dezelfde mannelijke en vrouwelijke woorden scheiden de beroepen nog, 1,11 → 0,96, namen met carrière en gezin geven nog 1,15, en 15 van de 20 beroepen houden een dichtstbijzijnde beroep van hun oude kant. Zie *Wat Neutralize veranderde*.",
            "worse": "Een getal onder 0 bij één test laat niet zien dat de bias omgedraaid is. Dezelfde mannelijke en vrouwelijke woorden scheiden de beroepen nog, 1,11 → 0,96, en namen met carrière en gezin geven nog 1,15. Zie *Wat Neutralize veranderde*.",
            "more": "Het punt van Gonen en Goldberg is dat de richting de genderassociatie van een woord meet, maar niet bepaalt: de woorden groeperen nog met de woorden van hun oude gender. Op Granite houden 15 van de 20 beroepen een dichtstbijzijnde beroep van hun oude kant. Zie *Wat Neutralize veranderde*.",
        },
    )
    choice_hidden
    return (choice_hidden,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Embeddings: betekenis als coördinaten](/course/open/ml-embeddings-idea?lang=nl) legden dingen die op elkaar lijken dicht bij elkaar; hier zag je dat dichtbij ook kan betekenen: vaak samen genoemd in teksten, stereotypen inbegrepen. [Bias in language models](/course/open/rai-bias-in-llms?lang=nl) testte wat een model antwoordt; WEAT test wat erin zit, en heeft dezelfde grens: een klein getal laat niet zien dat er geen bias is. De volgende les, *Bias in gezichtsanalyse en beeldgeneratoren*, kijkt naar modellen die beelden zien en maken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_door, choice_hidden, result_association, result_effect, result_neutralize):
    L.wrap_up(result_association, result_effect, choice_door, result_neutralize, choice_hidden)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
