# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-sts. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-sts", "nl")
    C = L.checks   # de paren van de STS Benchmark met de cosine similarities van Granite, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Een vector voor een hele zin](/course/open/nlp-sentence-embeddings?lang=nl) en [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sentence():
    quick_sentence = L.choice(
        "quick-sentence",
        question="In *Een vector voor een hele zin* gaf het gemiddelde van de GloVe-vectoren *the dog bit the man* en *the man bit the dog* precies dezelfde vector. Waarom?",
        options={
            "same": "De twee zinnen betekenen hetzelfde",
            "length": "Elk gemiddelde heeft lengte 1",
            "right": "Een gemiddelde ziet de volgorde van de woorden niet",
            "glove": "GloVe geeft alle woorden van een zin één gedeelde vector",
        },
        correct="right",
        why={
            "right": "Beide zinnen hebben dezelfde vijf woorden, en een gemiddelde telt de woorden in elke volgorde op. Dezelfde vector betekent dus niet dezelfde betekenis: deze les vergelijkt cosine similarities met wat mensen over de betekenis zeggen.",
            "same": "Dat doen de zinnen niet: in de ene bijt de hond, in de andere de man. De gemiddelden zijn gelijk omdat beide zinnen dezelfde vijf woorden hebben, elk even vaak, en een gemiddelde ziet de volgorde niet.",
            "length": "Een gemiddelde van woordvectoren hoeft geen lengte 1 te hebben, en een lengte zegt niets over welke woorden erin gingen. Beide zinnen hebben dezelfde vijf woorden, en een gemiddelde ziet de volgorde niet.",
            "glove": "GloVe geeft elk woord een eigen vector, in elke zin dezelfde. Het gemiddelde van die vectoren is voor beide zinnen gelijk, omdat beide dezelfde vijf woorden hebben, en een gemiddelde ziet de volgorde niet. Zie [Een vector voor een hele zin](/course/open/nlp-sentence-embeddings?lang=nl).",
        },
    )
    quick_sentence
    return (quick_sentence,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="In *Meer dan accuracy: precision en recall* noemt een spamfilter 10 mails spam, en 8 daarvan zijn spam. Er zijn in totaal 20 spammails. Wat zijn de precision en de recall?",
        options={
            "both": "Allebei 0,8",
            "right": "Precision 0,8, recall 0,4",
            "swap": "Precision 0,4, recall 0,8",
            "unknown": "Precision 0,8, en voor de recall zijn ook de andere mails nodig",
        },
        correct="right",
        why={
            "right": "Van de 10 mails die het filter spam noemt, zijn er 8 spam: precision 8 / 10. Van de 20 spammails vindt het filter er 8: recall 8 / 20. Deze les rekent beide uit voor een threshold op de cosine similarity.",
            "both": "De precision is 8 / 10, van de mails die het filter spam noemt. De recall is een deel van de 20 mails die spam zijn: hoeveel daarvan vindt het filter?",
            "swap": "Het is andersom. De precision is een deel van de 10 mails die het filter spam noemt, en de recall een deel van de 20 spammails. Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "unknown": "De recall is alleen een deel van de spammails: het filter vindt er 8 van de 20, dus de recall is 0,4. De andere mails tellen alleen mee voor de false positives, en de false positives zitten in de precision.",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een forum dat een vraag herkent die al eerder gesteld is, een tool die korte antwoorden op een examenvraag nakijkt, een controle of een vertaling de betekenis heeft behouden: elk moet weten of twee zinnen hetzelfde zeggen, in welke woorden dan ook. Daniel Cer en anderen, die de STS-taak van 2017 organiseerden, noemen zulke toepassingen: machinevertaling, samenvatten, tekst genereren, vragen beantwoorden, korte antwoorden nakijken, zoeken op betekenis, en systemen voor dialoog en gesprekken.

    Een cosine similarity van twee sentence embeddings is een getal, nog geen antwoord. Deze les vergelijkt die getallen met wat mensen vonden, en maakt er een ja of een nee van: dezelfde betekenis, of niet.

    **Een voorbeeld uit het dagelijks leven.** Bij een balie voor gevonden voorwerpen zijn *een zwarte paraplu met een houten handvat* en *een donkere paraplu, handvat van hout* dezelfde paraplu. *Een zwarte paraplu met een rood handvat* deelt meer woorden met de eerste, en is een andere paraplu.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="De STS Benchmark bevat paren zinnen die mensen een score gaven voor hoe sterk de twee zinnen in betekenis op elkaar lijken, van 0, helemaal verschillend, tot 5, dezelfde betekenis. Deze les gebruikt 1.750 van die paren: korte zinnen over foto's. Mensen gaven 167 daarvan een 0. Welke cosine similarity geeft Granite Embedding die 167 paren gemiddeld?",
        options={
            "zero": "Ongeveer 0, want de betekenissen hebben niets gemeen",
            "low": "Ongeveer 0,2",
            "right": "Ongeveer 0,56",
            "high": "Ongeveer 0,9",
        },
        reveal="**Ongeveer 0,56:** gemiddeld 0,560, en geen van de 167 lager dan 0,325. *A yellow school bus parked in a field.* en *A brown horse in a green field.* kregen een 0 van mensen en 0,772 van Granite: de twee delen *in a field*, en beide beschrijven een foto. Een cosine similarity van 0,56 betekent niet *half hetzelfde*. Het gaat erom hoe een paar zich verhoudt tot andere paren, en het idee laat zien hoe.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Hoe gelijk in betekenis: STS

    **Semantic textual similarity**, kort STS, is hoe sterk twee zinnen in betekenis op elkaar lijken, als getal. Cer en anderen omschrijven STS als de mate waarin twee zinnen semantisch gelijkwaardig zijn. Van 2012 tot 2017 lieten de STS-taken van SemEval, een reeks wedstrijden waarin teams systemen op dezelfde data vergelijken, mensen paren zinnen een score geven op een schaal van 0 tot 5. Cer en anderen geven de schaal met deze uitleg, van Agirre en anderen (2013), hier in het Nederlands:

    | Score | Uitleg |
    |---|---|
    | 5 | De twee zinnen zijn volledig gelijkwaardig: de zinnen betekenen hetzelfde. |
    | 4 | De twee zinnen zijn grotendeels gelijkwaardig, maar een paar onbelangrijke details verschillen. |
    | 3 | De twee zinnen zijn ruwweg gelijkwaardig, maar wat belangrijke informatie verschilt of ontbreekt. |
    | 2 | De twee zinnen zijn niet gelijkwaardig, maar hebben een paar details gemeen. |
    | 1 | De twee zinnen zijn niet gelijkwaardig, maar gaan over hetzelfde onderwerp. |
    | 0 | De twee zinnen zijn helemaal verschillend. |

    De paren van 2017 kregen de score van betaalde krachten op Amazon Mechanical Turk. Volgens Cer en anderen kreeg elk paar vijf scores, en is de score van het paar het gemiddelde van die vijf. Een score kan dus tussen twee hele getallen liggen, zoals 3,4.

    De **STS Benchmark** verzamelt 8.628 Engelse paren van die taken, van 2012 tot 2017, met de scores erbij: uit het nieuws, van fora, en bijschriften van foto's en video's. Slechts twee van de bronnen laten de course de zinnen doorgeven met een korte bronvermelding: bijschriften van foto's uit de PASCAL-collectie, gescoord in 2014 en 2015, en zinnen over foto's uit SNLI, gescoord in 2017. Dat zijn de 1.750 paren van deze les. Naast elke Engelse zin heeft de course een Nederlandse: een machinevertaling door DeepL, gemaakt door Philip May in 2021. De scores horen bij de Engelse paren, dus deze les rekent met de cosine similarity van de Engelse zinnen; de Nederlandse vertalingen komen terug onder *Let op*, in de laatste oefening en bij het zelf proberen.
    """)
    return


@app.cell(hide_code=True)
def idea_scatter():
    mo.Html(C.idea_scatter_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_scatter():
    mo.md(r"""
    Hoe hoger de score, hoe hoger de cosine similarity van Granite gemiddeld: van 0,580 voor de paren met een score onder 0,5 tot 0,926 voor de paren van 4,5 en meer. Maar de wolken overlappen. *The gate is blue.* en *The gate is yellow.* krijgen 0,951 bij een score van 1,6, meer dan *A man is laughing with a woman* en *A man and a woman laughing.* krijgen bij een score van 4,8: 0,786. Twee zinnen die in één woord verschillen, krijgen van Granite een hoge cosine similarity, ook als dat ene woord verandert wat de foto laat zien.
    """)
    return


@app.cell(hide_code=True)
def idea_examples():
    mo.Html(C.idea_examples_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_paraphrase():
    mo.md(r"""
    ### Waar een parafrase begint

    Een **parafrase** van een zin zegt hetzelfde in andere woorden. De STS-schaal zegt niet waar een parafrase begint; deze les trekt de lijn bij 4: een paar met een score van 4 of meer telt als parafrase, de twee zinnen gelijkwaardig op onbelangrijke details na. **Parafrasedetectie**, van een paar beslissen of het een parafrase is, is dan een classificatie met twee uitkomsten.

    Met sentence embeddings heeft zo'n classifier één getal nodig: een threshold op de cosine similarity. Een paar telt als parafrase als de cosine similarity minstens de threshold is, zoals een mail vanaf een kans van 0,5 als spam telde in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl). Elk paar komt dan in één vak van de confusion matrix: een parafrase die als parafrase telt, is een true positive, een parafrase onder de threshold een false negative, een ander paar dat als parafrase telt een false positive. Precision en recall volgen zoals in die les.

    De threshold kies je op de ene set paren en meet je op een andere, zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl): een threshold die op de ene set net een precision haalt, is afgestemd op de paren van die set, en alleen paren waarop de threshold niet gekozen is, laten zien wat de threshold op nieuwe paren geeft. Granite leerde elders, dus niets leert hier van het train-deel van de STS Benchmark: samen met het dev-deel vormt het de validation set van deze les, 1.375 paren, waarvan 352 parafrases. Het test-deel, 375 paren met 91 parafrases, is de test set: daarop meet je de gekozen threshold één keer, aan het eind.
    """)
    return


@app.cell(hide_code=True)
def idea_threshold():
    mo.Html(C.idea_threshold_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_choose():
    mo.md(r"""
    Bij de threshold 0,85 tellen 494 paren van de validation set als parafrase, en 291 daarvan zijn er een: precision 0,589. Van de 352 parafrases vindt de threshold er 291: recall 0,827. Een hogere threshold telt nooit meer paren, dus de recall stijgt nooit, en de precision meestal wel, net als bij het spamfilter.

    Welke threshold goed is, hangt af van welke fout meer kost. Een forum dat twee vragen samenvoegt als de vragen hetzelfde willen weten, moet zelden twee verschillende vragen samenvoegen: het heeft een hoge precision nodig. En van alle thresholds die die precision halen, houdt de laagste de meeste recall over, omdat de recall nooit stijgt als de threshold stijgt.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **semantic textual similarity (STS)** | hoe sterk twee zinnen in betekenis op elkaar lijken, als getal; in de STS-taken een score van mensen van 0 tot 5 |
        | **STS Benchmark** | 8.628 Engelse paren zinnen uit de STS-taken van 2012 tot 2017, elk met de score erbij |
        | **parafrase** | een zin die hetzelfde zegt als een andere zin, in andere woorden; in deze les een paar met een score van 4 of meer |
        | **parafrasedetectie** | van een paar zinnen beslissen of het een parafrase is |
        | `cos[score >= 4]` | de getallen van `cos` op de plekken waar `score` 4 of meer is, met een boolean mask zoals in [Delen van een array kiezen](/course/open/np-indexing?lang=nl) |
        | `sorted(cos)` | de getallen van de array `cos` als nieuwe lijst, van laag naar hoog |
        """),
        L.watch_out("De Nederlandse zinnen zijn machinevertalingen, en de scores horen bij de Engelse paren. Een vertaling kan een paar veranderen: DeepL maakte van *gate* in de ene zin *hek* en in de andere *poort*, dus *Het hek is blauw.* en *De poort is geel.* verschillen in meer dan een kleur. Philip May controleerde negen Nederlandse paren met de hand, en vond een vertaling die deels Engels is en één met slechte grammatica. Een getal dat op de Nederlandse paren gemeten is, meet Granite en de vertaling samen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Neem paren zinnen met scores van mensen, in een validation set en een test set.
    2. Reken de cosine similarity van de sentence embeddings van elk paar uit.
    3. Trek de lijn voor een parafrase op de schaal van de scores; hier bij 4.
    4. Reken op de validation set voor elke threshold de precision en de recall uit, en kies de threshold die past bij wat een fout kost: voor een precision die je nodig hebt, de laagste threshold die de precision haalt.
    5. Meet de precision en de recall van die threshold één keer, op de test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.sts("validation", "en")` geeft twee arrays van 1.375 getallen: `cos`, de cosine similarity van de embeddings die Granite van de twee Engelse zinnen van elk paar van de validation set maakt, en `score`, de score die mensen het paar gaven. `C.sts("test", "en")` geeft de 375 paren van de test set, en `"nl"` in plaats van `"en"` de cosine similarities van de Nederlandse vertalingen, met dezelfde scores. De code telt de parafrases, neemt de gemiddelde cosine similarity van de parafrases en van de andere paren, en telt de paren die de threshold 0,85 halen.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    cos, score = C.sts("validation", "en")
    print(len(cos), np.sum(score >= 4))
    print(np.mean(cos[score >= 4]))
    print(np.mean(cos[score < 4]))
    print(np.sum(cos >= 0.85))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Van de 1.375 paren zijn er 352 parafrases. De parafrases krijgen gemiddeld 0,899, de andere paren 0,731: gemiddeld ver uit elkaar, maar de twee wolken overlappen, zoals in de afbeelding. 494 paren halen de threshold 0,85, waaronder 203 die geen parafrase zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waar op de schaal?

    Elke regel bevat een paar zinnen. Zet elk paar bij de scores die mensen het zouden geven op de STS-schaal van het idee: 4 of 5, 1 tot 3, of 0.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_scale = L.sort("sts-scale")
    sort_scale
    return (sort_scale,)


@app.cell(hide_code=True)
def check_sort(sort_scale):
    result_sort = L.check("sts-scale", sort_scale)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    De STS-schaal vraagt wat twee zinnen zeggen. Een cosine similarity kan hoog blijven als één woord verandert wat een zin zegt, zoals bij het hek dat blauw of geel is in het idee: een threshold op dat getal moet daarmee leven. De volgende twee oefeningen meten hoe goed dat gaat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pr():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. precision_recall: wat een threshold goed doet

        Schrijf `precision_recall(cos, score, threshold)`: `cos` bevat de cosine similarity van elk paar, en `score` de score. Een paar is een parafrase als de score 4 of meer is, en telt als parafrase als de cosine similarity minstens `threshold` is. Geef twee waarden terug: eerst de precision, dan de recall. Een deel dat door 0 zou delen, is `None`: de precision als geen enkel paar als parafrase telt, de recall als geen enkel paar er een is.

        Voor `precision_recall(np.array([0.9, 0.8, 0.7, 0.6]), np.array([4.5, 2.0, 4.0, 5.0]), 0.75)` is de precision 0,5 en de recall 1 / 3: twee paren halen 0,75, en een daarvan is een parafrase; van de drie parafrases vindt de threshold er een. Met de threshold 0,95 telt geen enkel paar, dus is de precision `None` en de recall 0. De check roept beide aan, en daarna gevallen van de check zelf: een paar precies op de threshold, paren zonder een enkele parafrase, en `cos` en `score` van `C.sts("validation", "en")` en van de test set.
        """),
        L.task("Geef de precision en de recall van de threshold terug, met None voor een deel dat door 0 zou delen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pr():
    box_pr = L.code_box("precision_recall", lines=16)
    box_pr
    return (box_pr,)


@app.cell(hide_code=True)
def check_pr(box_pr):
    result_pr = L.check("precision_recall", box_pr.value)
    result_pr
    return (result_pr,)


@app.cell(hide_code=True)
def hints_pr():
    L.hints(
        "precision_recall",
        [
            "Splits de cosine similarities met een boolean mask: `para`, de cosine similarities van de paren met een score van 4 of meer, en `other`, de cosine similarities van de rest. Tel `tp`, `fp` en `fn` met `np.sum` en een vergelijking met `threshold`, zoals bij het spamfilter. Zet dan `precision` en `recall` eerst op `None`, en reken elk alleen uit als het getal waardoor je deelt meer dan 0 is.",
            "Een true positive is een parafrase op of boven de threshold: `np.sum(para >= threshold)`. Welke array en welke vergelijking geven de false positives, en welke de false negatives?",
            r"`    if tp + fp > 0:` met `precision = tp / (tp + fp)` eronder, en voor de recall `    if ___ > 0:`",
        ],
        why="`para = cos[score >= 4]` en `other = cos[score < 4]`; dan `tp = np.sum(para >= threshold)`, `fp = np.sum(other >= threshold)` en `fn = np.sum(para < threshold)`. Begin met `precision = None` en `recall = None`; `if tp + fp > 0:` zet `precision = tp / (tp + fp)`, en `if tp + fn > 0:` zet `recall = tp / (tp + fn)`. Als laatste `return precision, recall`. Voor het voorbeeld: `para` is `[0.9, 0.7, 0.6]` en `other` is `[0.8]`, dus `tp` is 1, `fp` 1 en `fn` 2: precision 1 / 2, recall 1 / 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_pr():
    mo.md(r"""
    In de grafiek daalt je recall als de threshold stijgt, en stijgt de recall daarna nooit meer. Je precision stijgt, met kleine dalingen, van 0,349 bij 0,70 tot 1 vlak bij de top. Bij de hoogste cosine similarity van de validation set, 0,9942, telt maar één paar: een parafrase, dus precision 1 en recall 1 / 352. Boven elk paar, bij 1,00, telt niets, en is de precision `None`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. lowest_threshold: de laagste threshold die precies genoeg is

        Schrijf `lowest_threshold(cos, score, target)`: de laagste threshold met een precision van minstens `target`, of `None` als geen enkele threshold het doel haalt. Probeer de cosine similarities van de paren zelf als threshold, van laag naar hoog: elke andere threshold telt dezelfde paren als de eerstvolgende cosine similarity erboven. Het vak mag `precision_recall` gebruiken: de versie van de course, die werkt zoals de jouwe hoort te werken.

        Voor `lowest_threshold(np.array([0.8, 0.6, 0.9, 0.7]), np.array([4.0, 1.0, 5.0, 4.2]), 0.8)` is de uitkomst `0.7`: bij 0,6 tellen alle vier de paren en zijn er drie parafrases, precision 0,75; bij 0,7 tellen er drie en zijn alle drie parafrases, precision 1. Met het doel 0,75 is de uitkomst `0.6`. De check roept beide aan, en daarna gevallen van de check zelf: paren waarbij geen enkele threshold het doel haalt, en de validation sets van de Engelse paren en van de Nederlandse vertalingen.
        """),
        L.task("Geef de laagste cosine similarity terug waarvan de precision target haalt, of None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_low():
    box_low = L.code_box("lowest_threshold", lines=10)
    box_low
    return (box_low,)


@app.cell(hide_code=True)
def check_low(box_low):
    result_low = L.check("lowest_threshold", box_low.value)
    result_low
    return (result_low,)


@app.cell(hide_code=True)
def hints_low():
    L.hints(
        "lowest_threshold",
        [
            "Stap 4 van het recept: probeer de cosine similarities van de paren als threshold `t`, van laag naar hoog, met `sorted(cos)`, en vraag `precision_recall` om de precision van elke `t`. De eerste `t` met genoeg precision is dan de laagste; heeft geen enkele `t` genoeg, geef dan `None` terug.",
            "Schrijf `for t in sorted(cos):` met `precision, recall = precision_recall(cos, score, t)` erin. Wanneer moet de loop `t` teruggeven? Lees de opdracht nog eens: wat moet de precision zijn?",
            r"`        if ___:` met `return t` eronder",
        ],
        why="`for t in sorted(cos):` met `precision, recall = precision_recall(cos, score, t)` erin, dan `if precision >= target:` met `return t` eronder; na de loop `return None`. Voor het voorbeeld: bij 0,6 is de precision 0,75, minder dan 0,8; bij 0,7 is de precision 1, dus geeft de functie 0,7 terug.",
    )
    return


@app.cell(hide_code=True)
def text_after_low():
    mo.md(r"""
    Voor een precision van 0,8 op de Engelse paren kiest je functie 0,9418, en dat kost recall: 0,233, minder dan een kwart van de parafrases. Op de test set geeft dezelfde threshold een precision van 0,875 en een recall van 0,385: andere paren, andere getallen, al gaat de threshold mee. Op de test set halen maar 40 paren de threshold, dus één paar meer of minder verschuift de precision met ongeveer 0,02. Voor 0,9 gaat de threshold omhoog naar 0,9555, en de recall omlaag naar 0,156. Veel paren die meetellen zonder parafrase te zijn, liggen net onder de lijn: bij 0,9418 tellen 20 van zulke paren mee, en 11 daarvan kregen 3,6 of meer van mensen. Op de Nederlandse vertalingen is 0,9318 de laagste threshold die een precision van 0,8 haalt; de recall is dan 0,270.
    """)
    return


@app.cell(hide_code=True)
def text_explore_threshold():
    mo.md(r"""
    ### Probeer het: schuif de threshold

    Kies een threshold, en de Engelse paren of de Nederlandse vertalingen van de validation set. De afbeelding laat de paren zien, de threshold, de confusion matrix, de precision en de recall. Zoek de threshold waar de recall snel begint te dalen, en vergelijk de twee talen bij dezelfde threshold.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_controls():
    explore_t = mo.ui.slider(0.70, 1.00, step=0.005, value=0.85, label="Threshold", show_value=True)
    explore_data = mo.ui.radio(options={"Engelse paren": "en", "Nederlandse vertalingen": "nl"}, value="Engelse paren", label="Paren")
    mo.hstack([explore_t, explore_data], justify="start", gap=2)
    return explore_data, explore_t


@app.cell(hide_code=True)
def explore_threshold(explore_data, explore_t):
    mo.Html(C.explore_picture(explore_t.value, explore_data.value, L.lang))
    return


@app.cell(hide_code=True)
def text_explore_granite():
    mo.md(r"""
    ### Probeer het: je zinnen in Granite

    Granite Embedding draait op je computer zodra het gedownload is. Typ tot zes zinnen, één per regel: de afbeelding vergelijkt elke twee, de paren die het meest op elkaar lijken bovenaan, en zegt welke paren de threshold halen die je doel voor precision geeft op de Nederlandse vertalingen van de validation set. Probeer een zin en een parafrase ervan, dezelfde zin met *niet*, en een zin over iets anders. De threshold komt van bijschriften bij foto's: past de threshold bij jouw zinnen?
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
    explore_lines = mo.ui.text_area(value="Het museum gaat om tien uur open.\nOm tien uur opent het museum de deuren.\nHet museum gaat om tien uur niet open.\nDe soep is te zout.", label="Je zinnen, één per regel", rows=6, full_width=True)
    explore_target = mo.ui.dropdown(options=C.TARGET_CHOICES, value="0.8", label="Doel voor precision")
    mo.vstack([explore_lines, explore_target])
    return explore_lines, explore_target


@app.cell(hide_code=True)
def explore_granite(explore_lines, explore_target, granite_here):
    _ = granite_here
    mo.Html(C.granite_picture(explore_lines.value, explore_target.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een vector voor een hele zin](/course/open/nlp-sentence-embeddings?lang=nl) gaf elke zin één vector, en twee zinnen een cosine similarity. Hier beoordelen de scores van mensen die getallen: gemiddeld stijgen de getallen met de score, maar een paar dat in één belangrijk woord verschilt, kan hoger uitkomen dan een parafrase. Een threshold maakt van het getal parafrasedetectie, met precision en recall zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), gekozen op de validation set en gemeten op de test set zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl). De volgende les van het domein, *Een model voor zinnen kiezen: MTEB lezen*, vergelijkt modellen voor sentence embeddings op veel taken, waarvan STS er een is.

    **Een voorbeeld uit het dagelijks leven.** Een metaaldetector op het strand piept bij elke munt, en ook bij elk kroonkurkje. Zet je de detector lager, dan piept de detector alleen bij grote stukken metaal: minder kroonkurken, en meer munten die in het zand blijven. Waar je de detector op zet, hangt af van wat een kroonkurk kost, en wat een munt waard is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_low, result_pr, result_sort):
    L.wrap_up(result_sort, result_pr, result_low)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
