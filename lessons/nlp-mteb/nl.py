# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-mteb. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-mteb", "nl")
    C = L.checks   # de scores van 15 embedding models op de 41 taken van MTEB(eng, v2), de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Dezelfde betekenis? Gelijkenis en parafrases](/course/open/nlp-sts?lang=nl) en [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sts():
    quick_sts = L.choice(
        "quick-sts",
        question="In *Dezelfde betekenis? Gelijkenis en parafrases* gaf Granite *The gate is blue.* en *The gate is yellow.* een cosine similarity van 0,951, en gaven mensen het paar een score van 1,6 op 5. Wat laat dat paar zien?",
        options={
            "people": "Mensen gaven het paar een te lage score",
            "always": "Een cosine similarity boven 0,9 betekent een parafrase, wat mensen ook van het paar vinden",
            "right": "Twee zinnen kunnen een hoge cosine similarity krijgen en toch verschillen in betekenis",
            "colors": "Granite ziet helemaal geen kleuren",
        },
        correct="right",
        why={
            "right": "De twee zinnen delen elk woord op één na, en dat ene woord verandert hoe het hek eruitziet. Een cosine similarity meet één ding aan twee vectoren; deze les vergelijkt modellen op veel taken, die elk iets anders meten.",
            "people": "Op de STS-schaal betekent een 1 hetzelfde onderwerp, en de twee zinnen beschrijven verschillende hekken. De score klopt; de cosine similarity is hoog omdat de twee zinnen elk woord op één na delen.",
            "always": "Het paar laat het omgekeerde zien: 0,951 en een score van 1,6. Een hoge cosine similarity zegt dat twee vectoren dezelfde kant op wijzen, niet dat twee zinnen hetzelfde betekenen. Zie [Dezelfde betekenis? Gelijkenis en parafrases](/course/open/nlp-sts?lang=nl).",
            "colors": "Dan zouden de twee zinnen precies dezelfde vector krijgen, en een cosine similarity van 1. De zinnen krijgen 0,951: het ene woord dat verschilt, verschuift de vector een beetje, en de betekenis veel.",
        },
    )
    quick_sts
    return (quick_sts,)


@app.cell(hide_code=True)
def quick_check_axis():
    quick_axis = L.choice(
        "quick-axis",
        question="In *Sommen en gemiddelden langs een axis* is `m` gelijk aan `np.array([[1, 2, 3], [4, 5, 6]])`. Wat geeft `m.mean(axis=1)`?",
        options={
            "columns": "`array([2.5, 3.5, 4.5])`: één gemiddelde per kolom",
            "all": "`3.5`: het gemiddelde van alle zes getallen",
            "first": "`array([1, 4])`: de eerste van elke rij",
            "right": "`array([2., 5.])`: één gemiddelde per rij",
        },
        correct="right",
        why={
            "right": "`axis=1` werkt langs elke rij: (1 + 2 + 3) / 3 = 2 en (4 + 5 + 6) / 3 = 5.",
            "columns": "Dat is `axis=0`: met `axis=0` werkt NumPy de rijen af, met één uitkomst per kolom. `axis=1` werkt langs elke rij: wat geeft dat voor de rijen 1, 2, 3 en 4, 5, 6?",
            "all": "Dat is `m.mean()` zonder axis. Met `axis=1` neemt NumPy een gemiddelde langs elke rij: één getal per rij.",
            "first": "`m[:, 0]` geeft de eerste kolom. `mean(axis=1)` neemt het gemiddelde langs elke rij. Zie [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
        },
    )
    quick_axis
    return (quick_axis,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Alles wat met sentence embeddings werkt, begint met een keuze: welk model maakt de vectoren? Honderden modellen die teksten in vectoren omzetten, zijn gratis te downloaden, van 20 miljoen parameters tot miljarden. Een leaderboard zet de modellen op volgorde met één getal, en de bovenste van de lijst ligt voor de hand. Die keuze kan ook verkeerd uitpakken: een model dat voor een ander soort taak gemaakt is, te groot voor je computer, of nooit getest op jouw taal.

    **Een voorbeeld uit het dagelijks leven.** Een restaurantgids geeft elk restaurant één cijfer voor eten, bediening, prijs en sfeer samen. Wil je alleen een snelle, goedkope lunch, dan kan het restaurant met het hoogste cijfer de verkeerde plek zijn.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="Een benchmark van embedding models meet de modellen op taken van verschillende types, zoals STS, waar de cosine similarities de scores van mensen moeten volgen, zoals in de vorige les, en retrieval, waar de vectoren bij een vraag de teksten moeten vinden die er antwoord op geven. Van negen modellen tot 300 miljoen parameters scoort paraphrase-multilingual-mpnet-base-v2 het best op de STS-taken. Waar komt dat model uit op de retrieval-taken, tussen dezelfde negen?",
        options={
            "first": "1e: het antwoord op een vraag vinden vraagt ook welke teksten in betekenis op elkaar lijken",
            "third": "3e",
            "fifth": "5e, in het midden",
            "right": "8e, op één na laatste",
        },
        reveal="**8e, op één na laatste:** 37,1 op retrieval, tegen 55,7 voor granite-embedding-125m-english, de beste van de negen daar, terwijl het model op STS eerste is met 80,5. Een model dat paren zinnen goed op volgorde zet, kan zwak zijn in documenten vinden bij een vraag. Het idee laat zien hoe één gemiddelde dat verbergt, en hoe je erdoorheen leest.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Eén benchmark, veel taken: MTEB

    Een model dat een tekst omzet in één vector, zoals Granite in de vorige les, heet een **embedding model**. **MTEB**, de Massive Text Embedding Benchmark, vergelijkt embedding models op veel taken tegelijk. Niklas Muennighoff, Nouamane Tazi, Loïc Magne en Nils Reimers publiceerden de benchmark in 2023, met 8 soorten taken over 58 datasets en 112 talen. Elke **taak** geeft een model de teksten van één dataset, gebruikt de vectoren op een vaste manier, en meet de uitkomst met één getal, de **hoofdscore** van de taak. Taken die de vectoren op dezelfde manier gebruiken, vormen een **taaktype**, met dezelfde hoofdscore:

    | Taaktype | Wat de vectoren moeten doen | Hoofdscore |
    |---|---|---|
    | retrieval | bij een zoekvraag de documenten vinden die er antwoord op geven, uit vele: de documenten met de hoogste cosine similarity met de vector van de zoekvraag | nDCG@10: hoe hoog de goede documenten staan binnen de eerste 10 |
    | reranking | een korte lijst kandidaten bij een zoekvraag op volgorde zetten | MAP: hoe hoog de goede kandidaten staan, als gemiddelde over de zoekvragen |
    | classificatie | dienen als de features van een logistische regressie die de labels van de teksten leert | accuracy |
    | clustering | groepen vormen die bij de labels van de teksten passen, met k-means: een methode die dicht bij elkaar liggende vectoren in één groep zet | V-measure: hoe goed de groepen en de labels overeenkomen |
    | paarclassificatie | beslissen of twee teksten dubbel zijn of parafrases van elkaar, met een threshold op de cosine similarity | average precision: de precision bij elke threshold waar er een echt dubbel paar bij komt, gemiddeld |
    | STS | paren zinnen cosine similarities geven in de volgorde van de scores van mensen, zoals in de vorige les | Spearman-correlatie: hoe goed de twee volgordes overeenkomen |
    | samenvatting | samenvattingen die een machine maakte, beoordelen op de cosine similarity met samenvattingen die mensen schreven, in de volgorde van het oordeel van mensen | Spearman-correlatie |

    Het achtste type van MTEB, bitext mining, koppelt zinnen in twee talen, en hoort niet in een Engelse benchmark. nDCG en MAP komen terug in latere lessen, over zoekmachines. Het leaderboard toont elke hoofdscore maal 100, dus de meeste scores lopen van 0 tot 100, en hoger is beter.

    ### Eén getal voor het leaderboard

    MMTEB, de meertalige opvolger, door Kenneth Enevoldsen en vele anderen (2025), heeft ruim 500 gecontroleerde taken in meer dan 250 talen. De Engelse benchmark van het leaderboard, **MTEB(eng, v2)**, neemt daaruit 41 Engelse taken van zeven types: 10 retrieval, 2 reranking, 8 classificatie, 8 clustering, 3 paarclassificatie, 9 STS en 1 samenvatting. De STS Benchmark van de vorige les is een van de 9 STS-taken. Een **leaderboard** zet modellen op volgorde met één getal, en dit leaderboard biedt er twee:

    - **Mean(Task)** is het gemiddelde over alle 41 taken. Het leaderboard noemt dat zelf een naïef gemiddelde over alle taken van de benchmark. Een type met veel taken telt zwaarder: retrieval telt voor 10 van de 41, samenvatting voor 1. Het leaderboard waarschuwt ook dat het gemiddelde taken met meer spreiding in de scores te zwaar kan laten wegen. Op een retrieval-taak hebben de scores van de 15 modellen van deze les gemiddeld een standaardafwijking van 12,8 punten; op een reranking-taak van 2,8.
    - **Mean(TaskType)** neemt eerst het gemiddelde van elk type, en dan het gemiddelde van die zeven taaktypegemiddelden: elk type telt voor een zevende.

    Beide zijn **gewogen gemiddelden** van de zeven taaktypegemiddelden: elk taaktypegemiddelde maal de weging, opgeteld, en gedeeld door de som van de wegingen. Mean(TaskType) geeft elk type de weging 1. Mean(Task) geeft elk type het aantal taken van dat type: 10 voor retrieval, 1 voor samenvatting. Een model met 40 op retrieval en 80 op STS krijgt met de wegingen 3 en 1: (3 × 40 + 1 × 80) / 4 = 50.

    ### Wat één getal verbergt

    De tabel toont de negen modellen tot 300 miljoen parameters, van de 15 waarvan de course de scores meelevert.
    """)
    return


@app.cell(hide_code=True)
def idea_board():
    mo.Html(C.board_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_board():
    mo.md(r"""
    granite-embedding-125m-english is eerste op beide gemiddelden, en eerste op retrieval, reranking en clustering. Op classificatie en op paarclassificatie is het model 8e van de negen. multilingual-e5-base is eerste op classificatie, multilingual-e5-small op paarclassificatie, paraphrase-multilingual-mpnet-base-v2 op STS en granite-embedding-278m-multilingual op samenvatting. Ook de twee gemiddelden zijn het niet overal eens: granite-embedding-30m-english is 4e op Mean(Task) en 5e op Mean(TaskType), multilingual-e5-small andersom. De lijnen hieronder volgen de plaatsen van elk model.
    """)
    return


@app.cell(hide_code=True)
def idea_ranks():
    mo.Html(C.ranks_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_read():
    mo.md(r"""
    In het onderzoek van Muennighoff en de anderen, met 33 modellen, gold hetzelfde: geen enkel model was het best op alle zeven Engelse taaktypes, en modellen die op STS gericht waren, deden het slecht op retrieval.

    De tabel laat de zes grotere modellen van de 15 weg. Het grootste, Qwen3-Embedding-8B, heeft 7,6 miljard parameters, 333 keer zoveel als all-MiniLM-L6-v2, en meer parameters kosten meer geheugen, en meer tijd per tekst.

    ### Lees de taken die op de jouwe lijken

    De blogpost waarmee het huidige leaderboard van MTEB in 2026 verscheen, zegt het zo: een benchmark meet zelden precies wat jij belangrijk vindt, en vaak doe je het een stuk beter als je het leaderboard op je eigen gebruik afstemt. Bepaal dus wat je systeem met de vectoren doet, geef elk taaktype een weging voor hoeveel het daarop lijkt, en zet de modellen op volgorde met het gewogen gemiddelde met jouw weging.

    Ook je taal telt: MTEB(eng, v2) bevat alleen Engelse taken. Voor het Nederlands is er MTEB-NL, van Nikolay Banar en anderen (2026), met 40 Nederlandse datasets. Daar komt de eigen Granite van de course, granite-embedding-107m-multilingual, als 25e van 45 modellen uit op het gemiddelde over de taaktypes, met 57,0; het beste model, Qwen3-Embedding-4B, haalt 69,2. granite-embedding-125m-english, de beste van de negen op de Engelse taken, is alleen voor het Engels gemaakt. De course gebruikt de eigen Granite omdat het model klein is en Nederlands kent, een van de 12 talen van het model.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **embedding model** | een model dat een tekst omzet in één vector, zoals Granite |
        | **MTEB** | de Massive Text Embedding Benchmark: taken die embedding models vergelijken; MTEB(eng, v2) is de Engelse benchmark van het leaderboard, met 41 taken |
        | **taak** | één dataset, een vaste manier om de vectoren van de teksten te gebruiken, en één hoofdscore |
        | **hoofdscore** | het ene getal dat een model op een taak meet, zoals accuracy bij classificatie |
        | **taaktype** | taken die de vectoren op dezelfde manier gebruiken, met dezelfde hoofdscore: retrieval, reranking, classificatie, clustering, paarclassificatie, STS en samenvatting |
        | **leaderboard** | een tabel die modellen op volgorde zet met de scores op een benchmark |
        | **Mean(Task)** | het gemiddelde van de hoofdscores van een model over alle taken van een benchmark |
        | **Mean(TaskType)** | het gemiddelde van de taaktypegemiddelden van een model: elk taaktype telt even zwaar |
        | **gewogen gemiddelde** | elk getal maal de weging, opgeteld, en gedeeld door de som van de wegingen |
        | **Spearman-correlatie** | hoe goed twee volgordes overeenkomen, van −1 tot 1: 1 als beide dezelfde dingen in dezelfde volgorde zetten |
        | **zero-shot** | van een model op een taak: niet getraind op de data van die taak |
        | `scores[:, types == "STS"]` | alle rijen van `scores`, en van de kolommen alleen de kolommen waar de mask True is |
        | `300_000_000` | een getal met `_` tussen groepjes cijfers, om het makkelijker te lezen; Python slaat de `_` over |
        | `np.array(columns)` | maakt van een lijst met arrays van dezelfde lengte één tabel, één rij per array, zoals bij een lijst met lijsten |
        """),
        L.watch_out("Een model kan getraind zijn op de data van een taak, en dan zegt de score op die taak minder over teksten die het model nooit zag. MMTEB bedoelde MTEB(eng, v2) als zero-shot benchmark, voor modellen die niet op de taken getraind zijn, om overfitting te voorkomen. Over MTEB zelf schrijven Muennighoff en de anderen dat de scores geen rekening houden met overlap tussen de datasets van MTEB en de trainingsdata van modellen. Het leaderboard markeert op welke taken elk model getraind is: voor de vier Granite-modellen van deze les 5 van de 41, waaronder AskUbuntuDupQuestions, een van de twee reranking-taken; voor de Qwen3- en e5-modellen en GritLM-7B 2. Zonder die vijf taken blijft granite-embedding-125m-english eerste van de negen op Mean(Task), met 61,6 tegen 61,4 voor multilingual-e5-base."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf op wat je systeem met de vectoren doet, en in welke taal.
    2. Geef elk taaktype een weging voor hoeveel het daarop lijkt: 0 voor wat je systeem nooit doet.
    3. Bereken voor elk model het gemiddelde per taaktype uit de scores per taak.
    4. Zet de modellen op volgorde met het gewogen gemiddelde van de taaktypegemiddelden, met jouw weging.
    5. Streep weg wat niet past: te veel parameters voor je computer, niet jouw taal, of getraind op de taken die je het zwaarst laat wegen.
    6. Probeer de beste twee of drie op een paar honderd eigen voorbeelden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.mteb()` geeft vijf waarden: `scores`, een array met één rij per model en één kolom per taak, elk getal de hoofdscore van het model op de taak maal 100; `models`, de 15 namen, zoals op Hugging Face; `tasks`, de 41 namen van de taken; `types`, een array met het taaktype van elke kolom; en `parameters`, het aantal parameters van elk model. De code telt de taken van twee types, zoekt het model met de hoogste Mean(Task), telt de modellen tot 300 miljoen parameters, en houdt de kolommen van de STS-taken over.
    """)
    return


@app.cell
def example_cell():
    import numpy as np

    scores, models, tasks, types, parameters = C.mteb()
    print(scores.shape)
    print(np.sum(types == "Retrieval"), np.sum(types == "Summarization"))
    print(models[scores.mean(axis=1).argmax()])
    small = parameters <= 300_000_000
    print(np.sum(small))
    sts = scores[:, types == "STS"]
    print(sts.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    15 modellen en 41 taken, waarvan 10 retrieval-taken en 1 samenvattingstaak. Over alle 41 taken heeft Qwen/Qwen3-Embedding-8B het hoogste gemiddelde: de eerste plaats op Mean(Task). Negen modellen hebben tot 300 miljoen parameters. `scores[:, types == "STS"]` houdt alle 15 rijen over en de 9 kolommen van de STS-taken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Op welk taaktype lijkt het?

    Elke regel is een gebruik van een embedding model. Zet elk gebruik bij het taaktype van MTEB dat er het meest op lijkt.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_uses = L.sort("use-cases")
    sort_uses
    return (sort_uses,)


@app.cell(hide_code=True)
def check_sort(sort_uses):
    result_sort = L.check("use-cases", sort_uses)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    Een echt systeem doet vaak meer dan één van deze dingen, met hetzelfde model. De volgende oefening berekent voor elk model het gemiddelde per taaktype; de oefening daarna laat de types wegen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. type_means: één gemiddelde per taaktype

        Schrijf `type_means(scores, types, kinds)`: `scores` bevat de scores van elk model op elke taak, één rij per model en één kolom per taak; `types` het taaktype van elke kolom; `kinds` een lijst met taaktypes. Geef een array terug met één rij per model en één kolom per type van `kinds`, in die volgorde: het gemiddelde van de scores van het model op de taken van dat type.

        Voor `type_means(np.array([[60.0, 70.0, 40.0], [80.0, 60.0, 30.0]]), np.array(["STS", "STS", "Retrieval"]), ["Retrieval", "STS"])` is de uitkomst `[[40.0, 65.0], [30.0, 70.0]]`: het eerste model scoort 40 op de ene retrieval-taak en gemiddeld 65 op de twee STS-taken. Met `["STS", "Retrieval"]` wisselen de twee kolommen van plaats. De check roept beide aan, en dan eigen gevallen: `kinds` met maar een deel van de types, en de scores van `C.mteb()` met `C.KINDS`, de zeven types in de volgorde van deze les, en met twee types.
        """),
        L.task("Geef het gemiddelde per model en per type van kinds terug, één rij per model en één kolom per type."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("type_means", lines=10)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("type_means", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "type_means",
        [
            "Begin met een lege lijst `columns`, en ga met een loop langs `kinds`. Voeg voor elke `kind` met `append` één kolom toe aan `columns`: het gemiddelde van elk model op de taken van dat type. Aan het eind heeft `np.array(columns)` één rij per type; draai de array om met `.T`.",
            "Het uitgewerkte voorbeeld houdt de STS-kolommen over met `scores[:, types == \"STS\"]`. Wat staat er in je loop in plaats van `\"STS\"`?",
            r"`        columns.append(___)`",
        ],
        why="`columns = []`, dan `for kind in kinds:` met daarin `columns.append(scores[:, types == kind].mean(axis=1))`; als laatste `return np.array(columns).T`. Voor het voorbeeld: retrieval houdt de derde kolom over, met de gemiddelden `[40, 30]`; STS de eerste twee, met de gemiddelden `[65, 70]`. `np.array(columns)` is dan `[[40, 30], [65, 70]]`, één rij per type, en `.T` draait dat om tot `[[40, 65], [30, 70]]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_means():
    mo.md(r"""
    Met alle 15 modellen heeft Qwen3-Embedding-8B het hoogste getal in vijf van de zeven kolommen, en e5-mistral-7b-instruct in de andere twee. Verder omlaag kruisen de rijen van de kleine modellen elkaar: granite-embedding-125m-english heeft 55,65 op retrieval, meer dan de 48,97 van multilingual-e5-base, en 68,29 op classificatie, minder dan de 73,81 van dat model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. best_for: het beste model bij jouw weging

        Schrijf `best_for(means, weights, models)`: `means` bevat de taaktypegemiddelden van elk model, één rij per model en één kolom per type, zoals `type_means` de gemiddelden geeft; `weights` één weging per kolom; `models` de namen van de modellen, in de volgorde van de rijen. Geef twee waarden terug: de naam van het model met het hoogste gewogen gemiddelde, en dat gewogen gemiddelde.

        Voor `best_for(np.array([[50.0, 80.0], [60.0, 60.0]]), np.array([3.0, 1.0]), ["a", "b"])` is de uitkomst `("b", 60.0)`: model a krijgt (3 × 50 + 1 × 80) / 4 = 57,5, en model b (3 × 60 + 1 × 60) / 4 = 60. Met de wegingen 1 en 1 wint a met 65,0. De check roept beide aan, en dan eigen gevallen: een weging van 0, de taaktypegemiddelden van de 15 modellen van MTEB met de weging 1 voor elk type en met het aantal taken van elk type, en die van de negen modellen tot 300 miljoen parameters.
        """),
        L.task("Geef de naam van het model met het hoogste gewogen gemiddelde terug, en dat gewogen gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best_for", lines=8)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best_for", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best_for",
        [
            "Stap 4 van het recept: bereken `total`, het gewogen gemiddelde van elke rij van `means` in één keer, als de gewogen sommen van de rijen gedeeld door `np.sum(weights)`. Dan geeft `total.argmax()` de plaats van de hoogste, `best`; geef `models[best]` en `total[best]` terug.",
            "In het voorbeeld is `np.sum(weights)` gelijk aan 4. Welk product uit [Een hele batch tegelijk: matrixvermenigvuldiging](/course/open/math-matrix-multiply?lang=nl) geeft de gewogen sommen 230 en 240 van de twee rijen in één keer?",
            r"`    total = ___ / np.sum(weights)`",
        ],
        why="`total = means @ weights / np.sum(weights)`: `means @ weights` is het inproduct van elke rij met `weights`, de gewogen som van elk model. Dan `best = total.argmax()` en `return models[best], total[best]`. Voor het voorbeeld: `means @ weights` is `[230, 240]`, gedeeld door 4 is dat `[57.5, 60]`, en de hoogste staat op plaats 1, dus `(\"b\", 60.0)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_best():
    mo.md(r"""
    Onder de negen kleine modellen kiest je functie granite-embedding-125m-english voor Mean(TaskType), Mean(Task), het zoekvak en groeperen op onderwerp. Voor post sorteren kiest de functie multilingual-e5-base, voor dubbele teksten vinden multilingual-e5-small, en voor gelijkenis scoren paraphrase-multilingual-mpnet-base-v2. Onder alle 15 wint Qwen3-Embedding-8B zes van de zeven wegingen, en e5-mistral-7b-instruct de weging voor dubbele teksten. Mean(Task) geeft Qwen3-Embedding-8B 75,22 en Mean(TaskType) 68,71: hetzelfde model op een andere hoogte, omdat samenvatting en reranking, waar elk model laag scoort, voor 1 en 2 van de 41 wegen in Mean(Task), en elk voor een zevende in Mean(TaskType).
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: je eigen weging

    Geef elk taaktype een weging van 0 tot 3 voor je eigen gebruik, en kies hoe groot een model mag zijn. De tabel zet de modellen op volgorde met jouw gewogen gemiddelde, en laat zien hoe ver je weging elk model verschuift ten opzichte van de plaats op Mean(TaskType). Vink het vakje aan om de vijf taken weg te laten waarop een deel van de 15 modellen getraind is. Hoe hoog krijg je de eigen Granite van de course onder de modellen tot 300 miljoen parameters?
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_weights = mo.ui.array([mo.ui.slider(0, 3, step=1, value=1, label=name, show_value=True) for name in C.type_labels(L.lang)])
    explore_limit = mo.ui.dropdown(options=list(C.LIMITS[L.lang]), value=list(C.LIMITS[L.lang])[0], label="Grootte")
    explore_zero = mo.ui.checkbox(value=False, label="Alleen taken waarop geen van de modellen getraind is")
    mo.vstack([explore_weights.vstack(gap=0.25), mo.hstack([explore_limit, explore_zero], justify="start", gap=2)])
    return explore_limit, explore_weights, explore_zero


@app.cell(hide_code=True)
def explore(explore_limit, explore_weights, explore_zero):
    mo.Html(C.explore_picture(explore_weights.value, explore_limit.value, explore_zero.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_explore():
    mo.md(r"""
    Geen enkele weging van 0 tot 3 zet granite-embedding-107m-multilingual op de eerste plaats van de negen, met of zonder de vijf taken: hoger dan 3e komt het model niet, bijvoorbeeld met retrieval 1, reranking 1 en samenvatting 2 op alle 41 taken. Voor Engelse teksten passen andere kleine modellen beter; de course heeft een model nodig dat ook Nederlands kent.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Dezelfde betekenis? Gelijkenis en parafrases](/course/open/nlp-sts?lang=nl) mat Granite op één taak, STS. MTEB meet embedding models op veel taken tegelijk, en een leaderboard zet de modellen op volgorde met één gemiddelde. Met een gemiddelde langs een axis, zoals in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl), berekende je zelf het gemiddelde per taaktype, en met een gewogen gemiddelde zette je de modellen op volgorde voor je eigen gebruik. De volgende les van het domein, *Een label per woord: sequence labeling en BIO*, gaat van hele zinnen naar losse woorden: een label voor elk woord.

    **Een voorbeeld uit het dagelijks leven.** Een tienkamp kroont de beste atleet over tien onderdelen. Een estafetteploeg die een sprinter zoekt, kijkt naar de 100 meter, en de winnaar van de tienkamp is daar misschien niet de snelste.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_best, result_means, result_sort):
    L.wrap_up(result_sort, result_means, result_best)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
