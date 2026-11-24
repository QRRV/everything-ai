# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-bert. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-bert", "nl")
    C = L.checks   # de scores van BERT-Tiny bij de gaten van de course, de rangen gemeten op Tatoeba, BERT-Tiny, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Encoders, decoders en de GPT-lijn](/course/open/lm-model-families?lang=nl) en [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_encoder():
    quick_encoder = L.choice(
        "quick-encoder",
        question="In *Encoders, decoders en de GPT-lijn* gaat een tekst van zes tokens een encoder in. Naar welke tokens kijkt het derde token?",
        options={
            "before": "De eerste twee, en zichzelf",
            "right": "Alle zes",
            "after": "De drie erna, en zichzelf",
        },
        correct="right",
        why={
            "right": "In een encoder kijkt elk token naar alle tokens, ervoor en erna. Dan wordt de manier waarop een decoder leert, het volgende token raden, veel te makkelijk: deze les laat zien waarom, en waar BERT in plaats daarvan van leert.",
            "before": "Zo werkt een decoder, zoals GPT: elk token kijkt naar zichzelf en de tokens ervoor. Welke tokens verbergt een decoder, en een encoder niet? Zie [Encoders, decoders en de GPT-lijn](/course/open/lm-model-families?lang=nl).",
            "after": "Geen van de modellen in *Encoders, decoders en de GPT-lijn* kijkt alleen vooruit. Verbergt een encoder de tokens vóór het derde? Zie [Encoders, decoders en de GPT-lijn](/course/open/lm-model-families?lang=nl).",
        },
    )
    quick_encoder
    return (quick_encoder,)


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question="Wat geeft `bert.pieces(\"kayak\")` in *Eén vector per woord, of één per gebruik*?",
        options={
            "whole": "`['[CLS]', 'kayak', '[SEP]']`",
            "bare": "`['kay', '##ak']`",
            "right": "`['[CLS]', 'kay', '##ak', '[SEP]']`",
            "letters": "`['[CLS]', 'k', '##a', '##y', '##a', '##k', '[SEP]']`",
        },
        correct="right",
        why={
            "right": "BERT-Tiny knipt *kayak* in *kay* en *##ak*, waarbij *##* betekent dat het stukje een woord voortzet, en zet *[CLS]* ervoor en *[SEP]* erachter. In deze les raadt BERT-Tiny zulke stukjes.",
            "whole": "*kayak* is geen van de 30.522 stukjes van BERT-Tiny. Wat doet `bert.pieces` met een woord dat het model niet heeft? Zie [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl).",
            "bare": "`bert.pieces` geeft de stukjes die het model leest, en het model leest meer dan het woord. Wat gaat er bij elke tekst nog meer BERT-Tiny in? Zie [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl).",
            "letters": "BERT-Tiny gebruikt losse letters alleen waar niets langers past. Uit welke langere stukjes kan *kayak* bestaan? Zie [Eén vector per woord, of één per gebruik](/course/open/nlp-static-vs-contextual?lang=nl).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een decoder leert door het volgende token te raden: elk token kijkt alleen naar de tokens ervoor, dus het volgende blijft verborgen. In een encoder kijkt elk token naar alle tokens, ook naar het volgende. Het volgende token raden leert het model dan niets: het antwoord staat al in de input. Jacob Devlin en anderen bij Google, die BERT in 2018 online zetten en in 2019 publiceerden, schreven dat elk woord zichzelf dan via een omweg kan zien, zodat het model het woord zonder moeite voorspelt.

    De uitweg van de makers was een paar woorden van een tekst te verbergen, en het model de verborgen woorden te laten raden uit de woorden aan beide kanten. In deze les maak je zo'n trainingsvoorbeeld, meet je hoe goed BERT-Tiny raadt, en reken je de loss uit waarvan het model leerde.

    **Een voorbeeld uit het dagelijks leven.** In *De ___ blafte de hele nacht, dus niemand sliep.* bepalen de woorden na het gat wat er ontbreekt: iets dat blaft.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_first = L.guess(
        "guess",
        question="De course nam 500 Engelse zinnen van Tatoeba, een verzameling zinnen die vrijwilligers schrijven en vertalen, en verborg elk woord dat BERT-Tiny als één stukje houdt, één woord tegelijk: 3.031 woorden in totaal. Bij hoeveel ervan is de eerste gok van BERT-Tiny, het stukje met de hoogste score, het verborgen woord?",
        options={
            "low": "Ongeveer 3 op de 100",
            "right": "Ongeveer 30 op de 100",
            "mid": "Ongeveer 60 op de 100",
            "high": "Bijna altijd: 90 op de 100",
        },
        reveal="**Ongeveer 30 op de 100**: 919 van de 3.031, 0,303. Bij 1.634 ervan, 0,539, zit het verborgen woord bij de eerste vijf gokken. Veel missers zijn toch goede gokken: in *I drink a cup of [MASK] every morning.* is de eerste gok van BERT-Tiny *breakfast* (ontbijt), en komt *coffee* tweede.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Masked language modeling

    **Masked language modeling** (MLM) traint een model door een paar stukjes van een tekst te verbergen en het model de stukjes te laten raden uit de rest. Daniel Jurafsky en James H. Martin beschrijven het in *Speech and Language Processing* zo: in plaats van het volgende woord te voorspellen, verberg je een woord in het midden, en vraag je het model dat woord te raden uit de woorden aan beide kanten. Zo'n invuloefening is ouder dan computers: het is de **cloze task**, in het Nederlands ook clozetoets, die W. L. Taylor in 1953 beschreef als manier om te meten hoe leesbaar een tekst is.

    BERT kiest willekeurig 15% van de stukjes van een tekst. Een gekozen stukje wordt in 80% van de gevallen **[MASK]**, in 10% een willekeurig stukje, en in 10% blijft het zoals het was. Het voorbeeld in het paper is *my dog is hairy* (mijn hond is harig), met *hairy* gekozen: BERT ziet dan *my dog is [MASK]*, *my dog is apple* of *my dog is hairy*. Op elke gekozen plek moet het model het oorspronkelijke stukje geven. De andere plekken helpen bij het raden, maar tellen niet mee in de loss. Waarom niet op elke gekozen plek [MASK]? Dat is oefening 2.

    ### Van een vector naar een gok

    In BERT-Tiny heeft elke plek na het laatste block een vector van 128 getallen. De **head** maakt daar een gok van. De head stuurt de vector nog door één kleine layer, neemt dan het inproduct met de vector van elk stukje uit de tabel van 30.522 stukjes van het model, dezelfde tabel waarmee het model begint, en telt er per stukje één getal bij op: 30.522 scores. Eén tabel aan beide kanten gebruiken heet weight tying; een latere les, *Weight tying*, gaat daarop in. Softmax maakt van de scores kansen. De loss op een gekozen plek is de cross-entropy: min de log van de kans op het goede stukje. De loss van een tekst is het gemiddelde over de gekozen plekken.

    De **rang** van een stukje is de plaats in de lijst van alle stukjes, gerangschikt van de hoogste score omlaag: 1 voor de eerste gok.

    ### Wat de woorden na het gat waard zijn

    BERT-Tiny leerde met beide kanten. Om te zien wat de woorden na een gat opleveren, verborg de course de 3.031 woorden van Tatoeba nog een keer, en maakte deze keer ook elk stukje na het gat een [MASK], zodat alleen de woorden ervoor overblijven. Dat vraagt veel van BERT-Tiny: het model leerde nooit met zoveel [MASK]s achter elkaar, terwijl een decoder juist leert met alleen de woorden ervoor. De afbeelding laat het verschil toch zien.
    """)
    return


@app.cell(hide_code=True)
def idea_sides():
    mo.Html(C.sides_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        Met beide kanten is de eerste gok van BERT-Tiny bij 0,303 van de 3.031 het verborgen woord; met alleen de woorden ervoor bij 0,140, nog niet half zo vaak. De helft van de verborgen woorden heeft rang 4 of beter met beide kanten, en rang 25 of beter met alleen de woorden ervoor.

        ### Next sentence prediction

        BERT leerde tegelijk een tweede taak: **next sentence prediction**. Het model kreeg twee zinnen, en moest zeggen of de tweede in de oorspronkelijke tekst direct na de eerste kwam, of van ergens anders; de helft van de paren was van elke soort. De makers van BERT vonden dat drie van de gemeten taken slechter gingen zonder deze tweede taak. In 2019 trainden Yinhan Liu en anderen BERT opnieuw, als RoBERTa, en vonden, anders dan de makers van BERT, dat de resultaten op andere taken zonder next sentence prediction gelijk bleven of iets beter werden. DistilBERT en ModernBERT laten de taak ook weg.

        ### De familie van BERT

        | Model | Jaar | Parameters | Wat er veranderde |
        |---|---|---|---|
        | BERT | 2019 | 110 miljoen, 340 miljoen | MLM met 15% van de stukjes, plus next sentence prediction; 3,3 miljard woorden uit boeken en de Engelse Wikipedia |
        | BERT-Tiny | 2019 | 4,4 miljoen | hetzelfde recept, met 2 blocks en vectoren van 128 getallen; uitgebracht in 2020 |
        | RoBERTa | 2019 | 355 miljoen | langer getraind, met grotere batches, op 160 GB tekst in plaats van 16 GB; geen next sentence prediction |
        | DistilBERT | 2019 | 66 miljoen | de helft van de blocks, getraind om dezelfde antwoorden te geven als BERT |
        | DeBERTa | 2020 | 134 miljoen, 1,5 miljard | één vector voor een stukje en één voor de plek, apart gehouden in de attention |
        | ModernBERT | 2024 | 149 miljoen, 395 miljoen | teksten tot 8.192 tokens; 2 biljoen tokens, met code erbij; 30% van de stukjes verborgen |
        | EuroBERT | 2025 | 210 miljoen, 610 miljoen, 2,1 miljard | 15 talen, met het Nederlands erbij; tot 8.192 tokens; 50% van de stukjes verborgen, 10% in de laatste fase |

        Benjamin Warner en anderen, de makers van ModernBERT, leggen uit waarom zulke modellen nog gemaakt worden: voor zoeken en classificeren presteren encoders zoals BERT volgens de makers sterk in verhouding tot de grootte, vergeleken met grotere decoders.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **masked language modeling** (MLM) | een model trainen door een paar stukjes van een tekst te verbergen en het model de stukjes te laten raden uit de stukjes aan beide kanten |
        | **cloze task** | een invuloefening: een tekst met gaten, in te vullen uit de woorden eromheen; in het Nederlands ook clozetoets |
        | **BERT** | een encoder van Google (2018, gepubliceerd in 2019) die leerde met MLM; de naam staat voor Bidirectional Encoder Representations from Transformers |
        | **[MASK]** | het stukje dat in de plaats van een verborgen stukje staat |
        | **head** | het laatste deel van het model: van de vector van een plek naar een score voor elk stukje |
        | **rang** | de plaats van een stukje in de lijst van alle stukjes, gerangschikt van de hoogste score omlaag; 1 voor de eerste gok |
        | **next sentence prediction** | de tweede taak van BERT: komt de tweede van twee zinnen direct na de eerste? |
        | `t[[2, 5]]` | rij 2 en rij 5 van de tensor of array `t`, in deze volgorde, als een nieuwe tabel van twee rijen |
        | `t[rows]` | hetzelfde met een lijst van rijnummers `rows`, zoals `rows = [2, 5]` |
        """),
        L.watch_out("De rang en de loss kijken naar één verborgen stukje. *breakfast* in *I drink a cup of [MASK] every morning.* past goed in de zin, maar telt als misser, omdat de course *coffee* verborg."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Knip de tekst in stukjes, en kies willekeurig 15% van de plekken.
    2. Zet op elke gekozen plek in 80% van de gevallen [MASK], in 10% een willekeurig stukje, en houd in 10% het stukje. Het label van een gekozen plek is het oorspronkelijke stukje; elke andere plek krijgt geen label.
    3. Laat het model rekenen, en laat de head op elke plek een score geven voor elk stukje.
    4. Neem op elke gekozen plek de cross-entropy van de scores met het goede stukje, en neem het gemiddelde over de gekozen plekken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De knop hieronder downloadt BERT-Tiny. De code kijkt dan naar de [MASK] van *I drink a cup of [MASK] every morning.* `bert.piece_scores` van de course geeft de scores van de head van BERT-Tiny op elke plek van een tekst: één rij van 30.522 scores per stukje. `bert.tokenizer().vocab` is de lijst van de 30.522 stukjes van BERT-Tiny, zodat `vocab[j]` het stukje met nummer `j` is. De code maakt met softmax kansen van de rij bij de [MASK], en print de beste gok, en dan de kans en de loss van *coffee*.
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
    import math
    import numpy as np

    sentence = "I drink a cup of [MASK] every morning."
    pieces = bert.pieces(sentence)
    k = pieces.index("[MASK]")
    scores = bert.piece_scores(sentence)[k]
    p = np.exp(scores) / np.sum(np.exp(scores))
    vocab = bert.tokenizer().vocab
    print(pieces)
    print(k, len(scores))
    best = np.argmax(p)
    print(vocab[best], round(float(p[best]), 3))
    coffee = vocab.index("coffee")
    print("coffee", round(float(p[coffee]), 3), round(-math.log(p[coffee]), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De zin heeft 11 stukjes, en de [MASK] staat op plek 6, geteld vanaf 0, met *[CLS]* op plek 0. De rij heeft 30.522 scores, één per stukje. De beste gok is *breakfast*, met een kans van 0,069. *coffee* krijgt 0,061 en komt dus tweede; de loss van *coffee*, min de log van de kans, is 2,803.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_example():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. make_example: een trainingsvoorbeeld

        Schrijf `make_example(pieces, replace)`. `pieces` is een lijst van stukjes; `replace` is een plan: een dict met een gekozen plek als key en als value wat daar in komt, `"[MASK]"`, een willekeurig stukje, of het stukje zelf. Geef twee lijsten terug, elk zo lang als `pieces`: `inputs`, wat het model op elke plek ziet, en `labels`, het stukje dat het model op elke gekozen plek moet geven, en None op elke andere plek.

        Voor `make_example(["the", "cat", "sat", "on", "the", "mat", "."], {1: "[MASK]", 5: "[MASK]"})` is de uitkomst `(["the", "[MASK]", "sat", "on", "the", "[MASK]", "."], [None, "cat", None, None, None, "mat", None])`. Vervang de `...`. De check roept dat aan en `make_example(["i", "drink", "a", "cup", "of", "coffee", "."], {1: "[MASK]", 3: "river", 5: "coffee"})`, met een willekeurig stukje op plek 3 en het stukje zelf op plek 5. Daarna eigen gevallen: geen plan, een plan voor elke plek, een stukje dat twee keer voorkomt, een plan in een andere volgorde geschreven met een stukje dat een woord voortzet, en de 18 stukjes van een zin van de course, `C.EXAMPLE_PIECES`, met het plan `C.EXAMPLE_PLAN`.
        """),
        L.task("Geef terug wat het model op elke plek ziet, en het stukje dat het model op elke gekozen plek moet geven, of None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_example():
    box_example = L.code_box("make_example", lines=11)
    box_example
    return (box_example,)


@app.cell(hide_code=True)
def check_example(box_example):
    result_example = L.check("make_example", box_example.value)
    result_example
    return (result_example,)


@app.cell(hide_code=True)
def hints_example():
    L.hints(
        "make_example",
        [
            "Twee keer `append` voor elke plek `k`: één keer naar `inputs`, één keer naar `labels`. Gebruik `if k in replace:` voor een gekozen plek, en `else:` voor elke andere plek. Op een gekozen plek zegt het plan wat er in `inputs` komt.",
            "Een label is het antwoord dat het model op een gekozen plek moet geven. Dat antwoord is hetzelfde of het plan nu `\"[MASK]\"`, een willekeurig stukje of het stukje zelf neerzet: het model moet het plan terugdraaien. Welk van de twee argumenten weet nog wat er stond?",
            "De regel onder `if k in replace:` die `labels` vult: `            labels.append(___)`",
        ],
        why=r"`labels.append(pieces[k])`: het label is het stukje dat op plek `k` stond, wat het plan daar ook neerzet. Helemaal, onder `for k in range(len(pieces)):`: `if k in replace:` met `inputs.append(replace[k])` en `labels.append(pieces[k])` eronder; dan `else:` met `inputs.append(pieces[k])` en `labels.append(None)`. In het voorbeeld krijgen plek 1 en 5 `\"[MASK]\"` in `inputs`, en *cat* en *mat* in `labels`.",
    )
    return


@app.cell(hide_code=True)
def text_after_example_exercise():
    mo.md(r"""
    In de afbeelding maakte je functie het trainingsvoorbeeld van een zin van de course. BERT maakt voor elke tekst zo'n plan: willekeurig 15% van de plekken, en op elk daarvan [MASK], een willekeurig stukje of het stukje zelf. Volgens de makers van RoBERTa trok de oorspronkelijke BERT de plannen één keer, vóór de training, en kopieerde de data 10 keer, zodat een tekst 10 verschillende plannen kreeg in de 40 keer dat de training door de data ging. RoBERTa trekt een nieuw plan elke keer dat een tekst langskomt, en de makers vonden dat even goed of iets beter werken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keep():
    mo.md(r"""
    ### Oefening 2. Waarom niet altijd [MASK]?
    """)
    return


@app.cell(hide_code=True)
def choice_keep_cell():
    choice_keep = L.choice(
        "keep-some",
        question="Op een gekozen plek zet BERT maar in 80% van de gevallen [MASK]; in 10% zet BERT er een willekeurig stukje neer, en in 10% blijft het stukje staan. Waarom niet op elke gekozen plek [MASK]?",
        options={
            "speed": "Met minder [MASK]s gaat de training sneller",
            "right": "Een tekst die BERT later leest, heeft geen [MASK]",
            "typos": "Van de willekeurige stukjes leert BERT tikfouten in een tekst te verbeteren",
        },
        correct="right",
        why={
            "right": "Een model als BERT wordt verder getraind en daarna gebruikt op gewone teksten, zonder [MASK]. Het paper noemt dat een verschil tussen pretraining en fine-tuning, want bij fine-tuning komt [MASK] niet voor; fine-tuning is een model verder trainen voor één taak, het onderwerp van een latere les, *Een encoder finetunen voor een taak*. Met willekeurige en ongewijzigde stukjes ertussen weet het model niet naar welke plekken gevraagd wordt, en moet het volgens het paper voor elk stukje van de input een beeld bijhouden dat uit de hele zin komt.",
            "speed": "Een stap kost evenveel werk: het model rekent een vector uit voor elke plek, wat daar ook staat, en de loss telt dezelfde gekozen plekken. Denk aan de teksten die BERT krijgt als de training klaar is.",
            "typos": "BERT leert wel het goede stukje te geven waar een willekeurig stukje staat, maar willekeurige stukjes zijn zeldzaam: 1,5% van alle stukjes, 10% van 15%. Volgens het paper lijkt dat het taalbegrip van het model niet te schaden. Denk aan de teksten die BERT krijgt als de training klaar is.",
        },
    )
    choice_keep
    return (choice_keep,)


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. rank: hoe ver omlaag staat het goede stukje?

        Schrijf `rank(scores, right)`. `scores` is een NumPy-array met één score per stukje, zoals de 30.522 scores van BERT-Tiny bij een [MASK], en `right` is het nummer van het goede stukje. Geef de rang ervan terug: de plaats in de lijst van alle stukjes, gerangschikt van de hoogste score omlaag, 1 voor het stukje met de hoogste score. Een stukje met dezelfde score als het goede stukje staat er niet boven.

        Voor `rank(np.array([0.5, 2.0, 1.0, -1.0]), 2)` is de uitkomst 2: 1,0 is de op een na hoogste score. Vervang de `...`. De check roept dat aan en `rank(np.array([0.5, 2.0, 1.0, -1.0]), 1)`, wat 1 moet geven. Daarna eigen gevallen: de laagste score, twee stukjes met dezelfde score bovenaan en twee lager, alleen negatieve scores, en de scores van BERT-Tiny bij de [MASK] van drie zinnen van de course, `C.blank_scores(1)`, `C.blank_scores(6)` en `C.blank_scores(7)`, met het nummer van het verborgen woord: *coffee*, *capital* en *rain*.
        """),
        L.task("Geef de rang van het stukje met nummer `right` terug: 1 plus het aantal stukjes dat erboven staat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rank():
    box_rank = L.code_box("rank", lines=7)
    box_rank
    return (box_rank,)


@app.cell(hide_code=True)
def check_rank(box_rank):
    result_rank = L.check("rank", box_rank.value)
    result_rank
    return (result_rank,)


@app.cell(hide_code=True)
def hints_rank():
    L.hints(
        "rank",
        [
            "Twee regels: `better = np.sum(...)`, het aantal stukjes dat boven het goede stukje staat, en dan `return better + 1`. In `np.sum` komt een vergelijking van de hele array `scores` tegelijk, zoals bij een boolean mask.",
            "Een vergelijking van een array met één getal, zoals `scores > 0`, geeft een array van `True` en `False`, één per stukje, en `np.sum` telt de `True`s. Met welk getal vergelijk je, zodat precies de stukjes boven het goede stukje `True` geven? En welke vergelijking laat een stukje met dezelfde score weg?",
            "De eerste regel: `    better = np.sum(___)`",
        ],
        why=r"`better = np.sum(scores > scores[right])`: `scores[right]` is de score van het goede stukje, en `>` geeft alleen `True` bij een hogere score, dus een stukje met dezelfde score telt niet mee, en het goede stukje zelf ook niet. Dan `return better + 1`. In het voorbeeld is alleen 2,0 hoger dan 1,0, dus de rang is 2.",
    )
    return


@app.cell(hide_code=True)
def text_after_rank():
    mo.md(r"""
    In de afbeelding gaf je functie het verborgen woord van alle acht zinnen van de course een rang. De eerste gok van BERT-Tiny is goed in 1 van de 8, en het verborgen woord zit in 4 ervan bij de eerste vijf gokken. Veel missers zijn woorden van de goede soort: *breakfast* voor *coffee*, *monday* voor *sunday*, *hot* voor *cold*. *rain* krijgt rang 341: de eerste gokken zijn *go*, *do*, *it*, *be* en *move*, en de meeste passen bij de grammatica van *It started to [MASK]*, maar geen ervan past bij de paraplu's. Victor Sanh en anderen geven de gokken van de versie van BERT met 110 miljoen parameters bij *I think this is the beginning of a beautiful [MASK]*: twee stukjes met een hoge kans, *day* en *life*, en een lange staart van andere goede antwoorden, zoals *future*, *story* en *world*. Ook rang 3 kan dus een goed antwoord zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_loss():
    mo.vstack([
        mo.md(r"""
        ### Oefening 4. mlm_loss: de loss op de gekozen plekken

        Schrijf `mlm_loss(scores, places, right)`. `scores` is een tensor met één rij scores per plek van een tekst, zoals de head het uitrekent; `places` is een lijst van de gekozen plekken, en `right` een lijst met het nummer van het goede stukje op elk daarvan, in dezelfde volgorde. Geef de gemiddelde cross-entropy over de gekozen plekken terug, en over geen enkele andere plek: één getal, of een tensor met één getal.

        Voor `mlm_loss(torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 2.0, 0.5]]), [2], [0])` is de uitkomst 1,4644, afgerond: min de log van de kans die softmax in rij 2 aan stukje 0 geeft. Vervang de `...`. De check roept dat aan en dezelfde tabel met `[0, 2]` en `[0, 1]`, wat 0,352 moet geven. Daarna eigen gevallen: dezelfde twee plekken in de andere volgorde, één plek waar het goede stukje de hoogste score heeft, elke plek, een tabel van 4 rijen van 5 met drie plekken, en de scores van BERT-Tiny op elke plek van een zin van de course met twee gaten, `C.week_scores()`, met de plekken 5 en 11 en de nummers van de verborgen woorden.
        """),
        L.task("Geef de gemiddelde cross-entropy over alleen de gekozen plekken terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_loss():
    box_loss = L.code_box("mlm_loss", lines=7)
    box_loss
    return (box_loss,)


@app.cell(hide_code=True)
def check_loss(box_loss):
    result_loss = L.check("mlm_loss", box_loss.value)
    result_loss
    return (result_loss,)


@app.cell(hide_code=True)
def hints_loss():
    L.hints(
        "mlm_loss",
        [
            "Eén regel: `return nn.CrossEntropyLoss()(...)`, zoals in [Project: een MLP dat handgeschreven cijfers leest](/course/open/torch-mlp-digits?lang=nl): eerst een tabel van scores met één rij per voorspelling, dan een tensor met het goede antwoord bij elke rij. Hier zijn de goede antwoorden `torch.tensor(right)`.",
            "`torch.tensor(right)` heeft één getal per gekozen plek, dus de tabel van scores heeft één rij per gekozen plek nodig, in dezelfde volgorde, en geen andere rijen. De tabel met nieuwe woorden laat zien hoe je meerdere rijen van een tensor tegelijk neemt. Welke lijst zegt welke rijen?",
            "De enige regel: `    return nn.CrossEntropyLoss()(___, torch.tensor(right))`",
        ],
        why=r"`return nn.CrossEntropyLoss()(scores[places], torch.tensor(right))`: `scores[places]` neemt de rijen van de gekozen plekken, in de volgorde van `places`, en `nn.CrossEntropyLoss()` geeft het gemiddelde van min de log van de kans op het goede stukje in elke rij. In het voorbeeld geeft softmax van rij 2, `[1.0, 2.0, 0.5]`, stukje 0 een kans van 0,231, en min de log daarvan is 1,4644.",
    )
    return


@app.cell(hide_code=True)
def text_after_loss():
    mo.md(r"""
    In de afbeelding rekende je functie de loss uit van *A week has seven [MASK] and a year has twelve [MASK].* (een week heeft zeven dagen en een jaar twaalf maanden). Op de twee verborgen plekken is de eerste gok van BERT-Tiny beide keren *years*, en de gemiddelde loss is 2,99. Waar BERT-Tiny het stukje ziet, is de loss lager dan 0,09: het antwoord staat gewoon in de input. Over alle 12 plekken is het gemiddelde 0,52, en bijna alles daarvan komt van de twee verborgen plekken. Een loss over elke plek zou vooral overschrijven belonen. Jurafsky en Martin wijzen op de prijs: maar 15% van de stukjes in de trainingsdata telt mee in het leren. ELECTRA, een later model van de familie, leert van elk stukje.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: vul het gat in

    Typ een Engelse zin met een of meer [MASK]s, in hoofdletters en met de vierkante haken. De afbeelding laat bij elke [MASK] de vijf beste stukjes van BERT-Tiny zien, met beide kanten en met alleen de woorden ervoor. De zin hieronder geeft met beide kanten eerst *winter*; met alleen *the* vóór het gat kan BERT-Tiny alleen veelvoorkomende woorden gokken. Probeer een gat waarvan het antwoord uit de woorden erna komt, en een gat waarvan de woorden ervoor het antwoord al geven.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_text = mo.ui.text(value="The [MASK] is very cold in winter.", label="Zin", full_width=True)
    explore_text
    return (explore_text,)


@app.cell(hide_code=True)
def explore_look(bert, explore_text):
    bert   # BERT-Tiny is gedownload: deze cel wacht op de knop hierboven
    mo.Html(C.explore_picture(explore_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Encoders, decoders en de GPT-lijn](/course/open/lm-model-families?lang=nl) keek elk token van een encoder naar alle tokens. Dan is het volgende token raden te makkelijk, dus leert BERT met masked language modeling: het model verbergt 15% van de stukjes, en de head raadt elk ervan uit beide kanten, met als loss de cross-entropy op alleen de gekozen plekken, zoals in [Hoe goed is een language model?](/course/open/lm-nll?lang=nl). De eerste gok van BERT-Tiny is bij 0,303 van de woorden van Tatoeba het verborgen woord, en bij 0,140 met alleen de woorden ervoor. De volgende les van het domein, *Een encoder finetunen voor een taak*, traint zo'n model verder voor één taak.

    **Een voorbeeld uit het dagelijks leven.** Google Search gebruikt BERT sinds oktober 2019, eerst voor Engelse zoekvragen in de Verenigde Staten. Pandu Nayak van Google schreef toen dat BERT de hele context van een woord kan meenemen, door naar de woorden ervoor en erna te kijken, en dat dat vooral helpt om te begrijpen wat iemand met een zoekvraag bedoelt. In *2019 brazil traveler to usa need a visa* beslist het kleine woord *to* wie waarheen reist.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_keep, result_example, result_loss, result_rank):
    L.wrap_up(result_example, choice_keep, result_rank, result_loss)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
