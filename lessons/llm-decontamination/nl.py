# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-decontamination. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-decontamination", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl) en [Dubbele tekst weghalen met MinHash](/course/open/llm-dedup?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="Waarom bevat een test set alleen voorbeelden waar het model niet van leerde?",
        options={
            "fast": "Zodat het trainen minder tijd kost: minder voorbeelden om van te leren",
            "hard": "Zodat de test moeilijker is, want een eerlijke score is een lage",
            "right": "Zodat de score zegt hoe goed het model het doet op nieuwe voorbeelden",
        },
        correct="right",
        why={
            "fast": "Minder voorbeelden om van te leren is een bijzaak. Het gaat om wat de score meet: zie [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).",
            "hard": "Een test set hoeft niet moeilijker te zijn, alleen nieuw voor het model. Zie [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl).",
            "right": "En als testvoorbeelden toch in de training data belanden, meet de score dat niet meer. Deze les vindt ze.",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def quick_check_copy():
    quick_copy = L.choice(
        "quick-copy",
        question="Een testvraag staat op een webpagina, gekopieerd met één andere naam erin. Welke controle vindt hem?",
        options={
            "both": "Zowel een controle op exacte kopieën als de overlap van hun shingles",
            "right": "Alleen de overlap van hun shingles: de meeste reeksen woorden zijn gelijk",
            "none": "Geen van beide: één andere naam maakt er een andere tekst van",
        },
        correct="right",
        why={
            "both": "Door één andere naam verschillen de teksten, dus een controle op exacte kopieën mist hem. Zie [Dubbele tekst weghalen met MinHash](/course/open/llm-dedup?lang=nl).",
            "none": "De reeksen woorden zonder die naam blijven gelijk, dus de twee teksten delen nog bijna al hun shingles. Zie [Dubbele tekst weghalen met MinHash](/course/open/llm-dedup?lang=nl).",
            "right": "Deze les gebruikt hetzelfde idee om testvragen te zoeken in training data, met langere reeksen woorden.",
        },
    )
    quick_copy
    return (quick_copy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een benchmark is alleen iets waard als het model zijn vragen niet gezien heeft. Maar een pretraining corpus is een groot deel van het web, en benchmarks staan op het web: hun bestanden, pagina's die ze bespreken, en de teksten waaruit ze gemaakt zijn. De makers van GPT-3 zochten in hun training data naar elke overlap met de test sets van hun benchmarks, en probeerden die weg te halen. Door een fout in het filter bleef een deel staan, en opnieuw trainen kostte te veel. Daarom maakten ze van elke benchmark een schone versie, zonder de vragen die gelekt konden zijn, en vergeleken ze de scores. Van de vier Wikipedia-benchmarks voor language modelling en de Children's Book Test zat bijna alle tekst in de training data, en die scores rapporteerden ze helemaal niet (Brown en anderen, 2020).

    **Een voorbeeld uit het dagelijks leven.** Een docent die een toets maakt uit de opgaven die de leerlingen thuis maakten, leert weinig van de cijfers: een hoog cijfer kan betekenen dat een leerling de antwoorden onthield, niet dat de leerling de sommen kan.
    """)
    return


@app.cell(hide_code=True)
def predict_seen():
    _g = C.MEASURED["gpt"]
    guess_seen = L.guess(
        "seen",
        question=f"De GPT op tokens van de course uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) trainde 10 keer op de eerste 90% van 2.519 verhalen uit Cosmopedia; de rest was zijn validation set. De course maakte {_g['of']} story-cloze-vragen uit verhalen waarop hij trainde en {_g['of']} uit verhalen die hij nooit zag: het begin van een verhaal, en vier zinnen, waarvan er één echt volgt. Op welke vragen doet hij het volgens jou beter?",
        options={
            "seen": "Veel beter op de verhalen waarop hij trainde",
            "right": "Ongeveer even goed op allebei",
            "new": "Veel beter op de nieuwe verhalen",
        },
        reveal=f"**Ongeveer even goed: {_g['seen']} van de {_g['of']} goed op de verhalen waarop hij trainde, {_g['items']} van de {_g['of']} op de nieuwe** (gokken haalt er ongeveer 1 op de 4). Een model van 1,3 miljoen parameters dat op 4,5 MB aan verhalen trainde, onthoudt daar woord voor woord bijna niets van. Grote modellen onthouden veel meer: hoe meer parameters een model heeft, en hoe vaker een tekst in zijn training data stond, hoe meer het ervan kan herhalen (Carlini en anderen, 2023). De course liet zijn GPT de zin kiezen waarvan de kans het meest stijgt als het begin van het verhaal ervoor staat; een latere les laat zien hoe een model opties een score geeft.",
    )
    guess_seen
    return (guess_seen,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Contaminatie** is als testvragen, of hun antwoorden, in de training data van een model zitten: dan meet een testscore geheugen, geen vaardigheid. Het kan een hele vraag met zijn antwoord zijn, of alleen de tekst waaruit een vraag gemaakt is. Dat is een kwestie van mate: de controle van GPT-3 markeerde meer dan 90% van de vragen van drie leestests, maar bij elke overlap die de makers bekeken, zat alleen de tekst waar de vragen over gaan in de training data, niet de vragen en antwoorden.

        **Vinden met n-grams.** Knip elke testvraag in reeksen van $n$ woorden, zoals de shingles van [Dubbele tekst weghalen met MinHash](/course/open/llm-dedup?lang=nl), in kleine letters, maar nu ook zonder de leestekens aan beide kanten van elk woord, zodat een vraag die in een andere opmaak gekopieerd is nog dezelfde reeksen heeft. Een vraag korter dan $n$ woorden telt als één geheel. Markeer dan een vraag als meer dan een bepaald deel van zijn n-grams in de training data voorkomt. GPT-3 markeerde een vraag als één van zijn 13-grams dat deed, of, bij benchmarks met korte vragen, één van zijn kortere reeksen.

        **De grootte doet ertoe.** Een kleine $n$ markeert gewone zinsdelen: "once upon a time in a small town" staat in veel verhalen, en zegt niets over een lek. Een grote $n$ mist een kopie waarin om de paar woorden een woord veranderd is. Geen controle met n-grams vindt een vraag die in andere woorden herschreven is, of vertaald.

        **Voor en na het trainen.** Haal voor het trainen elk document uit de training data dat een n-gram deelt met een testvraag: **decontaminatie**. Na het trainen, als opnieuw trainen te veel kost, markeer je de vragen die n-grams delen met de training data, en rapporteer je de score op de **clean subset**, de vragen die niet gemarkeerd zijn, naast de score op alle vragen. Is de score op de clean subset veel lager, dan hebben de gemarkeerde vragen de score misschien opgedreven.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **contaminatie** | | als testvragen, of hun antwoorden, in de training data van een model zitten, zodat een testscore geheugen meet in plaats van vaardigheid | |
        | **decontaminatie** | | voor het trainen alles uit de training data halen dat een n-gram deelt met een testvraag | |
        | **clean subset** | | de vragen van een benchmark die geen n-gram delen met de training data; hun score, naast de score op alle vragen, laat zien wat contaminatie deed | `report(flags, right)` |
        | **tekens strippen** | | `w.strip(chars)` haalt elk teken van `chars` van beide kanten van `w` af, zoals `strip()` spaties weghaalt | `"home!".strip(".!")` geeft `"home"` |
        """),
        mo.md(r"""
        Het beeld: de 500 story-cloze-vragen van de course uit verhalen die geen model van de course zag, tegen de 6.189 verhalen van TinyStories van de course. De twee zijn los van elkaar gemaakt, door verschillende modellen, en geen vraag is naar TinyStories gekopieerd.
        """),
        mo.Html(C.flagged_picture([(f"n = {_n}", _m["natural"][_n]) for _n in C.SIZES],
                                  {"title": "Gemarkeerde vragen, per n", "of": "van de {total}", "lang": L.lang,
                                   "note": "gemarkeerd: één n-gram staat ook in TinyStories",
                                   "alt": f"Balken met gemarkeerde vragen: met n = 5: {_m['natural'][5]} van de 500; met n = 8: {_m['natural'][8]}; met n = 13: {_m['natural'][13]}. Een vraag wordt gemarkeerd als één van zijn n-grams in TinyStories staat."})),
        mo.md(r"""
        Met $n = 5$ lijken 423 van de 500 vragen besmet; ongeveer de helft daarvan begint met "once upon a time, in", net als 638 van de verhalen van TinyStories. Met $n = 13$ geen enkele: deze false positives zijn weg.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de vragen van een benchmark uit de training data van een model te houden:

    1. Knip elke testvraag in n-grams van zijn woorden: in kleine letters, zonder leestekens aan hun uiteinden; een vraag korter dan $n$ woorden als één geheel.
    2. Voor het trainen: haal elk document uit de training data dat een n-gram deelt met een testvraag.
    3. Na het trainen: markeer de vragen met meer dan een bepaald deel van hun n-grams in de training data, en rapporteer de score op alle vragen en op de clean subset.
    4. Kijk naar wat gemarkeerd is: een gewoon zinsdeel is geen lek, een gekopieerde vraag wel.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 en de controle van stap 3 voor één korte vraag en één document, met reeksen van 4 woorden, met de hand.
    """)
    return


@app.cell
def example_overlap():
    _question = "Where did Tom hide the old key? Under the red flower pot."
    _document = "Tom looked around. He hid the old key under the red flower pot, by the door!"
    _marks = ".,!?"
    _sets = []
    for _text in (_question, _document):
        # 1. de woorden in kleine letters, zonder leestekens aan hun uiteinden
        _words = []
        for _w in _text.lower().split():
            _words.append(_w.strip(_marks))
        # elke reeks van 4 woorden, als set
        _runs = set()
        for _i in range(len(_words) - 3):
            _runs.add(" ".join(_words[_i:_i + 4]))
        _sets.append(_runs)
    _shared = _sets[0] & _sets[1]
    print(len(_shared), "van de", len(_sets[0]), "reeksen van de vraag staan in het document:", sorted(_shared))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vraag heeft 9 reeksen van 4 woorden, en 5 daarvan staan in het document, van "the old key under" tot "the red flower pot": zonder leestekens leest "key? Under" in de vraag als "key under", zoals in het document. De andere 4 reeksen bevatten "hide", waar het document "hid" zegt: één ander woord neemt tot 4 reeksen weg. Zonder `strip` zouden "key?" en "pot." andere woorden zijn dan "key" en "pot,", en zou maar 1 reeks overeenkomen. Het deel is 5 van de 9, ongeveer 0,56, dus deze vraag wordt gemarkeerd bij elke `share` daaronder. Met $n = 13$ is hij te kort voor één 13-gram, en telt hij als één geheel, dat het document niet bevat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ngrams():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. N-grams van een vraag

        Maak `ngrams(text, n)` af: stap 1 van het recept. Neem de woorden van `text` in kleine letters, haal de tekens van `PUNCT` van beide kanten van elk woord af, en houd een woord alleen als er iets van over is. Een tekst met minder dan `n` van zulke woorden geeft de hele tekst als één n-gram, zijn woorden met spaties ertussen, of geen n-grams als hij geen woorden heeft. Anders geeft hij de set van elke reeks van `n` woorden achter elkaar, elk als één tekst met een spatie tussen de woorden. De check roept `ngrams("The dog barked at the post-man, \"Go home!\"", 4)` aan, en andere.
        """),
        L.task("Laat `ngrams` elke reeks van `n` opgeschoonde woorden geven, en een korte tekst als één geheel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ngrams():
    box_ngrams = L.code_box("ngrams", lines=16)
    box_ngrams
    return (box_ngrams,)


@app.cell(hide_code=True)
def check_ngrams(box_ngrams):
    result_ngrams = L.check("ngrams", box_ngrams.value)
    result_ngrams
    return (result_ngrams,)


@app.cell(hide_code=True)
def hints_ngrams():
    L.hints(
        "ngrams",
        [
            "Het uitgewerkte voorbeeld doet het eerste deel en de reeksen. Een woord zoals een los streepje wordt leeg na `strip`: voeg het alleen toe `if w:`.",
            "Kijk vóór de reeksen of `len(words) < n`. Dan is de hele tekst één n-gram, `{\" \".join(words)}`, tenzij `words` leeg is.",
        ],
        why="Zonder hoofdletters en leestekens heeft een vraag die met een andere opmaak gekopieerd is dezelfde n-grams. Een korte vraag heeft geen reeks van n woorden, dus zonder de hele tekst als één n-gram kon hij nooit gemarkeerd worden; GPT-3 deed hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Markeer de vragen

        Maak `flag(test_grams, train_grams, share)` af: de controle van stap 3. `test_grams` is een lijst met de set n-grams van elke testvraag, en `train_grams` de set van alle n-grams van de training data. Hij geeft een lijst terug met `True` of `False` voor elke vraag: `True` als meer dan `share` van de n-grams van de vraag in `train_grams` zit, en `False` voor een vraag zonder n-grams. Met `share` 0 is één n-gram genoeg, zoals bij GPT-3. De check roept `flag([{"a b", "b c", "c d"}, {"x y"}, set()], {"b c", "q r"}, 0)` aan, en andere.
        """),
        L.task("Laat `flag` elke vraag markeren met meer dan `share` van zijn n-grams in de training data."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flag():
    box_flag = L.code_box("flag", lines=12)
    box_flag
    return (box_flag,)


@app.cell(hide_code=True)
def check_flag(box_flag):
    result_flag = L.check("flag", box_flag.value)
    result_flag
    return (result_flag,)


@app.cell(hide_code=True)
def hints_flag():
    L.hints(
        "flag",
        [
            "`grams & train_grams` is de set van de n-grams van de vraag die in de training data zitten. Zijn lengte, gedeeld door `len(grams)`, is het deel.",
            "Kijk eerst `if not grams:`: een vraag zonder n-grams zou door 0 delen.",
        ],
        why="Met een deel kies je hoeveel overlap meetelt: GPT-3 markeerde op één gedeeld n-gram, wat veel vragen markeert die alleen een zinsdeel delen; een groter deel markeert alleen vragen die grotendeels in de training data staan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De score op de clean subset

        Maak `report(flags, right)` af: het rapport van stap 3. `flags` zegt voor elke vraag of hij gemarkeerd is, `right` of het model hem goed had. Hij geeft drie getallen terug: het deel goed van alle vragen; het deel goed van de clean subset, de vragen die niet gemarkeerd zijn, of `None` als die er niet zijn; en het deel van de vragen dat gemarkeerd is. De check roept `report` aan met 8 vragen, waarvan er 2 gemarkeerd zijn, en andere.
        """),
        L.task("Laat `report` de score op alle vragen geven, op de clean subset, en het deel dat gemarkeerd is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=12)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Verzamel `right[i]` voor elke `i` waar `flags[i]` `False` is: dat is de clean subset.",
            "`sum` telt `True` als 1, dus `sum(clean) / len(clean)` is het deel goed; kijk eerst `if clean:`.",
        ],
        why="De makers van GPT-3 deden precies dit voor elke benchmark: is de score op de clean subset ongeveer gelijk aan die op alle vragen, dan dreef het lek de score nauwelijks op; is hij lager, dan misschien wel, of zijn de schone vragen gewoon moeilijker.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Wat je functies vinden

    De demo laat je drie functies los op de story-cloze-vragen van de course, het begin van elk verhaal met de zin die echt volgt. Eerst tegen de verhalen van TinyStories van de course, voor n = 5, 8 en 13. Dan tegen een crawl die de course met opzet besmet: de {C.number(len(C.stories()), 0, L.lang)} verhalen van TinyStories, met de verhalen van {C.PLANTED} van de vragen ertussen gekopieerd, zoals een pagina die ze overnam zou doen. Een look-up model, zoals in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl), leert die crawl: het onthoudt elke tekst, zoekt de laatste {C.KEY} woorden van het begin van een vraag op, en kiest de zin die daar volgt, of gokt als het niets vindt. Het duurt ongeveer tien seconden.
    """)
    return


@app.cell(hide_code=True)
def demo_natural(box_ngrams, box_flag, result_ngrams, result_flag):
    mo.stop(not (result_ngrams.state == "correct" and result_flag.state == "correct"),
            mo.md("*Maak eerst oefening 1 en 2 af: dit deel gebruikt je `ngrams` en je `flag`.*"))
    _out = C.natural(run_code(box_ngrams.value).names["ngrams"], run_code(box_flag.value).names["flag"])
    _rows = []
    for _n in C.SIZES:
        _ex = _out[_n]["example"]
        _rows.append(f"| {_n} | {C.number(_out[_n]['flagged'], 0, L.lang)} | {'geen' if _ex is None else '“' + _ex + '”'} |")
    mo.md("Je functies op de vragen tegen TinyStories, met `share` 0:\n\n| n | gemarkeerde vragen, van de 500 | het n-gram dat ze het vaakst delen |\n|---|---|---|\n" + "\n".join(_rows) +
          "\n\nGeen vraag is naar TinyStories gekopieerd, dus elke markering hier is een false positive: een zinsdeel dat veel verhalen gebruiken.")
    return


@app.cell(hide_code=True)
def demo_crawl(box_ngrams, box_flag, box_report, result_ngrams, result_flag, result_report):
    _ready = result_ngrams.state == "correct" and result_flag.state == "correct" and result_report.state == "correct"
    mo.stop(not _ready, mo.md("*Maak eerst oefening 1 tot en met 3 af: dit deel gebruikt je drie functies.*"))
    _fns = {_name: run_code(_box.value).names[_name] for _box, _name in ((box_ngrams, "ngrams"), (box_flag, "flag"), (box_report, "report"))}
    _r = C.contaminated(_fns["ngrams"], _fns["flag"], _fns["report"])
    _score, _clean, _share = _r["report"]
    _removed = "; ".join(f"n = {_n}: {C.number(_r['removed'][_n], 0, L.lang)}" for _n in C.SIZES)
    mo.md(f"""Met n = {C.N} markeert je `flag` {_r['flagged']} vragen: de {_r['planted_flagged']} die gekopieerd zijn, en {_r['flagged'] - _r['planted_flagged']} andere, die een reeks van 13 woorden delen met een gekopieerde, meestal aan het begin van het verhaal. Verhalen uit Cosmopedia beginnen vaak op dezelfde manier, zoals "once upon a time in a small town named Harmonyville lived two best friends".

Het look-up model dat de crawl leerde, heeft {C.number(100 * _score, 1, L.lang)}% van alle vragen goed, en {C.number(100 * _clean, 1, L.lang)}% van de clean subset: ongeveer 1 op de 4, wat gokken haalt. Het kent de gekopieerde vragen uit het hoofd, en verder niets, en dat drijft zijn score op alle vragen op. Je `report` zegt dat {C.number(100 * _share, 1, L.lang)}% van de vragen gemarkeerd is.

De crawl decontamineren, door elk document weg te halen dat een n-gram deelt met een vraag, haalt zoveel van zijn {C.number(_r['docs'], 0, L.lang)} documenten weg: {_removed}. Met n = 13 zijn dat precies de {_r['planted']} kopieën; met n = 5 gaat het grootste deel van TinyStories weg, om zinsdelen zoals "once upon a time there". Na decontaminatie met n = 13 heeft het look-up model {C.number(100 * _r['after'], 1, L.lang)}% goed: zijn score meet nu wat het kan.""")
    return


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Oefening 4. Gevonden na het trainen
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after-training",
        question="Na het trainen ontdek je dat een deel van een benchmark in de training data zat, en opnieuw trainen kost te veel. Wat kun je nog doen?",
        options={
            "remove": "Die vragen nu uit de training data halen",
            "nothing": "Niets: de score van het model op die benchmark is helemaal niet meer te vertrouwen",
            "right": "De score op de clean subset rapporteren naast de score op alle vragen",
        },
        correct="right",
        why={
            "remove": "Het model heeft er al van geleerd: ze nu weghalen verandert niets aan zijn parameters. Dat helpt alleen voor het trainen.",
            "nothing": "De clean subset geeft nog steeds een score op vragen die niet gelekt zijn, en de vergelijking met de score op alle vragen laat zien of het lek hem misschien opdreef.",
            "right": "Dat deden de makers van GPT-3. Waar de twee verschilden, keken ze beter, en ze markeerden twee uitkomsten, PIQA en Winograd, als mogelijk besmet. Waar bijna een hele benchmark in de training data zat, rapporteerden ze geen score.",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De test set van [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl), nu apart gehouden van een heel corpus; de shingles van [Dubbele tekst weghalen met MinHash](/course/open/llm-dedup?lang=nl), nu als n-grams van testvragen; de vraag uit [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl) of een benchmark in de training data zat, nu iets wat je kunt controleren. De volgende les zet de opgeschoonde verhalen om in een bestand met token-ids om op te trainen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_ngrams, result_flag, result_report, choice_after):
    L.wrap_up(result_ngrams, result_flag, result_report, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
