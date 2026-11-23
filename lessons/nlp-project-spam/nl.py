# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-project-spam. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("nlp-project-spam", "nl")
    C = L.checks   # de berichten, het filter van de course en naive Bayes, de demo's en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tf_idf():
    quick_tf_idf = L.choice(
        "quick-tf-idf",
        question="Een woord staat twee keer in een bericht, en in 10 van de 1.000 trainingsberichten. Wat is de TF-IDF-waarde, met de logaritme met grondtal 10?",
        options={
            "idf": "2",
            "right": "4",
            "no-log": "200",
            "upside": "0,02",
        },
        correct="right",
        why={
            "idf": "Dat is de inverse document frequency alleen, $\\log_{10}(1000/10) = 2$. De TF-IDF-waarde vermenigvuldigt dat getal met hoe vaak het woord in het bericht staat: $2 \\times 2 = 4$. Zie [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl), waar de rijen uit TF-IDF-waarden bestonden.",
            "right": "Het aantal, 2, keer $\\log_{10}(1000/10) = 2$. In dit project krijgt elk bericht zo'n waarde voor elk woord dat erin staat.",
            "no-log": "Dat is $2 \\times 1000/10$, zonder de logaritme. De inverse document frequency is $\\log_{10}(1000/10) = 2$, dus de waarde is $2 \\times 2 = 4$.",
            "upside": "Dat deelt andersom, $10/1000$, en laat de logaritme weg. De inverse document frequency is $\\log_{10}(1000/10) = 2$, dus de waarde is $2 \\times 2 = 4$.",
        },
    )
    quick_tf_idf
    return (quick_tf_idf,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="Een weight is 0,4, en de helling is 0,2. Wat is de weight na één stap gradient descent met learning rate 1,0?",
        options={
            "plus": "0,6",
            "right": "0,2",
            "times": "0,08",
            "same": "0,4",
        },
        correct="right",
        why={
            "plus": "Dat telt de helling op. Gradient descent gaat tegen de helling in, bergaf: $0{,}4 - 1{,}0 \\times 0{,}2 = 0{,}2$. Zie de demo van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "right": "$0{,}4 - 1{,}0 \\times 0{,}2 = 0{,}2$: de weight beweegt tegen de helling in, met de learning rate keer de helling. In dit project schrijf je die stap zelf.",
            "times": "Dat vermenigvuldigt de weight met de helling. Een stap trekt de learning rate keer de helling af: $0{,}4 - 1{,}0 \\times 0{,}2 = 0{,}2$. Zie de demo van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).",
            "same": "Alleen een helling van 0 laat een weight staan. Hier trekt de stap $1{,}0 \\times 0{,}2$ af, dus de weight wordt 0,2.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) trainde je logistische regressie op 2.400 zinnen uit reviews, met de code van de course voor de training. In dit project bouw je stukje bij beetje een eigen tekstclassifier, en de eerste taak is spam filteren: echte sms'jes, in 2011 tot een verzameling samengebracht, met echte spam ertussen. De functies die je schrijft, blijven in je project. De volgende stap, *Project: sentiment in het Engels en het Nederlands*, gebruikt de functies opnieuw op reviews.

        Een spamfilter maakt twee soorten fouten. Het kan spam doorlaten, en het kan een bericht tegenhouden dat geen spam is. De eerste fout kost een blik op je telefoon. De tweede kan veel meer kosten: een bericht van een vriend, een dokter of een bank dat nooit aankomt. Eén getal voor hoe vaak een filter goed zit, zegt niet welke fouten het maakt, dus je project telt beide soorten apart.

        **Een voorbeeld uit het dagelijks leven.** Iemand die de post van een kantoor sorteert, gooit de reclame weg. Een folder van een pizzeria die op een bureau belandt, kost een collega een blik; een brief van de Belastingdienst die in de prullenbak belandt, kan een boete kosten. Dus wie sorteert, gooit een brief alleen weg als de brief zeker reclame is, en laat een paar folders door. Dat is alleen een vergelijking; hieronder tel je wat je filter doet.
        """),
        draw.project_steps(["Je spamfilter", "Je sentimentclassifier"], 0,
                           here_label="← vandaag", title="Waar je project staat: text"),
    ])
    return


@app.cell(hide_code=True)
def predict_through():
    guess_through = L.guess(
        "through",
        question="Een filter dat niets tegenhoudt, laat elk bericht door. Bij hoeveel van de 1.034 testberichten van dit project zit het goed, denk je?",
        options={
            "spam": "Zo'n 13%",
            "half": "Zo'n 50%",
            "right": "Zo'n 87%",
            "all": "Bijna 100%",
        },
        reveal="**Zo'n 87%: 895 van de 1.034.** Maar 139 van de testberichten zijn spam, dus een filter dat niets doet, zit goed bij alle andere. Hoe vaak een filter goed zit, zegt weinig als één label zeldzaam is. Je project telt de spam die het doorlaat en de berichten die het tegenhoudt, elk apart.",
    )
    guess_through
    return (guess_through,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **De berichten.** De SMS Spam Collection, in 2011 samengesteld door Tiago A. Almeida en José María Gómez Hidalgo, bevat 5.574 sms'jes in het Engels, elk gelabeld als spam of niet. De 747 spamberichten komen van Grumbletext, een Brits forum waar mensen klaagden over de spam die binnenkwam, en uit een eerdere verzameling van Gómez Hidalgo. De andere 4.827 komen ook uit die verzameling, van vrijwilligers aan de National University of Singapore, vooral studenten, en uit een proefschrift over sms'jes van Caroline Tagg. De verzameling mag gebruikt worden onder de licentie CC BY 4.0, maar de berichten die geen spam zijn, zijn echte persoonlijke berichten, met voornamen en soms een telefoonnummer, dus de course host de set niet: in het uitgewerkte voorbeeld download je de set één keer van de UCI Machine Learning Repository. De les laat alleen berichten zien die de course koos, met lange nummers vervangen door [nummer].

    **Kopieën, en de splitsing.** 403 regels van het bestand zijn een kopie van een eerdere regel: dezelfde spam naar veel mensen gestuurd, of een kort bericht als *Sorry, I'll call later*, dat er 30 keer in staat. Als één kopie bij de trainingsberichten belandt en een andere bij de testberichten, heeft het filter dat testbericht al gezien, en laat de test het filter beter lijken dan het is. Dus je project splitst de 5.171 verschillende teksten: elke vijfde is een testbericht. Dat geeft 4.137 trainingsberichten, waarvan 514 spam, en 1.034 testberichten, waarvan 139 spam. Teksten die alleen in cijfers, tekens of hoofdletters verschillen, tellen als verschillend, maar hebben dezelfde woorden: 24 testberichten, waarvan 19 spam, hebben precies dezelfde woorden als een trainingsbericht, meestal dezelfde spam in een iets andere vorm. Zulke spam komt op echte telefoons ook terug, dus de course laat die berichten staan.

    **Je filter.** Het is de classifier van [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl): een rij met TF-IDF-waarden voor elk bericht, en één weight voor elk woord. Twee dingen zijn nieuw. De *document frequencies* komen alleen uit de trainingsberichten, want de testberichten staan voor berichten die later binnenkomen. En een woord van een testbericht dat in geen enkel trainingsbericht staat, krijgt geen kolom: het filter leerde er geen weight voor. 493 van de 1.034 testberichten hebben zo'n woord. Voor de training is elke weight 0 en krijgt elk bericht $p = 0{,}5$; elke stap gradient descent beweegt elke weight en de bias tegen de eigen helling in, zoals in [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl).

    **Twee soorten fouten.** Het filter geeft elk bericht een kans op spam, $p$, en houdt het tegen als $p$ op de **threshold** ligt of erboven. In [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) was dat 0,5. Een **doorgelaten spambericht** is spam die het filter doorlaat; een **onterecht tegengehouden bericht** is een bericht dat geen spam is en dat het filter tegenhoudt. Een hogere threshold houdt minder berichten tegen, dus minder onterecht, en laat meer spam door; een lagere threshold doet het omgekeerde. Waar de threshold hoort, hangt af van wat elke fout kost.

    **Iets om mee te vergelijken.** *Spam of niet: naive Bayes* deelde berichten in door te tellen hoe vaak elk woord in spam staat en in andere berichten. Je project traint naive Bayes op dezelfde trainingsberichten, met dezelfde woorden, en vergelijkt.

    De afbeelding volgt één testbericht door het filter, met de weights na de training, zoals je het hieronder zelf traint.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        *voicemail* staat in geen van de 4.137 trainingsberichten, dus het woord krijgt geen kolom en telt niet mee. *call* draagt het meest bij, 0,992, en *please* en *new* dragen ook wat bij, maar de bias van −3,344 houdt $z$ onder 0: $p$ is 0,253, onder 0,5, dus het filter laat deze spam door. Hetzelfde bericht met *message* in plaats van *voicemail*, een woord uit 54 trainingsberichten, krijgt 0,409.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **threshold** | de laagste kans op spam waarbij het filter een bericht tegenhoudt | `threshold` |
        | **doorgelaten spambericht** | een spambericht dat het filter doorlaat | `missed` |
        | **onterecht tegengehouden bericht** | een bericht dat geen spam is, en dat het filter tegenhoudt | `blocked` |
        """),
        L.watch_out("Zoek naar kopieën voordat je een dataset splitst. Had je project elke vijfde regel van het bestand bij de testberichten gezet, dan stonden 120 van die 1.114 testberichten ook bij de trainingsberichten, met precies dezelfde tekst."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Je spamfilter:

    1. Download de berichten, en splits de verschillende teksten: elke vijfde is een testbericht.
    2. Tel de *document frequencies* op de trainingsberichten. Dat is `document_frequencies`.
    3. Maak de rij met TF-IDF-waarden van elk bericht, zonder kolom voor een woord dat niet in de trainingsberichten staat. Dat is `features`.
    4. Geef elk woord weight 0 en de bias 0, en zet 200 stappen gradient descent met learning rate 1,0. Dat is `step`; `train` roept `step` aan.
    5. Tel bij een threshold op de testberichten de doorgelaten spam en de onterecht tegengehouden berichten. Dat is `mistakes`.
    6. Kies de threshold op basis van wat elke fout kost, en vergelijk met naive Bayes.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel hieronder volgt stap 1 en 2. `sms` is de module `everything_ai.sms` van de course; de cel onder de knop geeft de module zodra de berichten gedownload zijn. `sms.training()` geeft de trainingsberichten als paren van een lijst woorden en een label, 1 voor spam en 0 voor geen spam, en `sms.test()` geeft de testberichten. De woorden staan in kleine letters, zoals in [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl); cijfers en tekens vallen weg. De cel telt de spam bij de trainingsberichten, en daarna de *document frequencies*: van elk bericht elk verschillend woord één keer, met `set(words)`. Ten slotte print de cel de *document frequency* van vijf woorden. Voer de cel uit.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download de berichten, 0,2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def sms_cell(get_button):
    sms = C.sms_or_stop(get_button, {
        "ask": "*De SMS Spam Collection staat nog niet op je computer. De knop downloadt de set één keer van de UCI Machine Learning Repository, 0,2 MB. Het uitgewerkte voorbeeld en de demo's hebben de set nodig; de oefeningen werken zonder. De course host de set niet: de berichten die geen spam zijn, zijn echte persoonlijke berichten.*",
        "busy": "**De berichten worden gedownload** · {done} van {total} kB",
        "no_net": "*Het downloaden lukte niet: de UCI Machine Learning Repository was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam, is niet het vastgelegde bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 0,2 MB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
        "here": "*De SMS Spam Collection staat op je computer: 5.574 sms'jes, van Tiago A. Almeida en José María Gómez Hidalgo, onder CC BY 4.0.*",
    })
    return (sms,)


@app.cell
def example_counts(sms):
    # 1. de trainingsberichten en de testberichten: (woorden, label), label 1 voor spam
    _training = sms.training()
    _test = sms.test()
    _spam = 0
    for _words, _label in _training:
        _spam = _spam + _label
    print(len(_training), "trainingsberichten, waarvan", _spam, "spam;", len(_test), "testberichten")

    # 2. de document frequencies: in hoeveel trainingsberichten elk woord staat
    _df = {}
    for _words, _label in _training:
        for _word in set(_words):
            if _word not in _df:
                _df[_word] = 0
            _df[_word] = _df[_word] + 1
    print(len(_df), "verschillende woorden")
    for _word in ["call", "free", "i", "p", "lt"]:
        print(_word, _df[_word])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    6.957 verschillende woorden: je filter krijgt 6.957 weights. *call* staat in 367 trainingsberichten, waarvan 225 spam, en *i* in 1.289, waarvan maar 34 spam. Twee woorden vallen op. *p* staat in 121 berichten, waarvan 114 spam: spam schrijft prijzen als *150p*, 150 pence, en zonder de cijfers blijft er een *p* over. *lt* staat in 181 berichten, en geen daarvan is spam: in veel berichten die geen spam zijn, staat `&lt;#&gt;` op de plek van een weggehaald getal, zoals een webpagina `<#>` schrijft, en de woorden daarvan zijn *lt* en *gt*. Een filter leert alles wat de labels in de data uit elkaar houdt, ook sporen van hoe de data is samengesteld. Oefening 1 maakt van de telling van stap 2 en de rijen van stap 3 functies van je project.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. features: de rij van een bericht

        `features(words, df, n)` krijgt de woorden van één bericht, de *document frequencies* `df` van de trainingsberichten, en het aantal trainingsberichten `n`. De functie geeft de rij van het bericht als dictionary: voor elk woord van het bericht dat `df` kent, hoe vaak het in het bericht staat keer de inverse document frequency, met de logaritme met grondtal 10, zoals in *TF-IDF: woorden wegen*. Een woord dat `df` niet kent, krijgt geen kolom. Het vak heeft al `document_frequencies`, de loop van het uitgewerkte voorbeeld als functie, en het begin van `features`.

        `small_df`, de `SMALL_DF` van de course, bevat de *document frequencies* van acht kleine berichten van de course, zoals *win a free prize call now*, en `small_words`, de `SMALL_WORDS` van de course, is het bericht *free prize free call tonight*. *free* staat in 4 van de 8 kleine berichten, *prize* en *call* in 3, en *tonight* in geen enkel bericht. Dus `features(small_words, small_df, 8)` geeft `{'free': 0.602, 'prize': 0.426, 'call': 0.426}`: *free* twee keer $\log_{10}(8/4)$, de andere één keer $\log_{10}(8/3)$, en geen kolom voor *tonight*. Vervang de `...`. De check roept dat aan, en dan eigen gevallen: een bericht met alleen woorden die `df` niet kent, drie woorden met *document frequencies* van de echte trainingsberichten, en een bericht zonder woorden. Als het klopt, bewaart de check beide functies in je project, in `work/projects/text/features.py`.
        """),
        L.task("Geef elk woord van het bericht dat `df` kent de TF-IDF-waarde, en sla de andere over."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    box_features = L.code_box("features", lines=24)
    box_features
    return (box_features,)


@app.cell(hide_code=True)
def check_features(box_features):
    result_features = L.check("features", box_features.value)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def hints_features():
    L.hints(
        "features",
        [
            "Stap 3 van het recept. Begin `row` als lege dictionary en loop over `words`. Sla een woord over dat niet in `df` staat. De eerste keer dat een woord langskomt, geef je `row[word]` de waarde `0.0`; elke keer tel je er iets bij op.",
            "Elke keer dat een woord langskomt, krijgt de rij er één inverse document frequency van dat woord bij, dus een woord dat twee keer in het bericht staat, eindigt met twee keer die waarde. Welke functie van NumPy neemt de logaritme met grondtal 10, en van welke twee getallen gedeeld?",
            "`            row[word] = row[word] + ___`",
        ],
        why="`row = {}`; `for word in words:` met `if word in df:`, daaronder `if word not in row:` met `row[word] = 0.0`, en `row[word] = row[word] + np.log10(n / df[word])`; na de loop `return row`. Met de echte *document frequencies* krijgt *call* twee keer $2 \\times \\log_{10}(4137/367) = 2{,}104$.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. step: één stap training

        `step(examples, weights, bias, rate)` zet één stap gradient descent, zoals de code van de course in [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) deed. `examples` zijn rijen met TF-IDF-waarden en de labels. `gradients`, uit het bestand `work/projects/text/logistic.py` dat de course in je project zette, geeft `grad`, de helling van elke weight, en `grad_bias`, de helling van de bias. Elke weight en de bias zetten één stap tegen de eigen helling in, met `rate` als learning rate. Het vak heeft al de eerste regels van `step`, en `train`, die elk woord van de voorbeelden weight 0 geeft en de bias 0, en je `step` `steps` keer aanroept.

        `small_examples`, de `SMALL_EXAMPLES` van de course, bevat drie kleine rijen met TF-IDF-waarden van de woorden *free*, *prize*, *call* en *txt*, waarvan twee spam, en `start_weights` geeft die vier woorden weight 0. Met elke weight 0 krijgt elk voorbeeld $p = 0{,}5$. *prize* staat alleen in de twee spamvoorbeelden en krijgt de helling $-0{,}142$, dus de nieuwe weight is $0 - 1{,}0 \times (-0{,}142) = 0{,}142$. `step(small_examples, start_weights, 0.0, 1.0)` geeft `({'free': 0.050, 'prize': 0.142, 'call': -0.071, 'txt': 0.100}, 0.167)`. Vervang de `...`. De check roept dat aan, en dan eigen gevallen: andere weights, een andere bias en een andere learning rate, en een weight voor een woord dat geen enkel voorbeeld heeft. De check bewaart `step` en `train` in `work/projects/text/train.py`.
        """),
        L.task("Maak de nieuwe weights en de nieuwe bias van één stap gradient descent."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("step", lines=24)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "step",
        [
            "`gradients` gaf al `grad`, met een helling voor elk woord, en `grad_bias`. Loop `for word in weights:` en geef `new_weights[word]` de nieuwe waarde van elke weight; maak na de loop `new_bias`.",
            "Een stap gaat tegen de helling in, bergaf: een weight met een negatieve helling gaat omhoog. Hoe groot de stap is, bepaalt `rate`. Wat trek je af van `weights[word]`?",
            "`        new_weights[word] = weights[word] - ___`",
        ],
        why="`for word in weights:` met `new_weights[word] = weights[word] - rate * grad[word]`, en na de loop `new_bias = bias - rate * grad_bias`. In het tweede geval, met learning rate 0,5 vanaf de kleine weights en bias −0,6, gaat *prize* van 2,0 naar 2,036.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train je filter

    De knop traint je filter op de 4.137 trainingsberichten. Je `document_frequencies` en `features` maken de rijen, en je `step` zet 200 stappen met learning rate 1,0, vanaf weight 0 voor elk woord en bias 0. Voor een bestand dat nog niet in je project staat, zoals `train.py` voordat oefening 2 klopt, gebruikt de course de eigen versie. De grafiek laat de gemiddelde log loss van de trainingsberichten zien na elke stap. Ernaast zie je de vijf woorden met de sterkste weight naar elke kant, drie testberichten met de kans op spam, en bij hoeveel testberichten het filter bij 0,5 een fout maakt. Het duurt ongeveer tien seconden. Daarna kun je terug door de stappen.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train 200 stappen")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(demo_button, result_features, result_step, sms):
    _r = (result_features, result_step, sms)   # zodat deze cel na elke check opnieuw draait
    trained = None
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om te trainen.*")
    else:
        _words = C.live_words(L.lang)
        _now = {}

        def _progress(_step, _loss, _ws, _b, _test_rows):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                      C.STEPS, low=0.0, high=0.7)
            _now["live"].watch(lambda: C.live_parts(_test_rows, _ws, _b, L.lang))
            _now["live"].add(_step, _loss, _words["progress"].format(s=_step, n=C.STEPS, loss=L.number(_loss, 3)))

        mo.output.replace(mo.md("*De rijen van de berichten worden gemaakt …*"))
        _run = C.demo(_progress)
        if isinstance(_run, str):
            _note = C.broken(_run, L.lang)
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            trained = _run
            _m, _bl = C.rewind_counts(_run, C.STEPS)
            _who = {(): "de functies van de course", ("features",): "je `features` en de `step` van de course",
                    ("train",): "de `features` van de course en je `step`"}.get(
                tuple(m for m in _run["fns"].yours if m != "judge"), "je `features` en je `step`")
            _out = mo.vstack([
                mo.md(f"""Na {C.STEPS} stappen met {_who} is de gemiddelde log loss van de trainingsberichten {L.number(_run["curve"][-1][1], 3)}. Bij 0,5 laat het filter <span class="ea-num">{_m}</span> van de 139 spamberichten onder de testberichten door, en houdt het <span class="ea-num">{_bl}</span> van de 895 andere tegen."""),
                L.explore("rewind", "stap", lambda _s: (C.rewind_frame(_run, int(_s), L.lang),
                                                        "Na stap {s}: {m} van de 139 spamberichten doorgelaten, {b} van de 895 andere tegengehouden.".format(
                                                            s=int(_s), m=C.rewind_counts(_run, int(_s))[0], b=C.rewind_counts(_run, int(_s))[1])),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Ga terug naar stap 1, en dan weer vooruit: wanneer komt *Please CALL [nummer] immediately* boven 0,5, en welke woorden worden het eerst sterk?"),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met elke weight 0 krijgt elk bericht $p = 0{,}5$, en is de gemiddelde log loss $\log 2 = 0{,}693$; na 200 stappen is de log loss 0,031, en heeft het filter 4.114 van de 4.137 trainingsberichten goed. Bij 0,5 laat het 15 van de 139 spamberichten onder de testberichten door, en houdt het geen van de 895 andere tegen. De sterkste woorden richting spam zijn *call* (0,94), *p* (0,93), *txt* (0,73), *free* (0,71), *text* (0,71) en *stop* (0,66); richting geen spam *i* (−0,81), *my* (−0,50), *gt* en *lt* (allebei −0,42), *me* (−0,41) en *ok* (−0,38). *p* en *lt* zijn de opvallende woorden van het uitgewerkte voorbeeld. De bias eindigt op −3,344, lager dan het deel spam alleen zou geven: met 514 spamberichten op 4.137 zou dat deel elk bericht laten beginnen bij $p = 0{,}124$, en de bias laat het beginnen bij 0,034. Spamberichten hebben veel woorden met grote weights, en die woorden trekken een spambericht over 0,5 heen, dus het filter kan elk bericht ver van spam laten beginnen.

    Van de drie testberichten eindigt *Please CALL [nummer] immediately as there is an urgent message waiting for you* op 0,712 en wordt het tegengehouden. *You have 1 new voicemail. Please call [nummer]*, het bericht van de afbeelding, eindigt op 0,253 en komt erdoor. *Hey now am free you can call me.* is geen spam: het heeft *free* en *call*, maar de woorden tellen samen maar 1,29 op, tegen de bias van −3,344, dus het eindigt op 0,114.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. mistakes: de twee soorten fouten

        `mistakes(examples, weights, bias, threshold)` telt, bij rijen met de labels, de spamberichten die het filter doorlaat, `missed`, en de berichten die geen spam zijn en die het filter tegenhoudt, `blocked`. Het filter houdt een bericht tegen als de kans op spam, uit de `probability` van de course in je `logistic.py`, `threshold` is of meer. Het vak heeft al de loop en `p`.

        `more_examples`, de `MORE_EXAMPLES` van de course, bevat de drie kleine voorbeelden en nog drie, en `small_weights` geeft *free* 1,2, *prize* 2,0, *call* −0,8 en *txt* 1,5. Met bias −0,6 krijgen de zes voorbeelden de kansen 0,726, 0,359, 0,760, 0,281, 0,660 en 0,380; het tweede en het vijfde zijn geen spam. Bij 0,5 komen het vierde en het zesde, allebei spam, erdoor, en wordt het vijfde tegengehouden: `mistakes(more_examples, small_weights, -0.6, 0.5)` geeft `(2, 1)`. Vervang de `...`. De check roept dat aan, en dan eigen gevallen: de thresholds 0,3 en 0,8, en twee berichten met precies de threshold als kans. De check bewaart je functie in `work/projects/text/judge.py`.
        """),
        L.task("Tel de doorgelaten spam en de tegengehouden andere berichten, bij de threshold `threshold`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_mistakes = L.code_box("mistakes", lines=14)
    box_mistakes
    return (box_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_mistakes):
    result_mistakes = L.check("mistakes", box_mistakes.value)
    result_mistakes
    return (result_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "mistakes",
        [
            "Twee `if`-regels, elk met het label, `p` en `threshold` erin: spam is `label == 1`, een bericht dat geen spam is `label == 0`. Tel onder elke regel 1 op bij `missed` of bij `blocked`.",
            "Spam komt erdoor als `p` onder de threshold ligt. Een bericht dat geen spam is, wordt tegengehouden als `p` de threshold is of meer, ook als `p` precies de threshold is. Welke vergelijking is dat?",
            "`        if label == 0 and p ___ threshold:`",
        ],
        why="`if label == 1 and p < threshold:` met `missed = missed + 1` eronder, en `if label == 0 and p >= threshold:` met `blocked = blocked + 1`. Bij 0,3 wordt ook het tweede voorbeeld, met 0,359, tegengehouden, en van de spam komt alleen het vierde erdoor: `(1, 2)`.",
    )
    return


@app.cell(hide_code=True)
def text_project_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je project heeft nu alle drie de delen: de rijen, de training en de telling van de fouten. De schuif verplaatst de threshold van het filter dat je hierboven trainde, van 0,05 tot 0,95. De afbeelding laat elk testbericht zien als een stip bij de kans op spam: spam op de bovenste rij, de andere berichten op de onderste. Eronder staan de drie testberichten van de training en het testbericht dat geen spam is met de hoogste kans. De zin geeft wat je `mistakes` telt bij die threshold, naast naive Bayes, getraind op dezelfde trainingsberichten.
    """)
    return


@app.cell(hide_code=True)
def project_demo(result_mistakes, trained):
    _r = result_mistakes   # zodat deze cel opnieuw draait na de check van oefening 3
    if trained is None:
        _out = mo.md("*Train eerst je filter hierboven: de schuif verplaatst de threshold ervan.*")
    else:
        _out = L.explore("threshold", "threshold", lambda _t: C.threshold_frame(trained, float(_t), L.lang),
                         start=0.5, values=C.THRESHOLDS, role="",
                         task="Zoek de laagste threshold waarbij je filter geen van de 895 testberichten tegenhoudt die geen spam zijn. Hoeveel spam komt daar door?")
    _out
    return


@app.cell(hide_code=True)
def text_after_project_demo():
    mo.vstack([
        mo.md(r"""
        Bij 0,5 laat je filter 15 spamberichten door en houdt het er geen tegen; naive Bayes laat er 10 door en houdt er 5 tegen. Allebei hebben 1.019 van de 1.034 testberichten goed, met andere fouten. Verlaag de threshold, en je filter vangt meer spam: bij 0,35 laat het er 11 door en houdt het nog steeds niets tegen. Bij 0,3 laat het er 10 door en houdt het er één tegen, *Please protect yourself from e-threats. SIB never asks for sensitive information…*: een waarschuwing van een bank, die een filter moet doorlaten. Bij 0,1 laat het er 2 door, maar houdt het er 16 tegen. Naive Bayes houdt onder meer *Just taste fish curry :-P* tegen: de smiley geeft het woord *p*, dat in 114 spamberichten bij de trainingsberichten staat en in 7 andere.
        """),
        L.watch_out("Hier koos je een threshold door naar de testberichten te kijken, dus de getallen van die berichten laten je filter nu beter lijken dan het is: de threshold past bij deze 1.034 berichten. Wil je er zuiver een kiezen, zet dan een deel van de trainingsberichten apart om de threshold op te kiezen, en kijk pas aan het eind één keer naar de testberichten."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_phone():
    mo.md(r"""
    ### Oefening 4. Welk filter op een telefoon?
    """)
    return


@app.cell(hide_code=True)
def choice_phone_cell():
    choice_phone = L.choice(
        "phone",
        question="Bij 0,5 laat je filter 15 van de 139 spamberichten onder de testberichten door, en houdt het geen van de 895 andere tegen. Naive Bayes laat er 10 door en houdt er 5 tegen. Allebei hebben er 1.019 goed. Welk filter zet je op een telefoon, voor de sms'jes?",
        options={
            "nb": "Naive Bayes, want er komt minder spam door",
            "right": "Dat van jou, want een tegengehouden bericht mis je",
            "same": "Maakt niet uit: allebei hebben er 1.019 goed",
            "neither": "Geen van beide: een filter op een telefoon moet alle spam vangen",
        },
        correct="right",
        why={
            "nb": "Vijf spamberichten minder kosten vijf blikken minder. Maar naive Bayes betaalt dat met vijf tegengehouden berichten die geen spam zijn, zoals *Just taste fish curry :-P*, die de ontvanger misschien nooit ziet. Op een telefoon is dat de duurste fout.",
            "right": "Een spambericht dat erdoor komt, kost een blik; een tegengehouden bericht van een vriend, een dokter of een bank kan veel meer kosten, en niemand merkt dat het ontbreekt. Dus voor een telefoon wint het filter dat niets onterecht tegenhoudt, met een paar spamberichten meer als prijs. De schuif liet zien dat een lagere threshold meer spam vangt, en dat 0,3 al een waarschuwing van een bank tegenhoudt.",
            "same": "Hetzelfde aantal goed verbergt andere fouten: 15 doorgelaten spamberichten tegenover 10 doorgelaten en 5 andere tegengehouden. Wat elke fout kost, beslist, en op een telefoon kost een tegengehouden bericht meer.",
            "neither": "Geen filter hier vangt alle spam zonder andere berichten tegen te houden: bij 0,05 laat je filter nog één spambericht door, en houdt het er 53 andere tegen. Een filter weegt de twee soorten fouten af; het haalt de fouten niet weg.",
        },
    )
    choice_phone
    return (choice_phone,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Logistische regressie op woorden](/course/open/nlp-logistic-text?lang=nl) gaf je de classifier, de hellingen en de stappen van de training. Hier werd het een eigen filter, op echte berichten, met wat een project eromheen nodig heeft: een splitsing zonder kopieën, *document frequencies* die alleen op de trainingsberichten geteld zijn, eigen code voor de training, en een telling van de twee soorten fouten. *Spam of niet: naive Bayes* gaf iets om mee te vergelijken, en hoe vaak een filter goed zit, bleek weinig te zeggen als spam zeldzaam is. Je project `text` heeft nu `logistic.py`, `features.py`, `train.py` en `judge.py`. Het komt terug in *Project: sentiment in het Engels en het Nederlands*, waar dezelfde functies reviews indelen. De volgende les, *Tools en corpora voor het Nederlands*, gaat over de tools en de verzamelingen tekst die er voor het Nederlands zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_phone, result_features, result_mistakes, result_step):
    L.wrap_up(result_features, result_step, result_mistakes, choice_phone)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
