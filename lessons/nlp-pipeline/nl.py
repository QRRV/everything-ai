# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-pipeline. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-pipeline", "nl")
    C = L.checks   # de drie stappen van de course, en vier verzonnen zinnen met wat een mens bij elke stap geeft


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="Een programma leest de mails van een winkel en zet elke mail in een map: klachten, vragen of bestellingen. Welke kant van NLP is dat?",
        options={
            "right": "Begrijpen (NLU): er gaat tekst in, en er komt een label uit",
            "nlg": "Maken (NLG): het geeft de naam van een map terug",
            "both": "Beide kanten, want een mail is tekst en elke map heeft een naam in woorden",
        },
        correct="right",
        why={
            "right": "Het programma haalt iets uit tekst, de soort mail, en maakt zelf geen tekst. Deze les deelt zo'n programma op in stappen.",
            "nlg": "Tekst maken is taal schrijven uit data. De naam van een map is een label dat het programma kiest, geen tekst die het schrijft. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "both": "Er gaat tekst in, maar wat eruit komt is een label, geen tekst die het programma schrijft. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_reply():
    quick_reply = L.choice(
        "quick-reply",
        question="Het programma van de fietsenwinkel in *Wat NLP is* vindt eerst de soort verzoek, en kiest dan een antwoord bij die soort. Wat gebeurt er met het antwoord als het de verkeerde soort vindt?",
        options={
            "right": "Dat gaat ook over iets verkeerds, want het wordt gekozen bij de soort",
            "fixed": "De stap die het antwoord kiest ziet de fout, en kiest het antwoord dat bij het verzoek past",
            "unknown": "Het programma zegt dat het het verzoek niet begreep",
        },
        correct="right",
        why={
            "right": "De stap van het antwoord ziet alleen de soort die binnenkomt. Een verkeerde soort geeft een antwoord over iets verkeerds: een fout in de eerste stap werkt door tot het eind. Daar gaat deze les over.",
            "fixed": "De stap van het antwoord krijgt alleen de soort, niet het verzoek, dus de fout ziet die stap niet. Zie de demo in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "unknown": "Dat zegt het programma alleen bij de soort `\"unknown\"`. Met een verkeerde soort antwoordt het vol vertrouwen over iets verkeerds. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_reply
    return (quick_reply,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een zoekmachine, een spamfilter of een spraakassistent verwerkt tekst zelden in één keer. Het werk wordt opgedeeld in stappen: waar eindigen de zinnen, wat zijn de woorden, welke woorden zijn namen? Elke stap is klein genoeg om los te schrijven en te testen, en elke stap werkt met wat de stap ervoor teruggaf.

    Deze les bouwt zo'n keten van drie stappen, en volgt een fout van de eerste stap tot het eind.

    **Een voorbeeld uit het dagelijks leven.** Aan een lopende band krijgt iedereen wat de persoon ervoor maakte. Een onderdeel dat aan het begin scheef zit, zit scheef in de auto, ook als iedereen erna het werk goed doet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een keten heeft drie stappen. Elke stap doet 90 van elke 100 zinnen goed, zolang de invoer goed is. Als elke fout het eindresultaat fout maakt, hoeveel van 100 zinnen komen er dan ongeveer goed uit?",
        options={
            "most": "Ongeveer 90",
            "right": "Ongeveer 73",
            "half": "Ongeveer 50",
        },
        reveal="**Ongeveer 73.** Van 100 zinnen doet de eerste stap er 90 goed. De tweede stap doet 90 op de 100 daarvan goed, 81, en de derde 90 op de 100 daarvan, ongeveer 73. Deze simpele som is van de course zelf: ze neemt aan dat elke fout het eind haalt. *Fouten lopen door* in *Het idee* vertelt wat er in een echte keten gemeten is.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een keten van stappen

    Een **NLP-pipeline** is een keten van stappen waarin elke stap de uitvoer van de stap ervoor neemt. De documentatie van spaCy, een library die zo werkt, zegt het zonder omwegen: elk onderdeel geeft het bewerkte document terug, en dat gaat dan naar het volgende onderdeel van de pipeline. Een typische keten deelt een tekst op in zinnen, deelt de zinnen op in woorden, zegt wat voor woord elk woord is, en vindt de namen van mensen en plaatsen. Latere stappen kunnen alles gebruiken wat de eerdere stappen vonden.

    De keten van de course hieronder heeft drie stappen, elk met een eenvoudige regel die de course zelf bedacht. Stap 1 begint een nieuwe zin na elke punt, elk vraagteken of elk uitroepteken met een spatie erachter. Stap 2 knipt elke zin bij de spaties, en maakt van een komma aan het eind van een woord, en van het teken aan het eind van de zin, losse woorden. Stap 3 ziet een woord dat met een hoofdletter begint als (een deel van) een naam, maar slaat het eerste woord van elke zin over, omdat elke zin met een hoofdletter begint; woorden met een hoofdletter naast elkaar vormen samen één naam, zoals *Den Haag*.

    ### Drie tools

    Zulke stappen hoef je niet zelf te schrijven: libraries zoals spaCy en Stanza bieden hele ketens, met stappen die getraind zijn op zinnen die mensen met de hand gemarkeerd hebben.

    **spaCy** is een Python-library voor NLP die Matthew Honnibal in 2015 uitbracht, en die het bedrijf Explosion onderhoudt, onder de MIT-licentie. De kleine pipeline voor het Nederlands, `nl_core_news_sm` 3.8.0, is 12 MB en heeft een eigen licentie, CC BY-SA 4.0. Eerst knipt spaCy de tekst in woorden; daarna komen stappen met de namen `tok2vec`, `morphologizer`, `tagger`, `parser`, `lemmatizer`, `attribute_ruler` en `ner`; latere lessen komen op een aantal ervan terug.

    **NLTK**, de Natural Language Toolkit, is ouder. Volgens het boek van NLTK is het in 2001 gemaakt als onderdeel van een cursus computationele linguïstiek aan de University of Pennsylvania. De makers, Edward Loper en Steven Bird, bouwden het om mee les te geven. NLTK is meer een gereedschapskist dan één keten: je zet de stappen zelf in elkaar, en je kunt er teksten en woordenlijsten bij downloaden, ook in het Nederlands. In 2004 viel het onder de GPL; op 21 november 2026 viel het onder de Apache License 2.0.

    Een derde tool is Stanza, van de Stanford NLP Group. Tot versie 1.0.0, in 2020, heette Stanza StanfordNLP. De paper van dat jaar stelt het voor als een open-source toolkit voor NLP in Python, voor 66 talen; op 21 november 2026 noemde de homepage er 70. Deze course schrijft de stappen zelf, zodat je ziet wat er binnenin gebeurt.

    ### Fouten lopen door

    Een stap krijgt alleen wat de stap ervoor teruggaf, en ziet niet wat die stap gemist heeft. Jenny Rose Finkel, Christopher Manning en Andrew Ng (2006) schrijven dat NLP-systemen voor taken van meerdere stappen vaak slechter werken door een greedy pipeline die alleen de beste gok doorgeeft: daardoor lopen fouten bij elke stap door en stapelen ze zich op.

    Ze geven ook een meting, uit werk van Kristina Toutanova (2005): een stap die vindt wie wat met wie doet in een zin, scoorde 91,2 van de 100, op een schaal van 0 tot 100 en dus geen aantal zinnen, met zinsstructuren die mensen met de hand gemaakt hadden, en 80,0 met de structuren die een eerdere stap berekend had. Een score kan zo'n fout zelfs verbergen. De documentatie van spaCy waarschuwt ervoor: knipt de stap die woorden knipt zo dat een naam niet meer past, dan telt die naam niet mee in de score voor namen. Die score kan er dan nog goed uitzien, ook als de helft van de namen niet te vinden is.
    """)
    return


@app.cell(hide_code=True)
def pipeline_cell():
    mo.Html(C.pipeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **NLP-pipeline** | een keten van stappen waarin elke stap de uitvoer van de stap ervoor neemt |
        | **spaCy** | een Python-library voor NLP met getrainde pipelines, ook voor het Nederlands |
        | **NLTK** | de Natural Language Toolkit: een Python-library met tools, teksten en woordenlijsten, gemaakt om mee les te geven |
        | `C.STEPS` | de drie stappen van de course als lijst: `C.split_sentences`, `C.split_words` en `C.find_names` |
        | `C.SENTENCES`, `C.EXPECTED` | vier verzonnen Engelse zinnen, en voor elke zin de uitvoer die een mens bij elke stap geeft; de demo en de afbeeldingen tonen Nederlandse zinnen die bij dezelfde stappen misgaan |
        | `C.OUTPUTS` | voor elke zin de uitvoer van de drie stappen van de course |
        | `step(x)` | als `step` een functie bevat, roept `step(x)` die aan: een lijst kan functies bevatten zoals elke andere waarde |
        """),
        L.watch_out("De som bij de gok is van de course zelf, en te simpel. In een echte keten doen sommige fouten aan het eind geen kwaad, en gaan de stappen vaak op dezelfde moeilijke zinnen mis. Hoeveel een vroege fout kost, verschilt per keten: meet het."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf elke stap als een functie die de uitvoer van de stap ervoor neemt.
    2. Geef de tekst aan de eerste stap, en elke uitvoer aan de volgende stap.
    3. Bewaar de uitvoer van elke stap, niet alleen de laatste.
    4. Vergelijk elke uitvoer met wat een mens bij die stap geeft. De eerste die verschilt, is waar de fout begint.
    5. Verbeter de stap waar de fout begint. De stappen erna kunnen goed zijn, en alleen verkeerde invoer krijgen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel haalt de vierde zin van de course met de hand door de drie stappen, één regel per stap, en print wat elke stap teruggeeft.
    """)
    return


@app.cell
def example_steps():
    text = C.SENTENCES[3]   # twee zinnen
    sentences = C.split_sentences(text)
    words = C.split_words(sentences)
    names = C.find_names(words)
    print(sentences)
    print(words)
    print(names)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke regel geeft de volgende stap wat de regel ervoor terugkreeg: twee zinnen, dan twee lijsten woorden, dan de twee namen. In het Nederlands zou de zin luiden: *Ik zag Anna in Parijs. We dronken thee.* Elke stap doet deze zin goed. Oefening 1 doet hetzelfde met een loop, voor elke lijst stappen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_trace():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. trace: de uitvoer van elke stap

        `C.STEPS` is een lijst met de drie stappen van de course, elk een functie. Schrijf `trace(text, steps)`: geef `text` aan de eerste functie van de lijst `steps`, geef wat die teruggeeft aan de volgende functie, enzovoort, en geef een lijst terug met de uitvoer van elke stap, in volgorde. Begin met een lege lijst `outputs`, en bewaar wat de laatste stap teruggaf in een variabele `result`. Vervang de `...`. De check roept `trace(SENTENCES[0], STEPS)` aan, daarna de andere zinnen en eigen stappen.
        """),
        L.task("Ga de stappen langs, geef elke stap de uitvoer van de stap ervoor, en bewaar elke uitvoer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_trace():
    box_trace = L.code_box("trace", lines=7)
    box_trace
    return (box_trace,)


@app.cell(hide_code=True)
def check_trace(box_trace):
    result_trace = L.check("trace", box_trace.value)
    result_trace
    return (result_trace,)


@app.cell(hide_code=True)
def hints_trace():
    L.hints(
        "trace",
        [
            "Kijk naar het uitgewerkte voorbeeld: elke regel geeft een stap wat de regel ervoor terugkreeg. Een `for`-loop over `steps` doet dat voor elk aantal stappen.",
            "Wat krijgt elke stap: de tekst waarmee je begon, of wat de stap ervoor teruggaf? En wanneer zet je een uitvoer in de lijst?",
            "`        result = ___`",
        ],
        why="`outputs = []` en `result = text`, dan `for step in steps:` met `result = step(result)` en `outputs.append(result)` eronder, en na de loop `return outputs`. `trace(SENTENCES[0], STEPS)` geeft een lijst van drie, en de laatste is `['New York']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_trace():
    mo.md(r"""
    Bij de eerste zin, in de afbeelding *Dr. Jansen woont in Den Haag.*, zijn er na de eerste stap al twee zinnen waar er één is, en mist de laatste uitvoer *Jansen*. Oefening 2 vindt de eerste stap die misgaat.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_first_wrong():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. first_wrong: waar de fout begint

        `C.EXPECTED` bevat voor elke zin de uitvoer die een mens bij elke stap geeft, en `C.OUTPUTS` de uitvoer van de stappen van de course. Schrijf `first_wrong(outputs, expected)`: het nummer van de eerste stap waarvan de uitvoer niet gelijk is aan de verwachte uitvoer, geteld vanaf 0 zoals Python de plekken in een lijst telt, en `None` als elke stap goed is. Ga de nummers van de stappen langs met `for i in range(len(outputs)):`. Vervang de `...`. De check roept `first_wrong(OUTPUTS[0], EXPECTED[0])` aan, daarna de andere zinnen en eigen lijsten.
        """),
        L.task("Ga de stappen in volgorde langs, en geef het nummer terug van de eerste die fout is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first_wrong():
    box_first_wrong = L.code_box("first_wrong", lines=5)
    box_first_wrong
    return (box_first_wrong,)


@app.cell(hide_code=True)
def check_first_wrong(box_first_wrong):
    result_first_wrong = L.check("first_wrong", box_first_wrong.value)
    result_first_wrong
    return (result_first_wrong,)


@app.cell(hide_code=True)
def hints_first_wrong():
    L.hints(
        "first_wrong",
        [
            "Stap 4 van *Recept*: bekijk de uitvoer per stap, vanaf de eerste stap, en stop bij de eerste die fout is.",
            "Wat vergelijk je bij stapnummer `i`? En wanneer mag de functie zeggen dat geen enkele stap fout is?",
            "`        if ___:`",
        ],
        why="`for i in range(len(outputs)):`, dan `if outputs[i] != expected[i]:` met `return i` eronder, en na de loop `return None`. `first_wrong(OUTPUTS[0], EXPECTED[0])` geeft `0`: de fout begint bij de eerste stap.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: volg een zin door de keten

    Kies een zin, en zie wat elke stap teruggeeft. Een vak dat verschilt van wat een mens bij die stap geeft, heeft een paars kader, met eronder wat een mens geeft.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_sentence = mo.ui.dropdown(options=C.sentence_options(L.lang), value=list(C.sentence_options(L.lang))[0], label="Een zin")
    demo_sentence
    return (demo_sentence,)


@app.cell(hide_code=True)
def demo_look(demo_sentence):
    mo.Html(C.demo_picture(demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_fix():
    mo.md(r"""
    ### Oefening 3. Een reparatie in stap 3

    In de eerste zin mist stap 3 *Jansen*. Iemand wil dat verhelpen door stap 3 te veranderen: voortaan ziet die stap ook het eerste woord van een zin als naam, als het met een hoofdletter begint.
    """)
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    choice_fix = L.choice(
        "fix",
        question="Wat doet die verandering met de vier zinnen van de course?",
        options={
            "right": "*Jansen* wordt gevonden, maar ook eerste woorden die geen naam zijn, zoals *We*",
            "only": "Alleen *Jansen* komt erbij, en vanaf dan klopt de keten",
            "nothing": "Er verandert niets, want het woord *Jansen* bereikt stap 3 niet vanuit de stappen ervoor",
        },
        correct="right",
        why={
            "right": "Stap 3 neemt nu elk eerste woord met een hoofdletter, dus vindt stap 3 ook *Ik*, *We* en *Dr*. Wel komt er ook *Anna* bij in de derde zin, waar de fout echt in stap 3 begint. De fout van de eerste zin begon in stap 1, die de zin na *Dr.* afknipte; een stap 1 die weet dat *Dr.* geen einde van een zin is, vindt *Jansen* zonder nieuwe fouten. Stap 5 van *Recept*: verbeter de stap waar de fout begint.",
            "only": "Elke zin begint met een hoofdletter, dus de nieuwe regel neemt ook *Ik*, *We* en *Dr* als naam. Probeer het in je hoofd op de vierde zin. Stap 5 van *Recept* zegt waar je verbetert.",
            "nothing": "Stap 3 krijgt *Jansen* wel, maar als eerste woord van de zin die stap 1 maakte, dus de eigen regel slaat het over. Kijk naar de eerste zin in de demo.",
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_exercise_measure():
    mo.md(r"""
    ### Oefening 4. Een score op schone invoer

    Een team test een stap die namen vindt op zinnen die mensen met de hand in zinnen en woorden opdeelden. De stap scoort 95 van de 100. In de echte keten krijgt de stap wat stap 1 en 2 van de course teruggeven.
    """)
    return


@app.cell(hide_code=True)
def choice_measure_cell():
    choice_measure = L.choice(
        "measure",
        question="Welke score verwacht je in de echte keten?",
        options={
            "right": "Lager, want stap 1 en 2 maken fouten die het opdelen met de hand niet heeft",
            "same": "Dezelfde 95, want de stap zelf is niet veranderd",
            "higher": "Hoger, want drie stappen die samenwerken weten meer dan één stap die alles alleen doet",
        },
        correct="right",
        why={
            "right": "De stap krijgt invoer die in de test nooit voorkwam, zoals een zin die na *Dr.* is afgeknipt of een naam met aanhalingstekens eraan. De stap van Toutanova ging van 91,2 naar 80,0 met de structuren van een eerdere stap. Wil je weten wat een stap in de keten doet, test dan in de keten.",
            "same": "De stap is hetzelfde, maar de invoer niet: een zin die na *Dr.* is afgeknipt, of een woord met aanhalingstekens eraan. Zie *Fouten lopen door* in *Het idee*.",
            "higher": "De latere stap krijgt alleen de uitvoer van de eerdere, fouten en al; niets in de keten herstelt ze. Zie *Fouten lopen door* in *Het idee*.",
        },
    )
    choice_measure
    return (choice_measure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl) vond in één stap waar een verzoek over ging. Deze les deelde het werk op in een keten: een NLP-pipeline, waarin elke stap de uitvoer van de stap ervoor neemt, zoals de stappen van spaCy doen. NLTK biedt de onderdelen voor zulke ketens. Met de uitvoer van elke stap in handen vond je waar een fout begint, en zag je dat een fout in de eerste stap doorwerkt tot het eind, ook als de latere stappen hun regels volgen. De volgende les, *Woorden en hun delen*, kijkt in de woorden die stap 2 eruit knipt.

    **Een voorbeeld uit het dagelijks leven.** Als een recept mislukt, proef je na elke stap, niet alleen aan het eind: dan weet je welke stap je opnieuw moet doen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, choice_measure, result_first_wrong, result_trace):
    L.wrap_up(result_trace, result_first_wrong, choice_fix, choice_measure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
