# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-history. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-history", "nl")
    C = L.checks   # zes verzonnen paren van een Nederlandse en een Engelse zin, en een verzonnen shared task met drie systemen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_readings():
    quick_readings = L.choice(
        "quick-readings",
        question="In *we saw a bat* heeft *saw* drie betekenissen en *bat* twee. Op hoeveel manieren zijn de vier woorden te lezen?",
        options={
            "plus": "5, want de betekenissen tellen op",
            "right": "6, want de betekenissen vermenigvuldigen zich",
            "two": "2, want maar twee woorden hebben meer dan één betekenis",
        },
        correct="right",
        why={
            "right": "Elke betekenis van saw gaat samen met elke betekenis van bat: 3 × 2 = 6. Deze les laat zien hoe NLP overging van zulke keuzes in regels opschrijven naar ze leren uit data.",
            "plus": "Elke betekenis van saw gaat samen met elke betekenis van bat, dus de aantallen vermenigvuldigen zich: 3 × 2 = 6. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
            "two": "Twee woorden hebben meer dan één betekenis, en elke betekenis van het ene gaat samen met elke betekenis van het andere: 3 × 2 = 6. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
        },
    )
    quick_readings
    return (quick_readings,)


@app.cell(hide_code=True)
def quick_check_context():
    quick_context = L.choice(
        "quick-context",
        question="De raadsleden weigeren de boze betogers een vergunning omdat ze geweld vrezen. Waarom wijst *ze* naar de raadsleden?",
        options={
            "grammar": "Grammatica: ze wijst naar het onderwerp van de zin",
            "near": "Plek: ze wijst naar de groep die er in de zin het dichtst vóór staat",
            "right": "Kennis van de wereld: wie geweld vreest, weigert een vergunning",
        },
        correct="right",
        why={
            "right": "Met aanmoedigen in plaats van vrezen wijst ze naar de betogers, en de grammatica blijft dezelfde. Regels over vorm zien dat niet; deze les laat zien wat NLP daarna probeerde.",
            "grammar": "Met aanmoedigen in plaats van vrezen wijst ze naar de betogers, met dezelfde grammatica. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
            "near": "De dichtstbijzijnde groep vóór ze zijn de betogers, en toch zijn de raadsleden bedoeld. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
        },
    )
    quick_context
    return (quick_context,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tegenwoordig vertaalt een app op je telefoon een menukaart terwijl je wacht. In 1954 vertaalde een computer een paar dozijn Russische zinnen in het Engels, met een korte lijst woorden en een handvol regels. Daartussen liggen drie manieren om NLP te bouwen: regels die met de hand geschreven zijn, statistiek uit tellingen in grote hoeveelheden tekst, en neural networks. Een werkwijze maakte elke stap meetbaar: shared tasks, waarin elk systeem op dezelfde test wordt gescoord.

    Deze les loopt die geschiedenis door, laat een programma een vertaling leren door te tellen, en scoort drie systemen op één testset.

    **Een voorbeeld uit het dagelijks leven.** Een schoolexamen is voor alle leerlingen dezelfde opgave, nagekeken met hetzelfde antwoordmodel: alleen dan zijn twee cijfers te vergelijken.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In januari 1954 lieten IBM en Georgetown University in New York zien hoe een computer Russische zinnen in het Engels vertaalde. Hoeveel woorden kende de computer?",
        options={
            "right": "Ongeveer 250",
            "mid": "Ongeveer 2.500",
            "many": "Ongeveer 25.000",
        },
        reveal="**Ongeveer 250.** Volgens John Hutchins (2004) had het systeem een woordenschat van maar 250 lexicale eenheden (stammen en uitgangen) en maar zes regels. *Regels* in *Het idee* vertelt hoe dat werd ontvangen.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Regels

    Op 7 januari 1954 lieten IBM en Georgetown University op het hoofdkantoor van IBM in New York zien hoe een computer Russisch in het Engels vertaalde, met een woordenschat van 250 stammen en uitgangen en zes regels. De meeste van de ruim zestig zinnen gingen over organische chemie, schrijft John Hutchins in een geschiedenis van de demonstratie (2004). Een krant gaf een voorspelling door: over vijf, misschien drie jaar zou vertalen met een elektronisch proces in belangrijke vakgebieden van meerdere talen wel eens een voldongen feit kunnen zijn. Leon Dostert leidde het team van Georgetown; of die woorden van Dostert kwamen, is niet bekend, voegt Hutchins eraan toe. Het leerboek van Jurafsky en Martin, in de versie van 19 augustus 2026, noemt het de eerste openbare demonstratie van een prototype van een MT-systeem, waarbij MT staat voor machine translation, machinevertaling. Hutchins (2004) zag dat anders: het was geen prototype maar een etalage, bedoeld om aandacht en geld te trekken.

    Twaalf jaar later, in november 1966, concludeerde een commissie, ALPAC, in een rapport dat de Amerikaanse *National Academy of Sciences* uitgaf, dat er geen direct of voorspelbaar uitzicht was op bruikbare machinevertaling. Volgens Hutchins (1996) maakte het rapport in de Verenigde Staten voor zo'n twintig jaar een einde aan het grote geld voor onderzoek naar MT.

    Regels leverden wel opvallende programma's op. ELIZA, beschreven door Joseph Weizenbaum in januari 1966, antwoordde als een therapeut, door patronen te zoeken in wat mensen typten en die om te draaien; mensen raakten er emotioneel sterk bij betrokken, schrijven Jurafsky en Martin. SHRDLU van Terry Winograd, beschreven in 1971–72, volgde opdrachten over blokken op een tafel; volgens Winograd (1971) kon je het programma vragen speelgoed te verplaatsen. Maar elk programma werkte in een kleine wereld, en elke regel moest met de hand worden geschreven. Volgens Jurafsky en Martin overheersten symbolische aanpakken in AI, NLP en cognitiewetenschap ruwweg van 1965 tot begin jaren negentig. Symbolisch betekent hier: gebouwd uit regels en symbolen die mensen schreven.

    ### Statistiek

    De periode daarna noemen ze een lange terugkeer van het empirisme, waarin het vakgebied terugging naar statistische en neurale modellen; die begon op verschillende momenten tussen 1975 en 1988 en liep tot 2017. Het begon bij spraakherkenning, met de groep van Fred Jelinek bij IBM tussen 1975 en 1985, en volgens Jurafsky en Martin bedacht die groep de term language model. In 1988 presenteerde dezelfde groep op de conferentie COLING in Boedapest een statistische aanpak van vertalen, en in 1990 beschreven ze die aanpak van machinevertaling in een tijdschrift. Ze leerden uit de debatten van het Canadese parlement, die in het Engels en in het Frans verschijnen; die verslagen heten Hansards. In plaats van een woordenboek dat met de hand geschreven was, leerden ze uit 40.000 zinsparen welke woorden elkaars vertaling zijn. Dat begint met tellen welke woorden samen voorkomen in zinnen die elkaars vertaling zijn.

    Statistiek heeft tekst nodig, en tekst waar de goede antwoorden in staan. De Penn Treebank werd in 1993 beschreven als een corpus van ruim 4,5 miljoen woorden Amerikaans Engels, waarin mensen bij elk woord hadden aangegeven of het een zelfstandig naamwoord was, een werkwoord of iets anders. Een corpus is een grote verzameling tekst, bijeengebracht om te bestuderen.

    ### Neural networks

    In 2003 beschreven Yoshua Bengio en collega's een neural language model in het tijdschrift *Journal of Machine Learning Research* (een eerdere versie stond op de conferentie NIPS van 2000). Ze wilden een bekend probleem te lijf gaan: er zijn veel te veel rijtjes woorden om ze allemaal in de training te zien. Hun oplossing was elk woord een lijst getallen te geven, zodat wat het model van één zin leert, doorwerkt in zinnen met woorden die erop lijken. In 2014 lieten meerdere groepen zien dat een neural network hele zinnen kon vertalen; Jurafsky en Martin noemen Kalchbrenner en Blunsom (2013) de pioniers van die aanpak. Op 12 juni 2017 verscheen de Transformer, een architectuur die alleen op attention is gebouwd, zonder de recurrence en de convoluties van eerdere netwerken. En op 30 november 2022 presenteerde OpenAI ChatGPT als research preview: een model dat in de vorm van een gesprek reageert.
    """)
    return


@app.cell(hide_code=True)
def timeline_cell():
    mo.Html(C.timeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Shared tasks

    Elke stap telde pas als anderen konden zien dat die beter werkte. Een **shared task** is een wedstrijd met vaste data en een vaste score: de organisatoren geven elk team dezelfde trainingsdata en testdata, en scoren elk systeem op dezelfde manier. De organisatoren van de shared task van CoNLL in 2018 beschrijven het zo: elk jaar heeft de conferentie een shared task, waarin de deelnemers hun lerende systemen op dezelfde datasets trainen en testen. De eerste editie van WMT, de *Workshop on Statistical Machine Translation*, gaf in 2006 elk team dezelfde data om op te trainen, om een gemeenschappelijk kader te hebben met resultaten die te vergelijken zijn, en om de drempel om mee te doen te verlagen.

    Een paar bekende voorbeelden:

    - **Senseval** hield in 1998 een eerste ronde, over het kiezen van de betekenis van een woord in een zin. De vierde editie, in 2007, heette SemEval, omdat de taken breder waren geworden. De organisatoren schreven in 2001 dat het er niet in de eerste plaats om ging een winnaar aan te wijzen.
    - De lijst van shared tasks van **CoNLL** begint in 1999. In 2002 ging de taak over het vinden van namen van personen, plaatsen en organisaties in het Spaans en het Nederlands; de Nederlandse data kwamen uit vier edities van de Belgische krant De Morgen uit 2000, en twaalf systemen deden mee. In 2003 deden zestien systemen hetzelfde voor Engels en Duits.
    - **WMT** volgde op een shared task op een workshop in 2005; sinds 2016 heet het de *Conference on Machine Translation*.

    In 2010 schreef Mark Liberman over de nalatenschap van Jelinek. De methode noemde Liberman een wedstrijd tussen algoritmes, beoordeeld met precieze getallen die vooraf vastliggen, op een gedeelde verzameling trainings- en testmateriaal.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | regels, statistiek, neural networks | drie manieren om NLP te bouwen: kennis die met de hand is opgeschreven, geteld in tekst, of geleerd door een netwerk |
        | parallelle tekst | zinnen in de ene taal naast hun vertaling in een andere, zoals de Hansards |
        | **shared task** | een wedstrijd met vaste data en een vaste score, zodat de systemen van alle teams te vergelijken zijn |
        | `C.PAIRS` | zes verzonnen paren van een Nederlandse en een Engelse zin, elk als twee lijsten woorden |
        | `C.TEST`, `C.ANSWERS`, `C.OUTPUTS` | de tien testzinnen van een verzonnen shared task, de goede antwoorden, en de antwoorden van drie systemen |
        | `for a, b in pairs` | haalt elk paar in de lijst uit elkaar in de twee delen |
        | `x in a_list` | `True` als de lijst `x` bevat: `"kat" in ["de", "kat", "slaapt"]` is `True` |
        """),
        L.watch_out("De periodes lopen door elkaar, en hun jaartallen zijn ruw: Jurafsky en Martin schrijven zelf *ruwweg* en *op verschillende momenten*. Regels zijn nooit verdwenen, en woorden tellen wordt nog steeds gebruikt, naast neural networks."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een vertaling te leren door te tellen, zoals de statistiek deed:

    1. Neem paren zinnen die elkaars vertaling zijn.
    2. Houd de paren over waarvan de zin in de ene taal het woord bevat.
    3. Tel de woorden van de andere taal in die paren: het vaakste woord is een kandidaat-vertaling.

    Om systemen te scoren, zoals een shared task doet:

    1. Leg de testdata en de goede antwoorden vast voordat de systemen draaien.
    2. Leg de score vast: hier het deel van de antwoorden dat goed is.
    3. Laat elk systeem op dezelfde testdata draaien, en zet de scores naast elkaar.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel gaat de paren van `C.PAIRS` langs, haalt elk paar uit elkaar in de Nederlandse en de Engelse lijst, en print de paren waarvan de Nederlandse zin *kat* bevat.
    """)
    return


@app.cell
def example_pairs():
    for dutch, english in C.PAIRS:
        if "kat" in dutch:
            print(dutch, english)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Drie paren bevatten *kat*, en *cat* staat in alle drie; *the* staat in twee ervan. Wie telt welk Engels woord het vaakst bij *kat* staat, vindt de vertaling zonder woordenboek.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cooccur():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. cooccur: een vertaling door te tellen

        Schrijf `cooccur(word, pairs)`: een dict van elk Engels woord naar het aantal paren waarin het samen met het Nederlandse `word` voorkomt. Ga de paren langs met `for dutch, english in pairs:`; ga bij elk paar waarvan de Nederlandse lijst `word` bevat de Engelse woorden langs. Vervang de `...`. De check roept `cooccur("kat", PAIRS)` aan, daarna de andere Nederlandse woorden en eigen paren.
        """),
        L.task("Tel de Engelse woorden in de paren waarvan de Nederlandse zin het woord bevat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cooccur():
    box_cooccur = L.code_box("cooccur", lines=10)
    box_cooccur
    return (box_cooccur,)


@app.cell(hide_code=True)
def check_cooccur(box_cooccur):
    result_cooccur = L.check("cooccur", box_cooccur.value)
    result_cooccur
    return (result_cooccur,)


@app.cell(hide_code=True)
def hints_cooccur():
    L.hints(
        "cooccur",
        [
            "Het uitgewerkte voorbeeld vindt de paren; tel nu wat erin staat.",
            "Wat doe je met een woord dat al in `counts` staat, en met een woord dat er nog niet in staat?",
            "`                    counts[w] = ___`",
        ],
        why="`counts = {}`, `for dutch, english in pairs:`, `if word in dutch:`, `for w in english:`, dan `if w in counts:` met `counts[w] = counts[w] + 1`, `else:` met `counts[w] = 1`, en na de loops `return counts`. `cooccur(\"kat\", PAIRS)` geeft `{\"the\": 2, \"cat\": 3, \"sleeps\": 1, \"eats\": 1, \"a\": 1, \"plays\": 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_cooccur():
    mo.md(r"""
    *Cat* wint bij *kat*, en *dog* bij *hond*: geleerd uit zes paren, zonder woordenboek. Maar bij *slaapt* staan *the* en *sleeps* gelijk, omdat *the* in de meeste paren staat. De modellen van IBM deden meer dan tellen om door zulke woorden heen te kijken.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: tel voor andere woorden

    Kies een Nederlands woord, en zie welke Engelse woorden de `cooccur` van de course ernaast vindt.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.dropdown(options=C.word_options(L.lang), value=C.DUTCH[0], label="Een Nederlands woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. score: één test voor iedereen

        `C.TEST` bevat de tien Engelse testzinnen van een verzonnen shared task, en `C.TEST_NL` dezelfde zinnen in het Nederlands: is het gemarkeerde woord een zelfstandig naamwoord of een werkwoord? `C.ANSWERS` bevat de goede antwoorden, en `C.OUTPUTS` is een dict van elk van drie systemen naar de antwoorden van dat systeem. Schrijf `score(predicted, answers)`: het deel van de plekken waar de lijst `predicted` hetzelfde antwoord heeft als de lijst `answers`, een getal tussen 0 en 1. Tel met `for i in range(len(answers)):`, en deel door het aantal antwoorden. Vervang de `...`. De check roept `score(OUTPUTS["rules"], ANSWERS)` aan, daarna de andere systemen en eigen lijsten.
        """),
        L.task("Tel de plekken waar de twee lijsten overeenkomen, en deel door hun lengte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_score():
    box_score = L.code_box("score", lines=6)
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
            "*Recept* zegt wat de score hier is.",
            "Op welke plekken tel je er één bij, en waardoor deel je aan het eind?",
            "`        if ___:`",
        ],
        why="`right = 0`, `for i in range(len(answers)):`, dan `if predicted[i] == answers[i]:` met `right = right + 1` eronder, en na de loop `return right / len(answers)`. `score(OUTPUTS[\"rules\"], ANSWERS)` geeft `0.7`.",
    )
    return


@app.cell(hide_code=True)
def text_after_score():
    mo.md(r"""
    Het systeem dat telt haalt 0,8, dat met regels 0,7, en een systeem dat altijd zelfstandig naamwoord zegt haalt al 0,5. Dat laatste getal doet ertoe: een score zegt pas iets naast wat een heel eenvoudig systeem op dezelfde test haalt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_linguist():
    mo.md(r"""
    ### Oefening 3. Een opmerking over taalkundigen

    Een opmerking die aan Fred Jelinek wordt toegeschreven, wordt vaak zo naverteld: telkens als Jelinek een taalkundige ontsloeg, werd het systeem beter. Volgens Stevenson en Merlo (2022) plaatste Jelinek de opmerking zelf in een voordracht op een workshop in december 1988; de woorden verschillen per navertelling.
    """)
    return


@app.cell(hide_code=True)
def choice_linguist_cell():
    choice_linguist = L.choice(
        "linguist",
        question="Waar staat de opmerking voor in de geschiedenis van NLP?",
        options={
            "right": "Uit data geleerde modellen doen het beter dan regels van experts",
            "wrong": "Taalkunde blijkt ongelijk te hebben over hoe taal werkt",
            "people": "Snellere computers maken de kennis van mensen over taal overbodig",
        },
        correct="right",
        why={
            "right": "De opmerking gaat over systemen: uit tekst geleerde modellen deden het beter dan regels van taalkundigen. Over het gelijk van de taalkunde zegt ze niets, en de data waren vaak door taalkundigen gemarkeerd, zoals in de Penn Treebank.",
            "wrong": "Het gaat over systemen, niet over de wetenschap: met de hand geschreven regels deden het slechter dan uit data geleerde modellen. Zie *Statistiek* in *Het idee*.",
            "people": "De modellen leerden uit tekst die mensen hadden geschreven en vaak gemarkeerd, zoals de Penn Treebank. Wat veranderde, is wie de kennis opschrijft. Zie *Statistiek* in *Het idee*.",
        },
    )
    choice_linguist
    return (choice_linguist,)


@app.cell(hide_code=True)
def text_exercise_compare():
    mo.md(r"""
    ### Oefening 4. Twee scores

    Twee verzonnen teams rapporteren over het vinden van namen van personen, plaatsen en organisaties in Nederlands nieuws. Team A heeft 90 procent goed op nieuws dat het team zelf koos. Team B haalt een score van 80 op de Nederlandse testset van CoNLL-2002.
    """)
    return


@app.cell(hide_code=True)
def choice_compare_cell():
    choice_compare = L.choice(
        "compare",
        question="Welk team heeft het betere systeem?",
        options={
            "a": "Team A, want 90 is meer dan 80",
            "right": "Dat valt niet te zeggen, want de twee getallen komen van andere data",
            "b": "Team B, want de testset van een shared task is moeilijker dan zelfgekozen nieuws",
        },
        correct="right",
        why={
            "right": "Een score betekent pas iets naast scores op dezelfde data met dezelfde maat. Dat legt een shared task vast: WMT wilde een gemeenschappelijk kader, met resultaten die te vergelijken zijn.",
            "a": "Team A koos het eigen nieuws, en dat kan makkelijker zijn. Alleen scores op dezelfde testdata kun je naast elkaar zetten. Zie *Shared tasks* in *Het idee*.",
            "b": "Niets maakt de testset van een shared task moeilijker; wat die geeft, is dezelfde data voor iedereen. Zie *Shared tasks* in *Het idee*.",
        },
    )
    choice_compare
    return (choice_compare,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl) liet zien wat taal moeilijk maakt: ambiguïteit, variatie en context. Deze les liet drie manieren zien waarop NLP daarmee probeerde om te gaan: regels die met de hand geschreven zijn, statistiek uit tellingen in grote hoeveelheden tekst, en neural networks. Shared tasks maakten elke stap meetbaar op dezelfde data. Jouw `cooccur` leerde uit zes zinsparen dat *kat* *cat* is, en jouw `score` vergeleek drie systemen op één testset. De volgende les, *Een NLP-pipeline*, laat zien hoe klassieke NLP stappen aan elkaar rijgt, waarbij elke stap de volgende voedt.

    **Een voorbeeld uit het dagelijks leven.** Een recept dat een kok uit ervaring opschrijft, werkt in de keuken van die kok; een recept dat in honderd keukens is getest en overal op dezelfde manier is beoordeeld, zegt je meer.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_compare, choice_linguist, result_cooccur, result_score):
    L.wrap_up(result_cooccur, result_score, choice_linguist, choice_compare)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
