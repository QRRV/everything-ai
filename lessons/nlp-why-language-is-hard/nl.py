# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-why-language-is-hard. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-why-language-is-hard", "nl")
    C = L.checks   # verzonnen betekenissen van acht woorden, vijf zinnen, spellingvarianten en vijf berichten


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
        question="Een tool vat een lang rapport samen in vijf zinnen. Welke kant van NLP is dat, in de eenvoudige indeling van de course?",
        options={
            "nlu": "NLU, want de tool moet snappen wat er in het rapport staat",
            "right": "Allebei, want er gaat tekst in en er komt tekst uit",
            "nlg": "NLG, want de tool schrijft vijf nieuwe zinnen",
        },
        correct="right",
        why={
            "right": "Tekst erin en tekst eruit: beide kanten. Deze les vraagt waarom de kant van het begrijpen zo moeilijk is.",
            "nlu": "De tool moet snappen wat er in het rapport staat, maar schrijft ook vijf zinnen: tekst erin en tekst eruit is allebei. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "nlg": "Gatt en Krahmer zouden het NLG noemen, want de tool schrijft tekst; in de eenvoudige indeling van de course gaat er ook tekst in, dus het is allebei. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="Wie computationele linguïstiek en NLP uit elkaar houdt: wat wil computationele linguïstiek dan?",
        options={
            "systems": "Systemen die werken voor gebruikers, zoals een spellingapp of een vertaler",
            "replace": "Taalkundigen vervangen door programma's die woorden tellen",
            "right": "Weten hoe taal werkt, met computers als gereedschap",
        },
        correct="right",
        why={
            "right": "Dat is de kant van de wetenschap; NLP is dan de kant van de techniek. Deze les laat zien wat allebei moeilijk maakt: de taal zelf.",
            "systems": "Systemen die werken zijn NLP, als je de twee uit elkaar houdt. Computationele linguïstiek wil weten hoe taal werkt. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "replace": "Computationele linguïstiek gebruikt computers om vragen over taal te beantwoorden; ze vervangt niemand. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    *Teacher strikes idle kids*, een bekende Engelse kop. Lees je de kop één keer, dan laat een staking van leraren kinderen zonder les zitten. Lees je de kop nog eens, dan slaat een leraar kinderen die niets doen. Je koos een lezing zonder het te merken; een programma moet kiezen, en de woorden alleen zeggen niet welke bedoeld is. Het programma met trefwoorden uit de vorige les zag een slot dat niet open wil aan voor een vraag over openingstijden.

    Deze les laat zien waarom taal zulke fouten makkelijk maakt: één vorm kan meer dingen betekenen, één betekenis heeft veel vormen, en wat een zin betekent, hangt af van wie de zin zegt en waar. Ze telt ook hoe snel de keuzes oplopen.

    **Een voorbeeld uit het dagelijks leven.** Spreken we af bij de bank? Voor twee vrienden die altijd op hetzelfde bankje in het park zitten, is dat duidelijk, en voor niemand anders.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Kenneth Church en Ramesh Patil (1982) telden op hoeveel manieren je zinnen met een rij stukjes zoals *in the box* of *of products* kunt lezen, telkens met een stukje meer. Hoeveel lezingen vond hun programma bij negen van zulke stukjes achter elkaar?",
        options={
            "mid": "Ongeveer 500",
            "right": "Ongeveer 5000",
            "many": "Ongeveer 50.000",
        },
        reveal="**Ongeveer 5000**: 4862. Church en Patil schrijven dat hun voorspellingen bleven kloppen tot en met negen voorzetselgroepen, met 4862 manieren om de zin op te bouwen. *De keuzes tellen* in *Het idee* laat zien waar dat getal vandaan komt.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Drie redenen

    Hirschberg en Manning (2015) beschrijven de eerste decennia van het vakgebied: onderzoekers probeerden de woordenschat en de regels van menselijke talen voor computers op te schrijven. Dat bleek moeilijk, omdat menselijke taal varieert, dubbelzinnig is en afhangt van de context. Dat zijn de drie redenen van deze les, in een andere volgorde: ambiguïteit, variatie en context.

    ### Ambiguïteit: één vorm, meer betekenissen

    **Ambiguïteit** is als een woord of een zin meer dan één ding kan betekenen. Een woord kan dat: in het leerboek van Jurafsky en Martin kan het Engelse *bank* een geldinstelling betekenen of een glooiende oever. Het Nederlandse *bank* kan een bank zijn om op te zitten of een bank voor je geld. Een woord kan ook meer dan één rol in een zin hebben. Volgens Hirschberg en Manning kan *star* een hemellichaam zijn of een beroemd persoon, en een zelfstandig naamwoord of een werkwoord. *Teacher strikes idle kids* is hun voorbeeld van een kop met twee lezingen: welke lezing het wordt, hangt af van welke woorden je als zelfstandig naamwoord, werkwoord of bijvoeglijk naamwoord leest, en van de bouw van de zin.

    Krantenkoppen zijn kort, dus vaak dubbelzinnig. Op een forum van eindredacteuren plaatste iemand de kop *Violinist linked to JAL crash blossoms*, en een ander stelde voor zulke koppen crash blossoms te noemen. Eindredacteur John McIntyre vertelde het verhaal, en Ben Zimmer gaf het op 26 augustus 2009 door op het blog *Language Log*.

    Een zin kan ook dubbelzinnig zijn terwijl elk woord duidelijk is. Volgens Jurafsky en Martin is er structurele ambiguïteit als de grammatica een zin op meer dan één manier kan ontleden. Een parse is één manier om een zin uit zijn delen op te bouwen. In de grap van Groucho Marx *One morning I shot an elephant in my pajamas* kan *in my pajamas* bij *elephant* horen of bij *shot*. In het Nederlands kunnen *oude mannen en vrouwen* oude mannen en oude vrouwen zijn, of oude mannen en vrouwen van elke leeftijd.

    ### De keuzes tellen

    Elk woord met twee betekenissen verdubbelt het aantal lezingen. In *Teacher strikes idle kids* hebben drie woorden elk twee rollen: 2 × 2 × 2 = 8 lezingen, waarvan er twee zinnig zijn.

    De bouw van de zin vermenigvuldigt ook. Church en Patil (1982) bekeken zinnen zoals *Put the block in the box on the table*, waarin elke voorzetselgroep, een stukje dat met een voorzetsel begint (*in*, *on*, *by*), op meer dan één plek kan aanhaken. Volgens hen groeit die ambiguïteit combinatorisch met het aantal voorzetselgroepen: twee lezingen bij twee voorzetselgroepen, vijf bij drie, en veertien bij vier. Dat zijn de Catalan-getallen: 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862. Het stuk drukt 469 af in plaats van 429, een drukfout: de formule in hetzelfde stuk geeft 429. Hun parser, een programma dat de parses van een zin vindt, vond precies deze aantallen bij zinnen zoals *It was the number of products of products*, tot en met negen voorzetselgroepen. Dat het de Catalan-getallen zijn, hoorden ze van V. Pratt, schrijven ze; ze vermoeden dat het onder kenners van formele talen bekend is, maar weten niet waar het vandaan komt.
    """)
    return


@app.cell(hide_code=True)
def headline_cell():
    mo.Html(C.headline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Variatie: één betekenis, veel vormen

    Het werkt ook andersom: één betekenis kun je op veel manieren schrijven. Jurafsky en Martin wijzen erop dat elke tekst gemaakt is door bepaalde sprekers of schrijvers, in een bepaald dialect van een bepaalde taal, op een bepaalde tijd en plek, met een bepaald doel. De spelling verschilt: *color* en *colour*, of *e-mail* en *email*. Berichten op Twitter kunnen trekken hebben die sprekers van Afro-Amerikaans Engels vaak gebruiken, schrijven ze, zoals *iont* voor *I don't* en *talmbout* voor *talking about*. Mensen wisselen van taal binnen één bericht: hun voorbeeld *Por primera vez veo a @username actually being hateful! it was beautiful:)* begint in het Spaans en eindigt in het Engels. En het genre maakt verschil: een appje schrijf je anders dan een wet. Voor een programma dat letters vergelijkt, zijn *colour* en *color* twee verschillende woorden.

    ### Context: wie het zegt, en waar

    Wat een zin betekent, kan afhangen van wat een lezer weet over de spreker, de plek en de wereld. Hector Levesque, Ernest Davis en Leora Morgenstern (2012) bouwden daar een test op, de Winograd Schema Challenge, genoemd naar Terry Winograd, van wie in 1972 een eerste versie van dit voorbeeld kwam. De raadsleden weigeren de boze betogers een vergunning omdat ze geweld vrezen: wie vreest geweld? Met *aanmoedigen* in plaats van *vrezen* draait het antwoord om, van de raadsleden naar de betogers. Niets in de grammatica zegt wie *ze* zijn; een lezer weet dat raadsleden die geweld vrezen een vergunning weigeren. En *Kun je het raam opendoen?* is zelden een vraag naar wat iemand kan: in een warme kamer is het een verzoek.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **ambiguïteit** | als een woord of een zin meer dan één ding kan betekenen |
        | variatie | één betekenis in veel vormen: spelling, dialect, taal, genre |
        | context | wat een lezer weet over de spreker, de plek en de wereld |
        | parse | één manier om een zin uit zijn delen op te bouwen |
        | `C.SENSES` | verzonnen betekenissen van acht Engelse woorden: een dict van een woord naar een lijst betekenissen |
        | `C.SENTENCES`, `C.VARIANTS`, `C.MESSAGES` | vijf zinnen als lijsten woorden, spellingvarianten met hun standaardvorm, en vijf berichten |
        | `key in a_dict` | `True` als de dict die key heeft: `"bank" in C.SENSES` is `True` |
        | `total = total * n` | vermenigvuldigt `total` met `n` en bewaart de uitkomst onder dezelfde naam |
        """),
        L.watch_out("De meeste ambiguïteit valt een lezer nooit op: je kiest een betekenis uit de context zonder het te merken. Daardoor mis je die makkelijk als je regels voor een programma schrijft."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Tel de betekenissen of rollen van elk woord.
    2. Vermenigvuldig ze: zoveel lezingen zijn er, nog zonder de bouw van de zin.
    3. Tel de plekken waar elke voorzetselgroep kan aanhaken: met elke groep erbij groeit het aantal sneller.
    4. Breng varianten terug tot één vorm, zodat één betekenis niet als meer woorden telt.
    5. Vraag wat een lezer moet weten dat niet in de woorden staat: wie het zegt, waar, en waarover.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel gaat de woorden van de tweede zin langs, en drukt elk woord af dat een lijst betekenissen heeft in `C.SENSES`, met die lijst.
    """)
    return


@app.cell
def example_senses():
    words = C.SENTENCES[1]   # we saw a bat
    for word in words:
        if word in C.SENSES:
            print(word, C.SENSES[word])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De cel drukt de betekenissen in het Engels af. Twee woorden hebben meer dan één betekenis: *saw* drie (zag, zaagt, een zaag), *bat* twee (een vleermuis, een knuppel). Dus de vier woorden zijn op 3 × 2 = 6 manieren te lezen, en een lezer kiest er meteen een.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_readings():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. readings: op hoeveel manieren te lezen

        Schrijf `readings(words, senses)`: het aantal manieren om de lijst `words` te lezen, één betekenis voor elk woord, met de dict `senses` van een woord naar zijn lijst betekenissen. Een woord dat niet in `senses` staat, heeft één lezing. Begin een telling bij 1, ga de woorden langs met `for`, en vermenigvuldig de telling met het aantal betekenissen van elk woord dat in `senses` staat. Vervang de `...`. De check roept `readings(SENTENCES[0], SENSES)` aan, daarna de andere zinnen en eigen dicts.
        """),
        L.task("Vermenigvuldig de aantallen betekenissen van de woorden in senses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_readings():
    box_readings = L.code_box("readings", lines=6)
    box_readings
    return (box_readings,)


@app.cell(hide_code=True)
def check_readings(box_readings):
    result_readings = L.check("readings", box_readings.value)
    result_readings
    return (result_readings,)


@app.cell(hide_code=True)
def hints_readings():
    L.hints(
        "readings",
        [
            "*De keuzes tellen* in *Het idee* zegt hoe de betekenissen van twee woorden samengaan.",
            "Hoeveel keer zo groot wordt de telling bij een woord met drie betekenissen? En wat gebeurt er bij een woord dat niet in `senses` staat?",
            "`            count = ___`",
        ],
        why="`count = 1`, `for word in words:`, dan `if word in senses:` met `count = count * len(senses[word])` eronder, en na de loop `return count`. `readings(SENTENCES[0], SENSES)` geeft `8`.",
    )
    return


@app.cell(hide_code=True)
def text_after_readings():
    mo.md(r"""
    Acht lezingen voor de kop, zes voor *we saw a bat*, en twaalf voor *a fine light bat*: drie woorden geven al een dozijn keuzes. De woorden alleen zeggen niet welke bedoeld is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_canonical():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. canonical: één vorm voor één betekenis

        `C.VARIANTS` is een dict van een spellingvariant naar zijn standaardvorm, zoals `"colour"` naar `"color"`, en `C.MESSAGES` bevat vijf korte berichten als lijsten woorden. Schrijf `canonical(words, variants)`: een nieuwe lijst met elk woord van `words` in zijn standaardvorm: de waarde onder het woord in `variants` als het woord een key is, en anders het woord zelf. Gebruik `if` en `else`, en `.append` om aan de nieuwe lijst toe te voegen. Vervang de `...`. De check roept `canonical(MESSAGES[0], VARIANTS)` aan, daarna de andere berichten en eigen dicts.
        """),
        L.task("Vervang elke variant door zijn standaardvorm, en laat de andere woorden zoals ze zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_canonical():
    box_canonical = L.code_box("canonical", lines=8)
    box_canonical
    return (box_canonical,)


@app.cell(hide_code=True)
def check_canonical(box_canonical):
    result_canonical = L.check("canonical", box_canonical.value)
    result_canonical
    return (result_canonical,)


@app.cell(hide_code=True)
def hints_canonical():
    L.hints(
        "canonical",
        [
            "Kijk wat er met *thx* en met *for* in het eerste bericht moet gebeuren.",
            "Welk van de twee woorden is een key van `variants`, en wat voeg je voor elk van beide aan de nieuwe lijst toe?",
            "`            out.append(___)`",
        ],
        why="`out = []`, `for word in words:`, dan `if word in variants:` met `out.append(variants[word])` eronder, `else:` met `out.append(word)`, en na de loop `return out`. `canonical(MESSAGES[0], VARIANTS)` geeft `[\"thanks\", \"for\", \"the\", \"color\", \"tips\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_canonical():
    mo.md(r"""
    De eerste twee berichten waren anders geschreven en betekenen hetzelfde; nu ziet een programma dezelfde lijst. Maar de dict kent alleen de varianten die iemand heeft opgeschreven: *thnx* of *colr* glipt erdoor.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: voorzetselgroepen erbij

    Zet voorzetselgroepen zoals *in the box* achter *Put the block*, en zie op hoeveel manieren de zin te lezen is.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_phrases = mo.ui.slider(0, 9, value=2, label="Voorzetselgroepen", show_value=True)
    demo_phrases
    return (demo_phrases,)


@app.cell(hide_code=True)
def demo_look(demo_phrases):
    mo.Html(C.parses_picture(demo_phrases.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_trophy():
    mo.md(r"""
    ### Oefening 3. Een trofee en een koffer

    Levesque, Davis en Morgenstern geven dit paar, hier in het Nederlands: de trofee past niet in de bruine koffer, omdat die te groot is. Wat is te groot? Het antwoord is de trofee. Met *te klein* in plaats van *te groot* is het de koffer.
    """)
    return


@app.cell(hide_code=True)
def choice_trophy_cell():
    choice_trophy = L.choice(
        "trophy",
        question="Wat heeft een programma nodig om beide zinnen goed te krijgen?",
        options={
            "right": "Kennis van de wereld: iets past alleen in iets dat groter is",
            "grammar": "Een regel van de grammatica: *die* wijst naar het onderwerp van de zin",
            "near": "Een regel over de volgorde: *die* wijst naar het dichtstbijzijnde zelfstandig naamwoord ervoor, de koffer",
        },
        correct="right",
        why={
            "right": "De twee zinnen verschillen in één woord, en hun grammatica is dezelfde. Wat het antwoord omdraait, is weten dat grote dingen niet in kleine passen: context, geen vorm.",
            "grammar": "Dan zouden beide zinnen de trofee krijgen, maar met *te klein* is het antwoord de koffer. De grammatica van de twee is dezelfde. Zie *Context* in *Het idee*.",
            "near": "Dan zouden beide de koffer krijgen, maar met *te groot* is het de trofee. De plek van de woorden verandert niet tussen de twee. Zie *Context* in *Het idee*.",
        },
    )
    choice_trophy
    return (choice_trophy,)


@app.cell(hide_code=True)
def text_exercise_blossoms():
    mo.md(r"""
    ### Oefening 4. Een crash blossom

    De kop waaraan crash blossoms hun naam danken: *Violinist linked to JAL crash blossoms*. Het verhaal ging over een violist van wie de carrière opbloeide nadat een ouder in 1985 was omgekomen bij een vliegtuigongeluk van Japan Airlines (JAL).
    """)
    return


@app.cell(hide_code=True)
def choice_blossoms_cell():
    choice_blossoms = L.choice(
        "blossoms",
        question="Wat maakt de kop makkelijk verkeerd te lezen?",
        options={
            "know": "Lezers zouden van het ongeluk moeten weten om te zien welke violist er bedoeld wordt",
            "right": "Blossoms kan een zelfstandig naamwoord zijn, maar de kop bedoelt een werkwoord",
            "spelling": "*Linked* is op een ongewone manier geschreven die een programma niet herkent",
        },
        correct="right",
        why={
            "right": "Als zelfstandig naamwoord zijn *crash blossoms* een soort bloesems van een ongeluk, en wordt de violist daarmee in verband gebracht. Lees je *blossoms* als werkwoord, dan bloeit de carrière op. Het gaat om de rol van één woord, net als in *Teacher strikes idle kids*.",
            "know": "Welke violist bedoeld wordt, is niet waar lezers over struikelen: het gaat erom of *blossoms* een zelfstandig naamwoord is of een werkwoord. Zie *Ambiguïteit* in *Het idee*.",
            "spelling": "Elk woord is gewoon gespeld. Het probleem is dat *blossoms* een zelfstandig naamwoord of een werkwoord kan zijn. Zie *Ambiguïteit* in *Het idee*.",
        },
    )
    choice_blossoms
    return (choice_blossoms,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl) gaf NLP twee kanten, en een programma met trefwoorden dat twee verzoeken fout had. Deze les liet zien waarom zulke fouten makkelijk zijn: één vorm kan meer dingen betekenen (ambiguïteit), één betekenis heeft veel vormen (variatie), en wat een zin betekent, kan afhangen van wat een lezer weet (context). De lezingen vermenigvuldigen zich woord voor woord, en de manieren om voorzetselgroepen aan te haken groeien nog sneller. De volgende les, *Van regels via statistiek naar neural networks*, laat zien hoe NLP dat probeerde op te lossen: eerst met regels die met de hand geschreven waren, daarna door te tellen in grote hoeveelheden tekst.

    **Een voorbeeld uit het dagelijks leven.** Een cryptogram werkt omdat één woord twee dingen kan betekenen: de maker rekent erop dat je eerst de verkeerde kiest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_blossoms, choice_trophy, result_canonical, result_readings):
    L.wrap_up(result_readings, result_canonical, choice_trophy, choice_blossoms)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
