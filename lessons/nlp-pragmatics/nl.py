# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-pragmatics. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-pragmatics", "nl")
    C = L.checks   # de schalen en de wereld van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_some():
    quick_some = L.choice(
        "quick-some",
        question="Is *er slaapt een kat* in de logica waar in een wereld waarin elke kat slaapt?",
        options={
            "right": "Ja: er slaapt minstens één kat",
            "not-every": "Nee: *een kat* betekent niet elke kat",
            "no-value": "Nee: de zin heeft dan geen waarheidswaarde",
        },
        correct="right",
        why={
            "right": "In de logica betekent *een kat* hier minstens één, en als elke kat slaapt, slaapt er minstens één. Deze les kijkt waarom een luisteraar er toch *niet elke kat* in hoort.",
            "not-every": "Dat horen veel luisteraars, maar het is niet wat de logica zegt: *er slaapt een kat* is waar als er minstens één kat bij de slapers hoort, ook als ze er allemaal bij horen. Zie *Elke en een* in [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
            "no-value": "In een wereld met katten is elke zin van die les waar of onwaar. Hier hoort minstens één kat bij de slapers, dus de zin is waar. Zie *Elke en een* in [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
        },
    )
    quick_some
    return (quick_some,)


@app.cell(hide_code=True)
def quick_check_vp():
    quick_vp = L.choice(
        "quick-vp",
        question="Waar staat de VP *achtervolgt Tom* voor in de wereld van *Betekenis: wanneer is een zin waar?*?",
        options={
            "right": "De dieren die Tom achtervolgen",
            "pairs": "De paren van *achtervolgt* met Tom eerst",
            "truth": "Waar als Tom iemand achtervolgt, en anders onwaar",
        },
        correct="right",
        why={
            "right": "Een werkwoord met het lijdend voorwerp staat voor een lijst dingen, de doeners: hier alleen Rex. Een zin met die VP is waar als het onderwerp in de lijst staat. Deze les houdt die wereld, en vraagt wat een spreker bedoelt voorbij die wereld.",
            "pairs": "Het werkwoord *achtervolgt* alleen staat voor paren. Met het lijdend voorwerp erbij staat de VP voor een lijst dingen: de dieren die Tom achtervolgen, met Tom als tweede in hun paar. Zie *Het geheel uit de delen* in [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
            "truth": "Waar of onwaar geeft pas een hele zin, met het onderwerp erbij. De VP alleen staat voor een lijst dingen: de dieren die Tom achtervolgen. Zie *Het geheel uit de delen* in [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl).",
        },
    )
    quick_vp
    return (quick_vp,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    *Betekenis: wanneer is een zin waar?* controleerde wat een zin zegt tegen een wereld. Mensen bedoelen meer dan dat. De filosofen Kepa Korta en John Perry beginnen er hun bijdrage over pragmatiek in de Stanford Encyclopedia of Philosophy mee: wat iemand overbrengt, is meer dan wat er letterlijk gezegd wordt, en wat iemand bedoelt, is meer dan de gewone, vaste betekenis van de gebruikte woorden.

    Jurafsky en Martin laten zien wat dat van een programma vraagt. Een reisagent vraagt op welke dag in mei de klant wil reizen, en de klant antwoordt: *ik moet er zijn voor een vergadering van de 12e tot en met de 15e*. Jurafsky en Martin merken op dat de klant de vraag eigenlijk niet beantwoordt. Jurafsky en Martin leggen uit hoe de reisagent toch kan afleiden dat het een vlucht op de 11e wordt. Een programma dat alleen nakijkt wat de woorden zeggen, heeft geen datum om te boeken.

    **Een voorbeeld uit het dagelijks leven.** Je houdt een lege kop omhoog en vraagt *is er nog koffie?* Letterlijk is dat een vraag over koffie, en *ja* zou een antwoord zijn. Wie *ja* zegt en niets doet, heeft de woorden begrepen en gemist wat je bedoelde.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Laura Ruis en collega's (2023) testten large language models op 600 korte gesprekjes zoals dit. Esther vraagt *Kun je vrijdag naar mijn feestje komen?* en Juan antwoordt *Ik moet werken*. Bedoelt Juan ja of nee? Wie willekeurig gokt, heeft er 50 van de 100 goed. Hoeveel van de 100 hadden modellen die alleen op grote hoeveelheden tekst getraind zijn, zonder extra training om instructies te volgen en zonder voorbeelden in de prompt, ongeveer goed?",
        options={
            "right": "Ongeveer 60 van de 100",
            "people": "Ongeveer 85 van de 100: dicht bij mensen",
            "more": "Ongeveer 95 van de 100: meer dan mensen",
        },
        reveal="**Ongeveer 60 van de 100**: volgens de auteurs kwamen al die modellen zonder voorbeelden dicht bij gokken uit, rond 60 procent, en mensen op 86 procent; gemiddeld 86,2. Die modellen telden als goed als ze het antwoord dat klopt een hogere kans gaven dan het antwoord dat niet klopt: een ruimere manier van tellen dan soms gebruikt wordt, zoals de auteurs zelf zeggen. Het beste model dat ze testten, GPT-4, moest het antwoord zelf schrijven, omdat ze de kansen van dat model niet konden zien: het had er 82,3 van de 100 goed met één voorbeeld in de prompt, en 86,5 met vijf voorbeelden in de prompt die stap voor stap redeneerden.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Zeggen is doen

    De zinnen van *Betekenis: wanneer is een zin waar?* beschreven een wereld, en waren waar of onwaar. De filosoof J. L. Austin (1962) wees op uitingen die in plaats daarvan iets doen. Wie *ik noem dit schip de Queen Elizabeth* zegt en intussen een fles tegen de achtersteven slaat, beschrijft geen doop: onder de juiste omstandigheden is het zeggen van de woorden de doop zelf. Austin noemde zulke uitingen performatief, en schreef over de voorbeelden dat geen ervan waar of onwaar is. In dezelfde colleges wilde Austin, volgens de lezing van de Stanford Encyclopedia of Philosophy, juist laten zien dat er geen eenvoudige grens tussen beschrijven en doen te trekken is. Jurafsky en Martin noemen het idee erachter een sleutelinzicht, dat oorspronkelijk van de filosoof Ludwig Wittgenstein (1953) komt en door Austin is uitgewerkt: elke uiting in een gesprek is een soort handeling die de spreker verricht. Zo'n handeling is een **taalhandeling**: iets beweren, vragen, verzoeken, beloven, wedden.

    Een handeling kan ook via een andere lopen. *Kun je het zout aangeven?* vraagt of je dat kunt, maar een luisteraar hoort een verzoek. Volgens de encyclopedie wordt de vraag of je het zout kunt aangeven normaal opgevat als een verzoek om het te doen; naar de filosoof John Searle (1979) heten zulke gevallen indirecte taalhandelingen. De psycholoog Herbert Clark (1979) noemde de Engelse vormen van *kun je het zout aangeven?* en *zou je het zout kunnen aangeven?* heel gangbare, bijna vaste manieren om iemand om het zout te vragen. Taalhandelingen horen bij de **pragmatiek**: de studie van wat sprekers in een context bedoelen en doen met woorden.

    ### Samenwerken

    Hoe komt een luisteraar van de woorden bij wat bedoeld wordt? De filosoof Paul Grice (1975) stelde een globaal principe voor, waarvan je mag verwachten dat deelnemers aan een gesprek het volgen als er niets bijzonders speelt: het coöperatieprincipe. Dat luidt: lever je bijdrage zoals die nodig is, op het moment dat die komt, voor het doel of de richting van het gesprek waaraan je meedoet. Daaronder vallen vier soorten maximes, vuistregels:

    - **kwantiteit**: maak je bijdrage zo informatief als nodig is voor het gesprek, en niet informatiever dan nodig;
    - **kwaliteit**: probeer iets te zeggen wat waar is;
    - **relatie**: wees relevant;
    - **manier**: wees helder: vermijd duisterheid en dubbelzinnigheid, wees kort en ordelijk.

    Grice noemde de tweede maxime van kwantiteit zelf betwistbaar. Een luisteraar gaat ervan uit dat de spreker zich aan de maximes houdt, en werkt uit wat de spreker dan moet bedoelen. De klant van de reisagent noemde de data van een vergadering, geen reisdag. Als dat antwoord relevant is, bepaalt de vergadering de reis, en omdat mensen graag de dag vóór een vergadering aankomen, kan de reisagent op de 11e uitkomen. Wat een spreker zo suggereert, voorbij wat er gezegd is, en wat een luisteraar uitwerkt, heet een **implicatuur**.
    """)
    return


@app.cell(hide_code=True)
def text_idea_scale():
    mo.md(r"""
    ### Sommige, maar niet alle

    *Sommige katten slapen* is in de logica waar als er minstens één kat slaapt, ook als ze allemaal slapen: zo telde je in *Betekenis: wanneer is een zin waar?*. Toch maakt een luisteraar eruit op dat niet alle katten slapen. De maxime van kwantiteit verklaart dat: een spreker die wist dat alle katten slapen, en dat kon zeggen, had het informatievere *alle* moeten gebruiken.

    Volgens de Stanford Encyclopedia of Philosophy zag de taalkundige Laurence Horn (1972, 1989) dat de Engelse kwantoren voor *alle*, *de meeste*, *veel* en *sommige* een schaal vormen, het sterkste woord eerst. De encyclopedie geeft er de regel bij: wie een woord van de schaal gebruikt, suggereert dat de zin met elk woord links ervan onwaar is, maar niet met de woorden rechts ervan. *De meeste* suggereert dus *niet alle*, en *sommige* suggereert *niet veel*, *niet de meeste* en *niet alle*. De woorden rechts blijven waar: als de meeste katten slapen, slapen er ook sommige.

    Een implicatuur is niet wat de zin zegt, en dat blijkt: de spreker kan de implicatuur terugnemen. De encyclopedie geeft *sommige sporters roken*, gevolgd door *sterker nog, ze roken allemaal*. Die tweede zin spreekt de eerste niet tegen; die zin neemt alleen terug wat een luisteraar uit de eerste zou opmaken.
    """)
    return


@app.cell(hide_code=True)
def scale_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_presupposition():
    mo.md(r"""
    ### Als vanzelfsprekend aangenomen

    *Tom is gestopt met Pip achtervolgen* zegt dat Tom Pip nu niet achtervolgt. De zin neemt ook als vanzelfsprekend aan dat Tom Pip eerst wel achtervolgde: een luisteraar die dat niet wist, hoort het terloops. Wat een zin zo als vanzelfsprekend aanneemt, is een **presuppositie**. Woorden zoals *stoppen*, *weer* en *de* of *het* brengen er een mee; de Stanford Encyclopedia of Philosophy geeft als voorbeeld dat China gestopt is met het aanleggen van voorraden metaal, wat aanneemt dat China dat eerst deed.

    Een test onderscheidt een presuppositie van wat een zin zegt: voeg *niet* toe. *Tom is niet gestopt met Pip achtervolgen* zegt het omgekeerde over nu, en neemt nog steeds aan dat Tom Pip eerst achtervolgde. De encyclopedie noemt dat de negatietest, de gangbare manier om een presuppositie te herkennen. Andere plekken werken ook als test, zoals een vraag of na *als*, maar daar blijven presupposities niet altijd staan, en onder *niet* evenmin altijd; wanneer ze wel blijven staan, wordt nog volop onderzocht.
    """)
    return


@app.cell(hide_code=True)
def presupposition_picture():
    mo.Html(C.presup_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_the():
    mo.md(r"""
    *De* en *het* nemen aan dat er precies één zo'n ding is: *de kat die slaapt* verwijst naar de ene kat die slaapt. Maar hoe zit het met *de kat die slaapt achtervolgt Tom* in een wereld waarin geen kat slaapt? De filosoof Bertrand Russell (1905) vond zo'n zin onwaar. Over *de huidige koning van Frankrijk is kaal*, geschreven toen Frankrijk geen koning had, schreef Russell dat die zin zeker onwaar is. De filosoof Peter Strawson (1950) betoogde volgens de encyclopedie dat een zin geen waarheidswaarde kan hebben als zo'n woordgroep naar niets verwijst: de zin is dan waar noch onwaar. In code kan dat niets `None` zijn.

    In een echt gesprek wijst de context vaak één ding aan: *de kat* is de kat van het huis, ook in een wereld met veel katten. Deze les laat de context weg, en telt in de kleine wereld.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **pragmatiek** | de studie van wat sprekers in een context bedoelen en doen met woorden, voorbij wat de woorden zeggen |
        | **taalhandeling** | een uiting gezien als handeling, zoals beweren, vragen, verzoeken, beloven of wedden |
        | **implicatuur** | wat een spreker suggereert voorbij wat er gezegd is, en wat een luisteraar uitwerkt, zoals *niet alle* na *sommige* |
        | **presuppositie** | wat een zin als vanzelfsprekend aanneemt, zoals dat Tom Pip eerst achtervolgde, in *Tom is gestopt met Pip achtervolgen* |
        | **schaal** | hier: woorden op een rij van sterk naar zwak, zoals *alle*, *de meeste*, *veel*, *sommige* |
        | `return None` | geef `None` terug: helemaal geen waarde, hier voor een woordgroep die naar niets verwijst |
        """),
        L.watch_out("*Sommige* betekent nog steeds *minstens één*, zoals in de logica. Het *niet alle* is een implicatuur die daar bovenop komt: de spreker suggereert het, en een luisteraar werkt het uit. Code die nakijkt wat een zin zegt, zoals `some` in *Betekenis: wanneer is een zin waar?*, ziet die implicatuur niet."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Vraag je af wat de spreker met de uiting doet: beweren, vragen, verzoeken, beloven.
    2. Zoek wat de zin zegt: de waarheidsvoorwaarden in de wereld.
    3. Vraag je af wat een coöperatieve spreker in plaats daarvan had kunnen zeggen. Sterkere woorden op een schaal die niet gezegd zijn, vat een luisteraar op als onwaar.
    4. Zoek woorden die iets als vanzelfsprekend aannemen, zoals *stoppen*, *weer* en *de* of *het*. Wat ze aannemen, blijft meestal staan als je *niet* toevoegt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel schrijft drie schalen als lijsten, het sterkste woord eerst: de schaal van Horn en twee van de course zelf. De tweede schrijft de wereld van *Betekenis: wanneer is een zin waar?* en de nacht. De derde definieert twee functies: `says_some` is de `some` van die les, wat de zin zegt; `means_some` voegt toe wat een luisteraar eruit opmaakt, *maar niet alle*. Daarna probeert de cel beide op de katten in de twee werelden.
    """)
    return


@app.cell
def scales_cell():
    SCALE = ["alle", "de meeste", "veel", "sommige"]
    HEAT = ["heet", "warm"]
    OFTEN = ["altijd", "vaak", "soms"]
    return (SCALE,)


@app.cell
def world_cell():
    WORLD = {
        "kat": ["Tom", "Mia"],
        "hond": ["Rex"],
        "muis": ["Pip"],
        "slaapt": ["Mia", "Rex"],
        "achtervolgt": [("Tom", "Pip"), ("Rex", "Tom")],
    }
    NIGHT = {
        "kat": ["Tom", "Mia"],
        "hond": ["Rex"],
        "muis": ["Pip"],
        "slaapt": ["Tom", "Mia", "Rex"],
        "achtervolgt": [("Pip", "Tom")],
    }
    return NIGHT, WORLD


@app.cell
def example_look(NIGHT, SCALE, WORLD):
    def says_some(things, group):
        inside = [thing for thing in things if thing in group]
        return len(inside) > 0

    def means_some(things, group):
        inside = [thing for thing in things if thing in group]
        return len(inside) > 0 and len(inside) < len(things)

    # sommige katten slapen: in WORLD, dan 's nachts
    print(says_some(WORLD["kat"], WORLD["slaapt"]), means_some(WORLD["kat"], WORLD["slaapt"]))
    print(says_some(NIGHT["kat"], NIGHT["slaapt"]), means_some(NIGHT["kat"], NIGHT["slaapt"]))
    print(SCALE[0], SCALE[1:3])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In `WORLD` slaapt een van de twee katten: *sommige katten slapen* is waar, en wat een luisteraar eruit opmaakt ook. In `NIGHT` slapen beide katten. De zin is in de logica nog steeds waar, maar een luisteraar die de zin hoort, maakt eruit op dat niet alle katten slapen, en dat is onwaar. Een spreker die zich aan kwantiteit houdt, zegt daar *alle katten slapen*. De laatste regel kijkt naar de schaal: plek 0 bevat het sterkste woord, en een slice neemt er een rij woorden uit. Oefening 1 vindt bij een woord op een schaal de sterkere woorden die een luisteraar opvat als onwaar.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_implicated():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. implicated: de sterkere woorden

        Schrijf `implicated(scale, word)`: geef de woorden van `scale` terug die sterker zijn dan `word`, in de volgorde van de schaal. Bij *sommige* geeft `implicated(SCALE, "sommige")` dan `["alle", "de meeste", "veel"]`: een luisteraar vat *sommige* op als niet alle, niet de meeste en niet veel. Geef bij een woord dat niet op de schaal staat een lege lijst terug. Vervang de `...`. De check roept `implicated(SCALE, SCALE[3])` aan, en daarna eigen gevallen met de drie schalen, waaronder `implicated(SCALE, HEAT[1])`.
        """),
        L.task("Vind de woorden op de schaal die een luisteraar opvat als onwaar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_implicated():
    box_implicated = L.code_box("implicated", lines=5)
    box_implicated
    return (box_implicated,)


@app.cell(hide_code=True)
def check_implicated(box_implicated):
    result_implicated = L.check("implicated", box_implicated.value)
    result_implicated
    return (result_implicated,)


@app.cell(hide_code=True)
def hints_implicated():
    L.hints(
        "implicated",
        [
            "Loop met `for i in range(len(scale)):` langs de plekken van de schaal, en zoek de plek waar `scale[i]` gelijk is aan `word`. Geef na de loop een lege lijst terug.",
            "De schaal begint met het sterkste woord. Op welke plekken staan de woorden die sterker zijn dan het woord op plek `i`?",
            "`            return ___`",
        ],
        why="`for i in range(len(scale)):` met daaronder `if scale[i] == word:` en `return scale[:i]`; na de loop `return []`. `implicated(SCALE, SCALE[3])` geeft `['alle', 'de meeste', 'veel']`, en `implicated(SCALE, SCALE[0])` geeft `[]`: niets is sterker dan *alle*.",
    )
    return


@app.cell(hide_code=True)
def text_after_implicated():
    mo.md(r"""
    De afbeelding onder de check vraagt je functie naar elk woord van de schaal van Horn. Hoe zwakker het woord, hoe meer een luisteraar opvat als onwaar; bij *alle* niets. Een woord dat niet op de schaal staat, zegt er niets over: *warm* suggereert *niet heet*, en niets over hoeveel katten slapen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: zes katten

    Kies hoeveel van de zes katten slapen, en zie wat elke zin in de logica zegt en wat een luisteraar eruit opmaakt. Kies daarna een zin: de twee rijen vakjes laten bij 0 tot en met 6 slapende katten zien waar de zin waar is in de logica, en waar de zin waar is zoals een luisteraar de zin opvat. Zoek de aantallen waarbij *sommige katten slapen* in de logica waar is, maar niet zoals een luisteraar de zin opvat, en kijk welke zin een spreker die zich aan kwantiteit houdt daar zegt.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_count = mo.ui.slider(0, 6, value=6, label="Slapende katten", show_value=True)
    demo_sentence = mo.ui.dropdown(options=C.demo_options(L.lang), value=C.DEMO_LABELS[L.lang][2], label="Zin")
    mo.vstack([demo_count, demo_sentence])
    return demo_count, demo_sentence


@app.cell(hide_code=True)
def demo_look(demo_count, demo_sentence):
    mo.Html(C.demo_picture(demo_count.value, demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_the_one():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. the_one: waar *de* naar verwijst

        Schrijf `the_one(things, group)`: geef het ene ding uit `things` terug dat ook in `group` zit, en `None` als er niet precies één is. *De kat die slaapt* is dan `the_one(WORLD["kat"], WORLD["slaapt"])`, en dat geeft `"Mia"`. *De kat* alleen is `the_one(WORLD["kat"], ["Tom", "Mia", "Rex", "Pip"])`, met alle dieren als `group`, en dat geeft `None`: er zijn twee katten. Vervang de `...`. De check roept `the_one(["Tom", "Mia"], ["Mia", "Rex"])` aan, en daarna eigen gevallen, met één zo'n ding, met twee en met geen enkel.
        """),
        L.task("Vind het ene ding waar een woordgroep met *de* naar verwijst, of `None`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_the_one():
    box_the_one = L.code_box("the_one", lines=5)
    box_the_one
    return (box_the_one,)


@app.cell(hide_code=True)
def check_the_one(box_the_one):
    result_the_one = L.check("the_one", box_the_one.value)
    result_the_one
    return (result_the_one,)


@app.cell(hide_code=True)
def hints_the_one():
    L.hints(
        "the_one",
        [
            "Maak de lijst `inside` zoals `says_some` dat doet in het uitgewerkte voorbeeld. Kijk de lengte na met `if len(inside) == 1:`; geef anders `None` terug.",
            "Als `inside` precies één ding bevat, op welke plek in `inside` staat dat ding dan?",
            "`        return ___`",
        ],
        why="`inside = [thing for thing in things if thing in group]`; `if len(inside) == 1:` met `return inside[0]` eronder; aan het eind `return None`. `the_one([\"Tom\", \"Mia\"], [\"Mia\", \"Rex\"])` geeft `'Mia'`.",
    )
    return


@app.cell(hide_code=True)
def text_after_the_one():
    mo.md(r"""
    De afbeelding onder de check vraagt je functie waar vijf woordgroepen in de wereld naar verwijzen, en zet elke woordgroep in een zin: *... achtervolgt Tom*. Waar een woordgroep naar niets verwijst, geeft je functie `None`, en geeft de afbeelding de zin geen waarheidswaarde: het antwoord van Strawson. Volgens Russell is elk van die zinnen onwaar. Een programma moet een van de twee kiezen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_acts():
    mo.md(r"""
    ### Oefening 3. Wat doet de spreker?

    Zet elke vetgedrukte zin bij wat de spreker ermee doet. Kijk verder dan de vorm van de zin: een vraag kan meer doen dan vragen.
    """)
    return


@app.cell(hide_code=True)
def exercise_acts():
    sort_acts = L.sort("acts")
    sort_acts
    return (sort_acts,)


@app.cell(hide_code=True)
def check_acts(sort_acts):
    result_acts = L.check("acts", sort_acts)
    result_acts
    return (result_acts,)


@app.cell(hide_code=True)
def text_exercise_cancel():
    mo.md(r"""
    ### Oefening 4. Terugnemen

    Een implicatuur kun je terugnemen zonder tegenspraak; wat een zin zegt niet. Een presuppositie zit ertussenin: die blijft meestal staan onder *niet*, en wie de presuppositie ronduit ontkent, laat een zin met zichzelf botsen.
    """)
    return


@app.cell(hide_code=True)
def choice_cancel_cell():
    choice_cancel = L.choice(
        "cancel",
        question="Welke van deze zinnen neemt een implicatuur terug, en klopt dus nog?",
        options={
            "right": "*Sommige taartjes zijn op; eigenlijk zijn ze allemaal op.*",
            "stopped": "*Sam is gestopt met roken; eigenlijk rookte Sam vroeger niet.*",
            "all": "*Alle taartjes zijn op; eigenlijk zijn er nog twee.*",
            "bakery": "*De bakkerij is gestopt met taartjes verkopen; eigenlijk verkoopt de bakkerij ze nog elke dag van de week.*",
        },
        correct="right",
        why={
            "right": "*Sommige* zegt minstens één, en suggereert *niet alle*. *Eigenlijk zijn ze allemaal op* neemt de suggestie terug, niet wat er gezegd is: als alle taartjes op zijn, zijn er ook sommige op. Zo kun je een implicatuur terugnemen.",
            "stopped": "*Is gestopt met roken* neemt aan dat Sam vroeger rookte, en de tweede helft ontkent dat: de zin botst met zichzelf. Een presuppositie neem je niet zo makkelijk terug als een implicatuur. Zoek een zin die alleen terugneemt wat een luisteraar uitwerkt.",
            "all": "*Alle taartjes zijn op* zegt dat er geen meer over is, en *er zijn er nog twee* ontkent wat er gezegd is, niet iets wat een luisteraar uitwerkte: een tegenspraak. Zoek een zin die alleen terugneemt wat een luisteraar uitwerkt.",
            "bakery": "*Is gestopt met taartjes verkopen* zegt dat de bakkerij ze nu niet verkoopt, en *verkoopt ze nog* ontkent wat er gezegd is: een tegenspraak. Zoek een zin die alleen terugneemt wat een luisteraar uitwerkt.",
        },
    )
    choice_cancel
    return (choice_cancel,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Betekenis: wanneer is een zin waar?](/course/open/nlp-meaning-and-truth?lang=nl) controleerde wat een zin zegt tegen een wereld. Deze les voegde toe wat een spreker bedoelt: de handeling die met de woorden gedaan wordt, de sterkere woorden op een schaal die een luisteraar opvat als onwaar, en wat een zin als vanzelfsprekend aanneemt. Je zette de implicaturen van een schaal op een rij, vond waar *de* naar verwijst, deelde taalhandelingen in en nam een implicatuur terug. De lessen *Hoe een gesprek werkt* en *Een tekst is meer dan losse zinnen* bouwen hierop voort: een gesprek als een rij handelingen, en een tekst als meer dan een rij zinnen.

    **Een voorbeeld uit het dagelijks leven.** In een recensie staat dat het eten *eetbaar* was. Niemand noemt een geweldige maaltijd eetbaar: de recensent had *lekker* kunnen zeggen, en deed dat niet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cancel, result_acts, result_implicated, result_the_one):
    L.wrap_up(result_implicated, result_the_one, result_acts, choice_cancel)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
