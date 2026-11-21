# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-morphemes. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-morphemes", "nl")
    C = L.checks   # de lijsten uitgangen en korte woorden van de course, in het Engels en in het Nederlands


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
def quick_check_field():
    quick_field = L.choice(
        "quick-field",
        question="Wie computationele linguïstiek en NLP uit elkaar houdt: welk project is dan computationele linguïstiek?",
        options={
            "right": "Met een computer tellen welke uitgangen het Nederlands het meest gebruikt, om te leren hoe het Nederlands woorden maakt",
            "app": "Voor een webwinkel een zoekvak bouwen dat woorden in delen knipt, zodat elke zoekopdracht meer producten vindt dan eerst",
            "spell": "Een spellingchecker bouwen: onbekende woorden rood onderstrepen",
        },
        correct="right",
        why={
            "right": "Het stelt een vraag over taal zelf, en gebruikt een computer om die te beantwoorden. Over die vraag gaat deze les: hoe woorden gebouwd zijn.",
            "app": "Een zoekvak dat moet werken, is de technische kant: NLP. Computationele linguïstiek stelt, voor wie de twee uit elkaar houdt, vragen over taal zelf. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "spell": "Een spellingchecker is gereedschap dat moet werken: de technische kant, NLP. Computationele linguïstiek stelt vragen over taal zelf. Zie [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_field
    return (quick_field,)


@app.cell(hide_code=True)
def quick_check_keyword():
    quick_keyword = L.choice(
        "quick-keyword",
        question="In *Wat NLP is* zoekt `intent` elk keyword in het verzoek, en `\"repair\"` is een keyword. Wat geeft de functie voor het Engelse verzoek *My brakes need repairs.* (mijn remmen moeten gerepareerd worden)?",
        options={
            "right": "`\"repair\"`, want het keyword is een stuk van het woord *repairs*",
            "unknown": "`\"unknown\"`, want *repairs* en *repair* zijn twee verschillende woorden",
            "error": "Een fout, want het verzoek heeft een woord dat langer is dan het keyword",
        },
        correct="right",
        why={
            "right": "`intent` zoekt het keyword als stuk van de string, dus *repairs* telt ook: de *-s* zegt alleen dat het er meer dan één zijn. Deze les knipt woorden met opzet in zulke delen.",
            "unknown": "`intent` vergelijkt geen hele woorden: de functie zoekt het keyword overal in het verzoek, en *repair* is het begin van *repairs*. Zie de oefening *intent* in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
            "error": "Zoeken naar een stuk van een string geeft `True` of `False`, geen fout, hoe lang de woorden ook zijn. Zie de oefening *intent* in [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl).",
        },
    )
    quick_keyword
    return (quick_keyword,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Voor een computer zijn *fiets*, *fietst*, *fietste* en *fietsen* vier verschillende strings. Een zoekopdracht naar *fietsen* die alleen die string zoekt, mist een pagina over iemand die naar het werk *fietste*. En mensen maken steeds nieuwe woorden: misschien heb je *herwasbaar* nooit gezien, en toch weet je wat het betekent, omdat je de delen kent.

    Deze les knipt woorden in zulke delen, met een lijst bekende uitgangen en een lijst bekende woorden, en zet de drie manieren op een rij waarop een taal woorden maakt.

    **Een voorbeeld uit het dagelijks leven.** Met een doos bouwsteentjes bouw je dingen die nog niemand bouwde, uit steentjes die je kent. Een taal maakt op dezelfde manier nieuwe woorden uit bekende delen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een tabel uit 2003 telt de verschillende woorden in ongeveer 37 miljoen woorden Engelse tekst, en in evenveel Nederlandse tekst. De Engelse tekst heeft ongeveer 165 duizend verschillende woorden. Ongeveer hoeveel heeft de Nederlandse tekst?",
        options={
            "same": "Ongeveer evenveel, zo'n 170 duizend",
            "right": "Bijna drie keer zoveel",
            "ten": "Tien keer zoveel, ruim een miljoen",
        },
        reveal="**Bijna drie keer zoveel: 462 duizend.** De tabel staat in een paper van Roeland Ordelman, Arjan van Hessen en Franciska de Jong (2003), die de getallen voor de andere talen uit eerder werk van Adda-Decker en Lamel haalden; het Duits had er 650 duizend. Ze noemen samenstellen (woorden tot één woord samenvoegen), naast de vormen die uitgangen maken, de belangrijkste oorzaak van de vele verschillende woorden in het Nederlands en het Duits. Met een lijst van 65 duizend woorden ontbraken 0,4 van elke 100 woorden van de Engelse tekst in de lijst, en 2,85 van elke 100 van de Nederlandse.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Delen van woorden

    Je kunt *boeken* knippen in *boek* en *-en*, en *werkte* in *werk* en *-te*. Daniel Jurafsky en James H. Martin noemen zulke delen in hun leerboek *Speech and Language Processing* (versie van 19 augustus 2026) morfemen: de kleinste eenheden in een taal die betekenis dragen. **Morfologie** is de leer van hoe woorden uit morfemen zijn opgebouwd. De *-en* van *boeken* heeft geen eigen betekenis zoals *boek*: die zegt alleen dat het er meer dan één zijn. Het leerboek *Essentials of Linguistics* (tweede druk, 2022) spreekt dan ook van betekenis of grammaticale functie. De course neemt beide: een **morfeem** is het kleinste deel van een woord met een betekenis of een taak in de grammatica.

    De **wortel** is het morfeem met de hoofdbetekenis, zoals *werk* in *werkte*. Een **affix** voegt er iets aan toe: een **voorvoegsel** (prefix) staat ervoor, zoals *on-* in *ongewoon*, en een **achtervoegsel** (suffix) erachter, zoals *-en* in *boeken*. *boek* kan op zichzelf een woord zijn, en *-en* niet: de e-ANS, de naslaggrammatica van het Nederlands, noemt affixen gebonden morfemen, omdat ze niet zelfstandig als woord voorkomen. Een woord kan veel morfemen hebben: Jurafsky en Martin knippen het Engelse *carefully* (zorgvuldig) in *care-ful-ly*, en volgens *Essentials* heeft *non-renewability* er minstens vijf: *non-*, *re-*, *new*, *-abil* en *-ity*.

    ### Drie manieren om woorden te maken

    **Inflectie** (in de e-ANS: flexie) maakt de vormen van één woord die een zin nodig heeft: *loop*, *loopt*, *liep*. Bij werkwoorden heet dat vervoegen, bij zelfstandige en bijvoeglijke naamwoorden verbuigen. Jurafsky en Martin noemen voor het Engels de *-s* (of *-es*) voor meer dan één en de *-ed* voor de verleden tijd. Volgens *Essentials* verandert inflectie nooit het soort woord van het basiswoord, en de e-ANS zegt hetzelfde: een werkwoord blijft een werkwoord, en een zelfstandig naamwoord blijft een zelfstandig naamwoord. Een latere les komt terug op zulke soorten woorden.

    **Derivatie** (in de grammatica ook afleiding) maakt een nieuw woord. Van het bijvoeglijk naamwoord *schoon* komt het zelfstandig naamwoord *schoonheid*, en van het Engelse *care* komt *careful*, en daarvan weer *carefully*. Jurafsky en Martin schrijven dat derivatie meestal een ander soort woord geeft dan de wortel, vaak met een betekenis die moeilijk precies te voorspellen is. Meestal, niet altijd: *ongewoon* is een bijvoeglijk naamwoord, net als *gewoon*. Ook de e-ANS zegt dat het soort woord bij derivatie kan veranderen, maar dat niet hoeft. Jurafsky en Martin noemen de grens tussen inflectie en derivatie bovendien een glijdende schaal tussen twee polen: niet elke uitgang valt netjes aan één kant.

    Een **samenstelling** is een woord uit twee of meer woorden, zoals *voordeur*. Het laatste deel zegt meestal wat voor ding het is: een voordeur is een deur. Volgens *Essentials* schrijft het Engels samenstellingen niet op een vaste manier: oudere vaak aan elkaar, andere met een streepje, en veel nieuwe met spaties, alsof het losse woorden zijn. Het Nederlands schrijft ze als één woord: de Leidraad, de officiële spellingregels van de Taalunie, zegt dat je de delen van een samenstelling aan elkaar schrijft, dus het Engelse *science fiction* is in het Nederlands *sciencefiction*. De e-ANS noemt samenstellen de belangrijkste manier om in het Nederlands nieuwe woorden te maken.

    ### Waarom een programma de delen nodig heeft

    Jurafsky en Martin waarschuwen dat geen lijst woorden, hoe groot ook, elk woord bevat dat kan voorkomen, dus een programma komt steeds woorden tegen die het niet kent. Philipp Koehn en Kevin Knight (2003) schrijven over talen als het Duits en het Nederlands dat woorden die je vrij aan elkaar kunt plakken het aantal verschillende woorden enorm laten groeien, zodat er per woord weinig data is. Rico Sennrich, Barry Haddow en Alexandra Birch (2016) bekeken 100 zeldzame woorden, die niet bij de 50.000 meest voorkomende hoorden, in de Duitse tekst waarop ze een vertaalsysteem trainden: 56 waren samenstellingen, en 21 waren namen.

    Een programma dat de delen van woorden kent, kan iets met een woord dat het nooit zag: *werkte* via *werk*, en *voordeur* via *voor* en *deur*. Maar zo'n programma ziet alleen letters, dus het kan ook een woord knippen waar geen deel zit.
    """)
    return


@app.cell(hide_code=True)
def parts_cell():
    mo.Html(C.parts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **morfologie** | de leer van hoe woorden uit morfemen zijn opgebouwd |
        | **morfeem** | het kleinste deel van een woord met een betekenis of een taak in de grammatica, zoals *boek* en *-en* in *boeken* |
        | **wortel**, **affix** | het morfeem met de hoofdbetekenis, en een morfeem dat er iets aan toevoegt |
        | **voorvoegsel**, **achtervoegsel** | een affix vóór wat het aanvult, zoals *on-*, en een erachter, zoals *-te*; in het Engels prefix en suffix |
        | **inflectie** | de vormen van één woord maken, zoals *liep* van *lopen*; het soort woord blijft hetzelfde (ook: flexie) |
        | **derivatie** | een nieuw woord maken met een affix, zoals *schoonheid* van *schoon*; vaak een ander soort woord (ook: afleiding) |
        | **samenstelling** | een woord uit twee of meer woorden, zoals *voordeur* |
        | `C.SUFFIXES`, `C.WORDS` | de lijst Engelse uitgangen van de course, `["ing", "ed", "s"]`, en de lijst met twaalf korte Engelse woorden |
        | `C.SUFFIXES_NL`, `C.WORDS_NL` | hetzelfde voor het Nederlands: `["te", "en", "t"]`, en twaalf korte Nederlandse woorden |
        | `x in a_list` | `True` als `x` een van de items van de lijst is, en `False` als dat niet zo is |
        """),
        L.watch_out("In de wiskunde is een afleiding iets anders: een reeks stappen van de ene formule naar de volgende, en de afgeleide van een functie is weer iets anders. In deze les is derivatie het maken van een nieuw woord uit een woord. En in de schoolgrammatica is de stam alleen de basisvorm van een werkwoord, zoals *loop* van *lopen*; het deel met de hoofdbetekenis van elk woord heet hier de wortel."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Maak een lijst met bekende delen: uitgangen zoals *-te*, of korte woorden zoals *deur*.
    2. Probeer elk deel op het woord: knip het eraf, en plak het er weer aan om te zien of je het woord terugkrijgt.
    3. Kijk naar wat overblijft: lang genoeg voor een wortel, of ook een bekend woord.
    4. Geef de delen terug die je vond, of het hele woord als één deel als niets past.
    5. Kijk naar de delen: een lijst ziet letters, geen betekenis, en kan dus knippen waar geen morfeem zit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel knipt met een slice de uitgang *-t* van drie woorden af, plakt de uitgang er weer aan, en print of je dan het woord terugkrijgt.
    """)
    return


@app.cell
def example_cut():
    for word in ["werkt", "nacht", "werken"]:
        rest = word[:len(word) - len("t")]   # het woord zonder de laatste letter
        print(word, rest, rest + "t" == word)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij *werkt* en *nacht* krijg je het woord terug als je *-t* er weer aan plakt, en bij *werken*, dat op *-en* eindigt, niet. Maar *nacht* heeft geen uitgang: het is één morfeem, en *nach* is geen woord. Een lijst uitgangen ziet letters, geen betekenis. Oefening 1 probeert een hele lijst uitgangen, in volgorde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cut():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. cut_suffix: een bekende uitgang afknippen

        `C.SUFFIXES` is de lijst Engelse uitgangen van de course, `["ing", "ed", "s"]`, en `C.SUFFIXES_NL` de Nederlandse, `["te", "en", "t"]`. Schrijf `cut_suffix(word, suffixes)`: ga de uitgangen van de lijst `suffixes` in volgorde langs, en reken voor elke uitgang uit wat er zonder die uitgang van het woord overblijft, zoals in het uitgewerkte voorbeeld. Geef bij de eerste uitgang waarop het woord echt eindigt, met minstens 3 letters over, de lijst `[rest, suffix]` terug. Past er geen uitgang, geef dan `[word]` terug: het woord is één deel. De 3 letters zijn een keuze van de course zelf: zo wordt *kat* niet *ka* en *-t*. Vervang de `...`. De check roept `cut_suffix("walked", SUFFIXES)` aan (*walked* is Engels voor gewandeld), daarna andere woorden en eigen lijsten, ook Nederlandse woorden met `SUFFIXES_NL`.
        """),
        L.task("Probeer de uitgangen in volgorde, en geef terug wat overblijft en de eerste uitgang die past."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cut():
    box_cut = L.code_box("cut_suffix", lines=6)
    box_cut
    return (box_cut,)


@app.cell(hide_code=True)
def check_cut(box_cut):
    result_cut = L.check("cut_suffix", box_cut.value)
    result_cut
    return (result_cut,)


@app.cell(hide_code=True)
def hints_cut():
    L.hints(
        "cut_suffix",
        [
            "Gebruik een `for`-loop over `suffixes`, en doe voor elke uitgang de stappen van het uitgewerkte voorbeeld.",
            "Wat geef je terug zodra een uitgang past en er genoeg letters overblijven? En wat na de loop, als er geen paste?",
            "`        if ___ and len(rest) >= 3:`",
        ],
        why="`for suffix in suffixes:`, dan `rest = word[:len(word) - len(suffix)]` en `if rest + suffix == word and len(rest) >= 3:` met `return [rest, suffix]` eronder, en na de loop `return [word]`. `cut_suffix(\"walked\", SUFFIXES)` geeft `['walk', 'ed']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_cut():
    mo.md(r"""
    De afbeelding onder de check toont je functie op meer Nederlandse woorden. Net als de functie van de course knipt je functie *nacht*, en maakt ze van *lopen* de delen *lop* en *-en*, terwijl de wortel *loop* is: de spelling laat een *o* weg. Oefening 2 knipt een woord in twee bekende woorden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. split_compound: twee bekende woorden

        `C.WORDS` is de lijst korte Engelse woorden van de course, zoals `"door"` (deur) en `"bell"` (bel), en `C.WORDS_NL` de Nederlandse. Schrijf `split_compound(word, words)`: probeer de knippen van links naar rechts met `for i in range(1, len(word)):`, met het linkerstuk `word[:i]` en het rechterstuk `word[i:]`. Geef bij de eerste knip waarbij beide stukken in de lijst `words` staan, de lijst met de twee stukken terug; werkt geen enkele knip, geef dan `[word]` terug. Vervang de `...`. De check roept `split_compound("doorbell", WORDS)` aan (*doorbell* is Engels voor deurbel), daarna andere woorden en eigen lijsten, ook Nederlandse woorden met `WORDS_NL`.
        """),
        L.task("Probeer elke knip, en geef de eerste terug waarbij beide stukken bekende woorden zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split_compound", lines=7)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split_compound", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split_compound",
        [
            "Een knip na letter `i` geeft twee stukken. Probeer elke plek met de `for`-loop uit de oefening, en kijk elke keer naar beide stukken.",
            "Welke stukken moeten in `words` staan voordat je splitst? En wat geef je terug als geen enkele plek werkt?",
            "`        if ___:`",
        ],
        why="`for i in range(1, len(word)):`, dan `left = word[:i]` en `right = word[i:]`, en `if left in words and right in words:` met `return [left, right]` eronder, en na de loop `return [word]`. `split_compound(\"doorbell\", WORDS)` geeft `['door', 'bell']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_split():
    mo.md(r"""
    Je functie splitst een samenstelling in twee bekende woorden. Een Nederlandse samenstelling kan meer delen hebben, en soms een klank ertussen, zoals in *pannenkoek*: de volgende les, *Nederlandse samenstellingen splitsen*, komt daarop terug.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: knip een eigen woord

    Typ een woord, en zie wat de twee functies van de course ervan maken, met de Nederlandse uitgangen en woorden van de course. Probeer *werkte*, *kat* of *voordeuren*: elke functie knipt hoogstens één keer.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="werkt", label="Een woord")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 3. Drie manieren om een woord te maken

    Elk woord hieronder is gemaakt van het woord of de woorden erachter. Zet het in de groep van de manier waarop het gemaakt is.
    """)
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.md(r"""
    ### Oefening 4. Eén woord of twee?

    Een zoekvak behandelt *liep* en *lopen* als hetzelfde woord, zodat een zoekopdracht naar het ene ook het andere vindt. Iemand wil hetzelfde doen met *schoonheid* en *schoon*.
    """)
    return


@app.cell(hide_code=True)
def choice_merge_cell():
    choice_merge = L.choice(
        "merge",
        question="Waarom is dat riskanter dan bij *liep* en *lopen*?",
        options={
            "right": "*schoonheid* betekent iets anders dan *schoon*",
            "same": "Niet riskanter: in allebei zit hetzelfde basiswoord",
            "longer": "*-heid* is een lang achtervoegsel, dus een programma vindt het minder makkelijk aan het eind van een woord",
        },
        correct="right",
        why={
            "right": "*liep* is *lopen* in de verleden tijd. *schoonheid* is een nieuw woord, van een ander soort, met een eigen betekenis: wie *schoon* zoekt, zoekt meestal niet naar *schoonheid*. Jurafsky en Martin schrijven dat de betekenis van een afgeleid woord vaak moeilijk te voorspellen is, dus wie *schoonheid* en *schoon* samenvoegt, krijgt bij een zoekopdracht naar het ene pagina's over het andere.",
            "same": "In allebei zit een basiswoord, maar *liep* is een vorm van hetzelfde woord en *schoonheid* een nieuw woord. Zie *Drie manieren om woorden te maken* in *Het idee*.",
            "longer": "Een programma vindt *-heid* aan het eind van een woord net zo makkelijk als *-en* in *lopen*. Het gaat erom wat de uitgang maakt. Zie *Drie manieren om woorden te maken* in *Het idee*.",
        },
    )
    choice_merge
    return (choice_merge,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Wat NLP is](/course/open/nlp-what-is-nlp?lang=nl) vond keywords als stukken van een string, en *repairs* telde bij toeval mee voor *repair*. Deze les keek in woorden: morfemen, de kleinste delen met een betekenis of een taak in de grammatica, die de morfologie bestudeert. Inflectie maakt vormen van één woord, derivatie maakt nieuwe woorden, en een samenstelling voegt woorden samen; het Nederlands schrijft samenstellingen als één woord. Je knipte bekende uitgangen af met een lijst, splitste een woord in twee bekende woorden, en zag dat een lijst, die alleen letters ziet, kan knippen waar geen deel zit. De volgende les, *Nederlandse samenstellingen splitsen*, splitst woorden van meer dan twee delen, met de klanken die het Nederlands ertussen zet.

    **Een voorbeeld uit het dagelijks leven.** Een woordenboek heeft een trefwoord *boek*, maar geen trefwoord *boeken*: je vindt *boeken* omdat je de delen kent.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_merge, result_cut, result_kinds, result_split):
    L.wrap_up(result_cut, result_split, result_kinds, choice_merge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
