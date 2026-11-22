# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-stemming. Route: course/route/lessen/taal-en-zoeken.yaml. Controles: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-stemming", "nl")
    C = L.checks   # de woorden van de course, de afbeeldingen en de controles


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl), en een over [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_inflection():
    quick_inflection = L.choice(
        "quick-inflection",
        question="Welk paar is twee vormen van één woord, gemaakt door inflectie?",
        options={
            "derive": "*schoon* en *schoonheid*",
            "right": "*lopen* en *liep*",
            "compound": "*deur* en *voordeur*",
        },
        correct="right",
        why={
            "right": "*liep* is de verleden tijd van hetzelfde werkwoord: inflectie. Een woordenboek heeft één ingang voor *lopen* en *liep*, en deze les brengt zulke vormen samen.",
            "derive": "*-heid* maakt een nieuw woord, een zelfstandig naamwoord van een bijvoeglijk naamwoord: dat is derivatie. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
            "compound": "*voordeur* voegt twee woorden samen tot één: een samenstelling. Zie *Drie manieren om woorden te maken* in [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl).",
        },
    )
    quick_inflection
    return (quick_inflection,)


@app.cell(hide_code=True)
def quick_check_fold():
    quick_fold = L.choice(
        "quick-fold",
        question="Fold, uit *Normaliseren: hoofdletters, accenten en stop words*, maakt een woord klein en haalt de accenten eraf. Van welk paar maakt fold één string?",
        options={
            "right": "*één* (het getal) en *een*",
            "plural": "*Cafés* en *cafe*",
            "niet": "*niet* en *niets*, twee stop words",
        },
        correct="right",
        why={
            "right": "Zonder accenten is *één*, het getal, het lidwoord *een*. Fold verandert alleen letters, en knipt niets van een woord af: dat doet deze les.",
            "plural": "Fold maakt *cafes* van *Cafés*: kleine letters en geen accent. Maar de *s* van het meervoud blijft, dus *cafes* en *cafe* zijn nog twee strings. Zie *Accenten* in [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
            "niet": "Fold verandert letters, geen woorden: *niet* en *niets* blijven twee strings. Zie *Accenten* in [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl).",
        },
    )
    quick_fold
    return (quick_fold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een woord heeft veel vormen. In het Nederlands *lopen*, *loopt*, *liep* en *gelopen*; in het Engels *walk*, *walks*, *walked* en *walking*. Voor een programma dat strings vergelijkt, zijn dat vier verschillende woorden: wie zoekt op *gelopen*, mist een tekst met *liep*, en een programma dat woorden telt, telt één woord als vier. Christopher Manning, Prabhakar Raghavan en Hinrich Schütze schrijven in het hoofdstuk *The term vocabulary and postings lists* van *Introduction to Information Retrieval* dat teksten om grammaticale redenen verschillende vormen van een woord gebruiken, zoals *organize*, *organizes* en *organizing*. Deze les brengt de vormen van een woord op twee manieren samen. Een **stemmer** hakt uitgangen eraf met regels, en een **lemmatizer** zoekt de vorm op die een woordenboek gebruikt. Beide aanpakken maken van veel woorden minder woorden, en elke aanpak maakt eigen fouten: de stemmer van Martin Porter uit 1980 geeft *university* en *universe* dezelfde stam, en houdt *ran* apart van *run*.

    **Een voorbeeld uit het dagelijks leven.** Een webwinkel die *hardloopschoenen* laat zien als je zoekt op *hardloopschoen*, heeft het meervoud en het enkelvoud samengebracht.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="UD_English-EWT, een verzameling Engelse zinnen van het web, heeft 12.415 verschillende woordvormen, zonder namen, getallen en leestekens. De stemmer van Porter geeft elke vorm een stam. Hoeveel verschillende stammen komen eruit?",
        options={
            "few": "Ongeveer 4.000",
            "right": "8.206",
            "most": "Ongeveer 11.000",
            "all": "Bijna 12.000",
        },
        reveal="**8.206**: 33,9% minder dan de 12.415 vormen. Porter vond in 1980 ongeveer hetzelfde, op een lijst van 10.000 woorden: in *An algorithm for suffix stripping* schrijft Porter dat het weghalen van de uitgangen de woordenschat met ongeveer een derde kleiner maakte. Gegroepeerd op de vorm uit het woordenboek, die EWT bij elk woord noteert, maken dezelfde vormen 8.745 groepen. De stemmer maakt dus nog minder groepen dan er lemma's zijn: de stemmer brengt ook vormen van verschillende woorden samen, zoals de stukken hieronder laten zien.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Eén woord, veel vormen

    [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl) knipte woorden in morfemen. Inflectie maakt de vormen van één woord, zoals *liep* van *lopen*; derivatie maakt een nieuw woord, zoals *schoonheid* van *schoon*. Manning en collega's schrijven dat stemming en lemmatization hetzelfde doel hebben: de vormen die inflectie maakt, en soms ook woorden die door derivatie verwant zijn, terugbrengen tot één gedeelde basisvorm. Hun voorbeelden: *am*, *are* en *is* worden *be*, en *car*, *cars*, *car’s* en *cars’* worden *car*.

    De les telt woorden in twee **treebanks**: verzamelingen zinnen waarin bij elk woord staat wat mensen erover opschreven, onder meer de vorm die een woordenboek gebruikt. Beide komen van Universal Dependencies (UD), een project dat treebanks van veel talen in één formaat verzamelt, versie 2.19 (november 2026). *UD_English-EWT* heeft zinnen van het web: blogs, nieuwsgroepen, recensies, en vragen met de antwoorden erop; de course liet de e-mails weg, want daarin staan echte namen en adressen. *UD_Dutch-Alpino* heeft zinnen uit Nederlandse kranten en ander geschreven Nederlands. De les telt elk woord in kleine letters, behalve namen, getallen, leestekens, symbolen en de woorden die de treebank als overig labelt: 12.415 verschillende vormen in EWT en 19.982 in Alpino. De les *Corpora en treebanks* vertelt meer over treebanks.

    ### Stemming: hakken met regels

    Volgens Manning en collega's is stemming meestal een grof proces met vuistregels: het proces hakt de einden van woorden eraf, in de hoop dat de uitkomst meestal klopt, en haalt vaak ook affixen van derivatie weg. Een vuistregel werkt vaak, niet altijd. Wat een stemmer van een woord overlaat, is de **stam**. Het meest gebruikte algoritme voor stemming van het Engels is volgens Manning en collega's dat van Porter, en keer op keer blijkt het goed te werken. Martin Porter schreef het in 1979 in het Computer Laboratory in Cambridge, Engeland, als deel van een groter project over information retrieval, IR: het vakgebied van zoekmachines. In *An algorithm for suffix stripping* zegt Porter waar de regels voor dienen: om IR beter te laten werken, niet als oefening in taalkunde.

    De stemmer van Porter doorloopt vijf stappen, en elke stap heeft groepen regels. Een regel heeft de vorm (voorwaarde) S1 → S2: als een woord eindigt op de uitgang S1, en de stam vóór S1 voldoet aan de voorwaarde, dan komt S2 in de plaats van S1. De eerste groep, stap 1a, heeft geen voorwaarden:

    | regel | voorbeeld |
    |---|---|
    | SSES → SS | *caresses* → *caress* |
    | IES → I | *ponies* → *poni* |
    | SS → SS | *caress* → *caress* |
    | S → | *cats* → *cat* |

    *caresses* eindigt op *sses*, maar ook op *ss* en op *s*. Van regels die onder elkaar staan, werkt er volgens Porter maar één: de regel met de langste uitgang die bij het woord past. Dus *caresses* neemt de eerste regel. De regel SS → SS verandert niets: de regel staat er zodat *caress* deze regel neemt, en niet de regel S →, die *cares* van *caress* zou maken. Latere stappen halen lange uitgangen stukje voor stukje weg: Porter laat zien hoe *generalizations* in stap 1 *generalization* wordt, in stap 2 *generalize*, in stap 3 *general* en in stap 4 *gener*.

    Een stam hoeft geen woord te zijn: *ponies* wordt *poni*. Mensen zien dat vaak als een fout, zegt de pagina van Porter over de stemmer, maar het doel van stemming is de vormen van een woord samen te brengen, niet een woord om te zetten in de vorm uit het woordenboek. Manning en collega's zeggen het zo: welke vorm de stam precies heeft, maakt niet uit; het gaat om de groepen woorden die dezelfde stam krijgen.

    ### De measure m

    De meeste latere regels hebben een voorwaarde op wat er vóór de uitgang overblijft. Voor Porter is een medeklinker elke letter behalve a, e, i, o en u, en behalve een y na een medeklinker. Elke andere letter is een klinker: in *toy* zijn de t en de y medeklinkers, in *syzygy* de s, de z en de g. De **measure** m van een woord, of van een deel van een woord, telt hoe vaak een medeklinker op een klinker volgt. De voorbeelden van Porter: m = 0 voor *tr*, *ee*, *tree*, *y* en *by*; m = 1 voor *trouble*, *oats*, *trees* en *ivy*; m = 2 voor *troubles*, *private*, *oaten* en *orrery*. In *trouble* volgt alleen op de *ou* een medeklinker, de *b*. In *ivy* staat de *y* na een medeklinker, dus is de *y* een klinker, en volgt alleen op de *i* een medeklinker.

    Een regel met de voorwaarde (m > 1) werkt alleen als wat zou overblijven een m boven 1 heeft. Het voorbeeld van Porter is de regel (m > 1) EMENT →, die *ement* weghaalt: van *replacement* maakt de regel *replac*, want *replac* heeft m = 2. Manning en collega's voegen eraan toe dat de regel van *cement* geen *c* maakt: *c* heeft m = 0. De measure kijkt volgens Manning en collega's grofweg naar het aantal lettergrepen, om te zien of een woord lang genoeg is om het passende stuk als uitgang te zien en niet als deel van de stam. Porter zegt er zelf bij dat deze aanpak geen taalkundige basis heeft.

    ### Lemmatization: opzoeken

    Lemmatization doet het volgens Manning en collega's netjes: met een woordenlijst en een analyse van de opbouw van een woord, meestal om alleen de uitgangen van inflectie weg te halen en de basisvorm of woordenboekvorm van een woord terug te geven, het **lemma**. Daniel Jurafsky en James Martin geven in het hoofdstuk *Embeddings* van *Speech and Language Processing* een voorbeeld: *mouse* is ook het lemma van *mice*, want een woordenboek heeft geen aparte ingang voor een vorm als *mice*. Zo is *sing* het lemma van *sing*, *sang* en *sung*. Een programma dat lemma's vindt, is een **lemmatizer**. Een lemmatizer brengt samen wat geen regel voor uitgangen kan samenbrengen: *liep* en *lopen*, *ging* en *gaan*.

    Beide treebanks hebben een kolom LEMMA. Volgens de richtlijnen van UD staat daar de standaardvorm of basisvorm van een woord, de vorm die je meestal in een woordenboek vindt. Een lemma haalt geen derivatie weg: het lemma van *organizations* is *organization*, niet *organize* en ook niet *organ*. De stemmer van Porter maakt *organ* van *organization*. In het Nederlands is het lemma van een werkwoord de infinitief, zoals *lopen* voor *liep*. Volgens de pagina van EWT gaf een programma de lemma's, met wat verbeteringen door mensen, maar zonder dat mensen alles nakeken: sommige lemma's kunnen dus fout zijn.

    Een lemma kan van de zin afhangen. Manning en collega's: voor het token *saw* geeft stemming misschien alleen *s*, terwijl lemmatization *see* of *saw* probeert te geven, afhankelijk van de vraag of *saw* een werkwoord of een zelfstandig naamwoord is. De stemmer van Porter laat *saw* staan. In EWT heeft *saw* 25 keer het lemma *see* en één keer *saw*, en *best* 246 keer het lemma *good* en 24 keer *well*. In Alpino heeft *liep* 17 keer het lemma *lopen* en 7 keer *op_lopen*: Alpino plakt het losse deel van een scheidbaar werkwoord, zoals de *op* van *liep op*, aan het lemma vast.

    ### Te veel en te weinig

    Wessel Kraaij en Renée Pohlmann, die een Nederlandse versie van de stemmer van Porter maakten, noemen in *Porter's stemming algorithm for Dutch* (1994) de twee soorten fouten. Bij overstemming brengt de stemmer woorden samen die in betekenis niets met elkaar te maken hebben; bij understemming laat de stemmer woorden uit elkaar die in betekenis wel bij elkaar horen. **Overstemming**: de stemmer van Porter geeft *university* en *universe* de stam *univers*, en *general* en *generous* de stam *gener*; in een opgave noemen Manning en collega's *university* en *universe* bij de paren die de stemmer van Porter samenbrengt, en vragen welke paren niet bij elkaar horen. Porter zag het al aankomen: als *sand* en *sander* samenkomen, schrijft Porter, dan waarschijnlijk ook *wand* en *wander*. **Understemming**: *ran* houdt de stam *ran*, en *went* de stam *went*. Porter koos ervoor woorden waarvan de stam verandert, zoals *deceive* en *deception*, niet te behandelen: gezien het aantal fouten dat je toch al moet verwachten, leek dat de moeite niet waard. En wie de ene soort fout herstelt, krijgt meer van de andere, schrijven Kraaij en Pohlmann: regels die understemming verminderen, geven waarschijnlijk meer overstemming.

    Voor een zoekmachine vindt een stemmer meer van de teksten die de zoekmachine moet vinden, en ook meer teksten die niet gevonden hoeven te worden. Het voorbeeld van Manning en collega's: de stemmer van Porter geeft *operate*, *operating*, *operational* en vier andere vormen de stam *oper*, dus vindt een zoekopdracht naar *operating system* ook teksten met *operative* en *system* erin. Voor het Engels helpt stemming of lemmatization al met al niet veel, schrijven Manning en collega's: voor sommige zoekopdrachten helpt het veel, maar voor andere schaadt het net zo veel. Voor talen met veel meer morfologie, zoals Spaans, Duits en Fins, ligt dat anders: in de Europese evaluaties van CLEF gaven stemmers keer op keer flinke winst.

    ### Nederlands

    Een Nederlands werkwoord heeft veel vormen: *lopen*, *loop*, *loopt*, *liep*, *liepen*, *lopend* en *gelopen*. De spelling maakt het lastiger voor regels, merken Kraaij en Pohlmann op: een lange klinker schrijf je enkel in een open lettergreep en dubbel in een gesloten lettergreep, zoals in *schapen* en *schaap*. Dus schrijf je *loopt* met *oo* en *lopen* met één *o*. De course gebruikt de stemmer die Porter zelf voor het Nederlands schreef, zoals het project Snowball deze stemmer beschrijft, en niet die van Kraaij en Pohlmann. De stemmer haalt uitgangen weg zoals *-en*, *-e*, *-heid*, *-ing* en *-lijk*, en maakt een dubbele klinker vóór een laatste medeklinker enkel, zoals in *maan* → *man*. Zo krijgt *lopen* de stam *lop*, maar *loopt* blijft *loopt*, *gelopen* wordt *gelop*, en *liep* blijft *liep*: vier verschillende stammen voor één werkwoord.

    Sinds versie 3.0.0 (2025) is de standaard van Snowball voor het Nederlands de stemmer van Kraaij en Pohlmann, naar aanleiding van opmerkingen van Nederlandse gebruikers. Deze stemmer geeft *lopen*, *loopt* en *gelopen* de stam *loop*, en gaat verder: de stemmer haalt *ge* weg aan het begin en midden in een woord, zodat *luidvoel* overblijft van *geluidgevoelige*. De course gebruikt de stemmer van Porter, omdat de regels daarvan stap voor stap beschreven zijn; de stemmer van Kraaij en Pohlmann is volgens Snowball eigenlijk alleen vastgelegd in code, in ANSI C. Geen van beide stemmers zet *liep* bij *lopen*. Over werkwoorden zoals *drinken*, *dronk* en *gedronken* schrijven Kraaij en Pohlmann dat een eenvoudige stemmer de vormen nooit naar één stam kan brengen.

    Let op het woord stam. De schoolgrammatica kent de stam van een werkwoord: volgens de Woordenlijst is dat de infinitief zoals je die uitspreekt, zonder de uitgang, dus de stam van *lopen* is *loop*. Een stam in deze les is wat een stemmer overlaat, zoals *lop*.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **stemming** | uitgangen van woorden afhakken met regels, zodat de vormen van een woord hetzelfde uitkomen |
        | **stemmer** | een programma dat stemming toepast, zoals de stemmer van Porter |
        | **stam** | wat een stemmer van een woord overlaat, zoals *poni* voor *ponies*; dat hoeft geen woord te zijn |
        | **lemma** | de vorm van een woord die een woordenboek gebruikt: *lopen* voor *liep*, *kind* voor *kinderen*, *run* voor *ran* |
        | **lemmatization**, **lemmatizer** | het lemma van elk woord vinden, met een woordenboek of een tabel; een programma dat dat doet |
        | **overstemming**, **understemming** | een stemmer brengt vormen van verschillende woorden samen, zoals *university* en *universe*; een stemmer laat vormen van één woord uit elkaar, zoals *liep* en *lopen* |
        | **measure** m | de telling van Porter van hoe vaak een medeklinker op een klinker volgt: 0 voor *tree*, 1 voor *trouble*, 2 voor *troubles* |
        | **treebank** | een verzameling zinnen waarin bij elk woord staat wat mensen erover opschreven, zoals het lemma |
        | `word.endswith(suffix)` | `True` als de string `word` eindigt op de string `suffix`: `"cats".endswith("s")` is `True` |
        | `for suffix in rules` | een `for` over een dictionary gaat de keys langs, in de volgorde waarin de keys erin kwamen |
        | `a and not b`, `a or b` | `not` maakt van `True` `False` en andersom; `or` is `True` als minstens één kant `True` is |
        """),
        L.watch_out("*Stemming* is hier het Engelse woord uit de zoektechniek, niet de stemming bij een verkiezing of de stemming in een zaal; schrijf dus ook niet dat je woorden stemt, maar dat je stemming toepast. En een lemma in de wiskunde is iets anders: een kleine stelling op weg naar een grotere."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Vraag wat de taak nodig heeft: alleen groepen vormen die bij elkaar horen, of woorden die een mens leest?
    2. Voor groepen: geef elk woord een stam, met een stemmer voor de taal, en vergelijk de stammen.
    3. Voor de vorm uit het woordenboek: zoek elk woord op in een tabel met lemma's; een woord dat er niet in staat, blijft zoals het is.
    4. Tel wat de stap samenbrengt dat niet bij elkaar hoort, en wat de stap uit elkaar laat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt *ponies* met de hand door stap 1a: op welke uitgangen het woord eindigt, hoe lang het woord en de uitgang zijn, en wat er overblijft met de nieuwe uitgang.
    """)
    return


@app.cell
def example_ponies():
    W = "ponies"
    print(W.endswith("sses"), W.endswith("ies"), W.endswith("s"))
    print(len(W), len("ies"))
    print(W[:3] + "i")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *ponies* eindigt op *ies* en op *s*, en de langere uitgang beslist: de regel IES → I. Van de zes letters blijven de drie vóór de uitgang staan, en daarachter komt *i*: *poni*. Oefening 1 doet dit voor elk woord en elke groep regels.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rules():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. apply_rules: één groep regels

        `STEP_1A` hieronder bevat stap 1a als dictionary: elke uitgang met de nieuwe uitgang, de langste uitgang eerst, zodat de eerste regel die past de regel is die Porter neemt. Schrijf `apply_rules(word, rules)`: ga de uitgangen van `rules` in volgorde langs, en geef voor de eerste uitgang waarop `word` eindigt het woord zonder die uitgang terug, met de nieuwe uitgang van de regel eraan vast. Als geen uitgang past, geef dan `word` onveranderd terug. Voor `apply_rules("caresses", STEP_1A)` is de uitkomst `"caress"`. Vervang de `...`. De controle roept dat aan, en daarna eigen gevallen: de andere regels van stap 1a, een woord waar geen regel bij past met een *s* erin, een woord dat met een *s* begint, twee regels van stap 2 waarbij de langere uitgang eerst moet, Nederlandse regels, geen enkele regel, en een leeg woord.
        """),
        L.task("Pas de eerste regel van een groep toe waarvan de uitgang het woord afsluit: knip de uitgang eraf en plak de nieuwe uitgang eraan."),
    ])
    return


@app.cell
def step_1a_cell():
    STEP_1A = {"sses": "ss", "ies": "i", "ss": "ss", "s": ""}
    print(list(STEP_1A), STEP_1A["ies"])
    return


@app.cell(hide_code=True)
def exercise_rules():
    box_rules = L.code_box("apply-rules", lines=5)
    box_rules
    return (box_rules,)


@app.cell(hide_code=True)
def check_rules(box_rules):
    result_rules = L.check("apply-rules", box_rules.value)
    result_rules
    return (result_rules,)


@app.cell(hide_code=True)
def hints_rules():
    L.hints(
        "apply-rules",
        [
            "Ga de uitgangen langs met `for suffix in rules`, en vraag `word.endswith(suffix)`. Geef voor de eerste uitgang die past het deel van `word` ervoor terug, met `rules[suffix]` eraan vast; geef na de loop `word` terug.",
            "Bij *ponies* en *ies* blijven de eerste drie letters staan. Hoe krijg je die drie uit het woord en de uitgang, voor elk woord?",
            r'`            return word[:___] + rules[suffix]`',
        ],
        why=r'`for suffix in rules:`, `if word.endswith(suffix):`, `return word[:len(word) - len(suffix)] + rules[suffix]`, en na de loop `return word`.',
    )
    return


@app.cell(hide_code=True)
def text_after_rules():
    mo.md(r"""
    Met je apply_rules en `STEP_1A` veranderen 2.563 van de 12.415 vormen van EWT, en blijven er 10.854 groepen over. Vormen zoals *time* en *times* komen samen, en de treebank is het daarmee eens. Maar stap 1a knipt de *s* ook af van woorden die geen meervoud zijn: *is* wordt *i*, *as* wordt *a*, en *its* komt bij *it*. Het programma van Porter zelf laat volgens de pagina over de stemmer woorden van één of twee letters met rust, omdat de stemmer anders van *as* een *a* zou maken en van *is* een *i*. De stemmer van de course doet hetzelfde. En stap 1a is pas de eerste stap: *running* staat nog los van *run*, tot stap 1b *ing* eraf haalt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measure():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. measure: de m van Porter

        `vowel(word, i)` hieronder geeft `True` als de letter op plek `i` van `word` voor Porter een klinker is: *a*, *e*, *i*, *o* of *u*, of een *y* na een medeklinker. De loop gaat de letters tot en met plek `i` langs, en onthoudt in `is_vowel` of de letter ervoor een klinker was. Schrijf `measure(word)`: tel de plekken in het woord waar een medeklinker op een klinker volgt, zoals `vowel` het ziet, en geef dat aantal terug. Voor `measure("trouble")` is de uitkomst `1`. Vervang de `...`. De controle roept dat aan, en daarna eigen gevallen: de andere voorbeelden uit de paper van Porter, een *y* na een klinker en na een medeklinker, *crepuscular*, waarvoor Snowball m = 4 geeft, een kort woord, en een leeg woord.
        """),
        L.task("Tel hoe vaak een medeklinker op een klinker volgt, met de klinkers van Porter."),
    ])
    return


@app.cell
def vowel_cell():
    def vowel(word, i):
        is_vowel = False
        for j in range(i + 1):
            is_vowel = word[j] in "aeiou" or (word[j] == "y" and j > 0 and not is_vowel)
        return is_vowel


    print([vowel("toy", i) for i in range(3)])
    print([vowel("syzygy", i) for i in range(6)])
    return


@app.cell(hide_code=True)
def exercise_measure():
    box_measure = L.code_box("measure", lines=6)
    box_measure
    return (box_measure,)


@app.cell(hide_code=True)
def check_measure(box_measure):
    result_measure = L.check("measure", box_measure.value)
    result_measure
    return (result_measure,)


@app.cell(hide_code=True)
def hints_measure():
    L.hints(
        "measure",
        [
            "Tel met `m`: begin met `m = 0`, ga de plekken langs met `for i in range(1, len(word))`, en tel er één bij met `m = m + 1` waar een medeklinker op een klinker volgt. Geef `m` terug na de loop.",
            "Op plek `i` staat de medeklinker. Op welke plek staat de klinker ervoor, en hoe vraag je `vowel` naar die plek?",
            r'`        if ___ and not vowel(word, i):`',
        ],
        why=r'`m = 0`, `for i in range(1, len(word)):`, `if vowel(word, i - 1) and not vowel(word, i):` `m = m + 1`, en na de loop `return m`.',
    )
    return


@app.cell(hide_code=True)
def text_after_measure():
    mo.md(r"""
    Met je measure haalt de regel (m > 1) EMENT → *ement* weg uit *replacement* en *management*, waar m van wat overblijft 2 is. De regel laat *statement*, *agreement* en *settlement* met rust, waar m van wat zou overblijven 1 is, en *cement* ook. De hele stemmer van Porter met jouw measure geeft elke vorm van EWT dezelfde stam als met de measure van de course, en maakt de 8.206 groepen uit de gok. De pagina van Porter waarschuwt voor een veelgemaakte fout in programma's die de stemmer namaken: in een groep regels beslist de regel met de langste uitgang, of die regel nu iets verandert of niet. Zo blijft *argument* volgens de pagina *argument*: *ment* is de langste uitgang die past, *argu* heeft m = 1, dus *ment* blijft staan, en een kortere uitgang komt niet meer aan de beurt. Een programma dat daarna nog *ent* probeerde, zou er *argum* van maken.
    """)
    return


@app.cell(hide_code=True)
def compare_cell():
    mo.Html(C.compare_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_compare():
    mo.md(r"""
    Over heel EWT maakt de stemmer van Porter 8.206 groepen van de 12.415 vormen, en de lemma's maken er 8.745. 1.013 stammen brengen vormen van meer dan één lemma samen, zoals *use* en *us*, dat in EWT het lemma *we* heeft, en *business* en *busy*. 515 lemma's hebben vormen onder meer dan één stam, zoals *go*, *going*, *went* en *gone*. In het Nederlands gaat het andersom: de stemmer van Porter voor het Nederlands maakt 16.561 groepen van de 19.982 vormen van Alpino, meer dan de 14.928 lemma's. Minder stammen brengen vormen van meer dan één lemma samen, 899, zoals *zaal* en *zal*, en 1.789 lemma's hebben vormen onder meer dan één stam, zoals *werken*, *werkt*, *werkte* en *gewerkt*: de stemmer haalt *-en* en *-e* weg, maar laat de *t* van *werkt* en de *ge-* van *gewerkt* staan. De pagina van Porter zegt dat de oorspronkelijke stemmer voor het Engels als bevroren moet gelden, en raadt voor praktisch werk de nieuwere stemmer van Snowball aan. Die nieuwere stemmer voor het Engels verandert nog steeds: sinds Snowball 3.0.0 brengt de stemmer *university* en *universe* niet meer samen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lemmatize():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. lemmatize: opzoeken

        Een treebank bestaat uit delen, zodat je een programma uit het ene deel kunt maken, het trainingsdeel, en op een ander deel kunt uitproberen, het testdeel. `TABLE` hieronder bevat de tabel met lemma's van de course voor het Engels: elk van de 11.115 verschillende vormen in het trainingsdeel van EWT, met het meest voorkomende lemma daar. `TABLE_NL` bevat hetzelfde voor de 18.542 vormen in het trainingsdeel van Alpino. Schrijf `lemmatize(word, table)`: als `word` een key van de dictionary `table` is, geef dan de bijbehorende waarde terug, het lemma; geef anders `word` zelf terug, want van veel woorden is het woord zelf het lemma. Voor `lemmatize("liep", TABLE_NL)` is de uitkomst `"lopen"`. Vervang de `...`. De controle roept dat aan, en daarna eigen gevallen: Engelse en Nederlandse woorden uit de tabellen, een woord dat geen tabel heeft, een woord dat zelf het lemma is, een eigen tabel met een woord dat erin staat en een woord dat er niet in staat, en een lege tabel.
        """),
        L.task("Zoek een woord op in een tabel met lemma's, en houd het woord zelf als de tabel het woord niet heeft."),
    ])
    return


@app.cell
def table_cell():
    TABLE = C.table("en")
    TABLE_NL = C.table("nl")
    print(len(TABLE), len(TABLE_NL))
    print(TABLE["feet"], TABLE["went"], TABLE_NL["steden"])
    return


@app.cell(hide_code=True)
def exercise_lemmatize():
    box_lemmatize = L.code_box("lemmatize", lines=4)
    box_lemmatize
    return (box_lemmatize,)


@app.cell(hide_code=True)
def check_lemmatize(box_lemmatize):
    result_lemmatize = L.check("lemmatize", box_lemmatize.value)
    result_lemmatize
    return (result_lemmatize,)


@app.cell(hide_code=True)
def hints_lemmatize():
    L.hints(
        "lemmatize",
        [
            "Twee dingen die je van dictionaries weet: `table[word]` geeft de waarde van een key, en vragen naar een key die er niet in staat, stopt het programma met een fout. Dus eerst één `if` met een vraag over `table`, en dan twee wegen terug: het lemma, of `word`.",
            "Welke vraag over de tabel vertelt je of `table[word]` veilig is?",
            r'`    if ___:`',
        ],
        why=r'`if word in table:` geeft `table[word]` terug; anders `return word`.',
    )
    return


@app.cell(hide_code=True)
def text_after_lemmatize():
    mo.md(r"""
    Het testdeel van elke treebank bevat woorden waar de tabellen niet van gemaakt zijn: 14.801 Engelse en 8.912 Nederlandse. Daarop geeft je lemmatize voor 96,6% van de Engelse woorden het lemma van de treebank, en voor 90,5% van de Nederlandse. Van de Engelse woorden staat 94,7% in `TABLE`, maar van de Nederlandse maar 88,7% in `TABLE_NL`: het Nederlands schrijft een samenstelling als één woord, en een nieuwe samenstelling staat in geen enkele tabel. Alpino geeft in de lemma's de delen van een samenstelling aan met `_`: het lemma van *jeugdgezondheidszorg* is *jeugd_gezondheid_zorg*. Van de 1.008 Nederlandse woorden die niet in `TABLE_NL` staan, hebben er 484 zo'n lemma; van de 784 Engelse woorden die niet in `TABLE` staan, 2. Voor die woorden is het woord zelf nooit goed. Alles bij elkaar is het woord zelf goed voor 469 van de 784 in het Engels en voor 272 van de 1.008 in het Nederlands. En een tabel geeft elke vorm één lemma, het meest voorkomende: *best* krijgt *good*, ook waar de treebank *well* zegt, en *vindt* krijgt *vinden*, ook waar Alpino *plaats_vinden* zegt, voor *vindt plaats*. Een lemmatizer die naar de zin rond een woord kijkt, kan zulke gevallen uit elkaar houden.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: stam en lemma van je eigen woorden

    Typ een paar woorden en kies een taal. De afbeelding geeft elk woord een stam, van de stemmer van Porter voor die taal, en de lemma's die de treebank het woord geeft, met hoe vaak. Probeer *ging*, *gegaan* en *zag*, of Engelse woorden zoals *ran*, *mice* en *better*.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value="Ik loop, Tom loopt, wij liepen, en jullie hebben gelopen.", label="Woorden", full_width=True)
    demo_lang = mo.ui.dropdown(options={"Nederlands": "nl", "Engels": "en"}, value="Nederlands", label="Taal")
    mo.vstack([demo_text, demo_lang])
    return demo_lang, demo_text


@app.cell(hide_code=True)
def demo_look(demo_lang, demo_text):
    mo.Html(C.demo_picture(demo_text.value, demo_lang.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    In *Ik loop, Tom loopt, wij liepen, en jullie hebben gelopen.* krijgt *loop* de stam *lop*, net als *lopen*, maar *loopt* houdt de *t*, *liepen* wordt *liep*, en *gelopen* wordt *gelop*: vier verschillende stammen voor de vormen van één werkwoord. De treebank geeft *loopt*, *liepen* en *gelopen* het lemma *lopen*, en *loop* meestal het lemma *loop*: in Alpino is *loop* vaker het zelfstandig naamwoord. En *hebben* wordt *hebb*: de stemmer haalt *-en* weg, en laat beide b's staan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Oefening 4. Stam, lemma of geen van beide?

    Elke vetgedrukte taak vraagt iets van een programma dat met woorden werkt. Zet elke taak bij wat de taak nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_stem = L.sort("stem-or-lemma")
    sort_stem
    return (sort_stem,)


@app.cell(hide_code=True)
def check_sort(sort_stem):
    result_sort = L.check("stem-or-lemma", sort_stem)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Normaliseren: hoofdletters, accenten en stop words](/course/open/nlp-normalize?lang=nl) maakte van *Café* en *cafe* één woord. Deze les ging verder met de vormen van één woord, wat [Woorden en hun delen](/course/open/nlp-morphemes?lang=nl) inflectie noemde. Een stemmer hakt uitgangen eraf met regels: je schreef een functie die een groep regels van Porter toepast, en de measure m waar latere regels naar vragen, en zag dat de stemmer van Porter *university* en *universe* samenbrengt en *ran* apart houdt van *run*. Een lemmatizer zoekt de vorm uit het woordenboek op: jouw lemmatizer gebruikte een tabel uit een treebank, en miste de nieuwe woorden die niet in de tabel stonden. Net als case folding gooien beide aanpakken iets weg: de *-te* die de verleden tijd aangeeft, de *-en* van het meervoud. De volgende les van het domein, *Hoe ver liggen twee woorden uit elkaar? Edit distance*, vergelijkt twee woorden letter voor letter, en telt hoe ver het ene woord van het andere ligt, zoals een tikfout van het bedoelde woord.

    **Een voorbeeld uit het dagelijks leven.** Een bibliotheekcatalogus die een boek over *tuinieren* vindt als je *tuinier* typt, heeft beide woorden dezelfde stam gegeven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_lemmatize, result_measure, result_rules, result_sort):
    L.wrap_up(result_rules, result_measure, result_lemmatize, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
