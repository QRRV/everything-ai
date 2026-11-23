# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-dutch-tools. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dutch-tools", "nl")
    C = L.checks   # de zinnen van de course, de treebanks, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Corpora en treebanks](/course/open/nlp-corpora?lang=nl), en een over [Stam en lemma](/course/open/nlp-stemming?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_particles():
    quick_particles = L.choice(
        "quick-particles",
        question="In [Corpora en treebanks](/course/open/nlp-corpora?lang=nl) gaf `particles(sentence)` een paar `(head, i)` voor elk woord met de relatie `compound:prt`: voor *We haalden de kinderen op en zetten ze af* `[(2, 5), (7, 9)]`. Wat geeft de functie voor *Pak het op*, met de woorden *Pak*, *het* en *op*?",
        options={
            "turned": "`[(3, 1)]`",
            "words": "`[('Pak', 'op')]`",
            "right": "`[(1, 3)]`",
            "object": "`[(1, 2)]`",
        },
        correct="right",
        why={
            "right": "*op* is woord 3, en het hoofdwoord ervan, het werkwoord *Pak*, is woord 1: het paar `(1, 3)`, met het werkwoord eerst, zoals in `(2, 5)` voor *haalden* en *op*. Hier staat het partikel twee plaatsen na het werkwoord. Deze les meet in de hele treebank waar partikels staan.",
            "turned": "Het paar is `(head, i)`: eerst het nummer van het werkwoord, dan dat van het partikel. *Pak* is woord 1 en *op* woord 3. Zie Oefening 2 in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
            "words": "`particles` geeft de nummers van de woorden, niet de woorden zelf: *Pak* is woord 1 en *op* woord 3. Zie Oefening 2 in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
            "object": "Woord 2 is *het*, het lijdend voorwerp van *Pak*, met de relatie `obj`. Het partikel is *op*, woord 3. Zie Oefening 2 in [Corpora en treebanks](/course/open/nlp-corpora?lang=nl).",
        },
    )
    quick_particles
    return (quick_particles,)


@app.cell(hide_code=True)
def quick_check_lemma():
    quick_lemma = L.choice(
        "quick-lemma",
        question="In [Stam en lemma](/course/open/nlp-stemming?lang=nl) gaf de Nederlandse treebank *liep* 17 keer het lemma *lopen* en 7 keer *op_lopen*. Welk lemma geeft de treebank *liep* in *Het aantal liep snel op*?",
        options={
            "plain": "`lopen`",
            "right": "`op_lopen`",
            "participle": "`opgelopen`",
            "form": "`liep`",
        },
        correct="right",
        why={
            "right": "Hier horen *liep* en *op* bij elkaar: *oplopen*, stijgen. De lemma's van de treebank plakken het losse deel van een scheidbaar werkwoord, zoals *op*, met `_` aan het lemma vast. In deze les lees je zulke lemma's in de hele Nederlandse treebank.",
            "plain": "*lopen* is het lemma waar *liep* over wandelen gaat, zonder *op*. Hier hoort *op* bij *liep*, en de treebank plakt het aan het lemma vast. Zie [Stam en lemma](/course/open/nlp-stemming?lang=nl).",
            "participle": "*opgelopen* is een vorm van het werkwoord, het voltooid deelwoord. Een lemma is de vorm die in het woordenboek staat, en de treebank plakt de losse *op* eraan vast. Zie [Stam en lemma](/course/open/nlp-stemming?lang=nl).",
            "form": "Een lemma is de vorm die in het woordenboek staat, niet het woord zoals het in de zin staat. Zie [Stam en lemma](/course/open/nlp-stemming?lang=nl).",
        },
    )
    quick_lemma
    return (quick_lemma,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel van NLP is eerst voor het Engels gemaakt. Porter schreef een stemmer voor het Engels, en de richtlijnen van de Penn Treebank in *Corpora en treebanks* zijn voor Engelse zinnen geschreven. Het Nederlands doet dingen die regels voor het Engels niet verwachten. Een werkwoord als *opbellen* valt in een hoofdzin uit elkaar: *Ik bel je morgen op*. Een samenstelling schrijf je als één woord, zoals *grondwet*, en in een bijzin komt het werkwoord achteraan. Onderzoekers in Nederland en Vlaanderen hebben dus tools en corpora voor het Nederlands gebouwd: de eerste delen van de toolkit Frog stammen uit de jaren negentig, het Corpus Gesproken Nederlands is van 1998 tot 2004 verzameld, en de parser Alpino krijgt in 2026 nog nieuwe versies. Deze les laat de belangrijkste zien, en meet in een Nederlandse treebank wat een regel voor het Engels mist.

    **Een voorbeeld uit het dagelijks leven.** Een spellingcontrole die op Engels staat, onderstreept in een Nederlandse brief bijna elk woord. De controle is niet kapot; het is het verkeerde gereedschap voor deze tekst.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In de 11.723 zinnen van UD_English-EWT die de course gebruikt, een Engelse treebank van Universal Dependencies, staan 547 van de 724 partikels, zoals *up* in *pick up the phone*, direct na het werkwoord: 76 op de 100. Het Nederlands heeft ook zulke partikels: de *op* van *opbellen* in *Ik bel je morgen op*. Hoeveel op de 100 staan direct na het werkwoord in alle 13.603 zinnen van UD_Dutch-Alpino, de Nederlandse treebank van *Corpora en treebanks*?",
        options={
            "some": "Ongeveer 15",
            "same": "Ongeveer 75, zoals in het Engels",
            "right": "Ongeveer 8",
            "few": "Ongeveer 2",
        },
        reveal="**Ongeveer 8**: 151 van de 1.952 partikels in UD_Dutch-Alpino staan direct na het werkwoord, volgens de eigen telling van de course. 808 staan ervoor, zoals *terug* in *ik probeer je morgen terug te bellen*, en de rest verder erna, in één zin 32 plaatsen. Een regel voor het Engels, zoals *het partikel komt direct na het werkwoord*, mist er meer dan negen op de tien. In *Corpora en treebanks* stonden 300 van de 1.640 partikels direct na het werkwoord of één woord verder; hier zijn dat er 352 van de 1.952, en minder dan de helft daarvan staat direct erna.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Scheidbare werkwoorden

    Het Nederlands heeft veel **scheidbare werkwoorden**: werkwoorden als *opbellen*, gemaakt van een werkwoord, *bellen*, met een partikel ervoor, *op*. In een hoofdzin staat de persoonsvorm op de tweede plaats, en gaat het partikel naar het eind van de zin: *Ik bel je morgen op*. Staan werkwoord en partikel naast elkaar, dan schrijf je het geheel als één woord: *omdat ik je morgen opbel*, *opbellen*, *opgebeld*. In een bijzin is dat vaak zo, maar niet altijd: *omdat ik je morgen op wil bellen*. Het voltooid deelwoord krijgt de *ge-* na het partikel, en de infinitief met *te* krijgt de *te* ertussen: de naslaggrammatica van het Nederlands, de *Algemene Nederlandse Spraakkunst* (ANS), geeft *aangevallen* en niet *geaanvallen*, en *aan te vallen* en niet *te aanvallen*. Het Engels heeft ook partikels, *up* in *pick up*, maar een Engels partikel blijft dichtbij: direct na het werkwoord, of na een kort lijdend voorwerp, zoals in *pick it up*.

    ### Wat de Nederlandse treebank schrijft

    *Corpora en treebanks* liet zien dat Universal Dependencies een partikel de relatie `compound:prt` naar het werkwoord geeft. De Nederlandse treebank van deze les, UD_Dutch-Alpino, is aan de Rijksuniversiteit Groningen gelabeld met de tools en richtlijnen van de parser Alpino, en de lemma's ervan doen meer. Het partikel van een scheidbaar werkwoord zit vrijwel altijd met een underscore vast aan het lemma van het werkwoord: *bel* in *Ik bel je morgen op* heeft het lemma `op_bellen`. Het werkwoord houdt dat lemma als het als één woord in de zin staat: *opgelost* heeft `op_lossen`, zonder eigen partikel in de zin. Het lemma onderscheidt zelfs twee werkwoorden die er hetzelfde uitzien. *voorkomen* in de betekenis van verhinderen is niet scheidbaar, *dat voorkwam een ongeluk*, en heeft het lemma `voorkomen`; *voorkomen* in de betekenis van gebeuren is scheidbaar, *dat komt vaak voor*, en heeft `voor_komen`. De delen van een samengesteld zelfstandig naamwoord krijgen dezelfde markering, zoals *jeugd_gezondheid_zorg* in *Stam en lemma*: *grondwet* heeft `grond_wet`.

    UD geeft in het Nederlands ook `compound:prt` aan een woord dat met een werkwoord een vaste combinatie vormt: een zelfstandig naamwoord, zoals *plaats* in *de les vindt morgen plaats*, lemma `plaats_vinden`; een bijvoeglijk naamwoord, zoals *dicht* in *doe de deur dicht*; het voorzetsel van een vaste uitdrukking, zoals *in* in *in gebruik nemen*; en zelfs een werkwoord, zoals *zien* in *laten zien*. In de laatste twee houdt het werkwoord een eigen lemma, `nemen` en `laten`, zonder underscore. Deze les noemt al zulke woorden partikels, zoals UD doet. Hoe ver een partikel van het werkwoord staat, heet hier de **afstand**: het aantal plaatsen van het werkwoord tot het partikel, 1 als het partikel direct na het werkwoord komt, en negatief als het partikel ervoor komt. In *Ik bel je morgen op* is de afstand 3.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_tools():
    mo.md(r"""
    ### Tools voor het Nederlands

    - **Alpino** is een parser voor het Nederlands van de Rijksuniversiteit Groningen, van Gertjan van Noord, Gosse Bouma, Robert Malouf en anderen, begonnen in een onderzoeksproject van 2000 tot 2005. Anders dan de meeste parsers van nu begint Alpino bij een grammatica die met de hand is geschreven: in 2006 telde van Noord ongeveer 600 regels en een lexicon van ongeveer 100.000 woorden. Waar de grammatica meerdere analyses van een zin toelaat, kiest een statistisch model dat op een treebank is getraind er één. Alpino geeft elke zin een dependency-structuur; in de eigen uitvoer van Alpino krijgt *speelt ... op* (laait op) de grondvorm `speel_op`, die Alpino de stam noemt. Alpino is vrije software, onder de LGPL.
    - **Frog** is een toolkit voor het Nederlands, gemaakt aan Tilburg University en de Universiteit Antwerpen, verder ontwikkeld aan de Radboud Universiteit Nijmegen en nu onderhouden bij het KNAW Humanities Cluster. Frog knipt een tekst in zinnen en woorden, en voegt tags toe, lemma's, de delen van elk woord, een dependency tree, en namen van personen, plaatsen en organisaties. De delen van Frog leren van Nederlandse corpora: de tagger vooral van het Corpus Gesproken Nederlands, de parser van Lassy Klein en van tekst die Alpino heeft geparst, en het deel dat namen vindt, van SoNaR-1. Ook Frog is vrije software, onder de GPL.
    - spaCy, de library van [Een NLP-pipeline](/course/open/nlp-pipeline?lang=nl), heeft drie Nederlandse pipelines, `nl_core_news_sm`, `nl_core_news_md` en `nl_core_news_lg`, getraind op twee Nederlandse treebanks van UD; de kleinste heeft op de eigen test van spaCy 95,0% van de lemma's goed. Stanza, van Stanford University, heeft Nederlandse modellen die op dezelfde twee treebanks zijn getraind.

    ### Corpora van het Nederlands

    - **SoNaR** is een corpus van geschreven Nederlands uit Nederland en Vlaanderen, gebouwd van 2008 tot 2011: SoNaR-500 bevat meer dan 500 miljoen woorden uit veel soorten tekst, gekozen uit 1954 of later, met tags, lemma's en namen die programma's hebben toegevoegd. SoNaR-1, een miljoen woorden die er grotendeels uit komen, heeft labels die mensen hebben gecontroleerd, zoals welke woorden namen van personen of plaatsen zijn.
    - Het **Corpus Gesproken Nederlands** (CGN) bevat ongeveer 900 uur gesproken Nederlands, 8.940.098 woorden, verzameld van 1998 tot 2004 in Nederland en Vlaanderen: gesprekken, telefoongesprekken, lessen, nieuws, voorgelezen teksten. Mensen schreven elk woord uit, elk woord heeft een tag en een lemma, en een deel heeft bomen.
    - **Lassy** is de naam van twee treebanks van geschreven Nederlands. **Lassy Klein** heeft ongeveer een miljoen woorden, 65.200 zinnen, met bomen die mensen hebben gecontroleerd; **Lassy Groot** ongeveer 700 miljoen woorden, door Alpino geparst zonder controle. Beoordelaars van de Universiteit van Kopenhagen controleerden 500 zinnen van Lassy Klein en 250 van Lassy Groot: de analyse was goed voor 97,8% en voor 78,4% van de zinnen.
    - Het Instituut voor de Nederlandse Taal geeft deze drie gratis uit, voor onderzoek en onderwijs, niet voor commercieel gebruik: je logt in en accepteert een licentie, en voor het CGN teken je een licentie. Open Dutch WordNet, uit *Corpora en treebanks*, staat gratis op GitHub.

    De treebanks van UD zijn gratis te downloaden: UD_Dutch-Alpino, met krantenzinnen uit het Eindhovencorpus, vragen, zinnen om de grammatica van Alpino te testen, voorbeeldzinnen van de ANS en een deel van Lassy Klein, in totaal 13.603 zinnen; en UD_Dutch-LassySmall, uit Lassy Klein. Ook de language models van latere lessen leren van corpora: BERTje, een Nederlands model uit 2019, leerde deels van SoNaR-500, en de makers lieten de teksten weg die ook in SoNaR-1 en Lassy Klein staan, de delen die mensen hebben gecontroleerd.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **scheidbaar werkwoord** | een werkwoord als *opbellen* waarvan het partikel in een hoofdzin los staat: *Ik bel je op* |
        | **afstand** | in deze les: hoeveel plaatsen een partikel na het werkwoord staat; negatief als het partikel ervoor staat |
        | **Alpino** | een parser voor het Nederlands van de Rijksuniversiteit Groningen, gebouwd op een grammatica die met de hand is geschreven |
        | **Frog** | een toolkit voor het Nederlands die tags, lemma's, de delen van woorden, dependency trees en namen toevoegt |
        | **SoNaR** | een corpus van geschreven Nederlands: SoNaR-500, meer dan 500 miljoen woorden, en SoNaR-1, een miljoen woorden met labels die mensen hebben gecontroleerd |
        | **Corpus Gesproken Nederlands** | ongeveer 900 uur gesproken Nederlands uit Nederland en Vlaanderen, uitgeschreven; afgekort CGN |
        | **Lassy** | twee treebanks van geschreven Nederlands: Lassy Klein, ongeveer een miljoen woorden die mensen hebben gecontroleerd, en Lassy Groot, ongeveer 700 miljoen woorden die Alpino heeft geparst |
        | `(form, lemma, tag, head, relation)` | een woord van een zin in deze les: het woord, het lemma, de tag, het nummer van het hoofdwoord, en de relatie |
        | `"ge" in "opgebeld"` | `in` tussen twee teksten: `True` als de eerste tekst ergens in de tweede staat |
        | `"2,5".replace(",", ".")` | een kopie van de tekst met elke `,` vervangen door `.`: `"2.5"`; met `""` als tweede tekst valt de `,` weg |
        """),
        L.watch_out("UD noemt *plaats* in *vindt plaats* een partikel, maar het woordenboek noemt het een zelfstandig naamwoord. De labels van een treebank volgen de richtlijnen van die treebank, en die richtlijnen zetten sommige vaste combinaties bij de scheidbare werkwoorden. Als je partikels telt in een treebank, tel je wat de richtlijnen van die treebank een partikel noemen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Lees een woord van de treebank als vijf delen: `form, lemma, tag, head, relation`. Woordnummer `i` staat op plaats `i - 1` van de zin.
    2. Om de scheidbare werkwoorden te vinden, kijk je naar de werkwoorden en de lemma's ervan, niet alleen naar de partikels: een werkwoord dat als één woord in de zin staat, heeft geen eigen partikel.
    3. Om te zien waar de partikels staan, ga je de woordnummers van elke zin langs, neem je de woorden met de relatie `compound:prt`, en vergelijk je waar elk partikel staat met waar het hoofdwoord staat.
    4. Tel over een hele treebank in een dictionary, en vergelijk het Nederlands met het Engels.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel bevat drie zinnen als lijsten van woorden, elk woord met de vijf delen: `BEL`, *Ik bel je morgen op*; `OPGELOST`, *We hebben het probleem samen opgelost*; en `PICK`, *Pick it up*. De lemma's, tags en relaties zijn van de course, op de manier van de twee treebanks. De tweede cel definieert een nieuwe functie `particles`, die elk partikel van een zin geeft met de vorm van het werkwoord, niet met nummers zoals in *Corpora en treebanks*, en toont daarna het lemma van elk woord van `OPGELOST`.
    """)
    return


@app.cell
def sentences_cell():
    BEL = [
        ("Ik", "ik", "PRON", 2, "nsubj"),
        ("bel", "op_bellen", "VERB", 0, "root"),
        ("je", "je", "PRON", 2, "obj"),
        ("morgen", "morgen", "ADV", 2, "advmod"),
        ("op", "op", "ADP", 2, "compound:prt"),
        (".", ".", "PUNCT", 2, "punct"),
    ]
    OPGELOST = [
        ("We", "we", "PRON", 6, "nsubj"),
        ("hebben", "hebben", "AUX", 6, "aux"),
        ("het", "het", "DET", 4, "det"),
        ("probleem", "probleem", "NOUN", 6, "obj"),
        ("samen", "samen", "ADV", 6, "advmod"),
        ("opgelost", "op_lossen", "VERB", 0, "root"),
        (".", ".", "PUNCT", 6, "punct"),
    ]
    PICK = [
        ("Pick", "pick", "VERB", 0, "root"),
        ("it", "it", "PRON", 1, "obj"),
        ("up", "up", "ADP", 1, "compound:prt"),
        (".", ".", "PUNCT", 1, "punct"),
    ]
    return BEL, OPGELOST, PICK


@app.cell
def example_particles(BEL, OPGELOST, PICK):
    def particles(words):
        found = []
        for form, lemma, tag, head, relation in words:
            if relation == "compound:prt":
                found.append((form, words[head - 1][0]))
        return found

    print(particles(BEL))
    print(particles(PICK))
    print(particles(OPGELOST))
    for form, lemma, tag, head, relation in OPGELOST:
        print(form, lemma, tag)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *op* hangt aan *bel*, en *up* aan *Pick*: in de loop over de woorden is `words[head - 1]` het hoofdwoord, en `[0]` daarvan de vorm. `OPGELOST` heeft geen woord met de relatie `compound:prt`, dus `particles` geeft een lege lijst, hoewel *opgelost* een scheidbaar werkwoord is. Het lemma, `op_lossen`, laat dat nog wel zien. Ook de tags houden de werkwoorden uit elkaar: *opgelost* heeft de tag `VERB`, en *hebben*, een hulpwerkwoord, de tag `AUX`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_separable():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. separable: scheidbare werkwoorden, los of niet

        Schrijf `separable(words)`: geef een lijst terug met de scheidbare werkwoorden van de zin, van links naar rechts, elk als één woord geschreven, zoals het werkwoord in het woordenboek staat. `separable(BEL)` geeft `['opbellen']`, en `separable(OPGELOST)` geeft `['oplossen']`: een scheidbaar werkwoord telt ook mee als het als één woord in de zin staat. Vervang de `...`. De check roept `separable(BEL)` aan; daarna eigen zinnen, zoals `GRONDWET`, *De grondwet is aangepast*, de twee werkwoorden *voorkomen*, en `PICK`.
        """),
        L.task("Vind de scheidbare werkwoorden van een zin aan de lemma's."),
    ])
    return


@app.cell(hide_code=True)
def exercise_separable():
    box_separable = L.code_box("separable", lines=6)
    box_separable
    return (box_separable,)


@app.cell(hide_code=True)
def check_separable(box_separable):
    result_separable = L.check("separable", box_separable.value)
    result_separable
    return (result_separable,)


@app.cell(hide_code=True)
def hints_separable():
    L.hints(
        "separable",
        [
            "Ga de woorden langs, en haal elk woord uit elkaar in `form, lemma, tag, head, relation`. Houd een woord als de `tag` `VERB` is en het `lemma` een partikel laat zien, en voeg dat lemma toe, als één woord geschreven.",
            "Wat heeft het lemma van *bel* in `BEL`, `op_bellen`, dat het lemma van *hebben* in `OPGELOST` mist? En hoe haal je dat uit het lemma?",
            r'`        if tag == "VERB" and ___:`',
        ],
        why="`found = []`, dan `for form, lemma, tag, head, relation in words:` met daaronder `if tag == \"VERB\" and \"_\" in lemma:`, en daaronder `found.append(lemma.replace(\"_\", \"\"))`; na de loop `return found`. `separable(OPGELOST)` geeft `['oplossen']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_separable():
    mo.md(r"""
    De afbeelding onder de check laat je functie op alle zinnen van UD_Dutch-Alpino lopen. Slechts ongeveer een op de drie scheidbare werkwoorden die je functie vindt, heeft een eigen partikel in de zin, dat een zoektocht naar partikels zoals `particles` zou vinden. En het lemma zegt wat de vorm niet kan zeggen: *voorkomen* schrijf je voor beide werkwoorden hetzelfde, en alleen het lemma zegt welk van de twee het is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. gap_counts: hoe ver staat het partikel?

        Schrijf `gap_counts(sentences)`: geef een dictionary terug met voor elke afstand hoeveel partikels van de zinnen op die afstand staan. Een partikel is een woord met de relatie `compound:prt`, en het hoofdwoord ervan is het werkwoord. De afstand is 3 voor *op* in `BEL` en 2 voor *up* in `PICK`; staat het partikel twee plaatsen voor het werkwoord, zoals *dicht* in *We proberen de deur dicht te doen*, dan is de afstand −2. Ga de zinnen langs, en in elke zin de woordnummers `i`, zoals `dependents` deed in [Wie hangt van wie af: dependency trees](/course/open/nlp-dependencies?lang=nl); tel zoals `count_tags` deed in *Woordsoorten en woordgroepen*. `gap_counts([BEL, PICK, OPGELOST])` geeft `{3: 1, 2: 1}`. Vervang de `...`. De check roept dat aan; daarna eigen lijsten, ook met Engelse zinnen en een lege lijst.
        """),
        L.task("Tel hoeveel partikels op elke afstand van het werkwoord staan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gaps():
    box_gaps = L.code_box("gap_counts", lines=12)
    box_gaps
    return (box_gaps,)


@app.cell(hide_code=True)
def check_gaps(box_gaps):
    result_gaps = L.check("gap_counts", box_gaps.value)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gap_counts",
        [
            "Maak `counts = {}` voor beide loops. Reken voor elk woord met de relatie `compound:prt` de `gap` uit met `i` en `head`, en tel dan 1 op bij de telling van die `gap`, of begin de telling op 1.",
            "Welk nummer heeft *op* in `BEL`, en welk nummer het hoofdwoord *bel*? Hoe krijg je uit die twee nummers de afstand 3?",
            "`                gap = ___`",
        ],
        why=r'`counts = {}`, dan `for words in sentences:` met daarin `for i in range(1, len(words) + 1):` en `form, lemma, tag, head, relation = words[i - 1]`; `if relation == "compound:prt":` met daaronder `gap = i - head`, dan `if gap in counts:` met `counts[gap] = counts[gap] + 1`, en `else:` met `counts[gap] = 1`. Na beide loops `return counts`. `gap_counts([BEL, PICK, OPGELOST])` geeft `{3: 1, 2: 1}`.',
    )
    return


@app.cell(hide_code=True)
def text_after_gaps():
    mo.md(r"""
    De afbeelding onder de check laat je functie op beide treebanks lopen, en toont elke afstand als aandeel van alle partikels, zodat je de twee talen kunt vergelijken. In het Engels staat een partikel direct na het werkwoord of een of twee woorden later: *pick up the phone*, *pick it up*. In het Nederlands spreiden de staven zich uit aan beide kanten van het werkwoord, en de hoogste staaf staat ervoor.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: zinnen bij elke afstand

    De schuif kiest een afstand. Bij elke afstand toont de afbeelding een zin uit UD_Dutch-Alpino met een partikel op die afstand, door de course gekozen uit de korte zinnen zonder namen van mensen: de boom van de zin, met de boog naar het partikel vetgedrukt en het lemma onder het werkwoord.
    """)
    return


@app.cell(hide_code=True)
def demo_cell():
    demo_gap = L.explore("gap", "afstand", lambda _g: C.explore_frame(_g, L.lang), start=3, values=C.EXPLORE_GAPS, role="",
                         task="Schuif van 1 naar 12. Wat staat er tussen het werkwoord en het partikel als de afstand groot is?")
    demo_gap
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij een grote afstand staat het midden van de zin tussen het werkwoord en het partikel: het onderwerp als het na het werkwoord komt, voorwerpen, woorden als *altijd*, woordgroepen met een voorzetsel. De persoonsvorm staat op de tweede plaats, of vooraan in een vraag of een bevel, en het partikel aan het eind van de zin, dus hoe langer de zin, hoe groter de afstand: in *Ik sta 's morgens om zeven uur op* is de afstand 6. Een partikel vóór het werkwoord staat vlak voor een werkwoord aan het eind van de zin, en de afstand die het vaakst voorkomt is −2: in *ik probeer je morgen terug te bellen* staat *te* ertussen, zoals in *aan te vallen* van de ANS. 333 van de 495 partikels op −2 hebben *te* ertussen. De twee zinnen met *terugbellen* zijn ook voorbeelden in de richtlijnen van Lassy.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_corpus():
    mo.md(r"""
    ### Oefening 3. Welk corpus van het Nederlands?

    Elke taak hieronder heeft Nederlandse data nodig. Zet elke taak onder het corpus dat er het best bij past.
    """)
    return


@app.cell(hide_code=True)
def exercise_corpus():
    sort_corpus = L.sort("corpus")
    sort_corpus
    return (sort_corpus,)


@app.cell(hide_code=True)
def check_corpus(sort_corpus):
    result_corpus = L.check("corpus", sort_corpus)
    result_corpus
    return (result_corpus,)


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.md(r"""
    ### Oefening 4. Nieuwe zinnen, zonder boom
    """)
    return


@app.cell(hide_code=True)
def choice_rule_cell():
    choice_rule = L.choice(
        "rule",
        question="Je hebt 10.000 nieuwe Nederlandse zinnen, zonder bomen of lemma's. Van elk scheidbaar werkwoord wil je het werkwoord als één woord, ook als het partikel los staat, zoals in *Ik bel je morgen op*. Wat helpt het meest?",
        options={
            "stemmer": "De stemmer van Porter voor het Nederlands, woord voor woord",
            "right": "Een parser voor het Nederlands, zoals Alpino",
            "list": "Een lijst met elk Nederlands werkwoord en de vormen",
            "before": "Een regel die elk partikel aan het woord ervoor plakt",
        },
        correct="right",
        why={
            "right": "Een parser koppelt elk partikel aan het werkwoord, waar het partikel ook in de zin staat: Alpino geeft *speelt ... op* de grondvorm `speel_op`, en Frog en de Nederlandse pipelines van spaCy koppelen het partikel ook aan het werkwoord. Een parser maakt fouten, maar kijkt naar de hele zin.",
            "stemmer": "Een stemmer kijkt naar één woord tegelijk: *bel* en *op* blijven twee woorden, en niets vertelt de stemmer dat de twee bij elkaar horen. Zie *Stam en lemma*.",
            "list": "Een lijst vindt *opbellen*, *opbelt* en *opgebeld*. Maar in *Ik bel je morgen op* is het werkwoord *bel*, en een lijst kan niet zien dat de *op* drie woorden verder bij *bel* hoort, en geen voorzetsel is.",
            "before": "In de Nederlandse treebank staan 151 van de 1.952 partikels direct na het werkwoord; de andere staan verder weg, of ervoor. Zie de afbeelding onder Oefening 2.",
        },
    )
    choice_rule
    return (choice_rule,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Corpora en treebanks](/course/open/nlp-corpora?lang=nl) vond de partikels van een treebank aan de labels, en [Stam en lemma](/course/open/nlp-stemming?lang=nl) liet zien wat een lemma is. Het Nederlands heeft eigen tools en corpora: Alpino en Frog parsen Nederlands, spaCy en Stanza hebben Nederlandse modellen, getraind op Nederlandse treebanks van UD, en SoNaR, het Corpus Gesproken Nederlands en Lassy bevatten geschreven en gesproken Nederlands, deels door mensen gecontroleerd. In UD_Dutch-Alpino vond je de scheidbare werkwoorden aan de lemma's, ook als een werkwoord als één woord in de zin staat, en telde je waar de partikels staan: in het Engels bijna altijd hooguit drie plaatsen na het werkwoord, in het Nederlands aan beide kanten ervan, vaak ver weg. Een regel voor het Engels mist de meeste; een parser voor het Nederlands koppelt elk partikel aan het werkwoord. De volgende les begint de module over betekenis: *Een woord ken je aan het gezelschap waarin het verkeert*.

    **Een voorbeeld uit het dagelijks leven.** Een oplader uit het Verenigd Koninkrijk heeft in Nederland een verloopstekker nodig: de stroom is hetzelfde, de stekker niet. Een tool voor het Engels werkt op het Nederlands alleen voor zover het Nederlands werkt als het Engels.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rule, result_corpus, result_gaps, result_separable):
    L.wrap_up(result_separable, result_gaps, result_corpus, choice_rule)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
