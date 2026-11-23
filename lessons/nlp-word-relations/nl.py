# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-word-relations. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-word-relations", "nl")
    C = L.checks   # de betekenissen en relaties van WordNet, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie vragen, over [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl) en [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_readings():
    quick_readings = L.choice(
        "quick-readings",
        question="In *Waarom taal moeilijk is voor computers* vermenigvuldigde `readings` de betekenissen van de woorden van een zin. Een zin heeft vier woorden met meer dan één betekenis: drie met elk twee, en één met drie. Hoeveel lezingen krijgt de zin?",
        options={
            "sum": "9: de betekenissen opgeteld",
            "right": "24",
            "twelve": "12",
            "four": "4",
        },
        correct="right",
        why={
            "right": "2 × 2 × 2 × 3 = 24: elk woord vermenigvuldigt het aantal lezingen met het aantal betekenissen van dat woord. Deze les vraagt wat die betekenissen zijn, en hoe een woordenboek de betekenissen opsomt.",
            "sum": "Elk woord vermenigvuldigt het aantal lezingen: elke betekenis van het ene woord gaat samen met elke betekenis van de andere. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
            "twelve": "Dan laat je een van de drie woorden met twee betekenissen weg: 2 × 2 × 3 = 12. Alle vier tellen mee: 2 × 2 × 2 × 3.",
            "four": "Dat is het aantal woorden met meer dan één betekenis. De lezingen komen uit het vermenigvuldigen van de betekenissen van die woorden. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
        },
    )
    quick_readings
    return (quick_readings,)


@app.cell(hide_code=True)
def quick_check_variation():
    quick_variation = L.choice(
        "quick-variation",
        question="*Waarom taal moeilijk is voor computers* gaf drie redenen. Welke was variatie?",
        options={
            "ambiguity": "Eén vorm met meer betekenissen",
            "right": "Eén betekenis in veel vormen",
            "context": "Een betekenis die afhangt van wie iets zegt, en waar",
        },
        correct="right",
        why={
            "right": "*colour* en *color* zijn één betekenis in twee spellingen. Twee verschillende woorden kunnen ook één betekenis delen, zoals *couch* en *sofa*: deze les noemt dat synonymie.",
            "ambiguity": "Dat was ambiguïteit, precies andersom: één woord, zoals *bank*, met meer betekenissen. Variatie was één betekenis in veel vormen, zoals *colour* en *color*.",
            "context": "Dat was context. Variatie was één betekenis in veel vormen, zoals *colour* en *color*. Zie [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
        },
    )
    quick_variation
    return (quick_variation,)


@app.cell(hide_code=True)
def quick_check_while():
    quick_while = L.choice(
        "quick-while",
        question="Wat is `n` na deze code?\n\n```python\nn = 3\nwhile n < 20:\n    n = n * 2\n```",
        options={
            "twelve": "12, het laatste getal onder 20",
            "twenty": "20",
            "right": "24",
            "more": "48",
        },
        correct="right",
        why={
            "right": "3 wordt 6, dan 12 en dan 24. Vóór elke ronde vraagt de loop de voorwaarde opnieuw: bij 24 is de voorwaarde `False`, dus de loop stopt. Oefening 3 klimt met een `while`-loop omhoog tot er niets meer boven zit.",
            "twelve": "Bij 12 is de voorwaarde `n < 20` nog `True`, dus de loop draait nog één keer: 12 wordt 24. De loop stopt pas als de voorwaarde `False` is.",
            "twenty": "`n` verdubbelt elke ronde: 3, 6, 12, 24. Het wordt nooit precies 20; de loop stopt bij de eerste waarde die niet onder 20 ligt. Zie [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl).",
            "more": "Bij 24 is de voorwaarde `n < 20` `False`, dus de loop stopt daar en verdubbelt niet nog een keer. Zie [Herhalen tot er iets gebeurt: while](/course/open/py-while?lang=nl).",
        },
    )
    quick_while
    return (quick_while,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Zoek op *automobiel*, en een goede zoekmachine vindt ook pagina's waarop alleen *auto* staat: de twee woorden delen een betekenis. Vraag of een poedel een dier is, en een programma moet weten dat een poedel een soort hond is, en een hond een soort dier. Word embeddings, zoals in *Word embeddings: buren en analogieën*, zetten verwante woorden dicht bij elkaar, maar zeggen niet hoe de woorden samenhangen. Daniel Jurafsky en James H. Martin geven in *Speech and Language Processing* een voorbeeld: vraagt iemand een spraakassistent de muziek harder te zetten, dan is het jammer als de assistent het geluid juist zachter zet. Ook waarschuwen Jurafsky en Martin dat word embeddings zoals word2vec antoniemen, woorden met een tegengestelde betekenis, makkelijk door elkaar halen: een van de dichtstbijzijnde buren van *up* is vaak juist *down*.

    Deze les kijkt naar wat woordenboeken over betekenis opschrijven: de betekenissen van een woord, de woorden die een betekenis delen, waar iets een soort van is, en waar iets een deel van is.

    **Een voorbeeld uit het dagelijks leven.** Een woordenboek nummert de betekenissen van een woord, 1, 2, 3, elk met een eigen definitie, en een synoniemenwoordenboek geeft de woorden met dezelfde betekenis, één rijtje per betekenis.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="WordNet is een grote database van de betekenissen van Engelse woorden. Er staan 127.846 verschillende woorden in, van losse woorden zoals *dog* tot namen en termen van meer woorden. Welk deel daarvan heeft in WordNet precies één betekenis?",
        options={
            "fifth": "Ongeveer 20%",
            "half": "Ongeveer de helft",
            "right": "Ongeveer 80%",
            "nearly": "Ongeveer 95%",
        },
        reveal="**Ongeveer 80%**: 102.341 van de 127.846 woorden, 80,05%, hebben één betekenis. De meeste daarvan zijn zeldzame woorden, namen en termen van meer woorden. De woorden die mensen het meest gebruiken, hebben er veel: *break* heeft er 75, *cut* 70 en *run* 57. In oefening 2 tel je de betekenissen zelf.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Betekenissen

    Jurafsky en Martin omschrijven een **woordbetekenis**, in het Engels *word sense*, als een afgebakende weergave van één kant van wat een woord betekent. Een woordenboek zet de betekenissen van een woord onder elkaar, elk met een definitie. Deze les gebruikt de betekenissen van WordNet, een database van het Engels waar de psycholoog George A. Miller en anderen halverwege de jaren tachtig aan Princeton University mee begonnen, in de open versie die het team van Open English WordNet bijhoudt, de uitgave van 2025. Een latere les, *WordNet: een netwerk van betekenissen*, gaat er verder op in.

    WordNet noemt elke betekenis naar een woord, de woordsoort en een nummer: `bank.n.01` is de eerste betekenis van *bank* als zelfstandig naamwoord, met *n* van noun (*v* is een werkwoord, *a* en *s* een bijvoeglijk naamwoord, *r* een bijwoord). Een betekenis heet naar het eerste woord dat WordNet ervoor noemt, dus de bank voor geld heet `depository_financial_institution.n.01`. *bank* heeft 18 betekenissen: 10 als zelfstandig naamwoord en 8 als werkwoord. De eerste vijf zelfstandige naamwoorden:

    | Betekenis | Definitie in WordNet |
    |---|---|
    | `bank.n.01` | *sloping land (especially the slope beside a body of water)* |
    | `depository_financial_institution.n.01` | *a financial institution that accepts deposits and channels the money into lending activities* |
    | `bank.n.03` | *a long ridge or pile* |
    | `bank.n.04` | *an arrangement of similar objects in a row or in tiers* |
    | `bank.n.05` | *a supply or stock held in reserve for future use (especially in emergencies)* |

    ### Polysemie

    Een woord met meer dan één betekenis is **polyseem**, van het Griekse *poly*, veel, en *sema*, teken; de eigenschap heet **polysemie**. Jurafsky en Martin gebruiken het woord voor elk woord met meer dan één betekenis. Anderen gebruiken het alleen voor betekenissen die verwant zijn, zoals de bank als bedrijf en de bank als het gebouw van dat bedrijf, en noemen betekenissen die niets met elkaar te maken hebben **homonymie**, zoals de oever van een rivier en de bank voor geld. Verwante betekenissen volgen vaak een vast patroon: de organisatie en het gebouw horen ook samen bij *school*, *universiteit* en *ziekenhuis*, merken Jurafsky en Martin op.

    ### Synonymie

    Twee woorden met dezelfde betekenis, of bijna dezelfde, zijn **synoniemen**: *couch* en *sofa*. Jurafsky en Martin benadrukken dat synonymie eigenlijk een relatie tussen betekenissen is, en niet tussen woorden. *big* en *large* kunnen van plaats wisselen in *How big is that plane?*, maar niet in *a big sister*: *big* heeft een betekenis, ouder, die *large* niet heeft. WordNet geeft elke betekenis al de woorden ervan, dus synoniemen zijn woorden die een naam delen: `sofa.n.01` heeft de woorden *sofa*, *couch* en *lounge*, en de Nederlandse woorden *bank*, *canapé* en *sofa*.

    ### Hyponymie

    Een woord of betekenis is volgens Jurafsky en Martin een **hyponiem** van een andere als de eerste specifieker is, en een deelgroep van de andere aanduidt. Het voorbeeld in het boek: *dog* is een hyponiem van *animal*. De algemenere betekenis is het **hyperoniem**: *animal* is een hyperoniem van *dog*. De relatie heet **hyponymie**, of de is-een-relatie: een hond is een dier. Meestal loopt de relatie door; Jurafsky en Martin noemen de relatie transitief: is A een hyponiem van B en B een hyponiem van C, dan is A ook een hyponiem van C. In WordNet heeft bijna elke betekenis van een zelfstandig naamwoord of werkwoord een of meer hyperoniemen. Wie steeds het eerste neemt, klimt van `dog.n.01` in 13 stappen naar `entity.n.01`, de enige betekenis van een zelfstandig naamwoord zonder hyperoniem: canine, carnivore, placental, mammal, vertebrate, chordate, animal, organism, living thing, whole, object, physical entity, entity.

    ### Meronymie

    Een andere veelvoorkomende relatie is volgens Jurafsky en Martin **meronymie**, de deel-geheelrelatie: een poot is een deel van een stoel, een wiel een deel van een auto. Het deel is een **meroniem** van het geheel, en het geheel een **holoniem** van het deel. WordNet houdt drie soorten uit elkaar: een deel, zoals het gaspedaal, een van de 30 delen van `car.n.01`; een lid, zoals een schaap van een kudde; en een stof, zoals water gemaakt is van waterstof en zuurstof.

    Daarnaast hebben **antoniemen** een tegengestelde betekenis: *hot* en *cold*, *up* en *down*. WordNet koppelt een antoniem aan een woord in een betekenis: *hot* in `hot.a.01` heeft *cold* in `cold.a.01`.

    De afbeelding toont één betekenis uit WordNet met de buren ervan.
    """)
    return


@app.cell(hide_code=True)
def idea_dog():
    mo.Html(C.dog_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **woordbetekenis**, betekenis | één betekenis van een woord, zoals een woordenboek de betekenissen nummert; WordNet noemt er een bijvoorbeeld `bank.n.01` |
        | **polysemie**, polyseem | een woord dat meer dan één betekenis heeft |
        | **homonymie** | voor sommige auteurs: betekenissen van één woord die niets met elkaar te maken hebben |
        | **synonymie**, synoniem | twee woorden met dezelfde betekenis, of bijna dezelfde |
        | **hyponymie**, hyponiem, hyperoniem | de is-een-relatie: een hyponiem is een soort van het hyperoniem, de algemenere betekenis |
        | **meronymie**, meroniem, holoniem | de deel-geheelrelatie: een meroniem is een deel, lid of stof van het holoniem |
        | **antoniem** | een woord met de tegengestelde betekenis |
        | `wordnet.senses(word)` | de namen van de betekenissen van een Engels woord: eerst zelfstandige naamwoorden, dan werkwoorden, bijvoeglijke naamwoorden en bijwoorden, elk vanaf de meest gebruikte betekenis |
        | `wordnet.words(name)` | de woorden van een betekenis, zoals `["sofa", "couch", "lounge"]` |
        | `wordnet.synset(name)` | een betekenis met wat WordNet erover weet: `.definition`, `.hypernyms`, `.parts`, `.members`, `.substances` en `.dutch`; WordNet noemt een betekenis met de woorden ervan een synset |
        """),
        L.watch_out("Woordenboeken zijn het niet eens over hoeveel betekenissen een woord heeft. Volgens Jurafsky en Martin onderscheiden woordenboeken vaak veel fijne betekenissen, om kleine verschillen in betekenis te vangen; voor programma's zijn zulke fijne onderscheidingen vaak niet nodig, en worden betekenissen vaak samengenomen tot groepen. Ook WordNet maakt fijne onderscheidingen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zoek de betekenissen van een woord op: `wordnet.senses(word)` geeft de namen.
    2. Twee woorden zijn synoniemen als de twee een betekenis delen: dezelfde naam bij de betekenissen van allebei.
    3. Klim van een betekenis naar de algemeenste: neem het eerste hyperoniem, dan het eerste hyperoniem daarvan, enzovoort, tot er geen meer is.
    4. Voor een geheel en de delen ervan: kijk naar `.parts`, `.members` en `.substances` van de betekenis.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De code zoekt de betekenissen van *couch* op, met de woorden van elke betekenis. Dan neemt de code de eerste betekenis en print de definitie, de hyperoniemen en de Nederlandse woorden.
    """)
    return


@app.cell
def example_cell():
    from everything_ai import wordnet

    for name in wordnet.senses("couch"):
        print(name, wordnet.words(name))
    sofa = wordnet.synset("sofa.n.01")
    print(sofa.definition)
    print(sofa.hypernyms, sofa.dutch)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *couch* heeft vier betekenissen: drie als zelfstandig naamwoord en één als werkwoord. De eerste, `sofa.n.01`, heeft de woorden *sofa*, *couch* en *lounge*: drie synoniemen. De andere twee zelfstandige naamwoorden, `couch.n.02` en `couch.n.03`, hebben alleen *couch*. Het werkwoord `frame.v.04`, *formulate in a particular style or language*, heeft *frame*, *redact*, *cast*, *put* en *couch*. De definitie van `sofa.n.01` is *an upholstered seat for more than one person*; het hyperoniem is `seat.n.03`, en de Nederlandse woorden zijn *bank*, *canapé* en *sofa*. Het Nederlandse *bank* is dus ook polyseem: om op te zitten, en voor geld.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Welke relatie?

    Elke regel heeft twee woorden. Zet de regel onder de relatie tussen de twee.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_relations = L.sort("relations")
    sort_relations
    return (sort_relations,)


@app.cell(hide_code=True)
def check_sort(sort_relations):
    result_sort = L.check("relations", sort_relations)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_after_sort():
    mo.md(r"""
    In WordNet zijn *dokter* en *arts* twee van de Nederlandse woorden van `doctor.n.01`, en *fiets* en *rijwiel* van `bicycle.n.01`. `sparrow.n.01`, met het Nederlandse woord *mus*, klimt in twee stappen naar `bird.n.01`, en `salmon.n.01`, *zalm*, in vijf naar `fish.n.01`. `finger.n.01` is een van de delen van `hand.n.01`, en `tree.n.01` is een lid van `forest.n.01`, met onder meer de Nederlandse woorden *woud* en *bos*. *muis* staat onder meer bij `mouse.n.01`, het dier, en `mouse.n.04`, het apparaat. *slot* staat bij zeven betekenissen, waaronder `castle.n.02`, een kasteel, maar niet bij `lock.n.01`, het slot op een deur: daar geeft Open Dutch WordNet geen Nederlands woord bij.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. count_senses: hoeveel betekenissen per woord?

        Schrijf `count_senses(pairs)`. `pairs` is een lijst met paren `(word, name)`: een woord en de naam van een van de betekenissen ervan, zoals WordNet de paren geeft, één paar voor elk woord van elke betekenis. Geef een dict terug van elk woord naar het aantal betekenissen: het aantal paren met dat woord.

        Voor `count_senses([("bank", "bank.n.01"), ("couch", "sofa.n.01"), ("bank", "depository_financial_institution.n.01"), ("sofa", "sofa.n.01")])` is de uitkomst `{"bank": 2, "couch": 1, "sofa": 1}`. Het vak begint met een lege dict `counts` en loopt al over de paren als `word, name`; vervang de `...`. De check roept dat aan en `count_senses([])`, wat `{}` moet geven, en dan eigen gevallen: één woord met drie betekenissen, twee woorden die een betekenis delen, woorden die terugkomen na andere woorden, en `C.pairs_for(["break", "bank", "poodle"])`, alle betekenissen van drie woorden van WordNet. De afbeelding geeft je functie alle 184.845 paren van WordNet.
        """),
        L.task("Geef een dict terug van elk woord in pairs naar het aantal paren met dat woord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_senses", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_senses", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_senses",
        [
            "In de loop een `if` met een `else:` eronder: één weg voor een woord dat al eerder langskwam, één voor een nieuw woord. Beide wegen zetten `counts[word]`: één meer dan eerst, of een eerste telling.",
            "De eerste keer dat *bank* langskomt, heeft `counts` nog geen key `\"bank\"`, en `counts[\"bank\"]` lezen stopt dan met een KeyError. Wat kun je aan `counts` vragen voordat je het getal van een woord leest?",
            r"`        if ___:`",
        ],
        why="`if word in counts:` met `counts[word] = counts[word] + 1` eronder, en `else:` met `counts[word] = 1` daaronder. Voor het voorbeeld: *bank* krijgt 1, dan *couch* 1, dan *bank* 2, dan *sofa* 1, dus de uitkomst is `{\"bank\": 2, \"couch\": 1, \"sofa\": 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    In de afbeelding telde je functie de betekenissen van alle 127.846 woorden van WordNet. 80% heeft één betekenis, zoals bij het gokken, en de woorden met de meeste zijn korte, veelgebruikte woorden: *break*, *cut*, *run*, *play*, *make*, *light*. GloVe, een verzameling word embeddings geleerd uit een grote hoeveelheid tekst, zet de 50.000 woorden ervan op volgorde, van het meest tot het minst voorkomende woord in die teksten, en die volgorde laat het patroon zien: de 736 woorden van WordNet onder de 1.000 meest voorkomende woorden van GloVe hebben gemiddeld 8,25 betekenissen, de woorden van WordNet onder de volgende 4.000 hebben er 4,94, onder de volgende 15.000 2,84 en onder de volgende 30.000 1,94. Woorden die niet bij de 50.000 van GloVe horen, hebben er 1,17. GloVe schrijft elk woord met kleine letters, dus een woord met een hoofdletter, zoals *Monday*, telt bij de woorden die GloVe niet heeft. In 1945 beschreef de taalkundige George Kingsley Zipf dit verband: hoe vaker een woord gebruikt wordt, hoe meer betekenissen het meestal heeft.

    Gemiddeld per woordsoort heeft een zelfstandig naamwoord in WordNet 1,26 betekenissen en een werkwoord 2,17. Voor WordNet 3.0, de versie van 2006, geven Jurafsky en Martin hetzelfde beeld: gemiddeld 1,23 betekenissen per zelfstandig naamwoord en 2,16 per werkwoord.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. chain: de ladder op

        Schrijf `chain(name, parent)`. `parent` is een dict van een betekenis naar het eerste hyperoniem ervan; de top heeft er geen, dus de top is geen key van `parent`. Geef de lijst terug van `name` tot de top: `name`, de ouder ervan, de ouder daarvan, enzovoort, tot een betekenis die geen ouder heeft.

        Voor `chain("poodle", {"poodle": "dog", "dog": "animal"})` is de uitkomst `["poodle", "dog", "animal"]`. Het vak begint de lijst `out` met `name`; vervang de `...` door een `while`-loop. De check roept dat aan en `chain("animal", {"poodle": "dog", "dog": "animal"})`, wat `["animal"]` moet geven, en dan eigen gevallen: een langere ladder, een naam zonder ouder, een dict met vertakkingen, en drie betekenissen van WordNet met `C.PARENT`, het eerste hyperoniem van elk van de 85.586 betekenissen van WordNet die er een hebben.
        """),
        L.task("Geef de lijst terug van name tot de top, waarin elk volgend item de ouder is van het item ervoor."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chain():
    box_chain = L.code_box("chain", lines=6)
    box_chain
    return (box_chain,)


@app.cell(hide_code=True)
def check_chain(box_chain):
    result_chain = L.check("chain", box_chain.value)
    result_chain
    return (result_chain,)


@app.cell(hide_code=True)
def hints_chain():
    L.hints(
        "chain",
        [
            "Gebruik `while`, met een voorwaarde die vraagt of er nog een stap omhoog is, en voeg in de loop één ouder toe aan `out`, met `parent[...]`. Na de loop: `return out`.",
            "Als de loop `name` niet verandert, verandert een voorwaarde over `name` ook nooit. Welk item van `out` is het item waarvan de ouder nu komt, en hoe schrijf je dat?",
            r"`    while ___ in parent:`",
        ],
        why="`while out[-1] in parent:` met `out.append(parent[out[-1]])` eronder: `out[-1]` is de laatste naam tot nu toe, en de loop stopt bij de eerste naam zonder ouder. Voor het voorbeeld: `out` is `[\"poodle\"]`, dan `[\"poodle\", \"dog\"]`, dan `[\"poodle\", \"dog\", \"animal\"]`, en *animal* is geen key, dus de loop stopt.",
    )
    return


@app.cell(hide_code=True)
def text_after_chain():
    mo.md(r"""
    In de afbeelding klom je functie vanaf vier betekenissen omhoog. De ladders van `poodle.n.01`, 15 sporten, en `cat.n.01`, 14 sporten, komen samen bij `carnivore.n.01`: vanaf daar zijn de twee ladders gelijk. De twee betekenissen van *bank* delen niets onder `entity.n.01`: de oever van een rivier is een helling, een geologische formatie en een object; de bank voor geld een instelling, een organisatie, een sociale groep en een abstractie. Jurafsky en Martin laten hetzelfde zien voor twee betekenissen van *bass*, de zanger en het instrument: de ladders zijn helemaal verschillend en komen pas samen bij het heel abstracte *whole, unit*. Hoe hoog twee ladders samenkomen, zegt iets over hoe sterk twee betekenissen op elkaar lijken: *WordNet: een netwerk van betekenissen* maakt daar een maat van.
    """)
    return


@app.cell(hide_code=True)
def text_explore():
    mo.md(r"""
    ### Probeer het: de betekenissen van een woord

    Typ een woord, in het Engels of in het Nederlands. De tabel geeft de betekenissen in WordNet, met de woorden van elke betekenis, de Nederlandse woorden en de definitie. Daaronder: de ladder van de betekenis waarvan je het nummer kiest, met de delen, leden, stoffen en antoniemen, en de specifiekere betekenissen eronder. Kies English en probeer *head*, *spring* of *bass*, of kies Nederlands en probeer *bank* of *blad*. Sommige koppelingen met Nederlandse woorden kloppen niet: `head.n.01`, het hoofd van een lichaam, heeft de Nederlandse woorden *baas* en *leider*, die bij een andere betekenis van *head* horen.
    """)
    return


@app.cell(hide_code=True)
def explore_controls():
    explore_word = mo.ui.text(value="bank", label="Woord")
    explore_language = mo.ui.radio(options=["English", "Nederlands"], value="Nederlands", label="Taal", inline=True)
    explore_number = mo.ui.number(start=1, stop=99, step=1, value=1, label="Betekenis voor de ladder")
    mo.hstack([explore_word, explore_language, explore_number], justify="start", wrap=True)
    return explore_language, explore_number, explore_word


@app.cell(hide_code=True)
def explore_look(explore_language, explore_number, explore_word):
    mo.Html(C.explore_picture(explore_word.value, explore_language.value, explore_number.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl) telde de lezingen van woorden met meer dan één betekenis. Hier hebben de betekenissen een naam: de betekenissen van WordNet, 18 voor *bank*. De meeste woorden hebben één betekenis, en de meest gebruikte woorden hebben er veel. Meer woorden kunnen één betekenis delen: synonymie. Een betekenis is een soort van een algemenere, via een ladder omhoog tot `entity.n.01`: hyponymie, die je `while`-loop beklom. En een deel, lid of stof van een geheel: meronymie. De volgende les van het domein, *WordNet: een netwerk van betekenissen*, voegt deze relaties samen tot één netwerk, en meet hoe ver twee betekenissen uit elkaar liggen.

    **Een voorbeeld uit het dagelijks leven.** De synoniemenlijst in een tekstverwerker geeft de synoniemen van *groot* in groepjes: *fors* en *omvangrijk* voor de maat, *volwassen* voor een kind dat groot geworden is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chain, result_count, result_sort):
    L.wrap_up(result_sort, result_count, result_chain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
