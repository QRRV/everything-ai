# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-syntactic-ambiguity. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-syntactic-ambiguity", "nl")
    C = L.checks   # de bomen van de course, de aantallen, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen: een over [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl), een over [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_place():
    quick_place = L.choice(
        "quick-place",
        question="De grammatica van *Zinsbouw en eindeloze zinnen* heeft NP → DET NOM | PROPN, VP → VERB NP | VERB NP PP en PP → ADP NP. Waar komt *bij Breda* in de boom van *een lange trein passeerde de brug bij Breda*?",
        options={
            "np": "In de NP, samen met *de brug*",
            "right": "Onder de VP, naast de NP",
            "s": "Direct onder S",
        },
        correct="right",
        why={
            "right": "Alleen VP → VERB NP PP neemt een PP op, dus de PP staat in de werkwoordgroep, naast de NP *de brug*. Geen regel van die grammatica zet een PP in een naamwoordgroep. Deze les voegt er een toe, en dan krijgt de zin een tweede boom.",
            "np": "Daarvoor is een regel nodig die een PP in een naamwoordgroep zet, en de grammatica heeft er geen: een NP is DET NOM of PROPN. Alleen VP → VERB NP PP neemt een PP op. Zie *Regels die zinnen bouwen* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
            "s": "S → NP VP heeft geen PP: de PP komt erin met VP → VERB NP PP, onder de VP. Zie *Regels die zinnen bouwen* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
        },
    )
    quick_place
    return (quick_place,)


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Church en Patil telden de lezingen van zinnen zoals *Put the block in the box on the table*. Met twee voorzetselgroepen, zoals daar, vonden ze 2 lezingen. Hoeveel met drie?",
        options={
            "three": "3: één plek erbij voor de nieuwe groep",
            "four": "4",
            "right": "5",
        },
        correct="right",
        why={
            "right": "Vijf, en veertien met vier: elke nieuwe voorzetselgroep kan ook aanhaken bij de groepen ervoor, dus het aantal groeit steeds sneller. Deze les telt zulke bomen zelf.",
            "three": "Elke nieuwe voorzetselgroep kan ook aanhaken bij de groepen ervoor: Church en Patil vonden 5 bij drie groepen en 14 bij vier. Zie *De keuzes tellen* in [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
            "four": "Church en Patil vonden 5 bij drie voorzetselgroepen, en 14 bij vier: het aantal groeit sneller dan verdubbelen. Zie *De keuzes tellen* in [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De kleine grammatica van *Zinsbouw en eindeloze zinnen* gaf elke zin één boom. Een echte grammatica geeft veel zinnen meer dan één boom, en een programma moet kiezen. Volgens Jurafsky en Martin is ambiguïteit het grootste probleem van een parser. Ze laten een nieuwszin over president Kennedy zien uit het Brown-corpus, een verzameling Engelse teksten, met een hele rij van zulke ambiguïteiten. Aan het eind kun je *nationwide television and radio*, landelijke televisie en radio, lezen als [landelijke [televisie en radio]] of als [[landelijke televisie] en radio]. Een lezer ziet de andere lezingen nauwelijks, omdat de meeste nergens op slaan; een programma ziet ze allemaal. Jurafsky en Martin noemen die stroom aan lezingen die in de grammatica passen maar nergens op slaan een hinderlijk probleem voor elke parser.

    De boom zegt wat bij wat hoort. Een programma dat vragen over een tekst beantwoordt, heeft de boom nodig die de schrijver bedoelde.

    **Een voorbeeld uit het dagelijks leven.** De som 8 − 4 − 2. Lees je (8 − 4) − 2, dan krijg je 2; lees je 8 − (4 − 2), dan krijg je 6. Dezelfde getallen, in dezelfde volgorde: de haakjes beslissen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Ratnaparkhi, Reynar en Roukos (1994) gaven drie experts 300 gevallen uit zinnen van de *Wall Street Journal*, waarin een voorzetselgroep na een werkwoord en het lijdend voorwerp staat en bij allebei kan aanhaken. Met de hele zin kozen de experts gemiddeld in 93 van de 100 gevallen de plek die de treebank geeft. Hoe vaak ongeveer met alleen vier woorden van elk geval, het werkwoord, de kern van het lijdend voorwerp, het voorzetsel en de kern van de woordgroep erna, zoals *report million in charges*?",
        options={
            "coin": "Ongeveer 50 op de 100: niet beter dan een munt",
            "some": "Ongeveer 71 op de 100",
            "right": "Ongeveer 88 op de 100",
        },
        reveal="**Ongeveer 88 op de 100**: gemiddeld 88,2 over de drie experts, een getal dat het stuk met een ~ ervoor afdrukt, tegen 93,2 met de hele zin. In vier woorden zit al het meeste van wat bepaalt waar de voorzetselgroep aanhaakt. Het programma uit het stuk, dat leerde van de treebank en ook alleen de vier woorden zag, koos op dezelfde gevallen in 78,0 van de 100 de plek van de treebank. En de treebank zelf is niet foutloos: bij de 274 gevallen waarover de drie experts met de hele zin het allemaal eens waren, gaf de treebank in 95,7 van de 100 dezelfde plek.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Twee bomen voor één zin

    Geef de grammatica van *Zinsbouw en eindeloze zinnen* één regel erbij: NP → NP PP, een naamwoordgroep met een PP erachter is ook een naamwoordgroep. Nu heeft *een lange trein passeerde de brug bij Breda* twee bomen. In de ene maakt VP → VERB NP PP van *bij Breda* een deel van de werkwoordgroep: bij Breda passeerde de trein de brug. In de andere maakt NP → NP PP er een deel van de naamwoordgroep van: de trein passeerde de brug bij Breda, en niet een andere brug. De afbeelding laat beide zien. De grotere grammatica L1 van Jurafsky en Martin heeft een regel van dezelfde soort, één niveau lager: Nominal → Nominal PP, waarbij Nominal de NOM van *Zinsbouw en eindeloze zinnen* is.

    Jurafsky en Martin noemen dit **attachment ambiguity**: een zin heeft die als een constituent op meer dan één plek in de ontleedboom kan aanhaken. Is die constituent een voorzetselgroep, dan heet het **PP-attachment**, zoals bij de olifant van Groucho Marx in *Waarom taal moeilijk is voor computers*: de voorzetselgroep kan deel worden van de NP of van de VP. Ook andere woordgroepen kunnen op twee plekken aanhaken. In hun voorbeeld *We saw the Eiffel Tower flying to Paris*, we zagen de Eiffeltoren op weg naar Parijs, kan *flying to Paris* zeggen wat de toren deed, of wat wij deden terwijl we de toren zagen.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_kinds():
    mo.md(r"""
    ### Wat *en* verbindt

    Twee veelvoorkomende soorten ambiguïteit zijn volgens Jurafsky en Martin attachment ambiguity en coordination ambiguity. Bij coordination ambiguity kun je op meer dan één manier lezen welke woordgroepen een voegwoord als *en* verbindt. Hun voorbeeld is *old men and women*, in het Nederlands *oude mannen en vrouwen*, dat ook in *Waarom taal moeilijk is voor computers* stond. Met de haakjes [oude [mannen en vrouwen]] zijn de vrouwen ook oud; met [oude mannen] en [vrouwen] alleen de mannen.

    ### Een woord met twee rollen

    *Teacher strikes idle kids* heeft twee bomen omdat de woorden een andere rol kunnen spelen, zoals *Waarom taal moeilijk is voor computers* liet zien:

    ```text
    [S [NP Teacher strikes] [VP idle [NP kids]]]     een staking van leraren laat kinderen zonder les zitten
    [S [NP Teacher] [VP strikes [NP idle kids]]]     een leraar slaat kinderen die niets doen
    ```

    In de eerste boom is *strikes* een zelfstandig naamwoord en *idle* een werkwoord; in de tweede is *strikes* het werkwoord en *idle* een bijvoeglijk naamwoord. In een kop, met maar een paar woorden, gebeurt dat snel: de crash blossoms uit die les zijn koppen van deze soort.

    ### Bracketings tellen

    Hoeveel bomen kan een woordgroep hebben? Neem een samenstelling met drie delen, zoals *stadsbuschauffeur*, en zet telkens haakjes om twee groepen, tot één paar haakjes alles omvat. Dat kan op twee manieren, twee **bracketings**: [[stads bus] chauffeur], de chauffeur van een stadsbus, en [stads [bus chauffeur]], een buschauffeur die voor de stad werkt. Vier delen geven er vijf, vijf delen veertien. Met de woorden van een woordgroep werkt het net zo.

    Je kunt ze tellen zonder te tekenen. Het buitenste paar haakjes splitst het geheel in een linkerdeel en een rechterdeel: vier delen kun je splitsen na het eerste, het tweede of het derde deel. Links en rechts hebben elk hun eigen bracketings, en elke bracketing links gaat samen met elke bracketing rechts. Een splitsing geeft dus het aantal van het linkerdeel maal het aantal van het rechterdeel, en alle splitsingen samen geven het aantal van het geheel. Voor vier delen: 1 × 2 + 1 × 1 + 2 × 1 = 5.

    Het aantal voor vier delen heeft alleen de aantallen voor minder delen nodig. Begin dus klein, bewaar elk aantal in een lijst, en zoek het op als een langere woordgroep het nodig heeft. Het is dezelfde rij getallen als die van Church en Patil in *Waarom taal moeilijk is voor computers*: 1, 1, 2, 5, 14, 42 enzovoort, de Catalan-getallen.

    ### Welke boom is bedoeld?

    Een lezer kiest een boom zonder het te merken. Vaak beslissen de woorden, zoals de experts uit de gok lieten zien met maar vier woorden van elk geval. Een programma kan hetzelfde leren uit een treebank: welke bomen bij welke woorden horen.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **attachment ambiguity** | een woordgroep die op meer dan één plek in de boom van een zin kan aanhaken |
        | **PP-attachment** | attachment ambiguity van een voorzetselgroep: deel van de werkwoordgroep of van een naamwoordgroep |
        | **coordination ambiguity** | als je op meer dan één manier kunt lezen wat *en* of *of* verbindt, zoals in *oude mannen en vrouwen* |
        | **bracketing** | één manier om haakjes te zetten in een woordgroep, telkens om twee groepen, tot één paar haakjes alles omvat |
        | `["VP", "passeerde", ["NP", "de", "brug"]]` | een boom in deze les: een lijst met eerst het symbool, dan de delen; een woord is een string |
        | `vp[2][1]` | het item op plaats 1 van het item op plaats 2: één stap omlaag in de boom per paar blokhaken |
        | een `for` in een `for` | de binnenste loop doorloopt alle rondes opnieuw bij elke ronde van de buitenste loop |
        """),
        L.watch_out("De bomen van deze les laten de tags boven de woorden weg, zoals DET en NOUN, de NOM van *Zinsbouw en eindeloze zinnen* en de NP binnen een PP. Ze laten alleen S, NP, VP en PP zien: genoeg om te tonen waar de PP aanhaakt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zoek een woordgroep die op twee plekken kan aanhaken, een *en* met woorden aan beide kanten, of een woord dat twee rollen kan spelen.
    2. Teken een boom voor elke lezing: de woorden blijven in hun volgorde, alleen de haakjes verschuiven.
    3. Tel de bracketings van een woordgroep: deel de woordgroep bovenaan in een linker- en een rechterdeel, vermenigvuldig hun aantallen, en tel op over de splitsingen, van korte woordgroepen naar lange.
    4. Kies een boom aan de hand van de woorden: vaak zijn het werkwoord, het lijdend voorwerp en de woorden van de PP genoeg.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel bevat de werkwoordgroepen van de twee bomen uit de afbeelding, als lijsten: in `VP_VERB` hoort de PP bij de werkwoordgroep, in `VP_NOUN` bij de naamwoordgroep. De tweede kijkt erin.
    """)
    return


@app.cell
def trees_cell():
    VP_VERB = ["VP", "passeerde", ["NP", "de", "brug"], ["PP", "bij", "Breda"]]
    VP_NOUN = ["VP", "passeerde", ["NP", ["NP", "de", "brug"], ["PP", "bij", "Breda"]]]
    return VP_NOUN, VP_VERB


@app.cell
def example_look(VP_NOUN, VP_VERB):
    print(len(VP_VERB), len(VP_NOUN))
    print(VP_VERB[3])
    print(VP_NOUN[2])
    print(VP_NOUN[2][1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `VP_VERB` heeft vier items: het symbool, het werkwoord, de NP en de PP. `VP_NOUN` heeft er drie: het symbool, het werkwoord, en één naamwoordgroep met daarin de NP en de PP. `VP_NOUN[2][1]` gaat twee stappen omlaag: eerst naar die naamwoordgroep, dan naar de NP *de brug* daarin. De twee lijsten hebben dezelfde woorden, in dezelfde volgorde. In oefening 1 maak je van de ene boom de andere.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_other():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. other_tree: de andere lezing

        Schrijf `other_tree(vp)`: geef de andere boom van dezelfde woorden terug. Heeft `vp` vier items, dan hoort de PP bij de werkwoordgroep: geef dan de VP terug waarin de PP bij de naamwoordgroep hoort, zoals in `VP_NOUN`. Anders zit de PP in de naamwoordgroep: geef dan de VP terug waarin de PP weer bij de werkwoordgroep hoort, zoals in `VP_VERB`. Onderscheid de twee met `len(vp)`. Vervang de `...`. De check roept `other_tree(VP_VERB)` en `other_tree(VP_NOUN)` aan; daarna eigen gevallen, met tags op de plek van de woorden, zoals `["VP", "VERB", ["NP", "PROPN"], ["PP", "ADP", "PROPN"]]`.
        """),
        L.task("Zet de PP in de naamwoordgroep, of haal de PP eruit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_other():
    box_other = L.code_box("other_tree", lines=5)
    box_other
    return (box_other,)


@app.cell(hide_code=True)
def check_other(box_other):
    result_other = L.check("other_tree", box_other.value)
    result_other
    return (result_other,)


@app.cell(hide_code=True)
def hints_other():
    L.hints(
        "other_tree",
        [
            "Begin met `if len(vp) == 4:`. Bouw in dat geval een nieuwe naamwoordgroep uit de NP en de PP van `vp`. Haal in het andere geval de naamwoordgroep uit elkaar met `phrase = vp[2]`, en zet de twee delen in de VP.",
            "Wat zit er in `VP_NOUN[2]`, en wat moet `other_tree(VP_NOUN)` teruggeven?",
            "`    return [\"VP\", vp[1], phrase[1], ___]`",
        ],
        why="`if len(vp) == 4:` geeft `[\"VP\", vp[1], [\"NP\", vp[2], vp[3]]]` terug; anders `phrase = vp[2]` en `return [\"VP\", vp[1], phrase[1], phrase[2]]`. `other_tree(VP_VERB)` geeft `VP_NOUN`, en `other_tree(VP_NOUN)` geeft `VP_VERB`.",
    )
    return


@app.cell(hide_code=True)
def text_after_other():
    mo.md(r"""
    De afbeelding onder de check tekent `VP_VERB` en wat je functie ervan maakte. In beide bomen staat *passeerde de brug bij Breda*; alleen de dikke lijn naar de PP begint ergens anders. Een parser die beide bomen vindt, moet kiezen, en de woorden alleen laten niet altijd zien welke boom de schrijver bedoelde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. bracketings: hoeveel bomen?

        Schrijf `bracketings(n)`: het aantal bracketings van een woordgroep van `n` woorden. Begin met `counts = [0, 1]`: een woordgroep van nul woorden heeft geen bracketing, en een woordgroep van één woord heeft er één, het woord zelf. Ga dan voor elke `size` van 2 tot en met `n` de splitsingen langs, met `left` woorden in het linkerdeel, van 1 tot en met `size - 1`; tel op wat elke splitsing geeft, zoals *Bracketings tellen* in *Het idee* voor vier delen uitrekende; en voeg het totaal toe aan `counts`. Geef aan het eind het aantal voor `n` woorden terug. Vervang de `...`. De check roept `bracketings(4)` aan, daarna enkele `n` van 1 tot en met 12.
        """),
        L.task("Vul de lijst met aantallen, van korte woordgroepen naar lange."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("bracketings", lines=8)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("bracketings", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "bracketings",
        [
            "Twee loops: de buitenste over `size`, de binnenste over `left`. Zet `total` op 0 binnen de buitenste loop, tel in de binnenste loop op wat elke splitsing geeft, en voeg `total` toe aan `counts` na de binnenste loop.",
            "Wat moet `bracketings(3)` geven, en wat telt elk van de twee splitsingen erbij op?",
            "`            total = total + counts[left] * ___`",
        ],
        why="`counts = [0, 1]`; `for size in range(2, n + 1):` met daaronder `total = 0` en `for left in range(1, size):`, en in die loop `total = total + counts[left] * counts[size - left]`; na de binnenste loop `counts.append(total)`. Aan het eind `return counts[n]`. `bracketings(4)` geeft 5.",
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    De afbeelding onder de check zet je aantallen voor 1 tot en met 12 woorden naast de goede. De balken gebruiken een schaal waarop elke stap van dezelfde lengte tien keer zoveel betekent. `bracketings(10)` geeft 4.862: het aantal bomen dat de parser van Church en Patil vond bij een zin met negen voorzetselgroepen. Hun rij is de jouwe, één plek opgeschoven: n voorzetselgroepen geven evenveel bomen als n + 1 woorden bracketings hebben. Een woordgroep van twaalf woorden heeft al 58.786 bracketings, en een parser die elke boom uitschreef, zou erin verdrinken.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: elke bracketing

    Kies een aantal delen, en bekijk elke bracketing van het begin van *stadsbuschauffeursbondloonoverleg*. Zulke zelfstandige naamwoorden schrijf je in het Nederlands aan elkaar, dus de demo zet haakjes tussen de delen van één lange samenstelling. Zoek de bracketing die een lezer bedoelt, en tel hoeveel andere een programma moet uitsluiten.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_words = mo.ui.slider(2, 6, value=4, label="Delen", show_value=True)
    demo_words
    return (demo_words,)


@app.cell(hide_code=True)
def demo_look(demo_words):
    mo.Html(C.demo_picture(demo_words.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 3. Welke soort ambiguïteit?

    Elke vetgedrukte zin kun je op twee manieren lezen. Zet elke zin bij de soort ambiguïteit die er twee bomen van maakt.
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
def text_exercise_agree():
    mo.md(r"""
    ### Oefening 4. Als het werkwoord beslist

    Een bijzin die met *die* begint, kan net als een PP bij twee zelfstandige naamwoorden aanhaken. In *de vriend van de zanger die in Parijs woont* is niet duidelijk wie er in Parijs woont: de vriend of de zanger. Soms beslist een ander woord.
    """)
    return


@app.cell(hide_code=True)
def choice_agree_cell():
    choice_agree = L.choice(
        "agree",
        question="Welke van deze woordgroepen kun je maar op één manier lezen?",
        options={
            "right": "*de trainer van de spelers die te laat was*",
            "player": "*de trainer van de speler die te laat was*",
            "coaches": "*de trainers van de spelers die te laat waren*",
            "is": "*de trainer van de speler die te laat is*",
        },
        correct="right",
        why={
            "right": "*was* is de vorm voor één, en van de twee zelfstandige naamwoorden is alleen *trainer* één persoon: de trainer was te laat. De vorm van het werkwoord past bij één zelfstandig naamwoord, dus de bijzin kan alleen daar aanhaken.",
            "player": "*was* past bij *trainer* en bij *speler*: allebei één persoon. De bijzin kan dus nog bij allebei aanhaken. Zoek een woordgroep waarin de vorm van het werkwoord maar bij één van de twee zelfstandige naamwoorden past.",
            "coaches": "*waren* past bij *trainers* en bij *spelers*: allebei meer dan één. De bijzin kan dus nog bij allebei aanhaken. Zoek een woordgroep waarin de vorm van het werkwoord maar bij één van de twee zelfstandige naamwoorden past.",
            "is": "*is* past bij *trainer* en bij *speler*: allebei één persoon. De bijzin kan dus nog bij allebei aanhaken. Zoek een woordgroep waarin de vorm van het werkwoord maar bij één van de twee zelfstandige naamwoorden past.",
        },
    )
    choice_agree
    return (choice_agree,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl) gaf een zin een boom, en [Waarom taal moeilijk is voor computers](/course/open/nlp-why-language-is-hard?lang=nl) telde lezingen. Deze les bracht de twee samen: met één regel erbij krijgen dezelfde woorden twee bomen. Je maakte van de ene boom de andere, telde hoe snel de bracketings van een woordgroep groeien, deelde drie soorten ambiguïteit in, en zag hoe de vorm van een werkwoord één lezing overlaat. De volgende les gaat van de bouw van een zin naar de betekenis: wanneer is een zin waar?

    **Een voorbeeld uit het dagelijks leven.** Op een bord staat *Gratis koffie en gebak voor leden*. Is de koffie gratis voor iedereen en het gebak alleen voor leden, of is allebei voor leden? De meeste lezers kiezen een lezing zonder de andere te zien.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agree, result_count, result_kinds, result_other):
    L.wrap_up(result_other, result_count, result_kinds, choice_agree)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
