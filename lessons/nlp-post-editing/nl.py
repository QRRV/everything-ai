# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-post-editing. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-post-editing", "nl")
    C = L.checks   # de zinnen, de vertalingen van Qwen en van mensen, het geheugen, TER, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl), en één over [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_wer():
    quick_wer = L.choice(
        "quick-wer",
        question="In *Hoe ver liggen twee woorden uit elkaar? Edit distance* telt de word error rate de keren invoegen, weghalen en vervangen tussen wat iemand zei en wat een programma hoorde, als deel van het aantal gezegde woorden. Iemand zegt *turn on the light*, en het programma hoort *turn the lights*. Wat is de word error rate?",
        options={
            "place": "75%, want alleen *turn* staat in beide op dezelfde plaats",
            "right": "50%",
            "heard": "67%, voor 2 bewerkingen op 3 gehoorde woorden",
            "missing": "25%, voor de ontbrekende *on*",
        },
        correct="right",
        why={
            "right": "*on* invoegen en *lights* vervangen door *light*: 2 bewerkingen op 4 gezegde woorden, 50%. Tel je plaats voor plaats, dan staat *the* tegenover *on*, en telt elk woord daarna als fout; de edit distance lijnt de woorden eerst uit. Deze les lijnt een vertaling op dezelfde manier uit met een referentie, en voegt nog een soort bewerking toe: een blok woorden verplaatsen.",
            "place": "Tel je plaats voor plaats, dan staat *the* tegenover *on*, en telt elk woord daarna als fout. Wat doet de edit distance eerst, voordat er geteld wordt? Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
            "heard": "De word error rate deelt door de woorden die gezegd werden, niet door de woorden die het programma hoorde. Hoeveel woorden zei de persoon? Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
            "missing": "*on* ontbreekt, en nog een woord is anders: *lights* waar de persoon *light* zei. Hoeveel bewerkingen is dat? Zie [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl).",
        },
    )
    quick_wer
    return (quick_wer,)


@app.cell(hide_code=True)
def quick_check_bleu():
    quick_bleu = L.choice(
        "quick-bleu",
        question="De vertaling *take the next bus i will* heeft elk woord van de referentie *i will take the next bus*, en n-gram-precisions, zoals in *Een vertaling scoren: BLEU en chrF*, van 6/6, 4/5, 2/4 en 1/3. Waarom zijn de precisions van de langere n-grams lager?",
        options={
            "clip": "Door het aftoppen telt elk woord slechts één keer",
            "few": "Zes woorden hebben slechts drie groepjes van vier, dus het deel zakt",
            "right": "Twee blokken staan op elkaars plaats",
            "short": "De brevity penalty maakt de precisions lager",
        },
        correct="right",
        why={
            "right": "*i will* staat achteraan en *take the next bus* vooraan, dus de groepjes woorden die van het ene blok in het andere doorlopen, zoals *bus i*, staan niet in de referentie. Iemand herstelt het met één verplaatsing: *i will* naar voren. BLEU telt zo'n verplaatsing niet; deze les telt het als één bewerking.",
            "clip": "Aftoppen doet alleen iets bij een n-gram dat de vertaling vaker heeft dan de referentie, en elk woord komt hier één keer voor. Welke groepjes woorden op een rij ontbreken in de referentie? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
            "few": "Een precision is een deel: 1 van de 3 groepjes van vier is hetzelfde deel als 2 van de 6. Welke groepjes woorden op een rij ontbreken in de referentie? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
            "short": "De brevity penalty vermenigvuldigt BLEU als geheel, niet elke precision, en de vertaling heeft evenveel woorden als de referentie. Welke groepjes woorden op een rij ontbreken in de referentie? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
        },
    )
    quick_bleu
    return (quick_bleu,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    BLEU zegt hoe dicht een vertaling bij vertalingen door mensen ligt. Een vertaler die een machinevertaling krijgt, wil iets anders weten: hoeveel werk is het om de vertaling te verbeteren? In een overzicht van machinevertaling uit 1997 keek Martin Kay terug op de stap weg van machines die alleen vertalen, naar een werkwijze waarin mensen en machines het werk verdelen. Bij systemen die hele teksten vertalen, schreef Kay, gaat iedereen er gewoon van uit dat iemand de uitvoer van de machine nawerkt. In 2010 maten Mirko Plitt en François Masselot wat dat oplevert bij Autodesk, een softwarebedrijf, met twaalf vertalers die uit het Engels naar het Frans, Italiaans, Duits en Spaans vertaalden. Throughput is hoeveel een vertaler in een uur afkrijgt. Met machinevertaling lag de throughput gemiddeld 74% hoger, en dat scheelde 43% van de vertaaltijd. De winst verschilde per vertaler, van 20% tot 131%, en één uitkomst verraste de onderzoekers: in vertaalopdrachten zaten meer fouten dan in opdrachten met post-editing.

    In deze les tel je de bewerkingen die een vertaling nodig heeft, met TER, en vergelijk je twee beginpunten voor een vertaler: een vertaling van Qwen3-0.6B, het chatmodel van de course, en een vertaalgeheugen, dat opzoekt hoe een zin die op de nieuwe zin lijkt eerder vertaald is. De zinnen komen uit Tatoeba: het zijn de 3.707 Nederlandse zinnen van *Een vertaling scoren: BLEU en chrF*, zonder de 29 waarvoor de vertaling van Qwen een voornaamwoord gebruikt dat een geslacht aangeeft, want zulke zinnen laat de course weg; dat geeft 3.678. Het vertaalgeheugen bevat 120.728 andere Nederlandse zinnen van Tatoeba, elk met een Engelse vertaling door mensen; ook daar liet de course zinnen weg met zo'n voornaamwoord in een vertaling.

    **Een voorbeeld uit het dagelijks leven.** Een concept verbeteren dat een collega schreef: je houdt wat goed is, verandert hier een woord en verplaatst daar een zin. Is het concept goed, dan gaat dat sneller dan alles zelf schrijven; is het slecht, dan schrijf je het toch opnieuw.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_exact = L.guess(
        "guess",
        question="De course liet Qwen3-0.6B de 3.678 Nederlandse zinnen van deze les in het Engels vertalen, zin voor zin, met de prompt *Translate this Dutch sentence into English. Reply with only the English translation.* Elke zin heeft twee tot 14 vertalingen door mensen. Bij hoeveel van de 3.678 is de vertaling van Qwen, met de woorden in kleine letters en zonder leestekens, precies een van de vertalingen door mensen?",
        options={
            "almost": "Bijna geen enkele",
            "tenth": "Ongeveer een op de tien",
            "right": "Ongeveer drie op de tien",
            "most": "Meer dan twee derde ervan",
        },
        reveal="1.142 van de 3.678, of 31,0%. De vertalingen woord voor woord van *Vertalen met regels* haalden er 726. Een vertaling die anders is, is niet altijd fout, zoals *Ik moet naar bed.* laat zien: Qwen schreef *I must go to bed.*, en mensen schreven *I have to go to bed.* en *I need to go to sleep.*",
    )
    guess_exact
    return (guess_exact,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Post-editing** is een machinevertaling verbeteren in plaats van alles zelf te vertalen: de vertaler houdt wat goed is en verandert de rest. Het hoort bij **computer-assisted translation** (CAT): een mens vertaalt, met software die helpt. Een oudere vorm van hulp is een **vertaalgeheugen**: elke zin die een vertaler vertaald heeft, bewaard met de vertaling erbij. Bij een nieuwe zin zoekt het geheugen de opgeslagen zin op die er het dichtst bij ligt, een **fuzzy match**, en toont de vertaling daarvan, om na te werken zoals een machinevertaling. Programma's tonen hoe dichtbij een match ligt als percentage, met 100% voor dezelfde zin, en elk programma rekent dat op een eigen manier uit. Deze les gebruikt de edit distance van de woorden:

    $$\text{fuzzy} = 1 - \frac{\text{edit distance}}{\text{lengte van de langste zin}}$$

    Hoeveel heeft een vertaling nog nodig? In 2006 stelden Matthew Snover en vier collega's voor om bewerkingen te tellen. TER is het kleinste aantal bewerkingen waarmee een hypothese precies gelijk wordt aan een van de referenties, gedeeld door de gemiddelde lengte van de referenties. De **hypothese** is de vertaling die gescoord wordt. De bewerkingen zijn die van de word error rate, plus één: losse woorden invoegen, weghalen en vervangen, en reeksen woorden verplaatsen. Een **shift** verplaatst een blok woorden naar een andere plaats, als één bewerking. Volgens Snover en collega's kost elke bewerking evenveel, ook een shift van veel woorden over een grote afstand.

    TER probeert niet elke mogelijke reeks shifts. TER is **greedy**: het zet steeds de beste stap die het ziet, één tegelijk, zonder verder vooruit te kijken. Steeds kiest TER de shift die het aantal keren invoegen, weghalen en vervangen het meest verlaagt, tot geen enkele shift nog helpt. En een shift moet woorden op de goede plek zetten: op de nieuwe plaats moeten de verplaatste woorden precies gelijk zijn aan de woorden van de referentie. De edit distance telt de rest. Net als bij de word error rate schrijf je TER keer 100, lager is beter, en TER kan boven de 100 uitkomen als een vertaling veel woorden te veel heeft.

    Vertalingen door mensen die gemaakt zijn zonder de machinevertaling te zien, zeggen hetzelfde met eigen woorden, en TER telt ook deze verschillen als bewerkingen. Snover en collega's lieten dus ook mensen die de doeltaal goed spreken bij elke machinevertaling een referentie maken, speciaal gericht op de uitvoer van het systeem: de hypothese werd nagewerkt tot de zin vloeiend was en hetzelfde betekende als de referentie. TER tegen zo'n **gerichte referentie** heet **HTER**, human-targeted TER: het kleinste aantal bewerkingen van de vertaling naar de post-edit ervan, net als bij TER gedeeld door de gemiddelde lengte van de referenties die gemaakt zijn zonder de machinevertaling te zien. Het kost tijd: gemiddeld 3 tot 7 minuten per zin voor een goede gerichte referentie.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **post-editing** | een machinevertaling verbeteren, en houden wat goed is, in plaats van alles zelf te vertalen |
    | **computer-assisted translation** (CAT) | vertalen door een mens, met software die helpt, zoals een vertaalgeheugen |
    | **vertaalgeheugen** | eerder vertaalde zinnen, elk bewaard met de vertaling erbij |
    | **fuzzy match** | een opgeslagen zin die op een nieuwe zin lijkt, met een score voor hoe dichtbij de zin ligt; 100% is dezelfde zin |
    | **hypothese** | de vertaling die TER scoort |
    | **shift** | een blok woorden naar een andere plaats in een zin verplaatsen, als één bewerking |
    | **greedy** | steeds de beste stap nemen die je ziet, één tegelijk, zonder verder vooruit te kijken |
    | **TER** | translation edit rate: het kleinste aantal bewerkingen, shifts meegeteld, naar de dichtstbijzijnde referentie, gedeeld door de gemiddelde lengte van de referenties, keer 100 |
    | **gerichte referentie** | een referentie die ontstaat door de vertaling zelf na te werken |
    | **HTER** | TER tegen een gerichte referentie: het kleinste aantal bewerkingen van een vertaling naar de post-edit ervan |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo tel je de TER-bewerkingen van een vertaling tegen één referentie:

    1. Lijn de twee uit met de edit distance, zoals in *Hoe ver liggen twee woorden uit elkaar?*.
    2. Probeer elke shift van een blok van hoogstens 10 woorden met een woord dat niet uitlijnt, naar een plaats waar de referentie hetzelfde blok heeft, en reken uit wat de edit distance daarna is.
    3. Voer de shift uit die de edit distance het meest verlaagt; bij gelijke winst het langste blok. Ga terug naar stap 1 met de verschoven vertaling, tot geen enkele shift de edit distance nog verlaagt.
    4. De TER-bewerkingen zijn de shifts plus de edit distance die overblijft.

    Bij meer referenties neem je de referentie met de minste bewerkingen, en deel je door de gemiddelde lengte van de referenties. Bij veel zinnen tel je eerst de bewerkingen en de gemiddelde lengtes van alle zinnen op, en deel je één keer, zoals BLEU dat deed op corpusniveau.

    De TER van deze les werkt zoals de TER van sacreBLEU: blokken van hoogstens 10 woorden, hoogstens 50 plaatsen van hetzelfde blok in de referentie, en hoogstens 1.000 geprobeerde shifts. Op de woorden van de les, kleine letters zonder leestekens zoals in *Vertalen met regels*, geeft de TER van de les voor elke zin dezelfde bewerkingen als sacreBLEU.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Neem het voorbeeld van Snover en collega's, met letters in plaats van woorden: de hypothese *a d e b c f* en de referentie *a b c d e f c*. De woorden *b c* in de hypothese kunnen naar links. `C.distance` geeft de edit distance van twee lijsten woorden, en `C.ter_edits` de TER-bewerkingen:
    """)
    return


@app.cell
def example_cell():
    _hypothesis = ["a", "d", "e", "b", "c", "f"]
    _reference = ["a", "b", "c", "d", "e", "f", "c"]
    print(C.distance(_hypothesis, _reference))
    print(C.ter_edits(_hypothesis, _reference))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De edit distance geeft 4. Je kunt *b c* of *d e* gelijk zetten met de referentie, maar niet allebei, dus het andere blok kost woorden die je op de ene plek invoegt en op een andere plek weghaalt of vervangt. TER geeft 2: één shift zet *b c* op plaats 1, wat *a b c d e f* geeft, en met één keer invoegen komt de laatste *c* erbij. Met 7 woorden in de referentie is TER 2/7 × 100, ongeveer 28,6, en zonder shifts 4/7 × 100, ongeveer 57,1. Een vertaler die de hypothese ziet, doet hetzelfde: *b c* naar voren slepen en één woord typen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shift():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. shift: verplaats een blok woorden

        Schrijf `shift`, die `words` krijgt, een lijst woorden, en het blok van `length` woorden dat op plaats `start` begint naar plaats `to` verplaatst. Plaatsen tellen vanaf 0, en `to` telt in de lijst zonder het blok: met `to` 0 gaat het blok naar voren. In de begincode is `block` het blok, `rest` de lijst zonder het blok, en de `return` zet het blok tussen `rest[:to]`, de eerste `to` woorden van `rest`, en `rest[to:]`, de andere: `+` zet twee lijsten achter elkaar. Vervang de twee `...`.

        De check roept `shift(["a", "d", "e", "b", "c", "f"], 3, 2, 1)` aan, en dat geeft `["a", "b", "c", "d", "e", "f"]`: de shift van Snover en collega's, met `["b", "c"]` als blok en `["a", "d", "e", "f"]` als rest. Dan `shift(["can", "i", "this", "eat"], 2, 1, 3)`, en dat geeft `["can", "i", "eat", "this"]`: de vertaling woord voor woord van *Kan ik dit eten?*, met *this* naar het eind. En eigen gevallen: een blok vooraan of achteraan, een blok dat blijft waar het staat, en vertalingen woord voor woord met het werkwoord achteraan, zoals het Nederlands het zet.
        """),
        L.task("Verplaats een blok woorden in een lijst naar een andere plaats."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shift():
    box_shift = L.code_box("shift", lines=4)
    box_shift
    return (box_shift,)


@app.cell(hide_code=True)
def check_shift(box_shift):
    result_shift = L.check("shift", box_shift.value)
    result_shift
    return (result_shift,)


@app.cell(hide_code=True)
def hints_shift():
    L.hints(
        "shift",
        [
            "Een slice `words[a:b]` geeft de woorden van plaats `a` tot plaats `b`, zonder plaats `b` zelf, zoals in *Lijsten*. `block` is de slice van `length` woorden die op plaats `start` begint. `rest` is de woorden voor het blok met daarachter de woorden na het blok, aan elkaar gezet met `+`.",
            "Het blok stopt op plaats `start + length`, dus de woorden erna zijn `words[start + length:]`, en de woorden ervoor `words[:start]`. Een slice heeft na de dubbele punt een plaats nodig, en `length` is een aantal woorden, geen plaats.",
            "`    block = ___`",
        ],
        why="`block = words[start:start + length]` en `rest = words[:start] + words[start + length:]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_shift():
    mo.md(r"""
    Zodra je `shift` werkt, rekent het plaatje onder de check TER ermee uit op alle 3.678 zinnen, voor drie manieren van vertalen: woord voor woord zoals in *Vertalen met regels*, Qwen3-0.6B, en één vertaling door mensen tegen de andere, zoals in *Een vertaling scoren: BLEU en chrF*. Shifts doen het meest bij de vertalingen woord voor woord, omdat zo'n vertaling de Nederlandse woordvolgorde houdt: *Mijn vader heeft een nieuwe auto gekocht.* wordt *my father has a new car bought*, en dat vraagt één shift van *bought* en verder niets. 1.355 van de 3.678 vertalingen woord voor woord hebben minstens één shift nodig, en TER zakt van 47,7 zonder shifts naar 43,3. De vertalingen van Qwen hebben slechts in 343 zinnen een shift nodig, en scoren 36,9 zonder shifts en 36,2 met; de vertalingen door mensen 25,9.

    Qwen is niet bij elke zin beter. De vertaling van Qwen heeft minder TER-bewerkingen nodig dan de vertaling woord voor woord bij 1.586 zinnen, meer bij 876, en evenveel bij 1.216. Op de 1.772 zinnen van hoogstens vier Nederlandse woorden scoren de vertalingen woord voor woord beter, 33,7 tegen 39,8; op de 1.906 langere wint Qwen, 34,3 tegen 48,3. En Qwen vertaalt niet altijd: 51 keer geeft het de Nederlandse zin terug, zoals bij *Echt?*, en slechts twee keer is dat goed, zoals bij *Stop!*. 23 keer geeft het de laatste zin van de prompt, *Reply with only the English translation.*, zoals bij *Vergeet ons niet!*. En 5 keer herhaalt het *The answer is:* tot het stopt bij 64 tokens.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fuzzy():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. fuzzy: hoe dichtbij ligt een opgeslagen zin?

        Schrijf `fuzzy`, die `new` krijgt, de woorden van een nieuwe zin, en `stored`, de woorden van een zin in het geheugen, allebei lijsten, en de fuzzy match teruggeeft als getal van 0 tot 1. De `distance` van de course geeft de edit distance van twee lijsten woorden, zoals je `edit_distance` van *Hoe ver liggen twee woorden uit elkaar? Edit distance* deed: `distance(["ik", "ben", "moe"], ["ik", "ben", "blij"])` geeft 1. `max` geeft het grootste van de getallen die de functie krijgt: `max(3, 5)` is 5. Vervang de twee `...`: `edits`, en `longest`, de lengte van de langste lijst.

        De check roept `fuzzy(["ik", "heb", "geen", "tijd", "om", "te", "lezen"], ["ik", "heb", "geen", "tijd", "om", "te", "schrijven"])` aan, en dat geeft ongeveer 0,8571: één keer vervangen op 7 woorden. Dan `fuzzy(["ik", "weet", "niet", "wat", "dat", "is"], ["ik", "weet", "niet", "zeker", "wat", "dat", "is"])`, en dat geeft ook ongeveer 0,8571: één keer invoegen, en de langste lijst heeft 7 woorden. En eigen gevallen: twee zinnen die gelijk zijn, twee zonder één gemeenschappelijk woord, een langere nieuwe zin, en zinnen uit het geheugen.
        """),
        L.task("Geef een score voor hoe dicht een opgeslagen zin bij een nieuwe zin ligt, uit de edit distance van de twee."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fuzzy():
    box_fuzzy = L.code_box("fuzzy", lines=4)
    box_fuzzy
    return (box_fuzzy,)


@app.cell(hide_code=True)
def check_fuzzy(box_fuzzy):
    result_fuzzy = L.check("fuzzy", box_fuzzy.value)
    result_fuzzy
    return (result_fuzzy,)


@app.cell(hide_code=True)
def hints_fuzzy():
    L.hints(
        "fuzzy",
        [
            "`edits` is de edit distance van `new` en `stored`, uit de `distance` van de course. `longest` is de lengte van de langste van de twee lijsten: `len` geeft de lengte van een lijst, zoals in *Lijsten*, en `max` het grootste van twee getallen.",
            "Deel door de langste lengte, niet door de lengte van alleen `new` of alleen `stored`: twee zinnen hebben hoogstens zoveel bewerkingen nodig als de langste zin woorden heeft, en alleen de lengte van de langste lijst houdt de score tussen 0 en 1. `max` krijgt de twee lengtes, niet de twee lijsten.",
            "`    longest = ___`",
        ],
        why="`edits = distance(new, stored)` en `longest = max(len(new), len(stored))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_fuzzy():
    mo.md(r"""
    Het plaatje onder de check vergelijkt twee beginpunten voor een post-editor: de vertaling van de beste match in het vertaalgeheugen, ongewijzigd, en de vertaling van Qwen. Over alle 3.678 zinnen scoort het geheugen slechter, TER 42,5 tegen 36,2, maar dat hangt af van de match. 113 zinnen hebben een match van 100%: dezelfde woorden, met andere leestekens, zoals *Wees niet bang!* bij *Wees niet bang.*. TER is daar 16,7, en 85 van de 113 hebben geen enkele bewerking nodig. De andere 28 wel: de opgeslagen vertaling kan het met andere woorden zeggen dan de referenties, zoals *It can't be!* bij *Dat kan niet waar zijn.*, en de leestekens, die de match niet meetelt, kunnen de betekenis veranderen: *Tom wacht.* vindt *Tom, wacht!*, met *Tom, wait!*. Van 85% tot 99% wint het geheugen nog, 25,7 tegen 29,2. Van 70% tot 84% liggen de twee dicht bij elkaar, 35,5 tegen 33,8, en onder 70% wint Qwen ruim: 55,0 en 74,6 tegen 40,7 en 42,8. De meeste zinnen vallen in de middelste groepen: 2.015 van 70% tot 84%, en 1.308 van 50% tot 69%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_match():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. best_match: zoek de beste match op

        Schrijf `best_match`, die `new` krijgt, de woorden van een nieuwe zin, en `memory`, een lijst paren: de woorden van een opgeslagen zin, en de Engelse vertaling ervan. Geef de beste fuzzy match en de vertaling daarvan terug, als paar. Bij gelijke matches neem je de match die het eerst in het geheugen staat, en als geen enkele opgeslagen zin boven 0 scoort, geef je `(0, None)` terug: `None` is de waarde van Python voor niets. In de begincode begint `best_score` bij 0 en `best` bij `None`, en de loop neemt steeds één paar, met `score` de fuzzy match uit de `fuzzy` van de course, die net zo werkt als jouw `fuzzy`. Vervang de twee `...`: de `if`, en wat `best` bewaart.

        De check roept `best_match(["ik", "ben", "moe"], [(["ik", "ben", "blij"], "I'm happy."), (["ik", "ben", "erg", "moe"], "I'm very tired."), (["jij", "bent", "moe"], "You're tired.")])` aan, en dat geeft `(0.75, "I'm very tired.")`. Dan `best_match(["ik", "heb", "geen", "tijd", "om", "te", "lezen"], [(["ik", "heb", "geen", "tijd"], "I have no time."), (["ik", "heb", "geen", "tijd", "om", "te", "schrijven"], "I don't have time to write.")])`, en dat geeft een score van ongeveer 0,8571 met *I don't have time to write.* En eigen gevallen: twee gelijke matches, een leeg geheugen, een geheugen zonder match, en kleine geheugens van 20 zinnen uit het geheugen van de course, die `C.near` geeft.
        """),
        L.task("Zoek de opgeslagen zin met de beste fuzzy match, en geef de score en de vertaling terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_match():
    box_match = L.code_box("best_match", lines=9)
    box_match
    return (box_match,)


@app.cell(hide_code=True)
def check_match(box_match):
    result_match = L.check("best_match", box_match.value)
    result_match
    return (result_match,)


@app.cell(hide_code=True)
def hints_match():
    L.hints(
        "best_match",
        [
            "De `if` vraagt of `score` hoger is dan `best_score`, de beste score tot nu toe. Daaronder krijgt `best_score` de nieuwe score, en bewaart `best` de vertaling van hetzelfde paar.",
            "Vraag naar hoger, niet naar hoger of gelijk: met `>=` neemt een latere opgeslagen zin met dezelfde score de plaats van de eerste in. En bewaar `translation`, niet `stored`: een vertaler wil het Engels, niet de Nederlandse woorden van de match.",
            "`        if ___:`",
        ],
        why="In de `if`: `score > best_score`; daaronder: `best = translation`.",
    )
    return


@app.cell(hide_code=True)
def text_after_match():
    mo.md(r"""
    Het plaatje onder de check laat zien wat een fuzzy match wel en niet kan. *Wees niet bang.* vindt *Wees niet bang!*, en de vertaling *Don't be afraid!* heeft geen bewerking nodig. *Ik heb geen tijd om te lezen.* vindt *Ik heb geen tijd om te schrijven.*, met 85,7%, en *I don't have time to write.*: één bewerking, maar zonder deze bewerking zegt de vertaling iets anders. TER telt bewerkingen, geen betekenis: de vertaling van Qwen, *I have no time to read.*, klopt en heeft 2 bewerkingen nodig tegen de vertalingen door mensen. *Ik wist niet dat je zou komen.* vindt *Ik wist wel dat je zou komen.*, ook met 85,7%, en *I knew you'd come.*: één woord keert de betekenis om, en Qwen schreef *I knew that you would come.*, dezelfde fout. *Vergeet ons niet!* vindt *Vergeet mij niet!*, met 66,7%: *Don't forget about me!* heeft één bewerking nodig, waar Qwen een zin uit de prompt gaf. En *Als het nodig is, kom ik snel.* vindt *Als het mogelijk is, zou ik een uitleg willen.*, met 44,4%, en *If possible, I would like an explanation.* helpt weinig.

    Een fuzzy match vergelijkt de woorden van de brontekst, niet de betekenis, dus een vertaler leest elke match goed na.

    ### Probeer het: tel je eigen post-edit

    Typ een machinevertaling en je eigen post-edit ervan, en bekijk de shifts en de woordbewerkingen van TER tegen je post-edit: de bewerkingen die HTER telt. Omdat hier geen andere referenties zijn, deelt het vak door de lengte van je post-edit. De vakken beginnen met de vertaling van Qwen van *Ik hoop je snel te zien.*, *I hope you soon see.*, en een post-edit ervan, *I hope to see you soon.*: één shift van *see* en één keer invoegen van *to*, terwijl de edit distance zonder shifts er 3 nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def box_texts():
    try_translation = mo.ui.text(value="I hope you soon see.", label="De machinevertaling", full_width=True)
    try_post_edit = mo.ui.text(value="I hope to see you soon.", label="Je post-edit", full_width=True)
    mo.vstack([try_translation, try_post_edit])
    return try_post_edit, try_translation


@app.cell(hide_code=True)
def box_cell(try_post_edit, try_translation):
    mo.Html(C.box_picture(L.lang, try_translation.value, try_post_edit.value))
    return


@app.cell(hide_code=True)
def text_exercise_targeted():
    mo.md(r"""
    ### Oefening 4. Waarom is HTER lager?

    De course maakte zelf gerichte referenties. De course werkte de vertalingen van Qwen na van de eerste 40 zinnen van deze les die Qwen niet vertaalde zoals mensen deden, en veranderde niet meer dan vloeiend Engels met de betekenis van de Nederlandse zin nodig heeft. 8 van de 40 hoefden niet te veranderen, zoals *Can you show me another one?* bij *Kunt u me nog een andere laten zien?*: HTER 0, terwijl TER 2 bewerkingen telt tegen *Please show me another one.* en *Please show me another.*, op gemiddeld 4,5 woorden. Over de 40 is HTER 62,4 en TER 74,5: 123 bewerkingen tegen 147, allebei gedeeld door de gemiddelde lengtes van de vertalingen door mensen. Alleen bij *Ik ga.* is het andersom: *Go.* van Qwen werd *I'm going.*, 3 bewerkingen, terwijl de vertaling *I go.* door mensen er 1 nodig heeft, want TER neemt de dichtstbijzijnde van alle referenties en HTER heeft er één. Eén vertaling weegt zwaar: *The answer is:*, herhaald tot 64 tokens, bij *Het kan niet waar zijn.*, kost hoe dan ook 48 bewerkingen; zonder deze vertaling is HTER 39,0 en TER 51,5. Snover en collega's, met vier vertalingen door mensen per zin, vonden dat HTER een derde lager uitkomt dan TER met vier referenties die zonder de machinevertaling gemaakt zijn.
    """)
    return


@app.cell(hide_code=True)
def choice_targeted_cell():
    targeted = L.choice(
        "targeted",
        question="Waarom telt TER tegen vertalingen door mensen bewerkingen die bij post-editing niet nodig zijn?",
        options={
            "mistakes": "Mensen maken fouten bij het vertalen",
            "right": "Mensen zeggen hetzelfde met andere woorden",
            "shifts": "TER telt shifts, en HTER niet",
            "longer": "Vertalingen door mensen zijn langer, dus zijn er meer bewerkingen nodig",
        },
        correct="right",
        why={
            "right": "Elke vertaling door mensen is gemaakt zonder de vertaling van Qwen te zien, en zegt dus hetzelfde met andere woorden, en TER telt elk zulk verschil als een bewerking. Een post-editor begint bij de vertaling van Qwen en verandert alleen wat fout is: *Can you show me another one?* blijft staan, terwijl TER 2 bewerkingen telt tegen *Please show me another one.*",
            "mistakes": "Een slechte vertaling door mensen verandert weinig: TER neemt de referentie met de minste bewerkingen. Wat delen de vertalingen door mensen niet met de vertaling van Qwen, en een post-edit wel?",
            "shifts": "HTER is TER tegen een gerichte referentie, met shifts zoals elke TER. Wat bewaart een gerichte referentie dat een vertaling door mensen niet heeft?",
            "longer": "*Please show me another one.* heeft 5 woorden, en de vertaling van Qwen 6: aan de lengte ligt het niet. Wat is er nog meer anders tussen de twee?",
        },
    )
    targeted
    return (targeted,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De edit distance van [Hoe ver liggen twee woorden uit elkaar? Edit distance](/course/open/nlp-edit-distance?lang=nl) telde de bewerkingen van de word error rate; TER voegt shifts van blokken woorden toe, en dezelfde afstand gaf de fuzzy matches van een vertaalgeheugen. [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl) telde wat een vertaling gemeen heeft met vertalingen door mensen; TER telt wat een vertaler nog moet veranderen, en HTER alleen de bewerkingen naar de post-edit. De volgende les, *Vertalen met een LLM*, vertaalt met een large language model, zoals Qwen hier deed, en kijkt wat een prompt kan veranderen.

    **Een voorbeeld uit het dagelijks leven.** Een docent die een opstel nakijkt met een rode pen, markeert wat dit opstel nodig heeft, niet elk verschil met een voorbeeldopstel: twee goede opstellen kunnen in elke zin verschillen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_shift, result_fuzzy, result_match, targeted):
    L.wrap_up(result_shift, result_fuzzy, result_match, targeted)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
