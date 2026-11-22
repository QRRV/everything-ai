# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-dependencies. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dependencies", "nl")
    C = L.checks   # de zinnen van de course, de treebanks, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rewrite():
    quick_rewrite = L.choice(
        "quick-rewrite",
        question="In *Zinsbouw en eindeloze zinnen* zet `rewrite_first(symbols, grammar, k)` rechterkant nummer `k` op de plaats van de eerste non-terminal. De grammatica van die les heeft DET → *de* | *een*. Wat geeft `rewrite_first([\"DET\", \"NOM\", \"VP\"], GRAMMAR, 1)`?",
        options={
            "first": "`['de', 'NOM', 'VP']`",
            "right": "`['een', 'NOM', 'VP']`",
            "kept": "`['DET', 'een', 'NOM', 'VP']`",
        },
        correct="right",
        why={
            "right": "De rechterkanten tellen vanaf 0, dus nummer 1 van DET is `['een']`, en dat komt op de plaats van DET. In deze les krijgen ook de woorden van een zin nummers, en die tellen vanaf 1.",
            "first": "De rechterkanten tellen vanaf 0: nummer 0 van DET is `['de']`, en nummer 1 is `['een']`. Zie Oefening 1 in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
            "kept": "De rechterkant komt op de plaats van DET, dus DET zelf staat niet meer in de nieuwe lijst. Zie Oefening 1 in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
        },
    )
    quick_rewrite
    return (quick_rewrite,)


@app.cell(hide_code=True)
def quick_check_vp():
    quick_vp = L.choice(
        "quick-vp",
        question="De grammatica van *Zinsbouw en eindeloze zinnen* heeft S → NP VP en VP → VERB NP. Welke woorden beslaat de VP in *de late trein verliet de stad*?",
        options={
            "np": "*de stad*",
            "subject": "*de late trein verliet*",
            "right": "*verliet de stad*",
        },
        correct="right",
        why={
            "right": "De VP is het werkwoord met de naamwoordgroep erachter: *verliet de stad*. In deze les wordt het werkwoord *verliet* het middelpunt van de hele zin.",
            "np": "*de stad* is de NP binnen de VP, en VP → VERB NP zet het werkwoord ervoor: *verliet de stad*. Zie *Woordgroepen in woordgroepen* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
            "subject": "*de late trein* is de NP van S → NP VP, die vóór de VP komt. De VP is het werkwoord met de naamwoordgroep erachter. Zie *Woordgroepen in woordgroepen* in [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl).",
        },
    )
    quick_vp
    return (quick_vp,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    *Zinsbouw en eindeloze zinnen* zette de woorden van een zin in woordgroepen binnen woordgroepen. Je kunt de bouw van een zin ook anders tekenen: verbind elk woord met het woord waar het van afhangt. In *de late trein verliet de stad na middernacht* zegt *late* wat voor trein, is *trein* wat er vertrok, en is *stad* wat de trein verliet. Zulke verbindingen zeggen wie wat deed: volgens Jurafsky en Martin benaderen ze goed hoe een werkwoord en de woorden die erbij horen in betekenis samenhangen. Dat is volgens hen een belangrijke reden dat grammatica's van zulke verbindingen in NLP op dit moment vaker gebruikt worden dan grammatica's van woordgroepen. Ze kunnen ook beter om met een vrije woordvolgorde, zoals in het Tsjechisch: één soort verbinding kan een relatie weergeven waar het woord ook staat, terwijl een grammatica van woordgroepen voor elke plaats een aparte regel nodig heeft.

    Universal Dependencies tekent zulke bomen voor veel talen volgens dezelfde richtlijnen. *Woordsoorten en woordgroepen* gebruikte twee treebanks daarvan, een Engelse en een Nederlandse, voor de tags. Deze les leest de bomen ervan.

    **Een voorbeeld uit het dagelijks leven.** Een organigram: iedereen heeft één leidinggevende, behalve de persoon bovenaan. Bij wie je ook begint, je volgt de leidinggevenden stap voor stap omhoog tot je bovenaan bent.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Neem elk woord dat de Engelse treebank van Universal Dependencies markeert als lijdend voorwerp van een werkwoord, zoals *city* (stad) in *left the city*. Ongeveer 3 op de 100 van die lijdende voorwerpen staan vóór het werkwoord. Hoeveel ongeveer in de Nederlandse treebank van deze les?",
        options={
            "same": "Ongeveer 3 op de 100, net als in het Engels",
            "some": "Ongeveer 25 op de 100",
            "right": "Ongeveer 63 op de 100",
        },
        reveal="**Ongeveer 63 op de 100**: 4.289 van de 6.818 lijdende voorwerpen van een werkwoord in de Nederlandse treebank staan ervoor, tegen 325 van de 9.451 in de Engelse, volgens een eigen telling van de course. Het Nederlands zet het werkwoord vaak laat. In *de trein heeft de stad verlaten* staat het lijdend voorwerp *stad* vóór het werkwoord *verlaten*, en na *dat* komt het werkwoord aan het eind, zoals *Zinsbouw en eindeloze zinnen* liet zien. Wat de volgorde ook is, *stad* is het lijdend voorwerp van *verlaten*: de verbinding tussen de twee woorden blijft dezelfde.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Hoofdwoorden en afhankelijke woorden

    Teken een pijl van *trein* naar *late*: *late* hangt af van *trein*, want het zegt wat voor trein. Het woord waar de pijl vandaan komt, is het **hoofdwoord**, en het woord waar de pijl naar wijst, het **afhankelijke woord** (in het Engels *head* en *dependent*). Volgens Jurafsky en Martin is het hoofdwoord het woord waar alles om draait, en werkt het afhankelijke woord als een soort bepaling erbij. Elke pijl krijgt een label dat zegt wat het afhankelijke woord voor het hoofdwoord is: *late* is een bijvoeglijk naamwoord bij *trein*, label `amod`. Een verbinding van een hoofdwoord naar een afhankelijk woord, met het label, is een **dependency relation**. Je schrijft die als het label met het hoofdwoord en het afhankelijke woord: `amod(trein, late)`.

    ### Een boom van woorden

    In een **dependency tree** heeft elk woord één hoofdwoord, behalve één: de **root**, het middelpunt van de zin, vaak het hoofdwerkwoord. De root krijgt een pijl van boven, met het label `root`, vanuit ROOT: een punt boven de zin dat geen woord is. Jurafsky en Martin noemen drie voorwaarden, waarbij ROOT meetelt als punt van de boom: er gaat geen pijl naar ROOT; naar elk woord gaat precies één pijl; en vanaf ROOT loopt er naar elk woord één pad. Vanaf elk woord kun je dus de hoofdwoorden stap voor stap omhoog volgen, en dan kom je bij de root.

    De ontleedboom van de vorige les had symbolen voor groepen, zoals NP en VP. Een dependency tree heeft die niet: alleen de woorden, en de verbindingen ertussen. De afbeelding toont de boom van de zin, zoals Universal Dependencies die tekent.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_ud():
    mo.md(r"""
    ### Universal Dependencies

    Welk woord is het hoofdwoord, en welke labels zijn er? **Universal Dependencies** (UD) legt dat vast met richtlijnen die voor elke taal hetzelfde zijn: versie 2 heeft 37 universele relaties. De eerste richtlijnen kwamen uit in oktober 2014, en de eerste release, op 15 januari 2015, had 10 treebanks in 10 talen. Release 2.19, van 15 november 2026, heeft 402 treebanks in 226 talen, volgens de downloadpagina van UD; de homepage noemt nog meer dan 200 treebanks in meer dan 150 talen. UD bouwt voort op ouder werk, waaronder de Stanford dependencies en de universele tags van Google; de tags van *Woordsoorten en woordgroepen* zijn die van UD. Dit zijn de relaties uit de afbeelding:

    | Label | Het afhankelijke woord is | In de zin |
    |---|---|---|
    | `nsubj` | het onderwerp: vaak wie of wat iets doet | `nsubj(verliet, trein)` |
    | `obj` | het lijdend voorwerp: vaak waar de handeling mee gebeurt | `obj(verliet, stad)` |
    | `det` | een lidwoord of ander bepalend woord bij een zelfstandig naamwoord | `det(trein, de)` |
    | `amod` | een bijvoeglijk naamwoord bij een zelfstandig naamwoord | `amod(trein, late)` |
    | `obl` | een naamwoordgroep bij een werkwoord die geen onderwerp en geen lijdend voorwerp is, zoals een met een voorzetsel | `obl(verliet, middernacht)` |
    | `case` | een voorzetsel, dat hangt aan het zelfstandig naamwoord waar het bij hoort | `case(middernacht, na)` |
    | `root` | de root van de zin | `root(ROOT, verliet)` |

    ### Een voorzetsel hangt aan het zelfstandig naamwoord

    Eén keuze van UD verrast veel mensen. In *na middernacht* is *middernacht* het hoofdwoord, en het voorzetsel *na* het afhankelijke woord. De richtlijnen zeggen dat woorden die een naamval aangeven, zoals voorzetsels, afhangen van het zelfstandig naamwoord waar ze bij horen. *middernacht* hangt dus aan *verliet*, met `obl`, en *na* hangt aan *middernacht*, met `case`.

    ### De bestanden: CoNLL-U

    De treebanks van UD zijn tekstbestanden in het formaat **CoNLL-U**: één woord per regel, met 10 kolommen gescheiden door tabs, en een lege regel tussen twee zinnen. Een kolom zonder waarde bevat `_`. Dit zijn de kolommen die deze les gebruikt:

    - ID, het nummer van het woord, geteld vanaf 1 in elke zin;
    - FORM, het woord zelf, en UPOS, de tag ervan;
    - HEAD, het nummer van het hoofdwoord, met 0 voor de root;
    - DEPREL, de relatie met het hoofdwoord.

    De zin uit de afbeelding, in die vijf kolommen:

    ```
    ID  FORM         UPOS  HEAD  DEPREL
    1   de           DET   3     det
    2   late         ADJ   3     amod
    3   trein        NOUN  4     nsubj
    4   verliet      VERB  0     root
    5   de           DET   6     det
    6   stad         NOUN  4     obj
    7   na           ADP   8     case
    8   middernacht  NOUN  4     obl
    ```

    In deze les is een zin een lijst, en elk woord daarin een tuple `(form, head, relation)`. Woord nummer `i` staat op plaats `i - 1` van de lijst: de woorden tellen vanaf 1, en de plaatsen in een lijst vanaf 0. De relaties en hoofdwoorden van de zinnen in deze les zijn van de course zelf, volgens de richtlijnen van UD.

    ### Wie het bedacht

    Dependency-grammatica is oud. Jurafsky en Martin voeren die terug op Pāṇini, ergens tussen de 7e en de 4e eeuw v.Chr., en op de grammatici van het oude Griekenland. De moderne dependency-grammatica bouwt voort op de Franse taalkundige Lucien Tesnière (1893–1954), van wie het boek *Éléments de syntaxe structurale* in 1959 verscheen, na de dood van de schrijver. Jurafsky en Martin openen hun hoofdstuk met een passage daaruit, in hun eigen Engelse vertaling. Vrij weergegeven: tussen elk woord in een zin en de woorden eromheen ziet de geest verbindingen, en samen vormen die verbindingen het geraamte van de zin.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **hoofdwoord**, **afhankelijk woord** | de twee woorden van een verbinding: het centrale woord, en het woord dat er iets over zegt; in het Engels *head* en *dependent* |
        | **dependency relation** | een verbinding van een hoofdwoord naar een afhankelijk woord, met een label zoals `nsubj` |
        | **dependency tree** | de verbindingen van een zin: elk woord heeft één hoofdwoord, behalve de root |
        | **root** | het ene woord zonder hoofdwoord, vaak het hoofdwerkwoord; in CoNLL-U is de HEAD ervan 0 |
        | **Universal Dependencies** | richtlijnen voor tags en bomen die voor veel talen hetzelfde zijn, met treebanks die ze volgen; afgekort UD |
        | **CoNLL-U** | het bestandsformaat van UD: een woord per regel, met 10 kolommen |
        | `(form, head, relation)` | een woord van een zin in deze les: het woord, het nummer van het hoofdwoord, en het label |
        | `form, head, relation = word` | geeft elk van de drie delen van een tuple een eigen naam, net als bij een paar |
        | `for step in range(n)` | een loop die hoogstens `n` keer loopt; een `return` erin kan die eerder beëindigen |
        """),
        L.watch_out("In *Woorden en hun delen* was de wortel het morfeem met de hoofdbetekenis van een woord, zoals *werk* in *werkte*. In een dependency tree heet het woord van een zin zonder hoofdwoord de root, met het Engelse woord, zoals in de bestanden van UD."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf de zin als een lijst met tuples `(form, head, relation)`: woord nummer `i` op plaats `i - 1`, en hoofdwoord 0 voor de root.
    2. Zoek de afhankelijke woorden van een woord: loop langs alle woorden, en bewaar de woorden die dat woord als hoofdwoord hebben.
    3. Ga omhoog vanaf een woord: zoek het hoofdwoord op, dan het hoofdwoord daarvan, tot je bij het woord met hoofdwoord 0 bent.
    4. Lees de relaties af aan de labels: `nsubj` en `obj` van het werkwoord zeggen wie wat deed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste cel is de zin uit de afbeelding als lijst. De tweede definieert de functie `head_of`, die het woord teruggeeft dat het hoofdwoord is van woord nummer `i`, of `"ROOT"` voor de root.
    """)
    return


@app.cell
def sentence_cell():
    SENTENCE = [
        ("de", 3, "det"),
        ("late", 3, "amod"),
        ("trein", 4, "nsubj"),
        ("verliet", 0, "root"),
        ("de", 6, "det"),
        ("stad", 4, "obj"),
        ("na", 8, "case"),
        ("middernacht", 4, "obl"),
    ]
    return (SENTENCE,)


@app.cell
def example_head(SENTENCE):
    def head_of(sentence, i):
        form, head, relation = sentence[i - 1]
        if head == 0:
            return "ROOT"
        return sentence[head - 1][0]

    print(head_of(SENTENCE, 3))
    print(head_of(SENTENCE, 7))
    print(head_of(SENTENCE, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Woord 3, *trein*, heeft hoofdwoord 4, en woord nummer 4 staat op plaats 3: *verliet*. Woord 7, *na*, hangt aan *middernacht*, zoals de richtlijnen van UD willen. Woord 4, *verliet*, heeft hoofdwoord 0: dat is de root. In de laatste coderegel van `head_of` is `sentence[head - 1]` de tuple van het hoofdwoord, en de `[0]` erachter neemt het eerste deel daarvan, het woord zelf. Oefening 1 gaat de andere kant op: van een hoofdwoord naar de afhankelijke woorden ervan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dependents():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. dependents: wie hangt aan dit woord?

        Schrijf `dependents(sentence, h)`: geef een lijst terug met de nummers van de woorden met `h` als hoofdwoord, van links naar rechts. Loop langs de woordnummers `i`, van 1 tot en met het aantal woorden in de zin; woord nummer `i` staat op plaats `i - 1`. Met `h` gelijk aan 0 geeft de functie het nummer van de root, in een lijst. Vervang de `...`. De check roept `dependents(SENTENCE, 4)` aan; daarna eigen gevallen, ook met `ROOF`, de zin *het dak van het huis van de bakker lekte*, en met een zin van één woord.
        """),
        L.task("Verzamel de nummers van de woorden met `h` als hoofdwoord."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dependents():
    box_dependents = L.code_box("dependents", lines=7)
    box_dependents
    return (box_dependents,)


@app.cell(hide_code=True)
def check_dependents(box_dependents):
    result_dependents = L.check("dependents", box_dependents.value)
    result_dependents
    return (result_dependents,)


@app.cell(hide_code=True)
def hints_dependents():
    L.hints(
        "dependents",
        [
            "Maak vóór de loop een lege lijst. Geef in de loop elk van de drie delen van woord `i` een eigen naam, en voeg `i` toe als het hoofdwoord `h` is.",
            "Wat moet `dependents(SENTENCE, 4)` teruggeven? En komt je loop bij woord 8, *middernacht*?",
            "`    for i in range(1, ___):`",
        ],
        why="`found = []`, dan `for i in range(1, len(sentence) + 1):` met daaronder `form, head, relation = sentence[i - 1]`, en `if head == h:` met `found.append(i)`; na de loop `return found`. `dependents(SENTENCE, 4)` geeft `[3, 6, 8]`: *trein*, *stad* en *middernacht* hangen aan *verliet*.",
    )
    return


@app.cell(hide_code=True)
def text_after_dependents():
    mo.md(r"""
    De afbeelding onder de check is getekend met je functie: de check vroeg je functie om de afhankelijke woorden van elk hoofdwoord van 0 tot en met 8, en elk daarvan werd een boog. Naar elk woord gaat precies één boog, en naar *verliet* gaat die van boven. Een woord met twee bogen erheen, of geen, zou de boom breken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_path():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. path_to_root: omhoog naar de root

        Schrijf `path_to_root(sentence, i)`: geef de nummers terug van de woorden vanaf woord `i` tot en met de root. `path_to_root(SENTENCE, 1)` geeft `[1, 3, 4]`: woord 1, het hoofdwoord ervan, en het hoofdwoord daarvan. Zoek het hoofdwoord van woord `i` op: is dat 0, dan is woord `i` de root, en is het pad klaar; zo niet, voeg het hoofdwoord toe aan `path`, en ga daar verder, met het hoofdwoord als nieuwe `i`. Een pad heeft nooit meer woorden dan de zin, dus een loop `for step in range(len(sentence))` is lang genoeg, en een `return` erin beëindigt de loop bij de root. Vervang de `...`. De check roept `path_to_root(SENTENCE, 1)` aan; daarna eigen gevallen, ook met `ROOF`.
        """),
        L.task("Volg de hoofdwoorden vanaf woord `i` tot een woord hoofdwoord 0 heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_path():
    box_path = L.code_box("path_to_root", lines=9)
    box_path
    return (box_path,)


@app.cell(hide_code=True)
def check_path(box_path):
    result_path = L.check("path_to_root", box_path.value)
    result_path
    return (result_path,)


@app.cell(hide_code=True)
def hints_path():
    L.hints(
        "path_to_root",
        [
            "Maak de lijst vóór de loop. Zoek in de loop het hoofdwoord van woord `i` op: bij hoofdwoord 0 geef je het pad terug; anders voeg je het hoofdwoord toe, en maak je het de nieuwe `i`.",
            "Wat moet `path_to_root(SENTENCE, 4)` teruggeven, voor de root zelf?",
            "`    path = ___`",
        ],
        why="`path = [i]`, dan `for step in range(len(sentence)):` met daaronder `form, head, relation = sentence[i - 1]`; `if head == 0:` geeft `path` terug; anders `path.append(head)` en `i = head`. Na de loop `return path`. `path_to_root(SENTENCE, 1)` geeft `[1, 3, 4]`: *de*, *trein*, *verliet*.",
    )
    return


@app.cell(hide_code=True)
def text_after_path():
    mo.md(r"""
    De afbeelding onder de check neemt `ROOF`, de langere zin *het dak van het huis van de bakker lekte*. *huis* hangt aan *dak* met `nmod`, een naamwoordgroep bij een zelfstandig naamwoord: die zegt welk dak. Vanaf de *de* vóór *bakker* klimt het pad vier stappen: *bakker*, *huis*, *dak*, *lekte*. Een programma dat wil weten wat er lekte, vindt het onderwerp van *lekte*, *dak*, hoeveel woorden er ook tussen staan.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: bomen uit twee treebanks

    Kies een treebank, en trek er een zin uit, met hoogstens het aantal woorden dat je kiest. Onder de boom: wie wat deed, afgelezen aan de relaties, de kolommen van CoNLL-U waar de bogen uit komen, en de namen van de labels, ook van de labels die deze les niet uitlegt. De Engelse en de Nederlandse bomen gebruiken dezelfde labels; zoek in de Nederlandse een lijdend voorwerp vóór het werkwoord.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_bank = mo.ui.dropdown(options={name: key for key, name in C.TREEBANK_NAMES.items()}, value=C.TREEBANK_NAMES[L.lang], label="Treebank")
    demo_words = mo.ui.slider(5, 20, value=10, label="Woorden, hoogstens", show_value=True)
    demo_new = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Andere zin")
    mo.hstack([demo_bank, demo_words, demo_new], justify="start", gap=1.5, wrap=True)
    return demo_bank, demo_new, demo_words


@app.cell(hide_code=True)
def demo_look(demo_bank, demo_new, demo_words):
    mo.Html(C.demo_picture(demo_bank.value, demo_words.value, demo_new.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_relations():
    mo.md(r"""
    ### Oefening 3. Benoem de relaties

    Neem de zin *Een vermoeide bewaker opende de zware poort.* Elk vetgedrukt woord hangt aan een ander woord van de zin. Zet elk woord bij de relatie die het met het hoofdwoord heeft.
    """)
    return


@app.cell(hide_code=True)
def exercise_relations():
    sort_relations = L.sort("relations")
    sort_relations
    return (sort_relations,)


@app.cell(hide_code=True)
def check_relations(sort_relations):
    result_relations = L.check("relations", sort_relations)
    result_relations
    return (result_relations,)


@app.cell(hide_code=True)
def text_exercise_case():
    mo.md(r"""
    ### Oefening 4. Waar hangt het voorzetsel?

    Neem de zin *een schip zeilde naar de haven*, met de richtlijnen van UD.
    """)
    return


@app.cell(hide_code=True)
def choice_case_cell():
    choice_case = L.choice(
        "case",
        question="Welk woord maakt UD het hoofdwoord van *naar*?",
        options={
            "right": "*haven*",
            "verb": "*zeilde*, het werkwoord van de zin",
            "next": "*de*, het woord er direct na",
            "none": "Geen: *naar* is de root",
        },
        correct="right",
        why={
            "right": "In UD hangt een voorzetsel aan het zelfstandig naamwoord waar het bij hoort, met het label `case`: `case(haven, naar)`. *haven* zelf hangt aan *zeilde*, met `obl`, zoals *middernacht* aan *verliet* hangt.",
            "verb": "*haven* hangt aan *zeilde*, met `obl`. Maar *naar* hangt aan het zelfstandig naamwoord waar het bij hoort, *haven*: in UD hangt een voorzetsel af van het zelfstandig naamwoord. Zie *Een voorzetsel hangt aan het zelfstandig naamwoord* in *Het idee*.",
            "next": "*de* is een lidwoord: het hangt aan *haven*, met `det`, en er hangt niets aan. Zie *Een voorzetsel hangt aan het zelfstandig naamwoord* in *Het idee*.",
            "none": "De root is *zeilde*, het hoofdwerkwoord; elk ander woord heeft een hoofdwoord. Zie *Een boom van woorden* in *Het idee*.",
        },
    )
    choice_case
    return (choice_case,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Zinsbouw en eindeloze zinnen](/course/open/nlp-phrase-structure?lang=nl) zette woorden in woordgroepen binnen woordgroepen. Een dependency tree verbindt de woorden zelf: elk woord hangt aan het hoofdwoord, met een label voor de relatie, en één woord, de root, hangt nergens aan. Je vond de afhankelijke woorden van een woord, volgde de hoofdwoorden omhoog tot en met de root, en benoemde de relaties van een zin. Universal Dependencies tekent zulke bomen volgens dezelfde richtlijnen voor honderden treebanks, zodat de Engelse en de Nederlandse bomen van deze les dezelfde labels hebben. De volgende les laat zinnen zien die meer dan één boom kunnen krijgen.

    **Een voorbeeld uit het dagelijks leven.** Een rivierstelsel: elke beek stroomt uit in een grotere stroom, en al het water komt in zee. Volg een beek stroomafwaarts, welke ook, en je komt bij dezelfde monding uit, zoals elk woord bij de root uitkomt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_case, result_dependents, result_path, result_relations):
    L.wrap_up(result_dependents, result_path, result_relations, choice_case)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
