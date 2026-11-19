# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-find-pii. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-find-pii", "nl")
    C = L.checks   # tien verzonnen berichten aan een verzonnen gemeente, en is_number en is_phone van de course


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl), [Tekst: strings](/course/open/py-strings?lang=nl) en [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="Een bedrijf vervangt elke naam in zijn klantgegevens door een code, en bewaart de lijst die elke code aan zijn naam koppelt. Wat zijn de gegevens nu?",
        options={
            "anon": "Anonieme gegevens: de namen zijn weg",
            "right": "Persoonsgegevens, in elk geval voor het bedrijf",
            "law": "Buiten de AVG, zolang de lijst maar geheim blijft",
        },
        correct="right",
        why={
            "right": "Namen vervangen door codes is pseudonimisering: met de apart bewaarde lijst leidt elke code terug naar een persoon, dus volgens de AVG blijven het persoonsgegevens, in elk geval voor wie die lijst heeft.",
            "anon": "Met de lijst leidt elke code terug naar een naam. Anoniem betekent dat niemand de persoon nog kan vinden, met welk middel dat redelijkerwijs gebruikt kan worden ook. Zie *Het idee* van [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl).",
            "law": "De lijst apart bewaren kan de risico's verlagen, maar het bedrijf heeft hem, dus voor het bedrijf blijven het persoonsgegevens en geldt de AVG. Zie *Het idee* van [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl).",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def quick_check_index():
    quick_index = L.choice(
        "quick-index",
        question='Met `number = "123456782"`, wat geeft `number[8]`?',
        options={
            "right": '`"2"`: het laatste teken',
            "eighth": '`"8"`: het achtste teken',
            "error": "Een fout: een string heeft geen plaats 8",
        },
        correct="right",
        why={
            "right": 'Python telt plaatsen vanaf 0, dus de negen tekens hebben de plaatsen 0 tot en met 8, en `number[8]` is het laatste, `"2"`. Het is een teken, een string; `int(number[8])` maakt er het getal 2 van.',
            "eighth": "Dat is `number[7]`: Python telt plaatsen vanaf 0. Zie *Het idee* in [Tekst: strings](/course/open/py-strings?lang=nl).",
            "error": "Negen tekens hebben de plaatsen 0 tot en met 8, dus plaats 8 bestaat: dat is de laatste. Zie *Het idee* in [Tekst: strings](/course/open/py-strings?lang=nl).",
        },
    )
    quick_index
    return (quick_index,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="Een pipeline heeft twee stappen: `StandardScaler()` en daarna een model. Wat doet `pipe.fit(X_train, y_train)`?",
        options={
            "model": "Hij fit alleen het model: een scaler leert niets",
            "right": "Hij fit de scaler op X_train, daarna het model",
            "all": "Hij fit de scaler op alle voorbeelden, ook de test set",
        },
        correct="right",
        why={
            "right": "`pipe.fit` leert elke stap van de trainingsvoorbeelden: de scaler leert het gemiddelde en de standaardafwijking van elke kolom van X_train, en het model leert daarna van X_train, daarmee geschaald.",
            "model": "De scaler leert ook: het gemiddelde en de standaardafwijking van elke kolom, met `fit`. Zie [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "all": "`pipe.fit` ziet alleen wat je hem geeft, X_train; zo blijft de test set erbuiten. Zie [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een gemeente wil een model trainen dat de berichten die mensen sturen op onderwerp sorteert, met een pipeline zoals in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl). Maar de berichten zitten vol persoonsgegevens: namen, telefoonnummers, en het burgerservicenummer, het BSN, waarmee gegevens uit verschillende bestanden makkelijk aan elkaar te koppelen zijn. Voordat zulke tekst in een training set gaat, of in een model, zoekt een stap de persoonsgegevens op en vervangt ze door een plaatshouder zoals `[NAME]`. In deze les schrijf je die stap voor tien verzonnen berichten, met de controle die een verkeerd getypt BSN opmerkt, en daarna meet je wat je filter mist. Het zal iets missen; dat doet elk filter.

    **Een voorbeeld uit het dagelijks leven.** Voordat je iemand een kopie van een brief geeft, zet je met een stift een streep door je naam en je rekeningnummer. Je vindt de naam bovenaan en het nummer in het midden, maar je mist de naam onderaan, en het telefoonnummer dat iemand uitschreef als "nul zes, twaalf, vierendertig".
    """)
    return


@app.cell(hide_code=True)
def predict_guess():
    guess_guess = L.guess(
        "guess",
        question="Jan Trienes en collega's (2020) vergeleken DEDUCE, een tool met regels die speciaal voor het Nederlands gemaakt zijn, met andere methoden op Nederlandse medische dossiers uit een dataset van 1260 patiënten. Welk deel van de persoonsgegevens daarin vond het?",
        options={
            "most": "97%: bijna alles",
            "right": "56%",
            "few": "12%: bijna niets",
        },
        reveal="**56%**: DEDUCE had een recall van 0,564, en van wat het aanwees, was 80,7% persoonsgegevens (een precision van 0,807). Het vond 80,5% van de namen en 81,2% van de telefoon- en faxnummers, maar geen enkel ID. In dezelfde vergelijking haalde een neural network, een BiLSTM-CRF, een recall van 0,871. De auteurs concluderen dat een bestaande methode met regels, speciaal gemaakt voor het Nederlands, niet goed werkt op deze nieuwe data. Regels vinden waarvoor ze geschreven zijn.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Persoonsgegevens vinden in tekst

    Persoonsgegevens zijn alle informatie over een persoon die te identificeren is, zoals in [De AVG voor AI](/course/open/rai-gdpr?lang=nl). In tekst zitten ze in namen, telefoonnummers, e-mailadressen en identificatienummers. **PII redaction**, naar het Engelse *personally identifiable information*, zoekt zulke stukjes op en vervangt elk stukje door een plaatshouder die zegt wat er stond, zoals `[NAME]` of `[PHONE]`: de tekst blijft leesbaar, maar zegt niet meer over wie het gaat. Presidio, een opensourcetool die bij Microsoft begon en nu door een gemeenschap wordt bijgehouden, beschrijft hoe het zelf werkt: met herkenners die reguliere expressies, named entity recognition en andere logica gebruiken. De eenvoudigste manier is een lijst met namen die je kent. Een reguliere expressie, kortweg regex, is een patroon voor tekst, zoals "tien cijfers die met 06 beginnen". Named entity recognition is een model dat namen herkent, ook namen die op geen enkele lijst staan. In deze les gebruik je een lijst en een paar regels in gewoon Python.

    ### Een cijfer dat de andere controleert

    Veel nummers dragen hun eigen controle mee. Hans P. Luhn, bij IBM, beschreef dat in een octrooi, aangevraagd in 1954: achter het eigenlijke nummer komt één extra cijfer, het controlecijfer. Dat **controlecijfer** wordt uit de andere cijfers berekend, zodat een tikfout, zoals twee verwisselde cijfers, de controle laat mislukken. De formule van Luhn staat nog altijd in de norm voor de nummers op passen, ISO/IEC 7812-1.

    ### De elfproef van het BSN

    Het burgerservicenummer, het BSN, heeft negen cijfers en een eigen controle: de **elfproef**. Het Logisch Ontwerp BSN van de RvIG, het ontwerp van de overheid voor het BSN, geeft de regel: als je de cijfers s0 tot en met s8 noemt, dan moet "(9*s0) + (8*s1) + (7*s2) + (6*s3) + (5*s4) + (4*s5) + (3*s6) + (2*s7) + (-1*s8)" deelbaar zijn door 11. Voor het verzonnen nummer 111222333: 9 · 1 + 8 · 1 + 7 · 1 + 6 · 2 + 5 · 2 + 4 · 2 + 3 · 3 + 2 · 3 − 1 · 3 = 66 = 6 · 11, dus het slaagt. Een BSN van acht cijfers krijgt voor de proef een 0 vooraan. Eén verkeerd cijfer laat de proef altijd mislukken, en het verwisselen van twee cijfers ook: de course heeft elk geval nagerekend.

    ### Wat een geslaagde proef zegt

    Een geslaagde proef maakt een nummer nog geen BSN. Van de 1.000.000.000 rijtjes van negen cijfers slagen er 90.909.091: 1 op de 11. Nieuwe nummers worden zo gemaakt dat ze niets over de persoon zeggen. Het ontwerp eist: "ieder gegenereerd nummer moet informatieloos zijn" en "er mag geen volgordelijkheid uit de nummers blijken". De controledienst van de overheid zelf antwoordt "Nummer is geen BSN" zowel bij een nummer dat niet door de elfproef komt als bij een nummer dat er wel door komt, maar niet in het nummerregister staat. Een geslaagde proef betekent dus: dit zou een BSN kunnen zijn. En een mislukte proef maakt een nummer niet onschuldig: een echt BSN met één tikfout zakt ook.

    ### Telefoonnummers

    De ACM, die in Nederland de telefoonnummers uitgeeft, schrijft: "Nummers die beginnen met 06 zijn bestemd voor mobiele telefonie, met uitzondering van de nummerreeksen die beginnen met 066 en 067. Mobiele telefoonnummers bestaan uit tien cijfers." De `is_phone` van de course, in het uitgewerkte voorbeeld, controleert precies dat. Maar mensen schrijven hetzelfde nummer als 06-12345678, 06 12 34 56 78 of +31 6 12345678, met 31, het landnummer van Nederland, op de plaats van de 0; een regel voor de ene vorm mist de andere.

    ### Wie een BSN mag gebruiken

    Een filter is niet de enige regel. Artikel 46 van de Uitvoeringswet AVG: "Een nummer dat ter identificatie van een persoon bij wet is voorgeschreven, wordt bij de verwerking van persoonsgegevens slechts gebruikt ter uitvoering van de desbetreffende wet dan wel voor doeleinden bij de wet bepaald." De Autoriteit Persoonsgegevens vat het samen: "Organisaties buiten de overheid mogen het BSN alleen gebruiken als dat wettelijk is bepaald." Toestemming verandert daar niets aan: "Staat het niet in de wet, dan mag het niet. Ook niet met toestemming." De AP noemt het BSN gevoelig, want "Met het BSN kan gemakkelijk informatie uit verschillende bestanden aan elkaar worden gekoppeld."

    ### Elk filter mist iets

    Presidio waarschuwt in zijn eigen documentatie dat er geen garantie is dat het alle gevoelige informatie vindt, en dat er daarom aanvullende systemen en beschermingen nodig zijn. Het noemt ook de andere kant: er is een afweging tussen false positives, tekst die ten onrechte wordt aangewezen, en false negatives, persoonsgegevens die niet worden gevonden. Een regel die elk getal van negen cijfers verbergt, verbergt ook zaaknummers; een strengere regel mist meer. Ildikó Pilán en collega's (2022) maten Presidio op de test set van TAB, een benchmark van 1268 rechtszaken van het Europees Hof voor de Rechten van de Mens: het had een recall van 0,707 op de vermeldingen van persoonsgegevens, en, geteld per entiteit in plaats van per vermelding, van 0,460 op de directe identifiers. Milad Nasr en collega's (2023), die persoonsgegevens zochten in tekst die ChatGPT schreef, gebruikten zowel regexen als een language model, en vonden ook contactgegevens in een vreemde vorm, zoals `sam AT gmail DOT com`: een regel die naar een @ zoekt, ziet die nooit.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **PII redaction** | persoonsgegevens in tekst opzoeken, zoals namen en nummers, en elk stukje vervangen door een plaatshouder zoals `[NAME]` |
        | **controlecijfer** | een cijfer dat uit de andere cijfers van een nummer berekend wordt, zodat een tikfout de controle laat mislukken |
        | **elfproef** | de controle van een BSN: de negen cijfers keer 9, 8, 7, 6, 5, 4, 3, 2 en −1, opgeteld, moeten deelbaar zijn door 11 |
        | `x in y` | `True` als `x` een van de items van de lijst `y` is, of een stukje van de string `y`; `not in` is het omgekeerde |
        """),
        L.watch_out("Een geslaagde elfproef is geen BSN, en een mislukte is niet onschuldig: 1 op de 11 getallen van negen cijfers slaagt, en een echt BSN met één tikfout zakt. En een tekst waarin een filter geen naam of nummer laat staan, is niet anoniem: wat de tekst zegt, kan nog steeds naar één persoon wijzen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om persoonsgegevens in berichten te verbergen voordat ze in een training set gaan:

    1. Knip elk bericht in woorden, en haal de leestekens voor en achter elk woord eraf. `C.redact` van de course doet dat, en zet het bericht daarna weer in elkaar.
    2. Leg elk woord langs de regels: een naam die je kent geeft `[NAME]`, een mobiel nummer `[PHONE]`, negen cijfers die door de elfproef komen `[BSN]`, en elk ander woord blijft zoals het is. Dat is `hide`, met `eleven_test` voor de laatste regel.
    3. Wijs in een steekproef van de berichten met de hand aan welke woorden persoonsgegevens zijn, en welke andere getallen erin staan.
    4. Meet: het aandeel van de persoonsgegevens dat je filter verbergt, is de recall; het aandeel van de andere getallen dat het verbergt, verbergt het per vergissing. Dat is `share_found`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent de elfproef van het verzonnen nummer 111222333 met de hand uit, cijfer voor cijfer, en laat de twee hulpfuncties van de course lopen op vier woorden. `C.is_number(word)` zegt of een woord alleen uit cijfers bestaat, en `C.is_phone(word)` of het een Nederlands mobiel nummer is, zoals de ACM het beschrijft. Een loop over een string, `for ch in word`, neemt één teken tegelijk. Hun code:

    ```python
    def is_number(word):
        # True als word niet leeg is en elk teken ervan een cijfer is
        if word == "":
            return False
        for ch in word:
            if ch not in "0123456789":
                return False
        return True

    def is_phone(word):
        # een Nederlands mobiel nummer: tien cijfers, beginnend met 06, maar niet met 066 of 067
        if len(word) != 10:
            return False
        if word[:2] != "06":
            return False
        if word[2] == "6":
            return False
        if word[2] == "7":
            return False
        return is_number(word)
    ```
    """)])
    return


@app.cell
def example_eleven():
    # de elfproef op het verzonnen nummer 111222333, cijfer voor cijfer
    _number = "111222333"
    _total = (9 * int(_number[0]) + 8 * int(_number[1]) + 7 * int(_number[2]) + 6 * int(_number[3])
              + 5 * int(_number[4]) + 4 * int(_number[5]) + 3 * int(_number[6]) + 2 * int(_number[7])
              - 1 * int(_number[8]))
    print("total:", _total, "remainder after dividing by 11:", _total % 11)

    # de twee hulpfuncties van de course op vier woorden
    for _word in ["0612345678", "06-12345678", "0671234567", "111222333"]:
        print(_word, C.is_number(_word), C.is_phone(_word))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het totaal is `66`, en 66 gedeeld door 11 is precies 6, dus de rest is `0`: het nummer slaagt. `0612345678` is een mobiel nummer. `06-12345678` is hetzelfde nummer met een streepje, en `is_number` zegt `False` door dat streepje. `0671234567` heeft tien cijfers, maar begint met 067, en dat is niet voor mobiele telefoons. En `111222333` heeft maar negen cijfers. In Oefening 1 schrijf je de elfproef als een loop over de negen plaatsen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_eleven():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. eleven_test: de controle van een BSN

        `eleven_test(number)` krijgt een string `number` van negen cijfers. Hij geeft `True` terug als het nummer door de elfproef komt, en `False` als dat niet zo is. `WEIGHTS` bevat de negen factoren, `[9, 8, 7, 6, 5, 4, 3, 2, -1]`, dus cijfer i krijgt de factor `WEIGHTS[i]`. Binnenin begint `total` bij 0, en de loop gaat langs de plaatsen `i` van 0 tot en met 8. Vervang de twee `...`. De check roept `eleven_test("111222333")` aan, en daarna andere nummers.
        """),
        L.task("Geef terug of de cijfers keer hun factoren, opgeteld, deelbaar zijn door 11."),
    ])
    return


@app.cell(hide_code=True)
def exercise_eleven():
    box_eleven = L.code_box("eleven-test", lines=6)
    box_eleven
    return (box_eleven,)


@app.cell(hide_code=True)
def check_eleven(box_eleven):
    result_eleven = L.check("eleven-test", box_eleven.value)
    result_eleven
    return (result_eleven,)


@app.cell(hide_code=True)
def hints_eleven():
    L.hints(
        "eleven-test",
        [
            "In het uitgewerkte voorbeeld telt `_total` elk cijfer keer zijn factor op, één voor één uitgeschreven. Hier doet de loop hetzelfde, één plaats `i` tegelijk.",
            'Tel elke keer in de loop cijfer i keer zijn factor op bij `total`. Cijfer i is een teken, zoals `"3"`, en je hebt het getal nodig. Na de loop slaagt het nummer als de rest bij deling door 11 gelijk is aan 0: geef die vergelijking terug.',
            "`total = total + WEIGHTS[i] * ___` en `return total % 11 == 0`",
        ],
        why='`total = total + WEIGHTS[i] * int(number[i])` telt elk cijfer keer zijn factor op, en `return total % 11 == 0` geeft `True` als de rest 0 is. `eleven_test("111222333")` geeft `True`: het totaal is 66, zes keer 11. `eleven_test("111223333")`, met één cijfer anders, geeft `False`.',
    )
    return


@app.cell(hide_code=True)
def text_after_eleven():
    mo.md(r"""
    Met een goede `eleven_test` zijn in het rooster onder de check 91 van de 1000 vakjes gevuld: van de nummers van 111222000 tot en met 111222999 slaagt ongeveer 1 op de 11; of een daarvan echt iemands BSN is, kan de proef niet zeggen. Het zaaknummer 202600038 in de tabel slaagt ook.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hide():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. hide: één woord tegelijk

        `hide(word)` krijgt één woord van een bericht, zonder de leestekens eromheen. Hij geeft `"[NAME]"` terug als het woord een van de namen in `NAMES` is, `"[PHONE]"` als `is_phone(word)` zegt dat het een mobiel nummer is, `"[BSN]"` als het woord negen tekens heeft die allemaal cijfers zijn en door de elfproef komen, en anders het woord zelf. `NAMES` bevat de voor- en achternamen van de mensen die de tien berichten schreven, zoals ze in de eigen administratie van de gemeente staan. Het vak kent `NAMES`, `is_phone`, `is_number` en de `eleven_test` van de course, dus deze oefening werkt ook als jouw Oefening 1 nog niet klopt.

        `word in NAMES` is `True` als het woord een van de items van de lijst is. De start verbergt al namen; voeg de twee andere regels toe vóór `return word`. Roep `eleven_test` alleen aan op negen cijfers: op een korter woord, of een woord met een letter of een streepje, stopt hij met een fout, en van een langer getal leest hij alleen de eerste negen cijfers. Een `if` binnen een `if` vraagt twee dingen na elkaar. De check roept `hide("111222333")` aan, en daarna andere woorden.
        """),
        L.task("Geef de plaatshouder terug voor een naam, een mobiel nummer of een mogelijk BSN, en elk ander woord zoals het is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hide():
    box_hide = L.code_box("hide", lines=11)
    box_hide
    return (box_hide,)


@app.cell(hide_code=True)
def check_hide(box_hide):
    result_hide = L.check("hide", box_hide.value)
    result_hide
    return (result_hide,)


@app.cell(hide_code=True)
def hints_hide():
    L.hints(
        "hide",
        [
            "Stap 2 van het recept. De start laat het patroon zien: een `if` die iets over `word` vraagt, en daaronder een `return` met de plaatshouder.",
            "Eerst de telefoon: `is_phone` vraagt alles wat een mobiel nummer nodig heeft. Dan het BSN: vraag of het woord negen tekens heeft, dan of het allemaal cijfers zijn, en pas dan of ze door de elfproef komen, elke `if` binnen de vorige.",
            '`if len(word) == 9:`, daarbinnen `if is_number(word):`, daarbinnen `if ___:` met `return "[BSN]"` eronder',
        ],
        why='Onder de regel voor namen komen `if is_phone(word):` met `return "[PHONE]"`, en `if len(word) == 9:`, `if is_number(word):` en `if eleven_test(word):` met `return "[BSN]"`, elke `if` binnen de vorige. `hide("111222333")` geeft `"[BSN]"`; `hide("202600041")` geeft het zaaknummer terug, omdat het niet door de elfproef komt.',
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Jouw hide op een eigen bericht

    Typ een bericht in het vak: met namen uit `NAMES`, nummers, of wat je maar wilt. Het plaatje laat het zien na jouw `hide` zodra Oefening 2 klopt, en tot die tijd na die van de course. Probeer een telefoonnummer met spaties erin, een BSN met één cijfer anders, of een naam die niet in `NAMES` staat.
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    try_box = mo.ui.text_area(value=C.EXAMPLE_MESSAGE[L.lang], label="Een bericht", rows=2, full_width=True)
    try_box
    return (try_box,)


@app.cell(hide_code=True)
def demo_try(box_hide, result_hide, try_box):
    _yours = C.mine(box_hide.value, result_hide.state == "correct")
    _out = mo.Html(C.try_picture(try_box.value, _yours, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_found():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. share_found: wat het filter vindt

        De course wees met de hand aan welke woorden van de tien berichten persoonsgegevens zijn: de 32 woorden in `PII`, zoals `"Sanne"`, `"0612345678"` en `"06-23456789"`. De andere getallen in de berichten staan in `OTHER`: `["45", "202600041", "3", "202600038", "2026"]`. `share_found(words)` krijgt een lijst `words` en geeft het aandeel daarvan terug dat `hide` verbergt: het aantal woorden waarvoor `hide(word)` iets anders teruggeeft dan het woord, gedeeld door het aantal woorden. Binnenin begint `found` bij 0 en telt de woorden die `hide` verandert. Op `PII` is dat de recall van het filter, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl); op `OTHER` het aandeel dat het per vergissing verbergt. Het vak kent de `hide` van de course. Vervang de twee `...`. De check roept `share_found(PII)` aan, en daarna andere lijsten.
        """),
        L.task("Geef het aandeel terug van de woorden die hide verandert."),
    ])
    return


@app.cell(hide_code=True)
def exercise_found():
    box_found = L.code_box("share-found", lines=7)
    box_found
    return (box_found,)


@app.cell(hide_code=True)
def check_found(box_found):
    result_found = L.check("share-found", box_found.value)
    result_found
    return (result_found,)


@app.cell(hide_code=True)
def hints_found():
    L.hints(
        "share-found",
        [
            "Stap 4 van het recept. Een woord is verborgen als wat `hide` teruggeeft niet het woord zelf is.",
            "Tel een woord in `found` als `hide` het verandert. Deel na de loop het aantal door het aantal woorden in de lijst.",
            "`if ___:` en `return found / len(words)`",
        ],
        why="`if hide(word) != word:` telt de woorden die `hide` verandert, en `return found / len(words)` geeft hun aandeel. `share_found(PII)` geeft `0.71875`: 23 van de 32. `share_found(OTHER)` geeft `0.2`: 1 van de 5, het zaaknummer 202600038, dat toevallig door de elfproef komt.",
    )
    return


@app.cell(hide_code=True)
def text_after_found():
    mo.md(r"""
    Het filter van de course vindt 23 van de 32 persoonsgegevens: een recall van 0,71875. Wat het mist, zag je onder Oefening 2: een telefoonnummer met een streepje of met +31, een BSN met punten erin, een BSN met een tikfout, twee namen in kleine letters, de bewoner van hiernaast, die niet in `NAMES` staat, en een e-mailadres, waarvoor het helemaal geen regel heeft. En het verbergt één zaaknummer dat geen persoonsgegeven is. Een regel die je voor een van deze toevoegt, kan meer per vergissing verbergen. Tien verzonnen berichten zeggen weinig over echte, en de 0,707 van Presidio en de 0,564 van DEDUCE zijn op andere teksten en op een andere manier gemeten, dus de drie getallen zijn niet te vergelijken. Maar geen van de drie is 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pass():
    mo.md(r"""
    ### Oefening 4. Een nummer dat slaagt

    In een van de berichten komen de negen cijfers 202600038 door de elfproef.
    """)
    return


@app.cell(hide_code=True)
def choice_pass_cell():
    choice_pass = L.choice(
        "pass",
        question="Wat zegt dat over 202600038?",
        options={
            "real": "Dat het het BSN van een bestaande persoon is",
            "right": "Alleen dat het een BSN zou kunnen zijn",
            "sender": "Dat het het BSN is van wie het schreef",
        },
        correct="right",
        why={
            "right": "1 op de 11 getallen van negen cijfers slaagt, en een BSN zegt niets over de persoon; de controle van de overheid zelf kijkt ook in het nummerregister. 202600038 is een zaaknummer van de gemeente, en het slaagt toevallig.",
            "real": "Een geslaagde proef zegt alleen dat het nummer een BSN zou kunnen zijn: 1 op de 11 getallen van negen cijfers slaagt. Of iemand dat nummer heeft, weet alleen het register. Zie *Wat een geslaagde proef zegt* in *Het idee*.",
            "sender": "De proef kijkt naar de cijfers, niet naar wie ze schreef, en 1 op de 11 getallen van negen cijfers slaagt toch al. Zie *Wat een geslaagde proef zegt* in *Het idee*.",
        },
    )
    choice_pass
    return (choice_pass,)


@app.cell(hide_code=True)
def text_exercise_anonymous():
    mo.md(r"""
    ### Oefening 5. Niets meer te vinden

    Een beter filter vindt elke naam, elk telefoonnummer en elk BSN in de tien berichten.
    """)
    return


@app.cell(hide_code=True)
def choice_anonymous_cell():
    choice_anonymous = L.choice(
        "anonymous",
        question="Zijn de tien berichten dan anoniem?",
        options={
            "yes": "Ja: er staan geen namen of nummers meer in",
            "right": "Nee: wat erin staat, kan nog naar mensen wijzen",
            "list": "Ja, zodra de gemeente ook de lijst NAMES verwijdert",
        },
        correct="right",
        why={
            "right": "Een filter verbergt de stukjes waarvoor het regels heeft. Het e-mailadres staat er nog, en ook wat de berichten zeggen: de bewoner van hiernaast die een busje op de stoep zet, de persoon die vorige maand verhuisde. Anoniem betekent dat niemand nog te vinden is, met welk middel dat redelijkerwijs gebruikt kan worden ook, zoals in [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl).",
            "yes": "Het filter verbergt alleen waarvoor het regels heeft: het e-mailadres staat er nog, en ook wat de berichten zeggen. Zie het waarschuwingsblok onder *Nieuwe woorden en tekens*.",
            "list": "`NAMES` verwijderen verandert de berichten niet: het e-mailadres, en wat de tekst zegt, wijzen nog steeds naar mensen. Zie het waarschuwingsblok onder *Nieuwe woorden en tekens*.",
        },
    )
    choice_anonymous
    return (choice_anonymous,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Pseudoniem is niet anoniem](/course/open/rai-pseudonymization?lang=nl) verving je namen door codes in een tabel, en wezen een paar feiten mensen nog steeds aan. Hier verborg je namen en nummers in vrije tekst voordat die een pipeline in gaat: met een controlecijfer dat tikfouten opmerkt en een mogelijk BSN onderscheidt van de meeste andere getallen, en met een meting die laat zien wat erdoor glipt. Er glipt altijd iets door. In de volgende les, *Privacy by design*, wordt de vraag hoe je minder van zulke filters nodig hebt: door alleen te verzamelen wat het doel nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_anonymous, choice_pass, result_eleven, result_found, result_hide):
    L.wrap_up(result_eleven, result_hide, result_found, choice_pass, choice_anonymous)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
