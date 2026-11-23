# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-span-f1. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-span-f1", "nl")
    C = L.checks   # de tags en namen van de test set, vijf taggers, de scores van de course, de afbeeldingen en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen, over [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl) en [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_woodward():
    quick_woodward = L.choice(
        "quick-woodward",
        question="In *Namen vinden* gaf de tagger *Bob Woodward* de tags `B-PER I-PER`, hoewel *Woodward* nergens in de training set staat. Waar kwam de `I-PER` vandaan?",
        options={
            "gazetteer": "Uit de gazetteer, waarin *Bob Woodward* als persoon staat",
            "word": "Uit de tabel van woorden: *Woodward* na `B-PER`",
            "capital": "Uit een regel die mensen schreven: een woord met een hoofdletter na een voornaam is een achternaam",
            "right": "Uit de tabel van word shapes: `Xx` na `B-PER`",
        },
        correct="right",
        why={
            "right": "De tabel van woorden heeft geen paar voor een woord dat nergens in de training set staat, dus zocht de tagger de short word shape op: `Xx` na `B-PER` krijgt `I-PER`. Die les telde de goede namen, de foute vondsten en de gemiste namen. Deze les maakt van die aantallen een score.",
            "gazetteer": "De gazetteer heeft alleen namen uit de training set, en *Woodward* staat er niet in. Welke tabel gebruikte de tagger voor een woord dat de tagger nooit zag? Zie [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).",
            "word": "De tabel van woorden heeft alleen paren uit de training set, en *Woodward* staat nergens in de training set. Waar keek de tagger daarna? Zie [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).",
            "capital": "De tagger van die les heeft geen regels die mensen schreven: de tagger telt tags in de training set. Welke van de twee tabellen weet iets van een woord dat de tagger nooit zag? Zie [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl).",
        },
    )
    quick_woodward
    return (quick_woodward,)


@app.cell(hide_code=True)
def quick_check_f1():
    quick_f1 = L.choice(
        "quick-f1",
        question="Een detector heeft een precision van 1,0 en een recall van 0,1. Wat is de F1 score?",
        options={
            "mean": "0,55, precies tussen de twee getallen in",
            "low": "0,1, de kleinste van de twee",
            "right": "Ongeveer 0,18",
            "high": "1,0, de grootste van de twee",
        },
        correct="right",
        why={
            "right": "F1 is het harmonisch gemiddelde: 2 · 1,0 · 0,1 / (1,0 + 0,1) = 0,2 / 1,1, ongeveer 0,18. De F1 score blijft dicht bij het kleinste van de twee getallen, dus één hoog getal alleen geeft geen hoge F1 score. Deze les gebruikt de F1 score voor namen.",
            "mean": "Precies ertussen is het gewone gemiddelde. F1 blijft dicht bij het kleinste van de twee getallen. Welk gemiddelde gebruikt F1? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "low": "F1 blijft dicht bij het kleinste getal, maar niet erop: een precision van 1,0 telt nog mee. Wat geeft de formule? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "high": "F1 beloont niet één hoog getal: F1 blijft dicht bij het kleinste van de twee. Wat geeft de formule? Zie [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een tagger komt met een score, en de score beslist welke tagger gebruikt wordt. Dezelfde tagger kan er bijna perfect of nogal matig uitzien, afhankelijk van wat je telt. Jurafsky en Martin schrijven dat taggers van woordsoorten gescoord worden met de gewone accuracy, en taggers van named entities met recall, precision en F1. Deze les laat zien waarom namen de tweede manier nodig hebben: de meeste woorden van een tekst horen bij geen enkele naam, dus een score per woord meet vooral hoe goed een tagger woorden met rust laat.

    **Een voorbeeld uit het dagelijks leven.** Een rookmelder die nooit afgaat, heeft het bijna elke dag van het jaar goed, want op de meeste dagen is er geen brand. Toch wil niemand zo'n rookmelder: het gaat om de paar dagen met rook, en of de melder juist op die dagen afging, en alleen op die dagen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="De test set van Universal NER heeft 1.471 zinnen met 18.990 tokens, en 802 namen. Een tagger geeft elk token de tag `O`, en vindt dus geen enkele naam. Welk deel van de tokens tagt deze tagger goed?",
        options={
            "coin": "Ongeveer 50 van de 100, net als bij kop of munt",
            "quarter": "Ongeveer 75 van de 100",
            "right": "Ongeveer 94 van de 100",
            "almost": "Ongeveer 99 van de 100",
        },
        reveal="**Ongeveer 94:** 17.758 van de 18.990 tokens hebben de tag `O`, dus de tagger die nooit een naam vindt, heeft 93,5 van de 100 tokens goed. Maar 1.232 tokens, ongeveer 6 van de 100, horen bij een naam, en deze tagger vindt geen van de 802 namen.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Accuracy per token

    De **accuracy per token** van een tagger is het deel van alle tokens waarvan de tag de tag is die mensen gaven. Bij woordsoorten past dat: elk token heeft een woordsoort, en elke tag doet ertoe. Bij namen komt het grootste deel van dat getal van tokens met `O`. De tagger die elk token `O` geeft, krijgt 0,935 zonder één naam te vinden, en de tagger uit [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl) krijgt 0,958. Volgens de accuracy levert het vinden van 343 namen 0,023 op.

    ### De naam als eenheid

    Jurafsky en Martin schrijven dat bij het taggen van named entities niet het woord, maar de entity de eenheid is waarop je een tagger beoordeelt. Tel dus namen, zoals [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) cijfers telde. Noem elke span die een tagger markeert een **vondst**. Een vondst met dezelfde start, hetzelfde end en hetzelfde type als een naam die mensen markeerden, is een **exacte match**, een true positive. Elke andere vondst is een false positive. Een naam waar geen enkele vondst bij past, is een false negative. Precision is het deel van de vondsten dat goed is, recall het deel van de namen dat gevonden is, en de F1 score van de twee is de **span F1**.

    Erik Tjong Kim Sang en Fien De Meulder leidden de shared task van CoNLL-2003, waarin zestien systemen werden vergeleken op hetzelfde Engelse en Duitse nieuws. Volgens de twee is precision het percentage van de gevonden named entities dat goed is, en recall het percentage van de named entities in het corpus dat het systeem vindt. Een named entity telt alleen als goed bij een exacte match met de named entity in het bestand met de gegevens.

    De tagger uit *Namen vinden* doet 443 vondsten op de test set, waarvan 343 goed, en de test set heeft 802 namen. De precision van de tagger is 343 / 443 = 0,774, de recall 343 / 802 = 0,428, en de span F1 is 2 · 343 / (443 + 802) = 0,551. Dat is de formule met aantallen uit *Precision en recall in één getal*, 2 · tp / (2 · tp + fp + fn), want de vondsten (tp + fp) en de namen (tp + fn) zijn samen 443 + 802. De afbeelding zet beide scores van vijf taggers naast elkaar.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Met accuracy liggen alle vijf tussen 0,897 en 0,958; met span F1 tussen 0 en 0,551. De twee scores zijn het zelfs niet eens over de volgorde. De **hoofdletterregel** markeert elk woord met een hoofdletter dat niet het eerste woord van de zin is als deel van de naam van een persoon. Die regel geeft veel woorden een foute tag, dus de accuracy, 0,897, is lager dan de accuracy van de tagger die niets vindt. Maar 146 vondsten van de regel zijn goed, dus de span F1 is 0,136, en niet 0.

    ### Eén foute tag, twee fouten

    Volgens Jurafsky en Martin geeft het scoren van named entity recognition enkele problemen, omdat de taak ook segmentatie omvat: waar begint een naam, en waar houdt de naam op? Bij het indelen van teksten of het taggen van woordsoorten speelt dat niet. In het voorbeeld van Jurafsky en Martin maakt een systeem dat wel *Jane* maar niet *Jane Villanueva* als persoon markeert twee fouten. De afbeelding toont de voorbeeldzin uit het boek met een tagger die *Villanueva* de tag `O` geeft.
    """)
    return


@app.cell(hide_code=True)
def slip_cell():
    mo.Html(C.slip_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        Eén tag van de elf is fout, dus de accuracy is 0,909. Bij de namen is de vondst *Jane* geen exacte match, dus een false positive, en de naam *Jane Villanueva* is niet gevonden, een false negative. Jurafsky en Martin wijzen nog op een verschil: bij het scoren is de entity de eenheid, maar bij het leren het woord, dus training en test lopen niet gelijk. De tagger uit *Namen vinden* leerde tag voor tag, en wordt naam voor naam gescoord.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **accuracy per token** | het deel van alle tokens waarvan de tag de tag is die mensen gaven |
        | **vondst** | een span die een tagger als naam markeert, met het type |
        | **exacte match** | een vondst met dezelfde start, hetzelfde end en hetzelfde type als een naam die mensen markeerden |
        | **span F1** | precision, recall en de F1 score van de twee over namen, waarbij alleen een exacte match als goed telt |
        | **hoofdletterregel** | een tagger die elk woord met een hoofdletter dat niet het eerste woord van de zin is, markeert als deel van de naam van een persoon |
        | `C.GOLD` | de tags die mensen gaven in de test set: een lijst met een lijst van tags voor elke zin |
        | `C.TAGGERS` | de tags van de vijf taggers op de test set, in een dict met de keys `"all_o"`, `"one_word"`, `"gazetteer"`, `"tagger"` en `"capitals"`; elke waarde is gebouwd zoals `C.GOLD` |
        | `C.GOLD_SPANS`, `C.FOUND` | hetzelfde als spans: voor elke zin een lijst van spans `(start, end, kind)`; `C.FOUND` bevat de vondsten van elke tagger, onder dezelfde keys |
        | `C.span_f1` | de versie van de course van Oefening 2 |
        | `gold[s][k]` | item `k` van item `s`: hier de tag van token `k` in zin `s` |
        """),
        L.watch_out("Een score is maar zo goed als de namen die mensen markeerden. In Universal NER markeerden twee mensen een deel van de zinnen, en de makers, Mayhew en anderen, schrijven dat de twee het doorgaans minder eens zijn over organisaties en plaatsen dan over personen, omdat sommige namen beide types kunnen hebben: een restaurant is een plaats als je er gaat eten, maar een organisatie als het restaurant een nieuwe kok aanneemt. Een tagger die zo'n naam het andere type geeft, telt als fout."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Accuracy per token: ga alle tokens van alle zinnen langs, tel de tokens, en tel ook de tokens waarvan de twee tags gelijk zijn. Deel het tweede aantal door het eerste.
    2. Span F1: tel voor elke zin de vondsten, de namen, en de vondsten die ook een naam in dezelfde zin zijn. Tel de aantallen op over de zinnen. Precision is goede vondsten / vondsten, recall goede vondsten / namen, en F1 is 2 · precision · recall / (precision + recall). Is geen enkele vondst goed, dan zijn alle drie 0.
    3. Per type: houd voor elke zin alleen de spans van één type over, bij de namen en bij de vondsten, en bereken span F1 op wat overblijft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel telt de tokens van de test set en de tokens met de tag `O`, en print beide aantallen en het deel. Daarna telt de cel de namen die mensen markeerden.
    """)
    return


@app.cell
def example_count():
    tokens = 0
    outside = 0
    for sentence_tags in C.GOLD:
        for tag in sentence_tags:
            tokens = tokens + 1
            if tag == "O":
                outside = outside + 1
    print(tokens, outside, outside / tokens)

    names = 0
    for sentence_spans in C.GOLD_SPANS:
        names = names + len(sentence_spans)
    print(names)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het deel van de tokens met `O` is de accuracy van de tagger die elk token `O` geeft. Oefening 1 vergelijkt in plaats daarvan twee lijsten met tags: de tags die mensen gaven, en de tags van een tagger.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. token_accuracy: het deel goede tags

        Schrijf `token_accuracy(gold, guess)`: `gold` en `guess` bevatten elk één lijst met tags voor elke zin; de twee lijsten van een zin zijn even lang. Geef het deel van alle tokens van alle zinnen terug waarvan de tag in `guess` de tag in `gold` is. Ga de zinnen langs met een positie `s`, en de tokens van een zin met een positie `k`: `gold[s][k]` is de tag die mensen token `k` van zin `s` gaven. Vervang de `...`. De check roept `token_accuracy([["B-PER", "I-PER", "O"], ["O", "B-LOC"]], [["B-PER", "O", "O"], ["O", "B-LOC"]])` aan, wat `0.8` geeft; daarna eigen gevallen: elke tag goed, geen enkele tag goed in twee zinnen, één token met een ander type, een zin van negen tokens en een zin van twee, en `C.GOLD` met `C.TAGGERS["all_o"]` en met `C.TAGGERS["tagger"]`.
        """),
        L.task("Geef het deel van de tokens met een goede tag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    box_accuracy = L.code_box("token_accuracy", lines=9)
    box_accuracy
    return (box_accuracy,)


@app.cell(hide_code=True)
def check_accuracy(box_accuracy):
    result_accuracy = L.check("token_accuracy", box_accuracy.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "token_accuracy",
        [
            "Twee tellers, `same` en `total`, aan het begin allebei 0. Een loop `for s in range(len(gold)):` met daarin een loop `for k in range(len(gold[s])):`. `total` gaat omhoog bij elk token, `same` alleen als de tag in `gold` en de tag in `guess` op die positie gelijk zijn.",
            "Binnen de binnenste loop vergelijkt een `if` de twee tags van token `k` in zin `s`. Na beide loops is het deel de goede tokens gedeeld door alle tokens.",
            "`            if gold[s][k] == ___:`",
        ],
        why="`same = 0` en `total = 0`, dan `for s in range(len(gold)):` met `for k in range(len(gold[s])):` eronder; daarin `if gold[s][k] == guess[s][k]:` met `same = same + 1` eronder, en `total = total + 1`; na de loops `return same / total`. Het voorbeeld geeft `0.8`.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracy():
    mo.md(r"""
    De afbeelding onder de check draait je functie op de tags van de vijf taggers. Een zin van negen tokens weegt zwaarder dan een zin van twee, omdat elk token één keer telt. Oefening 2 telt namen in plaats van tokens.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_f1():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. span_f1: precision, recall en F1 over namen

        Schrijf `span_f1(gold, found)`: `gold` bevat de namen die mensen markeerden, en `found` de vondsten van een tagger, elk als één lijst van spans `(start, end, kind)` voor elke zin. Tel de vondsten, de namen, en de goede vondsten: een vondst is goed als de vondst een van de namen van dezelfde zin is, met dezelfde start, hetzelfde end en hetzelfde type. Geef `(precision, recall, f1)` terug, en `(0.0, 0.0, 0.0)` als geen enkele vondst goed is. Een tagger vindt elke span één keer. Vervang de `...`. De check roept `span_f1([[(0, 2, "PER"), (3, 5, "ORG"), (7, 9, "LOC")]], [[(0, 1, "PER"), (3, 5, "ORG")]])` aan: de namen van *Jane Villanueva of United Airlines flew to San Diego .*, met *Jane* gevonden als persoon, *United Airlines* goed en *San Diego* gemist, wat `(0.5, 0.3333333333333333, 0.4)` geeft. Daarna eigen gevallen: niets gevonden, elke naam gevonden in twee zinnen, dezelfde woorden met een ander type, een vondst die past bij een naam uit een andere zin, meer vondsten dan namen, een zin zonder namen en zonder vondsten, en `C.GOLD_SPANS` met `C.FOUND["gazetteer"]` en met `C.FOUND["tagger"]`.
        """),
        L.task("Scoor de vondsten van een tagger met precision, recall en F1 over namen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_f1():
    box_f1 = L.code_box("span_f1", lines=16)
    box_f1
    return (box_f1,)


@app.cell(hide_code=True)
def check_f1(box_f1):
    result_f1 = L.check("span_f1", box_f1.value)
    result_f1
    return (result_f1,)


@app.cell(hide_code=True)
def hints_f1():
    L.hints(
        "span_f1",
        [
            "Drie tellers: `right`, `n_found` en `n_gold`. Een loop `for s in range(len(gold)):`, en daarin een loop `for span in found[s]:` met een `if` die vraagt of `span` een van de namen is. Tel na de binnenste loop het aantal vondsten en het aantal namen van die zin bij de tellers op. Vang `right == 0` op voordat je deelt.",
            "Met welke namen mag je een vondst van zin `s` vergelijken: met de namen van alle zinnen, of alleen met de namen van één zin?",
            "`            if span in ___:`",
        ],
        why="`right = 0`, `n_found = 0` en `n_gold = 0`, dan `for s in range(len(gold)):` met `for span in found[s]:` en `if span in gold[s]:` met `right = right + 1` eronder; na de binnenste loop `n_found = n_found + len(found[s])` en `n_gold = n_gold + len(gold[s])`. Na de loop `if right == 0:` met `return (0.0, 0.0, 0.0)`; dan `precision = right / n_found`, `recall = right / n_gold`, `f1 = 2 * precision * recall / (precision + recall)` en `return (precision, recall, f1)`. Het voorbeeld geeft `(0.5, 0.3333333333333333, 0.4)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_f1():
    mo.md(r"""
    Met span F1 liggen de vijf taggers tussen 0 en 0,551 in plaats van binnen 0,061 van elkaar. Van de 100 foute vondsten van de tagger hebben er 26 de woorden van een naam met een ander type, overlappen er 40 een naam zonder exacte match, en liggen er 34 waar geen naam is; de tagger mist 459 namen. Oefening 3 splitst de score per type.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kind():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. of_kind: de namen van één type

        Schrijf `of_kind(per_sentence, kind)`: `per_sentence` bevat één lijst van spans voor elke zin, zoals `C.GOLD_SPANS` of `C.FOUND["tagger"]`. Geef een lijst terug met voor elke zin een lijst van alleen de spans van type `kind`, in de volgorde waarin de spans staan. Een zin zonder zulke spans krijgt een lege lijst, zodat de lijsten gelijk lopen met de zinnen. Dan kan `C.span_f1` één type scoren: `C.span_f1(of_kind(C.GOLD_SPANS, "ORG"), of_kind(C.FOUND["tagger"], "ORG"))`. Vervang de `...`. De check roept `of_kind([[(0, 2, "PER"), (3, 5, "ORG")], [(1, 2, "LOC")]], "PER")` aan, wat `[[(0, 2, 'PER')], []]` geeft; daarna eigen gevallen: een type dat geen enkele span heeft, geen zinnen, twee namen van het type in één zin met een ander type ertussen, een zin zonder namen tussen twee andere zinnen, en `C.GOLD_SPANS` met `"ORG"`.
        """),
        L.task("Houd voor elke zin alleen de namen van één type over."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kind():
    box_kind = L.code_box("of_kind", lines=5)
    box_kind
    return (box_kind,)


@app.cell(hide_code=True)
def check_kind(box_kind):
    result_kind = L.check("of_kind", box_kind.value)
    result_kind
    return (result_kind,)


@app.cell(hide_code=True)
def hints_kind():
    L.hints(
        "of_kind",
        [
            "Begin met een lege lijst `out` en een loop `for sentence_spans in per_sentence:`. Maak voor elke zin een lijst van de spans waarvan het type `kind` is, met een list comprehension die elke `span` van de zin langsgaat en een `if` aan het eind heeft, en voeg die lijst toe aan `out`, ook als de lijst leeg is.",
            "Een span is `(start, end, kind)`. Op welke positie in de span staat het type?",
            "`        out.append([span for span in sentence_spans if ___ == kind])`",
        ],
        why="`out = []`, dan `for sentence_spans in per_sentence:` met `out.append([span for span in sentence_spans if span[2] == kind])` eronder; na de loop `return out`. Het voorbeeld geeft `[[(0, 2, 'PER')], []]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_kind():
    mo.md(r"""
    Per type pakken de twee verschillend uit. De tagger is beter in personen, 0,527 tegen 0,334 voor de gazetteer: de tag ervoor zet een naam voort in woorden die de tagger nooit zag. De gazetteer is beter in organisaties, 0,491 tegen 0,349: een organisatie die de gazetteer kent, vindt de gazetteer helemaal. Over alle types liggen de twee dicht bij elkaar: 0,551 voor de tagger, 0,521 voor de gazetteer. Mayhew en anderen, de makers van Universal NER, geven beide soorten getallen voor een veel groter model, getraind op dezelfde Engelse training set: een span F1 van 0,858 over alle namen op de eigen test set, waarin ook de e-mails staan die deze course weglaat, en 0,950 voor personen, 0,871 voor plaatsen en 0,709 voor organisaties.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cut():
    mo.md(r"""
    ### Oefening 4. Een halve naam

    In *I have used Bright Futures for the last 7 years .* markeerden mensen *Bright Futures* als organisatie. Tagger A markeert *Bright Futures for the* als organisatie. Tagger B markeert niets in de zin.
    """)
    return


@app.cell(hide_code=True)
def choice_cut_cell():
    choice_cut = L.choice(
        "near-miss",
        question="Wat voegt elke tagger toe aan de aantallen van span F1?",
        options={
            "same": "Beide voegen één gemiste naam toe, verder niets",
            "right": "A voegt een foute vondst en een gemiste naam toe; B alleen een gemiste naam",
            "half": "A krijgt een halve goede vondst, omdat de vondst beide woorden van de naam bevat",
            "b": "B voegt een gemiste naam en een foute vondst toe; A een gemiste naam",
        },
        correct="right",
        why={
            "right": "De vondst van A is geen exacte match, dus de vondst telt als fout, en de naam zelf blijft gemist: precision en recall dalen allebei. B mist alleen de naam. Span F1 straft een verkeerd afgeknipte naam dus zwaarder dan een naam die helemaal niet gevonden is. Voor wie de uitkomst leest, is *Bright Futures for the* niet de naam van de organisatie.",
            "same": "Voor B klopt dat. Maar A markeert ook iets dat mensen niet als naam markeerden. Hoe telt die vondst?",
            "half": "Span F1 kent geen halve vondsten: een vondst is goed met dezelfde start, hetzelfde end en hetzelfde type, en anders fout. Welke van de twee is de vondst van A?",
            "b": "B markeert niets, dus B heeft geen vondsten om te tellen. Hoe telt de vondst van A, naast de naam die niet gevonden is?",
        },
    )
    choice_cut
    return (choice_cut,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: één zin, per token en per naam

    Kies een tagger en een zin uit de test set, en bekijk de tags van de tagger naast de tags die mensen gaven, en de vondsten naast de namen. Probeer *Dr. White is the best !* met de tagger: één foute tag, en twee fouten bij de namen. Kies daarna *Elk token O*, en de hoofdletterregel.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_tagger = mo.ui.dropdown(options=C.tagger_options(L.lang), value=C.TAGGER_NAMES[L.lang]["tagger"], label="Tagger")
    demo_sentence = mo.ui.dropdown(options=C.sentence_options(), value=list(C.sentence_options())[1], label="Zin")
    mo.hstack([demo_tagger, demo_sentence], justify="start", gap=1.5, wrap=True)
    return demo_sentence, demo_tagger


@app.cell(hide_code=True)
def demo_look(demo_sentence, demo_tagger):
    mo.Html(C.demo_picture(demo_tagger.value, demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Namen vinden: named entity recognition](/course/open/nlp-named-entities?lang=nl) telde de goede namen, de foute vondsten en de gemiste namen; hier werden die aantallen een score. Accuracy per token, de score van [Woordsoorten taggen](/course/open/nlp-pos-tagging?lang=nl), zet vijf taggers tussen 0,897 en 0,958, en geeft de tagger die niets vindt 0,935. Span F1 gebruikt de precision en recall van [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) met een naam als eenheid, alleen goed als exacte match: de tagger krijgt 0,551, en een verkeerd afgeknipte naam kost een foute vondst en een gemiste naam. Per type is de tagger beter in personen en de gazetteer in organisaties. De volgende les, *Volgt de ene zin uit de andere?*, gaat van namen naar de betekenis van hele zinnen.

    **Een voorbeeld uit het dagelijks leven.** Een envelop aan *Villanueba* heeft 9 van de 10 letters van *Villanueva* goed, en kan toch bij de verkeerde persoon terechtkomen. Bij een naam telt alleen de hele naam.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cut, result_accuracy, result_f1, result_kind):
    L.wrap_up(result_accuracy, result_f1, result_kind, choice_cut)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
