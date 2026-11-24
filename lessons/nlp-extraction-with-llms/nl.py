# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-extraction-with-llms. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-extraction-with-llms", "nl")
    C = L.checks   # de aankondigingen van de course, de run van Qwen3-0.6B, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl), en één over [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="Een regel haalt 40 rijen uit een tekst, en 30 daarvan kloppen. De gold standard heeft 50 rijen. Wat zijn de precision en de recall van de regel?",
        options={
            "swapped": "Precision 0,60, recall 0,75",
            "right": "Precision 0,75, recall 0,60",
            "found": "Precision 0,75, recall 0,80: 40 van de 50 rijen zijn gevonden",
            "same": "Allebei 0,75",
        },
        correct="right",
        why={
            "right": "Precision is het deel van de gevonden rijen dat klopt, 30 / 40 = 0,75; recall is het deel van de goede rijen dat gevonden is, 30 / 50 = 0,60. Deze les telt op dezelfde manier welke waarden van een chat model kloppen.",
            "swapped": "Welke van de twee kijkt naar de rijen die de regel vond, en welke naar de rijen van de gold standard? Zie [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl).",
            "found": "De regel vond 40 rijen, maar slechts 30 daarvan kloppen. Welke rijen telt recall? Zie [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl).",
            "same": "De twee delen dezelfde 30 goede rijen door verschillende getallen. Door welke? Zie [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl).",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def quick_check_shots():
    quick_shots = L.choice(
        "quick-shots",
        question="Wat is few-shot prompting?",
        options={
            "role": "Het model een rol geven, zoals docent",
            "steps": "Om de stappen vragen, en pas daarna om het antwoord",
            "right": "Een paar afgemaakte voorbeelden van de taak in de prompt zetten",
            "again": "Dezelfde vraag een paar keer stellen, en het antwoord nemen dat het vaakst terugkomt",
        },
        correct="right",
        why={
            "right": "Elk voorbeeld is een invoer met het antwoord dat je wilt, en daarna volgt de echte invoer, zodat het model hetzelfde patroon volgt. Deze les laat een chat model twee afgemaakte voorbeelden zien van een formulier om in te vullen.",
            "role": "Dat is role prompting. Waar staan de shots van few-shot prompting voor? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "steps": "Dat is chain of thought. Waar staan de shots van few-shot prompting voor? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "again": "Meer dan eens vragen is een manier om een antwoord na te kijken. Waar staan de shots van few-shot prompting voor? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
        },
    )
    quick_shots
    return (quick_shots,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In *Van tekst naar een tabel* haalde je de datums uit tekst met een reguliere expressie: één patroon voor elke vorm die een datum kan hebben. Een aankondiging van een activiteit heeft meer velden, en elk veld kan op veel manieren geschreven zijn: een tijd is *19.30 uur*, *vanaf 21.00 uur* of *de eerste ronde begint om 9.30 uur*, en een plaats is een gebouw, een straat of een park. Een patroon schrijven voor elk veld en elke vorm is traag werk.

    Een chat model kan de velden invullen met een prompt: zeg welke velden je wilt, laat een paar afgemaakte voorbeelden zien, en geef het model de tekst. Monica Agrawal en anderen probeerden dat in 2022 met GPT-3 op notities van artsen: op de klinische taken die de auteurs onderzochten, deden de systemen met GPT-3 het significant beter dan bestaande zero-shot- en few-shot-baselines. Maar een chat model schrijft vaak ook iets bij een veld dat niet in de tekst staat. In deze les vult Qwen3-0.6B, het kleine chat model van de course, vijf velden in voor twaalf aankondigingen, en zoek je elke waarde op in de aankondiging waar de waarde vandaan komt.

    **Een voorbeeld uit het dagelijks leven.** Onder een e-mail met een treinkaartje laat je telefoon een knop *Toevoegen aan agenda* zien: de telefoon heeft de datum, de tijd en het station uit de tekst gelezen. Een verkeerde tijd in je agenda is erger dan helemaal geen tijd, dus een zorgvuldige app laat eerst zien wat de app vond, en slaat pas daarna iets op.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_made_up = L.guess(
        "guess",
        question="De course schreef de twaalf aankondigingen van deze les, en liet met opzet acht velden weg: één plaats, vier prijzen, twee tijden en één datum. De prompt vraagt Qwen3 om - te schrijven bij een veld dat niet in de tekst staat, en laat twee voorbeelden zien, waarvan één met een -. Hoeveel van de acht velden vult Qwen3 toch in, in de antwoorden met seed 0?",
        options={
            "none": "Geen: de prompt zegt dat het model - moet schrijven",
            "few": "Eén of twee",
            "right": "Ongeveer de helft",
            "all": "Alle acht",
        },
        reveal="Vijf van de acht. Qwen3 schreef - bij de prijs van de voorjaarsschoonmaak en van de kookworkshop, zoals de prompt vraagt, en bij de prijs van de open dag, omdat Qwen3 daar het hele voorbeeld 2 overnam. Bij de andere vijf schreef Qwen3 een waarde: een plaats voor de zwemles, een prijs voor Sterrennacht, een tijd voor de filmavond en voor de tuinroute, en een datum voor jazz in het park. Over de vier seeds vult Qwen3 er 18 van de 32 in.",
    )
    guess_made_up
    return (guess_made_up,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Information extraction** vult de velden van een template in uit vrije tekst. Met een chat model gaat het template in de prompt, in de delen uit *Goed vragen*: een **instructie** die de velden noemt en zegt hoe het antwoord eruit moet zien, een paar **voorbeelden** met het afgemaakte antwoord, en dan de **tekst**. Tags zoals `<text>` en `</text>` om elke tekst heen zijn **delimiters**: de tags laten zien waar een tekst begint en eindigt. Het model gaat verder na de laatste tekst, en schrijft één regel per veld: `datum: vrijdag 20 maart`. Met voorbeelden is dat few-shot prompting; zonder voorbeelden zero-shot prompting.

    Het model schrijft vaak een waarde bij elk veld waar je om vraagt, ook als de tekst geen waarde geeft. Een waarde die in de tekst staat, heeft het model overgenomen. Een waarde die nergens in de tekst staat, komt ergens anders vandaan: uit een voorbeeld in de prompt, uit wat het model tijdens de training zag, of uit het niets. Een waarde die zeker klinkt maar geen bron heeft, is een **hallucinatie**. Dus **toets je elke waarde aan de bron**: je zoekt elke waarde op in de tekst waar de waarde vandaan moet komen, en elke waarde die er niet in staat, gaat naar een mens om naar te kijken, of blijft leeg.

    Het plaatje laat één aankondiging zien, de prompt die Qwen3 ervoor kreeg, en het antwoord van Qwen3.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Qwen3 houdt zich aan de vorm van de voorbeelden: één regel per veld. Vier van de waarden staan in de aankondiging. De plaats niet: de aankondiging zegt niet waar de lessen zijn, en Qwen3 schreef *de muziekschool*, van de open avond in voorbeeld 2 van de prompt. Als je de waarde opzoekt in de aankondiging, vind je de fout, want de woorden staan nergens in de aankondiging.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **hallucinatie** | uitvoer die zeker klinkt maar fout of verzonnen is, zoals een prijs die niet in de aankondiging staat |
    | **bron** | de tekst waar een waarde vandaan moet komen: hier de aankondiging |
    | **waarden toetsen aan de bron** | elke waarde die een model invulde opzoeken in de bron, en de waarden markeren die er niet in staan |
    | `a in b`, `a not in b` | voor twee strings: `True` als de tekst `a` wel, of juist niet, ergens in `b` staat |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf de instructie: de velden, met een paar woorden bij elk veld, de vorm van het antwoord, en wat er moet staan bij een veld dat niet in de tekst staat.
    2. Voeg twee of drie voorbeelden toe met het afgemaakte antwoord, één ervan met een veld dat ontbreekt.
    3. Zet de tekst tussen tags erachter.
    4. Lees het antwoord regel na regel in als een rij velden, en laat de velden met `-` weg.
    5. Zoek elke waarde op in de tekst, en markeer de waarden die er niet in staan.
    6. Laat een mens naar de gemarkeerde waarden kijken, of laat de velden leeg.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De aankondiging van de zwemles zegt wanneer de lessen beginnen en wat de lessen kosten, maar niet waar de lessen zijn. Hier is de rij uit het antwoord van Qwen3, met de twee voorbeelden en seed 0, en elke waarde met de hand opgezocht in de aankondiging:
    """)
    return


@app.cell
def example_cell():
    notice = "Zwemles voor volwassenen begint op maandag 16 maart om 19.30 uur. Tien lessen kosten € 60."
    row = {"wat": "Zwemles", "datum": "maandag 16 maart", "tijd": "19.30",
           "plaats": "de muziekschool", "prijs": "€ 60"}
    for field in row:
        print(field, "|", row[field], "|", row[field].lower() in notice.lower())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vier waarden geven `True`: de waarden staan in de aankondiging. *Zwemles* is korter dan de naam in de aankondiging, maar het woord staat erin. De plaats geeft `False`: *de muziekschool* staat niet in de aankondiging, wel in voorbeeld 2 van de prompt. De waarde en de aankondiging gaan allebei eerst naar kleine letters, zodat *Maandag* en *maandag* als hetzelfde tellen. In oefening 3 schrijf je deze check als functie voor een hele rij.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prompt():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. fields_prompt: de prompt

        Schrijf `fields_prompt(instruction, examples, text)`. `instruction` is een tekst, `examples` een lijst tuples van een tekst en het afgemaakte antwoord, en `text` de tekst waarvoor de velden ingevuld moeten worden. Geef één tekst terug: de instructie en een lege regel, dan voor elk voorbeeld `<text>` op een eigen regel, de tekst van het voorbeeld, `</text>` op een eigen regel, het antwoord en een lege regel, en aan het eind `<text>`, de tekst en `</text>`, elk op een eigen regel. `"\n"` in een string is een regeleinde, dus `"\n\n"` maakt een lege regel. Vervang de twee keer `...`.

        De check roept `fields_prompt("Vul twee velden in: wat en datum.", [("Quiz in het café op vrijdag 6 maart.", "wat: Quiz\ndatum: vrijdag 6 maart")], "Bingo op maandag 9 maart.")` aan, wat `"Vul twee velden in: wat en datum.\n\n<text>\nQuiz in het café op vrijdag 6 maart.\n</text>\nwat: Quiz\ndatum: vrijdag 6 maart\n\n<text>\nBingo op maandag 9 maart.\n</text>"` geeft, en hetzelfde in het Engels. Daarna eigen gevallen: geen voorbeelden, en de instructie en de twee voorbeelden van de course voor Sterrennacht, in het Engels en in het Nederlands (in de les: `C.INSTRUCTION`, `C.SHOTS` en `C.NOTICES`).
        """),
        L.task("Bouw de prompt die om de velden vraagt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prompt():
    box_prompt = L.code_box("fields_prompt", lines=9)
    box_prompt
    return (box_prompt,)


@app.cell(hide_code=True)
def check_prompt(box_prompt):
    result_prompt = L.check("fields_prompt", box_prompt.value)
    result_prompt
    return (result_prompt,)


@app.cell(hide_code=True)
def hints_prompt():
    L.hints(
        "fields_prompt",
        [
            "In de loop groeit `prompt` met één voorbeeld tegelijk: de tekst van het voorbeeld tussen de twee tags, dan het `answer`. Na de loop geeft `return` de `prompt` terug, met de `text` tussen de tags aan het eind.",
            'Strings plak je aan elkaar met `+`: `"<text>\\n" + example` zet de tag op een eigen regel, boven de tekst. Wat komt er na de tekst, en wat na het antwoord?',
            "`    return prompt + ___`",
        ],
        why='In de loop: `prompt = prompt + "<text>\\n" + example + "\\n</text>\\n" + answer + "\\n\\n"`. Erna: `return prompt + "<text>\\n" + text + "\\n</text>"`.',
    )
    return


@app.cell(hide_code=True)
def text_after_prompt():
    mo.md(r"""
    Zonder voorbeelden begint Qwen3 elke regel met een streepje, maar de regels `veld: waarde` zijn er: alle 48 antwoorden zonder voorbeelden (twaalf aankondigingen, vier seeds) hebben een regel voor elk van de vijf velden. Alleen schrijft Qwen3 zonder voorbeelden nooit `-` als waarde: bij een veld dat niet in de aankondiging staat, schrijft Qwen3 woorden uit de instructie, zoals *geen prijs (van een kaartje of de toegang)* bij Sterrennacht. Alle 32 velden die niet in de aankondigingen staan (acht velden, vier seeds), vult Qwen3 zonder voorbeelden in; met de twee voorbeelden 18. Voorbeeld 2 laat zien hoe een `-` eruitziet, waar de instructie het alleen beschrijft. De voorbeelden maken de prompt wel langer: voor Sterrennacht 369 tokens in plaats van 164.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. read_fields: het antwoord als rij

        Schrijf `read_fields(answer)`, met `answer` de tekst die het model schreef. Geef een dictionary terug van elk veld naar de waarde, voor elk deel van het antwoord in de vorm veld, dubbele punt, spatie en een waarde tot het eind van de regel, zoals `datum: vrijdag 20 maart`. Haal met `.strip()` de spaties aan beide kanten van elke waarde weg: Qwen3 eindigt een regel vaak met twee spaties. Laat een veld met de waarde `-` weg. De naam van een veld bestaat uit woordtekens, `\w`. Vervang de twee keer `...`: het patroon, met een groep voor het veld en een groep voor de waarde, en de regels in de loop.

        De check roept `read_fields("wat: Bingo  \ndatum: maandag 9 maart  \nprijs: -")` aan, wat `{"wat": "Bingo", "datum": "maandag 9 maart"}` geeft, en hetzelfde in het Engels. Daarna eigen gevallen: een leeg antwoord, en antwoorden van Qwen3 uit de run van de course, met en zonder voorbeelden (in de les: `C.answer(lang, way, i)["text"]`).
        """),
        L.task("Lees het antwoord van het model in als een rij velden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read_fields", lines=10)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read_fields", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read_fields",
        [
            "`re.findall` met een patroon van twee groepen geeft een tuple `(field, value)` voor elke match in `answer`. In de loop haalt `value.strip()` de spaties weg, en `if value != \"-\":` zet de waarde in `row` onder het `field`.",
            "Een groep is een deel van het patroon tussen ronde haakjes. `\\w+` matcht de naam van een veld, en `.` elk teken behalve een regeleinde: wat matcht de rest van een regel na `: `?",
            '`    for field, value in re.findall(r"(\\w+): ___", answer):`',
        ],
        why='Het patroon is `r"(\\w+): (.+)"`. In de loop: `value = value.strip()`, dan `if value != "-":` met `row[field] = value` eronder.',
    )
    return


@app.cell(hide_code=True)
def text_after_read():
    mo.md(r"""
    Zonder voorbeelden staat er bij de zwemles *plaats: niet in de tekst staat*. Je `read_fields` leest dat als een waarde, want alleen een `-` laat de functie weg: een rij bevat wat het antwoord zegt, ook als het antwoord de instructie herhaalt. Met de voorbeelden staat er bij Sterrennacht *wat: Open*, het begin van *Open avond* uit voorbeeld 2, en *prijs: warme kleren*. De vorm klopt, dus de rij is compleet; of de waarden kloppen, kijkt oefening 3 na. Van de twaalf antwoorden van seed 0 leest je functie zonder voorbeelden alle vijf velden uit alle twaalf; met de voorbeelden uit negen, en drie andere hebben een `-` waar de aankondiging niets geeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. unsupported: waarden die niet in de bron staan

        Schrijf `unsupported(row, text)`, met `row` een dictionary van velden en waarden, zoals `read_fields` een rij geeft, en `text` de bron. Geef een lijst terug van de velden waarvan de waarde niet in de tekst staat, in de volgorde van de rij. Hoofdletters mogen niet uitmaken: vergelijk allebei in kleine letters, met `.lower()`. Vervang de `...`.

        De check roept `unsupported({"wat": "Bingo", "datum": "maandag 9 maart", "prijs": "gratis"}, "Bingo op maandag 9 maart.")` aan, wat `["prijs"]` geeft, en hetzelfde in het Engels. Daarna eigen gevallen: een lege rij, een rij met andere hoofdletters, en rijen uit de antwoorden van Qwen3 met de aankondigingen (in de les: `C.read_fields(C.answer(lang, way, i)["text"])` en `C.NOTICES[lang][i][0]`).
        """),
        L.task("Markeer de waarden die niet in de bron staan."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flag():
    box_flag = L.code_box("unsupported", lines=8)
    box_flag
    return (box_flag,)


@app.cell(hide_code=True)
def check_flag(box_flag):
    result_flag = L.check("unsupported", box_flag.value)
    result_flag
    return (result_flag,)


@app.cell(hide_code=True)
def hints_flag():
    L.hints(
        "unsupported",
        [
            "In de loop is `value.lower()` de waarde in kleine letters. Met `if ... not in ...:` vraag je of de waarde ontbreekt in de tekst, ook in kleine letters, en dan voegt `fields.append(field)` het veld toe.",
            "`not in` werkt op twee strings: staat de ene nergens in de andere? Welke van de twee moet in welke gevonden worden, en wat hebben de twee allebei nodig, zodat hoofdletters niet tellen?",
            "`        if value.lower() not in ___:`",
        ],
        why="`if value.lower() not in text.lower():` met `fields.append(field)` eronder.",
    )
    return


@app.cell(hide_code=True)
def text_after_flag():
    mo.md(r"""
    Je functie markeert 12 van de 57 waarden. Daar zitten fouten tussen die nergens in de aankondiging staan: drie keer *Open* als naam van de activiteit, en bij de open dag het hele antwoord van voorbeeld 2, *Open avond* op donderdag 12 maart om 19.00 in de muziekschool. Van de acht velden die niet in de aankondigingen staan, vulde Qwen3 er vijf in, en drie daarvan markeert je functie: *de muziekschool*, *19.00* en *12 uur*. Twee komen erdoor, *warme kleren* en *14.00*, net als *tuinroute* als plaats van de tuinroute. Eén waarde met het goede bedrag erin markeert de check ook: *€ 2 (parkeren)* bij de rommelmarkt, met een woord erbij dat niet klopt, want parkeren kost € 4. Over de vier seeds vult Qwen3 met de voorbeelden 18 van de 32 velden in, en de check markeert er 14 van.
    """)
    return


@app.cell(hide_code=True)
def text_long():
    mo.md(r"""
    ### Een lang document

    Een echte tekst bevat vaak meer dan één activiteit. Hier staan alle twaalf aankondigingen in één nieuwsbrief van 1.577 tekens, en de prompt eindigt met een regel die de activiteit noemt: *Vul de velden alleen in voor deze activiteit: Kookworkshop*. De waarden moeten nu uit één deel van een langere tekst komen. Met seed 0 geeft Qwen3 de kookworkshop een prijs van € 5, het inschrijfgeld van het schaaktoernooi, en de open dag de datum, de tijd, de plaats en de prijs van de tuinroute en de kookworkshop. Zoek je de waarden op in de hele nieuwsbrief, dan staan de waarden er gewoon in, en komen de waarden erdoor: over de vier seeds markeert de check dan 26 waarden, waar de check er 47 markeert als je in de aankondiging van de activiteit zoekt. Zoek elke waarde dus op in het deel van de tekst waar de waarde vandaan moet komen, niet in het hele document. De tabel telt alle drie de manieren.
    """)
    return


@app.cell(hide_code=True)
def table_cell():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: elk antwoord uit de run van de course

    Kies een manier van vragen, een seed en een aankondiging, en bekijk de prompt, wat Qwen3 schreef, en elke waarde opgezocht in de aankondiging. Met de nieuwsbrief wordt elke waarde ook opgezocht in de hele nieuwsbrief. Zoek een waarde die de check doorlaat, hoewel de waarde fout is.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_way = mo.ui.radio(options={"zonder voorbeelden": "zero", "met de twee voorbeelden": "examples", "met de nieuwsbrief": "long"},
                          value="met de twee voorbeelden", label="Manier van vragen")
    try_seed = mo.ui.slider(0, 3, value=0, label="Seed", show_value=True)
    try_notice = mo.ui.dropdown(options=C.notice_options(L.lang), value="2. Zwemles voor volwassenen", label="Aankondiging")
    mo.hstack([try_way, try_seed, try_notice], justify="start", gap=1.5, wrap=True)
    return try_notice, try_seed, try_way


@app.cell(hide_code=True)
def try_look(try_notice, try_seed, try_way):
    mo.Html(C.try_picture(L.lang, try_way.value, try_seed.value, try_notice.value))
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Vraag het Qwen3 zelf

    Schrijf een eigen aankondiging, en laat Qwen3-0.6B de vijf velden invullen, met de instructie en de twee voorbeelden van de course en seed 0. Laat een veld weg, zoals de prijs, en kijk of Qwen3 `-` schrijft of iets anders invult. Qwen3 draait op je eigen computer, dus niets van wat je typt gaat de computer uit.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het model één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base. De rest van de les werkt zonder Qwen3: de les leest de antwoorden die de course opnam.*",
        "busy": "**Downloaden:** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het bestand weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def notice_box_cell():
    notice_box = mo.ui.text_area(value=C.YOUR_NOTICE[L.lang], label="Je aankondiging", rows=3, full_width=True)
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([notice_box, ask_button])
    return ask_button, notice_box


@app.cell(hide_code=True)
def demo_answer(ask_button, notice_box):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen.*")
    elif not notice_box.value.strip():
        _out = mo.md("*Schrijf eerst een aankondiging in het vak.*")
    else:
        _out = mo.Html(C.ask_live(notice_box.value, lambda _html: mo.output.replace(mo.Html(_html)), L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_passes():
    mo.md(r"""
    ### Oefening 4. Een waarde die erdoor komt

    In het plaatje onder oefening 3 heeft het antwoord van Qwen3 voor Sterrennacht, met de twee voorbeelden en seed 0, *prijs: warme kleren*. De aankondiging noemt helemaal geen prijs, en toch markeert `unsupported` de waarde niet.
    """)
    return


@app.cell(hide_code=True)
def choice_passes_cell():
    why_passes = L.choice(
        "why_passes",
        question="Waarom laat unsupported warme kleren door als prijs van Sterrennacht?",
        options={
            "made_up": "Qwen3 verzon de woorden, en een check op de tekst kan niet zien dat een waarde verzonnen is",
            "capitals": "De check vergelijkt in kleine letters, en dat verandert de waarde",
            "right": "De woorden staan in de aankondiging, en de check kijkt niet naar het veld",
            "long": "De aankondiging is te lang voor de check",
        },
        correct="right",
        why={
            "right": "De aankondiging eindigt met *Trek warme kleren aan.*, dus de waarde staat in de tekst. Een check of een waarde in de bron staat, vindt waarden die nergens vandaan komen of uit een voorbeeld, zoals *de muziekschool*, maar geen woorden uit de bron in het verkeerde veld. Daarvoor heb je een check per veld nodig, zoals een prijs die € of *gratis* moet bevatten, of een mens die kijkt.",
            "made_up": "Zoek de woorden op in de aankondiging van Sterrennacht: staan de woorden erin? Wat vraagt `unsupported` over een waarde?",
            "capitals": "Kleine letters maken alleen *Trek* en *trek* hetzelfde; de woorden zelf blijven gelijk. Staan de woorden van de waarde in de aankondiging?",
            "long": "De aankondiging heeft drie zinnen, en `in` kijkt door de hele tekst. Staan de woorden van de waarde in de aankondiging?",
        },
    )
    why_passes
    return (why_passes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Van tekst naar een tabel](/course/open/nlp-information-extraction?lang=nl) schreef je één patroon voor elke vorm van een datum, en hier vult een chat model vijf velden in met een prompt in de delen uit [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl); je `read_fields` leest het antwoord nog steeds met een reguliere expressie. Waarom een model een waarde zonder bron schrijft, legt [Waarom language models dingen verzinnen](/course/open/lm-hallucination?lang=nl) uit. De volgende les, *Soorten question answering*, stelt vragen over een tekst in plaats van velden in te vullen: ook daar maakt het uit waar het antwoord vandaan komt, uit de tekst of uit het model.

    **Een voorbeeld uit het dagelijks leven.** Een winkel die de facturen van leveranciers met een model leest, vergelijkt het totaal met de factuur, getal na getal, en betaalt pas daarna: een bedrag dat op geen enkele factuur staat, wordt nooit betaald.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_flag, result_prompt, result_read, why_passes):
    L.wrap_up(result_prompt, result_read, result_flag, why_passes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
