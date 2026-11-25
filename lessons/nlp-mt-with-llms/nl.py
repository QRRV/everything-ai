# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les nlp-mt-with-llms. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-mt-with-llms", "nl")
    C = L.checks   # de twaalf teksten van de course, de run van Qwen3-0.6B op drie manieren, de plaatjes en de checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl), en één over [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="Je zet de zinnen van een testset in een andere volgorde, en bij elke vertaling hoort nog steeds dezelfde referentie. Wat gebeurt er met de BLEU van de hele testset?",
        options={
            "lower": "BLEU gaat omlaag, want de n-grams volgen elkaar niet meer op",
            "right": "BLEU blijft gelijk",
            "penalty": "BLEU gaat omhoog, want de brevity penalty verandert",
            "far": "Dat hangt ervan af hoe ver elke zin van de oude plek is verschoven",
        },
        correct="right",
        why={
            "right": "BLEU van een testset telt de afgetopte tellingen van n-grams en de lengtes van alle zinnen bij elkaar op, en elk n-gram wordt geteld binnen één zin en de bijbehorende referentie. De volgorde van de zinnen speelt geen rol, dus BLEU kan niet zien of een naam in elke zin van een tekst op dezelfde manier geschreven is. Deze les telt dat op een andere manier.",
            "lower": "Elk n-gram wordt geteld binnen één zin, tegen de referentie van de zin zelf. Loopt een n-gram ooit door van de ene zin in de volgende? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
            "penalty": "De brevity penalty vergelijkt de lengte van alle vertalingen met de lengte van alle referenties. Verandert dat als de zinnen van plek wisselen? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
            "far": "Elke zin wordt nog steeds gescoord tegen de eigen referentie. Wat in BLEU kijkt naar de plek van een zin? Zie [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def quick_check_messages():
    quick_messages = L.choice(
        "quick-messages",
        question="In een chat stuur je je derde bericht. Wat krijgt het model als prompt?",
        options={
            "last": "Alleen je derde bericht",
            "yours": "Je drie berichten, zonder de eigen antwoorden van het model",
            "right": "Alle berichten tot nu toe, elk met een rol",
            "summary": "Een samenvatting van de chat tot nu toe, die het model na elk antwoord zelf schreef",
        },
        correct="right",
        why={
            "right": "In een chat is de prompt elk bericht tot nu toe: jouw berichten met de rol `user`, de eigen antwoorden van het model met de rol `assistant`. Deze les gebruikt dat om een tekst zin voor zin te vertalen in één chat, zodat het model de eerdere zinnen ziet, en ook de vertalingen die het model zelf schreef.",
            "last": "Dan zou het model niet weten wat je eerder vroeg. Wat is een prompt in een chat? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "yours": "De antwoorden van het model zijn ook berichten. Welke rol hebben de antwoorden, en staan de antwoorden in de prompt? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
            "summary": "Het model krijgt de berichten zelf, elk met een rol. Welke rollen zijn er? Zie [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl).",
        },
    )
    quick_messages
    return (quick_messages,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Machinevertaling werkte jarenlang met één zin tegelijk, en zo vertaalt Qwen3 ook in de les *Vertalers en machines samen*. Maar een zin hoort vaak bij een tekst: een naam komt terug, net als een term, en de tekst rond een zin zegt welke betekenis een woord daar heeft. In 2018 lieten Samuel Läubli, Rico Sennrich en Martin Volk professionele vertalers Chinees-Engelse vertalingen van nieuwsberichten vergelijken: vertalingen door een systeem dat even goed zou vertalen als mensen, en vertalingen door mensen. Bij losse zinnen, beoordeeld op hoe goed de betekenis bewaard blijft, vonden de beoordelaars geen duidelijk verschil. Bij de hele tekst wel: bij hele documenten kozen de beoordelaars duidelijker voor de vertaling door mensen dan bij losse zinnen. In een nieuwsbericht van zes zinnen over een app hield de vertaling door mensen één naam voor de app aan, van begin tot eind. In de vertaling door het systeem kreeg de app in hetzelfde bericht drie namen, schrijven Läubli en collega's: *Twitter Move Car*, *WeChat mobile* en *WeChat Move*.

    Een chat model krijgt de invoer als prompt, en een prompt kan een hele tekst bevatten. In deze les vertaalt Qwen3-0.6B, het kleine chat model van de course, twaalf korte Nederlandse teksten die de course schreef, op drie manieren, en telt de les wat er gebeurt met de namen en de termen die terugkomen.

    **Een voorbeeld uit het dagelijks leven.** In een handleiding heet een onderdeel op bladzijde 2 de *schakelaar* en op bladzijde 30 de *knop*: zijn het twee onderdelen? Vertalers van handleidingen houden een lijst met termen bij, zodat één woord van het origineel één woord krijgt in de vertaling, van begin tot eind.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_right = L.guess(
        "guess",
        question="Elke tekst van de course heeft een term met meer dan één betekenis, zoals blad (van een boom, een vel papier of een tijdschrift), die drie of vier keer terugkomt: 42 plekken in totaal. Met de hele tekst in één prompt gebruikt Qwen3 in bijna elke tekst één woord voor de term. Op hoeveel van de 42 plekken gebruikt Qwen3 een woord voor wat de term in de tekst aanduidt?",
        options={
            "all": "Alle 42",
            "most": "Meer dan driekwart",
            "between": "Tussen de helft en driekwart",
            "right": "Minder dan de helft",
        },
        reveal="15 van de 42: minder dan de helft. Met elke zin los zijn het er 12, en in één chat 14. Met de hele tekst houdt Qwen3 veel vaker één woord aan voor een term, maar in zes teksten duidt dat ene woord niet aan wat de term daar aanduidt: *bank* voor de zitbanken van Sluis, waar het Engels *bench* zegt, *forecast* voor de nachtvorst van Best, *wool* voor de stof van Buren, *newspaper* voor het tijdschrift van Hoorn, *rope* voor de veerpont van Velden, en *monster* voor het watermonster van Kampen.",
    )
    guess_right
    return (guess_right,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Vertalen op documentniveau** (*document-level translation*) vertaalt elke zin met de rest van de tekst als **context**; **vertalen op zinsniveau** (*sentence-level translation*) vertaalt elke zin op zichzelf. Een chat model kan de context op twee manieren krijgen. *In één chat*: elke zin is een nieuw bericht, en de chat bewaart alle eerdere berichten, dus het model ziet de eerdere Nederlandse zinnen en de vertalingen die het model zelf schreef. *De hele tekst*: alle zinnen in één prompt, elke zin op een genummerde regel, en het model schrijft bij elke zin een genummerde regel terug. In 2023 probeerden Longyue Wang en collega's drie van zulke prompts met ChatGPT, elk met één chat voor de hele tekst: één zin per bericht, of zoveel mogelijk zinnen per bericht, met of zonder een teken tussen de zinnen. De verschillen waren klein, en de prompt met veel zinnen per bericht en zonder tekens deed het op de meeste maten het best, ook op de consistentie van termen.

    Context helpt waar een zin op zichzelf twee dingen kan betekenen. In *Het blad verschijnt vier keer per jaar* kan *blad* een blad van een boom zijn, een vel papier of een tijdschrift; de eerste zin van de tekst zegt *tijdschrift*. En veel Nederlandse plaatsen hebben een naam die ook een woord is: *Bergen* is in het Engels *mountains*, *Echt* is *real*. Elk van de twaalf teksten van de course heeft zo'n plaats, en een term met meer dan één betekenis die drie of vier keer terugkomt.

    Een term die herhaald wordt, hoort volgens Wang en collega's in het hele document dezelfde vertaling te houden. De auteurs meten dat als de **consistentie** van een term: van alle paren vertalingen van de term in één tekst, het deel dat gelijk is. De consistentie is 1 als de vertaling elke keer hetzelfde woord is. Consistentie zegt niet of dat woord goed is: een term die elke keer met hetzelfde foute woord vertaald wordt, is ook consistent.

    Het plaatje laat één tekst zien op de drie manieren.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Hoorn blijft op alle drie de manieren een naam. Met elke zin los heeft Qwen3 verder geen houvast: *Het blad verschijnt vier keer per jaar* wordt *The leaf appears four times a year*, en in zin 4 schrijft Qwen3 *the blade*. In zin 3 en 5 laat Qwen3 de term weg. In één chat en met de hele tekst schrijft Qwen3 alle vier de keren *newspaper*, een krant: geen blad van een boom meer, maar iets dat regelmatig verschijnt, en elke keer hetzelfde woord. Toch gaat de tekst over een tijdschrift dat vier keer per jaar verschijnt, en dat is geen krant.

    Of een woord goed is, besliste de course met de hand voor elk woord dat Qwen3 gebruikte: een woord voor wat de term in de tekst aanduidt, is goed, ook een algemener woord, zoals *building* (gebouw) voor het kasteel van Bergen. Een woord voor een andere betekenis van de term, zoals *leaf*, of voor iets anders, zoals *newspaper* voor een tijdschrift, is niet goed.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | **vertalen op documentniveau** (*document-level translation*) | elke zin vertalen met de rest van de tekst als context |
    | **vertalen op zinsniveau** (*sentence-level translation*) | elke zin op zichzelf vertalen, zonder de zinnen eromheen |
    | **context** | hier: de andere zinnen van de tekst, die een model ziet terwijl het één zin vertaalt |
    | **consistentie** van een term | van alle paren vertalingen van een term in één tekst, het deel dat gelijk is; 1 als het elke keer hetzelfde woord is |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Zet elke zin van de tekst op een eigen regel, met vooraan het nummer: `1.`, `2.`, enzovoort.
    2. Geef de hele tekst in één prompt, en vraag bij elke zin een genummerde regel terug. Of vertaal de zinnen in één chat, één bericht per zin, en bewaar elk eerder bericht en antwoord.
    3. Lees de genummerde regels van het antwoord terug als zinnen, en controleer of elk nummer terugkwam.
    4. Noteer bij elke term het woord dat de vertaling in elke zin gebruikt.
    5. Bereken de consistentie van elke term, en controleer of de woorden goed zijn: consistent is niet hetzelfde als goed.
    6. Scoor de vertalingen ook met BLEU tegen referenties, en onthoud dat BLEU naar één zin tegelijk kijkt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    In de tekst over Echt betekent *bord* een verkeersbord: de eerste zin heeft *verkeersborden*. Met elke zin los schrijft Qwen3 drie keer *board* en één keer *sign*. Hier staan de paren van de vier woorden, met de hand uitgeschreven, en het deel van de paren dat gelijk is:
    """)
    return


@app.cell
def example_cell():
    alone = ["board", "board", "board", "sign"]
    pairs = [(alone[0], alone[1]), (alone[0], alone[2]), (alone[0], alone[3]),
             (alone[1], alone[2]), (alone[1], alone[3]),
             (alone[2], alone[3])]
    same = 0
    for a, b in pairs:
        if a == b:
            same = same + 1
    print(same, "van de", len(pairs), "paren zijn gelijk:", same / len(pairs))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vier woorden geven zes paren: het eerste woord met elk van de drie erna, het tweede met de twee erna, en het derde met het laatste. Drie van de zes zijn gelijk, dus de consistentie is 0,5. Met de hele tekst schrijft Qwen3 alle vier de keren *sign*, en dat geeft 1,0: elke keer hetzelfde woord, en hier ook het goede. De course schrijft elk woord in het enkelvoud, dus *boards* telt als *board*. In oefening 3 maken twee loops de paren.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_numbered():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. numbered: de regels van de hele tekst

        Schrijf `numbered(sentences)`, waarbij `sentences` de lijst met de zinnen van een tekst is. Geef één tekst terug met elke zin op een eigen regel: het nummer van de zin, geteld vanaf 1, een punt, een spatie en de zin. `"\n".join(lines)` voegt de teksten van de lijst `lines` samen, met telkens een regeleinde ertussen. Vervang de `...`. De course zet de eigen instructie voor de hele tekst, `C.DOCUMENT_INSTRUCTION`, en een lege regel boven je regels: *Translate this Dutch text into English. Translate each numbered line into one numbered line. Reply with only the English translation.*

        De check roept `numbered(["Het regent.", "De weg is glad."])` aan, wat `"1. Het regent.\n2. De weg is glad."` geeft. Daarna eigen gevallen: een lege lijst, één zin, elf zinnen, en de twaalf teksten van de course (in de les: `C.DOCUMENTS[i]["nl"]`).
        """),
        L.task("Nummer de regels van een tekst voor de prompt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_numbered():
    box_numbered = L.code_box("numbered", lines=6)
    box_numbered
    return (box_numbered,)


@app.cell(hide_code=True)
def check_numbered(box_numbered):
    result_numbered = L.check("numbered", box_numbered.value)
    result_numbered
    return (result_numbered,)


@app.cell(hide_code=True)
def hints_numbered():
    L.hints(
        "numbered",
        [
            "In de loop loopt `k` langs de plaatsen van de zinnen, 0, 1, 2 enzovoort, en `lines.append` voegt één regel toe aan `lines`: het nummer van de zin, een punt, een spatie en de zin `sentences[k]`. Een f-string zet de waarde van wat tussen `{` en `}` staat in de tekst.",
            "De eerste zin staat op plaats 0, maar de regel moet met 1 beginnen. Wat schrijf je tussen de accolades om het nummer van de zin op plaats `k` te krijgen?",
            '`        lines.append(f"{___}. {sentences[k]}")`',
        ],
        why='In de loop: `lines.append(f"{k + 1}. {sentences[k]}")`.',
    )
    return


@app.cell(hide_code=True)
def text_after_numbered():
    mo.md(r"""
    In alle twaalf teksten gaf Qwen3 bij elke zin één genummerde regel terug, zodat de course elke regel naast de Nederlandse zin kon zetten. Wie zich aan de nummers houdt, doet nog niet wat gevraagd is: bij de tekst over Haren gaf Qwen3 de Nederlandse zinnen zelf terug, onder de nummers. Met elke zin los deed Qwen3 dat slechts bij de eerste zin van de tekst.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chat():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. chat_messages: één chat voor een hele tekst

        Schrijf `chat_messages(instruction, sentences, answers)`. `instruction` is de instructie voor één zin, `sentences` zijn de zinnen van de tekst, en `answers` zijn de vertalingen die Qwen3 tot nu toe van de eerste zinnen maakte. Geef de berichten van de chat voor de volgende zin terug: bij elke zin tot nu toe een bericht met de rol `"user"` met de instructie, een lege regel en de zin, en dan het antwoord van Qwen3 met de rol `"assistant"`; en aan het eind de volgende zin met de rol `"user"`, in dezelfde vorm. Een bericht is een dictionary zoals `{"role": "user", "content": "Hoi"}`, zoals in *Goed vragen*. Vervang de twee `...`.

        De check roept `chat_messages("Translate.", ["Het regent.", "De weg is glad."], ["It is raining."])` aan, wat `[{"role": "user", "content": "Translate.\n\nHet regent."}, {"role": "assistant", "content": "It is raining."}, {"role": "user", "content": "Translate.\n\nDe weg is glad."}]` geeft. Daarna eigen gevallen: het begin van een tekst zonder antwoorden, een tekst van drie zinnen, en de chats van de course bij twee teksten (in de les: `C.INSTRUCTION`, `C.DOCUMENTS[i]["nl"]` en `C.answers(i, "chat")[:k]`).
        """),
        L.task("Bouw de chat waarin Qwen3 de volgende zin vertaalt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chat():
    box_chat = L.code_box("chat_messages", lines=8)
    box_chat
    return (box_chat,)


@app.cell(hide_code=True)
def check_chat(box_chat):
    result_chat = L.check("chat_messages", box_chat.value)
    result_chat
    return (result_chat,)


@app.cell(hide_code=True)
def hints_chat():
    L.hints(
        "chat_messages",
        [
            'Elk bericht is een dictionary met een `"role"` en een `"content"`. In de loop voegt `messages.append` eerst zin `k` toe met de rol `"user"`, wat al in de code staat; daarna komt de vertaling van de zin met de rol `"assistant"`: `answers` bevat de vertalingen van Qwen3 tot nu toe. Na de loop vraagt nog één bericht met de rol `"user"` om de volgende zin, in dezelfde vorm als in de loop: `instruction`, een lege regel en de zin.',
            "Qwen3 heeft zoveel zinnen vertaald als `answers` bevat, en de plaatsen in `sentences` beginnen bij 0. Welk antwoord hoort bij zin `k`, en op welke plaats staat de volgende zin?",
            '`        messages.append({"role": "assistant", "content": ___})`',
        ],
        why='In de loop: `messages.append({"role": "assistant", "content": answers[k]})`. Daarna: `messages.append({"role": "user", "content": instruction + "\\n\\n" + sentences[len(answers)]})`.',
    )
    return


@app.cell(hide_code=True)
def text_after_chat():
    mo.md(r"""
    Met elke zin los vertaalt Qwen3 *Duiven is de eerste plaats in de buurt met dit vak* als *First place in the neighborhood with this subject.*: de plaats is weg. In de chat begint het eerste antwoord van Qwen3 met *A school in Duiven*, en Qwen3 houdt de naam. Over de twaalf teksten blijft de plaats een naam op 22 van de 31 plekken met elke zin los, en op 29 in één chat. Met de hele tekst zijn het er 24: Qwen3 geeft het Nederlands van de tekst over Haren terug, schrijft twee keer *the sluice* voor Sluis, spelt Bergen één keer als *Bregen*, en laat Bergen één keer weg en Echt ook één keer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_consistency():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. consistency: elke keer hetzelfde woord?

        Schrijf `consistency(translations)`, waarbij `translations` de lijst is met de woorden die een vertaling voor één term in één tekst gebruikte, minstens twee. Geef het deel terug van de paren van twee verschillende plaatsen in de lijst waarvan de woorden gelijk zijn. Twee loops maken de paren: `i` loopt langs elke plaats, en `j` langs de plaatsen na `i`. Vervang de twee `...`.

        De check roept `consistency(["leaf", "paper", "paper"])` aan, wat `0.3333` geeft, en `consistency(["menu", "menu", "menu", "menu"])`, wat `1.0` geeft. Daarna eigen gevallen: twee vertalingen, gelijk of niet, langere lijsten, en de woorden die Qwen3 gebruikte in de run van de course (in de les: `C.words_found(i, way)`, zonder de plekken waar Qwen3 de term wegliet).
        """),
        L.task("Bereken de consistentie van een term in één tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_consistency():
    box_consistency = L.code_box("consistency", lines=9)
    box_consistency
    return (box_consistency,)


@app.cell(hide_code=True)
def check_consistency(box_consistency):
    result_consistency = L.check("consistency", box_consistency.value)
    result_consistency
    return (result_consistency,)


@app.cell(hide_code=True)
def hints_consistency():
    L.hints(
        "consistency",
        [
            "De buitenste loop neemt elke plaats `i` van `translations` om de beurt. De binnenste loop over `j` moet alleen de plaatsen na `i` nemen, tot het eind van de lijst, zodat elk paar één keer komt en geen woord een paar met zichzelf vormt: `range(a, b)` geeft de hele getallen van `a` tot `b`, zonder `b`, en `len(translations)` is het aantal woorden. De `if` vraagt of de twee woorden van het paar gelijk zijn.",
            "Bij drie woorden zijn de paren (0, 1), (0, 2) en (1, 2). Op welke plaats begint `j` als `i` 0 is, en als `i` 1 is?",
            "`        for j in range(___, len(translations)):`",
        ],
        why="`for j in range(i + 1, len(translations)):` en `if translations[i] == translations[j]:`.",
    )
    return


@app.cell(hide_code=True)
def text_after_consistency():
    mo.md(r"""
    Gemiddeld over de teksten is de consistentie 0,54 met elke zin los, 0,92 in één chat en 0,94 met de hele tekst, dat laatste over elf teksten, omdat Qwen3 de tekst over Haren in het Nederlands teruggaf. Maar in zes teksten heeft de hele tekst elke keer één woord dat niet aanduidt wat de term aanduidt: *bank* voor de zitbanken van Sluis, *forecast* voor de nachtvorst van Best, *wool* voor de stof van Buren, *newspaper* voor het tijdschrift van Hoorn, *rope* voor de veerpont van Velden en *monster* voor het watermonster van Kampen. De tabel telt alles.
    """)
    return


@app.cell(hide_code=True)
def table_cell():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.md(r"""
    Context helpt Qwen3 met namen en met een term hetzelfde houden, en BLEU stijgt van 27,4 met elke zin los naar 35,9 in één chat en 35,1 met de hele tekst. Met de betekenis helpt context veel minder: van de 42 plekken van een term krijgen er 12 een goed woord met elke zin los, 14 in één chat en 15 met de hele tekst. Een model dat niet weet dat het Nederlandse *monster* ook een kleine hoeveelheid om te onderzoeken kan zijn, in het Engels een *sample*, leert dat niet uit de zinnen eromheen. Wang en collega's maten ChatGPT, een veel groter model; deze les laat niet zien hoeveel context helpt bij een model van dat formaat.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: elke vertaling uit de run van de course

    Kies een tekst en een manier, en bekijk elke zin: het Nederlands, wat Qwen3 schreef, en de vertaling van de course. Zoek een plaats die een Engels woord werd, en een term die van zin tot zin verandert.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_text = mo.ui.dropdown(options=C.text_options(L.lang), value="Hoorn: blad", label="Tekst")
    try_way = mo.ui.radio(options={C.WAY_NAMES[L.lang][_way]: _way for _way in C.WAYS},
                          value=C.WAY_NAMES[L.lang]["alone"], label="Manier")
    mo.hstack([try_text, try_way], justify="start", gap=1.5, wrap=True)
    return try_text, try_way


@app.cell(hide_code=True)
def try_look(try_text, try_way):
    mo.Html(C.try_picture(L.lang, try_text.value, try_way.value))
    return


@app.cell(hide_code=True)
def text_live():
    mo.md(r"""
    ### Vertaal je eigen tekst met Qwen3

    Schrijf een korte Nederlandse tekst, één zin per regel, en laat Qwen3-0.6B de tekst op de drie manieren vertalen, met elke keer dezelfde uitkomst, zoals in de run van de course. Zet er een plaats in met een naam die ook een woord is, of een term met meer dan één betekenis, en kijk welke manier de plaats of de term goed houdt. Elke regel wordt twee keer vertaald, en daarna de hele tekst één keer, dus een langere tekst duurt langer. Qwen3 draait op je eigen computer, dus niets wat je typt verlaat je computer.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het model één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base. De rest van de les werkt zonder Qwen3: de les leest de vertalingen die de course opnam.*",
        "busy": "**Downloaden:** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course gooide het bestand weg. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def text_box_cell():
    text_box = mo.ui.text_area(value=C.YOUR_TEXT, label="Je Nederlandse tekst, één zin per regel", rows=4, full_width=True)
    translate_button = mo.ui.run_button(label="Vertaal met Qwen3")
    mo.vstack([text_box, translate_button])
    return text_box, translate_button


@app.cell(hide_code=True)
def demo_answer(text_box, translate_button):
    if not C.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not translate_button.value:
        _out = mo.md("*Druk op de knop om Qwen3 je tekst te laten vertalen.*")
    elif not C.your_lines(text_box.value):
        _out = mo.md("*Schrijf eerst een tekst in het vak.*")
    else:
        _out = mo.Html(C.translate_live(text_box.value, lambda _html: mo.output.replace(mo.Html(_html)), L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_same_wrong():
    mo.md(r"""
    ### Oefening 4. Consistent, maar goed?

    In de tekst over Kampen is een *monster* een kleine hoeveelheid water om te onderzoeken: de eerste zin heeft *watermonster*. Op alle drie de manieren is elk woord dat Qwen3 voor de term gebruikt *monster*, en je `consistency` geeft voor elke manier 1,0.
    """)
    return


@app.cell(hide_code=True)
def choice_same_wrong_cell():
    same_wrong = L.choice(
        "same_wrong",
        question="Wat zegt een consistentie van 1,0 over de vertalingen van monster in de tekst over Kampen?",
        options={
            "good": "Dat de term goed vertaald is, omdat de drie manieren het eens zijn",
            "right": "Dat Qwen3 elke keer één woord gebruikte, en niets over of dat woord goed is",
            "bleu": "Dat BLEU hoog is voor deze zinnen",
            "broken": "Dat consistentie niet werkt voor een woord dat in het Nederlands en het Engels hetzelfde geschreven wordt, dus de tekst moet eruit",
        },
        correct="right",
        why={
            "right": "Consistentie vergelijkt alleen de vertalingen met elkaar. Vier keer *monster* is helemaal consistent, en vier keer fout: een watermonster is in het Engels een *sample*. Elena Voita, Rico Sennrich en Ivan Titov zeggen iets vergelijkbaars, als uitleg waarom BLEU consistente en inconsistente vertalingen nauwelijks uit elkaar houdt: als een naam op meer manieren vertaald kan worden, maakt een model dat de vertaling gelijk houdt, de naam even makkelijk overal gelijk aan de referentie als overal anders dan de referentie. Controleer dus allebei: of een term hetzelfde blijft, en of de term goed is.",
            "good": "De drie manieren schrijven hetzelfde woord, maar betekent dat woord wat *monster* in deze tekst betekent? Wat haalt het waterbedrijf in de eerste zin uit de rivier?",
            "bleu": "BLEU vergelijkt met de vertaling van de course, en daar staat *sample*. Past *monster* daarbij?",
            "broken": "Consistentie vergelijkt de woorden van Qwen3 met elkaar, in welke taal de woorden ook lijken te staan. Wat zegt consistentie over of *monster* goed is?",
        },
    )
    same_wrong
    return (same_wrong,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Een vertaling scoren: BLEU en chrF](/course/open/nlp-bleu?lang=nl) scoorde je een vertaling zin voor zin tegen een referentie; hier vertaalt een chat model met de zinnen rond elke zin, in een chat met de rollen van berichten uit [Goed vragen: rollen, voorbeelden en stappen](/course/open/start-prompt-basics?lang=nl), en consistentie telt waar BLEU niet naar kijkt. De les *Vertalers en machines samen* laat Qwen3 zin voor zin vertalen, en verbetert de vertalingen met post-editing. De volgende les, *Van data naar tekst*, maakt tekst van data, zoals in een weerbericht.

    **Een voorbeeld uit het dagelijks leven.** Ondertitels worden regel voor regel vertaald, maar een goede ondertitelaar kijkt eerst de hele film: wie tegen wie praat, bepaalt of het Engelse *you* in het Nederlands *je* of *u* wordt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chat, result_consistency, result_numbered, same_wrong):
    L.wrap_up(result_numbered, result_chat, result_consistency, same_wrong)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
