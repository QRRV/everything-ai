# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-source-attribution. Route: course/route/lessen/verantwoord.yaml. Checks en de run van Qwen3 van de course: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import chat
    from everything_ai.lesson import lesson

    L = lesson("rai-source-attribution", "nl")
    C = L.checks   # de bronnen en vragen van de course, en de antwoorden van Qwen3 met alle bronnen en met telkens één weggelaten


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Kan een model zichzelf uitleggen?](/course/open/rai-llm-self-explanations?lang=nl) en [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_faithful():
    quick_faithful = L.choice(
        "quick-faithful",
        question="Een model legt zijn antwoord uit, en elk feit in de uitleg klopt. Is de uitleg faithful?",
        options={
            "yes": "Ja: kloppende feiten tonen wat tot het antwoord leidde",
            "right": "Niet altijd: misschien leidde dat er niet toe",
            "size": "Alleen als het model groot genoeg is om het te weten",
        },
        correct="right",
        why={
            "yes": "Kloppende feiten maken een uitleg plausibel. Faithful betekent dat hij laat zien wat echt tot het antwoord leidde, en de uitleg over de planeet in [Kan een model zichzelf uitleggen?](/course/open/rai-llm-self-explanations?lang=nl) had kloppende feiten die naar (A) leidden, terwijl het antwoord (B) was.",
            "right": "Plausibel is hoe overtuigend een uitleg is; faithful is of hij laat zien wat tot het antwoord leidde. Deze les vraagt hetzelfde over een citatie: een bron kan zeggen wat het antwoord zegt, zonder dat het antwoord eruit kwam.",
            "size": "Grootte maakt een uitleg niet faithful; Lanham en anderen vonden bij de meeste van hun taken zelfs minder faithful redeneringen in grotere modellen. Zie [Kan een model zichzelf uitleggen?](/course/open/rai-llm-self-explanations?lang=nl).",
        },
    )
    quick_faithful
    return (quick_faithful,)


@app.cell(hide_code=True)
def quick_check_change():
    quick_change = L.choice(
        "quick-change",
        question="Hoe testte de course of de uitleg van Qwen3 zegt wat zijn antwoord veranderde?",
        options={
            "ask": "Door Qwen3 te vragen of zijn uitleg wel de waarheid sprak",
            "right": "Door een hint toe te voegen en te kijken of de uitleg die noemde",
            "read": "Door de uitleggen te lezen en te beoordelen hoe overtuigend ze klonken",
        },
        correct="right",
        why={
            "ask": "Zijn antwoord daarop zou nog een self-explanation zijn, tekst die hij schrijft zoals alle andere. De test veranderde iets dat er niet toe zou moeten doen, een hint, en keek wat de uitleg zei. Zie [Kan een model zichzelf uitleggen?](/course/open/rai-llm-self-explanations?lang=nl).",
            "right": "Verander iets, en kijk of het antwoord en de uitleg mee veranderen. Deze les doet hetzelfde met bronnen: laat er één weg, en kijk of het antwoord verandert.",
            "read": "Door te lezen kun je plausibel en faithful niet uit elkaar houden. De test veranderde iets dat er niet toe zou moeten doen, een hint, en keek wat de uitleg zei. Zie [Kan een model zichzelf uitleggen?](/course/open/rai-llm-self-explanations?lang=nl).",
        },
    )
    quick_change
    return (quick_change,)


@app.cell(hide_code=True)
def quick_check_cite():
    quick_cite = L.choice(
        "quick-cite",
        question="Een model antwoordt uit een tekst die je het geeft. Waarom vraag je het om een citatie, zoals [2]?",
        options={
            "right": "Zodat je het antwoord op die plek kunt nakijken",
            "true": "Zodat het antwoord klopt: een model kan niet citeren wat er niet staat",
            "fast": "Zodat het model sneller antwoordt",
        },
        correct="right",
        why={
            "right": "Een citatie noemt de plek in de gegeven tekst waar een deel van het antwoord vandaan komt, zodat je die kunt opzoeken. Of die plek zegt wat het antwoord zegt, en of het antwoord er echt uit kwam, test deze les.",
            "true": "Een model kan elk getal tussen haken schrijven, ook een verkeerd getal. Met een citatie kun je het antwoord nakijken; het antwoord wordt er niet goed door. Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
            "fast": "Een citatie is een paar tokens meer om te schrijven, niet minder. Je kunt er het antwoord mee nakijken in de tekst. Zie [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl).",
        },
    )
    quick_cite
    return (quick_cite,)




@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Antwoorden met bronnen zien eruit alsof je ze kunt nakijken. Een chatbot die [2] achter een zin schrijft, lijkt te zeggen: hier heb ik het vandaan. Maar een citatie kan wijzen naar een bron die niet zegt wat het antwoord zegt. En zelfs als een citatie klopt, hoeft het antwoord niet uit die bron te komen: het model kan geantwoord hebben uit wat het onthouden had, en daarna een bron geciteerd hebben die past. In deze les test je allebei, op Qwen3-0.6B: zegt de bron die hij citeert wat het antwoord zegt, en verandert het antwoord als je die bron weglaat?

    **Een voorbeeld uit het dagelijks leven.** Een student levert een essay in met een lijst bronnen. Een docent kan opzoeken of een bron zegt wat het essay beweert. Of de student het idee uit die bron haalde, of eerst het essay schreef en daarna een bron zocht die past, kan de lijst niet laten zien.
    """)
    return


@app.cell(hide_code=True)
def predict_wrote():
    guess_wrote = L.guess(
        "wrote",
        question="De course stelde Qwen3 12 vragen, elk met vijf genummerde bronnen, en vroeg hem het te zeggen als de bronnen geen antwoord geven. Daarna stelde de course elke vraag opnieuw, zonder de bron die Qwen3 had geciteerd. Wat schreef Qwen3 het vaakst?",
        options={
            "says": "Dat de bronnen geen antwoord op de vraag geven",
            "memory": "Toch het goede antwoord, uit wat hij weet",
            "other": "Een ander antwoord, met een citatie erin",
        },
        reveal="Een ander antwoord, met een citatie erin: bij 8 van de 12, zoals \"The ticket for the ferry costs [5] dollars.\" Bij 3 zei Qwen3 dat de bronnen geen antwoord geven, en alleen de openingstijd van de bibliotheek bleef goed, omdat een andere bron die ook noemt. In de oefeningen hieronder zoek je zelf uit welke bronnen elk antwoord nodig had.",
    )
    guess_wrote
    return (guess_wrote,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Source attribution.** In [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl) antwoordde een model uit een tekst die je het gaf, met een citatie, zodat je het antwoord kon nakijken. Laten zien op welke bronnen een antwoord gebaseerd is, heet **source attribution**. Een citatie roept twee vragen op die je makkelijk door elkaar haalt: zegt de bron wat het antwoord zegt, en kwam het antwoord uit die bron?

    **Zegt de bron het?** Nelson F. Liu, Tianyi Zhang en Percy Liang (2023) noemen een antwoord verifieerbaar als elke bewering over de wereld buiten het antwoord volledig ondersteund wordt door citaties in de tekst, en elke citatie de bewering ondersteunt waar die citatie bij hoort. Ze maten allebei in vier generatieve zoekmachines van begin 2023, Bing Chat, NeevaAI, perplexity.ai en YouChat, op 1450 zoekvragen: citation recall, het deel van de beweringen die het nakijken waard zijn dat volledig door hun citaties ondersteund wordt, en **citation precision**, het deel van de citaties dat de bewering ondersteunt waar ze bij horen. Over de vier zoekmachines samen werd maar 51,5% van de beweringen volledig door citaties ondersteund (recall), en ondersteunde maar 74,5% van de citaties hun bewering volledig (precision). Per zoekmachine liep de recall van 68,7% bij perplexity.ai tot 11,1% bij YouChat. En de zoekmachines waarvan mensen de antwoorden nuttiger vonden, hadden minder precieze citaties (een correlatie van −0,96): Liu en collega's vermoeden dat dat komt doordat antwoorden die hun bronnen dicht volgen precieze citaties hebben, maar minder nuttig lijken. Hannah Rashkin en collega's (2023) geven ondersteuning een vaste test: een lezer moet kunnen zeggen "Volgens P, s", met P de bron en s de bewering; staat maar een deel van de bewering in de bron, dan telt het niet. Tianyu Gao en collega's (2023) lieten een model dat beoordelen, op hun benchmark ALCE; het kwam voor citation recall goed overeen met het oordeel van mensen, en voor citation precision matig. En ondersteund is niet hetzelfde als waar: Rashkin en collega's laten buiten beschouwing of een bewering waar is, en Liu en collega's schrijven dat verifieerbaar iets anders is dan feitelijk juist.

    **Kwam het antwoord eruit?** Een bron kan een bewering ondersteunen zonder dat het model die bron gebruikte. Jonas Wallat, Maria Heuss, Maarten de Rijke en Avishek Anand (2024) noemen de eerste vraag *citation correctness*, en gebruiken faithfulness alleen voor de tweede: een citatie is faithful als het geciteerde document de bewering ondersteunt en er ook oorzakelijk invloed op had. Een model kan antwoorden uit zijn eigen geheugen en daarna een document zoeken dat op een oppervlakkige manier past, bijvoorbeeld doordat de woorden overeenkomen; zulke citaties die niet faithful zijn, kunnen volgens hen toch feitelijk kloppen, en zijn daardoor moeilijk op te merken. Ze testten dit op Command-R+, een model van Cohere, door een korte bewering die het model met een citatie had geschreven, zoals "Emperor penguin", aan andere documenten toe te voegen. Als het model dezelfde bewering opnieuw gaf, citeerde het zo'n aangepast document in 12% van de gevallen als het een willekeurig document was, en in 57% als het document relevant was maar eerder niet geciteerd was. Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev en Aleksander Madry (2024) vatten de test van de tweede vraag samen in één zin: als een deel van de context echt tot een bepaald antwoord leidde, dan moet het weghalen ervan dat antwoord flink veranderen. Met woorden uit een publicatie die ze aanhalen, noemen ze een bron die het antwoord veroorzaakte *contributive*, en een bron die het ondersteunt *corroborative*.

    **Eén weglaten.** De eenvoudigste test van de tweede vraag: laat één bron weg, vraag opnieuw, en kijk of het antwoord verandert. Doe dat voor elke bron, en dat is **leave-one-out**. Cohen-Wang en collega's vergelijken hun methode, ContextCite, daarmee: gemeten aan hoeveel minder waarschijnlijk het antwoord wordt zonder de belangrijkste bron, is leave-one-out het beste wat er is, maar het kost voor elke bron één extra run van het model. ContextCite laat in plaats daarvan telkens een willekeurig groepje bronnen weg en laat het model dan draaien, in een van hun voorbeelden 32 keer voor 98 bronnen, en fit een klein lineair model op de uitkomsten. Ze vonden ook dat de attention van het model geen veilige kortere weg is: bij het ene model kwam die dicht bij ContextCite, bij een ander deed die het slecht.

    **Wat weglaten niet kan laten zien.** Als twee bronnen hetzelfde zeggen, verandert het weglaten van één ervan weinig: de andere zegt het nog. Cohen-Wang en collega's noemen dat geval voor hun eigen lineaire model, en het geldt ook voor leave-one-out. Als het model het antwoord al wist, verandert het weglaten van de bron misschien ook niets, al ondersteunt de bron het antwoord: dan kan ook een bron die een bewering ondersteunt een lage score krijgen, schrijven ze. En het weghalen van een zin kan veranderen waar een woord als "het" in een andere zin naar verwijst.

    **De test van de course.** De course schreef vier korte teksten over een verzonnen eiland, Kestrel Island, elk in vijf genummerde bronnen, en 12 vragen erover, die elk door één of twee van de bronnen beantwoord worden. Twee bronnen van de tekst over de bibliotheek geven dezelfde openingstijd in andere woorden, en de tekst over de quizavond bevat twee feiten die een model vaak gezien heeft: de hoofdstad van Frankrijk en het kookpunt van water. Qwen3 kreeg elke vraag met de vijf bronnen erbij, en de opdracht om in één korte zin te antwoorden met het nummer van de bron tussen vierkante haken, of te zeggen dat de bronnen geen antwoord geven. Daarna stelde de course elke vraag nog eens voor elke bron, met die bron weggelaten en de andere nummers ongewijzigd: samen met de eerste ronde 72 prompts. De course vroeg zonder denken en greedy, dus steeds met het waarschijnlijkste volgende token, zodat elke prompt één antwoord heeft. De model card van Qwen3 waarschuwt tegen greedy decoding in de denkmodus, en die gebruikt deze run niet. De course rekent een antwoord goed als het de woorden van het goede antwoord bevat, zoals `7:40` of `6 euros`.

    **Probeer het.** Hieronder kun je Qwen3 een van de vragen van de course stellen, met de vijf bronnen erbij, in het Engels, zoals in de run van de course. Haal een bron weg, vraag opnieuw, en kijk of het antwoord en de citatie veranderen.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Qwen3-0.6B (1,5 GB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    chat.download_or_stop(get_button, {
        "ask": "*Het chat model Qwen3-0.6B staat nog niet op je computer. De knop downloadt het één keer van Hugging Face, `Qwen/Qwen3-0.6B` in een vaste versie, 1,5 GB, onder de Apache-2.0-licentie, met de tokenizer van Qwen3-0.6B-Base. De rest van de les werkt ook zonder: die leest de antwoorden die de course opnam.*",
        "busy": "**Bezig met downloaden:** {done} van {total} MB …",
        "no_net": "*Het downloaden lukte niet: Hugging Face was niet bereikbaar, of de verbinding viel weg. Controleer je verbinding, en klik opnieuw.*",
        "not_pinned": "*Het downloaden lukte niet: het bestand dat binnenkwam is niet het vastgezette bestand van de course, dus de course heeft het weggegooid. Klik opnieuw; gebeurt het weer, meld het dan op GitHub.*",
        "no_room": "*Het downloaden lukte niet: er is geen ruimte op je schijf voor 1,5 GB. Maak ruimte, en klik opnieuw.*",
        "failed": "*Het downloaden lukte niet ({why}). Klik opnieuw.*",
    })
    mo.md("*Qwen3-0.6B is gedownload.*")
    return


@app.cell(hide_code=True)
def ask_box_cell():
    ask_box = mo.ui.text_area(value=C.ASK_EXAMPLE, label="Je bronnen en vraag", rows=13, full_width=True)
    ask_button = mo.ui.run_button(label="Vraag het Qwen3")
    mo.vstack([ask_box, ask_button])
    return ask_box, ask_button


@app.cell(hide_code=True)
def demo_ask(ask_box, ask_button):
    if not chat.downloaded():
        _out = mo.md("*Download Qwen3 eerst, met de knop hierboven.*")
    elif not ask_button.value:
        _out = mo.md("*Druk op de knop om het Qwen3 te vragen, zonder denken en greedy, zoals in de run van de course. Haal een regel met een bron weg om die bron weg te laten.*")
    elif not ask_box.value.strip():
        _out = mo.md("*Schrijf eerst bronnen en een vraag in het vak.*")
    else:
        _words = {"title": "Wat Qwen3 schrijft", "loading": "Qwen3 laden …", "writing": "Qwen3 schrijft …",
                  "done": "{tokens} tokens in {seconds} seconden.",
                  "load": "Qwen3 laden stopt met een fout. Sluit andere programma's en druk opnieuw op de knop; gebeurt het weer, meld het dan op GitHub.",
                  "memory": "Het geheugen van je apparaat is vol. Sluit andere programma's, en druk opnieuw op de knop."}
        _out = mo.Html(chat.demo(C.ask_messages(ask_box.value), lambda _html: mo.output.replace(mo.Html(_html)), _words,
                                 L.lang, max_tokens=C.MAX_TOKENS, temperature=0,
                                 note=lambda _text: C.cited_note(ask_box.value, _text, L.lang)))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **source attribution** | laten zien op welke bronnen een antwoord gebaseerd is, bijvoorbeeld met citaties, zodat lezers het kunnen nakijken |
        | **citation precision** | het deel van de citaties in antwoorden dat de bewering ondersteunt waar de citatie bij hoort |
        | **leave-one-out** | telkens één bron weglaten, opnieuw vragen, en kijken of het antwoord verandert |
        """),
        L.watch_out("Een citatie die het antwoord ondersteunt, laat niet zien dat het antwoord eruit kwam: het model kan het antwoord geweten hebben, of achteraf een bron gevonden hebben die past. En een bron waarvan het weglaten niets verandert, kan toch gebruikt zijn, als een andere bron hetzelfde zegt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De citaties van een model testen tegen zijn bronnen:

    1. Geef het model genummerde bronnen en een vraag, en vraag om een kort antwoord met het nummer van de bron tussen vierkante haken, of om te zeggen dat de bronnen geen antwoord geven.
    2. Lees welke nummers het antwoord citeert. Dat is `cited`.
    3. Kijk voor elke citatie of die bron zegt wat het antwoord zegt. Het deel van de citaties waarvoor dat geldt, is de citation precision.
    4. Stel de vraag nog eens voor elke bron, telkens met die bron weggelaten en de andere nummers ongewijzigd.
    5. Noteer de bronnen zonder welke het antwoord het goede antwoord niet meer bevat. Dat is `needed`.
    6. Tel het deel van de citaties waarvan het antwoord de bron nodig had: de citaties die weglaten bevestigt. Dat is `confirmed_share`.
    7. Schrijf op wat geen van beide kan laten zien: een geciteerde bron die het antwoord niet nodig had, kan toch de bron zijn die het gebruikte, als een andere bron hetzelfde zegt; en een bron die het antwoord ondersteunt, hoeft niet de bron te zijn waar het antwoord uit kwam.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een verzonnen antwoord dat twee bronnen citeert, en de antwoorden op dezelfde vraag met telkens één van de vijf bronnen weggelaten. De cel gaat de vijf langs, en print voor elke bron of het antwoord die bron citeert, en of het antwoord zonder die bron nog 9:00 zegt.
    """)
    return


@app.cell
def example_left_out():
    # een verzonnen antwoord met alle vijf de bronnen, en de antwoorden met telkens één bron weggelaten
    _answer = "Het museum gaat om 9:00 open [3][5]."
    _left_out = [(1, "Het museum gaat om 9:00 open [3][5]."),
                 (2, "Het museum gaat om 9:00 open [3][5]."),
                 (3, "De bronnen zeggen niet wanneer het museum opengaat."),
                 (4, "Het museum gaat om 9:00 open [3][5]."),
                 (5, "Het museum gaat om 9:00 open [3].")]

    for _number, _text in _left_out:
        _cited = f"[{_number}]" in _answer
        _still = "9:00" in _text
        print("bron", _number, "| geciteerd:", _cited, "| zonder deze bron nog 9:00:", _still)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het antwoord citeert [3] en [5]. Zonder bron 3 zegt het geen 9:00 meer: het antwoord had die bron nodig, dus weglaten bevestigt die citatie. Zonder bron 5 zegt het nog steeds 9:00: weglaten bevestigt [5] niet. Misschien gebruikte het antwoord die bron niet, of zegt een andere bron hetzelfde. Van de 2 citaties bevestigt weglaten er 1: een deel van 0,5. `f"[{_number}]" in _answer` is True als het antwoord die bron citeert.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cited():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. cited: welke bronnen citeert een antwoord?

        `cited(text, n)` krijgt een antwoord `text` en het aantal bronnen `n`, en geeft, de kleinste eerst, de getallen van 1 tot en met `n` terug die de tekst tussen vierkante haken citeert, zoals `[2]`. `found` begint als een lege lijst, en de `for`-loop gaat elk getal `i` van 1 tot en met `n` langs. Vervang de `...` door twee regels: als de tekst het getal `i` tussen vierkante haken bevat, voeg `i` dan toe aan `found`. De functie geeft `found` terug. De check roept `cited(ANSWER, 5)` aan, met `ANSWER` het antwoord van Qwen3 op de eerste vraag, "The ferry to Kestrel Island leaves at 7:40 every morning [1].", en daarna andere teksten.
        """),
        L.task("Geef de getallen van 1 tot en met `n` terug die de tekst tussen vierkante haken citeert."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cited():
    box_cited = L.code_box("cited", lines=7)
    box_cited
    return (box_cited,)


@app.cell(hide_code=True)
def check_cited(box_cited):
    result_cited = L.check("cited", box_cited.value)
    result_cited
    return (result_cited,)


@app.cell(hide_code=True)
def hints_cited():
    L.hints(
        "cited",
        [
            "Stap 2 van het recept, zoals `_cited` in het uitgewerkte voorbeeld. Zoek het getal met zijn vierkante haken, zodat de 4 in 7:40 niet meetelt.",
            "Een `if` die vraagt of de tekst `i` tussen vierkante haken bevat, en daaronder `found.append(i)`.",
            "`if ___ in text:`",
        ],
        why="`if f\"[{i}]\" in text:` en daaronder `found.append(i)`. `\"[\" + str(i) + \"]\"` werkt ook. Op de 12 antwoorden van Qwen3 met alle vijf de bronnen vindt `cited` in elk antwoord één citatie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_needed():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. needed: zonder welke bronnen kon het antwoord niet?

        `needed(without, value)` krijgt een lijst `without` met de antwoorden van Qwen3 op één vraag met telkens één bron weggelaten, elk een tuple van twee: het nummer van de weggelaten bron en het antwoord. De functie krijgt ook `value`, de woorden van het goede antwoord, zoals `"7:40"`. `found` begint als een lege lijst, en de `for`-loop splitst elke tuple in `number` en `text`. Vervang de `...` door twee regels: als `value not in text`, wat True is als het antwoord de woorden niet meer bevat, voeg `number` dan toe aan `found`. De functie geeft `found` terug. De check roept `needed(WITHOUT, "7:40")` aan, met `WITHOUT` de vijf antwoorden van Qwen3 op de eerste vraag, elk met één bron weggelaten, daarna `needed(OPENING, "10:00")`, zijn antwoorden over de openingstijd van de bibliotheek, en korte lijsten.
        """),
        L.task("Geef de nummers terug van de bronnen zonder welke het antwoord `value` niet meer bevat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_needed():
    box_needed = L.code_box("needed", lines=7)
    box_needed
    return (box_needed,)


@app.cell(hide_code=True)
def check_needed(box_needed):
    result_needed = L.check("needed", box_needed.value)
    result_needed
    return (result_needed,)


@app.cell(hide_code=True)
def hints_needed():
    L.hints(
        "needed",
        [
            "Stap 5 van het recept, zoals `_still` in het uitgewerkte voorbeeld, maar andersom: je wilt de bronnen zonder welke het antwoord `value` niet meer bevat.",
            "Een `if` met `not in`, die vraagt of de woorden van het goede antwoord in het antwoord ontbreken, en daaronder voeg je het nummer van de bron toe aan `found`.",
            "`if value not in ___:`",
        ],
        why="`if value not in text:` en daaronder `found.append(number)`. `needed(WITHOUT, \"7:40\")` is `[1]`: alleen zonder bron 1 verliest het antwoord 7:40. Voor de openingstijd van de bibliotheek is `needed(OPENING, \"10:00\")` gelijk aan `[]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_confirmed():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. confirmed_share: hoeveel citaties bevestigt weglaten?

        `confirmed_share(rows)` krijgt een lijst `rows` met een dict per vraag: `"cited"`, de nummers die het antwoord met alle vijf de bronnen citeert, zoals je `cited` ze vindt, en `"needed"`, de nummers zonder welke het antwoord het goede antwoord niet meer bevat, zoals je `needed` ze vindt. Elke dict heeft ook `"question"`, het nummer van de vraag, geteld vanaf 0, en `"says"`, de bronnen die het goede antwoord noemen; die twee heb je in deze oefening niet nodig. `citations` en `confirmed` beginnen op 0, en de `for`-loop gaat elke `row` langs. Vervang de `...` door vier regels: een tweede `for`-loop over elk `number` in `row["cited"]`, waarin je 1 optelt bij `citations`, en ook 1 bij `confirmed` als `number in row["needed"]`. De functie geeft `confirmed / citations` terug. De check roept `confirmed_share(ROWS)` aan, met `ROWS` de 12 vragen uit de run van de course, en daarna drie korte lijsten.
        """),
        L.task("Geef van alle citaties het deel terug waarvan het antwoord de bron nodig had."),
    ])
    return


@app.cell(hide_code=True)
def exercise_confirmed():
    box_confirmed = L.code_box("confirmed", lines=10)
    box_confirmed
    return (box_confirmed,)


@app.cell(hide_code=True)
def check_confirmed(box_confirmed):
    result_confirmed = L.check("confirmed", box_confirmed.value)
    result_confirmed
    return (result_confirmed,)


@app.cell(hide_code=True)
def hints_confirmed():
    L.hints(
        "confirmed",
        [
            "Stap 6 van het recept. Een citatie is één nummer in `row[\"cited\"]`, dus je hebt een tweede loop binnen de eerste nodig.",
            "`for number in row[\"cited\"]:`, daarin `citations = citations + 1`, en een `if` die vraagt of het nummer ook bij de bronnen hoort die het antwoord nodig had.",
            "`if number in ___:`",
        ],
        why="In de loop over de rijen: `for number in row[\"cited\"]:`, daarin `citations = citations + 1` en `if number in row[\"needed\"]:` met daaronder `confirmed = confirmed + 1`. `confirmed_share(ROWS)` is 0,917: weglaten bevestigt 11 van de 12 citaties.",
    )
    return


@app.cell(hide_code=True)
def text_measured():
    mo.vstack([
        mo.md(r"""
        ### Wat de antwoorden van Qwen3 laten zien

        Met alle vijf de bronnen citeerde Qwen3 in elk van de 12 antwoorden één bron, en elke geciteerde bron bevat de woorden van het goede antwoord: gemeten zoals de course het doet: een citation precision van 1,0. Het weglaten van de geciteerde bron nam het goede antwoord weg bij 11 van de 12; het weglaten van een andere bron nam het nergens weg. Leave-one-out bevestigt dus 11 van de 12 citaties, een deel van 0,917. De citatie die het niet bevestigt, hoort bij de openingstijd van de bibliotheek: zonder [2] gaf Qwen3 dezelfde tijd en citeerde hij [4], en zonder [4] antwoordde hij net als met alle vijf. Zelfs bij de hoofdstad van Frankrijk was de bron nodig: zonder [2] schreef Qwen3 "The capital of France is [1]."
        """),
        mo.Html(C.measured_picture("nl")),
        mo.md(r"""
        Zonder de bron die hij citeerde, zei Qwen3 maar bij 3 van de 12 vragen dat de bronnen geen antwoord geven, en ook dan zette hij er nummers tussen haken bij. Bij 8 schreef hij een ander antwoord met een citatie erin: een fout antwoord, zoals "The trip to the island takes 7 hours and 30 minutes. [1]", een kapot antwoord, zoals "The ticket for the ferry costs [5] dollars.", of een antwoord over iets anders, zoals "The bakery is closed on Mondays. [5]" op de vraag wanneer de bakkerij sluit. Een lezer die alleen kijkt of er een citatie staat, zou het niet merken. De test is klein: één klein model, greedy gevraagd, 12 makkelijke vragen met korte bronnen die de course schreef, en goed of fout beslist door woorden te vergelijken. Hij laat zien zonder welke bronnen de antwoorden niet konden; hij laat niet zien hoe Qwen3 ze gebruikte, en waar twee bronnen hetzelfde zeggen, zoals bij de bibliotheek, kan hij niet zien welke van de twee gebruikt werd.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_duplicate():
    mo.md(r"""
    ### Oefening 4. Twee bronnen die hetzelfde zeggen

    Twee bronnen van de tekst over de bibliotheek zeggen wanneer de bibliotheek op zaterdag opengaat: [2] "The library opens at 10:00 on Saturdays." en [4] "On Saturdays, the doors of the library open at 10:00." Met alle vijf de bronnen antwoordde Qwen3 "The library opens at 10:00 on Saturdays. [2]". Zonder [2] antwoordde hij "The library opens on Saturdays at 10:00. [4]", en zonder [4] hetzelfde als met alle vijf.
    """)
    return


@app.cell(hide_code=True)
def choice_duplicate_cell():
    choice_duplicate = L.choice(
        "duplicate",
        question="Wat vertelt leave-one-out je over waar dit antwoord vandaan kwam?",
        options={
            "neither": "Dat het uit geen van beide bronnen kwam",
            "right": "Weinig: één van de twee was genoeg",
            "both": "Dat het beide bronnen samen nodig had",
            "memory": "Dat Qwen3 de openingstijd al kende",
        },
        correct="right",
        why={
            "neither": "Het weglaten van één bron veranderde niets, omdat de andere nog 10:00 zegt. Dat laat niet zien dat geen van beide gebruikt werd; het laat zien dat elk van de twee alleen genoeg was. Zie *Wat weglaten niet kan laten zien* in *Het idee*.",
            "right": "Zonder de ene geeft de andere nog 10:00, dus geen enkele weglating neemt het goede antwoord weg, en `needed` geeft een lege lijst. Het antwoord kan uit [2] komen, zoals het citeert, of uit [4]; door er telkens één weg te laten kun je dat niet zien. Beide tegelijk weglaten zou laten zien of het er minstens één nodig heeft.",
            "both": "Dan had het weglaten van één ervan het goede antwoord weggenomen. Dat gebeurde niet: elk van de twee alleen was genoeg. Zie *Wat weglaten niet kan laten zien* in *Het idee*.",
            "memory": "Het eiland en zijn bibliotheek zijn verzonnen voor de course, dus Qwen3 kan de tijd niet gekend hebben. Het weglaten van één bron veranderde niets, omdat de andere nog 10:00 zegt.",
        },
    )
    choice_duplicate
    return (choice_duplicate,)


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Oefening 5. Een kop
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Een nieuwssite schrijft over het onderzoek van Liu en collega's: \"De helft van wat AI-zoekmachines zeggen, is fout.\" Wat ondersteunt het onderzoek?",
        options={
            "exact": "De kop, zoals hij er staat",
            "right": "Minder: bijna de helft miste volledige steun door citaties",
            "more": "Meer: een kwart van de citaties was verzonnen",
            "useful": "Niets: het mat alleen hoe nuttig mensen de antwoorden vonden",
        },
        correct="right",
        why={
            "exact": "Liu en collega's maten of beweringen ondersteund werden door hun citaties, niet of ze waar waren: verifieerbaar is iets anders dan feitelijk juist. 51,5% van de beweringen werd volledig ondersteund; de andere kunnen toch waar zijn. En het ging om vier zoekmachines, begin 2023. Zie *Zegt de bron het?* in *Het idee*.",
            "right": "51,5% van de beweringen die het nakijken waard zijn, werd volledig ondersteund door hun citaties, en 74,5% van de citaties ondersteunde hun bewering, in vier zoekmachines begin 2023. Een bewering zonder volledige steun is daarom nog niet fout: verifieerbaar is iets anders dan feitelijk juist.",
            "more": "74,5% van de citaties ondersteunde hun bewering volledig. De andere ondersteunden die niet volledig, wat niet betekent dat ze verzonnen waren. Zie *Zegt de bron het?* in *Het idee*.",
            "useful": "Ze lieten mensen citation recall en citation precision beoordelen. Hoe nuttig mensen de antwoorden vonden, was een aparte vraag, en dat ging samen met minder precieze citaties. Zie *Zegt de bron het?* in *Het idee*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Kan een model zichzelf uitleggen?](/course/open/rai-llm-self-explanations?lang=nl) testte je of een uitleg zegt wat het antwoord veranderde; een citatie is nog een uitleg die een model schrijft, en een bron weglaten test die op dezelfde manier: verander iets, en kijk of het antwoord mee verandert. [Prompts die zichzelf controleren](/course/open/start-prompt-chains?lang=nl) vroeg om citaties, zodat je een antwoord kunt nakijken; deze les liet zien dat het antwoord niet uit een geciteerde bron hoeft te komen, ook als die citatie klopt, en dat een antwoord zonder zijn bron toch een citatie kan bevatten. Dit is de laatste les van de module over modellen uitleggen; de volgende module begint met *Wat is privacy?*
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_duplicate, choice_headline, result_cited, result_confirmed, result_needed):
    L.wrap_up(result_cited, result_needed, result_confirmed, choice_duplicate, choice_headline)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
