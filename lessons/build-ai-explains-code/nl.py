# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les build-ai-explains-code. Route: course/route/lessen/bouwen.yaml. Checks, de 24 antwoorden en hun plaatje: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-ai-explains-code", "nl")
    C = L.checks   # wat Qwen3 antwoordde toen het twee kleine functies uitlegde, uitrekende en hernoemde, en het plaatje


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl), en één over [Functies](/course/open/py-functions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chat():
    quick_chat = L.choice(
        "quick-chat",
        question="Een collega stuurt je een functie die je niet begrijpt, en je wilt uitleg, met de code ernaast. Welke soort hulp past?",
        options={
            "completion": "Code completion",
            "chat": "Een chat over je code",
            "agent": "Een coding agent",
        },
        correct="chat",
        why={
            "chat": "Je wilt een antwoord dat je daarna zelf gebruikt: een chat, met de code erin geplakt.",
            "completion": "Code completion stelt de volgende regel voor waar je typt; het legt niets uit. Zie [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).",
            "agent": "Een agent verandert bestanden en draait ze; voor uitleg is dat meer dan je nodig hebt. Zie [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl).",
        },
    )
    quick_chat
    return (quick_chat,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="Een functie `double(x)` heeft één regel: `return x * 2`. Wat geeft `double(double(3))`?",
        options={
            "twelve": "12",
            "six": "6",
            "nine": "9",
        },
        correct="twelve",
        why={
            "twelve": "`double(3)` geeft 6, en `double(6)` geeft 12: wat een functie teruggeeft, is een waarde die je kunt doorgeven.",
            "six": "Dat is `double(3)`. Wat gebeurt er met die 6? Zie [Functies](/course/open/py-functions?lang=nl).",
            "nine": "`double` vermenigvuldigt met 2, twee keer. Zie [Functies](/course/open/py-functions?lang=nl).",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel code waarmee je gaat werken, heb je niet zelf geschreven: hij komt van een collega, van internet, of van een AI. Een assistent kan zulke code in gewone woorden uitleggen, samenvatten wat een bestand doet, en klusjes van je overnemen, zoals betere namen. Dat scheelt tijd. Maar een uitleg vertelt je wat de code *lijkt* te moeten doen; de code doet wat zijn regels zeggen.

    **Een voorbeeld uit het dagelijks leven.** Op een pot in de kast staat *aardbeienjam*. Meestal klopt dat. Of het deze keer klopt, merk je met een lepel, niet door het etiket nog eens te lezen. Dat is alleen een vergelijking; hieronder zie je wat Qwen3 zei over deze functie, die een leerling schreef om de hoogste en de laagste score weg te laten:

    ```python
    def average_without_extremes(scores):
        scores = sorted(scores)
        middle = scores[1:-1]
        return sum(middle) / len(scores)
    ```

    `sorted(scores)` geeft de scores op volgorde, van klein naar groot, en `sum(middle)` telt een lijst op. `scores[1:-1]` is een slice: vanaf index 1 tot net voor het laatste item, dus zonder de eerste en de laatste.
    """)
    return


@app.cell(hide_code=True)
def predict_explanations():
    guess_explanations = L.guess(
        "explanations",
        question="De course vroeg Qwen3 acht keer om in twee zinnen uit te leggen wat deze functie teruggeeft. Hoeveel van de acht uitleggen zagen wat er mis mee is?",
        options={
            "most": "De meeste",
            "none": "Geen enkele",
            "few": "Eén of twee",
        },
        reveal="**Geen enkele.** Alle acht zeggen dat hij het gemiddelde van de scores teruggeeft zonder de eerste en de laatste. Dat belooft de naam, en dat bedoelde de leerling. De laatste regel deelt door `len(scores)`, het aantal van *alle* scores: bij vijf scores deelt hij door 5, terwijl er maar drie worden opgeteld.",
    )
    guess_explanations
    return (guess_explanations,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent |
    |---|---|
    | `sorted(x)` | een nieuwe lijst met de items van `x` op volgorde, van klein naar groot |
    | `sum(x)` | de items van de lijst `x` bij elkaar opgeteld |
    | **docstring** | een tekst tussen drie aanhalingstekens, zoals `'''Return the area.'''`, direct onder de `def`-regel, die zegt wat de functie doet |

    ### Drie dingen om te vragen, en hoe je ze nakijkt

    | Je vraagt om | Bijvoorbeeld | Hoe je het nakijkt |
    |---|---|---|
    | uitleg | *Wat geeft deze functie terug?* | reken één klein geval met de hand uit, en voer het dan uit: geeft de code wat de uitleg zegt? |
    | een samenvatting | *Wat doet dit bestand?* | open de delen die de samenvatting noemt, en kijk of ze dat doen |
    | een klusje | betere namen, een docstring, een eerste test | voer dezelfde gevallen ervoor en erna uit: een klusje verandert hoe code leest, nooit wat hij geeft |

    Elke chatassistent doet dit, met de code erin geplakt, en de assistenten in code-editors ook (voorbeelden in november 2026: ChatGPT, Claude, Gemini en GitHub Copilot Chat).

    **De run.** De course vroeg Qwen3-0.6B drie dingen, elk acht keer: in twee zinnen uitleggen wat `average_without_extremes` teruggeeft; wat hij teruggeeft voor `[1, 2, 3, 4, 100]`, met alleen het getal; en deze functie en zijn variabelen duidelijke namen geven:

    ```python
    def f(a, b):
        c = 0
        for d in a:
            if d > b:
                c = c + 1
        return c
    ```
    """)
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.vstack([
        mo.Html(C.runs_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_answers():
    L.explore("answers", "antwoord", lambda _v: C.answer_frame(int(_v), L.lang), start=15, low=1, high=3 * C.SEEDS, step=1, role="",
              task="Schuif door de 24 antwoorden: acht uitleggen, acht antwoorden op wat hij teruggeeft, en acht keer hernoemen. Let op hoe zeker van zichzelf elk antwoord klinkt.")
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("Een grote assistent legt deze functie beter uit dan Qwen3-0.6B, en ziet de bug vaak wel. Maar een uitleg leunt op de namen en de vorm van de code, dus een bug die eruitziet als wat bedoeld was, mis je makkelijk, ook met een groot model. Welk model het ook is, één geval met de hand uitgerekend laat het zien.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: AI code laten uitleggen

    1. **Vraag om de uitleg in gewone woorden,** en wat de code teruggeeft voor één voorbeeld.
    2. **Kies een klein geval dat je met de hand kunt uitrekenen,** en reken het uit met de regels, niet met de uitleg.
    3. **Voer het uit.** Geeft de code wat jij uitrekende, en wat de uitleg zegt? Waar die drie verschillen, lees je die regel opnieuw.
    4. **Bewaar bij een klusje de gevallen.** Voer ze ervoor en erna uit: de uitkomst mag niet veranderen.
    5. **Lees wat er veranderde,** regel voor regel, voordat je het houdt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De functie van een collega heeft geen docstring. Je vraagt een assistent om er een. Stel dat hij schrijft: *Geeft het totaal van de prijzen terug, min de korting als percentage.*

    ```python
    def total_price(prices, discount):
        total = 0
        for price in prices:
            total = total + price
        return total - discount
    ```

    1. **De uitleg:** het totaal van de prijzen, min een percentage.
    2. **Met de hand:** voor `total_price([10, 20], 10)` tellen de regels 10 en 20 op, dat is 30, en trekken er 10 af: 20. Een korting van 10 procent zou 27 geven.
    3. **Voer het uit:** `total_price([10, 20], 10)` geeft `20`. De code trekt een bedrag af, geen percentage. Je collega bevestigt dat de korting een bedrag is: de docstring klopt niet, de code wel.
    4. **Het klusje:** je schrijft de docstring zelf, *Geeft de som van de prijzen terug, min de korting, een bedrag.* Ervoor en erna geeft `total_price([10, 20], 10)` `20`: een docstring verandert niets aan wat de functie geeft.
    5. **Lees wat er veranderde:** één regel, de docstring.
    """)
    return


@app.cell(hide_code=True)
def text_read_average():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Lees de regels, niet de uitleg
    """)
    return


@app.cell(hide_code=True)
def choice_read_average_cell():
    choice_read_average = L.choice(
        "read-average",
        question="Wat geeft `average_without_extremes([10, 20, 30])` terug, met de functie zoals hij is?",
        options={
            "twenty": "20",
            "third": "ongeveer 6,67",
            "ten": "10",
        },
        correct="third",
        why={
            "third": "Op volgorde: `[10, 20, 30]`. Het midden: `[20]`. De som, 20, wordt gedeeld door `len(scores)`, en dat is 3: ongeveer 6,67. De uitleg beloofde 20, het gemiddelde van het midden.",
            "twenty": "Dat belooft de uitleg: het gemiddelde van het midden. Door welk getal deelt de laatste regel?",
            "ten": "Kijk welke scores `scores[1:-1]` houdt: de eerste en de laatste gaan eraf.",
        },
    )
    choice_read_average
    return (choice_read_average,)


@app.cell(hide_code=True)
def text_fix_average():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Laat de code doen wat de uitleg zegt

        In het vak staat de functie die Qwen3 uitlegde. Verander hem zo dat hij doet wat alle acht uitleggen zeiden: het gemiddelde van de scores zonder de hoogste en de laagste. De check roept hem aan met `[1, 2, 3, 4, 100]`, en met andere lijsten.
        """),
        L.task("Laat `average_without_extremes([1, 2, 3, 4, 100])` `3.0` geven, en voor elke lijst het goede gemiddelde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fix_average():
    box_average = L.code_box("fix-average", lines=6)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_fix_average(box_average):
    result_average = L.check("fix-average", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_fix_average():
    L.hints(
        "fix-average",
        [
            "Reken `[1, 2, 3, 4, 100]` met de hand uit: welke scores zitten in `middle`, wat is hun som, en hoeveel zijn het er?",
            "De som klopt: 9. Hij wordt gedeeld door het verkeerde getal: `len(scores)` telt alle vijf de scores, en er zitten er maar drie in het midden.",
            "Deel door het aantal scores in `middle`: `return sum(middle) / len(___)`.",
        ],
        why="`return sum(middle) / len(middle)` deelt de som van de middelste scores door hoeveel het er zijn: 9 / 3 = 3.0. `len(scores) - 2` geeft hetzelfde getal, en is ook goed.",
    )
    return


@app.cell(hide_code=True)
def text_rename_count():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Doe het klusje dat Qwen3 niet deed

        Geen van de acht antwoorden hernoemde iets in `f`. Doe het zelf: noem de functie `count_above`, en geef zijn argumenten en variabelen namen die zeggen wat erin zit. Verander niets aan wat hij doet. De check roept `count_above([3, 8, 5, 10], 4)` aan, en andere gevallen.
        """),
        L.task("Hernoem `f` tot `count_above`, met duidelijke namen erin, zodat hij precies geeft wat `f` gaf."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rename_count():
    box_rename = L.code_box("rename-count", lines=7)
    box_rename
    return (box_rename,)


@app.cell(hide_code=True)
def check_rename_count(box_rename):
    result_rename = L.check("rename-count", box_rename.value)
    result_rename
    return (result_rename,)


@app.cell(hide_code=True)
def hints_rename_count():
    L.hints(
        "rename-count",
        [
            "Zoek eerst uit wat `f` doet: wat is `a`, wat is `b`, en wat telt `c`?",
            "`a` is een lijst getallen, `b` de grens, `d` één getal uit de lijst, en `c` telt de getallen boven de grens. Hernoem elk ervan overal waar het gebruikt wordt.",
            "`def count_above(numbers, limit):`, dan `count`, `for number in numbers:` en `if number > limit:`; de regels blijven hetzelfde, alleen de namen veranderen.",
        ],
        why="`count_above(numbers, limit)` met `count`, `number` en `if number > limit:` geeft voor elke lijst wat `f` gaf: alleen de namen veranderden. `count_above([3, 8, 5, 10], 4)` geeft `3`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Functies](/course/open/py-functions?lang=nl) schreef je zelf functies; hier las je er een die iemand anders schreef, met een AI die hem uitlegde, en keek je de uitleg na met één geval. In [Van aanvullen tot agents: hulp bij code](/course/open/build-coding-help?lang=nl) koos je een chat voor vragen als deze. De gevallen die je met de hand uitrekent, zijn tests die nog opgeschreven moeten worden: verderop in deze module laat [Code nakijken die AI schreef](/course/open/build-check-ai-code?lang=nl) zien hoe.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_read_average, result_average, result_rename):
    L.wrap_up(choice_read_average, result_average, result_rename)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
