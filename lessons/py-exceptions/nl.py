# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les py-exceptions. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-exceptions", "nl")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_last_line():
    quick_last_line = L.choice(
        "quick-last-line",
        question="Een traceback eindigt met `ZeroDivisionError: division by zero`. Wat vertelt die laatste regel je?",
        options={
            "what": "Wat er misging: er werd door nul gedeeld",
            "where": "Op welke regel van de cel het misging",
            "call": "Welke aanroep het allemaal begon",
        },
        correct="what",
        why={
            "what": "De laatste regel heeft de soort fout en de uitleg van Python: wat er misging.",
            "where": "De regel waar het gebeurde, staat één stap hoger in de traceback. Zie de figuur in [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
            "call": "De aanroepen staan hoger in de traceback. Zie de figuur in [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
        },
    )
    quick_last_line
    return (quick_last_line,)


@app.cell(hide_code=True)
def quick_check_no_error():
    quick_no_error = L.choice(
        "quick-no-error",
        question="Een functie geeft geen fout bij je drie tests. Klopt hij?",
        options={
            "yes": "Ja: geen fout betekent dat hij klopt",
            "not-sure": "Niet zeker: hij kan nog steeds verkeerde antwoorden geven; test met invoer waarvan je het antwoord weet",
            "red": "Alleen als marimo niets in rood laat zien onder de check",
        },
        correct="not-sure",
        why={
            "yes": "Een functie kan een verkeerd antwoord geven zonder enige fout. Zie het blok *Let op* in [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
            "not-sure": "Geen fout betekent niet dat het klopt: alleen een test waarvan je het antwoord weet, laat zien of het antwoord goed is.",
            "red": "Rood laat alleen fouten zien, en een verkeerd antwoord geeft er geen. Zie het uitgewerkte voorbeeld in [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
        },
    )
    quick_no_error
    return (quick_no_error,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sommige problemen zijn geen bugs, maar dingen die je kunt verwachten: iemand typt `twaalf` waar je om een getal vroeg, of een lijst blijkt leeg te zijn. Dan wil je niet dat je hele programma stopt. Met `try` en `except` vangt je programma zo'n probleem op en gaat het verder. En met `raise` kan je eigen functie een invoer weigeren die niet kan kloppen, met een duidelijke melding, in plaats van stilletjes onzin terug te geven.

    **Een voorbeeld uit het dagelijks leven.** Een snoepautomaat. Stop je er een buitenlandse munt in, dan gaat hij niet kapot: hij geeft de munt terug en laat een melding zien. En is een product uitverkocht, dan weigert hij het te verkopen, en zegt dat ook.
    """)
    return


@app.cell(hide_code=True)
def predict_try():
    guess_try = L.guess(
        "try",
        question="Wat denk je dat dit programma laat zien?\n\n```python\ntry:\n    print(int(\"12\"))\n    print(int(\"twelve\"))\n    print(\"done\")\nexcept ValueError:\n    print(\"not a number\")\n```",
        options={
            "stops": "`12`, dan een fout, en verder niets",
            "caught": "`12` en `not a number`",
            "all": "`12`, `not a number` en `done`",
        },
        reveal="**`12` en `not a number`.** `int(\"12\")` maakt van de tekst het getal 12. `int(\"twelve\")` kan dat niet, en stopt met een `ValueError`. Python springt meteen naar `except ValueError:`, dus `print(\"done\")` in de `try` draait nooit. Het programma zelf stopt niet.",
    )
    guess_try
    return (guess_try,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([mo.md(r"""
    ## Het idee

    `try` voert regels uit die mis kunnen gaan. Stopt er een met de soort fout die na `except` staat, dan springt Python naar de regels onder `except`, en gaat het programma verder. Met `raise` stopt je eigen code met een fout naar keuze.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **try** | voert de ingesprongen regels eronder uit; stopt er een met een fout, dan wordt de rest ervan overgeslagen | `try:` |
    | **except** | met een soort fout: de regels eronder draaien alleen als die soort fout in de `try` gebeurt; daarna gaat het programma verder | `except ValueError:` |
    | **raise** | stopt met een fout naar keuze, met een eigen melding | `raise ValueError("age below 0")` |
    | **ValueError** | de soort fout voor een waarde van de goede soort die toch niet bruikbaar is, zoals de tekst `"twaalf"` waar `int` een geheel getal nodig heeft | |
    | **int(...)** | maakt van een tekst een geheel getal; een tekst die dat niet is, geeft een `ValueError` | `int("12")` geeft `12` |

    Een tekst die geen geheel getal is, zoals `"twaalf"`, `"3.5"` of de lege tekst `""`, geeft een `ValueError`; `"-7"` gaat goed. Een fout van een andere soort dan die na `except` wordt niet opgevangen: het programma stopt met een traceback, net als eerst. Een `raise` in een functie stopt de functie meteen, net als `return`, maar dan met een fout; de code die de functie aanriep, kan die fout opvangen met `try` en `except`.
    """)])
    return


@app.cell(hide_code=True)
def picture_paths():
    mo.vstack([
        mo.md("```python\ntry:\n    n = int(text)\n    print(n * 2)\nexcept ValueError:\n    print(\"not a number\")\nprint(\"done\")\n```"),
        draw.if_else("n = int(text)", ["print(n * 2)"], ['print("not a number")'], ['print("done")'],
                     yes_label="geen fout", no_label="ValueError", title="try en except: twee wegen erdoorheen"),
        mo.md(r"""
        Werkt `int(text)`, dan draait de rest van de `try`, en wordt de `except` overgeslagen. Stopt het met een `ValueError`, dan slaat Python de rest van de `try` over en voert het de regels onder `except` uit. Hoe dan ook gaat het programma daarna verder.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Noem altijd de soort fout na `except`. Een kale `except:` vangt elke fout op, ook een `NameError` door een typefout, en verstopt zo je bugs. En houd de `try` klein: de regel die mis kan gaan om een reden die je verwacht, en wat de uitkomst ervan nodig heeft."),
        mo.md(r"""
        ### Recept

        Zo vang je een probleem op dat je kunt verwachten, of weiger je een invoer:

        1. Zoek de regel die mis kan gaan om een reden die je verwacht, niet door een bug: zoals `int(text)` met een tekst die iemand typte.
        2. Schrijf `try:`, met die regel ingesprongen eronder.
        3. Schrijf `except`, de soort fout en een dubbele punt, en daaronder wat er in plaats daarvan moet gebeuren.
        4. Kijk in je eigen functie eerst naar de invoer: kan die niet kloppen, dan `raise ValueError("wat er mis is")`.
        5. Test beide wegen: een invoer die werkt, en een die misgaat.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Uren die mensen typten, waarvan er één geen getal is; en een functie die weigert door nul te delen. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_try_raise():
    # 1. een getypt antwoord is misschien geen geheel getal
    def _minutes(text):
        # 2. de regel die mis kan gaan, onder try
        try:
            return int(text) * 60
        # 3. except, de soort fout, en wat er in plaats daarvan gebeurt
        except ValueError:
            return 0

    for _text in ["2", "twee", "5"]:
        print(f"{_text} uur is {_minutes(_text)} minuten")

    # 4. je eigen functie weigert een invoer die niet kan kloppen
    def _percent(part, whole):
        if whole == 0:
            raise ValueError("delen door 0: er is geen percentage")
        return part / whole * 100

    # 5. test beide wegen; de except vangt de fout op die de functie geeft
    print(_percent(1, 4))
    try:
        print(_percent(1, 0))
    except ValueError:
        print("geen percentage")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Onder de cel verschijnen vijf regels. Bij `twee` stopte `int` met een `ValueError`, dus gaf `_minutes` 0 terug, en ging de loop verder met `5`. `_percent(1, 4)` geeft `25.0`; `_percent(1, 0)` gaf zijn eigen `ValueError`, en de `except` om de aanroep ving die op: `geen percentage`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_safe_int():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een veilig getal

        Vul `safe_int(text, default)` in, in de cel hieronder: hij geeft `int(text)` terug als de tekst een geheel getal is, en `default` als `int` stopt met een `ValueError`. De check roept hem aan met `"42"`, met `"-7"`, en met teksten die geen geheel getal zijn.
        """),
        L.task("Laat `safe_int` het getal teruggeven, of `default` als de tekst geen geheel getal is."),
    ])
    return


@app.cell
def exercise_safe_int():
    def safe_int(text, default):
        ...
    return (safe_int,)


@app.cell(hide_code=True)
def check_safe_int(safe_int):
    try:
        _value = safe_int
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_safe_int = L.check("safe-int", _value)
    result_safe_int
    return (result_safe_int,)


@app.cell(hide_code=True)
def hints_safe_int():
    L.hints(
        "safe-int",
        [
            "Kijk naar de loop in het uitgewerkte voorbeeld: `int(...)` staat onder `try:`, en de `except` zegt wat er in plaats daarvan gebeurt.",
            "Onder `try:` geef je `int(text)` terug. Onder `except ___:` geef je de standaardwaarde terug.",
            "`try:`, daaronder `return int(text)`; dan `except ___:`, en daaronder `return default`.",
        ],
        why="`try: return int(text)` geeft het getal terug als de tekst een geheel getal is. Stopt `int` met een `ValueError`, dan springt Python naar `except ValueError:`, en geeft `return default` de standaardwaarde terug.",
    )
    return


@app.cell(hide_code=True)
def text_raise_caught():
    mo.md(r"""
    ### Oefening 2. Een eigen fout, opgevangen
    """)
    return


@app.cell(hide_code=True)
def choice_raise_caught_cell():
    choice_raise_caught = L.choice(
        "raise-caught",
        question="Wat laat dit programma zien?\n\n```python\ndef check_age(age):\n    if age < 0:\n        raise ValueError(\"age below 0\")\n    return age\n\ntry:\n    print(check_age(-5))\nexcept ValueError:\n    print(\"bad age\")\nprint(\"done\")\n```",
        options={
            "minus": "`-5` en `done`",
            "stops": "Alleen de fout `ValueError: age below 0`",
            "caught": "`bad age` en `done`",
        },
        correct="caught",
        why={
            "minus": "Bij -5 is de voorwaarde `age < 0` waar, dus `raise` draait vóór `return`. Kijk nog eens naar het woord *raise* in de tabel.",
            "stops": "De aanroep staat onder `try:`, en de soort fout past bij de `except`. Kijk nog eens naar de figuur: welke weg gaat Python?",
            "caught": "`check_age(-5)` geeft een `ValueError`, dus de `print` in de `try` laat niets zien. De `except` laat `bad age` zien, en daarna gaat het programma verder met `done`.",
        },
    )
    choice_raise_caught
    return (choice_raise_caught,)


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een eigen, duidelijke fout

        Bij een lege lijst stopt deze `average` met een `ZeroDivisionError`: het woord van Python voor wat het merkte, niet voor wat er mis is. Laat hem een `ValueError` geven met een eigen melding als de lijst leeg is, vóór de deling. 0 teruggeven is niet genoeg: 0 lijkt een echt gemiddelde, en de code die de functie aanroept, kan het verschil niet zien. De check roept hem ook aan met lijsten die wel getallen hebben. Wil je de lege lijst zelf testen, zet de aanroep dan onder `try:` met `except ValueError:`, zoals in stap 5 van het uitgewerkte voorbeeld; anders stopt je eigen fout de cel, en kan de check niet draaien.
        """),
        L.task("Laat `average` een `ValueError` met een melding geven bij een lege lijst."),
    ])
    return


@app.cell
def exercise_average():
    def average(numbers):
        total = 0
        for n in numbers:
            total = total + n
        return total / len(numbers)
    return (average,)


@app.cell(hide_code=True)
def check_average(average):
    try:
        _value = average
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_average = L.check("average", _value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average",
        [
            "Stap 4 van het recept: kijk eerst naar de invoer, bovenaan de functie. Bij welke lijst kan `average` geen goed antwoord geven?",
            "Bij de lege lijst: `len(numbers) == 0`. Stop dan met een eigen fout, vóór de deling.",
            "`if len(numbers) == 0:` en daaronder `raise ___(\"the list is empty\")`.",
        ],
        why="`if len(numbers) == 0: raise ValueError(\"the list is empty\")` stopt de functie vóór de deling, met een melding die zegt wat er mis is. Code die `average` aanroept, kan hem opvangen met `except ValueError:`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl) las je de fouten die Python gaf; nu vang je de fouten op die je verwacht, en geef je zelf een fout. Later in de course geeft kant-en-klare code een duidelijke fout als getallen niet bij elkaar passen, en controleer je in je eigen code de invoer op dezelfde manier. In de volgende les lees je een tekstbestand, en een bestand dat er niet is, geeft een fout die je kunt verwachten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_raise_caught, result_average, result_safe_int):
    L.wrap_up(result_safe_int, choice_raise_caught, result_average)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
