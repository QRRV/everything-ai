# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-check-ai-code. Route: course/route/lessen/bouwen.yaml. Checks, the 16 functions and their picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-check-ai-code", "nl")
    C = L.checks   # what the 16 functions Qwen3 wrote give for six inputs each, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl), en één over [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_examples():
    quick_examples = L.choice(
        "quick-examples",
        question="Een agent schreef een functie vanuit je spec, met drie voorbeelden van wat eruit moet komen. Wat doe je met die voorbeelden?",
        options={
            "tests": "Ze als tests draaien, na elke stap",
            "plan": "Ze in het plan zetten, zodat de agent ze twee keer leest",
            "nothing": "Niets: de agent heeft ze al gelezen",
        },
        correct="tests",
        why={
            "tests": "De voorbeelden zeggen wat eruit moet komen; draai ze op de code, en je ziet meteen of dat zo is.",
            "plan": "Het plan zegt hoe, in stappen. Of de code doet wat de voorbeelden zeggen, zie je alleen door ze te draaien. Zie [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl).",
            "nothing": "Geen van de 24 programma's die het voorbeeld voor zich hadden, had het goed. Zie [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl).",
        },
    )
    quick_examples
    return (quick_examples,)


@app.cell(hide_code=True)
def quick_check_traceback():
    quick_traceback = L.choice(
        "quick-traceback",
        question="Python stopt met een traceback van acht regels. Welke regel lees je eerst?",
        options={
            "first": "De eerste regel",
            "middle": "De regel in het midden, waar je functie staat",
            "last": "De laatste regel",
        },
        correct="last",
        why={
            "last": "In de laatste regel staan de soort fout en wat er misging; daarna lees je de regel uit je eigen code er net boven.",
            "first": "De eerste regel zegt alleen dat er een traceback volgt. Zie [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
            "middle": "Die regel doet ertoe, maar als tweede: eerst wil je weten wat er misging. Zie [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
        },
    )
    quick_traceback
    return (quick_traceback,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Code van AI ziet er meestal goed uit: nette namen, commentaar, geen fouten als je hem uitvoert. Of hij het goede antwoord geeft, is een andere vraag, en dat ontdek je door hem te proberen op gevallen waarvan je het antwoord weet. Teams die met AI werken doen dat bij elke verandering, met tests, een tool die de code leest, en iemand die hem ook leest.

    **Een voorbeeld uit het dagelijks leven.** Of een cake gaar is, ontdek je niet door het recept nog eens te lezen: je steekt er een satéprikker in. Dat is alleen een vergelijking; hieronder staat een echte functie. De course vroeg Qwen3 acht keer om een functie `count_vowels(text)` die de klinkers (a, e, i, o, u) in een tekst telt. Dit is wat Qwen3 schreef:

    ```python
    def count_vowels(text):
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in text if char in vowels)
    ```

    `{'a', 'e', 'i', 'o', 'u'}` is een *set*: een verzameling waarden, zoals een lijst, en hier alleen gebruikt om te vragen of een letter erin zit. `char in vowels` is `True` als de letter `char` in de set zit. De laatste regel telt 1 op voor elke letter van `text` die in `vowels` zit: `1 for char in text if char in vowels` is een `for` met een `if`, geschreven tussen de haakjes van `sum`. Sets krijgen later een eigen les, in [Tuples, sets en sorteren](/course/open/py-tuples-sets?lang=nl), en deze manier om een `for` te schrijven komt terug in [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def predict_apple():
    guess_apple = L.guess(
        "apple",
        question="De course riep elk van de acht functies aan met zes teksten, waaronder `\"Apple\"`. Bij hoeveel van de acht klopt `count_vowels(\"Apple\")`, wat 2 moet zijn?",
        options={
            "all": "Alle acht",
            "none": "Geen enkele",
            "half": "Ongeveer de helft",
        },
        reveal="**Geen enkele.** Alle acht zijn dezelfde functie, teken voor teken, en in de set zit `'a'` maar geen `'A'`: `count_vowels(\"Apple\")` geeft 1, en `count_vowels(\"AEIOU\")` geeft 0. Een test met alleen kleine letters, zoals `\"hello\"`, laat dat nooit zien.",
    )
    guess_apple
    return (guess_apple,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Nieuwe woorden en tekens

    | Woord of teken | Wat het betekent |
    |---|---|
    | **test** | een paar regels code die een functie op een geval uitvoeren en de uitkomst controleren; een **unit test** test één klein deel, zoals één functie, los van de rest |
    | `assert` | `assert` met een voorwaarde erachter: is de voorwaarde `True`, dan gebeurt er niets; is hij `False`, dan stopt Python met een `AssertionError` |
    | **code review** | iemand leest een verandering voordat die blijft: doet hij wat gevraagd was, en kunnen anderen hem volgen? |
    | **static analysis** | een tool leest de code zonder hem uit te voeren, en meldt fouten zoals een naam die Python niet kent |
    | **randomized controlled trial** | een experiment waarin het toeval bepaalt wie met iets werkt en wie zonder, zodat een verschil achteraf door dat ene ding komt |

    Een test met `assert` is één regel per geval:

    ```python
    assert count_vowels("hello") == 2
    assert count_vowels("Apple") == 2
    ```

    Met de `count_vowels` van Qwen3 slaagt de eerste regel, en stopt de tweede met een `AssertionError`: die test vond de bug. Met een goede `count_vowels` slagen ze allebei, en gebeurt er niets.

    ### Drie manieren om na te kijken, en wat elk ervan vindt

    | Controle | Wat hij doet | Wat hij vindt | Tools (voorbeelden in november 2026) |
    |---|---|---|---|
    | een test | voert de code uit op een geval dat jij koos, en vergelijkt de uitkomst | een verkeerd antwoord op dat geval | pytest |
    | static analysis | leest de code zonder hem uit te voeren | wat in elke code fout is, zoals een naam die Python niet kent, of een naam die een waarde krijgt en nooit gebruikt wordt | Ruff, mypy, CodeQL |
    | een review | iemand leest de verandering | of hij doet wat gevraagd was, niet meer en niet minder, en of anderen hem kunnen volgen | de reviewstap op GitHub of GitLab |

    Elk vindt wat de andere twee missen. Een test ziet alleen de gevallen die je schreef; een tool weet niet waar de code voor is; een mens mist wat er goed uitziet.

    **Wat onderzoek vond.** In een randomized controlled trial van Peng en anderen (2023) waren developers met GitHub Copilot 55,8% sneller klaar met één taak, een kleine webserver in JavaScript, dan developers zonder. Pearce en anderen (2022) lieten Copilot 1.689 programma's schrijven voor 89 situaties waarin beveiliging ertoe doet: ongeveer 40% had een bekend soort beveiligingslek.
    """)
    return


@app.cell(hide_code=True)
def predict_metr():
    guess_metr = L.guess(
        "metr",
        question="In 2025 deed METR een randomized controlled trial met 16 ervaren developers, op 246 taken in grote projecten die ze goed kenden; het toeval bepaalde per taak of AI-tools mochten. Vooraf verwachtten ze dat AI hen 24% sneller zou maken. Wat mat de trial?",
        options={
            "slower": "Met AI waren ze 19% langzamer",
            "faster": "Met AI waren ze ongeveer 24% sneller",
            "same": "Met en zonder AI duurde het even lang",
        },
        reveal="**19% langzamer.** Met AI-tools kostten de taken hun 19% meer tijd. Achteraf dachten ze nog steeds dat AI hen 20% sneller had gemaakt. Een gevoel van snelheid is geen meting, net zoals code die er goed uitziet geen test is. De trial met 55,8% sneller ging over één kleine taak die bij nul begon; wat geldt, hangt af van het werk.",
    )
    guess_metr
    return (guess_metr,)


@app.cell(hide_code=True)
def text_run():
    mo.md(r"""
    **De run.** De course vroeg Qwen3-0.6B acht keer om `count_vowels`, en acht keer om een functie `is_palindrome(text)` die zegt of een tekst achterstevoren hetzelfde leest, zonder op hoofdletters en spaties te letten. Daarna is elk van de 16 functies uitgevoerd op zes teksten, en wat eruit kwam vergeleken met het goede antwoord.
    """)
    return


@app.cell(hide_code=True)
def picture_checks():
    mo.vstack([
        mo.Html(C.checks_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_programs():
    L.explore("programs", "functie", lambda _v: C.program_frame(int(_v), L.lang), start=9, low=1, high=2 * C.SEEDS, step=1, role="",
              task="Schuif door de 16 functies: acht keer count_vowels, dan acht keer is_palindrome. Welke zien er goed uit, en welke zijn het ook?")
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("Commentaar in code van AI zegt wat de code zou moeten doen, niet wat hij doet: twee van de acht is_palindrome zeggen dat ze alles weghalen wat geen letter of cijfer is, en laten de spaties staan. En een test kan bij toeval slagen: in \"Ab ba\" staat de spatie precies in het midden, dus die twee halen hem ook. Kies gevallen waarin een fout zichtbaar zou worden: hoofdletters, spaties, een lege tekst, de randen.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept: code nakijken die AI schreef

    1. **Schrijf gevallen op voordat je de code leest,** vanuit de spec: gewone, en de randen, zoals een lege tekst, hoofdletters, een spatie of nul.
    2. **Reken de goede uitkomst van elk geval met de hand uit,** en maak er een `assert`-regel van.
    3. **Draai de tests.** Faalt er een, dan zit de fout in de code of in de test: reken dat geval nog eens met de hand uit.
    4. **Laat een tool de code lezen** op wat in elke code fout is, zoals een naam die Python niet kent.
    5. **Lees de verandering zelf, of laat iemand anders hem lezen:** doet hij wat gevraagd was, niet meer en niet minder, en kun je zien waarom?
    6. **Bewaar de tests.** Na elke verandering draaien ze opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        De spec: `is_palindrome(text)` geeft `True` als een tekst achterstevoren hetzelfde leest, zonder op hoofdletters en spaties te letten. Dit is de eerste van de acht versies die Qwen3 schreef. Je kunt hem testen zonder te weten hoe elke regel werkt:

        ```python
        def is_palindrome(text):
            # Remove all non-alphanumeric characters
            cleaned_text = ''.join([c for c in text if c.isalnum() or c == ' '])
            # Convert to lowercase
            cleaned_text = cleaned_text.lower()
            # Check if it's a palindrome
            return cleaned_text == cleaned_text[::-1]
        ```

        1. **Gevallen uit de spec:** `"racecar"` is een palindroom; `"Taco cat"` ook, als je de hoofdletter en de spatie niet meetelt; `"hello"` niet; en `""`, een lege tekst, leest achterstevoren hetzelfde.
        2. **Als tests:**

            ```python
            assert is_palindrome("racecar") == True
            assert is_palindrome("Taco cat") == True
            assert is_palindrome("hello") == False
            assert is_palindrome("") == True
            ```

        3. **Draai ze:** de tweede regel stopt met een `AssertionError`, want deze versie geeft `False` voor `"Taco cat"`. Klopt de test niet? Met de hand: zonder hoofdletter en spatie is *tacocat* achterstevoren *tacocat*. De test klopt.
        4. **Een tool die de code leest** vindt hier geen onbekende naam, en kan niet weten dat spaties weg moeten: dat staat in de spec, niet in de code.
        5. **Een review** legt het commentaar en de regel eronder naast elkaar: volgens het commentaar gaat alles weg wat geen letter of cijfer is, en de regel houdt `c == ' '`, de spaties. Dat is de bug.
        6. **Bewaar de vier tests.** Is de regel verbeterd, dan slagen ze alle vier, en ze blijven controleren na elke latere verandering.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_catch_vowels():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Schrijf een test die de bug vangt

        Het vak begint met één test. Die slaagt voor de `count_vowels` van Qwen3, en voor een goede, dus hij vangt niets. Voeg een test toe, één regel met `assert`, die slaagt voor een goede `count_vowels` en faalt voor die van Qwen3. De check draait je tests op allebei.
        """),
        L.task("Voeg een regel met `assert` toe die slaagt voor een goede `count_vowels` en faalt voor die van Qwen3."),
    ])
    return


@app.cell(hide_code=True)
def exercise_catch_vowels():
    box_tests = L.code_box("catch-vowels", lines=4)
    box_tests
    return (box_tests,)


@app.cell(hide_code=True)
def check_catch_vowels(box_tests):
    result_tests = L.check("catch-vowels", box_tests.value)
    result_tests
    return (result_tests,)


@app.cell(hide_code=True)
def hints_catch_vowels():
    L.hints(
        "catch-vowels",
        [
            "Kijk naar de tabel hierboven: welke van de zes teksten hadden alle acht versies fout?",
            "Teksten met een klinker als hoofdletter: `\"Apple\"` heeft twee klinkers, de A en de e, en de functie van Qwen3 telt er 1.",
            "Voeg de regel `assert count_vowels(\"Apple\") == ___` toe, met het aantal klinkers in Apple.",
        ],
        why="`assert count_vowels(\"Apple\") == 2` slaagt voor een goede functie, die de A en de e telt, en faalt voor die van Qwen3, die 1 geeft: deze test vangt de bug. `assert count_vowels(\"AEIOU\") == 5` doet dat ook.",
    )
    return


@app.cell(hide_code=True)
def text_fix_vowels():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Los de bug op

        In de cel hieronder staat de `count_vowels` van Qwen3. Verander hem zodat hij ook klinkers als hoofdletter telt, en voer de cel uit. De check roept hem aan met `"Apple"`, en met andere teksten.
        """),
        L.task("Zorg dat `count_vowels(\"Apple\")` `2` geeft, en het goede aantal voor elke tekst."),
    ])
    return


@app.cell
def exercise_fix_vowels():
    def count_vowels(text):
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in text if char in vowels)
    return (count_vowels,)


@app.cell(hide_code=True)
def check_fix_vowels(count_vowels):
    try:
        _value = count_vowels
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_fix = L.check("fix-vowels", _value)
    result_fix
    return (result_fix,)


@app.cell(hide_code=True)
def hints_fix_vowels():
    L.hints(
        "fix-vowels",
        [
            "Welke letters van `\"Apple\"` zijn klinkers, en welke daarvan zitten in de set `vowels`?",
            "Voor Python zijn `'A'` en `'a'` verschillende letters, en in de set zitten alleen de kleine.",
            "Zet de hoofdletters ook in de set: `{'a', 'e', 'i', 'o', 'u', 'A', ___}`.",
        ],
        why="Met `{'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}` is `char in vowels` `True` voor kleine klinkers en voor hoofdletters: `count_vowels(\"Apple\")` geeft 2, en `count_vowels(\"AEIOU\")` geeft 5.",
    )
    return


@app.cell(hide_code=True)
def text_which_check():
    mo.md(r"""
    ### Oefening 3. Welke controle vindt het?

    Zes problemen in code die AI schreef. Kies bij elk de controle die het vindt: een test, static analysis, of een review.
    """)
    return


@app.cell(hide_code=True)
def exercise_which_check():
    sort_checks = L.sort("which-check")
    sort_checks
    return (sort_checks,)


@app.cell(hide_code=True)
def check_which_check(sort_checks):
    result_sort = L.check("which-check", sort_checks)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def hints_which_check():
    L.hints(
        "which-check",
        [
            "Vraag bij elk probleem: zie je het als je de functie op één geval uitvoert, kan een tool het zien zonder te weten waar de code voor is, of is er iemand voor nodig die dat weet?",
            "Apple en Taco cat zie je als een verkeerd antwoord op één geval. Een naam die Python niet kent, of een naam die nooit gebruikt wordt, is in elke code fout.",
            "Test: Apple, Taco cat. Static analysis: Nooit gebruikt, Onbekende naam. Review: Meer dan gevraagd, 37.",
        ],
        why="Een test voert de code uit op een geval en ziet een verkeerd antwoord: Apple en Taco cat. Static analysis leest de code en vindt wat in elke code fout is: een naam die nooit gebruikt wordt, en een naam die Python niet kent, ook op een regel waar geen test komt. Voor een review is iemand nodig die weet waar de code voor is: meer dan de spec vroeg, en een getal dat niemand kan uitleggen.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl) las je een traceback nadat Python stopte; hier schreef je regels die Python expres laten stoppen als een antwoord niet klopt. In [Werken met een coding agent: plan, spec en context](/course/open/build-coding-workflow?lang=nl) draaide je de voorbeelden uit de spec na elke stap als tests: nu kun je die tests zelf schrijven, als `assert`-regels. Deed je [AI code laten uitleggen, en de klusjes laten doen](/course/open/build-ai-explains-code?lang=nl), dan zijn de gevallen die je daar met de hand uitrekende ook tests, zodra je ze opschrijft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_tests, result_fix, result_sort):
    L.wrap_up(result_tests, result_fix, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
