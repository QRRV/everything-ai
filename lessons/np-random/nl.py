# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les np-random. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("np-random", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Code uit modules gebruiken](/course/open/py-imports?lang=nl) en [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_randint():
    quick_randint = L.choice(
        "quick-randint",
        question="Welke getallen kan `random.randint(1, 6)` geven?",
        options={
            "five": "1, 2, 3, 4 en 5",
            "six": "1, 2, 3, 4, 5 en 6",
            "zero": "0, 1, 2, 3, 4, 5 en 6",
        },
        correct="six",
        why={
            "five": "Bij `randint` doet het tweede getal wel mee. Zie het idee in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
            "six": "`random.randint(1, 6)` geeft een geheel getal van 1 tot en met 6: een dobbelsteen.",
            "zero": "Hij begint bij zijn eerste getal, 1. Zie het idee in [Code uit modules gebruiken](/course/open/py-imports?lang=nl).",
        },
    )
    quick_randint
    return (quick_randint,)


@app.cell(hide_code=True)
def quick_check_named():
    quick_named = L.choice(
        "quick-named",
        question="Wat is `axis=0` in `m.sum(axis=0)`?",
        options={
            "named": "Een waarde met een naam: de naam `axis`, dan `=` en de waarde",
            "new": "Een nieuwe naam `axis` die het programma maakt, met de waarde 0",
            "compare": "Een controle of `axis` 0 is",
        },
        correct="named",
        why={
            "named": "Tussen de haakjes geeft `axis=0` aan `sum` de waarde 0, onder de naam `axis`.",
            "new": "Het maakt geen naam in je programma: het gaat alleen `sum` in. Zie de tabel met nieuwe woorden in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
            "compare": "Een controle zou `==` zijn, met twee tekens. Zie de tabel met nieuwe woorden in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl).",
        },
    )
    quick_named
    return (quick_named,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model begint met willekeurige getallen, voordat het iets geleerd heeft, en het ziet zijn voorbeelden in een willekeurige volgorde. Twee runs geven dus net iets andere modellen. Om een fout te vinden, of om twee ideeën eerlijk te vergelijken, wil je hetzelfde nog eens draaien, met dezelfde willekeurige getallen: dat doet een seed. Als je later in de course een wetenschappelijke paper nabouwt, hoort de seed bij het recept. Maar één seed laat maar één run zien: hoeveel de uitkomsten met andere seeds veranderen, zie je pas als je er meer probeert.

    **Een voorbeeld uit het dagelijks leven.** In veel games kun je een seed intypen voor een nieuwe wereld: wie dezelfde seed intypt, krijgt dezelfde wereld.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same",
        question="NumPy maakt willekeurige getallen met een random number generator, of RNG. Twee RNG's worden allebei gemaakt met seed 42, en je vraagt ze allebei om drie dobbelsteenworpen. Wat denk je dat er gebeurt?",
        options={
            "same": "Ze geven allebei dezelfde drie worpen",
            "different": "Ze geven verschillende worpen",
            "continue": "De tweede gaat verder waar de eerste stopte",
        },
        reveal="**Ze geven allebei dezelfde drie worpen.** Elke RNG begint bij zijn eigen seed. Met dezelfde seed maken ze dezelfde getallen, in dezelfde volgorde. Alleen als je dezelfde RNG opnieuw iets vraagt, krijg je nieuwe getallen.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    NumPy maakt willekeurige getallen met een random number generator, kort RNG; in code heet hij meestal `rng`. Je maakt er een met een seed, en vraagt hem dan om getallen: gehele getallen, floats, of een willekeurige volgorde. De getallen lijken willekeurig, maar volgen helemaal uit de seed.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent | In code |
    |---|---|---|
    | **random number generator (RNG)** | maakt getallen die willekeurig lijken maar volgen uit zijn seed; NumPy maakt er een met `np.random.default_rng` | `rng = np.random.default_rng(42)` |
    | **`rng.integers(low, high, size=n)`** | n gehele getallen vanaf `low` tot `high`; `high` zelf doet niet mee | `rng.integers(1, 7, size=3)`: drie dobbelsteenworpen |
    | **`rng.random(size=n)`** | n floats vanaf 0 tot 1; 1 zelf doet niet mee | `rng.random(size=3)` |
    | **`rng.permutation(items)`** | de items in een willekeurige volgorde, als nieuwe array | `rng.permutation(["a", "b", "c"])` |
    | **`size=(rijen, kolommen)`** | willekeurige getallen in een matrix met die shape | `rng.random(size=(2, 3))` |

    `size=3` is een waarde met een naam, net als `axis=0` in [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl). De seed is het startpunt, net als bij `random.seed` in [Code uit modules gebruiken](/course/open/py-imports?lang=nl). Een RNG gaat door: elke keer dat je erom vraagt, krijg je de volgende getallen; maak je hem opnieuw met dezelfde seed, dan begint hij opnieuw. En een array kan ook tekst bevatten, zoals namen: dan laat hij die tussen aanhalingstekens zien, en zegt zijn dtype hoe lang de langste tekst is, zoals `<U3`.
    """)
    return


@app.cell(hide_code=True)
def picture_seeds():
    mo.vstack([
        draw.value_rows(C.picture_rows({"first": "een nieuwe RNG, seed 7", "same": "nog een nieuwe, seed 7: dezelfde worpen",
                                        "asked": "eerste RNG opnieuw", "again": "de eerste RNG, opnieuw gevraagd: hij gaat door",
                                        "other": "seed 8: andere worpen"}),
                        title="Zes dobbelsteenworpen per keer, van RNG's"),
        mo.md(r"""
        Elke rij is zes dobbelsteenworpen, van `integers(1, 7, size=6)`. De eerste twee komen van twee nieuwe RNG's met seed 7, dus zijn ze gelijk, in dezelfde volgorde. De derde rij vraagt het de eerste RNG opnieuw: die gaat door, dus zijn de worpen nieuw. De laatste rij komt van een RNG met seed 8: andere worpen. NumPy rekent deze worpen uit als de les opent; met dezelfde versie van NumPy komen ze op elke computer hetzelfde uit.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`rng.integers(1, 6)` geeft nooit een 6: zijn tweede getal doet niet mee. Bij `random.randint(1, 6)` van Python doet het wel mee."),
        mo.md(r"""
        ### Recept

        Zo maak je willekeurige getallen die je kunt herhalen:

        1. Maak één random number generator met een seed, voordat je om getallen vraagt: `rng = np.random.default_rng(42)`. In een functie die een seed krijgt, maak je hem binnen de functie.
        2. Vraag hem om getallen: `rng.integers`, `rng.random` of `rng.permutation`, met `size=` voor hoeveel, of voor de shape.
        3. Gebruik die ene RNG voor alles wat willekeurig is, elke keer in dezelfde volgorde.
        4. Wil je dezelfde getallen nog eens, begin dan opnieuw: maak de RNG opnieuw met dezelfde seed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Dobbelstenen, namen in een willekeurige volgorde, een kleine matrix met floats, en een functie die één dobbelsteen gooit met een seed. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_random():
    import numpy as np

    # 1. één RNG, met een seed
    _rng = np.random.default_rng(7)
    # 2. en 3. tien dobbelsteenworpen: vanaf 1 tot 7; 7 zelf doet niet mee
    print(_rng.integers(1, 7, size=10))
    # 2. en 3. vijf namen in een willekeurige volgorde
    print(_rng.permutation(["Ada", "Bo", "Cas", "Dex", "Eva"]))
    # 2. en 3. floats vanaf 0 tot 1, in een matrix met shape (2, 3)
    print(_rng.random(size=(2, 3)))
    # 4. een nieuwe RNG met dezelfde seed begint opnieuw
    _again = np.random.default_rng(7)
    print(_again.integers(1, 7, size=10))


    # 1. in een functie die een seed krijgt: de RNG wordt binnenin gemaakt, met die seed
    def _one_roll(seed):
        return np.random.default_rng(seed).integers(1, 7)


    print(_one_roll(7), _one_roll(7), _one_roll(8))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste regel heeft tien worpen, van 1 tot en met 6. De namen staan in een andere volgorde dan in de lijst, en de matrix heeft zes floats tussen 0 en 1. De vierde regel heeft dezelfde tien worpen als de eerste: de nieuwe RNG, `_again`, begint opnieuw bij seed 7. De functie maakt elke keer een nieuwe RNG, met de seed die hij krijgt, dus geeft seed 7 twee keer dezelfde worp, en seed 8 meestal een andere. Voer de cel nog eens uit: elke regel blijft hetzelfde, omdat de seeds vastliggen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_number():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Eén getal van 1 tot en met 10
    """)
    return


@app.cell(hide_code=True)
def choice_number_cell():
    choice_number = L.choice(
        "one-number",
        question="`rng` is een random number generator. Welke regel geeft één geheel getal van 1 tot en met 10?",
        options={
            "short": "`rng.integers(1, 10)`",
            "zero": "`rng.integers(0, 10)`",
            "ten": "`rng.integers(1, 11)`",
        },
        correct="ten",
        why={
            "short": "Het tweede getal doet niet mee, dus geeft deze nooit een 10. Kijk nog eens naar het blok *Let op* boven het recept.",
            "zero": "Die begint bij 0, dus geeft hij 0 tot en met 9. Waar moet hij beginnen, en waar stoppen?",
            "ten": "Vanaf 1 tot 11, en 11 zelf doet niet mee: 1 tot en met 10.",
        },
    )
    choice_number
    return (choice_number,)


@app.cell(hide_code=True)
def text_exercise_rolls():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Twintig worpen die je kunt herhalen

        Maak een random number generator met seed 2026, vraag hem in één keer om twintig dobbelsteenworpen, en noem ze `rolls`: een vector van twintig gehele getallen van 1 tot en met 6. Vervang de `...`. De laatste regel is `rolls`, zodat de check het ziet.
        """),
        L.task("Maak `rolls`: twintig dobbelsteenworpen van een random number generator met seed 2026."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rolls():
    box_rolls = L.code_box("rolls", lines=6)
    box_rolls
    return (box_rolls,)


@app.cell(hide_code=True)
def check_rolls(box_rolls):
    result_rolls = L.check("rolls", box_rolls.value)
    result_rolls
    return (result_rolls,)


@app.cell(hide_code=True)
def hints_rolls():
    L.hints(
        "rolls",
        [
            "Stap 1 en 2 van het recept: eerst de RNG, dan de getallen.",
            "De worpen komen van `rng.integers`. Zijn tweede getal doet niet mee: welk getal maakt een 6 mogelijk? `size=20` vraagt er in één keer twintig.",
            "`rng = np.random.default_rng(___)`, dan `rolls = rng.integers(___, ___, size=20)`.",
        ],
        why="`np.random.default_rng(2026)` maakt een RNG die begint bij seed 2026. `rng.integers(1, 7, size=20)` vraagt hem om twintig gehele getallen van 1 tot en met 6. Met dezelfde seed en dezelfde versie van NumPy krijgt iedereen dezelfde twintig worpen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shuffled():
    mo.vstack([
        mo.md(f"""
        ### Oefening 3. Een willekeurige volgorde die je kunt herhalen

        Maak `shuffled(items, seed)` af: die maakt een random number generator met `seed`, en geeft `items` terug in een willekeurige volgorde. Dezelfde seed moet elke keer dezelfde volgorde geven. Vervang de `...`. De check roept `shuffled(["Ada", "Bo", "Cas", "Dex", "Eva"], 7)` aan: dat moet `{C.shuffled_example()}` geven. Dat is een andere volgorde dan in het uitgewerkte voorbeeld, met dezelfde seed: daar had de RNG eerst tien keer gegooid, dus was hij al verder.
        """),
        L.task("Laat `shuffled(items, seed)` de items teruggeven in een willekeurige volgorde die bij dezelfde seed steeds gelijk is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shuffled():
    box_shuffled = L.code_box("shuffled", lines=7)
    box_shuffled
    return (box_shuffled,)


@app.cell(hide_code=True)
def check_shuffled(box_shuffled):
    result_shuffled = L.check("shuffled", box_shuffled.value)
    result_shuffled
    return (result_shuffled,)


@app.cell(hide_code=True)
def hints_shuffled():
    L.hints(
        "shuffled",
        [
            "Stap 1 en 2 van het recept, binnen de functie: een RNG, dan een willekeurige volgorde.",
            "Maak de RNG met de seed die de functie krijgt, niet met een eigen getal. Voor een willekeurige volgorde: kijk in de tabel met nieuwe woorden.",
            "`rng = np.random.default_rng(___)`, en dan `return rng.___(items)`.",
        ],
        why="`np.random.default_rng(seed)` laat de RNG beginnen bij de seed die de check meegeeft, dus geeft dezelfde seed dezelfde volgorde. `rng.permutation(items)` geeft de items terug in die willekeurige volgorde, als nieuwe array.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Code uit modules gebruiken](/course/open/py-imports?lang=nl) maakte `random.seed` de willekeurige getallen van Python herhaalbaar; een random number generator van NumPy doet hetzelfde, en geeft hele arrays tegelijk. In [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) koos een waarde met een naam, `axis=0`, de richting; hier kiest `size=` hoeveel getallen, of hun shape. Later in de course, in *Sampling: laat het model schrijven*, kiest je language model zijn volgende letter met een RNG. In de volgende les, *Een afbeelding is een rooster van getallen*, bekijk je een plaatje als array.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_number, result_rolls, result_shuffled):
    L.wrap_up(choice_number, result_rolls, result_shuffled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
