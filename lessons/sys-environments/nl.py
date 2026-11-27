# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-environments. Route: course/route/lessen/systemen.yaml. Checks, de computers, de releases van NumPy, de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-environments", "nl")
    C = L.checks   # de eigen bestanden van de course, de computers, de releases van NumPy, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een tekstbestand lezen](/course/open/py-files?lang=nl) en [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_strip():
    quick_strip = L.choice(
        "quick-strip",
        question="Een bestand bevat de regel `numpy==2.5.3`, met een regeleinde erachter. Wat geeft `line.strip()` voor deze regel?",
        options={
            "right": "De string `numpy==2.5.3`, zonder het regeleinde",
            "number": "De versie `2.5.3`, als getal",
            "same": "De hele regel, met het regeleinde erbij",
            "empty": "Een lege string, want strip haalt alle letters weg",
        },
        correct="right",
        why={
            "right": "`strip()` haalt de spaties en regeleinden aan het begin en het eind van een string weg, en houdt de rest. In deze les lees je de lock file van de course regel voor regel, met `strip()` op elke regel.",
            "number": "`strip()` geeft een string, nooit een getal, en houdt de naam. Wat haalt `strip()` weg? Zie [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "same": "Het regeleinde is precies wat `strip()` aan het eind weghaalt. Wat blijft er over van de regel? Zie [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
            "empty": "`strip()` haalt slechts spaties en regeleinden aan het begin en het eind weg. Wat blijft er over van de regel? Zie [Een tekstbestand lezen](/course/open/py-files?lang=nl).",
        },
    )
    quick_strip
    return (quick_strip,)


@app.cell(hide_code=True)
def quick_check_package():
    quick_package = L.choice(
        "quick-package",
        question="Wat is een package in deze course?",
        options={
            "file": "Eén `.py`-bestand met functies erin",
            "right": "Een map met modules, met een bestand `__init__.py`",
            "function": "Een functie die je aanroept met `from`",
            "program": "Een programma dat andere programma's op je computer installeert",
        },
        correct="right",
        why={
            "right": "Een package is een map met modules. De packages die je installeert, zoals numpy, zijn ook mappen met modules, geschreven door anderen. Deze les gaat over welke versie daarvan op je computer terechtkomt.",
            "file": "Eén `.py`-bestand is een module. Hoe heet een map met modules? Zie [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
            "function": "`from` importeert een module uit een package. Wat is het package zelf? Zie [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
            "program": "Installeren is het werk van een tool als pip of uv, en zo'n tool installeert packages. Wat is een package? Zie [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).",
        },
    )
    quick_package
    return (quick_package,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op 16 juni 2024 kwam NumPy 2.0.0 uit. De release notes waarschuwden voor veranderingen die code breken, waaronder één voor gecompileerde packages: packages met delen die in C geschreven zijn en tegen één versie van NumPy in machinecode omgezet zijn. Packages die tegen NumPy 1.x gebouwd zijn werken niet met NumPy 2.0. Vanaf dat moment kreeg een installatie met de regel `numpy` zonder versie NumPy 2.0.0, en stopte een package dat tegen NumPy 1.x gecompileerd was al bij de import. De melding zei dat een module die met NumPy 1.x gecompileerd is niet in NumPy 2.0.0 kan draaien, omdat de module zou kunnen crashen. Dezelfde code, met dezelfde regel, draaide prima op een computer die een week eerder was ingericht.

    De melding noemde zelf de uitweg: een oudere versie, met `numpy<2`. Deze les gaat over versies opschrijven voordat er iets breekt: één virtual environment per project, en één exacte versie voor elk package, op elke computer. In ML telt dit dubbel, want ook een getraind model hangt van versies af: volgens scikit-learn is er geen ondersteunde manier om een model te laden dat met een andere versie van scikit-learn getraind is.

    **Een voorbeeld uit het dagelijks leven.** Een recept dat alleen "bloem" zegt lukt in de ene keuken en mislukt in de volgende, omdat de ene keuken tarwebloem heeft en de andere zelfrijzend bakmeel. Een recept dat het precieze pak noemt geeft in elke keuken hetzelfde brood.
    """)
    return


@app.cell(hide_code=True)
def predict_locked():
    guess_locked = L.guess(
        "locked",
        question="`pyproject.toml` is het bestand dat een project beschrijft, met de packages die het project nodig heeft. Dit bestand van de course noemt ongeveer twaalf packages, zoals `marimo`, `numpy` en `torch`. De lock file, `uv.lock`, legt ook packages vast. Hoeveel?",
        options={
            "same": "Ongeveer twaalf: dezelfde",
            "thirty": "Ongeveer 30",
            "right": "Ongeveer 100",
            "thousand": "Ongeveer 1.000",
        },
        reveal="**Ongeveer 100.** Elk package dat de course noemt heeft zelf packages nodig, en deze packages weer andere: matplotlib alleen al heeft er negen nodig, zoals pillow en kiwisolver. De lock file legt al deze packages vast, elk op één exacte versie. Het uitgewerkte voorbeeld hieronder telt de packages in je kopie van de course.",
    )
    guess_locked
    return (guess_locked,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Versies.** Packages komen van PyPI, de online index waar pip en uv packages vandaan halen. Elke release van een package heeft een versie, zoals 2.5.3: een major-, een minor- en een patchnummer. Veel projecten volgen semantic versioning, waarin slechts een nieuwe major-versie werkende code mag breken. Net als Python zelf volgt NumPy dit niet: ook een minor-versie mag functies weghalen, na minstens twee releases met een waarschuwing. Elke nieuwe versie kan dus iets veranderen wat je code nodig heeft.

        **Eén virtual environment per project.** Een installer zet packages in één map van één Python, dus kunnen twee projecten die twee versies van pandas nodig hebben deze map niet delen. Een **virtual environment** is een map met een eigen Python en eigen geïnstalleerde packages, voor één project. De module venv van Python maakt zulke mappen sinds Python 3.3: `python -m venv .venv`. De course heeft er ook een: `uv run everything-ai` gebruikt de map `.venv` in de map van de course. De map blijft buiten git, het programma dat de geschiedenis van de bestanden van een project bijhoudt: met de twee bestanden hieronder kun je de map altijd opnieuw maken.

        **Vaste versies.** Een regel in `pyproject.toml` zegt welke versies een project accepteert. `numpy` accepteert elke versie, `numpy>=1.24,<2` een bereik (de komma betekent *en*), en `numpy==1.26.4` één exacte versie. De exacte versie opschrijven heet **dependency pinning**. Een bereik geeft een andere versie zodra er nieuwe releases uitkomen, want een installer neemt de nieuwste release die een bereik toestaat.

        **Versies vergelijk je deel voor deel.** Versie 2.10.0 komt na 2.9.0, maar als strings staat `"2.10.0"` eerder dan `"2.9.0"`: Python vergelijkt strings teken voor teken, en `"1"` komt eerder dan `"9"`. Bij tuples vergelijkt Python deel voor deel: `(2, 10, 0) > (2, 9, 0)` is `True`, want de eerste delen zijn gelijk en 10 is groter dan 9. De oefeningen hieronder gebruiken tuples.

        **Een lock file.** Vaste versies voor de packages die je noemt zijn niet genoeg: marimo, numpy en torch hebben zelf packages nodig, en deze packages weer andere. Een **lock file** legt de exacte versie vast van elk package, ook van deze onderliggende packages. Een tool zoals uv rekent alle versies één keer uit, zodat elke installatie hetzelfde krijgt. uv schrijft `uv.lock` met `uv lock` en installeert de packages daaruit met `uv sync`; volgens de documentatie van uv hoort het bestand in versiebeheer, zodat elke machine dezelfde installatie krijgt. Bij pip print `pip freeze` wat er geïnstalleerd is, als regels zoals `numpy==2.5.3`, en de documentatie van pip zegt erbij dat `pip freeze` geen lock file uitrekent. Sinds 2025 heeft Python ook een standaardformaat voor lock files, `pylock.toml`, uit PEP 751.

        **Een model onthoudt de versies.** Een opgeslagen model hangt af van de versies waarmee het getraind is. Sinds versie 1.3 waarschuwt scikit-learn met een `InconsistentVersionWarning` als je een model laadt dat met een andere versie is opgeslagen. Versies horen dus naast het model, en de volgende lessen leggen deze bij elke run vast. De figuur laat één regel op twee computers zien, zonder en met lock file.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **virtual environment** | | een map met een eigen Python en eigen geïnstalleerde packages, voor één project | `python -m venv .venv` |
        | **dependency pinning** | | de exacte versie opschrijven van een package dat een project gebruikt | `numpy==1.26.4` |
        | **lock file** | | een bestand met de exacte versie van elk package dat een project installeert, de packages van packages meegeteld | `uv.lock` |
        """),
        L.watch_out("In reinforcement learning betekent environment iets anders: de wereld waarin een agent handelt. In deze les is een environment een map met een Python en eigen packages."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Dezelfde packages op elke computer:

    1. Maak één virtual environment per project (`uv venv`, of `python -m venv .venv`), en houd de map buiten git.
    2. Noem in `pyproject.toml` de packages die je code importeert.
    3. Laat de tool één keer elke versie uitrekenen en de lock file schrijven (`uv lock`). Commit `pyproject.toml` en de lock file in git.
    4. Installeer op elke computer uit de lock file (`uv sync`), niet uit de bereiken.
    5. Als twee computers zich anders gedragen, vergelijk dan de versies package voor package.
    6. Upgrade bewust: verander een versie, draai je tests, en commit de nieuwe lock file.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.LOCK_FILE` is het pad van de eigen `uv.lock` van de course, en `C.NAMED` bevat de packages die de `pyproject.toml` van de course noemt, ingelezen door de course. In `uv.lock` begint elk package met een regel `[[package]]`. De cel telt deze regels. Voer de cel uit.
    """)
    return


@app.cell
def example_lock():
    named = C.NAMED
    count = 0
    with open(C.LOCK_FILE) as _f:
        for _line in _f:
            if _line.strip() == "[[package]]":
                count = count + 1
    print(f"pyproject.toml noemt {len(named)} packages")
    print(f"uv.lock legt {count} packages vast")
    return count, named


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De lock file legt vele malen meer packages vast dan `pyproject.toml` noemt. Eén van de packages is de course zelf, en een paar, zoals pytest, zijn alleen voor de eigen tests van de course. Elk ander package is nodig voor de course, direct of via een ander package. Elk package heeft één exacte versie en een hash: een vingerafdruk van het bestand, zodat de installer kan controleren of het binnengekomen bestand precies klopt. Open `uv.lock` in een teksteditor en zoek `name = "numpy"`: de regel eronder zegt welke versie elke leerling van deze course krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_differences():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat is er anders?

        Code draait op je laptop en stopt op de server. `differences(mine, theirs)` krijgt twee dictionaries, één per computer, met de naam van elk geïnstalleerd package als key en de versie als waarde, zoals `pip freeze` de packages opsomt. De functie geeft een lijst `found` terug met de packages die in `mine` staan en in `theirs` ontbreken of daar een andere versie hebben. De loop gaat elke `name` in `mine` langs. De `if` voegt al een package toe dat `theirs` niet heeft, met `found.append(name)`; Python controleert de `elif` eronder, kort voor else if, alleen als de voorwaarde van de `if` niet klopt. Vervang de `...` door een voorwaarde die waar is als het package op de andere computer een andere versie heeft.

        De check roept `differences(C.LAPTOP, C.SERVER)` aan, daarna andere paren computers.
        """),
        L.task("Noem de packages met een andere versie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_differences():
    box_differences = L.code_box("differences", lines=8)
    box_differences
    return (box_differences,)


@app.cell(hide_code=True)
def check_differences(box_differences):
    result_differences = L.check("differences", box_differences.value)
    result_differences
    return (result_differences,)


@app.cell(hide_code=True)
def hints_differences():
    L.hints(
        "differences",
        [
            "Stap 5 van het recept: vergelijk de versies package voor package. `mine[name]` is de versie van `name` op jouw computer, als string, zoals `\"1.26.4\"`.",
            "`theirs` is ook een dictionary, met dezelfde soort keys. Hoe haal je de versie van `name` eruit, en welke vergelijking is waar als twee strings verschillen?",
            "`        elif mine[name] != ___:`",
        ],
        why="`mine[name] != theirs[name]` is waar als beide computers het package hebben, met verschillende versies. Voor `C.LAPTOP` en `C.SERVER` is dat numpy, 1.26.4 tegenover 2.0.1; joblib ontbreekt op de server.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_resolve():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De nieuwste versie die een bereik toestaat

        Bij `numpy>=1.24,<2` neemt een installer de nieuwste release die minstens 1.24 is en onder 2 blijft. `resolve(released, low, high)` doet hetzelfde. `released` is een lijst versies als tuples, van de oudste naar de nieuwste, en `low` en `high` zijn de twee kanten van het bereik, ook als tuples, zoals `(1, 24, 0)` en `(2, 0, 0)`. `chosen` begint als `None`. De loop gaat elke `version` langs, en als de voorwaarde klopt, houdt `chosen = version` deze versie vast, dus blijft de laatste toegestane versie over. Vervang de `...` door een voorwaarde die waar is als `version` minstens `low` is en onder `high` blijft.

        De check roept `resolve(C.RELEASED, (1, 24, 0), (2, 0, 0))` aan, met elke release van NumPy op PyPI van 1.24.0 tot en met 2.5.5, behalve 2.4.0: PyPI heeft deze versie teruggetrokken (yanked), dus slaan installers deze versie over. Daarna andere bereiken en lijsten.
        """),
        L.task("Houd de nieuwste versie die het bereik toestaat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resolve():
    box_resolve = L.code_box("resolve", lines=7)
    box_resolve
    return (box_resolve,)


@app.cell(hide_code=True)
def check_resolve(box_resolve):
    result_resolve = L.check("resolve", box_resolve.value)
    result_resolve
    return (result_resolve,)


@app.cell(hide_code=True)
def hints_resolve():
    L.hints(
        "resolve",
        [
            "Een installer neemt de nieuwste release die een bereik toestaat. Python vergelijkt tuples deel voor deel, dus kun je `version` direct met `low` vergelijken. Welke vergelijking met `low` laat `low` zelf toe?",
            "`<2` betekent onder 2.0.0, dus is 2.0.0 zelf niet toegestaan. Welke vergelijking met `high` laat `high` zelf weg, en welk woord verbindt twee voorwaarden?",
            "`        if version >= low and ___:`",
        ],
        why="`version >= low and version < high` staat de versies toe van `low` tot net onder `high`. Voor `numpy>=1.24,<2` is dat 1.26.4, de laatste NumPy 1; voor `numpy>=2.0,<3` is het 2.5.5.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Eén regel, geïnstalleerd op een gekozen dag

    De slider kiest een dag: de eerste van een maand, van januari 2023 tot en met november 2026. Op deze dag installeren vier projecten NumPy van PyPI, elk met een andere regel: `numpy` zonder versie, het bereik `numpy>=1.24,<2`, de vaste versie `numpy==1.24.1`, en een `uv.lock` die op 1 januari 2023 gemaakt is uit de regel `numpy`. Elk project neemt de nieuwste release die de regel van het project op deze dag toestaat, net als je `resolve`. De stippen op de tijdlijn zijn de releases van NumPy op PyPI, een rij in de kleur van fouten wijkt af van de lock file, en de onderste rij zegt of een package dat tegen NumPy 1.x gecompileerd is nog werkt naast de regel zonder versie.
    """)
    return


@app.cell(hide_code=True)
def demo_month():
    L.explore("month", "maanden na januari 2023", lambda _v: C.frame(_v, L.lang), start=0, low=0, high=46, step=1,
              role="", task="Zoek de eerste maand waarin het bereik een andere versie installeert dan de lock file, en de eerste waarin de regel zonder versie NumPy 2 krijgt.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 1 maart 2023 installeert het bereik een nieuwere NumPy dan de lock file, net als de regel zonder versie: eerst 1.24.2, en vanaf maart 2024 1.26.4. Twee computers die een maand na elkaar ingericht zijn verschillen al, zonder één foutmelding. Op 1 juli 2024 installeert de regel zonder versie voor het eerst NumPy 2.0.0, en vanaf dan stopt het package dat tegen NumPy 1.x gecompileerd is; het bereik blijft op 1.26.4, want het bereik stopt onder 2. Slechts de vaste versie en de lock file geven elke maand 1.24.1, tot en met november 2026, terwijl de regel zonder versie al bij 2.5.4 is. Een vaste versie en een lock file houden je niet tegen bij een upgrade: een upgrade wordt een stap die je bewust zet, met je tests, in plaats van een verrassing op de dag dat er een nieuwe release uitkomt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lock():
    mo.md(r"""
    ### Oefening 3. Vaste versies zijn nog geen lock file

    Een team schrijft elk package in de `pyproject.toml` met `==`: twaalf exacte versies. Een jaar later installeert een nieuwe collega het project met alleen `pyproject.toml`, zonder lock file.
    """)
    return


@app.cell(hide_code=True)
def choice_lock_cell():
    choice_lock = L.choice(
        "lock",
        question="Krijgt de collega dezelfde packages als het team?",
        options={
            "same": "Ja: elk package in het bestand heeft één exacte versie",
            "newest": "Nee: een installer neemt altijd de nieuwste release",
            "right": "Niet allemaal: onderliggende packages kunnen nieuwer zijn",
            "never": "Nee: dezelfde versies kun je nooit een tweede keer installeren",
        },
        correct="right",
        why={
            "right": "De twaalf regels leggen slechts de genoemde packages vast. Voor de onderliggende packages gelden eigen bereiken, en een installer neemt de nieuwste release die elk bereik toestaat. Een lock file legt ook deze packages vast.",
            "same": "De twaalf genoemde packages zijn hetzelfde. Welke packages noemt het bestand helemaal niet?",
            "newest": "Bij de twaalf vaste regels neemt de installer precies de vastgelegde versie. Voor welke packages kiest de installer zelf een versie?",
            "never": "Een oude versie blijft op PyPI staan, dus kun je deze opnieuw installeren. Welke packages liggen helemaal niet vast?",
        },
    )
    choice_lock
    return (choice_lock,)


@app.cell(hide_code=True)
def text_exercise_projects():
    mo.md(r"""
    ### Oefening 4. Twee projecten, één laptop

    Je oude project heeft pandas 1.5 nodig, omdat de code iets gebruikt wat pandas 2 weghaalde. Je nieuwe project heeft pandas 2.2 nodig.
    """)
    return


@app.cell(hide_code=True)
def choice_projects_cell():
    choice_projects = L.choice(
        "projects",
        question="Hoe draaien beide projecten op dezelfde laptop?",
        options={
            "both": "Installeer beide versies van pandas in dezelfde Python",
            "upgrade": "Installeer pandas 2.2 en repareer het oude project later",
            "copy": "Kopieer de bestanden van pandas naar elke projectmap",
            "right": "Maak een virtual environment voor elk project",
        },
        correct="right",
        why={
            "right": "Elk project krijgt een eigen map met een eigen Python en een eigen pandas, dus staan beide versies op één laptop zonder elkaar in de weg te zitten.",
            "both": "Eén Python bevat één versie van een package: de tweede installeren vervangt de eerste. Waar kan een tweede versie staan?",
            "upgrade": "Dan stopt het oude project vandaag. Hoe draaien beide nu?",
            "copy": "Dan installeer je met de hand, en nergens staat welke versie elk project heeft. Welke map bevat de packages van één project?",
        },
    )
    choice_projects
    return (choice_projects,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In *Het probleem formuleren voor productie* besliste je wat een model voorspelt. Deze les zorgt dat de code eromheen op elke computer hetzelfde draait: een virtual environment per project, en een lock file met elke versie. [Een tekstbestand lezen](/course/open/py-files?lang=nl) gaf de loop over de regels van `uv.lock`, en [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl) de packages, die nu versies hebben. In de volgende les, *Zelfde run, zelfde uitkomst: seeds en determinisme*, komen de vaste versies terug naast seeds: wat een training nog meer nodig heeft om twee keer dezelfde uitkomst te geven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lock, choice_projects, result_differences, result_resolve):
    L.wrap_up(result_differences, result_resolve, choice_lock, choice_projects)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
