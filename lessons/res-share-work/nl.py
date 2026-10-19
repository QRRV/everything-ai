# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-share-work. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-share-work", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_before():
    mo.md(r"""
    **Ook van buiten de course:** een paar woorden. **Git** is een programma dat elke versie van de bestanden in een map bewaart, zijn **repository**; **pushen** is die versies kopiëren naar een website zoals **GitHub**, waar anderen ze kunnen halen. Een **model hub**, zoals Hugging Face, is een website waar mensen getrainde modellen delen. Een **hexteken** is een van de 16 tekens 0 tot en met 9 en a tot en met f waarin een hash geschreven wordt. Een **terminal** is een venster waarin je opdrachten typt die programma's starten, zoals `python rerun.py`; een **script** is een Python-bestand dat je zo in zijn geheel uitvoert. [De terminal: een script uitvoeren](/course/open/py-terminal?lang=nl) laat beide zien. Is Git nieuw voor je, lees dan het eerste hoofdstuk van [Pro Git](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control) (in het Engels).
    """)
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een experimentlogboek bijhouden](/course/open/res-experiment-log?lang=nl) en [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_version():
    quick_version = L.choice(
        "quick-version",
        question="Een run in het experimentlogboek van een collega heeft codeversie `4f9a1c2e`, en je hebt een kopie van zijn code. Hoe zoek je uit of dat de code is die draaide?",
        options={
            "right": "Bereken de hash van je kopie op dezelfde manier, en vergelijk hem met `4f9a1c2e`",
            "date": "Kijk wanneer het bestand het laatst veranderde: de code van die dag is degene die draaide",
            "seed": "Draai hem met dezelfde seed: is de uitkomst hetzelfde, dan is de code dat ook",
        },
        correct="right",
        why={
            "date": "Een datum zegt wanneer een bestand veranderde, niet wat erin staat: een kopie van later, of een teruggedraaide wijziging, heeft een andere datum en dezelfde code. Zie [Een experimentlogboek bijhouden](/course/open/res-experiment-log?lang=nl).",
            "seed": "Dezelfde uitkomst betekent niet dezelfde code: een veranderd commentaar, of een wijziging waar deze run nooit komt, geeft dezelfde getallen met andere code. Zie [Een experimentlogboek bijhouden](/course/open/res-experiment-log?lang=nl).",
            "right": "Dezelfde tekst geeft dezelfde hash, en één veranderd teken een andere: begint de jouwe met `4f9a1c2e`, dan is het vrijwel zeker de code die draaide. Deze les doet hetzelfde voor elk bestand van een map.",
        },
    )
    quick_version
    return (quick_version,)


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="Je vond een dataset op een website die niets over een licentie zegt. Mag je er een kopie van in je openbare repository zetten?",
        options={
            "online": "Ja: hij staat al op internet",
            "right": "Nee: zonder licentie mag je hem niet doorgeven; link naar waar hij staat",
            "credit": "Ja, als je maar noemt waar hij vandaan komt",
        },
        correct="right",
        why={
            "online": "Wat op internet staat, is niet vrij om te kopiëren: zonder licentie blijven alle rechten bij wie hem maakte. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "credit": "De bron noemen is wat een licentie zoals CC BY vraagt; zonder licentie is er niets dat je toestaat hem überhaupt te kopiëren. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Anderen kunnen hem nog steeds bij de bron halen: je map zegt waar, en welke versie.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je verslag zegt wat je vond. Om het opnieuw te draaien, heeft iemand meer nodig: je code, de instellingen en de seeds, de versies van Python en PyTorch waarmee de runs draaiden, de data of de manier om die te maken, en een licentie die zegt dat het mag. Zonder dat heeft een lezer alleen jouw woord, en weet je over een jaar misschien zelf niet meer wat je draaide. Een verzoek per mail is geen manier om iets opnieuw te draaien: tegen die tijd is de code veranderd, of weg.

    **Een voorbeeld uit het dagelijks leven.** Een recept dat je doorgeeft: "een beetje bloem, bakken tot het klaar is" helpt niemand. De hoeveelheden, de oven op 180 graden voor 35 minuten, de grootte van de vorm, en dat ze het zelf ook mogen doorgeven: dan bakt iemand anders dezelfde taart.
    """)
    return


@app.cell(hide_code=True)
def predict_same_seed():
    guess_same_seed = L.guess(
        "same-seed",
        question="Iemand krijgt je kit en draait hem op zijn eigen computer, met je instellingen en seed 0. Krijgt die jouw accuracies?",
        options={
            "exact": "Precies dezelfde op elke computer, altijd: de seed legt elk willekeurig getal vast",
            "right": "Met jouw versies op één CPU-thread meestal dezelfde; elders kan de run een beetje anders uitvallen",
            "random": "Elke keer andere, ook op je eigen computer: trainen is willekeurig",
        },
        reveal=f"**Met jouw versies op één CPU-thread meestal dezelfde; elders een beetje anders.** De seed legt de splitsing, de beginwaarden van de weights en de batches vast. Maar in welke volgorde een som wordt opgeteld, hangt af van de CPU of GPU, het aantal threads en de versie van PyTorch, en dat kan het laatste bit van een getal veranderen. Zo'n piepklein verschil groeit tijdens het trainen. De course draaide seed 0 opnieuw met het laatste bit van één beginwaarde veranderd: validatie haalde 99% één gelogde stap later, bij stap {C.number(C.ONE_BIT['v'], 0, L.lang)} in plaats van {C.number(C.ONE_BIT['v_before'], 0, L.lang)}, en bij stap {C.number(C.ONE_BIT['v_before'], 0, L.lang)} stond hij op {round(100 * C.ONE_BIT['val_at'])}% in plaats van {C.number(100 * C.ONE_BIT['val_before'], 1, L.lang)}%. Daarom zegt de map welke versies de runs gebruikten, en draait `rerun.py` op één thread.",
    )
    guess_same_seed
    return (guess_same_seed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een open artefact** is één map met alles wat nodig is om je werk opnieuw te draaien, en een licentie die zegt dat anderen dat mogen. Voor je reproductie ziet hij er zo uit:

        ```
        share/
          repro/             je kit, met zijn instellingen
          llm/               de modules van je GPT die de kit importeert
          runs.jsonl         de logboeken van je runs: seed en accuracies
          ablation.jsonl     hetzelfde voor de ablation
          seeds.jsonl        hetzelfde voor de vergelijking over seeds
          report.md          je verslag, met zijn negatieve resultaten
          rerun.py           een script dat de belangrijkste figuur opnieuw draait
          requirements.txt   de versies van PyTorch en NumPy
          LICENSE            wat anderen met de code mogen
          README.md          wat het is, en hoe je het opnieuw draait
          SHA256SUMS         een checksum van elk bestand
        ```

        Wat niet van jou is om uit te delen, zoals de pdf van de paper of een dataset waarvan de licentie niet toestaat hem door te geven, krijgt een link en zijn versie, geen kopie. Geheimen, zoals een bestand met een login-token, en wat niemand nodig heeft om het te draaien, blijven erbuiten. Bewaarde een run een checkpoint, deel dat dan ook, onder een licentie zoals CC BY 4.0, die de course zijn eigen modellen geeft; een checkpoint dat te groot is voor de repository gaat naar een model hub zoals Hugging Face, met zijn link en checksum in de map.

        **Checksums.** Een **checksum** van een bestand is de hash van zijn inhoud, zoals de codeversie in je experimentlogboek de hash van je code was. Om je eigen versies uit elkaar te houden, waren 8 hextekens genoeg; om te laten zien dat een bestand precies het bestand is dat je deelde, zijn alle 64 van zijn SHA-256 nodig. Het programma `sha256sum` schrijft één regel per bestand, de hash, twee spaties en de naam, en `sha256sum -c SHA256SUMS` controleert elk bestand tegen zijn regel.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **open artefact** | alles wat nodig is om een stuk werk opnieuw te draaien, gedeeld met een licentie: code, instellingen, seeds, versies, data of hoe je die maakt, resultaten | de map `share/` |
        | **checksum** | de hash van de inhoud van een bestand; wie dezelfde checksum heeft, heeft hetzelfde bestand | `sha256(...).hexdigest()` |
        """),
        L.watch_out("Zoek naar geheimen voordat je deelt: een login-token of wachtwoord in een bestand of notebook is openbaar zodra je het naar een openbare repository pusht, en het blijft in de geschiedenis van Git staan, ook nadat je het bestand verwijdert. Lekte er een uit, trek het dan in en maak een nieuw aan."),
        mo.md(r"""
        ### Recept

        Zo deel je je werk zodat anderen het opnieuw kunnen draaien:

        1. Zet de code, de instellingen, de seeds, de logboeken en het verslag in één map.
        2. Schrijf de versies op waarmee het draaide, en voeg een script toe dat het opnieuw draait.
        3. Laat geheimen weg en wat niemand nodig heeft; geef wat niet van jou is een link en zijn versie.
        4. Voeg een licentie toe en een README die zegt hoe je het draait.
        5. Voeg de checksums van elk bestand toe, en draai het script één keer vanuit alleen de map.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De checksumregels van twee kleine bestanden met de hand: stap 5 van het recept. Voer de cel uit, verander dan één teken van een bestand en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_sums():
    import hashlib

    _files = {"settings.py": "SEED = 0\n", "LICENSE": "MIT License\n"}   # de naam van elk bestand, en zijn tekst
    for _name in sorted(_files):   # 5. gesorteerd, zodat de lijst voor iedereen hetzelfde is
        _hash = hashlib.sha256(_files[_name].encode()).hexdigest()
        print(_hash + "  " + _name)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `LICENSE` komt vóór `settings.py`: gesorteerd komen hoofdletters vóór kleine letters. Elke hash is 64 hextekens, en één veranderd teken geeft een heel andere, dus laat de lijst meteen zien welk bestand niet het bestand is dat je deelde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_checksums():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De checksums

        Maak `checksums(files)` af: voor elke naam in `sorted(files)` een regel met de SHA-256 van de tekst van het bestand als 64 hextekens, twee spaties, en zijn naam. `files` is een dictionary die elke naam aan zijn tekst koppelt. De demo hieronder gebruikt je `checksums` voor het bestand `SHA256SUMS` van je map zodra hij zijn check haalt.
        """),
        L.task("Maak `checksums` af: één regel per bestand, gesorteerd, met de hash van zijn tekst, twee spaties en zijn naam."),
    ])
    return


@app.cell(hide_code=True)
def exercise_checksums():
    box_checksums = L.code_box("checksums", lines=12)
    box_checksums
    return (box_checksums,)


@app.cell(hide_code=True)
def check_checksums(box_checksums):
    result_checksums = L.check("checksums", box_checksums.value)
    result_checksums
    return (result_checksums,)


@app.cell(hide_code=True)
def hints_checksums():
    L.hints(
        "checksums",
        [
            "Volg het uitgewerkte voorbeeld, met `files[name]` als de tekst en `lines.append(...)` in plaats van `print`.",
        ],
        why="De hash van de tekst, niet van de naam, verandert als het bestand verandert; en gesorteerd is de lijst zelf hetzelfde voor iedereen met dezelfde bestanden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_what_goes():
    mo.md(r"""
    ### Oefening 2. Wat komt er in de map?

    Zet elk item in een groep: in de map, een link ernaartoe in plaats van het bestand, of het blijft erbuiten.
    """)
    return


@app.cell(hide_code=True)
def exercise_what_goes():
    sort_what_goes = L.sort("what-goes")
    sort_what_goes
    return (sort_what_goes,)


@app.cell(hide_code=True)
def check_what_goes(sort_what_goes):
    result_what_goes = L.check("what-goes", sort_what_goes)
    result_what_goes
    return (result_what_goes,)


@app.cell(hide_code=True)
def text_exercise_license():
    mo.md(r"""
    ### Oefening 3. Een licentie voor de code
    """)
    return


@app.cell(hide_code=True)
def choice_license_cell():
    choice_license = L.choice(
        "license",
        question="Met welke licentie mag iedereen de code van je kit draaien, veranderen en delen, zolang je naam erop blijft?",
        options={
            "none": "Geen: de repository is openbaar, dus iedereen mag hem gebruiken",
            "nc": "CC BY-NC 4.0, zodat geen bedrijf er geld mee verdient",
            "right": "De MIT-licentie",
        },
        correct="right",
        why={
            "none": "Zonder licentie blijven alle rechten bij jou: anderen mogen je code lezen, maar niet kopiëren, veranderen of delen.",
            "nc": "Creative Commons raadt zelf af om zijn licenties voor code te gebruiken: ze zijn gemaakt voor tekst, beelden en data. En NC houdt meer tegen dan bedrijven: een school of een lab dat geld krijgt, weet misschien niet zeker of het mag.",
            "right": "Met MIT mag iedereen de code gebruiken, veranderen en delen, ook in eigen werk, zolang de licentie met je naam meegaat. Voor het verslag en de logboeken doet CC BY 4.0 hetzelfde voor tekst en data.",
        },
    )
    choice_license
    return (choice_license,)


@app.cell(hide_code=True)
def text_demo():
    mo.vstack([
        mo.md(r"""
        ### Je map

        De les voegt je reproductie samen in `work/projects/repro/share/`: elke module van je kit, je logboeken en je verslag, een script dat de belangrijkste figuur opnieuw draait, de versies van deze computer, de MIT-licentie met je naam, een README, en de checksums, gemaakt met je `checksums` zodra hij zijn check haalt. Ontbreekt een module of een logboek van jou, dan gaat die van de course erin, zoals in je verslag, en de README zegt dat; de vergelijking over seeds gebruikte altijd de runs van de course. De map is in het Engels, zoals je verslag. Schrijf de naam die in de licentie komt, en druk op de knop. Hij schrijft de map elke keer opnieuw; er gaat niets van je computer af.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def share_inputs():
    name_box = mo.ui.text(value="", label="Je naam, voor de licentie", full_width=True)
    build_button = mo.ui.run_button(label="Voeg de map samen")
    mo.vstack([name_box, build_button])
    return build_button, name_box


@app.cell(hide_code=True)
def demo_share(build_button, name_box, box_checksums, result_checksums):
    _out = mo.md("*Druk op de knop om de map samen te voegen.*")
    if build_button.value:
        _mine = C.checksums_of(box_checksums.value) if result_checksums.state == "correct" else None
        try:
            _r = C.build(name_box.value, _mine)
        except OSError:
            _r = None
        if _r is None:
            _out = mo.md("*De map kon niet geschreven worden: bestaat `work/projects/repro/`, en mag de course erin schrijven?*")
        else:
            _notes = [f"*De map staat in `{_r['where']}`, met {len(_r['files'])} bestanden.*",
                      "*De checksums zijn gemaakt met je `checksums`.*" if _r["mine"] else "*De checksums zijn gemaakt met de `checksums` van de course: de jouwe neemt het over zodra hij zijn check haalt.*"]
            if _r["missing"]:
                _notes.append("*Nog niet gemaakt, dus niet in de map: " + ", ".join(f"`{_m}`" for _m in _r["missing"]) + ". Maak het in de les hiervoor, en voeg de map dan opnieuw samen.*")
            _notes.append("*De runs in " + ", ".join(f"`{_m}`" for _m in _r["course_logs"]) + " zijn die van de course; de README zegt dat.*")
            if _r["course"]:
                _notes.append("*De versies van de course van " + ", ".join(f"`{_m}`" for _m in _r["course"]) + " zitten erin, omdat je er geen eigen versie van hebt; de README zegt dat.*")
            _sums = "\n".join(_r["files"]["SHA256SUMS"].splitlines()[:4]) + "\n…"
            _out = mo.vstack([
                mo.md(" ".join(_notes)),
                mo.Html(f"<div style='border: 1px solid var(--ea-line); border-radius: 8px; padding: 0.2rem 1rem; overflow-x: auto'>{mo.md(_r['files']['README.md'].replace(chr(10) + '## ', chr(10) + '#### ').replace('# ', '### ', 1)).text}</div>"),
                mo.md(f"De eerste regels van `SHA256SUMS`:\n\n```text\n{_sums}\n```"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Om hem te publiceren, maak je van de map een Git-repository en push je hem naar GitHub, of upload je hem naar Zenodo, dat hem een DOI geeft: een vaste link die een verslag of paper kan citeren, ook als de repository verhuist. Open daarvoor elk bestand één keer: staat er niets in dat je niet openbaar wilt hebben?
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De codeversie uit [Een experimentlogboek bijhouden](/course/open/res-experiment-log?lang=nl), nu als checksum van elk bestand; de licentie uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl), nu op je eigen werk; en het verslag uit [Begeleide reproductie 7](/course/open/repro-guided-report?lang=nl), nu met alles wat nodig is om het te controleren. In de volgende les, *Je reproductie 1: kies een paper*, begin je aan een eigen reproductie, met dezelfde stappen en minder hulp, en aan het eind een map zoals deze.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_checksums, result_what_goes, choice_license):
    L.wrap_up(result_checksums, result_what_goes, choice_license)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
