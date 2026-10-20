# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les repro-own-report. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("repro-own-report", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Je reproductie 7: vergelijk met de paper](/course/open/repro-own-compare?lang=nl) en [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_failed():
    quick_failed = L.choice(
        "quick-failed",
        question="Je methode verlaagt de fout voor elke seed, met 2 tot 3 procentpunt, en de winst van de paper is 6 procentpunt, hoger dan elk ervan. Een vriend zegt: dan is je reproductie mislukt. Wat zeg je?",
        options={
            "failed": "Dat dat klopt: een reproductie hoort het getal van de paper te geven, binnen zijn interval",
            "right": "Dat de claim hield, een lagere fout; de grootte verschilt, en dat is een bevinding om te verklaren",
            "paper": "Dat het getal van de paper wel fout moet zijn, want al je seeds komen met elkaar overeen",
        },
        correct="right",
        why={
            "failed": "Een reproductie test de claim, hier een duidelijk lagere testfout, en die hield. Een grootte die verschilt in een opzet die verschilt, is een bevinding, met zijn redenen. Zie [Je reproductie 7](/course/open/repro-own-compare?lang=nl).",
            "paper": "Dat je seeds overeenkomen, zegt hoe zeker je eigen getal is, niet dat dat van de paper fout is: zijn opzet verschilt van de jouwe. Kijk naar wat verschilt, zoals in [Je reproductie 7](/course/open/repro-own-compare?lang=nl).",
            "right": "Je verslag zegt allebei: wat hield, en wat verschilt, met de redenen die je kunt noemen.",
        },
    )
    quick_failed
    return (quick_failed,)


@app.cell(hide_code=True)
def quick_check_hash():
    quick_hash = L.choice(
        "quick-hash",
        question="Een lezer downloadt je `runs.jsonl`, en zijn SHA-256 is niet die in je manifest. Wat zegt hem dat?",
        options={
            "right": "Dat zijn bestand niet het jouwe is: minstens één teken verschilt",
            "older": "Dat zijn kopie ouder is dan de jouwe, minstens een dag",
            "close": "Dat de resultaten verschillen, maar weinig, want twee hashes van bijna hetzelfde bestand liggen dicht bij elkaar",
        },
        correct="right",
        why={
            "older": "Een hash zegt niets over tijd: hij verandert met elke verandering van de tekst, wanneer die ook gemaakt is. Zie [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl).",
            "close": "Twee teksten die in één teken verschillen, geven hashes die totaal niet op elkaar lijken: een hash zegt hetzelfde of niet, niet hoe dichtbij. Zie [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl).",
            "right": "Dezelfde tekst geeft dezelfde SHA-256; elke andere tekst een andere.",
        },
    )
    quick_hash
    return (quick_hash,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een reproductie die niemand kan lezen, had net zo goed niet kunnen gebeuren. Je verslag zegt wat je testte, hoe, wat hield en wat verschilde, en wat je niet controleerde, zodat een lezer het kan wegen zonder het jou te vragen. Publiceer je je code, je plan en je logboeken, dan kan een lezer alles opnieuw draaien en je getallen terugvinden. En de auteurs van de paper zijn de lezers die er het meest van kunnen leren: vertel ze, kort en vriendelijk, wat je vond.

        **Een voorbeeld uit het dagelijks leven.** Je probeerde een recept voor een kookclub, met ander meel en een kleinere oven. Je schrijft op wat je veranderde, hoe het uitpakte en wat je niet probeerde, voegt de foto's toe, en stuurt een briefje aan de schrijver van het recept.
        """),
        draw.project_steps(["Kies een paper", "Claim en plan", "Bouw het", "Sanity checks", "Het hoofdresultaat", "Ablation", "Vergelijk", "Verslag en delen"],
                           7, here_label="← vandaag", title="Waar je project staat: je eigen reproductie"),
    ])
    return


@app.cell(hide_code=True)
def predict_sentence():
    guess_sentence = L.guess(
        "sentence",
        question="De reproductie van de course van de dropout-paper steunt de claim volgens zijn regel. Welke zin hoort in de Results van zijn verslag?",
        options={
            "general": "Dropout verlaagt de testfout van neural networks, zoals de paper zegt, dus de claim geldt in het algemeen",
            "right": "Met dropout is de testfout bij 12.800 stappen lager voor alle drie de seeds, met 0,7 procentpunt in de mediaan",
            "works": "Dropout werkt duidelijk: de reproductie bevestigt de paper zonder enige twijfel, met een grotere winst dan die van de paper",
        },
        reveal="**Met dropout is de testfout bij 12.800 stappen lager voor alle drie de seeds, met 0,7 procentpunt in de mediaan.** Hij zegt wat gemeten is, bij welk budget en over welke seeds, en niets meer: één netwerk, op 6000 afbeeldingen van Fashion-MNIST. \"In het algemeen\" en \"zonder enige twijfel\" claimen wat drie seeds van één opzet niet kunnen laten zien.",
    )
    guess_sentence
    return (guess_sentence,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Het verslag van je kit.** De `report` van Begeleide reproductie 7 zet vijf delen samen: *Claim*, *Setup*, *Results*, *Differences* en *Not checked*. Vier ervan schrijf je zelf; de Results komen uit je kit: de regel van je plan en zijn oordeel bij je budget, de runs van elke seed in een tabel, het aandeel dat je ablation houdt, en je vergelijking met de paper. Het verslag is in het Engels, de taal van de paper en van zijn auteurs.

    **Getallen met hun precisie.** Een testfout over 1000 afbeeldingen beweegt in stappen van 0,1 procentpunt: één decimaal zegt alles wat hij weet. Een winst die een verschil van twee floats is, zoals `12.9 - 12.5`, komt uit op `0.40000000000000036`; afgerond is het 0,4. Je tabel rondt elk getal af op de decimalen die de metric heeft.

    **Wat openbaar wordt, en zijn manifest.** Je code, je plan met zijn datum, je budget, je logboeken en je verslag worden openbaar; een token of een wachtwoord nooit. Een manifest geeft de SHA-256 van elk bestand, zoals de `SHA256SUMS` van [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl) deed, zodat een lezer kan controleren dat zijn bestanden de jouwe zijn.

    **Vertel het de auteurs.** Een kort bericht: wat je reproduceerde en hoe, wat hield, wat verschilde en wat het volgens jou verklaart, en waar je code en verslag staan. Vriendelijk en precies: zij maakten de paper waarvan je leerde.

    ### Nieuwe woorden en tekens

    | Woord | Wat het betekent |
    |---|---|
    | manifest | een lijst van de bestanden die openbaar worden, elk met zijn SHA-256 |

    Hieronder staat het verslag dat de course zelf schreef over zijn reproductie van de dropout-paper, met zijn vijf delen als tabs, de Results uit zijn runs. Klik de delen door.
    """)
    return


@app.cell(hide_code=True)
def course_report_cell():
    _parts = {**C.COURSE_PARTS, "Results": C.course_results_text()}
    mo.vstack([
        mo.md(f"**{C.COURSE_PARTS['title']}**"),
        mo.ui.tabs({_name: mo.md(_parts[_name]) for _name in C.SECTIONS}),
    ])
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        Elk getal van de Results is terug te vinden in `runs.jsonl` en de andere bestanden van de kit. De Differences zeggen wat er met opzet veranderd is, en waar de getallen van de paper van zijn: twee netwerken, zonder een spreiding over runs. Not checked noemt wat een lezer getest zou kunnen denken en dat niet is: MNIST zelf, grotere netwerken, meer seeds.
        """),
        L.watch_out("Een verslag zegt wat de runs lieten zien, bij het budget en over de seeds die je draaide, en niets meer. Laat je een run weg omdat hij niet paste, dan gaat het verslag niet meer over je runs. Een resultaat dat niet hield, komt er net zo goed in als een dat wel hield: het is voor de volgende lezer net zo nuttig."),
        mo.md(r"""
        ### Recept

        Zo maak je je verslag en deel je je reproductie:

        1. Schrijf de Claim zoals de paper hem maakt, met waar hij staat; de Setup zo dat een lezer hem kan nabouwen.
        2. Laat je kit de Results schrijven: de regel en zijn oordeel, de runs als tabel met hun precisie, de ablation, de vergelijking.
        3. Schrijf de Differences, met opzet en niet, en wat je niet controleerde (Not checked).
        4. Publiceer je code, plan, budget, logboeken en verslag, met een manifest van hun SHA-256, en wat Je werk delen zodat anderen het kunnen herhalen erbij deed: de versies waarmee je draaide, waar de data vandaan komen, een README en een licentie. Geen tokens of wachtwoorden.
        5. Vertel de auteurs wat je vond, vriendelijk en precies, met de link.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén winst van het hoofdresultaat van de course, zoals Python hem uitrekent en zoals een verslag hem laat zien, in een tabel van één rij. Voer de cel uit.
    """)
    return


@app.cell
def example_round():
    _gain = 13.6 - 12.4          # zonder dropout min met, seed 2
    print("uitgerekend:", _gain)
    print("afgerond:", round(_gain, 1))
    _table = "| seed | gain |\n|---|---|\n| 2 | " + str(round(_gain, 1)) + " |"
    print(_table)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `13.6 - 12.4` is in een float niet precies 1,2: de laatste decimalen zijn ruis van de computer, niet van de runs. Afgerond op de ene decimaal die de testfout heeft, is het 1,2. De tweede regel van de tabel, `|---|---|`, vertelt Markdown dat de regel erboven de namen van de kolommen heeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_table():
    mo.vstack([
        mo.md(r"""
        ## Jouw beurt

        ### Oefening 1. De tabel van je verslag

        Maak `markdown_table(head, rows, digits)` af: een tabel in Markdown, met een regel met namen, een regel met `---` voor elke kolom, en een regel voor elke rij, elk getal afgerond op `digits` decimalen. `"---|" * 3` herhaalt een tekst drie keer: `"---|---|---|"`. De Results van je verslag gebruiken hem voor de runs van je seeds, met de decimalen die je runs hebben. De check probeert twee runs van de course, een tabel met drie rijen en twee decimalen, een tabel zonder rijen, en een met drie decimalen.
        """),
        L.task("Maak `markdown_table` af: de eerste twee regels, dan een regel voor elke rij, afgerond."),
    ])
    return


@app.cell(hide_code=True)
def exercise_table():
    box_table = L.code_box("table", lines=10)
    box_table
    return (box_table,)


@app.cell(hide_code=True)
def check_table(box_table):
    result_table = L.check("table", box_table.value)
    result_table
    return (result_table,)


@app.cell(hide_code=True)
def hints_table():
    L.hints(
        "table",
        [
            "Een regel is `\"| \" + \" | \".join(cells) + \" |\"`, met de cellen als tekst; de tweede regel is `\"|\" + \"---|\"` één keer voor elke kolom.",
            "Voor elke rij: rond elk getal af op `digits` decimalen en maak er tekst van met `str`, als zijn cellen; voeg aan het eind de regels samen met `\"\\n\"`.",
        ],
        why="De regel met `---` laat Markdown een tabel tekenen; afronden laat elk getal zien met de precisie die zijn metric heeft, zonder de ruis van floats.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je verslag

        Schrijf de vier delen van je verslag die je kit niet voor je kan schrijven, in het Engels, en zijn titel. De Claim zoals de paper hem maakt, met waar hij staat; de Setup zo dat een lezer hem kan nabouwen; de Differences, met opzet en niet; en wat je niet controleerde (Not checked). De check zet ze samen met de Results uit je kit, met de `report` van je kit, en bewaart het verslag als `work/projects/own/REPORT.md`.
        """),
        L.task("Schrijf de titel en de vier delen van je verslag."),
    ])
    return


@app.cell(hide_code=True)
def report_inputs():
    _old = C.saved_parts() or {}
    _start = C.start_parts()   # the claim and the setup of your plan, in English
    report_title = mo.ui.text(label="Titel", full_width=True, value=_old.get("title", ""))
    report_claim = mo.ui.text_area(label="Claim", full_width=True, rows=4, value=_old.get("Claim") or _start.get("Claim", ""))
    report_setup = mo.ui.text_area(label="Setup", full_width=True, rows=5, value=_old.get("Setup") or _start.get("Setup", ""))
    report_differences = mo.ui.text_area(label="Differences", full_width=True, rows=4, value=_old.get("Differences", ""))
    report_not_checked = mo.ui.text_area(label="Not checked", full_width=True, rows=3, value=_old.get("Not checked", ""))
    mo.vstack([report_title, report_claim, report_setup, report_differences, report_not_checked])
    return report_claim, report_differences, report_not_checked, report_setup, report_title


@app.cell(hide_code=True)
def check_report(report_claim, report_differences, report_not_checked, report_setup, report_title):
    result_report = L.check("report", {"title": report_title.value, "Claim": report_claim.value, "Setup": report_setup.value,
                                       "Differences": report_differences.value, "Not checked": report_not_checked.value})
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Begin de Claim en de Setup bij je plan van Je reproductie 2: zijn claim, waar de paper hem laat zien, en wat in beide settings hetzelfde blijft.",
            "Kijk voor de Differences naar Je reproductie 7: wat je met opzet veranderde, waar de getallen van de paper van zijn, en wat volgens jou het verschil verklaart.",
        ],
        why="De eigen delen van de course staan in de tabs hierboven, onder *Het idee*: een Claim met waar hij staat, een Setup die een lezer kan nabouwen, Differences met opzet en niet, en wat niet gecontroleerd is.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De knop schrijft je verslag opnieuw, met de Results die je kit nu heeft, en daarna het manifest van de bestanden die openbaar worden, `work/projects/own/MANIFEST.txt`. Zet de map `work/projects/own/` in een openbare repository met je licentie, zoals in Je werk delen zodat anderen het kunnen herhalen, en het manifest erbij.
    """)
    return


@app.cell(hide_code=True)
def button_share():
    share_button = mo.ui.run_button(label="Schrijf mijn verslag en zijn manifest")
    share_button
    return (share_button,)


@app.cell(hide_code=True)
def demo_share(share_button, result_table, result_report):
    result_table, result_report  # opnieuw uitvoeren als een oefening een deel van de kit bewaart
    if not share_button.value:
        _out = mo.md("*Druk op de knop om je verslag en zijn manifest te schrijven.*")
    else:
        _lines = C.manifest()
        if _lines is None:
            _out = mo.md("*Schrijf eerst je verslag in oefening 2: dan kan de knop het samenzetten.*")
        elif _lines == "report":
            _out = mo.md("*De `report` van je kit stopt met een fout, `work/projects/repro/report.py` van [Begeleide reproductie 7](/course/open/repro-guided-report?lang=nl). Check hem daar opnieuw.*")
        else:
            _title, _parts = C.report_sections(C.report_path().read_text(encoding="utf-8"))
            _out = mo.vstack([
                mo.md("**Je verslag, `work/projects/own/REPORT.md`:**"),
                mo.md(f"**{_title}**"),
                mo.ui.tabs({_name: mo.md(_body) for _name, _body in _parts.items()}),
                mo.md("**Zijn manifest, `work/projects/own/MANIFEST.txt`:**"),
                mo.md("| bestand | SHA-256, de eerste 16 van 64 hextekens |\n|---|---|\n" + "\n".join(f"| `{_l.split()[1]}` | `{_l.split()[0][:16]}…` |" for _l in _lines) + "\n\nHet bestand heeft ze alle 64."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_authors():
    mo.md(r"""
    ### Oefening 3. Vertel het de auteurs

    Je verslag en je code zijn openbaar. Schrijf het bericht aan de auteurs van de paper.
    """)
    return


@app.cell(hide_code=True)
def explain_to_authors():
    explain_to_authors_cell = L.explain(
        "to-authors",
        question="Schrijf het bericht aan de auteurs van je paper: wat je reproduceerde en hoe, wat hield, wat verschilde en waarom je dat denkt, en waar je verslag en code staan. In het Engels, zoals je verslag.",
        model_answer="Dear Dr. Srivastava and coauthors, I reproduced the claim of your dropout paper as part of a course: a smaller network (784-512-512-10, ReLU) on 6000 pictures of Fashion-MNIST, with dropout 0.2 at the input and 0.5 in the hidden layers, three seeds, 12,800 steps. With dropout, the test error was lower for all three seeds, by 0.7 points in the median, so the claim held. The gain is larger in points than your 1.60% to 1.35% on MNIST and smaller as a share of the error; I think the harder data and the smaller training set explain that. An ablation without the dropout at the input kept none of the gain. My report, code and logs: <link>. Thank you for a paper that was a pleasure to reproduce.",
        points=[
            "je zegt wat je reproduceerde, waarop, en met hoeveel seeds",
            "je zegt wat hield en wat verschilde, met je gok waarom",
            "je geeft de link naar je verslag en code",
            "je toon is vriendelijk en precies",
        ],
    )
    explain_to_authors_cell
    return (explain_to_authors_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het verslag van [Begeleide reproductie 7](/course/open/repro-guided-report?lang=nl) en het delen van [Je werk delen zodat anderen het kunnen herhalen](/course/open/res-share-work?lang=nl), nu voor je eigen paper; en elk deel van je kit, van het plan van [Je reproductie 2](/course/open/repro-own-claim?lang=nl) tot de vergelijking van [Je reproductie 7](/course/open/repro-own-compare?lang=nl). Je eigen reproductie is klaar. De volgende les, *Waar nu heen*, laat zien waar de andere takken van de course aansluiten op wat je nu weet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_table, result_report, explain_to_authors_cell):
    L.wrap_up(result_table, result_report, explain_to_authors_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
