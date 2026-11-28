# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-project-tested-model. Route: course/route/lessen/systemen.yaml. Checks, het model van de course en een versie zonder weekday, de demo die pytest uitvoert, en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("sys-project-tested-model", "nl")
    C = L.checks   # het model van de course, de checks van je onderdelen, de run van elke test, de card en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl) en [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Welke van deze tests is een directional expectation test van je model van de vraag?",
        options={
            "right": "Met regen in plaats van helder weer in dezelfde uren moet het model minder fietsen verwachten",
            "night": "Van 1 tot 4 uur 's nachts op een werkdag moet het model in elk uur minder dan 50 fietsen verwachten",
            "row": "Het rijnummer van elk uur een jaar opschuiven mag niet veranderen wat het model verwacht",
            "unit": "`inputs` moet uur 8 van een rij in kolom 8 van het one-hot-deel zetten, en in geen andere kolom",
        },
        correct="right",
        why={
            "right": "Een directional expectation test verandert iets wat uitmaakt, het weer, en vraagt dat het antwoord de bekende kant op beweegt: met regen minder fietsen. Je `test_rain` doet dat met het gemiddelde over 5.645 heldere uren.",
            "night": "Weinig fietsen 's nachts is een eenvoudig geval met een bekend antwoord, dat elk bruikbaar model goed moet doen: welke soort test vraagt dat een antwoord de bekende kant op beweegt? Zie [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl).",
            "row": "Een verandering die niet mag uitmaken, met de vraag om hetzelfde antwoord, is een invariance test: welke soort verandert iets wat wel uitmaakt, en vraagt dat het antwoord beweegt? Zie [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl).",
            "unit": "Dat test één functie van binnenuit, met één rij met een bekend antwoord: een unit test van de code, geen test van het model van buitenaf. Welke soort vraagt dat het antwoord van het model de bekende kant op beweegt? Zie [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="De cijferlezer leest 94,5% van 2.000 testcijfers goed. Wat hoort de model card naast dat ene getal te geven?",
        options={
            "decimals": "Dezelfde 94,5% met twee decimalen meer, zodat twee versies uit elkaar te houden zijn",
            "history": "De 94,5% van elke eerdere versie, zodat een lezer ziet hoe het model beter werd",
            "right": "Het deel goed per groep en per omstandigheid, elk met het aantal cijfers erachter",
            "train": "Het deel goed op de trainingscijfers ook, zodat een lezer ziet dat het model niet overfit",
        },
        correct="right",
        why={
            "right": "Een gemiddelde over 2.000 cijfers kan een groep verbergen: de cijfers van de scholieren gingen bijna twee keer zo vaak fout als de cijfers van de medewerkers. Met het aantal cijfers achter elk deel ziet een lezer ook welke getallen te klein zijn om veel te zeggen.",
            "decimals": "Meer decimalen maken het ene getal exacter, maar het getal verbergt nog steeds wie het model fout leest: wat voegt een card toe zodat een lezer dat ziet? Zie [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl).",
            "history": "Eerdere versies zeggen hoe het model zo ver kwam, niet voor wie het model nu werkt: wat voegt een card toe aan één getal over alle cijfers? Zie [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl).",
            "train": "De trainingscijfers zeggen hoe goed het model deze cijfers leerde, niet hoe het model het doet voor elke groep nieuwe cijfers: wat voegt een card toe aan één getal over alle testcijfers? Zie [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl).",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je model van de vraag is verpakt, en heeft vijf tests uit twee lessen: twee van de code en drie van het gedrag. Een team voert de tests niet een voor een uit. Het team voert alle tests uit met één opdracht, bij elke nieuwe versie, en wil één antwoord: groen als elke test deed wat de test moest doen, rood met de naam van de test die dat niet deed. Sommige fouten zijn bekend en vandaag niet op te lossen, zoals zaterdag tegenover zondag uit [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl), en zulke fouten mogen niet elke run rood maken.

        Wie het model gebruikt, heeft in woorden nodig wat de getallen en de tests vonden: waarvoor het model is en waarvoor niet, hoe ver het model ernaast zit in totaal en voor welke uren, en de bekende grenzen. Dat is een model card, zoals in [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl). Met de hand geschreven is een card verouderd zodra de volgende versie verpakt is. Dus schrijft code in deze stap van je project de card, uit het pakket en de getallen, en voert één opdracht elke test uit.

        **Een voorbeeld uit het dagelijks leven.** Een auto krijgt elk jaar een APK: één bezoek controleert de remmen, de lampen en de banden, allemaal op één lijst. Een bekend mankement dat voorlopig mag blijven, zoals een deuk, komt als opmerking op het rapport, niet als afkeuring. En het rapport gaat met de auto mee naar de volgende eigenaar.
        """),
        draw.project_steps(["Een model van de vraag, bijgehouden en ingepakt", "Tests en een model card", "Je model als service, bewaakt op drift"], 1,
                           here_label="← vandaag", title="Waar je project staat: ops"),
    ])
    return


@app.cell(hide_code=True)
def predict_days():
    guess_days = L.guess(
        "days",
        question="Over alle 1.733 uren van de validatiedagen zit je model 53,04 fietsen per uur ernaast. 1.119 van deze uren vallen op werkdagen, en 614 in weekenden en op feestdagen, met ongeveer evenveel fietsen per uur. Splits de RMSE in deze twee groepen. Wat komt eruit, denk je?",
        options={
            "same": "Ongeveer 53 voor allebei, omdat het model de soort dag als een van de inputs krijgt",
            "work": "Duidelijk hoger op werkdagen, omdat de spits daar de meeste fietsen van allemaal heeft",
            "right": "Duidelijk hoger in weekenden en op feestdagen, met ruim 10 fietsen per uur verschil",
            "count": "Hoger op werkdagen, want op bijna twee keer zoveel uren kan het model ernaast zitten",
        },
        reveal="**Hoger in weekenden en op feestdagen.** 60,9 fietsen per uur op de 614 uren van deze dagen, tegenover 48,2 op de 1.119 uren van de werkdagen, terwijl beide groepen ongeveer evenveel fietsen per uur hebben, 144,9 en 142,0. Het verst ernaast zitten de 5 feestdagen onder de validatiedagen: 78,7 op de 119 uren van deze dagen. Ook de trainingsdagen hadden maar 5 feestdagen, dus leerde het model feestdagen van 5 dagen. De ene RMSE van 53,04 verbergt dit allemaal. Oefening 1 meet het per groep.",
    )
    guess_days
    return (guess_days,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Elke test, één opdracht.** pytest zoekt in een map naar bestanden met een naam die begint met `test_`, en daarin naar functies met een naam die begint met `test`. pytest leest eerst `conftest.py`, voor fixtures zoals `expect`, voert elke gevonden test uit, en eindigt met één regel die de tests telt, zoals `5 passed in 1.35s`. In je project vindt pytest in de map `ops` vijf tests in vijf bestanden: `test_one_row` en `test_overfit_one_batch` uit [Tests voor ML-code](/course/open/sys-code-tests?lang=nl), en `test_quiet_night`, `test_row_number` en `test_rain` uit [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl). Alle tests van een project samen zijn de suite. De suite is groen als elke test doet wat de test moet doen, en rood zodra één test dat niet doet.

        **Een bekende fout.** De invariance test van het weekend faalt voor je model, in 333 van de 2.495 weekenduren, omdat de dag van de week als één getal binnenkomt. Dat oplossen vraagt een nieuwe versie met andere inputs. Wat doe je tot dan met de test? Verwijder de test, en niets vertelt je wanneer een versie het goed doet, of op een nieuwe manier fout. Laat de test falen, en elke run is rood, dus lijkt een nieuwe fout precies op de oude. pytest heeft een markering voor dit geval: `@pytest.mark.xfail(reason="...")` op de regel boven `def`, zoals `@pytest.fixture` boven `expect` staat in `conftest.py`. De markering zegt dat de test hoort te falen, en waarom. pytest voert de test nog steeds uit. Faalt de test, dan meldt pytest **XFAIL**, een `x` in plaats van een punt, en blijft de suite groen: `5 passed, 1 xfailed`.

        **Strict.** Op een dag kan een test met xfail slagen: een nieuwe versie heeft de grens niet meer. pytest meldt dan **XPASS**, en uit zichzelf houdt dat de suite groen, dus valt het niemand op, en houdt de card een grens die weg is. Met `strict=True` in de markering telt pytest XPASS als fout, `[XPASS(strict)]`, en wordt de suite rood: tijd om de markering weg te halen, en de regel op de card.

        **Getallen per groep.** In [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl) gaf de card de accuracy van de cijferlezer per groep en per omstandigheid, met het aantal cijfers achter elk deel. Voor je model is de metric de RMSE, en de groepen zijn uren: per soort dag, werkdag of niet, en per weer. Het plaatje splitst de validation RMSE van het model van de course zo.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        De lijn van streepjes is de RMSE van alle 1.733 uren samen. Weekenden en feestdagen, helder weer en lichte regen liggen erboven; werkdagen en mist liggen eronder. De uren onder elke balk zeggen hoeveel vertrouwen elk getal verdient: 182 uren met lichte regen zeggen minder dan 1.119 uren op werkdagen.

        **Een card die code maakt.** Met de hand geschreven klopt een card voor één versie. Je project verpakt een nieuwe versie zodra het model opnieuw getraind wordt, met andere getallen, en een card die niemand herschrijft geeft dan de getallen van een ouder model. Dus schrijft code de card, uit `config.json` van het pakket, de RMSE per groep en de bekende grenzen, elke keer als de tests draaien. De card komt naast het pakket als `README.md`: op de Hugging Face Hub is het bestand `README.md` van een model de model card. De card is in Markdown, platte tekst met een paar tekens: `#` voor een kop, `|` tussen de cellen van een tabel, en `-` voor elk punt van een lijst. En de card is in het Engels, in beide talen van de course, zoals de meeste model cards: wie het model gebruikt, leest misschien geen Nederlands.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **suite** | swiet | alle tests van een project samen, uitgevoerd met één opdracht | `pytest ops` |
        | **xfail** | eks-feel | een markering die zegt dat een test hoort te falen: pytest voert de test toch uit, en meldt XFAIL in plaats van een fout | `@pytest.mark.xfail(reason="...")` |
        | **XPASS** | eks-pas | wat pytest meldt als een test met xfail slaagt | |
        | `strict=True` | strikt | in een xfail-markering: een test die slaagt telt als fout, zodat de suite rood wordt als een bekende grens weg is | `@pytest.mark.xfail(..., strict=True)` |
        | **README.md** | riedmie | het tekstbestand dat zegt wat er in een map staat; op de Hugging Face Hub de model card van een model | `ops/model/README.md` |
        | **Markdown** | | platte tekst met een paar tekens voor koppen, tabellen en lijsten; een site laat deze tekst zien als opgemaakte pagina | `## Known limits` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Tests en een card voor een model in een project:

    1. Houd elke test van het model in één map, in bestanden met een naam die begint met `test_`, en voer alle tests uit met één opdracht.
    2. Meet de metric per groep en per omstandigheid, met het aantal gevallen achter elk getal.
    3. Haalt het model een test niet, en los je dat nu niet op, geef de test dan een xfail-markering met een reden en `strict=True`, en schrijf dezelfde grens op de card.
    4. Laat code de card schrijven, uit het pakket en de getallen, zodat de card met elke versie meeverandert.
    5. Wordt de strikte markering een fout, dan is de grens weg: haal de markering weg, en de regel op de card.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier uren: twee van een werkdag en twee van een zaterdag. Het model verwacht 30, 40, 220 en 110 fietsen; 10, 40, 250 en 100 werden er gehuurd. De fouten zijn 20, 0, −30 en 10.

    Voor de werkdag zijn de kwadraten 400 en 0, met een gemiddelde van 200, en de wortel uit 200 is 14,1 fietsen per uur. Voor de zaterdag zijn de kwadraten 900 en 100, met een gemiddelde van 500, en is de RMSE 22,4. Voor alle vier de uren samen is het gemiddelde van de vier kwadraten 1.400 / 4 = 350, en is de RMSE 18,7: niet 18,25, het gemiddelde van 14,1 en 22,4, want de wortel komt als laatste: het gemiddelde van de vier kwadraten is het gemiddelde van 200 en 500, en de wortel uit dat gemiddelde is niet het gemiddelde van de twee wortels. In de card wordt de werkdag de rij `| working day | 2 | 14.1 |`: de groep, de uren, en de RMSE op één decimaal.
    """)
    return


@app.cell(hide_code=True)
def text_val():
    mo.md(r"""
    ### De validatiedagen

    De cel leest de uren van 2011 met `read_hours` uit je project en houdt de validatiedagen over, elke vijfde dag, zoals in [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl). `C.expect` is het verpakte model van de course voor je project: het geeft wat het model voor elk uur verwacht. `targets` geeft de fietsen die echt gehuurd werden, als kolom, en `.reshape(-1)` maakt er één rij getallen van, net als `pred`. `sorted(set(...))` geeft elke waarde van een kolom één keer, op volgorde. Voer de cel uit.
    """)
    return


@app.cell
def example_val():
    from ops.demand import read_hours, targets

    hours = read_hours(2011)
    val = hours[hours["day"] % 5 == 4]   # elke vijfde dag: de 73 validatiedagen
    pred = C.expect(val)                 # wat het model van de course in elk uur verwacht
    y = targets(val).reshape(-1)         # de fietsen die echt gehuurd werden, één rij zoals pred
    print("hours:", len(y))
    print("kinds of day:", sorted(set(val["workingday"])))
    print("weather:", sorted(set(val["weathersit"])))
    print("first 6 hours, model:", pred[:6].round())
    print("first 6 hours, real: ", y[:6])
    return


@app.cell(hide_code=True)
def text_exercise_per_group():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De RMSE per groep

        `rmse_per(hours, pred, column)` meet hoe ver het model ernaast zit voor elke waarde van één kolom, zoals `workingday`. `hours` is een tabel met uren, `pred` wat het model voor elk uur verwacht, en `column` de naam van de kolom. `y` bevat de fietsen die echt gehuurd werden, uit `targets`, als één rij met `.reshape(-1)`. `table` begint leeg. De loop gaat elke waarde van de kolom één keer langs, op volgorde, met `sorted(set(hours[column]))`. `mask` is True voor de uren met deze waarde: `.to_numpy()` haalt de kolom uit pandas, en `torch.tensor` maakt er een tensor van. `error` bevat de fouten van deze uren, wat het model verwacht min de gehuurde fietsen. Elke waarde krijgt een plek in `table`, onder `int(value)`: een dict met het aantal uren, `len(error)`, onder de key `"hours"`, en de RMSE onder de key `"rmse"`, de namen die de card leest. De functie eindigt met `return table`. `import torch` en de import van `targets` uit `ops.demand` staan er voor de functie.

        Vervang de `...` door deze dict, met de RMSE van `error`, de wortel uit het gemiddelde van de gekwadrateerde fouten, als getal van Python.

        De check roept `rmse_per(val, pred, "workingday")` aan, met de validatiedagen van de cel hierboven en het model van de course, en daarna andere kolommen en een eigen kleine tabel. Onder de check splitst je functie de validatiedagen per soort dag en per weer. Klopt het, dan bewaart de check je functie als `work/projects/ops/report.py`.
        """),
        L.task("Vervang de `...` door de dict van de groep: de uren, en de RMSE van `error` als getal van Python."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_group():
    box_per_group = L.code_box("per-group", lines=13)
    box_per_group
    return (box_per_group,)


@app.cell(hide_code=True)
def check_per_group(box_per_group):
    result_per_group = L.check("per-group", box_per_group.value)
    result_per_group
    return (result_per_group,)


@app.cell(hide_code=True)
def hints_per_group():
    L.hints(
        "per-group",
        [
            "Na `table[int(value)] =` komt een dict met twee keys: `\"hours\"` met `len(error)`, en `\"rmse\"` met de RMSE van de groep. Deze RMSE heeft alleen `error` nodig, de fouten van de groep, en moet eruit komen als gewoon getal van Python, niet als tensor.",
            "De RMSE in drie stappen, met een operator of een methode van een tensor voor elke stap: `** 2` kwadrateert elke fout, `.mean()` neemt het gemiddelde van de kwadraten, en `.sqrt()` de wortel uit dat gemiddelde. Wat eruit komt is nog steeds een tensor met één getal erin; `.item()` geeft dat getal als gewoon getal van Python.",
            "`        table[int(value)] = {\"hours\": len(error), \"rmse\": ___}`",
        ],
        why="`\"rmse\": (error ** 2).mean().sqrt().item()`: de wortel uit de gemiddelde gekwadrateerde fout van de uren onder `mask`, als getal van Python. Voor de validatiedagen: 48,2 fietsen per uur op werkdagen en 60,9 op de andere dagen; 54,1 bij helder weer, 48,0 bij mist en 60,1 bij lichte regen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_weekend():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een bekende grens, gemarkeerd

        `test_weekend(expect)` is de invariance test van het weekend uit [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl). De test leest `HOURS` met `read_hours` uit `ops.demand`. `days_off` houdt de uren over van de dagen die geen werkdag zijn, en `weekend` houdt daarvan alleen de uren buiten de feestdagen over: 2.495 uren. `saturday` is een kopie van `weekend`, gemaakt met `.copy()`, met weekday 6 in elk uur, en `sunday` een kopie met weekday 0. De `assert` vraagt dat de antwoorden van `expect` voor de twee kopieën binnen 20 fietsen van elkaar liggen, in elk uur: `.abs()` maakt elk verschil positief, en `.all()` vraagt dat van elk uur. `import pytest` staat er voor de markering.

        Het model van de course haalt deze test niet, in 333 uren. Zet direct boven `def test_weekend(expect):` één regel die de test markeert als bekende grens: verwacht om te falen, met een reden, en strikt.

        De check leest de markering op je test zoals pytest de markering leest, en voert je test uit met twee modellen: het model van de course, `h16-lr0.01`, en een versie die op dezelfde manier getraind is zonder de kolom weekday, waarvoor zaterdag en zondag hetzelfde zijn. Onder de check zie je wat pytest voor elk model zou melden. Klopt het, dan bewaart de check je test als `work/projects/ops/test_weekend.py`.
        """),
        L.task("Zet de regel die `test_weekend` markeert als verwacht om te falen, met een reden, en strikt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weekend():
    box_weekend = L.code_box("weekend", lines=15)
    box_weekend
    return (box_weekend,)


@app.cell(hide_code=True)
def check_weekend(box_weekend):
    result_weekend = L.check("weekend", box_weekend.value)
    result_weekend
    return (result_weekend,)


@app.cell(hide_code=True)
def hints_weekend():
    L.hints(
        "weekend",
        [
            "De markering staat op de regel direct boven `def test_weekend(expect):`, en begint met `@`, zoals `@pytest.fixture` in `conftest.py`. `@pytest.mark.xfail` zegt dat de test hoort te falen; `reason` zegt waarom, en `strict=True` laat een geslaagde test als fout tellen.",
            "De reden is een korte tekst tussen aanhalingstekens: wat in de inputs van het model zet zaterdag en zondag ver uit elkaar? Geef de reden met de naam erbij, `reason=`, want pytest leest een eerste waarde zonder naam als iets anders.",
            "`@pytest.mark.xfail(reason=___, strict=True)`",
        ],
        why="`@pytest.mark.xfail(reason=\"weekday is one number, so Saturday and Sunday lie far apart\", strict=True)`, direct boven `def test_weekend(expect):`. Voor het model van de course meldt pytest XFAIL en blijft de suite groen; voor de versie zonder weekday meldt pytest `[XPASS(strict)]`, een fout die zegt dat de grens weg is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_card():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De card

        `card(config, groups, limits)` schrijft de card als één tekst in Markdown. `config` is de dict van `config.json` uit het pakket. `groups` is een dict met voor elke groep de uren en de RMSE, zoals `{"working day": {"hours": 1119, "rmse": 48.18}}`. `limits` is een lijst zinnen. `lines` begint met de kop, met de versie uit `config['version']`, dan twee zinnen over waarvoor het model is en waarvoor niet, met de run uit `config['run']`, een kop met de validation RMSE uit `config['val_rmse']`, met `round` afgerond op twee decimalen, en de eerste twee regels van een tabel: de kopregel `| Group | Hours | RMSE |`, en `|---|---|---|`, die Markdown onder een kopregel nodig heeft. De loop gaat `groups.items()` langs: elke `name` met de bijbehorende `row`. Na de loop komen een kop `## Known limits` en een regel met `- ` voor elke `limit`. `"\n".join(lines)` voegt de regels samen tot één tekst, met `\n`, het teken voor een nieuwe regel, ertussen.

        Vervang de `...` door een regel die één rij van de tabel aan `lines` toevoegt: de naam van de groep, de uren, en de RMSE afgerond op één decimaal, elk tussen `|`-tekens.

        De check roept je `card` aan met het pakket van de course, de RMSE per groep uit oefening 1 en de bekende grenzen, en nog een keer met andere groepen en zonder grenzen. Onder de check zie je je card zoals een site Markdown laat zien. Klopt het, dan bewaart de check je functie als `work/projects/ops/card.py`.
        """),
        L.task("Vervang de `...` door een regel die één rij van de tabel aan `lines` toevoegt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_card():
    box_card = L.code_box("card", lines=13)
    box_card
    return (box_card,)


@app.cell(hide_code=True)
def check_card(box_card):
    result_card = L.check("card", box_card.value)
    result_card
    return (result_card,)


@app.cell(hide_code=True)
def hints_card():
    L.hints(
        "card",
        [
            "`lines.append` voegt één regel aan de lijst toe. De rij is een f-string met `name` en `row['hours']` in de cellen, net als de kopregel `| Group | Hours | RMSE |`, en de RMSE uit `row` in de laatste cel.",
            "De RMSE moet afgerond worden op één decimaal: `round` met twee waarden, het getal en hoeveel decimalen. In een f-string komt de hele aanroep tussen `{` en `}`.",
            "`        lines.append(f\"| {name} | {row['hours']} | ___ |\")`",
        ],
        why="`lines.append(f\"| {name} | {row['hours']} | {round(row['rmse'], 1)} |\")`, ingesprongen in de loop: één rij per groep, zoals `| working day | 1119 | 48.2 |`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_heavy_rain():
    mo.md(r"""
    ### Oefening 4. Zware regen

    De tabel van je card heeft een rij voor elk weer van de validatiedagen: helder, mist, en lichte regen of sneeuw. Zware regen, weathersit 4, heeft geen rij. In heel 2011 had één uur zware regen: 4 uur 's middags op woensdag 26 januari, een trainingsdag, met 36 fietsen.
    """)
    return


@app.cell(hide_code=True)
def choice_heavy_rain_cell():
    choice_heavy_rain = L.choice(
        "heavy-rain",
        question="Wat hoort je card over zware regen te zeggen?",
        options={
            "nothing": "Helemaal niets: een weer zonder uren is geen grens van het model, dus heeft de card niets te melden",
            "train": "De RMSE van het ene uur met zware regen op een trainingsdag, omdat één getal beter is dan geen getal",
            "right": "Dat zware regen niet getest is, want geen validatie-uur laat zien hoe ver het model er dan naast zit",
            "light": "De RMSE van lichte regen ook voor zware regen, omdat zware regen hetzelfde soort weer is, alleen meer",
        },
        correct="right",
        why={
            "right": "Geen enkel validatie-uur had zware regen, dus zegt geen getal hoe ver het model er dan naast zit, en het model leerde zware regen van één uur. Laat een card zware regen weg, dan lijkt de card te zeggen dat er niets misgaat: een groep die niemand mat komt helemaal niet voor, zoals [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl) waarschuwde. Dus zegt de card wat er niet getest is, bijvoorbeeld met een regel bij de bekende grenzen, en de run hieronder voegt zo'n regel toe.",
            "nothing": "Zonder uren kan de card geen getal geven, terwijl een lezer zonder rij voor zware regen misschien denkt dat het model het dan goed doet: wat zegt een card over een groep die niemand mat?",
            "train": "Het model leerde van dat uur, dus zegt de fout daar hoe goed het model dat ene uur leerde, en één uur is veel te weinig voor een getal: wat blijft er over voor de card om te zeggen?",
            "light": "Zware regen houdt meer fietsers thuis dan lichte regen, en het model zag zware regen één keer: zegt het getal van lichte regen iets over zware regen, of is het een gok?",
        },
    )
    choice_heavy_rain
    return (choice_heavy_rain,)


@app.cell(hide_code=True)
def text_demo(result_card, result_per_group, result_weekend):
    result_per_group, result_weekend, result_card  # opnieuw gemaakt na de checks, die je onderdelen bewaren
    mo.md(r"""
    ### Elke test, en de card

    De knop voert elke test van je project tegelijk uit, met pytest. Eerst kopieert de knop je project naar `work/check/sys-project-tested-model/run`, en zet de course een eigen versie in de kopie van elk bestand dat je project nog niet heeft, zodat de run ook werkt als je een les oversloeg; heeft je project nog geen pakket, dan krijgt de kopie het pakket van de course. pytest draait in een eigen proces van Python, zoals in een terminal, en dat duurt een paar seconden. Daarna schrijven je `rmse_per` en je `card` de card van je pakket, en bewaart de course de card naast het pakket als `work/projects/ops/model/README.md`. Je `card` krijgt in `limits` de bekende grenzen van je model, en een regel voor elke groep zonder validatie-uren, zoals zware regen uit Oefening 4.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Voer elke test uit en schrijf de card")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_card, result_per_group, result_weekend, run_button):
    result_per_group, result_weekend, result_card   # opnieuw met je bestanden zodra een check een bestand bewaart
    if not run_button.value:
        _out = mo.md("*Druk op de knop om elke test van je project uit te voeren.*")
    else:
        mo.output.replace(mo.md("**pytest draait** …"))
        _r = C.demo()
        if _r.get("error"):
            _out = mo.md(C.summary(_r, L.lang))
        else:
            _out = mo.vstack([
                mo.Html(C.suite_picture(_r, L.lang)),
                mo.Html(C.card_view(_r, L.lang)),
                mo.md(C.summary(_r, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Met de bestanden van de course voert pytest zes tests uit: vijf slagen, en `test_weekend` faalt zoals de markering verwacht, dus is de suite groen, `5 passed, 1 xfailed`. De tests zelf duren ongeveer een seconde, het meeste daarvan `test_overfit_one_batch`, die een netwerk traint; het starten van Python en PyTorch kost de rest. De card heeft de versie van het pakket, `cb4f177c9137` voor het pakket van de course, de validation RMSE van 53,04, vijf rijen uit `rmse_per`, en vier bekende grenzen: minder dan 0 fietsen 's nachts, zaterdag tegenover zondag, 2012, en zware regen, niet getest.

        De derde grens komt uit uren die het model nooit zag: de 8.734 uren van 2012. Met veel meer fietsers dan in 2011 zit het model daar 130,7 fietsen per uur ernaast, meer dan twee keer de 53,04. De validatiedagen konden dat niet laten zien: het zijn dagen van 2011, tussen de trainingsdagen. *Project: je model als service, bewaakt op drift* komt terug op 2012.

        In een terminal, in de map van de course, voert `uv run pytest work/projects/ops` dezelfde tests uit op je project zelf, en met `-rx` erbij zet pytest de reden naast elke XFAIL.
        """),
        L.watch_out("Een xfail-markering telt elke fout als de verwachte fout, ook een fout die niets met de grens te maken heeft, zoals een tikfout in de naam van een kolom in de test. Dan blijft de suite groen terwijl de test niets test. `raises=AssertionError` in de markering maakt het smaller: alleen een `assert` die faalt telt als verwacht, en elke andere fout laat de test falen."),
    ])
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tests voor ML-code](/course/open/sys-code-tests?lang=nl) en [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl) keken je tests naar de code en naar het model, een voor een; nu voert één opdracht alle tests uit, en is een bekende grens gemarkeerd, niet verborgen. Nu schrijft code de card van [Voor wie werkt het? Model cards en system cards](/course/open/sys-model-card?lang=nl), uit het pakket en de getallen, en de card ligt naast het pakket. Je project *ops* heeft drie bestanden meer: `report.py`, `test_weekend.py` en `card.py`. In *Project: je model als service, bewaakt op drift* beantwoordt je model verzoeken, en komen de uren van 2012 terug, als data die is weggeschoven van wat het model leerde.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_heavy_rain, result_card, result_per_group, result_weekend):
    L.wrap_up(result_per_group, result_weekend, result_card, choice_heavy_rain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
