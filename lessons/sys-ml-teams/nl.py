# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-ml-teams. Route: course/route/lessen/systemen.yaml. Checks, de plannen van de vier manieren en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-ml-teams", "nl")
    C = L.checks   # de rollen, de plannen van de vier manieren om het werk in te richten, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_back():
    quick_back = L.choice(
        "quick-back",
        question="Waar stuurt een herbeoordeling in de levenscyclus een systeem heen dat nog nodig is, alleen niet meer goed genoeg?",
        options={
            "retire": "Door naar uitfasering",
            "right": "Terug naar ontwerp en ontwikkeling",
            "inception": "Terug naar het idee, om helemaal opnieuw te beginnen",
            "operation": "Meteen terug in gebruik, zonder verandering",
        },
        correct="right",
        why={
            "right": "Opnieuw door ontwerp en ontwikkeling, verificatie en validatie en deployment: het systeem krijgt een nieuwe versie. Deze les kijkt naar wie de nieuwe versie bouwt, en hoe lang dat duurt.",
            "retire": "Uitfasering is voor een systeem dat niet meer nodig is. Waar gaat een systeem heen dat nog wel nodig is? Zie [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).",
            "inception": "Het probleem en het doel zijn nog hetzelfde; alleen het systeem schiet tekort. Welke fase bouwt een nieuwe versie? Zie [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).",
            "operation": "Zonder verandering doordraaien houdt het systeem onder het doel. Welke fase verandert het systeem? Zie [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).",
        },
    )
    quick_back
    return (quick_back,)


@app.cell(hide_code=True)
def quick_check_producer():
    quick_producer = L.choice(
        "quick-producer",
        question="Een bedrijf bouwt de spamdienst en verkoopt de dienst aan een gemeente. Welke rollen heeft het bedrijf, in de woorden van ISO/IEC 22989?",
        options={
            "customer": "AI-klant en AI-subject",
            "partner": "Alleen AI-partner",
            "right": "AI-producent en AI-aanbieder",
            "all": "AI-producent, AI-aanbieder en relevante autoriteiten",
        },
        correct="right",
        why={
            "right": "Het bedrijf ontwerpt en bouwt de dienst (AI-producent) en biedt de dienst aan (AI-aanbieder). Binnen het bedrijf is de AI-producent niet één persoon: deze les kijkt naar de mensen die de dienst bouwen, en hoe de mensen samenwerken.",
            "customer": "De gemeente gebruikt de dienst, en de inwoners worden erdoor geraakt. Wat doet het bedrijf dat de dienst bouwt en verkoopt? Zie [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).",
            "partner": "Een AI-partner helpt rond de AI, zoals een labelbureau. Welke rollen horen bij een dienst bouwen en aanbieden? Zie [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).",
            "all": "Relevante autoriteiten maken de regels en zien erop toe dat de regels nageleefd worden, zoals de privacytoezichthouder. Is dat het bedrijf dat de dienst bouwt? Zie [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl).",
        },
    )
    quick_producer
    return (quick_producer,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2015 begon Uber aan Michelangelo, het eigen platform voor machine learning. In september 2017 beschreven Jeremy Hermann en Mike Del Balso op het blog van Uber hoe het tot dan toe ging. Data scientists bouwden modellen met allerlei tools, zoals R en scikit-learn, terwijl aparte teams van engineers losse systemen bouwden om de modellen in productie te gebruiken. Er was geen vaste weg om een model in gebruik te nemen, en wat machine learning voor Uber deed, bleef beperkt tot wat een paar data scientists en engineers in korte tijd konden bouwen. Een model wordt door mensen gebouwd, en hoe de mensen samenwerken, bepaalt hoe snel een nieuw model in gebruik komt, en of het er ooit komt. Deze les kijkt naar de rollen van de mensen die een model bouwen en draaiende houden, en naar vier manieren om het werk in te richten.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe keuken. Losse vakmensen, een loodgieter, een elektricien en een tegelzetter, werken elk snel in het eigen vak, en toch wacht de keuken tussen de vakmensen in: de tegelzetter kan pas beginnen als de elektricien klaar is, en de agenda van de elektricien zit vol. Eén klusjesman die alles doet, wacht nooit op iemand, maar doet langer over de bedrading. Een ploeg die samenwerkt, wacht minder, maar heeft voor elke keuken elk vak nodig. En een keuken uit de winkel, met kant-en-klare delen, gaat het snelst, zolang je wilt wat de winkel verkoopt.
    """)
    return


@app.cell(hide_code=True)
def predict_models():
    guess_models = L.guess(
        "models",
        question="In 2024 schreef Uber op het eigen engineeringblog hoeveel modellen er in productie draaiden. Hoeveel, denk je?",
        options={
            "fifty": "Ongeveer 50",
            "hundreds": "Ongeveer 500",
            "right": "Meer dan 5.000",
            "half-million": "Ongeveer een half miljoen",
        },
        reveal="**Meer dan 5.000.** In mei 2024 schreef Uber dat er meer dan 5.000 modellen in productie draaiden, met op de drukste momenten 10 miljoen voorspellingen per seconde, en dat er ongeveer 400 actieve ML-projecten op Michelangelo liepen. In 2017 schreef de blogpost over Michelangelo nog dat tientallen teams het platform gebruikten. De conclusie van Uber over de mensen erachter: de beste opzet heeft één centraal team voor het ML-platform, met data scientists en ML engineers in elk productteam.",
    )
    guess_models
    return (guess_models,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De rollen.** In [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl) was het bedrijf de AI-producent. Binnen het bedrijf is het werk over rollen verdeeld. Dominik Kreuzberger, Niklas Kühl en Sebastian Hirschl (2023) noemen zeven rollen rond MLOps, van de business stakeholder die het doel stelt tot de ML engineer/MLOps engineer. Deze les gebruikt vier rollen die een model bouwen en draaiende houden; de platform engineer is geen van de zeven:

        | Rol | Wat de rol doet | Bij de spamdienst |
        |---|---|---|
        | **data scientist** | maakt van een vraag een ML-probleem, verzamelt data, traint en evalueert het model | verzamelt en labelt berichten, en traint het spammodel |
        | **ML engineer** | maakt van het model betrouwbare software: een geteste dienst die snel genoeg antwoordt | maakt van het model een dienst die op elk bericht antwoordt |
        | **MLOps engineer** | bouwt de weg naar productie: pipelines, deployment en monitoring | zet de dienst in de mailbox, met alarmen |
        | **platform engineer** | bouwt de gedeelde tools die elk team gebruikt | bouwt de tools om elk model van het bedrijf te trainen, te deployen en in de gaten te houden |

        De grenzen tussen de rollen verschillen per bedrijf, en één persoon kan meerdere rollen hebben: in een klein bedrijf doet één persoon alles. Kreuzberger en anderen noemen ook een data engineer, die de pipelines bouwt die de data binnenhalen, en voegen de ML engineer en de MLOps engineer samen tot één rol.

        **Overdrachten.** Elke keer dat het werk van de ene persoon naar de andere gaat, is er een **overdracht** (handoff). Bij een overdracht wacht het werk tot de volgende persoon tijd heeft, en een deel van wat de eerste persoon wist, blijft achter: waarom een feature gekozen is, wat het model met vreemde berichten doet. In 2019 schreef Eric Colson van Stitch Fix dat de wachttijd tussen twee stukken werk meer schade aanricht dan de kosten van het afstemmen, en vaak in dagen, weken of zelfs maanden gemeten wordt.

        **Vier manieren.** Een bedrijf kan dezelfde rollen op verschillende manieren inrichten, en elke manier verplaatst de plek waar het werk vastloopt:
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        | Manier | Wie doet wat | Waar het werk vastloopt |
        |---|---|---|
        | **teams per functie** | een team van data scientists, een team van ML engineers en een team voor beheer; het model gaat van team naar team | in de wachtrijen tussen de teams, en in bugs die niemand kan plaatsen, omdat elk team één deel ziet |
        | **één team per product** | een data scientist, een ML engineer en een MLOps engineer in één team rond één product | de wachttijden zijn kort, maar elk product heeft elke rol nodig, en specialisten zijn schaars |
        | **één data scientist van begin tot eind** | één persoon doet elke stap: een **end-to-end** data scientist | niemand wacht, alleen duren de stappen buiten het eigen vak langer, en weinig mensen kunnen en willen alles doen |
        | **een data scientist met een platform** | één persoon doet elke stap, met de tools van een **ML-platform** dat platform engineers voor elk team bouwen | snel, zolang het platform kan wat het model nodig heeft; en het platform kost eerst maanden werk |

        Chip Huyen (2022) noemt de prijs van aparte teams: de extra communicatie. Data scientists die alles doen, schrijft Huyen, zijn moeilijk te vinden, en niet altijd blij met al dat werk, tenzij goede tools de infrastructuur uit handen nemen. Dat is de taak van een platform. Michelangelo is er een: het platform dekt bij Uber zes stappen voor elk team, van data beheren tot de voorspellingen in de gaten houden.

        ### Nieuwe woorden en tekens

        | Woord of symbool | Uitspraak | Betekenis | In code |
        |---|---|---|---|
        | **data scientist** | | iemand die een vraag omzet in een ML-probleem, en het model bouwt en evalueert | `"data scientist"` in `C.PLAN` |
        | **ML engineer** | em-el engineer | iemand die een model omzet in betrouwbare software die snel genoeg antwoordt, en de software draaiende houdt | `"ML engineer"` in `C.PLAN` |
        | **MLOps engineer** | em-el-ops engineer | iemand die de weg naar productie bouwt: pipelines, deployment en monitoring | `"MLOps engineer"` in `C.PLAN` |
        | **platform engineer** | | iemand die de gedeelde tools bouwt die elk team gebruikt | |
        | **ML-platform** (ML platform) | em-el-platform | de gedeelde tools die een bedrijf de eigen teams geeft om modellen te trainen, te deployen en in de gaten te houden | `C.PLATFORM_DAYS` |
        | **overdracht** (handoff) | | het moment dat het werk naar iemand anders gaat: het werk wacht daar, en een deel van wat de eerste persoon wist, blijft achter | twee stappen na elkaar met een andere `who` |
        | **end-to-end** | | gedaan door één persoon of team, van de data tot een model in gebruik | `C.PLANS["start to finish"]` |
        """),
        L.watch_out("Functietitels betekenen bij elk bedrijf iets anders: een data scientist bij het ene bedrijf doet wat een ML engineer bij het andere doet, en een MLOps engineer heet soms platform engineer. Kijk naar wat een functie inhoudt, niet naar hoe de functie heet."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Kiezen hoe je het werk aan modellen inricht:

    1. Schrijf de stappen op van de data tot een model in gebruik, en wie elke stap doet.
    2. Tel de overdrachten, en schat hoe lang het werk bij elke overdracht wacht: een dag of twee binnen één team, vaak meer tussen teams.
    3. Zoek de stappen die iemand buiten het eigen vak doet: deze stappen duren langer, en gaan vaker mis.
    4. Tel de modellen die het bedrijf gaat bouwen: een platform kost maanden werk voordat het iets bespaart, dus betaalt het zich alleen terug over veel modellen.
    5. Kijk wat het platform nog niet kan: werk dat erbuiten valt, wacht in de wachtrij van de platform engineers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.PLAN` is het plan voor een nieuwe versie van het spammodel van de course bij een bedrijf met teams per functie, door de course bedacht: een lijst met stappen `(step, who, days)`, op volgorde. De cel print elke stap en telt de dagen werk op. Voer de cel uit.
    """)
    return


@app.cell
def example_plan():
    plan = C.PLAN
    work = 0
    for step, who, days in plan:
        print(step, "-", who, "-", days, "dagen")
        work = work + days
    print("dagen werk:", work)
    return plan, work


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het plan heeft 28 dagen werk. Maar het werk gaat van de data scientist naar de ML engineer, en van de ML engineer naar de MLOps engineer, elk in een ander team. Als het werk in de wachtrij van elk team 10 dagen wacht, antwoordt de nieuwe versie na 28 + 10 + 10 = 48 dagen, en 20 van de 48 dagen werkt niemand eraan. Oefening 1 telt de overdrachten in een plan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_handoffs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tel de overdrachten

        `handoffs(plan)` krijgt een plan als een lijst met stappen `(step, who, days)`, op volgorde, en geeft terug hoe vaak het werk naar iemand anders gaat. `count` telt de overdrachten, en de loop haalt elke stap uit elkaar in `step`, `who` en `days`. `previous` bevat de `who` van de vorige stap; `previous` begint bij `None`, omdat de eerste stap geen vorige stap heeft, en de voorwaarde controleert dat met `previous is not None`. Vervang de `...` door de tweede helft van de voorwaarde, die zegt of deze stap een overdracht is.

        De check roept `handoffs(C.PLAN)` aan. Daarna een leeg plan, een plan van één stap, het plan van één data scientist van begin tot eind, `C.BACK_AND_FORTH`, waarin de ML engineer het model één keer terugstuurt, en een plan waarin het werk heen en weer gaat tussen twee mensen.
        """),
        L.task("Tel hoe vaak het werk naar iemand anders gaat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_handoffs():
    box_handoffs = L.code_box("handoffs", lines=8)
    box_handoffs
    return (box_handoffs,)


@app.cell(hide_code=True)
def check_handoffs(box_handoffs):
    result_handoffs = L.check("handoffs", box_handoffs.value)
    result_handoffs
    return (result_handoffs,)


@app.cell(hide_code=True)
def hints_handoffs():
    L.hints(
        "handoffs",
        [
            "Stap 2 van het recept: een overdracht is een stap waarvan `who` niet de `who` van de vorige stap is. In de loop bevat `previous` de persoon van de vorige stap.",
            "De voorwaarde vergelijkt de persoon van deze stap met de persoon van de vorige stap. Welke vergelijking zegt \"is niet gelijk aan\"?",
            "`        if previous is not None and ___:`",
        ],
        why="`who != previous` is waar bij elke stap waarvan de persoon verschilt van de persoon van de vorige stap. `C.PLAN` heeft 2 overdrachten, en `C.BACK_AND_FORTH` heeft er 4: het werk gaat naar de ML engineer, terug naar de data scientist, weer naar de ML engineer, en door naar de MLOps engineer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_payoff():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wanneer een platform zich terugbetaalt

        Een platform kost werk voordat het iets bespaart: in het voorbeeld van de course werken twee platform engineers er ongeveer zes maanden aan, 240 dagen werk. Daarna heeft een data scientist met het platform 3 dagen nodig om van een nieuwe versie een dienst te maken en 1 dag om de dienst te deployen, waar één data scientist zonder platform 10 en 8 dagen nodig heeft: een nieuwe versie kost 24 dagen werk in plaats van 38, en het platform bespaart zo bij elk model 14 dagen. Het voorbeeld laat het onderhoud van het platform weg.

        `models_to_pay_off(platform_days, saved)` geeft terug hoeveel modellen er nodig zijn tot de bespaarde dagen samen minstens zo veel zijn als de dagen die het platform kostte. `models` telt de modellen, en `gained` houdt bij hoeveel dagen er tot nu toe bespaard zijn: na elk model `saved` meer. Een `while`-loop gaat door zolang de voorwaarde waar is. Vervang de `...` door de voorwaarde waaronder het platform zich nog niet terugbetaald heeft.

        De check roept `models_to_pay_off(240, 14)` aan. Daarna een platform van 100 dagen dat bij elk model 10 dagen bespaart, helemaal geen platform, een platform dat zich met het eerste model terugbetaalt, een platform van 50 dagen dat 7 dagen bespaart, en het platform van 240 dagen vergeleken met teams per functie, waar het bij elk model 4 dagen werk bespaart.
        """),
        L.task("Tel de modellen tot de bespaarde dagen de dagen van het platform halen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_payoff():
    box_payoff = L.code_box("payoff", lines=8)
    box_payoff
    return (box_payoff,)


@app.cell(hide_code=True)
def check_payoff(box_payoff):
    result_payoff = L.check("payoff", box_payoff.value)
    result_payoff
    return (result_payoff,)


@app.cell(hide_code=True)
def hints_payoff():
    L.hints(
        "payoff",
        [
            "Stap 4 van het recept. De loop telt er één model per keer bij en moet stoppen zodra `gained` bij `platform_days` komt. De voorwaarde van een `while` zegt wanneer de loop doorgaat, niet wanneer de loop stopt.",
            "Vergelijk `gained` met de kosten van het platform. Welke vergelijking zegt \"nog kleiner dan\"?",
            "`    while ___:`",
        ],
        why="`while gained < platform_days:` telt modellen tot de bespaarde dagen de kosten van het platform halen. Met 14 bespaarde dagen per model besparen 17 modellen 238 dagen, net minder dan 240, dus betaalt het platform zich terug vanaf het 18e model.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Eén nieuwe versie, vier manieren

    De demo volgt één nieuwe versie van het spammodel, dag voor dag, bij vier bedrijven die het werk elk op een van de vier manieren inrichten, met de plannen van `C.PLANS`. Elke balk heeft de dagen werk ingekleurd, donker tot nu en licht voor wat nog komt, en de dagen wachten als vakken met een rand van streepjes; de lijn is de huidige dag. Onder elke balk staat wat er nu gebeurt. De dagen zijn door de course bedacht: 28 dagen werk, en 10 dagen wachten bij elke overdracht tussen teams, of 2 binnen één team.
    """)
    return


@app.cell(hide_code=True)
def demo_days():
    L.explore("day", "dag", lambda _v: C.frame(int(_v), L.lang), start=1, values=list(range(1, 49)), role="",
              task="Ga de 48 dagen langs: bij welke manier wacht het werk het langst, en waar doet de data scientist van begin tot eind langer over?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij teams per functie duurt de nieuwe versie 48 dagen, en 20 van de 48 dagen werkt niemand eraan. Bij één team per product krimpen de wachttijden tot 4 dagen, maar elk productteam heeft een eigen ML engineer en een eigen MLOps engineer nodig, voor 5 en 3 dagen werk aan deze versie. De data scientist van begin tot eind wacht nooit, maar is 18 dagen bezig met een dienst maken en deployen, waar de specialisten 8 dagen nodig hebben. Met een platform antwoordt de nieuwe versie na 24 dagen, omdat platform engineers eerst 240 dagen aan het platform werkten; in oefening 2 betaalde het platform zich terug vanaf het 18e model.

    Echte bedrijven mengen de manieren: de conclusie van Uber uit 2024 combineert één centraal platformteam met data scientists en ML engineers in elk productteam. En de wachttijden zijn echt. In een enquête onder 403 leidinggevenden bij bedrijven met een omzet van minstens 100 miljoen dollar, in december 2020 gepubliceerd door Algorithmia, een bedrijf dat MLOps-tools verkoopt, zei 64% dat het deployen van een model bij het eigen bedrijf een maand of langer duurt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_role():
    mo.md(r"""
    ### Oefening 3. Van wie is dit werk?

    Vijf productteams bij een bedrijf bouwden elk een eigen manier om een model in gebruik te nemen, met eigen servers en eigen alarmen. Het bedrijf wil één manier die elk team kan gebruiken.
    """)
    return


@app.cell(hide_code=True)
def choice_role_cell():
    choice_role = L.choice(
        "role",
        question="Welke rol bouwt de gedeelde manier?",
        options={
            "data": "Een data scientist in elk team, die de modellen het best kent",
            "mle": "De ML engineer van de spamdienst, die serving kent",
            "authority": "De relevante autoriteiten, die de regels maken",
            "right": "Een platform engineer, die tools voor elk team bouwt",
        },
        correct="right",
        why={
            "right": "Gedeelde tools voor elk team zijn het werk van platform engineers: een ML-platform, zoals Michelangelo bij Uber. De teams bouwen geen eigen manieren meer, en vijf manieren om te onderhouden worden er één.",
            "data": "Een data scientist kent de modellen van één team. Wie bouwt tools die alle vijf de teams kunnen gebruiken?",
            "mle": "De ML engineer van de spamdienst bouwt de dienst van één product. Wie maakt een tool voor elk product?",
            "authority": "De relevante autoriteiten maken de regels rond een systeem, buiten het bedrijf. Wie binnen het bedrijf bouwt tools voor alle teams?",
        },
    )
    choice_role
    return (choice_role,)


@app.cell(hide_code=True)
def text_exercise_stuck():
    mo.md(r"""
    ### Oefening 4. Waar het vastloopt

    Met het platform neemt een data scientist een nieuw getraind spammodel in 4 dagen in gebruik: 3 dagen om er een dienst van te maken en 1 dag om de dienst te deployen. Nu moet de spamdienst ook in de browser van de inwoners draaien, en het platform serveert modellen alleen op de eigen servers.
    """)
    return


@app.cell(hide_code=True)
def choice_stuck_cell():
    choice_stuck = L.choice(
        "stuck",
        question="Waar loopt het werk vast?",
        options={
            "right": "In de wachtrij van het platformteam, tot het platform het kan",
            "nowhere": "Nergens: met een platform kan één persoon elke stap doen",
            "handoff": "Tussen de ML engineer en de MLOps engineer",
            "data": "Bij het verzamelen van data: de browser heeft andere berichten nodig, dus begint de data scientist opnieuw",
        },
        correct="right",
        why={
            "right": "Een platform is snel voor wat het kan. Voor iets nieuws wacht de data scientist tot de platform engineers het bouwen, of bouwt het zelf, en is dan terug bij één data scientist van begin tot eind.",
            "nowhere": "De data scientist kan elke stap doen die het platform ondersteunt. Wat gebeurt er met een stap die het platform niet ondersteunt?",
            "handoff": "Met een platform doet de data scientist alle stappen. Wie moet het platform uitbreiden, zodat het modellen in een browser kan serveren?",
            "data": "De berichten blijven hetzelfde; alleen de plek waar het model draait, verandert. Welk deel van het werk dekt het platform niet?",
        },
    )
    choice_stuck
    return (choice_stuck,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Van idee tot afscheid: de levenscyclus van een AI-systeem](/course/open/sys-lifecycle?lang=nl) was het bedrijf dat de spamdienst bouwde één partij, de AI-producent. Deze les keek binnen in het bedrijf: de rollen van de mensen die een model bouwen en draaiende houden, de overdrachten tussen deze mensen, en vier manieren om het werk in te richten, elk met een eigen plek waar het werk vastloopt. In de volgende les, *Betrouwbaar, schaalbaar, onderhoudbaar, aanpasbaar*, kijk je naar wat zo'n team moet opleveren: een systeem dat blijft werken, meegroeit met de gebruikers, door anderen te veranderen is, en een veranderende wereld volgt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_role, choice_stuck, result_handoffs, result_payoff):
    L.wrap_up(result_handoffs, result_payoff, choice_role, choice_stuck)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
