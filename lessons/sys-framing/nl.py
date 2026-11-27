# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-framing. Route: course/route/lessen/systemen.yaml. Checks, de apps, de feed van de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-framing", "nl")
    C = L.checks   # de weergaven, de apps met de scores, de feed van de demo, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl) en [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_task():
    quick_task = L.choice(
        "quick-task",
        question="Een model voorspelt de prijs van een huis in euro. Wat voor taak is dat?",
        options={
            "classification": "Classificatie: elke prijs is een klasse",
            "right": "Regressie: de prijs is een getal in euro",
            "reward": "Reinforcement learning: een reward",
            "ranking": "Ranking: de huizen van goedkoop naar duur",
        },
        correct="right",
        why={
            "right": "Een prijs is een getal, dus is dit een regressie. Deze les laat zien dat je één wens vaak op meer dan één manier kunt formuleren, en dat de keuze later uitmaakt.",
            "classification": "Classificatie kiest één klasse uit een vaste set, zoals spam of ok. Wat geeft een model dat een prijs voorspelt? Zie [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "reward": "Reinforcement learning leert van rewards, en dit model leert van huizen met de prijs erbij. Welk soort taak voorspelt een getal? Zie [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "ranking": "Een ranking zet items op volgorde, en dit model geeft één prijs per huis. Welk soort taak voorspelt een getal? Zie [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
        },
    )
    quick_task
    return (quick_task,)


@app.cell(hide_code=True)
def quick_check_mae():
    quick_mae = L.choice(
        "quick-mae",
        question="Een model voorspelt hoeveel minuten een gebruiker naar een video kijkt. Op de test set is de MAE 3 minuten. Wat zegt dat?",
        options={
            "percent": "3% van de voorspellingen is fout",
            "every": "Het voorspelt 3 minuten voor elke video",
            "right": "Het zit meestal zo'n 3 minuten ernaast",
            "squared": "De gekwadrateerde missers zijn gemiddeld 3",
        },
        correct="right",
        why={
            "right": "De MAE is het gemiddelde van de missers zonder teken, in de eenheid van het label: minuten. Kijkminuten komen in deze les terug, bij een echt bedrijf.",
            "percent": "De MAE staat in de eenheid van het label, niet in procenten. In welke eenheid zijn de missers? Zie [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).",
            "every": "Dat zou een model zijn dat altijd hetzelfde voorspelt. Wat meet de MAE? Zie [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).",
            "squared": "Eerst kwadrateren hoort bij de MSE en de RMSE. Wat doet de MAE met het teken van elke misser? Zie [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).",
        },
    )
    quick_mae
    return (quick_mae,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2016 beschreven drie engineers van Google hoe YouTube de video's kiest die het platform aanraadt (Covington, Adams en Sargin). Een voor de hand liggende manier om de taak te formuleren is voorspellen of een gebruiker op een video klikt: een classificatie, ja of nee. Volgens de drie zet ranken op klikken vaak clickbait bovenaan: misleidende video's met een verleidelijk plaatje en een verleidelijke titel, video's die gebruikers niet afkijken. De engineers kozen dus een andere formulering: het model voorspelt hoe lang de gebruiker naar verwachting kijkt naar elke video die het platform laat zien, en de ranking draait vooral om deze voorspelling.

    De wens bleef hetzelfde: video's laten zien waar mensen naar willen kijken. Het model voorspelt nu iets anders, en daarmee veranderen de video's bovenaan. Deze keuze heet het formuleren van het probleem, en je maakt deze keuze voordat er iets getraind wordt. Deze les laat drie gangbare formuleringen zien, hoe de keuze bepaalt wat je later kunt veranderen zonder nieuw model, en wat je doet als een systeem twee doelen tegelijk heeft.

    **Een voorbeeld uit het dagelijks leven.** Een bibliotheek wil lezers helpen goede boeken te vinden. De bibliotheek kan voorspellen of een lezer een boek leent (ja of nee), hoeveel bladzijden de lezer leest (een getal), of in welke volgorde de bibliotheek tien nieuwe boeken laat zien (een ranking). Lenen voorspellen beloont mooie kaften; gelezen bladzijden voorspellen beloont boeken die mensen uitlezen.
    """)
    return


@app.cell(hide_code=True)
def predict_new_app():
    guess_new_app = L.guess(
        "new-app",
        question="Een telefoon heeft 40 apps. Een model voorspelt welke app de gebruiker als volgende opent: een classificatie met één output per app, 40 in totaal. Dan installeert de gebruiker een 41e app. Wat heeft het model nodig om de nieuwe app ooit te kiezen?",
        options={
            "right": "Opnieuw trainen, met één output meer",
            "nothing": "Niets: het model leert de nieuwe app zelf",
            "example": "Eén voorbeeld van de nieuwe app in de instellingen",
            "threshold": "Een lagere threshold voor nieuwe apps",
        },
        reveal="**Opnieuw trainen.** Het model heeft één output voor elk van de 40 apps waarop het getraind is, en geen voor de 41e. Chip Huyen (2022) gebruikt dit voorbeeld in een vak over ML-systemen aan Stanford, en formuleert het probleem anders: een model krijgt de gebruiker, het moment en één app, en geeft één getal van 0 tot 1, hoe waarschijnlijk de gebruiker de app opent. Om te kiezen geef je elke app een score en neem je de hoogste. Een nieuwe app is dan één app meer om te scoren, geen nieuwe output.",
    )
    guess_new_app
    return (guess_new_app,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een probleem formuleren.** Voordat een model getraind wordt, beslist iemand wat het model precies voorspelt: wat erin gaat, wat eruit komt, en van welk label het model leert. Dat is **probleemformulering** (problem framing). De meeste wensen kun je op meer dan één manier formuleren:

        | Formulering | Het model geeft | Op de videosite | Wat het later betekent |
        |---|---|---|---|
        | classificatie | één klasse uit een vaste set, of een kans per klasse | klikt de gebruiker: ja of nee | een nieuwe klasse, zoals een nieuwe app, betekent opnieuw trainen |
        | regressie | een getal | hoeveel minuten de gebruiker kijkt | een threshold kan in code staan, buiten het model |
        | ranking | een volgorde van items | de 10 video's in de volgorde op het scherm | heeft een score per item nodig, vaak uit een regressie of een kans |

        **Waar de threshold staat.** De cursus van Google over problem framing geeft dit voorbeeld: een videosite houdt een video in een dure cache, snelle opslag dicht bij de kijkers, als de video naar verwachting minstens 50 weergaven krijgt. Een classificatiemodel dat getraind is op "50 weergaven of meer: ja of nee" heeft de 50 in de labels zitten. Een regressiemodel voorspelt het aantal weergaven, en de 50 is een regel code. Het advies van Google: is de threshold vast, gebruik dan een classificatiemodel; kan de threshold veranderen, voorspel dan het getal en houd de threshold in de code van de app, zodat de threshold kan verschuiven zonder nieuw model. Ook de spamdienst uit [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) houdt de threshold buiten het model: het model geeft een kans, en de threshold op deze kans staat in de instellingen.

        **Nieuwe items.** In het voorbeeld van Huyen met de volgende app moet een classificatie met één output per app voor elke nieuwe app opnieuw getraind worden. Als regressie, met één score per gebruiker, moment en app, is een nieuwe app één item meer om te scoren. Huyen noemt dit een regressie; je kunt het ook zien als een classificatie met ja of nee per app. Hoe je het ook noemt, een nieuwe app heeft geen nieuwe output nodig. Een ranking neemt dan de app met de hoogste score, of de tien hoogste.

        **Het label staat in voor de wens.** Geen label is de wens zelf. Bij YouTube beloonden klikken clickbait; kijkminuten geven volgens de engineers beter weer hoe betrokken gebruikers zijn, en ook kijkminuten zijn niet de wens zelf. Vraag bij elk label hoe een model hoog kan scoren op het label zonder te doen wat je bedoelde.

        **Twee doelen.** Veel systemen hebben twee doelen die tegen elkaar in werken. Het voorbeeld van Huyen is een nieuwsfeed die het engagement hoog moet houden, hoeveel gebruikers klikken en reageren, en misinformatie en extreme meningen moet beperken. De ene manier is één model, getraind op één samengestelde loss: α × loss op kwaliteit + (1 − α) × loss op engagement. Elke keer dat het team een andere balans wil, moet het team het model opnieuw trainen. De andere manier is een kwaliteitsmodel en een engagementmodel trainen, elk op een eigen loss, en ranken op α × kwaliteit + (1 − α) × engagement, uitgerekend in code. Dat zijn **ontkoppelde doelen** (decoupled objectives): het team verandert α in een seconde, zonder iets te trainen. Huyen schrijft α en β voor de twee wegingen; deze les gebruikt α en 1 − α, zodat de wegingen samen 1 zijn.

        Huyen geeft een tweede reden: elk model kan in een eigen tempo bijgewerkt worden. In het voorbeeld van Huyen moet een spamfilter veel vaker bijgewerkt worden dan een kwaliteitsmodel, omdat spamtrucs veel sneller veranderen dan wat mensen een goede post vinden. Het rankingsysteem van YouTube uit 2019 (Zhao en anderen) traint één model met een output per doel, zoals klikken, kijktijd en likes, en voegt deze outputs samen in een formule met wegingen die het team met de hand instelt. De afbeelding laat beide manieren zien voor de feed.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **probleemformulering** (problem framing) | | beslissen wat een model voorspelt: wat erin gaat, wat eruit komt, en van welk label het model leert | `next_app(apps, scores)` |
        | **ontkoppelde doelen** (decoupled objectives) | | één model per doel, met de scores samengevoegd via wegingen in code, zodat de balans verandert zonder training | `combined(quality, engagement, alpha)` |
        | **α** | alfa | de weging van kwaliteit in de samengestelde score; engagement krijgt 1 − α | `alpha` |
        """),
        L.watch_out("De weging α is een keuze, niet iets wat training vindt: geen loss zegt hoeveel reacties één misleidende post waard is. Schrijf op wie α koos en waarom, net als bij een threshold."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een wens als ML-probleem formuleren:

    1. Schrijf de wens op als de beslissing die het systeem met de output neemt: wel of niet in de cache, spam of ok, welke volgorde.
    2. Zet de formuleringen op een rij: een klasse, een getal, een volgorde. Noem bij elke formulering het label en hoe je het label zou verzamelen.
    3. Vraag je af wat er later verandert: nieuwe items, een threshold die verschuift, een nieuwe balans tussen doelen. Kies liever de formulering waarbij de verandering in code gebeurt, niet in training.
    4. Train bij twee doelen één model per doel en voeg de scores samen met een weging in code.
    5. Kijk wat het label beloont: kan een model hoog scoren op het label door iets te doen wat je niet bedoelde?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.VIEWS` bevat de weergaven die een regressiemodel voorspelt voor 8 nieuwe video's in de eerste week, verzonnen door de course. Haalt het voorspelde aantal weergaven de threshold, dan gaat de video in de dure cache. De cel telt de video's bij een threshold van 50, en daarna bij 80. Voer de cel uit.
    """)
    return


@app.cell
def example_cache():
    views = C.VIEWS
    for threshold in [50, 80]:
        cached = 0
        for v in views:
            if v >= threshold:
                cached = cached + 1
        print(f"threshold {threshold}: {cached} van de {len(views)} video's in de cache")
    return cached, threshold, views


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij een threshold van 50 gaan 4 van de 8 video's in de cache; bij 80 zijn het er 2. Voor een andere threshold was één getal in code genoeg, en het model bleef hetzelfde. Een classificatiemodel dat getraind is op "50 weergaven of meer" weet niets van 80: dat model heeft nieuwe labels en nieuwe training nodig.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén score per app

        Het model dat de volgende app voorspelt is nu geformuleerd als regressie: voor elke app geeft het model een score van 0 tot 1, hoe waarschijnlijk de gebruiker de app als volgende opent. `next_app(apps, scores)` krijgt een lijst `apps` en een lijst `scores` in dezelfde volgorde, en geeft de app met de hoogste score terug. `best` houdt de plek bij van de beste app tot nu toe, te beginnen met de eerste. De loop gaat langs elke plek `i` uit `range(len(apps))`, en als de voorwaarde klopt, volgt `best = i`. Vervang de `...` door een voorwaarde die waar is als app `i` hoger scoort dan de beste app tot nu toe.

        De check roept `next_app(C.APPS, C.SCORES)` aan, met de 8 apps van één gebruiker. Daarna dezelfde apps met een nieuwe erbij, Bus, en andere lijsten.
        """),
        L.task("Houd de plek bij van de app met de hoogste score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next():
    box_next = L.code_box("next", lines=7)
    box_next
    return (box_next,)


@app.cell(hide_code=True)
def check_next(box_next):
    result_next = L.check("next", box_next.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next",
        [
            "Stap 3 van het recept: met één score per app is een nieuwe app één item meer om te scoren. Vergelijk `scores[i]`, de score van app `i`, met de score van de beste app tot nu toe.",
            "`best` is een plek, geen score. Hoe haal je de score op een plek uit de lijst `scores`?",
            "`        if scores[i] > ___:`",
        ],
        why="`scores[i] > scores[best]` vergelijkt app `i` met de beste app tot nu toe; scoort app `i` hoger, dan onthoudt `best = i` de plek. Voor `C.APPS` is dat Chat met 0,64; nadat de gebruiker Bus installeert met 0,81, is het Bus.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_combined():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Twee scores, één weging

        `combined(quality, engagement, alpha)` krijgt de scores van het kwaliteitsmodel en van het engagementmodel, één per post, en de weging `alpha` van kwaliteit. De functie geeft één samengestelde score per post terug, met `alpha` als weging van kwaliteit en de rest van de weging voor engagement. De loop gaat langs elke plek `i` uit `range(len(quality))`, en `scores.append` voegt de score van post `i` toe aan `scores`. Vervang de `...` door de score van post `i`.

        De check roept `combined(C.QUALITY, C.ENGAGEMENT, 0.3)` aan, de 8 posts van de demo hieronder. Daarna andere wegingen en lijsten.
        """),
        L.task("Geef elke post de kwaliteit keer alpha, plus het engagement keer de rest van de weging."),
    ])
    return


@app.cell(hide_code=True)
def exercise_combined():
    box_combined = L.code_box("combined", lines=6)
    box_combined
    return (box_combined,)


@app.cell(hide_code=True)
def check_combined(box_combined):
    result_combined = L.check("combined", box_combined.value)
    result_combined
    return (result_combined,)


@app.cell(hide_code=True)
def hints_combined():
    L.hints(
        "combined",
        [
            "Stap 4 van het recept: één model per doel, en een weging in code. In `scores.append` begint de score van post `i` met `alpha * quality[i]`; engagement krijgt de weging die overblijft.",
            "De twee wegingen zijn samen 1. Als kwaliteit `alpha` weegt, wat weegt engagement dan, en welk item van `engagement` hoort bij post `i`?",
            "`        scores.append(alpha * quality[i] + ___)`",
        ],
        why="`alpha * quality[i] + (1 - alpha) * engagement[i]` is de score van post `i`. Bij `alpha` 0,3 staan er twee misleidende posts in de top 3; bij 0,8 geen, en er is geen model opnieuw getraind.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Eén weging over de feed

    De app van de stad laat een feed van 8 posts zien, verzonnen door de course. Voor elke post geeft een kwaliteitsmodel hoe betrouwbaar de post is, en een engagementmodel hoe waarschijnlijk een gebruiker op de post reageert, allebei van 0 tot 1. Drie posts zijn misleidend. De slider stelt α in, de weging van kwaliteit, en de feed staat op volgorde van α × kwaliteit + (1 − α) × engagement. Elke balk laat de twee delen van de score van een post zien, zoals de legenda zegt, en de lijn van streepjes markeert de top 3. Er wordt niets getraind terwijl je de slider beweegt.
    """)
    return


@app.cell(hide_code=True)
def demo_alpha():
    L.explore("alpha", "α", lambda _v: C.frame(_v, L.lang), start=0.0, low=0.0, high=1.0, step=0.1, role="",
              task="Zoek de kleinste α zonder misleidende post in de top 3. Wat gebeurt er met het engagement van de top 3?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf α = 0,4 staat er geen misleidende post meer in de top 3, en heeft de top 3 nog gemiddeld een engagement van 0,64, tegen 0,88 bij α = 0. Vanaf α = 0,8 zakt het engagement van de top 3 naar 0,37. Het team kiest welke α goed is, samen met de mensen die de app gebruiken. Met twee modellen kost elke keuze geen training: 11 standen van de slider, 0 trainingsruns. Met één model op één samengestelde loss was elk van de 11 een trainingsrun geweest.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_framing():
    mo.md(r"""
    ### Oefening 3. Welke formulering?

    Een winkel houdt een product in het magazijn bij de stad als het product naar verwachting minstens 100 keer per week verkocht wordt. De 100 verandert elk seizoen: lager in de zomer, als het magazijn meer ruimte heeft, en hoger in de weken voor de feestdagen.
    """)
    return


@app.cell(hide_code=True)
def choice_framing_cell():
    choice_framing = L.choice(
        "framing",
        question="Hoe formuleer je het model, zodat een nieuw seizoen geen nieuwe training nodig heeft?",
        options={
            "classification": "Classificatie: minstens 100 keer verkocht, ja of nee",
            "right": "Regressie: de verkoop per week, met 100 in code",
            "ranking": "Ranking: de producten van meest naar minst verkocht",
            "groups": "Groepen producten die op dezelfde manier verkopen, gevonden zonder labels",
        },
        correct="right",
        why={
            "right": "Het model voorspelt een getal, en de 100 is een regel code: een nieuw seizoen verandert één getal. Dat is het advies van Google voor een threshold die verschuift.",
            "classification": "De 100 zit dan in de labels. Wat gebeurt er als de winkel de grens naar 120 verschuift?",
            "ranking": "Een ranking zegt welk product beter verkoopt dan een ander, niet of een product 100 keer verkocht wordt. Welke formulering voorspelt het aantal verkopen zelf?",
            "groups": "Groepen zonder labels voorspellen helemaal geen verkoop. Welke formulering voorspelt hoe vaak een product verkocht wordt?",
        },
    )
    choice_framing
    return (choice_framing,)


@app.cell(hide_code=True)
def text_exercise_goals():
    mo.md(r"""
    ### Oefening 4. Een nieuwe balans

    De feed van de app van de stad zet posts op volgorde met één model, getraind op één loss: 0,7 × loss op engagement + 0,3 × loss op kwaliteit. De gemeenteraad vraagt om meer nadruk op kwaliteit, en vraagt misschien volgende maand opnieuw.
    """)
    return


@app.cell(hide_code=True)
def choice_goals_cell():
    choice_goals = L.choice(
        "goals",
        question="Welke opzet maakt van zo'n verandering, nu en volgende maand, een verandering in code?",
        options={
            "again": "Hetzelfde model opnieuw trainen, nu met 0,5 voor elke loss",
            "bigger": "Een groter model, getraind op dezelfde samengestelde loss",
            "engagement": "De loss op kwaliteit weglaten en alleen op engagement ranken",
            "right": "Twee modellen, één per doel, en een weging in code",
        },
        correct="right",
        why={
            "right": "Elk model leert één doel, en de weging staat in code: de volgende vraag van de raad is één getal, en er wordt niets getraind. Dat zijn ontkoppelde doelen.",
            "again": "Dat werkt deze maand. Wat gebeurt er als de raad volgende maand opnieuw vraagt?",
            "bigger": "De weging zit nog steeds in de loss, dus betekent een nieuwe balans nog steeds training. Waar hoort de weging te staan?",
            "engagement": "Dan heeft kwaliteit helemaal geen weging meer, en komen de misleidende posts bovenaan. Welke opzet behoudt beide doelen?",
        },
    )
    choice_goals
    return (choice_goals,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In *Betrouwbaar, schaalbaar, onderhoudbaar, aanpasbaar* maakte je van de eisen aan een systeem getallen met een doel. Deze les ging een stap terug: hoe een wens überhaupt een ML-probleem wordt, en hoe de keuze bepaalt wat je later kunt veranderen zonder opnieuw te trainen. [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl) en [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl) gaven de twee formuleringen die je het vaakst tegenkwam. In de volgende les, *Overal dezelfde packages: omgevingen en vaste versies*, zorg je dat de code van een systeem op elke computer hetzelfde draait.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_framing, choice_goals, result_combined, result_next):
    L.wrap_up(result_next, result_combined, choice_framing, choice_goals)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
