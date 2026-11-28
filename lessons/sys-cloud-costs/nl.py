# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-cloud-costs. Route: course/route/lessen/systemen.yaml. Checks, de prijzen, de onderbrekingen van de spot-run en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-cloud-costs", "nl")
    C = L.checks   # de prijzen, de onderbrekingen van de spot-run, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl) en [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_job():
    quick_job = L.choice(
        "quick-job",
        question="Een job duurt 30 uur op een GPU die je kent. Een gehuurde GPU is 3 keer zo snel en kost 6 dollar per uur. Wat kost de job daarop?",
        options={
            "per-hour": "180 dollar: 30 uur voor 6 dollar per uur",
            "right": "60 dollar: 10 uur voor 6 dollar per uur",
            "times": "540 dollar: 30 uur, keer 3 voor de snelheid, keer 6 dollar",
            "divide": "2 dollar: 6 dollar gedeeld door 3",
        },
        correct="right",
        why={
            "right": "Op een GPU die 3 keer zo snel is duurt de job 30 / 3 = 10 uur, en 10 uur voor 6 dollar kost 60 dollar. Deze les voegt toe dat dezelfde GPU meer dan één prijs per uur heeft, afhankelijk van hoe je de GPU huurt.",
            "per-hour": "De prijs is per uur, maar op een snellere GPU duurt de job minder uren. Hoeveel? Zie [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).",
            "times": "Een snellere GPU maakt de job korter, niet langer. Wat deel je door de snelheid? Zie [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).",
            "divide": "De job duurt nog steeds uren, elk voor 6 dollar. Hoeveel uur duurt de job op de snellere GPU? Zie [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl).",
        },
    )
    quick_job
    return (quick_job,)


@app.cell(hide_code=True)
def quick_check_checkpoint():
    quick_checkpoint = L.choice(
        "quick-checkpoint",
        question="Een training stopt na epoch 6 van 10. Wat moet het checkpoint bevatten om vanaf epoch 7 verder te gaan alsof er niets gebeurd is?",
        options={
            "right": "De state_dict van het model en van de optimizer, en de epoch",
            "model": "Alleen de state_dict van het model, want daarin staat elke parameter",
            "class": "De code van de class van het model, en de loss van elke epoch tot nu toe voor de curve",
            "data": "De hele dataset, in de volgorde van de epochs",
        },
        correct="right",
        why={
            "right": "De parameters alleen zijn niet genoeg: de optimizer houdt eigen getallen bij voor elke parameter, en de loop moet weten waar de training was. Met alle drie gaat een run die stopt verder vanaf het laatste checkpoint, en zo worden goedkope machines bruikbaar: machines die op elk moment kunnen stoppen.",
            "model": "De parameters zijn het grootste deel. De optimizer houdt ook eigen getallen bij, en de loop moet weten waar de training was. Wat hoort er nog meer in? Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "class": "De code blijft in je eigen bestanden, en de curve is niet nodig om verder te trainen. Wat heeft de training nodig om door te gaan? Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
            "data": "De dataset verandert niet tijdens het trainen. Wat verandert er wel, en moet bewaard worden? Zie [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl).",
        },
    )
    quick_checkpoint
    return (quick_checkpoint,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model trainen kan een GPU dagenlang bezighouden, en deze GPU-uren zijn het grootste deel van de rekening. In [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl) rekende je uit wat een hele job kost bij één prijs per uur. Maar dezelfde GPU in hetzelfde datacenter heeft meer dan één prijs: de prijs hangt af van hoe je de GPU huurt. Betaal per uur wanneer je wilt, en je betaalt het meest. Beloof de GPU een jaar te huren, en het uur wordt veel goedkoper, ook de uren die je niet gebruikt. Neem een GPU die nu niemand gebruikt, en het uur kost een fractie, terwijl de cloud de GPU op elk moment kan terugnemen. Voor een model achter een API is er een vierde manier: geen eigen server, en een prijs per verzoek. Een goede keuze kan hetzelfde werk drie keer goedkoper maken; met een slechte keuze komt het werk nooit af.

    **Een voorbeeld uit het dagelijks leven.** Een hotelkamer per nacht aan de balie kost het meest. Een appartement dat je voor een jaar huurt kost minder per nacht, maar je betaalt ook de nachten dat je weg bent. Een kamer die deze nacht leeg staat kan goedkoop weg, op voorwaarde dat je vertrekt als een gast die de volle prijs betaalt aankomt. En een taxi kost per rit: goedkoper dan een eigen auto als je een paar keer per maand rijdt, duurder als je elke dag rijdt.
    """)
    return


@app.cell(hide_code=True)
def predict_spot():
    guess_spot = L.guess(
        "spot",
        question="Een training duurt 100 uur op een GPU die on-demand 4 dollar per uur kost. Als spot instance kost dezelfde GPU 1,20 dollar per uur, alleen neemt de cloud de GPU ongeveer eens per 10 uur terug. Je slaat elk uur een checkpoint op, wat 3 minuten duurt, en opnieuw starten op een nieuwe machine duurt 12 minuten. Wat kost de run als spot, denk je?",
        options={
            "same": "Ongeveer 400 dollar, net als on-demand, want het verloren werk eet de korting op",
            "price": "Ongeveer 120 dollar: 100 uur voor 1,20 dollar",
            "right": "Ongeveer 135 dollar: wat meer dan 100 uur voor 1,20 dollar",
            "more": "Meer dan 400 dollar, want elke onderbreking begint de run opnieuw",
        },
        reveal="**Ongeveer 135 dollar.** De run betaalt ongeveer 113 uur: 100 uur training, 5 uur om 100 checkpoints van 3 minuten op te slaan, en ongeveer 11 onderbrekingen die elk gemiddeld een half uur werk weggooien en 12 minuten kosten om opnieuw te starten. Voor 1,20 dollar per uur is dat 135,48 dollar, een derde van de 400 dollar on-demand. Zonder checkpoints zou elke onderbreking alles sinds het begin weggooien, en zou de run nauwelijks opschieten. De demo hieronder laat zien hoe vaak je opslaat.",
    )
    guess_spot
    return (guess_spot,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **On-demand, reserved en spot.** Een machine die je in de cloud huurt heet een **instance**. Een **on-demand instance** start en stop je wanneer je wilt, en je betaalt elk uur of elke seconde dat de instance draait: de hoogste prijs per uur, zonder beloftes. Voor een **reserved instance** beloof je een of drie jaar te betalen, gebruikt of niet; in ruil is het uur veel goedkoper, bij AWS tot 72% korting voor drie jaar. Een **spot instance** draait op capaciteit die op dat moment vrij is, voor een fractie van de prijs: bij AWS tot 90% korting, bij Google Cloud 60 tot 91%. Als de cloud de capaciteit terug nodig heeft neemt de cloud de machine terug, met een waarschuwing van twee minuten bij AWS en van 30 seconden bij Google Cloud. Deze kortingen komen van de pagina's van de aanbieders op 28 november 2026. In deze les kost een voorbeeld-GPU 4 dollar per uur on-demand, 2,40 dollar reserved (40% korting) en 1,20 dollar als spot (70% korting).
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Welke manier wanneer.** Reserved betaal je voor elk uur van de maand, alle 730, gebruikt of niet: 1.752 dollar voor 2,40 dollar per uur. On-demand kost 4 dollar voor elk gebruikt uur. Tot 438 uur per maand is on-demand goedkoper; vanaf daar reserved: met 40% korting vanaf 60% van de uren. Spot is per uur het goedkoopst, maar alleen voor werk dat kan stoppen en later verder kan, zoals training die checkpoints opslaat, of een grote klus in delen die elk opnieuw kunnen beginnen. Een dienst die op elk moment moet antwoorden draait nooit alleen op spot.

        **Checkpoints op spot.** Als de cloud een spot-machine terugneemt is alles sinds het laatste checkpoint weg: de training gaat verder vanaf dat checkpoint, op een nieuwe machine. Een checkpoint bevat, zoals in [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl), de state_dict van het model en van de optimizer, en de epoch. Vaker opslaan gooit bij elke onderbreking minder weg, maar ook opslaan kost tijd, en ook deze tijd wordt betaald.

        **Serverless inference.** Bij **serverless inference** heb je geen eigen server: de aanbieder start je model als er verzoeken binnenkomen, en je betaalt per verzoek of per seconde rekenwerk. Zonder verzoeken betaal je niets; het eerste verzoek na een stille periode wacht terwijl het model laadt, van seconden voor een klein model tot minuten voor een groot model. Een vaste server kost elk uur hetzelfde, druk of stil. Bij weinig verzoeken is serverless goedkoper, bij veel verzoeken een vaste server, en ergens daartussen kosten beide per dag even veel.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **instance** | | een machine die je in de cloud huurt | |
        | **on-demand instance** | | een instance die je start en stopt wanneer je wilt, betaald per uur of seconde dat de instance draait, voor de hoogste prijs | |
        | **reserved instance** | | een instance waarvoor je een of drie jaar belooft te betalen, gebruikt of niet, voor een lagere prijs per uur | |
        | **spot instance** | | een instance op vrije capaciteit, voor een fractie van de prijs; de cloud kan de instance op elk moment terugnemen | |
        | **serverless inference** | | een model draaien zonder eigen server, betaald per verzoek of per seconde | |
        """),
        L.watch_out("Een reserved instance wordt elk uur betaald, ook een uur waarin de instance stilstaat. Reserveer alleen wat bijna altijd draait, en alleen zo lang als je zeker weet: een project dat na drie maanden stopt betaalt de rest van het jaar toch."),
        L.watch_out("Een spot-machine die wordt teruggenomen neemt de eigen schijf mee, en een checkpoint op deze schijf gaat ook mee. Sla checkpoints op in opslag die blijft als de machine verdwijnt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Kiezen hoe je voor compute betaalt:

    1. Reken uit hoeveel uur per maand de GPU echt draait.
    2. Reserveer als dat meer is dan het deel waarbij reserved en on-demand even veel kosten (60% van de uren bij 40% korting), voor minstens zo lang als de belofte; betaal anders on-demand.
    3. Zet werk dat kan stoppen en doorgaan op spot instances: training die vaak genoeg checkpoints opslaat, in opslag die blijft, en grote klussen in delen die elk opnieuw kunnen beginnen.
    4. Vergelijk voor een model achter een API een dag: de verzoeken keer de prijs per verzoek voor serverless, tegen 24 uur van een server. Onder het aantal verzoeken waarbij beide even veel kosten is serverless goedkoper; daarboven een vaste server.
    5. Zet de datum bij elke prijs, en kijk later opnieuw: prijzen en kortingen veranderen vaak.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een team gebruikt de voorbeeld-GPU. Vanaf hoeveel uur per maand is reserved goedkoper dan on-demand? In de cel is `reserved` wat een maand reserved kost: alle 730 uur voor 2,40 dollar. De loop telt `hours` op zolang zoveel uur on-demand niet meer kost dan een maand reserved. Voer de cel uit.
    """)
    return


@app.cell
def example_reserved():
    on_demand = 4.00              # dollar per GPU-uur, voor elk uur dat je gebruikt
    reserved = 2.40 * 730         # 40% korting, maar voor alle 730 uur van een maand
    hours = 0
    while hours * on_demand <= reserved:
        hours = hours + 1
    print("reserved per maand:", reserved, "dollar")
    print("reserved is goedkoper vanaf", hours, "uur per maand")
    return hours, on_demand, reserved


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij 438 uur kost on-demand 438 × 4 = 1.752 dollar, precies zoveel als reserved, dus gaat de loop door; bij 439 uur kost on-demand 1.756 dollar, en is reserved goedkoper. 439 van de 730 uur is 60%: met 40% korting loont reserved als de GPU meer dan 60% van de tijd draait. Oefening 2 vindt zo'n punt voor serverless en een vaste server; oefening 1 telt eerst wat een spot-run betaalt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hours_paid():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wat een spot-run betaalt

        `hours_paid(hours, every, stops)` krijgt de uren training die een run nodig heeft, de uren training tussen twee checkpoints, en een lijst onderbrekingen: voor elke keer dat de cloud de machine terugnam, hoeveel uur training er op dat moment gedaan was. De functie geeft terug hoeveel uur de run betaalt, zonder de minuten om op te slaan en opnieuw te starten. `paid` begint bij `hours`. Voor elke `stop` zet de `while`-loop `saved` steeds `every` hoger, zolang het volgende checkpoint niet later kwam dan de onderbreking: `saved` eindigt bij de uren van het laatste checkpoint tot aan de onderbreking. Vervang de `...` door de uren werk die de onderbreking weggooit.

        `C.STOPS` bevat de tien onderbrekingen van een run van 100 uur: 9,5; 17,25; 26; 35,5; 43,75; 52; 61,5; 69,25; 78 en 87,5. De check roept `hours_paid(100, 2, C.STOPS)` aan: een checkpoint elke 2 uur. Daarna een checkpoint elke 4 uur en elk uur, een run zonder onderbrekingen, onderbrekingen precies op een checkpoint, een checkpoint elk half uur, en een onderbreking die komt als er nog geen checkpoint is.
        """),
        L.task("Tel de weggegooide uren van alle onderbrekingen op."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours_paid():
    box_hours_paid = L.code_box("hours_paid", lines=9)
    box_hours_paid
    return (box_hours_paid,)


@app.cell(hide_code=True)
def check_hours_paid(box_hours_paid):
    result_hours_paid = L.check("hours_paid", box_hours_paid.value)
    result_hours_paid
    return (result_hours_paid,)


@app.cell(hide_code=True)
def hints_hours_paid():
    L.hints(
        "hours_paid",
        [
            "Het idee, onder *Checkpoints op spot*: een onderbreking gooit alles sinds het laatste checkpoint weg. Bij elke `stop` moet `paid` met deze uren groeien. Waar viel het laatste checkpoint, en waar de onderbreking?",
            "In `C.STOPS` kwam de eerste onderbreking bij 9,5 uur. Met een checkpoint elke 2 uur lag het laatste checkpoint tot dan bij 8 uur: 1,5 uur werk is weg. Welke twee namen bevatten 9,5 en 8 in de loop?",
            "`        paid = paid + stop ___`",
        ],
        why="`paid = paid + stop - saved` telt de uren tussen het laatste checkpoint en de onderbreking op. Met een checkpoint elke 2 uur gooien de tien onderbrekingen van `C.STOPS` 1,5 + 1,25 + 0 + 1,5 + 1,75 + 0 + 1,5 + 1,25 + 0 + 1,5 = 10,25 uur weg: 110,25 betaalde uren. Een onderbreking bij 26 uur kost niets, want het checkpoint van 26 uur was net opgeslagen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_break_even():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wanneer serverless meer kost

        Een klein model beantwoordt verzoeken achter een API. Op serverless kost een verzoek 0,00035 dollar, ongeveer 1,5 seconde van een GPU; een vaste server met een GPU kost 0,75 dollar per uur, 18 dollar per dag. `break_even(per_request, per_day)` geeft het kleinste aantal verzoeken per dag terug waarbij serverless meer kost dan de server. `n` telt op vanaf 0, en de `while`-loop gaat door zolang `n` verzoeken niet meer kosten dan de server. Vervang de `...` door deze vraag.

        De check roept `break_even(0.00035, 18.0)` aan. Daarna een verzoek van 0,5 dollar en een server van 12 dollar, waarbij 24 verzoeken precies even veel kosten als de server; een verzoek van 0,0003 dollar en een server van 20 dollar; een verzoek dat duurder is dan de hele server; en een server die niets kost.
        """),
        L.task("Vind het eerste aantal verzoeken waarbij serverless meer kost."),
    ])
    return


@app.cell(hide_code=True)
def exercise_break_even():
    box_break_even = L.code_box("break_even", lines=6)
    box_break_even
    return (box_break_even,)


@app.cell(hide_code=True)
def check_break_even(box_break_even):
    result_break_even = L.check("break_even", box_break_even.value)
    result_break_even
    return (result_break_even,)


@app.cell(hide_code=True)
def hints_break_even():
    L.hints(
        "break_even",
        [
            "In het uitgewerkte voorbeeld gaat de loop door zolang `hours` on-demand niet meer kosten dan een maand reserved. Hier kosten `n` verzoeken `n * per_request` dollar op serverless. Waarmee vergelijkt de loop dat, en hoe?",
            "Niet meer dan is `<=` in Python, zoals in het uitgewerkte voorbeeld. Met 24 verzoeken van 0,5 dollar kost serverless 12 dollar, precies zoveel als een server van 12 dollar: gaat de loop daar door?",
            "`    while n * per_request ___:`",
        ],
        why="`while n * per_request <= per_day:` gaat door zolang serverless niet meer kost dan de server, en stopt bij de eerste `n` waarbij serverless meer kost. Met 0,00035 dollar per verzoek en 18 dollar per dag: 18 / 0,00035 = 51.428,57, dus kost serverless meer vanaf 51.429 verzoeken per dag.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fits():
    mo.md(r"""
    ### Oefening 3. Welke manier past?

    Acht klussen, elk met de manier van betalen die er het best bij past: on-demand, reserved, spot of serverless. Zet elke klus in de juiste kolom.
    """)
    return


@app.cell(hide_code=True)
def sort_fits_cell():
    sort_fits = L.sort("fits")
    sort_fits
    return (sort_fits,)


@app.cell(hide_code=True)
def check_fits(sort_fits):
    result_fits = L.check("fits", sort_fits)
    result_fits
    return (result_fits,)


@app.cell(hide_code=True)
def hints_fits():
    L.hints(
        "fits",
        [
            "Het recept. Vraag bij elke klus: hoe lang duurt de klus, mag de klus halverwege stoppen, en draait de klus elk uur of af en toe?",
            "In elke kolom horen twee klussen.",
            "On-demand: de test op het laatste moment en \\_\\_\\_. Reserved: de dienst voor drie jaar en \\_\\_\\_. Spot: de run van drie dagen en \\_\\_\\_. Serverless: de demo en \\_\\_\\_.",
        ],
        why="**On-demand:** de test op het laatste moment, want stoppen mag niet, en de week om een idee te proberen, want niemand kan een jaar beloven. **Reserved:** de dienst voor drie jaar en het team dat elke dag traint, want deze GPU's draaien bijna elk uur. **Spot:** de run met elk uur een checkpoint en de foto's in delen, want na een onderbreking gaat het werk gewoon verder. **Serverless:** de demo en het formulier, met zo weinig verzoeken dat een server vooral stil zou staan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rhythm():
    mo.md(r"""
    ### Oefening 4. Te zelden een checkpoint

    Een run op spot instances slaat elke 12 uur een checkpoint op. Gemiddeld neemt de cloud de machine eens per 6 uur terug.
    """)
    return


@app.cell(hide_code=True)
def choice_rhythm_cell():
    choice_rhythm = L.choice(
        "rhythm",
        question="Wat gebeurt er?",
        options={
            "discount": "De run is af met 70% korting, want alleen de prijs per uur telt",
            "right": "De run schiet nauwelijks op: de meeste onderbrekingen komen eerder dan een checkpoint",
            "double": "De run duurt ongeveer twee keer zo lang, want de helft van het werk gaat verloren",
            "wait": "De cloud wacht tot het volgende checkpoint is opgeslagen, en neemt de machine pas daarna terug",
        },
        correct="right",
        why={
            "right": "Met gemiddeld elke 6 uur een onderbreking zijn 12 uur zonder onderbreking zeldzaam: de meeste onderbrekingen komen eerder dan het eerste checkpoint, en elke keer begint de run opnieuw vanaf het laatste checkpoint, vaak het begin zelf. Sla veel vaker op dan de onderbrekingen komen: in de demo, met elke 10 uur een onderbreking, ongeveer elk uur. Een waarschuwing van twee minuten of 30 seconden is genoeg om een klein checkpoint op te slaan, zelden een groot checkpoint.",
            "discount": "De prijs per uur is lager, maar bij elke onderbreking is het werk sinds het laatste checkpoint weg. Hoeveel onderbrekingen komen er eerder dan het eerste checkpoint?",
            "double": "Dat zou kloppen als elke onderbreking een beetje kost. Hoe ver komt de run meestal tot de volgende onderbreking, met 12 uur tussen checkpoints en elke 6 uur een onderbreking?",
            "wait": "De cloud geeft hoogstens twee minuten waarschuwing en neemt de machine dan, opgeslagen of niet. Wat gebeurt er met de uren sinds het laatste checkpoint?",
        },
    )
    choice_rhythm
    return (choice_rhythm,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Hoe vaak opslaan?

    Een spot-run van 100 uur voor 1,20 dollar per uur, waarin de cloud de machine gemiddeld eens per 10 betaalde uren terugneemt. Een checkpoint opslaan duurt 3 minuten, opnieuw starten 12 minuten, en elke onderbreking gooit gemiddeld de helft van de tijd tussen twee checkpoints weg. Kies hoeveel uur training er tussen twee checkpoints zit, en zie wat de run betaalt.
    """)
    return


@app.cell(hide_code=True)
def demo_every():
    L.explore("every", "uren tussen checkpoints", lambda _v: C.frame(_v, L.lang), start=4,
              values=C.DEMO_VALUES, role="", task="Ga de zes waarden langs: bij welke waarde betaalt de run de minste uren, en waarom kosten beide uiteinden meer?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met elk kwartier een checkpoint kost opslaan 20 uur, veel meer dan de 1,6 uur die de onderbrekingen weggooien; met elke 8 uur een checkpoint gooien de onderbrekingen 69,4 uur weg. Daartussen ligt het laagste punt: elk uur een checkpoint, 112,9 betaalde uren en 135,48 dollar, ongeveer een derde van de 400 dollar on-demand. De beste afstand is korter als opslaan snel gaat of de onderbrekingen vaak komen, en langer als opslaan lang duurt of de onderbrekingen zelden komen.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De computer van een ander: de cloud](/course/open/sys-cloud?lang=nl) had een job één prijs per uur; nu heeft hetzelfde uur drie prijzen, en serverless een prijs per verzoek. Een spot instance loont alleen met de checkpoints van [Een model opslaan en laden](/course/open/torch-checkpoints?lang=nl): de state_dict van het model en van de optimizer, en de epoch, vaak opgeslagen, in opslag die blijft. In de volgende les, *MLOps: hoe ver is een team?*, komen de stappen van deze module samen in één kringloop die uit zichzelf draait.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rhythm, result_break_even, result_fits, result_hours_paid):
    L.wrap_up(result_hours_paid, result_break_even, result_fits, choice_rhythm)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
