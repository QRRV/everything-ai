# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-deployment-patterns. Route: course/route/lessen/systemen.yaml. Checks, de gesimuleerde dag van de app, de stream van station 31 en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-deployment-patterns", "nl")
    C = L.checks   # de gesimuleerde dag van de app voor deelfietsen, de stream van één station, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_serving():
    quick_serving = L.choice(
        "quick-serving",
        question="Welk deel van de spamdienst van de course zorgt dat elk bericht een antwoord krijgt, ook als het model niet kan helpen?",
        options={
            "features": "Features, dat de woorden van een bericht omzet in tellingen",
            "monitoring": "Monitoring, dat de antwoorden bijhoudt en alarm slaat",
            "right": "Serving, dat elk bericht een antwoord geeft: spam, ok of geen score",
            "feedback": "Feedback, dat elk bericht bewaart waarvan een gebruiker het antwoord later verbetert",
        },
        correct="right",
        why={
            "right": "Serving maakt van de kans van het model een antwoord, en antwoordt ook als het model niet kan helpen. Deze les vraagt wanneer het model het eigen deel van dat werk doet: op het moment van elk verzoek, of vooraf voor iedereen.",
            "features": "Features maakt van woorden tellingen voor het model. Welk deel maakt van wat het model zegt een antwoord, en antwoordt ook zonder het model? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "monitoring": "Monitoring houdt de antwoorden bij nadat elk antwoord gegeven is. Welk deel geeft de antwoorden? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "feedback": "Feedback verzamelt verbeteringen om later op te trainen. Welk deel beantwoordt elk bericht zodra het bericht binnenkomt? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_serving
    return (quick_serving,)


@app.cell(hide_code=True)
def quick_check_production():
    quick_production = L.choice(
        "quick-production",
        question="Wat vraagt productie van een model, en onderzoek niet?",
        options={
            "score": "De hoogste score op een test set die hetzelfde blijft",
            "right": "Snelle antwoorden voor veel gebruikers, op data die blijft veranderen",
            "bigger": "Een groter model, want meer gebruikers geeft meer data",
            "notebook": "Uitkomsten opgeschreven in een notebook, zodat anderen het werk kunnen herhalen",
        },
        correct="right",
        why={
            "right": "In productie telt elk afzonderlijk antwoord, voor veel gebruikers tegelijk, en de data blijft binnenkomen. Hoe snel een antwoord er moet zijn, is de eerste vraag van deze les: het antwoord op deze vraag bepaalt wanneer het model het werk doet.",
            "score": "Een vaste test set is waar onderzoek op mikt. Wat moet een model in gebruik elke dag doen voor echte gebruikers? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "bigger": "Meer gebruikers betekent vooral meer verzoeken tegelijk. Wat heeft elk verzoek nodig van het model? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "notebook": "Met de uitkomsten op papier is een onderzoeksproject klaar. Wanneer is een model in productie klaar? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_production
    return (quick_production,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model dat getraind en gemeten is, doet nog niets: het moet voorspellingen maken op een plek waar iemand de voorspellingen gebruikt. *Wanneer* het model de voorspellingen maakt, is de eerste keuze, en deze keuze bepaalt veel van de rest. Een app voor deelfietsen kan 's nachts voor elk lid een suggestie uitrekenen, zodat de app de suggestie meteen laat zien: een station met een vrije fiets, op de tijd dat het lid meestal fietst. Of de app vraagt het model om een suggestie op het moment dat een lid de app opent, zodat de suggestie bij dat moment past. Het eerste is goedkoop en niemand wacht, al is een deel van het werk misschien voor niemand. Het tweede is altijd vers, maar dan moet het model binnen een fractie van een seconde antwoorden, elke keer, terwijl het lid wacht.

    **Een voorbeeld uit het dagelijks leven.** Een bakker bakt het brood om 4 uur 's ochtends voor de hele dag: 's avonds is er brood over, en geen enkel brood is nog vers. Een pizzeria maakt elke pizza als iemand de pizza bestelt: altijd warm, al wacht de klant, en op een drukke avond moet de keuken het bijhouden. Een sushibar met een lopende band zet een nieuw bordje neer zodra iemand een bordje pakt: niemand bestelt, en de keuken reageert op wat er gebeurt. En een broodbakmachine thuis bakt in je eigen keuken, zonder winkel: klein, maar altijd bij de hand.
    """)
    return


@app.cell(hide_code=True)
def predict_used():
    guess_used = L.guess(
        "used",
        question="De app voor deelfietsen heeft 10.000 leden. Om 2 uur 's nachts maakt een batch voor elk lid een suggestie, en de app laat de suggestie zien zodra de app opengaat. Hoeveel van de 10.000 suggesties ziet een lid op een gesimuleerde dag, denk je?",
        options={
            "most": "De meeste: elk lid heeft een suggestie",
            "half": "Ongeveer de helft",
            "quarter": "Tussen 2.000 en 3.000",
            "right": "Minder dan 2.000",
        },
        reveal="**Minder dan 2.000.** Op de gesimuleerde dag openen 1.316 leden de app en zien de eigen suggestie. De andere 8.684, 86,8% van het werk van de nacht, zijn voor niemand gemaakt. En de 60 leden die overdag lid worden, vinden helemaal geen suggestie: de batch kende deze leden nog niet. De dag is een simulatie van de course, met leden die elke dag, elke week of zelden fietsen; een echte app heeft eigen getallen, maar het patroon komt vaak voor.",
    )
    guess_used
    return (guess_used,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Deployment.** **Deployment** is een model in gebruik nemen, op een plek waar mensen of programma's de voorspellingen krijgen. Eén vraag bepaalt de vorm meer dan alle andere: *wanneer* maakt het model een voorspelling? Er zijn drie antwoorden, en een vierde patroon zegt *waar*: op de eigen telefoon van de gebruiker.

        **Batch.** Bij **batch inference** maakt het model op een vast moment een voorspelling voor iedereen, zoals elke nacht, en worden de voorspellingen opgeslagen; de app zoekt er alleen één op. Niemand wacht op het model, een grote batch is goedkoop per voorspelling, en een traag model is minder erg, zolang elke run op tijd klaar is. De prijs: voorspellingen voor wie niet komt, voorspellingen die al uren oud zijn op het moment van zien, en niets voor wie nieuw is sinds de laatste run.

        **Online.** Bij **online inference** maakt het model een voorspelling op het moment dat er een verzoek binnenkomt, uit de invoer van dat moment. De voorspelling is nooit oud en past bij het verzoek, ook voor een nieuwe gebruiker. De prijs: iemand wacht, dus moet het model snel antwoorden, ook op het drukste uur, en elk verzoek kost een modelaanroep.

        **Streaming.** Bij **streaming inference** is de invoer een stream: gebeurtenissen die blijven binnenkomen, zoals verhuringen, posities of betalingen. Na elke gebeurtenis maakt het model of een regel een voorspelling, vaak uit wat er in een tijdvenster gebeurde, zoals de laatste 15 minuten, en als het ertoe doet, gebeurt er iets, zonder dat iemand erom vraagt. Streaming reageert binnen seconden op wat er nu gebeurt, maar moet elke gebeurtenis bijhouden, en kent alleen wat er tot nu toe binnenkwam.

        **Op het apparaat.** Het model kan ook op de eigen telefoon of laptop van de gebruiker draaien. Dan werkt het model zonder netwerk, antwoordt het zonder omweg via een server, en blijft de data bij de gebruiker. De prijs: het model moet klein genoeg zijn voor de telefoon, en elke telefoon draait een eigen kopie, en elke kopie moet apart bijgewerkt worden. Op de telefoon kan het model nog steeds op elk van de drie manieren werken: bij een verzoek, na elke gebeurtenis, of in een batch 's nachts.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Gemaakt tegenover gebruikt.** Bij een batch zeggen twee getallen veel: hoeveel voorspellingen er gemaakt worden tegenover hoeveel er gebruikt worden, en hoe oud een voorspelling is op het moment van gebruik. Online maakt evenveel voorspellingen als er verzoeken zijn, elk nul uur oud; daar is de vraag hoe snel elke voorspelling komt. Veel systemen combineren de patronen: een batch voor wat kan wachten, en online voor wat niet kan wachten.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **deployment** | | een model in gebruik nemen, op een plek waar mensen of programma's de voorspellingen krijgen | |
        | **batch inference** | | vooraf voorspellingen maken voor iedereen, op een vast moment zoals elke nacht, en de voorspellingen opslaan | |
        | **online inference** | | een voorspelling maken op het moment dat er een verzoek binnenkomt, uit de invoer van dat moment, terwijl iemand wacht | |
        | **streaming inference** | | een voorspelling maken na elke gebeurtenis van een stream, zoals elke verhuring bij een station, zonder dat iemand erom vraagt | |
        """),
        L.watch_out("*Streaming* betekent hier een stroom gebeurtenissen die binnenkomt. Van een chatbot die de woorden al laat zien tijdens het schrijven, zegt men ook dat het antwoord streamt: daar gaat het antwoord in stukjes naar buiten, maar het blijft één online verzoek."),
        L.watch_out("*Op het apparaat* betekent hier de eigen telefoon of laptop van de gebruiker. In code voor neural networks betekent *device* iets anders: de chip waarop een model draait, zoals de processor of een videokaart."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Kiezen wanneer een model de voorspellingen maakt:

    1. Zoek uit wanneer de invoer bekend is: de avond van tevoren, pas op het moment van het verzoek, of als een stream van gebeurtenissen.
    2. Vraag hoe snel het antwoord nodig is, en wie erop wacht: iemand in de app, een programma, of niemand.
    3. Tel bij een batch hoeveel voorspellingen er gebruikt worden, en hoe oud de voorspellingen dan zijn.
    4. Kies: alles vooraf bekend en het antwoord mag wachten, batch; iemand wacht op een antwoord op nieuwe invoer, online; gebeurtenissen blijven binnenkomen en er moet iets op gebeuren, streaming; geen netwerk, of data die bij de gebruiker moet blijven, op het apparaat.
    5. Combineer waar het loont: een batch voor wat kan wachten, online of streaming voor wat niet kan wachten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een kleine dag: de batch van 2:00 maakt een suggestie voor de leden a tot en met e. Overdag openen b om 8:00, d om 8:30, f om 12:00 en b nog een keer om 18:00 de app; f werd 's ochtends lid. De cel loopt langs elke keer dat de app openging en kijkt alleen naar leden uit de batch: voor elk van deze keren telt de cel op hoe oud de suggestie is. Voer de cel uit.
    """)
    return


@app.cell
def example_small_day():
    batch = ["a", "b", "c", "d", "e"]
    opened = ["b", "d", "f", "b"]
    hours = [8.0, 8.5, 12.0, 18.0]
    seen = set(opened)
    print("leden die de app openden:", len(seen))
    total = 0
    count = 0
    for i in range(len(opened)):
        if opened[i] in batch:
            total = total + hours[i] - 2
            count = count + 1
    print("keer geopend met een suggestie:", count)
    print("gemiddelde leeftijd in uren:", total / count)
    print("modelaanroepen online:", len(opened))
    return batch, count, hours, i, opened, seen, total


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Drie leden openden de app, maar alleen b en d hadden een suggestie: 2 van de 5 zijn gezien, en de suggesties van a, c en e zijn voor niemand gemaakt. b opende de app twee keer en zag twee keer dezelfde suggestie. f is nieuw sinds de nacht en kreeg geen suggestie. De suggesties waren 6 uur, 6,5 uur en 16 uur oud op het moment van zien, gemiddeld 9,5. Online had het model 4 voorspellingen gemaakt, elk op het moment dat de app openging, ook één voor f. Oefening 1 telt de suggesties die niemand zag.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_unused():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Suggesties voor niemand

        `unused(batch, opened)` krijgt de leden waarvoor de batch een suggestie maakte, en het lid van elke keer dat de app openging, met herhalingen en met nieuwe leden. De functie geeft terug hoeveel suggesties niemand zag. `seen = set(opened)` is een set van iedereen die de app opende, de loop gaat met `user` de leden van `batch` langs, en `used` telt de leden die de app openden, zodat `len(batch) - used` is wat niemand zag. Vervang de `...` door twee regels die deze leden tellen.

        De check roept `unused(C.BATCH, C.OPENED)` aan: de 10.000 leden van de nacht en de 2.473 keer dat de app openging op de gesimuleerde dag. Daarna het uitgewerkte voorbeeld, dat 3 moet geven, een dag waarop niemand de app opende, een dag waarop alleen nieuwe leden dat deden, een batch van niemand, en een dag met alleen herhalingen.
        """),
        L.task("Tel hoeveel leden uit de batch de app openden, en geef terug hoeveel leden dat niet deden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unused():
    box_unused = L.code_box("unused", lines=7)
    box_unused
    return (box_unused,)


@app.cell(hide_code=True)
def check_unused(box_unused):
    result_unused = L.check("unused", box_unused.value)
    result_unused
    return (result_unused,)


@app.cell(hide_code=True)
def hints_unused():
    L.hints(
        "unused",
        [
            "Stap 3 van het recept. Voor elke `user` van de batch stelt de loop één vraag: opende dit lid de app? Zo ja, dan gaat `used` 1 omhoog, zoals `count` in het uitgewerkte voorbeeld.",
            "Met `in` vraagt Python of een waarde bij de waarden van een set hoort: `\"b\" in set([\"b\", \"d\", \"b\"])` is `True`, en de set bevat `\"b\"` één keer. Welke set hierboven bevat iedereen die de app opende?",
            "`        if user ___:`",
        ],
        why="`if user in seen:` vraagt van elk lid van de batch of dat lid de app opende, en `used = used + 1` telt het lid één keer, hoe vaak de app ook openging. Nieuwe leden zitten niet in `batch`, dus telt de functie nieuwe leden nooit mee. Een set beantwoordt `in` meteen, zelfs bij 10.000 leden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_recent():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het laatste kwartier

        Hetzelfde bedrijf heeft een stream: elke verhuring bij een station komt binnen als gebeurtenis, met de minuut van de dag waarop de verhuring gebeurde, zoals 480 voor 8:00. Na elke verhuring telt een regel de verhuringen van de laatste 15 minuten; vanaf 10 gaat er een vrachtwagen met fietsen heen. `recent(times, now, minutes)` krijgt de minuten van verhuringen, de minuut `now` en de lengte van het venster in minuten. De functie geeft terug hoeveel verhuringen in het venster vallen: na `now - minutes`, en op `now` of eerder. De lijst mag na `now` doorgaan, zoals wanneer je een dag achteraf opnieuw afspeelt. De loop gaat met `t` de verhuringen langs. Vervang de `...` door twee regels die de verhuringen in het venster tellen in `count`.

        De check roept `recent(C.TIMES, 1050, 15)` aan: de 83 verhuringen van een gesimuleerde dag bij station 31, allemaal, met 17:30 als `now`. Daarna verhuringen om de 5 minuten, een venster zonder verhuringen, een venster dat langer is dan de dag tot nu toe, helemaal geen verhuringen, en verhuringen na `now`.
        """),
        L.task("Tel de verhuringen na `now - minutes`, en op `now` of eerder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recent():
    box_recent = L.code_box("recent", lines=6)
    box_recent
    return (box_recent,)


@app.cell(hide_code=True)
def check_recent(box_recent):
    result_recent = L.check("recent", box_recent.value)
    result_recent
    return (result_recent,)


@app.cell(hide_code=True)
def hints_recent():
    L.hints(
        "recent",
        [
            "Stap 1 van het recept: de invoer is een stream. De voorwaarde heeft twee delen met `and` ertussen, en elk deel vergelijkt `t` met een getal: het ene deel houdt de verhuringen van meer dan `minutes` minuten voor `now` buiten, het andere de verhuringen die nog niet gebeurd zijn.",
            "Een verhuring op precies `now` is de verhuring die net binnenkwam, dus telt deze verhuring mee; een verhuring na `now` is op dat moment nog niet gebeurd.",
            "`        if t > now - minutes and t ___ now:`",
        ],
        why="`if t > now - minutes and t <= now:` houdt de verhuringen van de laatste 15 minuten over, met de verhuring van deze minuut erbij: om 17:30 zijn dat 17 verhuringen, vanaf 17:16.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_patterns():
    mo.md(r"""
    ### Oefening 3. Welk patroon?

    Acht toepassingen van een model bij het deelfietsbedrijf. Zet elke toepassing onder het patroon dat het best past; moet het model op de telefoon zelf draaien, kies dan Op het apparaat.
    """)
    return


@app.cell(hide_code=True)
def sort_patterns_cell():
    sort_patterns = L.sort("patterns")
    sort_patterns
    return (sort_patterns,)


@app.cell(hide_code=True)
def check_patterns(sort_patterns):
    result_patterns = L.check("patterns", sort_patterns)
    result_patterns
    return (result_patterns,)


@app.cell(hide_code=True)
def hints_patterns():
    L.hints(
        "patterns",
        [
            "Stap 4 van het recept. Vraag bij elk gebruik: is alles vooraf bekend, wacht er iemand op het antwoord, blijven er gebeurtenissen binnenkomen, of is er misschien geen netwerk?",
            "Twee voorspellingen kunnen de avond van tevoren al klaar zijn, en twee hangen af van wat een fietser net intypte. Welke twee reageren op gebeurtenissen, zonder dat iemand iets vraagt?",
            "Batch: vrachtwagens en \\_\\_\\_. Online: prijs en \\_\\_\\_. Streaming: een fiets die rijdt en \\_\\_\\_. Op het apparaat: stem en \\_\\_\\_.",
        ],
        why="**Batch:** de vrachtwagens 's nachts en de mail op maandag, want alles is vooraf bekend en niemand wacht. **Online:** de prijs van een rit en de helpdesk, want iemand wacht op een antwoord op wat er net ingetypt is. **Streaming:** een fiets die rijdt en een station dat bijna leeg is, want gebeurtenissen blijven binnenkomen en er moet iets op gebeuren. **Op het apparaat:** de stem, want de opname moet op de telefoon blijven, en een val, want er is misschien geen netwerk.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hourly():
    mo.md(r"""
    ### Oefening 4. Elk uur

    Leden klagen dat de suggestie van de nacht 's avonds verouderd is. Het team laat de batch voortaan elk uur draaien, vanaf 2 uur 's nachts.
    """)
    return


@app.cell(hide_code=True)
def choice_hourly_cell():
    choice_hourly = L.choice(
        "hourly",
        question="Wat verandert er?",
        options={
            "all": "Elk lid, ook een nieuw lid, ziet nu een suggestie van hooguit een uur oud",
            "right": "Versere suggesties, 24 keer zoveel werk, en nieuwe leden wachten nog op een run",
            "cheaper": "Ongeveer evenveel werk: de batch draait alleen voor leden die de app openen",
            "online": "Hetzelfde als online: een uur is zo kort dat niemand ooit het verschil kan merken",
        },
        correct="right",
        why={
            "right": "Elke run maakt een suggestie voor elk lid, dus maken 24 runs 24 keer zoveel suggesties, terwijl leden de app even vaak openen als eerst: bijna alle extra suggesties zijn voor niemand. Een suggestie is verser op het moment van zien. En een lid dat om 14:20 lid wordt, zit pas in de run van 15:00 in een batch. De demo laat de getallen zien.",
            "all": "Een lid dat overdag lid wordt, zit tot de volgende run in geen enkele batch. Wanneer openen nieuwe leden de app?",
            "cheaper": "Een batch weet niet wie de app gaat openen: bij elke run maakt de batch een suggestie voor elk lid. Hoeveel suggesties maakt de batch in 24 runs?",
            "online": "Online maakt een suggestie op het moment van het verzoek, ook voor een lid dat een minuut geleden lid werd. Wat geeft de batch van elk uur aan dat lid?",
        },
    )
    choice_hourly
    return (choice_hourly,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Hoe vaak de batch draait

    De demo laat de batch van de app 1, 2, 4, 8 of 24 keer per dag draaien, op dezelfde gesimuleerde dag, vanaf 2 uur 's nachts met gelijke stappen. Bij elke keuze laat de afbeelding zien hoeveel suggesties het model maakt, hoeveel suggesties een lid ziet, en hoe oud een suggestie is op het moment van zien. Ernaast staat online, met één modelaanroep voor elke keer dat de app opengaat.
    """)
    return


@app.cell(hide_code=True)
def demo_runs():
    L.explore("runs", "runs per dag", lambda _v: C.frame(int(_v), L.lang), start=1, values=C.RUNS, role="",
              task="Laat de batch vaker draaien: wat gebeurt er met het werk, met de leeftijd van een suggestie, en met de nieuwe leden?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Eén keer per dag maakt de batch 10.000 suggesties; leden zien er 1.316, gemiddeld 11,8 uur oud. Elk uur maakt de batch er 240.491, ongeveer 24 keer zoveel, en leden zien er 2.379: 1,0%. Een suggestie is dan gemiddeld een halfuur oud, maar 60 keer vindt een lid nog steeds geen suggestie, want elk nieuw lid opent de app meteen na het aanmelden, nog voor de volgende run. Online doet 2.473 aanroepen, elk op het moment van het verzoek en ook voor de nieuwe leden: bijna honderd keer minder dan de batch van elk uur, maar bij elke aanroep wacht een lid.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) was serving het deel dat elk bericht een antwoord geeft. Deze les vroeg *wanneer* het model het werk doet: vooraf voor iedereen, bij elk verzoek, na elke gebeurtenis, of op de eigen telefoon van de gebruiker. In de volgende les, *Je model achter een API*, zet je een model online: een kleine server die op verzoeken wacht en elk verzoek beantwoordt. Modellen op telefoons, laptops en sensoren komen terug in *Aan de rand: apparaten, sensoren en de cloud*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hourly, result_patterns, result_recent, result_unused):
    L.wrap_up(result_unused, result_recent, result_patterns, choice_hourly)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
