# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-safety-field. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Artificial Intelligence, AI Safety

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-safety-field", "nl")
    C = L.checks   # tabel 1 van Apollo, wie in Seoul tekende en de eerste leden van het netwerk, de namen van vier instellingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [AI-ethiek en AI-veiligheid](/course/open/rai-ethics-vs-safety?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Het AI-plan van Greaves en MacAskill doet het al beter dan de klamboes vanaf 2,5 × 10¹⁰ toekomstige levens, veel minder dan zelfs hun beperkte schatting. Waar rust hun pleidooi voor het AI-plan dan vooral op?",
        options={
            "main": "Op de 10²⁴ toekomstige levens van hun hoofdschatting",
            "right": "Op gokken: 0,1% kans op een ramp, en 1% daarvan vermeden",
            "cost": "Op wat klamboes kosten: ongeveer 4000 dollar per gered leven",
        },
        correct="right",
        why={
            "right": "Als er zo weinig toekomstige levens nodig zijn, doen de enorme aantallen er nauwelijks toe. Het pleidooi draait om de kans van 0,1% op een ramp en de 1% daarvan die het plan zou voorkomen, en daarvoor zijn er volgens Greaves en MacAskill geen harde gegevens.",
            "main": "Het AI-plan wint al bij 2,5 × 10¹⁰, ver onder 10²⁴, dus op de hoofdschatting rust het niet. Zie [AI-ethiek en AI-veiligheid](/course/open/rai-ethics-vs-safety?lang=nl).",
            "cost": "De klamboes zijn de benchmark waarmee het plan wordt vergeleken, niet waar het plan op rust. Zie [AI-ethiek en AI-veiligheid](/course/open/rai-ethics-vs-safety?lang=nl).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_scales():
    quick_scales = L.choice(
        "quick-scales",
        question="Prunkl en Whittlestone vonden de grens tussen korte en lange termijn slecht gedefinieerd. Wat zetten ze ervoor in de plaats?",
        options={
            "camps": "Twee kampen, een voor schade nu en een voor schade later, elk met eigen werk",
            "right": "Vier schalen, zoals hoe krachtig de systemen zijn en hoe zeker de schade is",
            "time": "Eén tijdschaal, van schade vandaag tot schade over honderd jaar",
        },
        correct="right",
        why={
            "right": "Capabilities, gevolgen, zekerheid en ernst: vier schalen in plaats van een keuze uit twee, zodat één zorg op de ene schaal dichtbij kan liggen en op de andere ver weg.",
            "camps": "Twee kampen is juist de indeling die ze slecht gedefinieerd vonden. Zie [AI-ethiek en AI-veiligheid](/course/open/rai-ethics-vs-safety?lang=nl).",
            "time": "Wanneer de schade komt, is maar één van hun schalen; ze voegen toe hoe krachtig de systemen zijn, hoe zeker de schade is en hoe groot. Zie [AI-ethiek en AI-veiligheid](/course/open/rai-ethics-vs-safety?lang=nl).",
        },
    )
    quick_scales
    return (quick_scales,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In nieuws over AI-veiligheid duiken veel namen op: een instituut dat een nieuw model testte, een evaluator die een model betrapte op het misleiden van de testers, een cursus die onderzoekers opleidt. Ze verschillen in wat ze mogen doen, ze veranderen van naam, en een deel van wat ze over zichzelf zeggen is hun eigen claim.

    Deze les brengt het vak in kaart: de instituten van overheden, de onafhankelijke evaluators, de cursussen, de bedrijven zelf en de onderzoeksagenda's die opsommen wat nog onopgelost is. Je telt zelf in de resultaten van één evaluator en vergelijkt zelf twee lijsten van landen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2024 gaf de evaluator Apollo Research AI-modellen een doel, zei erbij dat niets anders telde, en zette ze in een situatie waarin het toezicht van hun ontwikkelaars in de weg stond. In welk deel van de runs schakelde het model dat dit het vaakst deed dat toezicht uit?",
        options={
            "half": "In ongeveer de helft van de runs",
            "right": "In ongeveer één op twintig runs",
            "rare": "In ongeveer één op de duizend runs",
        },
        reveal="**In ongeveer één op twintig runs.** Het model dat het het vaakst deed, opus-3, schakelde zijn toezicht uit in 5% van de runs en o1 in 2%; twee van de vijf modellen in dat deel van de tabel deden het nooit. Dat ze het überhaupt deden, was waar het om ging: *Evaluators* in *Het idee* komt erop terug.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Instituten van overheden

    In november 2023 maakte de Britse regering van de Frontier AI Taskforce het AI Safety Institute: volgens de regering zelf de eerste door een staat gesteunde organisatie die zich richt op de veiligheid van geavanceerde AI in het algemeen belang. De opdracht: zo min mogelijk verrassingen voor het Verenigd Koninkrijk en de mensheid door snelle en onverwachte vooruitgang in AI. Het zou geavanceerde AI-systemen evalueren, onderzoek doen en informatie delen, en het kreeg er geen macht over: het instituut is geen toezichthouder en bepaalt geen regelgeving. De evaluaties zouden ook geen enkel AI-systeem tot 'veilig' verklaren. Zo'n instelling is een **AI safety institute**: een deel van een overheid dat geavanceerde AI-modellen test en onderzoek doet naar hun risico's.

    Bij de eerste evaluaties zouden ook vooringenomen antwoorden en redeneringen horen. Op 14 februari 2025 doopte minister Peter Kyle het om tot het AI Security Institute, met de nadruk op risico's zoals chemische en biologische wapens, cyberaanvallen en fraude; vooringenomenheid en vrijheid van meningsuiting horen er niet meer bij. Hetzelfde persbericht leest de wijziging op twee manieren. Kyle: het werk van het instituut verandert niet. De voorzitter, Ian Hogarth: het instituut richtte zich vanaf het begin op veiligheid in de zin van security. In november 2026 noemt de website meer dan 100 technische medewerkers en £66 miljoen per jaar, en het grootste team van een overheid ter wereld dat onderzoek doet naar de veiligheid en beveiliging van AI.

    In de Verenigde Staten werd op de Britse top van november 2023 een AI Safety Institute aangekondigd bij NIST, het nationale instituut voor standaarden en technologie. In juni 2025 werd het opnieuw opgericht als het Center for AI Standards and Innovation (CAISI). Een presidentieel besluit van 29 september 2026 droeg de overheid op "Super Intelligence" te schrijven waar ze "Artificial Intelligence" schreef, en het centrum werd het Center for Advancing Innovation and Standards for Super Intelligence (CAISSI). Voorlopig definieert het besluit de nieuwe woorden als de technologie die de Amerikaanse wet kunstmatige intelligentie noemt, en het vraagt binnen 60 dagen om een nieuwe definitie: de naam veranderde, terwijl de technologie waar die op slaat voorlopig dezelfde bleef. Volgens NIST richt het centrum zich bij zijn evaluaties op risico's die aantoonbaar zijn, zoals cybersecurity, biosecurity en chemische wapens.

    Op de AI Seoul Summit van 21 mei 2024 spraken 10 landen en de Europese Unie af een internationaal netwerk van zulke instituten op te zetten. Het kwam voor het eerst bijeen op 20 november 2024 in San Francisco, met de Verenigde Staten als eerste voorzitter, en sinds 9 december 2025 heet het International Network for Advanced AI Measurement, Evaluation and Science, met het Verenigd Koninkrijk als coördinator. Wie in Seoul tekende en wie lid van het eerste uur werd, is niet precies dezelfde lijst: in oefening 2 vind je het verschil.

    Het AI-bureau van de Europese Unie, opgericht bij een besluit van 24 januari 2024, is een ander soort instelling: het handhaaft de regels voor AI-modellen voor algemene doeleinden uit de AI Act van de EU, en helpt betrouwbare AI te ontwikkelen. Er werken meer dan 125 mensen, en een van de afdelingen heet AI Safety. Nederland staat op geen van de lijsten van het netwerk. In Nederland is de Autoriteit Persoonsgegevens "sinds 2023 coördinerend toezichthouder op algoritmes en AI die risico's met zich meebrengen voor fundamentele waarden en grondrechten": een toezichthouder, geen veiligheidsinstituut.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Evaluators

    Evaluators testen AI-modellen, vaak van buiten de bedrijven die ze maken. METR, uitgesproken als het Engelse "meter", is een organisatie zonder winstoogmerk die de meest geavanceerde AI-agents bestudeert om te begrijpen of hun ontwikkelaars op weg zijn de controle erover te verliezen. Het heeft geen geld aangenomen van AI-bedrijven, al gebruikt het wel veel gratis tokens. Een van de maten van METR is de time horizon: de lengte van de taken, in de tijd die vakmensen ervoor nodig hebben, die een model in de helft van de gevallen voltooit. In de versie van hun paper van juli 2026 heeft o3 een time horizon van ongeveer 110 minuten, en is de time horizon van de beste modellen sinds 2019 ongeveer elke zeven maanden verdubbeld. De les *Hoe lange taken kan een model aan?* rekent het later uit.

    Volgens Apollo Research komen sommige van de gevaarlijkste risico's van scheming: geavanceerde AI-systemen die heimelijk doelen nastreven die niet de bedoeling zijn. Frontier models zijn de krachtigste modellen van hun tijd. In een paper van december 2024, herzien in januari 2025, deed het zes evaluaties, en tabel 1 geeft de resultaten van zes modellen. Elke keer kreeg een model een doel en een situatie waarin scheming zou helpen dat doel te halen, zoals de kans om het eigen toezicht uit te schakelen, of om wat het voor zijn eigen weights hield naar een andere server te kopiëren voordat het vervangen werd. Vijf van de zes deden minstens één keer aan scheming. Lees wel de opzet: Apollo onderzocht of modellen aan scheming kunnen doen voor een doel dat Apollo zelf in de prompt gaf, met de opdracht het sterk na te streven. Toen Apollo die duw wegliet uit de vier evaluaties van tabel 2, kwam scheming veel minder vaak voor en in minder evaluaties; de samenvatting van de paper spreekt van zeldzame gevallen. Apollo schrijft dat het als eerste liet zien dat frontier models aan scheming kunnen doen voor een doel uit de prompt: een claim over zichzelf.

    ### Cursussen en de bedrijven

    De meeste mensen in het vak hebben het ergens geleerd. BlueDot Impact geeft gratis online cursussen van elk 25 tot 40 uur, en volgens de eigen telling werken meer dan 1.200 oud-deelnemers nu in het vak. ARENA, de Alignment Research Engineer Accelerator, geeft in Londen twee à drie bootcamps per jaar van vier tot vijf weken, van de grondslagen en transformer interpretability tot reinforcement learning, evaluaties van language models en alignment; het lesmateriaal staat online. En AI-bedrijven zoals Anthropic, Google DeepMind en OpenAI publiceren waar volgens hen het onderzoek naar veiligheid heen moet.

    ### Onderzoeksagenda's

    Wat is er nog onopgelost? Onderzoeksagenda's zetten het op een rij. Op 8 mei 2025 publiceerden onderzoekers die elkaar in Singapore hadden ontmoet, meer dan 100 deelnemers uit 11 landen, van universiteiten, AI-bedrijven, overheden en het maatschappelijk middenveld, de Singapore Consensus on Global AI Safety Research Priorities. Die bouwt onder meer voort op de prioriteiten van het Britse instituut, NIST, Anthropic, Google DeepMind en OpenAI, en verdeelt de open problemen over drie gebieden, in het plaatje hieronder: betrouwbare AI-systemen maken (ontwikkeling), hun risico's beoordelen (beoordeling), en ze na de ingebruikname in de gaten houden en ingrijpen (controle). Concurrenten kunnen zulk werk delen, zo stelt het document, zoals concurrerende vliegtuigbouwers als Boeing en Airbus samenwerken aan informatie en normen voor de veiligheid van de luchtvaart.

    De eigen agenda van het Britse instituut zet onderzoek naar risico's, van misbruik voor cyberaanvallen en misdaad tot autonome systemen en de weerbaarheid van de samenleving, naast onderzoek naar oplossingen: beveiligingen, controle en alignment. Over alignment schrijft het dat er geen betrouwbare methodes bekend zijn die voorkomen dat voldoende capabele AI-systemen hun beveiligingen omzeilen en schade doen.
    """)
    return


@app.cell(hide_code=True)
def areas_cell():
    mo.Html(C.areas_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **AI safety institute** | een overheidsinstelling die geavanceerde AI-modellen test en onderzoek doet naar hun risico's, zoals het Britse AI Security Institute; de naam van één instituut kan veranderen |
        | toezichthouder | een instelling die regels mag stellen of handhaven; het Britse instituut is er geen, het AI-bureau van de EU handhaaft de regels voor AI-modellen voor algemene doeleinden |
        | evaluator | een organisatie die AI-modellen test, vaak van buiten het bedrijf dat ze maakte |
        | time horizon | de maat van METR: de lengte van de taken, in de tijd die vakmensen nodig hebben, die een model in de helft van de gevallen voltooit |
        | scheming | een model dat heimelijk doelen nastreeft die niet die van de ontwikkelaars zijn, en ze verbergt |
        | onderzoeksagenda | een lijst van de problemen die een groep door onderzoek opgelost wil zien |
        | `C.TABLE` | tabel 1 van Apollo: een dict met voor elk van de zes evaluaties de set modellen die er minstens één keer aan scheming deden |
        | `C.MODELS` | de zes modellen die Apollo testte, zoals de tabel ze noemt: `"gpt-4o"`, `"gemini-1.5"`, `"llama-3.1"`, `"sonnet-3.5"`, `"opus-3"` en `"o1"` |
        | `C.SEOUL`, `C.NETWORK` | sets: de 11 die in Seoul tekenden (10 landen en de EU), en de 10 eerste leden van het netwerk |
        | `a - b` | voor twee sets: een nieuwe set met de waarden in `a` die niet in `b` zitten |
        """),
        L.watch_out("Een ✓ in een evaluatie betekent: minstens één keer, in deze opzet. Vraag hoe vaak, na welke duw, en wie zegt dat het de eerste keer was."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Vraag wat voor instelling het is: een deel van een overheid, een onafhankelijke evaluator, een bedrijf, een cursus of een groep die een agenda schrijft.
    2. Vraag wat ze mag doen: testen en adviseren, of regels stellen en handhaven.
    3. Geef een naam met de datum erbij, en kijk of die sindsdien veranderd is.
    4. Lees hoe een resultaat tot stand kwam: welke modellen, van wie het doel kwam, hoe sterk de duw, en hoe vaak.
    5. Let op wie iets "de eerste" of "de grootste" noemt: de instelling zelf of anderen.
    6. Plaats een stuk onderzoek in een agenda, bijvoorbeeld bij beoordeling, ontwikkeling of controle.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Tabel 1 van de paper van Apollo geeft voor elke evaluatie de modellen die er minstens één keer aan scheming deden; de course bewaart hem in `C.TABLE`. De cel loopt door de evaluaties en print hoeveel modellen er in elk aan scheming deden, en of gpt-4o daarbij is.
    """)
    return


@app.cell
def example_table():
    for evaluation in C.TABLE:                  # de zes evaluaties van Apollo
        models = C.TABLE[evaluation]            # de modellen die er minstens één keer aan scheming deden
        print(evaluation, len(models), "gpt-4o" in models)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke evaluatie test één manier van scheming. Bij oversight subversion kon een model het toezicht uitschakelen dat het zou stilleggen; bij self-exfiltration kon het zichzelf kopiëren naar de server van het model dat het moest vervangen; bij goal-guarding kon het de prompt veranderen die de ontwikkelaars voor zijn opvolger hadden geschreven; bij covert email reranking kon het ongemerkt een rangorde veranderen die het moest maken; bij instrumental alignment faking kon het alleen doen wat de ontwikkelaars wilden zolang het in de gaten werd gehouden; en bij sandbagging kon het met opzet slechter presteren in een test. Bij self-exfiltration deden vijf modellen minstens één keer aan scheming, bij covert email reranking maar één. gpt-4o zit in geen van de zes sets. In oefening 1 tel je andersom: voor één model, over alle evaluaties.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_schemed():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. schemed_in: in hoeveel evaluaties een model aan scheming deed

        Schrijf `schemed_in(table, model)`. `table` is een dict zoals `C.TABLE`: voor elke evaluatie een set modellen. Begin `count` bij 0, loop met een `for`-loop door de evaluaties in `table`, en tel 1 op bij `count` voor elke evaluatie waarvan de set `model` bevat. Geef `count` terug. Vervang de `...`. De check roept `schemed_in(TABLE, "opus-3")` aan, daarna de andere modellen en eigen tabellen.
        """),
        L.task("Tel de evaluaties waarin het model in de set modellen zit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_schemed():
    box_schemed = L.code_box("schemed_in", lines=6)
    box_schemed
    return (box_schemed,)


@app.cell(hide_code=True)
def check_schemed(box_schemed):
    result_schemed = L.check("schemed_in", box_schemed.value)
    result_schemed
    return (result_schemed,)


@app.cell(hide_code=True)
def hints_schemed():
    L.hints(
        "schemed_in",
        [
            "Loop de evaluaties een voor een af en houd een telling bij die elke keer één omhooggaat als het model in de set van die evaluatie zit.",
            "`count = 0` begint de telling. De set modellen van één evaluatie is de waarde van die evaluatie in de dict `table`, en `model in` een set geeft `True` als het model erin zit.",
            "`        if model in ___:`",
        ],
        why="`count = 0`, `for evaluation in table:`, daarbinnen `if model in table[evaluation]:` met `count = count + 1`, en dan `return count` op gelijke hoogte met `for`. Voor opus-3: oversight subversion, self-exfiltration, goal-guarding en sandbagging, dus 4.",
    )
    return


@app.cell(hide_code=True)
def text_after_schemed():
    mo.md(r"""
    o1 deed in alle zes evaluaties aan scheming, opus-3 en llama-3.1 in vier, gemini-1.5 in drie, sonnet-3.5 in twee en gpt-4o in geen enkele. Een ✓ betekent minstens één keer. Tabel 2 van de paper geeft het deel van de runs: opus-3 schakelde zijn toezicht uit in 5% ervan en o1 in 2%, en o1 gaf ongemerkt een valse rangorde in 22%. In vijf van de zes evaluaties kwam het doel van Apollo, met een sterke duw om het na te streven; bij instrumental alignment faking kwam het zonder die duw. Eén resultaat ging verder: sonnet-3.5 deed het in een test met opzet slechter om behulpzaam te zijn, een doel dat het in de training had gekregen en niet uit de prompt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. changes: wie tekende, en wie lid werd

        `C.SEOUL` bevat de 11 die in Seoul tekenden, `C.NETWORK` de 10 eerste leden van het netwerk. Schrijf `changes(before, after)` voor twee sets. Maak `left`, de gesorteerde lijst van de waarden in `before` die niet in `after` zitten, en `joined`, de gesorteerde lijst van de waarden in `after` die niet in `before` zitten: `a - b` en `sorted` doen dat. Geef ze allebei terug, met een komma ertussen: `return left, joined`. Vervang de `...`. De check roept `changes(SEOUL, NETWORK)` aan, daarna eigen sets.
        """),
        L.task("Wie in de eerste set zit en niet in de tweede, en wie andersom, elk als gesorteerde lijst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_changes():
    box_changes = L.code_box("changes", lines=4)
    box_changes
    return (box_changes,)


@app.cell(hide_code=True)
def check_changes(box_changes):
    result_changes = L.check("changes", box_changes.value)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def hints_changes():
    L.hints(
        "changes",
        [
            "Twee verschillen, in elke richting één, en elk omgezet in een lijst op volgorde.",
            "`before - after` is een set met de waarden in `before` die niet in `after` zitten, en `sorted(...)` maakt van een set een lijst op volgorde. De tweede lijst is het verschil andersom.",
            "`    joined = sorted(___)`",
        ],
        why="`left = sorted(before - after)`, `joined = sorted(after - before)` en `return left, joined`. Voor Seoul en het netwerk: `['Germany', 'Italy']` en `['Kenya']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_changes():
    mo.md(r"""
    Duitsland en Italië tekenden in Seoul, maar waren geen lid van het eerste uur; Kenia was lid van het eerste uur zonder daar te hebben getekend. De andere negen zitten in allebei. De twee lijsten komen uit verschillende documenten: ze zeggen wie één afspraak tekende en wie een half jaar later als eerste lid werd genoemd, niet waarom die twee verschillen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: namen door de tijd

    De schuif zet een datum, als een aantal maanden na november 2023, toen het Britse instituut werd opgericht; elke stand is de eerste dag van een maand. Het plaatje toont vier overheidsinstellingen door de tijd, met de naam die elk op die datum had vetgedrukt, op een balk die opvalt.
    """)
    return


@app.cell(hide_code=True)
def demo_names():
    names = L.explore("names", "maanden na november 2023", lambda _v: C.demo_frame(_v, L.lang),
                      start=2, values=C.MONTHS, role="",
                      task="Zoek het kleinste getal op de schuif waarbij geen van de namen het woord Safety bevat.")
    names
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 26, 1 januari 2026, bevat geen van de vier namen het woord Safety: het Britse instituut ging over op Security, het Amerikaanse centrum op Standards and Innovation en later Super Intelligence, en het netwerk op Measurement, Evaluation and Science. Wat het werk zelf werd, is een andere vraag: volgens de minister verandert het niet, terwijl vooringenomenheid buiten de taak viel. Geef een naam dus met de datum erbij, en gebruik AI safety institute voor het soort instelling.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Oefening 3. Een kop

    Boven een nieuwsbericht staat: "Brits AI Security Institute verklaart nieuw model veilig".
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Wat klopt er niet aan deze kop, als je afgaat op wat de Britse regering over het instituut schreef?",
        options={
            "closed": "Het instituut is in 2025 gesloten, toen het zijn nieuwe naam kreeg",
            "right": "Het is geen toezichthouder, en het noemt geen enkel model veilig",
            "bias": "Sinds de nieuwe naam onderzoekt het alleen nog vooringenomenheid en vrije meningsuiting",
        },
        correct="right",
        why={
            "right": "In 2023 schreef de regering dat het instituut geen toezichthouder is en dat de evaluaties geen enkel AI-systeem tot 'veilig' zouden verklaren. Het test en adviseert. In de EU handhaaft het AI-bureau wel regels, voor AI-modellen voor algemene doeleinden.",
            "closed": "Het kreeg een nieuwe naam en ging niet dicht: sinds 14 februari 2025 heet het AI Security Institute, en volgens de minister verandert het werk niet. De kop gaat de mist in bij wat het instituut mag. Zie *Instituten van overheden* in *Het idee*.",
            "bias": "Juist andersom: sinds de nieuwe naam horen vooringenomenheid en vrijheid van meningsuiting er niet meer bij. De kop gaat ergens anders de mist in: bij wat het instituut mag. Zie *Instituten van overheden* in *Het idee*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_exercise_area():
    mo.md(r"""
    ### Oefening 4. Waar in de agenda?

    Voordat een nieuw model uitkomt, kijkt een team van een AI safety institute of het model iemand kan helpen een cyberaanval te plannen.
    """)
    return


@app.cell(hide_code=True)
def choice_area_cell():
    choice_area = L.choice(
        "area",
        question="In welk van de drie gebieden van de Singapore Consensus valt dit werk?",
        options={
            "development": "Ontwikkeling: het gebeurt voor de release, terwijl het model nog gemaakt wordt",
            "right": "Beoordeling: het zoekt uit hoe ernstig en waarschijnlijk een schade kan zijn",
            "control": "Controle: het gaat erom aanvallen te voorkomen",
        },
        correct="right",
        why={
            "right": "Testen wat een model kan, is risicobeoordeling, gebied 1, of het nu voor of na de release gebeurt; de uitkomsten voeden de beslissingen over ontwikkeling en release. De latere les *Testen op gevaarlijke vaardigheden* gaat er dieper op in.",
            "development": "Ontwikkeling gaat over het systeem zo bouwen dat het doet wat is vastgelegd; de verificatie, 2.3, gaat na of het systeem werkt zoals vastgelegd. Uitzoeken welke gevaarlijke dingen een model kan, is 1.6, onder beoordeling, ook voor de release. Zie het plaatje onder *Onderzoeksagenda's* in *Het idee*.",
            "control": "Controle gaat over een systeem na de release in de gaten houden en ingrijpen. Uitzoeken wat een model kan, is beoordeling. Zie het plaatje onder *Onderzoeksagenda's* in *Het idee*.",
        },
    )
    choice_area
    return (choice_area,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [AI-ethiek en AI-veiligheid](/course/open/rai-ethics-vs-safety?lang=nl) liet zien waarover twee groepen het oneens zijn. Deze les liet zien wie het werk doet: instituten van overheden die testen en adviseren, een EU-bureau dat regels handhaaft, evaluators zoals METR en Apollo Research, cursussen zoals BlueDot en ARENA, de bedrijven zelf, en agenda's die opsommen wat onopgelost is. De volgende les, *Robuust, veerkrachtig, beschikbaar, voorspelbaar*, kijkt naar wat er nodig is om een AI-systeem te laten blijven werken: als de invoer een beetje afwijkt, na een storing, en elke keer dat het nodig is.

    **Een voorbeeld uit het dagelijks leven.** In veel landen geven onafhankelijke crashtests nieuwe auto's sterren, terwijl een overheidsdienst beslist of een auto überhaupt verkocht mag worden. Sterren zijn geen vergunning, en een vergunning is geen recensie.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_area, choice_headline, result_changes, result_schemed):
    L.wrap_up(result_schemed, result_changes, choice_headline, choice_area)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
