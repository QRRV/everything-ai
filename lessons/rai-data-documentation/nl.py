# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-data-documentation. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rai-data-documentation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) en [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="Een dataset op een website heeft geen licentie. Waar ga je van uit?",
        options={
            "anything": "Alles mag",
            "right": "Je mag hem niet kopiëren of delen",
            "name": "Je mag hem gebruiken als je de maker noemt",
        },
        correct="right",
        why={
            "anything": "Gratis te downloaden is niet vrij te gebruiken. Zonder licentie ga je ervan uit dat je de data niet mag kopiëren of delen, en vraag je het de maker. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Zonder licentie ga je ervan uit dat je de data niet mag kopiëren of delen: vraag het de maker voordat je hem deelt of erop voortbouwt.",
            "name": "De maker noemen is een voorwaarde van CC BY, een licentie die deze dataset niet heeft. Zonder licentie ga je ervan uit dat je hem niet mag kopiëren of delen. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def quick_check_sample():
    quick_sample = L.choice(
        "quick-sample",
        question="Een spraakherkenner is getraind op opnames met bijna geen oudere sprekers, en werkt voor hen slechter. Welke bias is dit?",
        options={
            "right": "Representation bias",
            "historical": "Historical bias uit een ongelijke wereld",
            "measurement": "Measurement bias in de labels",
        },
        correct="right",
        why={
            "right": "Te weinig voorbeelden van een deel van de mensen voor wie het model gebruikt wordt: representation bias, in de stap van de steekproef.",
            "historical": "Historical bias zit in de wereld zelf, ook bij een perfecte steekproef. Hier zitten er te weinig oudere sprekers in de steekproef: representation bias. Zie [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).",
            "measurement": "Measurement bias komt van features of labels die voor sommige groepen iets anders meten. Hier komen oudere sprekers weinig voor in de data: representation bias. Zie [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).",
        },
    )
    quick_sample
    return (quick_sample,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    UCI Adult bevat 48.842 mensen uit een Amerikaanse enquête van 1994. Frances Ding en collega's (2021) vonden hem in meer dan driehonderd papers over fairness, en noemden als problemen onder meer zijn leeftijd, de magere documentatie en verouderde manieren om waarden te coderen. Wat de pagina in de UCI-repository zegt over hoe hij gemaakt is, past in twee zinnen: Barry Becker haalde de gegevens uit de "1994 Census database", en nam alleen redelijk schone records die voldeden aan `((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))`. Lees die voorwaarden langzaam, en ze zeggen wie er níet in zit. Ding en collega's filterden hun vervanger op bijna dezelfde manier: mensen ouder dan 16, die naar eigen zeggen meestal minstens 1 uur per week werkten in het jaar ervoor, met een inkomen van minstens 100 dollar. En ondanks het woord Census komt de data uit een enquête, de Current Population Survey. Een model dat op Adult getraind is, heeft niemand buiten deze grenzen gezien, en alleen documentatie vertelt dat aan wie de data hierna gebruikt. Emily Bender en Batya Friedman (2018) schreven over de problemen die ontstaan als data van de ene groep mensen gebruikt wordt om technologie te maken voor een andere.

    **Een voorbeeld uit het dagelijks leven.** De bijsluiter in een doosje medicijnen zegt wat erin zit, voor wie het bedoeld is, en wanneer je het niet moet nemen. Dat zie je niet aan een pil. Een dataset heeft hetzelfde nodig: wie erin zit, hoe hij gemaakt is, en wanneer je hem niet moet gebruiken.
    """)
    return


@app.cell(hide_code=True)
def predict_women():
    guess_women = L.guess(
        "women",
        question="UCI Adult komt uit een Amerikaanse enquête van 1994. Welk deel van de 48.842 mensen is vrouw?",
        options={
            "half": "Ongeveer de helft",
            "right": "Ongeveer een derde",
            "quarter": "Ongeveer één op de vier",
        },
        reveal="**Ongeveer een derde: 16.192 van de 48.842, of 0,33.** Wie er in een dataset zit, is niet hetzelfde als wie er in een land woont. Een data card zegt in getallen wie er in de data zit, zodat niemand hoeft te raden, of de helft hoeft aan te nemen.",
    )
    guess_women
    return (guess_women,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drie vragen.** In [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) schreef je een datasheet met vijf velden. De datasheets van Timnit Gebru en collega's (2021) verdelen hun vragen over zeven onderdelen die ruwweg het leven van een dataset volgen: motivatie, samenstelling, verzameling, voorbewerking (opschonen en labelen), gebruik, verspreiding en onderhoud. De vragen zijn niet bedoeld als voorschrift: wat niet van toepassing is, sla je over. Andere vormen vragen grotendeels hetzelfde, elk op hun eigen manier. Voor fairness tellen drie vragen het meest: wie er in de data zit, hoe die verzameld en gelabeld is, en waarvoor die niet geschikt is.

        **Data statements.** Emily Bender en Batya Friedman (2018) schreven data statements voor taaldata: waarom juist deze teksten, welke taalvariant met een code als `en-US`, de leeftijd, het gender en de moedertaal van de sprekers en van de annotators, de situatie waaruit de taal komt, en meer. Een lange vorm hoort in de paper die een dataset presenteert; elke paper die de dataset gebruikt, hoort een korte vorm van 60 tot 100 woorden te geven die naar de lange verwijst. Gaten mogen: alles volledig invullen is niet altijd haalbaar, en een bereik kan beter zijn dan een precieze waarde. Bender en Friedman schrijven data statements voor taaldata; voor een tabel als Adult past een datasheet of een data card beter.

        **Data Cards.** Mahima Pushkarna, Andrew Zaldivar en Oddur Kjartansson (2022) maakten met 12 teams in een groot techbedrijf 22 Data Cards: gestructureerde samenvattingen van de belangrijkste feiten over een dataset, voor iedereen die er tijdens het leven van die dataset mee werkt. Een card vraagt op drie niveaus. Telescopen geven een overzicht, zoals de vraag of de dataset gevoelige menselijke eigenschappen bevat; periscopen geven technische details; microscopen vragen naar de onzichtbare menselijke processen, beslissingen, aannames en regels die de dataset gevormd hebben, en die waren het moeilijkst te beantwoorden. Deelnemers aan hun onderzoek vonden "known unknowns", wat je weet dat je niet weet, even belangrijk als wat bekend is. Onder meer dit ging mis: mensen kopieerden een kant-en-klare card, zodat de fouten ervan in de kopieën terechtkwamen, en ze schreven "N/A" als ze het niet zeker wisten.

        **Voedingslabels.** Sarah Holland en collega's (2018) maakten het Dataset Nutrition Label, naar het etiket op voedsel: voorbeeldmodules zoals Metadata, de enige verplichte, Provenance, Variables en Statistics. In hun enquête zei 47 procent van de deelnemers dat ze enige analyse van de data deden voordat ze een model bouwden, maar zei 74 procent dat hun organisatie daar geen expliciete werkwijze voor had. Een tweede generatie, gepresenteerd in 2020, vraagt eerst waarvoor je de data wilt gebruiken, en laat dan de waarschuwingen zien die voor dat gebruik tellen.

        **Op de Hugging Face Hub** is een dataset card de `README.md` van een dataset, die de Hub op de pagina ervan laat zien. Het sjabloon heeft een deel *Uses* met *Direct Use* en *Out-of-Scope Use*, voor misbruik en gebruik waarvoor de dataset niet goed werkt, en een deel *Bias, Risks, and Limitations*, voor technische en sociotechnische beperkingen. Een veld dat niemand invulde, zegt "[More Information Needed]".

        | Vraag | Datasheet | Data Card | Hugging Face card |
        |---|---|---|---|
        | Wie zit erin? | *Composition* | *Breakdown of subgroups* | *Who are the source data producers?* |
        | Hoe is hij gemaakt? | *Collection process*; *labeling* | *Collection methods* | *Data Collection and Processing* |
        | Waarvoor is hij niet geschikt? | *Uses* | *Unacceptable uses* | *Out-of-Scope Use* |

        **FAIR is iets anders.** In 2014 leidde een workshop in Leiden, "Jointly Designing a Data Fairport", tot de FAIR-principes van Mark Wilkinson en collega's (2016): data moet vindbaar, toegankelijk, uitwisselbaar en herbruikbaar zijn (Findable, Accessible, Interoperable, Reusable), voor machines en voor mensen. Het zijn er vijftien, zoals F1, een wereldwijd unieke en blijvende identifier (Adult heeft de DOI 10.24432/C5XW20), en R1.1, een duidelijke en toegankelijke licentie voor het gebruik. Ze zijn zelf geen standaard of specificatie, en data hoeft niet open te zijn om FAIR te zijn: bij gevoelige data geven rijke metadata met duidelijke regels voor toegang al een hoge mate van FAIR. Over bias zegt FAIR niets. Een dataset kan aan alle vijftien voldoen en toch een groep missen.

        **De wet.** Volgens de AI Act van de EU moet de data om een AI-systeem met een hoog risico op te trainen, te valideren en te testen onder datagovernance vallen die ook een onderzoek omvat naar mogelijke bias die gezondheid, veiligheid of grondrechten kan schaden of tot verboden discriminatie kan leiden, en moet die data relevant, voldoende representatief en zo veel mogelijk foutloos en volledig zijn voor het beoogde doel. De technische documentatie vraagt, waar relevant, om "datasheets" die de training data beschrijven. Na een wijziging in 2026, de Digital Omnibus voor AI, gelden deze regels vanaf 2 december 2027 voor de AI-systemen met een hoog risico uit bijlage III, en vanaf 2 augustus 2028 voor de andere. Sinds 2 augustus 2025 moeten aanbieders van AI-modellen voor algemene doeleinden een voldoende gedetailleerde samenvatting publiceren van de inhoud waarop ze getraind zijn, volgens een sjabloon dat de Europese Commissie in 2025 publiceerde; voor modellen die al vóór die datum op de markt waren, uiterlijk 2 augustus 2027.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **data statement** | een beschrijving van een taaldataset: waarom deze teksten, welke taalvariant, wie ze sprak of schreef en wie ze annoteerde |
        | **data card** | een gestructureerde samenvatting van de belangrijkste feiten over een dataset, voor iedereen die ermee werkt, ook van wat niet bekend is |
        | **Dataset Nutrition Label** | een label voor een dataset, naar het etiket op voedsel, met modules zoals metadata, herkomst en statistieken |
        | **dataset card** | op de Hugging Face Hub de `README.md` van een dataset, op de pagina ervan te zien |
        | **out-of-scope use** | gebruik waarvoor een dataset niet bedoeld is, of waarvoor hij niet goed werkt |
        | **known unknown** | iets waarvan je weet dat je het niet weet over een dataset; je schrijft het op in plaats van het weg te laten |
        | **FAIR-principes** | vindbaar, toegankelijk, uitwisselbaar en herbruikbaar: hoe je data deelt zodat mensen en machines die kunnen vinden en hergebruiken; niet over fairness |
        """),
        L.watch_out("FAIR is geen fairness: data kan vindbaar, toegankelijk, uitwisselbaar en herbruikbaar zijn en toch een groep missen. En waar je iets niet weet, schrijf je op dat het onbekend is: een leeg veld of N/A verbergt wat de volgende gebruiker moet weten."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo schrijf je een data card voor een dataset die je gebruikt, ook een die je niet zelf maakte:

    1. Waarom hij gemaakt is, door wie en waarvan. Waar niemand het opschreef, schrijf je dat het voor jou onbekend is.
    2. Wie erin zit: tel elke groep, en het aandeel ervan in alle rijen.
    3. Wat onbekend is: per kolom hoe vaak een waarde ontbreekt, ook per groep.
    4. Waarvoor hij niet geschikt is: de mensen en situaties buiten zijn bereik, en wat volgt uit hoe hij gemaakt is.
    5. Waar je hem vindt en onder welke voorwaarden: een blijvende identifier, zoals een DOI, en de licentie.
    """)
    return


@app.cell(hide_code=True)
def picture_card():
    mo.vstack([
        mo.md("**Een data card voor UCI Adult, zoals deze les hem invult**"),
        draw.dict_boxes([("gemaakt door", "Barry Becker, Ronny Kohavi"), ("bron", "Current Population Survey, 1994"),
                         ("wie zit erin", "oefening 1"), ("onbekend", "oefening 3"),
                         ("ongeschikt voor", "het uitgewerkte voorbeeld"), ("toestemming", "onbekend voor ons"),
                         ("licentie", "CC BY 4.0"), ("identifier", "DOI 10.24432/C5XW20")],
                        key_label="veld", value_label="antwoord"),
        mo.md(r"""
        Elk veld beantwoordt een vraag uit het recept. Drie ervan komen uit de data zelf, met een paar regels code: het uitgewerkte voorbeeld en oefening 1 en 3. Toestemming is een known unknown: niets in de documentatie van Adult zegt waarmee de mensen in de enquête instemden.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4 voor Adult: de laagste en de hoogste waarde van twee kolommen, `age` en `hours_per_week`. Buiten dat bereik heeft de tabel niemand om van te leren. Voer de cel uit.
    """)
    return


@app.cell
def example_ranges():
    import pandas as pd

    # 48.842 mensen uit een Amerikaanse enquête van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")

    # 4. waarvoor hij niet geschikt is: het bereik van twee kolommen
    for _column in ["age", "hours_per_week"]:
        print(_column, _people[_column].min(), "tot", _people[_column].max())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Niemand in Adult is jonger dan 17, en niemand werkte 0 uur per week: dat doen de voorwaarden AAGE>16 en HRSWK>0. Dus onder out-of-scope use zegt de card: niet voor mensen jonger dan 17, of voor mensen die niet werken. Wie er binnen het bereik zit, is de volgende vraag: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie zit erin?

        `people` is UCI Adult, en `column` de kolom met de groepen, `"race"`. Reken `shares` uit: voor elke groep het aantal mensen gedeeld door het aantal van alle rijen. `people[column].value_counts()` telt de mensen per groep, zoals in [Vragen stellen aan data](/course/open/data-explore?lang=nl), en `len(people)` geeft het aantal rijen. De check probeert ook `column = "sex"`.
        """),
        L.task("Reken `shares` uit: per groep van `column` het aandeel in alle mensen in de tabel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=12)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Twee getallen per groep: hoeveel mensen erin zitten, en hoeveel mensen er in totaal zijn.",
            "`people[column].value_counts()` geeft het aantal per groep; `len(people)` het aantal van alle rijen.",
            "`shares = people[column].value_counts() / ___`",
        ],
        why="`counts = people[column].value_counts()`, dan `shares = counts / len(people)`. Voor race geeft dat `0.855` voor White, `0.096` voor Black, `0.031` voor Asian-Pac-Islander, `0.010` voor Amer-Indian-Eskimo en `0.008` voor Other.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Oefening 2. Wat zegt FAIR?
    """)
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair",
        question="Een dataset voldoet aan alle vijftien FAIR-principes. Wat weet je dan?",
        options={
            "group": "Er ontbreekt geen groep",
            "right": "Je kunt hem vinden en hergebruiken",
            "open": "Iedereen mag de data gratis downloaden",
        },
        correct="right",
        why={
            "group": "FAIR gaat over data vinden, openen, uitwisselen en hergebruiken, niet over wie erin zit. Een dataset kan aan alle vijftien principes voldoen en toch een groep missen. Zie *FAIR is iets anders* in *Het idee*.",
            "right": "Vindbaar, toegankelijk, uitwisselbaar en herbruikbaar, voor mensen en voor machines: FAIR zegt hoe je data kunt vinden en hergebruiken, en niets over bias.",
            "open": "FAIR-data hoeft niet open te zijn: een van de principes staat toe dat je je moet aanmelden en toestemming nodig hebt, waar dat nodig is, en gevoelige data kan FAIR zijn met rijke metadata en duidelijke regels voor toegang. Zie *FAIR is iets anders* in *Het idee*.",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_unknown():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wat is onbekend?

        Bij sommige mensen in Adult is de `occupation` onbekend. Reken `unknown` uit: voor elke groep van `column` het deel van de mensen van wie het beroep onbekend is. Houd de rijen waarin het onbekend is, `people[people["occupation"].isna()]`, tel ze per groep, en deel door het aantal mensen in elke groep. Als je het ene aantal per groep door het andere deelt, deelt pandas groep voor groep, op naam. De check probeert ook `column = "race"`.
        """),
        L.task("Reken `unknown` uit: per groep van `column` het deel van de mensen van wie het beroep onbekend is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_unknown():
    box_unknown = L.code_box("unknown", lines=12)
    box_unknown
    return (box_unknown,)


@app.cell(hide_code=True)
def check_unknown(box_unknown):
    result_unknown = L.check("unknown", box_unknown.value)
    result_unknown
    return (result_unknown,)


@app.cell(hide_code=True)
def hints_unknown():
    L.hints(
        "unknown",
        [
            "Eerst de mensen van wie het beroep onbekend is; dan per groep hun aantal gedeeld door het aantal mensen in de groep.",
            "`missing = people[people[\"occupation\"].isna()]` houdt die rijen; `missing[column].value_counts()` telt ze per groep.",
            "`unknown = missing[column].value_counts() / ___`",
        ],
        why="`missing = people[people[\"occupation\"].isna()]`, dan `unknown = missing[column].value_counts() / people[column].value_counts()`. pandas deelt groep voor groep, op naam. Dat geeft `0.079` voor vrouwen en `0.047` voor mannen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_write():
    mo.md(r"""
    ### Oefening 4. Wat schrijf je op?
    """)
    return


@app.cell(hide_code=True)
def choice_write_cell():
    choice_write = L.choice(
        "write",
        question="Je schrijft een datasheet voor UCI Adult. Nergens staat of de mensen in de enquête met dit gebruik instemden. Wat schrijf je bij toestemming?",
        options={
            "na": "N/A: niet van toepassing",
            "right": "Dat het voor jou onbekend is",
            "empty": "Niets; laat het veld leeg staan",
        },
        correct="right",
        why={
            "na": "N/A zegt dat de vraag niet van toepassing is, en bij data over mensen is toestemming dat wel. Teams in het onderzoek naar Data Cards schreven \"N/A\" als ze het niet zeker wisten, en verborgen zo wat ze niet wisten. Zie *Data Cards* in *Het idee*.",
            "right": "Schrijf op wat je niet weet. In hun eigen voorbeeld beantwoordden Gebru en collega's zulke vragen met \"onbekend voor de auteurs van de datasheet\", omdat ze de dataset niet zelf gemaakt hadden.",
            "empty": "Een leeg veld ziet eruit alsof niemand het vroeg. Schrijf op dat het voor jou onbekend is: de volgende gebruiker weet dan dat de vraag gesteld is en geen antwoord heeft. Zie stap 1 van het *Recept*.",
        },
    )
    choice_write
    return (choice_write,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) schreef je een datasheet met vijf velden tekst. Een data card voegt getallen toe: wie er in de data zit, als het aandeel van elke groep, en wat onbekend is, per groep. [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl) noemde representation bias in de stap van de steekproef; op een card schrijft een team die op, zodat het volgende team hem niet opnieuw hoeft te vinden. Oefening 3 liet nog een manier zien waarop die begint: een team dat de rijen met een onbekende waarde weglaat, laat een groter deel van de vrouwen weg dan van de mannen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, choice_write, result_shares, result_unknown):
    L.wrap_up(result_shares, choice_fair, result_unknown, choice_write)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
