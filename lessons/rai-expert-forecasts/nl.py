# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-expert-forecasts. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow Artificial Intelligence, Journal, Existential Risk, Risk, Expert Survey

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-expert-forecasts", "nl")
    C = L.checks   # twintig verzonnen antwoorden, de weging van Cotra met verzonnen kansen, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gap():
    quick_gap = L.choice(
        "quick-gap",
        question="In het verzonnen model van differential technology development uit de vorige les komt een gevaarlijke technologie in jaar 6, en de bescherming ertegen in jaar 4. Hoeveel jaar is het gevaar onbeschermd?",
        options={
            "minus": "−2: de bescherming komt twee jaar te vroeg, dus min twee",
            "right": "0: de bescherming is er eerder dan het gevaar",
            "gap": "2: de jaren tussen de twee jaartallen",
        },
        correct="right",
        why={
            "right": "De onbeschermde jaren zijn de jaren dat het gevaar er is voordat de bescherming komt, en nooit minder dan 0: `max(0, 4 - 6)` is 0. Een bescherming die vroeg komt, laat geen onbeschermde jaren over, en ook geen negatief aantal.",
            "minus": "Een aantal onbeschermde jaren kan niet negatief zijn: een bescherming die vroeg komt, laat er geen over, dus `max(0, 4 - 6)` is 0. Zie [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl).",
            "gap": "De bescherming komt in jaar 4, vóór het gevaar in jaar 6, dus het gevaar is nooit onbeschermd. Die 2 jaar zouden alleen tellen als de bescherming na het gevaar kwam. Zie [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl).",
        },
    )
    quick_gap
    return (quick_gap,)


@app.cell(hide_code=True)
def quick_check_median():
    quick_median = L.choice(
        "quick-median",
        question="Vijf leerlingen reizen in 10, 12, 13, 15 en 60 minuten naar school. Wat is groter, het gemiddelde of de mediaan van hun reistijden?",
        options={
            "right": "Het gemiddelde: de ene lange reis trekt het omhoog",
            "median": "De mediaan: dat is de middelste reis, 13 minuten",
            "same": "Geen van beide: ze beschrijven allebei het midden van dezelfde vijf reizen",
        },
        correct="right",
        why={
            "right": "Het gemiddelde is 110 / 5 = 22 minuten, en de mediaan 13, de middelste reis als je de reizen sorteert. Eén extreme waarde trekt aan het gemiddelde, maar verandert de mediaan nauwelijks.",
            "median": "De mediaan is inderdaad 13 minuten, de middelste reis. Maar het gemiddelde is 110 / 5 = 22 minuten: de reis van 60 minuten trekt het omhoog. Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "same": "Allebei beschrijven ze een midden, maar niet hetzelfde: het gemiddelde is 110 / 5 = 22 minuten, de mediaan 13. Eén lange reis trekt aan het gemiddelde, maar verandert de mediaan nauwelijks. Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
        },
    )
    quick_median
    return (quick_median,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Nieuws over de risico's van AI draait vaak om één getal, zoals een kans van 10% dat AI de mensheid uitroeit. Achter zo'n getal zit meestal een enquête onder AI-onderzoekers, en een enquête levert veel antwoorden op, niet één. In een enquête van AI Impacts in december 2024 beantwoordden 1.489 onderzoekers elk een vraag over de kans dat AI leidt tot het uitsterven van de mens, of tot iets wat even blijvend en ernstig is. De mediaan van hun antwoorden was 10%.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Wat was het gemiddelde van die 1.489 antwoorden?",
        options={
            "below": "Ongeveer 5%: onder de mediaan",
            "same": "Ongeveer 10%: gelijk aan de mediaan",
            "right": "Ongeveer 18%: boven de mediaan",
        },
        reveal="**Ongeveer 18%: boven de mediaan.** Volgens het rapport is de gemiddelde kans in de groep minstens ongeveer 18%. Het woord *minstens* staat er omdat twee van de drie vragen erachter smaller zijn dan de eerste, zodat het rapport de drie samen bij benadering als ondergrens ziet. 12% van de deelnemers gaf een kans van nul, en een derde gaf minstens 20%: de hoge antwoorden trekken het gemiddelde omhoog, zoals de ene lange reis in de instapcheck. Deze les kijkt naar zulke getallen: waar ze vandaan komen, hoe ver ze uiteenlopen, en wat ze waard zijn.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Expertenenquêtes

    AI Impacts houdt de Expert Survey on Progress in AI, kortweg ESPAI, volgens het rapport sinds 2016 met tussenpozen, met weinig aanpassingen. Elke keer krijgen onderzoekers die publiceerden bij de grote AI-conferenties en -tijdschriften dezelfde soort vragen: wanneer machines wat kunnen, en hoe waarschijnlijk heel goede of heel slechte uitkomsten zijn. In de preprint van hun paper over de enquête van 2023 (eerste versie januari 2024) schrijven Katja Grace en collega's dat ze in de grootste enquête van dit soort 2.778 onderzoekers ondervroegen die hadden gepubliceerd bij toonaangevende AI-conferenties en -tijdschriften. Dat was een respons van 15%, en niet iedereen zag elke vraag: volgens de auteurs kregen de deelnemers een willekeurig deel van de vragen, zodat de meeste vragen veel minder dan 2.778 antwoorden kregen. De drie vragen over uitsterven kregen er 1.321, 661 en 655.

    De volgende enquête liep in december 2024, onder mensen die in 2023 hadden gepubliceerd bij zes conferenties en tijdschriften: NeurIPS, ICML, ICLR, AAAI, JMLR en IJCAI. Van 19.874 adressen reageerden er 2.052, 10%; daarvan bleven er 1.580 over voor de hoofdanalyse. Het rapport, van Grace en collega's, verscheen in september 2026. De enquête vraagt op drie manieren naar uitsterven. De eerste vraag is welke kans de deelnemer geeft dat toekomstige vooruitgang van AI leidt tot het uitsterven van de mens, of tot een even blijvende en ernstige machteloosheid van de mensheid. De tweede vraagt hetzelfde over het onvermogen van mensen om toekomstige geavanceerde AI-systemen onder controle te houden; de derde beperkt zich tot de komende 100 jaar. Elke deelnemer kreeg één van de drie, de eerste twee keer zo vaak als elk van de andere. In procent:

    | Vraag | Antwoorden | Gemiddelde | Mediaan |
    |---|---|---|---|
    | vooruitgang van AI | 744 | 18,3 | 10 |
    | geen controle over AI | 392 | 18,5 | 9 |
    | binnen 100 jaar | 353 | 17,5 | 5 |
    | de drie samen | 1.489 | 18,2 | 10 |

    Volgens het rapport liep de middelste helft van de schattingen van 1% tot 25%. In de enquête van 2023 was de mediaan van de drie samen 5%. Over 2024 schrijft het rapport dat de AI-onderzoeker in het midden voor het eerst in deze enquêtes een kans van 10% geeft op uitsterven of blijvende en ernstige machteloosheid door toekomstige AI, zonder verdere voorwaarden.

    Dezelfde enquête vraagt ook wanneer. High-level machine intelligence, kortweg HLMI, is er volgens de beschrijving in de enquête als machines zonder hulp elke taak beter en goedkoper kunnen doen dan menselijke werknemers; de enquête vraagt naar wat kan, niet naar wat al gebruikt wordt. Hoe de vraag gesteld wordt, maakt uit: de helft van de deelnemers kreeg een jaartal en gaf een kans, de andere helft kreeg een kans en gaf een jaartal. Volgens het rapport geven die twee vormen in alle edities van de enquête steeds andere uitkomsten, met in het algemeen latere jaartallen als het jaartal vastligt. In de samengenomen voorspelling van het rapport heeft HLMI in 2027 een kans van 10% en in 2042 een kans van 50%, 18 jaar na de enquête. En in acht jaar kromp de horizon van die voorspelling volgens het rapport van 45 tot 18 jaar, zo'n 3,4 jaar voor elk jaar dat verstreek.

    In de versie van hun paper over de enquête van 2023 in het tijdschrift *Journal of Artificial Intelligence Research* (oktober 2025) zeggen de auteurs zelf wat zulke getallen waard zijn. De ondervraagden zijn volgens hen experts in AI-onderzoek, niet in het voorspellen van AI, en missen misschien algemene vaardigheid in voorspellen, of kennis van niet-technische factoren die de koers van AI bepalen. Voorspellingen van AI-experts zijn volgens de auteurs niet te zien als een betrouwbare gids naar de objectieve waarheid, maar ze kunnen wel één belangrijk stukje van de puzzel zijn.

    ### Superforecasters

    In het *Existential Risk Persuasion Tournament*, kortweg XPT, liet het Forecasting Research Institute experts en superforecasters van juni tot oktober 2022 dezelfde vragen voorspellen. Volgens de working paper (versie van augustus 2023) bestond de hoofdgroep uit 89 superforecasters die hoog scoorden in eerdere toernooien, en 80 experts op verschillende gebieden van existentieel risico; 32 van die experts waren AI-experts. Uitsterven betekent in de paper dat de wereldbevolking krimpt tot minder dan 5.000 mensen, en een catastrofe dat 10% of meer van de mensen sterft binnen 5 jaar. Voor AI, uiterlijk in 2100, gaven de superforecasters als mediaan 2,13% voor een catastrofe en 0,38% voor uitsterven; de AI-experts gaven als mediaan 12% en 3%. Voor uitsterven was het gemiddelde van de AI-experts 13,61%, meer dan vier keer hun mediaan. Bij de drie vragen van de paper over uitsterven (door AI, door kernoorlog en in totaal) gaf van vijf manieren om voorspellingen samen te voegen volgens de paper steeds het gemiddelde het hoogste getal, vaak twee keer zo groot als de mediaan of meer.

    Van sommige vragen uit het toernooi is de uitkomst inmiddels bekend. Op 2 september 2025 schreef het instituut dat de deelnemers de vooruitgang van AI hadden onderschat, vooral de superforecasters. AI-systemen presteerden in juli 2025 op het niveau van een gouden medaille op de Internationale Wiskunde Olympiade, een uitkomst waar superforecasters volgens het instituut maar 2,3% kans op gaven, en domeinexperts 8,6%.

    ### p(doom)

    Wikipedia (pagina laatst bewerkt op 15 november 2026) omschrijft P(doom), in het vakgebied van AI-veiligheid, als de kans op existentieel catastrofale uitkomsten door kunstmatige intelligentie. De term werd volgens Wikipedia vermoedelijk voor het eerst gebruikt door programmeur Tim Tyler, in een reactie op LessWrong in 2009; het woord *vermoedelijk* komt van Wikipedia. Isaac King waarschuwde in een post op LessWrong van 1 januari 2024 dat de term heel verschillende dingen kan betekenen, zoals de kans dat AI ons binnen 5 tot 10 jaar doodt, of binnen 80 tot 200 jaar.

    Arvind Narayanan en Sayash Kapoor stellen in een post van 26 juli 2024 dat voorspellingen van existentieel risico door AI veel te onbetrouwbaar zijn om bruikbaar te zijn voor beleid, en zelfs sterk misleidend. Een van hun redenen is dat er voor existentieel risico door AI geen referentieklasse bestaat, omdat het een gebeurtenis is zoals geen andere; wel is dat volgens hen een kwestie van gradatie, niet van soort. Volgens de titel van een van hun paragrafen zijn subjectieve kansen gevoelens die zich voordoen als getallen. Tegen zulke voorspellingen als academische bezigheid hebben ze volgens eigen zeggen geen bezwaar, en voor bedrijven en andere private beslissers kunnen zulke voorspellingen volgens hen nuttig zijn.

    Getallen veranderen onderweg. In het essay uit de vorige les, van november 2023, schrijft Buterin dat onderzoekers volgens een enquête uit 2022 gemiddeld een kans van 5 tot 10% geven dat AI ons letterlijk allemaal doodt. De paper van Grace en collega's over de enquête van 2023 geeft in een tabel ook de uitkomsten van 2022: twee van de vragen over uitsterven hadden een mediaan van 5% en 10%, en een gemiddelde van 15,7% en 20,5%. Met *gemiddeld* bedoelt Buterin dus kennelijk de medianen, niet de gemiddelden. Die lezing is van deze les. Melanie Mitchell zocht in een post van 23 april 2023 uit waar een veel gedeelde bewering vandaan kwam, dat de helft van de AI-onderzoekers AI een kans van minstens 10% geeft om ons allemaal te doden: de vraag erachter kreeg volgens Mitchell maar 162 antwoorden.

    ### Biological anchors

    Het rapport *Forecasting TAI with biological anchors* van Ajeya Cotra, gedateerd juli 2020, vraagt wanneer een transformatief model getraind zou kunnen worden. Cotra definieert transformative AI, kortweg TAI, naar het effect, als software (een computerprogramma of een verzameling programma's) met minstens zo'n diepe invloed op de koers van de wereld als de Industriële Revolutie. De methode vergelijkt de compute die training zou kosten met de biologie. Het lifetime anchor is het rekenwerk van een menselijk brein in de eerste miljard seconden, tot ongeveer 32 jaar, volgens Cotra 1e15 FLOP per seconde maal 1e9 seconden, samen 1e24 FLOP. Vanuit elk anchor komt Cotra met een aanpassing tot een hypothese over training: de Lifetime Anchor-hypothese schat een mediaan van ongeveer 1e29 FLOP. Het evolution anchor is het rekenwerk van de hele evolutie, ongeveer 1e41 FLOP.

    Cotra kiest niet één hypothese, maar weegt ze allemaal mee in één voorspelling. De drie hypotheses over neural networks krijgen 20% (Short Horizon), 30% (Medium Horizon) en 15% (Long Horizon). Verder geeft Cotra 5% aan het Lifetime Anchor, 10% aan het Genome Anchor, 10% aan het Evolution Anchor, en 10% aan de mogelijkheid dat het trainen van een transformatief model met de architecturen en algoritmen van 2020 meer rekenwerk kost dan elk van de hypotheses voorspelt, misschien zelfs astronomisch veel meer.

    Met de aannames die Cotra het waarschijnlijkst vindt, komt het rapport uit op een mediaan van ongeveer 2052, 32 jaar na het schrijven, voor wanneer de compute om een transformatief model te trainen betaalbaar zou kunnen worden; met agressieve aannames op ongeveer 2036, en met voorzichtige op ongeveer 2100. Cotra neemt voorlopig ongeveer 2050 als mediaan voor TAI. Over zichzelf zegt het rapport dat het werk in uitvoering is, en niet het standpunt van Open Philanthropy als instelling. Cotra deelde het in september 2020 op LessWrong als een conceptrapport over wanneer transformative AI komt. Twee jaar later, in augustus 2022, schoof Cotra de mediaan op naar ongeveer 2040.

    ### Waar ze het oneens zijn

    Over de vraag of je zulke getallen moet gebruiken: voor de auteurs van de paper over de enquête van 2023 zijn ze één belangrijk stukje van de puzzel, voor Narayanan en Kapoor veel te onbetrouwbaar om bruikbaar te zijn voor beleid. Over wie je het vraagt: voor uitsterven door AI, uiterlijk in 2100, was de mediaan van de AI-experts in het XPT 3% en die van de superforecasters 0,38%, en volgens de paper houden de verschillen tussen experts en superforecasters stand bij elke manier van samenvoegen. En over hoe je het vraagt: in elke editie van de enquête leverde de vraag naar de kans in een gegeven jaar latere voorspellingen op dan de vraag naar het jaar bij een gegeven kans. De drie vragen over uitsterven kregen daarentegen volgens het rapport vergelijkbare antwoorden; hun gemiddelden liggen tussen 17,5% en 18,5%.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **expertenenquête** | een enquête die veel onderzoekers in een vakgebied dezelfde vragen stelt, zoals hoe waarschijnlijk een uitkomst is of wanneer iets kan; de antwoorden zijn opgegeven verwachtingen, geen metingen |
        | **p(doom)** | de kans die iemand geeft dat AI tot een existentiële catastrofe leidt; waar die kans precies over gaat, en tot wanneer, verschilt van persoon tot persoon |
        | **biological anchors** | de methode van Cotra (2020) om transformative AI te voorspellen: schat de compute die training zou kosten door te vergelijken met de biologie, zoals een menselijk brein gedurende een leven of de hele evolutie, en vraag wanneer die compute betaalbaar wordt |
        | superforecaster | in het XPT iemand die hoog scoorde in eerdere toernooien in voorspellen |
        | HLMI | high-level machine intelligence, in de enquête: als machines zonder hulp elke taak beter en goedkoper kunnen doen dan menselijke werknemers |
        | TAI | transformative AI, in het rapport van Cotra: software met minstens zo'n diepe invloed op de wereld als de Industriële Revolutie |
        """),
        L.watch_out("Een antwoord in een enquête is een opgegeven kans, geen gemeten risico. Zeg wie er gevraagd is, welke vraag, en welk getal van alle antwoorden: een mediaan, een gemiddelde of een aandeel. De antwoorden in oefening 1 en de kansen in oefening 2 zijn verzonnen; alleen de weging in oefening 2 komt van Cotra."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo lees je een getal over wat experts verwachten:

    1. Zoek wie er gevraagd is, hoe ze gekozen zijn, en hoeveel er antwoordden.
    2. Lees de precieze vraag: welke uitkomst, en tot wanneer.
    3. Kijk welk ene getal voor alle antwoorden staat, en wat voor getal het is.
    4. Kijk hoe ver de antwoorden uiteenlopen, en hoeveel mensen nul gaven.
    5. Worden meerdere antwoorden of hypotheses samengevoegd tot één, zoek dan uit hoe zwaar elk meetelt.
    6. Lees wat de auteurs zelf zeggen dat het getal waard is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent het gemiddelde en de mediaan uit van vijf verzonnen antwoorden, in procent, op een vraag zoals die van de enquête.
    """)
    return


@app.cell
def example_answers():
    import numpy as np

    answers = np.array([1, 2, 5, 10, 90])   # verzonnen: vijf antwoorden, in procent
    print(answers.mean(), np.median(answers))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het gemiddelde is 21,6 en de mediaan 5. Vier van de vijf antwoorden zijn 10 of minder; het ene antwoord van 90 trekt het gemiddelde omhoog tot meer dan vier keer de mediaan. De mediaan blijft 5 als dat antwoord 50 of 100 wordt: het middelste antwoord telt, en dat blijft hetzelfde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_describe():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. describe: vier getallen voor veel antwoorden

        Twintig verzonnen antwoorden, in procent, op een vraag zoals die van de enquête. Schrijf `describe(answers)`, die een lijst van vier getallen teruggeeft: `mean`, het gemiddelde van de antwoorden; `median`, de mediaan; `low`, het 0,25-kwantiel; en `high`, het 0,75-kwantiel. Tussen `low` en `high` ligt de middelste helft van de antwoorden, zoals de middelste 50% in het rapport. Maak eerst een array `a` van `answers`, en geef `[mean, median, low, high]` terug.

        De check roept `describe([10, 0, 5, 50, 1, 20, 10, 2, 100, 5, 1, 10, 30, 0, 15, 5, 80, 1, 20, 10])` aan, de twintig verzonnen antwoorden, en daarna andere gevallen.
        """),
        L.task("Het gemiddelde, de mediaan en de middelste helft van de antwoorden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_describe():
    box_describe = L.code_box("describe", lines=10)
    box_describe
    return (box_describe,)


@app.cell(hide_code=True)
def check_describe(box_describe):
    result_describe = L.check("describe", box_describe.value)
    result_describe
    return (result_describe,)


@app.cell(hide_code=True)
def hints_describe():
    L.hints(
        "describe",
        [
            "Begin klein: de vijf antwoorden 1, 2, 5, 10 en 90 uit het uitgewerkte voorbeeld. Wat zijn hun gemiddelde en hun mediaan, en tussen welke twee antwoorden ligt de middelste helft?",
            "Maak een array `a` van `answers`. Reken `mean` uit met `.mean()` en `median` met `np.median`; `low` en `high` komen uit `np.quantile`, met als fractie het deel van de antwoorden dat eronder ligt. Geef de vier terug in een lijst.",
            "`    low = np.quantile(a, ___)`",
        ],
        why="`a = np.array(answers)`, dan `mean = a.mean()`, `median = np.median(a)`, `low = np.quantile(a, 0.25)` en `high = np.quantile(a, 0.75)`, en `return [mean, median, low, high]`. Voor de antwoorden 1, 2, 5, 10 en 90 is het gemiddelde 21,6, de mediaan 5, en ligt de middelste helft tussen 2 en 10.",
    )
    return


@app.cell(hide_code=True)
def text_after_describe():
    mo.md(r"""
    Voor de twintig verzonnen antwoorden is het gemiddelde 18,75 en de mediaan 10, en loopt de middelste helft van 1,75 tot 20. Dertien van de twintig antwoorden zijn 10 of minder, en toch is het gemiddelde bijna twee keer de mediaan: de antwoorden 50, 80 en 100 trekken het omhoog. De verzonnen antwoorden lijken expres op die van de enquête: in december 2024 was het gemiddelde 18,2, de mediaan 10, en liep de middelste helft van 1 tot 25.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mixture():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. mixture: één kans uit meerdere hypotheses

        Een verzonnen model van de stap waarin Cotra de hypotheses weegt, niet het model van Cotra zelf. De zeven getallen in `weights` zijn die van Cotra, in de volgorde van hierboven: korte, middellange en lange horizon, lifetime, genome, evolution, en meer dan al die hypotheses voorspellen. De kansen zijn verzonnen: voor elke mogelijkheid de kans dat de compute voor een transformatief model uiterlijk in een verzonnen jaar betaalbaar is. De kans in totaal is elke kans maal het gewicht, opgeteld.

        Schrijf `mixture(weights, chances)`, die die totale kans teruggeeft. `weights` en `chances` zijn lijsten van dezelfde lengte. Maak arrays `w` van `weights` en `c` van `chances`, en geef de som terug van elk gewicht maal de bijbehorende kans.

        De check roept `mixture([0.2, 0.3, 0.15, 0.05, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.9, 0.4, 0.0, 0.0])` aan, en daarna andere gevallen.
        """),
        L.task("De totale kans, uit `weights` en `chances`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mixture():
    box_mixture = L.code_box("mixture", lines=7)
    box_mixture
    return (box_mixture,)


@app.cell(hide_code=True)
def check_mixture(box_mixture):
    result_mixture = L.check("mixture", box_mixture.value)
    result_mixture
    return (result_mixture,)


@app.cell(hide_code=True)
def hints_mixture():
    L.hints(
        "mixture",
        [
            "Begin klein: twee hypotheses, de ene met gewicht 0,75 en kans 0,4, de andere met gewicht 0,25 en kans 0,8. Hoeveel levert elke hypothese op, en wat is de kans in totaal?",
            "Maak arrays `w` van `weights` en `c` van `chances`. Als je de twee arrays vermenigvuldigt, wordt elk gewicht vermenigvuldigd met de eigen kans; tel die producten op met NumPy en geef het totaal terug.",
            "`    return np.sum(___)`",
        ],
        why="`w = np.array(weights)` en `c = np.array(chances)`, dan `return np.sum(w * c)`. Met gewicht 0,75 en kans 0,4 levert de eerste hypothese 0,3 op, met gewicht 0,25 en kans 0,8 de tweede 0,2: samen 0,5.",
    )
    return


@app.cell(hide_code=True)
def text_after_mixture():
    mo.md(r"""
    Voor de verzonnen kansen is de uitkomst 0,29, of 29%. De twee mogelijkheden zonder enige kans tellen ook mee: hun 20% van het gewicht trekt de uitkomst omlaag. De eigen getallen van Cotra voor 2036 komen uit het hele model. Volgens het rapport is er ongeveer 23% kans dat de compute om een transformatief model te trainen uiterlijk in 2036 beschikbaar is, en Cotra stelt dat bij tot ongeveer 18% als kans op TAI uiterlijk in 2036, alles meegewogen. Een heel brede marge, van ongeveer 2% tot ongeveer 45%, zou volgens Cotra misschien te verdedigen zijn.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: een paar hoge antwoorden

    De twintig verzonnen antwoorden uit oefening 1 blijven hetzelfde. Voeg één voor één antwoorden van 100 toe, en kijk wat het gemiddelde en de mediaan doen.
    """)
    return


@app.cell(hide_code=True)
def demo_extra():
    extra = L.explore("extra", "extra: hoeveel antwoorden van 100 erbij komen",
                      lambda _v: C.demo_frame(_v, L.lang), start=1, values=C.DEMO_EXTRA, role="",
                      task="Zoek het kleinste aantal extra antwoorden van 100 waarbij het gemiddelde meer dan drie keer de mediaan is.")
    extra
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Het gemiddelde is 18,75 zonder extra antwoorden; 22,62 met 1 extra antwoord van 100; 26,14 met 2; 29,35 met 3; 32,29 met 4; en 35 met 5. De mediaan blijft de hele tijd 10. Vanaf 4 extra antwoorden is het gemiddelde meer dan drie keer de mediaan, terwijl 13 van de 24 antwoorden nog steeds 10 of minder zijn. De XPT-paper verklaart de eigen gemiddelden net zo: volgens de paper lagen de voorspellingen op deze vragen meestal tussen 0 en 5%, met een lange staart van hogere voorspellingen die het gemiddelde omhoog trekt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Oefening 3. Een kop in het nieuws

    Een nieuwssite schrijft dat AI-onderzoekers de kans dat AI de mensheid uitroeit op 18% schatten, volgens de enquête van december 2024.
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Wat had de nieuwssite erbij moeten zeggen?",
        options={
            "nothing": "Niets, want 1.489 antwoorden maken van 18% een gemeten risico",
            "right": "Dat 18% het gemiddelde is, en dat de mediaan 10% was",
            "few": "Dat maar een paar dozijn onderzoekers aan de enquête hebben meegedaan",
        },
        correct="right",
        why={
            "right": "Het rapport noemt minstens ongeveer 18% als gemiddelde van de groep, en 10% als mediaan, en volgens het rapport liep de middelste helft van de schattingen van 1% tot 25%. De vraag ging ook over uitsterven of een even blijvende en ernstige machteloosheid van de mensheid, niet alleen over het uitroeien van de mensheid.",
            "nothing": "1.489 antwoorden maken een grote enquête, maar elk antwoord is een opgegeven kans, geen meting; volgens de auteurs van de paper over de enquête van 2023 zijn voorspellingen van AI-experts niet te zien als een betrouwbare gids naar de objectieve waarheid. Zie *Expertenenquêtes*.",
            "few": "1.580 onderzoekers bleven over voor de hoofdanalyse, en 1.489 van hen beantwoordden een van de drie vragen over uitsterven. Wat de kop weglaat, is welk getal 18% is: het gemiddelde, met een mediaan van 10%. Zie *Expertenenquêtes*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_exercise_pdoom():
    mo.md(r"""
    ### Oefening 4. Een budget naar p(doom)

    Een regering wil het budget voor AI-veiligheid laten afhangen van de p(doom) van experts: hoe hoger de mediaan in de laatste enquête, hoe meer geld.
    """)
    return


@app.cell(hide_code=True)
def choice_pdoom_cell():
    choice_pdoom = L.choice(
        "pdoom",
        question="Welk bezwaar zouden Narayanan en Kapoor maken, volgens hun post van juli 2024?",
        options={
            "low": "Experts geven altijd veel te lage getallen",
            "right": "Zulke voorspellingen zijn te onbetrouwbaar voor beleid",
            "pointless": "AI-risico voorspellen heeft geen zin, ook niet als academisch onderzoek",
        },
        correct="right",
        why={
            "right": "Volgens hun centrale stelling zijn voorspellingen van existentieel risico door AI veel te onbetrouwbaar om bruikbaar te zijn voor beleid, en zelfs sterk misleidend. Voor existentieel risico door AI bestaat er volgens hen geen referentieklasse, omdat het een gebeurtenis is zoals geen andere.",
            "low": "Ze zeggen niet dat de getallen te laag zijn, of te hoog: ze zeggen dat zulke getallen te onbetrouwbaar zijn om bruikbaar te zijn voor beleid. Zie *p(doom)*.",
            "pointless": "Tegen zulke voorspellingen als academische bezigheid hebben ze volgens eigen zeggen geen bezwaar, en voor bedrijven en andere private beslissers kunnen zulke voorspellingen volgens hen nuttig zijn. Hun bezwaar geldt beleid dat op zulke getallen gebouwd is. Zie *p(doom)*.",
        },
    )
    choice_pdoom
    return (choice_pdoom,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Andere standpunten](/course/open/rai-xrisk-counterpositions?lang=nl) zette standpunten over AI-risico naast elkaar, en Buterin noemde daar getallen van anderen. Deze les keek waar zulke getallen vandaan komen: expertenenquêtes zoals de ESPAI van AI Impacts, een toernooi van superforecasters en experts, de persoonlijke p(doom), en de biological anchors van Cotra. Met [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl) kun je ze lezen: welke vraag, welk getal van alle antwoorden, en hoe ver ze uiteenlopen. De volgende les, *Kan een AI-systeem moreel ertoe doen?*, kijkt niet meer naar wat AI met mensen kan doen, maar naar hoe mensen met AI moeten omgaan.

    **Een voorbeeld uit het dagelijks leven.** Vijf vrienden schatten hoe lang een autorit duurt. Vier zeggen ongeveer 2 uur, één zegt 6. Het gemiddelde is 2,8 uur en de mediaan 2. Dat de vrienden gemiddeld 2,8 uur verwachten, klopt, en is toch misleidend. En als geen van hen die weg ooit heeft gereden, zegt geen van beide getallen veel over de rit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_headline, choice_pdoom, result_describe, result_mixture):
    L.wrap_up(result_describe, result_mixture, choice_headline, choice_pdoom)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
