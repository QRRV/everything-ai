# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-bias-speech-ads. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-speech-ads", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Bias in gezichtsanalyse en beeldgeneratoren](/course/open/rai-bias-faces-images?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pool():
    quick_pool = L.choice(
        "quick-pool",
        question="Een systeem maakt 30 fouten bij 200 gezichten uit groep A, en 0 fouten bij 100 gezichten uit groep B. Wat is het foutpercentage van A en B samen?",
        options={
            "right": "0,1",
            "mean": "0,075",
            "only-a": "0,15",
        },
        correct="right",
        why={
            "right": "30 fouten bij 300 gezichten: 0,1. Tel de aantallen van de vakjes op, en deel één keer.",
            "mean": "Dat is het gemiddelde van 0,15 en 0,0, alsof beide groepen even groot zijn. Tel eerst de fouten en de gezichten op: 30 / 300. Zie *Recept* in [Bias in gezichtsanalyse en beeldgeneratoren](/course/open/rai-bias-faces-images?lang=nl).",
            "only-a": "Dat is het foutpercentage van alleen groep A, 30 van de 200. Samen is het 30 fouten bij 300 gezichten. Zie *Recept* in [Bias in gezichtsanalyse en beeldgeneratoren](/course/open/rai-bias-faces-images?lang=nl).",
        },
    )
    quick_pool
    return (quick_pool,)


@app.cell(hide_code=True)
def quick_check_hide():
    quick_hide = L.choice(
        "quick-hide",
        question="Een rapport zegt dat een spraakherkenner even goed werkt voor vrouwen als voor mannen, en even goed voor oudere als voor jongere sprekers. Wat kan het dan nog verbergen?",
        options={
            "nothing": "Niets, als beide getallen precies gelijk zijn",
            "right": "Een slechter resultaat bij oudere vrouwen",
            "one": "Alleen een verschil per eigenschap",
        },
        correct="right",
        why={
            "nothing": "Gelijke getallen per eigenschap kunnen een combinatie verbergen: dat is fairness gerrymandering. Zie *Intersectional fairness* in [Bias in gezichtsanalyse en beeldgeneratoren](/course/open/rai-bias-faces-images?lang=nl).",
            "right": "Een getal per eigenschap mengt de combinaties, zoals de 0,107 van Microsoft voor vrouwen de 0,208 van vrouwen met een donkerdere huid verborg. Een combinatie zie je pas als je per combinatie telt.",
            "one": "Het rapport laat al zien dat er geen verschil per eigenschap is. Wat het kan verbergen, is een combinatie, zoals oudere vrouwen. Zie *Intersectional fairness* in [Bias in gezichtsanalyse en beeldgeneratoren](/course/open/rai-bias-faces-images?lang=nl).",
        },
    )
    quick_hide
    return (quick_hide,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je praat tegen een telefoon, een auto of een klantenservice, en een systeem schrijft op wat je zegt. Allison Koenecke en collega's (2020) gaven vijf van zulke systemen interviews met zwarte en met witte sprekers in de VS. Bij elk systeem waren de foutpercentages voor zwarte sprekers bijna twee keer zo groot. Een systeem dat je slechter verstaat, laat je alles herhalen, zet verkeerde woorden in je ondertitels, of laat je niet door. Advertenties hebben een stillere versie van hetzelfde probleem. Een adverteerder kan ervoor kiezen een vacature aan iedereen te laten zien, en toch beslist het platform wie van hen hem echt te zien krijgt. In een test van Muhammad Ali en collega's (2019) bereikten vijf advertenties voor banen in de houtindustrie meer dan 90 procent mannen, terwijl vijf advertenties voor schoonmakers, met hetzelfde publiek en dezelfde biedstrategie, meer dan 65 procent vrouwen bereikten.

    **Een voorbeeld uit het dagelijks leven.** Een winkel betaalt iemand om in de hele stad folders voor een vacature rond te brengen, en om voor dat geld zoveel mogelijk mensen te bereiken. De flats op de heuvel kosten meer loopwerk, dus de folders belanden in de straten beneden. Niemand koos ervoor de heuvel over te slaan, en toch horen de mensen daar nooit van de vacature.
    """)
    return


@app.cell(hide_code=True)
def predict_clicks():
    guess_clicks = L.guess(
        "clicks",
        question="Een vacature voor wetenschap en techniek moest vrouwen en mannen evengoed bereiken, maar minder vrouwen zagen hem. Klikten vrouwen die hem wel zagen er minder vaak of vaker op dan mannen?",
        options={
            "less": "Minder vaak",
            "same": "Even vaak",
            "right": "Vaker",
        },
        reveal="**Vaker.** In de data van Anja Lambrecht en Catherine Tucker (2019) klikten vrouwen 1,77 keer per 1.000 keer dat ze de advertentie zagen, mannen 1,41 keer. Het platform liet de advertentie dus niet minder aan vrouwen zien omdat ze er minder belangstelling voor hadden. *Het idee* legt uit wat er wel gebeurde.",
    )
    guess_clicks
    return (guess_clicks,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Word error rate.** Automatische spraakherkenning, kort ASR (*automatic speech recognition*), zet spraak om in geschreven tekst. Om die te beoordelen, leg je wat het systeem schreef naast een transcriptie die mensen maakten, en tel je de woorden die het systeem verving (S), wegliet (D) en toevoegde (I). De word error rate is hun som gedeeld door het aantal woorden in de juiste transcriptie (N): WER = (S + D + I) / N. Lager is beter. Omdat toegevoegde woorden ook meetellen, kan de WER boven 1 uitkomen.

        **Spraakherkenning.** Koenecke en collega's testten de systemen van Amazon, Apple, Google, IBM en Microsoft op interviews met 42 witte en 73 zwarte sprekers uit vijf Amerikaanse steden: 19,8 uur audio, afgestemd op de leeftijd en het gender van de sprekers, in 2.141 fragmenten per groep, elk ongeveer 17 seconden lang. De gemiddelde WER was 0,35 voor zwarte sprekers en 0,19 voor witte sprekers. Microsoft deed het het best, met 0,27 tegenover 0,15; Apple het slechtst, met 0,45 tegenover 0,23. Per combinatie, net als bij *Gender Shades*: 0,41 voor zwarte mannen, 0,30 voor zwarte vrouwen, 0,21 voor witte mannen en 0,17 voor witte vrouwen. Als een WER boven 0,5 een transcriptie onbruikbaar maakt, was 23 procent van de fragmenten van zwarte sprekers onbruikbaar, tegenover 1,6 procent van die van witte sprekers.

        Waar komt het verschil vandaan? In de interviews zaten ook 206 korte zinnetjes, van 5 tot 8 woorden, die zwarte en witte sprekers allebei zeiden. Op die identieke zinnetjes bleef het verschil:

        | Systeem | zwarte sprekers | witte sprekers |
        |---|---|---|
        | Apple | 0,28 | 0,12 |
        | IBM | 0,21 | 0,10 |
        | Google | 0,17 | 0,11 |
        | Amazon | 0,18 | 0,08 |
        | Microsoft | 0,13 | 0,07 |

        *Gemiddelde WER op 206 identieke zinnetjes, uit tabel 2 van Koenecke en collega's.*

        De woorden waren hetzelfde, dus het verschil wijst op hoe ze klinken. De auteurs herleiden het tot de acoustic models, het deel van een systeem dat korte stukjes audio koppelt aan spraakklanken: hun uitkomsten wijzen erop dat de systemen in de war raken van hoe African American Vernacular English klinkt, en niet van de grammatica of de woordkeus. Ze schrijven dat de WER op deze zinnetjes bij elk systeem ongeveer twee keer zo groot is. Lees ook de tabel: bij Google is het 0,17 tegenover 0,11, 1,55 keer.

        **Niet één richting.** Rachael Tatman (2017) testte de automatische ondertitels van YouTube op 80 sprekers, acht vrouwen en acht mannen uit elk van vijf regio's: Californië, Georgia, New England, Nieuw-Zeeland en Schotland. De WER was hoger voor vrouwen, en hoger voor sprekers uit Schotland. Tatman noemt de uitkomst voor vrouwen enigszins verrassend, omdat eerdere studies het omgekeerde vonden; Koenecke en collega's vonden hem hoger voor mannen. Er is geen vuistregel voor welke groep een systeem slechter verstaat: je meet het.

        **Targeting en delivery.** Op een advertentieplatform kiest de adverteerder wie een advertentie mag zien, zoals alle volwassenen in een stad: de targeting. Daarna beslist het platform, veiling na veiling, wie van hen hem echt te zien krijgt: de delivery. Facebook legde zijn veiling zo uit: de advertentie die een veiling wint en getoond wordt, is die met de hoogste totale waarde, opgebouwd uit het bod, de geschatte kans op een actie, en de kwaliteit en relevantie van de advertentie. Ali en collega's lieten hun eigen advertenties lopen, elk met hetzelfde publiek. Alleen al het budget verschoof het aandeel mannen van meer dan 55 procent, bij advertenties met een heel laag budget, naar minder dan 45 procent, bij advertenties met een hoog budget. Hun vacatures liepen 24 uur, allemaal met hetzelfde publiek en dezelfde biedstrategie: de banen in de houtindustrie bereikten meer dan 90 procent mannen en meer dan 70 procent witte gebruikers, de banen als schoonmaker meer dan 65 procent vrouwen en meer dan 75 procent zwarte gebruikers. Advertenties voor woningen bereikten naar schatting 27 procent witte gebruikers bij luxe huurwoningen, en 49 procent bij goedkope koophuizen.

        **Een goedkoper publiek.** Anja Lambrecht en Catherine Tucker (2019) bekeken een vacature voor wetenschap, techniek, engineering en wiskunde, die in 191 landen op Facebook werd getoond. Hij was genderneutraal bedoeld, en toch werd hij aan meer dan 20 procent meer mannen dan vrouwen getoond. Vrouwen die hem zagen, klikten vaker dan mannen, en dat sluit gebrek aan belangstelling uit. Hun verklaring: jongere vrouwen zijn een gewild publiek, dus een advertentie aan hen tonen kost meer. Een algoritme dat alleen op kosteneffectiviteit optimaliseert, toont de advertentie aan het goedkopere publiek: andere adverteerders drukken de vrouwen weg.

        **Niet de kwalificaties.** Is de scheefheid misschien terecht, omdat de banen beter bij één groep passen? Basileal Imana, Aleksandra Korolova en John Heidemann (2021) lieten paren advertenties lopen voor hetzelfde soort baan bij twee bedrijven met een ander aandeel vrouwen, zoals bezorgers voor Domino's, waar 98 procent man is, en voor Instacart, waar meer dan 50 procent vrouw is. Op Facebook bereikte de advertentie van Domino's een groter aandeel mannen dan die van Instacart. Beide advertenties liepen tegelijk, voor vergelijkbare banen, dus ze vragen vergelijkbare kwalificaties: die kunnen het verschil niet verklaren. Op LinkedIn vonden ze zo'n scheefheid niet; een van de negen runs daar lag net boven de grens voor statistische significantie, in de onverwachte richting.

        **Wat eraan gedaan werd.** In juni 2022 schikte het Amerikaanse ministerie van Justitie een zaak met Meta onder de Fair Housing Act, de Amerikaanse wet tegen discriminatie bij wonen: de eerste zaak waarin het algoritmische bias onder die wet aanvocht. Het ministerie stelde dat de delivery van Meta machine learning gebruikte die deels steunde op ras, nationale afkomst en geslacht, om mede te bepalen wie van het publiek van een adverteerder een advertentie voor een woning zag. In een schikking die een rechter goedkeurde, beloofde Meta te stoppen met een hulpmiddel voor woningadvertenties dat Special Ad Audience heet, de hoogst mogelijke boete van 115.054 dollar te betalen, en het Variance Reduction System (VRS) te bouwen: voor woningadvertenties verkleint het het verschil tussen het publiek dat in aanmerking komt, de gebruikers die bij de targeting van de adverteerder passen, en het werkelijke publiek, wie de advertentie zag, naar geslacht en naar geschat ras of geschatte etniciteit. Ras wordt geschat uit achternamen en woonplaatsen, niet gevraagd. In januari 2023 spraken de twee partijen doelen af voor het overgrote deel van de woningadvertenties: aan het eind van dat jaar een verschil van hooguit 10 procent bij 91,7 procent van die advertenties naar geslacht, en bij 81,0 procent naar ras of etniciteit. De schikking stelde dit tot 27 juni 2026 onder toezicht van een rechter.

        **In de EU.** In de EU verbiedt de Digital Services Act sinds 17 februari 2024 onlineplatforms om advertenties te tonen op basis van profilering met bijzondere persoonsgegevens, zoals ras of etnische afkomst, en advertenties op basis van profilering van minderjarigen; bij elke advertentie moet staan wat de belangrijkste factoren waren om te bepalen wie hem ziet. Een scheefheid die uit prijzen en optimalisatie voortkomt, zoals hierboven, is niet waar die regels over gaan.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **spraakherkenning** (ASR) | spraak omzetten in geschreven tekst, zoals ondertitels |
        | **word error rate** (WER) | (S + D + I) / N: de vervangen, weggelaten en toegevoegde woorden, gedeeld door de woorden in de juiste transcriptie; lager is beter, en hij kan boven 1 uitkomen |
        | **targeting** en **delivery** | de adverteerder kiest wie een advertentie mag zien: de targeting; het platform beslist wie van hen hem echt ziet: de delivery |
        | **ad delivery bias** | een scheefheid in wie een advertentie ziet die voortkomt uit de delivery van het platform, zijn voorspellingen en de prijzen in zijn veilingen, en niet uit de targeting van de adverteerder |
        """),
        L.watch_out("23 procent van de fragmenten, niet 23 procent van de sprekers. Het deel telt fragmenten van ongeveer 17 seconden, elk met zijn WER gemiddeld over de vijf systemen. Zeg wat je telt: fragmenten, sprekers of mensen die een advertentie zagen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo controleer je een spraaksysteem of een advertentie op bias:

    1. Kies de groepen en de maat: de WER per groep sprekers, of het aandeel van elke groep onder de mensen die een advertentie zagen.
    2. Vergelijk gelijk met gelijk: sprekers die op leeftijd en gender zijn afgestemd, of dezelfde zinnetjes; advertenties met hetzelfde publiek, budget en bod.
    3. Eerst tellen, dan één keer delen per groep: fouten gedeeld door woorden, of de impressions van de groep gedeeld door alle impressions.
    4. Kijk naar een grens die voor mensen telt, zoals een WER boven 0,5 die een transcriptie onbruikbaar maakt, en zeg of je fragmenten of mensen telt.
    5. Zoek de oorzaak met een tweede test: identieke zinnetjes sluiten de woorden uit, en een paar vergelijkbare vacatures sluit de kwalificaties uit.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee fragmenten, met de hand naast elkaar gelegd. In het eerste zei iemand 'we gaan op vrijdag samen naar de markt' en schreef het systeem 'we gaan vrijdag samen naar een markt hoor': *op* is weggelaten, *de* is vervangen door *een*, en *hoor* is toegevoegd. In het tweede zei iemand 'dank je' en schreef het systeem 'denk je wel echt heel': één woord vervangen en drie toegevoegd. De loop rekent de WER van elk uit. Voer de cel uit.
    """)
    return


@app.cell
def example_snippets():
    # twee fragmenten: de woorden in de juiste transcriptie, en de vervangen, weggelaten en toegevoegde woorden
    _snippets = [
        {"words": 8, "substituted": 1, "deleted": 1, "inserted": 1},
        {"words": 2, "substituted": 1, "deleted": 0, "inserted": 3},
    ]

    for _snippet in _snippets:
        _errors = _snippet["substituted"] + _snippet["deleted"] + _snippet["inserted"]
        print(_snippet["words"], "woorden, WER", round(_errors / _snippet["words"], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    0,375 voor het eerste fragment: drie fouten op acht woorden. 2,0 voor het tweede: vier fouten op twee woorden, dus de WER ligt boven 1. Koenecke en collega's noemden een transcriptie onbruikbaar als de WER boven 0,5 ligt: de eerste is nog bruikbaar, de tweede niet. Welk deel van een reeks fragmenten is onbruikbaar? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_wer():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Onbruikbare transcripties

        `snippets` bevat tien fragmenten van één spreker, verzonnen door de course. Reken `unusable` uit: het deel van de fragmenten met een WER boven `limit`. Begin met `count = 0`, loop met `snippet` over `snippets`, reken zijn `errors` en zijn `wer` uit zoals in het uitgewerkte voorbeeld, en tel 1 op bij `count` als `wer > limit`. Deel na de loop `count` door het aantal fragmenten, `len(snippets)`; `unusable` is de laatste regel, zodat de check het ziet. De check probeert ook een lagere grens.
        """),
        L.task("Reken `unusable` uit: het deel van de fragmenten met een WER boven `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wer():
    box_wer = L.code_box("wer", lines=25)
    box_wer
    return (box_wer,)


@app.cell(hide_code=True)
def check_wer(box_wer):
    result_wer = L.check("wer", box_wer.value)
    result_wer
    return (result_wer,)


@app.cell(hide_code=True)
def hints_wer():
    L.hints(
        "wer",
        [
            "Eerst de WER van één fragment, zoals in het uitgewerkte voorbeeld; tel dan hoeveel fragmenten boven de grens liggen, en deel door hoeveel er zijn.",
            "In de loop: `errors` is de som van de drie aantallen, `wer = errors / snippet[\"words\"]`, en onder `if wer > limit:` tel je 1 op bij `count`.",
            "Na de loop, niet erin: `unusable = count / ___`.",
        ],
        why="`count = 0`, dan `for snippet in snippets:` met daaronder `errors = snippet[\"substituted\"] + snippet[\"deleted\"] + snippet[\"inserted\"]` en `wer = errors / snippet[\"words\"]`, en `if wer > limit:` met `count = count + 1`. Na de loop `unusable = count / len(snippets)`. Dat geeft 4 / 10 = `0.4`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cause():
    mo.md(r"""
    ### Oefening 2. Dezelfde woorden, meer fouten
    """)
    return


@app.cell(hide_code=True)
def choice_cause_cell():
    choice_cause = L.choice(
        "cause",
        question="Op 206 identieke zinnetjes, gezegd door zwarte en door witte sprekers, maakten de vijf systemen nog steeds meer fouten bij de zwarte sprekers. Waar wijst dat op?",
        options={
            "words": "Zeldzamere woorden in wat de sprekers zeiden",
            "right": "Hoe de woorden klinken",
            "grammar": "De grammatica van de zinnen van de sprekers",
        },
        correct="right",
        why={
            "words": "De zinnetjes waren identiek, dus beide groepen zeiden dezelfde woorden. Zie *Spraakherkenning* bij *Het idee*.",
            "right": "De woorden en de grammatica waren hetzelfde, dus wat verschilt, is hoe ze klinken. Koenecke en collega's herleiden het verschil tot de acoustic models, het deel van een systeem dat korte stukjes audio koppelt aan spraakklanken.",
            "grammar": "De zinnetjes waren identiek, dus de grammatica was voor beide groepen hetzelfde. Zie *Spraakherkenning* bij *Het idee*.",
        },
    )
    choice_cause
    return (choice_cause,)


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Wie zag de advertentie?

        `cells` bevat tabel 2 van Lambrecht en Tucker: per leeftijdsgroep en gender de `"impressions"`, het aantal keer dat iemand de advertentie zag, en de `"clicks"`. Reken `share` uit: het aandeel van `gender` in alles wat `count` telt, bij `count = "impressions"` en `gender = "female"` het aandeel van de vrouwen in de impressions. Begin met `total = 0` en `mine = 0`, loop met `cell` over `cells`, tel van elk vakje `cell[count]` op bij `total`, en tel het bij `cell["gender"] == gender` ook op bij `mine`. Na de loop is `share` gelijk aan `mine` gedeeld door `total`. De check probeert ook de clicks, en de mannen.
        """),
        L.task("Reken `share` uit: de vakjes van `gender` opgeteld voor `count`, gedeeld door alle vakjes opgeteld voor `count`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=27)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Een aandeel is een deel gedeeld door het geheel: de impressions van de vrouwen gedeeld door alle impressions. Tel beide op in één loop, en deel één keer, erna.",
            "In de loop gaat elk vakje in `total`: `total = total + cell[count]`. Onder `if cell[\"gender\"] == gender:` gaat het vakje ook in `mine`.",
            "Na de loop: `share = ___ / total`.",
        ],
        why="`total = 0` en `mine = 0`, dan `for cell in cells:` met daaronder `total = total + cell[count]`, en `if cell[\"gender\"] == gender:` met `mine = mine + cell[count]`. Na de loop `share = mine / total`. Dat geeft 1.983.915 / 4.515.713 = `0.439`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_audience():
    mo.md(r"""
    ### Oefening 4. Wat vergelijkt het VRS?
    """)
    return


@app.cell(hide_code=True)
def choice_audience_cell():
    choice_audience = L.choice(
        "audience",
        question="Het Variance Reduction System van Meta voor woningadvertenties meet een verschil tussen twee groepen gebruikers. Welke twee?",
        options={
            "everyone": "Alle gebruikers en wie de advertentie zag",
            "right": "De doelgroep en wie de advertentie zag",
            "clicks": "Wie de advertentie zag en wie erop klikte",
        },
        correct="right",
        why={
            "everyone": "Het VRS begint bij de targeting van de adverteerder, niet bij alle gebruikers: het publiek dat in aanmerking komt, zijn de gebruikers die bij die targeting passen. Zie *Wat eraan gedaan werd* bij *Het idee*.",
            "right": "Het publiek dat in aanmerking komt, de gebruikers die bij de targeting van de adverteerder passen, tegenover het werkelijke publiek, wie de advertentie zag, naar geslacht en naar geschat ras of geschatte etniciteit. Het pakt de scheefheid aan die de delivery toevoegt; de targeting blijft de keuze van de adverteerder.",
            "clicks": "Clicks horen er niet bij: het VRS vergelijkt wie de advertentie had kunnen zien met wie hem zag. Zie *Wat eraan gedaan werd* bij *Het idee*.",
        },
    )
    choice_audience
    return (choice_audience,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Bias in gezichtsanalyse en beeldgeneratoren](/course/open/rai-bias-faces-images?lang=nl) telde je fouten per combinatie van groepen. Spraakherkenning heeft een eigen foutmaat, de WER, en dezelfde les: tel per groep en per combinatie, en kijk naar een grens die ertoe doet, zoals een transcriptie die je niet kunt gebruiken. Bij advertenties is de vraag niet hoe vaak een systeem het mis heeft, maar wie een kans te zien krijgt: de targeting van de adverteerder kan neutraal zijn terwijl de delivery dat niet is. Dat is ad delivery bias, en het komt voort uit optimaliseren op clicks en kosten, zonder dat iemand een groep kiest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_audience, choice_cause, result_share, result_wer):
    L.wrap_up(result_wer, choice_cause, result_share, choice_audience)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
