# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-lifecycle. Route: course/route/lessen/systemen.yaml. Checks, het leven van de spamdienst en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-lifecycle", "nl")
    C = L.checks   # het leven van de spamdienst van de course, de fasen, de rollen en de afbeeldingen


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
def quick_check_parts():
    quick_parts = L.choice(
        "quick-parts",
        question="Welk deel van de spamdienst van de course merkt dat de antwoorden er anders uitzien dan normaal?",
        options={
            "verification": "Controle, het deel dat elk binnenkomend bericht nakijkt",
            "right": "Monitoring, het deel dat de antwoorden bekijkt",
            "serving": "Serving, het deel dat altijd antwoordt",
            "collection": "Verzamelen, het deel dat elk bericht één keer aanneemt",
        },
        correct="right",
        why={
            "right": "Monitoring bekijkt de antwoorden door de tijd heen, zoals het aandeel spam of het aandeel antwoorden zonder score. In het leven van een systeem vertelt de monitoring het team dat het tijd is om opnieuw te kijken: deze les volgt een systeem door dat leven.",
            "verification": "Controle bekijkt één bericht tegelijk: is het bruikbaar? Welk deel bekijkt veel antwoorden samen? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "serving": "Serving maakt van de kans van het model een antwoord, één bericht tegelijk. Welk deel bekijkt de antwoorden die naar buiten gaan? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "collection": "Verzamelen zorgt dat elk bericht één keer behandeld wordt. Welk deel bekijkt de antwoorden door de tijd heen? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_parts
    return (quick_parts,)


@app.cell(hide_code=True)
def quick_check_debt():
    quick_debt = L.choice(
        "quick-debt",
        question="Een team neemt een shortcut in de code om een dienst een week eerder in gebruik te nemen. Wat zegt technische schuld over de shortcut?",
        options={
            "right": "Je wint nu tijd, en het kost later meer, elke keer dat iemand eromheen moet werken",
            "tests": "Het kost niets, zolang de tests nog slagen",
            "never": "Het telt alleen bij code die nooit meer verandert",
            "data": "De schuld lost zichzelf af zodra het model opnieuw getraind wordt op verse data, zonder extra werk",
        },
        correct="right",
        why={
            "right": "Een shortcut werkt als een lening: de rente komt later, elk jaar dat het systeem draait en iemand de code rond de shortcut moet veranderen. Deze les gaat over deze jaren.",
            "tests": "Tests laten zien wat de code nu doet. Wat gebeurt er elke keer dat iemand de code rond de shortcut moet veranderen? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "never": "Code die nooit verandert, kost weinig. Hoe lang wordt een systeem eigenlijk veranderd en onderhouden? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "data": "Opnieuw trainen verandert het model, niet de code eromheen. Wie betaalt voor een shortcut in deze code? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_debt
    return (quick_debt,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In 2008 begon Google met Flu Trends: uit wat mensen zochten, schatte Flu Trends hoeveel griep er in de Verenigde Staten was, sneller dan de CDC, de Amerikaanse gezondheidsdienst, die bezoeken aan artsen telt. In 2009 beschreven onderzoekers van Google het model in *Nature*. Bij de bouw kwamen de schattingen overeen met jaren aan cijfers van de CDC. In februari 2013 kwam Flu Trends in het nieuws omdat de schatting er ver naast zat, en in augustus 2015 stopte Google met het publiceren van de schattingen. Het model werd in 2008 gebouwd en in 2009 bijgewerkt; wat misging, gebeurde in de jaren erna, toen het model nauwelijks veranderde en de wereld eromheen wel. Deze les volgt een AI-systeem door het hele leven, van het eerste idee tot de dag dat het uitgaat, en kijkt wie er onderweg bij betrokken zijn.

    **Een voorbeeld uit het dagelijks leven.** Een auto wordt in een paar jaar ontworpen, getest en gebouwd, en daarna tien of twintig jaar gereden. Onderweg krijgt de auto elk jaar een APK en reparaties, en soms wordt een onderdeel vervangen omdat de regels veranderden. Op een dag gaat de auto naar de sloop, en moet iemand iets met de accu en de olie. Het grootste deel van het leven van een auto komt na de fabriek, en ook een groot deel van wat de auto kost. Onderweg doen veel partijen mee: de fabrikant, de dealer, de eigenaar, de garage, de keurders, en de mensen die langs de wegen wonen waarover de auto rijdt.
    """)
    return


@app.cell(hide_code=True)
def predict_estimate():
    guess_estimate = L.guess(
        "estimate",
        question="In het griepseizoen van 2012 tot 2013 schatte Google Flu Trends welk deel van de bezoeken aan artsen in de Verenigde Staten om griepachtige klachten ging. Hoe ver zat de schatting voor de piek naast wat de CDC telde, denk je?",
        options={
            "close": "Binnen 5%",
            "fifth": "Ongeveer 20% te hoog",
            "right": "Ongeveer het dubbele",
            "ten": "Ongeveer tien keer te hoog",
        },
        reveal="**Ongeveer het dubbele.** In februari 2013 meldde *Nature* dat de schatting van Flu Trends voor de piek van het seizoen bijna twee keer zo hoog lag als wat de CDC telde; in 2014 schreven David Lazer en anderen in *Science* zelfs over meer dan het dubbele. Van augustus 2011 tot september 2013 lag de schatting in 100 van de 108 weken te hoog. Het model was in 2009 bijgewerkt en draaide daarna bijna ongewijzigd door tot oktober 2013, terwijl Google de eigen zoekmachine steeds veranderde: 86 veranderingen in juni en juli 2012 alleen al, volgens het zoekblog van Google.",
    )
    guess_estimate
    return (guess_estimate,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een leven in fasen.** Een AI-systeem is niet af als het model getraind is. Het begint met een idee, wordt gebouwd en gecontroleerd, gaat in gebruik, wordt jarenlang in de gaten gehouden, wordt af en toe veranderd, en gaat op een dag uit. Deze fasen samen zijn de **AI-levenscyclus**. In eerdere lessen eindigde een project met één meting op de test set; in een echt systeem komt deze meting vroeg: bij de spamdienst in maand 5 van de 36.

        **ISO/IEC 22989.** **ISO/IEC 22989**, een internationale norm uit 2022, legt de basiswoorden en -begrippen van AI vast. De norm beschrijft de levenscyclus van een AI-systeem in acht fasen, en ISO/IEC 5338, uit 2023, bouwt daarop voort met het werk in elke fase, op basis van de normen voor de levenscyclus van systemen en software. De afbeelding laat de fasen zien; de tabel volgt de spamdienst van de course erdoorheen, zoals de dienst in de mailbox van een gemeente zou draaien.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        | Fase | Wat er gebeurt | De spamdienst |
        |---|---|---|
        | **idee** (inception) | het probleem, wat een fout kost, een meetbaar doel, en of AI wel nodig is | een echte vraag in de spammap is erger dan spam in de inbox; minstens 0,90 goed |
        | **ontwerp en ontwikkeling** | data, features, het model en alle code eromheen | de acht delen van de dienst, en gelabelde berichten om van te leren |
        | **verificatie en validatie** | verificatie: is het gebouwd zoals afgesproken? validatie: doet het wat de gebruikers nodig hebben? | accuracy boven het doel op testberichten; medewerkers van de gemeente proberen het een week op echte mail |
        | **deployment** | naar de plek waar het gebruikt wordt | naar de mailbox van de gemeente, en medewerkers leren fouten markeren |
        | **gebruik en monitoring** | het draait, en de antwoorden worden bekeken | elke maand kijken medewerkers een steekproef van berichten met de hand na |
        | **doorlopende validatie** | bij een systeem dat blijft leren: elke nieuwe versie wordt gecontroleerd voordat de versie antwoordt | het model dat opnieuw getraind is op markeringen, wordt gecontroleerd op de testberichten |
        | **herbeoordeling** | een nieuwe blik op het geheel: nog nodig, nog goed genoeg? | na het alarm: terug naar ontwerp; in maand 35 niet meer nodig: uitfasering |
        | **uitfasering** | uitgezet, gebruikers ingelicht, data gewist of bewaard zoals afgesproken | de bewaarde berichten en markeringen worden gewist |

        **Geen rechte lijn.** Een herbeoordeling kan uitwijzen dat alles goed is, en dan blijft het systeem draaien. Anders zijn er twee uitgangen. Is het systeem nog nodig, alleen niet meer goed genoeg, dan gaat het terug naar ontwerp en ontwikkeling, en opnieuw door verificatie en deployment. Is het niet meer nodig, of is het niet meer goed genoeg te krijgen, dan volgt uitfasering. De spamdienst leert bij van de markeringen van gebruikers. Een systeem dat zo blijft leren, krijgt naast het gebruik ook doorlopende validatie: een model dat vanzelf verandert, moet bij elke verandering gecontroleerd worden.

        **Wie er betrokken zijn.** In elke fase doen anderen mee, en elke partij heeft eigen belangen: een **AI-stakeholder**. ISO/IEC 22989 geeft de rollen namen:

        | Rol | Wat de rol doet | Rond de spamdienst |
        |---|---|---|
        | **AI-aanbieder** (AI provider) | biedt een AI-product of -dienst aan | het bedrijf |
        | **AI-producent** (AI producer) | ontwerpt en bouwt de AI | het bedrijf |
        | **AI-klant** (AI customer) | gebruikt een AI-product of -dienst, vaak na aankoop | de gemeente |
        | **AI-partner** (AI partner) | helpt rond de AI, bijvoorbeeld met data, met het inpassen in andere systemen, of met een audit | een labelbureau, dat berichten als spam of geen spam markeert |
        | **AI-subject** (AI subject) | wordt geraakt door het systeem: de mensen van wie het de data gebruikt of over wie het beslist | de inwoners die de gemeente mailen |
        | **relevante autoriteiten** (relevant authorities) | maken de regels en zien erop toe dat de regels nageleefd worden | de privacytoezichthouder |

        Eén partij kan meerdere rollen hebben: het bedrijf dat de dienst bouwt en verkoopt, is AI-producent en AI-aanbieder tegelijk. En een **AI-subject** ziet het systeem vaak nooit: de inwoners mailen alleen de gemeente, en toch raakt een echte vraag in de spammap de inwoners het eerst.

        ### Nieuwe woorden en tekens

        | Woord of symbool | Uitspraak | Betekenis | In code |
        |---|---|---|---|
        | **AI-levenscyclus** (AI lifecycle) | | de fasen die een AI-systeem doorloopt, van het eerste idee via bouwen, controleren, gebruik en monitoring tot uitfasering; ISO/IEC 22989 schrijft life cycle | |
        | **ISO/IEC 22989** | ie-es-oo ie-ee-cee tweeëntwintig negen negenentachtig | een internationale norm uit 2022 met de basiswoorden en -begrippen van AI, met daarin de fasen van de levenscyclus en de rollen van de partijen | |
        | **AI-stakeholder** | | een partij die bij een AI-systeem betrokken is of erdoor geraakt wordt, in een van de rollen van ISO/IEC 22989 | |
        | **AI-subject** | | een persoon of groep die door een AI-systeem geraakt wordt: van wie het de data gebruikt, of over wie het beslist | |
        """),
        L.watch_out("De Europese AI-verordening, de AI Act, noemt ook rollen, met andere woorden en een juridische betekenis: de aanbieder (provider) ontwikkelt een AI-systeem, of laat het ontwikkelen, en brengt het onder de eigen naam of het eigen merk in de handel of stelt het in gebruik; de gebruiksverantwoordelijke (deployer) gebruikt een AI-systeem onder de eigen verantwoordelijkheid. Houd de twee rijtjes uit elkaar: een partij kan in de norm AI-klant zijn en in de wet gebruiksverantwoordelijke, en de woorden vallen niet altijd samen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een AI-systeem door het leven volgen:

    1. Schrijf bij het idee het probleem op, wat elke soort fout kost, en een meetbaar doel; vraag je af of AI wel nodig is.
    2. Benoem de partijen en de rol van elke partij: wie het aanbiedt, wie het bouwt, wie het gebruikt, wie helpt, wie erdoor geraakt wordt, en wie de regels maakt.
    3. Controleer allebei voordat de deployment begint: dat het gebouwd is zoals afgesproken (verificatie), en dat het doet wat de gebruikers nodig hebben (validatie).
    4. Meet tijdens het gebruik regelmatig tegen het doel, en spreek vooraf af wat een daling onder het doel in gang zet.
    5. Kies bij elke herbeoordeling: laten draaien, terug naar ontwerp, of uitfaseren; en plan de uitfasering: wie wordt ingelicht, en wat gebeurt er met de data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.HISTORY` is het leven van de spamdienst van de course in de mailbox van een gemeente, verzonnen door de course: een lijst met regels `(stage, months)`, op volgorde. De fasen staan er in het Engels in, zoals in de norm. De cel print elke regel en telt de maanden van het hele leven op. Voer de cel uit.
    """)
    return


@app.cell
def example_history():
    history = C.HISTORY
    total = 0
    for stage, months in history:
        print(stage, months)
        total = total + months
    print("maanden in totaal:", total)
    return history, total


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De dienst leefde 36 maanden. Een paar fasen komen twee keer voor: na de eerste periode van gebruik stuurde een herbeoordeling de dienst terug naar ontwerp en ontwikkeling, via verificatie en validatie en deployment, naar een tweede periode van gebruik. Oefening 1 telt de maanden van elke fase op.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_months():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Waar de maanden heen gaan

        `months_per_stage(history)` krijgt een lijst met regels `(stage, months)`, en geeft een dict terug van elke fase naar het totaal aan maanden. De loop gaat de regels langs; komt een `stage` voor het eerst, dan begint `totals[stage]` bij 0. Vervang de `...` door de nieuwe waarde van `totals[stage]`: wat er tot nu toe in stond, plus de `months` van deze regel.

        De check roept `months_per_stage(C.HISTORY)` aan. Daarna een lege lijst, een geschiedenis van één regel, drie regels van dezelfde fase, halve maanden, en een andere geschiedenis.
        """),
        L.task("Tel de maanden van elke fase op, ook als een fase terugkomt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_months():
    box_months = L.code_box("months", lines=8)
    box_months
    return (box_months,)


@app.cell(hide_code=True)
def check_months(box_months):
    result_months = L.check("months", box_months.value)
    result_months
    return (result_months,)


@app.cell(hide_code=True)
def hints_months():
    L.hints(
        "months",
        [
            "`totals[stage]` bevat de maanden van `stage` tot nu toe, en `months` bevat de maanden van deze regel. De nieuwe waarde van `totals[stage]` gebruikt allebei.",
            "Optellen, niet vervangen: een fase die terugkomt, zoals gebruik en monitoring, moet de maanden van allebei de periodes krijgen. Welke waarde bevat de maanden van deze fase tot nu toe, en wat tel je daarbij op?",
            "`        totals[stage] = ___`",
        ],
        why="`totals[stage] = totals[stage] + months` telt de maanden van elke regel op bij wat de fase al had. Gebruik en monitoring krijgt 14 + 9 = 23 maanden, van de 36 in totaal.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_below():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wanneer opnieuw kijken

        `first_month_below(accuracies, goal)` krijgt de accuracy van elke maand van gebruik, op volgorde, en het doel dat bij het idee is afgesproken. De functie geeft het nummer terug van de eerste maand met een accuracy onder het doel, geteld vanaf 1, of `None` als geen enkele maand eronder ligt. `month` telt de maanden; in elke ronde van de loop is `accuracy` de accuracy van deze maand. Vervang de `...` door de voorwaarde: `accuracy` ligt onder `goal`. Een maand precies op het doel is nog goed genoeg.

        De check roept `first_month_below(C.ACCURACY, 0.9)` aan: de 14 maanden van de eerste periode van gebruik van de dienst. Daarna een maand precies op het doel, accuracy's die nooit onder het doel zakken, een lege lijst, een eerste maand onder het doel, en een doel van 0,8.
        """),
        L.task("Geef de eerste maand terug met een accuracy onder het doel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_below():
    box_below = L.code_box("below", lines=8)
    box_below
    return (box_below,)


@app.cell(hide_code=True)
def check_below(box_below):
    result_below = L.check("below", box_below.value)
    result_below
    return (result_below,)


@app.cell(hide_code=True)
def hints_below():
    L.hints(
        "below",
        [
            "Stap 4 van het recept. In elke ronde hoort `accuracy` bij maand `month`, en de voorwaarde vergelijkt de accuracy met `goal`.",
            "Onder het doel betekent kleiner dan het doel, en een maand precies op het doel telt niet mee. Welke vergelijking zegt dat?",
            "`        if ___:`",
        ],
        why="`if accuracy < goal:` stopt bij de eerste maand onder het doel en geeft het nummer terug. Bij de dienst is dat maand 13 van de eerste periode van gebruik, met een accuracy van 0,89.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Het leven van de dienst

    De demo gaat de 36 maanden van de dienst één voor één langs. De afbeelding heeft een rij per fase: de maanden die de dienst in de fase doorbracht, donker tot nu en licht voor wat nog komt, met een lijn bij de huidige maand. Onder de rijen staat de accuracy van elke maand van gebruik, met het doel als een lijn van streepjes, en daaronder wie er in deze fase iets doet.
    """)
    return


@app.cell(hide_code=True)
def demo_life():
    L.explore("month", "maand", lambda _v: C.frame(int(_v), L.lang), start=1, values=list(range(1, 37)), role="",
              task="Ga de 36 maanden langs: in welke maand verlaat de dienst voor het eerst het gebruik, en waarom?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De dienst is 23 van de 36 maanden in gebruik en monitoring. In maand 19 zakt de accuracy voor het eerst onder het doel, hoewel feedback het model in maand 12 en 17 opnieuw trainde. In maand 21 kijken de gemeente en het bedrijf opnieuw: de dienst is nog nodig, dus gaat de dienst terug naar ontwerp en ontwikkeling, en vanaf maand 26 antwoordt versie spam-2. In maand 35 eindigt een tweede herbeoordeling in uitfasering: niet omdat de dienst slechter werd, maar omdat de dienst niet meer nodig was. De gemeente doet in elke fase mee, behalve bij ontwerp en ontwikkeling; de inwoners zien de dienst nooit, maar over de berichten van inwoners beslist de dienst wel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flu():
    mo.md(r"""
    ### Oefening 3. Google Flu Trends

    Het model van Google Flu Trends werd in 2009 bijgewerkt en draaide daarna bijna ongewijzigd door tot oktober 2013. Ondertussen veranderde Google steeds de zoekmachine, en zochten mensen naar andere dingen. Van augustus 2011 tot september 2013 lagen de schattingen in 100 van de 108 weken te hoog, terwijl de CDC elke week de eigen cijfers publiceerde.
    """)
    return


@app.cell(hide_code=True)
def choice_flu_cell():
    choice_flu = L.choice(
        "flu",
        question="Welke fase van de levenscyclus schoot het meest tekort?",
        options={
            "inception": "Idee: zoekopdrachten konden vanaf het allereerste begin nooit iets zeggen over griep",
            "deploy": "Deployment: te weinig mensen bezochten de website",
            "right": "Herbeoordeling: het model werd zelden getoetst aan een veranderende wereld",
            "retire": "Uitfasering: Google zette de website te vroeg uit",
        },
        correct="right",
        why={
            "right": "Het model paste bij de wereld van 2009. Zoeken veranderde, en het model nauwelijks, terwijl de cijfers van de CDC elke week hadden kunnen laten zien dat de schattingen afdreven. Een regelmatige herbeoordeling tegen de cijfers van de CDC had het opgemerkt en het model teruggestuurd naar ontwerp; uiteindelijk bleef alleen uitfasering over.",
            "inception": "Bij de bouw kwamen de schattingen overeen met jaren aan cijfers van de CDC, dus zeiden zoekopdrachten wel iets over griep. Wat veranderde er in de jaren erna?",
            "deploy": "De schattingen stonden openbaar, voor iedereen te zien. Zat het probleem in wie de schattingen zag, of in wat de schattingen zeiden?",
            "retire": "In 2015 lagen de schattingen al jaren te hoog. Welke fase had jaren eerder dan de uitfasering moeten komen?",
        },
    )
    choice_flu
    return (choice_flu,)


@app.cell(hide_code=True)
def text_exercise_subject():
    mo.md(r"""
    ### Oefening 4. Over wie gaat het?

    In de demo mailen inwoners vragen naar de gemeente. Inwoners zien de spamdienst nooit en kozen de dienst niet: de gemeente kocht de dienst van het bedrijf.
    """)
    return


@app.cell(hide_code=True)
def choice_subject_cell():
    choice_subject = L.choice(
        "subject",
        question="Welke rol geeft ISO/IEC 22989 de inwoners?",
        options={
            "none": "Geen: alleen de partijen die een systeem bouwen, verkopen of gebruiken, hebben een rol",
            "customer": "AI-klant, want de mailbox is er voor de inwoners",
            "partner": "AI-partner, want de berichten zijn de data",
            "right": "AI-subject, want de dienst beslist over de berichten",
        },
        correct="right",
        why={
            "right": "Een AI-subject wordt geraakt door het systeem: door wat het systeem met data over het subject doet, of door wat het systeem over het subject beslist. De inwoners gebruiken de dienst nooit, en toch raakt een echte vraag in de spammap de inwoners het eerst. De belangen van inwoners komen bij het idee al aan bod, als de gemeente en het bedrijf afspreken dat een gemiste echte vraag de ergste fout is.",
            "none": "De rollen van de norm gaan ook over partijen die het systeem nooit aanraken. Van wie komen de vragen in de spammap als de dienst het fout heeft?",
            "customer": "De gemeente is de AI-klant: de gemeente kocht de dienst en gebruikt de dienst. Wat doen de inwoners zelf met de dienst?",
            "partner": "Een AI-partner helpt het systeem te maken of te controleren, zoals het bureau dat berichten labelt. Helpen de inwoners de dienst te maken, of beslist de dienst over de inwoners?",
        },
    )
    choice_subject
    return (choice_subject,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) was de spamdienst een verzameling delen op één moment. Deze les volgde de dienst door de tijd: van het idee, via bouwen, controleren en jaren gebruik, één keer terug naar ontwerp, en door naar uitfasering, met de partijen eromheen, elk in een eigen rol. In de volgende les, *Wie bouwt een ML-systeem: rollen en teams*, kijk je binnen in het bedrijf: welke mensen zo'n systeem bouwen en draaiende houden, en hoe deze mensen samenwerken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flu, choice_subject, result_below, result_months):
    L.wrap_up(result_months, result_below, choice_flu, choice_subject)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
